//! SQLite catalog: files, directory tree and FTS5 text chunks.
//! One mutex-guarded writer connection; readers open extra WAL connections on demand.

use crate::{CoreResult, DirRow, FileFilter, FileKind, FileRow, RagStatus, SortKey};
use rusqlite::{params, params_from_iter, Connection, OptionalExtension};
use std::path::{Path, PathBuf};

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS dirs(
  id INTEGER PRIMARY KEY,
  parent_id INTEGER REFERENCES dirs(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  path TEXT NOT NULL UNIQUE,
  mtime INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS dirs_parent ON dirs(parent_id);
CREATE TABLE IF NOT EXISTS files(
  id INTEGER PRIMARY KEY,
  dir_id INTEGER NOT NULL REFERENCES dirs(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  kind TEXT NOT NULL,
  size INTEGER NOT NULL,
  mtime INTEGER NOT NULL,
  duration REAL,
  rag INTEGER NOT NULL DEFAULT 0,
  seen INTEGER NOT NULL DEFAULT 1,
  UNIQUE(dir_id, name)
);
CREATE INDEX IF NOT EXISTS files_dir ON files(dir_id);
CREATE INDEX IF NOT EXISTS files_rag ON files(rag);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(text, file_id UNINDEXED, vec_slot UNINDEXED);
";

/// New file row for upsert (parent resolved by caller).
pub struct NewFile {
    pub dir_id: i64,
    pub name: String,
    pub kind: FileKind,
    pub size: i64,
    pub mtime: i64,
}

pub struct NewChunk<'a> {
    pub file_id: i64,
    pub text: &'a str,
}

pub struct Catalog {
    conn: std::sync::Mutex<Connection>,
    path: PathBuf,
}

fn open_connection(path: &Path, read_only: bool) -> CoreResult<Connection> {
        let conn = Connection::open(path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.pragma_update(None, "mmap_size", 268_435_456i64)?;
        conn.pragma_update(None, "cache_size", -65_536i64)?; // 64 MiB page cache
        conn.pragma_update(None, "foreign_keys", "ON")?;
        if read_only {
            conn.pragma_update(None, "query_only", true)?;
        }
        Ok(conn)
    }

impl Catalog {
    pub fn open(path: &Path) -> CoreResult<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = open_connection(path, false)?;
        conn.execute_batch(SCHEMA)?;
        // Rows left Busy by a previous crashed run go back to Pending.
        conn.execute("UPDATE files SET rag = 0 WHERE rag = 1", [])?;
        // Older catalogs: non-RAG kinds must not linger as phantom pending work.
        conn.execute(
            "UPDATE files SET rag = 2 WHERE rag = 0 AND kind NOT IN ('text','pdf','audio','video','image')",
            [],
        )?;
        // Images became searchable after v1: re-queue pre-existing image rows
        // that were marked Done before filename-context indexing existed.
        conn.execute(
            "UPDATE files SET rag = 0
             WHERE kind = 'image' AND rag = 2
               AND id NOT IN (SELECT DISTINCT file_id FROM chunks)",
            [],
        )?;
        Ok(Self { conn: conn.into(), path: path.to_path_buf() })
    }

    /// Fresh read connection for use on any thread (WAL permits concurrent readers).
    pub fn reader(&self) -> CoreResult<Connection> {
        open_connection(&self.path, true)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Marks every row unseen before a full rescan.
    pub fn begin_full_scan(&self) -> CoreResult<()> {
        self.conn.lock().unwrap().execute("UPDATE files SET seen = 0", [])?;
        Ok(())
    }

    /// Deletes rows not reached by the last scan; returns vector slots to free.
    pub fn finish_full_scan(&self) -> CoreResult<Vec<i64>> {
        let conn = self.conn.lock().unwrap();
        let tx = conn.unchecked_transaction()?;
        let stale_file_ids: Vec<i64> = {
            let mut statement = tx.prepare("SELECT id FROM files WHERE seen = 0")?;
            statement.query_map([], |row| row.get(0))?.collect::<Result<_, _>>()?
        };
        let slots = Self::delete_files_chunks(&tx, &stale_file_ids)?;
        tx.execute("DELETE FROM files WHERE seen = 0", [])?;
        tx.execute(
            "DELETE FROM dirs WHERE id NOT IN (SELECT DISTINCT dir_id FROM files)
             AND id NOT IN (SELECT DISTINCT parent_id FROM dirs WHERE parent_id IS NOT NULL)
             AND parent_id IS NOT NULL",
            [],
        )?;
        tx.commit()?;
        Ok(slots)
    }

    /// Removes chunks of files about to be re-indexed; returns freed vector slots.
    pub fn delete_chunks_for_files(&self, file_ids: &[i64]) -> CoreResult<Vec<i64>> {
        if file_ids.is_empty() {
            return Ok(Vec::new());
        }
        let conn = self.conn.lock().unwrap();
        let tx = conn.unchecked_transaction()?;
        let slots = Self::delete_files_chunks(&tx, file_ids)?;
        tx.commit()?;
        Ok(slots)
    }

    /// Atomically claims up to `count` pending jobs as Busy (safe under concurrency).
    pub fn claim_rag_jobs(&self, kinds: &[FileKind], count: i64) -> CoreResult<Vec<(i64, String)>> {
        let kind_list = kinds.iter().map(|kind| format!("'{}'", kind.token())).collect::<Vec<_>>().join(",");
        let claim_sql = format!(
            "UPDATE files SET rag = 1 WHERE id IN (
               SELECT f.id FROM files f WHERE f.rag = 0 AND f.kind IN ({kind_list}) ORDER BY f.id LIMIT {count}
             ) RETURNING id"
        );
        let conn = self.conn.lock().unwrap();
        let tx = conn.unchecked_transaction()?;
        let claimed_ids: Vec<i64> = {
            let mut statement = tx.prepare(&claim_sql)?;
            statement.query_map([], |row| row.get(0))?.collect::<Result<_, _>>()?
        };
        let mut jobs = Vec::with_capacity(claimed_ids.len());
        if !claimed_ids.is_empty() {
            let placeholders = vec!["?"; claimed_ids.len()].join(",");
            let path_sql = format!(
                "SELECT f.id, d.path || '/' || f.name FROM files f JOIN dirs d ON d.id = f.dir_id
                 WHERE f.id IN ({placeholders})"
            );
            let mut statement = tx.prepare(&path_sql)?;
            jobs = statement
                .query_map(params_from_iter(claimed_ids), |row| {
                    Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
                })?
                .collect::<Result<_, _>>()?;
        }
        tx.commit()?;
        Ok(jobs)
    }

    /// Directory id lookup by absolute path.
    pub fn dir_id_by_path(&self, path: &str) -> Option<i64> {
        self.conn
            .lock()
            .unwrap()
            .query_row("SELECT id FROM dirs WHERE path = ?1", params![path], |row| row.get(0))
            .optional()
            .unwrap_or(None)
    }

    /// Inserts chunk rows (vector slot pending); returns rowids in input order.
    /// FTS5's RETURNING rowid yields -1, so last_insert_rowid is used instead.
    pub fn insert_chunks(&self, chunks: &[(i64, &str)]) -> CoreResult<Vec<i64>> {
        let conn = self.conn.lock().unwrap();
        let mut statement =
            conn.prepare("INSERT INTO chunks(file_id, vec_slot, text) VALUES (?1, -1, ?2)")?;
        let mut rowids = Vec::with_capacity(chunks.len());
        for (file_id, text) in chunks {
            statement.execute(params![file_id, text])?;
            rowids.push(conn.last_insert_rowid());
        }
        Ok(rowids)
    }

    /// Binds persisted vector slots to their chunk rows.
    pub fn assign_vector_slots(&self, assignments: &[(i64, i64)]) -> CoreResult<()> {
        let conn = self.conn.lock().unwrap();
        let mut statement = conn.prepare("UPDATE chunks SET vec_slot = ?1 WHERE rowid = ?2")?;
        for (rowid, slot) in assignments {
            statement.execute(params![slot, rowid])?;
        }
        Ok(())
    }

    /// Removes chunks whose file vanished or whose vector write was interrupted;
    /// returns recyclable vector slots.
    pub fn delete_orphan_chunks(&self) -> CoreResult<Vec<i64>> {
        let conn = self.conn.lock().unwrap();
        let slots: Vec<i64> = {
            let mut statement = conn.prepare(
                "SELECT vec_slot FROM chunks WHERE file_id NOT IN (SELECT id FROM files) AND vec_slot >= 0",
            )?;
            statement.query_map([], |row| row.get(0))?.collect::<Result<_, _>>()?
        };
        conn.execute("DELETE FROM chunks WHERE file_id NOT IN (SELECT id FROM files) OR vec_slot < 0", [])?;
        Ok(slots)
    }

    /// Rag-status counts: (pending, busy, done, failed).
    pub fn rag_counters(&self) -> CoreResult<(i64, i64, i64, i64)> {
        let conn = self.conn.lock().unwrap();
        let counters = conn.query_row(
            "SELECT SUM(rag = 0), SUM(rag = 1), SUM(rag = 2), SUM(rag = 3) FROM files",
            [],
            |row| {
                Ok((
                    row.get::<_, Option<i64>>(0)?.unwrap_or(0),
                    row.get::<_, Option<i64>>(1)?.unwrap_or(0),
                    row.get::<_, Option<i64>>(2)?.unwrap_or(0),
                    row.get::<_, Option<i64>>(3)?.unwrap_or(0),
                ))
            },
        )?;
        Ok(counters)
    }

    pub fn total_files(&self) -> CoreResult<i64> {
        Ok(self.conn.lock().unwrap().query_row("SELECT COUNT(*) FROM files", [], |row| row.get(0))?)
    }

    /// Directory ids mapped by path, for parent resolution in the collector.
    pub fn dir_id_map(&self) -> CoreResult<std::collections::HashMap<String, i64>> {
        let conn = self.conn.lock().unwrap();
        let mut statement = conn.prepare("SELECT id, path FROM dirs")?;
        let map = statement
            .query_map([], |row| Ok((row.get::<_, String>(1)?, row.get::<_, i64>(0)?)))?
            .collect::<Result<_, _>>()?;
        Ok(map)
    }

    /// Upserts directories, creating any missing ancestor chain (order-independent).
    pub fn upsert_dirs(&self, dirs: &[(PathBuf, i64)]) -> CoreResult<()> {
        let conn = self.conn.lock().unwrap();
        for (path, mtime) in dirs {
            ensure_dir(&conn, path, *mtime)?;
        }
        Ok(())
    }

    /// Bulk file upsert; a changed size/mtime resets rag to Pending for reindexing.
    /// Non-RAG kinds (image/other) enter as Done so they never clog the queue.
    pub fn upsert_files(&self, files: &[NewFile]) -> CoreResult<()> {
        if files.is_empty() {
            return Ok(());
        }
        let conn = self.conn.lock().unwrap();
        let tx = conn.unchecked_transaction()?;
        {
            let mut statement = tx.prepare(
                "INSERT INTO files(dir_id, name, kind, size, mtime, seen, rag)
                 VALUES (?1, ?2, ?3, ?4, ?5, 1,
                         CASE WHEN ?3 IN ('text','pdf','audio','video','image') THEN 0 ELSE 2 END)
                 ON CONFLICT(dir_id, name) DO UPDATE SET
                   size = excluded.size,
                   mtime = excluded.mtime,
                   seen = 1,
                   rag = CASE WHEN files.size != excluded.size OR files.mtime != excluded.mtime
                              THEN CASE WHEN excluded.kind IN ('text','pdf','audio','video','image')
                                        THEN 0 ELSE 2 END
                              ELSE files.rag END",
            )?;
            for file in files {
                statement.execute(params![
                    file.dir_id,
                    file.name,
                    file.kind.token(),
                    file.size,
                    file.mtime
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// Files awaiting RAG processing in priority order (text/pdf before slow media).
    pub fn pending_rag(&self, kinds: &[FileKind], limit: i64) -> CoreResult<Vec<(i64, String)>> {
        let kind_list: Vec<String> = kinds.iter().map(|k| format!("'{}'", k.token())).collect();
        let sql = format!(
            "SELECT f.id, d.path || '/' || f.name FROM files f JOIN dirs d ON d.id = f.dir_id
             WHERE f.rag = 0 AND f.kind IN ({}) ORDER BY f.id LIMIT {}",
            kind_list.join(","),
            limit
        );
        let conn = self.conn.lock().unwrap();
        let mut statement = conn.prepare(&sql)?;
        let rows = statement
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<Result<_, _>>()?;
        Ok(rows)
    }

    pub fn set_rag(&self, ids: &[i64], status: RagStatus) -> CoreResult<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let conn = self.conn.lock().unwrap();
        conn.prepare_cached("UPDATE files SET rag = ?1 WHERE id = ?2")?;
        let mut statement = conn.prepare("UPDATE files SET rag = ?1 WHERE id = ?2")?;
        let tx = conn.unchecked_transaction()?;
        for id in ids {
            statement.execute(params![status as i64, id])?;
        }
        drop(statement);
        tx.commit()?;
        Ok(())
    }

    pub fn set_duration(&self, file_id: i64, duration_secs: f64) -> CoreResult<()> {
        self.conn.lock().unwrap().execute(
            "UPDATE files SET duration = ?1 WHERE id = ?2",
            params![duration_secs, file_id],
        )?;
        Ok(())
    }

    /// Lists children of a directory with filters and sorting applied in SQL.
    pub fn children(
        &self,
        conn: &Connection,
        dir_id: Option<i64>,
        filter: &FileFilter,
        sort: SortKey,
        ascending: bool,
        limit: i64,
    ) -> CoreResult<Vec<FileRow>> {
        let mut clauses: Vec<String> = Vec::new();
        let mut bind: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        match dir_id {
            Some(dir_id) => clauses.push(format!("f.dir_id = {}", dir_id)),
            None => clauses.push("f.dir_id IS NOT NULL".into()), // whole-library view
        }
        if !filter.kinds.is_empty() {
            let tokens: Vec<&str> = filter.kinds.iter().map(|k| k.token()).collect();
            clauses.push(format!(
                "f.kind IN ({})",
                tokens.iter().map(|t| format!("'{}'", t)).collect::<Vec<_>>().join(",")
            ));
        }
        if let Some(name) = filter.name_query.as_deref().filter(|q| !q.is_empty()) {
            let escaped = name.replace('\\', "\\\\").replace('%', "\\%").replace('_', "\\_");
            clauses.push("f.name LIKE ?1 ESCAPE '\\'".into());
            bind.push(Box::new(format!("%{}%", escaped)));
        }
        for (condition, value) in [
            ("f.size >= ?", filter.min_size),
            ("f.size <= ?", filter.max_size),
            ("f.mtime >= ?", filter.after_unix),
            ("f.mtime <= ?", filter.before_unix),
        ] {
            if let Some(value) = value {
                clauses.push(condition.into());
                bind.push(Box::new(value));
            }
        }
        let order_column = match sort {
            SortKey::Name => "f.name COLLATE NOCASE",
            SortKey::Kind => "f.kind",
            SortKey::Size => "f.size",
            SortKey::Modified => "f.mtime",
            SortKey::Duration => "f.duration",
        };
        let direction = if ascending { "ASC" } else { "DESC" };
        let sql = format!(
            "SELECT f.id, f.dir_id, f.name, f.kind, f.size, f.mtime, f.duration, f.rag
             FROM files f WHERE {} ORDER BY {} {}, f.name COLLATE NOCASE LIMIT {}",
            clauses.join(" AND "),
            order_column,
            direction,
            limit
        );
        let mut statement = conn.prepare(&sql)?;
        let rows = statement
            .query_map(params_from_iter(bind.iter().map(|b| b.as_ref())), file_row_from)?
            .collect::<Result<_, _>>()?;
        Ok(rows)
    }

    pub fn tree_children(&self, conn: &Connection, dir_id: Option<i64>) -> CoreResult<Vec<DirRow>> {
        let mut statement = conn.prepare(
            "SELECT d.id, d.parent_id, d.name, d.path,
                    (SELECT COUNT(*) FROM files f WHERE f.dir_id = d.id) AS file_count,
                    (SELECT COUNT(*) FROM dirs c WHERE c.parent_id = d.id) AS subdir_count
             FROM dirs d WHERE d.parent_id IS ?1 ORDER BY d.name COLLATE NOCASE",
        )?;
        let rows = statement
            .query_map(params![dir_id], |row| {
                Ok(DirRow {
                    id: row.get(0)?,
                    parent_id: row.get(1)?,
                    name: row.get(2)?,
                    path: row.get(3)?,
                    file_count: row.get(4)?,
                    subdir_count: row.get(5)?,
                })
            })?
            .collect::<Result<_, _>>()?;
        Ok(rows)
    }

    pub fn dir_by_path(&self, conn: &Connection, path: &str) -> CoreResult<Option<DirRow>> {
        let row = conn
            .query_row(
                "SELECT d.id, d.parent_id, d.name, d.path, 0, 0 FROM dirs d WHERE d.path = ?1",
                params![path],
                |row| {
                    Ok(DirRow {
                        id: row.get(0)?,
                        parent_id: row.get(1)?,
                        name: row.get(2)?,
                        path: row.get(3)?,
                        file_count: 0,
                        subdir_count: 0,
                    })
                },
            )
            .optional()?;
        Ok(row)
    }

    pub fn file_by_id(&self, conn: &Connection, file_id: i64) -> CoreResult<Option<(FileRow, String)>> {
        let row = conn
            .query_row(
                "SELECT f.id, f.dir_id, f.name, f.kind, f.size, f.mtime, f.duration, f.rag, d.path || '/' || f.name
                 FROM files f JOIN dirs d ON d.id = f.dir_id WHERE f.id = ?1",
                params![file_id],
                |row| {
                    Ok((
                        file_row_from(row)?,
                        row.get::<_, String>(8)?,
                    ))
                },
            )
            .optional()?;
        Ok(row)
    }

    /// Fetches full file rows (with directory path) for a set of ids.
    pub fn files_by_ids(&self, conn: &Connection, file_ids: &[i64]) -> CoreResult<Vec<(FileRow, String)>> {
        if file_ids.is_empty() {
            return Ok(Vec::new());
        }
        let placeholders = vec!["?"; file_ids.len()].join(",");
        let sql = format!(
            "SELECT f.id, f.dir_id, f.name, f.kind, f.size, f.mtime, f.duration, f.rag, d.path || '/' || f.name
             FROM files f JOIN dirs d ON d.id = f.dir_id WHERE f.id IN ({})",
            placeholders
        );
        let mut statement = conn.prepare(&sql)?;
        let rows = statement
            .query_map(params_from_iter(file_ids), |row| {
                Ok((file_row_from(row)?, row.get::<_, String>(8)?))
            })?
            .collect::<Result<_, _>>()?;
        Ok(rows)
    }

    /// Filename substring lookup across the whole catalog.
    pub fn name_search(&self, conn: &Connection, query: &str, limit: i64) -> CoreResult<Vec<(FileRow, String)>> {
        let escaped = query.replace('\\', "\\\\").replace('%', "\\%").replace('_', "\\_");
        let mut statement = conn.prepare(
            "SELECT f.id, f.dir_id, f.name, f.kind, f.size, f.mtime, f.duration, f.rag, d.path || '/' || f.name
             FROM files f JOIN dirs d ON d.id = f.dir_id
             WHERE f.name LIKE ?1 ESCAPE '\\' ORDER BY f.mtime DESC LIMIT ?2",
        )?;
        let rows = statement
            .query_map(params![format!("%{}%", escaped), limit], |row| {
                Ok((file_row_from(row)?, row.get::<_, String>(8)?))
            })?
            .collect::<Result<_, _>>()?;
        Ok(rows)
    }

    /// Removes a file or a whole directory subtree; returns vector slots to free.
    pub fn remove_path(&self, target: &str) -> CoreResult<Vec<i64>> {
        let conn = self.conn.lock().unwrap();
        let tx = conn.unchecked_transaction()?;
        let freed_files: Vec<i64> = {
            // Files below a removed directory, or the single file itself.
            let mut statement = tx.prepare(
                "SELECT f.id FROM files f JOIN dirs d ON d.id = f.dir_id
                 WHERE d.path = ?1 OR d.path LIKE ?1 || '/%' OR (d.path = ?2 AND f.name = ?3)",
            )?;
            let mut rows = statement.query(params![target, parent_of(target), file_name_of(target)])?;
            let mut ids = Vec::new();
            while let Some(row) = rows.next()? {
                ids.push(row.get(0)?);
            }
            ids
        };
        let slots = Self::delete_files_chunks(&tx, &freed_files)?;
        tx.execute(
            "DELETE FROM files WHERE id IN (SELECT f.id FROM files f JOIN dirs d ON d.id = f.dir_id
             WHERE d.path = ?1 OR d.path LIKE ?1 || '/%' OR (d.path = ?2 AND f.name = ?3))",
            params![target, parent_of(target), file_name_of(target)],
        )?;
        tx.execute(
            "DELETE FROM dirs WHERE path = ?1 OR path LIKE ?1 || '/%'",
            params![target],
        )?;
        tx.commit()?;
        Ok(slots)
    }

    fn delete_files_chunks(tx: &Connection, file_ids: &[i64]) -> CoreResult<Vec<i64>> {
        if file_ids.is_empty() {
            return Ok(Vec::new());
        }
        let placeholders = vec!["?"; file_ids.len()].join(",");
        let sql_files = format!("SELECT id FROM files WHERE id IN ({})", placeholders);
        let mut slots = Vec::new();
        {
            let mut statement = tx.prepare(&sql_files)?;
            let ids: Vec<i64> = statement
                .query_map(params_from_iter(file_ids), |row| row.get(0))?
                .collect::<Result<_, _>>()?;
            let mut chunk_slots = tx.prepare("SELECT vec_slot FROM chunks WHERE file_id = ?1")?;
            for file_id in ids {
                let file_slots: Vec<Option<i64>> = chunk_slots
                    .query_map(params![file_id], |row| row.get(0))?
                    .collect::<Result<_, _>>()?;
                slots.extend(file_slots.into_iter().flatten());
                tx.execute("DELETE FROM chunks WHERE file_id = ?1", params![file_id])?;
            }
        }
        Ok(slots)
    }

    /// Drops all chunks and vector-slot references (vector store rebuild).
    pub fn clear_chunks(&self) -> CoreResult<()> {
        self.conn.lock().unwrap().execute("DELETE FROM chunks", [])?;
        self.conn.lock().unwrap().execute("UPDATE files SET rag = 0, duration = NULL", [])?;
        Ok(())
    }

    pub fn chunk_counts(&self) -> CoreResult<(i64, i64)> {
        let conn = self.conn.lock().unwrap();
        let files: i64 = conn.query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0))?;
        let done: i64 = conn.query_row("SELECT COUNT(*) FROM files WHERE rag = 2", [], |r| r.get(0))?;
        Ok((done, files))
    }

    /// Per-kind (count, bytes) across the catalog.
    pub fn kind_stats(&self, conn: &Connection) -> CoreResult<Vec<(FileKind, i64, i64)>> {
        let mut statement =
            conn.prepare("SELECT kind, COUNT(*), COALESCE(SUM(size), 0) FROM files GROUP BY kind")?;
        let rows = statement
            .query_map([], |row| {
                Ok((FileKind::from_token(&row.get::<_, String>(0)?), row.get(1)?, row.get(2)?))
            })?
            .collect::<Result<_, _>>()?;
        Ok(rows)
    }
}

fn file_row_from(row: &rusqlite::Row<'_>) -> rusqlite::Result<FileRow> {

    Ok(FileRow {
        id: row.get(0)?,
        dir_id: row.get(1)?,
        name: row.get(2)?,
        kind: FileKind::from_token(&row.get::<_, String>(3)?),
        size: row.get(4)?,
        mtime: row.get(5)?,
        duration_secs: row.get(6)?,
        rag: match row.get::<_, i64>(7)? {
            2 => RagStatus::Done,
            3 => RagStatus::Failed,
            1 => RagStatus::Busy,
            _ => RagStatus::Pending,
        },
    })
}

/// Returns the directory id for `path`, recursively creating missing ancestors.
fn ensure_dir(conn: &Connection, path: &Path, mtime: i64) -> CoreResult<i64> {
    let path_text = path.to_string_lossy().into_owned();
    if let Some(id) = conn
        .query_row("SELECT id FROM dirs WHERE path = ?1", params![path_text], |r| r.get(0))
        .optional()?
    {
        return Ok(id);
    }
    let parent_id = match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() && parent != path => {
            Some(ensure_dir(conn, parent, mtime)?)
        }
        _ => None,
    };
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    conn.execute(
        "INSERT INTO dirs(parent_id, name, path, mtime) VALUES (?1, ?2, ?3, ?4)
         ON CONFLICT(path) DO NOTHING",
        params![parent_id, name, path_text, mtime],
    )?;
    Ok(conn.query_row("SELECT id FROM dirs WHERE path = ?1", params![path_text], |r| r.get(0))?)
}

fn parent_of(path: &str) -> &str {
    path.rsplit_once('/').map(|(parent, _)| parent).unwrap_or("")
}

fn file_name_of(path: &str) -> &str {
    path.rsplit_once('/').map(|(_, name)| name).unwrap_or(path)
}
