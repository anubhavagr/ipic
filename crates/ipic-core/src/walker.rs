//! Multi-threaded filesystem walker: a shared directory queue feeds N workers,
//! discovered files stream to a single collector channel. Symlinks never followed.

use crate::{CoreResult, FileKind};
use crossbeam_channel::{unbounded, Sender};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Debug)]
pub enum WalkItem {
    Dir { path: PathBuf, mtime: i64 },
    File { path: PathBuf, kind: FileKind, size: i64, mtime: i64 },
}

#[derive(Debug, Default)]
pub struct WalkStats {
    pub files: u64,
    pub dirs: u64,
    pub unreadable: u64,
    pub elapsed_secs: f64,
}

/// Walks `roots` in parallel using up to `threads` workers, streaming entries to `sink`.
/// Workers block on the directory queue; termination when no work remains (`pending == 0`).
pub fn walk_parallel(
    roots: &[PathBuf],
    skip_dir_names: &[String],
    threads: usize,
    sink: Sender<WalkItem>,
) -> CoreResult<WalkStats> {
    let threads = threads.max(1);
    let skip: HashSet<String> = skip_dir_names.iter().cloned().collect();
    // Deduplicate + canonicalize roots.
    let mut roots: Vec<PathBuf> = {
        let mut seen = HashSet::new();
        roots
            .iter()
            .filter_map(|r| std::fs::canonicalize(r).ok())
            .filter(|r| r.is_dir())
            .filter(|r| seen.insert(r.clone()))
            .collect()
    };
    roots.sort_unstable();
    // Drop roots nested inside another root.
    let mut compact: Vec<PathBuf> = Vec::new();
    for root in roots {
        if !compact.iter().any(|kept| root.starts_with(kept)) {
            compact.push(root);
        }
    }

    let (dir_tx, dir_rx) = unbounded::<PathBuf>();
    let dir_rx = Arc::new(Mutex::new(dir_rx));
    let skip = Arc::new(skip);
    let pending = Arc::new(AtomicU64::new(0));
    let files = Arc::new(AtomicU64::new(0));
    let dirs = Arc::new(AtomicU64::new(0));
    let unreadable = Arc::new(AtomicU64::new(0));

    for root in &compact {
        pending.fetch_add(1, Ordering::Release);
        dir_tx.send(root.clone())?;
    }

    let started = Instant::now();
    let mut workers = Vec::with_capacity(threads);
    for _ in 0..threads {
        let dir_rx = Arc::clone(&dir_rx);
        let dir_tx = dir_tx.clone();
        let skip = Arc::clone(&skip);
        let pending = Arc::clone(&pending);
        let files = Arc::clone(&files);
        let dirs = Arc::clone(&dirs);
        let unreadable = Arc::clone(&unreadable);
        let sink = sink.clone();
        workers.push(std::thread::spawn(move || loop {
            // Lock is held only for the receive, never while reading directory contents.
            let next = {
                let receiver = dir_rx.lock().unwrap();
                receiver.recv_timeout(std::time::Duration::from_millis(100))
            };
            match next {
                Ok(dir) => {
                process_dir(&dir, &skip, &dir_tx, &pending, &sink, &files, &dirs, &unreadable);
                    if pending.fetch_sub(1, Ordering::AcqRel) == 1 {
                        break; // was last outstanding directory: queue drained
                    }
                }
                Err(_) if pending.load(Ordering::Acquire) == 0 => break,
                Err(_) => continue, // timeout while others still work: re-check
            }
        }));
    }
    for worker in workers {
        worker.join().map_err(|_| anyhow::anyhow!("walker thread panicked"))?;
    }
    drop(sink); // signal collector that the walk is finished

    Ok(WalkStats {
        files: files.load(Ordering::Relaxed),
        dirs: dirs.load(Ordering::Relaxed),
        unreadable: unreadable.load(Ordering::Relaxed),
        elapsed_secs: started.elapsed().as_secs_f64(),
    })
}

#[allow(clippy::too_many_arguments)]
fn process_dir(
    dir: &Path,
    skip: &HashSet<String>,
    dir_tx: &Sender<PathBuf>,
    pending: &AtomicU64,
    sink: &Sender<WalkItem>,
    files: &AtomicU64,
    dirs: &AtomicU64,
    unreadable: &AtomicU64,
) {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(_) => {
            unreadable.fetch_add(1, Ordering::Relaxed);
            return;
        }
    };
    dirs.fetch_add(1, Ordering::Relaxed);
    let mtime = entry_mtime(dir);
    let _ = sink.send(WalkItem::Dir { path: dir.to_path_buf(), mtime });
    for entry in entries.flatten() {
        let Ok(file_type) = entry.file_type() else { continue };
        if file_type.is_symlink() {
            continue; // never traverse or index symlinks (cycle safety)
        }
        let path = entry.path();
        let Ok(meta) = entry.metadata() else { continue };
        let mtime = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        if file_type.is_dir() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if !skip.contains(name.as_ref()) {
                pending.fetch_add(1, Ordering::Release);
                if dir_tx.send(path).is_err() {
                    pending.fetch_sub(1, Ordering::AcqRel);
                }
            }
        } else {
            files.fetch_add(1, Ordering::Relaxed);
            let _ = sink.send(WalkItem::File {
                kind: crate::kind_for_path(&path),
                size: meta.len() as i64,
                mtime,
                path,
            });
        }
    }
}

fn entry_mtime(path: &Path) -> i64 {
    std::fs::symlink_metadata(path)
        .ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}
