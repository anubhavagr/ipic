//! Live filesystem watcher: debounced notify events keep the catalog in sync;
//! directory rescans and freed vector slots are forwarded to the engine.

use crate::catalog::{Catalog, NewFile};
use crate::CoreResult;
use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher as NotifyWatcher};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Debug)]
pub enum WatchUpdate {
    /// Directory subtree changed on disk; engine re-walks it incrementally.
    Rescan(PathBuf),
    /// Vector slots released by deletions; engine frees them in the vector store.
    FreedSlots(Vec<i64>),
}

/// Starts watching `roots` recursively. Updates flow through `updates`; errors are swallowed
/// (a watcher failure must never take the app down).
pub fn spawn(
    catalog: Arc<Catalog>,
    roots: Vec<PathBuf>,
    updates: crossbeam_channel::Sender<WatchUpdate>,
) -> CoreResult<Arc<AtomicBool>> {
    let (event_tx, event_rx) = crossbeam_channel::unbounded::<Event>();
    let mut watcher = RecommendedWatcher::new(
        move |event: notify::Result<Event>| {
            if let Ok(event) = event {
                let _ = event_tx.send(event);
            }
        },
        notify::Config::default(),
    )?;
    for root in &roots {
        let _ = watcher.watch(root, RecursiveMode::Recursive);
    }

    let stop = Arc::new(AtomicBool::new(false));
    let thread_stop = Arc::clone(&stop);
    let debouncer = std::thread::Builder::new().name("ipic-watcher".into()).spawn(move || {
        let catalog = Arc::clone(&catalog);
        let updates = updates.clone();
        let mut pending: HashSet<PathBuf> = HashSet::new();
        let mut removals: HashSet<PathBuf> = HashSet::new();
        let mut window_end = Instant::now();
        let _watcher = watcher; // keep alive for the thread's lifetime
        while !thread_stop.load(Ordering::Relaxed) {
            if let Ok(event) = event_rx.recv_timeout(Duration::from_millis(120)) {
                let removed = matches!(event.kind, EventKind::Remove(_));
                for path in event.paths {
                    if removed {
                        removals.insert(path);
                    } else {
                        pending.insert(path);
                    }
                }
                window_end = Instant::now() + Duration::from_millis(350);
            }
            if Instant::now() >= window_end && (!pending.is_empty() || !removals.is_empty()) {
                for path in removals.drain() {
                    if let Ok(slots) = catalog.remove_path(&path.to_string_lossy()) {
                        let _ = updates.send(WatchUpdate::FreedSlots(slots));
                    }
                }
                for path in pending.drain() {
                    if !path.exists() {
                        continue;
                    }
                    if path.is_dir() {
                        let _ = catalog.remove_path(&path.to_string_lossy());
                        let _ = updates.send(WatchUpdate::Rescan(path));
                    } else if let Err(error) = upsert_single_file(&catalog, &path) {
                        let _ = updates.send(WatchUpdate::Rescan(path.parent().map(|p| p.to_path_buf()).unwrap_or(path)));
                        let _ = error; // fall back to a parent rescan
                    }
                }
            }
        }
    })?;
    let _ = debouncer;
    Ok(stop)
}

/// Inserts/updates one file row, creating any missing ancestor directories.
fn upsert_single_file(catalog: &Catalog, path: &Path) -> CoreResult<()> {
    let reader = catalog.reader()?;
    let parent = path.parent().unwrap_or(path);
    // Walk up to the deepest already-known ancestor, then insert the missing chain.
    let mut missing = Vec::new();
    let mut current = Some(parent.to_path_buf());
    let mut known_dir_id: Option<i64> = None;
    while let Some(dir) = current {
        if let Some(row) = catalog.dir_by_path(&reader, &dir.to_string_lossy())? {
            known_dir_id = Some(row.id);
            break;
        }
        missing.push(dir.clone());
        current = dir.parent().map(|p| p.to_path_buf());
    }
    for dir in missing.iter().rev() {
        let mtime = std::fs::symlink_metadata(dir).ok().and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs() as i64).unwrap_or(0);
        catalog.upsert_dirs(&[(dir.clone(), mtime)])?;
    }
    let dir_id = match known_dir_id {
        Some(id) => id,
        None => catalog
            .dir_by_path(&reader, &parent.to_string_lossy())?
            .map(|row| row.id)
            .ok_or_else(|| anyhow::anyhow!("parent directory missing: {}", parent.display()))?,
    };
    let metadata = std::fs::symlink_metadata(path)?;
    let mtime = metadata.modified().ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64).unwrap_or(0);
    catalog.upsert_files(&[NewFile {
        dir_id,
        name: path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
        kind: crate::kind_for_path(path),
        size: metadata.len() as i64,
        mtime,
    }])?;
    Ok(())
}
