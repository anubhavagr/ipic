//! Live filesystem watcher: debounced notify events only — the engine owns
//! every catalog decision, this side never touches storage.

use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher as NotifyWatcher};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Debug)]
pub enum WatchUpdate {
    /// A file or directory appeared or changed on disk.
    Appeared(PathBuf),
    /// A file or directory vanished from disk.
    Vanished(PathBuf),
}

/// Watches `roots` recursively and forwards debounced updates. Set the
/// returned flag to stop; watcher failures are swallowed (never crash the app).
pub fn spawn(roots: Vec<PathBuf>, updates: crossbeam_channel::Sender<WatchUpdate>) -> Arc<AtomicBool> {
    let stop = Arc::new(AtomicBool::new(false));
    let (event_sender, event_receiver) = crossbeam_channel::unbounded::<Event>();
    let watcher = RecommendedWatcher::new(
        move |event: notify::Result<Event>| {
            if let Ok(event) = event {
                let _ = event_sender.send(event);
            }
        },
        notify::Config::default(),
    );
    let mut watcher = match watcher {
        Ok(watcher) => watcher,
        Err(_) => return stop,
    };
    for root in &roots {
        let _ = watcher.watch(root, RecursiveMode::Recursive);
    }

    let thread_stop = Arc::clone(&stop);
    std::thread::Builder::new().name("ipic-watcher".into()).spawn(move || {
        let mut appeared: HashSet<PathBuf> = HashSet::new();
        let mut vanished: HashSet<PathBuf> = HashSet::new();
        let mut window_end = Instant::now();
        let _watcher = watcher; // keeps the notify handle alive for the thread
        while !thread_stop.load(Ordering::Relaxed) {
            if let Ok(event) = event_receiver.recv_timeout(Duration::from_millis(120)) {
                let removed = matches!(event.kind, EventKind::Remove(_));
                for path in event.paths {
                    // A remove followed by a recreate inside one window is a
                    // rename/replace: treat it purely as an appearance.
                    if removed {
                        appeared.remove(&path);
                        vanished.insert(path);
                    } else {
                        vanished.remove(&path);
                        appeared.insert(path);
                    }
                }
                window_end = Instant::now() + Duration::from_millis(350);
            }
            if Instant::now() >= window_end && (!appeared.is_empty() || !vanished.is_empty()) {
                for path in vanished.drain() {
                    let _ = updates.send(WatchUpdate::Vanished(path));
                }
                for path in appeared.drain() {
                    let _ = updates.send(WatchUpdate::Appeared(path));
                }
            }
        }
    })
    .ok();
    stop
}
