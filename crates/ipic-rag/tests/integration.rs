//! Integration: engine end-to-end over a temp corpus — parallel walk →
//! per-root shards → chunk/embed → hybrid search, plus the background
//! lifecycle (modify re-index, delete frees embeddings). All local with the
//! deterministic hashing embedder and whisper/vision disabled (no downloads).

use ipic_core::{Config, FileFilter, FileKind, SortKey};
use ipic_rag::{Engine, EngineEvent};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn test_config(roots: Vec<PathBuf>, data_directory: PathBuf) -> (Config, PathBuf) {
    (
        Config {
            roots,
            whisper_model: "none".into(),
            embedder: "hashing".into(),
            image_embedder: "none".into(),
            ..Default::default()
        },
        data_directory,
    )
}

fn build_corpus(root: &Path) {
    let notes = root.join("documents/notes");
    std::fs::create_dir_all(&notes).unwrap();
    std::fs::write(
        notes.join("quarterly-roadmap.md"),
        "# Roadmap\n\nThe quarterly roadmap prioritizes on-device semantic search, \
         offline speech transcription and low memory vector indexing for the next release.",
    )
    .unwrap();
    std::fs::write(notes.join("grocery-list.txt"), "milk eggs bread butter").unwrap();
    let cache = root.join("node_modules/pkg");
    std::fs::create_dir_all(&cache).unwrap();
    std::fs::write(cache.join("noise.js"), "roadmap roadmap roadmap (should be skipped)").unwrap();
}

/// Drains events until the engine reports idle (or the deadline expires).
fn wait_until_idle(engine: &Arc<Engine>, timeout: Duration) -> bool {
    let started = Instant::now();
    let mut saw_scan = false;
    while started.elapsed() < timeout {
        while let Ok(event) = engine.events.try_recv() {
            if matches!(event, EngineEvent::ScanFinished { .. }) {
                saw_scan = true;
            }
        }
        let status = engine.status();
        if saw_scan && !status.scanning && status.pending == 0 && status.total_files > 0 {
            return true;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    false
}

#[test]
fn scan_shard_and_search_end_to_end() {
    let base = std::env::temp_dir().join(format!("ipic-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let corpus = base.join("corpus");
    build_corpus(&corpus);
    let (config, data_directory) = test_config(vec![corpus.clone()], base.join("data"));

    let engine = Engine::launch_with_data_dir(config, data_directory).unwrap();
    assert!(wait_until_idle(&engine, Duration::from_secs(30)), "engine must settle");
    let status = engine.status();
    assert_eq!(status.total_files, 2, "skipped directories must not be indexed");

    // Browsing: filters + sorting through the engine facade.
    let notes_path = corpus.join("documents/notes").canonicalize().unwrap();
    let notes_directory = engine.dir_by_path(&notes_path.to_string_lossy()).unwrap();
    let (_directories, all_entries) = engine
        .directory_children(Some(&notes_directory), &FileFilter::default(), SortKey::Name, true, 100);
    assert_eq!(all_entries.len(), 2);
    let text_filter = FileFilter { kinds: vec![FileKind::Text], ..Default::default() };
    let (_dirs, text_entries) =
        engine.directory_children(Some(&notes_directory), &text_filter, SortKey::Size, false, 100);
    assert!(text_entries.iter().all(|(entry, _)| entry.kind == FileKind::Text));
    assert!(text_entries[0].0.size >= text_entries[1].0.size, "size sort descending");

    // Hybrid search: semantic lane finds the roadmap document.
    let outcome = engine.semantic_search("quarterly roadmap priorities", 10).unwrap();
    assert!(!outcome.hits.is_empty(), "search must return hits");
    assert!(
        outcome.hits[0].path.ends_with("quarterly-roadmap.md"),
        "top hit should be the roadmap document, got {}",
        outcome.hits[0].path
    );
    assert!(outcome.hits[0].sources.semantic, "semantic lane must contribute");

    let keyword_outcome = engine.semantic_search("milk", 10).unwrap();
    assert!(keyword_outcome.hits.iter().any(|hit| hit.path.ends_with("grocery-list.txt")));

    let name_outcome = engine.semantic_search("roadmap", 10).unwrap();
    assert!(name_outcome.hits.iter().any(|hit| hit.path.ends_with("quarterly-roadmap.md")));

    engine.shutdown();
    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn shards_partition_roots_and_merge_search() {
    let base = std::env::temp_dir().join(format!("ipic-shards-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let left = base.join("left");
    let right = base.join("right");
    std::fs::create_dir_all(&left).unwrap();
    std::fs::create_dir_all(&right).unwrap();
    std::fs::write(left.join("alpha-notes.md"), "the alpha telescope observes distant nebulae").unwrap();
    std::fs::write(right.join("beta-notes.md"), "the beta telescope tracks asteroid orbits").unwrap();

    let (config, data_directory) = test_config(vec![left.clone(), right.clone()], base.join("data"));
    let engine = Engine::launch_with_data_dir(config, data_directory).unwrap();
    assert!(wait_until_idle(&engine, Duration::from_secs(30)), "engine must settle");
    assert_eq!(engine.status().total_files, 2);
    assert_eq!(engine.current_roots().len(), 2, "one shard per root");

    // Cross-shard query: both roots surface, ranked by content.
    let outcome = engine.semantic_search("telescope astronomy", 10).unwrap();
    let paths: Vec<&str> = outcome.hits.iter().map(|hit| hit.path.as_str()).collect();
    assert!(paths.iter().any(|path| path.ends_with("alpha-notes.md")));
    assert!(paths.iter().any(|path| path.ends_with("beta-notes.md")));

    engine.shutdown();
    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn vision_content_search_ranks_by_pixels() {
    // Opt-in (IPIC_VISION_TEST=1): uses the real CLIP towers, no network
    // after first download. Hermetic runs keep vision off.
    if std::env::var("IPIC_VISION_TEST").is_err() {
        return;
    }
    let base = std::env::temp_dir().join(format!("ipic-vision-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let corpus = base.join("corpus");
    std::fs::create_dir_all(&corpus).unwrap();
    // Two flat-color images with neutral names: only pixels can separate them.
    let red = image::RgbImage::from_pixel(128, 128, image::Rgb([200, 30, 30]));
    let blue = image::RgbImage::from_pixel(128, 128, image::Rgb([30, 30, 200]));
    red.save(corpus.join("asset-one.png")).unwrap();
    blue.save(corpus.join("asset-two.png")).unwrap();

    let config = Config {
        roots: vec![corpus.clone()],
        whisper_model: "none".into(),
        embedder: "hashing".into(),
        image_embedder: "clip-vit-b32".into(),
        ..Default::default()
    };
    let engine = Engine::launch_with_data_dir(config, base.join("data")).unwrap();
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        let status = engine.status();
        if status.pending == 0 && status.image_vector_count >= 2 {
            break;
        }
        assert!(Instant::now() < deadline, "vision indexing must finish: {status:?}");
        std::thread::sleep(Duration::from_millis(250));
    }
    let outcome = engine.semantic_search("solid red color", 5).unwrap();
    let top = outcome.hits.first().expect("vision search must return hits");
    assert!(
        top.path.ends_with("asset-one.png") && top.sources.vision,
        "red query must rank the red image by content, got {} (lanes {:?})",
        top.path,
        top.sources
    );
    engine.shutdown();
    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn modify_reindexes_and_delete_frees_embeddings() {
    let base = std::env::temp_dir().join(format!("ipic-lifecycle-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let corpus = base.join("corpus");
    std::fs::create_dir_all(&corpus).unwrap();
    let target = corpus.join("journal.md");
    std::fs::write(&target, "original entry about sailing").unwrap();

    let (config, data_directory) = test_config(vec![corpus.clone()], base.join("data"));
    let engine = Engine::launch_with_data_dir(config, data_directory).unwrap();
    assert!(wait_until_idle(&engine, Duration::from_secs(30)), "engine must settle");
    let vectors_after_first_index = engine.status().vector_count;
    assert!(vectors_after_first_index > 0, "text chunks must be embedded");

    // Modification on disk: the watcher must reset and re-embed the file.
    std::fs::write(&target, "revised entry about mountaineering expeditions").unwrap();
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut reindexed = false;
    while Instant::now() < deadline {
        let outcome = engine.semantic_search("mountaineering expeditions", 5).unwrap();
        if outcome.hits.iter().any(|hit| hit.path.ends_with("journal.md")) {
            reindexed = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    assert!(reindexed, "modified content must be re-indexed and findable");
    // Old content is gone: the stale chunk's slot was freed, not duplicated.
    let stale = engine.semantic_search("sailing", 5).unwrap();
    assert!(!stale.hits.iter().any(|hit| hit.path.ends_with("journal.md") && hit.sources.semantic));

    // Deletion: rows, chunks and vector slots all disappear.
    let vectors_before_delete = engine.status().vector_count;
    std::fs::remove_file(&target).unwrap();
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut deleted = false;
    while Instant::now() < deadline {
        while engine.events.try_recv().is_ok() {}
        let outcome = engine.semantic_search("mountaineering", 5).unwrap();
        if !outcome.hits.iter().any(|hit| hit.path.ends_with("journal.md")) {
            deleted = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    assert!(deleted, "deleted file must vanish from search");
    assert!(engine.status().vector_count < vectors_before_delete, "slots must be freed");

    engine.shutdown();
    let _ = std::fs::remove_dir_all(&base);
}
