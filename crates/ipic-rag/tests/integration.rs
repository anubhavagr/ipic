//! Integration: parallel walk → catalog → chunk/embed → hybrid search, all local
//! with the deterministic hashing embedder (no model downloads, no network).

use ipic_core::catalog::{Catalog, NewFile};
use ipic_core::walker::{self, WalkItem};
use ipic_core::{FileFilter, FileKind, SortKey};
use ipic_rag::embed::{HashingEmbedder, TextEmbedder};
use ipic_rag::extract;
use ipic_rag::search::semantic_search;
use ipic_rag::vector_store::VectorStore;

fn build_corpus(root: &std::path::Path) {
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

fn index_corpus(catalog: &Catalog, root: &std::path::Path, vector_store: &mut VectorStore) {
    let (item_sender, item_receiver) = crossbeam_channel::unbounded::<WalkItem>();
    let skip_names = vec!["node_modules".to_string()];
    let stats = walker::walk_parallel(&[root.to_path_buf()], &skip_names, 4, item_sender).unwrap();
    assert_eq!(stats.files, 2, "skipped directories must not be indexed");

    let mut directory_ids = catalog.dir_id_map().unwrap();
    for item in item_receiver.try_recv_iter() {
        match item {
            WalkItem::Dir { path, mtime } => {
                let path_text = path.to_string_lossy().into_owned();
                catalog.upsert_dirs(&[(path, mtime)]).unwrap();
                if let Some(id) = catalog.dir_id_by_path(&path_text) {
                    directory_ids.insert(path_text, id);
                }
            }
            WalkItem::File { path, kind, size, mtime } => {
                let parent = path.parent().unwrap().to_string_lossy().into_owned();
                let file_name = path.file_name().unwrap().to_string_lossy().into_owned();
                catalog
                    .upsert_files(&[NewFile {
                        dir_id: *directory_ids.get(&parent).unwrap(),
                        name: file_name,
                        kind,
                        size,
                        mtime,
                    }])
                    .unwrap();
                // Inline "pipeline": text files get chunked, embedded, indexed.
                if kind == FileKind::Text {
                    let text = extract::extract_text(&path, kind, 1 << 20).unwrap().unwrap_or_default();
                    let chunks = extract::chunk_text(&text);
                    if !chunks.is_empty() {
                        let references: Vec<(i64, &str)> =
                            chunks.iter().map(|chunk| (last_file_id(catalog), chunk.as_str())).collect();
                        let rowids = catalog.insert_chunks(&references).unwrap();
                        let embedder = HashingEmbedder;
                        let embedded: Vec<Vec<f32>> =
                            chunks.iter().map(|chunk| embedder.embed_batch(&[chunk.clone()]).unwrap()[0].clone()).collect();
                        let slots = vector_store.append_batch(&rowids, &embedded).unwrap();
                        let assignments: Vec<(i64, i64)> =
                            rowids.iter().cloned().zip(slots.iter().cloned()).collect();
                        catalog.assign_vector_slots(&assignments).unwrap();
                    }
                }
            }
        }
    }
}

fn last_file_id(catalog: &Catalog) -> i64 {
    let connection = catalog.reader().unwrap();
    connection.query_row("SELECT MAX(id) FROM files", [], |row| row.get(0)).unwrap()
}

// crossbeam receivers expose iter(); small helper for readability.
trait TryRecvIter: Sized {
    fn try_recv_iter(self) -> Vec<WalkItem>;
}

impl TryRecvIter for crossbeam_channel::Receiver<WalkItem> {
    fn try_recv_iter(self) -> Vec<WalkItem> {
        let mut items = Vec::new();
        while let Ok(item) = self.try_recv() {
            items.push(item);
        }
        items
    }
}

#[test]
fn scan_catalog_and_semantic_search_end_to_end() {
    let base = std::env::temp_dir().join(format!("ipic-e2e-{}", std::process::id()));
    let root = base.join("corpus");
    let _ = std::fs::remove_dir_all(&base);
    build_corpus(&root);

    let catalog = Catalog::open(&base.join("data/catalog.db")).unwrap();
    let embedder = HashingEmbedder;
    let (mut vector_store, compatible) =
        VectorStore::open(&base.join("data"), embedder.dim(), embedder.model_id()).unwrap();
    assert!(compatible);
    index_corpus(&catalog, &root, &mut vector_store);

    // Directory browsing: filters + sorting work against the catalog.
    let connection = catalog.reader().unwrap();
    let notes_path = root.join("documents/notes").canonicalize().unwrap();
    let notes_dir = catalog.dir_by_path(&connection, &notes_path.to_string_lossy()).unwrap().unwrap();
    let all_entries =
        catalog.children(&connection, Some(notes_dir.id), &FileFilter::default(), SortKey::Name, true, 100).unwrap();
    assert_eq!(all_entries.len(), 2);
    let text_only_filter = FileFilter { kinds: vec![FileKind::Text], ..Default::default() };
    let text_entries =
        catalog.children(&connection, Some(notes_dir.id), &text_only_filter, SortKey::Size, false, 100).unwrap();
    assert!(text_entries.iter().all(|entry| entry.kind == FileKind::Text));
    assert!(text_entries[0].size >= text_entries[1].size, "size sort descending");

    // Hybrid search: semantic lane finds the roadmap document; grocery list must not rank.
    let outcome = semantic_search(&catalog, &connection, &vector_store, &embedder, "quarterly roadmap priorities", 10).unwrap();
    assert!(!outcome.hits.is_empty(), "search must return hits");
    assert!(
        outcome.hits[0].path.ends_with("quarterly-roadmap.md"),
        "top hit should be the roadmap document, got {}",
        outcome.hits[0].path
    );
    assert!(outcome.hits[0].sources.semantic, "semantic lane must contribute");

    // Keyword-only query exercises the FTS5 lane.
    let keyword_outcome = semantic_search(&catalog, &connection, &vector_store, &embedder, "milk", 10).unwrap();
    assert!(keyword_outcome.hits.iter().any(|hit| hit.path.ends_with("grocery-list.txt")));

    // Filename lane.
    let name_outcome = semantic_search(&catalog, &connection, &vector_store, &embedder, "roadmap", 10).unwrap();
    assert!(name_outcome.hits.iter().any(|hit| hit.path.ends_with("quarterly-roadmap.md")));

    let _ = std::fs::remove_dir_all(&base);
}
