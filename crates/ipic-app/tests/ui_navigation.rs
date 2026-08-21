//! Drives the real GUI with synthetic pointer/keyboard input (egui_kittest)
//! and asserts the full navigation matrix: folder entry, breadcrumbs, history
//! buttons, sidebar locations/tree, and keyboard navigation.

use egui_kittest::kittest::Queryable;
use egui_kittest::Harness;
use ipic::{theme, IpicApp};
use ipic_core::Config;
use ipic_rag::{Engine, EngineStatus};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

struct TestWorld {
    corpus: PathBuf,
    data_directory: PathBuf,
}

impl TestWorld {
    fn create(tag: &str) -> Self {
        let base = std::env::temp_dir().join(format!("ipic-nav-{tag}-{}", std::process::id()));
        let corpus = base.join("corpus");
        let documents = corpus.join("documents");
        std::fs::create_dir_all(documents.join("notes")).unwrap();
        std::fs::create_dir_all(corpus.join("photos")).unwrap();
        std::fs::write(documents.join("alpha-zebra.md"), "Alphabetized canary document.").unwrap();
        std::fs::write(
            documents.join("mission-briefing.md"),
            "The orbital rendezvous plan covers launch windows and docking maneuvers.",
        )
        .unwrap();
        std::fs::write(documents.join("notes/deep-thoughts.txt"), "Nested folder resident.").unwrap();
        let png_header: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        std::fs::write(corpus.join("photos/beach-sunset.png"), png_header).unwrap();
        Self { corpus, data_directory: base.join("data") }
    }

    fn config(&self) -> Config {
        Config {
            roots: vec![self.corpus.clone()],
            whisper_model: "none".into(),
            embedder: "hashing".into(),
            ..Default::default()
        }
    }
}

impl Drop for TestWorld {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(self.corpus.parent().unwrap());
    }
}

fn build_harness(world: &TestWorld) -> (Harness<'static, IpicApp>, Arc<Engine>) {
    let config = world.config();
    let engine = Engine::launch_with_data_dir(config.clone(), world.data_directory.clone()).unwrap();
    let mut application = IpicApp::new(Arc::clone(&engine), config);
    // Capture open requests so keyboard tests never shell out to the OS.
    application.open_handler = Some(Arc::new(|_path| {}));
    // A small simulated clock keeps click sequences inside egui's 0.3 s
    // double-click window; a wide window keeps the breadcrumb from clipping.
    let harness = egui_kittest::HarnessBuilder::default()
        .with_size(egui::vec2(1600.0, 1000.0))
        .with_step_dt(0.001)
        .build_eframe(move |creation_context| {
            theme::apply(&creation_context.egui_ctx);
            application
        });
    (harness, engine)
}

fn wait_until_indexed(engine: &Arc<Engine>) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while Instant::now() < deadline {
        let status: EngineStatus = engine.status();
        if !status.scanning && status.pending == 0 && status.total_files > 0 {
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!("engine did not finish indexing in time: {:?}", engine.status());
}

/// Emits a click with real-input timing: move, press, release — one frame apart.
fn click_at(harness: &mut Harness<'_, IpicApp>, position: egui::Pos2) {
    let modifiers = egui::Modifiers::default();
    let button = egui::PointerButton::Primary;
    harness.event(egui::Event::PointerMoved(position));
    harness.run();
    harness.event(egui::Event::PointerButton { pos: position, button, pressed: true, modifiers });
    harness.run();
    harness.event(egui::Event::PointerButton { pos: position, button, pressed: false, modifiers });
    harness.run();
}

/// Two rapid clicks on one spot. Any earlier click must first fall outside
/// egui's 0.6 s triple-click window, or the second click counts as a triple.
fn double_click_at(harness: &mut Harness<'_, IpicApp>, position: egui::Pos2) {
    harness.run_steps(700);
    click_at(harness, position);
    click_at(harness, position);
}

fn current_directory_path(harness: &Harness<'_, IpicApp>) -> Option<String> {
    harness.state().current_directory.as_ref().map(|dir| dir.path.clone())
}

fn selected_file_name(harness: &Harness<'_, IpicApp>) -> Option<String> {
    harness.state().selected_file.as_ref().map(|(file, _)| file.name.clone())
}

/// Center of the unique label whose rect passes `keep` (regions disambiguate
/// the sidebar, the top bar, and the browse table, which reuse texts).
fn label_center(
    harness: &Harness<'_, IpicApp>,
    text: &str,
    keep: impl Fn(&egui::Rect) -> bool,
) -> egui::Pos2 {
    let matches: Vec<_> = harness
        .query_all_by_label(text)
        .filter(|node| keep(&node.rect()))
        .collect();
    assert_eq!(
        matches.len(),
        1,
        "expected exactly one label {text:?} in the requested region"
    );
    matches[0].rect().center()
}

/// Top-bar region (history buttons, breadcrumb segments and links).
fn top_bar_label(harness: &Harness<'_, IpicApp>, text: &str) -> egui::Pos2 {
    label_center(harness, text, |rect| rect.max.y < 45.0)
}

/// Top-bar history button; breadcrumb "›" separators share the forward
/// button's glyph, so only a button-sized rect (min 26 px wide) qualifies.
fn top_bar_button(harness: &Harness<'_, IpicApp>, glyph: &str) -> egui::Pos2 {
    label_center(harness, glyph, |rect| rect.max.y < 45.0 && rect.width() >= 20.0)
}

/// Sidebar region (locations, kind filters, directory tree).
fn sidebar_label(harness: &Harness<'_, IpicApp>, text: &str) -> egui::Pos2 {
    label_center(harness, text, |rect| rect.min.y > 45.0 && rect.max.x < 255.0)
}

/// Leftmost sidebar label — the root row of the directory tree.
fn sidebar_tree_root_label(harness: &Harness<'_, IpicApp>, glyph: &str) -> egui::Pos2 {
    let matches: Vec<_> = harness
        .query_all_by_label(glyph)
        .filter(|node| {
            let rect = node.rect();
            rect.min.y > 45.0 && rect.max.x < 255.0
        })
        .collect();
    assert!(!matches.is_empty(), "expected a tree toggle {glyph:?} in the sidebar");
    matches
        .iter()
        .min_by_key(|node| node.rect().min.x as i32)
        .unwrap()
        .rect()
        .center()
}

fn sidebar_location_root(harness: &Harness<'_, IpicApp>) -> egui::Pos2 {
    let matches: Vec<_> = harness.query_all_by_label_contains("Volume").collect();
    assert_eq!(matches.len(), 1, "expected exactly one location root row");
    matches[0].rect().center()
}

/// Browse-table region (central panel; the details panel lives further right).
fn table_row_position(harness: &Harness<'_, IpicApp>, text: &str) -> egui::Pos2 {
    label_center(harness, text, |rect| {
        rect.min.x > 260.0 && rect.max.x < 1290.0 && rect.min.y > 45.0
    })
}

fn has_table_label(harness: &Harness<'_, IpicApp>, text: &str) -> bool {
    harness
        .query_all_by_label(text)
        .any(|node| {
            let rect = node.rect();
            rect.min.x > 260.0 && rect.max.x < 1290.0 && rect.min.y > 45.0
        })
}

/// Clicks the sidebar location row of the corpus root.
fn enter_corpus_root(harness: &mut Harness<'_, IpicApp>) {
    let position = sidebar_location_root(harness);
    click_at(harness, position);
    harness.run();
}

/// Enters corpus/documents by double-clicking its row in the browse table.
fn enter_documents_via_table(harness: &mut Harness<'_, IpicApp>) {
    enter_corpus_root(harness);
    let folder_position = table_row_position(harness, "documents");
    double_click_at(harness, folder_position);
    harness.run();
}

#[test]
fn double_click_folder_row_navigates_into_it() {
    let world = TestWorld::create("dbl-folder");
    let (mut harness, engine) = build_harness(&world);
    wait_until_indexed(&engine);
    harness.run();
    enter_corpus_root(&mut harness);
    let folder_position = table_row_position(&harness, "documents");
    double_click_at(&mut harness, folder_position);
    let current = current_directory_path(&harness).unwrap_or_default();
    assert!(
        current.ends_with("/documents"),
        "double-click must enter the folder, got {current:?}"
    );
    // Listing must reload for the new directory.
    assert!(has_table_label(&harness, "alpha-zebra.md"), "listing must show the new folder's files");
}

#[test]
fn breadcrumb_segment_click_returns_to_ancestor() {
    let world = TestWorld::create("crumb-ancestor");
    let (mut harness, engine) = build_harness(&world);
    wait_until_indexed(&engine);
    harness.run();
    enter_documents_via_table(&mut harness);
    let corpus_segment = top_bar_label(&harness, "corpus");
    click_at(&mut harness, corpus_segment);
    let current = current_directory_path(&harness).unwrap_or_default();
    assert!(
        current.ends_with("/corpus"),
        "breadcrumb segment must navigate to the ancestor, got {current:?}"
    );
}

#[test]
fn all_files_breadcrumb_returns_to_library() {
    let world = TestWorld::create("crumb-library");
    let (mut harness, engine) = build_harness(&world);
    wait_until_indexed(&engine);
    harness.run();
    enter_documents_via_table(&mut harness);
    let all_files_link = top_bar_label(&harness, "All files");
    click_at(&mut harness, all_files_link);
    assert!(
        current_directory_path(&harness).is_none(),
        "All files link must return to the library view"
    );
}

#[test]
fn back_and_forward_traverse_history() {
    let world = TestWorld::create("history");
    let (mut harness, engine) = build_harness(&world);
    wait_until_indexed(&engine);
    harness.run();
    enter_corpus_root(&mut harness);
    let back_button = top_bar_button(&harness, "‹");
    click_at(&mut harness, back_button);
    assert!(
        current_directory_path(&harness).is_none(),
        "back must return to the previous view (library)"
    );
    assert_eq!(harness.state().forward_history.len(), 1);
    let forward_button = top_bar_button(&harness, "›");
    click_at(&mut harness, forward_button);
    let current = current_directory_path(&harness).unwrap_or_default();
    assert!(
        current.ends_with("/corpus"),
        "forward must re-enter the directory, got {current:?}"
    );
    assert!(harness.state().forward_history.is_empty());
}

#[test]
fn up_button_navigates_to_parent_then_library() {
    let world = TestWorld::create("up-parent");
    let (mut harness, engine) = build_harness(&world);
    wait_until_indexed(&engine);
    harness.run();
    enter_documents_via_table(&mut harness);
    let up_button = top_bar_button(&harness, "↑");
    click_at(&mut harness, up_button);
    let current = current_directory_path(&harness).unwrap_or_default();
    assert!(
        current.ends_with("/corpus"),
        "up must navigate to the parent directory, got {current:?}"
    );
    click_at(&mut harness, up_button);
    assert!(
        current_directory_path(&harness).is_none(),
        "up from a root must fall back to the library view"
    );
}

#[test]
fn up_from_library_view_is_safe() {
    let world = TestWorld::create("up-library");
    let (mut harness, engine) = build_harness(&world);
    wait_until_indexed(&engine);
    harness.run();
    let up_button = top_bar_button(&harness, "↑");
    click_at(&mut harness, up_button);
    assert!(
        current_directory_path(&harness).is_none(),
        "up from the library view must stay in the library view"
    );
    assert!(
        harness.state().back_history.is_empty(),
        "up from the library view must not pollute the history"
    );
}

#[test]
fn sidebar_location_root_click_navigates() {
    let world = TestWorld::create("sidebar-root");
    let (mut harness, engine) = build_harness(&world);
    wait_until_indexed(&engine);
    harness.run();
    let location_root = sidebar_location_root(&harness);
    click_at(&mut harness, location_root);
    let expected_root = std::fs::canonicalize(&world.corpus).unwrap();
    assert_eq!(
        current_directory_path(&harness),
        Some(expected_root.to_string_lossy().into_owned()),
        "location root click must browse the indexed root"
    );
    // The root's folder rows must appear once navigated there.
    assert!(has_table_label(&harness, "documents"));
}

#[test]
fn sidebar_tree_toggle_expands_and_child_click_navigates() {
    let world = TestWorld::create("sidebar-tree");
    let (mut harness, engine) = build_harness(&world);
    wait_until_indexed(&engine);
    harness.run();
    let expand_toggle = sidebar_tree_root_label(&harness, "▸");
    click_at(&mut harness, expand_toggle);
    harness.run();
    let child_position = sidebar_label(&harness, "documents");
    click_at(&mut harness, child_position);
    let current = current_directory_path(&harness).unwrap_or_default();
    assert!(
        current.ends_with("/documents"),
        "tree child click must navigate into the folder, got {current:?}"
    );
    // Collapsing the root must hide the child rows again.
    let collapse_toggle = sidebar_tree_root_label(&harness, "▾");
    click_at(&mut harness, collapse_toggle);
    harness.run();
    let sidebar_children = harness
        .query_all_by_label("documents")
        .filter(|node| {
            let rect = node.rect();
            rect.min.y > 45.0 && rect.max.x < 255.0
        })
        .count();
    assert_eq!(sidebar_children, 0, "collapsed tree must hide its children");
}

#[test]
fn arrow_keys_move_selection_and_enter_opens_folder() {
    let world = TestWorld::create("keyboard");
    let (mut harness, engine) = build_harness(&world);
    wait_until_indexed(&engine);
    harness.run();
    enter_corpus_root(&mut harness);
    // Root listing: [documents, photos] — both directories.
    harness.key_press(egui::Key::ArrowDown);
    harness.run();
    assert_eq!(harness.state().browse.selected_row, Some(0), "ArrowDown must select the first row");
    harness.key_press(egui::Key::ArrowDown);
    harness.run();
    assert_eq!(harness.state().browse.selected_row, Some(1), "ArrowDown must move down");
    harness.key_press(egui::Key::ArrowUp);
    harness.run();
    assert_eq!(harness.state().browse.selected_row, Some(0), "ArrowUp must move up");
    harness.key_press(egui::Key::Enter);
    harness.run();
    let current = current_directory_path(&harness).unwrap_or_default();
    assert!(
        current.ends_with("/documents"),
        "Enter must open the selected folder, got {current:?}"
    );
    assert!(
        selected_file_name(&harness).is_none(),
        "entering a folder must not leave a file selected"
    );
    assert!(has_table_label(&harness, "alpha-zebra.md"), "listing must show the entered folder");
}

#[test]
fn navigation_clears_stale_selection() {
    let world = TestWorld::create("clear-selection");
    let (mut harness, engine) = build_harness(&world);
    wait_until_indexed(&engine);
    harness.run();
    // Library listing: [alpha-zebra.md, beach-sunset.png, deep-thoughts.txt,
    // mission-briefing.md]; select the second row.
    let beach_sunset_row = table_row_position(&harness, "beach-sunset.png");
    click_at(&mut harness, beach_sunset_row);
    assert_eq!(selected_file_name(&harness).as_deref(), Some("beach-sunset.png"));
    // Enter documents via the sidebar tree; the stale row index must not
    // resurrect a selection in the new listing.
    let expand_toggle = sidebar_tree_root_label(&harness, "▸");
    click_at(&mut harness, expand_toggle);
    harness.run();
    let documents_child = sidebar_label(&harness, "documents");
    click_at(&mut harness, documents_child);
    harness.run();
    assert!(
        selected_file_name(&harness).is_none(),
        "navigation must clear the previous selection, got {:?}",
        selected_file_name(&harness)
    );
    assert_eq!(harness.state().browse.selected_row, None);
    // Selecting inside the folder and going back must clear it too.
    let briefing_row = table_row_position(&harness, "mission-briefing.md");
    click_at(&mut harness, briefing_row);
    assert_eq!(selected_file_name(&harness).as_deref(), Some("mission-briefing.md"));
    let back_button = top_bar_button(&harness, "‹");
    click_at(&mut harness, back_button);
    harness.run();
    assert!(
        selected_file_name(&harness).is_none(),
        "back must clear the selection, got {:?}",
        selected_file_name(&harness)
    );
}
