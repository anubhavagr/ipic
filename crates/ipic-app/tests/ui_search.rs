//! Drives the unified search bar through real keyboard/pointer input
//! (egui_kittest): typing filters the browse listing, Enter and the Search
//! button run the multimodal search, Esc returns to browsing.

use egui::accesskit::Role;
use egui_kittest::kittest::{by, Queryable};
use egui_kittest::Harness;
use ipic::{theme, IpicApp};
use ipic_core::Config;
use ipic_rag::{Engine, EngineStatus};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

struct TestWorld {
    corpus: PathBuf,
    data_directory: PathBuf,
}

impl TestWorld {
    fn create(tag: &str) -> Self {
        let base = std::env::temp_dir().join(format!("ipic-search-{tag}-{}", std::process::id()));
        let corpus = base.join("corpus");
        let documents = corpus.join("documents");
        std::fs::create_dir_all(&documents).unwrap();
        std::fs::create_dir_all(corpus.join("photos")).unwrap();
        std::fs::write(
            documents.join("mission-briefing.md"),
            "The orbital rendezvous plan covers launch windows and docking maneuvers.",
        )
        .unwrap();
        std::fs::write(documents.join("grocery-run.txt"), "oat milk and cardamom").unwrap();
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

fn build_harness(
    world: &TestWorld,
    opened: Arc<Mutex<Vec<String>>>,
) -> (Harness<'static, IpicApp>, Arc<Engine>) {
    let config = world.config();
    let engine = Engine::launch_with_data_dir(config.clone(), world.data_directory.clone()).unwrap();
    let mut application = IpicApp::new(Arc::clone(&engine), config);
    let capture = Arc::clone(&opened);
    application.open_handler = Some(Arc::new(move |path| {
        capture.lock().unwrap().push(path.display().to_string());
    }));
    // A small simulated clock keeps click sequences inside egui's 0.3 s
    // double-click window (kittest's default step is a quarter second).
    let harness = egui_kittest::HarnessBuilder::default()
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
fn click_at(harness: &mut Harness<'_, IpicApp>, position: egui::Pos2, button: egui::PointerButton) {
    let modifiers = egui::Modifiers::default();
    harness.event(egui::Event::PointerMoved(position));
    harness.run();
    harness.event(egui::Event::PointerButton { pos: position, button, pressed: true, modifiers });
    harness.run();
    harness.event(egui::Event::PointerButton { pos: position, button, pressed: false, modifiers });
    harness.run();
}

/// Gives the top-bar search TextEdit keyboard focus (AccessKit focus request).
/// Library view shows root folders; file tests enter the corpus root first.
fn enter_corpus_root(harness: &mut Harness<'_, IpicApp>, corpus: &std::path::Path) {
    let canonical = std::fs::canonicalize(corpus).unwrap_or_else(|_| corpus.to_path_buf());
    let root_row = harness
        .query_all_by_label_contains(&format!("Volume  {}", canonical.display()))
        .next()
        .expect("sidebar must list the corpus root")
        .rect()
        .center();
    click_at(harness, root_row, egui::PointerButton::Primary);
    harness.run_steps(3);
}

/// Enter the documents subfolder (double-click its table row).
fn enter_documents(harness: &mut Harness<'_, IpicApp>) {
    let position = harness
        .query_all_by_label("documents")
        .filter(|node| node.rect().min.x > 240.0)
        .min_by_key(|node| node.rect().min.x as i32)
        .expect("documents folder visible")
        .rect()
        .center();
    harness.run_steps(700);
    click_at(harness, position, egui::PointerButton::Primary);
    click_at(harness, position, egui::PointerButton::Primary);
    harness.run_steps(700);
}

fn focus_search_box(harness: &mut Harness<'_, IpicApp>) {
    harness.get(by().role(Role::TextInput)).focus();
    harness.run();
    // Focus takes effect at the end of the pass; the app records it next frame.
    harness.run();
    assert!(harness.state().search_box_has_focus, "search box must hold keyboard focus");
}

/// Types text into the focused search box, then lets the listing refresh settle.
fn type_query(harness: &mut Harness<'_, IpicApp>, text: &str) {
    harness.get(by().role(Role::TextInput)).type_text(text);
    harness.run();
    harness.run();
    assert_eq!(harness.state().search_edit, text, "typed text must land in the search box");
}

/// Steps the GUI until the running search finishes (worker sends the outcome).
fn wait_for_search_to_finish(harness: &mut Harness<'_, IpicApp>) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while Instant::now() < deadline {
        harness.run();
        if !harness.state().searching {
            return;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    panic!("search did not finish in time");
}

fn search_button_position(harness: &Harness<'_, IpicApp>) -> egui::Pos2 {
    harness.get_by_label("\u{e8b6}").rect().center()
}

#[test]
fn enter_key_runs_search() {
    let world = TestWorld::create("enter-runs");
    let (mut harness, engine) = build_harness(&world, Arc::new(Mutex::new(Vec::new())));
    wait_until_indexed(&engine);
    harness.run();
    focus_search_box(&mut harness);
    type_query(&mut harness, "rendezvous");
    harness.key_press(egui::Key::Enter);
    harness.run();
    assert_eq!(harness.state().active_query, "rendezvous", "Enter must activate the query");
    // The hashing embedder can finish before `searching` is observable; hits are
    // the proof that a search actually ran.
    wait_for_search_to_finish(&mut harness);
    let hit_paths: Vec<&str> =
        harness.state().results.hits.iter().map(|hit| hit.path.as_str()).collect();
    assert!(
        hit_paths.iter().any(|path| path.ends_with("mission-briefing.md")),
        "search must surface the matching document, got {hit_paths:?}"
    );
    assert!(
        harness.query_all_by_label("mission-briefing.md").next().is_some(),
        "results view must render the hit"
    );
}

#[test]
fn search_button_click_runs_search() {
    let world = TestWorld::create("button-runs");
    let (mut harness, engine) = build_harness(&world, Arc::new(Mutex::new(Vec::new())));
    wait_until_indexed(&engine);
    harness.run();
    focus_search_box(&mut harness);
    type_query(&mut harness, "docking");
    let button_position = search_button_position(&harness);
    click_at(&mut harness, button_position, egui::PointerButton::Primary);
    assert_eq!(harness.state().active_query, "docking", "Search button must activate the query");
    // The hashing embedder can finish before `searching` is observable; hits are
    // the proof that a search actually ran.
    wait_for_search_to_finish(&mut harness);
    assert!(
        harness
            .state()
            .results
            .hits
            .iter()
            .any(|hit| hit.path.ends_with("mission-briefing.md")),
        "button search must surface the matching document"
    );
}

#[test]
fn typing_filters_browse_listing_live() {
    let world = TestWorld::create("live-filter");
    let (mut harness, engine) = build_harness(&world, Arc::new(Mutex::new(Vec::new())));
    wait_until_indexed(&engine);
    harness.run_steps(2);
    enter_corpus_root(&mut harness, &world.corpus);
    enter_documents(&mut harness);
    let unfiltered_count = harness.state().browse.listing.len();
    assert!(unfiltered_count >= 2, "corpus must show both files, got {unfiltered_count}");
    focus_search_box(&mut harness);
    type_query(&mut harness, "mission");
    let state = harness.state();
    assert_eq!(state.active_query, "", "typing alone must not enter search mode");
    assert_eq!(state.browse.name_filter_active, "mission");
    assert!(
        state.browse.listing.iter().all(|entry| match entry {
            ipic::browse::ListingEntry::File(file) => file.name.contains("mission"),
            ipic::browse::ListingEntry::Directory(_) => false,
        }) && !state.browse.listing.is_empty(),
        "listing must shrink to name matches, got {:?}",
        state.browse.listing.len()
    );
}

#[test]
fn escape_clears_search_back_to_browse() {
    let world = TestWorld::create("escape-clears");
    let (mut harness, engine) = build_harness(&world, Arc::new(Mutex::new(Vec::new())));
    wait_until_indexed(&engine);
    harness.run_steps(2);
    enter_corpus_root(&mut harness, &world.corpus);
    enter_documents(&mut harness);
    focus_search_box(&mut harness);
    type_query(&mut harness, "rendezvous");
    harness.key_press(egui::Key::Enter);
    wait_for_search_to_finish(&mut harness);
    assert!(!harness.state().active_query.is_empty());
    harness.key_press(egui::Key::Escape);
    harness.run();
    harness.run();
    let state = harness.state();
    assert!(state.active_query.is_empty(), "Esc must clear the active query");
    assert!(state.search_edit.is_empty(), "Esc must clear the search box");
    assert!(state.results.hits.is_empty(), "Esc must drop the results");
    assert!(
        state.browse.listing.iter().any(|entry| matches!(entry,
            ipic::browse::ListingEntry::File(file) if file.name == "mission-briefing.md")),
        "Esc must restore the full browse listing"
    );
    assert!(
        harness.query_all_by_label("mission-briefing.md").next().is_some(),
        "browse view must be visible again"
    );
}

#[test]
fn enter_with_empty_query_starts_no_search() {
    let world = TestWorld::create("empty-query");
    let (mut harness, engine) = build_harness(&world, Arc::new(Mutex::new(Vec::new())));
    wait_until_indexed(&engine);
    harness.run();
    focus_search_box(&mut harness);
    harness.key_press(egui::Key::Enter);
    harness.run();
    let state = harness.state();
    assert!(state.active_query.is_empty(), "empty query must not activate a search");
    assert!(!state.searching, "empty query must not start a search");
    assert!(state.results.hits.is_empty());
}

#[test]
fn no_results_state_renders_without_panic() {
    let world = TestWorld::create("no-results");
    let (mut harness, engine) = build_harness(&world, Arc::new(Mutex::new(Vec::new())));
    wait_until_indexed(&engine);
    harness.run();
    focus_search_box(&mut harness);
    type_query(&mut harness, "zzzznothingshares");
    harness.key_press(egui::Key::Enter);
    wait_for_search_to_finish(&mut harness);
    let state = harness.state();
    assert_eq!(state.active_query, "zzzznothingshares");
    assert!(state.results.hits.is_empty());
    assert!(
        harness.query_all_by_label("no matches").next().is_some(),
        "no-results state must render its hint"
    );
}

#[test]
fn command_f_focuses_search_box() {
    let world = TestWorld::create("command-f");
    let (mut harness, engine) = build_harness(&world, Arc::new(Mutex::new(Vec::new())));
    wait_until_indexed(&engine);
    harness.run();
    assert!(!harness.state().search_box_has_focus, "search box must start unfocused");
    harness.key_combination_modifiers(egui::Modifiers::COMMAND, &[egui::Key::F]);
    harness.run();
    // The focus request is consumed while drawing; the next frame records it.
    harness.run();
    assert!(harness.state().search_box_has_focus, "⌘F must focus the search box");
}

#[test]
fn escape_returns_to_directory_before_search() {
    let world = TestWorld::create("esc-return");
    let (mut harness, engine) = build_harness(&world, Arc::new(Mutex::new(Vec::new())));
    wait_until_indexed(&engine);
    harness.run_steps(2);
    // Library view shows the corpus root; enter it, then its documents folder.
    let corpus_label = harness
        .query_all_by_label("corpus")
        .next()
        .expect("corpus root visible in library view")
        .rect()
        .center();
    for _ in 0..2 {
        click_at(&mut harness, corpus_label, egui::PointerButton::Primary);
    }
    // Let the double-click window close before the next pair.
    harness.run_steps(700);
    let documents_label = harness
        .query_all_by_label("documents")
        .next()
        .expect("documents folder visible inside corpus")
        .rect()
        .center();
    for _ in 0..2 {
        click_at(&mut harness, documents_label, egui::PointerButton::Primary);
    }
    harness.run_steps(700);
    let before = harness.state().current_directory.clone();
    assert!(before.as_ref().is_some_and(|dir| dir.path.ends_with("documents")), "entered documents");
    // Run a search from inside it.
    focus_search_box(&mut harness);
    type_query(&mut harness, "orbital rendezvous");
    harness.key_press(egui::Key::Enter);
    wait_for_search_to_finish(&mut harness);
    assert!(!harness.state().active_query.is_empty());
    // Esc must restore the documents folder, not the library view.
    harness.key_press(egui::Key::Escape);
    harness.run_steps(3);
    let restored = harness.state().current_directory.clone();
    assert!(
        restored.as_ref().is_some_and(|dir| dir.path.ends_with("documents")),
        "Esc must return to the pre-search directory, got {restored:?}"
    );
    assert!(harness.state().active_query.is_empty());
}
