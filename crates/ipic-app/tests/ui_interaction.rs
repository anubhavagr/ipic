//! Drives the real GUI with synthetic pointer/keyboard input (egui_kittest)
//! and asserts that selecting and opening files actually works.

use egui_kittest::kittest::Queryable;
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
        let base = std::env::temp_dir().join(format!("ipic-ui-{tag}-{}", std::process::id()));
        let corpus = base.join("corpus");
        let documents = corpus.join("documents");
        std::fs::create_dir_all(&documents).unwrap();
        std::fs::create_dir_all(corpus.join("photos")).unwrap();
        std::fs::write(
            documents.join("mission-briefing.md"),
            "The orbital rendezvous plan covers launch windows and docking maneuvers.",
        )
        .unwrap();
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

/// The table row's name label (leftmost match — the details panel also shows names).
/// Position of a row's name label in the browse table (leftmost match — the
/// details panel may also render the same name).
fn table_row_position(harness: &Harness<'_, IpicApp>, text: &str) -> egui::Pos2 {
    let matches: Vec<_> = harness.query_all_by_label(text).collect();
    assert!(!matches.is_empty(), "no visible label with text {text:?}");
    matches
        .iter()
        .min_by_key(|node| node.rect().min.x as i32)
        .unwrap()
        .rect()
        .center()
}

/// Emits a click with real-input timing: move, press, release — one frame apart.
fn click_at(harness: &mut Harness<'_, IpicApp>, position: egui::Pos2, button: egui::PointerButton) {
    let modifiers = egui::Modifiers::default();
    harness.event(egui::Event::PointerMoved(position));
    harness.run_steps(2);
    harness.event(egui::Event::PointerButton { pos: position, button, pressed: true, modifiers });
    harness.run_steps(2);
    harness.event(egui::Event::PointerButton { pos: position, button, pressed: false, modifiers });
    harness.run_steps(2);
}

#[test]
fn single_click_selects_file() {
    let world = TestWorld::create("select");
    let (mut harness, engine) = build_harness(&world, Arc::new(Mutex::new(Vec::new())));
    wait_until_indexed(&engine);
    harness.run_steps(2);
    let row_position = table_row_position(&harness, "mission-briefing.md");
    click_at(&mut harness, row_position, egui::PointerButton::Primary);
    let selected = harness.state().selected_file.clone();
    let (_, path) = selected.expect("single click must select the file");
    assert!(path.ends_with("mission-briefing.md"), "selected {path}");
}

#[test]
fn double_click_opens_file() {
    let world = TestWorld::create("double");
    let opened = Arc::new(Mutex::new(Vec::new()));
    let (mut harness, engine) = build_harness(&world, Arc::clone(&opened));
    wait_until_indexed(&engine);
    harness.run_steps(2);
    // Two rapid full clicks on the same spot register as a double click.
    let row_position = table_row_position(&harness, "mission-briefing.md");
    click_at(&mut harness, row_position, egui::PointerButton::Primary);
    click_at(&mut harness, row_position, egui::PointerButton::Primary);
    let opened_paths = opened.lock().unwrap().clone();
    assert!(
        opened_paths.iter().any(|path| path.ends_with("mission-briefing.md")),
        "double click must open the file, got {opened_paths:?}"
    );
}

#[test]
fn right_click_menu_opens_file() {
    let world = TestWorld::create("menu");
    let opened = Arc::new(Mutex::new(Vec::new()));
    let (mut harness, engine) = build_harness(&world, Arc::clone(&opened));
    wait_until_indexed(&engine);
    harness.run_steps(2);
    let row_position = table_row_position(&harness, "beach-sunset.png");
    click_at(&mut harness, row_position, egui::PointerButton::Secondary);
    let menu_position = harness
        .query_all_by_label("Open")
        .find(|node| node.rect().min.y > 30.0)
        .expect("context menu with an Open entry must appear")
        .rect()
        .center();
    click_at(&mut harness, menu_position, egui::PointerButton::Primary);
    let opened_paths = opened.lock().unwrap().clone();
    assert!(
        opened_paths.iter().any(|path| path.ends_with("beach-sunset.png")),
        "context-menu Open must open the file, got {opened_paths:?}"
    );
}

#[test]
fn enter_key_opens_selected_file() {
    let world = TestWorld::create("enter");
    let opened = Arc::new(Mutex::new(Vec::new()));
    let (mut harness, engine) = build_harness(&world, Arc::clone(&opened));
    wait_until_indexed(&engine);
    harness.run_steps(2);
    let row_position = table_row_position(&harness, "mission-briefing.md");
    click_at(&mut harness, row_position, egui::PointerButton::Primary);
    harness.key_press(egui::Key::Enter);
    harness.run_steps(2);
    let opened_paths = opened.lock().unwrap().clone();
    assert!(
        opened_paths.iter().any(|path| path.ends_with("mission-briefing.md")),
        "Enter must open the selected file, got {opened_paths:?}"
    );
}
