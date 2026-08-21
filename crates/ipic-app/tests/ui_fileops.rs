//! File-operation workflows driven through the real GUI: context menu,
//! duplicate, rename dialog, move to trash, new folder, details panel.
//! All operations run against a real temp corpus on disk.

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
        let base = std::env::temp_dir().join(format!("ipic-fileops-{tag}-{}", std::process::id()));
        let corpus = base.join("corpus");
        std::fs::create_dir_all(corpus.join("documents")).unwrap();
        std::fs::create_dir_all(corpus.join("photos")).unwrap();
        std::fs::write(
            corpus.join("documents/mission-briefing.md"),
            "The orbital rendezvous plan covers launch windows and docking maneuvers.",
        )
        .unwrap();
        std::fs::write(corpus.join("todo.txt"), "pack radiation shields\n").unwrap();
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

/// Rightmost match of an exact label (widgets in the details panel).
fn details_label_position(harness: &Harness<'_, IpicApp>, text: &str) -> egui::Pos2 {
    let matches: Vec<_> = harness.query_all_by_label(text).collect();
    assert!(!matches.is_empty(), "no visible label with text {text:?}");
    matches
        .iter()
        .max_by_key(|node| node.rect().min.x as i32)
        .unwrap()
        .rect()
        .center()
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

fn click_label(
    harness: &mut Harness<'_, IpicApp>,
    text: &str,
    button: egui::PointerButton,
) {
    let position = table_row_position(harness, text);
    click_at(harness, position, button);
}

/// Position of an entry inside the currently open context menu. Menus open at
/// the cursor over the table, so the leftmost match wins — the details panel
/// (far right) renders identically named buttons when a file is selected.
fn menu_entry_position(harness: &Harness<'_, IpicApp>, text: &str) -> egui::Pos2 {
    let matches: Vec<_> = harness.query_all_by_label(text).collect();
    assert!(!matches.is_empty(), "context menu must contain {text:?}");
    matches
        .iter()
        .min_by_key(|node| ((node.rect().min.x * 10.0) as i32, node.rect().min.y as i32))
        .unwrap()
        .rect()
        .center()
}

fn click_menu_entry(harness: &mut Harness<'_, IpicApp>, text: &str) {
    let position = menu_entry_position(harness, text);
    click_at(harness, position, egui::PointerButton::Primary);
}

/// Replaces the content of the focused text field: select all, then type.
fn type_into_focused_edit(harness: &mut Harness<'_, IpicApp>, text: &str) {
    harness.key_press_modifiers(egui::Modifiers::COMMAND, egui::Key::A);
    harness.run();
    if text.is_empty() {
        harness.key_press(egui::Key::Backspace);
    } else {
        harness.event(egui::Event::Text(text.into()));
    }
    harness.run();
}

/// Runs frames until a label containing `text` appears (background rescans
/// need wall-clock time to land in the catalog).
fn wait_for_label_contains(harness: &mut Harness<'_, IpicApp>, text: &str) {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        if harness.query_all_by_label_contains(text).next().is_some() {
            return;
        }
        if Instant::now() > deadline {
            panic!("label containing {text:?} never appeared");
        }
        harness.run();
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// Runs frames until no exact label `text` remains visible.
fn wait_for_label_gone(harness: &mut Harness<'_, IpicApp>, text: &str) {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        if harness.query_all_by_label(text).next().is_none() {
            return;
        }
        if Instant::now() > deadline {
            panic!("label {text:?} never disappeared");
        }
        harness.run();
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// Opens the rename dialog for `file_name` through the real context menu.
fn open_rename_dialog(harness: &mut Harness<'_, IpicApp>, file_name: &str) {
    click_label(harness, file_name, egui::PointerButton::Secondary);
    click_menu_entry(harness, "Rename…");
    wait_for_label_contains(harness, "New name:");
}

#[test]
fn context_menu_open_reveal_copy_path_act_without_panic() {
    let world = TestWorld::create("menuops");
    let opened = Arc::new(Mutex::new(Vec::new()));
    let (mut harness, engine) = build_harness(&world, Arc::clone(&opened));
    wait_until_indexed(&engine);
    harness.run();

    // All file-operation entries exist on the menu.
    click_label(&mut harness, "todo.txt", egui::PointerButton::Secondary);
    for entry in ["Open", "Reveal in Finder", "Copy path", "Duplicate", "Rename…", "Move to Trash"] {
        menu_entry_position(&harness, entry);
    }
    click_menu_entry(&mut harness, "Copy path");
    wait_for_label_contains(&mut harness, "path copied");

    // Reveal spawns the OS file manager — must not take the UI down.
    click_label(&mut harness, "todo.txt", egui::PointerButton::Secondary);
    click_menu_entry(&mut harness, "Reveal in Finder");
    harness.run();

    // Open dispatches through the injectable open handler.
    click_label(&mut harness, "todo.txt", egui::PointerButton::Secondary);
    click_menu_entry(&mut harness, "Open");
    let opened_paths = opened.lock().unwrap().clone();
    assert!(
        opened_paths.iter().any(|path| path.ends_with("todo.txt")),
        "context-menu Open must open the file, got {opened_paths:?}"
    );
}

#[test]
fn duplicate_creates_real_copy_on_disk_and_refreshes_listing() {
    let world = TestWorld::create("dup");
    let (mut harness, engine) = build_harness(&world, Arc::new(Mutex::new(Vec::new())));
    wait_until_indexed(&engine);
    harness.run();

    click_label(&mut harness, "beach-sunset.png", egui::PointerButton::Secondary);
    click_menu_entry(&mut harness, "Duplicate");

    let original = world.corpus.join("photos/beach-sunset.png");
    let copy = world.corpus.join("photos/beach-sunset copy.png");
    assert!(copy.exists(), "duplicate must exist on disk");
    assert!(original.exists(), "original must survive duplication");
    assert_eq!(
        std::fs::read(&original).unwrap(),
        std::fs::read(&copy).unwrap(),
        "duplicate must have identical content"
    );
    wait_for_label_contains(&mut harness, "beach-sunset copy");
}

#[test]
fn rename_dialog_enter_renames_file_without_leaking_open() {
    let world = TestWorld::create("rename-enter");
    let opened = Arc::new(Mutex::new(Vec::new()));
    let (mut harness, engine) = build_harness(&world, Arc::clone(&opened));
    wait_until_indexed(&engine);
    harness.run();

    // Select a different file first: Enter inside the rename dialog must act
    // on the dialog's target, not open the current selection.
    click_label(&mut harness, "todo.txt", egui::PointerButton::Primary);
    open_rename_dialog(&mut harness, "mission-briefing.md");
    type_into_focused_edit(&mut harness, "flight-plan.md");
    harness.key_press(egui::Key::Enter);
    harness.run();

    let old_path = world.corpus.join("documents/mission-briefing.md");
    let new_path = world.corpus.join("documents/flight-plan.md");
    assert!(!old_path.exists(), "old name must be gone from disk");
    assert!(new_path.exists(), "renamed file must exist on disk");
    assert!(std::fs::read_to_string(&new_path).unwrap().contains("orbital rendezvous"));
    wait_for_label_contains(&mut harness, "flight-plan.md");
    wait_for_label_gone(&mut harness, "mission-briefing.md");

    let opened_paths = opened.lock().unwrap().clone();
    assert!(
        opened_paths.is_empty(),
        "Enter in the rename dialog must not open the selected file, got {opened_paths:?}"
    );
}

#[test]
fn rename_dialog_rename_button_renames_file() {
    let world = TestWorld::create("rename-button");
    let (mut harness, engine) = build_harness(&world, Arc::new(Mutex::new(Vec::new())));
    wait_until_indexed(&engine);
    harness.run();

    open_rename_dialog(&mut harness, "mission-briefing.md");
    type_into_focused_edit(&mut harness, "docking-notes.md");
    let rename_button = harness
        .query_all_by_label("Rename")
        .max_by_key(|node| node.rect().min.y as i32)
        .expect("Rename dialog must have a Rename button")
        .rect()
        .center();
    click_at(&mut harness, rename_button, egui::PointerButton::Primary);

    assert!(!world.corpus.join("documents/mission-briefing.md").exists());
    assert!(world.corpus.join("documents/docking-notes.md").exists());
    wait_for_label_contains(&mut harness, "docking-notes.md");
}

#[test]
fn rename_dialog_cancel_and_escape_leave_file_untouched() {
    let world = TestWorld::create("rename-cancel");
    let (mut harness, engine) = build_harness(&world, Arc::new(Mutex::new(Vec::new())));
    wait_until_indexed(&engine);
    harness.run();
    let original_path = world.corpus.join("documents/mission-briefing.md");
    let original_bytes = std::fs::read(&original_path).unwrap();

    // Cancel button.
    open_rename_dialog(&mut harness, "mission-briefing.md");
    type_into_focused_edit(&mut harness, "should-not-exist.md");
    let cancel_button = menu_entry_position(&harness, "Cancel");
    click_at(&mut harness, cancel_button, egui::PointerButton::Primary);
    wait_for_label_gone(&mut harness, "New name:");
    assert!(original_path.exists(), "Cancel must not rename the file");

    // Escape key.
    open_rename_dialog(&mut harness, "mission-briefing.md");
    type_into_focused_edit(&mut harness, "also-not.md");
    harness.key_press(egui::Key::Escape);
    harness.run();
    wait_for_label_gone(&mut harness, "New name:");
    assert!(original_path.exists(), "Escape must not rename the file");
    assert!(!world.corpus.join("documents/also-not.md").exists());
    assert_eq!(std::fs::read(&original_path).unwrap(), original_bytes);
}

#[test]
fn rename_dialog_rejects_invalid_names_without_damage() {
    let world = TestWorld::create("rename-invalid");
    let (mut harness, engine) = build_harness(&world, Arc::new(Mutex::new(Vec::new())));
    wait_until_indexed(&engine);
    harness.run();
    let original_path = world.corpus.join("documents/mission-briefing.md");

    // A name containing a path separator is rejected, dialog stays available.
    open_rename_dialog(&mut harness, "mission-briefing.md");
    type_into_focused_edit(&mut harness, "evil/name.md");
    harness.key_press(egui::Key::Enter);
    harness.run();
    assert!(original_path.exists(), "file must be untouched by invalid rename");
    assert!(
        harness.query_all_by_label_contains("rename failed").next().is_some(),
        "invalid rename must surface a failure notice"
    );
    assert!(
        harness.query_all_by_label_contains("New name:").next().is_some(),
        "dialog must stay open after a rejected name"
    );

    // Empty name is also rejected.
    type_into_focused_edit(&mut harness, "");
    harness.key_press(egui::Key::Enter);
    harness.run();
    assert!(original_path.exists(), "file must be untouched by empty rename");
    assert!(harness.query_all_by_label_contains("New name:").next().is_some());

    // Escaping afterwards still closes the dialog cleanly.
    harness.key_press(egui::Key::Escape);
    harness.run();
    wait_for_label_gone(&mut harness, "New name:");
    assert!(original_path.exists());
}

#[test]
fn move_to_trash_removes_file_from_disk_and_listing() {
    let world = TestWorld::create("trash");
    let (mut harness, engine) = build_harness(&world, Arc::new(Mutex::new(Vec::new())));
    wait_until_indexed(&engine);
    harness.run();
    let target = world.corpus.join("todo.txt");
    assert!(target.exists());

    click_label(&mut harness, "todo.txt", egui::PointerButton::Secondary);
    click_menu_entry(&mut harness, "Move to Trash");

    assert!(!target.exists(), "file must be removed from disk (moved to trash)");
    wait_for_label_gone(&mut harness, "todo.txt");
    assert!(
        world.corpus.join("documents/mission-briefing.md").exists(),
        "other files must be untouched"
    );
}

/// Navigates into the corpus root through the sidebar locations row.
fn enter_corpus_root(harness: &mut Harness<'_, IpicApp>, corpus: &std::path::Path) {
    // The app canonicalizes roots (e.g. /var → /private/var on macOS), so the
    // sidebar shows the canonical form.
    let canonical = std::fs::canonicalize(corpus).unwrap_or_else(|_| corpus.to_path_buf());
    let root_row = harness
        .query_all_by_label_contains(&format!("Volume  {}", canonical.display()))
        .next()
        .expect("sidebar must list the corpus root")
        .rect()
        .center();
    click_at(harness, root_row, egui::PointerButton::Primary);
    harness.run();
}

#[test]
fn new_folder_enter_creates_directory_on_disk() {
    let world = TestWorld::create("folder-create");
    let (mut harness, engine) = build_harness(&world, Arc::new(Mutex::new(Vec::new())));
    wait_until_indexed(&engine);
    harness.run();
    enter_corpus_root(&mut harness, &world.corpus);

    let new_folder_button = table_row_position(&harness, "＋ New Folder");
    click_at(&mut harness, new_folder_button, egui::PointerButton::Primary);
    wait_for_label_contains(&mut harness, "Create");
    type_into_focused_edit(&mut harness, "Field Notes");
    harness.key_press(egui::Key::Enter);
    harness.run();

    let created = world.corpus.join("Field Notes");
    assert!(created.is_dir(), "folder must be created on disk");
    wait_for_label_contains(&mut harness, "Field Notes");
    wait_for_label_contains(&mut harness, "created “Field Notes”");
    assert!(
        harness.query_all_by_label("＋ New Folder").next().is_some(),
        "toolbar must return to the collapsed New Folder button"
    );
}

#[test]
fn new_folder_create_button_creates_directory() {
    let world = TestWorld::create("folder-button");
    let (mut harness, engine) = build_harness(&world, Arc::new(Mutex::new(Vec::new())));
    wait_until_indexed(&engine);
    harness.run();
    enter_corpus_root(&mut harness, &world.corpus);

    let new_folder_button = table_row_position(&harness, "＋ New Folder");
    click_at(&mut harness, new_folder_button, egui::PointerButton::Primary);
    wait_for_label_contains(&mut harness, "Create");
    type_into_focused_edit(&mut harness, "Sketches");
    let create_button = menu_entry_position(&harness, "Create");
    click_at(&mut harness, create_button, egui::PointerButton::Primary);
    harness.run();

    assert!(world.corpus.join("Sketches").is_dir(), "Create button must make the folder");
    wait_for_label_contains(&mut harness, "Sketches");
}

#[test]
fn new_folder_escape_and_close_button_cancel_without_creating() {
    let world = TestWorld::create("folder-cancel");
    let (mut harness, engine) = build_harness(&world, Arc::new(Mutex::new(Vec::new())));
    wait_until_indexed(&engine);
    harness.run();
    enter_corpus_root(&mut harness, &world.corpus);

    // Escape collapses the inline editor.
    let new_folder_button = table_row_position(&harness, "＋ New Folder");
    click_at(&mut harness, new_folder_button, egui::PointerButton::Primary);
    wait_for_label_contains(&mut harness, "Create");
    harness.key_press(egui::Key::Escape);
    harness.run();
    wait_for_label_gone(&mut harness, "Create");
    assert!(harness.query_all_by_label("＋ New Folder").next().is_some());

    // The ✕ button collapses it too.
    let new_folder_button = table_row_position(&harness, "＋ New Folder");
    click_at(&mut harness, new_folder_button, egui::PointerButton::Primary);
    wait_for_label_contains(&mut harness, "Create");
    harness
        .query_all_by_label("✕")
        .next()
        .expect("close button visible")
        .click_accesskit();
    harness.run();
    wait_for_label_gone(&mut harness, "Create");
    assert!(harness.query_all_by_label("＋ New Folder").next().is_some());

    assert!(
        !world.corpus.join("New Folder").exists(),
        "cancelled folder must not be created on disk"
    );
}

#[test]
fn new_folder_in_library_view_shows_notice_instead_of_panicking() {
    let world = TestWorld::create("folder-library");
    let (mut harness, engine) = build_harness(&world, Arc::new(Mutex::new(Vec::new())));
    wait_until_indexed(&engine);
    harness.run();
    // Library view (no current folder) is the default after launch.

    let new_folder_button = table_row_position(&harness, "＋ New Folder");
    click_at(&mut harness, new_folder_button, egui::PointerButton::Primary);
    wait_for_label_contains(&mut harness, "Create");
    type_into_focused_edit(&mut harness, "Nowhere");
    harness.key_press(egui::Key::Enter);
    harness.run();

    wait_for_label_contains(&mut harness, "open a folder first");
    assert!(
        harness.query_all_by_label("＋ New Folder").next().is_some(),
        "toolbar must return to the collapsed New Folder button"
    );
    assert!(
        !world.corpus.join("Nowhere").exists(),
        "no folder may be created while in library view"
    );
}

#[test]
fn details_panel_open_dispatches_through_open_handler_and_shows_text_preview() {
    let world = TestWorld::create("details");
    let opened = Arc::new(Mutex::new(Vec::new()));
    let (mut harness, engine) = build_harness(&world, Arc::clone(&opened));
    wait_until_indexed(&engine);
    harness.run();

    click_label(&mut harness, "mission-briefing.md", egui::PointerButton::Primary);
    // Wait for the extractor to land the first chunk in the catalog.
    wait_for_label_contains(&mut harness, "INDEXED CONTENT");
    assert!(
        harness.query_all_by_label_contains("orbital rendezvous").next().is_some(),
        "details panel must render the indexed text preview"
    );

    let open_button = details_label_position(&harness, "Open");
    click_at(&mut harness, open_button, egui::PointerButton::Primary);
    let opened_paths = opened.lock().unwrap().clone();
    assert!(
        opened_paths.iter().any(|path| path.ends_with("mission-briefing.md")),
        "details-panel Open must dispatch through the open handler, got {opened_paths:?}"
    );
}
