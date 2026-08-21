//! Toolbar, filtering, sorting, settings and status workflows driven through
//! the real GUI with synthetic pointer/keyboard input (egui_kittest).

use egui_kittest::kittest::{by, Queryable};
use egui_kittest::Harness;
use ipic::{theme, IpicApp};
use ipic_core::{Config, FileFilter, SortKey};
use ipic_rag::{Engine, EngineStatus};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Whole-library listing order under the default sort (Name, ascending).
const NAME_ASC: [&str; 6] = [
    "alpha-notes.md",
    "big-blob.bin",
    "bravo-symphony.mp3",
    "charlie-archive.zip",
    "delta-report.pdf",
    "echo-snapshot.png",
];

struct TestWorld {
    corpus: PathBuf,
    data_directory: PathBuf,
}

fn write_file(path: &Path, contents: &str, age_secs: u64) {
    std::fs::write(path, contents).unwrap();
    set_mtime(path, age_secs);
}

/// Creates a large file instantly by extending it (sparse on APFS).
fn write_sparse(path: &Path, len: u64, age_secs: u64) {
    let file = std::fs::File::create(path).unwrap();
    file.set_len(len).unwrap();
    drop(file);
    set_mtime(path, age_secs);
}

fn set_mtime(path: &Path, age_secs: u64) {
    let modified = std::time::SystemTime::now() - Duration::from_secs(age_secs);
    std::fs::File::options()
        .write(true)
        .open(path)
        .unwrap()
        .set_times(std::fs::FileTimes::new().set_modified(modified))
        .unwrap();
}

impl TestWorld {
    fn create(tag: &str) -> Self {
        let base = std::env::temp_dir().join(format!("ipic-ui-toolbar-{tag}-{}", std::process::id()));
        let corpus = base.join("corpus");
        std::fs::create_dir_all(&corpus).unwrap();
        // Six files with distinct kinds, sizes and ages so every sort key and
        // filter has a deterministic expected order.
        write_file(&corpus.join("alpha-notes.md"), &"a".repeat(120), 24 * 3600); // Text
        write_sparse(&corpus.join("big-blob.bin"), 101 * 1024 * 1024, 3 * 86400); // Other, >100 MB
        write_file(&corpus.join("bravo-symphony.mp3"), &"b".repeat(500), 6 * 86400); // Audio
        write_file(&corpus.join("charlie-archive.zip"), &"c".repeat(90), 100 * 86400); // Other, >30 days old
        write_file(&corpus.join("delta-report.pdf"), &"d".repeat(700), 2 * 3600); // Pdf
        write_file(&corpus.join("echo-snapshot.png"), &"e".repeat(300), 9 * 86400); // Image
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

/// Library view shows root folders; toolbar tests operate inside the corpus.
fn enter_corpus_root(harness: &mut Harness<'_, IpicApp>) {
    // The sidebar location row navigates reliably regardless of layout.
    let position = harness
        .query_all_by_label_contains("Volume")
        .next()
        .expect("sidebar must list the corpus root")
        .rect()
        .center();
    click_at(harness, position, egui::PointerButton::Primary);
    harness.run_steps(3);
}

fn build_harness(world: &TestWorld) -> (Harness<'static, IpicApp>, Arc<Engine>) {
    let config = world.config();
    let engine = Engine::launch_with_data_dir(config.clone(), world.data_directory.clone()).unwrap();
    let application = IpicApp::new(Arc::clone(&engine), config);
    // A small simulated clock keeps click sequences inside egui's 0.3 s
    // double-click window (kittest's default step is a quarter second).
    // The wide window keeps the toolbar and all table columns visible.
    let harness = egui_kittest::HarnessBuilder::default()
        .with_step_dt(0.001)
        .with_size(egui::vec2(1400.0, 900.0))
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
        if !status.scanning && status.pending == 0 && status.total_files >= 6 {
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
    harness.run_steps(2);
    harness.event(egui::Event::PointerButton { pos: position, button, pressed: true, modifiers });
    harness.run_steps(2);
    harness.event(egui::Event::PointerButton { pos: position, button, pressed: false, modifiers });
    harness.run_steps(2);
}

/// Clicks and lets a further frame run so `logic()` can apply the pending
/// listing refresh triggered by the click.
fn click_and_settle(harness: &mut Harness<'_, IpicApp>, position: egui::Pos2) {
    click_at(harness, position, egui::PointerButton::Primary);
    harness.run_steps(2);
}

/// Clicks the topmost exact-label match (see [`topmost_label_position`]).
fn click_topmost_label(harness: &mut Harness<'_, IpicApp>, text: &str) {
    let position = topmost_label_position(harness, text);
    click_and_settle(harness, position);
}

/// Clicks the bottom-most exact-label match (see [`bottom_label_position`]).
fn click_bottom_label(harness: &mut Harness<'_, IpicApp>, text: &str) {
    let position = bottom_label_position(harness, text);
    click_and_settle(harness, position);
}

/// Toolbar widgets sit above the table, whose cells reuse the same texts
/// (kind labels); the topmost exact-label match is the toolbar one.
fn topmost_label_position(harness: &Harness<'_, IpicApp>, text: &str) -> egui::Pos2 {
    let matches: Vec<_> = harness.query_all_by_label(text).collect();
    assert!(!matches.is_empty(), "no visible label with text {text:?}");
    matches
        .iter()
        .min_by_key(|node| node.rect().min.y as i32)
        .unwrap()
        .rect()
        .center()
}

/// Bottom-most exact-label match (e.g. the toolbar direction button, not the
/// identical top-bar parent-directory button).
fn bottom_label_position(harness: &Harness<'_, IpicApp>, text: &str) -> egui::Pos2 {
    let matches: Vec<_> = harness.query_all_by_label(text).collect();
    assert!(!matches.is_empty(), "no visible label with text {text:?}");
    matches
        .iter()
        .max_by_key(|node| node.rect().min.y as i32)
        .unwrap()
        .rect()
        .center()
}

fn listing_names(harness: &Harness<'_, IpicApp>) -> Vec<String> {
    harness
        .state()
        .browse
        .listing
        .iter()
        .map(|entry| match entry {
            ipic::browse::ListingEntry::File(file) => file.name.clone(),
            ipic::browse::ListingEntry::Directory(directory) => directory.name.clone(),
        })
        .collect()
}

/// Opens the sort ComboBox (identified by its current "Sort: …" value) and
/// clicks the popup entry. Popup contents are submitted after the main panels,
/// so the last exact-label match is the popup item.
fn click_sort_combo_item(harness: &mut Harness<'_, IpicApp>, combo_value: &str, item: &str) {
    let combo_center = harness.get(by().value(combo_value)).rect().center();
    let combo_bottom = harness.get(by().value(combo_value)).rect().max.y;
    click_at(harness, combo_center, egui::PointerButton::Primary);
    harness.run_steps(2);
    let matches: Vec<_> = harness
        .query_all_by_label(item)
        .filter(|node| node.rect().min.y > combo_bottom)
        .collect();
    let item_position = matches
        .last()
        .unwrap_or_else(|| panic!("sort popup item {item:?} not visible below the combo box"))
        .rect()
        .center();
    click_and_settle(harness, item_position);
}

fn notices_contain(harness: &Harness<'_, IpicApp>, prefix: &str) -> bool {
    harness.state().notices.iter().any(|(message, _)| message.starts_with(prefix))
}

/// Waits (running frames) until a notice starting with `prefix` appears.
fn wait_for_notice(harness: &mut Harness<'_, IpicApp>, prefix: &str) {
    let deadline = Instant::now() + Duration::from_secs(15);
    while Instant::now() < deadline {
        harness.run_steps(2);
        if notices_contain(harness, prefix) {
            return;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    panic!("notice starting with {prefix:?} never appeared: {:?}", harness.state().notices);
}

#[test]
fn kind_chip_filters_listing_and_clear_resets() {
    let world = TestWorld::create("chip");
    let (mut harness, engine) = build_harness(&world);
    wait_until_indexed(&engine);
    harness.run_steps(2);
    enter_corpus_root(&mut harness);
    assert_eq!(listing_names(&harness), NAME_ASC);

    click_topmost_label(&mut harness, "Audio");
    assert_eq!(listing_names(&harness), ["bravo-symphony.mp3"]);

    click_topmost_label(&mut harness, "clear");
    assert_eq!(listing_names(&harness), NAME_ASC);
}

#[test]
fn kind_chips_combine() {
    let world = TestWorld::create("combine");
    let (mut harness, engine) = build_harness(&world);
    wait_until_indexed(&engine);
    harness.run_steps(2);
    enter_corpus_root(&mut harness);

    click_topmost_label(&mut harness, "Text");
    click_topmost_label(&mut harness, "Other");
    assert_eq!(listing_names(&harness), ["alpha-notes.md", "big-blob.bin", "charlie-archive.zip"]);
}

#[test]
fn name_header_click_toggles_direction() {
    let world = TestWorld::create("header");
    let (mut harness, engine) = build_harness(&world);
    wait_until_indexed(&engine);
    harness.run_steps(2);
    enter_corpus_root(&mut harness);
    assert_eq!(listing_names(&harness), NAME_ASC);

    // The active Name header carries the direction arrow.
    click_topmost_label(&mut harness, "Name ↑");
    assert!(!harness.state().browse.sort_ascending, "header click must flip the direction");
    let mut expected = NAME_ASC;
    expected.reverse();
    assert_eq!(listing_names(&harness), expected);

    click_topmost_label(&mut harness, "Name ↓");
    assert!(harness.state().browse.sort_ascending);
    assert_eq!(listing_names(&harness), NAME_ASC);
}

#[test]
fn sort_combo_reorders_for_each_key() {
    let world = TestWorld::create("combo");
    let (mut harness, engine) = build_harness(&world);
    wait_until_indexed(&engine);
    harness.run_steps(2);
    enter_corpus_root(&mut harness);

    // Kind (SQL orders by the kind token: audio < image < other < pdf < text).
    click_sort_combo_item(&mut harness, "Sort: Name", "Kind");
    assert_eq!(harness.state().browse.sort_key, SortKey::Kind);
    assert_eq!(
        listing_names(&harness),
        ["bravo-symphony.mp3", "echo-snapshot.png", "big-blob.bin", "charlie-archive.zip", "delta-report.pdf", "alpha-notes.md"]
    );

    // Size ascending by the distinct byte counts.
    click_sort_combo_item(&mut harness, "Sort: Kind", "Size");
    assert_eq!(harness.state().browse.sort_key, SortKey::Size);
    assert_eq!(
        listing_names(&harness),
        ["charlie-archive.zip", "alpha-notes.md", "echo-snapshot.png", "bravo-symphony.mp3", "delta-report.pdf", "big-blob.bin"]
    );

    // Modified ascending: oldest first (charlie is 100 days old).
    click_sort_combo_item(&mut harness, "Sort: Size", "Modified");
    assert_eq!(harness.state().browse.sort_key, SortKey::Modified);
    assert_eq!(
        listing_names(&harness),
        ["charlie-archive.zip", "echo-snapshot.png", "bravo-symphony.mp3", "big-blob.bin", "alpha-notes.md", "delta-report.pdf"]
    );

    // Length: inject a duration for the audio file; SQL sorts NULLs first.
    let connection = engine.catalog.reader().unwrap();
    let bravo_id = engine
        .catalog
        .children(&connection, None, &FileFilter::default(), SortKey::Name, true, 100)
        .unwrap()
        .into_iter()
        .find(|file| file.name == "bravo-symphony.mp3")
        .unwrap()
        .id;
    engine.catalog.set_duration(bravo_id, 4800.0).unwrap();
    click_sort_combo_item(&mut harness, "Sort: Modified", "Length");
    assert_eq!(harness.state().browse.sort_key, SortKey::Duration);
    assert_eq!(
        listing_names(&harness),
        ["alpha-notes.md", "big-blob.bin", "charlie-archive.zip", "delta-report.pdf", "echo-snapshot.png", "bravo-symphony.mp3"]
    );

    // Back to Name via the combo box.
    click_sort_combo_item(&mut harness, "Sort: Length", "Name");
    assert_eq!(harness.state().browse.sort_key, SortKey::Name);
    assert_eq!(listing_names(&harness), NAME_ASC);
}

#[test]
fn direction_button_flips_order() {
    let world = TestWorld::create("direction");
    let (mut harness, engine) = build_harness(&world);
    wait_until_indexed(&engine);
    harness.run_steps(2);
    enter_corpus_root(&mut harness);

    click_sort_combo_item(&mut harness, "Sort: Name", "Size");
    assert_eq!(
        listing_names(&harness),
        ["charlie-archive.zip", "alpha-notes.md", "echo-snapshot.png", "bravo-symphony.mp3", "delta-report.pdf", "big-blob.bin"]
    );

    click_bottom_label(&mut harness, "↑");
    assert!(!harness.state().browse.sort_ascending, "direction button must flip the order");
    assert_eq!(
        listing_names(&harness),
        ["big-blob.bin", "delta-report.pdf", "bravo-symphony.mp3", "echo-snapshot.png", "alpha-notes.md", "charlie-archive.zip"]
    );
}

#[test]
fn large_filter_shows_only_big_file() {
    let world = TestWorld::create("large");
    let (mut harness, engine) = build_harness(&world);
    wait_until_indexed(&engine);
    harness.run_steps(2);
    enter_corpus_root(&mut harness);

    click_topmost_label(&mut harness, "Large");
    assert!(harness.state().browse.large_files_only);
    assert_eq!(harness.state().browse.active_filter().min_size, Some(100 * 1024 * 1024));
    assert_eq!(listing_names(&harness), ["big-blob.bin"]);
}

#[test]
fn recent_toggle_sets_recency_window() {
    let world = TestWorld::create("recent");
    let (mut harness, engine) = build_harness(&world);
    wait_until_indexed(&engine);
    harness.run_steps(2);
    enter_corpus_root(&mut harness);

    click_topmost_label(&mut harness, "Recent");
    let browse = &harness.state().browse;
    assert!(browse.recent_files_only);
    assert!(browse.active_filter().after_unix.is_some(), "recency window must be set");
    // charlie-archive.zip is 100 days old and must drop out of the listing.
    assert_eq!(
        listing_names(&harness),
        ["alpha-notes.md", "big-blob.bin", "bravo-symphony.mp3", "delta-report.pdf", "echo-snapshot.png"]
    );
}

#[test]
fn empty_state_message_when_filters_match_nothing() {
    let world = TestWorld::create("empty");
    let (mut harness, engine) = build_harness(&world);
    wait_until_indexed(&engine);
    harness.run_steps(2);
    enter_corpus_root(&mut harness);

    // PDF + Large matches nothing (the only PDF is small).
    click_topmost_label(&mut harness, "PDF");
    click_topmost_label(&mut harness, "Large");
    assert!(listing_names(&harness).is_empty());
    assert!(harness.query_by_label("nothing here").is_some(), "empty state must render");
    assert!(
        harness.query_by_label("no files match the current filters").is_some(),
        "empty state must blame the filters, not claim the folder is empty"
    );
}

#[test]
fn settings_window_opens_lists_roots_and_closes() {
    let world = TestWorld::create("settings");
    let (mut harness, engine) = build_harness(&world);
    wait_until_indexed(&engine);
    harness.run_steps(2);
    enter_corpus_root(&mut harness);

    click_topmost_label(&mut harness, "⚙");
    assert!(harness.state().show_settings, "gear button must open the settings window");
    assert!(harness.query_by_label("Indexed roots").is_some(), "roots section must render");
    let corpus_text = world.corpus.to_string_lossy().into_owned();
    assert!(
        harness.get_all_by_label_contains(&corpus_text).next().is_some(),
        "roots list must show the configured root"
    );

    click_topmost_label(&mut harness, "Close");
    assert!(!harness.state().show_settings, "Close must close the settings window");
}

fn type_into_settings_root_edit(harness: &mut Harness<'_, IpicApp>, text: &str) {
    let edit_center = harness
        .get(by().predicate(|node| node.placeholder() == Some("/absolute/path")))
        .rect()
        .center();
    click_and_settle(harness, edit_center);
    harness.event(egui::Event::Text(text.to_string()));
    harness.run_steps(2);
}

#[test]
fn settings_add_root_accepts_valid_and_ignores_invalid_path() {
    let world = TestWorld::create("addroot");
    let extra_root = world.corpus.parent().unwrap().join("extra-root");
    std::fs::create_dir_all(&extra_root).unwrap();
    let (mut harness, engine) = build_harness(&world);
    wait_until_indexed(&engine);
    harness.run_steps(2);
    enter_corpus_root(&mut harness);

    click_topmost_label(&mut harness, "⚙");

    // A valid typed path must be added (the buffer empties on success).
    type_into_settings_root_edit(&mut harness, &extra_root.to_string_lossy());
    click_topmost_label(&mut harness, "+ Add");
    let roots = &harness.state().config.roots;
    assert_eq!(roots.len(), 2, "valid path must be added, got {roots:?}");
    assert!(
        roots.iter().any(|root| root.ends_with("extra-root")),
        "added root must be the canonicalized typed path, got {roots:?}"
    );

    // An invalid path must be ignored.
    type_into_settings_root_edit(&mut harness, "/no/such/directory/anywhere");
    click_topmost_label(&mut harness, "+ Add");
    assert_eq!(harness.state().config.roots.len(), 2, "invalid path must not be added");
}

#[test]
fn settings_rescan_now_triggers_scan_and_closes() {
    let world = TestWorld::create("rescan");
    let (mut harness, engine) = build_harness(&world);
    wait_until_indexed(&engine);
    harness.run_steps(2);
    enter_corpus_root(&mut harness);

    click_topmost_label(&mut harness, "⚙");
    click_topmost_label(&mut harness, "Rescan now");
    assert!(!harness.state().show_settings, "Rescan now must close the settings window");
    wait_for_notice(&mut harness, "scan:");
}

#[test]
fn toolbar_rescan_button_announces_scan() {
    let world = TestWorld::create("toolbar-rescan");
    let (mut harness, engine) = build_harness(&world);
    wait_until_indexed(&engine);
    harness.run_steps(2);
    enter_corpus_root(&mut harness);

    click_topmost_label(&mut harness, "\u{e5d5}");
    assert!(notices_contain(&harness, "rescan started"));
    wait_for_notice(&mut harness, "scan:");
}

#[test]
fn status_bar_shows_file_and_vector_counts() {
    let world = TestWorld::create("status");
    let (mut harness, engine) = build_harness(&world);
    wait_until_indexed(&engine);
    harness.run_steps(2);
    enter_corpus_root(&mut harness);

    assert!(harness.query_by_label("● ready").is_some(), "idle status must be shown");
    assert!(harness.query_by_label("6 files").is_some(), "file count must be shown");
    assert!(
        harness.get_all_by_label_contains("vectors").next().is_some(),
        "vector count must be shown"
    );
}
