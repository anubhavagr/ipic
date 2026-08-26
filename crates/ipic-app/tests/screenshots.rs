//! Documentation screenshots against the real user index (IPIC_SCREENSHOTS=1).
//! Hermetic CI runs skip this entirely.

use egui::accesskit::Role;
use egui_kittest::kittest::{by, Queryable};
use egui_kittest::{Harness, HarnessBuilder, SnapshotOptions};
use ipic::{theme, IpicApp};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[test]
fn capture_documentation_screenshots() {
    if std::env::var("IPIC_SCREENSHOTS").is_err() {
        return;
    }
    let config = ipic_core::Config::load_or_create().unwrap();
    let engine = ipic_rag::Engine::launch(config.clone()).unwrap();
    let settled = Instant::now();
    while Instant::now() - settled < Duration::from_secs(240) {
        let status = engine.status();
        if !status.scanning && status.pending == 0 {
            break;
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    let application = IpicApp::new(Arc::clone(&engine), config);
    let options = SnapshotOptions::new().output_path("docs/img");
    let mut harness = HarnessBuilder::default()
        .with_size([1280.0, 800.0])
        .with_options(options)
        .build_eframe(move |creation_context| {
            theme::apply(&creation_context.egui_ctx);
            application
        });

    // Library view: roots in the sidebar, ready dot in the footer.
    harness.run_steps(4);
    harness.snapshot("ipic-library");

    // Enter the root and show the file table with status dots.
    let root_row = harness
        .query_all_by_label_contains(&format!("Volume  {}", engine.current_roots()[0].display()))
        .next()
        .expect("sidebar lists the root")
        .rect();
    click_at(&mut harness, root_row.center());
    harness.run_steps(6);
    harness.snapshot("ipic-browse");

    // Unified search over every modality.
    let query = "api pipeline screenshot";
    let search_input = harness.get(by().role(Role::TextInput));
    search_input.click();
    harness.run_steps(2);
    harness.get(by().role(Role::TextInput)).type_text(query);
    harness.run_steps(2);
    harness.key_press(egui::Key::Enter);
    harness.run_steps(1);
    let searched = Instant::now();
    while Instant::now() - searched < Duration::from_secs(30) {
        let state = harness.state();
        if !state.searching && !state.results.hits.is_empty() {
            break;
        }
        harness.run_steps(1);
    }
    assert!(
        !harness.state().results.hits.is_empty(),
        "search must produce hits for the screenshot"
    );
    harness.run_steps(6);
    harness.snapshot("ipic-search");

    engine.shutdown();
}

/// Emits a click with real-input timing: move, press, release — one frame apart.
fn click_at(harness: &mut Harness<'_, IpicApp>, position: egui::Pos2) {
    let modifiers = egui::Modifiers::default();
    harness.event(egui::Event::PointerMoved(position));
    harness.run_steps(2);
    harness.event(egui::Event::PointerButton {
        pos: position,
        button: egui::PointerButton::Primary,
        pressed: true,
        modifiers,
    });
    harness.run_steps(2);
    harness.event(egui::Event::PointerButton {
        pos: position,
        button: egui::PointerButton::Primary,
        pressed: false,
        modifiers,
    });
    harness.run_steps(2);
}
