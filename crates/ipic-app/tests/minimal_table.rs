// Regression: click-sensing labels register clicks, plain and inside TableBuilder.
// kittest queues press+release in one step; egui needs them in separate frames
// (real input always spans frames), so clicks are emitted phase by phase.
use egui_kittest::kittest::Queryable;
use egui_kittest::Harness;
use std::cell::Cell;

/// Emits a click with real-input timing: move, press, release — one frame apart.
pub fn click_at<State>(harness: &mut Harness<'_, State>, position: egui::Pos2, button: egui::PointerButton) {
    let modifiers = egui::Modifiers::default();
    harness.event(egui::Event::PointerMoved(position));
    harness.run();
    harness.event(egui::Event::PointerButton { pos: position, button, pressed: true, modifiers });
    harness.run();
    harness.event(egui::Event::PointerButton { pos: position, button, pressed: false, modifiers });
    harness.run();
}

#[test]
fn plain_label_click_registers() {
    let clicked = Cell::new(false);
    let mut harness = Harness::new_ui(|ui| {
        let response = ui.add(
            egui::Label::new(egui::RichText::new("target-file.md").strong()).sense(egui::Sense::click()),
        );
        if response.clicked() {
            clicked.set(true);
        }
    });
    harness.run();
    let position = harness.get_by_label("target-file.md").rect().center();
    click_at(&mut harness, position, egui::PointerButton::Primary);
    assert!(clicked.get(), "plain click-sensing label must register clicks");
}

#[test]
fn table_label_click_registers() {
    let clicked = Cell::new(false);
    let mut harness = Harness::new_ui(|ui| {
        egui_extras::TableBuilder::new(ui)
            .column(egui_extras::Column::remainder())
            .column(egui_extras::Column::exact(80.0))
            .header(24.0, |mut header| {
                header.col(|ui| {
                    ui.label("Name");
                });
                header.col(|_ui| {});
            })
            .body(|body| {
                body.rows(28.0, 1, |mut row| {
                    row.col(|ui| {
                        let response = ui.add(
                            egui::Label::new(egui::RichText::new("target-file.md").strong())
                                .sense(egui::Sense::click()),
                        );
                        if response.clicked() {
                            clicked.set(true);
                        }
                    });
                    row.col(|_ui| {});
                });
            });
    });
    harness.run();
    let position = harness.get_by_label("target-file.md").rect().center();
    click_at(&mut harness, position, egui::PointerButton::Primary);
    assert!(clicked.get(), "label click inside TableBuilder must register");
}
