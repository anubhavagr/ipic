//! Ask mode: natural-language (or spoken) hybrid search with scored result cards.

use crate::app::IpicApp;
use crate::browse::kind_glyph;
use crate::theme;
use egui::{Color32, RichText, Ui};
use ipic_rag::SearchOutcome;

#[derive(Default)]
pub struct AskPanel {
    pub query_edit: String,
    pub results: Option<SearchOutcome>,
}

pub fn draw(ui: &mut Ui, app: &mut IpicApp) {
    let has_results = app.ask.results.is_some();
    if !has_results {
        ui.add_space(ui.available_height() * 0.18);
    }
    ui.vertical_centered(|ui| {
        if !has_results {
            ui.label(RichText::new("Ask ipic anything").size(30.0).strong().color(theme::TEXT_PRIMARY));
            ui.label(RichText::new("semantic · keyword · filename — fused, fully on-device").color(theme::TEXT_DIM));
        }
        ui.add_space(16.0);
        draw_query_bar(ui, app);
    });
    if app.searching {
        ui.add_space(24.0);
        ui.vertical_centered(|ui| {
            ui.spinner();
            ui.label(RichText::new("searching…").color(theme::TEXT_DIM));
        });
        return;
    }
    let Some(outcome) = app.ask.results.clone() else { return };
    ui.add_space(10.0);
    if let Some(transcript) = &outcome.interpreted_query {
        ui.horizontal(|ui| {
            ui.label(RichText::new("🎙 heard").color(theme::ACCENT).strong());
            ui.label(RichText::new(format!("\"{transcript}\"")).color(theme::TEXT_PRIMARY));
        });
        ui.add_space(4.0);
    }
    ui.label(
        RichText::new(format!(
            "{} results · {:.0} ms · {} vectors",
            outcome.hits.len(),
            outcome.elapsed_millis,
            outcome.vector_count
        ))
        .color(theme::TEXT_DIM)
        .small(),
    );
    ui.add_space(8.0);
    let top_score = outcome.hits.first().map(|hit| hit.score).unwrap_or(1.0);
    egui::ScrollArea::vertical().show(ui, |ui| {
        for (position, hit) in outcome.hits.iter().enumerate() {
            draw_result_card(ui, app, position, hit, top_score);
            ui.add_space(6.0);
        }
    });
}

fn draw_query_bar(ui: &mut Ui, app: &mut IpicApp) {
    ui.horizontal(|ui| {
        let width = (ui.available_width() - 130.0).max(240.0);
        let edit = egui::TextEdit::singleline(&mut app.ask.query_edit)
            .hint_text("e.g. “notes about radiation shielding on mars”")
            .desired_width(width)
            .font(egui::TextStyle::Body)
            .clip_text(true);
        let response = ui.add(edit);
        if (response.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter)))
            || ui.input(|input| input.key_pressed(egui::Key::Enter) && app.ask.query_edit.len() > 2 && !response.has_focus())
        {
            app.start_search(app.ask.query_edit.clone());
        }
        let recording = app.recorder.is_some();
        let mic_label = if recording {
            RichText::new(format!("● {:.0}s", app.recorder.as_ref().map(|r| r.captured_seconds()).unwrap_or(0.0)))
                .color(theme::DANGER)
                .strong()
        } else {
            RichText::new("🎙 Speak").color(theme::TEXT_PRIMARY)
        };
        let mic_fill = if recording { theme::DANGER } else { theme::SURFACE_CARD };
        if ui.add(egui::Button::new(mic_label).fill(mic_fill)).clicked() {
            app.toggle_recording();
        }
        if ui.button("Search").clicked() {
            app.start_search(app.ask.query_edit.clone());
        }
        response.request_focus();
    });
}

fn draw_result_card(ui: &mut Ui, _app: &mut IpicApp, position: usize, hit: &ipic_rag::SearchHit, top_score: f32) {
    let frame = egui::Frame::new()
        .fill(theme::SURFACE_CARD)
        .corner_radius(8.0)
        .inner_margin(egui::Margin::same(12))
        .outer_margin(egui::Margin::symmetric(2, 0));
    frame.show(ui, |ui| {
        ui.horizontal_top(|ui| {
            ui.label(kind_glyph(hit.file.kind).size(18.0));
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(format!("{}.", position + 1)).color(theme::TEXT_DIM).small().strong());
                    ui.label(RichText::new(&hit.file.name).strong().color(theme::TEXT_PRIMARY));
                    ui.label(RichText::new(hit.file.kind.label()).color(theme::TEXT_DIM).small());
                    draw_lane_badges(ui, hit);
                });
                ui.label(RichText::new(&hit.path).color(theme::TEXT_DIM).small().monospace());
                if !hit.snippet.is_empty() {
                    ui.add_space(2.0);
                    ui.label(
                        RichText::new(hit.snippet.chars().take(240).collect::<String>())
                            .color(Color32::from_rgb(168, 175, 190))
                            .small(),
                    );
                }
                draw_score_bar(ui, hit.score / top_score.max(1e-6));
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    if ui.small_button("Open").clicked() {
                        crate::actions::open_file(std::path::Path::new(&hit.path));
                    }
                    if ui.small_button("Reveal").clicked() {
                        crate::actions::reveal_in_file_manager(std::path::Path::new(&hit.path));
                    }
                });
            });
        });
    });
}

fn draw_lane_badges(ui: &mut Ui, hit: &ipic_rag::SearchHit) {
    let lanes = [
        (hit.sources.semantic, "semantic", theme::ACCENT),
        (hit.sources.keyword, "keyword", theme::SUCCESS),
        (hit.sources.filename, "filename", theme::WARNING),
    ];
    for (active, label, color) in lanes {
        if active {
            ui.label(
                RichText::new(label)
                    .small()
                    .color(color)
                    .background_color(color.gamma_multiply(0.15)),
            );
        }
    }
}

fn draw_score_bar(ui: &mut Ui, relative_score: f32) {
    let width = 180.0;
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, 4.0), egui::Sense::hover());
    let painter = ui.painter();
    painter.rect_filled(rect, egui::CornerRadius::same(2), theme::SURFACE_PANEL);
    let filled = egui::Rect::from_min_size(rect.min, egui::vec2(width * relative_score.clamp(0.0, 1.0), 4.0));
    painter.rect_filled(filled, egui::CornerRadius::same(2), theme::ACCENT);
}
