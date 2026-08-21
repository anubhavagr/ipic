//! Unified multimodal search results — every modality in one ranked list.

use crate::app::IpicApp;
use crate::browse::kind_glyph;
use crate::theme;
use egui::{Color32, RichText, Ui};
use ipic_rag::SearchHit;

#[derive(Default)]
pub struct ResultsPanel {
    pub hits: Vec<SearchHit>,
    pub elapsed_millis: f32,
    pub interpreted_query: Option<String>,
    pub selected_index: Option<usize>,
    pub request_search_focus: bool,
}

/// Draws the results view when a unified query is active.
pub fn draw(ui: &mut Ui, app: &mut IpicApp) {
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(format!("Results for “{}”", app.active_query))
                .size(17.0)
                .strong()
                .color(theme::TEXT_PRIMARY),
        );
        if let Some(transcript) = &app.results.interpreted_query {
            ui.label(RichText::new(format!("🎙 heard: “{transcript}”")).color(theme::ACCENT).small());
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(
                RichText::new(format!(
                    "{} hits · {:.0} ms · Esc to clear",
                    app.results.hits.len(),
                    app.results.elapsed_millis
                ))
                .color(theme::TEXT_DIM)
                .small(),
            );
        });
    });
    ui.add_space(6.0);
    if app.results.hits.is_empty() && !app.searching {
        ui.add_space(40.0);
        ui.vertical_centered(|ui| {
            ui.label(RichText::new("no matches").size(18.0).color(theme::TEXT_DIM));
            ui.label(RichText::new("try different words, or press Esc to go back").color(theme::TEXT_DIM).small());
        });
        return;
    }
    let top_score = app.results.hits.first().map(|hit| hit.score).unwrap_or(1.0);
    let hits = app.results.hits.clone();
    egui::ScrollArea::vertical().show(ui, |ui| {
        for (position, hit) in hits.iter().enumerate() {
            let selected = app.results.selected_index == Some(position);
            draw_result_card(ui, app, position, hit, top_score, selected);
            ui.add_space(6.0);
        }
    });
}

fn draw_result_card(ui: &mut Ui, app: &mut IpicApp, position: usize, hit: &SearchHit, top_score: f32, selected: bool) {
    let fill = if selected { theme::ACCENT_SOFT } else { theme::SURFACE_CARD };
    let frame = egui::Frame::new()
        .fill(fill)
        .corner_radius(8)
        .inner_margin(egui::Margin::same(12))
        .outer_margin(egui::Margin::symmetric(2, 0))
        .stroke(if selected {
            egui::Stroke::new(1.0, theme::ACCENT)
        } else {
            egui::Stroke::NONE
        });
    let frame_response = frame.show(ui, |ui| {
        ui.horizontal_top(|ui| {
            ui.label(kind_glyph(hit.file.kind).size(18.0));
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(format!("{}.", position + 1)).color(theme::TEXT_DIM).small().strong());
                    let name = ui.add(
                        egui::Button::new(RichText::new(&hit.file.name).strong().color(theme::TEXT_PRIMARY))
                            .fill(Color32::TRANSPARENT),
                    );
                    if name.clicked() {
                        select_hit(app, position, hit, true);
                    }
                    ui.label(RichText::new(hit.file.kind.label()).color(theme::TEXT_DIM).small());
                    draw_lane_badges(ui, hit);
                });
                let path_button = ui.add(
                    egui::Button::new(RichText::new(&hit.path).color(theme::TEXT_DIM).small().monospace())
                        .fill(Color32::TRANSPARENT),
                );
                if path_button.clicked() {
                    select_hit(app, position, hit, false);
                    if let Some(parent) = std::path::Path::new(&hit.path).parent()
                        && let Ok(connection) = app.engine.catalog.reader()
                            && let Some(directory) =
                                app.engine.catalog.dir_by_path(&connection, &parent.to_string_lossy()).ok().flatten()
                            {
                                app.navigate_to(Some(directory));
                            }
                }
                if !hit.snippet.is_empty() {
                    ui.add_space(2.0);
                    ui.label(
                        RichText::new(hit.snippet.chars().take(240).collect::<String>())
                            .color(Color32::from_rgb(96, 102, 112))
                            .small(),
                    );
                }
                draw_score_bar(ui, hit.score / top_score.max(1e-6));
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    if ui.small_button("Open").clicked() {
                        app.dispatch_open(std::path::Path::new(&hit.path));
                    }
                    if ui.small_button("Reveal").clicked() {
                        app.dispatch_reveal(std::path::Path::new(&hit.path));
                    }
                });
            });
        });
    });
    if frame_response.response.clicked() {
        select_hit(app, position, hit, false);
    }
    frame_response.response.context_menu(|ui| {
        result_context_menu(ui, app, hit);
    });
    let enter_opens = !app.search_box_has_focus
        && ui.input(|input| input.key_pressed(egui::Key::Enter))
        && app.results.selected_index == Some(position);
    if enter_opens {
        app.dispatch_open(std::path::Path::new(&hit.path));
    }
}

fn select_hit(app: &mut IpicApp, position: usize, hit: &SearchHit, also_open: bool) {
    app.results.selected_index = Some(position);
    app.selected_file = Some((hit.file.clone(), hit.path.clone()));
    if also_open {
        app.dispatch_open(std::path::Path::new(&hit.path));
    }
}

fn result_context_menu(ui: &mut Ui, app: &mut IpicApp, hit: &SearchHit) {
    if ui.button("Open").clicked() {
        app.dispatch_open(std::path::Path::new(&hit.path));
        ui.close();
    }
    if ui.button("Reveal in file manager").clicked() {
        app.dispatch_reveal(std::path::Path::new(&hit.path));
        ui.close();
    }
    if ui.button("Copy path").clicked() {
        crate::actions::copy_path_to_clipboard(&hit.path);
        app.push_notice("path copied".into());
        ui.close();
    }
}

fn draw_lane_badges(ui: &mut Ui, hit: &SearchHit) {
    let lanes = [
        (hit.sources.semantic, "semantic", theme::ACCENT),
        (hit.sources.keyword, "keyword", theme::SUCCESS),
        (hit.sources.acoustic, "acoustic", theme::SUCCESS),
        (hit.sources.filename, "filename", theme::WARNING),
    ];
    for (active, label, color) in lanes {
        if active {
            ui.label(RichText::new(label).small().color(color));
        }
    }
}

fn draw_score_bar(ui: &mut Ui, relative_score: f32) {
    let width = 180.0;
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, 4.0), egui::Sense::hover());
    let painter = ui.painter();
    painter.rect_filled(rect, egui::CornerRadius::same(2), theme::SURFACE_PANEL);
    let filled =
        egui::Rect::from_min_size(rect.min, egui::vec2(width * relative_score.clamp(0.0, 1.0), 4.0));
    painter.rect_filled(filled, egui::CornerRadius::same(2), theme::ACCENT);
}
