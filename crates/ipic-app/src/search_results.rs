//! Search screen: query header with result telemetry, dense ranked result
//! rows with lane chips and score readouts, keyboard-first navigation.

use crate::app::IpicApp;
use crate::theme;
use egui::{Color32, RichText, Ui};
use ipic_core::FileKind;
use ipic_rag::SearchHit;

#[derive(Default)]
pub struct ResultsPanel {
    pub hits: Vec<SearchHit>,
    pub elapsed_millis: f32,
    pub interpreted_query: Option<String>,
    pub vector_count: u32,
    pub selected_index: Option<usize>,
    pub request_search_focus: bool,
}

/// Draws the search screen when a unified query is active.
pub fn draw(ui: &mut Ui, app: &mut IpicApp) {
    draw_query_header(ui, app);
    ui.add_space(8.0);
    // Keyboard affordance strip: the search screen is navigable without
    // touching the mouse.
    ui.label(theme::micro("↑↓ select · ↵ open · esc back to browsing"));
    ui.add_space(6.0);
    if app.searching {
        ui.add_space(ui.available_height() * 0.3);
        ui.vertical_centered(|ui| {
            ui.spinner();
            ui.label(RichText::new("searching").color(theme::TEXT_DIM).small());
        });
        return;
    }
    if app.results.hits.is_empty() {
        ui.add_space(ui.available_height() * 0.28);
        ui.vertical_centered(|ui| {
            ui.label(
                egui::RichText::new(theme::icons::SEARCH)
                    .font(egui::FontId::new(34.0, egui::FontFamily::Name("material-icons".into())))
                    .color(theme::SURFACE_HOVER),
            );
            ui.add_space(8.0);
            ui.label(RichText::new("no matches").size(17.0).color(theme::TEXT_DIM));
            ui.label(
                RichText::new("try different words, or press Esc to go back")
                    .color(theme::TEXT_DIM)
                    .small(),
            );
        });
        return;
    }
    draw_keyboard_navigation(ui, app);
    let top_score = app.results.hits.first().map(|hit| hit.score).unwrap_or(1.0);
    let hits = app.results.hits.clone();
    egui::ScrollArea::vertical()
        .auto_shrink(false)
        .show(ui, |ui| {
            for (position, hit) in hits.iter().enumerate() {
                let selected = app.results.selected_index == Some(position);
                draw_result_row(ui, app, position, hit, top_score, selected);
                ui.add_space(4.0);
            }
        });
}

/// Header: query echo, spoken transcript, right-aligned result telemetry.
fn draw_query_header(ui: &mut Ui, app: &mut IpicApp) {
    ui.horizontal_top(|ui| {
        ui.vertical(|ui| {
            ui.label(theme::micro("query"));
            ui.label(
                RichText::new(format!("“{}”", app.active_query))
                    .size(19.0)
                    .strong()
                    .color(theme::TEXT_PRIMARY),
            );
            if let Some(transcript) = &app.results.interpreted_query {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new(theme::icons::MIC)
                            .font(egui::FontId::new(13.0, egui::FontFamily::Name("material-icons".into())))
                            .color(theme::ACCENT),
                    );
                    ui.label(
                        RichText::new(format!("heard: “{transcript}”"))
                            .color(theme::ACCENT)
                            .small(),
                    );
                });
            }
        });
        ui.with_layout(egui::Layout::right_to_left(egui::Align::TOP), |ui| {
            let telemetry = format!(
                "{} results · {:.0} ms · {} vectors",
                app.results.hits.len(),
                app.results.elapsed_millis,
                format_count(app.results.vector_count),
            );
            ui.label(theme::mono_dim(telemetry));
        });
    });
    // Hairline separating the query block from the result list.
    let rect = ui.max_rect();
    ui.painter()
        .hline(rect.left()..=rect.right(), ui.cursor().top(), egui::Stroke::new(1.0, theme::BORDER));
    ui.add_space(6.0);
}

fn format_count(value: u32) -> String {
    if value >= 1_000_000 {
        format!("{:.1}M", value as f64 / 1_000_000.0)
    } else if value >= 1_000 {
        format!("{:.1}k", value as f64 / 1_000.0)
    } else {
        value.to_string()
    }
}

/// ↑/↓ move the selection, mirroring the selected file into the inspector.
fn draw_keyboard_navigation(ui: &mut Ui, app: &mut IpicApp) {
    if app.search_box_has_focus {
        return;
    }
    let count = app.results.hits.len();
    if count == 0 {
        return;
    }
    let (mut down, mut up) = (false, false);
    ui.input(|input| {
        down = input.key_pressed(egui::Key::ArrowDown);
        up = input.key_pressed(egui::Key::ArrowUp);
    });
    if !(down || up) {
        return;
    }
    let next = match app.results.selected_index {
        None => 0,
        Some(current) => {
            if down {
                (current + 1).min(count - 1)
            } else {
                current.saturating_sub(1)
            }
        }
    };
    app.results.selected_index = Some(next);
    if let Some(hit) = app.results.hits.get(next) {
        app.selected_file = Some((hit.file.clone(), hit.path.clone()));
    }
}

fn draw_result_row(
    ui: &mut Ui,
    app: &mut IpicApp,
    position: usize,
    hit: &SearchHit,
    top_score: f32,
    selected: bool,
) {
    let frame = egui::Frame::new()
        .fill(if selected { theme::ACCENT_SOFT } else { theme::SURFACE_CARD })
        .corner_radius(8)
        .inner_margin(egui::Margin::same(10))
        .outer_margin(egui::Margin::symmetric(2, 0))
        .stroke(if selected {
            egui::Stroke::new(1.0, theme::ACCENT)
        } else {
            egui::Stroke::new(1.0, theme::BORDER)
        });
    let frame_response = frame.show(ui, |ui| {
        ui.horizontal_top(|ui| {
            // Rank: zero-padded mono so digits align down the list.
            ui.label(
                RichText::new(format!("{:02}", position + 1))
                    .monospace()
                    .size(12.0)
                    .color(if selected { theme::ACCENT } else { theme::TEXT_DIM }),
            );
            ui.add_space(4.0);
            if hit.file.kind == FileKind::Image {
                ui.add(
                    egui::Image::new(format!("file://{}", hit.path))
                        .fit_to_exact_size(egui::vec2(48.0, 48.0))
                        .corner_radius(4.0),
                );
            } else {
                // Elevated kind tile: rounded square with the glyph centered.
                let (tile, _) = ui.allocate_exact_size(egui::vec2(48.0, 48.0), egui::Sense::hover());
                ui.painter()
                    .rect_filled(tile, egui::CornerRadius::same(4), theme::SURFACE_ELEVATED);
                ui.painter().text(
                    tile.center(),
                    egui::Align2::CENTER_CENTER,
                    crate::browse::kind_glyph_codepoint(hit.file.kind),
                    egui::FontId::new(22.0, egui::FontFamily::Name("material-icons".into())),
                    crate::browse::kind_glyph_color(hit.file.kind),
                );
            }
            ui.add_space(4.0);
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    let name = ui.add(
                        egui::Button::new(
                            RichText::new(&hit.file.name).strong().color(theme::TEXT_PRIMARY).size(13.5),
                        )
                        .fill(Color32::TRANSPARENT),
                    );
                    if name.clicked() {
                        select_hit(app, position, hit, true);
                    }
                    ui.label(RichText::new(hit.file.kind.label()).color(theme::TEXT_DIM).small());
                    draw_lane_chips(ui, hit);
                });
                let path_button = ui.add(
                    egui::Button::new(
                        RichText::new(&hit.path).color(theme::TEXT_DIM).small().monospace().size(11.0),
                    )
                    .fill(Color32::TRANSPARENT),
                );
                if path_button.clicked() {
                    select_hit(app, position, hit, false);
                    if let Some(parent) = std::path::Path::new(&hit.path).parent()
                        && let Some(directory) = app.engine.dir_by_path(&parent.to_string_lossy())
                    {
                        app.navigate_to(Some(directory));
                    }
                }
                if !hit.snippet.is_empty() {
                    ui.add_space(2.0);
                    ui.label(
                        RichText::new(hit.snippet.chars().take(220).collect::<String>())
                            .color(theme::TEXT_DIM)
                            .small(),
                    );
                }
            });
            // Right rail: score readout + proportional bar, right-aligned.
            ui.with_layout(egui::Layout::right_to_left(egui::Align::TOP), |ui| {
                ui.vertical(|ui| {
                    ui.label(
                        RichText::new(format!("{:.3}", hit.score))
                            .monospace()
                            .size(11.0)
                            .color(if selected { theme::ACCENT } else { theme::TEXT_DIM }),
                    );
                    theme::prop_bar(ui, 56.0, 3.0, hit.score / top_score.max(1e-6), theme::ACCENT);
                    ui.add_space(2.0);
                    if ui.small_button("Open").clicked() {
                        app.dispatch_open(std::path::Path::new(&hit.path));
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
    ui.set_min_width(200.0);
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

/// Lane chips: which retrieval lanes voted for this hit. Uppercase mono on a
/// tinted background; hues follow the lane's signal family.
fn draw_lane_chips(ui: &mut Ui, hit: &SearchHit) {
    let lanes = [
        (hit.sources.semantic, "semantic", theme::ACCENT),
        (hit.sources.vision, "content", theme::ACCENT),
        (hit.sources.keyword, "keyword", theme::SUCCESS),
        (hit.sources.acoustic, "acoustic", theme::SUCCESS),
        (hit.sources.filename, "filename", theme::WARNING),
    ];
    for (active, label, color) in lanes {
        if active {
            ui.label(
                RichText::new(label.to_uppercase())
                    .size(9.0)
                    .monospace()
                    .strong()
                    .color(color)
                    .background_color(color.gamma_multiply(0.15)),
            );
        }
    }
}
