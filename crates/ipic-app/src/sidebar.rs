//! Sidebar as an instrument panel: library scope, kind filters with live
//! counts and capacity bars, lazy directory tree, pinned engine readout.

use crate::app::IpicApp;
use crate::browse::kind_glyph_codepoint;
use crate::theme;
use egui::{Align2, Color32, Pos2, RichText, Ui};
use ipic_core::DirRow;

pub fn draw(ui: &mut Ui, app: &mut IpicApp) {
    // Pinned engine readout reserves the bottom first; everything above
    // scrolls when the window is short.
    egui::Panel::bottom("sidebar_engine")
        .frame(
            egui::Frame::new()
                .fill(theme::SURFACE_PANEL)
                .inner_margin(egui::Margin::symmetric(2, 8))
                .stroke(egui::Stroke::new(1.0, theme::BORDER)),
        )
        .show(ui, |ui| draw_engine_readout(ui, app));
    ui.vertical(|ui| {
        section_label(ui, "Library");
        if selectable_row(
            ui,
            app.current_directory.is_none(),
            theme::icon_label(theme::icons::FOLDER_OPEN, "All files", theme::TEXT_DIM),
        ) {
            app.navigate_to(None);
        }
        for root in app.config.roots.clone() {
            let root_text = root.to_string_lossy().into_owned();
            let is_current = app
                .current_directory
                .as_ref()
                .map(|dir| dir.path == root_text)
                .unwrap_or(false);
            // Accessible label must keep the exact "Volume  {path}" text
            // (UI tests query it by contains).
            let label = RichText::new(format!("Volume  {root_text}"))
                .color(if is_current { theme::TEXT_PRIMARY } else { theme::TEXT_DIM })
                .monospace();
            if selectable_row(ui, is_current, label)
                && let Some(dir) = app.engine.dir_by_path(&root_text) {
                    app.navigate_to(Some(dir));
                }
        }
        ui.add_space(12.0);
        section_label(ui, "Filters");
        draw_kind_filters(ui, app);
        ui.add_space(12.0);
        section_label(ui, "Tree");
        egui::ScrollArea::vertical()
            .auto_shrink(false)
            .show(ui, |ui| {
                for root in app.config.roots.clone() {
                    let root_text = root.to_string_lossy().into_owned();
                    if let Some(dir) = app.engine.dir_by_path(&root_text) {
                        draw_tree_node(ui, app, &dir, 0);
                    }
                }
            });
    });
}

fn section_label(ui: &mut Ui, label: &str) {
    ui.label(theme::micro(label));
    // Short accent tick under the section label: the one place the brand
    // hue appears in the sidebar chrome.
    let (rect, _) = ui.allocate_exact_size(egui::vec2(18.0, 2.0), egui::Sense::hover());
    ui.painter().rect_filled(rect, egui::CornerRadius::same(1), theme::ACCENT.gamma_multiply(0.55));
}

fn selectable_row(ui: &mut Ui, selected: bool, content: impl Into<egui::WidgetText>) -> bool {
    ui.add_sized(
        egui::vec2(ui.available_width(), 26.0),
        egui::Button::new(content).selected(selected),
    )
    .clicked()
}

/// Right-aligned mono text painted over a row rect (bypasses layout so the
/// label and the count share one visual line).
fn paint_row_readout(ui: &mut Ui, rect: egui::Rect, text: &str, color: Color32) {
    ui.painter().text(
        Pos2::new(rect.right() - 4.0, rect.center().y),
        Align2::RIGHT_CENTER,
        text,
        egui::FontId::monospace(11.0),
        color,
    );
}

/// Capacity bar painted along a row's bottom edge (share of total bytes).
fn paint_capacity_bar(ui: &mut Ui, rect: egui::Rect, fraction: f32) {
    let thickness = 2.0;
    let track = egui::Rect::from_min_size(
        Pos2::new(rect.left(), rect.bottom() - thickness),
        egui::vec2(rect.width(), thickness),
    );
    ui.painter().rect_filled(track, egui::CornerRadius::same(1), theme::SURFACE_HOVER);
    let filled = egui::Rect::from_min_size(
        track.min,
        egui::vec2(track.width() * fraction.clamp(0.0, 1.0), thickness),
    );
    ui.painter().rect_filled(filled, egui::CornerRadius::same(1), theme::ACCENT.gamma_multiply(0.55));
}

/// Icon + label composed in one widget (mixed fonts need a layout job; a
/// plain string would tofu the icon codepoint).
fn kind_row_content(kind: ipic_core::FileKind, text_color: Color32) -> egui::WidgetText {
    let mut job = egui::text::LayoutJob::default();
    job.append(
        kind_glyph_codepoint(kind),
        0.0,
        egui::TextFormat::simple(
            egui::FontId::new(16.0, egui::FontFamily::Name("material-icons".into())),
            crate::browse::kind_glyph_color(kind),
        ),
    );
    job.append(
        "  ",
        0.0,
        egui::TextFormat::simple(egui::FontId::new(13.0, egui::FontFamily::Proportional), text_color),
    );
    job.append(
        kind.label(),
        0.0,
        egui::TextFormat::simple(egui::FontId::new(13.0, egui::FontFamily::Proportional), text_color),
    );
    egui::WidgetText::LayoutJob(job.into())
}

fn draw_kind_filters(ui: &mut Ui, app: &mut IpicApp) {
    let stats = app.engine.kind_stats();
    let grand_total_bytes: i64 = stats.iter().map(|(_, _, bytes)| (*bytes).max(0)).sum();
    for (kind, count, kind_bytes) in stats {
        let selected = app.browse.kind_filter.contains(&kind);
        let content = kind_row_content(kind, if selected { theme::TEXT_PRIMARY } else { theme::TEXT_DIM });
        let response = ui
            .add_sized(
                egui::vec2(ui.available_width(), 26.0),
                egui::Button::new(content).selected(selected),
            )
            .on_hover_text(ipic_core::util::format_size(kind_bytes));
        let rect = response.rect;
        paint_row_readout(ui, rect, &count.to_string(), if selected { theme::TEXT_PRIMARY } else { theme::TEXT_DIM });
        if grand_total_bytes > 0 {
            paint_capacity_bar(ui, rect, kind_bytes as f32 / grand_total_bytes as f32);
        }
        if response.clicked() {
            if selected {
                app.browse.kind_filter.retain(|k| *k != kind);
            } else {
                app.browse.kind_filter = vec![kind];
            }
            // Kind filters apply to browsing, so leave any search view.
            app.active_query.clear();
            app.results.hits.clear();
            app.browse.listing_stale = true;
        }
    }
}

/// Pinned engine readout: which models are live and the compute budget.
fn draw_engine_readout(ui: &mut Ui, app: &mut IpicApp) {
    let status = app.engine.status();
    ui.label(theme::micro("Engine"));
    ui.add_space(4.0);
    readout_row(ui, "EMBEDDER", &status.embedder_model);
    match &status.vision_model {
        Some(model) => readout_row(ui, "VISION", model),
        None => readout_row(ui, "VISION", "off"),
    }
    if status.transcriber_ready {
        readout_row(ui, "WHISPER", &status.whisper_model);
    } else {
        readout_row(ui, "WHISPER", "off");
    }
    readout_row(
        ui,
        "COMPUTE",
        &format!(
            "{}w / {}t",
            status.compute.extract_workers, status.compute.search_threads
        ),
    );
}

fn readout_row(ui: &mut Ui, label: &str, value: &str) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(label).size(10.0).monospace().color(theme::TEXT_DIM));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(
                RichText::new(value)
                    .size(10.5)
                    .monospace()
                    .color(theme::TEXT_PRIMARY),
            );
        });
    });
}

fn draw_tree_node(ui: &mut Ui, app: &mut IpicApp, dir: &DirRow, depth: usize) {
    ui.horizontal(|ui| {
        ui.add_space((depth * 14) as f32);
        let expanded = app.expanded_tree_nodes.contains(&dir.id);
        let is_current = app.current_directory.as_ref().map(|current| current.id == dir.id).unwrap_or(false);
        let arrow = if expanded { "▾" } else { "▸" };
        let has_children = !app.engine.tree_children_empty(dir);
        let toggle = ui.add(
            egui::Button::new(RichText::new(arrow).color(theme::TEXT_DIM))
                .fill(egui::Color32::TRANSPARENT)
                .min_size(egui::vec2(16.0, 20.0)),
        );
        if has_children && toggle.clicked() {
            if expanded {
                app.expanded_tree_nodes.remove(&dir.id);
            } else {
                app.expanded_tree_nodes.insert(dir.id);
            }
        }
        let name = std::path::Path::new(&dir.path)
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| dir.path.clone());
        let label = RichText::new(name).color(if is_current { theme::TEXT_PRIMARY } else { theme::TEXT_DIM });
        if ui
            .add_sized(egui::vec2(ui.available_width() * 0.88, 22.0), egui::Button::new(label).selected(is_current))
            .on_hover_text(&dir.path)
            .clicked()
        {
            app.navigate_to(Some(dir.clone()));
        }
        if dir.file_count > 0 {
            ui.label(RichText::new(dir.file_count.to_string()).small().monospace().color(theme::TEXT_DIM));
        }
    });
    if app.expanded_tree_nodes.contains(&dir.id) {
        for child in app.engine.tree_children(dir) {
            draw_tree_node(ui, app, &child, depth + 1);
        }
    }
}
