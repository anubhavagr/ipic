//! Sidebar: locations, kind filters with live counts, lazy directory tree.

use crate::app::IpicApp;
use crate::browse::kind_glyph;
use crate::theme;
use egui::{RichText, Ui};
use ipic_core::DirRow;

pub fn draw(ui: &mut Ui, app: &mut IpicApp) {
    ui.vertical(|ui| {
        section_label(ui, "Locations");
        if selectable_row(ui, app.current_directory.is_none(), "🗄", "All files") {
            app.navigate_to(None);
        }
        for root in app.config.roots.clone() {
            let root_text = root.to_string_lossy().into_owned();
            let is_current = app
                .current_directory
                .as_ref()
                .map(|dir| dir.path == root_text)
                .unwrap_or(false);
            if selectable_row(ui, is_current, "Volume", &root_text)
                && let Some(dir) = app.engine.dir_by_path(&root_text) {
                    app.navigate_to(Some(dir));
                }
        }
        ui.add_space(10.0);
        section_label(ui, "Filter by kind");
        draw_kind_filters(ui, app);
        ui.add_space(10.0);
        section_label(ui, "Tree");
        egui::ScrollArea::vertical().show(ui, |ui| {
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
    ui.label(RichText::new(label.to_uppercase()).small().color(theme::TEXT_DIM).strong());
}

fn selectable_row(ui: &mut Ui, selected: bool, glyph: &str, label: &str) -> bool {
    let fill = if selected { theme::ACCENT_SOFT } else { egui::Color32::TRANSPARENT };
    let text_color = if selected { theme::TEXT_PRIMARY } else { theme::TEXT_DIM };
    ui.add(
        egui::Button::new(RichText::new(format!("{glyph}  {label}")).color(text_color))
            .fill(fill)
            .min_size(egui::vec2(ui.available_width(), 24.0)),
    )
    .clicked()
}

fn draw_kind_filters(ui: &mut Ui, app: &mut IpicApp) {
    for (kind, count, total_bytes) in app.engine.kind_stats() {
        let selected = app.browse.kind_filter.contains(&kind);
        let fill = if selected { theme::ACCENT_SOFT } else { theme::SURFACE_CARD };
        let label = format!("{}  {} · {}", kind_glyph(kind).text(), kind.label(), count);
        if ui
            .add(
                egui::Button::new(
                    RichText::new(label).color(if selected { theme::TEXT_PRIMARY } else { theme::TEXT_DIM }),
                )
                .fill(fill)
                .min_size(egui::vec2(ui.available_width(), 24.0)),
            )
            .on_hover_text(ipic_core::util::format_size(total_bytes))
            .clicked()
        {
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

fn draw_tree_node(ui: &mut Ui, app: &mut IpicApp, dir: &DirRow, depth: usize) {
    let indent = egui::Layout::left_to_right(egui::Align::Center);
    let _ = indent;
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
        let fill = if is_current { theme::ACCENT_SOFT } else { egui::Color32::TRANSPARENT };
        let name = std::path::Path::new(&dir.path)
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| dir.path.clone());
        if ui
            .add(
                egui::Button::new(
                    RichText::new(name).color(if is_current { theme::TEXT_PRIMARY } else { theme::TEXT_DIM }),
                )
                .fill(fill),
            )
            .on_hover_text(&dir.path)
            .clicked()
        {
            app.navigate_to(Some(dir.clone()));
        }
        if dir.file_count > 0 {
            ui.label(RichText::new(dir.file_count.to_string()).small().color(theme::TEXT_DIM));
        }
    });
    if app.expanded_tree_nodes.contains(&dir.id) {
        for child in app.engine.tree_children(dir) {
            draw_tree_node(ui, app, &child, depth + 1);
        }
    }
}
