//! Right-hand details panel: preview (images, first indexed text), metadata, actions.

use crate::app::IpicApp;
use crate::browse::kind_glyph;
use crate::theme;
use egui::{RichText, Ui};
use ipic_core::{FileKind, RagStatus};

pub fn draw(ui: &mut Ui, app: &mut IpicApp) {
    let Some((file, path)) = app.selected_file.clone() else {
        ui.vertical_centered(|ui| {
            ui.add_space(ui.available_height() * 0.4);
            ui.label(RichText::new("select a file").color(theme::TEXT_DIM));
        });
        return;
    };
    egui::ScrollArea::vertical().show(ui, |ui| {
        ui.label(kind_glyph(file.kind).size(30.0));
        ui.label(RichText::new(&file.name).size(17.0).strong().color(theme::TEXT_PRIMARY));
        ui.label(RichText::new(&path).small().color(theme::TEXT_DIM).monospace());
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            if ui.button("Open").clicked() {
                crate::actions::open_file(std::path::Path::new(&path));
            }
            if ui.button("Reveal").clicked() {
                crate::actions::reveal_in_file_manager(std::path::Path::new(&path));
            }
            if ui.button("Copy path").clicked() {
                crate::actions::copy_path_to_clipboard(&path);
            }
        });
        ui.add_space(8.0);
        metadata_row(ui, "Kind", file.kind.label());
        metadata_row(ui, "Size", &ipic_core::util::format_size(file.size));
        metadata_row(
            ui,
            "Modified",
            &ipic_core::util::format_local_timestamp(file.mtime),
        );
        if let Some(duration) = file.duration_secs {
            metadata_row(ui, "Duration", &ipic_core::util::format_duration(duration));
        }
        let status = match file.rag {
            RagStatus::Done => ("indexed — searchable", theme::SUCCESS),
            RagStatus::Busy => ("indexing…", theme::WARNING),
            RagStatus::Failed => ("index failed", theme::DANGER),
            RagStatus::Pending => (
                if file.kind.is_rag() { "queued for indexing" } else { "not RAG-indexable" },
                theme::TEXT_DIM,
            ),
        };
        metadata_row(ui, "RAG", status.0);
        ui.add_space(10.0);
        draw_preview(ui, app, &file, &path);
    });
}

fn metadata_row(ui: &mut Ui, label: &str, value: &str) {
    ui.horizontal(|ui| {
        ui.set_min_width(ui.available_width());
        ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
            ui.label(RichText::new(label).small().color(theme::TEXT_DIM).strong());
        });
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(RichText::new(value).small().color(theme::TEXT_PRIMARY).monospace());
        });
    });
}

fn draw_preview(ui: &mut Ui, app: &mut IpicApp, file: &ipic_core::FileRow, path: &str) {
    match file.kind {
        FileKind::Image => {
            let image = egui::Image::new(format!("file://{path}"))
                .fit_to_fraction(egui::Vec2::splat(1.0))
                .max_size(egui::vec2(ui.available_width(), ui.available_width()));
            ui.add(image);
        }
        FileKind::Text | FileKind::Pdf | FileKind::Audio | FileKind::Video => {
            section(ui, if file.kind.is_rag() { "Indexed content" } else { "Preview" });
            let excerpt = first_chunk_text(app, file.id);
            match excerpt {
                Some(text) => {
                    let frame = egui::Frame::new().fill(theme::SURFACE_CARD).corner_radius(egui::CornerRadius::same(8)).inner_margin(egui::Margin::same(10));
                    frame.show(ui, |ui| {
                        ui.set_max_width(ui.available_width());
                        ui.label(
                            RichText::new(text.chars().take(800).collect::<String>())
                                .small()
                                .color(theme::TEXT_PRIMARY),
                        );
                    });
                }
                None => {
                    ui.label(RichText::new(match file.kind {
                        FileKind::Audio | FileKind::Video => "transcription pending…",
                        _ => "no indexed text yet",
                    })
                    .color(theme::TEXT_DIM)
                    .small());
                }
            }
        }
        FileKind::Other => {}
    }
}

fn section(ui: &mut Ui, label: &str) {
    ui.label(RichText::new(label.to_uppercase()).small().strong().color(theme::TEXT_DIM));
}

/// First indexed chunk of a file (its extract or transcript).
fn first_chunk_text(app: &mut IpicApp, file_id: i64) -> Option<String> {
    let connection = app.engine.catalog.reader().ok()?;
    connection
        .query_row(
            "SELECT text FROM chunks WHERE file_id = ?1 ORDER BY rowid LIMIT 1",
            rusqlite::params![file_id],
            |row| row.get::<_, String>(0),
        )
        .ok()
}
