//! Browse mode: filter toolbar + virtualized, sortable file table with
//! context actions (open, reveal, copy path, rename, trash).

use crate::app::IpicApp;
use crate::theme;
use egui::{Color32, RichText, Ui};
use egui_extras::{Column, TableBuilder};
use ipic_core::DirRow;
use ipic_core::{FileFilter, FileKind, FileRow, RagStatus, SortKey};

#[derive(Debug, Clone)]
pub enum ListingEntry {
    Directory(DirRow),
    File(FileRow),
}

pub struct RenameDialog {
    pub file_id: i64,
    #[allow(dead_code)]
    pub current_name: String,
    pub edit_buffer: String,
}

pub struct BrowsePanel {
    pub active: bool,
    pub name_filter_edit: String,
    pub name_filter_active: String,
    pub kind_filter: Vec<FileKind>,
    pub large_files_only: bool,
    pub recent_files_only: bool,
    pub sort_key: SortKey,
    pub sort_ascending: bool,
    pub listing: Vec<ListingEntry>,
    pub listing_dir: Option<i64>,
    pub listing_stale: bool,
    pub rename_dialog: Option<RenameDialog>,
}

impl Default for BrowsePanel {
    fn default() -> Self {
        Self {
            active: true,
            name_filter_edit: String::new(),
            name_filter_active: String::new(),
            kind_filter: Vec::new(),
            large_files_only: false,
            recent_files_only: false,
            sort_key: SortKey::Name,
            sort_ascending: true,
            listing: Vec::new(),
            listing_dir: None,
            listing_stale: true,
            rename_dialog: None,
        }
    }
}

impl BrowsePanel {
    pub fn active_filter(&self) -> FileFilter {
        let mut filter = FileFilter {
            name_query: (!self.name_filter_active.is_empty()).then(|| self.name_filter_active.clone()),
            kinds: self.kind_filter.clone(),
            ..Default::default()
        };
        if self.large_files_only {
            filter.min_size = Some(100 * 1024 * 1024);
        }
        if self.recent_files_only {
            filter.after_unix = Some(ipic_core::util::unix_now() - 30 * 24 * 3600);
        }
        filter
    }

    /// Reloads the visible listing from the catalog (SQL-side filter + sort).
    pub fn refresh_listing(app: &mut IpicApp) {
        let browse = &mut app.browse;
        let connection = match app.engine.catalog.reader() {
            Ok(connection) => connection,
            Err(_) => return,
        };
        let directory_id = app.current_directory.as_ref().map(|dir| dir.id);
        let files = app
            .engine
            .catalog
            .children(
                &connection,
                directory_id,
                &browse.active_filter(),
                browse.sort_key,
                browse.sort_ascending,
                20_000,
            )
            .unwrap_or_default();
        let directories = if directory_id.is_none() {
            Vec::new()
        } else {
            app.engine
                .catalog
                .tree_children(&connection, directory_id)
                .unwrap_or_default()
                .into_iter()
                .filter(|dir| dir.file_count > 0 || dir.subdir_count > 0 || true)
                .collect::<Vec<_>>()
        };
        let mut listing: Vec<ListingEntry> = directories.into_iter().map(ListingEntry::Directory).collect();
        listing.extend(files.into_iter().map(ListingEntry::File));
        browse.listing = listing;
        browse.listing_dir = directory_id;
        browse.listing_stale = false;
    }
}

pub fn draw(ui: &mut Ui, app: &mut IpicApp) {
    draw_toolbar(ui, app);
    ui.add_space(2.0);
    draw_table(ui, app);
}

fn draw_toolbar(ui: &mut Ui, app: &mut IpicApp) {
    ui.horizontal(|ui| {
        let search_edit = egui::TextEdit::singleline(&mut app.browse.name_filter_edit)
            .hint_text("Filter by name…")
            .desired_width(ui.available_width() - 460.0)
            .clip_text(true);
        let response = ui.add(search_edit);
        if response.lost_focus() || app.browse.name_filter_edit.is_empty() {
            if app.browse.name_filter_active != app.browse.name_filter_edit {
                app.browse.name_filter_active = app.browse.name_filter_edit.clone();
                app.browse.listing_stale = true;
            }
        }
        if response.changed() && app.browse.name_filter_edit.len() % 3 == 0 {
            app.browse.name_filter_active = app.browse.name_filter_edit.clone();
            app.browse.listing_stale = true;
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let direction = if app.browse.sort_ascending { "↑" } else { "↓" };
            if ui.button(direction).clicked() {
                app.browse.sort_ascending = !app.browse.sort_ascending;
                app.browse.listing_stale = true;
            }
            let sort_names = [
                (SortKey::Name, "Name"),
                (SortKey::Kind, "Kind"),
                (SortKey::Size, "Size"),
                (SortKey::Modified, "Modified"),
                (SortKey::Duration, "Duration"),
            ];
            egui::ComboBox::from_id_salt("sort_key")
                .selected_text(sort_names.iter().find(|(key, _)| *key == app.browse.sort_key).map(|(_, name)| *name).unwrap_or("Name"))
                .width(100.0)
                .show_ui(ui, |ui| {
                    for (key, name) in sort_names {
                        ui.selectable_value(&mut app.browse.sort_key, key, name);
                    }
                });
            if ui.button("Large files").on_hover_text("> 100 MB").clicked() {
                app.browse.large_files_only = !app.browse.large_files_only;
                app.browse.listing_stale = true;
            }
            if ui.button("Recent").on_hover_text("modified in last 30 days").clicked() {
                app.browse.recent_files_only = !app.browse.recent_files_only;
                app.browse.listing_stale = true;
            }
        });
    });
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        for kind in FileKind::ALL {
            let selected = app.browse.kind_filter.contains(&kind);
            let label = kind.label();
            let chip = egui::Button::new(
                RichText::new(label).color(if selected { Color32::WHITE } else { theme::TEXT_DIM }),
            )
            .fill(if selected { theme::ACCENT_SOFT } else { theme::SURFACE_CARD });
            if ui.add(chip).clicked() {
                if selected {
                    app.browse.kind_filter.retain(|k| *k != kind);
                } else {
                    app.browse.kind_filter.push(kind);
                }
                app.browse.listing_stale = true;
            }
        }
        if !app.browse.kind_filter.is_empty() && ui.button("clear").clicked() {
            app.browse.kind_filter.clear();
            app.browse.listing_stale = true;
        }
    });
}

fn draw_table(ui: &mut Ui, app: &mut IpicApp) {
    let row_height = 30.0;
    let listing_count = app.browse.listing.len();
    let available_height = ui.available_height();
    let table = TableBuilder::new(ui)
        .striped(false)
        .cell_layout(egui::Layout::left_to_right(egui::Align::Center))
        .column(Column::remainder())
        .column(Column::exact(64.0))
        .column(Column::exact(78.0))
        .column(Column::exact(120.0))
        .column(Column::exact(64.0))
        .column(Column::exact(84.0))
        .min_scrolled_height(available_height);
    table
        .header(28.0, |mut header| {
            header.col(|ui| {
                sortable_header(ui, app, SortKey::Name, "Name");
            });
            header.col(|ui| {
                sortable_header(ui, app, SortKey::Kind, "Kind");
            });
            header.col(|ui| {
                sortable_header(ui, app, SortKey::Size, "Size");
            });
            header.col(|ui| {
                sortable_header(ui, app, SortKey::Modified, "Modified");
            });
            header.col(|ui| {
                sortable_header(ui, app, SortKey::Duration, "Length");
            });
            header.col(|_ui| {});
        })
        .body(|body| {
            body.rows(row_height, listing_count, |mut row| {
                let index = row.index();
                let Some(entry) = app.browse.listing.get(index).cloned() else { return };
                match &entry {
                    ListingEntry::Directory(directory) => {
                        let selected = false;
                        row.set_selected(selected);
                        row.col(|ui| {
                            ui.horizontal(|ui| {
                                ui.label(RichText::new("▸").color(theme::ACCENT));
                                if ui
                                    .add(egui::Button::new(
                                        RichText::new(&directory.name).color(theme::TEXT_PRIMARY).strong(),
                                    )
                                    .fill(Color32::TRANSPARENT))
                                    .clicked()
                                {
                                    app.navigate_to(Some(directory.clone()));
                                }
                                ui.label(
                                    RichText::new(format!(
                                        "{} items",
                                        directory.file_count + directory.subdir_count
                                    ))
                                    .color(theme::TEXT_DIM)
                                    .small(),
                                );
                            });
                        });
                        row.col(|ui| {
                            ui.label(RichText::new("Folder").color(theme::TEXT_DIM));
                        });
                        for _ in 0..4 {
                            row.col(|_ui| {});
                        }
                    }
                    ListingEntry::File(file) => {
                        let is_selected = app.selected_file.as_ref().map(|(selected, _)| selected.id) == Some(file.id);
                        row.set_selected(is_selected);
                        draw_file_row(&mut row, app, file);
                    }
                }
            });
        });
}

fn sortable_header(ui: &mut Ui, app: &mut IpicApp, key: SortKey, label: &str) {
    let active = app.browse.sort_key == key;
    let arrow = if active {
        if app.browse.sort_ascending { " ↑" } else { " ↓" }
    } else {
        ""
    };
    let text = RichText::new(format!("{label}{arrow}"))
        .color(if active { theme::TEXT_PRIMARY } else { theme::TEXT_DIM })
        .strong();
    if ui.add(egui::Button::new(text).fill(Color32::TRANSPARENT)).clicked() {
        if active {
            app.browse.sort_ascending = !app.browse.sort_ascending;
        } else {
            app.browse.sort_key = key;
            app.browse.sort_ascending = true;
        }
        app.browse.listing_stale = true;
    }
}

fn draw_file_row(row: &mut egui_extras::TableRow<'_, '_>, app: &mut IpicApp, file: &FileRow) {
    row.col(|ui| {
        ui.horizontal(|ui| {
            ui.label(kind_glyph(file.kind));
            ui.label(RichText::new(&file.name).color(theme::TEXT_PRIMARY));
        });
    });
    row.col(|ui| {
        ui.label(RichText::new(file.kind.label()).color(theme::TEXT_DIM).small());
    });
    row.col(|ui| {
        ui.label(RichText::new(ipic_core::util::format_size(file.size)).color(theme::TEXT_DIM).monospace());
    });
    row.col(|ui| {
        ui.label(
            RichText::new(ipic_core::util::format_local_timestamp(file.mtime))
                .color(theme::TEXT_DIM)
                .small(),
        );
    });
    row.col(|ui| {
        let text = file.duration_secs.map(ipic_core::util::format_duration).unwrap_or_default();
        ui.label(RichText::new(text).color(theme::TEXT_DIM).monospace());
    });
    row.col(|ui| {
        let (color, status) = match file.rag {
            RagStatus::Done => (theme::SUCCESS, "indexed"),
            RagStatus::Busy => (theme::WARNING, "working"),
            RagStatus::Failed => (theme::DANGER, "failed"),
            RagStatus::Pending => (theme::TEXT_DIM, "queued"),
        };
        ui.label(RichText::new(status).color(color).small());
    });
    // Selection + interactions use the row response captured via the last cell.
    let interact = row.response();
    let path = file_full_path(app, file);
    if interact.clicked() {
        app.selected_file = Some((file.clone(), path.clone()));
    }
    if interact.double_clicked() {
        crate::actions::open_file(std::path::Path::new(&path));
    }
    interact.context_menu(|ui| {
        if ui.button("Open").clicked() {
            crate::actions::open_file(std::path::Path::new(&path));
            ui.close();
        }
        if ui.button("Reveal in file manager").clicked() {
            crate::actions::reveal_in_file_manager(std::path::Path::new(&path));
            ui.close();
        }
        if ui.button("Copy path").clicked() {
            crate::actions::copy_path_to_clipboard(&path);
            ui.close();
        }
        if ui.button("Rename…").clicked() {
            app.browse.rename_dialog = Some(RenameDialog {
                file_id: file.id,
                current_name: file.name.clone(),
                edit_buffer: file.name.clone(),
            });
            ui.close();
        }
        if ui.button(egui::RichText::new("Move to Trash").color(theme::DANGER)).clicked() {
            if crate::actions::move_to_trash(std::path::Path::new(&path)).is_ok() {
                if let Ok(slots) = app.engine.catalog.remove_path(&path) {
                    app.engine.release_vector_slots(&slots);
                }
                app.selected_file = None;
                app.browse.listing_stale = true;
            }
            ui.close();
        }
    });
}

fn file_full_path(app: &IpicApp, file: &FileRow) -> String {
    let Ok(connection) = app.engine.catalog.reader() else { return file.name.clone() };
    app.engine
        .catalog
        .file_by_id(&connection, file.id)
        .ok()
        .flatten()
        .map(|(_, path)| path)
        .unwrap_or_else(|| file.name.clone())
}

pub fn kind_glyph(kind: FileKind) -> RichText {
    let glyph = match kind {
        FileKind::Text => "📝",
        FileKind::Pdf => "📕",
        FileKind::Audio => "🎧",
        FileKind::Video => "🎬",
        FileKind::Image => "🖼",
        FileKind::Other => "📄",
    };
    RichText::new(glyph)
}
