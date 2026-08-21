//! Browse view: toolbar (kind filters, sorting, new folder, refresh) plus a
//! virtualized sortable table. The top search bar drives name filtering.

use crate::app::IpicApp;
use crate::theme;
use egui::{Color32, RichText, Ui};
use egui_extras::{Column, TableBuilder};
use ipic_core::DirRow;
use ipic_core::{FileFilter, FileKind, FileRow, RagStatus, SortKey};
use std::path::Path;

#[derive(Debug, Clone)]
pub enum ListingEntry {
    Directory(DirRow),
    File(FileRow),
}

pub struct RenameDialog {
    pub file_id: i64,
    pub edit_buffer: String,
}

pub struct BrowsePanel {
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
    /// Flat index of the selected row for keyboard navigation.
    pub selected_row: Option<usize>,
    pub new_folder_buffer: Option<String>,
}

impl Default for BrowsePanel {
    fn default() -> Self {
        Self {
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
            selected_row: None,
            new_folder_buffer: None,
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
                &app.browse.active_filter(),
                app.browse.sort_key,
                app.browse.sort_ascending,
                20_000,
            )
            .unwrap_or_default();
        let directories = match directory_id {
            None => Vec::new(),
            Some(directory_id) => app
                .engine
                .catalog
                .tree_children(&connection, Some(directory_id))
                .unwrap_or_default(),
        };
        let mut listing: Vec<ListingEntry> = directories.into_iter().map(ListingEntry::Directory).collect();
        listing.extend(files.into_iter().map(ListingEntry::File));
        if let Some(selected) = app.browse.selected_row {
            app.selected_file = listing.get(selected).and_then(|entry| match entry {
                ListingEntry::File(file) => Some((file.clone(), full_path(app, file))),
                ListingEntry::Directory(_) => None,
            });
        }
        app.browse.listing = listing;
        app.browse.listing_dir = directory_id;
        app.browse.listing_stale = false;
    }
}

pub fn draw(ui: &mut Ui, app: &mut IpicApp) {
    draw_toolbar(ui, app);
    ui.add_space(2.0);
    if app.browse.listing.is_empty() {
        draw_empty_state(ui, app);
        return;
    }
    draw_table(ui, app);
    draw_keyboard_navigation(ui, app);
}

fn draw_empty_state(ui: &mut Ui, app: &mut IpicApp) {
    ui.add_space(ui.available_height() * 0.3);
    ui.vertical_centered(|ui| {
        ui.label(RichText::new("nothing here").size(18.0).color(theme::TEXT_DIM));
        let hint = if app.browse.name_filter_active.is_empty() {
            "this folder is empty — add files or create a new folder above"
        } else {
            "no files match the current filters"
        };
        ui.label(RichText::new(hint).color(theme::TEXT_DIM).small());
    });
}

fn draw_toolbar(ui: &mut Ui, app: &mut IpicApp) {
    ui.horizontal(|ui| {
        match &app.browse.new_folder_buffer {
            None => {
                if ui.button("＋ New Folder").clicked() {
                    app.browse.new_folder_buffer = Some("New Folder".into());
                }
            }
            Some(buffered) => {
                let mut name = buffered.clone();
                let edit = egui::TextEdit::singleline(&mut name).desired_width(160.0).hint_text("folder name");
                let response = ui.add(edit);
                response.request_focus();
                let mut confirm = ui.button("Create").clicked();
                if response.has_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter)) {
                    confirm = true;
                }
                let cancel = ui.button("✕").clicked()
                    || (response.has_focus() && ui.input(|input| input.key_pressed(egui::Key::Escape)));
                if confirm && !name.trim().is_empty() {
                    match &app.current_directory {
                        Some(directory) => {
                            if crate::actions::create_folder(Path::new(&directory.path), &name).is_ok() {
                                app.browse.listing_stale = true;
                                app.push_notice(format!("created “{name}”"));
                            }
                        }
                        None => app.push_notice("open a folder first, then create inside it".into()),
                    }
                } else if !confirm && !cancel {
                    app.browse.new_folder_buffer = Some(name);
                }
            }
        }
        if ui.button("⟳").on_hover_text("Rescan roots now").clicked() {
            app.engine.spawn_scan();
            app.push_notice("rescan started".into());
        }
        ui.separator();
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
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let direction = if app.browse.sort_ascending { "↑" } else { "↓" };
            if ui.button(direction).on_hover_text("Sort direction").clicked() {
                app.browse.sort_ascending = !app.browse.sort_ascending;
                app.browse.listing_stale = true;
            }
            let sort_names = [
                (SortKey::Name, "Name"),
                (SortKey::Kind, "Kind"),
                (SortKey::Size, "Size"),
                (SortKey::Modified, "Modified"),
                (SortKey::Duration, "Length"),
            ];
            let selected_name = sort_names
                .iter()
                .find(|(key, _)| *key == app.browse.sort_key)
                .map(|(_, name)| *name)
                .unwrap_or("Name");
            egui::ComboBox::from_id_salt("sort_key")
                .selected_text(format!("Sort: {selected_name}"))
                .width(140.0)
                .show_ui(ui, |ui| {
                    for (key, name) in sort_names {
                        if ui.selectable_label(key == app.browse.sort_key, name).clicked() {
                            app.browse.sort_key = key;
                            app.browse.listing_stale = true;
                        }
                    }
                });
            if ui
                .button(if app.browse.large_files_only { "Large ✓" } else { "Large" })
                .on_hover_text("larger than 100 MB")
                .clicked()
            {
                app.browse.large_files_only = !app.browse.large_files_only;
                app.browse.listing_stale = true;
            }
            if ui
                .button(if app.browse.recent_files_only { "Recent ✓" } else { "Recent" })
                .on_hover_text("modified in the last 30 days")
                .clicked()
            {
                app.browse.recent_files_only = !app.browse.recent_files_only;
                app.browse.listing_stale = true;
            }
        });
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
                let is_selected = app.browse.selected_row == Some(index);
                row.set_selected(is_selected);
                match &entry {
                    ListingEntry::Directory(directory) => {
                        let mut opened = false;
                        row.col(|ui| {
                            ui.horizontal(|ui| {
                                ui.label(RichText::new("▸").color(theme::ACCENT));
                                if ui
                                    .add(
                                        egui::Button::new(
                                            RichText::new(&directory.name).color(theme::TEXT_PRIMARY).strong(),
                                        )
                                        .fill(Color32::TRANSPARENT),
                                    )
                                    .clicked()
                                {
                                    opened = true;
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
                        if opened || row.response().double_clicked() {
                            app.navigate_to(Some(directory.clone()));
                        }
                    }
                    ListingEntry::File(file) => {
                        draw_file_row(&mut row, app, file, index);
                    }
                }
            });
        });
}

fn draw_keyboard_navigation(ui: &mut Ui, app: &mut IpicApp) {
    if app.search_box_has_focus {
        return;
    }
    let listing_len = app.browse.listing.len();
    if listing_len == 0 {
        return;
    }
    let (mut move_down, mut move_up, mut open) = (false, false, false);
    ui.input(|input| {
        move_down = input.key_pressed(egui::Key::ArrowDown);
        move_up = input.key_pressed(egui::Key::ArrowUp);
        open = input.key_pressed(egui::Key::Enter);
    });
    if move_down || move_up {
        let next = match app.browse.selected_row {
            None => 0,
            Some(current) => {
                if move_down {
                    (current + 1).min(listing_len - 1)
                } else {
                    current.saturating_sub(1)
                }
            }
        };
        app.browse.selected_row = Some(next);
        if let Some(entry) = app.browse.listing.get(next).cloned() {
            app.selected_file = match entry {
                ListingEntry::File(file) => Some((file.clone(), full_path(app, &file))),
                ListingEntry::Directory(_) => None,
            };
        }
    }
    if open
        && let Some(index) = app.browse.selected_row {
            if let Some(ListingEntry::Directory(directory)) = app.browse.listing.get(index).cloned() {
                app.navigate_to(Some(directory));
            } else {
                app.open_selected();
            }
        }
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

fn draw_file_row(row: &mut egui_extras::TableRow<'_, '_>, app: &mut IpicApp, file: &FileRow, index: usize) {
    let path = full_path(app, file);
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
    // Whole-row interactions via the union of cell responses.
    let interact = row.response();
    if interact.clicked() {
        app.browse.selected_row = Some(index);
        app.selected_file = Some((file.clone(), path.clone()));
    }
    if interact.double_clicked() {
        crate::actions::open_file(Path::new(&path));
    }
    interact.context_menu(|ui| {
        file_context_menu(ui, app, file, &path);
    });
}

fn file_context_menu(ui: &mut Ui, app: &mut IpicApp, file: &FileRow, path: &str) {
    if ui.button("Open").clicked() {
        crate::actions::open_file(Path::new(path));
        ui.close();
    }
    if ui.button("Reveal in Finder").clicked() {
        crate::actions::reveal_in_file_manager(Path::new(path));
        ui.close();
    }
    if ui.button("Copy path").clicked() {
        crate::actions::copy_path_to_clipboard(path);
        app.push_notice("path copied".into());
        ui.close();
    }
    if ui.button("Duplicate").clicked() {
        match crate::actions::duplicate_file(Path::new(path)) {
            Ok(new_path) => {
                app.push_notice(format!(
                    "duplicated → {}",
                    new_path.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default()
                ));
                app.browse.listing_stale = true;
            }
            Err(error) => app.push_notice(format!("duplicate failed: {error}")),
        }
        ui.close();
    }
    if ui.button("Rename…").clicked() {
        app.browse.rename_dialog = Some(RenameDialog {
            file_id: file.id,
            edit_buffer: file.name.clone(),
        });
        ui.close();
    }
    if ui
        .button(egui::RichText::new("Move to Trash").color(theme::DANGER))
        .clicked()
    {
        if crate::actions::move_to_trash(Path::new(path)).is_ok() {
            if let Ok(slots) = app.engine.catalog.remove_path(path) {
                app.engine.release_vector_slots(&slots);
            }
            app.selected_file = None;
            app.browse.selected_row = None;
            app.browse.listing_stale = true;
        }
        ui.close();
    }
}

pub fn full_path(app: &IpicApp, file: &FileRow) -> String {
    let Ok(connection) = app.engine.catalog.reader() else {
        return file.name.clone();
    };
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
