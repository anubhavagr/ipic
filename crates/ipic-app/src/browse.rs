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
    /// File row with its resolved path (no per-frame path lookups).
    File(FileRow, String),
}

pub struct RenameDialog {
    /// File the dialog acts on. Carried by the dialog itself: the row may be
    /// right-clicked without being the current selection.
    pub file: FileRow,
    pub path: String,
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
    /// Flat index of the focused row for keyboard navigation.
    pub selected_row: Option<usize>,
    /// All selected row indexes (⌘-click toggles, ⇧-click extends, ⌘A selects all).
    pub selected_rows: Vec<usize>,
    /// Anchor row for shift-range selection.
    pub selection_anchor: Option<usize>,
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
            selected_rows: Vec::new(),
            selection_anchor: None,
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

    /// Reloads the visible listing through the engine (SQL-side filter + sort).
    pub fn refresh_listing(app: &mut IpicApp) {
        let directory = app.current_directory.clone();
        let (directories, files) = app.engine.directory_children(
            directory.as_ref(),
            &app.browse.active_filter(),
            app.browse.sort_key,
            app.browse.sort_ascending,
            20_000,
        );
        let mut listing: Vec<ListingEntry> = directories.into_iter().map(ListingEntry::Directory).collect();
        listing.extend(files.into_iter().map(|(file, path)| ListingEntry::File(file, path)));
        if let Some(selected) = app.browse.selected_row {
            app.selected_file = listing.get(selected).and_then(|entry| match entry {
                ListingEntry::File(file, path) => Some((file.clone(), path.clone())),
                ListingEntry::Directory(_) => None,
            });
        }
        app.browse.listing = listing;
        app.browse.listing_dir = directory.as_ref().map(|dir| dir.id);
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
    ui.add_space(ui.available_height() * 0.28);
    ui.vertical_centered(|ui| {
        ui.label(
            egui::RichText::new(crate::theme::icons::FOLDER_OPEN)
                .font(egui::FontId::new(34.0, egui::FontFamily::Name("material-icons".into())))
                .color(theme::SURFACE_HOVER),
        );
        ui.add_space(8.0);
        ui.label(RichText::new("nothing here").size(17.0).color(theme::TEXT_DIM));
        let hint = if app.browse.active_filter().is_empty() {
            "this folder is empty, add files or create a new folder above"
        } else {
            "no files match the current filters"
        };
        ui.label(RichText::new(hint).color(theme::TEXT_DIM).small());
    });
}

fn draw_toolbar(ui: &mut Ui, app: &mut IpicApp) {
    // Two rows: on one row the right-aligned sort controls silently overlapped
    // the kind chips once the row overflowed, stealing each other's clicks.
    ui.horizontal(|ui| {
        match &app.browse.new_folder_buffer {
            None => {
                let icon_clicked = ui
                    .button(crate::theme::icon(crate::theme::icons::NEW_FOLDER, 16.0))
                    .on_hover_text("New folder in the open directory")
                    .clicked();
                let text_clicked = ui
                    .add(
                        egui::Button::new(
                            RichText::new("New Folder").color(theme::TEXT_PRIMARY).small(),
                        )
                        .fill(Color32::TRANSPARENT),
                    )
                    .clicked();
                if icon_clicked || text_clicked {
                    app.browse.new_folder_buffer = Some("New Folder".into());
                }
            }
            Some(buffered) => {
                let mut name = buffered.clone();
                let edit = egui::TextEdit::singleline(&mut name).desired_width(160.0).hint_text("folder name");
                let response = ui.add(edit);
                // Grab focus when the editor appears, but never steal it back
                // after Enter/Escape surrender it or the user clicks elsewhere.
                if !response.has_focus() && !response.lost_focus() {
                    response.request_focus();
                }
                let mut confirm = ui.button("Create").clicked();
                // Enter/Escape surrender focus mid-frame, so lost_focus counts
                // as "this edit had the keyboard" for that key press.
                if (response.has_focus() || response.lost_focus())
                    && ui.input(|input| input.key_pressed(egui::Key::Enter))
                {
                    confirm = true;
                }
                // Enter/Escape may surrender focus mid-frame, so both the
                // focused and just-unfocused states count as "had the keyboard".
                let escape_pressed = ui.input(|input| input.key_pressed(egui::Key::Escape));
                let cancel = ui.button("Cancel").clicked()
                    || escape_pressed && (response.has_focus() || response.lost_focus());
                if cancel {
                    app.browse.new_folder_buffer = None;
                } else if confirm {
                    let name = name.trim().to_string();
                    match &app.current_directory {
                        Some(directory) => {
                            match crate::actions::create_folder(Path::new(&directory.path), &name) {
                                Ok(created) => {
                                    app.browse.new_folder_buffer = None;
                                    app.browse.listing_stale = true;
                                    // A rescan is what registers the new folder
                                    // in the catalog (and thus the listing).
                                    app.engine.spawn_scan();
                                    app.push_notice(format!("created “{}”", created.file_name().map(|part| part.to_string_lossy().into_owned()).unwrap_or(name)));
                                }
                                Err(error) => app.push_notice(format!("folder failed: {error}")),
                            }
                        }
                        None => {
                            app.browse.new_folder_buffer = None;
                            app.push_notice("open a folder first, then create inside it".into());
                        }
                    }
                } else {
                    app.browse.new_folder_buffer = Some(name);
                }
            }
        }
        if ui
            .button(crate::theme::icon(crate::theme::icons::REFRESH, 16.0))
            .on_hover_text("Rescan roots now")
            .clicked()
        {
            app.engine.spawn_scan();
            app.push_notice("rescan started".into());
        }
        ui.separator();
        for kind in FileKind::ALL {
            let selected = app.browse.kind_filter.contains(&kind);
            let label = kind.label();
            let chip = egui::Button::new(
                RichText::new(label).color(if selected { theme::TEXT_PRIMARY } else { theme::TEXT_DIM }).strong(),
            )
            .fill(if selected { theme::ACCENT_SOFT } else { theme::SURFACE_CARD })
            .stroke(if selected {
                egui::Stroke::new(1.0, theme::ACCENT)
            } else {
                egui::Stroke::new(1.0, theme::BORDER)
            });
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
    ui.horizontal(|ui| {
        let toggle = |ui: &mut Ui, label: &str, active: bool, hover: &str| {
            let chip = egui::Button::new(
                RichText::new(label).color(if active { theme::TEXT_PRIMARY } else { theme::TEXT_DIM }).strong(),
            )
            .fill(if active { theme::ACCENT_SOFT } else { theme::SURFACE_CARD })
            .stroke(if active {
                egui::Stroke::new(1.0, theme::ACCENT)
            } else {
                egui::Stroke::new(1.0, theme::BORDER)
            });
            ui.add(chip).on_hover_text(hover).clicked()
        };
        if toggle(ui, "Large", app.browse.large_files_only, "larger than 100 MB") {
            app.browse.large_files_only = !app.browse.large_files_only;
            app.browse.listing_stale = true;
        }
        if toggle(ui, "Recent", app.browse.recent_files_only, "modified in the last 30 days") {
            app.browse.recent_files_only = !app.browse.recent_files_only;
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
        })
        .body(|body| {
            body.rows(row_height, listing_count, |mut row| {
                let index = row.index();
                let Some(entry) = app.browse.listing.get(index).cloned() else { return };
                let is_selected = app.browse.selected_row == Some(index);
                row.set_selected(is_selected);
                match &entry {
                    ListingEntry::Directory(directory) => {
                        let mut cell_response: Option<egui::Response> = None;
                        let extend = |response: egui::Response, acc: &mut Option<egui::Response>| {
                            *acc = Some(match acc.take() {
                                Some(accumulated) => accumulated | response,
                                None => response,
                            });
                        };
                        row.col(|ui| {
                            ui.horizontal(|ui| {
                                extend(
                                    clickable_label(
                                        ui,
                                        egui::RichText::new(crate::theme::icons::FOLDER)
                                            .font(egui::FontId::new(
                                                19.0,
                                                egui::FontFamily::Name("material-icons".into()),
                                            ))
                                            .color(theme::ACCENT),
                                    ),
                                    &mut cell_response,
                                );
                                extend(
                                    clickable_label(
                                        ui,
                                        RichText::new(&directory.name).color(theme::TEXT_PRIMARY).strong(),
                                    )
                                    .on_hover_cursor(egui::CursorIcon::PointingHand),
                                    &mut cell_response,
                                );
                                extend(
                                    clickable_label(
                                        ui,
                                        RichText::new(format!(
                                            "{} items",
                                            directory.file_count + directory.subdir_count
                                        ))
                                        .color(theme::TEXT_DIM)
                                        .small()
                                        .monospace(),
                                    ),
                                    &mut cell_response,
                                );
                            });
                        });
                        row.col(|ui| {
                            extend(
                                clickable_label(ui, RichText::new("Folder").color(theme::TEXT_DIM)),
                                &mut cell_response,
                            );
                        });
                        for _ in 0..3 {
                            row.col(|_ui| {});
                        }
                        let Some(interact) = cell_response else { return };
                        if interact.double_clicked() {
                            app.navigate_to(Some(directory.clone()));
                        }
                        interact.context_menu(|ui| {
                            ui.set_min_width(200.0);
                            if ui.button("Open").clicked() {
                                app.navigate_to(Some(directory.clone()));
                                ui.close();
                            }
                            if ui.button("Reveal in Finder").clicked() {
                                app.dispatch_reveal(Path::new(&directory.path));
                                ui.close();
                            }
                            if ui.button("Copy path").clicked() {
                                crate::actions::copy_path_to_clipboard(&directory.path);
                                app.push_notice("path copied".into());
                                ui.close();
                            }
                        });
                    }
                    ListingEntry::File(file, path) => {
                        draw_file_row(&mut row, app, file, path.clone(), index);
                    }
                }
            });
        });
}

fn draw_keyboard_navigation(ui: &mut Ui, app: &mut IpicApp) {
    // Enter is the confirm key of the rename dialog and the new-folder editor;
    // while either is open it must not also open the selected file.
    if app.search_box_has_focus
        || app.browse.rename_dialog.is_some()
        || app.browse.new_folder_buffer.is_some()
    {
        return;
    }
    let listing_len = app.browse.listing.len();
    if listing_len == 0 {
        return;
    }
    let (mut move_down, mut move_up, mut open, mut select_all) = (false, false, false, false);
    let command = ui.input(|input| input.modifiers.command);
    ui.input(|input| {
        move_down = input.key_pressed(egui::Key::ArrowDown);
        move_up = input.key_pressed(egui::Key::ArrowUp);
        open = input.key_pressed(egui::Key::Enter);
    });
    if command && ui.input(|input| input.key_pressed(egui::Key::A)) {
        select_all = true;
    }
    if select_all {
        app.browse.selected_rows = (0..listing_len).collect();
        return;
    }
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
                ListingEntry::File(file, path) => Some((file, path)),
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
    // Telemetry register: uppercase mono micro label; the active column is
    // the only one that lights up.
    let text = RichText::new(format!("{label}{arrow}"))
        .color(if active { theme::TEXT_PRIMARY } else { theme::TEXT_DIM })
        .strong()
        .small()
        .monospace();
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

fn ui_input_command(row: &egui_extras::TableRow<'_, '_>) -> bool {
    row.response().ctx.input(|input| input.modifiers.command)
}

fn row_shift_pressed(row: &egui_extras::TableRow<'_, '_>) -> bool {
    row.response().ctx.input(|input| input.modifiers.shift)
}

fn draw_file_row(
    row: &mut egui_extras::TableRow<'_, '_>,
    app: &mut IpicApp,
    file: &FileRow,
    path: String,
    index: usize,
) {
    // TableRow::col unions only the cell container (hover-sensed); row
    // interaction therefore hangs off the widgets' own responses.
    let mut cell_response: Option<egui::Response> = None;
    let extend = |response: egui::Response, acc: &mut Option<egui::Response>| {
        *acc = Some(match acc.take() {
            Some(accumulated) => accumulated | response,
            None => response,
        });
    };
    let row_id = egui::Id::new(("file-row", file.id));
    row.col(|ui| {
        ui.horizontal(|ui| {
            draw_status_dot(ui, file);
            extend(
                clickable_label_stable(ui, kind_glyph(file.kind), row_id.with("glyph")),
                &mut cell_response,
            );
            extend(
                clickable_label_stable(ui, file_name_job(&file.name), row_id.with("name")),
                &mut cell_response,
            );
        });
    });
    row.col(|ui| {
        extend(
            clickable_label_stable(
                ui,
                RichText::new(file.kind.label()).color(theme::TEXT_DIM).small(),
                row_id.with("kind"),
            ),
            &mut cell_response,
        );
    });
    row.col(|ui| {
        extend(
            clickable_label_stable(
                ui,
                RichText::new(ipic_core::util::format_size(file.size)).color(theme::TEXT_DIM).monospace(),
                row_id.with("size"),
            ),
            &mut cell_response,
        );
    });
    row.col(|ui| {
        extend(
            clickable_label_stable(
                ui,
                RichText::new(ipic_core::util::format_local_timestamp(file.mtime))
                    .color(theme::TEXT_DIM)
                    .small()
                    .monospace(),
                row_id.with("modified"),
            ),
            &mut cell_response,
        );
    });
    row.col(|ui| {
        let text = file.duration_secs.map(ipic_core::util::format_duration).unwrap_or_default();
        extend(
            clickable_label_stable(
                ui,
                RichText::new(text).color(theme::TEXT_DIM).monospace(),
                row_id.with("duration"),
            ),
            &mut cell_response,
        );
    });
    let Some(interact) = cell_response else { return };
    let command_pressed = ui_input_command(row);
    // A right-click opens the context menu without a prior left-click, so it
    // must select the row too (otherwise details/rename act on another file).
    if interact.secondary_clicked() {
        app.browse.selected_row = Some(index);
        app.selected_file = Some((file.clone(), path.clone()));
        if !app.browse.selected_rows.contains(&index) {
            app.browse.selected_rows = vec![index];
        }
    }
    if interact.clicked() {
        app.browse.selected_row = Some(index);
        app.selected_file = Some((file.clone(), path.clone()));
        if command_pressed {
            // Toggle membership, keep others.
            if let Some(position) = app.browse.selected_rows.iter().position(|row| *row == index) {
                app.browse.selected_rows.remove(position);
            } else {
                app.browse.selected_rows.push(index);
            }
            app.browse.selection_anchor = Some(index);
        } else if let Some(anchor) = app.browse.selection_anchor
            && row_shift_pressed(row)
        {
            let (low, high) = (anchor.min(index), anchor.max(index));
            app.browse.selected_rows = (low..=high).collect();
        } else {
            app.browse.selected_rows = vec![index];
            app.browse.selection_anchor = Some(index);
        }
    }
    if interact.double_clicked() {
        app.dispatch_open(Path::new(&path));
    }
    interact.context_menu(|ui| {
        file_context_menu(ui, app, file, &path);
    });
}

fn file_context_menu(ui: &mut Ui, app: &mut IpicApp, file: &FileRow, path: &str) {
    // Grouped instrument menu: navigation / clipboard / file ops / destructive.
    // Labels must stay exact (UI tests query them).
    ui.set_min_width(200.0);
    ui.set_min_height(4.0);
    if ui.button("Open").clicked() {
        app.dispatch_open(Path::new(path));
        ui.close();
    }
    if ui.button("Reveal in Finder").clicked() {
        app.dispatch_reveal(Path::new(path));
        ui.close();
    }
    if ui.button("Copy path").clicked() {
        crate::actions::copy_path_to_clipboard(path);
        app.push_notice("path copied".into());
        ui.close();
    }
    ui.separator();
    if ui.button("Duplicate").clicked() {
        match crate::actions::duplicate_file(Path::new(path)) {
            Ok(new_path) => {
                app.push_notice(format!(
                    "duplicated → {}",
                    new_path.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default()
                ));
                app.browse.listing_stale = true;
                // The copy exists only on disk; a rescan registers it in the
                // catalog so the listing picks it up.
                app.engine.spawn_scan();
            }
            Err(error) => app.push_notice(format!("duplicate failed: {error}")),
        }
        ui.close();
    }
    if ui.button("Rename…").clicked() {
        app.browse.rename_dialog = Some(RenameDialog {
            file: file.clone(),
            path: path.to_string(),
            edit_buffer: file.name.clone(),
        });
        ui.close();
    }
    ui.separator();
    if ui
        .button(egui::RichText::new("Move to Trash").color(theme::DANGER).strong())
        .clicked()
    {
        if crate::actions::move_to_trash(Path::new(path)).is_ok() {
            app.engine.remove_path(path);
            app.selected_file = None;
            app.browse.selected_row = None;
            app.browse.listing_stale = true;
            app.push_notice("moved to trash".into());
        }
        ui.close();
    }
}

/// Filename as a two-tone layout job: stem in primary, extension dimmed.
/// The concatenated text still equals the exact file name (UI tests query it).
fn file_name_job(name: &str) -> egui::WidgetText {
    let (stem, extension) = match name.rfind('.') {
        Some(index) if index > 0 => (&name[..index], &name[index..]),
        _ => (name, ""),
    };
    let mut job = egui::text::LayoutJob::default();
    job.append(
        stem,
        0.0,
        egui::TextFormat::simple(
            egui::FontId::new(13.5, egui::FontFamily::Proportional),
            theme::TEXT_PRIMARY,
        ),
    );
    if !extension.is_empty() {
        job.append(
            extension,
            0.0,
            egui::TextFormat::simple(
                egui::FontId::new(13.5, egui::FontFamily::Proportional),
                theme::TEXT_DIM,
            ),
        );
    }
    egui::WidgetText::LayoutJob(job.into())
}

/// Kind glyph from the embedded Material Icons font (consistent rendering,
/// no emoji fallback surprises across platforms).
pub fn kind_glyph(kind: FileKind) -> RichText {
    let glyph = match kind {
        FileKind::Text => theme::icons::DESCRIPTION,
        FileKind::Pdf => theme::icons::PICTURE_AS_PDF,
        FileKind::Audio => theme::icons::AUDIOTRACK,
        FileKind::Video => theme::icons::MOVIE,
        FileKind::Image => theme::icons::IMAGE,
        FileKind::Other => theme::icons::INSERT_DRIVE_FILE,
    };
    egui::RichText::new(glyph)
        .font(egui::FontId::new(
            19.0,
            egui::FontFamily::Name("material-icons".into()),
        ))
        .color(kind_glyph_color(kind))
}

/// Per-kind glyph tint. Pastel offsets from the saturated status hues so
/// file-type color never reads as index state (the status dot owns those):
/// slate for text, blush red for PDF, violet audio, rose video, mint image.
pub fn kind_glyph_color(kind: FileKind) -> egui::Color32 {
    match kind {
        FileKind::Text => egui::Color32::from_rgb(170, 180, 199),
        FileKind::Pdf => egui::Color32::from_rgb(250, 146, 146),
        FileKind::Audio => egui::Color32::from_rgb(182, 155, 255),
        FileKind::Video => egui::Color32::from_rgb(246, 140, 193),
        FileKind::Image => egui::Color32::from_rgb(118, 222, 174),
        FileKind::Other => theme::TEXT_DIM,
    }
}

/// Kind glyph codepoint (for composition into layout jobs).
pub fn kind_glyph_codepoint(kind: FileKind) -> &'static str {
    match kind {
        FileKind::Text => theme::icons::DESCRIPTION,
        FileKind::Pdf => theme::icons::PICTURE_AS_PDF,
        FileKind::Audio => theme::icons::AUDIOTRACK,
        FileKind::Video => theme::icons::MOVIE,
        FileKind::Image => theme::icons::IMAGE,
        FileKind::Other => theme::icons::INSERT_DRIVE_FILE,
    }
}

/// Index-status dot: green = indexed, red = failed, amber = working,
/// faint = queued; non-RAG kinds (archives, binaries) draw nothing.
fn draw_status_dot(ui: &mut Ui, file: &FileRow) {
    let color = match file.rag {
        RagStatus::Done => Some(theme::SUCCESS),
        RagStatus::Failed => Some(theme::DANGER),
        RagStatus::Busy => Some(theme::WARNING),
        RagStatus::Pending => file.kind.is_rag().then_some(theme::TEXT_DIM),
    };
    let Some(color) = color else { return };
    let (rect, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
    ui.painter().circle_filled(rect.center(), 3.0, color);
}

/// A label that participates in row clicking (plain labels ignore pointers).
fn clickable_label(ui: &mut Ui, contents: impl Into<egui::WidgetText>) -> egui::Response {
    ui.add(egui::Label::new(contents).sense(egui::Sense::click()))
}

/// Stable-id variant: keeps context menus and interaction state alive across
/// layout shifts (e.g. the details panel appearing when a row is selected).
fn clickable_label_stable(
    ui: &mut Ui,
    contents: impl Into<egui::WidgetText>,
    stable_id: egui::Id,
) -> egui::Response {
    ui.push_id(stable_id, |ui| ui.add(egui::Label::new(contents).sense(egui::Sense::click())))
        .inner
}
