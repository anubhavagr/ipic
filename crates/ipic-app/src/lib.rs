//! ipic — multimodal file manager with on-device semantic search.

pub mod actions;
pub mod app;
pub mod browse;
pub mod details;
pub mod recorder;
pub mod search_results;
pub mod sidebar;
pub mod theme;

pub use app::IpicApp;

use egui::Context;
use std::time::{Duration, Instant};

pub fn launch_gui() -> anyhow::Result<()> {
    let config = ipic_core::Config::load_or_create()?;
    let engine = ipic_rag::Engine::launch(config.clone())?;
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([900.0, 560.0])
            .with_title("ipic"),
        ..Default::default()
    };
    eframe::run_native(
        "ipic",
        native_options,
        Box::new(move |creation_context| {
            theme::apply(&creation_context.egui_ctx);
            egui_extras::install_image_loaders(&creation_context.egui_ctx);
            Ok(Box::new(IpicApp::new(engine, config)))
        }),
    )
    .map_err(|error| anyhow::anyhow!("gui run failed: {error}"))
}

impl eframe::App for IpicApp {
    /// State preparation, no painting (eframe 0.35 logic/ui split).
    fn logic(&mut self, context: &Context, _frame: &mut eframe::Frame) {
        let refreshed = self.drain_engine_events();
        if refreshed {
            self.browse.listing_stale = true;
        }
        self.poll_search(context);
        self.expire_notices();
        self.handle_shortcuts(context);

        // Refresh listing when stale, on navigation, or once per second
        // (index statuses advance in the background).
        let directory_changed =
            self.browse.listing_dir != self.current_directory.as_ref().map(|dir| dir.id);
        if self.browse.listing_stale
            || directory_changed
            || self.last_status_refresh.elapsed() > Duration::from_secs(1)
        {
            browse::BrowsePanel::refresh_listing(self);
            self.last_status_refresh = Instant::now();
        }

        // Keep the UI live while background work or recording runs.
        if self.searching || self.recorder.is_some() || self.engine.is_scanning() {
            context.request_repaint_after(Duration::from_millis(200));
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        app::draw_top_bar(ui, self);
        app::draw_status_bar(ui, self);
        egui::Panel::left("sidebar")
            .resizable(true)
            .default_size(250.0)
            .frame(egui::Frame::new().fill(theme::SURFACE_PANEL).inner_margin(egui::Margin::same(10)))
            .show(ui, |ui| sidebar::draw(ui, self));
        if self.show_details {
            egui::Panel::right("details")
                .resizable(true)
                .default_size(300.0)
                .frame(egui::Frame::new().fill(theme::SURFACE_PANEL).inner_margin(egui::Margin::same(10)))
                .show(ui, |ui| details::draw(ui, self));
        }
        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(theme::SURFACE_BASE).inner_margin(egui::Margin::same(14)))
            .show(ui, |ui| {
                if self.active_query.is_empty() {
                    browse::draw(ui, self);
                } else {
                    search_results::draw(ui, self);
                }
                draw_rename_dialog(ui, self);
            });

        if self.show_settings {
            draw_settings_window(ui.ctx(), self);
        }
    }
}

fn draw_rename_dialog(ui: &mut egui::Ui, app: &mut IpicApp) {
    let Some(dialog) = &mut app.browse.rename_dialog else { return };
    let mut apply = false;
    let mut cancel = false;
    egui::Window::new("Rename")
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .fixed_size([360.0, 0.0])
        .show(ui.ctx(), |ui| {
            ui.label("New name:");
            let response = ui.add(egui::TextEdit::singleline(&mut dialog.edit_buffer).clip_text(true));
            if response.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter)) {
                apply = true;
            }
            ui.horizontal(|ui| {
                if ui.button("Rename").clicked() {
                    apply = true;
                }
                if ui.button("Cancel").clicked() {
                    cancel = true;
                }
            });
        });
    if apply || cancel {
        let dialog = app.browse.rename_dialog.take().unwrap();
        if apply
            && let Some((file, path)) = &app.selected_file
                && file.id == dialog.file_id {
                    let new_path = actions::rename_file(std::path::Path::new(path), &dialog.edit_buffer);
                    if let Ok(new_path) = new_path {
                        // Update the catalog row so the UI reflects it instantly.
                        let new_name = new_path
                            .file_name()
                            .map(|name| name.to_string_lossy().into_owned())
                            .unwrap_or_default();
                        app.selected_file = Some((ipic_core::FileRow { name: new_name, ..file.clone() }, new_path.to_string_lossy().into_owned()));
                        app.browse.listing_stale = true;
                    }
                }
    }
}

fn draw_settings_window(context: &Context, app: &mut IpicApp) {
    egui::Window::new("Settings")
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
 .fixed_size([430.0, 0.0])
        .collapsible(false)
        .show(context, |ui| {
            ui.label(egui::RichText::new("Indexed roots").strong());
            let mut roots = app.config.roots.clone();
            let mut roots_changed = false;
            for index in (0..roots.len()).rev() {
                ui.horizontal(|ui| {
                    ui.monospace(roots[index].to_string_lossy().into_owned());
                    if ui.small_button("−").clicked() {
                        roots.remove(index);
                        roots_changed = true;
                    }
                });
            }
            let mut new_root = String::new();
            ui.horizontal(|ui| {
                let edit = egui::TextEdit::singleline(&mut new_root).hint_text("/absolute/path").desired_width(280.0);
                let response = ui.add(edit);
                let mut add = ui.button("+ Add").clicked();
                if response.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter)) {
                    add = true;
                }
                if add && !new_root.is_empty()
                    && let Ok(canonical) = std::fs::canonicalize(&new_root)
                        && !roots.contains(&canonical) {
                            roots.push(canonical);
                            roots_changed = true;
                        }
            });
            if roots_changed {
                app.config.roots = roots;
                let _ = app.config.save();
                app.push_notice("roots updated — rescan starts now".into());
                app.engine.spawn_scan();
            }
            ui.add_space(8.0);
            ui.label(egui::RichText::new("Whisper model").strong());
            let models = ipic_rag::transcribe::WHISPER_MODELS.to_vec();
            egui::ComboBox::from_id_salt("whisper_model")
                .selected_text(&app.config.whisper_model)
                .width(220.0)
                .show_ui(ui, |ui| {
                    for model in models {
                        ui.selectable_value(&mut app.config.whisper_model, model.to_string(), model);
                    }
                });
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label("Whisper workers");
                let mut workers = app.config.whisper_workers;
                ui.add(egui::Slider::new(&mut workers, 1..=8));
                if workers != app.config.whisper_workers {
                    app.config.whisper_workers = workers;
                    let _ = app.config.save();
                }
            });
            ui.add_space(10.0);
            ui.horizontal(|ui| {
                if ui.button("Rescan now").clicked() {
                    app.engine.spawn_scan();
                    app.show_settings = false;
                }
                if ui.button("Close").clicked() {
                    app.show_settings = false;
                }
            });
            let status = app.engine.status();
            ui.add_space(6.0);
            ui.label(
                egui::RichText::new(format!(
                    "embedder: {}  ·  vectors: {}  ·  indexed: {} files",
                    status.embedder_model, status.vector_count, status.done
                ))
                .small()
                .color(theme::TEXT_DIM),
            );
        });
}
