//! Application state and top-level layout. The GUI is a thin shell over the
//! engine (swap-friendly by design): all data flows through catalog readers
//! and engine events, never direct filesystem access.

use crate::ask::AskPanel;
use crate::browse::BrowsePanel;
use crate::recorder::AudioRecorder;
use egui::Context;
use ipic_core::{Config, DirRow, FileRow};
use ipic_rag::{Engine, EngineEvent, SearchOutcome};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub struct IpicApp {
    pub engine: Arc<Engine>,
    pub config: Config,
    pub browse: BrowsePanel,
    pub ask: AskPanel,
    pub recorder: Option<AudioRecorder>,
    /// Directory currently browsed; None = whole library.
    pub current_directory: Option<DirRow>,
    pub back_history: Vec<Option<DirRow>>,
    pub forward_history: Vec<Option<DirRow>>,
    pub expanded_tree_nodes: HashSet<i64>,
    pub notices: Vec<(String, Instant)>,
    pub searching: bool,
    pub search_receiver: std::sync::mpsc::Receiver<SearchOutcome>,
    pub search_sender: std::sync::mpsc::Sender<SearchOutcome>,
    pub selected_file: Option<(FileRow, String)>,
    pub show_details: bool,
    pub show_settings: bool,
    pub last_status_refresh: Instant,
}

impl IpicApp {
    pub fn new(engine: Arc<Engine>, config: Config) -> Self {
        let (search_sender, search_receiver) = std::sync::mpsc::channel();
        Self {
            browse: BrowsePanel::default(),
            ask: AskPanel::default(),
            engine,
            config,
            recorder: None,
            current_directory: None,
            back_history: Vec::new(),
            forward_history: Vec::new(),
            expanded_tree_nodes: HashSet::new(),
            notices: Vec::new(),
            searching: false,
            search_receiver,
            search_sender,
            selected_file: None,
            show_details: true,
            show_settings: false,
            last_status_refresh: Instant::now(),
        }
    }

    /// Consumes pending engine events; returns true when views must refresh.
    pub fn drain_engine_events(&mut self) -> bool {
        let mut refresh = false;
        while let Ok(event) = self.engine.events.try_recv() {
            match event {
                EngineEvent::ScanFinished { files_seen, directories_seen, elapsed_seconds } => {
                    self.push_notice(format!(
                        "scan finished: {files_seen} files in {directories_seen} directories ({elapsed_seconds:.1}s)"
                    ));
                    refresh = true;
                }
                EngineEvent::ScanStarted { roots } => {
                    self.push_notice(format!("scanning {}…", roots.first().map(|r| r.display().to_string()).unwrap_or_default()));
                }
                EngineEvent::ModelDownload { model, downloaded_bytes, total_bytes, finished } => {
                    if !finished {
                        if let Some(total) = total_bytes {
                            self.push_notice(format!("{model}: {}/{} MB", downloaded_bytes / 1_000_000, total / 1_000_000));
                        }
                    }
                }
                EngineEvent::TranscriberReady { model } => self.push_notice(format!("{model} ready")),
                EngineEvent::TranscriberFailed { reason } => self.push_notice(format!("whisper unavailable: {reason}")),
                EngineEvent::EmbedderReady { model_id, neural } => {
                    let kind = if neural { "neural" } else { "lexical fallback" };
                    self.push_notice(format!("embedder ready: {model_id} ({kind})"));
                }
                EngineEvent::Notice(message) => self.push_notice(message),
                EngineEvent::IndexingIdle => {
                    self.push_notice("index up to date".into());
                    refresh = true;
                }
                EngineEvent::CatalogChanged => refresh = true,
                EngineEvent::ScanProgress { .. } | EngineEvent::IndexProgress { .. } => {}
            }
        }
        refresh
    }

    pub fn push_notice(&mut self, message: String) {
        self.notices.push((message, Instant::now()));
        if self.notices.len() > 4 {
            self.notices.remove(0);
        }
    }

    pub fn navigate_to(&mut self, directory: Option<DirRow>) {
        if directory.as_ref().map(|d| d.id) == self.current_directory.as_ref().map(|d| d.id) {
            return;
        }
        self.back_history.push(self.current_directory.take());
        self.forward_history.clear();
        self.current_directory = directory;
        self.browse.listing_stale = true;
    }

    pub fn navigate_back(&mut self) {
        if let Some(previous) = self.back_history.pop() {
            self.forward_history.push(self.current_directory.take());
            self.current_directory = previous;
            self.browse.listing_stale = true;
        }
    }

    pub fn navigate_forward(&mut self) {
        if let Some(next) = self.forward_history.pop() {
            self.back_history.push(self.current_directory.take());
            self.current_directory = next;
            self.browse.listing_stale = true;
        }
    }

    pub fn navigate_up(&mut self) {
        let parent_path = self.current_directory.as_ref().and_then(|dir| {
            std::path::Path::new(&dir.path).parent().map(|p| p.to_path_buf())
        });
        if let Some(parent_path) = parent_path {
            if let Ok(connection) = self.engine.catalog.reader() {
                if let Some(parent) = self
                    .engine
                    .catalog
                    .dir_by_path(&connection, &parent_path.to_string_lossy())
                    .ok()
                    .flatten()
                {
                    if self
                        .config
                        .roots
                        .iter()
                        .any(|root| parent.path.starts_with(root.to_string_lossy().as_ref()))
                    {
                        self.navigate_to(Some(parent));
                        return;
                    }
                }
            }
        }
        self.navigate_to(None);
    }

    /// Runs a semantic search on a worker thread; results arrive via channel.
    pub fn start_search(&mut self, query: String) {
        if self.searching || query.trim().is_empty() {
            return;
        }
        self.searching = true;
        self.ask.results = None;
        let engine = Arc::clone(&self.engine);
        let sender = self.search_sender.clone();
        std::thread::spawn(move || {
            let outcome = engine.semantic_search(&query, 50).unwrap_or(SearchOutcome {
                hits: Vec::new(),
                elapsed_millis: 0.0,
                interpreted_query: None,
                vector_count: 0,
            });
            let _ = sender.send(outcome);
        });
    }

    pub fn poll_search(&mut self, context: &Context) {
        if self.searching {
            if let Ok(outcome) = self.search_receiver.try_recv() {
                self.ask.results = Some(outcome);
                self.searching = false;
                context.request_repaint();
            }
        }
    }

    pub fn toggle_recording(&mut self) {
        if self.recorder.is_none() {
            match AudioRecorder::start() {
                Ok(recorder) => {
                    self.recorder = Some(recorder);
                    self.push_notice("listening…".into());
                }
                Err(error) => self.push_notice(format!("microphone unavailable: {error}")),
            }
            return;
        }
        if let Some(recorder) = self.recorder.take() {
            let samples = recorder.stop();
            self.push_notice(format!("captured {:.1}s of audio", samples.len() as f32 / 16_000.0));
            self.searching = true;
            self.ask.results = None;
            let engine = Arc::clone(&self.engine);
            let sender = self.search_sender.clone();
            std::thread::spawn(move || {
                let outcome = engine.spoken_query_search(&samples, 50).unwrap_or(SearchOutcome {
                    hits: Vec::new(),
                    elapsed_millis: 0.0,
                    interpreted_query: None,
                    vector_count: 0,
                });
                let _ = sender.send(outcome);
            });
        }
    }

    /// Prunes stale notices (shown for 5 seconds).
    pub fn expire_notices(&mut self) {
        self.notices.retain(|(_, at)| at.elapsed() < Duration::from_secs(5));
    }
}

pub fn draw_top_bar(ui: &mut egui::Ui, app: &mut IpicApp) {
    egui::Panel::top("top_bar")
        .frame(egui::Frame::new().inner_margin(egui::Margin::symmetric(14, 8)))
        .show(ui, |ui| {
        ui.horizontal_centered(|ui| {
            ui.label(egui::RichText::new("◆ ipic").color(crate::theme::ACCENT).size(19.0).strong());
            ui.add_space(12.0);
            let back_enabled = !app.back_history.is_empty();
            if ui
                .add_enabled(back_enabled, egui::Button::new("‹").min_size(egui::vec2(26.0, 24.0)))
                .on_hover_text("Back")
                .clicked()
            {
                app.navigate_back();
            }
            if ui
                .add_enabled(!app.forward_history.is_empty(), egui::Button::new("›").min_size(egui::vec2(26.0, 24.0)))
                .on_hover_text("Forward")
                .clicked()
            {
                app.navigate_forward();
            }
            if ui.button("↑").on_hover_text("Parent directory").clicked() {
                app.navigate_up();
            }
            ui.add_space(6.0);
            draw_breadcrumb(ui, app);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("⚙").clicked() {
                    app.show_settings = !app.show_settings;
                }
                let details_label = if app.show_details { "◧ Details" } else { "◧" };
                if ui.button(details_label).clicked() {
                    app.show_details = !app.show_details;
                }
                ui.separator();
                let (browse_active, ask_active) = (app.browse.active, !app.browse.active);
                let browse = ui.selectable_label(browse_active, egui::RichText::new("Browse").strong());
                let ask = ui.selectable_label(ask_active, egui::RichText::new("Ask").strong());
                if browse.clicked() {
                    app.browse.active = true;
                }
                if ask.clicked() {
                    app.browse.active = false;
                }
            });
        });
    });
}

fn draw_breadcrumb(ui: &mut egui::Ui, app: &mut IpicApp) {
    ui.style_mut().spacing.item_spacing.x = 4.0;
    let roots: Vec<String> = app.config.roots.iter().map(|r| r.to_string_lossy().into_owned()).collect();
    let current_path = app.current_directory.as_ref().map(|dir| dir.path.clone());
    match current_path {
        None => {
            ui.label(egui::RichText::new("All files").color(crate::theme::TEXT_DIM));
        }
        Some(path_text) => {
            if ui.link("All files").clicked() {
                app.navigate_to(None);
            }
            let mut accumulated = String::new();
            for segment in path_text.split('/') {
                if segment.is_empty() {
                    continue;
                }
                accumulated.push('/');
                accumulated.push_str(segment);
                ui.label(egui::RichText::new("›").color(crate::theme::TEXT_DIM));
                let target = accumulated.clone();
                let is_current = target == path_text;
                let label = if is_current {
                    egui::RichText::new(segment).color(crate::theme::TEXT_PRIMARY).strong()
                } else {
                    egui::RichText::new(segment).color(crate::theme::TEXT_DIM)
                };
                if ui.add(egui::Button::new(label).fill(egui::Color32::TRANSPARENT)).clicked() {
                    if let Ok(connection) = app.engine.catalog.reader() {
                        if let Some(target_dir) = app
                            .engine
                            .catalog
                            .dir_by_path(&connection, &target)
                            .ok()
                            .flatten()
                        {
                            app.navigate_to(Some(target_dir));
                        }
                    }
                }
            }
            let _ = roots;
        }
    }
}

pub fn draw_status_bar(ui: &mut egui::Ui, app: &mut IpicApp) {
    egui::Panel::bottom("status_bar")
        .frame(
            egui::Frame::new()
                .fill(crate::theme::SURFACE_PANEL)
                .inner_margin(egui::Margin::symmetric(14, 5)),
        )
        .show(ui, |ui| {
            ui.horizontal_centered(|ui| {
                let status = app.engine.status();
                if status.scanning {
                    ui.label(egui::RichText::new("● scanning").color(crate::theme::ACCENT));
                } else if status.pending > 0 {
                    ui.label(
                        egui::RichText::new(format!("● indexing {} left", status.pending))
                            .color(crate::theme::WARNING),
                    );
                } else {
                    ui.label(egui::RichText::new("● ready").color(crate::theme::SUCCESS));
                }
                ui.separator();
                ui.label(egui::RichText::new(format!("{} files", status.total_files)).color(crate::theme::TEXT_DIM));
                ui.label(egui::RichText::new(format!("{} vectors", status.vector_count)).color(crate::theme::TEXT_DIM));
                ui.label(
                    egui::RichText::new(format!("{}× cores", status.core_count)).color(crate::theme::TEXT_DIM),
                );
                if let Some(outcome) = &app.ask.results {
                    ui.label(
                        egui::RichText::new(format!("search: {:.0} ms", outcome.elapsed_millis))
                            .color(crate::theme::TEXT_DIM),
                    );
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    for (message, _) in app.notices.iter().rev() {
                        ui.label(egui::RichText::new(message.clone()).color(crate::theme::TEXT_DIM));
                    }
                });
            });
        });
}
