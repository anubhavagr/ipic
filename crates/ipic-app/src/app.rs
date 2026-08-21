//! Application state and top-level layout. One unified search bar drives both
//! name filtering (while browsing) and full multimodal semantic search (Enter).

use crate::browse::BrowsePanel;
use crate::recorder::AudioRecorder;
use crate::search_results::ResultsPanel;
use egui::Ui;
use ipic_core::{Config, DirRow, FileRow};
use ipic_rag::{Engine, EngineEvent, SearchOutcome};
use std::collections::HashSet;

/// Injectable file-open action (tests capture requests instead of spawning).
pub type OpenHandler = std::sync::Arc<dyn Fn(&std::path::Path) + Send + Sync>;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub struct IpicApp {
    pub engine: Arc<Engine>,
    pub config: Config,
    pub browse: BrowsePanel,
    pub results: ResultsPanel,
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
    /// Search box contents; drives name filtering and semantic search.
    pub search_edit: String,
    /// Injection point for tests: captures open requests instead of spawning.
    pub open_handler: Option<OpenHandler>,
    /// Whether the search box currently holds keyboard focus.
    pub search_box_has_focus: bool,
    /// Active unified-search query (empty = browsing mode).
    pub active_query: String,
}

impl IpicApp {
    pub fn new(engine: Arc<Engine>, config: Config) -> Self {
        let (search_sender, search_receiver) = std::sync::mpsc::channel();
        Self {
            browse: BrowsePanel::default(),
            results: ResultsPanel::default(),
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
            search_edit: String::new(),
            open_handler: None,
            search_box_has_focus: false,
            active_query: String::new(),
        }
    }

    /// Consumes pending engine events; returns true when views must refresh.
    pub fn drain_engine_events(&mut self) -> bool {
        let mut refresh = false;
        while let Ok(event) = self.engine.events.try_recv() {
            match event {
                EngineEvent::ScanFinished { files_seen, directories_seen, elapsed_seconds } => {
                    self.push_notice(format!(
                        "scan: {files_seen} files in {directories_seen} dirs ({elapsed_seconds:.1}s)"
                    ));
                    refresh = true;
                }
                EngineEvent::TranscriberReady { model } => self.push_notice(format!("{model} ready")),
                EngineEvent::TranscriberFailed { reason } => {
                    self.push_notice(format!("whisper unavailable: {reason}"))
                }
                EngineEvent::EmbedderReady { model_id, neural } => {
                    let kind = if neural { "neural" } else { "lexical fallback" };
                    self.push_notice(format!("embedder: {model_id} ({kind})"));
                }
                EngineEvent::Notice(message) => self.push_notice(message),
                EngineEvent::IndexingIdle => {
                    self.push_notice("index up to date".into());
                    refresh = true;
                }
                EngineEvent::CatalogChanged => refresh = true,
                EngineEvent::ScanStarted { .. }
                | EngineEvent::ModelDownload { .. }
                | EngineEvent::ScanProgress { .. }
                | EngineEvent::IndexProgress { .. } => {}
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
        if directory.as_ref().map(|dir| dir.id) == self.current_directory.as_ref().map(|dir| dir.id) {
            return;
        }
        self.back_history.push(self.current_directory.take());
        self.forward_history.clear();
        self.current_directory = directory;
        self.browse.listing_stale = true;
        self.selected_file = None;
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
            std::path::Path::new(&dir.path).parent().map(|parent| parent.to_path_buf())
        });
        if let Some(parent_path) = parent_path
            && let Ok(connection) = self.engine.catalog.reader()
                && let Some(parent) = self
                    .engine
                    .catalog
                    .dir_by_path(&connection, &parent_path.to_string_lossy())
                    .ok()
                    .flatten()
                    && self
                        .config
                        .roots
                        .iter()
                        .any(|root| parent.path.starts_with(root.to_string_lossy().as_ref()))
                    {
                        self.navigate_to(Some(parent));
                        return;
                    }
        self.navigate_to(None);
    }

    /// Runs the unified multimodal search on a worker thread.
    pub fn start_search(&mut self, query: String) {
        if self.searching || query.trim().is_empty() {
            return;
        }
        self.searching = true;
        self.results.hits = Vec::new();
        let engine = Arc::clone(&self.engine);
        let sender = self.search_sender.clone();
        std::thread::spawn(move || {
            let outcome = engine.semantic_search(&query, 100).unwrap_or(SearchOutcome {
                hits: Vec::new(),
                elapsed_millis: 0.0,
                interpreted_query: None,
                vector_count: 0,
            });
            let _ = sender.send(outcome);
        });
    }

    pub fn poll_search(&mut self, context: &egui::Context) {
        if self.searching
            && let Ok(outcome) = self.search_receiver.try_recv() {
                self.results.hits = outcome.hits;
                self.results.elapsed_millis = outcome.elapsed_millis;
                self.results.interpreted_query = outcome.interpreted_query;
                self.results.selected_index = None;
                self.searching = false;
                context.request_repaint();
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
            self.results.hits = Vec::new();
            let engine = Arc::clone(&self.engine);
            let sender = self.search_sender.clone();
            std::thread::spawn(move || {
                let outcome = engine.spoken_query_search(&samples, 100).unwrap_or(SearchOutcome {
                    hits: Vec::new(),
                    elapsed_millis: 0.0,
                    interpreted_query: None,
                    vector_count: 0,
                });
                let _ = sender.send(outcome);
            });
        }
    }

    /// Currently selected file path, if any.
    pub fn selected_path(&self) -> Option<String> {
        self.selected_file.as_ref().map(|(_, path)| path.clone())
    }

    /// Opens a file through the injectable handler (tests) or the OS.
    pub fn dispatch_open(&self, path: &std::path::Path) {
        match &self.open_handler {
            Some(handler) => handler(path),
            None => crate::actions::open_file(path),
        }
    }

    pub fn open_selected(&mut self) {
        if let Some(path) = self.selected_path() {
            self.dispatch_open(std::path::Path::new(&path));
        }
    }

    pub fn trash_selected(&mut self) {
        if let Some((_file, path)) = self.selected_file.clone()
            && crate::actions::move_to_trash(std::path::Path::new(&path)).is_ok() {
                if let Ok(slots) = self.engine.catalog.remove_path(&path) {
                    self.engine.release_vector_slots(&slots);
                }
                self.selected_file = None;
                self.browse.listing_stale = true;
                self.push_notice("moved to trash".into());
            }
    }

    /// Prunes stale notices (shown for 5 seconds).
    pub fn expire_notices(&mut self) {
        self.notices.retain(|(_, at)| at.elapsed() < Duration::from_secs(5));
    }

    /// Global keyboard shortcuts.
    pub fn handle_shortcuts(&mut self, context: &egui::Context) {
        let command = context.input(|input| input.modifiers.command);
        let search_focused = self.search_box_has_focus;
        let mut focus_search = false;
        let mut trash = false;
        let mut reveal = false;
        let mut clear_search = false;
        context.input(|input| {
            if command && input.key_pressed(egui::Key::F) {
                focus_search = true;
            }
            if command && input.key_pressed(egui::Key::Backspace) {
                trash = true;
            }
            if command && input.key_pressed(egui::Key::ArrowUp) {
                reveal = true;
            }
            if !search_focused && input.key_pressed(egui::Key::Escape) && !self.active_query.is_empty() {
                clear_search = true;
            }
        });
        if focus_search {
            self.results.request_search_focus = true;
        }
        if trash {
            self.trash_selected();
        }
        if reveal
            && let Some(path) = self.selected_path() {
                crate::actions::reveal_in_file_manager(std::path::Path::new(&path));
            }
        if clear_search {
            self.active_query.clear();
            self.search_edit.clear();
            self.results.hits.clear();
            self.browse.name_filter_active.clear();
            self.browse.listing_stale = true;
        }
    }
}

pub fn draw_top_bar(ui: &mut Ui, app: &mut IpicApp) {
    egui::Panel::top("top_bar")
        .frame(egui::Frame::new().inner_margin(egui::Margin::symmetric(14, 8)))
        .show(ui, |ui| {
            ui.horizontal_centered(|ui| {
                ui.label(egui::RichText::new("◆ ipic").color(crate::theme::ACCENT).size(19.0).strong());
                ui.add_space(10.0);
                if ui
                    .add_enabled(
                        !app.back_history.is_empty(),
                        egui::Button::new("‹").min_size(egui::vec2(26.0, 24.0)),
                    )
                    .on_hover_text("Back (history)")
                    .clicked()
                {
                    app.navigate_back();
                }
                if ui
                    .add_enabled(
                        !app.forward_history.is_empty(),
                        egui::Button::new("›").min_size(egui::vec2(26.0, 24.0)),
                    )
                    .on_hover_text("Forward (history)")
                    .clicked()
                {
                    app.navigate_forward();
                }
                if ui.button("↑").on_hover_text("Parent directory").clicked() {
                    app.navigate_up();
                }
                ui.add_space(4.0);
                draw_breadcrumb(ui, app);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("⚙").on_hover_text("Settings").clicked() {
                        app.show_settings = !app.show_settings;
                    }
                    if ui
                        .button(if app.show_details { "◧" } else { "◨" })
                        .on_hover_text("Toggle details panel")
                        .clicked()
                    {
                        app.show_details = !app.show_details;
                    }
                    ui.separator();
                    draw_search_box(ui, app);
                });
            });
        });
}

fn draw_search_box(ui: &mut Ui, app: &mut IpicApp) {
    let recording = app.recorder.is_some();
    let mic_label = if recording {
        egui::RichText::new(format!(
            "● {:.0}s",
            app.recorder.as_ref().map(|recorder| recorder.captured_seconds()).unwrap_or(0.0)
        ))
        .color(crate::theme::DANGER)
        .strong()
    } else {
        egui::RichText::new("🎙").color(crate::theme::TEXT_PRIMARY)
    };
    if ui
        .add(egui::Button::new(mic_label).fill(if recording {
            crate::theme::DANGER
        } else {
            crate::theme::SURFACE_CARD
        }))
        .on_hover_text("Speak a query — transcribed on-device")
        .clicked()
    {
        app.toggle_recording();
    }
    let width = (ui.available_width() - 130.0).clamp(220.0, 520.0);
    let edit = egui::TextEdit::singleline(&mut app.search_edit)
        .hint_text("Search everything — files, documents, audio, video, images…  (⌘F)")
        .desired_width(width)
        .clip_text(true);
    let response = ui.add(edit);
    app.search_box_has_focus = response.has_focus();
    if app.results.request_search_focus {
        response.request_focus();
        app.results.request_search_focus = false;
    }
    if response.changed() {
        // Live name filtering while browsing; typing leaves search mode.
        app.active_query.clear();
        app.results.hits.clear();
        app.browse.name_filter_active = app.search_edit.clone();
        app.browse.listing_stale = true;
    }
    let enter_pressed = response.has_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter));
    if ui.button("Search").clicked() || enter_pressed {
        let query = app.search_edit.trim().to_string();
        if !query.is_empty() {
            app.active_query = query.clone();
            app.start_search(query);
        }
        response.surrender_focus();
    }
    if app.searching {
        ui.spinner();
    }
}

fn draw_breadcrumb(ui: &mut Ui, app: &mut IpicApp) {
    ui.style_mut().spacing.item_spacing.x = 4.0;
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
                if ui.add(egui::Button::new(label).fill(egui::Color32::TRANSPARENT)).clicked()
                    && let Ok(connection) = app.engine.catalog.reader()
                        && let Some(target_dir) = app
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
}

pub fn draw_status_bar(ui: &mut Ui, app: &mut IpicApp) {
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
                ui.label(egui::RichText::new(format!("{}× cores", status.core_count)).color(crate::theme::TEXT_DIM));
                if app.searching {
                    ui.label(egui::RichText::new("searching…").color(crate::theme::TEXT_DIM));
                } else if !app.results.hits.is_empty() {
                    ui.label(
                        egui::RichText::new(format!("search: {:.0} ms", app.results.elapsed_millis))
                            .color(crate::theme::TEXT_DIM),
                    );
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    for (message, _) in app.notices.iter().rev() {
                        ui.label(egui::RichText::new(message.clone()).color(crate::theme::TEXT_DIM).small());
                    }
                });
            });
        });
}
