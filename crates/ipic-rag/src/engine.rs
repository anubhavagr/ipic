//! Engine: the single orchestration point. Owns the catalog, embedder, vector
//! store and whisper transcriber; runs the scan → extract → transcribe → embed
//! pipeline on a self-serving worker pool sized to the CPU core count.

use crate::embed::{HashingEmbedder, NeuralEmbedder, TextEmbedder};
use crate::extract;
use crate::fingerprint::{self, FINGERPRINT_DIM};
use crate::search::{self, SearchOutcome};
use crate::transcribe::{self, Transcriber};
use crate::vector_store::VectorStore;
use anyhow::Result;
use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use ipic_core::catalog::{Catalog, NewFile};
use ipic_core::watcher::{self, WatchUpdate};
use crate::search::RRF_CONSTANT;
use ipic_core::walker::{self, WalkItem};
use ipic_core::{Config, FileKind, RagStatus};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub enum EngineEvent {
    ScanStarted { roots: Vec<PathBuf> },
    ScanProgress { files_seen: u64, directories_seen: u64, elapsed_seconds: f64 },
    ScanFinished { files_seen: u64, directories_seen: u64, elapsed_seconds: f64 },
    EmbedderReady { model_id: String, neural: bool },
    ModelDownload { model: String, downloaded_bytes: u64, total_bytes: Option<u64>, finished: bool },
    TranscriberReady { model: String },
    TranscriberFailed { reason: String },
    IndexProgress { pending: u64, busy: u64, done: u64, failed: u64 },
    IndexingIdle,
    CatalogChanged,
    Notice(String),
}

#[derive(Debug, Clone)]
pub struct EngineStatus {
    pub scanning: bool,
    pub total_files: i64,
    pub pending: u64,
    pub done: u64,
    pub failed: u64,
    pub vector_count: u32,
    pub embedder_model: String,
    pub neural_embedder: bool,
    pub whisper_model: String,
    pub transcriber_ready: bool,
    pub core_count: usize,
}

/// A claimed indexing job.
struct IndexingJob {
    file_id: i64,
    path: PathBuf,
    kind: FileKind,
}

/// Reciprocal-rank weight of the acoustic (audio-to-audio) lane.
const ACOUSTIC_WEIGHT: f32 = 0.9;

pub struct Engine {
    pub catalog: Arc<Catalog>,
    pub config: Config,
    /// Data directory this engine was launched with (config saves scoped to it).
    pub data_directory: PathBuf,
    /// Live scan roots, swappable from the GUI settings without a restart.
    roots: Arc<std::sync::RwLock<Vec<PathBuf>>>,
    pub events: Receiver<EngineEvent>,
    event_sender: Sender<EngineEvent>,
    vector_store: Arc<Mutex<VectorStore>>,
    /// Acoustic fingerprints for audio-to-audio matching.
    audio_store: Arc<Mutex<VectorStore>>,
    embedder: Arc<dyn TextEmbedder>,
    neural_embedder: bool,
    transcriber: Arc<Mutex<Option<Arc<Transcriber>>>>,
    transcriber_failure: Arc<Mutex<Option<String>>>,
    stop: Arc<AtomicBool>,
    scanning: Arc<AtomicBool>,
    scan_finished: Arc<AtomicBool>,
    idle_announced: Arc<AtomicBool>,
    /// Join handles for every background thread; joined on shutdown so the
    /// whisper/Metal context is destroyed while the process is still healthy.
    worker_handles: Mutex<Vec<std::thread::JoinHandle<()>>>,
}

impl Engine {
    /// Registers a spawned worker so shutdown() can join it.
    fn track_worker(&self, handle: std::thread::JoinHandle<()>) {
        self.worker_handles.lock().unwrap().push(handle);
    }

    /// Stops background work and joins all threads; call before process exit.
    pub fn shutdown(&self) {
        self.stop_background_work();
        let handles: Vec<_> = std::mem::take(&mut *self.worker_handles.lock().unwrap());
        for handle in handles {
            let _ = handle.join();
        }
    }

    /// Boots the engine against the default user data directory.
    pub fn launch(config: Config) -> Result<Arc<Engine>> {
        Self::launch_with_data_dir(config, ipic_core::data_dir())
    }

    /// Boots the engine against an explicit data directory (tests, portable installs).
    pub fn launch_with_data_dir(mut config: Config, data_directory: PathBuf) -> Result<Arc<Engine>> {
        // The walker indexes canonical roots; mirror that everywhere so
        // sidebar/tree/library lookups match the catalog's stored paths.
        config.roots = config
            .roots
            .iter()
            .map(|root| std::fs::canonicalize(root).unwrap_or_else(|_| root.clone()))
            .collect();
        let core_count = std::thread::available_parallelism().map(|count| count.get()).unwrap_or(4);
        let catalog = Arc::new(Catalog::open(&data_directory.join("catalog.db"))?);

        // Embedder: neural by default; hashing override (or offline failure) keeps
        // search functional without any model download. The intra-op pool keeps
        // headroom for extraction/whisper workers so ORT spin-waits never starve.
        let (embedder, neural_embedder, notice) = if config.embedder == "hashing" {
            (Arc::new(HashingEmbedder) as Arc<dyn TextEmbedder>, false, None)
        } else {
            match NeuralEmbedder::load(
                data_directory.join("models").join("embeddings"),
                true,
                (core_count / 2).max(2),
            ) {
                Ok(neural) => (Arc::new(neural) as Arc<dyn TextEmbedder>, true, None),
                Err(error) => (
                    Arc::new(HashingEmbedder) as Arc<dyn TextEmbedder>,
                    false,
                    Some(format!("neural embedder unavailable ({error}); using lexical fallback")),
                ),
            }
        };

        let (mut vector_store, compatible) =
            VectorStore::open(&data_directory, embedder.dim(), embedder.model_id())?;
        if !compatible {
            // Embedder changed: wipe derived data and reindex everything.
            catalog.clear_chunks()?;
            vector_store.clear()?;
        } else if let Ok(orphan_slots) = catalog.delete_orphan_chunks() {
            vector_store.free(&orphan_slots)?;
        }

        // Acoustic fingerprints live in their own quantized store.
        let (mut audio_store, audio_compatible) = VectorStore::open(
            &data_directory.join("audio-fingerprints"),
            FINGERPRINT_DIM,
            "audio-fingerprint-v1",
        )?;
        if !audio_compatible && let Ok(orphan_slots) = catalog.drain_fingerprints() {
            audio_store.free(&orphan_slots)?;
        }

        let (event_sender, events) = bounded(1024);
        let engine = Arc::new(Engine {
            catalog: Arc::clone(&catalog),
            config: config.clone(),
            data_directory: data_directory.clone(),
            roots: Arc::new(std::sync::RwLock::new(config.roots.clone())),
            events,
            event_sender: event_sender.clone(),
            vector_store: Arc::new(Mutex::new(vector_store)),
            audio_store: Arc::new(Mutex::new(audio_store)),
            embedder: Arc::clone(&embedder),
            neural_embedder,
            transcriber: Arc::new(Mutex::new(None)),
            transcriber_failure: Arc::new(Mutex::new(None)),
            stop: Arc::new(AtomicBool::new(false)),
            scanning: Arc::new(AtomicBool::new(false)),
            scan_finished: Arc::new(AtomicBool::new(false)),
            idle_announced: Arc::new(AtomicBool::new(false)),
            worker_handles: Mutex::new(Vec::new()),
        });

        // Warm the ONNX session before whisper/Accelerate threads exist: ORT's
        // lazily-initialized thread pool can stall inside libdispatch when it
        // first spins up concurrently with BLAS work queues.
        if neural_embedder {
            let _ = embedder.embed_batch(&["warmup".to_string()]);
        }
        let _ = event_sender.send(EngineEvent::EmbedderReady {
            model_id: embedder.model_id().to_string(),
            neural: neural_embedder,
        });
        if let Some(notice) = notice {
            let _ = event_sender.send(EngineEvent::Notice(notice));
        }

        let (chunk_sender, chunk_receiver) = unbounded::<(i64, Vec<String>)>();
        spawn_extraction_workers(&engine, chunk_sender, core_count);
        spawn_embed_writer(&engine, chunk_receiver);
        spawn_progress_reporter(&engine);
        spawn_transcriber_initializer(&engine);
        engine.spawn_scan();
        Ok(engine)
    }

    /// Kicks off a full rescan of all configured roots (no-op while one runs).
    pub fn spawn_scan(&self) {
        if self.scanning.swap(true, Ordering::AcqRel) {
            return;
        }
        self.scan_finished.store(false, Ordering::Release);
        let engine_scanning = Arc::clone(&self.scanning);
        let scan_finished = Arc::clone(&self.scan_finished);
        let catalog = Arc::clone(&self.catalog);
        let vector_store = Arc::clone(&self.vector_store);
        let audio_store = Arc::clone(&self.audio_store);
        let event_sender = self.event_sender.clone();
        let roots = self.current_roots();
        let skip_names = self.config.skip_dir_names.clone();
        let stop = Arc::clone(&self.stop);
        let core_count = self.core_count();
        std::thread::Builder::new().name("ipic-scan".into()).spawn(move || {
            let _ = event_sender.send(EngineEvent::ScanStarted { roots: roots.clone() });
            catalog.begin_full_scan().ok();
            let stats = run_parallel_scan(&catalog, &roots, &skip_names, core_count, &event_sender);
            if let Ok(freed_slots) = catalog.finish_full_scan() {
                vector_store.lock().unwrap().free(&freed_slots).ok();
            }
            if let Ok(freed_fingerprints) = catalog.drain_orphan_fingerprints() {
                audio_store.lock().unwrap().free(&freed_fingerprints).ok();
            }
            engine_scanning.store(false, Ordering::Release);
            scan_finished.store(true, Ordering::Release);
            let _ = event_sender.send(EngineEvent::ScanFinished {
                files_seen: stats.files,
                directories_seen: stats.dirs,
                elapsed_seconds: stats.elapsed_secs,
            });
            let _ = event_sender.send(EngineEvent::CatalogChanged);
            if !stop.load(Ordering::Relaxed) {
                start_watcher(&catalog, &roots, &vector_store, &audio_store, &event_sender, &stop);
            }
        }).ok();
    }

    pub fn core_count(&self) -> usize {
        std::thread::available_parallelism().map(|count| count.get()).unwrap_or(4)
    }

    /// Non-blocking event emit for background threads (telemetry may drop under pressure).
    fn emit(&self, event: EngineEvent) {
        let _ = self.event_sender.try_send(event);
    }

    /// Synchronous hybrid search on the caller's thread (typically tens of ms).
    pub fn semantic_search(&self, query: &str, limit: usize) -> Result<SearchOutcome> {
        let connection = self.catalog.reader()?;
        let vector_store = self.vector_store.lock().unwrap();
        search::semantic_search(&self.catalog, &connection, &vector_store, self.embedder.as_ref(), query, limit)
    }

    /// Audio query: transcribe locally with whisper, then hybrid search.
    /// Audio query with two fused signal groups: acoustic similarity against
    /// every indexed recording (audio-to-audio match) plus the transcript's
    /// usual hybrid lanes (semantic + keyword + filename).
    pub fn spoken_query_search(&self, audio_pcm: &[f32], limit: usize) -> Result<SearchOutcome> {
        let transcript = self.transcribe_pcm(audio_pcm)?;
        let mut outcome = if transcript.trim().is_empty() {
            SearchOutcome {
                hits: Vec::new(),
                elapsed_millis: 0.0,
                interpreted_query: None,
                vector_count: 0,
            }
        } else {
            self.semantic_search(&transcript, limit)?
        };
        outcome.interpreted_query = (!transcript.trim().is_empty()).then_some(transcript);

        // Acoustic lane: reciprocal-rank contribution, merged by file.
        let acoustic_matches = self.audio_match(audio_pcm);
        if !acoustic_matches.is_empty() {
            let connection = self.catalog.reader()?;
            let mut acoustic_scores: HashMap<i64, f32> = HashMap::new();
            for (rank, (file_id, _similarity)) in acoustic_matches.iter().enumerate() {
                acoustic_scores.insert(*file_id, ACOUSTIC_WEIGHT / (RRF_CONSTANT + rank as f32));
            }
            let audio_file_ids: Vec<i64> = acoustic_matches.iter().map(|(file_id, _)| *file_id).collect();
            let hydrated = self.catalog.files_by_ids(&connection, &audio_file_ids)?;
            let mut merged = outcome.hits;
            for (file_row, path) in hydrated {
                let acoustic = acoustic_scores.get(&file_row.id).copied().unwrap_or(0.0);
                match merged.iter_mut().find(|hit| hit.file.id == file_row.id) {
                    Some(hit) => {
                        hit.score += acoustic;
                        hit.sources.acoustic = true;
                    }
                    None => merged.push(crate::search::SearchHit {
                        file: file_row,
                        path,
                        snippet: "acoustic match".into(),
                        score: acoustic,
                        sources: crate::search::MatchSources { acoustic: true, ..Default::default() },
                    }),
                }
            }
            merged.sort_by(|left, right| right.score.total_cmp(&left.score));
            merged.truncate(limit);
            outcome.hits = merged;
        }
        Ok(outcome)
    }

    pub fn transcribe_pcm(&self, audio_pcm: &[f32]) -> Result<String> {
        self.ensure_transcriber()?.transcribe(audio_pcm)
    }

    /// Loads (downloading once if needed) the whisper model; guarded by a mutex
    /// so concurrent workers share a single download/load.
    pub fn ensure_transcriber(&self) -> Result<Arc<Transcriber>> {
        if self.config.whisper_model == "none" {
            return Err(anyhow::anyhow!("whisper disabled in configuration"));
        }
        let mut slot = self.transcriber.lock().unwrap();
        if let Some(transcriber) = slot.as_ref() {
            return Ok(Arc::clone(transcriber));
        }
        if let Some(reason) = self.transcriber_failure.lock().unwrap().as_ref() {
            return Err(anyhow::anyhow!("transcriber failed to initialize: {reason}"));
        }
        let model_name = self.config.whisper_model.clone();
        let event_sender = self.event_sender.clone();
        let model_path = transcribe::ensure_model_downloaded(&model_name, |downloaded_bytes, total_bytes| {
            let _ = event_sender.send(EngineEvent::ModelDownload {
                model: model_name.clone(),
                downloaded_bytes,
                total_bytes,
                finished: false,
            });
        })?;
        let threads_per_worker = (self.core_count() / self.config.whisper_workers.max(1)).max(1);
        let transcriber = Arc::new(Transcriber::load(
            &model_path,
            self.config.whisper_workers,
            threads_per_worker,
            self.config.whisper_use_gpu,
        )?);
        *slot = Some(Arc::clone(&transcriber));
        let _ = self.event_sender.send(EngineEvent::TranscriberReady { model: model_name });
        Ok(transcriber)
    }

    pub fn status(&self) -> EngineStatus {
        let (pending, busy, done, failed) = self.catalog.rag_counters().unwrap_or((0, 0, 0, 0));
        EngineStatus {
            scanning: self.scanning.load(Ordering::Relaxed),
            total_files: self.catalog.total_files().unwrap_or(0),
            pending: (pending + busy) as u64,
            done: done as u64,
            failed: failed as u64,
            vector_count: self.vector_store.lock().unwrap().count(),
            embedder_model: self.embedder.model_id().to_string(),
            neural_embedder: self.neural_embedder,
            whisper_model: self.config.whisper_model.clone(),
            transcriber_ready: self.transcriber.lock().unwrap().is_some(),
            core_count: self.core_count(),
        }
    }

    pub fn is_scanning(&self) -> bool {
        self.scanning.load(Ordering::Relaxed)
    }

    pub fn stop_background_work(&self) {
        self.stop.store(true, Ordering::Release);
    }

    /// Current scan roots (settings may swap them at runtime).
    pub fn current_roots(&self) -> Vec<PathBuf> {
        self.roots.read().unwrap().clone()
    }

    /// Swaps scan roots; the next rescan covers the new set.
    pub fn replace_roots(&self, roots: Vec<PathBuf>) {
        *self.roots.write().unwrap() = roots;
    }

    /// Persists a configuration into this engine's data directory.
    pub fn persist_config(&self, config: &Config) {
        if let Err(error) = config.save_to_directory(&self.data_directory) {
            self.emit(EngineEvent::Notice(format!("config save failed: {error}")));
        }
    }

    /// Frees text-vector slots after external deletions (GUI trash action).
    pub fn release_vector_slots(&self, slots: &[i64]) {
        if !slots.is_empty() {
            self.vector_store.lock().unwrap().free(slots).ok();
        }
    }

    /// Frees acoustic-fingerprint slots after external deletions (GUI trash).
    pub fn free_fingerprint_slots(&self, slots: &[i64]) {
        if !slots.is_empty() {
            self.audio_store.lock().unwrap().free(slots).ok();
        }
    }

    /// Acoustic-similarity lane: audio query → matching audio/video files.
    pub fn audio_match(&self, audio_pcm: &[f32]) -> Vec<(i64, f32)> {
        let query = fingerprint::fingerprint_from_pcm(audio_pcm, extract::SAMPLE_RATE as u32);
        if query.is_empty() {
            return Vec::new();
        }
        self.audio_store.lock().unwrap().top_k(&query, 32)
    }
}

/// Parallel walk + single-writer collector batching into SQLite.
fn run_parallel_scan(
    catalog: &Arc<Catalog>,
    roots: &[PathBuf],
    skip_names: &[String],
    thread_count: usize,
    event_sender: &Sender<EngineEvent>,
) -> walker::WalkStats {
    let (item_sender, item_receiver) = unbounded::<WalkItem>();
    let catalog_writer = Arc::clone(catalog);
    let event_sender_collector = event_sender.clone();
    let collector = std::thread::Builder::new().name("ipic-collect".into()).spawn(move || {
        let mut directory_ids: HashMap<String, i64> = catalog_writer.dir_id_map().unwrap_or_default();
        let mut pending_files: Vec<NewFile> = Vec::new();
        let mut directories_seen = 0u64;
        let mut files_seen = 0u64;
        let mut next_progress_at = 2_000u64;
        let mut last_progress_time = Instant::now();
        let started = Instant::now();
        loop {
            let item = match item_receiver.recv_timeout(Duration::from_millis(80)) {
                Ok(item) => item,
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                    if pending_files.is_empty() {
                        continue;
                    }
                    flush_files(&catalog_writer, &mut pending_files);
                    continue;
                }
            };
            match item {
                WalkItem::Dir { path, mtime } => {
                    // Register each directory immediately so child files resolve parents.
                    catalog_writer.upsert_dirs(&[(path.clone(), mtime)]).ok();
                    let path_text = path.to_string_lossy().into_owned();
                    if let Some(directory_id) = catalog_writer.dir_id_by_path(&path_text) {
                        directory_ids.insert(path_text, directory_id);
                    }
                    directories_seen += 1;
                }
                WalkItem::File { path, kind, size, mtime } => {
                    let parent_path =
                        path.parent().map(|parent| parent.to_string_lossy().into_owned()).unwrap_or_default();
                    let directory_id = match directory_ids.get(&parent_path) {
                        Some(directory_id) => *directory_id,
                        None => catalog_writer.dir_id_by_path(&parent_path).unwrap_or(0),
                    };
                    if directory_id != 0 {
                        pending_files.push(NewFile {
                            dir_id: directory_id,
                            name: path.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default(),
                            kind,
                            size,
                            mtime,
                        });
                    }
                    files_seen += 1;
                    let due = files_seen >= next_progress_at || last_progress_time.elapsed().as_secs_f64() >= 2.0;
                    if due {
                        next_progress_at = files_seen + 2_000;
                        last_progress_time = Instant::now();
                        let _ = event_sender_collector.send(EngineEvent::ScanProgress {
                            files_seen,
                            directories_seen,
                            elapsed_seconds: started.elapsed().as_secs_f64(),
                        });
                    }
                }
            }
            if pending_files.len() >= 8192 {
                flush_files(&catalog_writer, &mut pending_files);
            }
        }
        flush_files(&catalog_writer, &mut pending_files);
    });
    let stats = walker::walk_parallel(roots, skip_names, thread_count, item_sender).unwrap_or_default();
    if let Ok(collector) = collector {
        let _ = collector.join();
    }
    stats
}

fn flush_files(catalog: &Arc<Catalog>, pending_files: &mut Vec<NewFile>) {
    if !pending_files.is_empty() {
        catalog.upsert_files(pending_files).ok();
        pending_files.clear();
    }
}

/// Extraction workers self-serve jobs from the catalog (natural backpressure):
/// text/PDF first (fast lane), audio/video gated by a whisper slot semaphore.
fn spawn_extraction_workers(engine: &Arc<Engine>, chunk_sender: Sender<(i64, Vec<String>)>, core_count: usize) {
    let worker_count = core_count.saturating_sub(2).max(2);
    let whisper_permits = Arc::new((Mutex::new(0u32), Condvar::new()));
    let whisper_limit = engine.config.whisper_workers.max(1) as u32;
    for worker_index in 0..worker_count {
        let worker_engine = Arc::clone(engine);
        let chunk_sender = chunk_sender.clone();
        let whisper_permits = Arc::clone(&whisper_permits);
        let handle = std::thread::Builder::new()
            .name(format!("ipic-extract-{worker_index}"))
            .spawn(move || loop {
                if worker_engine.stop.load(Ordering::Relaxed) {
                    return;
                }
                let fast_lane_job = claim_job(&worker_engine, &[FileKind::Text, FileKind::Pdf, FileKind::Image]);
                let job = match fast_lane_job.or_else(|| claim_job(&worker_engine, &[FileKind::Audio, FileKind::Video])) {
                    Some(job) => job,
                    None => {
                        std::thread::sleep(Duration::from_millis(350));
                        continue;
                    }
                };
                process_job(&worker_engine, &chunk_sender, job, &whisper_permits, whisper_limit);
            });
        if let Ok(handle) = handle {
            engine.track_worker(handle);
        }
    }
}

fn claim_job(engine: &Arc<Engine>, kinds: &[FileKind]) -> Option<IndexingJob> {
    let (file_id, path) = engine.catalog.claim_rag_jobs(kinds, 1).ok()?.pop()?;
    let path = PathBuf::from(path);
    let kind = ipic_core::kind_for_path(&path);
    Some(IndexingJob { file_id, path, kind })
}

fn process_job(
    engine: &Arc<Engine>,
    chunk_sender: &Sender<(i64, Vec<String>)>,
    job: IndexingJob,
    whisper_permits: &(Mutex<u32>, Condvar),
    whisper_limit: u32,
) {
    // Re-indexing must first clear chunks from any earlier attempt.
    if let Ok(freed_slots) = engine.catalog.delete_chunks_for_files(&[job.file_id])
        && !freed_slots.is_empty() {
            engine.vector_store.lock().unwrap().free(&freed_slots).ok();
        }
    let chunk_texts: Vec<String> = match job.kind {
        FileKind::Text | FileKind::Pdf | FileKind::Image => {
            match extract::extract_text(&job.path, job.kind, engine.config.max_text_mb * 1024 * 1024) {
                Ok(Some(text)) => extract::chunk_text(&text),
                Ok(None) => Vec::new(),
                Err(_) => {
                    engine.catalog.set_rag(&[job.file_id], RagStatus::Failed).ok();
                    return;
                }
            }
        }
        FileKind::Audio | FileKind::Video => {
            // Bound concurrent whisper decodes to the whisper state pool size.
            let mut permits = whisper_permits.0.lock().unwrap();
            while *permits >= whisper_limit {
                permits = whisper_permits.1.wait(permits).unwrap();
            }
            *permits += 1;
            drop(permits);
            let transcription = transcribe_media(engine, &job);
            let mut permits = whisper_permits.0.lock().unwrap();
            *permits -= 1;
            whisper_permits.1.notify_one();
            drop(permits);
            match transcription {
                Ok(transcript) => extract::chunk_text(&transcript),
                Err(_) => {
                    engine.catalog.set_rag(&[job.file_id], RagStatus::Failed).ok();
                    return;
                }
            }
        }
        _ => Vec::new(),
    };
    if chunk_texts.is_empty() {
        engine.catalog.set_rag(&[job.file_id], RagStatus::Done).ok();
    } else {
        let _ = chunk_sender.send((job.file_id, chunk_texts));
    }
}

fn transcribe_media(engine: &Arc<Engine>, job: &IndexingJob) -> Result<String> {
    let audio_pcm = extract::decode_speech_pcm(&job.path)?;
    if audio_pcm.len() > extract::SAMPLE_RATE {
        engine
            .catalog
            .set_duration(job.file_id, audio_pcm.len() as f64 / extract::SAMPLE_RATE as f64)
            .ok();
    }
    let transcriber = engine.ensure_transcriber()?;
    transcriber.transcribe(&audio_pcm)
}

/// Single embed writer: batches chunks, embeds with ONNX (multi-threaded),
/// persists FTS rows + quantized vectors, then flips files to Done.
fn spawn_embed_writer(engine: &Arc<Engine>, chunk_receiver: Receiver<(i64, Vec<String>)>) {
    let worker_engine = Arc::clone(engine);
    let handle = std::thread::Builder::new().name("ipic-embed".into()).spawn(move || {
        let engine = worker_engine;
        let mut awaiting: Vec<(i64, Vec<String>)> = Vec::new();
        loop {
            let received = chunk_receiver.recv_timeout(Duration::from_millis(120));
            match received {
                Ok((file_id, chunk_texts)) => awaiting.push((file_id, chunk_texts)),
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    if awaiting.is_empty() {
                        return;
                    }
                }
            }
            // Flush when the batch budget is hit, or when the queue has gone
            // idle (non-destructive emptiness check — never consume here).
            let budget_reached = awaiting.iter().map(|(_, chunks)| chunks.len()).sum::<usize>()
                >= engine.config.embed_batch;
            let queue_drained = chunk_receiver.is_empty();
            if !awaiting.is_empty() && (budget_reached || queue_drained) {
                flush_embeddings(&engine, &mut awaiting);
            }
            if engine.stop_background_stopped() && awaiting.is_empty() {
                return;
            }
        }
    });
    if let Ok(handle) = handle {
        engine.track_worker(handle);
    }
}

impl Engine {
    fn stop_background_stopped(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }
}

fn flush_embeddings(engine: &Arc<Engine>, awaiting: &mut Vec<(i64, Vec<String>)>) {
    let flattened: Vec<(i64, String)> = awaiting
        .iter()
        .flat_map(|(file_id, chunks)| chunks.iter().map(move |chunk| (*file_id, chunk.clone())))
        .collect();
    if flattened.is_empty() {
        awaiting.clear();
        return;
    }
    let texts: Vec<String> = flattened.iter().map(|(_, text)| text.clone()).collect();
    let embedded = match engine.embedder.embed_batch(&texts) {
        Ok(embedded) => embedded,
        Err(_) => {
            let failed_files: Vec<i64> = awaiting.iter().map(|(file_id, _)| *file_id).collect();
            engine.catalog.set_rag(&failed_files, RagStatus::Failed).ok();
            awaiting.clear();
            return;
        }
    };
    let references: Vec<(i64, &str)> = flattened.iter().map(|(file_id, text)| (*file_id, text.as_str())).collect();
    let rowids = match engine.catalog.insert_chunks(&references) {
        Ok(rowids) => rowids,
        Err(_) => {
            awaiting.clear();
            return;
        }
    };
    let slots = {
        let mut vector_store = engine.vector_store.lock().unwrap();
        vector_store.append_batch(&rowids, &embedded).unwrap_or_default()
    };
    let assignments: Vec<(i64, i64)> = rowids.iter().cloned().zip(slots.iter().cloned()).collect();
    engine.catalog.assign_vector_slots(&assignments).ok();
    let completed_files: Vec<i64> = awaiting.iter().map(|(file_id, _)| *file_id).collect();
    engine.catalog.set_rag(&completed_files, RagStatus::Done).ok();
    awaiting.clear();
}

fn spawn_progress_reporter(engine: &Arc<Engine>) {
    let worker_engine = Arc::clone(engine);
    let handle = std::thread::Builder::new().name("ipic-progress".into()).spawn(move || {
        let engine = worker_engine;
        loop {
        if engine.stop_background_stopped() {
            return;
        }
        if let Ok((pending, busy, done, failed)) = engine.catalog.rag_counters() {
            engine.emit(EngineEvent::IndexProgress {
                pending: pending as u64,
                busy: busy as u64,
                done: done as u64,
                failed: failed as u64,
            });
            let idle = pending + busy == 0 && engine.scan_finished.load(Ordering::Relaxed);
            if idle && !engine.idle_announced.swap(true, Ordering::AcqRel) {
                engine.emit(EngineEvent::IndexingIdle);
            } else if pending + busy > 0 {
                engine.idle_announced.store(false, Ordering::Release);
            }
        }
        std::thread::sleep(Duration::from_secs(1));
    }
    });
    if let Ok(handle) = handle {
        engine.track_worker(handle);
    }
}

fn spawn_transcriber_initializer(engine: &Arc<Engine>) {
    if engine.config.whisper_model == "none" {
        return; // whisper explicitly disabled (tests, text-only deployments)
    }
    let worker_engine = Arc::clone(engine);
    let handle = std::thread::Builder::new().name("ipic-whisper-init".into()).spawn(move || {
        let engine = worker_engine;
        if let Err(error) = engine.ensure_transcriber() {
            *engine.transcriber_failure.lock().unwrap() = Some(error.to_string());
            engine.emit(EngineEvent::TranscriberFailed { reason: error.to_string() });
        }
    });
    if let Ok(handle) = handle {
        engine.track_worker(handle);
    }
}

fn start_watcher(
    catalog: &Arc<Catalog>,
    roots: &[PathBuf],
    vector_store: &Arc<Mutex<VectorStore>>,
    audio_store: &Arc<Mutex<VectorStore>>,
    event_sender: &Sender<EngineEvent>,
    stop: &Arc<AtomicBool>,
) {
    let (update_sender, update_receiver) = unbounded::<WatchUpdate>();
    let Ok(watcher_stop) = watcher::spawn(Arc::clone(catalog), roots.to_vec(), update_sender) else {
        return;
    };
    let catalog = Arc::clone(catalog);
    let vector_store = Arc::clone(vector_store);
    let audio_store = Arc::clone(audio_store);
    let event_sender = event_sender.clone();
    let stop = Arc::clone(stop);
    std::thread::Builder::new().name("ipic-watch".into()).spawn(move || loop {
        if stop.load(Ordering::Relaxed) {
            watcher_stop.store(true, Ordering::Release);
            return;
        }
        match update_receiver.recv_timeout(Duration::from_millis(500)) {
            Ok(WatchUpdate::Rescan(directory)) => {
                let stats = run_parallel_scan(&catalog, &[directory], &[], 4, &event_sender);
                if stats.files > 0 {
                    let _ = event_sender.try_send(EngineEvent::CatalogChanged);
                }
            }
            Ok(WatchUpdate::FreedSlots(slots)) => {
                vector_store.lock().unwrap().free(&slots).ok();
                if let Ok(freed_fingerprints) = catalog.drain_orphan_fingerprints() {
                    audio_store.lock().unwrap().free(&freed_fingerprints).ok();
                }
                let _ = event_sender.try_send(EngineEvent::CatalogChanged);
            }
            Err(_) => {}
        }
    }).ok();
}
