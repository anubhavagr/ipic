//! Engine: supervisor over per-root shards. Each shard owns its SQLite
//! catalog and vector stores; the engine owns the shared models (text, CLIP,
//! whisper), the global extraction pool, the persistent watcher, and the
//! aggregated progress/ETA reported to the UI.

use crate::embed::{HashingEmbedder, NeuralEmbedder, TextEmbedder};
use crate::extract;
use crate::fingerprint;
use crate::progress::{scan_eta, IndexingEstimator};
use crate::search::{self, MatchSources, SearchOutcome, RRF_CONSTANT};
use crate::shard::{migrate_legacy_layout, open_shard, ChunkDelivery, Shard};
use crate::transcribe::{self, Transcriber};
use crate::vision::VisionEmbedder;
use anyhow::Result;
use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use image::DynamicImage;
use ipic_core::catalog::{RemovedSlots, STORE_AUDIO, STORE_IMAGE};
use ipic_core::walker::{self, WalkItem};
use ipic_core::watcher::{self, WatchUpdate};
use ipic_core::{ComputeBudget, Config, DirRow, FileFilter, FileKind, FileRow, RagStatus, SortKey};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, RwLock};
use std::time::{Duration, Instant};

/// Images handed to the single CLIP inference thread in batches.
const IMAGE_BATCH: usize = 16;
/// Reciprocal-rank weight of the acoustic (audio-to-audio) lane.
const ACOUSTIC_WEIGHT: f32 = 0.9;

#[derive(Debug, Clone)]
pub enum EngineEvent {
    ScanStarted { roots: Vec<PathBuf> },
    ScanProgress { files_seen: u64, directories_seen: u64, elapsed_seconds: f64 },
    ScanFinished { files_seen: u64, directories_seen: u64, elapsed_seconds: f64 },
    EmbedderReady { model_id: String, neural: bool },
    VisionReady { model_id: String },
    VisionUnavailable { reason: String },
    ModelDownload { model: String, downloaded_bytes: u64, total_bytes: Option<u64>, finished: bool },
    TranscriberReady { model: String },
    TranscriberFailed { reason: String },
    IndexProgress {
        pending: u64,
        done: u64,
        failed: u64,
        files_per_second: f64,
        eta_seconds: Option<f64>,
    },
    IndexingIdle,
    CatalogChanged,
    Notice(String),
}

/// Aggregated, UI-facing snapshot of all shards.
#[derive(Debug, Clone)]
pub struct EngineStatus {
    pub scanning: bool,
    pub scan_files_seen: u64,
    pub scan_files_per_second: f64,
    pub scan_eta_seconds: Option<f64>,
    pub pending: u64,
    pub done: u64,
    pub failed: u64,
    pub index_files_per_second: f64,
    pub index_eta_seconds: Option<f64>,
    pub total_files: i64,
    pub vector_count: u32,
    pub image_vector_count: u32,
    pub embedder_model: String,
    pub neural_embedder: bool,
    pub vision_model: Option<String>,
    pub whisper_model: String,
    pub transcriber_ready: bool,
    pub core_count: usize,
    pub compute: ComputeBudget,
}

/// A claimed indexing job, bound to the shard that owns the file.
struct IndexingJob {
    shard: Arc<Shard>,
    file_id: i64,
    path: PathBuf,
    kind: FileKind,
}

pub struct Engine {
    pub config: Config,
    /// Data directory this engine was launched with (config saves scoped to it).
    pub data_directory: PathBuf,
    shards: RwLock<Vec<Arc<Shard>>>,
    pub events: Receiver<EngineEvent>,
    event_sender: Sender<EngineEvent>,
    embedder: Arc<dyn TextEmbedder>,
    neural_embedder: bool,
    /// None when image embedding is disabled in configuration.
    vision: Option<Arc<VisionEmbedder>>,
    image_sender: Option<Sender<(Arc<Shard>, i64, DynamicImage)>>,
    transcriber: Arc<Mutex<Option<Arc<Transcriber>>>>,
    transcriber_failure: Arc<Mutex<Option<String>>>,
    stop: Arc<AtomicBool>,
    scanning: Arc<AtomicBool>,
    scan_finished: Arc<AtomicBool>,
    idle_announced: Arc<AtomicBool>,
    estimator: Mutex<IndexingEstimator>,
    /// 1 Hz aggregate maintained by the progress reporter; the UI frame loop
    /// clones it instead of querying every shard's SQLite.
    status_cache: Mutex<EngineStatus>,
    scan_files_seen: Arc<AtomicU64>,
    scan_directories_seen: Arc<AtomicU64>,
    /// Catalog size before the running scan: the honest total for rescan ETAs.
    scan_total_hint: AtomicU64,
    scan_started: Mutex<Option<Instant>>,
    watcher_stop: Mutex<Option<Arc<AtomicBool>>>,
    compute: ComputeBudget,
    /// Late-bound self reference so &self methods can spawn Arc-owning threads.
    weak_self: OnceLock<std::sync::Weak<Engine>>,
    /// Join handles for every background thread; joined on shutdown so the
    /// whisper/Metal context is destroyed while the process is still healthy.
    worker_handles: Mutex<Vec<std::thread::JoinHandle<()>>>,
}

impl Engine {
    fn track_worker(&self, handle: std::thread::JoinHandle<()>) {
        self.worker_handles.lock().unwrap().push(handle);
    }

    /// Stops background work and joins all threads; call before process exit.
    pub fn shutdown(&self) {
        self.stop_background_work();
        if let Some(watcher_stop) = self.watcher_stop.lock().unwrap().take() {
            watcher_stop.store(true, Ordering::Release);
        }
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
        let compute = config.compute.resolve();
        // Saturate every core for queries: top_k scans are rayon-sharded.
        let _ = rayon::ThreadPoolBuilder::new().num_threads(compute.search_threads).build_global();

        config.roots = config.canonical_roots();
        let roots = config.roots.clone();
        migrate_legacy_layout(&data_directory, &roots)?;

        let (embedder, neural_embedder, notice) = if config.embedder == "hashing" {
            (Arc::new(HashingEmbedder) as Arc<dyn TextEmbedder>, false, None)
        } else {
            match NeuralEmbedder::load(
                data_directory.join("models").join("embeddings"),
                true,
                compute.ort_threads,
            ) {
                Ok(neural) => (Arc::new(neural) as Arc<dyn TextEmbedder>, true, None),
                Err(error) => (
                    Arc::new(HashingEmbedder) as Arc<dyn TextEmbedder>,
                    false,
                    Some(format!("neural embedder unavailable ({error}); using lexical fallback")),
                ),
            }
        };

        let mut shards = Vec::with_capacity(roots.len());
        for root in &roots {
            shards.push(Arc::new(open_shard(&data_directory, root, embedder.dim(), embedder.model_id())?));
        }

        let vision_enabled = config.image_embedder != "none";
        let vision = vision_enabled.then(|| {
            Arc::new(VisionEmbedder::new(data_directory.join("models").join("vision"), compute.ort_threads))
        });
        let (image_sender, image_receiver) = bounded::<(Arc<Shard>, i64, DynamicImage)>(1024);
        let (event_sender, events) = bounded(1024);
        let engine = Arc::new(Engine {
            config: config.clone(),
            data_directory,
            shards: RwLock::new(shards),
            events,
            event_sender: event_sender.clone(),
            embedder: Arc::clone(&embedder),
            neural_embedder,
            vision: vision.clone(),
            image_sender: vision_enabled.then_some(image_sender),
            transcriber: Arc::new(Mutex::new(None)),
            transcriber_failure: Arc::new(Mutex::new(None)),
            stop: Arc::new(AtomicBool::new(false)),
            scanning: Arc::new(AtomicBool::new(false)),
            scan_finished: Arc::new(AtomicBool::new(false)),
            idle_announced: Arc::new(AtomicBool::new(false)),
            estimator: Mutex::new(IndexingEstimator::default()),
            status_cache: Mutex::new(EngineStatus {
                scanning: false,
                scan_files_seen: 0,
                scan_files_per_second: 0.0,
                scan_eta_seconds: None,
                pending: 0,
                done: 0,
                failed: 0,
                index_files_per_second: 0.0,
                index_eta_seconds: None,
                total_files: 0,
                vector_count: 0,
                image_vector_count: 0,
                embedder_model: embedder.model_id().to_string(),
                neural_embedder,
                vision_model: None,
                whisper_model: config.whisper_model.clone(),
                transcriber_ready: false,
                core_count: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4),
                compute,
            }),
            scan_files_seen: Arc::new(AtomicU64::new(0)),
            scan_directories_seen: Arc::new(AtomicU64::new(0)),
            scan_total_hint: AtomicU64::new(0),
            scan_started: Mutex::new(None),
            watcher_stop: Mutex::new(None),
            compute,
            weak_self: OnceLock::new(),
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

        for shard in engine.shards_snapshot() {
            spawn_embed_writer(&engine, &shard);
        }
        let _ = engine.weak_self.set(Arc::downgrade(&engine));
        if vision_enabled {
            spawn_image_writer(&engine, image_receiver);
            spawn_vision_initializer(&engine, vision.expect("vision enabled"));
        }
        spawn_extraction_workers(&engine);
        spawn_progress_reporter(&engine);
        spawn_transcriber_initializer(&engine);
        restart_watcher(&engine);
        engine.spawn_scan();
        Ok(engine)
    }

    fn shards_snapshot(&self) -> Vec<Arc<Shard>> {
        self.shards.read().unwrap().clone()
    }

    /// Longest-prefix shard lookup for a filesystem path.
    fn shard_for_path(&self, path: &Path) -> Option<Arc<Shard>> {
        let text = path.to_string_lossy();
        self.shards_snapshot()
            .into_iter()
            .filter(|shard| text.starts_with(shard.root.to_string_lossy().as_ref()))
            .max_by_key(|shard| shard.root.as_os_str().len())
    }

    /// Kicks off a full rescan of all shards in parallel (no-op while one runs).
    pub fn spawn_scan(&self) {
        if self.scanning.swap(true, Ordering::AcqRel) {
            return;
        }
        self.scan_finished.store(false, Ordering::Release);
        let engine_shards = self.shards_snapshot();
        let hint: u64 = engine_shards.iter().map(|shard| shard.catalog.total_files().unwrap_or(0)).sum::<i64>() as u64;
        self.scan_total_hint.store(hint, Ordering::Release);
        self.scan_files_seen.store(0, Ordering::Release);
        self.scan_directories_seen.store(0, Ordering::Release);
        *self.scan_started.lock().unwrap() = Some(Instant::now());
        let scanning = Arc::clone(&self.scanning);
        let scan_finished = Arc::clone(&self.scan_finished);
        let files_seen = Arc::clone(&self.scan_files_seen);
        let directories_seen = Arc::clone(&self.scan_directories_seen);
        let event_sender = self.event_sender.clone();
        let roots: Vec<PathBuf> = engine_shards.iter().map(|shard| shard.root.clone()).collect();
        let skip_names = self.config.skip_dir_names.clone();
        let scan_threads = self.compute.scan_threads.max(1);
        let stop = Arc::clone(&self.stop);
        self.spawn_tracked("ipic-scan", move || {
            let _ = event_sender.send(EngineEvent::ScanStarted { roots: roots.clone() });
            let per_shard_threads = (scan_threads / engine_shards.len().max(1)).max(2);
        let mut total_files = 0u64;
        let mut total_dirs = 0u64;
        let started = Instant::now();
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(engine_shards.len());
            for shard in &engine_shards {
                let skip_names = skip_names.clone();
                let files_seen = Arc::clone(&files_seen);
                let directories_seen = Arc::clone(&directories_seen);
                let event_sender = event_sender.clone();
                let stop = Arc::clone(&stop);
                handles.push(scope.spawn(move || {
                    scan_shard(
                        shard, &skip_names, per_shard_threads, true, &files_seen,
                        &directories_seen, &event_sender, &stop,
                    )
                }));
            }
            for handle in handles {
                if let Ok(stats) = handle.join() {
                    total_files += stats.files;
                    total_dirs += stats.dirs;
                }
            }
        });
        scanning.store(false, Ordering::Release);
        scan_finished.store(true, Ordering::Release);
        let _ = event_sender.send(EngineEvent::ScanFinished {
            files_seen: total_files,
            directories_seen: total_dirs,
            elapsed_seconds: started.elapsed().as_secs_f64(),
        });
        let _ = event_sender.send(EngineEvent::CatalogChanged);
    });
    }

    pub fn core_count(&self) -> usize {
        std::thread::available_parallelism().map(|count| count.get()).unwrap_or(4)
    }

    pub fn compute_budget(&self) -> ComputeBudget {
        self.compute
    }

    /// Non-blocking event emit for background threads (telemetry may drop under pressure).
    fn emit(&self, event: EngineEvent) {
        let _ = self.event_sender.try_send(event);
    }

    /// Synchronous hybrid search across every shard (typically tens of ms).
    pub fn semantic_search(&self, query: &str, limit: usize) -> Result<SearchOutcome> {
        let started = Instant::now();
        let shards = self.shards_snapshot();
        if shards.is_empty() {
            return Ok(SearchOutcome {
                hits: Vec::new(),
                elapsed_millis: 0.0,
                interpreted_query: None,
                vector_count: 0,
            });
        }
        let depth = (limit * 4).clamp(48, 400);
        let text_query = self.embedder.embed_batch(&[query.to_string()])?.remove(0);
        let vision_query = self
            .vision
            .as_ref()
            .and_then(|vision| vision.query_embedding(query));
        let candidates: Vec<search::ShardCandidates> = shards
            .par_iter()
            .enumerate()
            .map(|(shard_index, shard)| {
                search::collect_shard_candidates(
                    shard,
                    shard_index,
                    &text_query,
                    vision_query.as_deref(),
                    query,
                    depth,
                )
            })
            .collect::<Result<_>>()?;
        let vector_count: u32 =
            shards.iter().map(|shard| shard.text_store.lock().unwrap().count()).sum();
        search::fuse_and_hydrate(&shards, candidates, limit, vector_count, started.elapsed())
    }

    /// Audio query: whisper transcription drives the usual lanes while the
    /// recording itself matches indexed audio acoustically (audio-to-audio).
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

        let acoustic_matches = self.audio_match(audio_pcm);
        if acoustic_matches.is_empty() {
            return Ok(outcome);
        }
        // Acoustic lane: global rank by similarity, reciprocal-rank contribution.
        let mut acoustic_scores: HashMap<String, f32> = HashMap::new();
        for (rank, (path, _similarity)) in acoustic_matches.iter().enumerate() {
            acoustic_scores.insert(path.clone(), ACOUSTIC_WEIGHT / (RRF_CONSTANT + rank as f32));
        }
        let mut merged = outcome.hits;
        for (path, acoustic) in acoustic_scores {
            let Some(shard) = self.shard_for_path(Path::new(&path)) else { continue };
            let Ok(connection) = shard.catalog.reader() else { continue };
            let Ok(Some((file_row, _))) = shard.catalog.file_by_path(&connection, &path) else { continue };
            match merged.iter_mut().find(|hit| hit.path == path) {
                Some(hit) => {
                    hit.score += acoustic;
                    hit.sources.acoustic = true;
                }
                None => merged.push(search::SearchHit {
                    file: file_row,
                    path,
                    snippet: "acoustic match".into(),
                    score: acoustic,
                    sources: MatchSources { acoustic: true, ..Default::default() },
                }),
            }
        }
        merged.sort_by(|left, right| right.score.total_cmp(&left.score));
        merged.truncate(limit);
        outcome.hits = merged;
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
        let threads_per_worker = (self.compute.extract_workers / self.config.whisper_workers.max(1)).max(1);
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
        // Cached by the 1 Hz reporter; only scan counters are read live so the
        // UI frame loop never touches SQLite.
        let mut status = self.status_cache.lock().unwrap().clone();
        status.scanning = self.scanning.load(Ordering::Relaxed);
        status.scan_files_seen = self.scan_files_seen.load(Ordering::Relaxed);
        status.scan_files_per_second = self.scan_rate();
        status.scan_eta_seconds = if status.scanning {
            scan_eta(status.scan_files_seen, self.scan_total_hint.load(Ordering::Relaxed), status.scan_files_per_second)
        } else {
            None
        };
        status
    }

    /// Recomputes the cached aggregate from every shard (reporter + shard changes).
    fn refresh_status_cache(&self) {
        let shards = self.shards_snapshot();
        let mut pending = 0i64;
        let mut done = 0i64;
        let mut failed = 0i64;
        let mut total_files = 0i64;
        let mut vector_count = 0u32;
        let mut image_vector_count = 0u32;
        let mut counters = Vec::new();
        for shard in &shards {
            if let Ok((shard_pending, _busy, shard_done, shard_failed)) = shard.catalog.rag_counters() {
                pending += shard_pending;
                done += shard_done;
                failed += shard_failed;
            }
            counters.extend(shard.catalog.rag_kind_counters().unwrap_or_default());
            total_files += shard.catalog.total_files().unwrap_or(0);
            vector_count += shard.text_store.lock().unwrap().count();
            image_vector_count += shard.image_store.lock().unwrap().count();
        }
        let estimate = self.estimator.lock().unwrap().latest(&counters);
        *self.status_cache.lock().unwrap() = EngineStatus {
            scanning: false, // patched live by status()
            scan_files_seen: 0,
            scan_files_per_second: 0.0,
            scan_eta_seconds: None,
            pending: pending as u64,
            done: done as u64,
            failed: failed as u64,
            index_files_per_second: estimate.map(|estimate| estimate.files_per_second).unwrap_or(0.0),
            index_eta_seconds: estimate.map(|estimate| estimate.eta_seconds),
            total_files,
            vector_count,
            image_vector_count,
            embedder_model: self.embedder.model_id().to_string(),
            neural_embedder: self.neural_embedder,
            vision_model: self.vision.as_ref().filter(|_| self.vision_ready()).map(|_| "clip-vit-b32".to_string()),
            whisper_model: self.config.whisper_model.clone(),
            transcriber_ready: self.transcriber.lock().unwrap().is_some(),
            core_count: self.core_count(),
            compute: self.compute,
        };
    }

    fn scan_rate(&self) -> f64 {
        let started = self.scan_started.lock().unwrap();
        let Some(started) = *started else { return 0.0 };
        let elapsed = started.elapsed().as_secs_f64();
        if elapsed < 0.25 {
            return 0.0;
        }
        self.scan_files_seen.load(Ordering::Relaxed) as f64 / elapsed
    }

    /// True when any background work exists (drives UI repaint cadence).
    pub fn has_background_work(&self) -> bool {
        self.scanning.load(Ordering::Relaxed) || self.status_cache.lock().unwrap().pending > 0
    }

    pub fn is_scanning(&self) -> bool {
        self.scanning.load(Ordering::Relaxed)
    }

    pub fn stop_background_work(&self) {
        self.stop.store(true, Ordering::Release);
    }

    fn vision_ready(&self) -> bool {
        self.vision.as_ref().is_some_and(|vision| vision.vision_ready())
    }

    /// Current scan roots (settings may swap them at runtime).
    pub fn current_roots(&self) -> Vec<PathBuf> {
        self.shards_snapshot().into_iter().map(|shard| shard.root.clone()).collect()
    }

    /// Swaps scan roots: missing shards open, removed roots unload (their
    /// shard data stays on disk and returns if the root is re-added).
    pub fn replace_roots(&self, roots: Vec<PathBuf>) {
        let engine = self
            .weak_self
            .get()
            .and_then(std::sync::Weak::upgrade)
            .expect("engine outlives its shards");
        let canonical = (Config { roots, ..self.config.clone() }).canonical_roots();
        let mut shards = self.shards.write().unwrap();
        let mut created = Vec::new();
        for root in &canonical {
            if !shards.iter().any(|shard| &shard.root == root) {
                match open_shard(&self.data_directory, root, self.embedder.dim(), self.embedder.model_id()) {
                    Ok(shard) => {
                        let shard = Arc::new(shard);
                        shards.push(Arc::clone(&shard));
                        created.push(shard);
                    }
                    Err(error) => self.emit(EngineEvent::Notice(format!(
                        "root {} unavailable: {error}",
                        root.display()
                    ))),
                }
            }
        }
        shards.retain(|shard| canonical.contains(&shard.root));
        drop(shards);
        for shard in created {
            spawn_embed_writer(&engine, &shard);
        }
        self.refresh_status_cache();
        restart_watcher(&engine);
    }

    /// Persists a configuration into this engine's data directory.
    pub fn persist_config(&self, config: &Config) {
        if let Err(error) = config.save_to_directory(&self.data_directory) {
            self.emit(EngineEvent::Notice(format!("config save failed: {error}")));
        }
    }

    /// Frees every derived slot a removal produced, across all three stores.
    fn release_removed_slots(&self, shard: &Shard, removed: &RemovedSlots) {
        Self::release_removed_slots_for(shard, removed);
    }

    /// Removes a file/subtree from its shard and frees every embedding.
    pub fn remove_path(&self, path: &str) {
        let Some(shard) = self.shard_for_path(Path::new(path)) else { return };
        if let Ok(removed) = shard.catalog.remove_path(path) {
            self.release_removed_slots(&shard, &removed);
            self.refresh_status_cache();
            self.emit(EngineEvent::CatalogChanged);
        }
    }

    /// Acoustic-similarity lane: audio query → matching audio/video files,
    /// fanned out across shards and hydrated to full paths.
    pub fn audio_match(&self, audio_pcm: &[f32]) -> Vec<(String, f32)> {
        let query = fingerprint::fingerprint_from_pcm(audio_pcm, extract::SAMPLE_RATE as u32);
        if query.is_empty() {
            return Vec::new();
        }
        let mut scored: Vec<(String, f32)> = Vec::new();
        for shard in self.shards_snapshot() {
            let matches = shard.audio_store.lock().unwrap().top_k(&query, 32);
            if matches.is_empty() {
                continue;
            }
            let Ok(connection) = shard.catalog.reader() else { continue };
            for (file_id, similarity) in matches {
                if let Ok(Some((_file_row, path))) = shard.catalog.file_by_id(&connection, file_id) {
                    scored.push((path, similarity));
                }
            }
        }
        scored.sort_by(|left, right| right.1.total_cmp(&left.1));
        scored.truncate(32);
        scored
    }

    /// First indexed chunk of a file (its extract or transcript).
    pub fn first_chunk_text(&self, file_id: i64, path: &str) -> Option<String> {
        let shard = self.shard_for_path(Path::new(path))?;
        let connection = shard.catalog.reader().ok()?;
        connection
            .query_row(
                "SELECT text FROM chunks WHERE file_id = ?1 ORDER BY rowid LIMIT 1",
                rusqlite::params![file_id],
                |row| row.get::<_, String>(0),
            )
            .ok()
    }

    /// File row lookup by absolute path.
    pub fn file_row_by_path(&self, path: &str) -> Option<FileRow> {
        let shard = self.shard_for_path(Path::new(path))?;
        let connection = shard.catalog.reader().ok()?;
        shard.catalog.file_by_path(&connection, path).ok().flatten().map(|(file, _)| file)
    }

    /// Media duration persistence (set once the transcript decode measured it).
    pub fn set_duration(&self, path: &str, seconds: f64) {
        let Some(shard) = self.shard_for_path(Path::new(path)) else { return };
        if let Some(file) = self.file_row_by_path(path) {
            shard.catalog.set_duration(file.id, seconds).ok();
        }
    }

    // ---- Browse routing (path-keyed across shards) ----

    pub fn dir_by_path(&self, path: &str) -> Option<DirRow> {
        let shard = self.shard_for_path(Path::new(path))?;
        let connection = shard.catalog.reader().ok()?;
        shard.catalog.dir_by_path(&connection, path).ok().flatten()
    }

    pub fn tree_children(&self, directory: &DirRow) -> Vec<DirRow> {
        let Some(shard) = self.shard_for_path(Path::new(&directory.path)) else { return Vec::new() };
        let Ok(connection) = shard.catalog.reader() else { return Vec::new() };
        shard.catalog.tree_children(&connection, Some(directory.id)).unwrap_or_default()
    }

    pub fn tree_children_empty(&self, directory: &DirRow) -> bool {
        self.tree_children(directory).is_empty()
    }

    /// Files of one directory (rows carry full paths), or the roots when no
    /// directory is open.
    pub fn directory_children(
        &self,
        directory: Option<&DirRow>,
        filter: &FileFilter,
        sort: SortKey,
        ascending: bool,
        limit: i64,
    ) -> (Vec<DirRow>, Vec<(FileRow, String)>) {
        match directory {
            None => {
                let roots = self
                    .shards_snapshot()
                    .into_iter()
                    .filter_map(|shard| {
                        let connection = shard.catalog.reader().ok()?;
                        shard.catalog.dir_by_path(&connection, &shard.root.to_string_lossy()).ok().flatten()
                    })
                    .collect();
                (roots, Vec::new())
            }
            Some(directory) => {
                let Some(shard) = self.shard_for_path(Path::new(&directory.path)) else {
                    return (Vec::new(), Vec::new());
                };
                let Ok(connection) = shard.catalog.reader() else { return (Vec::new(), Vec::new()) };
                let directories =
                    shard.catalog.tree_children(&connection, Some(directory.id)).unwrap_or_default();
                let files = shard
                    .catalog
                    .children(&connection, Some(directory.id), filter, sort, ascending, limit)
                    .unwrap_or_default();
                (directories, files)
            }
        }
    }

    pub fn kind_stats(&self) -> Vec<(FileKind, i64, i64)> {
        let mut totals: HashMap<FileKind, (i64, i64)> = HashMap::new();
        for shard in self.shards_snapshot() {
            let Ok(connection) = shard.catalog.reader() else { continue };
            for (kind, count, bytes) in shard.catalog.kind_stats(&connection).unwrap_or_default() {
                let entry = totals.entry(kind).or_insert((0, 0));
                entry.0 += count;
                entry.1 += bytes;
            }
        }
        let mut stats: Vec<_> = totals.into_iter().map(|(kind, (count, bytes))| (kind, count, bytes)).collect();
        stats.sort_by_key(|(kind, _, _)| *kind as i32);
        stats
    }

    /// Spawns a named worker thread tracked for shutdown.
    fn spawn_tracked<F: FnOnce() + Send + 'static>(&self, name: &str, body: F) {
        if let Ok(handle) = std::thread::Builder::new().name(name.into()).spawn(body) {
            self.track_worker(handle);
        }
    }
}

/// Walks one shard's root into its catalog. `full` wraps the seen-sweep
/// (incremental watcher scans skip it: rows must survive untouched).
#[allow(clippy::too_many_arguments)]
fn scan_shard(
    shard: &Arc<Shard>,
    skip_names: &[String],
    threads: usize,
    full: bool,
    files_seen: &Arc<AtomicU64>,
    directories_seen: &Arc<AtomicU64>,
    event_sender: &Sender<EngineEvent>,
    stop: &Arc<AtomicBool>,
) -> walker::WalkStats {
    if full {
        shard.catalog.begin_full_scan().ok();
    }
    let stats = collect_walk_items(
        shard,
        std::slice::from_ref(&shard.root),
        skip_names,
        threads,
        files_seen,
        directories_seen,
        event_sender,
        stop,
    );
    if full {
        if let Ok(removed) = shard.catalog.finish_full_scan() {
            Engine::release_removed_slots_for(shard, &removed);
        }
        let orphan_audio = shard.catalog.drain_orphan_derived(STORE_AUDIO).unwrap_or_default();
        shard.audio_store.lock().unwrap().free(&orphan_audio).ok();
        let orphan_image = shard.catalog.drain_orphan_derived(STORE_IMAGE).unwrap_or_default();
        shard.image_store.lock().unwrap().free(&orphan_image).ok();
        shard.catalog.analyze().ok();
    }
    stats
}

impl Engine {
    fn release_removed_slots_for(shard: &Shard, removed: &RemovedSlots) {
        if !removed.chunk_slots.is_empty() {
            shard.text_store.lock().unwrap().free(&removed.chunk_slots).ok();
        }
        let image_slots = removed.derived_in(STORE_IMAGE);
        if !image_slots.is_empty() {
            shard.image_store.lock().unwrap().free(&image_slots).ok();
        }
        let audio_slots = removed.derived_in(STORE_AUDIO);
        if !audio_slots.is_empty() {
            shard.audio_store.lock().unwrap().free(&audio_slots).ok();
        }
    }
}

/// Parallel walk + single-writer collector batching into one shard's SQLite.
#[allow(clippy::too_many_arguments)]
fn collect_walk_items(
    shard: &Arc<Shard>,
    roots: &[PathBuf],
    skip_names: &[String],
    threads: usize,
    global_files_seen: &Arc<AtomicU64>,
    global_directories_seen: &Arc<AtomicU64>,
    event_sender: &Sender<EngineEvent>,
    _stop: &AtomicBool,
) -> walker::WalkStats {
    let (item_sender, item_receiver) = unbounded::<WalkItem>();
    let catalog_writer = Arc::clone(shard);
    let global_files_seen = Arc::clone(global_files_seen);
    let global_directories_seen = Arc::clone(global_directories_seen);
    let event_sender_collector = event_sender.clone();
    let collector = std::thread::Builder::new().name("ipic-collect".into()).spawn(move || {
        let mut directory_ids: HashMap<String, i64> =
            catalog_writer.catalog.dir_id_map().unwrap_or_default();
        let mut pending_files: Vec<ipic_core::catalog::NewFile> = Vec::new();
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
                    catalog_writer.catalog.upsert_dirs(&[(path.clone(), mtime)]).ok();
                    let path_text = path.to_string_lossy().into_owned();
                    if let Some(directory_id) = catalog_writer.catalog.dir_id_by_path(&path_text) {
                        directory_ids.insert(path_text, directory_id);
                    }
                    global_directories_seen.fetch_add(1, Ordering::Relaxed);
                }
                WalkItem::File { path, kind, size, mtime } => {
                    let parent_path =
                        path.parent().map(|parent| parent.to_string_lossy().into_owned()).unwrap_or_default();
                    let directory_id = match directory_ids.get(&parent_path) {
                        Some(directory_id) => *directory_id,
                        None => catalog_writer.catalog.dir_id_by_path(&parent_path).unwrap_or(0),
                    };
                    if directory_id != 0 {
                        pending_files.push(ipic_core::catalog::NewFile {
                            dir_id: directory_id,
                            name: path.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default(),
                            kind,
                            size,
                            mtime,
                        });
                    }
                    let seen_now = global_files_seen.fetch_add(1, Ordering::Relaxed) + 1;
                    if seen_now >= next_progress_at || last_progress_time.elapsed().as_secs_f64() >= 2.0 {
                        next_progress_at = seen_now + 2_000;
                        last_progress_time = Instant::now();
                        let _ = event_sender_collector.send(EngineEvent::ScanProgress {
                            files_seen: seen_now,
                            directories_seen: global_directories_seen.load(Ordering::Relaxed),
                            elapsed_seconds: started.elapsed().as_secs_f64(),
                        });
                    }
                }
            }
            // Small batches keep each writer-mutex hold short enough for
            // worker claims to interleave (an 8k-row transaction starves them).
            if pending_files.len() >= 1024 {
                flush_files(&catalog_writer, &mut pending_files);
            }
        }
        flush_files(&catalog_writer, &mut pending_files);
    });
    let stats = walker::walk_parallel(roots, skip_names, threads, item_sender).unwrap_or_default();
    if let Ok(collector) = collector {
        let _ = collector.join();
    }
    stats
}

fn flush_files(shard: &Arc<Shard>, pending_files: &mut Vec<ipic_core::catalog::NewFile>) {
    if !pending_files.is_empty() {
        shard.catalog.upsert_files(pending_files).ok();
        pending_files.clear();
    }
}

/// Extraction workers self-serve jobs across every shard (natural
/// backpressure): text/PDF first (fast lane), images once CLIP is warm,
/// audio/video gated by the whisper slot semaphore. Claims travel in batches
/// so the per-claim transaction amortizes across many files.
const CLAIM_BATCH: i64 = 16;

fn spawn_extraction_workers(engine: &Arc<Engine>) {
    let worker_count = engine.compute.extract_workers;
    let whisper_permits = Arc::new((Mutex::new(0u32), Condvar::new()));
    let whisper_limit = engine.config.whisper_workers.max(1) as u32;
    for worker_index in 0..worker_count {
        let worker_engine = Arc::clone(engine);
        let whisper_permits = Arc::clone(&whisper_permits);
        engine.spawn_tracked(&format!("ipic-extract-{worker_index}"), move || {
            let mut local_queue: Vec<IndexingJob> = Vec::new();
            loop {
                if worker_engine.stop.load(Ordering::Relaxed) {
                    return;
                }
                if local_queue.is_empty() {
                    let fast = claim_jobs(&worker_engine, &[FileKind::Text, FileKind::Pdf])
                        .or_else(|| {
                            (worker_engine.vision_ready() || worker_engine.vision.is_none())
                                .then(|| claim_jobs(&worker_engine, &[FileKind::Image]))
                                .flatten()
                        })
                        .or_else(|| claim_jobs(&worker_engine, &[FileKind::Audio, FileKind::Video]));
                    match fast {
                        Some(jobs) => local_queue = jobs,
                        None => {
                            std::thread::sleep(Duration::from_millis(350));
                            continue;
                        }
                    }
                }
                // Whisper-gated media reorders to the back: a long transcription
                // must not hoard the locally claimed text jobs.
                let job = match local_queue
                    .iter()
                    .position(|job| !matches!(job.kind, FileKind::Audio | FileKind::Video))
                {
                    Some(position) => local_queue.swap_remove(position),
                    None => local_queue.pop().unwrap(),
                };
                process_job(&worker_engine, job, &whisper_permits, whisper_limit);
            }
        });
    }
}

fn claim_jobs(engine: &Arc<Engine>, kinds: &[FileKind]) -> Option<Vec<IndexingJob>> {
    for shard in engine.shards_snapshot() {
        let claimed = shard.catalog.claim_rag_jobs(kinds, CLAIM_BATCH).ok()?;
        if claimed.is_empty() {
            continue;
        }
        // One cleanup transaction per claimed batch (not per file): chunks and
        // derived vectors of a previous attempt go before the new extraction.
        let claimed_ids: Vec<i64> = claimed.iter().map(|(file_id, _)| *file_id).collect();
        if let Ok(removed) = shard.catalog.clear_file_derived(&claimed_ids) {
            Engine::release_removed_slots_for(&shard, &removed);
        }
        return Some(
            claimed
                .into_iter()
                .map(|(file_id, path)| {
                    let path = PathBuf::from(path);
                    let kind = ipic_core::kind_for_path(&path);
                    IndexingJob { shard: Arc::clone(&shard), file_id, path, kind }
                })
                .collect(),
        );
    }
    None
}

fn process_job(
    engine: &Arc<Engine>,
    job: IndexingJob,
    whisper_permits: &(Mutex<u32>, Condvar),
    whisper_limit: u32,
) {
    let IndexingJob { shard, file_id, path, kind } = job;
    let mark_done = |completed| {
        if completed {
            shard.catalog.set_rag(&[file_id], RagStatus::Done).ok();
        }
    };
    match kind {
        FileKind::Image => {
            let context = extract::extract_text(&path, kind, engine.config.max_text_mb * 1024 * 1024);
            let chunks = match context {
                Ok(Some(text)) => extract::chunk_text(&text),
                Ok(None) => Vec::new(),
                Err(_) => {
                    shard.catalog.set_rag(&[file_id], RagStatus::Failed).ok();
                    return;
                }
            };
            if engine.vision.is_some() {
                match extract::decode_image_thumbnail(&path, engine.config.max_image_mb * 1024 * 1024) {
                    Ok(Some(image)) => {
                        if !chunks.is_empty() {
                            let _ = shard.chunk_sender.send(ChunkDelivery { file_id, chunks, completion_pending: true });
                        }
                        if let Some(image_sender) = &engine.image_sender {
                            let _ = image_sender.send((Arc::clone(&shard), file_id, image));
                        } else {
                            mark_done(true);
                        }
                    }
                    Ok(None) => {
                        // No pixel content (svg): filename-context is the whole index.
                        if chunks.is_empty() {
                            mark_done(true);
                        } else {
                            let _ = shard.chunk_sender.send(ChunkDelivery { file_id, chunks, completion_pending: false });
                        }
                    }
                    Err(_) => {
                        shard.catalog.set_rag(&[file_id], RagStatus::Failed).ok();
                    }
                }
            } else if chunks.is_empty() {
                mark_done(true);
            } else {
                let _ = shard.chunk_sender.send(ChunkDelivery { file_id, chunks, completion_pending: false });
            }
        }
        FileKind::Text | FileKind::Pdf => {
            let extracted = extract::extract_text(&path, kind, engine.config.max_text_mb * 1024 * 1024);
            let chunks = match extracted {
                Ok(Some(text)) => extract::chunk_text(&text),
                Ok(None) => Vec::new(),
                Err(_) => {
                    shard.catalog.set_rag(&[file_id], RagStatus::Failed).ok();
                    return;
                }
            };
            if chunks.is_empty() {
                mark_done(true);
            } else {
                let _ = shard.chunk_sender.send(ChunkDelivery { file_id, chunks, completion_pending: false });
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
            let transcription = transcribe_media(engine, &shard, file_id, &path);
            let mut permits = whisper_permits.0.lock().unwrap();
            *permits -= 1;
            whisper_permits.1.notify_one();
            drop(permits);
            let chunks = match transcription {
                Ok(transcript) => extract::chunk_text(&transcript),
                Err(_) => {
                    shard.catalog.set_rag(&[file_id], RagStatus::Failed).ok();
                    return;
                }
            };
            if chunks.is_empty() {
                mark_done(true);
            } else {
                let _ = shard.chunk_sender.send(ChunkDelivery { file_id, chunks, completion_pending: false });
            }
        }
        _ => mark_done(true),
    }
}

fn transcribe_media(engine: &Arc<Engine>, shard: &Arc<Shard>, file_id: i64, path: &Path) -> Result<String> {
    let audio_pcm = extract::decode_speech_pcm(path)?;
    // Acoustic fingerprint: the audio-to-audio lane's index entry.
    let fingerprint = fingerprint::fingerprint_from_pcm(&audio_pcm, extract::SAMPLE_RATE as u32);
    if !fingerprint.is_empty() {
        let slots = shard.audio_store.lock().unwrap().append_batch(&[file_id], &[fingerprint])?;
        shard.catalog.set_derived_slot(STORE_AUDIO, file_id, slots[0]).ok();
    }
    // Sub-second clips carry no speech (UI sounds, game assets): the
    // fingerprint alone indexes them; whisper would only burn CPU.
    if audio_pcm.len() < extract::SAMPLE_RATE {
        return Ok(String::new());
    }
    shard
        .catalog
        .set_duration(file_id, audio_pcm.len() as f64 / extract::SAMPLE_RATE as f64)
        .ok();
    let transcriber = engine.ensure_transcriber()?;
    transcriber.transcribe(&audio_pcm)
}

/// Per-shard embed writer: batches chunks, embeds with ONNX (multi-threaded),
/// persists FTS rows + quantized vectors, then flips files to Done.
fn spawn_embed_writer(engine: &Arc<Engine>, shard: &Arc<Shard>) {
    let chunk_receiver = shard.take_chunk_receiver();
    let worker_engine = Arc::clone(engine);
    let shard = Arc::clone(shard);
    engine.spawn_tracked("ipic-embed", move || {
        let mut awaiting: Vec<ChunkDelivery> = Vec::new();
        loop {
            let received = chunk_receiver.recv_timeout(Duration::from_millis(120));
            match received {
                Ok(delivery) => awaiting.push(delivery),
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    if awaiting.is_empty() {
                        return;
                    }
                }
            }
            // Flush when the batch budget is hit, or when the queue has gone
            // idle (non-destructive emptiness check — never consume here).
            let budget_reached =
                awaiting.iter().map(|delivery| delivery.chunks.len()).sum::<usize>()
                    >= worker_engine.config.embed_batch;
            let queue_drained = chunk_receiver.is_empty();
            if !awaiting.is_empty() && (budget_reached || queue_drained) {
                flush_embeddings(&worker_engine, &shard, &mut awaiting);
            }
            if worker_engine.stop.load(Ordering::Relaxed) && awaiting.is_empty() {
                return;
            }
        }
    });
}

fn flush_embeddings(engine: &Arc<Engine>, shard: &Arc<Shard>, awaiting: &mut Vec<ChunkDelivery>) {
    let flattened: Vec<(i64, String)> = awaiting
        .iter()
        .flat_map(|delivery| {
            delivery.chunks.iter().map(move |chunk| (delivery.file_id, chunk.clone()))
        })
        .collect();
    if flattened.is_empty() {
        // No text at all: every delivery completes right here.
        let completed: Vec<i64> =
            awaiting.iter().filter(|delivery| !delivery.completion_pending).map(|d| d.file_id).collect();
        shard.catalog.set_rag(&completed, RagStatus::Done).ok();
        awaiting.clear();
        return;
    }
    let texts: Vec<String> = flattened.iter().map(|(_, text)| text.clone()).collect();
    let embedded = match engine.embedder.embed_batch(&texts) {
        Ok(embedded) => embedded,
        Err(_) => {
            let failed_files: Vec<i64> = awaiting.iter().map(|delivery| delivery.file_id).collect();
            shard.catalog.set_rag(&failed_files, RagStatus::Failed).ok();
            awaiting.clear();
            return;
        }
    };
    let references: Vec<(i64, &str)> = flattened.iter().map(|(file_id, text)| (*file_id, text.as_str())).collect();
    let rowids = match shard.catalog.insert_chunks(&references) {
        Ok(rowids) => rowids,
        Err(_) => {
            awaiting.clear();
            return;
        }
    };
    let slots = {
        let mut text_store = shard.text_store.lock().unwrap();
        text_store.append_batch(&rowids, &embedded).unwrap_or_default()
    };
    let assignments: Vec<(i64, i64)> = rowids.iter().cloned().zip(slots.iter().cloned()).collect();
    shard.catalog.assign_vector_slots(&assignments).ok();
    let completed: Vec<i64> =
        awaiting.iter().filter(|delivery| !delivery.completion_pending).map(|d| d.file_id).collect();
    shard.catalog.set_rag(&completed, RagStatus::Done).ok();
    awaiting.clear();
}

/// Single CLIP inference thread: decodes images arrive in parallel from the
/// extraction pool and leave in batches into their owning shards' stores.
fn spawn_image_writer(engine: &Arc<Engine>, image_receiver: Receiver<(Arc<Shard>, i64, DynamicImage)>) {
    let worker_engine = Arc::clone(engine);
    engine.spawn_tracked("ipic-image-embed", move || {
        let engine = worker_engine;
        let mut awaiting: Vec<(Arc<Shard>, i64, DynamicImage)> = Vec::new();
        loop {
            let received = image_receiver.recv_timeout(Duration::from_millis(150));
            match received {
                Ok(delivery) => awaiting.push(delivery),
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    if awaiting.is_empty() {
                        return;
                    }
                }
            }
            let queue_drained = image_receiver.is_empty();
            if !awaiting.is_empty() && (awaiting.len() >= IMAGE_BATCH || queue_drained) {
                flush_images(&engine, &mut awaiting);
            }
            if engine.stop.load(Ordering::Relaxed) && awaiting.is_empty() {
                return;
            }
        }
    });
}

fn flush_images(engine: &Arc<Engine>, awaiting: &mut Vec<(Arc<Shard>, i64, DynamicImage)>) {
    let Some(vision) = engine.vision.as_ref() else {
        awaiting.clear();
        return;
    };
    let images: Vec<DynamicImage> = awaiting.iter().map(|(_, _, image)| image.clone()).collect();
    let embedded = match vision.embed_images(images) {
        Ok(embedded) => embedded,
        Err(_) => {
            for (shard, file_id, _) in awaiting.iter() {
                shard.catalog.set_rag(&[*file_id], RagStatus::Failed).ok();
            }
            awaiting.clear();
            return;
        }
    };
    for ((shard, file_id, _), embedding) in awaiting.iter().zip(embedded) {
        match shard.image_store.lock().unwrap().append_batch(&[*file_id], &[embedding]) {
            Ok(slots) => {
                shard.catalog.set_derived_slot(STORE_IMAGE, *file_id, slots[0]).ok();
                shard.catalog.set_rag(&[*file_id], RagStatus::Done).ok();
            }
            Err(_) => {
                shard.catalog.set_rag(&[*file_id], RagStatus::Failed).ok();
            }
        }
    }
    awaiting.clear();
}

fn spawn_vision_initializer(engine: &Arc<Engine>, vision: Arc<VisionEmbedder>) {
    let worker_engine = Arc::clone(engine);
    engine.spawn_tracked("ipic-vision-init", move || {
        let engine = worker_engine;
        if let Err(error) = vision.load() {
            engine.emit(EngineEvent::VisionUnavailable { reason: error.to_string() });
        } else {
            engine.emit(EngineEvent::VisionReady { model_id: "clip-vit-b32".into() });
        }
    });
}

fn spawn_progress_reporter(engine: &Arc<Engine>) {
    let worker_engine = Arc::clone(engine);
    engine.spawn_tracked("ipic-progress", move || {
        let engine = worker_engine;
        loop {
            if engine.stop.load(Ordering::Relaxed) {
                return;
            }
            let counters: Vec<_> = engine
                .shards_snapshot()
                .iter()
                .flat_map(|shard| shard.catalog.rag_kind_counters().unwrap_or_default())
                .collect();
            let pending: i64 = counters.iter().map(|&(_, pending, _, _)| pending).sum();
            let done: i64 = counters.iter().map(|&(_, _, done, _)| done).sum();
            let failed: i64 = counters.iter().map(|&(_, _, _, failed)| failed).sum();
            let estimate = engine.estimator.lock().unwrap().tick(&counters, Instant::now());
            engine.emit(EngineEvent::IndexProgress {
                pending: pending as u64,
                done: done as u64,
                failed: failed as u64,
                files_per_second: estimate.map(|estimate| estimate.files_per_second).unwrap_or(0.0),
                eta_seconds: estimate.map(|estimate| estimate.eta_seconds),
            });
            engine.refresh_status_cache();
            let idle = pending == 0 && engine.scan_finished.load(Ordering::Relaxed);
            if idle && !engine.idle_announced.swap(true, Ordering::AcqRel) {
                engine.emit(EngineEvent::IndexingIdle);
            } else if pending > 0 {
                engine.idle_announced.store(false, Ordering::Release);
            }
            std::thread::sleep(Duration::from_secs(1));
        }
    });
}

fn spawn_transcriber_initializer(engine: &Arc<Engine>) {
    if engine.config.whisper_model == "none" {
        return; // whisper explicitly disabled (tests, text-only deployments)
    }
    let worker_engine = Arc::clone(engine);
    engine.spawn_tracked("ipic-whisper-init", move || {
        if let Err(error) = worker_engine.ensure_transcriber() {
            *worker_engine.transcriber_failure.lock().unwrap() = Some(error.to_string());
            worker_engine.emit(EngineEvent::TranscriberFailed { reason: error.to_string() });
        }
    });
}

/// One persistent watcher over the union of current roots; recreated (never
/// stacked) whenever roots change. Events route to shards by path prefix.
fn restart_watcher(engine: &Arc<Engine>) {
    let roots = engine.current_roots();
    if let Some(previous) = engine.watcher_stop.lock().unwrap().take() {
        previous.store(true, Ordering::Release);
    }
    if roots.is_empty() {
        return;
    }
    let (update_sender, update_receiver) = unbounded::<WatchUpdate>();
    let watcher_stop = watcher::spawn(roots, update_sender);
    *engine.watcher_stop.lock().unwrap() = Some(Arc::clone(&watcher_stop));
    let worker_engine = Arc::clone(engine);
    engine.spawn_tracked("ipic-watch", move || {
        let engine = worker_engine;
        loop {
            if engine.stop.load(Ordering::Relaxed) {
                watcher_stop.store(true, Ordering::Release);
                return;
            }
            match update_receiver.recv_timeout(Duration::from_millis(500)) {
                Ok(WatchUpdate::Vanished(path)) => {
                    let path_text = path.to_string_lossy().into_owned();
                    engine.remove_path(&path_text);
                }
                Ok(WatchUpdate::Appeared(path)) => handle_appeared(&engine, &path),
                Err(_) => {}
            }
        }
    });
}

fn handle_appeared(engine: &Arc<Engine>, path: &Path) {
    if !path.exists() {
        // FSEvents often reports deletions as modify events for the path —
        // whatever the event kind said, a path gone from disk is a removal.
        engine.remove_path(&path.to_string_lossy());
        return;
    }
    let Some(shard) = engine.shard_for_path(path) else { return };
    if path.is_dir() {
        // Directory replaced/renamed: drop stale rows, re-walk the subtree.
        if let Ok(removed) = shard.catalog.remove_path(&path.to_string_lossy()) {
            engine.release_removed_slots(&shard, &removed);
        }
        scan_shard_incremental(engine, &shard, path);
    } else if shard.catalog.upsert_single_file(path).is_ok() {
        engine.emit(EngineEvent::CatalogChanged);
    }
}

fn scan_shard_incremental(engine: &Arc<Engine>, shard: &Arc<Shard>, directory: &Path) {
    let files_seen = Arc::new(AtomicU64::new(0));
    let directories_seen = Arc::new(AtomicU64::new(0));
    let event_sender = engine.event_sender.clone();
    let stop = Arc::clone(&engine.stop);
    let stats = collect_walk_items(
        shard,
        &[directory.to_path_buf()],
        &engine.config.skip_dir_names,
        4,
        &files_seen,
        &directories_seen,
        &event_sender,
        &stop,
    );
    if stats.files > 0 {
        engine.emit(EngineEvent::CatalogChanged);
    }
}
