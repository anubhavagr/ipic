//! Whisper.cpp transcription (fully on-device, Metal-accelerated via the `metal`
//! feature). One shared WhisperContext; N worker states checked out of a pool.

use anyhow::{anyhow, Context, Result};
use crossbeam_channel::{Receiver, Sender};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState};

const WHISPER_BASE_URL: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";

/// Supported local models (ggml, q5_1 quantized: best size/accuracy tradeoff).
pub const WHISPER_MODELS: &[&str] = &[
    "tiny.en-q5_1",
    "base.en-q5_1",
    "small.en-q5_1",
    "tiny-q5_1",
    "base-q5_1",
];

pub fn whisper_model_path(model_name: &str) -> PathBuf {
    ipic_core::data_dir().join("models").join(format!("ggml-{model_name}.bin"))
}

/// Downloads the ggml model if missing, reporting progress through `on_bytes`.
pub fn ensure_model_downloaded(model_name: &str, mut on_bytes: impl FnMut(u64, Option<u64>) + Send) -> Result<PathBuf> {
    let target = whisper_model_path(model_name);
    if target.exists() {
        return Ok(target);
    }
    if !WHISPER_MODELS.contains(&model_name) {
        return Err(anyhow!("unknown whisper model '{model_name}'"));
    }
    std::fs::create_dir_all(target.parent().unwrap())?;
    let url = format!("{WHISPER_BASE_URL}/ggml-{model_name}.bin");
    let mut response = ureq::get(&url).call().map_err(|error| anyhow!("download failed: {error}"))?;
    let total = response
        .headers()
        .get("content-length")
        .and_then(|length| length.to_str().ok())
        .and_then(|length| length.parse::<u64>().ok());
    let mut reader = response.body_mut().as_reader();
    let mut file = std::io::BufWriter::new(std::fs::File::create(&target)?);
    let mut buffer = [0u8; 256 * 1024];
    let mut downloaded = 0u64;
    let mut last_report = 0u64;
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        std::io::Write::write_all(&mut file, &buffer[..read])?;
        downloaded += read as u64;
        if downloaded - last_report >= 512 * 1024 {
            on_bytes(downloaded, total);
            last_report = downloaded;
        }
    }
    drop(file);
    if downloaded < 1024 * 1024 {
        let _ = std::fs::remove_file(&target);
        return Err(anyhow!("model download too small, likely a network error"));
    }
    on_bytes(downloaded, total);
    Ok(target)
}

pub struct Transcriber {
    context: Arc<WhisperContext>,
    state_pool_tx: Sender<WhisperState>,
    state_pool_rx: Receiver<WhisperState>,
    threads_per_worker: i32,
}

impl Transcriber {
    /// Loads the model and prepares decode states sharing one context.
    /// GPU (Metal) is fast but only safe with a single concurrent state, so it
    /// forces serialization; the CPU path (Accelerate/BLAS) scales across workers.
    pub fn load(model_path: &Path, workers: usize, threads_per_worker: usize, use_gpu: bool) -> Result<Self> {
        let mut parameters = WhisperContextParameters::default();
        let (workers, threads_per_worker) = if use_gpu {
            parameters.use_gpu = true;
            (1, threads_per_worker.max(4))
        } else {
            parameters.use_gpu = false;
            (workers.max(1), threads_per_worker)
        };
        let context = WhisperContext::new_with_params(
            model_path.to_string_lossy().as_ref(),
            parameters,
        )
        .map_err(|error| anyhow!("whisper load failed: {error}"))
        .with_context(|| format!("loading {}", model_path.display()))?;
        let context = Arc::new(context);
        let (state_pool_tx, state_pool_rx) = crossbeam_channel::bounded::<WhisperState>(workers);
        for _ in 0..workers {
            let state = context.create_state().map_err(|error| anyhow!("state init failed: {error}"))?;
            state_pool_tx.send(state).map_err(|_| anyhow!("state pool closed"))?;
        }
        Ok(Self { context, state_pool_tx, state_pool_rx, threads_per_worker: threads_per_worker as i32 })
    }

    /// Transcribes 16 kHz mono f32 PCM to plain text (state checked out per call).
    pub fn transcribe(&self, pcm: &[f32]) -> Result<String> {
        if pcm.is_empty() {
            return Ok(String::new());
        }
        let mut state = self.state_pool_rx.recv().map_err(|_| anyhow!("state pool closed"))?;
        let result = transcribe_with_state(&mut state, pcm, self.threads_per_worker);
        let _ = self.state_pool_tx.send(state);
        result
    }

    /// For query-mode: a dedicated state is fine to keep permanently checked out
    /// while streaming; here we just expose the shared context for such callers.
    pub fn context(&self) -> Arc<WhisperContext> {
        Arc::clone(&self.context)
    }
}

pub fn transcribe_with_state(state: &mut WhisperState, pcm: &[f32], threads: i32) -> Result<String> {
    // Beam search: materially fewer word errors than greedy at modest cost.
    let mut params = FullParams::new(SamplingStrategy::BeamSearch { beam_size: 5, patience: -1.0 });
    params.set_n_threads(threads);
    params.set_language(Some("en"));
    params.set_print_progress(false);
    params.set_print_special(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_no_context(true);
    state
        .full(params, pcm)
        .map_err(|error| anyhow!("whisper decode failed: {error}"))?;
    let mut text = String::new();
    for segment_index in 0..state.full_n_segments() {
        if let Some(segment) = state.get_segment(segment_index)
            && let Ok(segment_text) = segment.to_str() {
                text.push_str(segment_text);
                text.push(' ');
            }
    }
    Ok(text.trim().to_string())
}
