//! Text embedders: neural (fastembed/ONNX, fully local) with a zero-dependency
//! feature-hashing fallback so the pipeline still works offline / in tests.

use anyhow::{anyhow, Context, Result};
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use std::path::PathBuf;
use std::sync::Mutex;

pub trait TextEmbedder: Send + Sync {
    fn model_id(&self) -> &'static str;
    fn dim(&self) -> usize;
    /// Embeds a batch of texts into L2-normalized vectors (cosine == dot product).
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

/// BAAI/bge-small-en-v1.5 via ONNX Runtime (NEON-accelerated on arm64).
pub struct NeuralEmbedder {
    model: Mutex<TextEmbedding>,
}

impl NeuralEmbedder {
    pub fn load(cache_dir: PathBuf, show_download_progress: bool, threads: usize) -> Result<Self> {
        let model = TextEmbedding::try_new(
            TextInitOptions::new(EmbeddingModel::BGESmallENV15)
                .with_cache_dir(cache_dir)
                .with_show_download_progress(show_download_progress)
                .with_intra_threads(threads),
        )
        .map_err(|error| anyhow!("fastembed init failed: {error}"))?;
        Ok(Self { model: Mutex::new(model) })
    }
}

impl TextEmbedder for NeuralEmbedder {
    fn model_id(&self) -> &'static str {
        "bge-small-en-v1.5"
    }

    fn dim(&self) -> usize {
        384
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.model
            .lock()
            .unwrap()
            .embed(texts, Some(texts.len().min(32)))
            .map_err(|error| anyhow!("embed failed: {error}"))
            .context("neural embed")
    }
}

const HASH_DIM: usize = 512;

/// Deterministic feature-hashing embedder (word unigrams + bigrams, signed hashing).
/// Not semantic, but keeps hybrid search functional with zero downloads.
pub struct HashingEmbedder;

impl TextEmbedder for HashingEmbedder {
    fn model_id(&self) -> &'static str {
        "hashing-512-v1"
    }

    fn dim(&self) -> usize {
        HASH_DIM
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|text| hash_embed(text)).collect())
    }
}

fn fnv1a(bytes: &[u8]) -> u64 {
    // FNV-1a 64-bit: tiny, fast, good avalanche for feature hashing.
    bytes.iter().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ *byte as u64).wrapping_mul(0x100000001b3)
    })
}

fn hash_embed(text: &str) -> Vec<f32> {
    let mut vector = vec![0.0f32; HASH_DIM];
    let words: Vec<u64> = text
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 1)
        .map(|w| fnv1a(w.as_bytes()))
        .collect();
    let mut bump = |hash: u64, weight: f32| {
        let slot = (hash % HASH_DIM as u64) as usize;
        let sign = if (hash >> 63) & 1 == 0 { 1.0 } else { -1.0 };
        vector[slot] += sign * weight;
    };
    for &word in &words {
        bump(word, 1.0);
    }
    for pair in words.windows(2) {
        bump(pair[0] ^ pair[1].rotate_left(17), 0.5);
    }
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for value in &mut vector {
            *value /= norm;
        }
    }
    vector
}
