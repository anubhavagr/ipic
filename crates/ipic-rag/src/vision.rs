//! CLIP towers (ViT-B/32): image pixels and typed queries embed into the same
//! 512-dim latent space, so "sunset" matches photographs by content.

use anyhow::{anyhow, Result};
use fastembed::{
    EmbeddingModel, ImageEmbedding, ImageEmbeddingModel, ImageInitOptions, TextEmbedding,
    TextInitOptions,
};
use image::DynamicImage;
use std::path::PathBuf;
use std::sync::Mutex;

pub const CLIP_DIM: usize = 512;
pub const VISION_MODEL_ID: &str = "clip-vit-b32-vision";
pub const CLIP_TEXT_MODEL_ID: &str = "clip-vit-b32-text";

/// Both towers share one struct: one model identity, lazily downloaded.
/// Inference is mutex-serialized (decode/extract parallelize outside).
pub struct VisionEmbedder {
    cache_dir: PathBuf,
    ort_threads: usize,
    text_tower: Mutex<Option<TextEmbedding>>,
    vision_tower: Mutex<Option<ImageEmbedding>>,
}

impl VisionEmbedder {
    pub fn new(cache_dir: PathBuf, ort_threads: usize) -> Self {
        Self { cache_dir, ort_threads, text_tower: Mutex::new(None), vision_tower: Mutex::new(None) }
    }

    /// Downloads/loads both towers; call once from a background initializer.
    pub fn load(&self) -> Result<()> {
        self.load_text_tower()?;
        self.load_vision_tower()?;
        Ok(())
    }

    pub fn load_text_tower(&self) -> Result<()> {
        let mut slot = self.text_tower.lock().unwrap();
        if slot.is_some() {
            return Ok(());
        }
        let model = TextEmbedding::try_new(
            TextInitOptions::new(EmbeddingModel::ClipVitB32)
                .with_cache_dir(self.cache_dir.clone())
                .with_intra_threads(self.ort_threads),
        )
        .map_err(|error| anyhow!("clip text tower init failed: {error}"))?;
        *slot = Some(model);
        Ok(())
    }

    pub fn load_vision_tower(&self) -> Result<()> {
        let mut slot = self.vision_tower.lock().unwrap();
        if slot.is_some() {
            return Ok(());
        }
        let model = ImageEmbedding::try_new(
            ImageInitOptions::new(ImageEmbeddingModel::ClipVitB32)
                .with_cache_dir(self.cache_dir.clone())
                .with_intra_threads(self.ort_threads),
        )
        .map_err(|error| anyhow!("clip vision tower init failed: {error}"))?;
        *slot = Some(model);
        Ok(())
    }

    pub fn towers_ready(&self) -> bool {
        self.text_tower.lock().unwrap().is_some() && self.vision_tower.lock().unwrap().is_some()
    }

    pub fn vision_ready(&self) -> bool {
        self.vision_tower.lock().unwrap().is_some()
    }

    /// Query embedding for the vision lane; None while the tower is still loading.
    pub fn query_embedding(&self, query: &str) -> Option<Vec<f32>> {
        let mut guard = self.text_tower.lock().unwrap();
        let model = guard.as_mut()?;
        model.embed(vec![query.to_string()], Some(1)).ok()?.into_iter().next()
    }

    /// Embeds a batch of pre-resized images (the parallel decode already ran).
    pub fn embed_images(&self, images: Vec<DynamicImage>) -> Result<Vec<Vec<f32>>> {
        let mut guard = self.vision_tower.lock().unwrap();
        let model = guard.as_mut().ok_or_else(|| anyhow!("clip vision tower not loaded"))?;
        model.embed_images(images).map_err(|error| anyhow!("clip embed failed: {error}"))
    }
}
