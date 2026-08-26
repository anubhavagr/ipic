//! Decentralized index: one shard per root. Each shard owns its SQLite
//! catalog, its embed-writer thread, and its three quantized vector stores —
//! roots scan, write and serve queries in parallel with no shared mutex.

use crate::fingerprint::FINGERPRINT_DIM;
use crate::vector_store::VectorStore;
use crate::vision::{CLIP_DIM, VISION_MODEL_ID};
use anyhow::Result;
use ipic_core::catalog::{Catalog, STORE_AUDIO, STORE_IMAGE};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub const AUDIO_MODEL_ID: &str = "audio-fingerprint-v1";

pub struct Shard {
    pub root: PathBuf,
    pub directory: PathBuf,
    pub catalog: Catalog,
    pub text_store: Mutex<VectorStore>,
    pub image_store: Mutex<VectorStore>,
    pub audio_store: Mutex<VectorStore>,
    /// Feed for this shard's embed-writer thread.
    pub chunk_sender: crossbeam_channel::Sender<ChunkDelivery>,
    chunk_receiver: Mutex<Option<crossbeam_channel::Receiver<ChunkDelivery>>>,
}

impl Shard {
    /// The embed-writer thread takes the receiving end exactly once.
    pub fn take_chunk_receiver(&self) -> crossbeam_channel::Receiver<ChunkDelivery> {
        self.chunk_receiver
            .lock()
            .unwrap()
            .take()
            .expect("chunk receiver is taken once per shard")
    }
}

/// One file's extracted text bound for the embed writer.
pub struct ChunkDelivery {
    pub file_id: i64,
    pub chunks: Vec<String>,
    /// False while a second stage (pixel embedding) still owes the Done mark.
    pub completion_pending: bool,
}

pub fn shards_directory(data_directory: &Path) -> PathBuf {
    data_directory.join("shards")
}

/// Stable directory name: readable segment + path hash (collision-safe).
pub fn shard_key(root: &Path) -> String {
    let hash = root.to_string_lossy().bytes().fold(0xcbf29ce484222325u64, |hash, byte| {
        (hash ^ byte as u64).wrapping_mul(0x100000001b3)
    });
    let segment = root
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "root".into());
    format!("{segment}-{hash:016x}")
}

/// Opens (creating if needed) the shard owning `root`. Incompatible store
/// headers (model/dim change) wipe derived data and re-queue their files.
pub fn open_shard(
    data_directory: &Path,
    root: &Path,
    text_dim: usize,
    text_model_id: &str,
) -> Result<Shard> {
    let directory = shards_directory(data_directory).join(shard_key(root));
    std::fs::create_dir_all(&directory)?;
    let catalog = Catalog::open(&directory.join("catalog.db"))?;

    let (mut text_store, text_compatible) = VectorStore::open(&directory, text_dim, text_model_id)?;
    let (mut image_store, image_compatible) =
        VectorStore::open(&directory.join("image-vectors"), CLIP_DIM, VISION_MODEL_ID)?;
    let (mut audio_store, audio_compatible) =
        VectorStore::open(&directory.join("audio-fingerprints"), FINGERPRINT_DIM, AUDIO_MODEL_ID)?;

    if !text_compatible {
        // Text embedder changed: every derived artifact belongs to the old space.
        catalog.clear_chunks()?;
        text_store.clear()?;
        image_store.clear()?;
        audio_store.clear()?;
    } else {
        let orphan_slots = catalog.delete_orphan_chunks().unwrap_or_default();
        text_store.free(&orphan_slots)?;
        if !image_compatible {
            catalog.drain_derived(STORE_IMAGE)?;
            image_store.clear()?;
        }
        if !audio_compatible {
            catalog.drain_derived(STORE_AUDIO)?;
            audio_store.clear()?;
        }
        let orphans = catalog.drain_orphan_derived(STORE_IMAGE)?;
        image_store.free(&orphans)?;
        let orphans = catalog.drain_orphan_derived(STORE_AUDIO)?;
        audio_store.free(&orphans)?;
    }

    // Bounded feeds give the pipeline natural backpressure: extraction blocks
    // instead of buffering unbounded chunk text in RAM when embedding lags.
    const CHUNK_QUEUE_BOUND: usize = 8192;
    let (chunk_sender, chunk_receiver) =
        crossbeam_channel::bounded::<ChunkDelivery>(CHUNK_QUEUE_BOUND);
    Ok(Shard {
        root: root.to_path_buf(),
        directory,
        catalog,
        text_store: Mutex::new(text_store),
        image_store: Mutex::new(image_store),
        audio_store: Mutex::new(audio_store),
        chunk_sender,
        chunk_receiver: Mutex::new(Some(chunk_receiver)),
    })
}

/// Moves the pre-sharding single-database layout into the first root's shard
/// when every indexed path lives under it; otherwise archives it for a clean
/// reindex (mixed-root legacy rows cannot be split safely).
pub fn migrate_legacy_layout(data_directory: &Path, roots: &[PathBuf]) -> Result<()> {
    let legacy_database = data_directory.join("catalog.db");
    if !legacy_database.exists() || roots.is_empty() {
        return Ok(());
    }
    let shards = shards_directory(data_directory);
    std::fs::create_dir_all(&shards)?;

    let movable = {
        let connection = rusqlite::Connection::open(&legacy_database)?;
        let first_root = roots[0].to_string_lossy().into_owned();
        // Ancestor directories of the root are structural, not foreign data;
        // only paths outside the root's subtree force an archive + reindex.
        let outside: i64 = connection.query_row(
            "SELECT COUNT(*) FROM dirs
             WHERE path != ?1 AND path NOT LIKE ?1 || '/%' AND ?1 NOT LIKE path || '/%'",
            rusqlite::params![first_root],
            |row| row.get(0),
        )?;
        outside == 0
    };

    let target = if movable {
        shards.join(shard_key(&roots[0]))
    } else {
        shards.join("legacy-backup")
    };
    std::fs::create_dir_all(&target)?;
    for name in ["catalog.db", "vectors.bin", "free-slots.bin", "audio-fingerprints"] {
        let source = data_directory.join(name);
        if source.exists() {
            std::fs::rename(source, target.join(name))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_keys_are_stable_and_collision_safe() {
        let downloads = Path::new("/Users/x/Downloads");
        assert_eq!(shard_key(downloads), shard_key(downloads));
        assert_ne!(shard_key(downloads), shard_key(Path::new("/Users/y/Downloads")));
        assert!(shard_key(downloads).starts_with("Downloads-"));
    }
}
