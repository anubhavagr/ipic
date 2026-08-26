//! ipic-rag: on-device multimodal RAG — embeddings (text + CLIP vision),
//! whisper ASR, quantized vector stores, per-root shards, hybrid search, and
//! the engine that orchestrates indexing.

pub mod embed;
pub mod engine;
pub mod extract;
pub mod fingerprint;
pub mod progress;
pub mod search;
pub mod shard;
pub mod transcribe;
pub mod vector_store;
pub mod vision;

pub use engine::{Engine, EngineEvent, EngineStatus};
pub use search::{SearchHit, SearchOutcome};
