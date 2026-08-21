//! ipic-rag: on-device multimodal RAG — embeddings, whisper ASR, quantized
//! vector index, hybrid search, and the engine that orchestrates indexing.

pub mod embed;
pub mod engine;
pub mod extract;
pub mod search;
pub mod transcribe;
pub mod vector_store;

pub use engine::{Engine, EngineEvent, EngineStatus};
pub use search::{SearchHit, SearchOutcome};
