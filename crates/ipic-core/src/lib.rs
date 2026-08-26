//! ipic-core: parallel filesystem scanning, SQLite catalog, change watching.

pub mod catalog;
pub mod util;
pub mod walker;
pub mod watcher;

use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::{Path, PathBuf};

pub type CoreResult<T> = anyhow::Result<T>;

/// File categories. Text/Pdf/Audio/Video/Image are RAG-indexable (images via
/// filename + folder context); the rest are catalog-only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FileKind {
    Text,
    Pdf,
    Audio,
    Video,
    Image,
    Other,
}

impl FileKind {
    pub const ALL: [FileKind; 6] = [Self::Text, Self::Pdf, Self::Audio, Self::Video, Self::Image, Self::Other];
    pub fn is_rag(self) -> bool {
        matches!(self, Self::Text | Self::Pdf | Self::Audio | Self::Video | Self::Image)
    }
    pub fn label(self) -> &'static str {
        match self {
            Self::Text => "Text",
            Self::Pdf => "PDF",
            Self::Audio => "Audio",
            Self::Video => "Video",
            Self::Image => "Image",
            Self::Other => "Other",
        }
    }
    fn from_token(token: &str) -> Self {
        match token {
            "text" => Self::Text,
            "pdf" => Self::Pdf,
            "audio" => Self::Audio,
            "video" => Self::Video,
            "image" => Self::Image,
            _ => Self::Other,
        }
    }
    fn token(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Pdf => "pdf",
            Self::Audio => "audio",
            Self::Video => "video",
            Self::Image => "image",
            Self::Other => "other",
        }
    }
}

impl fmt::Display for FileKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

pub fn kind_for_path(path: &Path) -> FileKind {
    let ext = path.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    kind_for_ext(&ext)
}

pub fn kind_for_ext(ext: &str) -> FileKind {
    const TEXT: &[&str] = &["txt", "md", "markdown", "rst", "tex", "log", "csv", "tsv", "json", "yaml", "yml", "toml", "xml", "html", "htm", "srt", "vtt", "org", "adoc", "ini", "cfg", "conf", "sql", "rs", "py", "js", "jsx", "ts", "tsx", "c", "h", "cpp", "hpp", "cc", "java", "kt", "swift", "go", "rb", "sh", "zsh", "bash", "php", "pl", "lua", "vim", "clj", "ex", "exs", "hs", "ml", "fs", "cs", "scala", "dart", "nim", "zig", "vue", "svelte", "graphql", "proto"];
    const AUDIO: &[&str] = &["mp3", "wav", "flac", "ogg", "oga", "opus", "m4a", "aac", "wma", "aiff", "aif", "alac", "amr", "caf"];
    const VIDEO: &[&str] = &["mp4", "m4v", "mov", "mkv", "webm", "avi", "wmv", "flv", "mpg", "mpeg", "mts", "ts", "m2ts", "3gp"];
    const IMAGE: &[&str] = &["png", "jpg", "jpeg", "gif", "webp", "bmp", "tiff", "tif", "heic", "heif", "avif", "svg", "ico"];
    if ext == "pdf" {
        FileKind::Pdf
    } else if TEXT.contains(&ext) {
        FileKind::Text
    } else if AUDIO.contains(&ext) {
        FileKind::Audio
    } else if VIDEO.contains(&ext) {
        FileKind::Video
    } else if IMAGE.contains(&ext) {
        FileKind::Image
    } else {
        FileKind::Other
    }
}

/// RAG indexing lifecycle persisted per file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i64)]
pub enum RagStatus {
    Pending = 0,
    Busy = 1,
    Done = 2,
    Failed = 3,
}

/// One catalog row for a file.
#[derive(Debug, Clone)]
pub struct FileRow {
    pub id: i64,
    pub dir_id: i64,
    pub name: String,
    pub kind: FileKind,
    pub size: i64,
    pub mtime: i64,
    pub duration_secs: Option<f64>,
    pub rag: RagStatus,
}

/// One catalog row for a directory.
#[derive(Debug, Clone)]
pub struct DirRow {
    pub id: i64,
    pub parent_id: Option<i64>,
    pub name: String,
    pub path: String,
    pub file_count: i64,
    pub subdir_count: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortKey {
    Name,
    Kind,
    Size,
    Modified,
    Duration,
}

/// Metadata filters applied in SQL when listing a directory.
#[derive(Debug, Clone, Default)]
pub struct FileFilter {
    pub name_query: Option<String>,
    pub kinds: Vec<FileKind>,
    pub min_size: Option<i64>,
    pub max_size: Option<i64>,
    pub after_unix: Option<i64>,
    pub before_unix: Option<i64>,
}

impl FileFilter {
    pub fn is_empty(&self) -> bool {
        self.name_query.is_none() && self.kinds.is_empty() && self.min_size.is_none()
            && self.max_size.is_none() && self.after_unix.is_none() && self.before_unix.is_none()
    }
}

/// User configuration persisted at ~/.ipic/config.toml.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub roots: Vec<PathBuf>,
    pub skip_dir_names: Vec<String>,
    pub whisper_model: String,
    pub whisper_workers: usize,
    /// GPU (Metal) whisper: very fast but transcriptions serialize on one state.
    #[serde(default)]
    pub whisper_use_gpu: bool,
    /// "neural" (default) or "hashing" (offline lexical fallback, no download).
    #[serde(default = "default_embedder")]
    pub embedder: String,
    /// "clip-vit-b32" (default) embeds image pixels; "none" keeps filename-only.
    #[serde(default = "default_image_embedder")]
    pub image_embedder: String,
    pub embed_batch: usize,
    pub max_text_mb: usize,
    /// Images above this size are skipped for pixel embedding (decode guard).
    #[serde(default = "default_max_image_mb")]
    pub max_image_mb: usize,
    /// Thread budget per pipeline stage; 0 entries derive saturating defaults.
    #[serde(default)]
    pub compute: ComputeConfig,
}

fn default_embedder() -> String {
    "neural".into()
}

fn default_image_embedder() -> String {
    "clip-vit-b32".into()
}

fn default_max_image_mb() -> usize {
    64
}

/// Per-stage thread allocation. Zero means "derive from the machine": the
/// resolved budget saturates every core — idle hardware is waste — while the
/// user can cap any stage explicitly.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct ComputeConfig {
    pub max_cores: usize,
    pub scan_threads: usize,
    pub extract_workers: usize,
    pub ort_threads: usize,
    pub search_threads: usize,
}

/// Concrete thread counts resolved from [`ComputeConfig`].
#[derive(Debug, Clone, Copy)]
pub struct ComputeBudget {
    pub scan_threads: usize,
    pub extract_workers: usize,
    pub ort_threads: usize,
    pub search_threads: usize,
}

impl ComputeConfig {
    pub fn resolve(&self) -> ComputeBudget {
        let logical = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let physical = (logical / 2).max(1);
        let cap = if self.max_cores == 0 { logical } else { self.max_cores.min(logical) };
        let at_least_one = |value: usize| value.max(1);
        ComputeBudget {
            scan_threads: at_least_one(if self.scan_threads == 0 { cap } else { self.scan_threads.min(cap) }),
            // Leave one core for the UI compositor; everything else extracts.
            extract_workers: at_least_one(
                if self.extract_workers == 0 { cap.saturating_sub(1).max(2) } else { self.extract_workers.min(cap) },
            ),
            // ONNX intra-op shares the machine with the decode workers.
            ort_threads: at_least_one(if self.ort_threads == 0 { physical } else { self.ort_threads.min(cap) }),
            search_threads: at_least_one(if self.search_threads == 0 { cap } else { self.search_threads.min(cap) }),
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            roots: vec![dirs::home_dir().unwrap_or_else(|| PathBuf::from("/")).join("Downloads")],
            skip_dir_names: [
                ".Trash", "Library", "Caches", "node_modules", ".git", ".cache", ".venv",
                "__pycache__", "target", "dist", ".npm", ".cargo", ".rustup", ".docker",
            ].iter().map(|s| s.to_string()).collect(),
            whisper_model: "small.en-q5_1".into(),
            whisper_workers: 3,
            whisper_use_gpu: false,
            embedder: default_embedder(),
            image_embedder: default_image_embedder(),
            embed_batch: 64,
            max_text_mb: 8,
            max_image_mb: default_max_image_mb(),
            compute: ComputeConfig::default(),
        }
    }
}

/// ~/.ipic/ data directory, created on demand.
pub fn data_dir() -> PathBuf {
    let dir = dirs::home_dir().unwrap_or_else(|| PathBuf::from(".")).join(".ipic");
    std::fs::create_dir_all(&dir).ok();
    dir
}

impl Config {
    pub fn path() -> PathBuf {
        data_dir().join("config.toml")
    }

    /// Config lives beside the data it configures (tests use temp data dirs).
    pub fn path_in(directory: &Path) -> PathBuf {
        directory.join("config.toml")
    }

    /// Non-overlapping canonical roots (nested roots are redundant work).
    pub fn canonical_roots(&self) -> Vec<PathBuf> {
        let mut roots: Vec<PathBuf> = self
            .roots
            .iter()
            .filter_map(|root| std::fs::canonicalize(root).ok())
            .filter(|root| root.is_dir())
            .collect();
        roots.sort_unstable();
        roots.dedup();
        let mut compact: Vec<PathBuf> = Vec::new();
        for root in roots {
            if !compact.iter().any(|kept| root.starts_with(kept)) {
                compact.push(root);
            }
        }
        compact
    }

    /// Loads config from disk, creating the default on first run.
    pub fn load_or_create() -> CoreResult<Self> {
        let path = Self::path();
        if !path.exists() {
            let config = Self::default();
            config.save()?;
            return Ok(config);
        }
        let config: Config = toml::from_str(&std::fs::read_to_string(&path)?)?;
        Ok(config)
    }

    pub fn save(&self) -> CoreResult<()> {
        self.save_to_directory(&data_dir())
    }

    pub fn save_to_directory(&self, directory: &Path) -> CoreResult<()> {
        std::fs::create_dir_all(directory)?;
        std::fs::write(Self::path_in(directory), toml::to_string_pretty(self)?)?;
        Ok(())
    }
}
