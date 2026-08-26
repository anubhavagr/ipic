//! ipic-cli: headless driver for the ipic engine — scan, index, search, status.

use clap::{Parser, Subcommand};
use ipic_core::{Config, FileKind, SortKey};
use ipic_rag::{Engine, EngineEvent};
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Parser)]
#[command(name = "ipic-cli", version, about = "Local multimodal file search & indexing")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Scan configured roots and index everything, then exit when idle.
    Scan {
        /// Additional roots to index for this run only.
        #[arg(long)]
        root: Option<PathBuf>,
        /// Keep running and watch for changes instead of exiting.
        #[arg(long)]
        watch: bool,
    },
    /// Hybrid semantic + keyword search.
    Search {
        query: String,
        /// Maximum results.
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    /// Ask with an audio file as the query (locally transcribed).
    Ask {
        /// Audio file containing the spoken query.
        #[arg(long)]
        audio: PathBuf,
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    /// Engine and index statistics.
    Status,
    /// List directory contents with filters/sorting (classic file-manager view).
    List {
        directory: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = KindFilter::All)]
        kind: KindFilter,
        #[arg(long, default_value_t = 500)]
        limit: i64,
    },
    /// Print / edit effective configuration.
    Config {
        /// One or more roots (repeats also accepted: --set-roots a --set-roots b).
        #[arg(long, num_args = 1..)]
        set_roots: Option<Vec<PathBuf>>,
    },
}

#[derive(clap::ValueEnum, Clone, Copy)]
enum KindFilter {
    All,
    Text,
    Pdf,
    Audio,
    Video,
    Image,
}

impl KindFilter {
    fn kinds(self) -> Vec<FileKind> {
        match self {
            KindFilter::All => Vec::new(),
            KindFilter::Text => vec![FileKind::Text],
            KindFilter::Pdf => vec![FileKind::Pdf],
            KindFilter::Audio => vec![FileKind::Audio],
            KindFilter::Video => vec![FileKind::Video],
            KindFilter::Image => vec![FileKind::Image],
        }
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Config { set_roots } => run_config(set_roots),
        Command::List { directory, kind, limit } => run_list(directory, kind, limit),
        Command::Status => run_status(),
        Command::Search { query, limit } => run_search(&query, limit, None),
        Command::Ask { audio, limit } => run_ask(audio, limit),
        Command::Scan { root, watch } => run_scan(root, watch),
    }
}

fn run_config(set_roots: Option<Vec<PathBuf>>) -> anyhow::Result<()> {
    let mut config = Config::load_or_create()?;
    if let Some(roots) = set_roots {
        config.roots = roots;
        config.save()?;
    }
    println!("{}", toml::to_string_pretty(&config)?);
    Ok(())
}

fn run_list(directory: Option<PathBuf>, kind: KindFilter, limit: i64) -> anyhow::Result<()> {
    let config = Config::load_or_create()?;
    let mut directory = directory;
    if let Some(path) = &directory {
        directory = Some(path.canonicalize()?);
    }
    // A read-only engine view over the shards (no scan: launch then stop it).
    let mut quiet_config = config.clone();
    quiet_config.whisper_model = "none".into();
    let engine = Engine::launch_with_data_dir(quiet_config, ipic_core::data_dir())?;
    engine.stop_background_work();
    let directory_row = match &directory {
        Some(directory) => engine.dir_by_path(&directory.to_string_lossy()),
        None => None,
    };
    let mut entries: Vec<(ipic_core::FileRow, String)> = Vec::new();
    if let Some(row) = &directory_row {
        let (_dirs, files) = engine.directory_children(
            Some(row),
            &ipic_core::FileFilter { kinds: kind.kinds(), ..Default::default() },
            SortKey::Name,
            true,
            limit,
        );
        entries = files;
    } else {
        for root in engine.current_roots() {
            if let Some(row) = engine.dir_by_path(&root.to_string_lossy()) {
                let (_dirs, files) = engine.directory_children(
                    Some(&row),
                    &ipic_core::FileFilter { kinds: kind.kinds(), ..Default::default() },
                    SortKey::Name,
                    true,
                    limit,
                );
                entries.extend(files);
            }
        }
    }
    let directory_display = directory.unwrap_or_else(|| PathBuf::from("(entire catalog)"));
    for (file_row, path) in entries {
        println!(
            "{:<6} {:>10}  {}",
            file_row.kind.label(),
            ipic_core::util::format_size(file_row.size),
            path
        );
    }
    let _ = directory_display;
    engine.shutdown();
    Ok(())
}

fn run_status() -> anyhow::Result<()> {
    let engine = Engine::launch(Config::load_or_create()?)?;
    // Give the boot events a moment to land, then report.
    std::thread::sleep(Duration::from_millis(600));
    while let Ok(event) = engine.events.try_recv() {
        print_event(&event);
    }
    let status = engine.status();
    println!("files indexed      : {}", status.total_files);
    println!("rag pending        : {}", status.pending);
    println!("rag completed      : {}", status.done);
    println!("rag failed         : {}", status.failed);
    println!("text vectors       : {}", status.vector_count);
    println!("image vectors      : {}", status.image_vector_count);
    println!("embedder           : {}{}", status.embedder_model, if status.neural_embedder { " (neural)" } else { " (lexical fallback)" });
    println!("vision             : {}", status.vision_model.as_deref().unwrap_or("off"));
    println!("whisper            : {}{}", status.whisper_model, if status.transcriber_ready { " (ready)" } else { " (loading)" });
    println!("cpu cores used     : {}", status.core_count);
    println!(
        "compute budget     : scan {} · extract {} · onnx {} · search {}",
        status.compute.scan_threads, status.compute.extract_workers, status.compute.ort_threads, status.compute.search_threads
    );
    engine.shutdown();
    Ok(())
}

fn run_search(query: &str, limit: usize, spoken_pcm: Option<Vec<f32>>) -> anyhow::Result<()> {
    let engine = Engine::launch(Config::load_or_create()?)?;
    // Let resume/scan settle briefly so fresh installs return something.
    std::thread::sleep(Duration::from_millis(300));
    while let Ok(event) = engine.events.try_recv() {
        print_event(&event);
    }
    let outcome = match spoken_pcm {
        Some(audio_pcm) => engine.spoken_query_search(&audio_pcm, limit)?,
        None => engine.semantic_search(query, limit)?,
    };
    if let Some(transcript) = &outcome.interpreted_query {
        println!("heard: \"{transcript}\"\n");
    }
    println!("{} results in {:.1} ms ({} vectors)", outcome.hits.len(), outcome.elapsed_millis, outcome.vector_count);
    for (position, hit) in outcome.hits.iter().enumerate() {
        let lanes = [
            hit.sources.semantic.then_some("semantic"),
            hit.sources.vision.then_some("content"),
            hit.sources.keyword.then_some("keyword"),
            hit.sources.acoustic.then_some("acoustic"),
            hit.sources.filename.then_some("filename"),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join("+");
        println!("{:>2}. [{:.4}] ({}) {}", position + 1, hit.score, lanes, hit.path);
        if !hit.snippet.is_empty() {
            println!("     {}", hit.snippet.replace('\n', " "));
        }
    }
    engine.shutdown();
    Ok(())
}

fn run_ask(audio_path: PathBuf, limit: usize) -> anyhow::Result<()> {
    let audio_pcm = ipic_rag::extract::decode_speech_pcm(&audio_path)?;
    run_search("", limit, Some(audio_pcm))
}

fn run_scan(extra_root: Option<PathBuf>, watch: bool) -> anyhow::Result<()> {
    let mut config = Config::load_or_create()?;
    if let Some(root) = extra_root {
        config.roots.push(root.canonicalize()?);
    }
    let started = Instant::now();
    let engine = Engine::launch(config)?;
    stream_events_until_idle(&engine, started);
    if watch {
        println!("watching for changes — ctrl-c to exit");
        loop {
            while let Ok(event) = engine.events.try_recv() {
                print_event(&event);
            }
            std::thread::sleep(Duration::from_millis(250));
        }
    }
    engine.shutdown();
    Ok(())
}

/// Live one-line progress renderer (\r-updated) with per-kind ETA.
fn render_progress_bar(label: &str, completed: u64, total: u64, rate_per_second: f64, eta_seconds: Option<f64>) {
    let total = total.max(completed).max(1);
    let fraction = (completed as f64 / total as f64).clamp(0.0, 1.0);
    let bar_width = 28;
    let filled = (fraction * bar_width as f64).round() as usize;
    let eta = match (eta_seconds, rate_per_second > 0.01) {
        (Some(seconds), true) => {
            let seconds = seconds.max((total - completed) as f64 / rate_per_second);
            if seconds >= 60.0 {
                format!("{:.0}m", seconds / 60.0)
            } else {
                format!("{:.0}s", seconds)
            }
        }
        _ => "—".to_string(),
    };
    let bar = "█".repeat(filled) + &"░".repeat(bar_width - filled);
    print!(
        "\r  {label} |{bar}| {completed}/{total} ({:.0}%) {rate_per_second:.0}/s, eta {eta}  ",
        fraction * 100.0
    );
    use std::io::Write;
    std::io::stdout().flush().ok();
}

/// Streams engine events plus a live indexing progress bar until fully idle.
fn stream_events_until_idle(engine: &std::sync::Arc<Engine>, started: Instant) {
    let mut idle_ticks = 0;
    let mut last_render = std::time::Instant::now();
    while idle_ticks < 3 {
        while let Ok(event) = engine.events.try_recv() {
            print_event(&event);
        }
        let status = engine.status();
        if last_render.elapsed().as_secs_f64() >= 0.5 {
            last_render = std::time::Instant::now();
            let scanning = status.scanning;
            let completed = if scanning { status.scan_files_seen } else { status.done + status.failed };
            let rate = if scanning { status.scan_files_per_second } else { status.index_files_per_second };
            let eta = if scanning { status.scan_eta_seconds } else { status.index_eta_seconds };
            let label = if scanning && status.pending > 0 { "scanning + indexing" } else if scanning { "scanning" } else { "indexing" };
            // First-run scans have no known total: the bar grows with discovery.
            let total = if scanning { completed.max(1) } else { (status.total_files.max(0) as u64).max(completed) };
            render_progress_bar(label, completed, total, rate, eta);
        }
        if !status.scanning && status.pending == 0 {
            idle_ticks += 1;
        } else {
            idle_ticks = 0;
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    println!();
    println!("all indexing complete in {:.1}s", started.elapsed().as_secs_f64());
}

fn print_event(event: &EngineEvent) {
    match event {
        EngineEvent::ScanStarted { roots } => println!("scanning {} root(s)", roots.len()),
        EngineEvent::ScanFinished { files_seen, directories_seen, elapsed_seconds } => println!(
            "\rscan done: {files_seen} files, {directories_seen} directories in {elapsed_seconds:.1}s   "
        ),
        EngineEvent::EmbedderReady { model_id, neural } => {
            println!("embedder ready: {model_id}{}", if *neural { "" } else { " (fallback)" })
        }
        EngineEvent::VisionReady { model_id } => println!("vision ready: {model_id}"),
        EngineEvent::VisionUnavailable { reason } => println!("vision unavailable: {reason}"),
        EngineEvent::ModelDownload { model, downloaded_bytes, total_bytes, finished } => {
            if *finished {
                println!("model ready: {model}");
            } else if let Some(total) = total_bytes {
                println!("downloading {model}: {:.1} / {:.1} MB", downloaded_bytes / 1_000_000, total / 1_000_000);
            } else {
                println!("downloading {model}: {:.1} MB", downloaded_bytes / 1_000_000);
            }
        }
        EngineEvent::TranscriberReady { model } => println!("whisper ready: {model}"),
        EngineEvent::TranscriberFailed { reason } => println!("whisper unavailable: {reason}"),
        EngineEvent::Notice(message) => println!("note: {message}"),
        EngineEvent::ScanProgress { .. }
        | EngineEvent::IndexProgress { .. }
        | EngineEvent::CatalogChanged
        | EngineEvent::IndexingIdle => {}
    }
}


