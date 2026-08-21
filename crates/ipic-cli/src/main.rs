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
    Config { #[arg(long)] set_roots: Option<Vec<PathBuf>> },
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
    let catalog = ipic_core::catalog::Catalog::open(&ipic_core::data_dir().join("catalog.db"))?;
    let connection = catalog.reader()?;
    let filter = ipic_core::FileFilter { kinds: kind.kinds(), ..Default::default() };
    let directory_row = match &directory {
        Some(directory) => catalog.dir_by_path(&connection, &directory.canonicalize()?.to_string_lossy())?,
        None => None,
    };
    let entries = catalog.children(&connection, directory_row.map(|row| row.id), &filter, SortKey::Name, true, limit)?;
    let directory_display = directory
        .as_ref()
        .map(|directory| directory.canonicalize().unwrap_or_else(|_| directory.clone()))
        .unwrap_or_else(|| PathBuf::from("(entire catalog)"));
    for file_row in entries {
        println!(
            "{:<6} {:>10}  {} / {}",
            file_row.kind.label(),
            ipic_core::util::format_size(file_row.size),
            directory_display.display(),
            file_row.name
        );
    }
    Ok(())
}

fn run_status() -> anyhow::Result<()> {
    let engine = Engine::launch(Config::load_or_create()?)?;
    // Give the boot events a moment to land, then report.
    std::thread::sleep(Duration::from_millis(600));
    drain_events(&engine, false);
    let status = engine.status();
    println!("files indexed      : {}", status.total_files);
    println!("rag pending        : {}", status.pending);
    println!("rag completed      : {}", status.done);
    println!("rag failed         : {}", status.failed);
    println!("vectors stored     : {}", status.vector_count);
    println!("embedder           : {}{}", status.embedder_model, if status.neural_embedder { " (neural)" } else { " (lexical fallback)" });
    println!("whisper            : {}{}", status.whisper_model, if status.transcriber_ready { " (ready)" } else { " (loading)" });
    println!("cpu cores used     : {}", status.core_count);
    engine.shutdown();
    Ok(())
}

fn run_search(query: &str, limit: usize, spoken_pcm: Option<Vec<f32>>) -> anyhow::Result<()> {
    let engine = Engine::launch(Config::load_or_create()?)?;
    // Let resume/scan settle briefly so fresh installs return something.
    std::thread::sleep(Duration::from_millis(300));
    drain_events(&engine, false);
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
            hit.sources.keyword.then_some("keyword"),
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

/// Live one-line progress renderer (\r-updated).
fn render_progress_bar(label: &str, completed: u64, total: u64, rate_per_second: f64) {
    let total = total.max(completed).max(1);
    let fraction = (completed as f64 / total as f64).clamp(0.0, 1.0);
    let bar_width = 28;
    let filled = (fraction * bar_width as f64).round() as usize;
    let eta = if rate_per_second > 0.01 {
        let seconds = (total - completed) as f64 / rate_per_second;
        if seconds >= 60.0 {
            format!("{:.0}m", seconds / 60.0)
        } else {
            format!("{:.0}s", seconds)
        }
    } else {
        "—".to_string()
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
    let mut completed_snapshot = 0u64;
    while idle_ticks < 3 {
        while let Ok(event) = engine.events.try_recv() {
            print_event(&event);
        }
        if let Ok((pending, busy, done, failed)) = engine.catalog.rag_counters() {
            let total_files = engine.catalog.total_files().unwrap_or(0) as u64;
            let completed = (done + failed) as u64;
            if last_render.elapsed().as_secs_f64() >= 0.5 {
                let elapsed = last_render.elapsed().as_secs_f64();
                let rate = completed.saturating_sub(completed_snapshot) as f64 / elapsed.max(0.001);
                completed_snapshot = completed;
                last_render = std::time::Instant::now();
                let label = if engine.is_scanning() { "scanning + indexing" } else { "indexing" };
                render_progress_bar(label, completed, total_files.max(completed), rate);
            }
            if !engine.is_scanning() && pending + busy == 0 {
                idle_ticks += 1;
            } else {
                idle_ticks = 0;
            }
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

/// Prints engine events; in verbose mode streams everything until idle.
fn drain_events(engine: &std::sync::Arc<Engine>, verbose: bool) {
    while let Ok(event) = engine.events.try_recv() {
        match event {
            EngineEvent::ScanStarted { roots } => {
                if verbose {
                    println!("scanning {} root(s)", roots.len());
                }
            }
            EngineEvent::ScanProgress { files_seen, directories_seen, elapsed_seconds } => {
                if verbose && files_seen % 100_000 == 0 {
                    println!("  scanned {files_seen} files in {directories_seen} dirs ({elapsed_seconds:.1}s)");
                }
            }
            EngineEvent::ScanFinished { files_seen, directories_seen, elapsed_seconds } => {
                println!("scan: {files_seen} files, {directories_seen} directories in {elapsed_seconds:.1}s");
            }
            EngineEvent::EmbedderReady { model_id, neural } => {
                println!("embedder ready: {model_id}{}", if neural { "" } else { " (fallback)" });
            }
            EngineEvent::ModelDownload { model, downloaded_bytes, total_bytes, finished } => {
                if finished {
                    println!("model ready: {model}");
                } else if let Some(total) = total_bytes {
                    println!("  downloading {model}: {:.1} / {:.1} MB", downloaded_bytes as f64 / 1e6, total as f64 / 1e6);
                } else {
                    println!("  downloading {model}: {:.1} MB", downloaded_bytes as f64 / 1e6);
                }
            }
            EngineEvent::TranscriberReady { model } => println!("whisper ready: {model}"),
            EngineEvent::TranscriberFailed { reason } => println!("whisper unavailable: {reason}"),
            EngineEvent::IndexProgress { pending, done, failed, .. } => {
                if verbose && pending % 500 == 0 && pending > 0 {
                    println!("  indexing… {pending} pending, {done} done, {failed} failed");
                }
            }
            EngineEvent::IndexingIdle => {
                if verbose {
                    println!("indexing idle");
                }
            }
            EngineEvent::Notice(message) => println!("note: {message}"),
            EngineEvent::CatalogChanged => {}
        }
    }
}
