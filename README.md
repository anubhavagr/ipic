# ipic

A lightweight, blazingly fast multimodal file manager with **fully on-device RAG search**.
Type or *speak* a natural-language query and ipic finds it across your text files, PDFs,
audio recordings and videos — no cloud, no API calls, no telemetry.

```
◆ ipic     Browse ‹ › ↑  documents › notes                Ask | ◧ | ⚙
┌────────────┬──────────────────────────────────────────┬─────────────┐
│ Locations  │  Filter by name…   Name ↑ Size Modified   │  preview    │
│ All files  │  📝 quarterly-roadmap.md   indexed        │  metadata   │
│ Filter     │  🎧 lecture.mp3            06:57 indexed   │  transcript │
│ Tree       │  🎬 quantum-talk.mp4       06:57 indexed   │  actions    │
├────────────┴──────────────────────────────────────────┴─────────────┤
│ ● ready   7 files   7 vectors   12× cores          search: 4 ms     │
└─────────────────────────────────────────────────────────────────────┘
```

## What it does

- **Standard file manager** — tree navigation, breadcrumbs, back/forward/up,
  metadata filtering (kind, name, size, recency), column sorting, virtualized
  table for huge directories, open/reveal/rename/trash.
- **Multimodal RAG search** — hybrid retrieval fusing three lanes:
  dense **semantic** vectors (bge-small-en-v1.5 via ONNX), **BM25 keyword**
  (SQLite FTS5), and **filename** match, merged with weighted reciprocal rank
  fusion. Typical latency: single-digit milliseconds.
- **Spoken queries** — record from the microphone; whisper (base.en, q5) transcribes
  locally and the transcript drives the same hybrid search.
- **Audio/video understanding** — speech in audio and video files is transcribed
  at index time (ffmpeg decode → whisper), so lectures, podcasts and talks become
  full-text + semantically searchable.
- **Everything else** (images, archives, binaries…) is catalogued with metadata,
  filterable and sortable — just not RAG-indexed.

## Design goals (and how they're met)

| Goal | Approach |
| --- | --- |
| Blazing fast | Parallel directory walker (all cores), SQLite WAL batched writes, mmap'd i8-quantized vector index scanned with rayon + integer SIMD-friendly dots |
| Minimum RAM | Vectors stored as 1-byte-per-component on disk (4× smaller than f32), scanned through the page cache — near-zero resident cost; ~60 MB model footprint total |
| On-device only | bge-small (~130 MB) + whisper base.en q5 (~57 MB), both cached under `~/.ipic/`; deterministic feature-hashing fallback keeps search alive offline |
| Modern CPUs | NEON/Accelerate BLAS for whisper, ONNX intra-op threading for embeddings, self-serving worker pool sized to core count |
| Swap-friendly UI | The GUI is a thin shell over `ipic-rag`'s `Engine` (channels + catalog readers) — replaceable without touching core logic |

## Architecture

```
crates/
├── ipic-core   parallel walker · SQLite catalog (files/dirs/FTS5 chunks) ·
│               change watcher · media probe · config
├── ipic-rag    embedders (neural + hashing) · mmap vector store (i8 quantized) ·
│               chunker · text/PDF/audio extraction · whisper transcriber ·
│               hybrid search (RRF) · Engine orchestration
├── ipic-cli    headless driver: scan · search · ask --audio · status · list · config
└── ipic-app    egui GUI: browse table, ask panel + mic, details preview, settings
```

**Indexing pipeline** (all background, resumable):
scan (N-core walk → batched upserts) → self-serving workers claim jobs
(text/PDF fast lane, audio/video gated by a whisper-state semaphore) →
chunk → batch-embed (ORT, multi-threaded) → FTS5 rows + quantized vectors → Done.
Crash-safe: interrupted work resets to Pending on next launch.

## Requirements

- **Rust** 1.94+ (`rustup`), and `cmake` (builds whisper.cpp automatically)
- **macOS** on Apple Silicon is the primary target (NEON/Accelerate, Metal optional);
  Linux works with the same code paths
- **ffmpeg** (optional) — broadens audio/video decode coverage; wav/mp3/flac/ogg/m4a
  decode fine without it via pure-Rust symphonia
- ~250 MB free disk for the two local models (downloaded once, then fully offline)

## Getting started

```bash
cargo build --release

# GUI
./target/release/ipic

# or headless
./target/release/ipic-cli scan                # index everything under roots
./target/release/ipic-cli search "radiation shielding on mars"
./target/release/ipic-cli ask --audio query.wav
./target/release/ipic-cli status
./target/release/ipic-cli list --kind video
./target/release/ipic-cli config --set-roots ~/Downloads ~/Movies
```

First run downloads the two local models (~190 MB total) into `~/.ipic/models/`;
everything after that is 100% offline. Default scope is `~/Downloads` — add or
change roots any time via settings or `config --set-roots`.

Configuration lives at `~/.ipic/config.toml` (roots, skip-list, whisper model,
worker counts, GPU toggle). Data: `~/.ipic/catalog.db` (SQLite/FTS5) and
`~/.ipic/vectors.bin` (quantized vector index).

## Notes & trade-offs

- Audio/video search covers **speech**. Non-speech audio (music, ambience) is
  metadata-only; a CLAP-style audio encoder is the natural extension.
- GPU (Metal) whisper is available (`whisper_use_gpu = true`): very fast, but
  transcriptions serialize on a single state — the default CPU path
  (Accelerate BLAS, ~35× realtime) parallelizes across workers.
- `ffmpeg` (optional) broadens decode coverage for exotic containers;
  pure-Rust symphonia handles wav/mp3/flac/ogg/m4a without it.

## Tests

```bash
cargo test --workspace   # unit tests + full scan→catalog→embed→search integration test
```
