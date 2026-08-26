//! Content extraction: text files, PDFs, and speech decoding to 16 kHz mono f32 PCM
//! (ffmpeg pipe when available — covers every container — else pure-Rust symphonia).

use anyhow::{anyhow, Context, Result};
use ipic_core::FileKind;
use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};

const CHUNK_WORDS: usize = 320; // ≈ 512 tokens for typical English
const OVERLAP_WORDS: usize = 24;
/// Hard character ceiling per chunk: tokenizers process the full string even
/// past the model's truncation point, and batch padding wastes compute on
/// length spread.
const CHUNK_CHAR_LIMIT: usize = 800;
/// Minified code emits megabyte "words"; split them before tokenization.
const WORD_CHAR_LIMIT: usize = 64;
/// Head chunks carry a document's gist; deep tails of huge files (bundles,
/// lockfiles, datasets) are not worth their embedding cost.
const MAX_CHUNKS_PER_FILE: usize = 8;

/// Reads text content for Text/Pdf files; returns None for empty/unreadable input.
/// pdf-extract panics on malformed PDFs, so its call is panic-guarded.
pub fn extract_text(path: &Path, kind: FileKind, max_bytes: usize) -> Result<Option<String>> {
    let text = match kind {
        FileKind::Text => {
            let mut buffer = vec![0u8; max_bytes + 1];
            let read = std::fs::File::open(path)?.read(&mut buffer)?;
            buffer.truncate(read);
            match String::from_utf8(buffer) {
                Ok(text) => text,
                Err(_) => return Ok(None), // binary masquerading as text
            }
        }
        FileKind::Pdf => {
            let path_display = path.display().to_string();
            match std::panic::catch_unwind(move || pdf_extract::extract_text(&path_display)) {
                Ok(Ok(text)) => text,
                Ok(Err(error)) => return Err(anyhow!("pdf extraction failed: {error}")),
                Err(_) => return Err(anyhow!("pdf extraction panicked (malformed pdf)")),
            }
        }
        FileKind::Image => {
            // No vision model on-device (yet): images are searchable through
            // their filename and folder context, e.g. "vacation 2024 beach sunset".
            let segments: Vec<String> = path
                .components()
                .filter_map(|component| {
                    component.as_os_str().to_str().map(|text| text.to_string())
                })
                .flat_map(|text| {
                    text.split(|character: char| !character.is_alphanumeric())
                        .filter(|word| word.len() > 2 && !word.eq_ignore_ascii_case("jpg")
                            && !word.eq_ignore_ascii_case("jpeg") && !word.eq_ignore_ascii_case("png")
                            && !word.eq_ignore_ascii_case("heic") && !word.eq_ignore_ascii_case("webp")
                            && !word.eq_ignore_ascii_case("tiff") && !word.eq_ignore_ascii_case("gif")
                            && !word.eq_ignore_ascii_case("avif"))
                        .map(|word| word.to_lowercase())
                        .collect::<Vec<_>>()
                })
                .collect();
            let context = segments.join(" ");
            return Ok((!context.is_empty()).then_some(context));
        }
        _ => return Ok(None),
    };
    let trimmed = text.trim();
    Ok((!trimmed.is_empty()).then(|| trimmed.to_string()))
}

/// Word-bounded chunks with a small trailing overlap so boundary context stays
/// searchable. Both word and character budgets bound tokenizer input.
pub fn chunk_text(text: &str) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().flat_map(split_long_word).collect();
    let mut chunks: Vec<String> = Vec::new();
    let mut start = 0;
    while start < words.len() {
        let mut end = start;
        let mut characters = 0;
        while end < words.len() && end - start < CHUNK_WORDS && characters < CHUNK_CHAR_LIMIT {
            characters += words[end].len() + 1;
            end += 1;
        }
        chunks.push(words[start..end].join(" "));
        if end == words.len() || chunks.len() >= MAX_CHUNKS_PER_FILE {
            break;
        }
        // Overlap capped below the chunk length: start always advances.
        start = end - OVERLAP_WORDS.min(end - start - 1);
    }
    chunks.retain(|chunk| chunk.split_whitespace().count() > 3);
    chunks
}

/// Char-safe splitter for pathological single tokens.
fn split_long_word(word: &str) -> Vec<&str> {
    if word.chars().count() <= WORD_CHAR_LIMIT {
        return vec![word];
    }
    let mut pieces = Vec::new();
    let mut collected = 0;
    let mut piece_start = 0;
    for (offset, _) in word.char_indices() {
        if collected >= WORD_CHAR_LIMIT {
            pieces.push(&word[piece_start..offset]);
            piece_start = offset;
            collected = 0;
        }
        collected += 1;
    }
    pieces.push(&word[piece_start..]);
    pieces
}

pub const SAMPLE_RATE: usize = 16_000;

/// Extensions the `image` crate decodes directly; everything else (HEIC,
/// HEIF, AVIF) rides the ffmpeg pipe when present.
const NATIVE_IMAGE_EXTENSIONS: &[&str] =
    &["jpg", "jpeg", "png", "webp", "gif", "bmp", "tiff", "tif", "ico"];

/// Decodes an image and downsamples to CLIP input resolution, bounding decode
/// memory. Returns Ok(None) for formats with no pixel content (svg); errors
/// mean the file is corrupt or oversized.
pub fn decode_image_thumbnail(path: &Path, max_bytes: usize) -> Result<Option<image::DynamicImage>> {
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| extension.to_ascii_lowercase())
        .unwrap_or_default();
    if extension == "svg" {
        return Ok(None); // no rasterizer; filename-context still applies
    }
    let metadata = std::fs::metadata(path)?;
    if metadata.len() as usize > max_bytes {
        return Err(anyhow!("image exceeds the decode size limit"));
    }
    let decoded = if NATIVE_IMAGE_EXTENSIONS.contains(&extension.as_str()) {
        image::ImageReader::open(path)
            .map_err(|error| anyhow!("image open failed: {error}"))?
            .decode()
            .map_err(|error| anyhow!("image decode failed: {error}"))?
    } else {
        decode_image_with_ffmpeg(path)?
    };
    Ok(Some(decoded.thumbnail(224, 224)))
}

fn decode_image_with_ffmpeg(path: &Path) -> Result<image::DynamicImage> {
    let output = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(["-vf", "scale=224:224:force_original_aspect_ratio=decrease", "-f", "image2", "-vcodec", "png", "-"])
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .map_err(|error| anyhow!("ffmpeg spawn failed: {error}"))?;
    if !output.status.success() || output.stdout.is_empty() {
        return Err(anyhow!("ffmpeg could not decode {}", path.display()));
    }
    image::load_from_memory(&output.stdout).map_err(|error| anyhow!("ffmpeg png decode failed: {error}"))
}

/// True when ffmpeg is usable on this machine (broadest container coverage).
/// Probing spawns a process — answered once per process lifetime.
pub fn ffmpeg_available() -> bool {
    static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        std::process::Command::new("ffprobe")
            .arg("-version")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    })
}

/// Decodes any audio/video file to mono 16 kHz f32 PCM for whisper.
pub fn decode_speech_pcm(path: &Path) -> Result<Vec<f32>> {
    if ffmpeg_available() {
        return decode_with_ffmpeg(path);
    }
    decode_with_symphonia(path).with_context(|| format!("decoding {} without ffmpeg", path.display()))
}

fn decode_with_ffmpeg(path: &Path) -> Result<Vec<f32>> {
    let output = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(["-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"])
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .map_err(|error| anyhow!("ffmpeg spawn failed: {error}"))?;
    if !output.status.success() && output.stdout.is_empty() {
        return Err(anyhow!("ffmpeg could not decode {}", path.display()));
    }
    Ok(bytes_to_f32(output.stdout))
}

fn bytes_to_f32(mut bytes: Vec<u8>) -> Vec<f32> {
    bytes.shrink_to_fit();
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn decode_with_symphonia(path: &Path) -> Result<Vec<f32>> {
    use symphonia::core::codecs::audio::AudioDecoderOptions;
    use symphonia::core::errors::Error;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::formats::TrackType;
    use symphonia::core::formats::probe::Hint;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;

    let source = MediaSourceStream::new(Box::new(std::fs::File::open(path)?), Default::default());
    let mut hint = Hint::new();
    if let Some(extension) = path.extension().and_then(|extension| extension.to_str()) {
        hint.with_extension(extension);
    }
    let mut format = symphonia::default::get_probe()
        .probe(&hint, source, FormatOptions::default(), MetadataOptions::default())?;
    let Some(track) = format.default_track(TrackType::Audio) else {
        return Err(anyhow!("no audio track"));
    };
    let audio_parameters = track
        .codec_params
        .as_ref()
        .and_then(|parameters| parameters.audio())
        .ok_or_else(|| anyhow!("missing audio parameters"))?;
    let track_id = track.id;
    let source_rate = audio_parameters.sample_rate.unwrap_or(SAMPLE_RATE as u32) as usize;
    let channels = audio_parameters.channels.as_ref().map(|channels| channels.count()).unwrap_or(1).max(1);
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(audio_parameters, &AudioDecoderOptions::default())?;
    let mut interleaved: Vec<f32> = Vec::new();
    let mut packet_samples: Vec<f32> = Vec::new();
    while let Some(packet) = format.next_packet()? {
        if packet.track_id != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(audio_buffer) => {
                packet_samples.resize(audio_buffer.samples_interleaved(), 0.0);
                audio_buffer.copy_to_slice_interleaved(&mut packet_samples);
                // Mono downmix: average channels per frame.
                interleaved.extend(
                    packet_samples
                        .chunks(channels)
                        .map(|frame| frame.iter().sum::<f32>() / channels as f32),
                );
            }
            Err(Error::DecodeError(_)) => continue,
            Err(error) => return Err(error.into()),
        }
    }
    Ok(resample_linear(interleaved, source_rate, SAMPLE_RATE))
}

/// Linear resampling (quality is ample for ASR; cheapest CPU path).
pub fn resample_linear(samples: Vec<f32>, source_rate: usize, target_rate: usize) -> Vec<f32> {
    if source_rate == target_rate || samples.is_empty() {
        return samples;
    }
    let output_len = (samples.len() as f64 * target_rate as f64 / source_rate as f64) as usize;
    let step = (samples.len() - 1) as f32 / output_len.max(1) as f32;
    (0..output_len)
        .map(|index| {
            let position = index as f32 * step;
            let left = position.floor() as usize;
            let right = (left + 1).min(samples.len() - 1);
            let fraction = position - left as f32;
            samples[left] * (1.0 - fraction) + samples[right] * fraction
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_are_bounded_and_overlapping() {
        let text = (0..2000).map(|i| format!("word{i}{} ", if i % 50 == 49 { "." } else { "" })).collect::<String>();
        let chunks = chunk_text(&text);
        assert!(chunks.len() >= 3);
        for chunk in &chunks {
            assert!(chunk.split_whitespace().count() <= CHUNK_WORDS + 8);
        }
    }

    #[test]
    fn tiny_text_is_single_chunk() {
        assert_eq!(chunk_text("hello world this is a test"), vec!["hello world this is a test"]);
        assert!(chunk_text("").is_empty());
    }

    #[test]
    fn pathological_words_are_split_and_bounded() {
        let minified = "a".repeat(100_000);
        let chunks = chunk_text(&minified);
        assert!(!chunks.is_empty(), "minified text still yields chunks");
        for chunk in &chunks {
            assert!(chunk.len() <= CHUNK_CHAR_LIMIT + WORD_CHAR_LIMIT, "chunk {} chars", chunk.len());
        }
        // A long run of normal words respects the character ceiling.
        let normal = "word ".repeat(10_000);
        for chunk in chunk_text(&normal) {
            assert!(chunk.len() <= CHUNK_CHAR_LIMIT + 8);
        }
    }

    #[test]
    fn resampler_changes_length_proportionally() {
        let samples = vec![0.5f32; 4800];
        let resampled = resample_linear(samples, 48_000, 16_000);
        assert!((resampled.len() as i64 - 1600).abs() < 3);
    }
}
