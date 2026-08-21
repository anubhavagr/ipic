//! Media metadata probing via ffprobe (present on most dev machines; absence degrades gracefully).

use serde_json::Value;
use std::path::Path;
use std::process::Command;

#[derive(Debug, Clone, Copy, Default)]
pub struct MediaProbe {
    pub duration_secs: Option<f64>,
    pub width: Option<i64>,
    pub height: Option<i64>,
}

pub fn probe_media(path: &Path) -> MediaProbe {
    let output = Command::new("ffprobe")
        .args(["-v", "quiet", "-print_format", "json", "-show_format", "-show_streams"])
        .arg(path)
        .output();
    let Ok(output) = output else { return MediaProbe::default() };
    if !output.status.success() {
        return MediaProbe::default();
    }
    let Ok(json) = serde_json::from_slice::<Value>(&output.stdout) else {
        return MediaProbe::default();
    };
    let duration = json
        .pointer("/format/duration")
        .and_then(Value::as_str)
        .and_then(|d| d.parse::<f64>().ok());
    let video = json
        .pointer("/streams")
        .and_then(Value::as_array)
        .and_then(|streams| {
            streams.iter().find(|stream| stream.get("codec_type").and_then(Value::as_str) == Some("video"))
        });
    MediaProbe {
        duration_secs: duration,
        width: video.and_then(|v| v.get("width")).and_then(Value::as_i64),
        height: video.and_then(|v| v.get("height")).and_then(Value::as_i64),
    }
}

/// True when ffmpeg/ffprobe are usable on this machine (used to gate video transcription).
pub fn ffmpeg_available() -> bool {
    Command::new("ffprobe").arg("-version").output().map(|o| o.status.success()).unwrap_or(false)
}
