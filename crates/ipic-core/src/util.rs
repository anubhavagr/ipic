//! Small display helpers: human sizes, durations, local timestamps.

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

pub fn format_size(bytes: i64) -> String {
    const UNITS: [&str; 6] = ["B", "KB", "MB", "GB", "TB", "PB"];
    if bytes < 1024 {
        return format!("{} B", bytes);
    }
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    format!("{:.1} {}", value, UNITS[unit])
}

pub fn format_duration(seconds: f64) -> String {
    let total = seconds.round() as i64;
    let (hours, minutes, secs) = (total / 3600, (total % 3600) / 60, total % 60);
    if hours > 0 {
        format!("{}:{:02}:{:02}", hours, minutes, secs)
    } else {
        format!("{:02}:{:02}", minutes, secs)
    }
}

/// Local-time "YYYY-MM-DD HH:MM" via libc localtime_r (no heavyweight chrono dependency).
pub fn format_local_timestamp(unix_secs: i64) -> String {
    unsafe {
        let time = unix_secs as libc::time_t;
        let mut tm: libc::tm = std::mem::zeroed();
        if libc::localtime_r(&time, &mut tm).is_null() {
            return unix_secs.to_string();
        }
        format!(
            "{:04}-{:02}-{:02} {:02}:{:02}",
            tm.tm_year as i64 + 1900,
            tm.tm_mon + 1,
            tm.tm_mday,
            tm.tm_hour,
            tm.tm_min
        )
    }
}

pub fn unix_now() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

/// ~/.ipic scratch path for tests: unique per test binary, cleaned by caller.
pub fn test_data_dir(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("ipic-test-{}-{}", tag, std::process::id()))
}
