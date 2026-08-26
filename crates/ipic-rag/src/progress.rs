//! Indexing ETA: per-kind completion rates (an hour of audio is not a JPEG)
//! smoothed with an exponential moving average; estimates appear only once
//! there is real data to extrapolate from.

use ipic_core::FileKind;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Weight of the newest window in the smoothed rate.
const RATE_SMOOTHING: f64 = 0.3;
/// Minimum observation time before an estimate is honest enough to show.
const WARMUP: Duration = Duration::from_secs(5);

#[derive(Default)]
pub struct IndexingEstimator {
    rates: HashMap<FileKind, f64>,
    last_done: HashMap<FileKind, i64>,
    last_tick: Option<Instant>,
    first_backlog: Option<Instant>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IndexingEstimate {
    pub files_per_second: f64,
    pub eta_seconds: f64,
}

impl IndexingEstimator {
    pub fn new() -> Self {
        Self { rates: HashMap::new(), last_done: HashMap::new(), last_tick: None, first_backlog: None }
    }

    /// Feeds per-kind counters (pending incl. busy, done, failed); returns an
    /// estimate once rates have warmed up and work remains.
    pub fn tick(
        &mut self,
        counters: &[(FileKind, i64, i64, i64)],
        now: Instant,
    ) -> Option<IndexingEstimate> {
        let window = match self.last_tick.replace(now) {
            Some(previous) => now.duration_since(previous).as_secs_f64(),
            None => return self.refresh(counters, now),
        };
        if window <= 0.0 {
            return None;
        }
        for &(kind, _pending, done, _failed) in counters {
            let delta = (done - self.last_done.get(&kind).copied().unwrap_or(0)).max(0) as f64;
            let observed = delta / window;
            let smoothed = match self.rates.get(&kind) {
                Some(previous) => previous * (1.0 - RATE_SMOOTHING) + observed * RATE_SMOOTHING,
                None if observed > 0.0 => observed,
                None => 0.0,
            };
            if smoothed > 0.0 {
                self.rates.insert(kind, smoothed);
            }
            self.last_done.insert(kind, done);
        }
        self.refresh(counters, now)
    }

    fn refresh(&mut self, counters: &[(FileKind, i64, i64, i64)], now: Instant) -> Option<IndexingEstimate> {
        let remaining: i64 = counters.iter().map(|&(_, pending, _, _)| pending).sum();
        if remaining == 0 {
            self.first_backlog = None;
            return None;
        }
        if self.first_backlog.is_none() {
            self.first_backlog = Some(now);
        }
        if now.duration_since(self.first_backlog?) < WARMUP {
            return None;
        }
        let mut seconds_remaining = 0.0;
        for &(kind, pending, _, _) in counters {
            if pending == 0 {
                continue;
            }
            // A kind with backlog but no observed rate yet makes the total
            // unknowable — show nothing rather than a fabricated number.
            let rate = self.rates.get(&kind).copied()?;
            seconds_remaining += pending as f64 / rate.max(1e-6);
        }
        let throughput: f64 = self.rates.values().sum();
        Some(IndexingEstimate { files_per_second: throughput, eta_seconds: seconds_remaining })
    }

    /// Latest estimate from observed rates without consuming a window.
    pub fn latest(&mut self, counters: &[(FileKind, i64, i64, i64)]) -> Option<IndexingEstimate> {
        self.refresh(counters, Instant::now())
    }
}

/// Scan ETA from discovery throughput against a known-catalog hint.
pub fn scan_eta(files_seen: u64, hint_total: u64, files_per_second: f64) -> Option<f64> {
    if hint_total == 0 || files_per_second < 0.01 || files_seen == 0 {
        return None;
    }
    // Discovery may exceed the stale hint (new files): clamp, never negative.
    Some(hint_total.saturating_sub(files_seen) as f64 / files_per_second)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counters(pending: i64, done: i64) -> Vec<(FileKind, i64, i64, i64)> {
        vec![(FileKind::Text, pending, done, 0), (FileKind::Audio, 0, 0, 0)]
    }

    #[test]
    fn estimate_appears_after_warmup_and_tracks_rate() {
        let start = Instant::now();
        let mut estimator = IndexingEstimator::new();
        // t=0: backlog starts, no estimate during warmup.
        let t = |secs: f64| start + Duration::from_secs_f64(secs);
        assert!(estimator.tick(&counters(100, 0), t(0.0)).is_none());
        assert!(estimator.tick(&counters(100, 50), t(4.5)).is_none()); // still warming up
        let estimate = estimator.tick(&counters(50, 100), t(10.0)).expect("estimate after warmup");
        // Two 50-file/5s windows hold the rate at 10/s → 50 left ≈ 5s.
        assert!((estimate.eta_seconds - 5.0).abs() < 1.0, "eta was {}", estimate.eta_seconds);
        assert!((estimate.files_per_second - 10.0).abs() < 1.0);
    }

    #[test]
    fn no_estimate_without_backlog_or_without_rate() {
        let start = Instant::now();
        let mut estimator = IndexingEstimator::new();
        assert!(estimator.tick(&counters(0, 10), start).is_none());
        // Backlog in a kind that has never completed anything: unknowable.
        let audio_only = vec![(FileKind::Audio, 40, 0, 0)];
        let much_later = start + Duration::from_secs(60);
        assert!(estimator.tick(&audio_only, much_later).is_none());
    }

    #[test]
    fn scan_eta_clamps_at_discovered_surplus() {
        assert_eq!(scan_eta(120, 100, 10.0), Some(0.0));
        assert_eq!(scan_eta(0, 100, 10.0), None);
        assert!((scan_eta(50, 100, 10.0).unwrap() - 5.0).abs() < 1e-9);
    }
}
