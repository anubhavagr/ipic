//! Acoustic fingerprints: 128-dim spectral-statistics vectors computed from
//! decoded PCM, enabling audio-to-audio similarity (find the same or similar
//! recording regardless of transcript). Pure Rust, no model download.

use rustfft::{FftPlanner, num_complex::Complex};

pub const FINGERPRINT_DIM: usize = 128;
const FRAME_SIZE: usize = 1024;
const HOP_SIZE: usize = 512;
const BAND_COUNT: usize = FINGERPRINT_DIM / 2; // mean + stddev per band
const LOWEST_HZ: f32 = 80.0;
const HIGHEST_HZ: f32 = 7600.0;

/// Returns an L2-normalized fingerprint, or an empty vector when the clip is
/// too short to characterize (callers skip storing those).
pub fn fingerprint_from_pcm(samples: &[f32], sample_rate: u32) -> Vec<f32> {
    if samples.len() < FRAME_SIZE * 6 || sample_rate == 0 {
        return Vec::new();
    }
    // Log-spaced band edges across the spectrum.
    let bins = FRAME_SIZE / 2;
    let bin_hz = sample_rate as f32 / FRAME_SIZE as f32;
    let low_bin = (LOWEST_HZ / bin_hz).ceil().max(1.0) as usize;
    let high_bin = (HIGHEST_HZ / bin_hz).floor().min(bins as f32 - 1.0) as usize;
    if high_bin <= low_bin {
        return Vec::new();
    }
    let mut edges = Vec::with_capacity(BAND_COUNT + 1);
    let ratio = (high_bin as f32 / low_bin as f32).powf(1.0 / BAND_COUNT as f32);
    let mut edge = low_bin as f32;
    for _ in 0..=BAND_COUNT {
        edges.push(edge as usize);
        edge *= ratio;
    }

    let window: Vec<f32> = (0..FRAME_SIZE)
        .map(|index| {
            (std::f32::consts::PI * index as f32 / (FRAME_SIZE - 1) as f32).sin().powi(2)
        })
        .collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(FRAME_SIZE);

    // Per band: mean log-energy plus spectral flux (frame-to-frame log-energy
    // change). Flux separates moving spectra (sweeps, speech, music) from
    // static ones far better than energy statistics alone.
    let mut band_sums = vec![0.0f64; BAND_COUNT];
    let mut flux_squared_sums = vec![0.0f64; BAND_COUNT];
    let mut previous_frame_bands: Option<Vec<f64>> = None;
    let mut frame_count = 0u64;
    let mut buffer: Vec<Complex<f32>> = vec![Complex::default(); FRAME_SIZE];

    for start in (0..samples.len().saturating_sub(FRAME_SIZE)).step_by(HOP_SIZE) {
        for (index, slot) in buffer.iter_mut().enumerate() {
            slot.re = samples[start + index] * window[index];
            slot.im = 0.0;
        }
        fft.process(&mut buffer);
        let mut frame_bands = vec![0.0f64; BAND_COUNT];
        for band in 0..BAND_COUNT {
            let mut energy = 0.0f64;
            for sample in buffer[edges[band]..edges[band + 1].max(edges[band] + 1)].iter() {
                energy += sample.norm_sqr() as f64;
            }
            let bins_in_band = (edges[band + 1] - edges[band]).max(1);
            // Log compression keeps loudness differences from dominating.
            frame_bands[band] = (energy / bins_in_band as f64 + 1e-10).ln();
        }
        for band in 0..BAND_COUNT {
            band_sums[band] += frame_bands[band];
            if let Some(previous) = &previous_frame_bands {
                let flux = frame_bands[band] - previous[band];
                flux_squared_sums[band] += flux * flux;
            }
        }
        previous_frame_bands = Some(frame_bands);
        frame_count += 1;
    }
    if frame_count < 2 {
        return Vec::new();
    }

    // Build the two statistic halves and unit-normalize each separately, so
    // spectral shape and temporal dynamics contribute equally to similarity.
    let frames = frame_count as f64;
    let mut means = Vec::with_capacity(BAND_COUNT);
    let mut fluxes = Vec::with_capacity(BAND_COUNT);
    for band in 0..BAND_COUNT {
        means.push((band_sums[band] / frames) as f32);
        fluxes.push((flux_squared_sums[band] / (frames - 1.0)).sqrt() as f32);
    }
    let normalize = |half: &mut Vec<f32>| {
        let norm = half.iter().map(|value| value * value).sum::<f32>().sqrt();
        if norm < f32::EPSILON {
            false
        } else {
            half.iter_mut().for_each(|value| *value /= norm);
            true
        }
    };
    if !normalize(&mut means) || !normalize(&mut fluxes) {
        return Vec::new();
    }
    means.extend(fluxes);
    means
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic white noise (seeded xorshift).
    fn seeded_noise(seed: u32, seconds: f32) -> Vec<f32> {
        let mut state = seed;
        (0..(16_000.0 * seconds) as usize)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// Heavily low-passed noise: spectrally distinct from broadband noise.
    fn muffled_noise(seed: u32, seconds: f32) -> Vec<f32> {
        let source = seeded_noise(seed, seconds);
        source
            .windows(129)
            .map(|window| window.iter().sum::<f32>() / window.len() as f32)
            .collect()
    }

    #[test]
    fn identical_audio_self_matches_and_different_audio_separates() {
        let broadband = seeded_noise(0x9E37_79B9, 3.0);
        let muffled = muffled_noise(0x9E37_79B9, 3.0);
        let first_fingerprint = fingerprint_from_pcm(&broadband, 16_000);
        let second_fingerprint = fingerprint_from_pcm(&muffled, 16_000);
        assert_eq!(first_fingerprint.len(), FINGERPRINT_DIM);
        let similarity = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        let self_similarity = similarity(&first_fingerprint, &fingerprint_from_pcm(&broadband, 16_000));
        let cross_similarity = similarity(&first_fingerprint, &second_fingerprint);
        // Similarity spans two unit-normalized halves: self-match ≈ 2.
        assert!(self_similarity > 1.9, "self-match should be ~2, got {self_similarity}");
        assert!(
            cross_similarity < 0.8 * self_similarity,
            "different audio must score well below self-match: {cross_similarity} vs {self_similarity}"
        );
    }

    #[test]
    fn short_clips_return_empty() {
        assert!(fingerprint_from_pcm(&[0.0; 100], 16_000).is_empty());
    }
}
