// Copyright (C) 2026 The orangu community
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Whisper's input: thirty seconds of 16 kHz sound as a log-mel
//! spectrogram, `[n_mels, 3000]`.
//!
//! A 400-sample periodic Hann window every 160 samples, centred (the
//! signal reflected 200 samples at each end), the last frame dropped; the
//! power of the 201 bins through a Slaney-normalised mel filterbank over
//! 0–8 kHz; `log10`, floored 8 below the peak, then `(x + 4) / 4`. The
//! transform is a plain DFT against a precomputed table: 400 points is not
//! a power of two, and 3000 frames of it is cheap.

use rayon::prelude::*;

pub const N_FFT: usize = 400;
pub const HOP: usize = 160;
pub const CHUNK_SAMPLES: usize = 30 * super::wav::SAMPLE_RATE as usize;
pub const FRAMES: usize = CHUNK_SAMPLES / HOP;
const BINS: usize = N_FFT / 2 + 1;

fn hz_to_mel(f: f64) -> f64 {
    let (f_sp, min_log_hz) = (200.0 / 3.0, 1000.0);
    let step = (6.4f64).ln() / 27.0;
    if f < min_log_hz {
        f / f_sp
    } else {
        min_log_hz / f_sp + (f / min_log_hz).ln() / step
    }
}

fn mel_to_hz(m: f64) -> f64 {
    let (f_sp, min_log_hz) = (200.0 / 3.0, 1000.0);
    let min_log_mel = min_log_hz / f_sp;
    let step = (6.4f64).ln() / 27.0;
    if m < min_log_mel {
        m * f_sp
    } else {
        min_log_hz * ((m - min_log_mel) * step).exp()
    }
}

/// `[n_mels, BINS]` triangular filters, Slaney scale and area norm.
pub fn filterbank(n_mels: usize) -> Vec<f32> {
    let sr = super::wav::SAMPLE_RATE as f64;
    let fft_hz: Vec<f64> = (0..BINS).map(|i| i as f64 * sr / N_FFT as f64).collect();
    let (lo, hi) = (hz_to_mel(0.0), hz_to_mel(sr / 2.0));
    let pts: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (n_mels + 1) as f64))
        .collect();
    let mut out = vec![0.0f32; n_mels * BINS];
    for m in 0..n_mels {
        let (l, c, r) = (pts[m], pts[m + 1], pts[m + 2]);
        let norm = 2.0 / (r - l);
        for (b, &f) in fft_hz.iter().enumerate() {
            let w = ((f - l) / (c - l)).min((r - f) / (r - c)).max(0.0);
            out[m * BINS + b] = (w * norm) as f32;
        }
    }
    out
}

/// Log-mel spectrogram of up to thirty seconds of `audio`, zero-padded,
/// as `[n_mels, FRAMES]`.
pub fn log_mel(audio: &[f32], filters: &[f32], n_mels: usize) -> Vec<f32> {
    let mut x = vec![0.0f32; CHUNK_SAMPLES];
    let n = audio.len().min(CHUNK_SAMPLES);
    x[..n].copy_from_slice(&audio[..n]);
    let pad = N_FFT / 2;
    let len = x.len();
    let padded: Vec<f32> = (0..len + 2 * pad)
        .map(|i| {
            let j = i as isize - pad as isize;
            let j = if j < 0 {
                -j
            } else if j >= len as isize {
                2 * (len as isize - 1) - j
            } else {
                j
            };
            x[j as usize]
        })
        .collect();
    let window: Vec<f32> = (0..N_FFT)
        .map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64).cos()) as f32)
        .collect();
    let (cos, sin): (Vec<f32>, Vec<f32>) = (0..BINS * N_FFT)
        .map(|i| {
            let (k, t) = (i / N_FFT, i % N_FFT);
            let a = 2.0 * std::f64::consts::PI * ((k * t) % N_FFT) as f64 / N_FFT as f64;
            (a.cos() as f32, a.sin() as f32)
        })
        .unzip();
    let mut mel = vec![0.0f32; n_mels * FRAMES];
    let columns: Vec<Vec<f32>> = (0..FRAMES)
        .into_par_iter()
        .map(|f| {
            let frame: Vec<f32> = padded[f * HOP..f * HOP + N_FFT]
                .iter()
                .zip(&window)
                .map(|(s, w)| s * w)
                .collect();
            let power: Vec<f32> = (0..BINS)
                .map(|k| {
                    let re = crate::engine::tensor::dot(&frame, &cos[k * N_FFT..][..N_FFT]);
                    let im = crate::engine::tensor::dot(&frame, &sin[k * N_FFT..][..N_FFT]);
                    re * re + im * im
                })
                .collect();
            (0..n_mels)
                .map(|m| {
                    let e = crate::engine::tensor::dot(&filters[m * BINS..][..BINS], &power);
                    e.max(1e-10).log10()
                })
                .collect()
        })
        .collect();
    for (f, col) in columns.iter().enumerate() {
        for (m, v) in col.iter().enumerate() {
            mel[m * FRAMES + f] = *v;
        }
    }
    let peak = mel.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for v in &mut mel {
        *v = (v.max(peak - 8.0) + 4.0) / 4.0;
    }
    mel
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_mel_scale_round_trips() {
        for f in [0.0, 440.0, 1000.0, 4000.0, 8000.0] {
            assert!((mel_to_hz(hz_to_mel(f)) - f).abs() < 1e-6);
        }
    }

    #[test]
    fn every_filter_has_weight_and_starts_at_zero_hz() {
        let fb = filterbank(80);
        for m in 0..80 {
            assert!(
                fb[m * BINS..][..BINS].iter().any(|&w| w > 0.0),
                "filter {m}"
            );
        }
        assert_eq!(fb[0], 0.0);
    }

    #[test]
    fn silence_is_a_flat_spectrogram() {
        let mel = log_mel(&[], &filterbank(80), 80);
        assert_eq!(mel.len(), 80 * FRAMES);
        assert!(mel.iter().all(|&v| (v - mel[0]).abs() < 1e-6));
    }
}
