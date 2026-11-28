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

//! RIFF WAVE in, mono 16 kHz `f32` out: 8/16/24/32-bit integer PCM and
//! 32/64-bit float, plain or `WAVE_FORMAT_EXTENSIBLE`, any channel count
//! (averaged) and any rate (resampled linearly).

use anyhow::{Result, bail, ensure};

pub const SAMPLE_RATE: u32 = 16_000;

/// The samples of a WAV file as mono at `SAMPLE_RATE`.
pub fn decode(bytes: &[u8]) -> Result<Vec<f32>> {
    ensure!(
        bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE",
        "not a RIFF WAVE file"
    );
    let mut fmt: Option<(u16, u16, u32, u16)> = None;
    let mut data: Option<&[u8]> = None;
    let mut at = 12;
    while at + 8 <= bytes.len() {
        let id = &bytes[at..at + 4];
        let len = u32::from_le_bytes(bytes[at + 4..at + 8].try_into()?) as usize;
        let body = &bytes[at + 8..(at + 8 + len).min(bytes.len())];
        match id {
            b"fmt " => {
                ensure!(body.len() >= 16, "the fmt chunk is too short");
                let u16_at = |i: usize| u16::from_le_bytes([body[i], body[i + 1]]);
                let mut format = u16_at(0);
                if format == 0xFFFE && body.len() >= 26 {
                    format = u16_at(24);
                }
                let rate = u32::from_le_bytes(body[4..8].try_into()?);
                fmt = Some((format, u16_at(2), rate, u16_at(14)));
            }
            b"data" => data = Some(body),
            _ => {}
        }
        at += 8 + len + (len & 1);
    }
    let Some((format, channels, rate, bits)) = fmt else {
        bail!("the WAV file has no fmt chunk");
    };
    let Some(data) = data else {
        bail!("the WAV file has no data chunk");
    };
    ensure!(
        channels > 0 && rate > 0,
        "the WAV file has no channels or rate"
    );
    let width = (bits as usize).div_ceil(8);
    let sample = |b: &[u8]| -> Result<f32> {
        Ok(match (format, bits) {
            (1, 8) => (b[0] as f32 - 128.0) / 128.0,
            (1, 16) => i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0,
            (1, 24) => (i32::from_le_bytes([0, b[0], b[1], b[2]]) >> 8) as f32 / 8_388_608.0,
            (1, 32) => i32::from_le_bytes(b.try_into()?) as f32 / 2_147_483_648.0,
            (3, 32) => f32::from_le_bytes(b.try_into()?),
            (3, 64) => f64::from_le_bytes(b.try_into()?) as f32,
            _ => bail!("WAV format {format} at {bits} bits is not read"),
        })
    };
    let frame = width * channels as usize;
    let mono = data
        .chunks_exact(frame)
        .map(|f| {
            let mut sum = 0.0;
            for c in f.chunks_exact(width) {
                sum += sample(c)?;
            }
            Ok(sum / channels as f32)
        })
        .collect::<Result<Vec<f32>>>()?;
    Ok(resample(&mono, rate, SAMPLE_RATE))
}

/// Linear interpolation from `from` Hz to `to` Hz.
pub fn resample(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    let n = (x.len() as u64 * to as u64 / from as u64) as usize;
    let step = from as f64 / to as f64;
    (0..n)
        .map(|i| {
            let p = i as f64 * step;
            let j = p as usize;
            let f = (p - j as f64) as f32;
            let a = x[j.min(x.len() - 1)];
            let b = x[(j + 1).min(x.len() - 1)];
            a + (b - a) * f
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav16(rate: u32, channels: u16, samples: &[i16]) -> Vec<u8> {
        let data: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        out.extend_from_slice(b"WAVEfmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&channels.to_le_bytes());
        out.extend_from_slice(&rate.to_le_bytes());
        out.extend_from_slice(&(rate * 2 * channels as u32).to_le_bytes());
        out.extend_from_slice(&(2 * channels).to_le_bytes());
        out.extend_from_slice(&16u16.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&data);
        out
    }

    #[test]
    fn stereo_is_averaged_to_mono() {
        let x = decode(&wav16(16_000, 2, &[16384, 0, -16384, -16384])).unwrap();
        assert_eq!(x, vec![0.25, -0.5]);
    }

    #[test]
    fn a_higher_rate_is_brought_down_to_16k() {
        let x = decode(&wav16(32_000, 1, &[0; 3200])).unwrap();
        assert_eq!(x.len(), 1600);
    }
}
