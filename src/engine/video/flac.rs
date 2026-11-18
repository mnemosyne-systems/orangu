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

//! A small FLAC encoder, for the sound track of a generated clip: 16-bit
//! PCM in frames of 4096 samples, each channel predicted by the best of
//! FLAC's fixed polynomial predictors (orders 0 to 4) and its residual
//! Rice-coded with one parameter per partition. Lossless, about half the
//! size of the PCM, and playable inside an MP4 by every current browser.
//!
//! Only what the MP4 needs is produced: the `STREAMINFO` block (for the
//! `dfLa` box) and the frames, one MP4 sample each.

/// Samples per frame (per channel).
pub const BLOCK: usize = 4096;

/// The encoded stream.
pub struct Flac {
    /// The 34-byte `STREAMINFO` body.
    pub streaminfo: Vec<u8>,
    /// Each frame's bytes and its sample count.
    pub frames: Vec<(Vec<u8>, usize)>,
    pub sample_rate: u32,
    pub channels: u32,
}

struct Bits {
    out: Vec<u8>,
    acc: u64,
    n: u32,
}

impl Bits {
    fn new() -> Self {
        Self {
            out: Vec::new(),
            acc: 0,
            n: 0,
        }
    }

    fn put(&mut self, value: u64, bits: u32) {
        debug_assert!(bits <= 32);
        if bits == 0 {
            return;
        }
        self.acc = (self.acc << bits) | (value & ((1u64 << bits) - 1));
        self.n += bits;
        while self.n >= 8 {
            self.n -= 8;
            self.out.push((self.acc >> self.n) as u8);
        }
    }

    fn put_signed(&mut self, value: i64, bits: u32) {
        self.put(value as u64 & ((1u64 << bits) - 1), bits);
    }

    fn unary(&mut self, zeros: u64) {
        let mut z = zeros;
        while z >= 32 {
            self.put(0, 32);
            z -= 32;
        }
        self.put(1, z as u32 + 1);
    }

    fn align(&mut self) {
        if self.n > 0 {
            self.put(0, 8 - self.n);
        }
    }

    fn bytes(mut self) -> Vec<u8> {
        self.align();
        self.out
    }
}

fn crc8(data: &[u8]) -> u8 {
    let mut crc = 0u8;
    for &b in data {
        crc ^= b;
        for _ in 0..8 {
            crc = if crc & 0x80 != 0 {
                (crc << 1) ^ 0x07
            } else {
                crc << 1
            };
        }
    }
    crc
}

fn crc16(data: &[u8]) -> u16 {
    let mut crc = 0u16;
    for &b in data {
        crc ^= (b as u16) << 8;
        for _ in 0..8 {
            crc = if crc & 0x8000 != 0 {
                (crc << 1) ^ 0x8005
            } else {
                crc << 1
            };
        }
    }
    crc
}

/// FLAC's "UTF-8" coding of a frame number.
fn utf8_number(v: u64) -> Vec<u8> {
    if v < 0x80 {
        return vec![v as u8];
    }
    let mut bytes = Vec::new();
    let mut v = v;
    let mut payload = 6;
    let mut n = 1;
    while v >= (1u64 << payload) {
        bytes.push(0x80 | (v & 0x3F) as u8);
        v >>= 6;
        n += 1;
        payload -= 1;
    }
    let lead = (0xFFu16 << (8 - n)) as u8 | v as u8;
    bytes.push(lead);
    bytes.reverse();
    bytes
}

/// The residual of the fixed predictor of `order` (FLAC's 0..=4).
fn fixed_residual(x: &[i32], order: usize) -> Vec<i64> {
    let x: Vec<i64> = x.iter().map(|&v| v as i64).collect();
    (order..x.len())
        .map(|i| match order {
            0 => x[i],
            1 => x[i] - x[i - 1],
            2 => x[i] - 2 * x[i - 1] + x[i - 2],
            3 => x[i] - 3 * x[i - 1] + 3 * x[i - 2] - x[i - 3],
            _ => x[i] - 4 * x[i - 1] + 6 * x[i - 2] - 4 * x[i - 3] + x[i - 4],
        })
        .collect()
}

fn zigzag(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

/// The bits a Rice code of `param` spends on `res`.
fn rice_bits(res: &[i64], param: u32) -> u64 {
    res.iter()
        .map(|&r| (zigzag(r) >> param) + 1 + param as u64)
        .sum()
}

fn best_param(res: &[i64]) -> (u32, u64) {
    (0..=14u32)
        .map(|p| (p, rice_bits(res, p)))
        .min_by_key(|&(_, bits)| bits)
        .expect("fifteen candidates")
}

/// The partition order (0..=4) and parameters that code `res` (after
/// `order` warm-up samples of a `block`) the smallest.
fn best_partitioning(res: &[i64], order: usize, block: usize) -> (u32, Vec<u32>, u64) {
    let mut best: Option<(u32, Vec<u32>, u64)> = None;
    for po in 0..=4u32 {
        let parts = 1usize << po;
        if !block.is_multiple_of(parts) || block / parts <= order {
            break;
        }
        let per = block / parts;
        let mut params = Vec::with_capacity(parts);
        let mut bits = 0u64;
        let mut start = 0usize;
        for p in 0..parts {
            let n = if p == 0 { per - order } else { per };
            let (param, b) = best_param(&res[start..start + n]);
            params.push(param);
            bits += b + 4;
            start += n;
        }
        if best.as_ref().is_none_or(|b| bits < b.2) {
            best = Some((po, params, bits));
        }
    }
    best.expect("partition order 0 always fits")
}

fn subframe(out: &mut Bits, x: &[i32]) {
    let block = x.len();
    // A constant block is one sample.
    if x.iter().all(|&v| v == x[0]) {
        out.put(0, 1);
        out.put(0, 6);
        out.put(0, 1);
        out.put_signed(x[0] as i64, 16);
        return;
    }
    let max_order = 4.min(block - 1);
    // (order, residual, (partition order, parameters, bits)).
    type Choice = (usize, Vec<i64>, (u32, Vec<u32>, u64));
    let mut best: Option<Choice> = None;
    for order in 0..=max_order {
        let res = fixed_residual(x, order);
        let part = best_partitioning(&res, order, block);
        let total = part.2 + 16 * order as u64;
        if best
            .as_ref()
            .is_none_or(|b| total < b.2.2 + 16 * b.0 as u64)
        {
            best = Some((order, res, part));
        }
    }
    let (order, res, (po, params, bits)) = best.expect("order 0 always fits");
    // Verbatim when prediction does not pay.
    if bits + 16 * order as u64 >= 16 * block as u64 {
        out.put(0, 1);
        out.put(1, 6);
        out.put(0, 1);
        for &v in x {
            out.put_signed(v as i64, 16);
        }
        return;
    }
    out.put(0, 1);
    out.put(0b001000 | order as u64, 6);
    out.put(0, 1);
    for &v in &x[..order] {
        out.put_signed(v as i64, 16);
    }
    // Rice coding, 4-bit parameters.
    out.put(0, 2);
    out.put(po as u64, 4);
    let parts = 1usize << po;
    let per = block / parts;
    let mut start = 0;
    for (p, &param) in params.iter().enumerate() {
        out.put(param as u64, 4);
        let n = if p == 0 { per - order } else { per };
        for &r in &res[start..start + n] {
            let z = zigzag(r);
            out.unary(z >> param);
            out.put(z & ((1u64 << param) - 1), param);
        }
        start += n;
    }
}

/// Encodes interleaved samples in `[-1, 1]`.
pub fn encode(samples: &[f32], channels: usize, sample_rate: u32) -> Flac {
    assert!((1..=8).contains(&channels));
    let pcm: Vec<i32> = samples
        .iter()
        .map(|&v| (v.clamp(-1.0, 1.0) * 32767.0).round() as i32)
        .collect();
    let total = pcm.len() / channels;
    let mut frames = Vec::new();
    let (mut min_frame, mut max_frame) = (u32::MAX, 0u32);
    let mut start = 0usize;
    let mut number = 0u64;
    while start < total {
        let block = BLOCK.min(total - start);
        let mut hdr = Bits::new();
        hdr.put(0b11111111111110, 14);
        hdr.put(0, 1);
        hdr.put(0, 1); // fixed blocksize
        let (bs_code, bs_tail) = if block == BLOCK {
            (12u64, None)
        } else {
            (7u64, Some(block - 1))
        };
        hdr.put(bs_code, 4);
        hdr.put(0, 4); // sample rate from STREAMINFO
        hdr.put(channels as u64 - 1, 4); // independent channels
        hdr.put(0b100, 3); // 16 bits per sample
        hdr.put(0, 1);
        let mut head = hdr.bytes();
        head.extend(utf8_number(number));
        if let Some(t) = bs_tail {
            head.extend((t as u16).to_be_bytes());
        }
        head.push(crc8(&head));
        let mut body = Bits::new();
        for c in 0..channels {
            let x: Vec<i32> = (0..block)
                .map(|i| pcm[(start + i) * channels + c])
                .collect();
            subframe(&mut body, &x);
        }
        let mut frame = head;
        frame.extend(body.bytes());
        frame.extend(crc16(&frame).to_be_bytes());
        min_frame = min_frame.min(frame.len() as u32);
        max_frame = max_frame.max(frame.len() as u32);
        frames.push((frame, block));
        start += block;
        number += 1;
    }
    let mut info = Bits::new();
    info.put(BLOCK as u64, 16);
    info.put(BLOCK as u64, 16);
    info.put(
        if frames.is_empty() {
            0
        } else {
            min_frame as u64
        },
        24,
    );
    info.put(max_frame as u64, 24);
    info.put(sample_rate as u64, 20);
    info.put(channels as u64 - 1, 3);
    info.put(15, 5);
    info.put(total as u64 >> 32, 4);
    info.put(total as u64 & 0xFFFF_FFFF, 32);
    let mut streaminfo = info.bytes();
    streaminfo.extend([0u8; 16]); // MD5 unknown
    debug_assert_eq!(streaminfo.len(), 34);
    Flac {
        streaminfo,
        frames,
        sample_rate,
        channels: channels as u32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_crcs_match_the_reference_vectors() {
        assert_eq!(crc8(b"123456789"), 0xF4);
        assert_eq!(crc16(b"123456789"), 0xFEE8);
    }

    #[test]
    fn frame_numbers_are_utf8_coded() {
        assert_eq!(utf8_number(5), vec![5]);
        assert_eq!(utf8_number(0x80), vec![0xC2, 0x80]);
        assert_eq!(utf8_number(0x800), vec![0xE0, 0xA0, 0x80]);
    }

    #[test]
    fn a_sine_compresses_and_frames_cover_every_sample() {
        let rate = 48000;
        let samples: Vec<f32> = (0..rate)
            .flat_map(|i| {
                let v = (i as f32 * 440.0 * std::f32::consts::TAU / rate as f32).sin() * 0.5;
                [v, -v]
            })
            .collect();
        let flac = encode(&samples, 2, rate as u32);
        let covered: usize = flac.frames.iter().map(|f| f.1).sum();
        assert_eq!(covered, rate);
        let bytes: usize = flac.frames.iter().map(|f| f.0.len()).sum();
        assert!(
            bytes < rate * 4 / 2,
            "{bytes} bytes for {} PCM bytes",
            rate * 4
        );
        assert_eq!(flac.streaminfo.len(), 34);
    }
}
