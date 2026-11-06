//! KV rows in less host memory than `f32`.
//!
//! A sealed page of a [`crate::engine::kv_pool::KvPool`] never changes, so it
//! can sit in a compact format while the tail still being written stays `f32`.
//! Attention reads `f32`, so a layer's sealed rows are decoded into a scratch
//! buffer for the span of one attention call.
//!
//! The block formats quantize 32 elements together under one scale. A row
//! whose width is not a multiple of 32 pads its last block with zeros, and
//! decoding drops the padding.

const BLOCK: usize = 32;

/// How one side (keys or values) of a sealed row is stored on the host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowFormat {
    F32,
    /// An `f32` scale and 32 `i8`: 36 bytes a block, the device's own layout.
    Q8_0,
    /// An `f32` scale and 32 evenly spaced 4-bit levels: 20 bytes a block.
    Q4_0,
}

/// The levels of [`RowFormat::Q4_0`], indexed by a stored nibble.
const Q4_0_LEVELS: [i8; 16] = [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7];

impl RowFormat {
    /// Bytes `kv_dim` elements take.
    pub fn row_bytes(self, kv_dim: usize) -> usize {
        match self {
            RowFormat::F32 => kv_dim * 4,
            RowFormat::Q8_0 => kv_dim.div_ceil(BLOCK) * 36,
            RowFormat::Q4_0 => kv_dim.div_ceil(BLOCK) * 20,
        }
    }

    /// Writes whole rows of `kv_dim` elements into `out`.
    pub fn encode(self, rows: &[f32], kv_dim: usize, out: &mut [u8]) {
        debug_assert!(rows.len().is_multiple_of(kv_dim));
        let row_bytes = self.row_bytes(kv_dim);
        debug_assert_eq!(out.len(), rows.len() / kv_dim * row_bytes);
        for (row, out) in rows
            .chunks_exact(kv_dim)
            .zip(out.chunks_exact_mut(row_bytes))
        {
            match self {
                RowFormat::F32 => out.copy_from_slice(bytemuck::cast_slice(row)),
                RowFormat::Q8_0 | RowFormat::Q4_0 => {
                    let width = self.row_bytes(BLOCK);
                    for (values, out) in row.chunks(BLOCK).zip(out.chunks_exact_mut(width)) {
                        let mut block = [0f32; BLOCK];
                        block[..values.len()].copy_from_slice(values);
                        if self == RowFormat::Q8_0 {
                            encode_q8(&block, out);
                        } else {
                            encode_q4(&block, out);
                        }
                    }
                }
            }
        }
    }

    /// Reads whole rows of `kv_dim` elements from `raw`.
    pub fn decode(self, raw: &[u8], kv_dim: usize, out: &mut [f32]) {
        debug_assert!(out.len().is_multiple_of(kv_dim));
        debug_assert_eq!(raw.len(), out.len() / kv_dim * self.row_bytes(kv_dim));
        match self {
            RowFormat::F32 => bytemuck::cast_slice_mut::<f32, u8>(out).copy_from_slice(raw),
            RowFormat::Q8_0 | RowFormat::Q4_0 => {
                let row_bytes = self.row_bytes(kv_dim);
                let width = self.row_bytes(BLOCK);
                for (raw, out) in raw
                    .chunks_exact(row_bytes)
                    .zip(out.chunks_exact_mut(kv_dim))
                {
                    for (raw, out) in raw.chunks_exact(width).zip(out.chunks_mut(BLOCK)) {
                        if let Ok(out) = <&mut [f32; BLOCK]>::try_from(&mut *out) {
                            self.decode_block(raw, out);
                        } else {
                            let mut whole = [0f32; BLOCK];
                            self.decode_block(raw, &mut whole);
                            out.copy_from_slice(&whole[..out.len()]);
                        }
                    }
                }
            }
        }
    }

    /// Reads the elements of one row from `start`, as many as `out` holds.
    pub fn decode_span(self, row: &[u8], start: usize, out: &mut [f32]) {
        let end = start + out.len();
        match self {
            RowFormat::F32 => {
                bytemuck::cast_slice_mut::<f32, u8>(out).copy_from_slice(&row[start * 4..end * 4]);
            }
            RowFormat::Q8_0 | RowFormat::Q4_0 => {
                let width = self.row_bytes(BLOCK);
                let (mut at, mut left) = (start, &mut out[..]);
                while !left.is_empty() {
                    let raw = &row[at / BLOCK * width..][..width];
                    let skip = at % BLOCK;
                    let take = (BLOCK - skip).min(left.len());
                    let (this, later) = left.split_at_mut(take);
                    if let Ok(whole) = <&mut [f32; BLOCK]>::try_from(&mut *this) {
                        self.decode_block(raw, whole);
                    } else {
                        let mut whole = [0f32; BLOCK];
                        self.decode_block(raw, &mut whole);
                        this.copy_from_slice(&whole[skip..skip + take]);
                    }
                    at += take;
                    left = later;
                }
            }
        }
    }

    #[inline]
    fn decode_block(self, raw: &[u8], out: &mut [f32; BLOCK]) {
        let scale = f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
        let body = &raw[4..];
        if self == RowFormat::Q8_0 {
            decode_q8_block(scale, body, out);
        } else {
            decode_q4_block(scale, body, out);
        }
    }
}

fn encode_q8(block: &[f32; BLOCK], out: &mut [u8]) {
    let amax = block.iter().fold(0f32, |a, &b| a.max(b.abs()));
    let scale = amax / 127.0;
    let inverse = if scale > 0.0 { 1.0 / scale } else { 0.0 };
    out[..4].copy_from_slice(&scale.to_le_bytes());
    for (slot, &v) in out[4..].iter_mut().zip(block) {
        *slot = ((v * inverse).round().clamp(-127.0, 127.0) as i8) as u8;
    }
}

/// The element furthest from zero sets a scale that reaches it exactly at the
/// lowest level, -8, whichever its sign: that gives the block its last level.
fn encode_q4(block: &[f32; BLOCK], out: &mut [u8]) {
    let peak = block
        .iter()
        .fold(0f32, |peak, &v| if v.abs() > peak.abs() { v } else { peak });
    let scale = peak / -8.0;
    let inverse = if scale != 0.0 { 1.0 / scale } else { 0.0 };
    out[..4].copy_from_slice(&scale.to_le_bytes());
    let mut levels = [0u8; BLOCK];
    for (level, &v) in levels.iter_mut().zip(block) {
        *level = ((v * inverse).round().clamp(-8.0, 7.0) as i32 + 8) as u8;
    }
    write_nibbles(&levels, &mut out[4..]);
}

fn write_nibbles(levels: &[u8; BLOCK], out: &mut [u8]) {
    for (pair, out) in levels.as_chunks::<2>().0.iter().zip(out.iter_mut()) {
        *out = pair[0] | (pair[1] << 4);
    }
}

#[inline]
fn decode_q8_block(scale: f32, body: &[u8], out: &mut [f32; BLOCK]) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: the CPU has AVX2, and `body` holds at least 32 bytes.
        unsafe { widen_i8_avx2(body.as_ptr().cast(), scale, out) };
        return;
    }
    for (out, &q) in out.iter_mut().zip(body) {
        *out = f32::from(q as i8) * scale;
    }
}

#[inline]
fn decode_q4_block(scale: f32, body: &[u8], out: &mut [f32; BLOCK]) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: the CPU has AVX2, and `body` holds at least 16 bytes.
        unsafe { decode_nibbles_avx2(body, scale, out) };
        return;
    }
    for (pair, &byte) in out.as_chunks_mut::<2>().0.iter_mut().zip(body) {
        pair[0] = f32::from(Q4_0_LEVELS[(byte & 15) as usize]) * scale;
        pair[1] = f32::from(Q4_0_LEVELS[(byte >> 4) as usize]) * scale;
    }
}

/// # Safety
///
/// The CPU has AVX2, and `from` can be read for 32 bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn widen_i8_avx2(from: *const i8, scale: f32, out: &mut [f32; BLOCK]) {
    use std::arch::x86_64::*;
    let scale = _mm256_set1_ps(scale);
    for (i, out) in out.as_chunks_mut::<8>().0.iter_mut().enumerate() {
        // SAFETY: `i` is below 4, so eight bytes at `i * 8` are inside `from`,
        // and `out` is eight floats.
        unsafe {
            let bytes = _mm_loadl_epi64(from.add(i * 8).cast());
            let floats = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes));
            _mm256_storeu_ps(out.as_mut_ptr(), _mm256_mul_ps(floats, scale));
        }
    }
}

/// Spreads 16 bytes to 32 nibbles, looks each up in [`Q4_0_LEVELS`], then widens.
///
/// # Safety
///
/// The CPU has AVX2, and `body` holds at least 16 bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn decode_nibbles_avx2(body: &[u8], scale: f32, out: &mut [f32; BLOCK]) {
    use std::arch::x86_64::*;
    debug_assert!(body.len() >= 16);
    let mut bytes = [0i8; BLOCK];
    // SAFETY: `body` holds 16 bytes, the levels 16 and `bytes` 32.
    unsafe {
        let packed = _mm256_cvtepu8_epi16(_mm_loadu_si128(body.as_ptr().cast()));
        let low_nibble = _mm256_set1_epi16(0x000f);
        let spread = _mm256_or_si256(
            _mm256_and_si256(packed, low_nibble),
            _mm256_slli_epi16::<8>(_mm256_srli_epi16::<4>(packed)),
        );
        let table = _mm256_broadcastsi128_si256(_mm_loadu_si128(Q4_0_LEVELS.as_ptr().cast()));
        let looked_up = _mm256_shuffle_epi8(table, spread);
        _mm256_storeu_si256(bytes.as_mut_ptr().cast(), looked_up);
        widen_i8_avx2(bytes.as_ptr(), scale, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [RowFormat; 3] = [RowFormat::F32, RowFormat::Q8_0, RowFormat::Q4_0];

    fn sample(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32) * 0.37).sin() * (1.0 + (i % 7) as f32 * 0.4))
            .collect()
    }

    fn round_trip(format: RowFormat, rows: &[f32], kv_dim: usize) -> Vec<f32> {
        let mut raw = vec![0u8; rows.len() / kv_dim * format.row_bytes(kv_dim)];
        format.encode(rows, kv_dim, &mut raw);
        let mut back = vec![f32::NAN; rows.len()];
        format.decode(&raw, kv_dim, &mut back);
        back
    }

    fn worst_error(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).fold(0f32, |m, (a, b)| m.max((a - b).abs()))
    }

    #[test]
    fn a_format_takes_the_bytes_it_says() {
        assert_eq!(RowFormat::F32.row_bytes(64), 256);
        assert_eq!(RowFormat::Q8_0.row_bytes(64), 72);
        assert_eq!(RowFormat::Q4_0.row_bytes(64), 40);
        // A short last block takes a whole block.
        assert_eq!(RowFormat::Q8_0.row_bytes(33), 72);
    }

    #[test]
    fn every_format_comes_back_within_its_step() {
        // Elements reach about 3.4, so the steps are these fractions of it.
        let limits = [
            (RowFormat::F32, 0.0),
            (RowFormat::Q8_0, 3.4 / 127.0),
            (RowFormat::Q4_0, 3.4 / 8.0),
        ];
        for kv_dim in [32, 64, 96, 320, 448] {
            let rows = sample(kv_dim * 5);
            for (format, limit) in limits {
                let back = round_trip(format, &rows, kv_dim);
                let worst = worst_error(&rows, &back);
                assert!(worst <= limit, "{format:?} at {kv_dim}: {worst}");
            }
        }
    }

    #[test]
    fn a_row_that_is_not_whole_blocks_keeps_its_last_elements() {
        for format in ALL {
            for kv_dim in [1, 7, 31, 33, 57, 100] {
                let rows = sample(kv_dim * 3);
                let back = round_trip(format, &rows, kv_dim);
                assert_eq!(back.len(), rows.len());
                assert!(back.iter().all(|v| v.is_finite()), "{format:?} at {kv_dim}");
                assert!(worst_error(&rows, &back) < 3.4, "{format:?} at {kv_dim}");
            }
        }
    }

    /// A head may start and end inside a block when its width is not a
    /// multiple of the block's.
    #[test]
    fn a_span_reads_what_the_whole_row_reads() {
        for format in ALL {
            for kv_dim in [32, 64, 96, 100] {
                let rows = sample(kv_dim);
                let mut raw = vec![0u8; format.row_bytes(kv_dim)];
                format.encode(&rows, kv_dim, &mut raw);
                let mut whole = vec![0f32; kv_dim];
                format.decode(&raw, kv_dim, &mut whole);
                for (start, len) in [
                    (0, kv_dim),
                    (0, 1),
                    (kv_dim - 1, 1),
                    (5, 27),
                    (16, 16),
                    (31, 1),
                ] {
                    let mut span = vec![f32::NAN; len];
                    format.decode_span(&raw, start, &mut span);
                    assert_eq!(
                        span,
                        whole[start..start + len],
                        "{format:?} {kv_dim} {start} {len}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_row_of_zeros_stays_zero() {
        for format in ALL {
            let rows = vec![0.0; 64];
            assert_eq!(round_trip(format, &rows, 64), rows, "{format:?}");
        }
    }

    /// Reading the same bytes must not depend on which kernel the CPU gets.
    #[test]
    fn the_fast_and_plain_decoders_agree() {
        let rows = sample(BLOCK);
        let mut raw = vec![0u8; RowFormat::Q4_0.row_bytes(BLOCK)];
        RowFormat::Q4_0.encode(&rows, BLOCK, &mut raw);
        let scale = f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
        let mut plain = [0f32; BLOCK];
        for (pair, &byte) in plain.as_chunks_mut::<2>().0.iter_mut().zip(&raw[4..]) {
            pair[0] = f32::from(Q4_0_LEVELS[(byte & 15) as usize]) * scale;
            pair[1] = f32::from(Q4_0_LEVELS[(byte >> 4) as usize]) * scale;
        }
        let mut fast = [0f32; BLOCK];
        decode_q4_block(scale, &raw[4..], &mut fast);
        assert_eq!(plain, fast);
        let mut raw = vec![0u8; RowFormat::Q8_0.row_bytes(BLOCK)];
        RowFormat::Q8_0.encode(&rows, BLOCK, &mut raw);
        let scale = f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
        let plain: Vec<f32> = raw[4..]
            .iter()
            .map(|&q| f32::from(q as i8) * scale)
            .collect();
        let mut fast = [0f32; BLOCK];
        decode_q8_block(scale, &raw[4..], &mut fast);
        assert_eq!(plain, fast);
    }

    /// The same bytes the device's upload path writes, so one layout has one
    /// meaning.
    #[test]
    fn q8_0_matches_the_device_layout() {
        let rows = sample(BLOCK * 4);
        let mut raw = vec![0u8; RowFormat::Q8_0.row_bytes(BLOCK) * 4];
        RowFormat::Q8_0.encode(&rows, BLOCK, &mut raw);
        assert_eq!(raw, crate::engine::kv_cache::f32_to_q8_0_bytes(&rows));
    }

    #[test]
    fn extreme_values_do_not_produce_garbage() {
        for format in [RowFormat::Q8_0, RowFormat::Q4_0] {
            let rows = vec![1e30f32; BLOCK];
            assert!(
                round_trip(format, &rows, BLOCK)
                    .iter()
                    .all(|v| v.is_finite()),
                "{format:?}"
            );
            let tiny = vec![1e-30f32; BLOCK];
            assert!(
                round_trip(format, &tiny, BLOCK)
                    .iter()
                    .all(|v| v.is_finite()),
                "{format:?}"
            );
        }
    }
}
