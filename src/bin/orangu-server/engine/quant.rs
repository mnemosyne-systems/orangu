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

//! Dequantization for the `ggml_type`s a GGUF tensor can be stored as.
//! Struct layouts and algorithms are taken directly from ggml's own
//! `ggml-common.h`/`ggml-quants.c` (`dequantize_row_*`), not reimplemented
//! from a description — bit-for-bit compatible with what llama.cpp itself
//! reads.
//!
//! Only the types actually shipped by the overwhelming majority of GGUF
//! releases in circulation are supported: `F32`, `F16`, `Q8_0`, `Q4_0`,
//! `Q4_K`, `Q6_K`. Anything else fails with a clear "not yet supported"
//! error naming the type, rather than silently misreading the bytes — see
//! `doc/SERVER_ROADMAP.md` for what's missing.

use anyhow::{Result, bail};
use half::f16;

use orangu::gguf::ggml_type_name;

// ggml_type ids, from ggml.h. `pub(crate)` so `engine::backend::vulkan`'s
// shader dispatch table can key off the exact same ids rather than a
// hand-copied second list that could drift from this one.
pub(crate) const GGML_TYPE_F32: u32 = 0;
pub(crate) const GGML_TYPE_F16: u32 = 1;
pub(crate) const GGML_TYPE_Q4_0: u32 = 2;
pub(crate) const GGML_TYPE_Q5_0: u32 = 6;
pub(crate) const GGML_TYPE_Q8_0: u32 = 8;
pub(crate) const GGML_TYPE_Q4_K: u32 = 12;
pub(crate) const GGML_TYPE_Q5_K: u32 = 13;
pub(crate) const GGML_TYPE_Q6_K: u32 = 14;
pub(crate) const GGML_TYPE_BF16: u32 = 30;

const QK4_0: usize = 32;
const QK5_0: usize = 32;
const QK8_0: usize = 32;
const QK_K: usize = 256;
const K_SCALE_SIZE: usize = 12;

/// Bytes per block, and elements per block, for a supported `ggml_type`.
/// `None` for a type this engine can't yet read.
fn block_layout(ggml_type: u32) -> Option<(usize, usize)> {
    match ggml_type {
        GGML_TYPE_F32 => Some((4, 1)),
        GGML_TYPE_F16 => Some((2, 1)),
        GGML_TYPE_BF16 => Some((2, 1)),
        GGML_TYPE_Q4_0 => Some((2 + QK4_0 / 2, QK4_0)),
        GGML_TYPE_Q5_0 => Some((2 + 4 + QK5_0 / 2, QK5_0)),
        GGML_TYPE_Q8_0 => Some((2 + QK8_0, QK8_0)),
        GGML_TYPE_Q4_K => Some((2 + 2 + K_SCALE_SIZE + QK_K / 2, QK_K)),
        GGML_TYPE_Q5_K => Some((2 + 2 + K_SCALE_SIZE + QK_K / 8 + QK_K / 2, QK_K)),
        GGML_TYPE_Q6_K => Some((QK_K / 2 + QK_K / 4 + QK_K / 16 + 2, QK_K)),
        _ => None,
    }
}

/// The exact byte length a tensor with `element_count` elements of
/// `ggml_type` occupies in the GGUF file's data section.
pub fn tensor_byte_size(ggml_type: u32, element_count: u64) -> Result<u64> {
    let Some((block_bytes, block_elems)) = block_layout(ggml_type) else {
        bail!(
            "tensor type {} is not yet supported by orangu-server (see doc/SERVER_ROADMAP.md)",
            ggml_type_name(ggml_type)
        );
    };
    if !(element_count as usize).is_multiple_of(block_elems) {
        bail!(
            "tensor element count {element_count} is not a multiple of the {} block size for {}",
            block_elems,
            ggml_type_name(ggml_type)
        );
    }
    let blocks = element_count / block_elems as u64;
    Ok(blocks * block_bytes as u64)
}

/// Dequantizes `bytes` (exactly `tensor_byte_size(ggml_type, element_count)`
/// long) to `element_count` `f32` values, in the tensor's original order.
pub fn dequantize(ggml_type: u32, bytes: &[u8], element_count: usize) -> Result<Vec<f32>> {
    match ggml_type {
        GGML_TYPE_F32 => Ok(bytes
            .chunks_exact(4)
            .take(element_count)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()),
        GGML_TYPE_F16 => Ok(bytes
            .chunks_exact(2)
            .take(element_count)
            .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect()),
        // bfloat16: the top 16 bits of an f32 (sign + 8-bit exponent + 7-bit
        // mantissa) — reconstruct by left-shifting into the low bits' place.
        GGML_TYPE_BF16 => Ok(bytes
            .chunks_exact(2)
            .take(element_count)
            .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
            .collect()),
        GGML_TYPE_Q4_0 => Ok(dequantize_q4_0(bytes, element_count)),
        GGML_TYPE_Q5_0 => Ok(dequantize_q5_0(bytes, element_count)),
        GGML_TYPE_Q8_0 => Ok(dequantize_q8_0(bytes, element_count)),
        GGML_TYPE_Q4_K => Ok(dequantize_q4_k(bytes, element_count)),
        GGML_TYPE_Q5_K => Ok(dequantize_q5_k(bytes, element_count)),
        GGML_TYPE_Q6_K => Ok(dequantize_q6_k(bytes, element_count)),
        _ => bail!(
            "tensor type {} is not yet supported by orangu-server (see doc/SERVER_ROADMAP.md)",
            ggml_type_name(ggml_type)
        ),
    }
}

fn read_f16(bytes: &[u8], offset: usize) -> f32 {
    f16::from_le_bytes([bytes[offset], bytes[offset + 1]]).to_f32()
}

/// `block_q4_0`: `{ d: f16, qs: [u8; 16] }`, 32 elements — mirrors ggml's
/// `dequantize_row_q4_0` exactly (signed nibbles, offset by 8).
fn dequantize_q4_0(bytes: &[u8], element_count: usize) -> Vec<f32> {
    const BLOCK_BYTES: usize = 2 + QK4_0 / 2;
    let mut out = Vec::with_capacity(element_count);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let d = read_f16(block, 0);
        let qs = &block[2..];
        let mut lo = [0f32; QK4_0 / 2];
        let mut hi = [0f32; QK4_0 / 2];
        for (j, &byte) in qs.iter().enumerate() {
            lo[j] = ((byte & 0x0F) as i32 - 8) as f32 * d;
            hi[j] = ((byte >> 4) as i32 - 8) as f32 * d;
        }
        out.extend_from_slice(&lo);
        out.extend_from_slice(&hi);
    }
    out.truncate(element_count);
    out
}

/// `block_q5_0`: `{ d: f16, qh: [u8; 4], qs: [u8; 16] }`, 32 elements — a
/// 5-bit nibble (4 low bits in `qs`, the 5th/high bit packed across `qh`),
/// offset by 16 (the 5-bit analogue of Q4_0's offset-by-8), mirrors ggml's
/// `dequantize_row_q5_0`.
fn dequantize_q5_0(bytes: &[u8], element_count: usize) -> Vec<f32> {
    const BLOCK_BYTES: usize = 2 + 4 + QK5_0 / 2;
    let mut out = Vec::with_capacity(element_count);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let d = read_f16(block, 0);
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        let qs = &block[6..];
        let mut lo = [0f32; QK5_0 / 2];
        let mut hi = [0f32; QK5_0 / 2];
        for (j, &byte) in qs.iter().enumerate() {
            let xh_0 = ((qh >> j) << 4) & 0x10;
            let xh_1 = (qh >> (j + 12)) & 0x10;
            lo[j] = (((byte & 0x0F) as u32 | xh_0) as i32 - 16) as f32 * d;
            hi[j] = (((byte >> 4) as u32 | xh_1) as i32 - 16) as f32 * d;
        }
        out.extend_from_slice(&lo);
        out.extend_from_slice(&hi);
    }
    out.truncate(element_count);
    out
}

/// `block_q8_0`: `{ d: f16, qs: [i8; 32] }`, 32 elements.
fn dequantize_q8_0(bytes: &[u8], element_count: usize) -> Vec<f32> {
    const BLOCK_BYTES: usize = 2 + QK8_0;
    let mut out = Vec::with_capacity(element_count);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let d = read_f16(block, 0);
        let qs = &block[2..];
        out.extend(qs.iter().map(|&q| (q as i8) as f32 * d));
    }
    out.truncate(element_count);
    out
}

/// ggml's `get_scale_min_k4`: unpacks the 6-bit scale and 6-bit min for
/// sub-block `j` (0..8) of a `Q4_K`/`Q5_K` super-block's 12-byte `scales`.
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// `block_q4_K`: `{ d: f16, dmin: f16, scales: [u8; 12], qs: [u8; 128] }`,
/// 256 elements (8 sub-blocks of 32) — mirrors ggml's `dequantize_row_q4_K`.
fn dequantize_q4_k(bytes: &[u8], element_count: usize) -> Vec<f32> {
    const BLOCK_BYTES: usize = 2 + 2 + K_SCALE_SIZE + QK_K / 2;
    let mut out = Vec::with_capacity(element_count);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let d = read_f16(block, 0);
        let dmin = read_f16(block, 2);
        let scales = &block[4..4 + K_SCALE_SIZE];
        let qs = &block[4 + K_SCALE_SIZE..];

        let mut is = 0;
        let mut q_offset = 0;
        while q_offset < QK_K {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (d1, m1) = (d * sc1 as f32, dmin * m1 as f32);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let (d2, m2) = (d * sc2 as f32, dmin * m2 as f32);

            let q = &qs[q_offset / 2..q_offset / 2 + 32];
            for &byte in q {
                out.push(d1 * (byte & 0x0F) as f32 - m1);
            }
            for &byte in q {
                out.push(d2 * (byte >> 4) as f32 - m2);
            }

            is += 2;
            q_offset += 64;
        }
    }
    out.truncate(element_count);
    out
}

/// `block_q5_K`: `{ d: f16, dmin: f16, scales: [u8; 12], qh: [u8; 32],
/// qs: [u8; 128] }`, 256 elements — mirrors ggml's `dequantize_row_q5_K`:
/// like `Q4_K`, plus a 5th quant bit packed across `qh` (each `qh` byte's 8
/// bits are consumed one pair per 64-element sub-group, over all 4 groups).
fn dequantize_q5_k(bytes: &[u8], element_count: usize) -> Vec<f32> {
    const BLOCK_BYTES: usize = 2 + 2 + K_SCALE_SIZE + QK_K / 8 + QK_K / 2;
    let mut out = Vec::with_capacity(element_count);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let d = read_f16(block, 0);
        let dmin = read_f16(block, 2);
        let scales = &block[4..4 + K_SCALE_SIZE];
        let qh = &block[4 + K_SCALE_SIZE..4 + K_SCALE_SIZE + QK_K / 8];
        let qs = &block[4 + K_SCALE_SIZE + QK_K / 8..];

        let mut is = 0;
        let (mut u1, mut u2) = (1u8, 2u8);
        let mut ql_offset = 0;
        let mut q_offset = 0;
        while q_offset < QK_K {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (d1, m1) = (d * sc1 as f32, dmin * m1 as f32);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let (d2, m2) = (d * sc2 as f32, dmin * m2 as f32);

            let ql = &qs[ql_offset..ql_offset + 32];
            for (l, &byte) in ql.iter().enumerate() {
                let hi_bit = if qh[l] & u1 != 0 { 16 } else { 0 };
                out.push(d1 * ((byte & 0x0F) as i32 + hi_bit) as f32 - m1);
            }
            for (l, &byte) in ql.iter().enumerate() {
                let hi_bit = if qh[l] & u2 != 0 { 16 } else { 0 };
                out.push(d2 * ((byte >> 4) as i32 + hi_bit) as f32 - m2);
            }

            ql_offset += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
            q_offset += 64;
        }
    }
    out.truncate(element_count);
    out
}

/// `block_q6_K`: `{ ql: [u8; 128], qh: [u8; 64], scales: [i8; 16], d: f16 }`,
/// 256 elements — mirrors ggml's `dequantize_row_q6_K`.
fn dequantize_q6_k(bytes: &[u8], element_count: usize) -> Vec<f32> {
    const BLOCK_BYTES: usize = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2;
    let mut out = Vec::with_capacity(element_count);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let ql = &block[0..QK_K / 2];
        let qh = &block[QK_K / 2..QK_K / 2 + QK_K / 4];
        let sc = &block[QK_K / 2 + QK_K / 4..QK_K / 2 + QK_K / 4 + QK_K / 16];
        let d = read_f16(block, QK_K / 2 + QK_K / 4 + QK_K / 16);

        let mut values = vec![0f32; QK_K];
        let (mut ql_off, mut qh_off, mut sc_off, mut y_off) = (0usize, 0usize, 0usize, 0usize);
        while y_off < QK_K {
            for l in 0..32 {
                let is = l / 16;
                // `qh >> 0` (a no-op) is kept out of the expression below —
                // this is the `is=0` case of ggml's reference shift amount
                // `2*is`, spelled out per `is` for a fixed-size 32-lane loop.
                let q1 = ((ql[ql_off + l] & 0xF) | ((qh[qh_off + l] & 3) << 4)) as i32 - 32;
                let q2 =
                    ((ql[ql_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 =
                    ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) as i32 - 32;
                // `scales` is `int8_t` in the reference struct — reading it
                // as `u8` and casting straight to `f32` silently turns
                // every negative scale into a large positive one (e.g. 0x82
                // -> 130 instead of -126). Must go through `i8` first.
                values[y_off + l] = d * (sc[sc_off + is] as i8) as f32 * q1 as f32;
                values[y_off + l + 32] = d * (sc[sc_off + is + 2] as i8) as f32 * q2 as f32;
                values[y_off + l + 64] = d * (sc[sc_off + is + 4] as i8) as f32 * q3 as f32;
                values[y_off + l + 96] = d * (sc[sc_off + is + 6] as i8) as f32 * q4 as f32;
            }
            y_off += 128;
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
        }
        out.extend_from_slice(&values);
    }
    out.truncate(element_count);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_byte_size_matches_known_block_layouts() {
        assert_eq!(tensor_byte_size(GGML_TYPE_F32, 8).unwrap(), 32);
        assert_eq!(tensor_byte_size(GGML_TYPE_F16, 8).unwrap(), 16);
        assert_eq!(tensor_byte_size(GGML_TYPE_Q8_0, 32).unwrap(), 34);
        assert_eq!(tensor_byte_size(GGML_TYPE_Q4_0, 32).unwrap(), 18);
        assert_eq!(tensor_byte_size(GGML_TYPE_Q4_K, 256).unwrap(), 144);
        assert_eq!(tensor_byte_size(GGML_TYPE_Q5_K, 256).unwrap(), 176);
        assert_eq!(tensor_byte_size(GGML_TYPE_Q6_K, 256).unwrap(), 210);
    }

    #[test]
    fn tensor_byte_size_rejects_unsupported_types() {
        let err = tensor_byte_size(99, 8).unwrap_err();
        assert!(err.to_string().contains("not yet supported"));
    }

    #[test]
    fn dequantize_f32_round_trips() {
        let values = [1.5f32, -2.0, 0.0, 42.25];
        let mut bytes = Vec::new();
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let out = dequantize(GGML_TYPE_F32, &bytes, 4).unwrap();
        assert_eq!(out, values);
    }

    #[test]
    fn dequantize_f16_round_trips() {
        let values = [1.5f32, -2.0, 0.5];
        let mut bytes = Vec::new();
        for v in values {
            bytes.extend_from_slice(&f16::from_f32(v).to_le_bytes());
        }
        let out = dequantize(GGML_TYPE_F16, &bytes, 3).unwrap();
        for (a, b) in out.iter().zip(values.iter()) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }

    #[test]
    fn dequantize_bf16_takes_the_top_16_bits_of_an_f32() {
        let values = [1.5f32, -2.0, 0.5];
        let mut bytes = Vec::new();
        for v in values {
            // bfloat16 truncates (rather than rounds) an f32's low 16 bits
            // for these exact values without loss, so round-tripping is exact.
            let bits = (v.to_bits() >> 16) as u16;
            bytes.extend_from_slice(&bits.to_le_bytes());
        }
        let out = dequantize(GGML_TYPE_BF16, &bytes, 3).unwrap();
        assert_eq!(out, values);
    }

    /// A block of all-zero nibbles at `d=1.0` must dequantize to every
    /// element being `-8.0` (Q4_0's fixed zero-point offset).
    #[test]
    fn dequantize_q4_0_applies_the_fixed_offset() {
        let mut block = Vec::new();
        block.extend_from_slice(&f16::from_f32(1.0).to_le_bytes());
        block.extend_from_slice(&[0u8; 16]);
        let out = dequantize(GGML_TYPE_Q4_0, &block, 32).unwrap();
        assert_eq!(out.len(), 32);
        assert!(out.iter().all(|&v| v == -8.0));
    }

    /// All-zero nibbles and all-zero high bits at `d=1.0` must dequantize
    /// to every element being `-16.0` (Q5_0's fixed zero-point offset).
    #[test]
    fn dequantize_q5_0_applies_the_fixed_offset() {
        let mut block = Vec::new();
        block.extend_from_slice(&f16::from_f32(1.0).to_le_bytes());
        block.extend_from_slice(&[0u8; 4]); // qh
        block.extend_from_slice(&[0u8; 16]); // qs
        let out = dequantize(GGML_TYPE_Q5_0, &block, 32).unwrap();
        assert_eq!(out.len(), 32);
        assert!(out.iter().all(|&v| v == -16.0));
    }

    #[test]
    fn dequantize_q8_0_scales_signed_bytes() {
        let mut block = Vec::new();
        block.extend_from_slice(&f16::from_f32(2.0).to_le_bytes());
        let mut qs = [0i8; 32];
        qs[0] = 1;
        qs[1] = -1;
        block.extend_from_slice(&qs.map(|v| v as u8));
        let out = dequantize(GGML_TYPE_Q8_0, &block, 32).unwrap();
        assert_eq!(out[0], 2.0);
        assert_eq!(out[1], -2.0);
    }

    /// A `Q4_K` super-block with `d=1.0`, `dmin=0.0`, every scale byte set to
    /// encode scale `1` (`get_scale_min_k4` returns `(1, 0)` when `scales[j]
    /// == 1` for `j<4`, and correspondingly for `j>=4`), and nibble `5`
    /// everywhere, must dequantize to `5.0` everywhere (`1.0 * 1 * 5 - 0`).
    #[test]
    fn dequantize_q4_k_matches_the_reference_scale_unpacking() {
        let mut block = Vec::new();
        block.extend_from_slice(&f16::from_f32(1.0).to_le_bytes()); // d
        block.extend_from_slice(&f16::from_f32(0.0).to_le_bytes()); // dmin
        // scales[0..4] = 1 (sc for sub-blocks 0..4, j<4 path: q[j]&63).
        // scales[4..8] = 0 (min for sub-blocks 0..4, and sc/min high bits for j>=4 path).
        // scales[8..12] = 1 (sc for sub-blocks 4..8, j>=4 path: q[j+4]&0xF).
        let scales = [1u8, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1];
        block.extend_from_slice(&scales);
        // 128 bytes of qs, nibble 5 in both halves of every byte -> 0x55.
        block.extend_from_slice(&[0x55u8; 128]);

        let out = dequantize(GGML_TYPE_Q4_K, &block, 256).unwrap();
        assert_eq!(out.len(), 256);
        assert!(
            out.iter().all(|&v| (v - 5.0).abs() < 1e-5),
            "expected every element to be 5.0, got {:?}",
            &out[..8]
        );
    }

    /// Same scale layout as the Q4_K test (scale=1, min=0 for every
    /// sub-block); `qh` all-ones sets the 5th bit for every element, and
    /// `qs` all-zero nibbles means the raw 4-bit value is 0 — so every
    /// element should be `d(1.0) * scale(1) * (0 + 16) - 0 = 16.0`.
    #[test]
    fn dequantize_q5_k_applies_the_high_bit_from_qh() {
        let mut block = Vec::new();
        block.extend_from_slice(&f16::from_f32(1.0).to_le_bytes()); // d
        block.extend_from_slice(&f16::from_f32(0.0).to_le_bytes()); // dmin
        let scales = [1u8, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1];
        block.extend_from_slice(&scales);
        block.extend_from_slice(&[0xFFu8; QK_K / 8]); // qh: every high bit set
        block.extend_from_slice(&[0x00u8; QK_K / 2]); // qs: every nibble 0

        let out = dequantize(GGML_TYPE_Q5_K, &block, 256).unwrap();
        assert_eq!(out.len(), 256);
        assert!(
            out.iter().all(|&v| (v - 16.0).abs() < 1e-5),
            "expected every element to be 16.0, got {:?}",
            &out[..8]
        );
    }

    #[test]
    fn dequantize_q6_k_zero_quant_gives_the_offset_scaled_value() {
        // ql/qh all zero -> raw 6-bit quant value is 0, minus the fixed
        // 32 offset -> every element is d * scale * (-32).
        let mut block = Vec::new();
        block.extend_from_slice(&[0u8; QK_K / 2]); // ql
        block.extend_from_slice(&[0u8; QK_K / 4]); // qh
        block.extend_from_slice(&[2i8 as u8; QK_K / 16]); // scales, all 2
        block.extend_from_slice(&f16::from_f32(1.0).to_le_bytes()); // d

        let out = dequantize(GGML_TYPE_Q6_K, &block, 256).unwrap();
        assert_eq!(out.len(), 256);
        assert!(out.iter().all(|&v| v == -64.0), "got {:?}", &out[..8]);
    }

    /// `scales` is `int8_t` in ggml's own struct — a negative scale byte
    /// (`0xFE` = -2) must dequantize as -2, not as the unsigned 254 a naive
    /// `u8 as f32` cast would silently produce. Regression test for a bug
    /// that reached real model output (Qwen2.5-0.5B's `ffn_down.weight`)
    /// before being caught by cross-checking against real llama.cpp.
    #[test]
    fn dequantize_q6_k_treats_scales_as_signed() {
        let mut block = Vec::new();
        block.extend_from_slice(&[0u8; QK_K / 2]); // ql
        block.extend_from_slice(&[0u8; QK_K / 4]); // qh
        block.extend_from_slice(&[0xFEu8; QK_K / 16]); // scales, all -2
        block.extend_from_slice(&f16::from_f32(1.0).to_le_bytes()); // d

        let out = dequantize(GGML_TYPE_Q6_K, &block, 256).unwrap();
        // d(1.0) * scale(-2) * q(0-32=-32) = 64.0, not -16256.0.
        assert!(out.iter().all(|&v| v == 64.0), "got {:?}", &out[..8]);
    }

    #[test]
    fn dequantize_rejects_unsupported_types() {
        let err = dequantize(99, &[], 0).unwrap_err();
        assert!(err.to_string().contains("not yet supported"));
    }
}
