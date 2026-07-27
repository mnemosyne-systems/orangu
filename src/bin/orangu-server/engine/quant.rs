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
//! Supported: the float types (`F32`, `F16`, `BF16`), the legacy
//! round-number quants (`Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`), the whole
//! K-quant family
//! (`Q2_K` through `Q6_K`), and the `IQ*` codebook quants a mixed
//! "dynamic" release reaches for at the low end (`IQ2_XS`, `IQ2_S`,
//! `IQ3_XXS`, `IQ3_S`, `IQ4_XS`). Anything else fails with a clear "not yet
//! supported" error naming the type, rather than silently misreading the
//! bytes.
//!
//! Both families are worth reading as a pair, because the `IQ*` ones are
//! shaped quite differently: a K-quant block stores its weights, an `IQ*`
//! block stores *indices into a codebook* of lattice points that lives in
//! [`crate::engine::iq_grids`], plus a sign pattern and a scale.

use anyhow::{Result, bail};
use half::f16;

use orangu::gguf::ggml_type_name;

use crate::engine::iq_grids::{
    IQ2S_GRID, IQ2XS_GRID, IQ3S_GRID, IQ3XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS, KVALUES_IQ4NL,
};

// ggml_type ids, from ggml.h. `pub(crate)` so `engine::backend::vulkan`'s
// shader dispatch table can key off the exact same ids rather than a
// hand-copied second list that could drift from this one.
pub(crate) const GGML_TYPE_F32: u32 = 0;
pub(crate) const GGML_TYPE_F16: u32 = 1;
pub(crate) const GGML_TYPE_Q4_0: u32 = 2;
pub(crate) const GGML_TYPE_Q4_1: u32 = 3;
pub(crate) const GGML_TYPE_Q5_0: u32 = 6;
pub(crate) const GGML_TYPE_Q5_1: u32 = 7;
pub(crate) const GGML_TYPE_Q8_0: u32 = 8;
pub(crate) const GGML_TYPE_Q2_K: u32 = 10;
pub(crate) const GGML_TYPE_Q3_K: u32 = 11;
pub(crate) const GGML_TYPE_Q4_K: u32 = 12;
pub(crate) const GGML_TYPE_Q5_K: u32 = 13;
pub(crate) const GGML_TYPE_Q6_K: u32 = 14;
pub(crate) const GGML_TYPE_IQ2_XS: u32 = 17;
pub(crate) const GGML_TYPE_IQ3_XXS: u32 = 18;
pub(crate) const GGML_TYPE_IQ3_S: u32 = 21;
pub(crate) const GGML_TYPE_IQ2_S: u32 = 22;
pub(crate) const GGML_TYPE_IQ4_XS: u32 = 23;
pub(crate) const GGML_TYPE_BF16: u32 = 30;

const QK4_0: usize = 32;
const QK4_1: usize = 32;
const QK5_0: usize = 32;
const QK5_1: usize = 32;
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
        GGML_TYPE_Q4_1 => Some((2 + 2 + QK4_1 / 2, QK4_1)),
        GGML_TYPE_Q5_0 => Some((2 + 4 + QK5_0 / 2, QK5_0)),
        GGML_TYPE_Q5_1 => Some((2 + 2 + 4 + QK5_1 / 2, QK5_1)),
        GGML_TYPE_Q8_0 => Some((2 + QK8_0, QK8_0)),
        GGML_TYPE_Q2_K => Some((QK_K / 16 + QK_K / 4 + 2 + 2, QK_K)),
        GGML_TYPE_Q3_K => Some((QK_K / 8 + QK_K / 4 + 12 + 2, QK_K)),
        GGML_TYPE_Q4_K => Some((2 + 2 + K_SCALE_SIZE + QK_K / 2, QK_K)),
        GGML_TYPE_Q5_K => Some((2 + 2 + K_SCALE_SIZE + QK_K / 8 + QK_K / 2, QK_K)),
        GGML_TYPE_Q6_K => Some((QK_K / 2 + QK_K / 4 + QK_K / 16 + 2, QK_K)),
        GGML_TYPE_IQ2_XS => Some((2 + (QK_K / 8) * 2 + QK_K / 32, QK_K)),
        GGML_TYPE_IQ2_S => Some((2 + QK_K / 4 + QK_K / 32 + QK_K / 32, QK_K)),
        GGML_TYPE_IQ3_XXS => Some((2 + 3 * (QK_K / 8), QK_K)),
        GGML_TYPE_IQ3_S => Some((2 + QK_K / 4 + QK_K / 32 + QK_K / 8 + QK_K / 64, QK_K)),
        GGML_TYPE_IQ4_XS => Some((2 + 2 + QK_K / 64 + QK_K / 2, QK_K)),
        _ => None,
    }
}

/// The exact byte length a tensor with `element_count` elements of
/// `ggml_type` occupies in the GGUF file's data section.
pub fn tensor_byte_size(ggml_type: u32, element_count: u64) -> Result<u64> {
    let Some((block_bytes, block_elems)) = block_layout(ggml_type) else {
        bail!(
            "tensor type {} is not yet supported by orangu-server",
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
        GGML_TYPE_Q4_1 => Ok(dequantize_q4_1(bytes, element_count)),
        GGML_TYPE_Q5_0 => Ok(dequantize_q5_0(bytes, element_count)),
        GGML_TYPE_Q5_1 => Ok(dequantize_q5_1(bytes, element_count)),
        GGML_TYPE_Q8_0 => Ok(dequantize_q8_0(bytes, element_count)),
        GGML_TYPE_Q2_K => Ok(dequantize_q2_k(bytes, element_count)),
        GGML_TYPE_Q3_K => Ok(dequantize_q3_k(bytes, element_count)),
        GGML_TYPE_Q4_K => Ok(dequantize_q4_k(bytes, element_count)),
        GGML_TYPE_Q5_K => Ok(dequantize_q5_k(bytes, element_count)),
        GGML_TYPE_Q6_K => Ok(dequantize_q6_k(bytes, element_count)),
        GGML_TYPE_IQ2_XS => Ok(dequantize_iq2_xs(bytes, element_count)),
        GGML_TYPE_IQ2_S => Ok(dequantize_iq2_s(bytes, element_count)),
        GGML_TYPE_IQ3_XXS => Ok(dequantize_iq3_xxs(bytes, element_count)),
        GGML_TYPE_IQ3_S => Ok(dequantize_iq3_s(bytes, element_count)),
        GGML_TYPE_IQ4_XS => Ok(dequantize_iq4_xs(bytes, element_count)),
        _ => bail!(
            "tensor type {} is not yet supported by orangu-server",
            ggml_type_name(ggml_type)
        ),
    }
}

/// `f16` -> `f32`, bit-exact with `half::f16::to_f32` for every finite
/// value, zero, subnormal and infinity — the only divergence is a NaN's
/// payload bits, which `half` canonicalizes and this does not. The test
/// below checks all 65536 inputs and pins that divergence to NaN alone.
///
/// This is not premature micro-optimization: `half`'s conversion is an
/// out-of-line call, and a `perf` profile of CPU decode after
/// `engine::vecdot` landed showed **27% of total runtime** sitting in this
/// one function. Every quantized block carries an `f16` scale, so the fused
/// kernels call this once per 32 or 256 weights — often enough that a
/// function call plus a branchy software path costs as much as the NEON dot
/// product it feeds.
///
/// The algorithm is the standard branchless-ish widening: shift the
/// exponent/mantissa into `f32` position, rebias by `127 - 15`, then fix up
/// the two special exponent cases. Real model weights are overwhelmingly
/// normalized, so both branches predict essentially perfectly.
#[inline(always)]
pub(crate) fn f16_bits_to_f32(bits: u16) -> f32 {
    const SHIFTED_EXP: u32 = 0x7c00 << 13; // f16 exponent mask, in f32 position
    let mut out = ((bits as u32) & 0x7fff) << 13;
    let exp = SHIFTED_EXP & out;
    out += (127 - 15) << 23;
    if exp == SHIFTED_EXP {
        // Inf or NaN: re-bias to f32's exponent-all-ones.
        out += (128 - 16) << 23;
    } else if exp == 0 {
        // Zero or subnormal: add an implicit leading bit, then subtract the
        // matching power of two to renormalize. `113 << 23` is 2^-14 scaled
        // so the subtraction lands exactly.
        out += 1 << 23;
        out = (f32::from_bits(out) - f32::from_bits(113 << 23)).to_bits();
    }
    // Sign last, so the arithmetic above stays on a positive magnitude.
    f32::from_bits(out | (((bits as u32) & 0x8000) << 16))
}

#[inline(always)]
pub(crate) fn read_f16(bytes: &[u8], offset: usize) -> f32 {
    f16_bits_to_f32(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]))
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

/// `block_q4_1`: `{ d: f16, m: f16, qs: [u8; 16] }`, 32 elements — mirrors
/// ggml's `dequantize_row_q4_1`.
///
/// The `_1` variants differ from `Q4_0`/`Q5_0` in *how the zero point is
/// found*, not in how the quants are packed: `Q4_0` assumes a symmetric
/// range and subtracts a fixed 8, while `Q4_1` stores a per-block minimum
/// `m` and adds it, so an asymmetric weight distribution does not waste half
/// its levels. Same 16 packed bytes, same low-nibbles-then-high-nibbles
/// order.
fn dequantize_q4_1(bytes: &[u8], element_count: usize) -> Vec<f32> {
    const BLOCK_BYTES: usize = 2 + 2 + QK4_1 / 2;
    let mut out = Vec::with_capacity(element_count);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let d = read_f16(block, 0);
        let m = read_f16(block, 2);
        let qs = &block[4..];
        let mut lo = [0f32; QK4_1 / 2];
        let mut hi = [0f32; QK4_1 / 2];
        for (j, &byte) in qs.iter().enumerate() {
            lo[j] = (byte & 0x0F) as f32 * d + m;
            hi[j] = (byte >> 4) as f32 * d + m;
        }
        out.extend_from_slice(&lo);
        out.extend_from_slice(&hi);
    }
    out.truncate(element_count);
    out
}

/// `block_q5_1`: `{ d: f16, m: f16, qh: [u8; 4], qs: [u8; 16] }`, 32
/// elements — mirrors ggml's `dequantize_row_q5_1`. `Q5_0`'s fifth bit
/// packed across `qh`, with `Q4_1`'s stored minimum in place of the fixed
/// offset.
fn dequantize_q5_1(bytes: &[u8], element_count: usize) -> Vec<f32> {
    const BLOCK_BYTES: usize = 2 + 2 + 4 + QK5_1 / 2;
    let mut out = Vec::with_capacity(element_count);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let d = read_f16(block, 0);
        let m = read_f16(block, 2);
        let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        let qs = &block[8..];
        let mut lo = [0f32; QK5_1 / 2];
        let mut hi = [0f32; QK5_1 / 2];
        for (j, &byte) in qs.iter().enumerate() {
            let xh_0 = ((qh >> j) << 4) & 0x10;
            let xh_1 = (qh >> (j + 12)) & 0x10;
            lo[j] = ((byte & 0x0F) as u32 | xh_0) as f32 * d + m;
            hi[j] = ((byte >> 4) as u32 | xh_1) as f32 * d + m;
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
pub(crate) fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
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

/// `block_q2_K`: `{ scales: [u8; 16], qs: [u8; 64], d: f16, dmin: f16 }`,
/// 256 elements as 16 sub-blocks of 16 — mirrors ggml's
/// `dequantize_row_q2_K`.
///
/// Two bits per weight. Each `scales` byte carries *two* 4-bit fields: the
/// low nibble scales its sub-block's quants, the high nibble its offset —
/// both against the block's own `d`/`dmin`. The 64 `qs` bytes are read four
/// times over, two bits at a time (`shift` 0/2/4/6), so one `qs` byte feeds
/// four different sub-blocks 32 elements apart.
fn dequantize_q2_k(bytes: &[u8], element_count: usize) -> Vec<f32> {
    const BLOCK_BYTES: usize = QK_K / 16 + QK_K / 4 + 2 + 2;
    let mut out = Vec::with_capacity(element_count);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let scales = &block[0..QK_K / 16];
        let qs = &block[QK_K / 16..QK_K / 16 + QK_K / 4];
        let d = read_f16(block, QK_K / 16 + QK_K / 4);
        let dmin = read_f16(block, QK_K / 16 + QK_K / 4 + 2);

        let mut is = 0;
        let mut q_off = 0;
        while q_off < QK_K / 4 {
            let q = &qs[q_off..q_off + 32];
            for shift in [0, 2, 4, 6] {
                for half in 0..2 {
                    let sc = scales[is];
                    is += 1;
                    let dl = d * (sc & 0xF) as f32;
                    let ml = dmin * (sc >> 4) as f32;
                    for l in 0..16 {
                        out.push(dl * ((q[half * 16 + l] >> shift) & 3) as f32 - ml);
                    }
                }
            }
            q_off += 32;
        }
    }
    out.truncate(element_count);
    out
}

/// `block_q3_K`: `{ hmask: [u8; 32], qs: [u8; 64], scales: [u8; 12],
/// d: f16 }`, 256 elements as 16 sub-blocks of 16 — mirrors ggml's
/// `dequantize_row_q3_K`.
///
/// Three bits per weight, split across two arrays: `qs` holds the low two
/// bits (same four-pass `shift` walk as `Q2_K`) and `hmask` the third,
/// one bit per weight, with the mask rotating left once per sub-block pair.
/// That third bit is *inverted* — set means "don't subtract 4" — which is
/// why the reference subtracts 4 when it is **clear**.
///
/// The 16 sub-block scales are 6-bit and packed into 12 bytes by
/// [`unpack_q3_k_scales`]. They are biased by 32, i.e. signed.
fn dequantize_q3_k(bytes: &[u8], element_count: usize) -> Vec<f32> {
    const BLOCK_BYTES: usize = QK_K / 8 + QK_K / 4 + 12 + 2;
    let mut out = Vec::with_capacity(element_count);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let hmask = &block[0..QK_K / 8];
        let qs = &block[QK_K / 8..QK_K / 8 + QK_K / 4];
        let scales = unpack_q3_k_scales(&block[QK_K / 8 + QK_K / 4..QK_K / 8 + QK_K / 4 + 12]);
        let d_all = read_f16(block, QK_K / 8 + QK_K / 4 + 12);

        let mut is = 0;
        let mut m = 1u8;
        let mut q_off = 0;
        while q_off < QK_K / 4 {
            for shift in [0, 2, 4, 6] {
                for half in 0..2 {
                    let dl = d_all * (scales[is] - 32) as f32;
                    is += 1;
                    for l in 0..16 {
                        let idx = half * 16 + l;
                        let hi = if hmask[idx] & m != 0 { 0 } else { 4 };
                        out.push(dl * (((qs[q_off + idx] >> shift) & 3) as i32 - hi) as f32);
                    }
                }
                m <<= 1;
            }
            q_off += 32;
        }
    }
    out.truncate(element_count);
    out
}

/// `Q3_K`'s 16 six-bit sub-block scales, unpacked from the 12 bytes they are
/// stored in and returned in sub-block order, still biased by 32.
///
/// The packing splits every scale in two: the low 4 bits of scales 0..8 live
/// in the low nibbles of bytes 0..8, of scales 8..16 in the high nibbles of
/// those same bytes, and all 16 pairs of high bits are stacked two-per-byte
/// into bytes 8..12. This is ggml's `kmask1`/`kmask2` shuffle written out one
/// scale at a time rather than four-at-a-time over a `uint32_t`, which is the
/// same permutation without the endianness assumption a `memcpy` into a
/// `uint32_t[4]` carries.
fn unpack_q3_k_scales(packed: &[u8]) -> [i32; 16] {
    let mut scales = [0i32; 16];
    for (i, scale) in scales.iter_mut().enumerate() {
        let low = if i < 8 {
            packed[i] & 0xF
        } else {
            packed[i - 8] >> 4
        };
        // Scale `i`'s two high bits sit at bit `2 * (i / 4)` of byte
        // `8 + i % 4`.
        let high = (packed[8 + (i % 4)] >> (2 * (i / 4))) & 3;
        *scale = (low | (high << 4)) as i32;
    }
    scales
}

/// One `IQ2_XS`/`IQ2_S`/`IQ3_XXS`/`IQ3_S` lattice point: `width` bytes of
/// `grid`, each scaled by `dl` and negated where `signs` has a bit set.
///
/// Every `IQ*` dequantizer below is this same three-line inner loop over a
/// different codebook, so it lives here once. `signs` is the 8-bit pattern
/// [`KSIGNS_IQ2XS`] expands a 7-bit field into (or, for `IQ2_S`/`IQ3_S`, the
/// byte stored directly in the block) — bit `j` negates element `j`.
#[inline]
fn push_iq_grid(out: &mut Vec<f32>, grid: u64, width: usize, dl: f32, signs: u8) {
    for (j, &g) in grid.to_le_bytes()[..width].iter().enumerate() {
        let sign = if signs & KMASK_IQ2XS[j] != 0 {
            -1.0
        } else {
            1.0
        };
        out.push(dl * g as f32 * sign);
    }
}

/// `block_iq2_xs`: `{ d: f16, qs: [u16; 32], scales: [u8; 8] }`, 256
/// elements as 8 groups of 32 — mirrors ggml's `dequantize_row_iq2_xs`.
///
/// Each `qs` entry is one codebook lookup covering 8 weights: its low 9 bits
/// index [`IQ2XS_GRID`] and its top 7 bits index [`KSIGNS_IQ2XS`] for the
/// sign pattern. Each `scales` byte holds two 4-bit scales, one per 16-weight
/// half of its 32-weight group.
fn dequantize_iq2_xs(bytes: &[u8], element_count: usize) -> Vec<f32> {
    const BLOCK_BYTES: usize = 2 + (QK_K / 8) * 2 + QK_K / 32;
    let mut out = Vec::with_capacity(element_count);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let d = read_f16(block, 0);
        let qs = &block[2..2 + (QK_K / 8) * 2];
        let scales = &block[2 + (QK_K / 8) * 2..];

        for ib32 in 0..QK_K / 32 {
            let db = [
                d * (0.5 + (scales[ib32] & 0xF) as f32) * 0.25,
                d * (0.5 + (scales[ib32] >> 4) as f32) * 0.25,
            ];
            for l in 0..4 {
                let q = u16::from_le_bytes([qs[2 * (4 * ib32 + l)], qs[2 * (4 * ib32 + l) + 1]]);
                let grid = IQ2XS_GRID[(q & 511) as usize];
                let signs = KSIGNS_IQ2XS[(q >> 9) as usize];
                push_iq_grid(&mut out, grid, 8, db[l / 2], signs);
            }
        }
    }
    out.truncate(element_count);
    out
}

/// `block_iq2_s`: `{ d: f16, qs: [u8; 64], qh: [u8; 8], scales: [u8; 8] }`,
/// 256 elements as 8 groups of 32 — mirrors ggml's `dequantize_row_iq2_s`.
///
/// The scale layout is `IQ2_XS`'s. The codebook index is wider (10 bits into
/// the 1024-entry [`IQ2S_GRID`]): 8 bits from `qs`, the top 2 from a
/// per-group `qh` byte. The signs are no longer a 7-bit index into
/// [`KSIGNS_IQ2XS`] but a full byte, stored in the *second half* of `qs` —
/// ggml spells that as `signs = qs + QK_K/8`, an alias into the same array.
fn dequantize_iq2_s(bytes: &[u8], element_count: usize) -> Vec<f32> {
    const BLOCK_BYTES: usize = 2 + QK_K / 4 + QK_K / 32 + QK_K / 32;
    let mut out = Vec::with_capacity(element_count);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let d = read_f16(block, 0);
        let qs = &block[2..2 + QK_K / 4];
        let qh = &block[2 + QK_K / 4..2 + QK_K / 4 + QK_K / 32];
        let scales = &block[2 + QK_K / 4 + QK_K / 32..];
        let signs = &qs[QK_K / 8..];

        for ib32 in 0..QK_K / 32 {
            let db = [
                d * (0.5 + (scales[ib32] & 0xF) as f32) * 0.25,
                d * (0.5 + (scales[ib32] >> 4) as f32) * 0.25,
            ];
            for l in 0..4 {
                let idx =
                    qs[4 * ib32 + l] as usize | (((qh[ib32] as usize) << (8 - 2 * l)) & 0x300);
                push_iq_grid(&mut out, IQ2S_GRID[idx], 8, db[l / 2], signs[4 * ib32 + l]);
            }
        }
    }
    out.truncate(element_count);
    out
}

/// `block_iq3_xxs`: `{ d: f16, qs: [u8; 96] }`, 256 elements as 8 groups of
/// 32 — mirrors ggml's `dequantize_row_iq3_xxs`.
///
/// The single `qs` array is two arrays end to end: 64 bytes of codebook
/// indices into the 256-entry [`IQ3XXS_GRID`] (4 weights each, so two
/// lookups per 8-weight run), then 32 bytes read as eight little-endian
/// `u32`s, one per group, each packing four 7-bit sign indices plus a 4-bit
/// scale in its top nibble.
fn dequantize_iq3_xxs(bytes: &[u8], element_count: usize) -> Vec<f32> {
    const BLOCK_BYTES: usize = 2 + 3 * (QK_K / 8);
    let mut out = Vec::with_capacity(element_count);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let d = read_f16(block, 0);
        let qs = &block[2..];
        let scales_and_signs = &qs[QK_K / 4..];

        for ib32 in 0..QK_K / 32 {
            let aux32 = u32::from_le_bytes(
                scales_and_signs[4 * ib32..4 * ib32 + 4]
                    .try_into()
                    .expect("4 bytes"),
            );
            let db = d * (0.5 + (aux32 >> 28) as f32) * 0.5;
            for l in 0..4 {
                let signs = KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
                let grid1 = IQ3XXS_GRID[qs[8 * ib32 + 2 * l] as usize];
                let grid2 = IQ3XXS_GRID[qs[8 * ib32 + 2 * l + 1] as usize];
                // The two 4-element lookups are the two halves of one
                // 8-element sign pattern, so they concatenate into a single
                // `u64` rather than taking `push_iq_grid` twice with `signs`
                // restarted at bit 0.
                push_iq_grid(
                    &mut out,
                    grid1 as u64 | ((grid2 as u64) << 32),
                    8,
                    db,
                    signs,
                );
            }
        }
    }
    out.truncate(element_count);
    out
}

/// `block_iq3_s`: `{ d: f16, qs: [u8; 64], qh: [u8; 8], signs: [u8; 32],
/// scales: [u8; 4] }`, 256 elements as 8 groups of 32 — mirrors ggml's
/// `dequantize_row_iq3_s`.
///
/// A 9-bit index into the 512-entry [`IQ3S_GRID`]: 8 bits from `qs`, the
/// ninth from `qh`, one bit per lookup. Signs are stored outright, one byte
/// per 8 weights. The 4 `scales` bytes hold two 4-bit scales each, one per
/// 32-weight group, and unlike `IQ2_*` the scale is `1 + 2*s` rather than
/// `(0.5 + s) * 0.25`.
fn dequantize_iq3_s(bytes: &[u8], element_count: usize) -> Vec<f32> {
    const BLOCK_BYTES: usize = 2 + QK_K / 4 + QK_K / 32 + QK_K / 8 + QK_K / 64;
    let mut out = Vec::with_capacity(element_count);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let d = read_f16(block, 0);
        let qs = &block[2..2 + QK_K / 4];
        let qh = &block[2 + QK_K / 4..2 + QK_K / 4 + QK_K / 32];
        let signs = &block[2 + QK_K / 4 + QK_K / 32..2 + QK_K / 4 + QK_K / 32 + QK_K / 8];
        let scales = &block[2 + QK_K / 4 + QK_K / 32 + QK_K / 8..];

        for ib32 in 0..QK_K / 32 {
            let scale = (scales[ib32 / 2] >> (4 * (ib32 % 2))) & 0xF;
            let db = d * (1 + 2 * scale as i32) as f32;
            for l in 0..4 {
                let lo = &qs[8 * ib32 + 2 * l..];
                let hb = qh[ib32] as usize;
                let idx1 = lo[0] as usize | ((hb << (8 - 2 * l)) & 256);
                let idx2 = lo[1] as usize | ((hb << (7 - 2 * l)) & 256);
                push_iq_grid(
                    &mut out,
                    IQ3S_GRID[idx1] as u64 | ((IQ3S_GRID[idx2] as u64) << 32),
                    8,
                    db,
                    signs[4 * ib32 + l],
                );
            }
        }
    }
    out.truncate(element_count);
    out
}

/// `block_iq4_xs`: `{ d: f16, scales_h: u16, scales_l: [u8; 4],
/// qs: [u8; 128] }`, 256 elements as 8 groups of 32 — mirrors ggml's
/// `dequantize_row_iq4_xs`.
///
/// The only new type here with no codebook of lattice points: a nibble picks
/// one of the 16 non-uniformly spaced levels in [`KVALUES_IQ4NL`]. Each
/// group's 6-bit scale is split, 4 low bits in `scales_l` (two groups per
/// byte) and 2 high bits in `scales_h` (eight groups per `u16`), and is
/// biased by 32. Within a group the low nibbles of the 16 `qs` bytes are the
/// first 16 weights and the high nibbles the next 16 — the same split-halves
/// order as `Q4_K`, not interleaved.
fn dequantize_iq4_xs(bytes: &[u8], element_count: usize) -> Vec<f32> {
    const BLOCK_BYTES: usize = 2 + 2 + QK_K / 64 + QK_K / 2;
    let mut out = Vec::with_capacity(element_count);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let d = read_f16(block, 0);
        let scales_h = u16::from_le_bytes([block[2], block[3]]);
        let scales_l = &block[4..4 + QK_K / 64];
        let qs = &block[4 + QK_K / 64..];

        let mut values = [0f32; QK_K];
        for ib in 0..QK_K / 32 {
            let low = (scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF;
            let high = ((scales_h >> (2 * ib)) & 3) as u8;
            let ls = (low | (high << 4)) as i32;
            let dl = d * (ls - 32) as f32;
            for j in 0..16 {
                let byte = qs[16 * ib + j];
                values[32 * ib + j] = dl * KVALUES_IQ4NL[(byte & 0xF) as usize] as f32;
                values[32 * ib + j + 16] = dl * KVALUES_IQ4NL[(byte >> 4) as usize] as f32;
            }
        }
        out.extend_from_slice(&values);
    }
    out.truncate(element_count);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Random quantized blocks and the `f32`s **ggml itself** produced from
    /// them, generated by `testdata/ggml-dequant-reference.c` against
    /// `libggml-base`'s `ggml_get_type_traits(t)->to_float` — the same entry
    /// point `llama.cpp` reads a weight tensor through.
    ///
    /// Hand-built blocks can only check the arm of a dequantizer the author
    /// already understood, which for a codebook quant is close to nothing:
    /// an `IQ3_S` element is a 9-bit index into a 512-entry lattice, a sign
    /// byte and a shared 4-bit scale, and a wrong shift produces numbers
    /// that look entirely reasonable. Random bytes across 8 blocks reach
    /// every grid entry, sign pattern and scale combination the format can
    /// express, and the reference values are not derived from anything in
    /// this file.
    ///
    /// Layout: `"ORQFIX02"`, `u32` type count, then per type a `u32`
    /// `ggml_type`, `u32` block count, `u32` block bytes, `u32` element
    /// count, the raw bytes, and the reference `f32`s. Regenerate with:
    ///
    /// ```text
    /// cc -O2 -o genfix testdata/ggml-dequant-reference.c \
    ///     -I/usr/local/include -L/usr/local/lib64 -lggml-base
    /// ./genfix testdata/ggml-dequant-reference.bin
    /// ```
    ///
    /// Read at run time rather than through `include_bytes!`, for the same
    /// reason as `arch::read_reference_fixture`: the fixture is ground truth
    /// captured from a machine with `libggml-base` installed, so a checkout
    /// may legitimately not have it, and a compile-time include turns that
    /// into a build failure for the *whole* test binary rather than a skip of
    /// the two tests that need it. Returns `None` (with a note on stderr)
    /// when it isn't there.
    fn read_ggml_reference() -> Option<Vec<u8>> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/bin/orangu-server/engine/testdata/ggml-dequant-reference.bin");
        match std::fs::read(&path) {
            Ok(bytes) => Some(bytes),
            Err(err) => {
                eprintln!("skipping: no reference fixture {} ({err})", path.display());
                None
            }
        }
    }

    /// One type's fixture entry: `(ggml_type, elements per block, raw bytes,
    /// expected f32s)`.
    type ReferenceCase = (u32, usize, Vec<u8>, Vec<f32>);

    /// Splits the fixture read by [`read_ggml_reference`] into one
    /// [`ReferenceCase`] per type, or `None` when the fixture isn't present
    /// in this checkout.
    fn ggml_reference_cases() -> Option<Vec<ReferenceCase>> {
        fn take<'a>(fixture: &'a [u8], at: &mut usize, n: usize) -> &'a [u8] {
            let s = &fixture[*at..*at + n];
            *at += n;
            s
        }
        fn u32_at(fixture: &[u8], at: &mut usize) -> u32 {
            u32::from_le_bytes(take(fixture, at, 4).try_into().expect("4 bytes"))
        }

        let fixture = read_ggml_reference()?;
        let mut at = 0;
        assert_eq!(take(&fixture, &mut at, 8), b"ORQFIX02", "fixture magic");
        let n_types = u32_at(&fixture, &mut at);

        let mut cases = Vec::new();
        for _ in 0..n_types {
            let ggml_type = u32_at(&fixture, &mut at);
            let n_blocks = u32_at(&fixture, &mut at) as usize;
            let block_bytes = u32_at(&fixture, &mut at) as usize;
            // Element count comes from the fixture rather than being derived
            // from `block_layout`: parsing the reference with the table under
            // test would turn a wrong block size into a misaligned read
            // somewhere downstream instead of a clean failure here.
            let n_elems = u32_at(&fixture, &mut at) as usize;
            let raw = take(&fixture, &mut at, n_blocks * block_bytes).to_vec();
            let floats = take(&fixture, &mut at, n_elems * 4)
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().expect("4 bytes")))
                .collect();
            cases.push((ggml_type, n_elems / n_blocks, raw, floats));
        }
        assert_eq!(at, fixture.len(), "trailing bytes in fixture");
        Some(cases)
    }

    /// Every K-quant and `IQ*` type, bit-for-bit against ggml's own output.
    ///
    /// Bit-for-bit, not approximately: both sides evaluate the same
    /// expression in the same order on the same `f32` hardware, so any
    /// tolerance here would be hiding a real divergence rather than
    /// absorbing rounding. The `Q4_0`/`Q5_0`/`Q8_0`/`Q4_K`/`Q5_K`/`Q6_K`
    /// rows are the control — they were correct before this fixture existed,
    /// so their passing is what says the harness is wired up rather than
    /// vacuously agreeing.
    #[test]
    fn dequantize_matches_ggml_for_every_quantized_type() {
        let Some(cases) = ggml_reference_cases() else {
            return;
        };
        assert_eq!(cases.len(), 15, "fixture should cover 15 types");

        for (ggml_type, block_elems, raw, want) in cases {
            let name = ggml_type_name(ggml_type);
            assert_eq!(
                tensor_byte_size(ggml_type, want.len() as u64).expect("supported type"),
                raw.len() as u64,
                "{name}: block layout disagrees with ggml's type_size",
            );

            let got = dequantize(ggml_type, &raw, want.len()).expect("supported type");
            assert_eq!(got.len(), want.len(), "{name}: element count");
            for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "{name}: element {i} (block {}, offset {}): got {g}, ggml says {w}",
                    i / block_elems,
                    i % block_elems,
                );
            }
        }
    }

    /// Guards the guard: a dequantizer that returned all zeros, or that
    /// collapsed every element of a block onto one value, would sail through
    /// a comparison whose reference was itself degenerate. Assert the
    /// fixture's own reference values are varied and non-trivial before
    /// trusting a match against them.
    #[test]
    fn ggml_reference_values_are_non_degenerate() {
        let Some(cases) = ggml_reference_cases() else {
            return;
        };
        for (ggml_type, _, _, want) in cases {
            let name = ggml_type_name(ggml_type);
            assert!(
                want.iter().all(|v| v.is_finite()),
                "{name}: reference contains a non-finite value"
            );
            assert!(
                want.iter().any(|&v| v > 0.0) && want.iter().any(|&v| v < 0.0),
                "{name}: reference is single-signed"
            );
            // 64 is a floor, not a ratio, because each type's ceiling is its
            // own: `Q4_1` can express 16 levels per 32-element block (128
            // across the fixture's 8), `Q2_K` only 4 per 16-element
            // sub-block (512), a `Q6_K` thousands. Any proportion of the
            // element count that the widest type passes, the narrowest fails.
            // What this needs to catch is a *degenerate* reference — all
            // zeros (1 distinct), one value per block (8), a stuck scale —
            // and 64 is an order of magnitude above all of those while
            // sitting below the tightest real type's 95.
            let distinct: std::collections::HashSet<u32> =
                want.iter().map(|v| v.to_bits()).collect();
            assert!(
                distinct.len() >= 64,
                "{name}: only {} distinct values across {} elements",
                distinct.len(),
                want.len()
            );
        }
    }

    /// The hand-rolled widening must agree with `half` across the *entire*
    /// input domain, not just a few samples — one wrong exponent case would
    /// silently corrupt a quantized block's scale.
    ///
    /// Bit-exact for every finite value, zero, subnormal and infinity. The
    /// one documented divergence is the *payload* of a NaN: `half`
    /// canonicalizes to a quiet NaN while this returns the shifted input
    /// payload. Both are NaN, and a NaN scale in a GGUF weight is already a
    /// corrupt file, so the distinction is immaterial — but assert it stays
    /// confined to NaN rather than silently widening.
    #[test]
    fn f16_conversion_agrees_with_half_for_all_65536_inputs() {
        let mut nan_payload_diffs = 0;
        for bits in 0..=u16::MAX {
            let want = f16::from_bits(bits).to_f32();
            let got = f16_bits_to_f32(bits);
            if got.to_bits() == want.to_bits() {
                continue;
            }
            assert!(
                got.is_nan() && want.is_nan(),
                "f16 0x{bits:04x}: got {got} ({:08x}), want {want} ({:08x})",
                got.to_bits(),
                want.to_bits()
            );
            nan_payload_diffs += 1;
        }
        // Every f16 NaN encoding, both signs, minus the ones that happen to
        // land on half's canonical payload anyway.
        assert_eq!(nan_payload_diffs, 1022);
    }

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
