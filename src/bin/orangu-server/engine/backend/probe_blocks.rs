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

//! The weight fixture every backend cross-check builds its matrix from —
//! one definition, shared, because a cross-check is only as good as the
//! bytes it feeds both sides.
//!
//! Not test-only. `VulkanBackend::decode_kernel_agrees` runs a cross-check
//! at **startup** to decide whether this device computes its tuned kernels
//! correctly, and it needs valid blocks for exactly the reason the tests do:
//! it once built its weights from arbitrary bytes and reported that `F16`,
//! `Q4_0`, `Q8_0` and five other types were all broken on a working GPU,
//! because a random `f16` scale is `NaN` about as often as not.
//!
//! This lived in `vulkan::tests` alone, and the difference mattered.
//! `vulkan.rs` is the backend that has always had real hardware under it
//! here, so it is the one whose fixture got corrected as problems showed
//! up: [`build_block`] lays out each `ggml_type` field by field and routes
//! every field that is *read back as a float* through
//! [`next_bounded_f32`]. `opencl.rs`, `cuda.rs` and `rocm.rs` — none of
//! which had a device on this machine when they were written — instead
//! filled a whole weight row with uniformly random bytes, which is not the
//! same test at all: a random `f16` scale is `NaN` or `Inf` about as often
//! as not, and a reference that is `NaN` compares equal to *no* kernel
//! output, including the `NaN` the device correctly produced. Those
//! cross-checks asserted nothing they could pass, right up until a machine
//! with an OpenCL device ran them.
//!
//! `opencl.rs` is converted. `cuda.rs` and `rocm.rs` are not: there is no
//! NVIDIA or AMD device here to run them against, and switching a fixture
//! blind would only move an unverified claim rather than retire one. They
//! will fail the same way `opencl.rs` did on the first machine that has the
//! hardware — `rocm.rs` additionally still references `CpuBackend::matmul`
//! instead of `matmul_dequant`, the second defect from that same run.
//!
//! Only float-valued fields are bounded. Quant nibbles, high-bit packs and
//! K-quant scale bytes stay fully random: they are read back as integers,
//! never reinterpreted as floats, so arbitrary bits there are signal rather
//! than poison.

use crate::engine::quant::{
    GGML_TYPE_BF16, GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_IQ1_M, GGML_TYPE_IQ1_S,
    GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ3_XXS,
    GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS, GGML_TYPE_MXFP4, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K,
    GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q4_K, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K, GGML_TYPE_Q8_0,
};

pub(crate) fn next_byte(seed: &mut u64) -> u8 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*seed >> 33) as u8
}

pub(crate) fn next_bytes(seed: &mut u64, n: usize) -> Vec<u8> {
    (0..n).map(|_| next_byte(seed)).collect()
}

/// A small positive value, bounded well away from zero, infinity, and
/// subnormals — safe to use for every type's `d`/`dmin` scale field
/// (and the whole value, for `F32`/`F16`/`BF16`) without risking a NaN
/// or Inf poisoning the dot product on either backend.
pub(crate) fn next_bounded_f32(seed: &mut u64) -> f32 {
    0.05 + (next_byte(seed) as f32 / 255.0) * 1.95
}

pub(crate) fn f16_bytes(v: f32) -> [u8; 2] {
    half::f16::from_f32(v).to_le_bytes()
}

/// Builds one block's raw bytes for `ggml_type`, matching the exact
/// layout `engine::quant::dequantize` reads. Scale/whole-value float
/// fields are bounded (see `next_bounded_f32`); every other field
/// (quant nibbles, high-bit packs, K-quant scale bytes) is safe with
/// arbitrary bits since it's read back as a plain integer, never
/// reinterpreted as a float.
pub(crate) fn build_block(ggml_type: u32, seed: &mut u64) -> Vec<u8> {
    let mut out = Vec::new();
    match ggml_type {
        t if t == GGML_TYPE_F32 => {
            out.extend_from_slice(&next_bounded_f32(seed).to_le_bytes());
        }
        t if t == GGML_TYPE_F16 => {
            out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
        }
        t if t == GGML_TYPE_BF16 => {
            let bits = (next_bounded_f32(seed).to_bits() >> 16) as u16;
            out.extend_from_slice(&bits.to_le_bytes());
        }
        t if t == GGML_TYPE_Q4_0 => {
            out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
            out.extend(next_bytes(seed, 16));
        }
        // `Q4_0`'s 16 `qs` bytes behind a one-byte `e8m0` exponent instead of
        // an `f16` scale. The exponent is **bounded** where the nibbles are
        // not: it is the only field here read as a float, and an unbounded
        // byte spans 2^-127..2^127, which overflows a dot product to `inf` on
        // both paths and would compare equal while testing nothing. `128`
        // decodes to exactly 1.0 (`(128-1) << 23` is f32 exponent 127), so
        // this is a symmetric 2^-4..2^4 around unity.
        t if t == GGML_TYPE_MXFP4 => {
            out.push(124 + (next_byte(seed) % 9));
            out.extend(next_bytes(seed, 16));
        }
        t if t == GGML_TYPE_Q4_1 => {
            out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
            out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
            out.extend(next_bytes(seed, 16));
        }
        t if t == GGML_TYPE_Q5_0 => {
            out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
            out.extend(next_bytes(seed, 4));
            out.extend(next_bytes(seed, 16));
        }
        t if t == GGML_TYPE_Q5_1 => {
            out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
            out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
            out.extend(next_bytes(seed, 4));
            out.extend(next_bytes(seed, 16));
        }
        t if t == GGML_TYPE_Q8_0 => {
            out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
            out.extend(next_bytes(seed, 32));
        }
        t if t == GGML_TYPE_Q4_K => {
            out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
            out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
            out.extend(next_bytes(seed, 12));
            out.extend(next_bytes(seed, 128));
        }
        t if t == GGML_TYPE_Q5_K => {
            out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
            out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
            out.extend(next_bytes(seed, 12));
            out.extend(next_bytes(seed, 32));
            out.extend(next_bytes(seed, 128));
        }
        t if t == GGML_TYPE_Q6_K => {
            out.extend(next_bytes(seed, 128));
            out.extend(next_bytes(seed, 64));
            out.extend(next_bytes(seed, 16));
            out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
        }
        t if t == GGML_TYPE_Q2_K => {
            out.extend(next_bytes(seed, 16));
            out.extend(next_bytes(seed, 64));
            out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
            out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
        }
        t if t == GGML_TYPE_Q3_K => {
            out.extend(next_bytes(seed, 32));
            out.extend(next_bytes(seed, 64));
            out.extend(next_bytes(seed, 12));
            out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
        }
        // Every `IQ*` field below is a codebook index, a sign pattern or
        // a packed scale, all of which are valid for any bit pattern —
        // no field needs constraining to keep the block well formed, so
        // random bytes reach the whole encoding space.
        t if t == GGML_TYPE_IQ2_XS => {
            out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
            out.extend(next_bytes(seed, 64));
            out.extend(next_bytes(seed, 8));
        }
        t if t == GGML_TYPE_IQ2_S => {
            out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
            out.extend(next_bytes(seed, 64));
            out.extend(next_bytes(seed, 8));
            out.extend(next_bytes(seed, 8));
        }
        t if t == GGML_TYPE_IQ3_XXS => {
            out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
            out.extend(next_bytes(seed, 96));
        }
        t if t == GGML_TYPE_IQ3_S => {
            out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
            out.extend(next_bytes(seed, 64));
            out.extend(next_bytes(seed, 8));
            out.extend(next_bytes(seed, 32));
            out.extend(next_bytes(seed, 4));
        }
        t if t == GGML_TYPE_IQ4_XS => {
            out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
            out.extend(next_bytes(seed, 2));
            out.extend(next_bytes(seed, 4));
            out.extend(next_bytes(seed, 128));
        }
        t if t == GGML_TYPE_IQ4_NL => {
            out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
            out.extend(next_bytes(seed, 16));
        }
        t if t == GGML_TYPE_IQ2_XXS => {
            out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
            out.extend(next_bytes(seed, 64));
        }
        t if t == GGML_TYPE_IQ1_S => {
            out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
            out.extend(next_bytes(seed, 32));
            out.extend(next_bytes(seed, 16));
        }
        // The one type with no `d` field at all: its `f16` block scale is
        // four nibbles scattered across the top of the four `scales`
        // `u16`s, so random bytes there *are* a random `f16` — exponent
        // included, where every other arm draws its scale through
        // `next_bounded_f32`. The top nibble of the last `u16` (byte 7's
        // high half) is the `f16`'s own top nibble, so pinning it to
        // `0x3` keeps the block scale a positive normal of order 1
        // instead of an occasional `inf`/`NaN`, which would make the
        // comparison below vacuous rather than strict. Every other bit,
        // including the rest of the exponent, stays random.
        t if t == GGML_TYPE_IQ1_M => {
            out.extend(next_bytes(seed, 32));
            out.extend(next_bytes(seed, 16));
            let mut scales = next_bytes(seed, 8);
            scales[7] = (scales[7] & 0x0F) | 0x30;
            out.extend(scales);
        }
        other => panic!("build_block: unhandled ggml_type {other}"),
    }
    out
}
