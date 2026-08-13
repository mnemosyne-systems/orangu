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
//! "dynamic" release reaches for at the low end (`IQ1_S`, `IQ1_M`,
//! `IQ1_XS`, `IQ1_XXS`, `IQ1_XXXS`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`,
//! `IQ3_XXS`, `IQ3_S`, `IQ4_NL`, `IQ4_XS`), plus `MXFP4`. Anything else fails with a
//! clear "not yet supported" error naming the type, rather than silently
//! misreading the bytes.
//!
//! Both families are worth reading as a pair, because the `IQ*` ones are
//! shaped quite differently: a K-quant block stores its weights, an `IQ*`
//! block stores *indices into a codebook* of lattice points that lives in
//! [`crate::engine::iq_grids`], plus a sign pattern and a scale.

use anyhow::{Result, bail};
use half::f16;

use orangu::gguf::{ggml_type_name, is_removed_ggml_type};

use crate::engine::iq_grids::{
    IQ1S_GRID, IQ1XS_GRID, IQ1XXS_GRID, IQ1XXXS_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID,
    IQ3S_GRID, IQ3XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS, KVALUES_IQ4NL,
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
pub(crate) const GGML_TYPE_IQ2_XXS: u32 = 16;
pub(crate) const GGML_TYPE_IQ2_XS: u32 = 17;
pub(crate) const GGML_TYPE_IQ3_XXS: u32 = 18;
pub(crate) const GGML_TYPE_IQ1_S: u32 = 19;
/// `IQ4_NL` is the one `IQ*` type that blocks at 32 rather than 256, which
/// is why it turns up in files whose *name* promises a pure K-quant: a
/// K-quant block needs 256 | `ne[0]`, so upstream's quantizer substitutes
/// `IQ4_NL` row by row when a tensor's row length isn't a multiple of 256
/// (`llama.cpp`'s `llama_tensor_get_type`). Qwen2.5-0.5B's 896-wide rows
/// are the common case — a `Q2_K` download of one is mostly `IQ4_NL`.
pub(crate) const GGML_TYPE_IQ4_NL: u32 = 20;
pub(crate) const GGML_TYPE_IQ3_S: u32 = 21;
pub(crate) const GGML_TYPE_IQ2_S: u32 = 22;
pub(crate) const GGML_TYPE_IQ4_XS: u32 = 23;
pub(crate) const GGML_TYPE_I32: u32 = 26;
/// `IQ1_M` is the one block with no `f16` scale field of its own — its
/// scale is reassembled from nibbles spread across `scales`.
pub(crate) const GGML_TYPE_IQ1_M: u32 = 29;
pub(crate) const GGML_TYPE_BF16: u32 = 30;
// The three ARM-SIMD *repacked* `Q4_0` layouts. ggml has retired these ids
// and never shipped a `to_float` for them (only `gemv`/`gemm` kernels), so
// upstream `llama.cpp` refuses a file carrying one outright. They are not
// lossy or damaged, though: the packing is a pure permutation of `Q4_0`'s
// own bytes plus a per-nibble `^ 8`, so [`deinterleave_repack`] turns
// a tensor back into exactly the `Q4_0` it was built from, and
// `engine::loader` does that once at load. Nothing downstream of the
// loader ever sees these ids.
//
// The trailing digits are (rows interleaved) x (bytes interleaved), except
// that `Q4_0_4_8` interleaves 4 rows in 8-byte runs — so the row count is
// **not** always the first digit's pair. See [`repack_layout`].
pub(crate) const GGML_TYPE_Q4_0_4_4: u32 = 31;
pub(crate) const GGML_TYPE_Q4_0_4_8: u32 = 32;
pub(crate) const GGML_TYPE_Q4_0_8_8: u32 = 33;
// The same three interleavings applied to `IQ4_NL` instead of `Q4_0`. Same
// 18-byte block, same record shape, same row grouping — and *no* `^ 8`,
// because an `IQ4_NL` nibble is an index into a codebook rather than a
// signed integer, so there is no sign convention to fold away
// (`make_block_iq4_nlx4`/`x8` copy the runs verbatim).
pub(crate) const GGML_TYPE_IQ4_NL_4_4: u32 = 36;
pub(crate) const GGML_TYPE_IQ4_NL_4_8: u32 = 37;
pub(crate) const GGML_TYPE_IQ4_NL_8_8: u32 = 38;
pub(crate) const GGML_TYPE_MXFP4: u32 = 39;
// `IQ1_S` with a narrower codebook index, and the only ids here that ggml
// itself does not define: 42..63 are left free for ggml to grow into, so a
// quantizer shipping types of its own starts at 64. A stock build rejects a
// file carrying one instead of misreading it, which is exactly what the
// reservation is for — and what makes these ids safe to read here.
pub(crate) const GGML_TYPE_IQ1_XS: u32 = 64;
pub(crate) const GGML_TYPE_IQ1_XXS: u32 = 65;
pub(crate) const GGML_TYPE_IQ1_XXXS: u32 = 66;

const QK4_0: usize = 32;
const QK4_1: usize = 32;
const QK5_0: usize = 32;
const QK5_1: usize = 32;
const QK8_0: usize = 32;
const QK4_NL: usize = 32;
const QK_MXFP4: usize = 32;
const QK_K: usize = 256;
const K_SCALE_SIZE: usize = 12;
const KVALUES_MXFP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// Bytes per block, and elements per block, for a supported `ggml_type`.
/// `None` for a type this engine can't yet read.
fn block_layout(ggml_type: u32) -> Option<(usize, usize)> {
    match ggml_type {
        GGML_TYPE_F32 => Some((4, 1)),
        GGML_TYPE_F16 => Some((2, 1)),
        GGML_TYPE_I32 => Some((4, 1)),
        GGML_TYPE_BF16 => Some((2, 1)),
        GGML_TYPE_Q4_0 => Some((2 + QK4_0 / 2, QK4_0)),
        GGML_TYPE_Q4_1 => Some((2 + 2 + QK4_1 / 2, QK4_1)),
        GGML_TYPE_Q5_0 => Some((2 + 4 + QK5_0 / 2, QK5_0)),
        GGML_TYPE_Q5_1 => Some((2 + 2 + 4 + QK5_1 / 2, QK5_1)),
        GGML_TYPE_Q8_0 => Some((2 + QK8_0, QK8_0)),
        GGML_TYPE_MXFP4 => Some((1 + QK_MXFP4 / 2, QK_MXFP4)),
        GGML_TYPE_Q2_K => Some((QK_K / 16 + QK_K / 4 + 2 + 2, QK_K)),
        GGML_TYPE_Q3_K => Some((QK_K / 8 + QK_K / 4 + 12 + 2, QK_K)),
        GGML_TYPE_Q4_K => Some((2 + 2 + K_SCALE_SIZE + QK_K / 2, QK_K)),
        GGML_TYPE_Q5_K => Some((2 + 2 + K_SCALE_SIZE + QK_K / 8 + QK_K / 2, QK_K)),
        GGML_TYPE_Q6_K => Some((QK_K / 2 + QK_K / 4 + QK_K / 16 + 2, QK_K)),
        GGML_TYPE_IQ2_XXS => Some((2 + (QK_K / 8) * 2, QK_K)),
        GGML_TYPE_IQ2_XS => Some((2 + (QK_K / 8) * 2 + QK_K / 32, QK_K)),
        GGML_TYPE_IQ1_S => Some((2 + QK_K / 8 + QK_K / 16, QK_K)),
        GGML_TYPE_IQ1_M => Some((QK_K / 8 + QK_K / 16 + QK_K / 32, QK_K)),
        GGML_TYPE_IQ1_XS => Some((2 + QK_K / 8 + QK_K / 32 + QK_K / 64, QK_K)),
        GGML_TYPE_IQ1_XXS => Some((2 + QK_K / 8 + QK_K / 32, QK_K)),
        GGML_TYPE_IQ1_XXXS => Some((2 + QK_K / 8 + QK_K / 64, QK_K)),
        GGML_TYPE_IQ2_S => Some((2 + QK_K / 4 + QK_K / 32 + QK_K / 32, QK_K)),
        GGML_TYPE_IQ3_XXS => Some((2 + 3 * (QK_K / 8), QK_K)),
        GGML_TYPE_IQ3_S => Some((2 + QK_K / 4 + QK_K / 32 + QK_K / 8 + QK_K / 64, QK_K)),
        // Byte-for-byte the same footprint as the `Q4_0` they repack — the
        // interleaving moves bytes around, it doesn't add or drop any — so
        // `tensor_byte_size` is right for them without knowing anything
        // about the interleaving itself.
        GGML_TYPE_Q4_0_4_4 | GGML_TYPE_Q4_0_4_8 | GGML_TYPE_Q4_0_8_8 | GGML_TYPE_IQ4_NL_4_4
        | GGML_TYPE_IQ4_NL_4_8 | GGML_TYPE_IQ4_NL_8_8 => Some((2 + QK4_0 / 2, QK4_0)),
        GGML_TYPE_IQ4_NL => Some((2 + QK4_NL / 2, QK4_NL)),
        GGML_TYPE_IQ4_XS => Some((2 + 2 + QK_K / 64 + QK_K / 2, QK_K)),
        _ => None,
    }
}

/// Whether [`dequantize`] can read this `ggml_type` at all.
///
/// Shares [`block_layout`]'s table with [`tensor_byte_size`], which is what
/// keeps the two from drifting: a type without a block layout has no
/// dequantizer either, and every type with one is matched below in
/// [`dequantize`]. Used by `engine::loader::model_load_support` so `list`
/// can say `No` for a file whose *architecture* is fine but whose tensors
/// this build can't decode — otherwise the column promises a load that then
/// fails partway through.
pub fn supports_type(ggml_type: u32) -> bool {
    block_layout(ggml_type).is_some()
}

/// Bytes in one `Q4_0` block, and elements it covers.
const Q4_0_BLOCK_BYTES: usize = 2 + QK4_0 / 2;

/// How a repacked type maps back to a plain one: `(base type, rows
/// interleaved, bytes per interleave run, per-byte XOR)`. `None` for
/// anything that isn't a repack.
///
/// Read off ggml's own `block<K, N>` / `block_iq4_nlx*` structs and the
/// `make_block_*` functions beside them, because two things about this
/// table do not follow from the type names:
///
/// - **Row counts.** `*_4_4` and `*_4_8` both pack **4** rows per record
///   and differ only in run length; only `*_8_8` packs 8. Reading the
///   second digit as a row count would scramble every tensor while still
///   producing plausible-looking output.
/// - **The XOR.** The `Q4_0` family flips both nibbles of every byte
///   (`^ 0x88`), which is what lets the ARM kernels treat a nibble as a
///   signed offset without a separate subtract. The `IQ4_NL` family does
///   **not**: its nibble is an index into [`KVALUES_IQ4NL`], so there is no
///   sign convention to fold, and `make_block_iq4_nlx4`/`x8` copy the runs
///   verbatim. Applying the `Q4_0` mask to an `IQ4_NL` tensor would pick
///   the wrong codebook entry for every single weight.
///
/// Both base types use an 18-byte, 32-element block, so all six share one
/// set of byte arithmetic.
///
/// `IQ4_NL_4_8` is the one entry without a live upstream implementation:
/// `make_block_iq4_nlx4`'s 8-byte branch is commented out and marked "this
/// branch seems wrong", so ggml has no executable definition of it. The
/// entry below is the straightforward generalization — the `x4` index math
/// with 8-byte runs, exactly as the `Q4_0` pair differ from each other.
const fn repack_layout(ggml_type: u32) -> Option<(u32, usize, usize, u8)> {
    match ggml_type {
        GGML_TYPE_Q4_0_4_4 => Some((GGML_TYPE_Q4_0, 4, 4, 0x88)),
        GGML_TYPE_Q4_0_4_8 => Some((GGML_TYPE_Q4_0, 4, 8, 0x88)),
        GGML_TYPE_Q4_0_8_8 => Some((GGML_TYPE_Q4_0, 8, 8, 0x88)),
        GGML_TYPE_IQ4_NL_4_4 => Some((GGML_TYPE_IQ4_NL, 4, 4, 0x00)),
        GGML_TYPE_IQ4_NL_4_8 => Some((GGML_TYPE_IQ4_NL, 4, 8, 0x00)),
        GGML_TYPE_IQ4_NL_8_8 => Some((GGML_TYPE_IQ4_NL, 8, 8, 0x00)),
        _ => None,
    }
}

/// Whether `ggml_type` is one of the row-interleaved repack layouts that
/// [`deinterleave_repack`] converts at load time.
pub fn is_repacked(ggml_type: u32) -> bool {
    repack_layout(ggml_type).is_some()
}

/// Rewrites a whole repacked tensor (`src`, an `[in_dim, out_dim]` matrix)
/// into the plain `Q4_0` or `IQ4_NL` bytes it was packed from, returning
/// `(base type, bytes)`.
///
/// The inverse of ggml's `repack_{q4_0,iq4_nl}_to_*_bl` plus the
/// `make_block_*` functions they call (`ggml/src/ggml-cpu/repack.cpp`),
/// read directly from those rather than reconstructed from the format's
/// description. Two things are going on at once:
///
/// - **Rows.** Records are emitted per group of `n_rows` consecutive rows:
///   for group `g` and block-column `x`, one record carries block `x` of
///   *every* row in the group. So a row's blocks are strided across the
///   group, not contiguous — which is why this cannot be done per row, and
///   why `QuantMatrix`'s `start + index * row_bytes` addressing would read
///   pure noise from an untouched tensor.
/// - **Bytes.** Within a record, the `qs` payload is emitted in `run`-byte
///   runs round-robin across the group's rows, each XORed with the mask
///   [`repack_layout`] gives for the type (`0x88` for the `Q4_0` family,
///   nothing for `IQ4_NL`). XOR is its own inverse, so undoing it is the
///   same operation.
///
/// The returned bytes are the same length as `src`; an error means the
/// tensor's shape can't have been produced by the packer at all.
pub fn deinterleave_repack(
    ggml_type: u32,
    src: &[u8],
    in_dim: usize,
    out_dim: usize,
) -> Result<(u32, Vec<u8>)> {
    let Some((base, n_rows, run, xor)) = repack_layout(ggml_type) else {
        bail!(
            "{} is not a row-interleaved repack layout",
            ggml_type_name(ggml_type)
        );
    };
    // Both are enforced by the packer itself (`repack_q4_0_to_q4_0_4_bl`
    // returns -1 rather than packing), so a file violating either is
    // malformed, not merely unusual.
    if !in_dim.is_multiple_of(QK4_0) {
        bail!(
            "{} row length {in_dim} is not a multiple of {QK4_0}",
            ggml_type_name(ggml_type)
        );
    }
    if !out_dim.is_multiple_of(n_rows) {
        bail!(
            "{} row count {out_dim} is not a multiple of the {n_rows} rows it interleaves",
            ggml_type_name(ggml_type)
        );
    }
    let n_blocks = in_dim / QK4_0;
    // One record holds `n_rows` whole `Q4_0` blocks: their `f16` scales
    // first, then their `qs` payloads interleaved.
    let scale_bytes = 2 * n_rows;
    let record_bytes = n_rows * Q4_0_BLOCK_BYTES;
    let expected = record_bytes * n_blocks * (out_dim / n_rows);
    if src.len() != expected {
        bail!(
            "{} tensor is {} bytes, expected {expected} for {in_dim} x {out_dim}",
            ggml_type_name(ggml_type),
            src.len()
        );
    }

    let mut out = vec![0u8; out_dim * n_blocks * Q4_0_BLOCK_BYTES];
    // Runs per record, straight from ggml's `end`: the group's whole `qs`
    // payload (`16` bytes per row) cut into `run`-byte pieces.
    let runs = (Q4_0_BLOCK_BYTES - 2) * n_rows / run;
    for group in 0..out_dim / n_rows {
        for x in 0..n_blocks {
            let record = &src[(group * n_blocks + x) * record_bytes..][..record_bytes];
            let (scales, qs) = record.split_at(scale_bytes);
            // Where row `r` of this group keeps its block `x`, in the
            // plain-`Q4_0` output.
            let block_at = |r: usize| ((group * n_rows + r) * n_blocks + x) * Q4_0_BLOCK_BYTES;
            for r in 0..n_rows {
                out[block_at(r)..block_at(r) + 2].copy_from_slice(&scales[2 * r..2 * r + 2]);
            }
            for i in 0..runs {
                let row = i % n_rows;
                let dst = block_at(row) + 2 + (i / n_rows) * run;
                for (b, &byte) in qs[i * run..][..run].iter().enumerate() {
                    out[dst + b] = byte ^ xor;
                }
            }
        }
    }
    Ok((base, out))
}

/// The error for a `ggml_type` this build can't read, phrased by *why*.
///
/// A type ggml itself removed is a different situation from one this engine
/// hasn't implemented, and the difference is the whole of what the reader
/// should do next: waiting for orangu to add `Q4_2` is waiting forever,
/// because upstream `llama.cpp` dropped it too and nothing succeeded it.
/// Saying "not yet supported" for both sends them to the wrong place.
fn unsupported_type_error(ggml_type: u32) -> anyhow::Error {
    let name = ggml_type_name(ggml_type);
    if is_removed_ggml_type(ggml_type) {
        return anyhow::anyhow!(
            "tensor type {name} was removed from ggml itself — upstream llama.cpp cannot read \
             this file either. It predates the K-quants and has no successor; re-quantize \
             from the source weights."
        );
    }
    anyhow::anyhow!("tensor type {name} is not yet supported by orangu-server")
}

/// The exact byte length a tensor with `element_count` elements of
/// `ggml_type` occupies in the GGUF file's data section.
pub fn tensor_byte_size(ggml_type: u32, element_count: u64) -> Result<u64> {
    let Some((block_bytes, block_elems)) = block_layout(ggml_type) else {
        return Err(unsupported_type_error(ggml_type));
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

/// [`dequantize`] into a caller-owned buffer, so a loop over many rows
/// allocates once instead of once per row.
///
/// `out` is overwritten, not appended to, and ends holding exactly
/// `element_count` values. Its existing capacity is reused — the whole point:
/// a MoE layer reads thousands of expert rows, and every one of them used to
/// mean a fresh allocation of `in_dim` floats.
/// [`dequantize`] into a caller-owned buffer, so a loop over many rows
/// allocates once instead of once per row.
///
/// `out` is overwritten, not appended to, and ends holding exactly
/// `element_count` values. Its existing capacity is reused — the whole point:
/// a MoE layer reads thousands of expert rows, and each one used to mean a
/// fresh allocation of `in_dim` floats.
///
/// This is the real implementation; [`dequantize`] is the thin wrapper that
/// supplies a fresh buffer, so the two can never disagree about a format.
pub fn dequantize_into(
    ggml_type: u32,
    bytes: &[u8],
    element_count: usize,
    out: &mut Vec<f32>,
) -> Result<()> {
    out.clear();
    out.reserve(element_count);
    match ggml_type {
        GGML_TYPE_F32 => out.extend(
            bytes
                .chunks_exact(4)
                .take(element_count)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
        ),
        GGML_TYPE_F16 => out.extend(
            bytes
                .chunks_exact(2)
                .take(element_count)
                .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32()),
        ),
        GGML_TYPE_I32 => out.extend(
            bytes
                .chunks_exact(4)
                .take(element_count)
                .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32),
        ),
        // bfloat16: the top 16 bits of an f32 (sign + 8-bit exponent + 7-bit
        // mantissa) — reconstruct by left-shifting into the low bits' place.
        GGML_TYPE_BF16 => out.extend(
            bytes
                .chunks_exact(2)
                .take(element_count)
                .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16)),
        ),
        GGML_TYPE_Q4_0 => dequantize_q4_0(bytes, element_count, out),
        GGML_TYPE_Q4_1 => dequantize_q4_1(bytes, element_count, out),
        GGML_TYPE_Q5_0 => dequantize_q5_0(bytes, element_count, out),
        GGML_TYPE_Q5_1 => dequantize_q5_1(bytes, element_count, out),
        GGML_TYPE_Q8_0 => dequantize_q8_0(bytes, element_count, out),
        GGML_TYPE_MXFP4 => dequantize_mxfp4(bytes, element_count, out),
        GGML_TYPE_Q2_K => dequantize_q2_k(bytes, element_count, out),
        GGML_TYPE_Q3_K => dequantize_q3_k(bytes, element_count, out),
        GGML_TYPE_Q4_K => dequantize_q4_k(bytes, element_count, out),
        GGML_TYPE_Q5_K => dequantize_q5_k(bytes, element_count, out),
        GGML_TYPE_Q6_K => dequantize_q6_k(bytes, element_count, out),
        GGML_TYPE_IQ2_XXS => dequantize_iq2_xxs(bytes, element_count, out),
        GGML_TYPE_IQ2_XS => dequantize_iq2_xs(bytes, element_count, out),
        GGML_TYPE_IQ1_S => dequantize_iq1_s(bytes, element_count, out),
        GGML_TYPE_IQ1_M => dequantize_iq1_m(bytes, element_count, out),
        GGML_TYPE_IQ1_XS => dequantize_iq1_xs(bytes, element_count, out),
        GGML_TYPE_IQ1_XXS => dequantize_iq1_xxs(bytes, element_count, out),
        GGML_TYPE_IQ1_XXXS => dequantize_iq1_xxxs(bytes, element_count, out),
        GGML_TYPE_IQ2_S => dequantize_iq2_s(bytes, element_count, out),
        GGML_TYPE_IQ3_XXS => dequantize_iq3_xxs(bytes, element_count, out),
        GGML_TYPE_IQ3_S => dequantize_iq3_s(bytes, element_count, out),
        // Deliberately not dequantized here. A repacked row's blocks are
        // strided across its 4- or 8-row group, so `bytes` for "one row" is
        // not a thing that exists — the unit is the whole tensor, whose
        // shape this function is never told. `engine::loader` converts them
        // to `Q4_0` when the model opens, so reaching this arm means a new
        // caller bypassed that; say so rather than return plausible noise.
        t if is_repacked(t) => bail!(
            "{} must be de-interleaved with quant::deinterleave_repack before any row of it \
             is read — engine::loader does this at load time, so this is a bug in a caller \
             that built a tensor view directly",
            ggml_type_name(ggml_type)
        ),
        GGML_TYPE_IQ4_NL => dequantize_iq4_nl(bytes, element_count, out),
        GGML_TYPE_IQ4_XS => dequantize_iq4_xs(bytes, element_count, out),
        _ => return Err(unsupported_type_error(ggml_type)),
    }
    Ok(())
}

/// Dequantizes `bytes` (exactly `tensor_byte_size(ggml_type, element_count)`
/// long) to `element_count` `f32` values, in the tensor's original order.
pub fn dequantize(ggml_type: u32, bytes: &[u8], element_count: usize) -> Result<Vec<f32>> {
    let mut out = Vec::new();
    dequantize_into(ggml_type, bytes, element_count, &mut out)?;
    Ok(out)
}

fn e8m0_to_fp32_half(x: u8) -> f32 {
    let bits = if x < 2 {
        0x0020_0000u32 << x
    } else {
        u32::from(x - 1) << 23
    };
    f32::from_bits(bits)
}

fn dequantize_mxfp4(bytes: &[u8], element_count: usize, out: &mut Vec<f32>) {
    const BLOCK_BYTES: usize = 1 + QK_MXFP4 / 2;
    out.clear();
    out.reserve(element_count);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let d = e8m0_to_fp32_half(block[0]);
        let qs = &block[1..];
        for &q in qs.iter().take(QK_MXFP4 / 2) {
            out.push(KVALUES_MXFP4[(q & 0x0F) as usize] as f32 * d);
        }
        for &q in qs.iter().take(QK_MXFP4 / 2) {
            out.push(KVALUES_MXFP4[(q >> 4) as usize] as f32 * d);
        }
    }
    out.truncate(element_count);
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
fn dequantize_q4_0(bytes: &[u8], element_count: usize, out: &mut Vec<f32>) {
    const BLOCK_BYTES: usize = 2 + QK4_0 / 2;
    out.clear();
    out.reserve(element_count);
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
}

/// `block_q5_0`: `{ d: f16, qh: [u8; 4], qs: [u8; 16] }`, 32 elements — a
/// 5-bit nibble (4 low bits in `qs`, the 5th/high bit packed across `qh`),
/// offset by 16 (the 5-bit analogue of Q4_0's offset-by-8), mirrors ggml's
/// `dequantize_row_q5_0`.
fn dequantize_q5_0(bytes: &[u8], element_count: usize, out: &mut Vec<f32>) {
    const BLOCK_BYTES: usize = 2 + 4 + QK5_0 / 2;
    out.clear();
    out.reserve(element_count);
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
fn dequantize_q4_1(bytes: &[u8], element_count: usize, out: &mut Vec<f32>) {
    const BLOCK_BYTES: usize = 2 + 2 + QK4_1 / 2;
    out.clear();
    out.reserve(element_count);
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
}

/// `block_q5_1`: `{ d: f16, m: f16, qh: [u8; 4], qs: [u8; 16] }`, 32
/// elements — mirrors ggml's `dequantize_row_q5_1`. `Q5_0`'s fifth bit
/// packed across `qh`, with `Q4_1`'s stored minimum in place of the fixed
/// offset.
fn dequantize_q5_1(bytes: &[u8], element_count: usize, out: &mut Vec<f32>) {
    const BLOCK_BYTES: usize = 2 + 2 + 4 + QK5_1 / 2;
    out.clear();
    out.reserve(element_count);
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
}

/// `block_q8_0`: `{ d: f16, qs: [i8; 32] }`, 32 elements.
fn dequantize_q8_0(bytes: &[u8], element_count: usize, out: &mut Vec<f32>) {
    const BLOCK_BYTES: usize = 2 + QK8_0;
    out.clear();
    out.reserve(element_count);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let d = read_f16(block, 0);
        let qs = &block[2..];
        out.extend(qs.iter().map(|&q| (q as i8) as f32 * d));
    }
    out.truncate(element_count);
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
fn dequantize_q4_k(bytes: &[u8], element_count: usize, out: &mut Vec<f32>) {
    const BLOCK_BYTES: usize = 2 + 2 + K_SCALE_SIZE + QK_K / 2;
    out.clear();
    out.reserve(element_count);
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
}

/// `block_q5_K`: `{ d: f16, dmin: f16, scales: [u8; 12], qh: [u8; 32],
/// qs: [u8; 128] }`, 256 elements — mirrors ggml's `dequantize_row_q5_K`:
/// like `Q4_K`, plus a 5th quant bit packed across `qh` (each `qh` byte's 8
/// bits are consumed one pair per 64-element sub-group, over all 4 groups).
fn dequantize_q5_k(bytes: &[u8], element_count: usize, out: &mut Vec<f32>) {
    const BLOCK_BYTES: usize = 2 + 2 + K_SCALE_SIZE + QK_K / 8 + QK_K / 2;
    out.clear();
    out.reserve(element_count);
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
}

/// `block_q6_K`: `{ ql: [u8; 128], qh: [u8; 64], scales: [i8; 16], d: f16 }`,
/// 256 elements — mirrors ggml's `dequantize_row_q6_K`.
fn dequantize_q6_k(bytes: &[u8], element_count: usize, out: &mut Vec<f32>) {
    const BLOCK_BYTES: usize = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2;
    out.clear();
    out.reserve(element_count);
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
fn dequantize_q2_k(bytes: &[u8], element_count: usize, out: &mut Vec<f32>) {
    const BLOCK_BYTES: usize = QK_K / 16 + QK_K / 4 + 2 + 2;
    out.clear();
    out.reserve(element_count);
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
fn dequantize_q3_k(bytes: &[u8], element_count: usize, out: &mut Vec<f32>) {
    const BLOCK_BYTES: usize = QK_K / 8 + QK_K / 4 + 12 + 2;
    out.clear();
    out.reserve(element_count);
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
pub(crate) fn unpack_q3_k_scales(packed: &[u8]) -> [i32; 16] {
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

/// `block_iq2_xxs`: `{ d: f16, qs: [u16; 32] }`, 256 elements as 8 groups of
/// 32 — mirrors ggml's `dequantize_row_iq2_xxs`.
///
/// The tightest of the `iq2*` formats: there is no `scales` array at all.
/// Each 32-weight group reads two `u32` out of `qs`; the first supplies four
/// 8-bit codebook indices into the 256-entry [`IQ2XXS_GRID`], and the second
/// packs four 7-bit [`KSIGNS_IQ2XS`] indices plus, in its top 4 bits, the
/// group's own scale.
fn dequantize_iq2_xxs(bytes: &[u8], element_count: usize, out: &mut Vec<f32>) {
    const BLOCK_BYTES: usize = 2 + (QK_K / 8) * 2;
    out.clear();
    out.reserve(element_count);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let d = read_f16(block, 0);
        let qs = &block[2..];
        for ib32 in 0..QK_K / 32 {
            // ggml indexes `qs` as `uint16_t*` and steps by 4, i.e. 8 bytes.
            let base = 8 * ib32;
            let aux0 = u32::from_le_bytes(qs[base..base + 4].try_into().expect("4 bytes"));
            let aux1 = u32::from_le_bytes(qs[base + 4..base + 8].try_into().expect("4 bytes"));
            let db = d * (0.5 + (aux1 >> 28) as f32) * 0.25;
            let aux8 = aux0.to_le_bytes();
            for l in 0..4 {
                let grid = IQ2XXS_GRID[aux8[l] as usize];
                let signs = KSIGNS_IQ2XS[((aux1 >> (7 * l)) & 127) as usize];
                push_iq_grid(out, grid, 8, db, signs);
            }
        }
    }
    out.truncate(element_count);
}

/// One `IQ1_S`/`IQ1_M` codebook entry: 8 **signed** bytes, each shifted by a
/// per-group `delta` before scaling.
///
/// The `iq1*` formats carry no sign field at all — the grid values are
/// already signed, and the only per-group freedom is whether `delta` is
/// added or subtracted, which is why this can't reuse [`push_iq_grid`].
fn push_iq1_grid(out: &mut Vec<f32>, grid: u64, scale: f32, delta: f32) {
    for byte in grid.to_le_bytes() {
        out.push(scale * ((byte as i8) as f32 + delta));
    }
}

/// The `±` offset every `IQ1_S`/`IQ1_M` weight carries on top of its
/// codebook value — ggml's `IQ1S_DELTA`/`IQ1M_DELTA`, both `0.125`.
const IQ1_DELTA: f32 = 0.125;

/// `block_iq1_s`: `{ d: f16, qs: [u8; 32], qh: [u16; 8] }`, 256 elements as
/// 8 groups of 32 — mirrors ggml's `dequantize_row_iq1_s`.
///
/// 1.5625 bpw. Each group's `qh` entry carries three things at once: a
/// 3-bit scale (bits 12..15), the sign of the group's `delta` (bit 15), and
/// the high 3 bits of each of the group's four 11-bit [`IQ1S_GRID`] indices
/// (bits 0..9, three per index).
fn dequantize_iq1_s(bytes: &[u8], element_count: usize, out: &mut Vec<f32>) {
    const BLOCK_BYTES: usize = 2 + QK_K / 8 + QK_K / 16;
    out.clear();
    out.reserve(element_count);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let d = read_f16(block, 0);
        let qs = &block[2..2 + QK_K / 8];
        let qh_bytes = &block[2 + QK_K / 8..];
        for ib in 0..QK_K / 32 {
            let qh = u16::from_le_bytes([qh_bytes[2 * ib], qh_bytes[2 * ib + 1]]);
            let dl = d * (2 * ((qh >> 12) & 7) + 1) as f32;
            let delta = if qh & 0x8000 != 0 {
                -IQ1_DELTA
            } else {
                IQ1_DELTA
            };
            for l in 0..4 {
                let idx = qs[4 * ib + l] as usize | ((((qh >> (3 * l)) & 7) as usize) << 8);
                push_iq1_grid(out, IQ1S_GRID[idx], dl, delta);
            }
        }
    }
    out.truncate(element_count);
}

/// `block_iq1_m`: `{ qs: [u8; 32], qh: [u8; 16], scales: [u8; 8] }`, 256
/// elements as 8 groups of 32 — mirrors ggml's `dequantize_row_iq1_m`.
///
/// 1.75 bpw, and the only quantization here with **no `d` field**: the
/// block's `f16` scale is scattered four nibbles at a time across the top of
/// the four `scales` `u16`s and has to be reassembled before it can be read
/// as a half. Each group also gets *two* 3-bit sub-scales (one per 16
/// weights) rather than one, and `delta`'s sign moves to two bits of each
/// `qh` byte.
fn dequantize_iq1_m(bytes: &[u8], element_count: usize, out: &mut Vec<f32>) {
    const BLOCK_BYTES: usize = QK_K / 8 + QK_K / 16 + QK_K / 32;
    out.clear();
    out.reserve(element_count);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let qs = &block[..QK_K / 8];
        let qh = &block[QK_K / 8..QK_K / 8 + QK_K / 16];
        let scales = &block[QK_K / 8 + QK_K / 16..];
        let sc: [u16; 4] =
            std::array::from_fn(|i| u16::from_le_bytes([scales[2 * i], scales[2 * i + 1]]));
        let packed =
            (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        let d = f16_bits_to_f32(packed);

        for ib in 0..QK_K / 32 {
            let s = sc[ib / 2];
            let shift = 6 * (ib % 2);
            let dl1 = d * (2 * ((s >> shift) & 7) + 1) as f32;
            let dl2 = d * (2 * ((s >> (shift + 3)) & 7) + 1) as f32;
            let (qh0, qh1) = (qh[2 * ib] as usize, qh[2 * ib + 1] as usize);
            let idx = [
                qs[4 * ib] as usize | ((qh0 << 8) & 0x700),
                qs[4 * ib + 1] as usize | ((qh0 << 4) & 0x700),
                qs[4 * ib + 2] as usize | ((qh1 << 8) & 0x700),
                qs[4 * ib + 3] as usize | ((qh1 << 4) & 0x700),
            ];
            let sign = |bit: usize, byte: usize| {
                if byte & bit != 0 {
                    -IQ1_DELTA
                } else {
                    IQ1_DELTA
                }
            };
            let delta = [
                sign(0x08, qh0),
                sign(0x80, qh0),
                sign(0x08, qh1),
                sign(0x80, qh1),
            ];
            for l in 0..4 {
                let scale = if l < 2 { dl1 } else { dl2 };
                push_iq1_grid(out, IQ1S_GRID[idx[l]], scale, delta[l]);
            }
        }
    }
    out.truncate(element_count);
}

// The three quantizations *below* `IQ1_S`, which reach 1.4375, 1.3125 and
// 1.1875 bpw by narrowing one field of it and nothing else.
//
// `IQ1_S` spends 11 of its 1.5625 bits per weight on an index into a
// 2048-point codebook — 88% of the whole budget — so the index is the only
// field with room to give. Each of these types keeps the 8 low bits in
// `qs` and stores fewer high bits (2, 1, then none at all), indexing a
// 1024-, 512- or 256-point selection from the very same lattice
// (`IQ1XS_GRID`, `IQ1XXS_GRID`, `IQ1XXXS_GRID`).
//
// Everything else keeps its `IQ1_S` meaning exactly: `d` is the `f16`
// super-block scale, a 3-bit sub-block code gives `d * (2*ls + 1)`, and a
// sign bit picks `±IQ1_DELTA`. Only the reassembly of the index differs,
// which is why all three end in the same `push_iq1_grid` call
// `dequantize_iq1_s` does.
//
// Where those two fields *live* is what the byte counts force. A 32-weight
// sub-block must carry four index-high fields plus a scale and a sign, and
// at 10 / 9 / 8 index bits that is 12 / 8 / 4 bits — so `IQ1_XXS` alone
// fits scale and sign into the same byte as its index-highs, while
// `IQ1_XS` and `IQ1_XXXS` put them in a separate nibble-per-sub-block
// array (`iq1_narrow_sub_scale`).

/// The `(scale multiplier, delta)` that `IQ1_XS`/`IQ1_XXXS` pack into one
/// nibble: bits 0..2 the 3-bit sub-block scale, bit 3 the sign of the
/// delta.
fn iq1_narrow_sub_scale(nibble: u8) -> (f32, f32) {
    let scale = (2 * (nibble & 7) + 1) as f32;
    let delta = if nibble & 8 != 0 {
        -IQ1_DELTA
    } else {
        IQ1_DELTA
    };
    (scale, delta)
}

/// `block_iq1_xs`: `{ d: f16, qs: [u8; 32], qh: [u8; 8], sc: [u8; 4] }`,
/// 1.4375 bpw. Each `qh` byte holds the four 2-bit index-highs of its
/// 32-weight group; the scale and sign live in `sc`.
fn dequantize_iq1_xs(bytes: &[u8], element_count: usize, out: &mut Vec<f32>) {
    const BLOCK_BYTES: usize = 2 + QK_K / 8 + QK_K / 32 + QK_K / 64;
    out.clear();
    out.reserve(element_count);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let d = read_f16(block, 0);
        let qs = &block[2..2 + QK_K / 8];
        let qh = &block[2 + QK_K / 8..2 + QK_K / 8 + QK_K / 32];
        let sc = &block[2 + QK_K / 8 + QK_K / 32..];
        for ib in 0..QK_K / 32 {
            let (scale, delta) = iq1_narrow_sub_scale((sc[ib / 2] >> (4 * (ib % 2))) & 0xf);
            for l in 0..4 {
                let idx = qs[4 * ib + l] as usize | ((((qh[ib] >> (2 * l)) & 3) as usize) << 8);
                push_iq1_grid(out, IQ1XS_GRID[idx], d * scale, delta);
            }
        }
    }
    out.truncate(element_count);
}

/// `block_iq1_xxs`: `{ d: f16, qs: [u8; 32], qh: [u8; 8] }`, 1.3125 bpw —
/// the one that packs perfectly. A single index-high bit per group of 8
/// leaves exactly four bits over in the same byte, which is where the
/// 3-bit scale (bits 4..6) and the delta's sign (bit 7) go, so there is no
/// separate scale array.
fn dequantize_iq1_xxs(bytes: &[u8], element_count: usize, out: &mut Vec<f32>) {
    const BLOCK_BYTES: usize = 2 + QK_K / 8 + QK_K / 32;
    out.clear();
    out.reserve(element_count);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let d = read_f16(block, 0);
        let qs = &block[2..2 + QK_K / 8];
        let qh = &block[2 + QK_K / 8..];
        for ib in 0..QK_K / 32 {
            let dl = d * (2 * ((qh[ib] >> 4) & 7) + 1) as f32;
            let delta = if qh[ib] & 0x80 != 0 {
                -IQ1_DELTA
            } else {
                IQ1_DELTA
            };
            for l in 0..4 {
                let idx = qs[4 * ib + l] as usize | ((((qh[ib] >> l) & 1) as usize) << 8);
                push_iq1_grid(out, IQ1XXS_GRID[idx], dl, delta);
            }
        }
    }
    out.truncate(element_count);
}

/// `block_iq1_xxxs`: `{ d: f16, qs: [u8; 32], sc: [u8; 4] }`, 1.1875 bpw —
/// 38 bytes per 256 weights, the narrowest this family goes. The index fits
/// a byte, so the block carries no index-high field at all and `qs` alone
/// selects the codebook entry.
fn dequantize_iq1_xxxs(bytes: &[u8], element_count: usize, out: &mut Vec<f32>) {
    const BLOCK_BYTES: usize = 2 + QK_K / 8 + QK_K / 64;
    out.clear();
    out.reserve(element_count);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let d = read_f16(block, 0);
        let qs = &block[2..2 + QK_K / 8];
        let sc = &block[2 + QK_K / 8..];
        for ib in 0..QK_K / 32 {
            let (scale, delta) = iq1_narrow_sub_scale((sc[ib / 2] >> (4 * (ib % 2))) & 0xf);
            for l in 0..4 {
                push_iq1_grid(out, IQ1XXXS_GRID[qs[4 * ib + l] as usize], d * scale, delta);
            }
        }
    }
    out.truncate(element_count);
}

/// `block_iq2_xs`: `{ d: f16, qs: [u16; 32], scales: [u8; 8] }`, 256
/// elements as 8 groups of 32 — mirrors ggml's `dequantize_row_iq2_xs`.
///
/// Each `qs` entry is one codebook lookup covering 8 weights: its low 9 bits
/// index [`IQ2XS_GRID`] and its top 7 bits index [`KSIGNS_IQ2XS`] for the
/// sign pattern. Each `scales` byte holds two 4-bit scales, one per 16-weight
/// half of its 32-weight group.
fn dequantize_iq2_xs(bytes: &[u8], element_count: usize, out: &mut Vec<f32>) {
    const BLOCK_BYTES: usize = 2 + (QK_K / 8) * 2 + QK_K / 32;
    out.clear();
    out.reserve(element_count);
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
                push_iq_grid(out, grid, 8, db[l / 2], signs);
            }
        }
    }
    out.truncate(element_count);
}

/// `block_iq2_s`: `{ d: f16, qs: [u8; 64], qh: [u8; 8], scales: [u8; 8] }`,
/// 256 elements as 8 groups of 32 — mirrors ggml's `dequantize_row_iq2_s`.
///
/// The scale layout is `IQ2_XS`'s. The codebook index is wider (10 bits into
/// the 1024-entry [`IQ2S_GRID`]): 8 bits from `qs`, the top 2 from a
/// per-group `qh` byte. The signs are no longer a 7-bit index into
/// [`KSIGNS_IQ2XS`] but a full byte, stored in the *second half* of `qs` —
/// ggml spells that as `signs = qs + QK_K/8`, an alias into the same array.
fn dequantize_iq2_s(bytes: &[u8], element_count: usize, out: &mut Vec<f32>) {
    const BLOCK_BYTES: usize = 2 + QK_K / 4 + QK_K / 32 + QK_K / 32;
    out.clear();
    out.reserve(element_count);
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
                push_iq_grid(out, IQ2S_GRID[idx], 8, db[l / 2], signs[4 * ib32 + l]);
            }
        }
    }
    out.truncate(element_count);
}

/// `block_iq3_xxs`: `{ d: f16, qs: [u8; 96] }`, 256 elements as 8 groups of
/// 32 — mirrors ggml's `dequantize_row_iq3_xxs`.
///
/// The single `qs` array is two arrays end to end: 64 bytes of codebook
/// indices into the 256-entry [`IQ3XXS_GRID`] (4 weights each, so two
/// lookups per 8-weight run), then 32 bytes read as eight little-endian
/// `u32`s, one per group, each packing four 7-bit sign indices plus a 4-bit
/// scale in its top nibble.
fn dequantize_iq3_xxs(bytes: &[u8], element_count: usize, out: &mut Vec<f32>) {
    const BLOCK_BYTES: usize = 2 + 3 * (QK_K / 8);
    out.clear();
    out.reserve(element_count);
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
                push_iq_grid(out, grid1 as u64 | ((grid2 as u64) << 32), 8, db, signs);
            }
        }
    }
    out.truncate(element_count);
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
fn dequantize_iq3_s(bytes: &[u8], element_count: usize, out: &mut Vec<f32>) {
    const BLOCK_BYTES: usize = 2 + QK_K / 4 + QK_K / 32 + QK_K / 8 + QK_K / 64;
    out.clear();
    out.reserve(element_count);
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
                    out,
                    IQ3S_GRID[idx1] as u64 | ((IQ3S_GRID[idx2] as u64) << 32),
                    8,
                    db,
                    signs[4 * ib32 + l],
                );
            }
        }
    }
    out.truncate(element_count);
}

/// `block_iq4_nl`: `{ d: f16, qs: [u8; 16] }`, 32 elements — mirrors ggml's
/// `dequantize_row_iq4_nl`.
///
/// Structurally this is `Q4_0` with the linear `nibble - 8` replaced by a
/// lookup into [`KVALUES_IQ4NL`], the same 16 non-uniformly spaced levels
/// `IQ4_XS` uses; it carries no codebook of lattice points and no
/// sub-block scales, just one `f16` per 32 weights. The 16 low nibbles are
/// the first 16 weights and the high nibbles the next 16 — split halves,
/// not interleaved, as in [`dequantize_q4_0`].
fn dequantize_iq4_nl(bytes: &[u8], element_count: usize, out: &mut Vec<f32>) {
    const BLOCK_BYTES: usize = 2 + QK4_NL / 2;
    out.clear();
    out.reserve(element_count);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let d = read_f16(block, 0);
        let qs = &block[2..];
        let mut lo = [0f32; QK4_NL / 2];
        let mut hi = [0f32; QK4_NL / 2];
        for (j, &byte) in qs.iter().enumerate() {
            lo[j] = d * KVALUES_IQ4NL[(byte & 0x0F) as usize] as f32;
            hi[j] = d * KVALUES_IQ4NL[(byte >> 4) as usize] as f32;
        }
        out.extend_from_slice(&lo);
        out.extend_from_slice(&hi);
    }
    out.truncate(element_count);
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
fn dequantize_iq4_xs(bytes: &[u8], element_count: usize, out: &mut Vec<f32>) {
    const BLOCK_BYTES: usize = 2 + 2 + QK_K / 64 + QK_K / 2;
    out.clear();
    out.reserve(element_count);
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
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every type [`dequantize`] handles, with an element count that is a
    /// whole number of its blocks, and a checksum over the exact bits it
    /// produced for [`fixture_bytes`].
    ///
    /// **These values were captured from the implementation that existed
    /// before `dequantize_into`**, so they are an independent reference for
    /// it rather than a restatement of it. A rewrite that changes any
    /// format's output by one mantissa bit fails here, naming the format.
    const GOLDEN: &[(u32, &str, usize, u64)] = &[
        (GGML_TYPE_F32, "f32", 64, 0x1eb2_b799_2ae6_e834),
        (GGML_TYPE_F16, "f16", 64, 0xa4b3_8b79_0d3e_e825),
        (GGML_TYPE_BF16, "bf16", 64, 0x7fe9_5a39_1f2c_0825),
        (GGML_TYPE_I32, "i32", 64, 0x2223_13f8_19a6_8159),
        (GGML_TYPE_Q4_0, "q4_0", 256, 0x1623_25c9_68c1_0725),
        (GGML_TYPE_Q4_1, "q4_1", 256, 0x282a_888b_e2fd_640b),
        (GGML_TYPE_Q5_0, "q5_0", 256, 0x813f_8949_e12b_3725),
        (GGML_TYPE_Q5_1, "q5_1", 256, 0xca1e_0ebe_3c12_ef0e),
        (GGML_TYPE_Q8_0, "q8_0", 256, 0xc5d5_8782_f54e_52e5),
        (GGML_TYPE_MXFP4, "mxfp4", 256, 0x0875_e119_a32b_b725),
        (GGML_TYPE_Q2_K, "q2_k", 512, 0x3bea_8df7_bf68_d288),
        (GGML_TYPE_Q3_K, "q3_k", 512, 0x2183_e0b0_538c_6be5),
        (GGML_TYPE_Q4_K, "q4_k", 512, 0x9a72_68a4_32a5_c5fa),
        (GGML_TYPE_Q5_K, "q5_k", 512, 0x41c7_0b9b_e169_6e6d),
        (GGML_TYPE_Q6_K, "q6_k", 512, 0x6100_1d33_3cb0_cfd1),
        (GGML_TYPE_IQ2_XXS, "iq2_xxs", 512, 0x3265_6ac4_f581_d19d),
        (GGML_TYPE_IQ2_XS, "iq2_xs", 512, 0x7722_6bb1_1c18_4a05),
        (GGML_TYPE_IQ2_S, "iq2_s", 512, 0xc86c_2e15_90b4_3a35),
        (GGML_TYPE_IQ1_S, "iq1_s", 512, 0xdc0c_1a7b_8609_7ba5),
        (GGML_TYPE_IQ1_M, "iq1_m", 512, 0xf7ca_fbf0_92bc_33a5),
        // The three sub-`IQ1_S` types have no "before" implementation to be
        // captured from, so these were taken once the fixture in
        // `dequantize_matches_ggml_for_every_quantized_type` had already
        // agreed with ggml bit-for-bit. That test is what says the values
        // are *right*; this one is what says they never quietly change.
        (GGML_TYPE_IQ1_XS, "iq1_xs", 512, 0xae4c_e46b_ecf2_83e5),
        (GGML_TYPE_IQ1_XXS, "iq1_xxs", 512, 0x6c8e_911c_a956_b465),
        (GGML_TYPE_IQ1_XXXS, "iq1_xxxs", 512, 0xa7f0_c934_d9fc_8825),
        (GGML_TYPE_IQ3_XXS, "iq3_xxs", 512, 0xb565_b973_da18_d32d),
        (GGML_TYPE_IQ3_S, "iq3_s", 512, 0xa05c_c44c_4ab8_a2f5),
        (GGML_TYPE_IQ4_NL, "iq4_nl", 256, 0x7956_c82f_eba1_2125),
        (GGML_TYPE_IQ4_XS, "iq4_xs", 512, 0xd0ca_6239_e2d8_f06d),
    ];

    /// Deterministic bytes for a dequantization fixture. Not random: the
    /// checksums above are only a reference if the input is reproducible from
    /// the source alone, with no file to lose.
    fn fixture_bytes(len: usize) -> Vec<u8> {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 33) as u8
            })
            .collect()
    }

    /// A checksum over the exact bits of every value, so two implementations
    /// agreeing here agree to the last mantissa bit — an `f32` sum would let
    /// a reordering pass.
    fn bit_checksum(values: &[f32]) -> u64 {
        values.iter().fold(0xcbf2_9ce4_8422_2325u64, |acc, v| {
            (acc ^ u64::from(v.to_bits())).wrapping_mul(0x0000_0100_0000_01b3)
        })
    }

    /// The guard on `dequantize_into`: 27 formats, each still producing the
    /// bits it produced before the buffer-reusing rewrite.
    #[test]
    fn dequantizing_every_type_is_unchanged() {
        for &(ggml_type, name, count, expected) in GOLDEN {
            let len = tensor_byte_size(ggml_type, count as u64).unwrap() as usize;
            let bytes = fixture_bytes(len);
            let values = dequantize(ggml_type, &bytes, count).expect(name);
            assert_eq!(values.len(), count, "{name} produced the wrong count");
            assert_eq!(
                bit_checksum(&values),
                expected,
                "{name} dequantized differently than it used to"
            );
        }
    }

    /// The buffer-reusing entry point must agree with the allocating one on
    /// every format, and must leave the buffer holding exactly the values —
    /// no leftovers from whatever it held before.
    ///
    /// Compared as **bits**, not as `f32`. Quantized formats decode arbitrary
    /// bit patterns, and this fixture's `F32` block contains a `NaN`: under
    /// `==` a `NaN` is unequal to itself, so a value comparison fails on
    /// identical output. Bits are also the stricter check — two different
    /// `NaN` payloads still differ.
    #[test]
    fn dequantize_into_agrees_with_dequantize_and_reuses_its_buffer() {
        let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<u32>>();
        // Deliberately starts longer than any output, and full of a value no
        // format produces here: a leftover tail would survive into `buffer`.
        let mut buffer = vec![f32::from_bits(0xDEAD_BEEF); 4096];
        for &(ggml_type, name, count, _) in GOLDEN {
            let len = tensor_byte_size(ggml_type, count as u64).unwrap() as usize;
            let bytes = fixture_bytes(len);
            let expected = dequantize(ggml_type, &bytes, count).expect(name);
            dequantize_into(ggml_type, &bytes, count, &mut buffer).expect(name);
            assert_eq!(
                buffer.len(),
                count,
                "{name} left the buffer the wrong length"
            );
            assert_eq!(bits(&buffer), bits(&expected), "{name}");
        }
    }

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
        assert_eq!(cases.len(), 20, "fixture should cover 20 types");

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
        assert_eq!(tensor_byte_size(GGML_TYPE_I32, 8).unwrap(), 32);
        assert_eq!(tensor_byte_size(GGML_TYPE_Q8_0, 32).unwrap(), 34);
        assert_eq!(tensor_byte_size(GGML_TYPE_MXFP4, 32).unwrap(), 17);
        assert_eq!(tensor_byte_size(GGML_TYPE_Q4_0, 32).unwrap(), 18);
        assert_eq!(tensor_byte_size(GGML_TYPE_Q4_K, 256).unwrap(), 144);
        assert_eq!(tensor_byte_size(GGML_TYPE_Q5_K, 256).unwrap(), 176);
        assert_eq!(tensor_byte_size(GGML_TYPE_Q6_K, 256).unwrap(), 210);
        // The one `IQ*` type that blocks at 32, not 256 — same 18 bytes as
        // `Q4_0`. Pinned here because assuming `IQ* => QK_K` is exactly the
        // mistake that would misread every row of an `IQ4_NL` tensor.
        assert_eq!(tensor_byte_size(GGML_TYPE_IQ4_NL, 32).unwrap(), 18);
        assert!(tensor_byte_size(GGML_TYPE_IQ4_NL, 896).is_ok());
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

    #[test]
    fn dequantize_i32_widens_signed_integers() {
        let values = [1i32, -2, 4096];
        let mut bytes = Vec::new();
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let out = dequantize(GGML_TYPE_I32, &bytes, 3).unwrap();
        assert_eq!(out, vec![1.0, -2.0, 4096.0]);
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

    /// `IQ4_NL` looks exactly like `Q4_0` on the wire — `f16` scale, 16
    /// nibble-pair bytes — so the failure mode to guard against is decoding
    /// it *as* `Q4_0`. All-zero nibbles at `d=1.0` are the cleanest
    /// separator: the codebook says `-127.0`, the linear path would say
    /// `-8.0`. The ascending nibbles then check that the lookup is by index
    /// rather than a rescaled arithmetic sequence — `KVALUES_IQ4NL` is
    /// non-uniformly spaced, so its successive gaps differ.
    #[test]
    fn dequantize_iq4_nl_uses_the_codebook_not_a_linear_offset() {
        let mut block = Vec::new();
        block.extend_from_slice(&f16::from_f32(1.0).to_le_bytes());
        block.extend_from_slice(&[0u8; 16]);
        let out = dequantize(GGML_TYPE_IQ4_NL, &block, 32).unwrap();
        assert_eq!(out.len(), 32);
        assert!(out.iter().all(|&v| v == -127.0), "{out:?}");

        // Byte `j` holds low nibble `j` and high nibble `j`, so the low half
        // is levels 0..16 in order and the high half repeats them.
        let mut block = Vec::new();
        block.extend_from_slice(&f16::from_f32(1.0).to_le_bytes());
        block.extend((0..16u8).map(|j| j | (j << 4)));
        let out = dequantize(GGML_TYPE_IQ4_NL, &block, 32).unwrap();
        let want: Vec<f32> = KVALUES_IQ4NL.iter().map(|&v| v as f32).collect();
        assert_eq!(out[..16], want[..], "low nibbles");
        assert_eq!(out[16..], want[..], "high nibbles");
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

    #[test]
    fn dequantize_mxfp4_uses_e8m0_scale_and_fp4_codebook() {
        let mut block = Vec::new();
        block.push(128u8);
        block.extend((0..16u8).map(|j| j | (j << 4)));
        let out = dequantize(GGML_TYPE_MXFP4, &block, 32).unwrap();
        let want: Vec<f32> = KVALUES_MXFP4.iter().map(|&v| v as f32).collect();
        assert_eq!(out[..16], want[..], "low nibbles");
        assert_eq!(out[16..], want[..], "high nibbles");
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

    /// ggml's own `make_block_q4_0x4`/`x8` and `make_block_iq4_nlx4`/`x8`
    /// plus the `repack_*_bl` loop around them
    /// (`ggml/src/ggml-cpu/repack.cpp`), transcribed line for line — the
    /// *forward* direction, so it is genuine ground truth for the inverse
    /// rather than the inverse restated. All four differ only in row count,
    /// run length, and whether a run is XORed, which is exactly what
    /// `repack_layout` claims. `plain` is `out_dim * n_blocks` consecutive
    /// 18-byte blocks.
    fn ggml_repack(plain: &[u8], n_rows: usize, run: usize, xor: u8, in_dim: usize) -> Vec<u8> {
        let n_blocks = in_dim / QK4_0;
        let out_dim = plain.len() / (n_blocks * Q4_0_BLOCK_BYTES);
        let mut out = Vec::with_capacity(plain.len());
        for b in (0..out_dim).step_by(n_rows) {
            for x in 0..n_blocks {
                let src = |i: usize| &plain[((b + i) * n_blocks + x) * Q4_0_BLOCK_BYTES..];
                for i in 0..n_rows {
                    out.extend_from_slice(&src(i)[..2]);
                }
                // ggml: `end = QK4_0 * (2 or 4) / blck_size_interleave`,
                // `src_id = i % N`, `src_offset = (i / N) * S`.
                let end = (QK4_0 / 2) * n_rows / run;
                for i in 0..end {
                    let src_id = i % n_rows;
                    let src_offset = (i / n_rows) * run;
                    let qs = &src(src_id)[2 + src_offset..2 + src_offset + run];
                    out.extend(qs.iter().map(|b| b ^ xor));
                }
            }
        }
        out
    }

    /// Round-trips every repacked layout through ggml's packer: pack a
    /// known plain-`Q4_0` tensor, then de-interleave it back and require
    /// byte equality.
    ///
    /// Shapes are the real `SmolLM2-360M` ones (`960` and `2560` wide,
    /// `320`/`960`/`2560` rows) rather than a tidy single group, so the
    /// group stride is exercised across many groups — a de-interleave that
    /// got the row striding wrong would still pass on one group. This same
    /// round-trip was also run against the actual
    /// `bartowski/SmolLM2-360M-Instruct-GGUF` `Q4_0_4_4`/`Q4_0_4_8`/
    /// `Q4_0_8_8` files and matched byte-for-byte on every tensor.
    #[test]
    fn deinterleave_inverts_ggmls_own_repack() {
        for (ggml_type, base, n_rows, run, xor) in [
            (GGML_TYPE_Q4_0_4_4, GGML_TYPE_Q4_0, 4usize, 4usize, 0x88u8),
            (GGML_TYPE_Q4_0_4_8, GGML_TYPE_Q4_0, 4, 8, 0x88),
            (GGML_TYPE_Q4_0_8_8, GGML_TYPE_Q4_0, 8, 8, 0x88),
            (GGML_TYPE_IQ4_NL_4_4, GGML_TYPE_IQ4_NL, 4, 4, 0x00),
            (GGML_TYPE_IQ4_NL_4_8, GGML_TYPE_IQ4_NL, 4, 8, 0x00),
            (GGML_TYPE_IQ4_NL_8_8, GGML_TYPE_IQ4_NL, 8, 8, 0x00),
        ] {
            for (in_dim, out_dim) in [(960usize, 320usize), (960, 2560), (2560, 960)] {
                let n_bytes = out_dim * (in_dim / QK4_0) * Q4_0_BLOCK_BYTES;
                // Deterministic pseudo-random block bytes; the scale field
                // is left random too, since nothing here interprets it.
                let mut s = 0x2545_F491_4F6C_DD1Du64;
                let plain: Vec<u8> = (0..n_bytes)
                    .map(|_| {
                        s ^= s >> 12;
                        s ^= s << 25;
                        s ^= s >> 27;
                        (s >> 33) as u8
                    })
                    .collect();

                let packed = ggml_repack(&plain, n_rows, run, xor, in_dim);
                assert_eq!(packed.len(), plain.len(), "repack must not change size");
                let (got_base, back) =
                    deinterleave_repack(ggml_type, &packed, in_dim, out_dim).unwrap();
                assert_eq!(got_base, base, "{}", ggml_type_name(ggml_type));
                assert_eq!(
                    back,
                    plain,
                    "{} {in_dim}x{out_dim} did not round-trip",
                    ggml_type_name(ggml_type)
                );
            }
        }
    }

    /// Two things here read backwards and would both corrupt every weight
    /// silently. `*_4_8` interleaves **4** rows in 8-byte runs, not 8 rows
    /// — the digits are (rows) x (run). And the `IQ4_NL` family carries
    /// **no** XOR, unlike `Q4_0`'s `^ 0x88`: its nibble is a codebook
    /// index, so flipping the high bit would select a different level for
    /// every weight rather than adjusting a sign.
    #[test]
    fn repack_layouts_pin_row_count_and_xor_per_family() {
        assert_eq!(
            repack_layout(GGML_TYPE_Q4_0_4_4),
            Some((GGML_TYPE_Q4_0, 4, 4, 0x88))
        );
        assert_eq!(
            repack_layout(GGML_TYPE_Q4_0_4_8),
            Some((GGML_TYPE_Q4_0, 4, 8, 0x88))
        );
        assert_eq!(
            repack_layout(GGML_TYPE_Q4_0_8_8),
            Some((GGML_TYPE_Q4_0, 8, 8, 0x88))
        );
        assert_eq!(
            repack_layout(GGML_TYPE_IQ4_NL_4_4),
            Some((GGML_TYPE_IQ4_NL, 4, 4, 0x00))
        );
        assert_eq!(
            repack_layout(GGML_TYPE_IQ4_NL_4_8),
            Some((GGML_TYPE_IQ4_NL, 4, 8, 0x00))
        );
        assert_eq!(
            repack_layout(GGML_TYPE_IQ4_NL_8_8),
            Some((GGML_TYPE_IQ4_NL, 8, 8, 0x00))
        );
        assert_eq!(repack_layout(GGML_TYPE_Q4_0), None);
        assert_eq!(repack_layout(GGML_TYPE_IQ4_NL), None);

        // A tensor with 4 rows can only be packed by a 4-row layout, so the
        // `8_8` pair must refuse it rather than read past the group.
        let bytes = vec![0u8; 4 * (64 / QK4_0) * Q4_0_BLOCK_BYTES];
        for four in [GGML_TYPE_Q4_0_4_8, GGML_TYPE_IQ4_NL_4_8] {
            assert!(deinterleave_repack(four, &bytes, 64, 4).is_ok());
        }
        for eight in [GGML_TYPE_Q4_0_8_8, GGML_TYPE_IQ4_NL_8_8] {
            let err = deinterleave_repack(eight, &bytes, 64, 4).unwrap_err();
            assert!(
                err.to_string().contains("not a multiple of the 8 rows"),
                "{err}"
            );
        }
    }

    /// Repacked tensors carry exactly as many bytes as the `Q4_0` they were
    /// built from — the loader's `row_bytes * out_dim == len` check depends
    /// on it, and so does `deinterleave_repack`'s own length guard.
    #[test]
    fn repacked_types_size_exactly_like_q4_0() {
        for t in [
            GGML_TYPE_Q4_0_4_4,
            GGML_TYPE_Q4_0_4_8,
            GGML_TYPE_Q4_0_8_8,
            GGML_TYPE_IQ4_NL_4_4,
            GGML_TYPE_IQ4_NL_4_8,
            GGML_TYPE_IQ4_NL_8_8,
        ] {
            assert!(supports_type(t), "{}", ggml_type_name(t));
            assert_eq!(
                tensor_byte_size(t, 960).unwrap(),
                tensor_byte_size(GGML_TYPE_Q4_0, 960).unwrap()
            );
        }
    }

    /// A repacked row has no contiguous byte range of its own, so reading
    /// one through `dequantize` cannot be made to work — it has to fail
    /// loudly instead of returning bytes that decode into plausible noise.
    #[test]
    fn dequantize_refuses_a_repacked_row_instead_of_misreading_it() {
        let err = dequantize(GGML_TYPE_Q4_0_4_4, &[0u8; 18], 32).unwrap_err();
        assert!(err.to_string().contains("de-interleaved"), "{err}");
    }

    #[test]
    fn dequantize_rejects_unsupported_types() {
        let err = dequantize(99, &[], 0).unwrap_err();
        assert!(err.to_string().contains("not yet supported"));
    }

    /// For a type ggml removed *and* this build can't read, the message
    /// must say the removal was ggml's rather than this build's;
    /// "not yet supported" would imply waiting is the fix. `Q4_2`/`Q4_3`
    /// are all that is left in that category now that both repack families
    /// are read by de-interleaving.
    #[test]
    fn removed_ggml_types_explain_themselves_rather_than_reading_as_unimplemented() {
        for (ggml_type, name) in [(4u32, "Q4_2"), (5, "Q4_3")] {
            let err = dequantize(ggml_type, &[], 0).unwrap_err().to_string();
            assert!(err.contains(name), "{err}");
            assert!(err.contains("removed from ggml itself"), "{err}");
            assert!(!err.contains("not yet supported"), "{err}");
            // `tensor_byte_size` is the other door into the same rejection —
            // it is what `loader` calls first — so it must say the same thing.
            let sized = tensor_byte_size(ggml_type, 32).unwrap_err().to_string();
            assert!(sized.contains("removed from ggml itself"), "{sized}");
        }
        // And a type that is merely unimplemented must NOT borrow that
        // wording, or it would tell the reader to give up on a gap that
        // could genuinely be filled.
        let err = dequantize(99, &[], 0).unwrap_err().to_string();
        assert!(err.contains("not yet supported"), "{err}");
        assert!(!err.contains("removed from ggml"), "{err}");
    }
}
