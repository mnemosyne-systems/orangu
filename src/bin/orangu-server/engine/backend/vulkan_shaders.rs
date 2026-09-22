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

//! WGSL compute shaders — one per supported `ggml_type` — that dequantize a
//! weight matrix's raw, still-quantized bytes and dot-product it against the
//! input activations directly on the GPU. Each type's dequantization math
//! (`dequant_element`, in each `*_COOP_MIDDLE` constant) is a line-for-line
//! port of its `engine::quant::dequantize_*` Rust counterpart (which itself
//! mirrors ggml's `dequantize_row_*` exactly) — read them side by side when
//! changing either.
//!
//! Two dispatch strategies share those same per-type `dequant_element`
//! functions:
//!
//! - **`MAIN_REDUCE_SUFFIX`** (small `n_tokens`, e.g. decode's `n_tokens ==
//!   1`, `VulkanBackend::COOP_MIN_N_TOKENS`): one workgroup per `(row,
//!   token)` pair, all 64 threads splitting that *row's own elements*
//!   (`k`, `k+64`, `k+128`, ...) and reducing their partial dot-product
//!   sums together. Adjacent threads read *adjacent elements of the same
//!   row*, so a wavefront's reads over the row are contiguous.
//! - **`MAIN_COOP_SUFFIX`** (large `n_tokens`, e.g. a long prompt's
//!   prefill): one workgroup per output row, looping over *all* tokens —
//!   dequantizes each block once into shared memory and reuses it across
//!   up to 64 tokens' worth of threads, avoiding the redundant
//!   re-dequantizing `MAIN_REDUCE_SUFFIX` would otherwise do once per
//!   `(row, token)` pair when many tokens genuinely share the same row.
//!
//! `PRELUDE` (buffer bindings + byte/half-float decode helpers) is shared
//! by both, concatenated with a type's `*_COOP_MIDDLE` (see [`coop_middle`])
//! and the relevant `MAIN_*_SUFFIX` once at `VulkanBackend` construction
//! time into one complete, self-contained WGSL module per (type, dispatch
//! strategy) pair. The `IQ*` types additionally get [`IQ_GRID_PRELUDE`],
//! which declares the codebook buffer their `dequant_element`s look their
//! lattice points up in.

use crate::engine::quant::{
    GGML_TYPE_BF16, GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_IQ1_M, GGML_TYPE_IQ1_S,
    GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ3_XXS,
    GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS, GGML_TYPE_MXFP4, GGML_TYPE_PQ2_0, GGML_TYPE_PTQ1_0,
    GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q4_K, GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0,
};

/// Storage/uniform bindings every shader shares, plus byte- and half-float-
/// decode helpers. Storage buffers only accept 4-byte-aligned element
/// types in WGSL, so `weights` is read as `array<u32>` and `read_u8` peels
/// individual bytes out of it — a block's byte size is rarely a multiple of
/// 4 (`Q4_0` is 18 bytes, `Q6_K` is 210), so per-byte reads sidestep
/// alignment entirely rather than requiring one.
const PRELUDE: &str = r#"
struct Meta {
    in_dim: u32,
    out_dim: u32,
    n_tokens: u32,
    row_bytes: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> params: Meta;

fn read_u8(byte_offset: u32) -> u32 {
    let word = weights[byte_offset >> 2u];
    let shift = (byte_offset & 3u) * 8u;
    return (word >> shift) & 0xFFu;
}

// Four consecutive bytes as one `u32`, at any alignment. A block layout is
// rarely a multiple of 4 (`Q8_0` is 34 bytes, `Q4_0` 18, `Q6_K` 210), so a
// run of bytes inside a row straddles a word boundary more often than not
// and `read_u8` is the only reader that needs no alignment at all. That
// costs a load per byte, and a kernel reading four bytes a lane spends four
// loads where the data occupies at most two words.
//
// Both words are read unconditionally rather than under a branch: `shift`
// is uniform across the lanes sharing a row (they differ by whole words),
// so the branch would not diverge, but the second read is in cache either
// way and a `select` keeps the generated code straight-line. The `& 31u`
// only exists to keep the shift count in range when `shift` is zero, where
// the result is discarded.
fn read_u32_at(byte_offset: u32) -> u32 {
    let index = byte_offset >> 2u;
    let shift = (byte_offset & 3u) * 8u;
    let lo = weights[index] >> shift;
    let hi = weights[index + 1u] << ((32u - shift) & 31u);
    return lo | select(hi, 0u, shift == 0u);
}

// Two consecutive bytes as one `u32` — a block's `f16` scale, in one load
// when it does not straddle a word and two when it does.
fn read_u16_at(byte_offset: u32) -> u32 {
    return read_u32_at(byte_offset) & 0xFFFFu;
}

// IEEE 754 binary16 -> f32: `unpack2x16float` is a core WGSL builtin that
// does the exact conversion in hardware, so this delegates to it rather
// than hand-rolling the exponent/mantissa math — bit-for-bit the same
// result as `half::f16::to_f32` for the same bits.
fn f16_to_f32(bits: u32) -> f32 {
    return unpack2x16float(bits & 0xFFFFu).x;
}

// bfloat16 -> f32: the top 16 bits of an f32, left-shifted into place —
// mirrors `quant::dequantize`'s `GGML_TYPE_BF16` arm exactly.
fn bf16_to_f32(bits: u32) -> f32 {
    return bitcast<f32>((bits & 0xFFFFu) << 16u);
}

// ggml's `get_scale_min_k4`: unpacks the 6-bit scale and 6-bit min for
// sub-block `j` (0..8) of a Q4_K/Q5_K super-block's 12-byte `scales` region
// starting at `base`. Mirrors `quant::get_scale_min_k4` exactly.
fn get_scale_min_k4(base: u32, j: u32) -> vec2<u32> {
    if (j < 4u) {
        let qj = read_u8(base + j);
        let qj4 = read_u8(base + j + 4u);
        return vec2<u32>(qj & 63u, qj4 & 63u);
    }
    let qj = read_u8(base + j);
    let qj4 = read_u8(base + j + 4u);
    let qjm4 = read_u8(base + j - 4u);
    let sc = (qj4 & 0xFu) | ((qjm4 >> 6u) << 4u);
    let m = (qj4 >> 4u) | ((qj >> 6u) << 4u);
    return vec2<u32>(sc, m);
}

// Q3_K's `i`th 6-bit sub-block scale (0..16), still biased by 32, out of the
// 12 bytes at `base`. Mirrors `quant::unpack_q3_k_scales` for one index.
fn q3k_scale(base: u32, i: u32) -> u32 {
    var low: u32;
    if (i < 8u) {
        low = read_u8(base + i) & 0xFu;
    } else {
        low = read_u8(base + i - 8u) >> 4u;
    }
    let high = (read_u8(base + 8u + (i % 4u)) >> (2u * (i / 4u))) & 3u;
    return low | (high << 4u);
}
"#;

/// The `IQ*` codebooks, appended to [`PRELUDE`] for the `IQ*` types only.
///
/// A K-quant block stores its weights; an `IQ*` block stores *indices* into
/// a fixed codebook of lattice points. That codebook cannot be computed —
/// it has to be read — so it is uploaded once at
/// `VulkanBackend::try_init` as one ~15 KiB storage buffer
/// (`VulkanBackend::iq_grid_buffer`, packed by `iq_grid_words`) and bound at
/// `@binding(4)` for every matmul pipeline. Only these shaders declare it;
/// the other types' modules never mention the binding, which WGSL permits
/// (a bind group layout may carry entries a shader does not use — the
/// reverse is what is rejected).
///
/// Uploading beats baking the numbers into the WGSL text: a module-scope
/// `array` initializer indexed by a runtime value is not a uniform-control-
/// flow constant, so RADV would place a per-invocation copy in scratch and
/// every lookup would be a scratch load. A read-only storage buffer is
/// cached like any other weight read.
///
/// The `u32` offsets below must stay in step with `iq_grid_words`.
/// `MXFP4`'s scale decode and codebook, appended to [`PRELUDE`] for that one
/// type by [`prelude_for`] — the same arrangement [`IQ_GRID_PRELUDE`] gets,
/// and for the same reason: both middles need it, neither should carry its
/// own copy of a table, and no other type's module should declare it.
///
/// **No binding.** The `IQ*` codebooks are large enough to live in a storage
/// buffer; this one is sixteen values and is folded into arithmetic instead,
/// so an `MXFP4` shader declares exactly the four bindings [`PRELUDE`] does
/// and `needs_iq_grids` stays false for it.
const MXFP4_PRELUDE: &str = r#"
// `quant::e8m0_to_fp32_half`, transcribed. A one-byte exponent, and the
// "half" is not a rounding step: `KVALUES_MXFP4` holds the codebook at twice
// its true value (integers, so the table stays exact), and biasing the
// exponent down by one is what divides it back. `x < 2` is the subnormal
// tail the shift would otherwise underflow.
fn mxfp4_scale(e: u32) -> f32 {
    var bits: u32;
    if (e < 2u) {
        bits = 0x00200000u << e;
    } else {
        bits = (e - 1u) << 23u;
    }
    return bitcast<f32>(bits);
}

// `quant::KVALUES_MXFP4[i]` = [0,1,2,3,4,6,8,12] for `i < 8`, negated above.
// The magnitudes are packed one per nibble of a single `u32` — every value
// fits in 4 bits — rather than read from a buffer, which is what keeps this
// type off the `iq_grids` binding.
//
// Index 8 yields `-0.0` where the table has `+0.0`. That cannot change a
// result: the products differ only in the sign of a zero, and `+0.0 +
// -0.0` is `+0.0`, so every accumulation reaching here is bit-identical to
// the host path's.
fn mxfp4_kvalue(i: u32) -> f32 {
    let mag = f32((0xC8643210u >> ((i & 7u) * 4u)) & 0xFu);
    if ((i & 8u) != 0u) {
        return -mag;
    }
    return mag;
}
"#;

const IQ_GRID_PRELUDE: &str = r#"
@group(0) @binding(4) var<storage, read> iq_grids: array<u32>;

const IQ2XS_GRID_OFF: u32 = 0u;
const IQ2S_GRID_OFF: u32 = 1024u;
const IQ3XXS_GRID_OFF: u32 = 3072u;
const IQ3S_GRID_OFF: u32 = 3328u;
const KSIGNS_OFF: u32 = 3840u;
const KVALUES_IQ4NL_OFF: u32 = 3872u;
const IQ2XXS_GRID_OFF: u32 = 3876u;
const IQ1S_GRID_OFF: u32 = 4388u;

// Byte `j` (0..8) of the 8-element lattice point `idx` in an `iq2*` grid,
// which stores two `u32` words per entry.
fn iq_grid8(base: u32, idx: u32, j: u32) -> u32 {
    let word = iq_grids[base + idx * 2u + (j >> 2u)];
    return (word >> ((j & 3u) * 8u)) & 0xFFu;
}

// Byte `j` (0..4) of the 4-element lattice point `idx` in an `iq3*` grid,
// one `u32` word per entry.
fn iq_grid4(base: u32, idx: u32, j: u32) -> u32 {
    let word = iq_grids[base + idx];
    return (word >> ((j & 3u) * 8u)) & 0xFFu;
}

// `ksigns_iq2xs[i]`: the 8 sign bits a 7-bit sign field expands to.
fn iq_ksigns(i: u32) -> u32 {
    return (iq_grids[KSIGNS_OFF + (i >> 2u)] >> ((i & 3u) * 8u)) & 0xFFu;
}

// `kvalues_iq4nl[i]`, sign-extended from the `int8_t` it is stored as.
fn iq_kvalue(i: u32) -> f32 {
    let b = (iq_grids[KVALUES_IQ4NL_OFF + (i >> 2u)] >> ((i & 3u) * 8u)) & 0xFFu;
    var v: i32 = i32(b);
    if (v >= 128) {
        v = v - 256;
    }
    return f32(v);
}

// Byte `j` (0..8) of an `iq1*` lattice point, sign-extended from the
// `int8_t` it is stored as. The `iq1*` grids carry **signed** values and no
// sign field at all, unlike every `iq2*`/`iq3*` grid above — see
// `quant::push_iq1_grid`.
fn iq_grid8_signed(base: u32, idx: u32, j: u32) -> f32 {
    let b = iq_grid8(base, idx, j);
    var v: i32 = i32(b);
    if (v >= 128) {
        v = v - 256;
    }
    return f32(v);
}

// `IQ1S_DELTA`/`IQ1M_DELTA` — the `±` offset every `iq1*` weight carries on
// top of its codebook value.
const IQ1_DELTA: f32 = 0.125;

// `kmask_iq2xs[j]` is `1 << j`, so the sign of element `j` is bit `j`.
fn iq_sign(signs: u32, j: u32) -> f32 {
    if ((signs & (1u << j)) != 0u) {
        return -1.0;
    }
    return 1.0;
}
"#;

/// Generates the shared final-combine block both `main_reduce_suffix` and
/// `unroll_suffix` use: `n_rows` independent 64-wide reductions of
/// `partial0..partial{n_rows-1}` into `y`, either the classic six-round
/// `workgroupBarrier` pairwise tree or (`subgroup: true`) the
/// `subgroupAdd`-based combine. Not hardcoded to a 64-wide subgroup:
/// `subgroupAdd`/`subgroupMax` first collapse each lane's contribution down
/// to one partial sum *per subgroup* (broadcast to every lane in that
/// subgroup), each subgroup's lane 0 writes that partial into
/// `partial_sums`, one `workgroupBarrier` makes every subgroup's partial
/// visible, and then (only) `local == 0u` sums the (small, `num_subgroups`-
/// many, ≤64) partials sequentially before writing `y`. On hardware where
/// the subgroup spans the whole 64-thread workgroup, `num_subgroups == 1`
/// and that final loop runs exactly once. On hardware with a narrower
/// subgroup this degrades gracefully to a couple of barriers and a short
/// sequential combine instead of silently returning the wrong sum —
/// deliberately not assuming subgroup size == workgroup size, since getting
/// that wrong would be a silent correctness bug this project's own
/// bit-for-bit cross-check discipline doesn't allow. (Measured as a real
/// end-to-end regression despite fewer barriers — see
/// `VulkanBackend::try_init` for why it ships opt-in, not default.)
/// Row count used to be a hardcoded `4` baked separately into the WGSL text
/// and the Rust-side dispatch-count math (`VulkanBackend::REDUCE_N_ROWS`),
/// two places that had to be changed together by hand; generating both from
/// the same `n_rows` here removes that footgun.
fn reduce_combine_block(n_rows: usize, subgroup: bool) -> String {
    let mut s = String::new();
    if subgroup {
        for i in 0..n_rows {
            s.push_str(&format!("    let sg{i} = subgroupAdd(partial{i});\n"));
        }
        s.push_str("    if (sg_lane == 0u) {\n");
        for i in 0..n_rows {
            s.push_str(&format!(
                "        partial_sums[{i}u * 64u + sg_id] = sg{i};\n"
            ));
        }
        s.push_str("    }\n    workgroupBarrier();\n    if (local == 0u) {\n");
        for i in 0..n_rows {
            s.push_str(&format!("        var t{i}: f32 = 0.0;\n"));
        }
        s.push_str("        var i: u32 = 0u;\n        loop {\n            if (i >= n_sg) {\n                break;\n            }\n");
        for i in 0..n_rows {
            s.push_str(&format!(
                "            t{i} = t{i} + partial_sums[{i}u * 64u + i];\n"
            ));
        }
        s.push_str("            i = i + 1u;\n        }\n");
        s.push_str("        y[t * params.out_dim + o0] = t0;\n");
        for i in 1..n_rows {
            s.push_str(&format!(
                "        if (o{i} < params.out_dim) {{\n            y[t * params.out_dim + o{i}] = t{i};\n        }}\n"
            ));
        }
        s.push_str("    }\n");
    } else {
        for i in 0..n_rows {
            s.push_str(&format!(
                "    partial_sums[{i}u * 64u + local] = partial{i};\n"
            ));
        }
        s.push_str("    workgroupBarrier();\n    var stride: u32 = 32u;\n    loop {\n        if (stride == 0u) {\n            break;\n        }\n        if (local < stride) {\n");
        for i in 0..n_rows {
            s.push_str(&format!(
                "            partial_sums[{i}u * 64u + local] = partial_sums[{i}u * 64u + local] + partial_sums[{i}u * 64u + local + stride];\n"
            ));
        }
        s.push_str(
            "        }\n        workgroupBarrier();\n        stride = stride / 2u;\n    }\n",
        );
        s.push_str("    if (local == 0u) {\n");
        s.push_str("        y[t * params.out_dim + o0] = partial_sums[0];\n");
        for i in 1..n_rows {
            s.push_str(&format!(
                "        if (o{i} < params.out_dim) {{\n            y[t * params.out_dim + o{i}] = partial_sums[{i}u * 64u];\n        }}\n"
            ));
        }
        s.push_str("    }\n");
    }
    s
}

/// The `@compute fn main` entry-point parameter list's subgroup-only
/// builtins — see `reduce_combine_block`'s own doc comment.
fn subgroup_entry_params(subgroup: bool) -> &'static str {
    if subgroup {
        "\n    @builtin(subgroup_invocation_id) sg_lane: u32,\n    @builtin(subgroup_id) sg_id: u32,\n    @builtin(num_subgroups) n_sg: u32,"
    } else {
        ""
    }
}

/// The compute entry point for the *reduction* path (small `n_tokens`,
/// e.g. decode's `n_tokens == 1` — see `VulkanBackend::COOP_MIN_N_TOKENS`
/// for the crossover into `MAIN_COOP_SUFFIX` instead), generated for an
/// arbitrary `n_rows` (rows-per-workgroup) — see [`reduce_combine_block`]'s
/// own doc comment for the combine step. One workgroup per `(output row
/// *group* of `n_rows` rows, token)` pair, not one row: all 64 threads
/// divide up `in_dim` elements the same grid-stride way a single-row design
/// would (`k = local, local + 64, local + 128, ...`), but at each `k` read
/// `x[x_base + k]` *once* and reuse it across all `n_rows` rows' dot
/// products — "multiple output rows per thread." Adjacent threads read
/// adjacent elements of the *same* row at every step, so a wavefront's
/// reads over the row are contiguous. The last group in a row an
/// `n_rows`-imperfect `out_dim` (e.g. `out_dim = 6`, `n_rows = 4` needs 2
/// groups, the second only half full) simply skips the out-of-range rows
/// via `o < params.out_dim` bounds checks — their `partial_sums` entries
/// are computed as `0.0` and never written to `y`, not read back by
/// anything. `VulkanBackend::build_op_resources` dispatches
/// `ceil(out_dim / n_rows) * n_tokens` workgroups using this same `n_rows`
/// value, so the two can no longer drift out of sync the way two separately
/// hardcoded `4`s used to risk.
fn main_reduce_suffix(n_rows: usize, subgroup: bool) -> String {
    let mut s = format!(
        "var<workgroup> partial_sums: array<f32, {}>;\n\n",
        n_rows * 64
    );
    s.push_str("@compute @workgroup_size(64)\nfn main(\n    @builtin(workgroup_id) wid: vec3<u32>,\n    @builtin(local_invocation_id) lid: vec3<u32>,\n    @builtin(num_workgroups) nwg: vec3<u32>,");
    s.push_str(subgroup_entry_params(subgroup));
    s.push_str("\n) {\n");
    s.push_str(&format!(
        "    let n_row_groups = (params.out_dim + {}u) / {n_rows}u;\n",
        n_rows - 1
    ));
    s.push_str("    let flat = wid.x + wid.y * nwg.x + wid.z * nwg.x * nwg.y;\n    if (flat >= n_row_groups * params.n_tokens) {\n        return;\n    }\n");
    s.push_str("    let rg = flat / params.n_tokens;\n    let t = flat % params.n_tokens;\n");
    s.push_str(&format!("    let o_base = rg * {n_rows}u;\n"));
    for i in 0..n_rows {
        s.push_str(&format!("    let o{i} = o_base + {i}u;\n"));
    }
    s.push_str("    let local = lid.x;\n    let x_base = t * params.in_dim;\n\n");
    for i in 0..n_rows {
        s.push_str(&format!("    var partial{i}: f32 = 0.0;\n"));
    }
    s.push_str("    var k: u32 = local;\n    loop {\n        if (k >= params.in_dim) {\n            break;\n        }\n");
    s.push_str("        let block_idx = k / BLOCK_ELEMS;\n        let local_k = k % BLOCK_ELEMS;\n        let block_off = block_idx * BLOCK_BYTES;\n        let xv = x[x_base + k];\n");
    s.push_str(
        "        partial0 = partial0 + dequant_element(o0 * params.row_bytes + block_off, local_k) * xv;\n",
    );
    for i in 1..n_rows {
        s.push_str(&format!(
            "        if (o{i} < params.out_dim) {{\n            partial{i} = partial{i} + dequant_element(o{i} * params.row_bytes + block_off, local_k) * xv;\n        }}\n"
        ));
    }
    s.push_str("        k = k + 64u;\n    }\n\n");
    s.push_str(&reduce_combine_block(n_rows, subgroup));
    s.push_str("}\n");
    s
}

/// The **block-hoisted reduce** `main`, generic over *any* block size — the
/// third algorithm written against the per-type contract, alongside
/// [`main_reduce_suffix`] and [`unroll_suffix`].
///
/// It exists because the other two do not cover the same set.
/// `main_reduce_suffix` works for every type and is slow; `unroll_suffix` is
/// fast and works only for the K-quants, because it hardcodes their 256-element
/// super-block as a fixed 4×64 geometry (`x0..x3` at `local`, `64+local`,
/// `128+local`, `192+local`). A 32-element block cannot be expressed in that
/// shape, so every legacy and `IQ` type fell back to the slow one.
///
/// **What makes the slow one slow is not bandwidth, it is the block header.**
/// `main_reduce_suffix` calls `dequant_element` once per *element*, and every
/// such call re-reads the block's scale bytes and re-runs `f16_to_f32` — for a
/// 32-element block, 32 times over. Measured across four quantizations of one
/// model, the types with a hoisted kernel stream at ~82 GiB/s and the types
/// without at ~26.
///
/// The fix needs no new geometry, only a different work assignment: **one lane
/// owns a whole block** and strides over blocks, so the header is decoded once
/// per block instead of once per element. The per-type contract shrinks
/// accordingly, from `dequant_element(byte_offset, k)` to a single
/// `block_dot(byte_offset, x_off) -> f32` — this block's whole contribution to
/// one output row, given where its bytes start and where its `BLOCK_ELEMS`
/// activations start. That contract says nothing about block size, scale
/// layout or bit packing, which is what makes it work for every type rather
/// than three.
///
/// Activations are read contiguously within a lane and strided across lanes,
/// the transpose of the element-wise path's access. That is the one thing
/// given up here, and it is cheap: `x` is a single `[in_dim]` row that stays
/// in cache, while `weights` — the operand that actually has to stream — keeps
/// the same per-lane contiguous run it always had.
/// Blocks a lane takes per trip round the block loop of
/// [`block_hoisted_suffix`] — the loads of all of them go out together, so
/// a row's dependent memory round trips fall by this factor.
/// `ORANGU_BLOCKS_PER_TRIP` (1, 2 or 4) pins it for measurement; the
/// default is one. Two halved the isolated probe's time on the FFN gate
/// shape and lost 28% on the same op inside a decode step (four lost
/// 130%): a kernel measured over its own hot weights is not the kernel
/// streaming cold ones behind thirty other dispatches, and the decode step
/// is the measurement that counts.
fn blocks_per_trip() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("ORANGU_BLOCKS_PER_TRIP")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|n| [1, 2, 4].contains(n))
            .unwrap_or(1)
    })
}

fn block_hoisted_suffix(n_rows: usize, subgroup: bool) -> String {
    let mut s = format!(
        "var<workgroup> partial_sums: array<f32, {}>;\n\n",
        n_rows * 64
    );
    s.push_str("@compute @workgroup_size(64)\nfn main(\n    @builtin(workgroup_id) wid: vec3<u32>,\n    @builtin(local_invocation_id) lid: vec3<u32>,\n    @builtin(num_workgroups) nwg: vec3<u32>,");
    s.push_str(subgroup_entry_params(subgroup));
    s.push_str("\n) {\n");
    s.push_str(&format!(
        "    let n_row_groups = (params.out_dim + {}u) / {n_rows}u;\n",
        n_rows - 1
    ));
    s.push_str("    let flat = wid.x + wid.y * nwg.x + wid.z * nwg.x * nwg.y;\n    if (flat >= n_row_groups * params.n_tokens) {\n        return;\n    }\n");
    s.push_str("    let rg = flat / params.n_tokens;\n    let t = flat % params.n_tokens;\n");
    s.push_str(&format!("    let o_base = rg * {n_rows}u;\n"));
    for i in 0..n_rows {
        s.push_str(&format!("    let o{i} = o_base + {i}u;\n"));
    }
    s.push_str("    let local = lid.x;\n    let x_base = t * params.in_dim;\n\n");
    for i in 0..n_rows {
        s.push_str(&format!("    var partial{i}: f32 = 0.0;\n"));
    }
    // Whole blocks only. A row whose `in_dim` is not a multiple of
    // `BLOCK_ELEMS` cannot exist — a quantized tensor is stored in whole
    // blocks — so there is no tail to handle, unlike the element-wise path
    // whose stride is the workgroup rather than the block.
    // `LANES_PER_BLOCK` **adjacent** lanes share one block, each taking a
    // contiguous slice of its bytes, so those lanes issue one coalesced burst
    // across the row instead of scattering.
    //
    // Giving a lane a whole block to itself was the first thing tried here and
    // it is **34% slower than the element-wise path it was meant to beat**
    // (5.76 against 8.73 tok/s on a `Q8_0` model, reproduced): sixty-four lanes
    // then read sixty-four *different* byte runs at once, and losing coalescing
    // costs far more than decoding a block header once per element saves. The
    // engine sat at 99% busy with memory at 9% — the signature of a kernel
    // waiting on scattered reads rather than streaming.
    //
    // The shape here is instead the one the *fast* K-quant kernel already uses:
    // `shader_source_reduce_q4k_light` splits its 32-lane workgroup as
    // `itid = tid % 16`, several lanes to a super-block. Header decoding is
    // still amortized — `LANES_PER_BLOCK` times per block instead of
    // `BLOCK_ELEMS` — but not at the cost of the access pattern.
    s.push_str("\n    let sub = local % LANES_PER_BLOCK;\n    let slot = local / LANES_PER_BLOCK;\n    let blocks_in_flight = 64u / LANES_PER_BLOCK;\n");
    // Every row and every block of a trip is computed unconditionally —
    // a row past `out_dim` on a clamped row (its result is never stored),
    // a block past the row on a clamped block with its contribution
    // masked to zero — so the trip's loads for all of them sit in one
    // basic block and go out together. Under a per-row `if`, each row's
    // loads waited for the row before it: a trip of four rows was four
    // memory latencies, not one.
    for i in 0..n_rows {
        s.push_str(&format!(
            "    let oc{i} = min(o{i}, params.out_dim - 1u);\n"
        ));
    }
    let trip = blocks_per_trip();
    s.push_str("    let n_blocks = params.in_dim / BLOCK_ELEMS;\n    var b: u32 = slot;\n    loop {\n        if (b >= n_blocks) {\n            break;\n        }\n");
    for u in 0..trip {
        s.push_str(&format!(
            "        let bu{u} = min(b + {u}u * blocks_in_flight, n_blocks - 1u);\n        let m{u} = select(0.0, 1.0, b + {u}u * blocks_in_flight < n_blocks);\n        let block_off{u} = bu{u} * BLOCK_BYTES;\n        let x_off{u} = x_base + bu{u} * BLOCK_ELEMS;\n"
        ));
        for i in 0..n_rows {
            s.push_str(&format!(
                "        partial{i} = partial{i} + m{u} * block_dot(oc{i} * params.row_bytes + block_off{u}, x_off{u}, sub);\n"
            ));
        }
    }
    s.push_str(&format!(
        "        b = b + {trip}u * blocks_in_flight;\n    }}\n\n"
    ));
    s.push_str(&reduce_combine_block(n_rows, subgroup));
    s.push_str("}\n");
    s
}

/// The block-unroll `main` shared by every block-unroll kernel (`Q4_K`/
/// `Q5_K`/`Q6_K`, scalar and packed-`f16`) for an arbitrary `n_rows` — see
/// [`main_reduce_suffix`]'s own doc comment for the `n_rows` generalization
/// itself. Each type's `*_UNROLL_MIDDLE` supplies its own `BLOCK_BYTES`/
/// `BLOCK_ELEMS` and a single uniform entry point `block_dot(byte_offset,
/// local, x0, x1, x2, x3) -> f32` — this thread's contribution to one
/// output row from one 256-element super-block, given the block's byte
/// offset, this lane's id, and the four activations for the four 64-groups
/// (positions `local`, `64+local`, `128+local`, `192+local`). `block_dot`'s
/// signature is untouched by `n_rows`: those four `x0..x3` activations come
/// from the K-quant super-block's fixed 4×64 internal geometry (a different
/// axis from how many *output rows* share a workgroup — element `g` of this
/// lane always lives at position `g*64 + local`, the same for every type,
/// which is why the activation gather here is identical across types; only
/// `block_dot`'s own dequant-and-dot differs), so generalizing `n_rows`
/// only changes how many times `block_dot` is called per block (once per
/// output row this workgroup handles), issuing its **four activation loads
/// up front** each block, before the dependent dots — the memory-level-
/// parallelism restructuring this kernel exists for: several independent
/// loads outstanding per lane per block, instead of the plain reduce path's
/// one outstanding load at a time.
fn unroll_suffix(n_rows: usize, subgroup: bool) -> String {
    let mut s = format!(
        "var<workgroup> partial_sums: array<f32, {}>;\n\n",
        n_rows * 64
    );
    s.push_str("@compute @workgroup_size(64)\nfn main(\n    @builtin(workgroup_id) wid: vec3<u32>,\n    @builtin(local_invocation_id) lid: vec3<u32>,\n    @builtin(num_workgroups) nwg: vec3<u32>,");
    s.push_str(subgroup_entry_params(subgroup));
    s.push_str("\n) {\n");
    s.push_str(&format!(
        "    let n_row_groups = (params.out_dim + {}u) / {n_rows}u;\n",
        n_rows - 1
    ));
    s.push_str("    let flat = wid.x + wid.y * nwg.x + wid.z * nwg.x * nwg.y;\n    if (flat >= n_row_groups * params.n_tokens) {\n        return;\n    }\n");
    s.push_str("    let rg = flat / params.n_tokens;\n    let t = flat % params.n_tokens;\n");
    s.push_str(&format!("    let o0 = rg * {n_rows}u;\n"));
    for i in 1..n_rows {
        s.push_str(&format!("    let o{i} = o0 + {i}u;\n"));
    }
    s.push_str("    let local = lid.x;\n    let x_base = t * params.in_dim;\n\n");
    for i in 0..n_rows {
        s.push_str(&format!("    var partial{i}: f32 = 0.0;\n"));
    }
    s.push_str("\n    let n_blocks = params.in_dim / BLOCK_ELEMS;\n    var b: u32 = 0u;\n    loop {\n        if (b >= n_blocks) {\n            break;\n        }\n");
    s.push_str(
        "        let block_off = b * BLOCK_BYTES;\n        let x_blk = x_base + b * BLOCK_ELEMS;\n",
    );
    s.push_str("        let x0 = x[x_blk + local];\n        let x1 = x[x_blk + 64u + local];\n        let x2 = x[x_blk + 128u + local];\n        let x3 = x[x_blk + 192u + local];\n");
    s.push_str(
        "        partial0 = partial0 + block_dot(o0 * params.row_bytes + block_off, local, x0, x1, x2, x3);\n",
    );
    for i in 1..n_rows {
        s.push_str(&format!(
            "        if (o{i} < params.out_dim) {{\n            partial{i} = partial{i} + block_dot(o{i} * params.row_bytes + block_off, local, x0, x1, x2, x3);\n        }}\n"
        ));
    }
    s.push_str("        b = b + 1u;\n    }\n\n");
    s.push_str(&reduce_combine_block(n_rows, subgroup));
    s.push_str("}\n");
    s
}

/// The compute entry point for the *cooperative* path — used instead of
/// `MAIN_REDUCE_SUFFIX` when `n_tokens` is large enough (see `VulkanBackend`'s
/// dispatch selection) that many tokens genuinely share the same weight
/// row's blocks. One workgroup per output row (not per `(row, token)`
/// pair): every thread cooperatively dequantizes its own slice of each
/// block into `shared_vals` (`var<workgroup>`, on-chip shared
/// memory — not the per-thread `array<f32, BLOCK_ELEMS>` the
/// non-cooperative path deliberately avoids, a different physical
/// resource with none of that spilling risk) via `dequant_element`
/// (type-specific, computes one output index directly rather than
/// filling a whole block sequentially — see each `*_COOP_MIDDLE`), then a
/// `workgroupBarrier` lets every thread read the *whole* block back to
/// accumulate *its own* token's dot product. Splitting the dequant work
/// this way (each of the 64 threads computes `BLOCK_ELEMS / 64` elements,
/// or for `BLOCK_ELEMS < 64` only the first `BLOCK_ELEMS` threads do
/// anything) is what makes this actually cooperative rather than just
/// having one thread do all the work while the other 63 wait on it — the
/// block is still dequantized once and shared, not redone per token, but
/// now the *dequantizing itself* is parallel too. Tokens beyond the first
/// 64 are handled by looping in tiles of 64 (one thread per token per
/// tile); `n_tokens`/`in_dim` are uniform-buffer values, so every thread
/// in the workgroup reaches every `workgroupBarrier` together, as WGSL
/// requires — the strided `dequant_element` loop below it varies its own
/// iteration count per thread, which is fine precisely because it has no
/// barrier of its own inside it.
///
/// This never reuses activations across output rows — every one of
/// `out_dim` per-row workgroups independently re-reads the entire
/// activation matrix from global memory, *and* its own per-workgroup
/// `tile_start` loop above runs the *entire* `n_tokens` range
/// sequentially, with no upper bound on prompt length — which is what
/// `Self::shader_source_coop_tiled` addresses (bounded, fixed-size tiles
/// instead, so per-workgroup GPU time no longer grows unboundedly with
/// prompt length). That kernel is now the default (opt out with
/// `ORANGU_NO_TILED_PREFILL=1`) — see `VulkanBackend::tiled_prefill`'s
/// own doc comment for why.
const MAIN_COOP_SUFFIX: &str = r#"
@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let o = wid.x + wid.y * nwg.x + wid.z * nwg.x * nwg.y;
    if (o >= params.out_dim) {
        return;
    }
    let local = lid.x;
    let row_byte_base = o * params.row_bytes;
    let n_blocks = params.in_dim / BLOCK_ELEMS;

    var tile_start: u32 = 0u;
    loop {
        if (tile_start >= params.n_tokens) {
            break;
        }
        let t = tile_start + local;
        let is_active = t < params.n_tokens;
        var acc: f32 = 0.0;
        for (var b: u32 = 0u; b < n_blocks; b = b + 1u) {
            let block_byte_offset = row_byte_base + b * BLOCK_BYTES;
            var k: u32 = local;
            loop {
                if (k >= BLOCK_ELEMS) {
                    break;
                }
                shared_vals[k] = dequant_element(block_byte_offset, k);
                k = k + 64u;
            }
            workgroupBarrier();
            if (is_active) {
                let x_off = t * params.in_dim + b * BLOCK_ELEMS;
                for (var j: u32 = 0u; j < BLOCK_ELEMS; j = j + 1u) {
                    acc = acc + shared_vals[j] * x[x_off + j];
                }
            }
            workgroupBarrier();
        }
        if (is_active) {
            y[t * params.out_dim + o] = acc;
        }
        tile_start = tile_start + 64u;
    }
}
"#;

/// Row-tile / token-tile output-tiling dimensions for `MAIN_COOP_TILED_
/// SUFFIX`'s prefill GEMM — templated into the
/// WGSL text (`%TILE_ROWS%`/`%TILE_TOKENS%`/`%CHUNK%`,
/// `shader_source_coop_tiled`) rather than duplicated as separate literals
/// in the shader and in `VulkanBackend::build_op_resources`'s dispatch-
/// count math. `VulkanBackend` imports these same three constants for its
/// own dispatch math instead of re-declaring the numbers, so the shader and
/// the dispatch-count math can't drift out of sync. `TILE_TOKENS` (64)
/// matches the per-row cooperative kernel's own implicit token-tile size
/// (it loops 64 tokens at a time per weight-block dequant), so weight-
/// dequant reuse matches that kernel; `TILE_ROWS` (32) additionally reuses
/// activations across output rows, which the per-row cooperative kernel
/// does not (one workgroup per row, so every row's workgroup re-reads
/// `x` from global memory independently).
///
/// The two together also set how much arithmetic each shared-memory load
/// feeds: a `TILE_ROWS × TILE_TOKENS` tile does `TILE_ROWS * TILE_TOKENS`
/// multiply-adds per `TILE_ROWS + TILE_TOKENS` loaded elements, so widening
/// the tile is what moves the kernel off being load-issue-bound. It is not
/// free to widen indefinitely: the dispatch is one workgroup per
/// `(row-tile, token-tile)` pair, so a tile too tall leaves a matmul with a
/// small `out_dim` (a `qkv` projection, say) with fewer workgroups than the
/// adapter has compute units, and the tail dominates. 32 keeps the smallest
/// real projection above that floor while still feeding the register block
/// below.
///
/// `CHUNK` (32) is the K-dimension
/// streaming granularity and is deliberately *smaller* than the K-quant
/// types' native super-block size (`BLOCK_ELEMS = 256` for `Q4_K`/`Q5_K`/
/// `Q6_K`) so `tile_w`/`tile_x`'s combined shared-memory footprint
/// (`(TILE_ROWS + TILE_TOKENS) * CHUNK * 4` bytes = 12 KiB) stays bounded
/// regardless of `BLOCK_ELEMS` — using `BLOCK_ELEMS` itself as the tile
/// depth for `Q4_K` would need `(32 + 64) * 256 * 4` = 96 KiB, well past
/// typical workgroup-shared-memory limits. `elem_at` (below) restates
/// `dequant_element`'s existing `block_idx = k / BLOCK_ELEMS; local_k = k %
/// BLOCK_ELEMS` split (already used by `MAIN_REDUCE_SUFFIX`) as a small
/// helper so the K-loop can stream in `CHUNK`-sized pieces without knowing
/// or caring how big a type's native block actually is.
/// The tiled prefill GEMM's geometry, as one value because the five numbers
/// are not independent: the output tile is exactly what the thread grid and
/// each thread's register block multiply out to, and the staging run length
/// falls out of the tile and the thread count. Setting one without the others
/// produces a kernel whose dispatch grid and whose shader disagree.
///
/// Env-tunable as one knob for sweeping — `ORANGU_COOP_GEOM=ty:tx:ry:rx:chunk`
/// — because the interesting axis is shared-memory footprint against
/// occupancy, and that moves only when several of them move together. An
/// unparseable or invalid setting falls back to the default with a warning
/// rather than building a kernel that is quietly wrong.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoopGeom {
    /// Threads down the output tile; `threads_y * threads_x` must be 64.
    pub threads_y: u32,
    pub threads_x: u32,
    /// Output rows and tokens each thread accumulates in registers.
    pub reg_rows: u32,
    pub reg_tokens: u32,
    /// K-dimension streaming granularity.
    pub chunk: u32,
}

impl CoopGeom {
    pub const fn tile_rows(self) -> u32 {
        self.threads_y * self.reg_rows
    }
    pub const fn tile_tokens(self) -> u32 {
        self.threads_x * self.reg_tokens
    }
    /// Bytes of workgroup-shared memory the two staging tiles occupy — the
    /// quantity that actually caps occupancy on this kernel.
    pub const fn lds_bytes(self) -> u32 {
        (self.tile_rows() + self.tile_tokens()) * self.chunk * 4
    }
    /// Weight elements one thread stages per k-chunk.
    pub const fn run(self) -> u32 {
        (self.tile_rows() * self.chunk) / COOP_THREADS
    }

    /// Every constraint the generated kernel relies on. Returns why, so a bad
    /// `ORANGU_COOP_GEOM` says which rule it broke instead of failing later as
    /// wrong output or a driver error.
    fn check(self) -> Result<(), String> {
        if self.threads_y * self.threads_x != COOP_THREADS {
            return Err(format!(
                "threads_y * threads_x must be {COOP_THREADS}, got {}",
                self.threads_y * self.threads_x
            ));
        }
        if self.reg_rows == 0 || self.reg_tokens == 0 || self.chunk == 0 {
            return Err("reg_rows, reg_tokens and chunk must be non-zero".into());
        }
        if !self.tile_rows().is_multiple_of(4) || !self.tile_tokens().is_multiple_of(4) {
            return Err("tile rows and tokens must be multiples of 4 (vec4 staging)".into());
        }
        // The register block reads `tile_w`/`tile_x` as `vec4`, indexing
        // `(k * TILE + base) / 4`, so each thread's `base` — `ty * reg_rows`
        // and `tx * reg_tokens` — has to be 4-aligned for every thread.
        if !self.reg_rows.is_multiple_of(4) || !self.reg_tokens.is_multiple_of(4) {
            return Err(
                "reg_rows and reg_tokens must be multiples of 4 (vec4 register block)".into(),
            );
        }
        // The recorded-dispatch path still assumes the default token tile; a
        // narrower one disagrees with it at `n_tokens` past one tile
        // (`recorded_matmul_matches_backend_matmul_model_shaped` catches it).
        if self.tile_tokens() != COOP_GEOM_DEFAULT.tile_tokens() {
            return Err(format!(
                "token tile must be {} for now",
                COOP_GEOM_DEFAULT.tile_tokens()
            ));
        }
        // `coop_tiled_x_fill` relies on `local % chunk` being the same on
        // every one of a thread's staging steps, which needs the chunk to
        // divide the thread count.
        if !COOP_THREADS.is_multiple_of(self.chunk) {
            return Err(format!(
                "chunk {} must divide the thread count {COOP_THREADS}",
                self.chunk
            ));
        }
        // `coop_tiled_x_fill` hands each thread whole four-token quads of the
        // token tile — one shared-memory vector per store, which is what
        // makes the fill portable to a backend without component-granular
        // shared stores (see `coop_vec4_tiles`). The quads have to divide
        // evenly among the `COOP_THREADS / chunk` threads that share a k
        // column, or some thread gets a partial quad and the store is no
        // longer whole.
        let x_threads_per_k = COOP_THREADS / self.chunk;
        if !(self.tile_tokens() / 4).is_multiple_of(x_threads_per_k) {
            return Err(format!(
                "chunk {} leaves {x_threads_per_k} threads per k column, which does not \
                 divide the token tile's {} four-token quads",
                self.chunk,
                self.tile_tokens() / 4
            ));
        }
        let run = self.run();
        if run == 0 || !self.chunk.is_multiple_of(run) {
            return Err(format!(
                "staging run {run} must divide chunk {}",
                self.chunk
            ));
        }
        // **16, not 32.** `Q4_K`/`Q5_K` change scale/min every 32 elements, but
        // `Q6_K` changes its 8-bit scale every **16** (`sc_idx` is `l / 16`),
        // and `coop_tiled_run_fill` hoists that scale out of the run. A run of
        // 32 therefore dequantizes half its `Q6_K` elements with the wrong
        // scale — silently, and only for that one type. This bound was `32`
        // until a geometry sweep produced `run == 32` and
        // `matmul_matches_cpu_backend_cooperative_path_q6_k` caught it.
        if !16u32.is_multiple_of(run) {
            return Err(format!(
                "staging run {run} must divide 16 — Q6_K's scale group, the \
                 narrowest a K-quant has"
            ));
        }
        if self.lds_bytes() > 32 * 1024 {
            return Err(format!(
                "{} B of shared memory exceeds 32 KiB",
                self.lds_bytes()
            ));
        }
        Ok(())
    }
}

/// Threads per tiled-GEMM workgroup. One wave64 / two wave32s.
pub const COOP_THREADS: u32 = 64;

/// The default geometry: a 32×64 output tile from an 8×8 thread grid with a
/// 4×8 register block per thread, streaming K **16** at a time — 6144 B of
/// shared memory.
///
/// The chunk was 32, and halving it is worth **+21–22% on prefill at every
/// prompt length**. It is the one knob here that changes shared-memory
/// footprint without changing the arithmetic: FMAs and staged elements both
/// scale linearly with the chunk, so the ratio between them — 21 multiply-adds
/// per staged element — is identical either way. What changes is that the two
/// tiles drop from 12 288 B to 6144 B, and this kernel's occupancy is capped by
/// exactly that: 6 subgroups/SIMD becomes 8, with the registers left over.
///
/// | chunk | LDS | waves/SIMD | pp 1120 | pp 3356 |
/// | ---: | ---: | ---: | ---: | ---: |
/// | 32 | 12 288 B | 6 | 130.7 | 127.7 |
/// | **16** | **6144 B** | **8** | **159.4 / 159.0** | **154.2 / 155.0** |
/// | 8 | 3072 B | — | 156.8 | 152.3 |
///
/// 8 is worse than 16: twice as many `workgroupBarrier` rounds over the K
/// dimension stops paying for the occupancy it buys.
pub const COOP_GEOM_DEFAULT: CoopGeom = CoopGeom {
    threads_y: 8,
    threads_x: 8,
    reg_rows: 4,
    reg_tokens: 8,
    // **32, and the static metrics all say it should be 16.** Doubling the
    // chunk doubles shared memory to 12 KiB, which takes occupancy from 10
    // subgroups per SIMD to 6, raises the instruction count 12-28% and makes
    // ACO's modelled inverse throughput two to three times worse. It is also
    // measurably faster: `+3.4%` on `gemma-4-E2B Q4_K_M` and `+4.3%` on
    // `Llama-3.2-3B Q4_K_M` at `pp512`, with all four of the second model's
    // arms above all four of its `chunk 16` arms.
    //
    // An earlier sweep recorded `32` as 3.1% *slower*. That sweep ran its
    // arms in a fixed order, and the later slot on this rig is worth about
    // 3% on its own — every arm here holds every position exactly once
    // instead. See `DISK.md`, D7.
    //
    // The honest summary is that this kernel's throughput is not predicted by
    // occupancy, instruction count or the scheduler's own model, so a
    // geometry can only be chosen by measuring it.
    chunk: 32,
};

/// Whether the tiled GEMM stages its two shared tiles as `f16` instead of
/// `f32`.
///
/// P5 established that this kernel is capped by shared-memory footprint, not by
/// registers or arithmetic — halving the K chunk from 32 to 16 took occupancy
/// from 6 to 8 subgroups/SIMD and prefill up 21%. Halving the element width
/// does the same thing again without touching the tile geometry, so weight
/// reuse and the FMA-per-staged-element ratio are unchanged; only the footprint
/// and the LDS traffic move.
///
/// Accumulation stays `f32` — only the staged values narrow. Weights arrive
/// from a K-quant dequant (a `f16`-derived scale times a 4- to 6-bit integer,
/// minus a `f16`-derived min), so `f16`'s 10-bit mantissa holds them almost
/// exactly; the activations are the side that can lose precision.
///
/// **Opt-in, and staying that way.** Measured: occupancy does rise a long way
/// — 6144 B and 10 subgroups/SIMD become 3072 B and 16–18 — and prefill moves
/// only **+2.4% at 1120 tokens, +3.3% at 2238**. That is the finding, not the
/// speedup: after P5 this kernel is no longer occupancy-bound, so halving the
/// footprint again buys almost nothing. The cost is real, though — staging the
/// inputs at `f16` puts ~1.7% relative error on a 512-term dot product of
/// adversarial random data (√512 × 2⁻¹¹), which is why
/// `matmul_matches_cpu_backend_*` and the fused-chain cross-checks fail under
/// this flag. Their tolerances are calibrated for `f32` staging and were left
/// alone deliberately: loosening a correctness bound to admit a 3% win is the
/// wrong trade, and real activations are not the adversarial case anyway.
///
/// Kept because it is written, measured and documented, and because a device
/// where LDS bandwidth rather than footprint is the limit would see more from
/// it than this one does.
///
/// `ORANGU_COOP_F16_TILES=1`. Needs `SHADER_F16`, which the caller checks.
pub fn coop_f16_tiles() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| crate::engine::env::flag_on("ORANGU_COOP_F16_TILES"))
}

/// Which of the tiled GEMM's two shared tiles may be held as `vec4` — see
/// [`coop_vec4_tiles`], which is where the reasoning lives.
///
/// Two flags rather than one because the two tiles are filled by different
/// code under different constraints, and only one of them has the hazard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoopVec4Tiles {
    /// `tile_w`, the weight tile.
    pub w: bool,
    /// `tile_x`, the activation tile.
    pub x: bool,
}

/// Whether the tiled GEMM may hold each of its shared tiles as `vec4` — **a
/// correctness question, not a tuning one**.
///
/// # The hazard
///
/// The tiled kernel is the only one that fills shared memory by having many
/// threads each write a single dynamically-indexed *component* of a shared
/// vector — four different threads writing the four components of one
/// `vec4`. On Vulkan that lowers to a 4-byte store, and the reads can then
/// take four values per load, which is what the `vec4` tiles are for.
///
/// On `wgpu`'s Metal backend it does not. Measured on a real device: of each
/// `vec4`, exactly one component survived the barrier and the other three
/// read back as their initial value — the signature of a read-modify-write of
/// the whole 16-byte vector, with the four writing threads clobbering one
/// another. Every tiled-path cross-check disagreed with the CPU backend
/// there, including plain `f32`, while the scalar-shared-memory reduce
/// kernels all passed. `vulkan::tests::coop_tiles_are_vec4_only_where_component_stores_work`
/// is that finding reduced to one kernel;
/// `vulkan::tests::shared_vec4_whole_stores_survive_a_barrier` is the control
/// that pins the
/// fault to the *component* store rather than to shared `vec4` memory, the
/// barrier, or the readback.
///
/// # Why that is a per-tile answer, not a per-backend one
///
/// The control is the interesting half: **whole-vector stores are fine
/// everywhere**. A tile whose fill can be arranged so each thread writes a
/// complete `vec4` therefore needs no gate at all.
///
/// - `tile_x` **can** be arranged that way, and is (`coop_tiled_x_fill`
///   stages four consecutive tokens per store). The tile is k-major, so four
///   consecutive tokens at one `k` are four consecutive slots — one vector.
///   Giving a thread that quad instead of four tokens a stride apart is a
///   pure permutation of which thread stages which element: same addresses,
///   same coalescing, same values, bit-identical output. So `x` is `true`
///   everywhere.
/// - `tile_w` **cannot**, not without giving something else up. A thread
///   stages a run of consecutive `k` from *one* row, because that is what
///   lets the quantized block's scale/min pair be derived once per run
///   instead of once per element (`coop_tiled_run_fill`). In a k-major tile
///   those values land `TILE_ROWS` apart — never one vector. Making them
///   contiguous means either a row-major tile (which moves the problem into
///   the register block's read, where it costs more) or a thread staging four
///   rows at once (which multiplies the per-run dequant setup by four). Both
///   are real designs; neither is a fallback. So `w` stays gated on the
///   store's actual behaviour.
///
/// The one exception is the `ORANGU_NO_TILE_X_STRAIGHT` control, which
/// restores the *old* per-element `tile_x` fill verbatim so the straightened
/// one can be A/B'd against the code it replaced. That fill writes
/// components, so on a backend where components clobber, asking for it also
/// drops `tile_x` back to scalar — the alternative is a control that silently
/// computes the wrong answer.
///
/// Defaulting an unrecognized backend to the safe form is the deliberate
/// direction: a new backend that silently computed wrong numbers would be far
/// worse than one that is a little slower until someone runs the probe on it.
///
/// # Forcing it
///
/// `ORANGU_COOP_SCALAR_TILES` forces the scalar form on: `w` or `x` for one
/// tile, anything else non-empty (`1`) for both.
///
/// Naming a single tile is what makes the combination a backend without
/// component stores runs — scalar `tile_w`, `vec4` `tile_x` — reachable on a
/// backend that *has* them. The generated WGSL is what differs between the
/// two, not the driver, so `ORANGU_COOP_SCALAR_TILES=w` puts the whole
/// cross-check suite over the code path an Apple GPU takes, on hardware that
/// is actually to hand. It is also the honest way to A/B the two tiles
/// separately: whichever one is worth converting, the measurement should say
/// which.
pub fn coop_vec4_tiles(backend: wgpu::Backend) -> CoopVec4Tiles {
    let (force_w, force_x) =
        forced_scalar_tiles(&std::env::var("ORANGU_COOP_SCALAR_TILES").unwrap_or_default());
    let component_stores_work = backend == wgpu::Backend::Vulkan;
    CoopVec4Tiles {
        w: component_stores_work && !force_w,
        // Whole-vector stores everywhere, so no gate — unless the fill that
        // makes them whole has been swapped out for the per-element control.
        x: (component_stores_work || !rolled_x_fill()) && !force_x,
    }
}

/// `(force_w, force_x)` from an `ORANGU_COOP_SCALAR_TILES` value — see
/// [`coop_vec4_tiles`]. Split out so the parse is testable without setting a
/// process-global variable that a concurrently-running test could observe.
///
/// An unrecognized value forces both tiles scalar rather than being ignored:
/// `=1` is what this variable meant before it could name a tile, and a typo
/// should land on the conservative form rather than quietly leaving on the
/// one the operator was trying to rule out.
fn forced_scalar_tiles(spec: &str) -> (bool, bool) {
    match spec.trim() {
        "" => (false, false),
        "w" => (true, false),
        "x" => (false, true),
        _ => (true, true),
    }
}

/// Whether `ORANGU_NO_TILE_X_STRAIGHT=1` asked for the pre-straightening
/// per-element activation fill — see [`ROLLED_X_FILL`] for what it restores
/// and [`coop_vec4_tiles`] for why the answer also decides `tile_x`'s form.
fn rolled_x_fill() -> bool {
    crate::engine::env::flag_on("ORANGU_NO_TILE_X_STRAIGHT")
}

/// The geometry in force, read once from `ORANGU_COOP_GEOM`.
pub fn coop_geom() -> CoopGeom {
    static G: std::sync::OnceLock<CoopGeom> = std::sync::OnceLock::new();
    *G.get_or_init(|| {
        let Some(spec) = std::env::var_os("ORANGU_COOP_GEOM") else {
            return COOP_GEOM_DEFAULT;
        };
        let spec = spec.to_string_lossy().to_string();
        let parts: Vec<u32> = spec
            .split(':')
            .filter_map(|f| f.trim().parse::<u32>().ok())
            .collect();
        if parts.len() != 5 {
            eprintln!(
                "orangu-server: ORANGU_COOP_GEOM={spec}: expected ty:tx:ry:rx:chunk, using default"
            );
            return COOP_GEOM_DEFAULT;
        }
        let g = CoopGeom {
            threads_y: parts[0],
            threads_x: parts[1],
            reg_rows: parts[2],
            reg_tokens: parts[3],
            chunk: parts[4],
        };
        match g.check() {
            Ok(()) => {
                eprintln!(
                    "orangu-server: coop tile {}x{} chunk {} ({} B LDS, run {})",
                    g.tile_rows(),
                    g.tile_tokens(),
                    g.chunk,
                    g.lds_bytes(),
                    g.run()
                );
                g
            }
            Err(why) => {
                eprintln!("orangu-server: ORANGU_COOP_GEOM={spec}: {why}; using default");
                COOP_GEOM_DEFAULT
            }
        }
    })
}

/// The tiled-GEMM alternative to `MAIN_COOP_SUFFIX` — see `Self::
/// shader_source_coop_tiled` and `MAIN_COOP_SUFFIX`'s own doc comment for
/// why this is now the default (opt out with `ORANGU_NO_TILED_
/// PREFILL=1`) rather than staying opt-in.
///
/// One workgroup computes a `TILE_ROWS × TILE_TOKENS` output tile,
/// streaming the K dimension through shared memory in `CHUNK`-sized
/// pieces: each of the 64 threads (arranged as an 8-per-column ×
/// 8-per-row grid — `THREADS_Y × THREADS_X`) cooperatively fills
/// `tile_w`/`tile_x` for the current chunk, then owns a
/// `REG_ROWS × REG_TOKENS` (4×8) register
/// block of the output tile — written out as named scalars by
/// `coop_tiled_register_block`, see its own doc comment for why an
/// `array<f32, N>` will not do — accumulating its own 32 output elements'
/// partial dot products against the shared chunk before the next chunk
/// overwrites `tile_w`/`tile_x`. Both tiles are stored **k-major and as
/// `vec4`**: k-major so the `REG_ROWS`/`REG_TOKENS` values one thread reads
/// per `k` sit in consecutive shared-memory addresses rather than `CHUNK`
/// apart (which keeps a subgroup's lanes off a single bank), and `vec4` so
/// those contiguous values are fetched a quarter as many instructions —
/// three loads per `k` instead of twelve, for the same arithmetic. The
/// fills write through a dynamic component index (`tile[i >> 2u][i & 3u]`),
/// which keeps each thread's *global*-side access pattern exactly as it was;
/// the fill is a small fraction of the loop's work and the global pattern is
/// the part worth protecting. This gives every weight element
/// `TILE_TOKENS`-way reuse (same as `MAIN_COOP_SUFFIX`) *and* every
/// activation element `TILE_ROWS`-way reuse (which `MAIN_COOP_SUFFIX` has
/// none of at all), at the cost of finer-grained K-streaming than that
/// kernel's native per-block granularity — more `workgroupBarrier` rounds
/// for the K-quant types, whose native block is 256 elements wide vs. this
/// kernel's fixed 32-element `CHUNK`.
///
/// The weight fill is **run-oriented**: one thread stages `RUN` consecutive
/// `k` of a single row through `fill_w_run` ([`coop_tiled_run_fill`]), rather
/// than one element each of many rows. That is what lets a quantized type
/// derive its shared per-block constants once per run. The earlier
/// element-at-a-time fill made `dequant_element` re-derive, for every one of
/// the 1024 weight elements staged per chunk, values that 32 of them (the
/// sub-block scale/min) and 256 of them (the block scale) share — seven to
/// eight dependent loads to produce one float, against a kernel with ~6
/// waves/SIMD to hide them behind. `ORANGU_NO_TILE_DEQUANT_RUN=1` restores it
/// for A/B.
///
/// Out-of-range rows/tokens (a tile straddling the matrix edge) are zero-
/// filled while loading and skipped while writing, the same bounds-check
/// idiom `MAIN_REDUCE_SUFFIX` already uses for `REDUCE_N_ROWS`-imperfect
/// `out_dim`; the weight fill takes a bounds-checked per-element path there
/// rather than complicating the run.
const MAIN_COOP_TILED_SUFFIX: &str = r#"
const TILE_ROWS: u32 = %TILE_ROWS%u;
const TILE_TOKENS: u32 = %TILE_TOKENS%u;
const CHUNK: u32 = %CHUNK%u;
const THREADS_Y: u32 = %THREADS_Y%u;
const THREADS_X: u32 = %THREADS_X%u;
const REG_ROWS: u32 = %REG_ROWS%u;
const REG_TOKENS: u32 = %REG_TOKENS%u;
// Weight elements one thread stages per k-chunk, all from a single row and
// contiguous in k — see fill_w_run below.
const RUN: u32 = %RUN%u;
const RUNS_PER_ROW: u32 = CHUNK / RUN;
const THREADS: u32 = %THREADS%u;
// Four-token quads in the token tile — the unit `store_x4` writes, and so the
// unit `coop_tiled_x_fill` hands out.
const X_QUADS: u32 = TILE_TOKENS / 4u;

%TILE_DECL%

fn elem_at(row_byte_base: u32, k: u32) -> f32 {
    let block_idx = k / BLOCK_ELEMS;
    let local_k = k % BLOCK_ELEMS;
    return dequant_element(row_byte_base + block_idx * BLOCK_BYTES, local_k);
}

// `tile_w` is k-major (`k * TILE_ROWS + row`); the fill and the dot-product
// loop agree on that through this one helper rather than by restating the
// index arithmetic at each site.
fn store_w(rr: u32, kk: u32, v: f32) {
    let widx = kk * TILE_ROWS + rr;
%STORE_W%
}

// `tile_x` is k-major too (`k * TILE_TOKENS + token`), same reason as
// `tile_w`: the values one thread reads per `k` end up in consecutive
// shared-memory addresses.
//
// One element at a time — only the pre-straightening control fill
// (`ORANGU_NO_TILE_X_STRAIGHT`) still writes this way, and only where a
// component store into a shared vector is a component-sized store. Everything
// else goes through `store_x4`.
fn store_x(tt: u32, kk: u32, v: f32) {
    let xidx = kk * TILE_TOKENS + tt;
%STORE_X%
}

// A whole four-token quad at one `k`. Because `tile_x` is k-major those four
// tokens are four consecutive slots, so this is a single **whole**-vector
// store — no thread ever writes a component of a vector another thread is
// also writing. That is what makes the activation tile's `vec4` form portable
// to a backend whose shared-vector component store is a read-modify-write of
// the whole 16 bytes; see `coop_vec4_tiles`.
fn store_x4(tt0: u32, kk: u32, v: vec4<f32>) {
    let xidx = kk * TILE_TOKENS + tt0;
%STORE_X4%
}

%RUN_FILL%

@compute @workgroup_size(%THREADS%)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let row_tiles = (params.out_dim + TILE_ROWS - 1u) / TILE_ROWS;
    let token_tiles = (params.n_tokens + TILE_TOKENS - 1u) / TILE_TOKENS;
    let flat = wid.x + wid.y * nwg.x + wid.z * nwg.x * nwg.y;
    if (flat >= row_tiles * token_tiles) {
        return;
    }
    let rtile = flat / token_tiles;
    let ttile = flat % token_tiles;
    let row_start = rtile * TILE_ROWS;
    let token_start = ttile * TILE_TOKENS;

    let local = lid.x;
    let ty = local / THREADS_X;
    let tx = local % THREADS_X;

%ACC_DECL%
    let in_dim = params.in_dim;
    var chunk_start: u32 = 0u;
    loop {
        if (chunk_start >= in_dim) {
            break;
        }

%W_FILL%

%X_FILL%

        workgroupBarrier();

        let w_base = ty * REG_ROWS;
        let x_base = tx * REG_TOKENS;
        var k: u32 = 0u;
        loop {
            if (k >= CHUNK) {
                break;
            }
%INNER%
            k = k + %K_UNROLL%u;
        }

        workgroupBarrier();
        chunk_start = chunk_start + CHUNK;
    }

%STORE%
}
"#;

/// The register-block half of [`MAIN_COOP_TILED_SUFFIX`], written out with
/// literal indices rather than as `array<f32, N>` locals walked by a loop
/// variable.
///
/// WGSL function-scope arrays indexed dynamically do not become registers:
/// naga emits them as `Function`-storage variables with dynamic access
/// chains, which the driver's compiler keeps in scratch memory unless it
/// manages to promote them. That turned the innermost `REG_ROWS × REG_TOKENS`
/// accumulator block — the loop that should be pure FMA on values already in
/// registers — into scratch traffic on every single multiply-add. Emitting
/// `acc_i_j` scalars keeps the whole 4×4 block in registers, which is the
/// entire point of an output-tiled GEMM.
///
/// Generated from the same `REG_ROWS`/`REG_TOKENS` the tile constants above
/// imply (4×4, from `TILE_ROWS`/`THREADS_Y` and `TILE_TOKENS`/`THREADS_X`),
/// so the unrolled text cannot drift from the tile geometry.
fn coop_tiled_register_block(
    g: CoopGeom,
    f16_tiles: bool,
    vec4_tiles: CoopVec4Tiles,
) -> (String, String, String, usize) {
    let reg_rows = g.reg_rows as usize;
    let reg_tokens = g.reg_tokens as usize;
    // K iterations fused into one loop body; must divide the chunk.
    let k_unroll: usize = if (g.chunk as usize).is_multiple_of(2) {
        2
    } else {
        1
    };

    let mut decl = String::new();
    for i in 0..reg_rows {
        for j in 0..reg_tokens {
            decl.push_str(&format!("    var acc_{i}_{j}: f32 = 0.0;\n"));
        }
    }

    // The K loop is unrolled `K_UNROLL`-wide as well: one iteration's 16 FMAs
    // depend on that iteration's 8 shared loads, so a single-step loop leaves
    // the FMA units waiting on LDS with nothing else in flight. Interleaving
    // `K_UNROLL` independent iterations gives the scheduler that other work,
    // and amortizes the loop's own compare/branch over 4× the arithmetic.
    // With `vec4` tiles one load covers four of the block's values; with
    // scalar tiles it takes four loads, so the load count — but nothing
    // else — differs between the two forms. Both still issue every load for
    // an unrolled step before the first FMA consumes one.
    let mut inner = String::new();
    for u in 0..k_unroll {
        if vec4_tiles.w {
            for i in 0..reg_rows.div_ceil(4) {
                inner.push_str(&format!(
                    "            let wv{u}_{i} = tile_w[((k + {u}u) * TILE_ROWS + w_base) / 4u + {i}u];\n"
                ));
            }
        } else {
            for i in 0..reg_rows {
                inner.push_str(&format!(
                    "            let wv{u}_{i} = tile_w[(k + {u}u) * TILE_ROWS + w_base + {i}u];\n"
                ));
            }
        }
        if vec4_tiles.x {
            for j in 0..reg_tokens.div_ceil(4) {
                inner.push_str(&format!(
                    "            let xv{u}_{j} = tile_x[((k + {u}u) * TILE_TOKENS + x_base) / 4u + {j}u];\n"
                ));
            }
        } else {
            for j in 0..reg_tokens {
                inner.push_str(&format!(
                    "            let xv{u}_{j} = tile_x[(k + {u}u) * TILE_TOKENS + x_base + {j}u];\n"
                ));
            }
        }
    }
    for u in 0..k_unroll {
        for i in 0..reg_rows {
            for j in 0..reg_tokens {
                let w = if vec4_tiles.w {
                    format!("wv{u}_{}[{}]", i / 4, i % 4)
                } else {
                    format!("wv{u}_{i}")
                };
                let x = if vec4_tiles.x {
                    format!("xv{u}_{}[{}]", j / 4, j % 4)
                } else {
                    format!("xv{u}_{j}")
                };
                let (w, x) = if f16_tiles {
                    (format!("f32({w})"), format!("f32({x})"))
                } else {
                    (w, x)
                };
                inner.push_str(&format!(
                    "            acc_{i}_{j} = fma({w}, {x}, acc_{i}_{j});\n"
                ));
            }
        }
    }

    let mut store = String::new();
    for i in 0..reg_rows {
        store.push_str(&format!(
            "    let row{i} = row_start + ty * REG_ROWS + {i}u;\n    if (row{i} < params.out_dim) {{\n"
        ));
        for j in 0..reg_tokens {
            store.push_str(&format!(
                "        let token{i}_{j} = token_start + tx * REG_TOKENS + {j}u;\n        \
                 if (token{i}_{j} < params.n_tokens) {{\n            \
                 y[token{i}_{j} * params.out_dim + row{i}] = acc_{i}_{j};\n        }}\n"
            ));
        }
        store.push_str("    }\n");
    }

    (decl, inner, store, k_unroll)
}

/// How many weight elements one thread stages per k-chunk in
/// [`MAIN_COOP_TILED_SUFFIX`]'s fill — the whole tile spread over the 64
/// threads.
///
/// The two divisibility conditions are what make the per-run hoisting in
/// [`coop_tiled_run_fill`] *correct*, not merely faster, so they are asserted
/// here rather than left as a comment:
///
/// - `CHUNK % RUN == 0` makes each run's first `k` a multiple of `RUN`, since
///   the chunk start already is a multiple of `CHUNK`.
/// - `16 % RUN == 0` keeps the whole run inside one scale group. `Q4_K`/`Q5_K`
///   change scale/min every 32 elements, but **`Q6_K` changes its 8-bit scale
///   every 16** (`sc_idx` is `l / 16`), so 16 is the binding figure. A run that
///   straddled two groups would silently dequantize half its `Q6_K` elements
///   with the wrong scale — which is exactly what `run == 32` did until a
///   geometry sweep produced it.
fn coop_tiled_run_len(g: CoopGeom) -> u32 {
    debug_assert!(g.check().is_ok(), "geometry validated before use");
    g.run()
}

/// The weight-staging loop [`MAIN_COOP_TILED_SUFFIX`] runs once per k-chunk,
/// and the `fill_w_run` it calls, as `(fill, run_fn)`.
///
/// `amortized == false` restores the element-at-a-time grid-strided fill this
/// replaced, verbatim, so `ORANGU_NO_TILE_DEQUANT_RUN=1` is a genuine control
/// rather than a different arrangement of the new one — an A/B against a
/// halfway variant would have measured neither the change nor the baseline.
fn coop_tiled_weight_fill(ggml_type: u32, run: u32, amortized: bool) -> (String, String) {
    if !amortized {
        return (
            r#"        var fi: u32 = local;
        loop {
            if (fi >= TILE_ROWS * CHUNK) {
                break;
            }
            let rr = fi / CHUNK;
            let kk = fi % CHUNK;
            let row_idx = row_start + rr;
            let k_global = chunk_start + kk;
            if (row_idx < params.out_dim && k_global < in_dim) {
                store_w(rr, kk, elem_at(row_idx * params.row_bytes, k_global));
            } else {
                store_w(rr, kk, 0.0);
            }
            // `THREADS`, the WGSL const, not the `%THREADS%` placeholder:
            // this text is inserted via `%W_FILL%`, which the substitution
            // chain applies *after* `%THREADS%`, so a placeholder here would
            // survive into the final source and fail to parse. `ROLLED_X_FILL`
            // — the sibling control, for `ORANGU_NO_TILE_X_STRAIGHT` — already
            // does it this way.
            fi = fi + THREADS;
        }
"#
            .to_string(),
            String::new(),
        );
    }
    let fill = r#"        // Each thread stages `RUN` consecutive k of ONE row, so every
        // constant a quantized weight element shares with its neighbours —
        // the block scale, and for the K-quants the sub-block scale/min pair
        // — is derived once per run instead of once per element. `RUN`
        // divides both `CHUNK` and the K-quants' 32-element sub-block, so a
        // run never straddles two sub-blocks and those constants really are
        // constant across it; `coop_tiled_run_len` enforces that.
        let fill_row = local / RUNS_PER_ROW;
        let fill_kk0 = (local % RUNS_PER_ROW) * RUN;
        let fill_row_idx = row_start + fill_row;
        let fill_k0 = chunk_start + fill_kk0;
        if (fill_row_idx < params.out_dim && fill_k0 + RUN <= in_dim) {
            fill_w_run(fill_row_idx * params.row_bytes, fill_k0, fill_row, fill_kk0);
        } else {
            // A tile straddling the matrix edge: zero-fill what is out of
            // range, one bounds-checked element at a time. Off the hot path —
            // only the last row-tile and, for a type whose block is smaller
            // than `CHUNK`, the last k-chunk can reach it.
            var i: u32 = 0u;
            loop {
                if (i >= RUN) {
                    break;
                }
                var v: f32 = 0.0;
                if (fill_row_idx < params.out_dim && fill_k0 + i < in_dim) {
                    v = elem_at(fill_row_idx * params.row_bytes, fill_k0 + i);
                }
                store_w(fill_row, fill_kk0 + i, v);
                i = i + 1u;
            }
        }
"#
    .to_string();
    (fill, coop_tiled_run_fill(ggml_type, run))
}

/// The activation fill exactly as it was before it was straightened — the
/// control for `ORANGU_NO_TILE_X_STRAIGHT`.
const ROLLED_X_FILL: &str = r#"        var fj: u32 = local;
        loop {
            if (fj >= TILE_TOKENS * CHUNK) {
                break;
            }
            let tt = fj / CHUNK;
            let kk = fj % CHUNK;
            let token_idx = token_start + tt;
            let k_global = chunk_start + kk;
            var v: f32 = 0.0;
            if (token_idx < params.n_tokens && k_global < in_dim) {
                v = x[token_idx * in_dim + k_global];
            }
            store_x(tt, kk, v);
            fj = fj + THREADS;
        }
"#;

/// The activation-staging loop for [`MAIN_COOP_TILED_SUFFIX`], as straight-line
/// code with **every global load issued before any of them is consumed**.
///
/// This was a rolled loop over `TILE_TOKENS * CHUNK / 64` iterations, each
/// loading one `x` element and immediately writing it to shared memory. That is
/// the shape LESSONS §7 is about: RADV/ACO gives such a loop one destination
/// register and emits `s_waitcnt vmcnt(0)` before each use, so the loads run one
/// at a time. An ISA census of the kernel found **12 `vmcnt(0)` full drains**
/// against 1 in the decode GEMV, and this loop is where they were.
///
/// Written out, the loads land in distinct registers with no branch between
/// them and drain with counted `vmcnt(N)`.
///
/// The bounds check is hoisted to a whole-tile test rather than being paid per
/// element: both conditions are uniform across the workgroup, so the fast path
/// takes no divergent branch at all. The edge takes a rolled, bounds-checked
/// path.
///
/// # One thread, one quad
///
/// Each thread stages **four consecutive tokens** at its `k` per store, and
/// writes them as one vector through `store_x4`. It used to stage four tokens
/// a `THREADS / CHUNK` stride apart, which put its four values four *separate*
/// slots apart in the k-major tile — so four different threads ended up
/// writing the four components of each shared `vec4`, which is the pattern
/// `coop_vec4_tiles` exists to talk about. Handing a thread the quad instead
/// makes every shared store a whole-vector store, and that is the difference
/// between a `vec4` activation tile that works on one API and one that works
/// on both.
///
/// It changes nothing else. The set of loads is identical — the same
/// `TILE_TOKENS * CHUNK` elements, the same addresses, the same count per
/// thread — only permuted across threads, so the tile ends up holding exactly
/// the same values and the output is bit-identical. Coalescing is unchanged
/// too: for any one of a thread's loads, the `CHUNK` threads sharing a token
/// row still read `CHUNK` consecutive floats of it.
fn coop_tiled_x_fill(g: CoopGeom) -> String {
    // `ORANGU_NO_TILE_X_STRAIGHT=1` restores the rolled loop verbatim, so the
    // change can be A/B'd in one binary against the code it replaced rather
    // than against an approximation of it (LESSONS §17). `coop_vec4_tiles`
    // drops `tile_x` to scalar alongside it where that matters.
    if rolled_x_fill() {
        return ROLLED_X_FILL.to_string();
    }
    // Threads sharing a k column, and the four-token quads each one stages.
    // `CoopGeom::check` has already established that the second divides the
    // first evenly, which is what keeps every store whole.
    let threads_per_k = COOP_THREADS / g.chunk;
    let quads_per_thread = (g.tile_tokens() / 4) / threads_per_k;
    let mut s = String::from(
        "        // Straight-line, loads-before-stores — see `coop_tiled_x_fill`.\n\
        \x20        let x_kk = local % CHUNK;\n\
        \x20        let x_tt0 = (local / CHUNK) * 4u;\n\
        \x20        let x_k = chunk_start + x_kk;\n\
        \x20        if (token_start + TILE_TOKENS <= params.n_tokens && chunk_start + CHUNK <= in_dim) {\n\
        \x20            let x_base = (token_start + x_tt0) * in_dim + x_k;\n",
    );
    // Every load first, then every store: the whole point of the straightened
    // form. A quad's four tokens are adjacent rows of `x`, hence `in_dim`
    // apart; consecutive quads of one thread are `threads_per_k` quads apart.
    let quad_token = |i: u32| i * threads_per_k * 4;
    for i in 0..quads_per_thread {
        for c in 0..4 {
            s.push_str(&format!(
                "            let xv{i}_{c} = x[x_base + {}u * in_dim];\n",
                quad_token(i) + c
            ));
        }
    }
    for i in 0..quads_per_thread {
        s.push_str(&format!(
            "            store_x4(x_tt0 + {}u, x_kk, vec4<f32>(xv{i}_0, xv{i}_1, xv{i}_2, xv{i}_3));\n",
            quad_token(i)
        ));
    }
    // The edge. Also quad-at-a-time, for the same whole-store reason — it is
    // the last row/token tile of a matmul, not a rare path, so it has to be
    // correct on every backend rather than merely reachable on one.
    s.push_str(
        "        } else {\n\
        \x20            var fj: u32 = local;\n\
        \x20            loop {\n\
        \x20                if (fj >= X_QUADS * CHUNK) {\n\
        \x20                    break;\n\
        \x20                }\n\
        \x20                let kk = fj / X_QUADS;\n\
        \x20                let tt0 = (fj % X_QUADS) * 4u;\n\
        \x20                let k_global = chunk_start + kk;\n\
        \x20                let in_k = k_global < in_dim;\n\
        \x20                var q0: f32 = 0.0;\n\
        \x20                var q1: f32 = 0.0;\n\
        \x20                var q2: f32 = 0.0;\n\
        \x20                var q3: f32 = 0.0;\n\
        \x20                if (in_k && token_start + tt0 + 0u < params.n_tokens) {\n\
        \x20                    q0 = x[(token_start + tt0 + 0u) * in_dim + k_global];\n\
        \x20                }\n\
        \x20                if (in_k && token_start + tt0 + 1u < params.n_tokens) {\n\
        \x20                    q1 = x[(token_start + tt0 + 1u) * in_dim + k_global];\n\
        \x20                }\n\
        \x20                if (in_k && token_start + tt0 + 2u < params.n_tokens) {\n\
        \x20                    q2 = x[(token_start + tt0 + 2u) * in_dim + k_global];\n\
        \x20                }\n\
        \x20                if (in_k && token_start + tt0 + 3u < params.n_tokens) {\n\
        \x20                    q3 = x[(token_start + tt0 + 3u) * in_dim + k_global];\n\
        \x20                }\n\
        \x20                store_x4(tt0, kk, vec4<f32>(q0, q1, q2, q3));\n\
        \x20                fj = fj + THREADS;\n\
        \x20            }\n\
        \x20        }\n",
    );
    s
}

/// The `fill_w_run` a tiled kernel of `ggml_type` uses: dequantize `RUN`
/// consecutive `k` of one weight row into `tile_w`.
///
/// This is where the tiled GEMM stopped re-deriving, per element, values that
/// a whole run shares. The generic form below is what every element used to
/// cost: for `Q4_K` that is four `read_u8` for `d`/`dmin`, a
/// `get_scale_min_k4` (two or three more), and one for the nibble — seven to
/// eight dependent loads to produce a single float, with only ~6 waves/SIMD to
/// hide them behind. The specializations lift all of that out and leave one
/// load per element in the body.
///
/// Each specialization must be read against its type's `dequant_element` in
/// the corresponding `*_COOP_MIDDLE`; they are two statements of one format,
/// and `matmul_matches_cpu_backend_*` is what holds them to it.
/// Whether a run's weight bytes can be pulled as whole dwords instead of one
/// `read_u8` each.
///
/// `read_u8` is a dword load plus `>>2`, `&3`, `*8`, a shift and a mask — so
/// four consecutive elements load **the same dword four times** and pay the
/// address arithmetic four times. An ISA census of this kernel put the tile
/// fill at 520 integer operations and 84 VMEM loads per outer iteration,
/// against 118 instructions of accumulation; this is what most of that is.
///
/// It is only sound when the stream is 4-byte aligned, and that is a fact
/// about the *type*, not the geometry:
///
/// | type | block bytes | aligned |
/// | :-- | --: | :-- |
/// | `Q4_K` | 144 | yes |
/// | `Q5_K` | 176 | yes |
/// | `Q6_K` | **210** | **no** |
///
/// `Q6_K`'s block is 2-aligned, so `byte_offset` lands on an odd dword
/// boundary for every odd block and the loads would straddle. It keeps the
/// per-byte path. The run also has to be a multiple of four for the byte
/// index within a dword to be a compile-time constant — with a dynamic index
/// the extraction would go through scratch and lose more than it saves.
fn tile_dword_fill_ok(ggml_type: u32, run: u32) -> bool {
    let aligned_block = ggml_type == GGML_TYPE_Q4_K || ggml_type == GGML_TYPE_Q5_K;
    aligned_block
        && run.is_multiple_of(4)
        && !crate::engine::env::flag_on("ORANGU_NO_TILE_DWORD_FILL")
}

/// `run` consecutive bytes of one stream, loaded as `run / 4` aligned dwords
/// bound to `{name}_d0..`, for [`dword_byte`] to slice.
fn dword_stream(name: &str, base: &str, run: u32) -> String {
    let mut s = format!("    let {name}_w = ({base}) >> 2u;\n");
    for d in 0..run / 4 {
        s.push_str(&format!(
            "    let {name}_d{d} = weights[{name}_w + {d}u];\n"
        ));
    }
    s
}

/// Byte `i` of a [`dword_stream`], as an expression. Both indices are
/// compile-time constants, so this is one shift and one mask on a value
/// already in a register.
fn dword_byte(name: &str, i: u32) -> String {
    format!("(({name}_d{} >> {}u) & 0xFFu)", i / 4, (i % 4) * 8)
}

fn coop_tiled_run_fill(ggml_type: u32, run: u32) -> String {
    let body = match ggml_type {
        t if t == GGML_TYPE_Q4_K => {
            let mut s = String::from(
                "    let block_idx = k0 / BLOCK_ELEMS;\n\
                 \x20   let local_k0 = k0 % BLOCK_ELEMS;\n\
                 \x20   let byte_offset = row_byte_base + block_idx * BLOCK_BYTES;\n\
                 \x20   let d = f16_to_f32(read_u8(byte_offset) | (read_u8(byte_offset + 1u) << 8u));\n\
                 \x20   let dmin = f16_to_f32(read_u8(byte_offset + 2u) | (read_u8(byte_offset + 3u) << 8u));\n\
                 \x20   // Sub-block index is `local_k / 32`, constant over the run.\n\
                 \x20   let sm = get_scale_min_k4(byte_offset + 4u, local_k0 / 32u);\n\
                 \x20   let ds = d * f32(sm.x);\n\
                 \x20   let dm = dmin * f32(sm.y);\n\
                 \x20   // Low nibbles serve the first 32 of each 64-element group, high the\n\
                 \x20   // second, and both halves index the same 32 packed bytes.\n\
                 \x20   let q_base = byte_offset + 16u + (local_k0 / 64u) * 32u + (local_k0 % 32u);\n\
                 \x20   var shift: u32 = 0u;\n\
                 \x20   if ((local_k0 % 64u) >= 32u) {\n        shift = 4u;\n    }\n",
            );
            let dwords = tile_dword_fill_ok(ggml_type, run);
            if dwords {
                s.push_str(&dword_stream("q", "q_base", run));
            }
            for i in 0..run {
                let byte = if dwords {
                    dword_byte("q", i)
                } else {
                    format!("read_u8(q_base + {i}u)")
                };
                s.push_str(&format!(
                    "    store_w(rr, kk0 + {i}u, ds * f32(({byte} >> shift) & 0xFu) - dm);\n"
                ));
            }
            s
        }
        t if t == GGML_TYPE_Q5_K => {
            let mut s = String::from(
                "    let block_idx = k0 / BLOCK_ELEMS;\n\
                 \x20   let local_k0 = k0 % BLOCK_ELEMS;\n\
                 \x20   let byte_offset = row_byte_base + block_idx * BLOCK_BYTES;\n\
                 \x20   let d = f16_to_f32(read_u8(byte_offset) | (read_u8(byte_offset + 1u) << 8u));\n\
                 \x20   let dmin = f16_to_f32(read_u8(byte_offset + 2u) | (read_u8(byte_offset + 3u) << 8u));\n\
                 \x20   let idx = local_k0 / 64u;\n\
                 \x20   let l0 = local_k0 % 32u;\n\
                 \x20   var sub: u32 = idx * 2u;\n\
                 \x20   var nib_shift: u32 = 0u;\n\
                 \x20   var umask: u32 = 1u << (2u * idx);\n\
                 \x20   if ((local_k0 % 64u) >= 32u) {\n        \
                 sub = sub + 1u;\n        nib_shift = 4u;\n        \
                 umask = 2u << (2u * idx);\n    }\n\
                 \x20   let sm = get_scale_min_k4(byte_offset + 4u, sub);\n\
                 \x20   let ds = d * f32(sm.x);\n\
                 \x20   let dm = dmin * f32(sm.y);\n\
                 \x20   let ql_base = byte_offset + 48u + idx * 32u + l0;\n\
                 \x20   let qh_base = byte_offset + 16u + l0;\n",
            );
            let dwords = tile_dword_fill_ok(ggml_type, run);
            if dwords {
                s.push_str(&dword_stream("ql", "ql_base", run));
                s.push_str(&dword_stream("qh", "qh_base", run));
            }
            for i in 0..run {
                let (ql, qh) = if dwords {
                    (dword_byte("ql", i), dword_byte("qh", i))
                } else {
                    (
                        format!("read_u8(ql_base + {i}u)"),
                        format!("read_u8(qh_base + {i}u)"),
                    )
                };
                s.push_str(&format!(
                    "    let lo{i} = ({ql} >> nib_shift) & 0xFu;\n\
                     \x20   var hi{i}: i32 = 0;\n\
                     \x20   if (({qh} & umask) != 0u) {{\n        hi{i} = 16;\n    }}\n\
                     \x20   store_w(rr, kk0 + {i}u, ds * f32(i32(lo{i}) + hi{i}) - dm);\n"
                ));
            }
            s
        }
        t if t == GGML_TYPE_Q6_K => {
            let mut s = String::from(
                "    let block_idx = k0 / BLOCK_ELEMS;\n\
                 \x20   let local_k0 = k0 % BLOCK_ELEMS;\n\
                 \x20   let byte_offset = row_byte_base + block_idx * BLOCK_BYTES;\n\
                 \x20   let d = f16_to_f32(read_u8(byte_offset + 208u) | (read_u8(byte_offset + 209u) << 8u));\n\
                 \x20   let idx = local_k0 / 128u;\n\
                 \x20   // Which of the four interleaved 32-wide output ranges this run sits\n\
                 \x20   // in, and where in it — both constant over the run.\n\
                 \x20   let which_q = (local_k0 % 128u) / 32u;\n\
                 \x20   let l0 = local_k0 % 32u;\n\
                 \x20   let ql_base = byte_offset + idx * 64u + l0;\n\
                 \x20   let qh_base = byte_offset + 128u + idx * 32u + l0;\n\
                 \x20   var sc: i32 = i32(read_u8(byte_offset + 192u + idx * 8u + l0 / 16u + which_q * 2u));\n\
                 \x20   if (sc >= 128) {\n        sc = sc - 256;\n    }\n\
                 \x20   let dsc = d * f32(sc);\n\
                 \x20   var ql_extra: u32 = 0u;\n\
                 \x20   var ql_shift: u32 = 0u;\n\
                 \x20   let qh_shift: u32 = which_q * 2u;\n\
                 \x20   if (which_q == 1u || which_q == 3u) {\n        ql_extra = 32u;\n    }\n\
                 \x20   if (which_q >= 2u) {\n        ql_shift = 4u;\n    }\n",
            );
            for i in 0..run {
                s.push_str(&format!(
                    "    let ql{i} = (read_u8(ql_base + ql_extra + {i}u) >> ql_shift) & 0xFu;\n\
                     \x20   let qh{i} = ((read_u8(qh_base + {i}u) >> qh_shift) & 3u) << 4u;\n\
                     \x20   store_w(rr, kk0 + {i}u, dsc * f32(i32(ql{i} | qh{i}) - 32));\n"
                ));
            }
            s
        }
        // Everything else already costs about one load per element — `d` is
        // per 32 for `Q4_0`/`Q5_0`/`Q8_0` and there is no block header at all
        // for the float types — so the run is only straightened out, not
        // restructured. Straight-line rather than a loop because a rolled loop
        // (RADV/ACO) keeps one destination register and drains with
        // `vmcnt(0)`, serializing what should be `RUN` loads in flight.
        _ => {
            let mut s = String::new();
            for i in 0..run {
                s.push_str(&format!(
                    "    store_w(rr, kk0 + {i}u, elem_at(row_byte_base, k0 + {i}u));\n"
                ));
            }
            s
        }
    };
    format!(
        "// Dequantize `RUN` consecutive k of one weight row into `tile_w`.\n\
         fn fill_w_run(row_byte_base: u32, k0: u32, rr: u32, kk0: u32) {{\n{body}}}\n"
    )
}

/// `{ f32 }`, 1 element. Only one element exists, so `MAIN_COOP_SUFFIX`'s
/// distributed dequant only ever has thread 0 (`k == 0`) call this.
const F32_COOP_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 4u;
const BLOCK_ELEMS: u32 = 1u;
var<workgroup> shared_vals: array<f32, BLOCK_ELEMS>;
fn dequant_element(byte_offset: u32, k: u32) -> f32 {
    let bits = read_u8(byte_offset) | (read_u8(byte_offset + 1u) << 8u)
        | (read_u8(byte_offset + 2u) << 16u) | (read_u8(byte_offset + 3u) << 24u);
    return bitcast<f32>(bits);
}
"#;

/// `{ f16 }`, 1 element.
const F16_COOP_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 2u;
const BLOCK_ELEMS: u32 = 1u;
var<workgroup> shared_vals: array<f32, BLOCK_ELEMS>;
fn dequant_element(byte_offset: u32, k: u32) -> f32 {
    let bits = read_u8(byte_offset) | (read_u8(byte_offset + 1u) << 8u);
    return f16_to_f32(bits);
}
"#;

/// `{ bf16 }`, 1 element.
const BF16_COOP_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 2u;
const BLOCK_ELEMS: u32 = 1u;
var<workgroup> shared_vals: array<f32, BLOCK_ELEMS>;
fn dequant_element(byte_offset: u32, k: u32) -> f32 {
    let bits = read_u8(byte_offset) | (read_u8(byte_offset + 1u) << 8u);
    return bf16_to_f32(bits);
}
"#;

/// `block_q4_0`: mirrors `quant::dequantize_q4_0`'s low/high-nibble split
/// (signed, offset by 8), restated as a direct function of the target
/// index `k` (`0..32`) — `k < 16` is the low nibble at byte `k`, `k >= 16`
/// is the high nibble at byte `k - 16` — so up to 32 threads (or, in
/// `MAIN_REDUCE_SUFFIX`, all 64 via the grid-stride loop) can each
/// compute one `k` independently.
const Q4_0_COOP_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 18u;
const BLOCK_ELEMS: u32 = 32u;
var<workgroup> shared_vals: array<f32, BLOCK_ELEMS>;
fn dequant_element(byte_offset: u32, k: u32) -> f32 {
    let d = f16_to_f32(read_u8(byte_offset) | (read_u8(byte_offset + 1u) << 8u));
    if (k < 16u) {
        let byte = read_u8(byte_offset + 2u + k);
        return f32(i32(byte & 0xFu) - 8) * d;
    }
    let byte = read_u8(byte_offset + 2u + (k - 16u));
    return f32(i32(byte >> 4u) - 8) * d;
}
"#;

/// `block_q4_1`: mirrors `quant::dequantize_q4_1` — `Q4_0`'s nibble split
/// with a stored per-block minimum added instead of a fixed 8 subtracted.
const Q4_1_COOP_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 20u;
const BLOCK_ELEMS: u32 = 32u;
var<workgroup> shared_vals: array<f32, BLOCK_ELEMS>;
fn dequant_element(byte_offset: u32, k: u32) -> f32 {
    let d = f16_to_f32(read_u8(byte_offset) | (read_u8(byte_offset + 1u) << 8u));
    let m = f16_to_f32(read_u8(byte_offset + 2u) | (read_u8(byte_offset + 3u) << 8u));
    if (k < 16u) {
        let byte = read_u8(byte_offset + 4u + k);
        return f32(byte & 0xFu) * d + m;
    }
    let byte = read_u8(byte_offset + 4u + (k - 16u));
    return f32(byte >> 4u) * d + m;
}
"#;

/// `block_q5_1`: mirrors `quant::dequantize_q5_1` — `Q5_0`'s fifth bit
/// packed across `qh`, with `Q4_1`'s stored minimum.
const Q5_1_COOP_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 24u;
const BLOCK_ELEMS: u32 = 32u;
var<workgroup> shared_vals: array<f32, BLOCK_ELEMS>;
fn dequant_element(byte_offset: u32, k: u32) -> f32 {
    let d = f16_to_f32(read_u8(byte_offset) | (read_u8(byte_offset + 1u) << 8u));
    let m = f16_to_f32(read_u8(byte_offset + 2u) | (read_u8(byte_offset + 3u) << 8u));
    let qh = read_u8(byte_offset + 4u) | (read_u8(byte_offset + 5u) << 8u)
        | (read_u8(byte_offset + 6u) << 16u) | (read_u8(byte_offset + 7u) << 24u);
    if (k < 16u) {
        let byte = read_u8(byte_offset + 8u + k);
        let xh_0 = ((qh >> k) << 4u) & 0x10u;
        return f32((byte & 0xFu) | xh_0) * d + m;
    }
    let j = k - 16u;
    let byte = read_u8(byte_offset + 8u + j);
    let xh_1 = (qh >> (j + 12u)) & 0x10u;
    return f32((byte >> 4u) | xh_1) * d + m;
}
"#;

/// `block_q5_0`: mirrors `quant::dequantize_q5_0` — same low/high-nibble
/// split as `Q4_0_COOP_MIDDLE`, plus the 5th bit packed across `qh`.
const Q5_0_COOP_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 22u;
const BLOCK_ELEMS: u32 = 32u;
var<workgroup> shared_vals: array<f32, BLOCK_ELEMS>;
fn dequant_element(byte_offset: u32, k: u32) -> f32 {
    let d = f16_to_f32(read_u8(byte_offset) | (read_u8(byte_offset + 1u) << 8u));
    let qh = read_u8(byte_offset + 2u) | (read_u8(byte_offset + 3u) << 8u)
        | (read_u8(byte_offset + 4u) << 16u) | (read_u8(byte_offset + 5u) << 24u);
    if (k < 16u) {
        let byte = read_u8(byte_offset + 6u + k);
        let xh_0 = ((qh >> k) << 4u) & 0x10u;
        return f32(i32((byte & 0xFu) | xh_0) - 16) * d;
    }
    let j = k - 16u;
    let byte = read_u8(byte_offset + 6u + j);
    let xh_1 = (qh >> (j + 12u)) & 0x10u;
    return f32(i32((byte >> 4u) | xh_1) - 16) * d;
}
"#;

/// `block_q8_0`: mirrors `quant::dequantize_q8_0` — already trivially
/// per-element, one thread (or grid-stride iteration) per `k` in `0..32`.
const Q8_0_COOP_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 34u;
const BLOCK_ELEMS: u32 = 32u;
var<workgroup> shared_vals: array<f32, BLOCK_ELEMS>;
fn dequant_element(byte_offset: u32, k: u32) -> f32 {
    let d = f16_to_f32(read_u8(byte_offset) | (read_u8(byte_offset + 1u) << 8u));
    let byte = read_u8(byte_offset + 2u + k);
    var v: i32 = i32(byte);
    if (v >= 128) {
        v = v - 256;
    }
    return f32(v) * d;
}
"#;

/// `block_pq2_0`: mirrors `quant::unpack_block_pq2_0` — Prism's 128-element
/// `Q2_0`, one `f16` scale then 32 bytes of four 2-bit fields each, field
/// `k % 4` of byte `k / 4`, `0..=3` standing for `-1..=2`.
const PQ2_0_COOP_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 34u;
const BLOCK_ELEMS: u32 = 128u;
var<workgroup> shared_vals: array<f32, BLOCK_ELEMS>;
fn dequant_element(byte_offset: u32, k: u32) -> f32 {
    let d = f16_to_f32(read_u8(byte_offset) | (read_u8(byte_offset + 1u) << 8u));
    let byte = read_u8(byte_offset + 2u + (k >> 2u));
    let q = (byte >> ((k & 3u) * 2u)) & 3u;
    return (f32(q) - 1.0) * d;
}
"#;

/// `block_ptq1_0`: mirrors `quant::unpack_block_ptq1_0` — `TQ1_0`'s base-3
/// trits at group 128 with the `f16` scale **last** (byte 26). Element `k`
/// is trit `n` (most significant first) of one byte: `k < 80` is byte
/// `k % 16`, trit `k / 16`; `80..120` is byte `16 + (k - 80) % 8`, trit
/// `(k - 80) / 8`; `120..128` is `qh` byte `(k - 120) % 2`, trit
/// `(k - 120) / 2`. A trit is read by pushing it to the top with `3^n`
/// (wrapping in a byte) and taking `(q · 3) >> 8`.
const PTQ1_0_COOP_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 28u;
const BLOCK_ELEMS: u32 = 128u;
var<workgroup> shared_vals: array<f32, BLOCK_ELEMS>;
fn ptq1_trit(byte: u32, n: u32) -> f32 {
    var pow3 = array<u32, 5>(1u, 3u, 9u, 27u, 81u);
    let q = (byte * pow3[n]) & 0xFFu;
    return f32((q * 3u) >> 8u) - 1.0;
}
fn dequant_element(byte_offset: u32, k: u32) -> f32 {
    let d = f16_to_f32(read_u8(byte_offset + 26u) | (read_u8(byte_offset + 27u) << 8u));
    var byte_index: u32;
    var n: u32;
    if (k < 80u) {
        byte_index = k % 16u;
        n = k / 16u;
    } else if (k < 120u) {
        byte_index = 16u + (k - 80u) % 8u;
        n = (k - 80u) / 8u;
    } else {
        byte_index = 24u + (k - 120u) % 2u;
        n = (k - 120u) / 2u;
    }
    return ptq1_trit(read_u8(byte_offset + byte_index), n) * d;
}
"#;

/// `block_q4_K`: mirrors `quant::dequantize_q4_k`, whose sequential form
/// visits `q_offset` in `{0, 64, 128, 192}`, each covering a 64-wide
/// output range split into a low-nibble half (scale/min pair `is`) and a
/// high-nibble half (pair `is + 1`) — restated per target index `k`
/// (`0..256`) directly: which 64-wide group `k` falls in fixes
/// `q_offset`/`is`; which half of that group fixes low vs. high nibble.
const Q4_K_COOP_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 144u;
const BLOCK_ELEMS: u32 = 256u;
var<workgroup> shared_vals: array<f32, BLOCK_ELEMS>;
fn dequant_element(byte_offset: u32, k: u32) -> f32 {
    let d = f16_to_f32(read_u8(byte_offset) | (read_u8(byte_offset + 1u) << 8u));
    let dmin = f16_to_f32(read_u8(byte_offset + 2u) | (read_u8(byte_offset + 3u) << 8u));
    let scales_off = byte_offset + 4u;
    let qs_off = byte_offset + 16u;
    let q_offset = (k / 64u) * 64u;
    let local_in_group = k % 64u;
    let is_base = (q_offset / 64u) * 2u;
    let q_base = qs_off + q_offset / 2u;
    if (local_in_group < 32u) {
        let byte = read_u8(q_base + local_in_group);
        let sm = get_scale_min_k4(scales_off, is_base);
        let d1 = d * f32(sm.x);
        let m1 = dmin * f32(sm.y);
        return d1 * f32(byte & 0xFu) - m1;
    }
    let l = local_in_group - 32u;
    let byte = read_u8(q_base + l);
    let sm = get_scale_min_k4(scales_off, is_base + 1u);
    let d2 = d * f32(sm.x);
    let m2 = dmin * f32(sm.y);
    return d2 * f32(byte >> 4u) - m2;
}
"#;

/// `block_q5_K`: mirrors `quant::dequantize_q5_k` — same per-`k`
/// restatement as `Q4_K_COOP_MIDDLE`, plus `Q5_K`'s 5th bit (`qh`, keyed
/// by the same `q_offset`-derived iteration index `idx` that also derives
/// `u1`/`u2` and `ql_offset` in `quant::dequantize_q5_k`).
const Q5_K_COOP_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 176u;
const BLOCK_ELEMS: u32 = 256u;
var<workgroup> shared_vals: array<f32, BLOCK_ELEMS>;
fn dequant_element(byte_offset: u32, k: u32) -> f32 {
    let d = f16_to_f32(read_u8(byte_offset) | (read_u8(byte_offset + 1u) << 8u));
    let dmin = f16_to_f32(read_u8(byte_offset + 2u) | (read_u8(byte_offset + 3u) << 8u));
    let scales_off = byte_offset + 4u;
    let qh_off = byte_offset + 16u;
    let qs_off = byte_offset + 48u;
    let q_offset = (k / 64u) * 64u;
    let idx = q_offset / 64u;
    let local_in_group = k % 64u;
    let is_base = idx * 2u;
    let ql_offset = idx * 32u;
    let u1 = 1u << (2u * idx);
    let u2 = 2u << (2u * idx);
    if (local_in_group < 32u) {
        let l = local_in_group;
        let byte = read_u8(qs_off + ql_offset + l);
        let qhbyte = read_u8(qh_off + l);
        var hi_bit: i32 = 0;
        if ((qhbyte & u1) != 0u) {
            hi_bit = 16;
        }
        let sm = get_scale_min_k4(scales_off, is_base);
        let d1 = d * f32(sm.x);
        let m1 = dmin * f32(sm.y);
        return d1 * f32(i32(byte & 0xFu) + hi_bit) - m1;
    }
    let l = local_in_group - 32u;
    let byte = read_u8(qs_off + ql_offset + l);
    let qhbyte = read_u8(qh_off + l);
    var hi_bit: i32 = 0;
    if ((qhbyte & u2) != 0u) {
        hi_bit = 16;
    }
    let sm = get_scale_min_k4(scales_off, is_base + 1u);
    let d2 = d * f32(sm.x);
    let m2 = dmin * f32(sm.y);
    return d2 * f32(i32(byte >> 4u) + hi_bit) - m2;
}
"#;

/// `block_q6_K`: mirrors `quant::dequantize_q6_k`, whose sequential form
/// visits `y_off` in `{0, 128}`, each producing 4 interleaved 32-wide
/// output ranges (`q1`..`q4`, at `y_off+l`/`+32`/`+64`/`+96`) from the
/// same `ql`/`qh` bytes — restated per `k`: `y_off` and which-of-4
/// (`q1..q4`) come from `k`'s position, `l` is shared across all four so
/// only needs computing once regardless of which one `k` picked.
const Q6_K_COOP_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 210u;
const BLOCK_ELEMS: u32 = 256u;
var<workgroup> shared_vals: array<f32, BLOCK_ELEMS>;
fn dequant_element(byte_offset: u32, k: u32) -> f32 {
    let ql_off = byte_offset;
    let qh_off = byte_offset + 128u;
    let sc_off = byte_offset + 192u;
    let d = f16_to_f32(read_u8(byte_offset + 208u) | (read_u8(byte_offset + 209u) << 8u));
    let y_off = (k / 128u) * 128u;
    let idx = y_off / 128u;
    let local_in_group = k % 128u;
    let which_q = local_in_group / 32u;
    let l = local_in_group % 32u;
    let ql_o = idx * 64u;
    let qh_o = idx * 32u;
    let sc_o = idx * 8u;
    let is = l / 16u;
    let ql_l = read_u8(ql_off + ql_o + l);
    let ql_l32 = read_u8(ql_off + ql_o + l + 32u);
    let qh_l = read_u8(qh_off + qh_o + l);
    var q: i32;
    var sc_idx: u32;
    if (which_q == 0u) {
        q = i32((ql_l & 0xFu) | ((qh_l & 3u) << 4u)) - 32;
        sc_idx = is;
    } else if (which_q == 1u) {
        q = i32((ql_l32 & 0xFu) | (((qh_l >> 2u) & 3u) << 4u)) - 32;
        sc_idx = is + 2u;
    } else if (which_q == 2u) {
        q = i32((ql_l >> 4u) | (((qh_l >> 4u) & 3u) << 4u)) - 32;
        sc_idx = is + 4u;
    } else {
        q = i32((ql_l32 >> 4u) | (((qh_l >> 6u) & 3u) << 4u)) - 32;
        sc_idx = is + 6u;
    }
    var sc: i32 = i32(read_u8(sc_off + sc_o + sc_idx));
    if (sc >= 128) {
        sc = sc - 256;
    }
    return d * f32(sc) * f32(q);
}
"#;

/// `block_q2_K`: mirrors `quant::dequantize_q2_k`. Its sequential form walks
/// two 128-element halves, each re-reading the same 32 `qs` bytes four times
/// at `shift` 0/2/4/6 and splitting each pass into two 16-element sub-blocks
/// — restated per `k`, that is exactly the four-digit decomposition below
/// (`n`, `s`, `h`, `l`), which fixes the scale index and the shift outright.
const Q2_K_COOP_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 84u;
const BLOCK_ELEMS: u32 = 256u;
var<workgroup> shared_vals: array<f32, BLOCK_ELEMS>;
fn dequant_element(byte_offset: u32, k: u32) -> f32 {
    let scales_off = byte_offset;
    let qs_off = byte_offset + 16u;
    let d = f16_to_f32(read_u8(byte_offset + 80u) | (read_u8(byte_offset + 81u) << 8u));
    let dmin = f16_to_f32(read_u8(byte_offset + 82u) | (read_u8(byte_offset + 83u) << 8u));
    let n = k / 128u;
    let r = k % 128u;
    let s = r / 32u;
    let h = (r % 32u) / 16u;
    let l = r % 16u;
    let sc = read_u8(scales_off + n * 8u + s * 2u + h);
    let dl = d * f32(sc & 0xFu);
    let ml = dmin * f32(sc >> 4u);
    let byte = read_u8(qs_off + n * 32u + h * 16u + l);
    return dl * f32((byte >> (2u * s)) & 3u) - ml;
}
"#;

/// `block_q3_K`: mirrors `quant::dequantize_q3_k` — `Q2_K`'s decomposition
/// with two differences. The third quant bit comes from `hmask`, whose byte
/// index deliberately excludes `n` (one 32-byte mask covers all 256 weights,
/// one bit per weight, selected by `m = 1 << (n * 4 + s)`), and it is
/// *inverted*: a set bit means "don't subtract 4".
const Q3_K_COOP_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 110u;
const BLOCK_ELEMS: u32 = 256u;
var<workgroup> shared_vals: array<f32, BLOCK_ELEMS>;
fn dequant_element(byte_offset: u32, k: u32) -> f32 {
    let hmask_off = byte_offset;
    let qs_off = byte_offset + 32u;
    let scales_off = byte_offset + 96u;
    let d_all = f16_to_f32(read_u8(byte_offset + 108u) | (read_u8(byte_offset + 109u) << 8u));
    let n = k / 128u;
    let r = k % 128u;
    let s = r / 32u;
    let h = (r % 32u) / 16u;
    let l = r % 16u;
    let idx = h * 16u + l;
    let m = 1u << (n * 4u + s);
    let dl = d_all * f32(i32(q3k_scale(scales_off, n * 8u + s * 2u + h)) - 32);
    var hi: i32 = 4;
    if ((read_u8(hmask_off + idx) & m) != 0u) {
        hi = 0;
    }
    let q = (read_u8(qs_off + n * 32u + idx) >> (2u * s)) & 3u;
    return dl * f32(i32(q) - hi);
}
"#;

/// `block_iq2_xs`: mirrors `quant::dequantize_iq2_xs`. Each of the 32 `u16`
/// in `qs` covers 8 weights — low 9 bits the lattice point, top 7 the sign
/// pattern — so `k`'s group (`ib32`), which of that group's four lookups
/// (`l`) and which element of the lookup (`j`) is the whole decomposition.
/// `l / 2` picks the scale nibble, since one nibble serves 16 weights.
const IQ2_XS_COOP_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 74u;
const BLOCK_ELEMS: u32 = 256u;
var<workgroup> shared_vals: array<f32, BLOCK_ELEMS>;
fn dequant_element(byte_offset: u32, k: u32) -> f32 {
    let d = f16_to_f32(read_u8(byte_offset) | (read_u8(byte_offset + 1u) << 8u));
    let qs_off = byte_offset + 2u;
    let scales_off = byte_offset + 66u;
    let ib32 = k / 32u;
    let l = (k % 32u) / 8u;
    let j = k % 8u;
    let qo = qs_off + 2u * (4u * ib32 + l);
    let q = read_u8(qo) | (read_u8(qo + 1u) << 8u);
    let sc = (read_u8(scales_off + ib32) >> (4u * (l / 2u))) & 0xFu;
    let db = d * (0.5 + f32(sc)) * 0.25;
    let g = iq_grid8(IQ2XS_GRID_OFF, q & 511u, j);
    return db * f32(g) * iq_sign(iq_ksigns(q >> 9u), j);
}
"#;

/// `block_iq2_xxs`: mirrors `quant::dequantize_iq2_xxs`. The tightest of the
/// `iq2*` formats — no `scales` array at all. Each 32-weight group reads two
/// `u32` out of `qs`: the first is four 8-bit codebook indices, the second
/// four 7-bit `ksigns` indices plus, in its top nibble, the group's scale.
const IQ2_XXS_COOP_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 66u;
const BLOCK_ELEMS: u32 = 256u;
var<workgroup> shared_vals: array<f32, BLOCK_ELEMS>;
fn dequant_element(byte_offset: u32, k: u32) -> f32 {
    let d = f16_to_f32(read_u8(byte_offset) | (read_u8(byte_offset + 1u) << 8u));
    let qs_off = byte_offset + 2u;
    let ib32 = k / 32u;
    let l = (k % 32u) / 8u;
    let j = k % 8u;
    let base = qs_off + 8u * ib32;
    let aux0 = read_u8(base) | (read_u8(base + 1u) << 8u)
        | (read_u8(base + 2u) << 16u) | (read_u8(base + 3u) << 24u);
    let aux1 = read_u8(base + 4u) | (read_u8(base + 5u) << 8u)
        | (read_u8(base + 6u) << 16u) | (read_u8(base + 7u) << 24u);
    let db = d * (0.5 + f32(aux1 >> 28u)) * 0.25;
    let idx = (aux0 >> (8u * l)) & 0xFFu;
    let signs = iq_ksigns((aux1 >> (7u * l)) & 127u);
    let g = iq_grid8(IQ2XXS_GRID_OFF, idx, j);
    return db * f32(g) * iq_sign(signs, j);
}
"#;

/// `block_iq1_s`: mirrors `quant::dequantize_iq1_s`. Each group's `qh` `u16`
/// carries three things — a 3-bit scale (bits 12..15), the sign of the
/// group's `delta` (bit 15), and the high 3 bits of each of its four 11-bit
/// lattice indices (bits 0..9). The grid values are already signed and there
/// is no sign field, hence `iq_grid8_signed` rather than `iq_sign`.
const IQ1_S_COOP_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 50u;
const BLOCK_ELEMS: u32 = 256u;
var<workgroup> shared_vals: array<f32, BLOCK_ELEMS>;
fn dequant_element(byte_offset: u32, k: u32) -> f32 {
    let d = f16_to_f32(read_u8(byte_offset) | (read_u8(byte_offset + 1u) << 8u));
    let qs_off = byte_offset + 2u;
    let qh_off = byte_offset + 34u;
    let ib = k / 32u;
    let l = (k % 32u) / 8u;
    let j = k % 8u;
    let qh = read_u8(qh_off + 2u * ib) | (read_u8(qh_off + 2u * ib + 1u) << 8u);
    let dl = d * f32(2u * ((qh >> 12u) & 7u) + 1u);
    var delta: f32 = IQ1_DELTA;
    if ((qh & 0x8000u) != 0u) {
        delta = -IQ1_DELTA;
    }
    let idx = read_u8(qs_off + 4u * ib + l) | (((qh >> (3u * l)) & 7u) << 8u);
    return dl * (iq_grid8_signed(IQ1S_GRID_OFF, idx, j) + delta);
}
"#;

/// `block_iq1_m`: mirrors `quant::dequantize_iq1_m`. The only quantization
/// here with **no `d` field**: the block's `f16` scale is scattered four
/// nibbles at a time across the top of the four `scales` `u16`s and has to be
/// reassembled before it can be read as a half. Each group also gets *two*
/// 3-bit sub-scales (one per 16 weights) rather than one, and `delta`'s sign
/// moves to two bits of each `qh` byte.
const IQ1_M_COOP_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 56u;
const BLOCK_ELEMS: u32 = 256u;
var<workgroup> shared_vals: array<f32, BLOCK_ELEMS>;
fn iq1m_scale_u16(scales_off: u32, i: u32) -> u32 {
    return read_u8(scales_off + 2u * i) | (read_u8(scales_off + 2u * i + 1u) << 8u);
}
fn dequant_element(byte_offset: u32, k: u32) -> f32 {
    let qs_off = byte_offset;
    let qh_off = byte_offset + 32u;
    let scales_off = byte_offset + 48u;
    let s0 = iq1m_scale_u16(scales_off, 0u);
    let s1 = iq1m_scale_u16(scales_off, 1u);
    let s2 = iq1m_scale_u16(scales_off, 2u);
    let s3 = iq1m_scale_u16(scales_off, 3u);
    let packed = (s0 >> 12u) | ((s1 >> 8u) & 0x00F0u) | ((s2 >> 4u) & 0x0F00u) | (s3 & 0xF000u);
    let d = f16_to_f32(packed);
    let ib = k / 32u;
    let l = (k % 32u) / 8u;
    let j = k % 8u;
    var s: u32 = s0;
    if (ib / 2u == 1u) {
        s = s1;
    } else if (ib / 2u == 2u) {
        s = s2;
    } else if (ib / 2u == 3u) {
        s = s3;
    }
    let shift = 6u * (ib % 2u);
    var sub: u32 = shift;
    if (l >= 2u) {
        sub = shift + 3u;
    }
    let dl = d * f32(2u * ((s >> sub) & 7u) + 1u);
    var qhb: u32 = read_u8(qh_off + 2u * ib);
    if (l >= 2u) {
        qhb = read_u8(qh_off + 2u * ib + 1u);
    }
    var hshift: u32 = 8u;
    var bit: u32 = 0x08u;
    if ((l % 2u) == 1u) {
        hshift = 4u;
        bit = 0x80u;
    }
    let idx = read_u8(qs_off + 4u * ib + l) | ((qhb << hshift) & 0x700u);
    var delta: f32 = IQ1_DELTA;
    if ((qhb & bit) != 0u) {
        delta = -IQ1_DELTA;
    }
    return dl * (iq_grid8_signed(IQ1S_GRID_OFF, idx, j) + delta);
}
"#;

/// `block_iq2_s`: mirrors `quant::dequantize_iq2_s` — `IQ2_XS`'s scales and
/// decomposition, but a 10-bit lattice index (8 bits from `qs`, 2 more from
/// the group's `qh` byte) and an explicit sign byte rather than a 7-bit
/// index into `ksigns`. Those sign bytes live in the *second half* of `qs`,
/// which ggml spells as the alias `signs = qs + QK_K/8` — here, the `+ 32u`.
const IQ2_S_COOP_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 82u;
const BLOCK_ELEMS: u32 = 256u;
var<workgroup> shared_vals: array<f32, BLOCK_ELEMS>;
fn dequant_element(byte_offset: u32, k: u32) -> f32 {
    let d = f16_to_f32(read_u8(byte_offset) | (read_u8(byte_offset + 1u) << 8u));
    let qs_off = byte_offset + 2u;
    let qh_off = byte_offset + 66u;
    let scales_off = byte_offset + 74u;
    let signs_off = qs_off + 32u;
    let ib32 = k / 32u;
    let l = (k % 32u) / 8u;
    let j = k % 8u;
    let qh = read_u8(qh_off + ib32);
    let idx = read_u8(qs_off + 4u * ib32 + l) | ((qh << (8u - 2u * l)) & 0x300u);
    let sc = (read_u8(scales_off + ib32) >> (4u * (l / 2u))) & 0xFu;
    let db = d * (0.5 + f32(sc)) * 0.25;
    let g = iq_grid8(IQ2S_GRID_OFF, idx, j);
    return db * f32(g) * iq_sign(read_u8(signs_off + 4u * ib32 + l), j);
}
"#;

/// `block_iq3_xxs`: mirrors `quant::dequantize_iq3_xxs`. The one `qs` array
/// is two: 64 bytes of lattice indices, then 32 read as eight `u32`, one per
/// group, packing four 7-bit sign fields and a 4-bit scale in the top
/// nibble. The lattice points are 4 elements wide, so an 8-element run is
/// two lookups — `j < 4` picks the first, `j >= 4` the second — while the
/// sign pattern spans all 8 and is indexed by `j` throughout.
const IQ3_XXS_COOP_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 98u;
const BLOCK_ELEMS: u32 = 256u;
var<workgroup> shared_vals: array<f32, BLOCK_ELEMS>;
fn dequant_element(byte_offset: u32, k: u32) -> f32 {
    let d = f16_to_f32(read_u8(byte_offset) | (read_u8(byte_offset + 1u) << 8u));
    let qs_off = byte_offset + 2u;
    let aux_off = qs_off + 64u;
    let ib32 = k / 32u;
    let l = (k % 32u) / 8u;
    let j = k % 8u;
    let ao = aux_off + 4u * ib32;
    let aux32 = read_u8(ao) | (read_u8(ao + 1u) << 8u)
        | (read_u8(ao + 2u) << 16u) | (read_u8(ao + 3u) << 24u);
    let db = d * (0.5 + f32(aux32 >> 28u)) * 0.5;
    let signs = iq_ksigns((aux32 >> (7u * l)) & 127u);
    let idx = read_u8(qs_off + 8u * ib32 + 2u * l + (j >> 2u));
    let g = iq_grid4(IQ3XXS_GRID_OFF, idx, j & 3u);
    return db * f32(g) * iq_sign(signs, j);
}
"#;

/// `block_iq3_s`: mirrors `quant::dequantize_iq3_s`. A 9-bit lattice index —
/// 8 bits from `qs`, the ninth from the group's `qh` byte, a *different* bit
/// per lookup — with signs stored outright and a scale of `1 + 2*s` rather
/// than `IQ2_*`'s `(0.5 + s) * 0.25`. The two 4-element lookups of an
/// 8-element run take their ninth bit from adjacent positions, hence the
/// `8 - 2*l` / `7 - 2*l` shift pair collapsing to one expression in `j`.
const IQ3_S_COOP_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 110u;
const BLOCK_ELEMS: u32 = 256u;
var<workgroup> shared_vals: array<f32, BLOCK_ELEMS>;
fn dequant_element(byte_offset: u32, k: u32) -> f32 {
    let d = f16_to_f32(read_u8(byte_offset) | (read_u8(byte_offset + 1u) << 8u));
    let qs_off = byte_offset + 2u;
    let qh_off = byte_offset + 66u;
    let signs_off = byte_offset + 74u;
    let scales_off = byte_offset + 106u;
    let ib32 = k / 32u;
    let l = (k % 32u) / 8u;
    let j = k % 8u;
    let sc = (read_u8(scales_off + ib32 / 2u) >> (4u * (ib32 % 2u))) & 0xFu;
    let db = d * f32(1u + 2u * sc);
    let hb = read_u8(qh_off + ib32);
    let half = j >> 2u;
    let idx = read_u8(qs_off + 8u * ib32 + 2u * l + half)
        | ((hb << (8u - 2u * l - half)) & 256u);
    let g = iq_grid4(IQ3S_GRID_OFF, idx, j & 3u);
    return db * f32(g) * iq_sign(read_u8(signs_off + 4u * ib32 + l), j);
}
"#;

/// `block_mxfp4`: mirrors `quant::dequantize_mxfp4`. Layout is `IQ4_NL`'s
/// exactly — 16 `qs` bytes, low nibbles the first 16 elements and high
/// nibbles the next 16 — with a **one-byte `e8m0` exponent** in place of the
/// `f16` scale, which is what makes the block 17 bytes rather than 18. The
/// odd size costs nothing: `read_u8` peels bytes out of an `array<u32>` and
/// never required a block to be 4-aligned (see [`PRELUDE`]).
const MXFP4_COOP_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 17u;
const BLOCK_ELEMS: u32 = 32u;
var<workgroup> shared_vals: array<f32, BLOCK_ELEMS>;
fn dequant_element(byte_offset: u32, k: u32) -> f32 {
    let d = mxfp4_scale(read_u8(byte_offset));
    let byte = read_u8(byte_offset + 1u + (k % 16u));
    var nib: u32 = byte & 0xFu;
    if (k >= 16u) {
        nib = byte >> 4u;
    }
    return d * mxfp4_kvalue(nib);
}
"#;

/// `block_iq4_nl`: mirrors `quant::dequantize_iq4_nl` — `Q4_0`'s 18-byte
/// block and low/high-nibble split, with the nibble selecting one of the 16
/// non-uniformly spaced levels in `iq_kvalue` instead of being read as a
/// signed integer offset by 8. The one `IQ*` type whose block is 32
/// elements rather than 256.
const IQ4_NL_COOP_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 18u;
const BLOCK_ELEMS: u32 = 32u;
var<workgroup> shared_vals: array<f32, BLOCK_ELEMS>;
fn dequant_element(byte_offset: u32, k: u32) -> f32 {
    let d = f16_to_f32(read_u8(byte_offset) | (read_u8(byte_offset + 1u) << 8u));
    let byte = read_u8(byte_offset + 2u + (k % 16u));
    var nib: u32 = byte & 0xFu;
    if (k >= 16u) {
        nib = byte >> 4u;
    }
    return d * iq_kvalue(nib);
}
"#;

/// `block_iq4_xs`: mirrors `quant::dequantize_iq4_xs`. The only new type
/// with no codebook of lattice points — a nibble selects one of 16
/// non-uniformly spaced levels. Its 6-bit group scale is split across
/// `scales_l` (4 low bits, two groups per byte) and `scales_h` (2 high bits,
/// eight groups per `u16`), and within a group the low nibbles are the first
/// 16 weights and the high nibbles the next 16 — split halves, not
/// interleaved.
const IQ4_XS_COOP_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 136u;
const BLOCK_ELEMS: u32 = 256u;
var<workgroup> shared_vals: array<f32, BLOCK_ELEMS>;
fn dequant_element(byte_offset: u32, k: u32) -> f32 {
    let d = f16_to_f32(read_u8(byte_offset) | (read_u8(byte_offset + 1u) << 8u));
    let scales_h = read_u8(byte_offset + 2u) | (read_u8(byte_offset + 3u) << 8u);
    let scales_l_off = byte_offset + 4u;
    let qs_off = byte_offset + 8u;
    let ib = k / 32u;
    let r = k % 32u;
    let low = (read_u8(scales_l_off + ib / 2u) >> (4u * (ib % 2u))) & 0xFu;
    let high = (scales_h >> (2u * ib)) & 3u;
    let dl = d * f32(i32(low | (high << 4u)) - 32);
    let byte = read_u8(qs_off + 16u * ib + (r % 16u));
    var nib: u32 = byte & 0xFu;
    if (r >= 16u) {
        nib = byte >> 4u;
    }
    return dl * iq_kvalue(nib);
}
"#;

/// The `dequant_element` (plus `BLOCK_BYTES`/`BLOCK_ELEMS`/`shared_vals`)
/// for `ggml_type`, or `None` if this backend has no shader for it.
///
/// Every dispatch strategy — reduce, thin-tile reduce, coop, coop-tiled —
/// shares one `dequant_element` per type and differs only in the `main` it
/// is concatenated with, so the type coverage lives here once. Four
/// entry points used to each carry their own copy of this match, which is
/// four places a newly supported type has to be remembered in.
/// `block_q4_0` for [`block_hoisted_suffix`]: the whole 32-element block
/// dotted against 32 contiguous activations, with the block's `f16` scale
/// read **once** instead of once per element.
///
/// The nibble split is `Q4_0_COOP_MIDDLE`'s: element `k < 16` is the low
/// nibble of byte `2+k` and element `k+16` the high nibble of the same byte,
/// so one byte read serves two elements — half the loads of the element-wise
/// path even before the header saving.
///
/// `d` is multiplied into each element rather than factored out of the sum,
/// so each *term* is bit-identical to `dequant_element`'s. The **order the
/// terms are summed in still changes** — a lane here accumulates a
/// contiguous run within a block, where the element-wise path accumulates a
/// workgroup-strided one — so this is not bit-identical output, and it does
/// flip a greedy argmax on a near-tie.
///
/// Measured rather than assumed, against an exact `f64` reference at
/// `in_dim = 2816`: worst relative error **6.3e-7 for this kernel against
/// 1.6e-6 for the element-wise path** it replaces (the CPU path is 4.5e-7).
/// It is the *more* accurate of the two, which is the direction a longer
/// contiguous run should go. The cross-check's 1% tolerance is far too loose
/// to have shown that, which is why it was measured separately.
const Q4_0_BLOCK_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 18u;
const BLOCK_ELEMS: u32 = 32u;
const LANES_PER_BLOCK: u32 = 8u;
fn block_dot(byte_offset: u32, x_off: u32, sub: u32) -> f32 {
    let d = f16_to_f32(read_u8(byte_offset) | (read_u8(byte_offset + 1u) << 8u));
    var acc: f32 = 0.0;
    var m: u32 = 0u;
    loop {
        if (m >= 2u) {
            break;
        }
        let j = sub * 2u + m;
        let byte = read_u8(byte_offset + 2u + j);
        acc = acc + (f32(i32(byte & 0xFu) - 8) * d) * x[x_off + j];
        acc = acc + (f32(i32(byte >> 4u) - 8) * d) * x[x_off + 16u + j];
        m = m + 1u;
    }
    return acc;
}
"#;

/// `block_q8_0` for [`block_hoisted_suffix`] — see [`Q4_0_BLOCK_MIDDLE`] for
/// why `d` stays inside the loop. One signed byte per element, and the same
/// `>= 128` sign fold `Q8_0_COOP_MIDDLE` uses.
const Q8_0_BLOCK_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 34u;
const BLOCK_ELEMS: u32 = 32u;
const LANES_PER_BLOCK: u32 = 8u;
fn block_dot(byte_offset: u32, x_off: u32, sub: u32) -> f32 {
    let d = f16_to_f32(read_u16_at(byte_offset));
    let j = sub * 4u;
    // This lane's four weights are four *consecutive* bytes, so they are one
    // unaligned word rather than four byte reads — `read_u32_at`. Sign
    // extension is the same fold `Q8_0_COOP_MIDDLE` does, arranged as a
    // vector so the four multiplies are one `dot`: a byte `b` stands for
    // `b - 256` once it is above 127, which is `(b ^ 0x80) - 128` without a
    // comparison.
    let packed = read_u32_at(byte_offset + 2u + j);
    let bytes = vec4<u32>(packed, packed >> 8u, packed >> 16u, packed >> 24u) & vec4<u32>(0xFFu);
    let w = vec4<f32>(bytes ^ vec4<u32>(0x80u)) - vec4<f32>(128.0);
    let xs = vec4<f32>(x[x_off + j], x[x_off + j + 1u], x[x_off + j + 2u], x[x_off + j + 3u]);
    return dot(w * d, xs);
}
"#;

/// `block_q4_1` for [`block_hoisted_suffix`]: `Q4_0`'s nibble split with a
/// stored per-block minimum added instead of a fixed 8 subtracted.
const Q4_1_BLOCK_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 20u;
const BLOCK_ELEMS: u32 = 32u;
const LANES_PER_BLOCK: u32 = 8u;
fn block_dot(byte_offset: u32, x_off: u32, sub: u32) -> f32 {
    let d = f16_to_f32(read_u8(byte_offset) | (read_u8(byte_offset + 1u) << 8u));
    let mn = f16_to_f32(read_u8(byte_offset + 2u) | (read_u8(byte_offset + 3u) << 8u));
    var acc: f32 = 0.0;
    var m: u32 = 0u;
    loop {
        if (m >= 2u) {
            break;
        }
        let j = sub * 2u + m;
        let byte = read_u8(byte_offset + 4u + j);
        acc = acc + (f32(byte & 0xFu) * d + mn) * x[x_off + j];
        acc = acc + (f32(byte >> 4u) * d + mn) * x[x_off + 16u + j];
        m = m + 1u;
    }
    return acc;
}
"#;

/// `block_q5_0` for [`block_hoisted_suffix`]: [`Q4_0_BLOCK_MIDDLE`]'s
/// nibble split with the fifth bit of each weight taken from the 32-bit
/// `qh` field, which is decoded **once** per block here rather than
/// reassembled from four `read_u8` calls per element.
const Q5_0_BLOCK_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 22u;
const BLOCK_ELEMS: u32 = 32u;
const LANES_PER_BLOCK: u32 = 8u;
fn block_dot(byte_offset: u32, x_off: u32, sub: u32) -> f32 {
    let d = f16_to_f32(read_u8(byte_offset) | (read_u8(byte_offset + 1u) << 8u));
    let qh = read_u8(byte_offset + 2u) | (read_u8(byte_offset + 3u) << 8u)
        | (read_u8(byte_offset + 4u) << 16u) | (read_u8(byte_offset + 5u) << 24u);
    var acc: f32 = 0.0;
    var m: u32 = 0u;
    loop {
        if (m >= 2u) {
            break;
        }
        let j = sub * 2u + m;
        let byte = read_u8(byte_offset + 6u + j);
        let xh_0 = ((qh >> j) << 4u) & 0x10u;
        let xh_1 = (qh >> (j + 12u)) & 0x10u;
        acc = acc + (f32(i32((byte & 0xFu) | xh_0) - 16) * d) * x[x_off + j];
        acc = acc + (f32(i32((byte >> 4u) | xh_1) - 16) * d) * x[x_off + 16u + j];
        m = m + 1u;
    }
    return acc;
}
"#;

/// `block_q5_1` for [`block_hoisted_suffix`]: [`Q5_0_BLOCK_MIDDLE`]'s fifth
/// bit with [`Q4_1_BLOCK_MIDDLE`]'s stored minimum.
const Q5_1_BLOCK_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 24u;
const BLOCK_ELEMS: u32 = 32u;
const LANES_PER_BLOCK: u32 = 8u;
fn block_dot(byte_offset: u32, x_off: u32, sub: u32) -> f32 {
    let d = f16_to_f32(read_u8(byte_offset) | (read_u8(byte_offset + 1u) << 8u));
    let mn = f16_to_f32(read_u8(byte_offset + 2u) | (read_u8(byte_offset + 3u) << 8u));
    let qh = read_u8(byte_offset + 4u) | (read_u8(byte_offset + 5u) << 8u)
        | (read_u8(byte_offset + 6u) << 16u) | (read_u8(byte_offset + 7u) << 24u);
    var acc: f32 = 0.0;
    var m: u32 = 0u;
    loop {
        if (m >= 2u) {
            break;
        }
        let j = sub * 2u + m;
        let byte = read_u8(byte_offset + 8u + j);
        let xh_0 = ((qh >> j) << 4u) & 0x10u;
        let xh_1 = (qh >> (j + 12u)) & 0x10u;
        acc = acc + (f32((byte & 0xFu) | xh_0) * d + mn) * x[x_off + j];
        acc = acc + (f32((byte >> 4u) | xh_1) * d + mn) * x[x_off + 16u + j];
        m = m + 1u;
    }
    return acc;
}
"#;

/// `block_iq4_nl` for [`block_hoisted_suffix`]: [`Q4_0_BLOCK_MIDDLE`]'s
/// 18-byte block and nibble split, the nibble selecting one of 16
/// non-uniformly spaced codebook levels. The one `IQ*` type whose block is
/// 32 elements rather than 256, so it takes the 8-lane geometry the legacy
/// types use rather than the 16-lane one every super-block type below does.
const IQ4_NL_BLOCK_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 18u;
const BLOCK_ELEMS: u32 = 32u;
const LANES_PER_BLOCK: u32 = 8u;
fn block_dot(byte_offset: u32, x_off: u32, sub: u32) -> f32 {
    let d = f16_to_f32(read_u8(byte_offset) | (read_u8(byte_offset + 1u) << 8u));
    var acc: f32 = 0.0;
    var m: u32 = 0u;
    loop {
        if (m >= 2u) {
            break;
        }
        let j = sub * 2u + m;
        let byte = read_u8(byte_offset + 2u + j);
        acc = acc + (d * iq_kvalue(byte & 0xFu)) * x[x_off + j];
        acc = acc + (d * iq_kvalue(byte >> 4u)) * x[x_off + 16u + j];
        m = m + 1u;
    }
    return acc;
}
"#;

/// `block_mxfp4` for [`block_hoisted_suffix`]: [`IQ4_NL_BLOCK_MIDDLE`]'s
/// shape with a 17-byte block and the `e8m0` scale — 8 lanes per block, each
/// taking 2 of the 16 `qs` bytes, exactly as the other 32-element types do.
const MXFP4_BLOCK_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 17u;
const BLOCK_ELEMS: u32 = 32u;
const LANES_PER_BLOCK: u32 = 8u;
fn block_dot(byte_offset: u32, x_off: u32, sub: u32) -> f32 {
    let d = mxfp4_scale(read_u8(byte_offset));
    var acc: f32 = 0.0;
    var m: u32 = 0u;
    loop {
        if (m >= 2u) {
            break;
        }
        let j = sub * 2u + m;
        let byte = read_u8(byte_offset + 1u + j);
        acc = acc + (d * mxfp4_kvalue(byte & 0xFu)) * x[x_off + j];
        acc = acc + (d * mxfp4_kvalue(byte >> 4u)) * x[x_off + 16u + j];
        m = m + 1u;
    }
    return acc;
}
"#;

/// `block_q2_K` for [`block_hoisted_suffix`], and the first of the
/// 256-element super-block types, which all share one geometry:
/// `LANES_PER_BLOCK = 16`, a lane owning 16 **consecutive** elements
/// `k = sub*16 .. sub*16+16`.
///
/// That choice is what makes these middles short. Every K-quant and `IQ*`
/// index decomposition in this file is built from `k / 128`,
/// `(k % 128) / 32`, `(k % 32) / 16`, `k / 32` and `(k % 32) / 8` — and
/// because 16 divides every one of those boundaries, a 16-element run holds
/// all of them constant except the innermost. So the sub-block index, the
/// scale, the shift and the nibble half are decoded once per lane and only
/// the byte offset moves inside the loop, which is the whole point of
/// hoisting. Sixteen lanes per block also keeps `blocks_in_flight` at 4 for
/// a 64-lane workgroup, so four super-blocks stream at once, and adjacent
/// lanes read adjacent (or identical) 16-byte runs — the coalescing shape
/// [`block_hoisted_suffix`] documents as the one that matters.
///
/// Here specifically: a lane's 16 elements share one `n`/`s`/`h`, so the
/// 4-bit scale and 4-bit min are unpacked once and the loop is a
/// shift-and-mask over 16 consecutive `qs` bytes.
const Q2_K_BLOCK_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 84u;
const BLOCK_ELEMS: u32 = 256u;
const LANES_PER_BLOCK: u32 = 16u;
fn block_dot(byte_offset: u32, x_off: u32, sub: u32) -> f32 {
    let scales_off = byte_offset;
    let qs_off = byte_offset + 16u;
    let d = f16_to_f32(read_u8(byte_offset + 80u) | (read_u8(byte_offset + 81u) << 8u));
    let dmin = f16_to_f32(read_u8(byte_offset + 82u) | (read_u8(byte_offset + 83u) << 8u));
    let n = sub / 8u;
    let s = (sub % 8u) / 2u;
    let h = sub % 2u;
    let sc = read_u8(scales_off + n * 8u + s * 2u + h);
    let dl = d * f32(sc & 0xFu);
    let ml = dmin * f32(sc >> 4u);
    let base = qs_off + n * 32u + h * 16u;
    let x_lane = x_off + sub * 16u;
    var acc: f32 = 0.0;
    var l: u32 = 0u;
    loop {
        if (l >= 16u) {
            break;
        }
        let byte = read_u8(base + l);
        acc = acc + (dl * f32((byte >> (2u * s)) & 3u) - ml) * x[x_lane + l];
        l = l + 1u;
    }
    return acc;
}
"#;

/// `block_q3_K` for [`block_hoisted_suffix`]. [`Q2_K_BLOCK_MIDDLE`]'s
/// decomposition with the third quant bit from `hmask` — whose byte index
/// deliberately excludes `n` (one 32-byte mask covers all 256 weights) and
/// whose bit is *inverted*: set means "don't subtract 4". The 6-bit scale is
/// still hoisted once per lane.
const Q3_K_BLOCK_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 110u;
const BLOCK_ELEMS: u32 = 256u;
const LANES_PER_BLOCK: u32 = 16u;
fn block_dot(byte_offset: u32, x_off: u32, sub: u32) -> f32 {
    let hmask_off = byte_offset;
    let qs_off = byte_offset + 32u;
    let scales_off = byte_offset + 96u;
    let d_all = f16_to_f32(read_u8(byte_offset + 108u) | (read_u8(byte_offset + 109u) << 8u));
    let n = sub / 8u;
    let s = (sub % 8u) / 2u;
    let h = sub % 2u;
    let m = 1u << (n * 4u + s);
    let dl = d_all * f32(i32(q3k_scale(scales_off, n * 8u + s * 2u + h)) - 32);
    let x_lane = x_off + sub * 16u;
    var acc: f32 = 0.0;
    var l: u32 = 0u;
    loop {
        if (l >= 16u) {
            break;
        }
        let idx = h * 16u + l;
        var hi: i32 = 4;
        if ((read_u8(hmask_off + idx) & m) != 0u) {
            hi = 0;
        }
        let q = (read_u8(qs_off + n * 32u + idx) >> (2u * s)) & 3u;
        acc = acc + (dl * f32(i32(q) - hi)) * x[x_lane + l];
        l = l + 1u;
    }
    return acc;
}
"#;

/// `block_iq4_xs` for [`block_hoisted_suffix`]. A lane's 16 elements are one
/// whole nibble half of one 32-element group, so the group's 6-bit scale —
/// split across `scales_l` and `scales_h` — is assembled once and the loop
/// reads 16 contiguous `qs` bytes, taking the same nibble from each.
const IQ4_XS_BLOCK_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 136u;
const BLOCK_ELEMS: u32 = 256u;
const LANES_PER_BLOCK: u32 = 16u;
fn block_dot(byte_offset: u32, x_off: u32, sub: u32) -> f32 {
    let d = f16_to_f32(read_u8(byte_offset) | (read_u8(byte_offset + 1u) << 8u));
    let scales_h = read_u8(byte_offset + 2u) | (read_u8(byte_offset + 3u) << 8u);
    let scales_l_off = byte_offset + 4u;
    let qs_off = byte_offset + 8u;
    let ib = sub / 2u;
    let high_half = sub % 2u;
    let low = (read_u8(scales_l_off + ib / 2u) >> (4u * (ib % 2u))) & 0xFu;
    let high = (scales_h >> (2u * ib)) & 3u;
    let dl = d * f32(i32(low | (high << 4u)) - 32);
    let base = qs_off + 16u * ib;
    let x_lane = x_off + sub * 16u;
    var acc: f32 = 0.0;
    var l: u32 = 0u;
    loop {
        if (l >= 16u) {
            break;
        }
        let byte = read_u8(base + l);
        var nib: u32 = byte & 0xFu;
        if (high_half == 1u) {
            nib = byte >> 4u;
        }
        acc = acc + (dl * iq_kvalue(nib)) * x[x_lane + l];
        l = l + 1u;
    }
    return acc;
}
"#;

/// `block_iq2_xs` for [`block_hoisted_suffix`]. A lane's 16 elements are
/// exactly two of the group's four 8-element lattice lookups, so the outer
/// loop runs twice and each iteration decodes one `u16` index, one scale
/// nibble and one 7-bit sign field before its eight codebook bytes.
const IQ2_XS_BLOCK_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 74u;
const BLOCK_ELEMS: u32 = 256u;
const LANES_PER_BLOCK: u32 = 16u;
fn block_dot(byte_offset: u32, x_off: u32, sub: u32) -> f32 {
    let d = f16_to_f32(read_u8(byte_offset) | (read_u8(byte_offset + 1u) << 8u));
    let qs_off = byte_offset + 2u;
    let scales_off = byte_offset + 66u;
    let ib32 = sub / 2u;
    let l0 = (sub % 2u) * 2u;
    let sc_byte = read_u8(scales_off + ib32);
    let x_lane = x_off + sub * 16u;
    var acc: f32 = 0.0;
    var lh: u32 = 0u;
    loop {
        if (lh >= 2u) {
            break;
        }
        let l = l0 + lh;
        let qo = qs_off + 2u * (4u * ib32 + l);
        let q = read_u8(qo) | (read_u8(qo + 1u) << 8u);
        let sc = (sc_byte >> (4u * (l / 2u))) & 0xFu;
        let db = d * (0.5 + f32(sc)) * 0.25;
        let signs = iq_ksigns(q >> 9u);
        let grid = q & 511u;
        var j: u32 = 0u;
        loop {
            if (j >= 8u) {
                break;
            }
            let g = iq_grid8(IQ2XS_GRID_OFF, grid, j);
            acc = acc + (db * f32(g) * iq_sign(signs, j)) * x[x_lane + lh * 8u + j];
            j = j + 1u;
        }
        lh = lh + 1u;
    }
    return acc;
}
"#;

/// `block_iq2_s` for [`block_hoisted_suffix`]: [`IQ2_XS_BLOCK_MIDDLE`]'s
/// two-lookup shape with a 10-bit lattice index (two bits from the group's
/// `qh` byte) and an explicit sign byte from the second half of `qs`.
const IQ2_S_BLOCK_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 82u;
const BLOCK_ELEMS: u32 = 256u;
const LANES_PER_BLOCK: u32 = 16u;
fn block_dot(byte_offset: u32, x_off: u32, sub: u32) -> f32 {
    let d = f16_to_f32(read_u8(byte_offset) | (read_u8(byte_offset + 1u) << 8u));
    let qs_off = byte_offset + 2u;
    let qh_off = byte_offset + 66u;
    let scales_off = byte_offset + 74u;
    let signs_off = qs_off + 32u;
    let ib32 = sub / 2u;
    let l0 = (sub % 2u) * 2u;
    let qh = read_u8(qh_off + ib32);
    let sc_byte = read_u8(scales_off + ib32);
    let x_lane = x_off + sub * 16u;
    var acc: f32 = 0.0;
    var lh: u32 = 0u;
    loop {
        if (lh >= 2u) {
            break;
        }
        let l = l0 + lh;
        let idx = read_u8(qs_off + 4u * ib32 + l) | ((qh << (8u - 2u * l)) & 0x300u);
        let sc = (sc_byte >> (4u * (l / 2u))) & 0xFu;
        let db = d * (0.5 + f32(sc)) * 0.25;
        let signs = read_u8(signs_off + 4u * ib32 + l);
        var j: u32 = 0u;
        loop {
            if (j >= 8u) {
                break;
            }
            let g = iq_grid8(IQ2S_GRID_OFF, idx, j);
            acc = acc + (db * f32(g) * iq_sign(signs, j)) * x[x_lane + lh * 8u + j];
            j = j + 1u;
        }
        lh = lh + 1u;
    }
    return acc;
}
"#;

/// `block_iq3_xxs` for [`block_hoisted_suffix`]. The group's `aux32` — four
/// 7-bit sign fields and a 4-bit scale in its top nibble — is read once per
/// lane instead of once per element. Its lattice points are 4 elements wide,
/// so an 8-element run is two index bytes, selected by `j >> 2`.
const IQ3_XXS_BLOCK_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 98u;
const BLOCK_ELEMS: u32 = 256u;
const LANES_PER_BLOCK: u32 = 16u;
fn block_dot(byte_offset: u32, x_off: u32, sub: u32) -> f32 {
    let d = f16_to_f32(read_u8(byte_offset) | (read_u8(byte_offset + 1u) << 8u));
    let qs_off = byte_offset + 2u;
    let aux_off = qs_off + 64u;
    let ib32 = sub / 2u;
    let l0 = (sub % 2u) * 2u;
    let ao = aux_off + 4u * ib32;
    let aux32 = read_u8(ao) | (read_u8(ao + 1u) << 8u)
        | (read_u8(ao + 2u) << 16u) | (read_u8(ao + 3u) << 24u);
    let db = d * (0.5 + f32(aux32 >> 28u)) * 0.5;
    let x_lane = x_off + sub * 16u;
    var acc: f32 = 0.0;
    var lh: u32 = 0u;
    loop {
        if (lh >= 2u) {
            break;
        }
        let l = l0 + lh;
        let signs = iq_ksigns((aux32 >> (7u * l)) & 127u);
        let idx_lo = read_u8(qs_off + 8u * ib32 + 2u * l);
        let idx_hi = read_u8(qs_off + 8u * ib32 + 2u * l + 1u);
        var j: u32 = 0u;
        loop {
            if (j >= 8u) {
                break;
            }
            var idx: u32 = idx_lo;
            if (j >= 4u) {
                idx = idx_hi;
            }
            let g = iq_grid4(IQ3XXS_GRID_OFF, idx, j & 3u);
            acc = acc + (db * f32(g) * iq_sign(signs, j)) * x[x_lane + lh * 8u + j];
            j = j + 1u;
        }
        lh = lh + 1u;
    }
    return acc;
}
"#;

/// `block_iq3_s` for [`block_hoisted_suffix`]: [`IQ3_XXS_BLOCK_MIDDLE`]'s
/// two-lookup shape with a 9-bit lattice index whose ninth bit comes from a
/// *different* bit of the group's `qh` byte per lookup half, signs stored
/// outright, and a `1 + 2*s` scale rather than `(0.5 + s) * 0.25`.
const IQ3_S_BLOCK_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 110u;
const BLOCK_ELEMS: u32 = 256u;
const LANES_PER_BLOCK: u32 = 16u;
fn block_dot(byte_offset: u32, x_off: u32, sub: u32) -> f32 {
    let d = f16_to_f32(read_u8(byte_offset) | (read_u8(byte_offset + 1u) << 8u));
    let qs_off = byte_offset + 2u;
    let qh_off = byte_offset + 66u;
    let signs_off = byte_offset + 74u;
    let scales_off = byte_offset + 106u;
    let ib32 = sub / 2u;
    let l0 = (sub % 2u) * 2u;
    let sc = (read_u8(scales_off + ib32 / 2u) >> (4u * (ib32 % 2u))) & 0xFu;
    let db = d * f32(1u + 2u * sc);
    let hb = read_u8(qh_off + ib32);
    let x_lane = x_off + sub * 16u;
    var acc: f32 = 0.0;
    var lh: u32 = 0u;
    loop {
        if (lh >= 2u) {
            break;
        }
        let l = l0 + lh;
        let signs = read_u8(signs_off + 4u * ib32 + l);
        let idx_lo = read_u8(qs_off + 8u * ib32 + 2u * l) | ((hb << (8u - 2u * l)) & 256u);
        let idx_hi = read_u8(qs_off + 8u * ib32 + 2u * l + 1u)
            | ((hb << (7u - 2u * l)) & 256u);
        var j: u32 = 0u;
        loop {
            if (j >= 8u) {
                break;
            }
            var idx: u32 = idx_lo;
            if (j >= 4u) {
                idx = idx_hi;
            }
            let g = iq_grid4(IQ3S_GRID_OFF, idx, j & 3u);
            acc = acc + (db * f32(g) * iq_sign(signs, j)) * x[x_lane + lh * 8u + j];
            j = j + 1u;
        }
        lh = lh + 1u;
    }
    return acc;
}
"#;

/// `block_iq2_xxs` for [`block_hoisted_suffix`]: [`IQ2_XS_BLOCK_MIDDLE`]'s
/// two-lookup shape, with both `u32` of the group's `qs` pair — the four
/// codebook indices and the four sign fields plus scale — decoded once per
/// lane instead of once per element. That is eight `read_u8` calls saved
/// fifteen times over per lane.
const IQ2_XXS_BLOCK_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 66u;
const BLOCK_ELEMS: u32 = 256u;
const LANES_PER_BLOCK: u32 = 16u;
fn block_dot(byte_offset: u32, x_off: u32, sub: u32) -> f32 {
    let d = f16_to_f32(read_u8(byte_offset) | (read_u8(byte_offset + 1u) << 8u));
    let qs_off = byte_offset + 2u;
    let ib32 = sub / 2u;
    let l0 = (sub % 2u) * 2u;
    let base = qs_off + 8u * ib32;
    let aux0 = read_u8(base) | (read_u8(base + 1u) << 8u)
        | (read_u8(base + 2u) << 16u) | (read_u8(base + 3u) << 24u);
    let aux1 = read_u8(base + 4u) | (read_u8(base + 5u) << 8u)
        | (read_u8(base + 6u) << 16u) | (read_u8(base + 7u) << 24u);
    let db = d * (0.5 + f32(aux1 >> 28u)) * 0.25;
    let x_lane = x_off + sub * 16u;
    var acc: f32 = 0.0;
    var lh: u32 = 0u;
    loop {
        if (lh >= 2u) {
            break;
        }
        let l = l0 + lh;
        let idx = (aux0 >> (8u * l)) & 0xFFu;
        let signs = iq_ksigns((aux1 >> (7u * l)) & 127u);
        var j: u32 = 0u;
        loop {
            if (j >= 8u) {
                break;
            }
            let g = iq_grid8(IQ2XXS_GRID_OFF, idx, j);
            acc = acc + (db * f32(g) * iq_sign(signs, j)) * x[x_lane + lh * 8u + j];
            j = j + 1u;
        }
        lh = lh + 1u;
    }
    return acc;
}
"#;

/// `block_iq1_s` for [`block_hoisted_suffix`]. A lane's 16 elements are two
/// of the group's four lattice lookups, so the group's `qh` `u16` — scale,
/// delta sign and index-high bits all at once — is decoded once rather than
/// sixteen times.
const IQ1_S_BLOCK_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 50u;
const BLOCK_ELEMS: u32 = 256u;
const LANES_PER_BLOCK: u32 = 16u;
fn block_dot(byte_offset: u32, x_off: u32, sub: u32) -> f32 {
    let d = f16_to_f32(read_u8(byte_offset) | (read_u8(byte_offset + 1u) << 8u));
    let qs_off = byte_offset + 2u;
    let qh_off = byte_offset + 34u;
    let ib = sub / 2u;
    let l0 = (sub % 2u) * 2u;
    let qh = read_u8(qh_off + 2u * ib) | (read_u8(qh_off + 2u * ib + 1u) << 8u);
    let dl = d * f32(2u * ((qh >> 12u) & 7u) + 1u);
    var delta: f32 = IQ1_DELTA;
    if ((qh & 0x8000u) != 0u) {
        delta = -IQ1_DELTA;
    }
    let x_lane = x_off + sub * 16u;
    var acc: f32 = 0.0;
    var lh: u32 = 0u;
    loop {
        if (lh >= 2u) {
            break;
        }
        let l = l0 + lh;
        let idx = read_u8(qs_off + 4u * ib + l) | (((qh >> (3u * l)) & 7u) << 8u);
        var j: u32 = 0u;
        loop {
            if (j >= 8u) {
                break;
            }
            acc = acc + (dl * (iq_grid8_signed(IQ1S_GRID_OFF, idx, j) + delta))
                * x[x_lane + lh * 8u + j];
            j = j + 1u;
        }
        lh = lh + 1u;
    }
    return acc;
}
"#;

/// `block_iq1_m` for [`block_hoisted_suffix`]. The scattered `f16` block
/// scale — four nibbles across the top of four `scales` `u16`s — costs eight
/// `read_u8` calls and a reassembly to recover, and the element-wise path
/// pays that **per element**. Here it is paid once per lane, which is the
/// largest header saving of any type in this file.
const IQ1_M_BLOCK_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 56u;
const BLOCK_ELEMS: u32 = 256u;
const LANES_PER_BLOCK: u32 = 16u;
fn iq1m_scale_u16(scales_off: u32, i: u32) -> u32 {
    return read_u8(scales_off + 2u * i) | (read_u8(scales_off + 2u * i + 1u) << 8u);
}
fn block_dot(byte_offset: u32, x_off: u32, sub: u32) -> f32 {
    let qs_off = byte_offset;
    let qh_off = byte_offset + 32u;
    let scales_off = byte_offset + 48u;
    let s0 = iq1m_scale_u16(scales_off, 0u);
    let s1 = iq1m_scale_u16(scales_off, 1u);
    let s2 = iq1m_scale_u16(scales_off, 2u);
    let s3 = iq1m_scale_u16(scales_off, 3u);
    let packed = (s0 >> 12u) | ((s1 >> 8u) & 0x00F0u) | ((s2 >> 4u) & 0x0F00u) | (s3 & 0xF000u);
    let d = f16_to_f32(packed);
    let ib = sub / 2u;
    let l0 = (sub % 2u) * 2u;
    var s: u32 = s0;
    if (ib / 2u == 1u) {
        s = s1;
    } else if (ib / 2u == 2u) {
        s = s2;
    } else if (ib / 2u == 3u) {
        s = s3;
    }
    let shift = 6u * (ib % 2u);
    let x_lane = x_off + sub * 16u;
    var acc: f32 = 0.0;
    var lh: u32 = 0u;
    loop {
        if (lh >= 2u) {
            break;
        }
        let l = l0 + lh;
        var sub_shift: u32 = shift;
        if (l >= 2u) {
            sub_shift = shift + 3u;
        }
        let dl = d * f32(2u * ((s >> sub_shift) & 7u) + 1u);
        var qhb: u32 = read_u8(qh_off + 2u * ib);
        if (l >= 2u) {
            qhb = read_u8(qh_off + 2u * ib + 1u);
        }
        var hshift: u32 = 8u;
        var bit: u32 = 0x08u;
        if ((l % 2u) == 1u) {
            hshift = 4u;
            bit = 0x80u;
        }
        let idx = read_u8(qs_off + 4u * ib + l) | ((qhb << hshift) & 0x700u);
        var delta: f32 = IQ1_DELTA;
        if ((qhb & bit) != 0u) {
            delta = -IQ1_DELTA;
        }
        var j: u32 = 0u;
        loop {
            if (j >= 8u) {
                break;
            }
            acc = acc + (dl * (iq_grid8_signed(IQ1S_GRID_OFF, idx, j) + delta))
                * x[x_lane + lh * 8u + j];
            j = j + 1u;
        }
        lh = lh + 1u;
    }
    return acc;
}
"#;

/// [`PRELUDE`] with the activations bound as `vec4`s: the word-reading
/// `block_dot`s below take four activations per load, so their prelude
/// binds `x` as `array<vec4<f32>>` (`xv`). Nothing else changes — the
/// weight buffer is still `array<u32>` and `read_u8`/`read_u32_at` still
/// apply — and the block-hoisted `main` never touches `x` itself.
const PRELUDE_XV: &str = r#"
struct Meta {
    in_dim: u32,
    out_dim: u32,
    n_tokens: u32,
    row_bytes: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> xv2: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> params: Meta;

// Four activations from the pair view.
fn xv(i: u32) -> vec4<f32> {
    return vec4<f32>(xv2[2u * i], xv2[2u * i + 1u]);
}

fn read_u8(byte_offset: u32) -> u32 {
    let word = weights[byte_offset >> 2u];
    let shift = (byte_offset & 3u) * 8u;
    return (word >> shift) & 0xFFu;
}

fn read_u32_at(byte_offset: u32) -> u32 {
    let index = byte_offset >> 2u;
    let shift = (byte_offset & 3u) * 8u;
    let lo = weights[index] >> shift;
    let hi = weights[index + 1u] << ((32u - shift) & 31u);
    return lo | select(hi, 0u, shift == 0u);
}

fn read_u16_at(byte_offset: u32) -> u32 {
    return read_u32_at(byte_offset) & 0xFFFFu;
}

fn f16_to_f32(bits: u32) -> f32 {
    return unpack2x16float(bits & 0xFFFFu).x;
}

// The four bytes of a word as floats.
fn unpack4_f(w: u32) -> vec4<f32> {
    return vec4<f32>(f32(w & 0xFFu), f32((w >> 8u) & 0xFFu), f32((w >> 16u) & 0xFFu), f32(w >> 24u));
}

// Bits 0..4 of `s` as a lane sign: +1 where clear, -1 where set.
fn sign4(s: u32) -> vec4<f32> {
    return vec4<f32>(1.0) - 2.0 * vec4<f32>(f32(s & 1u), f32((s >> 1u) & 1u), f32((s >> 2u) & 1u), f32((s >> 3u) & 1u));
}
"#;

/// `block_q2_K` for [`block_hoisted_suffix`], read as **words**: a lane's
/// 16 elements are sixteen consecutive 2-bit fields at one shift in one
/// 128-element group, so they are four words masked once each, against four
/// `vec4` activations — no loop, no byte reads. The block is 84 bytes, so
/// every field is word-aligned.
const Q2_K_WORDS_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 84u;
const BLOCK_ELEMS: u32 = 256u;
const LANES_PER_BLOCK: u32 = 16u;
// Lane `sub` = (v_im, v_in): the 128-element half `v_im`, and within it the
// element pairs `l0, l0 + 1` and `l0 + 16, l0 + 17` of every one of the four
// 32-element sub-blocks (`l0 = 2 * v_in`) — sixteen elements that are the
// four 2-bit fields of one word, whose bytes are `q_offset, +1, +16, +17`.
fn block_dot(byte_offset: u32, x_off: u32, sub: u32) -> f32 {
    let w0 = byte_offset / 4u;
    let hdr = weights[w0 + 20u];
    let d = f16_to_f32(hdr & 0xFFFFu);
    let dmin = f16_to_f32(hdr >> 16u);
    let v_im = sub / 8u;
    let l0 = 2u * (sub % 8u);
    // This half's eight scale bytes: two words.
    let sw0 = weights[w0 + 2u * v_im];
    let sw1 = weights[w0 + 2u * v_im + 1u];
    // qs bytes 32·v_im + l0 .. +2 and +16 .. +18: two halves of two words.
    let qw = w0 + 4u + 8u * v_im + l0 / 4u;
    let half_sel = 16u * ((l0 / 2u) % 2u);
    let lo = (weights[qw] >> half_sel) & 0xFFFFu;
    let hi = (weights[qw + 4u] >> half_sel) & 0xFFFFu;
    let q = lo | (hi << 16u);
    let q0 = unpack4_f(q & 0x03030303u);
    let q2 = unpack4_f((q >> 2u) & 0x03030303u);
    let q4 = unpack4_f((q >> 4u) & 0x03030303u);
    let q6 = unpack4_f((q >> 6u) & 0x03030303u);
    // Activations: for sub-block `s`, elements `y + 32s, +1` and `+16, +17`.
    let y = (x_off + 128u * v_im + l0) / 2u;
    let b0 = xv2[y];
    let b16 = xv2[y + 8u];
    let b32 = xv2[y + 16u];
    let b48 = xv2[y + 24u];
    let b64 = xv2[y + 32u];
    let b80 = xv2[y + 40u];
    let b96 = xv2[y + 48u];
    let b112 = xv2[y + 56u];
    // Scale `8·v_im + 2s` for the low pair, `+1` for the high pair.
    let sc0 = unpack4_f(sw0 & 0x0F0F0F0Fu);
    let sc1 = unpack4_f(sw1 & 0x0F0F0F0Fu);
    let mn0 = unpack4_f((sw0 >> 4u) & 0x0F0F0F0Fu);
    let mn1 = unpack4_f((sw1 >> 4u) & 0x0F0F0F0Fu);
    var sum1: f32 = dot(vec4<f32>(b0, b16) * vec4<f32>(sc0.x, sc0.x, sc0.y, sc0.y), q0);
    sum1 = sum1 + dot(vec4<f32>(b32, b48) * vec4<f32>(sc0.z, sc0.z, sc0.w, sc0.w), q2);
    sum1 = sum1 + dot(vec4<f32>(b64, b80) * vec4<f32>(sc1.x, sc1.x, sc1.y, sc1.y), q4);
    sum1 = sum1 + dot(vec4<f32>(b96, b112) * vec4<f32>(sc1.z, sc1.z, sc1.w, sc1.w), q6);
    var sum2: f32 = dot(vec4<f32>(b0, b16), vec4<f32>(mn0.x, mn0.x, mn0.y, mn0.y));
    sum2 = sum2 + dot(vec4<f32>(b32, b48), vec4<f32>(mn0.z, mn0.z, mn0.w, mn0.w));
    sum2 = sum2 + dot(vec4<f32>(b64, b80), vec4<f32>(mn1.x, mn1.x, mn1.y, mn1.y));
    sum2 = sum2 + dot(vec4<f32>(b96, b112), vec4<f32>(mn1.z, mn1.z, mn1.w, mn1.w));
    return d * sum1 - dmin * sum2;
}
"#;

/// See `block_hoisted_words_middle`'s `ORANGU_Q2K_STUB`.
const Q2_K_STUB_LOADS_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 84u;
const BLOCK_ELEMS: u32 = 256u;
const LANES_PER_BLOCK: u32 = 16u;
fn block_dot(byte_offset: u32, x_off: u32, sub: u32) -> f32 {
    let w0 = byte_offset / 4u;
    let hdr = weights[w0 + 20u];
    let v_im = sub / 8u;
    let l0 = 2u * (sub % 8u);
    let sw0 = weights[w0 + 2u * v_im];
    let sw1 = weights[w0 + 2u * v_im + 1u];
    let qw = w0 + 4u + 8u * v_im + l0 / 4u;
    let lo = weights[qw];
    let hi = weights[qw + 4u];
    let y = (x_off + 128u * v_im + l0) / 2u;
    let b0 = xv2[y];
    let b16 = xv2[y + 8u];
    let b32 = xv2[y + 16u];
    let b48 = xv2[y + 24u];
    let b64 = xv2[y + 32u];
    let b80 = xv2[y + 40u];
    let b96 = xv2[y + 48u];
    let b112 = xv2[y + 56u];
    let w = f32(hdr ^ sw0 ^ sw1 ^ lo ^ hi);
    return w * 1e-9 + b0.x + b16.y + b32.x + b48.y + b64.x + b80.y + b96.x + b112.y;
}
"#;

/// See `block_hoisted_words_middle`'s `ORANGU_Q2K_STUB`.
const Q2_K_STUB_ALU_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 84u;
const BLOCK_ELEMS: u32 = 256u;
const LANES_PER_BLOCK: u32 = 16u;
fn block_dot(byte_offset: u32, x_off: u32, sub: u32) -> f32 {
    let hdr = byte_offset * 2654435761u + sub;
    let d = f16_to_f32(hdr & 0x3FFFu);
    let dmin = f16_to_f32((hdr >> 16u) & 0x3FFFu);
    let v_im = sub / 8u;
    let l0 = 2u * (sub % 8u);
    let sw0 = hdr * 3u + 7u;
    let sw1 = hdr * 5u + 11u;
    let q = hdr * 7u + 13u;
    let q0 = unpack4_f(q & 0x03030303u);
    let q2 = unpack4_f((q >> 2u) & 0x03030303u);
    let q4 = unpack4_f((q >> 4u) & 0x03030303u);
    let q6 = unpack4_f((q >> 6u) & 0x03030303u);
    let y = (x_off + 128u * v_im + l0) / 2u;
    let b0 = xv2[y];
    let b16 = xv2[y + 8u];
    let b32 = xv2[y + 16u];
    let b48 = xv2[y + 24u];
    let b64 = xv2[y + 32u];
    let b80 = xv2[y + 40u];
    let b96 = xv2[y + 48u];
    let b112 = xv2[y + 56u];
    let sc0 = unpack4_f(sw0 & 0x0F0F0F0Fu);
    let sc1 = unpack4_f(sw1 & 0x0F0F0F0Fu);
    let mn0 = unpack4_f((sw0 >> 4u) & 0x0F0F0F0Fu);
    let mn1 = unpack4_f((sw1 >> 4u) & 0x0F0F0F0Fu);
    var sum1: f32 = dot(vec4<f32>(b0, b16) * vec4<f32>(sc0.x, sc0.x, sc0.y, sc0.y), q0);
    sum1 = sum1 + dot(vec4<f32>(b32, b48) * vec4<f32>(sc0.z, sc0.z, sc0.w, sc0.w), q2);
    sum1 = sum1 + dot(vec4<f32>(b64, b80) * vec4<f32>(sc1.x, sc1.x, sc1.y, sc1.y), q4);
    sum1 = sum1 + dot(vec4<f32>(b96, b112) * vec4<f32>(sc1.z, sc1.z, sc1.w, sc1.w), q6);
    var sum2: f32 = dot(vec4<f32>(b0, b16), vec4<f32>(mn0.x, mn0.x, mn0.y, mn0.y));
    sum2 = sum2 + dot(vec4<f32>(b32, b48), vec4<f32>(mn0.z, mn0.z, mn0.w, mn0.w));
    sum2 = sum2 + dot(vec4<f32>(b64, b80), vec4<f32>(mn1.x, mn1.x, mn1.y, mn1.y));
    sum2 = sum2 + dot(vec4<f32>(b96, b112), vec4<f32>(mn1.z, mn1.z, mn1.w, mn1.w));
    return d * sum1 - dmin * sum2;
}
"#;

/// `block_q3_K` for [`block_hoisted_suffix`], read as words: the 2-bit
/// fields as in `Q2_K`, the third bit from the `hmask` words (bit `4n + s`
/// of each byte, set meaning "no `-4`"), the 6-bit scale assembled once.
/// 110-byte blocks start at either word parity, so the reads go through
/// `read_u32_at`.
const Q3_K_WORDS_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 110u;
const BLOCK_ELEMS: u32 = 256u;
const LANES_PER_BLOCK: u32 = 16u;
// A `u16` at an even byte offset: one load, whichever word parity the
// block starts at.
fn read_u16_even(byte_offset: u32) -> u32 {
    return (weights[byte_offset >> 2u] >> ((byte_offset & 2u) * 8u)) & 0xFFFFu;
}
// Lane `sub` = (v_im, v_in), the same element mapping as `Q2_K`: sixteen
// elements from one `qs` word, their third bits from one `hmask` word (bit
// `4·v_im + s` of each byte), the eight scales of the half from its three
// scale words.
fn block_dot(byte_offset: u32, x_off: u32, sub: u32) -> f32 {
    let d_all = f16_to_f32(read_u16_even(byte_offset + 108u));
    let v_im = sub / 8u;
    let l0 = 2u * (sub % 8u);
    let qo = byte_offset + 32u + 32u * v_im + l0;
    let q = read_u16_even(qo) | (read_u16_even(qo + 16u) << 16u);
    let ho = byte_offset + l0;
    let hm = read_u16_even(ho) | (read_u16_even(ho + 16u) << 16u);
    // Scales `8·v_im + 2s + half`: low nibbles of bytes `0..8` (high for the
    // second half's eight), top two bits from bytes `8..12` at `2·(i / 4)`.
    let so = byte_offset + 96u;
    let sa = read_u16_even(so) | (read_u16_even(so + 2u) << 16u);
    let sb = read_u16_even(so + 4u) | (read_u16_even(so + 6u) << 16u);
    let sc = read_u16_even(so + 8u) | (read_u16_even(so + 10u) << 16u);
    let nib = 4u * v_im;
    let lo0 = (sa >> nib) & 0x0F0F0F0Fu;
    let lo1 = (sb >> nib) & 0x0F0F0F0Fu;
    let hi0 = (sc >> nib) & 0x03030303u;
    let hi1 = (sc >> (nib + 2u)) & 0x03030303u;
    let s0 = unpack4_f(lo0 | (hi0 << 4u)) - vec4<f32>(32.0);
    let s1 = unpack4_f(lo1 | (hi1 << 4u)) - vec4<f32>(32.0);
    let m = nib;
    let four = vec4<f32>(4.0);
    let q0 = unpack4_f(q & 0x03030303u) - four + 4.0 * unpack4_f((hm >> m) & 0x01010101u);
    let q2 = unpack4_f((q >> 2u) & 0x03030303u) - four + 4.0 * unpack4_f((hm >> (m + 1u)) & 0x01010101u);
    let q4 = unpack4_f((q >> 4u) & 0x03030303u) - four + 4.0 * unpack4_f((hm >> (m + 2u)) & 0x01010101u);
    let q6 = unpack4_f((q >> 6u) & 0x03030303u) - four + 4.0 * unpack4_f((hm >> (m + 3u)) & 0x01010101u);
    let y = (x_off + 128u * v_im + l0) / 2u;
    let b0 = xv2[y];
    let b16 = xv2[y + 8u];
    let b32 = xv2[y + 16u];
    let b48 = xv2[y + 24u];
    let b64 = xv2[y + 32u];
    let b80 = xv2[y + 40u];
    let b96 = xv2[y + 48u];
    let b112 = xv2[y + 56u];
    var sum: f32 = dot(vec4<f32>(b0, b16) * vec4<f32>(s0.x, s0.x, s0.y, s0.y), q0);
    sum = sum + dot(vec4<f32>(b32, b48) * vec4<f32>(s0.z, s0.z, s0.w, s0.w), q2);
    sum = sum + dot(vec4<f32>(b64, b80) * vec4<f32>(s1.x, s1.x, s1.y, s1.y), q4);
    sum = sum + dot(vec4<f32>(b96, b112) * vec4<f32>(s1.z, s1.z, s1.w, s1.w), q6);
    return d_all * sum;
}
"#;

/// `block_iq4_xs` for [`block_hoisted_suffix`], read as words: a lane's 16
/// elements are one nibble half of a 32-element group's 16 bytes — four
/// words, each nibble through the value table.
const IQ4_XS_WORDS_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 136u;
const BLOCK_ELEMS: u32 = 256u;
const LANES_PER_BLOCK: u32 = 16u;
fn kv4(w: u32) -> vec4<f32> {
    return vec4<f32>(iq_kvalue(w & 0xFu), iq_kvalue((w >> 8u) & 0xFu), iq_kvalue((w >> 16u) & 0xFu), iq_kvalue((w >> 24u) & 0xFu));
}
fn block_dot(byte_offset: u32, x_off: u32, sub: u32) -> f32 {
    let w0 = byte_offset / 4u;
    let hdr = weights[w0];
    let d = f16_to_f32(hdr & 0xFFFFu);
    let scales_h = hdr >> 16u;
    let ib = sub / 2u;
    let nsh = 4u * (sub % 2u);
    let low = (weights[w0 + 1u] >> (8u * (ib / 2u) + 4u * (ib % 2u))) & 0xFu;
    let high = (scales_h >> (2u * ib)) & 3u;
    let dl = d * (f32(low | (high << 4u)) - 32.0);
    let qb = w0 + 2u + 4u * ib;
    let xb = (x_off + sub * 16u) / 4u;
    let x0 = xv(xb);
    let x1 = xv(xb + 1u);
    let x2 = xv(xb + 2u);
    let x3 = xv(xb + 3u);
    let q0 = kv4((weights[qb] >> nsh) & 0x0F0F0F0Fu);
    let q1 = kv4((weights[qb + 1u] >> nsh) & 0x0F0F0F0Fu);
    let q2 = kv4((weights[qb + 2u] >> nsh) & 0x0F0F0F0Fu);
    let q3 = kv4((weights[qb + 3u] >> nsh) & 0x0F0F0F0Fu);
    return dl * (dot(q0, x0) + dot(q1, x1) + dot(q2, x2) + dot(q3, x3));
}
"#;

/// `block_iq3_s` for [`block_hoisted_suffix`], read as words: a lane's 16
/// elements are two 8-element runs of one 32-element group — four lattice
/// words, each index a `qs` byte with its bit of the group's `qh` byte
/// above it, each word's signs a nibble of the run's sign byte.
const IQ3_S_WORDS_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 110u;
const BLOCK_ELEMS: u32 = 256u;
const LANES_PER_BLOCK: u32 = 16u;
fn block_dot(byte_offset: u32, x_off: u32, sub: u32) -> f32 {
    let d = f16_to_f32(read_u16_at(byte_offset));
    let ib = sub / 2u;
    let h = sub % 2u;
    let sc = (read_u8(byte_offset + 106u + ib / 2u) >> (4u * (ib % 2u))) & 0xFu;
    let db = d * f32(1u + 2u * sc);
    let qh = read_u8(byte_offset + 66u + ib);
    // Index bytes 8·ib + 4h .. +4, sign bytes 4·ib + 2h .. +2.
    let idx = read_u32_at(byte_offset + 2u + 8u * ib + 4u * h);
    let sg = read_u16_at(byte_offset + 74u + 4u * ib + 2u * h);
    let bit = 4u * h;
    let g0 = iq_grids[IQ3S_GRID_OFF + ((idx & 0xFFu) | (((qh >> bit) & 1u) << 8u))];
    let g1 = iq_grids[IQ3S_GRID_OFF + (((idx >> 8u) & 0xFFu) | (((qh >> (bit + 1u)) & 1u) << 8u))];
    let g2 = iq_grids[IQ3S_GRID_OFF + (((idx >> 16u) & 0xFFu) | (((qh >> (bit + 2u)) & 1u) << 8u))];
    let g3 = iq_grids[IQ3S_GRID_OFF + ((idx >> 24u) | (((qh >> (bit + 3u)) & 1u) << 8u))];
    let xb = (x_off + sub * 16u) / 4u;
    let x0 = xv(xb);
    let x1 = xv(xb + 1u);
    let x2 = xv(xb + 2u);
    let x3 = xv(xb + 3u);
    let s = dot(unpack4_f(g0) * sign4(sg), x0) + dot(unpack4_f(g1) * sign4(sg >> 4u), x1)
        + dot(unpack4_f(g2) * sign4(sg >> 8u), x2) + dot(unpack4_f(g3) * sign4(sg >> 12u), x3);
    return db * s;
}
"#;

/// `block_iq2_s` for [`block_hoisted_suffix`], read as words: a lane's 16
/// elements are two 8-element runs of one 32-element group, each run one
/// 64-bit lattice entry (two words) with a sign byte, the pair sharing the
/// group's scale nibble for that half.
const IQ2_S_WORDS_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 82u;
const BLOCK_ELEMS: u32 = 256u;
const LANES_PER_BLOCK: u32 = 16u;
fn block_dot(byte_offset: u32, x_off: u32, sub: u32) -> f32 {
    let d = f16_to_f32(read_u16_at(byte_offset));
    let ib = sub / 2u;
    let h = sub % 2u;
    let l0 = 2u * h;
    let sc = (read_u8(byte_offset + 74u + ib) >> (4u * h)) & 0xFu;
    let db = d * (0.5 + f32(sc)) * 0.25;
    let qh = read_u8(byte_offset + 66u + ib);
    let idx2 = read_u16_at(byte_offset + 2u + 4u * ib + l0);
    let sg2 = read_u16_at(byte_offset + 34u + 4u * ib + l0);
    let i0 = (idx2 & 0xFFu) | ((qh << (8u - 2u * l0)) & 0x300u);
    let i1 = (idx2 >> 8u) | ((qh << (6u - 2u * l0)) & 0x300u);
    let e0 = IQ2S_GRID_OFF + 2u * i0;
    let e1 = IQ2S_GRID_OFF + 2u * i1;
    let xb = (x_off + sub * 16u) / 4u;
    let x0 = xv(xb);
    let x1 = xv(xb + 1u);
    let x2 = xv(xb + 2u);
    let x3 = xv(xb + 3u);
    let s = dot(unpack4_f(iq_grids[e0]) * sign4(sg2), x0) + dot(unpack4_f(iq_grids[e0 + 1u]) * sign4(sg2 >> 4u), x1)
        + dot(unpack4_f(iq_grids[e1]) * sign4(sg2 >> 8u), x2) + dot(unpack4_f(iq_grids[e1 + 1u]) * sign4(sg2 >> 12u), x3);
    return db * s;
}
"#;

/// The word-reading `block_dot`s (`ORANGU_BLOCK_DOT_WORDS`, on; `0` keeps
/// the byte-reading ones as the control arm): the same per-type contract
/// as [`block_hoisted_middle`], with a lane's sixteen elements taken as
/// four words against four `vec4` activations in straight-line code.
///
/// The byte-reading forms decode a lane's elements one byte at a time in a
/// rolled loop — sixteen word loads with a shift and mask each, sixteen
/// scalar activation loads, and the loop keeps every one of them in
/// order. Measured on the E2B UD-Q2_K_XL file, the FFN gate ran at 13 GB/s
/// that way against the Q4_K kernel's ~80.
/// `block_pq2_0` for [`block_hoisted_suffix`], read as words: a lane's 16
/// elements are the four 2-bit fields of each of four consecutive bytes —
/// one word of the payload — against four `vec4` activations. The block is
/// 34 bytes, so the word and the scale go through the unaligned readers.
const PQ2_0_WORDS_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 34u;
const BLOCK_ELEMS: u32 = 128u;
const LANES_PER_BLOCK: u32 = 8u;
// One byte's four fields as `-1..=2`, in element order.
fn fields4(b: u32) -> vec4<f32> {
    return vec4<f32>(f32(b & 3u), f32((b >> 2u) & 3u), f32((b >> 4u) & 3u), f32(b >> 6u)) - vec4<f32>(1.0);
}
fn block_dot(byte_offset: u32, x_off: u32, sub: u32) -> f32 {
    let d = f16_to_f32(read_u16_at(byte_offset));
    let w = read_u32_at(byte_offset + 2u + 4u * sub);
    let xi = (x_off + 16u * sub) / 4u;
    var acc: f32 = dot(fields4(w & 0xFFu), xv(xi));
    acc = acc + dot(fields4((w >> 8u) & 0xFFu), xv(xi + 1u));
    acc = acc + dot(fields4((w >> 16u) & 0xFFu), xv(xi + 2u));
    acc = acc + dot(fields4(w >> 24u), xv(xi + 3u));
    return acc * d;
}
"#;

/// `block_ptq1_0` for [`block_hoisted_suffix`], read as words. The block is
/// 28 bytes — seven words, always word-aligned — with the scale in the top
/// half of the last. Sixteen lanes per block and **no branch**: every lane
/// dots two words through the same `trit4v`, and only *which* words, which
/// trit multiplier and which eight activations differ, as lane-uniform
/// selects. Lanes `0..10` take half a trit plane of the 16-byte run (plane
/// `sub / 2`, words `2·(sub % 2)`, `+1`); lanes `10..15` a whole plane of
/// the 8-byte run (words 4 and 5); lane 15 the eight `qh` values, which
/// are the two `qh` bytes duplicated into a word and multiplied by
/// `[1 1 3 3]` then `[9 9 27 27]` — the same `(q · 3^n · 3) >> 8` as
/// everywhere else, with the multiplier per lane instead of per word.
///
/// The first version of this kernel had eight lanes and a three-way branch
/// on `sub`; on the Mali-G720 it streamed at 20 GB/s against `PQ2_0`'s
/// 32 at the same shape (`decode_matvec_format_sweep_gpu_versus_cpu`).
const PTQ1_0_WORDS_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 28u;
const BLOCK_ELEMS: u32 = 128u;
const LANES_PER_BLOCK: u32 = 16u;
// Trit `p[i]` (a power of three) of byte `i` of `w`, as `-1..=1`.
fn trit4v(w: u32, p: vec4<u32>) -> vec4<f32> {
    let b = vec4<u32>(w & 0xFFu, (w >> 8u) & 0xFFu, (w >> 16u) & 0xFFu, w >> 24u);
    let m = (b * p) & vec4<u32>(0xFFu);
    return vec4<f32>((m * vec4<u32>(3u)) >> vec4<u32>(8u)) - vec4<f32>(1.0);
}
fn block_dot(byte_offset: u32, x_off: u32, sub: u32) -> f32 {
    let w0 = byte_offset / 4u;
    let d = f16_to_f32(weights[w0 + 6u] >> 16u);
    var pow3 = array<u32, 5>(1u, 3u, 9u, 27u, 81u);
    let in16 = sub < 10u;
    let in8 = sub >= 10u && sub < 15u;
    let is_qh = sub == 15u;
    let n = select(select(0u, sub - 10u, in8), sub / 2u, in16);
    let h = sub & 1u;
    let p = pow3[n];
    let ia = select(select(6u, 4u, in8), 2u * h, in16);
    let ib = select(select(6u, 5u, in8), 2u * h + 1u, in16);
    var wa = weights[w0 + ia];
    var wb = weights[w0 + ib];
    let dup = (wa & 0xFFFFu) | (wa << 16u);
    wa = select(wa, dup, is_qh);
    wb = select(wb, dup, is_qh);
    let pa = select(vec4<u32>(p), vec4<u32>(1u, 1u, 3u, 3u), is_qh);
    let pb = select(vec4<u32>(p), vec4<u32>(9u, 9u, 27u, 27u), is_qh);
    let e = select(select(120u, 80u + 8u * n, in8), 16u * n + 8u * h, in16);
    let xi = (x_off + e) / 4u;
    return (dot(trit4v(wa, pa), xv(xi)) + dot(trit4v(wb, pb), xv(xi + 1u))) * d;
}
"#;

/// The **integer-dot** decode kernel for Prism's ternary pair, `PQ2_0`
/// and `PTQ1_0` — the same workgroup shape as [`block_hoisted_suffix`]
/// (64 lanes, `n_rows` rows, eight adjacent lanes to a block, eight blocks
/// in flight) with the arithmetic of the `mmvq` kernels: the activations
/// are quantized to `int8` and the weights' trits are packed four to a
/// word, so `dot4I8Packed` does four multiply-adds per instruction and the
/// `-1` offset of a `0/1/2` trit comes out as one subtraction per block
/// (`Σ t·q − Σ q`).
///
/// **Why.** The float words kernels above decode every trit to `f32`
/// before the multiply — for `PTQ1_0` that is a byte extract, a multiply,
/// a mask, a multiply, a shift, a convert and a subtract per weight, then
/// the FMA — and the Mali-G720 issues them at 7.5 GB/s against 37 for
/// `Q8_0` and 30 for `Q4_K` at the same shape
/// (`matmul_kernel_time_by_wall_clock`): the kernel is bound by
/// instructions per weight, not bytes. Here a lane owns one *word* of the
/// block and its whole trit chain: the two even bytes sit in the low
/// halves of two 16-bit lanes and the two odd bytes in another word, and
/// one multiply by three, one shift, one mask per pair of trits
/// (`((q · 3ⁿ) mod 256) · 3 >> 8`, the reference formula, as a remainder
/// chain) yields the four trits of a position already in byte lanes — two
/// shifts fewer than the float form and no conversion at all. Three
/// instructions per weight for `PTQ1_0`, under two for `PQ2_0`, whose
/// fields are already two bits in element order (`(w >> 2k) & 0x03030303`
/// is a whole position).
///
/// **The activations are quantized in the kernel**, per 128-block, by the
/// eight lanes that own the block: each loads exactly the elements its own
/// word multiplies (the block's element map partitions its 128 elements
/// over the lanes' words), takes their absolute maximum, and three
/// `subgroupShuffleXor` steps make that the block's; `pack4x8snorm` of
/// `x / amax` is then the `int8` row with `amax / 127` its scale. Nothing
/// is staged, no barrier is crossed, and the same lanes reuse the packed
/// row for every one of the workgroup's `n_rows` rows — the reason the
/// quantization is worth its ~20% of the block: its cost is paid once per
/// block per workgroup, the trit chain once per block per row. The `Q4_K`
/// `mmvq` path quantizes in a separate dispatch into a second buffer; this
/// binds the same `f32` activations as every other decode kernel, so every
/// caller (the batched path, the fused FFN, the recurrent tail) takes it
/// without knowing.
///
/// **Lanes never branch.** For `PTQ1_0` the seven words of a block have
/// three roles — the 16-byte run (four words, five trit positions of
/// sixteen elements), the 8-byte run (two words, five positions of eight)
/// and the `qh` word (two bytes of four trits each, beside the scale) —
/// and every lane runs the same five-position chain; the role only decides
/// which activations it pairs with, through five per-lane `vec4` indices
/// and validity masks computed once. The `qh` lane's two bytes go in as
/// `[qh0, qh0·3]` and `[qh1, qh1·3]`, so a position `n` of its chain packs
/// `[t(qh0,n), t(qh1,n), t(qh0,n+1), t(qh1,n+1)]` — exactly elements
/// `120 + 2n ..= 123 + 2n`, contiguous — and only positions 0 and 2 are
/// live. The eighth lane decodes the `qh` word again with every position
/// masked; its activation loads duplicate live elements so the block
/// maximum is unchanged.
///
/// Every workgroup runs the same number of block iterations (a block past
/// the row is masked, not skipped), so the shuffles are in uniform control
/// flow. `-1..=1` per trit and `int8` activations sum exactly in `i32`
/// across a block; the rounding is the activation quantization's alone,
/// `round(x · 127 / amax)` per 128 elements — the same step the CPU
/// backend's fused `sdot` kernels take at 256.
///
/// `ORANGU_TERNARY_IDOT=0` keeps the float words kernels — the control arm.
pub fn shader_source_ternary_idot(
    ggml_type: u32,
    n_rows: usize,
    groups: usize,
    subgroup: bool,
) -> Option<String> {
    let ptq = match ggml_type {
        t if t == GGML_TYPE_PTQ1_0 => true,
        t if t == GGML_TYPE_PQ2_0 => false,
        _ => return None,
    };
    let mut s = String::new();
    s.push_str(PRELUDE_XV);
    s.push_str(&format!(
        "\nvar<workgroup> partial_sums: array<f32, {}>;\n\n",
        n_rows * 64
    ));
    s.push_str("@compute @workgroup_size(64)\nfn main(\n    @builtin(workgroup_id) wid: vec3<u32>,\n    @builtin(local_invocation_id) lid: vec3<u32>,\n    @builtin(num_workgroups) nwg: vec3<u32>,");
    s.push_str(subgroup_entry_params(subgroup));
    s.push_str("\n) {\n");
    // A workgroup takes `groups` runs of `n_rows` rows one after another:
    // the lane's role constants, the launch and the drain are paid once
    // per workgroup, and at a 40-block row (the FFN gate) they were about
    // two fifths of it.
    let wg_rows = n_rows * groups;
    s.push_str(&format!(
        "    let n_row_groups = (params.out_dim + {}u) / {wg_rows}u;\n",
        wg_rows - 1
    ));
    s.push_str("    let flat = wid.x + wid.y * nwg.x + wid.z * nwg.x * nwg.y;\n    if (flat >= n_row_groups * params.n_tokens) {\n        return;\n    }\n");
    s.push_str("    let rg = flat / params.n_tokens;\n    let t = flat % params.n_tokens;\n");
    s.push_str("    let local = lid.x;\n    let x_base = t * params.in_dim;\n\n");
    s.push_str("    let sub = local % 8u;\n    let slot = local / 8u;\n    let n_blocks = params.in_dim / 128u;\n    let n_iter = (n_blocks + 7u) / 8u;\n");
    let slots = if ptq { 5 } else { 4 };
    if ptq {
        // Per-lane role constants: the word this lane decodes and, for
        // each of its five positions, the `vec4` (within the block) of the
        // activations that position pairs with. The `qh` lane's positions
        // pair with *two* elements each (`120 + 2n`, `121 + 2n`), which is
        // arranged on the activation side below; its fifth position and
        // every position of the eighth lane pair with zeros.
        s.push_str("    let w = min(sub, 6u);\n    let is16 = sub < 4u;\n    let is8 = sub >= 4u && sub < 6u;\n    let is_qh = sub == 6u;\n    let live = is16 || is8;\n");
        for n in 0..5 {
            s.push_str(&format!(
                "    let idx{n} = select(select(select({n}u, {}u + w, is16), {}u + (w & 1u), is8), {}u, is_qh);\n",
                4 * n,
                20 + 2 * n,
                // `vec4` 30 holds elements 120..124, 31 the last four; the
                // fifth position is masked and duplicates 30 so the block
                // maximum sees nothing outside the block.
                if (2..4).contains(&n) { 31 } else { 30 }
            ));
        }
    } else {
        for n in 0..4 {
            s.push_str(&format!("    let idx{n} = 4u * sub + {n}u;\n"));
        }
    }
    s.push_str(&format!(
        "\n    var g: u32 = 0u;\n    loop {{\n    if (g >= {groups}u) {{\n        break;\n    }}\n    let o_base = (rg * {groups}u + g) * {n_rows}u;\n"
    ));
    for i in 0..n_rows {
        s.push_str(&format!("    let o{i} = o_base + {i}u;\n"));
    }
    for i in 0..n_rows {
        s.push_str(&format!("    var partial{i}: f32 = 0.0;\n"));
    }
    if ptq {
        for r in 0..n_rows {
            s.push_str(&format!(
                "    let row{r} = o{r} * params.row_bytes / 4u + w;\n"
            ));
        }
    } else {
        for r in 0..n_rows {
            s.push_str(&format!("    let row{r} = o{r} * params.row_bytes;\n"));
        }
    }
    s.push_str("\n    var it: u32 = 0u;\n    loop {\n        if (it >= n_iter) {\n            break;\n        }\n");
    s.push_str("        let b = it * 8u + slot;\n        let bvalid = b < n_blocks;\n        let bb = min(b, n_blocks - 1u);\n        let xq_base = (x_base + bb * 128u) / 4u;\n");
    for n in 0..slots {
        s.push_str(&format!("        let v{n} = xv(xq_base + idx{n});\n"));
    }
    // The block's absolute maximum over the eight lanes.
    s.push_str("        var m = abs(v0);\n");
    for n in 1..slots {
        s.push_str(&format!("        m = max(m, abs(v{n}));\n"));
    }
    s.push_str("        let a0 = max(max(m.x, m.y), max(m.z, m.w));\n        let a1 = max(a0, subgroupShuffleXor(a0, 1u));\n        let a2 = max(a1, subgroupShuffleXor(a1, 2u));\n        let amax = max(a2, subgroupShuffleXor(a2, 4u));\n");
    s.push_str("        let inv = select(0.0, 1.0 / amax, amax > 0.0);\n        let dx = amax * (1.0 / 127.0);\n");
    if ptq {
        // A position's four trits come out of the chain in bytes 1 and 3 of
        // `l·3` (even bytes of the word) and of `h·3` (odd bytes), so the
        // activations are spread the same way once per block: `xe` holds
        // the even elements in bytes 1 and 3, `xo` the odd ones. The `qh`
        // lane's position `n` is elements `120 + 2n` and `121 + 2n` in
        // byte 1 of each; everything masked is a zero activation word.
        s.push_str("        let zero = select(0u, 0xFFFFFFFFu, bvalid);\n");
        for n in 0..5 {
            s.push_str(&format!(
                "        let p{n} = pack4x8snorm(v{n} * inv) & zero;\n"
            ));
        }
        // Live lanes take their packs whole; the `qh` lane's position `n`
        // is the low or high half of `vec4` 30 or 31 (two elements, the
        // rest zero); the eighth lane and the `qh` lane's fifth position
        // are zeros.
        for n in 0..5 {
            let qh = match n {
                0 | 2 => format!("p{n} & 0xFFFFu"),
                1 | 3 => format!("p{n} >> 16u"),
                _ => "0u".to_string(),
            };
            s.push_str(&format!(
                "        let xq{n} = select(select(0u, p{n}, live), {qh}, is_qh);\n"
            ));
        }
        for n in 0..5 {
            s.push_str(&format!(
                "        let xe{n} = (xq{n} & 0x00FF00FFu) << 8u;\n        let xo{n} = xq{n} & 0xFF00FF00u;\n"
            ));
        }
        s.push_str("        var qsum: i32 = 0;\n");
        for n in 0..5 {
            s.push_str(&format!(
                "        qsum = qsum + dot4I8Packed(0x01010101u, xq{n});\n"
            ));
        }
        s.push_str("        let blk = bb * 7u;\n");
        for r in 0..n_rows {
            // Every row guarded, row 0 included: with several runs per
            // workgroup a run past the last row starts past `out_dim`, and
            // an unguarded row 0 there wrote the next token's rows — the
            // start-up self-check's 2055-row case caught it.
            s.push_str(&format!("        if (o{r} < params.out_dim) {{\n"));
            s.push_str(&format!(
                "            let w0 = row{r} + blk;\n            let word = weights[w0];\n            let dw = f16_to_f32(weights[w0 + 6u - w] >> 16u);\n"
            ));
            s.push_str("            var l = word & 0x00FF00FFu;\n            var h = (word >> 8u) & 0x00FF00FFu;\n            var isum: i32 = 0;\n");
            for n in 0..5 {
                s.push_str(&format!(
                    "            let ml{n} = l * 3u;\n            let mh{n} = h * 3u;\n            isum = isum + dot4I8Packed(ml{n} & 0x03000300u, xe{n}) + dot4I8Packed(mh{n} & 0x03000300u, xo{n});\n"
                ));
                if n < 4 {
                    s.push_str(&format!(
                        "            l = ml{n} & 0x00FF00FFu;\n            h = mh{n} & 0x00FF00FFu;\n"
                    ));
                }
            }
            s.push_str(&format!(
                "            partial{r} = partial{r} + f32(isum - qsum) * (dw * dx);\n        }}\n"
            ));
        }
    } else {
        // `PQ2_0`: position `k` of the lane's word is field `k` of its four
        // bytes — elements `16·sub + 4j + k` — against the `k`-th component
        // of the four activation vectors.
        s.push_str("        let zero = select(0u, 0xFFFFFFFFu, bvalid);\n");
        for k in 0..4 {
            let c = ["x", "y", "z", "w"][k];
            s.push_str(&format!(
                "        let xq{k} = pack4x8snorm(vec4<f32>(v0.{c}, v1.{c}, v2.{c}, v3.{c}) * inv) & zero;\n"
            ));
        }
        s.push_str("        var qsum: i32 = 0;\n");
        for k in 0..4 {
            s.push_str(&format!(
                "        qsum = qsum + dot4I8Packed(0x01010101u, xq{k});\n"
            ));
        }
        s.push_str("        let blk = bb * 34u + 2u + 4u * sub;\n");
        for r in 0..n_rows {
            // Every row guarded, row 0 included: with several runs per
            // workgroup a run past the last row starts past `out_dim`, and
            // an unguarded row 0 there wrote the next token's rows — the
            // start-up self-check's 2055-row case caught it.
            s.push_str(&format!("        if (o{r} < params.out_dim) {{\n"));
            s.push_str(&format!(
                "            let byte_off = row{r} + blk;\n            let word = read_u32_at(byte_off);\n            let dw = f16_to_f32(read_u16_at(byte_off - 2u - 4u * sub));\n"
            ));
            s.push_str("            var isum: i32 = 0;\n");
            for k in 0..4 {
                s.push_str(&format!(
                    "            isum = isum + dot4I8Packed((word >> {}u) & 0x03030303u, xq{k});\n",
                    2 * k
                ));
            }
            s.push_str(&format!(
                "            partial{r} = partial{r} + f32(isum - qsum) * (dw * dx);\n        }}\n"
            ));
        }
    }
    s.push_str("        it = it + 1u;\n    }\n\n");
    // The shared combine writes row 0 unguarded (its other kernels never
    // start a workgroup past the last row); here a run can.
    s.push_str(
        &reduce_combine_block(n_rows, subgroup)
            .replace(
                "        y[t * params.out_dim + o0] = t0;\n",
                "        if (o0 < params.out_dim) {\n            y[t * params.out_dim + o0] = t0;\n        }\n",
            )
            .replace(
                "        y[t * params.out_dim + o0] = partial_sums[0];\n",
                "        if (o0 < params.out_dim) {\n            y[t * params.out_dim + o0] = partial_sums[0];\n        }\n",
            ),
    );
    // The next run's partial sums must not land before this run's combine
    // has read them.
    s.push_str("    workgroupBarrier();\n    g = g + 1u;\n    }\n");
    s.push_str("}\n");
    Some(s)
}

/// [`shader_source_ternary_idot`] against an activation already quantized
/// to `int8` per 32 (`shader_source_quantize_q8`'s layout, binding 1 —
/// the layout main's integer-dot kernels read, so the same quantize
/// dispatch or norm epilogue feeds this). The activation side of the
/// kernel above — five `vec4` loads, `abs`/`max`, three shuffles, five
/// packs per block iteration, redone by every workgroup — becomes five
/// word loads; and the scale is per 32 rather than per 128. The trit
/// chain is the same.
///
/// A lane's five positions fall into three scale groups whatever its
/// role (positions 0–1, 2–3 and 4 each lie in one 32-block: elements
/// `16n + 4w` for the 16-byte run, `80 + 8n + 4(w − 4)` for the 8-byte
/// run, `120 + 2n` for `qh`), so the integer sums are kept per group and
/// scaled once each per row.
pub fn shader_source_ternary_i8(
    ggml_type: u32,
    n_rows: usize,
    groups: usize,
    subgroup: bool,
) -> Option<String> {
    let ptq = match ggml_type {
        t if t == GGML_TYPE_PTQ1_0 => true,
        t if t == GGML_TYPE_PQ2_0 => false,
        _ => return None,
    };
    let mut s = String::new();
    s.push_str(PRELUDE_Q8);
    s.push_str(&format!(
        "\nvar<workgroup> partial_sums: array<f32, {}>;\n\n",
        n_rows * 64
    ));
    s.push_str("@compute @workgroup_size(64)\nfn main(\n    @builtin(workgroup_id) wid: vec3<u32>,\n    @builtin(local_invocation_id) lid: vec3<u32>,\n    @builtin(num_workgroups) nwg: vec3<u32>,");
    s.push_str(subgroup_entry_params(subgroup));
    s.push_str("\n) {\n");
    let wg_rows = n_rows * groups;
    s.push_str(&format!(
        "    let n_row_groups = (params.out_dim + {}u) / {wg_rows}u;\n",
        wg_rows - 1
    ));
    s.push_str("    let flat = wid.x + wid.y * nwg.x + wid.z * nwg.x * nwg.y;\n    if (flat >= n_row_groups * params.n_tokens) {\n        return;\n    }\n");
    s.push_str("    let rg = flat / params.n_tokens;\n    let t = flat % params.n_tokens;\n");
    s.push_str("    let local = lid.x;\n    let x_base = t * params.in_dim;\n\n");
    s.push_str("    let sub = local % 8u;\n    let slot = local / 8u;\n    let n_blocks = params.in_dim / 128u;\n    let n_iter = (n_blocks + 7u) / 8u;\n");
    if ptq {
        s.push_str("    let w = min(sub, 6u);\n    let is16 = sub < 4u;\n    let is8 = sub >= 4u && sub < 6u;\n    let is_qh = sub == 6u;\n    let live = is16 || is8;\n");
        for n in 0..5 {
            // Element offsets within the block, per role; the `qh` lane's
            // live positions 0 and 2 start at 120 and 124.
            s.push_str(&format!(
                "    let e{n} = select(select(select({}u, {}u + 4u * w, is16), {}u + 4u * (w & 1u), is8), {}u, is_qh);\n",
                4 * n,
                16 * n,
                80 + 8 * n,
                if (2..4).contains(&n) { 124 } else { 120 }
            ));
        }
    } else {
        for n in 0..4 {
            s.push_str(&format!("    let e{n} = 16u * sub + 4u * {n}u;\n"));
        }
    }
    s.push_str(&format!(
        "\n    var g: u32 = 0u;\n    loop {{\n    if (g >= {groups}u) {{\n        break;\n    }}\n    let o_base = (rg * {groups}u + g) * {n_rows}u;\n"
    ));
    for i in 0..n_rows {
        s.push_str(&format!("    let o{i} = o_base + {i}u;\n"));
    }
    for i in 0..n_rows {
        s.push_str(&format!("    var partial{i}: f32 = 0.0;\n"));
    }
    if ptq {
        for r in 0..n_rows {
            s.push_str(&format!(
                "    let row{r} = o{r} * params.row_bytes / 4u + w;\n"
            ));
        }
    } else {
        for r in 0..n_rows {
            s.push_str(&format!("    let row{r} = o{r} * params.row_bytes;\n"));
        }
    }
    s.push_str("\n    var it: u32 = 0u;\n    loop {\n        if (it >= n_iter) {\n            break;\n        }\n");
    s.push_str("        let b = it * 8u + slot;\n        let bvalid = b < n_blocks;\n        let bb = min(b, n_blocks - 1u);\n        let eb = x_base + bb * 128u;\n        let zero = select(0u, 0xFFFFFFFFu, bvalid);\n");
    if ptq {
        // The five activation words and their three scales; the `qh` lane's
        // words hold two live elements each, the eighth lane's none.
        for n in 0..5 {
            let qh = match n {
                0 | 2 => format!("q8_word(eb + e{n}) & 0xFFFFu"),
                1 | 3 => format!("q8_word(eb + e{n}) >> 16u"),
                _ => "0u".to_string(),
            };
            s.push_str(&format!(
                "        let xq{n} = select(select(0u, q8_word(eb + e{n}), live), {qh}, is_qh) & zero;\n"
            ));
        }
        s.push_str("        let d01 = q8_scale(eb + e0);\n        let d23 = q8_scale(eb + e2);\n        let d4 = q8_scale(eb + e4);\n");
        for n in 0..5 {
            s.push_str(&format!(
                "        let xe{n} = (xq{n} & 0x00FF00FFu) << 8u;\n        let xo{n} = xq{n} & 0xFF00FF00u;\n"
            ));
        }
        s.push_str("        let q01 = dot4I8Packed(0x01010101u, xq0) + dot4I8Packed(0x01010101u, xq1);\n        let q23 = dot4I8Packed(0x01010101u, xq2) + dot4I8Packed(0x01010101u, xq3);\n        let q4 = dot4I8Packed(0x01010101u, xq4);\n");
        s.push_str("        let blk = bb * 7u;\n");
        for r in 0..n_rows {
            s.push_str(&format!("        if (o{r} < params.out_dim) {{\n"));
            s.push_str(&format!(
                "            let w0 = row{r} + blk;\n            let word = weights[w0];\n            let dw = f16_to_f32(weights[w0 + 6u - w] >> 16u);\n"
            ));
            s.push_str("            var l = word & 0x00FF00FFu;\n            var h = (word >> 8u) & 0x00FF00FFu;\n            var s01: i32 = 0;\n            var s23: i32 = 0;\n            var s4: i32 = 0;\n");
            for n in 0..5 {
                let acc = match n {
                    0 | 1 => "s01",
                    2 | 3 => "s23",
                    _ => "s4",
                };
                s.push_str(&format!(
                    "            let ml{n} = l * 3u;\n            let mh{n} = h * 3u;\n            {acc} = {acc} + dot4I8Packed(ml{n} & 0x03000300u, xe{n}) + dot4I8Packed(mh{n} & 0x03000300u, xo{n});\n"
                ));
                if n < 4 {
                    s.push_str(&format!(
                        "            l = ml{n} & 0x00FF00FFu;\n            h = mh{n} & 0x00FF00FFu;\n"
                    ));
                }
            }
            s.push_str(&format!(
                "            partial{r} = partial{r} + dw * (f32(s01 - q01) * d01 + f32(s23 - q23) * d23 + f32(s4 - q4) * d4);\n        }}\n"
            ));
        }
    } else {
        // `PQ2_0`: position `k` of the lane's word is field `k` of its four
        // bytes — elements `16·sub + 4j + k` — so the activation words are
        // gathered transposed: word `k` holds elements `16·sub + k + 4j`.
        for j in 0..4 {
            s.push_str(&format!("        let a{j} = q8_word(eb + e{j}) & zero;\n"));
        }
        for k in 0..4 {
            let sh = 8 * k;
            s.push_str(&format!(
                "        let xq{k} = ((a0 >> {sh}u) & 0xFFu) | (((a1 >> {sh}u) & 0xFFu) << 8u) | (((a2 >> {sh}u) & 0xFFu) << 16u) | (((a3 >> {sh}u) & 0xFFu) << 24u);\n"
            ));
        }
        // The lane's sixteen elements lie in one 32-block: one scale.
        s.push_str("        let dx = q8_scale(eb + e0);\n        var qsum: i32 = 0;\n");
        for k in 0..4 {
            s.push_str(&format!(
                "        qsum = qsum + dot4I8Packed(0x01010101u, xq{k});\n"
            ));
        }
        s.push_str(
            "        let blk = bb * 34u + 2u + 4u * sub;\n        let fm = zero & 0x03030303u;\n",
        );
        for r in 0..n_rows {
            s.push_str(&format!("        if (o{r} < params.out_dim) {{\n"));
            s.push_str(&format!(
                "            let byte_off = row{r} + blk;\n            let word = read_u32_at(byte_off);\n            let dw = f16_to_f32(read_u16_even(byte_off - 2u - 4u * sub));\n"
            ));
            s.push_str("            var isum: i32 = 0;\n");
            for k in 0..4 {
                s.push_str(&format!(
                    "            isum = isum + dot4I8Packed((word >> {}u) & fm, xq{k});\n",
                    2 * k
                ));
            }
            s.push_str(&format!(
                "            partial{r} = partial{r} + f32(isum - qsum) * (dw * dx);\n        }}\n"
            ));
        }
    }
    s.push_str("        it = it + 1u;\n    }\n\n");
    s.push_str(
        &reduce_combine_block(n_rows, subgroup)
            .replace(
                "        y[t * params.out_dim + o0] = t0;\n",
                "        if (o0 < params.out_dim) {\n            y[t * params.out_dim + o0] = t0;\n        }\n",
            )
            .replace(
                "        y[t * params.out_dim + o0] = partial_sums[0];\n",
                "        if (o0 < params.out_dim) {\n            y[t * params.out_dim + o0] = partial_sums[0];\n        }\n",
            ),
    );
    s.push_str("    workgroupBarrier();\n    g = g + 1u;\n    }\n");
    s.push_str("}\n");
    Some(s)
}

/// Words per 128-element block of the transposed `int8` activation layout
/// the two-phase `PQ2_0` kernel reads (`shader_source_ternary_t`): `[d,
/// Σq, 0, 0]`, then for each of two phases nine groups of four words —
/// word `4 + 36·phase + 4·i + k` holds, in bytes `t = 0..3`, the quant of
/// element `4·j + k` with `j = 4·i − 4 + t` (phase 1) or `4·i − 2 + t`
/// (phase 0), zero where `j` is outside `0..32`. That is the activation
/// transposed to meet `PQ2_0`'s 2-bit fields *as the weight words land in
/// memory*: a block's 32 payload bytes start at byte `34·b + 2`, so an
/// even block's payload sits two bytes into its aligned words (phase 0)
/// and an odd block's is aligned (phase 1, with a dead word ahead of it).
/// The kernel reads nine aligned words either way and dots them against
/// the phase's groups, and no weight word is ever shifted into alignment.
/// One scale per 128 (`amax / 127`, the same rounding as the per-32
/// layout), 304 bytes a block.
///
/// Probe-only, with the kernel: measured slower than
/// [`shader_source_ternary_i8`] on the Mali-G720 (see there).
#[cfg(test)]
pub const TERNARY_T_WORDS: u32 = 76;

/// `shader_source_quantize_q8`'s counterpart for [`TERNARY_T_WORDS`]'
/// layout: one workgroup of 64 threads per 128-block, each thread one of
/// the 72 transposed words (eight idle), the scale and sum from a shared
/// reduction. `qmeta.len` elements, a multiple of 128.
#[cfg(test)]
pub fn shader_source_quantize_ternary_t() -> String {
    r#"
struct ElemMeta {
    len: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

@group(0) @binding(0) var<storage, read> xin: array<f32>;
@group(0) @binding(1) var<storage, read_write> qout: array<u32>;
@group(0) @binding(2) var<uniform> qmeta: ElemMeta;

var<workgroup> q: array<i32, 128>;
var<workgroup> red: array<f32, 64>;
var<workgroup> redi: array<i32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let blk = wid.x;
    let t = lid.x;
    let base = blk * 128u;
    let v0 = xin[base + t];
    let v1 = xin[base + 64u + t];
    red[t] = max(abs(v0), abs(v1));
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { red[t] = max(red[t], red[t + stride]); }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let amax = red[0];
    let d = amax / 127.0;
    let id = select(0.0, 1.0 / d, d > 0.0);
    let q0 = clamp(i32(round(v0 * id)), -127, 127);
    let q1 = clamp(i32(round(v1 * id)), -127, 127);
    q[t] = q0;
    q[64u + t] = q1;
    redi[t] = q0 + q1;
    workgroupBarrier();
    stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { redi[t] = redi[t] + redi[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let out = blk * 76u;
    if (t == 0u) {
        qout[out] = bitcast<u32>(d);
        qout[out + 1u] = bitcast<u32>(redi[0]);
        qout[out + 2u] = 0u;
        qout[out + 3u] = 0u;
    }
    // Thread `t` builds transposed word `t` (0..36) of both phases.
    if (t < 36u) {
        let i = t / 4u;
        let k = t % 4u;
        for (var phase: u32 = 0u; phase < 2u; phase = phase + 1u) {
            var word: u32 = 0u;
            for (var tt: u32 = 0u; tt < 4u; tt = tt + 1u) {
                let j = i32(4u * i + tt) - select(2, 4, phase == 1u);
                var qv: i32 = 0;
                if (j >= 0 && j < 32) {
                    qv = q[4u * u32(j) + k];
                }
                word = word | ((u32(qv) & 0xFFu) << (8u * tt));
            }
            qout[out + 4u + 36u * phase + t] = word;
        }
    }
}
"#
    .to_string()
}

/// The `PQ2_0` decode kernel over [`TERNARY_T_WORDS`]' activation layout:
/// a workgroup of four 16-lane subgroups, each subgroup its own eight
/// rows, each lane one block per iteration — nine aligned weight words
/// (the block's payload as it lies, with the scale in the first word's
/// low or high half by the block's phase) dotted field by field against
/// the phase's transposed activation words, one integer sum per row per
/// block, one scale multiply. Against [`shader_source_ternary_i8`] this
/// removes the unaligned word reads (two loads and three shifts each),
/// the per-lane activation transposes and the per-32 scale bookkeeping:
/// ~1.1 instructions a weight where that kernel spent ~1.9. Subgroups
/// reduce their own rows, so there is no shared memory and no barrier.
/// 32 rows a workgroup. **An even number of blocks per row only**: with
/// an odd count a row's start is two bytes off every other row and the
/// phase would depend on the row as well as the block (every width this
/// model family has is a multiple of 256). **A 16-lane subgroup only**:
/// the lane's block index and the subgroup's row slice count to 16, and
/// nothing rescales them for a wider device — its test skips elsewhere.
///
/// **Probe-only.** Measured on the Mali-G720 (`ternary_t_kernel_time`,
/// warm, best of 12): the FFN gate shape 951 µs against the 8-lane
/// kernel's 698, the down shape 876 against 535 — slower, with fewer
/// instructions a weight, because a lane's nine loads walk the row at a
/// 34-byte stride across the subgroup (nine cache lines a load
/// instruction where the 8-lane kernel touches two). Kept with its test
/// as the record of the attempt, for a device with a different memory
/// pipeline.
#[cfg(test)]
pub fn shader_source_ternary_t() -> String {
    let mut s = String::new();
    s.push_str(PRELUDE_Q8);
    s.push_str(
        "\n@compute @workgroup_size(64)\nfn main(\n    @builtin(workgroup_id) wid: vec3<u32>,\n    @builtin(local_invocation_id) lid: vec3<u32>,\n    @builtin(num_workgroups) nwg: vec3<u32>,\n    @builtin(subgroup_invocation_id) sg_lane: u32,\n    @builtin(subgroup_id) sg_id: u32,\n) {\n",
    );
    s.push_str("    let n_row_groups = (params.out_dim + 31u) / 32u;\n");
    s.push_str("    let flat = wid.x + wid.y * nwg.x + wid.z * nwg.x * nwg.y;\n    if (flat >= n_row_groups * params.n_tokens) {\n        return;\n    }\n");
    s.push_str("    let rg = flat / params.n_tokens;\n    let t = flat % params.n_tokens;\n");
    s.push_str(
        "    let n_blocks = params.in_dim / 128u;\n    let n_iter = (n_blocks + 15u) / 16u;\n",
    );
    s.push_str("    let x_base = t * n_blocks * 76u;\n    let o_base = rg * 32u + sg_id * 8u;\n");
    // A row past the last reads the last one (in bounds) and writes
    // nothing.
    for r in 0..8 {
        s.push_str(&format!(
            "    let o{r} = o_base + {r}u;\n    let row{r} = min(o{r}, params.out_dim - 1u) * params.row_bytes;\n    var partial{r}: f32 = 0.0;\n"
        ));
    }
    s.push_str("    var it: u32 = 0u;\n    loop {\n        if (it >= n_iter) {\n            break;\n        }\n");
    s.push_str("        let b = it * 16u + sg_lane;\n        let bvalid = b < n_blocks;\n        let bb = min(b, n_blocks - 1u);\n        let phase = bb & 1u;\n        let zero = select(0u, 0xFFFFFFFFu, bvalid);\n");
    // A lane past the last block reads the last one, masked: its
    // activation words and its scale are zero, so its `Σq` does not leak
    // through the `-1` offset.
    s.push_str("        let xb = x_base + bb * 76u;\n        let d = select(0.0, bitcast<f32>(q8x[xb]), bvalid);\n        let sumq = bitcast<i32>(q8x[xb + 1u]);\n        let tb = xb + 4u + 36u * phase;\n");
    // The block's nine aligned words start at byte 34·b, or two bytes
    // earlier for an odd block. Word by word: the word's four activation
    // words, then every row's weight word against them — nothing but the
    // eight integer sums stays live across words.
    s.push_str(
        "        let wstart = (34u * bb - 2u * phase) / 4u;\n        let fm = 0x03030303u;\n",
    );
    for r in 0..8 {
        s.push_str(&format!(
            "        let wi{r} = row{r} / 4u + wstart;\n        var isum{r}: i32 = 0;\n        var dw{r}: f32 = 0.0;\n"
        ));
    }
    for i in 0..9 {
        for k in 0..4 {
            s.push_str(&format!(
                "        let x{i}_{k} = q8x[tb + {}u] & zero;\n",
                4 * i + k
            ));
        }
        for r in 0..8 {
            s.push_str(&format!(
                "        {{\n            let w = weights[wi{r} + {i}u];\n"
            ));
            if i == 0 {
                s.push_str(&format!(
                    "            dw{r} = f16_to_f32(select(w & 0xFFFFu, w >> 16u, phase == 1u));\n"
                ));
            }
            for k in 0..4 {
                let src = if k == 0 {
                    "w & fm".to_string()
                } else {
                    format!("(w >> {}u) & fm", 2 * k)
                };
                s.push_str(&format!(
                    "            isum{r} = isum{r} + dot4I8Packed({src}, x{i}_{k});\n"
                ));
            }
            s.push_str("        }\n");
        }
    }
    for r in 0..8 {
        s.push_str(&format!(
            "        if (o{r} < params.out_dim) {{\n            partial{r} = partial{r} + f32(isum{r} - sumq) * (dw{r} * d);\n        }}\n"
        ));
    }
    s.push_str("        it = it + 1u;\n    }\n");
    for r in 0..8 {
        s.push_str(&format!("    let sg{r} = subgroupAdd(partial{r});\n"));
    }
    s.push_str("    if (sg_lane == 0u) {\n");
    for r in 0..8 {
        s.push_str(&format!(
            "        if (o{r} < params.out_dim) {{\n            y[t * params.out_dim + o{r}] = sg{r};\n        }}\n"
        ));
    }
    s.push_str("    }\n}\n");
    s
}

/// Words per 128-element block of the activation layout the octet `PQ2_0`
/// kernel reads (`shader_source_ternary_o`): `[d, 0, 0, 0]`, then for
/// each of the two phases nine groups of five words — the four transposed
/// activation words of [`TERNARY_T_WORDS`]' group `(phase, i)` and the
/// group's `Σq` as an `i32` — 94 words padded to 96 (384 bytes). With the
/// sum beside each group, a lane subtracts the `-1` offset of the fields
/// it holds by itself; no lane has to know where a block's words end.
///
/// Probe-only, with the kernel (see [`shader_source_ternary_o`]).
#[cfg(test)]
pub const TERNARY_O_WORDS: u32 = 96;

/// [`shader_source_quantize_ternary_t`] for [`TERNARY_O_WORDS`]' layout.
#[cfg(test)]
pub fn shader_source_quantize_ternary_o() -> String {
    shader_source_quantize_ternary_t()
        .replace("let out = blk * 76u;", "let out = blk * 96u;")
        .replace(
            "            qout[out + 4u + 36u * phase + t] = word;\n",
            "            qout[out + 4u + 45u * phase + 5u * i + k] = word;\n            if (k == 0u) {\n                var all: i32 = 0;\n                for (var kk: u32 = 0u; kk < 4u; kk = kk + 1u) {\n                    for (var tt: u32 = 0u; tt < 4u; tt = tt + 1u) {\n                        let j = i32(4u * i + tt) - select(2, 4, phase == 1u);\n                        if (j >= 0 && j < 32) {\n                            all = all + q[4u * u32(j) + kk];\n                        }\n                    }\n                }\n                qout[out + 4u + 45u * phase + 5u * i + 4u] = bitcast<u32>(all);\n            }\n",
        )
}

/// The `PQ2_0` decode kernel over [`TERNARY_O_WORDS`]' layout, reading the
/// weights the way the memory system likes them: a subgroup takes eight
/// consecutive blocks of a row — 272 bytes, sixteen-byte aligned when the
/// row is (a block count that is a multiple of eight) — as one `vec4<u32>`
/// a lane, lanes 0–3 the four words left over. Every word of the 68 is
/// the payload (and, for the first of a block, the scale) of exactly one
/// block: word `w` belongs to block `j = 2w / 17` at position `i = w −
/// (17j − (j & 1)) / 2` of its phase's transposed groups (an even block's
/// nine words start at its scale, an odd block's eight after it), and a
/// lane's four words fall in at most two blocks. The block scale reaches
/// every lane of the block by `subgroupShuffle` from the lane that loaded
/// it. Four subgroups a workgroup, eight rows each, reduced within the
/// subgroup. **Block count a multiple of eight only**, and, like
/// [`shader_source_ternary_t`], **a 16-lane subgroup only**.
///
/// **Probe-only.** Measured on the Mali-G720 (`ternary_o_kernel_time`):
/// 7978 µs at the gate shape with the shuffles — `subgroupShuffle` is
/// ruinous on this driver, three a row costing eight times the kernel —
/// and 1017 µs with them stubbed out (`ORANGU_O_EXPERIMENT=noshuffle`),
/// still behind the 8-lane kernel's 698 with the same coalescing, fewer
/// load instructions and fewer dots. The 8-lane kernel's own
/// decomposition (`ternary_i8_kernel_time` under `ORANGU_O_EXPERIMENT`)
/// says why neither redesign could win: with every load removed it still
/// takes 375 of its 693 µs — the per-block-per-row structure, not any of
/// the loads or the arithmetic, is the floor, and each of those removed
/// alone moves the time by less than the run-to-run noise. Kept with its
/// test as the record of the attempt.
#[cfg(test)]
pub fn shader_source_ternary_o() -> String {
    let mut s = String::new();
    // The weights as `vec4<u32>`: one load a lane an octet-row.
    s.push_str(
        r#"
struct Meta {
    in_dim: u32,
    out_dim: u32,
    n_tokens: u32,
    row_bytes: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> q8x: array<u32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> params: Meta;

fn f16_to_f32(bits: u32) -> f32 {
    return unpack2x16float(bits & 0xFFFFu).x;
}
"#,
    );
    s.push_str(
        "\n@compute @workgroup_size(64)\nfn main(\n    @builtin(workgroup_id) wid: vec3<u32>,\n    @builtin(local_invocation_id) lid: vec3<u32>,\n    @builtin(num_workgroups) nwg: vec3<u32>,\n    @builtin(subgroup_invocation_id) sg_lane: u32,\n    @builtin(subgroup_id) sg_id: u32,\n) {\n",
    );
    s.push_str("    let n_row_groups = (params.out_dim + 31u) / 32u;\n");
    s.push_str("    let flat = wid.x + wid.y * nwg.x + wid.z * nwg.x * nwg.y;\n    if (flat >= n_row_groups * params.n_tokens) {\n        return;\n    }\n");
    s.push_str("    let rg = flat / params.n_tokens;\n    let t = flat % params.n_tokens;\n");
    s.push_str("    let n_blocks = params.in_dim / 128u;\n    let n_oct = n_blocks / 8u;\n");
    s.push_str("    let x_base = t * n_blocks * 96u;\n    let o_base = rg * 32u + sg_id * 8u;\n    let l = sg_lane;\n    let extra = l < 4u;\n");
    // The lane's words and their blocks/positions, fixed per lane.
    for m in 0..4 {
        s.push_str(&format!(
            "    let w{m} = 4u * l + {m}u;\n    let j{m} = (2u * w{m}) / 17u;\n    let i{m} = w{m} - (17u * j{m} - (j{m} & 1u)) / 2u;\n    let g{m} = 4u + 45u * (j{m} & 1u) + 5u * i{m};\n"
        ));
    }
    // The extra word (lanes 0–3): word 64 + l, block 7, position 5 + l.
    s.push_str("    let we = 64u + (l & 3u);\n    let ge = 4u + 45u + 5u * (we - 59u);\n");
    // The scale word of the lane's first and last block: word S_j, held
    // by lane S_j / 4 at component S_j % 4, low or high half by phase.
    s.push_str(
        "    let sa = (17u * j0 - (j0 & 1u)) / 2u;\n    let sb = (17u * j3 - (j3 & 1u)) / 2u;\n",
    );
    for r in 0..8 {
        s.push_str(&format!(
            "    let o{r} = o_base + {r}u;\n    let row{r} = min(o{r}, params.out_dim - 1u) * params.row_bytes / 16u;\n    var partial{r}: f32 = 0.0;\n"
        ));
    }
    s.push_str("    var oct: u32 = 0u;\n    loop {\n        if (oct >= n_oct) {\n            break;\n        }\n");
    s.push_str("        let xb0 = x_base + (oct * 8u + j0) * 96u;\n        let xb3 = x_base + (oct * 8u + j3) * 96u;\n        let d0 = bitcast<f32>(q8x[xb0]);\n        let d3 = bitcast<f32>(q8x[xb3]);\n");
    // The activation groups of the lane's words (the same for every row).
    for m in 0..4 {
        s.push_str(&format!(
            "        let xb{m}_ = x_base + (oct * 8u + j{m}) * 96u + g{m};\n"
        ));
        for k in 0..4 {
            s.push_str(&format!("        let x{m}_{k} = q8x[xb{m}_ + {k}u];\n"));
        }
        s.push_str(&format!(
            "        let xs{m} = bitcast<i32>(q8x[xb{m}_ + 4u]);\n"
        ));
    }
    s.push_str("        let xbe = x_base + (oct * 8u + 7u) * 96u + ge;\n        let de = bitcast<f32>(q8x[x_base + (oct * 8u + 7u) * 96u]);\n");
    for k in 0..4 {
        s.push_str(&format!(
            "        let xe_{k} = select(0u, q8x[xbe + {k}u], extra);\n"
        ));
    }
    s.push_str("        let xse = select(0, bitcast<i32>(q8x[xbe + 4u]), extra);\n");
    s.push_str("        let fm = 0x03030303u;\n        let wbase = oct * 17u;\n");
    for r in 0..8 {
        s.push_str(&format!(
            "        {{\n            let v = weights[row{r} + wbase + l];\n            let ve = weights[row{r} + wbase + 16u][l & 3u];\n"
        ));
        // Block scales by shuffle: a scale word `S_j` lies in lane `S_j / 4`
        // at component `S_j % 4`, which for every one of the eight is that
        // lane's `l / 4` — so each lane offers component `l / 4` of its own
        // load, and a shuffle names the lane alone (the shuffled expression
        // is evaluated by the source lane, so it cannot depend on the
        // caller's index).
        s.push_str("            let mine = v[l / 4u];\n            let sw0 = subgroupShuffle(mine, sa / 4u);\n            let sw3 = subgroupShuffle(mine, sb / 4u);\n            let dw0 = f16_to_f32(select(sw0 & 0xFFFFu, sw0 >> 16u, (j0 & 1u) == 1u));\n            let dw3 = f16_to_f32(select(sw3 & 0xFFFFu, sw3 >> 16u, (j3 & 1u) == 1u));\n");
        s.push_str("            let swe = subgroupShuffle(mine, 14u);\n            let dwe = f16_to_f32(swe >> 16u);\n");
        s.push_str("            var acc0: i32 = 0;\n            var acc3: i32 = 0;\n");
        for m in 0..4 {
            s.push_str(&format!("            var s{m}: i32 = -xs{m};\n"));
            for k in 0..4 {
                let src = if k == 0 {
                    format!("v[{m}] & fm")
                } else {
                    format!("(v[{m}] >> {}u) & fm", 2 * k)
                };
                s.push_str(&format!(
                    "            s{m} = s{m} + dot4I8Packed({src}, x{m}_{k});\n"
                ));
            }
            s.push_str(&format!(
                "            acc0 = acc0 + select(0, s{m}, j{m} == j0);\n            acc3 = acc3 + select(0, s{m}, j{m} != j0);\n"
            ));
        }
        s.push_str("            var se: i32 = -xse;\n");
        for k in 0..4 {
            let src = if k == 0 {
                "ve & fm".to_string()
            } else {
                format!("(ve >> {}u) & fm", 2 * k)
            };
            s.push_str(&format!(
                "            se = se + dot4I8Packed({src}, xe_{k});\n"
            ));
        }
        s.push_str(&format!(
            "            partial{r} = partial{r} + f32(acc0) * (dw0 * d0) + f32(acc3) * (dw3 * d3) + f32(se) * (dwe * de);\n        }}\n"
        ));
    }
    s.push_str("        oct = oct + 1u;\n    }\n");
    for r in 0..8 {
        s.push_str(&format!("    let sg{r} = subgroupAdd(partial{r});\n"));
    }
    s.push_str("    if (sg_lane == 0u) {\n");
    for r in 0..8 {
        s.push_str(&format!(
            "        if (o{r} < params.out_dim) {{\n            y[t * params.out_dim + o{r}] = sg{r};\n        }}\n"
        ));
    }
    s.push_str("    }\n}\n");
    s
}

/// Prism's ternary pair, which the integer-dot kernels above serve.
pub fn is_ternary(ggml_type: u32) -> bool {
    ggml_type == GGML_TYPE_PTQ1_0 || ggml_type == GGML_TYPE_PQ2_0
}

/// Whether `ggml_type` decodes through a word-reading `block_dot`.
pub fn has_words_block_dot(ggml_type: u32) -> bool {
    block_hoisted_words_middle(ggml_type).is_some()
}

fn block_hoisted_words_middle(ggml_type: u32) -> Option<&'static str> {
    if !crate::engine::env::flag_on_unless_disabled("ORANGU_BLOCK_DOT_WORDS") {
        return None;
    }
    // `ORANGU_Q2K_STUB=loads|alu`: a **wrong** `Q2_K` routine for splitting
    // the kernel's time — every load with the arithmetic reduced to a sum
    // of the words, or the full arithmetic on words made up from the lane
    // (the activations still read). Timing only; pair with
    // `ORANGU_VK_NO_SELFCHECK=1` or the start-up check retires it.
    if ggml_type == GGML_TYPE_Q2_K {
        match std::env::var("ORANGU_Q2K_STUB").as_deref() {
            Ok("loads") => return Some(Q2_K_STUB_LOADS_MIDDLE),
            Ok("alu") => return Some(Q2_K_STUB_ALU_MIDDLE),
            _ => {}
        }
    }
    Some(match ggml_type {
        t if t == GGML_TYPE_Q2_K => Q2_K_WORDS_MIDDLE,
        t if t == GGML_TYPE_Q3_K => Q3_K_WORDS_MIDDLE,
        t if t == GGML_TYPE_IQ4_XS => IQ4_XS_WORDS_MIDDLE,
        t if t == GGML_TYPE_IQ3_S => IQ3_S_WORDS_MIDDLE,
        t if t == GGML_TYPE_IQ2_S => IQ2_S_WORDS_MIDDLE,
        t if t == GGML_TYPE_PQ2_0 => PQ2_0_WORDS_MIDDLE,
        t if t == GGML_TYPE_PTQ1_0 => PTQ1_0_WORDS_MIDDLE,
        _ => return None,
    })
}

/// [`PRELUDE_XV`]'s integer-dot twin: the activation row arrives already
/// quantized to 8 bits (`shader_source_quantize_q8`'s layout at binding 1 —
/// per 32-element block `[d, sumq, qs0..qs7]`, ten words), and a lane's
/// sixteen elements are four `dot4I8Packed`s of a weight word against a
/// quant word, rescaled by the block's `d`. The sum of the sixteen quants,
/// where a type's minimum term needs it, is the same dot against `0x01010101`.
const PRELUDE_Q8: &str = r#"
struct Meta {
    in_dim: u32,
    out_dim: u32,
    n_tokens: u32,
    row_bytes: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> q8x: array<u32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> params: Meta;

fn read_u8(byte_offset: u32) -> u32 {
    let word = weights[byte_offset >> 2u];
    let shift = (byte_offset & 3u) * 8u;
    return (word >> shift) & 0xFFu;
}

fn read_u16_even(byte_offset: u32) -> u32 {
    return (weights[byte_offset >> 2u] >> ((byte_offset & 2u) * 8u)) & 0xFFFFu;
}

fn read_u32_at(byte_offset: u32) -> u32 {
    let index = byte_offset >> 2u;
    let shift = (byte_offset & 3u) * 8u;
    let lo = weights[index] >> shift;
    let hi = weights[index + 1u] << ((32u - shift) & 31u);
    return lo | select(hi, 0u, shift == 0u);
}

// Four words starting at an even byte offset, from the five aligned words
// that cover them.
fn read_4words_even(byte_offset: u32) -> vec4<u32> {
    let i = byte_offset >> 2u;
    let w0 = weights[i];
    let w1 = weights[i + 1u];
    let w2 = weights[i + 2u];
    let w3 = weights[i + 3u];
    if ((byte_offset & 2u) == 0u) {
        return vec4<u32>(w0, w1, w2, w3);
    }
    let w4 = weights[i + 4u];
    return vec4<u32>((w0 >> 16u) | (w1 << 16u), (w1 >> 16u) | (w2 << 16u), (w2 >> 16u) | (w3 << 16u), (w3 >> 16u) | (w4 << 16u));
}

fn f16_to_f32(bits: u32) -> f32 {
    return unpack2x16float(bits & 0xFFFFu).x;
}

// The quant words of the sixteen activations at element `e` (a multiple of
// 16) and the block scale they carry.
fn q8_words(e: u32) -> vec4<u32> {
    let b = (e / 32u) * 10u + 2u + (e % 32u) / 4u;
    return vec4<u32>(q8x[b], q8x[b + 1u], q8x[b + 2u], q8x[b + 3u]);
}
fn q8_scale(e: u32) -> f32 {
    return bitcast<f32>(q8x[(e / 32u) * 10u]);
}
fn idot4(w: vec4<u32>, x: vec4<u32>) -> i32 {
    return dot4I8Packed(w.x, x.x) + dot4I8Packed(w.y, x.y) + dot4I8Packed(w.z, x.z) + dot4I8Packed(w.w, x.w);
}
// The quant word of the four activations at element `e` (a multiple of 4).
fn q8_word(e: u32) -> u32 {
    return q8x[(e / 32u) * 10u + 2u + (e % 32u) / 4u];
}
// Bits 0..4 of `x` to bit 4 of bytes 0..4 — a nibble's fifth bit per lane.
fn spread_bits4(x: u32) -> u32 {
    return ((x & 1u) << 4u) | ((x & 2u) << 11u) | ((x & 4u) << 18u) | ((x & 8u) << 25u);
}
// Bytewise `v - k` for bytes below 128, as signed bytes.
fn sub_bytes(v: u32, k: u32) -> u32 {
    return ((v | 0x80808080u) - k) ^ 0x80808080u;
}
fn isum4(x: vec4<u32>) -> i32 {
    return idot4(vec4<u32>(0x01010101u), x);
}
// Bits 0..4 of `x` to bit 0 of bytes 0..4.
fn spread_bits(x: u32) -> u32 {
    return (x & 1u) | ((x & 2u) << 7u) | ((x & 4u) << 14u) | ((x & 8u) << 21u);
}
// A lattice word's four small positive bytes, each negated where its bit
// of the four-bit `signs` is set (two's complement per lane, no carry).
fn iq_signed_bytes(g: u32, signs: u32) -> u32 {
    let m = spread_bits(signs) * 0xFFu;
    return (g ^ m) + (m & 0x01010101u);
}
"#;

/// `block_q2_K` on the integer dot: a lane's sixteen elements are one
/// sub-block half, four words at one shift; the minimum term is the quant
/// sum.
const Q2_K_I8_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 84u;
const BLOCK_ELEMS: u32 = 256u;
const LANES_PER_BLOCK: u32 = 16u;
fn block_dot(byte_offset: u32, x_off: u32, sub: u32) -> f32 {
    let w0 = byte_offset / 4u;
    let hdr = weights[w0 + 20u];
    let d = f16_to_f32(hdr & 0xFFFFu);
    let dmin = f16_to_f32(hdr >> 16u);
    let n = sub / 8u;
    let s = (sub % 8u) / 2u;
    let h = sub % 2u;
    let si = n * 8u + s * 2u + h;
    let sc = (weights[w0 + si / 4u] >> (8u * (si % 4u))) & 0xFFu;
    let qb = w0 + 4u + n * 8u + h * 4u;
    let sh = 2u * s;
    let q = (vec4<u32>(weights[qb], weights[qb + 1u], weights[qb + 2u], weights[qb + 3u]) >> vec4<u32>(sh)) & vec4<u32>(0x03030303u);
    let e = x_off + sub * 16u;
    let x = q8_words(e);
    return q8_scale(e) * (d * f32(sc & 0xFu) * f32(idot4(q, x)) - dmin * f32(sc >> 4u) * f32(isum4(x)));
}
"#;

/// `block_q3_K` on the integer dot: the 2-bit fields with the `hmask` bit
/// as the third, minus four bytewise, signed into the dot.
const Q3_K_I8_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 110u;
const BLOCK_ELEMS: u32 = 256u;
const LANES_PER_BLOCK: u32 = 16u;
fn block_dot(byte_offset: u32, x_off: u32, sub: u32) -> f32 {
    let d_all = f16_to_f32(read_u16_even(byte_offset + 108u));
    let n = sub / 8u;
    let s = (sub % 8u) / 2u;
    let h = sub % 2u;
    let lo_byte = read_u8(byte_offset + 96u + (sub % 8u));
    let low = select(lo_byte >> 4u, lo_byte & 0xFu, sub < 8u);
    let high = (read_u8(byte_offset + 104u + (sub % 4u)) >> (2u * (sub / 4u))) & 3u;
    let dl = d_all * (f32(low | (high << 4u)) - 32.0);
    let qs = read_4words_even(byte_offset + 32u + n * 32u + h * 16u);
    let hm = read_4words_even(byte_offset + h * 16u);
    let m = n * 4u + s;
    let q2 = (qs >> vec4<u32>(2u * s)) & vec4<u32>(0x03030303u);
    let hb = (hm >> vec4<u32>(m)) & vec4<u32>(0x01010101u);
    let v = q2 | (hb << vec4<u32>(2u));
    let q = ((v | vec4<u32>(0x80808080u)) - vec4<u32>(0x04040404u)) ^ vec4<u32>(0x80808080u);
    let e = x_off + sub * 16u;
    return q8_scale(e) * dl * f32(idot4(q, q8_words(e)));
}
"#;

/// `block_iq4_xs` on the integer dot: each nibble through the value table
/// to a signed byte.
const IQ4_XS_I8_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 136u;
const BLOCK_ELEMS: u32 = 256u;
const LANES_PER_BLOCK: u32 = 16u;
fn kv_byte(i: u32) -> u32 {
    return (iq_grids[KVALUES_IQ4NL_OFF + (i >> 2u)] >> ((i & 3u) * 8u)) & 0xFFu;
}
fn kv_word(n: u32) -> u32 {
    return kv_byte(n & 0xFu) | (kv_byte((n >> 8u) & 0xFu) << 8u) | (kv_byte((n >> 16u) & 0xFu) << 16u) | (kv_byte(n >> 24u) << 24u);
}
fn block_dot(byte_offset: u32, x_off: u32, sub: u32) -> f32 {
    let w0 = byte_offset / 4u;
    let hdr = weights[w0];
    let d = f16_to_f32(hdr & 0xFFFFu);
    let scales_h = hdr >> 16u;
    let ib = sub / 2u;
    let nsh = 4u * (sub % 2u);
    let low = (weights[w0 + 1u] >> (8u * (ib / 2u) + 4u * (ib % 2u))) & 0xFu;
    let high = (scales_h >> (2u * ib)) & 3u;
    let dl = d * (f32(low | (high << 4u)) - 32.0);
    let qb = w0 + 2u + 4u * ib;
    let nib = (vec4<u32>(weights[qb], weights[qb + 1u], weights[qb + 2u], weights[qb + 3u]) >> vec4<u32>(nsh)) & vec4<u32>(0x0F0F0F0Fu);
    let q = vec4<u32>(kv_word(nib.x), kv_word(nib.y), kv_word(nib.z), kv_word(nib.w));
    let e = x_off + sub * 16u;
    return q8_scale(e) * dl * f32(idot4(q, q8_words(e)));
}
"#;

/// `block_iq3_s` on the integer dot: four lattice words with their sign
/// nibbles applied.
const IQ3_S_I8_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 110u;
const BLOCK_ELEMS: u32 = 256u;
const LANES_PER_BLOCK: u32 = 16u;
fn block_dot(byte_offset: u32, x_off: u32, sub: u32) -> f32 {
    let d = f16_to_f32(read_u16_even(byte_offset));
    let ib = sub / 2u;
    let h = sub % 2u;
    let sc = (read_u8(byte_offset + 106u + ib / 2u) >> (4u * (ib % 2u))) & 0xFu;
    let db = d * f32(1u + 2u * sc);
    let qh = read_u8(byte_offset + 66u + ib);
    let io = byte_offset + 2u + 8u * ib + 4u * h;
    let idx = read_u16_even(io) | (read_u16_even(io + 2u) << 16u);
    let sg = read_u16_even(byte_offset + 74u + 4u * ib + 2u * h);
    let bit = 4u * h;
    let g0 = iq_grids[IQ3S_GRID_OFF + ((idx & 0xFFu) | (((qh >> bit) & 1u) << 8u))];
    let g1 = iq_grids[IQ3S_GRID_OFF + (((idx >> 8u) & 0xFFu) | (((qh >> (bit + 1u)) & 1u) << 8u))];
    let g2 = iq_grids[IQ3S_GRID_OFF + (((idx >> 16u) & 0xFFu) | (((qh >> (bit + 2u)) & 1u) << 8u))];
    let g3 = iq_grids[IQ3S_GRID_OFF + ((idx >> 24u) | (((qh >> (bit + 3u)) & 1u) << 8u))];
    let q = vec4<u32>(iq_signed_bytes(g0, sg), iq_signed_bytes(g1, sg >> 4u), iq_signed_bytes(g2, sg >> 8u), iq_signed_bytes(g3, sg >> 12u));
    let e = x_off + sub * 16u;
    return q8_scale(e) * db * f32(idot4(q, q8_words(e)));
}
"#;

/// `block_iq2_s` on the integer dot: two 64-bit lattice entries with their
/// sign bytes, the half's scale nibble.
const IQ2_S_I8_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 82u;
const BLOCK_ELEMS: u32 = 256u;
const LANES_PER_BLOCK: u32 = 16u;
fn block_dot(byte_offset: u32, x_off: u32, sub: u32) -> f32 {
    let d = f16_to_f32(read_u16_even(byte_offset));
    let ib = sub / 2u;
    let h = sub % 2u;
    let l0 = 2u * h;
    let sc = (read_u8(byte_offset + 74u + ib) >> (4u * h)) & 0xFu;
    let db = d * (0.5 + f32(sc)) * 0.25;
    let qh = read_u8(byte_offset + 66u + ib);
    let idx2 = read_u16_even(byte_offset + 2u + 4u * ib + l0);
    let sg2 = read_u16_even(byte_offset + 34u + 4u * ib + l0);
    let i0 = (idx2 & 0xFFu) | ((qh << (8u - 2u * l0)) & 0x300u);
    let i1 = (idx2 >> 8u) | ((qh << (6u - 2u * l0)) & 0x300u);
    let e0 = IQ2S_GRID_OFF + 2u * i0;
    let e1 = IQ2S_GRID_OFF + 2u * i1;
    let q = vec4<u32>(iq_signed_bytes(iq_grids[e0], sg2), iq_signed_bytes(iq_grids[e0 + 1u], sg2 >> 4u), iq_signed_bytes(iq_grids[e1], sg2 >> 8u), iq_signed_bytes(iq_grids[e1 + 1u], sg2 >> 12u));
    let e = x_off + sub * 16u;
    return q8_scale(e) * db * f32(idot4(q, q8_words(e)));
}
"#;

/// The legacy types on the integer dot: a 32-element block is one q8
/// block, and a lane's four elements one word of it. Lane `sub` (0..8)
/// takes elements `4·sub..`: the low nibbles of bytes `4·sub..` for
/// `sub < 4`, the high nibbles of bytes `4·(sub − 4)..` above — which is
/// why the fifth-bit word's bits `4·sub..` are the lane's in either case.
/// `qs` sits at a two-byte offset in every layout (`d` is an `f16`), so it
/// is read through `read_u32_at`.
const Q4_0_I8_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 18u;
const BLOCK_ELEMS: u32 = 32u;
const LANES_PER_BLOCK: u32 = 8u;
fn block_dot(byte_offset: u32, x_off: u32, sub: u32) -> f32 {
    let d = f16_to_f32(read_u16_even(byte_offset));
    let w = read_u32_at(byte_offset + 2u + 4u * (sub % 4u));
    let nib = select(w & 0x0F0F0F0Fu, (w >> 4u) & 0x0F0F0F0Fu, sub >= 4u);
    let e = x_off + 4u * sub;
    return q8_scale(e) * d * f32(dot4I8Packed(sub_bytes(nib, 0x08080808u), q8_word(e)));
}
"#;

/// See [`Q4_0_I8_MIDDLE`]; `m` is the stored minimum, applied through
/// the four activations' quant sum.
const Q4_1_I8_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 20u;
const BLOCK_ELEMS: u32 = 32u;
const LANES_PER_BLOCK: u32 = 8u;
fn block_dot(byte_offset: u32, x_off: u32, sub: u32) -> f32 {
    let hdr = weights[byte_offset / 4u];
    let d = f16_to_f32(hdr & 0xFFFFu);
    let m = f16_to_f32(hdr >> 16u);
    let w = weights[byte_offset / 4u + 1u + (sub % 4u)];
    let nib = select(w & 0x0F0F0F0Fu, (w >> 4u) & 0x0F0F0F0Fu, sub >= 4u);
    let e = x_off + 4u * sub;
    let xq = q8_word(e);
    return q8_scale(e) * (d * f32(dot4I8Packed(nib, xq)) + m * f32(dot4I8Packed(0x01010101u, xq)));
}
"#;

/// See [`Q4_0_I8_MIDDLE`]; the fifth bits from `qh` at bytes 2..6.
const Q5_0_I8_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 22u;
const BLOCK_ELEMS: u32 = 32u;
const LANES_PER_BLOCK: u32 = 8u;
fn block_dot(byte_offset: u32, x_off: u32, sub: u32) -> f32 {
    let d = f16_to_f32(read_u16_even(byte_offset));
    let qh = read_u32_at(byte_offset + 2u);
    let w = read_u32_at(byte_offset + 6u + 4u * (sub % 4u));
    let nib = select(w & 0x0F0F0F0Fu, (w >> 4u) & 0x0F0F0F0Fu, sub >= 4u) | spread_bits4(qh >> (4u * sub));
    let e = x_off + 4u * sub;
    return q8_scale(e) * d * f32(dot4I8Packed(sub_bytes(nib, 0x10101010u), q8_word(e)));
}
"#;

/// See [`Q5_0_I8_MIDDLE`] and [`Q4_1_I8_MIDDLE`].
const Q5_1_I8_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 24u;
const BLOCK_ELEMS: u32 = 32u;
const LANES_PER_BLOCK: u32 = 8u;
fn block_dot(byte_offset: u32, x_off: u32, sub: u32) -> f32 {
    let hdr = weights[byte_offset / 4u];
    let d = f16_to_f32(hdr & 0xFFFFu);
    let m = f16_to_f32(hdr >> 16u);
    let qh = weights[byte_offset / 4u + 1u];
    let w = weights[byte_offset / 4u + 2u + (sub % 4u)];
    let nib = select(w & 0x0F0F0F0Fu, (w >> 4u) & 0x0F0F0F0Fu, sub >= 4u) | spread_bits4(qh >> (4u * sub));
    let e = x_off + 4u * sub;
    let xq = q8_word(e);
    return q8_scale(e) * (d * f32(dot4I8Packed(nib, xq)) + m * f32(dot4I8Packed(0x01010101u, xq)));
}
"#;

/// `block_q8_0` on the integer dot: the lane's four bytes are already the
/// signed quants.
const Q8_0_I8_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 34u;
const BLOCK_ELEMS: u32 = 32u;
const LANES_PER_BLOCK: u32 = 8u;
fn block_dot(byte_offset: u32, x_off: u32, sub: u32) -> f32 {
    let d = f16_to_f32(read_u16_even(byte_offset));
    let w = read_u32_at(byte_offset + 2u + 4u * sub);
    let e = x_off + 4u * sub;
    return q8_scale(e) * d * f32(dot4I8Packed(w, q8_word(e)));
}
"#;

/// The integer-dot `block_dot`s: the same contract as
/// [`block_hoisted_middle`] against a q8 activation. The word-reading float
/// forms above are bound by the arithmetic per element, not the loads; the
/// packed dot does four multiply-adds per instruction and takes the weight
/// bytes as they are.
fn block_hoisted_i8_middle(ggml_type: u32) -> Option<&'static str> {
    Some(match ggml_type {
        t if t == GGML_TYPE_Q2_K => Q2_K_I8_MIDDLE,
        t if t == GGML_TYPE_Q3_K => Q3_K_I8_MIDDLE,
        t if t == GGML_TYPE_IQ4_XS => IQ4_XS_I8_MIDDLE,
        t if t == GGML_TYPE_IQ3_S => IQ3_S_I8_MIDDLE,
        t if t == GGML_TYPE_IQ2_S => IQ2_S_I8_MIDDLE,
        t if t == GGML_TYPE_Q4_0 => Q4_0_I8_MIDDLE,
        t if t == GGML_TYPE_Q4_1 => Q4_1_I8_MIDDLE,
        t if t == GGML_TYPE_Q5_0 => Q5_0_I8_MIDDLE,
        t if t == GGML_TYPE_Q5_1 => Q5_1_I8_MIDDLE,
        t if t == GGML_TYPE_Q8_0 => Q8_0_I8_MIDDLE,
        _ => return None,
    })
}

/// The integer-dot block-hoisted decode kernel for `ggml_type`, or `None`
/// for a type without one. Binding 1 is the q8 activation
/// (`shader_source_quantize_q8`); the `main` and its row batching are
/// [`block_hoisted_suffix`]'s.
pub fn shader_source_reduce_block_hoisted_i8(
    ggml_type: u32,
    n_rows: usize,
    subgroup: bool,
) -> Option<String> {
    let middle = block_hoisted_i8_middle(ggml_type)?;
    let grids = if needs_iq_grids(ggml_type) {
        IQ_GRID_PRELUDE
    } else {
        ""
    };
    let suffix = block_hoisted_suffix(n_rows, subgroup);
    Some(format!("{PRELUDE_Q8}\n{grids}\n{middle}\n{suffix}"))
}

/// The word-reading `block_dot`s on the **light** skeleton: the 32-lane,
/// sixteen-lanes-per-super-block, two-blocks-in-flight `main` of
/// `shader_source_reduce_q4k_light` (the `Q5_K`/`Q6_K` twins' form), with a
/// type's `block_dot(byte_offset, x_off, lane)` as the per-block routine.
///
/// The same `block_dot`s under [`block_hoisted_suffix`] — 64 lanes, four
/// blocks in flight, `n_rows` rows — reach half the `Q4_K` light kernel's
/// rate per byte inside a decode step, and every knob on that `main` was
/// measured without moving it (rows, blocks per trip, the reduce, the
/// element mapping, the integer dot). The light `main` is the one skeleton
/// at parity with the reference in situ, so this puts the same routines on
/// it — and measured level with the block-hoisted form in a decode step,
/// which rules the skeleton out too. `ORANGU_KQ_LIGHT=1` selects it.
pub fn shader_source_reduce_kq_light(
    ggml_type: u32,
    n_rows: usize,
    subgroup: bool,
) -> Option<String> {
    let middle = block_hoisted_words_middle(ggml_type)?;
    let grids = if needs_iq_grids(ggml_type) {
        IQ_GRID_PRELUDE
    } else {
        ""
    };
    let mut s = format!("{PRELUDE_XV}\n{grids}\n{middle}\n");
    s.push_str(&format!(
        "\nvar<workgroup> psums: array<f32, {}>;\n\n",
        n_rows * 32
    ));
    s.push_str("@compute @workgroup_size(32)\nfn main(\n    @builtin(workgroup_id) wid: vec3<u32>,\n    @builtin(local_invocation_id) lid: vec3<u32>,\n    @builtin(num_workgroups) nwg: vec3<u32>,");
    s.push_str(subgroup_entry_params(subgroup));
    s.push_str("\n) {\n");
    s.push_str(&format!(
        "    let n_row_groups = (params.out_dim + {}u) / {n_rows}u;\n",
        n_rows - 1
    ));
    s.push_str("    let flat = wid.x + wid.y * nwg.x + wid.z * nwg.x * nwg.y;\n    if (flat >= n_row_groups * params.n_tokens) {\n        return;\n    }\n");
    s.push_str("    let rg = flat / params.n_tokens;\n    let t = flat % params.n_tokens;\n");
    s.push_str(&format!("    let o0 = rg * {n_rows}u;\n"));
    for i in 1..n_rows {
        s.push_str(&format!("    let o{i} = o0 + {i}u;\n"));
    }
    // A row past the end is computed on a clamped row and never stored, so
    // every row's loads of a trip sit in one basic block.
    for i in 0..n_rows {
        s.push_str(&format!(
            "    let oc{i} = min(o{i}, params.out_dim - 1u);\n"
        ));
    }
    s.push_str("    let tid = lid.x;\n    let itid = tid % 16u;\n    let ix = tid / 16u;\n");
    s.push_str(
        "    let num_blocks = params.in_dim / BLOCK_ELEMS;\n    let x_tok = t * params.in_dim;\n",
    );
    for i in 0..n_rows {
        s.push_str(&format!("    var acc{i}: f32 = 0.0;\n"));
    }
    s.push_str("    var i: u32 = ix;\n    loop {\n        if (i >= num_blocks) {\n            break;\n        }\n        let xrb = x_tok + i * BLOCK_ELEMS;\n");
    for r in 0..n_rows {
        s.push_str(&format!(
            "        acc{r} = acc{r} + block_dot(oc{r} * params.row_bytes + i * BLOCK_BYTES, xrb, itid);\n"
        ));
    }
    s.push_str("        i = i + 2u;\n    }\n");
    s.push_str(&light_reduce(n_rows, subgroup));
    s.push_str("}\n");
    Some(s)
}

/// The per-type half of [`block_hoisted_suffix`], or `None` for a type that
/// has no block-hoisted implementation yet and so keeps the element-wise
/// path.
///
/// **Deliberately a growable list, not a closed one.** Every entry is a
/// dozen lines against a contract that says nothing about block size or bit
/// packing, and the list now covers every type this backend has a shader for
/// *except* the three the block-unroll already serves faster (`Q4_K`,
/// `Q5_K`, `Q6_K` — see [`unroll_suffix`]) and the pass-through float types,
/// which have no block header to hoist.
///
/// The `IQ*` entries are what a `UD`-style mixed quantization actually needs:
/// `unsloth/Qwen3.8-27B-GGUF:Q3_K_XL` is 357 `IQ4_XS` tensors and 130
/// `IQ3_S` against a single `Q3_K` one, so before these it decoded almost
/// entirely on the element-wise path while its name promised a K-quant.
fn block_hoisted_middle(ggml_type: u32) -> Option<&'static str> {
    Some(match ggml_type {
        t if t == GGML_TYPE_Q4_0 => Q4_0_BLOCK_MIDDLE,
        t if t == GGML_TYPE_Q4_1 => Q4_1_BLOCK_MIDDLE,
        t if t == GGML_TYPE_Q5_0 => Q5_0_BLOCK_MIDDLE,
        t if t == GGML_TYPE_Q5_1 => Q5_1_BLOCK_MIDDLE,
        t if t == GGML_TYPE_Q8_0 => Q8_0_BLOCK_MIDDLE,
        t if t == GGML_TYPE_Q2_K => Q2_K_BLOCK_MIDDLE,
        t if t == GGML_TYPE_Q3_K => Q3_K_BLOCK_MIDDLE,
        t if t == GGML_TYPE_IQ1_S => IQ1_S_BLOCK_MIDDLE,
        t if t == GGML_TYPE_IQ1_M => IQ1_M_BLOCK_MIDDLE,
        t if t == GGML_TYPE_IQ2_XXS => IQ2_XXS_BLOCK_MIDDLE,
        t if t == GGML_TYPE_IQ2_XS => IQ2_XS_BLOCK_MIDDLE,
        t if t == GGML_TYPE_IQ2_S => IQ2_S_BLOCK_MIDDLE,
        t if t == GGML_TYPE_IQ3_XXS => IQ3_XXS_BLOCK_MIDDLE,
        t if t == GGML_TYPE_IQ3_S => IQ3_S_BLOCK_MIDDLE,
        t if t == GGML_TYPE_IQ4_NL => IQ4_NL_BLOCK_MIDDLE,
        t if t == GGML_TYPE_IQ4_XS => IQ4_XS_BLOCK_MIDDLE,
        t if t == GGML_TYPE_MXFP4 => MXFP4_BLOCK_MIDDLE,
        _ => return None,
    })
}

/// The complete WGSL for `ggml_type`'s **block-hoisted** decode pipeline, or
/// `None` if it has no [`block_hoisted_middle`] yet.
///
/// Same three-layer assembly as [`shader_source_reduce`] — shared I/O
/// prelude, per-type quantization middle, algorithm suffix — with only the
/// middle and suffix differing. That is the point: a new decode algorithm
/// costs one suffix, and a new quantization costs one middle, and neither
/// costs a kernel per (type × algorithm).
pub fn shader_source_reduce_block_hoisted(
    ggml_type: u32,
    n_rows: usize,
    subgroup: bool,
) -> Option<String> {
    let suffix = block_hoisted_suffix(n_rows, subgroup);
    if let Some(middle) = block_hoisted_words_middle(ggml_type) {
        let grids = if needs_iq_grids(ggml_type) {
            IQ_GRID_PRELUDE
        } else {
            ""
        };
        return Some(format!("{PRELUDE_XV}\n{grids}\n{middle}\n{suffix}"));
    }
    let middle = block_hoisted_middle(ggml_type)?;
    let prelude = prelude_for(ggml_type);
    Some(format!("{prelude}\n{middle}\n{suffix}"))
}

fn coop_middle(ggml_type: u32) -> Option<&'static str> {
    Some(match ggml_type {
        t if t == GGML_TYPE_F32 => F32_COOP_MIDDLE,
        t if t == GGML_TYPE_F16 => F16_COOP_MIDDLE,
        t if t == GGML_TYPE_BF16 => BF16_COOP_MIDDLE,
        t if t == GGML_TYPE_Q4_0 => Q4_0_COOP_MIDDLE,
        t if t == GGML_TYPE_Q4_1 => Q4_1_COOP_MIDDLE,
        t if t == GGML_TYPE_Q5_0 => Q5_0_COOP_MIDDLE,
        t if t == GGML_TYPE_Q5_1 => Q5_1_COOP_MIDDLE,
        t if t == GGML_TYPE_Q8_0 => Q8_0_COOP_MIDDLE,
        t if t == GGML_TYPE_Q2_K => Q2_K_COOP_MIDDLE,
        t if t == GGML_TYPE_Q3_K => Q3_K_COOP_MIDDLE,
        t if t == GGML_TYPE_Q4_K => Q4_K_COOP_MIDDLE,
        t if t == GGML_TYPE_Q5_K => Q5_K_COOP_MIDDLE,
        t if t == GGML_TYPE_Q6_K => Q6_K_COOP_MIDDLE,
        t if t == GGML_TYPE_IQ1_S => IQ1_S_COOP_MIDDLE,
        t if t == GGML_TYPE_IQ1_M => IQ1_M_COOP_MIDDLE,
        t if t == GGML_TYPE_IQ2_XXS => IQ2_XXS_COOP_MIDDLE,
        t if t == GGML_TYPE_IQ2_XS => IQ2_XS_COOP_MIDDLE,
        t if t == GGML_TYPE_IQ2_S => IQ2_S_COOP_MIDDLE,
        t if t == GGML_TYPE_IQ3_XXS => IQ3_XXS_COOP_MIDDLE,
        t if t == GGML_TYPE_IQ3_S => IQ3_S_COOP_MIDDLE,
        t if t == GGML_TYPE_IQ4_NL => IQ4_NL_COOP_MIDDLE,
        t if t == GGML_TYPE_IQ4_XS => IQ4_XS_COOP_MIDDLE,
        t if t == GGML_TYPE_MXFP4 => MXFP4_COOP_MIDDLE,
        t if t == GGML_TYPE_PQ2_0 => PQ2_0_COOP_MIDDLE,
        t if t == GGML_TYPE_PTQ1_0 => PTQ1_0_COOP_MIDDLE,
        _ => return None,
    })
}

/// Whether `ggml_type`'s `dequant_element` reads the `IQ*` codebooks, and so
/// needs [`IQ_GRID_PRELUDE`]'s `@binding(4)` declaration.
fn needs_iq_grids(ggml_type: u32) -> bool {
    ggml_type == GGML_TYPE_IQ1_S
        || ggml_type == GGML_TYPE_IQ1_M
        || ggml_type == GGML_TYPE_IQ2_XXS
        || ggml_type == GGML_TYPE_IQ2_XS
        || ggml_type == GGML_TYPE_IQ2_S
        || ggml_type == GGML_TYPE_IQ3_XXS
        || ggml_type == GGML_TYPE_IQ3_S
        || ggml_type == GGML_TYPE_IQ4_NL
        || ggml_type == GGML_TYPE_IQ4_XS
}

/// [`PRELUDE`] for `ggml_type`, with [`IQ_GRID_PRELUDE`] appended for the
/// codebook types. Kept off the other types' modules so a shader only ever
/// declares the bindings it reads.
fn prelude_for(ggml_type: u32) -> String {
    if needs_iq_grids(ggml_type) {
        format!("{PRELUDE}\n{IQ_GRID_PRELUDE}")
    } else if ggml_type == GGML_TYPE_MXFP4 {
        format!("{PRELUDE}\n{MXFP4_PRELUDE}")
    } else {
        PRELUDE.to_string()
    }
}

/// A decode matvec for the two float weight types (`F32`, `F16`), with its
/// loads in flight. The generic reduce kernel serves them by reading each
/// weight *byte by byte* through `dequant_element` and each `x` element
/// alone, one dependent round trip per element per thread — on a model
/// whose per-layer-embedding gate and projection are F32 that ran at a
/// fraction of what the same bytes stream at through the K-quant kernels.
///
/// One workgroup of 64 per output row (the light kernels' geometry, so the
/// dispatch-count math is unchanged), the row read as `vec4` of the weight
/// type and `x` as `vec4<f32>`, four of each issued together per iteration
/// so eight loads are outstanding at once, then the shared combine every
/// reduce kernel uses. `row_bytes` and `in_dim` must be multiples of 16 and
/// 4 (every model width is); the caller falls back otherwise.
pub fn shader_source_reduce_float_wide(ggml_type: u32, subgroup: bool) -> Option<String> {
    // The weight array's element holds four weights; its byte width is what
    // turns `row_bytes` into a row's element base.
    let (weight_decl, load, elem_bytes) = match ggml_type {
        t if t == GGML_TYPE_F32 => (
            "@group(0) @binding(0) var<storage, read> weights: array<vec4<f32>>;",
            "weights[base + {k}]",
            16,
        ),
        t if t == GGML_TYPE_F16 => (
            // Two `f16` per `u32`; a `vec2<u32>` is four weights.
            "@group(0) @binding(0) var<storage, read> weights: array<vec2<u32>>;",
            "unpack_f16x4(weights[base + {k}])",
            8,
        ),
        _ => return None,
    };
    let mut s = String::new();
    s.push_str(
        "struct Meta {\n    in_dim: u32,\n    out_dim: u32,\n    n_tokens: u32,\n    row_bytes: u32,\n}\n\n",
    );
    s.push_str(weight_decl);
    s.push_str("\n@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;\n");
    s.push_str("@group(0) @binding(2) var<storage, read_write> y: array<f32>;\n");
    s.push_str("@group(0) @binding(3) var<uniform> params: Meta;\n\n");
    if ggml_type == GGML_TYPE_F16 {
        s.push_str(
            "fn unpack_f16x4(v: vec2<u32>) -> vec4<f32> {\n    let a = unpack2x16float(v.x);\n    let b = unpack2x16float(v.y);\n    return vec4<f32>(a.x, a.y, b.x, b.y);\n}\n\n",
        );
    }
    s.push_str("var<workgroup> partial_sums: array<f32, 64>;\n\n");
    s.push_str("@compute @workgroup_size(64)\nfn main(\n    @builtin(workgroup_id) wid: vec3<u32>,\n    @builtin(local_invocation_id) lid: vec3<u32>,\n    @builtin(num_workgroups) nwg: vec3<u32>,");
    s.push_str(subgroup_entry_params(subgroup));
    s.push_str("\n) {\n");
    s.push_str("    let flat = wid.x + wid.y * nwg.x + wid.z * nwg.x * nwg.y;\n    if (flat >= params.out_dim * params.n_tokens) {\n        return;\n    }\n");
    s.push_str("    let o0 = flat / params.n_tokens;\n    let t = flat % params.n_tokens;\n    let local = lid.x;\n");
    s.push_str(&format!(
        "    let n4 = params.in_dim / 4u;\n    let base = o0 * (params.row_bytes / {elem_bytes}u);\n    let x_base = t * n4;\n"
    ));
    s.push_str("    var partial0: f32 = 0.0;\n    var k: u32 = local;\n");
    // Four vec4 per iteration, all loads issued before any use.
    s.push_str("    loop {\n        if (k + 192u >= n4) {\n            break;\n        }\n");
    for i in 0..4 {
        s.push_str(&format!(
            "        let w{i} = {};\n",
            load.replace("{k}", &format!("k + {}u", i * 64))
        ));
    }
    for i in 0..4 {
        s.push_str(&format!(
            "        let x{i} = x[x_base + k + {}u];\n",
            i * 64
        ));
    }
    s.push_str("        partial0 = partial0 + dot(w0, x0) + dot(w1, x1) + dot(w2, x2) + dot(w3, x3);\n        k = k + 256u;\n    }\n");
    s.push_str("    loop {\n        if (k >= n4) {\n            break;\n        }\n");
    s.push_str(&format!("        let w = {};\n", load.replace("{k}", "k")));
    s.push_str(
        "        partial0 = partial0 + dot(w, x[x_base + k]);\n        k = k + 64u;\n    }\n\n",
    );
    s.push_str(&reduce_combine_block(1, subgroup));
    s.push_str("}\n");
    Some(s)
}

/// The complete, compiled-ready WGSL source for `ggml_type`'s *reduction*
/// pipeline (see `MAIN_REDUCE_SUFFIX`), or `None` if this backend has no
/// shader for it (the same set `engine::quant` supports on the CPU path —
/// see its module doc for what's missing). Reuses the same [`coop_middle`]
/// `shader_source_coop` does — both dispatch strategies share the
/// exact same `dequant_element` per type, only `MAIN_REDUCE_SUFFIX` vs.
/// `MAIN_COOP_SUFFIX` (and so the resulting compute `main`) differs.
pub fn shader_source_reduce(ggml_type: u32, n_rows: usize, subgroup: bool) -> Option<String> {
    let middle = coop_middle(ggml_type)?;
    let prelude = prelude_for(ggml_type);
    let suffix = main_reduce_suffix(n_rows, subgroup);
    Some(format!("{prelude}\n{middle}\n{suffix}"))
}

/// The **thin-tile reduce** entry point (`fn main`) for an arbitrary
/// `(n_rows, tile)` — the multi-position (speculative / small-chunk-prefill)
/// matmul kernel named as the missing lever. It keeps the reduce path's
/// occupancy (one workgroup per
/// `(n_rows-row group, tile-token group)`, so `ceil(out_dim / n_rows) *
/// ceil(n_tokens / tile)` workgroups — still many small groups, unlike the
/// tiled GEMM's few large ones) **and** amortizes each weight dequant across
/// the tile: at every `k` this workgroup's 64 threads grid-stride over
/// `in_dim`, dequantizing each of `n_rows` weight elements *once* and reusing
/// it across all `tile` tokens' activations. The plain reduce path
/// (`main_reduce_suffix`) re-runs the whole (expensive, for K-quants)
/// `dequant_element` once per token because its workgroups are keyed on a
/// single token; this one runs it `tile`× less. Same bindings as the reduce
/// path (`x`, `y`, `params`, weight) — `x[t*in_dim+k]` / `y[t*out_dim+o]`
/// layout unchanged — so it drops into the existing matmul bind group with
/// only the dispatch count and pipeline swapped. Tree-reduce combine (no
/// subgroup variant yet); the win is the dequant amortization, not the
/// reduction.
fn thin_tile_reduce_suffix(n_rows: usize, tile: usize, subgroup: bool) -> String {
    let (r, t) = (n_rows, tile);
    let mut s = format!(
        "var<workgroup> partial_sums: array<f32, {}>;\n\n",
        r * t * 64
    );
    s.push_str("@compute @workgroup_size(64)\nfn main(\n    @builtin(workgroup_id) wid: vec3<u32>,\n    @builtin(local_invocation_id) lid: vec3<u32>,\n    @builtin(num_workgroups) nwg: vec3<u32>,");
    s.push_str(subgroup_entry_params(subgroup));
    s.push_str("\n) {\n");
    s.push_str(&format!(
        "    let n_row_groups = (params.out_dim + {}u) / {r}u;\n",
        r - 1
    ));
    s.push_str(&format!(
        "    let n_tok_tiles = (params.n_tokens + {}u) / {t}u;\n",
        t - 1
    ));
    s.push_str("    let flat = wid.x + wid.y * nwg.x + wid.z * nwg.x * nwg.y;\n    if (flat >= n_row_groups * n_tok_tiles) {\n        return;\n    }\n");
    s.push_str("    let rg = flat / n_tok_tiles;\n    let tt = flat % n_tok_tiles;\n");
    s.push_str(&format!(
        "    let o_base = rg * {r}u;\n    let t_base = tt * {t}u;\n"
    ));
    for i in 0..r {
        s.push_str(&format!("    let o{i} = o_base + {i}u;\n"));
    }
    s.push_str("    let local = lid.x;\n\n");
    for i in 0..r {
        for j in 0..t {
            s.push_str(&format!("    var p{i}_{j}: f32 = 0.0;\n"));
        }
    }
    s.push_str("    var k: u32 = local;\n    loop {\n        if (k >= params.in_dim) {\n            break;\n        }\n");
    s.push_str("        let block_idx = k / BLOCK_ELEMS;\n        let local_k = k % BLOCK_ELEMS;\n        let block_off = block_idx * BLOCK_BYTES;\n");
    // Dequantize each row's weight element once, reuse across the token tile.
    s.push_str("        let w0 = dequant_element(o0 * params.row_bytes + block_off, local_k);\n");
    for i in 1..r {
        s.push_str(&format!(
            "        var w{i}: f32 = 0.0;\n        if (o{i} < params.out_dim) {{ w{i} = dequant_element(o{i} * params.row_bytes + block_off, local_k); }}\n"
        ));
    }
    for j in 0..t {
        s.push_str(&format!(
            "        var x{j}: f32 = 0.0;\n        if (t_base + {j}u < params.n_tokens) {{ x{j} = x[(t_base + {j}u) * params.in_dim + k]; }}\n"
        ));
    }
    for i in 0..r {
        for j in 0..t {
            s.push_str(&format!("        p{i}_{j} = p{i}_{j} + w{i} * x{j};\n"));
        }
    }
    s.push_str("        k = k + 64u;\n    }\n\n");
    if subgroup {
        // Each of the `r * t` outputs: sum this lane's 64-stride partial
        // across the subgroup, one write per subgroup, then thread 0 folds
        // the `n_sg` subgroup partials — the fast combine the plain reduce
        // path (`reduce_combine_block`) already uses, avoiding the tree
        // reduce's `log2(64)` `workgroupBarrier` rounds.
        for i in 0..r {
            for j in 0..t {
                s.push_str(&format!("    let sg{}_{} = subgroupAdd(p{i}_{j});\n", i, j));
            }
        }
        s.push_str("    if (sg_lane == 0u) {\n");
        for i in 0..r {
            for j in 0..t {
                s.push_str(&format!(
                    "        partial_sums[{}u * 64u + sg_id] = sg{}_{};\n",
                    i * t + j,
                    i,
                    j
                ));
            }
        }
        s.push_str("    }\n    workgroupBarrier();\n    if (local == 0u) {\n");
        for i in 0..r {
            for j in 0..t {
                s.push_str(&format!("        var acc{}_{}: f32 = 0.0;\n", i, j));
            }
        }
        s.push_str("        var si: u32 = 0u;\n        loop {\n            if (si >= n_sg) {\n                break;\n            }\n");
        for i in 0..r {
            for j in 0..t {
                s.push_str(&format!(
                    "            acc{}_{} = acc{}_{} + partial_sums[{}u * 64u + si];\n",
                    i,
                    j,
                    i,
                    j,
                    i * t + j
                ));
            }
        }
        s.push_str("            si = si + 1u;\n        }\n");
        for i in 0..r {
            for j in 0..t {
                s.push_str(&format!(
                    "        if (o{i} < params.out_dim && t_base + {j}u < params.n_tokens) {{\n            y[(t_base + {j}u) * params.out_dim + o{i}] = acc{}_{};\n        }}\n",
                    i, j
                ));
            }
        }
        s.push_str("    }\n}\n");
    } else {
        for i in 0..r {
            for j in 0..t {
                s.push_str(&format!(
                    "    partial_sums[{}u * 64u + local] = p{i}_{j};\n",
                    i * t + j
                ));
            }
        }
        s.push_str("    workgroupBarrier();\n    var stride: u32 = 32u;\n    loop {\n        if (stride == 0u) {\n            break;\n        }\n        if (local < stride) {\n");
        for idx in 0..(r * t) {
            s.push_str(&format!(
                "            partial_sums[{idx}u * 64u + local] = partial_sums[{idx}u * 64u + local] + partial_sums[{idx}u * 64u + local + stride];\n"
            ));
        }
        s.push_str(
            "        }\n        workgroupBarrier();\n        stride = stride / 2u;\n    }\n",
        );
        s.push_str("    if (local == 0u) {\n");
        for i in 0..r {
            for j in 0..t {
                let idx = i * t + j;
                s.push_str(&format!(
                    "        if (o{i} < params.out_dim && t_base + {j}u < params.n_tokens) {{\n            y[(t_base + {j}u) * params.out_dim + o{i}] = partial_sums[{idx}u * 64u];\n        }}\n"
                ));
            }
        }
        s.push_str("    }\n}\n");
    }
    s
}

/// A thin-tile reduce kernel (see [`thin_tile_reduce_suffix`]) for one quant
/// type — same `*_COOP_MIDDLE` `dequant_element` every other reduce/coop
/// kernel uses. `None` for a type this backend has no shader for.
pub fn shader_source_reduce_thin_tile(
    ggml_type: u32,
    n_rows: usize,
    tile: usize,
    subgroup: bool,
) -> Option<String> {
    let middle = coop_middle(ggml_type)?;
    let prelude = prelude_for(ggml_type);
    Some(format!(
        "{prelude}\n{middle}\n{}",
        thin_tile_reduce_suffix(n_rows, tile, subgroup)
    ))
}

/// `Q4_K` only.
/// Dequantizes weight elements *in pairs* (`dequant_pair_f16`, a `Q4_K`-
/// specific restatement of `Q4_K_COOP_MIDDLE`'s `dequant_element` that
/// also skips the redundant `get_scale_min_k4` lookup a pair's two
/// elements would otherwise repeat) and accumulates the dot product as
/// packed `vec2<f16>` instead of two scalar `f32` multiplies — half as
/// many multiply-accumulate ops in the inner loop. Opt-in via
/// `VulkanBackend::packed_dot_f16` (`ORANGU_PACKED_DOT=1`). Not reused by
/// `shader_source_reduce`/`Q4_K_COOP_MIDDLE`'s own `dequant_element`
/// (kept as a separate, self-contained kernel) since a per-pair, `f16`-
/// typed dequant function has a different signature and doesn't compose
/// with the scalar-per-element `MAIN_REDUCE_SUFFIX`/`MAIN_COOP_SUFFIX`
/// bodies both other kernels share.
pub fn shader_source_reduce_q4k_packed_f16() -> String {
    const MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 144u;
const BLOCK_ELEMS: u32 = 256u;

// `k` must be even — `k` and `k + 1` always land in the same low/high-
// nibble half of their 64-wide group (that boundary, 32, is itself even
// and pair-aligned), so this never needs to stitch together two different
// `get_scale_min_k4` lookups for one pair.
fn dequant_pair_f16(byte_offset: u32, k: u32) -> vec2<f16> {
    let d = f16_to_f32(read_u8(byte_offset) | (read_u8(byte_offset + 1u) << 8u));
    let dmin = f16_to_f32(read_u8(byte_offset + 2u) | (read_u8(byte_offset + 3u) << 8u));
    let scales_off = byte_offset + 4u;
    let qs_off = byte_offset + 16u;
    let q_offset = (k / 64u) * 64u;
    let local_in_group = k % 64u;
    let is_base = (q_offset / 64u) * 2u;
    let q_base = qs_off + q_offset / 2u;
    if (local_in_group < 32u) {
        let byte0 = read_u8(q_base + local_in_group);
        let byte1 = read_u8(q_base + local_in_group + 1u);
        let sm = get_scale_min_k4(scales_off, is_base);
        let d1 = d * f32(sm.x);
        let m1 = dmin * f32(sm.y);
        return vec2<f16>(
            f16(d1 * f32(byte0 & 0xFu) - m1),
            f16(d1 * f32(byte1 & 0xFu) - m1),
        );
    }
    let l = local_in_group - 32u;
    let byte0 = read_u8(q_base + l);
    let byte1 = read_u8(q_base + l + 1u);
    let sm = get_scale_min_k4(scales_off, is_base + 1u);
    let d2 = d * f32(sm.x);
    let m2 = dmin * f32(sm.y);
    return vec2<f16>(
        f16(d2 * f32(byte0 >> 4u) - m2),
        f16(d2 * f32(byte1 >> 4u) - m2),
    );
}
"#;
    const SUFFIX: &str = r#"
var<workgroup> partial_sums: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let flat = wid.x + wid.y * nwg.x + wid.z * nwg.x * nwg.y;
    if (flat >= params.out_dim * params.n_tokens) {
        return;
    }
    let o = flat / params.n_tokens;
    let t = flat % params.n_tokens;
    let local = lid.x;
    let row_byte_base = o * params.row_bytes;
    let x_base = t * params.in_dim;

    var partial: f32 = 0.0;
    var k: u32 = local * 2u;
    loop {
        if (k >= params.in_dim) {
            break;
        }
        let block_idx = k / BLOCK_ELEMS;
        let local_k = k % BLOCK_ELEMS;
        let wv = dequant_pair_f16(row_byte_base + block_idx * BLOCK_BYTES, local_k);
        let xv = vec2<f16>(f16(x[x_base + k]), f16(x[x_base + k + 1u]));
        partial = partial + f32(dot(wv, xv));
        k = k + 128u;
    }

    partial_sums[local] = partial;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (local < stride) {
            partial_sums[local] = partial_sums[local] + partial_sums[local + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (local == 0u) {
        y[t * params.out_dim + o] = partial_sums[0];
    }
}
"#;
    // `enable f16;` must precede every global declaration in the whole
    // module (a WGSL rule) — `PRELUDE` already has
    // `struct Meta`/global `var<...>` declarations, so this can't sit
    // inside `MIDDLE` the way the rest of `MIDDLE` conceptually belongs
    // there; it has to lead the concatenated string instead.
    format!("enable f16;\n{PRELUDE}\n{MIDDLE}\n{SUFFIX}")
}

/// Wide vectorized weight loads. Unlike
/// every other kernel in this file, `weights` is bound as
/// `array<vec4<u32>>` (16-byte elements) instead of `array<u32>`, so this
/// needs its own prelude (`PRELUDE_VEC4` below) rather than reusing the
/// shared `PRELUDE` — the WGSL binding type is fixed at module scope, not
/// something a shader can reinterpret per-call the way `read_u8`
/// reinterprets `array<u32>` byte-by-byte.
///
/// Every type's `dequant_element` keeps the exact same `(byte_offset: u32,
/// k: u32) -> f32` signature the byte-wise `*_COOP_MIDDLE` constants use —
/// deliberately, so this reuses `MAIN_REDUCE_SUFFIX` verbatim (the same
/// `REDUCE_N_ROWS`-batched, 4-rows-per-workgroup dispatch the byte-wise
/// reduce kernel already uses) instead of a separate, one-off dispatch
/// shape. `Q4_K`/`Q5_K` (whose block sizes, 144/176 bytes, are both exact
/// multiples of 16) compute `vec4_base = byte_offset / 16u` — always exact
/// for those two types, since every block *and* every row (`row_bytes` is
/// a multiple of `BLOCK_BYTES`) they ever index starts at a 16-byte
/// boundary — and their whole `d`/`dmin`/`scales`
/// header (16 bytes) loads in one `vec4` read instead of up to 9 `read_u8`
/// calls. The other 7 types' blocks aren't 16-byte multiples (`Q6_K`'s 210
/// in particular), so their block starts land at unpredictable, *varying*
/// alignment from one block to the next — `read_word_v4`/
/// `read_word_unaligned_v4`/`read_byte_v4` below handle any `byte_offset`
/// correctly regardless, and each type still consolidates whatever of its
/// own fields are provably word-safe to combine (worked out per type,
/// see each `*_WIDE_MIDDLE` constant's own comment).
///
/// Opt-in via `VulkanBackend::wide_load` (`ORANGU_WIDE_LOAD=1`).
const PRELUDE_VEC4: &str = r#"
struct Meta {
    in_dim: u32,
    out_dim: u32,
    n_tokens: u32,
    row_bytes: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> params: Meta;

fn f16_to_f32(bits: u32) -> f32 {
    return unpack2x16float(bits & 0xFFFFu).x;
}

// bfloat16 -> f32: the top 16 bits of an f32, left-shifted into place —
// mirrors `quant::dequantize`'s `GGML_TYPE_BF16` arm exactly.
fn bf16_to_f32(bits: u32) -> f32 {
    return bitcast<f32>((bits & 0xFFFFu) << 16u);
}

// WGSL supports a dynamic (non-const) index into a vector via `v[i]`, but
// this sticks to an explicit branch anyway — `idx` only ever ranges 0..4
// (or 0..3 for `vec3_word`), so the branch is cheap, and a dynamic vector
// index is avoided.
fn vec4_word(v: vec4<u32>, idx: u32) -> u32 {
    if (idx == 0u) { return v.x; }
    if (idx == 1u) { return v.y; }
    if (idx == 2u) { return v.z; }
    return v.w;
}

fn vec3_word(v: vec3<u32>, idx: u32) -> u32 {
    if (idx == 0u) { return v.x; }
    if (idx == 1u) { return v.y; }
    return v.z;
}

// Reads the little-endian u32 word starting at `byte_offset`, which must
// itself be a multiple of 4 — the caller's responsibility, same as
// `read_u8`'s own "byte_offset in range" contract in `PRELUDE`. This is
// correct for *any* word-aligned offset, whether or not the enclosing
// block itself starts at a 16-byte (`vec4`) boundary — `byte_offset / 16u`
// and `(byte_offset % 16u) / 4u` are well-defined for any non-negative
// `byte_offset`, not just ones a caller has separately proven are
// block-vec4-aligned.
fn read_word_v4(byte_offset: u32) -> u32 {
    return vec4_word(weights[byte_offset / 16u], (byte_offset % 16u) / 4u);
}

// The vec4-bound drop-in equivalent of `PRELUDE`'s `read_u8` — correct for
// *any* `byte_offset`, aligned or not.
fn read_byte_v4(byte_offset: u32) -> u32 {
    let word = read_word_v4(byte_offset - (byte_offset % 4u));
    return (word >> (8u * (byte_offset % 4u))) & 0xFFu;
}

// Reads the little-endian u32 starting at an *arbitrary* (not necessarily
// 4-byte-aligned) `byte_offset` — the standard "unaligned load via two
// aligned loads + shift" trick, needed for fields (`Q5_0`'s 4-byte `qh`)
// whose own start alignment isn't fixed the way `Q4_K`/`Q5_K`'s 16-byte
// block size makes their header alignment fixed.
fn read_word_unaligned_v4(byte_offset: u32) -> u32 {
    let shift = (byte_offset % 4u) * 8u;
    let aligned = byte_offset - (byte_offset % 4u);
    if (shift == 0u) {
        return read_word_v4(aligned);
    }
    let lo = read_word_v4(aligned);
    let hi = read_word_v4(aligned + 4u);
    return (lo >> shift) | (hi << (32u - shift));
}

// ggml's `get_scale_min_k4`, sourcing scale bytes from an already-loaded
// `vec3<u32>` (`Q4_K`/`Q5_K`'s header vec4's `.yzw`) instead of re-reading
// them from `array<u32>` via `read_u8` each time — mirrors `PRELUDE`'s own
// `get_scale_min_k4` exactly (see that function's doc comment /
// `quant::get_scale_min_k4`).
fn get_scale_min_k4_v4(scales: vec3<u32>, j: u32) -> vec2<u32> {
    if (j < 4u) {
        let qj = (vec3_word(scales, j / 4u) >> (8u * (j % 4u))) & 0xFFu;
        let qj4 = (vec3_word(scales, (j + 4u) / 4u) >> (8u * ((j + 4u) % 4u))) & 0xFFu;
        return vec2<u32>(qj & 63u, qj4 & 63u);
    }
    let qj = (vec3_word(scales, j / 4u) >> (8u * (j % 4u))) & 0xFFu;
    let qj4 = (vec3_word(scales, (j + 4u) / 4u) >> (8u * ((j + 4u) % 4u))) & 0xFFu;
    let qjm4 = (vec3_word(scales, (j - 4u) / 4u) >> (8u * ((j - 4u) % 4u))) & 0xFFu;
    let sc = (qj4 & 0xFu) | ((qjm4 >> 6u) << 4u);
    let m = (qj4 >> 4u) | ((qj >> 6u) << 4u);
    return vec2<u32>(sc, m);
}
"#;

/// `{ f32 }`, 1 element — the whole block *is* one word, so `dequant_
/// element` collapses to a single `read_word_v4` call instead of the
/// byte-wise kernel's 4 separate `read_u8` calls.
const F32_WIDE_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 4u;
const BLOCK_ELEMS: u32 = 1u;
fn dequant_element(byte_offset: u32, k: u32) -> f32 {
    return bitcast<f32>(read_word_v4(byte_offset));
}
"#;

/// `{ f16 }`, 1 element. `byte_offset` is always even (2-byte blocks) but
/// not necessarily 4-aligned, so this reads the containing word and
/// selects the low or high half — 1 word read replacing 2 byte reads.
const F16_WIDE_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 2u;
const BLOCK_ELEMS: u32 = 1u;
fn dequant_element(byte_offset: u32, k: u32) -> f32 {
    let word = read_word_v4(byte_offset - (byte_offset % 4u));
    let half = select(word & 0xFFFFu, word >> 16u, (byte_offset % 4u) != 0u);
    return f16_to_f32(half);
}
"#;

/// `{ bf16 }`, 1 element — same word-halving as `F16_WIDE_MIDDLE`.
const BF16_WIDE_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 2u;
const BLOCK_ELEMS: u32 = 1u;
fn dequant_element(byte_offset: u32, k: u32) -> f32 {
    let word = read_word_v4(byte_offset - (byte_offset % 4u));
    let half = select(word & 0xFFFFu, word >> 16u, (byte_offset % 4u) != 0u);
    return bf16_to_f32(half);
}
"#;

/// `block_q4_0`: `d` (2 bytes, byte offset 0 relative to the block) never
/// straddles a 4-byte word regardless of the block's own alignment (a
/// 2-byte field starting at word-offset 0 or 2 always fits inside one
/// word) — 1 word read replaces the byte-wise kernel's 2 `read_u8` calls
/// for `d`. `qs` (the actual nibbles) stays per-byte (`read_byte_v4`, just
/// `vec4`-typed rather than further consolidated) — mirrors
/// `Q4_0_COOP_MIDDLE`'s math exactly otherwise.
const Q4_0_WIDE_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 18u;
const BLOCK_ELEMS: u32 = 32u;
fn dequant_element(byte_offset: u32, k: u32) -> f32 {
    let dword = read_word_v4(byte_offset - (byte_offset % 4u));
    let d = f16_to_f32(select(dword & 0xFFFFu, dword >> 16u, (byte_offset % 4u) != 0u));
    if (k < 16u) {
        let byte = read_byte_v4(byte_offset + 2u + k);
        return f32(i32(byte & 0xFu) - 8) * d;
    }
    let byte = read_byte_v4(byte_offset + 2u + (k - 16u));
    return f32(i32(byte >> 4u) - 8) * d;
}
"#;

/// `block_q5_0`: `d` consolidated the same way as `Q4_0_WIDE_MIDDLE`;
/// `qh` (4 bytes, byte offset 2 relative to the block) *can* straddle a
/// word boundary depending on the block's own alignment, so it goes
/// through `read_word_unaligned_v4` instead — 1 logical read (backed by up
/// to 2 aligned word reads) replaces the byte-wise kernel's 4 `read_u8`
/// calls. `qs` stays per-byte, same as `Q4_0_WIDE_MIDDLE` — mirrors
/// `Q5_0_COOP_MIDDLE`'s math exactly otherwise.
const Q5_0_WIDE_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 22u;
const BLOCK_ELEMS: u32 = 32u;
fn dequant_element(byte_offset: u32, k: u32) -> f32 {
    let dword = read_word_v4(byte_offset - (byte_offset % 4u));
    let d = f16_to_f32(select(dword & 0xFFFFu, dword >> 16u, (byte_offset % 4u) != 0u));
    let qh = read_word_unaligned_v4(byte_offset + 2u);
    if (k < 16u) {
        let byte = read_byte_v4(byte_offset + 6u + k);
        let xh_0 = ((qh >> k) << 4u) & 0x10u;
        return f32(i32((byte & 0xFu) | xh_0) - 16) * d;
    }
    let j = k - 16u;
    let byte = read_byte_v4(byte_offset + 6u + j);
    let xh_1 = (qh >> (j + 12u)) & 0x10u;
    return f32(i32((byte >> 4u) | xh_1) - 16) * d;
}
"#;

/// `block_q8_0`: `d` consolidated the same way as `Q4_0_WIDE_MIDDLE`;
/// already trivially per-element otherwise, mirrors `Q8_0_COOP_MIDDLE`.
const Q8_0_WIDE_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 34u;
const BLOCK_ELEMS: u32 = 32u;
fn dequant_element(byte_offset: u32, k: u32) -> f32 {
    let dword = read_word_v4(byte_offset - (byte_offset % 4u));
    let d = f16_to_f32(select(dword & 0xFFFFu, dword >> 16u, (byte_offset % 4u) != 0u));
    let byte = read_byte_v4(byte_offset + 2u + k);
    var v: i32 = i32(byte);
    if (v >= 128) {
        v = v - 256;
    }
    return f32(v) * d;
}
"#;

/// `block_q4_K`: `144`-byte blocks are an exact multiple of 16, and so is
/// `row_bytes` (`BLOCK_BYTES` times an integer count of blocks per row) —
/// every block this kernel ever indexes starts at a 16-byte boundary, so
/// `vec4_base = byte_offset / 16u` is always exact (no truncated
/// remainder). The whole `d`/`dmin`/`scales` header (bytes 0..16) then
/// loads in **one** `vec4` read (`weights[vec4_base]`) instead of the
/// byte-wise kernel's up to 9 separate `read_u8` calls (2 for `d`, 2 for
/// `dmin`, up to 5 across both `get_scale_min_k4` calls one element
/// needs). `qs` (the 128-byte
/// nibble region) stays one word-extraction per queried byte (`qs_byte`) —
/// same granularity `read_u8` already had there, just `vec4`-typed.
/// Otherwise mirrors `Q4_K_COOP_MIDDLE`'s index math line-for-line.
const Q4_K_WIDE_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 144u;
const BLOCK_ELEMS: u32 = 256u;

fn qs_byte_q4k(vec4_base: u32, qi: u32) -> u32 {
    let v4i = vec4_base + 1u + qi / 16u;
    let word = vec4_word(weights[v4i], (qi % 16u) / 4u);
    return (word >> (8u * (qi % 4u))) & 0xFFu;
}

fn dequant_element(byte_offset: u32, k: u32) -> f32 {
    let vec4_base = byte_offset / 16u;
    let header = weights[vec4_base];
    let d = f16_to_f32(header.x & 0xFFFFu);
    let dmin = f16_to_f32(header.x >> 16u);
    let scales = vec3<u32>(header.y, header.z, header.w);
    let q_offset = (k / 64u) * 64u;
    let local_in_group = k % 64u;
    let is_base = (q_offset / 64u) * 2u;
    let qi_base = q_offset / 2u;
    if (local_in_group < 32u) {
        let byte = qs_byte_q4k(vec4_base, qi_base + local_in_group);
        let sm = get_scale_min_k4_v4(scales, is_base);
        let d1 = d * f32(sm.x);
        let m1 = dmin * f32(sm.y);
        return d1 * f32(byte & 0xFu) - m1;
    }
    let l = local_in_group - 32u;
    let byte = qs_byte_q4k(vec4_base, qi_base + l);
    let sm = get_scale_min_k4_v4(scales, is_base + 1u);
    let d2 = d * f32(sm.x);
    let m2 = dmin * f32(sm.y);
    return d2 * f32(byte >> 4u) - m2;
}
"#;

/// `block_q5_K`: `176`-byte blocks are also an exact multiple of 16
/// (`176 / 16 == 11`), so this gets the same whole-header-in-one-`vec4`
/// treatment `Q4_K_WIDE_MIDDLE` does, plus `qh` (32 bytes, immediately
/// after the header — 2 more whole `vec4`s) read the same
/// word-extraction way `qs` already is. Otherwise mirrors
/// `Q5_K_COOP_MIDDLE`'s index math line-for-line.
const Q5_K_WIDE_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 176u;
const BLOCK_ELEMS: u32 = 256u;

fn qh_byte_q5k(vec4_base: u32, l: u32) -> u32 {
    let v4i = vec4_base + 1u + l / 16u;
    let word = vec4_word(weights[v4i], (l % 16u) / 4u);
    return (word >> (8u * (l % 4u))) & 0xFFu;
}

fn qs_byte_q5k(vec4_base: u32, qi: u32) -> u32 {
    let v4i = vec4_base + 3u + qi / 16u;
    let word = vec4_word(weights[v4i], (qi % 16u) / 4u);
    return (word >> (8u * (qi % 4u))) & 0xFFu;
}

fn dequant_element(byte_offset: u32, k: u32) -> f32 {
    let vec4_base = byte_offset / 16u;
    let header = weights[vec4_base];
    let d = f16_to_f32(header.x & 0xFFFFu);
    let dmin = f16_to_f32(header.x >> 16u);
    let scales = vec3<u32>(header.y, header.z, header.w);
    let q_offset = (k / 64u) * 64u;
    let idx = q_offset / 64u;
    let local_in_group = k % 64u;
    let is_base = idx * 2u;
    let ql_offset = idx * 32u;
    let u1 = 1u << (2u * idx);
    let u2 = 2u << (2u * idx);
    if (local_in_group < 32u) {
        let l = local_in_group;
        let byte = qs_byte_q5k(vec4_base, ql_offset + l);
        let qhbyte = qh_byte_q5k(vec4_base, l);
        var hi_bit: i32 = 0;
        if ((qhbyte & u1) != 0u) {
            hi_bit = 16;
        }
        let sm = get_scale_min_k4_v4(scales, is_base);
        let d1 = d * f32(sm.x);
        let m1 = dmin * f32(sm.y);
        return d1 * f32(i32(byte & 0xFu) + hi_bit) - m1;
    }
    let l = local_in_group - 32u;
    let byte = qs_byte_q5k(vec4_base, ql_offset + l);
    let qhbyte = qh_byte_q5k(vec4_base, l);
    var hi_bit: i32 = 0;
    if ((qhbyte & u2) != 0u) {
        hi_bit = 16;
    }
    let sm = get_scale_min_k4_v4(scales, is_base + 1u);
    let d2 = d * f32(sm.x);
    let m2 = dmin * f32(sm.y);
    return d2 * f32(i32(byte >> 4u) + hi_bit) - m2;
}
"#;

/// `block_q6_K`: `210`-byte blocks are *not* a multiple of 16 (`210 / 16
/// == 13.125`), so — unlike `Q4_K`/`Q5_K` — block starts land at
/// unpredictable, per-block-varying alignment, and the whole-header-in-
/// one-`vec4` trick doesn't apply cleanly. Only `d` (2 bytes, at relative
/// offset 208 — always word-safe by the same reasoning `Q4_0_WIDE_MIDDLE`
/// uses for its own `d`) is consolidated here; `ql`/`qh`/`scales` stay
/// per-byte (`read_byte_v4`). Otherwise mirrors `Q6_K_COOP_MIDDLE`'s index
/// math line-for-line.
const Q6_K_WIDE_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 210u;
const BLOCK_ELEMS: u32 = 256u;
fn dequant_element(byte_offset: u32, k: u32) -> f32 {
    let ql_off = byte_offset;
    let qh_off = byte_offset + 128u;
    let sc_off = byte_offset + 192u;
    let d_offset = byte_offset + 208u;
    let dword = read_word_v4(d_offset - (d_offset % 4u));
    let d = f16_to_f32(select(dword & 0xFFFFu, dword >> 16u, (d_offset % 4u) != 0u));
    let y_off = (k / 128u) * 128u;
    let idx = y_off / 128u;
    let local_in_group = k % 128u;
    let which_q = local_in_group / 32u;
    let l = local_in_group % 32u;
    let ql_o = idx * 64u;
    let qh_o = idx * 32u;
    let sc_o = idx * 8u;
    let is = l / 16u;
    let ql_l = read_byte_v4(ql_off + ql_o + l);
    let ql_l32 = read_byte_v4(ql_off + ql_o + l + 32u);
    let qh_l = read_byte_v4(qh_off + qh_o + l);
    var q: i32;
    var sc_idx: u32;
    if (which_q == 0u) {
        q = i32((ql_l & 0xFu) | ((qh_l & 3u) << 4u)) - 32;
        sc_idx = is;
    } else if (which_q == 1u) {
        q = i32((ql_l32 & 0xFu) | (((qh_l >> 2u) & 3u) << 4u)) - 32;
        sc_idx = is + 2u;
    } else if (which_q == 2u) {
        q = i32((ql_l >> 4u) | (((qh_l >> 4u) & 3u) << 4u)) - 32;
        sc_idx = is + 4u;
    } else {
        q = i32((ql_l32 >> 4u) | (((qh_l >> 6u) & 3u) << 4u)) - 32;
        sc_idx = is + 6u;
    }
    var sc: i32 = i32(read_byte_v4(sc_off + sc_o + sc_idx));
    if (sc >= 128) {
        sc = sc - 256;
    }
    return d * f32(sc) * f32(q);
}
"#;

/// The complete, compile-ready WGSL source for `ggml_type`'s wide-load
/// reduce pipeline, or `None` if this
/// backend has no wide-load kernel for it — same type coverage as
/// [`shader_source_reduce`]. Reuses `MAIN_REDUCE_SUFFIX` verbatim (see
/// `PRELUDE_VEC4`'s own doc comment for why every `*_WIDE_MIDDLE`'s
/// `dequant_element` keeps the same signature that requires).
pub fn shader_source_reduce_wide_load(
    ggml_type: u32,
    n_rows: usize,
    subgroup: bool,
) -> Option<String> {
    let middle = match ggml_type {
        t if t == GGML_TYPE_F32 => F32_WIDE_MIDDLE,
        t if t == GGML_TYPE_F16 => F16_WIDE_MIDDLE,
        t if t == GGML_TYPE_BF16 => BF16_WIDE_MIDDLE,
        t if t == GGML_TYPE_Q4_0 => Q4_0_WIDE_MIDDLE,
        t if t == GGML_TYPE_Q5_0 => Q5_0_WIDE_MIDDLE,
        t if t == GGML_TYPE_Q8_0 => Q8_0_WIDE_MIDDLE,
        t if t == GGML_TYPE_Q4_K => Q4_K_WIDE_MIDDLE,
        t if t == GGML_TYPE_Q5_K => Q5_K_WIDE_MIDDLE,
        t if t == GGML_TYPE_Q6_K => Q6_K_WIDE_MIDDLE,
        _ => return None,
    };
    let suffix = main_reduce_suffix(n_rows, subgroup);
    Some(format!("{PRELUDE_VEC4}\n{middle}\n{suffix}"))
}

// `Q4_K`-only decode kernel that restructures the reduce inner loop for
// **memory-level parallelism** — issuing several independent memory loads
// before the dependent dequant-and-dot rather than one outstanding load
// per lane at a time. Builds on the
// wide-load path (`PRELUDE_VEC4`, `weights` bound as `array<vec4<u32>>`)
// but changes *how the loop is shaped*, which the `MAIN_REDUCE_SUFFIX`-
// based wide-load kernel (`shader_source_reduce_wide_load`) does not.
//
// The problem it targets: `MAIN_REDUCE_SUFFIX`'s inner loop reads **one**
// weight element per lane per iteration (`k += 64u`) and immediately
// consumes it in a dependent `dequant_element` + `fma` before looping —
// one outstanding memory request per lane at a time, which under-feeds the
// memory pipeline on a latency-bound DRAM stream. Wide loads reduce the
// *number* of transactions but do not add independent in-flight loads;
// this does.
//
// The restructuring, exploiting `Q4_K`'s fixed `256 = 4 × 64` super-block
// geometry: one workgroup still handles a `REDUCE_N_ROWS = 4`-row group
// (same dispatch shape as `MAIN_REDUCE_SUFFIX`, so
// `VulkanBackend::build_op_resources`' workgroup-count math is reused
// unchanged), but the loop now iterates whole 256-element super-blocks
// rather than striding single elements. Within each block, thread `local`
// (0..63) owns in-group position `local` of *all four* 64-groups, so the
// body issues its **four activation loads up front** (`x0..x3`, one per
// 64-group, reused across all four output rows) and, per row, `q4k_block_
// dot` loads that block's header **once** (not once per element, as the
// per-element `dequant_element` re-does) and issues its **four qs-byte
// loads together** before any dependent scale/min math. That is the
// memory-level parallelism: several independent loads outstanding per
// lane per block, and 4× less redundant header traffic. Explicit unrolling
// to whole blocks makes the header reuse unconditional, which the compiler
// could not hoist across the stride-64 element loop.
//
// Pure `f32` arithmetic, identical to the scalar/wide-load path
// element-for-element (just reordered loads), so it cross-checks
// bit-for-bit against `CpuBackend` at the same tight tolerance the
// wide-load kernel uses — no `f16` precision loss to widen for. **On by
// default** (`VulkanBackend::wide_unroll`, opt out with
// `ORANGU_NO_MLP_UNROLL=1`).
// The shared `main` for every block-unroll kernel (`Q4_K`/`Q5_K`/`Q6_K`,
// scalar and packed-`f16`). Each type's `*_UNROLL_MIDDLE` supplies its own
// `BLOCK_BYTES`/`BLOCK_ELEMS` and a single uniform entry point
// `block_dot(byte_offset, local, x0, x1, x2, x3) -> f32` — this thread's
// contribution to one output row from one 256-element super-block, given
// the block's byte offset, this lane's id, and the four activations for
// the four 64-groups (positions `local`, `64+local`, `128+local`,
// `192+local`). Because all three types share that 4×64 super-block
// geometry (element `g` of this lane always lives at position `g*64 +
// local`), the activation gather and the whole `REDUCE_N_ROWS = 4`-batched
// loop/reduction are identical across types; only the per-type
// dequant-and-dot inside `block_dot` differs. Kept `REDUCE_N_ROWS`-batched
// (four output rows per workgroup, four hoisted activations reused across
// them) so `VulkanBackend::build_op_resources`' existing dispatch-count
// math applies unchanged.

/// `Q4_K`'s `block_dot`: header loaded once, all four qs-byte loads (one per
/// 64-group) issued up front so they're in flight together, then the four
/// dependent dequant-and-multiply-adds. This lane owns in-group position
/// `local` of every 64-group — positions 0..31 are low nibbles (qs byte
/// `local`, scale `g*2`), 32..63 high nibbles (qs byte `local-32`, scale
/// `g*2+1`). `q4k_elem` mirrors `Q4_K_WIDE_MIDDLE::dequant_element`.
const Q4K_UNROLL_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 144u;
const BLOCK_ELEMS: u32 = 256u;

fn qs_byte_q4k(vec4_base: u32, qi: u32) -> u32 {
    let v4i = vec4_base + 1u + qi / 16u;
    let word = vec4_word(weights[v4i], (qi % 16u) / 4u);
    return (word >> (8u * (qi % 4u))) & 0xFFu;
}

fn q4k_elem(d: f32, dmin: f32, scales: vec3<u32>, g: u32, is_low: bool, byte: u32) -> f32 {
    let is_idx = g * 2u + select(1u, 0u, is_low);
    let sm = get_scale_min_k4_v4(scales, is_idx);
    let dd = d * f32(sm.x);
    let mm = dmin * f32(sm.y);
    let nib = select(byte >> 4u, byte & 0xFu, is_low);
    return dd * f32(nib) - mm;
}

fn block_dot(byte_offset: u32, local: u32, x0: f32, x1: f32, x2: f32, x3: f32) -> f32 {
    let is_low = local < 32u;
    let qsi = select(local - 32u, local, is_low);
    let vec4_base = byte_offset / 16u;
    let header = weights[vec4_base];
    let d = f16_to_f32(header.x & 0xFFFFu);
    let dmin = f16_to_f32(header.x >> 16u);
    let scales = vec3<u32>(header.y, header.z, header.w);
    let b0 = qs_byte_q4k(vec4_base, qsi);
    let b1 = qs_byte_q4k(vec4_base, 32u + qsi);
    let b2 = qs_byte_q4k(vec4_base, 64u + qsi);
    let b3 = qs_byte_q4k(vec4_base, 96u + qsi);
    return q4k_elem(d, dmin, scales, 0u, is_low, b0) * x0
         + q4k_elem(d, dmin, scales, 1u, is_low, b1) * x1
         + q4k_elem(d, dmin, scales, 2u, is_low, b2) * x2
         + q4k_elem(d, dmin, scales, 3u, is_low, b3) * x3;
}
"#;

/// `Q5_K`'s `block_dot`: same 4×64 geometry and vec4-aligned header as
/// `Q4_K`, plus the extra high bit each element gets from the block's `qh`
/// region. One `qh` byte (index `qsi`) is shared across all four 64-groups
/// — only the bit selected differs (`1<<2g` for the low nibble half,
/// `2<<2g` for the high) — so it loads once. Mirrors `Q5_K_WIDE_MIDDLE::
/// dequant_element`.
const Q5K_UNROLL_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 176u;
const BLOCK_ELEMS: u32 = 256u;

fn qh_byte_q5k(vec4_base: u32, l: u32) -> u32 {
    let v4i = vec4_base + 1u + l / 16u;
    let word = vec4_word(weights[v4i], (l % 16u) / 4u);
    return (word >> (8u * (l % 4u))) & 0xFFu;
}

fn qs_byte_q5k(vec4_base: u32, qi: u32) -> u32 {
    let v4i = vec4_base + 3u + qi / 16u;
    let word = vec4_word(weights[v4i], (qi % 16u) / 4u);
    return (word >> (8u * (qi % 4u))) & 0xFFu;
}

fn q5k_elem(d: f32, dmin: f32, scales: vec3<u32>, g: u32, is_low: bool, byte: u32, qh: u32) -> f32 {
    let is_idx = g * 2u + select(1u, 0u, is_low);
    let sm = get_scale_min_k4_v4(scales, is_idx);
    let dd = d * f32(sm.x);
    let mm = dmin * f32(sm.y);
    let bit = select(2u << (2u * g), 1u << (2u * g), is_low);
    var hi: i32 = 0;
    if ((qh & bit) != 0u) { hi = 16; }
    let nib = select(byte >> 4u, byte & 0xFu, is_low);
    return dd * f32(i32(nib) + hi) - mm;
}

fn block_dot(byte_offset: u32, local: u32, x0: f32, x1: f32, x2: f32, x3: f32) -> f32 {
    let is_low = local < 32u;
    let qsi = select(local - 32u, local, is_low);
    let vec4_base = byte_offset / 16u;
    let header = weights[vec4_base];
    let d = f16_to_f32(header.x & 0xFFFFu);
    let dmin = f16_to_f32(header.x >> 16u);
    let scales = vec3<u32>(header.y, header.z, header.w);
    let qh = qh_byte_q5k(vec4_base, qsi);
    let b0 = qs_byte_q5k(vec4_base, qsi);
    let b1 = qs_byte_q5k(vec4_base, 32u + qsi);
    let b2 = qs_byte_q5k(vec4_base, 64u + qsi);
    let b3 = qs_byte_q5k(vec4_base, 96u + qsi);
    return q5k_elem(d, dmin, scales, 0u, is_low, b0, qh) * x0
         + q5k_elem(d, dmin, scales, 1u, is_low, b1, qh) * x1
         + q5k_elem(d, dmin, scales, 2u, is_low, b2, qh) * x2
         + q5k_elem(d, dmin, scales, 3u, is_low, b3, qh) * x3;
}
"#;

/// `Q6_K`'s `block_dot`. `Q6_K`'s 210-byte block isn't 16-byte-aligned and
/// uses a 2×128 (not 4×64) internal geometry, so this maps this lane's four
/// positions (`local`, `64+local`, `128+local`, `192+local`) to `Q6_K_WIDE_
/// MIDDLE`'s `(idx, which_q, l)` scheme: `l = local % 32`, `w_lo = local /
/// 32` picks which of the two `which_q` pairs, `idx` (0/1) picks the 128-half.
/// The two positions sharing an `idx` share one `ql`/`qh` byte, so only two
/// `ql`+two `qh` loads are issued (hoisted), plus `d` once (`scales` stay
/// per-byte — `Q6_K` has no compact vec4 header to consolidate, so this
/// hoists loads rather than caching a header). Mirrors `Q6_K_WIDE_MIDDLE`.
const Q6K_UNROLL_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 210u;
const BLOCK_ELEMS: u32 = 256u;

// One Q6_K element: `ql` is this (idx,w_lo)'s pre-loaded low-or-high quant
// byte, `qh` its pre-loaded high-bit byte; `half` (0/1) selects the low or
// high `which_q` of the pair. `which_q = w_lo + 2*half`, matching
// `Q6_K_WIDE_MIDDLE`'s four branches.
fn q6k_elem(d: f32, sc_off: u32, idx: u32, w_lo: u32, half: u32, is: u32, ql: u32, qh: u32) -> f32 {
    let qh_shift = half * 4u + w_lo * 2u;
    let sc_idx = is + half * 4u + w_lo * 2u;
    let nib = select(ql >> 4u, ql & 0xFu, half == 0u);
    let q = i32(nib | (((qh >> qh_shift) & 3u) << 4u)) - 32;
    var sc: i32 = i32(read_byte_v4(sc_off + idx * 8u + sc_idx));
    if (sc >= 128) { sc = sc - 256; }
    return d * f32(sc) * f32(q);
}

fn block_dot(byte_offset: u32, local: u32, x0: f32, x1: f32, x2: f32, x3: f32) -> f32 {
    let ql_off = byte_offset;
    let qh_off = byte_offset + 128u;
    let sc_off = byte_offset + 192u;
    let d_offset = byte_offset + 208u;
    let dword = read_word_v4(d_offset - (d_offset % 4u));
    let d = f16_to_f32(select(dword & 0xFFFFu, dword >> 16u, (d_offset % 4u) != 0u));
    let l = local % 32u;
    let w_lo = local / 32u;
    let is = l / 16u;
    let qlA = read_byte_v4(ql_off + l + w_lo * 32u);
    let qhA = read_byte_v4(qh_off + l);
    let qlB = read_byte_v4(ql_off + 64u + l + w_lo * 32u);
    let qhB = read_byte_v4(qh_off + 32u + l);
    let e0 = q6k_elem(d, sc_off, 0u, w_lo, 0u, is, qlA, qhA);
    let e1 = q6k_elem(d, sc_off, 0u, w_lo, 1u, is, qlA, qhA);
    let e2 = q6k_elem(d, sc_off, 1u, w_lo, 0u, is, qlB, qhB);
    let e3 = q6k_elem(d, sc_off, 1u, w_lo, 1u, is, qlB, qhB);
    return e0 * x0 + e1 * x1 + e2 * x2 + e3 * x3;
}
"#;

pub fn shader_source_reduce_q4k_wide_unroll(n_rows: usize, subgroup: bool) -> String {
    let suffix = unroll_suffix(n_rows, subgroup);
    format!("{PRELUDE_VEC4}\n{Q4K_UNROLL_MIDDLE}\n{suffix}")
}

/// `Q4_K` decode kernel that reads every qs byte **once** and dequantizes
/// *both* its nibbles — the fix for `Q4K_UNROLL_MIDDLE`'s 2× redundant
/// weight streaming. The two-wave `block_dot` above splits a 64-thread
/// workgroup into two 32-lane halves that *each* load the whole 144-byte
/// block — one taking low nibbles (`is_low`), one high (`local - 32`) — so
/// every weight byte is fetched twice. Here one **32-thread** workgroup
/// owns a whole super-block: lane `local` (0..31) loads the four qs bytes at
/// in-group position `local` of the four 64-groups and, per group, emits
/// *both* the low-nibble element (position `g*64 + local`, activation
/// `xl_g`) and the high-nibble element (position `g*64 + 32 + local`,
/// activation `xh_g`), reusing the identical `q4k_elem`/`qs_byte_q4k` math.
/// One lane now sums a low+high pair the two halves previously summed
/// separately, so the float add order differs — not bit-identical, but
/// within the same cross-check tolerance the existing kernel variants
/// already have vs. each other and `CpuBackend`. A 32-thread workgroup fits
/// in a single subgroup, so the reduction is a barrier-free `subgroupAdd`
/// when `subgroup` is set (else a 32-wide barrier tree). `n_rows` output
/// rows share the workgroup and its hoisted activations, exactly as
/// `unroll_suffix` does.
/// The **MMVQ** (integer-dot quantized) `Q4_K` decode matmul-vec — a WGSL port
/// of llama.cpp's `mul_mat_vecq.comp`, the path llama actually runs for gemma
/// decode on this GPU. Unlike every prior orangu Q4_K kernel (all
/// floating-point), this does the dot in **integers**: the activation row is
/// pre-quantized to `q8` (int8 + per-32-block `f32` scale and quant-sum, binding
/// 1, layout `[d, sumq, qs0..qs7]` = 10 `u32`/block), the 4-bit weights are read
/// as int8 nibbles, and the products go through `dot4I8Packed` (SPIR-V `OpSDot`,
/// HW `v_dot4_i32_i8`): four int8×int8 into one `int32` per instruction,
/// rescaled to `f32` at the end. This attacks both prior dead ends at once —
/// 4 MACs/instruction (ALU) and int8 operands / int32 accumulator (register
/// pressure). Correct because ggml's Q4_K dequant is
/// **contiguous per sub-block** (`y[32g..32g+32]` all share scale/min `g`), so a
/// natural 32-element activation block aligns with one Q4_K sub-block, letting
/// the per-sub-block integer dot factor the scale out. Each of 32 lanes strides
/// over the row's `in_dim/32` sub-blocks; per sub-block it forms
/// `d·scale·d_b·Σ(nib·q) − dmin·min·d_b·Σq`, then `subgroupAdd` (or a 32-wide
/// tree) reduces to the output element. One output row per workgroup
/// (`NUM_ROWS = 1`, llama's `rm_kq_int`).
///
/// Selected for `Q4_K` decode when `ORANGU_Q4K_MMVQ=1`. Not bit-identical (q8
/// activation quantization is lossy — like llama), cross-checks within the same
/// tolerance as the dual kernel.
/// GPU activation-quantiser for the MMVQ path: reads an `f32` activation row
/// (binding 0) and writes the q8 block layout `shader_source_reduce_q4k_mmvq`
/// consumes (binding 1) — per 32-element block, 10 `u32`: `[d, sumq, qs0..qs7]`
/// (`d = max|x|/127` the scale, `sumq = Σq`, 32 int8 quants packed 4/word). The
/// GPU equivalent of `quantize_activation_q8`. **One thread per 32-block**
/// (`n_blocks = len/32`), so no cross-thread reduction: each thread scans its
/// block for the max, then quantizes/packs/sums it. `meta` (binding 2) carries
/// `len` (the activation length) in field 0 — same `ElemMeta` shape the norm/
/// elementwise kernels use, so it reuses `elem3_bind_group_layout`.
pub fn shader_source_quantize_q8() -> String {
    r#"
struct ElemMeta {
    len: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

@group(0) @binding(0) var<storage, read> xin: array<f32>;
@group(0) @binding(1) var<storage, read_write> qout: array<u32>;
@group(0) @binding(2) var<uniform> qmeta: ElemMeta;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let blk = gid.x;
    let n_blocks = qmeta.len / 32u;
    if (blk >= n_blocks) {
        return;
    }
    let base = blk * 32u;

    var amax: f32 = 0.0;
    var i: u32 = 0u;
    loop {
        if (i >= 32u) { break; }
        amax = max(amax, abs(xin[base + i]));
        i = i + 1u;
    }
    let d = amax / 127.0;
    let id = select(0.0, 1.0 / d, d > 0.0);

    var sumq: i32 = 0;
    let out_base = blk * 10u;
    var w: u32 = 0u;
    loop {
        if (w >= 8u) { break; }
        var word: u32 = 0u;
        var k: u32 = 0u;
        loop {
            if (k >= 4u) { break; }
            let v = xin[base + w * 4u + k];
            var q: i32 = i32(round(v * id));
            q = clamp(q, -127, 127);
            sumq = sumq + q;
            word = word | ((u32(q) & 0xFFu) << (8u * k));
            k = k + 1u;
        }
        qout[out_base + 2u + w] = word;
        w = w + 1u;
    }
    qout[out_base] = bitcast<u32>(d);
    qout[out_base + 1u] = bitcast<u32>(sumq);
}
"#
    .to_string()
}

/// Rows of a prefill activation `[n_tokens, in_dim]`, quantized to 8 bits
/// per 32-element block for [`shader_source_mmq_q4k`]. The layout is
/// per token, per 256-element super-block — 96 `u32`, so every super-block
/// starts on a 16-byte boundary and the GEMM tiles it with `vec4` loads:
///
/// - words 0..32: the eight sub-blocks' `(d, sumq_lo, sumq_hi, 0)` — `d =
///   max|x|/127` as `f32` bits, the sums of the quants over the block's
///   first and second sixteen as `i32` bits (`Q4_K` scales per 32 and adds
///   them; `Q6_K` scales per 16 and needs them apart) — sub-block `b` at `4b`;
/// - words 32..96: the quants, sub-block `b` at `32 + 8b`, four int8 a word,
///   little-endian, exactly as `shader_source_quantize_q8` packs them.
///
/// One thread per 32-block (`ElemMeta.len` = `n_tokens * in_dim`, `aux` =
/// `in_dim`). The same rounding as the decode quantizer, so the two paths
/// quantize an activation identically.
pub fn shader_source_quantize_q8_rows() -> String {
    r#"
struct ElemMeta {
    len: u32,
    aux: u32,
    _p1: u32,
    _p2: u32,
}

@group(0) @binding(0) var<storage, read> xin: array<f32>;
@group(0) @binding(1) var<storage, read_write> qout: array<u32>;
@group(0) @binding(2) var<uniform> qmeta: ElemMeta;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let blk = gid.x;
    let n_blocks = qmeta.len / 32u;
    if (blk >= n_blocks) {
        return;
    }
    let in_dim = qmeta.aux;
    let per_row = in_dim / 32u;
    let t = blk / per_row;
    let sb = blk % per_row;
    let super_blk = sb / 8u;
    let b = sb % 8u;
    let base = blk * 32u;

    var amax: f32 = 0.0;
    var i: u32 = 0u;
    loop {
        if (i >= 32u) { break; }
        amax = max(amax, abs(xin[base + i]));
        i = i + 1u;
    }
    let d = amax / 127.0;
    let id = select(0.0, 1.0 / d, d > 0.0);

    var sumq_lo: i32 = 0;
    var sumq_hi: i32 = 0;
    // Super-blocks per row rounded **up**: a row that is not a whole number
    // of them (a 704-wide expert down projection) still has a slot for its
    // last, partial one, and the kernels read only the sub-blocks it has.
    let out_base = (t * ((in_dim + 255u) / 256u) + super_blk) * 96u;
    var w: u32 = 0u;
    loop {
        if (w >= 8u) { break; }
        var word: u32 = 0u;
        var k: u32 = 0u;
        loop {
            if (k >= 4u) { break; }
            let v = xin[base + w * 4u + k];
            var q: i32 = i32(round(v * id));
            q = clamp(q, -127, 127);
            if (w < 4u) { sumq_lo = sumq_lo + q; } else { sumq_hi = sumq_hi + q; }
            word = word | ((u32(q) & 0xFFu) << (8u * k));
            k = k + 1u;
        }
        qout[out_base + 32u + b * 8u + w] = word;
        w = w + 1u;
    }
    qout[out_base + 4u * b] = bitcast<u32>(d);
    qout[out_base + 4u * b + 1u] = bitcast<u32>(sumq_lo);
    qout[out_base + 4u * b + 2u] = bitcast<u32>(sumq_hi);
    qout[out_base + 4u * b + 3u] = 0u;
}
"#
    .to_string()
}

/// Tile geometry of the integer-dot GEMMs: rows per workgroup, and each
/// thread's share of them; the token width is per kernel
/// ([`mmq_tile_tokens`]) — 16 threads across the tokens, each holding
/// `tokens / 16`.
pub const MMQ_TILE_ROWS: u32 = 64;
const MMQ_THREAD_ROWS: u32 = 8;
/// The `Q4_K` kernel's token tile: 32, so two workgroups fit a compute
/// unit's shared memory (measured 2.6 → 3.4 TFLOP/s-equivalent from 64);
/// the `Q6_K` kernel keeps 64, its unpacking staging being per step and
/// worth amortizing over more tokens (64 → 32 measured 1.31 → 1.27).
pub const MMQ_Q4K_TILE_TOKENS: u32 = 32;
pub const MMQ_Q6K_TILE_TOKENS: u32 = 64;

/// The widest token tile any integer-dot kernel stages — what the q8
/// activation buffer is padded to.
pub const MMQ_MAX_TILE_TOKENS: u32 = MMQ_WIDE_TILE_TOKENS;

/// The integer-dot `Q4_K` GEMM for a prefill batch — the tiled form of
/// [`shader_source_reduce_q4k_mmvq`]'s arithmetic, over activations
/// [`shader_source_quantize_q8_rows`] laid out. The float tiled kernel
/// dequantizes every weight to `f32` and does one multiply-add per element;
/// this one keeps the 4-bit weights as bytes and the activations as int8,
/// and `dot4I8Packed` does four multiply-adds per instruction on both, with
/// each sub-block's scales applied to the integer sum afterwards — the same
/// factoring the decode kernel relies on: ggml's `Q4_K` dequant is
/// contiguous per 32-element sub-block, so a sub-block's 32 products share
/// one `d·sc` and one `dmin·m`, and the min term is `dmin·m·Σq` with `Σq`
/// precomputed per activation block.
///
/// One workgroup per `MMQ_TILE_ROWS × MMQ_Q4K_TILE_TOKENS` output tile, 128
/// threads in an 8 × 16 grid, each holding an 8 × 2 micro-tile of `f32`
/// accumulators as named scalars (an `array` local would be scratch memory,
/// not registers). Per 256-element super-block the workgroup stages the
/// tile's rows (9 `vec4` each: the header and 32 words of nibbles) and
/// tokens (20 `vec4` each) in shared memory and unpacks every row's eight
/// `(scale, min)` pairs once into a shared table. Each thread then runs the
/// eight sub-blocks with its four tokens' quants held in registers and one
/// row's nibbles read at a time — two `vec4` per row against the resident
/// eight, so shared memory is read once per 32 dots rather than once per 4.
/// The first version of this kernel read both operands per pair and ran no
/// faster than the float one; the instruction census put three
/// address-and-load instructions on every dot.
///
/// The rows and tokens a thread owns are **interleaved** across the tile
/// (thread `tr` holds rows `tr, tr + 8, …`, thread `tt` tokens `tt, tt + 16,
/// …`), and a token's staged super-block is padded from 24 to 25 `vec4`:
/// with each thread on a contiguous run, the sixteen lanes of a subgroup
/// read shared memory at strides that are multiples of 32 words and all
/// land in the same bank — a sixteen-way conflict on every activation read,
/// which is where the second version's time went. The interleaved strides
/// (36 words for rows, 100 for tokens) spread eight lanes over the banks,
/// the two-way remainder being what a 16-byte read costs anyway.
///
/// Shapes: `in_dim` a multiple of 256 and `out_dim` of `MMQ_TILE_ROWS`;
/// any `n_tokens` — the last token tile's rows past it are staged from the
/// q8 buffer's padding (sized to whole tiles, never written) and their
/// results are not stored. Bindings are the matmul layout's: weights (`vec4<u32>`),
/// the q8 activations (`vec4<u32>`), the `f32` output `[n_tokens, out_dim]`,
/// and the op's `Meta`.
///
/// Not bit-identical to the float path: the activation is quantized to 8
/// bits, as it is on every engine that runs this kind of kernel.
pub fn shader_source_mmq_q4k() -> String {
    let tr = MMQ_THREAD_ROWS as usize;
    let tt = (MMQ_Q4K_TILE_TOKENS / 16) as usize;
    let tile_tokens = MMQ_Q4K_TILE_TOKENS;
    let mut src = String::new();
    src.push_str(&format!(
        r#"
struct Meta {{
    in_dim: u32,
    out_dim: u32,
    n_tokens: u32,
    row_bytes: u32,
}}

@group(0) @binding(0) var<storage, read> weights: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> q8x: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> params: Meta;

fn f16_to_f32(bits: u32) -> f32 {{
    return unpack2x16float(bits & 0xFFFFu).x;
}}
fn vec3_word(v: vec3<u32>, i: u32) -> u32 {{
    if (i == 0u) {{ return v.x; }}
    if (i == 1u) {{ return v.y; }}
    return v.z;
}}
fn get_scale_min_k4_v4(scales: vec3<u32>, j: u32) -> vec2<u32> {{
    if (j < 4u) {{
        let qj = (vec3_word(scales, j / 4u) >> (8u * (j % 4u))) & 0xFFu;
        let qj4 = (vec3_word(scales, (j + 4u) / 4u) >> (8u * ((j + 4u) % 4u))) & 0xFFu;
        return vec2<u32>(qj & 63u, qj4 & 63u);
    }}
    let qj = (vec3_word(scales, j / 4u) >> (8u * (j % 4u))) & 0xFFu;
    let qj4 = (vec3_word(scales, (j + 4u) / 4u) >> (8u * ((j + 4u) % 4u))) & 0xFFu;
    let qjm4 = (vec3_word(scales, (j - 4u) / 4u) >> (8u * ((j - 4u) % 4u))) & 0xFFu;
    let sc = (qj4 & 0xFu) | ((qjm4 >> 6u) << 4u);
    let m = (qj4 >> 4u) | ((qj >> 6u) << 4u);
    return vec2<u32>(sc, m);
}}

const TILE_ROWS: u32 = {rows}u;
const TILE_TOKENS: u32 = {tokens}u;
// One super-block per row: the header vec4 and 32 words of nibbles.
var<workgroup> wt: array<vec4<u32>, {wt_len}>;
// One super-block per token: 8 (d, sumq_lo, sumq_hi, 0), then 64 words of
// quants, padded to 25 vec4 so the lanes' reads spread across the banks.
var<workgroup> xt: array<vec4<u32>, {xt_len}>;
// Each row's eight (d * scale, dmin * min), unpacked once per super-block,
// at a stride of 9 for the same reason.
var<workgroup> sm: array<vec2<f32>, {sm_len}>;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    let row0 = wid.x * TILE_ROWS;
    let tok0 = wid.y * TILE_TOKENS;
    let local = lid.x;
    let tr = local / 16u;
    let tt = local % 16u;
    let n_super = (params.in_dim + 255u) / 256u;
    let row_vec4s = params.row_bytes / 16u;
    let wbase = tr * 9u;
    let xbase = tt * 25u;
    let sbase = tr * 9u;
"#,
        rows = MMQ_TILE_ROWS,
        tokens = tile_tokens,
        wt_len = MMQ_TILE_ROWS * 9,
        xt_len = tile_tokens * 25,
        sm_len = MMQ_TILE_ROWS * 9,
    ));
    for r in 0..tr {
        for t in 0..tt {
            src.push_str(&format!("    var a{r}{t}: f32 = 0.0;\n"));
        }
    }
    src.push_str(&format!(
        r#"
    var s: u32 = 0u;
    loop {{
        if (s >= n_super) {{ break; }}
        // Stage the tile: rows, then tokens.
        var i: u32 = local;
        loop {{
            if (i >= {wt_len}u) {{ break; }}
            let r = i / 9u;
            let c = i % 9u;
            wt[i] = weights[(row0 + r) * row_vec4s + s * 9u + c];
            i = i + 128u;
        }}
        i = local;
        loop {{
            if (i >= {xt_loads}u) {{ break; }}
            let t = i / 24u;
            let c = i % 24u;
            xt[t * 25u + c] = q8x[((tok0 + t) * n_super + s) * 24u + c];
            i = i + 128u;
        }}
        workgroupBarrier();
        i = local;
        loop {{
            if (i >= {sm_loads}u) {{ break; }}
            let r = i / 8u;
            let b = i % 8u;
            let h = wt[r * 9u];
            let p = get_scale_min_k4_v4(vec3<u32>(h.y, h.z, h.w), b);
            sm[r * 9u + b] = vec2<f32>(
                f16_to_f32(h.x & 0xFFFFu) * f32(p.x),
                f16_to_f32(h.x >> 16u) * f32(p.y));
            i = i + 128u;
        }}
        workgroupBarrier();
        let m4 = vec4<u32>(0x0F0F0F0Fu);
        let f4 = vec4<u32>(4u);
"#,
        wt_len = MMQ_TILE_ROWS * 9,
        xt_loads = tile_tokens * 24,
        sm_loads = MMQ_TILE_ROWS * 8,
    ));
    // `ORANGU_MMQ_STUB=1`: the same loads and rescales with the eight dots
    // replaced by one xor — a probe of everything but the dot, so the
    // kernel's time splits into the arithmetic and the rest. The output is
    // meaningless under it.
    let stub = crate::engine::env::flag_on("ORANGU_MMQ_STUB");
    for b in 0..8usize {
        let half = b / 2;
        let high = b % 2 == 1;
        src.push_str(&format!(
            "        {{\n        // sub-block {b}: the four tokens' quants and scales, resident\n"
        ));
        for t in 0..tt {
            src.push_str(&format!(
                "        let x{t}a = xt[xbase + {}u];\n        let x{t}b = xt[xbase + {}u];\n        let ds{t} = xt[xbase + {}u];\n        let db{t} = bitcast<f32>(ds{t}.x);\n        let dq{t} = db{t} * f32(bitcast<i32>(ds{t}.y) + bitcast<i32>(ds{t}.z));\n",
                t * 16 * 25 + 8 + 2 * b,
                t * 16 * 25 + 9 + 2 * b,
                t * 16 * 25 + b,
            ));
        }
        for r in 0..tr {
            // Row `r` of this thread is tile row `r * 8 + tr`.
            let (wa, wb) = if high {
                (
                    format!("(wt[wbase + {}u] >> f4) & m4", r * 8 * 9 + 1 + 2 * half),
                    format!("(wt[wbase + {}u] >> f4) & m4", r * 8 * 9 + 2 + 2 * half),
                )
            } else {
                (
                    format!("wt[wbase + {}u] & m4", r * 8 * 9 + 1 + 2 * half),
                    format!("wt[wbase + {}u] & m4", r * 8 * 9 + 2 + 2 * half),
                )
            };
            src.push_str(&format!(
                "        {{\n        let wa = {wa};\n        let wb = {wb};\n        let p = sm[sbase + {}u];\n",
                r * 8 * 9 + b
            ));
            for t in 0..tt {
                let dot = if stub {
                    format!("i32((wa.x ^ x{t}a.x) & 7u)")
                } else {
                    format!(
                        "dot4I8Packed(wa.x, x{t}a.x) + dot4I8Packed(wa.y, x{t}a.y) + dot4I8Packed(wa.z, x{t}a.z) + dot4I8Packed(wa.w, x{t}a.w) + dot4I8Packed(wb.x, x{t}b.x) + dot4I8Packed(wb.y, x{t}b.y) + dot4I8Packed(wb.z, x{t}b.z) + dot4I8Packed(wb.w, x{t}b.w)"
                    )
                };
                src.push_str(&format!(
                    "        a{r}{t} = a{r}{t} + fma(p.x * db{t}, f32({dot}), -p.y * dq{t});\n"
                ));
            }
            src.push_str("        }\n");
        }
        src.push_str("        }\n");
    }
    src.push_str("        workgroupBarrier();\n        s = s + 1u;\n    }\n");
    for t in 0..tt {
        src.push_str(&format!(
            "    if (tok0 + {}u + tt < params.n_tokens) {{\n",
            t * 16
        ));
        for r in 0..tr {
            src.push_str(&format!(
                "        y[(tok0 + {}u + tt) * params.out_dim + row0 + {}u + tr] = a{r}{t};\n",
                t * 16,
                r * 8
            ));
        }
        src.push_str("    }\n");
    }
    src.push_str("}\n");
    src
}

/// The wide integer-dot `Q4_K` GEMM: the same arithmetic as
/// [`shader_source_mmq_q4k`] over a **128 × 128** output tile, staged
/// **32 elements deep** at a time.
///
/// The narrow kernel stages a whole 256-element super-block per step, which
/// bounds its output tile to 64 × 32 under the shared-memory limit: every
/// staged byte serves 2 tokens × 8 rows, and a thread carries 16
/// accumulators. Measured on the served model's FFN shape at 512 tokens it
/// reached 3.4 TFLOP/s-equivalent while the same shape can run at ~6 on
/// this class of card. This kernel slices the reduction instead — `KS`
/// sub-blocks of 32 elements per barrier — so the tile can grow to 128 rows
/// × 128 tokens with the same shared memory, and each thread holds a
/// 4-row × 32-token micro-tile: 128 accumulators as named scalars, each
/// staged row read once against 32 tokens and each token read once against
/// 4 rows. Per staged byte the arithmetic is 16× the narrow kernel's, and
/// the tile reads its operands from device memory 3.3× fewer times.
///
/// Thread layout: two subgroups of 64; subgroup `w` owns tokens
/// `64w..64w+64`, lane `l % 32` owns rows `l % 32 + 32c` (`c < 4`) and lane
/// `l / 32` the odd or even four of each eight tokens. Rows interleaved by
/// 32 rather than contiguous per thread so the 32 lanes of a half-subgroup
/// read shared memory at a stride of one row entry (7 words — coprime with
/// the bank count, so conflict-free) and store their outputs to 32
/// consecutive rows (coalesced). A token entry is read by a whole
/// half-subgroup at once — a broadcast — so its stride (11) only needs to
/// keep the two halves' addresses in different banks.
///
/// Shared memory per row and sub-block: the 32 nibbles packed two per
/// byte into 4 words (`u_{2q} | u_{2q+1} << 4`, unpacked back to 8 words
/// once per sub-block per thread, not per dot) and the row's `(d·sc,
/// dmin·m)` for that sub-block, computed at staging — 7 words. Per token
/// and sub-block: the 8 quant words, `d`, and `Σq` — 11 words. At `KS = 4`
/// that is 14 + 22 KiB.
///
/// Shapes: `in_dim` a multiple of 256, `out_dim` of 128, any `n_tokens`
/// (the q8 buffer is padded to whole 128-token tiles; padding rows are
/// staged and their results discarded). Same bindings and activation
/// layout as the narrow kernel, so the two are interchangeable per
/// dispatch — `mmq_pipeline_for` picks by shape, keeping the narrow one for
/// the projections too small to fill the device with 128 × 128 tiles.
pub const MMQ_WIDE_TILE_ROWS: u32 = 128;
/// The half-height wide tile, for a projection too narrow to fill the
/// device with 128-row tiles: 2 rows per thread, 64 accumulators.
pub const MMQ_MID_TILE_ROWS: u32 = 64;
pub const MMQ_WIDE_TILE_TOKENS: u32 = 128;
/// Sub-blocks of 32 elements staged per barrier, per kernel. Must divide 8
/// and, for `Q4_K`, be even (a nibble pair is two sub-blocks). Two rather
/// than four: the deeper slice halves the barriers but doubles the shared
/// memory (37 KiB against 18), and the workgroups that no longer fit beside
/// each other were what hid the weight loads' latency — measured over
/// eight rotating matrices at 512 tokens, 3.9 against 5.1 TFLOP/s-eq.
pub const MMQ_WIDE_KS: u32 = 2;
pub const MMQ_WIDE_KS_Q6K: u32 = 2;

pub fn shader_source_mmq_q4k_wide(rows: u32) -> String {
    shader_source_mmq_q4k_wide_toks(rows, MMQ_WIDE_TILE_TOKENS)
}

/// [`shader_source_mmq_q4k_wide`] at `toks` tokens per tile: 128 for a
/// dense prefill batch, 32 for the indexed expert GEMM, where a tile is one
/// expert's routed tokens — a few dozen — and a 128-token tile would be
/// three quarters padding.
pub fn shader_source_mmq_q4k_wide_toks(rows: u32, toks: u32) -> String {
    assert!(
        toks.is_multiple_of(16) && toks <= 128,
        "a tile is 16 tokens per lane pair per warp"
    );
    let tgroups = (toks / 16) as usize;
    // `ORANGU_MMQ_WIDE_KS`: the slice depth, for measuring it.
    let ks = std::env::var("ORANGU_MMQ_WIDE_KS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(MMQ_WIDE_KS);
    assert!(
        8 % ks == 0 && ks.is_multiple_of(2),
        "KS must be an even divisor of 8"
    );
    assert!(
        rows.is_multiple_of(32) && rows <= 128,
        "a wide tile is 32 rows per lane group"
    );
    let trows = (rows / 32) as usize;
    let mut src = String::new();
    src.push_str(&format!(
        r#"
struct Meta {{
    in_dim: u32,
    out_dim: u32,
    n_tokens: u32,
    row_bytes: u32,
}}

@group(0) @binding(0) var<storage, read> weights: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> q8x: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> params: Meta;

fn f16_to_f32(bits: u32) -> f32 {{
    return unpack2x16float(bits & 0xFFFFu).x;
}}
fn vec3_word(v: vec3<u32>, i: u32) -> u32 {{
    if (i == 0u) {{ return v.x; }}
    if (i == 1u) {{ return v.y; }}
    return v.z;
}}
fn get_scale_min_k4_v4(scales: vec3<u32>, j: u32) -> vec2<u32> {{
    if (j < 4u) {{
        let qj = (vec3_word(scales, j / 4u) >> (8u * (j % 4u))) & 0xFFu;
        let qj4 = (vec3_word(scales, (j + 4u) / 4u) >> (8u * ((j + 4u) % 4u))) & 0xFFu;
        return vec2<u32>(qj & 63u, qj4 & 63u);
    }}
    let qj = (vec3_word(scales, j / 4u) >> (8u * (j % 4u))) & 0xFFu;
    let qj4 = (vec3_word(scales, (j + 4u) / 4u) >> (8u * ((j + 4u) % 4u))) & 0xFFu;
    let qjm4 = (vec3_word(scales, (j - 4u) / 4u) >> (8u * ((j - 4u) % 4u))) & 0xFFu;
    let sc = (qj4 & 0xFu) | ((qjm4 >> 6u) << 4u);
    let m = (qj4 >> 4u) | ((qj >> 6u) << 4u);
    return vec2<u32>(sc, m);
}}

const ROWS: u32 = {rows}u;
const TOKS: u32 = {toks}u;
const KS: u32 = {ks}u;
// Per sub-block staged, per row: 4 packed quant words, d*sc, dmin*m, pad.
var<workgroup> wt: array<u32, {wt_len}>;
// Per sub-block staged, per token: 8 quant words, d, sum of quants, pad.
var<workgroup> xt: array<u32, {xt_len}>;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    let row0 = wid.x * ROWS;
    let tok0 = wid.y * TOKS;
    let tid = lid.x;
    let warp = tid / 64u;
    let lane = tid % 64u;
    let tiwr = lane % 32u;
    let tiwc = lane / 32u;
    let n_super = (params.in_dim + 255u) / 256u;
    let row_vec4s = params.row_bytes / 16u;
    let n_slices = n_super * (8u / KS);
    let m4 = vec4<u32>(0x0F0F0F0Fu);
    let f4 = vec4<u32>(4u);
"#,
        rows = rows,
        toks = toks,
        wt_len = rows * ks * 7,
        xt_len = toks * ks * 11,
    ));
    for cr in 0..trows {
        for t in 0..tgroups * 4 {
            src.push_str(&format!("    var a{cr}_{t}: f32 = 0.0;\n"));
        }
    }
    // The staging as straight-line code, every device load issued before
    // any is consumed: a `loop` over the items stages them one round trip
    // at a time, and on weights that are not already in the cache those
    // round trips are what the kernel waits on. Measured over eight
    // rotating matrices at 256 tokens, the rolled form ran at 75% of its
    // warm-cache rate.
    // Items per thread, and whether the last round is partial — the
    // half-height tile at two sub-blocks has 64 row items for 128 threads.
    let row_total = rows * ks / 2;
    let row_items = row_total.div_ceil(128) as usize;
    let row_partial = !row_total.is_multiple_of(128);
    let tok_total = toks * ks;
    let tok_items = tok_total.div_ceil(128) as usize;
    let tok_partial = !tok_total.is_multiple_of(128);
    src.push_str(
        r#"
    var slice: u32 = 0u;
    loop {
        if (slice >= n_slices) { break; }
        let sb = slice / (8u / KS);
        let k0 = (slice % (8u / KS)) * KS;
"#,
    );
    // `ORANGU_MMQ_STAGE_ROLLED=1`: the staging as loops, one item per
    // iteration — the control arm for measuring the straight-line form.
    if crate::engine::env::flag_on("ORANGU_MMQ_STAGE_ROLLED") {
        src.push_str(
            r#"        var i: u32 = tid;
        loop {
            if (i >= ROWS * (KS / 2u)) { break; }
            let r = i % ROWS;
            let pr = i / ROWS;
            let base = (row0 + r) * row_vec4s + sb * 9u;
            let h = weights[base];
            let jp = (k0 + 2u * pr) / 2u;
            let v0 = weights[base + 1u + 2u * jp];
            let v1 = weights[base + 2u + 2u * jp];
            let d = f16_to_f32(h.x & 0xFFFFu);
            let dm = f16_to_f32(h.x >> 16u);
            let lo0 = v0 & m4;
            let lo1 = v1 & m4;
            let hi0 = (v0 >> f4) & m4;
            let hi1 = (v1 >> f4) & m4;
            let e = ((2u * pr) * ROWS + r) * 7u;
            wt[e] = lo0.x | (lo0.y << 4u);
            wt[e + 1u] = lo0.z | (lo0.w << 4u);
            wt[e + 2u] = lo1.x | (lo1.y << 4u);
            wt[e + 3u] = lo1.z | (lo1.w << 4u);
            let s0 = get_scale_min_k4_v4(vec3<u32>(h.y, h.z, h.w), k0 + 2u * pr);
            wt[e + 4u] = bitcast<u32>(d * f32(s0.x));
            wt[e + 5u] = bitcast<u32>(dm * f32(s0.y));
            let e1 = e + ROWS * 7u;
            wt[e1] = hi0.x | (hi0.y << 4u);
            wt[e1 + 1u] = hi0.z | (hi0.w << 4u);
            wt[e1 + 2u] = hi1.x | (hi1.y << 4u);
            wt[e1 + 3u] = hi1.z | (hi1.w << 4u);
            let s1 = get_scale_min_k4_v4(vec3<u32>(h.y, h.z, h.w), k0 + 2u * pr + 1u);
            wt[e1 + 4u] = bitcast<u32>(d * f32(s1.x));
            wt[e1 + 5u] = bitcast<u32>(dm * f32(s1.y));
            i = i + 128u;
        }
        i = tid;
        loop {
            if (i >= TOKS * KS) { break; }
            let t = i % TOKS;
            let k = i / TOKS;
            let b = k0 + k;
            let tb = ((tok0 + t) * n_super + sb) * 24u;
            let hd = q8x[tb + b];
            let q0 = q8x[tb + 8u + 2u * b];
            let q1 = q8x[tb + 9u + 2u * b];
            let e = (k * TOKS + t) * 11u;
            xt[e] = q0.x;
            xt[e + 1u] = q0.y;
            xt[e + 2u] = q0.z;
            xt[e + 3u] = q0.w;
            xt[e + 4u] = q1.x;
            xt[e + 5u] = q1.y;
            xt[e + 6u] = q1.z;
            xt[e + 7u] = q1.w;
            xt[e + 8u] = hd.x;
            xt[e + 9u] = bitcast<u32>(bitcast<i32>(hd.y) + bitcast<i32>(hd.z));
            i = i + 128u;
        }
        workgroupBarrier();
        var k: u32 = 0u;
        loop {
            if (k >= KS) { break; }
"#,
        );
    } else {
        // Rows: one item per (row, nibble pair) — a pair of sub-blocks shares
        // 32 bytes of the super-block, low and high nibbles, read once and
        // written as two entries.
        for it in 0..row_items {
            // A thread past the last item loads the last item again (an
            // address that exists) and skips the write below.
            let index = if row_partial && it + 1 == row_items {
                format!("min(tid + {}u, ROWS * (KS / 2u) - 1u)", it * 128)
            } else {
                format!("tid + {}u", it * 128)
            };
            src.push_str(&format!(
                "        let ri{it} = {index};
        let rr{it} = ri{it} % ROWS;
        let rp{it} = ri{it} / ROWS;
        let rb{it} = (row0 + rr{it}) * row_vec4s + sb * 9u + 2u * ((k0 + 2u * rp{it}) / 2u);
        let rh{it} = weights[rb{it} - 2u * ((k0 + 2u * rp{it}) / 2u)];
        let rv{it}a = weights[rb{it} + 1u];
        let rv{it}b = weights[rb{it} + 2u];
"
            ));
        }
        for it in 0..tok_items {
            // A partial last round (32-token tiles: 64 items over 128
            // threads) clamps the item so the load stays in bounds and
            // skips the store.
            let clamp = if tok_partial && it + 1 == tok_items {
                format!("min(tid + {}u, TOKS * KS - 1u)", it * 128)
            } else {
                format!("tid + {}u", it * 128)
            };
            src.push_str(&format!(
                "        let ti{it} = {clamp};
        let tt{it} = ti{it} % TOKS;
        let tk{it} = ti{it} / TOKS;
        let tb{it} = ((tok0 + tt{it}) * n_super + sb) * 24u;
        let th{it} = q8x[tb{it} + k0 + tk{it}];
        let tq{it}a = q8x[tb{it} + 8u + 2u * (k0 + tk{it})];
        let tq{it}b = q8x[tb{it} + 9u + 2u * (k0 + tk{it})];
"
            ));
        }
        for it in 0..row_items {
            let guard = if row_partial && it + 1 == row_items {
                format!("if (tid + {}u < ROWS * (KS / 2u)) ", it * 128)
            } else {
                String::new()
            };
            src.push_str(&format!(
            r#"        {guard}{{
            let d = f16_to_f32(rh{it}.x & 0xFFFFu);
            let dm = f16_to_f32(rh{it}.x >> 16u);
            let lo0 = rv{it}a & m4;
            let lo1 = rv{it}b & m4;
            let hi0 = (rv{it}a >> f4) & m4;
            let hi1 = (rv{it}b >> f4) & m4;
            let e = ((2u * rp{it}) * ROWS + rr{it}) * 7u;
            wt[e] = lo0.x | (lo0.y << 4u);
            wt[e + 1u] = lo0.z | (lo0.w << 4u);
            wt[e + 2u] = lo1.x | (lo1.y << 4u);
            wt[e + 3u] = lo1.z | (lo1.w << 4u);
            let s0 = get_scale_min_k4_v4(vec3<u32>(rh{it}.y, rh{it}.z, rh{it}.w), k0 + 2u * rp{it});
            wt[e + 4u] = bitcast<u32>(d * f32(s0.x));
            wt[e + 5u] = bitcast<u32>(dm * f32(s0.y));
            let e1 = e + ROWS * 7u;
            wt[e1] = hi0.x | (hi0.y << 4u);
            wt[e1 + 1u] = hi0.z | (hi0.w << 4u);
            wt[e1 + 2u] = hi1.x | (hi1.y << 4u);
            wt[e1 + 3u] = hi1.z | (hi1.w << 4u);
            let s1 = get_scale_min_k4_v4(vec3<u32>(rh{it}.y, rh{it}.z, rh{it}.w), k0 + 2u * rp{it} + 1u);
            wt[e1 + 4u] = bitcast<u32>(d * f32(s1.x));
            wt[e1 + 5u] = bitcast<u32>(dm * f32(s1.y));
        }}
"#
        ));
        }
        for it in 0..tok_items {
            let guard = if tok_partial && it + 1 == tok_items {
                format!("if (tid + {}u < TOKS * KS) ", it * 128)
            } else {
                String::new()
            };
            src.push_str(&format!(
                r#"        {guard}{{
            let e = (tk{it} * TOKS + tt{it}) * 11u;
            xt[e] = tq{it}a.x;
            xt[e + 1u] = tq{it}a.y;
            xt[e + 2u] = tq{it}a.z;
            xt[e + 3u] = tq{it}a.w;
            xt[e + 4u] = tq{it}b.x;
            xt[e + 5u] = tq{it}b.y;
            xt[e + 6u] = tq{it}b.z;
            xt[e + 7u] = tq{it}b.w;
            xt[e + 8u] = th{it}.x;
            xt[e + 9u] = bitcast<u32>(bitcast<i32>(th{it}.y) + bitcast<i32>(th{it}.z));
        }}
"#
            ));
        }
        src.push_str(
            r#"        workgroupBarrier();
        var k: u32 = 0u;
        loop {
            if (k >= KS) { break; }
"#,
        );
    }
    // The thread's rows for this sub-block: unpacked once.
    for cr in 0..trows {
        src.push_str(&format!(
            "            let re{cr} = (k * ROWS + tiwr + {}u) * 7u;\n",
            32 * cr
        ));
        for q in 0..4 {
            src.push_str(&format!(
                "            let p{cr}_{q} = wt[re{cr} + {q}u];\n            let u{cr}_{} = p{cr}_{q} & 0x0F0F0F0Fu;\n            let u{cr}_{} = (p{cr}_{q} >> 4u) & 0x0F0F0F0Fu;\n",
                2 * q,
                2 * q + 1
            ));
        }
        src.push_str(&format!(
            "            let sc{cr} = bitcast<f32>(wt[re{cr} + 4u]);\n            let mn{cr} = bitcast<f32>(wt[re{cr} + 5u]);\n"
        ));
    }
    // The thread's thirty-two tokens, each against the four rows.
    for g in 0..tgroups {
        for cc in 0..4 {
            let t = g * 4 + cc;
            src.push_str(&format!(
                "            {{\n            let te = (k * TOKS + warp * {}u + {}u + tiwc * 4u) * 11u;\n",
                toks / 2,
                g * 8 + cc
            ));
            for i in 0..8 {
                src.push_str(&format!("            let q{i} = xt[te + {i}u];\n"));
            }
            src.push_str(
                "            let d = bitcast<f32>(xt[te + 8u]);\n            let dq = d * f32(bitcast<i32>(xt[te + 9u]));\n",
            );
            for cr in 0..trows {
                let dot = (0..8)
                    .map(|i| format!("dot4I8Packed(u{cr}_{i}, q{i})"))
                    .collect::<Vec<_>>()
                    .join(" + ");
                src.push_str(&format!(
                    "            {{ a{cr}_{t} = a{cr}_{t} + fma(sc{cr} * d, f32({dot}), -mn{cr} * dq); }}\n"
                ));
            }
            src.push_str("            }\n");
        }
    }
    src.push_str(
        r#"            k = k + 1u;
        }
        workgroupBarrier();
        slice = slice + 1u;
    }
"#,
    );
    for g in 0..tgroups {
        for cc in 0..4 {
            let t = g * 4 + cc;
            src.push_str(&format!(
                "    {{\n    let tok = tok0 + warp * {}u + {}u + tiwc * 4u;\n    if (tok < params.n_tokens) {{\n",
                toks / 2,
                g * 8 + cc
            ));
            for cr in 0..trows {
                src.push_str(&format!(
                    "        y[tok * params.out_dim + row0 + tiwr + {}u] = a{cr}_{t};\n",
                    32 * cr
                ));
            }
            src.push_str("    }\n    }\n");
        }
    }
    src.push_str("}\n");
    src
}

/// The wide integer-dot `Q6_K` GEMM: [`shader_source_mmq_q4k_wide`]'s
/// tile over [`shader_source_mmq_q6k`]'s unpacking. Per row and sub-block
/// the staging writes the 32 six-bit values as bytes (8 words, straight
/// from the `ql` nibbles and `qh` bit pairs the way the narrow kernel
/// unpacks a whole super-block) and the two `d·scale` of its sixteen-value
/// halves — 10 words at a stride of 11. Per token and sub-block: the 8
/// quant words, `d`, and `32·d·Σq` for each half (the `−32` offset folded
/// as the narrow kernel folds it) — 11 words at a stride of 13. Same
/// bindings as the narrow kernel: the weights as words, since a 210-byte
/// block may start two bytes into one.
pub fn shader_source_mmq_q6k_wide(rows: u32) -> String {
    shader_source_mmq_q6k_wide_toks(rows, MMQ_WIDE_TILE_TOKENS)
}

/// [`shader_source_mmq_q6k_wide`] at `toks` tokens per tile — see
/// [`shader_source_mmq_q4k_wide_toks`].
pub fn shader_source_mmq_q6k_wide_toks(rows: u32, toks: u32) -> String {
    assert!(
        toks.is_multiple_of(16) && toks <= 128,
        "a tile is 16 tokens per lane pair per warp"
    );
    let tgroups = (toks / 16) as usize;
    let ks = std::env::var("ORANGU_MMQ_WIDE_KS_Q6K")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(MMQ_WIDE_KS_Q6K);
    assert!(8 % ks == 0, "KS must divide 8");
    assert!(
        rows.is_multiple_of(32) && rows <= 128,
        "a wide tile is 32 rows per lane group"
    );
    let trows = (rows / 32) as usize;
    let mut src = String::new();
    src.push_str(&format!(
        r#"
struct Meta {{
    in_dim: u32,
    out_dim: u32,
    n_tokens: u32,
    row_bytes: u32,
}}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> q8x: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> params: Meta;

fn f16_to_f32(bits: u32) -> f32 {{
    return unpack2x16float(bits & 0xFFFFu).x;
}}

const ROWS: u32 = {rows}u;
const TOKS: u32 = {toks}u;
const KS: u32 = {ks}u;
// Per sub-block staged, per row: 8 words of six-bit values, d*sc0, d*sc1, pad.
var<workgroup> wt: array<u32, {wt_len}>;
// Per sub-block staged, per token: 8 quant words, d, 32*d*sumq_lo, 32*d*sumq_hi, pad, pad.
var<workgroup> xt: array<u32, {xt_len}>;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    let row0 = wid.x * ROWS;
    let tok0 = wid.y * TOKS;
    let tid = lid.x;
    let warp = tid / 64u;
    let lane = tid % 64u;
    let tiwr = lane % 32u;
    let tiwc = lane / 32u;
    let n_super = (params.in_dim + 255u) / 256u;
    let n_slices = n_super * (8u / KS);
"#,
        rows = rows,
        toks = toks,
        wt_len = rows * ks * 11,
        xt_len = toks * ks * 13,
    ));
    for cr in 0..trows {
        for t in 0..tgroups * 4 {
            src.push_str(&format!("    var a{cr}_{t}: f32 = 0.0;\n"));
        }
    }
    src.push_str(
        r#"
    var slice: u32 = 0u;
    loop {
        if (slice >= n_slices) { break; }
        let sb = slice / (8u / KS);
        let k0 = (slice % (8u / KS)) * KS;
        // Stage the rows: one item per (row, sub-block) — the eight `ql`
        // words and eight `qh` words its values come from, read as two runs
        // of consecutive words (nine each when the block starts mid-word;
        // the alignment is uniform across a slice whenever the row stride
        // is a multiple of four bytes, which every in_dim that is a
        // multiple of 512 gives), then the two scales and `d`.
        var i: u32 = tid;
        loop {
            if (i >= ROWS * KS) { break; }
            let r = i % ROWS;
            let k = i / ROWS;
            let b = k0 + k;
            let h = b / 4u;
            let j = b % 4u;
            let byte0 = (row0 + r) * params.row_bytes + sb * 210u;
            let e = (k * ROWS + r) * 11u;
            let w0 = byte0 / 4u;
            let qlb = w0 + 16u * h + 8u * (j % 2u);
            let qhb = w0 + 32u + 8u * h;
            var ql0: u32; var ql1: u32; var ql2: u32; var ql3: u32;
            var ql4: u32; var ql5: u32; var ql6: u32; var ql7: u32;
            var qh0: u32; var qh1: u32; var qh2: u32; var qh3: u32;
            var qh4: u32; var qh5: u32; var qh6: u32; var qh7: u32;
            var sc_word: u32;
            var d_word: u32;
            if ((byte0 & 3u) == 0u) {
                ql0 = weights[qlb]; ql1 = weights[qlb + 1u]; ql2 = weights[qlb + 2u]; ql3 = weights[qlb + 3u];
                ql4 = weights[qlb + 4u]; ql5 = weights[qlb + 5u]; ql6 = weights[qlb + 6u]; ql7 = weights[qlb + 7u];
                qh0 = weights[qhb]; qh1 = weights[qhb + 1u]; qh2 = weights[qhb + 2u]; qh3 = weights[qhb + 3u];
                qh4 = weights[qhb + 4u]; qh5 = weights[qhb + 5u]; qh6 = weights[qhb + 6u]; qh7 = weights[qhb + 7u];
                sc_word = weights[w0 + 48u + b / 2u];
                d_word = weights[w0 + 52u];
            } else {
                let a0 = weights[qlb]; let a1 = weights[qlb + 1u]; let a2 = weights[qlb + 2u]; let a3 = weights[qlb + 3u];
                let a4 = weights[qlb + 4u]; let a5 = weights[qlb + 5u]; let a6 = weights[qlb + 6u]; let a7 = weights[qlb + 7u];
                let a8 = weights[qlb + 8u];
                ql0 = (a0 >> 16u) | (a1 << 16u); ql1 = (a1 >> 16u) | (a2 << 16u);
                ql2 = (a2 >> 16u) | (a3 << 16u); ql3 = (a3 >> 16u) | (a4 << 16u);
                ql4 = (a4 >> 16u) | (a5 << 16u); ql5 = (a5 >> 16u) | (a6 << 16u);
                ql6 = (a6 >> 16u) | (a7 << 16u); ql7 = (a7 >> 16u) | (a8 << 16u);
                let c0 = weights[qhb]; let c1 = weights[qhb + 1u]; let c2 = weights[qhb + 2u]; let c3 = weights[qhb + 3u];
                let c4 = weights[qhb + 4u]; let c5 = weights[qhb + 5u]; let c6 = weights[qhb + 6u]; let c7 = weights[qhb + 7u];
                let c8 = weights[qhb + 8u];
                qh0 = (c0 >> 16u) | (c1 << 16u); qh1 = (c1 >> 16u) | (c2 << 16u);
                qh2 = (c2 >> 16u) | (c3 << 16u); qh3 = (c3 >> 16u) | (c4 << 16u);
                qh4 = (c4 >> 16u) | (c5 << 16u); qh5 = (c5 >> 16u) | (c6 << 16u);
                qh6 = (c6 >> 16u) | (c7 << 16u); qh7 = (c7 >> 16u) | (c8 << 16u);
                let s0 = weights[w0 + 48u + b / 2u];
                let s1 = weights[w0 + 49u + b / 2u];
                sc_word = (s0 >> 16u) | (s1 << 16u);
                // `d` is the block's last half-word: only its own word is read.
                d_word = weights[w0 + 52u] >> 16u;
            }
            let nsh = 4u * (j / 2u);
            let hsh = 2u * j;
            wt[e] = ((ql0 >> nsh) & 0x0F0F0F0Fu) | (((qh0 >> hsh) & 0x03030303u) << 4u);
            wt[e + 1u] = ((ql1 >> nsh) & 0x0F0F0F0Fu) | (((qh1 >> hsh) & 0x03030303u) << 4u);
            wt[e + 2u] = ((ql2 >> nsh) & 0x0F0F0F0Fu) | (((qh2 >> hsh) & 0x03030303u) << 4u);
            wt[e + 3u] = ((ql3 >> nsh) & 0x0F0F0F0Fu) | (((qh3 >> hsh) & 0x03030303u) << 4u);
            wt[e + 4u] = ((ql4 >> nsh) & 0x0F0F0F0Fu) | (((qh4 >> hsh) & 0x03030303u) << 4u);
            wt[e + 5u] = ((ql5 >> nsh) & 0x0F0F0F0Fu) | (((qh5 >> hsh) & 0x03030303u) << 4u);
            wt[e + 6u] = ((ql6 >> nsh) & 0x0F0F0F0Fu) | (((qh6 >> hsh) & 0x03030303u) << 4u);
            wt[e + 7u] = ((ql7 >> nsh) & 0x0F0F0F0Fu) | (((qh7 >> hsh) & 0x03030303u) << 4u);
            let d = f16_to_f32(d_word);
            let sh = 16u * (b % 2u);
            wt[e + 8u] = bitcast<u32>(d * f32(i32((sc_word >> sh) << 24u) >> 24u));
            wt[e + 9u] = bitcast<u32>(d * f32(i32((sc_word >> (sh + 8u)) << 24u) >> 24u));
            i = i + 128u;
        }
        // Stage the tokens: one item per (token, sub-block).
        i = tid;
        loop {
            if (i >= TOKS * KS) { break; }
            let t = i % TOKS;
            let k = i / TOKS;
            let b = k0 + k;
            let tb = ((tok0 + t) * n_super + sb) * 24u;
            let hd = q8x[tb + b];
            let q0 = q8x[tb + 8u + 2u * b];
            let q1 = q8x[tb + 9u + 2u * b];
            let e = (k * TOKS + t) * 13u;
            xt[e] = q0.x;
            xt[e + 1u] = q0.y;
            xt[e + 2u] = q0.z;
            xt[e + 3u] = q0.w;
            xt[e + 4u] = q1.x;
            xt[e + 5u] = q1.y;
            xt[e + 6u] = q1.z;
            xt[e + 7u] = q1.w;
            let d = bitcast<f32>(hd.x);
            xt[e + 8u] = hd.x;
            xt[e + 9u] = bitcast<u32>(32.0 * d * f32(bitcast<i32>(hd.y)));
            xt[e + 10u] = bitcast<u32>(32.0 * d * f32(bitcast<i32>(hd.z)));
            i = i + 128u;
        }
        workgroupBarrier();
        var k: u32 = 0u;
        loop {
            if (k >= KS) { break; }
"#,
    );
    for cr in 0..trows {
        src.push_str(&format!(
            "            let re{cr} = (k * ROWS + tiwr + {}u) * 11u;\n",
            32 * cr
        ));
        for i in 0..8 {
            src.push_str(&format!("            let u{cr}_{i} = wt[re{cr} + {i}u];\n"));
        }
        src.push_str(&format!(
            "            let s{cr}a = bitcast<f32>(wt[re{cr} + 8u]);\n            let s{cr}b = bitcast<f32>(wt[re{cr} + 9u]);\n"
        ));
    }
    for g in 0..tgroups {
        for cc in 0..4 {
            let t = g * 4 + cc;
            src.push_str(&format!(
                "            {{\n            let te = (k * TOKS + warp * {}u + {}u + tiwc * 4u) * 13u;\n",
                toks / 2,
                g * 8 + cc
            ));
            for i in 0..8 {
                src.push_str(&format!("            let q{i} = xt[te + {i}u];\n"));
            }
            src.push_str(
                "            let d = bitcast<f32>(xt[te + 8u]);\n            let lo = bitcast<f32>(xt[te + 9u]);\n            let hi = bitcast<f32>(xt[te + 10u]);\n",
            );
            for cr in 0..trows {
                let dlo = (0..4)
                    .map(|i| format!("dot4I8Packed(u{cr}_{i}, q{i})"))
                    .collect::<Vec<_>>()
                    .join(" + ");
                let dhi = (4..8)
                    .map(|i| format!("dot4I8Packed(u{cr}_{i}, q{i})"))
                    .collect::<Vec<_>>()
                    .join(" + ");
                src.push_str(&format!(
                    "            {{ a{cr}_{t} = a{cr}_{t} + fma(s{cr}a, fma(d, f32({dlo}), -lo), s{cr}b * fma(d, f32({dhi}), -hi)); }}\n"
                ));
            }
            src.push_str("            }\n");
        }
    }
    src.push_str(
        r#"            k = k + 1u;
        }
        workgroupBarrier();
        slice = slice + 1u;
    }
"#,
    );
    for g in 0..tgroups {
        for cc in 0..4 {
            let t = g * 4 + cc;
            src.push_str(&format!(
                "    {{\n    let tok = tok0 + warp * {}u + {}u + tiwc * 4u;\n    if (tok < params.n_tokens) {{\n",
                toks / 2,
                g * 8 + cc
            ));
            for cr in 0..trows {
                src.push_str(&format!(
                    "        y[tok * params.out_dim + row0 + tiwr + {}u] = a{cr}_{t};\n",
                    32 * cr
                ));
            }
            src.push_str("    }\n    }\n");
        }
    }
    src.push_str("}\n");
    src
}

/// The wide integer-dot GEMM for every other weight type the prefill sees
/// often — `Q5_K`, `Q8_0`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1` — as one
/// generator with a per-type staging routine.
///
/// [`shader_source_mmq_q4k_wide`]'s tile and loop, with each row's
/// sub-block staged **unpacked**: its 32 values as bytes in 8 words (the
/// packed nibble form fits only 4-bit types) and two floats `a`, `b` such
/// that the sub-block's contribution is `a · dₓ · Σ q·x + b · dₓ · Σ x` —
/// `(d·sc, −dmin·m)` for the K-quants, `(d, −8d)` for `Q4_0`, `(d, −16d)`
/// for `Q5_0`, `(d, m)` for the `_1` types and `(d, 0)` for `Q8_0`, whose
/// bytes are signed already. So one compute loop serves them all; only
/// the staging, generated per type from the block's byte layout, differs.
///
/// Blocks of 18, 22 and 34 bytes make a row's odd blocks start two bytes
/// into a word, so the weights are bound as words and each block's fields
/// are read through a per-parity variant of the staging (the parity is
/// uniform across a slice whenever the row stride is a multiple of four
/// bytes, which a multiple-of-256 `in_dim` guarantees). Rows: 10 words at
/// a stride of 11; tokens as the `Q4_K` kernel's. Two sub-blocks per
/// slice, for the occupancy the `Q4_K` measurement settled on.
pub fn shader_source_mmq_wide_bytes(ggml_type: u32, rows: u32) -> String {
    shader_source_mmq_wide_bytes_toks(ggml_type, rows, MMQ_WIDE_TILE_TOKENS)
}

/// [`shader_source_mmq_wide_bytes`] at `toks` tokens per tile — see
/// [`shader_source_mmq_q4k_wide_toks`].
pub fn shader_source_mmq_wide_bytes_toks(ggml_type: u32, rows: u32, toks: u32) -> String {
    use crate::engine::quant::*;
    assert!(
        toks.is_multiple_of(16) && toks <= 128,
        "a tile is 16 tokens per lane pair per warp"
    );
    let tgroups = (toks / 16) as usize;
    let ks = 2u32;
    assert!(
        rows.is_multiple_of(32) && rows <= 128,
        "a wide tile is 32 rows per lane group"
    );
    let trows = (rows / 32) as usize;
    let (block_bytes, block_elems): (u32, u32) = match ggml_type {
        GGML_TYPE_Q5_K => (176, 256),
        GGML_TYPE_Q8_0 => (34, 32),
        GGML_TYPE_Q4_0 => (18, 32),
        GGML_TYPE_Q4_1 => (20, 32),
        GGML_TYPE_Q5_0 => (22, 32),
        GGML_TYPE_Q5_1 => (24, 32),
        GGML_TYPE_Q2_K => (84, 256),
        GGML_TYPE_Q3_K => (110, 256),
        GGML_TYPE_IQ4_XS => (136, 256),
        GGML_TYPE_IQ3_S => (110, 256),
        GGML_TYPE_IQ2_S => (82, 256),
        t => panic!("no byte-unpacked integer-dot kernel for ggml type {t}"),
    };
    // `Q2_K`, `Q3_K` and `IQ2_S` scale each *sixteen* values, so a sub-block
    // has two `(a, b)` pairs and the dot is taken per half — against the two
    // sums of quants the activation layout keeps per half for exactly this.
    let halves = matches!(ggml_type, GGML_TYPE_Q2_K | GGML_TYPE_Q3_K | GGML_TYPE_IQ2_S);
    let row_words: u32 = if halves { 13 } else { 11 };
    // Whether a block can start two bytes into a word: only when its size
    // is not a multiple of four.
    let may_shift = !block_bytes.is_multiple_of(4);

    // WGSL for "the four block bytes at `off`" and "the half-word at
    // `off`" under a given parity, with `w0` the block's first (possibly
    // partial) word.
    let word_at = |off: u32, parity: u32| -> String {
        let byte = 2 * parity + off;
        let idx = byte / 4;
        if byte.is_multiple_of(4) {
            format!("weights[w0 + {idx}u]")
        } else {
            format!(
                "((weights[w0 + {idx}u] >> 16u) | (weights[w0 + {}u] << 16u))",
                idx + 1
            )
        }
    };
    // Like `word_at`, with a runtime byte offset (a multiple of four) added
    // to the block byte.
    let word_at_g = |off: u32, parity: u32, plus_bytes: &str| -> String {
        let byte = 2 * parity + off;
        let idx = byte / 4;
        if byte.is_multiple_of(4) {
            format!("weights[w0 + {idx}u + ({plus_bytes}) / 4u]")
        } else {
            format!(
                "((weights[w0 + {idx}u + ({plus_bytes}) / 4u] >> 16u) | (weights[w0 + {}u + ({plus_bytes}) / 4u] << 16u))",
                idx + 1
            )
        }
    };
    let half_at = |off: u32, parity: u32| -> String {
        let byte = 2 * parity + off;
        let idx = byte / 4;
        if byte.is_multiple_of(4) {
            format!("(weights[w0 + {idx}u] & 0xFFFFu)")
        } else {
            format!("(weights[w0 + {idx}u] >> 16u)")
        }
    };
    // The staging body for one parity: sets `u0..u7`, `sa`, `sb` from the
    // block at word `w0` (and, for a 256-block, sub-block `sub`).
    let stage = |parity: u32| -> String {
        let mut o = String::new();
        match ggml_type {
            GGML_TYPE_Q8_0 => {
                for i in 0..8u32 {
                    o += &format!("            u{i} = {};\n", word_at(2 + 4 * i, parity));
                }
                o += &format!(
                    "            sa = f16_to_f32({});\n            sb = 0.0;\n",
                    half_at(0, parity)
                );
            }
            GGML_TYPE_Q4_0 | GGML_TYPE_Q4_1 => {
                let qs = if ggml_type == GGML_TYPE_Q4_0 { 2 } else { 4 };
                for i in 0..4u32 {
                    o += &format!("            let q{i} = {};\n", word_at(qs + 4 * i, parity));
                }
                for i in 0..4u32 {
                    o += &format!(
                        "            u{i} = q{i} & 0x0F0F0F0Fu;\n            u{} = (q{i} >> 4u) & 0x0F0F0F0Fu;\n",
                        i + 4
                    );
                }
                o += &format!("            let d = f16_to_f32({});\n", half_at(0, parity));
                if ggml_type == GGML_TYPE_Q4_0 {
                    o += "            sa = d;\n            sb = -8.0 * d;\n";
                } else {
                    o += &format!(
                        "            sa = d;\n            sb = f16_to_f32({});\n",
                        half_at(2, parity)
                    );
                }
            }
            GGML_TYPE_Q5_0 | GGML_TYPE_Q5_1 => {
                let (qh, qs) = if ggml_type == GGML_TYPE_Q5_0 {
                    (2, 6)
                } else {
                    (4, 8)
                };
                o += &format!("            let qh = {};\n", word_at(qh, parity));
                for i in 0..4u32 {
                    o += &format!("            let q{i} = {};\n", word_at(qs + 4 * i, parity));
                }
                // Element e takes bit e of `qh` as its fifth bit; spread the
                // four bits of each element group into byte lanes.
                for i in 0..8u32 {
                    let nib = if i < 4 {
                        format!("(q{i} & 0x0F0F0F0Fu)")
                    } else {
                        format!("((q{} >> 4u) & 0x0F0F0F0Fu)", i - 4)
                    };
                    o += &format!(
                        "            u{i} = {nib} | spread_bits(qh >> {}u);\n",
                        4 * i
                    );
                }
                o += &format!("            let d = f16_to_f32({});\n", half_at(0, parity));
                if ggml_type == GGML_TYPE_Q5_0 {
                    o += "            sa = d;\n            sb = -16.0 * d;\n";
                } else {
                    o += &format!(
                        "            sa = d;\n            sb = f16_to_f32({});\n",
                        half_at(2, parity)
                    );
                }
            }
            GGML_TYPE_Q5_K => {
                // d, dmin, scales[12] at 0; qh[32] at 16; qs[128] at 48.
                // Sub-block `sub`: nibble pair `sub / 2` (32 bytes of qs),
                // low or high nibble by `sub % 2`, fifth bit `sub` of qh.
                o += &format!(
                    "            let h = vec4<u32>({}, {}, {}, {});\n",
                    word_at(0, parity),
                    word_at(4, parity),
                    word_at(8, parity),
                    word_at(12, parity)
                );
                o += "            let jp = sub / 2u;\n            let nsh = 4u * (sub % 2u);\n";
                for i in 0..8u32 {
                    o += &format!(
                        "            u{i} = ((weights[w0 + 12u + 8u * jp + {i}u] >> nsh) & 0x0F0F0F0Fu) | (((weights[w0 + 4u + {i}u] >> sub) & 0x01010101u) << 4u);\n"
                    );
                }
                o += "            let d = f16_to_f32(h.x & 0xFFFFu);\n            let dm = f16_to_f32(h.x >> 16u);\n            let p = get_scale_min_k4_v4(vec3<u32>(h.y, h.z, h.w), sub);\n            sa = d * f32(p.x);\n            sb = -dm * f32(p.y);\n";
            }
            GGML_TYPE_Q2_K => {
                // scales[16] at 0, qs[64] at 16, d at 80, dmin at 82.
                // Sub-block `sub`: 128-group `sub / 4`, bit pair `sub % 4`;
                // its halves' scale bytes are `2·sub` and `2·sub + 1`, low
                // nibble the scale, high nibble the min.
                o += "            let grp = sub / 4u;
            let sh = 2u * (sub % 4u);
";
                // Word `i` of the sub-block: the group's bytes `4i..4i+4`
                // for the low half and `16 + 4(i-4)..` for the high — the
                // same word `4 + 8·grp + i` either way.
                for i in 0..8u32 {
                    o += &format!(
                        "            u{i} = (weights[w0 + 4u + 8u * grp + {i}u] >> sh) & 0x03030303u;
"
                    );
                }
                o += &format!(
                    "            let d = f16_to_f32({});
            let dm = f16_to_f32({});
",
                    half_at(80, 0),
                    half_at(82, 0)
                );
                o += "            let sc0 = (weights[w0 + (2u * sub) / 4u] >> (8u * ((2u * sub) % 4u))) & 0xFFu;
            let sc1 = (weights[w0 + (2u * sub + 1u) / 4u] >> (8u * ((2u * sub + 1u) % 4u))) & 0xFFu;
";
                o += "            sa = d * f32(sc0 & 0xFu);
            sb = -dm * f32(sc0 >> 4u);
            sa1 = d * f32(sc1 & 0xFu);
            sb1 = -dm * f32(sc1 >> 4u);
";
            }
            GGML_TYPE_Q3_K => {
                // hmask[32] at 0, qs[64] at 32, scales[12] at 96, d at 108.
                // A value is its two `qs` bits with the `hmask` bit `sub` as
                // the third, minus four — signed, so the dot takes it as is.
                o += "            let grp = sub / 4u;
            let sh = 2u * (sub % 4u);
";
                for i in 0..8u32 {
                    let qs_off = 32 + 4 * i; // + 32·g
                    let hm_off = 4 * i;
                    o += &format!(
                        "            {{
                let q2 = ({} >> sh) & 0x03030303u;
                let hb = ({} >> sub) & 0x01010101u;
                let v = q2 | (hb << 2u);
                u{i} = ((v | 0x80808080u) - 0x04040404u) ^ 0x80808080u;
            }}
",
                        word_at_g(qs_off, parity, "32u * grp"),
                        word_at(hm_off, parity)
                    );
                }
                o += &format!(
                    "            let a0 = {};
            let a1 = {};
            let a2 = {};
",
                    word_at(96, parity),
                    word_at(100, parity),
                    word_at(104, parity)
                );
                o += "            let x0 = (a0 & 0x0F0F0F0Fu) | ((a2 & 0x03030303u) << 4u);
            let x1 = (a1 & 0x0F0F0F0Fu) | (((a2 >> 2u) & 0x03030303u) << 4u);
            let x2 = ((a0 >> 4u) & 0x0F0F0F0Fu) | (((a2 >> 4u) & 0x03030303u) << 4u);
            let x3 = ((a1 >> 4u) & 0x0F0F0F0Fu) | (((a2 >> 6u) & 0x03030303u) << 4u);
";
                o += "            let is0 = 2u * sub;
            let w_lo = select(select(x0, x1, is0 >= 4u), select(x2, x3, is0 >= 12u), is0 >= 8u);
            let s0 = (w_lo >> (8u * (is0 % 4u))) & 0xFFu;
            let s1 = (w_lo >> (8u * ((is0 + 1u) % 4u))) & 0xFFu;
";
                o += &format!(
                    "            let d = f16_to_f32({});
",
                    half_at(108, parity)
                );
                o += "            sa = d * (f32(s0) - 32.0);
            sb = 0.0;
            sa1 = d * (f32(s1) - 32.0);
            sb1 = 0.0;
";
            }
            GGML_TYPE_IQ4_XS => {
                // d at 0, scales_h (u16) at 2, scales_l[4] at 4, qs[128] at 8;
                // a nibble indexes the sixteen-entry value table.
                o += "            let sw = weights[w0];
            let scales_h = sw >> 16u;
            let sl_word = weights[w0 + 1u];
            let low = (sl_word >> (8u * (sub / 2u) + 4u * (sub % 2u))) & 0xFu;
            let high = (scales_h >> (2u * sub)) & 3u;
            let ls = low | (high << 4u);
";
                for i in 0..8u32 {
                    let word = 2 + if i < 4 { i } else { i - 4 };
                    let shift = if i < 4 { "" } else { " >> 4u" };
                    o += &format!(
                        "            u{i} = iq4nl_bytes((weights[w0 + 4u * sub + {word}u]{shift}) & 0x0F0F0F0Fu);
"
                    );
                }
                o += "            let d = f16_to_f32(sw & 0xFFFFu);
            sa = d * (f32(ls) - 32.0);
            sb = 0.0;
";
            }
            GGML_TYPE_IQ3_S => {
                // d at 0, qs[64] at 2, qh[8] at 66, signs[32] at 74,
                // scales[4] at 106. Word `i` of sub-block `sub` is lattice
                // point `qs[8·sub + i]` with bit `i` of `qh[sub]` as its
                // ninth index bit, negated where nibble `i` of the
                // sub-block's four sign bytes says; one scale per sub-block.
                o += &format!(
                    "            let qh = ({} >> (8u * (sub % 4u))) & 0xFFu;\n",
                    word_at_g(66, parity, "sub & ~3u")
                );
                o += &format!(
                    "            let sg0 = {};\n",
                    word_at_g(74, parity, "4u * sub")
                );
                for i in 0..8u32 {
                    // Index bytes `8·sub + i`: word `2·sub + i/4`, byte `i%4`.
                    let idx_word = word_at_g(2 + 4 * (i / 4), parity, "8u * sub");
                    o += &format!(
                        "            u{i} = iq_signed_bytes(iq_grids[IQ3S_GRID_OFF + ((({idx_word} >> {}u) & 0xFFu) | (((qh >> {i}u) & 1u) << 8u))], (sg0 >> {}u) & 0xFu);\n",
                        8 * (i % 4),
                        4 * i
                    );
                }
                o += &format!(
                    "            let sc = ({} >> (8u * ((sub / 2u) % 4u) + 4u * (sub % 2u))) & 0xFu;\n",
                    word_at(106, parity)
                );
                o += &format!("            let d = f16_to_f32({});\n", half_at(0, parity));
                o += "            sa = d * f32(1u + 2u * sc);\n            sb = 0.0;\n";
            }
            GGML_TYPE_IQ2_S => {
                // d at 0, qs[32] at 2 then signs[32] at 34, qh[8] at 66,
                // scales[8] at 74. Run `l` (eight values) of sub-block
                // `sub` is lattice point `qs[4·sub + l]` with bits `2l..2l+2`
                // of `qh[sub]` above it, one sign bit per value in
                // `signs[4·sub + l]`; the scale nibbles are per half.
                o += &format!(
                    "            let qh = ({} >> (8u * (sub % 4u))) & 0xFFu;\n            let iw = {};\n            let sgw = {};\n",
                    word_at_g(66, parity, "sub & ~3u"),
                    word_at_g(2, parity, "4u * sub"),
                    word_at_g(34, parity, "4u * sub")
                );
                for i in 0..8u32 {
                    let l = i / 2;
                    o += &format!(
                        "            u{i} = iq_signed_bytes(iq_grids[IQ2S_GRID_OFF + 2u * (((iw >> {}u) & 0xFFu) | (((qh >> {}u) & 3u) << 8u)) + {}u], (sgw >> {}u) & 0xFu);\n",
                        8 * l,
                        2 * l,
                        i % 2,
                        8 * l + 4 * (i % 2)
                    );
                }
                o += &format!(
                    "            let sc = ({} >> (8u * (sub % 4u))) & 0xFFu;\n",
                    word_at_g(74, parity, "sub & ~3u")
                );
                o += &format!("            let d = f16_to_f32({});\n", half_at(0, parity));
                o += "            sa = d * (0.5 + f32(sc & 0xFu)) * 0.25;\n            sb = 0.0;\n            sa1 = d * (0.5 + f32(sc >> 4u)) * 0.25;\n            sb1 = 0.0;\n";
            }
            _ => unreachable!(),
        }
        o
    };
    let staging = if may_shift {
        format!(
            "            if ((byte0 & 3u) == 0u) {{\n{}            }} else {{\n{}            }}\n",
            stage(0),
            stage(1)
        )
    } else {
        stage(0)
    };

    let mut src = String::new();
    src.push_str(&format!(
        r#"
struct Meta {{
    in_dim: u32,
    out_dim: u32,
    n_tokens: u32,
    row_bytes: u32,
}}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> q8x: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> params: Meta;
@group(0) @binding(4) var<storage, read> iq_grids: array<u32>;

const IQ2S_GRID_OFF: u32 = 1024u;
const IQ3S_GRID_OFF: u32 = 3328u;
const KVALUES_IQ4NL_OFF: u32 = 3872u;

fn f16_to_f32(bits: u32) -> f32 {{
    return unpack2x16float(bits & 0xFFFFu).x;
}}
// Each nibble of `n` to its `kvalues_iq4nl` entry, a signed byte.
fn iq4nl_byte(i: u32) -> u32 {{
    return (iq_grids[KVALUES_IQ4NL_OFF + (i >> 2u)] >> ((i & 3u) * 8u)) & 0xFFu;
}}
fn iq4nl_bytes(n: u32) -> u32 {{
    return iq4nl_byte(n & 0xFu) | (iq4nl_byte((n >> 8u) & 0xFu) << 8u)
        | (iq4nl_byte((n >> 16u) & 0xFu) << 16u) | (iq4nl_byte((n >> 24u) & 0xFu) << 24u);
}}
// A lattice word's four (small, positive) bytes, each negated where its
// bit of the four-bit `signs` is set: two's complement per byte lane,
// which cannot carry because every value is below 128.
fn iq_signed_bytes(g: u32, signs: u32) -> u32 {{
    let m = (spread_bits(signs) >> 4u) * 0xFFu;
    return (g ^ m) + (m & 0x01010101u);
}}
// Bits 0..4 of `x` to bit 4 of bytes 0..4 — a nibble's fifth bit per lane.
fn spread_bits(x: u32) -> u32 {{
    return ((x & 1u) << 4u) | ((x & 2u) << 11u) | ((x & 4u) << 18u) | ((x & 8u) << 25u);
}}
fn vec3_word(v: vec3<u32>, i: u32) -> u32 {{
    if (i == 0u) {{ return v.x; }}
    if (i == 1u) {{ return v.y; }}
    return v.z;
}}
fn get_scale_min_k4_v4(scales: vec3<u32>, j: u32) -> vec2<u32> {{
    if (j < 4u) {{
        let qj = (vec3_word(scales, j / 4u) >> (8u * (j % 4u))) & 0xFFu;
        let qj4 = (vec3_word(scales, (j + 4u) / 4u) >> (8u * ((j + 4u) % 4u))) & 0xFFu;
        return vec2<u32>(qj & 63u, qj4 & 63u);
    }}
    let qj = (vec3_word(scales, j / 4u) >> (8u * (j % 4u))) & 0xFFu;
    let qj4 = (vec3_word(scales, (j + 4u) / 4u) >> (8u * ((j + 4u) % 4u))) & 0xFFu;
    let qjm4 = (vec3_word(scales, (j - 4u) / 4u) >> (8u * ((j - 4u) % 4u))) & 0xFFu;
    let sc = (qj4 & 0xFu) | ((qjm4 >> 6u) << 4u);
    let m = (qj4 >> 4u) | ((qj >> 6u) << 4u);
    return vec2<u32>(sc, m);
}}

const ROWS: u32 = {rows}u;
const TOKS: u32 = {toks}u;
const KS: u32 = {ks}u;
const BLOCK_BYTES: u32 = {block_bytes}u;
const SUBS_PER_BLOCK: u32 = {subs}u;
const ROW_WORDS: u32 = {row_words}u;
var<workgroup> wt: array<u32, {wt_len}>;
var<workgroup> xt: array<u32, {xt_len}>;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    let row0 = wid.x * ROWS;
    let tok0 = wid.y * TOKS;
    let tid = lid.x;
    let warp = tid / 64u;
    let lane = tid % 64u;
    let tiwr = lane % 32u;
    let tiwc = lane / 32u;
    let n_super = (params.in_dim + 255u) / 256u;
    let n_slices = params.in_dim / 32u / KS;
"#,
        rows = rows,
        toks = toks,
        ks = ks,
        block_bytes = block_bytes,
        subs = block_elems / 32,
        row_words = row_words,
        wt_len = rows * ks * row_words,
        xt_len = toks * ks * 11,
    ));
    for cr in 0..trows {
        for t in 0..tgroups * 4 {
            src.push_str(&format!("    var a{cr}_{t}: f32 = 0.0;\n"));
        }
    }
    src.push_str(&format!(
        r#"
    var slice: u32 = 0u;
    loop {{
        if (slice >= n_slices) {{ break; }}
        let g0 = slice * KS;
        // Stage the rows: one item per (row, sub-block).
        var i: u32 = tid;
        loop {{
            if (i >= ROWS * KS) {{ break; }}
            let r = i % ROWS;
            let k = i / ROWS;
            let g = g0 + k;
            let blk = g / SUBS_PER_BLOCK;
            let sub = g % SUBS_PER_BLOCK;
            let byte0 = (row0 + r) * params.row_bytes + blk * BLOCK_BYTES;
            let w0 = byte0 / 4u;
            var u0: u32; var u1: u32; var u2: u32; var u3: u32;
            var u4: u32; var u5: u32; var u6: u32; var u7: u32;
            var sa: f32; var sb: f32; var sa1: f32 = 0.0; var sb1: f32 = 0.0;
{staging}            let e = (k * ROWS + r) * ROW_WORDS;
            wt[e] = u0; wt[e + 1u] = u1; wt[e + 2u] = u2; wt[e + 3u] = u3;
            wt[e + 4u] = u4; wt[e + 5u] = u5; wt[e + 6u] = u6; wt[e + 7u] = u7;
            wt[e + 8u] = bitcast<u32>(sa);
            wt[e + 9u] = bitcast<u32>(sb);
{halves_write}            i = i + 128u;
        }}
        // Stage the tokens: one item per (token, sub-block), from the q8
        // layout's 256-element super-blocks.
        i = tid;
        loop {{
            if (i >= TOKS * KS) {{ break; }}
            let t = i % TOKS;
            let k = i / TOKS;
            let g = g0 + k;
            let sbk = g / 8u;
            let b = g % 8u;
            let tb = ((tok0 + t) * n_super + sbk) * 24u;
            let hd = q8x[tb + b];
            let q0 = q8x[tb + 8u + 2u * b];
            let q1 = q8x[tb + 9u + 2u * b];
            let e = (k * TOKS + t) * 11u;
            xt[e] = q0.x;
            xt[e + 1u] = q0.y;
            xt[e + 2u] = q0.z;
            xt[e + 3u] = q0.w;
            xt[e + 4u] = q1.x;
            xt[e + 5u] = q1.y;
            xt[e + 6u] = q1.z;
            xt[e + 7u] = q1.w;
            xt[e + 8u] = hd.x;
{token_sums}            i = i + 128u;
        }}
        workgroupBarrier();
        var k: u32 = 0u;
        loop {{
            if (k >= KS) {{ break; }}
"#,
        halves_write = if halves {
            "            wt[e + 10u] = bitcast<u32>(sa1);\n            wt[e + 11u] = bitcast<u32>(sb1);\n"
        } else {
            ""
        },
        token_sums = if halves {
            "            xt[e + 9u] = hd.y;\n            xt[e + 10u] = hd.z;\n"
        } else {
            "            xt[e + 9u] = bitcast<u32>(bitcast<i32>(hd.y) + bitcast<i32>(hd.z));\n"
        },
    ));
    for cr in 0..trows {
        src.push_str(&format!(
            "            let re{cr} = (k * ROWS + tiwr + {}u) * ROW_WORDS;\n",
            32 * cr
        ));
        for i in 0..8 {
            src.push_str(&format!("            let u{cr}_{i} = wt[re{cr} + {i}u];\n"));
        }
        src.push_str(&format!(
            "            let sa{cr} = bitcast<f32>(wt[re{cr} + 8u]);\n            let sb{cr} = bitcast<f32>(wt[re{cr} + 9u]);\n"
        ));
        if halves {
            src.push_str(&format!(
                "            let sc{cr} = bitcast<f32>(wt[re{cr} + 10u]);\n            let sd{cr} = bitcast<f32>(wt[re{cr} + 11u]);\n"
            ));
        }
    }
    for g in 0..tgroups {
        for cc in 0..4 {
            let t = g * 4 + cc;
            src.push_str(&format!(
                "            {{\n            let te = (k * TOKS + warp * {}u + {}u + tiwc * 4u) * 11u;\n",
                toks / 2,
                g * 8 + cc
            ));
            for i in 0..8 {
                src.push_str(&format!("            let q{i} = xt[te + {i}u];\n"));
            }
            if halves {
                src.push_str(
                    "            let d = bitcast<f32>(xt[te + 8u]);\n            let dqlo = d * f32(bitcast<i32>(xt[te + 9u]));\n            let dqhi = d * f32(bitcast<i32>(xt[te + 10u]));\n",
                );
                for cr in 0..trows {
                    let lo = (0..4)
                        .map(|i| format!("dot4I8Packed(u{cr}_{i}, q{i})"))
                        .collect::<Vec<_>>()
                        .join(" + ");
                    let hi = (4..8)
                        .map(|i| format!("dot4I8Packed(u{cr}_{i}, q{i})"))
                        .collect::<Vec<_>>()
                        .join(" + ");
                    src.push_str(&format!(
                        "            {{ a{cr}_{t} = a{cr}_{t} + fma(sa{cr} * d, f32({lo}), sb{cr} * dqlo) + fma(sc{cr} * d, f32({hi}), sd{cr} * dqhi); }}\n"
                    ));
                }
            } else {
                src.push_str(
                    "            let d = bitcast<f32>(xt[te + 8u]);\n            let dq = d * f32(bitcast<i32>(xt[te + 9u]));\n",
                );
                for cr in 0..trows {
                    let dot = (0..8)
                        .map(|i| format!("dot4I8Packed(u{cr}_{i}, q{i})"))
                        .collect::<Vec<_>>()
                        .join(" + ");
                    src.push_str(&format!(
                        "            {{ a{cr}_{t} = a{cr}_{t} + fma(sa{cr} * d, f32({dot}), sb{cr} * dq); }}\n"
                    ));
                }
            }
            src.push_str("            }\n");
        }
    }
    src.push_str(
        r#"            k = k + 1u;
        }
        workgroupBarrier();
        slice = slice + 1u;
    }
"#,
    );
    for g in 0..tgroups {
        for cc in 0..4 {
            let t = g * 4 + cc;
            src.push_str(&format!(
                "    {{\n    let tok = tok0 + warp * {}u + {}u + tiwc * 4u;\n    if (tok < params.n_tokens) {{\n",
                toks / 2,
                g * 8 + cc
            ));
            for cr in 0..trows {
                src.push_str(&format!(
                    "        y[tok * params.out_dim + row0 + tiwr + {}u] = a{cr}_{t};\n",
                    32 * cr
                ));
            }
            src.push_str("    }\n    }\n");
        }
    }
    src.push_str("}\n");
    src
}

/// An integer-dot GEMM kernel over a **stack of expert matrices**, each
/// multiplied by the rows routed to it, in one dispatch — from the source
/// of one of the wide kernels (`shader_source_mmq_q4k_wide`,
/// `shader_source_mmq_q6k_wide`, `shader_source_mmq_wide_bytes`).
///
/// The wide kernels address three things by their workgroup id: the weight
/// rows (`wid.x`), the activation rows and the output rows (both `wid.y`).
/// This rewrites those three addresses to come from a table instead:
///
/// - `ids[8 * wid.y ..]` is the workgroup's **group entry** — the row the
///   expert's rows start at in the stack, where its token list starts in
///   `ids`, how many of that list's tokens this tile really has, where its
///   output rows start, and (as `f32` bits) the scale its output rows are
///   multiplied by — a per-expert output scalar, `1.0` when there is none;
///   three words spare;
/// - an activation row is `ids[list + t]`, the routed token's row in the
///   one quantized activation buffer, so the activations are staged once
///   for the whole layer rather than gathered per expert;
/// - an output row is `out_base + t`, so every expert's rows land in one
///   contiguous `[sum of counts, out_dim]` result.
///
/// Everything else — the staging, the dot, the tiles — is the wide kernel
/// as generated, which is the point: the expert GEMM is the dense GEMM with
/// three addresses looked up, not a kernel of its own to keep right.
///
/// The token lists are padded to a whole token tile with row `0`, a valid
/// row whose products are computed and discarded by the count guard.
pub fn shader_source_mmq_indexed(wide: String) -> String {
    let src = wide.replacen(
        "@group(0) @binding(3) var<uniform> params: Meta;",
        "@group(0) @binding(3) var<uniform> params: Meta;\n\
         @group(0) @binding(5) var<storage, read> ids: array<u32>;",
        1,
    );
    assert!(
        src.contains("@binding(5)"),
        "the wide kernel's Meta binding"
    );
    let src = src.replacen(
        "    let row0 = wid.x * ROWS;\n    let tok0 = wid.y * TOKS;\n",
        "    let grp = wid.y * 8u;\n    let orow0 = wid.x * ROWS;\n    let row0 = orow0 + ids[grp];\n    \
         let list_base = ids[grp + 1u];\n    let n_count = ids[grp + 2u];\n    let out_base = ids[grp + 3u];\n    \
         let oscale = bitcast<f32>(ids[grp + 4u]);\n    let tok0 = 0u;\n",
        1,
    );
    assert!(src.contains("let orow0"), "the wide kernel's tile origin");
    // Activation rows through the token list.
    let re = regex::Regex::new(r"\(tok0 \+ (\w+)\) \* n_super").expect("a literal pattern");
    let src = re
        .replace_all(&src, "ids[list_base + $1] * n_super")
        .into_owned();
    assert!(
        !src.contains("(tok0 +"),
        "every activation row goes through the list"
    );
    // Output rows through the group's base, guarded by its count.
    let src = src.replace("if (tok < params.n_tokens) {", "if (tok < n_count) {");
    let src = src.replace(
        "y[tok * params.out_dim + row0 +",
        "y[(out_base + tok) * params.out_dim + orow0 +",
    );
    assert!(
        src.contains("(out_base + tok)"),
        "the wide kernel's output rows"
    );
    // The output scale, on every store: `... = aN_M;` → `... = aN_M * oscale;`.
    let re = regex::Regex::new(r"\] = (a\d+_\d+);").expect("a literal pattern");
    let src = re.replace_all(&src, "] = $1 * oscale;").into_owned();
    assert!(src.contains("* oscale;"), "the wide kernel's stores");
    src
}

/// The weight types [`shader_source_mmq_wide_bytes`] generates a kernel for.
pub fn mmq_wide_bytes_types() -> &'static [u32] {
    use crate::engine::quant::*;
    &[
        GGML_TYPE_Q5_K,
        GGML_TYPE_Q8_0,
        GGML_TYPE_Q4_0,
        GGML_TYPE_Q4_1,
        GGML_TYPE_Q5_0,
        GGML_TYPE_Q5_1,
        GGML_TYPE_Q2_K,
        GGML_TYPE_Q3_K,
        GGML_TYPE_IQ4_XS,
        GGML_TYPE_IQ3_S,
        GGML_TYPE_IQ2_S,
    ]
}

/// The integer-dot `Q6_K` GEMM — [`shader_source_mmq_q4k`]'s structure
/// over the other K-quant the served models' FFNs use (`ffn_down` on most
/// layers of a `Q4_K_M` file). Two things differ, both in the staging:
///
/// - a `Q6_K` block is 210 bytes — `ql[128]` (the low four bits of 256
///   values), `qh[64]` (the high two), sixteen signed 8-bit scales, one per
///   **sixteen** values, and `d` — so a row's blocks are not 16-byte aligned
///   and the odd ones start two bytes into a word. The weights are bound as
///   words, and a block's words are read through the per-block shift, the
///   way the decode kernel hoists its misalignment;
/// - the staging **unpacks** each block to bytes: for sub-block `b` (values
///   `32b..32b+32`), value `l` is `ql` byte `64(b/4) + 32(b%2) + l`'s low or
///   high nibble (`b%4 < 2` low) and `qh` byte `32(b/4) + l`'s bits `2(b%4)`,
///   the row order ggml's `dequantize_row_q6_K` writes — done per word with
///   masks (two `ql` words and one `qh` word make the four sub-blocks' words
///   of one half), so the tile holds `0..63` bytes the dot reads directly,
///   no masks in the inner loop. The `−32` is folded into the min term the way `Q4_K`'s
///   `dmin·m` is: `Σ(q−32)x = Σqx − 32Σx`, with `Σx` per sixteen because the
///   scales are.
///
/// The scale table holds `d·sc` per sixteen values, at a row stride of 17
/// floats, and the unpacked rows at a stride of 17 `vec4` — both so the
/// lanes' reads spread across the banks, as in the `Q4_K` kernel. Shared
/// memory: 17.4 KiB of rows, 25.6 KiB of tokens, 4.3 KiB of scales.
///
/// Shapes: `in_dim` a multiple of 256, `out_dim` of `MMQ_TILE_ROWS`, any
/// `n_tokens`. Bindings as the `Q4_K` kernel's, with the weights as
/// `array<u32>`.
pub fn shader_source_mmq_q6k() -> String {
    let tr = MMQ_THREAD_ROWS as usize;
    let tt = (MMQ_Q6K_TILE_TOKENS / 16) as usize;
    let tile_tokens = MMQ_Q6K_TILE_TOKENS;
    let mut src = String::new();
    src.push_str(&format!(
        r#"
struct Meta {{
    in_dim: u32,
    out_dim: u32,
    n_tokens: u32,
    row_bytes: u32,
}}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> q8x: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> params: Meta;

fn f16_to_f32(bits: u32) -> f32 {{
    return unpack2x16float(bits & 0xFFFFu).x;
}}
// Word `k` of the block whose first byte is `byte0` — the block may start
// two bytes into a word.
fn block_word(byte0: u32, k: u32) -> u32 {{
    let w0 = byte0 / 4u;
    let lo = weights[w0 + k];
    if ((byte0 & 3u) == 0u) {{
        return lo;
    }}
    // The last word only needs its low half (`d`), and reading past it
    // could run off the binding.
    if (k >= 52u) {{
        return lo >> 16u;
    }}
    return (lo >> 16u) | (weights[w0 + k + 1u] << 16u);
}}

const TILE_ROWS: u32 = {rows}u;
const TILE_TOKENS: u32 = {tokens}u;
// One super-block per row, unpacked to bytes: 16 vec4 of quants at a
// stride of 17.
var<workgroup> wt: array<vec4<u32>, {wt_len}>;
// One super-block per token: 8 (d, sumq_lo, sumq_hi, 0), then 64 words of
// quants, padded to 25 vec4.
var<workgroup> xt: array<vec4<u32>, {xt_len}>;
// Each row's sixteen d * scale, at a stride of 17.
var<workgroup> sm: array<f32, {sm_len}>;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    let row0 = wid.x * TILE_ROWS;
    let tok0 = wid.y * TILE_TOKENS;
    let local = lid.x;
    let tr = local / 16u;
    let tt = local % 16u;
    let n_super = (params.in_dim + 255u) / 256u;
    let wbase = tr * 17u;
    let xbase = tt * 25u;
    let sbase = tr * 17u;
"#,
        rows = MMQ_TILE_ROWS,
        tokens = tile_tokens,
        wt_len = MMQ_TILE_ROWS * 17,
        xt_len = tile_tokens * 25,
        sm_len = MMQ_TILE_ROWS * 17,
    ));
    for r in 0..tr {
        for t in 0..tt {
            src.push_str(&format!("    var a{r}{t}: f32 = 0.0;\n"));
        }
    }
    src.push_str(&format!(
        r#"
    var s: u32 = 0u;
    loop {{
        if (s >= n_super) {{ break; }}
        // Stage the rows, unpacked: one (row, half, word) per item — the two
        // `ql` words and one `qh` word that make four output words, one for
        // each of the half's sub-blocks.
        var i: u32 = local;
        loop {{
            if (i >= {row_items}u) {{ break; }}
            let r = i / 16u;
            let h = (i / 8u) % 2u;
            let w = i % 8u;
            let byte0 = (row0 + r) * params.row_bytes + s * 210u;
            let ql_a = block_word(byte0, h * 16u + w);
            let ql_b = block_word(byte0, h * 16u + 8u + w);
            let qh = block_word(byte0, 32u + h * 8u + w);
            let c = w % 4u;
            let v = r * 17u + 8u * h + w / 4u;
            wt[v][c] = (ql_a & 0x0F0F0F0Fu) | ((qh & 0x03030303u) << 4u);
            wt[v + 2u][c] = (ql_b & 0x0F0F0F0Fu) | (((qh >> 2u) & 0x03030303u) << 4u);
            wt[v + 4u][c] = ((ql_a >> 4u) & 0x0F0F0F0Fu) | (((qh >> 4u) & 0x03030303u) << 4u);
            wt[v + 6u][c] = ((ql_b >> 4u) & 0x0F0F0F0Fu) | (((qh >> 6u) & 0x03030303u) << 4u);
            i = i + 128u;
        }}
        // The scales: sixteen per row, four to a word, times d.
        i = local;
        loop {{
            if (i >= {scale_items}u) {{ break; }}
            let r = i / 4u;
            let k4 = i % 4u;
            let byte0 = (row0 + r) * params.row_bytes + s * 210u;
            let d = f16_to_f32(block_word(byte0, 52u));
            let sc_word = block_word(byte0, 48u + k4);
            sm[r * 17u + 4u * k4] = d * f32(i32(sc_word << 24u) >> 24u);
            sm[r * 17u + 4u * k4 + 1u] = d * f32(i32((sc_word >> 8u) << 24u) >> 24u);
            sm[r * 17u + 4u * k4 + 2u] = d * f32(i32((sc_word >> 16u) << 24u) >> 24u);
            sm[r * 17u + 4u * k4 + 3u] = d * f32(i32(sc_word >> 24u) << 24u >> 24u);
            i = i + 128u;
        }}
        i = local;
        loop {{
            if (i >= {xt_loads}u) {{ break; }}
            let t = i / 24u;
            let c = i % 24u;
            xt[t * 25u + c] = q8x[((tok0 + t) * n_super + s) * 24u + c];
            i = i + 128u;
        }}
        workgroupBarrier();
"#,
        row_items = MMQ_TILE_ROWS * 16,
        scale_items = MMQ_TILE_ROWS * 4,
        xt_loads = tile_tokens * 24,
    ));
    let stub = crate::engine::env::flag_on("ORANGU_MMQ_STUB");
    for b in 0..8usize {
        src.push_str(&format!("        {{\n        // sub-block {b}\n"));
        for t in 0..tt {
            src.push_str(&format!(
                "        let x{t}a = xt[xbase + {}u];\n        let x{t}b = xt[xbase + {}u];\n        let ds{t} = xt[xbase + {}u];\n        let db{t} = bitcast<f32>(ds{t}.x);\n        let lo{t} = 32.0 * db{t} * f32(bitcast<i32>(ds{t}.y));\n        let hi{t} = 32.0 * db{t} * f32(bitcast<i32>(ds{t}.z));\n",
                t * 16 * 25 + 8 + 2 * b,
                t * 16 * 25 + 9 + 2 * b,
                t * 16 * 25 + b,
            ));
        }
        for r in 0..tr {
            src.push_str(&format!(
                "        {{\n        let wa = wt[wbase + {}u];\n        let wb = wt[wbase + {}u];\n        let s0 = sm[sbase + {}u];\n        let s1 = sm[sbase + {}u];\n",
                r * 8 * 17 + 2 * b,
                r * 8 * 17 + 2 * b + 1,
                r * 8 * 17 + 2 * b,
                r * 8 * 17 + 2 * b + 1,
            ));
            for t in 0..tt {
                let (dlo, dhi) = if stub {
                    (
                        format!("i32((wa.x ^ x{t}a.x) & 7u)"),
                        format!("i32((wb.x ^ x{t}b.x) & 7u)"),
                    )
                } else {
                    (
                        format!(
                            "dot4I8Packed(wa.x, x{t}a.x) + dot4I8Packed(wa.y, x{t}a.y) + dot4I8Packed(wa.z, x{t}a.z) + dot4I8Packed(wa.w, x{t}a.w)"
                        ),
                        format!(
                            "dot4I8Packed(wb.x, x{t}b.x) + dot4I8Packed(wb.y, x{t}b.y) + dot4I8Packed(wb.z, x{t}b.z) + dot4I8Packed(wb.w, x{t}b.w)"
                        ),
                    )
                };
                src.push_str(&format!(
                    "        a{r}{t} = a{r}{t} + fma(s0, fma(db{t}, f32({dlo}), -lo{t}), s1 * fma(db{t}, f32({dhi}), -hi{t}));\n"
                ));
            }
            src.push_str("        }\n");
        }
        src.push_str("        }\n");
    }
    src.push_str("        workgroupBarrier();\n        s = s + 1u;\n    }\n");
    for t in 0..tt {
        src.push_str(&format!(
            "    if (tok0 + {}u + tt < params.n_tokens) {{\n",
            t * 16
        ));
        for r in 0..tr {
            src.push_str(&format!(
                "        y[(tok0 + {}u + tt) * params.out_dim + row0 + {}u + tr] = a{r}{t};\n",
                t * 16,
                r * 8
            ));
        }
        src.push_str("    }\n");
    }
    src.push_str("}\n");
    src
}

pub fn shader_source_reduce_q4k_mmvq(subgroup: bool) -> String {
    let subgroup_params = if subgroup {
        "@builtin(subgroup_invocation_id) sg_lane: u32,\n    @builtin(subgroup_id) sg_id: u32,\n    @builtin(num_subgroups) n_sg: u32,"
    } else {
        ""
    };
    let reduce = if subgroup {
        "    let total = subgroupAdd(partial);\n    if (sg_lane == 0u) {\n        y[t * params.out_dim + o] = total;\n    }\n"
    } else {
        "    psum[tid] = partial;\n    workgroupBarrier();\n    var stride: u32 = 16u;\n    loop {\n        if (stride == 0u) { break; }\n        if (tid < stride) { psum[tid] = psum[tid] + psum[tid + stride]; }\n        workgroupBarrier();\n        stride = stride / 2u;\n    }\n    if (tid == 0u) {\n        y[t * params.out_dim + o] = psum[0];\n    }\n"
    };
    format!(
        r#"
struct Meta {{
    in_dim: u32,
    out_dim: u32,
    n_tokens: u32,
    row_bytes: u32,
}}

@group(0) @binding(0) var<storage, read> weights: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> q8x: array<u32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> params: Meta;

fn f16_to_f32(bits: u32) -> f32 {{
    return unpack2x16float(bits & 0xFFFFu).x;
}}
fn vec3_word(v: vec3<u32>, i: u32) -> u32 {{
    if (i == 0u) {{ return v.x; }}
    if (i == 1u) {{ return v.y; }}
    return v.z;
}}
// ggml get_scale_min_k4, from the already-loaded 12-byte scales (header.yzw).
fn get_scale_min_k4_v4(scales: vec3<u32>, j: u32) -> vec2<u32> {{
    if (j < 4u) {{
        let qj = (vec3_word(scales, j / 4u) >> (8u * (j % 4u))) & 0xFFu;
        let qj4 = (vec3_word(scales, (j + 4u) / 4u) >> (8u * ((j + 4u) % 4u))) & 0xFFu;
        return vec2<u32>(qj & 63u, qj4 & 63u);
    }}
    let qj = (vec3_word(scales, j / 4u) >> (8u * (j % 4u))) & 0xFFu;
    let qj4 = (vec3_word(scales, (j + 4u) / 4u) >> (8u * ((j + 4u) % 4u))) & 0xFFu;
    let qjm4 = (vec3_word(scales, (j - 4u) / 4u) >> (8u * ((j - 4u) % 4u))) & 0xFFu;
    let sc = (qj4 & 0xFu) | ((qjm4 >> 6u) << 4u);
    let m = (qj4 >> 4u) | ((qj >> 6u) << 4u);
    return vec2<u32>(sc, m);
}}

var<workgroup> psum: array<f32, 32>;

@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    {subgroup_params}
) {{
    let flat = wid.x + wid.y * nwg.x + wid.z * nwg.x * nwg.y;
    if (flat >= params.out_dim * params.n_tokens) {{
        return;
    }}
    let o = flat / params.n_tokens;
    let t = flat % params.n_tokens;
    let tid = lid.x;

    let n_sub = params.in_dim / 32u;          // 32-element sub-blocks per row
    let q8_row_base = t * n_sub * 10u;        // 10 u32 per q8 block

    var partial: f32 = 0.0;
    var sb: u32 = tid;
    loop {{
        if (sb >= n_sub) {{
            break;
        }}
        let super_blk = sb / 8u;
        let b = sb % 8u;
        let block_byte_off = o * params.row_bytes + super_blk * 144u;
        let header = weights[block_byte_off / 16u];
        let d = f16_to_f32(header.x & 0xFFFFu);
        let dmin = f16_to_f32(header.x >> 16u);
        let sm = get_scale_min_k4_v4(vec3<u32>(header.y, header.z, header.w), b);
        let scale = f32(sm.x);
        let mn = f32(sm.y);

        // This sub-block's 32 nibbles: qs bytes 32*(b/2)..+32, low half for even
        // b, high half for odd b (ggml's contiguous-per-sub-block dequant order).
        let qbyte = block_byte_off + 16u + 32u * (b / 2u);
        let is_high = (b & 1u) == 1u;

        let q8base = q8_row_base + sb * 10u;
        let d_b = bitcast<f32>(q8x[q8base]);
        let sumq = bitcast<i32>(q8x[q8base + 1u]);

        // The sub-block's 32 nibbles are 8 contiguous `u32` = two `vec4<u32>`.
        // `qbyte` is 16-byte aligned (`row_bytes`, 144-byte super-blocks, the
        // +16 header and 32*(b/2) qs offset are all multiples of 16), so read
        // them as two direct vec4 loads and mask both nibble halves branchlessly
        // — no per-word divide/modulo/select indexing.
        let vidx = qbyte / 16u;
        let m4 = vec4<u32>(0x0F0F0F0Fu);
        let f4 = vec4<u32>(4u);
        let w0 = weights[vidx];
        let w1 = weights[vidx + 1u];
        let wv0 = select(w0 & m4, (w0 >> f4) & m4, is_high);
        let wv1 = select(w1 & m4, (w1 >> f4) & m4, is_high);

        var idot: i32 = 0;
        idot = idot + dot4I8Packed(wv0.x, q8x[q8base + 2u]);
        idot = idot + dot4I8Packed(wv0.y, q8x[q8base + 3u]);
        idot = idot + dot4I8Packed(wv0.z, q8x[q8base + 4u]);
        idot = idot + dot4I8Packed(wv0.w, q8x[q8base + 5u]);
        idot = idot + dot4I8Packed(wv1.x, q8x[q8base + 6u]);
        idot = idot + dot4I8Packed(wv1.y, q8x[q8base + 7u]);
        idot = idot + dot4I8Packed(wv1.z, q8x[q8base + 8u]);
        idot = idot + dot4I8Packed(wv1.w, q8x[q8base + 9u]);

        partial = partial + d * scale * d_b * f32(idot) - dmin * mn * d_b * f32(sumq);
        sb = sb + 32u;
    }}

{reduce}}}
"#
    )
}

/// Prelude + per-super-block dot for the **light** `Q4_K` decode kernel
/// ([`shader_source_reduce_q4k_light`]) — a WGSL port of llama.cpp's
/// `mul_mat_vec_q4_k.comp`. The activation `x` is rebound as
/// `array<vec4<f32>>` (binding shape unchanged — storage is element-type
/// agnostic), so B is read as `vec4` and each thread's dot is four `dot()`s.
/// A **profiling probe, not a correct kernel**: `Q4K_LIGHT_PRELUDE` with every
/// load kept and the arithmetic between them removed.
///
/// P4 has ruled out the memory system for the decode GEMV — the hardware serves
/// its exact access shape at 170 GB/s against the 48 GB/s it achieves — leaving
/// two candidates that no static measurement separates: the dequant/dot ALU,
/// and the fixed per-workgroup cost (launch, subgroup reduction, store) spread
/// over one output row's worth of work.
///
/// This separates them. The seven loads per block are identical — same header
/// `vec4`, same two `read_word_v4`, same four activation `vec4`s, same
/// addresses, so the memory traffic and the workgroup count are untouched — and
/// roughly 140 VALU ops of dequantization and dot product become about ten.
/// If throughput moves toward the 170 GB/s the pattern can sustain, the ALU is
/// the cost; if it stays near 48, the per-workgroup overhead is.
///
/// The output is deliberately meaningless. `ORANGU_GEMV_STUB=1`, and every
/// cross-check fails under it by design.
const Q4K_LIGHT_STUB_SB: &str = r#"
fn sb(block_byte_off: u32, x_block_elem: u32, y_offset: u32, q_offset: u32, v_im: u32) -> f32 {
    let header = weights[block_byte_off / 16u];
    let qs0 = read_word_v4(block_byte_off + 16u + q_offset);
    let qs64 = read_word_v4(block_byte_off + 16u + q_offset + 64u);
    let y1 = (x_block_elem + y_offset) / 4u;
    let y2 = (x_block_elem + y_offset + 128u) / 4u;
    // Every component of every load is consumed, or the compiler narrows the
    // `vec4` fetches to scalars and the probe stops reading what the real
    // kernel reads. The load census is asserted to match before it is trusted.
    let a1 = xv[y1];
    let a2 = xv[y1 + 8u];
    let a3 = xv[y2];
    let a4 = xv[y2 + 8u];
    let h = (header.x ^ header.y ^ header.z ^ header.w ^ qs0 ^ qs64) & 0xFFu;
    return f32(h)
        + (a1.x + a1.y + a1.z + a1.w)
        + (a2.x + a2.y + a2.z + a2.w)
        + (a3.x + a3.y + a3.z + a3.w)
        + (a4.x + a4.y + a4.z + a4.w);
}
"#;

const Q4K_LIGHT_PRELUDE: &str = r#"
struct Meta {
    in_dim: u32,
    out_dim: u32,
    n_tokens: u32,
    row_bytes: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> xv: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> params: Meta;

fn f16_to_f32(bits: u32) -> f32 {
    return unpack2x16float(bits & 0xFFFFu).x;
}

// Dynamic vector index — naga lowers `v[i]` to a branchless select, unlike the
// old 3-`if` chain that diverged across a subgroup's lanes (each owns a
// different component).
fn vec4_word(v: vec4<u32>, i: u32) -> u32 {
    return v[i];
}

// 4-byte-aligned u32 read from the vec4<u32>-bound weight buffer.
fn read_word_v4(byte_offset: u32) -> u32 {
    return weights[byte_offset / 16u][(byte_offset % 16u) / 4u];
}

// Unpack the four bytes of a u32 into a vec4<f32>.
fn unpack4_f(w: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(w & 0xFFu),
        f32((w >> 8u) & 0xFFu),
        f32((w >> 16u) & 0xFFu),
        f32((w >> 24u) & 0xFFu),
    );
}

// One of the six little-endian u16s of the 12-byte Q4_K scales region
// (bytes 4..16 of the block, i.e. header.y/.z/.w). k in 0..5.
fn scale_u16(header: vec4<u32>, k: u32) -> u32 {
    let word = vec4_word(header, (k / 2u) + 1u);
    return select(word >> 16u, word & 0xFFFFu, (k & 1u) == 0u);
}

// Contribution of super-block `i` of one output row to that row's dot,
// computed by the 16 threads that own this block. `block_byte_off` is the
// byte offset of the block in `weights` (row_base + i*144); `x_block_elem` is
// the element offset of the block's activations in `xv` (t*in_dim + i*256).
// Faithful port of llama.cpp's `calc_superblock` (one row, one column).
fn sb(block_byte_off: u32, x_block_elem: u32, y_offset: u32, q_offset: u32, v_im: u32) -> f32 {
    let header = weights[block_byte_off / 16u];
    let d = f16_to_f32(header.x & 0xFFFFu);
    let dmin = f16_to_f32(header.x >> 16u);

    // Scales/mins — llama.cpp's packed-u16 unpacking of get_scale_min_k4.
    let scale0_u32 = scale_u16(header, v_im);
    let scale4_u32 = scale_u16(header, v_im + 2u);
    let scale8_u32 = scale_u16(header, v_im + 4u);
    let scale_0_4_l = (scale4_u32 << 16u) | scale0_u32;
    let scale_0_4_h = (scale_0_4_l & 0xC0C0C0C0u) >> 2u;
    let sc_l = unpack4_f(scale_0_4_l & 0x3F3F3F3Fu);
    let sc_8 = unpack4_f((((scale8_u32 << 12u) | scale8_u32) & 0x0F0F0F0Fu) | scale_0_4_h);
    let sc0 = sc_l.x; let sc1 = sc_l.y; let sc2 = sc_l.z; let sc3 = sc_l.w;
    let sc4 = sc_8.x; let sc5 = sc_8.y; let sc6 = sc_8.z; let sc7 = sc_8.w;

    // 16 weight nibbles: two packed qs words (this thread's, and its +64 pair).
    let qs0 = read_word_v4(block_byte_off + 16u + q_offset);
    let qs64 = read_word_v4(block_byte_off + 16u + q_offset + 64u);
    let y1 = (x_block_elem + y_offset) / 4u;
    let y2 = (x_block_elem + y_offset + 128u) / 4u;
    let ones = vec4<f32>(1.0, 1.0, 1.0, 1.0);

    // Accumulate the four (weight, activation) vec4 groups one at a time so
    // only a single vec4 pair is live at once — the whole-superblock scale/min
    // sums stay in two scalars rather than four hoisted vec4s each. Keeps the
    // kernel's peak register footprint down (occupancy is the goal here).
    var main_sum: f32 = 0.0;
    var min_sum: f32 = 0.0;
    {
        let by = xv[y1];
        let q = unpack4_f(qs0 & 0x0F0F0F0Fu); // elements 0..3
        main_sum = main_sum + sc0 * dot(by, q);
        min_sum = min_sum + sc2 * dot(by, ones);
    }
    {
        let by = xv[y1 + 8u];
        let q = unpack4_f((qs0 >> 4u) & 0x0F0F0F0Fu); // 4..7
        main_sum = main_sum + sc1 * dot(by, q);
        min_sum = min_sum + sc3 * dot(by, ones);
    }
    {
        let by = xv[y2];
        let q = unpack4_f(qs64 & 0x0F0F0F0Fu); // 8..11
        main_sum = main_sum + sc4 * dot(by, q);
        min_sum = min_sum + sc6 * dot(by, ones);
    }
    {
        let by = xv[y2 + 8u];
        let q = unpack4_f((qs64 >> 4u) & 0x0F0F0F0Fu); // 12..15
        main_sum = main_sum + sc5 * dot(by, q);
        min_sum = min_sum + sc7 * dot(by, ones);
    }
    return d * main_sum - dmin * min_sum;
}

// One row's super-block against activations already in registers — the
// same arithmetic as `sb`, with the four `xv` loads hoisted to the caller so
// a workgroup that owns several rows loads each activation once per
// super-block rather than once per row.
fn sb_act(block_byte_off: u32, q_offset: u32, v_im: u32, by0: vec4<f32>, by1: vec4<f32>, by2: vec4<f32>, by3: vec4<f32>) -> f32 {
    let header = weights[block_byte_off / 16u];
    let d = f16_to_f32(header.x & 0xFFFFu);
    let dmin = f16_to_f32(header.x >> 16u);

    let scale0_u32 = scale_u16(header, v_im);
    let scale4_u32 = scale_u16(header, v_im + 2u);
    let scale8_u32 = scale_u16(header, v_im + 4u);
    let scale_0_4_l = (scale4_u32 << 16u) | scale0_u32;
    let scale_0_4_h = (scale_0_4_l & 0xC0C0C0C0u) >> 2u;
    let sc_l = unpack4_f(scale_0_4_l & 0x3F3F3F3Fu);
    let sc_8 = unpack4_f((((scale8_u32 << 12u) | scale8_u32) & 0x0F0F0F0Fu) | scale_0_4_h);

    let qs0 = read_word_v4(block_byte_off + 16u + q_offset);
    let qs64 = read_word_v4(block_byte_off + 16u + q_offset + 64u);
    let ones = vec4<f32>(1.0, 1.0, 1.0, 1.0);

    var main_sum: f32 = sc_l.x * dot(by0, unpack4_f(qs0 & 0x0F0F0F0Fu));
    main_sum = main_sum + sc_l.y * dot(by1, unpack4_f((qs0 >> 4u) & 0x0F0F0F0Fu));
    main_sum = main_sum + sc_8.x * dot(by2, unpack4_f(qs64 & 0x0F0F0F0Fu));
    main_sum = main_sum + sc_8.y * dot(by3, unpack4_f((qs64 >> 4u) & 0x0F0F0F0Fu));
    var min_sum: f32 = sc_l.z * dot(by0, ones);
    min_sum = min_sum + sc_l.w * dot(by1, ones);
    min_sum = min_sum + sc_8.z * dot(by2, ones);
    min_sum = min_sum + sc_8.w * dot(by3, ones);
    return d * main_sum - dmin * min_sum;
}
"#;

/// The `Q5_K` twin of [`Q4K_LIGHT_PRELUDE`].
///
/// `Q5_K` is `Q4_K` plus a fifth bit per weight: the block grows from 144 to
/// 176 bytes, a 32-byte `qh` plane is inserted between the scales and the
/// quants (so `qs` starts at 48 rather than 16), and each 4-bit nibble gains
/// `+16` when its `qh` bit is set. Everything else — the `d`/`dmin` header, the
/// 12-byte packed scale/min region and its `get_scale_min_k4` unpacking, the
/// thread-to-element mapping, the four `vec4` activation reads — is identical,
/// so this is that kernel with one extra `u32` load and four masked adds.
///
/// # Why not llama.cpp's own `l0`
///
/// The reference `mul_mat_vec_q5_k.comp` uses `l0 = 4*ir + 2*v_in`, which puts
/// a thread's elements at `{0,1,16,17,32,33,48,49}` — eight *pairs*, read there
/// as `vec2`. This engine binds activations as `array<vec4<f32>>`, and those
/// offsets are not 4-aligned, so the `vec2` layout would cost eight scalar
/// loads where four `vec4` loads do. `Q4_K`'s own `l0 = 4*(2*ir + v_in)` gives
/// four contiguous runs of four instead, which is exactly one `vec4` each. The
/// `qs`/`qh` bit extraction is rewritten to match that mapping rather than the
/// reference's; the arithmetic per element is the same either way.
const Q5K_LIGHT_PRELUDE: &str = r#"
struct Meta {
    in_dim: u32,
    out_dim: u32,
    n_tokens: u32,
    row_bytes: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> xv: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> params: Meta;

fn f16_to_f32(bits: u32) -> f32 {
    return unpack2x16float(bits & 0xFFFFu).x;
}

fn vec4_word(v: vec4<u32>, i: u32) -> u32 {
    return v[i];
}

// 4-byte-aligned u32 read from the vec4<u32>-bound weight buffer.
fn read_word_v4(byte_offset: u32) -> u32 {
    return weights[byte_offset / 16u][(byte_offset % 16u) / 4u];
}

fn unpack4_f(w: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(w & 0xFFu),
        f32((w >> 8u) & 0xFFu),
        f32((w >> 16u) & 0xFFu),
        f32((w >> 24u) & 0xFFu),
    );
}

// One of the six little-endian u16s of the 12-byte scales region (bytes 4..16
// of the block, i.e. header.y/.z/.w). Identical to Q4_K's — the scale/min
// encoding does not change between the two formats.
fn scale_u16(header: vec4<u32>, k: u32) -> u32 {
    let word = vec4_word(header, (k / 2u) + 1u);
    return select(word >> 16u, word & 0xFFFFu, (k & 1u) == 0u);
}

// Contribution of super-block `i` of one output row, computed by the 16
// threads that own it. `block_byte_off` is the block's byte offset in
// `weights` (row_base + i*176); `x_block_elem` is the element offset of the
// block's activations in `xv` (t*in_dim + i*256).
fn sb(block_byte_off: u32, x_block_elem: u32, y_offset: u32, q_offset: u32, v_im: u32, l0: u32) -> f32 {
    let header = weights[block_byte_off / 16u];
    let d = f16_to_f32(header.x & 0xFFFFu);
    let dmin = f16_to_f32(header.x >> 16u);

    let scale0_u32 = scale_u16(header, v_im);
    let scale4_u32 = scale_u16(header, v_im + 2u);
    let scale8_u32 = scale_u16(header, v_im + 4u);
    let scale_0_4_l = (scale4_u32 << 16u) | scale0_u32;
    let scale_0_4_h = (scale_0_4_l & 0xC0C0C0C0u) >> 2u;
    let sc_l = unpack4_f(scale_0_4_l & 0x3F3F3F3Fu);
    let sc_8 = unpack4_f((((scale8_u32 << 12u) | scale8_u32) & 0x0F0F0F0Fu) | scale_0_4_h);
    let sc0 = sc_l.x; let sc1 = sc_l.y; let sc2 = sc_l.z; let sc3 = sc_l.w;
    let sc4 = sc_8.x; let sc5 = sc_8.y; let sc6 = sc_8.z; let sc7 = sc_8.w;

    // `qs` starts 32 bytes later than in Q4_K: the `qh` plane sits between the
    // scales and the quants.
    let qs0 = read_word_v4(block_byte_off + 48u + q_offset);
    let qs64 = read_word_v4(block_byte_off + 48u + q_offset + 64u);
    // The four `qh` bytes covering this thread's four element positions. `l0`
    // is 4-aligned (see the type doc), so this is one aligned word.
    let qh = read_word_v4(block_byte_off + 16u + l0);

    // Which bit of each `qh` byte carries the fifth bit depends only on which
    // 64-element quarter the element sits in: quarter `v_im` uses bits
    // `2*v_im` (low 32) and `2*v_im + 1` (high 32), quarter `v_im + 2` — the
    // `+128` half — uses bits `2*v_im + 4` and `2*v_im + 5`. Shifted down to
    // bit 0 of each byte and scaled to 16, this is a masked add per group.
    let qhs = qh >> (2u * v_im);
    let hi0 = (qhs & 0x01010101u) << 4u;         // elements +0..3
    let hi32 = (qhs & 0x02020202u) << 3u;        // elements +32..35
    let hi128 = qhs & 0x10101010u;               // elements +128..131
    let hi160 = (qhs & 0x20202020u) >> 1u;       // elements +160..163

    let y1 = (x_block_elem + y_offset) / 4u;
    let y2 = (x_block_elem + y_offset + 128u) / 4u;
    let ones = vec4<f32>(1.0, 1.0, 1.0, 1.0);

    var main_sum: f32 = 0.0;
    var min_sum: f32 = 0.0;
    {
        let by = xv[y1];
        let q = unpack4_f((qs0 & 0x0F0F0F0Fu) + hi0);
        main_sum = main_sum + sc0 * dot(by, q);
        min_sum = min_sum + sc2 * dot(by, ones);
    }
    {
        let by = xv[y1 + 8u];
        let q = unpack4_f(((qs0 >> 4u) & 0x0F0F0F0Fu) + hi32);
        main_sum = main_sum + sc1 * dot(by, q);
        min_sum = min_sum + sc3 * dot(by, ones);
    }
    {
        let by = xv[y2];
        let q = unpack4_f((qs64 & 0x0F0F0F0Fu) + hi128);
        main_sum = main_sum + sc4 * dot(by, q);
        min_sum = min_sum + sc6 * dot(by, ones);
    }
    {
        let by = xv[y2 + 8u];
        let q = unpack4_f(((qs64 >> 4u) & 0x0F0F0F0Fu) + hi160);
        main_sum = main_sum + sc5 * dot(by, q);
        min_sum = min_sum + sc7 * dot(by, ones);
    }
    return d * main_sum - dmin * min_sum;
}
"#;

/// The **light** `Q5_K` decode matmul-vec kernel — [`Q5K_LIGHT_PRELUDE`]
/// wrapped in the same 32-thread, `n_rows`-batched dispatch as
/// [`shader_source_reduce_q4k_light`].
///
/// `Q5_K` was the only K-quant with no kernel of its own: it fell through to
/// the generic block-unroll, and it showed. Measured on one model across five
/// quantizations, as effective weight-streaming bandwidth (file size ×
/// decode tok/s), `Q4_K` on its light kernel and `Q8_0` on the *generic*
/// reduce both reached ~50 GiB/s while `Q5_K` managed 33.6 — a third short of
/// formats either side of it, which is not a bit-width effect.
///
/// Not bit-identical against `CpuBackend` (it reorders the float adds, exactly
/// as the `Q4_K` twin does), so it cross-checks within tolerance.
pub fn shader_source_reduce_q5k_light(n_rows: usize, subgroup: bool) -> String {
    let mut s = String::from(Q5K_LIGHT_PRELUDE);
    s.push_str(&format!(
        "\nvar<workgroup> psums: array<f32, {}>;\n\n",
        n_rows * 32
    ));
    s.push_str("@compute @workgroup_size(32)\nfn main(\n    @builtin(workgroup_id) wid: vec3<u32>,\n    @builtin(local_invocation_id) lid: vec3<u32>,\n    @builtin(num_workgroups) nwg: vec3<u32>,");
    s.push_str(subgroup_entry_params(subgroup));
    s.push_str("\n) {\n");
    s.push_str(&format!(
        "    let n_row_groups = (params.out_dim + {}u) / {n_rows}u;\n",
        n_rows - 1
    ));
    s.push_str("    let flat = wid.x + wid.y * nwg.x + wid.z * nwg.x * nwg.y;\n    if (flat >= n_row_groups * params.n_tokens) {\n        return;\n    }\n");
    s.push_str("    let rg = flat / params.n_tokens;\n    let t = flat % params.n_tokens;\n");
    s.push_str(&format!("    let o0 = rg * {n_rows}u;\n"));
    for i in 1..n_rows {
        s.push_str(&format!("    let o{i} = o0 + {i}u;\n"));
    }
    // Identical thread mapping to the Q4_K light kernel — see
    // `Q5K_LIGHT_PRELUDE`'s doc for why this rather than the reference's.
    s.push_str("    let tid = lid.x;\n    let itid = tid % 16u;\n    let ix = tid / 16u;\n");
    s.push_str("    let il = itid / 4u;\n    let ir = itid % 4u;\n    let v_im = il / 2u;\n    let v_in = il % 2u;\n    let l0 = 4u * (2u * ir + v_in);\n    let q_offset = 32u * v_im + l0;\n    let y_offset = 64u * v_im + l0;\n");
    s.push_str("    let num_blocks = params.in_dim / 256u;\n    let x_tok = t * params.in_dim;\n");
    for i in 0..n_rows {
        s.push_str(&format!("    var acc{i}: f32 = 0.0;\n"));
    }
    // 176-byte blocks, two super-blocks per 32-thread workgroup iteration.
    s.push_str("    var i: u32 = ix;\n    loop {\n        if (i >= num_blocks) {\n            break;\n        }\n        let xrb = x_tok + i * 256u;\n");
    for r in 0..n_rows {
        s.push_str(&format!(
            "        acc{r} = acc{r} + sb(o{r} * params.row_bytes + i * 176u, xrb, y_offset, q_offset, v_im, l0);\n"
        ));
    }
    s.push_str("        i = i + 2u;\n    }\n");
    for r in 0..n_rows {
        if subgroup {
            s.push_str(&format!(
                "    let sg{r} = subgroupAdd(acc{r});\n    if (sg_lane == 0u && o{r} < params.out_dim) {{\n        y[t * params.out_dim + o{r}] = sg{r};\n    }}\n"
            ));
        } else {
            s.push_str(&format!("    psums[tid * {n_rows}u + {r}u] = acc{r};\n"));
        }
    }
    if !subgroup {
        s.push_str("    workgroupBarrier();\n    var stride: u32 = 16u;\n    loop {\n        if (stride == 0u) {\n            break;\n        }\n        if (tid < stride) {\n");
        for r in 0..n_rows {
            s.push_str(&format!(
                "            psums[tid * {n_rows}u + {r}u] = psums[tid * {n_rows}u + {r}u] + psums[(tid + stride) * {n_rows}u + {r}u];\n"
            ));
        }
        s.push_str("        }\n        workgroupBarrier();\n        stride = stride / 2u;\n    }\n    if (tid == 0u) {\n");
        for r in 0..n_rows {
            s.push_str(&format!(
                "        if (o{r} < params.out_dim) {{\n            y[t * params.out_dim + o{r}] = psums[{r}u];\n        }}\n"
            ));
        }
        s.push_str("    }\n");
    }
    s.push_str("}\n");
    s
}

/// The `Q6_K` twin of [`Q4K_LIGHT_PRELUDE`] / [`Q5K_LIGHT_PRELUDE`].
///
/// `Q6_K`'s block is a different shape from the other two K-quants: 210 bytes
/// of `ql` (128) + `qh` (64) + sixteen **signed 8-bit** scales (16) + `d` (2),
/// with no `dmin` and so no min-sum term at all. Two bits per weight come from
/// the `qh` plane rather than one, and the scale is per-16-elements rather than
/// per-32.
///
/// # Alignment
///
/// 210 is not a multiple of 4, so a block's fields land at an offset whose
/// alignment alternates with the block index — unlike `Q4_K`'s 144 and
/// `Q5_K`'s 176, where every block starts 16-byte aligned and every field read
/// is naturally aligned. Every weight read here therefore goes through
/// `read_word_unaligned_v4`, which costs a second aligned load on the odd
/// blocks. That is still far cheaper than what it replaces: the dual kernel
/// reads `ql`/`qh` a **byte** at a time and re-reads a scale byte *per
/// element*, ~28 byte-granular loads per 16 elements against this kernel's
/// three `u32`s plus one for the scales.
const Q6K_LIGHT_PRELUDE: &str = r#"
struct Meta {
    in_dim: u32,
    out_dim: u32,
    n_tokens: u32,
    row_bytes: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> xv: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> params: Meta;

fn f16_to_f32(bits: u32) -> f32 {
    return unpack2x16float(bits & 0xFFFFu).x;
}

fn read_word_v4(byte_offset: u32) -> u32 {
    return weights[byte_offset / 16u][(byte_offset % 16u) / 4u];
}

// See the type doc: Q6_K blocks are 210 bytes, so nothing here is reliably
// 4-byte aligned.
fn read_word_u(byte_offset: u32) -> u32 {
    let shift = (byte_offset % 4u) * 8u;
    let aligned = byte_offset - (byte_offset % 4u);
    if (shift == 0u) {
        return read_word_v4(aligned);
    }
    let lo = read_word_v4(aligned);
    let hi = read_word_v4(aligned + 4u);
    return (lo >> shift) | (hi << (32u - shift));
}

fn read_byte_u(byte_offset: u32) -> u32 {
    let word = read_word_v4(byte_offset - (byte_offset % 4u));
    return (word >> (8u * (byte_offset % 4u))) & 0xFFu;
}

// The four bytes of a u32 as floats, each biased by -32: Q6_K stores its
// 6-bit weights unsigned and the dequant is `d * scale * (q - 32)`.
fn unpack4_f_bias32(w: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(w & 0xFFu) - 32.0,
        f32((w >> 8u) & 0xFFu) - 32.0,
        f32((w >> 16u) & 0xFFu) - 32.0,
        f32((w >> 24u) & 0xFFu) - 32.0,
    );
}

// One of the sixteen **signed** 8-bit scales.
fn scale_i8(sc_off: u32, i: u32) -> f32 {
    let b = read_byte_u(sc_off + i);
    return f32(i32(b << 24u) >> 24u);
}

// Contribution of super-block `i` of one output row, computed by the 16
// threads that own it. Faithful to llama.cpp's `mul_mat_vec_q6_k.comp`
// `calc_superblock`, minus its shared-memory scale cache: a thread needs only
// four of the sixteen scales, so it reads those four directly rather than
// staging all sixteen through LDS and a barrier.
fn sb(block_byte_off: u32, x_block_elem: u32, ql_offset: u32, qh_offset: u32,
      s_offset: u32, y_offset: u32) -> f32 {
    let ql_off = block_byte_off;
    let qh_off = block_byte_off + 128u;
    let sc_off = block_byte_off + 192u;
    let d = f16_to_f32(read_word_u(block_byte_off + 208u) & 0xFFFFu);

    let ql0 = read_word_u(ql_off + ql_offset);
    let ql32 = read_word_u(ql_off + ql_offset + 32u);
    let qh = read_word_u(qh_off + qh_offset);

    // Two bits per weight, four weights per qh byte: bits 0-1 feed the
    // elements at +0, bits 2-3 those at +32, bits 4-5 at +64, bits 6-7 at
    // +96 — shifted into position 4 of the 6-bit quant.
    let q0 = (ql0 & 0x0F0F0F0Fu) | ((qh & 0x03030303u) << 4u);
    let q1 = (ql32 & 0x0F0F0F0Fu) | ((qh & 0x0C0C0C0Cu) << 2u);
    let q2 = ((ql0 >> 4u) & 0x0F0F0F0Fu) | (qh & 0x30303030u);
    let q3 = ((ql32 >> 4u) & 0x0F0F0F0Fu) | ((qh & 0xC0C0C0C0u) >> 2u);

    let y0 = (x_block_elem + y_offset) / 4u;
    var sum: f32 = 0.0;
    sum = sum + scale_i8(sc_off, s_offset) * dot(xv[y0], unpack4_f_bias32(q0));
    sum = sum + scale_i8(sc_off, s_offset + 2u) * dot(xv[y0 + 8u], unpack4_f_bias32(q1));
    sum = sum + scale_i8(sc_off, s_offset + 4u) * dot(xv[y0 + 16u], unpack4_f_bias32(q2));
    sum = sum + scale_i8(sc_off, s_offset + 6u) * dot(xv[y0 + 24u], unpack4_f_bias32(q3));
    return d * sum;
}
"#;

/// [`Q6K_LIGHT_PRELUDE`] with the block read as **words**, the unaligned
/// case resolved once per thread rather than once per load.
///
/// The prelude above reads every field through a `vec4<u32>` load and a
/// per-load alignment test. Both are avoidable: a thread's block offsets are
/// all congruent modulo four — it walks blocks two at a time, and two blocks
/// are 420 bytes — so its misalignment is one of two values fixed for the
/// whole loop, and the fields it needs are all four-byte spans. So the shift
/// is computed once, each aligned field is one `u32` load (two when the
/// thread is the misaligned one), the four scale bytes come out of three
/// consecutive words, and `d` out of one. Roughly half the loads of the
/// prelude above, at a quarter of the width each. `ORANGU_Q6K_V4=1` keeps
/// the old prelude, as the control arm of a sweep.
const Q6K_LIGHT_PRELUDE_WORDS: &str = r#"
struct Meta {
    in_dim: u32,
    out_dim: u32,
    n_tokens: u32,
    row_bytes: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> xv: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> params: Meta;

fn f16_to_f32(bits: u32) -> f32 {
    return unpack2x16float(bits & 0xFFFFu).x;
}

// A four-byte field at `byte_offset`, whose misalignment in bits (`sh`, 0
// or 16) the caller has already established for every field it reads.
fn word_sh(byte_offset: u32, sh: u32) -> u32 {
    let w = byte_offset / 4u;
    if (sh == 0u) {
        return weights[w];
    }
    return (weights[w] >> sh) | (weights[w + 1u] << (32u - sh));
}

// The four bytes of a u32 as floats, each biased by -32: Q6_K stores its
// 6-bit weights unsigned and the dequant is `d * scale * (q - 32)`.
fn unpack4_f_bias32(w: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(w & 0xFFu) - 32.0,
        f32((w >> 8u) & 0xFFu) - 32.0,
        f32((w >> 16u) & 0xFFu) - 32.0,
        f32((w >> 24u) & 0xFFu) - 32.0,
    );
}

// Byte `k` (0..12) of the three consecutive words `w0 w1 w2`, as a signed
// 8-bit scale.
fn scale_of(w0: u32, w1: u32, w2: u32, k: u32) -> f32 {
    let w = select(select(w2, w1, k < 8u), w0, k < 4u);
    let b = (w >> (8u * (k % 4u))) & 0xFFu;
    return f32(i32(b << 24u) >> 24u);
}

// Contribution of super-block `i` of one output row, computed by the 16
// threads that own it; `sh` is this thread's misalignment, see the doc.
fn sb(block_byte_off: u32, x_block_elem: u32, ql_offset: u32, qh_offset: u32,
      s_offset: u32, y_offset: u32, sh: u32) -> f32 {
    let d = f16_to_f32(word_sh(block_byte_off + 208u, sh));

    let ql0 = word_sh(block_byte_off + ql_offset, sh);
    let ql32 = word_sh(block_byte_off + ql_offset + 32u, sh);
    let qh = word_sh(block_byte_off + 128u + qh_offset, sh);

    // The scale bytes at s_offset + {0, 2, 4, 6} span seven bytes: three
    // aligned words cover them from any start.
    let sb0 = block_byte_off + 192u + s_offset;
    let sw = sb0 / 4u;
    let sk = sb0 % 4u;
    let w0 = weights[sw];
    let w1 = weights[sw + 1u];
    let w2 = weights[sw + 2u];

    let q0 = (ql0 & 0x0F0F0F0Fu) | ((qh & 0x03030303u) << 4u);
    let q1 = (ql32 & 0x0F0F0F0Fu) | ((qh & 0x0C0C0C0Cu) << 2u);
    let q2 = ((ql0 >> 4u) & 0x0F0F0F0Fu) | (qh & 0x30303030u);
    let q3 = ((ql32 >> 4u) & 0x0F0F0F0Fu) | ((qh & 0xC0C0C0C0u) >> 2u);

    let y0 = (x_block_elem + y_offset) / 4u;
    var sum: f32 = 0.0;
    sum = sum + scale_of(w0, w1, w2, sk) * dot(xv[y0], unpack4_f_bias32(q0));
    sum = sum + scale_of(w0, w1, w2, sk + 2u) * dot(xv[y0 + 8u], unpack4_f_bias32(q1));
    sum = sum + scale_of(w0, w1, w2, sk + 4u) * dot(xv[y0 + 16u], unpack4_f_bias32(q2));
    sum = sum + scale_of(w0, w1, w2, sk + 6u) * dot(xv[y0 + 24u], unpack4_f_bias32(q3));
    return d * sum;
}
"#;

/// The **light** `Q6_K` decode matmul-vec kernel — [`Q6K_LIGHT_PRELUDE`] in
/// the same 32-thread, `n_rows`-batched dispatch as the `Q4_K`/`Q5_K` twins.
///
/// `Q6_K` was the slowest K-quant after `Q5_K` got its own kernel: 36.1 GiB/s
/// of effective weight streaming against 48–58 for the light kernels. It drags
/// every `_M`-suffixed mixed quantization down with it, since those store
/// `attn_v`/`ffn_down` as `Q6_K`.
///
/// Not bit-identical against `CpuBackend` (it reorders the float adds), so it
/// cross-checks within tolerance.
pub fn shader_source_reduce_q6k_light(n_rows: usize, subgroup: bool) -> String {
    let words = !crate::engine::env::flag_on("ORANGU_Q6K_V4");
    let mut s = String::from(if words {
        Q6K_LIGHT_PRELUDE_WORDS
    } else {
        Q6K_LIGHT_PRELUDE
    });
    s.push_str(&format!(
        "\nvar<workgroup> psums: array<f32, {}>;\n\n",
        n_rows * 32
    ));
    s.push_str("@compute @workgroup_size(32)\nfn main(\n    @builtin(workgroup_id) wid: vec3<u32>,\n    @builtin(local_invocation_id) lid: vec3<u32>,\n    @builtin(num_workgroups) nwg: vec3<u32>,");
    s.push_str(subgroup_entry_params(subgroup));
    s.push_str("\n) {\n");
    s.push_str(&format!(
        "    let n_row_groups = (params.out_dim + {}u) / {n_rows}u;\n",
        n_rows - 1
    ));
    s.push_str("    let flat = wid.x + wid.y * nwg.x + wid.z * nwg.x * nwg.y;\n    if (flat >= n_row_groups * params.n_tokens) {\n        return;\n    }\n");
    s.push_str("    let rg = flat / params.n_tokens;\n    let t = flat % params.n_tokens;\n");
    s.push_str(&format!("    let o0 = rg * {n_rows}u;\n"));
    for i in 1..n_rows {
        s.push_str(&format!("    let o{i} = o0 + {i}u;\n"));
    }
    // `Q6_K`'s own mapping, not `Q4_K`'s: 16 threads split the super-block into
    // two 128-element halves (`v_im`), each thread taking a 4-element run
    // (`l0`). Every offset it produces is 4-aligned, so the activation reads
    // are `vec4` without further work.
    s.push_str("    let tid = lid.x;\n    let itid = tid % 16u;\n    let ix = tid / 16u;\n");
    s.push_str("    let v_im = itid / 8u;\n    let v_in = itid % 8u;\n    let l0 = 4u * v_in;\n    let is = v_in / 4u;\n");
    s.push_str("    let ql_offset = 64u * v_im + l0;\n    let qh_offset = 32u * v_im + l0;\n    let s_offset = 8u * v_im + is;\n    let y_offset = 128u * v_im + l0;\n");
    s.push_str("    let num_blocks = params.in_dim / 256u;\n    let x_tok = t * params.in_dim;\n");
    for i in 0..n_rows {
        s.push_str(&format!("    var acc{i}: f32 = 0.0;\n"));
    }
    if words {
        // Fixed for the loop: two blocks are 420 bytes, a multiple of four,
        // and every row shares one row length.
        s.push_str("    let sh0 = ((o0 * params.row_bytes + ix * 210u) % 4u) * 8u;\n");
        for r in 1..n_rows {
            s.push_str(&format!(
                "    let sh{r} = ((o{r} * params.row_bytes + ix * 210u) % 4u) * 8u;\n"
            ));
        }
    }
    s.push_str("    var i: u32 = ix;\n    loop {\n        if (i >= num_blocks) {\n            break;\n        }\n        let xrb = x_tok + i * 256u;\n");
    for r in 0..n_rows {
        if words {
            s.push_str(&format!(
                "        acc{r} = acc{r} + sb(o{r} * params.row_bytes + i * 210u, xrb, ql_offset, qh_offset, s_offset, y_offset, sh{r});\n"
            ));
        } else {
            s.push_str(&format!(
                "        acc{r} = acc{r} + sb(o{r} * params.row_bytes + i * 210u, xrb, ql_offset, qh_offset, s_offset, y_offset);\n"
            ));
        }
    }
    s.push_str("        i = i + 2u;\n    }\n");
    for r in 0..n_rows {
        if subgroup {
            s.push_str(&format!(
                "    let sg{r} = subgroupAdd(acc{r});\n    if (sg_lane == 0u && o{r} < params.out_dim) {{\n        y[t * params.out_dim + o{r}] = sg{r};\n    }}\n"
            ));
        } else {
            s.push_str(&format!("    psums[tid * {n_rows}u + {r}u] = acc{r};\n"));
        }
    }
    if !subgroup {
        s.push_str("    workgroupBarrier();\n    var stride: u32 = 16u;\n    loop {\n        if (stride == 0u) {\n            break;\n        }\n        if (tid < stride) {\n");
        for r in 0..n_rows {
            s.push_str(&format!(
                "            psums[tid * {n_rows}u + {r}u] = psums[tid * {n_rows}u + {r}u] + psums[(tid + stride) * {n_rows}u + {r}u];\n"
            ));
        }
        s.push_str("        }\n        workgroupBarrier();\n        stride = stride / 2u;\n    }\n    if (tid == 0u) {\n");
        for r in 0..n_rows {
            s.push_str(&format!(
                "        if (o{r} < params.out_dim) {{\n            y[t * params.out_dim + o{r}] = psums[{r}u];\n        }}\n"
            ));
        }
        s.push_str("    }\n");
    }
    s.push_str("}\n");
    s
}

/// The **light** `Q4_K` decode matmul-vec kernel (`ORANGU_Q4K_LIGHT`): a WGSL
/// port of llama.cpp's `mul_mat_vec_q4_k.comp`, targeting register pressure /
/// occupancy rather than load count (which earlier experiments showed is a null
/// on this GPU). Where the default dual-nibble kernel gives one 32-thread workgroup a
/// whole super-block and keeps eight activations plus `n_rows` accumulators
/// live, this gives **16 threads** each a super-block (two run in parallel in
/// the 32-thread workgroup, `it_size = 2`), each thread owning 16 weight
/// nibbles from two packed `u32` `qs` reads and four `vec4` activation reads —
/// so the only state live across the block loop is the `n_rows` accumulators.
/// Same 6-binding shape, `Meta`, dispatch grid (`ceil(out_dim / n_rows)` × per
/// token, via `selects_wide_unroll`), and 144-byte block layout as the dual
/// kernel, so it drops into `pipeline_for` with no dispatch/bind-group change.
/// Not bit-identical (it reorders the float adds vs. `CpuBackend`), but
/// cross-checks within tolerance. Opt-in and default-off. As measured via
/// `RADV_DEBUG=shaderstats`, it compiles to *higher* register pressure than the
/// dual kernel (holding 16 nibbles + four `vec4` activations + eight scalar
/// scales live per super-block costs more registers than it saves in code), so
/// it lowers rather than raises occupancy. Source-level register-minimizing (accumulating each
/// `vec4` group in turn instead of hoisting all four) did not move the count —
/// the allocation is compiler-determined. It is therefore *not* the occupancy
/// lever it was meant to be; kept for reference / other GPUs.
pub fn shader_source_reduce_q4k_light(n_rows: usize, subgroup: bool) -> String {
    // `ORANGU_GEMV_STUB=1` swaps the block reader for a load-preserving,
    // arithmetic-free stub — see `Q4K_LIGHT_STUB_SB`. Wrong output on purpose.
    let mut s = if crate::engine::env::flag_on("ORANGU_GEMV_STUB") {
        let start = Q4K_LIGHT_PRELUDE
            .find("\nfn sb(")
            .expect("prelude defines sb");
        let end = Q4K_LIGHT_PRELUDE[start..]
            .find("\n}\n")
            .map(|e| start + e + 3)
            .expect("sb has a closing brace");
        let mut t = String::from(&Q4K_LIGHT_PRELUDE[..start]);
        // Keep the leading newline: the prelude's `fn sb` is preceded by a
        // `//` comment, and splicing onto that line comments the function out.
        t.push_str(Q4K_LIGHT_STUB_SB);
        t.push_str(&Q4K_LIGHT_PRELUDE[end..]);
        t
    } else {
        String::from(Q4K_LIGHT_PRELUDE)
    };
    s.push_str(&format!(
        "\nvar<workgroup> psums: array<f32, {}>;\n\n",
        n_rows * 32
    ));
    s.push_str("@compute @workgroup_size(32)\nfn main(\n    @builtin(workgroup_id) wid: vec3<u32>,\n    @builtin(local_invocation_id) lid: vec3<u32>,\n    @builtin(num_workgroups) nwg: vec3<u32>,");
    s.push_str(subgroup_entry_params(subgroup));
    s.push_str("\n) {\n");
    s.push_str(&format!(
        "    let n_row_groups = (params.out_dim + {}u) / {n_rows}u;\n",
        n_rows - 1
    ));
    s.push_str("    let flat = wid.x + wid.y * nwg.x + wid.z * nwg.x * nwg.y;\n    if (flat >= n_row_groups * params.n_tokens) {\n        return;\n    }\n");
    s.push_str("    let rg = flat / params.n_tokens;\n    let t = flat % params.n_tokens;\n");
    s.push_str(&format!("    let o0 = rg * {n_rows}u;\n"));
    for i in 1..n_rows {
        s.push_str(&format!("    let o{i} = o0 + {i}u;\n"));
    }
    // **Row lanes** (`ORANGU_Q4K_ROWLANES=1`, `n_rows == 2`): the workgroup's
    // two 16-lane halves take two *rows* rather than two super-blocks of one
    // row. Each half walks every super-block of its own row with stride 1.
    //
    // What it changes is the count of workgroups and what each one costs.
    // At a model's FFN shape a row is six super-blocks, so under the split
    // below each lane makes **three** trips round the block loop and then
    // pays the per-workgroup fixed cost — descriptor loads, the reduction,
    // the store — for 864 bytes of weights. Halving the workgroup count and
    // doubling the trips amortises that cost 2× at **the same register
    // footprint**: one accumulator per lane, exactly as before. The
    // accumulator-per-row form of `n_rows = 2` amortised the same cost by
    // growing every lane's live state and measured slower for it.
    //
    // The reduction is then per half — sixteen lanes, not thirty-two — done
    // with subgroup shuffles where the subgroup is available and a
    // half-width shared-memory tree where it is not.
    // **Shared activation** (`ORANGU_Q4K_SHARED_ACT=1`, `n_rows == 2`): the
    // ordinary split — two 16-lane halves, two super-blocks of a row per
    // trip — but with the workgroup owning two rows and loading each
    // super-block's four activation vectors **once**, applying them to both
    // rows. Per lane and per super-block a row costs ~9 bytes of unique
    // weight and 64 bytes of activation; across a whole matvec that is
    // seven times the weight bytes in activation traffic, re-read from cache
    // for every row. The accumulator form of `n_rows = 2` that measured
    // slower did not share those loads at all — it called `sb` once per row
    // and paid the traffic twice while also holding two accumulators.
    if shared_act() && n_rows == 2 {
        s.push_str("    let tid = lid.x;\n    let itid = tid % 16u;\n    let ix = tid / 16u;\n");
        s.push_str("    let il = itid / 4u;\n    let ir = itid % 4u;\n    let v_im = il / 2u;\n    let v_in = il % 2u;\n    let l0 = 4u * (2u * ir + v_in);\n    let q_offset = 32u * v_im + l0;\n    let y_offset = 64u * v_im + l0;\n");
        s.push_str(
            "    let num_blocks = params.in_dim / 256u;\n    let x_tok = t * params.in_dim;\n",
        );
        s.push_str("    var acc0: f32 = 0.0;\n    var acc1: f32 = 0.0;\n    let has1 = o1 < params.out_dim;\n");
        s.push_str("    var i: u32 = ix;\n    loop {\n        if (i >= num_blocks) {\n            break;\n        }\n        let xrb = x_tok + i * 256u;\n        let y1 = (xrb + y_offset) / 4u;\n        let y2 = (xrb + y_offset + 128u) / 4u;\n        let by0 = xv[y1];\n        let by1 = xv[y1 + 8u];\n        let by2 = xv[y2];\n        let by3 = xv[y2 + 8u];\n");
        s.push_str("        acc0 = acc0 + sb_act(o0 * params.row_bytes + i * 144u, q_offset, v_im, by0, by1, by2, by3);\n        if (has1) {\n            acc1 = acc1 + sb_act(o1 * params.row_bytes + i * 144u, q_offset, v_im, by0, by1, by2, by3);\n        }\n        i = i + 2u;\n    }\n\n");
        s.push_str(&light_reduce(2, subgroup));
        s.push_str("}\n");
        return s;
    }
    if row_lanes() && n_rows == 2 {
        s.push_str("    let tid = lid.x;\n    let itid = tid % 16u;\n    let half = tid / 16u;\n    let orow = o0 + half;\n");
        s.push_str("    let il = itid / 4u;\n    let ir = itid % 4u;\n    let v_im = il / 2u;\n    let v_in = il % 2u;\n    let l0 = 4u * (2u * ir + v_in);\n    let q_offset = 32u * v_im + l0;\n    let y_offset = 64u * v_im + l0;\n");
        s.push_str(
            "    let num_blocks = params.in_dim / 256u;\n    let x_tok = t * params.in_dim;\n",
        );
        s.push_str("    var acc: f32 = 0.0;\n    if (orow < params.out_dim) {\n        var i: u32 = 0u;\n        loop {\n            if (i >= num_blocks) {\n                break;\n            }\n            acc = acc + sb(orow * params.row_bytes + i * 144u, x_tok + i * 256u, y_offset, q_offset, v_im);\n            i = i + 1u;\n        }\n    }\n");
        if subgroup {
            // Butterfly within each 16-lane half: xor distances 8,4,2,1 never
            // cross the half boundary (bit 4 is untouched).
            s.push_str("    acc = acc + subgroupShuffleXor(acc, 8u);\n    acc = acc + subgroupShuffleXor(acc, 4u);\n    acc = acc + subgroupShuffleXor(acc, 2u);\n    acc = acc + subgroupShuffleXor(acc, 1u);\n");
            s.push_str("    if (itid == 0u && orow < params.out_dim) {\n        y[t * params.out_dim + orow] = acc;\n    }\n}\n");
        } else {
            s.push_str("    psums[tid] = acc;\n    workgroupBarrier();\n    var stride: u32 = 8u;\n    loop {\n        if (stride == 0u) {\n            break;\n        }\n        if (itid < stride) {\n            psums[tid] = psums[tid] + psums[tid + stride];\n        }\n        workgroupBarrier();\n        stride = stride / 2u;\n    }\n");
            s.push_str("    if (itid == 0u && orow < params.out_dim) {\n        y[t * params.out_dim + orow] = psums[tid];\n    }\n}\n");
        }
        return s;
    }
    s.push_str("    let tid = lid.x;\n    let itid = tid % 16u;\n    let ix = tid / 16u;\n");
    s.push_str("    let il = itid / 4u;\n    let ir = itid % 4u;\n    let v_im = il / 2u;\n    let v_in = il % 2u;\n    let l0 = 4u * (2u * ir + v_in);\n    let q_offset = 32u * v_im + l0;\n    let y_offset = 64u * v_im + l0;\n");
    s.push_str("    let num_blocks = params.in_dim / 256u;\n    let x_tok = t * params.in_dim;\n");
    // `ORANGU_REDUCE_ROW_LOOP=1`: one workgroup still covers `n_rows` output
    // rows, but walks them in a loop with a **single** accumulator live rather
    // than unrolling `acc0..accN` into registers.
    //
    // P4's remaining candidate is per-workgroup fixed cost — launch, reduction,
    // store — spread over a loop of only three iterations. Raising `n_rows` is
    // the obvious way to amortize it and makes things monotonically worse, but
    // the unrolled form pays for that amortization in register pressure, so the
    // two effects cannot be told apart. This form amortizes the *launch* over
    // `n_rows` rows at constant register cost; the reduction and store still
    // happen per row. If throughput now rises with `n_rows` where it fell
    // before, the launch is the cost.
    if row_loop() && n_rows > 1 {
        s.push_str(&format!(
            "    var row: u32 = 0u;\n    loop {{\n        if (row >= {n_rows}u) {{\n            break;\n        }}\n        let orow = o0 + row;\n        if (orow < params.out_dim) {{\n            var acc0: f32 = 0.0;\n            var i: u32 = ix;\n            loop {{\n                if (i >= num_blocks) {{\n                    break;\n                }}\n                let xrb = x_tok + i * 256u;\n                acc0 = acc0 + sb(orow * params.row_bytes + i * 144u, xrb, y_offset, q_offset, v_im);\n                i = i + 2u;\n            }}\n"
        ));
        if subgroup {
            s.push_str("            let sg0 = subgroupAdd(acc0);\n            if (sg_lane == 0u) {\n                y[t * params.out_dim + orow] = sg0;\n            }\n");
        } else {
            s.push_str("            psums[tid] = acc0;\n            workgroupBarrier();\n            var stride: u32 = 16u;\n            loop {\n                if (stride == 0u) {\n                    break;\n                }\n                if (tid < stride) {\n                    psums[tid] = psums[tid] + psums[tid + stride];\n                }\n                workgroupBarrier();\n                stride = stride / 2u;\n            }\n            if (tid == 0u) {\n                y[t * params.out_dim + orow] = psums[0];\n            }\n            workgroupBarrier();\n");
        }
        s.push_str("        }\n        row = row + 1u;\n    }\n}\n");
        return s;
    }
    for i in 0..n_rows {
        s.push_str(&format!("    var acc{i}: f32 = 0.0;\n"));
    }
    // it_size = workgroup_size(32) / 16 = 2 super-blocks per iteration, so a
    // thread walks blocks `ix, ix+2, ix+4, ...`.
    //
    // Unrolled `U` blocks at a time. Rolled, this loop drained every
    // outstanding load once per block — 12 sequential `vmcnt(0)` waits for an
    // `ffn_down` row (24 blocks, stride 2) with only 7 loads in flight between
    // them. `U` independent `sb()` calls have no dependency between them, so
    // their loads issue together and drain once. Same shape as the tiled GEMM's
    // staging fill, same reason (LESSONS §7).
    //
    // The accumulation stays **bit-exact**: `acc = acc + a0 + a1 + ...` is
    // left-associative, which is the same order the rolled loop's successive
    // `acc = acc + a` statements produced.
    let unroll = reduce_block_unroll();
    let sb_call = |row: usize, blk: usize| {
        format!(
            "sb(o{row} * params.row_bytes + (i + {}u) * 144u, x_tok + (i + {}u) * 256u, y_offset, q_offset, v_im)",
            blk * 2,
            blk * 2
        )
    };
    if unroll > 1 {
        s.push_str(&format!(
            "    var i: u32 = ix;\n    loop {{\n        if (i + {}u >= num_blocks) {{\n            break;\n        }}\n",
            (unroll - 1) * 2
        ));
        for row in 0..n_rows {
            let terms: Vec<String> = (0..unroll).map(|b| sb_call(row, b)).collect();
            let sum = terms.join("\n            + ");
            if row == 0 {
                s.push_str(&format!("        acc0 = acc0 + {sum};\n"));
            } else {
                s.push_str(&format!(
                    "        if (o{row} < params.out_dim) {{\n            acc{row} = acc{row} + {sum};\n        }}\n"
                ));
            }
        }
        s.push_str(&format!("        i = i + {}u;\n    }}\n", unroll * 2));
        // Remainder: the same body one block at a time.
        s.push_str("    loop {\n        if (i >= num_blocks) {\n            break;\n        }\n");
    } else {
        s.push_str("    var i: u32 = ix;\n    loop {\n        if (i >= num_blocks) {\n            break;\n        }\n");
    }
    s.push_str("        let xrb = x_tok + i * 256u;\n");
    s.push_str("        acc0 = acc0 + sb(o0 * params.row_bytes + i * 144u, xrb, y_offset, q_offset, v_im);\n");
    for i in 1..n_rows {
        s.push_str(&format!(
            "        if (o{i} < params.out_dim) {{\n            acc{i} = acc{i} + sb(o{i} * params.row_bytes + i * 144u, xrb, y_offset, q_offset, v_im);\n        }}\n"
        ));
    }
    s.push_str("        i = i + 2u;\n    }\n\n");
    s.push_str(&light_reduce(n_rows, subgroup));
    s.push_str("}\n");
    s
}

/// Super-blocks the decode `Q4_K` GEMV fuses into one loop body, so that many
/// blocks' loads are in flight at once instead of one block's.
///
/// `ORANGU_REDUCE_UNROLL`, clamped `1..=8`. `1` restores the rolled loop
/// verbatim and is the control.
fn reduce_block_unroll() -> usize {
    static U: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *U.get_or_init(|| {
        super::env_tuning_value(
            "ORANGU_REDUCE_UNROLL",
            REDUCE_BLOCK_UNROLL_DEFAULT,
            "an integer in 1..=8",
            |u| (1..=8).contains(&u),
        )
    })
}

/// Whether the light `Q4_K` GEMV loads each super-block's activation once and
/// applies it to two rows — see the emission site. `ORANGU_Q4K_SHARED_ACT=1`;
/// only meaningful with `ORANGU_REDUCE_N_ROWS=2`.
fn shared_act() -> bool {
    static R: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *R.get_or_init(|| crate::engine::env::flag_on("ORANGU_Q4K_SHARED_ACT"))
}

/// Whether the light `Q4_K` GEMV gives its two 16-lane halves two rows rather
/// than two super-blocks of one row — see the emission site.
/// `ORANGU_Q4K_ROWLANES=1`; only meaningful with `ORANGU_REDUCE_N_ROWS=2`.
fn row_lanes() -> bool {
    static R: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *R.get_or_init(|| crate::engine::env::flag_on("ORANGU_Q4K_ROWLANES"))
}

/// Whether the decode GEMV walks its workgroup's output rows in a loop with one
/// live accumulator instead of unrolling them into registers — see the emission
/// site. `ORANGU_REDUCE_ROW_LOOP=1`.
fn row_loop() -> bool {
    static R: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *R.get_or_init(|| crate::engine::env::flag_on("ORANGU_REDUCE_ROW_LOOP"))
}

/// See [`reduce_block_unroll`]. **1** — measured flat at 2, 4 and 8
/// (ffn-side 13.41–13.57 ms across all of them), so the plain loop ships.
/// Unlike the tiled GEMM's staging fill, this loop is not latency-exposed:
/// decode dispatches one workgroup per output row, thousands of them, and that
/// hides load latency without any help from unrolling.
const REDUCE_BLOCK_UNROLL_DEFAULT: usize = 1;

/// Reduction for [`shader_source_reduce_q4k_light`] — sums each row's `acc`
/// across all 32 lanes (a single 32-lane subgroup, so `subgroupAdd` when
/// available, else a 32-wide shared-memory tree) and lane 0 writes it out.
fn light_reduce(n_rows: usize, subgroup: bool) -> String {
    let mut s = String::new();
    if subgroup {
        for i in 0..n_rows {
            s.push_str(&format!("    let sg{i} = subgroupAdd(acc{i});\n"));
        }
        s.push_str("    if (sg_lane == 0u) {\n");
        s.push_str("        y[t * params.out_dim + o0] = sg0;\n");
        for i in 1..n_rows {
            s.push_str(&format!(
                "        if (o{i} < params.out_dim) {{\n            y[t * params.out_dim + o{i}] = sg{i};\n        }}\n"
            ));
        }
        s.push_str("    }\n");
    } else {
        for i in 0..n_rows {
            s.push_str(&format!("    psums[{i}u * 32u + tid] = acc{i};\n"));
        }
        s.push_str("    workgroupBarrier();\n    var stride: u32 = 16u;\n    loop {\n        if (stride == 0u) {\n            break;\n        }\n        if (tid < stride) {\n");
        for i in 0..n_rows {
            s.push_str(&format!(
                "            psums[{i}u * 32u + tid] = psums[{i}u * 32u + tid] + psums[{i}u * 32u + tid + stride];\n"
            ));
        }
        s.push_str(
            "        }\n        workgroupBarrier();\n        stride = stride / 2u;\n    }\n",
        );
        s.push_str("    if (tid == 0u) {\n");
        s.push_str("        y[t * params.out_dim + o0] = psums[0];\n");
        for i in 1..n_rows {
            s.push_str(&format!(
                "        if (o{i} < params.out_dim) {{\n            y[t * params.out_dim + o{i}] = psums[{i}u * 32u];\n        }}\n"
            ));
        }
        s.push_str("    }\n");
    }
    s
}

pub fn shader_source_reduce_q4k_dual_nibble(n_rows: usize, subgroup: bool) -> String {
    let suffix = dual_nibble_suffix(n_rows, subgroup);
    format!("{PRELUDE_VEC4}\n{Q4K_DUAL_MIDDLE}\n{suffix}")
}

/// `Q4_K` decode kernel with a **contiguous** thread→element mapping — a
/// load-efficiency variant of `shader_source_reduce_q4k_dual_nibble` that
/// measured *no faster* (this matmul is memory-latency-bound, not
/// load-issue-bound), kept opt-in. The dual kernel loads a full 16-byte
/// `vec4` per qs byte and keeps only one
/// byte (its four `qs_byte_q4k` calls each waste 15/16 of the fetched word)
/// and issues eight scalar activation loads. Here lane `local` (0..31) owns
/// the `u32` at qs offset `local*4` — four **consecutive** qs bytes (group
/// `g = local/8`, in-group base `p = (local%8)*4`), *all four used* — read
/// in one `read_word_v4`, and the four low/high positions it covers are
/// contiguous, so their activations are two `vec4<f32>` loads (`x` is bound
/// as `array<vec4<f32>>` on the same binding — storage layout is
/// type-agnostic). Per-super-block VMEM loads drop from ~5 weight + 8
/// activation to ~1 weight + 2 activation per output row. The per-element
/// arithmetic is `q4k_elem`'s, so it stays within the same cross-check
/// tolerance; the reduction is the shared `subgroupAdd`/tree over the
/// 32-lane subgroup. `n_rows` output rows share the two hoisted activation
/// `vec4`s.
pub fn shader_source_reduce_q4k_contig(n_rows: usize, subgroup: bool) -> String {
    // Reuse `PRELUDE_VEC4` verbatim except rebind `x` as a `vec4<f32>` view
    // (nothing in the prelude's helper bodies references the scalar `x`).
    let prelude = PRELUDE_VEC4.replace(
        "@group(0) @binding(1) var<storage, read> x: array<f32>;",
        "@group(0) @binding(1) var<storage, read> xv4: array<vec4<f32>>;",
    );
    let suffix = contig_suffix(n_rows, subgroup);
    format!("{prelude}\n{Q4K_CONTIG_MIDDLE}\n{suffix}")
}

const Q4K_CONTIG_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 144u;
const BLOCK_ELEMS: u32 = 256u;

// One 256-element super-block for lane `local` (0..31). `local` owns the
// `u32` at qs offset `local*4` — four consecutive qs bytes, all used. Group
// `g = local/8` picks the low/high scale sub-blocks (loaded once). Each qs
// byte's low nibble is weight `g*64 + p + c`, its high nibble weight
// `g*64 + 32 + p + c` (`p = (local%8)*4`, `c` the byte within the u32), so
// the four low positions are `xl.xyzw` and the four high positions
// `xh.xyzw`. Reuses `q4k_elem`'s exact per-element math (`d*scale*nib -
// dmin*min`), just factored out of the loop.
fn block_dot_contig(byte_offset: u32, local: u32, xl: vec4<f32>, xh: vec4<f32>) -> f32 {
    let vb = byte_offset / 16u;
    let header = weights[vb];
    let d = f16_to_f32(header.x & 0xFFFFu);
    let dmin = f16_to_f32(header.x >> 16u);
    let scales = vec3<u32>(header.y, header.z, header.w);
    let g = local / 8u;
    let w = read_word_v4(byte_offset + 16u + local * 4u);
    let slo = get_scale_min_k4_v4(scales, g * 2u);
    let shi = get_scale_min_k4_v4(scales, g * 2u + 1u);
    let dlo = d * f32(slo.x);
    let mlo = dmin * f32(slo.y);
    let dhi = d * f32(shi.x);
    let mhi = dmin * f32(shi.y);
    let b0 = w & 0xFFu;
    let b1 = (w >> 8u) & 0xFFu;
    let b2 = (w >> 16u) & 0xFFu;
    let b3 = (w >> 24u) & 0xFFu;
    return (dlo * f32(b0 & 0xFu) - mlo) * xl.x + (dhi * f32(b0 >> 4u) - mhi) * xh.x
         + (dlo * f32(b1 & 0xFu) - mlo) * xl.y + (dhi * f32(b1 >> 4u) - mhi) * xh.y
         + (dlo * f32(b2 & 0xFu) - mlo) * xl.z + (dhi * f32(b2 >> 4u) - mhi) * xh.z
         + (dlo * f32(b3 & 0xFu) - mlo) * xl.w + (dhi * f32(b3 >> 4u) - mhi) * xh.w;
}
"#;

/// The `@compute fn main` for the contiguous kernel — a 32-thread analogue
/// of [`dual_nibble_suffix`] that gathers **two `vec4<f32>` activations**
/// per super-block (the four contiguous low positions and the four high
/// positions lane `local` covers) instead of eight scalars, and calls
/// `block_dot_contig` once per output row. Same `ceil(out_dim / n_rows) *
/// n_tokens` workgroup dispatch and the shared `dual_nibble_reduce` combine.
fn contig_suffix(n_rows: usize, subgroup: bool) -> String {
    let mut s = format!(
        "var<workgroup> partial_sums: array<f32, {}>;\n\n",
        n_rows * 32
    );
    s.push_str("@compute @workgroup_size(32)\nfn main(\n    @builtin(workgroup_id) wid: vec3<u32>,\n    @builtin(local_invocation_id) lid: vec3<u32>,\n    @builtin(num_workgroups) nwg: vec3<u32>,");
    s.push_str(subgroup_entry_params(subgroup));
    s.push_str("\n) {\n");
    s.push_str(&format!(
        "    let n_row_groups = (params.out_dim + {}u) / {n_rows}u;\n",
        n_rows - 1
    ));
    s.push_str("    let flat = wid.x + wid.y * nwg.x + wid.z * nwg.x * nwg.y;\n    if (flat >= n_row_groups * params.n_tokens) {\n        return;\n    }\n");
    s.push_str("    let rg = flat / params.n_tokens;\n    let t = flat % params.n_tokens;\n");
    s.push_str(&format!("    let o0 = rg * {n_rows}u;\n"));
    for i in 1..n_rows {
        s.push_str(&format!("    let o{i} = o0 + {i}u;\n"));
    }
    s.push_str("    let local = lid.x;\n    let x_base = t * params.in_dim;\n");
    s.push_str("    let g = local / 8u;\n    let p = (local % 8u) * 4u;\n\n");
    for i in 0..n_rows {
        s.push_str(&format!("    var partial{i}: f32 = 0.0;\n"));
    }
    s.push_str("\n    let n_blocks = params.in_dim / BLOCK_ELEMS;\n    var b: u32 = 0u;\n    loop {\n        if (b >= n_blocks) {\n            break;\n        }\n");
    s.push_str(
        "        let block_off = b * BLOCK_BYTES;\n        let x_blk = x_base + b * BLOCK_ELEMS;\n",
    );
    s.push_str("        let xl = xv4[(x_blk + g * 64u + p) / 4u];\n        let xh = xv4[(x_blk + g * 64u + 32u + p) / 4u];\n");
    s.push_str(
        "        partial0 = partial0 + block_dot_contig(o0 * params.row_bytes + block_off, local, xl, xh);\n",
    );
    for i in 1..n_rows {
        s.push_str(&format!(
            "        if (o{i} < params.out_dim) {{\n            partial{i} = partial{i} + block_dot_contig(o{i} * params.row_bytes + block_off, local, xl, xh);\n        }}\n"
        ));
    }
    s.push_str("        b = b + 1u;\n    }\n\n");
    s.push_str(&dual_nibble_reduce(n_rows, subgroup));
    s.push_str("}\n");
    s
}

/// `Q6_K` decode kernel that loads each `qh` byte **once**, the analogue of
/// [`shader_source_reduce_q4k_dual_nibble`] for `Q6_K`. The two-wave
/// `Q6K_UNROLL_MIDDLE::block_dot` splits a 64-thread workgroup into two
/// `w_lo` halves that each re-read the same `qh` byte (`qh[l]`, `l =
/// local % 32`, identical for the two halves) — a redundant high-bit load
/// on every super-block. Here one 32-thread workgroup owns a whole
/// super-block: lane `l` (0..31) loads `qh` once and drives *both* `w_lo`
/// halves' eight elements from it, reusing the identical `q6k_elem` math
/// (so within the same cross-check tolerance the two-wave kernel already
/// has), with a barrier-free `subgroupAdd` reduction. Reuses
/// [`dual_nibble_suffix`]: its eight per-block activations
/// (`x[l]`,`x[32+l]`,…,`x[224+l]`) are exactly the positions the two merged
/// `w_lo` lanes covered — `w_lo=0`'s four elements map to `xl0..xl3`,
/// `w_lo=1`'s to `xh0..xh3`.
pub fn shader_source_reduce_q6k_dual(n_rows: usize, subgroup: bool) -> String {
    let suffix = dual_nibble_suffix(n_rows, subgroup);
    format!("{PRELUDE_VEC4}\n{Q6K_DUAL_MIDDLE}\n{suffix}")
}

const Q6K_DUAL_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 210u;
const BLOCK_ELEMS: u32 = 256u;

fn q6k_elem(d: f32, sc_off: u32, idx: u32, w_lo: u32, half: u32, is: u32, ql: u32, qh: u32) -> f32 {
    let qh_shift = half * 4u + w_lo * 2u;
    let sc_idx = is + half * 4u + w_lo * 2u;
    let nib = select(ql >> 4u, ql & 0xFu, half == 0u);
    let q = i32(nib | (((qh >> qh_shift) & 3u) << 4u)) - 32;
    var sc: i32 = i32(read_byte_v4(sc_off + idx * 8u + sc_idx));
    if (sc >= 128) { sc = sc - 256; }
    return d * f32(sc) * f32(q);
}

// One 256-element super-block for `local` in 0..31. `qhA`/`qhB` are each
// loaded once and shared across both `w_lo` halves (the two-wave kernel's
// redundant read); `ql` still differs per half. The eight elements map to
// the suffix's eight activations: `w_lo=0` -> `xl0..xl3`, `w_lo=1` ->
// `xh0..xh3`.
fn block_dot_dual(byte_offset: u32, local: u32,
                  xl0: f32, xh0: f32, xl1: f32, xh1: f32,
                  xl2: f32, xh2: f32, xl3: f32, xh3: f32) -> f32 {
    let ql_off = byte_offset;
    let qh_off = byte_offset + 128u;
    let sc_off = byte_offset + 192u;
    let d_offset = byte_offset + 208u;
    let dword = read_word_v4(d_offset - (d_offset % 4u));
    let d = f16_to_f32(select(dword & 0xFFFFu, dword >> 16u, (d_offset % 4u) != 0u));
    let l = local;
    let is = l / 16u;
    let qhA = read_byte_v4(qh_off + l);
    let qhB = read_byte_v4(qh_off + 32u + l);
    let qlA0 = read_byte_v4(ql_off + l);
    let qlA1 = read_byte_v4(ql_off + 32u + l);
    let qlB0 = read_byte_v4(ql_off + 64u + l);
    let qlB1 = read_byte_v4(ql_off + 96u + l);
    return q6k_elem(d, sc_off, 0u, 0u, 0u, is, qlA0, qhA) * xl0
         + q6k_elem(d, sc_off, 0u, 0u, 1u, is, qlA0, qhA) * xl1
         + q6k_elem(d, sc_off, 1u, 0u, 0u, is, qlB0, qhB) * xl2
         + q6k_elem(d, sc_off, 1u, 0u, 1u, is, qlB0, qhB) * xl3
         + q6k_elem(d, sc_off, 0u, 1u, 0u, is, qlA1, qhA) * xh0
         + q6k_elem(d, sc_off, 0u, 1u, 1u, is, qlA1, qhA) * xh1
         + q6k_elem(d, sc_off, 1u, 1u, 0u, is, qlB1, qhB) * xh2
         + q6k_elem(d, sc_off, 1u, 1u, 1u, is, qlB1, qhB) * xh3;
}
"#;

const Q4K_DUAL_MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 144u;
const BLOCK_ELEMS: u32 = 256u;

fn qs_byte_q4k(vec4_base: u32, qi: u32) -> u32 {
    let v4i = vec4_base + 1u + qi / 16u;
    let word = vec4_word(weights[v4i], (qi % 16u) / 4u);
    return (word >> (8u * (qi % 4u))) & 0xFFu;
}

fn q4k_elem(d: f32, dmin: f32, scales: vec3<u32>, g: u32, is_low: bool, byte: u32) -> f32 {
    let is_idx = g * 2u + select(1u, 0u, is_low);
    let sm = get_scale_min_k4_v4(scales, is_idx);
    let dd = d * f32(sm.x);
    let mm = dmin * f32(sm.y);
    let nib = select(byte >> 4u, byte & 0xFu, is_low);
    return dd * f32(nib) - mm;
}

// One 256-element super-block for `local` in 0..31. Each of the four qs
// bytes (one per 64-group, at group-relative position `local`) is loaded
// exactly once; both of its nibbles are consumed — the low nibble against
// `xl_g` (position `g*64 + local`), the high nibble against `xh_g`
// (position `g*64 + 32 + local`). Header + the four byte loads are issued
// before the dependent dequant-and-multiply-adds, same memory-level-
// parallelism idiom as the two-wave `block_dot`.
fn block_dot_dual(byte_offset: u32, local: u32,
                  xl0: f32, xh0: f32, xl1: f32, xh1: f32,
                  xl2: f32, xh2: f32, xl3: f32, xh3: f32) -> f32 {
    let vec4_base = byte_offset / 16u;
    let header = weights[vec4_base];
    let d = f16_to_f32(header.x & 0xFFFFu);
    let dmin = f16_to_f32(header.x >> 16u);
    let scales = vec3<u32>(header.y, header.z, header.w);
    let b0 = qs_byte_q4k(vec4_base, local);
    let b1 = qs_byte_q4k(vec4_base, 32u + local);
    let b2 = qs_byte_q4k(vec4_base, 64u + local);
    let b3 = qs_byte_q4k(vec4_base, 96u + local);
    return q4k_elem(d, dmin, scales, 0u, true, b0) * xl0
         + q4k_elem(d, dmin, scales, 0u, false, b0) * xh0
         + q4k_elem(d, dmin, scales, 1u, true, b1) * xl1
         + q4k_elem(d, dmin, scales, 1u, false, b1) * xh1
         + q4k_elem(d, dmin, scales, 2u, true, b2) * xl2
         + q4k_elem(d, dmin, scales, 2u, false, b2) * xh2
         + q4k_elem(d, dmin, scales, 3u, true, b3) * xl3
         + q4k_elem(d, dmin, scales, 3u, false, b3) * xh3;
}
"#;

/// The `@compute fn main` for the dual-nibble kernel — a 32-thread
/// (single-subgroup) analogue of [`unroll_suffix`]. Same
/// `ceil(out_dim / n_rows) * n_tokens` workgroup dispatch (so
/// `build_op_resources`/`selects_wide_unroll`'s existing count applies
/// unchanged — only the threads-per-workgroup differs, 32 vs 64), but each
/// lane gathers **eight** activations per block (a low/high pair per
/// 64-group) and calls `block_dot_dual` once per output row. The reduction
/// is `subgroupAdd` (no `workgroupBarrier`, since the whole 32-lane
/// workgroup is a single subgroup) when `subgroup`, else a 32-wide barrier
/// tree.
fn dual_nibble_suffix(n_rows: usize, subgroup: bool) -> String {
    let mut s = format!(
        "var<workgroup> partial_sums: array<f32, {}>;\n\n",
        n_rows * 32
    );
    s.push_str("@compute @workgroup_size(32)\nfn main(\n    @builtin(workgroup_id) wid: vec3<u32>,\n    @builtin(local_invocation_id) lid: vec3<u32>,\n    @builtin(num_workgroups) nwg: vec3<u32>,");
    s.push_str(subgroup_entry_params(subgroup));
    s.push_str("\n) {\n");
    s.push_str(&format!(
        "    let n_row_groups = (params.out_dim + {}u) / {n_rows}u;\n",
        n_rows - 1
    ));
    s.push_str("    let flat = wid.x + wid.y * nwg.x + wid.z * nwg.x * nwg.y;\n    if (flat >= n_row_groups * params.n_tokens) {\n        return;\n    }\n");
    s.push_str("    let rg = flat / params.n_tokens;\n    let t = flat % params.n_tokens;\n");
    s.push_str(&format!("    let o0 = rg * {n_rows}u;\n"));
    for i in 1..n_rows {
        s.push_str(&format!("    let o{i} = o0 + {i}u;\n"));
    }
    s.push_str("    let local = lid.x;\n    let x_base = t * params.in_dim;\n\n");
    for i in 0..n_rows {
        s.push_str(&format!("    var partial{i}: f32 = 0.0;\n"));
    }
    s.push_str("\n    let n_blocks = params.in_dim / BLOCK_ELEMS;\n    var b: u32 = 0u;\n    loop {\n        if (b >= n_blocks) {\n            break;\n        }\n");
    s.push_str(
        "        let block_off = b * BLOCK_BYTES;\n        let x_blk = x_base + b * BLOCK_ELEMS;\n",
    );
    s.push_str("        let xl0 = x[x_blk + local];\n        let xh0 = x[x_blk + 32u + local];\n        let xl1 = x[x_blk + 64u + local];\n        let xh1 = x[x_blk + 96u + local];\n        let xl2 = x[x_blk + 128u + local];\n        let xh2 = x[x_blk + 160u + local];\n        let xl3 = x[x_blk + 192u + local];\n        let xh3 = x[x_blk + 224u + local];\n");
    s.push_str(
        "        partial0 = partial0 + block_dot_dual(o0 * params.row_bytes + block_off, local, xl0, xh0, xl1, xh1, xl2, xh2, xl3, xh3);\n",
    );
    for i in 1..n_rows {
        s.push_str(&format!(
            "        if (o{i} < params.out_dim) {{\n            partial{i} = partial{i} + block_dot_dual(o{i} * params.row_bytes + block_off, local, xl0, xh0, xl1, xh1, xl2, xh2, xl3, xh3);\n        }}\n"
        ));
    }
    s.push_str("        b = b + 1u;\n    }\n\n");
    s.push_str(&dual_nibble_reduce(n_rows, subgroup));
    s.push_str("}\n");
    s
}

/// Combine step for [`dual_nibble_suffix`]: reduce each output row's 32
/// per-lane partials to one value. `subgroup` → a single `subgroupAdd` per
/// row (the workgroup is exactly one subgroup, so no `workgroupBarrier` and
/// no cross-subgroup pass is needed, unlike `reduce_combine_block`'s
/// 64-thread/2-subgroup case); otherwise a 32-wide shared-memory barrier
/// tree (`stride = 16,8,4,2,1`).
fn dual_nibble_reduce(n_rows: usize, subgroup: bool) -> String {
    let mut s = String::new();
    if subgroup {
        for i in 0..n_rows {
            s.push_str(&format!("    let sg{i} = subgroupAdd(partial{i});\n"));
        }
        s.push_str("    if (sg_lane == 0u) {\n");
        s.push_str("        y[t * params.out_dim + o0] = sg0;\n");
        for i in 1..n_rows {
            s.push_str(&format!(
                "        if (o{i} < params.out_dim) {{\n            y[t * params.out_dim + o{i}] = sg{i};\n        }}\n"
            ));
        }
        s.push_str("    }\n");
    } else {
        for i in 0..n_rows {
            s.push_str(&format!(
                "    partial_sums[{i}u * 32u + local] = partial{i};\n"
            ));
        }
        s.push_str("    workgroupBarrier();\n    var stride: u32 = 16u;\n    loop {\n        if (stride == 0u) {\n            break;\n        }\n        if (local < stride) {\n");
        for i in 0..n_rows {
            s.push_str(&format!(
                "            partial_sums[{i}u * 32u + local] = partial_sums[{i}u * 32u + local] + partial_sums[{i}u * 32u + local + stride];\n"
            ));
        }
        s.push_str(
            "        }\n        workgroupBarrier();\n        stride = stride / 2u;\n    }\n",
        );
        s.push_str("    if (local == 0u) {\n");
        s.push_str("        y[t * params.out_dim + o0] = partial_sums[0];\n");
        for i in 1..n_rows {
            s.push_str(&format!(
                "        if (o{i} < params.out_dim) {{\n            y[t * params.out_dim + o{i}] = partial_sums[{i}u * 32u];\n        }}\n"
            ));
        }
        s.push_str("    }\n");
    }
    s
}

/// See `shader_source_reduce_q4k_wide_unroll` — same memory-level-parallelism
/// restructuring, for `Q5_K` (`Q5K_UNROLL_MIDDLE`).
pub fn shader_source_reduce_q5k_wide_unroll(n_rows: usize, subgroup: bool) -> String {
    let suffix = unroll_suffix(n_rows, subgroup);
    format!("{PRELUDE_VEC4}\n{Q5K_UNROLL_MIDDLE}\n{suffix}")
}

/// See `shader_source_reduce_q4k_wide_unroll` — same restructuring, for
/// `Q6_K` (`Q6K_UNROLL_MIDDLE`); it hoists loads rather than caching a
/// header (`Q6_K` has no vec4-aligned header).
pub fn shader_source_reduce_q6k_wide_unroll(n_rows: usize, subgroup: bool) -> String {
    let suffix = unroll_suffix(n_rows, subgroup);
    format!("{PRELUDE_VEC4}\n{Q6K_UNROLL_MIDDLE}\n{suffix}")
}

/// The complete block-unroll reduce source for `ggml_type`, or `None` if
/// this type has no unroll kernel (only the three K-quants do — the block-
/// unroll exploits their 256-element super-block geometry; the smaller
/// legacy quants and float types keep the wide-load/scalar reduce path).
pub fn shader_source_reduce_wide_unroll(
    ggml_type: u32,
    n_rows: usize,
    subgroup: bool,
) -> Option<String> {
    match ggml_type {
        t if t == GGML_TYPE_Q4_K => Some(shader_source_reduce_q4k_wide_unroll(n_rows, subgroup)),
        t if t == GGML_TYPE_Q5_K => Some(shader_source_reduce_q5k_wide_unroll(n_rows, subgroup)),
        t if t == GGML_TYPE_Q6_K => Some(shader_source_reduce_q6k_wide_unroll(n_rows, subgroup)),
        _ => None,
    }
}

/// `Q4_K` block-unroll combined with the packed-`f16` dot: the unroll's
/// four scalar `f32` multiply-adds replaced with **two** `v_dot2_f32_f16`
/// packed dots (groups 0/1 and 2/3 paired), halving the multiply-accumulate
/// count while keeping the unroll's header-once + hoisted-load memory
/// structure — the packed-dot technique applied to the *unrolled*
/// structure rather than the byte-wise/`MAIN_REDUCE_SUFFIX` one. Selected
/// only when both
/// the block-unroll (default) and `ORANGU_PACKED_DOT=1` are on
/// (`VulkanBackend::pipeline_for`). `f16` dot loses precision, so its
/// cross-check uses the same widened tolerance the byte-wise packed kernel
/// needs. `enable f16;` must lead the whole module (WGSL rule), so it can't
/// sit inside the shared middle/suffix.
pub fn shader_source_reduce_q4k_wide_unroll_packed_f16(n_rows: usize, subgroup: bool) -> String {
    const MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 144u;
const BLOCK_ELEMS: u32 = 256u;

fn qs_byte_q4k(vec4_base: u32, qi: u32) -> u32 {
    let v4i = vec4_base + 1u + qi / 16u;
    let word = vec4_word(weights[v4i], (qi % 16u) / 4u);
    return (word >> (8u * (qi % 4u))) & 0xFFu;
}

fn q4k_elem(d: f32, dmin: f32, scales: vec3<u32>, g: u32, is_low: bool, byte: u32) -> f32 {
    let is_idx = g * 2u + select(1u, 0u, is_low);
    let sm = get_scale_min_k4_v4(scales, is_idx);
    let dd = d * f32(sm.x);
    let mm = dmin * f32(sm.y);
    let nib = select(byte >> 4u, byte & 0xFu, is_low);
    return dd * f32(nib) - mm;
}

fn block_dot(byte_offset: u32, local: u32, x0: f32, x1: f32, x2: f32, x3: f32) -> f32 {
    let is_low = local < 32u;
    let qsi = select(local - 32u, local, is_low);
    let vec4_base = byte_offset / 16u;
    let header = weights[vec4_base];
    let d = f16_to_f32(header.x & 0xFFFFu);
    let dmin = f16_to_f32(header.x >> 16u);
    let scales = vec3<u32>(header.y, header.z, header.w);
    let b0 = qs_byte_q4k(vec4_base, qsi);
    let b1 = qs_byte_q4k(vec4_base, 32u + qsi);
    let b2 = qs_byte_q4k(vec4_base, 64u + qsi);
    let b3 = qs_byte_q4k(vec4_base, 96u + qsi);
    let e0 = q4k_elem(d, dmin, scales, 0u, is_low, b0);
    let e1 = q4k_elem(d, dmin, scales, 1u, is_low, b1);
    let e2 = q4k_elem(d, dmin, scales, 2u, is_low, b2);
    let e3 = q4k_elem(d, dmin, scales, 3u, is_low, b3);
    let w01 = vec2<f16>(f16(e0), f16(e1));
    let w23 = vec2<f16>(f16(e2), f16(e3));
    let x01 = vec2<f16>(f16(x0), f16(x1));
    let x23 = vec2<f16>(f16(x2), f16(x3));
    return f32(dot(w01, x01)) + f32(dot(w23, x23));
}
"#;
    let suffix = unroll_suffix(n_rows, subgroup);
    format!("enable f16;\n{PRELUDE_VEC4}\n{MIDDLE}\n{suffix}")
}

/// Wide loads (this file's
/// `PRELUDE_VEC4`/`Q4_K_WIDE_MIDDLE`) combined with the packed-`f16`
/// pairwise dot (`shader_source_reduce_q4k_packed_f16`'s own `dequant_
/// pair_f16`) — one addresses memory access, the other the multiply-
/// accumulate count. `Q4_K`-only, like the packed-dot
/// kernel itself (no other type has
/// a packed-`f16` kernel to combine with). `dequant_pair_f16` below is a
/// direct transcription of the byte-wise kernel's own — same "`k` must be
/// even, `k`/`k+1` always share one nibble half" invariant, same math —
/// just sourcing `d`/`dmin`/`scales` from one `vec4` header load (`Q4_K`'s
/// block is always vec4-aligned, see `Q4_K_WIDE_MIDDLE`'s own doc comment)
/// and `qs` bytes via vec4-based extraction instead of `read_u8`. Dispatch
/// (`SUFFIX` below) mirrors the packed-`f16` kernel's own one-row-per-
/// workgroup shape exactly, just walking `vec4_base` instead of
/// `byte_offset` — deliberately *not* attempting `REDUCE_N_ROWS` batching
/// on top of this, which would be a much bigger, more error-prone rewrite.
///
/// Correctness-verified; kept available (like `kv_f16`/`gpu_sample`) as a
/// selectable combination — see `VulkanBackend::wide_packed_pipeline`'s
/// own doc comment.
pub fn shader_source_reduce_q4k_wide_packed_f16() -> String {
    const MIDDLE: &str = r#"
const BLOCK_BYTES: u32 = 144u;
const BLOCK_ELEMS: u32 = 256u;

fn qs_byte_q4k_packed(vec4_base: u32, qi: u32) -> u32 {
    let v4i = vec4_base + 1u + qi / 16u;
    let word = vec4_word(weights[v4i], (qi % 16u) / 4u);
    return (word >> (8u * (qi % 4u))) & 0xFFu;
}

// `k` must be even — mirrors `shader_source_reduce_q4k_packed_f16`'s own
// `dequant_pair_f16` exactly (see its doc comment for why `k`/`k+1` always
// share one nibble half); only the byte source differs.
fn dequant_pair_f16(vec4_base: u32, k: u32) -> vec2<f16> {
    let header = weights[vec4_base];
    let d = f16_to_f32(header.x & 0xFFFFu);
    let dmin = f16_to_f32(header.x >> 16u);
    let scales = vec3<u32>(header.y, header.z, header.w);
    let q_offset = (k / 64u) * 64u;
    let local_in_group = k % 64u;
    let is_base = (q_offset / 64u) * 2u;
    let qi_base = q_offset / 2u;
    if (local_in_group < 32u) {
        let byte0 = qs_byte_q4k_packed(vec4_base, qi_base + local_in_group);
        let byte1 = qs_byte_q4k_packed(vec4_base, qi_base + local_in_group + 1u);
        let sm = get_scale_min_k4_v4(scales, is_base);
        let d1 = d * f32(sm.x);
        let m1 = dmin * f32(sm.y);
        return vec2<f16>(
            f16(d1 * f32(byte0 & 0xFu) - m1),
            f16(d1 * f32(byte1 & 0xFu) - m1),
        );
    }
    let l = local_in_group - 32u;
    let byte0 = qs_byte_q4k_packed(vec4_base, qi_base + l);
    let byte1 = qs_byte_q4k_packed(vec4_base, qi_base + l + 1u);
    let sm = get_scale_min_k4_v4(scales, is_base + 1u);
    let d2 = d * f32(sm.x);
    let m2 = dmin * f32(sm.y);
    return vec2<f16>(
        f16(d2 * f32(byte0 >> 4u) - m2),
        f16(d2 * f32(byte1 >> 4u) - m2),
    );
}
"#;
    const SUFFIX: &str = r#"
var<workgroup> partial_sums: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let flat = wid.x + wid.y * nwg.x + wid.z * nwg.x * nwg.y;
    if (flat >= params.out_dim * params.n_tokens) {
        return;
    }
    let o = flat / params.n_tokens;
    let t = flat % params.n_tokens;
    let local = lid.x;
    let row_vec4_base = (o * params.row_bytes) / 16u;
    let x_base = t * params.in_dim;

    var partial: f32 = 0.0;
    var k: u32 = local * 2u;
    loop {
        if (k >= params.in_dim) {
            break;
        }
        let block_idx = k / BLOCK_ELEMS;
        let local_k = k % BLOCK_ELEMS;
        let block_vec4_base = row_vec4_base + block_idx * (BLOCK_BYTES / 16u);
        let wv = dequant_pair_f16(block_vec4_base, local_k);
        let xv = vec2<f16>(f16(x[x_base + k]), f16(x[x_base + k + 1u]));
        partial = partial + f32(dot(wv, xv));
        k = k + 128u;
    }

    partial_sums[local] = partial;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (local < stride) {
            partial_sums[local] = partial_sums[local] + partial_sums[local + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (local == 0u) {
        y[t * params.out_dim + o] = partial_sums[0];
    }
}
"#;
    // `enable f16;` must precede every global declaration in the whole
    // module — same WGSL rule `shader_source_reduce_q4k_packed_f16` deals
    // with the same way.
    format!("enable f16;\n{PRELUDE_VEC4}\n{MIDDLE}\n{SUFFIX}")
}

/// Like [`shader_source`], but the cooperative variant (see
/// `MAIN_COOP_SUFFIX`) — used when `n_tokens` is large enough that
/// dequantizing each block once per workgroup and sharing it across many
/// tokens beats each token's thread dequantizing it independently.
pub fn shader_source_coop(ggml_type: u32) -> Option<String> {
    let middle = coop_middle(ggml_type)?;
    let prelude = prelude_for(ggml_type);
    Some(format!("{prelude}\n{middle}\n{MAIN_COOP_SUFFIX}"))
}

/// The default (opt out with `ORANGU_NO_TILED_PREFILL=1`) tiled-GEMM
/// alternative to [`shader_source_coop`] — see `MAIN_COOP_TILED_SUFFIX`'s
/// own doc comment for the design, and `MAIN_COOP_SUFFIX`'s for why this
/// is the default now.
pub fn shader_source_coop_tiled(ggml_type: u32, vec4_tiles: CoopVec4Tiles) -> Option<String> {
    let middle = coop_middle(ggml_type)?;
    let prelude = prelude_for(ggml_type);
    let g = coop_geom();
    let f16_tiles = coop_f16_tiles();
    let (acc_decl, inner, store, k_unroll) = coop_tiled_register_block(g, f16_tiles, vec4_tiles);
    // Each tile's declaration and the body of the `store_*` helper that writes
    // into it — chosen per tile, since `coop_vec4_tiles` answers per tile.
    let (tile_w_decl, store_w) = if vec4_tiles.w {
        (
            "var<workgroup> tile_w: array<vec4<%TILE_T%>, (%TILE_ROWS%u * %CHUNK%u) / 4u>;",
            "    tile_w[widx >> 2u][widx & 3u] = %TILE_T%(v);",
        )
    } else {
        (
            "var<workgroup> tile_w: array<%TILE_T%, %TILE_ROWS%u * %CHUNK%u>;",
            "    tile_w[widx] = %TILE_T%(v);",
        )
    };
    // `store_x4` writes a whole quad either way — as one vector store into a
    // `vec4` tile, or as four independent element stores into a scalar one.
    // Neither is a component store, so the fill that calls it is the same
    // code in both forms and the tile's shape stays a pure layout choice.
    let (tile_x_decl, store_x, store_x4) = if vec4_tiles.x {
        (
            "var<workgroup> tile_x: array<vec4<%TILE_T%>, (%TILE_TOKENS%u * %CHUNK%u) / 4u>;",
            "    tile_x[xidx >> 2u][xidx & 3u] = %TILE_T%(v);",
            "    tile_x[xidx >> 2u] = vec4<%TILE_T%>(v);",
        )
    } else {
        (
            "var<workgroup> tile_x: array<%TILE_T%, %TILE_TOKENS%u * %CHUNK%u>;",
            "    tile_x[xidx] = %TILE_T%(v);",
            "    tile_x[xidx] = %TILE_T%(v.x);\n    \
             tile_x[xidx + 1u] = %TILE_T%(v.y);\n    \
             tile_x[xidx + 2u] = %TILE_T%(v.z);\n    \
             tile_x[xidx + 3u] = %TILE_T%(v.w);",
        )
    };
    let tile_decl = format!("{tile_w_decl}\n{tile_x_decl}");
    let run = coop_tiled_run_len(g);
    // `ORANGU_NO_TILE_DEQUANT_RUN=1` restores the per-element fill this
    // replaced, so the two can be A/B'd in one build. Same output either way —
    // it is a knob for measuring the change, not for choosing behaviour.
    let amortized = !crate::engine::env::flag_on("ORANGU_NO_TILE_DEQUANT_RUN");
    let (w_fill, run_fill) = coop_tiled_weight_fill(ggml_type, run, amortized);
    let suffix = MAIN_COOP_TILED_SUFFIX
        .replace("%TILE_DECL%", &tile_decl)
        .replace("%STORE_W%", store_w)
        .replace("%STORE_X4%", store_x4)
        .replace("%STORE_X%", store_x)
        .replace("%TILE_T%", if f16_tiles { "f16" } else { "f32" })
        .replace("%THREADS_Y%", &g.threads_y.to_string())
        .replace("%THREADS_X%", &g.threads_x.to_string())
        .replace("%REG_ROWS%", &g.reg_rows.to_string())
        .replace("%REG_TOKENS%", &g.reg_tokens.to_string())
        .replace("%THREADS%", &COOP_THREADS.to_string())
        .replace("%TILE_ROWS%", &g.tile_rows().to_string())
        .replace("%TILE_TOKENS%", &g.tile_tokens().to_string())
        .replace("%CHUNK%", &g.chunk.to_string())
        .replace("%RUN%", &run.to_string())
        .replace("%W_FILL%", &w_fill)
        .replace("%RUN_FILL%", &run_fill)
        .replace("%X_FILL%", &coop_tiled_x_fill(g))
        .replace("%ACC_DECL%", &acc_decl)
        .replace("%INNER%", &inner)
        .replace("%STORE%", &store)
        .replace("%K_UNROLL%", &k_unroll.to_string());
    let enable = if f16_tiles { "enable f16;\n" } else { "" };
    Some(format!("{enable}{prelude}\n{middle}\n{suffix}"))
}

/// Shared `Meta` layout for every elementwise/norm shader below: `len` is
/// the element count to process, `extra` is a single per-op float parameter
/// (`eps` for [`RMSNORM_SHADER`], the multiplier for [`SCALE_SHADER`],
/// unused — but still present, so one Rust-side struct fits every op — for
/// the rest). These exist to fuse the CPU-side steps between a gemma4
/// layer's GPU matmul calls (RMSNorm, residual add, GEGLU's GELU + mul,
/// PLE's output scale) directly onto the GPU, so a whole post-attention
/// sub-layer chain — `wo` through the next layer's normed input — can be
/// recorded into one command encoder and read back once, instead of once
/// per matmul call. See `VulkanBackend::fused_post_attention`.
/// [`ELEM_META`] for a kernel assembled outside this module.
pub const ELEM_META_SRC: &str = ELEM_META;

const ELEM_META: &str = r#"
struct ElemMeta {
    len: u32,
    /// A second integer whose meaning is the shader's: a destination offset for
    /// the KV cast, the row width to broadcast along for `BIAS_ADD_SHADER_BODY`.
    aux: u32,
    extra: f32,
    out_scale: f32,
}
"#;

/// `y[i] += bias[i % row]` — a projection bias broadcast down the rows of a
/// `[n_tokens, row]` tensor.
///
/// Qwen2 carries `attn_q/k/v.bias`; Llama, Mistral and gemma do not. Until this
/// existed the fused Q/K/V chain had nowhere to put them, so a Qwen2 layer had
/// to stay on the step-by-step path — the bias was the last of four conventions
/// separating the two families.
///
/// Same `elem3` binding shape as the per-head norm (read storage, read-write
/// storage, uniform), so it reuses that layout rather than needing its own.
const BIAS_ADD_SHADER_BODY: &str = r#"
@group(0) @binding(0) var<storage, read> bias: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> em: ElemMeta;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= em.len) {
        return;
    }
    y[i] = y[i] + bias[i % em.aux];
}
"#;

/// The blockwise Hadamard rotation of `engine::hadamard` on the device:
/// `x` is `n_rows` rows of `em.len` elements, each row `em.len / BLOCK`
/// independent blocks, and one workgroup transforms one block in place —
/// the sign multiply (when `em.aux & 1`, before the butterflies; `em.aux
/// & 2` puts it after, which is the inverse for a folded lookup table),
/// then the `log2(BLOCK)` butterfly stages in shared memory, then the
/// `1 / sqrt(BLOCK)` normalization. The same arithmetic as
/// `hadamard::fwht_normalized`, to reassociation.
///
/// `elem3` bindings: the sign vector (`em.len` long; unread when `em.aux`
/// has neither bit, but a buffer must still be bound), the rows, the meta.
/// Workgroup `wid.x` is a flat block index across the rows.
pub fn shader_source_hadamard(block: u32) -> String {
    assert!(block.is_power_of_two() && block >= 2 * HADAMARD_WG);
    format!(
        r#"{ELEM_META}
@group(0) @binding(0) var<storage, read> signs: array<f32>;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<uniform> em: ElemMeta;

const BLOCK: u32 = {block}u;
const WG: u32 = {wg}u;
const PER_THREAD: u32 = BLOCK / WG;
var<workgroup> v: array<f32, BLOCK>;

@compute @workgroup_size({wg})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let blocks_per_row = em.len / BLOCK;
    let row = wid.x / blocks_per_row;
    let base = row * em.len + (wid.x % blocks_per_row) * BLOCK;
    let sign_base = (wid.x % blocks_per_row) * BLOCK;
    let t = lid.x;
    let pre = (em.aux & 1u) != 0u;
    let post = (em.aux & 2u) != 0u;
    for (var k: u32 = 0u; k < PER_THREAD; k = k + 1u) {{
        let i = t + k * WG;
        var val = x[base + i];
        if (pre) {{
            val = val * signs[sign_base + i];
        }}
        v[i] = val;
    }}
    workgroupBarrier();
    // `BLOCK / 2` butterfly pairs per stage, `PER_THREAD / 2` per thread:
    // pair `p` is elements `(p / h) * 2h + (p % h)` and `+ h`.
    var h: u32 = 1u;
    loop {{
        if (h >= BLOCK) {{
            break;
        }}
        for (var k: u32 = 0u; k < PER_THREAD / 2u; k = k + 1u) {{
            let p = t + k * WG;
            let j = (p / h) * 2u * h + (p % h);
            let a = v[j];
            let b = v[j + h];
            v[j] = a + b;
            v[j + h] = a - b;
        }}
        workgroupBarrier();
        h = h * 2u;
    }}
    let scale = 1.0 / sqrt(f32(BLOCK));
    for (var k: u32 = 0u; k < PER_THREAD; k = k + 1u) {{
        let i = t + k * WG;
        var val = v[i] * scale;
        if (post) {{
            val = val * signs[sign_base + i];
        }}
        x[base + i] = val;
    }}
}}
"#,
        wg = HADAMARD_WG
    )
}

/// [`shader_source_hadamard`] with the consumers' epilogue folded in, for
/// the decode chain: after the butterfly the block is written back as
/// `f32` (in place, for the float readers) **and** as the per-32 q8 the
/// integer-dot kernels read (`shader_source_quantize_q8`'s layout —
/// `[d, Σq, 8 words]` a block, the same rounding), thread `t < BLOCK/32`
/// quantizing block `t` out of shared memory. With `silu_mul`, the input
/// is `silu(a) · b` over two buffers (the FFN's gate and up outputs)
/// rather than `x` — the activation, the fold and the quantize as one
/// dispatch where the chain ran three. Six bindings
/// (`elem6_bind_group_layout`): `a`, `b`, `signs` read-only, `x` and `q8`
/// read-write, `meta`; without `silu_mul` the first two are unread.
/// One row only.
pub fn shader_source_hadamard_q8(block: u32, silu_mul: bool) -> String {
    assert!(block.is_power_of_two() && block >= 2 * HADAMARD_WG && block.is_multiple_of(32));
    let load = if silu_mul {
        "let g = a[base + i]; var val = g / (1.0 + exp(-g)) * b[base + i];"
    } else {
        "var val = x[base + i];"
    };
    format!(
        r#"{ELEM_META}
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> signs: array<f32>;
@group(0) @binding(3) var<storage, read_write> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> q8: array<u32>;
@group(0) @binding(5) var<uniform> em: ElemMeta;

const BLOCK: u32 = {block}u;
const WG: u32 = {wg}u;
const PER_THREAD: u32 = BLOCK / WG;
const Q8_BLOCKS: u32 = BLOCK / 32u;
var<workgroup> v: array<f32, BLOCK>;

@compute @workgroup_size({wg})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let base = wid.x * BLOCK;
    let sign_base = base;
    let t = lid.x;
    let pre = (em.aux & 1u) != 0u;
    let post = (em.aux & 2u) != 0u;
    for (var k: u32 = 0u; k < PER_THREAD; k = k + 1u) {{
        let i = t + k * WG;
        {load}
        if (pre) {{
            val = val * signs[sign_base + i];
        }}
        v[i] = val;
    }}
    workgroupBarrier();
    var h: u32 = 1u;
    loop {{
        if (h >= BLOCK) {{
            break;
        }}
        for (var k: u32 = 0u; k < PER_THREAD / 2u; k = k + 1u) {{
            let p = t + k * WG;
            let j = (p / h) * 2u * h + (p % h);
            let a0 = v[j];
            let b0 = v[j + h];
            v[j] = a0 + b0;
            v[j + h] = a0 - b0;
        }}
        workgroupBarrier();
        h = h * 2u;
    }}
    let scale = 1.0 / sqrt(f32(BLOCK));
    for (var k: u32 = 0u; k < PER_THREAD; k = k + 1u) {{
        let i = t + k * WG;
        var val = v[i] * scale;
        if (post) {{
            val = val * signs[sign_base + i];
        }}
        v[i] = val;
        x[base + i] = val;
    }}
    workgroupBarrier();
    if (t < Q8_BLOCKS) {{
        let qb = t * 32u;
        var amax: f32 = 0.0;
        for (var i: u32 = 0u; i < 32u; i = i + 1u) {{
            amax = max(amax, abs(v[qb + i]));
        }}
        let d = amax / 127.0;
        let id = select(0.0, 1.0 / d, d > 0.0);
        var sumq: i32 = 0;
        let out_base = (base / 32u + t) * 10u;
        for (var w: u32 = 0u; w < 8u; w = w + 1u) {{
            var word: u32 = 0u;
            for (var kk: u32 = 0u; kk < 4u; kk = kk + 1u) {{
                var q: i32 = i32(round(v[qb + w * 4u + kk] * id));
                q = clamp(q, -127, 127);
                sumq = sumq + q;
                word = word | ((u32(q) & 0xFFu) << (8u * kk));
            }}
            q8[out_base + 2u + w] = word;
        }}
        q8[out_base] = bitcast<u32>(d);
        q8[out_base + 1u] = bitcast<u32>(sumq);
    }}
}}
"#,
        wg = HADAMARD_WG
    )
}

/// The gated-DeltaNet step of `engine::arch::qwen_hybrid` — one token, every
/// head — as one dispatch: workgroup `h` is value head `h`, thread `j` owns
/// column `j` of that head's `HD × HD` state, so every step of
/// `delta_head_step` is a loop over the thread's own column with the reads
/// coalesced across the workgroup (`state[i * HD + j]`, `j` adjacent):
///
/// ```text
///   sk[j]      = Σ_i k[i] · decay · S[i][j]
///   d[j]       = β · (v[j] − sk[j])
///   S[i][j]    = decay · S[i][j] + k[i] · d[j]
///   o[j]       = Σ_i q[i] · S[i][j]
/// ```
///
/// then the gated RMSNorm over the head — the one reduction, through
/// shared memory — times the output gate of `z`, and the store, into the
/// tiled position (`h`) or, for a `gdn_v_grouped` fold, the grouped one
/// (`h % NK · REP + h / NK`), so the Hadamard kernel that follows needs no
/// permutation of its own. `q`/`k` are per key head and shared by the
/// `REP` value heads tiled from it (`h % NK`), exactly as the host's
/// `kh = vh % n_k_heads`.
///
/// `elem4` bindings: `a` = the token's inputs packed as `q | k | v | beta |
/// decay | z`, `b` = the norm weight, `y` = the layer's scratch — the state
/// at `0`, the output at `NV·HD` — and the meta, whose `extra` is `eps`
/// and whose `aux & 1` selects the grouped store.
pub fn shader_source_gated_delta(hd: u32, n_k: u32, n_v: u32, sigmoid_gate: bool) -> String {
    assert!(hd <= 256 && hd.is_power_of_two() && n_v.is_multiple_of(n_k));
    let gate = if sigmoid_gate {
        "1.0 / (1.0 + exp(-zv))"
    } else {
        "zv / (1.0 + exp(-zv))"
    };
    // A head's `hd × hd` state, one thread per column `j`, walking the
    // rows four at a time. `split` threads per column (each a share of the
    // rows, meeting at two shared-memory reductions) is written in and was
    // measured at the 27B's shape (48 heads of 128): 0.68 ms a layer at
    // one thread per column, 0.6–0.8 at two, 1.3 at four — the workgroup
    // is not short of loads in flight, it is a fixed cost per head that
    // only more *heads* amortize (16 heads 0.29 ms, 192 heads 1.7 ms, the
    // last at 21 GB/s). One thread per column, then. Columns across
    // several workgroups with the head's norm as a dispatch of its own
    // (`shader_source_gated_delta_split` + `_norm`, probe-only) was the
    // next attempt, and measured *warm* (`gated_delta_kernel_time`, best
    // of four at 32 repetitions) the single workgroup is 245 µs a layer at
    // the 27B's shape, not 0.66 ms — the earlier figure was the clock
    // ramping — and the split forms are 250 (128 columns), 253 (64), 264
    // (32), 305 (16): no gain to have. 48 × 245 µs is 12 ms of a ~320 ms
    // token; the kernel is done.
    let split = 1u32;
    assert!((hd / split).is_multiple_of(4));
    let rows = hd / split;
    // The inputs `[q | k | v | beta | decay | z]` sit in the scratch at
    // `em.len`: written there by the host tail already L2-normed and
    // scaled, or by the prep kernel raw (`em.aux & 2`), in which case the
    // head norms its own `q` and `k` on load — `x / max(||x||, eps)`, then
    // `q / sqrt(HD)`, `tensor::l2_norm_inplace` and the trunk's `q_scale`.
    format!(
        r#"{ELEM_META}
@group(0) @binding(0) var<storage, read> unused: array<f32>;
@group(0) @binding(1) var<storage, read> norm_w: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> em: ElemMeta;

const HD: u32 = {hd}u;
const NK: u32 = {n_k}u;
const NV: u32 = {n_v}u;
const SPLIT: u32 = {split}u;
const ROWS: u32 = {rows}u;
const REP: u32 = NV / NK;
const Q_OFF: u32 = 0u;
const K_OFF: u32 = NK * HD;
const V_OFF: u32 = 2u * NK * HD;
const BETA_OFF: u32 = V_OFF + NV * HD;
const DECAY_OFF: u32 = BETA_OFF + NV;
const Z_OFF: u32 = DECAY_OFF + NV;
const OUT_OFF: u32 = NV * HD * HD;
var<workgroup> red: array<f32, HD>;
var<workgroup> qs: array<f32, HD>;
var<workgroup> ks: array<f32, HD>;
var<workgroup> parts: array<f32, {parts_len}>;

// The sum of `v` over the columns, from the `part == 0` threads only; every
// thread of the workgroup calls this (the barriers are uniform).
fn column_sum(j: u32, part: u32, v: f32) -> f32 {{
    if (part == 0u) {{
        red[j] = v;
    }}
    workgroupBarrier();
    var stride: u32 = HD / 2u;
    loop {{
        if (stride == 0u) {{
            break;
        }}
        if (part == 0u && j < stride) {{
            red[j] = red[j] + red[j + stride];
        }}
        workgroupBarrier();
        stride = stride / 2u;
    }}
    let total = red[0];
    workgroupBarrier();
    return total;
}}

// The sum of `v` over the `SPLIT` threads of column `j`.
fn part_sum(j: u32, part: u32, v: f32) -> f32 {{
    parts[part * HD + j] = v;
    workgroupBarrier();
    var total: f32 = 0.0;
    for (var p: u32 = 0u; p < SPLIT; p = p + 1u) {{
        total = total + parts[p * HD + j];
    }}
    workgroupBarrier();
    return total;
}}

@compute @workgroup_size({wg})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let h = wid.x;
    let kh = h % NK;
    let j = lid.x % HD;
    let part = lid.x / HD;
    let base = h * HD * HD;
    let inp = em.len;
    var qj = y[inp + Q_OFF + kh * HD + j];
    var kj = y[inp + K_OFF + kh * HD + j];
    if ((em.aux & 2u) != 0u) {{
        let qn = max(sqrt(column_sum(j, part, qj * qj)), em.extra);
        let kn = max(sqrt(column_sum(j, part, kj * kj)), em.extra);
        qj = qj / qn * inverseSqrt(f32(HD));
        kj = kj / kn;
    }}
    if (part == 0u) {{
        qs[j] = qj;
        ks[j] = kj;
    }}
    workgroupBarrier();
    let beta = y[inp + BETA_OFF + h];
    let decay = y[inp + DECAY_OFF + h];
    let v = y[inp + V_OFF + h * HD + j];
    let i0 = part * ROWS;
    // Four rows a step, loads first: the rows are independent, and issued
    // together they overlap, where one dependent load per row is one
    // round trip per row.
    var sk: f32 = 0.0;
    for (var i: u32 = i0; i < i0 + ROWS; i = i + 4u) {{
        let s0 = y[base + i * HD + j];
        let s1 = y[base + (i + 1u) * HD + j];
        let s2 = y[base + (i + 2u) * HD + j];
        let s3 = y[base + (i + 3u) * HD + j];
        sk = sk + ks[i] * s0 + ks[i + 1u] * s1 + ks[i + 2u] * s2 + ks[i + 3u] * s3;
    }}
    sk = part_sum(j, part, sk * decay);
    let d = beta * (v - sk);
    var o: f32 = 0.0;
    for (var i: u32 = i0; i < i0 + ROWS; i = i + 4u) {{
        let s0 = decay * y[base + i * HD + j] + ks[i] * d;
        let s1 = decay * y[base + (i + 1u) * HD + j] + ks[i + 1u] * d;
        let s2 = decay * y[base + (i + 2u) * HD + j] + ks[i + 2u] * d;
        let s3 = decay * y[base + (i + 3u) * HD + j] + ks[i + 3u] * d;
        y[base + i * HD + j] = s0;
        y[base + (i + 1u) * HD + j] = s1;
        y[base + (i + 2u) * HD + j] = s2;
        y[base + (i + 3u) * HD + j] = s3;
        o = o + qs[i] * s0 + qs[i + 1u] * s1 + qs[i + 2u] * s2 + qs[i + 3u] * s3;
    }}
    o = part_sum(j, part, o);
    // Gated RMSNorm over the head: rms of `o` across the columns.
    let inv = inverseSqrt(column_sum(j, part, o * o) / f32(HD) + em.extra);
    if (part == 0u) {{
        let zv = y[inp + Z_OFF + h * HD + j];
        let gate = {gate};
        let out_head = select(h, (h % NK) * REP + h / NK, (em.aux & 1u) != 0u);
        y[OUT_OFF + out_head * HD + j] = o * inv * norm_w[j] * gate;
    }}
}}
"#,
        wg = hd * split,
        parts_len = hd * split,
    )
}

/// [`shader_source_gated_delta`] with a head's columns spread over
/// `hd / cols` workgroups of `cols` threads: `n_v * hd / cols` workgroups,
/// so the device has more of them in flight than it has heads. Every
/// workgroup loads the head's whole `q` and `k` (and norms them itself
/// when the inputs are raw — a redundancy of `2 · hd` elements per
/// workgroup), walks its columns' rows once for `k·S` and once for the
/// update, and leaves the head's *un-normed* output `o` in the `v` slot
/// of the inputs (each thread's own, read before it is written); the
/// gated RMSNorm, which needs every column of the head, is
/// [`shader_source_gated_delta_norm`]'s dispatch after it. `cols == hd`
/// is the single-workgroup kernel's shape with the norm split off, for
/// the A/B. Probe-only: measured no faster than the single workgroup
/// (see there).
#[cfg(test)]
pub fn shader_source_gated_delta_split(hd: u32, n_k: u32, n_v: u32, cols: u32) -> String {
    assert!(hd <= 256 && hd.is_power_of_two() && n_v.is_multiple_of(n_k));
    assert!(cols.is_power_of_two() && cols <= hd && hd.is_multiple_of(4));
    format!(
        r#"{ELEM_META}
@group(0) @binding(0) var<storage, read> unused: array<f32>;
@group(0) @binding(1) var<storage, read> norm_w: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> em: ElemMeta;

const HD: u32 = {hd}u;
const NK: u32 = {n_k}u;
const NV: u32 = {n_v}u;
const COLS: u32 = {cols}u;
const CS: u32 = HD / COLS;
const Q_OFF: u32 = 0u;
const K_OFF: u32 = NK * HD;
const V_OFF: u32 = 2u * NK * HD;
const BETA_OFF: u32 = V_OFF + NV * HD;
const DECAY_OFF: u32 = BETA_OFF + NV;
var<workgroup> red: array<f32, COLS>;
var<workgroup> qs: array<f32, HD>;
var<workgroup> ks: array<f32, HD>;

// The sum of `v` over the workgroup; every thread calls it.
fn wg_sum(t: u32, v: f32) -> f32 {{
    red[t] = v;
    workgroupBarrier();
    var stride: u32 = COLS / 2u;
    loop {{
        if (stride == 0u) {{
            break;
        }}
        if (t < stride) {{
            red[t] = red[t] + red[t + stride];
        }}
        workgroupBarrier();
        stride = stride / 2u;
    }}
    let total = red[0];
    workgroupBarrier();
    return total;
}}

@compute @workgroup_size({cols})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let h = wid.x / CS;
    let c = wid.x % CS;
    let kh = h % NK;
    let t = lid.x;
    let j = c * COLS + t;
    let base = h * HD * HD;
    let inp = em.len;
    for (var i: u32 = t; i < HD; i = i + COLS) {{
        qs[i] = y[inp + Q_OFF + kh * HD + i];
        ks[i] = y[inp + K_OFF + kh * HD + i];
    }}
    workgroupBarrier();
    if ((em.aux & 2u) != 0u) {{
        var sq: f32 = 0.0;
        var sk2: f32 = 0.0;
        for (var i: u32 = t; i < HD; i = i + COLS) {{
            sq = sq + qs[i] * qs[i];
            sk2 = sk2 + ks[i] * ks[i];
        }}
        let qsc = inverseSqrt(f32(HD)) / max(sqrt(wg_sum(t, sq)), em.extra);
        let ksc = 1.0 / max(sqrt(wg_sum(t, sk2)), em.extra);
        for (var i: u32 = t; i < HD; i = i + COLS) {{
            qs[i] = qs[i] * qsc;
            ks[i] = ks[i] * ksc;
        }}
        workgroupBarrier();
    }}
    let beta = y[inp + BETA_OFF + h];
    let decay = y[inp + DECAY_OFF + h];
    let v = y[inp + V_OFF + h * HD + j];
    var sk: f32 = 0.0;
    for (var i: u32 = 0u; i < HD; i = i + 4u) {{
        let s0 = y[base + i * HD + j];
        let s1 = y[base + (i + 1u) * HD + j];
        let s2 = y[base + (i + 2u) * HD + j];
        let s3 = y[base + (i + 3u) * HD + j];
        sk = sk + ks[i] * s0 + ks[i + 1u] * s1 + ks[i + 2u] * s2 + ks[i + 3u] * s3;
    }}
    let d = beta * (v - sk * decay);
    var o: f32 = 0.0;
    for (var i: u32 = 0u; i < HD; i = i + 4u) {{
        let s0 = decay * y[base + i * HD + j] + ks[i] * d;
        let s1 = decay * y[base + (i + 1u) * HD + j] + ks[i + 1u] * d;
        let s2 = decay * y[base + (i + 2u) * HD + j] + ks[i + 2u] * d;
        let s3 = decay * y[base + (i + 3u) * HD + j] + ks[i + 3u] * d;
        y[base + i * HD + j] = s0;
        y[base + (i + 1u) * HD + j] = s1;
        y[base + (i + 2u) * HD + j] = s2;
        y[base + (i + 3u) * HD + j] = s3;
        o = o + qs[i] * s0 + qs[i + 1u] * s1 + qs[i + 2u] * s2 + qs[i + 3u] * s3;
    }}
    // The head's un-normed output, in this thread's own `v` slot.
    y[inp + V_OFF + h * HD + j] = o;
}}
"#
    )
}

/// The gated RMSNorm over each head's output that
/// [`shader_source_gated_delta_split`] leaves in the `v` slots: one
/// workgroup of `hd` threads per head, the same bindings, the result in
/// the delta kernel's output layout (with the head permutation of
/// `em.aux & 1`). Probe-only, with the split.
#[cfg(test)]
pub fn shader_source_gated_delta_norm(hd: u32, n_k: u32, n_v: u32, sigmoid_gate: bool) -> String {
    let gate = if sigmoid_gate {
        "1.0 / (1.0 + exp(-zv))"
    } else {
        "zv / (1.0 + exp(-zv))"
    };
    format!(
        r#"{ELEM_META}
@group(0) @binding(0) var<storage, read> unused: array<f32>;
@group(0) @binding(1) var<storage, read> norm_w: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> em: ElemMeta;

const HD: u32 = {hd}u;
const NK: u32 = {n_k}u;
const NV: u32 = {n_v}u;
const REP: u32 = NV / NK;
const V_OFF: u32 = 2u * NK * HD;
const Z_OFF: u32 = V_OFF + NV * HD + 2u * NV;
const OUT_OFF: u32 = NV * HD * HD;
var<workgroup> red: array<f32, HD>;

@compute @workgroup_size({hd})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let h = wid.x;
    let j = lid.x;
    let inp = em.len;
    let o = y[inp + V_OFF + h * HD + j];
    red[j] = o * o;
    workgroupBarrier();
    var stride: u32 = HD / 2u;
    loop {{
        if (stride == 0u) {{
            break;
        }}
        if (j < stride) {{
            red[j] = red[j] + red[j + stride];
        }}
        workgroupBarrier();
        stride = stride / 2u;
    }}
    let inv = inverseSqrt(red[0] / f32(HD) + em.extra);
    let zv = y[inp + Z_OFF + h * HD + j];
    let gate = {gate};
    let out_head = select(h, (h % NK) * REP + h / NK, (em.aux & 1u) != 0u);
    y[OUT_OFF + out_head * HD + j] = o * inv * norm_w[j] * gate;
}}
"#
    )
}

/// The device half of a recurrent layer's step between its projections
/// and the delta rule (`shader_source_gated_delta`), one thread per
/// element: the causal conv1d over the QKV mix against the rolling history
/// kept in the scratch (`Recurrent LayerState::conv_step` — the history's
/// `d_conv - 1` taps then the current input, and the window slid by one),
/// SiLU, `beta = sigmoid(b)`, `decay = exp(softplus(a + dt_bias) · A)`, and
/// the gate `z` copied — each landing in the delta kernel's input layout
/// `[q | k | v | beta | decay | z]` at `em.len` of the scratch, the history
/// at `em.aux`.
///
/// Bindings: 0 the projections `[qkv mix | z | beta | alpha]` as the four
/// matmuls left them, copied together; 1 the layer's constants `[conv
/// kernel (channel-major, `d_conv` fastest) | dt_bias | A]`; 2 the scratch;
/// 3 the meta. `em.extra` is unused.
pub fn shader_source_recurrent_prep(
    conv_channels: u32,
    d_conv: u32,
    value_dim: u32,
    n_v: u32,
) -> String {
    format!(
        r#"{ELEM_META}
@group(0) @binding(0) var<storage, read> proj: array<f32>;
@group(0) @binding(1) var<storage, read> consts: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> em: ElemMeta;

const CC: u32 = {conv_channels}u;
const DC: u32 = {d_conv}u;
const HW: u32 = DC - 1u;
const VD: u32 = {value_dim}u;
const NV: u32 = {n_v}u;
// The projections, back to back.
const P_Z: u32 = CC;
const P_BETA: u32 = CC + VD;
const P_ALPHA: u32 = CC + VD + NV;
// The constants.
const C_DT: u32 = CC * DC;
const C_A: u32 = CC * DC + NV;
// The delta inputs, relative to `em.len`.
const I_BETA: u32 = CC;
const I_DECAY: u32 = CC + NV;
const I_Z: u32 = CC + 2u * NV;

fn softplus(x: f32) -> f32 {{
    return select(log(1.0 + exp(x)), x, x > 20.0);
}}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let inp = em.len;
    let hist = em.aux;
    if (i < CC) {{
        let x = proj[i];
        var sum: f32 = 0.0;
        for (var t: u32 = 0u; t < HW; t = t + 1u) {{
            sum = sum + y[hist + i * HW + t] * consts[i * DC + t];
        }}
        sum = sum + x * consts[i * DC + HW];
        for (var t: u32 = 1u; t < HW; t = t + 1u) {{
            y[hist + i * HW + t - 1u] = y[hist + i * HW + t];
        }}
        if (HW > 0u) {{
            y[hist + i * HW + HW - 1u] = x;
        }}
        y[inp + i] = sum / (1.0 + exp(-sum));
    }} else if (i < CC + NV) {{
        let h = i - CC;
        y[inp + I_BETA + h] = 1.0 / (1.0 + exp(-proj[P_BETA + h]));
    }} else if (i < CC + 2u * NV) {{
        let h = i - CC - NV;
        y[inp + I_DECAY + h] = exp(softplus(proj[P_ALPHA + h] + consts[C_DT + h]) * consts[C_A + h]);
    }} else if (i < CC + 2u * NV + VD) {{
        let e = i - CC - 2u * NV;
        y[inp + I_Z + e] = proj[P_Z + e];
    }}
}}
"#
    )
}

/// Threads per Hadamard workgroup — half the smallest block this will be
/// asked for (`prism.hadamard.block_size` is 1024), and inside every
/// device's 256-invocation floor.
pub const HADAMARD_WG: u32 = 256;

pub fn shader_source_bias_add() -> String {
    format!("{ELEM_META}\n{BIAS_ADD_SHADER_BODY}")
}

/// `y[i] = a[i] + b[i]`, e.g. a residual add.
const ADD_SHADER_BODY: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> em: ElemMeta;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= em.len) {
        return;
    }
    y[i] = a[i] + b[i];
}
"#;

/// `y[i] = a[i] * b[i]` — GEGLU's gate/up combine, and PLE's per-layer gate.
const MUL_SHADER_BODY: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> em: ElemMeta;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= em.len) {
        return;
    }
    y[i] = a[i] * b[i];
}
"#;

/// Line-for-line port of `engine::tensor::gelu` (the tanh approximation, not the
/// exact erf form). WGSL `tanh` lowers to the SPIR-V `GLSLstd450 Tanh` ExtInst —
/// the hardware SFU, no `exp` polyfill (verified: naga `back/spv/block.rs` maps
/// `Mf::Tanh` straight to it). A `vec4` (4-elements-per-thread) rewrite of this
/// and `MUL_SHADER_BODY` was also tried; it measured *slower* because
/// this decode-shaped dispatch is launch/occupancy-bound, not compute-bound —
/// quartering the thread count underfills the GPU. Kept scalar; see
/// `_scratch_measure_ffn_elementwise_dispatch_cost`.
///
/// # Why the argument is clamped
///
/// The cubic inside `tanh` grows fast: `|v| = 1000` already asks for
/// `tanh(3.6e7)`. That is fine on a hardware SFU, which saturates, and fine
/// in Rust (`f32::tanh`), which is what `engine::tensor::gelu` uses. It is
/// **not** fine wherever `tanh` is lowered to `(exp(2x) - 1) / (exp(2x) + 1)`:
/// `exp` overflows to `inf` and `inf / inf` is `NaN`. Measured on `wgpu`'s
/// Metal backend, where every GELU-path cross-check returned `NaN` while
/// every SwiGLU one passed — SiLU's `v / (1 + exp(-v))` has no such form,
/// which is exactly why the two split.
///
/// `tanh` is already saturated to `±1.0` in `f32` well below `±20`, so the
/// clamp changes no representable result: it is the same function, minus an
/// overflow. Applied unconditionally rather than per-backend, because a
/// kernel that returns `NaN` for large-but-finite input is wrong everywhere
/// — it only happened to be unobservable on the one driver this was
/// developed against.
const GELU_SHADER_BODY: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> em: ElemMeta;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= em.len) {
        return;
    }
    let v = x[i];
    let sqrt_2_over_pi = 0.7978846;
    let coef_a = 0.044715;
    y[i] = 0.5 * v * (1.0 + tanh(clamp(sqrt_2_over_pi * v * (1.0 + coef_a * v * v), -20.0, 20.0)));
}
"#;

/// Fused GELU+multiply (dispatch fusion): `y[i] = gelu(a[i]) * b[i]` in
/// one dispatch instead of a separate GELU pass (writing a `gelu_out` scratch)
/// then a MUL pass. `a` = the gate projection, `b` = the up projection; byte-
/// identical to running `GELU_SHADER_BODY` then `MUL_SHADER_BODY` (same tanh
/// gelu, same multiply order) — the only difference is the intermediate stays in
/// a register rather than a round-trip through VRAM. Same `elem4` binding shape
/// as `MUL_SHADER_BODY`, so it reuses `elem4_bind_group`.
const GELU_MUL_SHADER_BODY: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> em: ElemMeta;

// Grid-stride: a dispatch may carry fewer workgroups than elements / 64
// (a device's workgroups-per-dimension limit is what bounds it), and each
// thread walks on by the grid's width until the end.
@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    var i: u32 = gid.x;
    let stride = nwg.x * 64u;
    loop {
        if (i >= em.len) {
            break;
        }
        let v = a[i];
        let sqrt_2_over_pi = 0.7978846;
        let coef_a = 0.044715;
        let g = 0.5 * v * (1.0 + tanh(clamp(sqrt_2_over_pi * v * (1.0 + coef_a * v * v), -20.0, 20.0)));
        y[i] = g * b[i];
        i = i + stride;
    }
}
"#;

/// `y[i] = x[i] * em.extra` — gemma4's per-layer `layer_output_scale`.
const SCALE_SHADER_BODY: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> em: ElemMeta;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= em.len) {
        return;
    }
    y[i] = x[i] * em.extra;
}
"#;

/// Weighted RMSNorm over a *single row* of `em.len` elements (this fused
/// path only ever runs at `n_tokens == 1` — decode), dispatched as exactly
/// one workgroup: all 64 threads grid-stride over the row to build a
/// partial sum of squares, tree-reduce it in `partial_sums` (same reduction
/// shape as `MAIN_REDUCE_SUFFIX`'s dot-product reduction), then every
/// thread rescales its own elements by the shared result — line-for-line
/// the same formula as `engine::tensor::rmsnorm_inplace`.
// Tree-reduce RMSNorm body, parameterized by workgroup size via `%WG%`
// (thread count / grid-stride) and `%HALF%` (the reduction's initial
// stride, `%WG% / 2`). A single workgroup grid-strides the whole `em.len`
// row, so more threads means fewer sequential iterations per thread and
// more SIMDs busy on this otherwise occupancy-starved
// `dispatch_workgroups(1,1,1)` norm — see `VulkanBackend::norm_wg`. `%WG%`
// must be a power of two.
const RMSNORM_SHADER_BODY_TEMPLATE: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> em: ElemMeta;

var<workgroup> partial_sums: array<f32, %WG%>;

@compute @workgroup_size(%WG%)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let local = lid.x;
    var partial: f32 = 0.0;
    var k: u32 = local;
    loop {
        if (k >= em.len) {
            break;
        }
        let v = x[k];
        partial = partial + v * v;
        k = k + %WG%u;
    }
    partial_sums[local] = partial;
    workgroupBarrier();
    var stride: u32 = %HALF%u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (local < stride) {
            partial_sums[local] = partial_sums[local] + partial_sums[local + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let mean_sq = partial_sums[0] / f32(em.len);
    let scale = 1.0 / sqrt(mean_sq + em.extra);
    k = local;
    loop {
        if (k >= em.len) {
            break;
        }
        y[k] = x[k] * scale * weight[k];
        k = k + %WG%u;
    }
}
"#;

/// Substitutes `%WG%`/`%HALF%` in a tree-reduce norm body template for a
/// concrete (power-of-two) workgroup size.
fn norm_body_for_wg(template: &str, wg: usize) -> String {
    template
        .replace("%WG%", &wg.to_string())
        .replace("%HALF%", &(wg / 2).to_string())
}

/// `RMSNORM_SHADER_BODY` with `subgroupAdd` replacing the 6-round
/// tree — see `reduce_combine_block`'s doc comment for the general-
/// subgroup-size rationale. Unlike the reduce kernels above (only lane 0
/// needs the combined total, to write `y`), every lane here needs the
/// combined `mean_sq`/`scale` to rescale its own slice of the row — so
/// instead of a second `if (local == 0u) { combine }` + barrier, every lane
/// just runs the same tiny (`num_subgroups`-long, ≤64, and 1 on hardware
/// where the subgroup already spans the whole workgroup) combine loop
/// itself. That keeps this at exactly one barrier — the one that makes each
/// subgroup's `subgroupAdd` partial visible workgroup-wide — the same
/// barrier count the fully-single-subgroup case would need anyway.
const RMSNORM_SHADER_BODY_SUBGROUP: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> em: ElemMeta;

var<workgroup> partial_sums: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_lane: u32,
    @builtin(subgroup_id) sg_id: u32,
    @builtin(num_subgroups) n_sg: u32,
) {
    let local = lid.x;
    var partial: f32 = 0.0;
    var k: u32 = local;
    loop {
        if (k >= em.len) {
            break;
        }
        let v = x[k];
        partial = partial + v * v;
        k = k + 64u;
    }
    let sg_sum = subgroupAdd(partial);
    if (sg_lane == 0u) {
        partial_sums[sg_id] = sg_sum;
    }
    workgroupBarrier();
    var total: f32 = 0.0;
    var i: u32 = 0u;
    loop {
        if (i >= n_sg) {
            break;
        }
        total = total + partial_sums[i];
        i = i + 1u;
    }
    let mean_sq = total / f32(em.len);
    let scale = 1.0 / sqrt(mean_sq + em.extra);
    k = local;
    loop {
        if (k >= em.len) {
            break;
        }
        y[k] = x[k] * scale * weight[k];
        k = k + 64u;
    }
}
"#;

pub fn shader_source_add() -> String {
    format!("{ELEM_META}\n{ADD_SHADER_BODY}")
}

pub fn shader_source_mul() -> String {
    format!("{ELEM_META}\n{MUL_SHADER_BODY}")
}

pub fn shader_source_gelu() -> String {
    format!("{ELEM_META}\n{GELU_SHADER_BODY}")
}

/// Fused SiLU+multiply — the SwiGLU counterpart of [`GELU_MUL_SHADER_BODY`],
/// for the Llama/Qwen2/Mistral/Phi families, whose FFN gate is
/// `silu(gate) * up` rather than gemma's `gelu(gate) * up`.
///
/// Line-for-line the same shape as the GELU twin (same bindings, same
/// `elem4_bind_group`, same workgroup size), so selecting between them is a
/// pipeline swap and nothing else. `silu(v) = v * sigmoid(v)`, written as
/// `v / (1 + exp(-v))` to match `engine::tensor::silu` exactly rather than
/// approximately — this kernel is cross-checked against that CPU function, and
/// an algebraically-equal-but-differently-rounded form would make the tolerance
/// carry a difference that is not the one being tested.
const SILU_MUL_SHADER_BODY: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> em: ElemMeta;

// Grid-stride: a dispatch may carry fewer workgroups than elements / 64
// (a device's workgroups-per-dimension limit is what bounds it), and each
// thread walks on by the grid's width until the end.
@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    var i: u32 = gid.x;
    let stride = nwg.x * 64u;
    loop {
        if (i >= em.len) {
            break;
        }
        let v = a[i];
        let g = v / (1.0 + exp(-v));
        y[i] = g * b[i];
        i = i + stride;
    }
}
"#;

pub fn shader_source_silu_mul() -> String {
    format!("{ELEM_META}\n{SILU_MUL_SHADER_BODY}")
}

/// Squared ReLU — `y[i] = max(x[i], 0)²`, the gate-less expert activation
/// (`nemotron_h_moe`'s `down(relu(up(x))²)`). Grid-stride like the paired
/// activations, so one dispatch covers a routed batch of any width.
/// `elem3` bindings: the up projection, the product, the meta.
const RELU_SQUARED_SHADER_BODY: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> em: ElemMeta;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    var i: u32 = gid.x;
    let stride = nwg.x * 64u;
    loop {
        if (i >= em.len) {
            break;
        }
        let v = max(x[i], 0.0);
        y[i] = v * v;
        i = i + stride;
    }
}
"#;

pub fn shader_source_relu_squared() -> String {
    format!("{ELEM_META}\n{RELU_SQUARED_SHADER_BODY}")
}

/// Attention's output gated **in place** by the sigmoid of its own
/// projection — `x[i] *= sigmoid(gate[i])`, the gate `muse-glimmer` (and
/// the Qwen 3.5 family's full-attention layers) applies before `wo`.
/// `elem3` bindings: the gate read-only, the rows read-write, the meta.
/// Written as `1 / (1 + exp(-g))` to match `engine::tensor::sigmoid`.
const SIGMOID_GATE_SHADER_BODY: &str = r#"
@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<uniform> em: ElemMeta;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= em.len) {
        return;
    }
    x[i] = x[i] * (1.0 / (1.0 + exp(-gate[i])));
}
"#;

pub fn shader_source_sigmoid_gate() -> String {
    format!("{ELEM_META}\n{SIGMOID_GATE_SHADER_BODY}")
}

pub fn shader_source_gelu_mul() -> String {
    format!("{ELEM_META}\n{GELU_MUL_SHADER_BODY}")
}

/// The routed experts' outputs combined per token: `y[t] = sum over the
/// token's picks r of w[t][r] * rows[slot[t][r]]`, where `rows` is the
/// down projection's `[total, n_embd]` result in group order, `table`
/// holds every token's `k` row slots and then, as `f32` bits, its `k`
/// weights (the routing weight times any per-expert output scale; `0` for
/// a pick the token does not have). `elem4` bindings: rows, table, the
/// combined rows, the meta — `len = n_tokens * n_embd`, `aux = n_embd`,
/// `extra = k`. Grid-stride like the activations.
const MOE_COMBINE_SHADER_BODY: &str = r#"
@group(0) @binding(0) var<storage, read> rows: array<f32>;
@group(0) @binding(1) var<storage, read> table: array<u32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> em: ElemMeta;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let n_embd = em.aux;
    let k = u32(em.extra);
    let n_tokens = em.len / n_embd;
    var i: u32 = gid.x;
    let stride = nwg.x * 64u;
    loop {
        if (i >= em.len) {
            break;
        }
        let t = i / n_embd;
        let e = i % n_embd;
        var acc: f32 = 0.0;
        var r: u32 = 0u;
        loop {
            if (r >= k) { break; }
            let slot = table[t * k + r];
            let w = bitcast<f32>(table[n_tokens * k + t * k + r]);
            acc = acc + w * rows[slot * n_embd + e];
            r = r + 1u;
        }
        y[i] = acc;
        i = i + stride;
    }
}
"#;

pub fn shader_source_moe_combine() -> String {
    format!("{ELEM_META}\n{MOE_COMBINE_SHADER_BODY}")
}

/// The fused activation+multiply that also writes the product's 8-bit form
/// (`shader_source_quantize_q8`'s layout, binding 3) for a down projection
/// on the integer dot: each 64-thread workgroup owns 64 consecutive
/// elements — two 32-element blocks — so a block's scale and quant sum are
/// two 32-lane reductions through workgroup memory. One workgroup per 64
/// elements (`len` a multiple of 32), no grid stride. `silu` selects the
/// SwiGLU form. Two outputs, so it sits on the norm pair's layout: `a` at
/// 0, `b` at 2, the product at 4, its q8 at 5, the meta at 6; 1, 3 and 7
/// take placeholders.
pub fn shader_source_activation_mul_q8(silu: bool) -> String {
    let act = if silu {
        "let g = v / (1.0 + exp(-v));"
    } else {
        "let sqrt_2_over_pi = 0.7978846;\n        let coef_a = 0.044715;\n        let g = 0.5 * v * (1.0 + tanh(clamp(sqrt_2_over_pi * v * (1.0 + coef_a * v * v), -20.0, 20.0)));"
    };
    format!(
        r#"{ELEM_META}
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<storage, read_write> q8: array<u32>;
@group(0) @binding(6) var<uniform> em: ElemMeta;

var<workgroup> amax: array<f32, 64>;
var<workgroup> sums: array<i32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    let i = gid.x;
    let local = lid.x;
    let inb = i < em.len;
    var p: f32 = 0.0;
    if (inb) {{
        let v = a[i];
        {act}
        p = g * b[i];
        y[i] = p;
    }}
    amax[local] = abs(p);
    workgroupBarrier();
    let c = (local / 32u) * 32u;
    var m: f32 = 0.0;
    var j: u32 = 0u;
    loop {{
        if (j >= 32u) {{ break; }}
        m = max(m, amax[c + j]);
        j = j + 1u;
    }}
    let d = m / 127.0;
    let id = select(0.0, 1.0 / d, d > 0.0);
    let q = clamp(i32(round(p * id)), -127, 127);
    sums[local] = q;
    workgroupBarrier();
    // Four lanes pack a word each; lane 0 of the block its header.
    let blk = i / 32u;
    let e = local % 32u;
    if (inb && (e % 4u) == 0u) {{
        let word = (u32(sums[local]) & 0xFFu) | ((u32(sums[local + 1u]) & 0xFFu) << 8u)
            | ((u32(sums[local + 2u]) & 0xFFu) << 16u) | ((u32(sums[local + 3u]) & 0xFFu) << 24u);
        q8[blk * 10u + 2u + e / 4u] = word;
    }}
    if (inb && e == 0u) {{
        var sum: i32 = 0;
        j = 0u;
        loop {{
            if (j >= 32u) {{ break; }}
            sum = sum + sums[c + j];
            j = j + 1u;
        }}
        q8[blk * 10u] = bitcast<u32>(d);
        q8[blk * 10u + 1u] = bitcast<u32>(sum);
    }}
}}
"#
    )
}

pub fn shader_source_scale() -> String {
    format!("{ELEM_META}\n{SCALE_SHADER_BODY}")
}

pub fn shader_source_rmsnorm(subgroup: bool, wg: usize) -> String {
    if subgroup {
        // The subgroup variant's own reduction is fixed to a 64-thread
        // workgroup; `wg` only tunes the default tree-reduce path.
        format!("{ELEM_META}\n{RMSNORM_SHADER_BODY_SUBGROUP}")
    } else {
        format!(
            "{ELEM_META}\n{}",
            norm_body_for_wg(RMSNORM_SHADER_BODY_TEMPLATE, wg)
        )
    }
}

/// `RMSNORM_SHADER_BODY_SUBGROUP` at a caller-chosen workgroup width —
/// tests whether a narrower `workgroup_size` (matching a GPU's native
/// subgroup/wavefront width) lets each workgroup fit in exactly one
/// subgroup, the same way llama.cpp's `USE_SUBGROUP_ADD_NO_SHMEM`
/// specifically skips its cross-subgroup merge/barrier when the workgroup
/// already fits in one subgroup — unlike the fixed 64-wide `RMSNORM_
/// SHADER_BODY_SUBGROUP` above, which always needs one whenever a
/// workgroup spans more than one subgroup. `%WG_SIZE%` substitutes both
/// the `@workgroup_size` attribute and the
/// grid-stride loops' stride — the reduction logic itself (per-subgroup
/// `subgroupAdd`, then every lane redundantly re-summing the `n_sg`-long
/// `partial_sums` combine) is already general to any subgroup count, not
/// touched here. `partial_sums` stays fixed at 64 slots regardless — a
/// safe upper bound (`num_subgroups <= workgroup_size <= 64`) for every
/// `workgroup_size` this is ever called with.
#[allow(dead_code)]
const RMSNORM_SHADER_BODY_SUBGROUP_WG_TEMPLATE: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> em: ElemMeta;

var<workgroup> partial_sums: array<f32, 64>;

@compute @workgroup_size(%WG_SIZE%)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_lane: u32,
    @builtin(subgroup_id) sg_id: u32,
    @builtin(num_subgroups) n_sg: u32,
) {
    let local = lid.x;
    var partial: f32 = 0.0;
    var k: u32 = local;
    loop {
        if (k >= em.len) {
            break;
        }
        let v = x[k];
        partial = partial + v * v;
        k = k + %WG_SIZE%u;
    }
    let sg_sum = subgroupAdd(partial);
    if (sg_lane == 0u) {
        partial_sums[sg_id] = sg_sum;
    }
    workgroupBarrier();
    var total: f32 = 0.0;
    var i: u32 = 0u;
    loop {
        if (i >= n_sg) {
            break;
        }
        total = total + partial_sums[i];
        i = i + 1u;
    }
    let mean_sq = total / f32(em.len);
    let scale = 1.0 / sqrt(mean_sq + em.extra);
    k = local;
    loop {
        if (k >= em.len) {
            break;
        }
        y[k] = x[k] * scale * weight[k];
        k = k + %WG_SIZE%u;
    }
}
"#;

/// Only ever called from the `#[ignore]`d scratch benchmark
/// (`VulkanBackend::_scratch_measure_rmsnorm_workgroup_size_and_subgroup`)
/// — **not** wired into `try_init`'s own pipeline set. The RMSNorm
/// dispatch is a single workgroup (`dispatch_workgroups(1, 1, 1)`)
/// covering the whole row via a grid-stride loop, so halving
/// `workgroup_size` halves the thread count doing that loop — twice the
/// sequential iterations per thread — with no offsetting barrier/merge
/// cost avoided, since the 64-wide subgroup variant's cross-subgroup
/// combine is already cheap next to the raw compute either way.
#[allow(dead_code)]
pub fn shader_source_rmsnorm_subgroup_wg(workgroup_size: u32) -> String {
    let body =
        RMSNORM_SHADER_BODY_SUBGROUP_WG_TEMPLATE.replace("%WG_SIZE%", &workgroup_size.to_string());
    format!("{ELEM_META}\n{body}")
}

/// `RMSNORM_SHADER_BODY_SUBGROUP` with the trailing rescale loop's write
/// changed to `y[k] = x[k] * scale * weight[k] + residual[k]` — RMSNorm
/// immediately followed by a residual add, in one dispatch instead of two
/// (`rmsnorm_pipeline` then `add_pipeline`). Only safe to merge this way
/// because both steps are single-workgroup, whole-row operations already
/// (`dispatch_workgroups(1, 1, 1)`, every one of the 64 threads
/// grid-striding the *entire* row) — the add's own per-thread output slice
/// exactly matches the norm's own, so no new cross-thread dependency is
/// introduced by folding the add into the same trailing loop. This is
/// *not* the same kind of fusion as folding a matmul in: the matmul that
/// produces `x` here is dispatched across many independent workgroups (one
/// per `REDUCE_N_ROWS`-row group, for occupancy), and there is no
/// cross-workgroup barrier in a single dispatch to make that matmul's own
/// output visible to a fused norm+add before every one of *those*
/// workgroups has finished — that would need collapsing the matmul itself
/// down to one workgroup, trading its current many-workgroup occupancy for
/// dispatch-count savings with an unclear (likely negative) net effect;
/// not attempted.
/// Needs its own bind group shape (`elem5_bind_group_layout`): `elem4`'s
/// four bindings (`x`, `weight`, `y`, `meta`) aren't enough room for the
/// extra `residual` input.
const RMSNORM_ADD_SHADER_BODY_SUBGROUP: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> residual: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> em: ElemMeta;

var<workgroup> partial_sums: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_lane: u32,
    @builtin(subgroup_id) sg_id: u32,
    @builtin(num_subgroups) n_sg: u32,
) {
    let local = lid.x;
    var partial: f32 = 0.0;
    var k: u32 = local;
    loop {
        if (k >= em.len) {
            break;
        }
        let v = x[k];
        partial = partial + v * v;
        k = k + 64u;
    }
    let sg_sum = subgroupAdd(partial);
    if (sg_lane == 0u) {
        partial_sums[sg_id] = sg_sum;
    }
    workgroupBarrier();
    var total: f32 = 0.0;
    var i: u32 = 0u;
    loop {
        if (i >= n_sg) {
            break;
        }
        total = total + partial_sums[i];
        i = i + 1u;
    }
    let mean_sq = total / f32(em.len);
    let scale = 1.0 / sqrt(mean_sq + em.extra);
    k = local;
    loop {
        if (k >= em.len) {
            break;
        }
        y[k] = x[k] * scale * weight[k] + residual[k];
        k = k + 64u;
    }
}
"#;

/// `RMSNORM_SHADER_BODY`'s shared-memory-tree-reduction fallback, fused
/// with the residual add the same way `RMSNORM_ADD_SHADER_BODY_SUBGROUP`
/// is — used when `subgroupAdd` isn't available. See that constant's own
/// doc comment for why this fusion is safe and what it deliberately
/// doesn't attempt.
// Tree-reduce RMSNorm+residual-add body — see `RMSNORM_SHADER_BODY_TEMPLATE`
// for the `%WG%`/`%HALF%` workgroup-size parameterization.
const RMSNORM_ADD_SHADER_BODY_TEMPLATE: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> residual: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> em: ElemMeta;

var<workgroup> partial_sums: array<f32, %WG%>;

@compute @workgroup_size(%WG%)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let local = lid.x;
    var partial: f32 = 0.0;
    var k: u32 = local;
    loop {
        if (k >= em.len) {
            break;
        }
        let v = x[k];
        partial = partial + v * v;
        k = k + %WG%u;
    }
    partial_sums[local] = partial;
    workgroupBarrier();
    var stride: u32 = %HALF%u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (local < stride) {
            partial_sums[local] = partial_sums[local] + partial_sums[local + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let mean_sq = partial_sums[0] / f32(em.len);
    let scale = 1.0 / sqrt(mean_sq + em.extra);
    k = local;
    loop {
        if (k >= em.len) {
            break;
        }
        y[k] = x[k] * scale * weight[k] + residual[k];
        k = k + %WG%u;
    }
}
"#;

/// See `RMSNORM_ADD_SHADER_BODY_SUBGROUP`'s own doc comment — RMSNorm
/// fused with the residual add that already always immediately follows it
/// at both of this codebase's two call sites (`wo`'s and `ffn_down`'s own
/// post-matmul norm+add, `VulkanBackend::build_fused_resources`), removing
/// one dispatch (`add_pipeline`'s own) from each.
/// The two norms of a prefill layer's post-attention chain, row-strided:
/// **one workgroup per token**, each normalising its own `[n_embd]` row at
/// `wid.x * em.len`. The decode-path bodies above normalise a single row and
/// are dispatched `(1, 1, 1)`; a prefill chain has `n_tokens` rows to do, and
/// running them as `n_tokens` separate dispatches (or separate submissions,
/// as the CPU-orchestrated path did) is exactly the round-trip cost
/// `VulkanBackend::fused_post_attention_prefill` exists to remove.
///
/// `weight` is indexed *without* the row offset — it is per-column, shared by
/// every row — which is the one thing a naive "add a base offset everywhere"
/// transformation of the single-row bodies would get wrong.
///
/// The tree reduction is used unconditionally rather than the `subgroupAdd`
/// variant the single-row bodies pick between: these dispatch `n_tokens`
/// workgroups, so the reduction is a small part of the work, and one body is
/// one thing to keep correct.
const RMSNORM_ROWS_SHADER_BODY: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> em: ElemMeta;

var<workgroup> partial_sums: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let base = wid.x * em.len;
    let local = lid.x;
    var partial: f32 = 0.0;
    var k: u32 = local;
    loop {
        if (k >= em.len) {
            break;
        }
        let v = x[base + k];
        partial = partial + v * v;
        k = k + 64u;
    }
    partial_sums[local] = partial;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (local < stride) {
            partial_sums[local] = partial_sums[local] + partial_sums[local + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    let mean_sq = partial_sums[0] / f32(em.len);
    let scale = 1.0 / sqrt(mean_sq + em.extra);
    k = local;
    loop {
        if (k >= em.len) {
            break;
        }
        y[base + k] = x[base + k] * scale * weight[k];
        k = k + 64u;
    }
}
"#;

/// [`RMSNORM_ROWS_SHADER_BODY`] with the residual add folded in, the
/// row-strided counterpart of `RMSNORM_ADD_SHADER_BODY_SUBGROUP`.
const RMSNORM_ADD_ROWS_SHADER_BODY: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> residual: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> em: ElemMeta;

var<workgroup> partial_sums: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let base = wid.x * em.len;
    let local = lid.x;
    var partial: f32 = 0.0;
    var k: u32 = local;
    loop {
        if (k >= em.len) {
            break;
        }
        let v = x[base + k];
        partial = partial + v * v;
        k = k + 64u;
    }
    partial_sums[local] = partial;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (local < stride) {
            partial_sums[local] = partial_sums[local] + partial_sums[local + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    let mean_sq = partial_sums[0] / f32(em.len);
    let scale = 1.0 / sqrt(mean_sq + em.extra);
    k = local;
    loop {
        if (k >= em.len) {
            break;
        }
        y[base + k] = x[base + k] * scale * weight[k] + residual[base + k];
        k = k + 64u;
    }
}
"#;

pub fn shader_source_rmsnorm_rows() -> String {
    format!("{ELEM_META}\n{RMSNORM_ROWS_SHADER_BODY}")
}

pub fn shader_source_rmsnorm_add_rows() -> String {
    format!("{ELEM_META}\n{RMSNORM_ADD_ROWS_SHADER_BODY}")
}

/// `shader_source_rmsnorm_add_rows` with the trailing write multiplied by
/// `em.out_scale` — the row-strided form of `shader_source_rmsnorm_add_scale`,
/// for a prefill layer's per-layer output scale.
/// gemma4's per-layer-embedding inputs for a prefill, in one dispatch per
/// (token, layer) row: the projection's row scaled by `proj_scale`,
/// RMS-normed against the shared `proj_norm` weight, added to the token's
/// gathered per-layer embedding row, scaled by `in_scale` — and written
/// **layer-major** (`[n_layer][row_cap][per_layer]`, `row_cap` being
/// `n_tokens` rounded up to whole stripes), so each layer's stage reads its
/// `[n_tokens, per_layer]` operand as one contiguous slice of the device
/// buffer where the host path gathered it per layer, padded tail included.
/// The inputs are token-major, the layout the projection GEMM writes.
///
/// One workgroup of 64 per row over `per_layer` (256) elements. Bindings:
/// the projection (read), the norm weight (read), the gathered rows (read),
/// the output (write), the meta.
const PLE_INPUTS_SHADER: &str = r#"
struct PleInMeta {
    n_tokens: u32,
    n_layer: u32,
    per_layer: u32,
    eps: f32,
    proj_scale: f32,
    in_scale: f32,
    // Rows per layer in the output — `n_tokens` rounded up to whole
    // stripes, so a striped stage's padded tail stays inside its layer.
    row_cap: u32,
    _p1: u32,
}

@group(0) @binding(0) var<storage, read> proj: array<f32>;
@group(0) @binding(1) var<storage, read> nw: array<f32>;
@group(0) @binding(2) var<storage, read> gathered: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> pm: PleInMeta;

var<workgroup> pi_partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wid.x;
    let t = row / pm.n_layer;
    let il = row % pm.n_layer;
    let local = lid.x;
    let base_in = (t * pm.n_layer + il) * pm.per_layer;
    let base_out = (il * pm.row_cap + t) * pm.per_layer;

    var partial: f32 = 0.0;
    var k: u32 = local;
    loop {
        if (k >= pm.per_layer) {
            break;
        }
        let v = proj[base_in + k] * pm.proj_scale;
        partial = partial + v * v;
        k = k + 64u;
    }
    pi_partial[local] = partial;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (local < stride) {
            pi_partial[local] = pi_partial[local] + pi_partial[local + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let mean_sq = pi_partial[0] / f32(pm.per_layer);
    let scale = 1.0 / sqrt(mean_sq + pm.eps);
    k = local;
    loop {
        if (k >= pm.per_layer) {
            break;
        }
        let v = proj[base_in + k] * pm.proj_scale;
        out[base_out + k] = (v * scale * nw[k] + gathered[base_in + k]) * pm.in_scale;
        k = k + 64u;
    }
}
"#;

pub fn shader_source_ple_inputs() -> String {
    PLE_INPUTS_SHADER.to_string()
}

pub fn shader_source_rmsnorm_add_scale_rows() -> String {
    shader_source_rmsnorm_add_rows().replace(
        "y[base + k] = x[base + k] * scale * weight[k] + residual[base + k];",
        "y[base + k] = (x[base + k] * scale * weight[k] + residual[base + k]) * em.out_scale;",
    )
}

pub fn shader_source_rmsnorm_add(subgroup: bool, wg: usize) -> String {
    if subgroup {
        // As in `shader_source_rmsnorm`, the subgroup variant stays at its
        // fixed 64-thread workgroup; `wg` tunes the default tree path only.
        format!("{ELEM_META}\n{RMSNORM_ADD_SHADER_BODY_SUBGROUP}")
    } else {
        format!(
            "{ELEM_META}\n{}",
            norm_body_for_wg(RMSNORM_ADD_SHADER_BODY_TEMPLATE, wg)
        )
    }
}

/// `shader_source_rmsnorm_add` with the trailing write multiplied by
/// `em.out_scale` (gemma4's per-layer `layer_output_scale`) — folds the
/// separate post-`rmsnorm_add` `scale` dispatch into the norm+add's own final
/// loop. The `(norm*w + residual) * out_scale` intermediate is computed in the
/// same f32 as the split `add`-then-`scale` path (the `add`'s f32 result then
/// `* out_scale`), so the fusion is byte-exact. `em.out_scale` reuses
/// `ElemMeta`'s former `_pad1` slot; every other elementwise shader leaves it
/// `0.0` and never reads it, so no other shader's output changes.
/// Threads in a wide whole-row norm workgroup (`shader_source_rmsnorm_wide`).
pub const NORM_WIDE_WG: usize = 256;

/// `vec4` slots each thread of a wide norm loads straight-line before the
/// tail loop: `NORM_WIDE_WG * NORM_WIDE_SLOTS * 4` elements, which covers
/// every model width this backend has met without the loop running once.
pub const NORM_WIDE_SLOTS: usize = 8;

/// The whole-row RMSNorm family (`norm`, `norm + residual`, `(norm +
/// residual) * out_scale`) rewritten for the way one workgroup actually
/// spends its time on a decode step, where every one of these dispatches
/// is a single workgroup over the model width and the step runs several
/// per layer.
///
/// The grid-stride templates above walk the row in a rolled loop whose
/// every iteration consumes its own load before the next is issued, so a
/// thread's `n_embd / wg` elements cost `n_embd / wg` memory round trips in
/// series — on a rolled WGSL loop the compiler gives each load one register
/// and a full wait, never two in flight. That serial chain, not the
/// arithmetic or the reduction, is the dispatch: measured back to back at a
/// model width, the same kernel is twice as slow at half the workgroup
/// width, and the whole dispatch is several times the cost of a trivial
/// one. This version:
///
/// - reads the row as `vec4<f32>` (a quarter of the loads),
/// - issues each thread's `NORM_WIDE_SLOTS` loads as *named* straight-line
///   values, guarded but independent, so they all leave together and are
///   waited for once,
/// - keeps them in registers for the rescale, so `x` is read once, not twice,
/// - reduces in two rounds of straight-line shared-memory sums (two
///   barriers) instead of a `log2(wg)`-round tree.
///
/// A row wider than the straight-line slots cover falls into a rolled tail
/// loop and is still correct. The row width must be a multiple of four —
/// the caller checks and takes the scalar kernel otherwise.
fn shader_source_rmsnorm_wide_kind(kind: NormKind) -> String {
    shader_source_rmsnorm_wide_shaped(kind, false)
}

/// [`shader_source_rmsnorm_wide_kind`], and its row-strided form: with
/// `rows`, workgroup `x` norms row `x` of `[rows, em.len]` (the prefill
/// chains' shape — one workgroup per token, as the rolled `_rows` kernels
/// dispatch), reading and writing at `x * em.len` and the weight from its
/// start. Same loads-in-flight body either way; the rolled rows kernel spent
/// 3 µs a row on one-at-a-time loads.
fn shader_source_rmsnorm_wide_shaped(kind: NormKind, rows: bool) -> String {
    let wg = NORM_WIDE_WG;
    let slots = NORM_WIDE_SLOTS;
    let mut src = String::new();
    src.push_str(ELEM_META);
    src.push_str("\n@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;\n");
    src.push_str("@group(0) @binding(1) var<storage, read> weight: array<vec4<f32>>;\n");
    match kind {
        NormKind::Plain => {
            src.push_str("@group(0) @binding(2) var<storage, read_write> y: array<vec4<f32>>;\n");
            src.push_str("@group(0) @binding(3) var<uniform> em: ElemMeta;\n");
        }
        NormKind::Add | NormKind::AddScale => {
            src.push_str("@group(0) @binding(2) var<storage, read> residual: array<vec4<f32>>;\n");
            src.push_str("@group(0) @binding(3) var<storage, read_write> y: array<vec4<f32>>;\n");
            src.push_str("@group(0) @binding(4) var<uniform> em: ElemMeta;\n");
        }
        NormKind::AddThenNorm => {
            src.push_str("@group(0) @binding(2) var<storage, read> residual: array<vec4<f32>>;\n");
            src.push_str(
                "@group(0) @binding(3) var<storage, read_write> x_out: array<vec4<f32>>;\n",
            );
            src.push_str("@group(0) @binding(4) var<storage, read_write> y: array<vec4<f32>>;\n");
            src.push_str("@group(0) @binding(5) var<uniform> em: ElemMeta;\n");
        }
    }
    // Round one: every thread's partial. Round two: sixteen threads each
    // sum a sixteen-slot stripe, then every thread sums those sixteen.
    let stripes = 16;
    let stripe = wg / stripes;
    src.push_str(&format!(
        "\nvar<workgroup> partial_sums: array<f32, {wg}>;\nvar<workgroup> stripe_sums: array<f32, {stripes}>;\n\n\
         @compute @workgroup_size({wg})\nfn main(\n\
         \x20   @builtin(workgroup_id) wid: vec3<u32>,\n\
         \x20   @builtin(local_invocation_id) lid: vec3<u32>,\n) {{\n\
         \x20   let local = lid.x;\n\
         \x20   let n4 = em.len / 4u;\n\
         \x20   let base = {};\n",
        if rows { "wid.x * n4" } else { "0u" }
    ));
    // The pre-norm form sums the stream and the sub-layer's output first,
    // keeps the sum in the slot and writes it out as the new stream.
    let load = |idx: &str| -> String {
        match kind {
            NormKind::AddThenNorm => format!(
                "x[base + {idx}] + residual[base + {idx}]; x_out[base + {idx}] = v{}",
                idx.strip_prefix('k')
                    .filter(|r| r.chars().all(|c| c.is_ascii_digit()))
                    .unwrap_or("")
            ),
            _ => format!("x[base + {idx}]"),
        }
    };
    for i in 0..slots {
        src.push_str(&format!(
            "    let k{i} = local + {}u;\n    var v{i} = vec4<f32>(0.0);\n    if (k{i} < n4) {{ v{i} = {}; }}\n",
            i * wg,
            load(&format!("k{i}"))
        ));
    }
    src.push_str("    var partial: f32 = 0.0;\n");
    for i in 0..slots {
        src.push_str(&format!("    partial = partial + dot(v{i}, v{i});\n"));
    }
    // The tail for a row wider than the slots cover.
    let tail_load = match kind {
        NormKind::AddThenNorm => "x[base + k] + residual[base + k]; x_out[base + k] = v",
        _ => "x[base + k]",
    };
    src.push_str(&format!(
        "    var k: u32 = local + {}u;\n    loop {{\n        if (k >= n4) {{ break; }}\n        let v = {tail_load};\n        partial = partial + dot(v, v);\n        k = k + {wg}u;\n    }}\n",
        slots * wg
    ));
    src.push_str(&format!(
        "    partial_sums[local] = partial;\n    workgroupBarrier();\n    if (local < {stripes}u) {{\n        let sb = local * {stripe}u;\n        var s: f32 = 0.0;\n"
    ));
    for j in 0..stripe {
        src.push_str(&format!("        s = s + partial_sums[sb + {j}u];\n"));
    }
    src.push_str("        stripe_sums[local] = s;\n    }\n    workgroupBarrier();\n    var total: f32 = 0.0;\n");
    for j in 0..stripes {
        src.push_str(&format!("    total = total + stripe_sums[{j}u];\n"));
    }
    src.push_str(
        "    let mean_sq = total / f32(em.len);\n    let scale = 1.0 / sqrt(mean_sq + em.extra);\n",
    );
    let store = |idx: &str, val: &str| -> String {
        match kind {
            NormKind::Plain | NormKind::AddThenNorm => {
                format!("y[base + {idx}] = {val} * scale * weight[{idx}];")
            }
            NormKind::Add => {
                format!("y[base + {idx}] = {val} * scale * weight[{idx}] + residual[base + {idx}];")
            }
            NormKind::AddScale => {
                format!(
                    "y[base + {idx}] = ({val} * scale * weight[{idx}] + residual[base + {idx}]) * em.out_scale;"
                )
            }
        }
    };
    for i in 0..slots {
        src.push_str(&format!(
            "    if (k{i} < n4) {{ {} }}\n",
            store(&format!("k{i}"), &format!("v{i}"))
        ));
    }
    src.push_str(&format!(
        "    k = local + {}u;\n    loop {{\n        if (k >= n4) {{ break; }}\n        {}\n        k = k + {wg}u;\n    }}\n}}\n",
        slots * wg,
        store(
            "k",
            match kind {
                NormKind::AddThenNorm => "x_out[base + k]",
                _ => "x[base + k]",
            }
        )
    ));
    src
}

/// The pre-norm pair as one dispatch: `x_out = x + residual` (the stream
/// after a sub-layer's residual add) and `y = rmsnorm(x_out) · weight` (the
/// next sub-layer's input), six bindings (`elem6_bind_group_layout`). The
/// Qwen hybrid trunk's decode chain ran these as two dependent dispatches
/// per sub-layer, 128 a token; the second booked the first's drain.
pub fn shader_source_add_rmsnorm_wide() -> String {
    shader_source_rmsnorm_wide_kind(NormKind::AddThenNorm)
}

/// The wide norms, one workgroup per row — see
/// `shader_source_rmsnorm_wide_shaped`.
pub fn shader_source_rmsnorm_rows_wide() -> String {
    shader_source_rmsnorm_wide_shaped(NormKind::Plain, true)
}

pub fn shader_source_rmsnorm_add_rows_wide() -> String {
    shader_source_rmsnorm_wide_shaped(NormKind::Add, true)
}

pub fn shader_source_rmsnorm_add_scale_rows_wide() -> String {
    shader_source_rmsnorm_wide_shaped(NormKind::AddScale, true)
}

/// Which member of the whole-row norm family a wide kernel is.
#[derive(Clone, Copy)]
enum NormKind {
    Plain,
    Add,
    AddScale,
    /// The add first, then the norm of the sum — see
    /// `shader_source_add_rmsnorm_wide`.
    AddThenNorm,
}

/// The wide whole-row RMSNorm — see `shader_source_rmsnorm_wide_kind`.
pub fn shader_source_rmsnorm_wide() -> String {
    shader_source_rmsnorm_wide_kind(NormKind::Plain)
}

/// The wide RMSNorm + residual add.
pub fn shader_source_rmsnorm_add_wide() -> String {
    shader_source_rmsnorm_wide_kind(NormKind::Add)
}

/// The wide `(RMSNorm + residual) * out_scale`.
pub fn shader_source_rmsnorm_add_scale_wide() -> String {
    shader_source_rmsnorm_wide_kind(NormKind::AddScale)
}

/// Two whole-row norms over one vector in one dispatch: `y1 = rmsnorm(x) *
/// w1 + residual`, then `y2 = rmsnorm(y1) * w2` — the attention post-norm
/// with its residual add, and the FFN norm that reads its result. Both were
/// single-workgroup dispatches over the same row; the second used to reload
/// from memory what the first had just stored, behind a dispatch boundary.
/// Here `y1` stays in the registers it was computed in for the second
/// reduce, and is stored once because the FFN's own residual reads it.
///
/// The same straight-line loads and two-round reduce as
/// `shader_source_rmsnorm_wide_kind`; a row wider than the slots cover
/// falls into the rolled tail, which re-reads its own `y1` stores for the
/// second stage. Bindings: `x`, `w1`, `residual`, `w2` read-only, `y1`,
/// `y2` read-write, `em` (`extra` is `eps`, the same for both norms).
pub fn shader_source_rmsnorm_add_norm_wide() -> String {
    shader_source_add_norm_wide(true)
}

/// [`shader_source_rmsnorm_add_norm_wide`] with or without the post-norm:
/// without it, stage one is the residual add alone (`y1 = x + residual`,
/// the llama family's shape) and `w1` is bound but never read, so the
/// same bind group layout serves both. What this buys the family that
/// has no post-norm is the pair's second half — the FFN norm and its
/// 8-bit epilogue in the same dispatch as the add, one dispatch a layer
/// fewer where the layer is sixteen.
pub fn shader_source_add_norm_wide(post_norm: bool) -> String {
    let wg = NORM_WIDE_WG;
    let slots = NORM_WIDE_SLOTS;
    let stripes = 16;
    let stripe = wg / stripes;
    let mut src = String::new();
    src.push_str(ELEM_META);
    src.push_str("\n@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;\n");
    src.push_str("@group(0) @binding(1) var<storage, read> w1: array<vec4<f32>>;\n");
    src.push_str("@group(0) @binding(2) var<storage, read> residual: array<vec4<f32>>;\n");
    src.push_str("@group(0) @binding(3) var<storage, read> w2: array<vec4<f32>>;\n");
    src.push_str("@group(0) @binding(4) var<storage, read_write> y1: array<vec4<f32>>;\n");
    src.push_str("@group(0) @binding(5) var<storage, read_write> y2: array<vec4<f32>>;\n");
    src.push_str("@group(0) @binding(6) var<uniform> em: ElemMeta;\n");
    // The 8-bit form of `y2` for the integer-dot matvecs that read it
    // (`shader_source_quantize_q8`'s layout), written when `em.aux == 1`:
    // eight adjacent lanes hold a 32-element block's eight `vec4`s, so its
    // scale and quant sum are two eight-lane reductions through workgroup
    // memory — the row never leaves the kernel that made it, and the
    // consumers need no quantize dispatch of their own.
    src.push_str("@group(0) @binding(7) var<storage, read_write> q8: array<u32>;\n");
    src.push_str(&format!(
        "\nvar<workgroup> partial_sums: array<f32, {wg}>;\nvar<workgroup> stripe_sums: array<f32, {stripes}>;\nvar<workgroup> q8_amax: array<f32, {wg}>;\nvar<workgroup> q8_sum: array<i32, {wg}>;\n\n\
         fn reduce_all(local: u32, partial: f32) -> f32 {{\n\
         \x20   partial_sums[local] = partial;\n    workgroupBarrier();\n\
         \x20   if (local < {stripes}u) {{\n        let base = local * {stripe}u;\n        var s: f32 = 0.0;\n"
    ));
    for j in 0..stripe {
        src.push_str(&format!("        s = s + partial_sums[base + {j}u];\n"));
    }
    src.push_str("        stripe_sums[local] = s;\n    }\n    workgroupBarrier();\n    var total: f32 = 0.0;\n");
    for j in 0..stripes {
        src.push_str(&format!("    total = total + stripe_sums[{j}u];\n"));
    }
    // The second call's writes to `partial_sums` must not overtake a slow
    // lane's reads of the stripe sums from the first.
    src.push_str("    workgroupBarrier();\n    return total;\n}\n\n");
    src.push_str(&format!(
        "@compute @workgroup_size({wg})\nfn main(@builtin(local_invocation_id) lid: vec3<u32>) {{\n\
         \x20   let local = lid.x;\n\
         \x20   let n4 = em.len / 4u;\n"
    ));
    for i in 0..slots {
        src.push_str(&format!(
            "    let k{i} = local + {}u;\n    var v{i} = vec4<f32>(0.0);\n    if (k{i} < n4) {{ v{i} = x[k{i}]; }}\n",
            i * wg
        ));
    }
    src.push_str("    var partial: f32 = 0.0;\n");
    for i in 0..slots {
        src.push_str(&format!("    partial = partial + dot(v{i}, v{i});\n"));
    }
    src.push_str(&format!(
        "    var k: u32 = local + {}u;\n    loop {{\n        if (k >= n4) {{ break; }}\n        let v = x[k];\n        partial = partial + dot(v, v);\n        k = k + {wg}u;\n    }}\n",
        slots * wg
    ));
    // Without the post-norm the first reduce is skipped and the branch
    // is added as it is: `scale1 * w1` becomes 1.
    let (scale_w1_slot, scale_w1_loop): (Box<dyn Fn(usize) -> String>, &str) = if post_norm {
        src.push_str(
            "    let scale1 = 1.0 / sqrt(reduce_all(local, partial) / f32(em.len) + em.extra);\n",
        );
        (
            Box::new(|i| format!("v{i} * scale1 * w1[k{i}]")),
            "x[k] * scale1 * w1[k]",
        )
    } else {
        (Box::new(|i| format!("v{i}")), "x[k]")
    };
    // Stage two: `y1` in the same registers, stored, and squared for the
    // second reduce.
    src.push_str("    partial = 0.0;\n");
    for i in 0..slots {
        src.push_str(&format!(
            "    if (k{i} < n4) {{ v{i} = {} + residual[k{i}]; y1[k{i}] = v{i}; partial = partial + dot(v{i}, v{i}); }}\n",
            scale_w1_slot(i)
        ));
    }
    src.push_str(&format!(
        "    k = local + {}u;\n    loop {{\n        if (k >= n4) {{ break; }}\n        let v = {scale_w1_loop} + residual[k];\n        y1[k] = v;\n        partial = partial + dot(v, v);\n        k = k + {wg}u;\n    }}\n",
        slots * wg
    ));
    src.push_str(
        "    let scale2 = 1.0 / sqrt(reduce_all(local, partial) / f32(em.len) + em.extra);\n",
    );
    for i in 0..slots {
        src.push_str(&format!(
            "    if (k{i} < n4) {{ v{i} = v{i} * scale2 * w2[k{i}]; y2[k{i}] = v{i}; }}\n"
        ));
    }
    src.push_str(&format!(
        "    k = local + {}u;\n    loop {{\n        if (k >= n4) {{ break; }}\n        y2[k] = y1[k] * scale2 * w2[k];\n        k = k + {wg}u;\n    }}\n",
        slots * wg
    ));
    src.push_str(&q8_epilogue(slots, wg));
    src.push_str("}\n");
    src
}

/// The q8 epilogue the norm kernels share: with `em.aux == 1`, the row the
/// slots hold as `v0..vN` (`k0..kN` their `vec4` indices, `n4` the row's
/// `vec4` count) is written in `shader_source_quantize_q8`'s layout to
/// `q8`. Eight adjacent lanes hold a 32-element block's eight `vec4`s, so
/// a block's scale and quant sum are two eight-lane reductions through
/// `q8_amax`/`q8_sum`. Only rows the slots cover whole (`n4 <= slots *
/// wg`) are quantized here; the caller binds the q8 output only for those.
fn q8_epilogue(slots: usize, wg: usize) -> String {
    let mut src = String::from("    if (em.aux == 1u) {\n");
    for i in 0..slots {
        // A slot no lane of the row reaches is skipped whole — uniformly, so
        // its barriers go with it: a 1536-wide row uses two of the eight.
        src.push_str(&format!(
            "        if ({}u < n4) {{\n        {{\n            let a = select(vec4<f32>(0.0), abs(v{i}), k{i} < n4);\n            q8_amax[local] = max(max(a.x, a.y), max(a.z, a.w));\n        }}\n        workgroupBarrier();\n        {{\n            let c = (local / 8u) * 8u;\n            var amax: f32 = q8_amax[c];\n",
            i * wg
        ));
        for j in 1..8 {
            src.push_str(&format!(
                "            amax = max(amax, q8_amax[c + {j}u]);\n"
            ));
        }
        src.push_str(&format!(
            "            let d = amax / 127.0;\n            let id = select(0.0, 1.0 / d, d > 0.0);\n            let q = clamp(vec4<i32>(round(v{i} * id)), vec4<i32>(-127), vec4<i32>(127));\n            q8_sum[local] = q.x + q.y + q.z + q.w;\n            let word = (u32(q.x) & 0xFFu) | ((u32(q.y) & 0xFFu) << 8u) | ((u32(q.z) & 0xFFu) << 16u) | ((u32(q.w) & 0xFFu) << 24u);\n            let blk = k{i} / 8u;\n            if (k{i} < n4) {{ q8[blk * 10u + 2u + (k{i} % 8u)] = word; }}\n            workgroupBarrier();\n            if (k{i} < n4 && (local % 8u) == 0u) {{\n                var sum: i32 = 0;\n"
        ));
        for j in 0..8 {
            src.push_str(&format!("                sum = sum + q8_sum[c + {j}u];\n"));
        }
        src.push_str(
            "                q8[blk * 10u] = bitcast<u32>(d);\n                q8[blk * 10u + 1u] = bitcast<u32>(sum);\n            }\n            workgroupBarrier();\n        }\n        }\n",
        );
    }
    src.push_str("    }\n");
    src
}

/// The plain wide RMSNorm with the q8 epilogue, for the attention norm
/// whose projections run on the integer dot: on the norm pair's layout
/// (`x` at 0, `weight` at 1, `y` at 4, `q8` at 5, the meta at 6;
/// placeholders at 2, 3 and 7). Same arithmetic as
/// `shader_source_rmsnorm_wide`.
pub fn shader_source_rmsnorm_wide_q8() -> String {
    let wg = NORM_WIDE_WG;
    let slots = NORM_WIDE_SLOTS;
    let stripes = 16;
    let stripe = wg / stripes;
    let mut src = String::new();
    src.push_str(ELEM_META);
    src.push_str("\n@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;\n");
    src.push_str("@group(0) @binding(1) var<storage, read> weight: array<vec4<f32>>;\n");
    src.push_str("@group(0) @binding(4) var<storage, read_write> y: array<vec4<f32>>;\n");
    src.push_str("@group(0) @binding(5) var<storage, read_write> q8: array<u32>;\n");
    src.push_str("@group(0) @binding(6) var<uniform> em: ElemMeta;\n");
    src.push_str(&format!(
        "\nvar<workgroup> partial_sums: array<f32, {wg}>;\nvar<workgroup> stripe_sums: array<f32, {stripes}>;\nvar<workgroup> q8_amax: array<f32, {wg}>;\nvar<workgroup> q8_sum: array<i32, {wg}>;\n\n\
         @compute @workgroup_size({wg})\nfn main(@builtin(local_invocation_id) lid: vec3<u32>) {{\n\
         \x20   let local = lid.x;\n\
         \x20   let n4 = em.len / 4u;\n"
    ));
    for i in 0..slots {
        src.push_str(&format!(
            "    let k{i} = local + {}u;\n    var v{i} = vec4<f32>(0.0);\n    if (k{i} < n4) {{ v{i} = x[k{i}]; }}\n",
            i * wg
        ));
    }
    src.push_str("    var partial: f32 = 0.0;\n");
    for i in 0..slots {
        src.push_str(&format!("    partial = partial + dot(v{i}, v{i});\n"));
    }
    src.push_str(&format!(
        "    var k: u32 = local + {}u;\n    loop {{\n        if (k >= n4) {{ break; }}\n        let v = x[k];\n        partial = partial + dot(v, v);\n        k = k + {wg}u;\n    }}\n",
        slots * wg
    ));
    src.push_str(&format!(
        "    partial_sums[local] = partial;\n    workgroupBarrier();\n    if (local < {stripes}u) {{\n        let sb = local * {stripe}u;\n        var s: f32 = 0.0;\n"
    ));
    for j in 0..stripe {
        src.push_str(&format!("        s = s + partial_sums[sb + {j}u];\n"));
    }
    src.push_str("        stripe_sums[local] = s;\n    }\n    workgroupBarrier();\n    var total: f32 = 0.0;\n");
    for j in 0..stripes {
        src.push_str(&format!("    total = total + stripe_sums[{j}u];\n"));
    }
    src.push_str("    let scale = 1.0 / sqrt(total / f32(em.len) + em.extra);\n");
    for i in 0..slots {
        src.push_str(&format!(
            "    if (k{i} < n4) {{ v{i} = v{i} * scale * weight[k{i}]; y[k{i}] = v{i}; }}\n"
        ));
    }
    src.push_str(&format!(
        "    k = local + {}u;\n    loop {{\n        if (k >= n4) {{ break; }}\n        y[k] = x[k] * scale * weight[k];\n        k = k + {wg}u;\n    }}\n",
        slots * wg
    ));
    src.push_str(&q8_epilogue(slots, wg));
    src.push_str("}\n");
    src
}

pub fn shader_source_rmsnorm_add_scale(subgroup: bool, wg: usize) -> String {
    shader_source_rmsnorm_add(subgroup, wg).replace(
        "y[k] = x[k] * scale * weight[k] + residual[k];",
        "y[k] = (x[k] * scale * weight[k] + residual[k]) * em.out_scale;",
    )
}

/// GPU-resident causal attention for a *single* query token (decode,
/// `n_tokens == 1`) against a GPU-resident KV cache — one workgroup per
/// query head, 64 threads. Online-softmax, **tiled** over the KV sequence in chunks of 64
/// positions (`TILE`, matching the workgroup width) rather than the old
/// design's two full passes over every candidate position (a max pass,
/// then a normalize-and-store pass, each independently recomputing every
/// position's `q·k`).
///
/// Per tile: each of the 64 threads computes **one** tile position's
/// score (`score_at`, unchanged — a single thread's sequential dot
/// product over `head_dim`, same as before; this is *never* recomputed
/// for a position once its tile has been processed), a workgroup tree
/// reduction finds the tile's max and (after subtracting it) sum, and the
/// running online-softmax state `(m, l)` — plain per-thread scalars, not
/// `var<workgroup>`, since every thread computes the identical update
/// from the same shared reduction results — absorbs the tile via the
/// standard rescale-and-merge rule. The running weighted-output
/// accumulator (`acc`, `head_dim`-long) lives in `var<workgroup>` shared
/// memory, split across head_dim the same way the old design's final pass
/// was (each thread owns `head_dim / 64` slots, a plain scalar loop, no
/// per-thread array), and only ever needs `MAX_HEAD_DIM` worth of shared
/// memory — bounded and small (a few KB) regardless of context length —
/// unlike a per-thread accumulator sized `head_dim` per *thread* would be
/// (64 of those, register-spill-prone for `E2B`'s real `head_dim = 512`).
/// `tile_probs` (also `var<workgroup>`, tile-sized — 64 entries, not
/// `n_pos`) holds this tile's normalized-to-`tile_max` weights just long
/// enough for the accumulator-update step to read them back.
///
/// Net effect vs. the two-pass design: every candidate position's score
/// is computed exactly **once** (not twice), and the working set is
/// bounded by `head_dim`/the tile size rather than by context length — no
/// `probs_scratch`-sized (`[n_head, capacity]`) buffer read or written at
/// all (that buffer is still allocated and bound at binding 3 for now,
/// simply unused by this shader — removing it is a separate, smaller
/// follow-up). Barrier count is
/// `O(n_pos / 64)` (a handful of barriers per tile), not the old design's
/// fixed `O(log 64)` — more barriers for a very long context, but each one
/// now amortizes 64 positions' worth of work instead of the whole
/// context's, which is the standard flash-attention trade-off. GQA is
/// resolved once per workgroup (`kv_head = h / (n_head / n_head_kv)`);
/// sliding-window attention is still just a nonzero `window_start`.
const ATTENTION_SHADER_TEMPLATE: &str = r#"
%KV_ENABLE%
struct AttnMeta {
    n_head: u32,
    n_head_kv: u32,
    head_dim: u32,
    window_start: u32,
    n_pos: u32,
    capacity: u32,
    scale: f32,
    // The four below are read only by the multi-query (prefill) variant,
    // which derives each query's own window from them; the single-query
    // variant takes `window_start`/`n_pos` above as given and ignores these.
    start_pos: u32,
    n_query: u32,
    n_swa: u32,
    causal: u32,
    // Read only by the paged form of this kernel; zero otherwise.
    kv_page_base: u32,
    kv_page_tokens: u32,
    kv_ring_rows: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> aq: array<f32>;
%KV_BINDINGS%
@group(0) @binding(3) var<storage, read_write> probs_scratch: array<f32>;
@group(0) @binding(4) var<storage, read_write> aout: array<f32>;
@group(0) @binding(5) var<uniform> am: AttnMeta;
%KV_PAGE_BINDING%

%KV_READ_FNS%
%KV_SLOT_FN%

// Size of the workgroup-shared `acc` accumulator, `MAX_HEAD_DIM * 4` bytes.
// The `2048u` here is a placeholder the shader-source builders substitute
// with the model's actual head_dim (`shader_source_attention`/`_split`'s
// `max_head_dim` argument) so `acc` isn't oversized — an oversized `acc`
// costs LDS and caps occupancy. The literal default only applies if a
// builder passes `2048` (the un-split test kernel does).
const MAX_HEAD_DIM: u32 = 2048u;

var<workgroup> shared_reduce: array<f32, 64>;
var<workgroup> tile_probs: array<f32, 64>;
var<workgroup> acc: array<f32, MAX_HEAD_DIM>;

fn score_at(q_off: u32, h: u32, kv_head: u32, p: u32) -> f32 {
    let head_dim = am.head_dim;
    let q_base = q_off + h * head_dim;
    let k_base = (kv_slot(p) * am.n_head_kv + kv_head) * head_dim;
    var s: f32 = 0.0;
    var d: u32 = 0u;
    loop {
        if (d >= head_dim) {
            break;
        }
        s = s + aq[q_base + d] * kv_read_k(k_base + d);
        d = d + 1u;
    }
    return s * am.scale;
}

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    %SUBGROUP_PARAMS%
) {
    let h = wid.x;
    let local = lid.x;
    let group_size = am.n_head / am.n_head_kv;
    let kv_head = h / group_size;
    let head_dim = am.head_dim;
%QUERY_SETUP%

    var zd: u32 = local;
    loop {
        if (zd >= head_dim) {
            break;
        }
        acc[zd] = 0.0;
        zd = zd + 64u;
    }

    var m: f32 = -1e30;
    var l: f32 = 0.0;

    var tile_start: u32 = 0u;
    loop {
        if (tile_start >= n_pos) {
            break;
        }
        let tile_len = min(64u, n_pos - tile_start);
        let has_pos = local < tile_len;
        let p = window_start + tile_start + local;

        var my_score: f32 = -1e30;
        if (has_pos) {
            my_score = score_at(q_off, h, kv_head, p);
        }
        %MAX_REDUCE_BLOCK%

        var my_prob: f32 = 0.0;
        if (has_pos) {
            my_prob = exp(my_score - tile_max);
        }
        tile_probs[local] = my_prob;
        %SUM_REDUCE_BLOCK%

        let new_m = max(m, tile_max);
        let alpha_old = exp(m - new_m);
        let alpha_tile = exp(tile_max - new_m);
        l = l * alpha_old + tile_sum * alpha_tile;

        var d2: u32 = local;
        loop {
            if (d2 >= head_dim) {
                break;
            }
            var tile_contribution: f32 = 0.0;
            var j: u32 = 0u;
            loop {
                if (j >= tile_len) {
                    break;
                }
                let vp = window_start + tile_start + j;
                let v_base = (kv_slot(vp) * am.n_head_kv + kv_head) * head_dim;
                tile_contribution = tile_contribution + tile_probs[j] * kv_read_v(v_base + d2);
                j = j + 1u;
            }
            acc[d2] = acc[d2] * alpha_old + alpha_tile * tile_contribution;
            d2 = d2 + 64u;
        }

        m = new_m;
        workgroupBarrier();
        tile_start = tile_start + 64u;
    }

    var d3: u32 = local;
    loop {
        if (d3 >= head_dim) {
            break;
        }
        aout[out_off + h * head_dim + d3] = acc[d3] / l;
        d3 = d3 + 64u;
    }
}
"#;

/// `kv_f16` selects whether `k_cache`/`v_cache` are bound as `array<f16>`
/// (the KV mirror's storage type when the adapter supports native WGSL
/// `f16`) or `array<f32>` (the
/// original, always-available path). Every read of either array already
/// goes through an `f32(...)` widening cast (a no-op when the array is
/// already `f32`), so the score/softmax/weighted-sum math itself is
/// identical either way — only the storage type, and hence the KV
/// mirror's memory traffic, changes.
/// The subgroup reduction for the attention softmax's per-tile max
/// and sum, substituted into `ATTENTION_SHADER_TEMPLATE`'s `%MAX_REDUCE_
/// BLOCK%`/`%SUM_REDUCE_BLOCK%` placeholders when `subgroup` is set — see
/// `reduce_combine_block`'s doc comment for the general-subgroup-
/// size rationale applied here too. Unlike the dot-product reduce kernels
/// (only lane 0 needs the total) or RMSNorm (every lane redundantly
/// recomputes the tiny combine, no second barrier), `shared_reduce` here is
/// reused twice more per tile iteration (the sum-phase, then next tile's
/// max-phase), so each phase keeps the classic design's two-barrier
/// discipline: one barrier after the subgroup partials are written (makes
/// them visible workgroup-wide), a second after every lane's own redundant
/// combine loop (a hazard barrier — protects against a fast lane starting
/// to overwrite `shared_reduce` for the *next* phase before a slow lane has
/// finished reading it for *this* one, exactly the reason the classic path
/// already had a barrier in the same spot). Four barriers per tile instead
/// of the classic path's sixteen, most of the win coming from the
/// eliminated pairwise-tree rounds themselves, not just their barriers.
fn attention_subgroup_blocks() -> (&'static str, &'static str) {
    let max_block = r#"
        let sg_max = subgroupMax(my_score);
        if (subgroup_invocation_id == 0u) {
            shared_reduce[subgroup_id] = sg_max;
        }
        workgroupBarrier();
        var tile_max: f32 = shared_reduce[0];
        var mi: u32 = 1u;
        loop {
            if (mi >= num_subgroups) {
                break;
            }
            tile_max = max(tile_max, shared_reduce[mi]);
            mi = mi + 1u;
        }
        workgroupBarrier();
"#;
    let sum_block = r#"
        let sg_sum = subgroupAdd(my_prob);
        if (subgroup_invocation_id == 0u) {
            shared_reduce[subgroup_id] = sg_sum;
        }
        workgroupBarrier();
        var tile_sum: f32 = shared_reduce[0];
        var si: u32 = 1u;
        loop {
            if (si >= num_subgroups) {
                break;
            }
            tile_sum = tile_sum + shared_reduce[si];
            si = si + 1u;
        }
        workgroupBarrier();
"#;
    (max_block, sum_block)
}

fn attention_classic_blocks() -> (&'static str, &'static str) {
    let max_block = r#"
        shared_reduce[local] = my_score;
        workgroupBarrier();
        var stride: u32 = 32u;
        loop {
            if (stride == 0u) {
                break;
            }
            if (local < stride) {
                shared_reduce[local] = max(shared_reduce[local], shared_reduce[local + stride]);
            }
            workgroupBarrier();
            stride = stride / 2u;
        }
        let tile_max = shared_reduce[0];
        workgroupBarrier();
"#;
    let sum_block = r#"
        shared_reduce[local] = my_prob;
        workgroupBarrier();
        var stride2: u32 = 32u;
        loop {
            if (stride2 == 0u) {
                break;
            }
            if (local < stride2) {
                shared_reduce[local] = shared_reduce[local] + shared_reduce[local + stride2];
            }
            workgroupBarrier();
            stride2 = stride2 / 2u;
        }
        let tile_sum = shared_reduce[0];
        workgroupBarrier();
"#;
    (max_block, sum_block)
}

/// Which of three ways `k_cache`/`v_cache` are stored — one fixed choice
/// per process (baked into the attention pipelines' WGSL text once at
/// `VulkanBackend::try_init`, the same way `kv_f16` alone used to be, not
/// a per-dispatch runtime branch). `attention_kv_bindings_and_reads`
/// substitutes both the bind-group's array element type (`%KV_BINDINGS%`)
/// and the `kv_read_k`/`kv_read_v` function bodies every score/weighted-
/// sum read in [`ATTENTION_SHADER_TEMPLATE`]/[`ATTENTION_SPLIT_SHADER_
/// TEMPLATE`] now goes through (`%KV_READ_FNS%`), so both templates share
/// one implementation of "how do I read one KV element" per storage kind
/// instead of duplicating it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KvStorage {
    F32,
    F16,
    /// A KV-cache-internal block quantization — **not** ggml's own
    /// `block_q8_0` byte layout (34 bytes: `f16` scale + 32 `i8`
    /// values), because nothing outside this process ever reads these
    /// bytes (no GGUF round-trip, no cross-process sharing), so there is
    /// no compatibility reason to match it. Instead: 36 bytes (9 `u32`
    /// words) per 32-element block — a plain `f32` scale (word 0, via
    /// `bitcast`, not a packed `f16`) followed by 32 `i8` values packed 4
    /// per word (words 1..9) — deliberately word-aligned throughout, so
    /// every read/write is a whole-`u32` load/store, never the
    /// byte-at-a-time read-modify-write ggml's own tighter 34-byte
    /// packing would force in WGSL (no byte-addressable storage writes).
    /// ~1.125 bytes/element — still a real ~44% reduction versus `f16`'s
    /// 2 bytes/element, just not quite `f16`'s exact halving-again ratio,
    /// the two extra scale bytes being the only difference from ggml's
    /// own ~1.0625 bytes/element. Requires `kv_dim % 32 == 0` (every
    /// GQA-shaped model this engine supports satisfies this in practice —
    /// `head_dim` is always a multiple of 32 — but `VulkanBackend::
    /// try_init` still checks rather than assuming it).
    Q8_0,
}

#[cfg(test)]
mod paging_tests {
    use super::*;

    /// **The contiguous kernel must be textually unchanged.** Paging is opt-in
    /// and its whole safety argument is that the un-paged shader is the one
    /// that has always run — so the identity form must emit no binding, and a
    /// `kv_slot` the compiler folds away.
    #[test]
    fn the_contiguous_form_adds_no_binding_and_no_lookup_at_all() {
        assert_eq!(KvPaging::Contiguous.binding(), "");
        assert_eq!(KvPaging::Contiguous.slot_fn(), "");
        let src = shader_source_attention_split(KvStorage::F16, false, 256, KvPaging::Contiguous);
        assert!(
            !src.contains("kv_page_table"),
            "the contiguous kernel declared a page-table binding it never reads"
        );
        // Not merely "no page table" — no `kv_slot` at anywhere either. An
        // identity function looked equivalent and cost 7.5% of decode, because
        // a per-position call that is not folded away is a per-position call.
        assert!(
            !src.contains("kv_slot("),
            "the contiguous kernel still routes its address through a lookup"
        );
        assert!(src.contains("(p * am.n_head_kv"), "the bare index is back");
    }

    /// And the paged form must declare the binding it reads, or the shader
    /// will not compile — which is the good failure, but only if the binding
    /// is emitted with the lookup rather than separately from it.
    #[test]
    fn the_paged_form_declares_the_table_it_reads() {
        let src = shader_source_attention_split(KvStorage::F16, false, 256, KvPaging::Paged);
        assert!(src.contains("var<storage, read> kv_page_table"));
        assert!(src.contains("kv_page_table[am.kv_page_base"));
        // Every position the kernel addresses must go through the lookup —
        // a site left as a bare multiply reads the wrong row, silently.
        assert!(
            !src.contains("(p * am.n_head_kv"),
            "a key index bypasses kv_slot"
        );
        assert!(
            !src.contains("(vp * am.n_head_kv"),
            "a value index bypasses kv_slot"
        );
    }

    /// The two forms must differ in exactly the address computation and
    /// nothing else — same reductions, same tiling, same storage handling.
    #[test]
    fn paging_changes_only_the_address() {
        let plain = shader_source_attention_split(KvStorage::F16, false, 256, KvPaging::Contiguous);
        let paged = shader_source_attention_split(KvStorage::F16, false, 256, KvPaging::Paged);
        // Rewrite the paged source into the contiguous one's spelling, by
        // substituting whole constructs rather than filtering lines. Line
        // filtering was tried first and was wrong in a way worth recording: it
        // removed `fn kv_slot`'s signature and body but left its closing brace,
        // so the test reported a difference that existed only in the test.
        let strip = |s: &str| {
            s.replace(KvPaging::Paged.slot_fn(), "")
                .replace(KvPaging::Paged.binding(), "")
                .replace("kv_slot(p)", "p")
                .replace("kv_slot(vp)", "vp")
                .lines()
                .filter(|l| !l.trim().is_empty())
                .collect::<Vec<_>>()
                .join("\n")
        };
        assert_eq!(
            strip(&plain),
            strip(&paged),
            "paging changed something other than how a position becomes a row"
        );
    }
}

/// Whether an attention kernel resolves a cached position through a page table.
///
/// A KV cache that shares pages between requests is not one contiguous run per
/// layer, so `position -> row` stops being a multiply. Every attention kernel
/// computes that address in the same shape —
/// `(p * n_head_kv + kv_head) * head_dim` — and this changes `p` and nothing
/// else: [`KvPaging::Contiguous`] emits an identity function that the compiler
/// folds away, so the un-paged shader is textually what it has always been and
/// cannot regress.
///
/// The lookup is per position rather than per tile. A tile is 64 positions and
/// a page is typically fewer, so a tile can straddle pages and the hoisted form
/// would be wrong; the per-position load is one `u32` from a small buffer that
/// every lane in the workgroup reads at nearly the same address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvPaging {
    /// One run per layer — position *is* row.
    Contiguous,
    /// Rows live in pool pages named by a per-sequence table.
    ///
    // Not selected by any dispatch yet: the device mirror is still one
    // contiguous run per request, so nothing has a page table to bind. This
    // variant is the kernel half of that change, landed and held to the
    // contiguous kernel's behaviour by `paging_tests` first, so that moving the
    // storage is a change to *where bytes live* and not simultaneously a change
    // to what the shader computes.
    #[allow(dead_code)]
    Paged,
    /// The mirror is a ring of `am.kv_ring_rows` rows, a power of two:
    /// position `p` is at row `p & (rows - 1)`.
    /// For a layer whose attention window is bounded
    /// (`LayerCache::set_mirror_ring`), so the mirror holds the window and
    /// the run being written rather than the whole context.
    Ring,
}

impl KvPaging {
    /// Applies whatever post-substitution rewriting this mode needs.
    ///
    /// For [`Self::Contiguous`] that is putting the address expressions back:
    /// the templates are written in terms of `kv_slot(..)` so the paged form
    /// has somewhere to hook, and the contiguous form must not pay for that
    /// spelling.
    fn finish(self, src: String) -> String {
        match self {
            KvPaging::Contiguous => src
                .replace("kv_slot(p)", "p")
                .replace("kv_slot(vp)", "vp")
                .replace("kv_slot(pp)", "pp"),
            KvPaging::Paged => src,
            // Inline, for the same reason the contiguous form is: an AND
            // against a uniform is one instruction, a call is not. A mask,
            // not a modulo — the ring is a power of two for it. The exact
            // size (window plus chunk) with a `%` per position was tried:
            // a third less memory on a 1024-token window, and the prefill
            // kernels, which take the address per position in their inner
            // loop, lost 12% on a 2238-token prompt.
            KvPaging::Ring => src
                .replace("kv_slot(p)", "(p & (am.kv_ring_rows - 1u))")
                .replace("kv_slot(vp)", "(vp & (am.kv_ring_rows - 1u))")
                .replace("kv_slot(pp)", "(pp & (am.kv_ring_rows - 1u))"),
        }
    }

    /// `%KV_PAGE_BINDING%` — the table, declared only when it is read.
    fn binding(self) -> &'static str {
        match self {
            KvPaging::Contiguous | KvPaging::Ring => "",
            KvPaging::Paged => {
                "@group(0) @binding(6) var<storage, read> kv_page_table: array<u32>;"
            }
        }
    }

    /// `%KV_SLOT_FN%` — logical position to physical row.
    ///
    /// `am.kv_page_base` is where this sequence's table starts, and
    /// `am.kv_page_tokens` how many positions one page covers.
    ///
    /// **Contiguous emits nothing at all**, and [`Self::finish`] rewrites the
    /// call sites back to the bare position instead. An identity function
    /// looked equivalent and was not: leaving `kv_slot(p)` in the contiguous
    /// kernel cost **7.5% of decode at depth 1024** against the same code
    /// without it, which is what a per-position call that does not get folded
    /// away costs. The contiguous shader is now textually what it was before
    /// paging existed, which is the only form that cannot regress.
    fn slot_fn(self) -> &'static str {
        match self {
            KvPaging::Contiguous | KvPaging::Ring => "",
            KvPaging::Paged => {
                "fn kv_slot(p: u32) -> u32 {\n    \
                 let page = kv_page_table[am.kv_page_base + p / am.kv_page_tokens];\n    \
                 return page * am.kv_page_tokens + p % am.kv_page_tokens;\n}"
            }
        }
    }
}

impl KvStorage {
    /// `%KV_ENABLE%` — `enable f16;` must lead the WGSL module when (and
    /// only when) an `f16`-typed binding is actually declared.
    fn enable_directive(self) -> &'static str {
        match self {
            KvStorage::F16 => "enable f16;",
            KvStorage::F32 | KvStorage::Q8_0 => "",
        }
    }

    /// `%KV_BINDINGS%` (bindings 1/2, `k_cache`/`v_cache`) and
    /// `%KV_READ_FNS%` (the `kv_read_k`/`kv_read_v` functions every score/
    /// weighted-sum read in both attention templates calls instead of
    /// indexing `k_cache`/`v_cache` directly) for this storage kind.
    fn bindings_and_read_fns(self) -> (String, String) {
        match self {
            KvStorage::F32 | KvStorage::F16 => {
                let ty = if self == KvStorage::F16 { "f16" } else { "f32" };
                let bindings = format!(
                    "@group(0) @binding(1) var<storage, read> k_cache: array<{ty}>;\n\
                     @group(0) @binding(2) var<storage, read> v_cache: array<{ty}>;"
                );
                let read_fns = "fn kv_read_k(idx: u32) -> f32 { return f32(k_cache[idx]); }\n\
                     fn kv_read_v(idx: u32) -> f32 { return f32(v_cache[idx]); }"
                    .to_string();
                (bindings, read_fns)
            }
            KvStorage::Q8_0 => {
                let bindings = "@group(0) @binding(1) var<storage, read> k_cache: array<u32>;\n\
                     @group(0) @binding(2) var<storage, read> v_cache: array<u32>;"
                    .to_string();
                // Mirrors `KV_QUANTIZE_Q8_0_SHADER`'s own write layout
                // exactly — see `KvStorage::Q8_0`'s own doc comment for
                // the block shape (9 words: 1 `f32` scale + 32 `i8`
                // values packed 4/word).
                let read_fns = r#"
fn kv_dequant_q8_0(word0: u32, word_rest: u32, in_block: u32) -> f32 {
    let d = bitcast<f32>(word0);
    let j = in_block % 4u;
    let byte = (word_rest >> (j * 8u)) & 0xFFu;
    var q: i32 = i32(byte);
    if (q >= 128) {
        q = q - 256;
    }
    return f32(q) * d;
}
fn kv_read_k(idx: u32) -> f32 {
    let block = idx / 32u;
    let in_block = idx % 32u;
    let word_base = block * 9u;
    return kv_dequant_q8_0(k_cache[word_base], k_cache[word_base + 1u + in_block / 4u], in_block);
}
fn kv_read_v(idx: u32) -> f32 {
    let block = idx / 32u;
    let in_block = idx % 32u;
    let word_base = block * 9u;
    return kv_dequant_q8_0(v_cache[word_base], v_cache[word_base + 1u + in_block / 4u], in_block);
}
"#
                .trim()
                .to_string();
                (bindings, read_fns)
            }
        }
    }
}

/// `kv_storage` selects whether `k_cache`/`v_cache` are bound as
/// `array<f16>` (the KV mirror's storage type when the adapter supports
/// native WGSL `f16`), `array<f32>` (the original, always-available
/// path), or a block-quantized `array<u32>` (see [`KvStorage::Q8_0`]).
/// Every read of any of the three already goes through `kv_read_k`/
/// `kv_read_v` (`f32`-returning either way), so the score/softmax/
/// weighted-sum math itself is identical regardless — only the storage
/// type, and hence the KV mirror's memory traffic, changes. `subgroup`
/// selects `attention_subgroup_blocks` over `attention_classic_blocks`
/// for the per-tile max/sum reductions — see `VulkanBackend::try_init`'s
/// own comment on its `subgroup_reduce` local for why this is opt-in.
pub fn shader_source_attention(
    kv_storage: KvStorage,
    subgroup: bool,
    max_head_dim: u32,
    paging: KvPaging,
) -> String {
    let (max_block, sum_block) = if subgroup {
        attention_subgroup_blocks()
    } else {
        attention_classic_blocks()
    };
    let subgroup_params = if subgroup {
        "@builtin(subgroup_invocation_id) subgroup_invocation_id: u32,\n    @builtin(subgroup_id) subgroup_id: u32,\n    @builtin(num_subgroups) num_subgroups: u32,"
    } else {
        ""
    };
    let (kv_bindings, kv_read_fns) = kv_storage.bindings_and_read_fns();
    let src = ATTENTION_SHADER_TEMPLATE
        .replace(
            "const MAX_HEAD_DIM: u32 = 2048u;",
            &format!("const MAX_HEAD_DIM: u32 = {max_head_dim}u;"),
        )
        .replace("%KV_ENABLE%", kv_storage.enable_directive())
        .replace("%KV_BINDINGS%", &kv_bindings)
        .replace("%KV_READ_FNS%", &kv_read_fns)
        .replace("%SUBGROUP_PARAMS%", subgroup_params)
        .replace("%MAX_REDUCE_BLOCK%", max_block)
        .replace("%SUM_REDUCE_BLOCK%", sum_block)
        .replace("%KV_PAGE_BINDING%", paging.binding())
        .replace("%KV_SLOT_FN%", paging.slot_fn())
        .replace("%QUERY_SETUP%", ATTENTION_SINGLE_QUERY_SETUP);
    paging.finish(src)
}

/// One workgroup per `(head, query)` instead of per head: `wid.y` selects the
/// query, and each one derives its **own** attention window from its absolute
/// position rather than being handed one in the uniform. That is the whole
/// difference between decoding a token and prefilling a prompt — every query
/// in a prompt attends over a different range — and it is what lets a prefill
/// layer's attention be a dispatch instead of a CPU loop over tokens.
///
/// The window rule mirrors `GemmaModel::attention_window` case for case; the
/// two must agree exactly, which `gpu_attention_prefill_matches_cpu_reference`
/// checks against the CPU implementation for causal, sliding-window, and
/// non-causal layers.
const ATTENTION_MULTI_QUERY_SETUP: &str = r#"
    let t = wid.y;
    let q_off = t * am.n_head * head_dim;
    let out_off = q_off;
    let pos = am.start_pos + t;
    var ws: u32 = 0u;
    var we: u32 = pos;
    if (am.causal == 0u) {
        if (am.n_swa > 0u) {
            let half = am.n_swa / 2u;
            ws = select(0u, pos - half, pos > half);
            we = min(pos + half, am.n_query - 1u);
        } else {
            ws = 0u;
            we = am.n_query - 1u;
        }
    } else if (am.n_swa > 0u) {
        ws = select(0u, pos - (am.n_swa - 1u), pos + 1u > am.n_swa);
    }
    let window_start = ws;
    let n_pos = we - ws + 1u;
"#;

/// The single-query counterpart: the caller already computed the window, and
/// there is exactly one query at offset zero.
const ATTENTION_SINGLE_QUERY_SETUP: &str = r#"
    let q_off = 0u;
    let out_off = 0u;
    let window_start = am.window_start;
    let n_pos = am.n_pos;
"#;

/// [`shader_source_attention`]'s multi-query variant — see
/// [`ATTENTION_MULTI_QUERY_SETUP`]. Deliberately the *same* template: the
/// online-softmax body, the KV reads, and the reductions are shared, so the
/// two cannot drift apart.
pub fn shader_source_attention_prefill(
    kv_storage: KvStorage,
    subgroup: bool,
    max_head_dim: u32,
    paging: KvPaging,
) -> String {
    shader_source_attention(kv_storage, subgroup, max_head_dim, paging).replace(
        ATTENTION_SINGLE_QUERY_SETUP.trim_end(),
        ATTENTION_MULTI_QUERY_SETUP.trim_end(),
    )
}

/// Split-k phase 1 of two. Same
/// per-tile online-softmax algorithm as [`ATTENTION_SHADER_TEMPLATE`]
/// (`score_at`, the tile loop, the rescale-and-merge update — all
/// unchanged line for line), but each workgroup now covers one `(head,
/// split)` pair instead of one whole head: `wid.x` selects the head (as
/// before), `wid.y` selects which of `am.k_num` roughly-equal slices of
/// `[0, n_pos)` this workgroup's tile loop runs over
/// (`split_start`/`split_end`, computed from `wid.y` and `am.k_num`).
/// A model with a low `n_head_kv` relative to `n_head` (an aggressive GQA
/// ratio) means the un-split kernel dispatches very few workgroups total
/// (one per query head), regardless of context length —
/// `_scratch_measure_attention_dispatch_cost` (`vulkan.rs`) isolates this
/// dispatch's own GPU time to check whether that's actually a meaningful
/// share of a decode layer's time before assuming it, the signature of an
/// occupancy-bound dispatch, not a compute-bound one, being worth
/// distinguishing from a dispatch that's merely doing little arithmetic.
/// `am.k_num` workgroups per
/// head instead of one raises that occupancy `k_num`-fold (`ATTN_SPLIT_K`
/// in `vulkan.rs`), the same split-k idea `flash_attn_split_k_reduce.comp`
/// implements in llama.cpp's own Vulkan backend (landed for the identical
/// reason, ["Implement split_k for coopmat2 flash
/// attention"](https://github.com/ggml-org/llama.cpp/pull/12627)).
///
/// Writes unnormalized partial results instead of the final softmax
/// output — this phase's own `(m, l, acc)` for its slice, not `acc / l`
/// — into `partial_ml`/`partial_acc` at index `h * am.k_num +
/// wid.y`, for [`ATTENTION_SPLIT_REDUCE_SHADER`] to merge. An empty
/// slice (`split_start >= split_end`, only possible when `am.n_pos <
/// am.k_num`, i.e. very early in a generation) leaves `m`/`l` at their
/// initial neutral values (`-1e30`/`0.0`) — the same identity element the
/// un-split kernel's own rescale-and-merge update already relies on
/// between tiles, so the reduce phase needs no special case for it.
///
/// Binding shape (3 read-only storage, 2 read-write storage, 1 uniform)
/// deliberately matches [`ATTENTION_SHADER_TEMPLATE`]'s own (`aq`/
/// `k_cache`/`v_cache` unchanged; `partial_ml`/`partial_acc` standing in
/// for `probs_scratch`/`aout`), so this reuses `VulkanBackend::
/// attn_bind_group_layout`/`attn_pipeline_layout` rather than needing new
/// ones.
const ATTENTION_SPLIT_SHADER_TEMPLATE: &str = r#"
%KV_ENABLE%
struct AttnSplitMeta {
    n_head: u32,
    n_head_kv: u32,
    head_dim: u32,
    window_start: u32,
    n_pos: u32,
    k_num: u32,
    scale: f32,
    // Where this sequence's block table starts, and how many positions one
    // page covers. Both read only by the paged form of this kernel; zero
    // otherwise.
    kv_page_base: u32,
    kv_page_tokens: u32,
    kv_ring_rows: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> aq: array<f32>;
%KV_BINDINGS%
@group(0) @binding(3) var<storage, read_write> partial_ml: array<f32>;
@group(0) @binding(4) var<storage, read_write> partial_acc: array<f32>;
@group(0) @binding(5) var<uniform> am: AttnSplitMeta;
%KV_PAGE_BINDING%

%KV_READ_FNS%
%KV_SLOT_FN%

const MAX_HEAD_DIM: u32 = 2048u;

var<workgroup> shared_reduce: array<f32, 64>;
var<workgroup> tile_probs: array<f32, 64>;
var<workgroup> acc: array<f32, MAX_HEAD_DIM>;

fn score_at(h: u32, kv_head: u32, p: u32) -> f32 {
    let head_dim = am.head_dim;
    let q_base = h * head_dim;
    let k_base = (kv_slot(p) * am.n_head_kv + kv_head) * head_dim;
    var s: f32 = 0.0;
    var d: u32 = 0u;
    loop {
        if (d >= head_dim) {
            break;
        }
        s = s + aq[q_base + d] * kv_read_k(k_base + d);
        d = d + 1u;
    }
    return s * am.scale;
}

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    %SUBGROUP_PARAMS%
) {
    let h = wid.x;
    let split_idx = wid.y;
    let local = lid.x;
    let group_size = am.n_head / am.n_head_kv;
    let kv_head = h / group_size;
    let head_dim = am.head_dim;
    let k_num = am.k_num;

    let split_len = (am.n_pos + k_num - 1u) / k_num;
    let split_start = split_idx * split_len;
    let split_end = min(split_start + split_len, am.n_pos);

    var zd: u32 = local;
    loop {
        if (zd >= head_dim) {
            break;
        }
        acc[zd] = 0.0;
        zd = zd + 64u;
    }

    var m: f32 = -1e30;
    var l: f32 = 0.0;

    if (split_start < split_end) {
        var tile_start: u32 = split_start;
        loop {
            if (tile_start >= split_end) {
                break;
            }
            let tile_len = min(64u, split_end - tile_start);
            let has_pos = local < tile_len;
            let p = am.window_start + tile_start + local;

            var my_score: f32 = -1e30;
            if (has_pos) {
                my_score = score_at(h, kv_head, p);
            }
            %MAX_REDUCE_BLOCK%

            var my_prob: f32 = 0.0;
            if (has_pos) {
                my_prob = exp(my_score - tile_max);
            }
            tile_probs[local] = my_prob;
            %SUM_REDUCE_BLOCK%

            let new_m = max(m, tile_max);
            let alpha_old = exp(m - new_m);
            let alpha_tile = exp(tile_max - new_m);
            l = l * alpha_old + tile_sum * alpha_tile;

            var d2: u32 = local;
            loop {
                if (d2 >= head_dim) {
                    break;
                }
                var tile_contribution: f32 = 0.0;
                var j: u32 = 0u;
                loop {
                    if (j >= tile_len) {
                        break;
                    }
                    let vp = am.window_start + tile_start + j;
                    let v_base = (kv_slot(vp) * am.n_head_kv + kv_head) * head_dim;
                    tile_contribution = tile_contribution + tile_probs[j] * kv_read_v(v_base + d2);
                    j = j + 1u;
                }
                acc[d2] = acc[d2] * alpha_old + alpha_tile * tile_contribution;
                d2 = d2 + 64u;
            }

            m = new_m;
            workgroupBarrier();
            tile_start = tile_start + 64u;
        }
    }

    let out_base = h * k_num + split_idx;
    if (local == 0u) {
        partial_ml[out_base * 2u] = m;
        partial_ml[out_base * 2u + 1u] = l;
    }
    var d3: u32 = local;
    loop {
        if (d3 >= head_dim) {
            break;
        }
        partial_acc[out_base * head_dim + d3] = acc[d3];
        d3 = d3 + 64u;
    }
}
"#;

pub fn shader_source_attention_split(
    kv_storage: KvStorage,
    subgroup: bool,
    max_head_dim: u32,
    paging: KvPaging,
) -> String {
    let (max_block, sum_block) = if subgroup {
        attention_subgroup_blocks()
    } else {
        attention_classic_blocks()
    };
    let subgroup_params = if subgroup {
        "@builtin(subgroup_invocation_id) subgroup_invocation_id: u32,\n    @builtin(subgroup_id) subgroup_id: u32,\n    @builtin(num_subgroups) num_subgroups: u32,"
    } else {
        ""
    };
    let (kv_bindings, kv_read_fns) = kv_storage.bindings_and_read_fns();
    let src = ATTENTION_SPLIT_SHADER_TEMPLATE
        .replace(
            "const MAX_HEAD_DIM: u32 = 2048u;",
            &format!("const MAX_HEAD_DIM: u32 = {max_head_dim}u;"),
        )
        .replace("%KV_ENABLE%", kv_storage.enable_directive())
        .replace("%KV_BINDINGS%", &kv_bindings)
        .replace("%KV_READ_FNS%", &kv_read_fns)
        .replace("%SUBGROUP_PARAMS%", subgroup_params)
        .replace("%MAX_REDUCE_BLOCK%", max_block)
        .replace("%SUM_REDUCE_BLOCK%", sum_block)
        .replace("%KV_PAGE_BINDING%", paging.binding())
        .replace("%KV_SLOT_FN%", paging.slot_fn());
    paging.finish(src)
}

/// **Cooperative-reduction** split-k phase 1 — on by default wherever the
/// adapter supports subgroups, opt out with `ORANGU_NO_ATTN_COOP=1`. A
/// from-scratch rewrite that targets the real bottleneck — the default
/// split kernel's *serial* per-thread `head_dim` dot product (one thread does
/// all `head_dim` MACs for one KV position, with the wave's 32 lanes reading K
/// at a `head_dim`-strided address each — badly uncoalesced) and its per-tile
/// `workgroupBarrier`s.
///
/// Here a **32-lane subgroup owns one KV position at a time, split across the
/// head dim**: lane `L` owns dims `{L, L+32, …}`, so consecutive lanes read
/// consecutive K/V elements — **fully coalesced** — and the score is a
/// barrier-free `subgroupAdd` of each lane's partial dot instead of a
/// 512-deep serial chain. `q`/`acc` live in registers (sized `HEAD_DIM/32`,
/// baked per-pipeline so the owned loops unroll to a compile-time trip count
/// and never spill to scratch). No workgroup shared memory, no
/// `workgroupBarrier`. Output layout (`partial_ml`/`partial_acc`) is identical
/// to [`ATTENTION_SPLIT_SHADER_TEMPLATE`], so the same phase-2
/// [`ATTENTION_SPLIT_REDUCE_SHADER`] merges it unchanged.
///
/// Subgroup-only and assumes a subgroup width `>= 32` (the 32 workgroup
/// threads always fall in one subgroup). `head_dim` must be a
/// multiple of 32 (gemma's 512/256 are). Not byte-identical to the serial
/// kernel (the `subgroupAdd` reduces the dot in a different order) — validated
/// by greedy-output match, like the flash/GQA variants.
/// Emits `count` straight-line `let {prefix}{i} = {read_fn}(kv_base + {i*32}u);`
/// bindings — one named scalar per owned dim, no loop.
///
/// This exists because a `for (var i = 0u; i < OWNED; i = i + 1u)` over the
/// owned dims does **not** compile to batched loads, even with `OWNED` baked in
/// and every value in registers. ACO reuses one destination register for the
/// load and emits a full `s_waitcnt vmcnt(0)` before each use, so the reads
/// serialise into `OWNED` dependent memory round trips per position — visible
/// only in `RADV_DEBUG=shaders`, since `shaderstats` reports no spills and no
/// scratch. Written out as separate `let` bindings the loads land in distinct
/// registers with no branch between them, and the wave issues all of them
/// before draining with counted `s_waitcnt vmcnt(N)`.
fn owned_dim_loads(prefix: &str, read_fn: &str, count: u32) -> String {
    (0..count)
        .map(|i| {
            format!(
                "        let {prefix}{i} = {read_fn}(kv_base + {}u);\n",
                i * 32
            )
        })
        .collect()
}

/// The left-to-right sum `q{0} * {k}0 + q{1} * {k}1 + …`, matching the
/// accumulation order of the serial loop it replaces so the result is
/// bit-identical to it.
fn owned_dim_dot(q_prefix: &str, k_prefix: &str, count: u32) -> String {
    (0..count)
        .map(|i| format!("{q_prefix}{i} * {k_prefix}{i}"))
        .collect::<Vec<_>>()
        .join(" + ")
}

/// **Cooperative-reduction** split-k phase 1 (the decode default). Targets the
/// classic split kernel's *serial* per-thread `head_dim` dot product (one
/// thread does all `head_dim` MACs for one KV position, with the wave's 32
/// lanes reading K at a `head_dim`-strided address each — badly uncoalesced)
/// and its per-tile `workgroupBarrier`s.
///
/// A **32-lane subgroup owns one KV position at a time, split across the head
/// dim**: lane `L` owns dims `{L, L+32, …}`, so consecutive lanes read
/// consecutive K/V elements — **fully coalesced** — and the score is a
/// barrier-free `subgroupAdd` of each lane's partial dot instead of a
/// `head_dim`-deep serial chain. Q and the accumulator live in registers, and
/// the owned dims are emitted **unrolled as named scalars** rather than as a
/// loop over an `array<f32, OWNED>` — see [`owned_dim_loads`] for why that is
/// not the same thing. No workgroup shared memory, no `workgroupBarrier`.
///
/// Output layout (`partial_ml`/`partial_acc`) is identical to
/// [`ATTENTION_SPLIT_SHADER_TEMPLATE`], so the same phase-2
/// [`ATTENTION_SPLIT_REDUCE_SHADER`] merges it unchanged.
///
/// Subgroup-only and assumes a subgroup width `>= 32` (the 32 workgroup threads
/// always fall in one subgroup, whatever the adapter's actual width). Requires
/// `head_dim % 32 == 0`. Not byte-identical to the *serial* kernel (the
/// `subgroupAdd` reduces the dot in a different order), but it is bit-identical
/// to the rolled cooperative kernel it replaces.
pub fn shader_source_attention_split_coop(
    kv_storage: KvStorage,
    head_dim: u32,
    paging: KvPaging,
) -> String {
    debug_assert_eq!(head_dim % 32, 0, "coop attention needs head_dim % 32 == 0");
    let owned = head_dim / 32;
    let (kv_bindings, kv_read_fns) = kv_storage.bindings_and_read_fns();
    let enable = kv_storage.enable_directive();
    let kv_page_binding = paging.binding();
    let kv_slot_fn = paging.slot_fn();

    let regs: String = (0..owned)
        .map(|i| {
            let off = i * 32;
            format!("    var q{i}: f32 = aq[q_base + {off}u + lane];\n    var a{i}: f32 = 0.0;\n")
        })
        .collect();
    let k_reads = owned_dim_loads("k", "kv_read_k", owned);
    let dot = owned_dim_dot("q", "k", owned);
    let v_reads = owned_dim_loads("v", "kv_read_v", owned);
    let acc: String = (0..owned)
        .map(|i| format!("        a{i} = a{i} * alpha + pw * v{i};\n"))
        .collect();

    // The per-position body, shared by both loop shapes below.
    let body = format!(
        "{k_reads}        let score = subgroupAdd({dot}) * am.scale;\n\
         \x20       let new_m = max(m, score);\n\
         \x20       let alpha = exp(m - new_m);\n\
         \x20       let pw = exp(score - new_m);\n\
         \x20       l = l * alpha + pw;\n\
         {v_reads}{acc}        m = new_m;\n"
    );

    // **Where the page lookup sits.**
    //
    // Contiguous keeps the single flat loop it has always had — one position
    // per iteration, no lookup, nothing to hoist.
    //
    // Paged splits it in two: an outer step per *page run* that resolves the
    // block table once, and an inner step over the positions inside that page.
    // A position's row is then an increment rather than a dependent load, which
    // is the whole point — the flat form made every position wait for
    // `kv_page_table[..]` before it could compute a KV address, and a
    // full-attention model pays that latency once per cached position per
    // layer. Measured on such a model, decode fell 22.1 -> 30.8 -> 32.1 tok/s
    // as the page size went 16 -> 64 -> 256, which is the same cost seen from
    // the other side: fewer lookups, less waiting.
    let pos_loop = match paging {
        KvPaging::Contiguous | KvPaging::Ring => format!(
            "    var pos: u32 = split_start;\n\
             \x20   loop {{\n\
             \x20       if (pos >= split_end) {{\n\
             \x20           break;\n\
             \x20       }}\n\
             \x20       let p = am.window_start + pos;\n\
             \x20       let kv_base = (kv_slot(p) * am.n_head_kv + kv_head) * HEAD_DIM + lane;\n\
             {body}        pos = pos + 1u;\n\
             \x20   }}\n"
        ),
        KvPaging::Paged => format!(
            "    var pos: u32 = split_start;\n\
             \x20   loop {{\n\
             \x20       if (pos >= split_end) {{\n\
             \x20           break;\n\
             \x20       }}\n\
             \x20       let p0 = am.window_start + pos;\n\
             \x20       let page = kv_page_table[am.kv_page_base + p0 / am.kv_page_tokens];\n\
             \x20       let in_page = p0 % am.kv_page_tokens;\n\
             \x20       let run = min(am.kv_page_tokens - in_page, split_end - pos);\n\
             \x20       let row0 = page * am.kv_page_tokens + in_page;\n\
             \x20       var j: u32 = 0u;\n\
             \x20       loop {{\n\
             \x20           if (j >= run) {{\n\
             \x20               break;\n\
             \x20           }}\n\
             \x20           let kv_base = ((row0 + j) * am.n_head_kv + kv_head) * HEAD_DIM + lane;\n\
             {body}            j = j + 1u;\n\
             \x20       }}\n\
             \x20       pos = pos + run;\n\
             \x20   }}\n"
        ),
    };

    let writes: String = (0..owned)
        .map(|i| {
            format!(
                "    partial_acc[out_base * HEAD_DIM + {}u + lane] = a{i};\n",
                i * 32
            )
        })
        .collect();

    let src = format!(
        r#"{enable}
struct AttnSplitMeta {{
    n_head: u32,
    n_head_kv: u32,
    head_dim: u32,
    window_start: u32,
    n_pos: u32,
    k_num: u32,
    scale: f32,
    kv_page_base: u32,
    kv_page_tokens: u32,
    kv_ring_rows: u32,
    _pad1: u32,
    _pad2: u32,
}}

@group(0) @binding(0) var<storage, read> aq: array<f32>;
{kv_bindings}
@group(0) @binding(3) var<storage, read_write> partial_ml: array<f32>;
@group(0) @binding(4) var<storage, read_write> partial_acc: array<f32>;
@group(0) @binding(5) var<uniform> am: AttnSplitMeta;
{kv_page_binding}

{kv_read_fns}
{kv_slot_fn}

const HEAD_DIM: u32 = {head_dim}u;

@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    let h = wid.x;
    let split_idx = wid.y;
    let lane = lid.x;
    let group_size = am.n_head / am.n_head_kv;
    let kv_head = h / group_size;
    let k_num = am.k_num;
    let q_base = h * HEAD_DIM;

    // Each lane owns dims {{ lane, lane+32, … }}. Stage the owned query
    // elements into registers and zero the owned accumulators.
{regs}
    let split_len = (am.n_pos + k_num - 1u) / k_num;
    let split_start = split_idx * split_len;
    let split_end = min(split_start + split_len, am.n_pos);

    var m: f32 = -1e30;
    var l: f32 = 0.0;

{pos_loop}

    let out_base = h * k_num + split_idx;
    if (lane == 0u) {{
        partial_ml[out_base * 2u] = m;
        partial_ml[out_base * 2u + 1u] = l;
    }}
{writes}}}
"#
    );
    paging.finish(src)
}

/// Multi-query attention with the **cooperative** reduction: 32 lanes split
/// `head_dim` between them for one `(head, query)` pair, so consecutive lanes
/// read consecutive K/V elements.
///
/// This is [`shader_source_attention_split_coop`] with the split-k machinery
/// removed — a prompt has queries to parallelise over, so it needs no splits —
/// and each query deriving its own window the way
/// [`ATTENTION_MULTI_QUERY_SETUP`] does.
///
/// The distinction from the classic template matters more here than anywhere
/// else: there, each lane walks its *own* K row, so a subgroup's 32 lanes touch
/// 32 different rows per step and every read is a separate transaction. That is
/// tolerable for one decode query and ruinous for a whole prompt's worth.
const ATTENTION_COOP_PREFILL_TEMPLATE: &str = r#"
%KV_ENABLE%
struct AttnMeta {
    n_head: u32,
    n_head_kv: u32,
    head_dim: u32,
    window_start: u32,
    n_pos: u32,
    capacity: u32,
    scale: f32,
    start_pos: u32,
    n_query: u32,
    n_swa: u32,
    causal: u32,
    kv_page_base: u32,
    kv_page_tokens: u32,
    kv_ring_rows: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> aq: array<f32>;
%KV_BINDINGS%
@group(0) @binding(3) var<storage, read_write> probs_scratch: array<f32>;
@group(0) @binding(4) var<storage, read_write> aout: array<f32>;
@group(0) @binding(5) var<uniform> am: AttnMeta;
%KV_PAGE_BINDING%

%KV_READ_FNS%
%KV_SLOT_FN%

const HEAD_DIM: u32 = %HEAD_DIM%u;
const OWNED: u32 = %OWNED%u;

@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let h = wid.x;
    let t = wid.y;
    let lane = lid.x;
    let group_size = am.n_head / am.n_head_kv;
    let kv_head = h / group_size;
    let q_base = (t * am.n_head + h) * HEAD_DIM;

    // Per-query window, same rule as `GemmaModel::attention_window`.
    let pos_abs = am.start_pos + t;
    var ws: u32 = 0u;
    var we: u32 = pos_abs;
    if (am.causal == 0u) {
        if (am.n_swa > 0u) {
            let half = am.n_swa / 2u;
            ws = select(0u, pos_abs - half, pos_abs > half);
            we = min(pos_abs + half, am.n_query - 1u);
        } else {
            ws = 0u;
            we = am.n_query - 1u;
        }
    } else if (am.n_swa > 0u) {
        ws = select(0u, pos_abs - (am.n_swa - 1u), pos_abs + 1u > am.n_swa);
    }

    var q_reg: array<f32, OWNED>;
    var acc_reg: array<f32, OWNED>;
    for (var i: u32 = 0u; i < OWNED; i = i + 1u) {
        q_reg[i] = aq[q_base + lane + i * 32u];
        acc_reg[i] = 0.0;
    }

    var m: f32 = -1e30;
    var l: f32 = 0.0;
    var p: u32 = ws;
    loop {
        if (p > we) {
            break;
        }
        let k_base = (kv_slot(p) * am.n_head_kv + kv_head) * HEAD_DIM;
        var partial: f32 = 0.0;
        for (var i: u32 = 0u; i < OWNED; i = i + 1u) {
            partial = partial + q_reg[i] * kv_read_k(k_base + lane + i * 32u);
        }
        let score = subgroupAdd(partial) * am.scale;
        let new_m = max(m, score);
        let alpha = exp(m - new_m);
        let pw = exp(score - new_m);
        l = l * alpha + pw;
        for (var i: u32 = 0u; i < OWNED; i = i + 1u) {
            acc_reg[i] = acc_reg[i] * alpha + pw * kv_read_v(k_base + lane + i * 32u);
        }
        m = new_m;
        p = p + 1u;
    }

    for (var i: u32 = 0u; i < OWNED; i = i + 1u) {
        aout[q_base + lane + i * 32u] = acc_reg[i] / l;
    }
}
"#;

/// Per-lane `f32` register budget the GQA prefill kernel sizes itself to.
/// Each query head a workgroup owns costs `head_dim / 32` registers for its
/// slice of Q and the same again for its accumulator, and the kernel spends
/// further registers on the staged K/V values and the per-head softmax state.
///
/// Sharing a KV read across more heads and keeping enough waves resident to
/// hide that read pull in opposite directions, and the useful range between
/// them is narrow: sweeping `ORANGU_GQA_HEADS` over an MQA-shaped model, one
/// head per workgroup (no sharing) and a whole eight-head group in one
/// workgroup were both clearly worse than the two settings between them, which
/// tied. This budget picks from inside that band at either head_dim — it is
/// well under the hardware ceiling on purpose, because the binding constraint
/// is occupancy, not correctness.
const GQA_PREFILL_REG_BUDGET: u32 = 64;

/// How many query heads of one KV group a single GQA prefill workgroup owns.
///
/// The kernel keeps every owned head's Q slice *and* running accumulator in
/// registers so a KV element read once serves all of them; that footprint is
/// `heads * (head_dim / 32) * 2` per lane and grows with both the group size
/// and the head dim. This returns the largest divisor of `group` that stays
/// inside [`GQA_PREFILL_REG_BUDGET`] — the fewest workgroups, hence the most
/// sharing, that still fits. `ORANGU_GQA_HEADS` pins it for measurement.
///
/// Splitting a group across several workgroups rather than across several
/// subgroups of one wide workgroup is deliberate: a 32-thread workgroup is at
/// most one subgroup whatever the adapter's subgroup size turns out to be, so
/// the `subgroupAdd` that reduces the score is correct without the kernel
/// having to know that size. A wider workgroup would have to assume it.
pub fn gqa_prefill_heads_per_workgroup(head_dim: u32, group: u32) -> u32 {
    if let Some(pinned) = std::env::var("ORANGU_GQA_HEADS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|&n| n >= 1 && n <= group && group.is_multiple_of(n))
    {
        return pinned;
    }
    let owned = (head_dim / 32).max(1);
    (1..=group)
        .rev()
        .filter(|heads| group.is_multiple_of(*heads))
        .find(|heads| heads * owned * 2 <= GQA_PREFILL_REG_BUDGET)
        .unwrap_or(1)
}

/// **GQA-sharing** multi-query attention: one workgroup per `(kv_head,
/// head-slice, query)` instead of per `(head, query)`, so the KV head that a
/// whole query-head group shares is read from global **once** for all the
/// heads in the slice rather than once per head.
///
/// [`ATTENTION_COOP_PREFILL_TEMPLATE`] already reads K and V coalesced, but
/// every query head reads them again for itself: a model whose `n_head_kv` is
/// far below its `n_head` therefore streams the same window `n_head /
/// n_head_kv` times per layer, and that redundancy — not the attention
/// arithmetic — is what the dispatch spends its time on. Here the 32 lanes
/// still split `head_dim` between them exactly as the cooperative kernel does
/// (lane `L` owns dims `{L, L+32, …}`, consecutive lanes on consecutive
/// elements), but each lane holds `heads` query slices at once: one
/// `kv_read_k` feeds `heads` partial dots, one `kv_read_v` feeds `heads`
/// accumulator updates.
///
/// `heads` is [`gqa_prefill_heads_per_workgroup`] — the whole group when its
/// registers fit, a divisor of it when they do not, with the remaining slices
/// covered by further workgroups along `x`. Every Q slice, accumulator and
/// per-head `(m, l)` is emitted as a **named scalar**, not an indexed
/// `array<f32, N>` local, because only the former reliably lands in registers.
///
/// Each query still derives its own window from `start_pos + t`, the same four
/// cases [`ATTENTION_MULTI_QUERY_SETUP`] and `GemmaModel::attention_window`
/// implement, and the per-position accumulation order is unchanged from the
/// cooperative kernel — so this produces bit-identical output to it, and only
/// changes which workgroup reads what.
pub fn shader_source_attention_prefill_gqa(
    kv_storage: KvStorage,
    head_dim: u32,
    group: u32,
    heads: u32,
    paging: KvPaging,
) -> String {
    debug_assert_eq!(head_dim % 32, 0, "gqa attention needs head_dim % 32 == 0");
    debug_assert!(heads >= 1 && group.is_multiple_of(heads));
    let owned = head_dim / 32;
    let (kv_bindings, kv_read_fns) = kv_storage.bindings_and_read_fns();
    let enable = kv_storage.enable_directive();
    let kv_page_binding = paging.binding();
    let kv_slot_fn = paging.slot_fn();

    let mut regs = String::new();
    for j in 0..heads {
        for i in 0..owned {
            let off = j * head_dim + i * 32;
            regs += &format!("    var q{j}_{i}: f32 = aq[q_base + {off}u + lane];\n");
            regs += &format!("    var a{j}_{i}: f32 = 0.0;\n");
        }
        regs += &format!("    var m{j}: f32 = -1e30;\n    var l{j}: f32 = 0.0;\n");
    }

    // One K read per owned dim, shared by every head's partial dot. The dot is
    // summed left to right, matching the cooperative kernel's serial loop.
    let k_reads = owned_dim_loads("k", "kv_read_k", owned);
    let mut scores = String::new();
    for j in 0..heads {
        let dot = owned_dim_dot(&format!("q{j}_"), "k", owned);
        scores += &format!("        let s{j} = subgroupAdd({dot}) * am.scale;\n");
    }
    // Per-head online-softmax update, then one V read per owned dim shared by
    // every head's accumulator rescale.
    let mut softmax = String::new();
    for j in 0..heads {
        softmax += &format!(
            "        let nm{j} = max(m{j}, s{j});\n\
             \x20       let al{j} = exp(m{j} - nm{j});\n\
             \x20       let pw{j} = exp(s{j} - nm{j});\n\
             \x20       l{j} = l{j} * al{j} + pw{j};\n\
             \x20       m{j} = nm{j};\n"
        );
    }
    // All V loads first, then every head's accumulator rescale — the loads are
    // independent of each other and of the rescales, so issuing them as one
    // block is what lets them go in flight together (see `owned_dim_loads`).
    let v_reads = owned_dim_loads("v", "kv_read_v", owned);
    let acc: String = (0..owned)
        .flat_map(|i| {
            (0..heads)
                .map(move |j| format!("        a{j}_{i} = a{j}_{i} * al{j} + pw{j} * v{i};\n"))
        })
        .collect();

    let mut writes = String::new();
    for j in 0..heads {
        for i in 0..owned {
            let off = j * head_dim + i * 32;
            writes += &format!("    aout[q_base + {off}u + lane] = a{j}_{i} / l{j};\n");
        }
    }

    let src = format!(
        r#"{enable}
struct AttnMeta {{
    n_head: u32,
    n_head_kv: u32,
    head_dim: u32,
    window_start: u32,
    n_pos: u32,
    capacity: u32,
    scale: f32,
    start_pos: u32,
    n_query: u32,
    n_swa: u32,
    causal: u32,
    kv_page_base: u32,
    kv_page_tokens: u32,
    kv_ring_rows: u32,
    _pad1: u32,
    _pad2: u32,
}}

@group(0) @binding(0) var<storage, read> aq: array<f32>;
{kv_bindings}
@group(0) @binding(3) var<storage, read_write> probs_scratch: array<f32>;
@group(0) @binding(4) var<storage, read_write> aout: array<f32>;
@group(0) @binding(5) var<uniform> am: AttnMeta;
{kv_page_binding}

{kv_read_fns}
{kv_slot_fn}

const HEAD_DIM: u32 = {head_dim}u;
const GROUP: u32 = {group}u;
const HEADS: u32 = {heads}u;

@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    // `x` enumerates (kv_head, head-slice); `y` the query token.
    let slices = GROUP / HEADS;
    let kv_head = wid.x / slices;
    let slice = wid.x - kv_head * slices;
    let t = wid.y;
    let lane = lid.x;
    // First query head this workgroup owns; the rest follow contiguously, so
    // one `q_base` plus a constant offset addresses all of them.
    let h0 = kv_head * GROUP + slice * HEADS;
    let q_base = (t * am.n_head + h0) * HEAD_DIM;

    // Per-query window, same rule as `GemmaModel::attention_window`.
    let pos_abs = am.start_pos + t;
    var ws: u32 = 0u;
    var we: u32 = pos_abs;
    if (am.causal == 0u) {{
        if (am.n_swa > 0u) {{
            let half = am.n_swa / 2u;
            ws = select(0u, pos_abs - half, pos_abs > half);
            we = min(pos_abs + half, am.n_query - 1u);
        }} else {{
            ws = 0u;
            we = am.n_query - 1u;
        }}
    }} else if (am.n_swa > 0u) {{
        ws = select(0u, pos_abs - (am.n_swa - 1u), pos_abs + 1u > am.n_swa);
    }}

{regs}
    var p: u32 = ws;
    loop {{
        if (p > we) {{
            break;
        }}
        let kv_base = (kv_slot(p) * am.n_head_kv + kv_head) * HEAD_DIM + lane;
{k_reads}{scores}{softmax}{v_reads}{acc}        p = p + 1u;
    }}

{writes}}}
"#
    );
    paging.finish(src)
}

/// The **tiled** prefill attention kernel: one workgroup per (query tile,
/// head), the tile's queries staged once in workgroup memory, the window
/// walked in blocks of positions, every thread owning a few rows × a few
/// columns × a slice of `head_dim`.
///
/// The per-query kernels above walk the window one position at a time and
/// pay a cross-lane reduction and an `exp` per head per position for a
/// handful of multiply-adds; every query also streams K and V for itself.
/// Here a K or V element loaded from the cache feeds one multiply-add per
/// owned row, the scores' only cross-lane work is a three-step shuffle over
/// the eight lanes that split `head_dim`, and the online softmax runs
/// **per thread over its own columns** — a partial `(m, l, o)` over a subset
/// of positions is a valid partial softmax, so the lanes that share a row
/// merge their partials once at the end instead of agreeing on a running
/// maximum every block.
///
/// Thread layout (four row groups of `lanes` threads): `row_tid` picks a
/// group of `rows` rows, `d_tid = tid % dlanes` one of the `head_dim`
/// slices (eight elements per lane per step, adjacent lanes adjacent in
/// memory, so a position's row is read as one contiguous run), `col_tid`
/// one of the `lanes / dlanes` column phases. The tile is `4 * rows`
/// queries by `cols` × phases positions per block; the accumulator is
/// `rows * head_dim / dlanes` registers and the scores `rows * cols`, which
/// is what bounds both — see [`FaPrefillTile`].
///
/// Every row derives its own window from `start_pos + t` (the same four cases
/// as [`ATTENTION_MULTI_QUERY_SETUP`]); a position outside a row's window
/// scores `-inf`, which the softmax turns into exactly zero, and the block
/// loop runs from the tile's earliest window start to its latest end. Rows
/// past `n_query` in a partial last tile compute on a clamped token and are
/// not written.
///
/// Q is read pre-scaled into workgroup memory as `vec4`s with a padded stride;
/// K and V are read straight from the cache eight elements at a time
/// (`kv_read_k8`/`v8`, per storage type), never staged — the four row groups
/// of a workgroup reread a position through the cache. The probabilities are
/// recomputed from the scores where they are used rather than kept, so the
/// live set is the accumulator and the scores.
pub fn shader_source_attention_prefill_tiled(
    kv_storage: KvStorage,
    head_dim: u32,
    paging: KvPaging,
    tile: FaPrefillTile,
) -> String {
    debug_assert_eq!(
        head_dim % (8 * tile.dlanes),
        0,
        "tiled attention needs head_dim % (8 * dlanes) == 0"
    );
    let rows = tile.rows;
    let cols = tile.cols;
    let lanes = tile.lanes;
    let dl = tile.dlanes;
    let col_lanes = lanes / dl;
    let wg = 4 * lanes;
    let br = rows * 4;
    let bc = cols * col_lanes;
    let dv = head_dim / (4 * dl); // vec4s of head_dim per thread
    let enable = kv_storage.enable_directive();
    let kv_page_binding = paging.binding();
    let kv_slot_fn = paging.slot_fn();
    let (kv_bindings, kv_read4) = match kv_storage {
        KvStorage::F32 => (
            "@group(0) @binding(1) var<storage, read> k_cache: array<vec4<f32>>;\n\
             @group(0) @binding(2) var<storage, read> v_cache: array<vec4<f32>>;"
                .to_string(),
            "fn kv_read_k8(e: u32) -> array<vec4<f32>, 2> { return array<vec4<f32>, 2>(k_cache[e / 4u], k_cache[e / 4u + 1u]); }\n\
             fn kv_read_v8(e: u32) -> array<vec4<f32>, 2> { return array<vec4<f32>, 2>(v_cache[e / 4u], v_cache[e / 4u + 1u]); }"
                .to_string(),
        ),
        // Eight halves as one 128-bit load, unpacked in pairs.
        KvStorage::F16 => (
            "@group(0) @binding(1) var<storage, read> k_cache: array<vec4<u32>>;\n\
             @group(0) @binding(2) var<storage, read> v_cache: array<vec4<u32>>;"
                .to_string(),
            "fn h8(w: vec4<u32>) -> array<vec4<f32>, 2> { return array<vec4<f32>, 2>(vec4<f32>(unpack2x16float(w.x), unpack2x16float(w.y)), vec4<f32>(unpack2x16float(w.z), unpack2x16float(w.w))); }\n\
             fn kv_read_k8(e: u32) -> array<vec4<f32>, 2> { return h8(k_cache[e / 8u]); }\n\
             fn kv_read_v8(e: u32) -> array<vec4<f32>, 2> { return h8(v_cache[e / 8u]); }"
                .to_string(),
        ),
        // `KV_QUANTIZE_Q8_0_SHADER`'s layout: 9 words per 32 values, the
        // scale then the bytes; four consecutive values are one word.
        KvStorage::Q8_0 => (
            "@group(0) @binding(1) var<storage, read> k_cache: array<u32>;\n\
             @group(0) @binding(2) var<storage, read> v_cache: array<u32>;"
                .to_string(),
            "fn q8x4(d: f32, w: u32) -> vec4<f32> {\n    \
                 return d * vec4<f32>(f32(i32(w << 24u) >> 24u), f32(i32(w << 16u) >> 24u), f32(i32(w << 8u) >> 24u), f32(i32(w) >> 24u));\n}\n\
             fn kv_read_k8(e: u32) -> array<vec4<f32>, 2> {\n    \
                 let b = (e / 32u) * 9u;\n    \
                 let d = bitcast<f32>(k_cache[b]);\n    \
                 let w = b + 1u + (e % 32u) / 4u;\n    \
                 return array<vec4<f32>, 2>(q8x4(d, k_cache[w]), q8x4(d, k_cache[w + 1u]));\n}\n\
             fn kv_read_v8(e: u32) -> array<vec4<f32>, 2> {\n    \
                 let b = (e / 32u) * 9u;\n    \
                 let d = bitcast<f32>(v_cache[b]);\n    \
                 let w = b + 1u + (e % 32u) / 4u;\n    \
                 return array<vec4<f32>, 2>(q8x4(d, v_cache[w]), q8x4(d, v_cache[w + 1u]));\n}"
                .to_string(),
        ),
    };

    // Registers: accumulators, per-row running max and sum, row windows.
    let mut decl = String::new();
    for r in 0..rows {
        for d in 0..dv {
            decl += &format!("    var o{r}_{d}: vec4<f32> = vec4<f32>(0.0);\n");
        }
        decl += &format!("    var m{r}: f32 = NEG_HALF_MAX;\n    var l{r}: f32 = 0.0;\n");
        decl += &format!("    let tr{r} = min(tb + {r}u, am.n_query - 1u);\n");
        decl +=
            &format!("    let ws{r} = window_start(tr{r});\n    let we{r} = window_end(tr{r});\n");
    }

    let mut s = String::new();
    for r in 0..rows {
        for c in 0..cols {
            s += &format!("        var s{r}_{c}: f32 = 0.0;\n");
        }
    }
    // Each column's K element base; the position itself is `pb + 4c + col_tid`.
    for c in 0..cols {
        s += &format!(
            "        let kb{c} = (kv_row(min(pb + {}u + col_tid, we_max)) * am.n_head_kv + kv_head) * HEAD_DIM + 8u * d_tid;\n",
            col_lanes * c
        );
    }
    // QK: per head_dim slice, the rows' Q from workgroup memory, then every
    // column's K once for all rows. A column past the window's end reads
    // a clamped row and is masked below.
    // Owned slot `d` (a vec4) sits at element `8 * (d / 2 * 8 + d_tid) + 4 *
    // (d % 2)`: a thread's two slots of a step are adjacent, one 8-element
    // load, and the eight lanes' loads are contiguous.
    let slot = |d: u32| -> String { format!("{}u + 2u * d_tid + {}u", 2 * dl * (d / 2), d % 2) };
    for d2 in 0..dv / 2 {
        for d in [2 * d2, 2 * d2 + 1] {
            for r in 0..rows {
                s += &format!(
                    "        let q{r}_{d} = qsh[(row_tid * ROWS + {r}u) * QSTRIDE + {}];\n",
                    slot(d)
                );
            }
        }
        for c in 0..cols {
            s += &format!(
                "        {{\n            let k = kv_read_k8(kb{c} + {}u);\n",
                8 * dl * d2
            );
            for r in 0..rows {
                s += &format!(
                    "            s{r}_{c} = s{r}_{c} + dot(q{r}_{}, k[0]) + dot(q{r}_{}, k[1]);\n",
                    2 * d2,
                    2 * d2 + 1
                );
            }
            s += "        }\n";
        }
    }
    // Scores summed over the eight head_dim lanes, then masked per row.
    for r in 0..rows {
        for c in 0..cols {
            let mut step = dl / 2;
            while step >= 1 {
                s += &format!(
                    "        s{r}_{c} = s{r}_{c} + subgroupShuffleXor(s{r}_{c}, {step}u);\n"
                );
                step /= 2;
            }
            s += &format!(
                "        s{r}_{c} = select(neg_inf(), s{r}_{c}, pb + {}u + col_tid >= ws{r} && pb + {}u + col_tid <= we{r});\n",
                col_lanes * c,
                col_lanes * c
            );
        }
    }
    // Online softmax per row over this thread's columns: the running max
    // moves, the accumulator and sum are rescaled.
    for r in 0..rows {
        let mut mx = format!("s{r}_0");
        for c in 1..cols {
            mx = format!("max({mx}, s{r}_{c})");
        }
        s += &format!(
            "        let nm{r} = max(m{r}, {mx});\n        let em{r} = exp(m{r} - nm{r});\n        m{r} = nm{r};\n        l{r} = l{r} * em{r};\n"
        );
        for d in 0..dv {
            s += &format!("        o{r}_{d} = o{r}_{d} * em{r};\n");
        }
    }
    // PV: every column's V once for all rows, its probabilities taken from
    // the scores here.
    for c in 0..cols {
        s += &format!(
            "        if (pb + {}u + col_tid <= we_max) {{\n",
            col_lanes * c
        );
        for r in 0..rows {
            s += &format!(
                "            let pw{r} = exp(s{r}_{c} - m{r});\n            l{r} = l{r} + pw{r};\n"
            );
        }
        for d2 in 0..dv / 2 {
            s += &format!(
                "            let v{d2} = kv_read_v8(kb{c} + {}u);\n",
                8 * dl * d2
            );
            for (i, d) in [2 * d2, 2 * d2 + 1].into_iter().enumerate() {
                for r in 0..rows {
                    s += &format!(
                        "            o{r}_{d} = fma(vec4<f32>(pw{r}), v{d2}[{i}], o{r}_{d});\n"
                    );
                }
            }
        }
        s += "        }\n";
    }
    // Merge the column phases' partials, then write. Phases 0..4 sit within
    // 32 lanes and merge by shuffle; with eight phases the upper four are a
    // second 32-lane half, merged by shuffle when the subgroup spans all 64
    // lanes and through workgroup memory (over the staged Q, no longer
    // needed) when it does not: each half reduces to its own `(max, sum,
    // acc)` first, then the lower half folds the upper's in.
    let wide = lanes == 64;
    // Sum (or max) `v` over the column phases: the steps within 32 lanes
    // unconditionally, the 32-lane step only on a 64-wide subgroup.
    let fold = |v: &str, op: &str| -> String {
        let mut o = String::new();
        let mut step = dl;
        while step < 32 {
            o += &format!("    {v} = {op}({v}, subgroupShuffleXor({v}, {step}u));\n");
            step *= 2;
        }
        if wide {
            o += &format!(
                "    if (sg_size >= 64u) {{ {v} = {op}({v}, subgroupShuffleXor({v}, 32u)); }}\n"
            );
        }
        o
    };
    let mut fin = String::new();
    for r in 0..rows {
        fin += &format!("    var gm{r} = m{r};\n");
        fin += &fold(&format!("gm{r}"), "max");
        fin += &format!("    let f{r} = exp(m{r} - gm{r});\n    var gl{r} = l{r} * f{r};\n");
        fin += &fold(&format!("gl{r}"), "fadd");
        for d in 0..dv {
            fin += &format!("    o{r}_{d} = o{r}_{d} * f{r};\n");
            fin += &fold(&format!("o{r}_{d}"), "vadd");
        }
    }
    if wide {
        let mut lds = String::new();
        lds += &format!(
            "        workgroupBarrier();\n        let xb = (row_tid * {dl}u + d_tid) * (ROWS * XROW);\n        if (col_tid == {}u) {{\n",
            col_lanes / 2
        );
        for r in 0..rows {
            lds += &format!(
                "            qsh[xb + {r}u * XROW + {dv}u] = vec4<f32>(gm{r}, gl{r}, 0.0, 0.0);\n"
            );
            for d in 0..dv {
                lds += &format!("            qsh[xb + {r}u * XROW + {d}u] = o{r}_{d};\n");
            }
        }
        lds += &format!(
            "        }}\n        workgroupBarrier();\n        if (col_tid < {}u) {{\n",
            col_lanes / 2
        );
        for r in 0..rows {
            lds += &format!(
                "            let u{r} = qsh[xb + {r}u * XROW + {dv}u];\n            let jm{r} = max(gm{r}, u{r}.x);\n            let fl{r} = exp(gm{r} - jm{r});\n            let fu{r} = exp(u{r}.x - jm{r});\n            gl{r} = gl{r} * fl{r} + u{r}.y * fu{r};\n"
            );
            for d in 0..dv {
                lds += &format!(
                    "            o{r}_{d} = o{r}_{d} * fl{r} + qsh[xb + {r}u * XROW + {d}u] * fu{r};\n"
                );
            }
        }
        lds += "        }\n";
        fin += &format!("    if (sg_size < 64u) {{\n{lds}    }}\n");
    }
    for r in 0..rows {
        fin += &format!(
            "    let inv{r} = select(1.0 / gl{r}, 0.0, gl{r} == 0.0);\n    if (col_tid == 0u && tb + {r}u < am.n_query) {{\n"
        );
        for d in 0..dv {
            fin += &format!(
                "        aout[((tb + {r}u) * am.n_head + h) * (HEAD_DIM / 4u) + {}] = o{r}_{d} * inv{r};\n",
                slot(d)
            );
        }
        fin += "    }\n";
    }

    let src = format!(
        r#"{enable}
struct AttnMeta {{
    n_head: u32,
    n_head_kv: u32,
    head_dim: u32,
    window_start: u32,
    n_pos: u32,
    capacity: u32,
    scale: f32,
    start_pos: u32,
    n_query: u32,
    n_swa: u32,
    causal: u32,
    kv_page_base: u32,
    kv_page_tokens: u32,
    kv_ring_rows: u32,
    _pad1: u32,
    _pad2: u32,
}}

@group(0) @binding(0) var<storage, read> aq: array<vec4<f32>>;
{kv_bindings}
@group(0) @binding(3) var<storage, read_write> probs_scratch: array<f32>;
@group(0) @binding(4) var<storage, read_write> aout: array<vec4<f32>>;
@group(0) @binding(5) var<uniform> am: AttnMeta;
{kv_page_binding}

{kv_read4}
{kv_slot_fn}
fn kv_row(p: u32) -> u32 {{ return kv_slot(p); }}

const HEAD_DIM: u32 = {head_dim}u;
const BR: u32 = {br}u;
const BC: u32 = {bc}u;
const ROWS: u32 = {rows}u;
const LANES: u32 = {lanes}u;
const DL: u32 = {dl}u;
const QSTRIDE: u32 = {qstride}u;
const XROW: u32 = {xrow}u;
// `-inf` and `-FLT_MAX / 2`: a masked score and the running maximum's
// start — far enough apart from anything finite that `exp(a - b)` is 0 or
// 1 exactly, never a NaN.
fn neg_inf() -> f32 {{ return bitcast<f32>(0xFF800000u); }}
fn fadd(a: f32, b: f32) -> f32 {{ return a + b; }}
fn vadd(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {{ return a + b; }}
const NEG_HALF_MAX: f32 = -1.7014117e38;
var<workgroup> qsh: array<vec4<f32>, {qsh_len}>;

// Per-query window, same rule as `GemmaModel::attention_window`.
fn window_start(t: u32) -> u32 {{
    let pos = am.start_pos + t;
    if (am.causal == 0u) {{
        if (am.n_swa > 0u) {{
            let half = am.n_swa / 2u;
            return select(0u, pos - half, pos > half);
        }}
        return 0u;
    }}
    if (am.n_swa > 0u) {{
        return select(0u, pos - (am.n_swa - 1u), pos + 1u > am.n_swa);
    }}
    return 0u;
}}
fn window_end(t: u32) -> u32 {{
    let pos = am.start_pos + t;
    if (am.causal == 0u) {{
        if (am.n_swa > 0u) {{
            return min(pos + am.n_swa / 2u, am.n_query - 1u);
        }}
        return am.n_query - 1u;
    }}
    return pos;
}}

@compute @workgroup_size({wg})
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(subgroup_size) sg_size: u32,
) {{
    let t0 = wid.x * BR;
    let h = wid.y;
    let kv_head = h / (am.n_head / am.n_head_kv);
    let tid = lid.x;
    let row_tid = tid / LANES;
    let col_tid = (tid % LANES) / DL;
    let d_tid = tid % DL;
    // This thread's first row (token) of the tile.
    let tb = t0 + row_tid * ROWS;

    // The tile's queries, scaled, one vec4 per thread per step.
    var i: u32 = tid;
    loop {{
        if (i >= BR * (HEAD_DIM / 4u)) {{ break; }}
        let r = i / (HEAD_DIM / 4u);
        let d = i % (HEAD_DIM / 4u);
        let t = min(t0 + r, am.n_query - 1u);
        let qv = aq[(t * am.n_head + h) * (HEAD_DIM / 4u) + d] * am.scale;
{q_stage}        i = i + {wg}u;
    }}
    workgroupBarrier();

{decl}
    let ws_min = window_start(t0);
    let we_max = window_end(min(t0 + BR - 1u, am.n_query - 1u));
    var pb: u32 = (ws_min / BC) * BC;
    loop {{
        if (pb > we_max) {{ break; }}
{s}        pb = pb + BC;
    }}

{fin}}}
"#,
        qstride = head_dim / 4 + 1,
        qsh_len = (br * (head_dim / 4 + 1)).max(if wide { 4 * dl * rows * (dv + 1) } else { 0 }),
        q_stage = "        qsh[r * QSTRIDE + d] = qv;\n",
        lanes = lanes,
        dl = dl,
        xrow = dv + 1,
        wg = wg,
    );
    paging.finish(src)
}

/// The tiled prefill kernel's per-thread tile: `rows` query rows and `cols`
/// positions per block (the workgroup covers four of each).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FaPrefillTile {
    pub rows: u32,
    pub cols: u32,
    /// Lanes per row group, 32 or 64: `dlanes` `head_dim` lanes by the
    /// column phases that fill the rest; the workgroup is four row groups.
    pub lanes: u32,
    /// Lanes that split `head_dim`, 8 or 16 — each owns `head_dim / dlanes`
    /// elements, so more of them means fewer accumulator registers per row.
    pub dlanes: u32,
}

impl FaPrefillTile {
    /// The tile for a `head_dim`: the accumulator is `rows * head_dim /
    /// dlanes` registers per thread. Two rows was the measured optimum at
    /// both `head_dim` 256 and 512 — four rows halve the loads per
    /// multiply-add but the registers they hold cost more occupancy than
    /// that buys, whichever way the lanes are split, and one row at 512
    /// (a 64-register accumulator) was a third slower than two. Narrower
    /// heads take four. `ORANGU_FA_ROWS`, `ORANGU_FA_COLS`,
    /// `ORANGU_FA_LANES` and `ORANGU_FA_DLANES` pin the tile for measurement.
    pub fn for_head_dim(head_dim: u32) -> Self {
        let pinned = |name: &str, allowed: &[u32]| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .filter(|n| allowed.contains(n))
        };
        Self {
            rows: pinned("ORANGU_FA_ROWS", &[1, 2, 3, 4, 8]).unwrap_or(if head_dim <= 128 {
                4
            } else {
                2
            }),
            cols: pinned("ORANGU_FA_COLS", &[2, 4, 8]).unwrap_or(8),
            lanes: pinned("ORANGU_FA_LANES", &[32, 64]).unwrap_or(32),
            dlanes: pinned("ORANGU_FA_DLANES", &[8, 16]).unwrap_or(8),
        }
    }
}

/// Builds [`ATTENTION_COOP_PREFILL_TEMPLATE`] for a specific `head_dim`
/// (baked in, so the owned loops unroll). Requires subgroups and
/// `head_dim % 32 == 0`; the caller falls back to
/// [`shader_source_attention_prefill`] otherwise.
pub fn shader_source_attention_prefill_coop(
    kv_storage: KvStorage,
    head_dim: u32,
    paging: KvPaging,
) -> String {
    debug_assert_eq!(head_dim % 32, 0, "coop attention needs head_dim % 32 == 0");
    let (kv_bindings, kv_read_fns) = kv_storage.bindings_and_read_fns();
    let src = ATTENTION_COOP_PREFILL_TEMPLATE
        .replace("%HEAD_DIM%", &head_dim.to_string())
        .replace("%OWNED%", &(head_dim / 32).to_string())
        .replace("%KV_ENABLE%", kv_storage.enable_directive())
        .replace("%KV_BINDINGS%", &kv_bindings)
        .replace("%KV_READ_FNS%", &kv_read_fns)
        .replace("%KV_PAGE_BINDING%", paging.binding())
        .replace("%KV_SLOT_FN%", paging.slot_fn());
    paging.finish(src)
}

/// Bytes the flash kernel's *other* workgroup arrays occupy, beside
/// `k_shmem`: `shared_reduce` and `tile_probs` (64 `f32` each) plus `acc`,
/// which is sized to this layer's `head_dim`.
///
/// Subtracted from the device's shared-memory limit rather than absorbed into
/// a round-number budget, because on a device whose limit is tight the
/// difference between "the tile fits" and "pipeline creation fails" is
/// exactly these three kilobytes.
fn flash_fixed_lds_bytes(head_dim: u32) -> u32 {
    (64 + 64 + head_dim) * 4
}

/// Positions per K-tile the **flash** split kernel
/// ([`ATTENTION_SPLIT_FLASH_SHADER_TEMPLATE`]) stages into LDS: as many as the
/// padded `f16` `k_shmem` (`tile_pos * (head_dim + 1) * 2` bytes) can have
/// without the workgroup's total shared memory exceeding `lds_limit`. Clamped
/// to a power of two in `8..=64`. The workgroup is always 64 threads, so
/// `tile_pos < 64` (only for large `head_dim`, e.g. gemma's `512`
/// full-attention layers) simply leaves the high lanes idle during the *score*
/// phase; they still cooperate on the coalesced staging and the
/// `head_dim`-strided `acc`/V loops.
///
/// # The limit is the device's, not a constant
///
/// This used to size the tile against a fixed ~34 KB budget. That is a
/// perfectly good number on a device with 64 KiB of shared memory per
/// workgroup and an impossible one on a device with 32 KiB — where the tile it
/// picks for `head_dim = 512` is over the ceiling before `acc` is even
/// counted, so the pipeline cannot be created at all. The flag would not be
/// slow there; it would fail.
///
/// A hardcoded budget also silently wastes headroom in the other direction, on
/// any device with more. Taking the number from
/// `wgpu::Limits::max_compute_workgroup_storage_size` makes the kernel fit
/// whatever it is actually running on, which is the only version of this that
/// is portable rather than tuned-for-one-machine.
///
/// `ORANGU_FLASH_LDS_KB` still overrides, to trade tile size for occupancy (a
/// full budget leaves one workgroup resident per SIMD, ~8 KB leaves ~7). A
/// smaller tile means more tiles and more barriers but higher occupancy. An
/// override *above* the device limit is clamped rather than obeyed — the
/// alternative is a flag whose effect is a failed pipeline.
fn flash_tile_positions(head_dim: u32, lds_limit: u32) -> u32 {
    let stride = head_dim + 1;
    let fixed = flash_fixed_lds_bytes(head_dim);
    let device_budget = lds_limit.saturating_sub(fixed);
    let requested =
        super::env_tuning_value("ORANGU_FLASH_LDS_KB", 0u32, "a positive integer", |kb| {
            kb > 0
        })
        .saturating_mul(1024);
    // `0` is the "unset" stand-in for the helper, so "no override" means the
    // whole device budget.
    let budget_bytes = if requested == 0 {
        device_budget
    } else {
        requested.min(device_budget)
    };
    let max_pos = (budget_bytes / 2 / stride).max(1);
    let mut tp = 64u32;
    while tp > max_pos && tp > 8 {
        tp /= 2;
    }
    tp.clamp(8, 64)
}

/// **Flash** variant of [`ATTENTION_SPLIT_SHADER_TEMPLATE`] (split-k phase 1).
/// Byte-for-byte the same online-softmax algorithm, the same 6-binding shape,
/// the same `AttnSplitMeta` uniform, the same `(head, split)` dispatch, and the
/// same unnormalized `partial_ml`/`partial_acc` outputs — so it drops straight
/// into `record_fused_attention`'s Pass C and [`ATTENTION_SPLIT_REDUCE_SHADER`]
/// merges it unchanged. **One difference, and it is the whole point:** the
/// un-split/split kernels compute each candidate position's `q·k` by reading
/// that position's `head_dim`-long K row *directly from global memory* inside
/// `score_at` — and with one thread per position, adjacent threads read K rows
/// `n_head_kv * head_dim` elements apart, i.e. **completely uncoalesced** (each
/// lane touches a different cache line for the same `d`). As the KV range grows
/// this uncoalesced K traffic comes to dominate the attention dispatch, while
/// the rest of the decode token is independent of context length.
///
/// This kernel instead **cooperatively stages each K tile into shared memory
/// with coalesced loads** (all 64 lanes stream `tile_len * head_dim` contiguous
/// f16 values, the classic flash-attention tiling llama.cpp's own WGSL
/// `flash_attn_tile.wgsl` `load_k_tile_block` does), then each lane computes its
/// position's score by reading K from LDS. The `k_shmem` row stride is padded to
/// `head_dim + 1` so the score read `k_shmem[local * KV_STRIDE + d]` lands in a
/// distinct LDS bank per lane (stride ≡ 1 (mod 32) for `head_dim` a multiple of
/// 32) instead of a 32-way bank conflict. V is left reading from global: its
/// existing access (`kv_read_v(v_base + d2)`, adjacent lanes → adjacent `d2`)
/// is *already* coalesced, so staging it would only add LDS pressure.
///
/// **Numerically identical, not merely close, for `f16` KV** (the default): the
/// staged value is `f16(kv_read_k(g))`, and `kv_read_k` already returns
/// `f32(k_cache[idx])` from an `f16` mirror, so the round-trip is exact and the
/// per-`d` accumulation order is unchanged — greedy output is byte-identical to
/// the split kernel. Only offered for `f16` storage (the generator is only
/// called for it): `f32` KV would blow the LDS budget at `head_dim = 512`, and
/// `q8_0` isn't `f16`-representable losslessly.
const ATTENTION_SPLIT_FLASH_SHADER_TEMPLATE: &str = r#"
%KV_ENABLE%
struct AttnSplitMeta {
    n_head: u32,
    n_head_kv: u32,
    head_dim: u32,
    window_start: u32,
    n_pos: u32,
    k_num: u32,
    scale: f32,
    kv_page_base: u32,
    kv_page_tokens: u32,
    kv_ring_rows: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> aq: array<f32>;
%KV_BINDINGS%
@group(0) @binding(3) var<storage, read_write> partial_ml: array<f32>;
@group(0) @binding(4) var<storage, read_write> partial_acc: array<f32>;
@group(0) @binding(5) var<uniform> am: AttnSplitMeta;
%KV_PAGE_BINDING%

%KV_READ_FNS%
%KV_SLOT_FN%

const MAX_HEAD_DIM: u32 = 2048u;
// Positions staged per tile, and the padded K-tile row stride (head_dim + 1).
const TILE_POS: u32 = %TILE_POS%u;
const KV_STRIDE: u32 = %KV_STRIDE%u;

var<workgroup> shared_reduce: array<f32, 64>;
var<workgroup> tile_probs: array<f32, 64>;
var<workgroup> acc: array<f32, MAX_HEAD_DIM>;
// Coalesced K-tile staging buffer: TILE_POS rows × head_dim, padded to
// KV_STRIDE so the strided score read is LDS-bank-conflict-free.
var<workgroup> k_shmem: array<f16, %K_SHMEM_LEN%>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    %SUBGROUP_PARAMS%
) {
    let h = wid.x;
    let split_idx = wid.y;
    let local = lid.x;
    let group_size = am.n_head / am.n_head_kv;
    let kv_head = h / group_size;
    let head_dim = am.head_dim;
    let k_num = am.k_num;
    let q_base = h * head_dim;

    let split_len = (am.n_pos + k_num - 1u) / k_num;
    let split_start = split_idx * split_len;
    let split_end = min(split_start + split_len, am.n_pos);

    var zd: u32 = local;
    loop {
        if (zd >= head_dim) {
            break;
        }
        acc[zd] = 0.0;
        zd = zd + 64u;
    }

    var m: f32 = -1e30;
    var l: f32 = 0.0;

    if (split_start < split_end) {
        var tile_start: u32 = split_start;
        loop {
            if (tile_start >= split_end) {
                break;
            }
            let tile_len = min(TILE_POS, split_end - tile_start);

            // Cooperatively stage this tile's K rows into LDS. All 64 lanes
            // stream tile_len*head_dim contiguous f16 values — consecutive
            // lanes read consecutive global addresses (a coalesced burst),
            // replacing the un-staged kernel's per-position, stride-head_dim
            // (uncoalesced) K fetches.
            var e: u32 = local;
            loop {
                if (e >= tile_len * head_dim) {
                    break;
                }
                let row = e / head_dim;
                let col = e - row * head_dim;
                let pp = am.window_start + tile_start + row;
                let g = (kv_slot(pp) * am.n_head_kv + kv_head) * head_dim + col;
                k_shmem[row * KV_STRIDE + col] = f16(kv_read_k(g));
                e = e + 64u;
            }
            workgroupBarrier();

            let has_pos = local < tile_len;
            var my_score: f32 = -1e30;
            if (has_pos) {
                var s: f32 = 0.0;
                var d: u32 = 0u;
                loop {
                    if (d >= head_dim) {
                        break;
                    }
                    s = s + aq[q_base + d] * f32(k_shmem[local * KV_STRIDE + d]);
                    d = d + 1u;
                }
                my_score = s * am.scale;
            }
            %MAX_REDUCE_BLOCK%

            var my_prob: f32 = 0.0;
            if (has_pos) {
                my_prob = exp(my_score - tile_max);
            }
            tile_probs[local] = my_prob;
            %SUM_REDUCE_BLOCK%

            let new_m = max(m, tile_max);
            let alpha_old = exp(m - new_m);
            let alpha_tile = exp(tile_max - new_m);
            l = l * alpha_old + tile_sum * alpha_tile;

            var d2: u32 = local;
            loop {
                if (d2 >= head_dim) {
                    break;
                }
                var tile_contribution: f32 = 0.0;
                var j: u32 = 0u;
                loop {
                    if (j >= tile_len) {
                        break;
                    }
                    let vp = am.window_start + tile_start + j;
                    let v_base = (kv_slot(vp) * am.n_head_kv + kv_head) * head_dim;
                    tile_contribution = tile_contribution + tile_probs[j] * kv_read_v(v_base + d2);
                    j = j + 1u;
                }
                acc[d2] = acc[d2] * alpha_old + alpha_tile * tile_contribution;
                d2 = d2 + 64u;
            }

            m = new_m;
            workgroupBarrier();
            tile_start = tile_start + TILE_POS;
        }
    }

    let out_base = h * k_num + split_idx;
    if (local == 0u) {
        partial_ml[out_base * 2u] = m;
        partial_ml[out_base * 2u + 1u] = l;
    }
    var d3: u32 = local;
    loop {
        if (d3 >= head_dim) {
            break;
        }
        partial_acc[out_base * head_dim + d3] = acc[d3];
        d3 = d3 + 64u;
    }
}
"#;

/// Builds [`ATTENTION_SPLIT_FLASH_SHADER_TEMPLATE`] for a specific `head_dim`
/// (which fixes the staged-tile size), the adapter's subgroup capability, and
/// the KV storage kind. Only meaningful for [`KvStorage::F16`] — see the
/// template's doc comment.
pub fn shader_source_attention_split_flash(
    kv_storage: KvStorage,
    subgroup: bool,
    head_dim: u32,
    lds_limit: u32,
    paging: KvPaging,
) -> String {
    let (max_block, sum_block) = if subgroup {
        attention_subgroup_blocks()
    } else {
        attention_classic_blocks()
    };
    let subgroup_params = if subgroup {
        "@builtin(subgroup_invocation_id) subgroup_invocation_id: u32,\n    @builtin(subgroup_id) subgroup_id: u32,\n    @builtin(num_subgroups) num_subgroups: u32,"
    } else {
        ""
    };
    let (kv_bindings, kv_read_fns) = kv_storage.bindings_and_read_fns();
    let tile_pos = flash_tile_positions(head_dim, lds_limit);
    let stride = head_dim + 1;
    let shmem_len = tile_pos * stride;
    let src = ATTENTION_SPLIT_FLASH_SHADER_TEMPLATE
        .replace(
            "const MAX_HEAD_DIM: u32 = 2048u;",
            &format!("const MAX_HEAD_DIM: u32 = {head_dim}u;"),
        )
        .replace("%KV_ENABLE%", kv_storage.enable_directive())
        .replace("%KV_BINDINGS%", &kv_bindings)
        .replace("%KV_READ_FNS%", &kv_read_fns)
        .replace("%SUBGROUP_PARAMS%", subgroup_params)
        .replace("%MAX_REDUCE_BLOCK%", max_block)
        .replace("%SUM_REDUCE_BLOCK%", sum_block)
        .replace("%TILE_POS%", &tile_pos.to_string())
        .replace("%KV_STRIDE%", &stride.to_string())
        .replace("%K_SHMEM_LEN%", &shmem_len.to_string())
        .replace("%KV_PAGE_BINDING%", paging.binding())
        .replace("%KV_SLOT_FN%", paging.slot_fn());
    paging.finish(src)
}

/// Split-k phase 2 of two — merges [`ATTENTION_SPLIT_SHADER_TEMPLATE`]'s
/// `k_num` partial `(m, l, acc)` triples for one head into the same final
/// `aout[h * head_dim .. (h+1) * head_dim]` the un-split kernel writes
/// directly, via the identical rescale-and-merge rule the un-split
/// kernel's own tile loop already uses between tiles (`m = max(...)`,
/// `alpha = exp(prev_m - new_m)`, rescale-and-add) — just applied across
/// `k_num` splits instead of `n_pos / 64` tiles.
///
/// One workgroup per head (`wid.x = h`, matching the un-split kernel's
/// own dispatch shape and `k_num=1`'s trivial case), but no
/// `workgroupBarrier` anywhere: every thread redundantly recomputes the
/// same tiny `m`/`l` merge from `partial_ml` (`k_num` is small — `ATTN_
/// SPLIT_K` in `vulkan.rs` — so this redundancy costs nothing measurable,
/// the same "every lane redundantly runs the tiny combine" trade-off
/// `PERHEAD_RMSNORM_SHADER_SUBGROUP` already makes), and each thread then
/// only *writes* the disjoint `head_dim / 64` slice of `aout` its own
/// `local` index owns — no cross-thread communication needed at all once
/// every thread has its own copy of the merged `m`/`l`.
///
/// Binding shape (2 read-only storage, 1 read-write storage, 1 uniform)
/// matches `elem4_bind_group_layout`'s (`add`/`mul`/`rmsnorm`/`vulkan_
/// shaders::FUSED_NORM_ROPE_SHADER`'s own shape), so this reuses
/// `VulkanBackend::elem4_bind_group_layout`/`elem4_pipeline_layout`
/// rather than needing a bind-group layout of its own.
const ATTENTION_SPLIT_REDUCE_SHADER: &str = r#"
struct AttnReduceMeta {
    head_dim: u32,
    k_num: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> rml: array<f32>;
@group(0) @binding(1) var<storage, read> racc: array<f32>;
@group(0) @binding(2) var<storage, read_write> raout: array<f32>;
@group(0) @binding(3) var<uniform> rm: AttnReduceMeta;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let h = wid.x;
    let local = lid.x;
    let head_dim = rm.head_dim;
    let k_num = rm.k_num;

    var m: f32 = -1e30;
    var s: u32 = 0u;
    loop {
        if (s >= k_num) {
            break;
        }
        m = max(m, rml[(h * k_num + s) * 2u]);
        s = s + 1u;
    }

    var l: f32 = 0.0;
    s = 0u;
    loop {
        if (s >= k_num) {
            break;
        }
        let base = h * k_num + s;
        l = l + rml[base * 2u + 1u] * exp(rml[base * 2u] - m);
        s = s + 1u;
    }

    var d: u32 = local;
    loop {
        if (d >= head_dim) {
            break;
        }
        var acc_val: f32 = 0.0;
        var s2: u32 = 0u;
        loop {
            if (s2 >= k_num) {
                break;
            }
            let base = h * k_num + s2;
            acc_val = acc_val + racc[base * head_dim + d] * exp(rml[base * 2u] - m);
            s2 = s2 + 1u;
        }
        raout[h * head_dim + d] = acc_val / l;
        d = d + 64u;
    }
}
"#;

/// **GQA-grouped** split-k phase 1. Same online-softmax math and same
/// `partial_ml`/`partial_acc` output layout (per *query* head) as
/// [`ATTENTION_SPLIT_SHADER_TEMPLATE`] — so [`ATTENTION_SPLIT_REDUCE_SHADER`]
/// and the bind groups are untouched — but each workgroup covers one **KV
/// head** and its whole group of `group = n_head / n_head_kv` query heads at
/// once (dispatched `(n_head_kv, k_num, 1)` instead of `(n_head, k_num, 1)`).
/// The point: the single shared KV head's K and V are read from global **once
/// per position** and reused across all `group` query heads, instead of the
/// un-grouped kernel re-reading them once per query-head workgroup (`group`×,
/// for MQA `group = n_head`). K is shared in the score loop (one `kv_read_k`
/// per element, dotted against every head's Q); V is shared in the accumulation
/// loop (one `kv_read_v` per `(d, position)`, weighted into every head's `acc`).
/// Numerically identical to the un-grouped kernel — it only changes which
/// workgroup reads what, not the arithmetic — so greedy output is byte-identical.
/// **Trade-off (measure it):** for MQA this drops the workgroup count from
/// `n_head·k_num` to `n_head_kv·k_num`, and keeps `group` heads' `(m, l, acc)`
/// state live — both cut occupancy, the very thing split-k exists to raise.
/// Opt-in (`ORANGU_ATTN_GQA=1`). `group` and `head_dim` are baked per pipeline.
/// Classic (tree) reductions only — the subgroup path isn't generated for it.
pub fn shader_source_attention_split_gqa(
    kv_storage: KvStorage,
    head_dim: u32,
    group: u32,
    paging: KvPaging,
) -> String {
    let (kv_bindings, kv_read_fns) = kv_storage.bindings_and_read_fns();
    let enable = kv_storage.enable_directive();
    let kv_page_binding = paging.binding();
    let kv_slot_fn = paging.slot_fn();
    let acc_len = group * head_dim;
    let probs_len = group * 64;
    let src = format!(
        r#"{enable}
struct AttnSplitMeta {{
    n_head: u32,
    n_head_kv: u32,
    head_dim: u32,
    window_start: u32,
    n_pos: u32,
    k_num: u32,
    scale: f32,
    kv_page_base: u32,
    kv_page_tokens: u32,
    kv_ring_rows: u32,
    _pad1: u32,
    _pad2: u32,
}}

@group(0) @binding(0) var<storage, read> aq: array<f32>;
{kv_bindings}
@group(0) @binding(3) var<storage, read_write> partial_ml: array<f32>;
@group(0) @binding(4) var<storage, read_write> partial_acc: array<f32>;
@group(0) @binding(5) var<uniform> am: AttnSplitMeta;
{kv_page_binding}

{kv_read_fns}
{kv_slot_fn}

const GROUP: u32 = {group}u;
const HEAD_DIM: u32 = {head_dim}u;

var<workgroup> shared_reduce: array<f32, 64>;
var<workgroup> probs_g: array<f32, {probs_len}>;
var<workgroup> acc: array<f32, {acc_len}>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    let kv_head = wid.x;
    let split_idx = wid.y;
    let local = lid.x;
    let head_dim = am.head_dim;
    let k_num = am.k_num;

    let split_len = (am.n_pos + k_num - 1u) / k_num;
    let split_start = split_idx * split_len;
    let split_end = min(split_start + split_len, am.n_pos);

    var z: u32 = local;
    loop {{
        if (z >= GROUP * head_dim) {{ break; }}
        acc[z] = 0.0;
        z = z + 64u;
    }}

    var m: array<f32, GROUP>;
    var l: array<f32, GROUP>;
    for (var hq: u32 = 0u; hq < GROUP; hq = hq + 1u) {{
        m[hq] = -1e30;
        l[hq] = 0.0;
    }}

    if (split_start < split_end) {{
        var tile_start: u32 = split_start;
        loop {{
            if (tile_start >= split_end) {{ break; }}
            let tile_len = min(64u, split_end - tile_start);
            let has_pos = local < tile_len;
            let p = am.window_start + tile_start + local;

            // Shared K: read each K element once, dot into every head's score.
            var scores: array<f32, GROUP>;
            for (var hq: u32 = 0u; hq < GROUP; hq = hq + 1u) {{ scores[hq] = -1e30; }}
            if (has_pos) {{
                for (var hq: u32 = 0u; hq < GROUP; hq = hq + 1u) {{ scores[hq] = 0.0; }}
                let k_base = (kv_slot(p) * am.n_head_kv + kv_head) * head_dim;
                var d: u32 = 0u;
                loop {{
                    if (d >= head_dim) {{ break; }}
                    let kval = kv_read_k(k_base + d);
                    for (var hq: u32 = 0u; hq < GROUP; hq = hq + 1u) {{
                        scores[hq] = scores[hq] + aq[(kv_head * GROUP + hq) * head_dim + d] * kval;
                    }}
                    d = d + 1u;
                }}
                for (var hq: u32 = 0u; hq < GROUP; hq = hq + 1u) {{ scores[hq] = scores[hq] * am.scale; }}
            }}

            // Per-head online-softmax update (tree reductions over the tile).
            var alpha_old: array<f32, GROUP>;
            var alpha_tile: array<f32, GROUP>;
            for (var hq: u32 = 0u; hq < GROUP; hq = hq + 1u) {{
                shared_reduce[local] = scores[hq];
                workgroupBarrier();
                var stm: u32 = 32u;
                loop {{
                    if (stm == 0u) {{ break; }}
                    if (local < stm) {{ shared_reduce[local] = max(shared_reduce[local], shared_reduce[local + stm]); }}
                    workgroupBarrier();
                    stm = stm / 2u;
                }}
                let tile_max = shared_reduce[0];
                workgroupBarrier();
                var prob: f32 = 0.0;
                if (has_pos) {{ prob = exp(scores[hq] - tile_max); }}
                probs_g[hq * 64u + local] = prob;
                shared_reduce[local] = prob;
                var st: u32 = 32u;
                workgroupBarrier();
                loop {{
                    if (st == 0u) {{ break; }}
                    if (local < st) {{ shared_reduce[local] = shared_reduce[local] + shared_reduce[local + st]; }}
                    workgroupBarrier();
                    st = st / 2u;
                }}
                let tile_sum = shared_reduce[0];
                workgroupBarrier();
                let new_m = max(m[hq], tile_max);
                alpha_old[hq] = exp(m[hq] - new_m);
                alpha_tile[hq] = exp(tile_max - new_m);
                l[hq] = l[hq] * alpha_old[hq] + tile_sum * alpha_tile[hq];
                m[hq] = new_m;
            }}

            // Shared V: read each V element once, weight into every head's acc.
            var d2: u32 = local;
            loop {{
                if (d2 >= head_dim) {{ break; }}
                var contrib: array<f32, GROUP>;
                for (var hq: u32 = 0u; hq < GROUP; hq = hq + 1u) {{ contrib[hq] = 0.0; }}
                var j: u32 = 0u;
                loop {{
                    if (j >= tile_len) {{ break; }}
                    let vp = am.window_start + tile_start + j;
                    let v_base = (kv_slot(vp) * am.n_head_kv + kv_head) * head_dim;
                    let vval = kv_read_v(v_base + d2);
                    for (var hq: u32 = 0u; hq < GROUP; hq = hq + 1u) {{
                        contrib[hq] = contrib[hq] + probs_g[hq * 64u + j] * vval;
                    }}
                    j = j + 1u;
                }}
                for (var hq: u32 = 0u; hq < GROUP; hq = hq + 1u) {{
                    acc[hq * head_dim + d2] = acc[hq * head_dim + d2] * alpha_old[hq] + alpha_tile[hq] * contrib[hq];
                }}
                d2 = d2 + 64u;
            }}

            workgroupBarrier();
            tile_start = tile_start + 64u;
        }}
    }}

    for (var hq: u32 = 0u; hq < GROUP; hq = hq + 1u) {{
        let h = kv_head * GROUP + hq;
        let out_base = h * k_num + split_idx;
        if (local == 0u) {{
            partial_ml[out_base * 2u] = m[hq];
            partial_ml[out_base * 2u + 1u] = l[hq];
        }}
        var d3: u32 = local;
        loop {{
            if (d3 >= head_dim) {{ break; }}
            partial_acc[out_base * head_dim + d3] = acc[hq * head_dim + d3];
            d3 = d3 + 64u;
        }}
    }}
}}
"#
    );
    paging.finish(src)
}

/// The split-k merge with its loads in flight — the same shape of rewrite
/// as the wide whole-row norm, for the same reason. The kernel above walks
/// `k_num` partials in a rolled loop for each of a thread's `head_dim / 64`
/// output elements, one dependent load per iteration; at a short context
/// that merge cost more than the attention it merged. Here a workgroup is
/// 256 threads — one per element of a 256-wide head, striding for wider —
/// the per-split `(m, l)` are read by one thread each and the softmax
/// weights computed once into shared memory (one barrier), and each thread's
/// walk over the partials is unrolled four wide so four loads leave together.
/// Same arithmetic, same bindings, same dispatch (`n_head` workgroups).
const ATTENTION_SPLIT_REDUCE_WIDE_SHADER: &str = r#"
struct AttnReduceMeta {
    head_dim: u32,
    k_num: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> rml: array<f32>;
@group(0) @binding(1) var<storage, read> racc: array<f32>;
@group(0) @binding(2) var<storage, read_write> raout: array<f32>;
@group(0) @binding(3) var<uniform> rm: AttnReduceMeta;

// A split's running max and sum; the cap on `k_num` is the scratch the
// backend sizes for.
var<workgroup> split_m: array<f32, 64>;
var<workgroup> split_l: array<f32, 64>;
var<workgroup> split_w: array<f32, 64>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let h = wid.x;
    let local = lid.x;
    let head_dim = rm.head_dim;
    let k_num = min(rm.k_num, 64u);

    if (local < k_num) {
        let base = h * k_num + local;
        split_m[local] = rml[base * 2u];
        split_l[local] = rml[base * 2u + 1u];
    }
    workgroupBarrier();
    var m: f32 = -1e30;
    var s: u32 = 0u;
    loop {
        if (s >= k_num) {
            break;
        }
        m = max(m, split_m[s]);
        s = s + 1u;
    }
    var l: f32 = 0.0;
    s = 0u;
    loop {
        if (s >= k_num) {
            break;
        }
        let w = exp(split_m[s] - m);
        l = l + split_l[s] * w;
        if (local == 0u) {
            split_w[s] = w;
        }
        s = s + 1u;
    }
    workgroupBarrier();

    var d: u32 = local;
    loop {
        if (d >= head_dim) {
            break;
        }
        var acc_val: f32 = 0.0;
        let row = h * k_num;
        var s2: u32 = 0u;
        loop {
            if (s2 + 4u > k_num) {
                break;
            }
            let a0 = racc[(row + s2) * head_dim + d];
            let a1 = racc[(row + s2 + 1u) * head_dim + d];
            let a2 = racc[(row + s2 + 2u) * head_dim + d];
            let a3 = racc[(row + s2 + 3u) * head_dim + d];
            acc_val = acc_val + a0 * split_w[s2] + a1 * split_w[s2 + 1u]
                + a2 * split_w[s2 + 2u] + a3 * split_w[s2 + 3u];
            s2 = s2 + 4u;
        }
        loop {
            if (s2 >= k_num) {
                break;
            }
            acc_val = acc_val + racc[(row + s2) * head_dim + d] * split_w[s2];
            s2 = s2 + 1u;
        }
        raout[h * head_dim + d] = acc_val / l;
        d = d + 256u;
    }
}
"#;

pub fn shader_source_attention_split_reduce() -> String {
    if crate::engine::env::flag_on("ORANGU_ATTN_REDUCE_NARROW") {
        return ATTENTION_SPLIT_REDUCE_SHADER.to_string();
    }
    ATTENTION_SPLIT_REDUCE_WIDE_SHADER.to_string()
}

/// Casts `cm.len` elements of a freshly RoPE'd/normed `f32` key or value
/// row (`csrc`) into the `f16`-stored KV mirror (`cdst`) at element offset
/// `cm.offset` — only ever built
/// when the adapter supports native WGSL `f16` (`VulkanBackend::kv_f16`).
/// Shares `elem3_bind_group_layout`'s three-binding shape (read-only
/// source, read-write destination, uniform meta) with `rope_pipeline`/
/// `perhead_rmsnorm_pipeline`, so it needs no bind-group layout of its
/// own — only `CastMeta`'s second field differs in *meaning* from
/// `ElemMeta`'s (an element offset into `cdst`, not `eps`/a scale
/// multiplier), not in byte layout, so the same `elem3_bind_group` helper
/// and buffer-building code build this shader's bind group too.
const KV_CAST_SHADER: &str = r#"
enable f16;
struct CastMeta {
    len: u32,
    offset: u32,
    extra: f32,
    // Where in `csrc` this dispatch starts. Zero for every writer that copies
    // a whole run at once; a paged write splits its range at page boundaries
    // and issues one dispatch per run, and each run reads from further into
    // the same source. Every existing caller leaves the word zeroed, which is
    // the offset that means "from the beginning".
    src_offset: u32,
}

@group(0) @binding(0) var<storage, read> csrc: array<f32>;
@group(0) @binding(1) var<storage, read_write> cdst: array<f16>;
@group(0) @binding(2) var<uniform> cm: CastMeta;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= cm.len) {
        return;
    }
    cdst[cm.offset + idx] = f16(csrc[cm.src_offset + idx]);
}
"#;

pub fn shader_source_kv_cast() -> String {
    KV_CAST_SHADER.to_string()
}

/// Quantizes `cm.n_blocks` 32-element blocks of a freshly RoPE'd/normed
/// `f32` key or value row (`csrc`) into the [`KvStorage::Q8_0`] mirror
/// (`cdst`, `array<u32>`) at block offset `cm.dst_block_offset` — only
/// ever built when `VulkanBackend::kv_storage` is `Q8_0`. One thread per
/// block (`csrc` is `kv_dim`-long, i.e. `kv_dim / 32` blocks — a handful
/// to a few dozen for any real model, so one block per thread is plenty
/// parallel without needing a workgroup-level reduction the way a much
/// wider quantize would). Each thread finds its own block's `amax`
/// sequentially (32 elements), derives the scale exactly the way
/// `engine::quant`'s own CPU-side quantizers do (`amax / 127`, `id = 1/d`
/// guarded against `d == 0`), then writes the word-aligned 9-word block
/// [`KvStorage::Q8_0`]'s own doc comment describes — see `kv_dequant_q8_0`
/// (`bindings_and_read_fns`) for the matching read side.
const KV_QUANTIZE_Q8_0_SHADER: &str = r#"
struct QuantMeta {
    n_blocks: u32,
    dst_block_offset: u32,
    _pad0: u32,
    /// Blocks into `csrc` this dispatch starts at — see `CastMeta::src_offset`.
    src_block_offset: u32,
}

@group(0) @binding(0) var<storage, read> csrc: array<f32>;
@group(0) @binding(1) var<storage, read_write> cdst: array<u32>;
@group(0) @binding(2) var<uniform> cm: QuantMeta;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let block = gid.x;
    if (block >= cm.n_blocks) {
        return;
    }
    let src_base = (cm.src_block_offset + block) * 32u;
    var amax: f32 = 0.0;
    var i: u32 = 0u;
    loop {
        if (i >= 32u) {
            break;
        }
        amax = max(amax, abs(csrc[src_base + i]));
        i = i + 1u;
    }
    let d = amax / 127.0;
    let inv_d = select(0.0, 1.0 / d, d > 0.0);
    let word_base = (cm.dst_block_offset + block) * 9u;
    cdst[word_base] = bitcast<u32>(d);
    var w: u32 = 0u;
    loop {
        if (w >= 8u) {
            break;
        }
        var packed: u32 = 0u;
        var j: u32 = 0u;
        loop {
            if (j >= 4u) {
                break;
            }
            let v = csrc[src_base + w * 4u + j];
            var q: i32 = i32(round(v * inv_d));
            q = clamp(q, -127, 127);
            packed = packed | ((u32(q) & 0xFFu) << (j * 8u));
            j = j + 1u;
        }
        cdst[word_base + 1u + w] = packed;
        w = w + 1u;
    }
}
"#;

pub fn shader_source_kv_quantize_q8_0() -> String {
    KV_QUANTIZE_Q8_0_SHADER.to_string()
}

/// Line-for-line port of `engine::tensor::rope_apply_params_inplace`, both
/// pairings (`RopeMeta::layout`: NEOX pairs element `i` with `i + rope_dim/2`,
/// NORM pairs `2i` with `2i+1` — `llama` and `mistral` are NORM, everything
/// else here is NEOX). Only the
/// leading `rope_dim` elements of each head rotate, any remainder passes
/// through untouched since this shader never touches it) — modifies `rx`
/// in place. `rff` (the proportional-RoPE per-frequency divisor,
/// Gemma4's `rope_freqs`) is *always* bound, even for layers that don't
/// use it: the caller fills it with `1.0`s in that case (a no-op divisor)
/// rather than making this shader branch on whether the tensor exists —
/// one fewer pipeline variant, and `x / 1.0 == x` exactly in IEEE 754, so
/// it's bit-for-bit identical to skipping the divide. Binding order
/// (read-only storage, read-write storage, uniform) deliberately matches
/// `elem3_bind_group_layout`'s shape so this reuses the same layout/
/// pipeline layout as `gelu`/`scale` rather than needing a new one.
/// ggml's `rope_yarn` angle, shared verbatim by [`ROPE_SHADER`] and
/// [`FUSED_NORM_ROPE_SHADER`] — prepended to both sources rather than
/// duplicated, because the two kernels rotating by *different* angles is
/// exactly the failure `RopeMeta::pairing`'s own comment records: a wrong
/// answer that matches to five significant figures before RoPE and not at all
/// after.
///
/// `corr_lo`/`corr_hi` (ggml's `ggml_rope_yarn_corr_dims`) and `mscale` (its
/// `rope_yarn` magnitude correction) are computed host-side in
/// [`super::vulkan::RopeYarn::from_params`]: they depend only on constants, so
/// deriving them per thread would burn a `log` to reach an answer that must
/// agree bit-for-bit with the CPU reference anyway.
///
/// With `ext_factor == 0` this returns `freq_scale * theta_extrap`, and every
/// non-YaRN caller passes `freq_scale = 1.0`, so the result is
/// `1.0 * theta_extrap` — exact in IEEE 754, hence bit-identical to the plain
/// rope these shaders computed before.
const ROPE_YARN_WGSL: &str = r#"
fn rope_yarn_theta(
    i: f32,
    theta_extrap: f32,
    freq_scale: f32,
    ext_factor: f32,
    corr_lo: f32,
    corr_hi: f32,
) -> f32 {
    let theta_interp = freq_scale * theta_extrap;
    if (ext_factor == 0.0) {
        return theta_interp;
    }
    // `rope_yarn_ramp`: 1 at the low end of the band (pure interpolation)
    // falling to 0 above it.
    let y = (i - corr_lo) / max(0.001, corr_hi - corr_lo);
    let ramp = 1.0 - clamp(y, 0.0, 1.0);
    let mix = ramp * ext_factor;
    return theta_interp * (1.0 - mix) + theta_extrap * mix;
}
"#;

const ROPE_SHADER: &str = r#"
struct RopeMeta {
    n_head: u32,
    head_dim: u32,
    rope_dim: u32,
    pos: u32,
    freq_base: f32,
    n_tokens: u32,
    /// `0` = NEOX (element `i` pairs with `i + rope_dim/2`), `1` = NORM
    /// (element `2i` pairs with `2i+1`). A uniform rather than a second
    /// pipeline: the two differ only in which two indices a thread touches,
    /// the branch is uniform across the whole dispatch, and a wrong answer
    /// here is the kind that matches upstream to five significant figures
    /// *before* RoPE and reads −354.6 against 6.66 after.
    ///
    /// Not named `layout`: that is a reserved word in WGSL, and naga rejects
    /// the module with `name \`layout\` is a reserved keyword`.
    pairing: u32,
    _pad2: u32,
    /// YaRN, all identity-valued for a model that does not use it — see
    /// `ROPE_YARN_WGSL`.
    freq_scale: f32,
    ext_factor: f32,
    /// ggml's `mscale`, already folded with `attn_factor` host-side.
    mscale: f32,
    corr_lo: f32,
    corr_hi: f32,
    _pad3: u32,
    _pad4: u32,
    _pad5: u32,
}

@group(0) @binding(0) var<storage, read> rff: array<f32>;
@group(0) @binding(1) var<storage, read_write> rx: array<f32>;
@group(0) @binding(2) var<uniform> rm: RopeMeta;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half = rm.rope_dim / 2u;
    // Rows are `[n_tokens, n_head, head_dim]`; `hg` is the flat row index over
    // the whole batch, so `hg / n_head` is the token and `pos + t` its
    // position. A decode step passes `n_tokens = 1`, where `t` is 0 and this
    // reduces to the single-position form.
    let total = rm.n_tokens * rm.n_head * half;
    let idx = gid.x;
    if (idx >= total) {
        return;
    }
    let hg = idx / half;
    let i = idx % half;
    let t = hg / rm.n_head;
    let base = hg * rm.head_dim;
    let freq = pow(rm.freq_base, -2.0 * f32(i) / f32(rm.rope_dim)) / rff[i];
    let theta_extrap = f32(rm.pos + t) * freq;
    let theta = rope_yarn_theta(
        f32(i), theta_extrap, rm.freq_scale, rm.ext_factor, rm.corr_lo, rm.corr_hi);
    let s = sin(theta) * rm.mscale;
    let c = cos(theta) * rm.mscale;
    // NEOX rotates `i` against `i + half`; NORM rotates the consecutive pair
    // `2i`/`2i+1`. Line-for-line the same choice `engine::tensor`'s
    // `RopeLayout` makes on the CPU.
    var lo = i;
    var hi = i + half;
    if (rm.pairing == 1u) {
        lo = 2u * i;
        hi = 2u * i + 1u;
    }
    let a = rx[base + lo];
    let b = rx[base + hi];
    rx[base + lo] = a * c - b * s;
    rx[base + hi] = a * s + b * c;
}
"#;

pub fn shader_source_rope() -> String {
    format!("{ROPE_YARN_WGSL}{ROPE_SHADER}")
}

/// Per-head weighted RMSNorm — Q-norm/K-norm applied independently to
/// each of `n_head`'s `head_dim`-length slices of `px`, one workgroup per
/// head (same reduction shape as `RMSNORM_SHADER_BODY`, just dispatched
/// `n_head` times instead of once — `RMSNORM_SHADER_BODY` only ever
/// handles a single row, which is all `fused_post_attention` needs, but
/// Q/K-norm need one independent normalization per head in a single
/// dispatch). `pw`, the learned scale, is the *same* `head_dim`-length
/// vector for every head — matches `tensor::rmsnorm_inplace(&mut q,
/// &layer.attn_q_norm, n_tokens * n_head, head_dim, eps)`'s treatment of
/// `q` as `n_tokens * n_head` independent rows all sharing one weight.
/// Binding order matches `elem3_bind_group_layout` for the same reuse
/// reason as [`ROPE_SHADER`].
const PERHEAD_RMSNORM_SHADER: &str = r#"
struct PerHeadNormMeta {
    n_head: u32,
    head_dim: u32,
    eps: f32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> pw: array<f32>;
@group(0) @binding(1) var<storage, read_write> px: array<f32>;
@group(0) @binding(2) var<uniform> pm: PerHeadNormMeta;

var<workgroup> ph_partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let h = wid.x;
    let local = lid.x;
    let base = h * pm.head_dim;

    var partial: f32 = 0.0;
    var k: u32 = local;
    loop {
        if (k >= pm.head_dim) {
            break;
        }
        let v = px[base + k];
        partial = partial + v * v;
        k = k + 64u;
    }
    ph_partial[local] = partial;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (local < stride) {
            ph_partial[local] = ph_partial[local] + ph_partial[local + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let mean_sq = ph_partial[0] / f32(pm.head_dim);
    let scale = 1.0 / sqrt(mean_sq + pm.eps);
    workgroupBarrier();
    k = local;
    loop {
        if (k >= pm.head_dim) {
            break;
        }
        px[base + k] = px[base + k] * scale * pw[k];
        k = k + 64u;
    }
}
"#;

/// Subgroup-reduce variant of `PERHEAD_RMSNORM_SHADER` — see
/// `RMSNORM_SHADER_BODY_SUBGROUP`'s doc comment for the "every lane
/// redundantly runs the tiny combine loop, one barrier total" pattern this
/// reuses.
const PERHEAD_RMSNORM_SHADER_SUBGROUP: &str = r#"
struct PerHeadNormMeta {
    n_head: u32,
    head_dim: u32,
    eps: f32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> pw: array<f32>;
@group(0) @binding(1) var<storage, read_write> px: array<f32>;
@group(0) @binding(2) var<uniform> pm: PerHeadNormMeta;

var<workgroup> ph_partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_lane: u32,
    @builtin(subgroup_id) sg_id: u32,
    @builtin(num_subgroups) n_sg: u32,
) {
    let h = wid.x;
    let local = lid.x;
    let base = h * pm.head_dim;

    var partial: f32 = 0.0;
    var k: u32 = local;
    loop {
        if (k >= pm.head_dim) {
            break;
        }
        let v = px[base + k];
        partial = partial + v * v;
        k = k + 64u;
    }
    let sg_sum = subgroupAdd(partial);
    if (sg_lane == 0u) {
        ph_partial[sg_id] = sg_sum;
    }
    workgroupBarrier();
    var total: f32 = 0.0;
    var i: u32 = 0u;
    loop {
        if (i >= n_sg) {
            break;
        }
        total = total + ph_partial[i];
        i = i + 1u;
    }
    let mean_sq = total / f32(pm.head_dim);
    let scale = 1.0 / sqrt(mean_sq + pm.eps);
    k = local;
    loop {
        if (k >= pm.head_dim) {
            break;
        }
        px[base + k] = px[base + k] * scale * pw[k];
        k = k + 64u;
    }
}
"#;

pub fn shader_source_perhead_rmsnorm(subgroup: bool) -> String {
    if subgroup {
        PERHEAD_RMSNORM_SHADER_SUBGROUP.to_string()
    } else {
        PERHEAD_RMSNORM_SHADER.to_string()
    }
}

/// Like [`PERHEAD_RMSNORM_SHADER`], but weightless (`ggml_rms_norm`, no
/// learned scale) — V's norm. One fewer binding (no weight vector), so
/// this needs its own 2-binding (read-write storage, uniform) layout —
/// see `elem2_bind_group_layout`.
const PERHEAD_RMSNORM_WEIGHTLESS_SHADER: &str = r#"
struct PerHeadNormMeta {
    n_head: u32,
    head_dim: u32,
    eps: f32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> px: array<f32>;
@group(0) @binding(1) var<uniform> pm: PerHeadNormMeta;

var<workgroup> ph_partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let h = wid.x;
    let local = lid.x;
    let base = h * pm.head_dim;

    var partial: f32 = 0.0;
    var k: u32 = local;
    loop {
        if (k >= pm.head_dim) {
            break;
        }
        let v = px[base + k];
        partial = partial + v * v;
        k = k + 64u;
    }
    ph_partial[local] = partial;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (local < stride) {
            ph_partial[local] = ph_partial[local] + ph_partial[local + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let mean_sq = ph_partial[0] / f32(pm.head_dim);
    let scale = 1.0 / sqrt(mean_sq + pm.eps);
    workgroupBarrier();
    k = local;
    loop {
        if (k >= pm.head_dim) {
            break;
        }
        px[base + k] = px[base + k] * scale;
        k = k + 64u;
    }
}
"#;

/// Subgroup-reduce variant of `PERHEAD_RMSNORM_WEIGHTLESS_SHADER` — see
/// `PERHEAD_RMSNORM_SHADER_SUBGROUP`'s doc comment.
const PERHEAD_RMSNORM_WEIGHTLESS_SHADER_SUBGROUP: &str = r#"
struct PerHeadNormMeta {
    n_head: u32,
    head_dim: u32,
    eps: f32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> px: array<f32>;
@group(0) @binding(1) var<uniform> pm: PerHeadNormMeta;

var<workgroup> ph_partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_lane: u32,
    @builtin(subgroup_id) sg_id: u32,
    @builtin(num_subgroups) n_sg: u32,
) {
    let h = wid.x;
    let local = lid.x;
    let base = h * pm.head_dim;

    var partial: f32 = 0.0;
    var k: u32 = local;
    loop {
        if (k >= pm.head_dim) {
            break;
        }
        let v = px[base + k];
        partial = partial + v * v;
        k = k + 64u;
    }
    let sg_sum = subgroupAdd(partial);
    if (sg_lane == 0u) {
        ph_partial[sg_id] = sg_sum;
    }
    workgroupBarrier();
    var total: f32 = 0.0;
    var i: u32 = 0u;
    loop {
        if (i >= n_sg) {
            break;
        }
        total = total + ph_partial[i];
        i = i + 1u;
    }
    let mean_sq = total / f32(pm.head_dim);
    let scale = 1.0 / sqrt(mean_sq + pm.eps);
    k = local;
    loop {
        if (k >= pm.head_dim) {
            break;
        }
        px[base + k] = px[base + k] * scale;
        k = k + 64u;
    }
}
"#;

pub fn shader_source_perhead_rmsnorm_weightless(subgroup: bool) -> String {
    if subgroup {
        PERHEAD_RMSNORM_WEIGHTLESS_SHADER_SUBGROUP.to_string()
    } else {
        PERHEAD_RMSNORM_WEIGHTLESS_SHADER.to_string()
    }
}

/// Fuses [`PERHEAD_RMSNORM_SHADER`] immediately followed by [`ROPE_SHADER`]
/// into one dispatch — Q-norm+Q-RoPE and (when this layer owns its own V
/// projection — see `VulkanBackend::build_fused_attn_layer_resources`'s own
/// comment for the one case this can't safely replace) K-norm+K-RoPE.
/// Concatenates the same two already-verified algorithms in the same
/// order — the reduce-then-scale-then-weight loop is a line-for-line copy
/// of `PERHEAD_RMSNORM_SHADER`'s, the rotation loop a line-for-line copy of
/// `ROPE_SHADER`'s (`half`/`freq`/`theta`/`sin`/`cos` all computed
/// identically) — so this produces bit-identical output to running the two
/// original shaders back to back, not just numerically-close output; no
/// operation is reordered or re-associated relative to either source.
///
/// The one real change: the normalized-but-not-yet-rotated head lives in
/// `fn_head` (`workgroup`-shared, not global) between the two stages, so
/// RoPE reads values a *different* thread just wrote without a trip through
/// global memory (`px`'s round-trip through VRAM between the old two
/// dispatches). `fn_head`'s `1024`-element bound matches llama.cpp's own
/// `rms_norm.comp` fused-rope shared array (`shared FLOAT_TYPE
/// rope_data_a[1024]`) — comfortably above every `head_dim` this project
/// loads (512 is gemma4-E2B's own largest, for its full-attention layers);
/// `VulkanBackend::build_fused_attn_layer_resources` asserts this before
/// ever dispatching, since `head_dim > 1024` would silently write past the
/// array on the GPU rather than fail loudly. Same 3-`workgroupBarrier()`-
/// per-head cost as the un-fused pair combined would already pay (reduce,
/// scale-visibility, rotate-visibility) — the saving is the *dispatch*
/// (one `begin_compute_pass`/pipeline-bind/launch instead of two) and the
/// eliminated intermediate global read+write, not fewer barriers.
///
/// **Dispatched `(n_head, n_tokens, 1)`.** `y` selects the token within a
/// prefill batch: row `(t, h)` lives at `(t * n_head + h) * head_dim` and takes
/// position `pos + t`, which is the layout and the position rule
/// `GemmaModel::run_layers_cpu`'s own per-token RoPE loop uses. A decode step
/// dispatches `(n_head, 1, 1)`, where `t` is 0 and both expressions collapse to
/// the single-token form this shader started as — so the decode path's output
/// is unchanged, not merely equivalent.
///
/// The token index cannot be folded into `x` the way the per-head norms fold
/// theirs (those are position-independent, so a prefill just dispatches
/// `n_tokens * n_head` workgroups against the same shader): RoPE's angle
/// depends on the row's position, so the shader has to be able to tell which
/// token it is looking at.
///
/// Binding order (`fnw` the learned norm weight, `fnff` RoPE's per-
/// frequency divisor, `fnx` the buffer normalized and rotated in place,
/// `fnm` the uniform meta) matches [`elem4_bind_group_layout`]'s shape —
/// the same one `add`/`mul`/`rmsnorm` already share — so this needs no
/// bind-group layout or pipeline layout of its own, only its own pipeline
/// (`VulkanBackend::fused_norm_rope_pipeline`).
const FUSED_NORM_ROPE_SHADER: &str = r#"
struct FusedNormRopeMeta {
    n_head: u32,
    head_dim: u32,
    rope_dim: u32,
    pos: u32,
    freq_base: f32,
    eps: f32,
    /// `0` = NEOX pairing, `1` = NORM. Same convention and same reason as
    /// `RopeMeta::pairing`: the QKV fusion serves `llama` and `mistral`, which
    /// are NORM, and this is the RoPE they actually reach — the standalone
    /// `ROPE_SHADER` is a different kernel and fixing that one alone would have
    /// left the fused path silently NEOX.
    pairing: u32,
    _pad1: u32,
    /// YaRN, all identity-valued for a model that does not use it — see
    /// `ROPE_YARN_WGSL`.
    freq_scale: f32,
    ext_factor: f32,
    /// ggml's `mscale`, already folded with `attn_factor` host-side.
    mscale: f32,
    corr_lo: f32,
    corr_hi: f32,
    _pad2: u32,
    _pad3: u32,
    _pad4: u32,
}

@group(0) @binding(0) var<storage, read> fnw: array<f32>;
@group(0) @binding(1) var<storage, read> fnff: array<f32>;
@group(0) @binding(2) var<storage, read_write> fnx: array<f32>;
@group(0) @binding(3) var<uniform> fnm: FusedNormRopeMeta;

var<workgroup> fn_head: array<f32, 1024>;
var<workgroup> fn_partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let h = wid.x;
    // `y` is the token within a prefill batch; a decode step dispatches
    // `(n_head, 1, 1)`, so `t` is 0 and both expressions below collapse to
    // exactly what the single-token version computed. The rows of a batch are
    // contiguous and each carries its own position, which is the only reason
    // this cannot be folded into `x` the way the per-head norms are.
    let t = wid.y;
    let local = lid.x;
    let base = (t * fnm.n_head + h) * fnm.head_dim;
    let pos = fnm.pos + t;

    // Stage 1 (= PERHEAD_RMSNORM_SHADER's own first stage): sum of
    // squares, staging each raw value into `fn_head` on the way so stage 2
    // doesn't need to re-read `fnx`.
    var partial: f32 = 0.0;
    var k: u32 = local;
    loop {
        if (k >= fnm.head_dim) {
            break;
        }
        let v = fnx[base + k];
        fn_head[k] = v;
        partial = partial + v * v;
        k = k + 64u;
    }
    fn_partial[local] = partial;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (local < stride) {
            fn_partial[local] = fn_partial[local] + fn_partial[local + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let mean_sq = fn_partial[0] / f32(fnm.head_dim);
    let norm_scale = 1.0 / sqrt(mean_sq + fnm.eps);

    // Stage 2 (= PERHEAD_RMSNORM_SHADER's own second stage): scale +
    // learned weight, written into `fn_head` instead of back to `fnx`.
    k = local;
    loop {
        if (k >= fnm.head_dim) {
            break;
        }
        fn_head[k] = fn_head[k] * norm_scale * fnw[k];
        k = k + 64u;
    }
    workgroupBarrier();

    // Stage 3 (= ROPE_SHADER's own body, unchanged): rotate the now-
    // normalized pairs, reading/writing `fn_head` instead of `rx`.
    let half = fnm.rope_dim / 2u;
    k = local;
    loop {
        if (k >= half) {
            break;
        }
        let freq = pow(fnm.freq_base, -2.0 * f32(k) / f32(fnm.rope_dim)) / fnff[k];
        let theta_extrap = f32(pos) * freq;
        let theta = rope_yarn_theta(
            f32(k), theta_extrap, fnm.freq_scale, fnm.ext_factor, fnm.corr_lo, fnm.corr_hi);
        let s = sin(theta) * fnm.mscale;
        let c = cos(theta) * fnm.mscale;
        var lo = k;
        var hi = k + half;
        if (fnm.pairing == 1u) {
            lo = 2u * k;
            hi = 2u * k + 1u;
        }
        let a = fn_head[lo];
        let b = fn_head[hi];
        fn_head[lo] = a * c - b * s;
        fn_head[hi] = a * s + b * c;
        k = k + 64u;
    }
    workgroupBarrier();

    // Stage 4: the one write back to global memory — normalized+rotated
    // for `[0, rope_dim)`, normalized-only pass-through (untouched by
    // stage 3) for `[rope_dim, head_dim)`, exactly matching what running
    // the two original shaders back to back would leave in `fnx`.
    k = local;
    loop {
        if (k >= fnm.head_dim) {
            break;
        }
        fnx[base + k] = fn_head[k];
        k = k + 64u;
    }
}
"#;

/// The maximum `head_dim` [`FUSED_NORM_ROPE_SHADER`]'s `fn_head` shared
/// array supports — see that constant's own doc comment.
pub const FUSED_NORM_ROPE_MAX_HEAD_DIM: usize = 1024;

pub fn shader_source_fused_norm_rope() -> String {
    format!("{ROPE_YARN_WGSL}{FUSED_NORM_ROPE_SHADER}")
}

/// The widest head the straight-line per-head kernels below cover: 64
/// threads, each holding `HEAD_WIDE_SLOTS` `vec4`s. A head wider than this
/// takes the rolled kernels above.
pub const HEAD_WIDE_MAX_DIM: usize = 64 * HEAD_WIDE_SLOTS * 4;
const HEAD_WIDE_SLOTS: usize = 2;

/// The per-head body [`FUSED_NORM_ROPE_SHADER`] and the K/V epilogue below
/// share, as text: one head's `vec4` loads issued straight-line into named
/// registers, a two-round shared reduce, and the reduce-then-scale that
/// leaves the head — scaled and, for a weighted norm, multiplied by its
/// weight — in `hd_head`. The rolled kernels' one-scalar-per-iteration
/// loop gave every load its own wait (see `shader_source_rmsnorm_wide_kind`
/// for the same finding on the whole-row norms); at eight heads of 256 a
/// workgroup of 64 lanes is the whole machine's worth of parallelism, so
/// the latency of one head *is* the kernel's time.
///
/// Expects, in scope: `local`, `n4` (the head's width in `vec4`s), `base4`
/// (the head's first `vec4` in `src`), a `src` array of `vec4<f32>` to read
/// the head from, `weighted` (whether `hw` scales the row), `normed`
/// (whether to normalize at all — `false` copies the head through
/// unscaled), and `hm.head_dim`/`hm.eps`.
const HEAD_WIDE_NORM_WGSL: &str = r#"
    let k0 = local;
    let k1 = local + 64u;
    var v0 = vec4<f32>(0.0);
    var v1 = vec4<f32>(0.0);
    if (k0 < n4) { v0 = src[base4 + k0]; }
    if (k1 < n4) { v1 = src[base4 + k1]; }
    hd_partial[local] = dot(v0, v0) + dot(v1, v1);
    workgroupBarrier();
    if (local < 8u) {
        let b = local * 8u;
        hd_stripe[local] = hd_partial[b] + hd_partial[b + 1u] + hd_partial[b + 2u]
            + hd_partial[b + 3u] + hd_partial[b + 4u] + hd_partial[b + 5u]
            + hd_partial[b + 6u] + hd_partial[b + 7u];
    }
    workgroupBarrier();
    let total = hd_stripe[0] + hd_stripe[1] + hd_stripe[2] + hd_stripe[3]
        + hd_stripe[4] + hd_stripe[5] + hd_stripe[6] + hd_stripe[7];
    var scale: f32 = 1.0;
    if (normed) {
        scale = 1.0 / sqrt(total / f32(hm.head_dim) + hm.eps);
    }
    if (k0 < n4) {
        var s0 = v0 * scale;
        if (weighted) { s0 = s0 * hw[k0]; }
        hd_head[4u * k0] = s0.x;
        hd_head[4u * k0 + 1u] = s0.y;
        hd_head[4u * k0 + 2u] = s0.z;
        hd_head[4u * k0 + 3u] = s0.w;
    }
    if (k1 < n4) {
        var s1 = v1 * scale;
        if (weighted) { s1 = s1 * hw[k1]; }
        hd_head[4u * k1] = s1.x;
        hd_head[4u * k1 + 1u] = s1.y;
        hd_head[4u * k1 + 2u] = s1.z;
        hd_head[4u * k1 + 3u] = s1.w;
    }
    workgroupBarrier();
"#;

/// The rotation stage the wide kernels share — [`FUSED_NORM_ROPE_SHADER`]'s
/// stage 3 verbatim, over `hd_head`, so the angles and the pair rule are the
/// same arithmetic in the same order. Expects `local`, `pos`, and the RoPE
/// fields of `hm`; leaves the head rotated in `hd_head`. The barrier that
/// publishes it is the caller's, so the K/V kernel can keep it outside the
/// branch that decides whether to rotate at all.
const HEAD_WIDE_ROTATE_WGSL: &str = r#"
    let half = hm.rope_dim / 2u;
    var k: u32 = local;
    loop {
        if (k >= half) {
            break;
        }
        let freq = pow(hm.freq_base, -2.0 * f32(k) / f32(hm.rope_dim)) / hff[k];
        let theta_extrap = f32(pos) * freq;
        let theta = rope_yarn_theta(
            f32(k), theta_extrap, hm.freq_scale, hm.ext_factor, hm.corr_lo, hm.corr_hi);
        let s = sin(theta) * hm.mscale;
        let c = cos(theta) * hm.mscale;
        var lo = k;
        var hi = k + half;
        if (hm.pairing == 1u) {
            lo = 2u * k;
            hi = 2u * k + 1u;
        }
        let a = hd_head[lo];
        let b = hd_head[hi];
        hd_head[lo] = a * c - b * s;
        hd_head[hi] = a * s + b * c;
        k = k + 64u;
    }
"#;

/// [`FUSED_NORM_ROPE_SHADER`] with the head loaded straight-line — the same
/// bindings (`elem4`), the same meta, the same output, dispatched
/// `(n_head, n_tokens, 1)` the same way, for heads up to
/// [`HEAD_WIDE_MAX_DIM`] wide and a multiple of four (the caller checks).
const HEAD_NORM_ROPE_WIDE_SHADER: &str = r#"
struct FusedNormRopeMeta {
    n_head: u32,
    head_dim: u32,
    rope_dim: u32,
    pos: u32,
    freq_base: f32,
    eps: f32,
    pairing: u32,
    _pad1: u32,
    freq_scale: f32,
    ext_factor: f32,
    mscale: f32,
    corr_lo: f32,
    corr_hi: f32,
    _pad2: u32,
    _pad3: u32,
    _pad4: u32,
}

@group(0) @binding(0) var<storage, read> hw: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> hff: array<f32>;
@group(0) @binding(2) var<storage, read_write> src: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> hm: FusedNormRopeMeta;

var<workgroup> hd_head: array<f32, HEAD_MAX>;
var<workgroup> hd_partial: array<f32, 64>;
var<workgroup> hd_stripe: array<f32, 8>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let h = wid.x;
    let t = wid.y;
    let local = lid.x;
    let n4 = hm.head_dim / 4u;
    let base4 = (t * hm.n_head + h) * n4;
    let pos = hm.pos + t;
    let weighted = true;
    let normed = true;
NORM
ROTATE
    workgroupBarrier();
    if (k0 < n4) {
        src[base4 + k0] = vec4<f32>(
            hd_head[4u * k0], hd_head[4u * k0 + 1u], hd_head[4u * k0 + 2u], hd_head[4u * k0 + 3u]);
    }
    if (k1 < n4) {
        src[base4 + k1] = vec4<f32>(
            hd_head[4u * k1], hd_head[4u * k1 + 1u], hd_head[4u * k1 + 2u], hd_head[4u * k1 + 3u]);
    }
}
"#;

pub fn shader_source_head_norm_rope_wide() -> String {
    let body = HEAD_NORM_ROPE_WIDE_SHADER
        .replace("HEAD_MAX", &HEAD_WIDE_MAX_DIM.to_string())
        .replace("NORM\n", HEAD_WIDE_NORM_WGSL)
        .replace("ROTATE\n", HEAD_WIDE_ROTATE_WGSL);
    format!("{ROPE_YARN_WGSL}{body}")
}

/// Everything that happens to a decode step's key and value after their
/// projections, in one dispatch: K's per-head norm and RoPE, V's weightless
/// per-head norm, and both rows' write into the KV mirror. Four dispatches
/// on the rolled path (`attn.k_norm_rope`, `attn.v_norm`, and one KV cast
/// each), each a few microseconds of device time on top of its work — and
/// on a model whose KV layers have one head, each one a single workgroup.
///
/// Workgroup `x < n_head_kv` is K head `x`; `x >= n_head_kv` is V head
/// `x - n_head_kv`. The processed row is written back to its projection
/// buffer (what the rolled path leaves there) and, cast, to the mirror at
/// element `dst_offset + head * head_dim` — the same address the cast
/// kernel writes. `DST` is the mirror's element type: `f16` or `f32`; a
/// `q8_0` mirror keeps the rolled path, its block quantization being a
/// different kernel.
const KV_EPILOGUE_SHADER: &str = r#"
struct KvEpilogueMeta {
    n_head_kv: u32,
    head_dim: u32,
    rope_dim: u32,
    pos: u32,
    freq_base: f32,
    eps: f32,
    pairing: u32,
    k_norm: u32,
    v_norm: u32,
    dst_offset: u32,
    _p0: u32,
    _p1: u32,
    freq_scale: f32,
    ext_factor: f32,
    mscale: f32,
    corr_lo: f32,
    corr_hi: f32,
    _p2: u32,
    _p3: u32,
    _p4: u32,
}

@group(0) @binding(0) var<storage, read_write> kx: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> vx: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> hw: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> hff: array<f32>;
@group(0) @binding(4) var<storage, read_write> kdst: array<DST>;
@group(0) @binding(5) var<storage, read_write> vdst: array<DST>;
@group(0) @binding(6) var<uniform> hm: KvEpilogueMeta;

var<workgroup> hd_head: array<f32, HEAD_MAX>;
var<workgroup> hd_partial: array<f32, 64>;
var<workgroup> hd_stripe: array<f32, 8>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let is_k = wid.x < hm.n_head_kv;
    var h = wid.x;
    if (!is_k) { h = wid.x - hm.n_head_kv; }
    let local = lid.x;
    let n4 = hm.head_dim / 4u;
    let base4 = h * n4;
    let pos = hm.pos;
    let weighted = is_k && hm.k_norm != 0u;
    var normed = hm.v_norm != 0u;
    if (is_k) { normed = hm.k_norm != 0u; }
    let k0 = local;
    let k1 = local + 64u;
    var v0 = vec4<f32>(0.0);
    var v1 = vec4<f32>(0.0);
    if (is_k) {
        if (k0 < n4) { v0 = kx[base4 + k0]; }
        if (k1 < n4) { v1 = kx[base4 + k1]; }
    } else {
        if (k0 < n4) { v0 = vx[base4 + k0]; }
        if (k1 < n4) { v1 = vx[base4 + k1]; }
    }
NORM
    if (is_k) {
ROTATE
    }
    workgroupBarrier();
    let dst = hm.dst_offset + h * hm.head_dim;
    if (k0 < n4) {
        let o = vec4<f32>(
            hd_head[4u * k0], hd_head[4u * k0 + 1u], hd_head[4u * k0 + 2u], hd_head[4u * k0 + 3u]);
        let e = dst + 4u * k0;
        if (is_k) {
            kx[base4 + k0] = o;
            kdst[e] = DST(o.x); kdst[e + 1u] = DST(o.y); kdst[e + 2u] = DST(o.z); kdst[e + 3u] = DST(o.w);
        } else {
            vx[base4 + k0] = o;
            vdst[e] = DST(o.x); vdst[e + 1u] = DST(o.y); vdst[e + 2u] = DST(o.z); vdst[e + 3u] = DST(o.w);
        }
    }
    if (k1 < n4) {
        let o = vec4<f32>(
            hd_head[4u * k1], hd_head[4u * k1 + 1u], hd_head[4u * k1 + 2u], hd_head[4u * k1 + 3u]);
        let e = dst + 4u * k1;
        if (is_k) {
            kx[base4 + k1] = o;
            kdst[e] = DST(o.x); kdst[e + 1u] = DST(o.y); kdst[e + 2u] = DST(o.z); kdst[e + 3u] = DST(o.w);
        } else {
            vx[base4 + k1] = o;
            vdst[e] = DST(o.x); vdst[e + 1u] = DST(o.y); vdst[e + 2u] = DST(o.z); vdst[e + 3u] = DST(o.w);
        }
    }
}
"#;

/// The K/V epilogue for an `f16` or `f32` mirror; `None` for `q8_0`.
pub fn shader_source_kv_epilogue(storage: KvStorage) -> Option<String> {
    let (enable, dst) = match storage {
        KvStorage::F16 => ("enable f16;\n", "f16"),
        KvStorage::F32 => ("", "f32"),
        KvStorage::Q8_0 => return None,
    };
    // The loads read from whichever projection the workgroup serves, so the
    // shared body's `src` reads are already done above it; only its reduce
    // and scale are wanted here.
    let norm = HEAD_WIDE_NORM_WGSL
        .lines()
        .skip_while(|l| !l.contains("hd_partial[local] ="))
        .collect::<Vec<_>>()
        .join("\n");
    let body = KV_EPILOGUE_SHADER
        .replace("HEAD_MAX", &HEAD_WIDE_MAX_DIM.to_string())
        .replace("DST", dst)
        .replace("NORM\n", &format!("{norm}\n"))
        .replace("ROTATE\n", HEAD_WIDE_ROTATE_WGSL);
    Some(format!("{enable}{ROPE_YARN_WGSL}{body}"))
}

/// Greedy (argmax) decode with repeat penalty, entirely on-GPU, so a
/// decode step that's going to sample greedily anyway never has to read
/// back the full `[n_vocab]` logits vector — just the one winning token
/// id (4 bytes instead of, for `E2B`'s 262144-entry vocabulary, ~1 MB).
///
/// Three dispatches, one command encoder (`VulkanBackend::record_argmax_
/// sample` — wgpu's automatic hazard tracking barriers each read-after-
/// write dependency between them, the same established pattern
/// `record_fused_attention`'s split-k phases use):
///
/// 1. **Repeat penalty** (`ARGMAX_PENALTY_SHADER`, one workgroup, thread 0
///    only), strictly sequential over `recent_tokens` in order — mirrors
///    `engine::sampling::apply_repeat_penalty`'s own loop exactly,
///    including its behavior on a repeated token id (penalized once per
///    occurrence, compounding, since each iteration reads the
///    *already-penalized* value the previous iteration just wrote). This
///    can't be parallelized without changing that compounding behavior,
///    but `recent_tokens` is tiny (`repeat_last_n`, 64 by default) next to
///    `n_vocab`, so a single thread doing it sequentially first costs
///    nothing worth optimizing.
/// 2. **Split argmax reduction** (`ARGMAX_SPLIT_SHADER`,
///    `ARGMAX_SPLIT_N` workgroups, 64 threads each — replacing an earlier
///    single-workgroup version that dispatched only 64 threads total over
///    the *whole* `[n_vocab]` buffer, drastically underusing the GPU).
///    Thread `wid.x * 64 + local` finds its own best `(value, index)`
///    globally strided by `ARGMAX_SPLIT_N * 64`, a workgroup tree
///    reduction combines each workgroup's 64 threads into one partial
///    winner, written to `partial_val[wid.x]`/`partial_idx[wid.x]`. A
///    workgroup with no in-range elements at all (`n_vocab` small enough
///    that `wid.x * 64 >= n_vocab`) writes the reduction's untouched
///    sentinel (`-3.4028235e38`, `f32::MIN`) — never a real logit, so
///    phase 3 correctly never picks it.
/// 3. **Merge** (`ARGMAX_REDUCE_SHADER`, reusing `elem4_bind_group_
///    layout`'s exact shape — read, read, read_write, uniform — so no new
///    bind-group plumbing was needed), one
///    workgroup, the identical tree-reduction shape as phase 2 but over
///    the `ARGMAX_SPLIT_N` partial winners instead of `n_vocab` — cheap,
///    since `ARGMAX_SPLIT_N` is tiny next to any real vocabulary.
///
/// Ties (any phase) are resolved arbitrarily (whichever candidate a given
/// comparison happens to keep) rather than matching `engine::sampling`'s
/// CPU `argmax` exactly (`Iterator::max_by`'s "last element wins" rule)
/// — two independently computed `f32` logits landing on the exact same
/// bit pattern doesn't happen with real model output, so this was never
/// worth the extra index-aware tie-break bookkeeping, now spread across
/// two reduction levels instead of one.
///
/// `logits` is mutated in place by phase 1 (the same buffer `record_full_
/// matmul` just produced) — safe because nothing else reads it afterward
/// in this submission, and the next decode step's own matmul dispatch
/// overwrites the whole buffer again before anything reads it.
/// Phase 0 of the GPU sample chain, dispatched only when the model sets a
/// final-logit softcap: `v = cap * tanh(v / cap)` over every logit, in
/// place, before the repeat-penalty and argmax phases. Reproduces the CPU
/// path's softcap → penalty → argmax order exactly (the softcap is
/// monotonic, so it can't change the greedy token, but applying it before
/// the value-dependent repeat penalty keeps byte-parity with the CPU sampler
/// when `repeat_penalty != 1`). Reuses the penalty phase's bind-group layout
/// — only `logits` (binding 0) and `sample_meta` (binding 3) are read.
const ARGMAX_SOFTCAP_SHADER: &str = r#"
struct SampleMeta {
    n_vocab: u32,
    n_recent: u32,
    repeat_penalty: f32,
    logit_softcap: f32,
}

@group(0) @binding(0) var<storage, read_write> logits: array<f32>;
@group(0) @binding(3) var<uniform> sample_meta: SampleMeta;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= sample_meta.n_vocab) {
        return;
    }
    let cap = sample_meta.logit_softcap;
    logits[idx] = cap * tanh(logits[idx] / cap);
}
"#;

const ARGMAX_PENALTY_SHADER: &str = r#"
struct SampleMeta {
    n_vocab: u32,
    n_recent: u32,
    repeat_penalty: f32,
    logit_softcap: f32,
}

@group(0) @binding(0) var<storage, read_write> logits: array<f32>;
@group(0) @binding(1) var<storage, read> recent_tokens: array<u32>;
@group(0) @binding(2) var<storage, read_write> out_token: array<u32>;
@group(0) @binding(3) var<uniform> sample_meta: SampleMeta;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    if (lid.x != 0u) {
        return;
    }
    var i: u32 = 0u;
    loop {
        if (i >= sample_meta.n_recent) {
            break;
        }
        let tok = recent_tokens[i];
        if (tok < sample_meta.n_vocab) {
            let v = logits[tok];
            if (v > 0.0) {
                logits[tok] = v / sample_meta.repeat_penalty;
            } else {
                logits[tok] = v * sample_meta.repeat_penalty;
            }
        }
        i = i + 1u;
    }
}
"#;

const ARGMAX_SPLIT_SHADER: &str = r#"
struct ArgmaxSplitMeta {
    n_vocab: u32,
    n_split: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> partial_val: array<f32>;
@group(0) @binding(2) var<storage, read_write> partial_idx: array<u32>;
@group(0) @binding(3) var<uniform> am: ArgmaxSplitMeta;

var<workgroup> best_val: array<f32, 64>;
var<workgroup> best_idx: array<u32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let local = lid.x;
    var my_best_val: f32 = -3.4028235e38;
    var my_best_idx: u32 = 0u;
    var k: u32 = wid.x * 64u + local;
    let global_stride: u32 = am.n_split * 64u;
    loop {
        if (k >= am.n_vocab) {
            break;
        }
        let v = logits[k];
        if (v > my_best_val) {
            my_best_val = v;
            my_best_idx = k;
        }
        k = k + global_stride;
    }
    best_val[local] = my_best_val;
    best_idx[local] = my_best_idx;
    workgroupBarrier();

    var stride: u32 = 32u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (local < stride && best_val[local + stride] > best_val[local]) {
            best_val[local] = best_val[local + stride];
            best_idx[local] = best_idx[local + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (local == 0u) {
        partial_val[wid.x] = best_val[0];
        partial_idx[wid.x] = best_idx[0];
    }
}
"#;

/// Merges `ARGMAX_SPLIT_SHADER`'s `ARGMAX_SPLIT_N` partial winners into
/// the final token id — reuses `elem4_bind_group_layout`'s exact shape
/// (two read-only storage inputs, one read_write storage output, one
/// uniform), so `em.len` (`ElemMeta`, prepended by `shader_source_
/// argmax_reduce`) is repurposed as the partial count instead of an
/// elementwise length.
const ARGMAX_REDUCE_SHADER_BODY: &str = r#"
@group(0) @binding(0) var<storage, read> partial_val: array<f32>;
@group(0) @binding(1) var<storage, read> partial_idx: array<u32>;
@group(0) @binding(2) var<storage, read_write> out_token: array<u32>;
@group(0) @binding(3) var<uniform> em: ElemMeta;

var<workgroup> best_val: array<f32, 64>;
var<workgroup> best_idx: array<u32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let local = lid.x;
    var my_best_val: f32 = -3.4028235e38;
    var my_best_idx: u32 = 0u;
    var k: u32 = local;
    loop {
        if (k >= em.len) {
            break;
        }
        let v = partial_val[k];
        if (v > my_best_val) {
            my_best_val = v;
            my_best_idx = partial_idx[k];
        }
        k = k + 64u;
    }
    best_val[local] = my_best_val;
    best_idx[local] = my_best_idx;
    workgroupBarrier();

    var stride: u32 = 32u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (local < stride && best_val[local + stride] > best_val[local]) {
            best_val[local] = best_val[local + stride];
            best_idx[local] = best_idx[local + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (local == 0u) {
        out_token[0] = best_idx[0];
    }
}
"#;

/// Phase 1 of the device top-k that a *sampled* decode step reads back
/// instead of the whole vocabulary: each workgroup takes a slice of the
/// (already penalized) logits into shared memory and extracts its `k`
/// largest by repeated argmax — `k` rounds of a per-thread scan and a tree
/// reduce, the winner blanked after each — writing `(value, index)` pairs
/// to `partial_*[wid * k ..]`. `k <= TOPK_MAX`, `slice <= TOPK_SLICE`; the
/// backend sizes the dispatch to both. A sampler that only ever looks at
/// the top `k` (top-k, then top-p/min-p over those) gets exactly the
/// candidates it would have found itself; temperature scales every logit
/// alike, so it can be applied to the `k` survivors afterwards.
const TOPK_SPLIT_SHADER: &str = r#"
struct TopkMeta {
    n_vocab: u32,
    n_split: u32,
    k: u32,
    slice: u32,
}

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> partial_val: array<f32>;
@group(0) @binding(2) var<storage, read_write> partial_idx: array<u32>;
@group(0) @binding(3) var<uniform> tm: TopkMeta;

var<workgroup> vals: array<f32, 4096>;
var<workgroup> best_val: array<f32, 256>;
var<workgroup> best_idx: array<u32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let local = lid.x;
    let base = wid.x * tm.slice;
    var j: u32 = local;
    loop {
        if (j >= tm.slice) {
            break;
        }
        let g = base + j;
        var v: f32 = -3.4028235e38;
        if (g < tm.n_vocab) {
            v = logits[g];
        }
        vals[j] = v;
        j = j + 256u;
    }
    workgroupBarrier();

    var r: u32 = 0u;
    loop {
        if (r >= tm.k) {
            break;
        }
        var bv: f32 = -3.4028235e38;
        var bi: u32 = 0u;
        j = local;
        loop {
            if (j >= tm.slice) {
                break;
            }
            let v = vals[j];
            if (v > bv) {
                bv = v;
                bi = j;
            }
            j = j + 256u;
        }
        best_val[local] = bv;
        best_idx[local] = bi;
        workgroupBarrier();
        var stride: u32 = 128u;
        loop {
            if (stride == 0u) {
                break;
            }
            if (local < stride) {
                let ov = best_val[local + stride];
                let oi = best_idx[local + stride];
                let mv = best_val[local];
                // Larger value wins; on a tie the lower index, so the order
                // is the one a stable descending sort gives.
                if (ov > mv || (ov == mv && oi < best_idx[local])) {
                    best_val[local] = ov;
                    best_idx[local] = oi;
                }
            }
            workgroupBarrier();
            stride = stride / 2u;
        }
        if (local == 0u) {
            partial_val[wid.x * tm.k + r] = best_val[0];
            partial_idx[wid.x * tm.k + r] = base + best_idx[0];
            vals[best_idx[0]] = -3.4028235e38;
        }
        workgroupBarrier();
        r = r + 1u;
    }
}
"#;

/// Phase 2 of the device top-k: one workgroup merges every slice's `k`
/// pairs (`em.len` of them) into the final `k` (`em.aux`), writing values
/// to `out[0..k]` and indices, as bits, to `out[k..2k]` — one buffer, one
/// readback. Reuses the `elem4` binding shape like the argmax merge.
const TOPK_REDUCE_SHADER_BODY: &str = r#"
@group(0) @binding(0) var<storage, read> partial_val: array<f32>;
@group(0) @binding(1) var<storage, read> partial_idx: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> em: ElemMeta;

var<workgroup> vals: array<f32, 4096>;
var<workgroup> ids: array<u32, 4096>;
var<workgroup> best_val: array<f32, 256>;
var<workgroup> best_idx: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let local = lid.x;
    let n = min(em.len, 4096u);
    let k = em.aux;
    var j: u32 = local;
    loop {
        if (j >= n) {
            break;
        }
        vals[j] = partial_val[j];
        ids[j] = partial_idx[j];
        j = j + 256u;
    }
    workgroupBarrier();

    var r: u32 = 0u;
    loop {
        if (r >= k) {
            break;
        }
        var bv: f32 = -3.4028235e38;
        var bi: u32 = 0u;
        j = local;
        loop {
            if (j >= n) {
                break;
            }
            let v = vals[j];
            if (v > bv || (v == bv && ids[j] < ids[bi])) {
                bv = v;
                bi = j;
            }
            j = j + 256u;
        }
        best_val[local] = bv;
        best_idx[local] = bi;
        workgroupBarrier();
        var stride: u32 = 128u;
        loop {
            if (stride == 0u) {
                break;
            }
            if (local < stride) {
                let ov = best_val[local + stride];
                let oi = best_idx[local + stride];
                let mv = best_val[local];
                if (ov > mv || (ov == mv && ids[oi] < ids[best_idx[local]])) {
                    best_val[local] = ov;
                    best_idx[local] = oi;
                }
            }
            workgroupBarrier();
            stride = stride / 2u;
        }
        if (local == 0u) {
            let slot = best_idx[0];
            out[r] = best_val[0];
            out[k + r] = bitcast<f32>(ids[slot]);
            vals[slot] = -3.4028235e38;
        }
        workgroupBarrier();
        r = r + 1u;
    }
}
"#;

/// Candidates one device top-k can return, and the elements one of its
/// slices holds — the shared-memory sizes baked into the two shaders.
pub const TOPK_MAX: u32 = 64;
pub const TOPK_SLICE: u32 = 4096;

pub fn shader_source_topk_split() -> String {
    TOPK_SPLIT_SHADER.to_string()
}

pub fn shader_source_topk_reduce() -> String {
    format!("{ELEM_META}\n{TOPK_REDUCE_SHADER_BODY}")
}

pub fn shader_source_argmax_softcap() -> String {
    ARGMAX_SOFTCAP_SHADER.to_string()
}

pub fn shader_source_argmax_penalty() -> String {
    ARGMAX_PENALTY_SHADER.to_string()
}

pub fn shader_source_argmax_split() -> String {
    ARGMAX_SPLIT_SHADER.to_string()
}

pub fn shader_source_argmax_reduce() -> String {
    format!("{ELEM_META}\n{ARGMAX_REDUCE_SHADER_BODY}")
}

#[cfg(test)]
mod flash_lds_tests {
    use super::*;

    /// Shared memory per workgroup on the two APIs this engine runs on.
    const LDS_64K: u32 = 64 * 1024;
    const LDS_32K: u32 = 32 * 1024;

    /// Total shared memory the flash kernel declares at a given tile size.
    fn total_lds(head_dim: u32, tile_pos: u32) -> u32 {
        tile_pos * (head_dim + 1) * 2 + flash_fixed_lds_bytes(head_dim)
    }

    /// **The whole point of taking the limit from the device.** The tile has to
    /// fit under whatever ceiling it is actually running on, including the one
    /// that is half the size the old fixed ~34 KB budget assumed.
    ///
    /// Under that budget, `head_dim = 512` produced a 32-position tile: 32 832
    /// bytes of `k_shmem` alone, over a 32 KiB ceiling before `acc` was
    /// counted. On such a device `ORANGU_FLASH_ATTN=1` did not run slowly, it
    /// failed to create the pipeline.
    #[test]
    fn the_tile_fits_the_devices_shared_memory_not_a_constant() {
        for head_dim in [128u32, 256, 512] {
            for limit in [LDS_32K, LDS_64K] {
                let tp = flash_tile_positions(head_dim, limit);
                assert!(
                    total_lds(head_dim, tp) <= limit,
                    "head_dim {head_dim} at a {limit} B limit picked {tp} positions, \
                     needing {} B",
                    total_lds(head_dim, tp)
                );
                assert!((8..=64).contains(&tp), "tile {tp} outside 8..=64");
                assert!(tp.is_power_of_two(), "tile {tp} is not a power of two");
            }
        }
    }

    /// A smaller ceiling must actually cost tile size where the tile is the
    /// thing that does not fit — otherwise the limit is being read and
    /// ignored. At `head_dim = 512` a 64 KiB device gets the full 32-position
    /// tile and a 32 KiB one cannot.
    #[test]
    fn a_tighter_limit_gives_a_smaller_tile() {
        let big = flash_tile_positions(512, LDS_64K);
        let small = flash_tile_positions(512, LDS_32K);
        assert!(
            small < big,
            "a 32 KiB device should stage fewer positions than a 64 KiB one, got {small} vs {big}"
        );
        // ...and where the tile fits either way, both get the maximum, so the
        // change is not a blanket shrink.
        assert_eq!(
            flash_tile_positions(128, LDS_32K),
            flash_tile_positions(128, LDS_64K)
        );
    }

    /// The `8..=64` clamp has a floor, so a head_dim large enough would return
    /// a tile that does not fit however tight the budget. Flash is only built
    /// for `head_dim == 512` (see `attn_split_pipeline_for`), where the floor
    /// is nowhere near binding — this pins that assumption so enabling the
    /// kernel for a wider head would fail here rather than at pipeline
    /// creation on someone's machine.
    #[test]
    fn the_tile_floor_is_not_binding_for_any_head_dim_flash_is_built_for() {
        assert!(
            total_lds(512, 8) <= LDS_32K,
            "the minimum tile must fit the tightest device at the head_dim flash uses"
        );
    }
}

#[cfg(test)]
mod coop_geom_tests {
    use super::*;

    /// The default has to satisfy its own rules, or every other check here is
    /// checking a shape nothing uses.
    #[test]
    fn the_default_geometry_is_valid() {
        assert_eq!(COOP_GEOM_DEFAULT.check(), Ok(()));
        assert_eq!(COOP_GEOM_DEFAULT.tile_rows(), 32);
        assert_eq!(COOP_GEOM_DEFAULT.tile_tokens(), 64);
        assert_eq!(COOP_GEOM_DEFAULT.lds_bytes(), 12_288);
        assert_eq!(COOP_GEOM_DEFAULT.run(), 16);
    }

    /// The default's shared memory has to fit the *smallest* device this
    /// engine runs on, not the one it was tuned on. `wgpu`'s default
    /// `max_compute_workgroup_storage_size` is 16 KiB — half the 32 KiB
    /// `check` allows — so a geometry can pass validation here and still fail
    /// pipeline creation on a conforming device. The default must not be one
    /// of those.
    #[test]
    fn the_default_fits_the_smallest_conforming_device() {
        const WGPU_DEFAULT_WORKGROUP_STORAGE: u32 = 16 * 1024;
        assert!(
            COOP_GEOM_DEFAULT.lds_bytes() <= WGPU_DEFAULT_WORKGROUP_STORAGE,
            "{} B of shared memory exceeds wgpu's default limit of {WGPU_DEFAULT_WORKGROUP_STORAGE} B",
            COOP_GEOM_DEFAULT.lds_bytes(),
        );
    }

    /// The rule that matters most, because breaking it is silent and
    /// type-specific: `coop_tiled_run_fill` hoists `Q6_K`'s 8-bit scale out of
    /// the staging run, and `Q6_K` changes that scale every **16** elements
    /// where `Q4_K`/`Q5_K` change theirs every 32. A geometry giving `run == 32`
    /// passes a "divides 32" test and dequantizes half of every `Q6_K` run with
    /// the wrong scale. Two such geometries came out of a real sweep.
    #[test]
    fn a_staging_run_wider_than_a_q6_k_scale_group_is_rejected() {
        for g in [
            // 64-row tile: run 32.
            CoopGeom {
                threads_y: 8,
                threads_x: 8,
                reg_rows: 8,
                reg_tokens: 8,
                chunk: 32,
            },
            // chunk 64: run 32.
            CoopGeom {
                threads_y: 8,
                threads_x: 8,
                reg_rows: 4,
                reg_tokens: 8,
                chunk: 64,
            },
        ] {
            assert_eq!(g.run(), 32, "geometry chosen to give run 32");
            let err = g.check().expect_err("run 32 must be rejected");
            assert!(
                err.contains("16"),
                "reason should name the 16-element group: {err}"
            );
        }
        // And the ones that do divide 16 are accepted.
        for chunk in [16u32, 32] {
            let g = CoopGeom {
                chunk,
                ..COOP_GEOM_DEFAULT
            };
            assert!(16u32.is_multiple_of(g.run()));
            assert_eq!(g.check(), Ok(()), "chunk {chunk} should be valid");
        }
    }

    /// The register block indexes the staging tiles as `vec4`, so each thread's
    /// base offset must be 4-aligned — `reg_rows = 2` gives odd threads a
    /// misaligned read and wrong output on 18 tests.
    #[test]
    fn a_register_block_that_breaks_vec4_alignment_is_rejected() {
        for (ry, rx) in [(2u32, 8u32), (4, 2), (1, 8), (8, 2)] {
            let g = CoopGeom {
                reg_rows: ry,
                reg_tokens: rx,
                ..COOP_GEOM_DEFAULT
            };
            assert!(g.check().is_err(), "reg {ry}x{rx} should be rejected");
        }
        assert_eq!(
            CoopGeom {
                threads_y: 4,
                threads_x: 16,
                reg_rows: 8,
                reg_tokens: 4,
                chunk: 32
            }
            .check(),
            Ok(()),
            "a transposed thread grid with 4-aligned register blocks is fine"
        );
    }

    #[test]
    fn thread_count_and_shared_memory_are_bounded() {
        assert!(
            CoopGeom {
                threads_y: 8,
                threads_x: 4,
                ..COOP_GEOM_DEFAULT
            }
            .check()
            .is_err()
        );
        // 32 KiB of shared memory is the cap; a big chunk blows it.
        let huge = CoopGeom {
            chunk: 256,
            ..COOP_GEOM_DEFAULT
        };
        assert!(huge.lds_bytes() > 32 * 1024);
        assert!(huge.check().is_err());
    }

    /// The activation fill hands out four-token quads, so the threads sharing
    /// a k column have to divide the tile's quads evenly — otherwise some
    /// thread owns part of a quad and its store stops being a whole-vector
    /// store, which is the entire property that makes `tile_x`'s `vec4` form
    /// portable (see [`coop_vec4_tiles`]).
    #[test]
    fn a_geometry_that_splits_a_token_quad_across_threads_is_rejected() {
        // chunk 2 leaves 32 threads sharing each k column, against 16 quads.
        let g = CoopGeom {
            chunk: 2,
            ..COOP_GEOM_DEFAULT
        };
        let err = g.check().expect_err("chunk 2 must be rejected");
        assert!(err.contains("quad"), "reason should name the quad: {err}");
        // 4 and up divide evenly and stay valid.
        for chunk in [4u32, 8, 16, 32] {
            let g = CoopGeom {
                chunk,
                ..COOP_GEOM_DEFAULT
            };
            assert_eq!(g.check(), Ok(()), "chunk {chunk} should be valid");
        }
    }
}

#[cfg(test)]
mod coop_tiled_fill_tests {
    use super::*;

    /// The dword fill is a correctness cliff, not a tuning knob: `Q6_K`'s
    /// 210-byte block puts every odd block on an odd dword boundary, and
    /// enabling it there fails
    /// `matmul_matches_cpu_backend_cooperative_path_q6_k` immediately —
    /// verified by temporarily removing the exclusion, not merely reasoned
    /// about. This pins the gate so a later type addition has to think about
    /// alignment rather than inherit it.
    #[test]
    fn only_four_byte_aligned_blocks_take_the_dword_fill() {
        assert!(
            tile_dword_fill_ok(GGML_TYPE_Q4_K, 8),
            "Q4_K blocks are 144B"
        );
        assert!(
            tile_dword_fill_ok(GGML_TYPE_Q5_K, 8),
            "Q5_K blocks are 176B"
        );
        assert!(
            !tile_dword_fill_ok(GGML_TYPE_Q6_K, 8),
            "Q6_K blocks are 210B — 2-aligned, so dword loads straddle"
        );
    }

    /// A run that is not a multiple of four would need a *dynamic* byte index
    /// into the loaded dwords, which goes through scratch and loses more than
    /// the loads save.
    #[test]
    fn a_run_that_is_not_a_multiple_of_four_keeps_the_byte_fill() {
        for run in [1u32, 2, 8, 16] {
            assert_eq!(
                tile_dword_fill_ok(GGML_TYPE_Q4_K, run),
                run.is_multiple_of(4),
                "run {run}"
            );
        }
    }

    /// What the two paths actually emit, read out of the generated text —
    /// the same reason the fill tests above parse WGSL rather than
    /// re-deriving it.
    #[test]
    fn the_dword_fill_replaces_four_byte_loads_with_one() {
        let src = coop_tiled_run_fill(GGML_TYPE_Q4_K, 8);
        assert_eq!(
            src.matches("weights[q_w +").count(),
            2,
            "eight bytes must come from two dwords, not eight loads:\n{src}"
        );
        assert!(
            !src.contains("read_u8(q_base"),
            "no per-byte load may survive on the aligned path:\n{src}"
        );

        // And Q6_K keeps every one of them.
        let q6 = coop_tiled_run_fill(GGML_TYPE_Q6_K, 8);
        assert_eq!(
            q6.matches("read_u8(ql_base").count(),
            8,
            "Q6_K stays per-byte"
        );
    }

    /// The straight-line activation fill, as the offsets it actually emits.
    ///
    /// Parsed out of the generated WGSL rather than re-derived from
    /// [`CoopGeom`], because a re-derivation would agree with a wrong
    /// generator by construction — the point is to check the text the driver
    /// is handed.
    fn emitted_quad_offsets(src: &str) -> Vec<u32> {
        src.lines()
            .filter_map(|l| l.trim().strip_prefix("store_x4(x_tt0 + "))
            .filter_map(|rest| rest.split_once('u'))
            .filter_map(|(n, _)| n.parse::<u32>().ok())
            .collect()
    }

    fn tiled_source(tiles: CoopVec4Tiles) -> String {
        shader_source_coop_tiled(crate::engine::quant::GGML_TYPE_F32, tiles)
            .expect("f32 has a tiled kernel")
    }

    /// Every token of the tile is staged exactly once.
    ///
    /// A thread's quads are `x_tt0 + <emitted offset>`, where `x_tt0` is
    /// `(local / CHUNK) * 4` and so ranges over `0, 4, .., (threads_per_k-1)*4`.
    /// Union those and the result has to be precisely the tile's quad starts —
    /// no token staged twice (one thread's write silently losing to another's)
    /// and none left holding whatever the previous k-chunk put there, which is
    /// the failure an off-by-one in the offsets produces and which no
    /// bounds-check would catch.
    #[test]
    fn the_activation_fill_stages_every_token_exactly_once() {
        let g = COOP_GEOM_DEFAULT;
        let offsets = emitted_quad_offsets(&tiled_source(CoopVec4Tiles { w: true, x: true }));
        let threads_per_k = COOP_THREADS / g.chunk;
        assert_eq!(
            offsets.len() as u32,
            (g.tile_tokens() / 4) / threads_per_k,
            "one store per quad the thread owns"
        );
        let mut covered: Vec<u32> = offsets
            .iter()
            .flat_map(|&off| (0..threads_per_k).map(move |grp| off + grp * 4))
            .collect();
        covered.sort_unstable();
        let want: Vec<u32> = (0..g.tile_tokens()).step_by(4).collect();
        assert_eq!(
            covered, want,
            "the fill must cover every four-token quad of the tile exactly once"
        );
    }

    /// **The invariant the Metal path depends on.** In the straightened fill,
    /// nothing writes a *component* of `tile_x` — every store goes through
    /// `store_x4`, which writes a whole vector (or, for a scalar tile, four
    /// independent elements). A backend whose shared-vector component store is
    /// a read-modify-write of all 16 bytes then computes the same answer as
    /// one whose is not.
    ///
    /// Checked on the generated text for every tile-form combination, since
    /// the `vec4`/scalar choice is per tile and only one of the four
    /// combinations is what any given device runs.
    #[test]
    fn the_activation_fill_never_writes_a_component_of_a_shared_vector() {
        for (w, x) in [(true, true), (false, true), (true, false), (false, false)] {
            let src = tiled_source(CoopVec4Tiles { w, x });
            let fill = src
                .split_once("// Straight-line, loads-before-stores")
                .expect("the straightened fill is in the generated source")
                .1
                .split_once("workgroupBarrier()")
                .expect("the fill ends at the barrier")
                .0;
            assert!(
                !fill.contains("store_x("),
                "tile_x vec4={x}: the fill still calls the per-element store_x, \
                 which writes one component of a shared vector"
            );
            assert!(
                fill.contains("store_x4("),
                "tile_x vec4={x}: the fill should write whole quads"
            );
            // And the helper it calls is a whole-vector store, not a
            // component one. The shape that breaks is a *double* index on the
            // left — array element, then vector component (`[i][c] =`); one
            // index is a whole element of whichever array shape the tile has,
            // which is always safe.
            let helper = src
                .split_once("fn store_x4(")
                .expect("store_x4 is generated")
                .1
                .split_once('}')
                .expect("store_x4 has a body")
                .0;
            assert!(
                !helper.contains("]["),
                "tile_w vec4={w}, tile_x vec4={x}: store_x4's body writes a \
                 vector component: {helper}"
            );
        }
    }

    /// `ORANGU_COOP_SCALAR_TILES` has to be able to name one tile, because the
    /// combination a backend without component stores runs — scalar `tile_w`,
    /// `vec4` `tile_x` — is otherwise unreachable on a backend that has them,
    /// and so untestable on the hardware that is to hand.
    ///
    /// An unrecognized value forces both, deliberately: `=1` is what the
    /// variable meant before it could name a tile, and a typo should give the
    /// conservative answer rather than silently leaving a tile on the form
    /// the operator was trying to rule out.
    #[test]
    fn the_scalar_tile_override_can_name_one_tile() {
        assert_eq!(forced_scalar_tiles(""), (false, false));
        assert_eq!(forced_scalar_tiles("  "), (false, false));
        assert_eq!(forced_scalar_tiles("w"), (true, false));
        assert_eq!(forced_scalar_tiles("x"), (false, true));
        assert_eq!(forced_scalar_tiles("1"), (true, true));
        assert_eq!(forced_scalar_tiles("wx"), (true, true));
    }
}

#[cfg(test)]
mod iq_grid_offset_tests {
    use crate::engine::iq_grids::packed;

    /// [`super::IQ_GRID_PRELUDE`]'s offsets must be the ones
    /// `engine::iq_grids::packed` actually lays the buffer out at.
    ///
    /// The packing moved to `iq_grids` so the CUDA/HIP/OpenCL backends could
    /// upload the identical bytes — `vendor_shaders::iq_grid_prelude`
    /// *formats* its constants in from there and so cannot drift. This WGSL
    /// still spells them as literals, which is the one place they can. A
    /// table that grows would shift every later offset, and a shader reading
    /// the old ones would return plausible, wrong weights rather than fail.
    ///
    /// Parsed out of the shader text rather than compared against a second
    /// copy of the numbers, because a second copy would agree with a wrong
    /// prelude by construction.
    #[test]
    fn iq_grid_prelude_offsets_match_the_packing() {
        let src = super::IQ_GRID_PRELUDE;
        let declared = |name: &str| -> u32 {
            let needle = format!("const {name}: u32 = ");
            let start = src
                .find(&needle)
                .unwrap_or_else(|| panic!("{name} is not declared in IQ_GRID_PRELUDE"))
                + needle.len();
            let digits: String = src[start..]
                .chars()
                .take_while(char::is_ascii_digit)
                .collect();
            digits.parse().expect("an offset literal")
        };
        for (name, expected) in [
            ("IQ2XS_GRID_OFF", packed::IQ2XS_GRID_OFF),
            ("IQ2S_GRID_OFF", packed::IQ2S_GRID_OFF),
            ("IQ3XXS_GRID_OFF", packed::IQ3XXS_GRID_OFF),
            ("IQ3S_GRID_OFF", packed::IQ3S_GRID_OFF),
            ("KSIGNS_OFF", packed::KSIGNS_OFF),
            ("KVALUES_IQ4NL_OFF", packed::KVALUES_IQ4NL_OFF),
            ("IQ2XXS_GRID_OFF", packed::IQ2XXS_GRID_OFF),
            ("IQ1S_GRID_OFF", packed::IQ1S_GRID_OFF),
        ] {
            assert_eq!(declared(name), expected, "IQ_GRID_PRELUDE's {name}");
        }
        // And the packing really is that long, so the last table is whole.
        assert_eq!(packed::words().len(), packed::WORDS);
    }
}

/// **Every generated kernel, translated to Metal.**
///
/// The kernels are written once in WGSL and translated per backend by the
/// same library `wgpu` uses, so a kernel that is valid WGSL and valid SPIR-V
/// can still be rejected by the Metal compiler — and on a machine without a
/// Metal device nothing here would ever notice. These tests run the
/// translation itself, which needs no GPU at all.
#[cfg(test)]
mod msl_tests {
    use super::*;

    /// The `dot4I8Packed` polyfill the Metal backend writes: one
    /// `packed_char4 <name> = as_type<packed_char4>(arg);` per call, named
    /// after the **argument**. Two calls on the same argument in one block
    /// therefore declare the same name twice, which the Metal compiler
    /// rejects as a redefinition — while WGSL, SPIR-V and every test on a
    /// non-Apple machine are perfectly happy.
    ///
    /// Reported from an `M2 Max`: the integer-dot GEMM failed to compile and
    /// took the request with it. A kernel's accumulate statements each sit in
    /// their own block for this reason; this test is what keeps them there.
    fn msl_of(label: &str, wgsl: &str) -> String {
        let module = naga::front::wgsl::parse_str(wgsl)
            .unwrap_or_else(|e| panic!("{label}: not valid WGSL: {e:?}"));
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{label}: did not validate: {e:?}"));
        let mut out = String::new();
        // Metal 2.4, which is what this project's own Apple machines report
        // and what `wgpu` then asks for. The version matters: below 2.1 the
        // backend expands `dot4I8Packed` inline, and at or above it writes a
        // `packed_char4` temporary per call — the form that collides.
        let options = naga::back::msl::Options {
            lang_version: (2, 4),
            ..naga::back::msl::Options::default()
        };
        let pipeline = naga::back::msl::PipelineOptions::default();
        let mut writer = naga::back::msl::Writer::new(&mut out);
        writer
            .write(&module, &info, &options, &pipeline)
            .unwrap_or_else(|e| panic!("{label}: no Metal translation: {e:?}"));
        out
    }

    /// The names declared twice **in one scope** in `msl`, with the lines
    /// they were declared on.
    ///
    /// Scope matters and is the whole point: the backend derives these names
    /// from an expression, so two statements that share an argument get the
    /// same name. In sibling blocks that is ordinary shadowing and Metal
    /// accepts it; in one block it is a redefinition and Metal refuses the
    /// kernel.
    fn redeclared(msl: &str) -> Vec<(String, Vec<usize>)> {
        let mut clashes: Vec<(String, Vec<usize>)> = Vec::new();
        let mut scopes: Vec<std::collections::HashMap<String, usize>> = vec![Default::default()];
        for (n, line) in msl.lines().enumerate() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("packed_char4 ")
                && let Some(name) = rest.split_whitespace().next()
                && let Some(scope) = scopes.last_mut()
                && let Some(first) = scope.insert(name.to_string(), n + 1)
            {
                clashes.push((name.to_string(), vec![first, n + 1]));
            }
            for c in trimmed.chars() {
                match c {
                    '{' => scopes.push(Default::default()),
                    '}' => {
                        scopes.pop();
                        if scopes.is_empty() {
                            scopes.push(Default::default());
                        }
                    }
                    _ => {}
                }
            }
        }
        clashes
    }

    /// The check above, on what the backend writes either way: two
    /// declarations in one block are the defect, the same two in sibling
    /// blocks are not. Without this the check passed on a kernel Metal
    /// rejects, which is how the defect reached a user.
    #[test]
    fn the_redefinition_check_reads_scopes() {
        let same_block = "\
fn() {
    packed_char4 reinterpreted_packed_char4_e1 = as_type<packed_char4>(q0);
    a = dot(...);
    packed_char4 reinterpreted_packed_char4_e1 = as_type<packed_char4>(q0);
    b = dot(...);
}";
        assert_eq!(
            redeclared(same_block)
                .iter()
                .map(|(n, _)| n.as_str())
                .collect::<Vec<_>>(),
            ["reinterpreted_packed_char4_e1"],
        );
        let sibling_blocks = "\
fn() {
    {
        packed_char4 reinterpreted_packed_char4_e1 = as_type<packed_char4>(q0);
        a = dot(...);
    }
    {
        packed_char4 reinterpreted_packed_char4_e1 = as_type<packed_char4>(q0);
        b = dot(...);
    }
}";
        assert!(redeclared(sibling_blocks).is_empty());
    }

    /// Writes the Metal translation of one kernel to
    /// `ORANGU_MSL_OUT=<path>` — for reading what the backend actually
    /// emitted, next to the WGSL `ORANGU_DUMP_SHADERS` writes.
    #[test]
    #[ignore]
    fn dump_one_kernel_as_metal() {
        let Ok(path) = std::env::var("ORANGU_MSL_OUT") else {
            eprintln!("set ORANGU_MSL_OUT=<path> and ORANGU_MSL_TYPE=<ggml type>");
            return;
        };
        let ggml_type: u32 = std::env::var("ORANGU_MSL_TYPE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(crate::engine::quant::GGML_TYPE_Q8_0);
        let src = shader_source_mmq_wide_bytes(ggml_type, MMQ_MID_TILE_ROWS);
        std::fs::write(&path, msl_of("dump", &src)).expect("write the translation");
        eprintln!("wrote {path}");
    }

    /// Writes the Metal translation of the `Q4_K` wide kernel to
    /// `ORANGU_MSL_OUT2`.
    #[test]
    #[ignore]
    fn dump_q4k_as_metal() {
        let Ok(path) = std::env::var("ORANGU_MSL_OUT2") else {
            return;
        };
        let src = shader_source_mmq_q4k_wide(MMQ_MID_TILE_ROWS);
        std::fs::write(&path, msl_of("q4k", &src)).expect("write the translation");
    }

    /// The integer-dot prefill GEMM, per type and row tile: the kernel the
    /// Metal compiler rejected.
    #[test]
    fn the_integer_dot_gemm_translates_to_metal() {
        for rows in [MMQ_MID_TILE_ROWS, MMQ_WIDE_TILE_ROWS] {
            let mut sources = vec![
                (
                    format!("mmq_q4k_rows{rows}"),
                    shader_source_mmq_q4k_wide(rows),
                ),
                (
                    format!("mmq_q6k_rows{rows}"),
                    shader_source_mmq_q6k_wide(rows),
                ),
            ];
            for &ggml_type in mmq_wide_bytes_types() {
                sources.push((
                    format!("mmq_bytes_{ggml_type}_rows{rows}"),
                    shader_source_mmq_wide_bytes(ggml_type, rows),
                ));
            }
            for (label, wgsl) in sources {
                let msl = msl_of(&label, &wgsl);
                let clashes = redeclared(&msl);
                assert!(
                    clashes.is_empty(),
                    "{label}: Metal would reject these redefinitions: {:?}",
                    clashes
                        .iter()
                        .map(|(n, at)| format!("{n} at {at:?}"))
                        .collect::<Vec<_>>()
                );
            }
        }
    }
}
