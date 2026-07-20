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
//! by both, concatenated with a type's `*_COOP_MIDDLE` and the relevant
//! `MAIN_*_SUFFIX` once at `VulkanBackend` construction time into 18
//! complete, self-contained WGSL modules (9 types × 2 dispatch strategies).

use crate::engine::quant::{
    GGML_TYPE_BF16, GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q4_K, GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0,
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
/// dequant reuse matches that kernel; `TILE_ROWS` (16) additionally reuses
/// activations across output rows, which the per-row cooperative kernel
/// does not (one workgroup per row, so every row's workgroup re-reads
/// `x` from global memory independently). `CHUNK` (32) is the K-dimension
/// streaming granularity and is deliberately *smaller* than the K-quant
/// types' native super-block size (`BLOCK_ELEMS = 256` for `Q4_K`/`Q5_K`/
/// `Q6_K`) so `tile_w`/`tile_x`'s combined shared-memory footprint
/// (`(TILE_ROWS + TILE_TOKENS) * CHUNK * 4` bytes = 10 KiB) stays bounded
/// regardless of `BLOCK_ELEMS` — using `BLOCK_ELEMS` itself as the tile
/// depth for `Q4_K` would need `(16 + 64) * 256 * 4` = 80 KiB, well past
/// typical workgroup-shared-memory limits. `elem_at` (below) restates
/// `dequant_element`'s existing `block_idx = k / BLOCK_ELEMS; local_k = k %
/// BLOCK_ELEMS` split (already used by `MAIN_REDUCE_SUFFIX`) as a small
/// helper so the K-loop can stream in `CHUNK`-sized pieces without knowing
/// or caring how big a type's native block actually is.
pub const COOP_TILE_ROWS: u32 = 16;
pub const COOP_TILE_TOKENS: u32 = 64;
pub const COOP_CHUNK: u32 = 32;

/// The tiled-GEMM alternative to `MAIN_COOP_SUFFIX` — see `Self::
/// shader_source_coop_tiled` and `MAIN_COOP_SUFFIX`'s own doc comment for
/// why this is now the default (opt out with `ORANGU_NO_TILED_
/// PREFILL=1`) rather than staying opt-in.
///
/// One workgroup computes a `TILE_ROWS × TILE_TOKENS` output tile,
/// streaming the K dimension through shared memory in `CHUNK`-sized
/// pieces: each of the 64 threads (arranged as an 8-per-column ×
/// 16-per-row grid — `THREADS_Y × THREADS_X`) cooperatively fills
/// `tile_w`/`tile_x` for the current chunk (`elem_at`/`x[...]`, grid-
/// strided across the tile the same way `MAIN_REDUCE_SUFFIX` grid-strides
/// across a row), then owns a small `REG_ROWS × REG_TOKENS` (4×4) register
/// block of the output tile, accumulating its own 16 output elements'
/// partial dot products against the shared chunk before the next chunk
/// overwrites `tile_w`/`tile_x`. This gives every weight element
/// `TILE_TOKENS`-way reuse (same as `MAIN_COOP_SUFFIX`) *and* every
/// activation element `TILE_ROWS`-way reuse (which `MAIN_COOP_SUFFIX` has
/// none of at all), at the cost of finer-grained K-streaming than that
/// kernel's native per-block granularity — more `workgroupBarrier` rounds
/// for the K-quant types, whose native block is 256 elements wide vs. this
/// kernel's fixed 32-element `CHUNK`, but no additional per-element
/// dequant cost (`dequant_element` already re-derives each element's
/// scale/min independently regardless of how the caller chunks its calls).
/// Out-of-range rows/tokens (a tile straddling the matrix edge) are zero-
/// filled while loading and skipped while writing, the same bounds-check
/// idiom `MAIN_REDUCE_SUFFIX` already uses for `REDUCE_N_ROWS`-imperfect
/// `out_dim`.
const MAIN_COOP_TILED_SUFFIX: &str = r#"
const TILE_ROWS: u32 = %TILE_ROWS%u;
const TILE_TOKENS: u32 = %TILE_TOKENS%u;
const CHUNK: u32 = %CHUNK%u;
const THREADS_Y: u32 = 4u;
const THREADS_X: u32 = 16u;
const REG_ROWS: u32 = 4u;
const REG_TOKENS: u32 = 4u;

var<workgroup> tile_w: array<f32, %TILE_ROWS%u * %CHUNK%u>;
var<workgroup> tile_x: array<f32, %TILE_TOKENS%u * %CHUNK%u>;

fn elem_at(row_byte_base: u32, k: u32) -> f32 {
    let block_idx = k / BLOCK_ELEMS;
    let local_k = k % BLOCK_ELEMS;
    return dequant_element(row_byte_base + block_idx * BLOCK_BYTES, local_k);
}

@compute @workgroup_size(64)
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

    var acc: array<f32, REG_ROWS * REG_TOKENS>;
    var zi: u32 = 0u;
    loop {
        if (zi >= REG_ROWS * REG_TOKENS) {
            break;
        }
        acc[zi] = 0.0;
        zi = zi + 1u;
    }

    let in_dim = params.in_dim;
    var chunk_start: u32 = 0u;
    loop {
        if (chunk_start >= in_dim) {
            break;
        }

        var fi: u32 = local;
        loop {
            if (fi >= TILE_ROWS * CHUNK) {
                break;
            }
            let rr = fi / CHUNK;
            let kk = fi % CHUNK;
            let row_idx = row_start + rr;
            let k_global = chunk_start + kk;
            if (row_idx < params.out_dim && k_global < in_dim) {
                tile_w[fi] = elem_at(row_idx * params.row_bytes, k_global);
            } else {
                tile_w[fi] = 0.0;
            }
            fi = fi + 64u;
        }

        var fj: u32 = local;
        loop {
            if (fj >= TILE_TOKENS * CHUNK) {
                break;
            }
            let tt = fj / CHUNK;
            let kk = fj % CHUNK;
            let token_idx = token_start + tt;
            let k_global = chunk_start + kk;
            if (token_idx < params.n_tokens && k_global < in_dim) {
                tile_x[fj] = x[token_idx * in_dim + k_global];
            } else {
                tile_x[fj] = 0.0;
            }
            fj = fj + 64u;
        }

        workgroupBarrier();

        var k: u32 = 0u;
        loop {
            if (k >= CHUNK) {
                break;
            }
            var wv: array<f32, REG_ROWS>;
            var i1: u32 = 0u;
            loop {
                if (i1 >= REG_ROWS) {
                    break;
                }
                wv[i1] = tile_w[(ty * REG_ROWS + i1) * CHUNK + k];
                i1 = i1 + 1u;
            }
            var xv: array<f32, REG_TOKENS>;
            var j1: u32 = 0u;
            loop {
                if (j1 >= REG_TOKENS) {
                    break;
                }
                xv[j1] = tile_x[(tx * REG_TOKENS + j1) * CHUNK + k];
                j1 = j1 + 1u;
            }
            i1 = 0u;
            loop {
                if (i1 >= REG_ROWS) {
                    break;
                }
                var j2: u32 = 0u;
                loop {
                    if (j2 >= REG_TOKENS) {
                        break;
                    }
                    acc[i1 * REG_TOKENS + j2] = acc[i1 * REG_TOKENS + j2] + wv[i1] * xv[j2];
                    j2 = j2 + 1u;
                }
                i1 = i1 + 1u;
            }
            k = k + 1u;
        }

        workgroupBarrier();
        chunk_start = chunk_start + CHUNK;
    }

    var i1: u32 = 0u;
    loop {
        if (i1 >= REG_ROWS) {
            break;
        }
        let row = row_start + ty * REG_ROWS + i1;
        if (row < params.out_dim) {
            var j1: u32 = 0u;
            loop {
                if (j1 >= REG_TOKENS) {
                    break;
                }
                let token = token_start + tx * REG_TOKENS + j1;
                if (token < params.n_tokens) {
                    y[token * params.out_dim + row] = acc[i1 * REG_TOKENS + j1];
                }
                j1 = j1 + 1u;
            }
        }
        i1 = i1 + 1u;
    }
}
"#;

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

/// The complete, compiled-ready WGSL source for `ggml_type`'s *reduction*
/// pipeline (see `MAIN_REDUCE_SUFFIX`), or `None` if this backend has no
/// shader for it (the same set `engine::quant` supports on the CPU path —
/// see its module doc for what's missing). Reuses the same `*_COOP_MIDDLE`
/// constant `shader_source_coop` does — both dispatch strategies share the
/// exact same `dequant_element` per type, only `MAIN_REDUCE_SUFFIX` vs.
/// `MAIN_COOP_SUFFIX` (and so the resulting compute `main`) differs.
pub fn shader_source_reduce(ggml_type: u32, n_rows: usize, subgroup: bool) -> Option<String> {
    let middle = match ggml_type {
        t if t == GGML_TYPE_F32 => F32_COOP_MIDDLE,
        t if t == GGML_TYPE_F16 => F16_COOP_MIDDLE,
        t if t == GGML_TYPE_BF16 => BF16_COOP_MIDDLE,
        t if t == GGML_TYPE_Q4_0 => Q4_0_COOP_MIDDLE,
        t if t == GGML_TYPE_Q5_0 => Q5_0_COOP_MIDDLE,
        t if t == GGML_TYPE_Q8_0 => Q8_0_COOP_MIDDLE,
        t if t == GGML_TYPE_Q4_K => Q4_K_COOP_MIDDLE,
        t if t == GGML_TYPE_Q5_K => Q5_K_COOP_MIDDLE,
        t if t == GGML_TYPE_Q6_K => Q6_K_COOP_MIDDLE,
        _ => return None,
    };
    let suffix = main_reduce_suffix(n_rows, subgroup);
    Some(format!("{PRELUDE}\n{middle}\n{suffix}"))
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
/// weight streaming (`SERVER_ROADMAP.md` Step 2). The two-wave
/// `block_dot` above splits a 64-thread workgroup into two wave32s that
/// *each* load the whole 144-byte block — one taking low nibbles
/// (`is_low`), one high (`local - 32`) — so every weight byte is fetched
/// twice. Here one **32-thread** workgroup owns a whole super-block: lane
/// `local` (0..31) loads the four qs bytes at in-group position `local` of
/// the four 64-groups and, per group, emits *both* the low-nibble element
/// (position `g*64 + local`, activation `xl_g`) and the high-nibble element
/// (position `g*64 + 32 + local`, activation `xh_g`), reusing the identical
/// `q4k_elem`/`qs_byte_q4k` math so the value stays within the same
/// float-reorder tolerance the existing kernel variants already have vs.
/// each other and `CpuBackend` (one lane now sums a low+high pair the two
/// waves previously summed separately, so the add order differs — not
/// bit-identical, but within `matmul_matches_cpu_backend_for_q4_k`'s
/// tolerance). 32 threads == one wave32 subgroup on this hardware, so the
/// reduction is a single barrier-free `subgroupAdd` when `subgroup` is set
/// (else a 32-wide barrier tree). `n_rows` output rows share the workgroup
/// and its hoisted activations, exactly as `unroll_suffix` does.
pub fn shader_source_reduce_q4k_dual_nibble(n_rows: usize, subgroup: bool) -> String {
    let suffix = dual_nibble_suffix(n_rows, subgroup);
    format!("{PRELUDE_VEC4}\n{Q4K_DUAL_MIDDLE}\n{suffix}")
}

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
/// (one-wave32-subgroup) analogue of [`unroll_suffix`]. Same
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
    let middle = match ggml_type {
        t if t == GGML_TYPE_F32 => F32_COOP_MIDDLE,
        t if t == GGML_TYPE_F16 => F16_COOP_MIDDLE,
        t if t == GGML_TYPE_BF16 => BF16_COOP_MIDDLE,
        t if t == GGML_TYPE_Q4_0 => Q4_0_COOP_MIDDLE,
        t if t == GGML_TYPE_Q5_0 => Q5_0_COOP_MIDDLE,
        t if t == GGML_TYPE_Q8_0 => Q8_0_COOP_MIDDLE,
        t if t == GGML_TYPE_Q4_K => Q4_K_COOP_MIDDLE,
        t if t == GGML_TYPE_Q5_K => Q5_K_COOP_MIDDLE,
        t if t == GGML_TYPE_Q6_K => Q6_K_COOP_MIDDLE,
        _ => return None,
    };
    Some(format!("{PRELUDE}\n{middle}\n{MAIN_COOP_SUFFIX}"))
}

/// The default (opt out with `ORANGU_NO_TILED_PREFILL=1`) tiled-GEMM
/// alternative to [`shader_source_coop`] — see `MAIN_COOP_TILED_SUFFIX`'s
/// own doc comment for the design, and `MAIN_COOP_SUFFIX`'s for why this
/// is the default now.
pub fn shader_source_coop_tiled(ggml_type: u32) -> Option<String> {
    let middle = match ggml_type {
        t if t == GGML_TYPE_F32 => F32_COOP_MIDDLE,
        t if t == GGML_TYPE_F16 => F16_COOP_MIDDLE,
        t if t == GGML_TYPE_BF16 => BF16_COOP_MIDDLE,
        t if t == GGML_TYPE_Q4_0 => Q4_0_COOP_MIDDLE,
        t if t == GGML_TYPE_Q5_0 => Q5_0_COOP_MIDDLE,
        t if t == GGML_TYPE_Q8_0 => Q8_0_COOP_MIDDLE,
        t if t == GGML_TYPE_Q4_K => Q4_K_COOP_MIDDLE,
        t if t == GGML_TYPE_Q5_K => Q5_K_COOP_MIDDLE,
        t if t == GGML_TYPE_Q6_K => Q6_K_COOP_MIDDLE,
        _ => return None,
    };
    let suffix = MAIN_COOP_TILED_SUFFIX
        .replace("%TILE_ROWS%", &COOP_TILE_ROWS.to_string())
        .replace("%TILE_TOKENS%", &COOP_TILE_TOKENS.to_string())
        .replace("%CHUNK%", &COOP_CHUNK.to_string());
    Some(format!("{PRELUDE}\n{middle}\n{suffix}"))
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
const ELEM_META: &str = r#"
struct ElemMeta {
    len: u32,
    _pad0: u32,
    extra: f32,
    _pad1: u32,
}
"#;

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

/// Line-for-line port of `engine::tensor::gelu` (the tanh approximation,
/// not the exact erf form).
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
    y[i] = 0.5 * v * (1.0 + tanh(sqrt_2_over_pi * v * (1.0 + coef_a * v * v)));
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
const RMSNORM_SHADER_BODY: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> em: ElemMeta;

var<workgroup> partial_sums: array<f32, 64>;

@compute @workgroup_size(64)
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
        k = k + 64u;
    }
}
"#;

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

pub fn shader_source_scale() -> String {
    format!("{ELEM_META}\n{SCALE_SHADER_BODY}")
}

pub fn shader_source_rmsnorm(subgroup: bool) -> String {
    let body = if subgroup {
        RMSNORM_SHADER_BODY_SUBGROUP
    } else {
        RMSNORM_SHADER_BODY
    };
    format!("{ELEM_META}\n{body}")
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
/// not attempted, see `doc/SERVER_ROADMAP.md`'s Priority 1 item 1 for why.
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
const RMSNORM_ADD_SHADER_BODY: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> residual: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> em: ElemMeta;

var<workgroup> partial_sums: array<f32, 64>;

@compute @workgroup_size(64)
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
        k = k + 64u;
    }
}
"#;

/// See `RMSNORM_ADD_SHADER_BODY_SUBGROUP`'s own doc comment — RMSNorm
/// fused with the residual add that already always immediately follows it
/// at both of this codebase's two call sites (`wo`'s and `ffn_down`'s own
/// post-matmul norm+add, `VulkanBackend::build_fused_resources`), removing
/// one dispatch (`add_pipeline`'s own) from each.
pub fn shader_source_rmsnorm_add(subgroup: bool) -> String {
    let body = if subgroup {
        RMSNORM_ADD_SHADER_BODY_SUBGROUP
    } else {
        RMSNORM_ADD_SHADER_BODY
    };
    format!("{ELEM_META}\n{body}")
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
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> aq: array<f32>;
%KV_BINDINGS%
@group(0) @binding(3) var<storage, read_write> probs_scratch: array<f32>;
@group(0) @binding(4) var<storage, read_write> aout: array<f32>;
@group(0) @binding(5) var<uniform> am: AttnMeta;

%KV_READ_FNS%

// Generous upper bound on `head_dim` — `E2B`'s real full-attention layers
// use 512; this leaves headroom for other models without costing more
// than a few KB of workgroup-shared memory (`MAX_HEAD_DIM * 4` bytes).
const MAX_HEAD_DIM: u32 = 2048u;

var<workgroup> shared_reduce: array<f32, 64>;
var<workgroup> tile_probs: array<f32, 64>;
var<workgroup> acc: array<f32, MAX_HEAD_DIM>;

fn score_at(h: u32, kv_head: u32, p: u32) -> f32 {
    let head_dim = am.head_dim;
    let q_base = h * head_dim;
    let k_base = (p * am.n_head_kv + kv_head) * head_dim;
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
    let n_pos = am.n_pos;
    let head_dim = am.head_dim;

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
                let v_base = (vp * am.n_head_kv + kv_head) * head_dim;
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
        aout[h * head_dim + d3] = acc[d3] / l;
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
pub fn shader_source_attention(kv_storage: KvStorage, subgroup: bool) -> String {
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
    ATTENTION_SHADER_TEMPLATE
        .replace("%KV_ENABLE%", kv_storage.enable_directive())
        .replace("%KV_BINDINGS%", &kv_bindings)
        .replace("%KV_READ_FNS%", &kv_read_fns)
        .replace("%SUBGROUP_PARAMS%", subgroup_params)
        .replace("%MAX_REDUCE_BLOCK%", max_block)
        .replace("%SUM_REDUCE_BLOCK%", sum_block)
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
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> aq: array<f32>;
%KV_BINDINGS%
@group(0) @binding(3) var<storage, read_write> partial_ml: array<f32>;
@group(0) @binding(4) var<storage, read_write> partial_acc: array<f32>;
@group(0) @binding(5) var<uniform> am: AttnSplitMeta;

%KV_READ_FNS%

const MAX_HEAD_DIM: u32 = 2048u;

var<workgroup> shared_reduce: array<f32, 64>;
var<workgroup> tile_probs: array<f32, 64>;
var<workgroup> acc: array<f32, MAX_HEAD_DIM>;

fn score_at(h: u32, kv_head: u32, p: u32) -> f32 {
    let head_dim = am.head_dim;
    let q_base = h * head_dim;
    let k_base = (p * am.n_head_kv + kv_head) * head_dim;
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
                    let v_base = (vp * am.n_head_kv + kv_head) * head_dim;
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

pub fn shader_source_attention_split(kv_storage: KvStorage, subgroup: bool) -> String {
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
    ATTENTION_SPLIT_SHADER_TEMPLATE
        .replace("%KV_ENABLE%", kv_storage.enable_directive())
        .replace("%KV_BINDINGS%", &kv_bindings)
        .replace("%KV_READ_FNS%", &kv_read_fns)
        .replace("%SUBGROUP_PARAMS%", subgroup_params)
        .replace("%MAX_REDUCE_BLOCK%", max_block)
        .replace("%SUM_REDUCE_BLOCK%", sum_block)
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

pub fn shader_source_attention_split_reduce() -> String {
    ATTENTION_SPLIT_REDUCE_SHADER.to_string()
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
    _pad1: u32,
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
    cdst[cm.offset + idx] = f16(csrc[idx]);
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
    _pad1: u32,
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
    let src_base = block * 32u;
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

/// Line-for-line port of `engine::tensor::rope_apply_scaled_inplace`
/// (NEOX-style pairing: element `i` pairs with `i + rope_dim/2`, only the
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
const ROPE_SHADER: &str = r#"
struct RopeMeta {
    n_head: u32,
    head_dim: u32,
    rope_dim: u32,
    pos: u32,
    freq_base: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> rff: array<f32>;
@group(0) @binding(1) var<storage, read_write> rx: array<f32>;
@group(0) @binding(2) var<uniform> rm: RopeMeta;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half = rm.rope_dim / 2u;
    let total = rm.n_head * half;
    let idx = gid.x;
    if (idx >= total) {
        return;
    }
    let h = idx / half;
    let i = idx % half;
    let base = h * rm.head_dim;
    let freq = pow(rm.freq_base, -2.0 * f32(i) / f32(rm.rope_dim)) / rff[i];
    let theta = f32(rm.pos) * freq;
    let s = sin(theta);
    let c = cos(theta);
    let a = rx[base + i];
    let b = rx[base + i + half];
    rx[base + i] = a * c - b * s;
    rx[base + i + half] = a * s + b * c;
}
"#;

pub fn shader_source_rope() -> String {
    ROPE_SHADER.to_string()
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
    _pad0: u32,
    _pad1: u32,
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
    let local = lid.x;
    let base = h * fnm.head_dim;

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
        let theta = f32(fnm.pos) * freq;
        let s = sin(theta);
        let c = cos(theta);
        let a = fn_head[k];
        let b = fn_head[k + half];
        fn_head[k] = a * c - b * s;
        fn_head[k + half] = a * s + b * c;
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
    FUSED_NORM_ROPE_SHADER.to_string()
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
const ARGMAX_PENALTY_SHADER: &str = r#"
struct SampleMeta {
    n_vocab: u32,
    n_recent: u32,
    repeat_penalty: f32,
    _pad: u32,
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

pub fn shader_source_argmax_penalty() -> String {
    ARGMAX_PENALTY_SHADER.to_string()
}

pub fn shader_source_argmax_split() -> String {
    ARGMAX_SPLIT_SHADER.to_string()
}

pub fn shader_source_argmax_reduce() -> String {
    format!("{ELEM_META}\n{ARGMAX_REDUCE_SHADER_BODY}")
}
