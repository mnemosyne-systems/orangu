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
//! functions (see each `*_COOP_MIDDLE`'s doc comment for why a *third*,
//! now-removed strategy — one thread computing a whole `(row, token)`
//! sequentially — was replaced):
//!
//! - **`MAIN_REDUCE_SUFFIX`** (small `n_tokens`, e.g. decode's `n_tokens ==
//!   1`, `VulkanBackend::COOP_MIN_N_TOKENS`): one workgroup per `(row,
//!   token)` pair, all 64 threads splitting that *row's own elements*
//!   (`k`, `k+64`, `k+128`, ...) and reducing their partial dot-product
//!   sums together. Replaced an earlier one-thread-per-row design once
//!   real measurements against `unsloth/gemma-4-E2B`'s actual shapes (its
//!   262144-entry vocabulary in particular — a single lm_head call cost
//!   ~138ms) showed that design's memory access pattern was badly
//!   uncoalesced: adjacent threads in the same wavefront, each owning a
//!   *different row*, read memory `row_bytes` (hundreds of bytes) apart at
//!   every step, instead of the current design's adjacent threads reading
//!   *adjacent elements of the same row*.
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

/// The compute entry point for the *reduction* path (small `n_tokens`,
/// e.g. decode's `n_tokens == 1` — see `VulkanBackend::COOP_MIN_N_TOKENS`
/// for the crossover into `MAIN_COOP_SUFFIX` instead). One workgroup per
/// `(output row *group* of `REDUCE_N_ROWS` rows, token)` pair, not one row:
/// all 64 threads divide up `in_dim` elements the same grid-stride way a
/// single-row design would (`k = local, local + 64, local + 128, ...`), but
/// at each `k` read `x[x_base + k]` *once* and reuse it across all
/// `REDUCE_N_ROWS` rows' dot products — `doc/SERVER_OPUS.md` Section 9's
/// Step 7's "multiple output rows per thread." A standard workgroup tree
/// reduction (`partial_sums`, now `REDUCE_N_ROWS` independent reductions
/// packed into one flat array, `partial_sums[row * 64 + lane]`) combines
/// each row's 64 partial sums into that row's final output, same as
/// before, just `REDUCE_N_ROWS` times per workgroup instead of once.
///
/// This still fixes the memory access pattern a one-thread-per-row design
/// has (adjacent threads read adjacent elements of the *same* row at every
/// step — see the single-row design's own history in this module's
/// top-of-file doc comment) and additionally: cuts the workgroup dispatch
/// count `REDUCE_N_ROWS`-fold (`VulkanBackend::build_op_resources` computes
/// `ceil(out_dim / REDUCE_N_ROWS) * n_tokens` workgroups now, not
/// `out_dim * n_tokens` — a real reduction for a wide `lm_head`'s
/// 262144-row output), amortizes each `workgroupBarrier` round over
/// `REDUCE_N_ROWS` rows' worth of useful reduction instead of one, and
/// reads each `x[k]` once per workgroup instead of once per row.
/// `REDUCE_N_ROWS` rows are handled with plain unrolled indices (0..4),
/// not a runtime loop over a dynamically-sized array, since it's a fixed
/// compile-time constant matching `VulkanBackend::REDUCE_N_ROWS` exactly —
/// see that constant's own doc comment for why the two must stay in sync.
/// The last group in a row a `REDUCE_N_ROWS`-imperfect `out_dim` (e.g.
/// `out_dim = 6` needs 2 groups of 4, the second only half full) simply
/// skips the out-of-range rows via `o < params.out_dim` bounds checks —
/// their `partial_sums` entries are computed as `0.0` and never written to
/// `y`, not read back by anything.
const MAIN_REDUCE_SUFFIX: &str = r#"
var<workgroup> partial_sums: array<f32, 256>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let n_row_groups = (params.out_dim + 3u) / 4u;
    let flat = wid.x + wid.y * nwg.x + wid.z * nwg.x * nwg.y;
    if (flat >= n_row_groups * params.n_tokens) {
        return;
    }
    let rg = flat / params.n_tokens;
    let t = flat % params.n_tokens;
    let o_base = rg * 4u;
    let o0 = o_base;
    let o1 = o_base + 1u;
    let o2 = o_base + 2u;
    let o3 = o_base + 3u;
    let local = lid.x;
    let x_base = t * params.in_dim;

    var partial0: f32 = 0.0;
    var partial1: f32 = 0.0;
    var partial2: f32 = 0.0;
    var partial3: f32 = 0.0;
    var k: u32 = local;
    loop {
        if (k >= params.in_dim) {
            break;
        }
        let block_idx = k / BLOCK_ELEMS;
        let local_k = k % BLOCK_ELEMS;
        let block_off = block_idx * BLOCK_BYTES;
        let xv = x[x_base + k];
        partial0 = partial0 + dequant_element(o0 * params.row_bytes + block_off, local_k) * xv;
        if (o1 < params.out_dim) {
            partial1 = partial1 + dequant_element(o1 * params.row_bytes + block_off, local_k) * xv;
        }
        if (o2 < params.out_dim) {
            partial2 = partial2 + dequant_element(o2 * params.row_bytes + block_off, local_k) * xv;
        }
        if (o3 < params.out_dim) {
            partial3 = partial3 + dequant_element(o3 * params.row_bytes + block_off, local_k) * xv;
        }
        k = k + 64u;
    }

    partial_sums[local] = partial0;
    partial_sums[64u + local] = partial1;
    partial_sums[128u + local] = partial2;
    partial_sums[192u + local] = partial3;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (local < stride) {
            partial_sums[local] = partial_sums[local] + partial_sums[local + stride];
            partial_sums[64u + local] = partial_sums[64u + local] + partial_sums[64u + local + stride];
            partial_sums[128u + local] = partial_sums[128u + local] + partial_sums[128u + local + stride];
            partial_sums[192u + local] = partial_sums[192u + local] + partial_sums[192u + local + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (local == 0u) {
        y[t * params.out_dim + o0] = partial_sums[0];
        if (o1 < params.out_dim) {
            y[t * params.out_dim + o1] = partial_sums[64u];
        }
        if (o2 < params.out_dim) {
            y[t * params.out_dim + o2] = partial_sums[128u];
        }
        if (o3 < params.out_dim) {
            y[t * params.out_dim + o3] = partial_sums[192u];
        }
    }
}
"#;

/// The compute entry point for the *cooperative* path — used instead of
/// `MAIN_REDUCE_SUFFIX` when `n_tokens` is large enough (see `VulkanBackend`'s
/// dispatch selection) that many tokens genuinely share the same weight
/// row's blocks. One workgroup per output row (not per `(row, token)`
/// pair): every thread cooperatively dequantizes its own slice of each
/// block into `shared_vals` (`var<workgroup>`, real fast on-chip shared
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
/// activation matrix from global memory — which is exactly what `Self::
/// shader_source_coop_tiled` (`doc/SERVER_OPUS.md` Section 9's Step 10,
/// `ORANGU_TILED_PREFILL=1`) fixes. That kernel is correctness-verified
/// (`VulkanBackend`'s `matmul_matches_cpu_backend_cooperative_path_*`
/// tests, run with the env var set) but its real end-to-end prefill
/// throughput is **unmeasured** — this project's own non-negotiable
/// verification discipline requires a real, back-to-back measurement on
/// the actual model before any default-affecting change ships, and that
/// measurement couldn't be safely completed this session: sending long
/// prompts (~1500+ tokens) through *either* kernel hit real `amdgpu`
/// ring-timeout/GPU-reset events on this laptop's shared GPU (also used
/// by the live desktop compositor) — a pre-existing hardware/driver
/// limit, confirmed to affect this unchanged kernel too, not something
/// Step 10 introduced, but one that made further large-prompt A/B
/// testing unsafe to keep pursuing. This kernel stays the default;
/// `shader_source_coop_tiled` ships opt-in, pending a real measurement
/// once this can be tested without risking the live desktop session.
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
/// SUFFIX`'s prefill GEMM (`doc/SERVER_OPUS.md` Section 9's Step 10,
/// `ORANGU_TILED_PREFILL=1`) — templated into the WGSL text (`%TILE_ROWS%`/
/// `%TILE_TOKENS%`/`%CHUNK%`, `shader_source_coop_tiled`) rather than
/// duplicated as separate literals in the shader and in `VulkanBackend::
/// build_op_resources`'s dispatch-count math. `REDUCE_N_ROWS` (Step 7) *is*
/// duplicated that way — a hand-kept-in-sync literal in both places — and
/// that exact drift is what caused Step 8's dispatch-count bug (`doc/
/// SERVER_ROADMAP.md`'s Step 8 entry): one formula got updated for a new
/// kernel, the other didn't. `VulkanBackend` imports these same three
/// constants for its own dispatch math instead of re-declaring the numbers,
/// closing off that failure mode structurally rather than just documenting
/// it. `TILE_TOKENS` (64) matches the old cooperative kernel's own implicit
/// token-tile size (it looped 64 tokens at a time per weight-block
/// dequant), so weight-dequant reuse is unchanged from before; `TILE_ROWS`
/// (16) is new — the old kernel never reused activations across output
/// rows at all (one workgroup per row, so every row's workgroup re-read
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
/// why this ships opt-in (`ORANGU_TILED_PREFILL=1`) rather than as the
/// default despite being correctness-verified.
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
pub fn shader_source_reduce(ggml_type: u32) -> Option<String> {
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
    Some(format!("{PRELUDE}\n{middle}\n{MAIN_REDUCE_SUFFIX}"))
}

/// `doc/SERVER_OPUS.md` Section 9's Step 8, Variant A, `Q4_K` only (`E2B`'s
/// default weight type — the doc's own "start with one type" guidance).
/// Dequantizes weight elements *in pairs* (`dequant_pair_f16`, a `Q4_K`-
/// specific restatement of `Q4_K_COOP_MIDDLE`'s `dequant_element` that
/// also skips the redundant `get_scale_min_k4` lookup a pair's two
/// elements would otherwise repeat) and accumulates the dot product as
/// packed `vec2<f16>` instead of two scalar `f32` multiplies — half as
/// many multiply-accumulate ops in the inner loop, gated behind
/// `VulkanBackend::packed_dot_f16` (`ORANGU_PACKED_DOT=1`) since, like
/// Step 6/7, this needs a real end-to-end measurement before it can be
/// trusted as a win, not just a plausible-sounding one. Not reused by
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
    // module (a WGSL rule, not just naga pedantry) — `PRELUDE` already has
    // `struct Meta`/global `var<...>` declarations, so this can't sit
    // inside `MIDDLE` the way the rest of `MIDDLE` conceptually belongs
    // there; it has to lead the concatenated string instead.
    format!("enable f16;\n{PRELUDE}\n{MIDDLE}\n{SUFFIX}")
}

/// `doc/SERVER_OPUS.md`'s Step 12 (wide vectorized weight loads). Unlike
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
/// boundary — and get the biggest win: their whole `d`/`dmin`/`scales`
/// header (16 bytes) loads in one `vec4` read instead of up to 9 `read_u8`
/// calls. The other 7 types' blocks aren't 16-byte multiples (`Q6_K`'s 210
/// in particular), so their block starts land at unpredictable, *varying*
/// alignment from one block to the next — `read_word_v4`/
/// `read_word_unaligned_v4`/`read_byte_v4` below handle any `byte_offset`
/// correctly regardless, and each type still consolidates whatever of its
/// own fields are provably word-safe to combine (worked out per type,
/// see each `*_WIDE_MIDDLE` constant's own comment) — smaller than the
/// aligned types' win, but real.
///
/// Gated behind `VulkanBackend::wide_load` (`ORANGU_WIDE_LOAD=1`), same
/// off-by-default discipline as every other unproven-until-measured kernel
/// in this file (Steps 6/8/10/11's GPU sampling and batching) — see that
/// field's own doc comment for why.
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
// (or 0..3 for `vec3_word`), so the branch is cheap, and it sidesteps
// relying on naga lowering `OpVectorExtractDynamic` correctly on a WGSL
// corner this project hasn't exercised before (see this file's own
// top-of-file history of real naga gaps — subgroups, still blocked).
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
/// byte-wise kernel's 4 separate `read_u8` calls: a clean 4x reduction in
/// memory-access instruction count for this type specifically.
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
/// for `d`. `qs` (the actual nibbles) stays per-byte (`read_byte_v4`,
/// still a real win over `array<u32>`-bound `read_u8`, just not further
/// consolidated) — mirrors `Q4_0_COOP_MIDDLE`'s math exactly otherwise.
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
/// needs) — this is where this type's real, measured ~17% win (see
/// `doc/SERVER_OPUS.md`'s Step 12 entry) comes from. `qs` (the 128-byte
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
/// reduce pipeline (`doc/SERVER_OPUS.md`'s Step 12), or `None` if this
/// backend has no wide-load kernel for it — same type coverage as
/// [`shader_source_reduce`]. Reuses `MAIN_REDUCE_SUFFIX` verbatim (see
/// `PRELUDE_VEC4`'s own doc comment for why every `*_WIDE_MIDDLE`'s
/// `dequant_element` keeps the same signature that requires).
pub fn shader_source_reduce_wide_load(ggml_type: u32) -> Option<String> {
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
    Some(format!("{PRELUDE_VEC4}\n{middle}\n{MAIN_REDUCE_SUFFIX}"))
}

/// `doc/SERVER_OPUS.md`'s Step 12, item 5: wide loads (this file's
/// `PRELUDE_VEC4`/`Q4_K_WIDE_MIDDLE`) combined with Step 8's packed-`f16`
/// pairwise dot (`shader_source_reduce_q4k_packed_f16`'s own `dequant_
/// pair_f16`) — the two are complementary (one fixes memory bandwidth, one
/// fixes ALU throughput), the pair the doc's original ~1.8–2.8× projection
/// was actually about. `Q4_K`-only, like Step 8 itself (no other type has
/// a packed-`f16` kernel to combine with). `dequant_pair_f16` below is a
/// direct transcription of the byte-wise kernel's own — same "`k` must be
/// even, `k`/`k+1` always share one nibble half" invariant, same math —
/// just sourcing `d`/`dmin`/`scales` from one `vec4` header load (`Q4_K`'s
/// block is always vec4-aligned, see `Q4_K_WIDE_MIDDLE`'s own doc comment)
/// and `qs` bytes via vec4-based extraction instead of `read_u8`. Dispatch
/// (`SUFFIX` below) mirrors the packed-`f16` kernel's own one-row-per-
/// workgroup shape exactly, just walking `vec4_base` instead of
/// `byte_offset` — deliberately *not* attempting `REDUCE_N_ROWS` batching
/// on top of this (a 4-row-batched *and* pair-packed kernel is a much
/// bigger, more error-prone rewrite for an increment not yet shown to be
/// worth it — see `doc/SERVER_OPUS.md`'s Step 12 entry for why `REDUCE_
/// N_ROWS` batching alone turned out close to a wash for wide loads, which
/// argues against assuming it would help here either without measuring).
///
/// **Correctness-verified, but a real, measured regression relative to
/// either technique alone** — see `VulkanBackend::wide_packed_pipeline`'s
/// own doc comment for the numbers and the likely (not chased down
/// further) cause. Not recommended; kept available (like `kv_f16`/
/// `gpu_sample`) as an honestly-reported negative result, not deleted.
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

/// The opt-in (`ORANGU_TILED_PREFILL=1`) tiled-GEMM alternative to
/// [`shader_source_coop`] — see `MAIN_COOP_TILED_SUFFIX`'s own doc comment
/// for the design, and `MAIN_COOP_SUFFIX`'s for why this isn't the default
/// yet despite being correctness-verified (`doc/SERVER_OPUS.md` Section 9's
/// Step 10).
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

pub fn shader_source_rmsnorm() -> String {
    format!("{ELEM_META}\n{RMSNORM_SHADER_BODY}")
}

/// GPU-resident causal attention for a *single* query token (decode,
/// `n_tokens == 1`) against a GPU-resident KV cache — one workgroup per
/// query head, 64 threads. `doc/SERVER_OPUS.md` Section 9's Step 9:
/// online-softmax, **tiled** over the KV sequence in chunks of 64
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
/// follow-up, not required for this step's win). Barrier count is
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
@group(0) @binding(1) var<storage, read> k_cache: array<%KV_TYPE%>;
@group(0) @binding(2) var<storage, read> v_cache: array<%KV_TYPE%>;
@group(0) @binding(3) var<storage, read_write> probs_scratch: array<f32>;
@group(0) @binding(4) var<storage, read_write> aout: array<f32>;
@group(0) @binding(5) var<uniform> am: AttnMeta;

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
        s = s + aq[q_base + d] * f32(k_cache[k_base + d]);
        d = d + 1u;
    }
    return s * am.scale;
}

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
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

        var my_prob: f32 = 0.0;
        if (has_pos) {
            my_prob = exp(my_score - tile_max);
        }
        tile_probs[local] = my_prob;
        shared_reduce[local] = my_prob;
        workgroupBarrier();
        stride = 32u;
        loop {
            if (stride == 0u) {
                break;
            }
            if (local < stride) {
                shared_reduce[local] = shared_reduce[local] + shared_reduce[local + stride];
            }
            workgroupBarrier();
            stride = stride / 2u;
        }
        let tile_sum = shared_reduce[0];
        workgroupBarrier();

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
                tile_contribution = tile_contribution + tile_probs[j] * f32(v_cache[v_base + d2]);
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
/// `f16` — `doc/SERVER_OPUS.md` Section 9's Step 6) or `array<f32>` (the
/// original, always-available path). Every read of either array already
/// goes through an `f32(...)` widening cast (a no-op when the array is
/// already `f32`), so the score/softmax/weighted-sum math itself is
/// identical either way — only the storage type, and hence the KV
/// mirror's memory traffic, changes.
pub fn shader_source_attention(kv_f16: bool) -> String {
    ATTENTION_SHADER_TEMPLATE
        .replace("%KV_ENABLE%", if kv_f16 { "enable f16;" } else { "" })
        .replace("%KV_TYPE%", if kv_f16 { "f16" } else { "f32" })
}

/// Casts `cm.len` elements of a freshly RoPE'd/normed `f32` key or value
/// row (`csrc`) into the `f16`-stored KV mirror (`cdst`) at element offset
/// `cm.offset` — `doc/SERVER_OPUS.md` Section 9's Step 6, only ever built
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

pub fn shader_source_perhead_rmsnorm() -> String {
    PERHEAD_RMSNORM_SHADER.to_string()
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

pub fn shader_source_perhead_rmsnorm_weightless() -> String {
    PERHEAD_RMSNORM_WEIGHTLESS_SHADER.to_string()
}

/// `doc/SERVER_OPUS.md` Section 9's Step 11's GPU-sampling follow-up:
/// greedy (argmax) decode with repeat penalty, entirely on-GPU, so a
/// decode step that's going to sample greedily anyway never has to read
/// back the full `[n_vocab]` logits vector — just the one winning token
/// id (4 bytes instead of, for `E2B`'s 262144-entry vocabulary, ~1 MB).
///
/// Two phases, one workgroup, 64 threads:
/// 1. **Repeat penalty**, thread 0 only, strictly sequential over
///    `recent_tokens` in order — mirrors `engine::sampling::
///    apply_repeat_penalty`'s own loop exactly, including its behavior on
///    a repeated token id (penalized once per occurrence, compounding,
///    since each iteration reads the *already-penalized* value the
///    previous iteration just wrote). This can't be parallelized without
///    changing that compounding behavior, but `recent_tokens` is tiny
///    (`repeat_last_n`, 64 by default) next to `n_vocab`, so a single
///    thread doing it sequentially before the parallel phase starts costs
///    nothing worth optimizing.
/// 2. **Argmax reduction** over the (now-penalized) logits: each thread
///    grid-strides its own `n_vocab / 64` share finding its own best
///    `(value, index)` pair, then a standard workgroup tree reduction
///    combines the 64 partial results into one. Ties are resolved
///    arbitrarily (whichever candidate a given comparison happens to keep)
///    rather than matching `engine::sampling`'s CPU `argmax` exactly
///    (`Iterator::max_by`'s "last element wins" rule) — two independently
///    computed `f32` logits landing on the exact same bit pattern doesn't
///    happen with real model output, so this was never worth the extra
///    bookkeeping an index-aware tie-break would need across the
///    grid-strided (non-contiguous) per-thread assignment.
///
/// `logits` is mutated in place by phase 1 (the same buffer `record_full_
/// matmul` just produced) — safe because nothing else reads it afterward
/// in this submission, and the next decode step's own matmul dispatch
/// overwrites the whole buffer again before anything reads it.
const ARGMAX_SAMPLE_SHADER: &str = r#"
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

var<workgroup> best_val: array<f32, 64>;
var<workgroup> best_idx: array<u32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let local = lid.x;

    if (local == 0u) {
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
    workgroupBarrier();

    var my_best_val: f32 = -3.4028235e38;
    var my_best_idx: u32 = 0u;
    var k: u32 = local;
    loop {
        if (k >= sample_meta.n_vocab) {
            break;
        }
        let v = logits[k];
        if (v > my_best_val) {
            my_best_val = v;
            my_best_idx = k;
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

pub fn shader_source_argmax_sample() -> String {
    ARGMAX_SAMPLE_SHADER.to_string()
}
