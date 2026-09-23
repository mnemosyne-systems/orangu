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

//! The always-available backend: dot products parallelized across output
//! rows via `rayon`. Also the fallback when no Vulkan-capable adapter is
//! found.
//!
//! Three paths, picked per weight tensor, in this order:
//!
//! * **Fused** (`engine::vecdot`) — for the quantized types that have a
//!   fused kernel, the weight row's raw quantized bytes are dotted straight
//!   against `int8`-quantized activations, ggml-style. No dequantize, no
//!   per-row `Vec<f32>`.
//! * **Float** (`vecdot::dot_row_f32`) — for `F32`/`F16`/`BF16`, which have
//!   no `int8` weight form to fuse against. Activations stay `f32`; the
//!   weights are widened a block at a time into the dot, so this also has
//!   no dequantize and no per-row `Vec<f32>`.
//! * **Dequantize** — the original path (`QuantMatrix::row` → a fresh
//!   `Vec<f32>` → `engine::tensor::dot` with runtime AVX2 dispatch), now
//!   reached only by types neither of the above claims. It stays as the
//!   full-precision *reference* other backends cross-check against, which is
//!   why its allocation is left alone rather than optimized in place.
//!
//! The fused path exists because a `perf` profile of CPU decode put 95% of
//! all time in this one function — see `engine::vecdot`'s module doc for the
//! measured breakdown.

use rayon::prelude::*;

use crate::engine::loader::QuantMatrix;
use crate::engine::tensor;
use crate::engine::vecdot;

use super::{Backend, MatmulOp, guarded_matmul_op_into};

#[derive(Default)]
pub struct CpuBackend;

/// The most tokens one `matmul_float_into` task takes: 258 pixels of a VAE
/// `im2col` band is under a megabyte of activations, read once per row
/// group. A multiple of the kernel's six-token tile.
const F32_TOKEN_BLOCK: usize = 258;
/// Rows per `matmul_float_into` task: eight quads, whose widened weights
/// (32 × `in_dim` floats — 430 KiB at 3×3×384) fit L2 beside the token
/// block.
const F32_ROW_GROUP: usize = 32;

/// A raw output pointer rayon tasks may share. Sound only under the
/// discipline `matmul_k_gemm_into`'s `i8mm` path keeps: every task writes a
/// column range no other task touches, inside the loop the pointer was
/// taken for.
#[derive(Clone, Copy)]
struct SharedOut(*mut f32);
unsafe impl Send for SharedOut {}
unsafe impl Sync for SharedOut {}

/// `ORANGU_PACKED_GEMM=1` runs the float prefill path through the packed,
/// cache-blocked GEMM in `rten-gemm` instead of this engine's own tiled
/// kernel. **Off by default, because it is not faster here.**
///
/// Measured at every shape a forward pass runs, with normal float values:
/// the two are level — between 0.92x and 1.28x depending on the shape, and
/// the spread between repeats of one shape is wider than the difference
/// between the two kernels at that shape. End to end on a mixture prefill
/// it is worth about 1% at a 624-token prompt and nothing at 2236, where an
/// interleaved A/B's closing control came back at the treatment's rate.
///
/// It is kept, and kept switchable, for two reasons. The packed kernel's
/// advantage is a property of the host: this one has `AVX2` and neither
/// `AVX-512` nor `VNNI`, so the crate selects its weakest x86 kernel, and
/// the comparison is worth repeating where it can select its best. And
/// re-answering the question costs one sweep with this variable rather than
/// a branch.
///
/// It does not apply to quantized weights at all. This engine multiplies
/// the packed bytes directly; a GEMM has to dequantize first, and the
/// dequantize alone costs more than the whole product.
fn packed_gemm_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::engine::env::flag_on("ORANGU_PACKED_GEMM"))
}
impl CpuBackend {
    /// The fused path: quantize each token's activations to `int8` **once**,
    /// then dot them against the still-quantized weight rows. Returns `None`
    /// for any weight type without a fused kernel, so the caller falls back.
    fn matmul_fused_into(
        &self,
        out: &mut Vec<f32>,
        x: &[f32],
        n_tokens: usize,
        w: &QuantMatrix,
    ) -> bool {
        let in_dim = w.in_dim;
        let out_dim = w.out_dim;
        let ggml_type = w.ggml_type();
        if !vecdot::supports(ggml_type, in_dim) {
            return false;
        }
        let raw = w.raw_bytes();
        let row_bytes = w.row_bytes();

        // K-quant prefill takes a dedicated GEMM: `q8_K`-style activations
        // (one scale per 256) accumulated in `i32` across the super-block,
        // laid out interleaved by token tile. Worth +45% to +84% per tensor
        // over the generic path on a `Llama-3.2-1B-Instruct-Q4_K_M`, which is
        // 100% `Q4_K`/`Q6_K`. See `engine::vecdot`'s section comment.
        if n_tokens > 1 && vecdot::supports_k(ggml_type, in_dim) {
            Self::matmul_k_gemm_into(out, x, n_tokens, ggml_type, raw, row_bytes, in_dim, out_dim);
            return true;
        }

        // The symmetric per-32 types take the same flat activation layout,
        // minus the per-256 accumulation their scales cannot support. Worth
        // it because a `Q8_0` prefill profile is 74.9% *loads* — see
        // `engine::vecdot`'s section comment.
        if n_tokens > 1 && vecdot::supports_flat(ggml_type, in_dim) {
            Self::matmul_flat_gemm_into(
                out, x, n_tokens, ggml_type, raw, row_bytes, in_dim, out_dim,
            );
            return true;
        }

        // The two-bit family decodes through a per-super-block activation
        // scale, which lets the whole 256-element accumulation stay in `i32`
        // instead of converting once per 32. Measured at 1.24x (`Q3_K`) and
        // 1.45x (`Q2_K`) in isolation — see `engine::vecdot`'s section comment
        // and `doc/perf/tiny-kernel`.
        //
        // Ahead of the `quantize_act` below because it needs a *different*
        // activation quantization, and doing both would pay for one twice.
        if n_tokens == 1 && vecdot::supports_k_row(ggml_type, in_dim) {
            let act = vecdot::quantize_act_k_row(&x[..in_dim]);
            // One token, so the accumulation is already `[1, out_dim]` —
            // write straight into the caller's buffer, no transpose and no
            // allocation at all.
            out.resize(out_dim, 0.0);
            // Two rows per task where the type has a two-row kernel — the
            // Prism ternary types, whose decode is instruction-bound and
            // shares its activation loads between the pair.
            if vecdot::supports_k_row_pair(ggml_type, in_dim) {
                out.par_chunks_mut(2)
                    .with_min_len(super::matmul_min_rows().div_ceil(2))
                    .enumerate()
                    .for_each(|(pair, dst)| {
                        let o0 = 2 * pair;
                        let row = |o: usize| &raw[o * row_bytes..(o + 1) * row_bytes];
                        if dst.len() == 2 {
                            let (y0, y1) =
                                vecdot::dot_k_row_pair(ggml_type, row(o0), row(o0 + 1), &act);
                            dst[0] = y0;
                            dst[1] = y1;
                        } else {
                            // `out_dim` is odd — the last row alone.
                            dst[0] = vecdot::dot_k_row(ggml_type, row(o0), &act);
                        }
                    });
                return true;
            }
            // A floor on how small a task may get, not a chunk size — see
            // `backend::matmul_min_rows`. One row per task is a few hundred
            // nanoseconds of arithmetic inside a job the pool has to create,
            // steal and join.
            out.par_chunks_mut(1)
                .with_min_len(super::matmul_min_rows())
                .enumerate()
                .for_each(|(o, dst)| {
                    dst[0] = vecdot::dot_k_row(
                        ggml_type,
                        &raw[o * row_bytes..(o + 1) * row_bytes],
                        &act,
                    );
                });
            return true;
        }

        // Once per call, not once per (row, token) — this is the whole point.
        let acts: Vec<vecdot::ActQ8> = (0..n_tokens)
            .map(|t| vecdot::quantize_act(&x[t * in_dim..(t + 1) * in_dim]))
            .collect();

        // Accumulate transposed (`[out_dim, n_tokens]`) so the rayon split is
        // over output rows, which is the only dimension with real parallelism
        // during decode (`n_tokens == 1`). Each output row is written exactly
        // once, so no scatter and no per-row allocation.
        let mut yt = vec![0f32; out_dim * n_tokens];

        // GEMV vs GEMM, the split any BLAS makes — and measured, not assumed:
        // routing decode through the row-at-a-time path below cost 2.05 ->
        // 1.46 tok/s on the reference Pi 4, because materializing a whole
        // unpacked row to memory only pays off once several tokens read it
        // back.
        if n_tokens == 1 {
            let act = &acts[0];
            yt.par_chunks_mut(1)
                .with_min_len(super::matmul_min_rows())
                .enumerate()
                .for_each(|(o, dst)| {
                    dst[0] =
                        vecdot::dot_row(ggml_type, &raw[o * row_bytes..(o + 1) * row_bytes], act);
                });
            // The transpose is the identity — hand the buffer straight back.
            Self::finish_into(out, yt, n_tokens, out_dim);
            return true;
        }

        // Two output rows per chunk: `dot_unpacked_pair` shares each token's
        // activation load between them, which is worth +14-28% over doing one
        // row at a time. A trailing odd row falls back to the single-row form.
        yt.par_chunks_mut(n_tokens * 2).enumerate().for_each_init(
            // Two scratch buffers per rayon worker, reused for every pair it
            // handles — `for_each_init` rather than `for_each` so unpacking
            // doesn't allocate per row.
            || (vecdot::UnpackedRow::new(), vecdot::UnpackedRow::new()),
            |(s0, s1), (pair, dst)| {
                let o0 = pair * 2;
                let row = |o: usize| &raw[o * row_bytes..(o + 1) * row_bytes];
                // Unpack each row **once**, then dot against every token.
                // Before this, a row was re-unpacked per token, which is why
                // prefill lagged decode so badly against llama.cpp.
                vecdot::unpack_row(ggml_type, row(o0), in_dim, s0);
                if dst.len() == n_tokens * 2 {
                    vecdot::unpack_row(ggml_type, row(o0 + 1), in_dim, s1);
                    let (d0, d1) = dst.split_at_mut(n_tokens);
                    vecdot::dot_unpacked_pair(s0, s1, &acts, d0, d1);
                } else {
                    // `out_dim` is odd — this chunk holds the last row alone.
                    vecdot::dot_unpacked_multi(s0, &acts, dst);
                }
            },
        );

        Self::finish_into(out, yt, n_tokens, out_dim);
        true
    }

    /// The K-quant prefill GEMM. Same shape as the generic path below it —
    /// accumulate transposed, two output rows per rayon chunk, one scratch
    /// row per worker — but over [`vecdot::ActQ8K`]/[`vecdot::KRow`].
    #[allow(clippy::too_many_arguments)]
    fn matmul_k_gemm_into(
        out: &mut Vec<f32>,
        x: &[f32],
        n_tokens: usize,
        ggml_type: u32,
        raw: &[u8],
        row_bytes: usize,
        in_dim: usize,
        out_dim: usize,
    ) {
        if vecdot::have_i8mm() {
            let acts = vecdot::ActQ8Mm::quantize(x, in_dim, n_tokens);
            Self::matmul_k_mm_into(
                out, &acts, n_tokens, ggml_type, raw, row_bytes, in_dim, out_dim,
            );
            return;
        }
        let acts = vecdot::ActQ8K::quantize(x, in_dim, n_tokens);
        let mut yt = vec![0f32; out_dim * n_tokens];
        yt.par_chunks_mut(n_tokens * 2).enumerate().for_each_init(
            || (vecdot::KRow::new(), vecdot::KRow::new(), Vec::new()),
            |(s0, s1, scratch), (pair, dst)| {
                let o0 = pair * 2;
                let row = |o: usize| &raw[o * row_bytes..(o + 1) * row_bytes];
                vecdot::unpack_k_row(ggml_type, row(o0), in_dim, s0);
                if dst.len() == n_tokens * 2 {
                    vecdot::unpack_k_row(ggml_type, row(o0 + 1), in_dim, s1);
                    let (d0, d1) = dst.split_at_mut(n_tokens);
                    vecdot::dot_k_pair(s0, s1, &acts, d0, d1);
                } else {
                    // `out_dim` is odd — this chunk holds the last row alone.
                    vecdot::dot_k_multi(s0, &acts, dst, scratch);
                }
            },
        );

        Self::finish_into(out, yt, n_tokens, out_dim);
    }

    /// The K-quant prefill GEMM on the `i8mm` kernel for activations
    /// already quantized — the entry a caller with its own `ActQ8Mm` uses
    /// (the VAE gathers its convolution windows straight into one).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn matmul_k_mm_into(
        out: &mut Vec<f32>,
        acts: &vecdot::ActQ8Mm,
        n_tokens: usize,
        ggml_type: u32,
        raw: &[u8],
        row_bytes: usize,
        in_dim: usize,
        out_dim: usize,
    ) {
        {
            // `k_rows_per_task` rows per task for the `smmla` kernel
            // (`vecdot::dot_k_rows_tiles`), unpacked once each, over a block
            // of token tiles — two dimensions, so a matmul of few rows
            // against many tokens (a VAE convolution: 96 rows, 65,536
            // pixels) is not six tasks on twelve cores. A short trailing
            // row group is padded up to a whole quad with the last row
            // repeated, its extra outputs going to scratch — the kernel's
            // arithmetic is the pair kernel's to the bit, so nothing else
            // changes.
            //
            // Each task accumulates its rows transposed in a small private
            // buffer (16 rows x n_tokens: L1-resident at prefill sizes) and
            // then writes them straight into `out`'s token-major layout —
            // for every token, its 16 rows are one contiguous 64-byte run,
            // one cache line. That is the whole transpose, done in parallel
            // and for free, where `finish_into`'s serial strided pass over
            // a 3 MiB `yt` was a quarter of a 256-token matmul's wall time.
            out.resize(n_tokens * out_dim, 0.0);
            let per_task = vecdot::k_rows_per_task();
            let quad = vecdot::ROW_QUAD;
            let n_groups = out_dim.div_ceil(per_task);
            let n_tiles = acts.n_tiles();
            // Token blocks only when the rows alone cannot occupy the pool:
            // the transformer's 3072-row linears are 192 groups already,
            // and every block re-reads the group's unpacked rows.
            let wanted = 4 * rayon::current_num_threads();
            let n_blocks = if n_groups >= wanted {
                1
            } else {
                wanted.div_ceil(n_groups).min(n_tiles).max(1)
            };
            let tiles_per_block = n_tiles.div_ceil(n_blocks).max(1);
            let n_blocks = n_tiles.div_ceil(tiles_per_block);
            let sink = SharedOut(out.as_mut_ptr());
            (0..n_groups * n_blocks).into_par_iter().for_each_init(
                || {
                    (
                        (0..per_task)
                            .map(|_| vecdot::KRow::new())
                            .collect::<Vec<_>>(),
                        vec![0f32; n_tokens * per_task],
                        vec![0f32; n_tokens * (quad - 1)],
                    )
                },
                move |(rows, yt, scratch), task| {
                    let sink = sink;
                    let (g, b) = (task / n_blocks, task % n_blocks);
                    let o0 = g * per_task;
                    let n_rows = per_task.min(out_dim - o0);
                    let row = |o: usize| &raw[o * row_bytes..(o + 1) * row_bytes];
                    for (i, slot) in rows.iter_mut().enumerate().take(n_rows) {
                        vecdot::unpack_k_row(ggml_type, row(o0 + i), in_dim, slot);
                    }
                    let padded = n_rows.div_ceil(quad) * quad;
                    let refs: Vec<&vecdot::KRow> =
                        (0..padded).map(|i| &rows[i.min(n_rows - 1)]).collect();
                    let mut outs: Vec<&mut [f32]> =
                        yt[..n_tokens * n_rows].chunks_mut(n_tokens).collect();
                    outs.extend(scratch.chunks_mut(n_tokens).take(padded - n_rows));
                    let tiles = b * tiles_per_block..((b + 1) * tiles_per_block).min(n_tiles);
                    let t0 = tiles.start * vecdot::MM_TILE;
                    let t1 = (tiles.end * vecdot::MM_TILE).min(n_tokens);
                    vecdot::dot_k_rows_tiles(&refs, acts, tiles, &mut outs);
                    // Safety: task `task` alone writes columns
                    // `o0..o0 + n_rows` of token rows `t0..t1` of a buffer
                    // sized `n_tokens * out_dim` above; the rectangles of
                    // different tasks are disjoint, and the pointer outlives
                    // the parallel loop it is used in.
                    for t in t0..t1 {
                        for r in 0..n_rows {
                            unsafe {
                                *sink.0.add(t * out_dim + o0 + r) = yt[r * n_tokens + t];
                            }
                        }
                    }
                },
            );
        }
    }

    /// The `Q8_0`/`Q5_0`/`IQ4_NL` prefill GEMM, over [`vecdot::ActQ8Flat`].
    /// Identical in shape to [`Self::matmul_k_gemm`]; they differ only in the
    /// activation type and the kernel called.
    #[allow(clippy::too_many_arguments)]
    fn matmul_flat_gemm_into(
        out: &mut Vec<f32>,
        x: &[f32],
        n_tokens: usize,
        ggml_type: u32,
        raw: &[u8],
        row_bytes: usize,
        in_dim: usize,
        out_dim: usize,
    ) {
        let acts = vecdot::ActQ8Flat::quantize(x, in_dim, n_tokens);
        let mut yt = vec![0f32; out_dim * n_tokens];
        yt.par_chunks_mut(n_tokens * 2).enumerate().for_each_init(
            || {
                (
                    vecdot::UnpackedRow::new(),
                    vecdot::UnpackedRow::new(),
                    Vec::new(),
                )
            },
            |(s0, s1, scratch), (pair, dst)| {
                let o0 = pair * 2;
                let row = |o: usize| &raw[o * row_bytes..(o + 1) * row_bytes];
                vecdot::unpack_row(ggml_type, row(o0), in_dim, s0);
                if dst.len() == n_tokens * 2 {
                    vecdot::unpack_row(ggml_type, row(o0 + 1), in_dim, s1);
                    let (d0, d1) = dst.split_at_mut(n_tokens);
                    vecdot::dot_flat_pair(s0, s1, &acts, d0, d1);
                } else {
                    // `out_dim` is odd — this chunk holds the last row alone.
                    vecdot::dot_flat_multi(s0, &acts, dst, scratch);
                }
            },
        );

        Self::finish_into(out, yt, n_tokens, out_dim);
    }

    /// [`Self::matmul_fused`]'s decode form: the GEMV kernel that function
    /// would pick at `n_tokens == 1` ([`vecdot::dot_k_row`] or
    /// [`vecdot::dot_row`]), run once per token whatever `n_tokens` is.
    ///
    /// A decode batch is `n` sequences each contributing a single token, so
    /// each row's result has to match what that sequence would have got
    /// decoding alone — see [`Backend::matmul_decode`]. Taking the same GEMV
    /// per `(row, token)` pair as the `n_tokens == 1` path makes that equality
    /// bit-for-bit by construction rather than by tolerance, which is also why
    /// the kernel choice below has to track `matmul_fused`'s branch for
    /// branch.
    ///
    /// Only the *unpack* half of the GEMM win is given up, not the whole of
    /// it: the row stays parallelized across workers and each packed row is
    /// still read once for all `n` tokens, close enough together to stay in
    /// cache. What it does not do is materialize the row as `int8` once and
    /// reuse it, which is what the GEMM paths buy and what costs the
    /// bit-exactness.
    fn matmul_fused_decode_into(
        &self,
        out: &mut Vec<f32>,
        x: &[f32],
        n_tokens: usize,
        w: &QuantMatrix,
    ) -> bool {
        let in_dim = w.in_dim;
        let out_dim = w.out_dim;
        let ggml_type = w.ggml_type();
        if !vecdot::supports(ggml_type, in_dim) {
            return false;
        }
        let raw = w.raw_bytes();
        let row_bytes = w.row_bytes();
        let row = |o: usize| &raw[o * row_bytes..(o + 1) * row_bytes];

        // Transposed accumulation and the row-major transpose back, exactly
        // as `matmul_fused` does — see its comments.
        let mut yt = vec![0f32; out_dim * n_tokens];

        // Which GEMV, branch for branch as `matmul_fused` picks it at
        // `n_tokens == 1`. The k-row kernel quantizes activations once per
        // super-block where `dot_row` does it once per 32, so the two do not
        // round alike — taking the wrong one here would break the very
        // equality this function exists to guarantee.
        if vecdot::supports_k_row(ggml_type, in_dim) {
            let acts: Vec<vecdot::ActQ8KRow> = (0..n_tokens)
                .map(|t| vecdot::quantize_act_k_row(&x[t * in_dim..(t + 1) * in_dim]))
                .collect();
            yt.par_chunks_mut(n_tokens)
                .enumerate()
                .for_each(|(o, dst)| {
                    for (slot, act) in dst.iter_mut().zip(&acts) {
                        *slot = vecdot::dot_k_row(ggml_type, row(o), act);
                    }
                });
        } else {
            let acts: Vec<vecdot::ActQ8> = (0..n_tokens)
                .map(|t| vecdot::quantize_act(&x[t * in_dim..(t + 1) * in_dim]))
                .collect();
            yt.par_chunks_mut(n_tokens)
                .enumerate()
                .for_each(|(o, dst)| {
                    for (slot, act) in dst.iter_mut().zip(&acts) {
                        *slot = vecdot::dot_row(ggml_type, row(o), act);
                    }
                });
        }

        Self::finish_into(out, yt, n_tokens, out_dim);
        true
    }

    /// One `[n_tokens, in_dim] x [in_dim, out_dim]` product through a
    /// packed, cache-blocked GEMM, writing token-major into `out`.
    ///
    /// `rows` is the weight as this engine stores it — `out_dim` rows of
    /// `in_dim` floats — which is the transpose of the `[k, n]` a GEMM
    /// wants. It is handed over as a transposed *view*: the packing step
    /// reads a strided operand at the same rate as a contiguous one
    /// (measured within 6%), so no weight is copied or reordered to use
    /// this.
    ///
    /// `false` when the shapes do not line up, which leaves the caller's
    /// own kernel to run — this is an optimization, never the only path to
    /// an answer.
    ///
    /// **Not for one token.** A mat-vec has no reuse to block for, and the
    /// packing is then pure overhead: measured level at one token and 6x
    /// worse once the weight is quantized, where reading the packed bytes
    /// directly beats widening them first.
    fn packed_gemm_f32(
        out: &mut [f32],
        x: &[f32],
        rows: &[f32],
        n_tokens: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> bool {
        use rten_gemm::{GemmInputA, GemmInputB, GemmOptions};
        use rten_tensor::NdTensorView;

        if !packed_gemm_enabled()
            || n_tokens < 2
            || rows.len() < in_dim * out_dim
            || x.len() < n_tokens * in_dim
            || out.len() < n_tokens * out_dim
        {
            return false;
        }
        // One executor per thread: building it picks a kernel for the
        // host's instruction set, which does not change while the process
        // runs, but the executor owns a boxed kernel that is not `Sync`, so
        // it cannot be shared from a `static`.
        thread_local! {
            static GEMM: rten_gemm::GemmExecutor<f32, f32, f32> =
                rten_gemm::GemmExecutor::default();
        }
        let a = NdTensorView::from_data([n_tokens, in_dim], &x[..n_tokens * in_dim]);
        let b = NdTensorView::from_data([out_dim, in_dim], &rows[..in_dim * out_dim]);
        GEMM.with(|gemm| {
            gemm.gemm(
                &mut out[..n_tokens * out_dim],
                GemmInputA::Unpacked(a),
                GemmInputB::Unpacked(b.transposed()),
                GemmOptions::default(),
            )
            .is_ok()
        })
    }

    /// The float path: `F32`/`F16`/`BF16` weights, widened a block at a time
    /// straight into the dot. Returns `None` for anything else.
    ///
    /// Separate from [`Self::matmul_fused`] because these types have no
    /// `int8` weight form — quantizing weights the file stores in full
    /// precision would be an accuracy change ggml does not make, so
    /// activations stay `f32` here. What it *does* remove is the same two
    /// costs the fused path removed for the quantized types: the per-row
    /// `Vec<f32>` and, on aarch64, `tensor::dot`'s single-accumulator scalar
    /// loop (that function has an AVX2 path and no NEON one).
    ///
    /// Reached in practice by `gemma-4`'s `per_layer_model_proj`, the last
    /// unfused matmul in the model set.
    fn matmul_float_into(
        &self,
        out: &mut Vec<f32>,
        x: &[f32],
        n_tokens: usize,
        w: &QuantMatrix,
    ) -> bool {
        let in_dim = w.in_dim;
        let out_dim = w.out_dim;
        let ggml_type = w.ggml_type();
        if !vecdot::supports_float(ggml_type) {
            return false;
        }
        let raw = w.raw_bytes();
        let row_bytes = w.row_bytes();

        // Prefill-sized: the tiled kernel (`vecdot::gemm_f32_rows`) over a
        // two-dimensional split — blocks of tokens × groups of rows — sized
        // so there are a few tasks per worker whatever the shape. A VAE
        // decode has both kinds: 96 rows against tens of thousands of
        // pixels at full resolution (split by rows alone, every task
        // streamed the whole activation band from memory for its four
        // rows), and 384 rows of 3×3×384 against a thousand latents at the
        // bottom (split by tokens alone, four tasks on twelve cores). Within
        // a task the token tile is applied to every quad of its row group
        // while it sits in L1, and the group's widened weights stay in L2.
        // Each task's outputs are a rectangle of the token-major result no
        // other task touches, written straight in — nothing to transpose.
        if n_tokens > 1 {
            let quad = vecdot::F32_ROWS;
            // `F32` weights are read in place — an `im2col` convolution's
            // weights already are `f32`, and copying 5 MiB of them per band
            // was most of a bad version of this path. The half-width types
            // are widened once per call, rows in parallel.
            let widened: Vec<f32>;
            // Safety: `f32` has no invalid bit patterns, so any aligned
            // four bytes are one; `align_to` reports the unaligned ends,
            // and only a fully aligned run is taken.
            let rows: &[f32] = match unsafe { raw.align_to::<f32>() } {
                ([], floats, []) if ggml_type == crate::engine::quant::GGML_TYPE_F32 => floats,
                _ => {
                    let mut all = vec![0f32; out_dim * in_dim];
                    all.par_chunks_mut(in_dim).enumerate().for_each(|(o, row)| {
                        let mut tmp = Vec::new();
                        vecdot::widen_float_row(
                            ggml_type,
                            &raw[o * row_bytes..(o + 1) * row_bytes],
                            in_dim,
                            &mut tmp,
                        );
                        row.copy_from_slice(&tmp);
                    });
                    widened = all;
                    &widened
                }
            };
            let row = |o: usize| &rows[o * in_dim..(o + 1) * in_dim];
            out.resize(n_tokens * out_dim, 0.0);
            // A packed, cache-blocked GEMM instead of the kernel below,
            // when asked for — see `packed_gemm_enabled` for what it is
            // worth here, which is not much. The weight goes over as a
            // transposed *view* of the rows above, so nothing is copied.
            if Self::packed_gemm_f32(out, x, rows, n_tokens, in_dim, out_dim) {
                return true;
            }
            let wanted = 4 * rayon::current_num_threads();
            // Rows per task: 32 unless the shape cannot make enough tasks
            // of them — a rank-64 LoRA's down projection has 64 rows and a
            // thousand tokens, which at 32 rows is two groups and, with the
            // token block floored at six, too few tasks to keep twelve
            // workers busy (measured: 2.7× one core on twelve). Halving
            // the group until there are `wanted` tasks fixes the tail.
            let max_blocks = n_tokens.div_ceil(6).max(1);
            let group = [F32_ROW_GROUP, 16, 8, 4]
                .into_iter()
                .find(|g| out_dim.div_ceil(*g) * max_blocks.min(wanted) >= wanted)
                .unwrap_or(4);
            let n_groups = out_dim.div_ceil(group);
            let blocks_wanted = wanted.div_ceil(n_groups).max(1);
            let block = n_tokens
                .div_ceil(blocks_wanted)
                .div_ceil(6)
                .saturating_mul(6)
                .clamp(6, F32_TOKEN_BLOCK);
            let n_blocks = n_tokens.div_ceil(block);
            let sink = SharedOut(out.as_mut_ptr());
            (0..n_blocks * n_groups).into_par_iter().for_each_init(
                || vec![0f32; block * quad],
                move |yt, task| {
                    let sink = sink;
                    let (b, g) = (task / n_groups, task % n_groups);
                    let t0 = b * block;
                    let nt = block.min(n_tokens - t0);
                    let xs = &x[t0 * in_dim..(t0 + nt) * in_dim];
                    let g0 = g * group;
                    let g1 = (g0 + group).min(out_dim);
                    let mut o0 = g0;
                    while o0 + quad <= g1 {
                        let (y0, rest) = yt.split_at_mut(nt);
                        let (y1, rest) = rest.split_at_mut(nt);
                        let (y2, rest) = rest.split_at_mut(nt);
                        let (y3, _) = rest.split_at_mut(nt);
                        vecdot::gemm_f32_rows(
                            [row(o0), row(o0 + 1), row(o0 + 2), row(o0 + 3)],
                            xs,
                            in_dim,
                            [y0, y1, y2, y3],
                        );
                        // Safety: this task alone writes rows `o0..o0 + 4` of
                        // tokens `t0..t0 + nt` in a buffer sized above; every
                        // other task's rectangle is disjoint.
                        for t in 0..nt {
                            for r in 0..quad {
                                unsafe {
                                    *sink.0.add((t0 + t) * out_dim + o0 + r) = yt[r * nt + t];
                                }
                            }
                        }
                        o0 += quad;
                    }
                    // A group whose end is not a multiple of four: the last
                    // rows one at a time.
                    for o in o0..g1 {
                        for t in 0..nt {
                            let v =
                                vecdot::dot_f32_slices(row(o), &xs[t * in_dim..(t + 1) * in_dim]);
                            // Safety: as above.
                            unsafe {
                                *sink.0.add((t0 + t) * out_dim + o) = v;
                            }
                        }
                    }
                },
            );
            return true;
        }

        // Transposed accumulation, as `matmul_fused` does: the rayon split is
        // over output rows, each written exactly once, so no scatter and no
        // per-row allocation. `matmul_dequant` collects a `Vec<f32>` per row
        // and scatters afterwards, which is the allocation this avoids.
        let mut yt = vec![0f32; out_dim * n_tokens];
        yt.par_chunks_mut(n_tokens)
            .enumerate()
            .for_each(|(o, dst)| {
                let row = &raw[o * row_bytes..(o + 1) * row_bytes];
                for (t, slot) in dst.iter_mut().enumerate() {
                    *slot = vecdot::dot_row_f32(ggml_type, row, &x[t * in_dim..(t + 1) * in_dim]);
                }
            });

        Self::finish_into(out, yt, n_tokens, out_dim);
        true
    }

    /// Materializes a `[out_dim, n_tokens]` accumulation into the caller's
    /// `[n_tokens, out_dim]` buffer — the one step every kernel above ends
    /// with, and the allocation this whole `_into` family exists to remove.
    ///
    /// Two cases, and the split is the point:
    ///
    /// * `n_tokens == 1`, where the transpose is the identity: the
    ///   accumulator *is* the answer, so it is swapped into place rather
    ///   than copied. Decode therefore allocates exactly what it allocated
    ///   before — no better, deliberately no worse.
    /// * `n_tokens > 1`: transposed straight into `out`, which is a caller
    ///   buffer reused across layers. This is where the saving is — one
    ///   allocate/free pair per projection per layer. See
    ///   [`Backend::matmul_into`] for why it is allocator bookkeeping rather
    ///   than `mmap` traffic at the default batch and stripe caps.
    ///
    /// `resize` rather than `clear` + `resize`: every element is written by
    /// the loop below, so zeroing them first is a wasted pass — the same
    /// reasoning as `tensor::rmsnorm_into`.
    fn finish_into(out: &mut Vec<f32>, mut yt: Vec<f32>, n_tokens: usize, out_dim: usize) {
        if n_tokens == 1 {
            std::mem::swap(out, &mut yt);
            return;
        }
        out.resize(n_tokens * out_dim, 0.0);
        for o in 0..out_dim {
            for t in 0..n_tokens {
                out[t * out_dim + o] = yt[o * n_tokens + t];
            }
        }
    }

    /// The dequantize path, on its own: every weight row widened to `f32`
    /// and dotted against the **unquantized** activations.
    ///
    /// Split out of [`Backend::matmul`] because it is the only
    /// full-precision matmul this backend has. `matmul` prefers
    /// [`Self::matmul_fused`], which rounds activations to `int8` and so
    /// carries a real (bounded, ggml-equivalent) error — fine in production,
    /// but useless as the *reference* in a cross-check of another backend's
    /// kernel, which would then be measuring quantization loss rather than
    /// correctness. Backend cross-checks call this instead; the `int8` error
    /// itself is bounded separately by `engine::vecdot`'s own tests.
    pub(crate) fn matmul_dequant(&self, x: &[f32], n_tokens: usize, w: &QuantMatrix) -> Vec<f32> {
        let in_dim = w.in_dim;
        let out_dim = w.out_dim;
        let mut y = vec![0f32; n_tokens * out_dim];
        // Parallelize over output rows (typically far more of these than
        // tokens) so each weight row is dequantized exactly once and reused
        // across every token, rather than once per (token, row) pair.
        let columns: Vec<(usize, Vec<f32>)> = (0..out_dim)
            .into_par_iter()
            .map(|o| {
                let wo = w.row(o);
                let column: Vec<f32> = (0..n_tokens)
                    .map(|t| tensor::dot(&x[t * in_dim..(t + 1) * in_dim], &wo))
                    .collect();
                (o, column)
            })
            .collect();
        for (o, column) in columns {
            for (t, value) in column.into_iter().enumerate() {
                y[t * out_dim + o] = value;
            }
        }
        y
    }
}

impl CpuBackend {
    /// [`Backend::matmul_batch_into`]'s one-token form: every op's rows in
    /// one parallel region, each op's activation quantized once (a shared
    /// input once for all). `false` when an op is not a one-token K-quant
    /// row dot, which is the per-op path's to handle.
    fn matmul_batch_one_token_into(&self, outs: &mut [Vec<f32>], ops: &[MatmulOp<'_>]) -> bool {
        // Two row kernels: the K-quants' per-super-block one and the per-32
        // one every other fused type takes; an op of neither is the per-op
        // path's.
        let k_row = |op: &MatmulOp<'_>| vecdot::supports_k_row(op.w.ggml_type(), op.w.in_dim);
        let flat = |op: &MatmulOp<'_>| vecdot::supports(op.w.ggml_type(), op.w.in_dim);
        if !ops
            .iter()
            .all(|op| op.n_tokens == 1 && op.x.len() == op.w.in_dim && (k_row(op) || flat(op)))
        {
            return false;
        }
        for op in ops {
            if let Some(layer) = op.w.layer() {
                crate::engine::dense_residency::touch(layer);
            }
        }
        // One quantized activation per distinct input, in whichever of the
        // two layouts the ops reading it need.
        struct Act {
            key: (usize, usize),
            k_row: Option<vecdot::ActQ8KRow>,
            flat: Option<vecdot::ActQ8>,
        }
        let mut acts: Vec<Act> = Vec::new();
        let act_of: Vec<usize> = ops
            .iter()
            .map(|op| {
                let key = (op.x.as_ptr() as usize, op.x.len());
                let i = match acts.iter().position(|a| a.key == key) {
                    Some(i) => i,
                    None => {
                        acts.push(Act {
                            key,
                            k_row: None,
                            flat: None,
                        });
                        acts.len() - 1
                    }
                };
                if k_row(op) {
                    if acts[i].k_row.is_none() {
                        acts[i].k_row = Some(vecdot::quantize_act_k_row(op.x));
                    }
                } else if acts[i].flat.is_none() {
                    acts[i].flat = Some(vecdot::quantize_act(op.x));
                }
                i
            })
            .collect();
        // Every op's rows end to end; `starts[i]` is where op `i` begins.
        let mut starts = Vec::with_capacity(ops.len() + 1);
        let mut total = 0usize;
        for op in ops {
            starts.push(total);
            total += op.w.out_dim;
        }
        starts.push(total);
        let mut flat = vec![0f32; total];
        flat.par_chunks_mut(1)
            .with_min_len(super::matmul_min_rows())
            .enumerate()
            .for_each(|(i, dst)| {
                let op_i = starts.partition_point(|&s| s <= i) - 1;
                let op = &ops[op_i];
                let o = i - starts[op_i];
                let row_bytes = op.w.row_bytes();
                let row = &op.w.raw_bytes()[o * row_bytes..(o + 1) * row_bytes];
                let act = &acts[act_of[op_i]];
                dst[0] = match &act.k_row {
                    Some(k) if k_row(op) => vecdot::dot_k_row(op.w.ggml_type(), row, k),
                    _ => vecdot::dot_row(
                        op.w.ggml_type(),
                        row,
                        act.flat.as_ref().expect("quantized for this op above"),
                    ),
                };
            });
        for (i, out) in outs.iter_mut().enumerate() {
            out.clear();
            out.extend_from_slice(&flat[starts[i]..starts[i + 1]]);
        }
        true
    }
}

impl Backend for CpuBackend {
    fn is_cpu(&self) -> bool {
        true
    }

    fn is_host(&self) -> bool {
        true
    }

    /// There is no ring to reset and no driver to give up on a submission:
    /// a long forward pass on the host is slow, not fatal. See the trait
    /// method for what the chunker does with this.
    fn has_submission_timeout(&self) -> bool {
        false
    }

    /// Both allocating entry points go through the `_into` forms, so there
    /// is one implementation of each kernel rather than two that could
    /// drift apart.
    fn matmul(&self, x: &[f32], n_tokens: usize, w: &QuantMatrix) -> Vec<f32> {
        let mut out = Vec::new();
        self.matmul_into(&mut out, x, n_tokens, w);
        out
    }

    fn matmul_into(&self, out: &mut Vec<f32>, x: &[f32], n_tokens: usize, w: &QuantMatrix) {
        debug_assert_eq!(x.len(), n_tokens * w.in_dim);
        // The one place every CPU weight read funnels through, and the only
        // point handed something that knows which layer it belongs to. That
        // is what lets `engine::dense_residency` follow the sweep without any
        // architecture's forward pass being aware of it — the same trick
        // `engine::placement` uses to route a split model.
        //
        // Inert unless `ORANGU_DENSE_WINDOW` is set: one `OnceLock` read and
        // one relaxed compare.
        if let Some(layer) = w.layer() {
            crate::engine::dense_residency::touch(layer);
        }
        if self.matmul_fused_into(out, x, n_tokens, w) {
            return;
        }
        if self.matmul_float_into(out, x, n_tokens, w) {
            return;
        }
        // Left allocating: this is the cross-check *reference* path, taken
        // only by types with no fused kernel at all, and it already
        // allocates a `Vec<f32>` per output row inside. Converting it would
        // save the smaller of its two allocations.
        *out = self.matmul_dequant(x, n_tokens, w);
    }

    fn matmul_decode(&self, x: &[f32], n_tokens: usize, w: &QuantMatrix) -> Vec<f32> {
        let mut out = Vec::new();
        self.matmul_decode_into(&mut out, x, n_tokens, w);
        out
    }

    fn matmul_decode_into(&self, out: &mut Vec<f32>, x: &[f32], n_tokens: usize, w: &QuantMatrix) {
        debug_assert_eq!(x.len(), n_tokens * w.in_dim);
        if self.matmul_fused_decode_into(out, x, n_tokens, w) {
            return;
        }
        // `matmul_float` needs no decode form of its own: it already dots one
        // `(row, token)` pair at a time and never materializes a row, so the
        // GEMV/GEMM split the quantized path makes has nothing to trade off.
        if self.matmul_float_into(out, x, n_tokens, w) {
            return;
        }
        // The dequantize path takes every `(row, token)` pair as its own
        // full-precision dot already, so it needs no decode form of its own.
        *out = self.matmul_dequant(x, n_tokens, w);
    }

    /// Per op and into the caller's buffers, which is the whole gain here:
    /// `CpuBackend` has no submission batching to do, so a batch is
    /// exactly `n` independent matmuls, and each of them can write into a
    /// buffer that outlives the layer. `matmul_batch` is this with fresh
    /// buffers, so the one-token region below serves both.
    ///
    /// `resize_with` rather than rebuilding the outer `Vec`: the inner
    /// buffers are the ones worth keeping, and a batch's shape does not
    /// change from layer to layer.
    ///
    /// **Except at one token**, where a batch's ops are one parallel region
    /// rather than one each. A layer's Q/K/V at decode are three matvecs of
    /// 8.8, 4.4 and 4.4 MB; as three regions each is a few dozen 64-row
    /// tasks over sixteen threads — a ramp, a tail and a join per op — and
    /// the three together read their 18 MB at 18.6 GB/s where the layer's
    /// 66 MB gate/up pair reads at 29 and the memory gives 35. One region
    /// over every op's rows is the same work with one ramp and one tail.
    fn matmul_batch(&self, ops: &[MatmulOp<'_>]) -> Vec<Vec<f32>> {
        let mut outs = Vec::new();
        self.matmul_batch_into(&mut outs, ops);
        outs
    }

    fn matmul_batch_into(&self, outs: &mut Vec<Vec<f32>>, ops: &[MatmulOp<'_>]) {
        outs.resize_with(ops.len(), Vec::new);
        if ops.len() > 1 && self.matmul_batch_one_token_into(outs, ops) {
            return;
        }
        for (out, op) in outs.iter_mut().zip(ops) {
            guarded_matmul_op_into(out, op, |dst, x, n_tokens, w| {
                self.matmul_into(dst, x, n_tokens, w)
            });
        }
    }

    /// Per op, since `CpuBackend` has nothing to amortize across a batch —
    /// the same reason it leaves [`Backend::matmul_batch`] at its default.
    fn matmul_batch_decode(&self, ops: &[MatmulOp<'_>]) -> Vec<Vec<f32>> {
        ops.iter()
            .map(|op| self.matmul_decode(op.x, op.n_tokens, op.w))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::loader::test_quant_matrix;
    use crate::engine::quant::{GGML_TYPE_Q4_K, GGML_TYPE_Q8_0};

    fn next_byte(seed: &mut u64) -> u8 {
        *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (*seed >> 33) as u8
    }

    /// Bytes per block and elements per block, for the two types this module
    /// tests. Hard-coded rather than read from `engine::quant`, whose own
    /// `block_layout` is private — the same thing the CUDA/OpenCL/ROCm test
    /// modules do with their `block_elems_for`.
    fn block_layout(ggml_type: u32) -> (usize, usize) {
        match ggml_type {
            GGML_TYPE_Q8_0 => (2 + 32, 32),
            GGML_TYPE_Q4_K => (2 + 2 + 12 + 128, 256),
            other => panic!("no fixture for ggml_type {other}"),
        }
    }

    /// `matmul_into` must produce exactly what `matmul` produces, from a
    /// buffer that is reused rather than fresh. That is the invariant every
    /// migrated call site depends on, and it has two halves worth testing
    /// separately: the values, and that nothing of the previous call
    /// survives.
    ///
    /// Both `n_tokens` cases matter and take different code paths —
    /// `n_tokens == 1` swaps the accumulator into place while `n_tokens > 1`
    /// transposes into the caller's buffer — and the K-quant GEMM at
    /// `n_tokens > 1` is a third kernel again. `77` is deliberately not a
    /// multiple of anything.
    #[test]
    fn matmul_into_matches_matmul_from_a_reused_buffer() {
        for ggml_type in [GGML_TYPE_Q8_0, GGML_TYPE_Q4_K] {
            let (_, block_elems) = block_layout(ggml_type);
            let in_dim = block_elems * 3;
            // Odd, so the two-rows-per-chunk kernels hit their trailing
            // single-row branch.
            let out_dim = 11;
            let mut seed = 0x9E37_79B9_7F4A_7C15u64;
            let bytes = weights(ggml_type, in_dim, out_dim, &mut seed);
            let w = test_quant_matrix(&bytes, ggml_type, in_dim, out_dim);

            // Reused across every shape below, exactly as a layer loop
            // reuses it — so a later, narrower call has to shrink it.
            let mut reused: Vec<f32> = vec![f32::NAN; 4096];

            for n_tokens in [1usize, 2, 5, 77, 3, 1] {
                let x: Vec<f32> = (0..n_tokens * in_dim)
                    .map(|i| ((i % 29) as f32 - 14.0) * 0.037)
                    .collect();

                let want = CpuBackend.matmul(&x, n_tokens, &w);
                CpuBackend.matmul_into(&mut reused, &x, n_tokens, &w);
                assert_eq!(
                    reused.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                    want.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                    "type {ggml_type} n_tokens {n_tokens}: matmul_into differs from matmul"
                );
                assert_eq!(reused.len(), n_tokens * out_dim);

                let want = CpuBackend.matmul_decode(&x, n_tokens, &w);
                CpuBackend.matmul_decode_into(&mut reused, &x, n_tokens, &w);
                assert_eq!(
                    reused.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                    want.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                    "type {ggml_type} n_tokens {n_tokens}: matmul_decode_into differs"
                );
            }
        }
    }

    /// A one-token batch — the shape of a decode layer's Q/K/V — comes out
    /// of the single-region path bit for bit as the ops would one at a
    /// time: two ops on one input, a third on another, widths that are not
    /// task multiples, and a mixed batch (a per-32 type among the K-quants)
    /// that has to take the per-op path.
    #[test]
    fn a_one_token_batch_matches_the_ops_one_at_a_time() {
        let mut seed = 0x0B47_C4ED_u64;
        let in_dim = 256 * 3;
        let mats: Vec<(u32, usize)> = vec![
            (GGML_TYPE_Q4_K, 200),
            (GGML_TYPE_Q4_K, 67),
            (GGML_TYPE_Q4_K, 129),
        ];
        let ws: Vec<_> = mats
            .iter()
            .map(|&(t, out_dim)| {
                let bytes = weights(t, in_dim, out_dim, &mut seed);
                (bytes, t, out_dim)
            })
            .collect();
        let ws: Vec<_> = ws
            .iter()
            .map(|(bytes, t, out_dim)| test_quant_matrix(bytes, *t, in_dim, *out_dim))
            .collect();
        let x1: Vec<f32> = (0..in_dim)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.037)
            .collect();
        let x2: Vec<f32> = (0..in_dim)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.051)
            .collect();
        let ops = [
            MatmulOp {
                x: &x1,
                n_tokens: 1,
                w: &ws[0],
            },
            MatmulOp {
                x: &x1,
                n_tokens: 1,
                w: &ws[1],
            },
            MatmulOp {
                x: &x2,
                n_tokens: 1,
                w: &ws[2],
            },
        ];
        let mut outs = Vec::new();
        CpuBackend.matmul_batch_into(&mut outs, &ops);
        assert_eq!(outs.len(), 3);
        for (out, op) in outs.iter().zip(&ops) {
            let want = CpuBackend.matmul(op.x, 1, op.w);
            assert_eq!(
                out.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                want.iter().map(|f| f.to_bits()).collect::<Vec<_>>()
            );
        }
        // A `Q8_0` op in the batch: not a K-quant row dot, so the whole
        // batch takes the per-op path — and still matches.
        let q8 = weights(GGML_TYPE_Q8_0, in_dim, 40, &mut seed);
        let q8 = test_quant_matrix(&q8, GGML_TYPE_Q8_0, in_dim, 40);
        let mixed = [
            MatmulOp {
                x: &x1,
                n_tokens: 1,
                w: &ws[0],
            },
            MatmulOp {
                x: &x1,
                n_tokens: 1,
                w: &q8,
            },
        ];
        let mut outs = Vec::new();
        CpuBackend.matmul_batch_into(&mut outs, &mixed);
        for (out, op) in outs.iter().zip(&mixed) {
            assert_eq!(*out, CpuBackend.matmul(op.x, 1, op.w));
        }
    }

    /// One weight matrix of `ggml_type`. Float scale fields get bounded,
    /// non-degenerate values; every other field is read back as an integer,
    /// so arbitrary bits are fine there.
    /// A timing loop for the K-quant prefill GEMM at a Qwen-Image linear's
    /// shape, for tuning the kernel without a model or a server:
    ///
    /// ```text
    /// ORANGU_EXPERT_K_ROWS=16 cargo test --profile release-with-debug \
    ///     --bin orangu-server k_gemm_throughput -- --ignored --nocapture
    /// ```
    ///
    /// Prints G MAC/s. Ignored because it takes seconds and proves nothing
    /// about correctness — the bit-identity tests do that.
    #[test]
    #[ignore]
    fn k_gemm_throughput() {
        // `ORANGU_BENCH_SHAPE=in,out,tokens` picks another shape — e.g.
        // `1024,96,65536`, a VAE convolution at full resolution; the type
        // is `Q4_K`, or `Q6_K` with `ORANGU_BENCH_Q6K=1`.
        let (in_dim, out_dim, n_tokens) = std::env::var("ORANGU_BENCH_SHAPE")
            .ok()
            .and_then(|v| {
                let mut it = v.split(',').map(|p| p.trim().parse::<usize>().ok());
                Some((it.next()??, it.next()??, it.next()??))
            })
            .unwrap_or((3072, 3072, 256));
        let ggml_type = if std::env::var("ORANGU_BENCH_Q6K").is_ok_and(|v| v == "1") {
            crate::engine::quant::GGML_TYPE_Q6_K
        } else {
            GGML_TYPE_Q4_K
        };
        let mut seed = 0x1234_5678_9ABC_DEF0u64;
        let bytes = if ggml_type == GGML_TYPE_Q4_K {
            weights(GGML_TYPE_Q4_K, in_dim, out_dim, &mut seed)
        } else {
            let values: Vec<f32> = (0..in_dim * out_dim)
                .map(|i| ((i * 31 % 17) as f32 - 8.0) * 0.01)
                .collect();
            orangu::quantize::encode(ggml_type, &values, in_dim)
        };
        let w = test_quant_matrix(&bytes, ggml_type, in_dim, out_dim);
        let x: Vec<f32> = (0..n_tokens * in_dim)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.037)
            .collect();
        let mut out = Vec::new();
        // Warm the pool and the caches once.
        CpuBackend.matmul_into(&mut out, &x, n_tokens, &w);
        let reps = std::env::var("ORANGU_BENCH_REPS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(5);
        let started = std::time::Instant::now();
        for _ in 0..reps {
            CpuBackend.matmul_into(&mut out, &x, n_tokens, &w);
        }
        let secs = started.elapsed().as_secs_f64() / reps as f64;
        let gmac = (in_dim * out_dim * n_tokens) as f64 / secs / 1e9;
        eprintln!(
            "k_gemm {in_dim}x{out_dim} x {n_tokens} tokens: {:.1} ms, {gmac:.1} G MAC/s \
             (i8mm {}, rows/task {})",
            secs * 1e3,
            vecdot::have_i8mm(),
            vecdot::k_rows_per_task()
        );
    }

    /// The tiled float path must agree with the dequantize reference at
    /// every awkward shape at once: an `out_dim` that is not a multiple of
    /// the row quad or the row group, an `in_dim` that is not a multiple of
    /// four, token counts that leave a short last tile and a short last
    /// block, and rows stored as `F32`, `F16` and `BF16`.
    #[test]
    fn float_prefill_matches_the_dequantize_reference_at_awkward_shapes() {
        use crate::engine::quant::{GGML_TYPE_BF16, GGML_TYPE_F16, GGML_TYPE_F32};
        for (ggml_type, out_dim, in_dim) in [
            (GGML_TYPE_F32, 11usize, 37usize),
            (GGML_TYPE_F32, 35, 64),
            (GGML_TYPE_F16, 9, 21),
            (GGML_TYPE_BF16, 4, 8),
            // Model-shaped, because the awkward shapes above are all
            // smaller than one cache tile and so never exercise a packed
            // GEMM's blocking: a mixture router is `[n_embd] -> [experts]`.
            (GGML_TYPE_F32, 128, 2048),
        ] {
            let value = |i: usize| ((i * 31 % 17) as f32 - 8.0) * 0.125;
            let mut bytes = Vec::new();
            for i in 0..out_dim * in_dim {
                let v = value(i);
                match ggml_type {
                    GGML_TYPE_F32 => bytes.extend_from_slice(&v.to_le_bytes()),
                    GGML_TYPE_F16 => bytes.extend_from_slice(&half::f16::from_f32(v).to_le_bytes()),
                    _ => bytes.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes()),
                }
            }
            let w = test_quant_matrix(&bytes, ggml_type, in_dim, out_dim);
            for n_tokens in [2usize, 5, 6, 7, 259, 1000] {
                let x: Vec<f32> = (0..n_tokens * in_dim).map(|i| value(i + 3) * 0.5).collect();
                let want = CpuBackend.matmul_dequant(&x, n_tokens, &w);
                let got = CpuBackend.matmul(&x, n_tokens, &w);
                assert_eq!(got.len(), want.len());
                for (i, (g, e)) in got.iter().zip(&want).enumerate() {
                    assert!(
                        (g - e).abs() <= 1e-4 * e.abs().max(1.0),
                        "type {ggml_type} {out_dim}x{in_dim} tokens {n_tokens} at {i}: {g} vs {e}"
                    );
                }
            }
        }
    }

    /// The packed GEMM has to actually run at the shapes it was adopted
    /// for. It reports failure by returning `false`, and the caller then
    /// falls through to the kernel below it and produces a correct answer —
    /// so the cross-check above passes whether or not this path is taken,
    /// and only this test would notice it silently switching itself off.
    ///
    /// One token is the deliberate exception: a mat-vec has no reuse to
    /// block for and measured level, so it stays on the kernel below.
    #[test]
    fn the_packed_gemm_runs_at_model_shapes_and_declines_a_single_token() {
        let (in_dim, out_dim) = (2048usize, 128usize);
        let rows = vec![0.5f32; in_dim * out_dim];
        for n_tokens in [2usize, 256, 2048] {
            let x = vec![0.25f32; n_tokens * in_dim];
            let mut out = vec![0f32; n_tokens * out_dim];
            assert!(
                CpuBackend::packed_gemm_f32(&mut out, &x, &rows, n_tokens, in_dim, out_dim),
                "the packed GEMM declined a model shape at {n_tokens} tokens"
            );
            let want = 0.5 * 0.25 * in_dim as f32;
            assert!((out[0] - want).abs() <= 1e-3 * want, "{} vs {want}", out[0]);
        }
        let x = vec![0.25f32; in_dim];
        let mut out = vec![0f32; out_dim];
        assert!(
            !CpuBackend::packed_gemm_f32(&mut out, &x, &rows, 1, in_dim, out_dim),
            "one token should stay on the mat-vec kernel"
        );
    }

    /// The float sibling of [`k_gemm_throughput`], at a Qwen-Image VAE
    /// convolution's shape: 96 output channels, a 3×3×96 `im2col` row,
    /// a band of 65,536 pixels.
    #[test]
    #[ignore]
    fn f32_gemm_throughput() {
        // `ORANGU_BENCH_SHAPE=in,out,tokens` picks another shape — e.g.
        // `3072,64,1054` and `64,3072,1054`, a rank-64 LoRA's two products.
        let (in_dim, out_dim, n_tokens) = std::env::var("ORANGU_BENCH_SHAPE")
            .ok()
            .and_then(|v| {
                let mut it = v.split(',').map(|p| p.trim().parse::<usize>().ok());
                Some((it.next()??, it.next()??, it.next()??))
            })
            .unwrap_or((864, 96, 65_536));
        let bytes: Vec<u8> = (0..in_dim * out_dim)
            .flat_map(|i| (((i * 31 % 17) as f32 - 8.0) * 0.01).to_le_bytes())
            .collect();
        let w = test_quant_matrix(&bytes, crate::engine::quant::GGML_TYPE_F32, in_dim, out_dim);
        let x: Vec<f32> = (0..n_tokens * in_dim)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.037)
            .collect();
        let mut out = Vec::new();
        CpuBackend.matmul_into(&mut out, &x, n_tokens, &w);
        let reps = std::env::var("ORANGU_BENCH_REPS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(5);
        let started = std::time::Instant::now();
        for _ in 0..reps {
            CpuBackend.matmul_into(&mut out, &x, n_tokens, &w);
        }
        let secs = started.elapsed().as_secs_f64() / reps as f64;
        let gmac = (in_dim * out_dim * n_tokens) as f64 / secs / 1e9;
        eprintln!(
            "f32_gemm {in_dim}x{out_dim} x {n_tokens} tokens: {:.1} ms, {gmac:.1} G MAC/s",
            secs * 1e3
        );
    }

    fn weights(ggml_type: u32, in_dim: usize, out_dim: usize, seed: &mut u64) -> Vec<u8> {
        let (block_bytes, block_elems) = block_layout(ggml_type);
        let mut bytes = Vec::new();
        for _ in 0..out_dim * (in_dim / block_elems) {
            let mut block = vec![0u8; block_bytes];
            let scale = 0.05 + (next_byte(seed) as f32 / 255.0) * 1.95;
            block[0..2].copy_from_slice(&half::f16::from_f32(scale).to_le_bytes());
            let quants_from = if ggml_type == GGML_TYPE_Q4_K {
                let dmin = 0.05 + (next_byte(seed) as f32 / 255.0) * 0.95;
                block[2..4].copy_from_slice(&half::f16::from_f32(dmin).to_le_bytes());
                4
            } else {
                2
            };
            for slot in block[quants_from..].iter_mut() {
                *slot = next_byte(seed);
            }
            bytes.extend(block);
        }
        bytes
    }
}
