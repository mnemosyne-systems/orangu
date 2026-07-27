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
//! Two paths, picked per weight tensor:
//!
//! * **Fused** (`engine::vecdot`) — for the quantized types that have a
//!   fused kernel, the weight row's raw quantized bytes are dotted straight
//!   against `int8`-quantized activations, ggml-style. No dequantize, no
//!   per-row `Vec<f32>`.
//! * **Dequantize** — the original path (`QuantMatrix::row` → a fresh
//!   `Vec<f32>` → `engine::tensor::dot` with runtime AVX2 dispatch), still
//!   used for every type `vecdot::supports` declines.
//!
//! The fused path exists because a `perf` profile of CPU decode put 95% of
//! all time in this one function — see `engine::vecdot`'s module doc for the
//! measured breakdown.

use rayon::prelude::*;

use crate::engine::loader::QuantMatrix;
use crate::engine::tensor;
use crate::engine::vecdot;

use super::Backend;

#[derive(Default)]
pub struct CpuBackend;

impl CpuBackend {
    /// The fused path: quantize each token's activations to `int8` **once**,
    /// then dot them against the still-quantized weight rows. Returns `None`
    /// for any weight type without a fused kernel, so the caller falls back.
    fn matmul_fused(&self, x: &[f32], n_tokens: usize, w: &QuantMatrix) -> Option<Vec<f32>> {
        let in_dim = w.in_dim;
        let out_dim = w.out_dim;
        let ggml_type = w.ggml_type();
        if !vecdot::supports(ggml_type, in_dim) {
            return None;
        }
        // Once per call, not once per (row, token) — this is the whole point.
        let acts: Vec<vecdot::ActQ8> = (0..n_tokens)
            .map(|t| vecdot::quantize_act(&x[t * in_dim..(t + 1) * in_dim]))
            .collect();
        let raw = w.raw_bytes();
        let row_bytes = w.row_bytes();

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
            yt.par_chunks_mut(1).enumerate().for_each(|(o, dst)| {
                dst[0] = vecdot::dot_row(ggml_type, &raw[o * row_bytes..(o + 1) * row_bytes], act);
            });
            // The transpose is the identity — hand the buffer straight back.
            return Some(yt);
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

        let mut y = vec![0f32; n_tokens * out_dim];
        for o in 0..out_dim {
            for t in 0..n_tokens {
                y[t * out_dim + o] = yt[o * n_tokens + t];
            }
        }
        Some(y)
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

impl Backend for CpuBackend {
    fn matmul(&self, x: &[f32], n_tokens: usize, w: &QuantMatrix) -> Vec<f32> {
        debug_assert_eq!(x.len(), n_tokens * w.in_dim);
        if let Some(y) = self.matmul_fused(x, n_tokens, w) {
            return y;
        }
        self.matmul_dequant(x, n_tokens, w)
    }
}
