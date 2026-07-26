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

//! The handful of numeric operations a Llama-style forward pass needs, on
//! plain `f32` slices — not a general ND-array library. Every tensor here
//! is row-major: an `[n_rows, n_cols]` matrix is `n_rows` contiguous rows
//! of `n_cols` elements, matching both ggml's own weight layout and how
//! this project's `engine::loader` returns dequantized tensors.
//!
//! The elementwise ops here (`rmsnorm_inplace`, `add_inplace`, `mul_inplace`,
//! `gelu_inplace`) parallelise across rows/elements with rayon **only above a
//! work threshold** — the multi-token prefill path (`run_layers_cpu`) exercises
//! them on `n_tokens × dim` buffers where the speedup is large, while the
//! single-token decode / CPU-fallback case stays serial so it never pays
//! rayon's task-dispatch overhead. The parallel and serial forms are
//! bit-for-bit identical (each row/element is independent).

use rayon::prelude::*;

/// Row count at/above which `rmsnorm_inplace` parallelises across rows.
pub(crate) const PAR_ROWS_THRESHOLD: usize = 32;
/// Element count at/above which `add`/`mul`/`gelu`_inplace parallelise.
const PAR_ELEMS_THRESHOLD: usize = 1 << 15;

/// Dot product of two equal-length `f32` slices, auto-vectorized via
/// AVX2+FMA where available (`RUSTFLAGS`-independent — checked once per
/// call site at runtime, not assumed from `.cargo/config.toml`'s
/// compile-time baseline; see `doc/BUILDING.md`), falling back to a
/// scalar loop everywhere else.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        // `dot_avx2` uses `_mm256_fmadd_ps`, which needs the `fma` CPUID
        // bit specifically — a real (if now rare) x86_64 CPU can have
        // AVX2 without it, so both must be checked, not just "avx2": an
        // earlier version of this function checked only `avx2`, which
        // would have executed an illegal instruction on such a CPU.
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // Safety: guarded by the runtime feature checks above.
            return unsafe { dot_avx2(a, b) };
        }
    }
    dot_scalar(a, b)
}

fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Four independent accumulators, not one.
///
/// A single accumulator makes the loop a **dependency chain**: each
/// `_mm256_fmadd_ps` has to wait for the previous one to retire, so a
/// `head_dim = 256` dot product serialises 32 FMAs at ~4 cycles of latency
/// each where the core could otherwise issue two per cycle. That is the loop's
/// real cost — the arithmetic is nowhere near the limit, the chain is — and
/// attention runs one of these per query × head × window position.
///
/// Four chains cover the latency (4 cycles × 2 per cycle needs ≥ 8 in flight,
/// and each iteration issues 4 independent FMAs plus their loads). Widening
/// past four stops paying: the loads become the constraint.
///
/// This changes the *summation order* — four partial sums combined pairwise at
/// the end rather than one running total — so it is not bit-identical to the
/// single-accumulator form. That is float reassociation, the same kind
/// `dot_scalar` and the vector path have always differed by (a sequential sum
/// versus eight lanes plus a horizontal reduction), and it is why `dot` has
/// never been the reference for a bit-exact claim. `axpy_inplace` next door
/// *is* bit-exact, because its reference is a specific scalar loop.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
        let n = a.len().min(b.len());
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        let pa = a.as_ptr();
        let pb = b.as_ptr();
        let mut i = 0usize;
        while i + 32 <= n {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
            acc1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(pa.add(i + 8)),
                _mm256_loadu_ps(pb.add(i + 8)),
                acc1,
            );
            acc2 = _mm256_fmadd_ps(
                _mm256_loadu_ps(pa.add(i + 16)),
                _mm256_loadu_ps(pb.add(i + 16)),
                acc2,
            );
            acc3 = _mm256_fmadd_ps(
                _mm256_loadu_ps(pa.add(i + 24)),
                _mm256_loadu_ps(pb.add(i + 24)),
                acc3,
            );
            i += 32;
        }
        while i + 8 <= n {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
            i += 8;
        }
        let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        let mut buf = [0f32; 8];
        _mm256_storeu_ps(buf.as_mut_ptr(), acc);
        let mut sum: f32 = buf.iter().sum();
        for j in i..n {
            sum += a[j] * b[j];
        }
        sum
    }
}

/// `out[i] += scale * v[i]` — attention's value accumulation.
///
/// The companion to [`dot`], and it exists for the same reason. Attention runs
/// exactly as many of these as it runs dot products (one per query × head ×
/// window position), but only the dot product was ever vectorized; this half
/// was a `out.iter_mut().zip(v.iter())` loop, and a CPU profile of prefill
/// found the zip's own iteration machinery — `next`, `spec_next`,
/// `unchecked_add` — accounting for more samples than the arithmetic it was
/// carrying.
///
/// **Bit-exact with the scalar form**, deliberately: a separate multiply and
/// add, not `_mm256_fmadd_ps`. An FMA rounds once where the scalar rounds
/// twice, so fusing here would change every attention output slightly and cost
/// the ability to verify this by byte-comparing a generation against the
/// previous build. [`dot`] can use FMA because its own reference is the same
/// AVX2 code; this one's reference is the loop it replaced.
pub fn axpy_inplace(out: &mut [f32], v: &[f32], scale: f32) {
    debug_assert_eq!(out.len(), v.len());
    #[cfg(target_arch = "x86_64")]
    {
        // AVX2 alone is enough — no `fma` bit needed, unlike `dot`.
        if is_x86_feature_detected!("avx2") {
            // Safety: guarded by the runtime feature check above.
            unsafe { axpy_avx2(out, v, scale) };
            return;
        }
    }
    axpy_scalar(out, v, scale);
}

fn axpy_scalar(out: &mut [f32], v: &[f32], scale: f32) {
    for (o, vi) in out.iter_mut().zip(v.iter()) {
        *o += scale * vi;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn axpy_avx2(out: &mut [f32], v: &[f32], scale: f32) {
    use std::arch::x86_64::*;
    unsafe {
        let n = out.len().min(v.len());
        let chunks = n / 8;
        let vs = _mm256_set1_ps(scale);
        for i in 0..chunks {
            let vo = _mm256_loadu_ps(out.as_ptr().add(i * 8));
            let vv = _mm256_loadu_ps(v.as_ptr().add(i * 8));
            // Multiply then add, matching `axpy_scalar`'s two roundings.
            let prod = _mm256_mul_ps(vs, vv);
            _mm256_storeu_ps(out.as_mut_ptr().add(i * 8), _mm256_add_ps(vo, prod));
        }
        for i in chunks * 8..n {
            *out.get_unchecked_mut(i) += scale * *v.get_unchecked(i);
        }
    }
}

/// In-place RMSNorm over each row of `x` (`[n_tokens, dim]`), scaled by
/// `weight` (`[dim]`) — `x[t,i] = x[t,i] / rms(x[t,:]) * weight[i]`.
pub fn rmsnorm_inplace(x: &mut [f32], weight: &[f32], n_tokens: usize, dim: usize, eps: f32) {
    debug_assert_eq!(x.len(), n_tokens * dim);
    debug_assert_eq!(weight.len(), dim);
    let norm_row = |row: &mut [f32]| {
        let mean_sq: f32 = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let scale = 1.0 / (mean_sq + eps).sqrt();
        for (v, w) in row.iter_mut().zip(weight.iter()) {
            *v = *v * scale * w;
        }
    };
    if n_tokens >= PAR_ROWS_THRESHOLD {
        x.par_chunks_mut(dim).for_each(norm_row);
    } else {
        x.chunks_mut(dim).for_each(norm_row);
    }
}

/// In-place softmax over a single row.
pub fn softmax_inplace(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v /= sum;
        }
    }
}

/// SiLU (`x * sigmoid(x)`), a.k.a. swish — the activation SwiGLU's gate
/// projection uses.
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Logistic sigmoid — MoE shared-expert gating and the gated-delta-net
/// layer gate (`engine::arch::qwen35moe`) both use this directly (unlike
/// `silu`, without multiplying back by `x`).
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `ln(1 + e^x)`, ggml's own `op_softplus` formula exactly
/// (`ggml-cpu/unary-ops.cpp`) including its overflow guard.
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

/// L2-normalizes `x` in place: `x[i] /= max(||x||_2, eps)` — ggml's
/// `ggml_l2_norm` (`ggml-cpu/ops.cpp`), used by gated-delta-net's Q/K norm
/// (distinct from RMSNorm: no averaging over `dim`, and no learned weight).
pub fn l2_norm_inplace(x: &mut [f32], eps: f32) {
    let norm = x.iter().map(|v| v * v).sum::<f32>().sqrt().max(eps);
    for v in x.iter_mut() {
        *v /= norm;
    }
}

/// GELU (tanh approximation), the activation Gemma's GEGLU FFN uses —
/// ggml's own `ggml_gelu_f32` formula exactly (`ggml-cpu/vec.h`), not the
/// erf-exact variant.
pub fn gelu(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const GELU_COEF_A: f32 = 0.044715;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * x * (1.0 + GELU_COEF_A * x * x)).tanh())
}

/// Applies rotary position embedding (RoPE, NEOX/GPT-NeoX-style pairing —
/// the convention llama.cpp's own GGUF tensors are laid out for) in place to
/// `x`, one token's `[n_head, head_dim]` block, at absolute position `pos`.
/// Only the leading `rope_dim` elements of each head rotate; any remainder
/// (`head_dim > rope_dim`, e.g. some partial-RoPE models) passes through
/// unchanged.
pub fn rope_apply_inplace(
    x: &mut [f32],
    n_head: usize,
    head_dim: usize,
    rope_dim: usize,
    pos: usize,
    freq_base: f32,
) {
    rope_apply_scaled_inplace(x, n_head, head_dim, rope_dim, pos, freq_base, None);
}

/// Like [`rope_apply_inplace`], but divides each pair's rotation frequency
/// by the matching entry of `freq_factors` (`[rope_dim/2]`) when given —
/// ggml's `theta/ff` in `ggml_rope_cache_init` — for models with a learned
/// "proportional RoPE" tensor (e.g. Gemma4's `rope_freqs`, only applied to
/// its full-attention layers).
pub fn rope_apply_scaled_inplace(
    x: &mut [f32],
    n_head: usize,
    head_dim: usize,
    rope_dim: usize,
    pos: usize,
    freq_base: f32,
    freq_factors: Option<&[f32]>,
) {
    debug_assert_eq!(x.len(), n_head * head_dim);
    let half = rope_dim / 2;
    for h in 0..n_head {
        let head = &mut x[h * head_dim..(h + 1) * head_dim];
        for i in 0..half {
            let mut freq = freq_base.powf(-2.0 * i as f32 / rope_dim as f32);
            if let Some(ff) = freq_factors {
                freq /= ff[i];
            }
            let theta = pos as f32 * freq;
            let (sin, cos) = theta.sin_cos();
            let a = head[i];
            let b = head[i + half];
            head[i] = a * cos - b * sin;
            head[i + half] = a * sin + b * cos;
        }
    }
}

/// Adds `bias` (`[dim]`) to every row of `x` (`[n_rows, dim]`) — a
/// projection bias, e.g. Qwen2/Qwen3's `attn_q.bias`/`attn_k.bias`/
/// `attn_v.bias` (plain Llama/Mistral GGUFs have no such tensors at all).
pub fn add_bias_per_row(x: &mut [f32], bias: &[f32], n_rows: usize) {
    let dim = bias.len();
    debug_assert_eq!(x.len(), n_rows * dim);
    for row in x.chunks_mut(dim) {
        add_inplace(row, bias);
    }
}

/// Elementwise `a[i] += b[i]`.
pub fn add_inplace(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    if a.len() >= PAR_ELEMS_THRESHOLD {
        a.par_iter_mut()
            .zip(b.par_iter())
            .for_each(|(x, y)| *x += y);
    } else {
        for (x, y) in a.iter_mut().zip(b.iter()) {
            *x += y;
        }
    }
}

/// Elementwise `a[i] *= b[i]`.
pub fn mul_inplace(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    if a.len() >= PAR_ELEMS_THRESHOLD {
        a.par_iter_mut()
            .zip(b.par_iter())
            .for_each(|(x, y)| *x *= y);
    } else {
        for (x, y) in a.iter_mut().zip(b.iter()) {
            *x *= y;
        }
    }
}

/// Elementwise in-place GELU (tanh approximation) — the FFN gate activation.
/// Parallelised above `PAR_ELEMS_THRESHOLD` (prefill applies it to the whole
/// `n_tokens × ffn_len` gate buffer, the single largest CPU-elementwise cost
/// there — each element an independent transcendental).
pub fn gelu_inplace(x: &mut [f32]) {
    if x.len() >= PAR_ELEMS_THRESHOLD {
        x.par_iter_mut().for_each(|v| *v = gelu(*v));
    } else {
        for v in x.iter_mut() {
            *v = gelu(*v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_matches_scalar_reference_for_odd_and_even_lengths() {
        for len in [1, 7, 8, 9, 16, 33] {
            let a: Vec<f32> = (0..len).map(|i| i as f32 * 0.5).collect();
            let b: Vec<f32> = (0..len).map(|i| (len - i) as f32 * 0.25).collect();
            let expected = dot_scalar(&a, &b);
            assert!((dot(&a, &b) - expected).abs() < 1e-3, "len={len}");
        }
    }

    #[test]
    fn rmsnorm_normalizes_each_row_independently() {
        // A real model's rms_eps is always nonzero (read from the GGUF
        // file, defaulting to 1e-5) — this is that realistic case, not
        // eps=0, which is a degenerate 0/0 input no real config produces.
        let eps = 1e-5f32;
        let mut x = [3.0, 4.0, 0.0, 0.0];
        let weight = [1.0, 1.0];
        rmsnorm_inplace(&mut x, &weight, 2, 2, eps);
        // Row 0: rms = sqrt((9+16)/2 + eps); 3/rms, 4/rms
        let rms = (12.5f32 + eps).sqrt();
        assert!((x[0] - 3.0 / rms).abs() < 1e-4);
        assert!((x[1] - 4.0 / rms).abs() < 1e-4);
        // Row 1 is all zero: normalized stays exactly zero.
        assert_eq!(x[2], 0.0);
        assert_eq!(x[3], 0.0);
    }

    #[test]
    fn softmax_sums_to_one_and_preserves_order() {
        let mut x = [1.0, 2.0, 3.0];
        softmax_inplace(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(x[0] < x[1] && x[1] < x[2]);
    }

    #[test]
    fn silu_matches_reference_values() {
        assert!((silu(0.0) - 0.0).abs() < 1e-6);
        // silu(1) = 1 * sigmoid(1) ≈ 0.7310586
        assert!((silu(1.0) - 0.7310586).abs() < 1e-5);
    }

    #[test]
    fn gelu_matches_reference_values() {
        assert!((gelu(0.0) - 0.0).abs() < 1e-6);
        // gelu(1) ≈ 0.841_192 (tanh approximation, matches ggml's own value)
        assert!((gelu(1.0) - 0.841_192).abs() < 1e-5);
        // gelu(-1) ≈ -0.158_808
        assert!((gelu(-1.0) - (-0.158_808)).abs() < 1e-5);
    }

    #[test]
    fn sigmoid_matches_reference_values() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        // sigmoid(1) ≈ 0.7310586
        assert!((sigmoid(1.0) - 0.7310586).abs() < 1e-5);
    }

    #[test]
    fn softplus_matches_reference_values() {
        // softplus(0) = ln(2)
        assert!((softplus(0.0) - std::f32::consts::LN_2).abs() < 1e-5);
        // Overflow guard: softplus(x) = x for x > 20.
        assert_eq!(softplus(25.0), 25.0);
    }

    #[test]
    fn l2_norm_inplace_produces_a_unit_vector() {
        let mut x = [3.0, 4.0];
        l2_norm_inplace(&mut x, 1e-6);
        assert!((x[0] - 0.6).abs() < 1e-5);
        assert!((x[1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn l2_norm_inplace_clamps_by_eps_for_a_near_zero_vector() {
        let mut x = [0.0, 0.0];
        l2_norm_inplace(&mut x, 1e-3);
        // norm=0 clamped to eps=1e-3, so x/eps = 0/1e-3 = 0.
        assert_eq!(x, [0.0, 0.0]);
    }

    #[test]
    fn rope_at_position_zero_is_the_identity() {
        let mut x = [1.0, 2.0, 3.0, 4.0];
        let original = x;
        rope_apply_inplace(&mut x, 1, 4, 4, 0, 10000.0);
        assert_eq!(x, original);
    }

    #[test]
    fn rope_preserves_pair_norm() {
        let mut x: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let norm_before = (x[0] * x[0] + x[2] * x[2]).sqrt();
        rope_apply_inplace(&mut x, 1, 4, 4, 5, 10000.0);
        let norm_after = (x[0] * x[0] + x[2] * x[2]).sqrt();
        assert!((norm_before - norm_after).abs() < 1e-5);
    }

    /// `axpy_inplace` replaced attention's value-accumulation loop, so its
    /// reference is that loop — and the claim is not "close", it is **equal**.
    /// A separate multiply and add rounds exactly where the scalar version
    /// does; had this used `_mm256_fmadd_ps` instead, this test would fail and
    /// every attention output would have moved by an amount too small to
    /// notice and too large to ignore when byte-comparing two builds.
    ///
    /// Lengths straddle the 8-wide vector body deliberately: `0` (no work),
    /// under one vector, exact multiples, and multiples plus a tail, including
    /// the 256 and 512 real `head_dim`s.
    #[test]
    fn axpy_is_bit_exact_with_the_scalar_loop_it_replaced() {
        let mut seed = 0x9E37_79B9_u32;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32) * 4.0 - 2.0
        };
        for len in [
            0usize, 1, 3, 7, 8, 9, 15, 16, 31, 33, 64, 255, 256, 257, 512,
        ] {
            let v: Vec<f32> = (0..len).map(|_| next()).collect();
            let base: Vec<f32> = (0..len).map(|_| next()).collect();
            for scale in [0.0f32, 1.0, -1.0, 0.37, 1e-8, 1e8, next()] {
                let mut got = base.clone();
                axpy_inplace(&mut got, &v, scale);
                let mut want = base.clone();
                axpy_scalar(&mut want, &v, scale);
                assert_eq!(
                    got.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                    want.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                    "len {len} scale {scale}: vectorized axpy is not bit-identical \
                     to the scalar loop it replaced"
                );
            }
        }
    }
}
