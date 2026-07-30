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
/// compile-time baseline; see `doc/BUILDING.md`), falling back to
/// [`dot_scalar`] everywhere else.
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
    // No runtime check on aarch64: NEON is mandatory in ARMv8-A, the same
    // reason `vecdot`'s dispatch table lists `ISA_BASELINE` as always
    // available. Written as a tail expression rather than an early `return`
    // because on aarch64 the other arm is `cfg`-ed away entirely and this
    // block *is* the tail — which `clippy::needless_return` correctly points
    // out.
    #[cfg(target_arch = "aarch64")]
    {
        dot_neon(a, b)
    }
    #[cfg(not(target_arch = "aarch64"))]
    dot_scalar(a, b)
}

/// Two `float32x4_t` accumulators and `vfmaq_f32`, for the reason [`dot_avx2`]
/// gives — this is the aarch64 half of the same argument.
///
/// It exists because the portable [`dot_scalar`] below **cannot be coaxed into
/// full-width NEON from source**. With the accumulator lanes written out LLVM
/// does vectorize it, but it picks de-interleaving `ld2` loads into four
/// 2-lane registers and separate `fmul`/`fadd` — correct, half the machine's
/// width, and twice the instruction count. Writing the accumulators as two
/// explicit arrays of four changes nothing; the SLP vectorizer canonicalises
/// both spellings to the same output. Verified by `objdump` across three
/// spellings before reaching for intrinsics — see
/// `doc/perf/annotate-dot_scalar.txt`.
///
/// Fused multiply-add, unlike `axpy_inplace` next door: an FMA rounds once
/// where a separate multiply and add round twice, and `dot` is allowed to
/// differ there because its reference is not a specific scalar loop.
#[cfg(target_arch = "aarch64")]
fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = a.len().min(b.len());
    // Safety: every load below is bounded by `i + 8 <= n <= a.len().min(b.len())`,
    // and NEON is unconditionally present on this target.
    unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut i = 0;
        while i + DOT_LANES <= n {
            acc0 = vfmaq_f32(acc0, vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
            acc1 = vfmaq_f32(acc1, vld1q_f32(pa.add(i + 4)), vld1q_f32(pb.add(i + 4)));
            i += DOT_LANES;
        }
        let mut total = vaddvq_f32(vaddq_f32(acc0, acc1));
        while i < n {
            total += a[i] * b[i];
            i += 1;
        }
        total
    }
}

/// Lanes in [`dot_scalar`]'s accumulator. Eight, for the reason spelled out
/// on [`dot_avx2`]: enough independent chains to cover FMA latency, and
/// two NEON `float32x4_t` registers' worth on aarch64.
const DOT_LANES: usize = 8;

/// Default right-hand operands per [`dot_multi`] call, when
/// `engine::attention` is not told otherwise.
///
/// **Four, measured.** Widths interleaved across rounds in one build with width 1
/// as the in-build control (`doc/perf/attn-batch/`), SmolLM2-360M `Q4_K_S`:
///
/// | prompt | width 1 | width 2 | width 4 |
/// |---|---:|---:|---:|
/// | 2237 tok, rounds 2-3 | 12.93 / 12.94 | +1.3% / +1.8% | **+1.9% / +1.9%** |
/// | 4447 tok, 2 rounds | 8.59 / 9.00 | — | **+3.5% / 0.0%** |
///
/// Four was never worse than two and never worse than one; two trails it by
/// 0.2-0.8%, consistently in sign and never by much, so **two is the conservative
/// choice if a narrower-bus part ever shows trouble** — the numbers do not
/// separate them sharply.
///
/// The trade-off runs both ways, which is why this is measured rather than
/// chosen. A wider batch amortizes the shared query load, the accumulator zeroing
/// and the output row's load/store; it also puts that many strided cache rows in
/// flight per thread, and `PERFORMANCE.md` records four-row blocking in the GEMV
/// kernel gaining +10% single-threaded and losing 4-10% at four threads for
/// exactly that reason. Here the two nearly cancel: the instruction shares
/// predicted ~10% end-to-end and ~2% arrived.
///
/// **This box is the worst case for a wide batch** — least bandwidth per core in
/// the range orangu targets — so the stream penalty is at its largest here while
/// the issue saving is machine-independent. Expect a wider batch to pay better on
/// an A76-or-later part, and re-run the sweep there rather than assuming it.
/// `ORANGU_ATTENTION_BATCH` overrides this without a rebuild.
pub const DOT_MULTI: usize = 4;

/// `N` dot products sharing one left operand, each **bit-identical** to [`dot`]
/// of the same pair.
///
/// `engine::attention`'s score loop is one `dot` per (query, head, position),
/// and `perf annotate` on a 2048-token prefill says where that goes: 21.3% of
/// the kernel is the *query* load `ldp q5, q6, [x0, #-0x10]` and 11.0% the key
/// load, against **0.8% for the `fmla`**. Not bandwidth — the L1 miss rate over
/// that window is 1.11%, and 671 M misses in 60 s is 0.72 GB/s against a
/// 3.40 GB/s bus. Load *issue*: two 128-bit loads per two FMAs, at IPC 1.02 on a
/// 3-wide core.
///
/// The query is the same `head_dim` floats for every position in a window, so
/// loading it once per `N` positions removes `(N-1)/N` of the larger of those two
/// lines. Per eight elements this issues 2 query loads and `2N` key loads for
/// `8N` MACs, against `4N` loads for the same work from `N` separate [`dot`]
/// calls. What it costs in exchange is `N` strided rows in flight; see
/// [`DOT_MULTI`].
///
/// **Bit-identical, not merely close**, by design: each output keeps its own pair
/// of accumulators laid out exactly as [`dot_neon`]'s, so lane assignment and
/// summation order match a single `dot` per output. `engine::attention`'s
/// bit-identity test against the head-outer reference therefore still holds with
/// this in the score loop — if it fails, this accumulator structure has drifted
/// from `dot_neon`'s, which is what it is there to catch.
///
/// `#[inline(always)]` because this is called from attention's innermost loop and
/// is big enough that LLVM declines to inline it unprompted; `engine::vecdot`
/// marks its per-block kernels the same way for the same reason.
#[inline(always)]
pub fn dot_multi<const N: usize>(a: &[f32], b: [&[f32]; N]) -> [f32; N] {
    #[cfg(target_arch = "aarch64")]
    {
        dot_multi_neon(a, b)
    }
    // Every other target defers to `dot`, the only way to stay consistent with
    // it without duplicating `dot_avx2`'s four-accumulator structure here. x86-64
    // loses the load sharing until someone writes `dot_multi_avx2`; it loses no
    // accuracy and no correctness.
    #[cfg(not(target_arch = "aarch64"))]
    std::array::from_fn(|j| dot(a, b[j]))
}

/// The aarch64 kernel behind [`dot_multi`]: `2N` accumulators, two per output,
/// mirroring [`dot_neon`]'s pair exactly.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn dot_multi_neon<const N: usize>(a: &[f32], b: [&[f32]; N]) -> [f32; N] {
    use std::arch::aarch64::*;
    let n = b.iter().fold(a.len(), |m, s| m.min(s.len()));
    // Safety: every load is bounded by `i + 8 <= n`, the minimum length over `a`
    // and all of `b`. NEON is unconditional on this target.
    unsafe {
        let mut acc0 = [vdupq_n_f32(0.0); N];
        let mut acc1 = [vdupq_n_f32(0.0); N];
        let pa = a.as_ptr();
        let pb: [*const f32; N] = std::array::from_fn(|j| b[j].as_ptr());
        let mut i = 0;
        while i + DOT_LANES <= n {
            // Loaded once, used `2N` times: the whole point of the function.
            let a0 = vld1q_f32(pa.add(i));
            let a1 = vld1q_f32(pa.add(i + 4));
            for j in 0..N {
                acc0[j] = vfmaq_f32(acc0[j], a0, vld1q_f32(pb[j].add(i)));
                acc1[j] = vfmaq_f32(acc1[j], a1, vld1q_f32(pb[j].add(i + 4)));
            }
            i += DOT_LANES;
        }
        let mut out = [0f32; N];
        for j in 0..N {
            // Same reduction and same scalar tail as `dot_neon`, in the same
            // order, which is what makes each result bit-identical to it.
            let mut total = vaddvq_f32(vaddq_f32(acc0[j], acc1[j]));
            let mut t = i;
            while t < n {
                total += a[t] * b[j][t];
                t += 1;
            }
            out[j] = total;
        }
        out
    }
}

/// The portable path: every target without one of the two vector kernels,
/// and the reference both of them are tested against.
///
/// It is no longer a plain `.sum()`. `dot_avx2` below documents at length why
/// one accumulator is the wrong shape here: a single running total makes the
/// loop a dependency chain of `n` FMAs at ~4 cycles of latency each, when the
/// core can issue two per cycle. That reasoning was never x86-specific, but
/// for most of this file's life the *implementation* was, so every other
/// target — including the aarch64 reference box, which has no AVX2 by
/// construction — silently took the serial form. It matters because of where
/// `dot` is called from: `engine::attention` runs one per (query, head,
/// window position).
///
/// Rust will not reassociate `f32` addition on its own — that is a
/// correctness guarantee, not a missed optimization — so the accumulator
/// lanes have to be written out before LLVM may vectorize at all.
///
/// **The `chunks_exact` pairing is load-bearing, not style.** An earlier
/// version of this loop indexed `a[i + lane]` against a length derived from
/// `a.len().min(b.len())`, and LLVM could not prove those indices in bounds:
/// it emitted a compare-and-branch per element and vectorized nothing, so the
/// rewrite bought eight scalar chains and sixteen bounds checks per iteration
/// and no vectors at all. `chunks_exact` yields slices whose length is known
/// at compile time, which eliminates the checks and lets the vectorizer run.
/// Verified by `objdump`, not assumed — see
/// `doc/perf/annotate-dot_scalar.txt`.
///
/// What it will *not* do is reach full width; that is why [`dot_neon`] exists
/// above rather than relying on this. Do not delete either vector kernel in
/// favour of this one on the strength of it "auto-vectorizing".
///
/// Summation order changes, exactly as it does on the vector paths, and for
/// the same reason it is fine there: `dot` has never been the reference for a
/// bit-exact claim (`axpy_inplace` next door is). Blocked summation is if
/// anything *more* accurate than a running total, not less.
///
/// On aarch64 [`dot_neon`] always wins the dispatch, so outside `cfg(test)`
/// nothing calls this — hence the narrow `allow`. It is kept, and kept
/// correct, because the tests check both vector kernels against it. On x86-64
/// it is live whenever AVX2 or FMA is absent.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    const HALF: usize = DOT_LANES / 2;
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);

    // Two arrays of `HALF`, not one of `DOT_LANES`. Given a single 8-element
    // accumulator LLVM keeps eight lanes but reaches them with de-interleaving
    // `ld2` loads into four 2-lane registers — correct, and half the width the
    // machine offers. Split explicitly, each half maps to one `float32x4_t`
    // with contiguous loads. Same eight chains either way.
    let mut acc0 = [0f32; HALF];
    let mut acc1 = [0f32; HALF];

    let mut ca = a.chunks_exact(DOT_LANES);
    let mut cb = b.chunks_exact(DOT_LANES);
    for (x, y) in ca.by_ref().zip(cb.by_ref()) {
        for lane in 0..HALF {
            acc0[lane] += x[lane] * y[lane];
            acc1[lane] += x[HALF + lane] * y[HALF + lane];
        }
    }

    // Pairwise rather than left-to-right, so the reduction itself does not
    // reintroduce a chain of `DOT_LANES` adds.
    let mut total =
        ((acc0[0] + acc1[0]) + (acc0[1] + acc1[1])) + ((acc0[2] + acc1[2]) + (acc0[3] + acc1[3]));
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        total += x * y;
    }
    total
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

/// `N` value rows accumulated into one output row in a single pass —
/// `out[i] += Σⱼ scale[j] · v[j][i]` — **bit-identical** to `N`
/// [`axpy_inplace`] calls in the same order.
///
/// [`dot_multi`]'s counterpart, for attention's other half. `perf annotate` put
/// 20.4% of the kernel in this half's value load, 4.9% in loading the output row
/// and 8.0% in storing it again, against 0.8% for the arithmetic. Per position
/// the output row is loaded, updated and stored; taking `N` positions at once
/// loads and stores it **once**.
///
/// Bit-identical because the per-element sequence is unchanged:
/// `(((o + s₀v₀) + s₁v₁) + …)`, each multiply and add rounding separately, in
/// ascending row order. That matters more here than for [`dot`]:
/// `axpy_inplace`'s documented reference is *the scalar loop it replaced*, so
/// this inherits that contract rather than getting to pick a new one. No FMA,
/// for the reason spelled out there.
#[inline(always)]
pub fn axpy_multi<const N: usize>(out: &mut [f32], v: [&[f32]; N], scale: [f32; N]) {
    #[cfg(target_arch = "aarch64")]
    {
        axpy_multi_neon(out, v, scale);
    }
    // Elsewhere, literally the calls this batches — trivially bit-identical.
    #[cfg(not(target_arch = "aarch64"))]
    for j in 0..N {
        axpy_inplace(out, v[j], scale[j]);
    }
}

/// The aarch64 kernel behind [`axpy_multi`].
///
/// Intrinsics rather than the vectorizer even though [`axpy_scalar`] *is*
/// auto-vectorized successfully: that loop has one independent output element per
/// iteration and nothing to reassociate, while this has an `N`-deep dependent
/// chain per element whose order must not be touched.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn axpy_multi_neon<const N: usize>(out: &mut [f32], v: [&[f32]; N], scale: [f32; N]) {
    use std::arch::aarch64::*;
    let n = v.iter().fold(out.len(), |m, s| m.min(s.len()));
    // Safety: every access is bounded by `n`, the minimum length across `out` and
    // all of `v`. NEON is unconditional on this target.
    unsafe {
        let s: [float32x4_t; N] = std::array::from_fn(|j| vdupq_n_f32(scale[j]));
        let po = out.as_mut_ptr();
        let pv: [*const f32; N] = std::array::from_fn(|j| v[j].as_ptr());
        let mut i = 0;
        while i + 4 <= n {
            // One load and one store of the output chunk for all `N` rows.
            let mut acc = vld1q_f32(po.add(i));
            for j in 0..N {
                acc = vaddq_f32(acc, vmulq_f32(s[j], vld1q_f32(pv[j].add(i))));
            }
            vst1q_f32(po.add(i), acc);
            i += 4;
        }
        while i < n {
            let mut acc = *po.add(i);
            for j in 0..N {
                acc += scale[j] * *pv[j].add(i);
            }
            *po.add(i) = acc;
            i += 1;
        }
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

/// Which two elements of a head RoPE rotates together — llama.cpp's own
/// `LLAMA_ROPE_TYPE_NEOX` vs `LLAMA_ROPE_TYPE_NORM`, chosen **per
/// architecture** by `llama_model_rope_type` (`src/llama-model.cpp`), not
/// per file and not globally.
///
/// This is not a stylistic detail: the two conventions rotate *different
/// pairs of numbers by the same angles*, so using the wrong one is silently
/// wrong rather than an error. It degrades gracefully in the worst way —
/// position 0 is the identity under both, and small positions rotate by
/// small angles, so a short prompt still produces plausible text while a
/// longer one collapses into repetition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeLayout {
    /// Pairs offset by `rope_dim / 2`: element `i` with element `i +
    /// rope_dim/2` (ggml's `rotate_pairs(n_dims, n_dims/2, ..)`). Used by
    /// `qwen2`/`qwen3`/`qwen3vl`/`phi3`/`falcon` and the whole gemma family.
    Neox,
    /// Pairs of *consecutive* elements: `2p` with `2p+1` (ggml's
    /// `rotate_pairs(n_dims, 1, .., scale = 1)`). Used by `llama` — every
    /// Llama 1/2/3/3.1/3.2 checkpoint — plus `mistral`, `deci`, `granite`,
    /// `cohere` and the other architectures in upstream's `NORM` arm.
    Norm,
}

/// Applies rotary position embedding (RoPE) in place to `x`, one token's
/// `[n_head, head_dim]` block, at absolute position `pos`, using NEOX
/// pairing. Only the leading `rope_dim` elements of each head rotate; any
/// remainder (`head_dim > rope_dim`, e.g. some partial-RoPE models) passes
/// through unchanged.
///
/// See [`RopeLayout`] — this entry point is NEOX-only for the callers whose
/// architecture is NEOX; a `llama`-architecture caller must go through
/// [`rope_apply_layout_inplace`] with [`RopeLayout::Norm`].
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
    rope_apply_mscale_inplace(
        x,
        n_head,
        head_dim,
        rope_dim,
        pos,
        freq_base,
        freq_factors,
        1.0,
    );
}

/// Like [`rope_apply_scaled_inplace`], but additionally scales every rotated
/// pair by `attn_factor` — ggml's own `mscale` in `rope_yarn` (`ggml-cpu/
/// ops.cpp`), which multiplies *both* `cos_theta` and `sin_theta` by it, so
/// the rotated pair comes out `attn_factor` times longer while its angle is
/// unchanged.
///
/// `phi3` (Phi-4-mini) is the architecture here that needs it: its GGUF
/// carries `phi3.rope.scaling.attn_factor` (1.1902381 for Phi-4-mini), which
/// upstream reads into `hparams.rope_attn_factor` and folds into the value
/// it passes as `mscale` (`llama-context.cpp`: `cparams.yarn_attn_factor *=
/// hparams.rope_attn_factor`). The `mscale *= 1 + 0.1*ln(1/freq_scale)`
/// correction next to it in `rope_yarn` is guarded by `ext_factor != 0`,
/// which is YaRN-only (`cparams.yarn_ext_factor` is 0 unless the file
/// declares `rope.scaling.type = yarn`, which `phi3` does not) — so for
/// every model this function currently serves, `attn_factor` reaches the
/// cos/sin unmodified.
///
/// Note this is *not* equivalent to scaling the attention logits: only the
/// leading `rope_dim` elements of each head rotate, so the un-rotated tail
/// (`head_dim > rope_dim`, exactly Phi-4-mini's case: 96 of 128) is
/// deliberately left alone.
#[allow(clippy::too_many_arguments)]
pub fn rope_apply_mscale_inplace(
    x: &mut [f32],
    n_head: usize,
    head_dim: usize,
    rope_dim: usize,
    pos: usize,
    freq_base: f32,
    freq_factors: Option<&[f32]>,
    attn_factor: f32,
) {
    rope_apply_layout_inplace(
        x,
        n_head,
        head_dim,
        rope_dim,
        pos,
        freq_base,
        freq_factors,
        attn_factor,
        RopeLayout::Neox,
    );
}

/// The full RoPE: [`rope_apply_mscale_inplace`] plus an explicit
/// [`RopeLayout`], for the architectures whose pairing isn't NEOX.
///
/// Both layouts use the *same* per-pair angle — pair `p` of `rope_dim/2`
/// rotates by `pos * freq_base^(-2p/rope_dim) / freq_factors[p]` — and
/// differ only in which two elements of the head that angle is applied to,
/// exactly as ggml's own `rotate_pairs` does with its `n_offset`/`scale`
/// arguments.
#[allow(clippy::too_many_arguments)]
pub fn rope_apply_layout_inplace(
    x: &mut [f32],
    n_head: usize,
    head_dim: usize,
    rope_dim: usize,
    pos: usize,
    freq_base: f32,
    freq_factors: Option<&[f32]>,
    attn_factor: f32,
    layout: RopeLayout,
) {
    rope_apply_params_inplace(
        x,
        n_head,
        head_dim,
        pos,
        freq_factors,
        &RopeParams {
            rope_dim,
            freq_base,
            attn_factor,
            layout,
            ..RopeParams::default()
        },
    );
}

/// Everything that shapes a RoPE rotation, so the growing set of knobs
/// travels as one value instead of nine positional arguments.
///
/// The defaults are the *unscaled* rope every non-YaRN model uses:
/// `freq_scale = 1`, `ext_factor = 0`, `attn_factor = 1`. With
/// `ext_factor == 0` the YaRN branch is skipped entirely and the result is
/// bit-identical to the plain rope, which is what keeps every existing
/// caller unchanged.
#[derive(Debug, Clone, Copy)]
pub struct RopeParams {
    pub rope_dim: usize,
    pub freq_base: f32,
    /// `1 / rope.scaling.factor` — how far positions are compressed. `1.0`
    /// for an unscaled model.
    pub freq_scale: f32,
    /// YaRN interpolation strength (upstream's `cparams.yarn_ext_factor`):
    /// `1.0` for a file declaring `rope.scaling.type = yarn`, `0.0` for
    /// everything else, which disables the ramp and the mscale correction.
    pub ext_factor: f32,
    /// Magnitude scale applied to cos/sin (ggml's `mscale`).
    pub attn_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    /// `rope.scaling.original_context_length` — the context the model was
    /// trained at before YaRN stretched it.
    pub n_ctx_orig: usize,
    pub layout: RopeLayout,
}

impl Default for RopeParams {
    fn default() -> Self {
        Self {
            rope_dim: 0,
            freq_base: 10000.0,
            freq_scale: 1.0,
            ext_factor: 0.0,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            n_ctx_orig: 0,
            layout: RopeLayout::Neox,
        }
    }
}

impl RopeParams {
    /// The three per-call YaRN constants — the `[corr_lo, corr_hi]` ramp band
    /// and ggml's `mscale` — that depend on the hyperparameters alone and not
    /// on the position, the head or the pair.
    ///
    /// Split out of [`rope_apply_params_inplace`] so the GPU shaders can be
    /// handed the *same* values rather than re-deriving them: they must agree
    /// bit-for-bit with this function for the cross-check tests to mean
    /// anything, and a second transcription of `ggml_rope_yarn_corr_dims` is
    /// exactly the kind that agrees on the models under test and diverges on
    /// the next one.
    ///
    /// With `ext_factor == 0` (every non-YaRN model) this is
    /// `(0.0, 0.0, attn_factor)`, which disables the ramp and leaves the
    /// magnitude scale as whatever the file asked for.
    pub fn yarn_terms(&self) -> (f32, f32, f32) {
        if self.ext_factor == 0.0 {
            return (0.0, 0.0, self.attn_factor);
        }
        let (lo, hi) = yarn_corr_dims(
            self.rope_dim,
            self.n_ctx_orig,
            self.freq_base,
            self.beta_fast,
            self.beta_slow,
        );
        (
            lo,
            hi,
            self.attn_factor * (1.0 + 0.1 * (1.0 / self.freq_scale).ln()),
        )
    }
}

/// ggml's `ggml_rope_yarn_corr_dim` — the pair index at which a rotation of
/// `n_rot` full turns fits inside the original context.
fn yarn_corr_dim(n_dims: usize, n_ctx_orig: usize, n_rot: f32, base: f32) -> f32 {
    n_dims as f32 * (n_ctx_orig as f32 / (n_rot * 2.0 * std::f32::consts::PI)).ln()
        / (2.0 * base.ln())
}

/// ggml's `ggml_rope_yarn_corr_dims`: the `[low, high]` pair-index band over
/// which YaRN ramps from pure interpolation to pure extrapolation.
fn yarn_corr_dims(
    n_dims: usize,
    n_ctx_orig: usize,
    base: f32,
    beta_fast: f32,
    beta_slow: f32,
) -> (f32, f32) {
    let start = yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, base).floor();
    let end = yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, base).ceil();
    (start.max(0.0), end.min(n_dims as f32 - 1.0))
}

/// The general RoPE: NEOX or NORM pairing, optional per-pair frequency
/// divisors, and optional YaRN interpolation — a direct transcription of
/// ggml's `ggml_rope_cache_init` + `rope_yarn` (`ggml-cpu/ops.cpp`).
pub fn rope_apply_params_inplace(
    x: &mut [f32],
    n_head: usize,
    head_dim: usize,
    pos: usize,
    freq_factors: Option<&[f32]>,
    params: &RopeParams,
) {
    debug_assert_eq!(x.len(), n_head * head_dim);
    let rope_dim = params.rope_dim;
    let half = rope_dim / 2;
    // ggml folds the mscale correction in once per call, not per pair, and
    // only when the YaRN ramp is active — see [`RopeParams::yarn_terms`],
    // which the GPU shaders are handed the results of.
    let (corr_lo, corr_hi, mscale) = params.yarn_terms();

    // The rotation for pair `i` depends on `i` and `pos` — **not** on the head.
    // Computed inside the head loop it is a `powf` and a `sin_cos` per
    // (head, pair) when only `half` of each are distinct: 24x redundant on
    // Llama-3.2-3B's Q, 8x on its K, and identical again for every layer at the
    // same position. It showed up as `__sincosf_fma` in decode profiles.
    //
    // Built once here and read by the head loop below, which keeps that loop's
    // sequential access to `x` — computing per-pair and applying across heads
    // would remove the same redundancy but stride through `x` instead.
    let mut rot = Vec::with_capacity(half);
    for i in 0..half {
        let mut freq = params.freq_base.powf(-2.0 * i as f32 / rope_dim as f32);
        if let Some(ff) = freq_factors {
            freq /= ff[i];
        }
        let theta_extrap = pos as f32 * freq;
        let theta = if params.ext_factor != 0.0 {
            // `rope_yarn_ramp`: 1 at the low end of the band (pure
            // interpolation) falling to 0 above it.
            let y = (i as f32 - corr_lo) / (corr_hi - corr_lo).max(0.001);
            let ramp = 1.0 - y.clamp(0.0, 1.0);
            let mix = ramp * params.ext_factor;
            let theta_interp = params.freq_scale * theta_extrap;
            theta_interp * (1.0 - mix) + theta_extrap * mix
        } else {
            params.freq_scale * theta_extrap
        };
        let (sin, cos) = theta.sin_cos();
        rot.push((sin * mscale, cos * mscale));
    }

    for h in 0..n_head {
        let head = &mut x[h * head_dim..(h + 1) * head_dim];
        for (i, &(sin, cos)) in rot.iter().enumerate() {
            // NEOX rotates `i` against `i + rope_dim/2`; NORM rotates the
            // consecutive pair `2i`/`2i+1`.
            let (lo, hi) = match params.layout {
                RopeLayout::Neox => (i, i + half),
                RopeLayout::Norm => (2 * i, 2 * i + 1),
            };
            let a = head[lo];
            let b = head[hi];
            head[lo] = a * cos - b * sin;
            head[hi] = a * sin + b * cos;
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

    /// [`dot_multi`] and [`axpy_multi`] must equal the calls they batch **to the
    /// bit**, at every width `engine::attention`'s dispatch can select — the
    /// property that keeps that module's own bit-identity test meaningful.
    ///
    /// Not tolerances. Both kernels exist to preserve lane assignment and
    /// summation order while restructuring the loads around them; a tolerance
    /// would admit exactly the drift these catch, and the failure would surface
    /// later as an attention test nobody could explain.
    #[test]
    fn batched_kernels_are_bit_identical_at_every_dispatchable_width() {
        check_dot_multi::<1>();
        check_dot_multi::<2>();
        check_dot_multi::<3>();
        check_dot_multi::<4>();
        check_dot_multi::<8>();
        check_axpy_multi::<1>();
        check_axpy_multi::<2>();
        check_axpy_multi::<3>();
        check_axpy_multi::<4>();
        check_axpy_multi::<8>();
    }

    /// Lengths straddle the 8-element step so the shared scalar tail runs; the
    /// right-hand operands differ from each other so a kernel that paired an
    /// accumulator with the wrong output would fail.
    fn check_dot_multi<const N: usize>() {
        for len in [1, 7, 8, 9, 64, 65, 71, 80, 128, 129, 256] {
            let a: Vec<f32> = (0..len)
                .map(|i| ((i * 13 % 37) as f32 - 18.0) * 0.0625)
                .collect();
            let b: Vec<Vec<f32>> = (0..N)
                .map(|j| {
                    (0..len)
                        .map(|i| ((i * (7 + j) % 31) as f32 - 15.0) * 0.03125)
                        .collect()
                })
                .collect();
            let refs: [&[f32]; N] = std::array::from_fn(|j| b[j].as_slice());
            let got = dot_multi(&a, refs);
            for j in 0..N {
                let want = dot(&a, &b[j]);
                assert_eq!(
                    got[j].to_bits(),
                    want.to_bits(),
                    "N={N} len={len} output {j}: dot_multi {} against dot {want}",
                    got[j]
                );
            }
        }
    }

    /// Scales are distinct and none is 1.0 or 0.0: equal scales would pass even
    /// if a scale were paired with the wrong row, and a 1.0 would hide a missing
    /// multiply. `out` starts non-zero, so dropping the initial load — precisely
    /// what this kernel restructures — would show.
    fn check_axpy_multi<const N: usize>() {
        for len in [1, 3, 4, 5, 63, 64, 65, 128] {
            let scale: [f32; N] = std::array::from_fn(|j| 0.375 - 0.8125 * j as f32);
            let rows: Vec<Vec<f32>> = (0..N)
                .map(|j| {
                    (0..len)
                        .map(|i| ((i * (5 + j) % 23) as f32 - 11.0) * 0.125)
                        .collect()
                })
                .collect();
            let start: Vec<f32> = (0..len).map(|i| ((i % 9) as f32 - 4.0) * 0.5).collect();

            let mut got = start.clone();
            let refs: [&[f32]; N] = std::array::from_fn(|j| rows[j].as_slice());
            axpy_multi(&mut got, refs, scale);

            let mut want = start.clone();
            for j in 0..N {
                axpy_inplace(&mut want, &rows[j], scale[j]);
            }
            assert_eq!(
                got.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                want.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                "N={N} len={len}: axpy_multi is not bit-identical to repeated axpy_inplace"
            );
        }
    }

    /// Whichever vector kernel this target uses must agree with `dot_scalar`
    /// at the widths attention actually runs — `head_dim` is 64/80/128/256 in
    /// the model set, and the window loop calls this once per position.
    ///
    /// Lengths deliberately straddle the 8-element step in both directions so
    /// the scalar tail is exercised: a kernel that dropped the tail entirely
    /// would still pass a multiples-of-8-only test.
    #[test]
    fn dot_agrees_with_the_scalar_reference_at_attention_widths() {
        for len in [64, 65, 71, 80, 127, 128, 129, 256, 260] {
            let a: Vec<f32> = (0..len)
                .map(|i| ((i * 31 % 41) as f32 - 20.0) * 0.03125)
                .collect();
            let b: Vec<f32> = (0..len)
                .map(|i| ((i * 17 % 29) as f32 - 14.0) * 0.0625)
                .collect();
            let want = dot_scalar(&a, &b);
            let got = dot(&a, &b);
            // Only summation order and FMA-vs-separate-rounding differ, so the
            // bound is tight relative to the summed term magnitude.
            let scale: f32 = a.iter().zip(&b).map(|(x, y)| (x * y).abs()).sum();
            assert!(
                (got - want).abs() <= 1e-5 * scale.max(1e-6),
                "len={len}: got {got}, want {want}"
            );
        }
    }

    /// `dot_scalar` accumulates in [`DOT_LANES`] lanes plus a scalar tail, so
    /// it needs a reference of its own — the test above compares `dot`
    /// against it, which on a target without AVX2 is comparing it to itself.
    ///
    /// Lengths straddle the lane count in both directions: shorter than one
    /// block (all tail), exactly one block (no tail), and a block plus a
    /// partial one. An off-by-one in the tail bound would drop or double the
    /// last few products, which nothing else here would catch.
    #[test]
    fn dot_scalar_matches_a_sequential_sum() {
        for len in [0, 1, 3, 7, 8, 9, 15, 16, 17, 64, 129, 256] {
            let a: Vec<f32> = (0..len)
                .map(|i| ((i * 13 % 17) as f32 - 8.0) * 0.125)
                .collect();
            let b: Vec<f32> = (0..len)
                .map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.25)
                .collect();
            let mut want = 0f32;
            for i in 0..len {
                want += a[i] * b[i];
            }
            let got = dot_scalar(&a, &b);
            // These inputs are exact binary fractions and the magnitudes are
            // small, so lane order costs nothing: this can be exact.
            assert_eq!(got, want, "len={len}");
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

    /// `attn_factor` (ggml's `mscale`) scales both `cos_theta` and
    /// `sin_theta`, so a rotated pair keeps its *angle* and comes out
    /// exactly `attn_factor` times longer — the property that distinguishes
    /// it from a plain post-RoPE scale of the whole head, and from rotating
    /// by a different angle.
    #[test]
    fn rope_attn_factor_scales_pair_length_without_rotating_further() {
        let plain = {
            let mut x: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
            rope_apply_mscale_inplace(&mut x, 1, 4, 4, 5, 10000.0, None, 1.0);
            x
        };
        let scaled = {
            let mut x: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
            rope_apply_mscale_inplace(&mut x, 1, 4, 4, 5, 10000.0, None, 1.1902381);
            x
        };
        for (p, s) in plain.iter().zip(&scaled) {
            assert!(
                (p * 1.1902381 - s).abs() < 1e-5,
                "expected {p} * 1.1902381, got {s}"
            );
        }
    }

    /// `RopeLayout::Neox` must stay bit-identical to what the NEOX-only
    /// implementation produced, since every gemma/qwen/phi caller and every
    /// GPU cross-check in `engine::backend::vulkan` is calibrated against
    /// it — the `llama` fix must not move any of them.
    #[test]
    fn rope_neox_layout_rotates_pairs_offset_by_half() {
        let mut x: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        rope_apply_layout_inplace(&mut x, 1, 8, 8, 3, 10000.0, None, 1.0, RopeLayout::Neox);
        // Pair p rotates element p against p+4.
        let mut expected = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        for p in 0..4 {
            let theta = 3.0 * 10000f32.powf(-2.0 * p as f32 / 8.0);
            let (sin, cos) = theta.sin_cos();
            let (a, b) = (expected[p], expected[p + 4]);
            expected[p] = a * cos - b * sin;
            expected[p + 4] = a * sin + b * cos;
        }
        for (got, want) in x.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-6, "{x:?} vs {expected:?}");
        }
    }

    /// `RopeLayout::Norm` rotates *consecutive* pairs (`2p`, `2p+1`) — the
    /// convention `llama`-architecture checkpoints are laid out for. It must
    /// differ from NEOX, or the `llama` fix is a no-op.
    #[test]
    fn rope_norm_layout_rotates_consecutive_pairs() {
        let mut x: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        rope_apply_layout_inplace(&mut x, 1, 8, 8, 3, 10000.0, None, 1.0, RopeLayout::Norm);
        let mut expected = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        for p in 0..4 {
            let theta = 3.0 * 10000f32.powf(-2.0 * p as f32 / 8.0);
            let (sin, cos) = theta.sin_cos();
            let (a, b) = (expected[2 * p], expected[2 * p + 1]);
            expected[2 * p] = a * cos - b * sin;
            expected[2 * p + 1] = a * sin + b * cos;
        }
        for (got, want) in x.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-6, "{x:?} vs {expected:?}");
        }

        let mut neox: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        rope_apply_layout_inplace(&mut neox, 1, 8, 8, 3, 10000.0, None, 1.0, RopeLayout::Neox);
        assert!(
            neox != x,
            "NORM and NEOX must not coincide, or the llama fix does nothing"
        );
    }

    /// `freq_factors` **divides** the frequency, so an entry of 32 rotates
    /// that pair 32x slower. This is the whole mechanism behind a
    /// Llama-3.1/3.2 checkpoint's `rope_freqs.weight`, and dropping it is not
    /// a subtle quality loss — `LlamaModel::rope_freq_factors` documents the
    /// `"I am I am I am I am"` it produces.
    ///
    /// Asserted as the *ratio* between two rotation angles rather than
    /// against a hand-computed constant: the ratio is the invariant, and a
    /// literal here would only re-encode `freq_base.powf(...)` in the test.
    #[test]
    fn rope_freq_factors_divide_the_rotation_frequency() {
        for layout in [RopeLayout::Neox, RopeLayout::Norm] {
            // Pair 1 of a 4-wide rotation. Only that pair's factor differs
            // between the two runs, so any change in it is attributable.
            let angle_of = |ff: Option<&[f32]>| {
                let mut x: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
                // Put the unit into whichever slot pair 1 reads as its "a".
                let a = match layout {
                    RopeLayout::Neox => 1,
                    RopeLayout::Norm => 2,
                };
                x[a] = 1.0;
                rope_apply_layout_inplace(&mut x, 1, 4, 4, 1, 10000.0, ff, 1.0, layout);
                // …and read the "b" slot, which holds sin(theta).
                x[match layout {
                    RopeLayout::Neox => 3,
                    RopeLayout::Norm => 3,
                }]
            };
            let plain = angle_of(None);
            let slowed = angle_of(Some(&[1.0, 32.0]));
            assert!(plain > 0.0 && slowed > 0.0, "{layout:?}: {plain}, {slowed}");
            // Both angles are small enough that sin(t) ~ t, so the ratio of
            // the sin components is the ratio of the frequencies.
            let ratio = plain / slowed;
            assert!(
                (ratio - 32.0).abs() < 1e-2,
                "{layout:?}: expected 32x slower, got {ratio}x"
            );

            // A factor of 1.0 is exactly the no-op, and `None` must agree
            // with it bit for bit rather than merely closely.
            assert_eq!(angle_of(Some(&[1.0, 1.0])), plain, "{layout:?}");
        }
    }

    /// Both layouts are pure rotations of *some* pair, so each leaves the
    /// summed squared magnitude of the rotated block unchanged. Guards
    /// against an indexing slip that drops or double-writes an element.
    #[test]
    fn rope_layouts_preserve_total_energy() {
        for layout in [RopeLayout::Neox, RopeLayout::Norm] {
            let mut x: [f32; 8] = [1.0, -2.0, 3.0, 4.0, -5.0, 6.0, 7.0, -8.0];
            let before: f32 = x.iter().map(|v| v * v).sum();
            rope_apply_layout_inplace(&mut x, 1, 8, 8, 7, 10000.0, None, 1.0, layout);
            let after: f32 = x.iter().map(|v| v * v).sum();
            assert!(
                (before - after).abs() < 1e-3,
                "{layout:?}: {before} vs {after}"
            );
        }
    }

    /// The un-rotated tail of a partial-RoPE head (`head_dim > rope_dim` —
    /// Phi-4-mini rotates 96 of 128) passes through untouched, `attn_factor`
    /// included: ggml's rope copies those channels verbatim rather than
    /// running them through `rope_yarn`, so scaling them too would silently
    /// change every `phi3` attention score.
    #[test]
    fn rope_attn_factor_leaves_the_unrotated_tail_alone() {
        let mut x: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        rope_apply_mscale_inplace(&mut x, 1, 8, 4, 5, 10000.0, None, 1.1902381);
        assert_eq!(&x[4..], &[5.0, 6.0, 7.0, 8.0]);
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
