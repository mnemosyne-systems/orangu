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

//! Multi-head attention over a populated [`LayerCache`], shared by every
//! architecture — [`attention`] picks the GPU kernel or the CPU loop,
//! [`multi_head_attention`] is the CPU loop itself.
//!
//! Each of the arches used to carry its own copy of the CPU loop, and they had
//! drifted: one was parallelised across tokens and three were not, one
//! supported a sliding window and three assumed causal-to-`pos`, and all were
//! head-outer — which on a grouped-query model re-reads the same K and V once
//! per query head. Attention is the largest CPU cost in a prefill and the only
//! one that grows quadratically with prompt length, so it is worth having
//! exactly one of it.
//!
//! **The same argument applies one level up, and had not been acted on.** The
//! choice of *where* attention runs was also per-architecture, and only
//! `gemma.rs` had ever made it: measured across eight models, the other
//! architectures spent **70–78% of their prefill CPU** in the loop below while
//! gemma spent 0.0%, because gemma alone reached for
//! `VulkanBackend::gpu_attention_prefill`. Nothing in that kernel is
//! gemma-specific — it takes head counts, a window and a scale — so the
//! selection belongs here, once, next to the fallback it selects against.

use rayon::prelude::*;

use crate::engine::backend::Backend;
use crate::engine::kv_cache::LayerCache;
use crate::engine::tensor;

/// Shapes and flags [`attention`] needs that are not the query or the cache.
///
/// A struct rather than eleven positional parameters because the GPU and CPU
/// paths must agree on every one of them, and a transposed pair in a call this
/// wide is a wrong answer rather than a compile error.
pub struct Params<'a> {
    pub backend: &'a dyn Backend,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub scale: f32,
    /// Whether a query may attend to positions after its own. The GPU kernel
    /// derives each query's window from this and `n_swa`; the CPU path takes
    /// the `window` closure instead, and the caller is responsible for the two
    /// describing the same thing.
    pub causal: bool,
    /// Sliding-window width, or `0` for none.
    pub n_swa: usize,
    pub start_pos: usize,
    pub n_tokens: usize,
}

/// Token count at or above which a **multi-token** attention call goes to the
/// GPU.
///
/// Not a safety margin — a measured crossover. The GPU path uploads this
/// layer's Q and reads its output back once per layer, and that cost does not
/// shrink with the prompt, so below some width the CPU loop wins outright.
///
/// **Measured, not argued.** This governs the *unfused* path only — where
/// attention is a round trip of its own, with no fused chain to amortise it.
/// Swept against batch width, the CPU loop stays ahead well past the previous
/// value of 32, so that value sent a whole band of widths to the slower path.
///
/// A caller that fuses attention into a longer chain has a different crossover
/// and does not consult this — see `arch::llama`'s own fusion width.
///
/// Sweepable as `ORANGU_ATTENTION_MIN_TOKENS`; `PERF-GAP.md` has the sweep.
const DEFAULT_MIN_GPU_TOKENS: usize = 64;

/// Window length at or above which a **single-token** (decode) attention call
/// goes to the GPU.
///
/// A decode step has one query, so token count is the wrong axis entirely: the
/// work is the *window*, and it grows with the conversation. The CPU loop reads
/// the whole window once per layer per token, which is what makes decode on
/// this path fall away with context while gemma — whose decode attention is on
/// the GPU — stays nearly flat. Below this length the round trip costs more
/// than the walk it saves.
///
/// **Measured, and 256 is measured to be the right value — not merely the
/// original one.** On a 3B model, pinning both paths on and off across context
/// makes the GPU path look flat while the CPU loop falls away, which argues for
/// a low threshold; but lowering it to 64 measured *neutral* there, because the
/// CPU loop's collapse happens further out, where both settings already use the
/// GPU. On a 360M model the same value is a **10.7% regression** at shallow
/// context: a small model's CPU attention loop is cheap (few heads, narrow
/// `kv_dim`) so it stays ahead much longer, while the GPU round trip costs the
/// same as it does for a large one.
///
/// So this threshold is genuinely model-dependent, and 256 is the value that is
/// no worse than the alternative on a large model and clearly better on a small
/// one. Sweepable as `ORANGU_DECODE_ATTENTION_MIN_POS` for a deployment that
/// wants to tune it per model; `PERF-GAP.md` has both sweeps.
const DEFAULT_MIN_GPU_DECODE_POS: usize = 256;

pub fn min_gpu_tokens() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("ORANGU_ATTENTION_MIN_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_MIN_GPU_TOKENS)
    })
}

fn min_gpu_decode_pos() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("ORANGU_DECODE_ATTENTION_MIN_POS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_MIN_GPU_DECODE_POS)
    })
}

/// The window the **GPU kernel** will use for token `t`, in Rust.
///
/// A transcription of `ATTENTION_MULTI_QUERY_SETUP` in
/// `engine::backend::vulkan_shaders`. It exists to be compared against the
/// caller's own closure, so the two descriptions of one window cannot drift
/// apart silently; if that shader's rule changes, this must change with it and
/// the arch tests will say so.
fn derived_window(params: &Params<'_>, t: usize) -> (usize, usize) {
    let pos = params.start_pos + t;
    if !params.causal {
        return if params.n_swa > 0 {
            let half = params.n_swa / 2;
            (
                pos.saturating_sub(half),
                (pos + half).min(params.n_tokens - 1),
            )
        } else {
            (0, params.n_tokens - 1)
        };
    }
    if params.n_swa > 0 {
        (pos.saturating_sub(params.n_swa - 1), pos)
    } else {
        (0, pos)
    }
}

/// Which path [`attention`] took. Returned rather than logged here so a
/// per-layer trace stays the caller's business, and so a test can assert the
/// selection rather than infer it from a timing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ran {
    OnGpu,
    OnCpu,
}

/// Decode attention that leaves its result **on the device**, when the GPU path
/// is the one this shape would have taken anyway.
///
/// Same decision as [`attention`]'s decode branch — the split kernel, the same
/// `ORANGU_DECODE_ATTENTION_MIN_POS` threshold, the same window — differing only
/// in that the `[n_head * head_dim]` output is returned as a buffer rather than
/// copied to the host. `None` means this shape does not take the GPU path and
/// the caller should use [`attention`] as before; it is not a failure.
///
/// Exists so a caller that is about to hand attention's output to another GPU
/// chain does not pay a readback and an upload to do it. The decision stays here
/// rather than in the architecture module, for the reason this module exists at
/// all: there must be exactly one place that decides where attention runs.
pub fn attention_decode_on_device(
    q: &[f32],
    cache: &mut LayerCache,
    params: &Params<'_>,
    window: impl Fn(usize) -> (usize, usize) + Sync,
) -> Option<wgpu::Buffer> {
    if params.n_tokens != 1 || std::env::var_os("ORANGU_NO_ATTN_ON_DEVICE").is_some() {
        return None;
    }
    // The same guard `attention` applies, so the two cannot disagree about the
    // window for a shape they both accept.
    debug_assert!(
        window(0) == derived_window(params, 0),
        "attention window disagrees with causal={} n_swa={} at start_pos={}",
        params.causal,
        params.n_swa,
        params.start_pos,
    );
    let vulkan = params.backend.as_vulkan()?;
    if !vulkan.prefill_attention_enabled() {
        return None;
    }
    let (window_start, window_end) = window(0);
    if window_end < window_start || window_end + 1 - window_start < min_gpu_decode_pos() {
        return None;
    }
    Some(
        vulkan.gpu_attention_split_on_device(crate::engine::backend::vulkan::GpuAttentionInput {
            q,
            cache,
            pos: window_end,
            window_start,
            n_head: params.n_head,
            n_head_kv: params.n_head_kv,
            head_dim: params.head_dim,
            scale: params.scale,
        }),
    )
}

/// Multi-head attention, on whichever processor is faster for this shape.
///
/// `out` is resized and fully written. `cache` must already hold every position
/// the window can reach, including this call's own tokens — both paths have
/// that precondition, since a prompt's later queries attend to its earlier
/// tokens.
///
/// `window(t)` is the CPU path's window and the authority on correctness: the
/// GPU kernel re-derives the same range from `causal`/`n_swa`, and the two are
/// cross-checked against each other for every window shape
/// (`gpu_attention_prefill_matches_cpu_reference_*`). An architecture whose
/// window cannot be expressed as `causal` plus `n_swa` must therefore *not* set
/// those fields to something approximate — it would get a silently different
/// answer on a Vulkan build than on a CPU one.
pub fn attention(
    out: &mut Vec<f32>,
    q: &[f32],
    cache: &mut LayerCache,
    params: &Params<'_>,
    window: impl Fn(usize) -> (usize, usize) + Sync,
) -> Ran {
    let Params {
        n_head,
        n_head_kv,
        head_dim,
        n_tokens,
        ..
    } = *params;

    // The two paths take the window two different ways — the CPU one from the
    // closure, the GPU one re-derived in the shader from `causal`/`n_swa`. If a
    // caller sets those inconsistently with its closure the result is not a
    // crash or a wrong shape; it is a *different answer on a Vulkan build than
    // on a CPU one*, which no cross-check between the two kernels can catch,
    // because both kernels are right. So check the caller instead, on every
    // token, in every debug build — which is every test run.
    debug_assert!(
        (0..n_tokens).all(|t| window(t) == derived_window(params, t)),
        "attention window disagrees with causal={} n_swa={} at start_pos={} \
         n_tokens={n_tokens}: closure {:?} against derived {:?}",
        params.causal,
        params.n_swa,
        params.start_pos,
        (0..n_tokens).map(&window).collect::<Vec<_>>(),
        (0..n_tokens)
            .map(|t| derived_window(params, t))
            .collect::<Vec<_>>(),
    );

    // A single query with a long window is a different kernel from a wide
    // batch with short ones, and the split ("flash-decode") shader is the one
    // shaped for it: it parallelises over *positions* rather than over queries,
    // and at `n_tokens == 1` there is exactly one query to parallelise over.
    if n_tokens == 1
        && let Some(vulkan) = params.backend.as_vulkan()
        && vulkan.prefill_attention_enabled()
    {
        let (window_start, window_end) = window(0);
        if window_end >= window_start && window_end + 1 - window_start >= min_gpu_decode_pos() {
            out.clear();
            out.resize(n_head * head_dim, 0.0);
            let got =
                vulkan.gpu_attention_split(crate::engine::backend::vulkan::GpuAttentionInput {
                    q,
                    cache,
                    pos: window_end,
                    window_start,
                    n_head,
                    n_head_kv,
                    head_dim,
                    scale: params.scale,
                });
            out.copy_from_slice(&got);
            return Ran::OnGpu;
        }
    }

    if n_tokens >= min_gpu_tokens()
        && let Some(vulkan) = params.backend.as_vulkan()
        && vulkan.prefill_attention_enabled()
    {
        *out = vulkan.gpu_attention_prefill(
            q,
            cache,
            params.start_pos,
            n_tokens,
            n_head,
            n_head_kv,
            head_dim,
            params.n_swa,
            params.causal,
            params.scale,
        );
        return Ran::OnGpu;
    }

    out.clear();
    out.resize(n_tokens * n_head * head_dim, 0.0);
    multi_head_attention(
        out,
        q,
        cache,
        n_head,
        n_head.div_ceil(n_head_kv),
        head_dim,
        params.scale,
        window,
    );
    Ran::OnCpu
}

/// Multi-head attention for `n_tokens` queries against `cache`, writing
/// `[n_tokens, n_head * head_dim]` into `out`.
///
/// `window(t)` gives the inclusive cached-position range token `t` attends to
/// — `(0, start_pos + t)` for plain causal attention, something narrower for a
/// sliding-window layer. An empty range (end before start) yields zeros for
/// that token, which is what the `start..=end` loops this replaced did.
///
/// `out` is fully written, so the caller need not pre-zero it.
///
/// # Loop order
///
/// **Position-outer, head-inner.** For a grouped-query model — `n_head_kv = 1`
/// against `n_head = 8` is common, and is what this project's own model does —
/// the head-outer form streams the whole window's K, then the whole window's
/// V, once per query head. That is `group_size` times more memory traffic than
/// the arithmetic needs, and at realistic window sizes the working set is well
/// past L2, so it is paid against L3 every time. Reading each cached position
/// once and fanning it out across the group costs nothing extra in arithmetic.
///
/// The result is **bit-identical** to the head-outer form: every score is an
/// independent dot product, softmax still sees one head's whole window in
/// order, and each output element still accumulates over positions in
/// ascending order. Only the order in which independent values are computed
/// changes.
///
/// # Parallelism
///
/// Across query tokens with rayon: each token reads the (already fully
/// populated, read-only) cache and writes only its own slice. Also bit-exact —
/// there is no cross-token dependency. The per-thread `scores` scratch is
/// carried by `for_each_init` rather than allocated per token, and it is one
/// `[n_head, window]` buffer rather than the per-head `Vec::with_capacity` the
/// arches were doing inside their innermost loop.
#[allow(clippy::too_many_arguments)]
pub fn multi_head_attention(
    out: &mut [f32],
    q: &[f32],
    cache: &LayerCache,
    n_head: usize,
    group_size: usize,
    head_dim: usize,
    scale: f32,
    window: impl Fn(usize) -> (usize, usize) + Sync,
) {
    debug_assert_eq!(out.len(), q.len());
    debug_assert!(group_size > 0);
    let row = n_head * head_dim;
    debug_assert!(row > 0 && out.len().is_multiple_of(row));
    let n_kv = n_head.div_ceil(group_size);

    out.par_chunks_mut(row)
        .enumerate()
        .for_each_init(Vec::<f32>::new, |scores, (t, out_t)| {
            out_t.fill(0.0);
            let (window_start, window_end) = window(t);
            // `saturating_sub`, not `-`: a `start..=end` loop is simply empty
            // when the end precedes the start, and sizing a buffer by the
            // wrapped difference instead would try to allocate ~2^64 floats.
            let n_pos = (window_end + 1).saturating_sub(window_start);
            if n_pos == 0 {
                return;
            }
            let q_t = &q[t * row..(t + 1) * row];

            // Scores as `[n_head, n_pos]`, so each head's row is the
            // contiguous slice `softmax_inplace` needs.
            scores.clear();
            scores.resize(n_head * n_pos, 0.0);
            for (offset, p) in (window_start..=window_end).enumerate() {
                for kv_head in 0..n_kv {
                    let kh = cache.key_at(p, kv_head, head_dim);
                    let h_end = ((kv_head + 1) * group_size).min(n_head);
                    for h in kv_head * group_size..h_end {
                        let qh = &q_t[h * head_dim..(h + 1) * head_dim];
                        scores[h * n_pos + offset] = tensor::dot(qh, kh) * scale;
                    }
                }
            }
            for h in 0..n_head {
                tensor::softmax_inplace(&mut scores[h * n_pos..(h + 1) * n_pos]);
            }
            for (offset, p) in (window_start..=window_end).enumerate() {
                for kv_head in 0..n_kv {
                    let vh = cache.value_at(p, kv_head, head_dim);
                    let h_end = ((kv_head + 1) * group_size).min(n_head);
                    for h in kv_head * group_size..h_end {
                        let weight = scores[h * n_pos + offset];
                        tensor::axpy_inplace(
                            &mut out_t[h * head_dim..(h + 1) * head_dim],
                            vh,
                            weight,
                        );
                    }
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The window rule the shader implements, restated independently of
    /// [`derived_window`] so the two are not the same code checking itself:
    /// this one is written from the *specification* (causal means "up to and
    /// including my own position"; a sliding window of `w` means "the `w`
    /// positions ending at mine"), not transcribed from the WGSL.
    fn spec_window(
        causal: bool,
        n_swa: usize,
        start_pos: usize,
        n_tokens: usize,
        t: usize,
    ) -> (usize, usize) {
        let pos = start_pos + t;
        match (causal, n_swa) {
            (true, 0) => (0, pos),
            (true, w) => (pos + 1 - w.min(pos + 1), pos),
            (false, 0) => (0, n_tokens - 1),
            (false, w) => {
                let half = w / 2;
                (pos.saturating_sub(half), (pos + half).min(n_tokens - 1))
            }
        }
    }

    #[test]
    fn the_derived_window_matches_the_specification_for_every_shape() {
        for &causal in &[true, false] {
            for &n_swa in &[0usize, 1, 2, 5, 8] {
                for &start_pos in &[0usize, 1, 7] {
                    let n_tokens = 9;
                    let params = Params {
                        backend: &crate::engine::backend::CpuBackend,
                        n_head: 2,
                        n_head_kv: 1,
                        head_dim: 4,
                        scale: 1.0,
                        causal,
                        n_swa,
                        start_pos,
                        n_tokens,
                    };
                    for t in 0..n_tokens {
                        assert_eq!(
                            derived_window(&params, t),
                            spec_window(causal, n_swa, start_pos, n_tokens, t),
                            "causal={causal} n_swa={n_swa} start_pos={start_pos} t={t}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "attention window disagrees")]
    fn a_caller_whose_closure_disagrees_with_its_flags_is_caught() {
        // The hazard this guard exists for: `n_swa: 0` says "attend to
        // everything up to my position" while the closure says "the last four".
        // On a CPU build the closure wins; on a Vulkan build the flags do. Both
        // kernels are correct and no cross-check between them can see it.
        let backend = crate::engine::backend::CpuBackend;
        let mut cache = crate::engine::kv_cache::KvCache::new_with_dims(16, &[4]);
        for _ in 0..8 {
            cache.layers[0].push(&[0.0; 4], &[0.0; 4]);
        }
        let params = Params {
            backend: &backend,
            n_head: 1,
            n_head_kv: 1,
            head_dim: 4,
            scale: 1.0,
            causal: true,
            n_swa: 0,
            start_pos: 0,
            n_tokens: 8,
        };
        let mut out = Vec::new();
        attention(&mut out, &[0.0; 32], &mut cache.layers[0], &params, |t| {
            (t.saturating_sub(3), t)
        });
    }

    #[test]
    fn a_matching_caller_passes_the_guard_and_gets_the_cpu_answer() {
        let backend = crate::engine::backend::CpuBackend;
        let mut cache = crate::engine::kv_cache::KvCache::new_with_dims(16, &[4]);
        let mut seed = 1u32;
        let mut next = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 16) as f32 / 32768.0 - 1.0
        };
        for _ in 0..8 {
            let k: Vec<f32> = (0..4).map(|_| next()).collect();
            let v: Vec<f32> = (0..4).map(|_| next()).collect();
            cache.layers[0].push(&k, &v);
        }
        let q: Vec<f32> = (0..32).map(|_| next()).collect();
        let params = Params {
            backend: &backend,
            n_head: 1,
            n_head_kv: 1,
            head_dim: 4,
            scale: 0.5,
            causal: true,
            n_swa: 0,
            start_pos: 0,
            n_tokens: 8,
        };
        let mut got = Vec::new();
        attention(&mut got, &q, &mut cache.layers[0], &params, |t| (0, t));

        let mut want = vec![0f32; 32];
        multi_head_attention(&mut want, &q, &cache.layers[0], 1, 1, 4, 0.5, |t| (0, t));
        assert_eq!(got, want);
    }

    /// The head-outer, token-serial form every architecture carried before
    /// this module existed — kept verbatim as the reference the shared
    /// implementation is held to.
    #[allow(clippy::too_many_arguments)]
    fn reference(
        out: &mut [f32],
        q: &[f32],
        cache: &LayerCache,
        n_head: usize,
        group_size: usize,
        head_dim: usize,
        scale: f32,
        window: impl Fn(usize) -> (usize, usize),
    ) {
        let row = n_head * head_dim;
        out.fill(0.0);
        for t in 0..out.len() / row {
            let (window_start, window_end) = window(t);
            for h in 0..n_head {
                let kv_head = h / group_size;
                let qh = &q[t * row + h * head_dim..t * row + (h + 1) * head_dim];
                let mut scores: Vec<f32> = Vec::new();
                for p in window_start..=window_end {
                    let kh = cache.key_at(p, kv_head, head_dim);
                    scores.push(tensor::dot(qh, kh) * scale);
                }
                if scores.is_empty() {
                    continue;
                }
                tensor::softmax_inplace(&mut scores);
                let o = &mut out[t * row + h * head_dim..t * row + (h + 1) * head_dim];
                for (offset, &weight) in scores.iter().enumerate() {
                    let vh = cache.value_at(window_start + offset, kv_head, head_dim);
                    tensor::axpy_inplace(o, vh, weight);
                }
            }
        }
    }

    fn build(
        n_pos: usize,
        n_head_kv: usize,
        head_dim: usize,
        seed: &mut u32,
    ) -> crate::engine::kv_cache::KvCache {
        let mut next = || {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 17;
            *seed ^= *seed << 5;
            (*seed as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let kv_dim = n_head_kv * head_dim;
        let mut cache = crate::engine::kv_cache::KvCache::new(1, n_pos, kv_dim);
        for _ in 0..n_pos {
            let k: Vec<f32> = (0..kv_dim).map(|_| next()).collect();
            let v: Vec<f32> = (0..kv_dim).map(|_| next()).collect();
            cache.layers[0].push(&k, &v);
        }
        cache
    }

    /// The claim this module rests on: reordering the loops changes nothing at
    /// all, not merely nothing much. Anything short of bit-equality here means
    /// the rewrite altered accumulation order somewhere.
    #[test]
    fn shared_attention_is_bit_identical_to_the_head_outer_form() {
        for &(n_head, n_head_kv) in &[(8usize, 1usize), (8, 2), (4, 4), (6, 3), (1, 1)] {
            for &head_dim in &[16usize, 64, 128] {
                for &n_tokens in &[1usize, 5, 33] {
                    let mut seed = 0x1234_5678u32 ^ (n_head * 31 + head_dim) as u32;
                    let n_pos = n_tokens + 7;
                    let kv = build(n_pos, n_head_kv, head_dim, &mut seed);
                    let cache = &kv.layers[0];
                    let group_size = n_head / n_head_kv;
                    let row = n_head * head_dim;
                    let q: Vec<f32> = (0..n_tokens * row)
                        .map(|i| ((i % 17) as f32 - 8.0) / 8.0)
                        .collect();
                    let start_pos = 3;
                    let win = |t: usize| (0usize, start_pos + t);

                    let mut got = vec![0f32; n_tokens * row];
                    multi_head_attention(
                        &mut got, &q, cache, n_head, group_size, head_dim, 0.125, win,
                    );
                    let mut want = vec![0f32; n_tokens * row];
                    reference(
                        &mut want, &q, cache, n_head, group_size, head_dim, 0.125, win,
                    );
                    assert_eq!(
                        got.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                        want.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                        "n_head {n_head} n_head_kv {n_head_kv} head_dim {head_dim} \
                         n_tokens {n_tokens}: position-outer attention is not \
                         bit-identical to the head-outer form"
                    );
                }
            }
        }
    }

    /// A sliding window, which only the gemma path exercised before — and an
    /// empty one, which the `start..=end` loops handled by simply not running
    /// and which a naive `end + 1 - start` would turn into a huge allocation.
    #[test]
    fn shared_attention_handles_sliding_and_empty_windows() {
        let mut seed = 0xC0FE_u32;
        let (n_head, n_head_kv, head_dim, n_tokens) = (8usize, 1usize, 32usize, 12usize);
        let kv = build(n_tokens + 4, n_head_kv, head_dim, &mut seed);
        let cache = &kv.layers[0];
        let row = n_head * head_dim;
        let q: Vec<f32> = (0..n_tokens * row)
            .map(|i| ((i % 13) as f32 - 6.0) / 6.0)
            .collect();
        let sliding = |t: usize| (t.saturating_sub(3), t);

        let mut got = vec![0f32; n_tokens * row];
        multi_head_attention(&mut got, &q, cache, n_head, 8, head_dim, 0.2, sliding);
        let mut want = vec![0f32; n_tokens * row];
        reference(&mut want, &q, cache, n_head, 8, head_dim, 0.2, sliding);
        assert_eq!(
            got.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|f| f.to_bits()).collect::<Vec<_>>()
        );

        // `end` before `start`: no positions, output stays zero rather than
        // the implementation trying to size a buffer by a wrapped subtraction.
        let mut empty = vec![7f32; n_tokens * row];
        multi_head_attention(&mut empty, &q, cache, n_head, 8, head_dim, 0.2, |_| (5, 4));
        assert!(empty.iter().all(|v| *v == 0.0));
    }
}
