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

//! CPU multi-head attention over a populated [`LayerCache`], shared by every
//! architecture.
//!
//! Each of the four arches used to carry its own copy of this loop, and they
//! had drifted: one was parallelised across tokens and three were not, one
//! supported a sliding window and three assumed causal-to-`pos`, and all four
//! were head-outer — which on a grouped-query model re-reads the same K and V
//! once per query head. Attention is the largest CPU cost in a prefill and the
//! only one that grows quadratically with prompt length, so it is worth having
//! exactly one of it.

use rayon::prelude::*;

use crate::engine::kv_cache::LayerCache;
use crate::engine::tensor;

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
