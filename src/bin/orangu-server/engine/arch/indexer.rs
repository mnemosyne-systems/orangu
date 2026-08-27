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

//! The **lightning indexer**: the small separate attention that decides
//! which earlier positions a sparse-attention layer is allowed to look at.
//! Not an architecture itself, the way [`kda`](super::kda) and
//! [`hyper`](super::hyper) are not — it is the DeepSeek sparse attention
//! (DSA) that `engine::arch::glm` and `engine::arch::glm5` both bolt onto
//! their latent attention.
//!
//! A scoring layer projects a small query per head off the *attention
//! query's own* LoRA intermediate, keeps its own narrow per-token key
//! cache, and scores every visible position with a `relu`'d per-head dot
//! product weighted by a per-head scalar read from the layer input. The
//! real attention then attends only the `attention.indexer.top_k`
//! best-scoring positions.
//!
//! Two shapes of it are in use, and the difference is one field:
//!
//! * **Per token** (`glm-dsa`) — one key per position, and the cut is over
//!   positions directly.
//! * **Pooled** ([`KeyPool`], `glm5next`) — positions are grouped into
//!   `attention.indexer.kpool` fixed pools, and one pooled key stands for
//!   each. The pool key is a per-channel convex mix of its members' keys,
//!   `softmax(gate + ape)` over the members, where the gate is a second,
//!   independent projection of each member's hidden state and `ape` is an
//!   intra-pool position bias. The cut is then over whole *pools*, never
//!   over single positions, and the query's own trailing incomplete pool is
//!   always attended on top of the budget. That the cut lands on pool
//!   boundaries is load-bearing: `relu` sends many pools to exactly `0.0`,
//!   and a top-k that split a pool would attend an arbitrary subset of one.
//!
//! Deliberately not carried over from upstream: the Hadamard rotation of
//! the indexer's queries and keys. It is orthonormal and applied to both
//! sides, so it changes no dot product — it exists there to spread
//! magnitude ahead of an `fp8` kernel, and the scoring here is `f32`.

use anyhow::{Context, Result};
use rayon::prelude::*;

use super::top_k_indices;
use crate::engine::backend::Backend;
use crate::engine::kv_cache::{KvCache, LayerCache};
use crate::engine::loader::{LoadedModel, QuantMatrix};
use crate::engine::tensor::{self, RopeParams};

/// LayerNorm over one row: `(x - mean)/sqrt(var + eps) * weight + bias`.
/// ggml's `ggml_norm` followed by the weight and bias `build_norm` applies
/// for `LLM_NORM` — distinct from the RMSNorm nearly every other
/// normalization in these models uses, which neither centers nor shifts.
/// The indexer key is the one place it appears.
pub(crate) fn layer_norm_inplace(x: &mut [f32], weight: &[f32], bias: &[f32], eps: f32) {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let scale = 1.0 / (var + eps).sqrt();
    for ((v, &w), &b) in x.iter_mut().zip(weight.iter()).zip(bias.iter()) {
        *v = (*v - mean) * scale * w + b;
    }
}

/// Pools one block from its members' `(value, score)` rows: a softmax over
/// the members *per feature dimension*, weighted-summed.
/// `values`/`scores` are `[n_members, width]`, with `-inf` scores for the
/// synthetic empty members an overlapping block's first window can have.
pub(crate) fn pool_block(values: &[f32], scores: &[f32], width: usize) -> Vec<f32> {
    let n = values.len() / width;
    debug_assert_eq!(scores.len(), n * width);
    let mut out = vec![0.0; width];
    let mut w = vec![0.0; n];
    for (d, o) in out.iter_mut().enumerate() {
        for (i, wi) in w.iter_mut().enumerate() {
            *wi = scores[i * width + d];
        }
        tensor::softmax_inplace(&mut w);
        *o = (0..n).map(|i| values[i * width + d] * w[i]).sum();
    }
    out
}

/// One pool's key: a per-channel convex mix of its members' keys, weighted
/// by `softmax(gate + ape)` over the members.
///
/// `keys`/`gates` are `[n_members, width]` in position order, and `ape` is
/// `[n_members, width]` — a member's intra-pool position bias, added
/// **before** the softmax. It is the only ordering signal a pooled key
/// carries in a model that rotates nothing, so dropping it would leave the
/// key a permutation-invariant blend of its members.
pub(crate) fn pool_key(keys: &[f32], gates: &[f32], ape: &[f32], width: usize) -> Vec<f32> {
    debug_assert_eq!(gates.len(), keys.len());
    debug_assert_eq!(ape.len(), keys.len());
    let scores: Vec<f32> = gates.iter().zip(ape.iter()).map(|(g, a)| g + a).collect();
    pool_block(keys, &scores, width)
}

/// The indexer hyperparameters every scoring layer of one model shares.
pub(crate) struct IndexerShape {
    /// `attention.indexer.head_count`.
    pub n_head: usize,
    /// `attention.indexer.key_length` — one indexer head's width, which is
    /// also the width of the single shared key.
    pub head_size: usize,
    /// `attention.indexer.top_k` — how many *positions* survive the cut.
    /// With a [`KeyPool`] the budget is spent in whole pools, so the pool
    /// count is `top_k / kpool`.
    pub top_k: usize,
    /// The LayerNorm epsilon for the key norm — `attention.
    /// layer_norm_epsilon`, **not** the model's RMS epsilon, and not
    /// discoverable from any output comparison.
    pub norm_eps: f32,
    /// The rotary parameters, or `None` for a model that rotates nothing.
    /// Unlike the main attention's, the indexer's rotary dimensions are the
    /// *leading* ones of each head.
    pub rope: Option<RopeParams>,
    /// `rope.dimension_count`, whether or not anything is rotated.
    pub rope_dim: usize,
}

impl IndexerShape {
    /// Reads the three required `attention.indexer.*` keys. `rope` is the
    /// model's own rotary parameters, or `None` on a position-free model.
    pub(crate) fn from_gguf(loaded: &LoadedModel, rope: Option<RopeParams>) -> Result<Self> {
        let head_size = loaded
            .metadata_u64("attention.indexer.key_length")
            .context("missing attention.indexer.key_length")? as usize;
        let rope_dim = loaded.config.rope_dim;
        anyhow::ensure!(
            head_size > rope_dim || rope.is_none(),
            "attention.indexer.key_length ({head_size}) must exceed rope.dimension_count ({rope_dim})"
        );
        Ok(Self {
            n_head: loaded
                .metadata_u64("attention.indexer.head_count")
                .context("missing attention.indexer.head_count")? as usize,
            head_size,
            top_k: loaded
                .metadata_u64("attention.indexer.top_k")
                .context("missing attention.indexer.top_k")? as usize,
            norm_eps: loaded
                .metadata_f32("attention.layer_norm_epsilon")
                .unwrap_or(0.0),
            rope,
            rope_dim: if rope.is_some() { rope_dim } else { 0 },
        })
    }

    /// The scale folded into the per-head weights rather than into the far
    /// larger score tensor — `relu(x*s) == s*relu(x)` for positive `s`.
    fn weight_scale(&self) -> f32 {
        1.0 / ((self.head_size * self.n_head) as f32).sqrt()
    }
}

/// The pooled variant's extra weights: one gate projection per token and
/// the intra-pool position bias, plus the cache slot the pooled keys live
/// in.
pub(crate) struct KeyPool {
    /// `[n_embd, indexer.key_length]` — the pooling gate. A **second,
    /// independent** projection of the hidden state, not the indexer key
    /// and not the attention query.
    gate: QuantMatrix,
    /// `[kpool, indexer.key_length]` — added to the gate before the
    /// softmax, selected by a position's offset *within* its pool. With no
    /// rotation anywhere in `glm5next`, this is the only ordering signal in
    /// the pooled key.
    ape: Vec<f32>,
    /// `attention.indexer.kpool` — how many positions one pool holds.
    pub size: usize,
    /// Where the pooled keys live in `KvCache::layers`, one row per
    /// completed pool (stride [`Self::size`]).
    pub cache_slot: usize,
}

/// One scoring layer's lightning indexer.
pub(crate) struct Indexer {
    /// `[q_lora_rank, indexer.head_count * indexer.key_length]` — shares
    /// the attention query's LoRA intermediate.
    q_b: QuantMatrix,
    /// `[n_embd, indexer.key_length]`: one key per token, all heads.
    attn_k: QuantMatrix,
    k_norm_weight: Vec<f32>,
    k_norm_bias: Vec<f32>,
    /// `[n_embd, indexer.head_count]` — the per-head score weights.
    proj: QuantMatrix,
    /// Where this layer's per-token indexer keys live in
    /// `KvCache::layers`. With a [`KeyPool`] the row's *value* half carries
    /// that token's pooling gate, which cannot be recomputed from the key.
    pub cache_slot: usize,
    pool: Option<KeyPool>,
}

/// One layer's per-token indexer projections, run once for a whole chunk.
pub(crate) struct IndexerInputs {
    /// `[n_tokens, n_head * head_size]`, roped when the model rotates.
    q: Vec<f32>,
    /// `[n_tokens, n_head]`, already scaled.
    weights: Vec<f32>,
    /// `[n_tokens, head_size]`, LayerNormed and roped.
    keys: Vec<f32>,
    /// `[n_tokens, head_size]` — the pooling gates, empty without a
    /// [`KeyPool`].
    gates: Vec<f32>,
}

impl Indexer {
    /// Loads a scoring layer's indexer. `pool_slot` is `Some` when the file
    /// declares `attention.indexer.kpool`, and names the cache slot the
    /// pooled keys were reserved in.
    pub(crate) fn load(
        loaded: &LoadedModel,
        layer: usize,
        kpool: Option<usize>,
        shape: &IndexerShape,
        cache_slot: usize,
        pool_slot: usize,
    ) -> Result<Self> {
        let get = |suffix: &str| -> Result<Vec<f32>> {
            let name = format!("blk.{layer}.{suffix}");
            Ok(loaded
                .tensor(&name)
                .with_context(|| format!("loading {name}"))?
                .0)
        };
        let get_matrix = |suffix: &str| -> Result<QuantMatrix> {
            let name = format!("blk.{layer}.{suffix}");
            loaded
                .matrix(&name)
                .with_context(|| format!("loading {name}"))
        };

        let pool = match kpool {
            Some(size) => {
                let ape = get("indexer_compressor_ape.weight")?;
                anyhow::ensure!(
                    ape.len() == size * shape.head_size,
                    "layer {layer}'s indexer_compressor_ape has {} values, not kpool * \
                     indexer.key_length ({})",
                    ape.len(),
                    size * shape.head_size
                );
                Some(KeyPool {
                    gate: get_matrix("indexer_compressor_gate.weight")?,
                    ape,
                    size,
                    cache_slot: pool_slot,
                })
            }
            None => None,
        };

        Ok(Self {
            q_b: get_matrix("indexer.attn_q_b.weight")?,
            attn_k: get_matrix("indexer.attn_k.weight")?,
            k_norm_weight: get("indexer.k_norm.weight")?,
            k_norm_bias: get("indexer.k_norm.bias")?,
            proj: get_matrix("indexer.proj.weight")?,
            cache_slot,
            pool,
        })
    }

    /// This layer's per-token indexer inputs: the query, the pre-scaled
    /// per-head weights, the key to cache, and — with a [`KeyPool`] — the
    /// pooling gate to cache beside it.
    ///
    /// `qr` is the attention query's own LoRA intermediate, which the
    /// indexer query is projected from; `normed` is the layer input.
    pub(crate) fn inputs(
        &self,
        backend: &dyn Backend,
        shape: &IndexerShape,
        qr: &[f32],
        normed: &[f32],
        n_tokens: usize,
        start_pos: usize,
    ) -> IndexerInputs {
        let dim = shape.head_size;
        let mut q = backend.matmul(qr, n_tokens, &self.q_b);
        let mut keys = backend.matmul(normed, n_tokens, &self.attn_k);
        for t in 0..n_tokens {
            let key = &mut keys[t * dim..(t + 1) * dim];
            layer_norm_inplace(key, &self.k_norm_weight, &self.k_norm_bias, shape.norm_eps);
        }
        if let Some(rope) = &shape.rope {
            let rope_dim = shape.rope_dim;
            for t in 0..n_tokens {
                let pos = start_pos + t;
                // Unlike the main attention, the indexer's rotary
                // dimensions are the *leading* ones of each head.
                let q_t = &mut q[t * shape.n_head * dim..(t + 1) * shape.n_head * dim];
                for h in 0..shape.n_head {
                    tensor::rope_apply_params_inplace(
                        &mut q_t[h * dim..h * dim + rope_dim],
                        1,
                        rope_dim,
                        pos,
                        None,
                        rope,
                    );
                }
                tensor::rope_apply_params_inplace(
                    &mut keys[t * dim..t * dim + rope_dim],
                    1,
                    rope_dim,
                    pos,
                    None,
                    rope,
                );
            }
        }
        let mut weights = backend.matmul(normed, n_tokens, &self.proj);
        let scale = shape.weight_scale();
        for w in weights.iter_mut() {
            *w *= scale;
        }
        let gates = match &self.pool {
            Some(pool) => backend.matmul(normed, n_tokens, &pool.gate),
            None => Vec::new(),
        };
        IndexerInputs {
            q,
            weights,
            keys,
            gates,
        }
    }

    /// The positions a query at `pos` may attend, in ascending order.
    ///
    /// Writes token `t`'s own indexer row (and, with a [`KeyPool`], the
    /// pooled key it may complete) into `cache` first: a pool is visible to
    /// the very token that finishes it, and the query attends itself.
    pub(crate) fn select(
        &self,
        shape: &IndexerShape,
        cache: &mut KvCache,
        inputs: &IndexerInputs,
        t: usize,
        pos: usize,
    ) -> Vec<usize> {
        let dim = shape.head_size;
        let key = &inputs.keys[t * dim..(t + 1) * dim];
        let value = match &self.pool {
            Some(_) => &inputs.gates[t * dim..(t + 1) * dim],
            None => key,
        };
        cache.layers[self.cache_slot].push(key, value);

        let q = &inputs.q[t * shape.n_head * dim..(t + 1) * shape.n_head * dim];
        let weights = &inputs.weights[t * shape.n_head..(t + 1) * shape.n_head];

        let Some(pool) = &self.pool else {
            // Scoring can only change the answer once there are more
            // positions than the indexer is allowed to keep: below that its
            // top-k is every visible position, and the mask it produces is
            // the causal mask upstream adds back anyway.
            if pos < shape.top_k {
                return (0..=pos).collect();
            }
            let scores = self.scores(&cache.layers[self.cache_slot], shape, q, weights, pos + 1);
            let mut chosen = top_k_indices(&scores, shape.top_k);
            chosen.sort_unstable();
            return chosen;
        };

        if (pos + 1).is_multiple_of(pool.size) {
            let pooled = pool.compress(&cache.layers[self.cache_slot], dim, pos);
            cache.layers[pool.cache_slot].push(&pooled, &pooled);
        }

        let visible = visible_pools(pos, pool.size);
        let budget = shape.top_k / pool.size;
        if visible <= budget {
            // Every visible pool fits, and the tail is what is left over:
            // together they are exactly the causal window, so there is
            // nothing for a score to change.
            return (0..=pos).collect();
        }

        let scores = self.scores(&cache.layers[pool.cache_slot], shape, q, weights, visible);
        pooled_selection(&scores, pool.size, budget, pos)
    }

    /// The indexer's score for each of the first `visible` rows of `keys`:
    /// a per-head `relu`'d dot product against that row's key, combined
    /// with this token's own per-head weights.
    fn scores(
        &self,
        keys: &LayerCache,
        shape: &IndexerShape,
        q: &[f32],
        weights: &[f32],
        visible: usize,
    ) -> Vec<f32> {
        let dim = shape.head_size;
        (0..visible)
            .into_par_iter()
            .map(|p| {
                let key = keys.key_at(p, 0, dim);
                (0..shape.n_head)
                    .map(|h| tensor::dot(&q[h * dim..(h + 1) * dim], key).max(0.0) * weights[h])
                    .sum()
            })
            .collect()
    }
}

/// How many pools a query at `pos` can see: every pool whose *last* member
/// is at or before it, which is what makes a pool the query sits inside get
/// dropped whole. Upstream tests visibility at the last member for exactly
/// this reason, and pools are position-aligned, so it collapses to this.
fn visible_pools(pos: usize, pool_size: usize) -> usize {
    (pos + 1) / pool_size
}

/// The positions a query at `pos` attends, given a score per visible pool.
///
/// `budget` whole pools by score, expanded to their member positions, plus
/// the query's own trailing incomplete pool unconditionally
/// (`index_kpool_always_select_tail`) — that pool has no pooled key and
/// could never be chosen. Ascending, and a whole number of pools wide
/// before the tail: the cut lands on pool boundaries, never inside one.
fn pooled_selection(scores: &[f32], pool_size: usize, budget: usize, pos: usize) -> Vec<usize> {
    let tail_start = visible_pools(pos, pool_size) * pool_size;
    let mut pools = top_k_indices(scores, budget);
    pools.sort_unstable();

    let mut chosen = Vec::with_capacity(budget * pool_size + pool_size - 1);
    for p in pools {
        chosen.extend(p * pool_size..(p + 1) * pool_size);
    }
    chosen.extend(tail_start..=pos);
    chosen
}

impl KeyPool {
    /// Pools the block that ends at `pos` out of its members' cached
    /// `(key, gate)` rows.
    fn compress(&self, state: &LayerCache, width: usize, pos: usize) -> Vec<f32> {
        let start = pos + 1 - self.size;
        let mut values = vec![0f32; self.size * width];
        let mut gates = vec![0f32; self.size * width];
        for j in 0..self.size {
            values[j * width..(j + 1) * width].copy_from_slice(state.key_at(start + j, 0, width));
            gates[j * width..(j + 1) * width].copy_from_slice(state.value_at(start + j, 0, width));
        }
        pool_key(&values, &gates, &self.ape, width)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// LayerNorm centers and shifts; RMSNorm does neither. Using the wrong
    /// one on the indexer key is a silent accuracy loss, not an error.
    #[test]
    fn layer_norm_centers_scales_and_shifts() {
        let mut x = vec![1.0, 3.0];
        layer_norm_inplace(&mut x, &[1.0, 1.0], &[0.0, 0.0], 0.0);
        // mean 2, variance 1 -> -1, +1
        assert!((x[0] + 1.0).abs() < 1e-5, "{x:?}");
        assert!((x[1] - 1.0).abs() < 1e-5, "{x:?}");

        let mut y = vec![1.0, 3.0];
        layer_norm_inplace(&mut y, &[2.0, 2.0], &[0.5, 0.5], 0.0);
        assert!((y[0] + 1.5).abs() < 1e-5, "{y:?}");
        assert!((y[1] - 2.5).abs() < 1e-5, "{y:?}");
    }

    #[test]
    fn pool_block_softmaxes_each_dimension_over_its_members() {
        // Dimension 0: member 1 wins outright; dimension 1: a tie.
        let values = vec![
            1.0, 1.0, //
            2.0, 3.0, //
        ];
        let scores = vec![
            0.0, 0.0, //
            100.0, 0.0, //
        ];
        let out = pool_block(&values, &scores, 2);
        assert!((out[0] - 2.0).abs() < 1e-5, "{out:?}");
        assert!((out[1] - 2.0).abs() < 1e-5, "{out:?}");
    }

    /// The pooled key must depend on *where* in its pool a position sits.
    /// Without the `ape` the mix is permutation-invariant, and two pools
    /// holding the same keys in a different order would be indistinguishable
    /// — which on a model that rotates nothing is the whole ordering signal
    /// gone, and it still produces perfectly fluent text.
    #[test]
    fn the_pool_key_position_bias_reorders_an_otherwise_symmetric_pool() {
        // Two members, one channel each. Equal gates, so the ape decides.
        let keys = vec![1.0, 5.0];
        let gates = vec![0.0, 0.0];
        let flat = pool_key(&keys, &gates, &[0.0, 0.0], 1);
        assert!((flat[0] - 3.0).abs() < 1e-5, "{flat:?}");

        let first = pool_key(&keys, &gates, &[10.0, 0.0], 1);
        assert!((first[0] - 1.0).abs() < 1e-3, "{first:?}");
        let second = pool_key(&keys, &gates, &[0.0, 10.0], 1);
        assert!((second[0] - 5.0).abs() < 1e-3, "{second:?}");
    }

    /// A pool becomes visible to the very query that completes it, and the
    /// pool a query sits *inside* is not visible at all — its positions
    /// arrive through the always-selected tail instead. Off by one either
    /// way and a query silently attends a pool built partly from its own
    /// future.
    #[test]
    fn a_pool_is_visible_exactly_when_its_last_member_is() {
        // Pools of 4: positions 0..3, 4..7, ...
        assert_eq!(visible_pools(2, 4), 0);
        assert_eq!(
            visible_pools(3, 4),
            1,
            "the query completing a pool sees it"
        );
        assert_eq!(visible_pools(4, 4), 1);
        assert_eq!(visible_pools(7, 4), 2);
    }

    /// The cut is over whole pools. `relu` sends many pools to exactly
    /// `0.0`, so a top-k over *positions* scored by their pool would split
    /// pools apart on the tie and attend an arbitrary subset of one.
    #[test]
    fn the_selection_is_a_whole_number_of_pools_plus_the_tail() {
        // Six visible pools of 4 (positions 0..23), a query at 25, budget 2.
        let scores = vec![0.0, 9.0, 0.0, 7.0, 0.0, 0.0];
        let chosen = pooled_selection(&scores, 4, 2, 25);
        assert_eq!(chosen, vec![4, 5, 6, 7, 12, 13, 14, 15, 24, 25]);
    }

    /// The tail is on top of the budget, not inside it: a model with
    /// `index_kpool_always_select_tail` attends its own trailing partial
    /// pool however the scores fall.
    #[test]
    fn the_query_always_attends_its_own_trailing_partial_pool() {
        let scores = vec![5.0, 0.0];
        // Two visible pools (0..7), query at 9, so positions 8 and 9 are the
        // tail. Budget 1 takes pool 0; the tail comes anyway.
        let chosen = pooled_selection(&scores, 4, 1, 9);
        assert_eq!(chosen, vec![0, 1, 2, 3, 8, 9]);
    }

    /// When the budget covers every visible pool the selection *is* the
    /// causal window — which is what lets the caller skip scoring entirely
    /// below that point without changing the answer.
    #[test]
    fn a_budget_that_covers_every_pool_is_the_dense_causal_window() {
        let pos = 13;
        let scores = vec![0.0, 0.0, 0.0];
        let chosen = pooled_selection(&scores, 4, 3, pos);
        assert_eq!(chosen, (0..=pos).collect::<Vec<usize>>());
    }

    #[test]
    fn pool_block_ignores_members_scored_negative_infinity() {
        let values = vec![
            9.0, 9.0, //
            2.0, 4.0, //
        ];
        let scores = vec![
            f32::NEG_INFINITY,
            f32::NEG_INFINITY, //
            0.0,
            0.0, //
        ];
        let out = pool_block(&values, &scores, 2);
        assert!((out[0] - 2.0).abs() < 1e-6, "{out:?}");
        assert!((out[1] - 4.0).abs() < 1e-6, "{out:?}");
    }
}
