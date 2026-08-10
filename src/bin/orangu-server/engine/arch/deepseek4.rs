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

//! DeepSeek-V4 (`general.architecture = "deepseek4"`), e.g.
//! `unsloth/DeepSeek-V4-Flash-0731-GGUF`.
//!
//! Four things make this architecture different from every other one in
//! this engine, and all four are load-bearing rather than variations on a
//! theme:
//!
//! - **Hyper-connections.** The residual stream is not one vector per token
//!   but `hyper_connection.count` (4) parallel streams. Each half-layer
//!   (attention, FFN) mixes the streams down to one vector on the way in
//!   (`hc_pre`) and back out to four on the way out (`hc_post`), with the
//!   mixing weights *predicted per token* from the streams themselves — the
//!   out-mix is a doubly-stochastic matrix produced by a Sinkhorn
//!   normalization. See [`Deepseek4Model::hc_pre`]/[`Deepseek4Model::hc_post`].
//! - **One shared key/value vector per token.** `attention.head_count_kv`
//!   is 1 and the value *is* the key (there is no separate `wv`): all 64
//!   query heads attend over the same `attention.key_length`-wide vector,
//!   whose trailing `rope.dimension_count` dimensions carry RoPE. The
//!   attention output is de-RoPEd again (the inverse rotation at the
//!   query's own position) before the grouped output projection, because
//!   the values it averaged were rotated.
//! - **Compressed attention.** `attention.compress_ratios` gives each layer
//!   a ratio: `0` is plain sliding-window attention over the last
//!   `attention.sliding_window` tokens; `128` (HCA) adds one compressed key
//!   per completed 128-token block; `4` (CSA) adds one compressed key per
//!   completed *overlapping* 8-token window, of which only the
//!   `attention.indexer.top_k` highest-scoring are attended — the "lightning
//!   indexer" is a second, narrower compressed cache scored per token to
//!   make that choice. Every layer still sees the raw sliding window; the
//!   compressed keys are what give it the rest of the context.
//! - **Hash-routed experts.** The first `hash_layer_count` layers do not
//!   score their experts at all: the token id indexes a fixed
//!   `ffn_gate_tid2eid` table for the expert *selection*, and only the
//!   weights come from the router.
//!
//! Transcribed from upstream `llama.cpp`'s `src/models/deepseek4.cpp`,
//! `llm_graph_context::build_moe_ffn`, and the DSV4 cache's block planner
//! in `src/llama-kv-cache-dsv4.cpp` (`dsv4_build_comp_plan`, which is what
//! fixes *which* block covers which tokens and when it becomes visible).
//!
//! What is **not** carried over from upstream is its Hadamard rotation of
//! the keys (`llama_mul_mat_hadamard`). That matrix is orthonormal and its
//! own inverse (`ggml_gen_hadamard`: "note: res^2 == I"), and upstream
//! applies it to the query and the key alike, so it leaves every dot
//! product — and therefore every attention weight and the indexer's scores
//! — unchanged. It exists there so a *quantized* K cache stores a rotated
//! basis; this engine's cache is `f32`, so rotating and unrotating would be
//! arithmetic with no effect. Upstream's multi-token-prediction (`nextn`)
//! blocks are also skipped: these checkpoints carry none.

use anyhow::{Context, Result};
use rayon::prelude::*;
use std::sync::Arc;

use super::{ExpertGating, ExpertRouting, ModelForward, attend, rms_norm_rows, top_k_indices};
use crate::engine::backend::Backend;
use crate::engine::kv_cache::KvCache;
use crate::engine::loader::{ExpertQuantMatrix, LoadedModel, ModelConfig, QuantMatrix};
use crate::engine::moe_stats;
use crate::engine::tensor::{self, RopeLayout, RopeParams};

/// `attention.compress_ratios` value for a compressed-sparse-attention
/// layer: 4-token blocks, compressed from an overlapping 8-token window,
/// attended through the lightning indexer's top-k selection. Upstream's
/// `DSV4_CSA_RATIO`.
const CSA_RATIO: usize = 4;
/// `attention.compress_ratios` value for a hierarchical-compressed-
/// attention layer: non-overlapping 128-token blocks, all of them attended.
/// Upstream's `DSV4_HCA_RATIO`.
const HCA_RATIO: usize = 128;

/// One block compressor: the projections that turn each token into a
/// (value, score) pair, and the norm applied to the block they are pooled
/// into. Shared shape between the attention compressor (`attn_compressor_*`)
/// and the lightning indexer's own (`indexer_compressor_*`) — they differ
/// only in width.
struct Compressor {
    wkv: QuantMatrix,
    wgate: QuantMatrix,
    /// `[ratio, width]` — an absolute positional encoding added to the
    /// score, selected by the token's offset *within* its block
    /// (`pos % ratio`).
    ape: Vec<f32>,
    /// RMSNorm weight applied to the pooled block, `[out_width]`.
    norm: Vec<f32>,
    /// Width of one token's compressor row. `2 * out_width` for the
    /// overlapping (ratio-4) compressors, whose first half is used when a
    /// token falls in a block's *previous* window and whose second half is
    /// used when it falls in the current one; `out_width` for the
    /// non-overlapping ratio-128 compressor.
    width: usize,
    /// Width of the compressed key this produces.
    out_width: usize,
}

impl Compressor {
    /// Whether one block pools an overlapping `2 * ratio` window (the
    /// ratio-4 compressors) rather than exactly `ratio` tokens.
    fn overlap(&self) -> bool {
        self.width == 2 * self.out_width
    }
}

/// The lightning indexer of a compressed-sparse-attention layer: a
/// per-token query/weight pair scored against its own compressed cache to
/// choose which of the layer's compressed blocks are worth attending.
struct Indexer {
    /// `[n_embd, indexer.head_count]` — the per-head score weights.
    proj: QuantMatrix,
    /// `[q_lora_rank, indexer.head_count * indexer.key_length]`.
    q_b: QuantMatrix,
    comp: Compressor,
}

/// One half-layer's hyper-connection mixer: the projection that predicts
/// the in-mix, out-mix, and stream-combination weights from the four
/// streams, plus that projection's affine post-scaling.
struct HyperConnection {
    /// `[hc * n_embd, (2 + hc) * hc]`.
    weights: QuantMatrix,
    /// `[(2 + hc) * hc]`.
    base: Vec<f32>,
    /// `[3]` — one scale each for the in-mix, out-mix, and combination
    /// parts of the projection's output.
    scale: Vec<f32>,
}

/// One layer's mixture-of-experts FFN: routed experts plus one always-on
/// shared expert, both SwiGLU with the clamp `swiglu_clamp_exp` /
/// `swiglu_clamp_shexp` ask for.
struct Moe {
    norm: Vec<f32>,
    gate_inp: QuantMatrix,
    /// `exp_probs_b.bias` — added to the probabilities for *selection*
    /// only. `None` on a hash-routed layer, which does not select by score.
    exp_probs_b: Option<Vec<f32>>,
    /// `ffn_gate_tid2eid.weight`, `[n_vocab, n_expert_used]` — the fixed
    /// per-token-id expert selection of a hash-routed layer.
    tid2eid: Option<Vec<i32>>,
    gate_exps: ExpertQuantMatrix,
    down_exps: ExpertQuantMatrix,
    up_exps: ExpertQuantMatrix,
    gate_shexp: QuantMatrix,
    down_shexp: QuantMatrix,
    up_shexp: QuantMatrix,
    clamp_exp: f32,
    clamp_shexp: f32,
}

/// Where one layer's slices of the positional KV cache live. Every entry is
/// an index into `KvCache::layers`; the compressed ones are absent on a
/// layer whose compression ratio is 0.
struct CacheSlots {
    /// Per token: the shared key/value vector (`k` and `v` both, since this
    /// architecture's value *is* its key).
    raw: usize,
    /// Per token: the attention compressor's value (`k`) and score (`v`).
    comp_state: Option<usize>,
    /// Per completed block (stride `ratio`): the compressed key.
    comp_blocks: Option<usize>,
    /// Per token: the indexer compressor's value (`k`) and score (`v`).
    lid_state: Option<usize>,
    /// Per completed block (stride [`CSA_RATIO`]): the indexer's key.
    lid_blocks: Option<usize>,
}

struct Deepseek4Layer {
    attn_norm: Vec<f32>,
    attn_sinks: Vec<f32>,
    wq_a: QuantMatrix,
    attn_q_a_norm: Vec<f32>,
    wq_b: QuantMatrix,
    wkv: QuantMatrix,
    attn_kv_norm: Vec<f32>,
    wo_a: QuantMatrix,
    wo_b: QuantMatrix,
    hc_attn: HyperConnection,
    hc_ffn: HyperConnection,
    /// This layer's `attention.compress_ratios` entry: 0, [`CSA_RATIO`], or
    /// [`HCA_RATIO`].
    ratio: usize,
    compressor: Option<Compressor>,
    indexer: Option<Indexer>,
    /// RoPE for this layer's own query and key: the compressed-rope
    /// parameters on a compressed layer, plain unscaled RoPE on a ratio-0
    /// one (upstream's `use_compress_rope` switch).
    rope: RopeParams,
    slots: CacheSlots,
    ffn: Moe,
}

pub struct Deepseek4Model {
    config: ModelConfig,
    backend: Arc<dyn Backend>,
    tok_embeddings: QuantMatrix,
    output_norm: Vec<f32>,
    output_weight: QuantMatrix,
    /// The final collapse of the four residual streams to one vector.
    hc_head: HyperConnection,
    /// `hyper_connection.count`.
    hc: usize,
    hc_sinkhorn_iters: usize,
    hc_eps: f32,
    n_expert_used: usize,
    routing: ExpertRouting,
    q_lora_rank: usize,
    o_group_count: usize,
    o_lora_rank: usize,
    indexer_n_head: usize,
    indexer_head_size: usize,
    indexer_top_k: usize,
    /// `attention.sliding_window` — how many raw (uncompressed) positions
    /// any layer attends, compressed blocks aside.
    n_swa: usize,
    /// RoPE for the compressed keys and the indexer query, which use
    /// `attention.compress_rope_freq_base` regardless of the layer.
    compress_rope: RopeParams,
    /// `(kv_dim, stride)` per `KvCache::layers` slot, in slot order.
    kv_dims: Vec<(usize, usize)>,
    layers: Vec<Deepseek4Layer>,
}

/// The `n_expert_used` expert ids a hash-routed layer assigns to `token`.
fn selected_experts_for_token(route: &[i32], token: u32, n_expert_used: usize) -> &[i32] {
    let start = token as usize * n_expert_used;
    &route[start..start + n_expert_used]
}

fn hc_affine(x: &mut [f32], scale: f32, base: &[f32]) {
    debug_assert_eq!(x.len(), base.len());
    for (v, &b) in x.iter_mut().zip(base.iter()) {
        *v = *v * scale + b;
    }
}

fn hc_sigmoid_eps_inplace(x: &mut [f32], eps: f32) {
    for v in x.iter_mut() {
        *v = tensor::sigmoid(*v) + eps;
    }
}

fn hc_sigmoid_times_two_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = tensor::sigmoid(*v) * 2.0;
    }
}

/// Sinkhorn-normalizes each token's `[hc, hc]` stream-combination matrix
/// (destination index fastest) into a doubly stochastic one: a softmax over
/// destinations, then alternating column/row normalizations. Upstream's
/// `build_hc_sinkhorn`.
fn hc_sinkhorn_inplace(comb: &mut [f32], hc: usize, n_tokens: usize, eps: f32, iters: usize) {
    debug_assert_eq!(comb.len(), hc * hc * n_tokens);
    for t in 0..n_tokens {
        let block = &mut comb[t * hc * hc..(t + 1) * hc * hc];
        for src in 0..hc {
            let mut max = f32::NEG_INFINITY;
            for dst in 0..hc {
                max = max.max(block[dst + src * hc]);
            }
            let mut sum = 0.0;
            for dst in 0..hc {
                let e = (block[dst + src * hc] - max).exp();
                block[dst + src * hc] = e;
                sum += e;
            }
            for dst in 0..hc {
                block[dst + src * hc] = block[dst + src * hc] / sum.max(f32::MIN_POSITIVE) + eps;
            }
        }

        let normalize_cols = |block: &mut [f32]| {
            for dst in 0..hc {
                let mut sum = eps;
                for src in 0..hc {
                    sum += block[dst + src * hc];
                }
                for src in 0..hc {
                    block[dst + src * hc] /= sum;
                }
            }
        };
        let normalize_rows = |block: &mut [f32]| {
            for src in 0..hc {
                let mut sum = eps;
                for dst in 0..hc {
                    sum += block[dst + src * hc];
                }
                for dst in 0..hc {
                    block[dst + src * hc] /= sum;
                }
            }
        };

        normalize_cols(block);
        for _ in 1..iters {
            normalize_rows(block);
            normalize_cols(block);
        }
    }
}

/// Splits one token's hyper-connection projection into its three parts: the
/// in-mix (`pre`), the out-mix (`post`), and the Sinkhorn-normalized
/// stream-combination matrix (`comb`).
fn hc_pre_parts(
    mixes: &[f32],
    hc: usize,
    eps: f32,
    sinkhorn_iters: usize,
    scale: &[f32],
    base: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    debug_assert_eq!(mixes.len(), hc * (2 + hc));
    let mut pre = mixes[..hc].to_vec();
    hc_affine(&mut pre, scale[0], &base[..hc]);
    hc_sigmoid_eps_inplace(&mut pre, eps);

    let mut post = mixes[hc..2 * hc].to_vec();
    hc_affine(&mut post, scale[1], &base[hc..2 * hc]);
    hc_sigmoid_times_two_inplace(&mut post);

    let mut comb = mixes[2 * hc..].to_vec();
    hc_affine(&mut comb, scale[2], &base[2 * hc..2 * hc + hc * hc]);
    hc_sinkhorn_inplace(&mut comb, hc, 1, eps, sinkhorn_iters);
    (pre, post, comb)
}

/// Pools one completed block from its members' `(value, score)` rows: a
/// softmax over the members *per feature dimension*, weighted-summed.
/// `values`/`scores` are `[n_members, width]`, with `-inf` scores for the
/// synthetic empty members an overlapping block's first window can have.
fn pool_block(values: &[f32], scores: &[f32], width: usize) -> Vec<f32> {
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

/// The raw (uncompressed) positions a query at `pos` attends: the last
/// `n_swa` of them, self included. Upstream's standard sliding-window mask
/// (`llama_hparams::is_masked_swa`) hides a key exactly when
/// `query_pos - key_pos >= n_swa`.
fn raw_window(pos: usize, n_swa: usize) -> std::ops::RangeInclusive<usize> {
    (pos + 1).saturating_sub(n_swa)..=pos
}

/// How many compressed blocks of `ratio` tokens each a query at `pos` can
/// see: every block that is *complete* at or before it. Upstream's
/// `n_visible = (pos + 1)/ratio` — note this includes the block the query
/// itself completes, which is why the compressor state is updated before
/// the softmax reads the cache.
fn visible_blocks(pos: usize, ratio: usize) -> usize {
    (pos + 1) / ratio
}

/// Upstream's `dsv4_rope_attn_factor`: the magnitude scale DSV4 passes to
/// `ggml_rope_ext`, chosen to cancel exactly the `1 + 0.1*ln(1/freq_scale)`
/// correction ggml's YaRN path applies on top of it — so DSV4 gets YaRN's
/// frequency ramp with no magnitude scaling at all.
fn rope_attn_factor(freq_scale: f32, ext_factor: f32) -> f32 {
    if ext_factor == 0.0 {
        return 1.0;
    }
    1.0 / (1.0 + 0.1 * (1.0 / freq_scale).ln())
}

impl Deepseek4Model {
    pub fn load_with_backend(loaded: &LoadedModel, backend: Arc<dyn Backend>) -> Result<Self> {
        let n_layer = loaded.config.n_layer;
        let head_dim = loaded.config.head_dim;
        let tok_embeddings = loaded
            .matrix("token_embd.weight")
            .context("loading token_embd.weight")?;
        let (output_norm, _) = loaded
            .tensor("output_norm.weight")
            .context("loading output_norm.weight")?;
        let output_weight = loaded
            .matrix("output.weight")
            .context("loading output.weight")?;
        let hc = loaded
            .metadata_u64("hyper_connection.count")
            .context("missing hyper_connection.count")? as usize;
        anyhow::ensure!(hc > 0, "hyper_connection.count must be at least 1");
        let hash_layer_count = loaded.metadata_u64("hash_layer_count").unwrap_or(0) as usize;
        let n_expert_used = loaded
            .metadata_u64("expert_used_count")
            .context("missing expert_used_count")? as usize;
        let q_lora_rank = loaded
            .metadata_u64("attention.q_lora_rank")
            .context("missing attention.q_lora_rank")? as usize;
        let o_group_count = loaded
            .metadata_u64("attention.output_group_count")
            .context("missing attention.output_group_count")? as usize;
        let o_lora_rank = loaded
            .metadata_u64("attention.output_lora_rank")
            .context("missing attention.output_lora_rank")? as usize;
        anyhow::ensure!(
            o_group_count > 0 && loaded.config.n_head.is_multiple_of(o_group_count),
            "attention.head_count ({}) must be a multiple of attention.output_group_count ({o_group_count})",
            loaded.config.n_head
        );
        let indexer_n_head = loaded
            .metadata_u64("attention.indexer.head_count")
            .context("missing attention.indexer.head_count")? as usize;
        let indexer_head_size = loaded
            .metadata_u64("attention.indexer.key_length")
            .context("missing attention.indexer.key_length")?
            as usize;
        let indexer_top_k = loaded
            .metadata_u64("attention.indexer.top_k")
            .context("missing attention.indexer.top_k")? as usize;
        let n_swa = loaded
            .metadata_u64("attention.sliding_window")
            .context("missing attention.sliding_window")? as usize;
        anyhow::ensure!(n_swa > 0, "attention.sliding_window must be at least 1");
        let compress_rope_base = loaded
            .metadata_f32("attention.compress_rope_freq_base")
            .unwrap_or(160000.0);
        let hc_sinkhorn_iters = loaded
            .metadata_u64("hyper_connection.sinkhorn_iterations")
            .context("missing hyper_connection.sinkhorn_iterations")?
            as usize;
        anyhow::ensure!(
            hc_sinkhorn_iters > 0,
            "hyper_connection.sinkhorn_iterations must be at least 1"
        );
        let hc_eps = loaded
            .metadata_f32("hyper_connection.epsilon")
            .unwrap_or(1e-6);
        let compress_ratios = loaded
            .metadata_array_f32("attention.compress_ratios")
            .unwrap_or_default();
        anyhow::ensure!(
            compress_ratios.len() >= n_layer,
            "attention.compress_ratios has {} entries, fewer than the {n_layer} layers",
            compress_ratios.len()
        );
        let gating_func = loaded
            .metadata_u64("expert_gating_func")
            .context("missing expert_gating_func")?;
        let expert_weights_scale = loaded.metadata_f32("expert_weights_scale").unwrap_or(1.0);
        let expert_weights_norm = loaded
            .metadata_u64("expert_weights_norm")
            .is_some_and(|v| v != 0);
        let swiglu_clamp_exp = loaded
            .metadata_array_f32("swiglu_clamp_exp")
            .unwrap_or_default();
        let swiglu_clamp_shexp = loaded
            .metadata_array_f32("swiglu_clamp_shexp")
            .unwrap_or_else(|| swiglu_clamp_exp.clone());

        // YaRN, when the file asks for it, applies to this model's own RoPE
        // and to the compressed keys alike; only the frequency base differs
        // between them (upstream's `use_compress_rope` picks the whole set,
        // and every compressed layer is on the compressed side of it).
        let is_yarn = loaded.metadata_string("rope.scaling.type").as_deref() == Some("yarn");
        let freq_scale = loaded
            .metadata_f32("rope.scaling.factor")
            .map_or(1.0, |f| 1.0 / f);
        let ext_factor = if is_yarn { 1.0 } else { 0.0 };
        let beta_fast = loaded
            .metadata_f32("rope.scaling.yarn_beta_fast")
            .unwrap_or(32.0);
        let beta_slow = loaded
            .metadata_f32("rope.scaling.yarn_beta_slow")
            .unwrap_or(1.0);
        let n_ctx_orig = loaded
            .metadata_u64("rope.scaling.original_context_length")
            .unwrap_or(loaded.config.n_ctx_train as u64) as usize;
        let scaled_rope = |freq_base: f32| RopeParams {
            rope_dim: loaded.config.rope_dim,
            freq_base,
            freq_scale,
            ext_factor,
            beta_fast,
            beta_slow,
            attn_factor: rope_attn_factor(freq_scale, ext_factor),
            n_ctx_orig,
            layout: RopeLayout::Norm,
        };
        // A ratio-0 layer ropes with the plain base and no scaling at all —
        // not merely a different base (upstream zeroes `freq_scale`,
        // `ext_factor`, both betas and `n_ctx_orig` together for it).
        let plain_rope = RopeParams {
            rope_dim: loaded.config.rope_dim,
            freq_base: loaded.config.rope_freq_base,
            freq_scale: 1.0,
            ext_factor: 0.0,
            beta_fast: 0.0,
            beta_slow: 0.0,
            attn_factor: 1.0,
            n_ctx_orig: 0,
            layout: RopeLayout::Norm,
        };
        let compress_rope = scaled_rope(compress_rope_base);

        let hyper_connection = |prefix: &str| -> Result<HyperConnection> {
            Ok(HyperConnection {
                weights: loaded
                    .matrix(&format!("{prefix}_fn.weight"))
                    .with_context(|| format!("loading {prefix}_fn.weight"))?,
                base: loaded
                    .tensor(&format!("{prefix}_base.weight"))
                    .with_context(|| format!("loading {prefix}_base.weight"))?
                    .0,
                scale: loaded
                    .tensor(&format!("{prefix}_scale.weight"))
                    .with_context(|| format!("loading {prefix}_scale.weight"))?
                    .0,
            })
        };
        let hc_head = hyper_connection("output_hc")?;

        let mut kv_dims: Vec<(usize, usize)> = Vec::new();
        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let get = |suffix: &str| -> Result<Vec<f32>> {
                let name = format!("blk.{i}.{suffix}");
                Ok(loaded
                    .tensor(&name)
                    .with_context(|| format!("loading {name}"))?
                    .0)
            };
            let get_matrix = |suffix: &str| -> Result<QuantMatrix> {
                let name = format!("blk.{i}.{suffix}");
                loaded
                    .matrix(&name)
                    .with_context(|| format!("loading {name}"))
            };
            let get_expert_matrix = |suffix: &str| -> Result<ExpertQuantMatrix> {
                let name = format!("blk.{i}.{suffix}");
                loaded
                    .expert_matrix(&name)
                    .with_context(|| format!("loading {name}"))
            };
            let compressor = |prefix: &str, out_width: usize| -> Result<Compressor> {
                let wkv = get_matrix(&format!("{prefix}_kv.weight"))?;
                Ok(Compressor {
                    width: wkv.out_dim,
                    out_width,
                    wkv,
                    wgate: get_matrix(&format!("{prefix}_gate.weight"))?,
                    ape: get(&format!("{prefix}_ape.weight"))?,
                    norm: get(&format!("{prefix}_norm.weight"))?,
                })
            };

            let ratio = compress_ratios[i] as usize;
            anyhow::ensure!(
                ratio == 0 || ratio == CSA_RATIO || ratio == HCA_RATIO,
                "layer {i} has compression ratio {ratio}; only 0, {CSA_RATIO} and {HCA_RATIO} are defined"
            );
            let compression = if ratio == 0 {
                None
            } else {
                Some(compressor("attn_compressor", head_dim)?)
            };
            let indexer = if ratio == CSA_RATIO {
                Some(Indexer {
                    proj: get_matrix("indexer.proj.weight")?,
                    q_b: get_matrix("indexer.attn_q_b.weight")?,
                    comp: compressor("indexer_compressor", indexer_head_size)?,
                })
            } else {
                None
            };

            // Slot order is fixed here and never derived again: a cache is
            // built from `kv_dims` alone, so the two must be appended in
            // lockstep.
            let mut push_slot = |dim: usize, stride: usize| {
                kv_dims.push((dim, stride));
                kv_dims.len() - 1
            };
            let slots = CacheSlots {
                raw: push_slot(head_dim, 1),
                comp_state: compression.as_ref().map(|c| push_slot(c.width, 1)),
                comp_blocks: compression.as_ref().map(|c| push_slot(c.out_width, ratio)),
                lid_state: indexer.as_ref().map(|ix| push_slot(ix.comp.width, 1)),
                lid_blocks: indexer
                    .as_ref()
                    .map(|ix| push_slot(ix.comp.out_width, CSA_RATIO)),
            };

            let hashed = i < hash_layer_count;
            layers.push(Deepseek4Layer {
                attn_norm: get("attn_norm.weight")?,
                attn_sinks: get("attn_sinks.weight")?,
                wq_a: get_matrix("attn_q_a.weight")?,
                attn_q_a_norm: get("attn_q_a_norm.weight")?,
                wq_b: get_matrix("attn_q_b.weight")?,
                wkv: get_matrix("attn_kv.weight")?,
                attn_kv_norm: get("attn_kv_a_norm.weight")?,
                wo_a: get_matrix("attn_output_a.weight")?,
                wo_b: get_matrix("attn_output_b.weight")?,
                hc_attn: hyper_connection(&format!("blk.{i}.hc_attn"))?,
                hc_ffn: hyper_connection(&format!("blk.{i}.hc_ffn"))?,
                ratio,
                compressor: compression,
                indexer,
                rope: if ratio == 0 {
                    plain_rope
                } else {
                    compress_rope
                },
                slots,
                ffn: Moe {
                    norm: get("ffn_norm.weight")?,
                    gate_inp: get_matrix("ffn_gate_inp.weight")?,
                    exp_probs_b: if hashed {
                        None
                    } else {
                        Some(get("exp_probs_b.bias")?)
                    },
                    tid2eid: if hashed {
                        Some(
                            loaded
                                .tensor_i32(&format!("blk.{i}.ffn_gate_tid2eid.weight"))
                                .with_context(|| {
                                    format!("loading blk.{i}.ffn_gate_tid2eid.weight")
                                })?
                                .0,
                        )
                    } else {
                        None
                    },
                    gate_exps: get_expert_matrix("ffn_gate_exps.weight")?,
                    down_exps: get_expert_matrix("ffn_down_exps.weight")?,
                    up_exps: get_expert_matrix("ffn_up_exps.weight")?,
                    gate_shexp: get_matrix("ffn_gate_shexp.weight")?,
                    down_shexp: get_matrix("ffn_down_shexp.weight")?,
                    up_shexp: get_matrix("ffn_up_shexp.weight")?,
                    clamp_exp: swiglu_clamp_exp.get(i).copied().unwrap_or(0.0),
                    clamp_shexp: swiglu_clamp_shexp.get(i).copied().unwrap_or(0.0),
                },
            });
            anyhow::ensure!(
                layers[i].wq_b.out_dim == loaded.config.n_head * head_dim,
                "layer {i}'s attn_q_b projects to {} outputs, not head_count * key_length ({})",
                layers[i].wq_b.out_dim,
                loaded.config.n_head * head_dim
            );
        }

        Ok(Self {
            config: loaded.config.clone(),
            backend,
            tok_embeddings,
            output_norm,
            output_weight,
            hc_head,
            hc,
            hc_sinkhorn_iters,
            hc_eps,
            n_expert_used,
            routing: ExpertRouting {
                n_expert_used,
                gating: ExpertGating::from_gguf(gating_func)?,
                weights_norm: expert_weights_norm,
                weights_scale: expert_weights_scale,
            },
            q_lora_rank,
            o_group_count,
            o_lora_rank,
            indexer_n_head,
            indexer_head_size,
            indexer_top_k,
            n_swa,
            compress_rope,
            kv_dims,
            layers,
        })
    }

    /// Runs `tokens` through every layer, returning each token's final
    /// hidden state (`[n_tokens, n_embd]`, collapsed out of the four
    /// residual streams and normed — what the output projection consumes).
    fn run_layers(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<Vec<f32>> {
        let n_tokens = tokens.len();
        let n_embd = self.config.n_embd;
        let hc = self.hc;
        let eps = self.config.rms_eps;

        // Every stream starts as a copy of the token embedding
        // (`ggml_repeat_4d` over the hyper-connection axis).
        let mut x = vec![0f32; n_tokens * hc * n_embd];
        for (t, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            anyhow::ensure!(
                tok < self.config.n_vocab,
                "token id {tok} is out of vocab range"
            );
            let embd = self.tok_embeddings.row(tok);
            for s in 0..hc {
                let at = (t * hc + s) * n_embd;
                x[at..at + n_embd].copy_from_slice(&embd);
            }
        }

        for layer in &self.layers {
            self.forward_layer(layer, cache, &mut x, tokens, start_pos)?;
        }

        let mut out = self.hc_collapse(&self.hc_head, &x, n_tokens, None);
        tensor::rmsnorm_inplace(&mut out, &self.output_norm, n_tokens, n_embd, eps);
        Ok(out)
    }

    fn forward_layer(
        &self,
        layer: &Deepseek4Layer,
        cache: &mut KvCache,
        x: &mut [f32],
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<()> {
        let n_tokens = tokens.len();
        let n_embd = self.config.n_embd;
        let eps = self.config.rms_eps;

        let residual = x.to_vec();
        let mut post = Vec::new();
        let mut comb = Vec::new();
        let mut cur = self.hc_collapse(&layer.hc_attn, x, n_tokens, Some((&mut post, &mut comb)));
        tensor::rmsnorm_inplace(&mut cur, &layer.attn_norm, n_tokens, n_embd, eps);
        let attn = self.attention(layer, cache, &cur, n_tokens, start_pos)?;
        self.hc_expand(x, &attn, &residual, &post, &comb, n_tokens);

        let residual = x.to_vec();
        let mut cur = self.hc_collapse(&layer.hc_ffn, x, n_tokens, Some((&mut post, &mut comb)));
        tensor::rmsnorm_inplace(&mut cur, &layer.ffn.norm, n_tokens, n_embd, eps);
        let ffn = self.moe_ffn(&layer.ffn, &cur, tokens);
        self.hc_expand(x, &ffn, &residual, &post, &comb, n_tokens);
        Ok(())
    }

    /// The hyper-connection in-mix: predicts this half-layer's mixing
    /// weights from the four streams and collapses them to one vector per
    /// token. When `out` is given it also yields the out-mix (`post`) and
    /// stream-combination (`comb`) weights [`Self::hc_expand`] needs; the
    /// final head passes `None`, which is upstream's separate
    /// `build_hc_head` (one scale, no `post`/`comb` at all).
    fn hc_collapse(
        &self,
        mixer: &HyperConnection,
        x: &[f32],
        n_tokens: usize,
        out: Option<(&mut Vec<f32>, &mut Vec<f32>)>,
    ) -> Vec<f32> {
        let n_embd = self.config.n_embd;
        let hc = self.hc;
        let flat_dim = hc * n_embd;

        let mut flat = x.to_vec();
        rms_norm_rows(&mut flat, flat_dim, self.config.rms_eps);
        let mixes = self.backend.matmul(&flat, n_tokens, &mixer.weights);
        let mix_dim = mixer.weights.out_dim;

        let mut cur = vec![0f32; n_tokens * n_embd];
        let (post_out, comb_out) = match out {
            Some((post, comb)) => {
                post.clear();
                post.resize(n_tokens * hc, 0.0);
                comb.clear();
                comb.resize(n_tokens * hc * hc, 0.0);
                (Some(post), Some(comb))
            }
            None => (None, None),
        };
        let mut post_out = post_out;
        let mut comb_out = comb_out;

        for t in 0..n_tokens {
            let mix = &mixes[t * mix_dim..(t + 1) * mix_dim];
            let pre = match (&mut post_out, &mut comb_out) {
                (Some(post), Some(comb)) => {
                    let (pre, p, c) = hc_pre_parts(
                        mix,
                        hc,
                        self.hc_eps,
                        self.hc_sinkhorn_iters,
                        &mixer.scale,
                        &mixer.base,
                    );
                    post[t * hc..(t + 1) * hc].copy_from_slice(&p);
                    comb[t * hc * hc..(t + 1) * hc * hc].copy_from_slice(&c);
                    pre
                }
                _ => {
                    let mut pre = mix.to_vec();
                    hc_affine(&mut pre, mixer.scale[0], &mixer.base);
                    hc_sigmoid_eps_inplace(&mut pre, self.hc_eps);
                    pre
                }
            };
            let dst = &mut cur[t * n_embd..(t + 1) * n_embd];
            for (s, &w) in pre.iter().enumerate() {
                let src = &x[(t * hc + s) * n_embd..(t * hc + s + 1) * n_embd];
                tensor::axpy_inplace(dst, src, w);
            }
        }
        cur
    }

    /// The hyper-connection out-mix: writes this half-layer's output back
    /// into the four streams, each a `post`-weighted copy of the output plus
    /// a `comb`-weighted mix of the streams it came from.
    fn hc_expand(
        &self,
        x: &mut [f32],
        sub_out: &[f32],
        residual: &[f32],
        post: &[f32],
        comb: &[f32],
        n_tokens: usize,
    ) {
        let n_embd = self.config.n_embd;
        let hc = self.hc;
        for t in 0..n_tokens {
            let out_t = &sub_out[t * n_embd..(t + 1) * n_embd];
            for dst in 0..hc {
                let target = &mut x[(t * hc + dst) * n_embd..(t * hc + dst + 1) * n_embd];
                let scale = post[t * hc + dst];
                for (o, &v) in target.iter_mut().zip(out_t.iter()) {
                    *o = v * scale;
                }
                for src in 0..hc {
                    let w = comb[t * hc * hc + src * hc + dst];
                    let from = &residual[(t * hc + src) * n_embd..(t * hc + src + 1) * n_embd];
                    tensor::axpy_inplace(target, from, w);
                }
            }
        }
    }

    /// One layer's attention: the shared-key MLA-style projection, this
    /// layer's compressed-block bookkeeping, the masked softmax over
    /// [sliding window + compressed blocks], and the grouped output
    /// projection.
    fn attention(
        &self,
        layer: &Deepseek4Layer,
        cache: &mut KvCache,
        cur: &[f32],
        n_tokens: usize,
        start_pos: usize,
    ) -> Result<Vec<f32>> {
        let n_embd = self.config.n_embd;
        let eps = self.config.rms_eps;
        let head_dim = self.config.head_dim;
        let n_head = self.config.n_head;
        let rope_dim = self.config.rope_dim;
        let nope = head_dim - rope_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Every projection here is token-independent, so all of them run
        // once for the whole chunk — the per-token loop below is only the
        // parts that depend on position (RoPE, the cache, the softmax).
        let mut qr = self.backend.matmul(cur, n_tokens, &layer.wq_a);
        tensor::rmsnorm_inplace(
            &mut qr,
            &layer.attn_q_a_norm,
            n_tokens,
            self.q_lora_rank,
            eps,
        );
        let mut q = self.backend.matmul(&qr, n_tokens, &layer.wq_b);
        rms_norm_rows(&mut q, head_dim, eps);
        let mut kv = self.backend.matmul(cur, n_tokens, &layer.wkv);
        tensor::rmsnorm_inplace(&mut kv, &layer.attn_kv_norm, n_tokens, head_dim, eps);

        for t in 0..n_tokens {
            let pos = start_pos + t;
            let q_t = &mut q[t * n_head * head_dim..(t + 1) * n_head * head_dim];
            for h in 0..n_head {
                tensor::rope_apply_params_inplace(
                    &mut q_t[h * head_dim + nope..(h + 1) * head_dim],
                    1,
                    rope_dim,
                    pos,
                    None,
                    &layer.rope,
                );
            }
            tensor::rope_apply_params_inplace(
                &mut kv[t * head_dim + nope..(t + 1) * head_dim],
                1,
                rope_dim,
                pos,
                None,
                &layer.rope,
            );
        }

        let comp = layer.compressor.as_ref().map(|c| {
            let values = self.backend.matmul(cur, n_tokens, &c.wkv);
            let mut scores = self.backend.matmul(cur, n_tokens, &c.wgate);
            for t in 0..n_tokens {
                let ape_row = (start_pos + t) % layer.ratio;
                tensor::add_inplace(
                    &mut scores[t * c.width..(t + 1) * c.width],
                    &c.ape[ape_row * c.width..(ape_row + 1) * c.width],
                );
            }
            (values, scores)
        });
        let lid = layer.indexer.as_ref().map(|ix| {
            let values = self.backend.matmul(cur, n_tokens, &ix.comp.wkv);
            let mut scores = self.backend.matmul(cur, n_tokens, &ix.comp.wgate);
            for t in 0..n_tokens {
                let ape_row = (start_pos + t) % CSA_RATIO;
                tensor::add_inplace(
                    &mut scores[t * ix.comp.width..(t + 1) * ix.comp.width],
                    &ix.comp.ape[ape_row * ix.comp.width..(ape_row + 1) * ix.comp.width],
                );
            }
            // The indexer's query shares the attention query's LoRA
            // intermediate, and ropes with the compressed-key parameters
            // whatever the layer's own are.
            let mut iq = self.backend.matmul(&qr, n_tokens, &ix.q_b);
            let inope = self.indexer_head_size - rope_dim;
            for t in 0..n_tokens {
                let pos = start_pos + t;
                let row = &mut iq[t * self.indexer_n_head * self.indexer_head_size
                    ..(t + 1) * self.indexer_n_head * self.indexer_head_size];
                for h in 0..self.indexer_n_head {
                    let head = &mut row
                        [h * self.indexer_head_size + inope..(h + 1) * self.indexer_head_size];
                    tensor::rope_apply_params_inplace(
                        head,
                        1,
                        rope_dim,
                        pos,
                        None,
                        &self.compress_rope,
                    );
                }
            }
            let mut weights = self.backend.matmul(cur, n_tokens, &ix.proj);
            let iscale = 1.0 / ((self.indexer_head_size * self.indexer_n_head) as f32).sqrt();
            for w in weights.iter_mut() {
                *w *= iscale;
            }
            (values, scores, iq, weights)
        });

        let mut attn_out = vec![0f32; n_tokens * n_head * head_dim];
        for t in 0..n_tokens {
            let pos = start_pos + t;

            cache.layers[layer.slots.raw].push(
                &kv[t * head_dim..(t + 1) * head_dim],
                &kv[t * head_dim..(t + 1) * head_dim],
            );

            // Compressor state first, then the block it may complete: the
            // block covering this token is visible *to* this token
            // (upstream's `n_visible = (pos + 1)/ratio`), so it has to be in
            // the cache before the softmax below reads it.
            if let (Some(c), Some((values, scores))) = (layer.compressor.as_ref(), comp.as_ref()) {
                let state = layer
                    .slots
                    .comp_state
                    .expect("compressor implies a state slot");
                cache.layers[state].push(
                    &values[t * c.width..(t + 1) * c.width],
                    &scores[t * c.width..(t + 1) * c.width],
                );
                if (pos + 1).is_multiple_of(layer.ratio) {
                    let block = self.compress_block(c, &cache.layers[state], pos, layer.ratio, eps);
                    cache.layers[layer.slots.comp_blocks.expect("compressor implies blocks")]
                        .push(&block, &block);
                }
            }
            if let (Some(ix), Some((values, scores, _, _))) = (layer.indexer.as_ref(), lid.as_ref())
            {
                let state = layer.slots.lid_state.expect("indexer implies a state slot");
                cache.layers[state].push(
                    &values[t * ix.comp.width..(t + 1) * ix.comp.width],
                    &scores[t * ix.comp.width..(t + 1) * ix.comp.width],
                );
                if (pos + 1).is_multiple_of(CSA_RATIO) {
                    let block =
                        self.compress_block(&ix.comp, &cache.layers[state], pos, CSA_RATIO, eps);
                    cache.layers[layer.slots.lid_blocks.expect("indexer implies blocks")]
                        .push(&block, &block);
                }
            }

            // The key set: the raw sliding window, plus this layer's
            // visible compressed blocks (all of them for HCA, the indexer's
            // top-k for CSA).
            let window = raw_window(pos, self.n_swa);
            let n_raw = window.end() + 1 - window.start();
            let blocks: Vec<usize> = match (layer.ratio, layer.slots.comp_blocks) {
                (0, _) | (_, None) => Vec::new(),
                (ratio, Some(_)) => {
                    let visible = visible_blocks(pos, ratio);
                    match (layer.indexer.as_ref(), lid.as_ref()) {
                        (Some(_), Some((_, _, iq, weights))) if visible > 0 => {
                            let lid_blocks = layer
                                .slots
                                .lid_blocks
                                .expect("indexer implies indexer blocks");
                            let scores = self.indexer_scores(
                                &cache.layers[lid_blocks],
                                &iq[t * self.indexer_n_head * self.indexer_head_size
                                    ..(t + 1) * self.indexer_n_head * self.indexer_head_size],
                                &weights[t * self.indexer_n_head..(t + 1) * self.indexer_n_head],
                                visible,
                            );
                            let mut chosen = top_k_indices(&scores, self.indexer_top_k);
                            chosen.sort_unstable();
                            chosen
                        }
                        _ => (0..visible).collect(),
                    }
                }
            };

            let mut keys = vec![0f32; (n_raw + blocks.len()) * head_dim];
            {
                let raw = &cache.layers[layer.slots.raw];
                for (i, p) in window.enumerate() {
                    keys[i * head_dim..(i + 1) * head_dim]
                        .copy_from_slice(raw.key_at(p, 0, head_dim));
                }
            }
            if let Some(slot) = layer.slots.comp_blocks {
                let comp_blocks = &cache.layers[slot];
                for (i, &b) in blocks.iter().enumerate() {
                    let at = (n_raw + i) * head_dim;
                    keys[at..at + head_dim].copy_from_slice(comp_blocks.key_at(b, 0, head_dim));
                }
            }

            let q_t = &q[t * n_head * head_dim..(t + 1) * n_head * head_dim];
            let heads: Vec<Vec<f32>> = (0..n_head)
                .into_par_iter()
                .map(|h| {
                    attend(
                        &q_t[h * head_dim..(h + 1) * head_dim],
                        &keys,
                        head_dim,
                        head_dim,
                        scale,
                        Some(layer.attn_sinks[h]),
                    )
                })
                .collect();
            let dst = &mut attn_out[t * n_head * head_dim..(t + 1) * n_head * head_dim];
            for (h, head) in heads.iter().enumerate() {
                dst[h * head_dim..(h + 1) * head_dim].copy_from_slice(head);
                // The values that were just averaged carry the *keys'*
                // rotations, so the output is de-RoPEd at the query's own
                // position before the output projection sees it
                // (`ggml_rope_ext_back`).
                tensor::rope_apply_params_inverse_inplace(
                    &mut dst[h * head_dim + nope..(h + 1) * head_dim],
                    1,
                    rope_dim,
                    pos,
                    None,
                    &layer.rope,
                );
            }
        }

        // Grouped output projection: each group of `n_head / o_groups`
        // heads goes through its own low-rank `wo_a` slice, then the
        // concatenation goes through the shared `wo_b`.
        let o_group_dim = n_head * head_dim / self.o_group_count;
        let mut oa = vec![0f32; n_tokens * self.o_group_count * self.o_lora_rank];
        for g in 0..self.o_group_count {
            let mut group = vec![0f32; n_tokens * o_group_dim];
            for t in 0..n_tokens {
                let at = t * n_head * head_dim + g * o_group_dim;
                group[t * o_group_dim..(t + 1) * o_group_dim]
                    .copy_from_slice(&attn_out[at..at + o_group_dim]);
            }
            let projected = self.backend.matmul(
                &group,
                n_tokens,
                &layer.wo_a.rows(g * self.o_lora_rank, self.o_lora_rank),
            );
            for t in 0..n_tokens {
                let at = t * self.o_group_count * self.o_lora_rank + g * self.o_lora_rank;
                oa[at..at + self.o_lora_rank]
                    .copy_from_slice(&projected[t * self.o_lora_rank..(t + 1) * self.o_lora_rank]);
            }
        }
        let out = self.backend.matmul(&oa, n_tokens, &layer.wo_b);
        debug_assert_eq!(out.len(), n_tokens * n_embd);
        Ok(out)
    }

    /// Pools the block that ends at `pos` out of its members' compressor
    /// rows, norms it, and ropes its positional tail — one compressed key.
    ///
    /// An overlapping compressor's block spans `2 * ratio` tokens: the
    /// `ratio` before it (read from the *first* half of each row) and its
    /// own `ratio` (read from the second half). The tokens before position
    /// zero are the synthetic all-zero / `-inf`-scored row upstream appends
    /// for exactly this case, which contributes nothing to the pool.
    fn compress_block(
        &self,
        c: &Compressor,
        state: &crate::engine::kv_cache::LayerCache,
        pos: usize,
        ratio: usize,
        eps: f32,
    ) -> Vec<f32> {
        let width = c.out_width;
        let start = pos + 1 - ratio;
        let n = if c.overlap() { 2 * ratio } else { ratio };
        let mut values = vec![0f32; n * width];
        let mut scores = vec![f32::NEG_INFINITY; n * width];
        let mut take = |slot: usize, p: usize, half: usize| {
            let row_k = state.key_at(p, 0, c.width);
            let row_v = state.value_at(p, 0, c.width);
            values[slot * width..(slot + 1) * width]
                .copy_from_slice(&row_k[half * width..(half + 1) * width]);
            scores[slot * width..(slot + 1) * width]
                .copy_from_slice(&row_v[half * width..(half + 1) * width]);
        };
        if c.overlap() {
            for j in 0..ratio {
                if let Some(p) = (start + j).checked_sub(ratio) {
                    take(j, p, 0);
                }
                take(ratio + j, start + j, 1);
            }
        } else {
            for j in 0..ratio {
                take(j, start + j, 0);
            }
        }

        let mut block = pool_block(&values, &scores, width);
        tensor::rmsnorm_inplace(&mut block, &c.norm, 1, width, eps);
        let nope = width - self.config.rope_dim;
        tensor::rope_apply_params_inplace(
            &mut block[nope..],
            1,
            self.config.rope_dim,
            start,
            None,
            &self.compress_rope,
        );
        block
    }

    /// The lightning indexer's score for each of the `visible` compressed
    /// blocks: a per-head `relu`'d dot product against the block's indexer
    /// key, combined with this token's own per-head weights.
    fn indexer_scores(
        &self,
        blocks: &crate::engine::kv_cache::LayerCache,
        q: &[f32],
        weights: &[f32],
        visible: usize,
    ) -> Vec<f32> {
        let dim = self.indexer_head_size;
        (0..visible)
            .into_par_iter()
            .map(|b| {
                let key = blocks.key_at(b, 0, dim);
                (0..self.indexer_n_head)
                    .map(|h| tensor::dot(&q[h * dim..(h + 1) * dim], key).max(0.0) * weights[h])
                    .sum()
            })
            .collect()
    }

    /// Routed experts plus the always-on shared expert. The routed
    /// selection is either the hash table's (the first `hash_layer_count`
    /// layers) or the top `expert_used_count` by biased probability; the
    /// weights always come from the *unbiased* probabilities.
    fn moe_ffn(&self, ffn: &Moe, cur: &[f32], tokens: &[u32]) -> Vec<f32> {
        let n_embd = self.config.n_embd;
        let mut out = vec![0f32; tokens.len() * n_embd];
        let mut experts =
            moe_stats::LayerRecorder::for_tensors(&[&ffn.gate_exps, &ffn.up_exps, &ffn.down_exps]);
        // Route the whole batch first, so the union can be taken before any
        // expert's weights are read — see `super::evaluate_routed_experts`.
        // The hash-routed layers route from the token id rather than the
        // logits, which changes nothing here: a selection is a selection.
        let mut selection: Vec<Vec<(usize, f32)>> = tokens
            .iter()
            .enumerate()
            .map(|(t, &token)| {
                let x_t = &cur[t * n_embd..(t + 1) * n_embd];
                let logits = self.backend.matmul(x_t, 1, &ffn.gate_inp);
                let hashed = ffn
                    .tid2eid
                    .as_ref()
                    .map(|route| selected_experts_for_token(route, token, self.n_expert_used));
                let (selected, weights) =
                    self.routing
                        .route(&logits, ffn.exp_probs_b.as_deref(), hashed);
                selected.into_iter().zip(weights).collect()
            })
            .collect();
        // Trim to the expert budget *before* anything is recorded or
        // read: the counters should describe the work actually done,
        // and a dropped expert's weights must never be fetched.
        super::apply_expert_budget(&mut selection, &ffn.gate_exps);
        for picks in &selection {
            picks.iter().for_each(|&(e, _)| experts.select(e));
        }

        // The GPU expert path batches the three projections across experts —
        // see `super::evaluate_routed_experts_batched`.
        let contribs = if super::gpu_experts() && self.backend.as_wgpu().is_some() {
            super::evaluate_routed_experts_batched(
                self.backend.as_ref(),
                &selection,
                cur,
                n_embd,
                &ffn.gate_exps,
                &ffn.up_exps,
                &ffn.down_exps,
                |gate, up| swiglu(gate, up, ffn.clamp_exp),
            )
        } else {
            super::evaluate_routed_experts(&selection, |expert, members| {
                let inputs: Vec<&[f32]> = members
                    .iter()
                    .map(|&(t, _)| &cur[t * n_embd..(t + 1) * n_embd])
                    .collect();
                let gate = super::project_expert(
                    self.backend.as_ref(),
                    &ffn.gate_exps,
                    expert,
                    0,
                    ffn.gate_exps.out_dim,
                    &inputs,
                );
                let up = super::project_expert(
                    self.backend.as_ref(),
                    &ffn.up_exps,
                    expert,
                    0,
                    ffn.up_exps.out_dim,
                    &inputs,
                );
                let hidden: Vec<Vec<f32>> = gate
                    .into_iter()
                    .zip(up)
                    .map(|(gate, up)| swiglu(&gate, &up, ffn.clamp_exp))
                    .collect();
                let hidden_refs: Vec<&[f32]> = hidden.iter().map(Vec::as_slice).collect();
                super::project_expert(
                    self.backend.as_ref(),
                    &ffn.down_exps,
                    expert,
                    0,
                    ffn.down_exps.out_dim,
                    &hidden_refs,
                )
                .into_iter()
                .zip(members)
                .map(|(mut contribution, &(_, weight))| {
                    contribution.iter_mut().for_each(|v| *v *= weight);
                    contribution
                })
                .collect()
            })
        };
        experts.loaded_once_per_distinct_expert();

        for t in 0..tokens.len() {
            let x_t = &cur[t * n_embd..(t + 1) * n_embd];
            let shexp_h = swiglu(
                &self.backend.matmul(x_t, 1, &ffn.gate_shexp),
                &self.backend.matmul(x_t, 1, &ffn.up_shexp),
                ffn.clamp_shexp,
            );
            let shexp = self.backend.matmul(&shexp_h, 1, &ffn.down_shexp);

            let dst = &mut out[t * n_embd..(t + 1) * n_embd];
            dst.copy_from_slice(&shexp);
            for contrib in &contribs[t] {
                tensor::add_inplace(dst, contrib);
            }
        }
        experts.commit(tokens.len());
        out
    }
}

/// SwiGLU with DeepSeek-V4's clamp: the up projection is clamped to
/// `[-limit, limit]` and the gate to `(-inf, limit]` **before** the SiLU,
/// not after it — upstream branches on the architecture for exactly this
/// (`arch == LLM_ARCH_DEEPSEEK4` in `build_moe_ffn`/`build_ffn`), and the
/// other branch clamps the activation instead. A limit of zero (no
/// `swiglu_clamp_*` key) means plain SwiGLU.
fn swiglu(gate: &[f32], up: &[f32], limit: f32) -> Vec<f32> {
    debug_assert_eq!(gate.len(), up.len());
    if limit <= 1e-6 {
        return gate
            .iter()
            .zip(up.iter())
            .map(|(&g, &u)| tensor::silu(g) * u)
            .collect();
    }
    gate.iter()
        .zip(up.iter())
        .map(|(&g, &u)| tensor::silu(g.min(limit)) * u.clamp(-limit, limit))
        .collect()
}

impl ModelForward for Deepseek4Model {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn new_kv_cache(&self, capacity: usize) -> KvCache {
        KvCache::new_with_strided_dims(capacity, &self.kv_dims)
    }

    fn forward(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        _slot_id: usize,
    ) -> Result<Vec<f32>> {
        anyhow::ensure!(!tokens.is_empty(), "forward called with no tokens");
        let hidden = self.run_layers(cache, tokens, start_pos)?;
        let n_embd = self.config.n_embd;
        let last = &hidden[(tokens.len() - 1) * n_embd..];
        Ok(self.backend.matmul(last, 1, &self.output_weight))
    }

    /// Every input position's next-token logits, for the speculative
    /// verify step. Sound here because this architecture's whole cache is
    /// positional — the compressed blocks included, each slot advancing
    /// once per `ratio` tokens — so `KvCache::truncate` rolls a rejected
    /// draft's tail back exactly, leaving the state the accepted prefix
    /// alone would have produced.
    fn forward_all_logits(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        _slot_id: usize,
    ) -> Result<Vec<Vec<f32>>> {
        anyhow::ensure!(
            !tokens.is_empty(),
            "forward_all_logits called with no tokens"
        );
        let hidden = self.run_layers(cache, tokens, start_pos)?;
        let n_embd = self.config.n_embd;
        Ok((0..tokens.len())
            .map(|t| {
                self.backend.matmul(
                    &hidden[t * n_embd..(t + 1) * n_embd],
                    1,
                    &self.output_weight,
                )
            })
            .collect())
    }

    fn forward_hidden_states(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        anyhow::ensure!(
            !tokens.is_empty(),
            "forward_hidden_states called with no tokens"
        );
        let mut cache = self.new_kv_cache(tokens.len());
        self.run_layers(&mut cache, tokens, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CSA_RATIO, HCA_RATIO, hc_affine, hc_pre_parts, hc_sigmoid_eps_inplace,
        hc_sigmoid_times_two_inplace, hc_sinkhorn_inplace, pool_block, raw_window,
        rope_attn_factor, selected_experts_for_token, swiglu, visible_blocks,
    };
    use crate::engine::arch::rms_norm_rows;
    use crate::engine::arch::{ExpertGating, ExpertRouting};

    #[test]
    fn selected_experts_for_token_reads_vocab_rows() {
        let route = vec![
            10, 11, 12, 13, 14, 15, //
            20, 21, 22, 23, 24, 25, //
        ];
        assert_eq!(
            selected_experts_for_token(&route, 0, 6),
            &[10, 11, 12, 13, 14, 15]
        );
        assert_eq!(
            selected_experts_for_token(&route, 1, 6),
            &[20, 21, 22, 23, 24, 25]
        );
    }

    /// `expert_gating_func = 4`, this architecture's own gating: the
    /// shared router has to compute `sqrt(softplus(x))` for it.
    #[test]
    fn the_sqrt_softplus_gating_matches_the_reference_formula() {
        let routing = ExpertRouting {
            n_expert_used: 1,
            gating: ExpertGating::SqrtSoftplus,
            weights_norm: false,
            weights_scale: 1.0,
        };
        let (_, weights) = routing.route(&[0.0], None, None);
        assert!((weights[0] - std::f32::consts::LN_2.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn hc_affine_scales_then_biases() {
        let mut x = vec![1.0, 2.0, 3.0];
        hc_affine(&mut x, 0.5, &[10.0, 20.0, 30.0]);
        assert_eq!(x, vec![10.5, 21.0, 31.5]);
    }

    #[test]
    fn hc_sigmoid_variants_match_reference_formulas() {
        let mut pre = vec![0.0, 1.0];
        hc_sigmoid_eps_inplace(&mut pre, 1e-6);
        assert!((pre[0] - 0.500001).abs() < 1e-6);
        assert!((pre[1] - (crate::engine::tensor::sigmoid(1.0) + 1e-6)).abs() < 1e-6);

        let mut post = vec![0.0, 1.0];
        hc_sigmoid_times_two_inplace(&mut post);
        assert!((post[0] - 1.0).abs() < 1e-6);
        assert!((post[1] - crate::engine::tensor::sigmoid(1.0) * 2.0).abs() < 1e-6);
    }

    #[test]
    fn hc_sinkhorn_produces_positive_column_normalized_weights() {
        let mut comb = vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
        ];
        hc_sinkhorn_inplace(&mut comb, 2, 2, 1e-6, 3);
        for t in 0..2 {
            let block = &comb[t * 4..(t + 1) * 4];
            for dst in 0..2 {
                let col = block[dst] + block[dst + 2];
                assert!((col - 1.0).abs() < 1e-4, "{col}");
            }
            assert!(block.iter().all(|v| *v > 0.0));
        }
    }

    /// This architecture's own routing settings through the shared
    /// router: sqrt-softplus probabilities, a hash-table selection, then
    /// renormalize and scale.
    #[test]
    fn deepseek4_router_weights_follow_sqrt_softplus_then_normalize_and_scale() {
        let routing = ExpertRouting {
            n_expert_used: 2,
            gating: ExpertGating::SqrtSoftplus,
            weights_norm: true,
            weights_scale: 1.5,
        };
        let logits = vec![0.0, 1.0, 2.0];
        let (selected, weights) = routing.route(&logits, None, Some(&[2, 0]));
        assert_eq!(selected, vec![2, 0]);
        let sqrt_softplus = |x: f32| crate::engine::tensor::softplus(x).sqrt();
        let raw0 = sqrt_softplus(2.0);
        let raw1 = sqrt_softplus(0.0);
        let denom = raw0 + raw1;
        assert!((weights[0] - (raw0 / denom) * 1.5).abs() < 1e-6);
        assert!((weights[1] - (raw1 / denom) * 1.5).abs() < 1e-6);
    }

    #[test]
    fn hc_pre_parts_split_and_normalize_the_mix_vector() {
        let mixes = vec![
            0.0, 1.0, //
            2.0, 3.0, //
            4.0, 5.0, 6.0, 7.0, //
        ];
        let scale = vec![1.0, 1.0, 1.0];
        let base = vec![0.0; 2 + 2 + 4];
        let (pre, post, comb) = hc_pre_parts(&mixes, 2, 1e-6, 3, &scale, &base);
        assert_eq!(pre.len(), 2);
        assert_eq!(post.len(), 2);
        assert_eq!(comb.len(), 4);
        assert!(pre.iter().all(|v| *v > 0.0));
        assert!(post.iter().all(|v| *v > 0.0));
        let col0 = comb[0] + comb[2];
        let col1 = comb[1] + comb[3];
        assert!((col0 - 1.0).abs() < 1e-4, "{col0}");
        assert!((col1 - 1.0).abs() < 1e-4, "{col1}");
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
        assert!((out[0] - 2.0).abs() < 1e-4, "{out:?}");
        assert!((out[1] - 2.0).abs() < 1e-4, "{out:?}");
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

    #[test]
    fn rms_norm_rows_normalizes_each_row_without_a_weight() {
        let mut x = vec![3.0, 4.0, 0.0, 0.0];
        rms_norm_rows(&mut x, 2, 0.0);
        let rms = ((9.0f32 + 16.0) / 2.0).sqrt();
        assert!((x[0] - 3.0 / rms).abs() < 1e-5);
        assert!((x[1] - 4.0 / rms).abs() < 1e-5);
    }

    #[test]
    fn the_rope_magnitude_scale_cancels_ggmls_yarn_correction() {
        // What `RopeParams::yarn_terms` will multiply back in.
        let freq_scale = 1.0 / 16.0;
        let attn_factor = rope_attn_factor(freq_scale, 1.0);
        let mscale = attn_factor * (1.0 + 0.1 * (1.0f32 / freq_scale).ln());
        assert!((mscale - 1.0).abs() < 1e-6, "{mscale}");
        // With YaRN off there is no correction to cancel.
        assert_eq!(rope_attn_factor(1.0, 0.0), 1.0);
    }

    #[test]
    fn swiglu_clamps_the_gate_before_the_activation_and_the_up_symmetrically() {
        // gate above the limit is clamped to it, then SiLU'd; up is
        // clamped on both sides.
        let out = swiglu(&[100.0, -100.0], &[100.0, 1.0], 10.0);
        assert!(
            (out[0] - crate::engine::tensor::silu(10.0) * 10.0).abs() < 1e-4,
            "{out:?}"
        );
        assert!(
            (out[1] - crate::engine::tensor::silu(-100.0)).abs() < 1e-6,
            "{out:?}"
        );
        // No limit: plain SwiGLU.
        let plain = swiglu(&[100.0], &[100.0], 0.0);
        assert!((plain[0] - crate::engine::tensor::silu(100.0) * 100.0).abs() < 1e-2);
    }

    #[test]
    fn the_two_compression_ratios_are_the_ones_upstream_defines() {
        assert_eq!(CSA_RATIO, 4);
        assert_eq!(HCA_RATIO, 128);
    }

    #[test]
    fn the_raw_window_holds_the_last_n_swa_positions_and_never_underflows() {
        // Before the window has filled: everything from zero.
        assert_eq!(raw_window(2, 128).collect::<Vec<_>>(), vec![0, 1, 2]);
        // Once it has: exactly `n_swa` positions, ending at the query.
        let full = raw_window(200, 128).collect::<Vec<_>>();
        assert_eq!(full.len(), 128);
        assert_eq!(*full.first().unwrap(), 73);
        assert_eq!(*full.last().unwrap(), 200);
        // The oldest kept key is `n_swa - 1` back, matching upstream's
        // "masked when query - key >= n_swa".
        assert_eq!(200 - 73, 127);
    }

    #[test]
    fn a_block_becomes_visible_to_the_token_that_completes_it() {
        // Ratio 4: tokens 0..=3 make block 0, which position 3 can see.
        assert_eq!(visible_blocks(2, CSA_RATIO), 0);
        assert_eq!(visible_blocks(3, CSA_RATIO), 1);
        assert_eq!(visible_blocks(4, CSA_RATIO), 1);
        assert_eq!(visible_blocks(7, CSA_RATIO), 2);
        // Ratio 128, the same rule at the other scale.
        assert_eq!(visible_blocks(126, HCA_RATIO), 0);
        assert_eq!(visible_blocks(127, HCA_RATIO), 1);
        assert_eq!(visible_blocks(255, HCA_RATIO), 2);
    }

    /// The block a position completes has to be the one a
    /// `stride`-strided cache slot stores it in, or a rolled-back or
    /// prefix-reused cache would read another block's key.
    #[test]
    fn block_visibility_agrees_with_the_strided_slots_own_row_count() {
        for ratio in [CSA_RATIO, HCA_RATIO] {
            for pos in 0..600usize {
                let rows_pushed = (0..=pos).filter(|p| (p + 1) % ratio == 0).count();
                assert_eq!(rows_pushed, visible_blocks(pos, ratio), "pos {pos}");
                // And that is what `LayerCache`'s stride arithmetic keeps
                // when the cache is rolled back to `pos + 1` tokens.
                assert_eq!((pos + 1) / ratio, visible_blocks(pos, ratio));
            }
        }
    }
}
