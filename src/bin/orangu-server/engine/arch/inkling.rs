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

//! Inkling (`general.architecture = "inkling"`, e.g.
//! `unsloth/Inkling-Small-GGUF`) — a sparse mixture-of-experts decoder that
//! carries **no rotary embedding at all**. Position reaches attention two
//! other ways, and both are new here:
//!
//! - **A learned relative-position bias.** Every layer projects its input
//!   to a small per-head vector (`attn_r`, `[n_embd, n_head * d_rel]`) and
//!   mixes it against a per-layer bank (`attn_rel_proj`, `[d_rel,
//!   rel_extent]`) into one additive logit per query/key *distance*. Keys
//!   further back than the bank is wide contribute no bias at all, which is
//!   how the model still attends a long prefix with a short bank.
//! - **A causal depthwise short convolution.** Four of them per layer, of
//!   width `inkling.shortconv_kernel`: on the raw key and value
//!   projections, and on the outputs of the attention and FFN sub-layers
//!   before their residual adds. Each is `conv(x) + x` — the residual is
//!   *inside* the convolution — and each carries a rolling window of the
//!   previous `kernel - 1` inputs, which is state that outlives a decode
//!   step exactly as a linear-attention layer's does. They therefore live
//!   in `KvCache::recurrent`, the same place `engine::arch::qwen35moe`
//!   keeps its conv history, and reuse its `RecurrentLayerState::
//!   conv_step`.
//!
//! Everything else is composed from pieces already here:
//!
//! - **Alternating local/global attention**, from a per-layer
//!   `attention.sliding_window_pattern` (`true` = sliding window), as
//!   `gemma`'s per-layer array does. The two kinds also use *different*
//!   relative-bias widths — `inkling.rel_extent_swa` and
//!   `inkling.rel_extent` — which is checked against each layer's own
//!   `attn_rel_proj` shape at load time (see [`InklingModel::
//!   load_with_backend`]), so a misread pattern is a load error rather
//!   than quietly wrong output.
//! - **Per-head query/key RMSNorm** (`attn_q_norm`/`attn_k_norm`,
//!   `[head_dim]`), as `gemma`/`qwen3` have. The attention scale is
//!   `1 / head_dim`, **not** `1 / sqrt(head_dim)`: the norm already fixes
//!   each head's magnitude, and upstream folds the second factor of
//!   `sqrt(head_dim)` into the same divisor.
//! - **Length-scaled attention on the full-attention layers only.** Past
//!   `inkling.log_scaling_n_floor` positions, every score in a global
//!   layer is multiplied by `1 + alpha * ln((pos + 1) / floor)` — an
//!   attention temperature that grows with the context, like
//!   `engine::arch::mistral`'s constant one but read per query position.
//! - **Sigmoid-routed experts with a correction bias**, as
//!   `engine::arch::glm` and `deepseek4` have (`ffn_gate_inp`,
//!   `exp_probs_b.bias`, `expert_weights_scale`), plus `dense_block_count`
//!   leading dense layers. Two differences from every other MoE here: the
//!   router emits `n_expert + n_expert_shared` logits, the trailing ones
//!   gating the shared experts, and the selected routed weights are
//!   normalized **together with** the shared ones rather than among
//!   themselves. Both shared experts live in one stacked
//!   `ffn_*_shexp` tensor, read through the same `ExpertQuantMatrix` view
//!   the routed experts use.
//! - **A logit divisor**, `inkling.logit_scale_denom`, applied to the final
//!   hidden state before the output projection — `engine::arch::muse`'s
//!   `logit_scale` with the multiply the other way round.
//!
//! The vocabulary is padded: `inkling.unpadded_vocab_size` names how many
//! of `token_embd`'s rows are real tokens, and the rest are masked out of
//! the logits so nothing can sample one (see
//! [`InklingModel::mask_padding_logits`]).
//!
//! CPU-orchestrated, like `engine::arch::glm`, `kimi3` and `qwen35moe`:
//! every matmul still dispatches to the `Backend` (GPU included), but the
//! relative bias is a per-`(token, head)` additive term no attention kernel
//! here can express, so the attention loop itself is the host's.
//!
//! Audio and image input are out of scope, as they are for every
//! architecture here: the `mmproj-*.gguf` shipped beside these weights is a
//! separate model this engine does not load, and the `<|audio|>` embedding
//! table is not part of the text GGUF.

use anyhow::{Context, Result};
use rayon::prelude::*;
use std::sync::Arc;

use super::{ExpertGating, ModelForward};
use crate::engine::backend::{Backend, MatmulOp};
use crate::engine::kv_cache::{KvCache, RecurrentLayerState};
use crate::engine::loader::{ExpertQuantMatrix, LoadedModel, ModelConfig, QuantMatrix};
use crate::engine::moe_stats;
use crate::engine::tensor;

/// Which of a layer's four short convolutions a `KvCache::recurrent` slot
/// holds. One layer owns [`CONV_PER_LAYER`] consecutive slots, so slot
/// `il * CONV_PER_LAYER + CONV_K` is layer `il`'s key convolution.
///
/// Named rather than numbered at the call sites because all four have the
/// same shape and the same type: swapping two of them is not a compile
/// error, not a crash, and not obviously wrong output.
const CONV_K: usize = 0;
const CONV_V: usize = 1;
const CONV_ATTN: usize = 2;
const CONV_MLP: usize = 3;
const CONV_PER_LAYER: usize = 4;

/// The sliding-window pattern when the file sets no
/// `attention.sliding_window_pattern` array — `(il + 1) % 6 != 0`, so every
/// sixth layer attends the whole prefix.
///
/// A fallback, not a guess to rely on: the released checkpoints all ship
/// the array, and a file that does not is cross-checked against its own
/// `attn_rel_proj` widths before anything runs (see
/// [`InklingModel::load_with_backend`]).
const DEFAULT_SWA_PERIOD: usize = 6;

/// A layer's feed-forward half — dense on the leading
/// `inkling.dense_block_count` layers, routed experts on the rest.
enum Ffn {
    Dense {
        gate: QuantMatrix,
        up: QuantMatrix,
        down: QuantMatrix,
    },
    Moe(Box<Moe>),
}

/// One MoE layer's routing and expert weights.
struct Moe {
    /// `[n_embd, n_expert + n_expert_shared]` — the trailing rows gate the
    /// shared experts, which is why this is wider than the expert count.
    gate_inp: QuantMatrix,
    /// `exp_probs_b.bias` — steers the expert *selection* only, never the
    /// weights, exactly as in `engine::arch::glm`.
    exp_probs_b: Vec<f32>,
    gate_exps: ExpertQuantMatrix,
    up_exps: ExpertQuantMatrix,
    down_exps: ExpertQuantMatrix,
    /// The shared experts, stacked the same way the routed ones are
    /// (`[in, out, n_expert_shared]`) — every token runs all of them.
    gate_shexp: ExpertQuantMatrix,
    up_shexp: ExpertQuantMatrix,
    down_shexp: ExpertQuantMatrix,
}

struct InklingLayer {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    wq: QuantMatrix,
    wk: QuantMatrix,
    wv: QuantMatrix,
    /// `attn_r`, `[n_embd, n_head * d_rel]` — this token's per-head mixing
    /// coefficients for the relative-bias bank below.
    wr: QuantMatrix,
    wo: QuantMatrix,
    /// `[head_dim]`, the same vector for every head.
    attn_q_norm: Vec<f32>,
    attn_k_norm: Vec<f32>,
    /// `attn_rel_proj`, `[d_rel, rel_extent]` in GGUF order, so
    /// `rel_proj[d * rel_extent + distance]`. Dequantized once at load:
    /// it is a few tens of kilobytes and every token reads all of it.
    rel_proj: Vec<f32>,
    /// How many distinct query/key distances the bank covers — this
    /// layer's own `rel_extent`, which differs between the sliding-window
    /// and full-attention layers.
    rel_extent: usize,
    conv_k: Vec<f32>,
    conv_v: Vec<f32>,
    conv_attn: Vec<f32>,
    conv_mlp: Vec<f32>,
    /// Whether this layer attends a sliding window rather than the whole
    /// prefix. Also selects [`InklingLayer::rel_extent`] and decides
    /// whether the length scaling applies.
    is_swa: bool,
    /// KV heads for *this* layer (`attention.head_count_kv` is a per-layer
    /// array). `n_head / n_head_kv` must divide evenly.
    n_head_kv: usize,
    /// `ffn_gscale.weight` — one scalar multiplying this layer's whole FFN
    /// output. Upstream folds it into the routing weights on a MoE layer
    /// and into the down projection's result on a dense one; both are the
    /// same multiply of the same branch, so it is applied in one place
    /// here.
    ffn_scale: f32,
    ffn: Ffn,
}

pub struct InklingModel {
    config: ModelConfig,
    backend: Arc<dyn Backend>,
    tok_embeddings: QuantMatrix,
    /// `token_embd_norm` — a *weighted* RMSNorm on the embeddings before
    /// the first layer (`engine::arch::muse` has the weightless form here,
    /// gemma a `sqrt(n_embd)` scale, and the llama family nothing).
    tok_embd_norm: Vec<f32>,
    output_norm: Vec<f32>,
    output_weight: QuantMatrix,
    layers: Vec<InklingLayer>,
    /// `attention.sliding_window` — the width a sliding-window layer
    /// attends.
    n_swa: usize,
    /// `inkling.d_rel` — the width of a head's relative-bias coefficients.
    d_rel: usize,
    /// `inkling.shortconv_kernel` — how many taps each short convolution
    /// has, and so how many prior inputs its rolling window carries.
    conv_kernel: usize,
    /// `inkling.logit_scale_denom` — the final hidden state is divided by
    /// this before the output projection.
    logit_denom: f32,
    /// `inkling.log_scaling_n_floor` / `.log_scaling_alpha` — the context
    /// length past which a full-attention layer's scores are scaled up, and
    /// by how much per natural log. A floor of `0` disables it.
    log_floor: f64,
    log_alpha: f32,
    n_expert: usize,
    n_expert_used: usize,
    n_expert_shared: usize,
    /// `expert_gating_func` — `2` (sigmoid) for every released checkpoint;
    /// read rather than assumed so a different one is a clear load error.
    gating: ExpertGating,
    /// `expert_weights_scale`, applied to the normalized routing weights.
    route_scale: f32,
    /// `inkling.unpadded_vocab_size` — how many of the vocabulary's rows
    /// are real tokens. The rest are padding and are masked out of every
    /// logits vector this model returns.
    unpadded_vocab: usize,
}

impl InklingModel {
    pub fn load_with_backend(loaded: &LoadedModel, backend: Arc<dyn Backend>) -> Result<Self> {
        let config = loaded.config.clone();
        let n_layer = config.n_layer;
        let n_head = config.n_head;
        let head_dim = config.head_dim;

        // `attention.head_count_kv` is a per-layer array in the released
        // checkpoints (every entry equal so far, but the model's own
        // configuration distinguishes the sliding-window layers' KV width
        // from the full-attention ones, so it is read per layer rather than
        // collapsed). A scalar — or nothing — is broadcast.
        let n_head_kv_default = loaded
            .metadata_u64("attention.head_count_kv")
            .unwrap_or(n_head as u64) as usize;
        let n_head_kv_per_layer = loaded.metadata_array_u64("attention.head_count_kv");

        let n_swa = loaded
            .metadata_u64("attention.sliding_window")
            .context("missing attention.sliding_window")? as usize;
        let swa_pattern = loaded.metadata_array_u64("attention.sliding_window_pattern");
        let d_rel = loaded.metadata_u64("d_rel").context("missing d_rel")? as usize;
        let rel_extent = loaded
            .metadata_u64("rel_extent")
            .context("missing rel_extent")? as usize;
        let rel_extent_swa = loaded
            .metadata_u64("rel_extent_swa")
            .unwrap_or(rel_extent as u64) as usize;
        let conv_kernel = loaded
            .metadata_u64("shortconv_kernel")
            .context("missing shortconv_kernel")? as usize;
        anyhow::ensure!(
            conv_kernel > 0,
            "shortconv_kernel must be positive (got {conv_kernel})"
        );
        let n_dense = loaded.metadata_u64("dense_block_count").unwrap_or(0) as usize;
        let logit_denom = loaded.metadata_f32("logit_scale_denom").unwrap_or(1.0);
        anyhow::ensure!(
            logit_denom != 0.0,
            "logit_scale_denom must be nonzero (got {logit_denom})"
        );
        let log_floor = loaded.metadata_f32("log_scaling_n_floor").unwrap_or(0.0) as f64;
        let log_alpha = loaded.metadata_f32("log_scaling_alpha").unwrap_or(0.0);
        let unpadded_vocab = loaded
            .metadata_u64("unpadded_vocab_size")
            .map(|v| v as usize)
            .unwrap_or(config.n_vocab)
            .min(config.n_vocab);

        let n_expert = loaded.metadata_u64("expert_count").unwrap_or(0) as usize;
        let n_expert_used = loaded.metadata_u64("expert_used_count").unwrap_or(0) as usize;
        let n_expert_shared = loaded.metadata_u64("expert_shared_count").unwrap_or(0) as usize;
        let route_scale = loaded.metadata_f32("expert_weights_scale").unwrap_or(1.0);
        let gating = ExpertGating::from_gguf(
            loaded
                .metadata_u64("expert_gating_func")
                .context("missing expert_gating_func")?,
        )?;

        let tok_embeddings = loaded
            .matrix("token_embd.weight")
            .context("loading token_embd.weight")?;
        let (tok_embd_norm, _) = loaded
            .tensor("token_embd_norm.weight")
            .context("loading token_embd_norm.weight")?;
        let (output_norm, _) = loaded
            .tensor("output_norm.weight")
            .context("loading output_norm.weight")?;
        let output_weight = if loaded.has_tensor("output.weight") {
            loaded
                .matrix("output.weight")
                .context("loading output.weight")?
        } else {
            tok_embeddings.clone()
        };

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

            let is_swa = match &swa_pattern {
                Some(pattern) => pattern.get(i).copied().unwrap_or(0) != 0,
                None => (i + 1) % DEFAULT_SWA_PERIOD != 0,
            };
            let n_head_kv = match &n_head_kv_per_layer {
                Some(per_layer) => per_layer.get(i).copied().unwrap_or(0) as usize,
                None => n_head_kv_default,
            };
            anyhow::ensure!(
                n_head_kv > 0 && n_head.is_multiple_of(n_head_kv),
                "layer {i}: {n_head} query heads do not divide into {n_head_kv} KV heads"
            );

            let (rel_proj, rel_dims) = loaded
                .tensor(&format!("blk.{i}.attn_rel_proj.weight"))
                .with_context(|| format!("loading blk.{i}.attn_rel_proj.weight"))?;
            // The layer's own bias bank is the independent witness for
            // which kind of layer this is: a sliding-window layer's bank is
            // `rel_extent_swa` wide and a full-attention layer's is
            // `rel_extent` wide. Read the pattern, then hold it to the
            // tensor — a pattern read the wrong way round produces a model
            // that loads and answers, badly.
            let expected_extent = if is_swa { rel_extent_swa } else { rel_extent };
            anyhow::ensure!(
                rel_dims == [expected_extent as u64, d_rel as u64],
                "layer {i}: attn_rel_proj is {rel_dims:?}, but attention.sliding_window_pattern \
                 says this layer is {} and so wants [{expected_extent}, {d_rel}]",
                if is_swa {
                    "sliding-window"
                } else {
                    "full-attention"
                }
            );

            let ffn_scale = {
                let (values, _) = loaded
                    .tensor(&format!("blk.{i}.ffn_gscale.weight"))
                    .with_context(|| format!("loading blk.{i}.ffn_gscale.weight"))?;
                *values
                    .first()
                    .with_context(|| format!("blk.{i}.ffn_gscale.weight is empty"))?
            };
            let ffn = if i < n_dense {
                Ffn::Dense {
                    gate: get_matrix("ffn_gate.weight")?,
                    up: get_matrix("ffn_up.weight")?,
                    down: get_matrix("ffn_down.weight")?,
                }
            } else {
                anyhow::ensure!(
                    n_expert > 0 && n_expert_used > 0,
                    "layer {i} has routed experts but the file sets expert_count = {n_expert} \
                     and expert_used_count = {n_expert_used}"
                );
                let (exp_probs_b, _) = loaded
                    .tensor(&format!("blk.{i}.exp_probs_b.bias"))
                    .with_context(|| format!("loading blk.{i}.exp_probs_b.bias"))?;
                let gate_inp = get_matrix("ffn_gate_inp.weight")?;
                // The router's *width* is what says the shared experts are
                // gated by it. A router of only `n_expert` rows would leave
                // the shared branch with no weights at all — which reads as
                // a slightly quieter FFN, not as an error.
                anyhow::ensure!(
                    gate_inp.out_dim == n_expert + n_expert_shared,
                    "layer {i}: ffn_gate_inp has {} rows, expected {n_expert} routed experts + \
                     {n_expert_shared} shared",
                    gate_inp.out_dim
                );
                Ffn::Moe(Box::new(Moe {
                    gate_inp,
                    exp_probs_b,
                    gate_exps: get_expert_matrix("ffn_gate_exps.weight")?,
                    up_exps: get_expert_matrix("ffn_up_exps.weight")?,
                    down_exps: get_expert_matrix("ffn_down_exps.weight")?,
                    gate_shexp: get_expert_matrix("ffn_gate_shexp.weight")?,
                    up_shexp: get_expert_matrix("ffn_up_shexp.weight")?,
                    down_shexp: get_expert_matrix("ffn_down_shexp.weight")?,
                }))
            };

            let kv_dim = n_head_kv * head_dim;
            let conv_k = get("shortconv_k.weight")?;
            let conv_v = get("shortconv_v.weight")?;
            let conv_attn = get("shortconv_attn.weight")?;
            let conv_mlp = get("shortconv_mlp.weight")?;
            for (name, kernel, channels) in [
                ("shortconv_k", &conv_k, kv_dim),
                ("shortconv_v", &conv_v, kv_dim),
                ("shortconv_attn", &conv_attn, config.n_embd),
                ("shortconv_mlp", &conv_mlp, config.n_embd),
            ] {
                anyhow::ensure!(
                    kernel.len() == channels * conv_kernel,
                    "layer {i}: {name} has {} values, expected {channels} channels x \
                     {conv_kernel} taps",
                    kernel.len()
                );
            }

            // One `d_rel`-wide coefficient vector per head, and the bias
            // loop reads it as exactly that: a narrower `attn_r` would have
            // every head after the first mixing the previous head's tail.
            let wr = get_matrix("attn_r.weight")?;
            anyhow::ensure!(
                wr.out_dim == n_head * d_rel,
                "layer {i}: attn_r projects to {}, expected {n_head} heads x {d_rel}",
                wr.out_dim
            );

            layers.push(InklingLayer {
                attn_norm: get("attn_norm.weight")?,
                ffn_norm: get("ffn_norm.weight")?,
                wq: get_matrix("attn_q.weight")?,
                wk: get_matrix("attn_k.weight")?,
                wv: get_matrix("attn_v.weight")?,
                wr,
                wo: get_matrix("attn_output.weight")?,
                attn_q_norm: get("attn_q_norm.weight")?,
                attn_k_norm: get("attn_k_norm.weight")?,
                rel_proj,
                rel_extent: expected_extent,
                conv_k,
                conv_v,
                conv_attn,
                conv_mlp,
                is_swa,
                n_head_kv,
                ffn_scale,
                ffn,
            });
        }

        Ok(Self {
            config,
            backend,
            tok_embeddings,
            tok_embd_norm,
            output_norm,
            output_weight,
            layers,
            n_swa,
            d_rel,
            conv_kernel,
            logit_denom,
            log_floor,
            log_alpha,
            n_expert,
            n_expert_used,
            n_expert_shared,
            gating,
            route_scale,
            unpadded_vocab,
        })
    }

    /// This layer's short-convolution channel counts, in
    /// `KvCache::recurrent` slot order — see [`CONV_K`] and friends.
    fn conv_channels(&self, layer: &InklingLayer) -> [usize; CONV_PER_LAYER] {
        let kv_dim = layer.n_head_kv * self.config.head_dim;
        [kv_dim, kv_dim, self.config.n_embd, self.config.n_embd]
    }

    /// The attention temperature for a query at absolute position `pos`.
    ///
    /// `1.0` everywhere on a sliding-window layer and everywhere below the
    /// floor; above it, `1 + alpha * ln((pos + 1) / floor)`, which grows
    /// without bound as the context does. Computed in `f64` because the
    /// ratio is a large integer over a large integer and the interesting
    /// part is its logarithm.
    fn length_scale(&self, is_swa: bool, pos: usize) -> f32 {
        if is_swa || self.log_floor <= 0.0 {
            return 1.0;
        }
        let ratio = (pos as f64 + 1.0) / self.log_floor;
        if ratio > 1.0 {
            1.0 + self.log_alpha * ratio.ln() as f32
        } else {
            1.0
        }
    }

    /// One layer's attention, returning `[n_tokens, n_embd]` — the output
    /// projection's result, before the short convolution and residual add
    /// the caller applies.
    fn attention(
        &self,
        cache: &mut KvCache,
        layer: &InklingLayer,
        il: usize,
        normed: &[f32],
        n_tokens: usize,
        start_pos: usize,
    ) -> Vec<f32> {
        let cfg = &self.config;
        let n_head = cfg.n_head;
        let head_dim = cfg.head_dim;
        let q_dim = n_head * head_dim;
        let kv_dim = layer.n_head_kv * head_dim;
        let eps = cfg.rms_eps;

        // Q, K, V and the relative-bias coefficients are four independent
        // projections of the same normed input — one batched dispatch
        // rather than four round trips (see `Backend::matmul_batch`).
        let mut qkvr = self.backend.matmul_batch(&[
            MatmulOp {
                x: normed,
                n_tokens,
                w: &layer.wq,
            },
            MatmulOp {
                x: normed,
                n_tokens,
                w: &layer.wk,
            },
            MatmulOp {
                x: normed,
                n_tokens,
                w: &layer.wv,
            },
            MatmulOp {
                x: normed,
                n_tokens,
                w: &layer.wr,
            },
        ]);
        let r = qkvr.pop().unwrap();
        let mut v = qkvr.pop().unwrap();
        let mut k = qkvr.pop().unwrap();
        let mut q = qkvr.pop().unwrap();

        // The key/value convolutions run on the *raw* projections, before
        // the per-head norm and before anything is cached — so what the
        // cache holds is already convolved, and a later token reads it
        // unchanged.
        let base = il * CONV_PER_LAYER;
        shortconv(
            &mut cache.recurrent[base + CONV_K],
            &mut k,
            n_tokens,
            kv_dim,
            &layer.conv_k,
        );
        shortconv(
            &mut cache.recurrent[base + CONV_V],
            &mut v,
            n_tokens,
            kv_dim,
            &layer.conv_v,
        );

        tensor::rmsnorm_inplace(&mut q, &layer.attn_q_norm, n_tokens * n_head, head_dim, eps);
        tensor::rmsnorm_inplace(
            &mut k,
            &layer.attn_k_norm,
            n_tokens * layer.n_head_kv,
            head_dim,
            eps,
        );

        {
            let layer_cache = &mut cache.layers[il];
            for t in 0..n_tokens {
                layer_cache.push(
                    &k[t * kv_dim..(t + 1) * kv_dim],
                    &v[t * kv_dim..(t + 1) * kv_dim],
                );
            }
        }
        let layer_cache = &cache.layers[il];

        let group = n_head / layer.n_head_kv;
        // `1 / head_dim`, not `1 / sqrt(head_dim)` — see this module's own
        // doc comment.
        let scale = 1.0 / head_dim as f32;
        let extent = layer.rel_extent;
        let d_rel = self.d_rel;

        let mut ctx = vec![0f32; n_tokens * q_dim];
        // One task per (token, head): each writes its own `head_dim` slice
        // and reads nothing another task writes. The two scratch buffers
        // are initialized once per rayon *job* rather than per task
        // (`for_each_init`), so a prefill's tens of thousands of tasks
        // refill two buffers instead of allocating two each.
        ctx.par_chunks_mut(head_dim).enumerate().for_each_init(
            || (Vec::<f32>::new(), Vec::<f32>::new()),
            |(rel, scores), (index, out)| {
                let t = index / n_head;
                let h = index % n_head;
                let qpos = start_pos + t;
                let first = if layer.is_swa && self.n_swa > 0 {
                    qpos.saturating_sub(self.n_swa - 1)
                } else {
                    0
                };
                let n_keys = qpos - first + 1;

                // This token's bias bank for this head, mixed from the
                // per-head coefficients. Only the distances actually
                // reachable from this query are built; past the bank's
                // width the bias is zero by definition, so the rest
                // would be computed and never read.
                let width = extent.min(n_keys);
                rel.clear();
                rel.resize(width, 0.0);
                let coeffs = &r[(t * n_head + h) * d_rel..(t * n_head + h + 1) * d_rel];
                for (d, &c) in coeffs.iter().enumerate() {
                    tensor::axpy_inplace(rel, &layer.rel_proj[d * extent..d * extent + width], c);
                }

                let tau = self.length_scale(layer.is_swa, qpos);
                let qv = &q[t * q_dim + h * head_dim..t * q_dim + (h + 1) * head_dim];
                let kv_head = h / group;
                scores.clear();
                scores.extend((first..=qpos).map(|p| {
                    let key = layer_cache.key_at(p, kv_head, head_dim);
                    let dist = qpos - p;
                    let bias = if dist < width { rel[dist] } else { 0.0 };
                    tau * (tensor::dot(qv, key) * scale + bias)
                }));
                tensor::softmax_inplace(scores);
                for (i, &weight) in scores.iter().enumerate() {
                    tensor::axpy_inplace(
                        out,
                        layer_cache.value_at(first + i, kv_head, head_dim),
                        weight,
                    );
                }
            },
        );

        self.backend.matmul(&ctx, n_tokens, &layer.wo)
    }

    /// One layer's feed-forward half, `[n_tokens, n_embd]`, with
    /// [`InklingLayer::ffn_scale`] already applied.
    fn ffn(&self, layer: &InklingLayer, normed: &[f32], n_tokens: usize) -> Vec<f32> {
        let mut out = match &layer.ffn {
            Ffn::Dense { gate, up, down } => {
                let mut gate_up = self.backend.matmul_batch(&[
                    MatmulOp {
                        x: normed,
                        n_tokens,
                        w: gate,
                    },
                    MatmulOp {
                        x: normed,
                        n_tokens,
                        w: up,
                    },
                ]);
                let up = gate_up.pop().unwrap();
                let mut act = gate_up.pop().unwrap();
                for g in act.iter_mut() {
                    *g = tensor::silu(*g);
                }
                tensor::mul_inplace(&mut act, &up);
                self.backend.matmul(&act, n_tokens, down)
            }
            Ffn::Moe(moe) => self.moe_ffn(moe, normed, n_tokens),
        };
        if layer.ffn_scale != 1.0 {
            for v in out.iter_mut() {
                *v *= layer.ffn_scale;
            }
        }
        out
    }

    /// Routed experts plus the always-on shared ones.
    ///
    /// The routing is `engine::arch::glm`'s — sigmoid probabilities, a
    /// correction bias that steers the *selection* only, top-k — with one
    /// difference that has to be got right: the router's last
    /// `n_expert_shared` logits gate the shared experts, and the
    /// normalization runs over the selected routed weights **and** those
    /// together. Normalizing the routed ones among themselves (which is
    /// what every other MoE here does) leaves the shared branch weighted as
    /// if the routed branch summed to one, and produces fluent, subtly
    /// wrong output.
    fn moe_ffn(&self, moe: &Moe, normed: &[f32], n_tokens: usize) -> Vec<f32> {
        let n_embd = self.config.n_embd;
        let mut experts =
            moe_stats::LayerRecorder::for_tensors(&[&moe.gate_exps, &moe.up_exps, &moe.down_exps]);

        // Route the whole batch first, so the union can be taken before any
        // expert's weights are read — see `super::evaluate_routed_experts`.
        let mut selection: Vec<Vec<(usize, f32)>> = Vec::with_capacity(n_tokens);
        // `[shared expert][token]`, the order the evaluation loop below
        // walks — one expert applied to every token, rather than one token
        // at a time — so its weights are read the way they are used.
        let mut shared_weights: Vec<Vec<f32>> =
            vec![Vec::with_capacity(n_tokens); self.n_expert_shared];
        for t in 0..n_tokens {
            let x_t = &normed[t * n_embd..(t + 1) * n_embd];
            let logits = self.backend.matmul(x_t, 1, &moe.gate_inp);
            let probs: Vec<f32> = logits.iter().map(|&l| self.gating.apply(l)).collect();
            let mut choice = probs[..self.n_expert].to_vec();
            tensor::add_inplace(&mut choice, &moe.exp_probs_b);
            let selected = super::top_k_indices(&choice, self.n_expert_used);

            let mut weights: Vec<f32> = selected.iter().map(|&e| probs[e]).collect();
            let mut shared: Vec<f32> = probs[self.n_expert..].to_vec();
            let sum: f32 = weights.iter().chain(shared.iter()).sum();
            let factor = self.route_scale / sum.max(super::MIN_EXPERT_WEIGHT_SUM);
            for w in weights.iter_mut().chain(shared.iter_mut()) {
                *w *= factor;
            }
            selection.push(selected.into_iter().zip(weights).collect());
            for (slot, weight) in shared_weights.iter_mut().zip(shared) {
                slot.push(weight);
            }
        }
        // Trim to the expert budget *before* anything is recorded or read:
        // the counters should describe the work actually done, and a
        // dropped expert's weights must never be fetched.
        super::apply_expert_budget(&mut selection, &moe.gate_exps);
        for picks in &selection {
            picks.iter().for_each(|&(e, _)| experts.select(e));
        }

        let swiglu = |gate: &[f32], up: &[f32]| -> Vec<f32> {
            let mut h: Vec<f32> = gate.iter().map(|&g| tensor::silu(g)).collect();
            tensor::mul_inplace(&mut h, up);
            h
        };
        // The GPU expert path batches the three projections across experts —
        // see `super::evaluate_routed_experts_batched`.
        let contribs = if super::gpu_experts() && self.backend.as_wgpu().is_some() {
            super::evaluate_routed_experts_batched(
                self.backend.as_ref(),
                &selection,
                normed,
                n_embd,
                &moe.gate_exps,
                &moe.up_exps,
                &moe.down_exps,
                swiglu,
            )
        } else {
            super::evaluate_routed_experts(&selection, |expert, members| {
                let inputs: Vec<&[f32]> = members
                    .iter()
                    .map(|&(t, _)| &normed[t * n_embd..(t + 1) * n_embd])
                    .collect();
                let hidden = self.expert_hidden(&moe.gate_exps, &moe.up_exps, expert, &inputs);
                let hidden_refs: Vec<&[f32]> = hidden.iter().map(Vec::as_slice).collect();
                super::project_expert(
                    self.backend.as_ref(),
                    &moe.down_exps,
                    expert,
                    0,
                    moe.down_exps.out_dim,
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

        // The shared experts, evaluated once per expert across the whole
        // batch rather than once per (token, expert) — the same grouping
        // the routed path gets from `evaluate_routed_experts`, which here
        // is simply "every token" since every token uses all of them.
        let inputs: Vec<&[f32]> = (0..n_tokens)
            .map(|t| &normed[t * n_embd..(t + 1) * n_embd])
            .collect();
        let mut out = vec![0f32; n_tokens * n_embd];
        for (j, weights) in shared_weights.iter().enumerate() {
            let hidden = self.expert_hidden(&moe.gate_shexp, &moe.up_shexp, j, &inputs);
            let hidden_refs: Vec<&[f32]> = hidden.iter().map(Vec::as_slice).collect();
            let down = super::project_expert(
                self.backend.as_ref(),
                &moe.down_shexp,
                j,
                0,
                moe.down_shexp.out_dim,
                &hidden_refs,
            );
            for ((t, contribution), &weight) in down.into_iter().enumerate().zip(weights) {
                tensor::axpy_inplace(
                    &mut out[t * n_embd..(t + 1) * n_embd],
                    &contribution,
                    weight,
                );
            }
        }
        for t in 0..n_tokens {
            let dst = &mut out[t * n_embd..(t + 1) * n_embd];
            for contribution in &contribs[t] {
                tensor::add_inplace(dst, contribution);
            }
        }
        experts.commit(n_tokens);
        out
    }

    /// One expert's SwiGLU hidden state for each of `inputs` — the gate and
    /// up projections plus `silu(gate) * up`, shared by the routed and the
    /// shared expert paths (they differ only in which stacked tensor and
    /// which index they read).
    fn expert_hidden(
        &self,
        gate_exps: &ExpertQuantMatrix,
        up_exps: &ExpertQuantMatrix,
        expert: usize,
        inputs: &[&[f32]],
    ) -> Vec<Vec<f32>> {
        let gate = super::project_expert(
            self.backend.as_ref(),
            gate_exps,
            expert,
            0,
            gate_exps.out_dim,
            inputs,
        );
        let up = super::project_expert(
            self.backend.as_ref(),
            up_exps,
            expert,
            0,
            up_exps.out_dim,
            inputs,
        );
        gate.into_iter()
            .zip(up)
            .map(|(gate, up)| {
                let mut h: Vec<f32> = gate.iter().map(|&g| tensor::silu(g)).collect();
                tensor::mul_inplace(&mut h, &up);
                h
            })
            .collect()
    }

    /// Every layer, from the token embeddings to the last residual —
    /// `[n_tokens, n_embd]`. Shared by `forward` and
    /// `forward_hidden_states`, which differ only in what they do with it.
    fn run_layers(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let n_tokens = tokens.len();
        let n_embd = cfg.n_embd;
        let eps = cfg.rms_eps;
        anyhow::ensure!(
            cache.recurrent.len() == self.layers.len() * CONV_PER_LAYER,
            "this cache has {} short-convolution states, expected {}",
            cache.recurrent.len(),
            self.layers.len() * CONV_PER_LAYER
        );

        let mut x = vec![0f32; n_tokens * n_embd];
        for (t, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            anyhow::ensure!(tok < cfg.n_vocab, "token id {tok} is out of vocab range");
            x[t * n_embd..(t + 1) * n_embd].copy_from_slice(&self.tok_embeddings.row(tok));
        }
        tensor::rmsnorm_inplace(&mut x, &self.tok_embd_norm, n_tokens, n_embd, eps);

        for (il, layer) in self.layers.iter().enumerate() {
            let mut normed = x.clone();
            tensor::rmsnorm_inplace(&mut normed, &layer.attn_norm, n_tokens, n_embd, eps);
            let mut attn = self.attention(cache, layer, il, &normed, n_tokens, start_pos);
            shortconv(
                &mut cache.recurrent[il * CONV_PER_LAYER + CONV_ATTN],
                &mut attn,
                n_tokens,
                n_embd,
                &layer.conv_attn,
            );
            tensor::add_inplace(&mut x, &attn);

            let mut ffn_normed = x.clone();
            tensor::rmsnorm_inplace(&mut ffn_normed, &layer.ffn_norm, n_tokens, n_embd, eps);
            let mut ffn_out = self.ffn(layer, &ffn_normed, n_tokens);
            shortconv(
                &mut cache.recurrent[il * CONV_PER_LAYER + CONV_MLP],
                &mut ffn_out,
                n_tokens,
                n_embd,
                &layer.conv_mlp,
            );
            tensor::add_inplace(&mut x, &ffn_out);
        }
        Ok(x)
    }

    /// Silences the vocabulary's padding rows.
    ///
    /// `token_embd` and `output` are padded out to a round `vocab_size`
    /// (201024 against 200058 real tokens on `Inkling-Small`), and those
    /// rows hold real numbers — left alone, one of them can and does win an
    /// argmax, and decoding it yields nothing at all. `-inf` rather than a
    /// truncation so the vector every caller receives is still
    /// `[n_vocab]` wide — the width the sampler and the logit-bias path
    /// both index by token id.
    fn mask_padding_logits(&self, logits: &mut [f32]) {
        if self.unpadded_vocab < logits.len() {
            for v in logits[self.unpadded_vocab..].iter_mut() {
                *v = f32::NEG_INFINITY;
            }
        }
    }

    /// The output norm, the logit divisor and the vocabulary projection,
    /// over one already-selected hidden state.
    fn project_logits(&self, hidden: &[f32]) -> Vec<f32> {
        let n_embd = self.config.n_embd;
        let mut last = hidden.to_vec();
        tensor::rmsnorm_inplace(&mut last, &self.output_norm, 1, n_embd, self.config.rms_eps);
        for v in last.iter_mut() {
            *v /= self.logit_denom;
        }
        let mut logits = self.backend.matmul(&last, 1, &self.output_weight);
        self.mask_padding_logits(&mut logits);
        logits
    }
}

/// One short convolution over `seq` (`[n_tokens, channels]`), in place,
/// advancing `state`'s rolling window as it goes.
///
/// `conv(x) + x`: the residual is part of the operator, not something the
/// caller adds — the same shape upstream's `sconv` has, and the reason a
/// layer's residual add is a *second*, separate add of the pre-convolution
/// activations.
///
/// Token by token, because that is what the state is: each output depends
/// on the previous `kernel - 1` inputs, and a decode step's window reaches
/// back into the previous call's. `RecurrentLayerState::conv_step` is the
/// same primitive `engine::arch::qwen35moe`'s linear-attention layers use.
fn shortconv(
    state: &mut RecurrentLayerState,
    seq: &mut [f32],
    n_tokens: usize,
    channels: usize,
    kernel: &[f32],
) {
    for t in 0..n_tokens {
        let row = &mut seq[t * channels..(t + 1) * channels];
        let convolved = state.conv_step(row, kernel);
        for (dst, src) in row.iter_mut().zip(convolved) {
            *dst += src;
        }
    }
}

impl ModelForward for InklingModel {
    fn vulkan_backend(&self) -> Option<&crate::engine::backend::vulkan::VulkanBackend> {
        self.backend.as_wgpu()
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn new_kv_cache(&self, capacity: usize) -> KvCache {
        let kv_dims: Vec<usize> = self
            .layers
            .iter()
            .map(|l| l.n_head_kv * self.config.head_dim)
            .collect();
        // Four convolution states per layer, in `CONV_*` order, each with
        // no delta-net state at all — this architecture's recurrence is the
        // convolution window and nothing else.
        let recurrent: Vec<(usize, usize, usize, usize)> = self
            .layers
            .iter()
            .flat_map(|l| {
                self.conv_channels(l)
                    .map(|channels| (channels, self.conv_kernel, 0, 0))
            })
            .collect();
        KvCache::new_mixed(capacity, &kv_dims, &recurrent)
    }

    fn forward(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        _slot_id: usize,
    ) -> Result<Vec<f32>> {
        anyhow::ensure!(!tokens.is_empty(), "forward called with no tokens");
        let n_embd = self.config.n_embd;
        let x = self.run_layers(cache, tokens, start_pos)?;
        Ok(self.project_logits(&x[(tokens.len() - 1) * n_embd..]))
    }

    fn forward_hidden_states(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        // A one-shot, whole-prompt pass — no KV cache reuse across calls,
        // the same convention `LlamaModel::forward_hidden_states` uses.
        let mut cache = self.new_kv_cache(tokens.len().max(1));
        let mut x = self.run_layers(&mut cache, tokens, 0)?;
        tensor::rmsnorm_inplace(
            &mut x,
            &self.output_norm,
            tokens.len(),
            self.config.n_embd,
            self.config.rms_eps,
        );
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The short convolution's residual is *inside* it, and its window
    /// spans calls. Both are checked against the arithmetic written out by
    /// hand rather than against a second copy of the implementation's own
    /// loop, and the whole point is the second call: a decode step's first
    /// token has to see the taps left by the prefill that preceded it.
    #[test]
    fn a_short_convolution_carries_its_window_across_calls() {
        // One channel, three taps, so a token's output is
        // `w0*x[t-2] + w1*x[t-1] + w2*x[t] + x[t]`.
        let mut cache = KvCache::new_mixed(4, &[0], &[(1, 3, 0, 0)]);
        let kernel = [0.5f32, 0.25, 2.0];
        let mut prefill = vec![1.0f32, 2.0, 3.0];
        shortconv(&mut cache.recurrent[0], &mut prefill, 3, 1, &kernel);
        assert_eq!(
            prefill,
            vec![
                2.0 * 1.0 + 1.0,
                0.25 * 1.0 + 2.0 * 2.0 + 2.0,
                0.5 * 1.0 + 0.25 * 2.0 + 2.0 * 3.0 + 3.0,
            ]
        );

        // A separate call, one token — the taps must still reach back to
        // the 2 and the 3 of the previous one.
        let mut decode = vec![4.0f32];
        shortconv(&mut cache.recurrent[0], &mut decode, 1, 1, &kernel);
        assert_eq!(decode, vec![0.5 * 2.0 + 0.25 * 3.0 + 2.0 * 4.0 + 4.0]);
    }

    /// The relative-bias bank is built per `(token, head)` and only out to
    /// the distances a query can actually reach, so the truncation must not
    /// change any value it keeps. Restated as an explicit dot product over
    /// the full bank rather than as the loop under test.
    #[test]
    fn the_truncated_bias_bank_matches_the_full_one() {
        let d_rel = 3;
        let extent = 5;
        let proj: Vec<f32> = (0..d_rel * extent)
            .map(|i| (i as f32 + 1.0) * 0.25)
            .collect();
        let coeffs = [0.5f32, -1.5, 2.0];
        let full: Vec<f32> = (0..extent)
            .map(|e| (0..d_rel).map(|d| coeffs[d] * proj[d * extent + e]).sum())
            .collect();

        for width in 1..=extent {
            let mut rel = vec![0f32; width];
            for (d, &c) in coeffs.iter().enumerate() {
                tensor::axpy_inplace(&mut rel, &proj[d * extent..d * extent + width], c);
            }
            assert_eq!(rel, full[..width], "width {width}");
        }
    }

    /// The padding rows of a padded vocabulary must be unreachable, and the
    /// vector must keep its full width — a caller indexing it by token id
    /// (logit bias, the speculative verifier) would otherwise read the
    /// wrong token's logit.
    #[test]
    fn padding_logits_are_masked_without_shortening_the_vector() {
        let mut logits: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let unpadded = 5;
        if unpadded < logits.len() {
            for v in logits[unpadded..].iter_mut() {
                *v = f32::NEG_INFINITY;
            }
        }
        assert_eq!(logits.len(), 8);
        assert_eq!(&logits[..5], &[0.0, 1.0, 2.0, 3.0, 4.0]);
        assert!(logits[5..].iter().all(|v| *v == f32::NEG_INFINITY));
        let top = logits
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(top, 4, "a padding row won the argmax");
    }

    /// The length scaling is a full-attention-layer mechanism and is `1.0`
    /// below the floor. A model whose prompt never reaches the floor cannot
    /// tell whether it is implemented at all, so the boundary is asserted
    /// directly.
    #[test]
    fn length_scaling_applies_only_to_full_attention_layers_above_the_floor() {
        let scale = |is_swa: bool, pos: usize, floor: f64, alpha: f32| -> f32 {
            if is_swa || floor <= 0.0 {
                return 1.0;
            }
            let ratio = (pos as f64 + 1.0) / floor;
            if ratio > 1.0 {
                1.0 + alpha * ratio.ln() as f32
            } else {
                1.0
            }
        };
        assert_eq!(scale(true, 1_000_000, 128_000.0, 0.1), 1.0);
        assert_eq!(scale(false, 127_999, 128_000.0, 0.1), 1.0);
        assert!(scale(false, 255_999, 128_000.0, 0.1) > 1.06);
        assert_eq!(scale(false, 1_000_000, 0.0, 0.1), 1.0);
    }
}

#[cfg(test)]
mod real_model_tests {
    use super::*;

    /// End-to-end against a real `unsloth/Inkling-Small-GGUF` file.
    ///
    /// The assertion is on the *predicted token*, not on logits within a
    /// tolerance, for the reason `arch::llama`'s equivalent test spells
    /// out: a wrong-but-plausible forward pass produces fluent output a
    /// tolerance check can be talked into accepting, while a factual
    /// one-word answer cannot survive a broken attention or FFN. This
    /// architecture has an unusual number of ways to be wrong-but-plausible
    /// — dropping the relative bias, dropping a short convolution or
    /// running it without its residual, normalizing the routed expert
    /// weights without the shared ones, using `1/sqrt(head_dim)` for the
    /// attention scale, or skipping the embedding norm each produce a model
    /// that loads and runs.
    ///
    /// Run with `ORANGU_TEST_INKLING_MODEL=/path/to/Inkling-Small-UD-Q4_K_M-00001-of-00005.gguf
    /// cargo test --release --bin orangu-server inkling::real_model_tests --
    /// --ignored`.
    #[test]
    #[ignore]
    fn inkling_predicts_paris_after_capital_of_france() {
        let path =
            std::env::var("ORANGU_TEST_INKLING_MODEL").expect("set ORANGU_TEST_INKLING_MODEL");
        let gguf = orangu::gguf::GgufFile::open(std::path::Path::new(&path)).expect("open gguf");
        let tokenizer =
            crate::engine::tokenizer::Tokenizer::from_gguf(&gguf).expect("build tokenizer");
        let loaded = LoadedModel::open(std::path::Path::new(&path)).expect("load model");
        assert_eq!(loaded.config.architecture, "inkling");
        let model =
            InklingModel::load_with_backend(&loaded, Arc::new(crate::engine::backend::CpuBackend))
                .expect("build model");
        // The sliding-window pattern is load-bearing twice over — it picks
        // the attention window *and* the relative-bias width — so a
        // checkpoint whose pattern differs is reported as such rather than
        // as a bad prediction.
        assert!(
            model.layers.iter().any(|l| l.is_swa) && model.layers.iter().any(|l| !l.is_swa),
            "expected a mix of sliding-window and full-attention layers"
        );

        let tokens = tokenizer.encode("The capital of France is", true);
        let mut cache = model.new_kv_cache(tokens.len() + 1);
        let logits = model.forward(&mut cache, &tokens, 0, 0).expect("forward");
        let (top_id, _) = logits
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        assert_eq!(
            tokenizer.decode(&[top_id as u32]),
            " Paris",
            "top prediction was {:?}",
            tokenizer.decode(&[top_id as u32])
        );
    }
}
