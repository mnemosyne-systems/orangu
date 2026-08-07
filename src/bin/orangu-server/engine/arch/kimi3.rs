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

//! Kimi-K3 (`general.architecture = "kimi-k3"`), e.g.
//! `unsloth/Kimi-K3-GGUF`.
//!
//! A hybrid of two attention layers this engine already has pieces of —
//! Kimi Delta Attention (a gated delta-net, three-in-four layers) and
//! absorbed multi-head latent attention (every fourth layer, as in
//! `engine::arch::glm`) — plus five things neither has:
//!
//! 1. **Cross-layer residual attention.** Every `attn_res.block_size`th
//!    layer *banks* its raw input, and the residual stream then restarts
//!    from that layer's attention output alone. Before each half-layer the
//!    stream is re-mixed with every banked checkpoint by a softmax over
//!    per-checkpoint scores, so a layer can reach back past the block it is
//!    in. See [`Kimi3Model::res_mix`].
//! 2. **Latent MoE.** The routed experts do not run at `n_embd`: the FFN
//!    input is projected down to `expert_latent_length`, the experts run
//!    there, and the result is normed and projected back up. The *router*
//!    still scores the full-width input, so the logits are computed before
//!    the down-projection.
//! 3. **The situ activation**, in place of SwiGLU everywhere — see
//!    [`situ`].
//! 4. **A sigmoid output gate on the MLA layers**, read from the normed
//!    layer input and applied before the output projection.
//! 5. **A full-rank KDA gate** (one `ssm_g`) where Kimi-Linear factors the
//!    same thing as `ssm_g_a`/`ssm_g_b`.
//!
//! Its MLA is also **nope-only**: `rope.dimension_count` still describes
//! the width of the key's second half, but nothing is rotated — there is no
//! position input in this model at all. Position enters only through the
//! KDA layers' recurrence and the causal mask.
//!
//! Transcribed from `src/models/kimi-k3.cpp` in upstream `llama.cpp`'s
//! Kimi-K3 pull request (ggml-org/llama.cpp#26185), together with the
//! `LLM_FFN_SITU` arm that PR adds to `llm_graph_context::build_moe_ffn`
//! and the delta-net recurrence in `src/models/delta-net-base.cpp`
//! (`build_delta_net_autoregressive`) that the released tree already
//! carries for Kimi-Linear. The multimodal projector these repos ship
//! alongside the text weights (`mmproj-*.gguf`) is out of scope, as
//! multimodal input is for every architecture here.

use anyhow::{Context, Result};
use rayon::prelude::*;
use std::sync::Arc;

use super::{ExpertGating, ExpertRouting, ModelForward, attend, rms_norm_scale, row_mean_sq};
use crate::engine::backend::Backend;
use crate::engine::kv_cache::KvCache;
use crate::engine::loader::{ExpertQuantMatrix, LoadedModel, ModelConfig, QuantMatrix};
use crate::engine::moe_stats;
use crate::engine::tensor;

/// Kimi Delta Attention: the linear-attention layer, three in every four.
struct KdaLayer {
    wq: QuantMatrix,
    wk: QuantMatrix,
    wv: QuantMatrix,
    /// The `ssm_conv1d_q`/`_k`/`_v` kernels concatenated into one
    /// `[3 * d_inner, d_conv]` buffer, channel-major.
    ///
    /// The convolution is depthwise — every channel is independent — so
    /// concatenating the three sets of channels and running one pass is
    /// exactly the three separate passes upstream runs over its one
    /// three-section conv state, and lets this share
    /// `RecurrentLayerState::conv_step` with `engine::arch::qwen35moe`
    /// unchanged.
    conv_kernel: Vec<f32>,
    /// The decay gate's low-rank projection: `[n_embd, kda_head_dim]` then
    /// `[kda_head_dim, d_inner]`.
    f_a: QuantMatrix,
    f_b: QuantMatrix,
    dt_bias: Vec<f32>,
    /// `ssm_a`, `[n_head]`. Holds `-exp(A_log)`, folded at conversion time.
    a: Vec<f32>,
    /// `[n_embd, n_head]` — the per-head delta-rule write strength.
    beta: QuantMatrix,
    /// `[n_embd, d_inner]` — the full-rank output gate.
    g: QuantMatrix,
    /// RMSNorm weight applied per head to the scan output, `[head_dim]`.
    o_norm: Vec<f32>,
    wo: QuantMatrix,
    /// Index into `KvCache::recurrent`.
    cache_index: usize,
}

/// The full-attention layer, every fourth: MLA in its absorbed form, as in
/// `engine::arch::glm`, but with no RoPE and with an output gate.
struct MlaLayer {
    wq_a: QuantMatrix,
    q_a_norm: Vec<f32>,
    wq_b: QuantMatrix,
    wkv_a_mqa: QuantMatrix,
    kv_a_norm: Vec<f32>,
    wk_b: ExpertQuantMatrix,
    wv_b: ExpertQuantMatrix,
    /// `attn_gate`, `[n_embd, n_head * value_length_mla]` — sigmoid-gates
    /// the attention output before the output projection.
    gate: Option<QuantMatrix>,
    wo: QuantMatrix,
    /// Index into `KvCache::layers`.
    cache_index: usize,
}

enum Attn {
    Kda(Box<KdaLayer>),
    Mla(Box<MlaLayer>),
}

/// The routed experts of one MoE layer, which run in the latent space,
/// plus the shared experts, which do not.
struct LatentMoe {
    /// `[n_embd, n_expert]` — scores the *full-width* input.
    gate_inp: QuantMatrix,
    exp_probs_b: Option<Vec<f32>>,
    /// `[n_embd, n_expert_latent]` / `[n_expert_latent, n_embd]`.
    routed_down: QuantMatrix,
    routed_norm: Option<Vec<f32>>,
    routed_up: QuantMatrix,
    gate_exps: ExpertQuantMatrix,
    down_exps: ExpertQuantMatrix,
    up_exps: ExpertQuantMatrix,
    gate_shexp: QuantMatrix,
    down_shexp: QuantMatrix,
    up_shexp: QuantMatrix,
}

enum Ffn {
    Dense {
        gate: QuantMatrix,
        up: QuantMatrix,
        down: QuantMatrix,
    },
    Moe(Box<LatentMoe>),
}

struct Kimi3Layer {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    /// `[n_embd]` each — the score vectors the cross-layer residual mix
    /// uses before the attention and the FFN respectively.
    attn_res_score: Vec<f32>,
    ffn_res_score: Vec<f32>,
    attn: Attn,
    ffn: Ffn,
}

pub struct Kimi3Model {
    config: ModelConfig,
    backend: Arc<dyn Backend>,
    tok_embeddings: QuantMatrix,
    output_norm: Vec<f32>,
    output_weight: QuantMatrix,
    /// The score vector for the last residual mix, after the final layer.
    output_res_score: Vec<f32>,
    /// `attn_res.block_size` — how often a layer banks its input. `0`
    /// disables cross-layer residuals entirely.
    res_block_size: usize,
    situ_beta: f32,
    situ_linear_beta: f32,
    routing: ExpertRouting,
    /// `kda.head_dim`, and the KDA layers' `n_head * kda.head_dim`.
    kda_head_dim: usize,
    d_inner: usize,
    /// `kda.gate_lower_bound`, when the file sets one.
    kda_gate_lower_bound: Option<f32>,
    kv_lora_rank: usize,
    head_k_mla: usize,
    head_v_mla: usize,
    /// Width of one cached MLA key row: `kv_lora_rank + rope dims`. The
    /// value is its leading `kv_lora_rank`.
    kv_row: usize,
    kq_scale: f32,
    kv_dims: Vec<usize>,
    recurrent_specs: Vec<(usize, usize, usize, usize)>,
    layers: Vec<Kimi3Layer>,
}

/// The situ activation, which replaces SwiGLU throughout this model:
///
/// ```text
/// situ(gate, up) = beta*tanh(gate/beta) * sigmoid(gate) * lb*tanh(up/lb)
/// ```
///
/// The gate branch is a soft-clipped SiLU — `beta*tanh(x/beta)` saturates
/// at `±beta` instead of growing linearly — and the up branch gets the same
/// soft clip at its own `linear_beta`. A `linear_beta` of zero or less
/// leaves the up branch alone, which is how upstream disables that half.
fn situ(gate: f32, up: f32, beta: f32, linear_beta: f32) -> f32 {
    let a = beta * (gate / beta).tanh() * tensor::sigmoid(gate);
    let up = if linear_beta > 0.0 {
        linear_beta * (up / linear_beta).tanh()
    } else {
        up
    };
    a * up
}

fn situ_vec(gate: &[f32], up: &[f32], beta: f32, linear_beta: f32) -> Vec<f32> {
    debug_assert_eq!(gate.len(), up.len());
    gate.iter()
        .zip(up.iter())
        .map(|(&g, &u)| situ(g, u, beta, linear_beta))
        .collect()
}

impl Kimi3Model {
    pub fn load_with_backend(loaded: &LoadedModel, backend: Arc<dyn Backend>) -> Result<Self> {
        let n_head = loaded.config.n_head;
        let n_layer = loaded.config.n_layer;

        let kda_head_dim = loaded
            .metadata_u64("kda.head_dim")
            .context("missing kda.head_dim")? as usize;
        let d_conv = loaded
            .metadata_u64("ssm.conv_kernel")
            .context("missing ssm.conv_kernel")? as usize;
        anyhow::ensure!(d_conv > 0, "ssm.conv_kernel must be at least 1");
        let d_inner = kda_head_dim * n_head;
        let kda_gate_lower_bound = loaded.metadata_f32("kda.gate_lower_bound");

        let kv_lora_rank = loaded
            .metadata_u64("attention.kv_lora_rank")
            .context("missing attention.kv_lora_rank")? as usize;
        let head_k_mla = loaded
            .metadata_u64("attention.key_length_mla")
            .context("missing attention.key_length_mla")? as usize;
        let head_v_mla = loaded
            .metadata_u64("attention.value_length_mla")
            .context("missing attention.value_length_mla")? as usize;
        let rope_dim = loaded.config.rope_dim;
        anyhow::ensure!(
            head_k_mla > rope_dim,
            "attention.key_length_mla ({head_k_mla}) must exceed rope.dimension_count ({rope_dim})"
        );
        // Not a RoPE parameter here: this model rotates nothing. It only
        // names how the cached key splits into its two halves.
        let kv_row = kv_lora_rank + rope_dim;
        let kq_scale = 1.0 / (head_k_mla as f32).sqrt();

        // `attention.head_count_kv` is a per-layer array: 0 marks a KDA
        // (recurrent) layer, non-zero a full-attention one.
        let head_count_kv = loaded
            .metadata_array_u64("attention.head_count_kv")
            .context("missing the per-layer attention.head_count_kv array")?;
        anyhow::ensure!(
            head_count_kv.len() >= n_layer,
            "attention.head_count_kv has {} entries, fewer than the {n_layer} layers",
            head_count_kv.len()
        );

        let n_expert_latent = loaded
            .metadata_u64("expert_latent_length")
            .context("missing expert_latent_length")? as usize;
        let n_layer_dense_lead = loaded
            .metadata_u64("leading_dense_block_count")
            .unwrap_or(0) as usize;
        let res_block_size = loaded.metadata_u64("attn_res.block_size").unwrap_or(0) as usize;
        let situ_beta = loaded.metadata_f32("activation.situ_beta").unwrap_or(1.0);
        let situ_linear_beta = loaded
            .metadata_f32("activation.situ_linear_beta")
            .unwrap_or(0.0);
        anyhow::ensure!(
            situ_beta > 0.0,
            "activation.situ_beta must be positive (got {situ_beta})"
        );

        let n_expert_groups = loaded.metadata_u64("expert_group_count").unwrap_or(1) as usize;
        anyhow::ensure!(
            n_expert_groups <= 1,
            "expert_group_count {n_expert_groups}: grouped expert selection is not implemented"
        );
        let routing = ExpertRouting {
            n_expert_used: loaded
                .metadata_u64("expert_used_count")
                .context("missing expert_used_count")? as usize,
            gating: ExpertGating::from_gguf(
                loaded
                    .metadata_u64("expert_gating_func")
                    .context("missing expert_gating_func")?,
            )?,
            weights_norm: loaded
                .metadata_u64("expert_weights_norm")
                .is_some_and(|v| v != 0),
            weights_scale: loaded.metadata_f32("expert_weights_scale").unwrap_or(1.0),
        };

        let tok_embeddings = loaded
            .matrix("token_embd.weight")
            .context("loading token_embd.weight")?;
        let (output_norm, _) = loaded
            .tensor("output_norm.weight")
            .context("loading output_norm.weight")?;
        let output_weight = loaded
            .matrix("output.weight")
            .context("loading output.weight")?;
        let output_res_score = if res_block_size > 0 {
            loaded
                .tensor("output_res_score.weight")
                .context("loading output_res_score.weight")?
                .0
        } else {
            Vec::new()
        };

        let mut kv_dims = Vec::new();
        let mut recurrent_specs = Vec::new();
        let mut layers = Vec::with_capacity(n_layer);
        for (i, &kv_heads) in head_count_kv.iter().take(n_layer).enumerate() {
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

            let attn = if kv_heads == 0 {
                // One conv kernel per channel, q's channels then k's then
                // v's, matching the concatenated projection fed to it.
                let mut conv_kernel = Vec::with_capacity(3 * d_inner * d_conv);
                for part in ["q", "k", "v"] {
                    let kernel = get(&format!("ssm_conv1d_{part}.weight"))?;
                    anyhow::ensure!(
                        kernel.len() == d_inner * d_conv,
                        "layer {i}'s ssm_conv1d_{part} has {} values, not d_inner * conv_kernel ({})",
                        kernel.len(),
                        d_inner * d_conv
                    );
                    conv_kernel.extend_from_slice(&kernel);
                }
                recurrent_specs.push((3 * d_inner, d_conv, n_head, kda_head_dim));
                Attn::Kda(Box::new(KdaLayer {
                    wq: get_matrix("attn_q.weight")?,
                    wk: get_matrix("attn_k.weight")?,
                    wv: get_matrix("attn_v.weight")?,
                    conv_kernel,
                    f_a: get_matrix("ssm_f_a.weight")?,
                    f_b: get_matrix("ssm_f_b.weight")?,
                    dt_bias: get("ssm_dt.bias")?,
                    a: get("ssm_a")?,
                    beta: get_matrix("ssm_beta.weight")?,
                    g: get_matrix("ssm_g.weight")?,
                    o_norm: get("ssm_norm.weight")?,
                    wo: get_matrix("attn_output.weight")?,
                    cache_index: recurrent_specs.len() - 1,
                }))
            } else {
                kv_dims.push(kv_row);
                let wkv_a_mqa = get_matrix("attn_kv_a_mqa.weight")?;
                anyhow::ensure!(
                    wkv_a_mqa.out_dim == kv_row,
                    "layer {i}'s attn_kv_a_mqa projects to {} outputs, not kv_lora_rank + rope dims ({kv_row})",
                    wkv_a_mqa.out_dim
                );
                Attn::Mla(Box::new(MlaLayer {
                    wq_a: get_matrix("attn_q_a.weight")?,
                    q_a_norm: get("attn_q_a_norm.weight")?,
                    wq_b: get_matrix("attn_q_b.weight")?,
                    wkv_a_mqa,
                    kv_a_norm: get("attn_kv_a_norm.weight")?,
                    wk_b: get_expert_matrix("attn_k_b.weight")?,
                    wv_b: get_expert_matrix("attn_v_b.weight")?,
                    gate: get_matrix("attn_gate.weight").ok(),
                    wo: get_matrix("attn_output.weight")?,
                    cache_index: kv_dims.len() - 1,
                }))
            };

            let ffn = if i < n_layer_dense_lead {
                Ffn::Dense {
                    gate: get_matrix("ffn_gate.weight")?,
                    up: get_matrix("ffn_up.weight")?,
                    down: get_matrix("ffn_down.weight")?,
                }
            } else {
                Ffn::Moe(Box::new(LatentMoe {
                    gate_inp: get_matrix("ffn_gate_inp.weight")?,
                    exp_probs_b: get("exp_probs_b.bias").ok(),
                    routed_down: get_matrix("ffn_routed_down.weight")?,
                    routed_norm: get("ffn_routed_norm.weight").ok(),
                    routed_up: get_matrix("ffn_routed_up.weight")?,
                    gate_exps: get_expert_matrix("ffn_gate_exps.weight")?,
                    down_exps: get_expert_matrix("ffn_down_exps.weight")?,
                    up_exps: get_expert_matrix("ffn_up_exps.weight")?,
                    gate_shexp: get_matrix("ffn_gate_shexp.weight")?,
                    down_shexp: get_matrix("ffn_down_shexp.weight")?,
                    up_shexp: get_matrix("ffn_up_shexp.weight")?,
                }))
            };
            if let Ffn::Moe(moe) = &ffn {
                anyhow::ensure!(
                    moe.gate_exps.in_dim == n_expert_latent,
                    "layer {i}'s routed experts read {} inputs, not expert_latent_length ({n_expert_latent})",
                    moe.gate_exps.in_dim
                );
            }

            layers.push(Kimi3Layer {
                attn_norm: get("attn_norm.weight")?,
                ffn_norm: get("ffn_norm.weight")?,
                attn_res_score: if res_block_size > 0 {
                    get("attn_res_score.weight")?
                } else {
                    Vec::new()
                },
                ffn_res_score: if res_block_size > 0 {
                    get("ffn_res_score.weight")?
                } else {
                    Vec::new()
                },
                attn,
                ffn,
            });
        }

        Ok(Self {
            config: loaded.config.clone(),
            backend,
            tok_embeddings,
            output_norm,
            output_weight,
            output_res_score,
            res_block_size,
            situ_beta,
            situ_linear_beta,
            routing,
            kda_head_dim,
            d_inner,
            kda_gate_lower_bound,
            kv_lora_rank,
            head_k_mla,
            head_v_mla,
            kv_row,
            kq_scale,
            kv_dims,
            recurrent_specs,
            layers,
        })
    }

    fn run_layers(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<Vec<f32>> {
        let n_tokens = tokens.len();
        let n_embd = self.config.n_embd;

        let mut x = vec![0f32; n_tokens * n_embd];
        for (t, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            anyhow::ensure!(
                tok < self.config.n_vocab,
                "token id {tok} is out of vocab range"
            );
            x[t * n_embd..(t + 1) * n_embd].copy_from_slice(&self.tok_embeddings.row(tok));
        }

        // The banked cross-layer residual checkpoints, each a full
        // `[n_tokens, n_embd]` slab.
        let mut banked: Vec<Vec<f32>> = Vec::new();
        for (il, layer) in self.layers.iter().enumerate() {
            let mut cur = self.res_mix(&banked, &x, &layer.attn_res_score, n_tokens);
            // A checkpoint layer banks its *raw* input, and the residual
            // stream then restarts from this layer's attention output —
            // the bank is what carries the old stream forward.
            let is_checkpoint = self.res_block_size > 0 && il % self.res_block_size == 0;
            if is_checkpoint {
                banked.push(x.clone());
            }

            tensor::rmsnorm_inplace(
                &mut cur,
                &layer.attn_norm,
                n_tokens,
                n_embd,
                self.config.rms_eps,
            );
            let attn = match &layer.attn {
                Attn::Kda(kda) => self.kda_layer(kda, cache, &cur, n_tokens),
                Attn::Mla(mla) => self.mla_layer(mla, cache, &cur, n_tokens, start_pos),
            };
            if is_checkpoint {
                x = attn;
            } else {
                tensor::add_inplace(&mut x, &attn);
            }

            let mut cur = self.res_mix(&banked, &x, &layer.ffn_res_score, n_tokens);
            tensor::rmsnorm_inplace(
                &mut cur,
                &layer.ffn_norm,
                n_tokens,
                n_embd,
                self.config.rms_eps,
            );
            let ffn = match &layer.ffn {
                Ffn::Dense { gate, up, down } => {
                    let gate = self.backend.matmul(&cur, n_tokens, gate);
                    let up = self.backend.matmul(&cur, n_tokens, up);
                    let h = situ_vec(&gate, &up, self.situ_beta, self.situ_linear_beta);
                    self.backend.matmul(&h, n_tokens, down)
                }
                Ffn::Moe(moe) => self.latent_moe(moe, &cur, n_tokens),
            };
            tensor::add_inplace(&mut x, &ffn);
        }

        let mut out = self.res_mix(&banked, &x, &self.output_res_score, n_tokens);
        tensor::rmsnorm_inplace(
            &mut out,
            &self.output_norm,
            n_tokens,
            n_embd,
            self.config.rms_eps,
        );
        Ok(out)
    }

    /// The cross-layer residual mix: a softmax over the banked
    /// checkpoints and the current stream, then that convex combination of
    /// them.
    ///
    /// Each candidate's score is its RMS-normalized value dotted with
    /// `score_w`, but the *weighted sum is over the raw values* — the norm
    /// only decides the weights. A no-op until the first checkpoint has
    /// been banked, which is why layer 0 sees the plain embedding.
    fn res_mix(
        &self,
        banked: &[Vec<f32>],
        cur: &[f32],
        score_w: &[f32],
        n_tokens: usize,
    ) -> Vec<f32> {
        if banked.is_empty() || self.res_block_size == 0 {
            return cur.to_vec();
        }
        let n_embd = self.config.n_embd;
        let eps = self.config.rms_eps;
        // `rms_norm(x) . w` is `(x . w) * scale`, so no normalized copy of
        // the row is ever materialized just to be dotted and dropped.
        let score = |row: &[f32]| tensor::dot(row, score_w) * rms_norm_scale(row_mean_sq(row), eps);

        let mut out = vec![0f32; n_tokens * n_embd];
        for t in 0..n_tokens {
            let span = t * n_embd..(t + 1) * n_embd;
            let mut scores: Vec<f32> = banked.iter().map(|b| score(&b[span.clone()])).collect();
            scores.push(score(&cur[span.clone()]));
            tensor::softmax_inplace(&mut scores);

            let dst = &mut out[span.clone()];
            for (b, &p) in banked.iter().zip(scores.iter()) {
                tensor::axpy_inplace(dst, &b[span.clone()], p);
            }
            tensor::axpy_inplace(dst, &cur[span.clone()], scores[banked.len()]);
        }
        out
    }

    /// One Kimi Delta Attention layer: a short causal convolution over each
    /// of Q/K/V, then the delta rule with a per-dimension decay, then a
    /// gated per-head norm.
    fn kda_layer(
        &self,
        layer: &KdaLayer,
        cache: &mut KvCache,
        normed: &[f32],
        n_tokens: usize,
    ) -> Vec<f32> {
        let n_head = self.config.n_head;
        let head_dim = self.kda_head_dim;
        let d_inner = self.d_inner;
        let eps = self.config.rms_eps;
        let q_scale = 1.0 / (head_dim as f32).sqrt();

        // Token-independent projections, batched over the whole chunk. The
        // three conv inputs are concatenated to match `conv_kernel`.
        let q = self.backend.matmul(normed, n_tokens, &layer.wq);
        let k = self.backend.matmul(normed, n_tokens, &layer.wk);
        let v = self.backend.matmul(normed, n_tokens, &layer.wv);
        let f = {
            let low = self.backend.matmul(normed, n_tokens, &layer.f_a);
            self.backend.matmul(&low, n_tokens, &layer.f_b)
        };
        let beta = self.backend.matmul(normed, n_tokens, &layer.beta);
        let gate = self.backend.matmul(normed, n_tokens, &layer.g);

        let mut out = vec![0f32; n_tokens * d_inner];
        let state = &mut cache.recurrent[layer.cache_index];
        for t in 0..n_tokens {
            let mut qkv = Vec::with_capacity(3 * d_inner);
            qkv.extend_from_slice(&q[t * d_inner..(t + 1) * d_inner]);
            qkv.extend_from_slice(&k[t * d_inner..(t + 1) * d_inner]);
            qkv.extend_from_slice(&v[t * d_inner..(t + 1) * d_inner]);
            let mut conv = state.conv_step(&qkv, &layer.conv_kernel);
            for value in conv.iter_mut() {
                *value = tensor::silu(*value);
            }
            let (q_t, rest) = conv.split_at_mut(d_inner);
            let (k_t, v_t) = rest.split_at_mut(d_inner);

            // The decay: `lower_bound * sigmoid(exp(A_log) * (f + dt_bias))`
            // when the file sets a lower bound, and `-exp(A_log) *
            // softplus(f + dt_bias)` when it does not. `ssm_a` holds
            // `-exp(A_log)` either way, so the first form negates it back.
            // The scan exponentiates this, making the per-dimension decay
            // `exp(lower_bound * sigmoid(..))`, which lives in
            // `(e^lower_bound, 1)`.
            let mut decay = vec![0f32; d_inner];
            for h in 0..n_head {
                for j in 0..head_dim {
                    let idx = h * head_dim + j;
                    let pre = f[t * d_inner + idx] + layer.dt_bias[idx];
                    decay[idx] = match self.kda_gate_lower_bound {
                        Some(bound) => bound * tensor::sigmoid(-(pre * layer.a[h])),
                        None => layer.a[h] * tensor::softplus(pre),
                    }
                    .exp();
                }
            }

            let dst = &mut out[t * d_inner..(t + 1) * d_inner];
            for h in 0..n_head {
                let q_h = &mut q_t[h * head_dim..(h + 1) * head_dim];
                tensor::l2_norm_inplace(q_h, eps);
                for value in q_h.iter_mut() {
                    *value *= q_scale;
                }
                let k_h = &mut k_t[h * head_dim..(h + 1) * head_dim];
                tensor::l2_norm_inplace(k_h, eps);
                let q_h = &q_t[h * head_dim..(h + 1) * head_dim];
                let k_h = &k_t[h * head_dim..(h + 1) * head_dim];
                let v_h = &v_t[h * head_dim..(h + 1) * head_dim];
                let beta_h = tensor::sigmoid(beta[t * n_head + h]);
                let decay_h = &decay[h * head_dim..(h + 1) * head_dim];

                // The delta rule, exactly `build_delta_net_autoregressive`:
                // state[i][j] decays by `decay[j]` (per *dimension* here,
                // where a plain gated delta-net decays a whole head by one
                // scalar), absorbs `k[i] * beta*(v - k^T state)[j]`, and is
                // read out with the query.
                let state_h = state.delta_state_mut(h);
                for i in 0..head_dim {
                    let row = &mut state_h[i * head_dim..(i + 1) * head_dim];
                    for (s, &d) in row.iter_mut().zip(decay_h.iter()) {
                        *s *= d;
                    }
                }
                let mut sk = vec![0f32; head_dim];
                for i in 0..head_dim {
                    tensor::axpy_inplace(
                        &mut sk,
                        &state_h[i * head_dim..(i + 1) * head_dim],
                        k_h[i],
                    );
                }
                let delta: Vec<f32> = (0..head_dim).map(|j| beta_h * (v_h[j] - sk[j])).collect();
                for i in 0..head_dim {
                    tensor::axpy_inplace(
                        &mut state_h[i * head_dim..(i + 1) * head_dim],
                        &delta,
                        k_h[i],
                    );
                }
                let mut o = vec![0f32; head_dim];
                for i in 0..head_dim {
                    tensor::axpy_inplace(
                        &mut o,
                        &state_h[i * head_dim..(i + 1) * head_dim],
                        q_h[i],
                    );
                }

                // Gated RMSNorm: norm the scan output, then scale it by a
                // sigmoid gate read from the layer input (Kimi-Linear
                // factors this gate into two matrices; K3 has one).
                tensor::rmsnorm_inplace(&mut o, &layer.o_norm, 1, head_dim, eps);
                let gate_h = &gate[t * d_inner + h * head_dim..t * d_inner + (h + 1) * head_dim];
                for (value, &g) in o.iter_mut().zip(gate_h.iter()) {
                    *value *= tensor::sigmoid(g);
                }
                dst[h * head_dim..(h + 1) * head_dim].copy_from_slice(&o);
            }
        }

        self.backend.matmul(&out, n_tokens, &layer.wo)
    }

    /// One absorbed-MLA layer. Identical in shape to `engine::arch::glm`'s,
    /// minus the RoPE (this model rotates nothing) and the lightning
    /// indexer (it has none), plus a sigmoid gate on the output.
    fn mla_layer(
        &self,
        layer: &MlaLayer,
        cache: &mut KvCache,
        normed: &[f32],
        n_tokens: usize,
        start_pos: usize,
    ) -> Vec<f32> {
        let n_head = self.config.n_head;
        let eps = self.config.rms_eps;
        let nope = self.head_k_mla - self.config.rope_dim;
        let absorbed_dim = self.kv_lora_rank + self.config.rope_dim;

        let mut qr = self.backend.matmul(normed, n_tokens, &layer.wq_a);
        tensor::rmsnorm_inplace(&mut qr, &layer.q_a_norm, n_tokens, layer.wq_a.out_dim, eps);
        let q = self.backend.matmul(&qr, n_tokens, &layer.wq_b);
        let mut kv = self.backend.matmul(normed, n_tokens, &layer.wkv_a_mqa);
        for t in 0..n_tokens {
            tensor::rmsnorm_inplace(
                &mut kv[t * self.kv_row..t * self.kv_row + self.kv_lora_rank],
                &layer.kv_a_norm,
                1,
                self.kv_lora_rank,
                eps,
            );
        }

        let mut attn_out = vec![0f32; n_tokens * n_head * self.head_v_mla];
        for t in 0..n_tokens {
            let kv_t = &kv[t * self.kv_row..(t + 1) * self.kv_row];
            cache.layers[layer.cache_index].push(kv_t, kv_t);

            let n_keys = start_pos + t + 1;
            let mut keys = vec![0f32; n_keys * self.kv_row];
            {
                let slot = &cache.layers[layer.cache_index];
                for p in 0..n_keys {
                    keys[p * self.kv_row..(p + 1) * self.kv_row].copy_from_slice(slot.key_at(
                        p,
                        0,
                        self.kv_row,
                    ));
                }
            }

            let q_t = &q[t * n_head * self.head_k_mla..(t + 1) * n_head * self.head_k_mla];
            let heads: Vec<Vec<f32>> = (0..n_head)
                .into_par_iter()
                .map(|h| {
                    // Absorb the query through this head's key
                    // decompression, then carry its second half unchanged.
                    let q_nope = &q_t[h * self.head_k_mla..h * self.head_k_mla + nope];
                    let mut q_h = vec![0f32; absorbed_dim];
                    for (j, out) in q_h[..self.kv_lora_rank].iter_mut().enumerate() {
                        *out = tensor::dot(q_nope, &layer.wk_b.row(h, j));
                    }
                    q_h[self.kv_lora_rank..].copy_from_slice(
                        &q_t[h * self.head_k_mla + nope..(h + 1) * self.head_k_mla],
                    );

                    let compressed = attend(
                        &q_h,
                        &keys,
                        self.kv_row,
                        self.kv_lora_rank,
                        self.kq_scale,
                        None,
                    );
                    (0..self.head_v_mla)
                        .map(|d| tensor::dot(&compressed, &layer.wv_b.row(h, d)))
                        .collect()
                })
                .collect();
            for (h, head) in heads.iter().enumerate() {
                let at = (t * n_head + h) * self.head_v_mla;
                attn_out[at..at + self.head_v_mla].copy_from_slice(head);
            }
        }

        if let Some(gate) = &layer.gate {
            let gate = self.backend.matmul(normed, n_tokens, gate);
            for (o, &g) in attn_out.iter_mut().zip(gate.iter()) {
                *o *= tensor::sigmoid(g);
            }
        }
        self.backend.matmul(&attn_out, n_tokens, &layer.wo)
    }

    /// The latent MoE: route on the full-width input, run the selected
    /// experts in the latent space, norm and project back up, then add the
    /// shared experts — which read the full-width input directly.
    fn latent_moe(&self, moe: &LatentMoe, normed: &[f32], n_tokens: usize) -> Vec<f32> {
        let n_embd = self.config.n_embd;
        let latent = moe.gate_exps.in_dim;
        let routed_in = self.backend.matmul(normed, n_tokens, &moe.routed_down);

        let mut latent_out = vec![0f32; n_tokens * latent];
        let mut experts =
            moe_stats::LayerRecorder::for_tensors(&[&moe.gate_exps, &moe.up_exps, &moe.down_exps]);
        // The router scores the full-width input, so routing the whole batch
        // up front costs nothing extra and lets the experts — which run in
        // the latent space — be grouped. See `super::evaluate_routed_experts`.
        let mut selection: Vec<Vec<(usize, f32)>> = (0..n_tokens)
            .map(|t| {
                let x_t = &normed[t * n_embd..(t + 1) * n_embd];
                let logits = self.backend.matmul(x_t, 1, &moe.gate_inp);
                let (selected, weights) =
                    self.routing
                        .route(&logits, moe.exp_probs_b.as_deref(), None);
                selected.into_iter().zip(weights).collect()
            })
            .collect();
        // Trim to the expert budget *before* anything is recorded or
        // read: the counters should describe the work actually done,
        // and a dropped expert's weights must never be fetched.
        super::apply_expert_budget(&mut selection, &moe.gate_exps);
        for picks in &selection {
            picks.iter().for_each(|&(e, _)| experts.select(e));
        }

        let contribs = super::evaluate_routed_experts(&selection, |expert, members| {
            let inputs: Vec<&[f32]> = members
                .iter()
                .map(|&(t, _)| &routed_in[t * latent..(t + 1) * latent])
                .collect();
            let gate =
                super::project_expert(&moe.gate_exps, expert, 0, moe.gate_exps.out_dim, &inputs);
            let up = super::project_expert(&moe.up_exps, expert, 0, moe.up_exps.out_dim, &inputs);
            let hidden: Vec<Vec<f32>> = gate
                .into_iter()
                .zip(up)
                .map(|(gate, up)| situ_vec(&gate, &up, self.situ_beta, self.situ_linear_beta))
                .collect();
            let hidden_refs: Vec<&[f32]> = hidden.iter().map(Vec::as_slice).collect();
            super::project_expert(
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
        });
        experts.loaded_once_per_distinct_expert();

        for t in 0..n_tokens {
            let dst = &mut latent_out[t * latent..(t + 1) * latent];
            for contrib in &contribs[t] {
                tensor::add_inplace(dst, contrib);
            }
        }
        experts.commit(n_tokens);

        if let Some(norm) = &moe.routed_norm {
            tensor::rmsnorm_inplace(&mut latent_out, norm, n_tokens, latent, self.config.rms_eps);
        }
        let mut out = self.backend.matmul(&latent_out, n_tokens, &moe.routed_up);

        let gate = self.backend.matmul(normed, n_tokens, &moe.gate_shexp);
        let up = self.backend.matmul(normed, n_tokens, &moe.up_shexp);
        let h = situ_vec(&gate, &up, self.situ_beta, self.situ_linear_beta);
        let shared = self.backend.matmul(&h, n_tokens, &moe.down_shexp);
        tensor::add_inplace(&mut out, &shared);
        out
    }
}

impl ModelForward for Kimi3Model {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn new_kv_cache(&self, capacity: usize) -> KvCache {
        KvCache::new_mixed(capacity, &self.kv_dims, &self.recurrent_specs)
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
    use super::{situ, situ_vec};

    /// The gate branch is a soft-clipped SiLU: it tracks `x*sigmoid(x)`
    /// near zero and saturates at `beta` instead of growing without bound.
    #[test]
    fn situ_saturates_the_gate_branch_at_beta() {
        let beta = 4.0;
        // Far above beta, `beta*tanh(gate/beta)` is beta and
        // `sigmoid(gate)` is 1, so the gate contributes ~beta.
        let big = situ(100.0, 1.0, beta, 0.0);
        assert!((big - beta).abs() < 1e-3, "{big}");
        // Far below, the sigmoid drives it to zero.
        assert!(situ(-100.0, 1.0, beta, 0.0).abs() < 1e-6);
        // Near zero it is close to SiLU (tanh(x/beta)*beta ~ x).
        let small = situ(0.5, 1.0, beta, 0.0);
        let silu = crate::engine::tensor::silu(0.5);
        assert!((small - silu).abs() < 0.02, "{small} vs {silu}");
    }

    /// A non-positive `linear_beta` leaves the up branch alone; a positive
    /// one soft-clips it the same way.
    #[test]
    fn the_linear_beta_transform_applies_only_when_positive() {
        // Disabled: the up value passes through untouched.
        let raw = situ(1.0, 1000.0, 4.0, 0.0);
        let gate_only = situ(1.0, 1.0, 4.0, 0.0);
        assert!((raw - gate_only * 1000.0).abs() < 1e-2, "{raw}");
        // Enabled: the up value saturates at linear_beta.
        let clipped = situ(1.0, 1000.0, 4.0, 25.0);
        assert!((clipped - gate_only * 25.0).abs() < 1e-2, "{clipped}");
    }

    #[test]
    fn situ_vec_matches_the_scalar_form_elementwise() {
        let gate = vec![-2.0, 0.0, 3.0];
        let up = vec![1.0, 2.0, 3.0];
        let got = situ_vec(&gate, &up, 4.0, 25.0);
        for i in 0..3 {
            assert!((got[i] - situ(gate[i], up[i], 4.0, 25.0)).abs() < 1e-9);
        }
    }
}
