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
//! `engine::arch::glm`). Both halves live in
//! [`engine::arch::kda`](super::kda), shared with
//! `engine::arch::bailingmoe`, which alternates the same pair — and which
//! is how this module's delta rule came to be checked against a real
//! checkpoint at all, since no `kimi-k3` GGUF was on hand when it was
//! written. What this module adds on top is five things neither half has:
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
//!    same thing as `ssm_g_a`/`ssm_g_b` — one of the load-time choices
//!    [`kda::KdaNames`](super::kda::KdaNames) carries.
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
use std::sync::Arc;

use super::kda::{KIMI3_KDA_NAMES, KdaLayer, KdaShape, MlaLayer, MlaShape};
use super::{ExpertGating, ExpertRouting, ModelForward, rms_norm_scale, row_mean_sq};
use crate::engine::backend::Backend;
use crate::engine::kv_cache::{KvCache, RecurrentSpec};
use crate::engine::loader::{ExpertQuantMatrix, LoadedModel, ModelConfig, QuantMatrix};
use crate::engine::moe_stats;
use crate::engine::tensor;

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
    /// The shapes the shared half-layers need — see `engine::arch::kda`.
    kda: KdaShape,
    mla: MlaShape,
    kv_dims: Vec<usize>,
    recurrent_specs: Vec<RecurrentSpec>,
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
        let kda = KdaShape {
            n_head,
            head_dim: kda_head_dim,
            d_inner: kda_head_dim * n_head,
            d_conv,
            gate_lower_bound: loaded.metadata_f32("kda.gate_lower_bound"),
            eps: loaded.config.rms_eps,
            // `kimi-k3.cpp` L2-normalizes with the model's own RMS epsilon.
            l2_eps: loaded.config.rms_eps,
        };

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
        let mla = MlaShape {
            n_head,
            kv_lora_rank,
            head_k_mla,
            head_v_mla,
            // Not a RoPE parameter here: this model rotates nothing, so
            // `rope` is `None` and `rope_dim` only names how the cached key
            // splits into its two halves.
            kv_row: kv_lora_rank + rope_dim,
            rope: None,
            rope_dim,
            kq_scale: 1.0 / (head_k_mla as f32).sqrt(),
            eps: loaded.config.rms_eps,
        };

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

        let routing = ExpertRouting {
            groups: super::ExpertGroups::from_gguf(
                loaded,
                loaded.metadata_u64("expert_count").unwrap_or(0) as usize,
            )?,
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
                recurrent_specs.push(kda.recurrent_spec());
                Attn::Kda(Box::new(KdaLayer::load(
                    loaded,
                    i,
                    &kda,
                    &KIMI3_KDA_NAMES,
                    recurrent_specs.len() - 1,
                )?))
            } else {
                kv_dims.push(mla.kv_row);
                Attn::Mla(Box::new(MlaLayer::load(
                    loaded,
                    i,
                    &mla,
                    kv_dims.len() - 1,
                )?))
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
            kda,
            mla,
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
        // Projection buffers, grown once and reused by every layer — see
        // `Backend::matmul_into`.
        let mut attn_out: Vec<f32> = Vec::new();
        let mut kda_scratch = super::kda::KdaScratch::default();
        let mut mla_scratch = super::kda::MlaScratch::default();
        let mut ffn_out: Vec<f32> = Vec::new();
        let mut ffn_gate: Vec<f32> = Vec::new();
        let mut ffn_up: Vec<f32> = Vec::new();
        // The residual mixes, one buffer per call site so none overwrites a
        // value another still needs.
        let mut attn_mix: Vec<f32> = Vec::new();
        let mut ffn_mix: Vec<f32> = Vec::new();

        for (il, layer) in self.layers.iter().enumerate() {
            self.res_mix_into(&mut attn_mix, &banked, &x, &layer.attn_res_score, n_tokens);
            let cur = &mut attn_mix;
            // A checkpoint layer banks its *raw* input, and the residual
            // stream then restarts from this layer's attention output —
            // the bank is what carries the old stream forward.
            let is_checkpoint = self.res_block_size > 0 && il % self.res_block_size == 0;
            if is_checkpoint {
                banked.push(x.clone());
            }

            tensor::rmsnorm_inplace(cur, &layer.attn_norm, n_tokens, n_embd, self.config.rms_eps);
            match &layer.attn {
                Attn::Kda(kda) => kda.forward_into(
                    self.backend.as_ref(),
                    &mut attn_out,
                    &mut kda_scratch,
                    &self.kda,
                    cache,
                    cur,
                    n_tokens,
                ),
                Attn::Mla(mla) => mla.forward_into(
                    self.backend.as_ref(),
                    &mut attn_out,
                    &mut mla_scratch,
                    &self.mla,
                    cache,
                    cur,
                    n_tokens,
                    start_pos,
                ),
            }
            if is_checkpoint {
                // The residual stream restarts from this layer's attention
                // output. A swap rather than a move: `x`'s old buffer was
                // already banked by value above, so handing it to `attn_out`
                // costs nothing and gives the next layer its capacity back.
                std::mem::swap(&mut x, &mut attn_out);
            } else {
                tensor::add_inplace(&mut x, &attn_out);
            }

            self.res_mix_into(&mut ffn_mix, &banked, &x, &layer.ffn_res_score, n_tokens);
            let cur = &mut ffn_mix;
            tensor::rmsnorm_inplace(cur, &layer.ffn_norm, n_tokens, n_embd, self.config.rms_eps);
            match &layer.ffn {
                Ffn::Dense { gate, up, down } => {
                    self.backend.matmul_into(&mut ffn_gate, cur, n_tokens, gate);
                    self.backend.matmul_into(&mut ffn_up, cur, n_tokens, up);
                    let h = situ_vec(&ffn_gate, &ffn_up, self.situ_beta, self.situ_linear_beta);
                    self.backend.matmul_into(&mut ffn_out, &h, n_tokens, down);
                }
                Ffn::Moe(moe) => ffn_out = self.latent_moe(moe, cur, n_tokens),
            }
            tensor::add_inplace(&mut x, &ffn_out);
        }

        let mut out = Vec::new();
        self.res_mix_into(&mut out, &banked, &x, &self.output_res_score, n_tokens);
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
    /// Into a caller-owned buffer — see `Backend::matmul_into`. Called
    /// three times per layer, each time for a `[n_tokens, n_embd]` result.
    fn res_mix_into(
        &self,
        out: &mut Vec<f32>,
        banked: &[Vec<f32>],
        cur: &[f32],
        score_w: &[f32],
        n_tokens: usize,
    ) {
        if banked.is_empty() || self.res_block_size == 0 {
            // Still a copy — the caller norms this in place and needs `cur`
            // itself untouched — but not an allocation. Fusing the copy into
            // that norm, the way `tensor::rmsnorm_into` does elsewhere,
            // would mean splitting this branch out at the call site.
            out.clear();
            out.extend_from_slice(cur);
            return;
        }
        let n_embd = self.config.n_embd;
        let eps = self.config.rms_eps;
        // `rms_norm(x) . w` is `(x . w) * scale`, so no normalized copy of
        // the row is ever materialized just to be dotted and dropped.
        let score = |row: &[f32]| tensor::dot(row, score_w) * rms_norm_scale(row_mean_sq(row), eps);

        // Accumulated into, not overwritten — see `tensor::zeroed_to`.
        let out = tensor::zeroed_to(out, n_tokens * n_embd);
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
    }

    /// The latent MoE: route on the full-width input, run the selected
    /// experts in the latent space, norm and project back up, then add the
    /// shared experts — which read the full-width input directly.
    fn latent_moe(&self, moe: &LatentMoe, normed: &[f32], n_tokens: usize) -> Vec<f32> {
        let latent = moe.gate_exps.in_dim;
        let routed_in = self.backend.matmul(normed, n_tokens, &moe.routed_down);

        let mut latent_out = vec![0f32; n_tokens * latent];
        let mut experts =
            moe_stats::LayerRecorder::for_tensors(&[&moe.gate_exps, &moe.up_exps, &moe.down_exps]);
        // The router scores the full-width input, so routing the whole batch
        // up front costs nothing extra and lets the experts — which run in
        // the latent space — be grouped. See `super::evaluate_routed_experts`.
        let logits =
            super::moe_router_logits(self.backend.as_ref(), normed, n_tokens, &moe.gate_inp);
        let mut selection =
            super::route_batch(&self.routing, &logits, n_tokens, moe.exp_probs_b.as_deref());
        // Trim to the expert budget *before* anything is recorded or
        // read: the counters should describe the work actually done,
        // and a dropped expert's weights must never be fetched.
        super::apply_expert_budget(&mut selection, &moe.gate_exps);
        for picks in &selection {
            picks.iter().for_each(|&(e, _)| experts.select(e));
        }

        // The GPU expert path batches the three projections across experts —
        // see `super::evaluate_routed_experts_batched`.
        let contribs = if super::gpu_experts() && self.backend.as_wgpu().is_some() {
            super::evaluate_routed_experts_batched(
                self.backend.as_ref(),
                &selection,
                &routed_in,
                latent,
                &moe.gate_exps,
                &moe.up_exps,
                &moe.down_exps,
                |gate, up| situ_vec(gate, up, self.situ_beta, self.situ_linear_beta),
            )
        } else {
            super::evaluate_routed_experts(&selection, |expert, members| {
                let inputs: Vec<&[f32]> = members
                    .iter()
                    .map(|&(t, _)| &routed_in[t * latent..(t + 1) * latent])
                    .collect();
                let gate = super::project_expert(
                    self.backend.as_ref(),
                    &moe.gate_exps,
                    expert,
                    0,
                    moe.gate_exps.out_dim,
                    &inputs,
                );
                let up = super::project_expert(
                    self.backend.as_ref(),
                    &moe.up_exps,
                    expert,
                    0,
                    moe.up_exps.out_dim,
                    &inputs,
                );
                let hidden: Vec<Vec<f32>> = gate
                    .into_iter()
                    .zip(up)
                    .map(|(gate, up)| situ_vec(&gate, &up, self.situ_beta, self.situ_linear_beta))
                    .collect();
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
    /// The instrumentation hook `engine::generate` reads to count GPU
    /// submissions per decode step. Only useful compared *between*
    /// architectures, which is why an architecture that answers `None` is
    /// invisible to exactly the measurement that would say whether a
    /// cross-architecture change helped it.
    fn vulkan_backend(&self) -> Option<&crate::engine::backend::vulkan::VulkanBackend> {
        self.backend.as_wgpu()
    }

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
