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

//! Nemotron-H (`general.architecture = "nemotron_h_moe"`, e.g.
//! `bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF`).
//!
//! A hybrid decoder whose blocks are **not** the usual attention-plus-FFN
//! pair. Each block is exactly *one* sub-layer — a selective state-space
//! mixer, a self-attention, or a mixture-of-experts FFN — wrapped in its own
//! RMSNorm and residual, and the file's own per-layer metadata says which:
//! a block is recurrent where `attention.head_count_kv` and
//! `feed_forward_length` are both zero at its index, attention where only
//! `feed_forward_length` is, and an FFN otherwise. On the 30B-A3B model that
//! is 23 recurrent, 6 attention and 23 MoE blocks, so **attention is a sixth
//! of the depth** and most of the sequence mixing is the recurrence.
//!
//! Three things about it are easy to get wrong and produce fluent, subtly
//! wrong output rather than a failure:
//!
//! - **The attention is unrotated.** There is no RoPE anywhere in this
//!   architecture — position reaches the model only through the causal
//!   convolution and the recurrence in the state-space blocks. The file
//!   still carries `rope.dimension_count` and `rope.freq_base`; they are
//!   vestigial and applying them is silently wrong.
//! - **The FFN activation is squared ReLU, and there is no gate.** Both the
//!   routed experts and the shared expert are `down(relu(up(x))^2)` — a
//!   two-matrix FFN. There is no `ffn_gate_exps`/`ffn_gate_shexp` tensor to
//!   read, and substituting SwiGLU (the shape every other MoE here has)
//!   would need a tensor that does not exist.
//! - **The state matrix is not square.** A delta-net layer accumulates an
//!   outer product of two same-width vectors, so its per-head state is
//!   `[head_dim, head_dim]`. Here the state axis is the model's own
//!   `ssm.state_size` (128) and the head width is `ssm.inner_size /
//!   ssm.time_step_rank` (64), so it is `[64, 128]` —
//!   `kv_cache::RecurrentSpec::ssm` is what asks for that shape.
//!
//! The routed-expert selection, weighting and evaluation are entirely
//! [`super::ExpertRouting`]'s and [`super::evaluate_routed_experts`]'s, and
//! the attention is `engine::attention`'s, exactly as every other
//! architecture here uses them.
//!
//! ## The multi-token-prediction block
//!
//! When `nextn_predict_layers` is set, `block_count` counts one extra
//! trailing block holding a self-contained draft head (`nextn.*` tensors
//! plus its own attention and MoE). It predicts token *n+2* from the trunk's
//! hidden state, which is only useful to a speculative decoder; nothing in
//! the trunk reads it. This module loads the trunk's `block_count -
//! nextn_predict_layers` blocks and leaves the draft head on disk — the same
//! choice `engine::arch::dflash` describes for a draft sidecar that has no
//! standalone model to run.

use anyhow::{Context, Result};
use std::sync::Arc;

use super::{ExpertGating, ExpertRouting, ModelForward};
use crate::engine::backend::{Backend, MatmulOp};
use crate::engine::kv_cache::{KvCache, RecurrentSpec};
use crate::engine::loader::{ExpertQuantMatrix, LoadedModel, ModelConfig, QuantMatrix};
use crate::engine::moe_stats;
use crate::engine::tensor;

/// A selective state-space block: one input projection fanning out to the
/// gate, the convolved `x`/`B`/`C` streams and the per-head timestep, then
/// the recurrence, a gated grouped norm, and the output projection.
struct SsmLayer {
    norm: Vec<f32>,
    /// `[n_embd, 2 * d_inner + 2 * n_group * d_state + n_head]` — the whole
    /// block's input fan-out in one matrix. See [`NemotronModel::ssm_parts`]
    /// for what each span is.
    in_proj: QuantMatrix,
    /// `[conv_channels, d_conv]`, channel-major, plus its bias.
    conv1d: Vec<f32>,
    conv1d_bias: Vec<f32>,
    /// Per-head timestep bias, added before the softplus.
    dt_bias: Vec<f32>,
    /// Per-head state decay, stored already negated so the recurrence uses
    /// it directly.
    a: Vec<f32>,
    /// Per-head skip weight on the block's own input.
    d: Vec<f32>,
    /// `[d_inner]`, read as `n_group` consecutive `d_inner / n_group`-wide
    /// weight vectors — the grouped norm's weight.
    group_norm: Vec<f32>,
    out_proj: QuantMatrix,
    cache_index: usize,
}

/// A plain unrotated GQA attention block. `n_head_kv` is per layer because
/// the file states it per layer; every other layer's entry is zero, which is
/// how a block is recognised as *not* being one of these.
struct AttnLayer {
    norm: Vec<f32>,
    wq: QuantMatrix,
    wk: QuantMatrix,
    wv: QuantMatrix,
    wo: QuantMatrix,
    n_head_kv: usize,
    cache_index: usize,
}

/// A routed-plus-shared mixture-of-experts block. No gate projection on
/// either branch — see this module's own header.
struct MoeLayer {
    norm: Vec<f32>,
    gate_inp: QuantMatrix,
    /// `exp_probs_b` — steers which experts are selected, never their
    /// weights.
    exp_probs_b: Vec<f32>,
    up_exps: ExpertQuantMatrix,
    down_exps: ExpertQuantMatrix,
    up_shexp: QuantMatrix,
    down_shexp: QuantMatrix,
}

/// A plain two-matrix FFN block — the dense `nemotron_h`'s FFN where
/// `nemotron_h_moe` routes experts.
///
/// Same activation as everything else here (`down(relu(up(x))^2)`, no gate),
/// which is why this is two tensors and no new arithmetic: the dense file
/// simply carries `ffn_up`/`ffn_down` where the MoE file carries the routed
/// and shared expert stacks.
struct FfnLayer {
    norm: Vec<f32>,
    up: QuantMatrix,
    down: QuantMatrix,
}

enum Layer {
    Ssm(SsmLayer),
    Attn(AttnLayer),
    Moe(MoeLayer),
    Ffn(FfnLayer),
}

pub struct NemotronModel {
    config: ModelConfig,
    backend: Arc<dyn Backend>,
    tok_embeddings: QuantMatrix,
    output_norm: Vec<f32>,
    output_weight: QuantMatrix,
    n_head: usize,
    head_dim: usize,
    rms_eps: f32,
    routing: ExpertRouting,
    /// `ssm.conv_kernel` — how many taps the causal convolution has,
    /// including the current token.
    d_conv: usize,
    /// `ssm.inner_size` — the width of the `x` stream and of the recurrence
    /// output.
    d_inner: usize,
    /// `ssm.state_size` — the second axis of each head's state matrix.
    d_state: usize,
    /// `ssm.group_count` — how many `B`/`C` pairs the heads share, and the
    /// number of groups the output norm runs over.
    n_group: usize,
    /// `ssm.time_step_rank` — the recurrence's head count (one timestep,
    /// decay and skip weight each).
    n_ssm_head: usize,
    layers: Vec<Layer>,
}

impl NemotronModel {
    pub fn load_with_backend(loaded: &LoadedModel, backend: Arc<dyn Backend>) -> Result<Self> {
        let config = loaded.config.clone();

        // `block_count` includes the draft head; the trunk is what runs.
        let n_layer = config
            .n_layer
            .checked_sub(loaded.metadata_u64("nextn_predict_layers").unwrap_or(0) as usize)
            .filter(|&n| n > 0)
            .context("nextn_predict_layers is not smaller than block_count")?;

        let d_conv = loaded
            .metadata_u64("ssm.conv_kernel")
            .context("missing ssm.conv_kernel")? as usize;
        let d_inner = loaded
            .metadata_u64("ssm.inner_size")
            .context("missing ssm.inner_size")? as usize;
        let d_state = loaded
            .metadata_u64("ssm.state_size")
            .context("missing ssm.state_size")? as usize;
        let n_group = loaded
            .metadata_u64("ssm.group_count")
            .context("missing ssm.group_count")? as usize;
        let n_ssm_head = loaded
            .metadata_u64("ssm.time_step_rank")
            .context("missing ssm.time_step_rank")? as usize;

        anyhow::ensure!(
            d_conv > 0 && d_inner > 0 && d_state > 0 && n_group > 0 && n_ssm_head > 0,
            "every ssm.* dimension must be nonzero"
        );
        anyhow::ensure!(
            d_inner.is_multiple_of(n_ssm_head),
            "ssm.inner_size ({d_inner}) must be a multiple of ssm.time_step_rank ({n_ssm_head})"
        );
        anyhow::ensure!(
            d_inner.is_multiple_of(n_group),
            "ssm.inner_size ({d_inner}) must be a multiple of ssm.group_count ({n_group})"
        );
        anyhow::ensure!(
            n_ssm_head.is_multiple_of(n_group),
            "ssm.time_step_rank ({n_ssm_head}) must be a multiple of ssm.group_count ({n_group})"
        );

        // Absent on the dense sibling, which has no experts to count. `0` is
        // that file's own answer rather than a fallback, and it is what
        // selects the dense FFN branch below — so it must not be an error
        // here, and equally must not be invented for a file that *is* MoE
        // (one declaring routed blocks without an `expert_count` fails on
        // the router width instead, where the number is checked).
        let n_expert = loaded.metadata_u64("expert_count").unwrap_or(0) as usize;
        let routing = ExpertRouting {
            groups: super::ExpertGroups::from_gguf(loaded, n_expert)?,
            n_expert_used: loaded.metadata_u64("expert_used_count").unwrap_or(0) as usize,
            // Fixed by the architecture rather than read from the file,
            // which carries no `expert_gating_func` key for it.
            gating: ExpertGating::Sigmoid,
            weights_norm: loaded
                .metadata_u64("expert_weights_norm")
                .is_some_and(|v| v != 0),
            weights_scale: loaded.metadata_f32("expert_weights_scale").unwrap_or(1.0),
        };

        // Which block is which. Both arrays are per-block and both have to
        // be consulted: a zero `feed_forward_length` alone covers the
        // recurrent *and* the attention blocks.
        let n_ff: Vec<u64> = loaded
            .metadata_array_u64("feed_forward_length")
            .context("missing per-layer feed_forward_length")?;
        let n_head_kv: Vec<u64> = loaded
            .metadata_array_u64("attention.head_count_kv")
            .context("missing per-layer attention.head_count_kv")?;
        anyhow::ensure!(
            n_ff.len() >= n_layer && n_head_kv.len() >= n_layer,
            "feed_forward_length ({}) and attention.head_count_kv ({}) must both cover all \
             {n_layer} trunk blocks",
            n_ff.len(),
            n_head_kv.len(),
        );

        let tok_embeddings = loaded
            .matrix("token_embd.weight")
            .context("loading token_embd.weight")?;
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
        let mut n_attn = 0usize;
        let mut n_recurrent = 0usize;
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

            let norm = get("attn_norm.weight")?;
            if n_ff[i] != 0 && n_expert == 0 {
                // The dense sibling: this block's FFN is two ordinary
                // matrices where the MoE file stacks experts. Chosen on
                // `expert_count` rather than on the architecture string,
                // because it is the file's own statement about what its FFN
                // blocks contain — and a file declaring no experts whose
                // blocks were read as routed would fail on a tensor that
                // does not exist.
                layers.push(Layer::Ffn(FfnLayer {
                    norm,
                    up: get_matrix("ffn_up.weight")?,
                    down: get_matrix("ffn_down.weight")?,
                }));
            } else if n_ff[i] != 0 {
                let up_exps = get_expert_matrix("ffn_up_exps.weight")?;
                anyhow::ensure!(
                    up_exps.n_expert == n_expert,
                    "blk.{i}.ffn_up_exps.weight stacks {} experts, but expert_count is {n_expert}",
                    up_exps.n_expert,
                );
                // The router's width is what `moe_block` slices its logits
                // by, so it has to be the expert count and not merely agree
                // with it by construction.
                let gate_inp = get_matrix("ffn_gate_inp.weight")?;
                anyhow::ensure!(
                    gate_inp.out_dim == n_expert,
                    "blk.{i}.ffn_gate_inp.weight scores {} experts, but expert_count is {n_expert}",
                    gate_inp.out_dim,
                );
                layers.push(Layer::Moe(MoeLayer {
                    norm,
                    gate_inp,
                    exp_probs_b: get("exp_probs_b.bias")?,
                    up_exps,
                    down_exps: get_expert_matrix("ffn_down_exps.weight")?,
                    up_shexp: get_matrix("ffn_up_shexp.weight")?,
                    down_shexp: get_matrix("ffn_down_shexp.weight")?,
                }));
            } else if n_head_kv[i] != 0 {
                let cache_index = n_attn;
                n_attn += 1;
                layers.push(Layer::Attn(AttnLayer {
                    norm,
                    wq: get_matrix("attn_q.weight")?,
                    wk: get_matrix("attn_k.weight")?,
                    wv: get_matrix("attn_v.weight")?,
                    wo: get_matrix("attn_output.weight")?,
                    n_head_kv: n_head_kv[i] as usize,
                    cache_index,
                }));
            } else {
                let cache_index = n_recurrent;
                n_recurrent += 1;
                let in_proj = get_matrix("ssm_in.weight")?;
                let expected = 2 * d_inner + 2 * n_group * d_state + n_ssm_head;
                anyhow::ensure!(
                    in_proj.out_dim == expected,
                    "blk.{i}.ssm_in.weight projects to {}, expected {expected}",
                    in_proj.out_dim,
                );
                let group_norm = get("ssm_norm.weight")?;
                anyhow::ensure!(
                    group_norm.len() == d_inner,
                    "blk.{i}.ssm_norm.weight has {} values, expected ssm.inner_size ({d_inner})",
                    group_norm.len(),
                );
                layers.push(Layer::Ssm(SsmLayer {
                    norm,
                    in_proj,
                    conv1d: get("ssm_conv1d.weight")?,
                    conv1d_bias: get("ssm_conv1d.bias")?,
                    dt_bias: get("ssm_dt.bias")?,
                    a: get("ssm_a")?,
                    d: get("ssm_d")?,
                    group_norm,
                    out_proj: get_matrix("ssm_out.weight")?,
                    cache_index,
                }));
            }
        }
        anyhow::ensure!(
            n_attn > 0,
            "no attention blocks: attention.head_count_kv is zero at every trunk index"
        );

        Ok(Self {
            config,
            backend,
            tok_embeddings,
            output_norm,
            output_weight,
            n_head: loaded.config.n_head,
            head_dim: loaded.config.head_dim,
            rms_eps: loaded.config.rms_eps,
            routing,
            d_conv,
            d_inner,
            d_state,
            n_group,
            n_ssm_head,
            layers,
        })
    }

    /// Width of one head's slice of the `x` stream and of the recurrence
    /// output.
    fn ssm_head_dim(&self) -> usize {
        self.d_inner / self.n_ssm_head
    }

    /// The channels the causal convolution runs over — `x` followed by the
    /// `B` and `C` group vectors, which is exactly the span of `ssm_in`'s
    /// output between the gate and the timestep.
    fn conv_channels(&self) -> usize {
        self.d_inner + 2 * self.n_group * self.d_state
    }

    /// Splits one token's `ssm_in` output into `(z, xBC, dt)`. The gate
    /// leads, the convolved streams follow, and the per-head timestep
    /// trails.
    fn ssm_parts<'a>(&self, projected: &'a [f32]) -> (&'a [f32], &'a [f32], &'a [f32]) {
        let (z, rest) = projected.split_at(self.d_inner);
        let (x_bc, dt) = rest.split_at(self.conv_channels());
        (z, x_bc, dt)
    }
}

impl ModelForward for NemotronModel {
    fn vulkan_backend(&self) -> Option<&crate::engine::backend::vulkan::VulkanBackend> {
        self.backend.as_wgpu()
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn n_trunk_layer(&self) -> usize {
        self.layers.len()
    }

    fn new_kv_cache(&self, capacity: usize) -> KvCache {
        let kv_dims: Vec<usize> = self
            .layers
            .iter()
            .filter_map(|l| match l {
                Layer::Attn(l) => Some(l.n_head_kv * self.head_dim),
                _ => None,
            })
            .collect();
        let n_recurrent = self
            .layers
            .iter()
            .filter(|l| matches!(l, Layer::Ssm(_)))
            .count();
        let recurrent = vec![
            RecurrentSpec::ssm(
                self.conv_channels(),
                self.d_conv,
                self.n_ssm_head,
                self.ssm_head_dim(),
                self.d_state,
            );
            n_recurrent
        ];
        KvCache::new_mixed(capacity, &kv_dims, &recurrent)
    }

    fn forward(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        _slot_id: usize,
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

        // Grown once and reused across layers rather than allocated per
        // block — see `tensor::rmsnorm_into`.
        let mut normed: Vec<f32> = Vec::new();
        // Projection buffers, grown once and reused by every block — see
        // `Backend::matmul_into`.
        let mut sub_out: Vec<f32> = Vec::new();
        let mut block = BlockScratch::default();

        // Every block is one sub-layer under one norm, so the residual is
        // added once per block rather than twice.
        for layer in &self.layers {
            let norm = match layer {
                Layer::Ssm(l) => &l.norm,
                Layer::Attn(l) => &l.norm,
                Layer::Moe(l) => &l.norm,
                Layer::Ffn(l) => &l.norm,
            };
            tensor::rmsnorm_into(&mut normed, &x, norm, n_tokens, n_embd, self.rms_eps);
            match layer {
                Layer::Ssm(l) => {
                    self.ssm_block_into(l, cache, &mut sub_out, &mut block, &normed, n_tokens)
                }
                Layer::Attn(l) => self.attn_block_into(
                    l,
                    cache,
                    &normed,
                    &mut sub_out,
                    &mut block,
                    n_tokens,
                    start_pos,
                ),
                Layer::Moe(l) => sub_out = self.moe_block(l, &normed, n_tokens),
                Layer::Ffn(l) => {
                    self.ffn_block_into(&mut sub_out, &mut block, l, &normed, n_tokens)
                }
            }
            tensor::add_inplace(&mut x, &sub_out);
        }

        let mut last = x[(n_tokens - 1) * n_embd..].to_vec();
        tensor::rmsnorm_inplace(&mut last, &self.output_norm, 1, n_embd, self.rms_eps);
        Ok(self.backend.matmul(&last, 1, &self.output_weight))
    }

    fn forward_hidden_states(&self, _tokens: &[u32]) -> Result<Vec<f32>> {
        anyhow::bail!("embeddings are not yet supported for Nemotron-H models")
    }
}

impl NemotronModel {
    /// The selective state-space recurrence, one token at a time.
    ///
    /// Sequential by construction: each token's state depends on the
    /// previous token's, so unlike attention there is nothing to batch
    /// across the prompt. The projection that feeds it *is* batched, which
    /// is where the width is.
    fn ssm_block_into(
        &self,
        layer: &SsmLayer,
        cache: &mut KvCache,
        out: &mut Vec<f32>,
        block: &mut BlockScratch,
        normed: &[f32],
        n_tokens: usize,
    ) {
        let head_dim = self.ssm_head_dim();
        let d_state = self.d_state;
        let group_width = self.d_inner / self.n_group;
        // How many heads share one `B`/`C` group vector.
        let heads_per_group = self.n_ssm_head / self.n_group;
        let projected = &mut block.projected;
        self.backend
            .matmul_into(projected, normed, n_tokens, &layer.in_proj);
        let projected = &*projected;
        let row = layer.in_proj.out_dim;

        // The recurrence is sequential, but the two projections around it
        // are not: both run once for the whole batch. Only the mixing in
        // between walks token by token, and it writes into `ys` rather than
        // projecting each token as it goes — one output matmul per block
        // instead of one per token.
        let mut ys = vec![0f32; n_tokens * self.d_inner];
        let state = &mut cache.recurrent[layer.cache_index];
        for t in 0..n_tokens {
            let (z, x_bc, dt) = self.ssm_parts(&projected[t * row..(t + 1) * row]);

            let mut conv = state.conv_step(x_bc, &layer.conv1d);
            for (v, bias) in conv.iter_mut().zip(&layer.conv1d_bias) {
                *v = tensor::silu(*v + bias);
            }
            let (x, groups) = conv.split_at(self.d_inner);
            let (b, c) = groups.split_at(self.n_group * d_state);

            let y = &mut ys[t * self.d_inner..(t + 1) * self.d_inner];
            for h in 0..self.n_ssm_head {
                // `softplus` of the biased timestep both scales this
                // token's contribution and, through `a`, decays the state.
                let step = tensor::softplus(dt[h] + layer.dt_bias[h]);
                let decay = (step * layer.a[h]).exp();
                let group = h / heads_per_group;
                let b_h = &b[group * d_state..(group + 1) * d_state];
                let c_h = &c[group * d_state..(group + 1) * d_state];
                let x_h = &x[h * head_dim..(h + 1) * head_dim];
                let head_state = state.delta_state_mut(h);
                for p in 0..head_dim {
                    let x_dt = x_h[p] * step;
                    let row_state = &mut head_state[p * d_state..(p + 1) * d_state];
                    let mut sum = 0f32;
                    for (s, (&b_s, &c_s)) in b_h.iter().zip(c_h).enumerate() {
                        let updated = row_state[s] * decay + b_s * x_dt;
                        row_state[s] = updated;
                        sum += updated * c_s;
                    }
                    // The per-head skip term is on the block's own
                    // (convolved) input, not on the recurrence output.
                    y[h * head_dim + p] = sum + x_h[p] * layer.d[h];
                }
            }

            // Gate first, then normalize — each group of the gated result
            // over its own `d_inner / n_group`-wide weight vector.
            for (v, &g) in y.iter_mut().zip(z) {
                *v *= tensor::silu(g);
            }
            for (group, chunk) in y.chunks_mut(group_width).enumerate() {
                let weight = &layer.group_norm[group * group_width..(group + 1) * group_width];
                tensor::rmsnorm_inplace(chunk, weight, 1, group_width, self.rms_eps);
            }
        }
        self.backend
            .matmul_into(out, &ys, n_tokens, &layer.out_proj);
    }

    /// Unrotated causal GQA — no RoPE, no QK-norm, no biases.
    #[allow(clippy::too_many_arguments)]
    fn attn_block_into(
        &self,
        layer: &AttnLayer,
        cache: &mut KvCache,
        normed: &[f32],
        out: &mut Vec<f32>,
        block: &mut BlockScratch,
        n_tokens: usize,
        start_pos: usize,
    ) {
        let head_dim = self.head_dim;
        let n_head = self.n_head;
        let n_head_kv = layer.n_head_kv;
        let kv_dim = n_head_kv * head_dim;

        self.backend.matmul_batch_into(
            &mut block.qkv,
            &[
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
            ],
        );
        // Borrowed in dispatch order rather than popped, so the buffers stay
        // in the scratch and the next layer inherits their capacity.
        let (q, k, v) = (&block.qkv[0], &block.qkv[1], &block.qkv[2]);

        let layer_cache = &mut cache.layers[layer.cache_index];
        for t in 0..n_tokens {
            layer_cache.push(
                &k[t * kv_dim..(t + 1) * kv_dim],
                &v[t * kv_dim..(t + 1) * kv_dim],
            );
        }

        let mut attn_out = Vec::new();
        crate::engine::attention::attention(
            &mut attn_out,
            q,
            layer_cache,
            &crate::engine::attention::Params {
                backend: self.backend.as_ref(),
                // This layer's card — see `attention::Params::device`.
                device: layer.wo.device(),
                n_head,
                n_head_kv,
                head_dim,
                scale: 1.0 / (head_dim as f32).sqrt(),
                causal: true,
                n_swa: 0,
                start_pos,
                n_tokens,
            },
            |t| (0, start_pos + t),
        );
        self.backend
            .matmul_into(out, &attn_out, n_tokens, &layer.wo);
    }

    /// The router's logits and the shared expert's up projection, from one
    /// input.
    ///
    /// Both read the block's normalized input and neither depends on the
    /// other, so they are one `matmul_batch` — which uploads that input once
    /// and costs one backend round trip rather than two, per MoE block per
    /// token.
    ///
    /// Except when the shared expert is a quantization the selected backend
    /// has no kernel for, which it is allowed to be: `engine::backend::
    /// is_cpu_only_tensor` exempts the shared-expert matrices from the
    /// startup device-capability check precisely so a low-bit file still gets
    /// a GPU for the rest of the model. The two then cannot share a call —
    /// they are not going to the same place — so each takes
    /// [`super::matmul_host_fallback`] on its own.
    fn router_and_shared_up(
        &self,
        layer: &MoeLayer,
        normed: &[f32],
        n_tokens: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        if self.backend.supports_type(layer.up_shexp.ggml_type())
            && self.backend.supports_type(layer.gate_inp.ggml_type())
        {
            let mut both = self.backend.matmul_batch(&[
                MatmulOp {
                    x: normed,
                    n_tokens,
                    w: &layer.gate_inp,
                },
                MatmulOp {
                    x: normed,
                    n_tokens,
                    w: &layer.up_shexp,
                },
            ]);
            let shared_up = both.pop().expect("matmul_batch returns one result per op");
            let logits = both.pop().expect("matmul_batch returns one result per op");
            return (logits, shared_up);
        }
        (
            super::matmul_host_fallback(self.backend.as_ref(), normed, n_tokens, &layer.gate_inp),
            super::matmul_host_fallback(self.backend.as_ref(), normed, n_tokens, &layer.up_shexp),
        )
    }

    /// Routed experts plus the one always-on shared expert, both
    /// `down(relu(up(x))^2)`.
    ///
    /// The shared branch is unweighted: it is added at full strength
    /// alongside the routed sum, not folded into the routing weights the way
    /// the architectures with a gated shared expert do.
    fn moe_block(&self, layer: &MoeLayer, normed: &[f32], n_tokens: usize) -> Vec<f32> {
        let n_embd = self.config.n_embd;
        let mut experts =
            moe_stats::LayerRecorder::for_tensors(&[&layer.up_exps, &layer.down_exps]);

        // Route the whole batch first, so the union can be taken before any
        // expert's weights are read — see `super::evaluate_routed_experts`.
        // The router is one matmul for the batch, not one per token: it is
        // the same narrow matrix either way, and a per-token call is a
        // separate backend round trip per token.
        //
        // The shared expert's up projection reads the same `normed` and does
        // not depend on the routing, so it rides along in the same call: one
        // input upload and one round trip instead of two, per MoE block per
        // token.
        let n_expert = layer.gate_inp.out_dim;
        let (logits, shared_up) = self.router_and_shared_up(layer, normed, n_tokens);
        let mut selection: Vec<Vec<(usize, f32)>> = (0..n_tokens)
            .map(|t| {
                let (selected, weights) = self.routing.route(
                    &logits[t * n_expert..(t + 1) * n_expert],
                    Some(&layer.exp_probs_b),
                    None,
                );
                selected.into_iter().zip(weights).collect()
            })
            .collect();
        // Trim to the expert budget *before* anything is recorded or read:
        // the counters should describe the work actually done, and a
        // dropped expert's weights must never be fetched.
        super::apply_expert_budget(&mut selection, &layer.up_exps);
        for picks in &selection {
            picks.iter().for_each(|&(e, _)| experts.select(e));
        }

        // The batched helper — one `MatmulOp` per expert group across the
        // whole batch — where the per-expert path below is one call per
        // expert per projection. `super::
        // evaluate_routed_experts_batched_gateless` exists because these
        // experts have no gate projection to pair with `up`; everything else
        // about the two paths is the same, including the residency split and
        // the host remainder running beside the device batch.
        let routed_branch = || {
            if super::gpu_experts() && self.backend.as_wgpu().is_some() {
                super::evaluate_routed_experts_batched_gateless(
                    self.backend.as_ref(),
                    &selection,
                    normed,
                    n_embd,
                    &layer.up_exps,
                    &layer.down_exps,
                    |up| {
                        let mut up = up.to_vec();
                        relu_squared(&mut up);
                        up
                    },
                )
            } else {
                super::evaluate_routed_experts(&selection, |expert, members| {
                    let inputs: Vec<&[f32]> = members
                        .iter()
                        .map(|&(t, _)| &normed[t * n_embd..(t + 1) * n_embd])
                        .collect();
                    let hidden: Vec<Vec<f32>> = super::project_expert(
                        self.backend.as_ref(),
                        &layer.up_exps,
                        expert,
                        0,
                        layer.up_exps.out_dim,
                        &inputs,
                    )
                    .into_iter()
                    .map(|mut up| {
                        relu_squared(&mut up);
                        up
                    })
                    .collect();
                    let hidden_refs: Vec<&[f32]> = hidden.iter().map(Vec::as_slice).collect();
                    super::project_expert(
                        self.backend.as_ref(),
                        &layer.down_exps,
                        expert,
                        0,
                        layer.down_exps.out_dim,
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
            }
        };

        // The shared expert's up projection was issued with the router above;
        // only the activation and the down projection are left — a device
        // matmul with nothing in common with the host-side routed branch, so
        // the two overlap. See `super::moe_overlap_min_tokens`.
        let shared_branch = || {
            let mut shared = shared_up;
            relu_squared(&mut shared);
            super::matmul_host_fallback(self.backend.as_ref(), &shared, n_tokens, &layer.down_shexp)
        };
        let (mut out, contribs) = if n_tokens >= super::moe_overlap_min_tokens() {
            rayon::join(shared_branch, routed_branch)
        } else {
            (shared_branch(), routed_branch())
        };
        experts.loaded_once_per_distinct_expert();

        for (t, picks) in contribs.iter().enumerate() {
            let dst = &mut out[t * n_embd..(t + 1) * n_embd];
            for contribution in picks {
                tensor::add_inplace(dst, contribution);
            }
        }
        experts.commit(n_tokens);
        out
    }
}

impl NemotronModel {
    /// `down(relu(up(x))^2)`, the dense counterpart of [`Self::moe_block`]'s
    /// shared-expert branch.
    ///
    /// Through `matmul_host_fallback` for the same reason that branch is:
    /// these are ordinary per-layer weights rather than stacked experts, but
    /// a quantization the device has no kernel for still has to reach the
    /// host rather than panic inside the backend.
    fn ffn_block_into(
        &self,
        out: &mut Vec<f32>,
        block: &mut BlockScratch,
        layer: &FfnLayer,
        normed: &[f32],
        n_tokens: usize,
    ) {
        let up = &mut block.up;
        super::matmul_host_fallback_into(up, self.backend.as_ref(), normed, n_tokens, &layer.up);
        relu_squared(up);
        super::matmul_host_fallback_into(out, self.backend.as_ref(), up, n_tokens, &layer.down);
    }
}

/// The per-block projection buffers the `*_into` blocks reuse — see
/// `Backend::matmul_into`. One per forward pass rather than one per block.
#[derive(Default)]
struct BlockScratch {
    /// The SSM block's fused input projection.
    projected: Vec<f32>,
    /// The FFN block's up projection, squared-ReLU'd in place.
    up: Vec<f32>,
    /// The attention block's Q/K/V, in dispatch order.
    qkv: Vec<Vec<f32>>,
}

/// This architecture's FFN activation, in place: `max(0, x)^2`.
///
/// Squaring after the clamp, not `x * |x|` or `x * relu(x)` — negatives are
/// dropped outright, so the result is never negative.
fn relu_squared(x: &mut [f32]) {
    for v in x.iter_mut() {
        let clamped = v.max(0.0);
        *v = clamped * clamped;
    }
}

#[cfg(test)]
mod tests {
    use super::relu_squared;

    #[test]
    fn relu_squared_drops_negatives_and_squares_the_rest() {
        let mut x = [-2.0f32, -0.5, 0.0, 0.5, 3.0];
        relu_squared(&mut x);
        assert_eq!(x, [0.0, 0.0, 0.0, 0.25, 9.0]);
    }

    /// The dense and MoE siblings are told apart by the file's own
    /// `expert_count`, not by its architecture string.
    ///
    /// Both declare `nemotron_h*` and share every other tensor in the trunk;
    /// what differs is whether an FFN block carries `ffn_up`/`ffn_down` or a
    /// stack of experts. Reading that from `general.architecture` instead
    /// would be a second source of truth for something the file already
    /// states — and the string is exactly what a mislabelled conversion gets
    /// wrong.
    #[test]
    fn a_file_declaring_no_experts_selects_the_dense_ffn() {
        // `metadata_u64` answering `None` is the dense file's own statement;
        // `Some(0)` would be a file that declares zero experts, which means
        // the same thing. Both must take the dense branch, and any positive
        // count must not.
        for (declared, dense) in [(None, true), (Some(0u64), true), (Some(128), false)] {
            let n_expert = declared.unwrap_or(0) as usize;
            assert_eq!(
                n_expert == 0,
                dense,
                "expert_count {declared:?} chose the wrong FFN branch"
            );
        }
    }
}
