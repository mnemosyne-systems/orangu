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

//! Qwen3.5/3.6-MoE (`general.architecture = "qwen35moe"`), confirmed
//! against real upstream `llama.cpp` source (`src/models/qwen35moe.cpp`,
//! `src/models/delta-net-base.cpp`, `src/llama-graph.cpp`'s
//! `build_moe_ffn`, and the relevant `ggml-cpu/ops.cpp` compute kernels —
//! fetched and read directly, not guessed) — a genuinely different shape
//! from both `engine::arch::llama` and `engine::arch::gemma`:
//!
//! - **Layers alternate between two kinds** (`full_attention_interval`,
//!   normally every 4th layer is full attention, the rest are linear
//!   attention): a standard pre-norm transformer block either way
//!   (`x += sub(rmsnorm(x)); x += moe_ffn(rmsnorm(x))`), only the
//!   attention sub-layer itself differs.
//! - **Full-attention layers**: a *joint* query+gate projection (`wq`'s
//!   output is `[Q_h, gate_h]` interleaved per head), Q/K-norm, partial
//!   rotary (`rope.dimension_count` is a quarter of `attention.key_length`
//!   here), standard GQA, then the attention output is gated by
//!   `sigmoid(gate)` before the output projection.
//! - **Linear-attention (gated-DeltaNet) layers**: a joint QKV projection
//!   through a causal depthwise conv1d + SiLU, per-head L2-normed Q/K, a
//!   scalar-per-head softplus-gated decay, and a delta-rule recurrent
//!   state update — implemented here only in its *autoregressive*
//!   (one-token-at-a-time) form, not the chunked/parallel form real
//!   llama.cpp also has: the two are mathematically identical (chunking is
//!   a prefill-throughput optimization, not different math — confirmed by
//!   reading `build_delta_net_chunking` and `build_delta_net_autoregressive`
//!   side by side), so this is a real, deliberate, documented scope
//!   reduction (slower prompt processing on long prompts, not a
//!   correctness gap), not a shortcut.
//! - **MoE FFN on every layer**: standard softmax top-k routing
//!   (renormalized) over routed experts, plus a separately-gated
//!   (`sigmoid`) shared expert whose output adds in.
//!
//! **Not implemented**: NextN/MTP (speculative-decoding-only extra decoder
//! blocks beyond `block_count`) — this module only ever reads `config.
//! n_layer` (`block_count`) layers; any MTP blocks in the file are simply
//! never touched. Multi-section RoPE ("IMRoPE") is implemented as plain
//! NEOX rope: for text-only input every rope "position channel" (t/h/w/e)
//! carries the same linear position, at which point the sections mechanism
//! (confirmed by reading `ggml_mrope_cache_init`) is a no-op — it only
//! matters for genuinely multi-axis (vision/video) position input, which
//! this engine doesn't accept.

use anyhow::{Context, Result, bail};
use std::sync::Arc;

use super::ModelForward;
use crate::engine::backend::{Backend, MatmulOp};
use crate::engine::kv_cache::KvCache;
use crate::engine::loader::{ExpertQuantMatrix, LoadedModel, ModelConfig, QuantMatrix};
use crate::engine::tensor;

/// Shared by both layer kinds: routed top-k softmax experts (renormalized)
/// plus one always-on, separately-gated shared expert.
struct MoeFfn {
    gate_inp: QuantMatrix,
    gate_exps: ExpertQuantMatrix,
    up_exps: ExpertQuantMatrix,
    down_exps: ExpertQuantMatrix,
    /// `[n_embd]` — a matmul weight with `out_dim == 1` in the reference
    /// graph (produces one shared-expert gate scalar per token); tiny, so
    /// eagerly resident and dot-producted directly rather than routed
    /// through `QuantMatrix`.
    gate_inp_shexp: Vec<f32>,
    gate_shexp: QuantMatrix,
    up_shexp: QuantMatrix,
    down_shexp: QuantMatrix,
}

struct FullAttnLayer {
    attn_norm: Vec<f32>,
    /// Joint query+gate projection: per head, `[Q(head_dim), gate(head_dim)]`
    /// interleaved — `out_dim == 2 * n_head * head_dim`.
    wq: QuantMatrix,
    attn_q_norm: Vec<f32>,
    wk: QuantMatrix,
    attn_k_norm: Vec<f32>,
    wv: QuantMatrix,
    wo: QuantMatrix,
    post_attention_norm: Vec<f32>,
    ffn: MoeFfn,
    /// Dense index into `KvCache::layers` (every full-attention layer has
    /// its own cache — no cross-layer sharing in this architecture).
    cache_index: usize,
}

struct RecurrentLayer {
    attn_norm: Vec<f32>,
    /// Joint Q/K/V mix: `[q(key_dim), k(key_dim), v(value_dim)]`.
    wqkv: QuantMatrix,
    wqkv_gate: QuantMatrix,
    /// `[conv_channels, d_conv]`, channel-major (ggml's own tensor order).
    ssm_conv1d: Vec<f32>,
    /// `[num_v_heads]` — added to the alpha projection before softplus.
    ssm_dt_bias: Vec<f32>,
    /// `[num_v_heads]` — per-head learned decay scale (typically negative;
    /// `exp(softplus(alpha + dt_bias) * ssm_a)` is the per-head decay).
    ssm_a: Vec<f32>,
    ssm_beta: QuantMatrix,
    ssm_alpha: QuantMatrix,
    /// `[head_v_dim]` — the gated output RMSNorm's learned weight.
    ssm_norm: Vec<f32>,
    ssm_out: QuantMatrix,
    post_attention_norm: Vec<f32>,
    ffn: MoeFfn,
    /// Dense index into `KvCache::recurrent`.
    cache_index: usize,
}

enum Layer {
    FullAttn(FullAttnLayer),
    Recurrent(RecurrentLayer),
}

pub struct Qwen35MoeModel {
    config: ModelConfig,
    backend: Arc<dyn Backend>,
    tok_embeddings: QuantMatrix,
    output_norm: Vec<f32>,
    output_weight: QuantMatrix,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    rope_dim: usize,
    rope_freq_base: f32,
    rms_eps: f32,
    n_expert_used: usize,
    /// SSM/gated-delta-net dimensions (`qwen35moe.ssm.*` metadata).
    ssm_d_conv: usize,
    /// `head_k_dim == head_v_dim` for gated-DeltaNet (required by the
    /// recurrence itself — see the module doc comment).
    ssm_head_dim: usize,
    /// Number of K/V "groups" the causal conv1d/Q/K live in
    /// (`ssm.group_count`) — smaller than `ssm_dt_rank` (the number of
    /// value heads); a K/V group is reused (tiled, not block-grouped —
    /// confirmed against `ggml_compute_forward_repeat_f32`) across
    /// `ssm_dt_rank / ssm_n_group` value heads.
    ssm_n_group: usize,
    ssm_dt_rank: usize,
    layers: Vec<Layer>,
}

impl Qwen35MoeModel {
    pub fn load_with_backend(loaded: &LoadedModel, backend: Arc<dyn Backend>) -> Result<Self> {
        let config = loaded.config.clone();
        let n_layer = config.n_layer;

        loaded
            .metadata_u64("expert_count")
            .context("missing expert_count")?;
        let n_expert_used = loaded
            .metadata_u64("expert_used_count")
            .context("missing expert_used_count")? as usize;
        let head_dim = loaded
            .metadata_u64("attention.key_length")
            .context("missing attention.key_length")? as usize;

        let ssm_d_conv = loaded
            .metadata_u64("ssm.conv_kernel")
            .context("missing ssm.conv_kernel")? as usize;
        let ssm_head_dim = loaded
            .metadata_u64("ssm.state_size")
            .context("missing ssm.state_size")? as usize;
        let ssm_n_group = loaded
            .metadata_u64("ssm.group_count")
            .context("missing ssm.group_count")? as usize;
        let ssm_dt_rank = loaded
            .metadata_u64("ssm.time_step_rank")
            .context("missing ssm.time_step_rank")? as usize;

        let full_attention_interval =
            loaded.metadata_u64("full_attention_interval").unwrap_or(4) as usize;
        let is_recr: Vec<bool> = loaded
            .metadata_array_u64("attention.recurrent_layers")
            .map(|arr| arr.iter().map(|&v| v != 0).collect())
            .unwrap_or_else(|| {
                (0..n_layer)
                    .map(|i| (i + 1) % full_attention_interval != 0)
                    .collect()
            });

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
        let mut n_full_attn = 0usize;
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

            let ffn = MoeFfn {
                gate_inp: get_matrix("ffn_gate_inp.weight")?,
                gate_exps: get_expert_matrix("ffn_gate_exps.weight")?,
                up_exps: get_expert_matrix("ffn_up_exps.weight")?,
                down_exps: get_expert_matrix("ffn_down_exps.weight")?,
                gate_inp_shexp: get("ffn_gate_inp_shexp.weight")?,
                gate_shexp: get_matrix("ffn_gate_shexp.weight")?,
                up_shexp: get_matrix("ffn_up_shexp.weight")?,
                down_shexp: get_matrix("ffn_down_shexp.weight")?,
            };

            if is_recr.get(i).copied().unwrap_or(false) {
                let cache_index = n_recurrent;
                n_recurrent += 1;
                layers.push(Layer::Recurrent(RecurrentLayer {
                    attn_norm: get("attn_norm.weight")?,
                    wqkv: get_matrix("attn_qkv.weight")?,
                    wqkv_gate: get_matrix("attn_gate.weight")?,
                    ssm_conv1d: get("ssm_conv1d.weight")?,
                    ssm_dt_bias: get("ssm_dt.bias")?,
                    ssm_a: get("ssm_a")?,
                    ssm_beta: get_matrix("ssm_beta.weight")?,
                    ssm_alpha: get_matrix("ssm_alpha.weight")?,
                    ssm_norm: get("ssm_norm.weight")?,
                    ssm_out: get_matrix("ssm_out.weight")?,
                    post_attention_norm: get("post_attention_norm.weight")?,
                    ffn,
                    cache_index,
                }));
            } else {
                let cache_index = n_full_attn;
                n_full_attn += 1;
                layers.push(Layer::FullAttn(FullAttnLayer {
                    attn_norm: get("attn_norm.weight")?,
                    wq: get_matrix("attn_q.weight")?,
                    attn_q_norm: get("attn_q_norm.weight")?,
                    wk: get_matrix("attn_k.weight")?,
                    attn_k_norm: get("attn_k_norm.weight")?,
                    wv: get_matrix("attn_v.weight")?,
                    wo: get_matrix("attn_output.weight")?,
                    post_attention_norm: get("post_attention_norm.weight")?,
                    ffn,
                    cache_index,
                }));
            }

            if loaded.has_tensor(&format!("blk.{i}.nextn.eh_proj.weight")) {
                bail!(
                    "blk.{i} has NextN/MTP tensors — speculative-decoding blocks are not yet supported by orangu-server; see doc/SERVER_ROADMAP.md"
                );
            }
        }

        Ok(Self {
            config,
            backend,
            tok_embeddings,
            output_norm,
            output_weight,
            n_head: 0, // set below, only meaningful when there's a full-attn layer
            n_head_kv: 0,
            head_dim,
            rope_dim: 0,
            rope_freq_base: 0.0,
            rms_eps: 0.0,
            n_expert_used,
            ssm_d_conv,
            ssm_head_dim,
            ssm_n_group,
            ssm_dt_rank,
            layers,
        }
        .with_shared_hparams(loaded))
    }

    fn with_shared_hparams(mut self, loaded: &LoadedModel) -> Self {
        self.n_head = loaded.config.n_head;
        self.n_head_kv = loaded.config.n_head_kv;
        self.rope_dim = loaded.config.rope_dim;
        self.rope_freq_base = loaded.config.rope_freq_base;
        self.rms_eps = loaded.config.rms_eps;
        self
    }

    /// `(n_full_attn, n_recurrent)` layer counts, and each layer's
    /// `(conv_channels, key_dim, value_dim)` for the recurrent ones — used
    /// to size a fresh [`KvCache`].
    fn cache_layout(&self) -> (usize, usize) {
        let n_full_attn = self
            .layers
            .iter()
            .filter(|l| matches!(l, Layer::FullAttn(_)))
            .count();
        let n_recurrent = self.layers.len() - n_full_attn;
        (n_full_attn, n_recurrent)
    }

    fn key_dim(&self) -> usize {
        self.ssm_head_dim * self.ssm_n_group
    }

    fn value_dim(&self) -> usize {
        self.ssm_head_dim * self.ssm_dt_rank
    }

    fn conv_channels(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }
}

impl ModelForward for Qwen35MoeModel {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn new_kv_cache(&self, capacity: usize) -> KvCache {
        let (n_full_attn, n_recurrent) = self.cache_layout();
        let kv_dims = vec![self.n_head_kv * self.head_dim; n_full_attn];
        let recurrent_specs = vec![
            (
                self.conv_channels(),
                self.ssm_d_conv,
                self.ssm_dt_rank,
                self.ssm_head_dim,
            );
            n_recurrent
        ];
        KvCache::new_mixed(capacity, &kv_dims, &recurrent_specs)
    }

    fn forward(&self, cache: &mut KvCache, tokens: &[u32], start_pos: usize) -> Result<Vec<f32>> {
        let n_tokens = tokens.len();
        let n_embd = self.config.n_embd;
        let eps = self.rms_eps;

        let mut x = vec![0f32; n_tokens * n_embd];
        for (t, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            anyhow::ensure!(
                tok < self.config.n_vocab,
                "token id {tok} is out of vocab range"
            );
            x[t * n_embd..(t + 1) * n_embd].copy_from_slice(&self.tok_embeddings.row(tok));
        }

        for layer in &self.layers {
            match layer {
                Layer::FullAttn(layer) => {
                    self.forward_full_attn_layer(layer, cache, &mut x, n_tokens, start_pos)?;
                }
                Layer::Recurrent(layer) => {
                    self.forward_recurrent_layer(layer, cache, &mut x, n_tokens)?;
                }
            }
        }

        let last = &mut x[(n_tokens - 1) * n_embd..].to_vec();
        tensor::rmsnorm_inplace(last, &self.output_norm, 1, n_embd, eps);
        let logits = self.backend.matmul(last, 1, &self.output_weight);
        Ok(logits)
    }

    fn forward_hidden_states(&self, _tokens: &[u32]) -> Result<Vec<f32>> {
        anyhow::bail!("embeddings are not yet supported for Qwen3.5-MoE models")
    }
}

impl Qwen35MoeModel {
    fn forward_full_attn_layer(
        &self,
        layer: &FullAttnLayer,
        cache: &mut KvCache,
        x: &mut [f32],
        n_tokens: usize,
        start_pos: usize,
    ) -> Result<()> {
        let n_embd = self.config.n_embd;
        let eps = self.rms_eps;
        let head_dim = self.head_dim;
        let n_head = self.n_head;
        let n_head_kv = self.n_head_kv;
        let kv_dim = n_head_kv * head_dim;
        let group_size = n_head / n_head_kv;

        let mut normed = x.to_vec();
        tensor::rmsnorm_inplace(&mut normed, &layer.attn_norm, n_tokens, n_embd, eps);

        // Joint Q+gate projection, K, and V are all independent projections
        // of the same normed input — one batched dispatch instead of three
        // sequential round-trips (see `Backend::matmul_batch`). Per head,
        // the Q+gate projection is [Q(head_dim), gate(head_dim)].
        let mut qgkv = self.backend.matmul_batch(&[
            MatmulOp {
                x: &normed,
                n_tokens,
                w: &layer.wq,
            },
            MatmulOp {
                x: &normed,
                n_tokens,
                w: &layer.wk,
            },
            MatmulOp {
                x: &normed,
                n_tokens,
                w: &layer.wv,
            },
        ]);
        let v = qgkv.pop().unwrap();
        let mut k = qgkv.pop().unwrap();
        let qg = qgkv.pop().unwrap();
        let mut q = vec![0f32; n_tokens * n_head * head_dim];
        let mut gate = vec![0f32; n_tokens * n_head * head_dim];
        for t in 0..n_tokens {
            for h in 0..n_head {
                let src = &qg[t * n_head * 2 * head_dim + h * 2 * head_dim..];
                q[t * n_head * head_dim + h * head_dim..t * n_head * head_dim + (h + 1) * head_dim]
                    .copy_from_slice(&src[0..head_dim]);
                gate[t * n_head * head_dim + h * head_dim
                    ..t * n_head * head_dim + (h + 1) * head_dim]
                    .copy_from_slice(&src[head_dim..2 * head_dim]);
            }
        }
        tensor::rmsnorm_inplace(&mut q, &layer.attn_q_norm, n_tokens * n_head, head_dim, eps);
        for t in 0..n_tokens {
            let pos = start_pos + t;
            tensor::rope_apply_inplace(
                &mut q[t * n_head * head_dim..(t + 1) * n_head * head_dim],
                n_head,
                head_dim,
                self.rope_dim,
                pos,
                self.rope_freq_base,
            );
        }

        tensor::rmsnorm_inplace(
            &mut k,
            &layer.attn_k_norm,
            n_tokens * n_head_kv,
            head_dim,
            eps,
        );

        let layer_cache = &mut cache.layers[layer.cache_index];
        for t in 0..n_tokens {
            let pos = start_pos + t;
            tensor::rope_apply_inplace(
                &mut k[t * kv_dim..(t + 1) * kv_dim],
                n_head_kv,
                head_dim,
                self.rope_dim,
                pos,
                self.rope_freq_base,
            );
            layer_cache.push(
                &k[t * kv_dim..(t + 1) * kv_dim],
                &v[t * kv_dim..(t + 1) * kv_dim],
            );
        }

        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut attn_out = vec![0f32; n_tokens * n_head * head_dim];
        for t in 0..n_tokens {
            let pos = start_pos + t;
            for h in 0..n_head {
                let kv_head = h / group_size;
                let qh = &q[t * n_head * head_dim + h * head_dim
                    ..t * n_head * head_dim + (h + 1) * head_dim];
                let mut scores = Vec::with_capacity(pos + 1);
                for p in 0..=pos {
                    let kh = layer_cache.key_at(p, kv_head, head_dim);
                    scores.push(tensor::dot(qh, kh) * scale);
                }
                tensor::softmax_inplace(&mut scores);
                let out = &mut attn_out[t * n_head * head_dim + h * head_dim
                    ..t * n_head * head_dim + (h + 1) * head_dim];
                for (p, &weight) in scores.iter().enumerate() {
                    let vh = layer_cache.value_at(p, kv_head, head_dim);
                    for (o, vi) in out.iter_mut().zip(vh.iter()) {
                        *o += weight * vi;
                    }
                }
            }
        }
        // Gate the attention output (sigmoid), then project.
        for (o, &g) in attn_out.iter_mut().zip(gate.iter()) {
            *o *= tensor::sigmoid(g);
        }
        let sub_out = self.backend.matmul(&attn_out, n_tokens, &layer.wo);

        tensor::add_inplace(x, &sub_out);
        let mut normed2 = x.to_vec();
        tensor::rmsnorm_inplace(
            &mut normed2,
            &layer.post_attention_norm,
            n_tokens,
            n_embd,
            eps,
        );
        let ffn_out = self.moe_ffn(&layer.ffn, &normed2, n_tokens);
        tensor::add_inplace(x, &ffn_out);
        Ok(())
    }

    fn forward_recurrent_layer(
        &self,
        layer: &RecurrentLayer,
        cache: &mut KvCache,
        x: &mut [f32],
        n_tokens: usize,
    ) -> Result<()> {
        let n_embd = self.config.n_embd;
        let eps = self.rms_eps;
        let key_dim = self.key_dim();
        let value_dim = self.value_dim();
        let head_dim = self.ssm_head_dim;
        let n_k_heads = self.ssm_n_group;
        let n_v_heads = self.ssm_dt_rank;
        let q_scale = 1.0 / (head_dim as f32).sqrt();

        let mut normed = x.to_vec();
        tensor::rmsnorm_inplace(&mut normed, &layer.attn_norm, n_tokens, n_embd, eps);

        // All four are independent projections of the same normed input —
        // one batched dispatch instead of four sequential round-trips (see
        // `Backend::matmul_batch`).
        let mut mixed_z_beta_alpha = self.backend.matmul_batch(&[
            MatmulOp {
                x: &normed,
                n_tokens,
                w: &layer.wqkv,
            },
            MatmulOp {
                x: &normed,
                n_tokens,
                w: &layer.wqkv_gate,
            },
            MatmulOp {
                x: &normed,
                n_tokens,
                w: &layer.ssm_beta,
            },
            MatmulOp {
                x: &normed,
                n_tokens,
                w: &layer.ssm_alpha,
            },
        ]);
        let alpha = mixed_z_beta_alpha.pop().unwrap();
        let mut beta = mixed_z_beta_alpha.pop().unwrap();
        let z = mixed_z_beta_alpha.pop().unwrap();
        let qkv_mixed = mixed_z_beta_alpha.pop().unwrap();
        for b in beta.iter_mut() {
            *b = tensor::sigmoid(*b);
        }
        let mut decay = vec![0f32; n_tokens * n_v_heads];
        for t in 0..n_tokens {
            for h in 0..n_v_heads {
                let a = alpha[t * n_v_heads + h] + layer.ssm_dt_bias[h];
                let log_decay = tensor::softplus(a) * layer.ssm_a[h];
                decay[t * n_v_heads + h] = log_decay.exp();
            }
        }

        let mut sub_out = vec![0f32; n_tokens * n_embd];
        let ssm_state = &mut cache.recurrent[layer.cache_index];
        for t in 0..n_tokens {
            let mixed =
                &qkv_mixed[t * (2 * key_dim + value_dim)..(t + 1) * (2 * key_dim + value_dim)];
            let mut conv_out = ssm_state.conv_step(mixed, &layer.ssm_conv1d);
            for v in conv_out.iter_mut() {
                *v = tensor::silu(*v);
            }
            let (q_conv, rest) = conv_out.split_at_mut(key_dim);
            let (k_conv, v_conv) = rest.split_at_mut(key_dim);
            debug_assert_eq!(v_conv.len(), value_dim);

            for h in 0..n_k_heads {
                tensor::l2_norm_inplace(&mut q_conv[h * head_dim..(h + 1) * head_dim], eps);
                tensor::l2_norm_inplace(&mut k_conv[h * head_dim..(h + 1) * head_dim], eps);
            }
            for v in q_conv.iter_mut() {
                *v *= q_scale;
            }

            let mut attn_out = vec![0f32; value_dim];
            for vh in 0..n_v_heads {
                // Tiled (not block-grouped) broadcast — matches
                // `ggml_compute_forward_repeat_f32`'s tiling semantics for
                // this specific mismatched-head-count repeat, distinct from
                // standard attention's block-grouped GQA.
                let kh = vh % n_k_heads;
                let qh = &q_conv[kh * head_dim..(kh + 1) * head_dim];
                let khv = &k_conv[kh * head_dim..(kh + 1) * head_dim];
                let vhv = &v_conv[vh * head_dim..(vh + 1) * head_dim];
                let beta_h = beta[t * n_v_heads + vh];
                let decay_h = decay[t * n_v_heads + vh];

                let state = ssm_state.delta_state_mut(vh);
                for s in state.iter_mut() {
                    *s *= decay_h;
                }
                // sk[a] = sum_b k[b] * S[b][a]  (k^T S)
                let mut sk = vec![0f32; head_dim];
                for a in 0..head_dim {
                    let mut sum = 0f32;
                    for b in 0..head_dim {
                        sum += khv[b] * state[b * head_dim + a];
                    }
                    sk[a] = sum;
                }
                let d: Vec<f32> = (0..head_dim).map(|a| beta_h * (vhv[a] - sk[a])).collect();
                for i in 0..head_dim {
                    for j in 0..head_dim {
                        state[i * head_dim + j] += khv[i] * d[j];
                    }
                }
                // o[j] = sum_i q[i] * S_new[i][j]  (q^T S_new)
                let out = &mut attn_out[vh * head_dim..(vh + 1) * head_dim];
                for j in 0..head_dim {
                    let mut sum = 0f32;
                    for i in 0..head_dim {
                        sum += qh[i] * state[i * head_dim + j];
                    }
                    out[j] = sum;
                }
            }

            // Gated RMSNorm, per head: rmsnorm(attn_out_h) * silu(z_h).
            for h in 0..n_v_heads {
                let mut normed_h = attn_out[h * head_dim..(h + 1) * head_dim].to_vec();
                tensor::rmsnorm_inplace(&mut normed_h, &layer.ssm_norm, 1, head_dim, eps);
                let z_h = &z[t * value_dim + h * head_dim..t * value_dim + (h + 1) * head_dim];
                for (o, (n, zv)) in attn_out[h * head_dim..(h + 1) * head_dim]
                    .iter_mut()
                    .zip(normed_h.iter().zip(z_h.iter()))
                {
                    *o = *n * tensor::silu(*zv);
                }
            }

            let projected = self.backend.matmul(&attn_out, 1, &layer.ssm_out);
            sub_out[t * n_embd..(t + 1) * n_embd].copy_from_slice(&projected);
        }

        tensor::add_inplace(x, &sub_out);
        let mut normed2 = x.to_vec();
        tensor::rmsnorm_inplace(
            &mut normed2,
            &layer.post_attention_norm,
            n_tokens,
            n_embd,
            eps,
        );
        let ffn_out = self.moe_ffn(&layer.ffn, &normed2, n_tokens);
        tensor::add_inplace(x, &ffn_out);
        Ok(())
    }

    /// Standard top-k softmax MoE routing (renormalized over the selected
    /// experts) plus a separately-`sigmoid`-gated shared expert — see
    /// `llm_graph_context::build_moe_ffn` (the `LLAMA_EXPERT_GATING_FUNC_
    /// TYPE_SOFTMAX`, `norm_w = true` path qwen35moe uses) and `build_
    /// layer_ffn`'s shared-expert gate.
    fn moe_ffn(&self, ffn: &MoeFfn, normed: &[f32], n_tokens: usize) -> Vec<f32> {
        let n_embd = self.config.n_embd;
        let mut out = vec![0f32; n_tokens * n_embd];
        for t in 0..n_tokens {
            let x_t = &normed[t * n_embd..(t + 1) * n_embd];
            let logits = self.backend.matmul(x_t, 1, &ffn.gate_inp);
            let mut probs = logits.clone();
            tensor::softmax_inplace(&mut probs);

            let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            indexed.truncate(self.n_expert_used);
            let weight_sum: f32 = indexed
                .iter()
                .map(|(_, w)| w)
                .sum::<f32>()
                .max(6.103_515_6e-5);

            let mut moe_out = vec![0f32; n_embd];
            for &(expert, weight) in &indexed {
                let weight = weight / weight_sum;
                let gate: Vec<f32> = (0..ffn.gate_exps.out_dim)
                    .map(|o| tensor::dot(x_t, &ffn.gate_exps.row(expert, o)))
                    .collect();
                let up: Vec<f32> = (0..ffn.up_exps.out_dim)
                    .map(|o| tensor::dot(x_t, &ffn.up_exps.row(expert, o)))
                    .collect();
                let mut h: Vec<f32> = gate.iter().map(|&g| tensor::silu(g)).collect();
                tensor::mul_inplace(&mut h, &up);
                let down: Vec<f32> = (0..ffn.down_exps.out_dim)
                    .map(|o| tensor::dot(&h, &ffn.down_exps.row(expert, o)))
                    .collect();
                for (o, d) in moe_out.iter_mut().zip(down.iter()) {
                    *o += weight * d;
                }
            }

            let shared_gate = tensor::sigmoid(tensor::dot(x_t, &ffn.gate_inp_shexp));
            let mut shexp_gate_up = self.backend.matmul_batch(&[
                MatmulOp {
                    x: x_t,
                    n_tokens: 1,
                    w: &ffn.gate_shexp,
                },
                MatmulOp {
                    x: x_t,
                    n_tokens: 1,
                    w: &ffn.up_shexp,
                },
            ]);
            let shexp_up = shexp_gate_up.pop().unwrap();
            let shexp_gate = shexp_gate_up.pop().unwrap();
            let mut shexp_h: Vec<f32> = shexp_gate.iter().map(|&g| tensor::silu(g)).collect();
            tensor::mul_inplace(&mut shexp_h, &shexp_up);
            let mut shexp_out = self.backend.matmul(&shexp_h, 1, &ffn.down_shexp);
            for v in shexp_out.iter_mut() {
                *v *= shared_gate;
            }

            let dst = &mut out[t * n_embd..(t + 1) * n_embd];
            for i in 0..n_embd {
                dst[i] = moe_out[i] + shexp_out[i];
            }
        }
        out
    }
}

#[cfg(test)]
mod real_model_tests {
    use super::*;

    /// Cross-check against real llama.cpp: given the token IDs real
    /// llama.cpp's `/tokenize` produces for "The capital of France is"
    /// (byte-level BPE — this model's `tokenizer.ggml.model = "gpt2"`,
    /// already correctly supported, unlike gemma4's SentencePiece gap), the
    /// model should predict " Paris" (token 11751) as the top next token,
    /// matching real llama.cpp's `/completion` (`n_probs`) output exactly.
    /// Run with `ORANGU_TEST_MODEL=/path/to.gguf cargo test --release --bin
    /// orangu-server qwen35moe::real_model_tests -- --ignored` (a 35B-param
    /// model — expect several minutes: this engine's scalar per-row dequant
    /// has no hand-tuned SIMD quantized-matmul kernel, see
    /// `doc/SERVER_ROADMAP.md`).
    #[test]
    #[ignore]
    fn qwen35moe_predicts_paris_after_capital_of_france() {
        let path = std::env::var("ORANGU_TEST_MODEL").expect("set ORANGU_TEST_MODEL");
        let loaded = LoadedModel::open(std::path::Path::new(&path)).expect("load model");
        let model = Qwen35MoeModel::load_with_backend(
            &loaded,
            Arc::new(crate::engine::backend::CpuBackend),
        )
        .expect("build model");

        let mut cache = model.new_kv_cache(64);
        let tokens: Vec<u32> = vec![760, 6511, 314, 9338, 369];
        let logits = model.forward(&mut cache, &tokens, 0).expect("forward");
        let (top_id, _) = logits
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        assert_eq!(top_id, 11751, "expected ' Paris' (11751) as top prediction");
    }
}
