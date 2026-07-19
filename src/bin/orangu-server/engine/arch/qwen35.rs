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

//! Qwen3.5 dense (`general.architecture = "qwen35"`), confirmed against
//! real upstream `llama.cpp` source (`src/models/qwen35.cpp`, read
//! directly, not guessed) — the non-MoE sibling of
//! `engine::arch::qwen35moe`. Every layer shape (hybrid full-attention /
//! gated-DeltaNet linear-attention alternation, the joint query+gate
//! projection, partial rotary, the delta-rule recurrent state update) is
//! *identical* to `qwen35moe` (`src/models/qwen35.cpp` and
//! `src/models/qwen35moe.cpp` share the same `llm_build_delta_net_base`
//! attention-layer code — only `build_layer_ffn` differs): this file only
//! exists because the FFN is plain dense SwiGLU (`ffn_gate`/`ffn_up`/
//! `ffn_down`, `LLM_FFN_SILU`/`LLM_FFN_PAR`, matching `engine::arch::
//! llama`'s own FFN exactly) instead of routed+shared-expert MoE, and a
//! GGUF's tensor names/shapes tell the loader which one to expect up
//! front rather than at graph-build time. See `qwen35moe`'s own module
//! doc comment for what's deliberately *not* implemented here too
//! (autoregressive-only gated-DeltaNet, no chunked/parallel prefill path;
//! plain NEOX rope in place of multi-section RoPE, provably identical for
//! text-only input; no NextN/MTP) — every one of those reasons applies
//! here unchanged, since the code they describe is shared.

use anyhow::{Context, Result, bail};
use std::sync::Arc;

use super::ModelForward;
use crate::engine::backend::{Backend, MatmulOp};
use crate::engine::kv_cache::KvCache;
use crate::engine::loader::{LoadedModel, ModelConfig, QuantMatrix};
use crate::engine::tensor;

/// Plain SwiGLU FFN (`gate`/`up`/`down`) — the dense counterpart of
/// `qwen35moe::MoeFfn`, one per layer.
struct DenseFfn {
    gate: QuantMatrix,
    up: QuantMatrix,
    down: QuantMatrix,
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
    ffn: DenseFfn,
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
    ffn: DenseFfn,
    /// Dense index into `KvCache::recurrent`.
    cache_index: usize,
}

enum Layer {
    FullAttn(FullAttnLayer),
    Recurrent(RecurrentLayer),
}

pub struct Qwen35Model {
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
    /// SSM/gated-delta-net dimensions (`qwen35.ssm.*` metadata).
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

impl Qwen35Model {
    pub fn load_with_backend(loaded: &LoadedModel, backend: Arc<dyn Backend>) -> Result<Self> {
        let config = loaded.config.clone();
        let n_layer = config.n_layer;

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

            let ffn = DenseFfn {
                gate: get_matrix("ffn_gate.weight")?,
                up: get_matrix("ffn_up.weight")?,
                down: get_matrix("ffn_down.weight")?,
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
                    "blk.{i} has NextN/MTP tensors — speculative-decoding blocks are not yet supported by orangu-server"
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

    /// `(n_full_attn, n_recurrent)` layer counts — used to size a fresh
    /// [`KvCache`].
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

impl ModelForward for Qwen35Model {
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

    fn forward(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        _slot_id: usize,
    ) -> Result<Vec<f32>> {
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
        anyhow::bail!("embeddings are not yet supported for Qwen3.5 models")
    }
}

impl Qwen35Model {
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
        let ffn_out = self.dense_ffn(&layer.ffn, &normed2, n_tokens);
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
        let ffn_out = self.dense_ffn(&layer.ffn, &normed2, n_tokens);
        tensor::add_inplace(x, &ffn_out);
        Ok(())
    }

    /// Plain SwiGLU FFN — `LLM_FFN_SILU`/`LLM_FFN_PAR` (`build_layer_ffn`,
    /// `src/models/qwen35.cpp`), the same computation as `engine::arch::
    /// llama`'s own dense FFN.
    fn dense_ffn(&self, ffn: &DenseFfn, normed: &[f32], n_tokens: usize) -> Vec<f32> {
        let mut gate_up = self.backend.matmul_batch(&[
            MatmulOp {
                x: normed,
                n_tokens,
                w: &ffn.gate,
            },
            MatmulOp {
                x: normed,
                n_tokens,
                w: &ffn.up,
            },
        ]);
        let up = gate_up.pop().unwrap();
        let mut gate = gate_up.pop().unwrap();
        for g in gate.iter_mut() {
            *g = tensor::silu(*g);
        }
        tensor::mul_inplace(&mut gate, &up);
        self.backend.matmul(&gate, n_tokens, &ffn.down)
    }
}

#[cfg(test)]
mod real_model_tests {
    use super::*;

    /// Cross-check against real llama.cpp (`unsloth/Ornith-1.0-9B-GGUF:
    /// Q4_K_M`, `llama-cli`/`llama-server` build b10066): given the token
    /// IDs real llama.cpp's `/tokenize` produces for "The capital of
    /// France is" (byte-level BPE — this model's `tokenizer.ggml.model =
    /// "gpt2"`), the model should predict the same top next token real
    /// llama.cpp's own `/completion` (`n_probs`) output does. Run with
    /// `ORANGU_TEST_MODEL=/path/to.gguf cargo test --release --bin
    /// orangu-server qwen35::real_model_tests -- --ignored` (a 9B-param
    /// model — expect a couple of minutes: this engine's scalar per-row
    /// dequant has no hand-tuned SIMD quantized-matmul kernel).
    #[test]
    #[ignore]
    fn qwen35_predicts_paris_after_capital_of_france() {
        let path = std::env::var("ORANGU_TEST_MODEL").expect("set ORANGU_TEST_MODEL");
        let loaded = LoadedModel::open(std::path::Path::new(&path)).expect("load model");
        assert_eq!(loaded.config.architecture, "qwen35");
        let model =
            Qwen35Model::load_with_backend(&loaded, Arc::new(crate::engine::backend::CpuBackend))
                .expect("build model");

        let mut cache = model.new_kv_cache(64);
        let tokens: Vec<u32> = vec![760, 6511, 314, 9338, 369];
        let logits = model.forward(&mut cache, &tokens, 0, 0).expect("forward");
        let (top_id, _) = logits
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        assert_eq!(top_id, 11751, "expected ' Paris' (11751) as top prediction");
    }
}
