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

//! Qwen3-Next (`general.architecture = "qwen3next"`).
//!
//! This reuses the same full-attention and MoE FFN shape as `qwen35moe`.
//! The recurrent path differs in two places:
//! - `ssm_ba.weight` packs beta and alpha together.
//! - older checkpoints can store recurrent QKV and z in `ssm_in.weight`
//!   instead of the split `attn_qkv.weight` and `attn_gate.weight`.

use anyhow::{Context, Result};
use std::sync::Arc;

use super::ModelForward;
use crate::engine::backend::{Backend, MatmulOp};
use crate::engine::kv_cache::KvCache;
use crate::engine::loader::{ExpertQuantMatrix, LoadedModel, ModelConfig, QuantMatrix};
use crate::engine::moe_stats;
use crate::engine::tensor;

struct MoeFfn {
    gate_inp: QuantMatrix,
    gate_exps: ExpertQuantMatrix,
    up_exps: ExpertQuantMatrix,
    down_exps: ExpertQuantMatrix,
    gate_inp_shexp: Vec<f32>,
    gate_shexp: QuantMatrix,
    up_shexp: QuantMatrix,
    down_shexp: QuantMatrix,
}

struct FullAttnLayer {
    attn_norm: Vec<f32>,
    wq: QuantMatrix,
    attn_q_norm: Vec<f32>,
    wk: QuantMatrix,
    attn_k_norm: Vec<f32>,
    wv: QuantMatrix,
    wo: QuantMatrix,
    post_attention_norm: Vec<f32>,
    ffn: MoeFfn,
    cache_index: usize,
}

struct RecurrentLayer {
    attn_norm: Vec<f32>,
    wqkv: QuantMatrix,
    wqkv_gate: QuantMatrix,
    ssm_beta_alpha: QuantMatrix,
    ssm_conv1d: Vec<f32>,
    ssm_dt_bias: Vec<f32>,
    ssm_a: Vec<f32>,
    ssm_norm: Vec<f32>,
    ssm_out: QuantMatrix,
    post_attention_norm: Vec<f32>,
    ffn: MoeFfn,
    cache_index: usize,
}

enum Layer {
    FullAttn(FullAttnLayer),
    Recurrent(RecurrentLayer),
}

pub struct Qwen3NextModel {
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
    ssm_d_conv: usize,
    ssm_head_dim: usize,
    ssm_n_group: usize,
    ssm_dt_rank: usize,
    layers: Vec<Layer>,
}

impl Qwen3NextModel {
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
        let ssm_inner_size = loaded
            .metadata_u64("ssm.inner_size")
            .context("missing ssm.inner_size")? as usize;

        anyhow::ensure!(
            ssm_dt_rank > 0 && ssm_n_group > 0,
            "ssm.time_step_rank and ssm.group_count must be nonzero"
        );
        anyhow::ensure!(
            ssm_dt_rank % ssm_n_group == 0,
            "ssm.time_step_rank {ssm_dt_rank} must be a multiple of ssm.group_count {ssm_n_group}"
        );
        anyhow::ensure!(
            ssm_inner_size == ssm_head_dim * ssm_dt_rank,
            "qwen3next expects ssm.inner_size ({ssm_inner_size}) = ssm.state_size ({ssm_head_dim}) * ssm.time_step_rank ({ssm_dt_rank})"
        );

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

        let key_dim = ssm_head_dim * ssm_n_group;
        let value_dim = ssm_head_dim * ssm_dt_rank;

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
                let (wqkv, wqkv_gate) = if loaded.has_tensor(&format!("blk.{i}.attn_qkv.weight")) {
                    (
                        get_matrix("attn_qkv.weight")?,
                        get_matrix("attn_gate.weight")?,
                    )
                } else {
                    let mixed = get_matrix("ssm_in.weight")?;
                    let qkv_out_dim = 2 * key_dim + value_dim;
                    anyhow::ensure!(
                        mixed.out_dim == qkv_out_dim + value_dim,
                        "blk.{i}.ssm_in.weight has out_dim {}, expected {}",
                        mixed.out_dim,
                        qkv_out_dim + value_dim,
                    );
                    (
                        mixed.rows(0, qkv_out_dim),
                        mixed.rows(qkv_out_dim, value_dim),
                    )
                };
                let ssm_beta_alpha = if loaded.has_tensor(&format!("blk.{i}.ssm_ba.weight")) {
                    get_matrix("ssm_ba.weight")?
                } else {
                    get_matrix("ssm_beta_alpha.weight")?
                };
                anyhow::ensure!(
                    ssm_beta_alpha.out_dim == 2 * ssm_dt_rank,
                    "blk.{i}.ssm_ba.weight has out_dim {}, expected {}",
                    ssm_beta_alpha.out_dim,
                    2 * ssm_dt_rank,
                );
                layers.push(Layer::Recurrent(RecurrentLayer {
                    attn_norm: get("attn_norm.weight")?,
                    wqkv,
                    wqkv_gate,
                    ssm_beta_alpha,
                    ssm_conv1d: get("ssm_conv1d.weight")?,
                    ssm_dt_bias: get("ssm_dt.bias")?,
                    ssm_a: get("ssm_a")?,
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
        }

        Ok(Self {
            config,
            backend,
            tok_embeddings,
            output_norm,
            output_weight,
            n_head: 0,
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

fn split_beta_alpha(
    mixed: &[f32],
    n_tokens: usize,
    n_k_heads: usize,
    n_v_heads: usize,
) -> (Vec<f32>, Vec<f32>) {
    let group = n_v_heads / n_k_heads;
    let mut beta = vec![0f32; n_tokens * n_v_heads];
    let mut alpha = vec![0f32; n_tokens * n_v_heads];
    for t in 0..n_tokens {
        let src = &mixed[t * 2 * n_v_heads..(t + 1) * 2 * n_v_heads];
        let beta_t = &mut beta[t * n_v_heads..(t + 1) * n_v_heads];
        let alpha_t = &mut alpha[t * n_v_heads..(t + 1) * n_v_heads];
        for kh in 0..n_k_heads {
            let src_off = kh * 2 * group;
            let dst_off = kh * group;
            beta_t[dst_off..dst_off + group].copy_from_slice(&src[src_off..src_off + group]);
            alpha_t[dst_off..dst_off + group]
                .copy_from_slice(&src[src_off + group..src_off + 2 * group]);
        }
    }
    (beta, alpha)
}

impl ModelForward for Qwen3NextModel {
    fn vulkan_backend(&self) -> Option<&crate::engine::backend::vulkan::VulkanBackend> {
        self.backend.as_wgpu()
    }

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
        Ok(self.backend.matmul(last, 1, &self.output_weight))
    }

    fn forward_hidden_states(&self, _tokens: &[u32]) -> Result<Vec<f32>> {
        anyhow::bail!("embeddings are not yet supported for Qwen3-Next models")
    }
}

impl Qwen3NextModel {
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

        let mut normed = x.to_vec();
        tensor::rmsnorm_inplace(&mut normed, &layer.attn_norm, n_tokens, n_embd, eps);

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

        let mut attn_out = Vec::new();
        crate::engine::attention::attention(
            &mut attn_out,
            &q,
            layer_cache,
            &crate::engine::attention::Params {
                backend: self.backend.as_ref(),
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

        let mut mixed_z_ba = self.backend.matmul_batch(&[
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
                w: &layer.ssm_beta_alpha,
            },
        ]);
        let mixed_ba = mixed_z_ba.pop().unwrap();
        let z = mixed_z_ba.pop().unwrap();
        let qkv_mixed = mixed_z_ba.pop().unwrap();
        let (mut beta, alpha) = split_beta_alpha(&mixed_ba, n_tokens, n_k_heads, n_v_heads);
        for b in &mut beta {
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
            for v in &mut conv_out {
                *v = tensor::silu(*v);
            }
            let (q_conv, rest) = conv_out.split_at_mut(key_dim);
            let (k_conv, v_conv) = rest.split_at_mut(key_dim);

            for h in 0..n_k_heads {
                tensor::l2_norm_inplace(&mut q_conv[h * head_dim..(h + 1) * head_dim], eps);
                tensor::l2_norm_inplace(&mut k_conv[h * head_dim..(h + 1) * head_dim], eps);
            }
            for v in q_conv.iter_mut() {
                *v *= q_scale;
            }

            let mut attn_out = vec![0f32; value_dim];
            for vh in 0..n_v_heads {
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
                let out = &mut attn_out[vh * head_dim..(vh + 1) * head_dim];
                for j in 0..head_dim {
                    let mut sum = 0f32;
                    for i in 0..head_dim {
                        sum += qh[i] * state[i * head_dim + j];
                    }
                    out[j] = sum;
                }
            }

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

    fn moe_ffn(&self, ffn: &MoeFfn, normed: &[f32], n_tokens: usize) -> Vec<f32> {
        let n_embd = self.config.n_embd;
        let mut out = vec![0f32; n_tokens * n_embd];
        let mut experts =
            moe_stats::LayerRecorder::for_tensors(&[&ffn.gate_exps, &ffn.up_exps, &ffn.down_exps]);
        // Route the whole batch first, so the union can be taken before any
        // expert's weights are read — see `super::evaluate_routed_experts`.
        let mut selection: Vec<Vec<(usize, f32)>> = (0..n_tokens)
            .map(|t| {
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
                indexed
                    .into_iter()
                    .map(|(expert, weight)| (expert, weight / weight_sum))
                    .collect()
            })
            .collect();
        // Trim to the expert budget *before* anything is recorded or
        // read: the counters should describe the work actually done,
        // and a dropped expert's weights must never be fetched.
        super::apply_expert_budget(&mut selection, &ffn.gate_exps);
        for picks in &selection {
            picks.iter().for_each(|&(e, _)| experts.select(e));
        }

        let contribs = super::evaluate_routed_experts(&selection, |expert, members| {
            let inputs: Vec<&[f32]> = members
                .iter()
                .map(|&(t, _)| &normed[t * n_embd..(t + 1) * n_embd])
                .collect();
            let gate =
                super::project_expert(&ffn.gate_exps, expert, 0, ffn.gate_exps.out_dim, &inputs);
            let up = super::project_expert(&ffn.up_exps, expert, 0, ffn.up_exps.out_dim, &inputs);
            let hidden: Vec<Vec<f32>> = gate
                .into_iter()
                .zip(up)
                .map(|(gate, up)| {
                    let mut h: Vec<f32> = gate.iter().map(|&g| tensor::silu(g)).collect();
                    tensor::mul_inplace(&mut h, &up);
                    h
                })
                .collect();
            let hidden_refs: Vec<&[f32]> = hidden.iter().map(Vec::as_slice).collect();
            super::project_expert(
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
        });
        experts.loaded_once_per_distinct_expert();

        for t in 0..n_tokens {
            let x_t = &normed[t * n_embd..(t + 1) * n_embd];
            let mut moe_out = vec![0f32; n_embd];
            for contrib in &contribs[t] {
                for (o, d) in moe_out.iter_mut().zip(contrib.iter()) {
                    *o += d;
                }
            }

            let shared_gate = tensor::sigmoid(tensor::dot(x_t, &ffn.gate_inp_shexp));
            let cpu = crate::engine::backend::CpuBackend;
            let use_cpu_shared = !self.backend.supports_type(ffn.gate_shexp.ggml_type())
                || !self.backend.supports_type(ffn.up_shexp.ggml_type())
                || !self.backend.supports_type(ffn.down_shexp.ggml_type());
            let (shexp_gate, shexp_up) = if use_cpu_shared {
                (
                    cpu.matmul(x_t, 1, &ffn.gate_shexp),
                    cpu.matmul(x_t, 1, &ffn.up_shexp),
                )
            } else {
                let mut gate_up = self.backend.matmul_batch(&[
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
                let up = gate_up.pop().unwrap();
                let gate = gate_up.pop().unwrap();
                (gate, up)
            };
            let mut shexp_h: Vec<f32> = shexp_gate.iter().map(|&g| tensor::silu(g)).collect();
            tensor::mul_inplace(&mut shexp_h, &shexp_up);
            let mut shexp_out = if use_cpu_shared {
                cpu.matmul(&shexp_h, 1, &ffn.down_shexp)
            } else {
                self.backend.matmul(&shexp_h, 1, &ffn.down_shexp)
            };
            for v in shexp_out.iter_mut() {
                *v *= shared_gate;
            }

            let dst = &mut out[t * n_embd..(t + 1) * n_embd];
            for i in 0..n_embd {
                dst[i] = moe_out[i] + shexp_out[i];
            }
        }
        experts.commit(n_tokens);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::split_beta_alpha;

    #[test]
    fn split_beta_alpha_unpacks_per_group_layout() {
        let mixed = vec![1.0, 2.0, 11.0, 12.0, 3.0, 4.0, 13.0, 14.0];
        let (beta, alpha) = split_beta_alpha(&mixed, 1, 2, 4);
        assert_eq!(beta, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(alpha, vec![11.0, 12.0, 13.0, 14.0]);
    }
}
