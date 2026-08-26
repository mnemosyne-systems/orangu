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

//! The hybrid full-attention / gated-DeltaNet trunk every Qwen 3.5-family
//! architecture shares — `qwen35` (dense FFN), `qwen35moe` and `qwen3next`
//! (both routed + shared-expert MoE). Confirmed against real upstream
//! `llama.cpp` source (`src/models/qwen35.cpp`, `src/models/qwen35moe.cpp`,
//! `src/models/qwen3next.cpp` and the `llm_build_delta_net_base` code all
//! three call, read directly rather than guessed): the three files build
//! *the same* attention sub-layer and differ only in `build_layer_ffn`.
//!
//! That is why this module exists. Each architecture module used to carry
//! its own copy of the layer shapes, the loader, and both forward halves —
//! about 500 duplicated lines each, three ways. A fix or an optimization in
//! one had no way to reach the other two, and two of the three had already
//! drifted (see [`trunk_layer_count`]). Here the trunk is written once and
//! the architecture modules supply only their FFN, through [`HybridFfn`].
//!
//! ## The layer shape
//!
//! Every layer is a standard pre-norm block either way — `x +=
//! sub(rmsnorm(x)); x += ffn(rmsnorm(x))` — and only the `sub` differs:
//!
//! - **Full-attention layers** (every `full_attention_interval`-th, and the
//!   file may instead name them outright in `attention.recurrent_layers`):
//!   a *joint* query+gate projection (`attn_q`'s output is `[Q_h, gate_h]`
//!   interleaved per head), Q/K-norm, partial rotary (`rope.dimension_count`
//!   is a fraction of `attention.key_length` here), standard GQA, then the
//!   attention output is gated by `sigmoid(gate)` before the output
//!   projection.
//! - **Linear-attention (gated-DeltaNet) layers**: a joint QKV projection
//!   through a causal depthwise conv1d + SiLU, per-head L2-normed Q/K, a
//!   scalar-per-head softplus-gated decay, and a delta-rule recurrent state
//!   update — implemented here only in its *autoregressive*
//!   (one-token-at-a-time) form, not the chunked/parallel form real
//!   `llama.cpp` also has. The two are mathematically identical (chunking
//!   is a prefill-throughput optimization, not different math — confirmed
//!   by reading `build_delta_net_chunking` and
//!   `build_delta_net_autoregressive` side by side), so this is a real,
//!   deliberate, documented scope reduction (slower prompt processing on
//!   long prompts, not a correctness gap), not a shortcut.
//!
//! ## Tensor-layout variation this absorbs
//!
//! The three architectures name the recurrent layer's projections
//! differently, and [`RecurrentWeights::load`] takes whichever a file
//! carries rather than making each architecture module decide:
//!
//! - Recurrent QKV and the output gate `z` are either the split
//!   `attn_qkv.weight` + `attn_gate.weight`, or one fused `ssm_in.weight`
//!   sliced into the two.
//! - Beta and alpha are either the split `ssm_beta.weight` +
//!   `ssm_alpha.weight`, or one `ssm_ba.weight` (`ssm_beta_alpha.weight` in
//!   older conversions) whose rows interleave the two per K/V group — see
//!   [`split_beta_alpha`].
//!
//! ## Not implemented
//!
//! **NextN/MTP** (speculative-decoding-only extra decoder blocks): the
//! trunk is the `block_count` layers a file declares *less* its
//! `nextn_predict_layers`, so an MTP block is never touched whether it sits
//! past `block_count` or is counted inside it — see [`trunk_layer_count`].
//!
//! **Multi-section RoPE** ("M-RoPE"/"IMRoPE", `rope.dimension_sections`) is
//! implemented as plain NEOX rope: for text-only input every rope "position
//! channel" (t/h/w/e) carries the same linear position, at which point the
//! sections mechanism (confirmed by reading `ggml_mrope_cache_init`) is a
//! no-op — it only matters for genuinely multi-axis (vision/video) position
//! input, which this engine doesn't accept.

use anyhow::{Context, Result};
use std::sync::Arc;

use rayon::prelude::*;

use crate::engine::backend::{Backend, MatmulOp};
use crate::engine::kv_cache::{KvCache, LayerCache, RecurrentSpec};
use crate::engine::loader::{ExpertQuantMatrix, LoadedModel, ModelConfig, QuantMatrix};
use crate::engine::moe_stats;
use crate::engine::tensor;

/// Everything the trunk's two forward halves need from a file's
/// hyper-parameters, read once at load rather than per layer.
pub(crate) struct Dims {
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    /// `attention.key_length` — the *attention* head dimension, distinct
    /// from the gated-DeltaNet one below.
    pub head_dim: usize,
    pub rope_dim: usize,
    pub rope_freq_base: f32,
    pub rms_eps: f32,
    pub ssm_d_conv: usize,
    /// `head_k_dim == head_v_dim` for gated-DeltaNet (required by the
    /// recurrence itself).
    pub ssm_head_dim: usize,
    /// Number of K/V "groups" the causal conv1d/Q/K live in
    /// (`ssm.group_count`) — smaller than `ssm_dt_rank` (the number of
    /// value heads); a K/V group is reused (tiled, not block-grouped —
    /// confirmed against `ggml_compute_forward_repeat_f32`) across
    /// `ssm_dt_rank / ssm_n_group` value heads.
    pub ssm_n_group: usize,
    pub ssm_dt_rank: usize,
}

impl Dims {
    pub(crate) fn from_loaded(loaded: &LoadedModel) -> Result<Self> {
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

        anyhow::ensure!(
            ssm_dt_rank > 0 && ssm_n_group > 0,
            "ssm.time_step_rank and ssm.group_count must be nonzero"
        );
        anyhow::ensure!(
            ssm_dt_rank.is_multiple_of(ssm_n_group),
            "ssm.time_step_rank {ssm_dt_rank} must be a multiple of ssm.group_count {ssm_n_group}"
        );
        // Optional in this family's older conversions; when a file does
        // carry it, it is a redundant statement of `state_size *
        // time_step_rank` and disagreeing with it means one of the three was
        // misread.
        if let Some(inner) = loaded.metadata_u64("ssm.inner_size") {
            anyhow::ensure!(
                inner as usize == ssm_head_dim * ssm_dt_rank,
                "ssm.inner_size ({inner}) should be ssm.state_size ({ssm_head_dim}) * ssm.time_step_rank ({ssm_dt_rank})"
            );
        }

        Ok(Self {
            n_embd: loaded.config.n_embd,
            n_head: loaded.config.n_head,
            n_head_kv: loaded.config.n_head_kv,
            head_dim,
            rope_dim: loaded.config.rope_dim,
            rope_freq_base: loaded.config.rope_freq_base,
            rms_eps: loaded.config.rms_eps,
            ssm_d_conv,
            ssm_head_dim,
            ssm_n_group,
            ssm_dt_rank,
        })
    }

    pub(crate) fn key_dim(&self) -> usize {
        self.ssm_head_dim * self.ssm_n_group
    }

    pub(crate) fn value_dim(&self) -> usize {
        self.ssm_head_dim * self.ssm_dt_rank
    }

    pub(crate) fn conv_channels(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }
}

/// The number of *trunk* layers in `loaded` — `block_count` less
/// `nextn_predict_layers`.
///
/// `block_count` counts the multi-token-prediction blocks too when a release
/// ships one, and the releases that ship one put it *inside* the count
/// rather than past the end: the last block is an MTP head carrying only
/// `ffn_*`, `post_attention_norm` and `nextn.*` tensors, so running it as a
/// trunk layer fails on the attention tensors it does not have. A file with
/// no MTP head is unaffected — the key is absent and this is a subtraction
/// of zero.
///
/// This was `qwen35moe`-only before the trunk was shared, which is exactly
/// the drift a shared trunk removes: `Qwen3.8-27B` is a `qwen35` *dense*
/// file with `nextn_predict_layers = 1`, and it failed to load on
/// `blk.64.attn_qkv.weight` — the MTP block, read as layer 64 of 65.
pub(crate) fn trunk_layer_count(loaded: &LoadedModel) -> Result<usize> {
    loaded
        .config
        .n_layer
        .checked_sub(loaded.metadata_u64("nextn_predict_layers").unwrap_or(0) as usize)
        .filter(|&n| n > 0)
        .context("nextn_predict_layers is not smaller than block_count")
}

/// Which of the `n_layer` trunk layers are gated-DeltaNet (linear
/// attention) rather than full attention. A file may name them outright;
/// otherwise every `full_attention_interval`-th layer is full attention and
/// the rest are recurrent.
pub(crate) fn recurrent_layer_mask(loaded: &LoadedModel, n_layer: usize) -> Vec<bool> {
    let interval = loaded.metadata_u64("full_attention_interval").unwrap_or(4) as usize;
    loaded
        .metadata_array_u64("attention.recurrent_layers")
        .map(|arr| arr.iter().map(|&v| v != 0).collect())
        .unwrap_or_else(|| {
            (0..n_layer)
                .map(|i| interval == 0 || (i + 1) % interval != 0)
                .collect()
        })
}

/// A full-attention layer of the trunk: the sub-layer itself, plus the two
/// RMSNorms this trunk brackets it and its FFN with.
pub(crate) struct FullAttnWeights {
    attn_norm: Vec<f32>,
    post_attention_norm: Vec<f32>,
    attn: FullAttn,
}

/// One full-attention sub-layer's projections — everything *between* the
/// norms that bracket it, and nothing about those norms.
///
/// Split out from [`FullAttnWeights`] because [`super::qwen4exp`] runs this
/// exact computation with no pre-norm at all: its residual stream is
/// `hyper_connection.count` parallel streams, and the mixer that collapses
/// them into the one vector a sub-layer reads is also the only
/// normalization that sub-layer gets. From the joint query+gate projection
/// through to the output projection the two are the same, so it is written
/// once here rather than copied — the same reason this module exists at
/// all.
pub(crate) struct FullAttn {
    /// Joint query+gate projection: per head, `[Q(head_dim), gate(head_dim)]`
    /// interleaved — `out_dim == 2 * n_head * head_dim`.
    wq: QuantMatrix,
    attn_q_norm: Vec<f32>,
    wk: QuantMatrix,
    attn_k_norm: Vec<f32>,
    wv: QuantMatrix,
    wo: QuantMatrix,
    /// Dense index into `KvCache::layers` (every full-attention layer has
    /// its own cache — no cross-layer sharing in this architecture).
    pub(crate) cache_index: usize,
}

/// Where a recurrent layer's per-head beta and alpha come from — two
/// projections, or one whose rows interleave them.
enum BetaAlpha {
    Split {
        beta: QuantMatrix,
        alpha: QuantMatrix,
    },
    Packed(QuantMatrix),
}

/// A gated-DeltaNet layer of the trunk: the sub-layer itself, plus the two
/// RMSNorms this trunk brackets it and its FFN with.
pub(crate) struct RecurrentWeights {
    attn_norm: Vec<f32>,
    post_attention_norm: Vec<f32>,
    recurrent: Recurrent,
}

/// One gated-DeltaNet sub-layer's weights — everything *between* the norms
/// that bracket it, for the same reason [`FullAttn`] is split out.
pub(crate) struct Recurrent {
    /// Joint Q/K/V mix: `[q(key_dim), k(key_dim), v(value_dim)]`.
    wqkv: QuantMatrix,
    wqkv_gate: QuantMatrix,
    beta_alpha: BetaAlpha,
    /// `[conv_channels, d_conv]`, channel-major (ggml's own tensor order).
    ssm_conv1d: Vec<f32>,
    /// `[num_v_heads]` — added to the alpha projection before softplus.
    ssm_dt_bias: Vec<f32>,
    /// `[num_v_heads]` — per-head learned decay scale (typically negative;
    /// `exp(softplus(alpha + dt_bias) * ssm_a)` is the per-head decay).
    ssm_a: Vec<f32>,
    /// `[head_v_dim]` — the gated output RMSNorm's learned weight.
    ssm_norm: Vec<f32>,
    ssm_out: QuantMatrix,
    /// Dense index into `KvCache::recurrent`.
    pub(crate) cache_index: usize,
}

/// What multiplies a gated-DeltaNet layer's normed output — the one
/// numerical difference between this trunk's own architectures and
/// [`super::qwen4exp`]'s otherwise identical delta net.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OutputGate {
    /// `silu(z)` — `qwen35`, `qwen35moe`, `qwen3next`.
    Silu,
    /// `sigmoid(z)` — `qwen4exp` (its `build_norm_gated`).
    Sigmoid,
}

impl OutputGate {
    fn apply(self, z: f32) -> f32 {
        match self {
            Self::Silu => tensor::silu(z),
            Self::Sigmoid => tensor::sigmoid(z),
        }
    }
}

/// Per-layer tensor readers, so a loader body reads
/// `blk.{i}.<suffix>` without repeating the name-building three times per
/// architecture.
pub(crate) struct LayerTensors<'a> {
    pub(crate) loaded: &'a LoadedModel,
    pub(crate) i: usize,
}

impl LayerTensors<'_> {
    pub(crate) fn vec(&self, suffix: &str) -> Result<Vec<f32>> {
        let name = format!("blk.{}.{suffix}", self.i);
        Ok(self
            .loaded
            .tensor(&name)
            .with_context(|| format!("loading {name}"))?
            .0)
    }

    pub(crate) fn matrix(&self, suffix: &str) -> Result<QuantMatrix> {
        let name = format!("blk.{}.{suffix}", self.i);
        self.loaded
            .matrix(&name)
            .with_context(|| format!("loading {name}"))
    }

    pub(crate) fn expert_matrix(&self, suffix: &str) -> Result<ExpertQuantMatrix> {
        let name = format!("blk.{}.{suffix}", self.i);
        self.loaded
            .expert_matrix(&name)
            .with_context(|| format!("loading {name}"))
    }

    pub(crate) fn has(&self, suffix: &str) -> bool {
        self.loaded.has_tensor(&format!("blk.{}.{suffix}", self.i))
    }
}

impl FullAttnWeights {
    fn load(t: &LayerTensors<'_>, cache_index: usize) -> Result<Self> {
        Ok(Self {
            attn_norm: t.vec("attn_norm.weight")?,
            post_attention_norm: t.vec("post_attention_norm.weight")?,
            attn: FullAttn::load(t, cache_index)?,
        })
    }
}

impl FullAttn {
    pub(crate) fn load(t: &LayerTensors<'_>, cache_index: usize) -> Result<Self> {
        Ok(Self {
            wq: t.matrix("attn_q.weight")?,
            attn_q_norm: t.vec("attn_q_norm.weight")?,
            wk: t.matrix("attn_k.weight")?,
            attn_k_norm: t.vec("attn_k_norm.weight")?,
            wv: t.matrix("attn_v.weight")?,
            wo: t.matrix("attn_output.weight")?,
            cache_index,
        })
    }

    /// The sub-layer itself: `normed` (`[n_tokens, n_embd]`, already
    /// whatever normalization the architecture puts in front of it) in, the
    /// vector to add back into the residual stream out.
    ///
    /// `selection`, when given, names per token exactly which cached
    /// positions that token may attend, in ascending order — the
    /// block-sparse key set [`super::qwen4exp`]'s indexer picks. `None` is
    /// ordinary causal attention over everything up to the token's own
    /// position, which is what this trunk's own three architectures do and
    /// what the shared (GPU-capable) `engine::attention` path implements.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward(
        &self,
        backend: &dyn Backend,
        dims: &Dims,
        cache: &mut KvCache,
        normed: &[f32],
        n_tokens: usize,
        start_pos: usize,
        selection: Option<&[Vec<usize>]>,
    ) -> Vec<f32> {
        let eps = dims.rms_eps;
        let head_dim = dims.head_dim;
        let n_head = dims.n_head;
        let n_head_kv = dims.n_head_kv;
        let kv_dim = n_head_kv * head_dim;

        // Joint Q+gate projection, K, and V are all independent projections
        // of the same normed input — one batched dispatch instead of three
        // sequential round-trips (see `Backend::matmul_batch`). Per head,
        // the Q+gate projection is [Q(head_dim), gate(head_dim)].
        let mut qgkv = backend.matmul_batch(&[
            MatmulOp {
                x: normed,
                n_tokens,
                w: &self.wq,
            },
            MatmulOp {
                x: normed,
                n_tokens,
                w: &self.wk,
            },
            MatmulOp {
                x: normed,
                n_tokens,
                w: &self.wv,
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
        tensor::rmsnorm_inplace(&mut q, &self.attn_q_norm, n_tokens * n_head, head_dim, eps);
        for t in 0..n_tokens {
            let pos = start_pos + t;
            tensor::rope_apply_inplace(
                &mut q[t * n_head * head_dim..(t + 1) * n_head * head_dim],
                n_head,
                head_dim,
                dims.rope_dim,
                pos,
                dims.rope_freq_base,
            );
        }

        tensor::rmsnorm_inplace(
            &mut k,
            &self.attn_k_norm,
            n_tokens * n_head_kv,
            head_dim,
            eps,
        );

        let layer_cache = &mut cache.layers[self.cache_index];
        for t in 0..n_tokens {
            let pos = start_pos + t;
            tensor::rope_apply_inplace(
                &mut k[t * kv_dim..(t + 1) * kv_dim],
                n_head_kv,
                head_dim,
                dims.rope_dim,
                pos,
                dims.rope_freq_base,
            );
            layer_cache.push(
                &k[t * kv_dim..(t + 1) * kv_dim],
                &v[t * kv_dim..(t + 1) * kv_dim],
            );
        }

        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut attn_out: Vec<f32> = Vec::new();
        match selection {
            // Plain causal attention, on the GPU when the batch is wide
            // enough -- `engine::attention` owns that choice for every
            // architecture.
            None => {
                crate::engine::attention::attention(
                    &mut attn_out,
                    &q,
                    layer_cache,
                    &crate::engine::attention::Params {
                        backend,
                        // This layer's card — see `attention::Params::device`.
                        device: self.wo.device(),
                        n_head,
                        n_head_kv,
                        head_dim,
                        scale,
                        causal: true,
                        n_swa: 0,
                        start_pos,
                        n_tokens,
                    },
                    |t| (0, start_pos + t),
                );
            }
            // A gathered key set cannot be expressed as a window, so the
            // shared path has nothing to offer here and this is the plain
            // per-head softmax over the chosen rows.
            Some(chosen) => {
                attn_out = attend_selected(&q, layer_cache, chosen, dims, scale);
            }
        }
        // Gate the attention output (sigmoid), then project.
        for (o, &g) in attn_out.iter_mut().zip(gate.iter()) {
            *o *= tensor::sigmoid(g);
        }
        backend.matmul(&attn_out, n_tokens, &self.wo)
    }
}

impl RecurrentWeights {
    fn load(t: &LayerTensors<'_>, dims: &Dims, cache_index: usize) -> Result<Self> {
        Ok(Self {
            attn_norm: t.vec("attn_norm.weight")?,
            post_attention_norm: t.vec("post_attention_norm.weight")?,
            recurrent: Recurrent::load(t, dims, cache_index)?,
        })
    }
}

impl Recurrent {
    pub(crate) fn load(t: &LayerTensors<'_>, dims: &Dims, cache_index: usize) -> Result<Self> {
        let qkv_out_dim = dims.conv_channels();
        let value_dim = dims.value_dim();
        let (wqkv, wqkv_gate) = if t.has("attn_qkv.weight") {
            (t.matrix("attn_qkv.weight")?, t.matrix("attn_gate.weight")?)
        } else {
            // One fused projection carrying the QKV mix and the output gate
            // `z` back to back.
            let mixed = t.matrix("ssm_in.weight")?;
            anyhow::ensure!(
                mixed.out_dim == qkv_out_dim + value_dim,
                "blk.{}.ssm_in.weight has out_dim {}, expected {}",
                t.i,
                mixed.out_dim,
                qkv_out_dim + value_dim,
            );
            (
                mixed.rows(0, qkv_out_dim),
                mixed.rows(qkv_out_dim, value_dim),
            )
        };

        let beta_alpha = if t.has("ssm_beta.weight") {
            BetaAlpha::Split {
                beta: t.matrix("ssm_beta.weight")?,
                alpha: t.matrix("ssm_alpha.weight")?,
            }
        } else {
            let packed = if t.has("ssm_ba.weight") {
                t.matrix("ssm_ba.weight")?
            } else {
                t.matrix("ssm_beta_alpha.weight")?
            };
            anyhow::ensure!(
                packed.out_dim == 2 * dims.ssm_dt_rank,
                "blk.{}'s packed beta/alpha projection has out_dim {}, expected {}",
                t.i,
                packed.out_dim,
                2 * dims.ssm_dt_rank,
            );
            BetaAlpha::Packed(packed)
        };

        Ok(Self {
            wqkv,
            wqkv_gate,
            beta_alpha,
            ssm_conv1d: t.vec("ssm_conv1d.weight")?,
            ssm_dt_bias: t.vec("ssm_dt.bias")?,
            ssm_a: t.vec("ssm_a")?,
            ssm_norm: t.vec("ssm_norm.weight")?,
            ssm_out: t.matrix("ssm_out.weight")?,
            cache_index,
        })
    }

    /// The sub-layer itself: `normed` (`[n_tokens, n_embd]`) in, the vector
    /// to add back into the residual stream out. `gate` picks the output
    /// gate's nonlinearity — see [`OutputGate`].
    pub(crate) fn forward(
        &self,
        backend: &dyn Backend,
        dims: &Dims,
        cache: &mut KvCache,
        normed: &[f32],
        n_tokens: usize,
        gate: OutputGate,
    ) -> Vec<f32> {
        let n_embd = dims.n_embd;
        let eps = dims.rms_eps;
        let key_dim = dims.key_dim();
        let value_dim = dims.value_dim();
        let head_dim = dims.ssm_head_dim;
        let n_k_heads = dims.ssm_n_group;
        let n_v_heads = dims.ssm_dt_rank;
        let q_scale = 1.0 / (head_dim as f32).sqrt();

        // Every projection here is of the same normed input — one batched
        // dispatch instead of three or four sequential round-trips (see
        // `Backend::matmul_batch`).
        let mut ops = vec![
            MatmulOp {
                x: normed,
                n_tokens,
                w: &self.wqkv,
            },
            MatmulOp {
                x: normed,
                n_tokens,
                w: &self.wqkv_gate,
            },
        ];
        match &self.beta_alpha {
            BetaAlpha::Split { beta, alpha } => {
                ops.push(MatmulOp {
                    x: normed,
                    n_tokens,
                    w: beta,
                });
                ops.push(MatmulOp {
                    x: normed,
                    n_tokens,
                    w: alpha,
                });
            }
            BetaAlpha::Packed(packed) => ops.push(MatmulOp {
                x: normed,
                n_tokens,
                w: packed,
            }),
        }
        let mut projected = backend.matmul_batch(&ops);
        let (mut beta, alpha) = match &self.beta_alpha {
            BetaAlpha::Split { .. } => {
                let alpha = projected.pop().unwrap();
                let beta = projected.pop().unwrap();
                (beta, alpha)
            }
            BetaAlpha::Packed(_) => {
                let mixed = projected.pop().unwrap();
                split_beta_alpha(&mixed, n_tokens, n_k_heads, n_v_heads)
            }
        };
        let z = projected.pop().unwrap();
        let qkv_mixed = projected.pop().unwrap();

        for b in beta.iter_mut() {
            *b = tensor::sigmoid(*b);
        }
        let mut decay = vec![0f32; n_tokens * n_v_heads];
        for t in 0..n_tokens {
            for h in 0..n_v_heads {
                let a = alpha[t * n_v_heads + h] + self.ssm_dt_bias[h];
                let log_decay = tensor::softplus(a) * self.ssm_a[h];
                decay[t * n_v_heads + h] = log_decay.exp();
            }
        }

        let mut sub_out = vec![0f32; n_tokens * n_embd];
        let ssm_state = &mut cache.recurrent[self.cache_index];
        for t in 0..n_tokens {
            let mixed =
                &qkv_mixed[t * (2 * key_dim + value_dim)..(t + 1) * (2 * key_dim + value_dim)];
            let mut conv_out = ssm_state.conv_step(mixed, &self.ssm_conv1d);
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

            // Gated RMSNorm, per head: rmsnorm(attn_out_h) * gate(z_h).
            for h in 0..n_v_heads {
                let mut normed_h = attn_out[h * head_dim..(h + 1) * head_dim].to_vec();
                tensor::rmsnorm_inplace(&mut normed_h, &self.ssm_norm, 1, head_dim, eps);
                let z_h = &z[t * value_dim + h * head_dim..t * value_dim + (h + 1) * head_dim];
                for (o, (n, zv)) in attn_out[h * head_dim..(h + 1) * head_dim]
                    .iter_mut()
                    .zip(normed_h.iter().zip(z_h.iter()))
                {
                    *o = *n * gate.apply(*zv);
                }
            }

            let projected = backend.matmul(&attn_out, 1, &self.ssm_out);
            sub_out[t * n_embd..(t + 1) * n_embd].copy_from_slice(&projected);
        }
        sub_out
    }
}

/// De-interleaves one `ssm_ba`-style projection's output into beta and
/// alpha. The rows are grouped by K/V group: within each of `n_k_heads`
/// groups, `group` beta values then `group` alpha values.
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

/// The one thing the three architectures on this trunk do differently.
///
/// `normed` is `[n_tokens, n_embd]`, already post-attention-normed; the
/// return is the same shape, to be added back into the residual stream.
pub(crate) trait HybridFfn: Send + Sync {
    fn forward(
        &self,
        backend: &dyn Backend,
        n_embd: usize,
        normed: &[f32],
        n_tokens: usize,
    ) -> Vec<f32>;

    /// [`Self::forward`] into caller-owned buffers — see
    /// `Backend::matmul_into`. Defaults to the allocating form, so an
    /// implementation only overrides it when it has something to reuse;
    /// `MoeFfn` builds its output by summing per-expert contributions and
    /// has no single projection to redirect.
    fn forward_into(
        &self,
        backend: &dyn Backend,
        out: &mut Vec<f32>,
        scratch: &mut super::FfnScratch,
        n_embd: usize,
        normed: &[f32],
        n_tokens: usize,
    ) {
        let _ = scratch;
        *out = self.forward(backend, n_embd, normed, n_tokens);
    }
}

/// Plain SwiGLU FFN (`gate`/`up`/`down`) — `LLM_FFN_SILU`/`LLM_FFN_PAR`
/// (`build_layer_ffn`, `src/models/qwen35.cpp`), the same computation
/// `engine::arch::llama` runs for Qwen2/Qwen3 and every other dense model in
/// that family, and shared with it through [`super::swiglu_ffn`].
pub(crate) struct DenseFfn {
    pub gate: QuantMatrix,
    pub up: QuantMatrix,
    pub down: QuantMatrix,
}

impl HybridFfn for DenseFfn {
    fn forward(
        &self,
        backend: &dyn Backend,
        _n_embd: usize,
        normed: &[f32],
        n_tokens: usize,
    ) -> Vec<f32> {
        super::swiglu_ffn(backend, normed, n_tokens, &self.gate, &self.up, &self.down)
    }

    fn forward_into(
        &self,
        backend: &dyn Backend,
        out: &mut Vec<f32>,
        scratch: &mut super::FfnScratch,
        _n_embd: usize,
        normed: &[f32],
        n_tokens: usize,
    ) {
        super::swiglu_ffn_into(
            backend, out, scratch, normed, n_tokens, &self.gate, &self.up, &self.down,
        );
    }
}

/// Routed top-k softmax experts (renormalized) plus one always-on,
/// separately-`sigmoid`-gated shared expert — `qwen35moe` and `qwen3next`
/// carry the identical FFN, only their recurrent tensor names differ.
pub(crate) struct MoeFfn {
    pub gate_inp: QuantMatrix,
    pub gate_exps: ExpertQuantMatrix,
    pub up_exps: ExpertQuantMatrix,
    pub down_exps: ExpertQuantMatrix,
    /// `[n_embd]` — a matmul weight with `out_dim == 1` in the reference
    /// graph (produces one shared-expert gate scalar per token); tiny, so
    /// eagerly resident and dot-producted directly rather than routed
    /// through `QuantMatrix`.
    pub gate_inp_shexp: Vec<f32>,
    pub gate_shexp: QuantMatrix,
    pub up_shexp: QuantMatrix,
    pub down_shexp: QuantMatrix,
    pub n_expert_used: usize,
}

impl MoeFfn {
    /// Loads one layer's MoE FFN. `n_expert_used` comes from the file's
    /// `expert_used_count`; every architecture that uses this FFN requires
    /// it.
    pub(crate) fn load(t: &LayerTensors<'_>, n_expert_used: usize) -> Result<Self> {
        Ok(Self {
            gate_inp: t.matrix("ffn_gate_inp.weight")?,
            gate_exps: t.expert_matrix("ffn_gate_exps.weight")?,
            up_exps: t.expert_matrix("ffn_up_exps.weight")?,
            down_exps: t.expert_matrix("ffn_down_exps.weight")?,
            gate_inp_shexp: t.vec("ffn_gate_inp_shexp.weight")?,
            gate_shexp: t.matrix("ffn_gate_shexp.weight")?,
            up_shexp: t.matrix("ffn_up_shexp.weight")?,
            down_shexp: t.matrix("ffn_down_shexp.weight")?,
            n_expert_used,
        })
    }
}

impl HybridFfn for MoeFfn {
    /// Standard top-k softmax MoE routing (renormalized over the selected
    /// experts) plus a separately-`sigmoid`-gated shared expert — see
    /// `llm_graph_context::build_moe_ffn` (the `LLAMA_EXPERT_GATING_FUNC_
    /// TYPE_SOFTMAX`, `norm_w = true` path this family uses) and
    /// `build_layer_ffn`'s shared-expert gate.
    fn forward(
        &self,
        backend: &dyn Backend,
        n_embd: usize,
        normed: &[f32],
        n_tokens: usize,
    ) -> Vec<f32> {
        let ffn = self;
        let mut out = vec![0f32; n_tokens * n_embd];
        let mut experts =
            moe_stats::LayerRecorder::for_tensors(&[&ffn.gate_exps, &ffn.up_exps, &ffn.down_exps]);
        // Route every position before touching any expert's weights, so the
        // batch's whole selection is known and the union can be taken.
        //
        // One router matmul for the whole batch, not one per token — see
        // `super::moe_router_logits`. Hoisted out of the parallel region
        // below for the same reason: it is a device submission at the head
        // of an otherwise host-side branch.
        let logits = super::moe_router_logits(backend, normed, n_tokens, &ffn.gate_inp);
        let n_expert = logits.len() / n_tokens.max(1);
        let mut selection: Vec<Vec<(usize, f32)>> = (0..n_tokens)
            .map(|t| {
                let mut probs = logits[t * n_expert..(t + 1) * n_expert].to_vec();
                tensor::softmax_inplace(&mut probs);

                let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                indexed.truncate(ffn.n_expert_used);
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

        // Routed experts — the CPU-scalar (per-row dequant + dot) bottleneck
        // of MoE decode. One weight read per *distinct* expert rather than
        // one per (token, expert): the rows are dequantized once and dotted
        // with every token that routed to this expert. See
        // `super::evaluate_routed_experts`, including why the contributions
        // come back in selection order and the summation below is unchanged.
        // The down projection's rows still fan out, so decode — where every
        // selection is distinct and the outer fan-out is only
        // `n_expert_used` wide — keeps filling every core.
        // The GPU expert path batches the three projections across experts
        // instead of issuing them one expert at a time — see
        // `super::evaluate_routed_experts_batched` for why that is the
        // whole question, and why it is only for the GPU.
        let routed_branch = || {
            if super::gpu_experts() && backend.as_wgpu().is_some() {
                super::evaluate_routed_experts_batched(
                    backend,
                    &selection,
                    normed,
                    n_embd,
                    &ffn.gate_exps,
                    &ffn.up_exps,
                    &ffn.down_exps,
                    |gate, up| {
                        let mut h: Vec<f32> = gate.iter().map(|&g| tensor::silu(g)).collect();
                        tensor::mul_inplace(&mut h, up);
                        h
                    },
                )
            } else {
                super::evaluate_routed_experts(&selection, |expert, members| {
                    let inputs: Vec<&[f32]> = members
                        .iter()
                        .map(|&(t, _)| &normed[t * n_embd..(t + 1) * n_embd])
                        .collect();
                    let gate = super::project_expert(
                        backend,
                        &ffn.gate_exps,
                        expert,
                        0,
                        ffn.gate_exps.out_dim,
                        &inputs,
                    );
                    let up = super::project_expert(
                        backend,
                        &ffn.up_exps,
                        expert,
                        0,
                        ffn.up_exps.out_dim,
                        &inputs,
                    );
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
                        backend,
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
            }
        };

        // The gated shared expert, for the whole batch in three matmuls
        // rather than three per token.
        //
        // A shared expert can be a type the device has no shader for —
        // `engine::backend::is_cpu_only_tensor` exempts these from the
        // startup check on purpose — and `VulkanBackend` panics on that
        // rather than returning zeros. See `super::matmul_host_fallback`,
        // which this repeats inline because it also decides which backend
        // the `down` projection goes to.
        let shared_branch = || {
            let cpu = crate::engine::backend::CpuBackend;
            let use_cpu_shared = !backend.supports_type(ffn.gate_shexp.ggml_type())
                || !backend.supports_type(ffn.up_shexp.ggml_type())
                || !backend.supports_type(ffn.down_shexp.ggml_type());
            let (shexp_gate, shexp_up) = if use_cpu_shared {
                (
                    cpu.matmul(normed, n_tokens, &ffn.gate_shexp),
                    cpu.matmul(normed, n_tokens, &ffn.up_shexp),
                )
            } else {
                let mut gate_up = backend.matmul_batch(&[
                    MatmulOp {
                        x: normed,
                        n_tokens,
                        w: &ffn.gate_shexp,
                    },
                    MatmulOp {
                        x: normed,
                        n_tokens,
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
                cpu.matmul(&shexp_h, n_tokens, &ffn.down_shexp)
            } else {
                backend.matmul(&shexp_h, n_tokens, &ffn.down_shexp)
            };
            // The per-token sigmoid gate on the shared branch.
            for t in 0..n_tokens {
                let x_t = &normed[t * n_embd..(t + 1) * n_embd];
                let gate = tensor::sigmoid(tensor::dot(x_t, &ffn.gate_inp_shexp));
                for v in shexp_out[t * n_embd..(t + 1) * n_embd].iter_mut() {
                    *v *= gate;
                }
            }
            shexp_out
        };

        // Different processors, nothing shared but the input, summed at the
        // end — see `super::moe_overlap_min_tokens`.
        let (shexp_out, contribs) = if n_tokens >= super::moe_overlap_min_tokens() {
            rayon::join(shared_branch, routed_branch)
        } else {
            (shared_branch(), routed_branch())
        };
        experts.loaded_once_per_distinct_expert();

        for t in 0..n_tokens {
            let dst = &mut out[t * n_embd..(t + 1) * n_embd];
            dst.copy_from_slice(&shexp_out[t * n_embd..(t + 1) * n_embd]);
            for contrib in &contribs[t] {
                for (o, d) in dst.iter_mut().zip(contrib.iter()) {
                    *o += d;
                }
            }
        }
        experts.commit(n_tokens);
        out
    }
}

/// GQA attention restricted, per query, to a gathered set of cached
/// positions — the block-sparse counterpart of the dense window
/// `engine::attention` implements, used only by [`super::qwen4exp`]'s
/// indexer-selected layers.
///
/// `chosen[t]` is token `t`'s key set, ascending. It is a gather rather than
/// a mask because the sets are small next to the cache they are drawn from —
/// that is the entire point of the indexer — so materializing an `-inf` mask
/// the width of the cache would cost more than the attention it guards.
///
/// Softmax is taken over the gathered scores alone, which is what masking
/// everything else to `-inf` means; the caller has already excluded any
/// position past the query's own, so there is no separate causal step here.
fn attend_selected(
    q: &[f32],
    cache: &LayerCache,
    chosen: &[Vec<usize>],
    dims: &Dims,
    scale: f32,
) -> Vec<f32> {
    let head_dim = dims.head_dim;
    let n_head = dims.n_head;
    let n_head_kv = dims.n_head_kv;
    let group = n_head / n_head_kv;
    let n_tokens = chosen.len();

    // Over (token, head) pairs rather than over tokens: decode is one token,
    // and parallelizing the outer axis alone would leave every core but one
    // idle for exactly the call that runs most often.
    let mut out = vec![0f32; n_tokens * n_head * head_dim];
    out.par_chunks_mut(head_dim)
        .enumerate()
        .for_each(|(i, dst)| {
            let (t, h) = (i / n_head, i % n_head);
            let kv_head = h / group;
            let keys = &chosen[t];
            let q_h = &q[i * head_dim..(i + 1) * head_dim];
            let mut scores: Vec<f32> = keys
                .iter()
                .map(|&p| tensor::dot(q_h, cache.key_at(p, kv_head, head_dim)) * scale)
                .collect();
            tensor::softmax_inplace(&mut scores);
            for (&p, &w) in keys.iter().zip(scores.iter()) {
                tensor::axpy_inplace(dst, cache.value_at(p, kv_head, head_dim), w);
            }
        });
    out
}

enum Layer<F> {
    FullAttn(FullAttnWeights, F),
    Recurrent(RecurrentWeights, F),
}

/// The shared trunk itself, parameterized by the architecture's FFN.
pub(crate) struct Trunk<F> {
    pub config: ModelConfig,
    pub backend: Arc<dyn Backend>,
    tok_embeddings: QuantMatrix,
    output_norm: Vec<f32>,
    output_weight: QuantMatrix,
    dims: Dims,
    layers: Vec<Layer<F>>,
}

/// The buffers every layer of a forward pass reuses, so the trunk allocates
/// once per pass rather than twice per layer. Threaded through the per-layer
/// helpers rather than held in `self`, because `forward` takes `&self`.
#[derive(Default)]
struct LayerScratch {
    /// The normed residual stream feeding attention or the recurrent mixer —
    /// what used to be `x.to_vec()`. See `tensor::rmsnorm_into`.
    normed: Vec<f32>,
    /// The same, for the post-attention FFN norm.
    ffn: Vec<f32>,
    /// The feed-forward output, and the gate/up intermediates behind it —
    /// see `Backend::matmul_into`.
    ffn_out: Vec<f32>,
    ffn_work: super::FfnScratch,
}

impl<F: HybridFfn> Trunk<F> {
    /// Trunk layers loaded — `block_count` less `nextn_predict_layers`, since
    /// `load` stops before any multi-token-prediction block.
    pub(crate) fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Loads every trunk layer, calling `make_ffn(layer_index)` for each
    /// layer's FFN. Everything else — the layer kind, the attention or
    /// recurrent tensors, the embedding and output heads — is the same for
    /// all three architectures and is read here.
    pub(crate) fn load(
        loaded: &LoadedModel,
        backend: Arc<dyn Backend>,
        mut make_ffn: impl FnMut(usize) -> Result<F>,
    ) -> Result<Self> {
        let dims = Dims::from_loaded(loaded)?;
        let n_layer = trunk_layer_count(loaded)?;
        let is_recr = recurrent_layer_mask(loaded, n_layer);

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
            let t = LayerTensors { loaded, i };
            let ffn = make_ffn(i)?;
            if is_recr.get(i).copied().unwrap_or(false) {
                let cache_index = n_recurrent;
                n_recurrent += 1;
                layers.push(Layer::Recurrent(
                    RecurrentWeights::load(&t, &dims, cache_index)?,
                    ffn,
                ));
            } else {
                let cache_index = n_full_attn;
                n_full_attn += 1;
                layers.push(Layer::FullAttn(
                    FullAttnWeights::load(&t, cache_index)?,
                    ffn,
                ));
            }
        }

        Ok(Self {
            config: loaded.config.clone(),
            backend,
            tok_embeddings,
            output_norm,
            output_weight,
            dims,
            layers,
        })
    }

    /// `(n_full_attn, n_recurrent)` layer counts — used to size a fresh
    /// [`KvCache`].
    fn cache_layout(&self) -> (usize, usize) {
        let n_full_attn = self
            .layers
            .iter()
            .filter(|l| matches!(l, Layer::FullAttn(..)))
            .count();
        (n_full_attn, self.layers.len() - n_full_attn)
    }

    pub(crate) fn new_kv_cache(&self, capacity: usize) -> KvCache {
        let (n_full_attn, n_recurrent) = self.cache_layout();
        let kv_dims = vec![self.dims.n_head_kv * self.dims.head_dim; n_full_attn];
        let recurrent_specs = vec![
            RecurrentSpec::delta_net(
                self.dims.conv_channels(),
                self.dims.ssm_d_conv,
                self.dims.ssm_dt_rank,
                self.dims.ssm_head_dim,
            );
            n_recurrent
        ];
        KvCache::new_mixed(capacity, &kv_dims, &recurrent_specs)
    }

    pub(crate) fn forward(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<Vec<f32>> {
        let n_tokens = tokens.len();
        let n_embd = self.dims.n_embd;

        let mut x = vec![0f32; n_tokens * n_embd];
        for (t, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            anyhow::ensure!(
                tok < self.config.n_vocab,
                "token id {tok} is out of vocab range"
            );
            x[t * n_embd..(t + 1) * n_embd].copy_from_slice(&self.tok_embeddings.row(tok));
        }

        // Grown once and reused by every layer rather than allocated per
        // norm. Threaded through the per-layer helpers rather than held in
        // `self`, because `forward` takes `&self` and two layers of the same
        // model may run concurrently. See `tensor::rmsnorm_into`.
        let mut scratch = LayerScratch::default();

        for layer in &self.layers {
            match layer {
                Layer::FullAttn(weights, ffn) => {
                    self.forward_full_attn_layer(
                        weights,
                        ffn,
                        cache,
                        &mut x,
                        n_tokens,
                        start_pos,
                        &mut scratch,
                    )?;
                }
                Layer::Recurrent(weights, ffn) => {
                    self.forward_recurrent_layer(
                        weights,
                        ffn,
                        cache,
                        &mut x,
                        n_tokens,
                        &mut scratch,
                    )?;
                }
            }
        }

        let last = &mut x[(n_tokens - 1) * n_embd..].to_vec();
        tensor::rmsnorm_inplace(last, &self.output_norm, 1, n_embd, self.dims.rms_eps);
        Ok(self.backend.matmul(last, 1, &self.output_weight))
    }

    // Eight because the scratch buffers are threaded rather than held in
    // `self`; grouping them into `LayerScratch` already removed one.
    #[allow(clippy::too_many_arguments)]
    fn forward_full_attn_layer(
        &self,
        layer: &FullAttnWeights,
        ffn: &F,
        cache: &mut KvCache,
        x: &mut [f32],
        n_tokens: usize,
        start_pos: usize,
        scratch: &mut LayerScratch,
    ) -> Result<()> {
        let LayerScratch {
            normed,
            ffn: ffn_scratch,
            ffn_out,
            ffn_work,
        } = scratch;
        let n_embd = self.dims.n_embd;

        tensor::rmsnorm_into(
            normed,
            x,
            &layer.attn_norm,
            n_tokens,
            n_embd,
            self.dims.rms_eps,
        );
        let sub_out = layer.attn.forward(
            self.backend.as_ref(),
            &self.dims,
            cache,
            normed,
            n_tokens,
            start_pos,
            None,
        );

        tensor::add_inplace(x, &sub_out);
        self.apply_ffn(
            ffn,
            &layer.post_attention_norm,
            x,
            n_tokens,
            ffn_scratch,
            ffn_out,
            ffn_work,
        );
        Ok(())
    }

    fn forward_recurrent_layer(
        &self,
        layer: &RecurrentWeights,
        ffn: &F,
        cache: &mut KvCache,
        x: &mut [f32],
        n_tokens: usize,
        scratch: &mut LayerScratch,
    ) -> Result<()> {
        let LayerScratch {
            normed,
            ffn: ffn_scratch,
            ffn_out,
            ffn_work,
        } = scratch;
        let n_embd = self.dims.n_embd;

        tensor::rmsnorm_into(
            normed,
            x,
            &layer.attn_norm,
            n_tokens,
            n_embd,
            self.dims.rms_eps,
        );
        let sub_out = layer.recurrent.forward(
            self.backend.as_ref(),
            &self.dims,
            cache,
            normed,
            n_tokens,
            OutputGate::Silu,
        );

        tensor::add_inplace(x, &sub_out);
        self.apply_ffn(
            ffn,
            &layer.post_attention_norm,
            x,
            n_tokens,
            ffn_scratch,
            ffn_out,
            ffn_work,
        );
        Ok(())
    }

    /// The second half of a pre-norm block, identical for both layer kinds:
    /// norm the residual stream, run the architecture's FFN, add it back.
    // Eight because the scratch buffers are threaded rather than held in
    // `self`, the same reason `forward_full_attn_layer` carries this allow:
    // `forward` takes `&self` and two layers may be in flight at once.
    #[allow(clippy::too_many_arguments)]
    fn apply_ffn(
        &self,
        ffn: &F,
        post_attention_norm: &[f32],
        x: &mut [f32],
        n_tokens: usize,
        normed: &mut Vec<f32>,
        out: &mut Vec<f32>,
        work: &mut super::FfnScratch,
    ) {
        let n_embd = self.dims.n_embd;
        tensor::rmsnorm_into(
            normed,
            x,
            post_attention_norm,
            n_tokens,
            n_embd,
            self.dims.rms_eps,
        );
        ffn.forward_into(self.backend.as_ref(), out, work, n_embd, normed, n_tokens);
        tensor::add_inplace(x, out);
    }
}

impl Trunk<MoeFfn> {
    /// Loads a MoE trunk — `qwen35moe` and `qwen3next` differ only in their
    /// recurrent tensor names, which [`RecurrentWeights::load`] already
    /// absorbs, so both come through here.
    pub(crate) fn load_moe(loaded: &LoadedModel, backend: Arc<dyn Backend>) -> Result<Self> {
        loaded
            .metadata_u64("expert_count")
            .context("missing expert_count")?;
        let n_expert_used = loaded
            .metadata_u64("expert_used_count")
            .context("missing expert_used_count")? as usize;
        Self::load(loaded, backend, |i| {
            MoeFfn::load(&LayerTensors { loaded, i }, n_expert_used)
        })
    }
}

impl Trunk<DenseFfn> {
    pub(crate) fn load_dense(loaded: &LoadedModel, backend: Arc<dyn Backend>) -> Result<Self> {
        Self::load(loaded, backend, |i| {
            let t = LayerTensors { loaded, i };
            Ok(DenseFfn {
                gate: t.matrix("ffn_gate.weight")?,
                up: t.matrix("ffn_up.weight")?,
                down: t.matrix("ffn_down.weight")?,
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::backend::CpuBackend;
    use crate::engine::kv_cache::KvCache;

    fn test_dims(n_head: usize, n_head_kv: usize, head_dim: usize) -> Dims {
        Dims {
            n_embd: n_head * head_dim,
            n_head,
            n_head_kv,
            head_dim,
            rope_dim: head_dim,
            rope_freq_base: 10000.0,
            rms_eps: 1e-6,
            ssm_d_conv: 4,
            ssm_head_dim: 1,
            ssm_n_group: 1,
            ssm_dt_rank: 1,
        }
    }

    /// **The gathered path must be a specialization of the dense one, not a
    /// second opinion.** Given a selection that names every visible
    /// position, `attend_selected` has to produce exactly what
    /// `engine::attention` produces for the same query and cache — because
    /// that is the case `qwen4exp` deliberately *skips* the indexer for, and
    /// the two paths would otherwise disagree at the threshold where one
    /// hands over to the other.
    ///
    /// This is also the only cheap check on the gather: with a real
    /// indexer-selected subset there is nothing to compare against, so a
    /// grouped-query-attention head mapping mistake (a `kv_head` off by a
    /// group) would be invisible. Here `n_head_kv` is deliberately smaller
    /// than `n_head`, so that mapping is exercised rather than degenerate.
    #[test]
    fn a_full_selection_reproduces_the_dense_attention_exactly() {
        let (n_head, n_head_kv, head_dim) = (4usize, 2usize, 8usize);
        let dims = test_dims(n_head, n_head_kv, head_dim);
        let n_tokens = 3usize;
        let kv_dim = n_head_kv * head_dim;

        let mut cache = KvCache::new(1, 16, kv_dim);
        for p in 0..n_tokens {
            let k: Vec<f32> = (0..kv_dim)
                .map(|i| ((p * 31 + i * 7) % 13) as f32 * 0.1)
                .collect();
            let v: Vec<f32> = (0..kv_dim)
                .map(|i| ((p * 17 + i * 5) % 11) as f32 * 0.2)
                .collect();
            cache.layers[0].push(&k, &v);
        }
        let q: Vec<f32> = (0..n_tokens * n_head * head_dim)
            .map(|i| ((i * 3) % 9) as f32 * 0.15 - 0.5)
            .collect();
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut dense: Vec<f32> = Vec::new();
        crate::engine::attention::attention(
            &mut dense,
            &q,
            &mut cache.layers[0],
            &crate::engine::attention::Params {
                backend: &CpuBackend,
                device: 0,
                n_head,
                n_head_kv,
                head_dim,
                scale,
                causal: true,
                n_swa: 0,
                start_pos: 0,
                n_tokens,
            },
            |t| (0, t),
        );

        let selection: Vec<Vec<usize>> = (0..n_tokens).map(|t| (0..=t).collect()).collect();
        let gathered = attend_selected(&q, &cache.layers[0], &selection, &dims, scale);

        assert_eq!(gathered.len(), dense.len());
        for (i, (g, d)) in gathered.iter().zip(dense.iter()).enumerate() {
            assert!(
                (g - d).abs() < 1e-5,
                "element {i} differs: gathered {g}, dense {d}"
            );
        }
    }

    /// A selection that drops a position must actually drop it: the softmax
    /// is taken over the gathered scores alone, so excluding the *only*
    /// other key leaves the remaining one with all the weight and the
    /// output is that key's value verbatim.
    #[test]
    fn an_excluded_position_contributes_nothing() {
        let (n_head, n_head_kv, head_dim) = (1usize, 1usize, 4usize);
        let dims = test_dims(n_head, n_head_kv, head_dim);
        let mut cache = KvCache::new(1, 8, head_dim);
        cache.layers[0].push(&[1.0, 0.0, 0.0, 0.0], &[9.0, 9.0, 9.0, 9.0]);
        cache.layers[0].push(&[0.0, 1.0, 0.0, 0.0], &[1.0, 2.0, 3.0, 4.0]);

        let q = vec![1.0f32, 1.0, 0.0, 0.0];
        let out = attend_selected(&q, &cache.layers[0], &[vec![1]], &dims, 1.0);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    use super::split_beta_alpha;

    /// A packed `ssm_ba` row is grouped by K/V group, not by kind: within
    /// each group its `group` beta values come first, then its `group` alpha
    /// values. Reading it as "all beta then all alpha" produces a
    /// well-formed vector of the right length carrying the wrong numbers,
    /// which is invisible downstream — decay and gating both still run.
    #[test]
    fn packed_beta_alpha_deinterleaves_per_group() {
        // 2 K/V groups, 4 value heads => group = 2.
        // Row: [b0 b1 a0 a1 | b2 b3 a2 a3]
        let mixed = vec![0.0, 1.0, 10.0, 11.0, 2.0, 3.0, 12.0, 13.0];
        let (beta, alpha) = split_beta_alpha(&mixed, 1, 2, 4);
        assert_eq!(beta, vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(alpha, vec![10.0, 11.0, 12.0, 13.0]);
    }

    /// Two tokens, so the per-token stride is exercised rather than
    /// assumed — an off-by-one there reads token 0's alpha as token 1's
    /// beta.
    #[test]
    fn packed_beta_alpha_strides_per_token() {
        let mixed = vec![0.0, 5.0, 1.0, 6.0];
        let (beta, alpha) = split_beta_alpha(&mixed, 2, 1, 1);
        assert_eq!(beta, vec![0.0, 1.0]);
        assert_eq!(alpha, vec![5.0, 6.0]);
    }
}
