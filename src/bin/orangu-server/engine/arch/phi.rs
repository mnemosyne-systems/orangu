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

//! The `phi3` forward pass (Microsoft Phi-3 / Phi-4-mini) — a GQA + RoPE +
//! RMSNorm + SwiGLU transformer like `arch::llama`, but with four
//! differences that each silently corrupt output if guessed rather than read
//! off upstream, all four confirmed directly against `llama.cpp/src/models/
//! phi3.cpp` and the ggml kernels it calls:
//!
//! 1. **Fused QKV.** One `attn_qkv.weight` of `[n_embd, n_embd + 2*kv_dim]`
//!    rather than three separate projections, sliced Q-then-K-then-V
//!    (`llm_graph_context::build_qkv`'s own view offsets: `0`, `n_embd_q`,
//!    `n_embd_q + n_embd_kv`). Upstream still accepts a checkpoint with
//!    separate `attn_q`/`attn_k`/`attn_v` instead (`create_tensor_qkv` falls
//!    back when `wqkv` is absent), so this module does too.
//! 2. **Fused gate/up.** `ffn_up.weight` is `[n_embd, 2*n_ff]` and there is
//!    no `ffn_gate.weight` at all; `LLM_FFN_SWIGLU` splits the *result* in
//!    half. The activated half is the **first** one — ggml's
//!    `ggml_compute_forward_swiglu` with `swapped = 0` sets `src0_p` to the
//!    row start and `src1_p` to `+nc`, and `ggml_vec_swiglu_f32` computes
//!    `silu(x) * g` over those in that order. Swapping the halves produces
//!    fluent-looking but wrong text, not a crash.
//! 3. **Partial NEOX RoPE with LongRoPE frequency factors.** Only the
//!    leading `rope_dim` of each head rotates (96 of 128 for Phi-4-mini) and
//!    each pair's frequency is divided by an entry of `rope_factors_long` or
//!    `rope_factors_short` — see [`PhiModel::rope_freq_factors`] for which,
//!    and why.
//! 4. **A RoPE magnitude factor.** `phi3.rope.scaling.attn_factor`
//!    (1.1902381 for Phi-4-mini) scales cos/sin, i.e. lengthens every
//!    rotated pair — see `tensor::rope_apply_mscale_inplace`.
//!
//! Sliding-window attention is deliberately **not** implemented, matching
//! upstream: Phi-4-mini's GGUF does carry `phi3.attention.sliding_window =
//! 262144`, but `llama_model_phi3::load_arch_hparams` reacts to finding that
//! key by *disabling* SWA outright (`swa_type = LLAMA_SWA_TYPE_NONE; n_swa =
//! 0`), warning that the conversion scripts populate it incorrectly
//! (ggml-org/llama.cpp#13676). Attention here is plain causal, as upstream's
//! is.
//!
//! Weight matrices stay `mmap`-backed and are dequantized one row at a time
//! via `QuantMatrix`, exactly as in `arch::llama`; only the small per-element
//! norm tensors are eagerly dequantized.
//!
//! **The chained decode path.** [`PhiModel::record_decode_forward`] records a
//! whole decode step as one GPU submission, the same shape `arch::llama` and
//! `arch::mistral` use. None of the four differences above stops it:
//!
//! - (3) and (4) are *parameters* of the fused chain — it takes `rope_dim` and
//!   `freq_factors`, and carries a RoPE magnitude scale via `vulkan::RopeYarn`,
//!   which `attn_factor` reaches as the `ext_factor == 0` case.
//! - (1) and (2) are one blocker twice, and it is a **naming** problem rather
//!   than a data one: the chain wants `wq`/`wk`/`wv` and `ffn_gate`/`ffn_up`
//!   separately, and this architecture stores each group concatenated. The
//!   bytes are already laid out correctly — rows are fixed-size and
//!   self-contained — so `QuantMatrix::rows` names the halves without copying
//!   any of them. See [`PhiLayer::qkv_views`] and [`PhiLayer::ffn_gate_up`].
//!
//! What that required first was widening `QuantMatrix::cache_key`, which was
//! `(mmap_ptr, start)` and so could not distinguish a view of the leading rows
//! from the whole tensor — every backend cache addressed by it would have
//! handed the view the parent's buffers.

use anyhow::{Context, Result};
use std::sync::Arc;

use super::ModelForward;
use crate::engine::backend::{Backend, MatmulOp};
use crate::engine::kv_cache::KvCache;
use crate::engine::loader::{LoadedModel, ModelConfig, QuantMatrix};
use crate::engine::tensor;

/// One layer's attention projections: either the fused `attn_qkv.weight`
/// upstream prefers, or the three separate ones it falls back to.
enum QkvProjection {
    /// `attn_qkv.weight`, `[n_embd, n_embd + 2*kv_dim]`.
    Fused(QuantMatrix),
    /// `attn_q.weight` / `attn_k.weight` / `attn_v.weight`.
    Split {
        wq: QuantMatrix,
        wk: QuantMatrix,
        wv: QuantMatrix,
    },
}

struct PhiLayer {
    attn_norm: Vec<f32>,
    qkv: QkvProjection,
    wo: QuantMatrix,
    ffn_norm: Vec<f32>,
    /// `ffn_up.weight`, `[n_embd, 2 * n_ff]` — gate and up concatenated;
    /// there is no separate `ffn_gate.weight` for this architecture.
    w_up: QuantMatrix,
    w_down: QuantMatrix,
}

impl PhiLayer {
    /// Q, K and V as three separate matrices, whichever way the checkpoint
    /// stored them — row-range views over the fused `attn_qkv.weight`, or the
    /// already-separate tensors.
    ///
    /// The offsets are upstream's (`build_qkv`'s three `ggml_view_3d` calls):
    /// Q at `0`, K at `n_embd`, V at `n_embd + kv_dim`. Applying them to the
    /// *weight* rather than to the projection's output is what lets prefill and
    /// decode share one set of matrices, and drops a whole-batch CPU copy.
    fn qkv_views(&self, n_embd: usize, kv_dim: usize) -> (QuantMatrix, QuantMatrix, QuantMatrix) {
        match &self.qkv {
            QkvProjection::Fused(wqkv) => (
                wqkv.rows(0, n_embd),
                wqkv.rows(n_embd, kv_dim),
                wqkv.rows(n_embd + kv_dim, kv_dim),
            ),
            QkvProjection::Split { wq, wk, wv } => (wq.clone(), wk.clone(), wv.clone()),
        }
    }

    /// The gate and up halves of `ffn_up.weight` as separate matrices.
    ///
    /// **Gate is the first half.** ggml's `ggml_vec_swiglu_f32` with
    /// `swapped = 0` activates the row start and multiplies by `+nc`; swapping
    /// these produces fluent-looking but wrong text, not a crash.
    fn ffn_gate_up(&self) -> (QuantMatrix, QuantMatrix) {
        let n_ff = self.w_up.out_dim / 2;
        (self.w_up.rows(0, n_ff), self.w_up.rows(n_ff, n_ff))
    }
}

pub struct PhiModel {
    config: ModelConfig,
    backend: Arc<dyn Backend>,
    tok_embeddings: QuantMatrix,
    output_norm: Vec<f32>,
    output_weight: QuantMatrix,
    layers: Vec<PhiLayer>,
    /// The chosen LongRoPE `[rope_dim / 2]` per-pair frequency divisor, or
    /// `None` for a `phi3` checkpoint that ships neither factor tensor.
    rope_freq_factors: Option<Vec<f32>>,
    /// `phi3.rope.scaling.attn_factor`, defaulting to `1.0` (a no-op) when
    /// the file doesn't set it — upstream's own `hparams.rope_attn_factor`
    /// default.
    rope_attn_factor: f32,
}

impl PhiModel {
    pub fn load_with_backend(loaded: &LoadedModel, backend: Arc<dyn Backend>) -> Result<Self> {
        let config = loaded.config.clone();
        let tok_embeddings = loaded
            .matrix("token_embd.weight")
            .context("loading token_embd.weight")?;
        let (output_norm, _) = loaded
            .tensor("output_norm.weight")
            .context("loading output_norm.weight")?;
        // Phi-4-mini ties its output projection to the input embedding and
        // ships no `output.weight` at all — upstream's own
        // `TENSOR_NOT_REQUIRED` + `TENSOR_DUPLICATED` fallback.
        let output_weight = if loaded.has_tensor("output.weight") {
            loaded
                .matrix("output.weight")
                .context("loading output.weight")?
        } else {
            tok_embeddings.clone()
        };

        let rope_freq_factors = Self::rope_freq_factors(loaded)?;
        if let Some(factors) = &rope_freq_factors {
            anyhow::ensure!(
                factors.len() >= config.rope_dim / 2,
                "rope factor tensor has {} entries, need {} for rope.dimension_count = {}",
                factors.len(),
                config.rope_dim / 2,
                config.rope_dim,
            );
        }
        let rope_attn_factor = loaded
            .metadata_f32("rope.scaling.attn_factor")
            .unwrap_or(1.0);

        let mut layers = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
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
            // Fused when present, three separate projections otherwise —
            // upstream's `create_tensor_qkv` makes the same choice on the
            // same condition.
            let qkv = if loaded.has_tensor(&format!("blk.{i}.attn_qkv.weight")) {
                QkvProjection::Fused(get_matrix("attn_qkv.weight")?)
            } else {
                QkvProjection::Split {
                    wq: get_matrix("attn_q.weight")?,
                    wk: get_matrix("attn_k.weight")?,
                    wv: get_matrix("attn_v.weight")?,
                }
            };
            layers.push(PhiLayer {
                attn_norm: get("attn_norm.weight")?,
                qkv,
                wo: get_matrix("attn_output.weight")?,
                ffn_norm: get("ffn_norm.weight")?,
                w_up: get_matrix("ffn_up.weight")?,
                w_down: get_matrix("ffn_down.weight")?,
            });
        }

        let model = Self {
            config,
            backend,
            tok_embeddings,
            output_norm,
            output_weight,
            layers,
            rope_freq_factors,
            rope_attn_factor,
        };
        model.validate_shapes()?;
        Ok(model)
    }

    /// Which of `rope_factors_long` / `rope_factors_short` this model rotates
    /// with — LongRoPE ships both and picks one.
    ///
    /// Upstream's rule (`llama_model::get_rope_factors`) is a property of the
    /// *serving* context, not of the request: `n_ctx_seq >
    /// hparams.n_ctx_orig_yarn ? long : short`, decided once at context
    /// creation so that every key already in the KV cache was rotated the
    /// same way as the query now reading it. This engine has no separate
    /// context-length knob — `engine::generate` caps a sequence at the
    /// model's own `n_ctx_train` — so `n_ctx_train` *is* this build's
    /// `n_ctx_seq`, and the comparison is made against it. For Phi-4-mini
    /// (131072 trained, 4096 original) that selects the long factors, which
    /// is also what `llama-server -c 0` (or any `-c` above 4096) selects;
    /// note upstream's own CLI default of `-c 4096` would instead select the
    /// short ones, so a logit-level comparison against llama.cpp has to pass
    /// a matching `-c`.
    ///
    /// `n_ctx_orig_yarn` itself defaults to `n_ctx_train` when the file omits
    /// `rope.scaling.original_context_length` (upstream sets it that way
    /// before the optional `get_key`), which makes the comparison false and
    /// selects the short factors — the right answer for a checkpoint that
    /// never declared a shorter original context.
    fn rope_freq_factors(loaded: &LoadedModel) -> Result<Option<Vec<f32>>> {
        let n_ctx_orig = loaded
            .metadata_u64("rope.scaling.original_context_length")
            .map(|v| v as usize)
            .unwrap_or(loaded.config.n_ctx_train);
        let name = if loaded.config.n_ctx_train > n_ctx_orig {
            "rope_factors_long.weight"
        } else {
            "rope_factors_short.weight"
        };
        if !loaded.has_tensor(name) {
            return Ok(None);
        }
        Ok(Some(
            loaded
                .tensor(name)
                .with_context(|| format!("loading {name}"))?
                .0,
        ))
    }

    fn head_dim(&self) -> usize {
        self.config.n_embd / self.config.n_head
    }

    fn kv_dim(&self) -> usize {
        self.config.n_head_kv * self.head_dim()
    }

    /// Rejects, at load time, a checkpoint whose tensor shapes don't match
    /// the hyperparameters — the fused QKV and fused gate/up splits are both
    /// computed from `n_embd`/`n_head_kv`/`out_dim` rather than read, so a
    /// mismatch would otherwise slice at the wrong offsets and produce
    /// confidently wrong tokens instead of an error.
    fn validate_shapes(&self) -> Result<()> {
        let n_embd = self.config.n_embd;
        let qkv_dim = n_embd + 2 * self.kv_dim();
        for (i, layer) in self.layers.iter().enumerate() {
            match &layer.qkv {
                QkvProjection::Fused(wqkv) => anyhow::ensure!(
                    wqkv.in_dim == n_embd && wqkv.out_dim == qkv_dim,
                    "blk.{i}.attn_qkv.weight is [{}, {}], expected [{n_embd}, {qkv_dim}]",
                    wqkv.in_dim,
                    wqkv.out_dim,
                ),
                QkvProjection::Split { wq, wk, wv } => {
                    anyhow::ensure!(
                        wq.out_dim == n_embd
                            && wk.out_dim == self.kv_dim()
                            && wv.out_dim == self.kv_dim(),
                        "blk.{i} attn_q/attn_k/attn_v produce [{}, {}, {}], expected [{n_embd}, {}, {}]",
                        wq.out_dim,
                        wk.out_dim,
                        wv.out_dim,
                        self.kv_dim(),
                        self.kv_dim(),
                    );
                }
            }
            anyhow::ensure!(
                layer.w_up.out_dim % 2 == 0,
                "blk.{i}.ffn_up.weight has an odd output width ({}) — this architecture's \
                 SwiGLU splits it into equal gate and up halves",
                layer.w_up.out_dim,
            );
            anyhow::ensure!(
                layer.w_up.out_dim / 2 == layer.w_down.in_dim,
                "blk.{i}.ffn_up.weight's half-width {} doesn't match ffn_down.weight's input {}",
                layer.w_up.out_dim / 2,
                layer.w_down.in_dim,
            );
        }
        Ok(())
    }

    /// Runs every transformer layer and returns the pre-final-norm hidden
    /// state for every token (`[n_tokens, n_embd]`) — shared by next-token
    /// prediction and pooled embeddings, as in `arch::llama`.
    /// This architecture's RoPE as the fused chain's terms.
    ///
    /// `phi3` sets no YaRN ramp — `ext_factor` stays 0, which disables the
    /// interpolation band — but it does set `rope.scaling.attn_factor`, a
    /// magnitude scale on cos/sin. That is the `ext_factor == 0` case of the
    /// same `RopeYarn` the chain gained for `arch::mistral`, so it is built
    /// through the shared `RopeParams` rather than assembled by hand.
    fn rope_yarn(&self) -> crate::engine::backend::vulkan::RopeYarn {
        crate::engine::backend::vulkan::RopeYarn::from_params(&tensor::RopeParams {
            rope_dim: self.config.rope_dim,
            freq_base: self.config.rope_freq_base,
            attn_factor: self.rope_attn_factor,
            layout: tensor::RopeLayout::Neox,
            ..tensor::RopeParams::default()
        })
    }

    /// One decode step as a single GPU submission, or `None` when this model or
    /// this step is not one the fused chain can describe.
    ///
    /// The two tensors this architecture concatenates — `attn_qkv.weight` and
    /// `ffn_up.weight` — reach the chain as row-range views
    /// ([`PhiLayer::qkv_views`], [`PhiLayer::ffn_gate_up`]), so nothing is
    /// copied and the chain sees the five separate matrices it expects.
    /// The whole decode step recorded into one encoder, **not submitted** —
    /// the caller decides what else joins the submission.
    ///
    /// Split out from [`Self::record_decode_forward`] so
    /// [`Self::forward_maybe_sampling`] can append the GPU argmax to this same
    /// encoder. Reading the `[n_vocab]` logits back to sample on the CPU is a
    /// second round trip and, for a 128k-vocab model, half a megabyte of
    /// transfer per token.
    ///
    /// Returns the encoder plus the logits buffer and its **byte** offset.
    fn record_decode_chain(
        &self,
        vulkan: &crate::engine::backend::VulkanBackend,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        slot_id: usize,
    ) -> Option<(wgpu::CommandEncoder, wgpu::Buffer, u64)> {
        if tokens.len() != 1 {
            return None;
        }
        let tok = tokens[0] as usize;
        if tok >= self.config.n_vocab {
            return None;
        }
        let x0 = self.tok_embeddings.row(tok).to_vec();
        self.record_decode_run(
            vulkan,
            cache,
            0..self.layers.len(),
            &x0,
            start_pos,
            slot_id,
            true,
        )
    }

    /// One *run* of the decode chain — see `LlamaModel::record_decode_run`,
    /// which this mirrors: layers `layers`, starting from the host vector
    /// `x_in`, recorded into one encoder on one device. `with_tail` appends
    /// `output_norm` and the vocab projection.
    #[allow(clippy::too_many_arguments)]
    fn record_decode_run(
        &self,
        vulkan: &crate::engine::backend::VulkanBackend,
        cache: &mut KvCache,
        layers: std::ops::Range<usize>,
        x_in: &[f32],
        start_pos: usize,
        slot_id: usize,
        with_tail: bool,
    ) -> Option<(wgpu::CommandEncoder, wgpu::Buffer, u64)> {
        use crate::engine::backend::vulkan::{
            FfnActivation, FusedAttnProjection, FusedLayerInput, GpuInput,
        };

        if crate::engine::arch::llama::no_fused_qkv()
            || crate::engine::arch::llama::no_fused_post_attention()
        {
            return None;
        }
        if !vulkan.prefill_attention_enabled() {
            return None;
        }
        let cfg = &self.config;
        let n_embd = cfg.n_embd;
        let head_dim = self.head_dim();
        let kv_dim = self.kv_dim();
        let yarn = self.rope_yarn();

        let mut encoder = vulkan.new_encoder("orangu-server phi decode");
        // Per-stage GPU timing for this step, when `ORANGU_GPU_TIMESTAMPS=1`
        // and the adapter has the query; inert otherwise. See
        // `VulkanBackend::begin_step_timestamps` for why the slot arithmetic
        // lives there rather than here.
        let n_layer = self.layers.len();
        let ts = vulkan.begin_step_timestamps(&mut encoder, n_layer);
        // The views must outlive the recording that borrows them, so they are
        // built into a vector alongside the output buffers rather than inside
        // the loop body.
        let views: Vec<_> = self
            .layers
            .iter()
            .map(|layer| (layer.qkv_views(n_embd, kv_dim), layer.ffn_gate_up()))
            .collect();
        let mut bufs: Vec<(wgpu::Buffer, u64)> = Vec::with_capacity(self.layers.len());
        for il in layers.clone() {
            let layer = &self.layers[il];
            let ((wq, wk, wv), (ffn_gate, ffn_up)) = &views[il];
            let x_input = match bufs.last() {
                Some((buf, offset)) => GpuInput::Gpu(buf, (*offset / 4) as usize),
                None => GpuInput::Cpu(x_in),
            };
            let out = vulkan.record_fused_layer(
                &mut encoder,
                FusedLayerInput {
                    x: x_input,
                    // Partial NEOX RoPE: only the leading `rope_dim` of each
                    // head rotates, which the chain takes as a parameter.
                    pairing: tensor::RopeLayout::Neox,
                    yarn,
                    // SwiGLU, no per-head Q/K norms, no post-norms, no
                    // projection biases, and V is not normalized.
                    activation: FfnActivation::Swiglu,
                    normalize_v: false,
                    attn_norm: &layer.attn_norm,
                    wq,
                    q_bias: None,
                    q_norm: None,
                    kv: Some(FusedAttnProjection {
                        wk,
                        wv: Some(wv),
                        k_bias: None,
                        v_bias: None,
                        k_norm: None,
                    }),
                    n_head: cfg.n_head,
                    n_head_kv: cfg.n_head_kv,
                    head_dim,
                    rope_dim: cfg.rope_dim,
                    rope_freq_base: cfg.rope_freq_base,
                    // LongRoPE's per-pair divisor, chosen once at load time.
                    freq_factors: self.rope_freq_factors.as_deref(),
                    eps: cfg.rms_eps,
                    pos: start_pos,
                    // Plain causal attention — upstream disables SWA for this
                    // architecture, and so does this module.
                    window_start: 0,
                    window: None,
                    scale: 1.0 / (head_dim as f32).sqrt(),
                    cache: &mut cache.layers[il],
                    wo: &layer.wo,
                    attn_post_norm: None,
                    ffn_norm: &layer.ffn_norm,
                    ffn_gate,
                    ffn_up,
                    ffn_down: &layer.w_down,
                    ffn_post_norm: None,
                    ple: None,
                    layer_output_scale: None,
                    batch_slot: slot_id,
                    attn_ts: ts.attn_slot(il, n_layer),
                },
            );
            ts.after_layer(&mut encoder, il);
            bufs.push(out);
        }

        let (last_buf, last_offset) = bufs.last()?;
        if !with_tail {
            let (buf, offset) = (last_buf.clone(), *last_offset);
            ts.finish(vulkan, &mut encoder, n_layer);
            return Some((encoder, buf, offset));
        }
        let normed = vulkan.record_output_norm(
            &mut encoder,
            GpuInput::Gpu(last_buf, (*last_offset / 4) as usize),
            &self.output_norm,
            cfg.rms_eps,
            n_embd,
        );
        // `slot_id + 1`: op resources are keyed by `(weight, batch_slot)`, and
        // the vocab projection must not share a slot with the layer chain.
        let (logits_buf, logits_offset) = vulkan.record_full_matmul(
            &mut encoder,
            GpuInput::Gpu(&normed, 0),
            &self.output_weight,
            slot_id + 1,
        );
        ts.finish(vulkan, &mut encoder, n_layer);
        Some((encoder, logits_buf, logits_offset))
    }

    /// A decode step as one GPU submission, returning the full `[n_vocab]`
    /// logits — the path taken when the caller is not greedy-sampling. See
    /// [`Self::forward_maybe_sampling`] for the one that is.
    fn record_decode_forward(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        slot_id: usize,
    ) -> Option<Vec<f32>> {
        let Some(vulkan) = self.backend.as_wgpu() else {
            // No single device holds the whole model: either there is no GPU
            // at all, or the model is split.
            return self.record_split_decode(cache, tokens, start_pos, slot_id);
        };
        let (encoder, _, _) =
            self.record_decode_chain(vulkan, cache, tokens, start_pos, slot_id)?;
        let logits = vulkan.submit_and_readback_for(encoder, &self.output_weight, slot_id + 1);
        if vulkan.gpu_timestamps() {
            vulkan.report_timestamps(start_pos, self.layers.len());
        }
        Some(logits)
    }

    /// The fused per-layer decode chain on a split model — see
    /// `LlamaModel::record_split_decode`, which this mirrors: one encoder
    /// per run of consecutive layers sharing a device, with the hidden
    /// state crossing to host memory in between.
    fn record_split_decode(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        slot_id: usize,
    ) -> Option<Vec<f32>> {
        if tokens.len() != 1 {
            return None;
        }
        let tok = tokens[0] as usize;
        if tok >= self.config.n_vocab {
            return None;
        }
        let runs = super::decode_device_runs(
            self.backend.as_ref(),
            self.layers.iter().map(|layer| layer.wo.device()),
        )?;
        if runs.len() < 2 {
            return None;
        }
        let tail_device = self.output_weight.device();

        let mut x = self.tok_embeddings.row(tok).to_vec();
        for (index, (device, layers)) in runs.iter().enumerate() {
            let vulkan = self.backend.as_wgpu_on(*device)?;
            let with_tail = index + 1 == runs.len() && *device == tail_device;
            let (encoder, buf, offset) = self.record_decode_run(
                vulkan,
                cache,
                layers.clone(),
                &x,
                start_pos,
                slot_id,
                with_tail,
            )?;
            if with_tail {
                return Some(vulkan.submit_and_readback_for(
                    encoder,
                    &self.output_weight,
                    slot_id + 1,
                ));
            }
            x = vulkan.submit_and_read_at(encoder, &buf, offset, self.config.n_embd);
        }

        // The last layers were not on the vocab projection's device, so the
        // tail is a run of its own.
        let vulkan = self.backend.as_wgpu_on(tail_device)?;
        let mut encoder = vulkan.new_encoder("orangu-server phi decode tail");
        let normed = vulkan.record_output_norm(
            &mut encoder,
            crate::engine::backend::vulkan::GpuInput::Cpu(&x),
            &self.output_norm,
            self.config.rms_eps,
            self.config.n_embd,
        );
        vulkan.record_full_matmul(
            &mut encoder,
            crate::engine::backend::vulkan::GpuInput::Gpu(&normed, 0),
            &self.output_weight,
            slot_id + 1,
        );
        Some(vulkan.submit_and_readback_for(encoder, &self.output_weight, slot_id + 1))
    }

    fn run_layers(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let n_tokens = tokens.len();
        let n_embd = cfg.n_embd;
        let head_dim = self.head_dim();
        let n_head = cfg.n_head;
        let n_head_kv = cfg.n_head_kv;
        let kv_dim = self.kv_dim();

        let mut x = vec![0f32; n_tokens * n_embd];
        for (t, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            anyhow::ensure!(tok < cfg.n_vocab, "token id {tok} is out of vocab range");
            x[t * n_embd..(t + 1) * n_embd].copy_from_slice(&self.tok_embeddings.row(tok));
        }

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let mut normed = x.clone();
            tensor::rmsnorm_inplace(&mut normed, &layer.attn_norm, n_tokens, n_embd, cfg.rms_eps);

            // Q/K/V as three matrices whichever way the checkpoint stored
            // them, then one batched dispatch — independent given the same
            // normed input — rather than three round trips, matching
            // `arch::llama`.
            //
            // For a fused checkpoint these are row *views*, so this both drops
            // `split_qkv`'s whole-batch CPU copy and — the reason it matters
            // more — keeps prefill and decode addressing the **same** weight
            // bytes. Uploading the fused tensor here and its views in the
            // decode chain would leave the GPU holding two copies of the
            // largest weights in the model.
            let (wq, wk, wv) = layer.qkv_views(n_embd, kv_dim);

            // Deliberately *not* `fused_attention_prefill`, which this shape is
            // eligible for and which `arch::llama` does use: measured neutral
            // here (−0.6% on Phi-4-mini) and a clear loss on the sibling this
            // was ported alongside. See PERF-GAP.md item 10.
            let (mut q, mut k, v) = {
                let mut out = self.backend.matmul_batch(&[
                    MatmulOp {
                        x: &normed,
                        n_tokens,
                        w: &wq,
                    },
                    MatmulOp {
                        x: &normed,
                        n_tokens,
                        w: &wk,
                    },
                    MatmulOp {
                        x: &normed,
                        n_tokens,
                        w: &wv,
                    },
                ]);
                let v = out.pop().unwrap();
                let k = out.pop().unwrap();
                let q = out.pop().unwrap();
                (q, k, v)
            };

            let layer_cache = &mut cache.layers[layer_idx];
            for t in 0..n_tokens {
                let pos = start_pos + t;
                tensor::rope_apply_mscale_inplace(
                    &mut q[t * n_head * head_dim..(t + 1) * n_head * head_dim],
                    n_head,
                    head_dim,
                    cfg.rope_dim,
                    pos,
                    cfg.rope_freq_base,
                    self.rope_freq_factors.as_deref(),
                    self.rope_attn_factor,
                );
                tensor::rope_apply_mscale_inplace(
                    &mut k[t * kv_dim..(t + 1) * kv_dim],
                    n_head_kv,
                    head_dim,
                    cfg.rope_dim,
                    pos,
                    cfg.rope_freq_base,
                    self.rope_freq_factors.as_deref(),
                    self.rope_attn_factor,
                );
                layer_cache.push(
                    &k[t * kv_dim..(t + 1) * kv_dim],
                    &v[t * kv_dim..(t + 1) * kv_dim],
                );
            }

            // Plain causal attention -- no sliding window, matching
            // upstream's own disabling of it for this architecture.
            let mut attn_out: Vec<f32> = Vec::new();
            crate::engine::attention::attention(
                &mut attn_out,
                &q,
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

            // One submission for `wo` → residual → FFN norm → gate/up → SwiGLU
            // → `down` → residual, instead of three blocking round trips.
            //
            // The chain wants gate and up as separate matrices, which is what
            // `ffn_gate_up`'s row views name — the same halves the `else`
            // branch below splits out of the *result*, and the sequence that
            // branch runs is what the chain is cross-checked against.
            let (w_gate, w_up_half) = layer.ffn_gate_up();
            let fused = self
                .backend
                // This layer's card — see `Backend::as_wgpu_on`.
                .as_wgpu_on(layer.wo.device())
                .filter(|_| !crate::engine::arch::llama::no_fused_post_attention())
                .and_then(|vulkan| {
                    vulkan.fused_post_attention_prefill(
                        crate::engine::backend::vulkan::AttnOutSrc::Host(&attn_out),
                        &x,
                        n_tokens,
                        &layer.wo,
                        None,
                        &layer.ffn_norm,
                        &w_gate,
                        &w_up_half,
                        &layer.w_down,
                        None,
                        cfg.rms_eps,
                        crate::engine::backend::vulkan::FfnActivation::Swiglu,
                    )
                });
            if let Some(out) = fused {
                x = out;
                continue;
            }

            let attn_proj = self.backend.matmul(&attn_out, n_tokens, &layer.wo);
            tensor::add_inplace(&mut x, &attn_proj);

            let mut normed2 = x.clone();
            tensor::rmsnorm_inplace(&mut normed2, &layer.ffn_norm, n_tokens, n_embd, cfg.rms_eps);
            // One `[n_embd, 2*n_ff]` projection, then SwiGLU over the two
            // halves of each row: `silu(first) * second`.
            let gate_up = self.backend.matmul(&normed2, n_tokens, &layer.w_up);
            let n_ff = layer.w_up.out_dim / 2;
            let mut activated = vec![0f32; n_tokens * n_ff];
            for t in 0..n_tokens {
                let row = &gate_up[t * 2 * n_ff..(t + 1) * 2 * n_ff];
                let (gate, up) = row.split_at(n_ff);
                let out = &mut activated[t * n_ff..(t + 1) * n_ff];
                for ((o, g), u) in out.iter_mut().zip(gate).zip(up) {
                    *o = tensor::silu(*g) * u;
                }
            }
            let down = self.backend.matmul(&activated, n_tokens, &layer.w_down);
            tensor::add_inplace(&mut x, &down);
        }

        Ok(x)
    }
}

impl ModelForward for PhiModel {
    /// A greedy decode step that never transfers the logits.
    ///
    /// The default implementation returns `[n_vocab]` logits for the caller to
    /// sample on the CPU. That is a second round trip on every token plus, for
    /// this family's larger vocabularies, half a megabyte of transfer — and it
    /// was measured at **+17.7% throughput** on the one architecture that
    /// already avoided it, once the quantized decode kernels stopped dominating
    /// the step. Here the argmax joins the same encoder as the forward, and a
    /// single `u32` comes back.
    ///
    /// Falls through to the logits path whenever the fast path does not apply —
    /// no `wgpu` backend, not greedy, more than one token, or `gpu_sample`
    /// turned off — so behaviour is unchanged wherever it cannot help.
    fn forward_maybe_sampling(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        greedy_sample: Option<super::GreedySampleParams<'_>>,
        slot_id: usize,
    ) -> Result<super::ForwardOutcome> {
        if tokens.len() == 1
            && let Some(params) = &greedy_sample
            && let Some(vulkan) = self.backend.as_wgpu()
            && vulkan.gpu_sample()
            && let Some((mut encoder, logits_buf, logits_offset)) =
                self.record_decode_chain(vulkan, cache, tokens, start_pos, slot_id)
        {
            let sample_buf = vulkan.record_argmax_sample(
                &mut encoder,
                crate::engine::backend::vulkan::GpuArgmaxSampleInput {
                    // `GpuInput::Gpu`'s offset is in elements; the arena aligns
                    // every output to at least 4 bytes, so this divides evenly.
                    logits: crate::engine::backend::vulkan::GpuInput::Gpu(
                        &logits_buf,
                        (logits_offset / 4) as usize,
                    ),
                    n_vocab: self.output_weight.out_dim,
                    recent_tokens: params.recent_tokens,
                    repeat_penalty: params.repeat_penalty,
                    // This family has no final-logit softcap.
                    logit_softcap: None,
                },
                // Per-slot, so two concurrently-decoding sequences never share
                // the cached sample scratch — same reason the op cache keys on
                // `slot_id + 1` just above.
                slot_id + 1,
            );
            let next = vulkan.submit_and_readback_u32(encoder, &sample_buf);
            if vulkan.gpu_timestamps() {
                vulkan.report_timestamps(start_pos, self.layers.len());
            }
            return Ok(super::ForwardOutcome::Token(next));
        }
        self.forward(cache, tokens, start_pos, slot_id)
            .map(super::ForwardOutcome::Logits)
    }

    fn vulkan_backend(&self) -> Option<&crate::engine::backend::vulkan::VulkanBackend> {
        self.backend.as_wgpu()
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn new_kv_cache(&self, capacity: usize) -> KvCache {
        KvCache::new(self.config.n_layer, capacity, self.kv_dim())
    }

    fn forward(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        slot_id: usize,
    ) -> Result<Vec<f32>> {
        // A single-token step the fused chain can describe goes through
        // `record_decode_forward`; `None` falls through to the path below.
        if let Some(logits) = self.record_decode_forward(cache, tokens, start_pos, slot_id) {
            return Ok(logits);
        }
        let cfg = &self.config;
        let n_tokens = tokens.len();
        let n_embd = cfg.n_embd;
        let x = self.run_layers(cache, tokens, start_pos)?;

        let last = &mut x[(n_tokens - 1) * n_embd..].to_vec();
        tensor::rmsnorm_inplace(last, &self.output_norm, 1, n_embd, cfg.rms_eps);
        let logits = self.backend.matmul(last, 1, &self.output_weight);
        Ok(logits)
    }

    fn forward_hidden_states(&self, tokens: &[u32]) -> Result<Vec<f32>> {
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

    /// The fused-QKV slice offsets, against a hand-built projection whose
    /// every element names its own (token, section, index) — so a swapped
    /// K/V, an off-by-`kv_dim` offset, or a row-vs-column transposition each
    /// produce a visibly wrong value rather than a plausible one.
    /// The offsets `qkv_views` cuts at are upstream's (`build_qkv`'s three
    /// `ggml_view_3d` calls): Q first at `0`, then K at `n_embd`, then V at
    /// `n_embd + kv_dim`. Getting them wrong slices every head at the wrong
    /// place and produces confidently wrong text, not an error.
    #[test]
    fn qkv_views_slice_q_then_k_then_v() {
        use crate::engine::loader::test_quant_matrix;
        let (n_embd, kv_dim, in_dim) = (4usize, 2usize, 64usize);
        let out_dim = n_embd + 2 * kv_dim;
        // Q8_0: 32 weights per block, 2 scale bytes + 32 value bytes. Row `r`
        // is filled with the byte `r + 1`, so a view's first row names itself.
        let row_bytes = in_dim / 32 * 34;
        let mut bytes = vec![0u8; row_bytes * out_dim];
        for r in 0..out_dim {
            for b in &mut bytes[r * row_bytes..(r + 1) * row_bytes] {
                *b = (r + 1) as u8;
            }
        }
        let layer = PhiLayer {
            attn_norm: Vec::new(),
            qkv: QkvProjection::Fused(test_quant_matrix(&bytes, 8, in_dim, out_dim)),
            wo: test_quant_matrix(&bytes, 8, in_dim, out_dim),
            ffn_norm: Vec::new(),
            w_up: test_quant_matrix(&bytes, 8, in_dim, out_dim),
            w_down: test_quant_matrix(&bytes, 8, in_dim, out_dim),
        };

        let (wq, wk, wv) = layer.qkv_views(n_embd, kv_dim);
        assert_eq!(
            (wq.out_dim, wk.out_dim, wv.out_dim),
            (n_embd, kv_dim, kv_dim)
        );
        // Each view's first row is the row at its offset in the fused tensor.
        assert_eq!(wq.raw_bytes()[0], 1);
        assert_eq!(wk.raw_bytes()[0], (n_embd + 1) as u8);
        assert_eq!(wv.raw_bytes()[0], (n_embd + kv_dim + 1) as u8);
        // And the three name disjoint byte ranges, so no backend cache can
        // confuse them for each other or for the tensor they came from.
        let keys = [wq.cache_key(), wk.cache_key(), wv.cache_key()];
        assert_eq!(
            keys.iter().collect::<std::collections::HashSet<_>>().len(),
            3
        );
    }

    /// Gate is the **first** half of `ffn_up.weight`: ggml's
    /// `ggml_vec_swiglu_f32` with `swapped = 0` activates the row start.
    /// Swapping these produces fluent-looking but wrong text, not a crash.
    #[test]
    fn ffn_gate_up_puts_gate_first() {
        use crate::engine::loader::test_quant_matrix;
        let (in_dim, n_ff) = (64usize, 3usize);
        let row_bytes = in_dim / 32 * 34;
        let mut bytes = vec![0u8; row_bytes * 2 * n_ff];
        for r in 0..2 * n_ff {
            for b in &mut bytes[r * row_bytes..(r + 1) * row_bytes] {
                *b = (r + 1) as u8;
            }
        }
        let layer = PhiLayer {
            attn_norm: Vec::new(),
            qkv: QkvProjection::Fused(test_quant_matrix(&bytes, 8, in_dim, 2 * n_ff)),
            wo: test_quant_matrix(&bytes, 8, in_dim, 2 * n_ff),
            ffn_norm: Vec::new(),
            w_up: test_quant_matrix(&bytes, 8, in_dim, 2 * n_ff),
            w_down: test_quant_matrix(&bytes, 8, in_dim, 2 * n_ff),
        };
        let (gate, up) = layer.ffn_gate_up();
        assert_eq!((gate.out_dim, up.out_dim), (n_ff, n_ff));
        assert_eq!(gate.raw_bytes()[0], 1, "gate must be the first half");
        assert_eq!(up.raw_bytes()[0], (n_ff + 1) as u8);
    }
}

#[cfg(test)]
mod real_model_tests {
    use super::*;

    /// End-to-end greedy decode against a real `unsloth/Phi-4-mini-instruct-
    /// GGUF` file, checking this module's four architecture-specific
    /// decisions actually hold together on real weights rather than only in
    /// isolation: the fused-QKV split, the SwiGLU half order, the LongRoPE
    /// factor choice, and the `attn_factor` magnitude scale.
    ///
    /// The prompt is the model's own chat format (`<|user|>...<|end|>
    /// <|assistant|>`), and the assertion is on *text*, not logits, because
    /// that is what actually breaks: getting the SwiGLU halves backwards or
    /// the QKV offsets wrong yields fluent-but-wrong output that a
    /// tolerance-based logit check can be talked into accepting.
    ///
    /// Run with `ORANGU_TEST_PHI_MODEL=/path/to/Phi-4-mini-instruct-Q4_K_M.gguf
    /// cargo test --release --bin orangu-server real_model_tests -- --ignored`.
    #[test]
    #[ignore]
    fn phi4_mini_answers_a_factual_question() {
        let path = std::env::var("ORANGU_TEST_PHI_MODEL").expect("set ORANGU_TEST_PHI_MODEL");
        let loaded = LoadedModel::open(std::path::Path::new(&path)).expect("load model");
        assert_eq!(loaded.config.architecture, "phi3");
        let gguf = orangu::gguf::GgufFile::open(std::path::Path::new(&path)).expect("open gguf");
        let tokenizer =
            crate::engine::tokenizer::Tokenizer::from_gguf(&gguf).expect("build tokenizer");
        let model =
            PhiModel::load_with_backend(&loaded, Arc::new(crate::engine::backend::CpuBackend))
                .expect("build model");

        let prompt = "<|user|>What is the capital of France? Answer in one word.<|end|>\
                      <|assistant|>";
        // `false`, as every chat-templated path in this server does: the
        // template already carries whatever prefix the model wants, and
        // Phi-4-mini's own `tokenizer.ggml.add_bos_token` is false.
        let tokens = tokenizer.encode(prompt, false);
        let mut cache = model.new_kv_cache(tokens.len() + 16);
        let mut logits = model.forward(&mut cache, &tokens, 0, 0).expect("prefill");

        let mut generated = Vec::new();
        for step in 0..16 {
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).expect("logits are finite"))
                .expect("non-empty logits")
                .0 as u32;
            if tokenizer.stop_token_ids().contains(&next) {
                break;
            }
            generated.push(next);
            logits = model
                .forward(&mut cache, &[next], tokens.len() + step, 0)
                .expect("decode");
        }

        let text = tokenizer.decode(&generated);
        assert!(
            text.contains("Paris"),
            "expected the answer to name Paris, got {text:?}"
        );
    }
}
