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

//! The `mistral3` forward pass (Mistral 3 / Ministral-3, e.g.
//! `unsloth/Ministral-3-3B-Instruct-2512-GGUF`), confirmed against upstream
//! `llama.cpp/src/models/mistral3.cpp`.
//!
//! Node for node the block shape *is* `arch::llama`'s — RMSNorm, GQA with
//! RoPE, a separate-gate SwiGLU FFN (`LLM_FFN_SILU` + `LLM_FFN_PAR`), an
//! optionally-tied output projection. What earns it its own module is four
//! hyperparameters that `arch::llama` has no reason to carry, each of which
//! is wrong-by-default rather than absent:
//!
//! 1. **A head width that isn't `n_embd / n_head`.** Ministral-3-3B has
//!    `n_embd = 3072` and `n_head = 32` — implying 96 — but declares
//!    `attention.key_length = 128`, and its `attn_q.weight` really is
//!    `[3072, 4096]`. Deriving the head width instead of reading it slices
//!    every head at the wrong offset. See [`ModelConfig::head_dim`].
//! 2. **YaRN RoPE scaling.** `rope.scaling.type = "yarn"` turns on ggml's
//!    interpolation ramp between `beta_fast` and `beta_slow` plus a
//!    magnitude correction — see `tensor::RopeParams`.
//! 3. **NORM RoPE pairing.** `mistral3` sits in upstream's
//!    `LLAMA_ROPE_TYPE_NORM` arm (consecutive pairs), not the NEOX one.
//! 4. **An attention temperature scale.** `attention.temperature_scale`
//!    damps Q as the position runs past the trained context.
//!
//! Multimodal input is out of scope, per this project's existing deferred-
//! multimodal decision: the checkpoint ships a separate `mmproj-*.gguf`
//! vision encoder this engine does not load, and the text graph above is
//! complete without it.

use anyhow::{Context, Result};
use std::sync::Arc;

use super::ModelForward;
use crate::engine::backend::{Backend, MatmulOp};
use crate::engine::kv_cache::KvCache;
use crate::engine::loader::{LoadedModel, ModelConfig, QuantMatrix};
use crate::engine::tensor;

/// **The chained decode path.**
///
/// A Mistral layer is structurally identical to a Llama one — no per-head Q/K
/// norms, no post-norm on either residual, SwiGLU, no projection biases — so
/// [`MistralModel::record_decode_forward`] is `arch::llama`'s chain with this
/// module's own hyperparameters substituted in.
///
/// Two of those hyperparameters are not shape, and both had to be dealt with
/// before the chain could be handed a `mistral3` checkpoint at all:
///
/// 1. **YaRN.** This module builds `RopeParams` with a `freq_scale`, an
///    `attn_factor` and an `ext_factor` ramp whenever the GGUF says
///    `rope.scaling.type = "yarn"`, which Ministral-3-3B does. The fused
///    chain's RoPE now carries those terms (`vulkan::RopeYarn`); before it did,
///    handing it this checkpoint would have rotated by the unscaled angle — no
///    wrong shape, no assertion, just a quietly wrong answer.
/// 2. **The attention temperature scale.** The chain has no term for it, so
///    the path is *gated* on it being the identity rather than taught it. That
///    is not a compromise here: see [`MistralModel::attn_temperature`], it is
///    exactly 1.0 for every position below `n_ctx_orig`.
struct MistralLayer {
    attn_norm: Vec<f32>,
    wq: QuantMatrix,
    wk: QuantMatrix,
    wv: QuantMatrix,
    wo: QuantMatrix,
    ffn_norm: Vec<f32>,
    w_gate: QuantMatrix,
    w_up: QuantMatrix,
    w_down: QuantMatrix,
}

pub struct MistralModel {
    config: ModelConfig,
    backend: Arc<dyn Backend>,
    tok_embeddings: QuantMatrix,
    output_norm: Vec<f32>,
    output_weight: QuantMatrix,
    layers: Vec<MistralLayer>,
    rope: tensor::RopeParams,
    /// `<arch>.attention.temperature_scale`, `None` when the file doesn't
    /// set it. See [`MistralModel::attn_temperature`].
    attn_temp_scale: Option<f32>,
}

impl MistralModel {
    pub fn load_with_backend(loaded: &LoadedModel, backend: Arc<dyn Backend>) -> Result<Self> {
        let config = loaded.config.clone();
        let tok_embeddings = loaded
            .matrix("token_embd.weight")
            .context("loading token_embd.weight")?;
        let (output_norm, _) = loaded
            .tensor("output_norm.weight")
            .context("loading output_norm.weight")?;
        // Ministral-3 ties its output projection to the input embedding;
        // upstream's own `TENSOR_NOT_REQUIRED` + `TENSOR_DUPLICATED` fallback.
        let output_weight = if loaded.has_tensor("output.weight") {
            loaded
                .matrix("output.weight")
                .context("loading output.weight")?
        } else {
            tok_embeddings.clone()
        };

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
            layers.push(MistralLayer {
                attn_norm: get("attn_norm.weight")?,
                wq: get_matrix("attn_q.weight")?,
                wk: get_matrix("attn_k.weight")?,
                wv: get_matrix("attn_v.weight")?,
                wo: get_matrix("attn_output.weight")?,
                ffn_norm: get("ffn_norm.weight")?,
                w_gate: get_matrix("ffn_gate.weight")?,
                w_up: get_matrix("ffn_up.weight")?,
                w_down: get_matrix("ffn_down.weight")?,
            });
        }

        let n_ctx_orig = loaded
            .metadata_u64("rope.scaling.original_context_length")
            .map(|v| v as usize)
            .unwrap_or(config.n_ctx_train);
        let rope = Self::rope_params(loaded, &config, n_ctx_orig);

        // Upstream refuses to load rather than guess when the floor scale
        // would be zero (`mistral3.cpp`: "invalid n_ctx_orig_yarn for
        // attention temperature scaling"), since it divides by it.
        let attn_temp_scale = loaded
            .metadata_f32("attention.temperature_scale")
            .filter(|s| *s != 0.0);
        anyhow::ensure!(
            attn_temp_scale.is_none() || n_ctx_orig != 0,
            "attention.temperature_scale is set but the original context length is zero"
        );

        let model = Self {
            config,
            backend,
            tok_embeddings,
            output_norm,
            output_weight,
            layers,
            rope,
            attn_temp_scale,
        };
        model.validate_shapes()?;
        Ok(model)
    }

    /// Reads the RoPE hyperparameters, including YaRN when the file asks
    /// for it.
    ///
    /// YaRN is opt-in per file: upstream only sets a nonzero
    /// `yarn_ext_factor` when `rope.scaling.type` is literally `"yarn"`
    /// (`llama-context.cpp`), and with `ext_factor == 0` both the ramp and
    /// the magnitude correction are skipped, leaving a plain rope.
    ///
    /// `attn_factor` mirrors that same function: it starts at 1.0, then
    /// upstream pre-cancels the `1 + 0.1*ln(factor)` that `rope_yarn` is
    /// about to multiply back in, then folds in the file's own
    /// `rope.scaling.attn_factor`. The `rope_yarn_log_mul` branch next to it
    /// computes a ratio of two `get_mscale` calls that is 1.0 for every
    /// architecture except DeepSeek-V2, so it drops out here.
    fn rope_params(
        loaded: &LoadedModel,
        config: &ModelConfig,
        n_ctx_orig: usize,
    ) -> tensor::RopeParams {
        let yarn = loaded.metadata_string("rope.scaling.type").as_deref() == Some("yarn");
        let scale_factor = loaded.metadata_f32("rope.scaling.factor").unwrap_or(0.0);
        let freq_scale = if scale_factor == 0.0 {
            1.0
        } else {
            1.0 / scale_factor
        };
        let mut attn_factor = 1.0f32;
        if yarn {
            attn_factor *= 1.0 / (1.0 + 0.1 * scale_factor.ln());
        }
        attn_factor *= loaded
            .metadata_f32("rope.scaling.attn_factor")
            .unwrap_or(1.0);
        tensor::RopeParams {
            rope_dim: config.rope_dim,
            freq_base: config.rope_freq_base,
            freq_scale,
            ext_factor: if yarn { 1.0 } else { 0.0 },
            attn_factor,
            beta_fast: loaded
                .metadata_f32("rope.scaling.yarn_beta_fast")
                .unwrap_or(32.0),
            beta_slow: loaded
                .metadata_f32("rope.scaling.yarn_beta_slow")
                .unwrap_or(1.0),
            n_ctx_orig,
            // `mistral3` is in upstream's `LLAMA_ROPE_TYPE_NORM` arm.
            layout: tensor::RopeLayout::Norm,
        }
    }

    /// Llama-4-style temperature tuning, from
    /// `llm_graph_input_attn_temp::set_input`:
    /// `log(floor(pos / floor_scale) + 1) * scale + 1`, applied to Q after
    /// RoPE.
    ///
    /// Note this is exactly `1.0` — a no-op — for every position below
    /// `floor_scale` (`n_ctx_orig`, 16384 for Ministral-3), because the
    /// floor is then 0 and `log(1) == 0`. It only starts damping attention
    /// past the trained context, which is why a short-prompt test cannot
    /// tell whether it is implemented at all.
    fn attn_temperature(&self, pos: usize) -> f32 {
        match self.attn_temp_scale {
            Some(scale) => {
                let floor_scale = self.rope.n_ctx_orig.max(1);
                ((pos / floor_scale) as f32 + 1.0).ln() * scale + 1.0
            }
            None => 1.0,
        }
    }

    fn head_dim(&self) -> usize {
        self.config.head_dim
    }

    fn kv_dim(&self) -> usize {
        self.config.n_head_kv * self.head_dim()
    }

    /// Rejects a checkpoint whose projections don't match the declared head
    /// width at load time. `head_dim` is read rather than derived here
    /// precisely because the two disagree for this architecture, so a
    /// mismatch would otherwise slice heads at the wrong offsets and
    /// produce confidently wrong tokens instead of an error.
    fn validate_shapes(&self) -> Result<()> {
        let q_dim = self.config.n_head * self.head_dim();
        let kv_dim = self.kv_dim();
        for (i, layer) in self.layers.iter().enumerate() {
            anyhow::ensure!(
                layer.wq.out_dim == q_dim,
                "blk.{i}.attn_q.weight produces {} values, expected n_head * head_dim = {q_dim}",
                layer.wq.out_dim,
            );
            anyhow::ensure!(
                layer.wk.out_dim == kv_dim && layer.wv.out_dim == kv_dim,
                "blk.{i}.attn_k/attn_v produce {}/{}, expected n_head_kv * head_dim = {kv_dim}",
                layer.wk.out_dim,
                layer.wv.out_dim,
            );
            anyhow::ensure!(
                layer.wo.in_dim == q_dim,
                "blk.{i}.attn_output.weight takes {} inputs, expected {q_dim}",
                layer.wo.in_dim,
            );
        }
        Ok(())
    }

    /// One decode step as a single GPU submission, or `None` when this model or
    /// this step is not one the fused chain can describe — see this module's
    /// [`MistralLayer`] note for which hyperparameters that turns on.
    ///
    /// The hidden state never returns to the host: each layer's output buffer
    /// is the next layer's input, so depth costs submissions nothing.
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
        use crate::engine::backend::vulkan::{
            FfnActivation, FusedAttnProjection, FusedLayerInput, GpuInput, RopeYarn,
        };

        if tokens.len() != 1
            || crate::engine::arch::llama::no_fused_qkv()
            || crate::engine::arch::llama::no_fused_post_attention()
        {
            return None;
        }
        // The chain applies no temperature to Q. Below `n_ctx_orig` this is
        // exactly 1.0 and the omission is not an approximation; at or past it
        // the step-by-step path takes over rather than dropping the term.
        if self.attn_temperature(start_pos) != 1.0 {
            return None;
        }
        if !vulkan.prefill_attention_enabled() {
            return None;
        }
        let cfg = &self.config;
        let n_embd = cfg.n_embd;
        let head_dim = self.head_dim();
        let tok = tokens[0] as usize;
        if tok >= cfg.n_vocab {
            return None;
        }
        let x0 = self.tok_embeddings.row(tok).to_vec();

        let mut encoder = vulkan.new_encoder("orangu-server mistral decode");
        // Per-stage GPU timing for this step, when `ORANGU_GPU_TIMESTAMPS=1`
        // and the adapter has the query; inert otherwise. See
        // `VulkanBackend::begin_step_timestamps` for why the slot arithmetic
        // lives there rather than here.
        let n_layer = self.layers.len();
        let ts = vulkan.begin_step_timestamps(&mut encoder, n_layer);
        // Each layer's output buffer, kept alive until the submission that
        // reads them is recorded: a layer's `GpuInput` borrows the previous
        // layer's buffer, so they cannot be dropped inside the loop.
        let mut bufs: Vec<(wgpu::Buffer, u64)> = Vec::with_capacity(self.layers.len());
        for (il, layer) in self.layers.iter().enumerate() {
            let x_input = match bufs.last() {
                Some((buf, offset)) => GpuInput::Gpu(buf, (*offset / 4) as usize),
                None => GpuInput::Cpu(&x0),
            };
            let out = vulkan.record_fused_layer(
                &mut encoder,
                FusedLayerInput {
                    x: x_input,
                    // `mistral3` sits in upstream's NORM arm.
                    pairing: self.rope.layout,
                    yarn: RopeYarn::from_params(&self.rope),
                    // SwiGLU, no per-head Q/K norms, no post-norm on either
                    // residual, and — the convention that is invisible from
                    // every shape — no V normalization.
                    activation: FfnActivation::Swiglu,
                    normalize_v: false,
                    attn_norm: &layer.attn_norm,
                    wq: &layer.wq,
                    q_bias: None,
                    q_norm: None,
                    kv: Some(FusedAttnProjection {
                        wk: &layer.wk,
                        wv: Some(&layer.wv),
                        k_bias: None,
                        v_bias: None,
                        k_norm: None,
                    }),
                    n_head: cfg.n_head,
                    n_head_kv: cfg.n_head_kv,
                    // Read, not derived — the two disagree for this
                    // architecture. See [`MistralModel::validate_shapes`].
                    head_dim,
                    rope_dim: cfg.rope_dim,
                    rope_freq_base: cfg.rope_freq_base,
                    freq_factors: None,
                    eps: cfg.rms_eps,
                    pos: start_pos,
                    // Causal to this position, no sliding window.
                    window_start: 0,
                    window: None,
                    scale: 1.0 / (head_dim as f32).sqrt(),
                    cache: &mut cache.layers[il],
                    wo: &layer.wo,
                    attn_post_norm: None,
                    ffn_norm: &layer.ffn_norm,
                    ffn_gate: &layer.w_gate,
                    ffn_up: &layer.w_up,
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
        let normed = vulkan.record_output_norm(
            &mut encoder,
            GpuInput::Gpu(last_buf, (*last_offset / 4) as usize),
            &self.output_norm,
            cfg.rms_eps,
            n_embd,
        );
        // `slot_id + 1`, not `slot_id`: op resources are keyed by
        // `(weight, batch_slot)`, and the vocab projection must not share a
        // slot with the layer chain that runs into it.
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
        let vulkan = self.backend.as_wgpu()?;
        let (encoder, _, _) =
            self.record_decode_chain(vulkan, cache, tokens, start_pos, slot_id)?;
        let logits = vulkan.submit_and_readback_for(encoder, &self.output_weight, slot_id + 1);
        if vulkan.gpu_timestamps() {
            vulkan.report_timestamps(start_pos, self.layers.len());
        }
        Some(logits)
    }

    /// Runs every transformer layer and returns the pre-final-norm hidden
    /// state for every token (`[n_tokens, n_embd]`).
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
        let q_dim = n_head * head_dim;

        let mut x = vec![0f32; n_tokens * n_embd];
        for (t, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            anyhow::ensure!(tok < cfg.n_vocab, "token id {tok} is out of vocab range");
            x[t * n_embd..(t + 1) * n_embd].copy_from_slice(&self.tok_embeddings.row(tok));
        }

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let mut normed = x.clone();
            tensor::rmsnorm_inplace(&mut normed, &layer.attn_norm, n_tokens, n_embd, cfg.rms_eps);

            // Q/K/V are independent given the same normed input — one batched
            // dispatch rather than three sequential round trips.
            //
            // Deliberately *not* `fused_attention_prefill`, which this shape is
            // otherwise eligible for: measured at −3.4% on Ministral-3-3B,
            // losing all three paired reps. See PERF-GAP.md item 10.
            let mut qkv = self.backend.matmul_batch(&[
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
            let v = qkv.pop().unwrap();
            let mut k = qkv.pop().unwrap();
            let mut q = qkv.pop().unwrap();

            let layer_cache = &mut cache.layers[layer_idx];
            for t in 0..n_tokens {
                let pos = start_pos + t;
                tensor::rope_apply_params_inplace(
                    &mut q[t * q_dim..(t + 1) * q_dim],
                    n_head,
                    head_dim,
                    pos,
                    None,
                    &self.rope,
                );
                // After RoPE, before attention — exactly where upstream's
                // graph puts `Qcur_attn_temp_scaled`.
                let temp = self.attn_temperature(pos);
                if temp != 1.0 {
                    for value in q[t * q_dim..(t + 1) * q_dim].iter_mut() {
                        *value *= temp;
                    }
                }
                tensor::rope_apply_params_inplace(
                    &mut k[t * kv_dim..(t + 1) * kv_dim],
                    n_head_kv,
                    head_dim,
                    pos,
                    None,
                    &self.rope,
                );
                layer_cache.push(
                    &k[t * kv_dim..(t + 1) * kv_dim],
                    &v[t * kv_dim..(t + 1) * kv_dim],
                );
            }

            // Plain causal attention, on the GPU when the batch is wide
            // enough — `engine::attention` owns that choice for every
            // architecture.
            let mut attn_out: Vec<f32> = Vec::new();
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

            // One submission for `wo` → residual → FFN norm → gate/up → SwiGLU
            // → `down` → residual, instead of three blocking round trips. Same
            // call and same shape as `arch::llama` — this family has no
            // post-norms and a SwiGLU gate — and the `else` branch below is the
            // sequence the fused chain is cross-checked against.
            let fused = self
                .backend
                .as_wgpu()
                .filter(|_| !crate::engine::arch::llama::no_fused_post_attention())
                .and_then(|vulkan| {
                    vulkan.fused_post_attention_prefill(
                        crate::engine::backend::vulkan::AttnOutSrc::Host(&attn_out),
                        &x,
                        n_tokens,
                        &layer.wo,
                        None,
                        &layer.ffn_norm,
                        &layer.w_gate,
                        &layer.w_up,
                        &layer.w_down,
                        None,
                        cfg.rms_eps,
                        crate::engine::backend::vulkan::FfnActivation::Swiglu,
                    )
                });
            if let Some(out) = fused {
                x = out;
            } else {
                let attn_proj = self.backend.matmul(&attn_out, n_tokens, &layer.wo);
                tensor::add_inplace(&mut x, &attn_proj);

                let mut normed2 = x.clone();
                tensor::rmsnorm_inplace(
                    &mut normed2,
                    &layer.ffn_norm,
                    n_tokens,
                    n_embd,
                    cfg.rms_eps,
                );
                let mut gate_up = self.backend.matmul_batch(&[
                    MatmulOp {
                        x: &normed2,
                        n_tokens,
                        w: &layer.w_gate,
                    },
                    MatmulOp {
                        x: &normed2,
                        n_tokens,
                        w: &layer.w_up,
                    },
                ]);
                let up = gate_up.pop().unwrap();
                let mut gate = gate_up.pop().unwrap();
                for g in gate.iter_mut() {
                    *g = tensor::silu(*g);
                }
                tensor::mul_inplace(&mut gate, &up);
                let down = self.backend.matmul(&gate, n_tokens, &layer.w_down);
                tensor::add_inplace(&mut x, &down);
            }
        }

        Ok(x)
    }
}

impl ModelForward for MistralModel {
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
    /// The temperature curve, against the formula's own definition rather
    /// than a remembered constant: flat at exactly 1.0 across the whole
    /// trained context (the floor is 0 there, and `log(1) == 0`), then
    /// rising once positions run past it.
    #[test]
    fn attn_temperature_is_identity_within_the_trained_context() {
        let model_temp =
            |pos: usize, floor: usize, scale: f32| ((pos / floor) as f32 + 1.0).ln() * scale + 1.0;
        for pos in [0usize, 1, 100, 16383] {
            assert_eq!(model_temp(pos, 16384, 0.1), 1.0, "pos {pos}");
        }
        assert!(model_temp(16384, 16384, 0.1) > 1.0);
        assert!(model_temp(65536, 16384, 0.1) > model_temp(16384, 16384, 0.1));
    }
}

#[cfg(test)]
mod real_model_tests {
    use super::*;

    /// End-to-end greedy decode against a real `mistral3` `.gguf`
    /// (verified against every published `Ministral-3-3B-Instruct-2512`
    /// quantization this engine can dequantize).
    ///
    /// The prompt comes from the model's own `tokenizer.chat_template`,
    /// rendered the way `http::openai` renders it — Mistral's `[INST]`
    /// format, which also injects a long default system prompt when the
    /// caller supplies none.
    ///
    /// **Not** valid for the `UD-IQ1_S` quantization, and that is a
    /// property of the file rather than of this module: at 1.5625 bits per
    /// weight it still produces fluent English but no longer recalls the
    /// answer, and real llama.cpp fed the exact same file answers just as
    /// wrongly ("The answer of the question about \"What is the answer of
    /// the French language..."). It was checked the level below instead —
    /// its layer-0 tensor sums match llama.cpp's own
    /// (`embd` -2.216335 vs -2.216340, `attn_norm-0` -82.465033 vs
    /// -82.465195, `Kcur-0` post-RoPE -277.0898 vs -277.0491), which is
    /// what actually exercises the `IQ1_S`/`IQ1_M` dequantizers.
    ///
    /// Run with `ORANGU_TEST_MISTRAL_MODEL=/path/to/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf
    /// cargo test --release --bin orangu-server real_model_tests -- --ignored`.
    #[test]
    #[ignore]
    fn mistral3_answers_a_factual_question() {
        let path =
            std::env::var("ORANGU_TEST_MISTRAL_MODEL").expect("set ORANGU_TEST_MISTRAL_MODEL");
        let loaded = LoadedModel::open(std::path::Path::new(&path)).expect("load model");
        assert_eq!(loaded.config.architecture, "mistral3");
        let gguf = orangu::gguf::GgufFile::open(std::path::Path::new(&path)).expect("open gguf");
        let tokenizer =
            crate::engine::tokenizer::Tokenizer::from_gguf(&gguf).expect("build tokenizer");
        let model =
            MistralModel::load_with_backend(&loaded, Arc::new(crate::engine::backend::CpuBackend))
                .expect("build model");

        let template_source = gguf
            .metadata
            .iter()
            .find_map(|(k, v)| match (k.as_str(), v) {
                ("tokenizer.chat_template", orangu::gguf::GgufValue::String(s)) => Some(s.clone()),
                _ => None,
            })
            .expect("model has a chat template");
        let prompt = crate::engine::chat_template::ChatTemplate::new(template_source)
            .render(
                &[crate::engine::chat_template::ChatMessage::text(
                    "user",
                    "What is the capital of France? Answer in one word.",
                )],
                true,
                tokenizer
                    .bos_token
                    .and_then(|id| tokenizer.token_text(id))
                    .unwrap_or(""),
                tokenizer
                    .eos_token
                    .and_then(|id| tokenizer.token_text(id))
                    .unwrap_or(""),
                None,
            )
            .expect("render chat template");
        let tokens = tokenizer.encode(&prompt, false);
        let mut cache = model.new_kv_cache(tokens.len() + 16);
        let mut logits = model.forward(&mut cache, &tokens, 0, 0).expect("prefill");

        let stop = tokenizer.stop_token_ids();
        let mut generated = Vec::new();
        for step in 0..16 {
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).expect("logits are finite"))
                .expect("non-empty logits")
                .0 as u32;
            if stop.contains(&next) {
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
