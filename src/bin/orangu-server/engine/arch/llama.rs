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

//! The Llama-style forward pass: grouped-query attention, RoPE, RMSNorm,
//! SwiGLU — the shape shared by Llama/Llama3/Qwen2/Qwen3/Mistral GGUFs
//! (tensor names confirmed against `llama.cpp/src/llama-arch.cpp`'s
//! `LLM_TENSOR_NAMES` table for `LLM_ARCH_LLAMA`).
//!
//! Weight matrices and embedding tables stay `mmap`-backed and are
//! dequantized one row at a time, on demand, via `QuantMatrix` — not
//! eagerly materialized to `f32` at load time. Only small per-element
//! tensors (norms, biases) are eagerly dequantized. This keeps resident
//! memory close to the file's own size rather than the ~4x an eager,
//! fully-dequantized-to-`f32` approach costs — the difference between a
//! large (tens-of-billions-of-parameters) model fitting in RAM at all or
//! not.

use anyhow::{Context, Result};
use std::sync::Arc;

use super::ModelForward;
use crate::engine::backend::{Backend, MatmulOp};
use crate::engine::kv_cache::KvCache;
use crate::engine::loader::{LoadedModel, ModelConfig, QuantMatrix};
use crate::engine::tensor;

struct LlamaLayer {
    attn_norm: Vec<f32>,
    wq: QuantMatrix,
    wk: QuantMatrix,
    wv: QuantMatrix,
    wo: QuantMatrix,
    /// Q/K/V projection biases — present on Qwen2/Qwen3-shaped GGUFs,
    /// absent on plain Llama/Mistral ones (`attn_*.bias` tensors simply
    /// don't exist in the file for those; confirmed directly against a
    /// downloaded Qwen2.5 GGUF, which has all three).
    q_bias: Option<Vec<f32>>,
    k_bias: Option<Vec<f32>>,
    v_bias: Option<Vec<f32>>,
    /// Per-head RMSNorm on Q/K after projection, before RoPE — present on
    /// Qwen3/Qwen3VL-shaped GGUFs (`attn_q_norm.weight`/`attn_k_norm.
    /// weight`, each `[head_dim]`), absent on Qwen2/Llama/Mistral ones
    /// (confirmed directly against a real downloaded `Qwen3-VL-Embedding-
    /// 8B` GGUF's `src/models/qwen3vl.cpp` graph: `Qcur = build_norm(Qcur,
    /// attn_q_norm, ..., LLM_NORM_RMS, il)` runs immediately after `build_
    /// qkv`, before `ggml_rope_multi`).
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
    ffn_norm: Vec<f32>,
    w_gate: QuantMatrix,
    w_up: QuantMatrix,
    w_down: QuantMatrix,
}

pub struct LlamaModel {
    config: ModelConfig,
    backend: Arc<dyn Backend>,
    tok_embeddings: QuantMatrix,
    output_norm: Vec<f32>,
    output_weight: QuantMatrix,
    layers: Vec<LlamaLayer>,
    /// `rope_freqs.weight` (`[rope_dim / 2]`) — the per-pair frequency
    /// divisor a Llama-3.1/3.2 checkpoint carries because its RoPE uses
    /// Meta's `"llama3"` scaling, which `convert_hf_to_gguf.py` bakes into
    /// this tensor at conversion time rather than leaving as scalar
    /// hyperparameters for the runtime to re-derive.
    ///
    /// Upstream applies it unconditionally when present:
    /// `llama_model::get_rope_factors` returns `layers[il].rope_freqs`
    /// from its *first* branch, before any context-length test, and
    /// `src/models/llama.cpp` hands the result to `ggml_rope_ext` as
    /// `freq_factors` for both Q and K.
    ///
    /// `None` for every checkpoint without the tensor — plain Llama 2,
    /// Qwen2/Qwen3, Mistral, qwen3vl — which is why loading it is purely
    /// additive: those models rotate exactly as they did before.
    ///
    /// Ignoring it is not a subtle quality regression. Llama-3.2-1B answers
    /// "What is the capital of France?" with `"I am I am I am I am"` when
    /// this is left unapplied, and correctly when it is.
    rope_freq_factors: Option<Vec<f32>>,
    /// Rotary width, base and pairing, bundled so the rope call doesn't
    /// take nine positional arguments. `arch::mistral` builds a richer one
    /// of these for YaRN; nothing this module serves needs that.
    rope: tensor::RopeParams,
}

/// `llama_model_rope_type`'s answer (`llama.cpp/src/llama-model.cpp`) for
/// the architectures `engine::loader::LLAMA_STYLE_ARCHITECTURES` routes
/// here.
///
/// `llama` sits in upstream's `LLAMA_ROPE_TYPE_NORM` arm ("use what we call
/// a normal RoPE, operating on pairs of consecutive head values"), together
/// with `mistral`; `qwen2`/`qwen3`/`qwen3vl` sit in the
/// `LLAMA_ROPE_TYPE_NEOX` one ("the pairs of head values are offset by
/// n_rot/2"). Treating them alike — which this module did, rotating
/// everything NEOX-style — leaves Qwen correct and every Llama checkpoint
/// quietly wrong: `Qcur` matches real llama.cpp to 5 significant figures
/// *before* RoPE and comes out at `-354.6` against upstream's `6.66`
/// after. Unknown architectures keep the previous NEOX default rather than
/// failing to load, since that is the majority answer upstream and the
/// behavior everything here already had.
fn rope_layout_for(architecture: &str) -> tensor::RopeLayout {
    match architecture {
        "llama" | "mistral" => tensor::RopeLayout::Norm,
        _ => tensor::RopeLayout::Neox,
    }
}

impl LlamaModel {
    pub fn load_with_backend(loaded: &LoadedModel, backend: Arc<dyn Backend>) -> Result<Self> {
        let config = loaded.config.clone();
        let tok_embeddings = loaded
            .matrix("token_embd.weight")
            .context("loading token_embd.weight")?;
        let (output_norm, _) = loaded
            .tensor("output_norm.weight")
            .context("loading output_norm.weight")?;
        // Some models tie the output projection to the input embedding and
        // simply omit a separate "output.weight" tensor.
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
            let get_optional = |suffix: &str| -> Result<Option<Vec<f32>>> {
                let name = format!("blk.{i}.{suffix}");
                if !loaded.has_tensor(&name) {
                    return Ok(None);
                }
                Ok(Some(
                    loaded
                        .tensor(&name)
                        .with_context(|| format!("loading {name}"))?
                        .0,
                ))
            };
            layers.push(LlamaLayer {
                attn_norm: get("attn_norm.weight")?,
                wq: get_matrix("attn_q.weight")?,
                wk: get_matrix("attn_k.weight")?,
                wv: get_matrix("attn_v.weight")?,
                wo: get_matrix("attn_output.weight")?,
                q_bias: get_optional("attn_q.bias")?,
                k_bias: get_optional("attn_k.bias")?,
                v_bias: get_optional("attn_v.bias")?,
                q_norm: get_optional("attn_q_norm.weight")?,
                k_norm: get_optional("attn_k_norm.weight")?,
                ffn_norm: get("ffn_norm.weight")?,
                w_gate: get_matrix("ffn_gate.weight")?,
                w_up: get_matrix("ffn_up.weight")?,
                w_down: get_matrix("ffn_down.weight")?,
            });
        }

        let rope_freq_factors = if loaded.has_tensor("rope_freqs.weight") {
            let (factors, _) = loaded
                .tensor("rope_freqs.weight")
                .context("loading rope_freqs.weight")?;
            anyhow::ensure!(
                factors.len() >= config.rope_dim / 2,
                "rope_freqs.weight has {} entries, need {} for rope.dimension_count = {}",
                factors.len(),
                config.rope_dim / 2,
                config.rope_dim,
            );
            Some(factors)
        } else {
            None
        };

        let rope_layout = rope_layout_for(&config.architecture);
        let rope = tensor::RopeParams {
            rope_dim: config.rope_dim,
            freq_base: config.rope_freq_base,
            layout: rope_layout,
            ..tensor::RopeParams::default()
        };
        Ok(Self {
            config,
            backend,
            tok_embeddings,
            output_norm,
            output_weight,
            layers,
            rope_freq_factors,
            rope,
        })
    }

    fn head_dim(&self) -> usize {
        self.config.head_dim
    }
}

/// Whether `ORANGU_NO_FUSED_POST_ATTN` forces the unfused `wo`/FFN sequence.
///
/// Exists to be the **control** for the fused chain's A/B: the alternative is
/// comparing against a different build, and LESSONS §17 is that the control
/// should be the code the change replaced rather than an approximation of it.
/// Read once — an env lookup per layer per token would itself be measurable.
/// Whether `ORANGU_NO_FUSED_QKV` restores the step-by-step Q/K/V, RoPE,
/// KV-write and attention sequence.
///
/// The control for the fused chain's A/B, and the fallback for anything the
/// chain does not implement. Read once — an env lookup per layer per token
/// would itself be measurable.
pub fn no_fused_qkv() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| crate::engine::env::flag_on("ORANGU_NO_FUSED_QKV"))
}

/// Narrowest batch the fused pre-attention chain is worth taking.
///
/// The chain's own crossover, and **not** the same question
/// `engine::attention`'s `ORANGU_ATTENTION_MIN_TOKENS` answers: that one asks
/// whether GPU attention alone beats the CPU loop, and the answer is "not until
/// much wider than this". Fusing attention into a single submission with Q/K/V,
/// RoPE and the KV write changes the trade — the saved round trips pay for GPU
/// attention well before GPU attention pays for itself — so this threshold sits
/// far below that one. Two thresholds because there are two crossovers, both
/// swept; `PERF-GAP.md` has them.
///
/// A short continuation of a cached prompt is the shape that lands here, and it
/// is the common one in multi-turn chat: everything but the newest message
/// comes from the prefix cache.
const MIN_FUSED_TOKENS: usize = 24;

/// [`MIN_FUSED_TOKENS`], overridable per run with `ORANGU_FUSED_MIN_TOKENS`.
///
/// The knob exists so the A/B for this threshold has the *shipping* code as its
/// control rather than a rebuild that differs in a constant — LESSONS §17. Read
/// once; a lookup per layer per token would itself be measurable.
pub fn min_fused_tokens() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("ORANGU_FUSED_MIN_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(MIN_FUSED_TOKENS)
    })
}

pub fn no_fused_post_attention() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| crate::engine::env::flag_on("ORANGU_NO_FUSED_POST_ATTN"))
}

impl LlamaModel {
    /// One decode step as a single GPU submission, or `None` when this model or
    /// this step is not one the fused chain can describe.
    ///
    /// The hidden state never returns to the host: each layer's output buffer is
    /// the next layer's input, so depth costs submissions nothing. `PERF-GAP.md`
    /// G3 measures that as the difference between an engine that can fill the
    /// device and one that cannot — the generic path costs one GPU round trip
    /// per layer per chain and never passes 66% engine occupancy however many
    /// concurrent requests it is given, while this form reaches 98% with two.
    ///
    /// `None` is the ordinary answer for anything the chain does not cover, and
    /// the caller then takes the step-by-step path unchanged.
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

    /// One *run* of the decode chain: layers `layers`, starting from the
    /// host vector `x_in`, recorded into one encoder on one device.
    ///
    /// `with_tail` appends `output_norm` and the vocab projection, so the
    /// returned buffer is the logits; without it the buffer is the last
    /// layer's hidden state, for the caller to bring to the host and hand
    /// to the next device (`VulkanBackend::submit_and_read_at`).
    ///
    /// A single-device model is one run over every layer with the tail —
    /// exactly what [`Self::record_decode_chain`] asks for, and byte-for-
    /// byte the code that ran before this took a range. A split model is
    /// one run per device.
    ///
    /// Why runs at all, rather than one encoder that switches device: a
    /// measured decode step on this project's own hardware is *faster* at
    /// one submission per layer than at one per token
    /// (`ORANGU_DECODE_CHUNKS`), because early work executes while the CPU
    /// is still recording later work. Submission count is not the cost a
    /// split pays; losing this fused per-layer chain was.
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
            FfnActivation, FusedAttnProjection, FusedLayerInput, GpuInput, RopeYarn,
        };

        if no_fused_qkv() || no_fused_post_attention() {
            return None;
        }
        if !vulkan.prefill_attention_enabled() {
            return None;
        }
        let cfg = &self.config;
        let n_embd = cfg.n_embd;
        let head_dim = self.head_dim();

        let mut encoder = vulkan.new_encoder("orangu-server llama decode");
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
        for il in layers.clone() {
            let layer = &self.layers[il];
            let x_input = match bufs.last() {
                Some((buf, offset)) => GpuInput::Gpu(buf, (*offset / 4) as usize),
                None => GpuInput::Cpu(x_in),
            };
            let out = vulkan.record_fused_layer(
                &mut encoder,
                FusedLayerInput {
                    x: x_input,
                    // `llama`/`mistral` are NORM; `qwen2` and the rest NEOX.
                    pairing: self.rope.layout,
                    // Derived rather than asserted identity: this family sets
                    // no YaRN today, and `from_params` keeps the chain correct
                    // rather than merely lucky if that changes.
                    yarn: RopeYarn::from_params(&self.rope),
                    // This family is SwiGLU throughout, has no per-head Q/K
                    // norms, no post-norm on either residual, and — the
                    // convention that is invisible from every shape — does not
                    // normalize V.
                    activation: FfnActivation::Swiglu,
                    normalize_v: false,
                    attn_norm: &layer.attn_norm,
                    wq: &layer.wq,
                    q_bias: layer.q_bias.as_deref(),
                    q_norm: None,
                    kv: Some(FusedAttnProjection {
                        wk: &layer.wk,
                        wv: Some(&layer.wv),
                        k_bias: layer.k_bias.as_deref(),
                        v_bias: layer.v_bias.as_deref(),
                        k_norm: None,
                    }),
                    n_head: cfg.n_head,
                    n_head_kv: cfg.n_head_kv,
                    head_dim,
                    rope_dim: cfg.rope_dim,
                    rope_freq_base: cfg.rope_freq_base,
                    freq_factors: self.rope_freq_factors.as_deref(),
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
        if !with_tail {
            // This run's hidden state, for the caller to read back and hand
            // to the next device. The timestamp resolve still has to be
            // recorded, or the query set this encoder wrote into is never
            // resolved.
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
        // `slot_id + 1`, not `slot_id`: op resources are keyed by
        // `(weight, batch_slot)`, and the vocab projection must not share a slot
        // with the layer chain that runs into it. gemma keys its own output
        // projection the same way.
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
            // at all, or the model is split. `Self::record_split_decode`
            // answers the second case and `None` the first.
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

    /// The same fused per-layer decode chain, on a model whose layers live
    /// on more than one device: one encoder per run of consecutive layers
    /// sharing a device, with the hidden state crossing to host memory in
    /// between.
    ///
    /// This is what a split was missing. Without it `Backend::as_wgpu`
    /// answering `None` took *every* layer — not just the ones near a
    /// boundary — off `record_fused_layer` and onto the step-by-step path,
    /// which round-trips through host memory between individual ops. The
    /// boundary crossings a split really owes are one per device, and they
    /// are the two `submit_and_read_at` calls below.
    ///
    /// `None` — falling back to the step-by-step path — whenever anything
    /// here is not exactly expressible: a layer with no GPU behind it (a
    /// CPU overflow tier), or a device that declines the chain.
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
        // One device is not a split; `as_wgpu` would have answered it.
        if runs.len() < 2 {
            return None;
        }
        // The vocab projection runs where its own weights are, which is
        // device 0 (`LoadedModel::device_for_tensor` keeps every non-layer
        // tensor there). When the last layer is elsewhere, that is one more
        // hand-off, and it is already counted in `runs`.
        let tail_device = self.output_weight.device();

        let mut x = self.tok_embeddings.row(tok).to_vec();
        for (index, (device, layers)) in runs.iter().enumerate() {
            let vulkan = self.backend.as_wgpu_on(*device)?;
            let last = index + 1 == runs.len();
            let with_tail = last && *device == tail_device;
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

        // The last layers were not on the tail's device, so the projection
        // is a run of its own with no layers in front of it.
        let vulkan = self.backend.as_wgpu_on(tail_device)?;
        let mut encoder = vulkan.new_encoder("orangu-server llama decode tail");
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

    /// Runs every transformer layer and returns the pre-final-norm hidden
    /// state for every token (`[n_tokens, n_embd]`) — the shared core of
    /// both next-token prediction ([`ModelForward::forward`]) and pooled
    /// embeddings ([`LlamaModel::forward_hidden_states`]).
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
        let kv_dim = n_head_kv * head_dim;

        // Embedding lookup: x[t, :] = tok_embeddings[token[t], :].
        let mut x = vec![0f32; n_tokens * n_embd];
        for (t, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            anyhow::ensure!(tok < cfg.n_vocab, "token id {tok} is out of vocab range");
            x[t * n_embd..(t + 1) * n_embd].copy_from_slice(&self.tok_embeddings.row(tok));
        }

        // Grown once and reused across layers rather than allocated per layer:
        // at prefill widths this is megabytes a layer. The two norm scratch
        // buffers are the same trick applied to what used to be `x.clone()`
        // — see `tensor::rmsnorm_into`.
        let mut attn_out: Vec<f32> = Vec::new();
        let mut normed: Vec<f32> = Vec::new();
        let mut normed2: Vec<f32> = Vec::new();
        // The projection outputs, on the same principle — see
        // `Backend::matmul_into`. `ffn` is the big one: `n_tokens * n_ff`.
        let mut attn_proj: Vec<f32> = Vec::new();
        let mut ffn_out: Vec<f32> = Vec::new();
        let mut ffn_scratch = super::FfnScratch::default();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // `Some` when attention left its output in a device buffer; the
            // post-attention chain then consumes it without a host bounce.
            let mut attn_on_device: Option<wgpu::Buffer> = None;
            tensor::rmsnorm_into(
                &mut normed,
                &x,
                &layer.attn_norm,
                n_tokens,
                n_embd,
                cfg.rms_eps,
            );

            // The whole pre-attention half — Q/K/V, RoPE, the KV-cache write
            // and attention itself — as one GPU submission, when this layer's
            // conventions match what the fused chain implements.
            //
            // Three of them, and they are not visible from the signature: the
            // per-head Q/K norms (Qwen3 has them, this family does not), the
            // per-head weightless V norm (gemma has it, this family does not),
            // and the RoPE pairing (`llama`/`mistral` are NORM, everything else
            // NEOX). Projection **biases** are a fourth, and the chain does
            // support them — Qwen2 has them and takes this path; they are
            // cross-checked per projection against the step-by-step sequence.
            // **Wide prefill only**, and both bounds are measured rather than
            // assumed.
            //
            // Below `MIN_FUSED_TOKENS` the chain loses to the step-by-step
            // path. It always runs attention on the GPU, and at narrow widths
            // the CPU loop beats that by more than the fusion's saved round
            // trips are worth; above it the saving dominates and the chain
            // wins outright.
            //
            // At `n_tokens == 1` it is worse than merely slower: running
            // attention itself routes a decode step away from the split
            // ("flash-decode") kernel `engine::attention` would otherwise pick,
            // and that kernel is what keeps decode flat as context grows. One
            // submission is not worth the wrong kernel. `MIN_FUSED_TOKENS`
            // already excludes decode; the width bound and the decode bound are
            // separate facts, so this does not lean on that coincidence.
            // Letting a *decode* step take this chain where the split kernel
            // would not have run was tried and measured neutral (−0.6% at
            // depth 0, −0.2% at 512): at shallow context the submission count
            // is unchanged either way, and the CPU work moved to the GPU — one
            // token's RoPE, one KV row, a short window — is too small to show.
            // So the bound stays a plain width test.
            let fusable = n_tokens > 1
                && n_tokens >= min_fused_tokens()
                && !no_fused_qkv()
                && layer.q_norm.is_none()
                && layer.k_norm.is_none();
            let fused_qkv = self
                .backend
                // This layer's card — see `Backend::as_wgpu_on`.
                .as_wgpu_on(layer.wo.device())
                .filter(|_| fusable && !no_fused_post_attention())
                .and_then(|vulkan| {
                    vulkan.fused_attention_prefill(
                        crate::engine::backend::vulkan::FusedAttnPrefillInput {
                            q_bias: layer.q_bias.as_deref(),
                            pairing: self.rope.layout,
                            yarn: crate::engine::backend::vulkan::RopeYarn::from_params(&self.rope),
                            normalize_v: false,
                            normed: &normed,
                            n_tokens,
                            start_pos,
                            wq: &layer.wq,
                            q_norm: None,
                            kv: Some(crate::engine::backend::vulkan::FusedAttnPrefillKv {
                                k_bias: layer.k_bias.as_deref(),
                                v_bias: layer.v_bias.as_deref(),
                                wk: &layer.wk,
                                k_norm: None,
                                wv: Some(&layer.wv),
                            }),
                            n_head,
                            n_head_kv,
                            head_dim,
                            rope_dim: cfg.rope_dim,
                            rope_freq_base: cfg.rope_freq_base,
                            freq_factors: self.rope_freq_factors.as_deref(),
                            eps: cfg.rms_eps,
                            n_swa: 0,
                            causal: true,
                            scale: 1.0 / (head_dim as f32).sqrt(),
                            want_attn_out_host: true,
                        },
                        &mut cache.layers[layer_idx],
                    )
                });

            if let Some(fused) = fused_qkv {
                // The recorder has already committed each stripe's K/V into the
                // cache — it has to, since a later stripe's attention reads
                // them — so there is nothing to commit here. Doing it again
                // pushes every position twice and fills the cache.
                attn_out = fused.attn_out;
            } else {
                // Independent given the same normed input — one batched
                // dispatch instead of three sequential round-trips (matters
                // most for a GPU backend; see `Backend::matmul_batch`).
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
                let mut v = qkv.pop().unwrap();
                let mut k = qkv.pop().unwrap();
                let mut q = qkv.pop().unwrap();
                if let Some(bias) = &layer.q_bias {
                    tensor::add_bias_per_row(&mut q, bias, n_tokens);
                }
                if let Some(bias) = &layer.k_bias {
                    tensor::add_bias_per_row(&mut k, bias, n_tokens);
                }
                if let Some(bias) = &layer.v_bias {
                    tensor::add_bias_per_row(&mut v, bias, n_tokens);
                }
                // Per-head RMSNorm, before RoPE — `Qwen3-VL-Embedding-8B`'s own
                // `src/models/qwen3vl.cpp` graph runs this immediately after
                // `build_qkv`, before `ggml_rope_multi`; `None` (Qwen2/Llama/
                // Mistral) is a no-op.
                if let Some(q_norm) = &layer.q_norm {
                    tensor::rmsnorm_inplace(
                        &mut q,
                        q_norm,
                        n_tokens * n_head,
                        head_dim,
                        cfg.rms_eps,
                    );
                }
                if let Some(k_norm) = &layer.k_norm {
                    tensor::rmsnorm_inplace(
                        &mut k,
                        k_norm,
                        n_tokens * n_head_kv,
                        head_dim,
                        cfg.rms_eps,
                    );
                }

                // RoPE, then append this token's K/V to the sequence's cache —
                // one token (one row) at a time, in prompt order, since a later
                // token's cache entry must exist before an even-later token's
                // attention can see it.
                let layer_cache = &mut cache.layers[layer_idx];
                for t in 0..n_tokens {
                    let pos = start_pos + t;
                    tensor::rope_apply_params_inplace(
                        &mut q[t * n_head * head_dim..(t + 1) * n_head * head_dim],
                        n_head,
                        head_dim,
                        pos,
                        self.rope_freq_factors.as_deref(),
                        &self.rope,
                    );
                    tensor::rope_apply_params_inplace(
                        &mut k[t * kv_dim..(t + 1) * kv_dim],
                        n_head_kv,
                        head_dim,
                        pos,
                        self.rope_freq_factors.as_deref(),
                        &self.rope,
                    );
                    layer_cache.push(
                        &k[t * kv_dim..(t + 1) * kv_dim],
                        &v[t * kv_dim..(t + 1) * kv_dim],
                    );
                }

                // Causal attention: token t (now at absolute position
                // start_pos+t) attends to every cached position up to and
                // including its own. `engine::attention` decides whether that runs
                // on the GPU or as the CPU loop; the closure is the CPU window and
                // `causal`/`n_swa` describe the same range to the kernel.
                let params = crate::engine::attention::Params {
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
                };
                // A decode step's attention output goes straight into the
                // `wo`/FFN chain, which is itself on the GPU — so when the
                // split kernel runs it, leaving the result on the device saves
                // reading `[n_head * head_dim]` floats to the host and
                // uploading them again one statement later. `None` means this
                // shape did not take the GPU path; the host vector below is
                // then the only answer, exactly as before.
                attn_on_device = crate::engine::attention::attention_decode_on_device(
                    &q,
                    layer_cache,
                    &params,
                    |t| (0, start_pos + t),
                );
                if attn_on_device.is_none() {
                    crate::engine::attention::attention(
                        &mut attn_out,
                        &q,
                        layer_cache,
                        &params,
                        |t| (0, start_pos + t),
                    );
                }
            }

            // The whole second half of the layer — `wo`, the residual add, the
            // FFN norm, gate/up, SwiGLU, `down`, the second residual add — as
            // **one** GPU submission with nothing in between reaching the host.
            //
            // Unfused this is three blocking submit→fence→readback cycles
            // (`wo`, `gate`/`up`, `down`) out of the five a layer costs, and
            // `PERF-GAP.md` prices a round trip on this stack at ~260 µs. The
            // fused chain is cross-checked against exactly the sequence in the
            // `else` branch below
            // (`fused_post_attention_prefill_matches_the_unfused_sequence_swiglu_*`).
            let fused = self
                .backend
                .as_wgpu_on(layer.wo.device())
                .filter(|_| !no_fused_post_attention())
                .and_then(|vulkan| {
                    vulkan.fused_post_attention_prefill(
                        match &attn_on_device {
                            Some(buf) => {
                                crate::engine::backend::vulkan::AttnOutSrc::Gpu(buf, 0, n_tokens)
                            }
                            None => crate::engine::backend::vulkan::AttnOutSrc::Host(&attn_out),
                        },
                        &x,
                        n_tokens,
                        &layer.wo,
                        // Llama-style has no post-norm on either residual add,
                        // and no norm is not a norm with weights of one.
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
                // The fused chain declined after attention had already left its
                // output on the device, so bring it back — the CPU sequence
                // below reads `attn_out`, and it would otherwise read zeros.
                if let (Some(buf), Some(vulkan)) =
                    (&attn_on_device, self.backend.as_wgpu_on(layer.wo.device()))
                {
                    attn_out = vulkan.read_buffer_f32(buf, n_tokens * n_head * head_dim);
                }
                self.backend
                    .matmul_into(&mut attn_proj, &attn_out, n_tokens, &layer.wo);
                tensor::add_inplace(&mut x, &attn_proj);

                tensor::rmsnorm_into(
                    &mut normed2,
                    &x,
                    &layer.ffn_norm,
                    n_tokens,
                    n_embd,
                    cfg.rms_eps,
                );
                // Shared with the dense FFN of the Qwen 3.5 hybrid trunk —
                // `LLM_FFN_SILU`/`LLM_FFN_PAR` is one computation and this
                // family (Llama, Mistral, Qwen2, Qwen3) and that one run the
                // same one.
                super::swiglu_ffn_into(
                    self.backend.as_ref(),
                    &mut ffn_out,
                    &mut ffn_scratch,
                    &normed2,
                    n_tokens,
                    &layer.w_gate,
                    &layer.w_up,
                    &layer.w_down,
                );
                tensor::add_inplace(&mut x, &ffn_out);
            }
        }

        Ok(x)
    }
}

impl ModelForward for LlamaModel {
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
        let kv_dim = self.config.n_head_kv * self.head_dim();
        KvCache::new(self.config.n_layer, capacity, kv_dim)
    }

    fn forward(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        slot_id: usize,
    ) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let n_tokens = tokens.len();
        let n_embd = cfg.n_embd;

        // A decode step, whole, as one GPU submission. See
        // `record_decode_forward`; `None` falls through to the path below.
        if let Some(logits) = self.record_decode_forward(cache, tokens, start_pos, slot_id) {
            return Ok(logits);
        }

        let x = self.run_layers(cache, tokens, start_pos)?;

        // Only the last token's hidden state is needed for next-token
        // logits — a batched prefill doesn't need every position's output.
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
mod fusion_width_tests {
    use super::*;

    /// The fused chain must never claim a decode step, and it is gated on two
    /// independent clauses that both happen to exclude one: `n_tokens > 1` and
    /// `n_tokens >= MIN_FUSED_TOKENS`. This asserts the second one does the job
    /// on its own, so deleting the first — which is exactly the edit that
    /// caused the regression this gate was added for — cannot silently let
    /// decode back onto the prefill chain.
    #[test]
    fn the_width_bound_excludes_decode_without_help_from_the_token_count_clause() {
        // A `const` block, so this is a *compile* error rather than a test
        // failure — the constant is known at build time and there is no reason
        // to let a build that violates the invariant exist at all.
        const {
            assert!(
                MIN_FUSED_TOKENS > 1,
                "MIN_FUSED_TOKENS would admit a decode step if the \
                 `n_tokens > 1` clause were ever removed"
            );
        }
    }

    /// The two thresholds answer different questions and were swept
    /// separately, but their *order* is a measured fact: fusing attention into
    /// a longer chain pays at a narrower batch than bare GPU attention does.
    /// If a future sweep inverted them, every width between the two would be
    /// taking a path neither sweep found best, and nothing else would say so.
    #[test]
    fn fusing_pays_off_at_a_narrower_batch_than_bare_gpu_attention() {
        assert!(
            MIN_FUSED_TOKENS < crate::engine::attention::min_gpu_tokens(),
            "fusion width {MIN_FUSED_TOKENS} is not below the bare-attention \
             threshold {}; one of the two sweeps is stale",
            crate::engine::attention::min_gpu_tokens()
        );
    }
}

#[cfg(test)]
mod real_model_tests {
    use super::*;
    use crate::engine::loader::PoolingType;

    /// End-to-end greedy decode against a real Llama-3-family `.gguf`
    /// (`bartowski/Llama-3.2-1B-Instruct-GGUF` and its 3B sibling are what
    /// this was verified against, every published quantization of each).
    ///
    /// This file's forward pass is shared by five `general.architecture`
    /// values, so most of it is already exercised by other models — what a
    /// Llama-3.2 checkpoint specifically adds is a head dimension that is
    /// *not* implied by the usual `n_embd / n_head` reading of a GGUF
    /// (`llama.attention.key_length`/`value_length` are set explicitly),
    /// tied output embeddings on the 1B, and an `IQ3_S`-carrying
    /// quantization (`IQ3_M`) that no other model in this suite reaches.
    ///
    /// The assertion is on *text*, not logits, for the reason spelled out in
    /// `arch::phi`'s equivalent test: a wrong-but-plausible forward pass
    /// produces fluent output that a tolerance-based logit check can be
    /// talked into accepting, while a factual one-word answer cannot survive
    /// a genuinely broken attention or FFN.
    ///
    /// The prompt comes from the model's own `tokenizer.chat_template`,
    /// rendered the way `http::openai` renders it (`add_generation_prompt`,
    /// the vocab's own BOS/EOS text, then `encode(.., add_bos: false)`
    /// because the template emits `{{- bos_token }}` itself). Hand-writing
    /// the `<|start_header_id|>` framing instead is what an earlier version
    /// of this test did, and it is *wrong in a way that looks right*: Llama
    /// 3.2's template unconditionally injects a `system` block ("Cutting
    /// Knowledge Date… Today Date…") even when the caller passes no system
    /// message, and without it the 1B answers "What is the capital of
    /// France?" with `"I hope that the capital\nThe capital\nThe"` — a
    /// failure that looks exactly like a broken forward pass but is purely
    /// a missing preamble. Rendering the real template is also what the
    /// server does, so this exercises the path users actually hit.
    ///
    /// Run with `ORANGU_TEST_LLAMA_MODEL=/path/to/Llama-3.2-1B-Instruct-Q4_K_M.gguf
    /// cargo test --release --bin orangu-server real_model_tests -- --ignored`.
    #[test]
    #[ignore]
    fn llama3_answers_a_factual_question() {
        let path = std::env::var("ORANGU_TEST_LLAMA_MODEL").expect("set ORANGU_TEST_LLAMA_MODEL");
        let loaded = LoadedModel::open(std::path::Path::new(&path)).expect("load model");
        // Any architecture this module serves — the test is about the
        // shared forward pass, and `mistral3` exercises strictly more of it
        // (YaRN, a head_dim that isn't n_embd/n_head, temperature scaling).
        assert!(
            matches!(
                loaded.config.architecture.as_str(),
                "llama" | "mistral" | "qwen2" | "qwen3" | "qwen3vl"
            ),
            "unexpected architecture {}",
            loaded.config.architecture
        );
        let gguf = orangu::gguf::GgufFile::open(std::path::Path::new(&path)).expect("open gguf");
        let tokenizer =
            crate::engine::tokenizer::Tokenizer::from_gguf(&gguf).expect("build tokenizer");
        let model =
            LlamaModel::load_with_backend(&loaded, Arc::new(crate::engine::backend::CpuBackend))
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
                crate::engine::chat_template::Reasoning::default(),
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

    /// Cross-check against real llama.cpp (`mradermacher/Qwen3-VL-
    /// Embedding-8B-GGUF:Q4_K_M`, `llama-server --embedding --pooling
    /// last`): tokenizing "The quick brown fox jumps over the lazy dog"
    /// with `add_special=true` gives `[785, 3974, 13876, 38835, 34208, 916,
    /// 279, 15678, 5562, 151643]` — no BOS (`qwen3vl`'s `tokenizer.ggml.
    /// add_bos_token` is `false`, unlike every other model this engine has
    /// been tested against) but *does* get a trailing EOS (151643,
    /// `add_eos_token = true`) — real llama.cpp's `LLAMA_POOLING_TYPE_LAST`
    /// pools whatever the actual last position is, so it's pooling the
    /// *EOS* token's hidden state here, not "dog"'s (the first version of
    /// this test used only the 9 content tokens, no EOS, and — pooling the
    /// wrong position entirely — got a real, wrong 0.15 cosine; this list
    /// must match `Tokenizer::encode_for_embedding`'s actual output
    /// exactly, not just the content tokens).
    ///
    /// Also exercises `Tokenizer::encode_for_embedding`'s BOS handling:
    /// an earlier version hardcoded `add_bos: true`, silently prepending a
    /// token real llama.cpp never adds for this model, and *that* bug
    /// alone (independent of the EOS one above) dropped cosine similarity
    /// to real llama.cpp's own embedding to ~0.47.
    ///
    /// This is the *last transformer hidden state* (`Self::run_layers`'s
    /// output, post-`output_norm`, no `lm_head`) at the final token
    /// position, L2-normalized — `LLAMA_POOLING_TYPE_LAST`, matching
    /// `PoolingType::Last`'s own dispatch in `http::openai::
    /// pooled_embedding`. Exercises this file's Q/K-norm addition (`Self::
    /// run_layers`'s `q_norm`/`k_norm` handling) and confirms M-RoPE
    /// degenerates to plain single-position RoPE for text-only input, as
    /// argued in `engine::loader`'s own `LLAMA_STYLE_ARCHITECTURES` doc
    /// comment. Run with `ORANGU_TEST_QWEN3VL_MODEL=/path/to/Qwen3-VL-
    /// Embedding-8B.Q4_K_M.gguf cargo test --release --bin orangu-server
    /// real_model_tests -- --ignored`.
    #[test]
    #[ignore]
    fn qwen3vl_embedding_matches_real_llama_cpp() {
        let path =
            std::env::var("ORANGU_TEST_QWEN3VL_MODEL").expect("set ORANGU_TEST_QWEN3VL_MODEL");
        let loaded = LoadedModel::open(std::path::Path::new(&path)).expect("load model");
        assert_eq!(loaded.config.architecture, "qwen3vl");
        assert_eq!(loaded.config.pooling_type, PoolingType::Last);
        let model =
            LlamaModel::load_with_backend(&loaded, Arc::new(crate::engine::backend::CpuBackend))
                .expect("build model");

        let tokens: Vec<u32> = vec![
            785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 151643,
        ];
        let n_embd = model.config().n_embd;
        let hidden = model
            .forward_hidden_states(&tokens)
            .expect("forward_hidden_states");
        assert_eq!(hidden.len(), tokens.len() * n_embd);

        let mut pooled = hidden[(tokens.len() - 1) * n_embd..].to_vec();
        let norm = pooled.iter().map(|v| v * v).sum::<f32>().sqrt();
        for v in pooled.iter_mut() {
            *v /= norm;
        }

        let Some(csv) =
            crate::engine::arch::read_reference_fixture("qwen3vl_embedding_reference.csv")
        else {
            return;
        };
        let reference: Vec<f32> = csv
            .trim()
            .split(',')
            .map(|v| v.parse().expect("reference fixture value"))
            .collect();
        assert_eq!(
            reference.len(),
            n_embd,
            "reference fixture has wrong length"
        );

        let cosine: f32 = pooled.iter().zip(&reference).map(|(a, b)| a * b).sum();
        assert!(
            cosine > 0.99,
            "cosine similarity to real llama.cpp's embedding was only {cosine}, expected > 0.99"
        );
    }
}
