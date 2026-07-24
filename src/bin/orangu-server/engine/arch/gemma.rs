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

//! Gemma-style forward pass, targeting `gemma4` (confirmed against real
//! upstream `llama.cpp` source — `src/models/gemma4.cpp`, fetched and read
//! directly, not guessed) as well as the simpler `gemma`/`gemma2`/`gemma3`
//! predecessors, whose hyperparameters are a subset of gemma4's.
//!
//! Substantially more involved than the Llama-style family
//! (`engine::arch::llama`) — per the real graph-building code, a gemma4
//! layer has:
//! - **QK-norm**: `attn_q_norm`/`attn_k_norm` (weighted RMSNorm) applied to
//!   Q/K per-head, before RoPE; V gets a *weightless* RMSNorm.
//! - **Per-layer-varying head dimension and RoPE**: SWA layers and
//!   full-attention layers use different head sizes, RoPE dimensions, and
//!   RoPE frequency bases (`attention.key_length` vs `.key_length_swa`,
//!   etc.) — not a single value for the whole model.
//! - **Cross-layer KV cache sharing**: the last `attention.shared_kv_layers`
//!   layers have no K/V projections of their own at all; they reuse the
//!   last layer before them that did.
//! - **Attention scale override**: `1.0`, not `1/sqrt(head_dim)`.
//! - **Dual sub-layer norms**: `attn_post_norm`/`ffn_post_norm` applied
//!   *after* each sub-layer, before its residual add (on top of the usual
//!   pre-norms).
//! - **Per-layer embeddings (PLE)**: a second embedding table
//!   (`per_layer_token_embd`), projected from the main hidden state,
//!   normed, gated, and added into *every* layer's residual stream — a
//!   mechanism with no equivalent anywhere else in this engine.
//! - **GEGLU FFN** (GELU, not SiLU) and **final logit softcapping**
//!   (`tanh`-based).
//! - **MoE FFN** (`gemma-4-26B-A4B`): a MoE layer (`ffn_gate_inp` present)
//!   runs *two* parallel FFN branches off the post-attention residual and
//!   sums them — a dense GEGLU "shared" MLP (this layer's always-present
//!   `ffn_gate`/`ffn_up`/`ffn_down`, its own `post_ffw_norm_1`) plus a
//!   routed-expert branch (`pre_ffw_norm_2` input norm, softmax top-k
//!   routing over `ffn_gate_*_exps`, renormalized, GELU experts, its own
//!   `post_ffw_norm_2`). The router logits are computed the way gemma4.cpp
//!   does — a *weightless* RMSNorm of the residual, `1/sqrt(n_embd)`-scaled
//!   and multiplied by the learned per-dim `ffn_gate_inp.scale`, then
//!   projected through `ffn_gate_inp` — reading the residual directly, not
//!   the expert branch's own pre-normed input. See [`GemmaModel::
//!   moe_ffn_result`].
//!
//! The gate+up experts come either fused (`ffn_gate_up_exps`, as in the QAT
//! checkpoint) or separate (`ffn_{gate,up}_exps`), each optionally carrying
//! a per-expert `.scale` companion (a QAT scalar folded in per
//! `build_lora_mm_id`); the `gemma-4-26B-A4B` QAT GGUF ships fused Q4_0
//! experts plus a per-expert `ffn_down_exps.scale`. Its `head_count_kv` also
//! varies per layer (full-attention layers use fewer KV heads than SWA
//! layers), read per [`GemmaLayer::n_head_kv`].
//!
//! A model with any MoE layer runs entirely through the CPU-orchestrated
//! forward paths ([`GemmaModel::run_layers_cpu`] and the CPU branch of
//! [`ModelForward::forward_batch_decode`]) — the matmuls still dispatch to
//! the GPU backend, but the fully-fused single-submission decode/replay/
//! batched Vulkan paths ([`GemmaModel::record_one_sequence_decode`] etc.)
//! are dense-FFN-only and are skipped when [`GemmaModel::is_moe`], the same
//! way [`super::qwen35moe`] (also MoE) is wholly CPU-orchestrated.

use anyhow::{Context, Result, bail};
use std::sync::Arc;
use std::time::Instant;

use super::{BatchDecodeItem, ForwardOutcome, GreedySampleParams, ModelForward};
use crate::engine::backend::vulkan::{
    FusedAttnProjection, FusedLayerInput, FusedPle, GpuArgmaxSampleInput, GpuInput, VulkanBackend,
};
use crate::engine::backend::vulkan_replay::{
    CaptureStep, ComputeProgram, ReplayContext, ReplayGraph,
};
use crate::engine::backend::{Backend, MatmulOp};
use crate::engine::kv_cache::KvCache;
use crate::engine::loader::{ExpertQuantMatrix, LoadedModel, ModelConfig, QuantMatrix};
use crate::engine::tensor;
use rayon::prelude::*;

/// State for the opt-in raw-Vulkan decode replay path (`ORANGU_REPLAY`):
/// the persistent command buffer captured from the first token's real
/// recording, replayed every subsequent token with no `wgpu` submit on the
/// forward itself. Built lazily on the first single-token decode.
struct DecodeReplay {
    ctx: ReplayContext,
    graph: ReplayGraph,
    /// Kept alive for the graph's lifetime (the persistent command buffer
    /// references their pipelines/descriptor sets).
    _programs: Vec<ComputeProgram>,
    /// The captured op-list — kept alive because it holds `wgpu::Buffer` clones
    /// of every buffer the graph's descriptor sets reference by raw handle.
    /// Some (e.g. `output_norm`'s scratch buffers) live *only* here, not in
    /// orangu's caches, so dropping this would free `VkBuffer`s still bound.
    _captured_steps: Vec<CaptureStep>,
    /// Every buffer the wgpu path fills from the scaled token embedding via a
    /// `GpuInput::Cpu` upload (layer-0 input, and the PLE projection input for
    /// PLE models) — this token's embedding is written to all of them each step.
    /// Uncaptured as GPU ops (see [`vulkan_replay::HostInputTag`]).
    embd_inputs: Vec<(wgpu::Buffer, u64)>,
    /// PLE models only: the PLE projection's gathered per-layer-embedding input,
    /// re-gathered and re-uploaded each token. Empty for non-PLE models.
    gathered_inputs: Vec<(wgpu::Buffer, u64)>,
    /// The `lm_head` output — read back after each `run_token` for sampling.
    logits_buf: wgpu::Buffer,
    logits_off: u64,
    n_vocab: usize,
    /// Identity of the `(KvCache, slot)` this graph was captured against — the
    /// graph binds that request's KV-cache and op-cache buffers by raw handle,
    /// so a different request (new cache object or slot) needs a fresh capture.
    cache_ptr: usize,
    slot_id: usize,
    /// The position the next replayed token must be at (`start_pos` increments by
    /// one per decode token within a sequence). A call whose `start_pos` isn't
    /// this — a *new* request that reused the same pooled `(KvCache, slot)`, so
    /// `cache_ptr` alone can't tell it apart — means the graph would replay at
    /// positions it wasn't built for, so it's rebuilt. Updated each token.
    expected_pos: usize,
}

/// The routed experts' gate+up projection. `gemma-4-26B-A4B` ships a
/// **fused** `ffn_gate_up_exps` (`[n_embd, 2*n_ff_exp, n_expert]`), whose
/// output rows `[0, n_ff_exp)` are the gate and `[n_ff_exp, 2*n_ff_exp)` the
/// up (matching gemma4.cpp's `ggml_view` split); the plain gemma4.cpp path
/// instead has separate `ffn_gate_exps`/`ffn_up_exps`. Both carry an
/// optional per-expert `.scale` companion (`[n_expert]`), a QAT scalar
/// multiplied into that expert's output *before* the GELU (per
/// `build_lora_mm_id`) — `None` when absent (the Q4_0 gate/up experts here
/// have inline scales and ship no companion).
enum GemmaExpertGateUp {
    Fused {
        gate_up: ExpertQuantMatrix,
        scale: Option<Vec<f32>>,
    },
    Separate {
        gate: ExpertQuantMatrix,
        up: ExpertQuantMatrix,
        gate_scale: Option<Vec<f32>>,
        up_scale: Option<Vec<f32>>,
    },
}

/// A gemma4 MoE layer's routed-expert branch (`gemma-4-26B-A4B`). The
/// dense "shared" MLP branch reuses the layer's always-present
/// `ffn_norm`/`ffn_gate`/`ffn_up`/`ffn_down`, so only the routed-expert-
/// specific tensors live here. See [`GemmaModel::moe_ffn_result`].
struct GemmaMoe {
    /// Router projection, `[n_embd, n_expert]`.
    gate_inp: QuantMatrix,
    /// `ffn_gate_inp.scale`, `[n_embd]` — a learned per-dim scale applied to
    /// the (weightless-RMSNormed, `1/sqrt(n_embd)`-scaled) router input
    /// before the `gate_inp` projection, per gemma4.cpp's custom router.
    gate_inp_scale: Vec<f32>,
    /// `pre_ffw_norm_2` — RMSNorm on the residual feeding the *experts*
    /// (distinct from the shared MLP's `ffn_norm`).
    pre_norm_2: Vec<f32>,
    /// `post_ffw_norm_1` — RMSNorm applied to the shared MLP branch's output.
    post_norm_1: Vec<f32>,
    /// `post_ffw_norm_2` — RMSNorm applied to the routed-expert branch's output.
    post_norm_2: Vec<f32>,
    gate_up: GemmaExpertGateUp,
    down_exps: ExpertQuantMatrix,
    /// `ffn_down_exps.scale`, `[n_expert]` — a per-expert QAT scalar
    /// multiplied into that expert's whole down-projection output (per
    /// `build_lora_mm_id`'s `w_s`). `None` when absent.
    down_scale: Option<Vec<f32>>,
}

struct GemmaLayer {
    attn_norm: Vec<f32>,
    wq: QuantMatrix,
    wk: Option<QuantMatrix>,
    wv: Option<QuantMatrix>,
    wo: QuantMatrix,
    attn_q_norm: Vec<f32>,
    attn_k_norm: Option<Vec<f32>>,
    attn_post_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    ffn_gate: QuantMatrix,
    ffn_up: QuantMatrix,
    ffn_down: QuantMatrix,
    ffn_post_norm: Vec<f32>,
    /// `Some` only for MoE layers (`gemma-4-26B-A4B`) — the routed-expert
    /// branch that runs alongside the dense FFN above. `None` (dense-only)
    /// for every other Gemma variant.
    moe: Option<GemmaMoe>,
    layer_output_scale: Option<f32>,
    per_layer_inp_gate: Option<QuantMatrix>,
    per_layer_proj: Option<QuantMatrix>,
    per_layer_post_norm: Option<Vec<f32>>,

    is_swa: bool,
    head_dim: usize,
    /// KV heads for *this* layer. Gemma4 can vary this per layer
    /// (`attention.head_count_kv` is an array — e.g. `gemma-4-26B-A4B`'s
    /// full-attention layers use 2, its SWA layers 8); a scalar (or absent)
    /// `head_count_kv` is broadcast to every layer. `n_head / n_head_kv`
    /// (this layer's GQA group size) must divide evenly.
    n_head_kv: usize,
    rope_dim: usize,
    rope_freq_base: f32,
    has_kv: bool,
    /// When `!has_kv`, the layer index whose KV cache this one reads from.
    kv_donor: usize,
}

pub struct GemmaModel {
    config: ModelConfig,
    backend: Arc<dyn Backend>,
    tok_embeddings: QuantMatrix,
    output_norm: Vec<f32>,
    output_weight: QuantMatrix,
    n_head: usize,
    n_swa: usize,
    /// Routed experts evaluated per token (`expert_used_count`) — `0` for
    /// dense models. Only read on the MoE path.
    n_expert_used: usize,
    /// `true` iff any layer carries a routed-expert branch
    /// ([`GemmaLayer::moe`]). Gates *off* the fully-fused single-submission
    /// Vulkan decode/replay/batched paths (dense-FFN-only), routing MoE
    /// models through the CPU-orchestrated forward instead.
    is_moe: bool,
    attention_scale: f32,
    final_logit_softcapping: Option<f32>,
    /// `false` only for `gemma-embedding` — every other Gemma family member
    /// is a causal decoder. Gates attention masking (causal window vs. full/
    /// symmetric-windowed bidirectional, see [`GemmaModel::run_layers_cpu`])
    /// and whether [`ModelForward::forward`] (generation) is even allowed.
    causal: bool,
    /// `gemma-embedding`'s sentence-transformers "Dense" adapter layers,
    /// applied to the *pooled* embedding by [`ModelForward::
    /// post_pool_projection`] — `None` for every other Gemma family member,
    /// and `None` here too unless the file was converted with
    /// `--sentence-transformers-dense-modules` (both tensors are optional
    /// in upstream `llama.cpp`, `TENSOR_NOT_REQUIRED`).
    dense_2: Option<QuantMatrix>,
    dense_3: Option<QuantMatrix>,
    /// Shared across every full-attention (non-SWA) layer — one tensor in
    /// the file, per `llama.cpp`'s `TENSOR_DUPLICATED` handling.
    rope_freqs: Option<Vec<f32>>,
    n_embd_per_layer: usize,
    per_layer_tok_embd: Option<QuantMatrix>,
    per_layer_model_proj: Option<QuantMatrix>,
    per_layer_proj_norm: Option<Vec<f32>>,
    layers: Vec<GemmaLayer>,
    /// Opt-in raw-Vulkan decode replay (`ORANGU_REPLAY`), built lazily on the
    /// first single-token decode. `None` until then / when disabled.
    decode_replay: std::sync::Mutex<Option<DecodeReplay>>,
}

impl GemmaModel {
    pub fn load_with_backend(loaded: &LoadedModel, backend: Arc<dyn Backend>) -> Result<Self> {
        let config = loaded.config.clone();
        let n_layer = config.n_layer;

        let n_head = loaded
            .metadata_u64("attention.head_count")
            .context("missing attention.head_count")? as usize;
        // `attention.head_count_kv` is a *scalar* for most Gemma variants but
        // a per-layer *array* for `gemma-4-26B-A4B` (full-attention layers use
        // fewer KV heads than SWA layers). Read both: the array wins per layer
        // when present, else the scalar (defaulting to `n_head`) is broadcast.
        let n_head_kv_default = loaded
            .metadata_u64("attention.head_count_kv")
            .unwrap_or(n_head as u64) as usize;
        let n_head_kv_per_layer = loaded.metadata_array_u64("attention.head_count_kv");
        let rms_eps = loaded
            .metadata_f32("attention.layer_norm_rms_epsilon")
            .unwrap_or(1e-6);
        let n_swa = loaded.metadata_u64("attention.sliding_window").unwrap_or(0) as usize;
        // `expert_used_count` — how many routed experts each token evaluates
        // (`gemma-4-26B-A4B`). `0`/absent for dense Gemma variants; a MoE
        // layer with this still `0` is rejected after the layer loop below.
        let n_expert_used = loaded.metadata_u64("expert_used_count").unwrap_or(0) as usize;
        let final_logit_softcapping = loaded.metadata_f32("final_logit_softcapping");
        let n_embd_per_layer = loaded
            .metadata_u64("embedding_length_per_layer_input")
            .unwrap_or(0) as usize;

        let head_dim_full = loaded.metadata_u64("attention.key_length").unwrap_or(0) as usize;
        let head_dim_swa = loaded
            .metadata_u64("attention.key_length_swa")
            .unwrap_or(head_dim_full as u64) as usize;
        let rope_dim_full = loaded
            .metadata_u64("rope.dimension_count")
            .unwrap_or(head_dim_full as u64) as usize;
        let rope_dim_swa = loaded
            .metadata_u64("rope.dimension_count_swa")
            .unwrap_or(rope_dim_full as u64) as usize;
        let rope_freq_base_full = loaded.metadata_f32("rope.freq_base").unwrap_or(10000.0);
        let rope_freq_base_swa = loaded.metadata_f32("rope.freq_base_swa").unwrap_or(10000.0);

        let is_embedding_arch = config.architecture == "gemma-embedding";
        let is_swa: Vec<bool> = loaded
            .metadata_array_u64("attention.sliding_window_pattern")
            .map(|arr| arr.iter().map(|&v| v != 0).collect())
            .unwrap_or_else(|| {
                if is_embedding_arch {
                    // Upstream `llama.cpp`'s `src/models/gemma-embedding.cpp`
                    // hardcodes a period-6 SWA pattern (`swa_period = 6`)
                    // when this key is absent from the file — which it
                    // always is for `embeddinggemma-300M` (confirmed
                    // directly against the real GGUF's metadata dump: no
                    // `attention.sliding_window_pattern` key at all). Every
                    // 6th layer (last of each group of 6) is full attention,
                    // the rest SWA — `llama_hparams::set_swa_pattern`'s own
                    // formula, `dense_first = false`.
                    (0..n_layer).map(|il| il % 6 < 5).collect()
                } else {
                    vec![false; n_layer]
                }
            });
        let n_shared_kv_layers = loaded
            .metadata_u64("attention.shared_kv_layers")
            .unwrap_or(0) as usize;
        let n_layer_kv_from_start = n_layer.saturating_sub(n_shared_kv_layers);

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

        let rope_freqs = loaded.tensor("rope_freqs.weight").ok().map(|(v, _)| v);

        // `gemma-embedding`'s sentence-transformers Dense adapters —
        // `TENSOR_NOT_REQUIRED` upstream, so a model converted without
        // `--sentence-transformers-dense-modules` simply lacks them.
        let dense_2 = loaded
            .has_tensor("dense_2.weight")
            .then(|| loaded.matrix("dense_2.weight"))
            .transpose()
            .context("loading dense_2.weight")?;
        let dense_3 = loaded
            .has_tensor("dense_3.weight")
            .then(|| loaded.matrix("dense_3.weight"))
            .transpose()
            .context("loading dense_3.weight")?;

        let n_embd_per_layer_total = n_embd_per_layer * n_layer;
        let per_layer_tok_embd = if n_embd_per_layer > 0 {
            Some(
                loaded
                    .matrix("per_layer_token_embd.weight")
                    .context("loading per_layer_token_embd.weight")?,
            )
        } else {
            None
        };
        let per_layer_model_proj = if n_embd_per_layer > 0 {
            Some(
                loaded
                    .matrix("per_layer_model_proj.weight")
                    .context("loading per_layer_model_proj.weight")?,
            )
        } else {
            None
        };
        let per_layer_proj_norm = if n_embd_per_layer > 0 {
            Some(
                loaded
                    .tensor("per_layer_proj_norm.weight")
                    .context("loading per_layer_proj_norm.weight")?
                    .0,
            )
        } else {
            None
        };
        let _ = n_embd_per_layer_total; // used by callers via n_embd_per_layer * n_layer

        let mut layers = Vec::with_capacity(n_layer);
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
            let get_optional_matrix = |suffix: &str| -> Result<Option<QuantMatrix>> {
                let name = format!("blk.{i}.{suffix}");
                if !loaded.has_tensor(&name) {
                    return Ok(None);
                }
                Ok(Some(
                    loaded
                        .matrix(&name)
                        .with_context(|| format!("loading {name}"))?,
                ))
            };
            let get_expert_matrix = |suffix: &str| -> Result<ExpertQuantMatrix> {
                let name = format!("blk.{i}.{suffix}");
                loaded
                    .expert_matrix(&name)
                    .with_context(|| format!("loading {name}"))
            };
            // An optional `[n_expert]` (etc.) F32 companion scale tensor.
            let get_optional_vec = |suffix: &str| -> Result<Option<Vec<f32>>> {
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

            // MoE layer (`gemma-4-26B-A4B`): the presence of the router
            // (`ffn_gate_inp`) marks this layer as running the routed-expert
            // branch alongside the always-present dense FFN. See
            // [`GemmaModel::moe_ffn_result`] for the graph. The gate+up
            // experts are either fused (`ffn_gate_up_exps`, as in the QAT
            // checkpoint) or separate (`ffn_{gate,up}_exps`), each with an
            // optional per-expert `.scale` companion.
            let moe = if loaded.has_tensor(&format!("blk.{i}.ffn_gate_inp.weight")) {
                let gate_up = if loaded.has_tensor(&format!("blk.{i}.ffn_gate_up_exps.weight")) {
                    GemmaExpertGateUp::Fused {
                        gate_up: get_expert_matrix("ffn_gate_up_exps.weight")?,
                        scale: get_optional_vec("ffn_gate_up_exps.scale")?,
                    }
                } else {
                    GemmaExpertGateUp::Separate {
                        gate: get_expert_matrix("ffn_gate_exps.weight")?,
                        up: get_expert_matrix("ffn_up_exps.weight")?,
                        gate_scale: get_optional_vec("ffn_gate_exps.scale")?,
                        up_scale: get_optional_vec("ffn_up_exps.scale")?,
                    }
                };
                Some(GemmaMoe {
                    gate_inp: get_matrix("ffn_gate_inp.weight")?,
                    gate_inp_scale: get("ffn_gate_inp.scale")?,
                    pre_norm_2: get("pre_ffw_norm_2.weight")?,
                    post_norm_1: get("post_ffw_norm_1.weight")?,
                    post_norm_2: get("post_ffw_norm_2.weight")?,
                    gate_up,
                    down_exps: get_expert_matrix("ffn_down_exps.weight")?,
                    down_scale: get_optional_vec("ffn_down_exps.scale")?,
                })
            } else {
                None
            };

            let swa = is_swa.get(i).copied().unwrap_or(false);
            let has_kv = i < n_layer_kv_from_start;
            // Real llama.cpp's donor-layer formula (llama-model.cpp, the
            // GEMMA3N/GEMMA4 KV `reuse` callback): a non-KV layer reuses the
            // *last KV-owning layer of its own attention type* (SWA and
            // full-attention layers have different head dims/RoPE params, so
            // a SWA layer can't reuse a full-attention layer's cache or vice
            // versa) — `n_layer_kv_from_start - (is_swa(il) ? 2 : 1)`, keyed
            // off the *current* (donee) layer's own SWA-ness, not a single
            // fixed donor for every non-KV layer.
            let kv_donor = if has_kv {
                i
            } else if swa {
                n_layer_kv_from_start.saturating_sub(2)
            } else {
                n_layer_kv_from_start.saturating_sub(1)
            };

            layers.push(GemmaLayer {
                attn_norm: get("attn_norm.weight")?,
                wq: get_matrix("attn_q.weight")?,
                wk: if has_kv {
                    get_optional_matrix("attn_k.weight")?
                } else {
                    None
                },
                wv: if has_kv {
                    get_optional_matrix("attn_v.weight")?
                } else {
                    None
                },
                wo: get_matrix("attn_output.weight")?,
                attn_q_norm: get("attn_q_norm.weight")?,
                attn_k_norm: if has_kv {
                    get_optional("attn_k_norm.weight")?
                } else {
                    None
                },
                attn_post_norm: get("post_attention_norm.weight")?,
                ffn_norm: get("ffn_norm.weight")?,
                ffn_gate: get_matrix("ffn_gate.weight")?,
                ffn_up: get_matrix("ffn_up.weight")?,
                ffn_down: get_matrix("ffn_down.weight")?,
                ffn_post_norm: get("post_ffw_norm.weight")?,
                moe,
                layer_output_scale: get_optional("layer_output_scale.weight")?.map(|v| v[0]),
                per_layer_inp_gate: if n_embd_per_layer > 0 {
                    Some(get_matrix("inp_gate.weight")?)
                } else {
                    None
                },
                per_layer_proj: if n_embd_per_layer > 0 {
                    Some(get_matrix("proj.weight")?)
                } else {
                    None
                },
                per_layer_post_norm: if n_embd_per_layer > 0 {
                    Some(get("post_norm.weight")?)
                } else {
                    None
                },
                is_swa: swa,
                head_dim: if swa { head_dim_swa } else { head_dim_full },
                n_head_kv: n_head_kv_per_layer
                    .as_ref()
                    .and_then(|a| a.get(i).copied())
                    .map(|v| v as usize)
                    .unwrap_or(n_head_kv_default),
                rope_dim: if swa { rope_dim_swa } else { rope_dim_full },
                rope_freq_base: if swa {
                    rope_freq_base_swa
                } else {
                    rope_freq_base_full
                },
                has_kv,
                kv_donor,
            });
        }

        let is_moe = layers.iter().any(|l| l.moe.is_some());
        if is_moe && n_expert_used == 0 {
            bail!(
                "MoE gemma model (layers carry ffn_gate_inp) is missing \
                 {}.expert_used_count",
                config.architecture
            );
        }

        Ok(Self {
            config,
            backend,
            tok_embeddings,
            output_norm,
            output_weight,
            n_head,
            n_swa,
            n_expert_used,
            is_moe,
            // Gemma4 uses self.scaling = 1.0 (no 1/sqrt(head_dim) scaling).
            // `gemma-embedding` is the one exception: `hparams.
            // f_attention_scale = 1/sqrt(n_embd_head_k)`, applied via an
            // explicit `ggml_scale` on Q in upstream `llama.cpp`'s
            // `src/models/gemma-embedding.cpp` (confirmed directly against
            // that file, not guessed).
            attention_scale: if is_embedding_arch {
                1.0 / (head_dim_full as f32).sqrt()
            } else {
                1.0
            },
            final_logit_softcapping,
            causal: !is_embedding_arch,
            dense_2,
            dense_3,
            rope_freqs,
            n_embd_per_layer,
            per_layer_tok_embd,
            per_layer_model_proj,
            per_layer_proj_norm,
            layers,
            decode_replay: std::sync::Mutex::new(None),
        })
        .inspect(|_: &Self| {
            let _ = rms_eps; // used inline below via self.config.rms_eps override per layer call sites
        })
    }

    fn rms_eps(&self) -> f32 {
        self.config.rms_eps
    }

    /// Per-layer KV cache dimensions (`n_head_kv * head_dim`, that layer's
    /// own SWA-or-full head size) — passed to [`KvCache::new_with_dims`].
    fn kv_dims(&self) -> Vec<usize> {
        self.layers
            .iter()
            .map(|l| l.n_head_kv * l.head_dim)
            .collect()
    }

    /// Records a decode-shaped (`n_tokens == 1`) full-forward pass — PLE
    /// input projection (if this model has one), every layer, `output_norm`,
    /// `lm_head` — into one fresh command encoder, *not yet submitted*,
    /// returning the encoder plus the GPU-resident, not-yet-read-back
    /// `[n_vocab]` logits buffer. This is every layer's `record_fused_layer`
    /// plus `record_output_norm`/`record_full_matmul` chained into one
    /// command encoder
    /// with the residual stream threaded GPU-resident from one layer
    /// straight into the next, so nothing bounces back to the CPU between
    /// layers.
    ///
    /// Shared by two callers: `Self::forward`'s decode branch submits the
    /// returned encoder immediately and reads back the full logits vector
    /// (the general case — any sampling strategy, any caller); `Self::
    /// forward_maybe_sampling`'s GPU-argmax fast path instead appends one
    /// more dispatch (`VulkanBackend::record_argmax_sample`) *before*
    /// submitting, and reads back a single token id instead of the whole
    /// vector.
    ///
    /// `x` is the caller's already-computed, already-`sqrt(n_embd)`-scaled
    /// embedding row for `token` (shared prep work `Self::forward` also
    /// needs for its own CPU-orchestrated `else` branch, so it stays
    /// computed once, outside this method, rather than recomputed here);
    /// `token` itself is still needed separately for the per-layer-
    /// embedding gather, which does its own independent lookup into a
    /// *different* embedding table.
    /// How many `queue.submit()` calls one decode step's layer loop is
    /// split across (`ORANGU_DECODE_CHUNKS`; see
    /// `record_one_sequence_decode`). Read once and cached. Clamped to
    /// `1..=n_layers` — `1` submits the whole token once, and no more than
    /// one submit per layer is meaningful. A malformed value falls back to
    /// the default rather than erroring a live decode. More chunks overlap
    /// more of the CPU-side submission cost with GPU execution but add a
    /// little per-submission barrier overhead, so the default sits below one
    /// submit per layer.
    fn decode_submit_chunks(n_layers: usize) -> usize {
        // The CPU↔GPU submission overlap this buys saturates early:
        // throughput climbs as chunks go from 1 to 3 and is flat from 3
        // upward. `3` sits at that knee, so it keeps the full overlap while
        // paying only 3 `queue.submit()` calls (and allocating only 3 command
        // encoders) per token — cutting the per-token `vkQueueSubmit` *and*
        // `radv_BeginCommandBuffer`-`memset` cost, both of which scale with
        // the submitted-command-buffer count.
        const DEFAULT_CHUNKS: usize = 3;
        static CHUNKS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        let requested = *CHUNKS.get_or_init(|| {
            std::env::var("ORANGU_DECODE_CHUNKS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&n| n >= 1)
                .unwrap_or(DEFAULT_CHUNKS)
        });
        requested.clamp(1, n_layers.max(1))
    }

    fn record_decode_forward(
        &self,
        vulkan: &VulkanBackend,
        cache: &mut KvCache,
        token: u32,
        start_pos: usize,
        x: &[f32],
        slot_id: usize,
    ) -> Result<(wgpu::CommandEncoder, wgpu::Buffer, u64)> {
        let mut encoder = vulkan.new_encoder("orangu-server full forward encoder");

        // See `VulkanBackend::gpu_timestamps`'s own doc comment for what
        // this measures and `ORANGU_GPU_TIMESTAMPS=1` to enable it —
        // `timestamps` is `None` (and every `write_timestamp` below a
        // no-op) unless it's set. Fetched once per decode step, not
        // cached across steps, since the query set itself is what's
        // cached (`VulkanBackend::timestamp_query_set` — cheap to clone,
        // built once for the model's lifetime). Single-sequence-only: see
        // `record_one_sequence_decode`'s own doc comment for why a batched
        // decode step's own timing isn't captured this same way (yet).
        let timestamps = vulkan
            .gpu_timestamps()
            .then(|| vulkan.timestamp_query_set(self.layers.len()));
        if let Some(t) = &timestamps {
            encoder.write_timestamp(t, 0);
        }

        let (logits_buf, logits_offset) = self.record_one_sequence_decode(
            vulkan,
            &mut encoder,
            cache,
            token,
            start_pos,
            x,
            slot_id + 1,
            timestamps.as_ref(),
        );

        if let Some(t) = &timestamps {
            encoder.write_timestamp(t, (2 + self.layers.len()) as u32);
            vulkan.finish_timestamps(&mut encoder);
        }
        Ok((encoder, logits_buf, logits_offset))
    }

    /// One sequence's whole decode step — PLE projection, every layer,
    /// `output_norm`, `lm_head` — recorded into the caller's own `encoder`
    /// (does **not** create or submit one) at `batch_slot`, returning the
    /// GPU buffer holding this sequence's own `[n_vocab]` logits. The
    /// recording half of [`Self::record_decode_forward`] (`batch_slot ==
    /// this request's own `SlotGuard::id() + 1` — see [`BatchDecodeItem::
    /// slot_id`]'s doc comment for why a shared constant here would let two
    /// `slots > 1` requests decoding concurrently corrupt each other's
    /// cached GPU buffers) *and* [`Self::record_batched_decode_
    /// forward`] (`batch_slot` likewise each item's own `slot_id + 1`, one
    /// call per sequence in the batch, all sharing *one* encoder/submission
    /// — see that method's own doc comment for why `batch_slot` has to
    /// differ per sequence at all, not just per caller). `timestamps`, unlike
    /// `record_decode_forward`'s own copy, is only ever threaded through
    /// from the single-sequence caller (`Some` there iff `ORANGU_GPU_
    /// TIMESTAMPS=1`) — `record_batched_decode_forward` always passes
    /// `None`: `timestamp_query_set`'s own `wgpu::QuerySet` is sized for
    /// exactly one sequence's `n_layer + 3` boundary points, with no batch
    /// dimension, and a shared query set written from *M* concurrently-
    /// recorded sequences' worth of `write_timestamp` calls into the same
    /// fixed slots would just overwrite each other's timings, not add a
    /// useful per-sequence
    /// breakdown — a real per-sequence batched-decode timing breakdown
    /// would need its own, wider query set, not implemented here.
    #[allow(clippy::too_many_arguments)]
    fn record_one_sequence_decode(
        &self,
        vulkan: &VulkanBackend,
        encoder: &mut wgpu::CommandEncoder,
        cache: &mut KvCache,
        token: u32,
        start_pos: usize,
        x: &[f32],
        batch_slot: usize,
        timestamps: Option<&wgpu::QuerySet>,
    ) -> (wgpu::Buffer, u64) {
        let n_embd = self.config.n_embd;
        let eps = self.rms_eps();
        let per_layer = self.n_embd_per_layer;
        // `ORANGU_SKIP_PLE=1` is a **reduce-work diagnostic** (WRONG output —
        // gemma-3n needs PLE): skips the per-token PLE projection *and* every
        // layer's PLE sub-chain (ple_gate matmul + gelu + mul + ple_proj matmul
        // + norm), so the throughput delta vs default isolates PLE's
        // recoverable cost. Not a correctness knob.
        let has_ple = per_layer > 0 && std::env::var_os("ORANGU_SKIP_PLE").is_none();

        let ple_buf = if has_ple {
            let gathered = self.gather_per_layer_tok_embd(&[token], 1);
            Some(
                vulkan.record_ple_projection(
                    encoder,
                    crate::engine::backend::vulkan::PleProjectionInput {
                        x: GpuInput::Cpu(x),
                        proj_w: self
                            .per_layer_model_proj
                            .as_ref()
                            .expect("has_ple implies per_layer_model_proj is Some"),
                        proj_norm: self
                            .per_layer_proj_norm
                            .as_ref()
                            .expect("has_ple implies per_layer_proj_norm is Some"),
                        gathered: &gathered,
                        n_layer: self.layers.len(),
                        per_layer,
                        eps,
                    },
                    batch_slot,
                ),
            )
        } else {
            None
        };
        if let Some(t) = timestamps {
            encoder.write_timestamp(t, 1);
        }

        // Number of `queue.submit()` calls the layer loop is split across
        // this decode step (`ORANGU_DECODE_CHUNKS`). `1` submits the whole
        // token once; `> 1` submits the first `chunks - 1` groups of layers
        // as soon as they're recorded (`VulkanBackend::submit_intermediate`),
        // so the GPU starts executing early chunks while the CPU is still
        // recording and paying `wgpu-core`'s per-submission validation cost
        // for the later ones — overlapping the CPU submission cost with GPU
        // execution instead of serialising it in front of one end-of-token
        // submit.
        let n_layers = self.layers.len();
        let chunks = Self::decode_submit_chunks(n_layers);
        let layers_per_chunk = n_layers.div_ceil(chunks);

        let mut prev_buf: Option<(wgpu::Buffer, u64)> = None;
        for (il, layer) in self.layers.iter().enumerate() {
            let head_dim = layer.head_dim;
            // Proportional RoPE (a learned per-frequency divisor) only
            // applies to full-attention layers, matching gemma4.cpp's
            // `if (!hparams.is_swa(il)) { freq_factors = ...rope_freqs; }`.
            let freq_factors = (!layer.is_swa)
                .then_some(self.rope_freqs.as_deref())
                .flatten();
            let cache_index = layer.kv_donor;
            let pos = start_pos;
            let window_start = if layer.is_swa && self.n_swa > 0 {
                pos.saturating_sub(self.n_swa - 1)
            } else {
                0
            };
            // Raw-replay capture: SWA layers carry their window size so the
            // replay recomputes `n_pos`/`window_start` each token (see
            // `FusedAttnInput::window`); full-attention layers pass `None`.
            let window = (layer.is_swa && self.n_swa > 0).then_some(self.n_swa as u32);
            let kv = layer.has_kv.then(|| FusedAttnProjection {
                wk: layer
                    .wk
                    .as_ref()
                    .expect("layer has_kv but no attn_k.weight"),
                k_norm: layer
                    .attn_k_norm
                    .as_ref()
                    .expect("layer has_kv but no attn_k_norm"),
                wv: layer.wv.as_ref(),
            });
            // `il`'s per-layer-embedding slice, read straight out of
            // `ple_buf` (`VulkanBackend::record_ple_projection`'s
            // `[n_layer, per_layer]` output) at a `GpuInput` offset —
            // no copy, no per-token CPU slicing. Only valid at `n_tokens
            // == 1`, which every caller of this method already guarantees.
            // The step-by-step CPU path (`Self::forward`'s `else` branch)
            // needs a *different* slice per token, so it re-derives its
            // own per-`t` CPU slice inside its own loop instead of reusing
            // this.
            let ple = if let (Some(ple_buf), Some(gate_w), Some(proj_w), Some(post_norm)) = (
                &ple_buf,
                &layer.per_layer_inp_gate,
                &layer.per_layer_proj,
                &layer.per_layer_post_norm,
            ) {
                Some(FusedPle {
                    gate_w,
                    proj_w,
                    post_norm,
                    per_layer_slice: GpuInput::Gpu(ple_buf, il * per_layer),
                    per_layer_dim: per_layer,
                })
            } else {
                None
            };

            let x_input = match &prev_buf {
                Some((buf, offset)) => GpuInput::Gpu(buf, (*offset / 4) as usize),
                None => GpuInput::Cpu(x),
            };
            let out = vulkan.record_fused_layer(
                encoder,
                FusedLayerInput {
                    x: x_input,
                    attn_norm: &layer.attn_norm,
                    wq: &layer.wq,
                    q_norm: &layer.attn_q_norm,
                    kv,
                    n_head: self.n_head,
                    n_head_kv: layer.n_head_kv,
                    head_dim,
                    rope_dim: layer.rope_dim,
                    rope_freq_base: layer.rope_freq_base,
                    freq_factors,
                    eps,
                    pos,
                    window_start,
                    window,
                    scale: self.attention_scale,
                    cache: &mut cache.layers[cache_index],
                    wo: &layer.wo,
                    attn_post_norm: &layer.attn_post_norm,
                    ffn_norm: &layer.ffn_norm,
                    ffn_gate: &layer.ffn_gate,
                    ffn_up: &layer.ffn_up,
                    ffn_down: &layer.ffn_down,
                    ffn_post_norm: &layer.ffn_post_norm,
                    ple,
                    layer_output_scale: layer.layer_output_scale,
                    batch_slot,
                    // Per-op timestamp bracket for this layer's attention
                    // dispatch: two slots per layer past the existing
                    // `n_layers + 3` per-layer slots (see
                    // `VulkanBackend::timestamp_query_set`/`report_timestamps`).
                    attn_ts: timestamps.map(|t| (t, (n_layers + 3 + 2 * il) as u32)),
                },
            );
            prev_buf = Some(out);
            if let Some(t) = timestamps {
                encoder.write_timestamp(t, (2 + il) as u32);
            }
            // Chunk boundary: submit everything recorded so far (including
            // this layer's end-of-layer timestamp, which is why the flush
            // follows the `write_timestamp` above) and continue recording
            // the next chunk into a fresh encoder. The already-submitted
            // work is now executing on the GPU. Skipped for the final layer
            // — its chunk carries `output_norm`/`lm_head` (and, on the
            // sampling path, argmax) and is returned unsubmitted so the
            // caller owns the terminal submit + readback. Timestamp writes
            // span the fresh encoders and are resolved once, on the final
            // one (`finish_timestamps`); every intermediate encoder is
            // submitted before that resolve executes, so the whole query set
            // is populated by then.
            if chunks > 1 && il + 1 < n_layers && (il + 1) % layers_per_chunk == 0 {
                let finished =
                    std::mem::replace(encoder, vulkan.new_encoder("orangu-server decode chunk"));
                vulkan.submit_intermediate(finished);
            }
        }
        let (last_buf, last_offset) =
            prev_buf.expect("a gemma4 model always has at least one layer");
        let normed_buf = vulkan.record_output_norm(
            encoder,
            GpuInput::Gpu(&last_buf, (last_offset / 4) as usize),
            &self.output_norm,
            eps,
            n_embd,
        );
        vulkan.record_full_matmul(
            encoder,
            GpuInput::Gpu(&normed_buf, 0),
            &self.output_weight,
            batch_slot,
        )
    }

    /// The CPU-orchestrated core of a Gemma forward pass — every layer,
    /// returning the pre-`output_norm` hidden state for every token
    /// (`[n_tokens, n_embd]`). Shared by [`ModelForward::forward`]'s own
    /// prefill/CPU-backend `else` branch (which then takes just the last
    /// token, norms it, and projects to vocab logits) and
    /// [`ModelForward::forward_hidden_states`] (which norms and returns
    /// *every* token, no logits projection) — mirrors `engine::arch::llama`'s
    /// `LlamaModel::run_layers` split for the same reason.
    ///
    /// `x0` is the caller's already-computed, already-`sqrt(n_embd)`-scaled
    /// embedding for every token in `tokens` (shared prep work `forward`'s
    /// top already does for its own GPU-branch use, so it isn't recomputed
    /// here); this method clones it into its own working copy since every
    /// layer mutates the residual stream in place.
    fn run_layers_cpu(
        &self,
        cache: &mut KvCache,
        x0: &[f32],
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<Vec<f32>> {
        let n_tokens = tokens.len();
        let n_embd = self.config.n_embd;
        let eps = self.rms_eps();
        let mut x = x0.to_vec();

        // Hoisted scratch, refilled (clear + extend/resize) each layer
        // rather than freshly allocated — the prefill's dominant CPU cost was
        // allocating and first-touching several ~n_tokens×n_embd clones per
        // layer (malloc + page-fault memset). After the first layer these
        // allocate nothing.
        let mut normed: Vec<f32> = Vec::with_capacity(x.len());
        let mut ffn_normed: Vec<f32> = Vec::with_capacity(x.len());
        let mut attn_out: Vec<f32> = Vec::new();

        let per_layer = self.n_embd_per_layer;
        let has_ple = per_layer > 0;
        let inp_per_layer = if has_ple {
            Some(self.compute_per_layer_inputs(&x, tokens, n_tokens))
        } else {
            None
        };

        // CPU-side wall-clock around each GPU submission this
        // (CPU-orchestrated) prefill path makes — unlike the fused decode
        // path, there's no single encoder/timestamp-query-set to instrument
        // here, but every `Backend::matmul`/`matmul_batch` call already
        // blocks (`device.poll(wait_indefinitely)`) until its own GPU work
        // finishes, so timing around the call is an accurate proxy for
        // that submission's own GPU time. Opt in with
        // `ORANGU_PREFILL_TRACE=1`; off by default (`eprintln!` per
        // submission is real overhead at high layer/token counts).
        static PREFILL_TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let prefill_trace =
            *PREFILL_TRACE.get_or_init(|| std::env::var_os("ORANGU_PREFILL_TRACE").is_some());

        for (il, layer) in self.layers.iter().enumerate() {
            let head_dim = layer.head_dim;
            let freq_factors = (!layer.is_swa)
                .then_some(self.rope_freqs.as_deref())
                .flatten();
            let cache_index = layer.kv_donor;
            let group_size = self.n_head / layer.n_head_kv;

            normed.clear();
            normed.extend_from_slice(&x);
            tensor::rmsnorm_inplace(&mut normed, &layer.attn_norm, n_tokens, n_embd, eps);

            let wk = layer.has_kv.then(|| {
                layer
                    .wk
                    .as_ref()
                    .context("layer has_kv but no attn_k.weight")
            });
            let wk = wk.transpose()?;
            let owns_v = layer.has_kv && layer.wv.is_some();

            let mut ops = vec![MatmulOp {
                x: &normed,
                n_tokens,
                w: &layer.wq,
            }];
            if let Some(wk) = wk {
                ops.push(MatmulOp {
                    x: &normed,
                    n_tokens,
                    w: wk,
                });
            }
            if owns_v {
                ops.push(MatmulOp {
                    x: &normed,
                    n_tokens,
                    w: layer.wv.as_ref().unwrap(),
                });
            }
            let t0 = Instant::now();
            let mut results = self.backend.matmul_batch(&ops).into_iter();
            if prefill_trace {
                eprintln!(
                    "orangu-server: [prefill-trace] layer {il} qkv_matmul_batch \
                     n_tokens={n_tokens}: {:.1}ms",
                    t0.elapsed().as_secs_f64() * 1000.0
                );
            }
            let mut q = results.next().unwrap();
            tensor::rmsnorm_inplace(
                &mut q,
                &layer.attn_q_norm,
                n_tokens * self.n_head,
                head_dim,
                eps,
            );
            for t in 0..n_tokens {
                let pos = start_pos + t;
                tensor::rope_apply_scaled_inplace(
                    &mut q[t * self.n_head * head_dim..(t + 1) * self.n_head * head_dim],
                    self.n_head,
                    head_dim,
                    layer.rope_dim,
                    pos,
                    layer.rope_freq_base,
                    freq_factors,
                );
            }

            if layer.has_kv {
                let kv_dim = layer.n_head_kv * head_dim;
                let mut k = results.next().unwrap();
                tensor::rmsnorm_inplace(
                    &mut k,
                    layer
                        .attn_k_norm
                        .as_ref()
                        .context("layer has_kv but no attn_k_norm")?,
                    n_tokens * layer.n_head_kv,
                    head_dim,
                    eps,
                );
                let mut v = if owns_v {
                    results.next().unwrap()
                } else {
                    k.clone()
                };
                rmsnorm_weightless_inplace(&mut v, n_tokens * layer.n_head_kv, head_dim, eps);

                for t in 0..n_tokens {
                    let pos = start_pos + t;
                    tensor::rope_apply_scaled_inplace(
                        &mut k[t * kv_dim..(t + 1) * kv_dim],
                        layer.n_head_kv,
                        head_dim,
                        layer.rope_dim,
                        pos,
                        layer.rope_freq_base,
                        freq_factors,
                    );
                    cache.layers[cache_index].push(
                        &k[t * kv_dim..(t + 1) * kv_dim],
                        &v[t * kv_dim..(t + 1) * kv_dim],
                    );
                }
            }
            // Every token's K/V for this layer is already in `cache` by this
            // point (the push loop above ran for the full `0..n_tokens`
            // range before this loop starts reading), so a non-causal
            // model's attention window can freely include positions *after*
            // `pos`, not just up to it — see `Self::attention_window`.
            attn_out.clear();
            attn_out.resize(n_tokens * self.n_head * head_dim, 0.0);
            let t0 = Instant::now();
            // Prefill attention is O(n_tokens²) and single-threaded was the
            // second-largest prefill cost (after the GPU matmuls) and the one
            // that grows quadratically with prompt length. Each query token's
            // attention is independent — reads the (now fully-populated,
            // read-only) KV cache + `q`, writes only its own `attn_out` slice —
            // so parallelise across tokens with rayon (byte-exact: no cross-token
            // dependency, per-token accumulation order unchanged). Every thread
            // keeps its own `scores` scratch.
            let n_head = self.n_head;
            let attention_scale = self.attention_scale;
            let is_swa = layer.is_swa;
            let cache_layer = &cache.layers[cache_index];
            attn_out
                .par_chunks_mut(n_head * head_dim)
                .enumerate()
                .for_each(|(t, out_t)| {
                    let pos = start_pos + t;
                    let (window_start, window_end) = self.attention_window(is_swa, pos, n_tokens);
                    let mut scores: Vec<f32> = Vec::new();
                    for h in 0..n_head {
                        let kv_head = h / group_size;
                        let qh = &q[t * n_head * head_dim + h * head_dim
                            ..t * n_head * head_dim + (h + 1) * head_dim];

                        scores.clear();
                        for p in window_start..=window_end {
                            let kh = cache_layer.key_at(p, kv_head, head_dim);
                            scores.push(tensor::dot(qh, kh) * attention_scale);
                        }
                        tensor::softmax_inplace(&mut scores);

                        let out = &mut out_t[h * head_dim..(h + 1) * head_dim];
                        for (offset, &weight) in scores.iter().enumerate() {
                            let p = window_start + offset;
                            let vh = cache_layer.value_at(p, kv_head, head_dim);
                            for (o, vi) in out.iter_mut().zip(vh.iter()) {
                                *o += weight * vi;
                            }
                        }
                    }
                });
            if prefill_trace {
                eprintln!(
                    "orangu-server: [prefill-trace] layer {il} cpu_attention \
                     n_tokens={n_tokens}: {:.1}ms",
                    t0.elapsed().as_secs_f64() * 1000.0
                );
            }

            let t0 = Instant::now();
            let mut attn_proj = self.backend.matmul(&attn_out, n_tokens, &layer.wo);
            if prefill_trace {
                eprintln!(
                    "orangu-server: [prefill-trace] layer {il} wo_matmul \
                     n_tokens={n_tokens}: {:.1}ms",
                    t0.elapsed().as_secs_f64() * 1000.0
                );
            }
            tensor::rmsnorm_inplace(&mut attn_proj, &layer.attn_post_norm, n_tokens, n_embd, eps);
            tensor::add_inplace(&mut x, &attn_proj);

            // FFN. Dense (GEGLU) for most Gemma variants; a MoE layer
            // (`gemma-4-26B-A4B`) instead runs a dense shared MLP plus routed
            // experts and sums them (`moe_ffn_result`). Either way the shared
            // `ffn_post_norm` and the residual add follow.
            if let Some(moe) = &layer.moe {
                let mut ffn_out = self.moe_ffn_result(layer, moe, &x, n_tokens);
                tensor::rmsnorm_inplace(&mut ffn_out, &layer.ffn_post_norm, n_tokens, n_embd, eps);
                tensor::add_inplace(&mut x, &ffn_out);
            } else {
                // `x` is the post-attention residual and is *not* mutated
                // again until the FFN residual add below (the norm runs on the
                // `ffn_normed` copy, not `x`), so the old `attn_out_residual =
                // x.clone(); …; x = attn_out_residual` round-trip was a redundant
                // ~n_tokens×n_embd clone per layer — dropped. `ffn_normed` reuses a
                // hoisted scratch buffer instead of allocating a fresh clone.
                ffn_normed.clear();
                ffn_normed.extend_from_slice(&x);
                tensor::rmsnorm_inplace(&mut ffn_normed, &layer.ffn_norm, n_tokens, n_embd, eps);
                let t0 = Instant::now();
                let mut gate_up = self.backend.matmul_batch(&[
                    MatmulOp {
                        x: &ffn_normed,
                        n_tokens,
                        w: &layer.ffn_gate,
                    },
                    MatmulOp {
                        x: &ffn_normed,
                        n_tokens,
                        w: &layer.ffn_up,
                    },
                ]);
                if prefill_trace {
                    eprintln!(
                        "orangu-server: [prefill-trace] layer {il} gate_up_matmul_batch \
                         n_tokens={n_tokens}: {:.1}ms",
                        t0.elapsed().as_secs_f64() * 1000.0
                    );
                }
                let up = gate_up.pop().unwrap();
                let mut gate = gate_up.pop().unwrap();
                tensor::gelu_inplace(&mut gate);
                tensor::mul_inplace(&mut gate, &up);
                let t0 = Instant::now();
                let mut ffn_out = self.backend.matmul(&gate, n_tokens, &layer.ffn_down);
                if prefill_trace {
                    eprintln!(
                        "orangu-server: [prefill-trace] layer {il} ffn_down_matmul \
                         n_tokens={n_tokens}: {:.1}ms",
                        t0.elapsed().as_secs_f64() * 1000.0
                    );
                }
                tensor::rmsnorm_inplace(&mut ffn_out, &layer.ffn_post_norm, n_tokens, n_embd, eps);
                tensor::add_inplace(&mut x, &ffn_out);
            }

            if let (Some(inp_per_layer), Some(gate_w), Some(proj_w), Some(post_norm)) = (
                &inp_per_layer,
                &layer.per_layer_inp_gate,
                &layer.per_layer_proj,
                &layer.per_layer_post_norm,
            ) {
                // Same redundant-clone removal as the FFN residual
                // above — `x` (the post-FFN residual) is read by the PLE
                // matmuls below but never mutated until the `+= proj` add, so
                // the `pe_in = x.clone(); …; x = pe_in` round-trip was dropped.
                let t0 = Instant::now();
                let mut g = self.backend.matmul(&x, n_tokens, gate_w);
                if prefill_trace {
                    eprintln!(
                        "orangu-server: [prefill-trace] layer {il} ple_gate_matmul \
                         n_tokens={n_tokens}: {:.1}ms",
                        t0.elapsed().as_secs_f64() * 1000.0
                    );
                }
                tensor::gelu_inplace(&mut g);
                for t in 0..n_tokens {
                    let slice = &inp_per_layer[(t * self.layers.len() + il) * per_layer
                        ..(t * self.layers.len() + il + 1) * per_layer];
                    tensor::mul_inplace(&mut g[t * per_layer..(t + 1) * per_layer], slice);
                }
                let t0 = Instant::now();
                let mut proj = self.backend.matmul(&g, n_tokens, proj_w);
                if prefill_trace {
                    eprintln!(
                        "orangu-server: [prefill-trace] layer {il} ple_proj_matmul \
                         n_tokens={n_tokens}: {:.1}ms",
                        t0.elapsed().as_secs_f64() * 1000.0
                    );
                }
                tensor::rmsnorm_inplace(&mut proj, post_norm, n_tokens, n_embd, eps);
                tensor::add_inplace(&mut x, &proj);
            }

            if let Some(scale) = layer.layer_output_scale {
                for v in x.iter_mut() {
                    *v *= scale;
                }
            }
        }

        Ok(x)
    }

    /// The inclusive `[start, end]` key/value position range a query at
    /// absolute position `pos` may attend to, for a layer that either is or
    /// isn't SWA. Causal models (`self.causal`) attend backward-only, as
    /// generation requires — unchanged from before this method existed.
    /// `gemma-embedding`'s bidirectional attention (`!self.causal`) attends
    /// across the *whole* prompt on full-attention layers, or a *symmetric*
    /// window on SWA layers — confirmed directly against upstream
    /// `llama.cpp`'s `llama_hparams::is_masked_swa`'s `LLAMA_SWA_TYPE_
    /// SYMMETRIC` case: masked when `|p1 - p0| > n_swa/2`, i.e. a window of
    /// radius `n_swa/2` centered on the query position, not `n_swa`
    /// trailing positions the way causal SWA works.
    fn attention_window(&self, is_swa: bool, pos: usize, n_tokens: usize) -> (usize, usize) {
        if !self.causal {
            return if is_swa && self.n_swa > 0 {
                let half = self.n_swa / 2;
                (pos.saturating_sub(half), (pos + half).min(n_tokens - 1))
            } else {
                (0, n_tokens - 1)
            };
        }
        if is_swa && self.n_swa > 0 {
            (pos.saturating_sub(self.n_swa - 1), pos)
        } else {
            (0, pos)
        }
    }

    /// The GPU-resident batched-decode path: every sequence's own PLE/
    /// layer-stack/`output_norm`/`lm_head` chain (`Self::record_one_
    /// sequence_decode`) recorded into **one shared encoder**, at a
    /// distinct `batch_slot` per sequence (`1..=items.len()` — `0` is
    /// reserved for the single-sequence path, see `OpCacheKey`'s own doc
    /// comment for why two sequences, or a batched and an unbatched
    /// decode, can never safely share a `batch_slot`), submitted
    /// **once**, with every sequence's own `[n_vocab]` logits read back
    /// together (`VulkanBackend::submit_and_readback_batch`). This is
    /// what actually eliminates the CPU↔GPU round trips `Self::forward_
    /// batch_decode`'s own doc comment describes the plain `Backend::
    /// matmul`/`matmul_batch`-based path taking on every op of every
    /// layer: instead of that, this is **one** round trip for the
    /// *entire* batch's *entire* forward pass — the same one-round-trip
    /// shape `record_decode_forward` already gives a single sequence,
    /// just run `items.len()` times into the same encoder before
    /// submitting, rather than once per sequence with its own
    /// submission.
    ///
    /// Each sequence's own attention/RoPE/per-head-norm work stays
    /// genuinely per-sequence — recorded once per sequence, not widened
    /// into a single cross-sequence dispatch the way the plain-matmul
    /// path's QKV/`wo`/FFN projections already batch across sequences
    /// (see `Self::forward_batch_decode`'s own doc comment) — only the
    /// round trips *between* those per-sequence dispatches are
    /// eliminated here, by sharing one encoder/submission across the
    /// whole batch instead of one per weight per sequence. Never
    /// GPU-samples (matches `forward_batch_decode`'s own contract) —
    /// always returns raw logits, sampled by the caller (`engine::
    /// batch::BatchCoordinator`) on the CPU.
    fn record_batched_decode_forward(
        &self,
        vulkan: &VulkanBackend,
        items: &mut [BatchDecodeItem<'_>],
    ) -> Vec<Vec<f32>> {
        let n_embd = self.config.n_embd;
        let n_vocab = self.output_weight.out_dim;
        let mut encoder = vulkan.new_encoder("orangu-server batched decode encoder");

        let logits_bufs: Vec<(wgpu::Buffer, u64)> = items
            .iter_mut()
            .map(|item| {
                let mut x = self.tok_embeddings.row(item.token as usize);
                for v in x.iter_mut() {
                    *v *= (n_embd as f32).sqrt();
                }
                self.record_one_sequence_decode(
                    vulkan,
                    &mut encoder,
                    item.cache,
                    item.token,
                    item.start_pos,
                    &x,
                    item.slot_id + 1,
                    None,
                )
            })
            .collect();

        let sources: Vec<(&wgpu::Buffer, u64, usize)> = logits_bufs
            .iter()
            .map(|(buf, offset)| (buf, *offset, n_vocab))
            .collect();
        let mut logits = vulkan.submit_and_readback_batch(encoder, &sources);

        // Matches `forward`'s own tail — softcapping is applied to the
        // read-back logits there too, never inside the recording chain
        // itself.
        if let Some(cap) = self.final_logit_softcapping {
            for row in &mut logits {
                for v in row.iter_mut() {
                    *v = (*v / cap).tanh() * cap;
                }
            }
        }
        logits
    }
}

/// Reads `len` f32s back from a device-local wgpu buffer (the replay logits) —
/// a small transfer submit that references only the logits + readback, no
/// weights, so it doesn't reintroduce the per-token weight-VM cost.
fn read_gpu_f32(vulkan: &VulkanBackend, buf: &wgpu::Buffer, offset: u64, len: usize) -> Vec<f32> {
    let device = vulkan.wgpu_device();
    let queue = vulkan.wgpu_queue();
    let rb = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("orangu-server replay logits readback"),
        size: (len * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("orangu-server replay logits readback enc"),
    });
    enc.copy_buffer_to_buffer(buf, offset, &rb, 0, (len * 4) as u64);
    queue.submit(Some(enc.finish()));
    rb.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let out = {
        let view = rb
            .slice(..)
            .get_mapped_range()
            .expect("map logits readback");
        bytemuck::cast_slice::<u8, f32>(&view).to_vec()
    };
    rb.unmap();
    out
}

impl GemmaModel {
    /// The raw-Vulkan decode replay path (`ORANGU_REPLAY`). On the first
    /// single-token decode it captures orangu's real recording of the whole
    /// forward and builds a persistent command buffer; every subsequent token it
    /// writes this token's embedding into the captured first-layer buffer,
    /// patches the per-token uniforms (`pos`), resubmits the same command buffer
    /// (no `wgpu` submit on the forward), and reads the logits back. Returns the
    /// `[n_vocab]` logits for the caller to sample.
    fn decode_forward_replay(
        &self,
        vulkan: &VulkanBackend,
        cache: &mut KvCache,
        token: u32,
        start_pos: usize,
        greedy_sample: Option<&GreedySampleParams<'_>>,
        slot_id: usize,
    ) -> Result<ForwardOutcome> {
        let n_embd = self.config.n_embd;
        anyhow::ensure!(
            (token as usize) < self.config.n_vocab,
            "token id {token} out of vocab range"
        );
        let mut x = self.tok_embeddings.row(token as usize).to_vec();
        for v in x.iter_mut() {
            *v *= (n_embd as f32).sqrt();
        }

        let cache_ptr = std::ptr::from_ref::<KvCache>(cache) as usize;
        let mut guard = self.decode_replay.lock().expect("decode_replay poisoned");
        // Rebuild whenever this is a different request than the one captured: the
        // graph binds the captured request's KV-cache + op-cache buffers by raw
        // handle, so reusing it for another `(cache, slot)` would replay over the
        // wrong (or freed) memory. A fresh request re-captures on its first token.
        let stale = guard.as_ref().is_some_and(|r| {
            r.cache_ptr != cache_ptr || r.slot_id != slot_id || r.expected_pos != start_pos
        });
        if stale && let Some(old) = guard.take() {
            // Free the previous request's raw-Vulkan objects (fence, command/
            // descriptor pools, pipelines, per-token buffers) before rebuilding —
            // `ReplayContext` only clones wgpu's shared device/instance, so it
            // owns nothing to destroy.
            unsafe {
                old.graph.destroy(&old.ctx);
                for p in old._programs {
                    p.destroy(&old.ctx);
                }
            }
        }
        if guard.is_none() {
            // First token: record + submit the real wgpu forward while capturing
            // it, then build the replay graph from the captured op-list.
            vulkan.begin_decode_capture();
            let (encoder, logits_buf, logits_off) =
                self.record_decode_forward(vulkan, cache, token, start_pos, &x, slot_id)?;
            let logits = vulkan.submit_and_readback_for(encoder, &self.output_weight, slot_id + 1);
            let steps = vulkan
                .take_decode_capture()
                .context("ORANGU_REPLAY: no decode capture produced")?;
            anyhow::ensure!(!steps.is_empty(), "ORANGU_REPLAY: empty capture");
            use crate::engine::backend::vulkan_replay::HostInputTag;
            let host = crate::engine::backend::vulkan_replay::host_inputs(&steps);
            let embd_inputs: Vec<(wgpu::Buffer, u64)> = host
                .iter()
                .filter(|(t, ..)| *t == HostInputTag::EmbeddingX)
                .map(|(_, b, o, _)| (b.clone(), *o))
                .collect();
            let gathered_inputs: Vec<(wgpu::Buffer, u64)> = host
                .iter()
                .filter(|(t, ..)| *t == HostInputTag::Gathered)
                .map(|(_, b, o, _)| (b.clone(), *o))
                .collect();
            anyhow::ensure!(
                !embd_inputs.is_empty(),
                "ORANGU_REPLAY: capture has no per-token embedding input"
            );
            let ctx = unsafe { ReplayContext::from_wgpu(vulkan.wgpu_device()) }
                .context("ORANGU_REPLAY: device is not the Vulkan backend")?;
            let (graph, programs) = unsafe { ReplayGraph::from_capture(&ctx, &steps) }
                .map_err(|e| anyhow::anyhow!("ORANGU_REPLAY: build graph: {e}"))?;
            {
                use crate::engine::backend::vulkan_replay::CaptureStep;
                let n_dispatch = steps
                    .iter()
                    .filter(|s| matches!(s, CaptureStep::Dispatch { .. }))
                    .count();
                let n_copy = steps
                    .iter()
                    .filter(|s| matches!(s, CaptureStep::Copy { .. }))
                    .count();
                let n_host = steps
                    .iter()
                    .filter(|s| matches!(s, CaptureStep::HostInput { .. }))
                    .count();
                eprintln!(
                    "orangu-server: [replay] decode graph — {} steps/token = {} dispatch + {} copy + {} host ({} layers ⇒ {:.1} dispatch/layer, {:.1} copy/layer)",
                    steps.len(),
                    n_dispatch,
                    n_copy,
                    n_host,
                    self.layers.len(),
                    n_dispatch as f64 / self.layers.len() as f64,
                    n_copy as f64 / self.layers.len() as f64,
                );
                if std::env::var_os("ORANGU_REPLAY_HISTO").is_some() {
                    use std::collections::BTreeMap;
                    let mut histo: BTreeMap<String, u32> = BTreeMap::new();
                    for s in steps.iter() {
                        if let CaptureStep::Dispatch { wgsl, .. } = s {
                            let compact: String =
                                wgsl.split_whitespace().collect::<Vec<_>>().join(" ");
                            let n = compact.len();
                            let sig = compact[n.saturating_sub(70)..].to_string();
                            *histo.entry(sig).or_insert(0) += 1;
                        }
                    }
                    let mut rows: Vec<(&String, &u32)> = histo.iter().collect();
                    rows.sort_by(|a, b| b.1.cmp(a.1));
                    eprintln!(
                        "orangu-server: [replay] dispatch histogram (by shader-body signature):"
                    );
                    for (sig, cnt) in rows {
                        eprintln!("  {:>4}x  …{}", cnt, sig);
                    }
                }
            }
            eprintln!(
                "orangu-server: [replay] built persistent decode graph — {} steps ({} embd + {} gathered host inputs)",
                steps.len(),
                embd_inputs.len(),
                gathered_inputs.len()
            );
            *guard = Some(DecodeReplay {
                ctx,
                graph,
                _programs: programs,
                _captured_steps: steps,
                embd_inputs,
                gathered_inputs,
                logits_buf,
                logits_off,
                n_vocab: self.output_weight.out_dim,
                cache_ptr,
                slot_id,
                // Captured at this token's position; the next replayed token of
                // this sequence must be at `start_pos + 1`.
                expected_pos: start_pos + 1,
            });
            // First token of a request runs the full wgpu forward; hand its
            // logits back for the caller to sample (once per request — the hot
            // path is the replayed tokens below).
            return Ok(ForwardOutcome::Logits(logits));
        }

        let r = guard.as_mut().expect("just checked Some");
        // This sequence's next replayed token must land at `start_pos + 1`.
        r.expected_pos = start_pos + 1;
        for (buf, off) in &r.embd_inputs {
            vulkan
                .wgpu_queue()
                .write_buffer(buf, *off, bytemuck::cast_slice(&x));
        }
        if !r.gathered_inputs.is_empty() {
            let gathered = self.gather_per_layer_tok_embd(&[token], 1);
            for (buf, off) in &r.gathered_inputs {
                vulkan
                    .wgpu_queue()
                    .write_buffer(buf, *off, bytemuck::cast_slice(&gathered));
            }
        }
        // Flush the per-token embedding/gathered `write_buffer`s to the shared
        // `VkQueue` — but do NOT block on it. `run_token`'s raw submit
        // goes to the same queue right after, so submission order + the command
        // buffer's entry barrier (`TRANSFER` → `SHADER_READ`) already make the
        // transfer visible to the first dispatch; the old `poll(wait)` here just
        // idled the CPU (and the GPU) between tokens for no correctness benefit.
        vulkan.wgpu_queue().submit(std::iter::empty());
        unsafe {
            r.graph.update_per_token(start_pos as u32);
            r.graph
                .run_token(&r.ctx)
                .map_err(|e| anyhow::anyhow!("ORANGU_REPLAY: run_token: {e}"))?;
        }
        // Argmax tail on the GPU: reclaims the per-token `[n_vocab]` logits
        // readback + the CPU argmax (`total_cmp`/`max_by`, a measurable slice
        // of decode CPU in the replay profile). Runs the same sample kernel the non-replay
        // GPU-sample path uses, reading the replay's `logits_buf` (visible after
        // `run_token`'s final barrier) and reading back only the winning token id.
        // Falls back to the full logits readback when the caller isn't greedy
        // sampling or GPU sampling is disabled.
        if let Some(params) = greedy_sample
            && vulkan.gpu_sample()
        {
            let mut encoder =
                vulkan
                    .wgpu_device()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("orangu-server replay argmax tail"),
                    });
            let sample_buf = vulkan.record_argmax_sample(
                &mut encoder,
                GpuArgmaxSampleInput {
                    logits: GpuInput::Gpu(&r.logits_buf, (r.logits_off / 4) as usize),
                    n_vocab: r.n_vocab,
                    recent_tokens: params.recent_tokens,
                    repeat_penalty: params.repeat_penalty,
                    logit_softcap: self.final_logit_softcapping,
                },
                slot_id + 1,
            );
            let next = vulkan.submit_and_readback_u32(encoder, &sample_buf);
            return Ok(ForwardOutcome::Token(next));
        }
        Ok(ForwardOutcome::Logits(read_gpu_f32(
            vulkan,
            &r.logits_buf,
            r.logits_off,
            r.n_vocab,
        )))
    }
}

impl ModelForward for GemmaModel {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn new_kv_cache(&self, capacity: usize) -> KvCache {
        KvCache::new_with_dims(capacity, &self.kv_dims())
    }

    fn forward(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        slot_id: usize,
    ) -> Result<Vec<f32>> {
        anyhow::ensure!(
            self.causal,
            "'{}' is an embeddings-only architecture (bidirectional attention, no causal \
             masking) and does not support text generation — use the embeddings endpoints \
             instead",
            self.config.architecture
        );
        let n_tokens = tokens.len();
        let n_embd = self.config.n_embd;
        let eps = self.rms_eps();

        // Counts GPU submissions per token directly rather than inferring
        // the round-trip count indirectly — set `ORANGU_GPU_TRACE=1` to log
        // it. Only reads an env var (via a
        // cached `OnceLock`, not a fresh lookup every call) and an atomic
        // load/subtract when a Vulkan backend is in use; free otherwise.
        static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let trace = *TRACE.get_or_init(|| std::env::var("ORANGU_GPU_TRACE").is_ok());
        let submissions_before = (trace && n_tokens == 1)
            .then(|| self.backend.as_vulkan())
            .flatten()
            .map(|v| v.submission_count());

        // Splits a decode step's CPU-side wall-clock time into "recording"
        // (building the whole-layer-loop `wgpu::CommandEncoder` — every
        // `set_pipeline`/`set_bind_group`/`dispatch_workgroups` call the
        // Rust `wgpu` API itself costs, not GPU execution) vs. "submit+wait"
        // (`queue.submit()` plus `poll(wait_indefinitely())`, which spans
        // real GPU execution time *and* whatever CPU-side driver/kernel
        // scheduling latency sits between the CPU handing work off and the
        // GPU actually finishing it) — set `ORANGU_CPU_TIMESTAMPS=1` to log
        // it. `ORANGU_GPU_TIMESTAMPS` (ahead of this in the codebase)
        // already measures GPU *execution* time between layers; this
        // measures the two halves neither that flag nor `ORANGU_GPU_TRACE`'s
        // submission count can see at all — specifically, how much of a
        // decode step's wall clock is CPU-side command-buffer construction,
        // a cost `wgpu`'s API (unlike raw Vulkan's resubmittable
        // `VkCommandBuffer`s) requires paying fresh every single token, with
        // no capture/replay primitive to amortize it across steps that
        // share the exact same dispatch sequence.
        static CPU_TIMESTAMPS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let cpu_timestamps =
            *CPU_TIMESTAMPS.get_or_init(|| std::env::var("ORANGU_CPU_TIMESTAMPS").is_ok());
        let record_start = (cpu_timestamps && n_tokens == 1).then(std::time::Instant::now);

        // Embedding lookup, scaled by sqrt(n_embd) — every real-token input
        // path (Gemma never leaves this unscaled outside multimodal input).
        let mut x = vec![0f32; n_tokens * n_embd];
        for (t, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            anyhow::ensure!(
                tok < self.config.n_vocab,
                "token id {tok} is out of vocab range"
            );
            x[t * n_embd..(t + 1) * n_embd].copy_from_slice(&self.tok_embeddings.row(tok));
        }
        for v in x.iter_mut() {
            *v *= (n_embd as f32).sqrt();
        }

        // Per-layer embeddings (PLE), if this model has them: the decode/
        // GPU-fused branch folds the whole projection into the same
        // encoder/submission as the rest of the forward pass
        // (`VulkanBackend::record_ple_projection`) instead of calling
        // `compute_per_layer_inputs`
        // — a separate, CPU-orchestrated submit-and-wait — the way `Self::
        // run_layers_cpu` (used by the CPU-orchestrated `else` branch below,
        // and by `Self::forward_hidden_states`) still does internally.
        let mut logits = if n_tokens == 1
            && !self.is_moe
            && let Some(vulkan) = self.backend.as_vulkan()
        {
            // See `Self::record_decode_forward`'s own doc comment for
            // what's recorded; GPU submissions per decode token dropped
            // from ~37 to ~2 with whole-layer fusion, then to **1** with
            // PLE fusion folded into the same encoder. Prefill (`n_tokens
            // > 1`) and the CPU backend still take the fully-CPU-
            // orchestrated `else` branch below.
            let (encoder, _logits_buf, _logits_offset) =
                self.record_decode_forward(vulkan, cache, tokens[0], start_pos, &x, slot_id)?;
            let record_elapsed = record_start.map(|t| t.elapsed());
            let submit_start = cpu_timestamps.then(std::time::Instant::now);
            let logits = vulkan.submit_and_readback_for(encoder, &self.output_weight, slot_id + 1);
            // `submit_and_readback_for`'s own `poll(wait_indefinitely())`
            // already blocked until this whole submission (timestamp
            // resolve included) finished, so the readback here is never
            // premature.
            if let (Some(record), Some(submit_start)) = (record_elapsed, submit_start) {
                let submit = submit_start.elapsed();
                eprintln!(
                    "orangu-server: [cpu-trace] pos {start_pos}: record {:.3}ms, submit+wait {:.3}ms, cpu-total {:.3}ms",
                    record.as_secs_f64() * 1000.0,
                    submit.as_secs_f64() * 1000.0,
                    (record + submit).as_secs_f64() * 1000.0
                );
            }
            if vulkan.gpu_timestamps() {
                vulkan.report_timestamps(start_pos, self.layers.len());
            }
            logits
        } else {
            let x = self.run_layers_cpu(cache, &x, tokens, start_pos)?;
            let last = &mut x[(n_tokens - 1) * n_embd..].to_vec();
            tensor::rmsnorm_inplace(last, &self.output_norm, 1, n_embd, eps);
            self.backend.matmul(last, 1, &self.output_weight)
        };
        if let Some(cap) = self.final_logit_softcapping {
            for v in logits.iter_mut() {
                *v = (*v / cap).tanh() * cap;
            }
        }
        if let Some(before) = submissions_before
            && let Some(vulkan) = self.backend.as_vulkan()
        {
            eprintln!(
                "orangu-server: [gpu-trace] {} GPU submissions for this decode step (pos {start_pos})",
                vulkan.submission_count() - before
            );
        }
        Ok(logits)
    }

    fn forward_all_logits(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        _slot_id: usize,
    ) -> Result<Vec<Vec<f32>>> {
        anyhow::ensure!(
            self.causal,
            "'{}' is an embeddings-only architecture and does not support text generation",
            self.config.architecture
        );
        let n_tokens = tokens.len();
        let n_embd = self.config.n_embd;
        let n_vocab = self.config.n_vocab;
        let eps = self.rms_eps();

        // Same embedding lookup + sqrt(n_embd) scaling as `forward`. This path
        // is deliberately the CPU-orchestrated one (never the single-token
        // GPU-fused decode branch): the keys/values it appends stay CPU-side,
        // so a caller can read them back or roll them off with
        // `KvCache::truncate`, and one weight stream covers every position.
        let mut x = vec![0f32; n_tokens * n_embd];
        for (t, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            anyhow::ensure!(tok < n_vocab, "token id {tok} is out of vocab range");
            x[t * n_embd..(t + 1) * n_embd].copy_from_slice(&self.tok_embeddings.row(tok));
        }
        for v in x.iter_mut() {
            *v *= (n_embd as f32).sqrt();
        }

        // One projection of every position through the output norm + vocab
        // matrix, batched — the weight-heavy `lm_head` read is amortized across
        // the whole draft in a single `matmul`, not one per position.
        let mut h = self.run_layers_cpu(cache, &x, tokens, start_pos)?;
        tensor::rmsnorm_inplace(&mut h, &self.output_norm, n_tokens, n_embd, eps);
        let flat = self.backend.matmul(&h, n_tokens, &self.output_weight);
        anyhow::ensure!(
            flat.len() == n_tokens * n_vocab,
            "output projection produced {} logits, expected {}",
            flat.len(),
            n_tokens * n_vocab
        );

        let mut out = Vec::with_capacity(n_tokens);
        for t in 0..n_tokens {
            let mut row = flat[t * n_vocab..(t + 1) * n_vocab].to_vec();
            if let Some(cap) = self.final_logit_softcapping {
                for v in row.iter_mut() {
                    *v = (*v / cap).tanh() * cap;
                }
            }
            out.push(row);
        }
        Ok(out)
    }

    /// Takes the GPU-argmax fast path only when every one of its
    /// preconditions holds: `tokens.len() == 1` (`Self::record_decode_
    /// forward` is decode-shaped only), a `Vulkan` backend is in use
    /// without `ORANGU_NO_GPU_SAMPLE=1` set (`VulkanBackend::gpu_sample`
    /// — **on by default**; correctness-verified and no measured
    /// end-to-end regression, see that method's own doc comment for the
    /// numbers), the caller actually wants greedy sampling
    /// (`greedy_sample.is_some()`), and this model has **no** final-logit
    /// softcapping configured.
    ///
    /// That last check matters: `tanh`-based softcapping
    /// (`x -> tanh(x / cap) * cap`) is strictly increasing, so it never
    /// changes which logit is the argmax *on its own* — but the real
    /// pipeline doesn't apply it on its own. `Self::forward` applies
    /// softcapping first, then the repeat penalty is applied afterward
    /// (by the caller, over in `engine::generate`) to the *softcapped*
    /// values. Applying the penalty to *raw* values instead (which is all
    /// this fast path does — it has no softcapping step of its own) is
    /// not guaranteed to pick the same token, since the penalty only
    /// touches specific positions and softcapping's squashing changes how
    /// much those positions' *raw* magnitude differs from the rest before
    /// the penalty ever sees them. Rather than prove that reordering is
    /// safe (or unsafe) in general, this simply doesn't take the fast path
    /// at all when softcapping is configured, falling back to the exact
    /// existing CPU-verified pipeline instead — `E2B` and every other
    /// model this project has tested against leave softcapping unset, so
    /// this costs nothing in practice today.
    fn forward_maybe_sampling(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        greedy_sample: Option<GreedySampleParams<'_>>,
        slot_id: usize,
    ) -> Result<ForwardOutcome> {
        // Raw-Vulkan decode replay (`ORANGU_REPLAY`): capture the
        // forward once, then resubmit the persistent command buffer every token
        // with no `wgpu` submit on the forward — returns logits for the caller
        // to sample (bypasses the GPU-argmax fast path).
        //
        // The capture now covers the *whole* gemma4 forward — the per-layer-
        // embedding (PLE) projection + sub-chain and the `layer_output_scale`
        // dispatch included, with PLE's gathered per-layer
        // embeddings re-uploaded each token as a second host input — so
        // `ORANGU_REPLAY` engages for every gemma variant this backend records,
        // gemma4-E2B (PLE + `layer_output_scale`) included.
        let replay_supported = true;
        // Opt-in (`ORANGU_REPLAY`). The replay path removes the per-token wgpu
        // record/submit, but single-token decode is bound by the `Q4_K`
        // matmul-vec kernel — not by CPU submit — so removing the submit is not
        // the bottleneck. It also pins the captured decode kernel, which
        // suppresses the faster kernel `pipeline_for` would otherwise select, so
        // it is off by default and kept for capture/replay study.
        // `ORANGU_REPLAY_FORCE` still bypasses the support check for debugging
        // incomplete captures.
        let force = std::env::var_os("ORANGU_REPLAY_FORCE").is_some();
        let replay_on = std::env::var_os("ORANGU_REPLAY").is_some();
        // MoE models never take the fully-fused Vulkan paths (the fused
        // record path is dense-FFN-only) — they run CPU-orchestrated via
        // `Self::forward`'s `else` branch, so short-circuit to it here.
        if tokens.len() == 1
            && !self.is_moe
            && (force || (replay_supported && replay_on))
            && let Some(vulkan) = self.backend.as_vulkan()
        {
            return self.decode_forward_replay(
                vulkan,
                cache,
                tokens[0],
                start_pos,
                greedy_sample.as_ref(),
                slot_id,
            );
        }
        // A `final_logit_softcapping` model no longer forces the slow CPU
        // path here: the softcap is `cap * tanh(v / cap)`, monotonic, so it
        // can't change the greedy token, and the GPU sample kernel applies it
        // (before the repeat penalty, matching the CPU order) so a softcapped
        // model keeps the single-`u32` readback instead of transferring the
        // whole `[n_vocab]` logits vector to `tanh` it on the CPU every token.
        if tokens.len() == 1
            && !self.is_moe
            && let Some(params) = &greedy_sample
            && let Some(vulkan) = self.backend.as_vulkan()
            && vulkan.gpu_sample()
        {
            let n_embd = self.config.n_embd;
            let token = tokens[0];
            anyhow::ensure!(
                (token as usize) < self.config.n_vocab,
                "token id {token} is out of vocab range"
            );
            let mut x = self.tok_embeddings.row(token as usize).to_vec();
            for v in x.iter_mut() {
                *v *= (n_embd as f32).sqrt();
            }
            let (mut encoder, logits_buf, logits_offset) =
                self.record_decode_forward(vulkan, cache, token, start_pos, &x, slot_id)?;
            // `GpuInput::Gpu`'s own offset is in elements, not bytes —
            // `logits_offset` (from `Self::record_full_matmul`'s own
            // `CachedOpResources::output_offset`) is always a multiple of 4
            // (the arena's own minimum alignment), so this divides evenly.
            let sample_buf = vulkan.record_argmax_sample(
                &mut encoder,
                GpuArgmaxSampleInput {
                    logits: GpuInput::Gpu(&logits_buf, (logits_offset / 4) as usize),
                    n_vocab: self.output_weight.out_dim,
                    recent_tokens: params.recent_tokens,
                    repeat_penalty: params.repeat_penalty,
                    logit_softcap: self.final_logit_softcapping,
                },
                // Per-slot key so two concurrently-decoding sequences never
                // share the cached sample scratch (same rationale as the
                // `slot_id + 1` batch_slot the op cache uses just above).
                slot_id + 1,
            );
            let next = vulkan.submit_and_readback_u32(encoder, &sample_buf);
            return Ok(ForwardOutcome::Token(next));
        }
        self.forward(cache, tokens, start_pos, slot_id)
            .map(ForwardOutcome::Logits)
    }

    /// See [`ModelForward::forward_batch_decode`]'s own doc comment for
    /// the shape of what this does and why.
    ///
    /// `items.len() <= 1` falls back to `Self::forward_maybe_sampling`
    /// (preserving its GPU-argmax fast path, on by default, for the
    /// common single-sequence case) rather than taking either batched
    /// path with a batch of one — there's nothing to amortize across a
    /// batch that doesn't have at least two members, and neither batched
    /// path below ever attempts GPU sampling at all (always returns
    /// `Logits`, letting the caller — `engine::batch::BatchCoordinator` —
    /// sample on the CPU), so a batch-of-one here would be strictly worse
    /// than the existing single-sequence path for no benefit.
    ///
    /// For a real batch (`items.len() >= 2`) against the Vulkan backend,
    /// `Self::record_batched_decode_forward` (that method's own doc
    /// comment has the details) is used — every sequence's whole decode
    /// step chained into one shared GPU submission. Every other backend
    /// (in practice, just `CpuBackend`) falls back to the CPU-orchestrated
    /// path below: structurally, this mirrors `Self::forward`'s CPU-
    /// orchestrated `else` branch almost exactly — same per-layer
    /// sequence of matmul/norm/RoPE/attention/residual steps, same math —
    /// except every place that branch loops `for t in 0..n_tokens` over
    /// *one* sequence's multiple positions, this loops over `items` — *N
    /// different sequences'* own single position each — and every matmul
    /// call's `n_tokens` argument becomes `items.len()` (the batch width)
    /// instead of a prompt's length.
    ///
    /// Both batched paths are correctness-verified
    /// (`forward_batch_decode_matches_independent_forward_calls_*`,
    /// below) against independent per-sequence `forward` calls. One
    /// honest observation from real-model testing, true of the CPU-
    /// orchestrated fallback specifically (not the Vulkan path, which
    /// reuses the exact same `gpu_attention` kernel per sequence the
    /// single-sequence decode path uses): generating many tokens (~100)
    /// through it can *diverge* from what the single-sequence path would
    /// have generated for the exact same prompt — not a bug (the per-step
    /// logits already match within the tight tolerance the tests below
    /// check), just the expected consequence of greedy decoding being
    /// sensitive to tiny floating-point differences: the CPU-orchestrated
    /// fallback's attention step is its own independently-written CPU
    /// loop, not the single-sequence path's GPU kernel — two
    /// independently-written, both-correct implementations of the same
    /// math whose tiny per-step differences can compound, over enough
    /// autoregressive steps, into an argmax flipping to a different
    /// (still fluent, still coherent) token somewhere along the way.
    fn forward_batch_decode(
        &self,
        items: &mut [BatchDecodeItem<'_>],
    ) -> Result<Vec<ForwardOutcome>> {
        let n = items.len();
        if n <= 1 {
            return items
                .iter_mut()
                .map(|item| {
                    self.forward_maybe_sampling(
                        item.cache,
                        &[item.token],
                        item.start_pos,
                        item.greedy_sample.take(),
                        item.slot_id,
                    )
                })
                .collect();
        }

        if !self.is_moe
            && let Some(vulkan) = self.backend.as_vulkan()
        {
            return Ok(self
                .record_batched_decode_forward(vulkan, items)
                .into_iter()
                .map(ForwardOutcome::Logits)
                .collect());
        }

        let n_embd = self.config.n_embd;
        let eps = self.rms_eps();

        // N embedding lookups, stacked into one `[n, n_embd]`
        // buffer — the same "n_tokens" shape `Self::forward`'s CPU path
        // builds for a multi-position prompt, just one row per *sequence*
        // instead of one row per *position*.
        let mut x = vec![0f32; n * n_embd];
        for (i, item) in items.iter().enumerate() {
            anyhow::ensure!(
                (item.token as usize) < self.config.n_vocab,
                "token id {} is out of vocab range",
                item.token
            );
            x[i * n_embd..(i + 1) * n_embd]
                .copy_from_slice(&self.tok_embeddings.row(item.token as usize));
        }
        for v in x.iter_mut() {
            *v *= (n_embd as f32).sqrt();
        }

        // Per-layer-embedding input, per sequence — `per_layer_
        // model_proj`/`per_layer_proj_norm` are small next to the main
        // attention/FFN weights, so batching this too wasn't worth the
        // extra bookkeeping; `compute_per_layer_inputs` is already
        // n_tokens-generic, just called once per sequence with n_tokens=1
        // here instead of once for a whole prompt.
        let per_layer = self.n_embd_per_layer;
        let has_ple = per_layer > 0;
        let inp_per_layer: Vec<Vec<f32>> = if has_ple {
            items
                .iter()
                .enumerate()
                .map(|(i, item)| {
                    self.compute_per_layer_inputs(
                        &x[i * n_embd..(i + 1) * n_embd],
                        &[item.token],
                        1,
                    )
                })
                .collect()
        } else {
            Vec::new()
        };

        for (il, layer) in self.layers.iter().enumerate() {
            let head_dim = layer.head_dim;
            let freq_factors = (!layer.is_swa)
                .then_some(self.rope_freqs.as_deref())
                .flatten();
            let cache_index = layer.kv_donor;
            let group_size = self.n_head / layer.n_head_kv;

            let mut normed = x.clone();
            tensor::rmsnorm_inplace(&mut normed, &layer.attn_norm, n, n_embd, eps);

            let wk = layer.has_kv.then(|| {
                layer
                    .wk
                    .as_ref()
                    .context("layer has_kv but no attn_k.weight")
            });
            let wk = wk.transpose()?;
            let owns_v = layer.has_kv && layer.wv.is_some();

            // The cross-sequence GEMM batching win: QKV projected for all
            // `n` sequences in one `matmul_batch` call instead of `n`
            // independent ones.
            let mut ops = vec![MatmulOp {
                x: &normed,
                n_tokens: n,
                w: &layer.wq,
            }];
            if let Some(wk) = wk {
                ops.push(MatmulOp {
                    x: &normed,
                    n_tokens: n,
                    w: wk,
                });
            }
            if owns_v {
                ops.push(MatmulOp {
                    x: &normed,
                    n_tokens: n,
                    w: layer.wv.as_ref().unwrap(),
                });
            }
            let mut results = self.backend.matmul_batch(&ops).into_iter();
            let mut q = results.next().unwrap();
            tensor::rmsnorm_inplace(&mut q, &layer.attn_q_norm, n * self.n_head, head_dim, eps);
            // RoPE stays per-sequence: each sequence has its own position.
            for (i, item) in items.iter().enumerate() {
                let pos = item.start_pos;
                tensor::rope_apply_scaled_inplace(
                    &mut q[i * self.n_head * head_dim..(i + 1) * self.n_head * head_dim],
                    self.n_head,
                    head_dim,
                    layer.rope_dim,
                    pos,
                    layer.rope_freq_base,
                    freq_factors,
                );
            }

            if layer.has_kv {
                let kv_dim = layer.n_head_kv * head_dim;
                let mut k = results.next().unwrap();
                tensor::rmsnorm_inplace(
                    &mut k,
                    layer
                        .attn_k_norm
                        .as_ref()
                        .context("layer has_kv but no attn_k_norm")?,
                    n * layer.n_head_kv,
                    head_dim,
                    eps,
                );
                let mut v = if owns_v {
                    results.next().unwrap()
                } else {
                    k.clone()
                };
                rmsnorm_weightless_inplace(&mut v, n * layer.n_head_kv, head_dim, eps);

                // RoPE + KV-cache write: per-sequence, each into its *own*
                // cache — there is no shared cache to batch across here.
                for (i, item) in items.iter_mut().enumerate() {
                    let pos = item.start_pos;
                    tensor::rope_apply_scaled_inplace(
                        &mut k[i * kv_dim..(i + 1) * kv_dim],
                        layer.n_head_kv,
                        head_dim,
                        layer.rope_dim,
                        pos,
                        layer.rope_freq_base,
                        freq_factors,
                    );
                    item.cache.layers[cache_index].push(
                        &k[i * kv_dim..(i + 1) * kv_dim],
                        &v[i * kv_dim..(i + 1) * kv_dim],
                    );
                }
            }

            // Attention: inherently per-sequence (each sequence attends
            // only to its own cache) — no weight matrix here to amortize
            // across the batch, so this stays a plain per-sequence loop,
            // same math as `Self::forward`'s CPU attention loop.
            let mut attn_out = vec![0f32; n * self.n_head * head_dim];
            for (i, item) in items.iter().enumerate() {
                let pos = item.start_pos;
                let window_start = if layer.is_swa && self.n_swa > 0 {
                    pos.saturating_sub(self.n_swa - 1)
                } else {
                    0
                };
                for h in 0..self.n_head {
                    let kv_head = h / group_size;
                    let qh = &q[i * self.n_head * head_dim + h * head_dim
                        ..i * self.n_head * head_dim + (h + 1) * head_dim];

                    let mut scores = Vec::with_capacity(pos + 1 - window_start);
                    for p in window_start..=pos {
                        let kh = item.cache.layers[cache_index].key_at(p, kv_head, head_dim);
                        scores.push(tensor::dot(qh, kh) * self.attention_scale);
                    }
                    tensor::softmax_inplace(&mut scores);

                    let out = &mut attn_out[i * self.n_head * head_dim + h * head_dim
                        ..i * self.n_head * head_dim + (h + 1) * head_dim];
                    for (offset, &weight) in scores.iter().enumerate() {
                        let p = window_start + offset;
                        let vh = item.cache.layers[cache_index].value_at(p, kv_head, head_dim);
                        for (o, vi) in out.iter_mut().zip(vh.iter()) {
                            *o += weight * vi;
                        }
                    }
                }
            }

            let mut attn_proj = self.backend.matmul(&attn_out, n, &layer.wo);
            tensor::rmsnorm_inplace(&mut attn_proj, &layer.attn_post_norm, n, n_embd, eps);
            tensor::add_inplace(&mut x, &attn_proj);

            // FFN — same dense/MoE split as `run_layers_cpu`, here batched
            // across the `n` sequences instead of a prompt's positions. `x`
            // is the post-attention residual at this point.
            if let Some(moe) = &layer.moe {
                let mut ffn_out = self.moe_ffn_result(layer, moe, &x, n);
                tensor::rmsnorm_inplace(&mut ffn_out, &layer.ffn_post_norm, n, n_embd, eps);
                tensor::add_inplace(&mut x, &ffn_out);
            } else {
                let attn_out_residual = x.clone();
                let mut ffn_normed = x.clone();
                tensor::rmsnorm_inplace(&mut ffn_normed, &layer.ffn_norm, n, n_embd, eps);
                let mut gate_up = self.backend.matmul_batch(&[
                    MatmulOp {
                        x: &ffn_normed,
                        n_tokens: n,
                        w: &layer.ffn_gate,
                    },
                    MatmulOp {
                        x: &ffn_normed,
                        n_tokens: n,
                        w: &layer.ffn_up,
                    },
                ]);
                let up = gate_up.pop().unwrap();
                let mut gate = gate_up.pop().unwrap();
                tensor::gelu_inplace(&mut gate);
                tensor::mul_inplace(&mut gate, &up);
                let mut ffn_out = self.backend.matmul(&gate, n, &layer.ffn_down);
                tensor::rmsnorm_inplace(&mut ffn_out, &layer.ffn_post_norm, n, n_embd, eps);
                x = attn_out_residual;
                tensor::add_inplace(&mut x, &ffn_out);
            }

            if let (Some(gate_w), Some(proj_w), Some(post_norm)) = (
                &layer.per_layer_inp_gate,
                &layer.per_layer_proj,
                &layer.per_layer_post_norm,
            ) {
                let pe_in = x.clone();
                let mut g = self.backend.matmul(&x, n, gate_w);
                tensor::gelu_inplace(&mut g);
                for (i, per_layer_input) in inp_per_layer.iter().enumerate() {
                    let slice = &per_layer_input[il * per_layer..(il + 1) * per_layer];
                    tensor::mul_inplace(&mut g[i * per_layer..(i + 1) * per_layer], slice);
                }
                let mut proj = self.backend.matmul(&g, n, proj_w);
                tensor::rmsnorm_inplace(&mut proj, post_norm, n, n_embd, eps);
                x = pe_in;
                tensor::add_inplace(&mut x, &proj);
            }

            if let Some(scale) = layer.layer_output_scale {
                for v in x.iter_mut() {
                    *v *= scale;
                }
            }
        }

        tensor::rmsnorm_inplace(&mut x, &self.output_norm, n, n_embd, eps);
        let mut logits = self.backend.matmul(&x, n, &self.output_weight);
        if let Some(cap) = self.final_logit_softcapping {
            for v in logits.iter_mut() {
                *v = (*v / cap).tanh() * cap;
            }
        }
        let n_vocab = self.output_weight.out_dim;
        Ok(logits
            .chunks(n_vocab)
            .map(|row| ForwardOutcome::Logits(row.to_vec()))
            .collect())
    }

    fn forward_hidden_states(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let n_tokens = tokens.len();
        let n_embd = self.config.n_embd;
        let eps = self.rms_eps();

        // Embedding lookup, scaled by sqrt(n_embd) — same prep `forward`
        // does at its own top; recomputed independently here rather than
        // threaded through, matching `engine::arch::llama::LlamaModel::
        // run_layers`'s own independent-embedding-lookup style.
        let mut x0 = vec![0f32; n_tokens * n_embd];
        for (t, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            anyhow::ensure!(
                tok < self.config.n_vocab,
                "token id {tok} is out of vocab range"
            );
            x0[t * n_embd..(t + 1) * n_embd].copy_from_slice(&self.tok_embeddings.row(tok));
        }
        for v in x0.iter_mut() {
            *v *= (n_embd as f32).sqrt();
        }

        // A one-shot, whole-prompt pass — no KV cache reuse across calls,
        // same convention `LlamaModel::forward_hidden_states` uses.
        let mut cache = self.new_kv_cache(n_tokens.max(1));
        let mut x = self.run_layers_cpu(&mut cache, &x0, tokens, 0)?;
        tensor::rmsnorm_inplace(&mut x, &self.output_norm, n_tokens, n_embd, eps);
        Ok(x)
    }

    fn post_pool_projection(&self, pooled: Vec<f32>) -> Result<Vec<f32>> {
        let Some(dense_2) = &self.dense_2 else {
            return Ok(pooled);
        };
        let mut cur = self.backend.matmul(&pooled, 1, dense_2);
        if let Some(dense_3) = &self.dense_3 {
            cur = self.backend.matmul(&cur, 1, dense_3);
        }
        Ok(cur)
    }
}

impl GemmaModel {
    /// Computes the combined per-layer-embedding input for every token and
    /// layer (`project_per_layer_inputs` + `build_inp_per_layer` in the
    /// reference graph), flattened as `[n_tokens, n_layer, n_embd_per_layer]`
    /// row-major.
    /// The first phase of `compute_per_layer_inputs`: gathers each token's
    /// per-layer embedding row, scaled by `sqrt(per_layer)` —
    /// `[n_tokens, n_layer, per_layer]` row-major, same shape and content
    /// `compute_per_layer_inputs` itself would produce this piece of. Split
    /// out so the decode (`n_tokens == 1`) GPU-fused path
    /// (`VulkanBackend::record_ple_projection`) can reuse it too, without
    /// also running the remaining phases on the CPU (those move to the GPU there
    /// instead) — it's a
    /// tiny embedding-table lookup, cheap enough to stay a plain CPU
    /// gather + upload rather than needing its own GPU kernel.
    fn gather_per_layer_tok_embd(&self, tokens: &[u32], n_tokens: usize) -> Vec<f32> {
        let per_layer = self.n_embd_per_layer;
        let n_layer = self.layers.len();
        let tok_embd_scale = (per_layer as f32).sqrt();
        let per_layer_tok_embd = self.per_layer_tok_embd.as_ref().expect("checked by caller");

        let row_width = per_layer * n_layer;
        let mut gathered = vec![0f32; n_tokens * row_width];
        for (t, &tok) in tokens.iter().enumerate() {
            let row = per_layer_tok_embd.row(tok as usize);
            let dst = &mut gathered[t * row_width..(t + 1) * row_width];
            dst.copy_from_slice(&row);
        }
        for v in gathered.iter_mut() {
            *v *= tok_embd_scale;
        }
        gathered
    }

    fn compute_per_layer_inputs(
        &self,
        x_scaled_embd: &[f32],
        tokens: &[u32],
        n_tokens: usize,
    ) -> Vec<f32> {
        let n_embd = self.config.n_embd;
        let per_layer = self.n_embd_per_layer;
        let n_layer = self.layers.len();
        let per_layer_projection_scale = 1.0 / (n_embd as f32).sqrt();
        let per_layer_input_scale = 1.0 / 2f32.sqrt();

        let per_layer_model_proj = self
            .per_layer_model_proj
            .as_ref()
            .expect("checked by caller");
        let per_layer_proj_norm = self
            .per_layer_proj_norm
            .as_ref()
            .expect("checked by caller");

        // First, gather each token's per-layer embedding row, scaled.
        let gathered = self.gather_per_layer_tok_embd(tokens, n_tokens);

        // Then project the (already sqrt(n_embd)-scaled) hidden state.
        let mut proj = self
            .backend
            .matmul(x_scaled_embd, n_tokens, per_layer_model_proj);
        for v in proj.iter_mut() {
            *v *= per_layer_projection_scale;
        }
        tensor::rmsnorm_inplace(
            &mut proj,
            per_layer_proj_norm,
            n_tokens * n_layer,
            per_layer,
            self.rms_eps(),
        );

        // Finally, combine and scale.
        tensor::add_inplace(&mut proj, &gathered);
        for v in proj.iter_mut() {
            *v *= per_layer_input_scale;
        }
        proj
    }

    /// A MoE gemma4 layer's router logits (`[n_tokens, n_expert]`
    /// row-major), computed the way gemma4.cpp does — deliberately reading
    /// the post-attention residual `attn_out` (`[n_tokens, n_embd]`), *not*
    /// the expert branch's own pre-normed input: a **weightless** RMSNorm,
    /// scaled by `1/sqrt(n_embd)` and multiplied elementwise by the learned
    /// per-dim `ffn_gate_inp.scale`, then projected through the router
    /// (`ffn_gate_inp`).
    fn moe_router_logits(&self, moe: &GemmaMoe, attn_out: &[f32], n_tokens: usize) -> Vec<f32> {
        let n_embd = self.config.n_embd;
        let eps = self.rms_eps();
        let scale = 1.0 / (n_embd as f32).sqrt();

        let mut tmp = attn_out.to_vec();
        rmsnorm_weightless_inplace(&mut tmp, n_tokens, n_embd, eps);
        for row in tmp.chunks_mut(n_embd) {
            for (v, s) in row.iter_mut().zip(moe.gate_inp_scale.iter()) {
                *v *= scale * s;
            }
        }
        // `[n_tokens, n_expert]` — one router score per expert per token.
        self.backend.matmul(&tmp, n_tokens, &moe.gate_inp)
    }

    /// A MoE gemma4 layer's FFN contribution *before* the shared
    /// `ffn_post_norm` (which the caller applies, exactly as it does for a
    /// dense layer): the elementwise sum of two branches computed off the
    /// same post-attention residual `attn_out` (`[n_tokens, n_embd]`), per
    /// gemma4.cpp's `is_moe_layer` graph:
    /// - a **dense GEGLU "shared" MLP** — this layer's always-present
    ///   `ffn_norm`/`ffn_gate`/`ffn_up`/`ffn_down` (identical to a dense
    ///   layer's FFN), then its own `post_ffw_norm_1`;
    /// - a **routed-expert branch** — `pre_ffw_norm_2` input norm, softmax
    ///   top-`n_expert_used` routing (renormalized over the selected experts,
    ///   the same `LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX`/`norm_w=true` path
    ///   [`super::qwen35moe`] implements), GELU experts, then its own
    ///   `post_ffw_norm_2`. The routing weights come from
    ///   [`Self::moe_router_logits`] (which reads `attn_out`, not this
    ///   branch's `pre_ffw_norm_2`-normed input).
    fn moe_ffn_result(
        &self,
        layer: &GemmaLayer,
        moe: &GemmaMoe,
        attn_out: &[f32],
        n_tokens: usize,
    ) -> Vec<f32> {
        let n_embd = self.config.n_embd;
        let eps = self.rms_eps();

        // Dense shared-MLP branch (GEGLU) — the exact dense-FFN computation,
        // using this layer's ffn_norm/ffn_gate/ffn_up/ffn_down.
        let mut mlp_normed = attn_out.to_vec();
        tensor::rmsnorm_inplace(&mut mlp_normed, &layer.ffn_norm, n_tokens, n_embd, eps);
        let mut gate_up = self.backend.matmul_batch(&[
            MatmulOp {
                x: &mlp_normed,
                n_tokens,
                w: &layer.ffn_gate,
            },
            MatmulOp {
                x: &mlp_normed,
                n_tokens,
                w: &layer.ffn_up,
            },
        ]);
        let up = gate_up.pop().unwrap();
        let mut gate = gate_up.pop().unwrap();
        tensor::gelu_inplace(&mut gate);
        tensor::mul_inplace(&mut gate, &up);
        let mut result = self.backend.matmul(&gate, n_tokens, &layer.ffn_down);
        tensor::rmsnorm_inplace(&mut result, &moe.post_norm_1, n_tokens, n_embd, eps);

        // Routed-expert branch. Expert input is its own `pre_ffw_norm_2`-normed
        // residual; the routing weights come from the (differently-normed)
        // `attn_out` — see `moe_router_logits`.
        let mut expert_in = attn_out.to_vec();
        tensor::rmsnorm_inplace(&mut expert_in, &moe.pre_norm_2, n_tokens, n_embd, eps);
        let logits = self.moe_router_logits(moe, attn_out, n_tokens);
        let n_expert = moe.gate_inp.out_dim;

        // Route every token first (cheap, sequential): softmax its logits,
        // take the top `n_expert_used`, renormalize their weights over the
        // selection (clamped like the reference's `ggml_clamp` against a zero
        // denominator). Flatten into a `(token, expert, weight)` work list.
        let mut work: Vec<(usize, usize, f32)> = Vec::with_capacity(n_tokens * self.n_expert_used);
        for t in 0..n_tokens {
            let mut probs = logits[t * n_expert..(t + 1) * n_expert].to_vec();
            tensor::softmax_inplace(&mut probs);
            let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            indexed.truncate(self.n_expert_used);
            let weight_sum: f32 = indexed
                .iter()
                .map(|(_, w)| w)
                .sum::<f32>()
                .max(6.103_515_6e-5);
            for (expert, weight) in indexed {
                work.push((t, expert, weight / weight_sum));
            }
        }

        // Evaluate each selected expert in parallel — this is the routed
        // FFN's dominant cost and the only part of the MoE forward still on
        // the CPU (per-row `Q*_0` dequant + dot; the shared MLP, attention,
        // and router all dispatch to the GPU backend). Each `(token, expert)`
        // pair is independent (reads shared weights + its own token's input,
        // writes its own contribution vector), so they fan out across every
        // core; the tiny weighted sum back into `moe_out` stays sequential.
        let contribs: Vec<(usize, Vec<f32>)> = work
            .par_iter()
            .map(|&(t, expert, weight)| {
                let x_t = &expert_in[t * n_embd..(t + 1) * n_embd];

                // gate/up projection (fused or separate). A per-expert
                // `.scale`, if present, multiplies that expert's raw gate/up
                // output *before* the GELU (matches `build_lora_mm_id`).
                let (mut gate, up) = match &moe.gate_up {
                    GemmaExpertGateUp::Fused { gate_up, scale } => {
                        let n_ff = gate_up.out_dim / 2;
                        let mut gate: Vec<f32> = (0..n_ff)
                            .map(|o| tensor::dot(x_t, &gate_up.row(expert, o)))
                            .collect();
                        let mut up: Vec<f32> = (0..n_ff)
                            .map(|o| tensor::dot(x_t, &gate_up.row(expert, n_ff + o)))
                            .collect();
                        if let Some(s) = scale {
                            let se = s[expert];
                            gate.iter_mut().for_each(|v| *v *= se);
                            up.iter_mut().for_each(|v| *v *= se);
                        }
                        (gate, up)
                    }
                    GemmaExpertGateUp::Separate {
                        gate,
                        up,
                        gate_scale,
                        up_scale,
                    } => {
                        let mut g: Vec<f32> = (0..gate.out_dim)
                            .map(|o| tensor::dot(x_t, &gate.row(expert, o)))
                            .collect();
                        let mut u: Vec<f32> = (0..up.out_dim)
                            .map(|o| tensor::dot(x_t, &up.row(expert, o)))
                            .collect();
                        if let Some(s) = gate_scale {
                            let se = s[expert];
                            g.iter_mut().for_each(|v| *v *= se);
                        }
                        if let Some(s) = up_scale {
                            let se = s[expert];
                            u.iter_mut().for_each(|v| *v *= se);
                        }
                        (g, u)
                    }
                };
                tensor::gelu_inplace(&mut gate);
                tensor::mul_inplace(&mut gate, &up);
                let h = gate;

                // Down projection, then the per-expert down `.scale` (if any)
                // and the routing weight — both scalars, folded into one.
                // Row-parallel *within* the expert as well: decode routes only
                // `n_expert_used` (8) tokens' worth of work, too few to fill
                // every core from the outer fan-out alone, so the down
                // projection's `n_embd` rows fan out too (rayon's work-stealing
                // keeps this from oversubscribing when the outer list is
                // already wide, e.g. during prefill).
                let dscale = moe.down_scale.as_ref().map_or(1.0, |s| s[expert]) * weight;
                let contrib: Vec<f32> = (0..moe.down_exps.out_dim)
                    .into_par_iter()
                    .map(|o| dscale * tensor::dot(&h, &moe.down_exps.row(expert, o)))
                    .collect();
                (t, contrib)
            })
            .collect();

        let mut moe_out = vec![0f32; n_tokens * n_embd];
        for (t, contrib) in &contribs {
            let dst = &mut moe_out[t * n_embd..(t + 1) * n_embd];
            for (d, v) in dst.iter_mut().zip(contrib) {
                *d += v;
            }
        }
        tensor::rmsnorm_inplace(&mut moe_out, &moe.post_norm_2, n_tokens, n_embd, eps);

        tensor::add_inplace(&mut result, &moe_out);
        result
    }
}

/// A plain (unweighted) RMSNorm — Gemma4's `Vcur` normalization
/// (`ggml_rms_norm` with no following `ggml_mul` by a learned weight,
/// unlike every other norm in this architecture).
fn rmsnorm_weightless_inplace(x: &mut [f32], n_rows: usize, dim: usize, eps: f32) {
    debug_assert_eq!(x.len(), n_rows * dim);
    for row in x.chunks_mut(dim) {
        let mean_sq: f32 = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let scale = 1.0 / (mean_sq + eps).sqrt();
        for v in row.iter_mut() {
            *v *= scale;
        }
    }
}

#[cfg(test)]
mod real_model_tests {
    use super::*;
    use crate::engine::arch::ModelForward;

    /// Cross-check against real llama.cpp: given the correct token IDs for
    /// "The capital of France is" (BOS=2 prepended, matching real
    /// llama.cpp's `/tokenize?add_special=true` and `/completion` default —
    /// this test feeds token IDs directly, sidestepping the separate,
    /// already-known SentencePiece tokenizer gap), the model should
    /// predict " Paris" (token
    /// 9079) as the single dominant next token, exactly as real llama.cpp's
    /// `/completion` (`n_probs`) does. This is what caught a real bug: the
    /// donor layer for Gemma4's shared-KV layers must be chosen per the
    /// *current* layer's own SWA-ness (SWA and full-attention layers have
    /// different head dims and can't share a cache) — run with
    /// `ORANGU_TEST_MODEL=/path/to.gguf cargo test --release --bin
    /// orangu-server real_model_tests -- --ignored`.
    #[test]
    #[ignore]
    fn gemma4_predicts_paris_after_capital_of_france() {
        let path = std::env::var("ORANGU_TEST_MODEL").expect("set ORANGU_TEST_MODEL");
        let loaded = LoadedModel::open(std::path::Path::new(&path)).expect("load model");
        let model =
            GemmaModel::load_with_backend(&loaded, Arc::new(crate::engine::backend::CpuBackend))
                .expect("build model");

        let mut cache = model.new_kv_cache(64);
        let tokens: Vec<u32> = vec![2, 818, 5279, 529, 7001, 563];
        let logits = model.forward(&mut cache, &tokens, 0, 0).expect("forward");
        let (top_id, _) = logits
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        assert_eq!(top_id, 9079, "expected ' Paris' (9079) as top prediction");
    }

    /// Regression guard for **dense gemma-4 with a per-layer
    /// `attention.head_count_kv` array** — `gemma-4-12B` and `gemma-4-31B`,
    /// whose full-attention layers use far fewer KV heads than their SWA
    /// layers (12B: 1 vs 8; 31B: 4 vs 16). If [`GemmaLayer::n_head_kv`] ever
    /// regresses to the scalar `head_count` fallback, every full-attention
    /// layer's GQA grouping is wrong and the logits collapse to a flat,
    /// near-tie mush (observed top-gap ~0.05, with reserved/whitespace ids
    /// winning) instead of a confident prediction.
    ///
    /// Rather than a specific argmax (these instruct models pick different
    /// raw-completion tokens — 12B ` a`, 31B ` France`), the invariant is
    /// *confidence*: fed "Paris is the capital of"
    /// (`[2, 50429, 563, 506, 5279, 529]`, same Gemma tokenizer, ids fed
    /// directly), a correctly-wired forward puts its top token well clear of
    /// the runner-up. Verified against real `llama-server` `/completion`,
    /// which is likewise confident here (12B ` a` at logprob -0.08, gap ~4;
    /// 31B/`E4B` ` France`, gap 8+). The bar (top − second ≥ 2.0 raw logits)
    /// sits far above a broken run's flat spread and far below every healthy
    /// model's margin. Run with `ORANGU_TEST_PLKV_MODEL=/path/to/
    /// gemma-4-{12B,31B}.gguf cargo test --release --bin orangu-server
    /// real_model_tests -- --ignored`.
    #[test]
    #[ignore]
    fn gemma4_per_layer_kv_dense_is_confident() {
        let path = std::env::var("ORANGU_TEST_PLKV_MODEL").expect("set ORANGU_TEST_PLKV_MODEL");
        let loaded = LoadedModel::open(std::path::Path::new(&path)).expect("load model");
        let model =
            GemmaModel::load_with_backend(&loaded, Arc::new(crate::engine::backend::CpuBackend))
                .expect("build model");
        assert!(
            !model.is_moe,
            "ORANGU_TEST_PLKV_MODEL should be a dense per-layer-KV model (12B/31B)"
        );
        assert!(
            model.layers.iter().any(|l| l.n_head_kv != model.n_head),
            "expected a per-layer head_count_kv array (some layer with n_head_kv != n_head)"
        );

        let mut cache = model.new_kv_cache(64);
        let tokens: Vec<u32> = vec![2, 50429, 563, 506, 5279, 529];
        let logits = model.forward(&mut cache, &tokens, 0, 0).expect("forward");
        let mut ranked: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let gap = ranked[0].1 - ranked[1].1;
        assert!(
            gap >= 2.0,
            "expected a confident prediction (top {} ahead by >=2.0), got gap {gap:.3}; \
             ranked[..5]={:?} — a flat top usually means per-layer head_count_kv regressed",
            ranked[0].0,
            &ranked[..5]
        );
    }

    /// The MoE sibling of the test above, for `gemma-4-26B-A4B`
    /// (`unsloth/gemma-4-26B-A4B-it-qat-GGUF`): exercises the routed-expert
    /// FFN path (dense shared MLP + softmax top-k experts, fused
    /// `ffn_gate_up_exps` + per-expert down `.scale`, `moe_ffn_result`)
    /// against real llama.cpp. The 26B-A4B uses the same Gemma tokenizer, so
    /// "The capital of France is" tokenizes to the identical ids (BOS=2
    /// prepended) as the dense test's; ids are fed directly to sidestep the
    /// tokenizer. Also asserts the model actually took the MoE path
    /// (`is_moe`), so a checkpoint that silently loaded dense-only wouldn't
    /// pass by accident.
    ///
    /// The bar is the **top-2 token set**, not the single argmax, because on
    /// this exact prompt the top two are a genuine near-tie that real
    /// llama.cpp resolves the *other* way — verified directly against
    /// `llama-server`'s `/completion` (`n_probs`) on this same GGUF: it
    /// returns ` Paris` (9079) at logprob -1.1775 then ` the` (506) at
    /// -1.2291 (Paris ahead by 0.05), with `.`/`\n`/` a` next. orangu
    /// produces the *identical ranking* except ` the` and ` Paris` swap at
    /// the very top (` the` ahead by ~0.07) — the two straddle a ~0.05-0.07
    /// gap, and orangu lands on the far side of it because llama.cpp's CPU
    /// `ggml_gelu` rounds through an f16 lookup table while this engine keeps
    /// GELU in full f32 (harmless everywhere the top logit isn't a tie — the
    /// dense test above, not a tie, matches llama.cpp's argmax exactly). So
    /// asserting a single argmax here would be asserting an f16-rounding
    /// artifact; the meaningful, stable invariant is that the forward puts
    /// exactly `{9079, 506}` on top, clear of the rest. Run with
    /// `ORANGU_TEST_MOE_MODEL=/path/to/gemma-4-26B-A4B.gguf cargo test
    /// --release --bin orangu-server real_model_tests -- --ignored` (a
    /// 26B-param model — expect several minutes on this engine's scalar
    /// per-row dequant).
    #[test]
    #[ignore]
    fn gemma4_moe_ranks_paris_and_the_after_capital_of_france() {
        let path = std::env::var("ORANGU_TEST_MOE_MODEL").expect("set ORANGU_TEST_MOE_MODEL");
        let loaded = LoadedModel::open(std::path::Path::new(&path)).expect("load model");
        let model =
            GemmaModel::load_with_backend(&loaded, Arc::new(crate::engine::backend::CpuBackend))
                .expect("build model");
        assert!(
            model.is_moe,
            "ORANGU_TEST_MOE_MODEL should be a MoE (A4B) checkpoint, but no layer had \
             ffn_gate_inp"
        );

        let mut cache = model.new_kv_cache(64);
        let tokens: Vec<u32> = vec![2, 818, 5279, 529, 7001, 563];
        let logits = model.forward(&mut cache, &tokens, 0, 0).expect("forward");
        let mut ranked: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top2: std::collections::HashSet<usize> = ranked[..2].iter().map(|&(i, _)| i).collect();
        assert_eq!(
            top2,
            std::collections::HashSet::from([9079usize, 506usize]),
            "expected the top-2 next tokens to be {{' Paris' 9079, ' the' 506}} (matching real \
             llama.cpp's top-2), got ranked[..5]={:?}",
            &ranked[..5]
        );
    }

    /// Cross-check against real llama.cpp (build 9959, `ggml-org/
    /// embeddinggemma-300M-GGUF:Q8_0`, `llama-server --embedding --pooling
    /// mean --ctx-size 2048`): tokenizing "The quick brown fox jumps over
    /// the lazy dog" via real llama.cpp's own `/tokenize?add_special=true`
    /// gives token ids `[2, 818, 3823, 8864, 37423, 38167, 1024, 506,
    /// 31770, 4799, 1]` — BOS=2 *and* EOS=1, since `embeddinggemma`'s
    /// `add_bos_token`/`add_eos_token` are both `true` (this is what
    /// motivated `Tokenizer::encode_for_embedding`, not just `encode`).
    /// `/embedding` on that same content returns the 768-value, L2-
    /// normalized vector in `testdata/embeddinggemma_reference.csv`.
    ///
    /// Feeds those exact token ids directly (sidestepping the tokenizer,
    /// matching this file's other real-model tests' convention) and runs
    /// this module's full non-causal path — symmetric-windowed SWA on 20 of
    /// 24 layers, `1/sqrt(head_dim)` attention scale, mean pooling,
    /// `dense_2`/`dense_3`, L2 norm — checking cosine similarity against
    /// the real vector rather than exact equality (independent Q8_0
    /// dequant and f32 accumulation-order implementations, not the same
    /// code path reordered). Run with `ORANGU_TEST_EMBEDDING_MODEL=/path/
    /// to/embeddinggemma-300M-Q8_0.gguf cargo test --release --bin
    /// orangu-server real_model_tests -- --ignored`.
    #[test]
    #[ignore]
    fn gemma_embedding_matches_real_llama_cpp() {
        let path =
            std::env::var("ORANGU_TEST_EMBEDDING_MODEL").expect("set ORANGU_TEST_EMBEDDING_MODEL");
        let loaded = LoadedModel::open(std::path::Path::new(&path)).expect("load model");
        let model =
            GemmaModel::load_with_backend(&loaded, Arc::new(crate::engine::backend::CpuBackend))
                .expect("build model");
        assert_eq!(loaded.config.architecture, "gemma-embedding");

        let tokens: Vec<u32> = vec![2, 818, 3823, 8864, 37423, 38167, 1024, 506, 31770, 4799, 1];
        let n_embd = model.config().n_embd;
        let hidden = model
            .forward_hidden_states(&tokens)
            .expect("forward_hidden_states");
        assert_eq!(hidden.len(), tokens.len() * n_embd);

        let mut pooled = vec![0f32; n_embd];
        for row in hidden.chunks(n_embd) {
            for (p, v) in pooled.iter_mut().zip(row.iter()) {
                *p += v;
            }
        }
        for v in pooled.iter_mut() {
            *v /= tokens.len() as f32;
        }
        let mut pooled = model
            .post_pool_projection(pooled)
            .expect("post_pool_projection");
        let norm = pooled.iter().map(|v| v * v).sum::<f32>().sqrt();
        for v in pooled.iter_mut() {
            *v /= norm;
        }

        let reference: Vec<f32> = include_str!("testdata/embeddinggemma_reference.csv")
            .trim()
            .split(',')
            .map(|v| v.parse().expect("reference fixture value"))
            .collect();
        assert_eq!(
            reference.len(),
            n_embd,
            "reference fixture has wrong length"
        );

        // 0.85, not something tighter, because this is a real cross-
        // implementation comparison (independent Q8_0 dequant, independent
        // f32 accumulation order over 24 layers, then a 4x-wide dense_2
        // expansion that amplifies small input differences) — not a GPU-
        // vs-CPU comparison of the *same* code path this project's other
        // tolerance-based checks use. A genuine structural bug (wrong
        // attention masking, wrong scale, wrong pooling) was ruled out by
        // varying each suspect independently (attention_scale 1.0 vs 1/
        // sqrt(head_dim), the SWA layer pattern's `dense_first` true vs
        // false) and observing the final cosine barely move (0.929-0.931)
        // — a real structural mismatch would show much more sensitivity to
        // getting these right. Also confirmed (the hard way): `llama-
        // server --pooling none`'s per-token output is *not* the raw pre-
        // dense hidden state — `llm_graph_context::build_dense_out` runs
        // unconditionally whenever `cparams.embeddings` is set and dense
        // tensors exist, regardless of pooling type, so it's already
        // dense-projected too.
        let cosine: f32 = pooled.iter().zip(&reference).map(|(a, b)| a * b).sum();
        assert!(
            cosine > 0.85,
            "cosine similarity to real llama.cpp's embedding was only {cosine}, expected > 0.85"
        );
    }

    /// Cross-checks `ModelForward::forward_batch_decode`
    /// (multiple independent sequences' decode steps fused into one call)
    /// against running `forward` independently for each sequence, on the
    /// real `E2B` model. Two separate, freshly prefilled sets of caches
    /// (rather than cloning one set) since `KvCache` isn't `Clone` —
    /// prefill is fully deterministic here (`forward`'s raw logits,
    /// argmax'd directly, no `Sampler`/RNG involved), so both sets reach
    /// identical starting state regardless. Run against both backends
    /// this project ships, expecting **bit-for-bit** equality on both:
    /// - On `CpuBackend`, both paths compute attention via the exact same
    ///   CPU loop, and `Backend::matmul`/`matmul_batch` compute every
    ///   `(row, token)` pair via an independent dot product (`CpuBackend::
    ///   matmul`'s own doc comment), so batching sequences together
    ///   doesn't change any individual result's arithmetic at all.
    /// - On `VulkanBackend` (skipped if no adapter is available),
    ///   `forward_batch_decode` now takes `GemmaModel::record_batched_
    ///   decode_forward` for a real batch — the *exact same* per-sequence
    ///   GPU chain (`record_one_sequence_decode`, including the same
    ///   `gpu_attention` WGSL kernel) `forward`'s own single-sequence path
    ///   uses, just recorded once per sequence into one shared submission
    ///   instead of a separate submission per sequence. Not two
    ///   independently-written implementations of the same math
    ///   converging within a tolerance — literally the same dispatches
    ///   and per-sequence buffers/bind groups, so bit-for-bit equality is
    ///   the right bar here too, not just a plausible one.
    #[test]
    #[ignore]
    fn forward_batch_decode_matches_independent_forward_calls_cpu() {
        let backend: Arc<dyn crate::engine::backend::Backend> =
            Arc::new(crate::engine::backend::CpuBackend);
        check_forward_batch_decode_matches_independent(backend);
    }

    #[test]
    #[ignore]
    fn forward_batch_decode_matches_independent_forward_calls_vulkan() {
        let Some(vulkan) = crate::engine::backend::vulkan::VulkanBackend::try_init() else {
            eprintln!("skipping: no Vulkan adapter available in this environment");
            return;
        };
        let backend: Arc<dyn crate::engine::backend::Backend> = Arc::new(vulkan);
        check_forward_batch_decode_matches_independent(backend);
    }

    fn check_forward_batch_decode_matches_independent(
        backend: Arc<dyn crate::engine::backend::Backend>,
    ) {
        let path = std::env::var("ORANGU_TEST_MODEL").expect("set ORANGU_TEST_MODEL");
        let loaded = LoadedModel::open(std::path::Path::new(&path)).expect("load model");
        let model = GemmaModel::load_with_backend(&loaded, backend).expect("build model");

        let prompts: Vec<Vec<u32>> = vec![
            vec![2, 818, 5279, 529, 7001, 563],
            vec![2, 818, 1963, 529, 5279, 3778, 563],
            vec![2, 818, 6870, 529, 8319, 563],
        ];

        let prefill = |model: &GemmaModel| -> (Vec<KvCache>, Vec<u32>) {
            let mut caches: Vec<KvCache> = prompts.iter().map(|_| model.new_kv_cache(64)).collect();
            let mut next = Vec::new();
            for (cache, prompt) in caches.iter_mut().zip(&prompts) {
                let logits = model.forward(cache, prompt, 0, 0).expect("prefill");
                let (top, _) = logits
                    .iter()
                    .copied()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap();
                next.push(top as u32);
            }
            (caches, next)
        };

        let (mut independent_caches, next_tokens) = prefill(&model);
        let (mut batched_caches, next_tokens_2) = prefill(&model);
        assert_eq!(next_tokens, next_tokens_2, "prefill is not deterministic");

        let mut expected = Vec::new();
        for (i, cache) in independent_caches.iter_mut().enumerate() {
            let pos = prompts[i].len();
            let logits = model
                .forward(cache, &[next_tokens[i]], pos, i)
                .expect("independent decode");
            expected.push(logits);
        }

        let mut items: Vec<_> = batched_caches
            .iter_mut()
            .enumerate()
            .map(|(i, cache)| crate::engine::arch::BatchDecodeItem {
                cache,
                token: next_tokens[i],
                start_pos: prompts[i].len(),
                greedy_sample: None,
                slot_id: i,
            })
            .collect();
        let outcomes = model
            .forward_batch_decode(&mut items)
            .expect("batched decode");

        assert_eq!(outcomes.len(), prompts.len());
        for (i, outcome) in outcomes.into_iter().enumerate() {
            let got = match outcome {
                crate::engine::arch::ForwardOutcome::Logits(l) => l,
                crate::engine::arch::ForwardOutcome::Token(_) => {
                    panic!("expected Logits — the batched path never GPU-samples")
                }
            };
            assert_eq!(expected[i].len(), got.len());
            for (j, (a, b)) in expected[i].iter().zip(got.iter()).enumerate() {
                // Bit-for-bit on both backends — see this test function's
                // own doc comment for why the Vulkan case is no longer
                // just "close": `record_batched_decode_forward` records
                // the *same* per-sequence GPU chain `forward` itself uses,
                // just sharing one submission across the batch.
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "sequence {i}, logit {j}: independent={a} batched={b}"
                );
            }
        }
    }

    /// A cheaper, stronger invariant than comparing against `forward`:
    /// `n` *identical* prompts, batched together, greedy-decoded for
    /// several sequential steps — every sequence must produce the exact
    /// same token trajectory as every other, at every step, trivially
    /// (same input, same deterministic greedy math, no RNG anywhere in
    /// this call chain), regardless of what the "correct" trajectory
    /// even is. Doesn't need a second, independent `forward` call to
    /// compare against — a single wrong output would still make two
    /// identical sequences disagree with *each other* — so this is a
    /// direct test of whether `Self::record_batched_decode_forward`
    /// keeps sequences correctly isolated across *many* calls (batch
    /// composition changing turn to turn is the norm in
    /// `engine::batch::BatchCoordinator`'s real usage, not the
    /// exception this test's own single-batch-call sibling above never
    /// exercises).
    #[test]
    #[ignore]
    fn forward_batch_decode_identical_prompts_stay_identical_over_many_steps_vulkan() {
        let Some(vulkan) = crate::engine::backend::vulkan::VulkanBackend::try_init() else {
            eprintln!("skipping: no Vulkan adapter available in this environment");
            return;
        };
        let backend: Arc<dyn crate::engine::backend::Backend> = Arc::new(vulkan);
        let path = std::env::var("ORANGU_TEST_MODEL").expect("set ORANGU_TEST_MODEL");
        let loaded = LoadedModel::open(std::path::Path::new(&path)).expect("load model");
        let model = GemmaModel::load_with_backend(&loaded, backend).expect("build model");

        const N: usize = 2;
        const STEPS: usize = 8;
        let prompt = vec![2u32, 818, 5279, 529, 7001, 563];

        let mut caches: Vec<KvCache> = (0..N).map(|_| model.new_kv_cache(64)).collect();
        let mut tokens = Vec::with_capacity(N);
        for cache in &mut caches {
            let logits = model.forward(cache, &prompt, 0, 0).expect("prefill");
            let (top, _) = logits
                .iter()
                .copied()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();
            tokens.push(top as u32);
        }
        assert!(
            tokens.iter().all(|&t| t == tokens[0]),
            "identical prompts must prefill to the identical first token, got {tokens:?}"
        );

        for step in 0..STEPS {
            let pos = prompt.len() + step;
            let mut items: Vec<_> = caches
                .iter_mut()
                .enumerate()
                .map(|(i, cache)| crate::engine::arch::BatchDecodeItem {
                    cache,
                    token: tokens[i],
                    start_pos: pos,
                    greedy_sample: None,
                    slot_id: i,
                })
                .collect();
            let outcomes = model
                .forward_batch_decode(&mut items)
                .expect("batched decode");
            assert_eq!(outcomes.len(), N);

            let mut next_tokens = Vec::with_capacity(N);
            for outcome in outcomes {
                let crate::engine::arch::ForwardOutcome::Logits(logits) = outcome else {
                    panic!("expected Logits — the batched path never GPU-samples");
                };
                let (top, _) = logits
                    .iter()
                    .copied()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap();
                next_tokens.push(top as u32);
            }
            assert!(
                next_tokens.iter().all(|&t| t == next_tokens[0]),
                "step {step}: identical prompts must stay identical, got {next_tokens:?} \
                 (pos={pos})"
            );
            tokens = next_tokens;
        }
    }
}
