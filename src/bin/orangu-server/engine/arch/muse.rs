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

//! Muse-Glimmer (`general.architecture = "muse-glimmer"`, e.g.
//! `unsloth/Muse-Glimmer-30B-GGUF`), confirmed against real upstream
//! `llama.cpp` source (`src/models/muse-glimmer.cpp`, fetched and read
//! directly, not guessed).
//!
//! A dense GQA decoder — `engine::arch::llama`'s block plus five things,
//! four of which some other architecture here already has, so most of this
//! module is composition over the shared pieces (`engine::attention`,
//! `tensor`'s rope/norm/activation kernels, `super::rms_norm_rows`) rather
//! than new math:
//!
//! - **Sandwich norms.** `attn_norm`/`post_attention_norm` around
//!   attention and `ffn_norm`/`post_ffw_norm` around the FFN — gemma2's
//!   arrangement, the post-norm applied to the sub-layer's output before
//!   its residual add. All four are present on every layer, which is what
//!   distinguishes this from `engine::arch::qwen35`'s two-norm block
//!   (where `post_attention_norm` *is* the pre-FFN norm and there is no
//!   `ffn_norm`). The two **post**-norms use their own epsilon,
//!   [`POST_NORM_EPS`], not the file's `attention.layer_norm_rms_epsilon`.
//! - **A sigmoid output gate on attention.** `attn_gate`
//!   (`[n_embd, n_head * head_dim]`) is projected from the same normed
//!   layer input as Q/K/V, and `sigmoid` of it multiplies attention's
//!   output element-wise *before* `attn_output` — the gate
//!   `engine::arch::kimi3`'s MLA layers and `engine::arch::qwen35`'s
//!   full-attention layers also carry (they fold it into the query
//!   projection; here it is its own tensor).
//! - **Per-head QK-norm**, `[head_dim]` RMSNorm weights on Q and K before
//!   the rotation, as in `gemma`/`qwen3`. Nothing extra is folded in here
//!   at run time: upstream notes this model's query-key scale factor was
//!   baked into `attn_q_norm`'s weights at conversion, and `attn_k_norm`
//!   is a vector of ones.
//! - **Alternating local/global attention.** A *scalar*
//!   `attention.sliding_window_pattern` (default 4 when absent) drives
//!   upstream's `llama_hparams::set_swa_pattern`: `il % n_pattern <
//!   n_pattern - 1` is sliding-window, so every `n_pattern`th layer
//!   attends the whole prefix. The same formula `gemma-embedding` uses at
//!   period 6.
//! - **RoPE on the sliding-window layers only** — the full-attention
//!   layers are NoPE, rotating neither Q nor K. This is the one mechanism
//!   with no counterpart elsewhere in this engine, and it is invisible
//!   from the file: nothing in the metadata or the tensor directory says
//!   a layer does not rotate.
//!
//! The tail applies both scalar logit transforms this engine has otherwise
//! seen separately: `logit_scale` (a plain multiply, as Command-R has) and
//! then `final_logit_softcapping` (`tanh`-based, as gemma2 has). The token
//! embeddings are **weightlessly** RMS-normed on the way in — an ordinary
//! `ggml_rms_norm` with no learned weight, where gemma instead scales by
//! `sqrt(n_embd)` and the llama family does nothing at all.
//!
//! Deliberately **not** taken: the whole-half fused Vulkan prefill chains
//! (`fused_attention_prefill`/`fused_post_attention_prefill`) that
//! `engine::arch::llama` and `engine::arch::gemma` run on. Neither can
//! express this layer — the pre-attention chain rotates unconditionally
//! and this architecture skips RoPE on a quarter of its layers, and the
//! post-attention chain takes a single epsilon for all three of its norms
//! where this needs [`POST_NORM_EPS`] for two of them and the file's own
//! epsilon for the third. So this is CPU-orchestrated like
//! `engine::arch::qwen35`, `glm`, `kimi3`, `deepseek4` and `qwen35moe`:
//! every matmul still dispatches to the `Backend` (GPU included), and
//! attention still goes wherever `engine::attention` decides, but the
//! layer is driven from the host.
//!
//! Multimodal input is out of scope, as it is for every architecture here:
//! the `mmproj-*.gguf` shipped beside these weights is a separate
//! `clip`-architecture model this engine does not load.

use anyhow::{Context, Result};
use std::sync::Arc;

use super::ModelForward;
use crate::engine::backend::{Backend, MatmulOp};
use crate::engine::kv_cache::KvCache;
use crate::engine::loader::{LoadedModel, ModelConfig, QuantMatrix};
use crate::engine::tensor;

/// The epsilon the two **post**-norms use, hardcoded upstream as
/// `post_norm_eps` rather than read from the file — which sets
/// `attention.layer_norm_rms_epsilon` (1e-5 for the 30B checkpoint) and is
/// what every *other* norm here uses. Three orders of magnitude apart, so
/// this is not a rounding choice somebody made twice.
const POST_NORM_EPS: f32 = 1e-8;

/// The sliding-window period when the file sets no
/// `attention.sliding_window_pattern`, matching upstream's own default for
/// this architecture.
///
/// Not a cosmetic fallback: the pattern decides which layers are
/// sliding-window, and *that* decides which layers rotate at all (see
/// [`GatedAttention::is_swa`]). Defaulting it to "no pattern" would leave a
/// file without the key with no RoPE anywhere.
const DEFAULT_SWA_PATTERN: usize = 4;

struct MuseLayer {
    attn_norm: Vec<f32>,
    /// Q/K/V, the output gate, QK-norm, and whether this layer is
    /// sliding-window — and so whether it rotates — three layers in every
    /// four at the default period.
    attn: GatedAttention,
    /// `post_attention_norm` — on attention's *output*, before the residual
    /// add, on top of the usual `attn_norm` pre-norm. Uses
    /// [`POST_NORM_EPS`].
    attn_post_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    ffn_gate: QuantMatrix,
    ffn_up: QuantMatrix,
    ffn_down: QuantMatrix,
    /// `post_ffw_norm` — the FFN's half of the same sandwich, same epsilon.
    ffn_post_norm: Vec<f32>,
}

pub struct MuseModel {
    config: ModelConfig,
    backend: Arc<dyn Backend>,
    tok_embeddings: QuantMatrix,
    output_norm: Vec<f32>,
    output_weight: QuantMatrix,
    layers: Vec<MuseLayer>,
    /// `attention.sliding_window` — the width a [`GatedAttention::is_swa`]
    /// layer attends.
    n_swa: usize,
    logits: LogitTransforms,
    /// Rotary width, base and pairing. `muse-glimmer` sits in upstream's
    /// `LLAMA_ROPE_TYPE_NORM` arm — pairs of *consecutive* head values,
    /// like `llama`/`mistral` and unlike the gemma and Qwen families this
    /// block shape otherwise resembles.
    rope: tensor::RopeParams,
}

/// The attention half both Muse architectures share: Q/K/V and a gate
/// projected from the same normed input, per-head QK-norm, rotation on the
/// sliding-window layers only, and `sigmoid(gate)` multiplied into
/// attention's output before `attn_output`.
///
/// `muse-spark` puts one thing in this sequence that `muse-glimmer` does
/// not — a weightless per-head RMSNorm on attention's output, before the
/// gate — which is [`GatedAttention::forward_into`]'s `out_norm_eps`.
pub(super) struct GatedAttention {
    pub(super) wq: QuantMatrix,
    pub(super) wk: QuantMatrix,
    pub(super) wv: QuantMatrix,
    /// `attn_gate`, `[n_embd, n_head * head_dim]` — projected from the same
    /// normed input as Q/K/V; `sigmoid` of it gates attention's output
    /// before `wo`.
    pub(super) w_gate: QuantMatrix,
    pub(super) wo: QuantMatrix,
    /// `[head_dim]`, the same vector for every head, applied before RoPE.
    pub(super) attn_q_norm: Vec<f32>,
    pub(super) attn_k_norm: Vec<f32>,
    /// Whether this layer attends a sliding window rather than the whole
    /// prefix.
    ///
    /// It also decides whether this layer rotates: upstream's `use_rope =
    /// hparams.is_swa(il)`. A full-attention layer here is NoPE, and its
    /// cached keys are therefore stored unrotated.
    pub(super) is_swa: bool,
}

/// The shape [`GatedAttention::forward_into`] needs, the same for every
/// layer of a model.
pub(super) struct GatedAttentionShape<'a> {
    pub(super) n_head: usize,
    pub(super) n_head_kv: usize,
    pub(super) head_dim: usize,
    /// The QK-norm's epsilon — the file's `attention.layer_norm_rms_epsilon`.
    pub(super) eps: f32,
    /// `attention.sliding_window`.
    pub(super) n_swa: usize,
    pub(super) rope: &'a tensor::RopeParams,
}

impl GatedAttention {
    pub(super) fn load(loaded: &LoadedModel, il: usize, is_swa: bool) -> Result<Self> {
        let get = |suffix: &str| -> Result<Vec<f32>> {
            let name = format!("blk.{il}.{suffix}");
            Ok(loaded
                .tensor(&name)
                .with_context(|| format!("loading {name}"))?
                .0)
        };
        let get_matrix = |suffix: &str| -> Result<QuantMatrix> {
            let name = format!("blk.{il}.{suffix}");
            loaded
                .matrix(&name)
                .with_context(|| format!("loading {name}"))
        };
        Ok(Self {
            wq: get_matrix("attn_q.weight")?,
            wk: get_matrix("attn_k.weight")?,
            wv: get_matrix("attn_v.weight")?,
            w_gate: get_matrix("attn_gate.weight")?,
            wo: get_matrix("attn_output.weight")?,
            attn_q_norm: get("attn_q_norm.weight")?,
            attn_k_norm: get("attn_k_norm.weight")?,
            is_swa,
        })
    }

    /// The layer's attention from its normed input `normed`
    /// (`[n_tokens, n_embd]`) to the output projection's result, appending
    /// each token's K/V to `layer_cache` on the way. `attn_out` is a
    /// caller-owned buffer reused across layers.
    ///
    /// `out_norm_eps` adds a weightless RMSNorm over each head of
    /// attention's output, before the gate — `muse-spark`'s `attn_o_norm`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_into(
        &self,
        backend: &dyn Backend,
        shape: &GatedAttentionShape<'_>,
        layer_cache: &mut crate::engine::kv_cache::LayerCache,
        attn_out: &mut Vec<f32>,
        normed: &[f32],
        n_tokens: usize,
        start_pos: usize,
        out_norm_eps: Option<f32>,
    ) -> Vec<f32> {
        let GatedAttentionShape {
            n_head,
            n_head_kv,
            head_dim,
            eps,
            n_swa,
            rope,
        } = *shape;
        let kv_dim = n_head_kv * head_dim;

        // Q, K, V and the gate are four independent projections of the
        // same normed input — one batched dispatch instead of four
        // sequential round trips (see `Backend::matmul_batch`).
        let mut qkvg = backend.matmul_batch(&[
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
            MatmulOp {
                x: normed,
                n_tokens,
                w: &self.w_gate,
            },
        ]);
        let gate = qkvg.pop().unwrap();
        let v = qkvg.pop().unwrap();
        let mut k = qkvg.pop().unwrap();
        let mut q = qkvg.pop().unwrap();

        tensor::rmsnorm_inplace(&mut q, &self.attn_q_norm, n_tokens * n_head, head_dim, eps);
        tensor::rmsnorm_inplace(
            &mut k,
            &self.attn_k_norm,
            n_tokens * n_head_kv,
            head_dim,
            eps,
        );

        // RoPE — on the sliding-window layers only; a full-attention
        // layer rotates neither Q nor K, and its keys go into the cache
        // exactly as projected. Then append each token's K/V in prompt
        // order, since a later token's attention reads them.
        for t in 0..n_tokens {
            let pos = start_pos + t;
            if self.is_swa {
                tensor::rope_apply_params_inplace(
                    &mut q[t * n_head * head_dim..(t + 1) * n_head * head_dim],
                    n_head,
                    head_dim,
                    pos,
                    None,
                    rope,
                );
                tensor::rope_apply_params_inplace(
                    &mut k[t * kv_dim..(t + 1) * kv_dim],
                    n_head_kv,
                    head_dim,
                    pos,
                    None,
                    rope,
                );
            }
            layer_cache.push(
                &k[t * kv_dim..(t + 1) * kv_dim],
                &v[t * kv_dim..(t + 1) * kv_dim],
            );
        }

        // `engine::attention` owns the GPU/CPU choice for every
        // architecture; the closure is the CPU window, and
        // `causal`/`n_swa` describe the same range to the GPU kernel.
        let is_swa = self.is_swa;
        crate::engine::attention::attention(
            attn_out,
            &q,
            layer_cache,
            &crate::engine::attention::Params {
                backend,
                // This layer's card — see `attention::Params::device`.
                device: self.wo.device(),
                n_head,
                n_head_kv,
                head_dim,
                scale: 1.0 / (head_dim as f32).sqrt(),
                causal: true,
                n_swa: if is_swa { n_swa } else { 0 },
                start_pos,
                n_tokens,
            },
            |t| attention_window(n_swa, is_swa, start_pos + t),
        );

        if let Some(out_eps) = out_norm_eps {
            super::rms_norm_rows(attn_out, head_dim, out_eps);
        }

        // The output gate, element-wise: attention's output and the gate
        // projection are both `[n_tokens, n_head * head_dim]`, so the
        // two line up position by position.
        debug_assert_eq!(attn_out.len(), gate.len());
        for (o, &g) in attn_out.iter_mut().zip(gate.iter()) {
            *o *= tensor::sigmoid(g);
        }

        backend.matmul(attn_out, n_tokens, &self.wo)
    }
}

/// Which layers attend a sliding window, from
/// `attention.sliding_window_pattern`: a per-layer array names each layer
/// directly (`muse-spark`), and a *scalar* period (`muse-glimmer`, default
/// [`DEFAULT_SWA_PATTERN`] when absent) drives upstream's
/// `llama_hparams::set_swa_pattern` — `il % n_pattern < n_pattern - 1` is
/// sliding-window, so every `n_pattern`th layer attends the whole prefix.
pub(super) fn swa_layers(loaded: &LoadedModel, n_layer: usize) -> Vec<bool> {
    let per_layer = loaded.metadata_array_u64("attention.sliding_window_pattern");
    let period = loaded
        .metadata_u64("attention.sliding_window_pattern")
        .map(|v| v as usize)
        .unwrap_or(DEFAULT_SWA_PATTERN);
    (0..n_layer)
        .map(|il| match &per_layer {
            Some(per_layer) => per_layer.get(il).copied().unwrap_or(0) != 0,
            None => period == 0 || il % period < period - 1,
        })
        .collect()
}

/// The inclusive `[start, end]` key/value position range a query at
/// absolute position `pos` may attend. Causal throughout — this is a
/// decoder — so the only variable is whether the window is the whole
/// prefix or the trailing `n_swa` positions, which is upstream's
/// `is_masked_swa` for `LLAMA_SWA_TYPE_STANDARD` (`p0 < p1 - n_swa` is
/// masked).
pub(super) fn attention_window(n_swa: usize, is_swa: bool, pos: usize) -> (usize, usize) {
    if is_swa && n_swa > 0 {
        (pos.saturating_sub(n_swa - 1), pos)
    } else {
        (0, pos)
    }
}

/// The two scalar logit transforms on the tail: `logit_scale` (a plain
/// multiply, as Command-R has) and then `final_logit_softcapping`
/// (`tanh`-based, as gemma2 has).
pub(super) struct LogitTransforms {
    /// `logit_scale`, multiplied into the output projection's result.
    logit_scale: Option<f32>,
    /// `final_logit_softcapping` — `x -> tanh(x / cap) * cap`, applied
    /// after the scale and never inside the layer stack.
    final_logit_softcapping: Option<f32>,
}

impl LogitTransforms {
    pub(super) fn from_gguf(loaded: &LoadedModel) -> Self {
        Self {
            logit_scale: loaded.metadata_f32("logit_scale"),
            final_logit_softcapping: loaded.metadata_f32("final_logit_softcapping"),
        }
    }

    /// `logit_scale` then `final_logit_softcapping`, in that order, over one
    /// `[n_vocab]` row — upstream's own order, and the one that makes the
    /// cap mean what it says: a scale applied *after* it would push the
    /// logits straight back outside the bound. Both are no-ops when the
    /// file sets neither.
    pub(super) fn apply(&self, logits: &mut [f32]) {
        if let Some(scale) = self.logit_scale {
            for v in logits.iter_mut() {
                *v *= scale;
            }
        }
        if let Some(cap) = self.final_logit_softcapping {
            for v in logits.iter_mut() {
                *v = (*v / cap).tanh() * cap;
            }
        }
    }
}

impl MuseModel {
    pub fn load_with_backend(loaded: &LoadedModel, backend: Arc<dyn Backend>) -> Result<Self> {
        let config = loaded.config.clone();
        let n_layer = config.n_layer;

        let n_swa = loaded
            .metadata_u64("attention.sliding_window")
            .context("missing attention.sliding_window")? as usize;
        let is_swa = swa_layers(loaded, n_layer);

        let tok_embeddings = loaded
            .matrix("token_embd.weight")
            .context("loading token_embd.weight")?;
        let (output_norm, _) = loaded
            .tensor("output_norm.weight")
            .context("loading output_norm.weight")?;
        // Upstream requires `output.weight` for this architecture; the tied
        // fallback costs nothing and matches every other module here.
        let output_weight = if loaded.has_tensor("output.weight") {
            loaded
                .matrix("output.weight")
                .context("loading output.weight")?
        } else {
            tok_embeddings.clone()
        };

        let mut layers = Vec::with_capacity(n_layer);
        for (i, &is_swa) in is_swa.iter().enumerate() {
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
            layers.push(MuseLayer {
                attn_norm: get("attn_norm.weight")?,
                attn: GatedAttention::load(loaded, i, is_swa)?,
                attn_post_norm: get("post_attention_norm.weight")?,
                ffn_norm: get("ffn_norm.weight")?,
                ffn_gate: get_matrix("ffn_gate.weight")?,
                ffn_up: get_matrix("ffn_up.weight")?,
                ffn_down: get_matrix("ffn_down.weight")?,
                ffn_post_norm: get("post_ffw_norm.weight")?,
            });
        }

        let rope = tensor::RopeParams {
            rope_dim: config.rope_dim,
            freq_base: config.rope_freq_base,
            layout: tensor::RopeLayout::Norm,
            ..tensor::RopeParams::default()
        };
        Ok(Self {
            config,
            backend,
            tok_embeddings,
            output_norm,
            output_weight,
            layers,
            n_swa,
            logits: LogitTransforms::from_gguf(loaded),
            rope,
        })
    }

    fn head_dim(&self) -> usize {
        self.config.head_dim
    }

    fn attention_shape(&self) -> GatedAttentionShape<'_> {
        GatedAttentionShape {
            n_head: self.config.n_head,
            n_head_kv: self.config.n_head_kv,
            head_dim: self.head_dim(),
            eps: self.config.rms_eps,
            n_swa: self.n_swa,
            rope: &self.rope,
        }
    }

    /// One decode step as **one submission per device**: every layer
    /// through `record_fused_layer`, the output norm and the logits
    /// projection recorded behind them, read back once. The chain takes
    /// what this architecture adds to the llama block — no rotation on a
    /// full-attention layer (`rope_dim: 0`), the sigmoid gate on
    /// attention's output (`attn_gate`), the post-norms' own epsilon
    /// (`post_norm_eps`) — so the whole layer stays on the device; before,
    /// each layer was five `Backend` calls with the activations home
    /// between them, two hundred submissions a token. `None` where the
    /// chain does not apply (a multi-token step, `ORANGU_MUSE_FUSED=0`).
    /// The logit scale and softcap are applied on the host, as `forward`
    /// applies them.
    fn record_decode_forward(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        slot_id: usize,
    ) -> Option<Vec<f32>> {
        if tokens.len() != 1 || !muse_fused() {
            return None;
        }
        let tok = tokens[0] as usize;
        if tok >= self.config.n_vocab {
            return None;
        }
        let n_embd = self.config.n_embd;
        let mut x = self.tok_embeddings.row(tok).to_vec();
        super::rms_norm_rows(&mut x, n_embd, self.config.rms_eps);

        let runs = super::decode_device_runs(
            self.backend.as_ref(),
            self.layers.iter().map(|layer| layer.attn.wo.device()),
        )?;
        let tail_device = self.output_weight.device();
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
                let mut logits =
                    vulkan.submit_and_readback_for(encoder, &self.output_weight, slot_id + 1);
                self.logits.apply(&mut logits);
                if vulkan.gpu_timestamps() {
                    vulkan.report_timestamps(start_pos, self.layers.len());
                }
                return Some(logits);
            }
            x = vulkan.submit_and_read_at(encoder, &buf, offset, n_embd);
        }
        // The output projection lives on a device that ran no layers: norm
        // and project there from the host copy of the residual.
        let vulkan = self.backend.as_wgpu_on(tail_device)?;
        let mut encoder = vulkan.new_encoder("orangu-server muse decode tail");
        let normed = vulkan.record_output_norm(
            &mut encoder,
            crate::engine::backend::vulkan::GpuInput::Cpu(&x),
            &self.output_norm,
            self.config.rms_eps,
            n_embd,
        );
        let _ = vulkan.record_full_matmul(
            &mut encoder,
            crate::engine::backend::vulkan::GpuInput::Gpu(&normed, 0),
            &self.output_weight,
            slot_id + 1,
        );
        let mut logits = vulkan.submit_and_readback_for(encoder, &self.output_weight, slot_id + 1);
        self.logits.apply(&mut logits);
        Some(logits)
    }

    /// The layers `layers` of one decode step on one device, from the
    /// residual `x_in` — with the output norm and logits projection behind
    /// them when `with_tail`, else ending at the last layer's residual.
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
            FfnActivation, FusedAttnProjection, FusedLayerInput, GpuInput, PassCursor, RopeYarn,
        };
        if !vulkan.prefill_attention_enabled() {
            return None;
        }
        let cfg = &self.config;
        let n_embd = cfg.n_embd;
        let head_dim = self.head_dim();
        let mut encoder = vulkan.new_encoder("orangu-server muse decode");
        let n_layer = self.layers.len();
        let ts = vulkan.begin_step_timestamps(&mut encoder, n_layer);
        let mut bufs: Vec<(wgpu::Buffer, u64)> = Vec::with_capacity(layers.len());
        // Submitted in chunks and recorded as one pass per chunk — see
        // `super::decode_chunk_ends` and `LlamaModel::record_decode_run`.
        let chunk_ends = super::decode_chunk_ends(layers.len());
        let mut cursor = PassCursor::new(&mut encoder);
        for il in layers.clone() {
            let layer = &self.layers[il];
            let x_input = match bufs.last() {
                Some((buf, offset)) => GpuInput::Gpu(buf, (*offset / 4) as usize),
                None => GpuInput::Cpu(x_in),
            };
            let (window_start, window) = if layer.attn.is_swa && self.n_swa > 0 {
                (
                    start_pos.saturating_sub(self.n_swa - 1),
                    Some(self.n_swa as u32),
                )
            } else {
                (0, None)
            };
            let ffn_gate_up = super::ffn_gate_up_pair(&layer.ffn_gate, &layer.ffn_up);
            let out = vulkan.record_fused_layer(
                &mut cursor,
                FusedLayerInput {
                    stop_at_ffn_norm: false,
                    x: x_input,
                    pairing: self.rope.layout,
                    yarn: RopeYarn::from_params(&self.rope),
                    activation: FfnActivation::Swiglu,
                    normalize_v: false,
                    attn_gate: Some(&layer.attn.w_gate),
                    attn_norm: &layer.attn_norm,
                    wq: &layer.attn.wq,
                    q_bias: None,
                    q_norm: Some(&layer.attn.attn_q_norm),
                    kv: Some(FusedAttnProjection {
                        wk: &layer.attn.wk,
                        wv: Some(&layer.attn.wv),
                        k_bias: None,
                        v_bias: None,
                        k_norm: Some(&layer.attn.attn_k_norm),
                    }),
                    n_head: cfg.n_head,
                    n_head_kv: cfg.n_head_kv,
                    head_dim,
                    // A full-attention layer of this architecture is NoPE:
                    // nothing to rotate, which the chain reads as zero
                    // rotated dimensions.
                    rope_dim: if layer.attn.is_swa {
                        self.rope.rope_dim
                    } else {
                        0
                    },
                    rope_freq_base: self.rope.freq_base,
                    freq_factors: None,
                    eps: cfg.rms_eps,
                    pos: start_pos,
                    window_start,
                    window,
                    scale: 1.0 / (head_dim as f32).sqrt(),
                    cache: &mut cache.layers[il],
                    wo: &layer.attn.wo,
                    attn_post_norm: Some(&layer.attn_post_norm),
                    ffn_norm: &layer.ffn_norm,
                    ffn_gate: Some(&layer.ffn_gate),
                    ffn_up: Some(&layer.ffn_up),
                    ffn_gate_up: ffn_gate_up.as_ref(),
                    ffn_down: Some(&layer.ffn_down),
                    ffn_post_norm: Some(&layer.ffn_post_norm),
                    ple: None,
                    layer_output_scale: None,
                    post_norm_eps: Some(POST_NORM_EPS),
                    batch_slot: slot_id,
                    attn_ts: ts.attn_slot(il, n_layer),
                },
            );
            ts.after_layer(cursor.encoder(), il);
            if il + 1 < layers.end && chunk_ends.contains(&(il - layers.start + 1)) {
                drop(cursor);
                let finished = std::mem::replace(
                    &mut encoder,
                    vulkan.new_encoder("orangu-server muse decode chunk"),
                );
                vulkan.submit_intermediate(finished);
                cursor = PassCursor::new(&mut encoder);
            }
            bufs.push(out);
        }
        drop(cursor);
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
        let (logits_buf, logits_offset) = vulkan.record_full_matmul(
            &mut encoder,
            GpuInput::Gpu(&normed, 0),
            &self.output_weight,
            slot_id + 1,
        );
        ts.finish(vulkan, &mut encoder, n_layer);
        Some((encoder, logits_buf, logits_offset))
    }

    /// Every layer, from the token embeddings to the last residual —
    /// `[n_tokens, n_embd]`. Shared by `forward`, `forward_all_logits` and
    /// `forward_hidden_states`, which differ only in what they do with it.
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
        let eps = cfg.rms_eps;

        let mut x = vec![0f32; n_tokens * n_embd];
        for (t, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            anyhow::ensure!(tok < cfg.n_vocab, "token id {tok} is out of vocab range");
            x[t * n_embd..(t + 1) * n_embd].copy_from_slice(&self.tok_embeddings.row(tok));
        }
        // A *weightless* RMSNorm on the embeddings, before any layer runs —
        // gemma scales by `sqrt(n_embd)` here and the llama family does
        // nothing, so neither of those shapes would produce this.
        super::rms_norm_rows(&mut x, n_embd, eps);

        // Grown once and reused across layers rather than allocated per
        // layer: at prefill widths this is megabytes a layer.
        let mut attn_out: Vec<f32> = Vec::new();
        // Same reuse as `attn_out`, replacing what used to be `x.clone()`
        // per norm — see `tensor::rmsnorm_into`.
        let mut normed: Vec<f32> = Vec::new();
        let mut ffn_normed: Vec<f32> = Vec::new();
        // Projection outputs, on the same principle — see
        // `Backend::matmul_into`.
        let mut ffn_out: Vec<f32> = Vec::new();
        let mut ffn_scratch = super::FfnScratch::default();

        for (il, layer) in self.layers.iter().enumerate() {
            tensor::rmsnorm_into(&mut normed, &x, &layer.attn_norm, n_tokens, n_embd, eps);

            // The two fused prefill chains, as `arch::llama` runs them:
            // attention with the gate applied on the device and its output
            // left there, then `wo`, both post-norms at their own epsilon,
            // and the FFN in one submission. `None` from either falls
            // through to the step-by-step sequence below.
            if muse_fused()
                && let Some(vulkan) = self.backend.as_wgpu_on(layer.attn.wo.device())
                && vulkan.prefill_fused_attention_enabled()
                && let Some(attn) = vulkan.fused_attention_prefill(
                    crate::engine::backend::vulkan::FusedAttnPrefillInput {
                        x_gpu: None,
                        attn_norm: None,
                        q_bias: None,
                        pairing: self.rope.layout,
                        yarn: crate::engine::backend::vulkan::RopeYarn::from_params(&self.rope),
                        normalize_v: false,
                        attn_gate: Some(&layer.attn.w_gate),
                        normed: &normed,
                        n_tokens,
                        start_pos,
                        wq: &layer.attn.wq,
                        q_norm: Some(&layer.attn.attn_q_norm),
                        kv: Some(crate::engine::backend::vulkan::FusedAttnPrefillKv {
                            k_bias: None,
                            v_bias: None,
                            wk: &layer.attn.wk,
                            k_norm: Some(&layer.attn.attn_k_norm),
                            wv: Some(&layer.attn.wv),
                        }),
                        n_head,
                        n_head_kv,
                        head_dim,
                        rope_dim: if layer.attn.is_swa {
                            self.rope.rope_dim
                        } else {
                            0
                        },
                        rope_freq_base: self.rope.freq_base,
                        freq_factors: None,
                        eps,
                        n_swa: if layer.attn.is_swa { self.n_swa } else { 0 },
                        causal: true,
                        scale: 1.0 / (head_dim as f32).sqrt(),
                        want_attn_out_host: false,
                    },
                    &mut cache.layers[il],
                )
                && let Some(out) = vulkan.fused_post_attention_prefill_rows(
                    crate::engine::backend::vulkan::AttnOutSrc::Gpu(
                        &attn.attn_out_buf,
                        0,
                        n_tokens,
                    ),
                    crate::engine::backend::vulkan::AttnOutSrc::Host(&x),
                    n_tokens,
                    &layer.attn.wo,
                    Some(&layer.attn_post_norm),
                    &layer.ffn_norm,
                    &layer.ffn_gate,
                    &layer.ffn_up,
                    &layer.ffn_down,
                    Some(&layer.ffn_post_norm),
                    eps,
                    crate::engine::backend::vulkan::FfnActivation::Swiglu,
                    None,
                    None,
                    Some(POST_NORM_EPS),
                )
            {
                x = out;
                continue;
            }

            let mut attn_proj = layer.attn.forward_into(
                self.backend.as_ref(),
                &self.attention_shape(),
                &mut cache.layers[il],
                &mut attn_out,
                &normed,
                n_tokens,
                start_pos,
                None,
            );
            tensor::rmsnorm_inplace(
                &mut attn_proj,
                &layer.attn_post_norm,
                n_tokens,
                n_embd,
                POST_NORM_EPS,
            );
            tensor::add_inplace(&mut x, &attn_proj);

            tensor::rmsnorm_into(&mut ffn_normed, &x, &layer.ffn_norm, n_tokens, n_embd, eps);
            // Was an open-coded copy of `super::swiglu_ffn` — same batched
            // gate/up, same in-place SiLU, same multiply, same down.
            super::swiglu_ffn_into(
                self.backend.as_ref(),
                &mut ffn_out,
                &mut ffn_scratch,
                &ffn_normed,
                n_tokens,
                &layer.ffn_gate,
                &layer.ffn_up,
                &layer.ffn_down,
            );
            tensor::rmsnorm_inplace(
                &mut ffn_out,
                &layer.ffn_post_norm,
                n_tokens,
                n_embd,
                POST_NORM_EPS,
            );
            tensor::add_inplace(&mut x, &ffn_out);
        }

        Ok(x)
    }

    /// Layer `il` as the device-resident stream runs it in a batched decode
    /// step.
    fn resident_layer(&self, il: usize) -> super::ResidentLayer<'_> {
        let layer = &self.layers[il];
        let head_dim = self.head_dim();
        super::ResidentLayer {
            attn_norm: &layer.attn_norm,
            wq: &layer.attn.wq,
            wk: &layer.attn.wk,
            wv: &layer.attn.wv,
            q_bias: None,
            k_bias: None,
            v_bias: None,
            q_norm: Some(&layer.attn.attn_q_norm),
            k_norm: Some(&layer.attn.attn_k_norm),
            pairing: self.rope.layout,
            yarn: crate::engine::backend::vulkan::RopeYarn::from_params(&self.rope),
            rope_dim: if layer.attn.is_swa {
                self.rope.rope_dim
            } else {
                0
            },
            rope_freq_base: self.rope.freq_base,
            freq_factors: None,
            n_swa: if layer.attn.is_swa { self.n_swa } else { 0 },
            scale: 1.0 / (head_dim as f32).sqrt(),
            wo: &layer.attn.wo,
            ffn_norm: &layer.ffn_norm,
            ffn_gate: &layer.ffn_gate,
            ffn_up: &layer.ffn_up,
            ffn_down: &layer.ffn_down,
            activation: crate::engine::backend::vulkan::FfnActivation::Swiglu,
            attn_gate: Some(&layer.attn.w_gate),
            attn_post_norm: Some(&layer.attn_post_norm),
            ffn_post_norm: Some(&layer.ffn_post_norm),
            post_norm_eps: Some(POST_NORM_EPS),
        }
    }

    /// The devices a batched decode step runs on, in layer order, with the
    /// layers each holds: the fused chains on, every layer on a device, and
    /// the prefill attention on each of them.
    fn batch_runs(
        &self,
    ) -> Option<
        Vec<(
            &crate::engine::backend::VulkanBackend,
            std::ops::Range<usize>,
        )>,
    > {
        if !muse_fused() {
            return None;
        }
        super::decode_device_runs(
            self.backend.as_ref(),
            self.layers.iter().map(|layer| layer.attn.wo.device()),
        )?
        .into_iter()
        .map(|(device, layers)| {
            let vulkan = self.backend.as_wgpu_on(device)?;
            vulkan
                .prefill_fused_attention_enabled()
                .then_some((vulkan, layers))
        })
        .collect()
    }

    /// One decode step for several sequences on the device-resident stream,
    /// the weights read once for all of them, as `arch::mistral`'s, one run
    /// of layers per device with the rows' residual brought home between
    /// them: `widths` as `super::run_layers_resident_batch` takes them
    /// (empty: one token a row), `samples` one per token. With `samples`, no
    /// logit scale to apply first and the head on the last run's device, the
    /// head and each row's sampler run there after the layers; otherwise
    /// the residual comes home for the head. The outcomes are one per token.
    fn decode_batch(
        &self,
        rows: &mut [super::DecodeRow<'_>],
        widths: &[usize],
        tokens: &[u32],
        samples: Option<&[Option<super::GreedySampleParams<'_>>]>,
    ) -> Result<Option<Vec<super::ForwardOutcome>>> {
        if rows.len() < 2 {
            return Ok(None);
        }
        let Some(runs) = self.batch_runs() else {
            return Ok(None);
        };
        let cfg = &self.config;
        let n_embd = cfg.n_embd;
        let n_rows = tokens.len();
        anyhow::ensure!(
            n_rows
                == if widths.is_empty() {
                    rows.len()
                } else {
                    widths.iter().sum()
                },
            "one token per row of the pass"
        );
        let mut x = vec![0f32; n_rows * n_embd];
        for (t, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            anyhow::ensure!(tok < cfg.n_vocab, "token id {tok} is out of vocab range");
            x[t * n_embd..(t + 1) * n_embd].copy_from_slice(&self.tok_embeddings.row(tok));
        }
        super::rms_norm_rows(&mut x, n_embd, cfg.rms_eps);
        // Each row of the pass is its request's: a verifying request's
        // tokens all take its slot.
        let slots: Vec<usize> = if widths.is_empty() {
            rows.iter().map(|r| r.slot).collect()
        } else {
            rows.iter()
                .zip(widths)
                .flat_map(|(r, &w)| std::iter::repeat_n(r.slot, w))
                .collect()
        };
        let pos = rows.first().map_or(0, |r| r.pos);
        let (last, before) = runs.split_last().expect("a model has layers");
        for (index, (vulkan, layers)) in before.iter().enumerate() {
            let Some(h_dev) = super::run_layers_resident_batch(
                vulkan,
                rows,
                widths,
                &x,
                cfg.n_head,
                cfg.n_head_kv,
                self.head_dim(),
                cfg.rms_eps,
                0,
                layers.clone(),
                |il| self.resident_layer(il),
            )?
            else {
                // Only the first run may decline: the caches before a later
                // one already hold this step's rows.
                anyhow::ensure!(
                    index == 0,
                    "layers {layers:?} cannot take a batched decode step"
                );
                return Ok(None);
            };
            x = vulkan.readback_rows(&h_dev, n_rows * n_embd);
            vulkan.finish_op_span(pos);
        }
        let (vulkan, layers) = last;
        let Some(h_dev) = super::run_layers_resident_batch(
            vulkan,
            rows,
            widths,
            &x,
            cfg.n_head,
            cfg.n_head_kv,
            self.head_dim(),
            cfg.rms_eps,
            0,
            layers.clone(),
            |il| self.resident_layer(il),
        )?
        else {
            anyhow::ensure!(
                before.is_empty(),
                "layers {layers:?} cannot take a batched decode step"
            );
            return Ok(None);
        };
        let tail_on_device = self
            .backend
            .as_wgpu_on(self.output_weight.device())
            .is_some_and(|tail| std::ptr::eq(tail, *vulkan));
        if let Some(samples) = samples
            .filter(|_| tail_on_device && vulkan.gpu_sample() && self.logits.logit_scale.is_none())
            && let Some(outcomes) = vulkan.decode_batch_tail(
                crate::engine::backend::vulkan::AttnOutSrc::Gpu(&h_dev, 0, n_rows),
                n_rows,
                &self.output_norm,
                cfg.rms_eps,
                &self.output_weight,
                self.logits.final_logit_softcapping,
                samples,
                &slots,
            )
        {
            vulkan.finish_op_span(pos);
            return Ok(Some(outcomes));
        }
        let mut h = vulkan.readback_rows(&h_dev, n_rows * n_embd);
        vulkan.finish_op_span(pos);
        tensor::rmsnorm_inplace(&mut h, &self.output_norm, n_rows, n_embd, cfg.rms_eps);
        let flat = self.backend.matmul(&h, n_rows, &self.output_weight);
        let n_vocab = flat.len() / n_rows;
        Ok(Some(
            flat.chunks_exact(n_vocab)
                .map(|row| {
                    let mut row = row.to_vec();
                    self.logits.apply(&mut row);
                    super::ForwardOutcome::Logits(row)
                })
                .collect(),
        ))
    }
}

/// `ORANGU_MUSE_FUSED=0` keeps every decode step on the step-by-step path —
/// one `Backend` call per matmul, the activations home between them — the
/// control arm for the fused chain. On unless `0`.
fn muse_fused() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::engine::env::flag_on_unless_disabled("ORANGU_MUSE_FUSED"))
}

impl ModelForward for MuseModel {
    fn vulkan_backend(&self) -> Option<&crate::engine::backend::vulkan::VulkanBackend> {
        self.backend.as_wgpu()
    }

    fn decode_step_is_chunked(&self) -> bool {
        true
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn new_kv_cache(&self, capacity: usize) -> KvCache {
        // Every layer caches the same shape — a sliding-window layer still
        // *stores* every position, it just attends fewer of them.
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

        // A one-token step goes through the fused chain when it can;
        // `None` falls through to the step-by-step path below.
        if let Some(logits) = self.record_decode_forward(cache, tokens, start_pos, slot_id) {
            return Ok(logits);
        }
        let x = self.run_layers(cache, tokens, start_pos)?;
        // Only the last token's hidden state is needed for next-token
        // logits — a batched prefill doesn't need every position's output.
        let last = &mut x[(n_tokens - 1) * n_embd..].to_vec();
        tensor::rmsnorm_inplace(last, &self.output_norm, 1, n_embd, cfg.rms_eps);
        let mut logits = self.backend.matmul(last, 1, &self.output_weight);
        self.logits.apply(&mut logits);
        Ok(logits)
    }

    fn batches_decode(&self) -> bool {
        self.batch_runs().is_some()
    }

    fn forward_decode_batch(
        &self,
        rows: &mut [super::DecodeRow<'_>],
        tokens: &[u32],
    ) -> Result<Option<Vec<Vec<f32>>>> {
        Ok(self.decode_batch(rows, &[], tokens, None)?.map(|outcomes| {
            outcomes
                .into_iter()
                .map(|outcome| match outcome {
                    super::ForwardOutcome::Logits(logits) => logits,
                    _ => unreachable!("a batch asked for no sampling returns logits"),
                })
                .collect()
        }))
    }

    fn forward_decode_batch_sampling(
        &self,
        rows: &mut [super::DecodeRow<'_>],
        tokens: &[u32],
        samples: &[Option<super::GreedySampleParams<'_>>],
    ) -> Result<Option<Vec<super::ForwardOutcome>>> {
        self.decode_batch(rows, &[], tokens, Some(samples))
    }

    fn batches_verify(&self) -> bool {
        super::batch_verify() && self.batch_runs().is_some()
    }

    fn forward_decode_batch_verify(
        &self,
        rows: &mut [super::DecodeRow<'_>],
        tokens: &[&[u32]],
        samples: &[Option<super::GreedySampleParams<'_>>],
    ) -> Result<Option<Vec<Vec<super::ForwardOutcome>>>> {
        if !self.batches_verify() || tokens.len() != rows.len() || samples.len() != rows.len() {
            return Ok(None);
        }
        let widths: Vec<usize> = tokens.iter().map(|t| t.len()).collect();
        let flat: Vec<u32> = tokens.iter().flat_map(|t| t.iter().copied()).collect();
        // A verifying row's tokens all come back as logits.
        let row_samples: Vec<Option<super::GreedySampleParams<'_>>> = samples
            .iter()
            .zip(&widths)
            .flat_map(|(sample, &w)| {
                let one = if w == 1 { *sample } else { None };
                std::iter::once(one).chain(std::iter::repeat_n(None, w - 1))
            })
            .collect();
        let Some(outcomes) = self.decode_batch(rows, &widths, &flat, Some(&row_samples))? else {
            return Ok(None);
        };
        let mut outcomes = outcomes.into_iter();
        Ok(Some(
            widths
                .iter()
                .map(|&w| outcomes.by_ref().take(w).collect())
                .collect(),
        ))
    }

    fn forward_all_logits(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        _slot_id: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let cfg = &self.config;
        let n_tokens = tokens.len();
        let n_embd = cfg.n_embd;
        let n_vocab = cfg.n_vocab;

        // One projection of every position through the output norm + vocab
        // matrix, batched — the weight-heavy `lm_head` read is amortized
        // across the whole draft in a single `matmul`, not one per position.
        let mut h = self.run_layers(cache, tokens, start_pos)?;
        tensor::rmsnorm_inplace(&mut h, &self.output_norm, n_tokens, n_embd, cfg.rms_eps);
        let flat = self.backend.matmul(&h, n_tokens, &self.output_weight);
        anyhow::ensure!(
            flat.len() == n_tokens * n_vocab,
            "output projection produced {} logits, expected {}",
            flat.len(),
            n_tokens * n_vocab
        );
        Ok((0..n_tokens)
            .map(|t| {
                let mut row = flat[t * n_vocab..(t + 1) * n_vocab].to_vec();
                self.logits.apply(&mut row);
                row
            })
            .collect())
    }

    fn forward_hidden_states(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        // A one-shot, whole-prompt pass — no KV cache reuse across calls,
        // the same convention `LlamaModel::forward_hidden_states` uses.
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
    /// The sliding-window pattern is read from a *scalar* period, and its
    /// answer is load-bearing twice over: it selects the attention window
    /// **and** whether the layer rotates at all. Upstream's formula makes
    /// the **last** layer of each group of `n_pattern` the full-attention
    /// one; the off-by-one alternative (`il % n_pattern != 0`, upstream's
    /// `dense_first` form, which several other architectures do use) picks
    /// a different set of layers entirely and is not distinguishable from
    /// this one by layer count.
    #[test]
    fn every_fourth_layer_is_full_attention_and_it_is_the_last_of_its_group() {
        let is_swa = |il: usize, n_pattern: usize| il % n_pattern < n_pattern - 1;
        let pattern: Vec<bool> = (0..8).map(|il| is_swa(il, 4)).collect();
        assert_eq!(
            pattern,
            vec![true, true, true, false, true, true, true, false]
        );
        assert_eq!((0..52).filter(|&il| is_swa(il, 4)).count(), 39);
    }
}

#[cfg(test)]
mod real_model_tests {
    use super::*;

    /// End-to-end against a real `unsloth/Muse-Glimmer-30B-GGUF` file.
    ///
    /// The assertion is on the *predicted token*, not on logits within a
    /// tolerance, for the reason `arch::llama`'s equivalent test spells
    /// out: a wrong-but-plausible forward pass produces fluent output a
    /// tolerance check can be talked into accepting, while a factual
    /// one-word answer cannot survive a broken attention or FFN. This
    /// architecture has an unusual number of ways to be wrong-but-plausible
    /// and every one of them was observed while it was being written — the
    /// wrong FFN activation, the wrong rotary pairing, rotating the
    /// full-attention layers, and skipping the weightless embedding norm
    /// each produce a model that loads, runs, and answers "The capital of
    /// France is" with a newline or a code token instead of " Paris".
    ///
    /// Run with `ORANGU_TEST_MUSE_MODEL=/path/to/Muse-Glimmer-30B-UD-Q4_K_XL.gguf
    /// cargo test --release --bin orangu-server muse::real_model_tests --
    /// --ignored`.
    #[test]
    #[ignore]
    fn muse_glimmer_predicts_paris_after_capital_of_france() {
        let path = std::env::var("ORANGU_TEST_MUSE_MODEL").expect("set ORANGU_TEST_MUSE_MODEL");
        let gguf = orangu::gguf::GgufFile::open(std::path::Path::new(&path)).expect("open gguf");
        let tokenizer =
            crate::engine::tokenizer::Tokenizer::from_gguf(&gguf).expect("build tokenizer");
        let loaded = LoadedModel::open(std::path::Path::new(&path)).expect("load model");
        assert_eq!(loaded.config.architecture, "muse-glimmer");
        let model =
            MuseModel::load_with_backend(&loaded, Arc::new(crate::engine::backend::CpuBackend))
                .expect("build model");
        // Three layers in four windowed, and those are exactly the ones
        // that rotate — asserted here as well as in the unit test above, so
        // a checkpoint whose pattern key differs is reported as such rather
        // than as a bad prediction.
        assert_eq!(
            model.layers.iter().filter(|l| l.attn.is_swa).count(),
            model.layers.len() * 3 / 4,
        );

        let tokens = tokenizer.encode("The capital of France is", true);
        let mut cache = model.new_kv_cache(tokens.len() + 1);
        let logits = model.forward(&mut cache, &tokens, 0, 0).expect("forward");
        let (top_id, _) = logits
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        assert_eq!(
            tokenizer.decode(&[top_id as u32]),
            " Paris",
            "top prediction was {:?}",
            tokenizer.decode(&[top_id as u32])
        );
    }
}

#[cfg(test)]
mod batch_tests {
    use super::*;

    /// A batch's rows take the GEMM kernels and a solo step the decode ones,
    /// and on this model that difference depends on the token: its own
    /// prompt pass over three tokens is 7e-3 to 8.8e-2 off its three decode
    /// steps.
    const TOLERANCE: f64 = 1.5e-1;

    fn model() -> Option<MuseModel> {
        let vulkan = crate::engine::backend::vulkan::VulkanBackend::try_init()?;
        let path = std::env::var("ORANGU_TEST_MUSE_MODEL").expect("set ORANGU_TEST_MUSE_MODEL");
        let loaded = LoadedModel::open(std::path::Path::new(&path)).expect("load model");
        Some(MuseModel::load_with_backend(&loaded, Arc::new(vulkan)).expect("build model"))
    }

    #[test]
    #[ignore]
    fn a_decode_batch_matches_each_rows_solo_step() {
        let Some(model) = model() else {
            println!("no Vulkan device; nothing to check");
            return;
        };
        super::super::batch_checks::decode_batch_matches_each_rows_solo_step(&model, TOLERANCE);
    }

    #[test]
    #[ignore]
    fn a_decode_batch_verifies_a_draft_as_its_own_pass_would() {
        let Some(model) = model() else {
            println!("no Vulkan device; nothing to check");
            return;
        };
        super::super::batch_checks::decode_batch_verifies_a_draft_as_its_own_pass_would(
            &model, TOLERANCE,
        );
    }
}
