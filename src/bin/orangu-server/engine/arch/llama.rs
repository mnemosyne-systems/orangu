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
use crate::npu_tool;

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
    ffn: Ffn,
}

/// What a layer's feed-forward is: one SwiGLU network, or a mixture of
/// them chosen per token.
///
/// The same block otherwise. `qwen3moe` (Qwen3-MoE, e.g.
/// `unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF`) is `qwen3` node for node —
/// the per-head Q/K norms this module already carries for it, GQA with
/// NEOX RoPE, the same norms and residuals — with `build_moe_ffn` where
/// the dense `build_ffn` would be (confirmed against upstream's
/// `src/models/qwen3moe.cpp`, read rather than guessed). So it is this
/// module's block with one substitution, not a family of its own: the
/// attention half, the KV cache, the split placement and the tail are
/// shared, and only the twelve lines that project the feed-forward differ.
enum Ffn {
    /// `ffn_gate`/`ffn_up`/`ffn_down` — every architecture here but
    /// `qwen3moe`.
    Dense {
        w_gate: QuantMatrix,
        w_up: QuantMatrix,
        w_down: QuantMatrix,
    },
    /// `ffn_gate_inp` routing to `ffn_*_exps` stacks. No shared expert and
    /// no selection bias: Qwen3-MoE has neither, and the file says so by
    /// carrying no `ffn_*_shexp` or `exp_probs_b` tensor.
    Moe(Box<MoeFfn>),
}

impl LlamaLayer {
    /// This layer's dense projections, or `None` where the feed-forward is
    /// a mixture.
    ///
    /// Every path that fuses the feed-forward onto the device — the decode
    /// chain, the resident prefill chain, the post-attention chain — asks
    /// through this and declines when it answers `None`: those chains
    /// compute a dense SwiGLU network themselves and have nowhere to put a
    /// per-token expert choice. A mixture layer takes the step-by-step
    /// route, where the routed feed-forward is one call
    /// (`arch::swiglu_moe_ffn`), exactly as the other mixture
    /// architectures do.
    fn dense(&self) -> Option<(&QuantMatrix, &QuantMatrix, &QuantMatrix)> {
        match &self.ffn {
            Ffn::Dense {
                w_gate,
                w_up,
                w_down,
            } => Some((w_gate, w_up, w_down)),
            Ffn::Moe(_) => None,
        }
    }
}

/// One layer's routed feed-forward.
struct MoeFfn {
    gate_inp: QuantMatrix,
    gate_exps: crate::engine::loader::ExpertQuantMatrix,
    up_exps: crate::engine::loader::ExpertQuantMatrix,
    down_exps: crate::engine::loader::ExpertQuantMatrix,
}

/// The scalar multipliers an IBM Granite checkpoint sprinkles through an
/// otherwise ordinary Llama block.
///
/// Granite 3.x *is* `arch::llama` node for node — RMSNorm, GQA with RoPE, a
/// separate-gate SwiGLU FFN, an optionally-tied output projection — and
/// then multiplies four things by constants the GGUF carries. They are
/// stored here rather than in `arch::granite` because they are the only
/// difference, and duplicating a thousand lines of forward pass to change
/// four numbers would leave two copies to keep in step.
///
/// [`Multipliers::NONE`] is what every other architecture routed here gets:
/// every field the identity, so the arithmetic is exactly what it was.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Multipliers {
    /// `granite.embedding_scale` (12 for 3.1-2B) — multiplies the token
    /// embeddings once, before the first layer.
    pub embedding: f32,
    /// `granite.residual_scale` (0.22) — multiplies each sub-layer's output
    /// *before* it is added to the residual stream, on both the attention
    /// and the FFN branch.
    ///
    /// Not the same thing as gemma's `layer_output_scale`, which the fused
    /// chain already carries: that scales `x` once after both adds, where
    /// this scales each branch before its own add. `x*(1 + a + f)` and
    /// `x + s*a + s*f` are different functions.
    pub residual: f32,
    /// `granite.logit_scale` (8) — the final logits are *divided* by this.
    pub logit: f32,
    /// `granite.attention.scale` (0.015625) — the softmax scale, replacing
    /// `1/sqrt(head_dim)`. `None` means derive it as everything else does.
    ///
    /// Granite does not merely rename the default: 3.1-2B has `head_dim =
    /// 64`, so the derived scale would be 0.125 and the checkpoint asks for
    /// 0.015625 — eight times smaller.
    pub attention: Option<f32>,
}

impl Multipliers {
    /// Every multiplier the identity — plain Llama, Qwen, Mistral.
    pub const NONE: Self = Self {
        embedding: 1.0,
        residual: 1.0,
        logit: 1.0,
        attention: None,
    };

    /// Whether these change anything at all.
    fn is_identity(&self) -> bool {
        *self == Self::NONE
    }
}

pub struct LlamaModel {
    config: ModelConfig,
    backend: Arc<dyn Backend>,
    tok_embeddings: QuantMatrix,
    output_norm: Vec<f32>,
    output_weight: QuantMatrix,
    layers: Vec<LlamaLayer>,
    /// How a mixture layer picks its experts — `expert_used_count` of them
    /// by a softmax over the router's logits, renormalized. Read from the
    /// file for a `qwen3moe`, and the defaults for every dense
    /// architecture here, which never ask.
    routing: super::ExpertRouting,
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
    /// [`Multipliers::NONE`] for everything but Granite — see
    /// `engine::arch::granite`.
    mul: Multipliers,
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
        "llama" | "mistral" | "granite" => tensor::RopeLayout::Norm,
        _ => tensor::RopeLayout::Neox,
    }
}

/// What [`LlamaModel::record_decode_run`] puts at the end of a run.
///
/// This was a `bool` — "append the tail" — and the two branches inside the
/// run disagreed about what the other value meant: without the NPU it left
/// the last layer's hidden state in the buffer, with the NPU it left
/// `output_norm` of it. Nothing caught that, because the only caller that
/// asked for no tail was the split path and the only machine with an NPU
/// here has one device. Three named states, so a branch cannot answer a
/// question it was not asked.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Tail {
    /// `output_norm` and the vocabulary projection: the buffer is logits.
    Device,
    /// `output_norm` only: the buffer is `[n_embd]`, for a caller that will
    /// read it back and project it on the host. See
    /// [`crate::engine::backend::tail_prefers_host`].
    Host,
    /// Neither: the buffer is this run's own hidden state, for the next
    /// device in a split to carry on from.
    None,
}

impl LlamaModel {
    pub fn load_with_backend(loaded: &LoadedModel, backend: Arc<dyn Backend>) -> Result<Self> {
        Self::load_with_multipliers(loaded, backend, Multipliers::NONE)
    }

    /// [`Self::load_with_backend`] for an architecture that scales parts of
    /// the block by constants — `engine::arch::granite`, and nothing else so
    /// far.
    pub fn load_with_multipliers(
        loaded: &LoadedModel,
        backend: Arc<dyn Backend>,
        mul: Multipliers,
    ) -> Result<Self> {
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
            let get_expert_matrix =
                |suffix: &str| -> Result<crate::engine::loader::ExpertQuantMatrix> {
                    let name = format!("blk.{i}.{suffix}");
                    loaded
                        .expert_matrix(&name)
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
                // The router is what says this layer is a mixture: a dense
                // file has no `ffn_gate_inp` at all.
                ffn: if loaded.has_tensor(&format!("blk.{i}.ffn_gate_inp.weight")) {
                    Ffn::Moe(Box::new(MoeFfn {
                        gate_inp: get_matrix("ffn_gate_inp.weight")?,
                        gate_exps: get_expert_matrix("ffn_gate_exps.weight")?,
                        up_exps: get_expert_matrix("ffn_up_exps.weight")?,
                        down_exps: get_expert_matrix("ffn_down_exps.weight")?,
                    }))
                } else {
                    Ffn::Dense {
                        w_gate: get_matrix("ffn_gate.weight")?,
                        w_up: get_matrix("ffn_up.weight")?,
                        w_down: get_matrix("ffn_down.weight")?,
                    }
                },
            });
        }

        // Only a mixture reads these, and only a mixture file carries
        // them: `build_moe_ffn`'s softmax gating with the selected weights
        // renormalized, which is what upstream's `qwen3moe` graph asks for
        // (`LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX`, `norm_w = true`). A
        // file that declares its own gating function is believed instead.
        let routing = super::ExpertRouting {
            n_expert_used: loaded.metadata_u64("expert_used_count").unwrap_or(0) as usize,
            gating: match loaded.metadata_u64("expert_gating_func") {
                Some(value) => super::ExpertGating::from_gguf(value)?,
                None => super::ExpertGating::Softmax,
            },
            weights_norm: loaded
                .metadata_u64("expert_weights_norm")
                .is_none_or(|v| v != 0),
            weights_scale: loaded.metadata_f32("expert_weights_scale").unwrap_or(1.0),
            groups: super::ExpertGroups::from_gguf(
                loaded,
                loaded.metadata_u64("expert_count").unwrap_or(0) as usize,
            )?,
        };

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
        // The streaming region the grouped expert GEMM will need, ahead of
        // the weights — see `super::reserve_expert_region`. A mixture whose
        // region is missing declines the grouped path and walks the groups
        // one at a time instead, which is slower than never having asked;
        // a file with no mixture layer reserves nothing.
        super::reserve_expert_region(
            backend.as_ref(),
            layers
                .iter()
                .filter_map(|l| match &l.ffn {
                    Ffn::Dense { .. } => None,
                    Ffn::Moe(moe) => Some(moe),
                })
                .flat_map(|moe| {
                    [
                        (moe.gate_exps.stack_matrix().raw_bytes().len()
                            + moe.up_exps.stack_matrix().raw_bytes().len())
                            as u64,
                        moe.down_exps.stack_matrix().raw_bytes().len() as u64,
                    ]
                }),
        );
        Ok(Self {
            config,
            backend,
            tok_embeddings,
            output_norm,
            output_weight,
            layers,
            routing,
            rope_freq_factors,
            rope,
            mul,
        })
    }

    /// The softmax scale this checkpoint wants — its own, or the usual
    /// `1/sqrt(head_dim)`.
    fn attn_scale(&self) -> f32 {
        self.mul
            .attention
            .unwrap_or_else(|| 1.0 / (self.head_dim() as f32).sqrt())
    }

    /// Multiplies one sub-layer's output before its residual add. A no-op
    /// for every architecture but Granite — see [`Multipliers::residual`].
    fn scale_residual(&self, branch: &mut [f32]) {
        if self.mul.residual != 1.0 {
            for v in branch.iter_mut() {
                *v *= self.mul.residual;
            }
        }
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
/// `ORANGU_LLAMA_RESIDENT_PREFILL=0` keeps a prefill chunk on the
/// step-by-step path — the control arm for the device-resident stream.
pub(crate) fn resident_prefill_off() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| {
        !crate::engine::env::flag_on_unless_disabled("ORANGU_LLAMA_RESIDENT_PREFILL")
    })
}

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
/// swept.
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
        crate::engine::backend::tuning_profile::env_var("ORANGU_FUSED_MIN_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(MIN_FUSED_TOKENS)
    })
}

pub fn no_fused_post_attention() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| crate::engine::env::flag_on("ORANGU_NO_FUSED_POST_ATTN"))
}

/// Where a decode step's wall clock goes across the seam, under
/// `ORANGU_NPU_TIME=1`.
///
/// The two halves that matter and that nothing else measures: bringing the
/// post-attention residual back from the GPU, and running the network on
/// the device. `record_decode_run` breaks the fused per-layer chain to make
/// the seam possible, so the readback is paid once per **layer** rather
/// than once per token, and whether that is affordable is the whole
/// question of whether decode belongs on the NPU at all.
fn seam_clock() -> Option<std::time::Instant> {
    seam_timing().then(std::time::Instant::now)
}

fn seam_elapsed(at: Option<std::time::Instant>) -> u64 {
    at.map_or(0, |t| t.elapsed().as_nanos() as u64)
}

fn seam_timing() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::engine::env::flag_on("ORANGU_NPU_TIME"))
}

/// Accumulates [`seam_clock`] and reports once per decode step.
fn record_seam_timing(read_ns: u64, device_ns: u64, n_layer: usize) {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    static READ: AtomicU64 = AtomicU64::new(0);
    static DEVICE: AtomicU64 = AtomicU64::new(0);
    static LAYERS: AtomicU64 = AtomicU64::new(0);
    if !seam_timing() {
        return;
    }
    READ.fetch_add(read_ns, Relaxed);
    DEVICE.fetch_add(device_ns, Relaxed);
    let seen = LAYERS.fetch_add(1, Relaxed) + 1;
    // Once per step, and only every eighth step so a long generation does
    // not drown the log it is being read out of.
    if !seen.is_multiple_of(n_layer as u64 * 8) {
        return;
    }
    let per_step = |v: u64| v as f64 / (seen as f64 / n_layer as f64) / 1e6;
    eprintln!(
        "orangu-server: [npu] seam over {} step(s): {:.1} ms/step reading back, \
         {:.1} ms/step on the device",
        seen / n_layer as u64,
        per_step(READ.load(Relaxed)),
        per_step(DEVICE.load(Relaxed))
    );
}

/// How much of a model the device must hold at width 1 before the decode
/// seam is worth breaking the fused GPU chain for.
///
/// Every layer pays the seam's submit-and-read whether or not the device
/// holds it, so the fraction that *is* held is what decides. Set so that a
/// checkpoint losing its massive-activation layer — one of 16 on llama 3.2
/// 1B — still uses the device, while one that lost half of itself does not.
///
/// With 15 of 16 layers held, llama 3.2 1B `Q8_0` decodes at 7.11 tok/s
/// against 7.44 with the device off, so the seam is close to free at that
/// coverage and the feature is at least *reachable* on the family. It was
/// `100` first, which meant one unrepresentable block took decode off the
/// device for every llama checkpoint there is.
const DECODE_SEAM_COVERAGE_PERCENT: usize = 80;

/// Diagnostic: take the decode seam, but run the network on the host.
///
/// `ORANGU_NPU_DECODE_HOST_FFN=1` keeps every other part of the seam — the
/// fused layer stopping at `ffn_norm`, the per-layer submit-and-read, the
/// host norm and the host residual add — and computes the feed-forward
/// network the way the backend would. It bisects a broken decode in one
/// run: wrong with this on is a seam that is wired wrong, wrong only with
/// it off is the device's arithmetic. Nothing but a measurement should set
/// it, and it is slower than either.
pub fn decode_seam_host_ffn() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| crate::engine::env::flag_on("ORANGU_NPU_DECODE_HOST_FFN"))
}

/// Diagnostic: measure the device against the backend on every layer's
/// **real** input, in prefill and in decode alike.
///
/// `ORANGU_NPU_CHECK=1` runs the network both ways at each layer,
/// reports what they disagree by, and keeps the backend's answer — so the
/// trajectory stays on the path the model would actually have taken and
/// every layer is measured on a correct input. That separates a block that
/// is wrong from a block that is merely being fed a hidden state some
/// earlier block already ruined.
///
/// The number to compare it against is the one the compiler recorded from
/// captured activations. A block that measures well there and badly here is
/// being asked for something its calibration never saw.
pub fn npu_block_check() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| crate::engine::env::flag_on("ORANGU_NPU_CHECK"))
}

/// One layer's device-against-backend disagreement, with the peak magnitude
/// of the input that produced it.
///
/// `|x| peak` is the diagnostic half: a projection on this device holds one
/// `uint8` activation scale chosen when the block was compiled, so an input
/// that runs past the calibrated range saturates rather than rounds, and
/// saturation is the failure that destroys a result instead of blunting it.
fn report_npu_block_error(
    layer: usize,
    tokens: usize,
    x: &[f32],
    device: &[f32],
    reference: &[f32],
) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static REPORTED: AtomicUsize = AtomicUsize::new(0);
    // A few decode steps' worth: enough to see whether the device is off on
    // the very first generated token or only once the trajectory has moved.
    if REPORTED.fetch_add(1, Ordering::Relaxed) >= 256 {
        return;
    }
    let (mut num, mut den, mut dev_sq, mut dot) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (d, r) in device.iter().zip(reference) {
        num += f64::from(d - r).powi(2);
        den += f64::from(*r).powi(2);
        dev_sq += f64::from(*d).powi(2);
        dot += f64::from(*d) * f64::from(*r);
    }
    let rms = if den > 0.0 { (num / den).sqrt() } else { 0.0 };
    // The two numbers that say *how* it is wrong rather than how much.
    // `gain` is the device's own magnitude against the reference's, and
    // `cos` is how much of the right direction is in it at all. Zero gain
    // is a block returning nothing; gain far from one with `cos` near one
    // is a scale that did not cancel; low `cos` is arithmetic.
    let gain = if den > 0.0 {
        (dev_sq / den).sqrt()
    } else {
        0.0
    };
    let cos = if dev_sq > 0.0 && den > 0.0 {
        dot / (dev_sq.sqrt() * den.sqrt())
    } else {
        0.0
    };
    let peak = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    eprintln!(
        "orangu-server: [npu] check layer {layer} at {tokens} tokens: \
         {:.1}% rms, gain {gain:.3}, cos {cos:.3}, |x| peak {peak:.2}",
        rms * 100.0
    );
}

/// `ORANGU_MOE_DECODE_CHAIN=0` sends a mixture's decode step back to the
/// step-by-step path, where attention, `wo` and the residual add are three
/// host turns instead of one submission.
///
/// The control arm, and the only way to compare the two on one binary: a
/// mixture either has the seam or it does not, so without this the
/// comparison would be two builds — and two builds differ by more than the
/// thing under test.
fn moe_decode_chain() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::engine::env::flag_on_unless_disabled("ORANGU_MOE_DECODE_CHAIN"))
}
impl LlamaModel {
    /// One decode step as a single GPU submission, or `None` when this model or
    /// this step is not one the fused chain can describe.
    ///
    /// The hidden state never returns to the host: each layer's output buffer is
    /// the next layer's input, so depth costs submissions nothing. That is the
    /// difference between an engine that can fill the device and one that
    /// cannot — the generic path costs one GPU round trip per layer per chain
    /// and never passes 66% engine occupancy however many concurrent requests
    /// it is given, while this form reaches 98% with two.
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
        tail: Tail,
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
            None,
            start_pos,
            slot_id,
            tail,
        )
    }

    /// Whether this model's vocabulary projection belongs on the host — see
    /// [`crate::engine::backend::tail_prefers_host`], which runs decode steps
    /// both ways and keeps the faster rather than inferring it from the
    /// weight's size.
    fn tail_on_host(&self) -> bool {
        crate::engine::backend::tail_prefers_host()
    }

    /// The vocabulary projection on the host, from a run that stopped after
    /// `output_norm`.
    fn host_tail(&self, normed: &[f32]) -> Vec<f32> {
        crate::engine::backend::CpuBackend.matmul(normed, 1, &self.output_weight)
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
        first: Option<&wgpu::Buffer>,
        start_pos: usize,
        slot_id: usize,
        tail: Tail,
    ) -> Option<(wgpu::CommandEncoder, wgpu::Buffer, u64)> {
        use crate::engine::backend::vulkan::{
            FfnActivation, FusedAttnProjection, FusedLayerInput, GpuInput, PassCursor, RopeYarn,
        };

        if no_fused_qkv() || no_fused_post_attention() {
            return None;
        }
        // The chain does its own embedding lookup and its own residual adds,
        // and has no term for scaling either. Granite needs both scaled, so
        // it takes the step-by-step path in `run_layers` instead — correct
        // and slower, rather than fast and quietly wrong.
        if !self.mul.is_identity() {
            return None;
        }
        if !vulkan.prefill_attention_enabled() {
            return None;
        }
        let cfg = &self.config;
        let n_embd = cfg.n_embd;
        let head_dim = self.head_dim();

        // **Where the recording's elapsed time goes**, under
        // `ORANGU_PREFILL_TRACE`. The stage timer puts this call at a third
        // of a small model's token while a CPU profile puts its *work* at a
        // fortieth of that, so most of it is the thread blocked rather than
        // busy — and the three candidates block in different places. The
        // first layer is timed apart from the rest because that is the
        // discriminator: a host waiting for the previous submission's
        // resources to retire pays at the token's first allocation, where a
        // per-dispatch driver cost is spread evenly over the layers.
        // **`device.record` is opened here, not by the caller, because the
        // chain submits from inside its own loop.** A guard held across
        // the whole call charges those submissions to recording, which is
        // how half this stage came to be submission time wearing the wrong
        // name. Closed and reopened around each one instead; the stages
        // accumulate, so several openings report as one total.
        let mut recording =
            crate::engine::decode_stages::enter(crate::engine::decode_stages::Stage::DeviceRecord);
        let trace = super::gemma::submission_trace();
        let t_pre = std::time::Instant::now();
        let t_enc = std::time::Instant::now();
        let mut encoder = vulkan.new_encoder("orangu-server llama decode");
        let enc_ms = t_enc.elapsed().as_secs_f64() * 1000.0;
        let (mut first_ms, mut rest_ms) = (0.0f64, 0.0f64);
        let mut pre_ms = 0.0f64;
        vulkan.begin_op_step(&mut encoder);
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
        // The NPU takes the feed-forward network when it holds this model's
        // blocks at this width, and the GPU keeps everything else. Measured
        // on Llama 3.2 3B: the FFN is 473 ms of a 582 ms decode step — 16.9
        // ms a layer — against 3.15 ms for the same block on the device.
        //
        // The cost is one submit-and-read per layer instead of one per
        // token, which `_scratch_measure_submit_roundtrip` puts at 0.199 ms.
        // 27 extra round trips is 5.4 ms against ~385 ms saved.
        //
        // llama has neither an attention post-norm nor an FFN post-norm, so
        // the part this has to do on the host is a residual add. An
        // architecture with either — gemma — cannot use this path as it
        // stands; `FusedPostAttentionInput::stop_at_ffn_norm` says so.
        // **Nearly every layer.** Taking this seam costs a submit-and-read
        // for the whole step rather than one per token, and a layer the
        // device does not hold pays that round trip and then computes the
        // network on the backend anyway — so a set with half the model
        // missing is worse than no set at all, and testing layer zero alone
        // was not enough.
        //
        // Not *every* layer, though, which is what this asked for first.
        // Llama's layer 1 carries the family's massive activations: its
        // `silu(gate) * up` is peaked enough that one channel sets the
        // `uint8` range and the rest quantizes to nothing, so the block
        // comes back exactly zero and `NpuFfn` withdraws it. That is one
        // layer of 16 on llama 3.2 1B and one of 28 on the 3B, and
        // demanding all of them meant a single unrepresentable block took
        // decode off the device for every checkpoint in the family.
        let npu = crate::engine::npu_ffn_service().filter(|npu| {
            let held = layers.clone().filter(|il| npu.has(*il, 1)).count();
            held * 100 >= layers.len() * DECODE_SEAM_COVERAGE_PERCENT
        });
        let mut host_x: Vec<f32> = Vec::new();
        let mut ffn_scratch = super::FfnScratch::default();
        let mut ffn_normed: Vec<f32> = Vec::new();
        let mut ffn_out: Vec<f32> = Vec::new();
        // Only ever filled under `npu_block_check`; an ordinary run leaves
        // it empty and never allocates it.
        let mut ffn_device: Vec<f32> = Vec::new();
        // The same, for the width-16 comparison the check makes.
        let (mut tiled, mut wide): (Vec<f32>, Vec<f32>) = (Vec::new(), Vec::new());
        // The layers after which what is recorded so far is submitted
        // (`super::decode_chunk_ends`): the device starts on the first
        // layer while the host records the rest, and the token's recording
        // stops standing in front of its execution. One compute pass per
        // chunk rather than per layer — nothing between two layers needs
        // the encoder, and a pass boundary is device time. The NPU seam
        // submits per layer on its own and keeps the single pass.
        // A mixture's experts are a per-token choice this chain cannot
        // record, so it stops at the norm on every layer and runs them at
        // the seam — the same shape the NPU seam uses, and with the same
        // consequence for chunking: a seam submits per layer already.
        // **Declined outright when the seam is off**, not half-taken: the
        // chain would still stop at the norm on a mixture layer (it has no
        // dense network to record) while the loop fed the next layer from a
        // device buffer the seam never wrote. That is not the step path,
        // it is a third thing that reads a stale residual — measured at a
        // third of the rate and emitting a stop token after nineteen.
        if self.is_moe() && !moe_decode_chain() {
            return None;
        }
        let moe_seam = self.is_moe();
        let chunk_ends = if npu.is_some() || moe_seam {
            Vec::new()
        } else {
            super::decode_chunk_ends(layers.len())
        };
        let mut cursor = PassCursor::new(&mut encoder);
        if trace {
            pre_ms = t_pre.elapsed().as_secs_f64() * 1000.0;
        }
        let t_tail = std::time::Instant::now();
        for il in layers.clone() {
            let layer = &self.layers[il];
            let x_input = if (npu.is_some() || moe_seam) && !host_x.is_empty() {
                GpuInput::Cpu(&host_x)
            } else {
                match (bufs.last(), first) {
                    (Some((buf, offset)), _) => GpuInput::Gpu(buf, (*offset / 4) as usize),
                    (None, Some(buf)) => GpuInput::Gpu(buf, 0),
                    (None, None) => GpuInput::Cpu(x_in),
                }
            };
            // `None` on a mixture layer, which is what tells the chain to
            // stop at the norm and hand back the post-attention residual.
            let dense = layer.dense();
            let ffn_gate_up = dense.and_then(|(gate, up, _)| super::ffn_gate_up_pair(gate, up));
            let t_layer = trace.then(std::time::Instant::now);
            let out = vulkan.record_fused_layer(
                &mut cursor,
                FusedLayerInput {
                    stop_at_ffn_norm: npu.is_some() || dense.is_none(),
                    x: x_input,
                    // `llama`/`mistral` are NORM; `qwen2` and the rest NEOX.
                    pairing: self.rope.layout,
                    // Derived rather than asserted identity: this family sets
                    // no YaRN today, and `from_params` keeps the chain correct
                    // rather than merely lucky if that changes.
                    yarn: RopeYarn::from_params(&self.rope),
                    // This family is SwiGLU throughout, has no post-norm on
                    // either residual, and — the convention that is invisible
                    // from every shape — does not normalize V.
                    activation: FfnActivation::Swiglu,
                    normalize_v: false,
                    attn_norm: &layer.attn_norm,
                    wq: &layer.wq,
                    q_bias: layer.q_bias.as_deref(),
                    // **Passed, not `None`.** These were hardcoded `None`
                    // under a comment claiming this family has no per-head
                    // Q/K norms. `LlamaLayer::q_norm` exists precisely
                    // because Qwen3 and Qwen3VL do, and `run_layers` applies
                    // them — so prefill was right and decode quietly skipped
                    // them. Nothing about that is visible in a shape: the
                    // prompt is processed correctly and generation is token
                    // soup from the first token. Qwen3-1.7B answered "what
                    // is a hash table" with `บทuyếtuyếtжеuyết}}{{Про胞...`.
                    q_norm: layer.q_norm.as_deref(),
                    kv: Some(FusedAttnProjection {
                        wk: &layer.wk,
                        wv: Some(&layer.wv),
                        k_bias: layer.k_bias.as_deref(),
                        v_bias: layer.v_bias.as_deref(),
                        k_norm: layer.k_norm.as_deref(),
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
                    scale: self.attn_scale(),
                    cache: &mut cache.layers[il],
                    wo: &layer.wo,
                    attn_post_norm: None,
                    ffn_norm: &layer.ffn_norm,
                    ffn_gate: dense.map(|(gate, _, _)| gate),
                    ffn_up: dense.map(|(_, up, _)| up),
                    ffn_gate_up: ffn_gate_up.as_ref(),
                    ffn_down: dense.map(|(_, _, down)| down),
                    ffn_post_norm: None,
                    ple: None,
                    layer_output_scale: None,
                    attn_gate: None,
                    post_norm_eps: None,
                    batch_slot: slot_id,
                    attn_ts: ts.attn_slot(il, n_layer),
                },
            );
            if let Some(at) = t_layer {
                let ms = at.elapsed().as_secs_f64() * 1000.0;
                if il == layers.start {
                    first_ms = ms;
                } else {
                    rest_ms += ms;
                }
            }
            ts.after_layer(cursor.encoder(), il);
            let chunk_done = il + 1 < layers.end && chunk_ends.contains(&(il - layers.start + 1));
            if chunk_done {
                // Chunk boundary: what is recorded so far executes while
                // the next chunk is recorded. The final chunk carries the
                // tail and is returned unsubmitted for the caller's own
                // submit-and-read.
                drop(cursor);
                let finished = std::mem::replace(
                    &mut encoder,
                    vulkan.new_encoder("orangu-server llama decode chunk"),
                );
                // Out of `device.record` for the submission, and back in
                // after it — see the guard's comment above.
                drop(recording.take());
                {
                    let _submit = crate::engine::decode_stages::enter(
                        crate::engine::decode_stages::Stage::DeviceSubmit,
                    );
                    vulkan.submit_intermediate(finished);
                }
                recording = crate::engine::decode_stages::enter(
                    crate::engine::decode_stages::Stage::DeviceRecord,
                );
                cursor = PassCursor::new(&mut encoder);
            }

            // With the FFN on the device, `out` is `x1` — the post-attention
            // residual — rather than the layer's output. Finish the layer
            // here: normalise, run the network, add the residual back.
            // The NPU seam runs the *dense* network at the seam, so it
            // needs this layer's three projections; a mixture never reaches
            // here, because `npu` is only set for a model whose blocks the
            // device holds and those are dense.
            if let (Some(npu), Some((w_gate, w_up, w_down))) = (npu, dense) {
                drop(cursor);
                let finished = std::mem::replace(
                    &mut encoder,
                    vulkan.new_encoder("orangu-server llama decode layer"),
                );
                let t_read = seam_clock();
                host_x = vulkan.submit_and_read_at(finished, &out.0, out.1, n_embd);
                let read_ns = seam_elapsed(t_read);
                cursor = PassCursor::new(&mut encoder);

                ffn_normed.clear();
                ffn_normed.extend_from_slice(&host_x);
                crate::engine::tensor::rmsnorm_inplace(
                    &mut ffn_normed,
                    &layer.ffn_norm,
                    1,
                    n_embd,
                    cfg.rms_eps,
                );
                // Both ways, when a measurement asked for it: the device's
                // answer is compared against the backend's and then thrown
                // away, so every layer is judged on an input the model would
                // really have produced. See `npu_block_check`.
                let checked = if npu_block_check() {
                    npu.forward_into(il, 1, &ffn_normed, &mut ffn_device)
                } else {
                    false
                };
                let t_dev = seam_clock();
                let on_device = !checked
                    && !decode_seam_host_ffn()
                    && npu.forward_into(il, 1, &ffn_normed, &mut ffn_out);
                record_seam_timing(read_ns, seam_elapsed(t_dev), n_layer);
                if on_device {
                    crate::engine::tensor::add_inplace(&mut host_x, &ffn_out);
                } else {
                    // The device declined or failed. `host_x` is `x1` and the
                    // network still has to run, so fall back to the backend
                    // for this layer rather than dropping it.
                    super::swiglu_ffn_into(
                        self.backend.as_ref(),
                        &mut ffn_out,
                        &mut ffn_scratch,
                        &ffn_normed,
                        1,
                        w_gate,
                        w_up,
                        w_down,
                    );
                    if checked {
                        report_npu_block_error(il, 1, &ffn_normed, &ffn_device, &ffn_out);
                        // **The same row through this layer's other graph.**
                        // A width-16 graph fed sixteen copies of one row
                        // computes the same function on each of them, so its
                        // first row is directly comparable with the width-1
                        // answer above. Two numbers that differ are a graph
                        // problem; two that agree are a quantization problem
                        // the width does not change.
                        if npu.has(il, npu_tool::prefill_width()) {
                            tiled.clear();
                            for _ in 0..npu_tool::prefill_width() {
                                tiled.extend_from_slice(&ffn_normed);
                            }
                            if npu.forward_into(il, npu_tool::prefill_width(), &tiled, &mut wide) {
                                wide.truncate(ffn_out.len());
                                report_npu_block_error(il, 16, &ffn_normed, &wide, &ffn_out);
                            }
                        }
                    }
                    crate::engine::tensor::add_inplace(&mut host_x, &ffn_out);
                }
                continue;
            }
            // The mixture seam: the chain stopped at the norm because a
            // routed feed-forward is a per-token choice it cannot record,
            // so `out` is `x1`. Normalise it, route it, add the result
            // back — the same three steps the step-by-step path takes,
            // with attention, `wo` and the residual add already done in
            // one submission rather than three host turns around them.
            if let Ffn::Moe(moe) = &layer.ffn {
                drop(cursor);
                let finished = std::mem::replace(
                    &mut encoder,
                    vulkan.new_encoder("orangu-server llama decode moe layer"),
                );
                host_x = vulkan.submit_and_read_at(finished, &out.0, out.1, n_embd);
                cursor = PassCursor::new(&mut encoder);

                ffn_normed.clear();
                ffn_normed.extend_from_slice(&host_x);
                crate::engine::tensor::rmsnorm_inplace(
                    &mut ffn_normed,
                    &layer.ffn_norm,
                    1,
                    n_embd,
                    cfg.rms_eps,
                );
                let routed = super::swiglu_moe_ffn(
                    self.backend.as_ref(),
                    &self.routing,
                    &ffn_normed,
                    1,
                    n_embd,
                    &super::SwigluMoe {
                        gate_inp: &moe.gate_inp,
                        exp_probs_b: None,
                        gate_exps: &moe.gate_exps,
                        up_exps: &moe.up_exps,
                        down_exps: &moe.down_exps,
                        shared: None,
                        clamp_exp: super::SwigluLimit::None,
                        clamp_shexp: super::SwigluLimit::None,
                        scales: super::ExpertScales::NONE,
                    },
                    il,
                );
                crate::engine::tensor::add_inplace(&mut host_x, &routed);
                continue;
            }
            bufs.push(out);
        }
        // The loop is done submitting; what follows is the tail's own
        // recording, under a guard of its own so the two cannot be
        // confused again.
        drop(recording.take());
        let _tail_recording =
            crate::engine::decode_stages::enter(crate::engine::decode_stages::Stage::DeviceRecord);
        if trace {
            let loop_ms = t_tail.elapsed().as_secs_f64() * 1000.0;
            eprintln!(
                "orangu-server: [prefill-trace]   decode-record pre={pre_ms:.2}ms \
                 encoder={enc_ms:.2}ms layer0={first_ms:.2}ms rest={rest_ms:.2}ms \
                 loop={loop_ms:.2}ms over {} layers",
                layers.len().saturating_sub(1)
            );
        }
        let t_after = std::time::Instant::now();
        let after_trace = |what: &str| {
            if trace {
                eprintln!(
                    "orangu-server: [prefill-trace]   decode-record tail[{what}]={:.2}ms",
                    t_after.elapsed().as_secs_f64() * 1000.0
                );
            }
        };
        drop(cursor);

        // On the NPU path the layer loop ends with the hidden state on the
        // host, not in `bufs` — every layer was finished there. Hand it to
        // the tail as a CPU input; `record_output_norm` takes either.
        if npu.is_some() || moe_seam {
            // `Tail::None` wants this run's *pre-norm* hidden state, which on
            // this path is on the host, so it goes back in a buffer of its
            // own. Declining is not an option here: every layer has already
            // committed this position to its cache, and a caller that reran
            // the layers would commit it a second time.
            if tail == Tail::None {
                let buf = vulkan.prefill_rows_buffer(n_embd);
                vulkan.upload_rows(&buf, &host_x);
                ts.finish(vulkan, &mut encoder, n_layer);
                return Some((encoder, buf, 0));
            }
            let normed = vulkan.record_output_norm(
                &mut encoder,
                GpuInput::Cpu(&host_x),
                &self.output_norm,
                cfg.rms_eps,
                n_embd,
            );
            if tail == Tail::Host {
                ts.finish(vulkan, &mut encoder, n_layer);
                after_trace("host");
                return Some((encoder, normed, 0));
            }
            let (logits_buf, logits_offset) = vulkan.record_full_matmul(
                &mut encoder,
                GpuInput::Gpu(&normed, 0),
                &self.output_weight,
                slot_id + 1,
            );
            ts.finish(vulkan, &mut encoder, n_layer);
            after_trace("device");
            return Some((encoder, logits_buf, logits_offset));
        }

        let (last_buf, last_offset) = bufs.last()?;
        if tail == Tail::None {
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
        if tail == Tail::Host {
            ts.finish(vulkan, &mut encoder, n_layer);
            return Some((encoder, normed, 0));
        }
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
        // Both routes below fuse the residual adds — see
        // `record_decode_chain`.
        if !self.mul.is_identity() {
            return None;
        }
        let Some(vulkan) = self.backend.as_wgpu() else {
            // No single device holds the whole model: either there is no GPU
            // at all, or the model is split. `Self::record_split_decode`
            // answers the second case and `None` the first.
            return self.record_split_decode(cache, tokens, start_pos, slot_id);
        };
        // A decode step after a prompt whose last chunk parked its rows
        // (a prompt served in one no-logits chunk, say) fills them first.
        vulkan.fill_deferred_kv_rows(cache);
        let host_tail = self.tail_on_host();
        let tail = if host_tail { Tail::Host } else { Tail::Device };
        // **The two halves of a chained token.** This path is one
        // submission, so the finer stages an unfused architecture reports
        // have nothing to divide: every layer's work is inside a command
        // buffer that has not run yet. What *is* divisible, and what
        // decides whether a token is short of cores or of card, is the
        // host building that buffer against the host waiting for it —
        // which is the split `Stage::DeviceRecord` and
        // `Stage::DeviceSubmit` exist for. Without them this family
        // reported its whole token under `other`.
        let (encoder, buf, offset) =
            self.record_decode_chain(vulkan, cache, tokens, start_pos, slot_id, tail)?;
        let mut encoder = encoder;
        vulkan.finish_op_step(&mut encoder);
        // `device.submit` and `head` are **siblings, not nested**: the
        // guard closes before the host projection runs. A stage inside
        // another is counted twice and the breakdown stops summing to the
        // pass, which shows up as `other` pinned at exactly 0.00.
        let logits = if host_tail {
            // `[n_embd]` back instead of `[n_vocab]` — on a 128k
            // vocabulary that is half a megabyte of readback this no
            // longer does.
            let normed = {
                let _submit = crate::engine::decode_stages::enter(
                    crate::engine::decode_stages::Stage::DeviceSubmit,
                );
                vulkan.submit_and_read_at(encoder, &buf, offset, self.config.n_embd)
            };
            crate::engine::decode_stages::scope(crate::engine::decode_stages::Stage::Head, || {
                self.host_tail(&normed)
            })
        } else {
            let _submit = crate::engine::decode_stages::enter(
                crate::engine::decode_stages::Stage::DeviceSubmit,
            );
            vulkan.submit_and_readback_for(encoder, &self.output_weight, slot_id + 1)
        };
        if vulkan.gpu_timestamps() {
            vulkan.report_timestamps(start_pos, self.layers.len());
        }
        vulkan.report_op_step(start_pos);
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
            // `Tail::Device` rather than `Tail::Host` even where the host
            // would be faster: the rule is measured, but a split is the one
            // shape this project has no machine to measure it on, and an
            // untested path is not worth the megabyte it would save.
            let with_tail = last && *device == tail_device;
            let (encoder, buf, offset) = self.record_decode_run(
                vulkan,
                cache,
                layers.clone(),
                &x,
                None,
                start_pos,
                slot_id,
                if with_tail { Tail::Device } else { Tail::None },
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
    /// The tokens' embedding rows, scaled by the family's multiplier
    /// (Granite's; 1 elsewhere) — dequantized a token per task.
    fn scaled_token_embeddings(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        use rayon::prelude::*;
        let cfg = &self.config;
        let n_embd = cfg.n_embd;
        if let Some(&tok) = tokens.iter().find(|&&t| t as usize >= cfg.n_vocab) {
            anyhow::bail!("token id {tok} is out of vocab range");
        }
        let scale = self.mul.embedding;
        let mut x = vec![0f32; tokens.len() * n_embd];
        let embed = |(dst, &tok): (&mut [f32], &u32)| {
            let row = self.tok_embeddings.row(tok as usize);
            for (d, r) in dst.iter_mut().zip(row.iter()) {
                *d = r * scale;
            }
        };
        // A decode batch is a few rows: waking the pool for them costs more
        // than the rows, and leaves its workers spinning after.
        if tokens.len() <= 8 {
            x.chunks_mut(n_embd).zip(tokens.iter()).for_each(embed);
        } else {
            x.par_chunks_mut(n_embd)
                .zip(tokens.par_iter())
                .for_each(embed);
        }
        Ok(x)
    }

    /// Whether any layer of this file routes its feed-forward — what
    /// separates a mixture from the dense models that share this block.
    fn is_moe(&self) -> bool {
        self.layers.iter().any(|l| l.dense().is_none())
    }

    /// Whether a prefill chunk of `n_tokens` runs on the device-resident
    /// stream (`run_layers_resident`): every layer's chains on one card,
    /// nothing of the layer on the host, and no diagnostic that wants the
    /// host's copy.
    fn resident_prefill_serves(&self, n_tokens: usize) -> bool {
        // Any width above one: the step-by-step path's advantage at
        // narrow widths was over the per-layer chain that read back at
        // every layer, and the stream reads back nothing. (The sizer's
        // 16-token probe chunk took the step path otherwise, and pushed
        // its rows ahead of the previous chunk's parked ones.)
        n_tokens > 1
            && !no_fused_qkv()
            && !no_fused_post_attention()
            && !resident_prefill_off()
            // The stream records a dense feed-forward per layer; a mixture
            // chooses its experts per token, which nothing in the chain can
            // express. See `LlamaLayer::dense`.
            && self.layers.iter().all(|layer| layer.dense().is_some())
            && self.mul.residual == 1.0
            && crate::engine::dump_ffn_dir().is_none()
            && orangu::npu_ffn::service().is_none()
            && self
                .backend
                .as_wgpu()
                .is_some_and(|v| v.prefill_fused_attention_enabled())
            && self.layers.iter().all(|l| l.wo.device() == 0)
    }

    /// The device-resident prefill stream for this family — see
    /// [`super::run_layers_resident`]. `None` hands the chunk to
    /// [`Self::run_layers`].
    fn run_layers_resident(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        want_x: bool,
    ) -> Result<Option<Vec<f32>>> {
        let cfg = &self.config;
        if !self.resident_prefill_serves(tokens.len()) {
            return Ok(None);
        }
        let Some(vulkan) = self.backend.as_wgpu() else {
            return Ok(None);
        };
        let x = self.scaled_token_embeddings(tokens)?;
        super::run_layers_resident(
            vulkan,
            cache,
            tokens,
            &x,
            start_pos,
            want_x,
            cfg.n_head,
            cfg.n_head_kv,
            self.head_dim(),
            cfg.rms_eps,
            0..self.layers.len(),
            |il| self.resident_layer(il),
        )
    }

    /// Layer `il` as the device-resident stream records it. Only for a model
    /// [`Self::resident_prefill_serves`] admitted, whose layers are all dense.
    fn resident_layer(&self, il: usize) -> super::ResidentLayer<'_> {
        let cfg = &self.config;
        let layer = &self.layers[il];
        let dense = layer
            .dense()
            .expect("resident_prefill_serves admits dense models only");
        super::ResidentLayer {
            attn_norm: &layer.attn_norm,
            wq: &layer.wq,
            wk: &layer.wk,
            wv: &layer.wv,
            q_bias: layer.q_bias.as_deref(),
            k_bias: layer.k_bias.as_deref(),
            v_bias: layer.v_bias.as_deref(),
            q_norm: layer.q_norm.as_deref(),
            k_norm: layer.k_norm.as_deref(),
            pairing: self.rope.layout,
            yarn: crate::engine::backend::vulkan::RopeYarn::from_params(&self.rope),
            rope_dim: cfg.rope_dim,
            rope_freq_base: cfg.rope_freq_base,
            freq_factors: self.rope_freq_factors.as_deref(),
            n_swa: 0,
            scale: self.attn_scale(),
            wo: &layer.wo,
            ffn_norm: &layer.ffn_norm,
            ffn_gate: dense.0,
            ffn_up: dense.1,
            ffn_down: dense.2,
            activation: crate::engine::backend::vulkan::FfnActivation::Swiglu,
            attn_gate: None,
            attn_post_norm: None,
            ffn_post_norm: None,
            post_norm_eps: None,
        }
    }

    /// [`Self::run_layers_resident`] over `layers` only, from `x`, the
    /// residual entering the first of them: the whole model's stream, or
    /// one node's range in a tree of workers (`forward_layers`).
    fn run_range_resident(
        &self,
        cache: &mut KvCache,
        x: &[f32],
        tokens: &[u32],
        start_pos: usize,
        layers: std::ops::Range<usize>,
        want_x: bool,
    ) -> Result<Option<Vec<f32>>> {
        let cfg = &self.config;
        let Some(vulkan) = self.backend.as_wgpu() else {
            return Ok(None);
        };
        super::run_layers_resident(
            vulkan,
            cache,
            tokens,
            x,
            start_pos,
            want_x,
            cfg.n_head,
            cfg.n_head_kv,
            self.head_dim(),
            cfg.rms_eps,
            layers,
            |il| self.resident_layer(il),
        )
    }

    /// One row of the last layer's residual stream through the final norm
    /// and the output projection — normalized in place — as next-token
    /// logits. One row whether this was a decode step or a thousand-token
    /// prefill, so the same host-or-device rule applies here as to the
    /// fused path's tail.
    /// `ModelForward::head` as one projection of every row, so the output
    /// matrix is read once for the pass rather than once a row; a device
    /// model whose tail runs on the host keeps the row-by-row form.
    /// [`ModelForward::forward_decode_batch_sampling`]'s body, and with no
    /// `samples` [`ModelForward::forward_decode_batch`]'s. The batch runs on
    /// the device-resident stream, and with `samples` the head and each
    /// row's sampler run after it on the device
    /// (`VulkanBackend::decode_batch_tail`); otherwise the residual comes
    /// home for the head.
    ///
    /// `widths` as `run_layers_resident_batch` takes them (empty: one token
    /// a row), `tokens` every row's tokens in order, and `samples` one per
    /// token; the outcomes are one per token too.
    fn decode_batch(
        &self,
        rows: &mut [super::DecodeRow<'_>],
        widths: &[usize],
        tokens: &[u32],
        samples: Option<&[Option<super::GreedySampleParams<'_>>]>,
    ) -> Result<Option<Vec<super::ForwardOutcome>>> {
        // The batch runs on the device-resident stream, so it needs what that
        // stream needs, and a device head: the host tail is one row's.
        if rows.len() < 2 || !self.resident_prefill_serves(tokens.len()) || self.tail_on_host() {
            return Ok(None);
        }
        let Some(vulkan) = self.backend.as_wgpu() else {
            return Ok(None);
        };
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
        let cfg = &self.config;
        let (n_embd, n_vocab) = (cfg.n_embd, cfg.n_vocab);
        let x = self.scaled_token_embeddings(tokens)?;
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
        // `batch_slot` is the request's slot, as `record_decode_chain` uses.
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
            0..self.layers.len(),
            |il| self.resident_layer(il),
        )?
        else {
            return Ok(None);
        };
        let pos = rows.first().map_or(0, |r| r.pos);
        // Granite divides its logits, which the device samplers do not.
        if let Some(samples) = samples.filter(|_| vulkan.gpu_sample() && self.mul.logit == 1.0)
            && let Some(outcomes) = vulkan.decode_batch_tail(
                crate::engine::backend::vulkan::AttnOutSrc::Gpu(&h_dev, 0, n_rows),
                n_rows,
                &self.output_norm,
                cfg.rms_eps,
                &self.output_weight,
                None,
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
        anyhow::ensure!(
            flat.len() == n_rows * n_vocab,
            "output projection produced {} logits, expected {}",
            flat.len(),
            n_rows * n_vocab
        );
        Ok(Some(
            flat.chunks_exact(n_vocab)
                .map(|row| {
                    // Granite divides; everything else has `logit == 1.0`.
                    super::ForwardOutcome::Logits(if self.mul.logit == 1.0 {
                        row.to_vec()
                    } else {
                        row.iter().map(|v| v / self.mul.logit).collect()
                    })
                })
                .collect(),
        ))
    }

    fn head_rows(&self, mut hidden: Vec<f32>, n_rows: usize) -> Result<Vec<Vec<f32>>> {
        let cfg = &self.config;
        if n_rows == 1 || (self.backend.as_wgpu().is_some() && self.tail_on_host()) {
            return self.head(&hidden, n_rows);
        }
        anyhow::ensure!(
            hidden.len() == n_rows * cfg.n_embd,
            "hidden state holds {} values, expected {n_rows} rows of {}",
            hidden.len(),
            cfg.n_embd
        );
        tensor::rmsnorm_inplace(
            &mut hidden,
            &self.output_norm,
            n_rows,
            cfg.n_embd,
            cfg.rms_eps,
        );
        let flat = self.backend.matmul(&hidden, n_rows, &self.output_weight);
        let n_vocab = flat.len() / n_rows;
        Ok(flat
            .chunks_exact(n_vocab)
            .map(|row| {
                let mut row = row.to_vec();
                // Granite divides; everything else has `logit == 1.0`.
                if self.mul.logit != 1.0 {
                    for v in row.iter_mut() {
                        *v /= self.mul.logit;
                    }
                }
                row
            })
            .collect())
    }

    fn row_logits(&self, row: &mut [f32]) -> Vec<f32> {
        let cfg = &self.config;
        tensor::rmsnorm_inplace(row, &self.output_norm, 1, cfg.n_embd, cfg.rms_eps);
        let mut logits = match self.backend.as_wgpu() {
            Some(_) if self.tail_on_host() => self.host_tail(row),
            _ => self.backend.matmul(row, 1, &self.output_weight),
        };
        // Granite divides; everything else has `logit == 1.0`.
        if self.mul.logit != 1.0 {
            for v in logits.iter_mut() {
                *v /= self.mul.logit;
            }
        }
        logits
    }

    fn run_layers(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<Vec<f32>> {
        // Embedding lookup: x[t, :] = tok_embeddings[token[t], :].
        let x = self.scaled_token_embeddings(tokens)?;
        self.run_layer_range(cache, x, tokens.len(), start_pos, 0..self.layers.len())
    }

    /// Layers `layers` of the host step path over `x`, the residual stream
    /// of `n_tokens` rows entering the first of them, returning it as it
    /// leaves the last. [`Self::run_layers`] is this over every layer after
    /// the embedding; `ModelForward::forward_layers` is this over the range
    /// one node of a worker tree owns (see `crate::workers`).
    fn run_layer_range(
        &self,
        cache: &mut KvCache,
        mut x: Vec<f32>,
        n_tokens: usize,
        start_pos: usize,
        layers: std::ops::Range<usize>,
    ) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let n_embd = cfg.n_embd;
        anyhow::ensure!(
            layers.end <= self.layers.len() && layers.start <= layers.end,
            "layer range {layers:?} is outside this model's {} layers",
            self.layers.len()
        );
        anyhow::ensure!(
            x.len() == n_tokens * n_embd,
            "hidden state holds {} values, expected {n_tokens} rows of {n_embd}",
            x.len()
        );
        let head_dim = self.head_dim();
        let n_head = cfg.n_head;
        let n_head_kv = cfg.n_head_kv;
        let kv_dim = n_head_kv * head_dim;

        // The per-dispatch device table over this whole pass — see
        // `super::OpSpan`. Nothing unless the timer is on; it is what makes
        // `perf/llamacpp/ops.sh` able to see a mixture at all, since a
        // mixture never takes the chains that resolve their own stamps.
        let _ops = super::OpSpan::open(self.backend.as_ref(), start_pos);
        // Rows a resident chunk parked for the chunk after it are brought
        // home before anything here pushes past them.
        if let Some(vulkan) = self.backend.as_wgpu() {
            vulkan.fill_deferred_kv_rows(cache);
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
        // Only ever filled under `npu_block_check`.
        let mut ffn_device: Vec<f32> = Vec::new();
        let mut ffn_scratch = super::FfnScratch::default();

        for layer_idx in layers {
            let layer = &self.layers[layer_idx];
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
            // What `Q`/`K`/`V` read, captured for the device to calibrate
            // against. Formed here whatever the fused chain decides below,
            // so this sees every layer of every chunk.
            crate::engine::dump_attn_input(layer_idx, n_tokens, &normed);

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
            // **The device takes `Q`/`K`/`V` when it has them**, which means
            // declining this fusion for the layer: the chain computes the
            // projections itself, so there is no way to have both. Measured
            // before it was wired — llama 3.2 1B `Q8_0`, prefill of 1024
            // tokens with the feed-forward already on the device — giving up
            // the fusion costs 77.75 ± 0.82 against 77.09 ± 0.96 tok/s,
            // which is nothing: with the network gone the GPU's remaining
            // work is small and a 128-token chunk amortises the extra
            // submissions.
            let attn_on_npu = crate::npu_tool::attention_enabled()
                && orangu::npu_ffn::service().is_some_and(|npu| npu.has_attn(layer_idx, n_tokens));
            let fused_qkv = self
                .backend
                // This layer's card — see `Backend::as_wgpu_on`.
                .as_wgpu_on(layer.wo.device())
                .filter(|_| fusable && !no_fused_post_attention() && !attn_on_npu)
                .and_then(|vulkan| {
                    vulkan.fused_attention_prefill(
                        crate::engine::backend::vulkan::FusedAttnPrefillInput {
                            x_gpu: None,
                            attn_norm: None,
                            q_bias: layer.q_bias.as_deref(),
                            pairing: self.rope.layout,
                            yarn: crate::engine::backend::vulkan::RopeYarn::from_params(&self.rope),
                            normalize_v: false,
                            attn_gate: None,
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
                            scale: self.attn_scale(),
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
                // The device first, when it holds this layer's projections:
                // one call for all three, answered as `q|k|v` back to back.
                // Falls through to the backend on any refusal, with the same
                // result and only slower.
                // Timed: at decode these three are host matmuls on any
                // model whose projections sit under the host-matmul
                // crossover, and they are most of what the host does
                // before it waits for attention.
                let mut qkv = crate::engine::decode_stages::scope(
                    crate::engine::decode_stages::Stage::AttnProject,
                    || {
                        attn_on_npu
                            .then(|| {
                                let mut packed = Vec::new();
                                let served = orangu::npu_ffn::service().is_some_and(|npu| {
                                    npu.forward_attn_into(layer_idx, n_tokens, &normed, &mut packed)
                                });
                                let q_len = n_tokens * cfg.n_head * head_dim;
                                let kv_len = n_tokens * cfg.n_head_kv * head_dim;
                                (served && packed.len() == q_len + 2 * kv_len).then(|| {
                                    let v = packed[q_len + kv_len..].to_vec();
                                    let k = packed[q_len..q_len + kv_len].to_vec();
                                    packed.truncate(q_len);
                                    vec![packed, k, v]
                                })
                            })
                            .flatten()
                            .unwrap_or_else(|| {
                                // Independent given the same normed input — one
                                // batched dispatch instead of three sequential round
                                // trips (matters most for a GPU backend; see
                                // `Backend::matmul_batch`).
                                self.backend.matmul_batch(&[
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
                                ])
                            })
                    },
                );
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
                    scale: self.attn_scale(),
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
            // a round trip on this stack was measured at ~260 µs. The
            // fused chain is cross-checked against exactly the sequence in the
            // `else` branch below
            // (`fused_post_attention_prefill_matches_the_unfused_sequence_swiglu_*`).
            //
            // **Declined when the NPU has this layer at this width.** The
            // fused chain computes the FFN itself, so taking it is what kept
            // this whole family off the device: gemma reached the NPU only
            // because its own chain is declined the same way. Routing to the
            // `else` arm is how the device gets asked at all; if it then
            // declines or fails, that arm computes the block as before.
            let ffn_on_npu =
                orangu::npu_ffn::service().is_some_and(|npu| npu.has(layer_idx, n_tokens));
            let fused = self
                .backend
                .as_wgpu_on(layer.wo.device())
                // `mul.residual` scales each branch before its add and this
                // chain does both adds internally, so Granite declines it.
                .filter(|_| !no_fused_post_attention() && self.mul.residual == 1.0)
                .filter(|_| !ffn_on_npu)
                // A mixture layer's feed-forward is not a network this
                // chain can record — see `LlamaLayer::dense`. Zipped rather
                // than filtered so the projections the chain needs come
                // from the same answer that admitted it.
                .zip(layer.dense())
                // The chain never forms `normed2` on the host, so a run that
                // is capturing activations takes the slow arm instead.
                .filter(|_| crate::engine::dump_ffn_dir().is_none())
                .and_then(|(vulkan, dense_ffn)| {
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
                        dense_ffn.0,
                        dense_ffn.1,
                        dense_ffn.2,
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
                //
                // Timed as `attn.out`, and it is not only arithmetic: the
                // readback below is where a caller first *waits* for the
                // attention kernel it dispatched. On the device path the
                // dispatch is recorded under `attn` and costs almost
                // nothing; what the context length actually buys is paid
                // here, and attributing it anywhere else leaves a stage
                // that grows with depth for no visible reason.
                // **Beside `attn.out`, not inside it.** This is where the
                // layer waits for the attention it dispatched, and it is the
                // only part of the block that recording `wo` into
                // attention's own submission could remove — so it earns a
                // line of its own. Nested it would be counted twice: every
                // stage but the pass itself is a disjoint sibling, and the
                // first version of this left `other` clamped at zero and the
                // reported stages summing past the pass.
                crate::engine::decode_stages::scope(
                    crate::engine::decode_stages::Stage::AttnRead,
                    || {
                        if let (Some(buf), Some(vulkan)) =
                            (&attn_on_device, self.backend.as_wgpu_on(layer.wo.device()))
                        {
                            attn_out = vulkan.read_buffer_f32(buf, n_tokens * n_head * head_dim);
                        }
                    },
                );
                crate::engine::decode_stages::scope(
                    crate::engine::decode_stages::Stage::AttnOut,
                    || {
                        self.backend
                            .matmul_into(&mut attn_proj, &attn_out, n_tokens, &layer.wo);
                        self.scale_residual(&mut attn_proj);
                        tensor::add_inplace(&mut x, &attn_proj);

                        tensor::rmsnorm_into(
                            &mut normed2,
                            &x,
                            &layer.ffn_norm,
                            n_tokens,
                            n_embd,
                            cfg.rms_eps,
                        );
                    },
                );
                // The NPU, when this model's block was compiled for exactly
                // this width. It runs the three projections and applies
                // SwiGLU on the host — see `orangu::npu_ffn`. Anything it
                // does not have, or any failure, falls through to the
                // backend below with the same result, only slower.
                //
                // Both widths reach the device now — the decode step has
                // its own graph and its own seam, see
                // `npu_tool::decode_enabled` — so this hook fires at 16 and
                // `record_decode_run` at 1.
                crate::engine::dump_ffn_input(layer_idx, n_tokens, &normed2);
                // Both ways under `npu_block_check`, keeping the backend's
                // answer, so prefill is measured on the same footing as the
                // decode seam.
                let checked = npu_block_check()
                    && orangu::npu_ffn::service().is_some_and(|npu| {
                        npu.forward_into(layer_idx, n_tokens, &normed2, &mut ffn_device)
                    });
                let from_npu = !checked
                    && orangu::npu_ffn::service().is_some_and(|npu| {
                        npu.forward_into(layer_idx, n_tokens, &normed2, &mut ffn_out)
                    });
                // Shared with the dense FFN of the Qwen 3.5 hybrid trunk —
                // `LLM_FFN_SILU`/`LLM_FFN_PAR` is one computation and this
                // family (Llama, Mistral, Qwen2, Qwen3) and that one run the
                // same one.
                if !from_npu {
                    match &layer.ffn {
                        Ffn::Dense {
                            w_gate,
                            w_up,
                            w_down,
                        } => super::swiglu_ffn_into(
                            self.backend.as_ref(),
                            &mut ffn_out,
                            &mut ffn_scratch,
                            &normed2,
                            n_tokens,
                            w_gate,
                            w_up,
                            w_down,
                        ),
                        // The routed feed-forward, shared with every other
                        // mixture here: one router matmul for the batch,
                        // then each token's chosen experts. No shared
                        // expert and no selection bias — Qwen3-MoE has
                        // neither.
                        Ffn::Moe(moe) => {
                            ffn_out = super::swiglu_moe_ffn(
                                self.backend.as_ref(),
                                &self.routing,
                                &normed2,
                                n_tokens,
                                cfg.n_embd,
                                &super::SwigluMoe {
                                    gate_inp: &moe.gate_inp,
                                    exp_probs_b: None,
                                    gate_exps: &moe.gate_exps,
                                    up_exps: &moe.up_exps,
                                    down_exps: &moe.down_exps,
                                    shared: None,
                                    clamp_exp: super::SwigluLimit::None,
                                    clamp_shexp: super::SwigluLimit::None,
                                    scales: super::ExpertScales::NONE,
                                },
                                layer_idx,
                            );
                        }
                    }
                    if checked {
                        report_npu_block_error(
                            layer_idx,
                            n_tokens,
                            &normed2,
                            &ffn_device,
                            &ffn_out,
                        );
                    }
                }
                self.scale_residual(&mut ffn_out);
                tensor::add_inplace(&mut x, &ffn_out);
            }
        }

        Ok(x)
    }
}

impl ModelForward for LlamaModel {
    fn reads_rows_only_through_attention(&self) -> bool {
        true
    }

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
        // Timed, when it is a decode step, so that
        // `backend::tail_prefers_host` can compare its two arms on whole
        // steps — the only comparison that charges each of them for
        // exactly what it costs. A prefill runs through here too and is
        // not a step of this kind.
        let at = (tokens.len() == 1).then(std::time::Instant::now);
        // A mixture's decode step alternates between the card and the
        // host's expert turns, and the card drops to its parked clock on
        // every one of them; it is held up across the step. A dense
        // model's step never leaves the card and arms nothing.
        let _clock = self
            .is_moe()
            .then(|| super::hold_clock_for_step(self.backend.as_ref(), tokens.len()))
            .flatten();
        let outcome = (|| -> Result<super::ForwardOutcome> {
            if tokens.len() == 1
                && self.mul.is_identity()
                && let Some(params) = &greedy_sample
                // A `top_k > 0` asks for candidates, which only the device top-k
                // path answers; this architecture has the greedy path alone.
                && params.top_k == 0
                && let Some(vulkan) = self.backend.as_wgpu()
                && vulkan.gpu_sample()
                // Sampling on the device is only worth having if the logits are
                // *made* there. When the projection belongs on the host
                // (`tail_prefers_host`) this path would drag it back onto the
                // device to save a readback a tenth its cost, so it stands down
                // and `record_decode_forward` returns host logits for the
                // caller to sample — which is the same arithmetic in the same
                // order, on the processor that is faster at it.
                && !self.tail_on_host()
                && let Some((mut encoder, logits_buf, logits_offset)) =
                    self.record_decode_chain(vulkan, cache, tokens, start_pos, slot_id, Tail::Device)
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
                let mut encoder = encoder;
                vulkan.finish_op_step(&mut encoder);
                let next = {
                    let _submit = crate::engine::decode_stages::enter(
                        crate::engine::decode_stages::Stage::DeviceSubmit,
                    );
                    vulkan.submit_and_readback_u32(encoder, &sample_buf)
                };
                if vulkan.gpu_timestamps() {
                    vulkan.report_timestamps(start_pos, self.layers.len());
                }
                vulkan.report_op_step(start_pos);
                return Ok(super::ForwardOutcome::Token(next));
            }
            self.forward(cache, tokens, start_pos, slot_id)
                .map(super::ForwardOutcome::Logits)
        })();
        if let Some(at) = at {
            crate::engine::backend::note_tail_step(at.elapsed());
        }
        outcome
    }

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
        let kv_dim = self.config.n_head_kv * self.head_dim();
        KvCache::new(self.config.n_layer, capacity, kv_dim)
    }

    fn batches_decode(&self) -> bool {
        self.backend.as_wgpu().is_some() && !self.tail_on_host() && self.resident_prefill_serves(2)
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
        super::batch_verify() && self.backend.as_wgpu().is_some() && !self.tail_on_host()
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

        let x = match self.run_layers_resident(cache, tokens, start_pos, true)? {
            Some(x) if super::resident_prefill_check() => {
                // The same chunk again on the step path, into a copy of the
                // cache, and the two residuals compared.
                let mut again = cache.duplicate();
                again.truncate(start_pos)?;
                let step = self.run_layers(&mut again, tokens, start_pos)?;
                super::report_resident_check(&x, &step, n_embd, start_pos);
                x
            }
            Some(x) => x,
            None => self.run_layers(cache, tokens, start_pos)?,
        };

        // Only the last token's hidden state is needed for next-token
        // logits — a batched prefill doesn't need every position's output.
        let mut last = x[(n_tokens - 1) * n_embd..].to_vec();
        Ok(self.row_logits(&mut last))
    }

    fn forward_no_logits(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        slot_id: usize,
    ) -> Result<()> {
        // A chunk before the last: on the resident stream its residual
        // stays on the device and its K/V readback is parked for the next
        // chunk to complete.
        if self
            .run_layers_resident(cache, tokens, start_pos, false)?
            .is_some()
        {
            return Ok(());
        }
        self.forward(cache, tokens, start_pos, slot_id).map(|_| ())
    }

    /// Every position's logits for a multi-token forward — what a
    /// speculative pair's verification pass needs (a draft model's, a
    /// prompt lookup's), and what lets a Llama-family model be either half
    /// of the pair. The host step path over every layer, then one output
    /// projection per row. A cache holding rows only the device wrote (a
    /// fused decode step before this) is read where those rows are: a
    /// layer's attention runs on its device while it has such rows
    /// (`engine::attention`).
    fn forward_all_logits(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        _slot_id: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let x = self.run_layers(cache, tokens, start_pos)?;
        self.head(&x, tokens.len())
    }

    /// On the device-resident prefill stream with every row's residual
    /// kept, then the head over all of them in one projection: a draft of
    /// `n` tokens costs one pass that reads the weights once, where the step
    /// path above reads them per projection call. One token is a decode
    /// step. Anything the stream does not serve takes the step path.
    fn forward_verify(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        slot_id: usize,
    ) -> Result<Vec<Vec<f32>>> {
        if tokens.len() == 1 {
            return Ok(vec![self.forward(cache, tokens, start_pos, slot_id)?]);
        }
        match self.run_layers_resident(cache, tokens, start_pos, true)? {
            Some(x) => self.head_rows(x, tokens.len()),
            None => self.forward_all_logits(cache, tokens, start_pos, slot_id),
        }
    }

    fn decode_lookahead(
        &self,
        cache: &mut KvCache,
        input: super::LookaheadInput<'_>,
        start_pos: usize,
        slot_id: usize,
    ) -> Result<Option<crate::engine::backend::vulkan::PendingToken>> {
        let Some(vulkan) = self.backend.as_wgpu() else {
            return Ok(None);
        };
        // The pick has to be made where the logits are, and the chain has
        // to be the whole step on one device.
        if !self.mul.is_identity() || !vulkan.gpu_sample() || self.tail_on_host() || self.is_moe() {
            return Ok(None);
        }
        let n_embd = self.config.n_embd;
        let (x0, gathered) = match input {
            super::LookaheadInput::Token(token) => {
                if token as usize >= self.config.n_vocab {
                    return Ok(None);
                }
                (self.tok_embeddings.row(token as usize).to_vec(), None)
            }
            super::LookaheadInput::Pending(prev) => {
                let mut encoder = vulkan.new_encoder("orangu-server lookahead gather");
                let Some(row) =
                    vulkan.record_row_gather(&mut encoder, &prev.ids, &self.tok_embeddings, 1.0)
                else {
                    return Ok(None);
                };
                vulkan.submit_now(encoder);
                (vec![0.0; n_embd], Some(row))
            }
        };
        let Some((mut encoder, logits_buf, logits_offset)) = self.record_decode_run(
            vulkan,
            cache,
            0..self.layers.len(),
            &x0,
            gathered.as_ref(),
            start_pos,
            slot_id,
            Tail::Device,
        ) else {
            return Ok(None);
        };
        let sample_buf = vulkan.record_argmax_sample(
            &mut encoder,
            crate::engine::backend::vulkan::GpuArgmaxSampleInput {
                logits: crate::engine::backend::vulkan::GpuInput::Gpu(
                    &logits_buf,
                    (logits_offset / 4) as usize,
                ),
                n_vocab: self.output_weight.out_dim,
                recent_tokens: &[],
                repeat_penalty: 1.0,
                logit_softcap: None,
            },
            slot_id + 1,
        );
        Ok(Some(vulkan.submit_token(encoder, &sample_buf)))
    }

    fn forward_verify_sampled(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        slot_id: usize,
        samples: &[super::GreedySampleParams<'_>],
    ) -> Result<Option<Vec<super::ForwardOutcome>>> {
        let Some(vulkan) = self.backend.as_wgpu() else {
            return Ok(None);
        };
        // Granite divides its logits, which the device samplers do not, and a
        // head on the host has no device rows to sample.
        if tokens.len() < 2
            || samples.len() != tokens.len()
            || !vulkan.gpu_sample()
            || self.mul.logit != 1.0
            || self.tail_on_host()
        {
            return Ok(None);
        }
        let Some(x) = self.run_layers_resident(cache, tokens, start_pos, true)? else {
            return Ok(None);
        };
        // Each row its own sampler scratch: the verify's rows are one
        // request's, and share its slot everywhere else.
        let slots: Vec<usize> = (0..tokens.len())
            .map(|r| super::VERIFY_SAMPLE_SLOTS + slot_id * super::MAX_VERIFY_ROWS + r)
            .collect();
        let rows: Vec<Option<super::GreedySampleParams<'_>>> =
            samples.iter().cloned().map(Some).collect();
        if let Some(outcomes) = vulkan.decode_batch_tail(
            crate::engine::backend::vulkan::AttnOutSrc::Host(&x),
            tokens.len(),
            &self.output_norm,
            self.config.rms_eps,
            &self.output_weight,
            None,
            &rows,
            &slots,
        ) {
            return Ok(Some(outcomes));
        }
        // The layers ran and the rows are in the cache; the head runs on the
        // host's copy of the residual instead.
        Ok(Some(
            self.head_rows(x, tokens.len())?
                .into_iter()
                .map(super::ForwardOutcome::Logits)
                .collect(),
        ))
    }

    fn supports_layer_split(&self) -> bool {
        true
    }

    fn new_kv_cache_for_layers(&self, layers: std::ops::Range<usize>, capacity: usize) -> KvCache {
        let kv_dim = self.config.n_head_kv * self.head_dim();
        let dims: Vec<usize> = (0..self.config.n_layer)
            .map(|il| if layers.contains(&il) { kv_dim } else { 0 })
            .collect();
        KvCache::new_with_dims(capacity, &dims)
    }

    fn embed(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        self.scaled_token_embeddings(tokens)
    }

    /// A decode step on a Vulkan device records the range as one chain
    /// (`record_decode_run`, the same the multi-device split decode uses)
    /// and reads the residual back. A prompt chunk runs the range on the
    /// device-resident prefill stream (`run_range_resident`), the residual
    /// uploaded once and read back once, as the whole model's does.
    /// Everything else — the CPU, a mixture, a chain or a stream that
    /// declines — takes the host step path (`run_layer_range`).
    ///
    /// The chain's per-sequence GPU resources are cached by slot, and a
    /// worker runs several sequences' forwards at once without knowing
    /// their slots: each chain holds a range slot of its own
    /// ([`super::range_turn`]) while it is recorded and run, so sequences
    /// record side by side; prompt chunks take turns
    /// ([`super::RANGE_PREFILL`]).
    fn forward_layers(
        &self,
        cache: &mut KvCache,
        hidden: Vec<f32>,
        tokens: &[u32],
        layers: std::ops::Range<usize>,
        start_pos: usize,
    ) -> Result<Vec<f32>> {
        if tokens.len() == 1
            && !layers.is_empty()
            && super::range_chains()
            && let Some(vulkan) = self.backend.as_wgpu()
        {
            let turn = super::range_turn(cache);
            if let Some((encoder, buf, offset)) = self.record_decode_run(
                vulkan,
                cache,
                layers.clone(),
                &hidden,
                None,
                start_pos,
                turn.slot,
                Tail::None,
            ) {
                return Ok(vulkan.submit_and_read_at(encoder, &buf, offset, self.config.n_embd));
            }
        }
        if tokens.len() > 1
            && !layers.is_empty()
            && super::range_chains()
            && self.resident_prefill_serves(tokens.len())
        {
            let _turn = super::RANGE_PREFILL
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            if let Some(x) =
                self.run_range_resident(cache, &hidden, tokens, start_pos, layers.clone(), true)?
            {
                return Ok(x);
            }
        }
        self.run_layer_range(cache, hidden, tokens.len(), start_pos, layers)
    }

    fn final_norm(&self, hidden: &mut [f32], n_rows: usize) -> Result<()> {
        tensor::rmsnorm_inplace(
            hidden,
            &self.output_norm,
            n_rows,
            self.config.n_embd,
            self.config.rms_eps,
        );
        Ok(())
    }

    fn head(&self, hidden: &[f32], n_rows: usize) -> Result<Vec<Vec<f32>>> {
        let n_embd = self.config.n_embd;
        anyhow::ensure!(
            hidden.len() == n_rows * n_embd,
            "hidden state holds {} values, expected {n_rows} rows of {n_embd}",
            hidden.len()
        );
        Ok(hidden
            .chunks_exact(n_embd)
            .map(|row| self.row_logits(&mut row.to_vec()))
            .collect())
    }

    fn forward_hidden_states(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let mut cache = self.new_kv_cache(tokens.len().max(1));
        self.forward_hidden_states_at(&mut cache, tokens, 0)
    }

    /// Attention here is causal at every layer, so a chunk's hidden states
    /// are the ones the whole input would give those positions.
    fn hidden_states_are_causal(&self) -> bool {
        true
    }

    fn forward_hidden_states_at(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<Vec<f32>> {
        let mut x = self.run_layers(cache, tokens, start_pos)?;
        tensor::rmsnorm_inplace(
            &mut x,
            &self.output_norm,
            tokens.len(),
            self.config.n_embd,
            self.config.rms_eps,
        );
        Ok(x)
    }

    fn forward_hidden_states_pre_norm(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let mut cache = self.new_kv_cache(tokens.len().max(1));
        self.run_layers(&mut cache, tokens, 0)
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

    /// The device-resident prefill stream against the step path on a model
    /// with per-head Q/K norms: the repository's qwen3 smoke model, one
    /// chunk, the two residuals compared. They round their activations at
    /// different places, so the bound is a tolerance; a norm left out puts
    /// whole rows off.
    /// `cargo test --release --bin orangu-server resident_stream_applies -- --ignored`.
    #[test]
    #[ignore]
    fn the_resident_stream_applies_the_per_head_qk_norms() {
        let Some(vulkan) = crate::engine::backend::vulkan::VulkanBackend::try_init() else {
            println!("no Vulkan device; nothing to check");
            return;
        };
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("contrib/orangu-model/smoke/orangu-code-smoke-Q4_K_M.gguf");
        let loaded = LoadedModel::open(&path).expect("load model");
        let model = LlamaModel::load_with_backend(&loaded, Arc::new(vulkan)).expect("build model");
        assert!(
            model
                .layers
                .iter()
                .all(|l| l.q_norm.is_some() && l.k_norm.is_some())
        );
        let tokens: Vec<u32> = (0..96u32).map(|i| 100 + (i * 37) % 2000).collect();
        let n_embd = model.config.n_embd;
        let mut cache = model.new_kv_cache(tokens.len() + 1);
        let resident = model
            .run_layers_resident(&mut cache, &tokens, 0, true)
            .expect("resident pass")
            .expect("the stream serves this model");
        let mut cache = model.new_kv_cache(tokens.len() + 1);
        let step = model.run_layers(&mut cache, &tokens, 0).expect("step pass");
        let (mut diff2, mut val2) = (0f64, 0f64);
        for (x, y) in resident.iter().zip(&step) {
            diff2 += f64::from((x - y) * (x - y));
            val2 += f64::from(y * y);
        }
        let rms = (diff2 / val2).sqrt();
        let worst = resident
            .chunks_exact(n_embd)
            .zip(step.chunks_exact(n_embd))
            .map(|(a, b)| {
                let mag = b.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
                a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / mag
            })
            .fold(0f32, f32::max);
        println!("resident against step: rms {rms:.2e}, worst row {worst:.2e}");
        assert!(rms < 2e-2, "rms {rms}");
        assert!(worst < 0.1, "worst row {worst}");
    }

    /// Two sequences' decode steps taken together (`forward_decode_batch`)
    /// give each the logits of its own solo steps, to the rounding of the
    /// narrow-pass kernels the batch runs on, and the same next tokens —
    /// over a batched step, a batched step with the rows in the other order,
    /// and a solo step after them, which is how a cache meets a different
    /// row of a batch and goes back to stepping alone. On the GPU, so
    /// ignored by default:
    /// `cargo test --release --bin orangu-server decode_batch_matches -- --ignored`.
    #[test]
    #[ignore]
    fn a_decode_batch_matches_each_rows_solo_step() {
        let Some(vulkan) = crate::engine::backend::vulkan::VulkanBackend::try_init() else {
            println!("no Vulkan device; nothing to check");
            return;
        };
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("contrib/orangu-model/smoke/orangu-code-smoke-Q4_K_M.gguf");
        let loaded = LoadedModel::open(&path).expect("load model");
        let model = LlamaModel::load_with_backend(&loaded, Arc::new(vulkan)).expect("build model");
        let prompts: [Vec<u32>; 2] = [
            (0..40u32).map(|i| 100 + (i * 37) % 2000).collect(),
            (0..23u32).map(|i| 300 + (i * 53) % 1700).collect(),
        ];
        // Each sequence's next three tokens, fed the same way to both.
        let next: [[u32; 3]; 2] = [[911, 57, 1302], [1207, 640, 88]];
        let prefilled = || -> Vec<KvCache> {
            prompts
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let mut cache = model.new_kv_cache(64);
                    model.forward(&mut cache, p, 0, i).expect("prefill");
                    cache
                })
                .collect()
        };
        let mut solo_caches = prefilled();
        let solo: Vec<Vec<Vec<f32>>> = solo_caches
            .iter_mut()
            .enumerate()
            .map(|(i, cache)| {
                (0..3)
                    .map(|step| {
                        model
                            .forward(cache, &next[i][step..=step], prompts[i].len() + step, i)
                            .expect("solo step")
                    })
                    .collect()
            })
            .collect();
        let mut caches = prefilled();
        let mut got: Vec<Vec<Vec<f32>>> = vec![Vec::new(), Vec::new()];
        for (step, order) in [[0usize, 1], [1, 0]].iter().enumerate() {
            let (first, second) = caches.split_at_mut(1);
            let mut by_seq = [Some(&mut first[0]), Some(&mut second[0])];
            let mut rows: Vec<super::super::DecodeRow<'_>> = order
                .iter()
                .map(|&i| super::super::DecodeRow {
                    cache: by_seq[i].take().expect("each sequence once"),
                    pos: prompts[i].len() + step,
                    slot: i,
                })
                .collect();
            let tokens: Vec<u32> = order.iter().map(|&i| next[i][step]).collect();
            let batched = model
                .forward_decode_batch(&mut rows, &tokens)
                .expect("batched step")
                .expect("the batch serves this model");
            for (&i, logits) in order.iter().zip(batched) {
                got[i].push(logits);
            }
        }
        for (i, cache) in caches.iter_mut().enumerate() {
            got[i].push(
                model
                    .forward(cache, &next[i][2..=2], prompts[i].len() + 2, i)
                    .expect("solo step after the batch"),
            );
        }
        let argmax = |v: &[f32]| crate::engine::sampling::argmax(v);
        for i in 0..2 {
            for step in 0..3 {
                let (b, s) = (&got[i][step], &solo[i][step]);
                let (mut diff2, mut val2) = (0f64, 0f64);
                for (x, y) in b.iter().zip(s) {
                    diff2 += f64::from((x - y) * (x - y));
                    val2 += f64::from(y * y);
                }
                let rms = (diff2 / val2).sqrt();
                println!(
                    "sequence {i} step {step}: rms {rms:.2e}, argmax {} / {}",
                    argmax(b),
                    argmax(s)
                );
                // ~1–2e-3 measured; a K norm left out of the batch is 5–7e-3.
                assert!(rms < 3.5e-3, "sequence {i} step {step} rms {rms}");
                assert_eq!(
                    argmax(b),
                    argmax(s),
                    "sequence {i} step {step} picks another token"
                );
            }
        }
        for (b, s) in caches.iter().zip(&solo_caches) {
            assert_eq!(b.committed_len(), s.committed_len());
        }
    }

    /// A verify taken into a batched step (`forward_decode_batch_verify`)
    /// gives the verifying sequence the logits of its own `forward_verify`
    /// at every position and the decoding one the logits of its own step,
    /// to the narrow-pass kernels' rounding — then, after the verify's
    /// rejected tail is rolled back, the same with the roles swapped, and a
    /// solo step on each after that. On the GPU, so ignored by default:
    /// `cargo test --release --bin orangu-server decode_batch_verifies -- --ignored`.
    #[test]
    #[ignore]
    fn a_decode_batch_verifies_a_draft_as_its_own_pass_would() {
        let Some(vulkan) = crate::engine::backend::vulkan::VulkanBackend::try_init() else {
            println!("no Vulkan device; nothing to check");
            return;
        };
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("contrib/orangu-model/smoke/orangu-code-smoke-Q4_K_M.gguf");
        let loaded = LoadedModel::open(&path).expect("load model");
        let model = LlamaModel::load_with_backend(&loaded, Arc::new(vulkan)).expect("build model");
        let prompts: [Vec<u32>; 2] = [
            (0..40u32).map(|i| 100 + (i * 37) % 2000).collect(),
            (0..23u32).map(|i| 300 + (i * 53) % 1700).collect(),
        ];
        let prefilled = || -> Vec<KvCache> {
            prompts
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let mut cache = model.new_kv_cache(64);
                    model.forward(&mut cache, p, 0, i).expect("prefill");
                    cache
                })
                .collect()
        };
        let compare = |what: &str, b: &[f32], s: &[f32]| {
            let (mut diff2, mut val2) = (0f64, 0f64);
            for (x, y) in b.iter().zip(s) {
                diff2 += f64::from((x - y) * (x - y));
                val2 += f64::from(y * y);
            }
            let rms = (diff2 / val2).sqrt();
            let (ab, as_) = (
                crate::engine::sampling::argmax(b),
                crate::engine::sampling::argmax(s),
            );
            println!("{what}: rms {rms:.2e}, argmax {ab} / {as_}");
            assert!(rms < 3.5e-3, "{what} rms {rms}");
            assert_eq!(ab, as_, "{what} picks another token");
        };
        let logits_of = |o: &super::super::ForwardOutcome| match o {
            super::super::ForwardOutcome::Logits(l) => l.clone(),
            _ => panic!("no sampling was asked for"),
        };
        // Round one: sequence 0 decodes, sequence 1 verifies five tokens;
        // round two, after sequence 1 keeps two of them: the other way round.
        // The rounds after those reuse what each layer cached on the first.
        let rounds: [(usize, Vec<u32>, usize, Vec<u32>); 6] = [
            (0, vec![911], 1, vec![1207, 640, 88, 1500, 7]),
            (1, vec![33], 0, vec![57, 1302, 400, 12]),
            (0, vec![12], 1, vec![5, 1606, 7]),
            (1, vec![13], 0, vec![808, 9, 1010]),
            (0, vec![14], 1, vec![11, 1212, 13]),
            (1, vec![15], 0, vec![16, 1717, 18]),
        ];
        let mut solo = prefilled();
        let mut batched = prefilled();
        let mut pos = [prompts[0].len(), prompts[1].len()];
        for (round, (dec, dec_tokens, ver, ver_tokens)) in rounds.iter().enumerate() {
            let solo_dec = model
                .forward(&mut solo[*dec], dec_tokens, pos[*dec], *dec)
                .expect("solo step");
            let solo_ver = model
                .forward_verify(&mut solo[*ver], ver_tokens, pos[*ver], *ver)
                .expect("solo verify");
            let (lo, hi) = batched.split_at_mut(1);
            let (dec_cache, ver_cache) = if *dec == 0 {
                (&mut lo[0], &mut hi[0])
            } else {
                (&mut hi[0], &mut lo[0])
            };
            let mut rows = vec![
                super::super::DecodeRow {
                    cache: dec_cache,
                    pos: pos[*dec],
                    slot: *dec,
                },
                super::super::DecodeRow {
                    cache: ver_cache,
                    pos: pos[*ver],
                    slot: *ver,
                },
            ];
            let tokens: [&[u32]; 2] = [dec_tokens, ver_tokens];
            let out = model
                .forward_decode_batch_verify(&mut rows, &tokens, &[None, None])
                .expect("batched verify")
                .expect("the batch serves this model");
            assert_eq!(out[0].len(), 1);
            assert_eq!(out[1].len(), ver_tokens.len());
            compare(
                &format!("round {round}: the decoding row"),
                &logits_of(&out[0][0]),
                &solo_dec,
            );
            for (k, (o, s)) in out[1].iter().zip(&solo_ver).enumerate() {
                compare(
                    &format!("round {round}: verify position {k}"),
                    &logits_of(o),
                    s,
                );
            }
            pos[*dec] += 1;
            // Two of the draft kept: the token and one more.
            let kept = pos[*ver] + 2;
            for caches in [&mut solo, &mut batched] {
                caches[*ver].truncate(kept).expect("roll back");
            }
            pos[*ver] = kept;
        }
        // Both verifying, no one-token row in the step: as wide as each
        // other, then not.
        let both: [[Vec<u32>; 2]; 2] = [
            [vec![71, 802, 13, 1999], vec![640, 5, 1201, 77]],
            [vec![18, 990, 4], vec![1450, 3, 760, 41, 1102, 9]],
        ];
        for (round, drafts) in both.iter().enumerate() {
            let solo_out: Vec<Vec<Vec<f32>>> = (0..2)
                .map(|i| {
                    model
                        .forward_verify(&mut solo[i], &drafts[i], pos[i], i)
                        .expect("solo verify")
                })
                .collect();
            let (lo, hi) = batched.split_at_mut(1);
            let mut rows = vec![
                super::super::DecodeRow {
                    cache: &mut lo[0],
                    pos: pos[0],
                    slot: 0,
                },
                super::super::DecodeRow {
                    cache: &mut hi[0],
                    pos: pos[1],
                    slot: 1,
                },
            ];
            let tokens: [&[u32]; 2] = [&drafts[0], &drafts[1]];
            let out = model
                .forward_decode_batch_verify(&mut rows, &tokens, &[None, None])
                .expect("batched verify")
                .expect("the batch serves this model");
            for i in 0..2 {
                assert_eq!(out[i].len(), drafts[i].len());
                for (k, (o, s)) in out[i].iter().zip(&solo_out[i]).enumerate() {
                    compare(
                        &format!("both verify {round}: sequence {i} position {k}"),
                        &logits_of(o),
                        s,
                    );
                }
                let kept = pos[i] + 1;
                for caches in [&mut solo, &mut batched] {
                    caches[i].truncate(kept).expect("roll back");
                }
                pos[i] = kept;
            }
        }
        for i in 0..2 {
            let s = model
                .forward(&mut solo[i], &[5], pos[i], i)
                .expect("solo step");
            let b = model
                .forward(&mut batched[i], &[5], pos[i], i)
                .expect("solo step after the batch");
            compare(&format!("sequence {i} after the batch"), &b, &s);
            assert_eq!(batched[i].committed_len(), solo[i].committed_len());
        }
    }

    /// A batched step that samples on the device
    /// (`forward_decode_batch_sampling`) picks what the host would pick from
    /// the batch's logits: the greedy token after the repeat penalty, the
    /// top `k` candidates, and a row that asks for nothing gets its logits.
    /// On the GPU, so ignored by default:
    /// `cargo test --release --bin orangu-server decode_batch_samples -- --ignored`.
    #[test]
    #[ignore]
    fn a_decode_batch_samples_each_row_as_the_host_would() {
        let Some(vulkan) = crate::engine::backend::vulkan::VulkanBackend::try_init() else {
            println!("no Vulkan device; nothing to check");
            return;
        };
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("contrib/orangu-model/smoke/orangu-code-smoke-Q4_K_M.gguf");
        let loaded = LoadedModel::open(&path).expect("load model");
        let model = LlamaModel::load_with_backend(&loaded, Arc::new(vulkan)).expect("build model");
        let prompts: [Vec<u32>; 3] = [
            (0..40u32).map(|i| 100 + (i * 37) % 2000).collect(),
            (0..23u32).map(|i| 300 + (i * 53) % 1700).collect(),
            (0..31u32).map(|i| 700 + (i * 29) % 1500).collect(),
        ];
        let tokens = [911u32, 1207, 64];
        let step = |samples: Option<&[Option<super::super::GreedySampleParams<'_>>]>| {
            let mut caches: Vec<KvCache> = prompts
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let mut cache = model.new_kv_cache(64);
                    model.forward(&mut cache, p, 0, i).expect("prefill");
                    cache
                })
                .collect();
            let mut rows: Vec<super::super::DecodeRow<'_>> = caches
                .iter_mut()
                .enumerate()
                .map(|(i, cache)| super::super::DecodeRow {
                    cache,
                    pos: prompts[i].len(),
                    slot: i,
                })
                .collect();
            match samples {
                Some(samples) => model
                    .forward_decode_batch_sampling(&mut rows, &tokens, samples)
                    .expect("sampled batch")
                    .expect("the batch serves this model"),
                None => model
                    .forward_decode_batch(&mut rows, &tokens)
                    .expect("batched step")
                    .expect("the batch serves this model")
                    .into_iter()
                    .map(super::super::ForwardOutcome::Logits)
                    .collect(),
            }
        };
        let logits: Vec<Vec<f32>> = step(None)
            .into_iter()
            .map(|o| match o {
                super::super::ForwardOutcome::Logits(l) => l,
                _ => unreachable!("no sampling asked for"),
            })
            .collect();
        let argmax = |v: &[f32]| crate::engine::sampling::argmax(v);
        // Row 0's penalty covers its own favourite, so the penalty decides.
        let recent = [argmax(&logits[0]), 5, 17];
        let penalty = 1.3f32;
        let mut penalized = logits[0].clone();
        for &t in &recent {
            let v = &mut penalized[t as usize];
            *v = if *v > 0.0 { *v / penalty } else { *v * penalty };
        }
        let k = 8u32;
        let samples = [
            Some(super::super::GreedySampleParams {
                recent_tokens: &recent,
                repeat_penalty: penalty,
                top_k: 0,
            }),
            Some(super::super::GreedySampleParams {
                recent_tokens: &[],
                repeat_penalty: 1.0,
                top_k: k,
            }),
            None,
        ];
        let sampled = step(Some(&samples));
        match &sampled[0] {
            super::super::ForwardOutcome::Token(t) => {
                println!("greedy with a penalty: {t} / {}", argmax(&penalized));
                assert_eq!(*t, argmax(&penalized));
            }
            _ => panic!("row 0 asked for the greedy token"),
        }
        match &sampled[1] {
            super::super::ForwardOutcome::Candidates(c) => {
                let mut want: Vec<u32> = (0..logits[1].len() as u32).collect();
                want.sort_by(|a, b| logits[1][*b as usize].total_cmp(&logits[1][*a as usize]));
                want.truncate(k as usize);
                let got: Vec<u32> = c.iter().map(|(t, _)| *t).collect();
                println!("top {k}: {got:?} / {want:?}");
                assert_eq!(got, want);
                for (t, v) in c {
                    let host = logits[1][*t as usize];
                    assert!(
                        (v - host).abs() <= 1e-2 * host.abs().max(1.0),
                        "{t}: {v} / {host}"
                    );
                }
            }
            _ => panic!("row 1 asked for candidates"),
        }
        match &sampled[2] {
            super::super::ForwardOutcome::Logits(l) => {
                let (mut diff2, mut val2) = (0f64, 0f64);
                for (x, y) in l.iter().zip(&logits[2]) {
                    diff2 += f64::from((x - y) * (x - y));
                    val2 += f64::from(y * y);
                }
                let rms = (diff2 / val2).sqrt();
                println!("logits row: rms {rms:.2e}");
                assert!(rms < 3.5e-3, "rms {rms}");
            }
            _ => panic!("row 2 asked for nothing"),
        }
    }

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
    /// The control for `bailingmoe3`'s equivalence probe: how far apart a
    /// one-pass forward and the same tokens fed one at a time are on a
    /// **dense** model, where nothing carries state between positions.
    ///
    /// The two take different kernels — a multi-token batch goes through
    /// the tiled path, a single token through the decode one — so they are
    /// not bit-identical and never were. What this pins is the *size* of
    /// that, which is the number the recurrent model's drift has to be read
    /// against: without it, ordinary kernel rounding and a real batching
    /// bug look the same.
    ///
    /// Run with `ORANGU_TEST_LLAMA_MODEL=/path/to/Llama-3.2-1B-Instruct-
    /// Q4_K_M.gguf cargo test --release --bin orangu-server
    /// one_pass_equals_one_token -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn one_pass_equals_one_token_at_a_time_on_a_dense_model() {
        let path = std::env::var("ORANGU_TEST_LLAMA_MODEL").expect("set ORANGU_TEST_LLAMA_MODEL");
        let loaded = LoadedModel::open(std::path::Path::new(&path)).expect("load model");
        let model =
            LlamaModel::load_with_backend(&loaded, Arc::new(crate::engine::backend::CpuBackend))
                .expect("build model");
        let gguf = orangu::gguf::GgufFile::open(std::path::Path::new(&path)).expect("open gguf");
        let tokenizer =
            crate::engine::tokenizer::Tokenizer::from_gguf(&gguf).expect("build tokenizer");
        let tokens = tokenizer.encode(
            "The capital of France is Paris, and the capital of Germany is Berlin. \
             Lists of European capitals are a common way to test whether a model \
             can recall a fact stated earlier in the same prompt.",
            tokenizer.wants_bos(),
        );
        println!("{:>6}  {:>12}  {:>10}", "tokens", "worst logit", "top-1");
        for n in [1usize, 2, 4, 8, 16, 32] {
            if n > tokens.len() {
                break;
            }
            let part = &tokens[..n];
            let mut batched = model.new_kv_cache(n + 1);
            let whole = model.forward(&mut batched, part, 0, 0).expect("one pass");
            let mut cache = model.new_kv_cache(n + 1);
            let mut stepped = Vec::new();
            for (pos, token) in part.iter().enumerate() {
                stepped = model
                    .forward(&mut cache, &[*token], pos, 0)
                    .expect("one token");
            }
            let worst = whole
                .iter()
                .zip(&stepped)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let top = |v: &[f32]| {
                v.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).expect("finite"))
                    .expect("non-empty")
                    .0
            };
            println!(
                "{n:>6}  {worst:>12.6}  {:>10}",
                if top(&whole) == top(&stepped) {
                    "same"
                } else {
                    "DIFFER"
                }
            );
        }
    }

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
                "llama" | "mistral" | "qwen2" | "qwen3" | "qwen3vl" | "qwen3moe"
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

    /// **The mixture of this family**, end to end on real weights:
    /// `qwen3moe` is this module's block with routed experts in place of
    /// the dense feed-forward, and what that replacement has to get right
    /// is a per-token choice of eight experts out of 128 — a router read
    /// the wrong way round, a softmax taken per logit, or weights left
    /// unnormalized all produce fluent text rather than an error.
    ///
    /// A one-word factual answer is the check for the same reason the test
    /// above gives, and this one is stricter than it looks: every layer of
    /// this model is a mixture, so an expert path that is wrong anywhere
    /// cannot be hidden by the ones that are right.
    ///
    /// Verified against real `llama.cpp` on the same file before this test
    /// was written: `llama-server` and this engine produce **byte-identical
    /// greedy continuations** for "The capital of France is" and "def
    /// fibonacci(n):" at `temperature = 0`.
    ///
    /// Run with `ORANGU_TEST_QWEN3MOE_MODEL=/path/to/Qwen3-Coder-30B-A3B-
    /// Instruct-Q4_K_M.gguf cargo test --release --bin orangu-server
    /// real_model_tests -- --ignored`.
    #[test]
    #[ignore]
    fn qwen3moe_answers_a_factual_question() {
        let path =
            std::env::var("ORANGU_TEST_QWEN3MOE_MODEL").expect("set ORANGU_TEST_QWEN3MOE_MODEL");
        let loaded = LoadedModel::open(std::path::Path::new(&path)).expect("load model");
        assert_eq!(
            loaded.config.architecture, "qwen3moe",
            "this test is about the routed feed-forward"
        );
        let gguf = orangu::gguf::GgufFile::open(std::path::Path::new(&path)).expect("open gguf");
        let tokenizer =
            crate::engine::tokenizer::Tokenizer::from_gguf(&gguf).expect("build tokenizer");
        let model =
            LlamaModel::load_with_backend(&loaded, Arc::new(crate::engine::backend::CpuBackend))
                .expect("build model");
        // Every layer routes: the file carries no dense feed-forward at
        // all, and reading one as dense would have failed to load rather
        // than answered anything.
        assert!(
            model.layers.iter().all(|layer| layer.dense().is_none()),
            "every qwen3moe layer is a mixture"
        );
        assert_eq!(
            model.routing.gating,
            crate::engine::arch::ExpertGating::Softmax
        );
        assert!(model.routing.weights_norm, "top-k weights are renormalized");
        assert_eq!(model.routing.n_expert_used, 8);

        let tokens = tokenizer.encode("The capital of France is", true);
        let mut cache = model.new_kv_cache(tokens.len() + 8);
        let mut logits = model.forward(&mut cache, &tokens, 0, 0).expect("prefill");
        let mut generated = Vec::new();
        for step in 0..4 {
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).expect("logits are finite"))
                .expect("non-empty logits")
                .0 as u32;
            generated.push(next);
            logits = model
                .forward(&mut cache, &[next], tokens.len() + step, 0)
                .expect("decode");
        }
        let text = tokenizer.decode(&generated);
        assert!(
            text.contains("Paris"),
            "expected the continuation to name Paris, got {text:?}"
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

#[cfg(test)]
mod batch_tests {
    use super::*;

    fn model() -> Option<LlamaModel> {
        let vulkan = crate::engine::backend::vulkan::VulkanBackend::try_init()?;
        let path = std::env::var("ORANGU_TEST_LLAMA_MODEL").expect("set ORANGU_TEST_LLAMA_MODEL");
        let loaded = LoadedModel::open(std::path::Path::new(&path)).expect("load model");
        Some(LlamaModel::load_with_backend(&loaded, Arc::new(vulkan)).expect("build model"))
    }

    #[test]
    #[ignore]
    fn a_decode_batch_matches_each_rows_solo_step() {
        let Some(model) = model() else {
            println!("no Vulkan device; nothing to check");
            return;
        };
        super::super::batch_checks::decode_batch_matches_each_rows_solo_step(&model, 3.5e-3);
    }

    /// A verify sampled on the device picks, row by row, the token the
    /// rows' own logits pick — greedy as the argmax, sampled as the largest
    /// candidates, largest first.
    #[test]
    #[ignore]
    fn a_verify_sampled_on_the_device_picks_what_its_logits_pick() {
        let Some(model) = model() else {
            println!("no Vulkan device; nothing to check");
            return;
        };
        let tokens: Vec<u32> = vec![128000, 791, 6864, 315, 9822, 374, 12366, 13, 578, 6864];
        let (prompt, draft) = tokens.split_at(4);
        let mut base = model.new_kv_cache(64);
        model.forward(&mut base, prompt, 0, 0).expect("prompt");
        let mut plain = base.duplicate();
        let logits = model
            .forward_verify(&mut plain, draft, prompt.len(), 0)
            .expect("verify");
        for top_k in [0u32, 8] {
            let mut cache = base.duplicate();
            let params = super::super::GreedySampleParams {
                recent_tokens: &[],
                repeat_penalty: 1.0,
                top_k,
            };
            let rows = model
                .forward_verify_sampled(
                    &mut cache,
                    draft,
                    prompt.len(),
                    0,
                    &vec![params; draft.len()],
                )
                .expect("sampled verify")
                .expect("the device path runs");
            for (row, logits) in rows.iter().zip(&logits) {
                let want = crate::engine::sampling::argmax(logits);
                match row {
                    super::super::ForwardOutcome::Token(t) => assert_eq!(*t, want),
                    super::super::ForwardOutcome::Candidates(c) => {
                        assert_eq!(c.len(), top_k as usize);
                        assert_eq!(c[0].0, want);
                        assert!(c.windows(2).all(|w| w[0].1 >= w[1].1));
                    }
                    super::super::ForwardOutcome::Logits(_) => panic!("read back whole rows"),
                }
            }
        }
    }

    /// Steps submitted before the previous pick is read, each gathering its
    /// input from that pick on the device, decode the tokens plain greedy
    /// steps decode.
    #[test]
    #[ignore]
    fn steps_taken_ahead_decode_what_plain_greedy_steps_decode() {
        let Some(model) = model() else {
            println!("no Vulkan device; nothing to check");
            return;
        };
        let vulkan = model.backend.as_wgpu().expect("a device");
        let prompt: Vec<u32> = vec![128000, 791, 6864, 315, 9822, 374];
        let mut base = model.new_kv_cache(64);
        let logits = model.forward(&mut base, &prompt, 0, 0).expect("prompt");
        let first = crate::engine::sampling::argmax(&logits);
        let greedy = super::super::GreedySampleParams {
            recent_tokens: &[],
            repeat_penalty: 1.0,
            top_k: 0,
        };
        const STEPS: usize = 16;
        let mut plain = vec![first];
        let mut cache = base.duplicate();
        for i in 0..STEPS {
            let pos = prompt.len() + i;
            let out = model
                .forward_maybe_sampling(&mut cache, &[plain[i]], pos, Some(greedy), 0)
                .expect("step");
            match out {
                super::super::ForwardOutcome::Token(t) => plain.push(t),
                _ => panic!("the device did not pick"),
            }
        }
        let mut ahead = vec![first];
        let mut cache = base.duplicate();
        let mut pending = model
            .decode_lookahead(
                &mut cache,
                super::super::LookaheadInput::Token(first),
                prompt.len(),
                0,
            )
            .expect("step")
            .expect("the lookahead path runs");
        for i in 0..STEPS {
            let next = (i + 1 < STEPS).then(|| {
                model
                    .decode_lookahead(
                        &mut cache,
                        super::super::LookaheadInput::Pending(&pending),
                        prompt.len() + i + 1,
                        0,
                    )
                    .expect("step")
                    .expect("the lookahead path runs")
            });
            ahead.push(vulkan.read_token(pending));
            match next {
                Some(p) => pending = p,
                None => break,
            }
        }
        assert_eq!(ahead, plain);
    }

    #[test]
    #[ignore]
    fn a_decode_batch_verifies_a_draft_as_its_own_pass_would() {
        let Some(model) = model() else {
            println!("no Vulkan device; nothing to check");
            return;
        };
        super::super::batch_checks::decode_batch_verifies_a_draft_as_its_own_pass_would(
            &model, 3.5e-3,
        );
    }
}
