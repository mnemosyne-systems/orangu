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

//! Inkling 2 (`general.architecture = "inkling2"` or `"inkling2_loop"`,
//! e.g. `unsloth/Inkling-2-Mini-GGUF`) — the successor of
//! `engine::arch::inkling`, and a different network rather than a revision
//! of it. What carries over is the rotation-free attention: position still
//! enters through a learned relative-position bias and causal short
//! convolutions, never a rotary embedding. Almost everything around it is
//! new:
//!
//! - **Block attention residuals.** The residual stream is cut into blocks
//!   of `attn_res.block_size` layers. Every sub-layer's input is not the
//!   running residual but an *attention over depth*: the RMS-normed outputs
//!   of every earlier block plus the current stream, mixed by a softmax
//!   against a learned query (`attn_res_query`, `ffn_res_query`,
//!   `output_res_query`). See [`mix`].
//! - **Separate Q, K and V inputs on the full-attention layers.** Their
//!   `attn_res_query` has three rows, one mixed input per projection, each
//!   with its own input norm (`attn_{q,k,v}_inp_norm`). The sliding-window
//!   layers mix one input and norm it once (`attn_norm`).
//! - **L2-clipped queries and keys** instead of a QK norm: each head's row
//!   is scaled down to a norm of at most `attention.qk_clip` (after a
//!   per-channel scale on the query side, `attn_q_clip`), and the value is
//!   RMS-normed without a weight. The relative-bias coefficients are
//!   clipped the same way to `attention.relative_clip`.
//! - **A per-query attention sink.** `attn_sink_proj` gives every
//!   `(token, head)` one extra logit that takes part in the softmax and
//!   contributes no value — the probability mass the head may put
//!   nowhere.
//! - **An output gate**, `sigmoid(attn_gate · x)`, on the attention result
//!   before its output projection.
//! - **Input-dependent convolution taps.** On the full-attention layers the
//!   key and value convolutions add a second, per-head set of taps
//!   projected from the layer input (`attn_{k,v}_conv_proj`). Every
//!   convolution keeps its residual and is then divided by
//!   `sqrt(taps + 1)` — `sqrt(2 * taps + 1)` with the dynamic taps.
//! - **SiTU experts**: `beta * tanh(gate / beta) * sigmoid(gate) * lb *
//!   tanh(up / lb)` on a fused `ffn_gate_up_exps`, sigmoid-routed with a
//!   selection bias, normalized and scaled by `expert_weights_scale`. NVFP4
//!   expert stacks carry a per-expert output scale (`*_exps.scale`).
//! - **A final logit softcap**, after the output projection.
//! - **A looped core** (`inkling2_loop`, and the `inkling2` files that carry
//!   `num_loops`): layers `[core_start, core_end)` run `passes` times. Every
//!   pass after the first restarts from the prelude's output and mixes a
//!   *message* from the previous pass into each sub-layer input, gated by a
//!   low-rank controller (see [`Inkling2Model::control`]). Each pass has its
//!   own KV and convolution state, so the cache holds
//!   `n_layer + (passes - 1) * (core_end - core_start)` layer slots.
//!
//! The reference is upstream llama.cpp's `src/models/inkling2.cpp` and
//! `inkling2-loop.cpp`. Two tensor layouts exist for the loop's message
//! queries — a per-layer `loop_msg_query` (upstream's converter) and the
//! global `loop_res_query` / `loop_kv_res_query` pair the first published
//! GGUFs carry, alongside `loop_{attn,k,v,mlp}_*` controller names — and
//! both are read.
//!
//! CPU-orchestrated like `engine::arch::inkling`: every matmul goes to the
//! `Backend`, but the attention loop is the host's, since the relative bias
//! and the sink are per-`(token, head)` terms no attention kernel here
//! takes. The vision projector shipped beside these weights is not loaded.

use anyhow::{Context, Result};
use rayon::prelude::*;
use std::sync::Arc;

use super::{ExpertProjection, ModelForward, Seqs};
use crate::engine::backend::{Backend, MatmulOp};
use crate::engine::kv_cache::{KvCache, RecurrentSpec};
use crate::engine::loader::{ExpertQuantMatrix, LoadedModel, ModelConfig, QuantMatrix};
use crate::engine::moe_stats;
use crate::engine::tensor;

/// Which of a layer slot's four short convolutions a `KvCache::recurrent`
/// entry holds: slot `s` owns entries `s * CONV_PER_SLOT ..`.
const CONV_K: usize = 0;
const CONV_V: usize = 1;
const CONV_ATTN: usize = 2;
const CONV_FFN: usize = 3;
const CONV_PER_SLOT: usize = 4;

/// The epsilon of the weightless RMS norms in the depth mixing and the loop
/// controllers — fixed upstream, not the file's `layer_norm_rms_epsilon`.
const MIX_EPS: f32 = 1e-6;

/// Added to the routing weights' sum before it divides them — upstream's
/// `expert_weights_norm_eps` for this architecture.
const ROUTE_NORM_EPS: f32 = 1e-20;

/// The message controllers of a looped layer, in the order upstream indexes
/// them and the message queries are stacked.
const CTRL_Q: usize = 0;
const CTRL_K: usize = 1;
const CTRL_V: usize = 2;
const CTRL_FFN: usize = 3;
const CTRL_COUNT: usize = 4;

struct Inkling2Layer {
    is_swa: bool,
    n_head: usize,
    n_head_kv: usize,
    /// `attn_norm` on a sliding-window layer; the three input norms on a
    /// full-attention one.
    attn_norm: Option<Vec<f32>>,
    q_inp_norm: Option<Vec<f32>>,
    k_inp_norm: Option<Vec<f32>>,
    v_inp_norm: Option<Vec<f32>>,
    wq: QuantMatrix,
    wk: QuantMatrix,
    wv: QuantMatrix,
    wo: QuantMatrix,
    /// `attn_gate`, `[n_embd, n_head * head_dim]` — the output gate.
    wgate: QuantMatrix,
    /// `attn_sink_proj`, `[n_embd, n_head]` — one sink logit per head.
    wsink: QuantMatrix,
    /// `attn_rel`, `[n_embd, n_head * n_rel]`.
    wrel: QuantMatrix,
    /// `attn_q_clip`, `[head_dim, n_head]` — the query's per-channel scale
    /// before its clip.
    q_clip: Vec<f32>,
    /// `attn_rel_clip`, `[n_rel, n_head]`.
    rel_clip: Vec<f32>,
    /// `attn_rel_bias`, `[n_rel, extent]`: `rel_bias[d * n_rel + i]`.
    rel_bias: Vec<f32>,
    /// Distances the bias bank covers — `attention.sliding_window` on a
    /// sliding-window layer, `attention.relative_extent` otherwise.
    extent: usize,
    conv_k: Vec<f32>,
    conv_v: Vec<f32>,
    conv_attn: Vec<f32>,
    conv_ffn: Vec<f32>,
    /// `attn_{k,v}_conv_proj`, `[n_embd, taps * n_head_kv]` — the
    /// input-dependent taps of a full-attention layer.
    k_conv_proj: Option<QuantMatrix>,
    v_conv_proj: Option<QuantMatrix>,
    /// `attn_res_query`, one `n_embd` row per mixed input (one or three);
    /// absent on layer 0, which reads the embeddings directly.
    attn_res_query: Option<Vec<f32>>,
    ffn_res_query: Vec<f32>,
    ffn_norm: Vec<f32>,
    gate_inp: QuantMatrix,
    exp_probs_b: Option<Vec<f32>>,
    gate_up_exps: ExpertQuantMatrix,
    down_exps: ExpertQuantMatrix,
    gate_up_scale: Option<Vec<f32>>,
    down_scale: Option<Vec<f32>>,
}

/// One looped layer's message machinery.
struct LoopLayer {
    /// `[CTRL_COUNT, n_embd]` — the queries that mix the previous pass into
    /// the q, k, v and ffn messages.
    msg_query: Vec<f32>,
    /// `[2 * n_embd, rank]` per controller.
    down: Vec<QuantMatrix>,
    /// `[rank, n_embd]` per controller.
    gain: Vec<QuantMatrix>,
    forget: Vec<QuantMatrix>,
}

struct LoopCore {
    start: usize,
    end: usize,
    passes: usize,
    /// Indexed by `layer - start`.
    layers: Vec<LoopLayer>,
}

pub struct Inkling2Model {
    config: ModelConfig,
    backend: Arc<dyn Backend>,
    tok_embeddings: QuantMatrix,
    tok_embd_norm: Vec<f32>,
    output_norm: Vec<f32>,
    output_res_query: Vec<f32>,
    output_weight: QuantMatrix,
    layers: Vec<Inkling2Layer>,
    core: Option<LoopCore>,
    n_swa: usize,
    n_rel: usize,
    conv_taps: usize,
    qk_clip: f32,
    rel_clip: f32,
    log_floor: f64,
    log_alpha: f32,
    block_size: usize,
    n_expert: usize,
    n_expert_used: usize,
    route_scale: f32,
    situ_beta: f32,
    situ_linear_beta: f32,
    softcap: f32,
    unpadded_vocab: usize,
}

/// The first of `keys` the file sets.
fn first_u64(loaded: &LoadedModel, keys: &[&str]) -> Option<u64> {
    keys.iter().find_map(|k| loaded.metadata_u64(k))
}

impl Inkling2Model {
    pub fn load_with_backend(loaded: &LoadedModel, backend: Arc<dyn Backend>) -> Result<Self> {
        let config = loaded.config.clone();
        // `block_count` and the per-layer arrays are the physical layers in
        // upstream's files, but the cache slots — physical layers plus one
        // per core layer for every extra pass — in the first published
        // looped ones. The tensors say which: the physical layers are the
        // ones that exist, and the arrays' leading entries describe them
        // either way.
        let n_layer = config.n_layer;
        let n_phys = (0..n_layer)
            .take_while(|i| loaded.has_tensor(&format!("blk.{i}.attn_q.weight")))
            .count();
        anyhow::ensure!(n_phys > 0, "inkling2: no blk.0.attn_q.weight");
        let n_embd = config.n_embd;
        let head_dim = config.head_dim;

        let per_layer = |key: &str, default: usize| -> Vec<usize> {
            match loaded.metadata_array_u64(key) {
                Some(values) => (0..n_layer)
                    .map(|i| values.get(i).copied().unwrap_or(default as u64) as usize)
                    .collect(),
                None => vec![
                    loaded.metadata_u64(key).map_or(default, |v| v as usize);
                    n_layer
                ],
            }
        };
        let n_head_per_layer = per_layer("attention.head_count", config.n_head);
        let n_head_kv_per_layer = per_layer("attention.head_count_kv", config.n_head_kv);
        let swa_pattern = loaded
            .metadata_array_u64("attention.sliding_window_pattern")
            .context("missing attention.sliding_window_pattern")?;
        anyhow::ensure!(
            swa_pattern.len() == n_layer,
            "attention.sliding_window_pattern has {} entries for {n_layer} layers",
            swa_pattern.len()
        );
        if let Some(k_swa) = loaded.metadata_u64("attention.key_length_swa") {
            anyhow::ensure!(
                k_swa as usize == head_dim,
                "a sliding-window head width of {k_swa} differs from {head_dim}, which this \
                 implementation does not support"
            );
        }

        let required = |key: &str| -> Result<u64> {
            loaded
                .metadata_u64(key)
                .with_context(|| format!("missing {key}"))
        };
        let required_f32 = |key: &str| -> Result<f32> {
            loaded
                .metadata_f32(key)
                .with_context(|| format!("missing {key}"))
        };
        let n_swa = required("attention.sliding_window")? as usize;
        let n_rel = required("attention.relative_length")? as usize;
        let rel_extent = required("attention.relative_extent")? as usize;
        let conv_taps = required("shortconv.l_cache")? as usize;
        let block_size = required("attn_res.block_size")? as usize;
        let qk_clip = required_f32("attention.qk_clip")?;
        let rel_clip = required_f32("attention.relative_clip")?;
        let log_floor = required_f32("attention.log_scale_floor")? as f64;
        let log_alpha = required_f32("attention.log_scale_alpha")?;
        let situ_beta = required_f32("activation.situ_beta")?;
        let situ_linear_beta = required_f32("activation.situ_linear_beta")?;
        let route_scale = loaded
            .metadata_f32("expert_weights_scale")
            .filter(|s| *s != 0.0)
            .unwrap_or(1.0);
        let softcap = loaded.metadata_f32("final_logit_softcapping").unwrap_or(0.0);
        let n_expert = required("expert_count")? as usize;
        let n_expert_used = required("expert_used_count")? as usize;
        let unpadded_vocab = first_u64(loaded, &["vocab_size_unpadded", "unpadded_vocab_size"])
            .map_or(config.n_vocab, |v| v as usize)
            .min(config.n_vocab);
        anyhow::ensure!(
            n_swa > 0 && n_rel > 0 && rel_extent > 0 && block_size > 0 && log_floor > 0.0,
            "inkling2: zero sliding window, relative width, extent, block size or log floor"
        );
        anyhow::ensure!(
            (2..=64).contains(&conv_taps),
            "inkling2: shortconv.l_cache must be in 2..=64 (got {conv_taps})"
        );
        for (name, value) in [
            ("qk_clip", qk_clip),
            ("relative_clip", rel_clip),
            ("situ_beta", situ_beta),
            ("situ_linear_beta", situ_linear_beta),
        ] {
            anyhow::ensure!(
                value.is_finite() && value > 0.0,
                "inkling2: {name} must be finite and positive (got {value})"
            );
        }
        anyhow::ensure!(
            n_expert > 0 && n_expert_used > 0 && n_expert_used <= n_expert,
            "inkling2: {n_expert_used} of {n_expert} experts"
        );

        let tok_embeddings = loaded
            .matrix("token_embd.weight")
            .context("loading token_embd.weight")?;
        let vector = |name: &str| -> Result<Vec<f32>> {
            Ok(loaded
                .tensor(name)
                .with_context(|| format!("loading {name}"))?
                .0)
        };
        let optional_vector = |name: &str| -> Result<Option<Vec<f32>>> {
            if loaded.has_tensor(name) {
                vector(name).map(Some)
            } else {
                Ok(None)
            }
        };
        let matrix = |name: &str| -> Result<QuantMatrix> {
            loaded
                .matrix(name)
                .with_context(|| format!("loading {name}"))
        };
        let sized = |name: &str, len: usize| -> Result<Vec<f32>> {
            let v = vector(name)?;
            anyhow::ensure!(
                v.len() == len,
                "{name} has {} values, expected {len}",
                v.len()
            );
            Ok(v)
        };

        let tok_embd_norm = sized("token_embd_norm.weight", n_embd)?;
        let output_norm = sized("output_norm.weight", n_embd)?;
        let output_res_query = sized("output_res_query.weight", n_embd)?;
        let output_weight = if loaded.has_tensor("output.weight") {
            matrix("output.weight")?
        } else {
            tok_embeddings.clone()
        };

        let mut layers = Vec::with_capacity(n_phys);
        for i in 0..n_phys {
            let name = |suffix: &str| format!("blk.{i}.{suffix}");
            let is_swa = swa_pattern[i] != 0;
            let n_head = n_head_per_layer[i];
            let n_head_kv = n_head_kv_per_layer[i];
            anyhow::ensure!(
                n_head > 0 && n_head_kv > 0 && n_head.is_multiple_of(n_head_kv),
                "layer {i}: {n_head} query heads do not divide into {n_head_kv} KV heads"
            );
            let kv_dim = n_head_kv * head_dim;
            let extent = if is_swa { n_swa } else { rel_extent };

            let wq = matrix(&name("attn_q.weight"))?;
            anyhow::ensure!(
                wq.out_dim == n_head * head_dim,
                "layer {i}: attn_q projects to {}, expected {n_head} heads x {head_dim}",
                wq.out_dim
            );
            let wrel = matrix(&name("attn_rel.weight"))?;
            anyhow::ensure!(
                wrel.out_dim == n_head * n_rel,
                "layer {i}: attn_rel projects to {}, expected {n_head} heads x {n_rel}",
                wrel.out_dim
            );
            let wsink = matrix(&name("attn_sink_proj.weight"))?;
            anyhow::ensure!(
                wsink.out_dim == n_head,
                "layer {i}: attn_sink_proj has {} outputs, expected {n_head}",
                wsink.out_dim
            );
            // The bias bank's width is the independent witness for which
            // kind of layer this is, as in `engine::arch::inkling`.
            let rel_bias = vector(&name("attn_rel_bias.weight"))?;
            anyhow::ensure!(
                rel_bias.len() == n_rel * extent,
                "layer {i}: attn_rel_bias has {} values, but the sliding-window pattern says \
                 this layer is {} and so wants {n_rel} x {extent}",
                rel_bias.len(),
                if is_swa {
                    "sliding-window"
                } else {
                    "full-attention"
                }
            );

            let (attn_norm, q_inp_norm, k_inp_norm, v_inp_norm, k_conv_proj, v_conv_proj) =
                if is_swa {
                    (
                        Some(sized(&name("attn_norm.weight"), n_embd)?),
                        None,
                        None,
                        None,
                        None,
                        None,
                    )
                } else {
                    let k_conv_proj = matrix(&name("attn_k_conv_proj.weight"))?;
                    let v_conv_proj = matrix(&name("attn_v_conv_proj.weight"))?;
                    for (what, m) in [("k", &k_conv_proj), ("v", &v_conv_proj)] {
                        anyhow::ensure!(
                            m.out_dim == conv_taps * n_head_kv,
                            "layer {i}: attn_{what}_conv_proj has {} outputs, expected \
                             {conv_taps} taps x {n_head_kv} heads",
                            m.out_dim
                        );
                    }
                    (
                        None,
                        Some(sized(&name("attn_q_inp_norm.weight"), n_embd)?),
                        Some(sized(&name("attn_k_inp_norm.weight"), n_embd)?),
                        Some(sized(&name("attn_v_inp_norm.weight"), n_embd)?),
                        Some(k_conv_proj),
                        Some(v_conv_proj),
                    )
                };

            let attn_res_query = if i > 0 {
                let rows = if is_swa { 1 } else { 3 };
                Some(sized(&name("attn_res_query.weight"), rows * n_embd)?)
            } else {
                None
            };

            let gate_up_exps = loaded
                .expert_matrix(&name("ffn_gate_up_exps.weight"))
                .with_context(|| format!("loading {}", name("ffn_gate_up_exps.weight")))?;
            let down_exps = loaded
                .expert_matrix(&name("ffn_down_exps.weight"))
                .with_context(|| format!("loading {}", name("ffn_down_exps.weight")))?;
            anyhow::ensure!(
                gate_up_exps.n_expert == n_expert
                    && down_exps.n_expert == n_expert
                    && gate_up_exps.out_dim.is_multiple_of(2)
                    && down_exps.in_dim == gate_up_exps.out_dim / 2,
                "layer {i}: expert stacks do not match {n_expert} experts with a fused gate/up"
            );
            let gate_up_scale = optional_vector(&name("ffn_gate_up_exps.scale"))?;
            let down_scale = optional_vector(&name("ffn_down_exps.scale"))?;
            for scale in [&gate_up_scale, &down_scale].into_iter().flatten() {
                anyhow::ensure!(
                    scale.len() == n_expert,
                    "layer {i}: an expert scale has {} values, expected {n_expert}",
                    scale.len()
                );
            }
            let gate_inp = matrix(&name("ffn_gate_inp.weight"))?;
            anyhow::ensure!(
                gate_inp.out_dim == n_expert,
                "layer {i}: ffn_gate_inp scores {} experts, expected {n_expert}",
                gate_inp.out_dim
            );

            layers.push(Inkling2Layer {
                is_swa,
                n_head,
                n_head_kv,
                attn_norm,
                q_inp_norm,
                k_inp_norm,
                v_inp_norm,
                wq,
                wk: matrix(&name("attn_k.weight"))?,
                wv: matrix(&name("attn_v.weight"))?,
                wo: matrix(&name("attn_output.weight"))?,
                wgate: matrix(&name("attn_gate.weight"))?,
                wsink,
                wrel,
                q_clip: sized(&name("attn_q_clip.weight"), n_head * head_dim)?,
                rel_clip: sized(&name("attn_rel_clip.weight"), n_head * n_rel)?,
                rel_bias,
                extent,
                conv_k: sized(&name("attn_k_conv.weight"), kv_dim * conv_taps)?,
                conv_v: sized(&name("attn_v_conv.weight"), kv_dim * conv_taps)?,
                conv_attn: sized(&name("attn_post_conv.weight"), n_embd * conv_taps)?,
                conv_ffn: sized(&name("ffn_post_conv.weight"), n_embd * conv_taps)?,
                k_conv_proj,
                v_conv_proj,
                attn_res_query,
                ffn_res_query: sized(&name("ffn_res_query.weight"), n_embd)?,
                ffn_norm: sized(&name("ffn_norm.weight"), n_embd)?,
                gate_inp,
                exp_probs_b: optional_vector(&name("exp_probs_b.bias"))?,
                gate_up_exps,
                down_exps,
                gate_up_scale,
                down_scale,
            });
        }

        let core = Self::load_core(loaded, n_phys, n_embd, block_size)?;
        let n_slots = n_phys
            + core
                .as_ref()
                .map_or(0, |c| (c.passes - 1) * (c.end - c.start));
        anyhow::ensure!(
            n_layer == n_phys || n_layer == n_slots,
            "inkling2: block_count {n_layer} is neither the {n_phys} layers present nor their \
             {n_slots} cache slots"
        );

        Ok(Self {
            config,
            backend,
            tok_embeddings,
            tok_embd_norm,
            output_norm,
            output_res_query,
            output_weight,
            layers,
            core,
            n_swa,
            n_rel,
            conv_taps,
            qk_clip,
            rel_clip,
            log_floor,
            log_alpha,
            block_size,
            n_expert,
            n_expert_used,
            route_scale,
            situ_beta,
            situ_linear_beta,
            softcap,
            unpadded_vocab,
        })
    }

    /// The looped core, when the file declares one with more than one pass.
    fn load_core(
        loaded: &LoadedModel,
        n_layer: usize,
        n_embd: usize,
        block_size: usize,
    ) -> Result<Option<LoopCore>> {
        let Some(passes) = first_u64(loaded, &["loop.count", "num_loops"]) else {
            return Ok(None);
        };
        let passes = passes as usize;
        if passes < 2 {
            return Ok(None);
        }
        let start = first_u64(loaded, &["loop.start", "loop.core_start"])
            .context("a looped file without loop.start")? as usize;
        let end = first_u64(loaded, &["loop.end", "loop.core_end"])
            .context("a looped file without loop.end")? as usize;
        anyhow::ensure!(
            0 < start
                && start < end
                && end <= n_layer
                && start.is_multiple_of(block_size)
                && end.is_multiple_of(block_size),
            "inkling2: invalid loop core [{start}, {end}) with block size {block_size} for \
             {n_layer} layers"
        );
        let n_core = end - start;

        // Message queries: per layer (`loop_msg_query`, `[n_embd, 4]`), or
        // the two stacked tensors of the first published files, where core
        // layer `j` reads `q = res[2j]`, `ffn = res[2j + 1]`,
        // `k = kv[2j]`, `v = kv[2j + 1]`.
        let stacked = if loaded.has_tensor("loop_res_query.weight") {
            let (res, _) = loaded.tensor("loop_res_query.weight")?;
            let (kv, _) = loaded
                .tensor("loop_kv_res_query.weight")
                .context("loading loop_kv_res_query.weight")?;
            anyhow::ensure!(
                res.len() == 2 * n_core * n_embd && kv.len() == res.len(),
                "loop_res_query / loop_kv_res_query do not hold two queries per core layer"
            );
            Some((res, kv))
        } else {
            None
        };

        let names: [(&str, &str); CTRL_COUNT] = [
            ("loop_q_ctrl", "loop_attn"),
            ("loop_k_ctrl", "loop_k"),
            ("loop_v_ctrl", "loop_v"),
            ("loop_ffn_ctrl", "loop_mlp"),
        ];
        let mut layers = Vec::with_capacity(n_core);
        for il in start..end {
            let j = il - start;
            let msg_query = match &stacked {
                Some((res, kv)) => {
                    let row = |src: &Vec<f32>, r: usize| src[r * n_embd..(r + 1) * n_embd].to_vec();
                    [
                        row(res, 2 * j),
                        row(kv, 2 * j),
                        row(kv, 2 * j + 1),
                        row(res, 2 * j + 1),
                    ]
                    .concat()
                }
                None => {
                    let name = format!("blk.{il}.loop_msg_query.weight");
                    let (q, _) = loaded
                        .tensor(&name)
                        .with_context(|| format!("loading {name}"))?;
                    anyhow::ensure!(
                        q.len() == CTRL_COUNT * n_embd,
                        "{name} has {} values, expected {CTRL_COUNT} x {n_embd}",
                        q.len()
                    );
                    q
                }
            };
            let mut down = Vec::with_capacity(CTRL_COUNT);
            let mut gain = Vec::with_capacity(CTRL_COUNT);
            let mut forget = Vec::with_capacity(CTRL_COUNT);
            for (current, legacy) in names {
                let pick = |part: &str, legacy_part: &str| -> Result<QuantMatrix> {
                    let a = format!("blk.{il}.{current}_{part}.weight");
                    let b = format!("blk.{il}.{legacy}_{legacy_part}.weight");
                    let name = if loaded.has_tensor(&a) { a } else { b };
                    loaded
                        .matrix(&name)
                        .with_context(|| format!("loading {name}"))
                };
                let d = pick("down", "down")?;
                let g = pick("gain", "gain")?;
                let f = pick("forget", "forget")?;
                anyhow::ensure!(
                    d.in_dim == 2 * n_embd
                        && g.in_dim == d.out_dim
                        && f.in_dim == d.out_dim
                        && g.out_dim == n_embd
                        && f.out_dim == n_embd,
                    "layer {il}: loop controller shapes do not match n_embd {n_embd}"
                );
                down.push(d);
                gain.push(g);
                forget.push(f);
            }
            layers.push(LoopLayer {
                msg_query,
                down,
                gain,
                forget,
            });
        }
        Ok(Some(LoopCore {
            start,
            end,
            passes,
            layers,
        }))
    }

    /// How many KV/convolution slots one sequence needs: one per physical
    /// layer, plus one per core layer for every pass after the first.
    fn n_slots(&self) -> usize {
        let n = self.layers.len();
        match &self.core {
            Some(c) => n + (c.passes - 1) * (c.end - c.start),
            None => n,
        }
    }

    /// The physical layer whose weights slot `slot` runs.
    fn slot_layer(&self, slot: usize) -> usize {
        let n = self.layers.len();
        match &self.core {
            Some(c) if slot >= n => c.start + (slot - n) % (c.end - c.start),
            _ => slot,
        }
    }

    /// The attention temperature of a full-attention layer for a query at
    /// `pos`: `1 + alpha * ln(max(1, (pos + 1) / floor))`.
    fn length_scale(&self, pos: usize) -> f32 {
        let ratio = (pos as f64 + 1.0) / self.log_floor;
        if ratio > 1.0 {
            1.0 + self.log_alpha * ratio.ln() as f32
        } else {
            1.0
        }
    }

    /// `(1 - tanh(forget · g)) * cur + tanh(gain · g) * message`, with
    /// `g = down · [rms(cur), rms(message)]` — how a looped layer takes in
    /// the previous pass.
    fn control(
        &self,
        backend: &dyn Backend,
        cur: &[f32],
        message: &[f32],
        lp: &LoopLayer,
        ctrl: usize,
        n_tokens: usize,
    ) -> Vec<f32> {
        let n_embd = self.config.n_embd;
        let mut joined = vec![0f32; n_tokens * 2 * n_embd];
        joined
            .par_chunks_mut(2 * n_embd)
            .enumerate()
            .for_each(|(t, row)| {
                let (a, b) = row.split_at_mut(n_embd);
                rms_norm_plain(a, &cur[t * n_embd..(t + 1) * n_embd]);
                rms_norm_plain(b, &message[t * n_embd..(t + 1) * n_embd]);
            });
        let g = backend.matmul(&joined, n_tokens, &lp.down[ctrl]);
        let mut gf = backend.matmul_batch(&[
            MatmulOp {
                x: &g,
                n_tokens,
                w: &lp.gain[ctrl],
            },
            MatmulOp {
                x: &g,
                n_tokens,
                w: &lp.forget[ctrl],
            },
        ]);
        let forget = gf.pop().unwrap();
        let gain = gf.pop().unwrap();
        cur.iter()
            .zip(message)
            .zip(gain.iter().zip(&forget))
            .map(|((&c, &m), (&g, &f))| c - f.tanh() * c + g.tanh() * m)
            .collect()
    }

    /// One layer slot: attention then the experts, over the pass's tokens.
    #[allow(clippy::too_many_arguments)]
    fn layer(
        &self,
        backend: &dyn Backend,
        seqs: &mut Seqs<'_, '_>,
        slot: usize,
        ip: usize,
        cur: &mut Vec<f32>,
        blocks: &mut Vec<Vec<f32>>,
        messages: Option<(&[Vec<f32>], &LoopLayer)>,
        n_tokens: usize,
    ) {
        let cfg = &self.config;
        let n_embd = cfg.n_embd;
        let eps = cfg.rms_eps;
        let layer = &self.layers[ip];

        // The three attention inputs: q, k and v.
        let mut inputs: Vec<Vec<f32>> = match &layer.attn_res_query {
            Some(query) => {
                let mixed = mix(blocks, cur, query, n_embd, n_tokens);
                if layer.is_swa {
                    vec![mixed[0].clone(), mixed[0].clone(), mixed[0].clone()]
                } else {
                    mixed
                }
            }
            None => vec![cur.clone(), cur.clone(), cur.clone()],
        };
        if let Some((msgs, lp)) = messages {
            for (input, ctrl) in inputs.iter_mut().zip([CTRL_Q, CTRL_K, CTRL_V]) {
                *input = self.control(backend, input, &msgs[ctrl], lp, ctrl, n_tokens);
            }
        }
        let shared_input = layer.is_swa && messages.is_none();
        match (&layer.attn_norm, &layer.q_inp_norm) {
            (Some(norm), _) => {
                for input in inputs.iter_mut().take(if shared_input { 1 } else { 3 }) {
                    tensor::rmsnorm_inplace(input, norm, n_tokens, n_embd, eps);
                }
                if shared_input {
                    inputs.truncate(1);
                }
            }
            (None, Some(q_norm)) => {
                let norms = [
                    q_norm,
                    layer.k_inp_norm.as_ref().unwrap(),
                    layer.v_inp_norm.as_ref().unwrap(),
                ];
                for (input, norm) in inputs.iter_mut().zip(norms) {
                    tensor::rmsnorm_inplace(input, norm, n_tokens, n_embd, eps);
                }
            }
            (None, None) => unreachable!("every layer has its input norms"),
        }
        let (in_q, in_k, in_v) = if inputs.len() == 1 {
            (&inputs[0], &inputs[0], &inputs[0])
        } else {
            (&inputs[0], &inputs[1], &inputs[2])
        };

        let attn = self.attention(backend, seqs, layer, slot, in_q, in_k, in_v, n_tokens);

        let inp_sa = std::mem::take(cur);
        if ip.is_multiple_of(self.block_size) {
            // The first layer of a block starts a new residual stream; the
            // stream it ends joins the mixed blocks.
            blocks.push(normalized(&inp_sa, n_embd));
            *cur = attn;
        } else {
            *cur = inp_sa;
            tensor::add_inplace(cur, &attn);
        }

        let mut x = mix(blocks, cur, &layer.ffn_res_query, n_embd, n_tokens)
            .pop()
            .unwrap();
        if let Some((msgs, lp)) = messages {
            x = self.control(backend, &x, &msgs[CTRL_FFN], lp, CTRL_FFN, n_tokens);
        }
        tensor::rmsnorm_inplace(&mut x, &layer.ffn_norm, n_tokens, n_embd, eps);
        let mut out = self.moe(backend, layer, &x, n_tokens);
        self.convolve(
            seqs,
            slot * CONV_PER_SLOT + CONV_FFN,
            &mut out,
            n_tokens,
            n_embd,
            &layer.conv_ffn,
            None,
        );
        tensor::add_inplace(cur, &out);
    }

    /// One layer's attention, through the output projection and its short
    /// convolution — `[n_tokens, n_embd]`.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        backend: &dyn Backend,
        seqs: &mut Seqs<'_, '_>,
        layer: &Inkling2Layer,
        slot: usize,
        in_q: &[f32],
        in_k: &[f32],
        in_v: &[f32],
        n_tokens: usize,
    ) -> Vec<f32> {
        let cfg = &self.config;
        let n_embd = cfg.n_embd;
        let head_dim = cfg.head_dim;
        let n_head = layer.n_head;
        let n_head_kv = layer.n_head_kv;
        let q_dim = n_head * head_dim;
        let kv_dim = n_head_kv * head_dim;
        let n_rel = self.n_rel;

        let mut ops = vec![
            MatmulOp {
                x: in_q,
                n_tokens,
                w: &layer.wq,
            },
            MatmulOp {
                x: in_k,
                n_tokens,
                w: &layer.wk,
            },
            MatmulOp {
                x: in_v,
                n_tokens,
                w: &layer.wv,
            },
            MatmulOp {
                x: in_q,
                n_tokens,
                w: &layer.wrel,
            },
            MatmulOp {
                x: in_q,
                n_tokens,
                w: &layer.wgate,
            },
            MatmulOp {
                x: in_q,
                n_tokens,
                w: &layer.wsink,
            },
        ];
        if let (Some(kp), Some(vp)) = (&layer.k_conv_proj, &layer.v_conv_proj) {
            ops.push(MatmulOp {
                x: in_k,
                n_tokens,
                w: kp,
            });
            ops.push(MatmulOp {
                x: in_v,
                n_tokens,
                w: vp,
            });
        }
        let mut outs = backend.matmul_batch(&ops).into_iter();
        let mut q = outs.next().unwrap();
        let mut k = outs.next().unwrap();
        let mut v = outs.next().unwrap();
        let mut r = outs.next().unwrap();
        let gate = outs.next().unwrap();
        let sinks = outs.next().unwrap();
        let k_dyn = outs.next();
        let v_dyn = outs.next();

        let base = slot * CONV_PER_SLOT;
        self.convolve(
            seqs,
            base + CONV_K,
            &mut k,
            n_tokens,
            kv_dim,
            &layer.conv_k,
            k_dyn.as_deref().map(|d| (d, n_head_kv)),
        );
        self.convolve(
            seqs,
            base + CONV_V,
            &mut v,
            n_tokens,
            kv_dim,
            &layer.conv_v,
            v_dyn.as_deref().map(|d| (d, n_head_kv)),
        );

        let positions: Vec<usize> = (0..n_tokens).map(|t| seqs.pos(t)).collect();
        // Queries and bias coefficients: per-channel scale, clip, and the
        // length temperature on a full-attention layer.
        q.par_chunks_mut(q_dim)
            .zip(r.par_chunks_mut(n_head * n_rel))
            .enumerate()
            .for_each(|(t, (qt, rt))| {
                let tau = if layer.is_swa {
                    1.0
                } else {
                    self.length_scale(positions[t])
                };
                for (h, row) in qt.chunks_exact_mut(head_dim).enumerate() {
                    tensor::mul_inplace(row, &layer.q_clip[h * head_dim..(h + 1) * head_dim]);
                    clip_l2(row, self.qk_clip, tau);
                }
                for (h, row) in rt.chunks_exact_mut(n_rel).enumerate() {
                    tensor::mul_inplace(row, &layer.rel_clip[h * n_rel..(h + 1) * n_rel]);
                    clip_l2(row, self.rel_clip, tau);
                }
            });
        k.par_chunks_mut(head_dim)
            .for_each(|row| clip_l2(row, self.qk_clip, 1.0));
        v.par_chunks_mut(head_dim)
            .for_each(|row| rms_norm_plain_inplace(row, cfg.rms_eps));

        for t in 0..n_tokens {
            seqs.cache(t).layers[slot].push(
                &k[t * kv_dim..(t + 1) * kv_dim],
                &v[t * kv_dim..(t + 1) * kv_dim],
            );
        }
        let layer_caches: Vec<&crate::engine::kv_cache::LayerCache> = match &*seqs {
            Seqs::One { cache, .. } => vec![&cache.layers[slot]; n_tokens],
            Seqs::Rows(rows) => rows.iter().map(|row| &row.cache.layers[slot]).collect(),
            Seqs::Runs { rows, owner } => owner
                .iter()
                .map(|&(row, _)| &rows[row].cache.layers[slot])
                .collect(),
        };

        let group = n_head / n_head_kv;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let extent = layer.extent;

        let mut ctx = vec![0f32; n_tokens * q_dim];
        ctx.par_chunks_mut(head_dim).enumerate().for_each_init(
            || (Vec::<f32>::new(), Vec::<f32>::new()),
            |(bias, scores), (index, out)| {
                let t = index / n_head;
                let h = index % n_head;
                let qpos = positions[t];
                let cache = layer_caches[t];
                let first = if layer.is_swa {
                    qpos.saturating_sub(self.n_swa - 1)
                } else {
                    0
                };
                let n_keys = qpos - first + 1;

                // This (token, head)'s bias per distance, out to the
                // distances this query reaches; zero past the bank.
                let width = extent.min(n_keys);
                let coeffs = &r[(t * n_head + h) * n_rel..(t * n_head + h + 1) * n_rel];
                bias.clear();
                bias.extend((0..width).map(|d| {
                    tensor::dot(&layer.rel_bias[d * n_rel..(d + 1) * n_rel], coeffs)
                }));

                let qv = &q[t * q_dim + h * head_dim..t * q_dim + (h + 1) * head_dim];
                let kv_head = h / group;
                scores.clear();
                scores.extend((first..=qpos).map(|p| {
                    let dist = qpos - p;
                    let b = if dist < width { bias[dist] } else { 0.0 };
                    tensor::dot(qv, cache.key_at(p, kv_head, head_dim)) * scale + b
                }));
                // The sink takes part in the normalization and adds no
                // value.
                let sink = sinks[t * n_head + h];
                let max = scores.iter().copied().fold(sink, f32::max);
                let mut denom = (sink - max).exp();
                for s in scores.iter_mut() {
                    *s = (*s - max).exp();
                    denom += *s;
                }
                let inv = 1.0 / denom;
                for (i, &w) in scores.iter().enumerate() {
                    tensor::axpy_inplace(
                        out,
                        cache.value_at(first + i, kv_head, head_dim),
                        w * inv,
                    );
                }
            },
        );
        ctx.par_iter_mut()
            .zip(gate.par_iter())
            .for_each(|(c, &g)| *c *= tensor::sigmoid(g));

        let mut out = backend.matmul(&ctx, n_tokens, &layer.wo);
        self.convolve(
            seqs,
            base + CONV_ATTN,
            &mut out,
            n_tokens,
            n_embd,
            &layer.conv_attn,
            None,
        );
        out
    }

    /// A causal short convolution with its residual, in place over `x`
    /// (`[n_tokens, channels]`), each token through its own sequence's
    /// window `index`: `(x + conv(x) [+ dynamic(x)]) / sqrt(taps [* 2] + 1)`.
    ///
    /// `dynamic`, `([n_tokens, n_heads * taps], n_heads)`, adds per-head,
    /// per-token taps: `taps[h * T + lag]` weights the input `lag` steps
    /// back of every channel in head `h`.
    #[allow(clippy::too_many_arguments)]
    fn convolve(
        &self,
        seqs: &mut Seqs<'_, '_>,
        index: usize,
        x: &mut [f32],
        n_tokens: usize,
        channels: usize,
        kernel: &[f32],
        dynamic: Option<(&[f32], usize)>,
    ) {
        let taps = self.conv_taps;
        let hist_w = taps - 1;
        let factor = if dynamic.is_some() { 2 } else { 1 };
        let scale = 1.0 / ((factor * taps + 1) as f32).sqrt();
        for t in 0..n_tokens {
            let state = &mut seqs.cache(t).recurrent[index];
            let row = &mut x[t * channels..(t + 1) * channels];
            let mut extra = vec![0f32; channels];
            if let Some((dyn_taps, n_heads)) = dynamic {
                let width = channels / n_heads;
                let dt = &dyn_taps[t * n_heads * taps..(t + 1) * n_heads * taps];
                let hist = state.conv_window();
                for c in 0..channels {
                    let tap = &dt[(c / width) * taps..(c / width + 1) * taps];
                    let past = &hist[c * hist_w..(c + 1) * hist_w];
                    let mut sum = tap[0] * row[c];
                    for lag in 1..taps {
                        sum += tap[lag] * past[hist_w - lag];
                    }
                    extra[c] = sum;
                }
            }
            let conv = state.conv_step(row, kernel);
            for ((v, c), e) in row.iter_mut().zip(conv).zip(extra) {
                *v = (*v + c + e) * scale;
            }
        }
    }

    /// The routed SiTU experts over `normed` (`[n_tokens, n_embd]`).
    fn moe(
        &self,
        backend: &dyn Backend,
        layer: &Inkling2Layer,
        normed: &[f32],
        n_tokens: usize,
    ) -> Vec<f32> {
        let n_embd = self.config.n_embd;
        let n_ff = layer.down_exps.in_dim;
        let mut recorder =
            moe_stats::LayerRecorder::for_tensors(&[&layer.gate_up_exps, &layer.down_exps]);

        let logits = super::moe_router_logits(backend, normed, n_tokens, &layer.gate_inp);
        let n_logit = self.n_expert;
        let mut selection: Vec<Vec<(usize, f32)>> = (0..n_tokens)
            .map(|t| {
                let probs: Vec<f32> = logits[t * n_logit..(t + 1) * n_logit]
                    .iter()
                    .map(|&l| tensor::sigmoid(l))
                    .collect();
                let mut choice = probs.clone();
                if let Some(b) = &layer.exp_probs_b {
                    tensor::add_inplace(&mut choice, b);
                }
                let selected = super::top_k_indices(&choice, self.n_expert_used);
                let sum: f32 = selected.iter().map(|&e| probs[e]).sum();
                let factor = self.route_scale / (sum + ROUTE_NORM_EPS);
                selected
                    .into_iter()
                    .map(|e| (e, probs[e] * factor))
                    .collect()
            })
            .collect();
        super::apply_expert_budget(&mut selection, &layer.gate_up_exps);
        crate::engine::expert_store::read_ahead_selected(
            &selection,
            &[&layer.gate_up_exps, &layer.down_exps],
        );
        for picks in &selection {
            picks.iter().for_each(|&(e, _)| recorder.select(e));
        }

        let (beta, lb) = (self.situ_beta, self.situ_linear_beta);
        let situ = |gate: &[f32], up: &[f32]| -> Vec<f32> {
            gate.iter()
                .zip(up)
                .map(|(&g, &u)| situ(g, u, beta, lb))
                .collect()
        };
        let gu_scale = layer.gate_up_scale.as_deref();
        let down_scale = layer.down_scale.as_deref();
        let contribs = if (super::gpu_experts() || super::expert_gemm_wide(selection.len()))
            && backend.as_wgpu().is_some()
        {
            super::evaluate_routed_experts_batched_views(
                backend,
                &selection,
                normed,
                n_embd,
                Some(&ExpertProjection {
                    exps: &layer.gate_up_exps,
                    first_row: 0,
                    n_rows: n_ff,
                    scale: gu_scale,
                }),
                &ExpertProjection {
                    exps: &layer.gate_up_exps,
                    first_row: n_ff,
                    n_rows: n_ff,
                    scale: gu_scale,
                },
                &ExpertProjection {
                    scale: down_scale,
                    ..ExpertProjection::whole(&layer.down_exps)
                },
                None,
                None,
                situ,
            )
        } else {
            super::evaluate_routed_experts(&selection, |expert, members| {
                let inputs: Vec<&[f32]> = members
                    .iter()
                    .map(|&(t, _)| &normed[t * n_embd..(t + 1) * n_embd])
                    .collect();
                let both = super::project_expert(
                    backend,
                    &layer.gate_up_exps,
                    expert,
                    0,
                    2 * n_ff,
                    &inputs,
                );
                let s = gu_scale.map_or(1.0, |s| s[expert]);
                let hidden: Vec<Vec<f32>> = both
                    .into_iter()
                    .map(|mut gu| {
                        if s != 1.0 {
                            gu.iter_mut().for_each(|v| *v *= s);
                        }
                        let (g, u) = gu.split_at(n_ff);
                        situ(g, u)
                    })
                    .collect();
                let hidden_refs: Vec<&[f32]> = hidden.iter().map(Vec::as_slice).collect();
                let ds = down_scale.map_or(1.0, |s| s[expert]);
                super::project_expert(
                    backend,
                    &layer.down_exps,
                    expert,
                    0,
                    layer.down_exps.out_dim,
                    &hidden_refs,
                )
                .into_iter()
                .zip(members)
                .map(|(mut contribution, &(_, weight))| {
                    let w = weight * ds;
                    contribution.iter_mut().for_each(|v| *v *= w);
                    contribution
                })
                .collect()
            })
        };
        recorder.loaded_once_per_distinct_expert();

        let mut out = vec![0f32; n_tokens * n_embd];
        for (t, parts) in contribs.iter().enumerate() {
            let dst = &mut out[t * n_embd..(t + 1) * n_embd];
            for part in parts {
                tensor::add_inplace(dst, part);
            }
        }
        recorder.commit(n_tokens);
        out
    }

    /// Every layer slot over `tokens`, then the output's depth mixing and
    /// norm — `[n_tokens, n_embd]`, ready for the vocabulary projection.
    fn run_layers(&self, seqs: &mut Seqs<'_, '_>, tokens: &[u32]) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let n_tokens = tokens.len();
        let n_embd = cfg.n_embd;
        let expected = self.n_slots() * CONV_PER_SLOT;
        for t in 0..n_tokens {
            let held = seqs.cache(t).recurrent.len();
            anyhow::ensure!(
                held == expected,
                "this cache has {held} short-convolution states, expected {expected}"
            );
        }

        let mut cur = vec![0f32; n_tokens * n_embd];
        for (t, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            anyhow::ensure!(tok < cfg.n_vocab, "token id {tok} is out of vocab range");
            cur[t * n_embd..(t + 1) * n_embd].copy_from_slice(&self.tok_embeddings.row(tok));
        }
        tensor::rmsnorm_inplace(&mut cur, &self.tok_embd_norm, n_tokens, n_embd, cfg.rms_eps);

        let decode = crate::engine::backend::DecodeKernels::for_batch(
            self.backend.as_ref(),
            seqs.is_batch(),
        );
        let backend: &dyn Backend = &decode;
        let mut blocks: Vec<Vec<f32>> = Vec::new();
        let n_phys = self.layers.len();

        match &self.core {
            None => {
                for il in 0..n_phys {
                    self.layer(backend, seqs, il, il, &mut cur, &mut blocks, None, n_tokens);
                }
            }
            Some(core) => {
                for il in 0..core.start {
                    self.layer(backend, seqs, il, il, &mut cur, &mut blocks, None, n_tokens);
                }
                // Every pass starts from the prelude's output.
                let prelude = cur.clone();
                let prelude_blocks = blocks.clone();
                let n_core = core.end - core.start;
                for pass in 0..core.passes {
                    // The previous pass's bank: its blocks after the first
                    // core block, and its output.
                    let mut bank: Vec<Vec<f32>> = Vec::new();
                    let mut bank_out: Vec<f32> = Vec::new();
                    if pass > 0 {
                        bank = blocks.split_off(prelude_blocks.len() + 1);
                        bank_out = std::mem::replace(&mut cur, prelude.clone());
                        blocks = prelude_blocks.clone();
                    }
                    for ip in core.start..core.end {
                        if pass == 0 {
                            self.layer(backend, seqs, ip, ip, &mut cur, &mut blocks, None, n_tokens);
                            continue;
                        }
                        let slot = n_phys + (pass - 1) * n_core + (ip - core.start);
                        let lp = &core.layers[ip - core.start];
                        let msgs = mix(&bank, &bank_out, &lp.msg_query, n_embd, n_tokens);
                        self.layer(
                            backend,
                            seqs,
                            slot,
                            ip,
                            &mut cur,
                            &mut blocks,
                            Some((&msgs, lp)),
                            n_tokens,
                        );
                    }
                }
                for il in core.end..n_phys {
                    self.layer(backend, seqs, il, il, &mut cur, &mut blocks, None, n_tokens);
                }
            }
        }

        let mut out = mix(&blocks, &cur, &self.output_res_query, n_embd, n_tokens)
            .pop()
            .unwrap();
        tensor::rmsnorm_inplace(&mut out, &self.output_norm, n_tokens, n_embd, cfg.rms_eps);
        Ok(out)
    }

    /// The vocabulary projection, softcap and padding mask over `n_rows`
    /// final hidden states.
    fn project_logits(&self, backend: &dyn Backend, hidden: &[f32], n_rows: usize) -> Vec<f32> {
        let mut logits = backend.matmul(hidden, n_rows, &self.output_weight);
        if self.softcap > 0.0 {
            tensor::softcap_inplace(&mut logits, self.softcap);
        }
        let n_vocab = logits.len() / n_rows.max(1);
        if self.unpadded_vocab < n_vocab {
            for row in logits.chunks_exact_mut(n_vocab) {
                row[self.unpadded_vocab..].fill(f32::NEG_INFINITY);
            }
        }
        logits
    }
}

/// `x / sqrt(mean(x²) + 1e-6)`, out of place.
fn rms_norm_plain(dst: &mut [f32], src: &[f32]) {
    let mean_sq = src.iter().map(|v| v * v).sum::<f32>() / src.len() as f32;
    let s = 1.0 / (mean_sq + MIX_EPS).sqrt();
    for (d, &v) in dst.iter_mut().zip(src) {
        *d = v * s;
    }
}

/// A weightless RMS norm in place, with the given epsilon.
fn rms_norm_plain_inplace(x: &mut [f32], eps: f32) {
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let s = 1.0 / (mean_sq + eps).sqrt();
    x.iter_mut().for_each(|v| *v *= s);
}

/// Every row of `x` (`[n_tokens, n_embd]`) RMS-normed without a weight.
fn normalized(x: &[f32], n_embd: usize) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    out.par_chunks_mut(n_embd)
        .zip(x.par_chunks(n_embd))
        .for_each(|(d, s)| rms_norm_plain(d, s));
    out
}

/// Scales `row` down to an L2 norm of at most `threshold`, then by `tau`.
/// Upstream adds `1e-6` to the norm before dividing.
fn clip_l2(row: &mut [f32], threshold: f32, tau: f32) {
    let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt() + 1e-6;
    let factor = (threshold / norm).clamp(0.0, 1.0) * tau;
    if factor != 1.0 {
        row.iter_mut().for_each(|v| *v *= factor);
    }
}

/// `beta * tanh(g / beta) * sigmoid(g) * lb * tanh(u / lb)`.
fn situ(g: f32, u: f32, beta: f32, lb: f32) -> f32 {
    beta * (g / beta).tanh() * tensor::sigmoid(g) * lb * (u / lb).tanh()
}

/// Attention over depth: for each token, the already-normed `blocks` and
/// the RMS-normed `cur` are scored against each `n_embd`-wide row of
/// `queries` (scaled by `1 / sqrt(n_embd)`), softmaxed, and summed —
/// one `[n_tokens, n_embd]` output per query row. The values are the
/// normed states, as the scores are.
fn mix(
    blocks: &[Vec<f32>],
    cur: &[f32],
    queries: &[f32],
    n_embd: usize,
    n_tokens: usize,
) -> Vec<Vec<f32>> {
    let n_q = queries.len() / n_embd;
    let n_states = blocks.len() + 1;
    let scale = 1.0 / (n_embd as f32).sqrt();
    // `[n_tokens][n_q * n_embd]`, transposed into per-query outputs below.
    let mut joined = vec![0f32; n_tokens * n_q * n_embd];
    joined
        .par_chunks_mut(n_q * n_embd)
        .enumerate()
        .for_each_init(
            || (vec![0f32; n_embd], vec![0f32; n_states]),
            |(normed_cur, scores), (t, out)| {
                rms_norm_plain(normed_cur, &cur[t * n_embd..(t + 1) * n_embd]);
                let state = |j: usize| -> &[f32] {
                    if j < blocks.len() {
                        &blocks[j][t * n_embd..(t + 1) * n_embd]
                    } else {
                        normed_cur
                    }
                };
                for (qi, dst) in out.chunks_exact_mut(n_embd).enumerate() {
                    let query = &queries[qi * n_embd..(qi + 1) * n_embd];
                    for (j, s) in scores.iter_mut().enumerate() {
                        *s = tensor::dot(query, state(j)) * scale;
                    }
                    tensor::softmax_inplace(scores);
                    for (j, &w) in scores.iter().enumerate() {
                        tensor::axpy_inplace(dst, state(j), w);
                    }
                }
            },
        );
    (0..n_q)
        .map(|qi| {
            let mut v = vec![0f32; n_tokens * n_embd];
            v.par_chunks_mut(n_embd).enumerate().for_each(|(t, dst)| {
                let src = (t * n_q + qi) * n_embd;
                dst.copy_from_slice(&joined[src..src + n_embd]);
            });
            v
        })
        .collect()
}

impl ModelForward for Inkling2Model {
    fn vulkan_backend(&self) -> Option<&crate::engine::backend::vulkan::VulkanBackend> {
        self.backend.as_wgpu()
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn new_kv_cache(&self, capacity: usize) -> KvCache {
        let head_dim = self.config.head_dim;
        let n_embd = self.config.n_embd;
        let slots: Vec<&Inkling2Layer> = (0..self.n_slots())
            .map(|s| &self.layers[self.slot_layer(s)])
            .collect();
        let kv_dims: Vec<usize> = slots.iter().map(|l| l.n_head_kv * head_dim).collect();
        let recurrent: Vec<RecurrentSpec> = slots
            .iter()
            .flat_map(|l| {
                let kv = l.n_head_kv * head_dim;
                [kv, kv, n_embd, n_embd]
                    .map(|channels| RecurrentSpec::delta_net(channels, self.conv_taps, 0, 0))
            })
            .collect();
        KvCache::new_mixed(capacity, &kv_dims, &recurrent)
    }

    fn forward(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        _slot_id: usize,
    ) -> Result<Vec<f32>> {
        anyhow::ensure!(!tokens.is_empty(), "forward called with no tokens");
        let n_embd = self.config.n_embd;
        let x = self.run_layers(&mut Seqs::One { cache, start_pos }, tokens)?;
        Ok(self.project_logits(
            self.backend.as_ref(),
            &x[(tokens.len() - 1) * n_embd..],
            1,
        ))
    }

    fn batches_decode(&self) -> bool {
        true
    }

    fn forward_decode_batch(
        &self,
        rows: &mut [super::DecodeRow<'_>],
        tokens: &[u32],
    ) -> Result<Option<Vec<Vec<f32>>>> {
        let n_rows = rows.len();
        if n_rows < 2 || tokens.len() != n_rows {
            return Ok(None);
        }
        let x = self.run_layers(&mut Seqs::Rows(rows), tokens)?;
        let backend = crate::engine::backend::DecodeKernels::for_batch(self.backend.as_ref(), true);
        let flat = self.project_logits(&backend, &x, n_rows);
        let n_vocab = flat.len() / n_rows;
        Ok(Some(
            flat.chunks_exact(n_vocab).map(<[f32]>::to_vec).collect(),
        ))
    }

    fn forward_hidden_states(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let mut cache = self.new_kv_cache(tokens.len().max(1));
        self.run_layers(
            &mut Seqs::One {
                cache: &mut cache,
                start_pos: 0,
            },
            tokens,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The depth mixing of a single state is that state, normed: the
    /// softmax over one score is one.
    #[test]
    fn mixing_one_state_returns_it_normed() {
        let cur = [3.0f32, -4.0, 0.0, 0.0];
        let out = mix(&[], &cur, &[1.0, 2.0, 3.0, 4.0], 4, 1);
        let mut expected = [0f32; 4];
        rms_norm_plain(&mut expected, &cur);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], expected);
    }

    /// Two states mix by the softmax of their scores, each query row on
    /// its own — written out by hand.
    #[test]
    fn mixing_weights_states_by_their_scores() {
        let n = 4;
        let a = [1.0f32, 1.0, 1.0, 1.0]; // already unit RMS
        let cur = [2.0f32, -2.0, 2.0, -2.0]; // normed: [1, -1, 1, -1]
        let queries = [2.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let out = mix(&[a.to_vec()], &cur, &queries, n, 1);
        let (sa, sc) = (2.0f32 / 2.0, 2.0f32 / 2.0);
        let wa = sa.exp() / (sa.exp() + sc.exp());
        let expected: Vec<f32> = a
            .iter()
            .zip([1.0f32, -1.0, 1.0, -1.0])
            .map(|(&x, y)| wa * x + (1.0 - wa) * y)
            .collect();
        for (o, e) in out[0].iter().zip(&expected) {
            assert!((o - e).abs() < 1e-5, "{:?} vs {expected:?}", out[0]);
        }
        // The zero query weights both states equally.
        for (o, e) in out[1].iter().zip([1.0f32, 0.0, 1.0, 0.0]) {
            assert!((o - e).abs() < 1e-5, "{:?}", out[1]);
        }
    }

    #[test]
    fn clipping_shrinks_only_long_rows() {
        let mut long = [3.0f32, 4.0];
        clip_l2(&mut long, 1.0, 1.0);
        assert!((long[0] - 0.6).abs() < 1e-5 && (long[1] - 0.8).abs() < 1e-5);
        let mut short = [0.3f32, 0.4];
        clip_l2(&mut short, 1.0, 2.0);
        assert!((short[0] - 0.6).abs() < 1e-5 && (short[1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn situ_saturates_both_branches() {
        let v = situ(1e6, 1e6, 4.0, 25.0);
        assert!((v - 100.0).abs() < 1e-3, "{v}");
        assert_eq!(situ(0.0, 5.0, 4.0, 25.0), 0.0);
        let g = 0.5f32;
        let u = -1.5f32;
        let expected = 4.0 * (g / 4.0).tanh() * (1.0 / (1.0 + (-g).exp())) * 25.0 * (u / 25.0).tanh();
        assert!((situ(g, u, 4.0, 25.0) - expected).abs() < 1e-6);
    }
}

#[cfg(test)]
mod real_model_tests {
    use super::*;

    /// End-to-end against a real `unsloth/Inkling-2-Mini-GGUF` file: the
    /// predicted token, not logits within a tolerance — see
    /// `engine::arch::inkling`'s equivalent test for why.
    ///
    /// Run with `ORANGU_TEST_INKLING2_MODEL=/path/to/Inkling-2-Mini-UD-Q4_K_M.gguf
    /// cargo test --release --bin orangu-server inkling2::real_model_tests --
    /// --ignored`.
    #[test]
    #[ignore]
    fn inkling2_predicts_paris_after_capital_of_france() {
        let path =
            std::env::var("ORANGU_TEST_INKLING2_MODEL").expect("set ORANGU_TEST_INKLING2_MODEL");
        let gguf = crate::gguf::GgufFile::open(std::path::Path::new(&path)).expect("open gguf");
        let tokenizer =
            crate::engine::tokenizer::Tokenizer::from_gguf(&gguf).expect("build tokenizer");
        let loaded = LoadedModel::open(std::path::Path::new(&path)).expect("load model");
        let model =
            Inkling2Model::load_with_backend(&loaded, Arc::new(crate::engine::backend::CpuBackend))
                .expect("build model");

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
