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

//! DeepSeek-V4.1 (`general.architecture = "deepseek41"`), e.g.
//! `unsloth/DeepSeek-V4.1-Flash-1017-GGUF`.
//!
//! The DeepSeek-V4 block of `engine::arch::deepseek4` — four hyper-connected
//! residual streams, one shared key/value vector per token, a raw sliding
//! window plus compressed keys, the sqrt-softplus mixture of experts — with
//! five changes, each of which alters the numbers rather than the speed:
//!
//! - **Single-pass hyper-connections.** Every sub-layer still predicts its
//!   three mixes from the streams it receives, but collapses the streams with
//!   the *previous* sub-layer's in-mix: the first attention reads stream 0,
//!   each FFN collapses with its own layer's attention in-mix, each attention
//!   with the previous layer's FFN in-mix, and the final collapse uses the
//!   last FFN's. There is no `output_hc` head.
//! - **Shared compressed keys.** Only the layers named in
//!   `attention.kv_source_layers` compress; every later compressed layer up to
//!   the next source reads that source's blocks. The ratios are 2 (two-token
//!   blocks) and 1 (one latent per token), never overlapping, with no
//!   positional bias on the scores. A block's row is its normed latent
//!   followed by the indexer's key, which is projected *from that latent*
//!   rather than from a compressor of its own.
//! - **Shared top-k.** Only the layers in `attention.indexer.source_layers`
//!   score the compressed keys; every later layer up to the next source
//!   attends the same positions. No query head is RMS-normalized.
//! - **Candidate blocks.** Past `indexer.candidate_source_layer`, an index
//!   source scores only inside the `candidate_top_k_blocks` blocks of
//!   `candidate_block_size` positions that the candidate source ranked best
//!   (by each block's best position, the newest block always kept).
//! - **Engram.** Before each `ple.layers` layer, every token gathers one
//!   `embedding_length_per_layer_input`-wide row per hash head from one
//!   shared table, keyed by a hash of the token and its predecessors in a
//!   compressed vocabulary (`engram.token_map`). A projection of the rows
//!   gives a key per stream and one shared value, and each stream adds the
//!   value under a gate scored from its own normed state against its key.

use anyhow::{Context, Result};
use rayon::prelude::*;
use std::sync::Arc;

use super::deepseek4::{
    Moe, grouped_output_projection, moe_ffn, raw_window, rope_attn_factor, visible_blocks,
};
use super::hyper::{HyperConnection, HyperShape};
use super::indexer::pool_block;
use super::qwen4exp::ple_gate;
use super::{ExpertGating, ExpertRouting, ModelForward, Seqs, attend, top_k_indices};
use crate::engine::backend::Backend;
use crate::engine::kv_cache::{KvCache, LayerCache};
use crate::engine::loader::{LoadedModel, ModelConfig, QuantMatrix};
use crate::engine::tensor::{self, RopeLayout, RopeParams};

/// A compressed-key source layer's own compressor: the per-token value and
/// score projections, the norm of the pooled latent, and the indexer key
/// projected from it.
struct Compressor {
    wkv: QuantMatrix,
    /// `None` at ratio 1, whose one-member blocks need no score.
    wgate: Option<QuantMatrix>,
    norm: Vec<f32>,
    /// `[key_length, indexer.key_length]` — the indexer key from the latent.
    index_k: QuantMatrix,
    index_k_norm: Vec<f32>,
}

/// A top-k source layer's indexer query and per-head weights.
struct Indexer {
    /// `[q_lora_rank, indexer.head_count * indexer.key_length]`.
    q_b: QuantMatrix,
    /// `[n_embd, indexer.head_count]`.
    proj: QuantMatrix,
}

/// One Engram layer's projection and gate.
struct EngramLayer {
    /// Which of the table's hash modules this layer reads.
    module: usize,
    /// `[ple.n_heads * head_dim, (hc + 1) * n_embd]` — a key per stream,
    /// then the shared value.
    kv: QuantMatrix,
    /// `[hc, n_embd]` — the per-stream weight of the gate's dot product.
    gate: Vec<f32>,
}

/// The Engram hash, with no table attached: pure arithmetic over token ids.
struct EngramHash {
    ngram_size: usize,
    heads_per_ngram: usize,
    /// `(ngram_size - 1) * heads_per_ngram` columns per module.
    n_cols: usize,
    /// `[n_modules, ngram_size]`.
    multipliers: Vec<u64>,
    /// `[n_modules, n_cols]` — where each column's slice of the table starts.
    offsets: Vec<u64>,
    /// `[n_modules, n_cols]` — each column's bucket count.
    buckets: Vec<u64>,
    /// `engram.token_map`: token id to compressed id.
    token_map: Vec<u64>,
    /// The compressed id of `engram.pad_token_id`, which stands in for every
    /// predecessor past the start of the sequence or past an unmapped token.
    pad: u64,
}

impl EngramHash {
    /// The table rows module `module` gathers for one token, `[n_cols]`.
    ///
    /// `prev(s)` is the token `s` positions back, `None` before the start of
    /// the sequence. Once one shift is blocked — no predecessor, or a token
    /// the map does not cover — every longer one is too, and reads as `pad`.
    fn rows(&self, module: usize, token: u32, prev: impl Fn(usize) -> Option<u32>) -> Vec<usize> {
        let n_gram = self.ngram_size;
        let map = |t: u32| self.token_map.get(t as usize).copied();
        let mut ids = vec![self.pad; n_gram];
        let mut blocked = false;
        for (s, id) in ids.iter_mut().enumerate() {
            let c = if s == 0 {
                map(token)
            } else {
                prev(s).and_then(map)
            };
            blocked |= c.is_none();
            if !blocked {
                *id = c.expect("not blocked");
            }
        }
        let mult = &self.multipliers[module * n_gram..(module + 1) * n_gram];
        let offsets = &self.offsets[module * self.n_cols..(module + 1) * self.n_cols];
        let buckets = &self.buckets[module * self.n_cols..(module + 1) * self.n_cols];
        let mut out = vec![0usize; self.n_cols];
        let mut rolling = ids[0].wrapping_mul(mult[0]);
        for o in 1..n_gram {
            rolling ^= ids[o].wrapping_mul(mult[o]);
            for h in 0..self.heads_per_ngram {
                let col = (o - 1) * self.heads_per_ngram + h;
                out[col] = (rolling % buckets[col] + offsets[col]) as usize;
            }
        }
        out
    }
}

/// The shared Engram table and its hash.
struct Engram {
    /// `[head_dim, rows]`, every module's rows back to back. Read one row at
    /// a time and never resident: only `n_cols` rows per token are touched.
    table: QuantMatrix,
    head_dim: usize,
    hash: EngramHash,
}

/// Two-level selection: which index sources narrow their scores to the
/// blocks the candidate source ranked best.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Candidates {
    source: usize,
    top_k_blocks: usize,
    block_size: usize,
}

/// Where one layer's slices of the positional KV cache live — indices into
/// `KvCache::layers`.
struct CacheSlots {
    /// Per token: the shared key/value vector.
    raw: usize,
    /// Per token, source layers only: the compressor's value (`k`) and
    /// score (`v`).
    comp_state: Option<usize>,
    /// Per completed block (stride `ratio`), source layers only: the
    /// latent followed by the indexer key.
    comp_blocks: Option<usize>,
}

struct Deepseek41Layer {
    attn_norm: Vec<f32>,
    attn_sinks: Vec<f32>,
    wq_a: QuantMatrix,
    attn_q_a_norm: Vec<f32>,
    wq_b: QuantMatrix,
    wkv: QuantMatrix,
    attn_kv_norm: Vec<f32>,
    wo_a: QuantMatrix,
    wo_b: QuantMatrix,
    hc_attn: HyperConnection,
    hc_ffn: HyperConnection,
    /// `attention.compress_ratios`: 0, 1 or 2.
    ratio: usize,
    /// The layer whose compressed blocks this one attends (itself on a
    /// source); `None` at ratio 0.
    kv_source: Option<usize>,
    /// The layer whose top-k this one attends (itself on a source).
    topk_source: Option<usize>,
    compressor: Option<Compressor>,
    indexer: Option<Indexer>,
    engram: Option<EngramLayer>,
    /// The compressed-rope parameters on a compressed layer, plain unscaled
    /// RoPE at ratio 0.
    rope: RopeParams,
    slots: CacheSlots,
    ffn: Moe,
}

pub struct Deepseek41Model {
    config: ModelConfig,
    backend: Arc<dyn Backend>,
    tok_embeddings: QuantMatrix,
    output_norm: Vec<f32>,
    output_weight: QuantMatrix,
    hyper: HyperShape,
    routing: ExpertRouting,
    q_lora_rank: usize,
    o_group_count: usize,
    o_lora_rank: usize,
    indexer_n_head: usize,
    indexer_head_size: usize,
    indexer_top_k: usize,
    n_swa: usize,
    candidates: Option<Candidates>,
    engram: Option<Engram>,
    /// A one-wide per-token slot holding each position's token id, which the
    /// Engram hash reads its predecessors from. Positional like every other
    /// slot, so a truncated or reused cache carries exactly the history its
    /// keys were made from. `None` without Engram layers.
    token_slot: Option<usize>,
    /// `(kv_dim, stride)` per `KvCache::layers` slot, in slot order.
    kv_dims: Vec<(usize, usize)>,
    layers: Vec<Deepseek41Layer>,
}

/// For each layer, the most recent source at or before it among `sources`,
/// for every layer with a nonzero ratio: the `kv_source_layers` and
/// `indexer.source_layers` resolution. A source must itself compress, and a
/// layer's source must share its ratio.
fn resolve_sources(ratios: &[usize], sources: &[usize]) -> Result<Vec<Option<usize>>> {
    let mut last = None;
    let mut out = Vec::with_capacity(ratios.len());
    for (il, &ratio) in ratios.iter().enumerate() {
        if sources.contains(&il) {
            anyhow::ensure!(ratio != 0, "source layer {il} has compression ratio 0");
            last = Some(il);
        }
        if ratio == 0 {
            out.push(None);
            continue;
        }
        let src = last
            .filter(|&s| ratios[s] == ratio)
            .with_context(|| format!("layer {il}: ratio {ratio} without a matching source"))?;
        out.push(Some(src));
    }
    Ok(out)
}

/// The candidate source's block choice for one query: the best score inside
/// each `block_size`-wide block, the newest block (holding the last visible
/// position) pinned so a partly filled one is always kept, then the
/// `top_k_blocks` best blocks.
fn candidate_blocks(scores: &[f32], block_size: usize, top_k_blocks: usize) -> Vec<usize> {
    let n_blk = scores.len().div_ceil(block_size);
    let mut best: Vec<f32> = scores
        .chunks(block_size)
        .map(|b| b.iter().copied().fold(f32::NEG_INFINITY, f32::max))
        .collect();
    if let Some(newest) = best.last_mut() {
        *newest += 1e30;
    }
    top_k_indices(&best, top_k_blocks.min(n_blk))
}

/// `scores` with every position outside `blocks` (each `block_size` wide)
/// set to `-inf`, so a top-k cannot pick it.
fn restrict_to_blocks(scores: &mut [f32], blocks: &[usize], block_size: usize) {
    let mut keep = vec![false; scores.len().div_ceil(block_size)];
    for &b in blocks {
        keep[b] = true;
    }
    for (c, s) in scores.iter_mut().enumerate() {
        if !keep[c / block_size] {
            *s = f32::NEG_INFINITY;
        }
    }
}

/// The top `k` positions of `scores`, ascending, leaving out any a mask
/// sent to `-inf`.
fn select_positions(scores: &[f32], k: usize) -> Vec<usize> {
    let mut chosen: Vec<usize> = top_k_indices(scores, k)
        .into_iter()
        .filter(|&c| scores[c] > f32::NEG_INFINITY)
        .collect();
    chosen.sort_unstable();
    chosen
}

/// Per-pass state the source layers hand to the layers after them.
struct PassState {
    /// Per layer: each token's attended positions, on a top-k source.
    topk: Vec<Option<Vec<Vec<usize>>>>,
    /// Each token's candidate blocks, once the candidate source has run.
    candidates: Option<Vec<Vec<usize>>>,
}

impl Deepseek41Model {
    pub fn load_with_backend(loaded: &LoadedModel, backend: Arc<dyn Backend>) -> Result<Self> {
        let n_layer = loaded.config.n_layer;
        let head_dim = loaded.config.head_dim;
        let n_embd = loaded.config.n_embd;
        let u64_key = |key: &str| -> Result<usize> {
            Ok(loaded
                .metadata_u64(key)
                .with_context(|| format!("missing {key}"))? as usize)
        };
        let tok_embeddings = loaded
            .matrix("token_embd.weight")
            .context("loading token_embd.weight")?;
        let (output_norm, _) = loaded
            .tensor("output_norm.weight")
            .context("loading output_norm.weight")?;
        let output_weight = loaded
            .matrix("output.weight")
            .context("loading output.weight")?;
        let hyper = HyperShape::from_gguf(loaded)?;
        let n_expert_used = u64_key("expert_used_count")?;
        let q_lora_rank = u64_key("attention.q_lora_rank")?;
        let o_group_count = u64_key("attention.output_group_count")?;
        let o_lora_rank = u64_key("attention.output_lora_rank")?;
        anyhow::ensure!(
            o_group_count > 0 && loaded.config.n_head.is_multiple_of(o_group_count),
            "attention.head_count ({}) must be a multiple of attention.output_group_count ({o_group_count})",
            loaded.config.n_head
        );
        let indexer_n_head = u64_key("attention.indexer.head_count")?;
        let indexer_head_size = u64_key("attention.indexer.key_length")?;
        let indexer_top_k = u64_key("attention.indexer.top_k")?;
        anyhow::ensure!(
            indexer_top_k > 0,
            "attention.indexer.top_k must be at least 1"
        );
        let n_swa = u64_key("attention.sliding_window")?;
        anyhow::ensure!(n_swa > 0, "attention.sliding_window must be at least 1");
        let compress_rope_base = loaded
            .metadata_f32("attention.compress_rope_freq_base")
            .unwrap_or(160000.0);
        let ratios: Vec<usize> = loaded
            .metadata_array_f32("attention.compress_ratios")
            .unwrap_or_default()
            .into_iter()
            .take(n_layer)
            .map(|r| r as usize)
            .collect();
        anyhow::ensure!(
            ratios.len() == n_layer,
            "attention.compress_ratios has {} entries, fewer than the {n_layer} layers",
            ratios.len()
        );
        if let Some((il, r)) = ratios.iter().enumerate().find(|(_, r)| **r > 2) {
            anyhow::bail!("layer {il} has compression ratio {r}; only 0, 1 and 2 are defined");
        }
        let layer_list = |key: &str| -> Result<Vec<usize>> {
            let v: Vec<usize> = loaded
                .metadata_array_u64(key)
                .with_context(|| format!("missing {key}"))?
                .into_iter()
                .map(|l| l as usize)
                .collect();
            anyhow::ensure!(
                v.iter().all(|&l| l < n_layer),
                "{key} names a layer past the {n_layer} layers"
            );
            Ok(v)
        };
        let kv_sources = resolve_sources(&ratios, &layer_list("attention.kv_source_layers")?)?;
        let topk_sources =
            resolve_sources(&ratios, &layer_list("attention.indexer.source_layers")?)?;
        for il in 0..n_layer {
            if let (Some(kv), Some(tk)) = (kv_sources[il], topk_sources[il]) {
                anyhow::ensure!(
                    tk >= kv,
                    "layer {il}: top-k source {tk} is older than its key source {kv}"
                );
            }
        }
        let candidates = match loaded.metadata_u64("attention.indexer.candidate_source_layer") {
            Some(source) => {
                let source = source as usize;
                let c = Candidates {
                    source,
                    top_k_blocks: u64_key("attention.indexer.candidate_top_k_blocks")?,
                    block_size: u64_key("attention.indexer.candidate_block_size")?,
                };
                anyhow::ensure!(
                    source < n_layer
                        && topk_sources[source] == Some(source)
                        && c.top_k_blocks > 0
                        && c.block_size > 0,
                    "attention.indexer.candidate_source_layer {source} must be a top-k source, with nonzero block size and block count"
                );
                for il in source + 1..n_layer {
                    anyhow::ensure!(
                        topk_sources[il] != Some(il) || ratios[il] == ratios[source],
                        "layer {il} reads candidate blocks at a different ratio from their source"
                    );
                }
                Some(c)
            }
            None => None,
        };

        let gating_func = loaded
            .metadata_u64("expert_gating_func")
            .context("missing expert_gating_func")?;
        let expert_weights_scale = loaded.metadata_f32("expert_weights_scale").unwrap_or(1.0);
        // Renormalizing a single expert's weight would only undo the scale.
        let expert_weights_norm = loaded
            .metadata_u64("expert_weights_norm")
            .is_some_and(|v| v != 0)
            && n_expert_used > 1;
        let swiglu_clamp_exp = loaded
            .metadata_array_f32("swiglu_clamp_exp")
            .unwrap_or_default();
        let swiglu_clamp_shexp = loaded
            .metadata_array_f32("swiglu_clamp_shexp")
            .unwrap_or_else(|| swiglu_clamp_exp.clone());

        let is_yarn = loaded.metadata_string("rope.scaling.type").as_deref() == Some("yarn");
        let freq_scale = loaded
            .metadata_f32("rope.scaling.factor")
            .map_or(1.0, |f| 1.0 / f);
        let ext_factor = if is_yarn { 1.0 } else { 0.0 };
        let compress_rope = RopeParams {
            rope_dim: loaded.config.rope_dim,
            freq_base: compress_rope_base,
            freq_scale,
            ext_factor,
            beta_fast: loaded
                .metadata_f32("rope.scaling.yarn_beta_fast")
                .unwrap_or(32.0),
            beta_slow: loaded
                .metadata_f32("rope.scaling.yarn_beta_slow")
                .unwrap_or(1.0),
            attn_factor: rope_attn_factor(freq_scale, ext_factor),
            n_ctx_orig: loaded
                .metadata_u64("rope.scaling.original_context_length")
                .unwrap_or(loaded.config.n_ctx_train as u64) as usize,
            layout: RopeLayout::Norm,
        };
        let plain_rope = RopeParams {
            rope_dim: loaded.config.rope_dim,
            freq_base: loaded.config.rope_freq_base,
            freq_scale: 1.0,
            ext_factor: 0.0,
            beta_fast: 0.0,
            beta_slow: 0.0,
            attn_factor: 1.0,
            n_ctx_orig: 0,
            layout: RopeLayout::Norm,
        };

        let engram = Self::load_engram(loaded)?;
        let engram_layers: Vec<usize> = match engram {
            Some(_) => layer_list("ple.layers")?,
            None => Vec::new(),
        };

        // Slot order is fixed here and never derived again: a cache is built
        // from `kv_dims` alone.
        let mut kv_dims: Vec<(usize, usize)> = Vec::new();
        let mut push_slot = |dim: usize, stride: usize| {
            kv_dims.push((dim, stride));
            kv_dims.len() - 1
        };
        let token_slot = engram.as_ref().map(|_| push_slot(1, 1));

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
            let get_expert_matrix = |suffix: &str| {
                let name = format!("blk.{i}.{suffix}");
                loaded
                    .expert_matrix(&name)
                    .with_context(|| format!("loading {name}"))
            };
            let ratio = ratios[i];
            let compressor = if kv_sources[i] == Some(i) {
                Some(Compressor {
                    wkv: get_matrix("attn_compressor_kv.weight")?,
                    wgate: if ratio > 1 {
                        Some(get_matrix("attn_compressor_gate.weight")?)
                    } else {
                        None
                    },
                    norm: get("attn_compressor_norm.weight")?,
                    index_k: get_matrix("indexer.attn_k.weight")?,
                    index_k_norm: get("indexer.k_norm.weight")?,
                })
            } else {
                None
            };
            let indexer = if topk_sources[i] == Some(i) {
                Some(Indexer {
                    q_b: get_matrix("indexer.attn_q_b.weight")?,
                    proj: get_matrix("indexer.proj.weight")?,
                })
            } else {
                None
            };
            let engram_layer = match engram_layers.iter().position(|&l| l == i) {
                Some(module) => {
                    let kv = get_matrix("engram_kv.weight")?;
                    let gate = get("engram_gate.weight")?;
                    let hash = &engram.as_ref().expect("engram layers imply a table").hash;
                    anyhow::ensure!(
                        kv.out_dim == (hyper.count + 1) * n_embd
                            && gate.len() == hyper.count * n_embd,
                        "blk.{i}.engram_kv/engram_gate do not match {} streams of {n_embd}",
                        hyper.count
                    );
                    anyhow::ensure!(
                        module * hash.n_cols < hash.offsets.len(),
                        "blk.{i} is Engram module {module}, past the hash constants"
                    );
                    Some(EngramLayer { module, kv, gate })
                }
                None => None,
            };
            let slots = CacheSlots {
                raw: push_slot(head_dim, 1),
                comp_state: compressor.as_ref().map(|_| push_slot(head_dim, 1)),
                comp_blocks: compressor
                    .as_ref()
                    .map(|_| push_slot(head_dim + indexer_head_size, ratio)),
            };
            let layer = Deepseek41Layer {
                attn_norm: get("attn_norm.weight")?,
                attn_sinks: get("attn_sinks.weight")?,
                wq_a: get_matrix("attn_q_a.weight")?,
                attn_q_a_norm: get("attn_q_a_norm.weight")?,
                wq_b: get_matrix("attn_q_b.weight")?,
                wkv: get_matrix("attn_kv.weight")?,
                attn_kv_norm: get("attn_kv_a_norm.weight")?,
                wo_a: get_matrix("attn_output_a.weight")?,
                wo_b: get_matrix("attn_output_b.weight")?,
                hc_attn: HyperConnection::load(loaded, &format!("blk.{i}.hc_attn"))?,
                hc_ffn: HyperConnection::load(loaded, &format!("blk.{i}.hc_ffn"))?,
                ratio,
                kv_source: kv_sources[i],
                topk_source: topk_sources[i],
                compressor,
                indexer,
                engram: engram_layer,
                rope: if ratio == 0 {
                    plain_rope
                } else {
                    compress_rope
                },
                slots,
                ffn: Moe {
                    norm: get("ffn_norm.weight")?,
                    gate_inp: get_matrix("ffn_gate_inp.weight")?,
                    exp_probs_b: Some(get("exp_probs_b.bias")?),
                    tid2eid: None,
                    gate_exps: get_expert_matrix("ffn_gate_exps.weight")?,
                    down_exps: get_expert_matrix("ffn_down_exps.weight")?,
                    up_exps: get_expert_matrix("ffn_up_exps.weight")?,
                    gate_shexp: get_matrix("ffn_gate_shexp.weight")?,
                    down_shexp: get_matrix("ffn_down_shexp.weight")?,
                    up_shexp: get_matrix("ffn_up_shexp.weight")?,
                    clamp_exp: swiglu_clamp_exp.get(i).copied().unwrap_or(0.0),
                    clamp_shexp: swiglu_clamp_shexp.get(i).copied().unwrap_or(0.0),
                },
            };
            anyhow::ensure!(
                layer.wq_b.out_dim == loaded.config.n_head * head_dim,
                "layer {i}'s attn_q_b projects to {} outputs, not head_count * key_length ({})",
                layer.wq_b.out_dim,
                loaded.config.n_head * head_dim
            );
            layers.push(layer);
        }

        super::reserve_expert_region(
            backend.as_ref(),
            layers.iter().flat_map(|layer| {
                [
                    (layer.ffn.gate_exps.stack_matrix().raw_bytes().len()
                        + layer.ffn.up_exps.stack_matrix().raw_bytes().len())
                        as u64,
                    layer.ffn.down_exps.stack_matrix().raw_bytes().len() as u64,
                ]
            }),
        );
        Ok(Self {
            config: loaded.config.clone(),
            backend,
            tok_embeddings,
            output_norm,
            output_weight,
            hyper,
            routing: ExpertRouting {
                n_expert_used,
                gating: ExpertGating::from_gguf(gating_func)?,
                weights_norm: expert_weights_norm,
                weights_scale: expert_weights_scale,
                groups: super::ExpertGroups::from_gguf(
                    loaded,
                    loaded.metadata_u64("expert_count").unwrap_or(0) as usize,
                )?,
            },
            q_lora_rank,
            o_group_count,
            o_lora_rank,
            indexer_n_head,
            indexer_head_size,
            indexer_top_k,
            n_swa,
            candidates,
            engram,
            token_slot,
            kv_dims,
            layers,
        })
    }

    /// The Engram table and hash constants, `None` for a file with no
    /// `ple.layers`.
    fn load_engram(loaded: &LoadedModel) -> Result<Option<Engram>> {
        let Some(layers) = loaded.metadata_array_u64("ple.layers") else {
            return Ok(None);
        };
        if layers.is_empty() {
            return Ok(None);
        }
        let n_modules = layers.len();
        let key = |k: &str| -> Result<usize> {
            Ok(loaded
                .metadata_u64(k)
                .with_context(|| format!("missing {k}"))? as usize)
        };
        let array = |k: &str| -> Result<Vec<u64>> {
            loaded
                .metadata_array_u64(k)
                .with_context(|| format!("missing {k}"))
        };
        let ngram_size = key("ple.ngram_size")?;
        let heads_per_ngram = key("ple.heads_per_ngram")?;
        let head_dim = key("embedding_length_per_layer_input")?;
        anyhow::ensure!(
            ngram_size >= 2 && heads_per_ngram > 0 && head_dim > 0,
            "ple.ngram_size must be at least 2, ple.heads_per_ngram and embedding_length_per_layer_input nonzero"
        );
        let n_cols = (ngram_size - 1) * heads_per_ngram;
        let multipliers = array("ple.layer_multipliers")?;
        let offsets = array("ple.head_offsets")?;
        let buckets = array("ple.head_vocab_sizes")?;
        let token_map = array("engram.token_map")?;
        let pad_id = key("engram.pad_token_id")?;
        anyhow::ensure!(
            multipliers.len() == n_modules * ngram_size
                && offsets.len() == n_modules * n_cols
                && buckets.len() == n_modules * n_cols,
            "the Engram hash constants do not match {n_modules} modules of {n_cols} heads"
        );
        anyhow::ensure!(
            token_map.len() == loaded.config.n_vocab,
            "engram.token_map has {} entries for a {}-token vocabulary",
            token_map.len(),
            loaded.config.n_vocab
        );
        anyhow::ensure!(
            pad_id < token_map.len(),
            "engram.pad_token_id is out of range"
        );
        anyhow::ensure!(
            buckets.iter().all(|&b| b >= 2),
            "an Engram head has fewer than two buckets"
        );
        let table = loaded
            .matrix("per_layer_token_embd.weight")
            .context("loading per_layer_token_embd.weight")?;
        let rows_needed = offsets
            .iter()
            .zip(&buckets)
            .map(|(o, b)| o + b)
            .max()
            .unwrap_or(0);
        anyhow::ensure!(
            table.in_dim == head_dim && table.out_dim as u64 >= rows_needed,
            "per_layer_token_embd.weight is [{}, {}], the Engram heads need [{head_dim}, {rows_needed}]",
            table.in_dim,
            table.out_dim
        );
        Ok(Some(Engram {
            table,
            head_dim,
            hash: EngramHash {
                ngram_size,
                heads_per_ngram,
                n_cols,
                multipliers,
                offsets,
                buckets,
                pad: token_map[pad_id],
                token_map,
            },
        }))
    }

    /// Every layer over `tokens`, to the final norm.
    fn run_layers(&self, seqs: &mut Seqs<'_, '_>, tokens: &[u32]) -> Result<Vec<f32>> {
        let n_tokens = tokens.len();
        let n_embd = self.config.n_embd;
        let eps = self.config.rms_eps;

        let mut embeddings = vec![0f32; n_tokens * n_embd];
        for (t, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            anyhow::ensure!(
                tok < self.config.n_vocab,
                "token id {tok} is out of vocab range"
            );
            embeddings[t * n_embd..(t + 1) * n_embd].copy_from_slice(&self.tok_embeddings.row(tok));
        }
        // The pass's own token ids go in before any layer runs, so the Engram
        // hash finds every predecessor in the cache, this pass's included.
        if let Some(slot) = self.token_slot {
            for (t, &tok) in tokens.iter().enumerate() {
                let id = [tok as f32];
                seqs.cache(t).layers[slot].push(&id, &id);
            }
        }
        let mut x = self.hyper.seed(&embeddings, n_tokens);

        let decode = crate::engine::backend::DecodeKernels::for_batch(
            self.backend.as_ref(),
            seqs.is_batch(),
        );
        let backend: &dyn Backend = &decode;
        let mut pass = PassState {
            topk: vec![None; self.layers.len()],
            candidates: None,
        };
        let mut pre_prev: Option<Vec<f32>> = None;
        for (il, layer) in self.layers.iter().enumerate() {
            pre_prev = Some(self.forward_layer(
                backend,
                il,
                layer,
                seqs,
                &mut x,
                tokens,
                pre_prev.as_deref(),
                &mut pass,
            )?);
        }

        let mut out = Vec::new();
        self.hyper.collapse_with(
            &mut out,
            &x,
            pre_prev.as_deref().context("a model with no layers")?,
            n_tokens,
        );
        tensor::rmsnorm_inplace(&mut out, &self.output_norm, n_tokens, n_embd, eps);
        Ok(out)
    }

    /// One layer, returning its FFN in-mix — the next layer's attention
    /// collapses with it.
    #[allow(clippy::too_many_arguments)]
    fn forward_layer(
        &self,
        backend: &dyn Backend,
        il: usize,
        layer: &Deepseek41Layer,
        seqs: &mut Seqs<'_, '_>,
        x: &mut [f32],
        tokens: &[u32],
        pre_prev: Option<&[f32]>,
        pass: &mut PassState,
    ) -> Result<Vec<f32>> {
        let n_tokens = tokens.len();
        let n_embd = self.config.n_embd;
        let eps = self.config.rms_eps;
        let hc = self.hyper.count;

        if let (Some(e), Some(engram)) = (layer.engram.as_ref(), self.engram.as_ref()) {
            self.apply_engram(backend, engram, e, seqs, x, tokens);
        }

        let (mut flat, mut mixes) = (Vec::new(), Vec::new());
        let (mut pre_a, mut post_a, mut comb_a) = (Vec::new(), Vec::new(), Vec::new());
        self.hyper.mixes_into(
            backend,
            &layer.hc_attn,
            x,
            n_tokens,
            &mut pre_a,
            &mut post_a,
            &mut comb_a,
            &mut flat,
            &mut mixes,
        );
        let mut cur = Vec::new();
        match pre_prev {
            Some(pre) => self.hyper.collapse_with(&mut cur, x, pre, n_tokens),
            None => {
                cur.resize(n_tokens * n_embd, 0.0);
                for t in 0..n_tokens {
                    let at = t * hc * n_embd;
                    cur[t * n_embd..(t + 1) * n_embd].copy_from_slice(&x[at..at + n_embd]);
                }
            }
        }
        tensor::rmsnorm_inplace(&mut cur, &layer.attn_norm, n_tokens, n_embd, eps);
        let attn = self.attention(backend, il, layer, seqs, &cur, n_tokens, pass)?;
        let residual = x.to_vec();
        self.hyper
            .expand(x, &attn, &residual, &post_a, &comb_a, n_tokens);

        let (mut pre_f, mut post_f, mut comb_f) = (Vec::new(), Vec::new(), Vec::new());
        self.hyper.mixes_into(
            backend,
            &layer.hc_ffn,
            x,
            n_tokens,
            &mut pre_f,
            &mut post_f,
            &mut comb_f,
            &mut flat,
            &mut mixes,
        );
        self.hyper.collapse_with(&mut cur, x, &pre_a, n_tokens);
        tensor::rmsnorm_inplace(&mut cur, &layer.ffn.norm, n_tokens, n_embd, eps);
        let ffn = moe_ffn(backend, &self.routing, n_embd, &layer.ffn, &cur, tokens);
        let residual = x.to_vec();
        self.hyper
            .expand(x, &ffn, &residual, &post_f, &comb_f, n_tokens);
        Ok(pre_f)
    }

    /// Adds one Engram layer's gated value into every stream.
    fn apply_engram(
        &self,
        backend: &dyn Backend,
        engram: &Engram,
        layer: &EngramLayer,
        seqs: &mut Seqs<'_, '_>,
        x: &mut [f32],
        tokens: &[u32],
    ) {
        let n_tokens = tokens.len();
        let n_embd = self.config.n_embd;
        let hc = self.hyper.count;
        let eps = self.config.rms_eps;
        let slot = self.token_slot.expect("Engram layers imply a token slot");

        let n_cols = engram.hash.n_cols;
        let mut rows = Vec::with_capacity(n_tokens * n_cols);
        for (t, &tok) in tokens.iter().enumerate() {
            let pos = seqs.pos(t);
            let history = &seqs.cache(t).layers[slot];
            let prev = |s: usize| {
                pos.checked_sub(s)
                    .map(|p| history.key_at(p, 0, 1)[0] as u32)
            };
            rows.extend(engram.hash.rows(layer.module, tok, prev));
        }
        let width = n_cols * engram.head_dim;
        let mut embd = vec![0f32; n_tokens * width];
        embd.par_chunks_mut(width)
            .zip(rows.par_chunks(n_cols))
            .for_each(|(dst, token_rows)| {
                for (h, &row) in token_rows.iter().enumerate() {
                    dst[h * engram.head_dim..(h + 1) * engram.head_dim]
                        .copy_from_slice(&engram.table.row(row));
                }
            });

        let kv = super::matmul_host_fallback(backend, &embd, n_tokens, &layer.kv);
        let kv_width = (hc + 1) * n_embd;
        let scale = 1.0 / (n_embd as f32).sqrt();
        for t in 0..n_tokens {
            let kv_t = &kv[t * kv_width..(t + 1) * kv_width];
            let value = &kv_t[hc * n_embd..];
            for s in 0..hc {
                let key = &kv_t[s * n_embd..(s + 1) * n_embd];
                let stream = &mut x[(t * hc + s) * n_embd..(t * hc + s + 1) * n_embd];
                let gate_w = &layer.gate[s * n_embd..(s + 1) * n_embd];
                let dot: f32 = stream
                    .iter()
                    .zip(key)
                    .zip(gate_w)
                    .map(|((a, b), w)| a * b * w)
                    .sum();
                let dot = dot
                    * super::rms_norm_scale(super::row_mean_sq(stream), eps)
                    * super::rms_norm_scale(super::row_mean_sq(key), eps)
                    * scale;
                tensor::axpy_inplace(stream, value, ple_gate(dot));
            }
        }
    }

    /// One layer's attention: the shared-key projection, the source layer's
    /// compressed blocks, the top-k choice (or the source's), the masked
    /// softmax over [sliding window + chosen blocks], and the grouped output
    /// projection.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        backend: &dyn Backend,
        il: usize,
        layer: &Deepseek41Layer,
        seqs: &mut Seqs<'_, '_>,
        cur: &[f32],
        n_tokens: usize,
        pass: &mut PassState,
    ) -> Result<Vec<f32>> {
        let n_embd = self.config.n_embd;
        let eps = self.config.rms_eps;
        let head_dim = self.config.head_dim;
        let n_head = self.config.n_head;
        let rope_dim = self.config.rope_dim;
        let nope = head_dim - rope_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let idx_dim = self.indexer_head_size;
        let idx_heads = self.indexer_n_head;

        let mut qr = backend.matmul(cur, n_tokens, &layer.wq_a);
        tensor::rmsnorm_inplace(
            &mut qr,
            &layer.attn_q_a_norm,
            n_tokens,
            self.q_lora_rank,
            eps,
        );
        let mut q = backend.matmul(&qr, n_tokens, &layer.wq_b);
        let mut kv = backend.matmul(cur, n_tokens, &layer.wkv);
        tensor::rmsnorm_inplace(&mut kv, &layer.attn_kv_norm, n_tokens, head_dim, eps);
        for t in 0..n_tokens {
            let pos = seqs.pos(t);
            let q_t = &mut q[t * n_head * head_dim..(t + 1) * n_head * head_dim];
            for h in 0..n_head {
                tensor::rope_apply_params_inplace(
                    &mut q_t[h * head_dim + nope..(h + 1) * head_dim],
                    1,
                    rope_dim,
                    pos,
                    None,
                    &layer.rope,
                );
            }
            tensor::rope_apply_params_inplace(
                &mut kv[t * head_dim + nope..(t + 1) * head_dim],
                1,
                rope_dim,
                pos,
                None,
                &layer.rope,
            );
        }

        if let Some(c) = layer.compressor.as_ref() {
            self.compress(backend, layer, c, seqs, cur, n_tokens);
        }

        // An index source's query and per-head weights, for every token.
        let index_query = layer.indexer.as_ref().map(|ix| {
            let mut iq = backend.matmul(&qr, n_tokens, &ix.q_b);
            for t in 0..n_tokens {
                let pos = seqs.pos(t);
                let row = &mut iq[t * idx_heads * idx_dim..(t + 1) * idx_heads * idx_dim];
                for h in 0..idx_heads {
                    tensor::rope_apply_params_inplace(
                        &mut row[h * idx_dim + idx_dim - rope_dim..(h + 1) * idx_dim],
                        1,
                        rope_dim,
                        pos,
                        None,
                        &layer.rope,
                    );
                }
            }
            let mut weights = backend.matmul(cur, n_tokens, &ix.proj);
            let iscale = 1.0 / ((idx_dim * idx_heads) as f32).sqrt();
            weights.iter_mut().for_each(|w| *w *= iscale);
            (iq, weights)
        });
        let blocks_slot = layer
            .kv_source
            .map(|src| {
                self.layers[src]
                    .slots
                    .comp_blocks
                    .context("a key source keeps compressed blocks")
            })
            .transpose()?;
        let is_topk_source = layer.topk_source == Some(il);
        let is_candidate_source = self.candidates.is_some_and(|c| c.source == il);
        let narrows = self.candidates.is_some_and(|c| c.source < il) && is_topk_source;
        let mut own_topk: Vec<Vec<usize>> = Vec::new();
        let mut own_candidates: Vec<Vec<usize>> = Vec::new();

        let mut attn_out = vec![0f32; n_tokens * n_head * head_dim];
        for t in 0..n_tokens {
            let pos = seqs.pos(t);
            let cache = seqs.cache(t);
            cache.layers[layer.slots.raw].push(
                &kv[t * head_dim..(t + 1) * head_dim],
                &kv[t * head_dim..(t + 1) * head_dim],
            );

            let chosen: Vec<usize> = match blocks_slot {
                None => Vec::new(),
                Some(slot) => {
                    let visible = visible_blocks(pos, layer.ratio);
                    if is_topk_source {
                        let (iq, weights) = index_query.as_ref().expect("a source has a query");
                        let mut scores = self.indexer_scores(
                            &cache.layers[slot],
                            &iq[t * idx_heads * idx_dim..(t + 1) * idx_heads * idx_dim],
                            &weights[t * idx_heads..(t + 1) * idx_heads],
                            visible,
                        );
                        if let Some(c) = self.candidates {
                            if is_candidate_source {
                                own_candidates.push(candidate_blocks(
                                    &scores,
                                    c.block_size,
                                    c.top_k_blocks,
                                ));
                            } else if narrows {
                                let blocks = &pass
                                    .candidates
                                    .as_ref()
                                    .context("candidate blocks before their source ran")?[t];
                                restrict_to_blocks(&mut scores, blocks, c.block_size);
                            }
                        }
                        let chosen = select_positions(&scores, self.indexer_top_k);
                        own_topk.push(chosen.clone());
                        chosen
                    } else {
                        let src = layer.topk_source.expect("a compressed layer has a source");
                        pass.topk[src].as_ref().with_context(|| {
                            format!("layer {il} ran before its top-k source {src}")
                        })?[t]
                            .clone()
                    }
                }
            };

            let window = raw_window(pos, self.n_swa);
            let n_raw = window.end() + 1 - window.start();
            let mut keys = vec![0f32; (n_raw + chosen.len()) * head_dim];
            let raw = &cache.layers[layer.slots.raw];
            for (i, p) in window.enumerate() {
                keys[i * head_dim..(i + 1) * head_dim].copy_from_slice(raw.key_at(p, 0, head_dim));
            }
            if let Some(slot) = blocks_slot {
                let blocks = &cache.layers[slot];
                for (i, &b) in chosen.iter().enumerate() {
                    let at = (n_raw + i) * head_dim;
                    keys[at..at + head_dim]
                        .copy_from_slice(&blocks.key_at(b, 0, head_dim + idx_dim)[..head_dim]);
                }
            }

            let q_t = &q[t * n_head * head_dim..(t + 1) * n_head * head_dim];
            let heads: Vec<Vec<f32>> = (0..n_head)
                .into_par_iter()
                .map(|h| {
                    attend(
                        &q_t[h * head_dim..(h + 1) * head_dim],
                        &keys,
                        head_dim,
                        head_dim,
                        scale,
                        Some(layer.attn_sinks[h]),
                    )
                })
                .collect();
            let dst = &mut attn_out[t * n_head * head_dim..(t + 1) * n_head * head_dim];
            for (h, head) in heads.iter().enumerate() {
                dst[h * head_dim..(h + 1) * head_dim].copy_from_slice(head);
                tensor::rope_apply_params_inverse_inplace(
                    &mut dst[h * head_dim + nope..(h + 1) * head_dim],
                    1,
                    rope_dim,
                    pos,
                    None,
                    &layer.rope,
                );
            }
        }
        if is_topk_source && blocks_slot.is_some() {
            pass.topk[il] = Some(own_topk);
        }
        if is_candidate_source {
            pass.candidates = Some(own_candidates);
        }

        let (mut projected, mut out) = (Vec::new(), Vec::new());
        grouped_output_projection(
            backend,
            &attn_out,
            n_tokens,
            n_head * head_dim,
            self.o_group_count,
            self.o_lora_rank,
            &layer.wo_a,
            &layer.wo_b,
            &mut projected,
            &mut out,
        );
        debug_assert_eq!(out.len(), n_tokens * n_embd);
        Ok(out)
    }

    /// A key source's compressor over this pass's tokens: each token's
    /// (value, score) row into the state slot, and each block a token
    /// completes — pooled, normed, its indexer key projected from it, both
    /// roped at the block's first position — into the blocks slot, before
    /// any token of the pass attends.
    fn compress(
        &self,
        backend: &dyn Backend,
        layer: &Deepseek41Layer,
        c: &Compressor,
        seqs: &mut Seqs<'_, '_>,
        cur: &[f32],
        n_tokens: usize,
    ) {
        let head_dim = self.config.head_dim;
        let eps = self.config.rms_eps;
        let rope_dim = self.config.rope_dim;
        let idx_dim = self.indexer_head_size;
        let ratio = layer.ratio;
        let state_slot = layer
            .slots
            .comp_state
            .expect("a compressor has a state slot");
        let blocks_slot = layer
            .slots
            .comp_blocks
            .expect("a compressor has a blocks slot");

        let values = backend.matmul(cur, n_tokens, &c.wkv);
        let scores = match c.wgate.as_ref() {
            Some(g) => backend.matmul(cur, n_tokens, g),
            None => vec![0f32; values.len()],
        };
        // The latents of the blocks this pass completes, with their owning
        // token, in token order.
        let mut completed: Vec<(usize, Vec<f32>)> = Vec::new();
        for t in 0..n_tokens {
            let pos = seqs.pos(t);
            let state = &mut seqs.cache(t).layers[state_slot];
            state.push(
                &values[t * head_dim..(t + 1) * head_dim],
                &scores[t * head_dim..(t + 1) * head_dim],
            );
            if (pos + 1).is_multiple_of(ratio) {
                let mut latent = pool_block_from_state(state, pos + 1 - ratio, ratio, head_dim);
                tensor::rmsnorm_inplace(&mut latent, &c.norm, 1, head_dim, eps);
                completed.push((t, latent));
            }
        }
        if completed.is_empty() {
            return;
        }
        let latents: Vec<f32> = completed
            .iter()
            .flat_map(|(_, l)| l.iter().copied())
            .collect();
        let mut index_keys = backend.matmul(&latents, completed.len(), &c.index_k);
        tensor::rmsnorm_inplace(
            &mut index_keys,
            &c.index_k_norm,
            completed.len(),
            idx_dim,
            eps,
        );
        for (i, (t, mut latent)) in completed.into_iter().enumerate() {
            let start = seqs.pos(t) + 1 - ratio;
            let mut row = Vec::with_capacity(head_dim + idx_dim);
            tensor::rope_apply_params_inplace(
                &mut latent[head_dim - rope_dim..],
                1,
                rope_dim,
                start,
                None,
                &layer.rope,
            );
            row.extend_from_slice(&latent);
            row.extend_from_slice(&index_keys[i * idx_dim..(i + 1) * idx_dim]);
            tensor::rope_apply_params_inplace(
                &mut row[head_dim + idx_dim - rope_dim..],
                1,
                rope_dim,
                start,
                None,
                &layer.rope,
            );
            seqs.cache(t).layers[blocks_slot].push(&row, &row);
        }
    }

    /// The indexer's score for each of the `visible` compressed blocks: a
    /// per-head `relu`'d dot product against the block's indexer key (the
    /// tail of its row), combined with this token's per-head weights.
    fn indexer_scores(
        &self,
        blocks: &LayerCache,
        q: &[f32],
        weights: &[f32],
        visible: usize,
    ) -> Vec<f32> {
        let head_dim = self.config.head_dim;
        let dim = self.indexer_head_size;
        (0..visible)
            .into_par_iter()
            .map(|b| {
                let key = &blocks.key_at(b, 0, head_dim + dim)[head_dim..];
                (0..self.indexer_n_head)
                    .map(|h| tensor::dot(&q[h * dim..(h + 1) * dim], key).max(0.0) * weights[h])
                    .sum()
            })
            .collect()
    }
}

/// Pools the `ratio` state rows from `start` into one latent: a per-feature
/// softmax over the members' scores, weighted-summed over their values.
fn pool_block_from_state(state: &LayerCache, start: usize, ratio: usize, width: usize) -> Vec<f32> {
    let mut values = Vec::with_capacity(ratio * width);
    let mut scores = Vec::with_capacity(ratio * width);
    for p in start..start + ratio {
        values.extend_from_slice(state.key_at(p, 0, width));
        scores.extend_from_slice(state.value_at(p, 0, width));
    }
    pool_block(&values, &scores, width)
}

impl ModelForward for Deepseek41Model {
    fn vulkan_backend(&self) -> Option<&crate::engine::backend::vulkan::VulkanBackend> {
        self.backend.as_wgpu()
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn n_trunk_layer(&self) -> usize {
        self.layers.len()
    }

    fn new_kv_cache(&self, capacity: usize) -> KvCache {
        KvCache::new_with_strided_dims(capacity, &self.kv_dims)
    }

    fn forward(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        _slot_id: usize,
    ) -> Result<Vec<f32>> {
        anyhow::ensure!(!tokens.is_empty(), "forward called with no tokens");
        let hidden = self.run_layers(&mut Seqs::One { cache, start_pos }, tokens)?;
        let n_embd = self.config.n_embd;
        let last = &hidden[(tokens.len() - 1) * n_embd..];
        Ok(self.backend.matmul(last, 1, &self.output_weight))
    }

    /// Every input position's next-token logits, for the speculative verify
    /// step. Sound because the whole cache is positional — the token ids, the
    /// compressor state and the strided blocks included — so
    /// `KvCache::truncate` rolls a rejected draft back exactly.
    fn forward_all_logits(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        _slot_id: usize,
    ) -> Result<Vec<Vec<f32>>> {
        anyhow::ensure!(
            !tokens.is_empty(),
            "forward_all_logits called with no tokens"
        );
        let hidden = self.run_layers(&mut Seqs::One { cache, start_pos }, tokens)?;
        let n_tokens = tokens.len();
        let logits = self.backend.matmul(&hidden, n_tokens, &self.output_weight);
        let n_vocab = logits.len() / n_tokens;
        Ok(logits.chunks_exact(n_vocab).map(<[f32]>::to_vec).collect())
    }

    fn forward_hidden_states(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        anyhow::ensure!(
            !tokens.is_empty(),
            "forward_hidden_states called with no tokens"
        );
        let mut cache = self.new_kv_cache(tokens.len());
        self.run_layers(
            &mut Seqs::One {
                cache: &mut cache,
                start_pos: 0,
            },
            tokens,
        )
    }

    /// Every layer runs on the host's turn of the step, so a batch is the
    /// same layer loop with each row's keys, blocks and token history on its
    /// own cache.
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
        let hidden = self.run_layers(&mut Seqs::Rows(rows), tokens)?;
        let backend = crate::engine::backend::DecodeKernels::for_batch(self.backend.as_ref(), true);
        let flat = backend.matmul(&hidden, n_rows, &self.output_weight);
        let n_vocab = flat.len() / n_rows;
        Ok(Some(
            flat.chunks_exact(n_vocab).map(<[f32]>::to_vec).collect(),
        ))
    }

    fn batches_verify(&self) -> bool {
        super::batch_verify()
    }

    fn forward_decode_batch_verify(
        &self,
        rows: &mut [super::DecodeRow<'_>],
        tokens: &[&[u32]],
        _samples: &[Option<super::GreedySampleParams<'_>>],
    ) -> Result<Option<Vec<Vec<super::ForwardOutcome>>>> {
        if !self.batches_verify()
            || tokens.len() != rows.len()
            || tokens.iter().any(|t| t.is_empty())
        {
            return Ok(None);
        }
        let widths: Vec<usize> = tokens.iter().map(|t| t.len()).collect();
        let flat: Vec<u32> = tokens.iter().flat_map(|t| t.iter().copied()).collect();
        let n_tokens = flat.len();
        let hidden = self.run_layers(&mut Seqs::runs(rows, &widths), &flat)?;
        let backend = crate::engine::backend::DecodeKernels::for_batch(self.backend.as_ref(), true);
        let logits = backend.matmul(&hidden, n_tokens, &self.output_weight);
        let n_vocab = logits.len() / n_tokens;
        let mut rows_out = logits
            .chunks_exact(n_vocab)
            .map(|row| super::ForwardOutcome::Logits(row.to_vec()));
        Ok(Some(
            widths
                .iter()
                .map(|&w| rows_out.by_ref().take(w).collect())
                .collect(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        EngramHash, candidate_blocks, resolve_sources, restrict_to_blocks, select_positions,
    };

    /// The released file's layout: two window-only layers, ratio 2 through
    /// layer 19 sourced from 2, 8 and 14, then ratio 1 sourced from 20.
    #[test]
    fn each_compressed_layer_reads_the_latest_source_at_or_before_it() {
        let mut ratios = vec![0, 0];
        ratios.extend(std::iter::repeat_n(2, 18));
        ratios.extend(std::iter::repeat_n(1, 20));
        let kv = resolve_sources(&ratios, &[2, 8, 14, 20]).unwrap();
        assert_eq!(kv[0], None);
        assert_eq!(kv[1], None);
        assert_eq!(kv[2], Some(2));
        assert_eq!(kv[7], Some(2));
        assert_eq!(kv[8], Some(8));
        assert_eq!(kv[19], Some(14));
        assert_eq!(kv[20], Some(20));
        assert_eq!(kv[39], Some(20));
        let topk = resolve_sources(&ratios, &[2, 8, 14, 20, 24, 28, 32, 36]).unwrap();
        assert_eq!(topk[23], Some(20));
        assert_eq!(topk[24], Some(24));
        assert_eq!(topk[39], Some(36));
    }

    #[test]
    fn a_source_must_compress_and_match_its_readers_ratio() {
        assert!(resolve_sources(&[0, 2], &[0]).is_err());
        // Layer 2 has ratio 1 but its only source compresses at 2.
        assert!(resolve_sources(&[0, 2, 1], &[1]).is_err());
        // A compressed layer before any source.
        assert!(resolve_sources(&[2, 2], &[1]).is_err());
    }

    #[test]
    fn candidate_blocks_rank_by_best_member_and_always_keep_the_newest() {
        // Blocks of 2: [0.1, 0.9] [0.5, 0.2] [0.3] — the last is partial.
        let scores = [0.1, 0.9, 0.5, 0.2, 0.3];
        assert_eq!(candidate_blocks(&scores, 2, 2), vec![2, 0]);
        // More blocks asked for than exist.
        assert_eq!(candidate_blocks(&scores, 2, 9).len(), 3);
    }

    #[test]
    fn a_restricted_score_can_only_pick_inside_its_blocks() {
        let mut scores = vec![5.0, 4.0, 3.0, 2.0, 1.0, 0.0];
        restrict_to_blocks(&mut scores, &[2], 2);
        assert_eq!(select_positions(&scores, 3), vec![4, 5]);
    }

    #[test]
    fn selected_positions_are_ascending() {
        assert_eq!(select_positions(&[0.0, 3.0, 1.0, 2.0], 3), vec![1, 2, 3]);
        assert_eq!(select_positions(&[0.0, 3.0], 9), vec![0, 1]);
    }

    fn hash() -> EngramHash {
        // Two modules, trigrams, two heads per n-gram width.
        EngramHash {
            ngram_size: 3,
            heads_per_ngram: 2,
            n_cols: 4,
            multipliers: vec![3, 5, 7, 11, 13, 17],
            offsets: vec![0, 100, 200, 300, 1000, 1100, 1200, 1300],
            buckets: vec![97, 89, 83, 79, 97, 89, 83, 79],
            token_map: vec![10, 11, 12, 12, 14],
            pad: 12,
        }
    }

    /// `rolling = t0*m0 ^ t1*m1 [^ t2*m2]`, reduced per head into its own
    /// slice — the same arithmetic written out by hand.
    #[test]
    fn the_engram_hash_folds_compressed_ids_into_each_heads_slice() {
        let h = hash();
        // Token 4 (id 14) after tokens 1 (11) and 0 (10).
        let prev = |s: usize| [None, Some(1u32), Some(0u32)][s];
        let rows = h.rows(0, 4, prev);
        let bi = (14u64 * 3) ^ (11 * 5);
        let tri = bi ^ (10 * 7);
        assert_eq!(
            rows,
            vec![
                (bi % 97) as usize,
                (bi % 89 + 100) as usize,
                (tri % 83 + 200) as usize,
                (tri % 79 + 300) as usize,
            ]
        );
        // The second module uses its own multipliers and slices.
        let rows = h.rows(1, 4, prev);
        let bi = (14u64 * 11) ^ (11 * 13);
        assert_eq!(rows[0], (bi % 97 + 1000) as usize);
    }

    #[test]
    fn a_missing_or_unmapped_predecessor_pads_it_and_every_one_before_it() {
        let h = hash();
        // At the start of the sequence: both predecessors are the pad.
        let start = h.rows(0, 4, |_| None);
        let bi = (14u64 * 3) ^ (12 * 5);
        assert_eq!(start[0], (bi % 97) as usize);
        assert_eq!(start[2], ((bi ^ (12 * 7)) % 83 + 200) as usize);
        // An unmapped first predecessor blocks the second too, even though
        // the second is a real token.
        let blocked = h.rows(0, 4, |s| [None, Some(99u32), Some(0u32)][s]);
        assert_eq!(blocked, start);
        // An unmapped token pads itself (36 ^ 60 = 24, under every bucket
        // count).
        let own = h.rows(0, 99, |_| None);
        assert_eq!(own[0], ((12u64 * 3) ^ (12 * 5)) as usize);
    }
}

#[cfg(test)]
mod real_model_tests {
    use super::*;
    use crate::engine::loader::LoadedModel;

    /// The real checkpoint completes "The capital of France is" with
    /// " Paris". The Engram table is what this guards: it is stored as MXFP8
    /// under id 43, and read as unscaled E4M3 every row comes out about 2^8
    /// too large, which turns the continuation into fluent nonsense
    /// (" scout") while every vector stays well-formed.
    ///
    /// Run with `ORANGU_TEST_DEEPSEEK41_MODEL=/path/to/first-shard.gguf
    /// cargo test --profile release-with-debug --bin orangu-server
    /// deepseek41::real_model_tests -- --ignored`. Expect minutes: every
    /// token streams its experts off disk.
    #[test]
    #[ignore]
    fn deepseek41_continues_the_capital_of_france_with_paris() {
        let path = std::env::var("ORANGU_TEST_DEEPSEEK41_MODEL")
            .expect("set ORANGU_TEST_DEEPSEEK41_MODEL to a DeepSeek-V4.1 GGUF");
        let loaded = LoadedModel::open(std::path::Path::new(&path)).expect("load model");
        assert_eq!(loaded.config.architecture, "deepseek41");
        let gguf = orangu::gguf::GgufFile::open(std::path::Path::new(&path)).expect("open gguf");
        let tokenizer =
            crate::engine::tokenizer::Tokenizer::from_gguf(&gguf).expect("build tokenizer");
        // This file sets `tokenizer.ggml.add_bos_token = false`.
        let tokens = tokenizer.encode("The capital of France is", false);
        let model = Deepseek41Model::load_with_backend(
            &loaded,
            Arc::new(crate::engine::backend::CpuBackend),
        )
        .expect("build model");
        let mut cache = model.new_kv_cache(64);
        let logits = model.forward(&mut cache, &tokens, 0, 0).expect("forward");
        let (top_id, _) = logits
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        assert_eq!(tokenizer.decode(&[top_id as u32]), " Paris");
    }
}

#[cfg(test)]
mod batch_tests {
    use super::*;
    use orangu::gguf::GgufValue;
    use orangu::gguf_write::TensorPlan;

    /// A batch runs every projection and expert over all its rows at once and
    /// a solo step over one; on the host both take the same kernels.
    const TOLERANCE: f64 = 3.5e-3;

    const VOCAB: usize = 2200;
    const EMBD: u64 = 64;
    const HC: u64 = 4;
    const HEADS: u64 = 4;
    const HEAD_DIM: u64 = 32;
    const Q_LORA: u64 = 16;
    const GROUPS: u64 = 2;
    const O_LORA: u64 = 8;
    const EXPERTS: u64 = 4;
    const FF: u64 = 16;
    const IDX_HEADS: u64 = 2;
    const IDX_DIM: u64 = 16;
    const PLE_DIM: u64 = 8;
    /// Window-only, two ratio-2 layers sourced from 1, three ratio-1 layers
    /// sourced from 3 with a second top-k source at 5 narrowed by 3's
    /// candidate blocks.
    const RATIOS: [u32; 6] = [0, 2, 2, 1, 1, 1];
    const KV_SOURCES: [u32; 2] = [1, 3];
    const INDEX_SOURCES: [u32; 3] = [1, 3, 5];
    const ENGRAM_LAYERS: [u32; 2] = [1, 4];
    const BUCKETS: [u64; 8] = [7, 11, 13, 17, 19, 23, 29, 31];

    fn u32s(v: &[u32]) -> GgufValue {
        GgufValue::Array(v.iter().map(|&x| GgufValue::U32(x)).collect())
    }

    fn u64s(v: &[u64]) -> GgufValue {
        GgufValue::Array(v.iter().map(|&x| GgufValue::U64(x)).collect())
    }

    /// Deterministic weights in `[-scale, scale]`, different per tensor.
    fn weights(name: &str, n: usize, scale: f32) -> Vec<f32> {
        let mut h: u64 = name.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
        });
        (0..n)
            .map(|_| {
                h ^= h << 13;
                h ^= h >> 7;
                h ^= h << 17;
                ((h >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
            })
            .collect()
    }

    /// A six-layer `deepseek41` file with every mechanism the released one
    /// has — both ratios, shared keys and top-k, candidate blocks, Engram —
    /// small enough to build and run in a unit test.
    fn synthetic_model() -> Deepseek41Model {
        Deepseek41Model::load_with_backend(
            &synthetic_loaded(),
            Arc::new(crate::engine::backend::CpuBackend),
        )
        .expect("build the synthetic model")
    }

    fn synthetic_loaded() -> LoadedModel {
        let p = |s: &str| format!("deepseek41.{s}");
        let offsets: Vec<u64> = BUCKETS
            .iter()
            .scan(0, |acc, &b| {
                let o = *acc;
                *acc += b;
                Some(o)
            })
            .collect();
        let table_rows: u64 = BUCKETS.iter().sum();
        let metadata: Vec<(String, GgufValue)> = vec![
            (
                "general.architecture".into(),
                GgufValue::String("deepseek41".into()),
            ),
            (p("vocab_size"), GgufValue::U32(VOCAB as u32)),
            (p("block_count"), GgufValue::U32(RATIOS.len() as u32)),
            (p("context_length"), GgufValue::U32(256)),
            (p("embedding_length"), GgufValue::U32(EMBD as u32)),
            (p("attention.head_count"), GgufValue::U32(HEADS as u32)),
            (p("attention.head_count_kv"), GgufValue::U32(1)),
            (p("attention.key_length"), GgufValue::U32(HEAD_DIM as u32)),
            (p("attention.value_length"), GgufValue::U32(HEAD_DIM as u32)),
            (p("rope.dimension_count"), GgufValue::U32(16)),
            (p("rope.freq_base"), GgufValue::F32(10000.0)),
            (p("attention.layer_norm_rms_epsilon"), GgufValue::F32(1e-6)),
            (p("attention.q_lora_rank"), GgufValue::U32(Q_LORA as u32)),
            (p("attention.sliding_window"), GgufValue::U32(4)),
            (
                p("attention.output_group_count"),
                GgufValue::U32(GROUPS as u32),
            ),
            (
                p("attention.output_lora_rank"),
                GgufValue::U32(O_LORA as u32),
            ),
            (p("attention.compress_ratios"), u32s(&RATIOS)),
            (
                p("attention.compress_rope_freq_base"),
                GgufValue::F32(160000.0),
            ),
            (p("attention.kv_source_layers"), u32s(&KV_SOURCES)),
            (p("attention.indexer.source_layers"), u32s(&INDEX_SOURCES)),
            (
                p("attention.indexer.head_count"),
                GgufValue::U32(IDX_HEADS as u32),
            ),
            (
                p("attention.indexer.key_length"),
                GgufValue::U32(IDX_DIM as u32),
            ),
            (p("attention.indexer.top_k"), GgufValue::U32(4)),
            (
                p("attention.indexer.candidate_source_layer"),
                GgufValue::U32(3),
            ),
            (
                p("attention.indexer.candidate_top_k_blocks"),
                GgufValue::U32(2),
            ),
            (
                p("attention.indexer.candidate_block_size"),
                GgufValue::U32(2),
            ),
            (p("expert_count"), GgufValue::U32(EXPERTS as u32)),
            (p("expert_used_count"), GgufValue::U32(2)),
            (p("expert_gating_func"), GgufValue::U32(4)),
            (p("expert_feed_forward_length"), GgufValue::U32(FF as u32)),
            (p("expert_shared_count"), GgufValue::U32(1)),
            (p("expert_weights_scale"), GgufValue::F32(1.5)),
            (p("expert_weights_norm"), GgufValue::Bool(true)),
            (
                p("swiglu_clamp_exp"),
                GgufValue::Array(vec![GgufValue::F32(10.0); RATIOS.len()]),
            ),
            (p("hyper_connection.count"), GgufValue::U32(HC as u32)),
            (p("hyper_connection.sinkhorn_iterations"), GgufValue::U32(3)),
            (p("hyper_connection.epsilon"), GgufValue::F32(1e-6)),
            (p("ple.layers"), u32s(&ENGRAM_LAYERS)),
            (p("ple.ngram_size"), GgufValue::U32(3)),
            (p("ple.heads_per_ngram"), GgufValue::U32(2)),
            (
                p("embedding_length_per_layer_input"),
                GgufValue::U32(PLE_DIM as u32),
            ),
            (p("ple.layer_multipliers"), u64s(&[3, 5, 7, 11, 13, 17])),
            (p("ple.head_offsets"), u64s(&offsets)),
            (p("ple.head_vocab_sizes"), u64s(&BUCKETS)),
            (p("engram.pad_token_id"), GgufValue::U32(2)),
            (
                p("engram.token_map"),
                u32s(&(0..VOCAB as u32).map(|t| t % 50).collect::<Vec<_>>()),
            ),
        ];

        let mut plans: Vec<(TensorPlan, f32)> = Vec::new();
        let mut add = |name: String, dims: &[u64], scale: f32| {
            plans.push((
                TensorPlan {
                    name,
                    dims: dims.to_vec(),
                    ggml_type: crate::engine::quant::GGML_TYPE_F32,
                },
                scale,
            ));
        };
        // A scale of 0 marks a norm weight, written as ones.
        add("token_embd.weight".into(), &[EMBD, VOCAB as u64], 1.0);
        add("output_norm.weight".into(), &[EMBD], 0.0);
        add("output.weight".into(), &[EMBD, VOCAB as u64], 0.3);
        add(
            "per_layer_token_embd.weight".into(),
            &[PLE_DIM, table_rows],
            1.0,
        );
        let mix = (2 + HC) * HC;
        for (i, &ratio) in RATIOS.iter().enumerate() {
            let mut layer = |s: &str, dims: &[u64], scale: f32| {
                add(format!("blk.{i}.{s}"), dims, scale);
            };
            layer("attn_norm.weight", &[EMBD], 0.0);
            layer("attn_sinks.weight", &[HEADS], 0.5);
            layer("attn_q_a.weight", &[EMBD, Q_LORA], 0.3);
            layer("attn_q_a_norm.weight", &[Q_LORA], 0.0);
            layer("attn_q_b.weight", &[Q_LORA, HEADS * HEAD_DIM], 0.5);
            layer("attn_kv.weight", &[EMBD, HEAD_DIM], 0.3);
            layer("attn_kv_a_norm.weight", &[HEAD_DIM], 0.0);
            layer(
                "attn_output_a.weight",
                &[HEADS * HEAD_DIM / GROUPS, GROUPS * O_LORA],
                0.2,
            );
            layer("attn_output_b.weight", &[GROUPS * O_LORA, EMBD], 0.3);
            for hc in ["hc_attn", "hc_ffn"] {
                layer(&format!("{hc}_fn.weight"), &[HC * EMBD, mix], 0.2);
                layer(&format!("{hc}_base.weight"), &[mix], 0.3);
                layer(&format!("{hc}_scale.weight"), &[3], 0.5);
            }
            if KV_SOURCES.contains(&(i as u32)) {
                layer("attn_compressor_kv.weight", &[EMBD, HEAD_DIM], 0.3);
                layer("attn_compressor_norm.weight", &[HEAD_DIM], 0.0);
                if ratio > 1 {
                    layer("attn_compressor_gate.weight", &[EMBD, HEAD_DIM], 0.3);
                }
                layer("indexer.attn_k.weight", &[HEAD_DIM, IDX_DIM], 0.3);
                layer("indexer.k_norm.weight", &[IDX_DIM], 0.0);
            }
            if INDEX_SOURCES.contains(&(i as u32)) {
                layer(
                    "indexer.attn_q_b.weight",
                    &[Q_LORA, IDX_HEADS * IDX_DIM],
                    0.5,
                );
                layer("indexer.proj.weight", &[EMBD, IDX_HEADS], 0.5);
            }
            if ENGRAM_LAYERS.contains(&(i as u32)) {
                layer("engram_kv.weight", &[4 * PLE_DIM, (HC + 1) * EMBD], 0.2);
                layer("engram_gate.weight", &[EMBD, HC], 1.0);
            }
            layer("ffn_norm.weight", &[EMBD], 0.0);
            layer("ffn_gate_inp.weight", &[EMBD, EXPERTS], 0.5);
            layer("exp_probs_b.bias", &[EXPERTS], 0.1);
            layer("ffn_gate_exps.weight", &[EMBD, FF, EXPERTS], 0.3);
            layer("ffn_up_exps.weight", &[EMBD, FF, EXPERTS], 0.3);
            layer("ffn_down_exps.weight", &[FF, EMBD, EXPERTS], 0.3);
            layer("ffn_gate_shexp.weight", &[EMBD, FF], 0.3);
            layer("ffn_up_shexp.weight", &[EMBD, FF], 0.3);
            layer("ffn_down_shexp.weight", &[FF, EMBD], 0.3);
        }

        // One file per call: the tests run in parallel in one process.
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "orangu-deepseek41-synthetic-{}-{}.gguf",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let scales: std::collections::HashMap<String, f32> =
            plans.iter().map(|(t, s)| (t.name.clone(), *s)).collect();
        let tensors: Vec<TensorPlan> = plans.into_iter().map(|(t, _)| t).collect();
        orangu::gguf_write::write(
            &path,
            &metadata,
            &tensors,
            |t| {
                let scale = scales[&t.name];
                Ok(if scale == 0.0 {
                    vec![1.0; t.elements()]
                } else {
                    weights(&t.name, t.elements(), scale)
                })
            },
            &|_, _, _| {},
        )
        .expect("write the synthetic model");
        let loaded = LoadedModel::open(&path).expect("open the synthetic model");
        std::fs::remove_file(&path).ok();
        loaded
    }

    #[test]
    fn the_synthetic_model_resolves_its_sources_and_engram_layers() {
        let model = synthetic_model();
        let kv: Vec<_> = model.layers.iter().map(|l| l.kv_source).collect();
        assert_eq!(kv, [None, Some(1), Some(1), Some(3), Some(3), Some(3)]);
        let tk: Vec<_> = model.layers.iter().map(|l| l.topk_source).collect();
        assert_eq!(tk, [None, Some(1), Some(1), Some(3), Some(3), Some(5)]);
        let engram: Vec<_> = model
            .layers
            .iter()
            .map(|l| l.engram.as_ref().map(|e| e.module))
            .collect();
        assert_eq!(engram, [None, Some(0), None, None, Some(1), None]);
        assert!(model.candidates.is_some());
    }

    /// Windowed as it is, the file prefills in full-width chunks: each pass
    /// reads every layer's experts, so narrowing a chunk only adds passes.
    #[test]
    fn the_synthetic_model_prefills_in_full_width_chunks() {
        let loaded = synthetic_loaded();
        assert!(super::super::prefill_in_full_width(&loaded));
        assert_eq!(super::super::prefill_chunk_tokens_for(&loaded), Some(2048));
    }

    /// A batch row must read its own keys, blocks, top-k and token history:
    /// the prompts here are long enough (40 and 23 tokens) for the top-k and
    /// the candidate blocks to cut.
    #[test]
    fn a_decode_batch_matches_each_rows_solo_step() {
        let model = synthetic_model();
        super::super::batch_checks::decode_batch_matches_each_rows_solo_step(&model, TOLERANCE);
    }

    #[test]
    fn a_batched_verify_matches_its_own_pass() {
        let model = synthetic_model();
        super::super::batch_checks::decode_batch_verifies_a_draft_as_its_own_pass_would(
            &model, TOLERANCE,
        );
    }

    /// A prompt split across two passes gives the same logits as one pass:
    /// the Engram history at the seam comes from the cache, and the
    /// compressed blocks completed in the first pass are read by the second.
    #[test]
    fn a_chunked_prefill_matches_a_single_pass() {
        let model = synthetic_model();
        let prompt: Vec<u32> = (0..31u32).map(|i| 100 + (i * 53) % 2000).collect();
        let mut whole = model.new_kv_cache(64);
        let a = model.forward(&mut whole, &prompt, 0, 0).unwrap();
        let mut split = model.new_kv_cache(64);
        model.forward(&mut split, &prompt[..17], 0, 0).unwrap();
        let b = model.forward(&mut split, &prompt[17..], 17, 0).unwrap();
        let worst = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-3, "chunked prefill differs by {worst}");
    }
}
