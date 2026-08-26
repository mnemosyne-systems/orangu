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

//! Qwen4-preview (`general.architecture = "qwen4exp"`), e.g.
//! `unsloth/Qwen3.8-Flash-Next-GGUF` — confirmed against real upstream
//! source (`src/models/qwen4exp.cpp`, `src/llama-memory-hybrid-idx.cpp`,
//! `src/llama-hparams.cpp` and the `llm_build_delta_net_base` code it
//! inherits, read directly rather than inferred from tensor shapes).
//!
//! Three of the four halves of a block are already written elsewhere and
//! are used from here rather than copied:
//!
//! - the **full-attention** sub-layer (joint query+gate projection,
//!   Q/K-norm, partial rotary, GQA, `sigmoid` output gate) is
//!   [`qwen_hybrid::FullAttn`],
//! - the **gated-DeltaNet** sub-layer is [`qwen_hybrid::Recurrent`], which
//!   differs here by exactly one nonlinearity — the output gate is
//!   `sigmoid`, not `silu` ([`qwen_hybrid::OutputGate`]),
//! - the **FFN** — softmax top-k routed experts plus a separately
//!   `sigmoid`-gated shared expert — is [`qwen_hybrid::MoeFfn`], the same
//!   FFN `qwen35moe` and `qwen3next` run, at 512 experts and top-10.
//!
//! What is only here is what has no counterpart in that trunk:
//!
//! ## Hyper-connections
//!
//! There is no residual *vector*. The state between sub-layers is
//! `hyper_connection.count` (4) parallel streams of `n_embd`, seeded as
//! four copies of the token embedding, and there is no `output_norm`: the
//! final mixer's own norm is the last normalization in the model. Each
//! sub-layer is bracketed by [`HcMixer`] instead of by RMSNorms — a
//! grouped norm, a low-rank `down`/`silu`/`up` `sigmoid` gate, and a mean
//! collapse to the one vector the sub-layer reads, plus the per-stream
//! injection weights that scatter its output back ([`HcMixer::combine`]).
//!
//! `engine::arch::deepseek4` also has four streams, and deliberately does
//! **not** share this code: DeepSeek-V4 mixes at full rank and normalizes
//! its stream-combination matrix with Sinkhorn iterations, where this is a
//! low-rank gate with a plain mean and a `2 * sigmoid` scatter. The two
//! agree on the shape of the idea and on none of the arithmetic.
//!
//! ## Query-sparse attention (QSA)
//!
//! Each full-attention layer carries a small **indexer** — four heads of
//! its own `q`/`k` projections — that scores whole blocks of
//! `attention.compress_ratios[layer]` (4) consecutive cached positions,
//! each represented by its members' mean key, and lets the real attention
//! see only the best `attention.indexer.top_k` of them plus the always
//! visible incomplete tail. Below `top_k + ratio - 1` cached positions the
//! result is exactly dense attention, which is what short contexts get.
//!
//! ## Per-layer embeddings (PLE)
//!
//! The layers named by `ple.layers` (layer 1 alone in the released
//! checkpoints) inject a second embedding read from a 320-million-row
//! n-gram hash table: the current token and its two predecessors are mixed
//! into one 64-bit hash per head, and each head's row is looked up in its
//! own slice of the table. See [`Ple`].
//!
//! ## Not implemented
//!
//! **Multi-section RoPE** (`rope.dimension_sections`) is plain NEOX rope
//! here for the same reason it is on the shared trunk: with text-only
//! input every position channel carries the same value, at which point the
//! sections mechanism is a no-op. Image and video input is out of scope
//! for this engine, and with it `ple.image_token_id`, which only names the
//! placeholder id a vision batch would hash.
//!
//! **The chunked/parallel form of the delta rule** — the trunk runs the
//! autoregressive form, which is the same arithmetic; see
//! [`qwen_hybrid`]'s own note.

use anyhow::{Context, Result};
use rayon::prelude::*;
use std::sync::Arc;

use super::ModelForward;
use super::qwen_hybrid::{
    self, Dims, FullAttn, LayerTensors, MoeFfn, OutputGate, Recurrent, recurrent_layer_mask,
    trunk_layer_count,
};
use crate::engine::backend::Backend;
use crate::engine::kv_cache::{KvCache, RecurrentSpec};
use crate::engine::loader::{LoadedModel, ModelConfig, QuantMatrix};
use crate::engine::tensor;

/// One hyper-connection mixer: the grouped norm and low-rank gate that
/// collapse the `hc` residual streams into the single vector a sub-layer
/// reads, plus (except on the model's final head) the projection that
/// predicts how that sub-layer's output is scattered back across them.
///
/// `hc_attn_*` and `hc_ffn_*` per layer, `output_hc_*` once at the end.
struct HcMixer {
    /// `[hc * n_embd]` — one gamma per stream *and* channel, applied after
    /// a norm taken over each stream separately. The converter folded the
    /// `1 + w` these were trained with, so this is a plain multiply.
    norm: Vec<f32>,
    /// `[hc * n_embd, hyper_connection.low_rank]`.
    down: QuantMatrix,
    /// `[hyper_connection.low_rank, hc * n_embd]`.
    up: QuantMatrix,
    /// `[hc * n_embd, hc]` — the per-stream scatter weights. `None` on the
    /// final head, which collapses without writing anything back.
    inject: Option<QuantMatrix>,
}

impl HcMixer {
    fn load(loaded: &LoadedModel, prefix: &str, with_inject: bool) -> Result<Self> {
        let get = |suffix: &str| -> Result<Vec<f32>> {
            let name = format!("{prefix}_{suffix}.weight");
            Ok(loaded
                .tensor(&name)
                .with_context(|| format!("loading {name}"))?
                .0)
        };
        let matrix = |suffix: &str| -> Result<QuantMatrix> {
            let name = format!("{prefix}_{suffix}.weight");
            loaded
                .matrix(&name)
                .with_context(|| format!("loading {name}"))
        };
        Ok(Self {
            norm: get("norm")?,
            down: matrix("down")?,
            up: matrix("up")?,
            inject: with_inject.then(|| matrix("inject")).transpose()?,
        })
    }
}

/// One full-attention layer's indexer: its own query and key projections
/// and their norms, and where its raw per-position keys are cached.
///
/// The keys are cached *raw* — before the norm and before the rotation —
/// because pooling comes first: a block's key is the mean of its members'
/// unrotated keys, and only that mean is then normed and rotated, at the
/// block's own first position.
struct Indexer {
    /// `[n_embd, indexer.head_count * indexer.key_length]`.
    q_proj: QuantMatrix,
    /// `[n_embd, indexer.key_length]` — one key per position, shared by
    /// every indexer head.
    k_proj: QuantMatrix,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    /// This layer's `attention.compress_ratios` entry: how many consecutive
    /// positions one scored block covers.
    ratio: usize,
    /// Dense index into `KvCache::layers` for the raw keys.
    cache_index: usize,
}

/// The token-mixing half of a block: full attention (with its indexer) or
/// the gated delta net, chosen per layer by `full_attention_interval`.
enum Mixer {
    FullAttn { attn: FullAttn, indexer: Indexer },
    Recurrent(Recurrent),
}

/// The n-gram hash embedding table and the parameters of its hash, read
/// once for the whole model.
///
/// Each token contributes `ple_n_heads = (ngram_size - 1) * heads_per_ngram`
/// rows of `head_dim` values, concatenated head-major into one `n_embd`
/// vector. For each n from 2 to `ngram_size`, the token and its n-1
/// predecessors are folded into a single 64-bit value
///
/// ```text
/// mixed_n = (t[0] * m[0]) ^ (t[1] * m[1]) ^ ... ^ (t[n-1] * m[n-1])
/// ```
///
/// and that value indexes each of the `heads_per_ngram` heads belonging to
/// that n, modulo the head's own vocabulary size and offset by where the
/// head's slice of the table starts. Predecessors stop at a segment
/// boundary: an EOS anywhere in the window replaces it and everything
/// before it, and a position before the start of the sequence reads as EOS.
struct Ple {
    /// `[head_dim, rows]` — one shared table, ~320 M rows, read one row at
    /// a time. Never dense-resident: only `n_heads` rows per token are
    /// ever touched.
    table: QuantMatrix,
    /// `embedding_length_per_layer_input`.
    head_dim: usize,
    hash: NgramHash,
}

/// The hash itself, with no table attached — the half of [`Ple`] that is
/// pure arithmetic over token ids, kept separate so it can be checked
/// without a 25-gigabyte tensor behind it.
struct NgramHash {
    /// `(ngram_size - 1) * heads_per_ngram`.
    n_heads: usize,
    ngram_size: usize,
    heads_per_ngram: usize,
    multipliers: Vec<u64>,
    head_offsets: Vec<u64>,
    head_vocab_sizes: Vec<u64>,
    eos: u32,
}

/// One PLE layer's own weights: the two projections off the gathered
/// embedding, the three grouped norms, and the dilated depthwise
/// convolution over the result.
struct PleLayer {
    /// `[n_embd, hc * n_embd]`.
    key: QuantMatrix,
    /// `[n_embd, n_embd]`.
    value: QuantMatrix,
    norm_key: Vec<f32>,
    norm_query: Vec<f32>,
    norm_conv: Vec<f32>,
    /// The `ple_conv1d` kernel re-laid as a dense `[hc * n_embd, taps]`
    /// channel-major kernel with zeros between the taps — see
    /// [`expand_dilated_kernel`].
    conv1d: Vec<f32>,
    /// Dense index into `KvCache::recurrent` for the convolution's rolling
    /// history.
    cache_index: usize,
}

struct Qwen4ExpLayer {
    hc_attn: HcMixer,
    hc_ffn: HcMixer,
    mixer: Mixer,
    ple: Option<PleLayer>,
    ffn: MoeFfn,
}

pub struct Qwen4ExpModel {
    config: ModelConfig,
    backend: Arc<dyn Backend>,
    dims: Dims,
    /// `hyper_connection.count`.
    hc: usize,
    /// `attention.indexer.key_length`.
    indexer_head_size: usize,
    /// `attention.indexer.head_count`.
    indexer_n_head: usize,
    /// `attention.indexer.top_k`.
    indexer_top_k: usize,
    tok_embeddings: QuantMatrix,
    output_weight: QuantMatrix,
    /// `output_hc_*` — the final collapse of the streams, and the model's
    /// last normalization.
    head: HcMixer,
    ple: Option<Ple>,
    layers: Vec<Qwen4ExpLayer>,
    /// `(kv_dim, stride)` per `KvCache::layers` slot, in slot order.
    kv_dims: Vec<usize>,
    recurrent_specs: Vec<RecurrentSpec>,
}

/// Re-lays a `[channels, kernel]` depthwise kernel as a `[channels, (kernel
/// - 1) * dilation + 1]` one whose unused taps are zero.
///
/// The PLE convolution is dilated by the n-gram size, and
/// `RecurrentLayerState::conv_step` — the rolling per-sequence history every
/// other convolution in this engine goes through — is not. Writing the
/// dilation into the kernel instead of into the stepper is what lets this
/// architecture reuse that machinery unchanged: the history it keeps, the
/// carryover across a chunked prefill's seams, and the slot persistence all
/// come along, and the extra taps are exact zeros rather than an
/// approximation. The kernel is `hc * n_embd` channels of four taps, so the
/// padding costs a few tens of thousands of floats, once, at load.
///
/// `out[c, t] = sum_k w[k, c] * x[c, t - (kernel - 1 - k) * dilation]`, and
/// `conv_step`'s tap `i` weights `x[c, t - (taps - 1 - i)]`, so tap `k` of
/// the original lands at `(taps - 1) - (kernel - 1 - k) * dilation`.
fn expand_dilated_kernel(w: &[f32], channels: usize, kernel: usize, dilation: usize) -> Vec<f32> {
    let taps = (kernel - 1) * dilation + 1;
    let mut out = vec![0f32; channels * taps];
    for c in 0..channels {
        for k in 0..kernel {
            let tap = (taps - 1) - (kernel - 1 - k) * dilation;
            out[c * taps + tap] = w[c * kernel + k];
        }
    }
    out
}

/// The `2 * sigmoid(inject / hc)` a hyper-connection scatters one
/// sub-layer's output into one stream with.
///
/// Doubling a sigmoid centres the weights on 1, which is what makes the
/// mechanism a *generalization* of the residual add it replaces rather than
/// a different thing: an all-zero injection projection reproduces
/// `x += sub(x)` exactly, per stream.
fn hc_scatter_weight(inject: f32, hc: usize) -> f32 {
    2.0 * tensor::sigmoid(inject / hc as f32)
}

/// The PLE value gate: a signed square root of the per-stream score,
/// squashed. `|s|` is floored at `1e-6` before the root, so the derivative
/// stays finite at zero — upstream's `clamp(abs(s), 1e-6, INFINITY)`.
fn ple_gate(score: f32) -> f32 {
    tensor::sigmoid(score.signum() * score.abs().max(1e-6).sqrt())
}

/// Which of a query's `visible` cached positions it attends, ascending.
///
/// `block_scores[b]` scores the block covering the `ratio` positions
/// starting at `b * ratio`; the positions past the last complete block are
/// the incomplete tail, which is always attended. `width` is the budget —
/// `indexer.top_k + ratio - 1`, a whole budget of blocks plus room for that
/// tail.
///
/// Every member of a block carries its block's score, rather than the
/// blocks being ranked and then expanded, because the budget is a whole
/// number of blocks and the members tie — so the cut lands on a block
/// boundary either way, and this is the arithmetic upstream's graph does.
/// Ties break towards the earlier position ([`super::top_k_indices`] keeps
/// input order), which is the one place a deterministic choice is made
/// where upstream's `ggml_top_k` leaves it open.
fn select_positions(
    block_scores: &[f32],
    visible: usize,
    ratio: usize,
    width: usize,
) -> Vec<usize> {
    if visible <= width {
        return (0..visible).collect();
    }
    let tail_start = visible / ratio * ratio;
    let scores: Vec<f32> = (0..visible)
        .map(|j| {
            if j >= tail_start {
                f32::INFINITY
            } else {
                block_scores[j / ratio]
            }
        })
        .collect();
    let mut chosen = super::top_k_indices(&scores, width);
    chosen.sort_unstable();
    chosen
}

impl NgramHash {
    /// How many predecessors the hash reaches back for.
    fn lookback(&self) -> usize {
        self.ngram_size - 1
    }

    /// The table rows every token of this batch gathers, `[n_tokens,
    /// n_heads]`.
    ///
    /// `history` is the tail of the tokens committed to the cache before
    /// this batch, oldest first — empty at the start of a sequence, and at
    /// most [`Self::lookback`] long. Predecessors are taken from the batch
    /// where the batch has them and from `history` otherwise, so a chunked
    /// prefill's seam and a decode step hash the same n-grams a single-shot
    /// prefill would.
    fn rows(&self, tokens: &[u32], history: &[u32]) -> Vec<usize> {
        let n_gram = self.ngram_size;
        let eos = self.eos;
        let mut rows = vec![0usize; tokens.len() * self.n_heads];
        for (i, &token) in tokens.iter().enumerate() {
            // Predecessor `s` (1-based) of token `i`: inside the batch when
            // it is there, otherwise the `s - i`-th entry back from the end
            // of the carried history, otherwise EOS.
            let prev = |s: usize| -> u32 {
                if let Some(j) = i.checked_sub(s) {
                    return tokens[j];
                }
                let back = s - i;
                history.len().checked_sub(back).map_or(eos, |k| history[k])
            };
            // An EOS in the window hides everything at or before it. The
            // token's *own* value never cuts its context: upstream takes
            // the last EOS strictly before this position, so a segment
            // boundary is only invisible to the positions after it.
            let mut ctx = vec![token as u64; n_gram];
            let mut cut = false;
            for (s, slot) in ctx.iter_mut().enumerate().skip(1) {
                let tok = if cut { eos } else { prev(s) };
                *slot = tok as u64;
                cut |= tok == eos;
            }
            for n in 2..=n_gram {
                let mut mixed = ctx[0].wrapping_mul(self.multipliers[0]);
                for (c, m) in ctx[1..n].iter().zip(self.multipliers[1..n].iter()) {
                    mixed ^= c.wrapping_mul(*m);
                }
                let base = (n - 2) * self.heads_per_ngram;
                for g in 0..self.heads_per_ngram {
                    let h = base + g;
                    rows[i * self.n_heads + h] =
                        (mixed % self.head_vocab_sizes[h] + self.head_offsets[h]) as usize;
                }
            }
        }
        rows
    }
}

impl Ple {
    fn load(loaded: &LoadedModel) -> Result<Option<Self>> {
        let Some(layers) = loaded.metadata_array_u64("ple.layers") else {
            return Ok(None);
        };
        if layers.is_empty() {
            return Ok(None);
        }
        let ngram_size = loaded
            .metadata_u64("ple.ngram_size")
            .context("missing ple.ngram_size")? as usize;
        let heads_per_ngram = loaded
            .metadata_u64("ple.heads_per_ngram")
            .context("missing ple.heads_per_ngram")? as usize;
        let head_dim = loaded
            .metadata_u64("embedding_length_per_layer_input")
            .context("missing embedding_length_per_layer_input")? as usize;
        anyhow::ensure!(
            ngram_size >= 2,
            "ple.ngram_size ({ngram_size}) must be at least 2"
        );
        let n_heads = (ngram_size - 1) * heads_per_ngram;
        anyhow::ensure!(n_heads > 0, "ple.heads_per_ngram must be nonzero");
        anyhow::ensure!(
            n_heads * head_dim == loaded.config.n_embd,
            "the PLE gather is {n_heads} heads of {head_dim}, which does not fill n_embd ({})",
            loaded.config.n_embd
        );
        let multipliers = loaded
            .metadata_array_u64("ple.layer_multipliers")
            .context("missing ple.layer_multipliers")?;
        let head_offsets = loaded
            .metadata_array_u64("ple.head_offsets")
            .context("missing ple.head_offsets")?;
        let head_vocab_sizes = loaded
            .metadata_array_u64("ple.head_vocab_sizes")
            .context("missing ple.head_vocab_sizes")?;
        anyhow::ensure!(
            multipliers.len() >= ngram_size,
            "ple.layer_multipliers has {} entries, need ple.ngram_size ({ngram_size})",
            multipliers.len()
        );
        anyhow::ensure!(
            head_offsets.len() >= n_heads && head_vocab_sizes.len() >= n_heads,
            "ple.head_offsets/head_vocab_sizes are shorter than the {n_heads} hash heads"
        );
        anyhow::ensure!(
            head_vocab_sizes.iter().all(|&v| v > 0),
            "a ple.head_vocab_sizes entry is zero, which the hash divides by"
        );
        let eos = loaded
            .metadata_u64("ple.eos_token_id")
            .context("missing ple.eos_token_id")? as u32;
        Ok(Some(Self {
            table: loaded
                .matrix("per_layer_token_embd.weight")
                .context("loading per_layer_token_embd.weight")?,
            head_dim,
            hash: NgramHash {
                n_heads,
                ngram_size,
                heads_per_ngram,
                multipliers,
                head_offsets,
                head_vocab_sizes,
                eos,
            },
        }))
    }

    /// The gathered embedding for each token, `[n_tokens, n_embd]` — the
    /// heads' rows concatenated head-major, which is the order
    /// `ggml_get_rows` produces for a `[n_heads * n_tokens]` index vector
    /// and the order the projections below expect.
    fn gather(&self, rows: &[usize], n_tokens: usize) -> Vec<f32> {
        let width = self.hash.n_heads * self.head_dim;
        let mut out = vec![0f32; n_tokens * width];
        out.par_chunks_mut(width)
            .zip(rows.par_chunks(self.hash.n_heads))
            .for_each(|(dst, token_rows)| {
                for (h, &row) in token_rows.iter().enumerate() {
                    dst[h * self.head_dim..(h + 1) * self.head_dim]
                        .copy_from_slice(&self.table.row(row));
                }
            });
        out
    }
}

impl Qwen4ExpModel {
    pub fn load_with_backend(loaded: &LoadedModel, backend: Arc<dyn Backend>) -> Result<Self> {
        let dims = Dims::from_loaded(loaded)?;
        let n_layer = trunk_layer_count(loaded)?;
        let is_recr = recurrent_layer_mask(loaded, n_layer);

        let hc = loaded
            .metadata_u64("hyper_connection.count")
            .context("missing hyper_connection.count")? as usize;
        anyhow::ensure!(hc > 0, "hyper_connection.count must be at least 1");
        loaded
            .metadata_u64("hyper_connection.low_rank")
            .context("missing hyper_connection.low_rank")?;

        let indexer_n_head = loaded
            .metadata_u64("attention.indexer.head_count")
            .context("missing attention.indexer.head_count")? as usize;
        let indexer_head_size = loaded
            .metadata_u64("attention.indexer.key_length")
            .context("missing attention.indexer.key_length")?
            as usize;
        let indexer_top_k = loaded
            .metadata_u64("attention.indexer.top_k")
            .context("missing attention.indexer.top_k")? as usize;
        anyhow::ensure!(
            indexer_n_head > 0 && indexer_head_size > 0 && indexer_top_k > 0,
            "attention.indexer.head_count/key_length/top_k must all be nonzero"
        );
        // Every full-attention layer in the released checkpoints declares a
        // ratio; a zero would be upstream's "this layer attends densely",
        // which is also what this engine does below `top_k + ratio - 1`
        // positions, so it is rejected rather than silently reinterpreted.
        let compress_ratios = loaded
            .metadata_array_u64("attention.compress_ratios")
            .context("missing attention.compress_ratios")?;

        let n_expert_used = loaded
            .metadata_u64("expert_used_count")
            .context("missing expert_used_count")? as usize;
        loaded
            .metadata_u64("expert_count")
            .context("missing expert_count")?;

        let ple = Ple::load(loaded)?;
        let ple_layers: Vec<usize> = loaded
            .metadata_array_u64("ple.layers")
            .unwrap_or_default()
            .iter()
            .map(|&v| v as usize)
            .collect();

        let tok_embeddings = loaded
            .matrix("token_embd.weight")
            .context("loading token_embd.weight")?;
        let output_weight = if loaded.has_tensor("output.weight") {
            loaded
                .matrix("output.weight")
                .context("loading output.weight")?
        } else {
            tok_embeddings.clone()
        };
        // There is no `output_norm.weight` in this architecture at all: the
        // head mixer's own grouped norm is the last one in the model.
        let head = HcMixer::load(loaded, "output_hc", false)?;

        let mut kv_dims: Vec<usize> = Vec::new();
        let mut recurrent_specs: Vec<RecurrentSpec> = Vec::new();
        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let t = LayerTensors { loaded, i };
            let mixer = if is_recr.get(i).copied().unwrap_or(false) {
                let cache_index = recurrent_specs.len();
                recurrent_specs.push(RecurrentSpec::delta_net(
                    dims.conv_channels(),
                    dims.ssm_d_conv,
                    dims.ssm_dt_rank,
                    dims.ssm_head_dim,
                ));
                Mixer::Recurrent(Recurrent::load(&t, &dims, cache_index)?)
            } else {
                let attn_slot = kv_dims.len();
                kv_dims.push(dims.n_head_kv * dims.head_dim);
                let key_slot = kv_dims.len();
                kv_dims.push(indexer_head_size);
                let ratio = compress_ratios.get(i).copied().unwrap_or(0) as usize;
                anyhow::ensure!(
                    ratio > 0,
                    "layer {i} is a full-attention layer with attention.compress_ratios[{i}] = 0"
                );
                Mixer::FullAttn {
                    attn: FullAttn::load(&t, attn_slot)?,
                    indexer: Indexer {
                        q_proj: t.matrix("indexer.q_proj.weight")?,
                        k_proj: t.matrix("indexer.k_proj.weight")?,
                        q_norm: t.vec("indexer.q_norm.weight")?,
                        k_norm: t.vec("indexer.k_norm.weight")?,
                        ratio,
                        cache_index: key_slot,
                    },
                }
            };

            let ple_layer = match (ple.as_ref(), ple_layers.contains(&i)) {
                (Some(p), true) => {
                    let kernel = loaded
                        .metadata_u64("ple.conv_kernel")
                        .context("missing ple.conv_kernel")?
                        as usize;
                    anyhow::ensure!(kernel > 0, "ple.conv_kernel must be nonzero");
                    let hc_dim = hc * dims.n_embd;
                    let raw = t.vec("ple_conv1d.weight")?;
                    anyhow::ensure!(
                        raw.len() == hc_dim * kernel,
                        "blk.{i}.ple_conv1d.weight is {} values, expected {hc_dim} channels of {kernel}",
                        raw.len()
                    );
                    let cache_index = recurrent_specs.len();
                    // A convolution and nothing else: `num_heads == 0`
                    // allocates the rolling history without a state matrix.
                    recurrent_specs.push(RecurrentSpec::delta_net(
                        hc_dim,
                        (kernel - 1) * p.hash.ngram_size + 1,
                        0,
                        0,
                    ));
                    Some(PleLayer {
                        key: t.matrix("ple_key.weight")?,
                        value: t.matrix("ple_value.weight")?,
                        norm_key: t.vec("ple_norm_key.weight")?,
                        norm_query: t.vec("ple_norm_query.weight")?,
                        norm_conv: t.vec("ple_norm_conv.weight")?,
                        conv1d: expand_dilated_kernel(&raw, hc_dim, kernel, p.hash.ngram_size),
                        cache_index,
                    })
                }
                _ => None,
            };

            layers.push(Qwen4ExpLayer {
                hc_attn: HcMixer::load(loaded, &format!("blk.{i}.hc_attn"), true)?,
                hc_ffn: HcMixer::load(loaded, &format!("blk.{i}.hc_ffn"), true)?,
                mixer,
                ple: ple_layer,
                ffn: MoeFfn::load(&t, n_expert_used)?,
            });
        }

        Ok(Self {
            config: loaded.config.clone(),
            backend,
            dims,
            hc,
            indexer_head_size,
            indexer_n_head,
            indexer_top_k,
            tok_embeddings,
            output_weight,
            head,
            ple,
            layers,
            kv_dims,
            recurrent_specs,
        })
    }

    /// The hyper-connection in-mix: normalizes the streams, gates them, and
    /// collapses them to the one `[n_tokens, n_embd]` vector the sub-layer
    /// reads. When the mixer has an injection projection, the `[n_tokens,
    /// hc]` scatter weights [`Self::hc_combine`] needs come back with it.
    ///
    /// `x` is `[n_tokens, hc, n_embd]`, streams contiguous within a token.
    fn hc_mix(&self, mixer: &HcMixer, x: &[f32], n_tokens: usize) -> (Vec<f32>, Option<Vec<f32>>) {
        let n_embd = self.dims.n_embd;
        let hc = self.hc;
        let hc_dim = hc * n_embd;

        // Grouped RMSNorm: the norm is taken over one stream at a time, and
        // the gamma that follows spans all of them.
        let mut normed = Vec::new();
        super::rms_norm_rows_into(&mut normed, x, n_embd, self.dims.rms_eps);
        for row in normed.chunks_mut(hc_dim) {
            tensor::mul_inplace(row, &mixer.norm);
        }

        let mut low =
            super::matmul_host_fallback(self.backend.as_ref(), &normed, n_tokens, &mixer.down);
        let inv_hc = 1.0 / hc as f32;
        for v in low.iter_mut() {
            *v = tensor::silu(*v * inv_hc);
        }
        let mut gate =
            super::matmul_host_fallback(self.backend.as_ref(), &low, n_tokens, &mixer.up);
        for v in gate.iter_mut() {
            *v = tensor::sigmoid(*v);
        }
        tensor::mul_inplace(&mut gate, &normed);
        let gated = gate;

        // Collapse the streams by their mean.
        let mut mixed = vec![0f32; n_tokens * n_embd];
        for (t, dst) in mixed.chunks_mut(n_embd).enumerate() {
            for c in 0..hc {
                let src = &gated[(t * hc + c) * n_embd..(t * hc + c + 1) * n_embd];
                tensor::axpy_inplace(dst, src, inv_hc);
            }
        }

        let inject = mixer
            .inject
            .as_ref()
            .map(|w| super::matmul_host_fallback(self.backend.as_ref(), &normed, n_tokens, w));
        (mixed, inject)
    }

    /// The hyper-connection out-mix: adds this sub-layer's output back into
    /// every stream, weighted per stream.
    ///
    /// `2 * sigmoid(inject / hc)` centres the weights on 1, so an untrained
    /// injection matrix reproduces the plain residual add this replaces.
    fn hc_combine(&self, x: &mut [f32], sub_out: &[f32], inject: &[f32], n_tokens: usize) {
        let n_embd = self.dims.n_embd;
        let hc = self.hc;
        for t in 0..n_tokens {
            let out_t = &sub_out[t * n_embd..(t + 1) * n_embd];
            for c in 0..hc {
                let w = hc_scatter_weight(inject[t * hc + c], hc);
                let dst = &mut x[(t * hc + c) * n_embd..(t * hc + c + 1) * n_embd];
                tensor::axpy_inplace(dst, out_t, w);
            }
        }
    }

    /// Which cached positions each token of this batch may attend, or
    /// `None` when every one of them can see everything it could anyway.
    ///
    /// The indexer's raw keys for this batch are cached here, as they must
    /// be before any of its queries can score the block they complete.
    fn indexer_selection(
        &self,
        indexer: &Indexer,
        cache: &mut KvCache,
        cur: &[f32],
        n_tokens: usize,
        start_pos: usize,
    ) -> Option<Vec<Vec<usize>>> {
        let dim = self.indexer_head_size;
        let n_head = self.indexer_n_head;
        let ratio = indexer.ratio;
        let eps = self.dims.rms_eps;
        let rope_dim = self.dims.rope_dim;
        let rope_base = self.dims.rope_freq_base;

        let keys = self.backend.matmul(cur, n_tokens, &indexer.k_proj);
        {
            let slot = &mut cache.layers[indexer.cache_index];
            for t in 0..n_tokens {
                let row = &keys[t * dim..(t + 1) * dim];
                slot.push(row, row);
            }
        }

        let n_pos = start_pos + n_tokens;
        // The widest key set the layer will ever gather. Below it the top-k
        // is every visible position, and the mask it would produce is the
        // causal mask the attention applies anyway — so the batch skips the
        // indexer outright and takes the dense (GPU-capable) path.
        let width = self.indexer_top_k + ratio - 1;
        if n_pos <= width {
            return None;
        }

        // One block key per *complete* block of `ratio` positions: the mean
        // of its members' raw keys, then normed, then rotated at the
        // block's own first position. Built once for the batch rather than
        // once per token; every query below reads a prefix of it.
        //
        // Recomputed each call rather than cached alongside the raw keys: a
        // strided cache slot would hold whole blocks only, which would put
        // this architecture's *recurrent* state up to `ratio - 1` positions
        // ahead of what the attention slots report as committed, and prefix
        // reuse reads that number. The pooling is `n_pos / ratio` short
        // vector sums against a step that reads every routed expert it
        // selected, so keeping the two lengths honest is worth more here
        // than the arithmetic it saves.
        let n_blocks = n_pos / ratio;
        let slot = &cache.layers[indexer.cache_index];
        let mut blocks = vec![0f32; n_blocks * dim];
        blocks.par_chunks_mut(dim).enumerate().for_each(|(b, dst)| {
            for p in b * ratio..(b + 1) * ratio {
                tensor::axpy_inplace(dst, slot.key_at(p, 0, dim), 1.0 / ratio as f32);
            }
            tensor::rmsnorm_inplace(dst, &indexer.k_norm, 1, dim, eps);
            tensor::rope_apply_inplace(dst, 1, dim, rope_dim, b * ratio, rope_base);
        });

        let mut queries = self.backend.matmul(cur, n_tokens, &indexer.q_proj);
        tensor::rmsnorm_inplace(&mut queries, &indexer.q_norm, n_tokens * n_head, dim, eps);
        for t in 0..n_tokens {
            tensor::rope_apply_inplace(
                &mut queries[t * n_head * dim..(t + 1) * n_head * dim],
                n_head,
                dim,
                rope_dim,
                start_pos + t,
                rope_base,
            );
        }

        Some(
            (0..n_tokens)
                .into_par_iter()
                .map(|t| {
                    let pos = start_pos + t;
                    let visible = pos + 1;
                    if visible <= width {
                        return (0..visible).collect();
                    }
                    // The incomplete block at the end is always attended,
                    // which is what lands the cut on a block boundary.
                    let scored = visible / ratio;
                    let q_t = &queries[t * n_head * dim..(t + 1) * n_head * dim];
                    // Parallel over blocks, not just over tokens: a decode
                    // step is one token and thousands of blocks.
                    let block_scores: Vec<f32> = (0..scored)
                        .into_par_iter()
                        .map(|b| {
                            let key = &blocks[b * dim..(b + 1) * dim];
                            (0..n_head)
                                .map(|h| tensor::dot(&q_t[h * dim..(h + 1) * dim], key).max(0.0))
                                .sum()
                        })
                        .collect();
                    select_positions(&block_scores, visible, ratio, width)
                })
                .collect(),
        )
    }

    /// The per-layer embedding injection: gathers each token's n-gram hash
    /// rows, gates a value against the residual streams, runs the dilated
    /// causal convolution over the result, and adds both back.
    fn forward_ple(
        &self,
        ple: &Ple,
        layer: &PleLayer,
        cache: &mut KvCache,
        x: &mut [f32],
        tokens: &[u32],
    ) {
        let n_embd = self.dims.n_embd;
        let hc = self.hc;
        let hc_dim = hc * n_embd;
        let eps = self.dims.rms_eps;
        let n_tokens = tokens.len();

        let rows = ple.hash.rows(tokens, &cache.recent_tokens);
        let embd = ple.gather(&rows, n_tokens);

        let key = super::matmul_host_fallback(self.backend.as_ref(), &embd, n_tokens, &layer.key);
        let value =
            super::matmul_host_fallback(self.backend.as_ref(), &embd, n_tokens, &layer.value);

        // Both norms are grouped over one stream with a gamma spanning all
        // of them, exactly as in `hc_mix`.
        let grouped_norm = |src: &[f32], w: &[f32]| -> Vec<f32> {
            let mut out = Vec::new();
            super::rms_norm_rows_into(&mut out, src, n_embd, eps);
            for row in out.chunks_mut(hc_dim) {
                tensor::mul_inplace(row, w);
            }
            out
        };
        let key = grouped_norm(&key, &layer.norm_key);
        let query = grouped_norm(x, &layer.norm_query);

        // A per-stream dot product, then a signed square root before the
        // sigmoid.
        let scale = 1.0 / (n_embd as f32).sqrt();
        let mut gated = vec![0f32; n_tokens * hc_dim];
        for t in 0..n_tokens {
            for c in 0..hc {
                let at = (t * hc + c) * n_embd;
                let s = tensor::dot(&key[at..at + n_embd], &query[at..at + n_embd]) * scale;
                let gate = ple_gate(s);
                let dst = &mut gated[at..at + n_embd];
                for (o, &v) in dst.iter_mut().zip(&value[t * n_embd..(t + 1) * n_embd]) {
                    *o = v * gate;
                }
            }
        }

        let normed = grouped_norm(&gated, &layer.norm_conv);
        let state = &mut cache.recurrent[layer.cache_index];
        for t in 0..n_tokens {
            let mut conv = state.conv_step(&normed[t * hc_dim..(t + 1) * hc_dim], &layer.conv1d);
            for (o, (&g, c)) in x[t * hc_dim..(t + 1) * hc_dim].iter_mut().zip(
                gated[t * hc_dim..(t + 1) * hc_dim]
                    .iter()
                    .zip(conv.iter_mut()),
            ) {
                *o += g + tensor::silu(*c);
            }
        }
    }
}

impl ModelForward for Qwen4ExpModel {
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
        KvCache::new_mixed(capacity, &self.kv_dims, &self.recurrent_specs)
    }

    fn forward(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        _slot_id: usize,
    ) -> Result<Vec<f32>> {
        let n_tokens = tokens.len();
        let n_embd = self.dims.n_embd;
        let hc = self.hc;
        let hc_dim = hc * n_embd;

        // The wide residual starts as `hc` identical copies of the token
        // embedding.
        let mut x = vec![0f32; n_tokens * hc_dim];
        for (t, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            anyhow::ensure!(
                tok < self.config.n_vocab,
                "token id {tok} is out of vocab range"
            );
            let embd = self.tok_embeddings.row(tok);
            for c in 0..hc {
                let at = (t * hc + c) * n_embd;
                x[at..at + n_embd].copy_from_slice(&embd);
            }
        }

        for layer in &self.layers {
            if let (Some(ple), Some(layer_ple)) = (self.ple.as_ref(), layer.ple.as_ref()) {
                self.forward_ple(ple, layer_ple, cache, &mut x, tokens);
            }

            let (cur, inject) = self.hc_mix(&layer.hc_attn, &x, n_tokens);
            let inject = inject.expect("a per-layer mixer always carries its injection weights");
            let sub_out = match &layer.mixer {
                Mixer::FullAttn { attn, indexer } => {
                    let selection =
                        self.indexer_selection(indexer, cache, &cur, n_tokens, start_pos);
                    attn.forward(
                        self.backend.as_ref(),
                        &self.dims,
                        cache,
                        &cur,
                        n_tokens,
                        start_pos,
                        selection.as_deref(),
                    )
                }
                Mixer::Recurrent(recurrent) => recurrent.forward(
                    self.backend.as_ref(),
                    &self.dims,
                    cache,
                    &cur,
                    n_tokens,
                    OutputGate::Sigmoid,
                ),
            };
            self.hc_combine(&mut x, &sub_out, &inject, n_tokens);

            let (cur, inject) = self.hc_mix(&layer.hc_ffn, &x, n_tokens);
            let inject = inject.expect("a per-layer mixer always carries its injection weights");
            let ffn_out = {
                use qwen_hybrid::HybridFfn as _;
                layer
                    .ffn
                    .forward(self.backend.as_ref(), n_embd, &cur, n_tokens)
            };
            self.hc_combine(&mut x, &ffn_out, &inject, n_tokens);
        }

        // Only the last position's logits are wanted, and the head mixer is
        // per token, so only that token's streams are collapsed.
        let last = &x[(n_tokens - 1) * hc_dim..];
        let (out, _) = self.hc_mix(&self.head, last, 1);

        if let Some(ple) = self.ple.as_ref() {
            for &tok in tokens {
                cache.push_recent_token(tok, ple.hash.lookback());
            }
        }
        Ok(self.backend.matmul(&out, 1, &self.output_weight))
    }

    fn forward_hidden_states(&self, _tokens: &[u32]) -> Result<Vec<f32>> {
        anyhow::bail!("embeddings are not yet supported for Qwen4-preview models")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The dilation has to end up *in the kernel*, because the stepper that
    /// consumes it has no notion of one. Reading the taps back at the wrong
    /// stride is not a crash and not obviously wrong output — it is a
    /// convolution over the wrong three predecessors — so the placement is
    /// asserted rather than assumed.
    #[test]
    fn a_dilated_kernel_places_its_taps_at_the_dilation_stride() {
        // Two channels, four taps, dilation 3 => nine zeros' worth of
        // reach, ten taps total.
        let w = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let out = expand_dilated_kernel(&w, 2, 4, 3);
        assert_eq!(out.len(), 2 * 10);
        // `conv_step`'s last tap is the current token, and the current
        // token is the *last* tap of the original kernel.
        assert_eq!(
            &out[..10],
            &[1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0]
        );
        assert_eq!(
            &out[10..],
            &[10.0, 0.0, 0.0, 20.0, 0.0, 0.0, 30.0, 0.0, 0.0, 40.0]
        );
    }

    /// A dilation of 1 is the undilated kernel unchanged — the property
    /// that makes this a re-layout rather than a reinterpretation.
    #[test]
    fn an_undilated_kernel_is_copied_verbatim() {
        let w = vec![1.0, 2.0, 3.0];
        assert_eq!(expand_dilated_kernel(&w, 1, 3, 1), w);
    }

    /// An untrained (all-zero) injection projection has to reproduce the
    /// plain residual add the hyper-connection generalizes. If the factor
    /// of two were dropped the whole model would run at half residual gain,
    /// which converges to fluent, wrong output rather than to an error.
    #[test]
    fn a_zero_injection_scatters_the_plain_residual_add() {
        for hc in [1usize, 2, 4] {
            assert!((hc_scatter_weight(0.0, hc) - 1.0).abs() < 1e-6);
        }
    }

    /// The gate is odd about zero and squashed into (0, 1) — a sign error
    /// in the signed square root would flip which per-layer embeddings are
    /// let through.
    #[test]
    fn the_ple_gate_is_a_signed_root_around_a_half() {
        assert!((ple_gate(0.0) - 0.5).abs() < 1e-3);
        assert!(ple_gate(4.0) > 0.85);
        assert!(ple_gate(-4.0) < 0.15);
        // sigmoid(sqrt(4)) and sigmoid(-sqrt(4)) are symmetric about 0.5.
        assert!((ple_gate(4.0) + ple_gate(-4.0) - 1.0).abs() < 1e-6);
    }

    /// Below the budget the indexer cannot change the answer: its top-k is
    /// every visible position, and the mask that produces is the causal
    /// mask attention applies anyway. The dense path is taken there, so
    /// this is the contract that lets `indexer_selection` return `None`.
    #[test]
    fn a_short_context_selects_every_visible_position() {
        assert_eq!(select_positions(&[], 5, 4, 8), vec![0, 1, 2, 3, 4]);
        assert_eq!(
            select_positions(&[1.0, 2.0], 8, 4, 8),
            (0..8).collect::<Vec<_>>()
        );
    }

    /// The incomplete tail is always attended, whatever the blocks score,
    /// and the rest of the budget goes to whole high-scoring blocks.
    #[test]
    fn the_tail_is_always_attended_and_the_budget_buys_whole_blocks() {
        // 14 visible positions, blocks of 4: blocks 0..3 are complete
        // (positions 0-11), positions 12-13 are the tail. A budget of 6
        // buys the tail plus one block.
        let chosen = select_positions(&[0.5, 9.0, 0.25, 1.0], 14, 4, 6);
        assert_eq!(chosen, vec![4, 5, 6, 7, 12, 13]);
    }

    /// Ties between equally-scoring blocks resolve towards the earlier
    /// position, so a selection is reproducible across runs rather than
    /// depending on how a sort happened to order equal keys.
    #[test]
    fn equal_block_scores_break_towards_the_earlier_position() {
        let chosen = select_positions(&[1.0, 1.0, 1.0], 13, 4, 5);
        assert_eq!(chosen, vec![0, 1, 2, 3, 12]);
    }

    fn test_hash(ngram_size: usize, heads_per_ngram: usize, eos: u32) -> NgramHash {
        let n_heads = (ngram_size - 1) * heads_per_ngram;
        NgramHash {
            n_heads,
            ngram_size,
            heads_per_ngram,
            multipliers: vec![3, 5, 7, 11],
            head_offsets: (0..n_heads as u64).map(|h| h * 1000).collect(),
            head_vocab_sizes: vec![1000; n_heads],
            eos,
        }
    }

    /// Each n-gram width owns its own heads, and each head indexes its own
    /// slice of the shared table. Folding the wrong multiplier in, or
    /// dropping a head's offset, gathers a real row of a 320-million-row
    /// table — a plausible embedding for the wrong n-gram, which no shape
    /// check downstream can see.
    #[test]
    fn the_ngram_hash_offsets_every_head_into_its_own_slice() {
        let ple = test_hash(3, 1, 99);
        // One token, no history: both predecessors are EOS.
        let rows = ple.rows(&[7], &[]);
        assert_eq!(rows.len(), 2);
        let bigram = (7u64 * 3) ^ (99u64 * 5);
        let trigram = bigram ^ (99u64 * 7);
        assert_eq!(rows[0], (bigram % 1000) as usize);
        assert_eq!(rows[1], (trigram % 1000 + 1000) as usize);
    }

    /// A token near the start of a batch has to reach into the tokens the
    /// cache already committed, or a chunked prefill would hash different
    /// n-grams than a single-shot one for the same prompt — the same text,
    /// two different answers, depending only on `ORANGU_PREFILL_BATCH`.
    #[test]
    fn predecessors_come_from_the_carried_history_at_a_batch_seam() {
        let ple = test_hash(3, 1, 99);
        let whole = ple.rows(&[1, 2, 3, 4], &[]);
        let split_tail = ple.rows(&[3, 4], &[1, 2]);
        assert_eq!(&whole[2 * 2..], &split_tail[..]);
    }

    /// Only as far back as the hash reaches: a history longer than the
    /// lookback is read from its *end*, not its start.
    #[test]
    fn a_longer_history_is_read_from_its_most_recent_end() {
        let ple = test_hash(3, 1, 99);
        assert_eq!(ple.rows(&[9], &[1, 2]), ple.rows(&[9], &[7, 8, 1, 2]));
    }

    /// An EOS hides everything at or before it, but never hides the token
    /// carrying it: upstream cuts on the last EOS strictly *before* the
    /// position being hashed.
    #[test]
    fn an_eos_cuts_the_context_before_it_but_not_the_token_itself() {
        let ple = test_hash(3, 1, 99);
        // Token 5 preceded by [4, EOS]: the EOS at distance 1 hides the 4
        // at distance 2, so both predecessors read as EOS.
        assert_eq!(ple.rows(&[5], &[4, 99]), ple.rows(&[5], &[]));
        // The EOS token itself still hashes as its own value.
        let eos_rows = ple.rows(&[99], &[]);
        let bigram = (99u64 * 3) ^ (99u64 * 5);
        assert_eq!(eos_rows[0], (bigram % 1000) as usize);
    }

    /// Every head of one n-gram width shares that width's hash, and the
    /// heads are laid out width-major — the order the gather below relies
    /// on to fill `n_embd` in the order the projections were trained for.
    #[test]
    fn heads_of_one_ngram_width_share_a_hash_and_sit_together() {
        let ple = test_hash(3, 4, 99);
        let rows = ple.rows(&[7], &[]);
        assert_eq!(rows.len(), 8);
        // Heads 0..4 are the bigram's, 4..8 the trigram's; within a width
        // they differ only by their own offset.
        for h in 0..4 {
            assert_eq!(rows[h] % 1000, rows[0] % 1000);
            assert_eq!(rows[h] / 1000, h);
        }
        for h in 4..8 {
            assert_eq!(rows[h] % 1000, rows[4] % 1000);
            assert_eq!(rows[h] / 1000, h);
        }
        assert_ne!(rows[0] % 1000, rows[4] % 1000);
    }
}

#[cfg(test)]
mod real_model_tests {
    use super::*;

    /// Cross-check against the real checkpoint: given the token ids for
    /// "The capital of France is", the model should predict " Paris".
    ///
    /// This is the only test that can catch the whole class of mistakes
    /// this architecture invites — a hyper-connection mixed with the wrong
    /// mean, an indexer scoring blocks at the wrong positions, a PLE hash
    /// off by one multiplier — because every one of them produces a
    /// well-formed vector of the right length and fluent, wrong text. No
    /// shape check downstream sees any of them.
    ///
    /// Run with `ORANGU_TEST_QWEN4EXP_MODEL=/path/to/first-shard.gguf cargo
    /// test --release --bin orangu-server qwen4exp::real_model_tests --
    /// --ignored`. Expect minutes, not seconds: this is a 125B-parameter
    /// MoE read through a scalar host path.
    #[test]
    #[ignore]
    fn qwen4exp_predicts_paris_after_capital_of_france() {
        let path = std::env::var("ORANGU_TEST_QWEN4EXP_MODEL")
            .expect("set ORANGU_TEST_QWEN4EXP_MODEL to a Qwen3.8-Flash-Next GGUF");
        let loaded = LoadedModel::open(std::path::Path::new(&path)).expect("load model");
        assert_eq!(loaded.config.architecture, "qwen4exp");
        let gguf = orangu::gguf::GgufFile::open(std::path::Path::new(&path)).expect("open gguf");
        let tokenizer =
            crate::engine::tokenizer::Tokenizer::from_gguf(&gguf).expect("build tokenizer");
        // No BOS: this file sets `tokenizer.ggml.add_bos_token = 0`.
        let tokens = tokenizer.encode("The capital of France is", false);
        let model =
            Qwen4ExpModel::load_with_backend(&loaded, Arc::new(crate::engine::backend::CpuBackend))
                .expect("build model");

        let mut cache = model.new_kv_cache(64);
        let logits = model.forward(&mut cache, &tokens, 0, 0).expect("forward");
        let (top_id, _) = logits
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        assert_eq!(
            tokenizer.decode(&[top_id as u32]).trim(),
            "Paris",
            "expected ' Paris' as the top prediction"
        );
    }
}
