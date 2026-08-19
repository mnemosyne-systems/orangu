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

//! GLM with DeepSeek sparse attention (`general.architecture = "glm-dsa"`),
//! e.g. `unsloth/GLM-5.2-GGUF`.
//!
//! The block shape is a plain pre-norm transformer — attention, residual,
//! FFN, residual — and the FFN is the same routed-experts-plus-one-shared-
//! expert MoE `qwen35moe` and `deepseek4` already build, with the first
//! `leading_dense_block_count` layers dense instead. Two things are new:
//!
//! - **MLA (multi-head latent attention), in its absorbed form.** The keys
//!   and values of a whole layer are one `attention.kv_lora_rank`-wide
//!   compressed vector per token plus a shared `rope.dimension_count`-wide
//!   rotary part — 576 floats for GLM-5.2, for all 64 heads together,
//!   rather than 64 separate key and value heads. Instead of decompressing
//!   that back to per-head keys, the query is pushed *through* the key
//!   decompression matrix (`attn_k_b`, per head) so it can be dotted with
//!   the compressed vector directly, and the attention output is pushed
//!   through the value decompression matrix (`attn_v_b`) afterwards. That
//!   is what makes the KV cache small enough for a 79-layer model; it also
//!   means the cache is K-only, since the value is the leading
//!   `kv_lora_rank` of the same row.
//! - **The lightning indexer.** A small separate attention (32 heads of
//!   128, its own per-token key cache) scores every earlier position, and
//!   the real attention attends only the `attention.indexer.top_k` best.
//!   Only some layers run it: GLM-5.2's first three and then every fourth,
//!   with the layers in between reusing the previous scoring layer's
//!   choice (`attention.indexer.types`, defaulted from the reference
//!   config when the file doesn't carry it).
//!
//! Transcribed from upstream `llama.cpp`'s `src/models/glm-dsa.cpp` and the
//! DSA-specific `llm_graph_context::build_attn` overload that turns the
//! indexer's top-k into an attention mask. Two upstream details are
//! deliberately not carried over, for the same reasons as in
//! `engine::arch::deepseek4`: the Hadamard rotation of the indexer's
//! queries and keys (orthonormal, self-inverse, applied to both sides, so
//! it changes no dot product — it exists there for quantized caches), and
//! the multi-token-prediction block (`blk.78` here), which is a draft head
//! this engine has no second-model speculative path for.

use anyhow::{Context, Result};
use rayon::prelude::*;
use std::sync::Arc;

use super::{ExpertGating, ExpertRouting, ModelForward, attend, top_k_indices};
use crate::engine::backend::Backend;
use crate::engine::kv_cache::KvCache;
use crate::engine::loader::{ExpertQuantMatrix, LoadedModel, ModelConfig, QuantMatrix};
use crate::engine::tensor::{self, RopeLayout, RopeParams};

/// `attention.indexer.types` for GLM-5.2, whose GGUFs don't carry the key:
/// layers 0, 1 and 2 score, then every fourth layer after that, with the
/// three layers in between reusing the last scored selection. Upstream's
/// `GLM_5_2_DEFAULT_INDEXER_TYPES`, which reads it from the reference
/// `config.json`.
fn default_indexer_is_full(layer: usize) -> bool {
    layer < 2 || (layer - 2).is_multiple_of(4)
}

/// The lightning indexer of a scoring layer: its own query/key projections,
/// key normalization and per-head score weights, plus the cache slot its
/// keys live in.
struct Indexer {
    /// `[q_lora_rank, indexer.head_count * indexer.key_length]` — shares
    /// the attention query's LoRA intermediate.
    q_b: QuantMatrix,
    /// `[n_embd, indexer.key_length]`: one key per token, all heads.
    attn_k: QuantMatrix,
    /// LayerNorm (not RMS — this is the one normalization in the model
    /// that subtracts a mean and adds a bias) over the indexer key.
    k_norm_weight: Vec<f32>,
    k_norm_bias: Vec<f32>,
    /// `[n_embd, indexer.head_count]` — the per-head score weights.
    proj: QuantMatrix,
    cache_slot: usize,
}

/// A layer's FFN: dense for the leading blocks, routed MoE for the rest.
enum Ffn {
    Dense {
        gate: QuantMatrix,
        up: QuantMatrix,
        down: QuantMatrix,
    },
    /// Boxed: a `Moe` is nearly three times the size of the dense arm,
    /// and every layer of a 79-layer model carries one `Ffn`.
    Moe(Box<Moe>),
}

struct Moe {
    gate_inp: QuantMatrix,
    /// `exp_probs_b.bias` — steers the expert *selection* only.
    exp_probs_b: Option<Vec<f32>>,
    gate_exps: ExpertQuantMatrix,
    down_exps: ExpertQuantMatrix,
    up_exps: ExpertQuantMatrix,
    gate_shexp: QuantMatrix,
    down_shexp: QuantMatrix,
    up_shexp: QuantMatrix,
}

struct GlmLayer {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    wq_a: QuantMatrix,
    q_a_norm: Vec<f32>,
    /// `[q_lora_rank, n_head * key_length_mla]`.
    wq_b: QuantMatrix,
    /// `[n_embd, kv_lora_rank + rope_dim]` — the whole layer's keys and
    /// values for every head, compressed.
    wkv_a_mqa: QuantMatrix,
    kv_a_norm: Vec<f32>,
    /// `[key_length_mla - rope_dim, kv_lora_rank, n_head]` — per head, the
    /// matrix that decompresses a compressed KV vector into that head's
    /// non-rotary key. Used the other way round here: it absorbs the query.
    wk_b: ExpertQuantMatrix,
    /// `[kv_lora_rank, value_length_mla, n_head]` — per head, the matrix
    /// that decompresses the attention output back to a real value head.
    wv_b: ExpertQuantMatrix,
    wo: QuantMatrix,
    /// `Some` on a layer that scores positions itself; `None` on one that
    /// reuses the previous scoring layer's choice.
    indexer: Option<Indexer>,
    /// Where this layer's compressed keys live in `KvCache::layers`.
    kv_slot: usize,
    ffn: Ffn,
}

pub struct GlmModel {
    config: ModelConfig,
    backend: Arc<dyn Backend>,
    tok_embeddings: QuantMatrix,
    output_norm: Vec<f32>,
    output_weight: QuantMatrix,
    /// `attention.kv_lora_rank`: the width of one token's compressed KV.
    kv_lora_rank: usize,
    /// `attention.key_length_mla` / `attention.value_length_mla`: the head
    /// widths the model would have if the compression were undone.
    head_k_mla: usize,
    head_v_mla: usize,
    /// Width of one cached key row: `kv_lora_rank + rope_dim`. The value is
    /// its leading `kv_lora_rank`.
    kv_row: usize,
    rope: RopeParams,
    /// The attention softmax scale, which is **not** `1/sqrt(kv_row)`: it
    /// follows the pre-decompression head width, plus YaRN's magnitude
    /// correction squared. See [`GlmModel::load_with_backend`].
    kq_scale: f32,
    routing: ExpertRouting,
    indexer_n_head: usize,
    indexer_head_size: usize,
    indexer_top_k: usize,
    /// LayerNorm epsilon for the indexer key norm — upstream leaves
    /// `f_norm_eps` at its default for this architecture, since the GGUF
    /// carries only the RMS one.
    norm_eps: f32,
    kv_dims: Vec<(usize, usize)>,
    layers: Vec<GlmLayer>,
}

/// LayerNorm over one row: `(x - mean)/sqrt(var + eps) * weight + bias`.
/// ggml's `ggml_norm` followed by the weight and bias `build_norm` applies
/// for `LLM_NORM` — distinct from the RMSNorm every other normalization in
/// this model uses, which neither centers nor shifts.
fn layer_norm_inplace(x: &mut [f32], weight: &[f32], bias: &[f32], eps: f32) {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let scale = 1.0 / (var + eps).sqrt();
    for ((v, &w), &b) in x.iter_mut().zip(weight.iter()).zip(bias.iter()) {
        *v = (*v - mean) * scale * w + b;
    }
}

/// The buffers every layer of a forward pass reuses, so the trunk allocates
/// once per pass rather than three times per layer.
#[derive(Default)]
struct LayerScratch {
    /// Per-token expert picks, reused by the MoE router.
    selection: Vec<Vec<usize>>,
    /// The normed residual stream — what used to be `x.to_vec()`, twice a
    /// layer. See `tensor::rmsnorm_into`.
    normed: Vec<f32>,
    /// The feed-forward output, and the gate/up intermediates behind it —
    /// see `Backend::matmul_into`.
    ffn_out: Vec<f32>,
    ffn: super::FfnScratch,
    /// The attention half's per-layer projections.
    attn: AttnScratch,
}

/// [`LayerScratch`]'s attention half — the MLA projections and the output
/// projection, each `[n_tokens, ..]` and each run once per layer.
///
/// The lightning indexer's own three projections stay allocating:
/// `indexer_inputs` hands back an owned triple that the caller destructures,
/// and this architecture has no local reference model to check a
/// restructuring against.
#[derive(Default)]
struct AttnScratch {
    /// The query LoRA intermediate, shared with the indexer's query.
    qr: Vec<f32>,
    q: Vec<f32>,
    kv: Vec<f32>,
    out: Vec<f32>,
}

impl GlmModel {
    pub fn load_with_backend(loaded: &LoadedModel, backend: Arc<dyn Backend>) -> Result<Self> {
        let n_head = loaded.config.n_head;
        let rope_dim = loaded.config.rope_dim;

        // `block_count` counts the multi-token-prediction blocks too; the
        // trunk is everything before them. Running one as a trunk layer
        // would be silently wrong rather than a load failure — its tensors
        // are all present and correctly shaped.
        let n_layer_nextn = loaded.metadata_u64("nextn_predict_layers").unwrap_or(0) as usize;
        anyhow::ensure!(
            n_layer_nextn < loaded.config.n_layer,
            "nextn_predict_layers ({n_layer_nextn}) must be fewer than block_count ({})",
            loaded.config.n_layer
        );
        let n_layer = loaded.config.n_layer - n_layer_nextn;

        let kv_lora_rank = loaded
            .metadata_u64("attention.kv_lora_rank")
            .context("missing attention.kv_lora_rank")? as usize;
        let head_k_mla = loaded
            .metadata_u64("attention.key_length_mla")
            .context("missing attention.key_length_mla (this build requires MLA)")?
            as usize;
        let head_v_mla = loaded
            .metadata_u64("attention.value_length_mla")
            .context("missing attention.value_length_mla (this build requires MLA)")?
            as usize;
        anyhow::ensure!(
            head_k_mla > rope_dim,
            "attention.key_length_mla ({head_k_mla}) must exceed rope.dimension_count ({rope_dim})"
        );
        let n_layer_dense_lead = loaded
            .metadata_u64("leading_dense_block_count")
            .unwrap_or(0) as usize;
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
            indexer_head_size > rope_dim,
            "attention.indexer.key_length ({indexer_head_size}) must exceed rope.dimension_count ({rope_dim})"
        );

        let n_expert_used = loaded
            .metadata_u64("expert_used_count")
            .context("missing expert_used_count")? as usize;
        let gating = ExpertGating::from_gguf(
            loaded
                .metadata_u64("expert_gating_func")
                .context("missing expert_gating_func")?,
        )?;
        let expert_groups = super::ExpertGroups::from_gguf(
            loaded,
            loaded.metadata_u64("expert_count").unwrap_or(0) as usize,
        )?;

        // YaRN, when a file uses it, scales the attention logits as well as
        // the frequencies: upstream folds `mscale^2` into the softmax
        // scale. GLM-5.2 declares no rope scaling at all, so every term
        // below is 1 and this reduces to `1/sqrt(key_length_mla)` — note
        // that width, not the wider `kv_lora_rank + rope_dim` the absorbed
        // query and key actually are.
        let is_yarn = loaded.metadata_string("rope.scaling.type").as_deref() == Some("yarn");
        let freq_scale = loaded
            .metadata_f32("rope.scaling.factor")
            .map_or(1.0, |f| 1.0 / f);
        let ext_factor = if is_yarn { 1.0 } else { 0.0 };
        let yarn_log_mul = loaded
            .metadata_f32("rope.scaling.yarn_log_multiplier")
            .unwrap_or(0.0);
        let mscale = if ext_factor == 0.0 {
            1.0
        } else {
            (1.0 + 0.1 * (1.0f32 / freq_scale).ln())
                * (1.0 + 0.1 * yarn_log_mul * (1.0f32 / freq_scale).ln())
        };
        let kq_scale = mscale * mscale / (head_k_mla as f32).sqrt();
        let rope = RopeParams {
            rope_dim,
            freq_base: loaded.config.rope_freq_base,
            freq_scale,
            ext_factor,
            attn_factor: 1.0,
            beta_fast: loaded
                .metadata_f32("rope.scaling.yarn_beta_fast")
                .unwrap_or(32.0),
            beta_slow: loaded
                .metadata_f32("rope.scaling.yarn_beta_slow")
                .unwrap_or(1.0),
            n_ctx_orig: loaded
                .metadata_u64("rope.scaling.original_context_length")
                .unwrap_or(loaded.config.n_ctx_train as u64) as usize,
            layout: RopeLayout::Norm,
        };

        let indexer_types = loaded.metadata_array_u64("attention.indexer.types");
        let tok_embeddings = loaded
            .matrix("token_embd.weight")
            .context("loading token_embd.weight")?;
        let (output_norm, _) = loaded
            .tensor("output_norm.weight")
            .context("loading output_norm.weight")?;
        // Tied embeddings are the documented fallback upstream, though the
        // GLM-5.2 quants all carry a separate head.
        let output_weight = match loaded.matrix("output.weight") {
            Ok(matrix) => matrix,
            Err(_) => tok_embeddings.clone(),
        };

        let kv_row = kv_lora_rank + rope_dim;
        let mut kv_dims: Vec<(usize, usize)> = Vec::new();
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
            let get_expert_matrix = |suffix: &str| -> Result<ExpertQuantMatrix> {
                let name = format!("blk.{i}.{suffix}");
                loaded
                    .expert_matrix(&name)
                    .with_context(|| format!("loading {name}"))
            };

            kv_dims.push((kv_row, 1));
            let kv_slot = kv_dims.len() - 1;

            let scores_itself = match &indexer_types {
                Some(types) => types.get(i).copied().unwrap_or(1) != 0,
                None => default_indexer_is_full(i),
            };
            let indexer = if scores_itself {
                kv_dims.push((indexer_head_size, 1));
                Some(Indexer {
                    q_b: get_matrix("indexer.attn_q_b.weight")?,
                    attn_k: get_matrix("indexer.attn_k.weight")?,
                    k_norm_weight: get("indexer.k_norm.weight")?,
                    k_norm_bias: get("indexer.k_norm.bias")?,
                    proj: get_matrix("indexer.proj.weight")?,
                    cache_slot: kv_dims.len() - 1,
                })
            } else {
                None
            };

            let ffn = if i < n_layer_dense_lead {
                Ffn::Dense {
                    gate: get_matrix("ffn_gate.weight")?,
                    up: get_matrix("ffn_up.weight")?,
                    down: get_matrix("ffn_down.weight")?,
                }
            } else {
                Ffn::Moe(Box::new(Moe {
                    gate_inp: get_matrix("ffn_gate_inp.weight")?,
                    exp_probs_b: get("exp_probs_b.bias").ok(),
                    gate_exps: get_expert_matrix("ffn_gate_exps.weight")?,
                    down_exps: get_expert_matrix("ffn_down_exps.weight")?,
                    up_exps: get_expert_matrix("ffn_up_exps.weight")?,
                    gate_shexp: get_matrix("ffn_gate_shexp.weight")?,
                    down_shexp: get_matrix("ffn_down_shexp.weight")?,
                    up_shexp: get_matrix("ffn_up_shexp.weight")?,
                }))
            };

            let wq_b = get_matrix("attn_q_b.weight")?;
            anyhow::ensure!(
                wq_b.out_dim == n_head * head_k_mla,
                "layer {i}'s attn_q_b projects to {} outputs, not head_count * key_length_mla ({})",
                wq_b.out_dim,
                n_head * head_k_mla
            );
            let wkv_a_mqa = get_matrix("attn_kv_a_mqa.weight")?;
            anyhow::ensure!(
                wkv_a_mqa.out_dim == kv_row,
                "layer {i}'s attn_kv_a_mqa projects to {} outputs, not kv_lora_rank + rope dims ({kv_row})",
                wkv_a_mqa.out_dim
            );

            layers.push(GlmLayer {
                attn_norm: get("attn_norm.weight")?,
                ffn_norm: get("ffn_norm.weight")?,
                wq_a: get_matrix("attn_q_a.weight")?,
                q_a_norm: get("attn_q_a_norm.weight")?,
                wq_b,
                wkv_a_mqa,
                kv_a_norm: get("attn_kv_a_norm.weight")?,
                wk_b: get_expert_matrix("attn_k_b.weight")?,
                wv_b: get_expert_matrix("attn_v_b.weight")?,
                wo: get_matrix("attn_output.weight")?,
                indexer,
                kv_slot,
                ffn,
            });
        }

        Ok(Self {
            config: loaded.config.clone(),
            backend,
            tok_embeddings,
            output_norm,
            output_weight,
            kv_lora_rank,
            head_k_mla,
            head_v_mla,
            kv_row,
            rope,
            kq_scale,
            routing: ExpertRouting {
                n_expert_used,
                gating,
                weights_norm: loaded
                    .metadata_u64("expert_weights_norm")
                    .is_some_and(|v| v != 0),
                weights_scale: loaded.metadata_f32("expert_weights_scale").unwrap_or(1.0),
                groups: expert_groups,
            },
            indexer_n_head,
            indexer_head_size,
            indexer_top_k,
            norm_eps: loaded
                .metadata_f32("attention.layer_norm_epsilon")
                .unwrap_or(0.0),
            kv_dims,
            layers,
        })
    }

    /// Runs `tokens` through every trunk layer and returns each one's final
    /// normed hidden state (`[n_tokens, n_embd]`).
    fn run_layers(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<Vec<f32>> {
        let n_tokens = tokens.len();
        let n_embd = self.config.n_embd;

        let mut x = vec![0f32; n_tokens * n_embd];
        for (t, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            anyhow::ensure!(
                tok < self.config.n_vocab,
                "token id {tok} is out of vocab range"
            );
            x[t * n_embd..(t + 1) * n_embd].copy_from_slice(&self.tok_embeddings.row(tok));
        }

        // A layer that doesn't score positions itself attends whatever the
        // last scoring layer chose, so the choice is carried down the stack
        // — per token, since each token scores its own history.
        let mut scratch = LayerScratch::default();
        for layer in &self.layers {
            self.forward_layer(layer, cache, &mut x, n_tokens, start_pos, &mut scratch)?;
        }

        tensor::rmsnorm_inplace(
            &mut x,
            &self.output_norm,
            n_tokens,
            n_embd,
            self.config.rms_eps,
        );
        Ok(x)
    }

    fn forward_layer(
        &self,
        layer: &GlmLayer,
        cache: &mut KvCache,
        x: &mut [f32],
        n_tokens: usize,
        start_pos: usize,
        scratch: &mut LayerScratch,
    ) -> Result<()> {
        let n_embd = self.config.n_embd;
        let eps = self.config.rms_eps;
        let LayerScratch {
            selection,
            normed,
            ffn_out,
            ffn,
            attn: attn_scratch,
        } = scratch;

        tensor::rmsnorm_into(normed, x, &layer.attn_norm, n_tokens, n_embd, eps);
        let attn = self.attention(
            layer,
            cache,
            attn_scratch,
            normed,
            n_tokens,
            start_pos,
            selection,
        )?;
        tensor::add_inplace(x, &attn);

        tensor::rmsnorm_into(normed, x, &layer.ffn_norm, n_tokens, n_embd, eps);
        match &layer.ffn {
            // The shared one, rather than this module's own copy of it. The
            // gate and up projections now go out as one `matmul_batch`
            // instead of two sequential `matmul`s — numerically identical,
            // since they are independent products of the same input, and one
            // submission instead of two on any backend that batches.
            Ffn::Dense { gate, up, down } => super::swiglu_ffn_into(
                self.backend.as_ref(),
                ffn_out,
                ffn,
                normed,
                n_tokens,
                gate,
                up,
                down,
            ),
            Ffn::Moe(moe) => *ffn_out = self.moe_ffn(moe, normed, n_tokens),
        }
        tensor::add_inplace(x, ffn_out);
        Ok(())
    }

    /// One layer's MLA attention, in the absorbed form: the query is pushed
    /// through `attn_k_b` so it can be dotted with the cached compressed KV
    /// directly, and the output is pushed back through `attn_v_b`.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        layer: &GlmLayer,
        cache: &mut KvCache,
        scratch: &mut AttnScratch,
        normed: &[f32],
        n_tokens: usize,
        start_pos: usize,
        selection: &mut Vec<Vec<usize>>,
    ) -> Result<Vec<f32>> {
        let n_head = self.config.n_head;
        let eps = self.config.rms_eps;
        let rope_dim = self.config.rope_dim;
        let nope = self.head_k_mla - rope_dim;
        let q_lora_rank = layer.wq_a.out_dim;

        let AttnScratch {
            qr,
            q,
            kv,
            out: out_buf,
        } = scratch;
        // Token-independent projections run once for the whole chunk.
        self.backend.matmul_into(qr, normed, n_tokens, &layer.wq_a);
        tensor::rmsnorm_inplace(qr, &layer.q_a_norm, n_tokens, q_lora_rank, eps);
        let qr = &*qr;
        self.backend.matmul_into(q, qr, n_tokens, &layer.wq_b);
        self.backend
            .matmul_into(kv, normed, n_tokens, &layer.wkv_a_mqa);
        for t in 0..n_tokens {
            let pos = start_pos + t;
            let q_t = &mut q[t * n_head * self.head_k_mla..(t + 1) * n_head * self.head_k_mla];
            for h in 0..n_head {
                tensor::rope_apply_params_inplace(
                    &mut q_t[h * self.head_k_mla + nope..(h + 1) * self.head_k_mla],
                    1,
                    rope_dim,
                    pos,
                    None,
                    &self.rope,
                );
            }
            // The compressed half is normed; the rotary half is roped. The
            // two together are one cache row.
            let row = &mut kv[t * self.kv_row..(t + 1) * self.kv_row];
            tensor::rmsnorm_inplace(
                &mut row[..self.kv_lora_rank],
                &layer.kv_a_norm,
                1,
                self.kv_lora_rank,
                eps,
            );
            tensor::rope_apply_params_inplace(
                &mut row[self.kv_lora_rank..],
                1,
                rope_dim,
                pos,
                None,
                &self.rope,
            );
        }

        // The absorbed query: per head, `attn_k_b` maps the query's
        // non-rotary part into the compressed KV space, and the rotary part
        // rides along unchanged.
        let absorbed_dim = self.kv_lora_rank + rope_dim;

        // `wk_b`/`wv_b` are weights and their rows do not depend on `t`, but
        // `ExpertQuantMatrix::row` inside the token loops below dequantized
        // each one afresh — into a newly allocated `Vec<f32>`, dotted once,
        // dropped — `n_tokens` times over. See `engine::arch::kda`'s copy of
        // this hoist for the measurement that found it.
        //
        // Bit-exact: same bytes, same dequantizer, same slice handed to
        // `tensor::dot`.
        let absorb = |m: &crate::engine::loader::ExpertQuantMatrix, rows: usize| -> Vec<Vec<f32>> {
            (0..n_head)
                .into_par_iter()
                .map(|h| {
                    let span = m.expert_span(h);
                    let mut flat = Vec::with_capacity(rows * m.in_dim);
                    let mut row = Vec::new();
                    for j in 0..rows {
                        m.row_from(span, j, &mut row);
                        flat.extend_from_slice(&row);
                    }
                    flat
                })
                .collect()
        };
        // **Only when there is more than one token to amortize it over.**
        // At `n_tokens == 1` there is no reuse: the row is dotted once, and
        // gathering it into a contiguous buffer first costs an extra copy and
        // makes the dot read cold memory instead of the line just written.
        // Measured **-6.8% on decode** before this guard, against +2.3% on
        // prefill with it — the hoist is a prefill optimization and saying so
        // in the type is cheaper than re-deriving it.
        let hoist = n_tokens > 1;
        let wk_b = hoist.then(|| absorb(&layer.wk_b, self.kv_lora_rank));
        let wv_b = hoist.then(|| absorb(&layer.wv_b, self.head_v_mla));

        let mut absorbed = vec![0f32; n_tokens * n_head * absorbed_dim];
        for t in 0..n_tokens {
            let q_t = &q[t * n_head * self.head_k_mla..(t + 1) * n_head * self.head_k_mla];
            let heads: Vec<Vec<f32>> = (0..n_head)
                .into_par_iter()
                .map(|h| {
                    let q_nope = &q_t[h * self.head_k_mla..h * self.head_k_mla + nope];
                    let mut head = vec![0f32; absorbed_dim];
                    super::kda::project_rows(
                        &layer.wk_b,
                        wk_b.as_deref(),
                        h,
                        q_nope,
                        nope,
                        &mut head[..self.kv_lora_rank],
                    );
                    head[self.kv_lora_rank..].copy_from_slice(
                        &q_t[h * self.head_k_mla + nope..(h + 1) * self.head_k_mla],
                    );
                    head
                })
                .collect();
            for (h, head) in heads.iter().enumerate() {
                let at = (t * n_head + h) * absorbed_dim;
                absorbed[at..at + absorbed_dim].copy_from_slice(head);
            }
        }

        let indexer = layer
            .indexer
            .as_ref()
            .map(|ix| self.indexer_inputs(ix, qr, normed, n_tokens, start_pos));

        let mut attn_out = vec![0f32; n_tokens * n_head * self.head_v_mla];
        selection.resize(n_tokens, Vec::new());
        for t in 0..n_tokens {
            let pos = start_pos + t;
            let kv_t = &kv[t * self.kv_row..(t + 1) * self.kv_row];
            cache.layers[layer.kv_slot].push(kv_t, kv_t);

            if let (Some(ix), Some((iq, weights, keys))) =
                (layer.indexer.as_ref(), indexer.as_ref())
            {
                cache.layers[ix.cache_slot].push(
                    &keys[t * self.indexer_head_size..(t + 1) * self.indexer_head_size],
                    &keys[t * self.indexer_head_size..(t + 1) * self.indexer_head_size],
                );
                // Scoring can only change the answer once there are more
                // positions than the indexer is allowed to keep: below
                // that, its top-k is every visible position and the mask it
                // produces is the causal mask upstream adds back anyway.
                selection[t] = if pos < self.indexer_top_k {
                    (0..=pos).collect()
                } else {
                    let scores = self.indexer_scores(
                        &cache.layers[ix.cache_slot],
                        &iq[t * self.indexer_n_head * self.indexer_head_size
                            ..(t + 1) * self.indexer_n_head * self.indexer_head_size],
                        &weights[t * self.indexer_n_head..(t + 1) * self.indexer_n_head],
                        pos + 1,
                    );
                    let mut chosen = top_k_indices(&scores, self.indexer_top_k);
                    chosen.sort_unstable();
                    chosen
                };
            }
            anyhow::ensure!(
                !selection[t].is_empty(),
                "layer reuses a lightning-indexer selection, but no earlier layer produced one"
            );

            let chosen = &selection[t];
            let mut keys = vec![0f32; chosen.len() * self.kv_row];
            {
                let slot = &cache.layers[layer.kv_slot];
                for (i, &p) in chosen.iter().enumerate() {
                    keys[i * self.kv_row..(i + 1) * self.kv_row].copy_from_slice(slot.key_at(
                        p,
                        0,
                        self.kv_row,
                    ));
                }
            }

            // Per head: attend over the compressed rows (the value is each
            // row's leading `kv_lora_rank`), then decompress the result
            // back to a real value head with `attn_v_b`.
            let heads: Vec<Vec<f32>> = (0..n_head)
                .into_par_iter()
                .map(|h| {
                    let q_h = &absorbed
                        [(t * n_head + h) * absorbed_dim..(t * n_head + h + 1) * absorbed_dim];
                    let compressed = attend(
                        q_h,
                        &keys,
                        self.kv_row,
                        self.kv_lora_rank,
                        self.kq_scale,
                        None,
                    );
                    let mut head = vec![0f32; self.head_v_mla];
                    super::kda::project_rows(
                        &layer.wv_b,
                        wv_b.as_deref(),
                        h,
                        &compressed,
                        self.kv_lora_rank,
                        &mut head,
                    );
                    head
                })
                .collect();
            for (h, head) in heads.iter().enumerate() {
                let at = (t * n_head + h) * self.head_v_mla;
                attn_out[at..at + self.head_v_mla].copy_from_slice(head);
            }
        }

        self.backend
            .matmul_into(out_buf, &attn_out, n_tokens, &layer.wo);
        Ok(std::mem::take(out_buf))
    }

    /// A scoring layer's per-token indexer inputs: the roped query, the
    /// pre-scaled per-head weights, and the key to cache.
    fn indexer_inputs(
        &self,
        ix: &Indexer,
        qr: &[f32],
        normed: &[f32],
        n_tokens: usize,
        start_pos: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let rope_dim = self.config.rope_dim;
        let dim = self.indexer_head_size;
        let mut q = self.backend.matmul(qr, n_tokens, &ix.q_b);
        let mut keys = self.backend.matmul(normed, n_tokens, &ix.attn_k);
        for t in 0..n_tokens {
            let pos = start_pos + t;
            // Unlike the main attention, the indexer's rotary dimensions
            // are the *leading* ones of each head.
            let q_t = &mut q[t * self.indexer_n_head * dim..(t + 1) * self.indexer_n_head * dim];
            for h in 0..self.indexer_n_head {
                tensor::rope_apply_params_inplace(
                    &mut q_t[h * dim..h * dim + rope_dim],
                    1,
                    rope_dim,
                    pos,
                    None,
                    &self.rope,
                );
            }
            let key = &mut keys[t * dim..(t + 1) * dim];
            layer_norm_inplace(key, &ix.k_norm_weight, &ix.k_norm_bias, self.norm_eps);
            tensor::rope_apply_params_inplace(
                &mut key[..rope_dim],
                1,
                rope_dim,
                pos,
                None,
                &self.rope,
            );
        }
        let mut weights = self.backend.matmul(normed, n_tokens, &ix.proj);
        let scale = 1.0 / ((dim * self.indexer_n_head) as f32).sqrt();
        for w in weights.iter_mut() {
            *w *= scale;
        }
        (q, weights, keys)
    }

    /// The lightning indexer's score for each visible position: a per-head
    /// `relu`'d dot product against that position's indexer key, combined
    /// with this token's own per-head weights.
    fn indexer_scores(
        &self,
        keys: &crate::engine::kv_cache::LayerCache,
        q: &[f32],
        weights: &[f32],
        visible: usize,
    ) -> Vec<f32> {
        let dim = self.indexer_head_size;
        (0..visible)
            .into_par_iter()
            .map(|p| {
                let key = keys.key_at(p, 0, dim);
                (0..self.indexer_n_head)
                    .map(|h| tensor::dot(&q[h * dim..(h + 1) * dim], key).max(0.0) * weights[h])
                    .sum()
            })
            .collect()
    }

    /// The routed + shared-expert SwiGLU MoE FFN — `super::swiglu_moe_ffn`,
    /// shared with `engine::arch::bailingmoe`, which runs the same
    /// computation over the same tensor names under whatever routing rules
    /// its own file declares.
    fn moe_ffn(&self, moe: &Moe, normed: &[f32], n_tokens: usize) -> Vec<f32> {
        super::swiglu_moe_ffn(
            self.backend.as_ref(),
            &self.routing,
            normed,
            n_tokens,
            self.config.n_embd,
            &super::SwigluMoe {
                gate_inp: &moe.gate_inp,
                exp_probs_b: moe.exp_probs_b.as_deref(),
                gate_exps: &moe.gate_exps,
                up_exps: &moe.up_exps,
                down_exps: &moe.down_exps,
                shared: Some(super::SwigluSharedExpert {
                    gate: &moe.gate_shexp,
                    up: &moe.up_shexp,
                    down: &moe.down_shexp,
                }),
                clamp_exp: 0.0,
                clamp_shexp: 0.0,
            },
        )
    }
}

impl ModelForward for GlmModel {
    /// The instrumentation hook `engine::generate` reads to count GPU
    /// submissions per decode step. Only useful compared *between*
    /// architectures, which is why an architecture that answers `None` is
    /// invisible to exactly the measurement that would say whether a
    /// cross-architecture change helped it.
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
        let hidden = self.run_layers(cache, tokens, start_pos)?;
        let n_embd = self.config.n_embd;
        let last = &hidden[(tokens.len() - 1) * n_embd..];
        Ok(self.backend.matmul(last, 1, &self.output_weight))
    }

    /// Every input position's next-token logits, for the speculative
    /// verify step — sound because this architecture's cache is purely
    /// positional, so `KvCache::truncate` rolls a rejected draft's tail
    /// back exactly.
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
        let hidden = self.run_layers(cache, tokens, start_pos)?;
        let n_embd = self.config.n_embd;
        Ok((0..tokens.len())
            .map(|t| {
                self.backend.matmul(
                    &hidden[t * n_embd..(t + 1) * n_embd],
                    1,
                    &self.output_weight,
                )
            })
            .collect())
    }

    fn forward_hidden_states(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        anyhow::ensure!(
            !tokens.is_empty(),
            "forward_hidden_states called with no tokens"
        );
        let mut cache = self.new_kv_cache(tokens.len());
        self.run_layers(&mut cache, tokens, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::{default_indexer_is_full, layer_norm_inplace};

    /// GLM-5.2's pattern: the first three layers score, then every fourth.
    #[test]
    fn the_default_indexer_pattern_matches_the_reference_config() {
        let full: Vec<usize> = (0..24).filter(|&il| default_indexer_is_full(il)).collect();
        assert_eq!(full, vec![0, 1, 2, 6, 10, 14, 18, 22]);
    }

    /// The first layer must score: every later layer either scores or
    /// reuses, so a stack that never scored would have nothing to attend.
    #[test]
    fn the_first_layer_always_scores() {
        assert!(default_indexer_is_full(0));
    }

    /// LayerNorm centers and shifts; RMSNorm does neither. Using the wrong
    /// one on the indexer key is a silent accuracy loss, not an error.
    #[test]
    fn layer_norm_centers_scales_and_shifts() {
        let mut x = vec![1.0, 3.0];
        layer_norm_inplace(&mut x, &[1.0, 1.0], &[0.0, 0.0], 0.0);
        // mean 2, variance 1 -> -1, +1
        assert!((x[0] + 1.0).abs() < 1e-5, "{x:?}");
        assert!((x[1] - 1.0).abs() < 1e-5, "{x:?}");

        let mut y = vec![1.0, 3.0];
        layer_norm_inplace(&mut y, &[2.0, 2.0], &[0.5, 0.5], 0.0);
        assert!((y[0] + 1.5).abs() < 1e-5, "{y:?}");
        assert!((y[1] - 2.5).abs() < 1e-5, "{y:?}");
    }
}
