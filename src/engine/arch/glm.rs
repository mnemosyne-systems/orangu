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
//! Both run in [`engine::arch::dsa`](super::dsa), the trunk this module
//! shares with `engine::arch::mistral_large4`; what lives here is how a
//! `glm-dsa` file spells its shape — the query LoRA the indexer shares, the
//! DeepSeek YaRN convention, the default indexer pattern.
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
use std::sync::Arc;

use super::dsa::{DsaModel, DsaShape, LayerPlan};
use super::indexer::IndexerShape;
use super::kda::MlaShape;
use crate::engine::backend::Backend;
use crate::engine::loader::LoadedModel;
use crate::engine::tensor::{RopeLayout, RopeParams};

/// `attention.indexer.types` for GLM-5.2, whose GGUFs don't carry the key:
/// layers 0, 1 and 2 score, then every fourth layer after that, with the
/// three layers in between reusing the last scored selection. Upstream's
/// `GLM_5_2_DEFAULT_INDEXER_TYPES`, which reads it from the reference
/// `config.json`.
fn default_indexer_is_full(layer: usize) -> bool {
    layer < 2 || (layer - 2).is_multiple_of(4)
}

/// Builds a `glm-dsa` model on the shared [`DsaModel`] trunk.
pub fn load_with_backend(loaded: &LoadedModel, backend: Arc<dyn Backend>) -> Result<DsaModel> {
    let n_head = loaded.config.n_head;
    let rope_dim = loaded.config.rope_dim;
    let n_layer = super::dsa::trunk_layer_count(loaded)?;

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

    // YaRN, when a file uses it, scales the attention logits as well as the
    // frequencies: upstream folds `mscale^2` into the softmax scale (the
    // DeepSeek convention, with the rope kernel's own magnitude left at 1).
    // GLM-5.2 declares no rope scaling at all, so every term below is 1 and
    // this reduces to `1/sqrt(key_length_mla)` — note that width, not the
    // wider `kv_lora_rank + rope_dim` the absorbed query and key actually
    // are.
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
        n_ctx_orig: super::position::original_context_length(loaded),
        layout: RopeLayout::Norm,
    };

    let shape = DsaShape {
        mla: MlaShape {
            n_head,
            kv_lora_rank,
            head_k_mla,
            head_v_mla,
            kv_row: kv_lora_rank + rope_dim,
            rope: Some(rope),
            rope_dim,
            freq_factors: None,
            kq_scale: mscale * mscale / (head_k_mla as f32).sqrt(),
            eps: loaded.config.rms_eps,
        },
        indexer: IndexerShape::from_gguf(loaded, Some(rope))?,
    };

    let scores = super::dsa::indexer_types(loaded, n_layer, default_indexer_is_full);
    let plans: Vec<LayerPlan> = (0..n_layer)
        .map(|il| LayerPlan {
            shape: 0,
            kv_source: il,
            scores: scores[il],
            dense: il < n_layer_dense_lead,
        })
        .collect();
    DsaModel::load(loaded, backend, vec![shape], &plans)
}

#[cfg(test)]
mod tests {
    use super::default_indexer_is_full;

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
}
