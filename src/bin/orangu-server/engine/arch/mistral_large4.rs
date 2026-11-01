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

//! Mistral Large 4 (`general.architecture = "mistral-large4"`), e.g.
//! `unsloth/Mistral-Large-4-1T-A52B-GGUF`.
//!
//! Despite the name, the block is not `mistral3`'s: it is the DeepSeek
//! sparse-attention decoder `glm-dsa` is — absorbed multi-head latent
//! attention over a compressed key/value cache, a per-token lightning
//! indexer choosing which positions each layer attends, one leading dense
//! FFN block and then routed experts with an always-on shared one. It runs
//! on that same trunk, [`engine::arch::dsa`](super::dsa). What it shares
//! with `mistral3` is its rotary convention: YaRN read through
//! [`super::position::yarn_rope_params`], NORM pairing, and the
//! `log(pos + 1)` basis of upstream's attention-temperature input.
//!
//! Six things set it apart from `glm-dsa`, every one declared in the file:
//!
//! 1. **No query LoRA.** `attn_q` projects the layer input straight to the
//!    heads, so the indexer has no LoRA intermediate to share: it
//!    LayerNorms the layer input itself (`indexer.q_norm`, weight and
//!    bias) and projects its query off that.
//! 2. **SSMax** on both queries — a learned per-head
//!    `s * log(pos + 1) + b` scale (`attn_ssmax`, `indexer.ssmax`), see
//!    [`super::position::Ssmax`].
//! 3. **NoPE layers.** `attention.rope_pattern` leaves every fourth layer
//!    unrotated. Such a layer spends its whole query head on the non-rotary
//!    part, and its cache row is the compressed vector alone — upstream
//!    zero-pads both to the rotary layers' width, which contributes nothing
//!    to a dot product, so here it is simply a second attention shape.
//! 4. **Shared key/value rows.** `attention.kv_source_map` lets a layer
//!    attend the rows an earlier layer of the same kind wrote, with no
//!    `attn_kv_a_mqa` of its own.
//! 5. **LeRoPE**: learned rotary frequency divisors, one shared
//!    `rope_freqs.weight` for every layer and both the attention and the
//!    indexer.
//! 6. **`final_logit_softcapping`** on the head.
//!
//! YaRN here is used without magnitude scaling: the rope kernel's
//! `1 + 0.1 * ln(factor)` is pre-cancelled (`yarn_log_multiplier = 1`), and
//! the softmax scale is the plain `1/sqrt(key_length_mla)`.
//!
//! Transcribed from upstream `llama.cpp`'s `src/models/mistral-large4.cpp`.
//! As in `engine::arch::glm`, the Hadamard rotation of the indexer's
//! queries and keys is not carried over: it is orthonormal and applied to
//! both sides, so it changes no dot product. Nor is the vision tower
//! shipped beside the model as a separate `mmproj-*.gguf`.

use anyhow::{Context, Result};
use std::sync::Arc;

use super::dsa::{DsaModel, DsaShape, LayerPlan};
use super::indexer::IndexerShape;
use super::kda::MlaShape;
use crate::engine::backend::Backend;
use crate::engine::loader::LoadedModel;
use crate::engine::tensor::RopeLayout;

/// [`DsaShape`] indices.
const ROPE: usize = 0;
const NOPE: usize = 1;

/// The per-layer plan from the file's three per-layer arrays, each
/// defaulting the way upstream does when the key is absent: every layer
/// rotates, writes its own rows, and scores.
fn layer_plans(
    n_layer: usize,
    n_layer_dense_lead: usize,
    rope_pattern: Option<&[u64]>,
    kv_source_map: Option<&[u64]>,
    scores: &[bool],
) -> Vec<LayerPlan> {
    (0..n_layer)
        .map(|il| {
            let rotates = rope_pattern.is_none_or(|p| p.get(il).copied().unwrap_or(1) != 0);
            LayerPlan {
                shape: if rotates { ROPE } else { NOPE },
                kv_source: kv_source_map
                    .and_then(|m| m.get(il).copied())
                    .map_or(il, |s| s as usize),
                scores: scores[il],
                dense: il < n_layer_dense_lead,
            }
        })
        .collect()
}

/// Builds a `mistral-large4` model on the shared [`DsaModel`] trunk.
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

    // NORM pairing, as `mistral3` — upstream's `LLAMA_ROPE_TYPE_NORM` arm.
    let n_ctx_orig = super::position::original_context_length(loaded);
    let rope = super::position::yarn_rope_params(loaded, n_ctx_orig, RopeLayout::Norm);
    let (freq_factors, _) = loaded
        .tensor("rope_freqs.weight")
        .context("loading rope_freqs.weight")?;
    anyhow::ensure!(
        freq_factors.len() == rope_dim / 2,
        "rope_freqs.weight has {} values, not rope.dimension_count / 2 ({})",
        freq_factors.len(),
        rope_dim / 2
    );
    let freq_factors: Arc<[f32]> = freq_factors.into();
    // No mscale: YaRN runs without magnitude scaling here.
    let kq_scale = 1.0 / (head_k_mla as f32).sqrt();
    let mla = |rotates: bool| MlaShape {
        n_head,
        kv_lora_rank,
        head_k_mla,
        head_v_mla,
        kv_row: kv_lora_rank + if rotates { rope_dim } else { 0 },
        rope: rotates.then_some(rope),
        rope_dim: if rotates { rope_dim } else { 0 },
        freq_factors: rotates.then(|| freq_factors.clone()),
        kq_scale,
        eps: loaded.config.rms_eps,
    };
    let mut rope_indexer = IndexerShape::from_gguf(loaded, Some(rope))?;
    rope_indexer.freq_factors = Some(freq_factors.clone());
    let shapes = vec![
        DsaShape {
            mla: mla(true),
            indexer: rope_indexer,
        },
        DsaShape {
            mla: mla(false),
            indexer: IndexerShape::from_gguf(loaded, None)?,
        },
    ];

    let rope_pattern = loaded.metadata_array_u64("attention.rope_pattern");
    let kv_source_map = loaded.metadata_array_u64("attention.kv_source_map");
    let scores = super::dsa::indexer_types(loaded, n_layer, |_| true);
    let plans = layer_plans(
        n_layer,
        n_layer_dense_lead,
        rope_pattern.as_deref(),
        kv_source_map.as_deref(),
        &scores,
    );
    DsaModel::load(loaded, backend, shapes, &plans)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The file's own arrays for its first eight layers and a shared-row
    /// layer: every fourth unrotated, layer 39 reading layer 35's rows,
    /// layers 17-18 reusing layer 16's selection.
    #[test]
    fn the_plans_follow_the_rope_pattern_and_the_kv_source_map() {
        let n = 40;
        let rope: Vec<u64> = (0..n).map(|il| u64::from(il % 4 != 3)).collect();
        let mut source: Vec<u64> = (0..n as u64).collect();
        source[39] = 35;
        let scores: Vec<bool> = (0..n).map(|il| !(17..=18).contains(&il)).collect();
        let plans = layer_plans(n, 1, Some(&rope), Some(&source), &scores);

        assert!(plans[0].dense && !plans[1].dense);
        assert_eq!(plans[2].shape, ROPE);
        assert_eq!(plans[3].shape, NOPE);
        assert_eq!(plans[39].shape, NOPE);
        assert_eq!(plans[39].kv_source, 35);
        assert_eq!(plans[35].kv_source, 35);
        assert!(!plans[17].scores && plans[19].scores);
        super::super::dsa::validate_plans(&plans, 2).expect("a valid stack");
    }

    /// With none of the three arrays the stack is a plain rotary one where
    /// every layer writes its own rows — upstream's defaults.
    #[test]
    fn missing_arrays_default_to_rotary_self_sourced_layers() {
        let plans = layer_plans(4, 0, None, None, &[true; 4]);
        for (il, p) in plans.iter().enumerate() {
            assert_eq!(p.shape, ROPE);
            assert_eq!(p.kv_source, il);
            assert!(!p.dense);
        }
    }

    /// A shared-row layer that reads a layer of the other shape is refused
    /// at load: their rows are different widths.
    #[test]
    fn a_kv_source_across_rope_kinds_is_refused() {
        let rope = [1u64, 0];
        let source = [0u64, 0];
        let plans = layer_plans(2, 0, Some(&rope), Some(&source), &[true, true]);
        assert!(super::super::dsa::validate_plans(&plans, 2).is_err());
    }
}
