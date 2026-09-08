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

//! IBM Granite 3.x (`ibm-granite/granite-3.1-2b-instruct` and relatives).
//!
//! **This module is four numbers and their justification.** Granite's block
//! is `arch::llama`'s, node for node — RMSNorm, GQA with RoPE, a
//! separate-gate SwiGLU FFN, an untied output projection — and upstream
//! implements it in `llama.cpp/src/models/granite.cpp` as the llama graph
//! with four scalar multipliers threaded through. So this file reads those
//! four out of the GGUF and hands them to [`LlamaModel`], which does the
//! forward pass. There is no second copy of the block to keep in step.
//!
//! The four, with `granite-3.1-2b-instruct`'s values:
//!
//! | key | value | where it lands |
//! |---|---|---|
//! | `embedding_scale` | 12 | token embeddings, once, before layer 0 |
//! | `residual_scale` | 0.22 | each sub-layer output, before its residual add |
//! | `logit_scale` | 8 | the final logits are *divided* by it |
//! | `attention.scale` | 0.015625 | the softmax scale |
//!
//! Every one of them is **wrong-by-default rather than absent**, which is
//! the reason this cannot simply be added to
//! `loader::LLAMA_STYLE_ARCHITECTURES` and forgotten. A Granite checkpoint
//! read as plain llama loads, runs, and produces fluent nonsense: nothing
//! about its shapes is unusual, so there is no tensor to be missing and no
//! assertion to trip. The attention scale is the sharpest of the four —
//! 3.1-2B has `head_dim = 64`, so the derived `1/sqrt(64)` is `0.125` and
//! the checkpoint asks for `0.015625`, eight times smaller.
//!
//! **What this costs.** `residual_scale` multiplies each branch *before* its
//! residual add, and the fused GPU chains do those adds internally with no
//! term for it — see [`crate::engine::arch::llama::Multipliers::residual`]
//! for why gemma's existing `layer_output_scale` is not the same function
//! and cannot stand in. So a Granite model declines the fused decode chain,
//! the fused post-attention prefill chain, and the GPU argmax, and runs the
//! step-by-step path in `LlamaModel::run_layers`. That path still uses the
//! GPU for every matmul and for attention; what it gives up is the
//! round-trip fusion. Correct and slower was the right way round to build
//! this first, and teaching the chain a residual multiplier is a contained
//! follow-up: the shader already carries an `out_scale` for gemma.

use anyhow::Result;
use std::sync::Arc;

use crate::engine::arch::llama::{LlamaModel, Multipliers};
use crate::engine::backend::Backend;
use crate::engine::loader::LoadedModel;

/// Granite's four multipliers, or the identity for any that is absent.
///
/// Absent means absent, not zero: a checkpoint that omits `logit_scale` is
/// asking for no division, and defaulting it to `0.0` would divide every
/// logit by zero. Each key is read through `LoadedModel::metadata_f32`,
/// which prefixes `general.architecture` — so these are `granite.*` here and
/// would be `granitemoe.*` in a MoE sibling, without this module naming
/// either.
fn multipliers(loaded: &LoadedModel) -> Multipliers {
    multipliers_from(|key| loaded.metadata_f32(key))
}

/// [`multipliers`] over a bare lookup, so the rules above can be tested
/// without a checkpoint to load.
fn multipliers_from(lookup: impl Fn(&str) -> Option<f32>) -> Multipliers {
    let scale = |key: &str| lookup(key).filter(|v| *v != 0.0);
    Multipliers {
        embedding: scale("embedding_scale").unwrap_or(1.0),
        residual: scale("residual_scale").unwrap_or(1.0),
        logit: scale("logit_scale").unwrap_or(1.0),
        attention: scale("attention.scale"),
    }
}

/// Builds the Granite forward pass: [`LlamaModel`] carrying
/// [`multipliers`].
pub fn load_with_backend(loaded: &LoadedModel, backend: Arc<dyn Backend>) -> Result<LlamaModel> {
    LlamaModel::load_with_multipliers(loaded, backend, multipliers(loaded))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with(pairs: &'static [(&'static str, f32)]) -> Multipliers {
        multipliers_from(move |key| pairs.iter().find_map(|(k, v)| (*k == key).then_some(*v)))
    }

    /// The values `granite-3.1-2b-instruct` actually carries, read back.
    #[test]
    fn granite_3_1_multipliers_are_read_from_the_file() {
        let mul = with(&[
            ("embedding_scale", 12.0),
            ("residual_scale", 0.22),
            ("logit_scale", 8.0),
            ("attention.scale", 0.015625),
        ]);
        assert_eq!(mul.embedding, 12.0);
        assert_eq!(mul.residual, 0.22);
        assert_eq!(mul.logit, 8.0);
        assert_eq!(mul.attention, Some(0.015625));
        // Not the identity, which is what keeps Granite off the fused
        // chains that cannot express `residual_scale`.
        assert_ne!(mul, Multipliers::NONE);
    }

    /// A checkpoint carrying none of them is plain llama, and in particular
    /// must not end up dividing by zero.
    #[test]
    fn a_checkpoint_without_the_keys_is_the_identity() {
        assert_eq!(with(&[]), Multipliers::NONE);
    }

    /// `0.0` is not a scale anyone means: `logit_scale = 0` would divide
    /// every logit by zero and `embedding_scale = 0` would erase the prompt.
    #[test]
    fn a_zero_scale_is_treated_as_absent() {
        let mul = with(&[("logit_scale", 0.0), ("embedding_scale", 0.0)]);
        assert_eq!(mul.logit, 1.0);
        assert_eq!(mul.embedding, 1.0);
    }

    /// The attention scale is the one that stays `None` when absent, so
    /// `LlamaModel` derives `1/sqrt(head_dim)` as it always has.
    #[test]
    fn an_absent_attention_scale_is_none_not_one() {
        assert_eq!(with(&[("logit_scale", 8.0)]).attention, None);
    }
}
