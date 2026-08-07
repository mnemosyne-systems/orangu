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

//! DFlash / DSpark (`general.architecture = "dflash"`), e.g. the
//! `dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf` sidecar that ships alongside
//! `unsloth/DeepSeek-V4-Flash-0731-GGUF`.
//!
//! These are **not** standalone models, and not by omission: the checkpoint
//! contains no `token_embd` and no `output` tensor at all (81 tensors, all
//! of them block weights plus `fc`, the two Markov-head matrices and the
//! confidence projection). A DFlash draft reads *the target model's* hidden
//! states — `dflash.target_layers` names which, `[41, 42, 43]` for the
//! DeepSeek-V4-Flash sidecar — fuses them through `fc`, and drafts
//! `dflash.block_size` tokens at a time using the target's own embedding
//! table and LM head. Upstream runs one only as the draft half of a
//! speculative pair (`src/models/dflash.cpp` takes both its embeddings and
//! its output projection from `cparams.ctx_other`, the target's context).
//!
//! So "serving a dflash file" means serving its target. `main::
//! auto_pair_dflash_target` does exactly that: selecting one of these
//! resolves the paired base model from the same Hugging Face repo —
//! downloading it if the models directory doesn't have it yet — and serves
//! that instead, which for the DeepSeek-V4-Flash sidecar is the
//! `deepseek4` model `engine::arch::deepseek4` runs natively. This module
//! is the honest failure for the one case that bypasses the pairing: a
//! dflash file handed over directly, from outside a Hugging Face cache
//! layout, with no target to pair it to.
//!
//! Executing the draft *as* a draft would need a second-model speculative
//! path, which this engine does not have — `engine::generate`'s speculative
//! decoding is prompt-lookup (n-gram) drafting against the served model
//! itself, with no notion of a separate draft model to run.

use anyhow::Result;

use super::ModelForward;
use crate::engine::kv_cache::KvCache;
use crate::engine::loader::{LoadedModel, ModelConfig};

pub struct DFlashModel {
    config: ModelConfig,
}

fn standalone_error(loaded: Option<&LoadedModel>) -> anyhow::Error {
    let targets = loaded
        .and_then(|l| l.metadata_array_u64("target_layers"))
        .map(|layers| {
            layers
                .iter()
                .map(|l| l.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_else(|| "unknown".to_string());
    anyhow::anyhow!(
        "this is a dflash draft sidecar, not a servable model: it carries no token embeddings \
         and no output projection, and reads its target model's hidden states (layers {targets}). \
         Serve the paired target model instead — selecting a draft from a Hugging Face cache \
         directory does that automatically"
    )
}

impl DFlashModel {
    pub fn load_with_backend(loaded: &LoadedModel) -> Result<Self> {
        Err(standalone_error(Some(loaded)))
    }
}

impl ModelForward for DFlashModel {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn new_kv_cache(&self, capacity: usize) -> KvCache {
        let _ = capacity;
        KvCache::new_with_dims(0, &[])
    }

    fn forward(
        &self,
        _cache: &mut KvCache,
        _tokens: &[u32],
        _start_pos: usize,
        _slot_id: usize,
    ) -> Result<Vec<f32>> {
        Err(standalone_error(None))
    }

    fn forward_hidden_states(&self, _tokens: &[u32]) -> Result<Vec<f32>> {
        Err(standalone_error(None))
    }
}

#[cfg(test)]
mod tests {
    use super::standalone_error;

    /// The message has to say what to do about it, not just that it
    /// failed: the only way a user reaches it is by naming a draft file
    /// directly, and the fix is to name the target.
    #[test]
    fn the_standalone_error_points_at_the_paired_target_model() {
        let message = standalone_error(None).to_string();
        assert!(message.contains("dflash draft sidecar"), "{message}");
        assert!(message.contains("paired target model"), "{message}");
    }
}
