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

//! Qwen3.5/3.6-MoE (`general.architecture = "qwen35moe"`), e.g.
//! `unsloth/Qwen3.6-35B-A3B-GGUF` — confirmed against real upstream
//! `llama.cpp` source (`src/models/qwen35moe.cpp`,
//! `src/models/delta-net-base.cpp`, `src/llama-graph.cpp`'s
//! `build_moe_ffn`, and the relevant `ggml-cpu/ops.cpp` compute kernels —
//! fetched and read directly, not guessed).
//!
//! The hybrid full-attention / gated-DeltaNet trunk is
//! [`engine::arch::qwen_hybrid`](super::qwen_hybrid), shared with the dense
//! `qwen35` and with `qwen3next` exactly as upstream shares
//! `llm_build_delta_net_base` between the same three. This module is only
//! the FFN choice: a mixture-of-experts FFN on every layer — standard
//! softmax top-k routing (renormalized) over routed experts, plus a
//! separately-`sigmoid`-gated shared expert whose output adds in. That FFN
//! is itself shared with `qwen3next` as
//! [`qwen_hybrid::MoeFfn`](super::qwen_hybrid::MoeFfn), the two having been
//! byte-for-byte identical implementations before.

use anyhow::Result;
use std::sync::Arc;

use super::ModelForward;
use super::qwen_hybrid::{MoeFfn, Trunk};
use crate::engine::backend::Backend;
use crate::engine::kv_cache::KvCache;
use crate::engine::loader::{LoadedModel, ModelConfig};

pub struct Qwen35MoeModel {
    trunk: Trunk<MoeFfn>,
}

impl Qwen35MoeModel {
    pub fn load_with_backend(loaded: &LoadedModel, backend: Arc<dyn Backend>) -> Result<Self> {
        Ok(Self {
            trunk: Trunk::load_moe(loaded, backend)?,
        })
    }
}

impl ModelForward for Qwen35MoeModel {
    fn vulkan_backend(&self) -> Option<&crate::engine::backend::vulkan::VulkanBackend> {
        self.trunk.backend.as_wgpu()
    }

    fn config(&self) -> &ModelConfig {
        &self.trunk.config
    }

    fn n_trunk_layer(&self) -> usize {
        self.trunk.layer_count()
    }

    fn new_kv_cache(&self, capacity: usize) -> KvCache {
        self.trunk.new_kv_cache(capacity)
    }

    fn forward(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        slot_id: usize,
    ) -> Result<Vec<f32>> {
        // A decode step alternates between the card and the host's expert
        // turns; the card's clock is held up through it.
        let _clock = super::hold_clock_for_step(self.trunk.backend.as_ref(), tokens.len());
        self.trunk.forward(cache, tokens, start_pos, slot_id)
    }

    /// Every layer's experts run on the host's turn of the step, so a batch
    /// is the host layer loop with each row's sub-layers on its own cache.
    fn batches_decode(&self) -> bool {
        true
    }

    /// One decode step for several sequences (`Trunk::decode_batch_host`), the card's clock held as for a solo step.
    fn forward_decode_batch(
        &self,
        rows: &mut [super::DecodeRow<'_>],
        tokens: &[u32],
    ) -> Result<Option<Vec<Vec<f32>>>> {
        let _clock = super::hold_clock_for_step(self.trunk.backend.as_ref(), 1);
        self.trunk.decode_batch_host(rows, tokens)
    }

    fn forward_hidden_states(&self, _tokens: &[u32]) -> Result<Vec<f32>> {
        anyhow::bail!("embeddings are not yet supported for Qwen3.5-MoE models")
    }

    /// The same cut as `qwen35`'s and `qwen3next`'s, through the same trunk:
    /// each layer keeps its own state — a full-attention layer its KV rows, a
    /// gated-DeltaNet layer its recurrent state — and passes on only the
    /// residual stream, the experts included, so any cut is exact.
    fn supports_layer_split(&self) -> bool {
        true
    }

    fn new_kv_cache_for_layers(&self, layers: std::ops::Range<usize>, capacity: usize) -> KvCache {
        self.trunk.new_kv_cache_for_layers(layers, capacity)
    }

    fn cache_slot(&self, il: usize) -> crate::engine::kv_cache::CacheSlot {
        self.trunk.cache_slot(il)
    }

    fn embed(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        self.trunk.embed(tokens)
    }

    /// The host path over `layers`, the card's clock held as for a whole
    /// step.
    fn forward_layers(
        &self,
        cache: &mut KvCache,
        hidden: Vec<f32>,
        tokens: &[u32],
        layers: std::ops::Range<usize>,
        start_pos: usize,
    ) -> Result<Vec<f32>> {
        let _clock = super::hold_clock_for_step(self.trunk.backend.as_ref(), tokens.len());
        self.trunk
            .forward_range(cache, hidden, tokens.len(), start_pos, layers, 0)
    }

    fn head(&self, hidden: &[f32], n_rows: usize) -> Result<Vec<Vec<f32>>> {
        self.trunk.head_rows(hidden, n_rows)
    }

    /// The trunk's prompt loop with the head over every row; a recurrent
    /// layer's state takes in every token, and speculation returns it to a
    /// copy saved before the verify (`KvCache::rollback`).
    fn forward_all_logits(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        slot_id: usize,
    ) -> Result<Vec<Vec<f32>>> {
        self.trunk
            .forward_all_logits(cache, tokens, start_pos, slot_id)
    }
}

#[cfg(test)]
mod real_model_tests {
    use super::*;

    /// Cross-check against real llama.cpp: given the token IDs real
    /// llama.cpp's `/tokenize` produces for "The capital of France is"
    /// (byte-level BPE — this model's `tokenizer.ggml.model = "gpt2"`,
    /// already correctly supported, unlike gemma4's SentencePiece gap), the
    /// model should predict " Paris" (token 11751) as the top next token,
    /// matching real llama.cpp's `/completion` (`n_probs`) output exactly.
    /// Run with `ORANGU_TEST_MODEL=/path/to.gguf cargo test --release --bin
    /// orangu-server qwen35moe::real_model_tests -- --ignored` (a 35B-param
    /// model — expect several minutes: this engine's scalar per-row dequant
    /// has no hand-tuned SIMD quantized-matmul kernel).
    #[test]
    #[ignore]
    fn qwen35moe_predicts_paris_after_capital_of_france() {
        let path = std::env::var("ORANGU_TEST_MODEL").expect("set ORANGU_TEST_MODEL");
        let loaded = LoadedModel::open(std::path::Path::new(&path)).expect("load model");
        let model = Qwen35MoeModel::load_with_backend(
            &loaded,
            Arc::new(crate::engine::backend::CpuBackend),
        )
        .expect("build model");

        let mut cache = model.new_kv_cache(64);
        let tokens: Vec<u32> = vec![760, 6511, 314, 9338, 369];
        let logits = model.forward(&mut cache, &tokens, 0, 0).expect("forward");
        let (top_id, _) = logits
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        assert_eq!(top_id, 11751, "expected ' Paris' (11751) as top prediction");
    }
}

#[cfg(test)]
mod batch_tests {
    use super::*;

    /// A batch runs each layer's FFN over every row at once and a solo
    /// step over one, and the recurrence carries the difference.
    const TOLERANCE: f64 = 4e-2;

    #[test]
    #[ignore]
    fn a_decode_batch_matches_each_rows_solo_step() {
        let Some(vulkan) = crate::engine::backend::vulkan::VulkanBackend::try_init() else {
            println!("no Vulkan device; nothing to check");
            return;
        };
        let path =
            std::env::var("ORANGU_TEST_QWEN35MOE_MODEL").expect("set ORANGU_TEST_QWEN35MOE_MODEL");
        let loaded = LoadedModel::open(std::path::Path::new(&path)).expect("load model");
        let model =
            Qwen35MoeModel::load_with_backend(&loaded, Arc::new(vulkan)).expect("build model");
        super::super::batch_checks::decode_batch_matches_each_rows_solo_step(&model, TOLERANCE);
    }
}
