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
        _slot_id: usize,
    ) -> Result<Vec<f32>> {
        self.trunk.forward(cache, tokens, start_pos)
    }

    fn forward_hidden_states(&self, _tokens: &[u32]) -> Result<Vec<f32>> {
        anyhow::bail!("embeddings are not yet supported for Qwen3.5-MoE models")
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
