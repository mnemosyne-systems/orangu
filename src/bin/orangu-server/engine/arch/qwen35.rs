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

//! Qwen3.5-family **dense** models (`general.architecture = "qwen35"`),
//! e.g. `unsloth/Ornith-1.0-9B-GGUF` and `unsloth/Qwen3.8-27B-GGUF` —
//! confirmed against real upstream `llama.cpp` source
//! (`src/models/qwen35.cpp`, read directly, not guessed).
//!
//! Every layer shape — the hybrid full-attention / gated-DeltaNet
//! alternation, the joint query+gate projection, partial rotary, the
//! delta-rule recurrent state update — lives in
//! [`engine::arch::qwen_hybrid`](super::qwen_hybrid), which
//! `src/models/qwen35.cpp` and `src/models/qwen35moe.cpp` share upstream
//! too (both call the same `llm_build_delta_net_base`). This module is only
//! the FFN choice: plain dense SwiGLU (`ffn_gate`/`ffn_up`/`ffn_down`,
//! `LLM_FFN_SILU`/`LLM_FFN_PAR` — the same computation `engine::arch::llama`
//! runs for Qwen2/Qwen3, shared with it through `super::swiglu_ffn`) instead
//! of routed + shared-expert MoE, and a GGUF's tensor names tell the loader
//! which one to expect up front rather than at graph-build time.
//!
//! See the trunk module's own doc comment for what is deliberately *not*
//! implemented (autoregressive-only gated-DeltaNet, plain NEOX rope in place
//! of multi-section RoPE, and NextN/MTP blocks — which `Qwen3.8-27B` does
//! ship, counted inside `block_count`, and which the trunk trims).

use anyhow::Result;
use std::sync::Arc;

use super::ModelForward;
use super::qwen_hybrid::{DenseFfn, Trunk};
use crate::engine::backend::Backend;
use crate::engine::kv_cache::KvCache;
use crate::engine::loader::{LoadedModel, ModelConfig};

pub struct Qwen35Model {
    trunk: Trunk<DenseFfn>,
}

impl Qwen35Model {
    pub fn load_with_backend(loaded: &LoadedModel, backend: Arc<dyn Backend>) -> Result<Self> {
        Ok(Self {
            trunk: Trunk::load_dense(loaded, backend)?,
        })
    }
}

impl ModelForward for Qwen35Model {
    fn vulkan_backend(&self) -> Option<&crate::engine::backend::vulkan::VulkanBackend> {
        self.trunk.backend.as_wgpu()
    }

    fn config(&self) -> &ModelConfig {
        &self.trunk.config
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
        anyhow::bail!("embeddings are not yet supported for Qwen3.5 models")
    }
}

#[cfg(test)]
mod real_model_tests {
    use super::*;

    /// Cross-check against real llama.cpp (`unsloth/Ornith-1.0-9B-GGUF:
    /// Q4_K_M`, `llama-cli`/`llama-server` build b10066): given the token
    /// IDs real llama.cpp's `/tokenize` produces for "The capital of
    /// France is" (byte-level BPE — this model's `tokenizer.ggml.model =
    /// "gpt2"`), the model should predict the same top next token real
    /// llama.cpp's own `/completion` (`n_probs`) output does. Run with
    /// `ORANGU_TEST_MODEL=/path/to.gguf cargo test --release --bin
    /// orangu-server qwen35::real_model_tests -- --ignored` (a 9B-param
    /// model — expect a couple of minutes: this engine's scalar per-row
    /// dequant has no hand-tuned SIMD quantized-matmul kernel).
    #[test]
    #[ignore]
    fn qwen35_predicts_paris_after_capital_of_france() {
        let path = std::env::var("ORANGU_TEST_MODEL").expect("set ORANGU_TEST_MODEL");
        let loaded = LoadedModel::open(std::path::Path::new(&path)).expect("load model");
        assert_eq!(loaded.config.architecture, "qwen35");
        let model =
            Qwen35Model::load_with_backend(&loaded, Arc::new(crate::engine::backend::CpuBackend))
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

    /// The same cross-check for `unsloth/Qwen3.8-27B-GGUF:Q4_K_M`, which is
    /// the file that made the trunk's NextN/MTP trim load-bearing for a
    /// *dense* `qwen35`: `block_count` is 65 and `blk.64` is a
    /// multi-token-prediction head carrying no attention tensors, so before
    /// the trim this model failed to load at all on `blk.64.attn_qkv.weight`.
    /// The layer-count assertion is therefore part of the test, not a
    /// precondition — a file where it does not hold is not exercising this.
    ///
    /// Reference taken from real `llama.cpp` (`llama-server` build 10423,
    /// this exact file, `--device none`): `/tokenize` on "The capital of
    /// France is" gives `[760, 6511, 314, 9338, 369]`, and `/completion`
    /// with `n_probs` ranks ` Paris` (11751) first at logprob -0.354, ahead
    /// of ` not` (524) at -3.619.
    ///
    /// Run with `ORANGU_TEST_QWEN38_MODEL=/path/to/Qwen3.8-27B-Q4_K_M.gguf
    /// cargo test --release --bin orangu-server qwen35::real_model_tests --
    /// --ignored` (a 27B-param model on the CPU path — expect minutes).
    #[test]
    #[ignore]
    fn qwen38_dense_predicts_paris_after_capital_of_france() {
        let path = std::env::var("ORANGU_TEST_QWEN38_MODEL")
            .expect("set ORANGU_TEST_QWEN38_MODEL to a Qwen3.8-27B GGUF");
        let loaded = LoadedModel::open(std::path::Path::new(&path)).expect("load model");
        assert_eq!(loaded.config.architecture, "qwen35");
        assert_eq!(
            super::super::qwen_hybrid::trunk_layer_count(&loaded).expect("trunk layers"),
            loaded.config.n_layer - 1,
            "this file should declare one NextN/MTP block inside block_count"
        );
        let model =
            Qwen35Model::load_with_backend(&loaded, Arc::new(crate::engine::backend::CpuBackend))
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
