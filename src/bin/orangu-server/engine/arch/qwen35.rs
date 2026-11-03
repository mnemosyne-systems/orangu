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

    fn n_trunk_layer(&self) -> usize {
        self.trunk.layer_count()
    }

    fn new_kv_cache(&self, capacity: usize) -> KvCache {
        self.trunk.new_kv_cache(capacity)
    }

    fn batches_decode(&self) -> bool {
        self.trunk.batches_decode()
    }

    fn forward_decode_batch(
        &self,
        rows: &mut [super::DecodeRow<'_>],
        tokens: &[u32],
    ) -> Result<Option<Vec<Vec<f32>>>> {
        Ok(self
            .trunk
            .decode_batch(rows, &[], tokens, None)?
            .map(|outcomes| {
                outcomes
                    .into_iter()
                    .map(|outcome| match outcome {
                        super::ForwardOutcome::Logits(logits) => logits,
                        _ => unreachable!("a batch asked for no sampling returns logits"),
                    })
                    .collect()
            }))
    }

    fn forward_decode_batch_sampling(
        &self,
        rows: &mut [super::DecodeRow<'_>],
        tokens: &[u32],
        samples: &[Option<super::GreedySampleParams<'_>>],
    ) -> Result<Option<Vec<super::ForwardOutcome>>> {
        self.trunk.decode_batch(rows, &[], tokens, Some(samples))
    }

    fn batches_verify(&self) -> bool {
        super::batch_verify() && self.trunk.verifies_on_device()
    }

    fn forward_decode_batch_verify(
        &self,
        rows: &mut [super::DecodeRow<'_>],
        tokens: &[&[u32]],
        samples: &[Option<super::GreedySampleParams<'_>>],
    ) -> Result<Option<Vec<Vec<super::ForwardOutcome>>>> {
        if !self.batches_verify() || tokens.len() != rows.len() || samples.len() != rows.len() {
            return Ok(None);
        }
        let widths: Vec<usize> = tokens.iter().map(|t| t.len()).collect();
        let flat: Vec<u32> = tokens.iter().flat_map(|t| t.iter().copied()).collect();
        // A verifying row's tokens all come back as logits.
        let row_samples: Vec<Option<super::GreedySampleParams<'_>>> = samples
            .iter()
            .zip(&widths)
            .flat_map(|(sample, &w)| {
                let one = if w == 1 { *sample } else { None };
                std::iter::once(one).chain(std::iter::repeat_n(None, w - 1))
            })
            .collect();
        let Some(outcomes) = self
            .trunk
            .decode_batch(rows, &widths, &flat, Some(&row_samples))?
        else {
            return Ok(None);
        };
        let mut outcomes = outcomes.into_iter();
        Ok(Some(
            widths
                .iter()
                .map(|&w| outcomes.by_ref().take(w).collect())
                .collect(),
        ))
    }

    fn forward(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        slot_id: usize,
    ) -> Result<Vec<f32>> {
        self.trunk.forward(cache, tokens, start_pos, slot_id)
    }

    /// Every token's hidden state after the final norm — what MOSS-Image
    /// conditions on (its text encoder is Qwen3.5-9B).
    fn forward_hidden_states(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        self.trunk.hidden_states(tokens)
    }

    /// Each layer keeps its own state — an attention layer its KV rows, a
    /// gated-DeltaNet layer its recurrent state — and passes on only the
    /// residual stream, so any cut is exact.
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

    /// The host path over `layers`; the one-token device loop runs the
    /// whole model and has no range to stop at.
    fn forward_layers(
        &self,
        cache: &mut KvCache,
        hidden: Vec<f32>,
        tokens: &[u32],
        layers: std::ops::Range<usize>,
        start_pos: usize,
    ) -> Result<Vec<f32>> {
        self.trunk
            .forward_range(cache, hidden, tokens.len(), start_pos, layers, 0)
    }

    fn head(&self, hidden: &[f32], n_rows: usize) -> Result<Vec<Vec<f32>>> {
        self.trunk.head_rows(hidden, n_rows)
    }

    fn verifies_recurrent_on_device(&self) -> bool {
        self.trunk.verifies_on_device()
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

    /// A verify on the device (`Trunk::rows_on_device`) gives every
    /// position the logits the one-token loop gives it stepped a token at a
    /// time, to the batched kernels' rounding, and leaves the recurrent
    /// states where the steps leave them: the steps after it agree, and so
    /// do the steps after a verify that keeps two of its rows
    /// (`KvCache::keep_verified`) and after one rolled back to its
    /// checkpoint with its kept tokens run again. The bound is the host verify's own: it sits
    /// 1.2–2.1e-2 (rms) from the steps on Qwen3.5-0.8B, the batched
    /// kernels' rounding carried through the recurrence; a wrong conv
    /// shift or decay is orders of magnitude past it. On the GPU, so
    /// ignored by default: run with
    /// `ORANGU_TEST_MODEL=/path/to/Qwen3.5-0.8B.gguf cargo test --release
    /// --bin orangu-server verify_on_the_device -- --ignored`.
    #[test]
    #[ignore]
    fn a_verify_on_the_device_steps_as_the_token_loop_does() {
        let Some(vulkan) = crate::engine::backend::vulkan::VulkanBackend::try_init() else {
            println!("no Vulkan device; nothing to check");
            return;
        };
        let path = std::env::var("ORANGU_TEST_MODEL").expect("set ORANGU_TEST_MODEL");
        let loaded = LoadedModel::open(std::path::Path::new(&path)).expect("load model");
        assert_eq!(loaded.config.architecture, "qwen35");
        let model = Qwen35Model::load_with_backend(&loaded, Arc::new(vulkan)).expect("build model");
        let compare = |what: &str, a: &[f32], b: &[f32]| {
            let (mut diff2, mut val2) = (0f64, 0f64);
            for (x, y) in a.iter().zip(b) {
                diff2 += f64::from((x - y) * (x - y));
                val2 += f64::from(y * y);
            }
            let rms = (diff2 / val2).sqrt();
            let (pa, pb) = (
                crate::engine::sampling::argmax(a),
                crate::engine::sampling::argmax(b),
            );
            println!("{what}: rms {rms:.2e}, argmax {pa} / {pb}");
            assert!(rms < 4e-2, "{what} rms {rms}");
            assert!(
                pa == pb || (a[pb as usize] - a[pa as usize]).abs() < 0.5,
                "{what} picks {pa} where the steps pick {pb}"
            );
        };
        let prompt: Vec<u32> = (0..37u32).map(|i| 200 + (i * 41) % 5000).collect();
        // Both caches past the prompt and three one-token steps, so the
        // attention layers hold rows on the device only.
        let prepared = || {
            let mut cache = model.new_kv_cache(128);
            model.forward(&mut cache, &prompt, 0, 0).expect("prompt");
            for (k, t) in [11u32, 4000, 77].into_iter().enumerate() {
                model
                    .forward(&mut cache, &[t], prompt.len() + k, 0)
                    .expect("step");
            }
            cache
        };
        // A prompt through the rows, against the same prompt a token at a
        // time.
        {
            let mut rows = model.new_kv_cache(128);
            let a = model.forward(&mut rows, &prompt, 0, 0).expect("prompt");
            let mut steps = model.new_kv_cache(128);
            let mut b = Vec::new();
            for (k, &t) in prompt.iter().enumerate() {
                b = model.forward(&mut steps, &[t], k, 0).expect("step");
            }
            compare("prompt", &a, &b);
        }
        let mut verified = prepared();
        let mut stepped = prepared();
        let mut pos = prompt.len() + 3;
        let draft = [1500u32, 9, 2701, 640, 33];
        let step = |cache: &mut KvCache, tokens: &[u32], at: usize| -> Vec<Vec<f32>> {
            tokens
                .iter()
                .enumerate()
                .map(|(k, &t)| model.forward(cache, &[t], at + k, 0).expect("step"))
                .collect()
        };
        let rows = model
            .forward_all_logits(&mut verified, &draft, pos, 0)
            .expect("verify");
        for (k, (v, s)) in rows.iter().zip(step(&mut stepped, &draft, pos)).enumerate() {
            compare(&format!("verify row {k}"), v, &s);
        }
        pos += draft.len();
        for (k, t) in [5u32, 812].into_iter().enumerate() {
            let a = model
                .forward(&mut verified, &[t], pos + k, 0)
                .expect("step");
            let b = model.forward(&mut stepped, &[t], pos + k, 0).expect("step");
            compare(&format!("step {k} after the verify"), &a, &b);
        }
        pos += 2;
        // A verify of four with two kept, from the states behind them.
        let point = verified.rollback_point(pos).expect("checkpoint");
        model
            .forward_all_logits(&mut verified, &[40, 41, 42, 43], pos, 0)
            .expect("verify");
        assert!(
            verified.keep_verified(&point, 2),
            "the device kept the rows"
        );
        step(&mut stepped, &[40, 41], pos);
        pos += 2;
        let a = model.forward(&mut verified, &[98], pos, 0).expect("step");
        let b = model.forward(&mut stepped, &[98], pos, 0).expect("step");
        compare("step after keeping two rows", &a, &b);
        pos += 1;
        // The same, rolled back to the anchor and the two run again.
        let point = verified.rollback_point(pos).expect("checkpoint");
        model
            .forward_all_logits(&mut verified, &[70, 71, 72, 73], pos, 0)
            .expect("verify");
        verified.rollback(&point).expect("roll back");
        let a = step(&mut verified, &[70, 71], pos);
        let b = step(&mut stepped, &[70, 71], pos);
        for (k, (a, b)) in a.iter().zip(&b).enumerate() {
            compare(&format!("replay {k}"), a, b);
        }
        pos += 2;
        let a = model.forward(&mut verified, &[99], pos, 0).expect("step");
        let b = model.forward(&mut stepped, &[99], pos, 0).expect("step");
        compare("step after the replay", &a, &b);
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

/// The batched decode and verify against this model's own solo path, on a
/// real file and the GPU (`ORANGU_TEST_DEVICE` picks the adapter):
/// `ORANGU_TEST_QWEN35_MODEL=/path/to/Ornith-1.5-9B-Q4_K_M.gguf cargo test --release --bin orangu-server
/// qwen35::batch_tests -- --ignored`.
#[cfg(test)]
mod batch_tests {
    use super::*;

    /// The bound `a_verify_on_the_device_steps_as_the_token_loop_does`
    /// holds the rows path to: a batch's rows take the GEMM kernels and a
    /// solo step the decode ones, and the recurrence carries the difference.
    const TOLERANCE: f64 = 4e-2;

    fn model() -> Option<Qwen35Model> {
        let vulkan = crate::engine::backend::vulkan::VulkanBackend::try_init()?;
        let path = std::env::var("ORANGU_TEST_QWEN35_MODEL").expect("set ORANGU_TEST_QWEN35_MODEL");
        let loaded = LoadedModel::open(std::path::Path::new(&path)).expect("load model");
        Some(Qwen35Model::load_with_backend(&loaded, Arc::new(vulkan)).expect("build model"))
    }

    #[test]
    #[ignore]
    fn a_decode_batch_matches_each_rows_solo_step() {
        let Some(model) = model() else {
            println!("no Vulkan device; nothing to check");
            return;
        };
        super::super::batch_checks::decode_batch_matches_each_rows_solo_step(&model, TOLERANCE);
    }

    #[test]
    #[ignore]
    fn a_decode_batch_verifies_a_draft_as_its_own_pass_would() {
        let Some(model) = model() else {
            println!("no Vulkan device; nothing to check");
            return;
        };
        super::super::batch_checks::decode_batch_verifies_a_draft_as_its_own_pass_would(
            &model, TOLERANCE,
        );
    }
}
