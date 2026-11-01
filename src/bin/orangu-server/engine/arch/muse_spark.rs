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

//! Muse-Spark (`general.architecture = "muse-spark"`, e.g.
//! `unsloth/Muse-Spark-Open-816B-A42B-GGUF`), transcribed from upstream
//! `llama.cpp`'s `src/models/muse-spark.cpp` and the `post_exp_norm` arm of
//! `llm_graph_context::build_moe_ffn` (`src/llama-graph.cpp`), read
//! directly rather than guessed.
//!
//! `muse-glimmer`'s attention with a mixture of experts behind it, so the
//! attention is `engine::arch::muse`'s [`GatedAttention`] — the same Q/K/V
//! and sigmoid gate, the same per-head QK-norm, the same NoPE on the
//! full-attention layers — and the experts go through
//! [`super::swiglu_moe_ffn_with`], the routed SwiGLU mixture `glm` and
//! `bailingmoe` share. What is new is around them:
//!
//! - **A weightless per-head norm on attention's output**, before the gate
//!   (`attn_o_norm`, at the file's own epsilon).
//! - **Weightless post-norms.** The post-attention norm has no weight at
//!   all; `post_ffw_norm` multiplies by its weight *first* and then
//!   normalizes — the reverse of every other sandwich norm here. Both use
//!   `post_norm_epsilon` from the file.
//! - **A norm-preserving residual.** Each sub-layer's residual add is
//!   `stream * alpha + branch * beta`, both per-channel vectors
//!   (`attn_resid_alpha`/`_beta`, `ffn_resid_alpha`/`_beta`) resolved from
//!   one learned gate at conversion so that `alpha^2 + beta^2 = 1`.
//! - **A latent MoE**, the shape `kimi3` also has: every layer routes, the
//!   router scores the full-width normed input, and the experts run in a
//!   `moe_latent_size`-wide space entered through `ffn_latent_down` (then
//!   `ffn_pre_exp_norm`) and left through `ffn_latent_up`. Each expert's
//!   output is multiplied by `ffn_post_exp_norm` and normalized **before**
//!   its router weight applies. There is no shared expert.
//! - **The sliding-window pattern is a per-layer array**, not
//!   `muse-glimmer`'s scalar period — [`super::muse::swa_layers`] reads
//!   either — and the rotated layers take `rope.freq_base_swa`.
//!
//! The embeddings get the same weightless RMSNorm on the way in, and the
//! tail the same `logit_scale` then `final_logit_softcapping`.
//!
//! None of the fused Vulkan chains can express this layer — the residual
//! is not an add, and the FFN is a mixture — so it is CPU-orchestrated like
//! `kimi3`, `glm` and `deepseek4`: every matmul still dispatches to the
//! `Backend`, and attention still goes wherever `engine::attention`
//! decides.
//!
//! Multimodal input is out of scope, as it is for every architecture here:
//! the `mmproj-*.gguf` shipped beside these weights is a separate model
//! this engine does not load.

use anyhow::{Context, Result};
use std::sync::Arc;

use super::muse::{GatedAttention, GatedAttentionShape, LogitTransforms, swa_layers};
use super::{ExpertRouting, ModelForward, SwigluLimit, SwigluMoe};
use crate::engine::backend::Backend;
use crate::engine::kv_cache::KvCache;
use crate::engine::loader::{ExpertQuantMatrix, LoadedModel, ModelConfig, QuantMatrix};
use crate::engine::tensor;

/// The routed experts of one layer, which run in the latent space.
struct LatentMoe {
    /// `[n_embd, n_expert]` — scores the *full-width* normed input.
    gate_inp: QuantMatrix,
    exp_probs_b: Option<Vec<f32>>,
    /// `[n_embd, moe_latent_size]` / `[moe_latent_size, n_embd]`.
    latent_down: QuantMatrix,
    latent_up: QuantMatrix,
    /// `[moe_latent_size]` — on the experts' input, at the file's epsilon.
    pre_exp_norm: Vec<f32>,
    /// `[moe_latent_size]` — on each expert's output, before its router
    /// weight, at `post_norm_epsilon`.
    post_exp_norm: Vec<f32>,
    gate_exps: ExpertQuantMatrix,
    up_exps: ExpertQuantMatrix,
    down_exps: ExpertQuantMatrix,
}

struct MuseSparkLayer {
    attn_norm: Vec<f32>,
    attn: GatedAttention,
    attn_resid_alpha: Vec<f32>,
    attn_resid_beta: Vec<f32>,
    ffn_norm: Vec<f32>,
    moe: LatentMoe,
    /// `post_ffw_norm` — multiplied in *before* the normalization.
    ffn_post_norm: Vec<f32>,
    ffn_resid_alpha: Vec<f32>,
    ffn_resid_beta: Vec<f32>,
}

pub struct MuseSparkModel {
    config: ModelConfig,
    backend: Arc<dyn Backend>,
    tok_embeddings: QuantMatrix,
    output_norm: Vec<f32>,
    output_weight: QuantMatrix,
    layers: Vec<MuseSparkLayer>,
    /// `attention.sliding_window`.
    n_swa: usize,
    /// `post_norm_epsilon` — the post-attention, post-FFN and per-expert
    /// norms. Every other norm uses the file's
    /// `attention.layer_norm_rms_epsilon`.
    post_norm_eps: f32,
    routing: ExpertRouting,
    logits: LogitTransforms,
    /// NORM pairing, as `muse-glimmer`: the converter un-permutes Q and K
    /// out of the reference's rotate-half layout. Applied on the
    /// sliding-window layers only, at `rope.freq_base_swa`.
    rope: tensor::RopeParams,
}

impl MuseSparkModel {
    pub fn load_with_backend(loaded: &LoadedModel, backend: Arc<dyn Backend>) -> Result<Self> {
        let config = loaded.config.clone();
        let n_layer = config.n_layer;

        let n_swa = loaded
            .metadata_u64("attention.sliding_window")
            .context("missing attention.sliding_window")? as usize;
        let post_norm_eps = loaded
            .metadata_f32("post_norm_epsilon")
            .context("missing post_norm_epsilon")?;
        let n_latent = loaded
            .metadata_u64("moe_latent_size")
            .context("missing moe_latent_size")? as usize;
        let is_swa = swa_layers(loaded, n_layer);

        let routing = ExpertRouting::from_gguf(loaded)?;

        let tok_embeddings = loaded
            .matrix("token_embd.weight")
            .context("loading token_embd.weight")?;
        let (output_norm, _) = loaded
            .tensor("output_norm.weight")
            .context("loading output_norm.weight")?;
        let output_weight = loaded
            .matrix("output.weight")
            .context("loading output.weight")?;

        let mut layers = Vec::with_capacity(n_layer);
        for (i, &is_swa) in is_swa.iter().enumerate() {
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
            let moe = LatentMoe {
                gate_inp: get_matrix("ffn_gate_inp.weight")?,
                exp_probs_b: get("exp_probs_b.bias").ok(),
                latent_down: get_matrix("ffn_latent_down.weight")?,
                latent_up: get_matrix("ffn_latent_up.weight")?,
                pre_exp_norm: get("ffn_pre_exp_norm.weight")?,
                post_exp_norm: get("ffn_post_exp_norm.weight")?,
                gate_exps: get_expert_matrix("ffn_gate_exps.weight")?,
                up_exps: get_expert_matrix("ffn_up_exps.weight")?,
                down_exps: get_expert_matrix("ffn_down_exps.weight")?,
            };
            anyhow::ensure!(
                moe.gate_exps.in_dim == n_latent && moe.down_exps.out_dim == n_latent,
                "layer {i}'s routed experts map {} -> {}, not moe_latent_size ({n_latent})",
                moe.gate_exps.in_dim,
                moe.down_exps.out_dim
            );
            layers.push(MuseSparkLayer {
                attn_norm: get("attn_norm.weight")?,
                attn: GatedAttention::load(loaded, i, is_swa)?,
                attn_resid_alpha: get("attn_resid_alpha.weight")?,
                attn_resid_beta: get("attn_resid_beta.weight")?,
                ffn_norm: get("ffn_norm.weight")?,
                moe,
                ffn_post_norm: get("post_ffw_norm.weight")?,
                ffn_resid_alpha: get("ffn_resid_alpha.weight")?,
                ffn_resid_beta: get("ffn_resid_beta.weight")?,
            });
        }

        let rope = tensor::RopeParams {
            rope_dim: config.rope_dim,
            freq_base: loaded
                .metadata_f32("rope.freq_base_swa")
                .unwrap_or(config.rope_freq_base),
            layout: tensor::RopeLayout::Norm,
            ..tensor::RopeParams::default()
        };
        Ok(Self {
            config,
            backend,
            tok_embeddings,
            output_norm,
            output_weight,
            layers,
            n_swa,
            post_norm_eps,
            routing,
            logits: LogitTransforms::from_gguf(loaded),
            rope,
        })
    }

    fn attention_shape(&self) -> GatedAttentionShape<'_> {
        GatedAttentionShape {
            n_head: self.config.n_head,
            n_head_kv: self.config.n_head_kv,
            head_dim: self.config.head_dim,
            eps: self.config.rms_eps,
            n_swa: self.n_swa,
            rope: &self.rope,
        }
    }

    /// Every layer, from the token embeddings to the last residual —
    /// `[n_tokens, n_embd]`.
    fn run_layers(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let n_tokens = tokens.len();
        let n_embd = cfg.n_embd;
        let eps = cfg.rms_eps;
        // Host-orchestrated with a few small device dispatches a token —
        // the shape `kimi3` holds the card's clock for. See there.
        let _clock = super::hold_clock_for_step(self.backend.as_ref(), n_tokens);
        let _ops = super::OpSpan::open(self.backend.as_ref(), start_pos);

        let mut x = vec![0f32; n_tokens * n_embd];
        for (t, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            anyhow::ensure!(tok < cfg.n_vocab, "token id {tok} is out of vocab range");
            x[t * n_embd..(t + 1) * n_embd].copy_from_slice(&self.tok_embeddings.row(tok));
        }
        // The same weightless embedding norm `muse-glimmer` has.
        super::rms_norm_rows(&mut x, n_embd, eps);

        let shape = self.attention_shape();
        let mut normed: Vec<f32> = Vec::new();
        let mut attn_out: Vec<f32> = Vec::new();
        for (il, layer) in self.layers.iter().enumerate() {
            tensor::rmsnorm_into(&mut normed, &x, &layer.attn_norm, n_tokens, n_embd, eps);
            let mut attn = layer.attn.forward_into(
                self.backend.as_ref(),
                &shape,
                &mut cache.layers[il],
                &mut attn_out,
                &normed,
                n_tokens,
                start_pos,
                Some(eps),
            );
            // The post-attention norm holds no weight.
            super::rms_norm_rows(&mut attn, n_embd, self.post_norm_eps);
            norm_preserving_residual(
                &mut x,
                &attn,
                &layer.attn_resid_alpha,
                &layer.attn_resid_beta,
            );

            tensor::rmsnorm_into(&mut normed, &x, &layer.ffn_norm, n_tokens, n_embd, eps);
            let mut ffn = self.latent_moe(&layer.moe, &normed, n_tokens, il);
            // Scale, *then* normalize — not the weighted norm.
            for row in ffn.chunks_mut(n_embd) {
                tensor::mul_inplace(row, &layer.ffn_post_norm);
            }
            super::rms_norm_rows(&mut ffn, n_embd, self.post_norm_eps);
            norm_preserving_residual(&mut x, &ffn, &layer.ffn_resid_alpha, &layer.ffn_resid_beta);
        }
        Ok(x)
    }

    /// Down into the latent space, the routed experts there, and back up.
    /// The router reads `normed`, the full-width input — not the latent one.
    fn latent_moe(&self, moe: &LatentMoe, normed: &[f32], n_tokens: usize, il: usize) -> Vec<f32> {
        let n_latent = moe.latent_down.out_dim;
        let mut latent = self.backend.matmul(normed, n_tokens, &moe.latent_down);
        tensor::rmsnorm_inplace(
            &mut latent,
            &moe.pre_exp_norm,
            n_tokens,
            n_latent,
            self.config.rms_eps,
        );
        let routed = super::swiglu_moe_ffn_with(
            self.backend.as_ref(),
            &self.routing,
            normed,
            &latent,
            n_tokens,
            n_latent,
            &SwigluMoe {
                gate_inp: &moe.gate_inp,
                exp_probs_b: moe.exp_probs_b.as_deref(),
                gate_exps: &moe.gate_exps,
                up_exps: &moe.up_exps,
                down_exps: &moe.down_exps,
                shared: None,
                clamp_exp: SwigluLimit::None,
                clamp_shexp: SwigluLimit::None,
            },
            Some((&moe.post_exp_norm, self.post_norm_eps)),
            il,
        );
        self.backend.matmul(&routed, n_tokens, &moe.latent_up)
    }

    /// The output norm and projection over `[n_tokens, n_embd]` hidden
    /// states, each row's logits scaled and capped.
    fn project(&self, hidden: &mut [f32], n_tokens: usize) -> Vec<f32> {
        let n_embd = self.config.n_embd;
        tensor::rmsnorm_inplace(
            hidden,
            &self.output_norm,
            n_tokens,
            n_embd,
            self.config.rms_eps,
        );
        let mut logits = self.backend.matmul(hidden, n_tokens, &self.output_weight);
        for row in logits.chunks_mut(self.config.n_vocab) {
            self.logits.apply(row);
        }
        logits
    }
}

/// `x = x * alpha + branch * beta`, per channel — upstream's
/// `ggml_add(ggml_mul(inp, alpha), ggml_mul(cur, beta))`, with `alpha` and
/// `beta` broadcast over the rows.
fn norm_preserving_residual(x: &mut [f32], branch: &[f32], alpha: &[f32], beta: &[f32]) {
    debug_assert_eq!(x.len(), branch.len());
    let n_embd = alpha.len();
    for (row, branch_row) in x.chunks_mut(n_embd).zip(branch.chunks(n_embd)) {
        for (((v, &b), &a), &w) in row.iter_mut().zip(branch_row).zip(alpha).zip(beta) {
            *v = *v * a + b * w;
        }
    }
}

impl ModelForward for MuseSparkModel {
    fn vulkan_backend(&self) -> Option<&crate::engine::backend::vulkan::VulkanBackend> {
        self.backend.as_wgpu()
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn new_kv_cache(&self, capacity: usize) -> KvCache {
        // Every layer caches the same shape — a sliding-window layer still
        // *stores* every position, it just attends fewer of them.
        let kv_dim = self.config.n_head_kv * self.config.head_dim;
        KvCache::new(self.config.n_layer, capacity, kv_dim)
    }

    fn forward(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        _slot_id: usize,
    ) -> Result<Vec<f32>> {
        anyhow::ensure!(!tokens.is_empty(), "forward called with no tokens");
        let n_embd = self.config.n_embd;
        let x = self.run_layers(cache, tokens, start_pos)?;
        // Only the last token's logits are needed for the next token.
        let mut last = x[(tokens.len() - 1) * n_embd..].to_vec();
        Ok(self.project(&mut last, 1))
    }

    fn forward_all_logits(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        _slot_id: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let n_tokens = tokens.len();
        let n_vocab = self.config.n_vocab;
        let mut h = self.run_layers(cache, tokens, start_pos)?;
        let flat = self.project(&mut h, n_tokens);
        anyhow::ensure!(
            flat.len() == n_tokens * n_vocab,
            "output projection produced {} logits, expected {}",
            flat.len(),
            n_tokens * n_vocab
        );
        Ok(flat.chunks(n_vocab).map(<[f32]>::to_vec).collect())
    }

    fn forward_hidden_states(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        anyhow::ensure!(
            !tokens.is_empty(),
            "forward_hidden_states called with no tokens"
        );
        let mut cache = self.new_kv_cache(tokens.len());
        let mut x = self.run_layers(&mut cache, tokens, 0)?;
        tensor::rmsnorm_inplace(
            &mut x,
            &self.output_norm,
            tokens.len(),
            self.config.n_embd,
            self.config.rms_eps,
        );
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::norm_preserving_residual;

    /// Alpha scales the stream and beta the branch — swapping them is the
    /// easy mistake, and with `alpha^2 + beta^2 = 1` it still keeps the
    /// norm, so nothing downstream would catch it.
    #[test]
    fn the_residual_scales_the_stream_by_alpha_and_the_branch_by_beta() {
        let mut x = vec![1.0, 2.0, 10.0, 20.0];
        let branch = vec![3.0, 4.0, 30.0, 40.0];
        norm_preserving_residual(&mut x, &branch, &[0.6, 0.8], &[0.8, 0.6]);
        let want = [
            1.0 * 0.6 + 3.0 * 0.8,
            2.0 * 0.8 + 4.0 * 0.6,
            10.0 * 0.6 + 30.0 * 0.8,
            20.0 * 0.8 + 40.0 * 0.6,
        ];
        for (got, want) in x.iter().zip(want) {
            assert!((got - want).abs() < 1e-5, "{got} vs {want}");
        }
    }
}

#[cfg(test)]
mod real_model_tests {
    use super::*;

    /// End-to-end against a real `unsloth/Muse-Spark-Open-816B-A42B-GGUF`
    /// file, on the predicted token for the reason `arch::muse`'s
    /// equivalent test gives: a wrong-but-plausible forward pass survives a
    /// logit tolerance, not a factual one-word answer.
    ///
    /// Run with `ORANGU_TEST_MUSE_SPARK_MODEL=/path/to/Muse-Spark-Open-816B-A42B-UD-Q4_K_XL-00001-of-00012.gguf
    /// cargo test --release --bin orangu-server muse_spark::real_model_tests --
    /// --ignored`.
    #[test]
    #[ignore]
    fn muse_spark_predicts_paris_after_capital_of_france() {
        let path = std::env::var("ORANGU_TEST_MUSE_SPARK_MODEL")
            .expect("set ORANGU_TEST_MUSE_SPARK_MODEL");
        let gguf = orangu::gguf::GgufFile::open(std::path::Path::new(&path)).expect("open gguf");
        let tokenizer =
            crate::engine::tokenizer::Tokenizer::from_gguf(&gguf).expect("build tokenizer");
        let loaded = LoadedModel::open(std::path::Path::new(&path)).expect("load model");
        assert_eq!(loaded.config.architecture, "muse-spark");
        let model = MuseSparkModel::load_with_backend(
            &loaded,
            Arc::new(crate::engine::backend::CpuBackend),
        )
        .expect("build model");
        // The pattern is a per-layer array here — the second layer of every
        // four attends the whole prefix — so 46 of the 62 layers rotate.
        assert_eq!(model.layers.iter().filter(|l| l.attn.is_swa).count(), 46);

        let tokens = tokenizer.encode("The capital of France is", true);
        let mut cache = model.new_kv_cache(tokens.len() + 1);
        let logits = model.forward(&mut cache, &tokens, 0, 0).expect("forward");
        let (top_id, _) = logits
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        assert_eq!(
            tokenizer.decode(&[top_id as u32]),
            " Paris",
            "top prediction was {:?}",
            tokenizer.decode(&[top_id as u32])
        );
    }

    /// `forward_all_logits` — what speculative decoding verifies a draft
    /// with — must pick, at every position, the token one-at-a-time decoding
    /// through the KV cache picks. A cache that rotates or windows the wrong
    /// positions agrees on the first token and parts company later.
    ///
    /// Same file and variable as the test above; about five minutes on CPU.
    #[test]
    #[ignore]
    fn muse_spark_all_logits_agree_with_step_by_step_decoding() {
        let path = std::env::var("ORANGU_TEST_MUSE_SPARK_MODEL")
            .expect("set ORANGU_TEST_MUSE_SPARK_MODEL");
        let gguf = orangu::gguf::GgufFile::open(std::path::Path::new(&path)).expect("open gguf");
        let tokenizer =
            crate::engine::tokenizer::Tokenizer::from_gguf(&gguf).expect("build tokenizer");
        let loaded = LoadedModel::open(std::path::Path::new(&path)).expect("load model");
        let model = MuseSparkModel::load_with_backend(
            &loaded,
            Arc::new(crate::engine::backend::CpuBackend),
        )
        .expect("build model");
        let argmax = |row: &[f32]| {
            row.iter()
                .copied()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap()
                .0
        };

        let tokens = tokenizer.encode("The capital of France is Paris. The capital", true);
        let mut cache = model.new_kv_cache(tokens.len());
        let batched: Vec<usize> = model
            .forward_all_logits(&mut cache, &tokens, 0, 0)
            .expect("forward_all_logits")
            .iter()
            .map(|row| argmax(row))
            .collect();

        let mut cache = model.new_kv_cache(tokens.len());
        let stepped: Vec<usize> = tokens
            .iter()
            .enumerate()
            .map(|(pos, &tok)| argmax(&model.forward(&mut cache, &[tok], pos, 0).expect("forward")))
            .collect();
        assert_eq!(batched, stepped);
    }
}
