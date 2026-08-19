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

//! Muse-Glimmer (`general.architecture = "muse-glimmer"`, e.g.
//! `unsloth/Muse-Glimmer-30B-GGUF`), confirmed against real upstream
//! `llama.cpp` source (`src/models/muse-glimmer.cpp`, fetched and read
//! directly, not guessed).
//!
//! A dense GQA decoder — `engine::arch::llama`'s block plus five things,
//! four of which some other architecture here already has, so most of this
//! module is composition over the shared pieces (`engine::attention`,
//! `tensor`'s rope/norm/activation kernels, `super::rms_norm_rows`) rather
//! than new math:
//!
//! - **Sandwich norms.** `attn_norm`/`post_attention_norm` around
//!   attention and `ffn_norm`/`post_ffw_norm` around the FFN — gemma2's
//!   arrangement, the post-norm applied to the sub-layer's output before
//!   its residual add. All four are present on every layer, which is what
//!   distinguishes this from `engine::arch::qwen35`'s two-norm block
//!   (where `post_attention_norm` *is* the pre-FFN norm and there is no
//!   `ffn_norm`). The two **post**-norms use their own epsilon,
//!   [`POST_NORM_EPS`], not the file's `attention.layer_norm_rms_epsilon`.
//! - **A sigmoid output gate on attention.** `attn_gate`
//!   (`[n_embd, n_head * head_dim]`) is projected from the same normed
//!   layer input as Q/K/V, and `sigmoid` of it multiplies attention's
//!   output element-wise *before* `attn_output` — the gate
//!   `engine::arch::kimi3`'s MLA layers and `engine::arch::qwen35`'s
//!   full-attention layers also carry (they fold it into the query
//!   projection; here it is its own tensor).
//! - **Per-head QK-norm**, `[head_dim]` RMSNorm weights on Q and K before
//!   the rotation, as in `gemma`/`qwen3`. Nothing extra is folded in here
//!   at run time: upstream notes this model's query-key scale factor was
//!   baked into `attn_q_norm`'s weights at conversion, and `attn_k_norm`
//!   is a vector of ones.
//! - **Alternating local/global attention.** A *scalar*
//!   `attention.sliding_window_pattern` (default 4 when absent) drives
//!   upstream's `llama_hparams::set_swa_pattern`: `il % n_pattern <
//!   n_pattern - 1` is sliding-window, so every `n_pattern`th layer
//!   attends the whole prefix. The same formula `gemma-embedding` uses at
//!   period 6.
//! - **RoPE on the sliding-window layers only** — the full-attention
//!   layers are NoPE, rotating neither Q nor K. This is the one mechanism
//!   with no counterpart elsewhere in this engine, and it is invisible
//!   from the file: nothing in the metadata or the tensor directory says
//!   a layer does not rotate.
//!
//! The tail applies both scalar logit transforms this engine has otherwise
//! seen separately: `logit_scale` (a plain multiply, as Command-R has) and
//! then `final_logit_softcapping` (`tanh`-based, as gemma2 has). The token
//! embeddings are **weightlessly** RMS-normed on the way in — an ordinary
//! `ggml_rms_norm` with no learned weight, where gemma instead scales by
//! `sqrt(n_embd)` and the llama family does nothing at all.
//!
//! Deliberately **not** taken: the whole-half fused Vulkan prefill chains
//! (`fused_attention_prefill`/`fused_post_attention_prefill`) that
//! `engine::arch::llama` and `engine::arch::gemma` run on. Neither can
//! express this layer — the pre-attention chain rotates unconditionally
//! and this architecture skips RoPE on a quarter of its layers, and the
//! post-attention chain takes a single epsilon for all three of its norms
//! where this needs [`POST_NORM_EPS`] for two of them and the file's own
//! epsilon for the third. So this is CPU-orchestrated like
//! `engine::arch::qwen35`, `glm`, `kimi3`, `deepseek4` and `qwen35moe`:
//! every matmul still dispatches to the `Backend` (GPU included), and
//! attention still goes wherever `engine::attention` decides, but the
//! layer is driven from the host.
//!
//! Multimodal input is out of scope, as it is for every architecture here:
//! the `mmproj-*.gguf` shipped beside these weights is a separate
//! `clip`-architecture model this engine does not load.

use anyhow::{Context, Result};
use std::sync::Arc;

use super::ModelForward;
use crate::engine::backend::{Backend, MatmulOp};
use crate::engine::kv_cache::KvCache;
use crate::engine::loader::{LoadedModel, ModelConfig, QuantMatrix};
use crate::engine::tensor;

/// The epsilon the two **post**-norms use, hardcoded upstream as
/// `post_norm_eps` rather than read from the file — which sets
/// `attention.layer_norm_rms_epsilon` (1e-5 for the 30B checkpoint) and is
/// what every *other* norm here uses. Three orders of magnitude apart, so
/// this is not a rounding choice somebody made twice.
const POST_NORM_EPS: f32 = 1e-8;

/// The sliding-window period when the file sets no
/// `attention.sliding_window_pattern`, matching upstream's own default for
/// this architecture.
///
/// Not a cosmetic fallback: the pattern decides which layers are
/// sliding-window, and *that* decides which layers rotate at all (see
/// [`MuseLayer::is_swa`]). Defaulting it to "no pattern" would leave a file
/// without the key with no RoPE anywhere.
const DEFAULT_SWA_PATTERN: usize = 4;

struct MuseLayer {
    attn_norm: Vec<f32>,
    wq: QuantMatrix,
    wk: QuantMatrix,
    wv: QuantMatrix,
    /// `attn_gate`, `[n_embd, n_head * head_dim]` — projected from the same
    /// normed input as Q/K/V; `sigmoid` of it gates attention's output
    /// before `wo`.
    w_gate: QuantMatrix,
    wo: QuantMatrix,
    /// `[head_dim]`, the same vector for every head, applied before RoPE.
    attn_q_norm: Vec<f32>,
    attn_k_norm: Vec<f32>,
    /// `post_attention_norm` — on attention's *output*, before the residual
    /// add, on top of the usual `attn_norm` pre-norm. Uses
    /// [`POST_NORM_EPS`].
    attn_post_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    ffn_gate: QuantMatrix,
    ffn_up: QuantMatrix,
    ffn_down: QuantMatrix,
    /// `post_ffw_norm` — the FFN's half of the same sandwich, same epsilon.
    ffn_post_norm: Vec<f32>,
    /// Whether this layer attends a sliding window rather than the whole
    /// prefix — three layers in every four at the default period.
    ///
    /// It also decides whether this layer rotates: upstream's `use_rope =
    /// hparams.is_swa(il)`. A full-attention layer here is NoPE, and its
    /// cached keys are therefore stored unrotated.
    is_swa: bool,
}

pub struct MuseModel {
    config: ModelConfig,
    backend: Arc<dyn Backend>,
    tok_embeddings: QuantMatrix,
    output_norm: Vec<f32>,
    output_weight: QuantMatrix,
    layers: Vec<MuseLayer>,
    /// `attention.sliding_window` — the width a [`MuseLayer::is_swa`] layer
    /// attends.
    n_swa: usize,
    /// `logit_scale`, multiplied into the output projection's result.
    logit_scale: Option<f32>,
    /// `final_logit_softcapping` — `x -> tanh(x / cap) * cap`, applied
    /// after [`MuseModel::logit_scale`] and never inside the layer stack.
    final_logit_softcapping: Option<f32>,
    /// Rotary width, base and pairing. `muse-glimmer` sits in upstream's
    /// `LLAMA_ROPE_TYPE_NORM` arm — pairs of *consecutive* head values,
    /// like `llama`/`mistral` and unlike the gemma and Qwen families this
    /// block shape otherwise resembles.
    rope: tensor::RopeParams,
}

impl MuseModel {
    pub fn load_with_backend(loaded: &LoadedModel, backend: Arc<dyn Backend>) -> Result<Self> {
        let config = loaded.config.clone();
        let n_layer = config.n_layer;

        let n_swa = loaded
            .metadata_u64("attention.sliding_window")
            .context("missing attention.sliding_window")? as usize;
        // A *scalar* period, not the per-layer array `gemma4` ships.
        // Upstream reads it as either and prefers the scalar; only the
        // scalar form is seen in the released checkpoints, and an array
        // would name each layer directly.
        let swa_pattern = loaded
            .metadata_u64("attention.sliding_window_pattern")
            .map(|v| v as usize)
            .unwrap_or(DEFAULT_SWA_PATTERN);
        let is_swa_per_layer = loaded.metadata_array_u64("attention.sliding_window_pattern");
        let logit_scale = loaded.metadata_f32("logit_scale");
        let final_logit_softcapping = loaded.metadata_f32("final_logit_softcapping");

        let tok_embeddings = loaded
            .matrix("token_embd.weight")
            .context("loading token_embd.weight")?;
        let (output_norm, _) = loaded
            .tensor("output_norm.weight")
            .context("loading output_norm.weight")?;
        // Upstream requires `output.weight` for this architecture; the tied
        // fallback costs nothing and matches every other module here.
        let output_weight = if loaded.has_tensor("output.weight") {
            loaded
                .matrix("output.weight")
                .context("loading output.weight")?
        } else {
            tok_embeddings.clone()
        };

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
            let is_swa = match &is_swa_per_layer {
                Some(per_layer) => per_layer.get(i).copied().unwrap_or(0) != 0,
                None => swa_pattern == 0 || i % swa_pattern < swa_pattern - 1,
            };
            layers.push(MuseLayer {
                attn_norm: get("attn_norm.weight")?,
                wq: get_matrix("attn_q.weight")?,
                wk: get_matrix("attn_k.weight")?,
                wv: get_matrix("attn_v.weight")?,
                w_gate: get_matrix("attn_gate.weight")?,
                wo: get_matrix("attn_output.weight")?,
                attn_q_norm: get("attn_q_norm.weight")?,
                attn_k_norm: get("attn_k_norm.weight")?,
                attn_post_norm: get("post_attention_norm.weight")?,
                ffn_norm: get("ffn_norm.weight")?,
                ffn_gate: get_matrix("ffn_gate.weight")?,
                ffn_up: get_matrix("ffn_up.weight")?,
                ffn_down: get_matrix("ffn_down.weight")?,
                ffn_post_norm: get("post_ffw_norm.weight")?,
                is_swa,
            });
        }

        let rope = tensor::RopeParams {
            rope_dim: config.rope_dim,
            freq_base: config.rope_freq_base,
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
            logit_scale,
            final_logit_softcapping,
            rope,
        })
    }

    fn head_dim(&self) -> usize {
        self.config.head_dim
    }

    /// The inclusive `[start, end]` key/value position range a query at
    /// absolute position `pos` may attend. Causal throughout — this is a
    /// decoder — so the only variable is whether the window is the whole
    /// prefix or the trailing `n_swa` positions, which is upstream's
    /// `is_masked_swa` for `LLAMA_SWA_TYPE_STANDARD` (`p0 < p1 - n_swa` is
    /// masked).
    fn attention_window(&self, is_swa: bool, pos: usize) -> (usize, usize) {
        if is_swa && self.n_swa > 0 {
            (pos.saturating_sub(self.n_swa - 1), pos)
        } else {
            (0, pos)
        }
    }

    /// Every layer, from the token embeddings to the last residual —
    /// `[n_tokens, n_embd]`. Shared by `forward`, `forward_all_logits` and
    /// `forward_hidden_states`, which differ only in what they do with it.
    fn run_layers(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let n_tokens = tokens.len();
        let n_embd = cfg.n_embd;
        let head_dim = self.head_dim();
        let n_head = cfg.n_head;
        let n_head_kv = cfg.n_head_kv;
        let kv_dim = n_head_kv * head_dim;
        let eps = cfg.rms_eps;

        let mut x = vec![0f32; n_tokens * n_embd];
        for (t, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            anyhow::ensure!(tok < cfg.n_vocab, "token id {tok} is out of vocab range");
            x[t * n_embd..(t + 1) * n_embd].copy_from_slice(&self.tok_embeddings.row(tok));
        }
        // A *weightless* RMSNorm on the embeddings, before any layer runs —
        // gemma scales by `sqrt(n_embd)` here and the llama family does
        // nothing, so neither of those shapes would produce this.
        super::rms_norm_rows(&mut x, n_embd, eps);

        // Grown once and reused across layers rather than allocated per
        // layer: at prefill widths this is megabytes a layer.
        let mut attn_out: Vec<f32> = Vec::new();
        // Same reuse as `attn_out`, replacing what used to be `x.clone()`
        // per norm — see `tensor::rmsnorm_into`.
        let mut normed: Vec<f32> = Vec::new();
        let mut ffn_normed: Vec<f32> = Vec::new();
        // Projection outputs, on the same principle — see
        // `Backend::matmul_into`.
        let mut ffn_out: Vec<f32> = Vec::new();
        let mut ffn_scratch = super::FfnScratch::default();

        for (il, layer) in self.layers.iter().enumerate() {
            tensor::rmsnorm_into(&mut normed, &x, &layer.attn_norm, n_tokens, n_embd, eps);

            // Q, K, V and the gate are four independent projections of the
            // same normed input — one batched dispatch instead of four
            // sequential round trips (see `Backend::matmul_batch`).
            let mut qkvg = self.backend.matmul_batch(&[
                MatmulOp {
                    x: &normed,
                    n_tokens,
                    w: &layer.wq,
                },
                MatmulOp {
                    x: &normed,
                    n_tokens,
                    w: &layer.wk,
                },
                MatmulOp {
                    x: &normed,
                    n_tokens,
                    w: &layer.wv,
                },
                MatmulOp {
                    x: &normed,
                    n_tokens,
                    w: &layer.w_gate,
                },
            ]);
            let gate = qkvg.pop().unwrap();
            let v = qkvg.pop().unwrap();
            let mut k = qkvg.pop().unwrap();
            let mut q = qkvg.pop().unwrap();

            tensor::rmsnorm_inplace(&mut q, &layer.attn_q_norm, n_tokens * n_head, head_dim, eps);
            tensor::rmsnorm_inplace(
                &mut k,
                &layer.attn_k_norm,
                n_tokens * n_head_kv,
                head_dim,
                eps,
            );

            // RoPE — on the sliding-window layers only; a full-attention
            // layer rotates neither Q nor K, and its keys go into the cache
            // exactly as projected. Then append each token's K/V in prompt
            // order, since a later token's attention reads them.
            let layer_cache = &mut cache.layers[il];
            for t in 0..n_tokens {
                let pos = start_pos + t;
                if layer.is_swa {
                    tensor::rope_apply_params_inplace(
                        &mut q[t * n_head * head_dim..(t + 1) * n_head * head_dim],
                        n_head,
                        head_dim,
                        pos,
                        None,
                        &self.rope,
                    );
                    tensor::rope_apply_params_inplace(
                        &mut k[t * kv_dim..(t + 1) * kv_dim],
                        n_head_kv,
                        head_dim,
                        pos,
                        None,
                        &self.rope,
                    );
                }
                layer_cache.push(
                    &k[t * kv_dim..(t + 1) * kv_dim],
                    &v[t * kv_dim..(t + 1) * kv_dim],
                );
            }

            // `engine::attention` owns the GPU/CPU choice for every
            // architecture; the closure is the CPU window, and
            // `causal`/`n_swa` describe the same range to the GPU kernel.
            let is_swa = layer.is_swa;
            crate::engine::attention::attention(
                &mut attn_out,
                &q,
                layer_cache,
                &crate::engine::attention::Params {
                    backend: self.backend.as_ref(),
                    // This layer's card — see `attention::Params::device`.
                    device: layer.wo.device(),
                    n_head,
                    n_head_kv,
                    head_dim,
                    scale: 1.0 / (head_dim as f32).sqrt(),
                    causal: true,
                    n_swa: if is_swa { self.n_swa } else { 0 },
                    start_pos,
                    n_tokens,
                },
                |t| self.attention_window(is_swa, start_pos + t),
            );

            // The output gate, element-wise: attention's output and the gate
            // projection are both `[n_tokens, n_head * head_dim]`, so the
            // two line up position by position.
            debug_assert_eq!(attn_out.len(), gate.len());
            for (o, &g) in attn_out.iter_mut().zip(gate.iter()) {
                *o *= tensor::sigmoid(g);
            }

            let mut attn_proj = self.backend.matmul(&attn_out, n_tokens, &layer.wo);
            tensor::rmsnorm_inplace(
                &mut attn_proj,
                &layer.attn_post_norm,
                n_tokens,
                n_embd,
                POST_NORM_EPS,
            );
            tensor::add_inplace(&mut x, &attn_proj);

            tensor::rmsnorm_into(&mut ffn_normed, &x, &layer.ffn_norm, n_tokens, n_embd, eps);
            // Was an open-coded copy of `super::swiglu_ffn` — same batched
            // gate/up, same in-place SiLU, same multiply, same down.
            super::swiglu_ffn_into(
                self.backend.as_ref(),
                &mut ffn_out,
                &mut ffn_scratch,
                &ffn_normed,
                n_tokens,
                &layer.ffn_gate,
                &layer.ffn_up,
                &layer.ffn_down,
            );
            tensor::rmsnorm_inplace(
                &mut ffn_out,
                &layer.ffn_post_norm,
                n_tokens,
                n_embd,
                POST_NORM_EPS,
            );
            tensor::add_inplace(&mut x, &ffn_out);
        }

        Ok(x)
    }

    /// `logit_scale` then `final_logit_softcapping`, in that order, over one
    /// `[n_vocab]` row — upstream's own order, and the one that makes the
    /// cap mean what it says: a scale applied *after* it would push the
    /// logits straight back outside the bound. Both are no-ops when the
    /// file sets neither.
    fn apply_logit_transforms(&self, logits: &mut [f32]) {
        if let Some(scale) = self.logit_scale {
            for v in logits.iter_mut() {
                *v *= scale;
            }
        }
        if let Some(cap) = self.final_logit_softcapping {
            for v in logits.iter_mut() {
                *v = (*v / cap).tanh() * cap;
            }
        }
    }
}

impl ModelForward for MuseModel {
    fn vulkan_backend(&self) -> Option<&crate::engine::backend::vulkan::VulkanBackend> {
        self.backend.as_wgpu()
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn new_kv_cache(&self, capacity: usize) -> KvCache {
        // Every layer caches the same shape — a sliding-window layer still
        // *stores* every position, it just attends fewer of them.
        let kv_dim = self.config.n_head_kv * self.head_dim();
        KvCache::new(self.config.n_layer, capacity, kv_dim)
    }

    fn forward(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        _slot_id: usize,
    ) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let n_tokens = tokens.len();
        let n_embd = cfg.n_embd;

        let x = self.run_layers(cache, tokens, start_pos)?;
        // Only the last token's hidden state is needed for next-token
        // logits — a batched prefill doesn't need every position's output.
        let last = &mut x[(n_tokens - 1) * n_embd..].to_vec();
        tensor::rmsnorm_inplace(last, &self.output_norm, 1, n_embd, cfg.rms_eps);
        let mut logits = self.backend.matmul(last, 1, &self.output_weight);
        self.apply_logit_transforms(&mut logits);
        Ok(logits)
    }

    fn forward_all_logits(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        _slot_id: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let cfg = &self.config;
        let n_tokens = tokens.len();
        let n_embd = cfg.n_embd;
        let n_vocab = cfg.n_vocab;

        // One projection of every position through the output norm + vocab
        // matrix, batched — the weight-heavy `lm_head` read is amortized
        // across the whole draft in a single `matmul`, not one per position.
        let mut h = self.run_layers(cache, tokens, start_pos)?;
        tensor::rmsnorm_inplace(&mut h, &self.output_norm, n_tokens, n_embd, cfg.rms_eps);
        let flat = self.backend.matmul(&h, n_tokens, &self.output_weight);
        anyhow::ensure!(
            flat.len() == n_tokens * n_vocab,
            "output projection produced {} logits, expected {}",
            flat.len(),
            n_tokens * n_vocab
        );
        Ok((0..n_tokens)
            .map(|t| {
                let mut row = flat[t * n_vocab..(t + 1) * n_vocab].to_vec();
                self.apply_logit_transforms(&mut row);
                row
            })
            .collect())
    }

    fn forward_hidden_states(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        // A one-shot, whole-prompt pass — no KV cache reuse across calls,
        // the same convention `LlamaModel::forward_hidden_states` uses.
        let mut cache = self.new_kv_cache(tokens.len().max(1));
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
    /// The sliding-window pattern is read from a *scalar* period, and its
    /// answer is load-bearing twice over: it selects the attention window
    /// **and** whether the layer rotates at all. Upstream's formula makes
    /// the **last** layer of each group of `n_pattern` the full-attention
    /// one; the off-by-one alternative (`il % n_pattern != 0`, upstream's
    /// `dense_first` form, which several other architectures do use) picks
    /// a different set of layers entirely and is not distinguishable from
    /// this one by layer count.
    #[test]
    fn every_fourth_layer_is_full_attention_and_it_is_the_last_of_its_group() {
        let is_swa = |il: usize, n_pattern: usize| il % n_pattern < n_pattern - 1;
        let pattern: Vec<bool> = (0..8).map(|il| is_swa(il, 4)).collect();
        assert_eq!(
            pattern,
            vec![true, true, true, false, true, true, true, false]
        );
        assert_eq!((0..52).filter(|&il| is_swa(il, 4)).count(), 39);
    }
}

#[cfg(test)]
mod real_model_tests {
    use super::*;

    /// End-to-end against a real `unsloth/Muse-Glimmer-30B-GGUF` file.
    ///
    /// The assertion is on the *predicted token*, not on logits within a
    /// tolerance, for the reason `arch::llama`'s equivalent test spells
    /// out: a wrong-but-plausible forward pass produces fluent output a
    /// tolerance check can be talked into accepting, while a factual
    /// one-word answer cannot survive a broken attention or FFN. This
    /// architecture has an unusual number of ways to be wrong-but-plausible
    /// and every one of them was observed while it was being written — the
    /// wrong FFN activation, the wrong rotary pairing, rotating the
    /// full-attention layers, and skipping the weightless embedding norm
    /// each produce a model that loads, runs, and answers "The capital of
    /// France is" with a newline or a code token instead of " Paris".
    ///
    /// Run with `ORANGU_TEST_MUSE_MODEL=/path/to/Muse-Glimmer-30B-UD-Q4_K_XL.gguf
    /// cargo test --release --bin orangu-server muse::real_model_tests --
    /// --ignored`.
    #[test]
    #[ignore]
    fn muse_glimmer_predicts_paris_after_capital_of_france() {
        let path = std::env::var("ORANGU_TEST_MUSE_MODEL").expect("set ORANGU_TEST_MUSE_MODEL");
        let gguf = orangu::gguf::GgufFile::open(std::path::Path::new(&path)).expect("open gguf");
        let tokenizer =
            crate::engine::tokenizer::Tokenizer::from_gguf(&gguf).expect("build tokenizer");
        let loaded = LoadedModel::open(std::path::Path::new(&path)).expect("load model");
        assert_eq!(loaded.config.architecture, "muse-glimmer");
        let model =
            MuseModel::load_with_backend(&loaded, Arc::new(crate::engine::backend::CpuBackend))
                .expect("build model");
        // Three layers in four windowed, and those are exactly the ones
        // that rotate — asserted here as well as in the unit test above, so
        // a checkpoint whose pattern key differs is reported as such rather
        // than as a bad prediction.
        assert_eq!(
            model.layers.iter().filter(|l| l.is_swa).count(),
            model.layers.len() * 3 / 4,
        );

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
}
