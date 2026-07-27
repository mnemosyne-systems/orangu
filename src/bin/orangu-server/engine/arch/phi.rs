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

//! The `phi3` forward pass (Microsoft Phi-3 / Phi-4-mini) — a GQA + RoPE +
//! RMSNorm + SwiGLU transformer like `arch::llama`, but with four
//! differences that each silently corrupt output if guessed rather than read
//! off upstream, all four confirmed directly against `llama.cpp/src/models/
//! phi3.cpp` and the ggml kernels it calls:
//!
//! 1. **Fused QKV.** One `attn_qkv.weight` of `[n_embd, n_embd + 2*kv_dim]`
//!    rather than three separate projections, sliced Q-then-K-then-V
//!    (`llm_graph_context::build_qkv`'s own view offsets: `0`, `n_embd_q`,
//!    `n_embd_q + n_embd_kv`). Upstream still accepts a checkpoint with
//!    separate `attn_q`/`attn_k`/`attn_v` instead (`create_tensor_qkv` falls
//!    back when `wqkv` is absent), so this module does too.
//! 2. **Fused gate/up.** `ffn_up.weight` is `[n_embd, 2*n_ff]` and there is
//!    no `ffn_gate.weight` at all; `LLM_FFN_SWIGLU` splits the *result* in
//!    half. The activated half is the **first** one — ggml's
//!    `ggml_compute_forward_swiglu` with `swapped = 0` sets `src0_p` to the
//!    row start and `src1_p` to `+nc`, and `ggml_vec_swiglu_f32` computes
//!    `silu(x) * g` over those in that order. Swapping the halves produces
//!    fluent-looking but wrong text, not a crash.
//! 3. **Partial NEOX RoPE with LongRoPE frequency factors.** Only the
//!    leading `rope_dim` of each head rotates (96 of 128 for Phi-4-mini) and
//!    each pair's frequency is divided by an entry of `rope_factors_long` or
//!    `rope_factors_short` — see [`PhiModel::rope_freq_factors`] for which,
//!    and why.
//! 4. **A RoPE magnitude factor.** `phi3.rope.scaling.attn_factor`
//!    (1.1902381 for Phi-4-mini) scales cos/sin, i.e. lengthens every
//!    rotated pair — see `tensor::rope_apply_mscale_inplace`.
//!
//! Sliding-window attention is deliberately **not** implemented, matching
//! upstream: Phi-4-mini's GGUF does carry `phi3.attention.sliding_window =
//! 262144`, but `llama_model_phi3::load_arch_hparams` reacts to finding that
//! key by *disabling* SWA outright (`swa_type = LLAMA_SWA_TYPE_NONE; n_swa =
//! 0`), warning that the conversion scripts populate it incorrectly
//! (ggml-org/llama.cpp#13676). Attention here is plain causal, as upstream's
//! is.
//!
//! Weight matrices stay `mmap`-backed and are dequantized one row at a time
//! via `QuantMatrix`, exactly as in `arch::llama`; only the small per-element
//! norm tensors are eagerly dequantized.

use anyhow::{Context, Result};
use std::sync::Arc;

use super::ModelForward;
use crate::engine::backend::{Backend, MatmulOp};
use crate::engine::kv_cache::KvCache;
use crate::engine::loader::{LoadedModel, ModelConfig, QuantMatrix};
use crate::engine::tensor;

/// One layer's attention projections: either the fused `attn_qkv.weight`
/// upstream prefers, or the three separate ones it falls back to.
enum QkvProjection {
    /// `attn_qkv.weight`, `[n_embd, n_embd + 2*kv_dim]`.
    Fused(QuantMatrix),
    /// `attn_q.weight` / `attn_k.weight` / `attn_v.weight`.
    Split {
        wq: QuantMatrix,
        wk: QuantMatrix,
        wv: QuantMatrix,
    },
}

struct PhiLayer {
    attn_norm: Vec<f32>,
    qkv: QkvProjection,
    wo: QuantMatrix,
    ffn_norm: Vec<f32>,
    /// `ffn_up.weight`, `[n_embd, 2 * n_ff]` — gate and up concatenated;
    /// there is no separate `ffn_gate.weight` for this architecture.
    w_up: QuantMatrix,
    w_down: QuantMatrix,
}

pub struct PhiModel {
    config: ModelConfig,
    backend: Arc<dyn Backend>,
    tok_embeddings: QuantMatrix,
    output_norm: Vec<f32>,
    output_weight: QuantMatrix,
    layers: Vec<PhiLayer>,
    /// The chosen LongRoPE `[rope_dim / 2]` per-pair frequency divisor, or
    /// `None` for a `phi3` checkpoint that ships neither factor tensor.
    rope_freq_factors: Option<Vec<f32>>,
    /// `phi3.rope.scaling.attn_factor`, defaulting to `1.0` (a no-op) when
    /// the file doesn't set it — upstream's own `hparams.rope_attn_factor`
    /// default.
    rope_attn_factor: f32,
}

impl PhiModel {
    pub fn load_with_backend(loaded: &LoadedModel, backend: Arc<dyn Backend>) -> Result<Self> {
        let config = loaded.config.clone();
        let tok_embeddings = loaded
            .matrix("token_embd.weight")
            .context("loading token_embd.weight")?;
        let (output_norm, _) = loaded
            .tensor("output_norm.weight")
            .context("loading output_norm.weight")?;
        // Phi-4-mini ties its output projection to the input embedding and
        // ships no `output.weight` at all — upstream's own
        // `TENSOR_NOT_REQUIRED` + `TENSOR_DUPLICATED` fallback.
        let output_weight = if loaded.has_tensor("output.weight") {
            loaded
                .matrix("output.weight")
                .context("loading output.weight")?
        } else {
            tok_embeddings.clone()
        };

        let rope_freq_factors = Self::rope_freq_factors(loaded)?;
        if let Some(factors) = &rope_freq_factors {
            anyhow::ensure!(
                factors.len() >= config.rope_dim / 2,
                "rope factor tensor has {} entries, need {} for rope.dimension_count = {}",
                factors.len(),
                config.rope_dim / 2,
                config.rope_dim,
            );
        }
        let rope_attn_factor = loaded
            .metadata_f32("rope.scaling.attn_factor")
            .unwrap_or(1.0);

        let mut layers = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
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
            // Fused when present, three separate projections otherwise —
            // upstream's `create_tensor_qkv` makes the same choice on the
            // same condition.
            let qkv = if loaded.has_tensor(&format!("blk.{i}.attn_qkv.weight")) {
                QkvProjection::Fused(get_matrix("attn_qkv.weight")?)
            } else {
                QkvProjection::Split {
                    wq: get_matrix("attn_q.weight")?,
                    wk: get_matrix("attn_k.weight")?,
                    wv: get_matrix("attn_v.weight")?,
                }
            };
            layers.push(PhiLayer {
                attn_norm: get("attn_norm.weight")?,
                qkv,
                wo: get_matrix("attn_output.weight")?,
                ffn_norm: get("ffn_norm.weight")?,
                w_up: get_matrix("ffn_up.weight")?,
                w_down: get_matrix("ffn_down.weight")?,
            });
        }

        let model = Self {
            config,
            backend,
            tok_embeddings,
            output_norm,
            output_weight,
            layers,
            rope_freq_factors,
            rope_attn_factor,
        };
        model.validate_shapes()?;
        Ok(model)
    }

    /// Which of `rope_factors_long` / `rope_factors_short` this model rotates
    /// with — LongRoPE ships both and picks one.
    ///
    /// Upstream's rule (`llama_model::get_rope_factors`) is a property of the
    /// *serving* context, not of the request: `n_ctx_seq >
    /// hparams.n_ctx_orig_yarn ? long : short`, decided once at context
    /// creation so that every key already in the KV cache was rotated the
    /// same way as the query now reading it. This engine has no separate
    /// context-length knob — `engine::generate` caps a sequence at the
    /// model's own `n_ctx_train` — so `n_ctx_train` *is* this build's
    /// `n_ctx_seq`, and the comparison is made against it. For Phi-4-mini
    /// (131072 trained, 4096 original) that selects the long factors, which
    /// is also what `llama-server -c 0` (or any `-c` above 4096) selects;
    /// note upstream's own CLI default of `-c 4096` would instead select the
    /// short ones, so a logit-level comparison against llama.cpp has to pass
    /// a matching `-c`.
    ///
    /// `n_ctx_orig_yarn` itself defaults to `n_ctx_train` when the file omits
    /// `rope.scaling.original_context_length` (upstream sets it that way
    /// before the optional `get_key`), which makes the comparison false and
    /// selects the short factors — the right answer for a checkpoint that
    /// never declared a shorter original context.
    fn rope_freq_factors(loaded: &LoadedModel) -> Result<Option<Vec<f32>>> {
        let n_ctx_orig = loaded
            .metadata_u64("rope.scaling.original_context_length")
            .map(|v| v as usize)
            .unwrap_or(loaded.config.n_ctx_train);
        let name = if loaded.config.n_ctx_train > n_ctx_orig {
            "rope_factors_long.weight"
        } else {
            "rope_factors_short.weight"
        };
        if !loaded.has_tensor(name) {
            return Ok(None);
        }
        Ok(Some(
            loaded
                .tensor(name)
                .with_context(|| format!("loading {name}"))?
                .0,
        ))
    }

    fn head_dim(&self) -> usize {
        self.config.n_embd / self.config.n_head
    }

    fn kv_dim(&self) -> usize {
        self.config.n_head_kv * self.head_dim()
    }

    /// Rejects, at load time, a checkpoint whose tensor shapes don't match
    /// the hyperparameters — the fused QKV and fused gate/up splits are both
    /// computed from `n_embd`/`n_head_kv`/`out_dim` rather than read, so a
    /// mismatch would otherwise slice at the wrong offsets and produce
    /// confidently wrong tokens instead of an error.
    fn validate_shapes(&self) -> Result<()> {
        let n_embd = self.config.n_embd;
        let qkv_dim = n_embd + 2 * self.kv_dim();
        for (i, layer) in self.layers.iter().enumerate() {
            match &layer.qkv {
                QkvProjection::Fused(wqkv) => anyhow::ensure!(
                    wqkv.in_dim == n_embd && wqkv.out_dim == qkv_dim,
                    "blk.{i}.attn_qkv.weight is [{}, {}], expected [{n_embd}, {qkv_dim}]",
                    wqkv.in_dim,
                    wqkv.out_dim,
                ),
                QkvProjection::Split { wq, wk, wv } => {
                    anyhow::ensure!(
                        wq.out_dim == n_embd
                            && wk.out_dim == self.kv_dim()
                            && wv.out_dim == self.kv_dim(),
                        "blk.{i} attn_q/attn_k/attn_v produce [{}, {}, {}], expected [{n_embd}, {}, {}]",
                        wq.out_dim,
                        wk.out_dim,
                        wv.out_dim,
                        self.kv_dim(),
                        self.kv_dim(),
                    );
                }
            }
            anyhow::ensure!(
                layer.w_up.out_dim % 2 == 0,
                "blk.{i}.ffn_up.weight has an odd output width ({}) — this architecture's \
                 SwiGLU splits it into equal gate and up halves",
                layer.w_up.out_dim,
            );
            anyhow::ensure!(
                layer.w_up.out_dim / 2 == layer.w_down.in_dim,
                "blk.{i}.ffn_up.weight's half-width {} doesn't match ffn_down.weight's input {}",
                layer.w_up.out_dim / 2,
                layer.w_down.in_dim,
            );
        }
        Ok(())
    }

    /// Runs every transformer layer and returns the pre-final-norm hidden
    /// state for every token (`[n_tokens, n_embd]`) — shared by next-token
    /// prediction and pooled embeddings, as in `arch::llama`.
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
        let kv_dim = self.kv_dim();
        let group_size = n_head / n_head_kv;

        let mut x = vec![0f32; n_tokens * n_embd];
        for (t, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            anyhow::ensure!(tok < cfg.n_vocab, "token id {tok} is out of vocab range");
            x[t * n_embd..(t + 1) * n_embd].copy_from_slice(&self.tok_embeddings.row(tok));
        }

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let mut normed = x.clone();
            tensor::rmsnorm_inplace(&mut normed, &layer.attn_norm, n_tokens, n_embd, cfg.rms_eps);

            let (mut q, mut k, v) = match &layer.qkv {
                QkvProjection::Fused(wqkv) => {
                    let qkv = self.backend.matmul(&normed, n_tokens, wqkv);
                    split_qkv(&qkv, n_tokens, n_embd, kv_dim)
                }
                QkvProjection::Split { wq, wk, wv } => {
                    // Independent given the same normed input — one batched
                    // dispatch rather than three round-trips, matching
                    // `arch::llama`.
                    let mut out = self.backend.matmul_batch(&[
                        MatmulOp {
                            x: &normed,
                            n_tokens,
                            w: wq,
                        },
                        MatmulOp {
                            x: &normed,
                            n_tokens,
                            w: wk,
                        },
                        MatmulOp {
                            x: &normed,
                            n_tokens,
                            w: wv,
                        },
                    ]);
                    let v = out.pop().unwrap();
                    let k = out.pop().unwrap();
                    let q = out.pop().unwrap();
                    (q, k, v)
                }
            };

            let layer_cache = &mut cache.layers[layer_idx];
            for t in 0..n_tokens {
                let pos = start_pos + t;
                tensor::rope_apply_mscale_inplace(
                    &mut q[t * n_head * head_dim..(t + 1) * n_head * head_dim],
                    n_head,
                    head_dim,
                    cfg.rope_dim,
                    pos,
                    cfg.rope_freq_base,
                    self.rope_freq_factors.as_deref(),
                    self.rope_attn_factor,
                );
                tensor::rope_apply_mscale_inplace(
                    &mut k[t * kv_dim..(t + 1) * kv_dim],
                    n_head_kv,
                    head_dim,
                    cfg.rope_dim,
                    pos,
                    cfg.rope_freq_base,
                    self.rope_freq_factors.as_deref(),
                    self.rope_attn_factor,
                );
                layer_cache.push(
                    &k[t * kv_dim..(t + 1) * kv_dim],
                    &v[t * kv_dim..(t + 1) * kv_dim],
                );
            }

            // Plain causal attention — no sliding window, matching upstream's
            // own disabling of it for this architecture (see the module doc).
            let mut attn_out = vec![0f32; n_tokens * n_head * head_dim];
            let scale = 1.0 / (head_dim as f32).sqrt();
            crate::engine::attention::multi_head_attention(
                &mut attn_out,
                &q,
                layer_cache,
                n_head,
                group_size,
                head_dim,
                scale,
                |t| (0, start_pos + t),
            );

            let attn_proj = self.backend.matmul(&attn_out, n_tokens, &layer.wo);
            tensor::add_inplace(&mut x, &attn_proj);

            let mut normed2 = x.clone();
            tensor::rmsnorm_inplace(&mut normed2, &layer.ffn_norm, n_tokens, n_embd, cfg.rms_eps);
            // One `[n_embd, 2*n_ff]` projection, then SwiGLU over the two
            // halves of each row: `silu(first) * second`.
            let gate_up = self.backend.matmul(&normed2, n_tokens, &layer.w_up);
            let n_ff = layer.w_up.out_dim / 2;
            let mut activated = vec![0f32; n_tokens * n_ff];
            for t in 0..n_tokens {
                let row = &gate_up[t * 2 * n_ff..(t + 1) * 2 * n_ff];
                let (gate, up) = row.split_at(n_ff);
                let out = &mut activated[t * n_ff..(t + 1) * n_ff];
                for ((o, g), u) in out.iter_mut().zip(gate).zip(up) {
                    *o = tensor::silu(*g) * u;
                }
            }
            let down = self.backend.matmul(&activated, n_tokens, &layer.w_down);
            tensor::add_inplace(&mut x, &down);
        }

        Ok(x)
    }
}

/// Splits a fused `[n_tokens, n_embd + 2*kv_dim]` QKV projection into
/// separate, per-token-contiguous `q`/`k`/`v` buffers.
///
/// The offsets are upstream's (`build_qkv`'s three `ggml_view_3d` calls): Q
/// first at `0`, then K at `n_embd`, then V at `n_embd + kv_dim`. The
/// resulting layout is what the rest of this module — RoPE, the KV cache
/// push, `multi_head_attention` — already expects from a split projection,
/// so nothing downstream needs to know which path produced it.
fn split_qkv(
    qkv: &[f32],
    n_tokens: usize,
    n_embd: usize,
    kv_dim: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let stride = n_embd + 2 * kv_dim;
    debug_assert_eq!(qkv.len(), n_tokens * stride);
    let mut q = Vec::with_capacity(n_tokens * n_embd);
    let mut k = Vec::with_capacity(n_tokens * kv_dim);
    let mut v = Vec::with_capacity(n_tokens * kv_dim);
    for row in qkv.chunks_exact(stride) {
        q.extend_from_slice(&row[..n_embd]);
        k.extend_from_slice(&row[n_embd..n_embd + kv_dim]);
        v.extend_from_slice(&row[n_embd + kv_dim..]);
    }
    (q, k, v)
}

impl ModelForward for PhiModel {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn new_kv_cache(&self, capacity: usize) -> KvCache {
        KvCache::new(self.config.n_layer, capacity, self.kv_dim())
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

        let last = &mut x[(n_tokens - 1) * n_embd..].to_vec();
        tensor::rmsnorm_inplace(last, &self.output_norm, 1, n_embd, cfg.rms_eps);
        let logits = self.backend.matmul(last, 1, &self.output_weight);
        Ok(logits)
    }

    fn forward_hidden_states(&self, tokens: &[u32]) -> Result<Vec<f32>> {
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
    use super::*;

    /// The fused-QKV slice offsets, against a hand-built projection whose
    /// every element names its own (token, section, index) — so a swapped
    /// K/V, an off-by-`kv_dim` offset, or a row-vs-column transposition each
    /// produce a visibly wrong value rather than a plausible one.
    #[test]
    fn split_qkv_slices_q_then_k_then_v() {
        let (n_tokens, n_embd, kv_dim) = (2, 4, 2);
        let stride = n_embd + 2 * kv_dim;
        // Token t's row: q = 100*t + 0..4, k = 100*t + 10..12, v = 100*t + 20..22.
        let mut qkv = Vec::new();
        for t in 0..n_tokens {
            let base = 100.0 * t as f32;
            qkv.extend((0..n_embd).map(|i| base + i as f32));
            qkv.extend((0..kv_dim).map(|i| base + 10.0 + i as f32));
            qkv.extend((0..kv_dim).map(|i| base + 20.0 + i as f32));
        }
        assert_eq!(qkv.len(), n_tokens * stride);

        let (q, k, v) = split_qkv(&qkv, n_tokens, n_embd, kv_dim);
        assert_eq!(q, vec![0.0, 1.0, 2.0, 3.0, 100.0, 101.0, 102.0, 103.0]);
        assert_eq!(k, vec![10.0, 11.0, 110.0, 111.0]);
        assert_eq!(v, vec![20.0, 21.0, 120.0, 121.0]);
    }
}

#[cfg(test)]
mod real_model_tests {
    use super::*;

    /// End-to-end greedy decode against a real `unsloth/Phi-4-mini-instruct-
    /// GGUF` file, checking this module's four architecture-specific
    /// decisions actually hold together on real weights rather than only in
    /// isolation: the fused-QKV split, the SwiGLU half order, the LongRoPE
    /// factor choice, and the `attn_factor` magnitude scale.
    ///
    /// The prompt is the model's own chat format (`<|user|>...<|end|>
    /// <|assistant|>`), and the assertion is on *text*, not logits, because
    /// that is what actually breaks: getting the SwiGLU halves backwards or
    /// the QKV offsets wrong yields fluent-but-wrong output that a
    /// tolerance-based logit check can be talked into accepting.
    ///
    /// Run with `ORANGU_TEST_PHI_MODEL=/path/to/Phi-4-mini-instruct-Q4_K_M.gguf
    /// cargo test --release --bin orangu-server real_model_tests -- --ignored`.
    #[test]
    #[ignore]
    fn phi4_mini_answers_a_factual_question() {
        let path = std::env::var("ORANGU_TEST_PHI_MODEL").expect("set ORANGU_TEST_PHI_MODEL");
        let loaded = LoadedModel::open(std::path::Path::new(&path)).expect("load model");
        assert_eq!(loaded.config.architecture, "phi3");
        let gguf = orangu::gguf::GgufFile::open(std::path::Path::new(&path)).expect("open gguf");
        let tokenizer =
            crate::engine::tokenizer::Tokenizer::from_gguf(&gguf).expect("build tokenizer");
        let model =
            PhiModel::load_with_backend(&loaded, Arc::new(crate::engine::backend::CpuBackend))
                .expect("build model");

        let prompt = "<|user|>What is the capital of France? Answer in one word.<|end|>\
                      <|assistant|>";
        // `false`, as every chat-templated path in this server does: the
        // template already carries whatever prefix the model wants, and
        // Phi-4-mini's own `tokenizer.ggml.add_bos_token` is false.
        let tokens = tokenizer.encode(prompt, false);
        let mut cache = model.new_kv_cache(tokens.len() + 16);
        let mut logits = model.forward(&mut cache, &tokens, 0, 0).expect("prefill");

        let mut generated = Vec::new();
        for step in 0..16 {
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).expect("logits are finite"))
                .expect("non-empty logits")
                .0 as u32;
            if tokenizer.stop_token_ids().contains(&next) {
                break;
            }
            generated.push(next);
            logits = model
                .forward(&mut cache, &[next], tokens.len() + step, 0)
                .expect("decode");
        }

        let text = tokenizer.decode(&generated);
        assert!(
            text.contains("Paris"),
            "expected the answer to name Paris, got {text:?}"
        );
    }
}
