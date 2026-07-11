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

//! The Llama-style forward pass: grouped-query attention, RoPE, RMSNorm,
//! SwiGLU — the shape shared by Llama/Llama3/Qwen2/Qwen3/Mistral GGUFs
//! (tensor names confirmed against `llama.cpp/src/llama-arch.cpp`'s
//! `LLM_TENSOR_NAMES` table for `LLM_ARCH_LLAMA`).
//!
//! Weight matrices and embedding tables stay `mmap`-backed and are
//! dequantized one row at a time, on demand, via `QuantMatrix` — not
//! eagerly materialized to `f32` at load time. Only small per-element
//! tensors (norms, biases) are eagerly dequantized. This keeps resident
//! memory close to the file's own size rather than the ~4x an eager,
//! fully-dequantized-to-`f32` approach costs — the difference between a
//! large (tens-of-billions-of-parameters) model fitting in RAM at all or
//! not, see `doc/SERVER_ROADMAP.md`.

use anyhow::{Context, Result};
use std::sync::Arc;

use super::ModelForward;
use crate::engine::backend::{Backend, MatmulOp};
use crate::engine::kv_cache::KvCache;
use crate::engine::loader::{LoadedModel, ModelConfig, QuantMatrix};
use crate::engine::tensor;

struct LlamaLayer {
    attn_norm: Vec<f32>,
    wq: QuantMatrix,
    wk: QuantMatrix,
    wv: QuantMatrix,
    wo: QuantMatrix,
    /// Q/K/V projection biases — present on Qwen2/Qwen3-shaped GGUFs,
    /// absent on plain Llama/Mistral ones (`attn_*.bias` tensors simply
    /// don't exist in the file for those; confirmed directly against a
    /// downloaded Qwen2.5 GGUF, which has all three).
    q_bias: Option<Vec<f32>>,
    k_bias: Option<Vec<f32>>,
    v_bias: Option<Vec<f32>>,
    /// Per-head RMSNorm on Q/K after projection, before RoPE — present on
    /// Qwen3/Qwen3VL-shaped GGUFs (`attn_q_norm.weight`/`attn_k_norm.
    /// weight`, each `[head_dim]`), absent on Qwen2/Llama/Mistral ones
    /// (confirmed directly against a real downloaded `Qwen3-VL-Embedding-
    /// 8B` GGUF's `src/models/qwen3vl.cpp` graph: `Qcur = build_norm(Qcur,
    /// attn_q_norm, ..., LLM_NORM_RMS, il)` runs immediately after `build_
    /// qkv`, before `ggml_rope_multi`).
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
    ffn_norm: Vec<f32>,
    w_gate: QuantMatrix,
    w_up: QuantMatrix,
    w_down: QuantMatrix,
}

pub struct LlamaModel {
    config: ModelConfig,
    backend: Arc<dyn Backend>,
    tok_embeddings: QuantMatrix,
    output_norm: Vec<f32>,
    output_weight: QuantMatrix,
    layers: Vec<LlamaLayer>,
}

impl LlamaModel {
    pub fn load_with_backend(loaded: &LoadedModel, backend: Arc<dyn Backend>) -> Result<Self> {
        let config = loaded.config.clone();
        let tok_embeddings = loaded
            .matrix("token_embd.weight")
            .context("loading token_embd.weight")?;
        let (output_norm, _) = loaded
            .tensor("output_norm.weight")
            .context("loading output_norm.weight")?;
        // Some models tie the output projection to the input embedding and
        // simply omit a separate "output.weight" tensor.
        let output_weight = if loaded.has_tensor("output.weight") {
            loaded
                .matrix("output.weight")
                .context("loading output.weight")?
        } else {
            tok_embeddings.clone()
        };

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
            let get_optional = |suffix: &str| -> Result<Option<Vec<f32>>> {
                let name = format!("blk.{i}.{suffix}");
                if !loaded.has_tensor(&name) {
                    return Ok(None);
                }
                Ok(Some(
                    loaded
                        .tensor(&name)
                        .with_context(|| format!("loading {name}"))?
                        .0,
                ))
            };
            layers.push(LlamaLayer {
                attn_norm: get("attn_norm.weight")?,
                wq: get_matrix("attn_q.weight")?,
                wk: get_matrix("attn_k.weight")?,
                wv: get_matrix("attn_v.weight")?,
                wo: get_matrix("attn_output.weight")?,
                q_bias: get_optional("attn_q.bias")?,
                k_bias: get_optional("attn_k.bias")?,
                v_bias: get_optional("attn_v.bias")?,
                q_norm: get_optional("attn_q_norm.weight")?,
                k_norm: get_optional("attn_k_norm.weight")?,
                ffn_norm: get("ffn_norm.weight")?,
                w_gate: get_matrix("ffn_gate.weight")?,
                w_up: get_matrix("ffn_up.weight")?,
                w_down: get_matrix("ffn_down.weight")?,
            });
        }

        Ok(Self {
            config,
            backend,
            tok_embeddings,
            output_norm,
            output_weight,
            layers,
        })
    }

    fn head_dim(&self) -> usize {
        self.config.n_embd / self.config.n_head
    }
}

impl LlamaModel {
    /// Runs every transformer layer and returns the pre-final-norm hidden
    /// state for every token (`[n_tokens, n_embd]`) — the shared core of
    /// both next-token prediction ([`ModelForward::forward`]) and pooled
    /// embeddings ([`LlamaModel::forward_hidden_states`]).
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
        let group_size = n_head / n_head_kv;

        // Embedding lookup: x[t, :] = tok_embeddings[token[t], :].
        let mut x = vec![0f32; n_tokens * n_embd];
        for (t, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            anyhow::ensure!(tok < cfg.n_vocab, "token id {tok} is out of vocab range");
            x[t * n_embd..(t + 1) * n_embd].copy_from_slice(&self.tok_embeddings.row(tok));
        }

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let mut normed = x.clone();
            tensor::rmsnorm_inplace(&mut normed, &layer.attn_norm, n_tokens, n_embd, cfg.rms_eps);

            // Independent given the same normed input — one batched
            // dispatch instead of three sequential round-trips (matters
            // most for a GPU backend; see `Backend::matmul_batch`).
            let mut qkv = self.backend.matmul_batch(&[
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
            ]);
            let mut v = qkv.pop().unwrap();
            let mut k = qkv.pop().unwrap();
            let mut q = qkv.pop().unwrap();
            if let Some(bias) = &layer.q_bias {
                tensor::add_bias_per_row(&mut q, bias, n_tokens);
            }
            if let Some(bias) = &layer.k_bias {
                tensor::add_bias_per_row(&mut k, bias, n_tokens);
            }
            if let Some(bias) = &layer.v_bias {
                tensor::add_bias_per_row(&mut v, bias, n_tokens);
            }
            // Per-head RMSNorm, before RoPE — `Qwen3-VL-Embedding-8B`'s own
            // `src/models/qwen3vl.cpp` graph runs this immediately after
            // `build_qkv`, before `ggml_rope_multi`; `None` (Qwen2/Llama/
            // Mistral) is a no-op.
            if let Some(q_norm) = &layer.q_norm {
                tensor::rmsnorm_inplace(&mut q, q_norm, n_tokens * n_head, head_dim, cfg.rms_eps);
            }
            if let Some(k_norm) = &layer.k_norm {
                tensor::rmsnorm_inplace(
                    &mut k,
                    k_norm,
                    n_tokens * n_head_kv,
                    head_dim,
                    cfg.rms_eps,
                );
            }

            // RoPE, then append this token's K/V to the sequence's cache —
            // one token (one row) at a time, in prompt order, since a later
            // token's cache entry must exist before an even-later token's
            // attention can see it.
            let layer_cache = &mut cache.layers[layer_idx];
            for t in 0..n_tokens {
                let pos = start_pos + t;
                tensor::rope_apply_inplace(
                    &mut q[t * n_head * head_dim..(t + 1) * n_head * head_dim],
                    n_head,
                    head_dim,
                    cfg.rope_dim,
                    pos,
                    cfg.rope_freq_base,
                );
                tensor::rope_apply_inplace(
                    &mut k[t * kv_dim..(t + 1) * kv_dim],
                    n_head_kv,
                    head_dim,
                    cfg.rope_dim,
                    pos,
                    cfg.rope_freq_base,
                );
                layer_cache.push(
                    &k[t * kv_dim..(t + 1) * kv_dim],
                    &v[t * kv_dim..(t + 1) * kv_dim],
                );
            }

            // Causal attention: token t (now at absolute position
            // start_pos+t) attends to every cached position up to and
            // including its own.
            let mut attn_out = vec![0f32; n_tokens * n_head * head_dim];
            for t in 0..n_tokens {
                let pos = start_pos + t;
                for h in 0..n_head {
                    let kv_head = h / group_size;
                    let qh = &q[t * n_head * head_dim + h * head_dim
                        ..t * n_head * head_dim + (h + 1) * head_dim];

                    let mut scores = Vec::with_capacity(pos + 1);
                    let scale = 1.0 / (head_dim as f32).sqrt();
                    for p in 0..=pos {
                        let kh = layer_cache.key_at(p, kv_head, head_dim);
                        scores.push(tensor::dot(qh, kh) * scale);
                    }
                    tensor::softmax_inplace(&mut scores);

                    let out = &mut attn_out[t * n_head * head_dim + h * head_dim
                        ..t * n_head * head_dim + (h + 1) * head_dim];
                    for (p, &weight) in scores.iter().enumerate() {
                        let vh = layer_cache.value_at(p, kv_head, head_dim);
                        for (o, vi) in out.iter_mut().zip(vh.iter()) {
                            *o += weight * vi;
                        }
                    }
                }
            }

            let attn_proj = self.backend.matmul(&attn_out, n_tokens, &layer.wo);
            tensor::add_inplace(&mut x, &attn_proj);

            let mut normed2 = x.clone();
            tensor::rmsnorm_inplace(&mut normed2, &layer.ffn_norm, n_tokens, n_embd, cfg.rms_eps);
            let mut gate_up = self.backend.matmul_batch(&[
                MatmulOp {
                    x: &normed2,
                    n_tokens,
                    w: &layer.w_gate,
                },
                MatmulOp {
                    x: &normed2,
                    n_tokens,
                    w: &layer.w_up,
                },
            ]);
            let up = gate_up.pop().unwrap();
            let mut gate = gate_up.pop().unwrap();
            for g in gate.iter_mut() {
                *g = tensor::silu(*g);
            }
            tensor::mul_inplace(&mut gate, &up);
            let down = self.backend.matmul(&gate, n_tokens, &layer.w_down);
            tensor::add_inplace(&mut x, &down);
        }

        Ok(x)
    }
}

impl ModelForward for LlamaModel {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn new_kv_cache(&self, capacity: usize) -> KvCache {
        let kv_dim = self.config.n_head_kv * self.head_dim();
        KvCache::new(self.config.n_layer, capacity, kv_dim)
    }

    fn forward(&self, cache: &mut KvCache, tokens: &[u32], start_pos: usize) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let n_tokens = tokens.len();
        let n_embd = cfg.n_embd;
        let x = self.run_layers(cache, tokens, start_pos)?;

        // Only the last token's hidden state is needed for next-token
        // logits — a batched prefill doesn't need every position's output.
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
mod real_model_tests {
    use super::*;
    use crate::engine::loader::PoolingType;

    /// Cross-check against real llama.cpp (`mradermacher/Qwen3-VL-
    /// Embedding-8B-GGUF:Q4_K_M`, `llama-server --embedding --pooling
    /// last`): tokenizing "The quick brown fox jumps over the lazy dog"
    /// with `add_special=true` gives `[785, 3974, 13876, 38835, 34208, 916,
    /// 279, 15678, 5562, 151643]` — no BOS (`qwen3vl`'s `tokenizer.ggml.
    /// add_bos_token` is `false`, unlike every other model this engine has
    /// been tested against) but *does* get a trailing EOS (151643,
    /// `add_eos_token = true`) — real llama.cpp's `LLAMA_POOLING_TYPE_LAST`
    /// pools whatever the actual last position is, so it's pooling the
    /// *EOS* token's hidden state here, not "dog"'s (the first version of
    /// this test used only the 9 content tokens, no EOS, and — pooling the
    /// wrong position entirely — got a real, wrong 0.15 cosine; this list
    /// must match `Tokenizer::encode_for_embedding`'s actual output
    /// exactly, not just the content tokens).
    ///
    /// Also exercises `Tokenizer::encode_for_embedding`'s BOS handling:
    /// an earlier version hardcoded `add_bos: true`, silently prepending a
    /// token real llama.cpp never adds for this model, and *that* bug
    /// alone (independent of the EOS one above) dropped cosine similarity
    /// to real llama.cpp's own embedding to ~0.47.
    ///
    /// This is the *last transformer hidden state* (`Self::run_layers`'s
    /// output, post-`output_norm`, no `lm_head`) at the final token
    /// position, L2-normalized — `LLAMA_POOLING_TYPE_LAST`, matching
    /// `PoolingType::Last`'s own dispatch in `http::openai::
    /// pooled_embedding`. Exercises this file's Q/K-norm addition (`Self::
    /// run_layers`'s `q_norm`/`k_norm` handling) and confirms M-RoPE
    /// degenerates to plain single-position RoPE for text-only input, as
    /// argued in `engine::loader`'s own `LLAMA_STYLE_ARCHITECTURES` doc
    /// comment. Run with `ORANGU_TEST_QWEN3VL_MODEL=/path/to/Qwen3-VL-
    /// Embedding-8B.Q4_K_M.gguf cargo test --release --bin orangu-server
    /// real_model_tests -- --ignored`.
    #[test]
    #[ignore]
    fn qwen3vl_embedding_matches_real_llama_cpp() {
        let path =
            std::env::var("ORANGU_TEST_QWEN3VL_MODEL").expect("set ORANGU_TEST_QWEN3VL_MODEL");
        let loaded = LoadedModel::open(std::path::Path::new(&path)).expect("load model");
        assert_eq!(loaded.config.architecture, "qwen3vl");
        assert_eq!(loaded.config.pooling_type, PoolingType::Last);
        let model =
            LlamaModel::load_with_backend(&loaded, Arc::new(crate::engine::backend::CpuBackend))
                .expect("build model");

        let tokens: Vec<u32> = vec![
            785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 151643,
        ];
        let n_embd = model.config().n_embd;
        let hidden = model
            .forward_hidden_states(&tokens)
            .expect("forward_hidden_states");
        assert_eq!(hidden.len(), tokens.len() * n_embd);

        let mut pooled = hidden[(tokens.len() - 1) * n_embd..].to_vec();
        let norm = pooled.iter().map(|v| v * v).sum::<f32>().sqrt();
        for v in pooled.iter_mut() {
            *v /= norm;
        }

        let reference: Vec<f32> = include_str!("testdata/qwen3vl_embedding_reference.csv")
            .trim()
            .split(',')
            .map(|v| v.parse().expect("reference fixture value"))
            .collect();
        assert_eq!(
            reference.len(),
            n_embd,
            "reference fixture has wrong length"
        );

        let cosine: f32 = pooled.iter().zip(&reference).map(|(a, b)| a * b).sum();
        assert!(
            cosine > 0.99,
            "cosine similarity to real llama.cpp's embedding was only {cosine}, expected > 0.99"
        );
    }
}
