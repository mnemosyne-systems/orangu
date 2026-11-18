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

//! The **multi-token-prediction (NextN/MTP) draft head** of a dense
//! [`super::qwen35`] model — e.g. `MTP/mtp-Qwen3.8-27B-*.gguf` beside
//! `unsloth/Qwen3.8-27B-GGUF`, or the `blk.N.nextn.*` block a trunk file
//! carries past its last layer. Confirmed against upstream's `qwen35` MTP
//! graph (`llama_model_qwen35::graph_mtp`), read directly.
//!
//! ## What a head is
//!
//! One full-attention decoder block of the trunk's own shape — the joint
//! query+gate projection, per-head Q/K norms, multi-section rotary, the
//! sigmoid output gate, a dense SwiGLU FFN — with three tensors in front of
//! it and one norm behind it:
//!
//! | tensor | what it does |
//! | :-- | :-- |
//! | `nextn.enorm` | RMSNorm weight on the embedding of the token just produced |
//! | `nextn.hnorm` | RMSNorm weight on the served model's state behind it |
//! | `nextn.eh_proj` | `[2 * n_embd, n_embd]` — the two, concatenated embedding first, projected back to one row |
//! | `nextn.shared_head_norm` | the norm in front of the output projection (the trunk's `output_norm` when absent) |
//!
//! A draft step reads a **pair**: the token at position `p` and the served
//! model's state at `p - 1`, which is its last layer's residual stream
//! *after* the final norm — the row the trunk's output projection reads.
//! The head's own state for a chained step is the same thing on its side:
//! its block's output after `shared_head_norm`.
//!
//! ## What it never changes
//!
//! The output. Every drafted token is checked against what the served model
//! would itself have produced at that position (`engine::generate`'s
//! `speculative_next`), so a head can only make an answer arrive sooner.
//!
//! ## Borrowed tensors
//!
//! The token embedding and the output projection are always the served
//! model's, whether or not the head's file carries copies: the trunk's
//! output projection is where the trunk placed it (a device when it fits),
//! and a second copy of a `[n_embd, n_vocab]` table read once per draft
//! token would cost about as much as the rest of the head together.

use anyhow::{Context, Result};
use std::sync::Arc;

use super::qwen_hybrid::{DenseFfn, Dims, FullAttn, HybridFfn as _, LayerTensors, Trunk};
use super::{MtpHead, MtpStep};
use crate::engine::backend::Backend;
use crate::engine::kv_cache::KvCache;
use crate::engine::loader::{LoadedModel, QuantMatrix};
use crate::engine::tensor;

pub struct Qwen35MtpHead {
    backend: Arc<dyn Backend>,
    dims: Dims,
    n_vocab: usize,
    /// `nextn.enorm` — `[n_embd]`, on the token embedding.
    enorm: Vec<f32>,
    /// `nextn.hnorm` — `[n_embd]`, on the served model's state.
    hnorm: Vec<f32>,
    /// `nextn.eh_proj` — `[2 * n_embd, n_embd]`.
    eh_proj: QuantMatrix,
    attn_norm: Vec<f32>,
    attn: FullAttn,
    post_attention_norm: Vec<f32>,
    ffn: DenseFfn,
    /// `nextn.shared_head_norm`, or the served model's `output_norm`.
    head_norm: Vec<f32>,
    tok_embeddings: QuantMatrix,
    output_weight: QuantMatrix,
}

impl Qwen35MtpHead {
    /// Builds the head `loaded` holds against the trunk it drafts for. Every
    /// dimension is checked against `target`: a head paired with the wrong
    /// model still produces well-formed rows, and its only symptom would be
    /// an acceptance rate of zero.
    pub fn load(
        loaded: &LoadedModel,
        backend: Arc<dyn Backend>,
        target: &Trunk<DenseFfn>,
    ) -> Result<Self> {
        let dims = Dims::from_loaded(loaded)?;
        let n_layer_all = loaded
            .metadata_u64("block_count")
            .context("missing block_count")? as usize;
        let n_nextn = loaded.metadata_u64("nextn_predict_layers").unwrap_or(0) as usize;
        anyhow::ensure!(
            n_nextn > 0,
            "this file declares no nextn_predict_layers, so it carries no draft head"
        );
        anyhow::ensure!(
            n_nextn == 1,
            "this file declares {n_nextn} NextN/MTP blocks; only a single-block head is supported"
        );
        let block = n_layer_all
            .checked_sub(n_nextn)
            .context("nextn_predict_layers is not smaller than block_count")?;

        let target_dims = target.dims();
        anyhow::ensure!(
            dims.n_embd == target_dims.n_embd
                && dims.n_head == target_dims.n_head
                && dims.n_head_kv == target_dims.n_head_kv
                && dims.head_dim == target_dims.head_dim
                && dims.rope_dim == target_dims.rope_dim,
            "draft head shape disagrees with the served model's — this head belongs to a \
             different model"
        );

        let t = LayerTensors { loaded, i: block };
        anyhow::ensure!(
            t.has("nextn.eh_proj.weight"),
            "{} is missing: this file carries no draft head",
            t.name("nextn.eh_proj.weight")
        );

        // A draft head of a Hadamard-folded file would need the fold on its
        // own projections too; none is wired here, so `finish` refuses one.
        let hadamard = crate::engine::hadamard::from_loaded(loaded)?;
        let mut fold = crate::engine::hadamard::FoldLedger::new(hadamard.as_ref());
        // The head's one attention layer: cache slot 0 of its own cache.
        let attn = FullAttn::load(&t, 0, &mut fold)?;
        fold.finish()?;
        anyhow::ensure!(
            target.head_is_unfolded(),
            "the served model's output projection is folded; a draft head cannot borrow it"
        );

        let head_norm = if t.has("nextn.shared_head_norm.weight") {
            t.vec("nextn.shared_head_norm.weight")?
        } else {
            target.output_norm().to_vec()
        };
        let output_weight = target.output_weight().clone();
        let n_vocab = output_weight.out_dim;

        Ok(Self {
            dims,
            n_vocab,
            enorm: t.vec("nextn.enorm.weight")?,
            hnorm: t.vec("nextn.hnorm.weight")?,
            eh_proj: t.matrix("nextn.eh_proj.weight")?,
            attn_norm: t.vec("attn_norm.weight")?,
            attn,
            post_attention_norm: t.vec("post_attention_norm.weight")?,
            ffn: DenseFfn {
                gate: t.matrix("ffn_gate.weight")?,
                up: t.matrix("ffn_up.weight")?,
                down: t.matrix("ffn_down.weight")?,
                gate_up_in: None,
                down_in: None,
            },
            head_norm,
            tok_embeddings: target.tok_embeddings().clone(),
            output_weight,
            backend,
        })
    }

    /// The pre-block: `eh_proj` over the normed embedding of each token
    /// beside the normed state behind it.
    fn pair(&self, tokens: &[u32], states: &[f32]) -> Result<Vec<f32>> {
        let n_embd = self.dims.n_embd;
        let eps = self.dims.rms_eps;
        let n_tokens = tokens.len();

        let mut h_norm = Vec::new();
        super::rms_norm_rows_into(&mut h_norm, states, n_embd, eps);
        for row in h_norm.chunks_mut(n_embd) {
            tensor::mul_inplace(row, &self.hnorm);
        }

        let mut concat = vec![0f32; n_tokens * 2 * n_embd];
        for (t, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            anyhow::ensure!(tok < self.n_vocab, "token id {tok} is out of vocab range");
            let mut e = self.tok_embeddings.row(tok);
            super::rms_norm_rows(&mut e, n_embd, eps);
            tensor::mul_inplace(&mut e, &self.enorm);
            let at = t * 2 * n_embd;
            concat[at..at + n_embd].copy_from_slice(&e);
            concat[at + n_embd..at + 2 * n_embd]
                .copy_from_slice(&h_norm[t * n_embd..(t + 1) * n_embd]);
        }
        Ok(crate::engine::backend::CpuBackend.matmul(&concat, n_tokens, &self.eh_proj))
    }
}

/// Which cached positions each token may attend, or `None` when that is
/// everything up to its own position anyway: `Some` only where a prefix
/// the head never saw sits in front of the rows it did, whose placeholder
/// rows must take no share of the softmax.
fn window_for(n_tokens: usize, start_pos: usize, first_pos: usize) -> Option<Vec<Vec<usize>>> {
    (first_pos > 0).then(|| {
        (0..n_tokens)
            .map(|t| (first_pos..=start_pos + t).collect())
            .collect()
    })
}

impl MtpHead for Qwen35MtpHead {
    fn state_width(&self) -> usize {
        self.dims.n_embd
    }

    fn new_kv_cache(&self, capacity: usize) -> KvCache {
        KvCache::new_mixed(capacity, &[self.dims.n_head_kv * self.dims.head_dim], &[])
    }

    fn forward(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        states: &[f32],
        start_pos: usize,
        first_pos: usize,
    ) -> Result<MtpStep> {
        let n_tokens = tokens.len();
        let n_embd = self.dims.n_embd;
        let eps = self.dims.rms_eps;
        anyhow::ensure!(n_tokens > 0, "a draft step needs at least one token");
        anyhow::ensure!(
            states.len() == n_tokens * n_embd,
            "expected {} state values for {n_tokens} positions, got {}",
            n_tokens * n_embd,
            states.len(),
        );

        // Rows the head was never fed, held so its cache stays indexed by
        // the served model's absolute positions; `window_for` keeps every
        // token from attending them.
        let slot = &mut cache.layers[0];
        let kv_dim = self.dims.n_head_kv * self.dims.head_dim;
        let zeros = vec![0f32; kv_dim];
        while slot.len < start_pos {
            slot.push(&zeros, &zeros);
        }
        anyhow::ensure!(
            slot.len == start_pos,
            "draft head cache holds {} positions but the step starts at {start_pos}",
            slot.len,
        );

        // The block runs on the host: its weights are the head file's own,
        // mapped there, and its cache is host rows alone — placeholders
        // and truncation included — which a device's attention, keeping
        // rows of its own, would not follow. Only the borrowed output
        // projection runs where the served model placed it.
        let host = &crate::engine::backend::CpuBackend;
        let mut x = self.pair(tokens, states)?;

        let mut normed = Vec::new();
        tensor::rmsnorm_into(&mut normed, &x, &self.attn_norm, n_tokens, n_embd, eps);
        let selection = window_for(n_tokens, start_pos, first_pos);
        let attn_out = self.attn.forward(
            host,
            &self.dims,
            cache,
            &normed,
            n_tokens,
            start_pos,
            selection.as_deref(),
            0,
        );
        tensor::add_inplace(&mut x, &attn_out);

        tensor::rmsnorm_into(
            &mut normed,
            &x,
            &self.post_attention_norm,
            n_tokens,
            n_embd,
            eps,
        );
        // Not one of the model's layers: no compiled block, and its
        // activations are not a layer's to calibrate on.
        let ffn_out = self.ffn.forward(
            host,
            n_embd,
            &normed,
            n_tokens,
            crate::engine::NOT_A_MODEL_LAYER,
        );
        tensor::add_inplace(&mut x, &ffn_out);

        let mut state = x[(n_tokens - 1) * n_embd..].to_vec();
        tensor::rmsnorm_inplace(&mut state, &self.head_norm, 1, n_embd, eps);
        let logits =
            super::matmul_host_fallback(self.backend.as_ref(), &state, 1, &self.output_weight);
        Ok(MtpStep { logits, state })
    }
}

#[cfg(test)]
mod tests {
    use super::window_for;

    #[test]
    fn a_window_is_only_narrowed_when_there_is_a_prefix_to_exclude() {
        assert_eq!(window_for(3, 10, 0), None);
        assert_eq!(
            window_for(2, 10, 4),
            Some(vec![
                vec![4, 5, 6, 7, 8, 9, 10],
                vec![4, 5, 6, 7, 8, 9, 10, 11]
            ])
        );
    }
}
