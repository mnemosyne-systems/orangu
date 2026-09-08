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

//! The **multi-token-prediction (NextN/MTP) draft head** that ships beside a
//! [`super::qwen4exp`] model — e.g. the `MTP/mtp-*.gguf` files in
//! `unsloth/Qwen3.8-Flash-Next-GGUF`. Confirmed against real upstream source
//! (the `qwen4exp` MTP graph and the draft driver that feeds it), read
//! directly rather than inferred from tensor shapes.
//!
//! ## What a head is
//!
//! One decoder block of exactly the trunk's shape — full attention, the two
//! hyper-connection mixers, the 512-expert top-10 FFN — with three tensors
//! in front of it and one mixer behind it that the trunk has no counterpart
//! for:
//!
//! | tensor | what it does |
//! | :-- | :-- |
//! | `nextn.enorm` | RMSNorm weight on the embedding of the token just produced |
//! | `nextn.hnorm` | the same grouped norm the mixers use, on the trunk's wide state |
//! | `nextn.eh_proj` | `[2 * n_embd, n_embd]` — projects the two, concatenated per stream, back to one stream's width |
//! | `nextn.hc_head_*` | the head's own final collapse, standing in for the trunk's `output_hc_*` |
//!
//! So a draft step reads a **pair**: the token at position `p`, and the
//! served model's own hidden state at position `p - 1`. That pairing is the
//! whole trick — the state already carries the context, which is what lets
//! one block stand in for forty-eight of them. It is also why this is not a
//! [`ModelForward`](super::ModelForward): there is no trunk here to run.
//!
//! ## What it never changes
//!
//! The output. Every drafted token is checked against what the served model
//! would itself have produced at that position and dropped when it differs
//! (`engine::generate`'s `speculative_next`), so a head can only make an
//! answer arrive sooner. A head that drafts badly costs throughput; it
//! cannot cost correctness. That is what licenses the two deliberate
//! departures below.
//!
//! ## Two deliberate departures
//!
//! **No query-sparse attention.** The block carries `indexer.*` tensors and
//! this runs dense attention instead, as upstream's own MTP graph does.
//! Below `attention.indexer.top_k` (2048) cached positions the indexer
//! selects everything anyway, so on short contexts the two are the same
//! arithmetic; past that they differ, and the difference lands in how often
//! a guess is accepted rather than in what is emitted.
//!
//! **The head sees only what it was fed.** Its attention cache is indexed by
//! the served model's absolute positions and filled as the trunk commits
//! them, so a prompt prefix that a previous request already cached
//! (`engine::prefix_cache`) never reaches it: the state rows it would need
//! were never recomputed. Those positions are held as placeholders and
//! excluded from every window rather than attended as zeros — attending a
//! zero row is not "no contribution", it is a key that scores zero and takes
//! a share of the softmax.
//!
//! ## Shared heads
//!
//! `qwen4exp.nextn_shared_target_tensors` marks a head exported without a
//! token embedding or an output projection of its own: it borrows the served
//! model's, which is around 1.3 GB it does not have to hold. Borrowing is
//! why [`Qwen4ExpMtpHead::load`] takes the model rather than being handed a
//! path — and why a head loaded on its own has nothing to be: see
//! `engine::loader`'s draft-head refusal.

use anyhow::{Context, Result};
use std::sync::Arc;

use super::qwen_hybrid::{Dims, FullAttn, HybridFfn as _, LayerTensors, MoeFfn};
use super::qwen4exp::{self, HcMixer, Qwen4ExpModel};
use super::{ModelForward as _, MtpHead, MtpStep};
use crate::engine::backend::Backend;
use crate::engine::kv_cache::KvCache;
use crate::engine::loader::{LoadedModel, QuantMatrix};
use crate::engine::tensor;

pub struct Qwen4ExpMtpHead {
    backend: Arc<dyn Backend>,
    dims: Dims,
    /// `hyper_connection.count`.
    hc: usize,
    n_vocab: usize,
    /// `nextn.enorm` — `[n_embd]`, on the token embedding.
    enorm: Vec<f32>,
    /// `nextn.hnorm` — `[hc * n_embd]`, on the wide state.
    hnorm: Vec<f32>,
    /// `nextn.eh_proj` — `[2 * n_embd, n_embd]`.
    eh_proj: QuantMatrix,
    hc_attn: HcMixer,
    hc_ffn: HcMixer,
    attn: FullAttn,
    ffn: MoeFfn,
    /// `nextn.hc_head_*` — this head's own final collapse.
    head: HcMixer,
    /// The head's own, or the served model's when the export is a `shared-`
    /// one. Cheap to clone either way — a [`QuantMatrix`] is a handle.
    tok_embeddings: QuantMatrix,
    output_weight: QuantMatrix,
}

impl Qwen4ExpMtpHead {
    /// Builds the head `loaded` holds against the model it drafts for.
    ///
    /// Every dimension is checked against `target` rather than assumed: a
    /// head paired with the wrong model produces a well-formed state row of
    /// the wrong width, and the only symptom of the ones that happen to
    /// match would be an acceptance rate of zero.
    pub fn load(
        loaded: &LoadedModel,
        backend: Arc<dyn Backend>,
        target: &Qwen4ExpModel,
    ) -> Result<Self> {
        let dims = Dims::from_loaded(loaded)?;
        let hc = loaded
            .metadata_u64("hyper_connection.count")
            .context("missing hyper_connection.count")? as usize;
        anyhow::ensure!(hc > 0, "hyper_connection.count must be at least 1");

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

        let target_width = target.mtp_state_width();
        anyhow::ensure!(
            target_width == Some(hc * dims.n_embd),
            "draft head state is {} wide but the served model's is {:?} — this head belongs to \
             a different model",
            hc * dims.n_embd,
            target_width,
        );
        let target_dims = target.dims();
        anyhow::ensure!(
            dims.n_head == target_dims.n_head
                && dims.n_head_kv == target_dims.n_head_kv
                && dims.head_dim == target_dims.head_dim
                && dims.rope_dim == target_dims.rope_dim,
            "draft head attention shape disagrees with the served model's"
        );

        let n_expert_used = loaded
            .metadata_u64("expert_used_count")
            .context("missing expert_used_count")? as usize;

        let t = LayerTensors { loaded, i: block };
        let prefix = format!("blk.{block}");

        // A `shared-` export carries neither, and borrowing them is the
        // point of it. A self-contained one carries both, and then they are
        // its own — but they still have to agree with the model whose token
        // ids this head is about to predict.
        let borrowed = !loaded.has_tensor("token_embd.weight");
        let (tok_embeddings, output_weight) = if borrowed {
            (
                target.tok_embeddings().clone(),
                target.output_weight().clone(),
            )
        } else {
            let embd = loaded
                .matrix("token_embd.weight")
                .context("loading the draft head's token_embd.weight")?;
            let out = if loaded.has_tensor("output.weight") {
                loaded
                    .matrix("output.weight")
                    .context("loading the draft head's output.weight")?
            } else {
                embd.clone()
            };
            (embd, out)
        };
        anyhow::ensure!(
            output_weight.out_dim == target.config().n_vocab,
            "draft head predicts over {} tokens, the served model over {} — a pair that \
             disagrees about what an id means cannot be verified",
            output_weight.out_dim,
            target.config().n_vocab,
        );

        Ok(Self {
            dims,
            hc,
            n_vocab: target.config().n_vocab,
            enorm: t.vec("nextn.enorm.weight")?,
            hnorm: t.vec("nextn.hnorm.weight")?,
            eh_proj: t.matrix("nextn.eh_proj.weight")?,
            hc_attn: HcMixer::load(loaded, &format!("{prefix}.hc_attn"), true)?,
            hc_ffn: HcMixer::load(loaded, &format!("{prefix}.hc_ffn"), true)?,
            // The one attention layer this head has, so cache slot 0.
            attn: FullAttn::load(&t, 0)?,
            ffn: MoeFfn::load(&t, n_expert_used)?,
            head: HcMixer::load(loaded, &format!("{prefix}.nextn.hc_head"), false)?,
            tok_embeddings,
            output_weight,
            backend,
        })
    }

    /// The pre-block: the two normalized halves of the pair, concatenated
    /// per stream and projected back to one stream's width.
    ///
    /// `states` is the *previous* position's wide state, one row of `hc *
    /// n_embd` per token. The embedding is normalized once and repeated
    /// across the streams; the state is normalized per stream, exactly as
    /// the hyper-connection mixers normalize it.
    fn pair(&self, tokens: &[u32], states: &[f32]) -> Result<Vec<f32>> {
        let n_embd = self.dims.n_embd;
        let hc = self.hc;
        let hc_dim = hc * n_embd;
        let n_tokens = tokens.len();

        let mut h_norm = Vec::new();
        super::rms_norm_rows_into(&mut h_norm, states, n_embd, self.dims.rms_eps);
        for row in h_norm.chunks_mut(hc_dim) {
            tensor::mul_inplace(row, &self.hnorm);
        }

        // One row per (token, stream): the normed embedding, then that
        // stream's normed state. `eh_proj` reads `2 * n_embd` at a time, so
        // the streams are just more rows to it.
        let mut concat = vec![0f32; n_tokens * hc * 2 * n_embd];
        for (t, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            anyhow::ensure!(tok < self.n_vocab, "token id {tok} is out of vocab range");
            let mut e = self.tok_embeddings.row(tok);
            super::rms_norm_rows(&mut e, n_embd, self.dims.rms_eps);
            tensor::mul_inplace(&mut e, &self.enorm);
            for c in 0..hc {
                let at = (t * hc + c) * 2 * n_embd;
                concat[at..at + n_embd].copy_from_slice(&e);
                concat[at + n_embd..at + 2 * n_embd]
                    .copy_from_slice(&h_norm[(t * hc + c) * n_embd..(t * hc + c + 1) * n_embd]);
            }
        }
        Ok(super::matmul_host_fallback(
            self.backend.as_ref(),
            &concat,
            n_tokens * hc,
            &self.eh_proj,
        ))
    }

    /// Which cached positions each token may attend, or `None` when that is
    /// everything up to its own position anyway.
    ///
    /// `Some` only where a prefix the head never saw sits in front of the
    /// rows it did — see this module's note on placeholder rows.
    fn window(
        &self,
        n_tokens: usize,
        start_pos: usize,
        first_pos: usize,
    ) -> Option<Vec<Vec<usize>>> {
        (first_pos > 0).then(|| {
            (0..n_tokens)
                .map(|t| (first_pos..=start_pos + t).collect())
                .collect()
        })
    }
}

impl MtpHead for Qwen4ExpMtpHead {
    fn state_width(&self) -> usize {
        self.hc * self.dims.n_embd
    }

    fn new_kv_cache(&self, capacity: usize) -> KvCache {
        // One full-attention layer, no recurrent state of any kind: a head
        // is a block, not a trunk.
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
        anyhow::ensure!(n_tokens > 0, "a draft step needs at least one token");
        anyhow::ensure!(
            states.len() == n_tokens * self.state_width(),
            "expected {} state values for {n_tokens} positions, got {}",
            n_tokens * self.state_width(),
            states.len(),
        );

        // Rows the head was never fed, held so its cache stays indexed by
        // the served model's absolute positions. Never attended — `window`
        // excludes them — so their contents are irrelevant and zero is the
        // cheapest thing to write.
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

        let n_embd = self.dims.n_embd;
        let mut x = self.pair(tokens, states)?;

        let selection = self.window(n_tokens, start_pos, first_pos);
        let (cur, inject) = qwen4exp::hc_mix(
            self.backend.as_ref(),
            n_embd,
            self.hc,
            self.dims.rms_eps,
            &self.hc_attn,
            &x,
            n_tokens,
        );
        let inject = inject.expect("a block mixer always carries its injection weights");
        let attn_out = self.attn.forward(
            self.backend.as_ref(),
            &self.dims,
            cache,
            &cur,
            n_tokens,
            start_pos,
            selection.as_deref(),
        );
        qwen4exp::hc_combine(n_embd, self.hc, &mut x, &attn_out, &inject, n_tokens);

        let (cur, inject) = qwen4exp::hc_mix(
            self.backend.as_ref(),
            n_embd,
            self.hc,
            self.dims.rms_eps,
            &self.hc_ffn,
            &x,
            n_tokens,
        );
        let inject = inject.expect("a block mixer always carries its injection weights");
        let ffn_out = self
            .ffn
            // The draft head is not one of the model's layers: no compiled
            // block, and its activations are not a layer's to calibrate on.
            .forward(
                self.backend.as_ref(),
                n_embd,
                &cur,
                n_tokens,
                crate::engine::NOT_A_MODEL_LAYER,
            );
        qwen4exp::hc_combine(n_embd, self.hc, &mut x, &ffn_out, &inject, n_tokens);

        let hc_dim = self.hc * n_embd;
        let last = &x[(n_tokens - 1) * hc_dim..];
        // The wide stream, before it is collapsed: the next chained draft
        // step re-enters here in place of the served model's own state.
        let state = last.to_vec();
        let (out, _) = qwen4exp::hc_mix(
            self.backend.as_ref(),
            n_embd,
            self.hc,
            self.dims.rms_eps,
            &self.head,
            last,
            1,
        );
        let logits = self.backend.matmul(&out, 1, &self.output_weight);
        Ok(MtpStep { logits, state })
    }
}

#[cfg(test)]
mod tests {
    /// A cache the head never saw the front of attends only what it did —
    /// and when it saw everything, it attends everything, which is the
    /// (GPU-capable) dense path.
    #[test]
    fn a_window_is_only_narrowed_when_there_is_a_prefix_to_exclude() {
        let dense: Option<Vec<Vec<usize>>> = None;
        assert_eq!(window_for(3, 10, 0), dense);
        assert_eq!(
            window_for(2, 10, 4),
            Some(vec![
                vec![4, 5, 6, 7, 8, 9, 10],
                vec![4, 5, 6, 7, 8, 9, 10, 11]
            ])
        );
    }

    /// The window rule alone, so it can be checked without a head.
    fn window_for(n_tokens: usize, start_pos: usize, first_pos: usize) -> Option<Vec<Vec<usize>>> {
        (first_pos > 0).then(|| {
            (0..n_tokens)
                .map(|t| (first_pos..=start_pos + t).collect())
                .collect()
        })
    }
}

#[cfg(test)]
mod real_model_tests {
    use super::*;

    /// How many decode steps the acceptance measurement below runs.
    const STEPS: usize = 12;

    /// Cross-check against the real checkpoint and the real head: how often
    /// the head's guess is the token the served model itself goes on to
    /// produce.
    ///
    /// **This is the only test that can catch what this head invites**, and
    /// what it invites is silence. A head fed the state of the wrong
    /// position, normalized over the wrong axis, or concatenated in the
    /// wrong order still produces a well-formed `[n_vocab]` row and a
    /// fluent, plausible token — and because every guess is verified, a
    /// completely broken head *still emits the correct answer*, just no
    /// faster. Nothing downstream can tell the difference. Acceptance can.
    ///
    /// Upstream measures 0.66 for this head against this model at greedy;
    /// the bar here is far below that, because the point is to separate
    /// "wired correctly" from "wired plausibly", and every one of the
    /// mistakes above lands at roughly zero.
    ///
    /// Run with `ORANGU_TEST_QWEN4EXP_MODEL=/path/to/first-shard.gguf
    /// ORANGU_TEST_QWEN4EXP_MTP=/path/to/mtp-head.gguf cargo test --release
    /// --bin orangu-server qwen4exp_mtp::real_model_tests -- --ignored
    /// --nocapture`. Expect minutes, not seconds.
    #[test]
    #[ignore]
    fn the_head_guesses_what_the_model_goes_on_to_say() {
        let model_path = std::env::var("ORANGU_TEST_QWEN4EXP_MODEL")
            .expect("set ORANGU_TEST_QWEN4EXP_MODEL to a Qwen3.8-Flash-Next GGUF");
        let head_path = std::env::var("ORANGU_TEST_QWEN4EXP_MTP")
            .expect("set ORANGU_TEST_QWEN4EXP_MTP to one of its MTP/mtp-*.gguf heads");
        let backend: Arc<dyn Backend> = Arc::new(crate::engine::backend::CpuBackend);

        let loaded = LoadedModel::open(std::path::Path::new(&model_path)).expect("load model");
        let gguf =
            orangu::gguf::GgufFile::open(std::path::Path::new(&model_path)).expect("open gguf");
        let tokenizer =
            crate::engine::tokenizer::Tokenizer::from_gguf(&gguf).expect("build tokenizer");
        let model =
            Qwen4ExpModel::load_with_backend(&loaded, backend.clone()).expect("build model");

        let head_loaded = LoadedModel::open(std::path::Path::new(&head_path)).expect("load head");
        let head = Qwen4ExpMtpHead::load(&head_loaded, backend, &model).expect("build head");
        let width = head.state_width();

        // No BOS: this file sets `tokenizer.ggml.add_bos_token = 0`.
        let prompt = tokenizer.encode(
            "The capital of France is Paris, and the capital of Italy is",
            false,
        );
        let n = prompt.len();

        let mut cache = model.new_kv_cache(256);
        let (logits, states) = model
            .forward_with_states(&mut cache, &prompt, 0, 0)
            .expect("prefill");

        // The head catches up over the whole prompt, each token paired with
        // the state of the position before it — zeros at position 0, which
        // is the only position that has no predecessor.
        let mut rows = vec![0f32; width];
        rows.extend_from_slice(&states[..(n - 1) * width]);
        let mut head_cache = head.new_kv_cache(256);
        let step = head
            .forward(&mut head_cache, &prompt, &rows, 0, 0)
            .expect("head catch-up");

        let mut next = crate::engine::sampling::argmax(&logits);
        // The catch-up's last row reads the prompt's last token, so what it
        // predicts is the position the model has just predicted too.
        let mut guess = crate::engine::sampling::argmax(&step.logits);
        let mut accepted = 0usize;
        let mut tried = 0usize;
        let mut state = states[(n - 1) * width..].to_vec();

        for pos in (n..).take(STEPS) {
            tried += 1;
            if guess == next {
                accepted += 1;
            }
            eprintln!(
                "pos {pos}: model {:?}, head {:?}{}",
                tokenizer.decode(&[next]),
                tokenizer.decode(&[guess]),
                if guess == next { "  <-- accepted" } else { "" },
            );
            // The head drafts the token after `next`, from `next` itself and
            // the state behind it — the pairing the whole thing rests on.
            let step = head
                .forward(&mut head_cache, &[next], &state, pos, 0)
                .expect("head step");
            guess = crate::engine::sampling::argmax(&step.logits);

            let (logits, produced) = model
                .forward_with_states(&mut cache, &[next], pos, 0)
                .expect("decode");
            next = crate::engine::sampling::argmax(&logits);
            state = produced;
        }

        eprintln!("acceptance: {accepted}/{tried}");
        assert!(
            accepted * 3 >= tried,
            "the head agreed with the model {accepted} times out of {tried}; a head wired \
             correctly agrees roughly two thirds of the time and a head wired wrongly agrees \
             almost never"
        );
    }
}
