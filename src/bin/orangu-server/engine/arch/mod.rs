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

//! One implementor per architecture family — `llama` (GQA/RoPE/RMSNorm/
//! SwiGLU), `gemma` (soft-capping/sliding-window/GEGLU), `qwen35`
//! (hybrid full-attention/gated-DeltaNet, dense FFN), and `qwen35moe`
//! (the same hybrid attention shape, mixture-of-experts FFN) — so adding
//! a family is additive rather than a rewrite of the others.

pub mod gemma;
pub mod llama;
pub mod qwen35;
pub mod qwen35moe;

use crate::engine::kv_cache::KvCache;
use crate::engine::loader::ModelConfig;
use anyhow::Result;

/// Repeat-penalty state a caller passes to [`ModelForward::forward_maybe_
/// sampling`] when it wants greedy sampling done for it.
/// `recent_tokens` must already be trimmed to the sampler's own
/// `repeat_last_n` window (mirroring `engine::sampling`'s own
/// `apply_repeat_penalty`, which does the same trim before applying the
/// penalty) — the callee applies the penalty to exactly the slice it's
/// given, nothing more.
pub struct GreedySampleParams<'a> {
    pub recent_tokens: &'a [u32],
    pub repeat_penalty: f32,
}

/// [`ModelForward::forward_maybe_sampling`]'s result: either the callee
/// already picked the next token itself (`Token`, only possible when the
/// caller asked for greedy sampling *and* the backend has a GPU fast path
/// for it), or it didn't and the caller must run `engine::sampling::
/// Sampler::sample` over the returned logits itself, exactly as a plain
/// `forward` call would have required.
pub enum ForwardOutcome {
    Token(u32),
    Logits(Vec<f32>),
}

/// One sequence's pending single-token decode step, as an element of a
/// cross-sequence batch (see `engine::batch::BatchCoordinator`).
/// Each sequence keeps its own `cache`/`start_pos`/`greedy_sample`
/// (attention, RoPE, and the KV-cache write all stay per-sequence even
/// when the *matmul* steps in between are fused across every item in the
/// batch — see [`ModelForward::forward_batch_decode`]'s own doc comment).
pub struct BatchDecodeItem<'a> {
    pub cache: &'a mut KvCache,
    pub token: u32,
    pub start_pos: usize,
    pub greedy_sample: Option<GreedySampleParams<'a>>,
    /// The submitting request's own `engine::scheduler::SlotGuard::id()`
    /// — **not** this item's position within `items`. `BatchCoordinator`
    /// can run two *different* batches' own `forward_batch_decode` calls
    /// genuinely concurrently (its own doc comment: the coordinator's
    /// lock is released before processing a batch, specifically so a new
    /// window can start collecting while an older one is still being
    /// processed) — a GPU-resident implementation that caches per-
    /// sequence resources by *array index within one call*
    /// (`GemmaModel::record_batched_decode_forward`'s first, broken
    /// attempt at this: `1..=items.len()`) would let two unrelated
    /// sequences from two concurrently-running batches collide on the
    /// exact same cached buffers — since the array index resets to `0`
    /// every call, two overlapping batches' own "index 0" always
    /// coincide, corrupting whichever one loses the race, silently. A
    /// `SlotGuard`'s own `id()` is unique across every *concurrently
    /// held* slot for that slot's entire lifetime (`SlotPool::acquire`'s
    /// own guarantee — capped by the semaphore, one live guard per index
    /// at a time), which is exactly the uniqueness overlapping batches
    /// need and a per-call array index doesn't provide.
    pub slot_id: usize,
}

pub trait ModelForward: Send + Sync {
    fn config(&self) -> &ModelConfig;

    /// A fresh KV cache sized for `capacity` positions, for a new sequence.
    fn new_kv_cache(&self, capacity: usize) -> KvCache;

    /// Runs `tokens` (a contiguous chunk of one sequence, starting at
    /// absolute position `start_pos`) through the model, appending their
    /// key/value vectors to `cache` as it goes, and returns the next-token
    /// logits (`[n_vocab]`) for the *last* token in `tokens` only — the one
    /// prediction a caller doing either prefill (find where generation
    /// starts) or decode (one token at a time) actually needs.
    ///
    /// `slot_id` is the caller's own `engine::scheduler::SlotGuard::id()` —
    /// see [`BatchDecodeItem::slot_id`]'s doc comment for why a real,
    /// per-request slot id (not a shared constant) is load-bearing here:
    /// `GemmaModel`'s Vulkan decode path threads it into the same
    /// per-sequence GPU resource cache the batched path uses, so two
    /// `slots > 1` requests decoding concurrently don't collide on the same
    /// cached buffers. Architectures/backends with no such per-caller cache
    /// (every non-Vulkan-decode path) simply ignore it.
    fn forward(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        slot_id: usize,
    ) -> Result<Vec<f32>>;

    /// Like `forward`, but lets the implementor sample the next token
    /// itself when `greedy_sample` is `Some` — skipping the full
    /// `[n_vocab]` logits readback entirely when it can. The default
    /// implementation always falls back to `forward` plus
    /// `ForwardOutcome::Logits`, so every architecture and backend
    /// combination keeps working correctly with no override needed; only
    /// `GemmaModel`'s Vulkan decode path currently overrides this to fuse
    /// the argmax into the same GPU submission.
    fn forward_maybe_sampling(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        greedy_sample: Option<GreedySampleParams<'_>>,
        slot_id: usize,
    ) -> Result<ForwardOutcome> {
        let _ = greedy_sample;
        self.forward(cache, tokens, start_pos, slot_id)
            .map(ForwardOutcome::Logits)
    }

    /// Like [`Self::forward`], but returns next-token logits for *every* input
    /// position (`Vec` of `n_tokens` `[n_vocab]` rows), not just the last —
    /// what speculative decoding's verify step needs to check each drafted
    /// token against the model's own prediction at that position. Appends all
    /// `tokens` to `cache` (which a later [`KvCache::truncate`] can roll back
    /// to the accepted prefix), and runs the multi-token / CPU-orchestrated
    /// path so one weight stream covers the whole draft at once.
    ///
    /// The default errors: an architecture opts in only by overriding this. A
    /// caller must not reach it for a model whose `forward` uses a KV path this
    /// can't stay consistent with (e.g. one that leaves keys/values GPU-only) —
    /// speculative decoding is gated to the plain CPU-KV path for that reason.
    fn forward_all_logits(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        slot_id: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let _ = (cache, tokens, start_pos, slot_id);
        anyhow::bail!("this architecture does not support multi-position (speculative) decoding")
    }

    /// Every token's final hidden state (`[n_tokens, n_embd]`, before the
    /// output projection to vocab logits) — what an embeddings request
    /// pools over. A one-shot call: no KV cache reuse across calls.
    fn forward_hidden_states(&self, tokens: &[u32]) -> Result<Vec<f32>>;

    /// Applied to the pooled embedding vector (`[n_embd]`, after mean/CLS/
    /// last-token pooling) before L2 normalization. The default is the
    /// identity — most architectures have nothing here — but a model
    /// converted with extra sentence-transformers "Dense" adapter layers
    /// (e.g. `gemma-embedding`'s `dense_2`/`dense_3`, confirmed against
    /// upstream `llama.cpp`'s `llm_graph_context::build_dense_out`: applied
    /// *after* pooling, not before) overrides this to run them. May change
    /// the vector's length (`gemma-embedding`'s `dense_2` widens
    /// `n_embd -> 4*n_embd` before `dense_3` narrows it back).
    fn post_pool_projection(&self, pooled: Vec<f32>) -> Result<Vec<f32>> {
        Ok(pooled)
    }

    /// Runs a *cross-sequence batch* of independent single-token decode
    /// steps, driven by `engine::batch::BatchCoordinator`.
    /// Each `items[i]` is one sequence's own pending decode step (its own
    /// KV cache, position, token); the *matmul* steps every layer needs
    /// (QKV, `wo`, FFN, PLE, `lm_head` — the weight-bandwidth-heavy ones,
    /// the actual thing "GEMM batching" amortizes) are fused into one call
    /// across every item in the batch instead of one call per sequence,
    /// while attention, RoPE, and the KV-cache write stay per-sequence
    /// (each sequence has its own cache and its own position — there is
    /// no shared state to fuse there, unlike the matmuls).
    ///
    /// The default implementation just loops over `items` calling
    /// `forward_maybe_sampling` on each independently (correct, no fusion,
    /// no batching win) — so `llama`/`qwen35moe` need no override to keep
    /// working; only `GemmaModel` currently overrides this with the real
    /// batched implementation. That override is correctness-verified but
    /// **measured slower** under real concurrent load than not batching at
    /// all — `engine::generate::Engine::batch_coordinator`'s own doc
    /// comment has the numbers and the likely cause — so it's only ever
    /// reached when a caller opts in (`ORANGU_BATCH_DECODE=1`); this
    /// default implementation is what every other configuration actually
    /// runs, including a `slots > 1` deployment that hasn't opted in.
    fn forward_batch_decode(
        &self,
        items: &mut [BatchDecodeItem<'_>],
    ) -> Result<Vec<ForwardOutcome>> {
        items
            .iter_mut()
            .map(|item| {
                self.forward_maybe_sampling(
                    item.cache,
                    &[item.token],
                    item.start_pos,
                    item.greedy_sample.take(),
                    item.slot_id,
                )
            })
            .collect()
    }
}
