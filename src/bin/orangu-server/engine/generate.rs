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

//! Wires the model, tokenizer, and sampler into the one operation the HTTP
//! layer actually needs: take a prompt (already tokenized), stream back
//! generated tokens. Each call acquires a slot from the `SlotPool` (waiting
//! if every slot is busy), runs prefill+decode on its own blocking-pool
//! thread against its own KV cache, and reports throughput the same way
//! llama-server's own console log does.

use anyhow::Result;
use std::collections::VecDeque;
use std::io::Write;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc::{self, UnboundedReceiver};

use super::arch::{ForwardOutcome, GreedySampleParams, ModelForward};
use super::batch::{BatchCoordinator, BatchDecodeRequest, OwnedGreedySample};
use super::kv_cache::KvCache;
use super::prefix_cache::PrefixCache;
use super::sampling::{Sampler, SamplingParams};
use super::scheduler::SlotPool;
use super::tokenizer::Tokenizer;
use super::tool_calls;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
}

#[derive(Clone, Debug)]
pub struct GenerateStats {
    pub prompt_tokens: usize,
    /// How many of `prompt_tokens` came from a cache (the cross-slot prefix
    /// pool or this slot's own retained cache) and so never went through a
    /// forward pass. The difference between this and `prompt_tokens` is what
    /// `prompt_time` was actually spent on, which is what makes a prefill
    /// measurement interpretable — a fast prompt that was 99% cached says
    /// nothing about prefill speed.
    pub cached_tokens: usize,
    pub prompt_time: Duration,
    pub generated_tokens: usize,
    pub generate_time: Duration,
}

impl GenerateStats {
    pub fn prompt_tokens_per_second(&self) -> f64 {
        self.prompt_tokens as f64 / self.prompt_time.as_secs_f64().max(1e-9)
    }

    pub fn generate_tokens_per_second(&self) -> f64 {
        self.generated_tokens as f64 / self.generate_time.as_secs_f64().max(1e-9)
    }

    /// Prompt tokens that actually needed a forward pass this request.
    pub fn prefilled_tokens(&self) -> usize {
        self.prompt_tokens.saturating_sub(self.cached_tokens)
    }

    /// The line printed to stdout per completed request — llama-server's
    /// own console log carries the same two figures.
    pub fn log_line(&self) -> String {
        format!(
            "prompt {} tokens in {:.2}s ({:.2} tok/s), generated {} tokens in {:.2}s ({:.2} tok/s)",
            self.prompt_tokens,
            self.prompt_time.as_secs_f64(),
            self.prompt_tokens_per_second(),
            self.generated_tokens,
            self.generate_time.as_secs_f64(),
            self.generate_tokens_per_second(),
        )
    }
}

pub struct GenerateRequest {
    pub prompt_tokens: Vec<u32>,
    pub sampling: SamplingParams,
    /// Constrain the output to valid JSON — OpenAI's
    /// `response_format: {"type": "json_object"}`.
    ///
    /// Not a sampler parameter, because it is not a knob on the distribution:
    /// it decides which tokens *exist* at each step rather than how likely
    /// they are, and it needs the tokenizer, which `SamplingParams` has no
    /// business knowing about.
    pub json_output: bool,
    pub max_tokens: usize,
    pub stop_token_ids: Vec<u32>,
    /// Whether this request may reuse an already-computed KV cache for
    /// whatever prefix of its prompt one is available for (llama.cpp's field
    /// of the same name, and its default of `true`). Setting it `false` forces
    /// every prompt token through a real forward pass, which is what makes a
    /// prefill measurement mean anything — a cached prompt "prefills" in
    /// microseconds and measures only the lookup. It does not stop this
    /// request's own cache from being *stored* for later requests.
    pub cache_prompt: bool,
    /// Pin this request to one specific slot (llama.cpp's field of the same
    /// name), waiting for it rather than taking whichever slot is free.
    ///
    /// What it buys is cache affinity, not fairness: a slot retains the
    /// `(tokens, KvCache)` of the last request that ran on it
    /// (`engine::slot_store`), so a conversation that returns to its own slot
    /// continues from a warm prefix. Landing on a neighbour instead finds
    /// another conversation's cache and reprefills the whole prompt — which is
    /// exactly what happened for as long as this field did not exist and the
    /// client's `id_slot` was parsed away.
    ///
    /// `None` keeps the old behaviour: any free slot.
    pub id_slot: Option<usize>,
    /// Emit a [`StreamEvent::Timings`] after every generated token
    /// (llama.cpp's `timings_per_token`). Off by default: it costs one extra
    /// channel message per token, which is nothing next to a forward pass but
    /// is pure waste for a caller that never reads it.
    pub timings_per_token: bool,
}

impl Default for GenerateRequest {
    fn default() -> Self {
        Self {
            prompt_tokens: Vec::new(),
            sampling: SamplingParams::default(),
            json_output: false,
            max_tokens: 0,
            stop_token_ids: Vec::new(),
            cache_prompt: true,
            id_slot: None,
            timings_per_token: false,
        }
    }
}

pub enum StreamEvent {
    Token(String),
    /// The admission queue was full, so this request was refused before it
    /// ever reached a slot.
    ///
    /// Its own event rather than an `Error`, because the two mean different
    /// things to a caller: an error is a request that cannot be served, and
    /// this is one that could be served later. It becomes a `503` with
    /// `Retry-After`, which a client can act on, where a `500` invites a
    /// bug report.
    Overloaded,
    /// Emitted after each prefill chunk, while the prompt is still being
    /// processed. Until this existed the only progress report was the `Done`
    /// event at the very end, so a client had nothing to show during the part
    /// of a turn that takes the longest — a 30-second prefill looked
    /// identical to a hung server.
    PromptProgress {
        total: usize,
        cached: usize,
        processed: usize,
        elapsed: Duration,
    },
    /// Emitted after each generated token when the request asked for
    /// per-token timings. Carries the server's own measurement, so a client
    /// never has to estimate a decode rate from its own wall clock — or, as
    /// this project's own client did, by re-tokenising the whole accumulated
    /// answer on every redraw.
    Timings(GenerateStats),
    Done {
        stats: GenerateStats,
        finish_reason: FinishReason,
    },
    Error(String),
}

/// A second, smaller model used to draft tokens the served model then
/// verifies — `[orangu-server].draft_model`.
///
/// Held whole rather than as a bare `Arc<dyn ModelForward>` because the two
/// things a speculative step needs beyond the forward pass — how many tokens
/// to draft, and what to call the pair in a log line — have nowhere else to
/// live that both the request path and the startup banner can reach.
pub struct DraftModel {
    pub model: Arc<dyn ModelForward>,
    /// How many tokens to draft per verification. See
    /// `[orangu-server].draft_tokens`.
    pub tokens: usize,
    /// The draft's own model label, for the banner and the acceptance log.
    pub label: String,
}

pub struct Engine {
    pub model: Arc<dyn ModelForward>,
    /// The draft half of a speculative pair, when one is configured.
    ///
    /// `None` — the default — leaves decoding exactly as it was.
    ///
    /// **Both** models must implement `ModelForward::forward_all_logits`, not
    /// only the target. The target obviously needs it — verification is one
    /// multi-position forward — but so does the draft, for a less obvious
    /// reason: it is the only entry point that keeps the KV rows on the host,
    /// and a draft cache is rolled back and re-read on every single step. A
    /// draft running through the single-token `forward` would take the fused
    /// GPU decode path instead, whose rows exist only on the device. See
    /// [`draft_forward`].
    pub draft: Option<Arc<DraftModel>>,
    pub tokenizer: Arc<Tokenizer>,
    pub chat_template_source: Option<String>,
    pub slots: Arc<SlotPool>,
    /// The cross-sequence GEMM batching coordinator — `Some` only when
    /// `slots.total() > 1` *and*
    /// `ORANGU_BATCH_DECODE=1` is set; a single-slot deployment, or
    /// `slots > 1` without the env var (the default), keeps calling
    /// `ModelForward::forward_maybe_sampling` directly, unchanged.
    ///
    /// **Off by default**, unlike every other GPU-fused change in this
    /// project. `GemmaModel::record_batched_decode_forward` *is*
    /// GPU-resident (every item in a batch chained into one shared
    /// encoder/submission — the one-round-trip design every single-
    /// sequence decode step already uses, not the old CPU-orchestrated
    /// per-layer-round-trip path an earlier version of this comment
    /// described), and is correctness-verified bit-for-bit against
    /// independent per-sequence `forward` calls
    /// (`engine::arch::gemma`'s own `forward_batch_decode_matches_
    /// independent_forward_calls_*` tests) as well as against itself
    /// across many autoregressive steps
    /// (`forward_batch_decode_identical_prompts_stay_identical_over_
    /// many_steps_vulkan`). It still measures **slower** than not
    /// batching under real concurrent load, though: a reproducible
    /// concurrent-load A/B (4 concurrent 100-token generations, `slots =
    /// 4` either way) measured it consistently slower batched than not.
    /// Likely cause: fusing *M* sequences' matmuls into shared dispatches
    /// amortizes weight bandwidth, but the GPU is fast enough per
    /// single-sequence step that the extra
    /// synchronization needed to chain *M* independent sequences into one
    /// encoder — and the coordinator's own up-to-`MAX_BATCH_WAIT`
    /// rendezvous wait before a batch can even start — costs more than
    /// the amortization saves. Left available behind the flag,
    /// correctness-verified, for hardware or batch sizes where that
    /// balance tips the other way, rather than deleted.
    ///
    /// Getting a trustworthy measurement here required fixing a real bug
    /// first: both this batched path *and* the pre-existing single-
    /// sequence GPU-resident decode path (`GemmaModel::record_decode_
    /// forward`) used to key their cached per-layer GPU
    /// buffers by weight shape alone, with no per-caller distinction.
    /// `BatchCoordinator` deliberately allows two of its own `process_
    /// batch` calls to run concurrently (see its own doc comment), and
    /// ordinary `slots > 1` decode is concurrent by construction — so two
    /// requests decoding at the same time could end up sharing the same
    /// cached buffer. Because that cache's mutex guard is only held
    /// during the cheap *recording* step, not across the deferred GPU
    /// *submission* (`queue.write_buffer` takes effect immediately, not
    /// in encoder-submission order), one request's write could silently
    /// corrupt another's not-yet-executed dispatch — no crash, just wrong
    /// tokens, on *any* `slots > 1` deployment regardless of whether
    /// `ORANGU_BATCH_DECODE` was ever set. Fixed by threading each
    /// request's own `SlotGuard::id()` through as `BatchDecodeItem::
    /// slot_id` (see its own doc comment) into every cache key, so
    /// concurrent callers never share a buffer. Verified fixed with a
    /// live reproduction: 4 concurrent identical greedy prompts, which
    /// diverged after a few tokens before the fix (in *both* the batched
    /// and non-batched configurations) and are byte-identical after it.
    pub batch_coordinator: Option<Arc<BatchCoordinator>>,
    /// Cross-request KV-cache prefix reuse (`engine::prefix_cache`) —
    /// `None` disables it entirely (same as `Some(PrefixCache::new(0))`,
    /// just without even the pool's own mutex/lookup cost). See that
    /// module's own doc comment for what it does and doesn't cover.
    pub prefix_cache: Option<Arc<PrefixCache>>,
    /// Durable per-slot KV-cache persistence (`engine::slot_store`) — `Some`
    /// by default (unless `ORANGU_NO_SLOT_SAVE` is set or the home directory
    /// can't be resolved). Backs the `POST /slots/{id}?action=save|restore` endpoints
    /// and, while live, lets a slot's own retained cache serve as a prefix-
    /// reuse source for the next request on that slot (independent of the
    /// cross-slot `prefix_cache` pool). See that module's own doc comment.
    pub slot_store: Option<Arc<super::slot_store::SlotStore>>,
    /// Which of `--all`/`--code`/`--review`/`--explorer`/`--embedding` this
    /// deployment was started with — read by the HTTP layer for default
    /// sampling parameters, generation-endpoint gating, and (`Review`
    /// only) reasoning suppression. See `config::Role`'s own doc comment.
    pub role: crate::config::Role,
    /// Latency distributions and totals for `/metrics` — see
    /// `engine::metrics`. Always present: an unscraped deployment pays a
    /// handful of relaxed atomic adds per request and one per token, which is
    /// nothing beside a forward pass, and a metric that only exists when
    /// something is configured is one nobody can ask for after the fact.
    pub metrics: Arc<super::metrics::ServerMetrics>,
}

/// What a caught generation panic is reported to the caller as: the panic's
/// own captured `detail` (message, location, backtrace — see
/// `crate::panic_capture`), except when the GPU device has been lost.
///
/// A lost device is the one panic whose detail tells the caller nothing:
/// the backtrace names whichever `wgpu` readback happened to be in flight
/// when the driver reset the device, which is about the driver, not about
/// their request — and the process is already on its way out
/// (`crate::device_lost`), so what they actually need to know is that
/// retrying in a moment will work. The full detail is not lost: it is in
/// this server's own log, written by `device_lost::fail` before the unwind
/// even started.
fn panic_report(detail: String, device_lost: bool) -> String {
    if device_lost {
        crate::device_lost::CLIENT_MESSAGE.to_string()
    } else {
        detail
    }
}

impl Engine {
    /// Starts generating in the background (on tokio's blocking pool) and
    /// returns a channel of [`StreamEvent`]s — waits for a free slot first
    /// if every one is already busy.
    pub async fn generate(&self, req: GenerateRequest) -> UnboundedReceiver<StreamEvent> {
        let (tx, rx) = mpsc::unbounded_channel();
        // A request that arrives in the short window between the GPU going
        // away and this process exiting (`device_lost`) is answered with
        // the same one sentence the request that hit the loss got, rather
        // than being started, run into the same dead device, and answered
        // with a second panic's worth of detail.
        if crate::device_lost::is_lost() {
            let _ = tx.send(StreamEvent::Error(
                crate::device_lost::CLIENT_MESSAGE.to_string(),
            ));
            return rx;
        }
        let model = self.model.clone();
        let tokenizer = self.tokenizer.clone();
        let slots = self.slots.clone();
        let draft = self.draft.clone();
        let batch_coordinator = self.batch_coordinator.clone();
        let prefix_cache = self.prefix_cache.clone();
        let slot_store = self.slot_store.clone();
        // Whether a reasoning message reaches the client is this server's
        // role's call, not the request's — the same place every other
        // reasoning decision is made (`Role::enable_thinking`, which the
        // HTTP layers pass to the chat template).
        let role = self.role;
        let metrics = self.metrics.clone();
        // Before the spawn, not inside it: what an operator means by "how long
        // did this request take" starts when the request arrives, and a task
        // that has not been scheduled yet is already waiting.
        let arrived = Instant::now();

        let id_slot = req.id_slot;
        tokio::spawn(async move {
            let guard = match id_slot {
                // A pinned request is waiting for one specific slot's warm
                // cache and is not competing for admission, so the queue limit
                // does not apply to it — the same reason it bypasses the
                // ticket queue entirely.
                Some(index) => slots.acquire_slot(index).await,
                None => match slots.try_acquire().await {
                    Some(guard) => guard,
                    None => {
                        // Refused, not queued. The caller gets an answer in
                        // microseconds instead of joining a pile — see
                        // `SlotPool::with_queue_limit`.
                        metrics.observe_refusal();
                        let _ = tx.send(StreamEvent::Overloaded);
                        return;
                    }
                },
            };
            // Only for a request that actually queued. A pinned request waits
            // for one named slot's warm cache rather than for capacity, so
            // folding its wait in here would make cache affinity read as
            // server overload.
            if id_slot.is_none() {
                metrics.observe_queue_wait(arrived.elapsed());
            }
            let task_tx = tx.clone();
            let result = tokio::task::spawn_blocking(move || {
                // `catch_unwind` here (not left to `spawn_blocking`'s own
                // panic-to-`JoinError` conversion below) so a panic's real
                // detail can be recovered at all: this closure runs to
                // completion on the *same* blocking-pool thread the panic
                // hook (`crate::panic_capture`) just stashed its message/
                // backtrace on, so `take_last_panic_detail` can only read
                // it back correctly from right here — by the time this
                // propagated out as a `JoinError` on a different
                // (async-runtime) thread, there would be no way to
                // associate that stash with this specific panic at all.
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run(
                        model.as_ref(),
                        tokenizer.as_ref(),
                        draft.as_deref(),
                        batch_coordinator.as_deref(),
                        prefix_cache.as_deref(),
                        slot_store.as_deref(),
                        &guard,
                        req,
                        role,
                        &metrics,
                        arrived,
                        task_tx.clone(),
                    )
                }));
                if let Err(_panic) = result {
                    let detail =
                        crate::panic_capture::take_last_panic_detail().unwrap_or_else(|| {
                            "generation task panicked (no detail captured)".to_string()
                        });
                    let message = panic_report(detail, crate::device_lost::is_lost());
                    let _ = task_tx.send(StreamEvent::Error(message));
                }
            })
            .await;
            if let Err(join_err) = result {
                // `spawn_blocking` itself failed *without* the closure
                // above panicking (e.g. the task was cancelled) — the
                // panic case is already handled and reported from inside
                // the closure, so this is only ever the non-panic
                // fallback now.
                let _ = tx.send(StreamEvent::Error(format!(
                    "generation task failed: {join_err}"
                )));
            }
        });

        rx
    }
}

/// Records exactly one outcome for a request, however it leaves [`run`].
///
/// A `Drop` rather than a call at each exit, because `run` has five places
/// that report an error and return, one that returns on a client disconnect,
/// and a `catch_unwind` above it that turns a panic into a reply — and an
/// outcome counter with a path that forgets to increment it is worse than no
/// counter, since the total silently stops matching the request count.
/// Defaulting to `Error` and being told otherwise means a new failure path is
/// counted correctly without anyone noticing it needed to be.
struct OutcomeGuard<'a> {
    metrics: &'a super::metrics::ServerMetrics,
    arrived: Instant,
    /// `None` until something says how this ended.
    finished: Option<(super::metrics::Outcome, usize, usize, usize)>,
}

impl OutcomeGuard<'_> {
    fn finish(
        &mut self,
        outcome: super::metrics::Outcome,
        prompt: usize,
        cached: usize,
        generated: usize,
    ) {
        self.finished = Some((outcome, prompt, cached, generated));
    }
}

impl Drop for OutcomeGuard<'_> {
    fn drop(&mut self) {
        // A request that failed contributes to the error counter but not to
        // the token totals: those count work *delivered*, and a rate taken
        // over them should not move because something broke.
        let (outcome, prompt, cached, generated) =
            self.finished
                .take()
                .unwrap_or((super::metrics::Outcome::Error, 0, 0, 0));
        self.metrics
            .observe_request(self.arrived.elapsed(), outcome, prompt, cached, generated);
    }
}

#[allow(clippy::too_many_arguments)]
fn run(
    model: &dyn ModelForward,
    tokenizer: &Tokenizer,
    draft: Option<&DraftModel>,
    batch_coordinator: Option<&BatchCoordinator>,
    prefix_cache: Option<&PrefixCache>,
    slot_store: Option<&super::slot_store::SlotStore>,
    guard: &super::scheduler::SlotGuard,
    req: GenerateRequest,
    role: crate::config::Role,
    metrics: &super::metrics::ServerMetrics,
    arrived: Instant,
    tx: mpsc::UnboundedSender<StreamEvent>,
) -> Result<()> {
    let mut outcome = OutcomeGuard {
        metrics,
        arrived,
        finished: None,
    };
    let config = model.config();
    let capacity = (req.prompt_tokens.len() + req.max_tokens).min(config.n_ctx_train.max(1));
    if req.prompt_tokens.len() > capacity {
        let _ = tx.send(StreamEvent::Error(format!(
            "prompt ({} tokens) exceeds the model's context length ({})",
            req.prompt_tokens.len(),
            config.n_ctx_train
        )));
        return Ok(());
    }

    guard.set_prompt_tokens(req.prompt_tokens.len());
    // Reuse a previous request's already-computed KV cache for however
    // much of this prompt matches one — see `engine::prefix_cache`'s own
    // doc comment. Always allocate this request's own cache fresh, at its
    // own `capacity` (never reused directly: two requests' capacities can
    // differ), then copy the matched prefix into it — `reused_len` tokens'
    // worth of the prompt never need a forward pass at all. Left at 1
    // fewer than the full matched length whenever it would otherwise equal
    // this prompt's own length, so there's always at least one real
    // forward call to produce fresh logits for the first sampled token
    // from (this only matters for the degenerate case of re-sending a
    // prompt identical to one already fully cached).
    //
    // `cache_prompt: false` skips both reuse paths below, so the prompt is
    // prefilled in full. Storing this request's own cache afterwards is
    // unaffected — the flag governs what this request may *read*.
    let mut new_cache = model.new_kv_cache(capacity);
    let mut reused_len = 0usize;
    if req.cache_prompt
        && let Some(pool) = prefix_cache
        && let Some((matched, entry)) = pool.take_best_match(&req.prompt_tokens)
    {
        let matched = matched.min(req.prompt_tokens.len().saturating_sub(1));
        if matched > 0 {
            // Moved, not copied: `take_best_match` removed this entry from the
            // pool, so nothing else can still be reading it and its buffers
            // can become this request's own. The slot store below cannot do
            // the same — it keeps its snapshot for the slot's next request.
            new_cache.adopt_prefix(entry.cache, matched);
            reused_len = matched;
        }
    }
    // The cross-slot pool (above) has first claim; only if it found nothing
    // does this slot's own durably-retained cache (`engine::slot_store`, the
    // source a `restore` populated) get consulted — it applies the same
    // leave-one-token and recurrent-state rules internally.
    if req.cache_prompt
        && reused_len == 0
        && let Some(store) = slot_store
    {
        reused_len = store.reuse_into(guard.id(), &req.prompt_tokens, &mut new_cache);
    }
    // `Option` (not a plain `KvCache`) so the decode loop can *move* it
    // into a `BatchDecodeRequest` when a `batch_coordinator` is in use —
    // that call crosses to a different thread (whichever one ends up
    // leading this batch), which needs ownership, not a borrow. `.take()`/
    // reassignment stands in for a borrow everywhere else, at zero real
    // cost (this is never actually `None` except mid-swap).
    let mut cache = Some(new_cache);
    let mut sampler = Sampler::new(req.sampling);
    if req.json_output {
        // The byte table is built on the tokenizer's first constrained
        // request and shared from then on, so an unconstrained deployment
        // never pays for it.
        sampler = sampler.with_constraint(crate::engine::sampling::Constraint::json(
            tokenizer.token_bytes_shared(),
            req.stop_token_ids.clone(),
        ));
    }
    let mut history = req.prompt_tokens.clone();

    let prompt_start = Instant::now();
    let total_prompt = req.prompt_tokens.len();
    let progress_tx = tx.clone();
    let mut on_chunk = |processed: usize| {
        // Cached tokens never went through a forward pass, so they are
        // already "processed" as far as a progress bar is concerned —
        // otherwise a mostly-cached prompt appears to start at zero and jump.
        let _ = progress_tx.send(StreamEvent::PromptProgress {
            total: total_prompt,
            cached: reused_len,
            processed: reused_len + processed,
            elapsed: prompt_start.elapsed(),
        });
    };
    let logits = match prefill(
        model,
        cache.as_mut().expect("cache is always Some here"),
        &req.prompt_tokens[reused_len..],
        reused_len,
        guard.id(),
        &mut on_chunk,
    ) {
        Ok(l) => l,
        Err(err) => {
            // `{err:?}` (anyhow's chain-plus-backtrace Debug format, not
            // `{err}`'s bare top-level message) — `main`'s own unconditional
            // `RUST_BACKTRACE=1` means this always includes a captured
            // backtrace, not just whatever `.context()` calls happened to
            // add, matching the detail a panic's own captured backtrace
            // (`panic_capture`) gives for a debug report worth saving.
            let _ = tx.send(StreamEvent::Error(format!("{err:?}")));
            return Ok(());
        }
    };
    let prompt_time = prompt_start.elapsed();
    // Prefill is never decode-shaped (`n_tokens > 1`), so it never takes a
    // GPU-fused sampling fast path either way — this first sample always
    // runs the plain CPU chain.
    let mut next = sampler.sample(&logits, &history);

    let generate_start = Instant::now();
    let mut generated = 0usize;
    let finish_reason;
    let mut last_report = Instant::now();
    let mut reported = false;
    // Speculative decoding, greedy-only, and not while fused batching is
    // running the decode step. `spec_buf` holds tokens a speculative step
    // already verified and committed to the KV cache, waiting to be emitted
    // before the next forward.
    //
    // Not under a constraint: speculation drafts several tokens and accepts
    // them where they match what greedy decoding *would* have produced, and
    // greedy decoding knows nothing about the grammar. A drafted token the
    // constraint forbids would be accepted on that comparison alone.
    let may_speculate =
        sampler.is_greedy() && !sampler.is_constrained() && batch_coordinator.is_none();
    // A configured draft model wins over prompt-lookup rather than being
    // combined with it. Both are guesses at the same tokens, and running the
    // free one first would only turn its misses into a second, wasted
    // verification forward — the drafter that is always available is the one
    // worth having when there is one.
    let mut drafter = match (may_speculate, draft, speculative_config()) {
        (false, _, _) => None,
        (true, Some(draft), _) => Some(Drafter::Model {
            model: draft.model.as_ref(),
            tokens: draft.tokens,
            cache: draft.model.new_kv_cache(capacity),
            committed: 0,
        }),
        (true, None, Some((ngram, max_draft))) => Some(Drafter::PromptLookup { ngram, max_draft }),
        (true, None, None) => None,
    };
    let mut spec_buf: VecDeque<u32> = VecDeque::new();
    let mut spec_accepted = 0usize;
    let mut spec_steps = 0usize;
    let mut header = MessageHeader::for_prompt(tokenizer, &req.prompt_tokens, role);
    // When the previous token was produced, so the gap to the next one can be
    // observed. `None` until there is a previous one to measure from.
    let mut last_token_at: Option<Instant> = None;
    loop {
        if generated >= req.max_tokens {
            finish_reason = FinishReason::Length;
            break;
        }
        if req.stop_token_ids.contains(&next) {
            finish_reason = FinishReason::Stop;
            break;
        }
        history.push(next);
        generated += 1;
        guard.set_generated_tokens(generated);
        // Time to first token counts from arrival, so it carries the queue
        // wait and the prefill together — which is what an interactive caller
        // actually waits through. Every later token contributes a gap.
        //
        // The first token *produced*, not the first the client sees: a chat
        // format's structural prefix is filtered out of the stream
        // (`MessageHeader`), and charging the template's shape to the
        // server's latency would make two models with the same speed report
        // different numbers.
        let now = Instant::now();
        match last_token_at {
            None => metrics.observe_first_token(now.duration_since(arrived)),
            Some(previous) => metrics.observe_inter_token(now.duration_since(previous)),
        }
        last_token_at = Some(now);
        // Structural tokens (a chat format's turn/channel/tool markers, and
        // any stray BOS/EOS) still go into `history` so the KV cache and any
        // continued generation stay correct, but are not rendered to the
        // user — otherwise a gemma-4 reply spills literal `<turn|>`/
        // `<channel|>` tokens into the stream (`skip_special_tokens`).
        let emitted = if tokenizer.is_special(next) {
            // A message marker opens or closes a header; it is hidden
            // either way, like every other structural token, but ending one
            // can call for a separator between two visible messages.
            match header.observe_marker(next) {
                Some(separator) => Some(separator.to_string()),
                // Almost every special token is structural and stays hidden.
                // The exception is a tool-call marker: the model writes its
                // call *between* special tokens, so suppressing them would
                // leave the arguments as loose prose with nothing to say
                // they were a call. Rendered back to literal text here and
                // read by `engine::tool_calls`; the HTTP layer removes the
                // whole span from what the user sees, so this never reaches
                // a chat client as content.
                None => tokenizer
                    .token_text(next)
                    .and_then(tool_calls::marker_text)
                    .map(str::to_string),
            }
        } else {
            // Suppressed while this is a message *header* (the recipient the
            // model wrote for the format's benefit, not the reader's), and
            // while it is a message body the role hides.
            header.observe_text(tokenizer.decode(&[next]))
        };
        if let Some(text) = emitted {
            let _ = tx.send(StreamEvent::Token(text));
        }
        // Whether anyone is still listening, asked **every token** rather than
        // only when there was text to send.
        //
        // The check used to ride on the send, which meant a token rendering to
        // nothing never made it. That is not a rare case: a chat format's
        // structural markers are suppressed, and a whole reasoning body is
        // suppressed under `--review`, so a request could generate to
        // `max_tokens` — holding the slot every other request is queued for —
        // for a client that hung up at the first token. Measured on a model
        // whose output was almost entirely suppressed: a disconnected 2000-token
        // request ran to completion; on one with visible text the same request
        // stopped within a token.
        //
        // Its own outcome, not an error: nothing went wrong, and the tokens
        // generated so far were real work a rate should see.
        if tx.is_closed() {
            outcome.finish(
                super::metrics::Outcome::Cancelled,
                req.prompt_tokens.len(),
                reused_len,
                generated,
            );
            return Ok(());
        }
        // Not after the first token: it was sampled from the prefill's own
        // logits, so `generate_time` at that point is a few microseconds of
        // bookkeeping and the "rate" comes out in the tens of thousands of
        // tokens per second. A rate needs an interval between two tokens
        // before it means anything.
        if req.timings_per_token && generated >= 2 {
            let _ = tx.send(StreamEvent::Timings(GenerateStats {
                prompt_tokens: req.prompt_tokens.len(),
                cached_tokens: reused_len,
                prompt_time,
                generated_tokens: generated,
                generate_time: generate_start.elapsed(),
            }));
        }
        if last_report.elapsed() >= Duration::from_secs(1) {
            let partial = GenerateStats {
                prompt_tokens: req.prompt_tokens.len(),
                cached_tokens: reused_len,
                prompt_time,
                generated_tokens: generated,
                generate_time: generate_start.elapsed(),
            };
            // \x1b[K ("erase to end of line") clears any leftover tail from
            // a longer previous update before the cursor returns to the
            // start of the line — plain \r alone can't shrink a line, only
            // overwrite its prefix.
            print!(
                "\rorangu-server: [slot {}] {}\x1b[K",
                guard.id(),
                partial.log_line()
            );
            std::io::stdout().flush().ok();
            last_report = Instant::now();
            reported = true;
        }
        if history.len() >= capacity {
            finish_reason = FinishReason::Length;
            break;
        }
        // When the sampler is greedy, let the model pick the
        // next token itself (a GPU-fused argmax, for backends that have
        // one) instead of always reading back the full `[n_vocab]` logits
        // vector just to immediately re-derive the same argmax on the
        // CPU. `recent_tokens` is trimmed to `repeat_last_n` here, not
        // inside the callee, matching `engine::sampling::
        // apply_repeat_penalty`'s own trim exactly.
        let repeat_last_n = sampler.repeat_last_n();
        let recent_start = history.len().saturating_sub(repeat_last_n);
        let start_pos = history.len() - 1;

        next = if let Some(tok) = spec_buf.pop_front() {
            // A token an earlier speculative step already verified and
            // committed to the KV cache — emit it without another forward.
            tok
        } else if let Some(drafter) = drafter.as_mut() {
            // Draft a continuation and verify it in one multi-position
            // forward; returns the model's own next token (any further
            // accepted tokens are queued in `spec_buf`).
            match speculative_next(
                model,
                cache
                    .as_mut()
                    .expect("cache is always Some between iterations"),
                &mut sampler,
                &history,
                next,
                start_pos,
                capacity,
                drafter,
                guard.id(),
                &mut spec_buf,
                &mut spec_accepted,
                &mut spec_steps,
            ) {
                Ok(t) => t,
                Err(err) => {
                    let _ = tx.send(StreamEvent::Error(format!("{err:?}")));
                    return Ok(());
                }
            }
        } else if let Some(coordinator) = batch_coordinator {
            // Submit this decode step to the shared coordinator instead of
            // calling `forward_maybe_sampling` directly, so it can be fused
            // with whatever other sequences submit their own next step
            // within the same short window.
            let request = BatchDecodeRequest {
                cache: cache
                    .take()
                    .expect("cache is always Some between iterations"),
                token: next,
                start_pos,
                greedy_sample: (sampler.is_greedy() && !sampler.is_constrained()).then(|| {
                    OwnedGreedySample {
                        recent_tokens: history[recent_start..].to_vec(),
                        repeat_penalty: sampler.repeat_penalty(),
                    }
                }),
                slot_id: guard.id(),
            };
            let response = coordinator.submit(model, request);
            cache = Some(response.cache);
            match response.outcome {
                Ok(ForwardOutcome::Token(t)) => t,
                Ok(ForwardOutcome::Logits(l)) => sampler.sample(&l, &history),
                Err(err) => {
                    let _ = tx.send(StreamEvent::Error(err));
                    return Ok(());
                }
            }
        } else {
            // The device-side argmax picks a token without ever consulting
            // the grammar, so a constrained request has to come back to the
            // CPU sampler for every step. It costs the fast path; the
            // alternative is a constraint that silently does not apply.
            let greedy_sample =
                (sampler.is_greedy() && !sampler.is_constrained()).then(|| GreedySampleParams {
                    recent_tokens: &history[recent_start..],
                    repeat_penalty: sampler.repeat_penalty(),
                });
            // GPU submissions for this one decode step. `gemma.rs` had this
            // instrumentation privately; it belongs here, because the number it
            // reports is *the* difference between the two decode paths and
            // cannot be compared across architectures from inside one of them.
            // A decode step that is one submission lets concurrent requests
            // interleave on the GPU; one that is two hundred does not, and
            // `PERF-GAP.md` G3 measures that as a 2x aggregate-throughput
            // ceiling. Costs a cached env read and one atomic load when the
            // flag is on, nothing when it is off.
            let submissions_before = gpu_trace()
                .then(|| model.vulkan_backend())
                .flatten()
                .map(|v| v.submission_count());
            let outcome = model.forward_maybe_sampling(
                cache
                    .as_mut()
                    .expect("cache is always Some between iterations"),
                &[next],
                start_pos,
                greedy_sample,
                guard.id(),
            );
            if let Some(before) = submissions_before
                && let Some(v) = model.vulkan_backend()
            {
                eprintln!(
                    "orangu-server: [gpu-trace] {} GPU submissions for this decode step (pos {start_pos})",
                    v.submission_count() - before
                );
            }
            match outcome {
                Ok(ForwardOutcome::Token(t)) => t,
                Ok(ForwardOutcome::Logits(l)) => sampler.sample(&l, &history),
                Err(err) => {
                    let _ = tx.send(StreamEvent::Error(format!("{err:?}")));
                    return Ok(());
                }
            }
        };
    }
    let generate_time = generate_start.elapsed();
    if spec_steps > 0 {
        // Draft acceptance: `spec_accepted` drafted tokens confirmed across
        // `spec_steps` forwards, i.e. this many extra tokens produced beyond the
        // one each forward always yields — the whole payoff of speculation.
        // The drafter is named because the two have completely different cost
        // profiles: prompt-lookup's misses are free, a draft model's are a
        // forward pass each, so the same acceptance figure is a win for one
        // and a loss for the other.
        eprintln!(
            "orangu-server: [speculative/{}] {spec_accepted} drafted tokens accepted over \
             {spec_steps} steps ({:.2} extra tokens/forward)",
            drafter.as_ref().map_or("none", Drafter::label),
            spec_accepted as f64 / spec_steps as f64
        );
    }

    // Offer this request's own final (full token sequence, resulting KV
    // cache) to the pool for a later request to reuse — win or not this
    // time, it's a candidate prefix for whatever comes next (most
    // obviously the same conversation's following turn, whose prompt will
    // be exactly `history` plus a short new suffix). The same completed
    // cache is also retained as this slot's durable snapshot (`slot_store`)
    // so a later `save` can persist it; when both features are on, the pool
    // gets a `duplicate()` and the slot keeps the original, since each needs
    // to own its copy.
    if let Some(final_cache) = cache.take() {
        let history = std::mem::take(&mut history);
        match (prefix_cache, slot_store) {
            (Some(pool), Some(store)) => {
                store.retain(guard.id(), history.clone(), final_cache.duplicate());
                pool.store(history, final_cache);
            }
            (Some(pool), None) => pool.store(history, final_cache),
            (None, Some(store)) => store.retain(guard.id(), history, final_cache),
            (None, None) => {}
        }
    }

    let stats = GenerateStats {
        prompt_tokens: req.prompt_tokens.len(),
        cached_tokens: reused_len,
        prompt_time,
        generated_tokens: generated,
        generate_time,
    };
    // The trailing \r + \x1b[K only matter if a live update above already
    // moved the cursor onto this line; harmless (a no-op) otherwise.
    let prefix = if reported { "\r" } else { "" };
    println!(
        "{prefix}orangu-server: [slot {}] {}\x1b[K",
        guard.id(),
        stats.log_line()
    );
    outcome.finish(
        match finish_reason {
            FinishReason::Stop => super::metrics::Outcome::Stop,
            FinishReason::Length => super::metrics::Outcome::Length,
        },
        stats.prompt_tokens,
        stats.cached_tokens,
        stats.generated_tokens,
    );
    let _ = tx.send(StreamEvent::Done {
        stats,
        finish_reason,
    });
    Ok(())
}

/// The recipient text that marks a message as the model's own reasoning
/// rather than something addressed to the caller.
///
/// `muse-glimmer` writes `<|start|>assistant to=self<|message|>` for a
/// chain-of-thought message and ` to=user` for the answer; the gpt-oss
/// family spells the same distinction with a channel name. Matching on the
/// text the model writes, the way `Tokenizer::message_framing` matches on
/// the vocabulary's own token names — a format that doesn't use it simply
/// never produces a header containing it.
const REASONING_RECIPIENT: &str = "to=self";

/// How much text may be withheld as a message header before
/// [`MessageHeader`] concludes it is not looking at one. A recipient is a
/// role plus a name (` to=self`, ` to=weather.get_current`); this is far
/// above any of them and far below a reply worth losing.
const MAX_HEADER_LEN: usize = 128;

/// Splits a `<|start|>…<|message|>`-framed reply into its parts as it
/// streams: the message *headers*, which are framing and never shown, and
/// the message *bodies*, which are shown or not depending on who the model
/// addressed them to.
///
/// The markers themselves are CONTROL tokens and were always hidden. Two
/// things are new:
///
/// - **The header text is hidden.** `muse-glimmer` stops its generation
///   prompt at `<|start|>assistant` so the model can pick its own
///   recipient, and the ` to=self` / ` to=user` it writes there is framing,
///   not prose. Left visible, every reply from that model began
///   `" to=self"`.
/// - **A reasoning message is hidden when the role says so.** An assistant
///   turn in this format is *several* messages — reasoning addressed
///   `to=self`, then the answer addressed `to=user` — and which of them the
///   caller sees is the same question `Role::enable_thinking` already
///   answers for models that mark reasoning with `<think>`. A role that
///   suppresses reasoning gets the answer alone; every other role gets
///   both, separated by a blank line (they are different messages, and
///   running them together produced `…Final.Three primes larger than…`).
///
/// A second format asks the same question with tokens instead of text, and
/// is answered here too: `inkling` opens each body with a marker naming its
/// *kind* (`<|content_thinking|>` for reasoning, `<|content_text|>` for the
/// answer) and writes no header at all. Same rule — a reasoning body is
/// hidden when the role says so, and two visible bodies are separated by a
/// blank line — read off `Tokenizer::content_kinds` rather than off a
/// recipient string.
///
/// Inert (`framing` and `kinds` both `None`) for every vocabulary with
/// neither, which is every other model this server serves — those keep
/// byte-for-byte the behavior they had.
struct MessageHeader {
    /// `(<|start|>, <|message|>)`, or `None` when this vocabulary has no
    /// such framing and nothing here applies.
    framing: Option<(u32, u32)>,
    /// The body-kind markers, for a vocabulary that types its bodies
    /// instead of naming a recipient. `None` for every other.
    kinds: Option<crate::engine::tokenizer::ContentKinds>,
    /// Whether the stream is currently inside a header.
    inside: bool,
    /// The current header's text so far, accumulated to be read once at
    /// `<|message|>` and then discarded. Bounded by the header's own length
    /// (a recipient name), not by the reply's.
    recipient: String,
    /// Whether the message body now streaming is suppressed.
    hidden_body: bool,
    /// Whether any *visible* body has already been emitted this turn —
    /// what decides if the next one needs a separator in front of it.
    emitted_body: bool,
    suppress_reasoning: bool,
}

impl MessageHeader {
    fn for_prompt(tokenizer: &Tokenizer, prompt_tokens: &[u32], role: crate::config::Role) -> Self {
        Self {
            framing: tokenizer.message_framing(),
            kinds: tokenizer.content_kinds(),
            inside: Self::prompt_ends_in_header(tokenizer, prompt_tokens),
            recipient: String::new(),
            hidden_body: false,
            emitted_body: false,
            suppress_reasoning: role.suppresses_reasoning(),
        }
    }

    /// Whether generation resumes *inside* a header.
    ///
    /// Read off the prompt rather than taken as a flag from the caller: the
    /// prompt is the ground truth, every caller renders one, and a flag is
    /// one more thing three HTTP paths would have to keep consistent.
    ///
    /// A prompt whose last marker is `<|start|>` (the ordinary generation
    /// prompt, `…<|eot|><|start|>assistant`) resumes inside a header. One
    /// whose last marker is `<|message|>` — a prefilled partial reply —
    /// resumes inside a body. One with neither, which includes every
    /// raw-completion prompt, likewise resumes in a body.
    fn prompt_ends_in_header(tokenizer: &Tokenizer, prompt_tokens: &[u32]) -> bool {
        tokenizer.message_framing().is_some_and(|(start, message)| {
            prompt_tokens
                .iter()
                .rev()
                .find_map(|&t| (t == start || t == message).then_some(t == start))
                .unwrap_or(false)
        })
    }

    /// Feed every *special* token the model generates through this.
    /// Returns text to emit in the marker's place — a separator between two
    /// visible messages, and otherwise nothing.
    fn observe_marker(&mut self, id: u32) -> Option<&'static str> {
        // A body-kind marker settles the same question a header's recipient
        // does, and settles it on its own — a format that has these writes
        // no header, so this is checked first and returns rather than
        // falling through to the header machinery.
        if let Some(kinds) = &self.kinds
            && (id == kinds.reasoning || kinds.other.contains(&id))
        {
            self.hidden_body = self.suppress_reasoning && id == kinds.reasoning;
            let separator = (!self.hidden_body && self.emitted_body).then_some("\n\n");
            self.emitted_body |= !self.hidden_body;
            return separator;
        }
        let (start, message) = self.framing?;
        if id == start {
            self.inside = true;
            self.recipient.clear();
            return None;
        }
        if id != message {
            return None;
        }
        // The header just ended: its text says who this message is for.
        self.inside = false;
        self.hidden_body = self.suppress_reasoning
            && self
                .recipient
                .replace(' ', "")
                .contains(REASONING_RECIPIENT);
        let separator = (!self.hidden_body && self.emitted_body).then_some("\n\n");
        self.emitted_body |= !self.hidden_body;
        separator
    }

    /// Feed every *ordinary* (non-special) token's text through this.
    /// Returns what to emit: nothing while inside a header, and nothing
    /// inside a body the role suppresses.
    fn observe_text(&mut self, text: String) -> Option<String> {
        if !self.inside {
            return (!self.hidden_body).then_some(text);
        }
        self.recipient.push_str(&text);
        if self.recipient.len() <= MAX_HEADER_LEN {
            return None;
        }
        // Long past any recipient's length and still no `<|message|>`: this
        // is not a header, so whatever was withheld is content and is
        // released now. Withholding is the one failure worth bounding —
        // a stray recipient in the reply is cosmetic, a reply swallowed
        // whole is not, and only the second can happen without a bound.
        self.inside = false;
        self.hidden_body = false;
        self.emitted_body = true;
        Some(std::mem::take(&mut self.recipient))
    }
}

/// Prompt-lookup speculative-decode settings, read once at the start of a
/// request. `None` (the default) leaves decoding exactly as it was. Setting
/// `ORANGU_SPECULATIVE` turns it on; `ORANGU_SPEC_NGRAM` (default 2) is how
/// many trailing tokens must match a earlier spot in the context to trigger a
/// draft, and `ORANGU_SPEC_DRAFT` (default 4) how many tokens to draft from
/// there. See `Self::speculative_next`.
/// How many prompt tokens go into one forward pass — `ORANGU_PREFILL_BATCH`,
/// default [`PREFILL_BATCH_DEFAULT`]. `0` means no limit: the whole prompt in
/// one pass, which is what this code did unconditionally before.
fn prefill_batch() -> usize {
    static BATCH: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *BATCH.get_or_init(|| {
        std::env::var("ORANGU_PREFILL_BATCH")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(PREFILL_BATCH_DEFAULT)
    })
}

/// The default prompt-chunk size.
///
/// A forward pass's work and scratch both scale with the tokens in it, and
/// handing the model a whole long prompt at once therefore sizes them by
/// however much text the caller happened to send. On a GPU that is already
/// short of memory, that is a promise nothing can keep — and the way it
/// breaks is not a clean allocation failure but a **device reset**: under
/// paging pressure a single large submission stops finishing inside
/// `amdgpu`'s ~10s ring timeout, the kernel resets the ring and names this
/// process as the guilty context (`ring gfx_0.0.0 timeout … Ring gfx_0.0.0
/// reset succeeded`), and every buffer on the lost device dies with it (see
/// `crate::device_lost`).
///
/// Measured on a 4 GiB RX 5500M holding 2.5 GiB of weights, with a
/// 17.5k-token prompt — the size an `orangu` TUI turn reaches once it
/// carries a workspace diff:
///
/// | chunk | outcome |
/// | :-- | :-- |
/// | whole prompt at once | device lost after 21s |
/// | 2048 | device lost after 3m54s |
/// | 512 | **completed in 4m13s** |
///
/// Peak VRAM was the same (3.67 GiB of 4.08) in every case, which is the
/// point: what changes is how long any *one* submission holds the ring
/// while the driver pages, not how much is resident. So the chunk is a
/// timeout budget, not a memory budget, and 512 is the value that kept
/// every submission comfortably inside it.
///
/// It is not a throughput tax either — at an 8k-token prompt on the same
/// card, 512 measured *faster* than no chunking at all (115.4 vs 105.5
/// tok/s prefill), since a smaller working set pages less. The matmuls were
/// already striped at
/// `engine::backend::vulkan`'s own per-phase submission cap regardless of
/// this, so what a bigger chunk buys is fewer attention and PLE calls, not
/// bigger GEMMs.
///
/// Prompts shorter than this are unaffected: one chunk *is* the old path.
/// `ORANGU_PREFILL_BATCH=0` restores it outright, and a machine with VRAM
/// to spare can raise it — nothing here is specific to one card beyond the
/// numbers that set it.
const PREFILL_BATCH_DEFAULT: usize = 512;

/// Runs `tokens` through the model as consecutive [`prefill_batch`]-sized
/// forward passes, returning the last one's logits — the only ones a caller
/// wants, since sampling continues from the end of the prompt.
///
/// Each chunk is fed at the position the previous chunk ended at, so the KV
/// cache accumulates exactly as it would from one pass over the whole
/// prompt: this is the same `(cache, tokens, start_pos)` shape the prefix
/// cache already uses to resume a prompt it has partly seen, not a new
/// contract with the architectures.
fn prefill(
    model: &dyn ModelForward,
    cache: &mut KvCache,
    tokens: &[u32],
    start_pos: usize,
    slot_id: usize,
    on_chunk: &mut dyn FnMut(usize),
) -> Result<Vec<f32>> {
    prefill_in_chunks(
        model,
        cache,
        tokens,
        start_pos,
        slot_id,
        prefill_batch(),
        on_chunk,
    )
}

/// [`prefill`] with the chunk size passed in rather than read from the
/// environment, so the splitting itself is testable — the property that
/// matters (every token fed exactly once, in order, at the right position)
/// is the one that silently corrupts a KV cache when it's wrong.
fn prefill_in_chunks(
    model: &dyn ModelForward,
    cache: &mut KvCache,
    tokens: &[u32],
    start_pos: usize,
    slot_id: usize,
    batch: usize,
    on_chunk: &mut dyn FnMut(usize),
) -> Result<Vec<f32>> {
    // The one shape that needs no bounding: a prompt that fits in a single
    // chunk *and* starts at position zero is the least work a prefill can be.
    // `batch == 0` is an explicit opt-out — see [`prefill_batch`].
    if batch == 0 || (tokens.len() <= batch && start_pos == 0) {
        let logits = model.forward(cache, tokens, start_pos, slot_id)?;
        on_chunk(tokens.len());
        return Ok(logits);
    }

    let budget = prefill_chunk_budget();
    let mut logits = Vec::new();
    let mut pos = start_pos;
    let mut done = 0usize;
    // Start with a probe rather than a full-width chunk. Nothing here knows
    // this machine's cost curve, and a full-width chunk at a deep position is
    // exactly the submission that hangs the GPU; a probe is cheap at any
    // depth and turns the next choice into arithmetic instead of a guess.
    let mut width = PREFILL_PROBE_TOKENS.min(batch);
    while done < tokens.len() {
        let n = width.min(tokens.len() - done);
        let started = Instant::now();
        logits = model.forward(cache, &tokens[done..done + n], pos, slot_id)?;
        let elapsed = started.elapsed();
        pos += n;
        done += n;
        // After the forward, not before: progress means work finished.
        on_chunk(done);
        width = next_chunk_width(n, elapsed, budget, batch);
    }
    Ok(logits)
}

/// How wide the next chunk should be, given how long a chunk of `n` tokens
/// just took.
///
/// Cost per token is not constant: a prefill chunk attends over everything
/// before it, so the same 512 tokens that take 2.3 s at position 0 take 10.1 s
/// at position 6 656 — past the ~10 s the graphics driver allows a single
/// submission before it resets the device. Sizing by token count alone, as
/// `ORANGU_PREFILL_BATCH` did on its own, therefore stops protecting anything
/// once a prompt gets long: the *count* is bounded but the *work* is not.
///
/// Scaling the width by the rate just measured keeps the work per submission
/// roughly constant instead. The rate only drifts upward, and slowly, so
/// sizing from the previous chunk lands close; the budget is set well under
/// the driver's limit to absorb what it misses, and the estimate itself aims
/// below that budget again because the *next* chunk always starts deeper into
/// the prompt than the one just timed.
fn next_chunk_width(n: usize, elapsed: Duration, budget: Duration, max_width: usize) -> usize {
    let per_token = elapsed.as_secs_f64() / (n.max(1) as f64);
    if per_token <= 0.0 {
        return max_width;
    }
    let fits = (budget.as_secs_f64() * PREFILL_BUDGET_HEADROOM / per_token) as usize;
    // `min(max_width)` on the floor as well: a caller that configured a batch
    // smaller than the floor asked for chunks that small, and `clamp` panics
    // outright when its own min exceeds its max.
    fits.clamp(PREFILL_MIN_CHUNK_TOKENS.min(max_width), max_width)
}

/// Only a fraction of the configured budget is spent when sizing the next
/// chunk. The measured chunk just completed at a shallower position than the
/// next one will start from, so using the full budget as the target leaves no
/// room for the cost curve to rise between them.
const PREFILL_BUDGET_HEADROOM: f64 = 0.75;

/// Tokens in the opening probe chunk. Small enough to stay well inside the
/// driver's limit even at the deepest context this server will accept.
const PREFILL_PROBE_TOKENS: usize = 16;

/// The floor on an adapted chunk. At a deep enough context even this exceeds
/// the budget — nothing can be done about that except take longer — but it
/// keeps the submission small enough that the driver does not give up on it.
const PREFILL_MIN_CHUNK_TOKENS: usize = 16;

/// Wall-clock target for one prefill submission — `ORANGU_PREFILL_CHUNK_MS`,
/// default [`PREFILL_CHUNK_BUDGET_MS_DEFAULT`].
///
/// This is a *timeout* budget, like `ORANGU_PREFILL_BATCH` before it, not a
/// memory one. The number that matters is the graphics driver's own limit on
/// how long one submission may run — around 10 s on the amdgpu/RADV stack this
/// was measured on, and not queryable from userspace. The default leaves room
/// for the estimate to be wrong and for the machine to be busier next time.
fn prefill_chunk_budget() -> Duration {
    static MS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    Duration::from_millis(*MS.get_or_init(|| {
        std::env::var("ORANGU_PREFILL_CHUNK_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|ms| *ms > 0)
            .unwrap_or(PREFILL_CHUNK_BUDGET_MS_DEFAULT)
    }))
}

const PREFILL_CHUNK_BUDGET_MS_DEFAULT: u64 = 3_000;

/// Whether `ORANGU_GPU_TRACE` is set, cached.
///
/// Read once — this sits in the decode loop, which runs once per token.
fn gpu_trace() -> bool {
    static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *TRACE.get_or_init(|| crate::engine::env::flag_on("ORANGU_GPU_TRACE"))
}

fn speculative_config() -> Option<(usize, usize)> {
    if std::env::var("ORANGU_SPECULATIVE").is_err() {
        return None;
    }
    let read = |name: &str, default: usize| {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(default)
    };
    Some((read("ORANGU_SPEC_NGRAM", 2), read("ORANGU_SPEC_DRAFT", 4)))
}

/// Where a speculative step's candidate tokens come from.
///
/// The two sources have opposite cost profiles, and that is the whole reason
/// both exist. Prompt-lookup calls no model at all, so a wrong guess costs
/// nothing beyond the wider verification forward — but it produces nothing
/// unless the context happens to repeat itself. A draft model always produces
/// a draft and always pays a forward pass per token for it, so it wins on
/// ordinary prose and loses whenever the pair agrees too rarely to cover the
/// drafting cost.
///
/// Verification does not care which one produced the tokens: it re-derives
/// what the served model would have said and keeps the matching prefix, so the
/// emitted text is identical either way. A drafter can only change how fast
/// the answer arrives, never what it is.
enum Drafter<'a> {
    /// Copy the continuation of an earlier occurrence of the trailing n-gram.
    PromptLookup { ngram: usize, max_draft: usize },
    /// Run a second, smaller model autoregressively.
    Model {
        model: &'a dyn ModelForward,
        tokens: usize,
        /// The draft's own KV cache, which tracks the same committed token
        /// sequence as the target's.
        cache: KvCache,
        /// How many of `history` this cache holds. Kept explicitly rather
        /// than read off the cache, because a block-compressed slot's `len`
        /// is a row count and one row can stand for several tokens
        /// (`KvCache::committed_tokens`) — a draft architecture with a stride
        /// would otherwise desynchronise silently.
        committed: usize,
    },
}

impl Drafter<'_> {
    /// The tokens to speculate on, continuing after all of `history`.
    ///
    /// `room` caps the draft at what the KV cache can still hold. It is not a
    /// tuning knob: a verification forward appends the whole draft to the
    /// cache at once, so a draft longer than the remaining context writes past
    /// the allocation — on the GPU mirror, which is sized to the request's
    /// capacity, that is a write into whatever is next rather than an error.
    /// Only reachable in the last few tokens of a full context, which is why
    /// it survived unnoticed on the prompt-lookup path.
    fn draft(&mut self, history: &[u32], room: usize, slot_id: usize) -> Result<Vec<u32>> {
        if room == 0 {
            return Ok(Vec::new());
        }
        match self {
            Drafter::PromptLookup { ngram, max_draft } => {
                Ok(ngram_draft(history, *ngram, (*max_draft).min(room)))
            }
            Drafter::Model {
                model,
                tokens,
                cache,
                committed,
            } => {
                let tokens = &(*tokens).min(room);
                // Everything committed that this cache has not seen yet — the
                // whole prompt on the first step, one token on every step
                // after.
                let catch_up = &history[*committed..];
                if catch_up.is_empty() {
                    // `history` always ends with the token the target is about
                    // to forward, and `committed` never runs past it.
                    anyhow::bail!("draft cache is ahead of the committed history");
                }
                let mut logits = draft_forward(*model, cache, catch_up, *committed, slot_id)?;
                *committed = history.len();
                let mut drafted = Vec::with_capacity(*tokens);
                for i in 0..*tokens {
                    // Plain argmax, not the request's sampler: a draft is a
                    // guess, and its only effect is how often the guess is
                    // accepted. Running the caller's temperature here would
                    // make the *draft* random without making the output any
                    // less determined by the target.
                    let token = crate::engine::sampling::argmax(&logits);
                    drafted.push(token);
                    // The last drafted token is never fed back: nothing would
                    // read its logits, and committing it to the draft cache
                    // would put the cache a token ahead of what the target can
                    // possibly accept.
                    if i + 1 == *tokens {
                        break;
                    }
                    logits = draft_forward(*model, cache, &[token], *committed, slot_id)?;
                    *committed += 1;
                }
                Ok(drafted)
            }
        }
    }

    /// Rolls the draft's own cache back to the tokens the target actually
    /// committed.
    ///
    /// Without this the rejected tail stays in the draft cache and every
    /// later position is written one slot too far along — the draft would
    /// keep producing tokens, they would simply stop resembling anything the
    /// target would say, and the only symptom would be an acceptance rate
    /// quietly falling to zero.
    fn commit(&mut self, committed_tokens: usize) {
        if let Drafter::Model {
            cache, committed, ..
        } = self
            && *committed > committed_tokens
        {
            cache.truncate(committed_tokens);
            *committed = committed_tokens;
        }
    }

    /// What the acceptance log calls this drafter.
    fn label(&self) -> &'static str {
        match self {
            Drafter::PromptLookup { .. } => "prompt-lookup",
            Drafter::Model { .. } => "draft model",
        }
    }
}

/// Runs the draft model forward and returns the last position's logits.
///
/// **`forward_all_logits`, never `forward`, and that is the whole point of
/// this function existing.** A single-token `forward` takes the GPU-fused
/// decode path on the backends that have one, and that path writes the key and
/// value straight into the GPU mirror without populating the host-side rows —
/// `KvCache::advance_gpu_only` says so, and warns in as many words that a
/// cache treated this way cannot later be read by a CPU attention pass.
///
/// A draft cache is read that way constantly: every step rolls it back to what
/// the target accepted and then runs a multi-position pass over it. Mixing the
/// two produced exactly the two failures the warning predicts — a read one row
/// past the end of the host buffer, and, where the buffer happened to be long
/// enough, an acceptance rate of zero from a model drafting for *itself*.
///
/// Chunked for the same reason [`prefill`] is: the first call carries the
/// whole prompt, and one large submission is what resets the device.
fn draft_forward(
    model: &dyn ModelForward,
    cache: &mut KvCache,
    tokens: &[u32],
    start_pos: usize,
    slot_id: usize,
) -> Result<Vec<f32>> {
    let batch = prefill_batch();
    let chunk = if batch == 0 { tokens.len() } else { batch };
    let mut last = Vec::new();
    for (i, part) in tokens.chunks(chunk.max(1)).enumerate() {
        let logits = model.forward_all_logits(cache, part, start_pos + i * chunk, slot_id)?;
        last = logits
            .into_iter()
            .next_back()
            .ok_or_else(|| anyhow::anyhow!("the draft model returned no logits"))?;
    }
    Ok(last)
}

/// Prompt-lookup draft: find the most recent earlier occurrence of the last
/// `ngram` tokens of `history`, and return up to `max_draft` tokens that
/// followed it there — a zero-cost (no model call) guess at what comes next,
/// which pays off whenever the output echoes the context (code, quotations,
/// structured/repetitive text). Empty when there's no match.
fn ngram_draft(history: &[u32], ngram: usize, max_draft: usize) -> Vec<u32> {
    if history.len() <= ngram {
        return Vec::new();
    }
    let suffix = &history[history.len() - ngram..];
    // Scan match-start positions newest-first; the most recent occurrence is
    // the best predictor of what follows now.
    for start in (0..history.len() - ngram).rev() {
        if &history[start..start + ngram] == suffix {
            let from = start + ngram;
            let to = (from + max_draft).min(history.len());
            return history[from..to].to_vec();
        }
    }
    Vec::new()
}

/// One speculative step. Takes a draft from `drafter`, verifies the whole of
/// it in a single multi-position forward, keeps the longest prefix the model
/// would itself have produced greedily, and rolls the rejected tail off both
/// KV caches. Returns the model's own next token (identical to what plain
/// greedy decoding produces here) and pushes any further accepted tokens onto
/// `spec_buf` for the loop to emit before its next forward.
/// `accepted`/`steps` accumulate acceptance stats for the final log.
///
/// Only sound for greedy sampling: a draft token is accepted only when it
/// equals the sampler's own pick at that position, so the emitted sequence is
/// byte-for-byte what non-speculative greedy decoding would emit.
#[allow(clippy::too_many_arguments)]
fn speculative_next(
    model: &dyn ModelForward,
    cache: &mut KvCache,
    sampler: &mut Sampler,
    history: &[u32],
    current: u32,
    start_pos: usize,
    capacity: usize,
    drafter: &mut Drafter<'_>,
    slot_id: usize,
    spec_buf: &mut VecDeque<u32>,
    accepted: &mut usize,
    steps: &mut usize,
) -> Result<u32> {
    // `current` takes the first of the remaining rows; the draft gets the rest.
    let room = capacity.saturating_sub(start_pos + 1);
    let draft = drafter.draft(history, room, slot_id)?;
    let mut input = Vec::with_capacity(1 + draft.len());
    input.push(current);
    input.extend_from_slice(&draft);

    // Per-position logits for `current` and every drafted token, from one
    // forward that appends all of them to the cache.
    let logits = model.forward_all_logits(cache, &input, start_pos, slot_id)?;

    // Verify greedily. `recent` mirrors what `history` would be as accepted
    // tokens are committed, so the repeat penalty (if any) sees exactly the
    // context plain decoding would — keeping the output identical.
    let rl = sampler.repeat_last_n();
    let mut recent: Vec<u32> = history[history.len().saturating_sub(rl)..].to_vec();
    let mut chosen = vec![sampler.sample(&logits[0], &recent)];
    recent.push(chosen[0]);
    let mut matched = 0usize;
    while matched < draft.len() && draft[matched] == chosen[matched] {
        let next = sampler.sample(&logits[matched + 1], &recent);
        recent.push(next);
        chosen.push(next);
        matched += 1;
    }

    // Keep `current` plus the `matched` accepted drafts; drop the rest. The
    // last element of `chosen` is the model's own token past the accepted
    // prefix — its key/value is not committed (it was never an accepted input),
    // so it becomes the next frontier the loop forwards.
    let committed_tokens = start_pos + chosen.len();
    cache.truncate(committed_tokens);
    // The draft's cache follows the target's exactly, which is what keeps the
    // two models looking at the same context. A drafter with no cache of its
    // own ignores this.
    drafter.commit(committed_tokens);
    *accepted += matched;
    *steps += 1;
    for &t in &chosen[1..] {
        spec_buf.push_back(t);
    }
    Ok(chosen[0])
}

#[cfg(test)]
mod message_header_tests {
    use super::{MessageHeader, REASONING_RECIPIENT};

    const START: u32 = 1;
    const MESSAGE: u32 = 2;
    const EOM: u32 = 3;
    /// The body-kind markers of the second format (`inkling`'s
    /// `<|content_thinking|>` and `<|content_text|>`).
    const THINKING: u32 = 4;
    const TEXT: u32 = 5;

    /// One thing the model produced: a structural marker, or a run of
    /// ordinary text.
    enum Out {
        Marker(u32),
        Text(&'static str),
    }
    use Out::{Marker, Text};

    /// Everything a client would see for `stream`, given a reply that
    /// resumes inside a header (the ordinary generation prompt) and a role
    /// that either shows reasoning or suppresses it.
    fn shown(suppress_reasoning: bool, resumes_in_header: bool, stream: &[Out]) -> String {
        let mut header = MessageHeader {
            framing: Some((START, MESSAGE)),
            kinds: None,
            inside: resumes_in_header,
            recipient: String::new(),
            hidden_body: false,
            emitted_body: false,
            suppress_reasoning,
        };
        let mut out = String::new();
        for item in stream {
            match item {
                Marker(id) => {
                    if let Some(separator) = header.observe_marker(*id) {
                        out.push_str(separator);
                    }
                }
                Text(text) => {
                    if let Some(text) = header.observe_text((*text).to_string()) {
                        out.push_str(&text);
                    }
                }
            }
        }
        out
    }

    /// A whole `muse-glimmer` assistant turn: a reasoning message, then the
    /// answer. Every header is dropped — this is the bug that made replies
    /// begin `" to=self"` — and the two bodies are separated rather than
    /// run together (`…Final.Three primes larger than…`).
    #[test]
    fn a_showing_role_gets_both_messages_with_their_headers_dropped() {
        let turn = [
            Text(" to=self"),
            Marker(MESSAGE),
            Text("Let me think."),
            Marker(EOM),
            Marker(START),
            Text(" to=user"),
            Marker(MESSAGE),
            Text("23, 29, 31."),
        ];
        assert_eq!(shown(false, true, &turn), "Let me think.\n\n23, 29, 31.");
    }

    /// The same turn under a reasoning-suppressing role: the answer alone,
    /// with no separator in front of it (there is nothing to separate).
    #[test]
    fn a_suppressing_role_gets_only_the_message_addressed_to_the_caller() {
        let turn = [
            Text(" to=self"),
            Marker(MESSAGE),
            Text("Let me think."),
            Marker(EOM),
            Marker(START),
            Text(" to=user"),
            Marker(MESSAGE),
            Text("23, 29, 31."),
        ];
        assert_eq!(shown(true, true, &turn), "23, 29, 31.");
    }

    /// The recipient arrives one token at a time and need not land as a
    /// single piece, so the test that reads it must not assume it did.
    #[test]
    fn a_recipient_split_across_tokens_is_still_recognized() {
        let turn = [
            Text(" to"),
            Text("="),
            Text("se"),
            Text("lf"),
            Marker(MESSAGE),
            Text("thinking"),
        ];
        assert_eq!(shown(true, true, &turn), "");
        assert_eq!(shown(false, true, &turn), "thinking");
    }

    /// A prompt that does *not* end in a header — a prefilled partial reply,
    /// or any raw-completion prompt — resumes inside a body, and hiding its
    /// first tokens would swallow the start of the reply.
    #[test]
    fn a_reply_resuming_inside_a_body_hides_nothing() {
        assert_eq!(
            shown(false, false, &[Text("already answering")]),
            "already answering"
        );
        assert_eq!(
            shown(true, false, &[Text("already answering")]),
            "already answering"
        );
    }

    /// Every vocabulary without both markers keeps exactly its previous
    /// behavior: nothing is hidden and nothing is inserted, whatever the
    /// model emits and whatever the role is.
    #[test]
    fn a_vocabulary_without_the_markers_is_unaffected() {
        let mut header = MessageHeader {
            framing: None,
            kinds: None,
            inside: false,
            recipient: String::new(),
            hidden_body: false,
            emitted_body: false,
            suppress_reasoning: true,
        };
        assert_eq!(header.observe_marker(START), None);
        assert_eq!(header.observe_marker(MESSAGE), None);
        assert_eq!(
            header.observe_text("plain".to_string()),
            Some("plain".to_string())
        );
    }

    /// Everything a client would see from an `inkling`-style reply, where
    /// each body is opened by a marker naming its kind and there is no
    /// header at all.
    fn shown_by_kind(suppress_reasoning: bool, stream: &[Out]) -> String {
        let mut header = MessageHeader {
            framing: None,
            kinds: Some(crate::engine::tokenizer::ContentKinds {
                reasoning: THINKING,
                other: vec![TEXT],
            }),
            inside: false,
            recipient: String::new(),
            hidden_body: false,
            emitted_body: false,
            suppress_reasoning,
        };
        let mut out = String::new();
        for item in stream {
            match item {
                Marker(id) => {
                    if let Some(separator) = header.observe_marker(*id) {
                        out.push_str(separator);
                    }
                }
                Text(text) => {
                    if let Some(text) = header.observe_text((*text).to_string()) {
                        out.push_str(&text);
                    }
                }
            }
        }
        out
    }

    /// A whole `inkling` assistant turn: a thinking body, then the answer.
    /// The showing role gets both, separated; the suppressing role gets the
    /// answer alone, with no leading separator.
    #[test]
    fn a_typed_body_is_shown_or_hidden_by_its_own_marker() {
        let turn = [
            Marker(THINKING),
            Text("Let me think."),
            Marker(EOM),
            Marker(TEXT),
            Text("23, 29, 31."),
        ];
        assert_eq!(shown_by_kind(false, &turn), "Let me think.\n\n23, 29, 31.");
        assert_eq!(shown_by_kind(true, &turn), "23, 29, 31.");
    }

    /// The markers that are *not* body kinds — a turn marker, an end
    /// marker — must not be read as the start of a visible body. Taking
    /// `<|end_message|>` for one would un-hide the reasoning that follows
    /// it under a suppressing role.
    #[test]
    fn a_marker_that_is_not_a_body_kind_does_not_reopen_a_body() {
        let turn = [
            Marker(THINKING),
            Text("first thought."),
            Marker(EOM),
            Marker(START),
            Text("second thought."),
        ];
        assert_eq!(shown_by_kind(true, &turn), "");
        assert_eq!(shown_by_kind(false, &turn), "first thought.second thought.");
    }

    /// A reply that never writes `<|message|>` must not be swallowed. The
    /// prompt still ends in a header, so the filter starts by withholding —
    /// but only up to `MAX_HEADER_LEN`, after which it concludes this is
    /// not a header and releases what it held.
    ///
    /// The failure this guards is not hypothetical: appending an empty
    /// `<think>` block to a `<|start|>assistant` prompt (the old blanket
    /// reasoning-suppression prefill) produced exactly such a reply, and
    /// with no bound the whole answer came back empty.
    #[test]
    fn a_reply_that_never_leaves_the_header_is_released_rather_than_swallowed() {
        let long = "x".repeat(super::MAX_HEADER_LEN + 1);
        let stream = [Text(Box::leak(long.into_boxed_str()) as &'static str)];
        let out = shown(false, true, &stream);
        assert!(
            out.len() > super::MAX_HEADER_LEN,
            "withheld text must be released, got {} chars",
            out.len()
        );
        assert!(out.chars().all(|c| c == 'x'));
    }

    /// The recipient this module keys on is the one the format writes; a
    /// header naming anyone else is an ordinary message, shown under every
    /// role.
    #[test]
    fn only_the_reasoning_recipient_is_suppressed() {
        assert_eq!(REASONING_RECIPIENT, "to=self");
        let turn = [Text(" to=user"), Marker(MESSAGE), Text("the answer")];
        assert_eq!(shown(true, true, &turn), "the answer");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::kv_cache::KvCache;
    use crate::engine::loader::{ModelConfig, PoolingType};
    use crate::engine::scheduler::SlotPool;
    use orangu::gguf::{GgufFile, GgufValue};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A deterministic, model-math-free `ModelForward`: each position's
    /// key/value is a pure function of `(token, position)`, and the
    /// returned logits are a pure function of every cached key so far —
    /// so whether an earlier position's key was computed by *this* call or
    /// copied in from a previous request's cache
    /// (`KvCache::copy_prefix_from`) can't matter, exactly the property
    /// `prefix_cache_reuse_matches_a_full_recompute` below needs to isolate
    /// prefix reuse's own correctness from any real model's floating-point
    /// non-associativity across different batch shapes (a separate,
    /// already-present property of the real GPU backends, not something
    /// this module's own plumbing introduces).
    struct DeterministicModel {
        config: ModelConfig,
        /// Total tokens ever passed to `forward` — lets a test confirm
        /// prefix reuse actually skipped work, not just that it didn't
        /// change the result.
        forwarded_tokens: AtomicUsize,
        /// Disagree with the base rule on every `n`th call, so an instance
        /// can stand in for a *draft* model: one that mostly predicts what
        /// the target would say and sometimes does not. `0` never disagrees,
        /// which makes the draft a perfect oracle.
        ///
        /// Counted per call rather than per token so the disagreement lands
        /// at positions the caller cannot arrange for, which is the point —
        /// a draft that failed only where a test expected it to would not
        /// exercise the rollback at all.
        disagree_every: usize,
        calls: AtomicUsize,
        /// Entries into `forward`/`forward_all_logits`, however many tokens
        /// each carried. This — not the token count — is what speculation
        /// reduces: a verification forward is one call carrying the whole
        /// draft, where plain decoding would have made one call per token.
        forward_calls: AtomicUsize,
    }

    impl DeterministicModel {
        fn new(n_vocab: usize) -> Self {
            Self {
                config: ModelConfig {
                    architecture: "test".to_string(),
                    n_vocab,
                    n_embd: 1,
                    n_layer: 1,
                    n_head: 1,
                    n_head_kv: 1,
                    head_dim: 4,
                    n_ctx_train: 1000,
                    rope_dim: 1,
                    rope_freq_base: 10000.0,
                    rms_eps: 1e-6,
                    pooling_type: PoolingType::Mean,
                },
                forwarded_tokens: AtomicUsize::new(0),
                disagree_every: 0,
                calls: AtomicUsize::new(0),
                forward_calls: AtomicUsize::new(0),
            }
        }

        /// The same model, wrong every `n`th call — a stand-in for a draft
        /// model that is usually but not always right.
        fn drafting(n_vocab: usize, disagree_every: usize) -> Self {
            Self {
                disagree_every,
                ..Self::new(n_vocab)
            }
        }

        /// The winner this model's rule picks for a cache in its current
        /// state, with the deliberate disagreement applied.
        fn winner(&self, layer: &crate::engine::kv_cache::LayerCache) -> usize {
            let mut acc = 0f32;
            for p in 0..layer.len {
                acc += layer.key_at(p, 0, 1)[0];
            }
            let base = (acc.abs() as u64 as usize) % self.config.n_vocab;
            let call = self.calls.fetch_add(1, Ordering::Relaxed);
            if self.disagree_every > 0 && call.is_multiple_of(self.disagree_every) {
                (base + 1) % self.config.n_vocab
            } else {
                base
            }
        }

        fn logits_for(&self, winner: usize) -> Vec<f32> {
            let mut logits = vec![0f32; self.config.n_vocab];
            logits[winner] = 10.0;
            logits
        }
    }

    impl ModelForward for DeterministicModel {
        fn config(&self) -> &ModelConfig {
            &self.config
        }

        fn new_kv_cache(&self, capacity: usize) -> KvCache {
            KvCache::new(1, capacity, 1)
        }

        fn forward(
            &self,
            cache: &mut KvCache,
            tokens: &[u32],
            start_pos: usize,
            _slot_id: usize,
        ) -> Result<Vec<f32>> {
            self.forwarded_tokens
                .fetch_add(tokens.len(), Ordering::Relaxed);
            self.forward_calls.fetch_add(1, Ordering::Relaxed);
            let layer = &mut cache.layers[0];
            for (i, &t) in tokens.iter().enumerate() {
                let val = t as f32 * 1000.0 + (start_pos + i) as f32;
                layer.push(&[val], &[val]);
            }
            let winner = self.winner(layer);
            Ok(self.logits_for(winner))
        }

        /// Per-position logits, each identical to what `forward` would have
        /// returned had the tokens been fed one at a time — the property
        /// speculative verification is built on, and the reason this is
        /// written as the same incremental rule rather than a batched one.
        fn forward_all_logits(
            &self,
            cache: &mut KvCache,
            tokens: &[u32],
            start_pos: usize,
            _slot_id: usize,
        ) -> Result<Vec<Vec<f32>>> {
            self.forwarded_tokens
                .fetch_add(tokens.len(), Ordering::Relaxed);
            self.forward_calls.fetch_add(1, Ordering::Relaxed);
            let layer = &mut cache.layers[0];
            let mut out = Vec::with_capacity(tokens.len());
            for (i, &t) in tokens.iter().enumerate() {
                let val = t as f32 * 1000.0 + (start_pos + i) as f32;
                layer.push(&[val], &[val]);
                let winner = self.winner(layer);
                out.push(self.logits_for(winner));
            }
            Ok(out)
        }

        fn forward_hidden_states(&self, _tokens: &[u32]) -> Result<Vec<f32>> {
            unimplemented!("not exercised by this test")
        }
    }

    /// A minimal real `Tokenizer` (plain single-letter tokens, `"llama"`
    /// vocab kind so `decode` needs no byte-mapping table) — only
    /// `Tokenizer::decode` is exercised by `run`, to turn each sampled
    /// token id back into the streamed text this test compares.
    fn letter_tokenizer(n_vocab: usize) -> Tokenizer {
        let tokens: Vec<GgufValue> = (0..n_vocab)
            .map(|i| GgufValue::String(char::from_u32('a' as u32 + i as u32).unwrap().to_string()))
            .collect();
        let gguf = GgufFile {
            version: 3,
            metadata: vec![
                (
                    "tokenizer.ggml.tokens".to_string(),
                    GgufValue::Array(tokens),
                ),
                (
                    "tokenizer.ggml.model".to_string(),
                    GgufValue::String("llama".to_string()),
                ),
            ],
            tensors: vec![],
            alignment: 32,
            data_offset: 0,
        };
        Tokenizer::from_gguf(&gguf).unwrap()
    }

    /// [`letter_tokenizer`], but every token is a CONTROL token.
    ///
    /// Which makes every one of them *suppressed*: nothing reaches the stream,
    /// exactly as a chat format's structural markers and a hidden reasoning
    /// body do. That is the shape in which "stop when the client goes away"
    /// used to fail, so it is the shape a test for it has to use.
    fn suppressed_tokenizer(n_vocab: usize) -> Tokenizer {
        let tokens: Vec<GgufValue> = (0..n_vocab)
            .map(|i| GgufValue::String(char::from_u32('a' as u32 + i as u32).unwrap().to_string()))
            .collect();
        let gguf = GgufFile {
            version: 3,
            metadata: vec![
                (
                    "tokenizer.ggml.tokens".to_string(),
                    GgufValue::Array(tokens),
                ),
                (
                    "tokenizer.ggml.model".to_string(),
                    GgufValue::String("llama".to_string()),
                ),
                (
                    // 3 is llama.cpp's `LLAMA_TOKEN_TYPE_CONTROL`.
                    "tokenizer.ggml.token_type".to_string(),
                    GgufValue::Array((0..n_vocab).map(|_| GgufValue::I32(3)).collect()),
                ),
            ],
            tensors: vec![],
            alignment: 32,
            data_offset: 0,
        };
        Tokenizer::from_gguf(&gguf).unwrap()
    }

    fn greedy_params() -> SamplingParams {
        SamplingParams {
            temperature: 0.0,
            repeat_penalty: 1.0,
            repeat_last_n: 0,
            ..SamplingParams::default()
        }
    }

    /// Drains every event `run` already sent (it only returns after
    /// sending `Done`, so nothing is still in flight) into the
    /// concatenated streamed text plus whether it finished without error.
    fn drain(mut rx: UnboundedReceiver<StreamEvent>) -> (String, bool) {
        let mut text = String::new();
        let mut ok = false;
        while let Ok(event) = rx.try_recv() {
            match event {
                StreamEvent::Token(t) => text.push_str(&t),
                StreamEvent::PromptProgress { .. } | StreamEvent::Timings(_) => {}
                StreamEvent::Done { .. } => ok = true,
                StreamEvent::Error(e) => panic!("unexpected generation error: {e}"),
                StreamEvent::Overloaded => panic!("unexpected refusal: these pools are unbounded"),
            }
        }
        (text, ok)
    }

    fn run_request(
        model: &DeterministicModel,
        tokenizer: &Tokenizer,
        prefix_cache: Option<&PrefixCache>,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
    ) -> (String, bool) {
        run_request_cached(
            model,
            tokenizer,
            prefix_cache,
            prompt_tokens,
            max_tokens,
            true,
        )
    }

    /// `run_request`, with a draft model attached.
    ///
    /// Everything else is identical, which is what makes the comparison
    /// worth anything: speculation is only allowed to change how many
    /// forwards a request costs, never a byte of what it emits.
    fn run_request_drafted(
        model: &DeterministicModel,
        draft: Arc<DeterministicModel>,
        tokenizer: &Tokenizer,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        draft_tokens: usize,
    ) -> (String, bool) {
        let slots = SlotPool::new(1);
        let guard = pollster::block_on(slots.acquire());
        let (tx, rx) = mpsc::unbounded_channel();
        let req = GenerateRequest {
            json_output: false,
            prompt_tokens,
            sampling: greedy_params(),
            max_tokens,
            stop_token_ids: vec![],
            cache_prompt: true,
            id_slot: None,
            timings_per_token: false,
        };
        let draft = DraftModel {
            model: draft,
            tokens: draft_tokens,
            label: "test-draft".to_string(),
        };
        run(
            model,
            tokenizer,
            Some(&draft),
            None,
            None,
            None,
            &guard,
            req,
            crate::config::Role::default(),
            &crate::engine::metrics::ServerMetrics::new(),
            Instant::now(),
            tx,
        )
        .unwrap();
        drain(rx)
    }

    /// `run_request` with explicit control over whether the request is allowed
    /// to read a cache (`GenerateRequest::cache_prompt`).
    fn run_request_cached(
        model: &DeterministicModel,
        tokenizer: &Tokenizer,
        prefix_cache: Option<&PrefixCache>,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        cache_prompt: bool,
    ) -> (String, bool) {
        let slots = SlotPool::new(1);
        let guard = pollster::block_on(slots.acquire());
        let (tx, rx) = mpsc::unbounded_channel();
        let req = GenerateRequest {
            json_output: false,
            prompt_tokens,
            sampling: greedy_params(),
            max_tokens,
            stop_token_ids: vec![],
            cache_prompt,
            id_slot: None,
            timings_per_token: false,
        };
        run(
            model,
            tokenizer,
            None,
            None,
            prefix_cache,
            None,
            &guard,
            req,
            crate::config::Role::default(),
            &crate::engine::metrics::ServerMetrics::new(),
            Instant::now(),
            tx,
        )
        .unwrap();
        drain(rx)
    }

    /// The correctness property prefix reuse must never break: a second,
    /// growing-conversation request (this exact model's own full first-
    /// turn history, plus a short new suffix — the shape `engine::
    /// prefix_cache`'s own doc comment calls the primary use case) must
    /// stream back *exactly* the same text whether or not a `PrefixCache`
    /// let it skip re-prefilling the shared part, and reuse must actually
    /// have skipped real work when it's available.
    #[test]
    fn prefix_cache_reuse_matches_a_full_recompute() {
        let n_vocab = 32;
        let tokenizer = letter_tokenizer(n_vocab);
        let turn1_prompt = vec![1u32, 2, 3, 4, 5];
        let turn2_suffix = vec![6u32, 7];

        // Baseline: no prefix cache at all, turn 2 is a full reprefill of
        // its own complete prompt from position 0 — today's behavior.
        let model = DeterministicModel::new(n_vocab);
        let (turn1_text, ok1) = run_request(&model, &tokenizer, None, turn1_prompt.clone(), 3);
        assert!(ok1);
        let mut turn2_prompt_baseline = turn1_prompt.clone();
        for ch in turn1_text.chars() {
            turn2_prompt_baseline.push(ch as u32 - 'a' as u32);
        }
        turn2_prompt_baseline.extend(turn2_suffix.clone());
        let (turn2_text_baseline, ok2) =
            run_request(&model, &tokenizer, None, turn2_prompt_baseline.clone(), 3);
        assert!(ok2);

        // Same two turns, this time through a shared `PrefixCache` — turn
        // 2's prompt is byte-for-byte `turn2_prompt_baseline` (same
        // tokenizer, same deterministic turn-1 output), so it should find
        // and reuse turn 1's entire cached history.
        let model = DeterministicModel::new(n_vocab);
        let pool = PrefixCache::new(4);
        let (turn1_text_reuse, ok1) =
            run_request(&model, &tokenizer, Some(&pool), turn1_prompt.clone(), 3);
        assert!(ok1);
        assert_eq!(
            turn1_text_reuse, turn1_text,
            "turn 1 has no prefix to reuse yet"
        );
        let mut turn2_prompt_reuse = turn1_prompt.clone();
        for ch in turn1_text_reuse.chars() {
            turn2_prompt_reuse.push(ch as u32 - 'a' as u32);
        }
        turn2_prompt_reuse.extend(turn2_suffix.clone());
        assert_eq!(
            turn2_prompt_reuse, turn2_prompt_baseline,
            "both runs' turn-2 prompts must be identical for this comparison to mean anything"
        );
        let forwarded_before_turn2 = model.forwarded_tokens.load(Ordering::Relaxed);
        let (turn2_text_reuse, ok2) =
            run_request(&model, &tokenizer, Some(&pool), turn2_prompt_reuse, 3);
        assert!(ok2);
        let reuse_forwarded = model.forwarded_tokens.load(Ordering::Relaxed);

        assert_eq!(
            turn2_text_reuse, turn2_text_baseline,
            "prefix reuse must produce byte-identical output to a full recompute"
        );
        // Turn 2's own forward-pass token count: reuse must have skipped
        // all but the very last of turn 1's 8-token history. `run`'s
        // decode loop stops as soon as `history.len()` reaches its target
        // capacity (`prompt.len() + max_tokens`) — which happens right
        // after the *last* generated token is appended to `history` but
        // *before* the forward call that would have pushed its own
        // key/value into the cache (`PrefixCache::take_best_match`'s own
        // doc comment covers this). Both turns here use `max_tokens = 3`
        // with no stop token ever reached, so this fires identically for
        // both: turn 1 leaves only 7 of its own 8 tokens actually cached,
        // and turn 2's own decode loop likewise only reaches 2 real
        // forward calls (its 3rd generated token's own forward call is
        // the one skipped this time). So turn 2 must forward: turn 1's
        // uncached 8th token (1), the 2 brand-new suffix tokens, plus 2
        // (not 3) decode-step forwards.
        let turn2_forwarded_reuse = reuse_forwarded - forwarded_before_turn2;
        assert_eq!(
            turn2_forwarded_reuse,
            1 + turn2_suffix.len() + 2,
            "reuse must skip turn 1's first 7 cached positions, forwarding only its own uncached 8th token, the new suffix, and this turn's own 2 real decode steps"
        );
    }

    /// `cache_prompt: false` must put every prompt token through a real
    /// forward pass even when a `PrefixCache` holds an exact match for the
    /// whole prompt — the property a prefill benchmark depends on, since a
    /// request served from cache reports a prompt time that measures the
    /// lookup and nothing else. Checked by counting forwarded tokens, not by
    /// timing: the same run repeated with the flag left at its default must
    /// skip that work, so the two counts bracket what the flag controls.
    #[test]
    fn cache_prompt_false_reprefills_a_prompt_the_cache_already_holds() {
        let n_vocab = 32;
        let tokenizer = letter_tokenizer(n_vocab);
        let prompt = vec![1u32, 2, 3, 4, 5];
        let model = DeterministicModel::new(n_vocab);
        let pool = PrefixCache::new(4);

        // Populate the pool, then re-send the identical prompt twice.
        let (first_text, ok) = run_request(&model, &tokenizer, Some(&pool), prompt.clone(), 3);
        assert!(ok);

        let before_uncached = model.forwarded_tokens.load(Ordering::Relaxed);
        let (uncached_text, ok) = run_request_cached(
            &model,
            &tokenizer,
            Some(&pool),
            prompt.clone(),
            3,
            /* cache_prompt */ false,
        );
        assert!(ok);
        let uncached_forwarded = model.forwarded_tokens.load(Ordering::Relaxed) - before_uncached;

        let before_cached = model.forwarded_tokens.load(Ordering::Relaxed);
        let (cached_text, ok) = run_request(&model, &tokenizer, Some(&pool), prompt.clone(), 3);
        assert!(ok);
        let cached_forwarded = model.forwarded_tokens.load(Ordering::Relaxed) - before_cached;

        // Skipping the cache changes cost, never output.
        assert_eq!(uncached_text, first_text);
        assert_eq!(cached_text, first_text);
        // All 5 prompt tokens forwarded, plus this run's own 2 real decode
        // steps (the 3rd generated token's forward call is skipped — see
        // `prefix_cache_reuse_matches_a_full_recompute` for why).
        assert_eq!(
            uncached_forwarded,
            prompt.len() + 2,
            "cache_prompt: false must prefill the whole prompt"
        );
        assert!(
            cached_forwarded < uncached_forwarded,
            "the same request with caching allowed must forward fewer tokens \
             ({cached_forwarded} vs {uncached_forwarded}) — otherwise this test \
             proves nothing about the flag"
        );
    }

    /// A `ModelForward` whose `forward` always panics — the deliberately
    /// broken model this module's own panic-recovery path
    /// (`Engine::generate`'s `catch_unwind` around `run`, `crate::
    /// panic_capture`) needs a real panic to exercise end to end, not
    /// just unit-test in isolation.
    struct PanickingModel {
        config: ModelConfig,
    }

    impl PanickingModel {
        fn new() -> Self {
            Self {
                config: ModelConfig {
                    architecture: "test".to_string(),
                    n_vocab: 8,
                    n_embd: 1,
                    n_layer: 1,
                    n_head: 1,
                    n_head_kv: 1,
                    head_dim: 4,
                    n_ctx_train: 1000,
                    rope_dim: 1,
                    rope_freq_base: 10000.0,
                    rms_eps: 1e-6,
                    pooling_type: PoolingType::Mean,
                },
            }
        }
    }

    impl ModelForward for PanickingModel {
        fn config(&self) -> &ModelConfig {
            &self.config
        }

        fn new_kv_cache(&self, capacity: usize) -> KvCache {
            KvCache::new(1, capacity, 1)
        }

        fn forward(
            &self,
            _cache: &mut KvCache,
            _tokens: &[u32],
            _start_pos: usize,
            _slot_id: usize,
        ) -> Result<Vec<f32>> {
            panic!("PANICKING_MODEL_DELIBERATE_TEST_PANIC");
        }

        fn forward_hidden_states(&self, _tokens: &[u32]) -> Result<Vec<f32>> {
            unimplemented!("not exercised by this test")
        }
    }

    /// A `ModelForward` that records the `(tokens, start_pos)` of every
    /// `forward` call instead of computing anything, so a test can see
    /// exactly how a prompt was fed to the model.
    struct RecordingModel {
        config: ModelConfig,
        calls: std::sync::Mutex<Vec<(Vec<u32>, usize)>>,
    }

    impl RecordingModel {
        fn new() -> Self {
            Self {
                config: PanickingModel::new().config,
                calls: std::sync::Mutex::new(Vec::new()),
            }
        }
    }

    impl ModelForward for RecordingModel {
        fn config(&self) -> &ModelConfig {
            &self.config
        }

        fn new_kv_cache(&self, capacity: usize) -> KvCache {
            KvCache::new(1, capacity, 1)
        }

        fn forward(
            &self,
            _cache: &mut KvCache,
            tokens: &[u32],
            start_pos: usize,
            _slot_id: usize,
        ) -> Result<Vec<f32>> {
            self.calls
                .lock()
                .unwrap()
                .push((tokens.to_vec(), start_pos));
            // Logits that identify which call produced them, so the caller
            // can be checked to keep the *last* chunk's.
            Ok(vec![start_pos as f32 + tokens.len() as f32; 8])
        }

        fn forward_hidden_states(&self, _tokens: &[u32]) -> Result<Vec<f32>> {
            unimplemented!("not exercised by this test")
        }
    }

    /// A prompt longer than one chunk is fed as consecutive `forward`
    /// calls, each starting where the previous one ended — covering every
    /// token exactly once, in order, with no gap or overlap. Anything else
    /// would corrupt the KV cache rather than merely slow things down.
    #[test]
    fn a_long_prompt_is_prefilled_in_consecutive_chunks() {
        let model = RecordingModel::new();
        let mut cache = model.new_kv_cache(64);
        let tokens: Vec<u32> = (0..25).collect();

        let logits = prefill_in_chunks(&model, &mut cache, &tokens, 0, 0, 10, &mut |_| {}).unwrap();

        let calls = model.calls.lock().unwrap().clone();
        assert_eq!(
            calls,
            vec![
                ((0..10).collect::<Vec<u32>>(), 0),
                ((10..20).collect::<Vec<u32>>(), 10),
                ((20..25).collect::<Vec<u32>>(), 20),
            ]
        );
        // The last chunk's logits are what sampling continues from; an
        // earlier chunk's would sample from the middle of the prompt.
        assert_eq!(logits[0], 25.0);
    }

    /// A prompt that starts partway in (the prefix cache reused its head)
    /// keeps counting positions from there — the chunk boundaries are
    /// relative to the prompt, the positions are absolute.
    #[test]
    fn chunking_starts_from_the_reused_prefix_position() {
        let model = RecordingModel::new();
        let mut cache = model.new_kv_cache(64);
        let tokens: Vec<u32> = (0..7).collect();

        prefill_in_chunks(&model, &mut cache, &tokens, 100, 0, 3, &mut |_| {}).unwrap();

        let starts: Vec<usize> = model
            .calls
            .lock()
            .unwrap()
            .iter()
            .map(|(_, pos)| *pos)
            .collect();
        assert_eq!(starts, vec![100, 103, 106]);
    }

    /// The whole point: a chunk that ran long makes the next one smaller, so
    /// the *work* per submission stays bounded even though the cost per token
    /// climbs with context. Sizing by token count alone does not — 512 tokens
    /// cost 2.3 s at position 0 and 10.1 s at position 6 656, past the ~10 s
    /// the driver allows before it resets the device.
    #[test]
    fn a_slow_chunk_shrinks_the_next_one() {
        let budget = Duration::from_millis(3_000);
        // 512 tokens took 10.1 s: ~19.7 ms each, so ~114 fit once the next
        // chunk leaves headroom for starting deeper in the prompt.
        let next = next_chunk_width(512, Duration::from_millis(10_069), budget, 512);
        assert!((105..=125).contains(&next), "sized {next}");
        // And a fast one is allowed back up to the configured maximum.
        assert_eq!(
            next_chunk_width(16, Duration::from_millis(70), budget, 512),
            512
        );
    }

    /// Whatever the measurement says, the width stays inside both bounds — the
    /// floor keeps a submission the driver can still finish, and the ceiling is
    /// the operator's own `ORANGU_PREFILL_BATCH`.
    #[test]
    fn an_adapted_width_stays_within_its_bounds() {
        let budget = Duration::from_millis(3_000);
        // Absurdly slow: would size to zero without the floor.
        assert_eq!(
            next_chunk_width(16, Duration::from_secs(600), budget, 512),
            PREFILL_MIN_CHUNK_TOKENS
        );
        // A batch smaller than the floor is honoured, not clamped upward —
        // and must not panic, which `clamp` does when min exceeds max.
        assert_eq!(next_chunk_width(3, Duration::from_secs(600), budget, 3), 3);
        // A free chunk cannot widen past the maximum.
        assert_eq!(next_chunk_width(8, Duration::ZERO, budget, 128), 128);
    }

    /// A deep continuation is the shape that used to hang the GPU: a full-width
    /// chunk submitted at a position where each token costs ten times what it
    /// did at the start. It must open with the small probe instead.
    #[test]
    fn a_continuation_opens_with_a_probe_not_a_full_width_chunk() {
        let model = RecordingModel::new();
        let mut cache = model.new_kv_cache(200_000);
        let tokens: Vec<u32> = (0..600).collect();

        prefill_in_chunks(&model, &mut cache, &tokens, 100_000, 0, 512, &mut |_| {}).unwrap();

        let first = model.calls.lock().unwrap()[0].0.len();
        assert_eq!(first, PREFILL_PROBE_TOKENS, "opened {first} tokens wide");
    }

    /// Progress must be reported *after* each chunk's forward and count
    /// cumulatively, so a client can render a bar that only ever moves
    /// forward and reaches the total exactly once.
    #[test]
    fn every_prefill_chunk_reports_cumulative_progress() {
        let model = RecordingModel::new();
        let mut cache = model.new_kv_cache(64);
        let tokens: Vec<u32> = (0..25).collect();
        let mut seen = Vec::new();

        prefill_in_chunks(&model, &mut cache, &tokens, 0, 0, 10, &mut |done| {
            seen.push((done, model.calls.lock().unwrap().len()))
        })
        .unwrap();

        // (tokens done, forwards completed) — the second element proves the
        // report follows the work rather than announcing it.
        assert_eq!(seen, vec![(10, 1), (20, 2), (25, 3)]);
    }

    /// A one-chunk prompt still reports, once, at the total — otherwise a
    /// short prompt's progress bar would never appear at all.
    #[test]
    fn a_single_chunk_prompt_still_reports_once() {
        let model = RecordingModel::new();
        let mut cache = model.new_kv_cache(64);
        let tokens: Vec<u32> = (0..5).collect();
        let mut seen = Vec::new();

        prefill_in_chunks(&model, &mut cache, &tokens, 0, 0, 512, &mut |done| {
            seen.push(done)
        })
        .unwrap();

        assert_eq!(seen, vec![5]);
    }

    /// A prompt that fits in one chunk — and `ORANGU_PREFILL_BATCH=0` at
    /// any length — is exactly one `forward` call, the same single pass
    /// this code made before chunking existed.
    #[test]
    fn a_short_prompt_is_still_one_forward_call() {
        for (len, batch) in [(5usize, 512usize), (5000, 0)] {
            let model = RecordingModel::new();
            let mut cache = model.new_kv_cache(8192);
            let tokens: Vec<u32> = (0..len as u32).collect();

            prefill_in_chunks(&model, &mut cache, &tokens, 0, 0, batch, &mut |_| {}).unwrap();

            let calls = model.calls.lock().unwrap();
            assert_eq!(calls.len(), 1, "len {len}, batch {batch}");
            assert_eq!(calls[0].0.len(), len);
        }
    }

    /// A panic during generation must reach the client as a real,
    /// detailed `StreamEvent::Error` — the panic's own message plus a
    /// captured backtrace (`panic_capture`) — not the generic "task
    /// panicked" note `tokio::task::JoinError`'s own `Display` would give
    /// on its own, and the generation channel must still terminate
    /// cleanly (one `Error` event, not a hang or a second event).
    #[tokio::test]
    async fn a_panic_during_generation_reaches_the_client_with_a_captured_backtrace() {
        // Prints the panic to stderr too (`panic_capture::install`'s hook
        // chains to, not replaces, the default one) — expected noise for a
        // test that deliberately panics, left alone rather than swapped
        // out for a silencing hook: `std::panic::set_hook` is a process-
        // global slot, and a hook that panics itself (or races another
        // concurrently-running test's own hook swap) aborts the whole
        // process outright, which a first attempt at silencing this
        // actually hit.
        crate::panic_capture::install();

        let engine = Engine {
            metrics: Arc::new(crate::engine::metrics::ServerMetrics::new()),
            draft: None,
            model: Arc::new(PanickingModel::new()),
            tokenizer: Arc::new(letter_tokenizer(8)),
            chat_template_source: None,
            slots: SlotPool::new(1),
            batch_coordinator: None,
            prefix_cache: None,
            slot_store: None,
            role: crate::config::Role::default(),
        };

        let mut rx = engine
            .generate(GenerateRequest {
                json_output: false,
                prompt_tokens: vec![1, 2, 3],
                sampling: greedy_params(),
                max_tokens: 4,
                stop_token_ids: vec![],
                cache_prompt: true,
                id_slot: None,
                timings_per_token: false,
            })
            .await;

        let event = rx
            .recv()
            .await
            .expect("generate() must send exactly one event before closing the channel on a panic");
        let StreamEvent::Error(detail) = event else {
            panic!("expected a StreamEvent::Error, got something else");
        };
        assert!(
            detail.contains("PANICKING_MODEL_DELIBERATE_TEST_PANIC"),
            "error detail must include the panic's own message, got: {detail}"
        );
        assert!(
            detail.contains("backtrace:"),
            "error detail must include a captured backtrace, got: {detail}"
        );

        assert!(
            rx.recv().await.is_none(),
            "the channel must close after the one error event, not send anything further"
        );
    }

    /// An ordinary panic keeps its full detail (the test above proves that
    /// end to end); a lost GPU device is replaced by the one sentence that
    /// is actually actionable. The panic's `wgpu` backtrace is not a
    /// diagnosis of the caller's request, and it must not be what they get
    /// instead of an explanation — the detail stays in the server's log.
    #[test]
    fn a_device_lost_panic_is_reported_as_one_sentence_not_a_backtrace() {
        let detail = "panicked at vulkan.rs:6183:\nmapping a buffer failed\n\nbacktrace:\n...";

        let ordinary = panic_report(detail.to_string(), false);
        assert_eq!(ordinary, detail, "an ordinary panic keeps its detail");

        let lost = panic_report(detail.to_string(), true);
        assert_eq!(lost, crate::device_lost::CLIENT_MESSAGE);
        assert!(!lost.contains("backtrace"));
        assert!(!lost.contains("vulkan.rs"));
    }

    /// A request whose client has gone away stops, even when every token it
    /// produces is suppressed.
    ///
    /// **This is the case the old check missed**, and it missed it silently.
    /// The disconnect test used to ride on sending a token, so a generation
    /// that rendered nothing — structural markers, or a reasoning body the
    /// role hides — never asked whether anyone was listening and ran to
    /// `max_tokens`, holding the slot every other request is queued for.
    ///
    /// Asserted on the target's forward calls rather than on elapsed time, so
    /// it fails on a slow machine for the right reason or not at all.
    #[test]
    fn generation_stops_when_the_client_goes_away_even_with_nothing_to_send() {
        let model = DeterministicModel::new(8);
        let tokenizer = suppressed_tokenizer(8);
        let slots = SlotPool::new(1);
        let guard = pollster::block_on(slots.acquire());
        let (tx, rx) = mpsc::unbounded_channel();
        // The client is already gone before the first token.
        drop(rx);

        let max_tokens = 500;
        run(
            &model,
            &tokenizer,
            None,
            None,
            None,
            None,
            &guard,
            GenerateRequest {
                prompt_tokens: vec![1, 2, 3],
                sampling: greedy_params(),
                max_tokens,
                ..Default::default()
            },
            crate::config::Role::default(),
            &crate::engine::metrics::ServerMetrics::new(),
            Instant::now(),
            tx,
        )
        .unwrap();

        // One prefill plus a couple of decode steps, not five hundred.
        let calls = model.forward_calls.load(Ordering::Relaxed);
        assert!(
            calls < 5,
            "kept generating for a client that had gone: {calls} forward calls"
        );
    }

    /// **The property the whole feature rests on.** A draft model may only
    /// change how fast an answer arrives, never what it says — so greedy
    /// output with a draft attached must be byte-identical to greedy output
    /// without one, whatever the draft proposes.
    ///
    /// Swept over how often the draft is wrong (never, every other call,
    /// every third, always) and over the draft depth, because the interesting
    /// failures are all in the rollback: a cache left one token long, or a
    /// draft cache that quietly desynchronises from the target's, shows up as
    /// divergence only at particular acceptance patterns.
    #[test]
    fn a_draft_model_never_changes_what_greedy_decoding_emits() {
        let tokenizer = letter_tokenizer(8);
        let prompt = vec![1u32, 2, 3, 4, 1, 2];
        let plain = {
            let model = DeterministicModel::new(8);
            run_request(&model, &tokenizer, None, prompt.clone(), 40)
        };
        assert!(plain.1, "the baseline run must finish");
        assert!(!plain.0.is_empty(), "the baseline run must emit something");

        for disagree_every in [0usize, 1, 2, 3, 5] {
            for draft_tokens in [1usize, 2, 4, 7] {
                let model = DeterministicModel::new(8);
                let draft = Arc::new(DeterministicModel::drafting(8, disagree_every));
                let drafted = run_request_drafted(
                    &model,
                    draft.clone(),
                    &tokenizer,
                    prompt.clone(),
                    40,
                    draft_tokens,
                );
                assert_eq!(
                    drafted.0, plain.0,
                    "speculation changed the output (disagree_every={disagree_every}, \
                     draft_tokens={draft_tokens})"
                );
                assert!(drafted.1, "the drafted run must finish");
                assert!(
                    draft.forwarded_tokens.load(Ordering::Relaxed) > 0,
                    "the draft model was never actually run"
                );
            }
        }
    }

    /// A perfect draft must actually save forwards, or the path is inert and
    /// the test above would pass on a `Drafter` that returned nothing.
    ///
    /// Counted in *forward calls*, not tokens: the target still processes
    /// every token — a verification forward carries the whole draft — so
    /// tokens are exactly what speculation does not reduce. What it reduces
    /// is how many times the model has to be entered, which on a
    /// bandwidth-bound decode is one full pass over the weights each.
    #[test]
    fn a_perfect_draft_costs_the_target_fewer_forward_calls() {
        let tokenizer = letter_tokenizer(8);
        let prompt = vec![1u32, 2, 3, 4, 1, 2];

        let plain_model = DeterministicModel::new(8);
        let plain = run_request(&plain_model, &tokenizer, None, prompt.clone(), 40);
        let plain_calls = plain_model.forward_calls.load(Ordering::Relaxed);

        let model = DeterministicModel::new(8);
        // `disagree_every = 0` never disagrees, so every drafted token is
        // exactly what the target would have produced.
        let draft = Arc::new(DeterministicModel::drafting(8, 0));
        let drafted = run_request_drafted(&model, draft, &tokenizer, prompt, 40, 4);
        assert_eq!(drafted.0, plain.0);

        let calls = model.forward_calls.load(Ordering::Relaxed);
        // One prefill plus one verification per five tokens, against one
        // prefill plus one call per token.
        assert!(
            calls * 3 < plain_calls,
            "a perfect 4-token draft should cut the target's forward calls several-fold: \
             {calls} vs {plain_calls}"
        );
        assert!(
            model.forwarded_tokens.load(Ordering::Relaxed) > 0,
            "the target must still have run"
        );
    }

    /// A draft that is always wrong must still be correct, and must not
    /// wedge: every step rejects everything, so the loop makes progress only
    /// through the target's own token.
    #[test]
    fn an_always_wrong_draft_still_terminates_with_the_right_answer() {
        let tokenizer = letter_tokenizer(8);
        let prompt = vec![5u32, 5, 5];
        let plain = {
            let model = DeterministicModel::new(8);
            run_request(&model, &tokenizer, None, prompt.clone(), 25)
        };
        let model = DeterministicModel::new(8);
        let draft = Arc::new(DeterministicModel::drafting(8, 1));
        let drafted = run_request_drafted(&model, draft, &tokenizer, prompt, 25, 4);
        assert_eq!(drafted.0, plain.0);
        assert!(drafted.1);
    }

    /// No drafter may propose more tokens than the KV cache can still hold.
    ///
    /// A verification forward appends `1 + draft.len()` positions at once, so
    /// a draft longer than the remaining context writes past the allocation —
    /// and on the GPU mirror, sized to the request's capacity, that is a write
    /// into whatever is next rather than an error. Only reachable in the last
    /// few tokens of a full context, which is exactly why it needs a test
    /// rather than a run to find it.
    #[test]
    fn a_draft_never_runs_past_the_end_of_the_context() {
        let history: Vec<u32> = vec![1, 2, 3, 1, 2, 3, 1, 2, 3];
        for room in 0..6 {
            let mut lookup = Drafter::PromptLookup {
                ngram: 2,
                max_draft: 8,
            };
            let drafted = lookup.draft(&history, room, 0).expect("prompt lookup");
            assert!(
                drafted.len() <= room,
                "prompt-lookup drafted {} with room for {room}",
                drafted.len()
            );

            let draft_model = DeterministicModel::new(8);
            let cache = draft_model.new_kv_cache(64);
            let mut model = Drafter::Model {
                model: &draft_model,
                tokens: 8,
                cache,
                committed: 0,
            };
            let drafted = model.draft(&history, room, 0).expect("draft model");
            assert!(
                drafted.len() <= room,
                "the draft model drafted {} with room for {room}",
                drafted.len()
            );
        }
    }

    /// The draft cache is rolled back to exactly what the target committed.
    ///
    /// Asserted directly rather than through output, because the symptom of
    /// getting it wrong is not a wrong answer — the target re-derives every
    /// token either way — but an acceptance rate silently falling to zero as
    /// the draft reads a context shifted out from under it.
    #[test]
    fn commit_rolls_the_draft_cache_back_to_what_the_target_kept() {
        let draft_model = DeterministicModel::new(8);
        let mut cache = draft_model.new_kv_cache(64);
        // Ten tokens in the draft's cache, standing for eight committed plus
        // two drafted-and-about-to-be-rejected.
        for pos in 0..10 {
            draft_model
                .forward(&mut cache, &[1], pos, 0)
                .expect("test model");
        }
        let mut drafter = Drafter::Model {
            model: &draft_model,
            tokens: 4,
            cache,
            committed: 10,
        };
        drafter.commit(8);
        match &drafter {
            Drafter::Model {
                cache, committed, ..
            } => {
                assert_eq!(*committed, 8);
                assert_eq!(
                    cache.layers[0].len, 8,
                    "the rejected tail was not rolled off"
                );
            }
            _ => unreachable!(),
        }
        // Committing further ahead than the cache reaches is the
        // all-accepted case, and must leave the cache alone for the next
        // step's catch-up to fill in rather than pretend it holds more.
        drafter.commit(12);
        match &drafter {
            Drafter::Model {
                cache, committed, ..
            } => {
                assert_eq!(*committed, 8);
                assert_eq!(cache.layers[0].len, 8);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn ngram_draft_copies_the_continuation_of_the_latest_matching_context() {
        // Last two tokens are [1, 2]; their most recent earlier occurrence is
        // at index 4, followed by [3, 9, 9] — so a 3-token draft is [3, 9, 9].
        let history = [1u32, 2, 3, 7, 1, 2, 3, 9, 9, 1, 2];
        assert_eq!(ngram_draft(&history, 2, 3), vec![3, 9, 9]);
        // max_draft caps the length.
        assert_eq!(ngram_draft(&history, 2, 1), vec![3]);
    }

    #[test]
    fn ngram_draft_is_empty_without_a_match_or_enough_history() {
        assert!(ngram_draft(&[1, 2], 2, 4).is_empty()); // suffix is the whole history
        assert!(ngram_draft(&[1, 2, 3, 4, 5], 2, 4).is_empty()); // [4,5] never recurs
        assert!(ngram_draft(&[], 2, 4).is_empty());
    }

    #[test]
    fn kv_cache_truncate_rolls_back_length_and_regrows_cleanly() {
        // A plain (no-GPU-mirror) cache: push a few positions, roll back, and
        // confirm the length moves and re-pushing overwrites in place.
        let mut cache = KvCache::new(2, 8, 4);
        for i in 0..6u32 {
            for layer in &mut cache.layers {
                let v = vec![i as f32; 4];
                layer.push(&v, &v);
            }
        }
        assert_eq!(cache.layers[0].len, 6);
        cache.truncate(4);
        assert_eq!(cache.layers[0].len, 4);
        cache.truncate(10); // no-op: never grows
        assert_eq!(cache.layers[0].len, 4);
    }
}
