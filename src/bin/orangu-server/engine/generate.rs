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
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc::{self, UnboundedReceiver};

use super::arch::{ForwardOutcome, GreedySampleParams, ModelForward};
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
    /// The paged KV pool and its prefix index (`ORANGU_PAGED_KV=1`), or `None`
    /// for the per-request contiguous caches this engine has always used.
    ///
    /// Both together or neither. A pool without an index pages without
    /// sharing, which pays paging's indirection for none of its benefit; an
    /// index without a pool has nothing to point at.
    pub paged_kv: Option<(
        Arc<super::kv_pool::KvPool>,
        Arc<super::prefix_index::PrefixIndex>,
    )>,
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
        let prefix_cache = self.prefix_cache.clone();
        let slot_store = self.slot_store.clone();
        let paged_kv = self.paged_kv.clone();
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
                        prefix_cache.as_deref(),
                        slot_store.as_deref(),
                        paged_kv.as_ref(),
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
    prefix_cache: Option<&PrefixCache>,
    slot_store: Option<&super::slot_store::SlotStore>,
    paged_kv: Option<&(
        Arc<super::kv_pool::KvPool>,
        Arc<super::prefix_index::PrefixIndex>,
    )>,
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
    // The paged path, when a pool exists. It replaces both reuse mechanisms
    // below rather than layering on them: those hand a whole cache to one
    // request at a time, and the point of the pool is that a prefix has as many
    // holders as want it.
    let mut page_tags: Vec<u64> = Vec::new();
    let mut new_cache;
    let mut reused_len = 0usize;
    if let Some((pool, index)) = paged_kv {
        // The architecture builds its own cache — recurrent state included —
        // and only its positional layers move into the pool.
        new_cache = model.new_kv_cache(capacity).into_paged(pool.clone());
        if req.cache_prompt {
            // `keep_last`: a fully matched prompt still needs one page of real
            // work to produce fresh logits from — the same reason the
            // contiguous path clamps a full match by one token.
            let resolved = index.resolve(&req.prompt_tokens, true);
            if !resolved.shared.is_empty()
                && let Some(tokens) =
                    new_cache.adopt_shared_pages(&resolved.shared, pool.page_tokens())
            {
                reused_len = tokens;
            }
            page_tags = super::prefix_index::page_tags(&req.prompt_tokens, pool.page_tokens());
        } else {
            page_tags = super::prefix_index::page_tags(&req.prompt_tokens, pool.page_tokens());
        }
        // Pages this request seals become findable under these identities. Set
        // even when nothing was adopted: this request is the one that makes the
        // prefix available to the next.
        new_cache.set_page_tags(&page_tags);
    } else {
        new_cache = model.new_kv_cache(capacity);
    }
    if paged_kv.is_none()
        && req.cache_prompt
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
    if paged_kv.is_none()
        && req.cache_prompt
        && reused_len == 0
        && let Some(store) = slot_store
    {
        reused_len = store.reuse_into(guard.id(), &req.prompt_tokens, &mut new_cache);
    }
    // `Option` (not a plain `KvCache`) so the decode loop can *move* it
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
    // Committed after the whole prefill, not inside it. Prefill is
    // **layer-major** — `arch::gemma` pushes every token of layer 0, then every
    // token of layer 1 — so during it a page can be complete for one layer and
    // untouched by the rest. The only moment a span of positions is known to
    // have been through every layer is when the pass returns, which is why this
    // is the caller's call to make and not something the cache can infer.
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
    if paged_kv.is_some()
        && let Some(cache) = cache.as_mut()
    {
        cache.commit_pages();
    }
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
    let may_speculate = sampler.is_greedy() && !sampler.is_constrained();
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
            // Every layer has run over this position now, so whichever page it
            // completed may be published. A no-op on all but one step in
            // `page_tokens`, and a lock plus a scan of the fill counts when it
            // is not.
            if paged_kv.is_some()
                && let Some(cache) = cache.as_mut()
            {
                cache.commit_pages();
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
        if let Some((pool, index)) = paged_kv {
            // Publish what this request sealed, so the next one with the same
            // prompt shares it instead of recomputing it. Only whole pages of
            // the *prompt* are recorded: the generated tail is still being
            // produced page by page and the last one is partial, and a page is
            // shareable exactly when it is complete.
            //
            // Recorded here rather than as each page seals, because a page is
            // only worth advertising once the request that built it is known to
            // have finished with it — an aborted request leaves pages the pool
            // reclaims, and an index entry for one of those promises content
            // that is gone.
            let page_tokens = pool.page_tokens();
            for (i, &tag) in page_tags.iter().enumerate() {
                // Only what the pool actually holds. The index and the pool are
                // separate structures, and an index entry for a page the pool
                // never sealed is worse than no entry: `resolve` promises it,
                // the adoption then fails, and the request pays a full prefill
                // having been told it would not have to.
                if !pool.holds(tag) {
                    continue;
                }
                let run = &history[i * page_tokens..(i + 1) * page_tokens];
                index.remember(tag, run);
            }
            drop(final_cache);
        } else {
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
/// A third format frames the same distinction with a *channel*: gemma-4
/// writes `<|channel>thought\n`, its chain of thought, then `<channel|>`,
/// and may open a `tool_code` channel the same way. Both markers are already
/// hidden, so what was reaching the reader was the channel name and its body
/// run together with the answer — replies that began `"thought"` and ended
/// `"tool_code\nprint(create_directory(path='x'))"`. The name is framing and
/// is always dropped; the body is a reasoning message and follows the same
/// rule as the other two formats' — hidden when the role suppresses
/// reasoning, and otherwise separated from the answer rather than glued to
/// it.
///
/// A second format asks the same question with tokens instead of text, and
/// is answered here too: `inkling` opens each body with a marker naming its
/// *kind* (`<|content_thinking|>` for reasoning, `<|content_text|>` for the
/// answer) and writes no header at all. Same rule — a reasoning body is
/// hidden when the role says so, and two visible bodies are separated by a
/// blank line — read off `Tokenizer::content_kinds` rather than off a
/// recipient string.
///
/// Inert (`framing`, `kinds` and `channel` all `None`) for every vocabulary
/// with none of them, which is every other model this server serves — those
/// keep byte-for-byte the behavior they had. That is also what `Default`
/// gives, which is how the tests build one framing at a time.
#[derive(Default)]
struct MessageHeader {
    /// `(<|start|>, <|message|>)`, or `None` when this vocabulary has no
    /// such framing and nothing here applies.
    framing: Option<(u32, u32)>,
    /// The body-kind markers, for a vocabulary that types its bodies
    /// instead of naming a recipient. `None` for every other.
    kinds: Option<crate::engine::tokenizer::ContentKinds>,
    /// `(<|channel>, <channel|>)`, for a vocabulary that frames the model's
    /// side channels with them. `None` for every other.
    channel: Option<(u32, u32)>,
    /// Whether the stream is currently inside a channel *name* — between
    /// `<|channel>` and the newline that ends it.
    naming_channel: bool,
    /// A separator owed to the next visible text because a message boundary
    /// was crossed without a marker to hang it on. A channel closes with
    /// `<channel|>` and the answer simply resumes, so unlike the other two
    /// formats there is no opening marker later to emit it at; holding it
    /// until there is text to put it in front of also keeps it off the end
    /// of a reply that stops right after the channel.
    owed_separator: bool,
    /// Whether the stream is currently inside a header.
    inside: bool,
    /// The current header's text so far, accumulated to be read once at
    /// `<|message|>` and then discarded. Bounded by the header's own length
    /// (a recipient name), not by the reply's.
    recipient: String,
    /// The current channel's name so far, accumulated only so that a run of
    /// text long past any name's length can be released as content instead
    /// of swallowed. Bounded the same way [`Self::recipient`] is.
    channel_name: String,
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
            channel: tokenizer.channel_framing(),
            naming_channel: false,
            owed_separator: false,
            inside: Self::prompt_ends_in_header(tokenizer, prompt_tokens),
            recipient: String::new(),
            channel_name: String::new(),
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
        // A channel opens and closes messages the way the other two formats'
        // markers do, so it is settled here and returns rather than falling
        // through to them — a vocabulary has at most one of the three.
        if let Some((open, close)) = self.channel {
            if id == open {
                self.naming_channel = true;
                self.channel_name.clear();
                self.hidden_body = self.suppress_reasoning;
                self.owed_separator |= !self.hidden_body && self.emitted_body;
                return None;
            }
            if id == close {
                // Whatever follows is the answer again: a new message, and
                // one no marker will announce.
                self.naming_channel = false;
                self.hidden_body = false;
                self.owed_separator |= self.emitted_body;
                return None;
            }
        }
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
    fn observe_text(&mut self, mut text: String) -> Option<String> {
        if self.naming_channel {
            // The name runs to the first newline. One token can carry both
            // it and the start of the body, so the tail is kept rather than
            // dropped with the name.
            match text.find('\n') {
                Some(at) => {
                    self.naming_channel = false;
                    text = text.split_off(at + 1);
                }
                None => {
                    self.channel_name.push_str(&text);
                    // Same bound, and for the same reason, as a recipient's:
                    // withholding is what has to stay bounded. A name this
                    // long is not a name, so it is released as content.
                    if self.channel_name.len() <= MAX_HEADER_LEN {
                        return None;
                    }
                    self.naming_channel = false;
                    self.hidden_body = false;
                    text = std::mem::take(&mut self.channel_name);
                }
            }
        }
        if !self.inside {
            if self.hidden_body || text.is_empty() {
                return None;
            }
            self.emitted_body = true;
            if std::mem::take(&mut self.owed_separator) {
                text.insert_str(0, "\n\n");
            }
            return Some(text);
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
    prefill_batch_override().unwrap_or(PREFILL_BATCH_DEFAULT)
}

/// `ORANGU_PREFILL_BATCH` if it was set, `None` if it was not — the
/// distinction [`flat_width`] needs and [`prefill_batch`] throws away. An
/// operator who wrote `512` gets 512 everywhere; one who wrote nothing gets a
/// width chosen for the regime.
fn prefill_batch_override() -> Option<usize> {
    static BATCH: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *BATCH.get_or_init(|| {
        std::env::var("ORANGU_PREFILL_BATCH")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
    })
}

/// How much of the model must be in RAM before an extra prefill pass is
/// cheap.
///
/// Not a tuning knob so much as a statement about which of two measured
/// regimes a run is in. Above it an extra pass costs locality; below it an
/// extra pass costs a re-read of everything missing, which is the whole model
/// when residency is poor.
const RESIDENT_ENOUGH: f64 = 0.95;

/// The width one flat-policy prefill should use.
///
/// **The two regimes disagree, and both were measured.** With the model
/// resident, a narrow chunk is faster *and* smaller — a smaller working set
/// pages less — and the ordering is monotone in width. With the model
/// streamed from disk, every extra pass re-reads what is not resident, so the
/// ordering reverses and one pass wins by a factor rather than a percentage.
/// No single constant is right for both, so this asks which regime it is in.
///
/// Two things keep the question cheap. A prompt that fits the narrow default
/// is one pass at any width, so the answer cannot matter and is not asked
/// for — and that is every short prompt, while the probe is `mincore` over
/// the whole model. And residency that cannot be established keeps **today's**
/// width, on the same principle as [`Backend::has_submission_timeout`]'s
/// default: a component that cannot answer must not be read as having
/// answered.
///
/// [`Backend::has_submission_timeout`]: crate::engine::backend::Backend::has_submission_timeout
fn flat_width(prompt_tokens: usize) -> usize {
    let configured = prefill_batch_override();
    // Short-circuited before the probe, not inside it: `resident_fraction` is
    // `mincore` over every shard, and a prompt this size does not care.
    let resident = (configured.is_none() && prompt_tokens > PREFILL_BATCH_DEFAULT)
        .then(resident_fraction)
        .flatten();
    flat_width_for(configured, prompt_tokens, resident)
}

/// The mapping alone, so each direction can be tested without a process-wide
/// `OnceLock` a test cannot then undo — the same shape as [`policy_for`], and
/// for the same reason: getting this backwards is a factor, not a percentage.
fn flat_width_for(
    configured: Option<usize>,
    prompt_tokens: usize,
    resident_fraction: Option<f64>,
) -> usize {
    match (configured, resident_fraction) {
        // An operator who named a width gets it, in either regime.
        (Some(width), _) => width,
        // One pass at any width: the regime cannot change the pass count.
        _ if prompt_tokens <= PREFILL_BATCH_DEFAULT => PREFILL_BATCH_DEFAULT,
        // Streamed: every extra pass re-reads what is not resident.
        (None, Some(fraction)) if fraction < RESIDENT_ENOUGH => 0,
        // Resident, or residency unknowable — keep today's width.
        (None, _) => PREFILL_BATCH_DEFAULT,
    }
}

/// What fraction of the model's bytes are in RAM, or `None` if that cannot be
/// established for every shard — a partial answer would understate residency
/// and read as a colder run than happened.
fn resident_fraction() -> Option<f64> {
    let shards = crate::engine::page_cache::residency();
    let (bytes, resident) = crate::engine::page_cache::residency_totals(&shards);
    if bytes == 0 {
        return None;
    }
    resident.map(|r| r as f64 / bytes as f64)
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
/// How a prefill splits a prompt across forward passes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ChunkPolicy {
    /// Size each chunk from the last one's measured rate, aiming at
    /// [`prefill_chunk_budget`]. For a backend whose driver resets a device
    /// that takes too long over one submission — see
    /// [`Backend::has_submission_timeout`].
    ///
    /// [`Backend::has_submission_timeout`]: crate::engine::backend::Backend::has_submission_timeout
    Adaptive,
    /// Every chunk the configured width. For a backend with no submission
    /// timeout, where the wall clock carries no information the sizer can
    /// use: a streamed model's per-pass cost is fixed rather than per-token,
    /// so adapting to it shrinks the chunk and multiplies the reads. The
    /// width still bounds a pass's scratch; nothing infers a rate.
    Flat,
}

/// How one prefill splits: how wide a chunk may be, and whether that width is
/// a starting point or the whole rule.
///
/// The two travel together because neither means anything alone — a width with
/// no policy does not say whether it is a ceiling or a probe, and a policy with
/// no width has nothing to bound a pass's scratch by.
#[derive(Clone, Copy, Debug)]
struct Chunking {
    /// Tokens per forward pass. `0` opts out of chunking entirely — see
    /// [`prefill_batch`].
    width: usize,
    policy: ChunkPolicy,
}

impl Chunking {
    /// How this process should split a prompt of `prompt_tokens`.
    ///
    /// The width is regime-chosen only under [`ChunkPolicy::Flat`]. Under
    /// `Adaptive` it stays a ceiling the sizer works below, and widening it
    /// would hand the driver exactly the long submission the chunker exists
    /// to prevent.
    fn for_prompt(prompt_tokens: usize) -> Self {
        let policy = chunk_policy();
        let width = match policy {
            ChunkPolicy::Adaptive => prefill_batch(),
            ChunkPolicy::Flat => flat_width(prompt_tokens),
        };
        Self { width, policy }
    }
}

/// Whether the backend this process selected can lose its device to a
/// submission timeout, recorded once at startup by [`set_chunk_policy`].
///
/// A process-wide fact rather than a parameter because the prefill call site
/// holds a `&dyn ModelForward` and no backend handle, and threading one down
/// would touch every architecture — the thing this whole design is trying not
/// to do. Unset means [`ChunkPolicy::Adaptive`]: a test, a benchmark harness
/// or any caller that never registered gets today's behaviour.
static CHUNK_POLICY: std::sync::OnceLock<ChunkPolicy> = std::sync::OnceLock::new();

/// Records what the selected backend answered for
/// [`Backend::has_submission_timeout`]. Called once, from `main`, right after
/// the backend is chosen.
///
/// [`Backend::has_submission_timeout`]: crate::engine::backend::Backend::has_submission_timeout
pub fn set_chunk_policy(has_submission_timeout: bool) {
    let _ =
        CHUNK_POLICY.set(env_chunk_policy().unwrap_or_else(|| policy_for(has_submission_timeout)));
}

/// `ORANGU_CHUNK_POLICY=adaptive|flat`, the override that lets a sweep put
/// both policies on one backend and compare them.
///
/// Without it the policy is a function of the hardware, so the two arms of the
/// A/B would have to be two machines — and a cross-machine A/B is not a
/// measurement. Anything unrecognised, including an empty value, is ignored
/// rather than treated as one of the two: a typo in a harness must not
/// silently select an arm.
fn env_chunk_policy() -> Option<ChunkPolicy> {
    match std::env::var("ORANGU_CHUNK_POLICY")
        .ok()?
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "adaptive" => Some(ChunkPolicy::Adaptive),
        "flat" => Some(ChunkPolicy::Flat),
        _ => None,
    }
}

/// The mapping alone, so the direction can be tested without writing the
/// process-wide `OnceLock` a test cannot then undo. Getting it backwards
/// costs a device.
fn policy_for(has_submission_timeout: bool) -> ChunkPolicy {
    if has_submission_timeout {
        ChunkPolicy::Adaptive
    } else {
        ChunkPolicy::Flat
    }
}

/// The registered policy, defaulting to the safe one.
fn chunk_policy() -> ChunkPolicy {
    *CHUNK_POLICY.get().unwrap_or(&ChunkPolicy::Adaptive)
}

fn prefill(
    model: &dyn ModelForward,
    cache: &mut KvCache,
    tokens: &[u32],
    start_pos: usize,
    slot_id: usize,
    on_chunk: &mut dyn FnMut(usize),
) -> Result<Vec<f32>> {
    // Priced once and carried forward: see [`CHUNK_COST`].
    let mut cost = load_chunk_cost();
    let out = prefill_in_chunks(
        &mut cost,
        model,
        cache,
        tokens,
        start_pos,
        slot_id,
        Chunking::for_prompt(tokens.len()),
        on_chunk,
    );
    // Written back on the error path too: a chunk that ran is a chunk that
    // was measured, and what it cost is true whether or not a later one
    // failed.
    store_chunk_cost(cost);
    out
}

/// What prefill has learned about the cost of one submission here, carried
/// across requests.
///
/// The curve belongs to this machine and this model, not to one request, so
/// re-deriving it per request throws away an answer already paid for — and
/// that is what made every prompt open with a probe and climb its way back to
/// a workable width. One `orangu-server` process serves one model, so one
/// estimate describes every prefill it will ever run.
static CHUNK_COST: Mutex<ChunkCost> = Mutex::new(ChunkCost::new());

/// The carried estimate, or a fresh one if another thread poisoned the lock.
///
/// A poisoned lock costs a ramp, not a wrong answer: the worst an unusable
/// estimate can do is start the next prompt at the probe, which is where every
/// prompt started before this existed.
fn load_chunk_cost() -> ChunkCost {
    if chunk_cost_fit_disabled() {
        return ChunkCost::proportional_only();
    }
    *CHUNK_COST
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Whether `ORANGU_NO_CHUNK_COST_FIT` asked for the single-point sizer back.
///
/// The control arm for the two-point fit, so the change can be measured
/// against what it replaced in one `orangu-bench --sweep` rather than across
/// two sessions, where this machine's drift is larger than the effect.
fn chunk_cost_fit_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| crate::engine::env::flag_on("ORANGU_NO_CHUNK_COST_FIT"))
}

fn store_chunk_cost(cost: ChunkCost) {
    *CHUNK_COST
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = cost;
}

/// [`prefill`] with the chunk size passed in rather than read from the
/// environment, so the splitting itself is testable — the property that
/// matters (every token fed exactly once, in order, at the right position)
/// is the one that silently corrupts a KV cache when it's wrong.
#[allow(clippy::too_many_arguments)]
fn prefill_in_chunks(
    cost: &mut ChunkCost,
    model: &dyn ModelForward,
    cache: &mut KvCache,
    tokens: &[u32],
    start_pos: usize,
    slot_id: usize,
    chunking: Chunking,
    on_chunk: &mut dyn FnMut(usize),
) -> Result<Vec<f32>> {
    let Chunking {
        width: batch,
        policy,
    } = chunking;
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
    //
    // Under `ChunkPolicy::Flat` there is nothing to probe for: the probe
    // exists to price a submission against a driver limit this backend does
    // not have, and pricing a streamed model by the clock is what shrinks the
    // chunk into a read spiral. Start at the full width and stay there.
    let mut width = match policy {
        ChunkPolicy::Adaptive => cost
            .opening_width(start_pos, budget, batch)
            .unwrap_or(PREFILL_PROBE_TOKENS.min(batch)),
        ChunkPolicy::Flat => batch,
    };
    // One line per prefill, not one per submission — which is the whole point.
    // `ORANGU_PREFILL_TRACE` answers a different question and answers it by
    // writing to stderr inside the submission loop, so it changes the cost it
    // is measuring: under it this sizer reads inflated chunk costs and picks
    // narrower chunks, which is a real effect on the number being checked.
    // Recording the widths and printing them once afterwards costs nothing
    // measurable and is enough to see what the sizer actually chose.
    let chunks_report = chunk_widths_reported();
    let mut widths: Vec<usize> = Vec::new();
    while done < tokens.len() {
        let n = width.min(tokens.len() - done);
        let started = Instant::now();
        logits = model.forward(cache, &tokens[done..done + n], pos, slot_id)?;
        let elapsed = started.elapsed();
        pos += n;
        done += n;
        // After the forward, not before: progress means work finished.
        on_chunk(done);
        if chunks_report {
            widths.push(n);
        }
        if policy == ChunkPolicy::Adaptive {
            cost.observe(n, elapsed);
            width = cost.next_width(budget, batch);
        }
    }
    if chunks_report {
        let total: usize = widths.iter().sum();
        eprintln!(
            "orangu-server: [prefill-chunks] {} tokens from {} in {} submissions: {:?}",
            total,
            start_pos,
            widths.len(),
            widths
        );
    }
    Ok(logits)
}

/// Whether `ORANGU_PREFILL_CHUNKS=1` asked for one line per prefill naming the
/// widths the sizer chose.
fn chunk_widths_reported() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::engine::env::flag_on("ORANGU_PREFILL_CHUNKS"))
}

/// How wide the next chunk should be, reading one chunk's cost as if all of it
/// were per-token work.
///
/// This is the estimate available before two observations exist to separate a
/// submission's fixed cost from its marginal one ([`ChunkCost`]), and the one
/// deliberately used again whenever a chunk overruns its budget — charging
/// everything to per-token work under-sizes, and under-sizing is the safe
/// direction.
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
    clamp_chunk_width(
        budget.as_secs_f64() * PREFILL_BUDGET_HEADROOM / per_token,
        max_width,
    )
}

/// What one prefill submission costs on this machine, as a line through its
/// token count: `fixed + per_token · n` seconds.
///
/// A chunk's cost is not proportional to its width. A submission pays for
/// itself before it processes a single token — the encoder, the bind groups,
/// the host round trip, and the per-layer dispatches whose grids do not shrink
/// with the batch — and only then pays per token. Sizing from `elapsed / n`,
/// which is what this did while it modelled cost as pure proportion, reads all
/// of that fixed cost as if it were per-token work. The error is worst exactly
/// where the sizing starts: the opening probe is almost entirely fixed cost,
/// so its apparent per-token rate comes out several times the real one and the
/// next chunk is sized at a fraction of what the budget allows. The width then
/// climbs one chunk at a time, and on a prompt only a few chunks long the
/// climb *is* the prompt.
///
/// Two observations at different widths separate the two terms. That
/// separation is also what keeps this portable: `fixed` and `per_token` are
/// measured here, against this machine and this model, rather than carried in
/// as constants from whichever machine the code was written on. A card with a
/// cheaper round trip, a model with fewer layers, or a backend with no
/// per-submission cost at all describes itself correctly through the same two
/// numbers.
#[derive(Clone, Copy, Debug, Default)]
struct ChunkCost {
    /// The most recent observation: how wide a chunk was, and what it cost.
    last: Option<(usize, Duration)>,
    /// The line through the two most recent observations wide enough apart to
    /// support one, kept until a better-separated pair replaces it.
    fit: Option<CostFit>,
    /// Never form a fit, and so never offer an opening width — which leaves
    /// the proportional sizer this replaced, exactly. The control arm for
    /// `ORANGU_NO_CHUNK_COST_FIT`, carried in the value rather than read from
    /// the environment down here so the type stays pure and a test can have
    /// either behaviour without touching a process-wide switch.
    proportional_only: bool,
}

/// `fixed + per_token · n` seconds, both non-negative.
#[derive(Clone, Copy, Debug, PartialEq)]
struct CostFit {
    fixed: f64,
    per_token: f64,
}

/// How much wider than its partner the wider of two observations must be
/// before the pair is allowed to set a slope.
///
/// Two points near the same width put a small, noisy denominator under the
/// subtraction — the slope they imply is mostly the difference between two
/// timings of the same thing. Requiring real separation costs nothing, because
/// the widths this sizer produces vary by much more than this on the way up,
/// and a pair that fails the test leaves the previous fit standing rather than
/// replacing it with a coincidence.
const COST_FIT_SEPARATION: usize = 2;

impl CostFit {
    /// The line through two observations, or `None` when the pair cannot
    /// support one.
    ///
    /// Rejects a pair that is too close together to carry a slope, and one
    /// where the wider chunk did not cost more — which is noise, not a
    /// negative marginal cost, and would size the next chunk off a downward
    /// line.
    fn through(a: (usize, Duration), b: (usize, Duration)) -> Option<Self> {
        let (narrow, wide) = if a.0 <= b.0 { (a, b) } else { (b, a) };
        let (n_narrow, n_wide) = (narrow.0, wide.0);
        if n_narrow == 0 || n_wide < n_narrow.saturating_mul(COST_FIT_SEPARATION) {
            return None;
        }
        let (t_narrow, t_wide) = (narrow.1.as_secs_f64(), wide.1.as_secs_f64());
        let per_token = (t_wide - t_narrow) / ((n_wide - n_narrow) as f64);
        if !per_token.is_finite() || per_token <= 0.0 {
            return None;
        }
        // Clamped rather than rejected: a measured intercept below zero means
        // the narrow point sat above the line, which is ordinary scatter, and
        // the slope it came with is still the useful half of the answer.
        let fixed = (t_wide - per_token * n_wide as f64).max(0.0);
        if !fixed.is_finite() {
            return None;
        }
        Some(Self { fixed, per_token })
    }

    /// This line moved to pass through `point`, keeping its fixed term.
    ///
    /// The marginal cost is whatever is left of the observation once the fixed
    /// cost is taken off it. Floored above zero rather than at it: a point
    /// that lands under the intercept — ordinary scatter on a narrow chunk —
    /// would otherwise produce a zero or negative slope and size the next
    /// chunk at the ceiling on the strength of one cheap measurement.
    fn through_holding_fixed(self, (n, elapsed): (usize, Duration)) -> Self {
        if n == 0 {
            return self;
        }
        let marginal = (elapsed.as_secs_f64() - self.fixed) / n as f64;
        if !marginal.is_finite() || marginal <= 0.0 {
            return self;
        }
        Self {
            fixed: self.fixed,
            per_token: marginal,
        }
    }

    /// The widest chunk whose predicted cost still fits `target` seconds.
    fn fits_in(&self, target: f64) -> f64 {
        let room = target - self.fixed;
        if room <= 0.0 {
            return 0.0;
        }
        room / self.per_token
    }
}

impl ChunkCost {
    /// An estimate that has measured nothing yet.
    const fn new() -> Self {
        Self {
            last: None,
            fit: None,
            proportional_only: false,
        }
    }

    /// An estimate that will never fit — the control arm, which restores the
    /// single-point sizer this replaced.
    const fn proportional_only() -> Self {
        Self {
            last: None,
            fit: None,
            proportional_only: true,
        }
    }

    /// Record what a chunk of `n` tokens actually cost.
    fn observe(&mut self, n: usize, elapsed: Duration) {
        let point = (n, elapsed);
        if !self.proportional_only {
            match self.last.and_then(|prev| CostFit::through(prev, point)) {
                // Two well-separated widths price both terms.
                Some(fit) => self.fit = Some(fit),
                // One width cannot, but it must still be allowed to move the
                // estimate, and this is the case that occurs *most* — once the
                // sizer settles at a width it keeps choosing it, so every
                // later pair is the same width twice and can never separate.
                // Leaving the fit untouched there freezes it: whatever the
                // first two chunks happened to say becomes permanent, a single
                // unlucky pair sizes every prompt the server will ever serve,
                // and because the frozen width is then the only width chosen,
                // nothing can ever dislodge it. Re-derive the marginal cost
                // through the new point instead, holding the fixed cost — the
                // decomposition is what makes that sound. Fixed cost is a
                // property of the submission and is stable; marginal cost
                // climbs with context, so it is the term that has to track.
                None => {
                    if let Some(fit) = self.fit {
                        self.fit = Some(fit.through_holding_fixed(point));
                    }
                }
            }
        }
        self.last = Some(point);
    }

    /// How wide the next chunk of the prompt being processed should be.
    fn next_width(&self, budget: Duration, max_width: usize) -> usize {
        let target = budget.as_secs_f64() * PREFILL_BUDGET_HEADROOM;
        // No separate "did it overrun?" branch, because the fit already
        // shrinks on one and cannot fail to. Every observation moves the line
        // through the newest point, so a chunk of `n` tokens that cost `t`
        // sizes the next at `n · (target − fixed) / (t − fixed)`, which is
        // below `n` exactly when `t` exceeds the target — which is what an
        // overrun is. A guard in front of that would be safety code no test
        // could ever reach: it would read as a promise, and the honest way to
        // keep this one is the arithmetic above rather than a branch nothing
        // exercises.
        match (self.fit, self.last) {
            (Some(fit), _) => clamp_chunk_width(fit.fits_in(target), max_width),
            // One observation cannot separate the terms. Charging all of it to
            // per-token work is the conservative reading — it under-sizes, and
            // under-sizing costs a chunk where over-sizing costs a device.
            (None, Some((n, elapsed))) => next_chunk_width(n, elapsed, budget, max_width),
            (None, None) => max_width,
        }
    }

    /// How wide to open a *new* prompt, for a machine already priced by an
    /// earlier one — or `None` to open with the probe.
    ///
    /// Offered only from position zero. The probe exists to keep the opening
    /// submission small at a depth where a token costs many times what it did
    /// at the start, and an estimate carried in from an earlier request knows
    /// nothing about the depth this one begins at. So the continuation path —
    /// the one the probe was built for, and the one that used to reset the
    /// GPU — keeps probing, and only a prompt starting from nothing opens at
    /// the width this machine has already been shown to sustain.
    fn opening_width(&self, start_pos: usize, budget: Duration, max_width: usize) -> Option<usize> {
        if start_pos != 0 {
            return None;
        }
        self.fit.map(|_| self.next_width(budget, max_width))
    }
}

/// The bounds every adapted width lives inside, in one place.
///
/// `min(max_width)` on the floor as well: a caller that configured a batch
/// smaller than the floor asked for chunks that small, and `clamp` panics
/// outright when its own min exceeds its max.
fn clamp_chunk_width(fits: f64, max_width: usize) -> usize {
    let fits = if fits.is_finite() && fits >= 0.0 {
        fits as usize
    } else {
        0
    };
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
    // The draft model runs the same forward passes over the same prompt, so it
    // wants the same width for the same reasons — it just never adapted.
    let batch = Chunking::for_prompt(tokens.len()).width;
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
    /// The framing of the third format (gemma-4's `<|channel>` /
    /// `<channel|>`).
    const CHANNEL_OPEN: u32 = 6;
    const CHANNEL_CLOSE: u32 = 7;

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
            inside: resumes_in_header,
            suppress_reasoning,
            ..Default::default()
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
            suppress_reasoning: true,
            ..Default::default()
        };
        assert_eq!(header.observe_marker(START), None);
        assert_eq!(header.observe_marker(MESSAGE), None);
        assert_eq!(
            header.observe_text("plain".to_string()),
            Some("plain".to_string())
        );
    }

    /// Everything a client would see from a gemma-4 reply, where a side
    /// channel is framed by `<|channel>` / `<channel|>` and names itself in
    /// the text right after the opener.
    fn shown_in_channels(suppress_reasoning: bool, stream: &[Out]) -> String {
        let mut header = MessageHeader {
            channel: Some((CHANNEL_OPEN, CHANNEL_CLOSE)),
            suppress_reasoning,
            ..Default::default()
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

    /// The channel *name* is framing and never reaches the reader, whatever
    /// the role. This is the bug that ended replies with a bare
    /// `"tool_code\nprint(create_directory(path='pacman_game'))"`.
    #[test]
    fn a_channel_name_is_never_shown() {
        let turn = [
            Text("Here is the plan."),
            Marker(CHANNEL_OPEN),
            Text("tool_code\nprint(create_directory(path='x'))"),
            Marker(CHANNEL_CLOSE),
        ];
        assert_eq!(
            shown_in_channels(false, &turn),
            "Here is the plan.\n\nprint(create_directory(path='x'))"
        );
        assert_eq!(shown_in_channels(true, &turn), "Here is the plan.");
    }

    /// A channel body is a reasoning message, so it follows the same rule as
    /// the other two formats' — shown separated from the answer, or dropped
    /// entirely when the role suppresses reasoning.
    #[test]
    fn a_channel_body_follows_the_reasoning_rule() {
        let turn = [
            Marker(CHANNEL_OPEN),
            Text("thought\nLet me think."),
            Marker(CHANNEL_CLOSE),
            Text("23, 29, 31."),
        ];
        assert_eq!(
            shown_in_channels(false, &turn),
            "Let me think.\n\n23, 29, 31."
        );
        assert_eq!(shown_in_channels(true, &turn), "23, 29, 31.");
    }

    /// The name arrives one token at a time like everything else, and the
    /// token that ends it can carry the first of the body with it.
    #[test]
    fn a_channel_name_split_across_tokens_is_still_dropped() {
        let turn = [
            Marker(CHANNEL_OPEN),
            Text("tho"),
            Text("ught"),
            Text("\nLet me"),
            Text(" think."),
            Marker(CHANNEL_CLOSE),
            Text("Done."),
        ];
        assert_eq!(shown_in_channels(false, &turn), "Let me think.\n\nDone.");
    }

    /// A channel whose name never ends must not swallow the rest of the
    /// reply: past any name's length, what was withheld is content. Driven
    /// directly rather than through the helper, because the text it needs is
    /// longer than a literal worth writing out.
    #[test]
    fn an_unterminated_channel_name_releases_what_it_withheld() {
        let mut header = MessageHeader {
            channel: Some((CHANNEL_OPEN, CHANNEL_CLOSE)),
            ..Default::default()
        };
        assert_eq!(header.observe_marker(CHANNEL_OPEN), None);
        let long = "x".repeat(super::MAX_HEADER_LEN + 1);
        assert_eq!(header.observe_text(long.clone()), Some(long));
    }

    /// A reply that stops right after a channel gets no separator hung off
    /// the end of it.
    #[test]
    fn a_reply_ending_in_a_channel_has_no_trailing_separator() {
        let turn = [
            Text("Answer."),
            Marker(CHANNEL_OPEN),
            Text("thought\nafterthought"),
            Marker(CHANNEL_CLOSE),
        ];
        assert_eq!(shown_in_channels(true, &turn), "Answer.");
    }

    /// Everything a client would see from an `inkling`-style reply, where
    /// each body is opened by a marker naming its kind and there is no
    /// header at all.
    fn shown_by_kind(suppress_reasoning: bool, stream: &[Out]) -> String {
        let mut header = MessageHeader {
            kinds: Some(crate::engine::tokenizer::ContentKinds {
                reasoning: THINKING,
                other: vec![TEXT],
            }),
            suppress_reasoning,
            ..Default::default()
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
            prefix_cache,
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

        let logits = prefill_in_chunks(
            &mut ChunkCost::new(),
            &model,
            &mut cache,
            &tokens,
            0,
            0,
            Chunking {
                width: 10,
                policy: ChunkPolicy::Adaptive,
            },
            &mut |_| {},
        )
        .unwrap();

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

    /// `Flat` does not probe. The probe exists to price one submission
    /// against a driver limit, and on a backend with no such limit it costs
    /// an extra pass over the whole model for nothing — which on a streamed
    /// model is the entire expense of a prefill.
    ///
    /// Asserted against `Adaptive` on the same input rather than in
    /// isolation, because the property is a *difference*: if both policies
    /// ever produce the same call sequence, this change did nothing.
    #[test]
    fn the_flat_policy_does_not_open_with_a_probe() {
        let widths = |policy| {
            let model = RecordingModel::new();
            let mut cache = model.new_kv_cache(128);
            let tokens: Vec<u32> = (0..40).collect();
            prefill_in_chunks(
                &mut ChunkCost::new(),
                &model,
                &mut cache,
                &tokens,
                0,
                0,
                Chunking { width: 32, policy },
                &mut |_| {},
            )
            .unwrap();
            let calls = model.calls.lock().unwrap().clone();
            calls.into_iter().map(|(t, _)| t.len()).collect::<Vec<_>>()
        };

        assert_eq!(widths(ChunkPolicy::Flat), vec![32, 8]);
        assert_eq!(
            widths(ChunkPolicy::Adaptive)[0],
            PREFILL_PROBE_TOKENS,
            "the adaptive path must still probe — this test is only \
             meaningful while the two policies differ"
        );
    }

    /// Dropping the adaptation must not drop the contract every prefill has:
    /// each token fed exactly once, in order, at its absolute position. This
    /// is the property whose violation corrupts a KV cache silently instead
    /// of failing, so it is asserted separately for each policy.
    #[test]
    fn the_flat_policy_still_feeds_every_token_once_in_order() {
        let model = RecordingModel::new();
        let mut cache = model.new_kv_cache(128);
        let tokens: Vec<u32> = (0..25).collect();

        prefill_in_chunks(
            &mut ChunkCost::new(),
            &model,
            &mut cache,
            &tokens,
            100,
            0,
            Chunking {
                width: 10,
                policy: ChunkPolicy::Flat,
            },
            &mut |_| {},
        )
        .unwrap();

        let calls = model.calls.lock().unwrap().clone();
        assert_eq!(
            calls,
            vec![
                ((0..10).collect::<Vec<u32>>(), 100),
                ((10..20).collect::<Vec<u32>>(), 110),
                ((20..25).collect::<Vec<u32>>(), 120),
            ]
        );
    }

    /// The direction of the mapping, which is the one thing here that is
    /// dangerous to get wrong: a backend that *has* a timeout must keep the
    /// chunker adapting, and an unregistered process must behave as though it
    /// does. Everything that is not the CPU backend — including CUDA, OpenCL
    /// and ROCm, none of which `as_wgpu` can distinguish from it — reaches
    /// this with `true`.
    /// The width choice is the one place D9 can still be catastrophically
    /// wrong, and in opposite directions in the two regimes: a narrow width on
    /// a streamed model re-reads the model per chunk (measured: 61x the
    /// bytes), while dropping the bound on a resident model costs tok/s and a
    /// third more peak memory. Both directions are asserted.
    #[test]
    fn a_streamed_model_prefills_in_one_pass() {
        assert_eq!(flat_width_for(None, 100_000, Some(0.5)), 0);
        assert_eq!(flat_width_for(None, 100_000, Some(0.0)), 0);
    }

    #[test]
    fn a_resident_model_keeps_the_narrow_width() {
        assert_eq!(
            flat_width_for(None, 100_000, Some(1.0)),
            PREFILL_BATCH_DEFAULT
        );
        assert_eq!(
            flat_width_for(None, 100_000, Some(RESIDENT_ENOUGH)),
            PREFILL_BATCH_DEFAULT
        );
    }

    /// The fail-safe: a platform where `mincore` is absent or refused must not
    /// be read as having said "streamed".
    #[test]
    fn unknowable_residency_keeps_todays_width() {
        assert_eq!(flat_width_for(None, 100_000, None), PREFILL_BATCH_DEFAULT);
    }

    #[test]
    fn an_explicit_batch_setting_outranks_the_regime() {
        // Including `0`, which is how an operator asks for one pass outright.
        assert_eq!(flat_width_for(Some(2048), 100_000, Some(0.1)), 2048);
        assert_eq!(flat_width_for(Some(2048), 100_000, Some(1.0)), 2048);
        assert_eq!(flat_width_for(Some(0), 100_000, Some(1.0)), 0);
    }

    /// A prompt that fits one pass at the narrow width must not be able to
    /// reach the streamed branch — it is the guard that keeps `mincore` off
    /// the path of every short prompt.
    #[test]
    fn a_short_prompt_ignores_the_regime_entirely() {
        for resident in [None, Some(0.0), Some(1.0)] {
            assert_eq!(
                flat_width_for(None, PREFILL_BATCH_DEFAULT, resident),
                PREFILL_BATCH_DEFAULT,
                "resident={resident:?}"
            );
        }
    }

    #[test]
    fn a_backend_with_a_submission_timeout_keeps_adapting() {
        assert_eq!(policy_for(true), ChunkPolicy::Adaptive);
        assert_eq!(policy_for(false), ChunkPolicy::Flat);
    }

    /// A prompt that starts partway in (the prefix cache reused its head)
    /// keeps counting positions from there — the chunk boundaries are
    /// relative to the prompt, the positions are absolute.
    #[test]
    fn chunking_starts_from_the_reused_prefix_position() {
        let model = RecordingModel::new();
        let mut cache = model.new_kv_cache(64);
        let tokens: Vec<u32> = (0..7).collect();

        prefill_in_chunks(
            &mut ChunkCost::new(),
            &model,
            &mut cache,
            &tokens,
            100,
            0,
            Chunking {
                width: 3,
                policy: ChunkPolicy::Adaptive,
            },
            &mut |_| {},
        )
        .unwrap();

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

    /// The bug this sizer was rebuilt around: a chunk's cost is not
    /// proportional to its width, and reading it as if it were turns the
    /// opening probe's fixed cost into a per-token rate several times the real
    /// one.
    ///
    /// The numbers are a real measurement — a 1120-token prompt on a 35-layer
    /// model, where a 16-token chunk cost 270 ms and a 122-token chunk 733 ms.
    /// Proportional sizing reads the probe at 16.9 ms/token and asks for ~133;
    /// the two-point fit recovers ~200 ms fixed and ~4.4 ms/token and asks for
    /// what the budget actually affords.
    #[test]
    fn two_observations_separate_fixed_cost_from_per_token_cost() {
        let budget = Duration::from_millis(3_000);
        let mut cost = ChunkCost::new();

        cost.observe(16, Duration::from_millis(270));
        let one_point = cost.next_width(budget, 512);

        cost.observe(122, Duration::from_millis(733));
        let two_points = cost.next_width(budget, 512);

        assert!(
            (120..=145).contains(&one_point),
            "one observation cannot do better than proportional: {one_point}"
        );
        assert!(
            two_points > one_point * 3,
            "the fit must reach a working width, not climb to it: \
             {one_point} then {two_points}"
        );
        let fit = cost.fit.expect("two separated observations must fit");
        assert!(
            (0.15..0.25).contains(&fit.fixed),
            "fixed cost {} s off the measured ~0.2 s",
            fit.fixed
        );
    }

    /// The safety property the budget exists for, preserved through the
    /// rewrite: a chunk that overran must shrink the next one. Over-sizing
    /// here is what resets the device, so it is asserted against a *standing*
    /// fit that still calls the wider chunk affordable.
    #[test]
    fn an_overrunning_chunk_shrinks_the_next_one_despite_a_fit() {
        let budget = Duration::from_millis(3_000);
        let mut cost = ChunkCost::new();
        // Two cheap, well-separated chunks: a fit that says 512 is affordable.
        cost.observe(16, Duration::from_millis(120));
        cost.observe(256, Duration::from_millis(600));
        assert_eq!(cost.next_width(budget, 512), 512);

        // Then the cost curve rises underneath it and 512 blows the budget.
        // This repeats the previous width deliberately: two observations that
        // close together cannot separate the two terms, so the *only* thing
        // that can shrink this is the fit tracking through the new point.
        cost.observe(512, Duration::from_millis(9_000));
        let after = cost.next_width(budget, 512);
        assert!(
            after < 256,
            "an overrun must shrink the next chunk, sized {after}"
        );
    }

    /// The estimate must never freeze.
    ///
    /// Once the sizer settles on a width it keeps choosing it, so every later
    /// pair is the same width twice and cannot separate the two terms. If that
    /// case left the fit untouched, whatever the opening pair happened to
    /// measure would size every prompt the process ever served — and since the
    /// frozen width is then the only width chosen, nothing could ever dislodge
    /// it. Measured on hardware before this was handled: a server whose
    /// opening pair was taken under profiling overhead pinned itself to
    /// 117-token chunks for every later request and ran *slower* than carrying
    /// no estimate at all.
    #[test]
    fn a_width_frozen_by_an_unlucky_opening_pair_recovers() {
        let budget = Duration::from_millis(3_000);
        let mut cost = ChunkCost::new();
        // An opening pair measured while something else had the machine: the
        // costs are inflated and the fit they produce is far too pessimistic.
        cost.observe(16, Duration::from_millis(270));
        cost.observe(106, Duration::from_millis(1_900));
        let frozen = cost.next_width(budget, 512);
        assert!(frozen < 200, "the unlucky pair should size low: {frozen}");

        // The interference goes away and that width is now cheap. Every later
        // observation is the *same* width, so nothing can separate the terms —
        // and the width must still climb back toward what the budget affords.
        let mut width = frozen;
        for _ in 0..4 {
            cost.observe(width, Duration::from_millis(300));
            width = cost.next_width(budget, 512);
        }
        assert_eq!(
            width, 512,
            "a frozen width must recover to the ceiling, reached {width}"
        );
    }

    /// A pair of observations too close in width cannot set a slope: the
    /// subtraction that produces it has a small, noisy denominator under it.
    /// Such a pair must leave the previous fit standing rather than replace it.
    #[test]
    fn observations_too_close_together_do_not_set_a_slope() {
        assert_eq!(
            CostFit::through(
                (500, Duration::from_millis(2_000)),
                (512, Duration::from_millis(2_010))
            ),
            None,
            "near-identical widths must not produce a fit"
        );
        // Nor may a wider chunk that somehow cost less — noise, not a negative
        // marginal cost.
        assert_eq!(
            CostFit::through(
                (16, Duration::from_millis(300)),
                (256, Duration::from_millis(200))
            ),
            None,
        );
    }

    /// A machine already priced by an earlier prompt opens the next one at a
    /// working width instead of re-running the ramp — that is the whole point
    /// of carrying the estimate — but only from position zero, because a
    /// carried fit knows nothing about the depth a continuation starts at.
    #[test]
    fn a_priced_machine_opens_a_fresh_prompt_at_width_but_still_probes_deep() {
        let budget = Duration::from_millis(3_000);
        let mut cost = ChunkCost::new();
        cost.observe(16, Duration::from_millis(270));
        cost.observe(122, Duration::from_millis(733));

        let fresh = cost
            .opening_width(0, budget, 512)
            .expect("a priced machine must offer an opening width");
        assert!(fresh > 400, "opened only {fresh} tokens wide");
        assert_eq!(
            cost.opening_width(100_000, budget, 512),
            None,
            "a continuation must still open with the probe"
        );
        // And an unpriced one has nothing to offer, at any position.
        assert_eq!(ChunkCost::new().opening_width(0, budget, 512), None);
    }

    /// End to end through the chunk loop: a cold estimate opens with the probe
    /// and a carried one does not. Asserted on the widths the model was
    /// actually called with, so it covers the wiring and not just the
    /// arithmetic.
    ///
    /// Only the *opening* width is asserted, because only it is decided before
    /// any forward runs. Every later width comes from timing a `RecordingModel`
    /// call, which is a few microseconds of noise on an idle machine and
    /// whatever the scheduler decides on a busy one — an earlier version of
    /// this test asserted the submission *count* and failed intermittently for
    /// exactly that reason. The count belongs to
    /// [`the_carried_estimate_is_what_removes_the_submissions`], which models
    /// the cost curve instead of racing it.
    #[test]
    fn a_carried_estimate_opens_the_next_prompt_without_the_probe() {
        let widths = |cost: &mut ChunkCost| {
            let model = RecordingModel::new();
            let mut cache = model.new_kv_cache(4096);
            let tokens: Vec<u32> = (0..1120).collect();
            prefill_in_chunks(
                cost,
                &model,
                &mut cache,
                &tokens,
                0,
                0,
                Chunking {
                    width: 512,
                    policy: ChunkPolicy::Adaptive,
                },
                &mut |_| {},
            )
            .unwrap();
            let calls = model.calls.lock().unwrap().clone();
            calls.into_iter().map(|(t, _)| t.len()).collect::<Vec<_>>()
        };

        let cold = widths(&mut ChunkCost::new());
        assert_eq!(
            cold[0], PREFILL_PROBE_TOKENS,
            "a cold estimate must still probe: {cold:?}"
        );
        assert_eq!(cold.iter().sum::<usize>(), 1120, "every token fed once");

        // Primed with a fixed pair rather than with whatever the previous run
        // happened to measure, so the opening width is decided by arithmetic.
        let mut primed = ChunkCost::new();
        primed.observe(16, Duration::from_millis(270));
        primed.observe(122, Duration::from_millis(733));
        let warm = widths(&mut primed);
        // 469, not 512: that pair prices the machine at 200 ms fixed and
        // 4.37 ms a token, and 469 is what the budget then affords. The point
        // is that it opens at a working width instead of the 16-token probe.
        assert_eq!(
            warm[0], 469,
            "a priced machine must open at a working width: {warm:?}"
        );
        assert_eq!(warm.iter().sum::<usize>(), 1120);
    }

    /// The submission count, against a cost curve rather than a clock: a
    /// prefill chunk really costs `fixed + per_token · n`, so the sizer is
    /// driven here with the numbers measured on hardware
    /// (`fixed` 297 ms, `per_token` 3.57 ms) instead of by timing a mock.
    ///
    /// This is the claim the change is for, so it is the one worth pinning
    /// deterministically: carrying the estimate turns a 1120-token prompt from
    /// five submissions into three.
    #[test]
    fn the_carried_estimate_is_what_removes_the_submissions() {
        let budget = Duration::from_millis(3_000);
        let cost_of = |n: usize| Duration::from_secs_f64(0.297 + 0.00357 * n as f64);

        // Run the sizer over a 1120-token prompt and report the widths it picks.
        let run = |cost: &mut ChunkCost| {
            let mut widths = Vec::new();
            let mut done = 0usize;
            let mut width = cost
                .opening_width(0, budget, 512)
                .unwrap_or(PREFILL_PROBE_TOKENS.min(512));
            while done < 1120 {
                let n = width.min(1120 - done);
                widths.push(n);
                done += n;
                cost.observe(n, cost_of(n));
                width = cost.next_width(budget, 512);
            }
            widths
        };

        let mut carried = ChunkCost::new();
        let cold = run(&mut carried);
        let warm = run(&mut carried);

        assert_eq!(cold, vec![16, 101, 512, 491], "cold ramp: {cold:?}");
        assert_eq!(warm, vec![512, 512, 96], "warm run: {warm:?}");
        assert!(
            warm.len() < cold.len(),
            "carrying the estimate must cost fewer submissions"
        );
        // And it stays converged: a third prompt does not drift off it.
        assert_eq!(run(&mut carried), vec![512, 512, 96]);
        assert_eq!(cold.iter().sum::<usize>(), 1120);
        assert_eq!(warm.iter().sum::<usize>(), 1120);
    }

    /// A deep continuation is the shape that used to hang the GPU: a full-width
    /// chunk submitted at a position where each token costs ten times what it
    /// did at the start. It must open with the small probe instead.
    #[test]
    fn a_continuation_opens_with_a_probe_not_a_full_width_chunk() {
        let model = RecordingModel::new();
        let mut cache = model.new_kv_cache(200_000);
        let tokens: Vec<u32> = (0..600).collect();

        prefill_in_chunks(
            &mut ChunkCost::new(),
            &model,
            &mut cache,
            &tokens,
            100_000,
            0,
            Chunking {
                width: 512,
                policy: ChunkPolicy::Adaptive,
            },
            &mut |_| {},
        )
        .unwrap();

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

        prefill_in_chunks(
            &mut ChunkCost::new(),
            &model,
            &mut cache,
            &tokens,
            0,
            0,
            Chunking {
                width: 10,
                policy: ChunkPolicy::Adaptive,
            },
            &mut |done| seen.push((done, model.calls.lock().unwrap().len())),
        )
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

        prefill_in_chunks(
            &mut ChunkCost::new(),
            &model,
            &mut cache,
            &tokens,
            0,
            0,
            Chunking {
                width: 512,
                policy: ChunkPolicy::Adaptive,
            },
            &mut |done| seen.push(done),
        )
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

            prefill_in_chunks(
                &mut ChunkCost::new(),
                &model,
                &mut cache,
                &tokens,
                0,
                0,
                Chunking {
                    width: batch,
                    policy: ChunkPolicy::Adaptive,
                },
                &mut |_| {},
            )
            .unwrap();

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
            paged_kv: None,
            metrics: Arc::new(crate::engine::metrics::ServerMetrics::new()),
            draft: None,
            model: Arc::new(PanickingModel::new()),
            tokenizer: Arc::new(letter_tokenizer(8)),
            chat_template_source: None,
            slots: SlotPool::new(1),
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
