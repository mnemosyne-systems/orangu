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

pub struct Engine {
    pub model: Arc<dyn ModelForward>,
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
        let batch_coordinator = self.batch_coordinator.clone();
        let prefix_cache = self.prefix_cache.clone();
        let slot_store = self.slot_store.clone();

        let id_slot = req.id_slot;
        tokio::spawn(async move {
            let guard = match id_slot {
                Some(index) => slots.acquire_slot(index).await,
                None => slots.acquire().await,
            };
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
                        batch_coordinator.as_deref(),
                        prefix_cache.as_deref(),
                        slot_store.as_deref(),
                        &guard,
                        req,
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

#[allow(clippy::too_many_arguments)]
fn run(
    model: &dyn ModelForward,
    tokenizer: &Tokenizer,
    batch_coordinator: Option<&BatchCoordinator>,
    prefix_cache: Option<&PrefixCache>,
    slot_store: Option<&super::slot_store::SlotStore>,
    guard: &super::scheduler::SlotGuard,
    req: GenerateRequest,
    tx: mpsc::UnboundedSender<StreamEvent>,
) -> Result<()> {
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
            new_cache.copy_prefix_from(&entry.cache, matched);
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
    // Prompt-lookup speculative decoding (opt-in, greedy-only, single-slot).
    // `spec_buf` holds tokens a speculative step already verified and committed
    // to the KV cache, waiting to be emitted before the next forward.
    let speculative =
        speculative_config().filter(|_| sampler.is_greedy() && batch_coordinator.is_none());
    let mut spec_buf: VecDeque<u32> = VecDeque::new();
    let mut spec_accepted = 0usize;
    let mut spec_steps = 0usize;
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
        // Structural tokens (a chat format's turn/channel/tool markers, and
        // any stray BOS/EOS) still go into `history` so the KV cache and any
        // continued generation stay correct, but are not rendered to the
        // user — otherwise a gemma-4 reply spills literal `<turn|>`/
        // `<channel|>` tokens into the stream (`skip_special_tokens`).
        let emitted = if tokenizer.is_special(next) {
            // Almost every special token is structural and stays hidden. The
            // exception is a tool-call marker: the model writes its call
            // *between* special tokens, so suppressing them would leave the
            // arguments as loose prose with nothing to say they were a call.
            // Rendered back to literal text here and read by
            // `engine::tool_calls`; the HTTP layer removes the whole span
            // from what the user sees, so this never reaches a chat client
            // as content.
            tokenizer
                .token_text(next)
                .and_then(tool_calls::marker_text)
                .map(str::to_string)
        } else {
            Some(tokenizer.decode(&[next]))
        };
        if let Some(text) = emitted
            && tx.send(StreamEvent::Token(text)).is_err()
        {
            // Receiver dropped (client disconnected) — stop generating.
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
        } else if let Some((ngram, max_draft)) = speculative {
            // Draft a continuation from the context and verify it in one
            // multi-position forward; returns the model's own next token (any
            // further accepted tokens are queued in `spec_buf`).
            match speculative_next(
                model,
                cache
                    .as_mut()
                    .expect("cache is always Some between iterations"),
                &mut sampler,
                &history,
                next,
                start_pos,
                ngram,
                max_draft,
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
                greedy_sample: sampler.is_greedy().then(|| OwnedGreedySample {
                    recent_tokens: history[recent_start..].to_vec(),
                    repeat_penalty: sampler.repeat_penalty(),
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
            let greedy_sample = sampler.is_greedy().then(|| GreedySampleParams {
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
        eprintln!(
            "orangu-server: [speculative] {spec_accepted} drafted tokens accepted over \
             {spec_steps} steps ({:.2} extra tokens/forward)",
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
    let _ = tx.send(StreamEvent::Done {
        stats,
        finish_reason,
    });
    Ok(())
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
/// `engine::backend::vulkan`'s own 128-token submission cap regardless of
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
/// the driver's limit to absorb what it misses.
fn next_chunk_width(n: usize, elapsed: Duration, budget: Duration, max_width: usize) -> usize {
    let per_token = elapsed.as_secs_f64() / (n.max(1) as f64);
    if per_token <= 0.0 {
        return max_width;
    }
    let fits = (budget.as_secs_f64() / per_token) as usize;
    // `min(max_width)` on the floor as well: a caller that configured a batch
    // smaller than the floor asked for chunks that small, and `clamp` panics
    // outright when its own min exceeds its max.
    fits.clamp(PREFILL_MIN_CHUNK_TOKENS.min(max_width), max_width)
}

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
    *TRACE.get_or_init(|| std::env::var_os("ORANGU_GPU_TRACE").is_some())
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

/// One prompt-lookup speculative step. Drafts a continuation of `current` from
/// the context, verifies the whole draft in a single multi-position forward,
/// keeps the longest prefix the model would itself have produced greedily, and
/// rolls the rejected tail off the KV cache. Returns the model's own next
/// token (identical to what plain greedy decoding produces here) and pushes any
/// further accepted tokens onto `spec_buf` for the loop to emit before its next
/// forward. `accepted`/`steps` accumulate acceptance stats for the final log.
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
    ngram: usize,
    max_draft: usize,
    slot_id: usize,
    spec_buf: &mut VecDeque<u32>,
    accepted: &mut usize,
    steps: &mut usize,
) -> Result<u32> {
    let draft = ngram_draft(history, ngram, max_draft);
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
    cache.truncate(start_pos + chosen.len());
    *accepted += matched;
    *steps += 1;
    for &t in &chosen[1..] {
        spec_buf.push_back(t);
    }
    Ok(chosen[0])
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
            }
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
            let layer = &mut cache.layers[0];
            for (i, &t) in tokens.iter().enumerate() {
                let val = t as f32 * 1000.0 + (start_pos + i) as f32;
                layer.push(&[val], &[val]);
            }
            let len = layer.len;
            let mut acc = 0f32;
            for p in 0..len {
                acc += layer.key_at(p, 0, 1)[0];
            }
            let winner = (acc.abs() as u64 as usize) % self.config.n_vocab;
            let mut logits = vec![0f32; self.config.n_vocab];
            logits[winner] = 10.0;
            Ok(logits)
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
            prompt_tokens,
            sampling: greedy_params(),
            max_tokens,
            stop_token_ids: vec![],
            cache_prompt,
            id_slot: None,
            timings_per_token: false,
        };
        run(model, tokenizer, None, prefix_cache, None, &guard, req, tx).unwrap();
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
        // 512 tokens took 10.1 s: ~19.7 ms each, so ~152 fit in the budget.
        let next = next_chunk_width(512, Duration::from_millis(10_069), budget, 512);
        assert!((140..=165).contains(&next), "sized {next}");
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
