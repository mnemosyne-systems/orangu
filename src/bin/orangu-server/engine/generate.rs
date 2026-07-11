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
use std::io::Write;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc::{self, UnboundedReceiver};

use super::arch::{ForwardOutcome, GreedySampleParams, ModelForward};
use super::batch::{BatchCoordinator, BatchDecodeRequest, OwnedGreedySample};
use super::sampling::{Sampler, SamplingParams};
use super::scheduler::SlotPool;
use super::tokenizer::Tokenizer;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
}

#[derive(Clone, Debug)]
pub struct GenerateStats {
    pub prompt_tokens: usize,
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
}

pub enum StreamEvent {
    Token(String),
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
    /// `doc/SERVER_OPUS.md` Section 9's Step 11's cross-sequence GEMM
    /// batching item — `Some` only when `slots.total() > 1` *and*
    /// `ORANGU_BATCH_DECODE=1` is set (`main.rs`'s own comment has the
    /// numbers); a single-slot deployment, or `slots > 1` without the env
    /// var (the default), keeps calling `ModelForward::forward_maybe_
    /// sampling` directly, unchanged.
    ///
    /// **Off by default**, unlike every other Step 9/11 GPU-fused change:
    /// a real, reproducible concurrent-load A/B (4 concurrent 100-token
    /// generations, same `slots` count either way) measured this ~60%
    /// *slower*, not faster (74–78s vs. 48.4–48.5s wall time). Correctness
    /// isn't in question — `engine::arch::gemma`'s own `forward_batch_
    /// decode_matches_independent_forward_calls_*` tests, plus a real
    /// concurrent multi-request run against `E2B` where each request got
    /// back its own, correctly-attributed answer, both confirm that — but
    /// `ModelForward::forward_batch_decode`'s batched matmul steps go
    /// through the generic `Backend::matmul`/`matmul_batch` trait methods,
    /// which always read results back to the CPU between steps (built for
    /// the CPU-orchestrated prefill path, not for staying GPU-resident).
    /// That's roughly six CPU↔GPU round trips per layer for the *whole
    /// batch* combined, vs. `GemmaModel::record_decode_forward`'s **one**
    /// round trip for a whole single-sequence forward pass — the exact
    /// round-trip elimination this project's entire Steps 3–11 effort was
    /// built around, reintroduced here in exchange for weight-bandwidth
    /// amortization that apparently doesn't outweigh it at this batch
    /// size/hardware. A genuinely faster version exists in principle (keep
    /// the batched matmuls GPU-resident across the whole layer loop, the
    /// same way the single-sequence path already does) but wasn't
    /// attempted — this is left available behind the flag, correctness-
    /// verified, for future work rather than deleted.
    pub batch_coordinator: Option<Arc<BatchCoordinator>>,
    /// Which of `--all`/`--code`/`--review`/`--explorer`/`--embedding` this
    /// deployment was started with — read by the HTTP layer for default
    /// sampling parameters, generation-endpoint gating, and (`Review`
    /// only) reasoning suppression. See `config::Role`'s own doc comment.
    pub role: crate::config::Role,
}

impl Engine {
    /// Starts generating in the background (on tokio's blocking pool) and
    /// returns a channel of [`StreamEvent`]s — waits for a free slot first
    /// if every one is already busy.
    pub async fn generate(&self, req: GenerateRequest) -> UnboundedReceiver<StreamEvent> {
        let (tx, rx) = mpsc::unbounded_channel();
        let model = self.model.clone();
        let tokenizer = self.tokenizer.clone();
        let slots = self.slots.clone();
        let batch_coordinator = self.batch_coordinator.clone();

        tokio::spawn(async move {
            let guard = slots.acquire().await;
            let task_tx = tx.clone();
            let result = tokio::task::spawn_blocking(move || {
                run(
                    model.as_ref(),
                    tokenizer.as_ref(),
                    batch_coordinator.as_deref(),
                    &guard,
                    req,
                    task_tx,
                )
            })
            .await;
            if let Err(join_err) = result {
                // The blocking task panicked (e.g. an internal invariant
                // violation) — surface it instead of leaving the client
                // hanging with no terminal event.
                let _ = tx.send(StreamEvent::Error(format!(
                    "generation task failed: {join_err}"
                )));
            }
        });

        rx
    }
}

fn run(
    model: &dyn ModelForward,
    tokenizer: &Tokenizer,
    batch_coordinator: Option<&BatchCoordinator>,
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
    // `Option` (not a plain `KvCache`) so the decode loop can *move* it
    // into a `BatchDecodeRequest` when a `batch_coordinator` is in use —
    // that call crosses to a different thread (whichever one ends up
    // leading this batch), which needs ownership, not a borrow. `.take()`/
    // reassignment stands in for a borrow everywhere else, at zero real
    // cost (this is never actually `None` except mid-swap).
    let mut cache = Some(model.new_kv_cache(capacity));
    let mut sampler = Sampler::new(req.sampling);
    let mut history = req.prompt_tokens.clone();

    let prompt_start = Instant::now();
    let logits = match model.forward(
        cache.as_mut().expect("cache is always Some here"),
        &req.prompt_tokens,
        0,
    ) {
        Ok(l) => l,
        Err(err) => {
            let _ = tx.send(StreamEvent::Error(err.to_string()));
            return Ok(());
        }
    };
    let prompt_time = prompt_start.elapsed();
    // Prefill is never decode-shaped (`n_tokens > 1`), so it never takes a
    // GPU-fused sampling fast path either way — this first sample always
    // runs the plain CPU chain, same as before Step 11's GPU-sampling
    // follow-up existed.
    let mut next = sampler.sample(&logits, &history);

    let generate_start = Instant::now();
    let mut generated = 0usize;
    let finish_reason;
    let mut last_report = Instant::now();
    let mut reported = false;
    loop {
        if generated >= req.max_tokens {
            finish_reason = FinishReason::Length;
            break;
        }
        if req.stop_token_ids.contains(&next) {
            finish_reason = FinishReason::Stop;
            break;
        }
        let text = tokenizer.decode(&[next]);
        history.push(next);
        generated += 1;
        guard.set_generated_tokens(generated);
        if tx.send(StreamEvent::Token(text)).is_err() {
            // Receiver dropped (client disconnected) — stop generating.
            return Ok(());
        }
        if last_report.elapsed() >= Duration::from_secs(1) {
            let partial = GenerateStats {
                prompt_tokens: req.prompt_tokens.len(),
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
        // `doc/SERVER_OPUS.md` Section 9's Step 11's GPU-sampling
        // follow-up: when the sampler is greedy, let the model pick the
        // next token itself (a GPU-fused argmax, for backends that have
        // one) instead of always reading back the full `[n_vocab]` logits
        // vector just to immediately re-derive the same argmax on the
        // CPU. `recent_tokens` is trimmed to `repeat_last_n` here, not
        // inside the callee, matching `engine::sampling::
        // apply_repeat_penalty`'s own trim exactly.
        let repeat_last_n = sampler.repeat_last_n();
        let recent_start = history.len().saturating_sub(repeat_last_n);
        let start_pos = history.len() - 1;

        next = if let Some(coordinator) = batch_coordinator {
            // `doc/SERVER_OPUS.md` Section 9's Step 11's cross-sequence
            // GEMM batching item: submit this decode step to the shared
            // coordinator instead of calling `forward_maybe_sampling`
            // directly, so it can be fused with whatever other sequences
            // submit their own next step within the same short window.
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
            match model.forward_maybe_sampling(
                cache
                    .as_mut()
                    .expect("cache is always Some between iterations"),
                &[next],
                start_pos,
                greedy_sample,
            ) {
                Ok(ForwardOutcome::Token(t)) => t,
                Ok(ForwardOutcome::Logits(l)) => sampler.sample(&l, &history),
                Err(err) => {
                    let _ = tx.send(StreamEvent::Error(err.to_string()));
                    return Ok(());
                }
            }
        };
    }
    let generate_time = generate_start.elapsed();

    let stats = GenerateStats {
        prompt_tokens: req.prompt_tokens.len(),
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
