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

//! Cross-sequence batched decode: the rendezvous that turns several slots'
//! decode steps into one forward pass.
//!
//! Every slot generates on its own thread (`engine::generate`), and a decode
//! step is bound by reading the weights, so two slots stepping one after the
//! other read them twice for two tokens. Measured on the served model: one
//! stream 65 tok/s, two streams 70 aggregate, four 67 — concurrency bought
//! nothing, because the device only ever saw one token at a time.
//!
//! Here, a slot that is decoding **registers**, and each of its steps is
//! offered to the batch. The step runs when every registered slot has
//! offered one: the first thread to see the batch complete runs it for all
//! of them (`ModelForward::forward_decode_batch`, the weights read once) and
//! hands each waiter its logits. A slot that leaves (its answer ended)
//! unregisters, and the waiters re-check — so a batch never waits for
//! something that is not coming. Between steps a slot only samples and
//! streams a token, microseconds against a step's milliseconds, which is why
//! there is no timeout: the join is deterministic.
//!
//! With one slot registered its step runs at once on the solo path,
//! unchanged. A batched step samples as the solo step does: each row's
//! sampling rides along, and an architecture that batches on the device
//! picks each row's token (or its top-k candidates) in the batch's own
//! submission (`ModelForward::forward_decode_batch_sampling`), so a row
//! reads back a token rather than its logits.
//!
//! A slot verifying a speculative draft offers that to the same rendezvous
//! (`DecodeBatcher::verify`): its token and draft become rows of the step,
//! one offer like any other, so the others step with it instead of waiting
//! for its own pass. A model that does not batch a verify
//! (`ModelForward::batches_verify`) is never offered one.
//!
//! **On unless disabled** (`ORANGU_DECODE_BATCH=0` turns it off). The batched
//! step runs every row's projections and feed-forward on the narrow-pass
//! kernels a speculative verify uses, and each row's attention as its solo
//! step would. So a batched answer is the solo answer to the rounding of
//! those kernels, the same floor a verify pass has: greedy output can differ
//! with which requests happen to share its steps.

use std::collections::HashMap;
use std::sync::{Condvar, Mutex};

use anyhow::Result;

use super::arch::{DecodeRow, ForwardOutcome, GreedySampleParams, ModelForward};
use super::kv_cache::KvCache;

pub struct DecodeBatcher {
    inner: Mutex<Inner>,
    joined: Condvar,
    enabled: bool,
}

#[derive(Default)]
struct Inner {
    /// Slots currently decoding — how many steps a batch waits for.
    registered: usize,
    pending: Vec<Pending>,
    results: HashMap<u64, Result<Answer>>,
    /// How long each answered ticket's batch took to compute, its wait for
    /// the other rows left out — see [`DecodeBatcher::take_last_compute`].
    computed: HashMap<u64, std::time::Duration>,
    running: bool,
    next_ticket: u64,
}

/// What a ticket gets back: a step's outcome, or a verify's logits at
/// every position.
enum Answer {
    Step(ForwardOutcome),
    Verify(Vec<Vec<f32>>),
}

struct Pending {
    ticket: u64,
    slot: usize,
    /// One token for a step; the token and its draft for a verify.
    tokens: Vec<u32>,
    verify: bool,
    pos: usize,
    /// The waiter's cache. The waiter is parked on `joined` for as long as
    /// this entry exists, so the leader is the only thread touching it —
    /// that is what makes the pointer usable from another thread.
    cache: *mut KvCache,
    /// The step's sampling, copied: the waiter's history is borrowed from a
    /// thread that is parked, and a window of recent tokens is short.
    sample: Option<OwnedSample>,
}

struct OwnedSample {
    recent_tokens: Vec<u32>,
    repeat_penalty: f32,
    top_k: u32,
}

impl OwnedSample {
    fn of(params: &GreedySampleParams<'_>) -> Self {
        Self {
            recent_tokens: params.recent_tokens.to_vec(),
            repeat_penalty: params.repeat_penalty,
            top_k: params.top_k,
        }
    }

    fn params(&self) -> GreedySampleParams<'_> {
        GreedySampleParams {
            recent_tokens: &self.recent_tokens,
            repeat_penalty: self.repeat_penalty,
            top_k: self.top_k,
        }
    }
}

// `Pending::cache` is only dereferenced while its owner is parked (see the
// field), which is the whole invariant.
unsafe impl Send for Pending {}

/// A slot's membership in the batch while it decodes; dropping it leaves.
pub struct Registration<'a> {
    batcher: &'a DecodeBatcher,
}

impl Drop for Registration<'_> {
    fn drop(&mut self) {
        let mut inner = self.batcher.lock();
        inner.registered = inner.registered.saturating_sub(1);
        // A batch may have been waiting for this slot's step.
        self.batcher.joined.notify_all();
    }
}

impl Default for DecodeBatcher {
    fn default() -> Self {
        Self::new()
    }
}

thread_local! {
    /// The compute time of the batch that last answered this thread's step.
    static LAST_COMPUTE: std::cell::Cell<Option<std::time::Duration>> =
        const { std::cell::Cell::new(None) };
}

impl DecodeBatcher {
    /// On unless `ORANGU_DECODE_BATCH=0` — see the module doc.
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner::default()),
            joined: Condvar::new(),
            enabled: super::env::flag_on_unless_disabled("ORANGU_DECODE_BATCH"),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// How long the last batch this thread's step was answered by took to
    /// compute, without the time the step waited for the other rows — `None`
    /// when the step ran alone. Read once: a later step that runs alone
    /// leaves nothing behind.
    pub fn take_last_compute(&self) -> Option<std::time::Duration> {
        LAST_COMPUTE.with(std::cell::Cell::take)
    }

    /// How many slots a decode step is shared by: those registered, or one
    /// when batching is off.
    pub fn rows(&self) -> usize {
        if self.enabled {
            self.lock().registered.max(1)
        } else {
            1
        }
    }

    /// Joins the batch for the decode loop about to run. `None` when
    /// batching is off, in which case every step takes the solo path.
    pub fn register(&self) -> Option<Registration<'_>> {
        if !self.enabled {
            return None;
        }
        self.lock().registered += 1;
        Some(Registration { batcher: self })
    }

    /// One decode step — this slot's `token` at `pos` into `cache` — as part
    /// of whatever batch forms, or alone when this slot is the only one
    /// decoding. `greedy` asks for the device's sampling, on the solo path
    /// and in a batch alike; where the model cannot, the step returns
    /// logits for the host sampler.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &self,
        registration: Option<&Registration<'_>>,
        model: &dyn ModelForward,
        cache: &mut KvCache,
        token: u32,
        pos: usize,
        greedy: Option<GreedySampleParams<'_>>,
        slot: usize,
    ) -> Result<ForwardOutcome> {
        let solo = |cache: &mut KvCache, greedy: Option<GreedySampleParams<'_>>| {
            // Timed, for the chunk-plan experiment: the whole step, which is
            // the only comparison that charges each plan for exactly what
            // it costs.
            let at = model.decode_step_is_chunked().then(std::time::Instant::now);
            let outcome = model.forward_maybe_sampling(cache, &[token], pos, greedy, slot);
            if let Some(at) = at {
                crate::engine::arch::note_decode_step(at.elapsed());
            }
            outcome
        };
        if registration.is_none() {
            return solo(cache, greedy);
        }
        let entry = |ticket| Pending {
            ticket,
            slot,
            tokens: vec![token],
            verify: false,
            pos,
            cache: cache as *mut KvCache,
            sample: greedy.as_ref().map(OwnedSample::of),
        };
        match self.offer(model, entry) {
            Some(answer) => answer.map(|answer| match answer {
                Answer::Step(outcome) => outcome,
                Answer::Verify(_) => unreachable!("a step is answered as a step"),
            }),
            None => solo(cache, greedy),
        }
    }

    /// A speculative verify — `tokens` (the slot's token and its draft) from
    /// `pos` into `cache` — as rows of whatever batch forms, or as its own
    /// pass (`ModelForward::forward_verify`) when this slot is decoding
    /// alone. Returns every position's logits, with all of `tokens` in the
    /// cache for the caller to roll back past what it accepts.
    pub fn verify(
        &self,
        registration: Option<&Registration<'_>>,
        model: &dyn ModelForward,
        cache: &mut KvCache,
        tokens: &[u32],
        pos: usize,
        slot: usize,
    ) -> Result<Vec<Vec<f32>>> {
        if registration.is_none() || tokens.len() < 2 || !crate::engine::arch::batch_verify() {
            return model.forward_verify(cache, tokens, pos, slot);
        }
        let entry = |ticket| Pending {
            ticket,
            slot,
            tokens: tokens.to_vec(),
            verify: true,
            pos,
            cache: cache as *mut KvCache,
            sample: None,
        };
        match self.offer(model, entry) {
            Some(answer) => answer.map(|answer| match answer {
                Answer::Verify(logits) => logits,
                Answer::Step(_) => unreachable!("a verify is answered as a verify"),
            }),
            None => model.forward_verify(cache, tokens, pos, slot),
        }
    }

    /// Offers `entry` (built with its ticket) to the batch and waits for its
    /// answer, running the batch when this offer completes it. `None` when
    /// this slot is the only one decoding, and the caller runs alone.
    fn offer(
        &self,
        model: &dyn ModelForward,
        entry: impl FnOnce(u64) -> Pending,
    ) -> Option<Result<Answer>> {
        let ticket = {
            let mut inner = self.lock();
            // Alone: nothing to wait for, and nothing to gain.
            if inner.registered <= 1 && inner.pending.is_empty() && !inner.running {
                return None;
            }
            let ticket = inner.next_ticket;
            inner.next_ticket += 1;
            inner.pending.push(entry(ticket));
            ticket
        };
        let mut inner = self.lock();
        loop {
            if let Some(result) = inner.results.remove(&ticket) {
                let took = inner.computed.remove(&ticket);
                LAST_COMPUTE.with(|last| last.set(took));
                return Some(result);
            }
            let complete = !inner.running
                && !inner.pending.is_empty()
                && inner.pending.len() >= inner.registered;
            if complete {
                inner.running = true;
                let batch = std::mem::take(&mut inner.pending);
                drop(inner);
                let started = std::time::Instant::now();
                let results = Self::run_batch(model, batch);
                let took = started.elapsed();
                inner = self.lock();
                for (ticket, _) in &results {
                    inner.computed.insert(*ticket, took);
                }
                inner.results.extend(results);
                inner.running = false;
                self.joined.notify_all();
                continue;
            }
            inner = self.joined.wait(inner).unwrap_or_else(|p| p.into_inner());
        }
    }

    /// Runs one batch — as one forward where the model batches, one pass
    /// per sequence otherwise — and returns each ticket's answer.
    fn run_batch(model: &dyn ModelForward, mut batch: Vec<Pending>) -> Vec<(u64, Result<Answer>)> {
        // The one-token rows first, as a batch with verifies takes them.
        batch.sort_by_key(|p| p.verify);
        let verifies = batch.iter().any(|p| p.verify);
        // Safety: every owner of a cache here is parked on `joined` until
        // its ticket is answered, so these are the only live references.
        let mut rows: Vec<DecodeRow<'_>> = batch
            .iter()
            .map(|p| DecodeRow {
                cache: unsafe { &mut *p.cache },
                pos: p.pos,
                slot: p.slot,
            })
            .collect();
        let samples: Vec<Option<GreedySampleParams<'_>>> = batch
            .iter()
            .map(|p| p.sample.as_ref().map(OwnedSample::params))
            .collect();
        let batched: Result<Option<Vec<Answer>>> = if verifies {
            let tokens: Vec<&[u32]> = batch.iter().map(|p| p.tokens.as_slice()).collect();
            model
                .forward_decode_batch_verify(&mut rows, &tokens, &samples)
                .map(|answers| {
                    answers.map(|answers| {
                        batch
                            .iter()
                            .zip(answers)
                            .map(|(p, mut outcomes)| {
                                if p.verify {
                                    Answer::Verify(
                                        outcomes
                                            .into_iter()
                                            .map(|o| match o {
                                                ForwardOutcome::Logits(l) => l,
                                                _ => unreachable!("a verify's rows are logits"),
                                            })
                                            .collect(),
                                    )
                                } else {
                                    Answer::Step(outcomes.remove(0))
                                }
                            })
                            .collect()
                    })
                })
        } else {
            let tokens: Vec<u32> = batch.iter().map(|p| p.tokens[0]).collect();
            model
                .forward_decode_batch_sampling(&mut rows, &tokens, &samples)
                .map(|outcomes| outcomes.map(|o| o.into_iter().map(Answer::Step).collect()))
        };
        drop(rows);
        match batched {
            Ok(Some(answers)) => batch
                .iter()
                .zip(answers)
                .map(|(p, a)| (p.ticket, Ok(a)))
                .collect(),
            Ok(None) => batch
                .iter()
                .map(|p| {
                    // Safety: as above.
                    let cache = unsafe { &mut *p.cache };
                    let answer = if p.verify {
                        model
                            .forward_verify(cache, &p.tokens, p.pos, p.slot)
                            .map(Answer::Verify)
                    } else {
                        let sample = p.sample.as_ref().map(OwnedSample::params);
                        model
                            .forward_maybe_sampling(cache, &p.tokens, p.pos, sample, p.slot)
                            .map(Answer::Step)
                    };
                    (p.ticket, answer)
                })
                .collect(),
            Err(err) => {
                let msg = format!("{err:#}");
                batch
                    .iter()
                    .map(|p| (p.ticket, Err(anyhow::anyhow!("{msg}"))))
                    .collect()
            }
        }
    }
}
