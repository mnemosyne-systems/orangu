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
//! unchanged — device sampling and all. A batched step comes back as logits
//! and is sampled on the host, which is the same result the device sampler
//! computes.
//!
//! **Opt-in for now** (`ORANGU_DECODE_BATCH=1`). The batched step is correct
//! — three different prompts decoded together give exactly their solo
//! answers — but it runs on the prefill chains, which build every bind
//! group, meta, scratch buffer and norm-weight upload afresh per call, and
//! at two to four rows that host work outweighs the weights saved: two
//! streams measured 48 tok/s aggregate against 54 for one, four streams 66.
//! Caching those resources per layer is what makes it pay, and is the next
//! step.

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
    results: HashMap<u64, Result<Vec<f32>>>,
    running: bool,
    next_ticket: u64,
}

struct Pending {
    ticket: u64,
    slot: usize,
    token: u32,
    pos: usize,
    /// The waiter's cache. The waiter is parked on `joined` for as long as
    /// this entry exists, so the leader is the only thread touching it —
    /// that is what makes the pointer usable from another thread.
    cache: *mut KvCache,
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

impl DecodeBatcher {
    /// Off unless `ORANGU_DECODE_BATCH=1` — see the module doc.
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner::default()),
            joined: Condvar::new(),
            enabled: super::env::flag_on("ORANGU_DECODE_BATCH"),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
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
    /// decoding. `greedy` is honoured on the solo path (the device samples);
    /// a batched step returns logits for the host sampler.
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
        let solo = |cache: &mut KvCache| {
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
            return solo(cache);
        }
        let ticket = {
            let mut inner = self.lock();
            // Alone: nothing to wait for, and nothing to gain.
            if inner.registered <= 1 && inner.pending.is_empty() && !inner.running {
                drop(inner);
                return solo(cache);
            }
            let ticket = inner.next_ticket;
            inner.next_ticket += 1;
            inner.pending.push(Pending {
                ticket,
                slot,
                token,
                pos,
                cache: cache as *mut KvCache,
            });
            ticket
        };
        let mut inner = self.lock();
        loop {
            if let Some(result) = inner.results.remove(&ticket) {
                return result.map(ForwardOutcome::Logits);
            }
            let complete = !inner.running
                && !inner.pending.is_empty()
                && inner.pending.len() >= inner.registered;
            if complete {
                inner.running = true;
                let batch = std::mem::take(&mut inner.pending);
                drop(inner);
                let results = Self::run_batch(model, batch);
                inner = self.lock();
                inner.results.extend(results);
                inner.running = false;
                self.joined.notify_all();
                continue;
            }
            inner = self.joined.wait(inner).unwrap_or_else(|p| p.into_inner());
        }
    }

    /// Runs one batch — as one forward where the model batches, one step
    /// per sequence otherwise — and returns each ticket's logits.
    fn run_batch(model: &dyn ModelForward, batch: Vec<Pending>) -> Vec<(u64, Result<Vec<f32>>)> {
        let tokens: Vec<u32> = batch.iter().map(|p| p.token).collect();
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
        match model.forward_decode_batch(&mut rows, &tokens) {
            Ok(Some(logits)) => batch
                .iter()
                .zip(logits)
                .map(|(p, l)| (p.ticket, Ok(l)))
                .collect(),
            Ok(None) => batch
                .iter()
                .map(|p| {
                    // Safety: as above.
                    let cache = unsafe { &mut *p.cache };
                    (p.ticket, model.forward(cache, &[p.token], p.pos, p.slot))
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
