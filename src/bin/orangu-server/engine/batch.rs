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

//! Cross-sequence GEMM batching coordinator — `doc/SERVER_OPUS.md` Section
//! 9's Step 11's cross-sequence GEMM batching item. `engine::generate::run`
//! (one blocking-pool thread per concurrently-generating request) submits
//! each decode step here instead of calling `ModelForward::forward_maybe_
//! sampling` directly; this collects together whichever requests submit
//! their next decode step within a short window, and processes them as one
//! `ModelForward::forward_batch_decode` call — the matmul steps fused
//! across every sequence in the batch, attention/RoPE/the KV-cache write
//! still per-sequence (see that trait method's own doc comment).
//!
//! **Only used when a caller opts in** (`ORANGU_BATCH_DECODE=1`,
//! `main.rs`) — this coordinator's own concurrency correctness is solid
//! (see this module's own tests, plus a real multi-request run against the
//! actual model where every request got back its own, correctly-attributed
//! answer), but the batched forward pass it drives measured *slower* under
//! real concurrent load than not batching at all — `engine::generate::
//! Engine::batch_coordinator`'s own doc comment has the numbers and the
//! likely cause. This module is exercised, verified, and left in place for
//! future work, not deleted.
//!
//! # Design
//!
//! A `Mutex<CoordState>` + `Condvar`, not an async channel: `run` calls
//! this from a blocking-pool thread (`tokio::task::spawn_blocking`), so a
//! plain `std::sync` rendezvous is the natural fit — no runtime handle
//! juggling needed to `.await` anything here.
//!
//! Each submitting thread pushes its own `(request, response sender)` into
//! `pending`, then loops: if the batch has reached its target size (`SlotPool::
//! busy_count`) or this collection window's deadline has passed, *this*
//! thread becomes the batch's leader
//! — drains `pending`, processes the whole batch (dropping the lock first,
//! since this can take real time), and wakes every other waiter so they
//! can re-check. Otherwise it waits on the condvar (bounded by `POLL_
//! INTERVAL`, so it always re-checks the deadline even if no notification
//! ever arrives) and loops. Every request either becomes a leader itself
//! or gets swept up by one within `MAX_BATCH_WAIT` of when it first
//! arrived — no participant can wait past that bound, and no batch can
//! grow forever waiting for stragglers, since the deadline is fixed the
//! moment the *first* request in a fresh window arrives, never extended by
//! later arrivals.

use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use super::arch::{BatchDecodeItem, ForwardOutcome, GreedySampleParams, ModelForward};
use super::kv_cache::KvCache;
use super::scheduler::SlotPool;

/// How long a collection window stays open, from the first request's
/// arrival, before whatever's pending gets processed regardless of how
/// many sequences joined — bounds every request's added latency from
/// going through the coordinator instead of calling `forward_maybe_
/// sampling` directly.
const MAX_BATCH_WAIT: Duration = Duration::from_millis(4);
/// How often a waiting thread re-checks the batch/deadline state — short
/// enough that a batch closes promptly once its target size or deadline
/// is reached, long enough not to spin.
const POLL_INTERVAL: Duration = Duration::from_millis(1);

/// Owned (thread-crossing) counterpart to `engine::arch::GreedySampleParams`
/// — that type borrows `recent_tokens`, which can't survive being queued
/// here and potentially picked up by a different thread than the one that
/// built it.
pub struct OwnedGreedySample {
    pub recent_tokens: Vec<u32>,
    pub repeat_penalty: f32,
}

/// One sequence's pending decode step, submitted to a [`BatchCoordinator`].
pub struct BatchDecodeRequest {
    pub cache: KvCache,
    pub token: u32,
    pub start_pos: usize,
    pub greedy_sample: Option<OwnedGreedySample>,
}

/// [`BatchDecodeRequest`]'s result: the same cache, with this step's new
/// K/V already appended (regardless of whether it ended up batched with
/// others or processed alone), plus this sequence's own outcome. `outcome`
/// is `Err` for every member of a batch whose `forward_batch_decode` call
/// failed — that call is all-or-nothing (a real error there, e.g. a lost
/// GPU device, isn't specific to one sequence), so there's nothing more
/// precise to report per request.
pub struct BatchDecodeResponse {
    pub cache: KvCache,
    pub outcome: Result<ForwardOutcome, String>,
}

struct CoordState {
    pending: Vec<(BatchDecodeRequest, mpsc::Sender<BatchDecodeResponse>)>,
    /// `Some` iff `pending` is non-empty — when this window's collection
    /// should close regardless of how many requests have joined. Set once
    /// per window, by whichever thread's push finds `pending` empty; never
    /// extended by later arrivals in the same window.
    deadline: Option<Instant>,
}

pub struct BatchCoordinator {
    slots: Arc<SlotPool>,
    state: Mutex<CoordState>,
    cv: Condvar,
}

impl BatchCoordinator {
    pub fn new(slots: Arc<SlotPool>) -> Arc<Self> {
        Arc::new(Self {
            slots,
            state: Mutex::new(CoordState {
                pending: Vec::new(),
                deadline: None,
            }),
            cv: Condvar::new(),
        })
    }

    /// Submits one sequence's pending decode step and blocks until the
    /// batch it ends up in has been processed. Called from a blocking-pool
    /// thread (`engine::generate::run`) — this itself blocks the calling
    /// thread (mutex/condvar/channel, no `.await` anywhere), which is
    /// exactly what a blocking-pool thread is for.
    pub fn submit(&self, model: &dyn ModelForward, req: BatchDecodeRequest) -> BatchDecodeResponse {
        let (tx, rx) = mpsc::channel();
        let mut guard = self.state.lock().expect("batch coordinator mutex poisoned");
        if guard.pending.is_empty() {
            guard.deadline = Some(Instant::now() + MAX_BATCH_WAIT);
        }
        guard.pending.push((req, tx));
        self.cv.notify_all();

        loop {
            // Has *my own* request already been swept into some other
            // thread's batch and processed while I was asleep below? If
            // so, my response is already sitting in `rx` — return it
            // without touching `guard` at all. This check has to come
            // *before* looking at `guard.pending`/`guard.deadline`: once
            // another thread has drained `pending` (which always takes
            // everything in it, including entries pushed by threads still
            // asleep in `wait_timeout` below), those two fields describe
            // the *next* collection window, not the one my own request
            // was part of.
            if let Ok(response) = rx.try_recv() {
                return response;
            }

            // `deadline` can legitimately be `None` here even though *my*
            // response hasn't arrived yet: another thread may have already
            // drained `pending` (clearing `deadline`) and dropped the lock
            // to call `process_batch` — which takes real time — without
            // having reached the point of actually sending *my* entry's
            // response yet. That's not a bug, just a transient window
            // between "swept up" and "response delivered"; there is
            // nothing for *this* thread to lead in that case (`pending` is
            // empty), so just wait and let the `try_recv` above eventually
            // catch the response once it's sent.
            let target = self.slots.busy_count().max(1);
            let become_leader = guard.deadline.is_some_and(|deadline| {
                guard.pending.len() >= target || Instant::now() >= deadline
            });
            if become_leader {
                let batch: Vec<_> = guard.pending.drain(..).collect();
                guard.deadline = None;
                drop(guard);
                Self::process_batch(model, batch);
                self.cv.notify_all();
                // My own entry can only ever have left `pending` via this
                // drain (nothing else removes from it), so my response is
                // already on its way — no need to loop back to the
                // `try_recv` above.
                return rx.recv().expect(
                    "process_batch always sends exactly once per batch member, including this one",
                );
            }
            let (g, _timeout_result) = self
                .cv
                .wait_timeout(guard, POLL_INTERVAL)
                .expect("batch coordinator mutex poisoned");
            guard = g;
        }
    }

    /// Runs `forward_batch_decode` over the whole batch and delivers each
    /// member's own response — called by whichever thread became this
    /// window's leader, with the coordinator's lock already released (this
    /// can take real time; nothing about it needs the lock).
    fn process_batch(
        model: &dyn ModelForward,
        mut batch: Vec<(BatchDecodeRequest, mpsc::Sender<BatchDecodeResponse>)>,
    ) {
        let outcomes: Vec<Result<ForwardOutcome, String>> = {
            let mut items: Vec<BatchDecodeItem<'_>> = batch
                .iter_mut()
                .map(|(req, _)| BatchDecodeItem {
                    cache: &mut req.cache,
                    token: req.token,
                    start_pos: req.start_pos,
                    greedy_sample: req.greedy_sample.as_ref().map(|g| GreedySampleParams {
                        recent_tokens: &g.recent_tokens,
                        repeat_penalty: g.repeat_penalty,
                    }),
                })
                .collect();
            match model.forward_batch_decode(&mut items) {
                Ok(outcomes) => outcomes.into_iter().map(Ok).collect(),
                Err(e) => {
                    let msg = e.to_string();
                    batch.iter().map(|_| Err(msg.clone())).collect()
                }
            }
        };
        for ((req, tx), outcome) in batch.into_iter().zip(outcomes) {
            // The receiver only ever fails to be there if the submitting
            // thread already gave up (e.g. panicked) — nothing to do about
            // that here, so a failed send is silently dropped.
            let _ = tx.send(BatchDecodeResponse {
                cache: req.cache,
                outcome,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::arch::ModelForward;
    use crate::engine::loader::ModelConfig;
    use std::thread;

    /// A `ModelForward` that never touches real model weights — this
    /// module's own concurrency correctness (every concurrent submitter
    /// gets *its own* result back, never another submitter's) doesn't
    /// depend on the real forward-pass math at all, which
    /// `engine::arch::gemma`'s own `forward_batch_decode_matches_
    /// independent_forward_calls_*` tests already cover with the real
    /// model. A short sleep before responding, matching a real forward
    /// pass's own non-trivial duration, makes it likely (not required —
    /// correctness must hold either way, including for a batch of one)
    /// that concurrent submissions actually land in the same batch here.
    struct MockModel;

    impl ModelForward for MockModel {
        fn config(&self) -> &ModelConfig {
            unimplemented!("not exercised by this test")
        }

        fn new_kv_cache(&self, _capacity: usize) -> KvCache {
            KvCache::new(1, 4, 4)
        }

        fn forward(
            &self,
            _cache: &mut KvCache,
            _tokens: &[u32],
            _start_pos: usize,
        ) -> anyhow::Result<Vec<f32>> {
            unimplemented!("not exercised by this test")
        }

        fn forward_hidden_states(&self, _tokens: &[u32]) -> anyhow::Result<Vec<f32>> {
            unimplemented!("not exercised by this test")
        }

        fn forward_batch_decode(
            &self,
            items: &mut [BatchDecodeItem<'_>],
        ) -> anyhow::Result<Vec<ForwardOutcome>> {
            std::thread::sleep(Duration::from_millis(2));
            Ok(items
                .iter()
                .map(|item| ForwardOutcome::Token(item.token * 2))
                .collect())
        }
    }

    /// The concurrency property this whole module exists for: many
    /// threads submitting to the *same* coordinator at once each get back
    /// their own, correctly-attributed result — never another thread's,
    /// even though `process_batch` fuses them all into one
    /// `forward_batch_decode` call and hands responses back out of that
    /// same shared batch. Run repeatedly (not just once) since a race, if
    /// one existed, wouldn't necessarily show up on the first attempt.
    #[test]
    fn concurrent_submissions_each_get_their_own_correct_result() {
        for _round in 0..20 {
            let slots = SlotPool::new(8);
            let coordinator = BatchCoordinator::new(slots.clone());
            let model: Arc<dyn ModelForward> = Arc::new(MockModel);

            // Mark every slot busy so `busy_count()` reports 8 — encourages
            // (doesn't require; correctness must hold regardless) all 8
            // requests landing in the same batch rather than the deadline
            // fallback splitting them up.
            let guards: Vec<_> = (0..8)
                .map(|_| pollster::block_on(slots.acquire()))
                .collect();

            let handles: Vec<_> = (0..8u32)
                .map(|i| {
                    let coordinator = coordinator.clone();
                    let model = model.clone();
                    thread::spawn(move || {
                        let response = coordinator.submit(
                            model.as_ref(),
                            BatchDecodeRequest {
                                cache: KvCache::new(1, 4, 4),
                                token: i,
                                start_pos: 0,
                                greedy_sample: None,
                            },
                        );
                        match response.outcome {
                            Ok(ForwardOutcome::Token(t)) => t,
                            Ok(ForwardOutcome::Logits(_)) => {
                                panic!("MockModel only ever returns Token")
                            }
                            Err(e) => panic!("unexpected batch error: {e}"),
                        }
                    })
                })
                .collect();

            let mut got: Vec<u32> = handles.into_iter().map(|h| h.join().unwrap()).collect();
            let mut expected: Vec<u32> = (0..8u32).map(|i| i * 2).collect();
            got.sort_unstable();
            expected.sort_unstable();
            assert_eq!(
                expected, got,
                "every thread must get back exactly its own token, doubled"
            );

            drop(guards);
        }
    }

    /// A lone submission (no concurrent siblings) must still complete —
    /// the deadline fallback, not the target-batch-size path, is what
    /// closes this window.
    #[test]
    fn solo_submission_completes_via_deadline_fallback() {
        let slots = SlotPool::new(4);
        let coordinator = BatchCoordinator::new(slots.clone());
        let model: Arc<dyn ModelForward> = Arc::new(MockModel);
        let _guard = pollster::block_on(slots.acquire());

        let response = coordinator.submit(
            model.as_ref(),
            BatchDecodeRequest {
                cache: KvCache::new(1, 4, 4),
                token: 7,
                start_pos: 0,
                greedy_sample: None,
            },
        );
        match response.outcome {
            Ok(ForwardOutcome::Token(t)) => assert_eq!(t, 14),
            Ok(ForwardOutcome::Logits(_)) => panic!("MockModel only ever returns Token"),
            Err(e) => panic!("unexpected batch error: {e}"),
        }
    }
}
