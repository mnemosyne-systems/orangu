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

//! Bounds how many requests generate concurrently, and tracks each one's
//! progress for the `/slots` endpoint. Each of `slots` concurrent
//! generations runs its own prefill+decode loop against its own KV cache on
//! its own blocking-pool thread (`engine::generate`) — real concurrency,
//! bounded fairly by slot count, but not llama.cpp's fused single-GEMM
//! cross-sequence batching (a distinct performance optimization —
//! see `engine::batch::BatchCoordinator`).

use serde::Serialize;
use std::sync::{Arc, Mutex};
use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore};

#[derive(Clone, Debug, Default, Serialize)]
pub struct SlotState {
    pub id: usize,
    pub busy: bool,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
}

pub struct SlotPool {
    slots: Vec<Mutex<SlotState>>,
    /// One single-permit semaphore per slot. Holding slot *i*'s permit **is**
    /// owning slot *i*, and since there are exactly as many permits as slots,
    /// it is also the concurrency bound — there is no second, global
    /// semaphore.
    ///
    /// There used to be one, and pinning is what ruled it out: a request
    /// waiting for a *named* slot would sit on a global permit while it
    /// waited, so a pinned waiter subtracted from everyone else's concurrency
    /// without doing any work. Taking the global permit second instead just
    /// moves the problem (an unpinned `acquire` holds a permit while looking
    /// for a lock). One semaphore, held for exactly as long as the slot is
    /// owned, has neither failure mode.
    locks: Vec<Arc<Semaphore>>,
    /// Signalled when any slot is released, so [`SlotPool::acquire`] — which
    /// wants *whichever* slot frees first — can wait without polling.
    released: Notify,
}

impl SlotPool {
    pub fn new(n: usize) -> Arc<Self> {
        Arc::new(Self {
            slots: (0..n)
                .map(|id| {
                    Mutex::new(SlotState {
                        id,
                        ..Default::default()
                    })
                })
                .collect(),
            locks: (0..n).map(|_| Arc::new(Semaphore::new(1))).collect(),
            released: Notify::new(),
        })
    }

    pub fn total(&self) -> usize {
        self.slots.len()
    }

    /// How many slots are currently busy (prefilling or decoding) —
    /// `engine::batch::BatchCoordinator`'s hint for how many concurrent
    /// decode steps to expect in the *current* cross-sequence batch.
    /// A live count, not a request-time snapshot — it can
    /// briefly overestimate during a mixed prefill/decode moment (a
    /// prefilling slot is "busy" but not yet submitting decode steps),
    /// which just means a batch waits out its own timeout instead of
    /// closing early; never a correctness concern, only a latency one.
    pub fn busy_count(&self) -> usize {
        self.slots.iter().filter(|s| s.lock().unwrap().busy).count()
    }

    /// Waits for whichever slot frees first, marks it busy, and returns a
    /// guard that releases it on drop.
    pub async fn acquire(self: &Arc<Self>) -> SlotGuard {
        loop {
            if let Some((index, lock)) = self.try_take_any() {
                self.mark_busy(index);
                return SlotGuard {
                    pool: self.clone(),
                    index,
                    lock: Some(lock),
                };
            }
            // `notify_one` (not `notify_waiters`) on the release side, so a
            // slot freed between the scan above and this await stores a
            // permit rather than being missed — the woken waiter simply
            // rescans. Without that, a release landing in the gap would
            // leave a request asleep next to an idle slot.
            self.released.notified().await;
        }
    }

    fn try_take_any(&self) -> Option<(usize, OwnedSemaphorePermit)> {
        self.locks
            .iter()
            .enumerate()
            .find_map(|(i, lock)| lock.clone().try_acquire_owned().ok().map(|l| (i, l)))
    }

    /// Waits for **this** slot specifically, however long its current request
    /// takes, then marks it busy exactly as [`SlotPool::acquire`] does.
    ///
    /// Waiting rather than falling back to any free slot is the whole point:
    /// a slot carries the KV cache of the last request that ran on it
    /// (`engine::slot_store`), so a conversation that keeps landing on its own
    /// slot continues from a warm prefix, while one that gets bounced to a
    /// neighbour finds a stranger's cache there and reprefills from scratch.
    /// Trading a little queueing for that is the trade a caller asking for a
    /// named slot has already decided to make.
    ///
    /// A waiter here holds nothing while it waits, so it costs no other
    /// request any concurrency and cannot deadlock: the only thing it needs is
    /// the one lock, and whoever holds that lock owns everything required to
    /// finish.
    ///
    /// Panics on an out-of-range `index`; callers reject that at the HTTP
    /// boundary, where it can be reported properly.
    pub async fn acquire_slot(self: &Arc<Self>, index: usize) -> SlotGuard {
        let lock = self.locks[index]
            .clone()
            .acquire_owned()
            .await
            .expect("a slot lock is never closed");
        self.mark_busy(index);
        SlotGuard {
            pool: self.clone(),
            index,
            lock: Some(lock),
        }
    }

    fn mark_busy(&self, index: usize) {
        let mut state = self.slots[index].lock().unwrap();
        state.busy = true;
        state.prompt_tokens = 0;
        state.generated_tokens = 0;
    }

    pub fn snapshot(&self) -> Vec<SlotState> {
        self.slots
            .iter()
            .map(|s| s.lock().unwrap().clone())
            .collect()
    }
}

pub struct SlotGuard {
    pool: Arc<SlotPool>,
    index: usize,
    /// `Option` only so [`Drop`] can hand the permit back *before* it signals
    /// `released`. A struct's fields drop after its `Drop::drop` body, so
    /// notifying first would wake a waiter that then finds the lock still
    /// held, fails its rescan, and sleeps again with no notification pending —
    /// a request asleep beside the idle slot it was just told about.
    lock: Option<OwnedSemaphorePermit>,
}

impl SlotGuard {
    pub fn id(&self) -> usize {
        self.index
    }

    pub fn set_prompt_tokens(&self, n: usize) {
        self.pool.slots[self.index].lock().unwrap().prompt_tokens = n;
    }

    pub fn set_generated_tokens(&self, n: usize) {
        self.pool.slots[self.index].lock().unwrap().generated_tokens = n;
    }
}

impl Drop for SlotGuard {
    fn drop(&mut self) {
        self.pool.slots[self.index].lock().unwrap().busy = false;
        drop(self.lock.take());
        self.pool.released.notify_one();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn acquire_marks_a_slot_busy_and_release_frees_it() {
        let pool = SlotPool::new(2);
        let guard = pool.acquire().await;
        assert!(pool.snapshot()[guard.id()].busy);
        drop(guard);
        assert!(pool.snapshot().iter().all(|s| !s.busy));
    }

    #[tokio::test]
    async fn acquire_slot_gives_back_the_slot_that_was_asked_for() {
        let pool = SlotPool::new(4);
        for id in [3, 0, 2] {
            let guard = pool.acquire_slot(id).await;
            assert_eq!(guard.id(), id);
        }
    }

    /// The point of pinning: a request must wait for *its* slot rather than
    /// be handed a free neighbour, because the neighbour holds someone else's
    /// KV cache.
    #[tokio::test]
    async fn acquire_slot_waits_for_that_slot_even_when_others_are_free() {
        let pool = SlotPool::new(3);
        let holder = pool.acquire_slot(1).await;
        let pool2 = pool.clone();
        let pinned = tokio::spawn(async move { pool2.acquire_slot(1).await.id() });
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        assert!(
            !pinned.is_finished(),
            "took a different slot instead of waiting"
        );
        drop(holder);
        assert_eq!(pinned.await.unwrap(), 1);
    }

    /// The deadlock this ordering exists to avoid: a pinned request waiting on
    /// a busy slot while unpinned requests keep arriving. `acquire` takes its
    /// concurrency permit before looking for a lock, and whoever holds the
    /// contended lock already holds a permit, so everyone makes progress.
    #[tokio::test]
    async fn a_pinned_waiter_does_not_starve_or_deadlock_unpinned_requests() {
        let pool = SlotPool::new(2);
        let holder = pool.acquire_slot(0).await;

        let p = pool.clone();
        let pinned = tokio::spawn(async move { p.acquire_slot(0).await.id() });
        let p = pool.clone();
        let anyone = tokio::spawn(async move { p.acquire().await.id() });

        // The unpinned request must get the other slot while the pinned one
        // is still queued behind slot 0.
        assert_eq!(
            tokio::time::timeout(std::time::Duration::from_secs(5), anyone)
                .await
                .expect("unpinned request deadlocked")
                .unwrap(),
            1
        );
        assert!(!pinned.is_finished());
        drop(holder);
        assert_eq!(
            tokio::time::timeout(std::time::Duration::from_secs(5), pinned)
                .await
                .expect("pinned request deadlocked")
                .unwrap(),
            0
        );
    }

    /// `acquire` must never hand the same slot to two live requests — the
    /// invariant `engine::slot_store` relies on to borrow a slot's retained
    /// cache without locking it.
    #[tokio::test]
    async fn every_concurrent_acquire_gets_a_distinct_slot() {
        let pool = SlotPool::new(4);
        let mut guards = Vec::new();
        for _ in 0..4 {
            guards.push(pool.acquire().await);
        }
        let mut ids: Vec<usize> = guards.iter().map(|g| g.id()).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![0, 1, 2, 3]);
    }

    #[tokio::test]
    async fn a_third_request_waits_when_both_slots_are_busy() {
        let pool = SlotPool::new(1);
        let guard = pool.acquire().await;
        let pool2 = pool.clone();
        let acquired_second = tokio::spawn(async move {
            tokio::time::timeout(std::time::Duration::from_millis(50), pool2.acquire())
                .await
                .is_ok()
        });
        // The single slot is held, so the second acquire must still be
        // pending when we check.
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        assert!(!acquired_second.is_finished());
        drop(guard);
        assert!(acquired_second.await.unwrap());
    }
}
