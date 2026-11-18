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
//! ).

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
    /// Ticketing for [`SlotPool::acquire`], so waiters are served in the order
    /// they arrived.
    ///
    /// Without it the wait is a scramble: a released slot wakes one sleeper,
    /// which then rescans — and a request that arrived in the meantime can
    /// call `try_take_any` first and take the slot out from under it. Under
    /// steady load that is unbounded waiting for whoever is unlucky, with
    /// nothing in the system that would ever report it.
    ///
    /// Not solvable with a global semaphore, fair though `tokio`'s is:
    /// [`SlotPool::acquire_slot`] bypasses admission by design (a pinned
    /// request is waiting for one specific slot's warm cache), so a global
    /// permit count would drift out of step with the slots actually free —
    /// the failure this pool's own comment above records having removed one
    /// for.
    admission: Mutex<Admission>,
    /// How many waiters [`SlotPool::acquire`] will hold before refusing.
    /// `0` means unbounded, which is the previous behaviour.
    max_queue: usize,
}

/// The ticket dispenser. Only the holder of `serving` may take a slot, which
/// is what makes the order first-come-first-served rather than first-to-wake.
#[derive(Default)]
struct Admission {
    next: u64,
    serving: u64,
    /// Waiters holding a ticket and not yet served — the queue depth an
    /// operator can see and an admission limit is compared against.
    depth: usize,
}

/// A place in the admission queue.
///
/// Its `Drop` is the load-bearing part: a waiter can vanish at any moment —
/// the client hung up, the request timed out, the runtime cancelled the task —
/// and a ticket that outlived its holder would leave `serving` pointing at a
/// number nobody will ever claim, stalling every request behind it for the
/// life of the process. Releasing it on drop makes cancellation safe by
/// construction rather than by remembering.
struct Ticket {
    pool: Arc<SlotPool>,
    number: u64,
    /// Set once this ticket has taken a slot, so `Drop` knows whether to
    /// advance `serving` (it already did) or to step over an abandoned place.
    served: bool,
}

impl Ticket {
    /// Whether it is this ticket's turn.
    fn is_serving(&self) -> bool {
        self.pool.admission.lock().unwrap().serving == self.number
    }

    /// Records that this ticket claimed a slot and hands the queue on.
    fn serve(&mut self) {
        self.served = true;
        let mut admission = self.pool.admission.lock().unwrap();
        admission.serving = admission.serving.max(self.number + 1);
        admission.depth = admission.depth.saturating_sub(1);
        // The guard is dropped before waking anyone, so a woken waiter does
        // not immediately block on a lock this thread still holds.
        drop(admission);
        self.pool.released.notify_waiters();
    }
}

impl Drop for Ticket {
    fn drop(&mut self) {
        if self.served {
            return;
        }
        let mut admission = self.pool.admission.lock().unwrap();
        // Step over this place if it was at the head, so the next waiter is
        // not blocked behind a request that no longer exists.
        admission.serving = admission.serving.max(self.number + 1);
        admission.depth = admission.depth.saturating_sub(1);
        drop(admission);
        self.pool.released.notify_waiters();
    }
}

impl SlotPool {
    /// An unbounded pool. Test-only since production reads
    /// `[orangu-server].queue_limit` and goes through
    /// [`with_queue_limit`](Self::with_queue_limit); kept because a test
    /// saying `new(2)` reads better than one repeating `, 0` it does not care
    /// about.
    #[cfg(test)]
    pub fn new(n: usize) -> Arc<Self> {
        Self::with_queue_limit(n, 0)
    }

    /// A pool that refuses a request once `max_queue` are already waiting.
    ///
    /// `0` keeps the previous unbounded behaviour. A bound is what turns
    /// overload from "every client waits, indefinitely, with no signal" into
    /// an answer the caller can act on — a `503` arriving in milliseconds is
    /// more useful than a token arriving in minutes, and it is the difference
    /// between a queue and a pile.
    pub fn with_queue_limit(n: usize, max_queue: usize) -> Arc<Self> {
        Arc::new(Self {
            admission: Mutex::new(Admission::default()),
            max_queue,
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

    /// How many slots are currently busy (prefilling or decoding).
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
    /// [`try_acquire`](Self::try_acquire) on a pool that cannot refuse.
    ///
    /// Test-only for the same reason as [`new`](Self::new): production has an
    /// admission limit to honour and must handle a refusal.
    #[cfg(test)]
    pub async fn acquire(self: &Arc<Self>) -> SlotGuard {
        self.try_acquire()
            .await
            .expect("an unbounded pool never refuses")
    }

    /// [`SlotPool::acquire`], returning `None` when the queue is already full.
    ///
    /// Waiters are served **in arrival order**. A ticket is taken up front and
    /// only its holder may claim a slot, so a request that arrives while
    /// someone is asleep cannot take the slot they were woken for.
    ///
    /// The ticket is released on drop, which matters more than it looks:
    /// dropping this future is exactly what a disconnected client does, and a
    /// cancelled waiter that kept its ticket would stall everyone behind it
    /// forever.
    pub async fn try_acquire(self: &Arc<Self>) -> Option<SlotGuard> {
        let mut ticket = self.take_ticket()?;
        loop {
            if ticket.is_serving()
                && let Some((index, lock)) = self.try_take_any()
            {
                self.mark_busy(index);
                ticket.serve();
                return Some(SlotGuard {
                    pool: self.clone(),
                    index,
                    lock: Some(lock),
                });
            }
            // `notify_one` (not `notify_waiters`) on the release side, so a
            // slot freed between the scan above and this await stores a
            // permit rather than being missed — the woken waiter simply
            // rescans. Without that, a release landing in the gap would
            // leave a request asleep next to an idle slot.
            //
            // Registered *before* the checks above would be tidier, but the
            // ticket makes it unnecessary: a waiter that is not yet at the
            // head has nothing to race for, and the head is woken by every
            // release.
            self.released.notified().await;
        }
    }

    /// Takes a queue ticket, or `None` when the queue is full.
    fn take_ticket(self: &Arc<Self>) -> Option<Ticket> {
        let mut admission = self.admission.lock().unwrap();
        if self.max_queue > 0 && admission.depth >= self.max_queue {
            return None;
        }
        let number = admission.next;
        admission.next += 1;
        admission.depth += 1;
        Some(Ticket {
            pool: self.clone(),
            number,
            served: false,
        })
    }

    /// Waiters holding a ticket right now — queue depth, for `/metrics` and
    /// for anyone deciding whether this server is the bottleneck.
    pub fn queued(&self) -> usize {
        self.admission.lock().unwrap().depth
    }

    /// The configured admission limit, `0` for unbounded.
    pub fn queue_limit(&self) -> usize {
        self.max_queue
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
    /// Overload has to be answerable. With a bound, the request beyond it is
    /// refused in microseconds instead of joining a pile nothing measures.
    #[tokio::test]
    async fn a_full_queue_refuses_instead_of_growing() {
        let pool = SlotPool::with_queue_limit(1, 2);
        let _busy = pool.acquire().await;

        // Two waiters fill the queue.
        let a = tokio::spawn({
            let p = pool.clone();
            async move { p.try_acquire().await.is_some() }
        });
        let b = tokio::spawn({
            let p = pool.clone();
            async move { p.try_acquire().await.is_some() }
        });
        // Give both a chance to take their tickets before asking about depth.
        for _ in 0..100 {
            if pool.queued() == 2 {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(pool.queued(), 2, "both waiters should be queued");

        // The third is refused rather than queued.
        assert!(
            pool.try_acquire().await.is_none(),
            "a full queue must refuse"
        );

        drop(_busy);
        assert!(
            a.await.unwrap() && b.await.unwrap(),
            "queued waiters served"
        );
    }

    /// An unbounded pool is the previous behaviour and must stay reachable —
    /// a limit nobody asked for would turn working deployments into ones that
    /// start refusing.
    #[tokio::test]
    async fn an_unbounded_pool_never_refuses() {
        let pool = SlotPool::new(1);
        let _busy = pool.acquire().await;
        assert_eq!(pool.queue_limit(), 0);
        let waiters: Vec<_> = (0..8)
            .map(|_| {
                let p = pool.clone();
                tokio::spawn(async move { p.try_acquire().await.is_some() })
            })
            .collect();
        for _ in 0..200 {
            if pool.queued() == 8 {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(pool.queued(), 8);
        drop(_busy);
        for w in waiters {
            assert!(w.await.unwrap(), "nobody is refused without a limit");
        }
    }

    /// Only the head of the queue may take a slot, even when one is free.
    ///
    /// **This is the test that discriminates.** The end-to-end order test
    /// below does not: with every waiter already asleep, `tokio`'s `Notify`
    /// wakes them first-in-first-out anyway, so it passes with the ticket
    /// check removed. What first-to-wake actually loses to is a request
    /// *arriving* in the gap between a release and the woken waiter's rescan,
    /// and that race cannot be staged deterministically. So the rule is
    /// asserted directly instead: a ticket that is not at the head must not be
    /// serving, however free the pool is.
    #[test]
    fn only_the_head_of_the_queue_may_take_a_slot() {
        let pool = SlotPool::new(4);
        let first = pool.take_ticket().expect("unbounded");
        let second = pool.take_ticket().expect("unbounded");
        assert_eq!(pool.queued(), 2);

        // Four slots are free and `second` still may not take one.
        assert!(first.is_serving());
        assert!(!second.is_serving(), "a later arrival must wait its turn");

        drop(first);
        assert!(second.is_serving(), "the queue advances when the head goes");
        drop(second);
        assert_eq!(pool.queued(), 0);
    }

    /// Waiters are served in arrival order, end to end.
    ///
    /// A regression guard for the observable property rather than proof the
    /// ticket is doing the work — see
    /// [`only_the_head_of_the_queue_may_take_a_slot`], which is the one that
    /// fails without it.
    #[tokio::test]
    async fn waiters_are_served_in_arrival_order() {
        let pool = SlotPool::with_queue_limit(1, 0);
        let busy = pool.acquire().await;
        let order = Arc::new(Mutex::new(Vec::new()));

        let mut handles = Vec::new();
        for i in 0..5u32 {
            // Each waiter is fully queued before the next one starts, so
            // arrival order is unambiguous.
            let p = pool.clone();
            let seen = order.clone();
            let want = i as usize + 1;
            handles.push(tokio::spawn(async move {
                let guard = p.acquire().await;
                seen.lock().unwrap().push(i);
                drop(guard);
            }));
            for _ in 0..500 {
                if pool.queued() == want {
                    break;
                }
                tokio::task::yield_now().await;
            }
            assert_eq!(pool.queued(), want, "waiter {i} did not queue");
        }

        drop(busy);
        for h in handles {
            h.await.unwrap();
        }
        assert_eq!(
            *order.lock().unwrap(),
            vec![0, 1, 2, 3, 4],
            "served out of arrival order"
        );
    }

    /// A waiter that goes away must not stall the queue behind it.
    ///
    /// This is not hypothetical — dropping the future is exactly what a
    /// disconnected client does, and a ticket that outlived its holder would
    /// leave the head pointing at a number nobody will ever claim, hanging
    /// every later request for the life of the process.
    #[tokio::test]
    async fn an_abandoned_waiter_does_not_stall_the_queue() {
        let pool = SlotPool::with_queue_limit(1, 0);
        let busy = pool.acquire().await;

        let doomed = tokio::spawn({
            let p = pool.clone();
            async move { p.acquire().await }
        });
        for _ in 0..500 {
            if pool.queued() == 1 {
                break;
            }
            tokio::task::yield_now().await;
        }
        let behind = tokio::spawn({
            let p = pool.clone();
            async move { p.acquire().await.id() }
        });
        for _ in 0..500 {
            if pool.queued() == 2 {
                break;
            }
            tokio::task::yield_now().await;
        }

        // The head of the queue vanishes.
        doomed.abort();
        let _ = doomed.await;
        drop(busy);

        let served = tokio::time::timeout(std::time::Duration::from_secs(5), behind)
            .await
            .expect("the queue stalled behind an abandoned ticket")
            .unwrap();
        assert_eq!(served, 0);
        assert_eq!(pool.queued(), 0, "no ticket should be left behind");
    }

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
