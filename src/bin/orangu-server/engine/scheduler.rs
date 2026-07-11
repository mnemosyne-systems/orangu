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
//! cross-sequence batching (a distinct performance optimization; see
//! `doc/SERVER_ROADMAP.md`).

use serde::Serialize;
use std::sync::{Arc, Mutex};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

#[derive(Clone, Debug, Default, Serialize)]
pub struct SlotState {
    pub id: usize,
    pub busy: bool,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
}

pub struct SlotPool {
    semaphore: Arc<Semaphore>,
    slots: Vec<Mutex<SlotState>>,
}

impl SlotPool {
    pub fn new(n: usize) -> Arc<Self> {
        Arc::new(Self {
            semaphore: Arc::new(Semaphore::new(n)),
            slots: (0..n)
                .map(|id| {
                    Mutex::new(SlotState {
                        id,
                        ..Default::default()
                    })
                })
                .collect(),
        })
    }

    pub fn total(&self) -> usize {
        self.slots.len()
    }

    /// How many slots are currently busy (prefilling or decoding) —
    /// `engine::batch::BatchCoordinator`'s hint for how many concurrent
    /// decode steps to expect in the *current* cross-sequence batch
    /// (`doc/SERVER_OPUS.md` Section 9's Step 11's cross-sequence GEMM
    /// batching item). A live count, not a request-time snapshot — it can
    /// briefly overestimate during a mixed prefill/decode moment (a
    /// prefilling slot is "busy" but not yet submitting decode steps),
    /// which just means a batch waits out its own timeout instead of
    /// closing early; never a correctness concern, only a latency one.
    pub fn busy_count(&self) -> usize {
        self.slots.iter().filter(|s| s.lock().unwrap().busy).count()
    }

    /// Waits for a free slot, marks it busy, and returns a guard that
    /// releases it (and the underlying concurrency permit) on drop.
    pub async fn acquire(self: &Arc<Self>) -> SlotGuard {
        let permit = self
            .semaphore
            .clone()
            .acquire_owned()
            .await
            .expect("SlotPool's semaphore is never closed");
        let index = self
            .slots
            .iter()
            .position(|s| !s.lock().unwrap().busy)
            .expect("a permit guarantees at least one slot is free");
        {
            let mut state = self.slots[index].lock().unwrap();
            state.busy = true;
            state.prompt_tokens = 0;
            state.generated_tokens = 0;
        }
        SlotGuard {
            pool: self.clone(),
            index,
            _permit: permit,
        }
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
    _permit: OwnedSemaphorePermit,
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
