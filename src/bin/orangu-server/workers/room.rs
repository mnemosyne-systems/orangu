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

//! What a node has promised of its memory to the sequences it holds rows for.
//!
//! A sequence's rows are written as its prompt runs, and a large allocation
//! costs nothing until then, so a node that takes a prompt it cannot hold
//! finds out far into it, when the kernel kills it. Its parent then sees a lost
//! worker and replays the same prompt. So a prompt asks first, for its whole
//! length, and a node compares that with what it has free now, less what it
//! has promised and not yet written ([`KvRoom::reserve`]).
//!
//! Memory is read fresh each time rather than kept in books, since other
//! processes use it too. Only the promises are kept here.

use crate::engine::arch::ModelForward;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// The share of memory no promise reaches into: scratch, the activations in
/// flight, everything else on the machine.
const MARGIN_SHARE: f64 = 0.05;

/// A promise nothing has written to for this long is from a parent that went
/// away. Sessions are evicted after the same time.
const PROMISE_LIFETIME: Duration = Duration::from_secs(600);

/// Prompts refused for memory on this node's tree, since it started.
static REFUSED: AtomicU64 = AtomicU64::new(0);

pub fn note_refused() {
    REFUSED.fetch_add(1, Ordering::Relaxed);
}

pub fn refused() -> u64 {
    REFUSED.load(Ordering::Relaxed)
}

/// Why a prompt does not fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shortfall {
    pub needed: u64,
    pub promised: u64,
    pub free: u64,
}

impl std::fmt::Display for Shortfall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use orangu::format::format_bytes;
        write!(
            f,
            "the prompt's KV cache needs {} here, with {} free and {} promised to other \
             sequences: send a shorter prompt, or give this node more memory",
            format_bytes(self.needed),
            format_bytes(self.free),
            format_bytes(self.promised)
        )
    }
}

struct Promise {
    positions: usize,
    written: usize,
    touched: Instant,
}

impl Promise {
    fn outstanding(&self) -> usize {
        self.positions.saturating_sub(self.written)
    }
}

type FreeMemory = Box<dyn Fn() -> u64 + Send + Sync>;

pub struct KvRoom {
    /// What a position of a sequence costs on this node's own layers.
    per_position: u64,
    free: FreeMemory,
    promises: Mutex<HashMap<u64, Promise>>,
}

impl KvRoom {
    /// The room for `layers` of `model`, as this machine has it.
    pub fn of(model: &dyn ModelForward, layers: &Range<usize>) -> Self {
        Self::with_free_memory(per_position(model, layers), Box::new(free_memory))
    }

    pub fn with_free_memory(per_position: u64, free: FreeMemory) -> Self {
        Self {
            per_position,
            free,
            promises: Mutex::new(HashMap::new()),
        }
    }

    /// Promises `session` memory for `positions`, past what it has already
    /// written. A session asking again replaces its promise.
    pub fn reserve(&self, session: u64, positions: usize) -> Result<(), Shortfall> {
        let mut promises = self.promises.lock().unwrap();
        let now = Instant::now();
        promises.retain(|_, p| now.duration_since(p.touched) < PROMISE_LIFETIME);
        let written = promises.get(&session).map_or(0, |p| p.written);
        let needed = self
            .per_position
            .saturating_mul(positions.saturating_sub(written) as u64);
        if needed > 0 {
            let promised = promises
                .iter()
                .filter(|(id, _)| **id != session)
                .map(|(_, p)| self.per_position.saturating_mul(p.outstanding() as u64))
                .sum::<u64>();
            let free = (self.free)();
            if needed.saturating_add(promised) > free {
                return Err(Shortfall {
                    needed,
                    promised,
                    free,
                });
            }
        }
        promises.insert(
            session,
            Promise {
                positions,
                written,
                touched: now,
            },
        );
        Ok(())
    }

    /// `session` now holds `len` positions: they are in the free memory read
    /// from here on, and no longer owed.
    pub fn wrote(&self, session: u64, len: usize) {
        if let Some(promise) = self.promises.lock().unwrap().get_mut(&session) {
            promise.written = promise.written.max(len);
            promise.touched = Instant::now();
        }
    }

    pub fn forget(&self, session: u64) {
        self.promises.lock().unwrap().remove(&session);
    }

    /// Whether `session` is still owed memory: a prompt on its way, which
    /// a session that only looks idle must not be evicted from under.
    pub fn owes(&self, session: u64) -> bool {
        self.promises
            .lock()
            .unwrap()
            .get(&session)
            .is_some_and(|p| p.outstanding() > 0)
    }

    /// Bytes promised and not yet written.
    pub fn promised_bytes(&self) -> u64 {
        self.promises
            .lock()
            .unwrap()
            .values()
            .map(|p| self.per_position.saturating_mul(p.outstanding() as u64))
            .sum()
    }
}

/// What a position costs on `layers`: its rows on the host, and their copy on
/// the device, which on a board that shares its memory comes out of the same
/// RAM. A recurrent layer's state does not grow with the sequence.
fn per_position(model: &dyn ModelForward, layers: &Range<usize>) -> u64 {
    if layers.is_empty() {
        return 0;
    }
    let probe = model.new_kv_cache_for_layers(layers.clone(), 1);
    let rows: u64 = layers
        .clone()
        .map(|layer| probe.hold_bytes(model.cache_slot(layer)).0)
        .sum();
    let mirror = model.vulkan_backend().map_or(0, |backend| {
        const STEP: usize = 1024;
        let n_head = model.config().n_head;
        probe.gpu_mirror_bytes_where(STEP, n_head, backend.kv_storage(), |layer| {
            layers.contains(&layer)
        }) / STEP as u64
    });
    rows + mirror
}

/// What this process could still commit: what the machine has available, or
/// what its cgroup leaves if that is less, each less a margin.
fn free_memory() -> u64 {
    let Ok(mut system) = sysinfo::System::new() else {
        return u64::MAX;
    };
    system.refresh_memory();
    let margin = (system.total_memory() as f64 * MARGIN_SHARE) as u64;
    let machine = system.available_memory().saturating_sub(margin);
    machine.min(cgroup_free().unwrap_or(u64::MAX))
}

/// The tightest of this process's cgroup and its ancestors, under cgroup v2;
/// `None` when none of them has a limit.
fn cgroup_free() -> Option<u64> {
    let own = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let path = own.lines().find_map(|line| line.strip_prefix("0::"))?;
    let mut dir = std::path::PathBuf::from("/sys/fs/cgroup").join(path.trim_start_matches('/'));
    let mut tightest = None;
    loop {
        let read = |name: &str| std::fs::read_to_string(dir.join(name)).ok();
        if let (Some(max), Some(current), Some(stat)) = (
            read("memory.max"),
            read("memory.current"),
            read("memory.stat"),
        ) && let Some(free) = free_in_cgroup(&max, &current, &stat)
        {
            tightest = Some(free.min(tightest.unwrap_or(u64::MAX)));
        }
        if !dir.pop() || !dir.starts_with("/sys/fs/cgroup") {
            return tightest;
        }
    }
}

/// What a cgroup has left, from its `memory.max`, `memory.current` and
/// `memory.stat`, less a margin: page cache can be taken back, as a machine's
/// available memory counts it. `None` with no limit.
fn free_in_cgroup(max: &str, current: &str, stat: &str) -> Option<u64> {
    let max: u64 = max.trim().parse().ok()?;
    let current: u64 = current.trim().parse().ok()?;
    let file = stat.lines().find_map(|line| {
        let (key, value) = line.split_once(' ')?;
        (key == "file").then(|| value.trim().parse::<u64>().ok())?
    })?;
    let margin = (max as f64 * MARGIN_SHARE) as u64;
    Some(
        max.saturating_sub(current.saturating_sub(file))
            .saturating_sub(margin),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    fn room(per_position: u64, free: u64) -> KvRoom {
        KvRoom::with_free_memory(per_position, Box::new(move || free))
    }

    #[test]
    fn a_prompt_that_fits_is_promised_and_one_that_does_not_is_told_why() {
        let room = room(1 << 20, GIB);
        assert_eq!(room.reserve(1, 1024), Ok(()));
        let short = room.reserve(2, 2048).unwrap_err();
        assert_eq!(short.needed, 2 * GIB);
        assert_eq!(short.free, GIB);
        assert_eq!(short.promised, GIB);
        assert!(short.to_string().contains("2.00 GiB"), "{short}");
    }

    #[test]
    fn a_refusal_promises_nothing() {
        let room = room(1 << 20, GIB);
        assert!(room.reserve(1, 2048).is_err());
        assert_eq!(room.promised_bytes(), 0);
        assert!(!room.owes(1));
    }

    #[test]
    fn what_is_written_stops_counting_against_the_others() {
        let room = room(1 << 20, 2 * GIB);
        room.reserve(1, 1536).unwrap();
        assert!(room.reserve(2, 1024).is_err(), "1.5 + 1 of 2 GiB");
        // Written, it is in the machine's used memory, not in the promises.
        room.wrote(1, 1536);
        assert_eq!(room.promised_bytes(), 0);
        assert!(!room.owes(1));
        assert_eq!(room.reserve(2, 1024), Ok(()));
    }

    #[test]
    fn a_session_going_on_needs_only_its_new_positions() {
        let room = room(1 << 20, GIB);
        room.reserve(1, 1024).unwrap();
        room.wrote(1, 1024);
        assert_eq!(room.reserve(1, 2000), Ok(()), "976 more positions");
        assert_eq!(room.promised_bytes(), 976 << 20);
        room.wrote(1, 1000);
        assert_eq!(room.reserve(1, 800), Ok(()), "rolled back, already held");
        assert_eq!(room.promised_bytes(), 0);
    }

    #[test]
    fn forgetting_gives_a_promise_back() {
        let room = room(1 << 20, GIB);
        room.reserve(1, 1024).unwrap();
        assert!(room.owes(1));
        room.forget(1);
        assert!(!room.owes(1));
        assert_eq!(room.reserve(2, 1024), Ok(()));
    }

    #[test]
    fn a_promise_from_a_parent_that_went_away_runs_out() {
        let room = room(1 << 20, GIB);
        room.reserve(1, 1024).unwrap();
        room.promises.lock().unwrap().get_mut(&1).unwrap().touched =
            Instant::now() - PROMISE_LIFETIME;
        assert_eq!(room.reserve(2, 1024), Ok(()));
        assert!(!room.owes(1));
    }

    #[test]
    fn layers_elsewhere_cost_nothing_and_read_no_memory() {
        let room = KvRoom::with_free_memory(0, Box::new(|| panic!("read")));
        assert_eq!(room.reserve(1, 1 << 20), Ok(()));
    }

    #[test]
    fn a_cgroup_counts_page_cache_as_free_less_a_margin() {
        let stat = "anon 700\nfile 200\nkernel 10\n";
        assert_eq!(free_in_cgroup("1000\n", "900\n", stat), Some(250));
        assert_eq!(free_in_cgroup("max\n", "900\n", stat), None, "no limit");
        assert_eq!(free_in_cgroup("1000", "5000", "file 0"), Some(0), "over");
        assert_eq!(
            free_in_cgroup("1000", "900", "anon 900"),
            None,
            "no file line"
        );
    }

    #[test]
    fn this_machine_has_some_memory_free() {
        assert!(free_memory() > 0);
    }
}
