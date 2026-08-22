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

//! Which transformer layers stay in memory, for a model too large to hold.
//!
//! # Why this exists, in one number
//!
//! A dense forward pass is a **strict cyclic sweep**: every layer, in order,
//! for every token, forever. That is the access pattern LRU handles worst —
//! the least-recently-used page is precisely the one the next sweep wants
//! first — and `DISK.md`'s D1a measured what it costs. On a 6.76 GiB model
//! under caps from 6 down to 3 GiB the kernel read **1.5× to 3.3× more bytes
//! than it had to**, and below about 60% residency it read *the whole model
//! for every single token*.
//!
//! Nothing else on that page is a lever any more. The read *mechanism* is not:
//! `mmap`, `pread` and `MADV_WILLNEED` measured 318, 320 and 321 MB/s on the
//! same cold span, a tie (D1e). The read *rate* is not something to optimise
//! either — the drive collapses from 207 MB/s to 31 after forty seconds of
//! continuous reading whatever asks for the bytes. **Bytes moved is the one
//! quantity a policy controls, and it is a count rather than a rate, which is
//! why it is the only figure on that page that survived re-measurement.**
//!
//! # The policy
//!
//! *(Measured and negative — see "Measured, and it does not pay" below before
//! acting on any of this.)*
//!
//! Release *behind* the sweep instead of letting the kernel evict *ahead* of
//! it. As layer `L` is entered, the layer `window` places back is handed to
//! `MADV_DONTNEED`; the pages it frees are the ones with the longest wait
//! until they are next wanted, which is exactly the choice LRU gets wrong.
//!
//! Deliberately **not** a prefetcher. `MADV_WILLNEED` was measured to reach no
//! better rate than a demand fault does, so reading ahead buys overlap at
//! best and cannot buy bandwidth. Release is the half with the number behind
//! it.
//!
//! # Measured, and it does not pay — so it stays off
//!
//! **Three windows, and every one of them loses.** A window of 4 saved 4.9%
//! of the bytes and cost **10.9% on tokens per second**. A cap-sized window of
//! 32 was **87% slower and read 60% more** — after layer `L` is used it is not
//! wanted again for a full cycle, so releasing the layer half a cycle from
//! reuse is nearly the worst choice available. And `window = 1`, the
//! MRU-optimal setting this design argues for above, saved 3.3% of the bytes
//! and **won nothing**: its apparent edge was smaller than the interleaved
//! control's own 19% spread, and the model's own decode timings put it 3.4%
//! *slower*.
//!
//! The column that explains all three is `pgsteal`, which moved **3.3%**. The
//! kernel was already evicting 8.4 million pages either way; one control arm
//! refaulted **exactly zero** times. At the caps this matters at, the kernel
//! has already stopped trying to cache — so there is no reuse left to
//! reclaim, and `MADV_DONTNEED` can only evict pages it would have kept. That
//! is what 202,000 refaults against zero looks like.
//!
//! So the paragraphs above describe a mechanism that works and a lever that
//! is not there. **Do not turn this on expecting throughput.**
//!
//! # A control arm rather than a placeholder
//!
//! `ORANGU_DENSE_WINDOW=<layers>` turns it on. Unset, [`touch`] is a load and
//! a compare and the behaviour is exactly the kernel's own — which is the
//! control arm D1a measured, reachable through the same code path as its
//! replacement so the comparison is against a measurement rather than a
//! reconstruction. `engine::expert_store` earned that shape twice over and
//! this follows it. It is kept for that reason: the seam is what made the
//! answer measurable, and re-deriving it to ask the question again on
//! different hardware would cost more than leaving it here.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

/// One layer's mapped weight bytes, as `(address, length)` spans.
///
/// Addresses rather than offsets: the release is an `madvise` on this
/// process's own mapping, and the mapping is pinned for the model's lifetime
/// (`engine::page_cache::register_shard` holds each shard's `Arc<Mmap>`), so
/// an address recorded at load is valid for as long as anything can ask.
type LayerSpans = Vec<(usize, usize)>;

static SPANS: OnceLock<Vec<LayerSpans>> = OnceLock::new();
static WINDOW: OnceLock<Option<usize>> = OnceLock::new();
/// The layer most recently entered, so the release runs once per layer rather
/// than once per matmul — a layer issues five or more of them.
static CURRENT: AtomicUsize = AtomicUsize::new(usize::MAX);
static RELEASED_LAYERS: AtomicU64 = AtomicU64::new(0);
static RELEASED_BYTES: AtomicU64 = AtomicU64::new(0);
/// Serialises the release so two threads entering a layer together cannot
/// both advise the same range. Uncontended in the common case: only the
/// thread that wins the `CURRENT` swap takes it.
static RELEASING: Mutex<()> = Mutex::new(());

/// How many layers stay resident behind the sweep, from
/// `ORANGU_DENSE_WINDOW`. `None` — the default — leaves residency entirely to
/// the kernel.
fn window() -> Option<usize> {
    *WINDOW.get_or_init(|| {
        std::env::var("ORANGU_DENSE_WINDOW")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|w| *w > 0)
    })
}

/// Records each layer's weight spans. Called once, after the model is built.
///
/// Ignored when called twice: the spans belong to one loaded model and this
/// server serves one.
pub fn register(spans: Vec<LayerSpans>) {
    let _ = SPANS.set(spans);
}

/// Whether a policy is active at all — for the report, so "off" and "measured
/// zero" cannot be confused.
pub fn enabled() -> bool {
    window().is_some() && SPANS.get().is_some_and(|s| !s.is_empty())
}

/// Layers released and bytes advised away since start.
pub fn stats() -> Option<(u64, u64)> {
    enabled().then(|| {
        (
            RELEASED_LAYERS.load(Ordering::Relaxed),
            RELEASED_BYTES.load(Ordering::Relaxed),
        )
    })
}

/// Called as layer `layer`'s weights are first used.
///
/// Cheap enough for the hot path when the policy is off: one `OnceLock` read
/// and one relaxed compare. When it is on, the release runs only for the
/// thread that observes the layer change, and only once per layer.
pub fn touch(layer: u32) {
    let Some(window) = window() else { return };
    let layer = layer as usize;
    if CURRENT.load(Ordering::Relaxed) == layer {
        return;
    }
    // Whoever swaps in the new layer owns the release for it. A racing thread
    // sees its own swap fail and simply carries on into the matmul.
    if CURRENT.swap(layer, Ordering::Relaxed) == layer {
        return;
    }
    let Some(spans) = SPANS.get() else { return };
    // The layer `window` behind: far enough back that the sweep will not want
    // it again until it has been round every other layer.
    let Some(stale) = layer.checked_sub(window) else {
        return;
    };
    let Some(ranges) = spans.get(stale) else {
        return;
    };
    let _guard = RELEASING.lock();
    let mut freed = 0u64;
    for &(addr, len) in ranges {
        freed += release(addr, len);
    }
    if freed > 0 {
        RELEASED_LAYERS.fetch_add(1, Ordering::Relaxed);
        RELEASED_BYTES.fetch_add(freed, Ordering::Relaxed);
    }
}

/// Hands one mapped range back to the kernel, returning the bytes advised.
///
/// The same call and the same reasoning as `loader::release_mapped_range`:
/// `MADV_DONTNEED` on a read-only `MAP_PRIVATE` file mapping frees the page
/// table references without invalidating the address, so the next touch
/// faults the page back rather than reading stale data. Rounded *inward* to
/// whole pages, since a page is the unit `madvise` works in and a partial
/// page at either end belongs to a neighbour that may still be live.
fn release(addr: usize, len: usize) -> u64 {
    #[cfg(target_os = "linux")]
    {
        // Safety: `sysconf` is a pure query with no preconditions.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if page <= 0 {
            return 0;
        }
        let page = page as usize;
        let start = addr.next_multiple_of(page);
        let end = (addr + len) - ((addr + len) % page);
        if end <= start {
            return 0;
        }
        // Safety: the range lies inside a live read-only mapping this process
        // created and holds for its lifetime, the bounds are page-aligned as
        // `madvise` requires, and `MADV_DONTNEED` on a read-only file mapping
        // discards only clean pages — it cannot lose data.
        unsafe {
            libc::madvise(start as *mut libc::c_void, end - start, libc::MADV_DONTNEED);
        }
        (end - start) as u64
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (addr, len);
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With no window configured the policy is inert, which is the control
    /// arm D1a measured — and `touch` must stay cheap enough to sit in front
    /// of every matmul.
    #[test]
    fn a_model_with_no_window_configured_releases_nothing() {
        assert!(window().is_none() || !enabled());
        touch(0);
        touch(5);
        assert_eq!(stats(), None, "an inactive policy reports no counters");
    }

    /// Rounding is inward, so a span never advises away a byte outside
    /// itself. The neighbour it would take is another layer's weight.
    #[test]
    fn a_span_shorter_than_a_page_releases_nothing() {
        assert_eq!(release(4097, 10), 0, "no whole page inside the span");
    }
}
