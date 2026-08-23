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

//! A fixed pool of KV pages, reference counted so two requests can hold the
//! same one, with least-recently-used reclaim over the pages nobody holds.
//!
//! # The problem this exists for
//!
//! A `KvCache` today is private to one request: one contiguous buffer per
//! layer, sized from that request's own prompt plus `max_tokens`, and dropped
//! when it finishes. `engine::prefix_cache` softens that for *sequential*
//! requests — a finished cache goes into a pool and the next request with a
//! matching prompt prefix adopts its buffers — but `PrefixCache::
//! take_best_match` **removes** the entry it returns, deliberately, so that
//! two requests can never race to extend one cache.
//!
//! That removal is what makes concurrent sharing impossible. Two requests in
//! flight against the same long system prompt: the first takes the pooled
//! cache and reuses the prefix for nothing, the second finds an empty pool and
//! prefills every one of those tokens again. Measured on a prompt of a couple
//! of thousand tokens, the second request's time to first token is more than
//! twenty times the first's — not a tax on sharing, the *whole* prefill, paid
//! again per concurrent peer.
//!
//! Reference counting is the fix, and it is the only fix: the entry cannot be
//! left in the pool for a second reader until something tracks how many
//! readers there are.
//!
//! Nor is it enough to keep *more* entries. The pool can hold several finished
//! caches, and when it does, concurrent requests with a common prefix each take
//! their own — which avoids the recompute by storing the same keys once per
//! request. That trades prefill for memory, and memory is what runs out first:
//! measured at the point where the pool is against its ceiling, the slowest of
//! four concurrent requests on one shared prefix waited more than two hundred
//! times as long as the fastest, because the entries the others needed had been
//! evicted to make room for the copies. One copy with a count on it costs
//! neither.
//!
//! # Why a page and not a cache
//!
//! Sharing whole caches would only help requests whose prompts match
//! end-to-end. Prefixes diverge — a shared system prompt, then a different
//! question — so the unit of sharing has to be smaller than a request and
//! bigger than a token. A page is `page_tokens` consecutive positions, and a
//! sequence is a *list* of page indices rather than a range, which is what
//! lets two sequences agree on their first N pages and disagree after.
//!
//! # What a page is
//!
//! **One page index addresses every layer.** Page `p` holds token positions
//! `[p * page_tokens, (p + 1) * page_tokens)` for layer 0, and for layer 1,
//! and so on; each layer has its own storage region because `kv_dim` varies
//! along a model's depth. The alternative — a page per (layer, position) run —
//! would need a block table per layer per request and a reference count per
//! layer per page, to describe state that is always identical across layers,
//! since a token's key exists in every layer or in none.
//!
//! # The contiguous cache stays the default, and that is deliberate
//!
//! Paging is not an improvement on the single-client case and must not be
//! allowed to become a regression in it. Profiled at one client, the entire KV
//! machinery — the buffers, the device sync, the upload conversion — is a fifth
//! of one percent of decode CPU, and the device mirror already grows to the
//! rows in use rather than to a request's `max_tokens`, so its footprint does
//! not vary with how many tokens a request asked for. There is no cost there to
//! remove. A page table can only add an indirection to reads that are currently
//! a multiply.
//!
//! What paging addresses is the case one client never reaches: several requests
//! wanting the same prefix at the same time. So the engagement rule is that a
//! lone sequence keeps the contiguous cache, and the pool is what a *shared*
//! prefix moves into. The flag exists to make that switch measurable before it
//! is made automatic.
//!
//! # What this module does not do yet
//!
//! Allocation, reference counting and reclaim only. It holds host storage and
//! hands out page indices; it does not know what a token id is, does not index
//! prefixes, and nothing in the forward pass reads it. The prefix index that
//! decides *which* pages a new request should be given, and the device-side
//! mirror, are separate pieces built on top of this one.
//!
//! Off unless `ORANGU_PAGED_KV` is on.

// Nothing in the forward pass calls this yet — the flag reader and the page
// geometry are used only by this module's own tests until `LayerCache` learns
// to read through a page table. Allowed at module level rather than per item so
// the allow disappears in one edit when the wiring lands, instead of leaving a
// scatter of them to find.
#![allow(dead_code)]

use std::collections::VecDeque;
use std::sync::Mutex;

/// Whether the paged KV path is switched on. **On unless disabled.**
///
/// `ORANGU_PAGED_KV=0` opts out and is a real control arm — the off-list is
/// shared with every other tuning flag, so a sweep of `0,1` compares the
/// feature with its absence rather than with itself.
///
/// It defaults on because the measurements say the trade is one-sided: decode
/// is at parity with the contiguous path on both architectures tested, and four
/// concurrent requests sharing a prefix are served thirty times faster. The
/// opt-out exists because a configuration this has not been measured on should
/// have somewhere to go.
pub fn paged_kv_enabled() -> bool {
    crate::engine::env::flag_on_unless_disabled("ORANGU_PAGED_KV")
}

/// Which replacement rule the pool reclaims by — `ORANGU_KV_POLICY`.
///
/// `arc` (the default), `lru`, or `fifo`.
///
/// ARC is the default because it is the only rule that keeps a hot prefix
/// across a scan of unrelated traffic. Measured on two architectures with the
/// pool held at the same geometry — 73 pages, about four requests' worth — the
/// second use of a hot prefix after two or four unrelated requests took:
///
/// | | ARC | LRU | FIFO |
/// |---|---|---|---|
/// | one model | 1.2-1.4 s | 10.3-10.9 s | 10.4-12.0 s |
/// | the other | 1.6-2.2 s | 12.9-23.3 s | 13.2-29.5 s |
///
/// Seven to twelve times, in the same direction on both, with each rule
/// measured twice in different sweep positions so the ordering is not doing the
/// work. Where the pool is ample all three tie, so the cost of this default is
/// nothing and the benefit is an order of magnitude.
///
/// `lru` and `fifo` are kept because they are the controls this was measured
/// against, and they are indistinguishable from each other: across four
/// comparisons the winner alternates and every margin is inside the noise.
pub fn policy_from_env() -> Policy {
    policy_from_str(&std::env::var("ORANGU_KV_POLICY").unwrap_or_default())
}

/// The parse behind [`policy_from_env`], split out so it can be tested without
/// touching the environment — a test that set `ORANGU_KV_POLICY` would be
/// setting it for every other test running in the same process.
fn policy_from_str(raw: &str) -> Policy {
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "arc" => Policy::Arc,
        "lru" => Policy::Lru,
        "fifo" => Policy::Fifo,
        other => {
            // Named rather than silently defaulted: a sweep that mistypes a
            // policy would otherwise measure the default twice and report the
            // difference between a thing and itself.
            eprintln!(
                "orangu-server: ORANGU_KV_POLICY={other:?} is not arc, lru or fifo \
                 — using arc. This run measures the default, not the value you \
                 asked for."
            );
            Policy::Arc
        }
    }
}

/// Token positions one page covers.
///
/// A tuning value, not a constant of the design, and the trade has three
/// terms. Internal fragmentation on a sequence's last page costs
/// `page_tokens / 2` positions on average per sequence, which argues small. A
/// block table and the indirections a kernel does through it argue large. And
/// a prefix can only be shared in whole pages, so a coarse page means a longer
/// unshared remainder — which is latency, not throughput.
///
/// One term is specific to this engine and it settled the value. The GPU
/// attention kernels walk their window in tiles of 64 positions, so at
/// `page_tokens >= 64` a tile lies inside one page whenever the window start is
/// page aligned and the page lookup hoists out of the inner loop; below that a
/// tile can straddle a boundary and the loop restarts mid-tile. Measured at
/// four concurrent streams, aggregate decode was 50.4 / 53.8 / 53.6 tok/s at
/// 16 / 64 / 256 on one model and 53.4 / 52.9 / 50.4 on another. The two
/// disagree about the best size; 64 is the one within a percent of the best on
/// both, and it is where the tile stops straddling.
pub fn page_tokens() -> usize {
    crate::engine::backend::env_tuning_value(
        "ORANGU_KV_PAGE_TOKENS",
        64usize,
        "a positive power of two",
        |v: usize| v > 0 && v.is_power_of_two(),
    )
}

/// One layer's KV geometry: how wide a row is, and how many token positions a
/// row stands for.
///
/// `stride` is [`crate::engine::kv_cache::LayerCache`]'s own — 1 for an
/// ordinary per-token slot, 4 or 128 for a block-compressed one. It is carried
/// here rather than assumed to be 1 because a page is defined in *tokens*: a
/// strided layer stores `page_tokens / stride` rows where a plain one stores
/// `page_tokens`, and a pool that sized every layer the same would over-
/// allocate the strided ones and, worse, index them wrongly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerGeometry {
    pub kv_dim: usize,
    pub stride: usize,
}

impl LayerGeometry {
    /// Floats one page needs for this layer, for keys — and the same again for
    /// values.
    ///
    /// `div_ceil`, not a plain divide: a layer whose stride does not divide
    /// `page_tokens` still needs somewhere to put the partial row, and
    /// rounding down would make the last row of every page alias the first row
    /// of the next.
    fn floats_per_page(&self, page_tokens: usize) -> usize {
        page_tokens.div_ceil(self.stride) * self.kv_dim
    }

    /// The geometry of every layer of `probe`, which is meant to be a
    /// `ModelForward::new_kv_cache(1)` — a cache built at one token, which
    /// costs nothing and carries the whole shape.
    ///
    /// Read off the model rather than described a second time. `kv_dim` and
    /// `stride` vary along a model's depth, and the architectures where they
    /// vary most are the ones a pool most needs to size correctly, so a
    /// summary figure would be wrong exactly where it matters.
    pub fn of(probe: &crate::engine::kv_cache::KvCache) -> Vec<Self> {
        probe
            .layers
            .iter()
            .map(|l| Self {
                kv_dim: l.kv_dim(),
                stride: l.row_stride(),
            })
            .collect()
    }
}

/// Device bytes `floats` elements occupy in `storage`.
fn device_bytes_for(
    floats: usize,
    storage: crate::engine::backend::vulkan_shaders::KvStorage,
) -> u64 {
    use crate::engine::backend::vulkan_shaders::KvStorage;
    match storage {
        KvStorage::F32 => (floats * 4) as u64,
        KvStorage::F16 => (floats * 2) as u64,
        KvStorage::Q8_0 => (floats / 32 * 36) as u64,
    }
}

/// **Device** bytes one page costs across `layers`, in `storage`'s width.
///
/// Separate from [`page_bytes`] because the two are different sizes of the same
/// page: the host holds `f32`, the device holds whatever the KV storage setting
/// says. Sizing the pool from one and allocating the other is a mistake that
/// costs a card its headroom — see `KvPool::pages_within`.
pub fn device_page_bytes(
    layers: &[LayerGeometry],
    page_tokens: usize,
    storage: crate::engine::backend::vulkan_shaders::KvStorage,
) -> u64 {
    layers
        .iter()
        .map(|l| 2 * device_bytes_for(l.floats_per_page(page_tokens), storage))
        .sum()
}

/// Host bytes one page costs across `layers` — keys and values both.
pub fn page_bytes(layers: &[LayerGeometry], page_tokens: usize) -> usize {
    layers
        .iter()
        .map(|l| 2 * l.floats_per_page(page_tokens) * std::mem::size_of::<f32>())
        .sum()
}

/// A page nobody holds, and when it was last released.
///
/// The `seq` is a monotonic counter rather than a clock: it only ever has to
/// order two releases against each other, and a counter cannot go backwards,
/// tie, or cost a syscall the way reading a clock can.
#[derive(Clone, Copy, Debug)]
struct Reclaimable {
    page: u32,
    seq: u64,
}

/// Which zero-reference page a reclaim takes.
///
/// A swappable choice rather than a decision, following
/// [`crate::engine::expert_store::Policy`] — which exists because the first
/// replacement rule written there lost to plain recency at every budget, and
/// then *won* on a different model. The lesson recorded with it is that which
/// rule wins depends on the regime rather than on the rule, so the shape to
/// build is a knob with a measured default, not a winner.
///
/// The regime here is not the expert cache's. Expert traffic is a routing scan:
/// every expert is touched once per pass, in an order nothing controls. KV
/// prefix traffic has structure — a system prompt reused by every request, and
/// behind it each conversation's own tail, which is read once and never again.
/// That is the shape where recency alone is known to do badly, because the
/// one-shot tails are always the most recently used thing in the pool.
///
/// Measured on the engine as it stands: one unrelated request between two uses
/// of a hot prefix costs the second one a full re-prefill. That is not a policy
/// failure yet — today's pool has no capacity to speak of — but it is the
/// access pattern both rules below are being asked about.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Policy {
    /// First filled, first out. A page's position is fixed when it is given
    /// content and never moves again, however often it is used after that.
    ///
    /// Worth measuring rather than dismissing, for two reasons. It is what the
    /// engine's existing prefix pool already does — `prefix_cache` evicts with
    /// `entries.remove(0)`, which is insertion order, not use order — so it is
    /// the incumbent rather than a straw man. And it is the only rule here that
    /// needs *no bookkeeping on a hit*: recency has to move a page to the back
    /// on every access and ARC has to maintain four lists, while this touches
    /// nothing until something is reclaimed.
    ///
    /// Where it should lose is a working set that is re-used steadily: a page
    /// asked for on every request still ages out on schedule, because being
    /// used does not renew it.
    Fifo,
    /// Least recently released. No notion of how often a page was used.
    ///
    /// It has one piece of state per page, it is the rule the rest of this
    /// engine already uses for its other cache, and where the pool is merely
    /// *large enough* it ties with everything else — capacity is the
    /// first-order fix and this is the policy that does not pretend otherwise.
    ///
    /// It was the default until the regime it does badly in was measured on the
    /// engine rather than argued about. The prediction two paragraphs up turned
    /// out to be exactly right: one-shot conversation tails are always the most
    /// recently used thing in the pool, so a scan of unrelated requests evicts
    /// the shared prefix and every later request re-prefills it. See
    /// [`Arc`](Self::Arc).
    Lru,
    /// Adaptive replacement: split the reclaimable pages into those seen once
    /// and those seen again, and let the workload move the boundary.
    ///
    /// Four lists. `T1` holds pages released after a single use and `T2` those
    /// released after being reused; `B1` and `B2` are *ghosts* — the content
    /// tags of pages already reclaimed out of each, holding no data. A tag
    /// arriving that is in `B1` says the recency half was trimmed too far and
    /// grows its target `p`; one in `B2` says the same of the frequency half
    /// and shrinks it. Nothing is tuned by hand and the balance follows the
    /// traffic.
    ///
    /// The property being bought is scan resistance: a burst of one-shot
    /// prompts fills `T1` and evicts out of `T1`, leaving a prefix that has
    /// been used more than once sitting in `T2` where the burst cannot reach
    /// it. Under [`Lru`](Self::Lru) that burst is simply the most recent thing
    /// in the pool and the prefix goes first.
    ///
    /// Ghosts need a *content* identity, because a physical page is recycled
    /// and its index says nothing about what it last held. That is what
    /// [`KvPool::set_tag`] supplies, and why this module carries a tag it never
    /// interprets.
    ///
    /// **The default**, on two architectures agreeing — see
    /// [`policy_from_env`] for the numbers. The scan resistance above is not a
    /// theoretical property here: it is worth seven to twelve times on the
    /// second use of a hot prefix, and it is the only rule of the three whose
    /// ratio against a cold prefix stays at one. The reservation recorded on
    /// [`crate::engine::expert_store::Policy`] — that a rule winning on one
    /// model may lose on another — is why this waited for the second model
    /// rather than shipping on the first.
    #[default]
    Arc,
}

/// Which list a reclaimable page is sitting in, under [`Policy::Arc`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ArcList {
    /// Released after one use.
    T1,
    /// Released after being used again.
    T2,
}

/// [`Policy::Arc`]'s bookkeeping.
///
/// The two ghost lists are bounded by the pool's page count — the standard
/// bound, and the reason ARC's memory overhead is a constant per page rather
/// than growing with how much history it has seen.
struct ArcState {
    /// Target size of the recency half, in pages. Moves toward whichever ghost
    /// list is being hit.
    p: usize,
    t1: VecDeque<u32>,
    t2: VecDeque<u32>,
    /// Tags reclaimed out of `t1` / `t2`, oldest first.
    b1: VecDeque<u64>,
    b2: VecDeque<u64>,
    /// Which list each page is in, when it is reclaimable.
    list_of: Vec<Option<ArcList>>,
}

impl ArcState {
    fn new(num_pages: usize) -> Self {
        Self {
            p: 0,
            t1: VecDeque::new(),
            t2: VecDeque::new(),
            b1: VecDeque::new(),
            b2: VecDeque::new(),
            list_of: vec![None; num_pages],
        }
    }

    fn forget_ghost(&mut self, tag: u64) -> Option<bool> {
        if let Some(i) = self.b1.iter().position(|&t| t == tag) {
            self.b1.remove(i);
            return Some(true);
        }
        if let Some(i) = self.b2.iter().position(|&t| t == tag) {
            self.b2.remove(i);
            return Some(false);
        }
        None
    }

    fn push_ghost(&mut self, from: ArcList, tag: u64, cap: usize) {
        // A tag of 0 means "nothing meaningful was in this page" — a caller
        // that never set one. Recording it would make every untagged page look
        // like the same item and collapse the ghost lists onto one entry.
        if tag == 0 {
            return;
        }
        let list = match from {
            ArcList::T1 => &mut self.b1,
            ArcList::T2 => &mut self.b2,
        };
        list.push_back(tag);
        while list.len() > cap {
            list.pop_front();
        }
    }

    fn remove_page(&mut self, page: u32) {
        if let Some(list) = self.list_of[page as usize].take() {
            let q = match list {
                ArcList::T1 => &mut self.t1,
                ArcList::T2 => &mut self.t2,
            };
            if let Some(i) = q.iter().position(|&p| p == page) {
                q.remove(i);
            }
        }
    }
}

struct PoolInner {
    /// How many holders each page has. `0` means reclaimable, and a page at
    /// `0` is in exactly one of `never_used` or `lru`.
    refs: Vec<u32>,
    /// Pages that have never held anything, handed out before any reclaim.
    ///
    /// Kept apart from `lru` on purpose. A never-used page costs nothing to
    /// take; reclaiming one that still holds a prefix throws away work some
    /// future request might have reused. Draining this first means the pool
    /// fills before it starts forgetting.
    never_used: Vec<u32>,
    /// Zero-reference pages that *do* hold data, oldest release first.
    ///
    /// A queue rather than a sorted structure because releases arrive in
    /// increasing `seq` by construction, so pushing to the back keeps it
    /// ordered for free. Entries can be stale — a page released, re-acquired
    /// and released again appears twice — so [`PoolInner::take_reclaimable`]
    /// validates against `refs` and `release_seq` instead of trusting it.
    lru: VecDeque<Reclaimable>,
    /// The `seq` each page was last released at, so a stale `lru` entry can be
    /// recognised and dropped.
    release_seq: Vec<u64>,
    next_seq: u64,
    /// Pages currently held by at least one holder.
    live: usize,
    policy: Policy,
    /// [`Policy::Arc`]'s lists, `None` under [`Policy::Lru`] so the recency
    /// path carries none of its bookkeeping.
    arc: Option<ArcState>,
    /// What each page currently holds, as an opaque identity the pool never
    /// interprets. `0` means "nothing named".
    tag: Vec<u64>,
    /// Which page holds a given tag, for pages that are still resident —
    /// held or merely cached. A hit is a lookup here.
    by_tag: std::collections::HashMap<u64, u32>,
    /// Whether this page has been acquired more than once since it was filled.
    /// Decides `T1` against `T2` when it is released.
    seen_again: Vec<bool>,
    /// [`Policy::Fifo`]'s order: pages in the order they were *filled*, which
    /// is the one thing that separates it from recency — nothing here moves
    /// when a page is used.
    fifo: VecDeque<u32>,
    /// Whether a page has been published. An unsealed page is being written by
    /// its sole holder and is deliberately absent from `by_tag`.
    sealed: Vec<bool>,
    /// Next unused slot in the shared block-table buffer, and the regions
    /// finished sequences have handed back.
    table_next: usize,
    free_tables: Vec<(usize, usize)>,
}

impl PoolInner {
    /// One reclaimable page, preferring never-used, or `None` when every page
    /// is held.
    ///
    /// Never-used first under either policy: a page that has never held
    /// anything cannot be a cache hit for anybody, so taking it costs nothing,
    /// and taking a filled one instead would discard a prefix to leave an
    /// empty page sitting unused.
    fn take_reclaimable(&mut self) -> Option<u32> {
        if let Some(page) = self.never_used.pop() {
            return Some(page);
        }
        match self.policy {
            Policy::Fifo => self.take_fifo(),
            Policy::Lru => self.take_lru(),
            Policy::Arc => self.take_arc(),
        }
    }

    /// Oldest fill first, skipping anything currently held.
    fn take_fifo(&mut self) -> Option<u32> {
        let mut skipped = Vec::new();
        let page = loop {
            let candidate = self.fifo.pop_front()?;
            if self.refs[candidate as usize] == 0 {
                break candidate;
            }
            // Held right now: it cannot be reclaimed, but it has not been
            // refilled either, so it keeps its place rather than losing it.
            skipped.push(candidate);
        };
        for p in skipped.into_iter().rev() {
            self.fifo.push_front(p);
        }
        let tag = self.tag[page as usize];
        if tag != 0 {
            self.by_tag.remove(&tag);
        }
        Some(page)
    }

    /// Records that `page` has just been given content — the only event
    /// [`Policy::Fifo`] orders by.
    fn on_fill(&mut self, page: u32) {
        if self.policy == Policy::Fifo {
            if let Some(i) = self.fifo.iter().position(|&p| p == page) {
                self.fifo.remove(i);
            }
            self.fifo.push_back(page);
        }
    }

    fn take_lru(&mut self) -> Option<u32> {
        while let Some(entry) = self.lru.pop_front() {
            // Stale if the page was re-acquired since this entry was pushed
            // (`refs > 0`), or released again later (a newer entry for it is
            // still in the queue, and that one carries the current `seq`).
            if self.refs[entry.page as usize] == 0
                && self.release_seq[entry.page as usize] == entry.seq
            {
                return Some(entry.page);
            }
        }
        None
    }

    /// ARC's replacement rule: take from the recency list while it is at or
    /// over its target `p`, otherwise from the frequency list.
    ///
    /// This is the whole of the scan resistance. A burst of one-shot content
    /// arrives as misses, and a miss is admitted to `T1`; so the burst grows
    /// `T1` past `p` and then evicts *itself*, leaving whatever has been used
    /// more than once sitting in `T2` untouched. `p` starts at zero, which
    /// means the frequency half is protected from the first eviction onward
    /// rather than after some warm-up.
    fn take_arc(&mut self) -> Option<u32> {
        let cap = self.refs.len();
        let (from, page) = {
            let arc = self.arc.as_mut()?;
            let t1_len = arc.t1.len();
            let take_from_t1 = t1_len > 0 && t1_len >= arc.p.max(1).min(t1_len);
            let picked = if take_from_t1 {
                arc.t1.pop_front().map(|p| (ArcList::T1, p))
            } else {
                arc.t2
                    .pop_front()
                    .map(|p| (ArcList::T2, p))
                    .or_else(|| arc.t1.pop_front().map(|p| (ArcList::T1, p)))
            };
            picked?
        };
        let tag = self.tag[page as usize];
        if let Some(arc) = self.arc.as_mut() {
            arc.list_of[page as usize] = None;
            arc.push_ghost(from, tag, cap);
        }
        if tag != 0 {
            self.by_tag.remove(&tag);
        }
        Some(page)
    }

    /// Records a page's release into whichever structure the policy reads.
    fn on_release(&mut self, page: u32) {
        let seq = self.next_seq;
        self.next_seq += 1;
        self.release_seq[page as usize] = seq;
        match self.policy {
            // Nothing: a page's place was fixed when it was filled.
            Policy::Fifo => {}
            Policy::Lru => self.lru.push_back(Reclaimable { page, seq }),
            Policy::Arc => {
                let list = if self.seen_again[page as usize] {
                    ArcList::T2
                } else {
                    ArcList::T1
                };
                if let Some(arc) = self.arc.as_mut() {
                    arc.remove_page(page);
                    match list {
                        ArcList::T1 => arc.t1.push_back(page),
                        ArcList::T2 => arc.t2.push_back(page),
                    }
                    arc.list_of[page as usize] = Some(list);
                }
            }
        }
    }

    /// Takes a *specific* cached page back into use — the hit path.
    fn resurrect(&mut self, page: u32) {
        if self.policy == Policy::Arc
            && let Some(arc) = self.arc.as_mut()
        {
            arc.remove_page(page);
        }
        self.seen_again[page as usize] = true;
        self.refs[page as usize] = 1;
        self.live += 1;
    }

    /// Drops whatever identity a page carried, so a page being refilled is not
    /// still findable under the content it used to hold.
    fn forget_tag(&mut self, page: u32) {
        let old = std::mem::replace(&mut self.tag[page as usize], 0);
        if old != 0 {
            self.by_tag.remove(&old);
        }
        self.sealed[page as usize] = false;
    }

    /// A miss for `tag`: let the ghost lists move `p` before anything is
    /// reclaimed.
    ///
    /// A tag in `B1` means this page was thrown out of the recency half and
    /// wanted again, so the recency half is too small — grow `p`. A tag in
    /// `B2` says the same of the frequency half — shrink it. The step sizes
    /// are ARC's own: the ratio of the two ghost lists, so a rarely-hit ghost
    /// list moves `p` further per hit than a busy one.
    fn adapt(&mut self, tag: u64) {
        if self.policy != Policy::Arc || tag == 0 {
            return;
        }
        let cap = self.refs.len();
        let Some(arc) = self.arc.as_mut() else {
            return;
        };
        match arc.forget_ghost(tag) {
            Some(true) => {
                let delta = if arc.b1.is_empty() {
                    1
                } else {
                    (arc.b2.len() / arc.b1.len()).max(1)
                };
                arc.p = (arc.p + delta).min(cap);
            }
            Some(false) => {
                let delta = if arc.b2.is_empty() {
                    1
                } else {
                    (arc.b1.len() / arc.b2.len()).max(1)
                };
                arc.p = arc.p.saturating_sub(delta);
            }
            None => {}
        }
    }
}

/// Pages, their reference counts, and the host bytes behind them.
pub struct KvPool {
    page_tokens: usize,
    num_pages: usize,
    layers: Vec<LayerGeometry>,
    /// `layers[l]`'s keys for every page, one flat buffer:
    /// page `p`'s rows start at `p * layers[l].floats_per_page(page_tokens)`.
    ///
    /// One allocation per layer rather than one per page: a page is a few
    /// kilobytes, a pool has thousands, and the addressing is the same
    /// multiply either way.
    k: Vec<std::cell::UnsafeCell<Vec<f32>>>,
    v: Vec<std::cell::UnsafeCell<Vec<f32>>>,
    /// The same pages on the device, when a device was attached.
    ///
    /// Allocated once, for the whole pool, rather than once per request —
    /// which is the point. Two sequences sharing a prefix currently mirror it
    /// twice, because the mirror is per request; sharing on the device means
    /// sharing *this*.
    device: Option<DevicePages>,
    inner: Mutex<PoolInner>,
}

/// The pool's device-side pages: one buffer per layer, plus the block tables
/// every live sequence's kernels read.
pub struct DevicePages {
    /// Per layer, `num_pages` pages of rows in the backend's KV storage width.
    pub layers: Vec<wgpu::Buffer>,
    /// Every sequence's block table, concatenated. A sequence is handed a base
    /// offset into this and reads `table[base + logical_page]`.
    ///
    /// One buffer rather than one per sequence: a bind group names a buffer,
    /// and rebuilding bind groups per request is what the per-layer attention
    /// dispatch cache exists to avoid. A base offset in the meta uniform costs
    /// nothing and keeps the binding stable for the process's life.
    pub table: wgpu::Buffer,
    /// How many `u32` entries `table` holds.
    pub table_len: usize,
    storage: crate::engine::backend::vulkan_shaders::KvStorage,
}

/// # Safety
///
/// The page storage is in `UnsafeCell`s, which are not `Sync` by default. What
/// makes sharing this across threads sound is a single invariant, enforced by
/// the state machine in [`PoolInner`] rather than by convention:
///
/// **A page is either being written by exactly one holder, or published and
/// read-only. It is never both, and never neither.**
///
/// - A page becomes writable only by coming out of [`KvPool::acquire`] as a
///   *miss*. At that moment its reference count is 1, it is in no reclaim list,
///   and — the load-bearing part — it is **not in `by_tag`**, so no other
///   caller can find it. Its sole holder is the one that just took it.
/// - [`KvPool::fill`] is the only writer, and it refuses any page that is
///   already sealed or that has more than one holder.
/// - [`KvPool::seal`] publishes the page into `by_tag` and marks it immutable.
///   Only after that can a second holder reach it, and from then on nothing
///   writes it: the next write can only follow a reclaim, which first removes
///   it from `by_tag` and can only happen at zero references.
///
/// So the window in which writes occur is exactly the window in which the page
/// is unreachable to everyone else, and every path into that window goes
/// through the one mutex. Publishing before filling would break this, which is
/// why the tag is recorded on the page at acquire time but inserted into the
/// lookup only at seal.
unsafe impl Sync for KvPool {}

/// Why an allocation could not be served.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocError {
    /// Every page in the pool is held by somebody. The caller's options are to
    /// wait, to refuse the request, or to shorten it — never to grow the pool,
    /// which is fixed so that the memory an operator was promised at startup
    /// stays the memory in use.
    Exhausted { wanted: usize, available: usize },
}

impl std::fmt::Display for AllocError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AllocError::Exhausted { wanted, available } => write!(
                f,
                "KV pool exhausted: {wanted} pages wanted, {available} reclaimable"
            ),
        }
    }
}

/// One page from [`KvPool::acquire`], and whether its content was already
/// there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Acquired {
    pub page: u32,
    /// `true` when the content was already resident — the caller must not
    /// overwrite it, and did not have to compute it.
    pub hit: bool,
}

/// A snapshot of what the pool is doing, for tests and for an operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolStats {
    pub total_pages: usize,
    /// Pages with at least one holder.
    pub live_pages: usize,
    /// Zero-reference pages that still hold data — reusable if something asks
    /// for exactly them, reclaimable otherwise.
    pub cached_pages: usize,
    /// Pages that have never held anything.
    pub free_pages: usize,
}

impl KvPool {
    /// A pool of `num_pages` pages over `layers`, with host storage allocated
    /// up front.
    ///
    /// Allocated, not reserved: the whole point of a fixed pool is that its
    /// footprint is decided once, at startup, where an operator can be told
    /// about it — rather than growing under a request and failing at whatever
    /// moment the machine happens to run out.
    pub fn new(num_pages: usize, page_tokens: usize, layers: Vec<LayerGeometry>) -> Self {
        Self::with_policy(num_pages, page_tokens, layers, Policy::default())
    }

    /// [`Self::new`] under a named replacement policy — the entry point a sweep
    /// uses, and the one the tests compare arms through.
    pub fn with_policy(
        num_pages: usize,
        page_tokens: usize,
        layers: Vec<LayerGeometry>,
        policy: Policy,
    ) -> Self {
        assert!(num_pages > 0, "a KV pool needs at least one page");
        assert!(page_tokens > 0, "a KV page must cover at least one token");
        assert!(
            layers.iter().all(|l| l.stride > 0),
            "every layer's stride must be at least one token"
        );
        let cells = || -> Vec<std::cell::UnsafeCell<Vec<f32>>> {
            layers
                .iter()
                .map(|l| {
                    std::cell::UnsafeCell::new(vec![
                        0.0;
                        num_pages * l.floats_per_page(page_tokens)
                    ])
                })
                .collect()
        };
        let k = cells();
        let v = cells();
        let device = None;
        Self {
            page_tokens,
            num_pages,
            layers,
            k,
            v,
            device,
            inner: Mutex::new(PoolInner {
                refs: vec![0; num_pages],
                never_used: (0..num_pages as u32).rev().collect(),
                lru: VecDeque::new(),
                release_seq: vec![0; num_pages],
                next_seq: 1,
                live: 0,
                policy,
                arc: (policy == Policy::Arc).then(|| ArcState::new(num_pages)),
                tag: vec![0; num_pages],
                by_tag: std::collections::HashMap::new(),
                seen_again: vec![false; num_pages],
                fifo: VecDeque::new(),
                sealed: vec![false; num_pages],
                table_next: 0,
                free_tables: Vec::new(),
            }),
        }
    }

    /// Gives the pool device-side pages, so a shared prefix is mirrored once
    /// rather than once per request.
    ///
    /// `max_table_entries` bounds the concatenated block tables: one sequence's
    /// table is `pages_for(its context)` entries, so this is the slot count
    /// times the longest context a slot may hold. Fixed at attach time for the
    /// same reason the page count is — a buffer that grew under a request would
    /// invalidate every bind group naming it, which is exactly the per-token
    /// cost the attention dispatch cache exists to avoid.
    ///
    /// Returns `false` if the device declines an allocation, leaving the pool
    /// host-only and usable rather than half-attached.
    pub fn attach_device(
        &mut self,
        device: &wgpu::Device,
        storage: crate::engine::backend::vulkan_shaders::KvStorage,
        max_table_entries: usize,
    ) -> bool {
        use crate::engine::backend::vulkan_shaders::KvStorage;
        let elem_bytes = |floats: usize| -> u64 {
            match storage {
                KvStorage::F32 => (floats * 4) as u64,
                KvStorage::F16 => (floats * 2) as u64,
                // 9 words per 32-element block — see `KvStorage::Q8_0`.
                KvStorage::Q8_0 => (floats / 32 * 36) as u64,
            }
        };
        let mut layers = Vec::with_capacity(self.layers.len() * 2);
        for geom in &self.layers {
            let per_page = geom.floats_per_page(self.page_tokens);
            // Keys and values in one buffer per layer, as the per-request
            // mirror already does: a per-token decode submission re-validates
            // every referenced buffer, so two regions of one is cheaper than
            // two buffers.
            let size = elem_bytes(per_page * self.num_pages) * 2;
            layers.push(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("orangu-server kv pool layer"),
                size: size.max(4),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let table = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("orangu-server kv page table"),
            size: (max_table_entries.max(1) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.device = Some(DevicePages {
            layers,
            table,
            table_len: max_table_entries.max(1),
            storage,
        });
        true
    }

    /// The device pages, if a device was attached.
    pub fn device_pages(&self) -> Option<&DevicePages> {
        self.device.as_ref()
    }

    /// Device bytes the pool holds — the figure an operator compares against
    /// the card, and the one that no longer scales with the number of requests.
    pub fn device_bytes(&self) -> u64 {
        self.device
            .as_ref()
            .map_or(0, |d| d.layers.iter().map(|b| b.size()).sum())
    }

    /// Where `layer`'s keys and values for `page` start in that layer's device
    /// buffer.
    ///
    /// Keys occupy the first half of the buffer and values the second, so a
    /// page's two regions are one page-stride apart plus the half-buffer
    /// offset — the same split `GpuLayerCache` uses, kept identical so the
    /// shader's `k_cache`/`v_cache` bindings mean the same thing under either.
    pub fn device_page_offsets(&self, layer: usize, page: u32) -> Option<(u64, u64, u64)> {
        let d = self.device.as_ref()?;
        let per_page = self.layers[layer].floats_per_page(self.page_tokens);
        let stride = device_bytes_for(per_page, d.storage);
        let half = stride * self.num_pages as u64;
        let base = stride * u64::from(page);
        Some((base, half + base, stride))
    }

    /// Writes one page's rows for one layer into the device pages.
    ///
    /// The host-side counterpart is [`Self::fill`], and this deliberately does
    /// **not** re-check the seal: the two are called together for the same page
    /// under the same exclusive hold, and duplicating the check would let them
    /// disagree about which one is authoritative.
    pub fn fill_device(
        &self,
        queue: &wgpu::Queue,
        layer: usize,
        page: u32,
        k_rows: &[f32],
        v_rows: &[f32],
    ) {
        use crate::engine::backend::vulkan_shaders::KvStorage;
        let Some(d) = self.device.as_ref() else {
            return;
        };
        let Some((k_off, v_off, _)) = self.device_page_offsets(layer, page) else {
            return;
        };
        let convert = |rows: &[f32]| -> Vec<u8> {
            match d.storage {
                KvStorage::F32 => bytemuck::cast_slice(rows).to_vec(),
                KvStorage::F16 => crate::engine::kv_cache::f32_to_f16_bytes(rows),
                KvStorage::Q8_0 => crate::engine::kv_cache::f32_to_q8_0_bytes(rows),
            }
        };
        queue.write_buffer(&d.layers[layer], k_off, &convert(k_rows));
        queue.write_buffer(&d.layers[layer], v_off, &convert(v_rows));
    }

    /// Reserves `entries` consecutive slots in the shared block-table buffer.
    ///
    /// A bump allocator with a free list, not a page allocator: table regions
    /// are per *sequence* and there are at most a few of them (one per slot),
    /// where pages are per sixteen tokens and there are thousands. Returns
    /// `None` when the buffer is full, which the caller must treat as "this
    /// sequence cannot use the paged device path" rather than as a failure —
    /// the per-request mirror still serves it correctly.
    pub fn alloc_table(&self, entries: usize) -> Option<usize> {
        let mut inner = self.inner.lock().expect("kv pool poisoned");
        let cap = self.device.as_ref()?.table_len;
        if let Some(i) = inner
            .free_tables
            .iter()
            .position(|&(_, len)| len >= entries)
        {
            let (base, len) = inner.free_tables.remove(i);
            if len > entries {
                inner.free_tables.push((base + entries, len - entries));
            }
            return Some(base);
        }
        if inner.table_next + entries <= cap {
            let base = inner.table_next;
            inner.table_next += entries;
            return Some(base);
        }
        None
    }

    /// Returns a region taken by [`Self::alloc_table`].
    pub fn free_table(&self, base: usize, entries: usize) {
        let mut inner = self.inner.lock().expect("kv pool poisoned");
        inner.free_tables.push((base, entries));
    }

    /// Writes a *range of rows* of one page, rather than the whole page.
    ///
    /// The tail of a sequence grows a row at a time, and re-sending the whole
    /// page on every decode step is `page_tokens` times the traffic the change
    /// actually made — per layer, per token. This writes only what is new.
    ///
    /// `first_row` must land on a block boundary under `Q8_0`, which it does
    /// for every real model: that storage already requires `kv_dim % 32 == 0`
    /// (`VulkanBackend::try_init` checks it), so a whole row is a whole number
    /// of blocks.
    pub fn fill_device_rows(
        &self,
        queue: &wgpu::Queue,
        layer: usize,
        page: u32,
        first_row: usize,
        k_rows: &[f32],
        v_rows: &[f32],
    ) {
        use crate::engine::backend::vulkan_shaders::KvStorage;
        let Some(d) = self.device.as_ref() else {
            return;
        };
        let Some((k_off, v_off, _)) = self.device_page_offsets(layer, page) else {
            return;
        };
        let kv_dim = self.layers[layer].kv_dim;
        if kv_dim == 0 || k_rows.is_empty() {
            return;
        }
        if d.storage == KvStorage::Q8_0 {
            assert_eq!(
                (first_row * kv_dim) % 32,
                0,
                "a Q8_0 row range must start on a block boundary"
            );
        }
        let within = device_bytes_for(first_row * kv_dim, d.storage);
        let convert = |rows: &[f32]| -> Vec<u8> {
            match d.storage {
                KvStorage::F32 => bytemuck::cast_slice(rows).to_vec(),
                KvStorage::F16 => crate::engine::kv_cache::f32_to_f16_bytes(rows),
                KvStorage::Q8_0 => crate::engine::kv_cache::f32_to_q8_0_bytes(rows),
            }
        };
        queue.write_buffer(&d.layers[layer], k_off + within, &convert(k_rows));
        queue.write_buffer(&d.layers[layer], v_off + within, &convert(v_rows));
    }

    /// Uploads one sequence's block table at `base`.
    ///
    /// The kernel reads `table[base + logical_page]`, so `base` is what the
    /// meta uniform carries and the buffer itself never changes shape.
    pub fn write_table(&self, queue: &wgpu::Queue, base: usize, pages: &[u32]) {
        let Some(d) = self.device.as_ref() else {
            return;
        };
        assert!(
            base + pages.len() <= d.table_len,
            "block table overflow: sequence at {base} needs {} entries of {}",
            pages.len(),
            d.table_len
        );
        queue.write_buffer(&d.table, (base * 4) as u64, bytemuck::cast_slice(pages));
    }

    /// A pool that fits in `budget_bytes`, with at least one page.
    ///
    /// **The budget is not divided by slots, and that is the whole point.**
    /// `footprint::DeviceFootprint::kv_tokens_in` multiplies its per-token cost
    /// by the slot count, because today every slot owns a private cache and the
    /// memory has to cover the worst case of all of them at once. That is what
    /// makes a four-slot server advertise a quarter of the context a one-slot
    /// server does, on the same card, for the same model. A pool is the
    /// opposite arrangement: one budget, drawn against by whoever is running,
    /// so a lone deep request can use all of it and four shallow ones share it.
    ///
    /// Returns `None` when the budget will not hold a single page — which the
    /// caller must treat as "do not build a pool", not as "build an empty one".
    /// An empty pool cannot serve any request at all, and finding that out at
    /// the first token rather than at startup is the failure this refuses to
    /// hand back.
    pub fn sized_for(
        budget_bytes: u64,
        page_tokens: usize,
        layers: Vec<LayerGeometry>,
        policy: Policy,
    ) -> Option<Self> {
        let pages = Self::pages_in(budget_bytes, &layers, page_tokens);
        (pages > 0).then(|| Self::with_policy(pages, page_tokens, layers, policy))
    }

    /// Pages that fit in **both** budgets — the host bytes the pool may take
    /// and, when there is a device, the device bytes its pages will occupy.
    ///
    /// The two are not the same size and the device one is easy to forget: a
    /// pool sized from a host budget allocates that many *device* pages too,
    /// and on a small card that quietly takes headroom the model needs.
    /// Measured on a 4 GiB card with a 1.9 GiB model, a 2 GiB host budget put
    /// about a gigabyte of pages on the device and cost **31% of decode** with a
    /// spread eight times the contiguous path's; the same pool at 512 MiB ran
    /// at parity. The regression was never the paging — it was the pool
    /// crowding out the weights.
    pub fn pages_within(
        host_bytes: u64,
        device_bytes: Option<u64>,
        layers: &[LayerGeometry],
        page_tokens: usize,
        storage: crate::engine::backend::vulkan_shaders::KvStorage,
    ) -> usize {
        let by_host = Self::pages_in(host_bytes, layers, page_tokens);
        match device_bytes {
            None => by_host,
            Some(bytes) => {
                let per = device_page_bytes(layers, page_tokens, storage);
                let by_device = bytes.checked_div(per).unwrap_or(0) as usize;
                by_host.min(by_device)
            }
        }
    }

    /// How many pages of this geometry fit in `budget_bytes`.
    pub fn pages_in(budget_bytes: u64, layers: &[LayerGeometry], page_tokens: usize) -> usize {
        let per = page_bytes(layers, page_tokens) as u64;
        if per == 0 {
            return 0;
        }
        (budget_bytes / per) as usize
    }

    /// Token positions the whole pool can hold at once, across every sequence
    /// drawing on it — the number an operator compares against a context
    /// length.
    pub fn token_capacity(&self) -> usize {
        self.num_pages * self.page_tokens
    }

    pub fn page_tokens(&self) -> usize {
        self.page_tokens
    }

    pub fn num_pages(&self) -> usize {
        self.num_pages
    }

    pub fn layers(&self) -> &[LayerGeometry] {
        &self.layers
    }

    /// Host bytes this pool holds, both halves.
    pub fn host_bytes(&self) -> usize {
        // SAFETY: reads only the length, which no writer changes — `fill`
        // overwrites elements in place and never resizes.
        self.k
            .iter()
            .chain(self.v.iter())
            .map(|b| {
                let buf: &Vec<f32> = unsafe { &*b.get() };
                buf.len() * std::mem::size_of::<f32>()
            })
            .sum()
    }

    /// How many pages a sequence of `tokens` positions needs.
    pub fn pages_for(&self, tokens: usize) -> usize {
        tokens.div_ceil(self.page_tokens)
    }

    /// The largest number of pages that could be served right now — held pages
    /// excluded, cached ones included, since those are reclaimable.
    pub fn available(&self) -> usize {
        let inner = self.inner.lock().expect("kv pool poisoned");
        self.num_pages - inner.live
    }

    pub fn stats(&self) -> PoolStats {
        let inner = self.inner.lock().expect("kv pool poisoned");
        let free = inner.never_used.len();
        PoolStats {
            total_pages: self.num_pages,
            live_pages: inner.live,
            cached_pages: self.num_pages - inner.live - free,
            free_pages: free,
        }
    }

    /// Takes `n` pages, each returned with a reference count of one.
    ///
    /// All or nothing. A partial allocation would leave the caller holding
    /// pages it cannot use and having to unwind them itself, and every caller
    /// would write that unwind slightly differently.
    pub fn alloc(&self, n: usize) -> Result<Vec<u32>, AllocError> {
        let mut inner = self.inner.lock().expect("kv pool poisoned");
        let available = self.num_pages - inner.live;
        if n > available {
            return Err(AllocError::Exhausted {
                wanted: n,
                available,
            });
        }
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let page = inner
                .take_reclaimable()
                .expect("availability was checked under this same lock");
            inner.forget_tag(page);
            inner.refs[page as usize] = 1;
            inner.seen_again[page as usize] = false;
            inner.sealed[page as usize] = false;
            inner.live += 1;
            inner.on_fill(page);
            out.push(page);
        }
        Ok(out)
    }

    /// Takes the pages holding `tags`, filling in whatever is not resident.
    ///
    /// This is the call a prefix index makes, and it is where sharing actually
    /// happens. For each tag, in order:
    ///
    /// - **resident and held** — someone else is using this exact content, so
    ///   take another reference to *their* page. One copy, two holders. This is
    ///   the case today's engine cannot express at all: its pool hands a cached
    ///   prefix to one request and removes it, so every concurrent peer
    ///   recomputes.
    /// - **resident and idle** — a cached page holding this content: take it
    ///   back. Under [`Policy::Arc`] this is what promotes a page from the
    ///   recency list to the frequency list, which is the signal a scan cannot
    ///   fake.
    /// - **not resident** — a miss. The ghost lists get their say on `p`
    ///   (under ARC), a page is reclaimed, and it is stamped with this tag.
    ///
    /// Returns one entry per tag, in order, saying which page and whether it
    /// was a hit — the caller needs the distinction to know which pages it must
    /// still fill with real keys and values.
    ///
    /// All or nothing, like [`Self::alloc`]: on exhaustion nothing is taken,
    /// including the hits, because a caller holding half a prefix has to unwind
    /// it and every caller would do that differently.
    ///
    /// A tag of `0` is reserved for "unnamed" and always misses.
    pub fn acquire(&self, tags: &[u64]) -> Result<Vec<Acquired>, AllocError> {
        let mut inner = self.inner.lock().expect("kv pool poisoned");

        // Price the request before touching anything: only the misses need a
        // page, and only the ones that are not already resident.
        let mut needed = 0usize;
        let mut seen_misses: Vec<u64> = Vec::new();
        for &tag in tags {
            if tag != 0 && inner.by_tag.contains_key(&tag) {
                continue;
            }
            // Two misses on the same tag in one call need one page, not two.
            if tag != 0 && seen_misses.contains(&tag) {
                continue;
            }
            if tag != 0 {
                seen_misses.push(tag);
            }
            needed += 1;
        }
        let available = self.num_pages - inner.live;
        if needed > available {
            return Err(AllocError::Exhausted {
                wanted: needed,
                available,
            });
        }

        let mut out = Vec::with_capacity(tags.len());
        // Tags this call has already given a page to. Needed because a page is
        // published only at `seal`, so the second occurrence of a tag within
        // one call cannot find the first through `by_tag` — and allocating it a
        // second page would both duplicate the content and take one more page
        // than the pricing loop above reserved.
        let mut assigned: Vec<(u64, u32)> = Vec::new();
        for &tag in tags {
            if tag != 0
                && let Some(&(_, page)) = assigned.iter().find(|(t, _)| *t == tag)
            {
                inner.refs[page as usize] += 1;
                inner.seen_again[page as usize] = true;
                out.push(Acquired { page, hit: true });
                continue;
            }
            if tag != 0
                && let Some(&page) = inner.by_tag.get(&tag)
            {
                if inner.refs[page as usize] > 0 {
                    inner.refs[page as usize] += 1;
                    inner.seen_again[page as usize] = true;
                } else {
                    inner.resurrect(page);
                }
                assigned.push((tag, page));
                out.push(Acquired { page, hit: true });
                continue;
            }
            inner.adapt(tag);
            let page = inner
                .take_reclaimable()
                .expect("availability was checked under this same lock");
            inner.forget_tag(page);
            inner.refs[page as usize] = 1;
            inner.seen_again[page as usize] = false;
            inner.live += 1;
            // Recorded on the page, but **not** inserted into `by_tag`: an
            // unfilled page must not be findable, or a second request would
            // share zeroes. `seal` publishes it once it holds something.
            inner.tag[page as usize] = tag;
            inner.sealed[page as usize] = false;
            inner.on_fill(page);
            if tag != 0 {
                assigned.push((tag, page));
            }
            out.push(Acquired { page, hit: false });
        }
        Ok(out)
    }

    /// The replacement policy in force.
    pub fn policy(&self) -> Policy {
        self.inner.lock().expect("kv pool poisoned").policy
    }

    /// Whether `tag`'s content is resident — a test and introspection helper,
    /// and what a hit-rate counter would read.
    pub fn holds(&self, tag: u64) -> bool {
        self.inner
            .lock()
            .expect("kv pool poisoned")
            .by_tag
            .contains_key(&tag)
    }

    /// Takes one more reference to each of `pages` — how a second request
    /// comes to share a prefix instead of recomputing it.
    ///
    /// Panics on a page nobody holds. That is not defensiveness: retaining a
    /// zero-reference page means the caller is handing out indices it read
    /// from something that did not keep them alive, and the pages it thinks it
    /// is sharing may already describe a different sequence. Failing loudly
    /// here is the difference between a bug and a request that quietly answers
    /// from another conversation's keys.
    pub fn retain(&self, pages: &[u32]) {
        let mut inner = self.inner.lock().expect("kv pool poisoned");
        for &p in pages {
            assert!(
                inner.refs[p as usize] > 0,
                "retained page {p} has no holder; it is not this caller's to share"
            );
            inner.refs[p as usize] += 1;
        }
    }

    /// Drops one reference to each of `pages`. A page reaching zero keeps its
    /// contents and joins the reclaim queue, so a later request whose prefix
    /// matches can still be given it.
    pub fn release(&self, pages: &[u32]) {
        let mut inner = self.inner.lock().expect("kv pool poisoned");
        for &p in pages {
            let refs = &mut inner.refs[p as usize];
            assert!(*refs > 0, "released page {p} was not held");
            *refs -= 1;
            if *refs == 0 {
                inner.live -= 1;
                inner.on_release(p);
            }
        }
    }

    /// The reference count of `page`, for tests and introspection.
    pub fn refs(&self, page: u32) -> u32 {
        self.inner.lock().expect("kv pool poisoned").refs[page as usize]
    }

    /// Keys for `page` in `layer`, the whole page's rows.
    ///
    /// Valid for a sealed page (which is immutable, so any number of readers
    /// may hold this at once) or for an unsealed page read by its own sole
    /// holder. Reading an unsealed page held by *somebody else* is what the
    /// type system cannot express here and what the debug assertion below
    /// catches: it is the one way to observe a half-written page.
    pub fn page_k(&self, layer: usize, page: u32) -> &[f32] {
        self.debug_assert_readable(page);
        // SAFETY: see the `unsafe impl Sync` above. A sealed page is never
        // written again until it is reclaimed, which requires zero holders.
        unsafe { Self::page_slice(&self.k[layer], self.layers[layer], self.page_tokens, page) }
    }

    pub fn page_v(&self, layer: usize, page: u32) -> &[f32] {
        self.debug_assert_readable(page);
        // SAFETY: as `page_k`.
        unsafe { Self::page_slice(&self.v[layer], self.layers[layer], self.page_tokens, page) }
    }

    /// A layer's whole page-storage buffer, for a caller that indexes it with
    /// [`Self::row_offset`] rather than a page at a time.
    ///
    /// Exists because a paged `LayerCache` returns one *row* — the same shape
    /// the contiguous path returns — and slicing a row out of a per-page slice
    /// would mean computing the page base twice. The safety argument is the
    /// same as [`Self::page_k`]'s and no weaker: this hands out a shared
    /// reference to storage whose written pages are sealed and whose unsealed
    /// pages are reachable only by their sole holder.
    pub fn page_k_all(&self, layer: usize) -> &[f32] {
        // SAFETY: see the `unsafe impl Sync` on this type.
        let buf: &Vec<f32> = unsafe { &*self.k[layer].get() };
        buf
    }

    pub fn page_v_all(&self, layer: usize) -> &[f32] {
        // SAFETY: as `page_k_all`.
        let buf: &Vec<f32> = unsafe { &*self.v[layer].get() };
        buf
    }

    /// # Safety
    ///
    /// The caller must hold the invariant in this type's `unsafe impl Sync`:
    /// the page is sealed, or is unsealed and held only by this caller.
    unsafe fn page_slice(
        cell: &std::cell::UnsafeCell<Vec<f32>>,
        geom: LayerGeometry,
        page_tokens: usize,
        page: u32,
    ) -> &[f32] {
        let per = geom.floats_per_page(page_tokens);
        let base = page as usize * per;
        // Through the raw pointer's own `as_ref`, not an autoref on a place
        // expression: the latter is the pattern that hides an aliasing
        // requirement behind an indexing operator.
        let buf: &Vec<f32> = unsafe { &*cell.get() };
        &buf[base..base + per]
    }

    fn debug_assert_readable(&self, page: u32) {
        debug_assert!(
            {
                let inner = self.inner.lock().expect("kv pool poisoned");
                inner.sealed[page as usize] || inner.refs[page as usize] <= 1
            },
            "page {page} is unsealed and shared; reading it can observe a \
             half-written page"
        );
    }

    /// Writes one layer's rows of a page. The only writer.
    ///
    /// Refuses a sealed page or one with more than one holder — the two states
    /// in which somebody else could be reading it. That is not defensive
    /// programming; it is the check that makes [`page_k`](Self::page_k) sound,
    /// so it is an assertion rather than an `Option`.
    ///
    /// `rows` must be the whole page's worth for this layer. A partially filled
    /// page is fine to *hold* — a sequence's tail is one — but it is filled by
    /// writing the whole span with the unused tail left as it is, so that no
    /// reader has to know how much of a page is meaningful. How many of its
    /// positions are live is the sequence's business, not the pool's.
    pub fn fill(&self, layer: usize, page: u32, k_rows: &[f32], v_rows: &[f32]) {
        let per = self.layers[layer].floats_per_page(self.page_tokens);
        assert_eq!(k_rows.len(), per, "k rows must fill the page exactly");
        assert_eq!(v_rows.len(), per, "v rows must fill the page exactly");
        {
            let inner = self.inner.lock().expect("kv pool poisoned");
            assert!(
                !inner.sealed[page as usize],
                "page {page} is sealed; a published page is read by others and \
                 must never be written again"
            );
            assert_eq!(
                inner.refs[page as usize], 1,
                "page {page} is not exclusively held; writing it could be \
                 observed half-done"
            );
        }
        let base = page as usize * per;
        // SAFETY: checked above — unsealed and singly held, so this page is in
        // the window described by this type's `unsafe impl Sync`, unreachable
        // to every other caller.
        let k: &mut Vec<f32> = unsafe { &mut *self.k[layer].get() };
        k[base..base + per].copy_from_slice(k_rows);
        let v: &mut Vec<f32> = unsafe { &mut *self.v[layer].get() };
        v[base..base + per].copy_from_slice(v_rows);
    }

    /// Publishes a filled page: it becomes immutable and findable by its tag,
    /// so other requests can share it.
    ///
    /// Separate from [`fill`](Self::fill) because a page spans every layer and
    /// is only complete once all of them are written. Sealing per layer would
    /// publish a page whose later layers are still zeroes.
    pub fn seal(&self, page: u32) {
        let mut inner = self.inner.lock().expect("kv pool poisoned");
        assert!(
            inner.refs[page as usize] > 0,
            "page {page} was sealed after being released"
        );
        inner.sealed[page as usize] = true;
        let tag = inner.tag[page as usize];
        if tag != 0 {
            inner.by_tag.insert(tag, page);
        }
    }

    /// Whether `page` has been published.
    pub fn is_sealed(&self, page: u32) -> bool {
        self.inner.lock().expect("kv pool poisoned").sealed[page as usize]
    }

    /// Where `layer`'s row `row_in_page` of `page` starts in the flat buffer —
    /// the one place the page addressing is written down.
    pub fn row_offset(&self, layer: usize, page: u32, row_in_page: usize) -> usize {
        let g = self.layers[layer];
        let rows = self.page_tokens.div_ceil(g.stride);
        debug_assert!(
            row_in_page < rows,
            "row {row_in_page} is past this page's {rows} rows"
        );
        page as usize * g.floats_per_page(self.page_tokens) + row_in_page * g.kv_dim
    }
}

#[cfg(test)]
mod tests {
    /// The default policy has two independent definitions — the `Default`
    /// derive on the enum and the empty-string arm of the parse — and nothing
    /// makes them agree. They have to be pinned together, because a change to
    /// one and not the other is silent: the server would run one rule and
    /// anything reaching for `Policy::default()` would get the other.
    #[test]
    fn the_default_policy_is_arc_by_both_routes() {
        assert_eq!(super::Policy::default(), super::Policy::Arc);
        assert_eq!(super::policy_from_str(""), super::Policy::Arc);
        assert_eq!(
            super::policy_from_str(""),
            super::Policy::default(),
            "the parse and the derive must name the same default"
        );
    }

    /// Each rule is selectable, spelling is forgiving, and a value that is not
    /// a rule falls back to the default rather than to something else.
    #[test]
    fn every_policy_is_selectable_and_junk_falls_back() {
        assert_eq!(super::policy_from_str("arc"), super::Policy::Arc);
        assert_eq!(super::policy_from_str("lru"), super::Policy::Lru);
        assert_eq!(super::policy_from_str("fifo"), super::Policy::Fifo);
        // Case and surrounding space are how these arrive from a shell.
        assert_eq!(super::policy_from_str("  LRU \n"), super::Policy::Lru);
        assert_eq!(super::policy_from_str("Fifo"), super::Policy::Fifo);
        // A mistyped sweep arm must not silently become a second copy of some
        // *other* rule — it becomes the default, and says so on stderr.
        assert_eq!(super::policy_from_str("lur"), super::Policy::Arc);
    }

    use super::*;

    fn pool(pages: usize) -> KvPool {
        KvPool::new(
            pages,
            4,
            vec![
                LayerGeometry {
                    kv_dim: 8,
                    stride: 1,
                },
                LayerGeometry {
                    kv_dim: 8,
                    stride: 1,
                },
            ],
        )
    }

    /// **The measurement this module exists for.** Two holders of one page is
    /// the whole mechanism: today's pool hands a cached prefix to one request
    /// and leaves the next to recompute it, because the entry is removed
    /// rather than shared.
    #[test]
    fn two_holders_share_one_page_and_it_survives_the_first_leaving() {
        let p = pool(4);
        let pages = p.alloc(1).unwrap();
        p.retain(&pages);
        assert_eq!(p.refs(pages[0]), 2);

        p.release(&pages);
        assert_eq!(p.refs(pages[0]), 1, "the second holder still has it");
        assert_eq!(p.stats().live_pages, 1);

        p.release(&pages);
        assert_eq!(p.refs(pages[0]), 0);
        assert_eq!(p.stats().live_pages, 0);
    }

    /// A held page must never be handed to somebody else, even under
    /// exhaustion — the failure mode is not a crash but a request reading
    /// another conversation's keys.
    #[test]
    fn a_held_page_is_never_reclaimed_even_when_the_pool_is_empty() {
        let p = pool(2);
        let held = p.alloc(2).unwrap();
        assert_eq!(p.available(), 0);
        assert_eq!(
            p.alloc(1),
            Err(AllocError::Exhausted {
                wanted: 1,
                available: 0
            })
        );
        p.release(&held[..1]);
        let got = p.alloc(1).unwrap();
        assert_eq!(got, vec![held[0]], "the released page, not the held one");
    }

    /// Reclaim order: never-used pages first, then least-recently-released.
    /// Taking a cached page while a never-used one exists throws away a prefix
    /// for no reason.
    #[test]
    fn never_used_pages_are_taken_before_cached_ones() {
        let p = pool(3);
        let a = p.alloc(1).unwrap();
        p.release(&a);
        // One page cached, two never used.
        let next = p.alloc(2).unwrap();
        assert!(
            !next.contains(&a[0]),
            "a cached page was taken while never-used ones remained"
        );
        // Now only the cached one is left.
        let last = p.alloc(1).unwrap();
        assert_eq!(last, a);
    }

    #[test]
    fn cached_pages_are_reclaimed_oldest_release_first() {
        let p = pool(3);
        let all = p.alloc(3).unwrap();
        p.release(&all[0..1]);
        p.release(&all[1..2]);
        p.release(&all[2..3]);
        assert_eq!(p.alloc(1).unwrap(), vec![all[0]]);
        assert_eq!(p.alloc(1).unwrap(), vec![all[1]]);
        assert_eq!(p.alloc(1).unwrap(), vec![all[2]]);
    }

    /// A page released, taken again and released again must be reclaimed on
    /// its *new* position in the order, not its old one — the stale-entry case
    /// the queue is allowed to contain.
    #[test]
    fn a_re_released_page_moves_to_the_back_of_the_reclaim_order() {
        let p = pool(2);
        let all = p.alloc(2).unwrap();
        p.release(&all[0..1]); // page A oldest
        p.release(&all[1..2]); // page B newer

        // Re-take A and release it again: it is now the newest, so B is next.
        let a = p.alloc(1).unwrap();
        assert_eq!(a, vec![all[0]]);
        p.release(&a);

        assert_eq!(
            p.alloc(1).unwrap(),
            vec![all[1]],
            "the stale queue entry for A was trusted over its new release order"
        );
    }

    /// An allocation that cannot be served in full must take nothing — a
    /// partial one leaves the caller unwinding pages it never asked to own.
    #[test]
    fn a_refused_allocation_takes_no_pages() {
        let p = pool(2);
        let _held = p.alloc(1).unwrap();
        assert!(p.alloc(2).is_err());
        assert_eq!(p.stats().live_pages, 1, "the refused call took a page");
        assert_eq!(p.available(), 1);
    }

    #[test]
    fn stats_account_for_every_page_exactly_once() {
        let p = pool(4);
        let held = p.alloc(3).unwrap();
        p.release(&held[0..2]);
        let s = p.stats();
        assert_eq!(s.live_pages, 1);
        assert_eq!(s.cached_pages, 2);
        assert_eq!(s.free_pages, 1);
        assert_eq!(s.live_pages + s.cached_pages + s.free_pages, s.total_pages);
    }

    /// A block-compressed layer stores fewer rows per page than a per-token
    /// one, and its rows must not run into the next page's.
    #[test]
    fn a_strided_layer_gets_its_own_row_count_per_page() {
        let p = KvPool::new(
            2,
            8,
            vec![
                LayerGeometry {
                    kv_dim: 4,
                    stride: 1,
                },
                LayerGeometry {
                    kv_dim: 4,
                    stride: 4,
                },
            ],
        );
        // 8 tokens per page: 8 rows at stride 1, 2 rows at stride 4.
        assert_eq!(p.page_k(0, 0).len(), 8 * 4);
        assert_eq!(p.page_k(1, 0).len(), 2 * 4);
        // Page 1's first row must start exactly where page 0's last one ends.
        assert_eq!(p.row_offset(1, 1, 0), 2 * 4);
        assert_eq!(p.row_offset(0, 1, 0), 8 * 4);
    }

    /// A stride that does not divide the page must round *up*, or the last
    /// row of a page aliases the first row of the next.
    #[test]
    fn a_stride_that_does_not_divide_the_page_rounds_up() {
        let g = LayerGeometry {
            kv_dim: 2,
            stride: 3,
        };
        // 8 tokens at stride 3 is 2 whole rows plus a partial one.
        assert_eq!(g.floats_per_page(8), 3 * 2);
    }

    #[test]
    fn pages_for_rounds_up_to_whole_pages() {
        let p = pool(8);
        assert_eq!(p.pages_for(0), 0);
        assert_eq!(p.pages_for(1), 1);
        assert_eq!(p.pages_for(4), 1);
        assert_eq!(p.pages_for(5), 2);
    }

    fn pool_with(pages: usize, policy: Policy) -> KvPool {
        KvPool::with_policy(
            pages,
            4,
            vec![LayerGeometry {
                kv_dim: 8,
                stride: 1,
            }],
            policy,
        )
    }

    /// Acquire one tag, fill it if it was a miss, publish it, put it back —
    /// the whole lifetime of one page, which is what a request does.
    fn touch(p: &KvPool, tag: u64) -> bool {
        let got = p.acquire(&[tag]).expect("pool has room");
        let hit = got[0].hit;
        if !hit {
            let per = p.layers()[0].floats_per_page(p.page_tokens());
            let rows = vec![tag as f32; per];
            p.fill(0, got[0].page, &rows, &rows);
            p.seal(got[0].page);
        }
        p.release(&[got[0].page]);
        hit
    }

    /// **What sharing means.** Two holders of the same content get the same
    /// page — one copy, two references — instead of one being told to
    /// recompute it.
    #[test]
    fn two_requests_wanting_the_same_content_get_the_same_page() {
        let p = pool_with(4, Policy::Lru);
        let first = p.acquire(&[7]).unwrap();
        // Published before anyone else can find it — an unfilled page must not
        // be shareable, or the second holder reads zeroes.
        let per = p.layers()[0].floats_per_page(p.page_tokens());
        let rows = vec![7.0f32; per];
        p.fill(0, first[0].page, &rows, &rows);
        p.seal(first[0].page);
        let second = p.acquire(&[7]).unwrap();
        assert_eq!(first[0].page, second[0].page, "the content was duplicated");
        assert!(!first[0].hit, "the first acquire cannot be a hit");
        assert!(second[0].hit, "the second must be served from the first");
        assert_eq!(p.refs(first[0].page), 2);
        // And it survives the first leaving, which is the case today's pool
        // cannot express: it removes its entry when it hands it out.
        p.release(&[first[0].page]);
        assert_eq!(p.refs(second[0].page), 1);
        assert!(p.holds(7));
    }

    /// **The scan.** A prefix used more than once, then a burst of one-shot
    /// content that fills the pool, then the prefix again.
    ///
    /// This is the access pattern measured on the running engine — one
    /// unrelated request between two uses of a hot prefix costs the second a
    /// full re-prefill — expressed as a pool trace. Recency alone cannot
    /// survive it: the one-shot pages are by definition the most recently used
    /// things in the pool.
    #[test]
    fn arc_keeps_a_twice_used_prefix_through_a_scan_and_lru_does_not() {
        const HOT: u64 = 1;
        // A pool of four, and a scan of four one-shot tags: exactly enough to
        // turn over the whole pool once.
        let scan: Vec<u64> = (100..104).collect();

        let lru = pool_with(4, Policy::Lru);
        assert!(!touch(&lru, HOT));
        assert!(touch(&lru, HOT), "second use of the prefix must hit");
        for &t in &scan {
            touch(&lru, t);
        }
        let lru_survived = lru.holds(HOT);

        let arc = pool_with(4, Policy::Arc);
        assert!(!touch(&arc, HOT));
        assert!(touch(&arc, HOT), "second use of the prefix must hit");
        for &t in &scan {
            touch(&arc, t);
        }
        let arc_survived = arc.holds(HOT);

        assert!(
            !lru_survived,
            "recency was expected to lose the prefix to the scan; if it now \
             keeps it, this test no longer distinguishes the policies and the \
             claim that ARC is worth its complexity needs re-measuring"
        );
        assert!(
            arc_survived,
            "ARC lost a twice-used prefix to a scan of one-shot content, which \
             is the single property it is being carried for"
        );
        assert!(touch(&arc, HOT), "and it must still be a hit");
    }

    /// The other half of the comparison, and the one that stops this from
    /// being a one-sided test: with no scan pressure, ARC must not be *worse*.
    /// A pure working set that fits keeps hitting under either rule.
    #[test]
    fn both_policies_keep_a_working_set_that_fits() {
        for policy in [Policy::Fifo, Policy::Lru, Policy::Arc] {
            let p = pool_with(4, policy);
            for t in 1..=3u64 {
                touch(&p, t);
            }
            for t in 1..=3u64 {
                assert!(touch(&p, t), "{policy:?} lost a resident working set");
            }
        }
    }

    /// A scan longer than the pool must not evict the frequency half either —
    /// the burst should keep evicting itself.
    #[test]
    fn arc_survives_a_scan_several_times_the_pool_size() {
        const HOT: u64 = 1;
        let arc = pool_with(4, Policy::Arc);
        touch(&arc, HOT);
        touch(&arc, HOT);
        for t in 100..140u64 {
            touch(&arc, t);
        }
        assert!(
            arc.holds(HOT),
            "a long scan reached into the frequency half"
        );
    }

    /// Ghost hits move the target. A tag evicted from the recency half and
    /// then wanted again should grow `p` — otherwise ARC never adapts and is
    /// just a two-list cache.
    #[test]
    fn a_ghost_hit_moves_the_recency_target() {
        let arc = pool_with(2, Policy::Arc);
        let p_before = arc.inner.lock().unwrap().arc.as_ref().unwrap().p;
        // Fill, then push the first tag out, then ask for it again: that second
        // ask is a miss whose tag is sitting in B1.
        touch(&arc, 10);
        touch(&arc, 11);
        touch(&arc, 12);
        assert!(!arc.holds(10), "10 should have been evicted");
        touch(&arc, 10);
        let p_after = arc.inner.lock().unwrap().arc.as_ref().unwrap().p;
        assert!(
            p_after > p_before,
            "a B1 hit did not grow p ({p_before} -> {p_after})"
        );
    }

    /// Acquire is all-or-nothing like `alloc`, hits included — a caller left
    /// holding half a prefix has to unwind it.
    #[test]
    fn a_refused_acquire_takes_nothing() {
        let p = pool_with(2, Policy::Lru);
        let held = p.acquire(&[1, 2]).unwrap();
        assert!(p.acquire(&[3]).is_err());
        assert_eq!(p.stats().live_pages, 2);
        p.release(&held.iter().map(|a| a.page).collect::<Vec<_>>());
    }

    /// The same tag twice in one call is one page, not two.
    #[test]
    fn a_repeated_tag_in_one_call_costs_one_page() {
        let p = pool_with(2, Policy::Lru);
        let got = p.acquire(&[5, 5]).unwrap();
        assert_eq!(got[0].page, got[1].page);
        assert_eq!(p.refs(got[0].page), 2);
        assert_eq!(p.stats().live_pages, 1);
    }

    /// A refilled page must not still be findable under what it used to hold —
    /// the silent-wrong-answer case.
    #[test]
    fn a_reclaimed_page_forgets_its_old_content() {
        let p = pool_with(1, Policy::Lru);
        touch(&p, 42);
        assert!(p.holds(42));
        touch(&p, 43);
        assert!(
            !p.holds(42),
            "the pool still claims to hold evicted content"
        );
        assert!(p.holds(43));
    }

    /// **Steady traffic: recency and ARC tie, and insertion order loses.**
    ///
    /// Every request carries some private conversation; one in `every` also
    /// carries a shared system prompt. Swept across `every`, the two rules
    /// produce identical hit rates at every point — and at `every >= 3` both
    /// produce *zero*.
    ///
    /// Both halves of that are worth understanding before reaching for a
    /// cleverer policy:
    ///
    /// - At `every == 1` the shared pages are touched by every request, so
    ///   plain recency already has them at the front. There is nothing for a
    ///   frequency signal to add.
    /// - At `every >= 3` the pages arriving between two uses of the prefix
    ///   outnumber the pool. Nothing can hold content across a gap wider than
    ///   the cache, so this is a capacity result, not a policy one — and no
    ///   replacement rule is the fix for it.
    ///
    /// ARC's advantage needs an item to be hit *while still resident* to reach
    /// the frequency list at all. Under steady traffic it either never needs
    /// protecting or is gone before it can be promoted. See
    /// [`hit_rate_when_traffic_is_bursty`] for the shape where the difference
    /// is real.
    ///
    /// [`Policy::Fifo`] is the one that separates here, and downward: at 1 in 2
    /// it serves half what recency does, because a page asked for on every
    /// other request still ages out on the schedule it was filled on. That is
    /// the rule the engine's existing prefix pool uses.
    #[test]
    fn hit_rate_under_steady_traffic() {
        const SHARED: u64 = 8;
        const PRIVATE: u64 = 6;
        const REQUESTS: u64 = 200;
        const PAGES: usize = 20;

        let run = |policy: Policy, every: u64| -> (usize, usize) {
            let pool = pool_with(PAGES, policy);
            let (mut hits, mut total) = (0usize, 0usize);
            for r in 0..REQUESTS {
                let mut tags: Vec<u64> = Vec::new();
                if r % every == 0 {
                    tags.extend(1..=SHARED);
                }
                tags.extend((0..PRIVATE).map(|i| 1000 + r * PRIVATE + i));
                let got = pool.acquire(&tags).expect("pool sized for one request");
                total += got.len();
                hits += got.iter().filter(|a| a.hit).count();
                let pages: Vec<u32> = got.iter().map(|a| a.page).collect();
                pool.release(&pages);
            }
            (hits, total)
        };

        eprintln!("steady traffic: shared {SHARED}, private {PRIVATE}, pool {PAGES}");
        eprintln!("  1 in N uses the prefix |   FIFO |    LRU |    ARC");
        for every in [1u64, 2, 3, 4, 6] {
            let (fifo, total) = run(Policy::Fifo, every);
            let (lru, _) = run(Policy::Lru, every);
            let (arc, _) = run(Policy::Arc, every);
            let pct = |h: usize| 100.0 * h as f64 / total as f64;
            eprintln!(
                "  {every:>21} | {:>5.1}% | {:>5.1}% | {:>5.1}%",
                pct(fifo),
                pct(lru),
                pct(arc)
            );
            assert!(arc >= lru, "at 1 in {every}, ARC served fewer than LRU");
            assert!(
                lru >= fifo,
                "at 1 in {every}, insertion order ({fifo}) beat recency ({lru}) — \
                 if that is real it changes which policy the engine should keep, \
                 since insertion order is what it ships today"
            );
        }
    }

    /// **Bursty traffic: this is where the difference lives.**
    ///
    /// The shape a served deployment actually has, and the one measured on the
    /// running engine: a prefix used repeatedly for a while, then a burst of
    /// unrelated work, then the prefix wanted again. A system prompt used all
    /// morning, a batch of one-off requests, then the prompt again.
    ///
    /// Recency loses the prefix to every burst, because the burst is by
    /// definition the most recent thing in the pool. ARC does not: the repeated
    /// use before the burst promotes those pages to the frequency list, the
    /// burst arrives as misses into the recency list, and the recency list
    /// evicts itself.
    ///
    /// Swept across burst length so the result is a curve rather than one
    /// point — a single burst size could be one that happens to favour either
    /// rule.
    #[test]
    fn hit_rate_when_traffic_is_bursty() {
        const SHARED: u64 = 8;
        const PAGES: usize = 20;
        const CYCLES: u64 = 20;
        /// Uses of the prefix before each burst — enough to promote it.
        const WARM: u64 = 3;

        let run = |policy: Policy, burst: u64| -> (usize, usize) {
            let pool = pool_with(PAGES, policy);
            let (mut hits, mut total) = (0usize, 0usize);
            let mut unique = 10_000u64;
            for _ in 0..CYCLES {
                for _ in 0..WARM {
                    let tags: Vec<u64> = (1..=SHARED).collect();
                    let got = pool.acquire(&tags).expect("room");
                    total += got.len();
                    hits += got.iter().filter(|a| a.hit).count();
                    let pages: Vec<u32> = got.iter().map(|a| a.page).collect();
                    pool.release(&pages);
                }
                for _ in 0..burst {
                    let tags: Vec<u64> = (0..4).map(|i| unique + i).collect();
                    unique += 4;
                    let got = pool.acquire(&tags).expect("room");
                    total += got.len();
                    hits += got.iter().filter(|a| a.hit).count();
                    let pages: Vec<u32> = got.iter().map(|a| a.page).collect();
                    pool.release(&pages);
                }
            }
            (hits, total)
        };

        eprintln!(
            "bursty traffic: shared {SHARED} pages, pool {PAGES}, {WARM} warm uses per cycle"
        );
        eprintln!("  burst (4-page requests) |   FIFO |    LRU |    ARC");
        for burst in [1u64, 2, 3, 5, 8] {
            let (fifo, total) = run(Policy::Fifo, burst);
            let (lru, _) = run(Policy::Lru, burst);
            let (arc, _) = run(Policy::Arc, burst);
            let pct = |h: usize| 100.0 * h as f64 / total as f64;
            eprintln!(
                "  {burst:>23} | {:>5.1}% | {:>5.1}% | {:>5.1}%",
                pct(fifo),
                pct(lru),
                pct(arc)
            );
            assert!(
                arc >= lru,
                "at burst {burst}, ARC ({arc}) served fewer than LRU ({lru})"
            );
            assert!(
                lru >= fifo,
                "at burst {burst}, insertion order ({fifo}) beat recency ({lru})"
            );
        }
    }

    /// Geometry comes from the model, not from a second description of it.
    #[test]
    fn geometry_is_read_off_a_probe_cache() {
        use crate::engine::kv_cache::KvCache;
        // Two layers of different width, one of them block-compressed — the
        // variation a summary figure would flatten.
        let probe = KvCache::new_with_strided_dims(1, &[(64, 1), (32, 4)]);
        let geom = LayerGeometry::of(&probe);
        assert_eq!(
            geom,
            vec![
                LayerGeometry {
                    kv_dim: 64,
                    stride: 1
                },
                LayerGeometry {
                    kv_dim: 32,
                    stride: 4
                },
            ]
        );
    }

    /// The budget buys what it buys, and a budget too small for one page
    /// yields no pool rather than an empty one.
    #[test]
    fn sizing_fits_the_budget_and_refuses_an_unusable_one() {
        let layers = vec![LayerGeometry {
            kv_dim: 16,
            stride: 1,
        }];
        // 8 tokens per page x 16 floats x 4 bytes x 2 (k and v) = 1024 bytes.
        assert_eq!(page_bytes(&layers, 8), 1024);

        let pool = KvPool::sized_for(10 * 1024, 8, layers.clone(), Policy::Lru)
            .expect("ten pages' worth is a usable pool");
        assert_eq!(pool.num_pages(), 10);
        assert_eq!(pool.token_capacity(), 80);
        assert!(pool.host_bytes() <= 10 * 1024);

        assert!(
            KvPool::sized_for(1023, 8, layers, Policy::Lru).is_none(),
            "a budget under one page must not produce a pool that can serve nothing"
        );
    }

    /// **The structural difference from today's accounting.** A per-slot budget
    /// divides by the slot count; a pool does not. Same bytes, same model: four
    /// slots each get a quarter of the context, or all four draw on the whole
    /// of it.
    #[test]
    fn a_pool_budget_is_not_divided_by_the_slot_count() {
        let layers = vec![LayerGeometry {
            kv_dim: 16,
            stride: 1,
        }];
        const BUDGET: u64 = 64 * 1024;
        const SLOTS: u64 = 4;

        let pool = KvPool::sized_for(BUDGET, 8, layers.clone(), Policy::Lru).unwrap();
        // What the same bytes buy one slot when every slot must be covered
        // independently, which is what `footprint::kv_tokens_in` computes.
        let per_slot_tokens = KvPool::pages_in(BUDGET / SLOTS, &layers, 8) * 8;

        assert_eq!(pool.token_capacity(), 512);
        assert_eq!(per_slot_tokens, 128);
        assert_eq!(
            pool.token_capacity(),
            per_slot_tokens * SLOTS as usize,
            "the pool should hold exactly what the divided budgets add up to — \
             the difference is that any one sequence may use all of it"
        );
    }

    /// **The invariant the `unsafe impl Sync` rests on.** An unfilled page must
    /// not be findable, or a second request shares zeroes and answers from
    /// them — a wrong answer with no crash and no wrong shape, which is the
    /// failure mode this whole design is arranged to avoid.
    #[test]
    fn an_unsealed_page_is_not_shareable_and_a_sealed_one_is() {
        let p = pool_with(4, Policy::Lru);
        let first = p.acquire(&[9]).unwrap();
        assert!(!p.is_sealed(first[0].page));
        assert!(
            !p.holds(9),
            "an unfilled page was published; another request could read it"
        );

        let per = p.layers()[0].floats_per_page(p.page_tokens());
        let rows = vec![1.5f32; per];
        p.fill(0, first[0].page, &rows, &rows);
        p.seal(first[0].page);

        assert!(p.holds(9));
        let second = p.acquire(&[9]).unwrap();
        assert!(second[0].hit);
        assert_eq!(second[0].page, first[0].page);
        // And the sharer reads what the filler wrote, not zeroes.
        assert_eq!(p.page_k(0, second[0].page)[0], 1.5);
    }

    /// Writing a page somebody else can read is refused rather than risked.
    #[test]
    #[should_panic(expected = "is sealed")]
    fn filling_a_published_page_panics() {
        let p = pool_with(2, Policy::Lru);
        let got = p.acquire(&[1]).unwrap();
        let per = p.layers()[0].floats_per_page(p.page_tokens());
        let rows = vec![0.0f32; per];
        p.fill(0, got[0].page, &rows, &rows);
        p.seal(got[0].page);
        p.fill(0, got[0].page, &rows, &rows);
    }

    /// The other half: a page with two holders is not exclusively owned, so it
    /// is not writable even if it were somehow still unsealed.
    #[test]
    #[should_panic(expected = "not exclusively held")]
    fn filling_a_shared_page_panics() {
        let p = pool_with(2, Policy::Lru);
        let got = p.acquire(&[0]).unwrap();
        p.retain(&[got[0].page]);
        let per = p.layers()[0].floats_per_page(p.page_tokens());
        let rows = vec![0.0f32; per];
        p.fill(0, got[0].page, &rows, &rows);
    }

    /// Reclaiming a page must un-publish it, so its next holder starts from
    /// "unsealed and mine" rather than inheriting the last one's permissions.
    #[test]
    fn a_reclaimed_page_is_writable_again() {
        let p = pool_with(1, Policy::Lru);
        touch(&p, 1);
        assert!(p.is_sealed(0));
        let next = p.acquire(&[2]).unwrap();
        assert!(!next[0].hit);
        assert!(!p.is_sealed(next[0].page), "a reused page stayed sealed");
        let per = p.layers()[0].floats_per_page(p.page_tokens());
        let rows = vec![2.0f32; per];
        p.fill(0, next[0].page, &rows, &rows);
    }

    /// Content written through one layer must not appear in another — the
    /// per-layer regions have to be disjoint.
    #[test]
    fn layers_do_not_alias_each_other() {
        let p = KvPool::with_policy(
            2,
            4,
            vec![
                LayerGeometry {
                    kv_dim: 8,
                    stride: 1,
                },
                LayerGeometry {
                    kv_dim: 8,
                    stride: 1,
                },
            ],
            Policy::Lru,
        );
        let got = p.acquire(&[1]).unwrap();
        let per = p.layers()[0].floats_per_page(p.page_tokens());
        p.fill(0, got[0].page, &vec![1.0; per], &vec![2.0; per]);
        p.fill(1, got[0].page, &vec![3.0; per], &vec![4.0; per]);
        assert!(p.page_k(0, got[0].page).iter().all(|&x| x == 1.0));
        assert!(p.page_v(0, got[0].page).iter().all(|&x| x == 2.0));
        assert!(p.page_k(1, got[0].page).iter().all(|&x| x == 3.0));
        assert!(p.page_v(1, got[0].page).iter().all(|&x| x == 4.0));
    }

    /// Two pages of one layer must not overlap either.
    #[test]
    fn pages_do_not_alias_each_other() {
        let p = pool_with(2, Policy::Lru);
        let a = p.acquire(&[1]).unwrap();
        let b = p.acquire(&[2]).unwrap();
        assert_ne!(a[0].page, b[0].page);
        let per = p.layers()[0].floats_per_page(p.page_tokens());
        p.fill(0, a[0].page, &vec![7.0; per], &vec![7.0; per]);
        p.fill(0, b[0].page, &vec![8.0; per], &vec![8.0; per]);
        assert!(p.page_k(0, a[0].page).iter().all(|&x| x == 7.0));
        assert!(p.page_k(0, b[0].page).iter().all(|&x| x == 8.0));
    }

    /// Byte widths per storage kind — pure arithmetic, so it runs without a
    /// device and pins the one thing a wrong answer here would corrupt: where
    /// every page begins.
    #[test]
    fn device_widths_match_each_storage_format() {
        use crate::engine::backend::vulkan_shaders::KvStorage;
        assert_eq!(device_bytes_for(64, KvStorage::F32), 256);
        assert_eq!(device_bytes_for(64, KvStorage::F16), 128);
        // 9 words per 32 elements: two blocks of 32 is 72 bytes, not 64.
        assert_eq!(device_bytes_for(64, KvStorage::Q8_0), 72);
    }

    /// Pages and the key/value halves must not overlap on the device, for the
    /// same reason they must not on the host — an overlap is a wrong answer
    /// with no wrong shape to catch it.
    #[test]
    fn device_pages_do_not_overlap() {
        use crate::engine::backend::vulkan_shaders::KvStorage;
        let Some(vulkan) = crate::engine::backend::vulkan::shared_test_backend() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let mut pool = KvPool::with_policy(
            8,
            16,
            vec![LayerGeometry {
                kv_dim: 32,
                stride: 1,
            }],
            Policy::Lru,
        );
        let (device, _) = vulkan.device_and_queue();
        assert!(pool.attach_device(device, KvStorage::F16, 256));

        let per_page = 16 * 32; // page_tokens * kv_dim
        let stride = device_bytes_for(per_page, KvStorage::F16);
        let mut seen: Vec<(u64, u64)> = Vec::new();
        for page in 0..8u32 {
            let (k, v, got_stride) = pool.device_page_offsets(0, page).expect("attached");
            assert_eq!(got_stride, stride);
            seen.push((k, k + stride));
            seen.push((v, v + stride));
        }
        seen.sort();
        for pair in seen.windows(2) {
            assert!(
                pair[0].1 <= pair[1].0,
                "device regions overlap: {:?} and {:?}",
                pair[0],
                pair[1]
            );
        }
        // And every region is inside the buffer that was allocated for it.
        let end = seen.last().unwrap().1;
        assert!(end <= pool.device_bytes());
    }

    /// A sequence's block table must fit where it is put. Overflowing it would
    /// write over another sequence's table, which the kernel then reads as page
    /// indices — the quiet cross-conversation failure this whole design keeps
    /// trying to make unreachable.
    #[test]
    #[should_panic(expected = "block table overflow")]
    fn a_table_that_does_not_fit_is_refused() {
        use crate::engine::backend::vulkan_shaders::KvStorage;
        let Some(vulkan) = crate::engine::backend::vulkan::shared_test_backend() else {
            // The assertion cannot be reached without a device, and a test that
            // silently passes for that reason is worse than one that is loud.
            panic!("block table overflow (no GPU adapter; assertion not exercised)");
        };
        let mut pool = KvPool::with_policy(
            4,
            16,
            vec![LayerGeometry {
                kv_dim: 32,
                stride: 1,
            }],
            Policy::Lru,
        );
        let (device, queue) = vulkan.device_and_queue();
        pool.attach_device(device, KvStorage::F16, 4);
        pool.write_table(queue, 2, &[0, 1, 2]);
    }

    /// Attaching is optional, and a host-only pool must stay fully usable —
    /// every CPU-backed path still works with no device in sight.
    #[test]
    fn a_pool_without_a_device_still_works() {
        let p = pool_with(4, Policy::Lru);
        assert!(p.device_pages().is_none());
        assert_eq!(p.device_bytes(), 0);
        assert!(p.device_page_offsets(0, 0).is_none());
        assert!(!touch(&p, 1), "the first touch of a tag cannot be a hit");
        assert!(touch(&p, 1), "the second must be");
    }

    #[test]
    #[should_panic(expected = "has no holder")]
    fn retaining_a_page_nobody_holds_panics() {
        let p = pool(2);
        let pages = p.alloc(1).unwrap();
        p.release(&pages);
        p.retain(&pages);
    }

    #[test]
    #[should_panic(expected = "was not held")]
    fn releasing_a_page_twice_panics() {
        let p = pool(2);
        let pages = p.alloc(1).unwrap();
        p.release(&pages);
        p.release(&pages);
    }
}
