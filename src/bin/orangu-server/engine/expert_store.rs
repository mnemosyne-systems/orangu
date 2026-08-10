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

//! Where a routed expert's weights live, separated from which expert is
//! wanted.
//!
//! A model whose experts do not fit in RAM needs a *policy*: which experts to
//! keep close, which to let go, and what to fetch before it is asked for.
//! orangu has no such policy — placement is whatever the OS page cache does
//! with an `mmap`. This module is the seam that policy will attach to, and its
//! first implementation is the incumbent: [`MmapExpertStore`], which changes
//! nothing and *measures* what the page cache is already achieving.
//!
//! That measurement is the point of building the seam before the policy. A
//! residency policy is only worth having if it beats the page cache, and the
//! page cache's own hit rate cannot be compared against a replacement's unless
//! both are measured through the same interface. Built afterwards, the
//! comparison would be retrospective and the baseline a reconstruction.
//!
//! # The lease
//!
//! Adapted from `colibri/c/expert_store.h`, whose contract is four rules
//! enforced by convention:
//!
//! > After a successful `lookup()`, the caller must call `release()` exactly
//! > once. Do not copy the view; the lease is not shareable. `prefetch()` is
//! > advisory, holds no lease, and must not evict a slot that still has an
//! > active lease.
//!
//! In Rust three of those four stop being rules and become types.
//! [`ExpertLease`] releases on `Drop`, so "exactly once" is not a thing a
//! caller can get wrong; it is not `Clone`, so "not shareable" is a compile
//! error; and it borrows the store, so a future evicting implementation cannot
//! be handed a `&mut self` while a lease is outstanding. Only "prefetch is
//! advisory" stays a rule, because it is a statement about behaviour rather
//! than about lifetimes.
//!
//! # Granularity
//!
//! colibri's unit is `(layer, expert)`, because it stores an expert's gate, up
//! and down matrices adjacent so one `pread` fetches all three. GGUF does not:
//! `ffn_gate_exps`, `ffn_up_exps` and `ffn_down_exps` are three separate
//! tensors, so one expert's weights are three byte ranges in three places.
//! The unit here is therefore `(tensor, expert)` — honest to the layout the
//! files actually have. The three ranges of one expert are always touched in
//! the same instant with the same routing weight, so any heat-based policy
//! will keep them together without being told they belong together.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::engine::loader::ExpertQuantMatrix;

/// Prefetch hints issued, process-wide. Separate from the per-store counters
/// because the question they answer — how many bytes a lookahead asked for
/// that nothing then used — is about the *predictor*, not about any one store.
static PREFETCHED_EXPERTS: AtomicU64 = AtomicU64::new(0);
static PREFETCHED_BYTES: AtomicU64 = AtomicU64::new(0);

/// One expert's slice of one per-expert tensor.
///
/// `tensor` is that tensor's stable process-lifetime identity (the address its
/// mapped bytes start at), which is what [`crate::engine::backend`] already
/// uses to key a weight, so two views of the same tensor agree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExpertKey {
    pub tensor: usize,
    pub expert: usize,
}

/// What a store has been asked for and what it did about it.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ExpertStoreStats {
    pub acquisitions: u64,
    /// Acquisitions whose weights were already in RAM.
    pub hits: u64,
    /// Acquisitions that had to come from the disk.
    pub misses: u64,
    /// Expert weight bytes found resident at acquisition time.
    pub resident_bytes: u64,
    /// Expert weight bytes acquired in total, resident or not.
    pub acquired_bytes: u64,
    /// Acquisitions where residency could not be determined — the platform
    /// has no `mincore`, or probing is switched off. Counted rather than
    /// folded into misses: "not measured" and "not resident" are different
    /// claims, and only one of them says a policy would have helped.
    pub unmeasured: u64,
    /// Misses the tier chose to keep. Always 0 for a store with no tier.
    pub admitted: u64,
    /// Misses the tier chose **not** to keep, because the newcomer did not
    /// clear the replacement margin. A high count next to a low hit rate is
    /// the signature of a budget too small for the workload — distinct from
    /// a policy that is simply picking badly, which shows as evictions.
    pub declined: u64,
    pub evicted: u64,
    /// Bytes the tier is holding right now — a level, not a window, so it is
    /// reported rather than drained.
    pub tier_bytes: u64,
    /// Experts a lookahead asked the kernel to read ahead, and their bytes.
    pub prefetched: u64,
    pub prefetched_bytes: u64,
}

impl ExpertStoreStats {
    /// Hits over acquisitions that were actually measured. `None` when none
    /// were — never `0.0`, which would read as "never hit".
    pub fn hit_rate(&self) -> Option<f64> {
        let measured = self.hits + self.misses;
        (measured > 0).then(|| self.hits as f64 / measured as f64)
    }

    pub fn to_json(self) -> serde_json::Value {
        serde_json::json!({
            "acquisitions": self.acquisitions,
            "hits": self.hits,
            "misses": self.misses,
            "unmeasured": self.unmeasured,
            "resident_bytes": self.resident_bytes,
            "acquired_bytes": self.acquired_bytes,
            "admitted": self.admitted,
            "declined": self.declined,
            "evicted": self.evicted,
            "tier_bytes": self.tier_bytes,
            "prefetched": self.prefetched,
            "prefetched_bytes": self.prefetched_bytes,
            "hit_rate": self.hit_rate(),
        })
    }
}

/// A held claim on one expert's weights.
///
/// Exists to bracket the read: while it is alive, the store's placement
/// decision for that expert must not change. [`MmapExpertStore`] has no
/// decision to hold still, so its lease is bookkeeping — but the bracket is
/// what a tiered implementation needs, and putting it in afterwards would mean
/// finding every read again.
///
/// Deliberately not `Clone`: a lease is a claim by one reader, and two
/// readers holding "the same" claim is how a use-after-evict gets written.
pub struct ExpertLease<'a> {
    store: &'a dyn ExpertStore,
    key: ExpertKey,
    /// The tier's own copy of the expert's bytes, when it has one. `None`
    /// means "read them where they already are" — the mapping.
    ///
    /// An `Arc` rather than a borrow because it is what makes colibri's
    /// "never evict a slot with an active lease" rule unnecessary: eviction
    /// drops the *tier's* reference, and a reader holding this one keeps the
    /// bytes alive until it is done. The rule becomes a refcount.
    bytes: Option<Arc<[u8]>>,
}

impl ExpertLease<'_> {
    /// The expert's still-quantized bytes if the store is holding them, else
    /// `None` — the caller then reads from the mapping as it always has.
    pub fn bytes(&self) -> Option<&[u8]> {
        self.bytes.as_deref()
    }
}

impl Drop for ExpertLease<'_> {
    fn drop(&mut self) {
        self.store.release(self.key);
    }
}

pub trait ExpertStore: Send + Sync {
    /// Claims one expert's weights for as long as the lease lives.
    ///
    /// `weights` is passed rather than looked up so the store can see the
    /// bytes it is being asked about without owning a tensor registry.
    fn acquire<'a>(&'a self, weights: &ExpertQuantMatrix, expert: usize) -> ExpertLease<'a>;

    /// Ends a claim. Called by [`ExpertLease`]'s `Drop`, never directly.
    fn release(&self, key: ExpertKey);

    /// Whether this expert's weights are already held, cheaply enough to ask
    /// on the hot path.
    ///
    /// Exists for the expert budget, whose rule is that a *hit costs nothing*
    /// and so should never be dropped — only misses are worth trimming. A
    /// store with no tier has no cheap answer (the page cache would need a
    /// syscall per expert) and says `false`, which makes the budget trim by
    /// gate weight alone. That is a weaker policy, not a wrong one.
    fn is_resident(&self, weights: &ExpertQuantMatrix, expert: usize) -> bool {
        let _ = (weights, expert);
        false
    }

    /// Asks for experts a lookahead expects to want shortly.
    ///
    /// **Advisory, and colibri's rule holds: it takes no lease and must never
    /// evict an expert that has one.** A prediction is a guess, and a guess
    /// that can displace weights a reader is currently using would make the
    /// engine slower the more confidently it was wrong.
    ///
    /// Deliberately does **not** count as use. Heat is a record of what the
    /// router actually chose; letting predictions raise it would let a
    /// prefetcher talk itself into keeping the experts it likes, and the
    /// residency policy would be scoring its own homework.
    ///
    /// The default does nothing, so a store with no prefetch is silent rather
    /// than wrong.
    fn prefetch(&self, weights: &ExpertQuantMatrix, experts: &[usize]) {
        let _ = (weights, experts);
    }

    /// Reads and resets the counters, like every other window in this engine.
    fn take_stats(&self) -> ExpertStoreStats;
}

/// The incumbent: weights are read straight out of the `mmap`, and placement
/// is whatever the kernel decides.
///
/// This is not a placeholder. It is the control arm — the policy a tiered
/// store has to beat — and it reports that policy's own hit rate so the
/// comparison is against a measurement rather than an assumption.
#[derive(Debug, Default)]
pub struct MmapExpertStore {
    acquisitions: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
    unmeasured: AtomicU64,
    resident_bytes: AtomicU64,
    acquired_bytes: AtomicU64,
}

impl ExpertStore for MmapExpertStore {
    fn acquire<'a>(&'a self, weights: &ExpertQuantMatrix, expert: usize) -> ExpertLease<'a> {
        let span = weights.expert_span(expert);
        self.acquisitions.fetch_add(1, Ordering::Relaxed);
        self.acquired_bytes
            .fetch_add(span.len() as u64, Ordering::Relaxed);
        if residency_probe_enabled() {
            match crate::engine::page_cache::resident_bytes_of(span) {
                Some(resident) => {
                    self.resident_bytes.fetch_add(resident, Ordering::Relaxed);
                    // Wholly resident is a hit; anything less means at least
                    // one page has to come off the disk, and a partial read is
                    // a miss for the purpose it is counted for.
                    if resident as usize == span.len() {
                        self.hits.fetch_add(1, Ordering::Relaxed);
                    } else {
                        self.misses.fetch_add(1, Ordering::Relaxed);
                    }
                }
                None => {
                    self.unmeasured.fetch_add(1, Ordering::Relaxed);
                }
            }
        } else {
            self.unmeasured.fetch_add(1, Ordering::Relaxed);
        }
        ExpertLease {
            store: self,
            key: ExpertKey {
                tensor: weights.tensor_id(),
                expert,
            },
            // No tier, so nothing to serve from: the caller reads the mapping.
            bytes: None,
        }
    }

    fn release(&self, _key: ExpertKey) {}

    /// `MADV_WILLNEED` over each predicted expert's byte range: the kernel
    /// starts the readahead and returns, which is the only asynchrony
    /// available without an I/O thread — and is exactly the "cheap version
    /// first" this was scoped as.
    ///
    /// Nothing is copied and nothing is pinned, so a wrong guess costs the
    /// readahead and no memory. On a model already in the page cache it is a
    /// no-op, which is why its payoff cannot be seen on a model that fits.
    fn prefetch(&self, weights: &ExpertQuantMatrix, experts: &[usize]) {
        // A hint warms the page cache; an `O_DIRECT` read path never looks at
        // it. Hinting anyway cost 4.15 GB of readahead per short request when
        // measured — bandwidth spent on the resource a streaming model has
        // least of.
        if !crate::engine::expert_read::source().uses_page_cache() {
            return;
        }
        for &expert in experts {
            let span = weights.expert_span(expert);
            crate::engine::page_cache::advise_willneed(span);
            PREFETCHED_EXPERTS.fetch_add(1, Ordering::Relaxed);
            PREFETCHED_BYTES.fetch_add(span.len() as u64, Ordering::Relaxed);
        }
    }

    fn take_stats(&self) -> ExpertStoreStats {
        ExpertStoreStats {
            acquisitions: self.acquisitions.swap(0, Ordering::Relaxed),
            hits: self.hits.swap(0, Ordering::Relaxed),
            misses: self.misses.swap(0, Ordering::Relaxed),
            unmeasured: self.unmeasured.swap(0, Ordering::Relaxed),
            resident_bytes: self.resident_bytes.swap(0, Ordering::Relaxed),
            acquired_bytes: self.acquired_bytes.swap(0, Ordering::Relaxed),
            // No tier: nothing is admitted, declined, evicted or held.
            admitted: 0,
            declined: 0,
            evicted: 0,
            tier_bytes: 0,
            prefetched: PREFETCHED_EXPERTS.swap(0, Ordering::Relaxed),
            prefetched_bytes: PREFETCHED_BYTES.swap(0, Ordering::Relaxed),
        }
    }
}

/// LFRU score, transliterated from `colibri/c/tier.h`'s `tier_lfru_score`.
///
/// Frequency is the primary signal and recency only breaks close calls: a
/// recent access is worth at most 255 points while one frequency count is
/// worth 256, so an expert that was merely *touched* recently can never
/// displace one that is genuinely hotter. That ordering is the whole design —
/// plain LRU throws out a heavily-used expert because one scan happened to
/// touch something else last.
fn lfru_score(heat: u32, last: u32, clock: u32) -> u64 {
    let age = clock.wrapping_sub(last);
    let recent = 255_u32.saturating_sub(age);
    (u64::from(heat) << 8) | u64::from(recent)
}

/// The margin a newcomer must beat the coldest resident by before it displaces
/// it: **25% + 4 frequency counts**, in score units, exactly colibri's
/// `hs <= cs + (cs>>2) + (4u<<8)`.
///
/// Without it two experts of similar heat evict each other forever, each
/// paying a full copy to take a seat it loses again immediately — the cache
/// does maximum work for zero hit rate. The fixed `+4` is what handles tiny
/// samples, where 25% of a small number is not a margin at all.
fn beats_with_hysteresis(newcomer: u64, coldest: u64) -> bool {
    newcomer > coldest + (coldest >> 2) + (4u64 << 8)
}

/// Which replacement rule the tier follows.
///
/// `Lru` exists to be *compared against*, not because anyone should run it:
/// BIG.md's acceptance for this work is "beats plain LRU at equal budget", and
/// a claim like that needs the loser implemented and selectable, or the
/// comparison is against a description of LRU rather than LRU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Policy {
    /// Frequency first, recency as a tie-break, replacement guarded by a
    /// margin — `colibri/c/tier.h`.
    ///
    /// **Measured worse than [`Policy::Lru`] at every budget tried**, on real
    /// MoE routing, by 12x at a small budget and 1.3x at a large one. The
    /// counters say why: the 25%+4 admission margin declines essentially every
    /// newcomer once the budget is full (42,654 declines against 381
    /// evictions), so the cache freezes around whatever it admitted first and
    /// stops adapting. Kept, not deleted — colibri reports it working on a
    /// different model, expert count and access pattern, and the margin is
    /// doing what it was designed to do; it is simply the wrong trade here.
    /// See BIG.md's M8 for the table.
    Lfru,
    /// Textbook LRU: every miss is admitted, and the least recently used
    /// resident makes room. No notion of how often anything was used.
    ///
    /// The default, on evidence rather than on principle.
    #[default]
    Lru,
}

/// How many acquisitions between halvings of every expert's heat.
///
/// `tier_decay` exists so history ages: without it the experts that were hot
/// during the first prompt of a session keep their seats forever, and the
/// cache stops adapting to what is being asked now. Halving is cheap and
/// keeps the *ordering* of long-lived differences while letting recent
/// activity catch up.
const DECAY_INTERVAL: u64 = 4096;

/// One expert the tier is holding, with the counters that decide whether it
/// keeps its place.
struct Resident {
    bytes: Arc<[u8]>,
    heat: u32,
    last: u32,
}

/// Routing heat that survives a restart, keyed by something a restart cannot
/// invalidate.
///
/// `ExpertKey::tensor` is an address, so the in-memory maps are useless the
/// moment the process exits. The sidecar keys on the tensor's GGUF *name*
/// instead, which also makes the file readable and diffable — a routing
/// history nobody can inspect is one nobody can check for overfitting.
///
/// Tab-separated, one `tensor<TAB>expert<TAB>heat` per line, because it is
/// meant to be read in a terminal like every other append-only record here.
type LearnedHeat = std::collections::HashMap<(Arc<str>, usize), u32>;

#[derive(Default)]
struct TierState {
    /// Names for the tensors seen this run, so heat can be written out under
    /// a key that means something next time.
    names: std::collections::HashMap<usize, Arc<str>>,
    /// What the previous sessions learned, consulted the first time each
    /// expert is touched.
    learned: LearnedHeat,
    resident: std::collections::HashMap<ExpertKey, Resident>,
    /// Heat for experts that are *not* resident, so a newcomer arrives with
    /// the history that justifies admitting it. Without this a cold expert
    /// always scores zero and could never beat anything.
    outside: std::collections::HashMap<ExpertKey, (u32, u32)>,
    bytes: u64,
    clock: u32,
    acquisitions: u64,
}

/// Expert weights held in owned memory under a byte budget, replaced by
/// frequency-first LFRU with hysteresis.
///
/// **Why owned copies rather than pinned pages.** The bytes are already
/// mapped, so the obvious tier is "tell the kernel which pages to keep" —
/// `mlock`. That is not available: `RLIMIT_MEMLOCK` is 8 MB by default (8192
/// KB on this box), which is two experts of a 434 GB model. `madvise` hints
/// are advisory and the kernel is free to ignore them, so a policy built on
/// them cannot be *measured* as a policy. Owning the bytes is what colibri
/// does — it `pread`s experts into its own slabs — and it is the only
/// mechanism here that actually decides anything.
///
/// **Admission, not just eviction.** colibri re-evaluates a fixed set of
/// pinned slots in a periodic repin pass. This is demand-paged, so the same
/// margin guards a different question: not "should these two slots swap" but
/// "should this newcomer displace the coldest resident". A newcomer that does
/// not clear the margin is simply served from the mapping and left out — the
/// cache declines to churn for it.
pub struct TieredExpertStore {
    budget_bytes: u64,
    policy: Policy,
    state: Mutex<TierState>,
    acquisitions: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
    admitted: AtomicU64,
    declined: AtomicU64,
    evicted: AtomicU64,
    resident_bytes_at_acquire: AtomicU64,
    acquired_bytes: AtomicU64,
}

impl TieredExpertStore {
    /// A tier under the default policy. Test-only: production goes through
    /// [`with_policy`](Self::with_policy) so the policy is always an explicit
    /// choice at the one place that reads the environment.
    #[cfg(test)]
    pub fn new(budget_bytes: u64) -> Self {
        Self::with_policy(budget_bytes, Policy::default())
    }

    pub fn with_policy(budget_bytes: u64, policy: Policy) -> Self {
        Self {
            budget_bytes,
            policy,
            state: Mutex::new(TierState {
                learned: load_usage(),
                ..TierState::default()
            }),
            acquisitions: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            admitted: AtomicU64::new(0),
            declined: AtomicU64::new(0),
            evicted: AtomicU64::new(0),
            resident_bytes_at_acquire: AtomicU64::new(0),
            acquired_bytes: AtomicU64::new(0),
        }
    }

    /// Bytes the tier is currently holding.
    pub fn resident_bytes(&self) -> u64 {
        self.state.lock().expect("expert tier poisoned").bytes
    }

    /// This tier's score for an expert, under whichever policy it follows.
    fn score(&self, heat: u32, last: u32, clock: u32) -> u64 {
        match self.policy {
            Policy::Lfru => lfru_score(heat, last, clock),
            // Recency alone, and *not* run through the recency window:
            // textbook LRU orders by when, without saturating, so the oldest
            // resident is always identifiable however long ago it was used.
            Policy::Lru => u64::from(last),
        }
    }

    /// Bumps `key`'s heat and reports whether it is resident, together with
    /// the admission decision if it is not.
    fn touch(&self, key: ExpertKey, name: &Arc<str>, span_len: u64) -> Touch {
        let mut state = self.state.lock().expect("expert tier poisoned");
        state
            .names
            .entry(key.tensor)
            .or_insert_with(|| Arc::clone(name));
        state.clock = state.clock.wrapping_add(1);
        state.acquisitions += 1;
        let decayed = state.acquisitions.is_multiple_of(DECAY_INTERVAL);
        if decayed {
            for resident in state.resident.values_mut() {
                resident.heat >>= 1;
            }
            for (heat, _) in state.outside.values_mut() {
                *heat >>= 1;
            }
        }
        let clock = state.clock;

        if let Some(resident) = state.resident.get_mut(&key) {
            resident.heat = resident.heat.saturating_add(1);
            resident.last = clock;
            let bytes = Arc::clone(&resident.bytes);
            drop(state);
            if decayed {
                self.save_usage();
            }
            return Touch::Hit(bytes);
        }

        // First sight of this expert in this session: start from whatever
        // previous sessions learned about it, so a genuinely hot expert does
        // not have to win its seat back from zero every restart. That is the
        // whole point of a learned store — and also its whole risk, which is
        // why the acceptance is a *held-out* workload.
        let seed = if state.outside.contains_key(&key) {
            0
        } else {
            state
                .names
                .get(&key.tensor)
                .and_then(|name| state.learned.get(&(Arc::clone(name), key.expert)))
                .copied()
                .unwrap_or(0)
        };
        let entry = state.outside.entry(key).or_insert((seed, clock));
        entry.0 = entry.0.saturating_add(1);
        entry.1 = clock;
        let newcomer = self.score(entry.0, entry.1, clock);

        // Room to spare: admit without displacing anyone. Hysteresis is about
        // *replacement*, and there is nothing to replace.
        if state.bytes + span_len <= self.budget_bytes {
            return Touch::Admit;
        }
        // Otherwise the newcomer has to be worth more than what it would push
        // out, by the margin.
        match state
            .resident
            .iter()
            .map(|(k, r)| (self.score(r.heat, r.last, clock), *k))
            .min_by_key(|(score, _)| *score)
        {
            // LRU has no admission decision to make: every miss goes in, and
            // the oldest resident is what pays for it. The margin is LFRU's,
            // and giving LRU one would make it a different policy.
            Some(_) if self.policy == Policy::Lru => Touch::Admit,
            Some((coldest, _)) if beats_with_hysteresis(newcomer, coldest) => Touch::Admit,
            Some(_) => Touch::Decline,
            // Budget too small to hold even one expert.
            None => Touch::Decline,
        }
    }

    /// Inserts an already-copied expert, evicting the coldest residents until
    /// it fits.
    fn admit(&self, key: ExpertKey, bytes: Arc<[u8]>) -> Arc<[u8]> {
        let mut state = self.state.lock().expect("expert tier poisoned");
        let len = bytes.len() as u64;
        while state.bytes + len > self.budget_bytes && !state.resident.is_empty() {
            let clock = state.clock;
            let Some((_, coldest)) = state
                .resident
                .iter()
                .map(|(k, r)| (self.score(r.heat, r.last, clock), *k))
                .min_by_key(|(score, _)| *score)
            else {
                break;
            };
            if let Some(gone) = state.resident.remove(&coldest) {
                state.bytes -= gone.bytes.len() as u64;
                // Its heat survives eviction: an expert that keeps being
                // asked for should not have to build its case from zero
                // every time it loses a seat.
                state.outside.insert(coldest, (gone.heat, gone.last));
                self.evicted.fetch_add(1, Ordering::Relaxed);
            }
        }
        if state.bytes + len <= self.budget_bytes {
            let (heat, last) = state.outside.remove(&key).unwrap_or((1, state.clock));
            state.bytes += len;
            state.resident.insert(
                key,
                Resident {
                    bytes: Arc::clone(&bytes),
                    heat,
                    last,
                },
            );
        }
        bytes
    }
}

impl TieredExpertStore {
    /// Every expert's heat, keyed by tensor name — the shape the sidecar
    /// stores and the shape a later run can use.
    ///
    /// Residents and non-residents both: an expert that earned a seat and one
    /// that keeps being asked for and keeps being declined are both things the
    /// next session wants to know about.
    fn heat_snapshot(&self) -> Vec<(Arc<str>, usize, u32)> {
        let state = self.state.lock().expect("expert tier poisoned");
        let named = |key: &ExpertKey| state.names.get(&key.tensor).map(Arc::clone);
        let resident = state
            .resident
            .iter()
            .filter_map(|(k, r)| named(k).map(|n| (n, k.expert, r.heat)));
        let outside = state
            .outside
            .iter()
            .filter_map(|(k, (heat, _))| named(k).map(|n| (n, k.expert, *heat)));
        resident.chain(outside).filter(|(_, _, h)| *h > 0).collect()
    }

    /// Writes the routing history where the operator asked for it.
    ///
    /// Through a temporary file and a rename, so a crash mid-write leaves the
    /// previous history intact rather than a truncated one: a half-written
    /// history is worse than none, because it looks usable.
    pub fn save_usage(&self) {
        let Some(path) = usage_path() else {
            return;
        };
        let mut rows = self.heat_snapshot();
        if rows.is_empty() {
            return;
        }
        rows.sort_by(|a, b| {
            b.2.cmp(&a.2)
                .then_with(|| a.0.cmp(&b.0))
                .then(a.1.cmp(&b.1))
        });
        let mut out = String::from("# tensor\texpert\theat\n");
        for (name, expert, heat) in rows {
            out.push_str(&format!("{name}\t{expert}\t{heat}\n"));
        }
        let temp = path.with_extension("tmp");
        if std::fs::write(&temp, out).is_ok() {
            let _ = std::fs::rename(&temp, path);
        }
    }
}

/// The routing history `ORANGU_EXPERT_USAGE` names, keyed by `(tensor
/// name, expert)` — empty when there is no history to read.
///
/// Public because a *device* expert tier has to choose its resident set
/// before the first token, from whatever previous sessions learned
/// (`main::plan_expert_tier`). The alternative is filling by size, which
/// colibri measured at 3-5x worse than filling by heat.
pub fn learned_heat() -> std::collections::HashMap<(Arc<str>, usize), u32> {
    load_usage()
}

/// Reads a routing history, or an empty one when there is nothing to read.
///
/// A malformed line is skipped rather than fatal. This is an optimisation
/// hint: a corrupt history must cost a cold cache, never a failed startup.
fn load_usage() -> LearnedHeat {
    let mut learned = LearnedHeat::new();
    let Some(path) = usage_path() else {
        return learned;
    };
    let Ok(text) = std::fs::read_to_string(&path) else {
        return learned;
    };
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let mut fields = line.split('\t');
        let (Some(name), Some(expert), Some(heat)) = (fields.next(), fields.next(), fields.next())
        else {
            continue;
        };
        let (Ok(expert), Ok(heat)) = (expert.parse::<usize>(), heat.parse::<u32>()) else {
            continue;
        };
        learned.insert((Arc::from(name), expert), heat);
    }
    learned
}

/// `ORANGU_EXPERT_USAGE` — where the routing history lives.
///
/// An explicit path rather than a location derived from the model, for two
/// reasons. A model directory is often a read-only Hugging Face blob cache,
/// and writing a sidecar into one is how a cache gets corrupted. And the
/// acceptance for this feature is a *held-out* comparison — history built on
/// one workload, measured on another — which needs two histories side by side,
/// not one magic file per model.
fn usage_path() -> Option<std::path::PathBuf> {
    std::env::var_os("ORANGU_EXPERT_USAGE").map(std::path::PathBuf::from)
}

enum Touch {
    Hit(Arc<[u8]>),
    Admit,
    Decline,
}

impl ExpertStore for TieredExpertStore {
    fn acquire<'a>(&'a self, weights: &ExpertQuantMatrix, expert: usize) -> ExpertLease<'a> {
        let span = weights.expert_span(expert);
        let key = ExpertKey {
            tensor: weights.tensor_id(),
            expert,
        };
        self.acquisitions.fetch_add(1, Ordering::Relaxed);
        self.acquired_bytes
            .fetch_add(span.len() as u64, Ordering::Relaxed);

        let bytes = match self.touch(key, weights.name(), span.len() as u64) {
            Touch::Hit(bytes) => {
                self.hits.fetch_add(1, Ordering::Relaxed);
                self.resident_bytes_at_acquire
                    .fetch_add(bytes.len() as u64, Ordering::Relaxed);
                Some(bytes)
            }
            Touch::Admit => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                self.admitted.fetch_add(1, Ordering::Relaxed);
                // Copied outside the lock: an expert is megabytes, and
                // holding the tier's mutex across the read would serialize
                // every other thread's *hits* behind one thread's miss.
                //
                // Where the copy comes from is `engine::expert_read`'s choice:
                // a `memcpy` from the mapping by default, or a `pread` off the
                // shard — with `O_DIRECT` — when the page cache is the thing
                // being avoided rather than relied on.
                let copy: Arc<[u8]> = Arc::from(crate::engine::expert_read::read_expert(span));
                Some(self.admit(key, copy))
            }
            Touch::Decline => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                self.declined.fetch_add(1, Ordering::Relaxed);
                None
            }
        };
        ExpertLease {
            store: self,
            key,
            bytes,
        }
    }

    fn release(&self, _key: ExpertKey) {}

    /// Asks the kernel to read the predicted experts in ahead of use.
    ///
    /// Still advisory, and still not use: heat is untouched, so a prediction
    /// cannot talk the replacement policy into keeping the experts it likes.
    ///
    /// **Only hints when the read path will actually use the page cache.**
    /// Under `O_DIRECT` it will not, so `MADV_WILLNEED` would warm a cache
    /// nothing reads from — measured at 4.15 GB of readahead per short
    /// request before this was checked. Hinting a bypassed cache is not a
    /// wasted opportunity, it is spent bandwidth on the one resource a
    /// streaming model has least of.
    fn prefetch(&self, weights: &ExpertQuantMatrix, experts: &[usize]) {
        if !crate::engine::expert_read::source().uses_page_cache() {
            return;
        }
        // Skip anything already held: re-reading a resident expert is the one
        // thing a prefetch must never spend bandwidth on.
        for &expert in experts {
            if self.is_resident(weights, expert) {
                continue;
            }
            let span = weights.expert_span(expert);
            crate::engine::page_cache::advise_willneed(span);
            PREFETCHED_EXPERTS.fetch_add(1, Ordering::Relaxed);
            PREFETCHED_BYTES.fetch_add(span.len() as u64, Ordering::Relaxed);
        }
    }

    fn is_resident(&self, weights: &ExpertQuantMatrix, expert: usize) -> bool {
        let key = ExpertKey {
            tensor: weights.tensor_id(),
            expert,
        };
        self.state
            .lock()
            .expect("expert tier poisoned")
            .resident
            .contains_key(&key)
    }

    fn take_stats(&self) -> ExpertStoreStats {
        ExpertStoreStats {
            acquisitions: self.acquisitions.swap(0, Ordering::Relaxed),
            hits: self.hits.swap(0, Ordering::Relaxed),
            misses: self.misses.swap(0, Ordering::Relaxed),
            unmeasured: 0,
            resident_bytes: self.resident_bytes_at_acquire.swap(0, Ordering::Relaxed),
            acquired_bytes: self.acquired_bytes.swap(0, Ordering::Relaxed),
            admitted: self.admitted.swap(0, Ordering::Relaxed),
            declined: self.declined.swap(0, Ordering::Relaxed),
            evicted: self.evicted.swap(0, Ordering::Relaxed),
            tier_bytes: self.resident_bytes(),
            prefetched: PREFETCHED_EXPERTS.swap(0, Ordering::Relaxed),
            prefetched_bytes: PREFETCHED_BYTES.swap(0, Ordering::Relaxed),
        }
    }
}

/// Whether to ask the kernel about residency on every acquisition.
///
/// **Off by default.** The probe is a `mincore` syscall per expert per layer —
/// cheap against a MoE layer's thousands of dot products, but "cheap" is a
/// claim, and this engine's own rule is not to add unmeasured cost to the
/// hottest path. `ORANGU_EXPERT_RESIDENCY=1` turns it on for the runs that
/// want the number; with it off, acquisitions are counted and every one is
/// reported as `unmeasured` rather than as a miss.
fn residency_probe_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("ORANGU_EXPERT_RESIDENCY").is_some_and(|v| v == "1"))
}

/// The store every expert read goes through.
///
/// Process-wide, like `engine::moe_stats`, so a residency policy does not have
/// to be threaded through six architecture modules to exist. When a tiered
/// store arrives it will need per-model configuration (a byte budget), at
/// which point this becomes the place that reads it.
/// The store every expert read goes through, chosen once from the
/// environment.
///
/// **`ORANGU_EXPERT_CACHE_GB` unset or `0` keeps the incumbent**, so nothing
/// changes for anyone who has not asked for a tier. That default is not
/// timidity: on a model that fits in RAM the page cache already holds every
/// expert, so a tier can only duplicate memory and lose. It earns its place
/// exactly where the page cache cannot hold the model, and nowhere else.
pub fn global() -> &'static dyn ExpertStore {
    // Boxed: a `TieredExpertStore` is six times the size of the mmap one, and
    // a process holds exactly one of these for its whole life — the
    // indirection costs a pointer hop that never happens on a hot path.
    enum Chosen {
        Mmap(Box<MmapExpertStore>),
        Tiered(Box<TieredExpertStore>),
    }
    static STORE: OnceLock<Chosen> = OnceLock::new();
    match STORE.get_or_init(|| match tier_budget_bytes() {
        Some(budget) => Chosen::Tiered(Box::new(TieredExpertStore::with_policy(
            budget,
            tier_policy(),
        ))),
        None => Chosen::Mmap(Box::default()),
    }) {
        Chosen::Mmap(store) => store.as_ref(),
        Chosen::Tiered(store) => store.as_ref(),
    }
}

/// `ORANGU_EXPERT_CACHE_GB` as a byte budget, or `None` for no tier.
///
/// Gibibytes because that is the unit the decision is made in — "how much of
/// this box am I giving to expert weights" — and a value that does not parse
/// is treated as no tier rather than as a default size, so a typo cannot
/// quietly hand the cache a budget nobody chose.
/// `ORANGU_EXPERT_CACHE_POLICY=lfru` selects the frequency-first policy.
///
/// The comparison this exists for has now been run, and it went the other way:
/// LRU won at every budget, so LRU is the default and LFRU is the arm. Being
/// one env var apart rather than two builds apart is what made that A/B
/// trustworthy — two arms of one binary cannot be confounded by a stale
/// build, which is a failure this project has paid for before.
fn tier_policy() -> Policy {
    match std::env::var("ORANGU_EXPERT_CACHE_POLICY").as_deref() {
        Ok("lfru") => Policy::Lfru,
        _ => Policy::Lru,
    }
}

fn tier_budget_bytes() -> Option<u64> {
    let raw = std::env::var("ORANGU_EXPERT_CACHE_GB").ok()?;
    let gib: f64 = raw.trim().parse().ok()?;
    (gib > 0.0).then_some((gib * 1024.0 * 1024.0 * 1024.0) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::loader::test_expert_matrix;

    /// `ORANGU_EXPERT_USAGE` is process-wide, so the tests that set it cannot
    /// run alongside each other or alongside the one that asserts it is
    /// *unset* — one would clear the variable out from under another and the
    /// failure would read as a persistence bug rather than as interference.
    static USAGE_ENV: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn usage_env() -> std::sync::MutexGuard<'static, ()> {
        USAGE_ENV.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn a_lease_releases_itself_exactly_once() {
        /// Counts releases, which `MmapExpertStore` has no reason to.
        #[derive(Default)]
        struct Counting {
            released: AtomicU64,
        }
        impl ExpertStore for Counting {
            fn acquire<'a>(&'a self, w: &ExpertQuantMatrix, expert: usize) -> ExpertLease<'a> {
                ExpertLease {
                    store: self,
                    key: ExpertKey {
                        tensor: w.tensor_id(),
                        expert,
                    },
                    bytes: None,
                }
            }
            fn release(&self, _key: ExpertKey) {
                self.released.fetch_add(1, Ordering::Relaxed);
            }
            fn take_stats(&self) -> ExpertStoreStats {
                ExpertStoreStats::default()
            }
        }

        let weights = test_expert_matrix(2, 4, 8);
        let store = Counting::default();
        {
            let _lease = store.acquire(&weights, 1);
            assert_eq!(store.released.load(Ordering::Relaxed), 0, "released early");
        }
        assert_eq!(store.released.load(Ordering::Relaxed), 1);
        {
            let _a = store.acquire(&weights, 0);
            let _b = store.acquire(&weights, 1);
        }
        assert_eq!(store.released.load(Ordering::Relaxed), 3);
    }

    /// The key has to tell two experts of one tensor apart *and* the same
    /// expert index of two different tensors — a policy keyed on the index
    /// alone would confuse `ffn_gate_exps[3]` with `ffn_down_exps[3]`.
    #[test]
    fn keys_separate_experts_and_tensors() {
        let a = test_expert_matrix(4, 4, 8);
        let b = test_expert_matrix(4, 4, 8);
        let key = |m: &ExpertQuantMatrix, e: usize| ExpertKey {
            tensor: m.tensor_id(),
            expert: e,
        };
        assert_ne!(key(&a, 0), key(&a, 1), "two experts of one tensor");
        assert_ne!(key(&a, 0), key(&b, 0), "one expert index, two tensors");
        assert_eq!(key(&a, 2), key(&a, 2), "the same expert is the same key");
    }

    #[test]
    fn acquisitions_and_bytes_are_counted_and_drained() {
        let weights = test_expert_matrix(3, 4, 8);
        let store = MmapExpertStore::default();
        let expert_bytes = weights.expert_bytes();
        drop(store.acquire(&weights, 0));
        drop(store.acquire(&weights, 2));

        let stats = store.take_stats();
        assert_eq!(stats.acquisitions, 2);
        assert_eq!(stats.acquired_bytes, 2 * expert_bytes);
        assert_eq!(
            stats.hits + stats.misses + stats.unmeasured,
            2,
            "every acquisition lands in exactly one bucket"
        );
        assert_eq!(store.take_stats(), ExpertStoreStats::default(), "drained");
    }

    /// "Nothing measured" must not read as "never hit" — the same rule the
    /// rest of this engine's counters follow.
    #[test]
    fn an_unmeasured_window_reports_no_hit_rate() {
        let unmeasured = ExpertStoreStats {
            acquisitions: 9,
            unmeasured: 9,
            ..ExpertStoreStats::default()
        };
        assert_eq!(unmeasured.hit_rate(), None);

        let measured = ExpertStoreStats {
            acquisitions: 4,
            hits: 1,
            misses: 3,
            ..ExpertStoreStats::default()
        };
        assert_eq!(measured.hit_rate(), Some(0.25));
    }

    /// The ordering the whole policy rests on: frequency first, recency only
    /// as a tie-break. One extra access outranks any amount of recency, which
    /// is exactly what plain LRU gets wrong.
    #[test]
    fn frequency_outranks_recency_but_recency_breaks_ties() {
        let clock = 1000;
        // Twice as hot, and untouched for ages, still wins against a
        // once-used expert touched this instant.
        assert!(lfru_score(2, 0, clock) > lfru_score(1, clock, clock));
        // Equal heat: the more recent one wins.
        assert!(lfru_score(5, clock, clock) > lfru_score(5, clock - 10, clock));
        // Recency saturates, so an ancient access is not worth less than an
        // even more ancient one — beyond the window they are simply both cold.
        assert_eq!(lfru_score(5, clock - 300, clock), lfru_score(5, 0, clock));
    }

    /// Without hysteresis two experts of similar heat evict each other
    /// forever, each paying a full copy for a seat it immediately loses.
    #[test]
    fn a_marginally_hotter_newcomer_does_not_displace_the_incumbent() {
        let clock = 100;
        let incumbent = lfru_score(40, clock, clock);
        assert!(
            !beats_with_hysteresis(lfru_score(41, clock, clock), incumbent),
            "one extra access is not a reason to churn"
        );
        assert!(
            !beats_with_hysteresis(lfru_score(49, clock, clock), incumbent),
            "still inside the 25% margin"
        );
        assert!(
            beats_with_hysteresis(lfru_score(60, clock, clock), incumbent),
            "clearly hotter should take the seat"
        );
    }

    /// 25% of a tiny number is not a margin, which is what the fixed `+4`
    /// frequency counts are for.
    #[test]
    fn the_margin_still_bites_when_the_numbers_are_small() {
        let clock = 10;
        let incumbent = lfru_score(1, clock, clock);
        assert!(!beats_with_hysteresis(
            lfru_score(2, clock, clock),
            incumbent
        ));
        assert!(!beats_with_hysteresis(
            lfru_score(4, clock, clock),
            incumbent
        ));
        assert!(beats_with_hysteresis(
            lfru_score(9, clock, clock),
            incumbent
        ));
    }

    /// A budget that fits everything should never evict, and every
    /// acquisition after the first is a hit.
    #[test]
    fn a_budget_that_fits_the_model_holds_all_of_it() {
        let weights = test_expert_matrix(4, 4, 8);
        let store = TieredExpertStore::new(weights.expert_bytes() * 4);
        for _ in 0..3 {
            for expert in 0..4 {
                drop(store.acquire(&weights, expert));
            }
        }
        let stats = store.take_stats();
        assert_eq!(stats.acquisitions, 12);
        assert_eq!(stats.misses, 4, "one miss per expert, the first time");
        assert_eq!(stats.hits, 8);
        assert_eq!(stats.evicted, 0);
        assert_eq!(stats.admitted, 4);
        assert_eq!(stats.tier_bytes, weights.expert_bytes() * 4);
    }

    /// The tier must serve exactly the bytes the mapping would have. Anything
    /// else is a wrong answer that only appears once a cache is enabled.
    #[test]
    fn a_served_expert_is_byte_identical_to_the_mapping() {
        let weights = test_expert_matrix(3, 4, 8);
        let store = TieredExpertStore::new(weights.expert_bytes() * 3);
        for expert in 0..3 {
            drop(store.acquire(&weights, expert));
            let lease = store.acquire(&weights, expert);
            let served = lease.bytes().expect("resident after the first touch");
            assert_eq!(served, weights.expert_span(expert), "expert {expert}");
        }
    }

    /// A budget smaller than one expert cannot hold anything, and must say so
    /// by declining rather than by thrashing.
    #[test]
    fn a_budget_below_one_expert_declines_everything() {
        let weights = test_expert_matrix(2, 4, 8);
        let store = TieredExpertStore::new(weights.expert_bytes() / 2);
        let lease = store.acquire(&weights, 0);
        assert!(lease.bytes().is_none(), "served from the mapping");
        drop(lease);
        let stats = store.take_stats();
        assert_eq!(stats.declined, 1);
        assert_eq!(stats.admitted, 0);
        assert_eq!(stats.tier_bytes, 0);
    }

    /// One expert used over and over, interleaved with single-use visitors —
    /// a scan running past a working set, which is the access pattern MoE
    /// routing actually produces and the one plain LRU handles worst.
    ///
    /// Run under both policies against the same trace and the same budget,
    /// because "LFRU beats LRU" is the claim this task exists to make and it
    /// has to be measured against LRU rather than asserted about it.
    #[test]
    fn lfru_keeps_the_repeatedly_used_expert_where_lru_loses_it() {
        fn run(policy: Policy) -> ExpertStoreStats {
            let weights = test_expert_matrix(64, 4, 8);
            // Room for exactly one expert: the budget where a replacement
            // rule has to actually choose.
            let store = TieredExpertStore::with_policy(weights.expert_bytes(), policy);
            for visitor in 1..40 {
                drop(store.acquire(&weights, 0));
                drop(store.acquire(&weights, visitor));
            }
            store.take_stats()
        }

        let lfru = run(Policy::Lfru);
        let lru = run(Policy::Lru);
        assert!(
            lfru.hits > lru.hits,
            "LFRU {} hits vs LRU {} — the policy is not earning its place",
            lfru.hits,
            lru.hits
        );
        // LRU admits every visitor and pays a copy for each; LFRU declines
        // them, which is where the difference comes from.
        assert!(
            lfru.declined > 0 && lru.declined == 0,
            "lfru declined {}, lru declined {}",
            lfru.declined,
            lru.declined
        );
        assert!(
            lru.evicted > lfru.evicted,
            "LRU should be the one churning: {} vs {}",
            lru.evicted,
            lfru.evicted
        );
    }

    /// The history has to survive a restart, which means keying it on
    /// something a restart cannot invalidate: the tensor *name*, never the
    /// runtime address the in-memory maps use.
    #[test]
    fn learned_heat_round_trips_through_the_sidecar() {
        let _env = usage_env();
        let dir = std::env::temp_dir().join(format!("orangu-usage-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("usage.tsv");
        // SAFETY: single-threaded test setup, before any tier is built.
        unsafe { std::env::set_var("ORANGU_EXPERT_USAGE", &path) };

        let weights =
            crate::engine::loader::test_expert_matrix_named("blk.0.ffn_gate_exps.weight", 8, 4, 8);
        {
            let store = TieredExpertStore::new(weights.expert_bytes() * 2);
            for _ in 0..25 {
                drop(store.acquire(&weights, 3));
            }
            for _ in 0..5 {
                drop(store.acquire(&weights, 6));
            }
            store.save_usage();
        }

        let text = std::fs::read_to_string(&path).expect("history written");
        assert!(
            text.contains("blk.0.ffn_gate_exps.weight\t3\t"),
            "the hot expert is not in the file:\n{text}"
        );

        // A *fresh* tier, as after a restart: the addresses are new, the names
        // are not.
        let reloaded = TieredExpertStore::new(weights.expert_bytes() * 2);
        let seeded = reloaded
            .state
            .lock()
            .unwrap()
            .learned
            .get(&(Arc::from("blk.0.ffn_gate_exps.weight"), 3))
            .copied();
        assert!(
            seeded.is_some_and(|heat| heat >= 25),
            "expert 3's history did not survive: {seeded:?}"
        );

        unsafe { std::env::remove_var("ORANGU_EXPERT_USAGE") };
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A history that cannot be parsed must cost a cold cache, never a failed
    /// startup — it is an optimisation hint, not model data.
    #[test]
    fn a_corrupt_history_is_skipped_line_by_line() {
        let _env = usage_env();
        let dir = std::env::temp_dir().join(format!("orangu-usage-bad-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("usage.tsv");
        std::fs::write(
            &path,
            "# tensor\texpert\theat\ngood.weight\t2\t99\nrubbish\nalso.weight\tNaN\t3\n",
        )
        .unwrap();
        // SAFETY: single-threaded test setup.
        unsafe { std::env::set_var("ORANGU_EXPERT_USAGE", &path) };

        let learned = load_usage();
        assert_eq!(learned.get(&(Arc::from("good.weight"), 2)), Some(&99));
        assert_eq!(learned.len(), 1, "only the parseable line survived");

        unsafe { std::env::remove_var("ORANGU_EXPERT_USAGE") };
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Without a path nothing is read and nothing is written: this is opt-in,
    /// and a feature that quietly starts writing files next to a model is how
    /// a read-only Hugging Face cache gets corrupted.
    #[test]
    fn no_path_means_no_history_at_all() {
        let _env = usage_env();
        // SAFETY: single-threaded test setup.
        unsafe { std::env::remove_var("ORANGU_EXPERT_USAGE") };
        assert!(load_usage().is_empty());
        let weights = crate::engine::loader::test_expert_matrix(4, 4, 8);
        let store = TieredExpertStore::new(weights.expert_bytes());
        drop(store.acquire(&weights, 0));
        store.save_usage(); // must not panic, must not write anywhere
    }

    /// The same thing stated as residency rather than as a rate: after a
    /// parade of one-off visitors, the expert that kept being asked for is
    /// still the one in the tier.
    #[test]
    fn a_hot_expert_survives_a_stream_of_one_off_visitors() {
        let weights = test_expert_matrix(16, 4, 8);
        let store = TieredExpertStore::new(weights.expert_bytes() * 2);
        for _ in 0..40 {
            drop(store.acquire(&weights, 0));
        }
        for visitor in 1..16 {
            drop(store.acquire(&weights, visitor));
        }
        let lease = store.acquire(&weights, 0);
        assert!(
            lease.bytes().is_some(),
            "the hot expert was evicted by traffic that never came back"
        );
    }

    /// Eviction must not lose an expert's history — one that keeps being
    /// asked for should not have to argue its case from zero every time it
    /// loses a seat, or it can never win one back.
    #[test]
    fn heat_survives_eviction() {
        let weights = test_expert_matrix(8, 4, 8);
        let store = TieredExpertStore::new(weights.expert_bytes());
        for _ in 0..20 {
            drop(store.acquire(&weights, 0));
        }
        // A much hotter expert takes the single seat.
        for _ in 0..200 {
            drop(store.acquire(&weights, 1));
        }
        let evicted_state = {
            let state = store.state.lock().unwrap();
            state
                .outside
                .get(&ExpertKey {
                    tensor: weights.tensor_id(),
                    expert: 0,
                })
                .copied()
        };
        let (heat, _) = evicted_state.expect("an evicted expert is still remembered");
        assert!(heat >= 20, "history was thrown away with the bytes: {heat}");
    }

    /// A partially resident expert is a miss: some of its pages still have to
    /// come off the disk, and counting it as a hit would flatter any policy
    /// measured this way.
    #[test]
    fn a_partially_resident_expert_is_not_a_hit() {
        let stats = ExpertStoreStats {
            acquisitions: 1,
            misses: 1,
            resident_bytes: 100,
            acquired_bytes: 200,
            ..ExpertStoreStats::default()
        };
        assert_eq!(stats.hit_rate(), Some(0.0));
        assert!(stats.resident_bytes < stats.acquired_bytes);
    }
}
