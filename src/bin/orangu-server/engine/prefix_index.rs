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

//! What a page of KV *is*, as an identity two requests can agree on without
//! talking to each other.
//!
//! [`crate::engine::kv_pool`] shares a page between holders that ask for the
//! same tag, and never interprets the tag. This is where the tag comes from.
//!
//! # A chained hash, not a tree
//!
//! A page holds `page_tokens` consecutive positions, and what those keys and
//! values *are* is a pure function of every token from the start of the
//! sequence up to the end of that page — attention is causal, so position `n`'s
//! key depends on tokens `0..=n` and on nothing after them.
//!
//! That gives the identity directly: page `i`'s tag is a hash of page `i-1`'s
//! tag together with page `i`'s own token ids. Two sequences that begin with
//! the same tokens produce the same tags for as many whole pages as they agree
//! on, and diverge from the first page where they differ — without either of
//! them consulting a shared structure, and without a tree to walk, split or
//! rebalance.
//!
//! # Why the token ids are stored as well
//!
//! A 64-bit hash collision would mean two different prefixes claiming one
//! page, and the request that lost would answer from another conversation's
//! keys. Not a crash, not a wrong shape — a fluent answer to the wrong
//! context, which is the failure mode this whole design is arranged to make
//! unreachable rather than unlikely.
//!
//! So a tag is a *candidate*, and [`PrefixIndex::resolve`] confirms it against
//! the token ids the page was built from before letting anything share it.
//! That costs one comparison of `page_tokens` ids per page on a hit, against a
//! prefill of the same tokens on a miss.
//!
//! The hash is therefore doing lookup, not identity, and does not have to be
//! cryptographic — only well mixed enough that collisions are rare enough for
//! the verification to be a formality rather than a hot path.

// Nothing calls this yet — `generate::run` still builds contiguous caches and
// consults `prefix_cache`. The scheduler step is what wires it in and removes
// this; until then the module is held to its behaviour by its own tests.
#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::Mutex;

/// The tag of the empty prefix — the seed every chain starts from.
///
/// Deliberately not `0`: `kv_pool` reserves `0` for "unnamed", so a page whose
/// tag happened to hash to it would be silently unshareable, which is the kind
/// of bug that shows up as a hit rate that is merely disappointing.
const SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// Mixes one token into a running prefix hash.
///
/// A 64-bit variant of the usual multiply-xor-shift finalizer. Chosen for
/// mixing rather than for strength: [`PrefixIndex::resolve`] confirms every hit
/// against the stored token ids, so a collision costs a missed reuse, never a
/// wrong answer.
fn mix(state: u64, token: u32) -> u64 {
    let mut h = state ^ (u64::from(token).wrapping_mul(0xFF51_AFD7_ED55_8CCD));
    h ^= h >> 33;
    h = h.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    h ^= h >> 29;
    h
}

/// The tag of each **whole** page of `tokens`, in order.
///
/// A trailing partial page gets no tag: it is still being written, and a page
/// is shareable exactly when it is complete. Handing out an identity for a
/// partial page would let a second request claim content the first has not
/// finished producing.
pub fn page_tags(tokens: &[u32], page_tokens: usize) -> Vec<u64> {
    assert!(page_tokens > 0, "a page must cover at least one token");
    let mut out = Vec::with_capacity(tokens.len() / page_tokens);
    let mut state = SEED;
    for page in tokens.chunks_exact(page_tokens) {
        for &t in page {
            state = mix(state, t);
        }
        out.push(state);
    }
    out
}

/// What a resolved prefix is worth to a request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resolved {
    /// Tags of the leading pages this request may share, longest first match
    /// wins. Shorter than the request's own page count whenever the index does
    /// not have the rest.
    pub shared: Vec<u64>,
    /// Tags for the pages this request must build itself, in order — the ones
    /// after `shared`.
    pub fresh: Vec<u64>,
}

impl Resolved {
    /// Token positions the shared prefix covers.
    pub fn shared_tokens(&self, page_tokens: usize) -> usize {
        self.shared.len() * page_tokens
    }
}

/// Which token run each live tag was built from, so a candidate can be
/// confirmed before it is shared.
pub struct PrefixIndex {
    page_tokens: usize,
    known: Mutex<HashMap<u64, Vec<u32>>>,
}

impl PrefixIndex {
    pub fn new(page_tokens: usize) -> Self {
        Self {
            page_tokens,
            known: Mutex::new(HashMap::new()),
        }
    }

    pub fn page_tokens(&self) -> usize {
        self.page_tokens
    }

    /// Splits `tokens` into the leading pages that are already known and the
    /// rest.
    ///
    /// A prefix is only worth what its *leading* pages are worth: page 3 of a
    /// sequence is meaningless without pages 0 to 2, because its keys were
    /// computed from them. So this stops at the first page the index does not
    /// have, or the first whose stored token run disagrees — a disagreement
    /// being a hash collision, which is answered by declining to share rather
    /// than by trusting the hash.
    ///
    /// `keep_last` leaves the final whole page unshared even when it matches.
    /// The reuse path needs at least one page of real work to produce fresh
    /// logits from, the same reason `generate::run` clamps a full prompt match
    /// by one token.
    pub fn resolve(&self, tokens: &[u32], keep_last: bool) -> Resolved {
        let tags = page_tags(tokens, self.page_tokens);
        let known = self.known.lock().expect("prefix index poisoned");
        let mut shared = Vec::new();
        for (i, &tag) in tags.iter().enumerate() {
            let page = &tokens[i * self.page_tokens..(i + 1) * self.page_tokens];
            match known.get(&tag) {
                Some(stored) if stored == page => shared.push(tag),
                // Absent, or present under a colliding hash. Either way this
                // page and everything after it has to be built.
                _ => break,
            }
        }
        if keep_last && shared.len() == tags.len() && !shared.is_empty() {
            shared.pop();
        }
        let fresh = tags[shared.len()..].to_vec();
        Resolved { shared, fresh }
    }

    /// Records what a page was built from, so a later request can confirm a
    /// candidate against it.
    ///
    /// Called as a sequence seals each page. Storing the token run rather than
    /// only the tag is what makes [`Self::resolve`] exact.
    pub fn remember(&self, tag: u64, page: &[u32]) {
        assert_eq!(
            page.len(),
            self.page_tokens,
            "a remembered page must be a whole page"
        );
        let mut known = self.known.lock().expect("prefix index poisoned");
        known.insert(tag, page.to_vec());
    }

    /// Drops a tag — for a page the pool has reclaimed, whose content is gone.
    ///
    /// Without this the index grows without bound and, worse, starts promising
    /// pages that are no longer resident, turning a cheap miss into a lookup
    /// that fails after the caller has already committed to sharing.
    pub fn forget(&self, tag: u64) {
        self.known
            .lock()
            .expect("prefix index poisoned")
            .remove(&tag);
    }

    /// How many tags are remembered.
    pub fn len(&self) -> usize {
        self.known.lock().expect("prefix index poisoned").len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_shared_token_prefix_produces_shared_page_tags() {
        let a: Vec<u32> = (0..32).collect();
        let mut b = a.clone();
        // Diverge inside the third page.
        b[20] = 999;
        let ta = page_tags(&a, 8);
        let tb = page_tags(&b, 8);
        assert_eq!(ta.len(), 4);
        assert_eq!(ta[0], tb[0], "page 0 is the same tokens");
        assert_eq!(ta[1], tb[1], "page 1 is the same tokens");
        assert_ne!(ta[2], tb[2], "page 2 differs and must not be shared");
        assert_ne!(
            ta[3], tb[3],
            "page 3 has the same tokens but a different history, so it is a \
             different page — this is what chaining is for"
        );
    }

    /// The property the chain exists for: identical tokens in a *different
    /// position* are not the same page.
    #[test]
    fn the_same_tokens_at_a_different_offset_are_a_different_page() {
        let a: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let b: Vec<u32> = vec![9, 9, 9, 9, 1, 2, 3, 4];
        let ta = page_tags(&a, 4);
        let tb = page_tags(&b, 4);
        assert_ne!(
            ta[0], tb[1],
            "a page's identity must include everything before it"
        );
    }

    #[test]
    fn a_partial_trailing_page_gets_no_tag() {
        let tokens: Vec<u32> = (0..10).collect();
        assert_eq!(
            page_tags(&tokens, 4).len(),
            2,
            "8 of 10 tokens are whole pages"
        );
        assert!(page_tags(&tokens[..3], 4).is_empty());
    }

    #[test]
    fn resolve_stops_at_the_first_page_the_index_does_not_have() {
        let index = PrefixIndex::new(4);
        let tokens: Vec<u32> = (0..16).collect();
        let tags = page_tags(&tokens, 4);
        index.remember(tags[0], &tokens[0..4]);
        index.remember(tags[1], &tokens[4..8]);
        // Page 2 deliberately not remembered, page 3 is — it must not be used,
        // because its keys depend on page 2.
        index.remember(tags[3], &tokens[12..16]);

        let r = index.resolve(&tokens, false);
        assert_eq!(r.shared, vec![tags[0], tags[1]]);
        assert_eq!(r.fresh, vec![tags[2], tags[3]]);
        assert_eq!(r.shared_tokens(4), 8);
    }

    /// **The collision case.** A tag that is present but was built from
    /// different tokens must not be shared — this is the difference between a
    /// missed reuse and a wrong answer.
    #[test]
    fn a_colliding_tag_is_declined_rather_than_trusted() {
        let index = PrefixIndex::new(4);
        let tokens: Vec<u32> = (0..8).collect();
        let tags = page_tags(&tokens, 4);
        // Same tag, different content — what a collision looks like from here.
        index.remember(tags[0], &[100, 101, 102, 103]);

        let r = index.resolve(&tokens, false);
        assert!(
            r.shared.is_empty(),
            "a tag was shared without confirming what it was built from"
        );
        assert_eq!(r.fresh.len(), 2);
    }

    #[test]
    fn a_fully_matching_prompt_keeps_its_last_page_when_asked() {
        let index = PrefixIndex::new(4);
        let tokens: Vec<u32> = (0..12).collect();
        let tags = page_tags(&tokens, 4);
        for (i, &t) in tags.iter().enumerate() {
            index.remember(t, &tokens[i * 4..(i + 1) * 4]);
        }
        let all = index.resolve(&tokens, false);
        assert_eq!(all.shared.len(), 3, "every page matches");
        assert!(all.fresh.is_empty());

        let kept = index.resolve(&tokens, true);
        assert_eq!(
            kept.shared.len(),
            2,
            "the last page must be left for the forward pass to redo"
        );
        assert_eq!(kept.fresh, vec![tags[2]]);
    }

    /// A longer prompt that extends a known one shares its whole known part —
    /// the growing-conversation case.
    #[test]
    fn a_continuation_shares_everything_it_extends() {
        let index = PrefixIndex::new(4);
        let first: Vec<u32> = (0..8).collect();
        for (i, &t) in page_tags(&first, 4).iter().enumerate() {
            index.remember(t, &first[i * 4..(i + 1) * 4]);
        }
        let second: Vec<u32> = (0..16).collect();
        let r = index.resolve(&second, false);
        assert_eq!(r.shared.len(), 2, "both pages of the earlier turn");
        assert_eq!(r.fresh.len(), 2);
    }

    #[test]
    fn forgetting_a_reclaimed_page_stops_it_being_promised() {
        let index = PrefixIndex::new(4);
        let tokens: Vec<u32> = (0..8).collect();
        let tags = page_tags(&tokens, 4);
        for (i, &t) in tags.iter().enumerate() {
            index.remember(t, &tokens[i * 4..(i + 1) * 4]);
        }
        assert_eq!(index.resolve(&tokens, false).shared.len(), 2);
        index.forget(tags[0]);
        assert!(
            index.resolve(&tokens, false).shared.is_empty(),
            "page 1 was still promised after page 0 was reclaimed"
        );
    }
}
