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

//! Prompt lookup's match, from an index of the request's own history.
//!
//! A prompt-lookup draft copies what followed an earlier occurrence of the
//! trailing context. Finding that occurrence by scanning the whole history
//! costs a pass over every token per decode step, and a step asks more than
//! once (whether the reply is echoing, then the draft itself). The index maps
//! each `shortest`-token window to the positions it starts at, grows by the
//! windows the committed tokens complete, and remembers its last answer, so a
//! step visits only the positions whose window equals the trailing one, once.

use rustc_hash::{FxHashMap, FxHasher};
use std::hash::Hasher;

/// The longest trailing context prompt lookup compares.
pub const NGRAM_LONGEST: usize = 64;

/// Window starts by window, for one request's history.
#[derive(Debug, Clone)]
pub struct NgramIndex {
    shortest: usize,
    /// Start positions of every indexed window, ascending, by the window's
    /// hash. A hash names a window only up to a collision, so a match is
    /// compared token for token before it is taken.
    starts: FxHashMap<u64, Vec<u32>>,
    /// How many tokens of the history the index covers.
    indexed: usize,
    /// The last token it covers, to notice a history that was not appended
    /// to but replaced (a check of the tail only: within a request the
    /// history is only appended to).
    last: Option<u32>,
    /// The answer for a history of this length.
    memo: Option<(usize, Option<(usize, usize)>)>,
}

impl NgramIndex {
    /// An empty index matching at least `shortest` tokens of context.
    pub fn new(shortest: usize) -> Self {
        Self {
            shortest: shortest.max(1),
            starts: FxHashMap::default(),
            indexed: 0,
            last: None,
            memo: None,
        }
    }

    fn window_hash(window: &[u32]) -> u64 {
        let mut h = FxHasher::default();
        for &t in window {
            h.write_u32(t);
        }
        h.finish()
    }

    /// Brings the index up to `history`. The history of a request only grows,
    /// so this adds the windows its new tokens complete; a history that is
    /// shorter, or whose last covered token changed, is indexed afresh.
    fn sync(&mut self, history: &[u32]) {
        let replaced = history.len() < self.indexed
            || (self.indexed > 0 && Some(history[self.indexed - 1]) != self.last);
        if replaced {
            self.starts.clear();
            self.indexed = 0;
            self.memo = None;
        }
        let n = self.shortest;
        // Windows starting at `indexed + 1 - n ..` are the ones the new
        // tokens complete.
        let first = (self.indexed + 1).saturating_sub(n);
        for start in first..history.len().saturating_sub(n - 1) {
            self.starts
                .entry(Self::window_hash(&history[start..start + n]))
                .or_default()
                .push(start as u32);
        }
        self.indexed = history.len();
        self.last = history.last().copied();
    }

    /// How many tokens of context the best match covers, and where in
    /// `history` the continuation it copies starts: of the earlier windows
    /// equal to the trailing `shortest` tokens, the one whose context matches
    /// furthest back (at most [`NGRAM_LONGEST`]), the most recent of equal
    /// ones. `None` without one.
    pub fn find(&mut self, history: &[u32]) -> Option<(usize, usize)> {
        self.sync(history);
        if let Some((len, answer)) = self.memo
            && len == history.len()
        {
            return answer;
        }
        let answer = self.search(history);
        self.memo = Some((history.len(), answer));
        answer
    }

    fn search(&self, history: &[u32]) -> Option<(usize, usize)> {
        let shortest = self.shortest;
        if history.len() <= shortest {
            return None;
        }
        let end = history.len();
        let suffix = &history[end - shortest..];
        let starts = self.starts.get(&Self::window_hash(suffix))?;
        // `(context length, start of the continuation)`.
        let mut best: Option<(usize, usize)> = None;
        // Most recent first; the trailing window itself is not an earlier
        // occurrence.
        for &start in starts.iter().rev() {
            let start = start as usize;
            if start >= end - shortest || &history[start..start + shortest] != suffix {
                continue;
            }
            let from = start + shortest;
            let longest = NGRAM_LONGEST.min(from);
            let mut len = shortest;
            while len < longest && history[from - len - 1] == history[end - len - 1] {
                len += 1;
            }
            if best.is_none_or(|(l, _)| len > l) {
                best = Some((len, from));
                if len == longest {
                    break;
                }
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The match by a scan of the whole history, which the index must agree
    /// with.
    fn scan(history: &[u32], shortest: usize) -> Option<(usize, usize)> {
        let shortest = shortest.max(1);
        if history.len() <= shortest {
            return None;
        }
        let end = history.len();
        let suffix = &history[end - shortest..];
        let mut best: Option<(usize, usize)> = None;
        for start in (0..end - shortest).rev() {
            if &history[start..start + shortest] != suffix {
                continue;
            }
            let from = start + shortest;
            let longest = NGRAM_LONGEST.min(from);
            let mut len = shortest;
            while len < longest && history[from - len - 1] == history[end - len - 1] {
                len += 1;
            }
            if best.is_none_or(|(l, _)| len > l) {
                best = Some((len, from));
                if len == longest {
                    break;
                }
            }
        }
        best
    }

    /// A small alphabet repeats windows often, with long runs and copies.
    fn history(seed: u64, len: usize, alphabet: u32) -> Vec<u32> {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut out: Vec<u32> = Vec::with_capacity(len);
        while out.len() < len {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            if out.len() > 8 && x.is_multiple_of(5) {
                // Copy an earlier passage, as an echo does.
                let from = (x >> 8) as usize % (out.len() - 4);
                let n = 4 + (x >> 20) as usize % 40;
                for i in 0..n.min(len - out.len()) {
                    out.push(out[from + i]);
                }
            } else {
                out.push((x >> 32) as u32 % alphabet);
            }
        }
        out
    }

    #[test]
    fn grown_token_by_token_it_answers_as_a_full_scan() {
        for (seed, alphabet, shortest) in [(1, 3, 2), (2, 8, 2), (3, 2, 1), (4, 5, 3), (5, 40, 2)] {
            let h = history(seed, 600, alphabet);
            let mut index = NgramIndex::new(shortest);
            for end in 0..=h.len() {
                assert_eq!(
                    index.find(&h[..end]),
                    scan(&h[..end], shortest),
                    "seed {seed}, alphabet {alphabet}, shortest {shortest}, length {end}"
                );
                // Asked again at the same length: the same answer.
                assert_eq!(index.find(&h[..end]), scan(&h[..end], shortest));
            }
        }
    }

    #[test]
    fn grown_several_tokens_at_a_time_it_answers_as_a_full_scan() {
        let h = history(7, 900, 4);
        let mut index = NgramIndex::new(2);
        let mut end = 0;
        let mut step = 1;
        while end < h.len() {
            end = (end + step).min(h.len());
            step = step % 17 + 1;
            assert_eq!(index.find(&h[..end]), scan(&h[..end], 2), "length {end}");
        }
    }

    #[test]
    fn a_replaced_history_is_indexed_afresh() {
        let a = history(11, 300, 4);
        let b = history(12, 200, 4);
        let mut index = NgramIndex::new(2);
        assert_eq!(index.find(&a), scan(&a, 2));
        // Shorter.
        assert_eq!(index.find(&b), scan(&b, 2));
        // As long, different tail.
        let mut c = b.clone();
        *c.last_mut().unwrap() ^= 1;
        assert_eq!(index.find(&c), scan(&c, 2));
        // Then grown again from there.
        let mut d = c.clone();
        d.extend_from_slice(&a[..80]);
        assert_eq!(index.find(&d), scan(&d, 2));
    }

    #[test]
    fn too_short_or_unmatched_has_no_answer() {
        let mut index = NgramIndex::new(2);
        assert_eq!(index.find(&[]), None);
        assert_eq!(index.find(&[1, 2]), None);
        assert_eq!(index.find(&[1, 2, 3, 4, 5]), None);
    }
}
