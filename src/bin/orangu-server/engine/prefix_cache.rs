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

//! A small, global pool of recently finished requests' own (prompt-plus-
//! generated token ids, KV cache) pairs, reused by a later request whose
//! own prompt tokens share a prefix with one already in the pool —
//! `engine::generate::run` skips the forward pass entirely for however
//! much of the new prompt matches (`KvCache::copy_prefix_from`), instead of
//! always reprefilling from position 0 the way every request does today.
//! Covers both the common growing-conversation case (turn N+1's prompt is
//! turn N's own prompt-plus-response plus a short new suffix — the whole
//! previous turn becomes a free prefix) and two otherwise-unrelated
//! requests that happen to share a long system prompt (a `--cache-reuse`-
//! style win, not just a same-session one) — matching is plain token-id
//! comparison, with no notion of "session" involved at all.

use std::path::Path;
use std::sync::Mutex;

use super::kv_cache::KvCache;

/// One pool entry: a finished request's full token sequence (prompt plus
/// whatever it generated) alongside the KV cache that resulted from
/// processing every one of those positions.
pub struct CachedPrefill {
    pub tokens: Vec<u32>,
    pub cache: KvCache,
}

impl CachedPrefill {
    /// How many leading tokens of `prompt` this entry can serve for free —
    /// the shared token prefix, capped by how much of the cache is actually
    /// committed, and forced to all-or-nothing when the cache carries
    /// recurrent (SSM / gated-delta-net) state. `0` means "no usable reuse."
    ///
    /// This is the single source of truth for prefix matching, shared by
    /// [`PrefixCache::take_best_match`] (the cross-request pool) and
    /// [`crate::engine::slot_store`] (per-slot persistence), so both honor
    /// the exact same committed-length and recurrent-state rules — see
    /// [`PrefixCache::take_best_match`]'s own doc comment for why each cap
    /// matters.
    pub fn reusable_prefix_len(&self, prompt: &[u32]) -> usize {
        let cached_len = self
            .cache
            .layers
            .iter()
            .map(|l| l.len)
            .max()
            .unwrap_or(self.tokens.len());
        let prefix_len = common_prefix_len(&self.tokens, prompt).min(cached_len);
        if prefix_len == 0 {
            return 0;
        }
        // Recurrent state has no per-position history to rewind to a shorter
        // prefix — only a full-length carryover is valid.
        if !self.cache.recurrent.is_empty() && prefix_len != cached_len {
            return 0;
        }
        prefix_len
    }
}

/// Bounded by `max_entries` (a fixed small number — each entry holds a
/// whole `KvCache`'s worth of `f32` K/V buffers, easily hundreds of MB at
/// real context lengths, so this is sized to stay well within ordinary
/// system RAM, not tuned per-deployment). `max_entries == 0` disables the
/// feature entirely at zero runtime cost beyond the `Option` check at each
/// call site.
pub struct PrefixCache {
    entries: Mutex<Vec<CachedPrefill>>,
    max_entries: usize,
}

impl PrefixCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Mutex::new(Vec::new()),
            max_entries,
        }
    }

    /// Removes and returns whichever pool entry shares the longest token
    /// prefix with `tokens`, plus that shared length — `None` if the pool
    /// is empty, disabled (`max_entries == 0`), or every entry's prefix
    /// match is empty. Removed (not just read), not reused in place: two
    /// concurrent requests must never race to extend the same cached
    /// generation, and the caller is expected to [`Self::store`] a fresh
    /// entry back once it's done, whether or not it ends up reusing this
    /// one.
    ///
    /// An entry's own `tokens.len()` can be one *more* than how much of
    /// it is actually reflected in `cache` — `engine::generate::run`'s
    /// decode loop stops as soon as `history.len()` reaches its target
    /// capacity, which happens right after that final token is appended
    /// to the token sequence but *before* the forward call that would
    /// have pushed its own key/value into the cache. So the reusable
    /// bound is always `cache`'s own actually-committed length, never
    /// `tokens.len()` directly — capped here rather than trusted to the
    /// caller. Taken as the *maximum* `len` across every layer, not just
    /// the first one: an architecture with cross-layer KV-donor layers
    /// (`engine::arch::gemma`'s `kv_donor`) gives some layers their own
    /// array slot that's never pushed to at all (writes always redirect
    /// to the donor target's own slot instead), permanently stuck at
    /// `len == 0` regardless of how far the model has actually
    /// progressed — every layer that *does* own its cache shares the
    /// same real `len`, so the maximum across all of them is exactly that
    /// shared value and simply ignores any always-zero donor slots.
    ///
    /// An entry whose `cache` has recurrent (SSM / gated-delta-net) layer
    /// state only matches when the *entire* committed cache is reusable
    /// (`prefix_len == cached_len`) — that state has no per-position
    /// history to rewind to a shorter, older prefix, so a partial match on
    /// such an entry is skipped entirely rather than passed to
    /// [`KvCache::copy_prefix_from`] with a `len` it can't honor correctly
    /// (see that method's own doc comment).
    pub fn take_best_match(&self, tokens: &[u32]) -> Option<(usize, CachedPrefill)> {
        if self.max_entries == 0 {
            return None;
        }
        let mut entries = self.entries.lock().unwrap();
        let mut best: Option<(usize, usize)> = None; // (pool index, prefix len)
        for (i, entry) in entries.iter().enumerate() {
            let prefix_len = entry.reusable_prefix_len(tokens);
            if prefix_len == 0 {
                continue;
            }
            if best.is_none_or(|(_, best_len)| prefix_len > best_len) {
                best = Some((i, prefix_len));
            }
        }
        let (index, prefix_len) = best?;
        Some((prefix_len, entries.remove(index)))
    }

    /// Stores a finished request's own (full token sequence, resulting KV
    /// cache) for a later request to reuse, evicting the oldest entry
    /// first if the pool is already at `max_entries`. A no-op when the
    /// feature is disabled (`max_entries == 0`).
    pub fn store(&self, tokens: Vec<u32>, cache: KvCache) {
        if self.max_entries == 0 {
            return;
        }
        let mut entries = self.entries.lock().unwrap();
        if entries.len() >= self.max_entries {
            entries.remove(0);
        }
        entries.push(CachedPrefill { tokens, cache });
    }

    /// Writes the pool to `dir`, one file per entry, so a restart can reopen
    /// a conversation warm instead of re-prefilling it.
    ///
    /// **Why this is worth bytes on disk.** Re-prefilling a conversation is
    /// not just recomputing attention: on a model whose experts stream, every
    /// replayed position pulls its experts off the disk again, and that is the
    /// dominant cost rather than a rounding error. colibri measures a second
    /// turn reusing 82% of its prompt at **61 s instead of 320 s** on
    /// DeepSeek V4.
    ///
    /// `fingerprint` is the model identity `slot_store` already computes. A
    /// snapshot is refused rather than misread if it does not match: a KV
    /// cache from another model is not corrupt data, it is *plausible* data,
    /// which is worse.
    ///
    /// Each entry is written through a temporary file and renamed, and the
    /// directory is swept of anything that is not one of this pool's files —
    /// so a crash mid-write leaves the previous snapshot rather than a
    /// half-written one that still loads.
    pub fn save_to(&self, dir: &Path, fingerprint: &str) -> std::io::Result<usize> {
        let entries = self.entries.lock().unwrap();
        std::fs::create_dir_all(dir)?;
        let mut written = 0usize;
        let mut keep: Vec<std::ffi::OsString> = Vec::new();
        for (index, entry) in entries.iter().enumerate() {
            // Only the committed prefix is worth storing, and `tokens` can run
            // one ahead of the cache — the same off-by-one
            // `reusable_prefix_len` caps for live matching. Storing the extra
            // token would make a reloaded entry claim a position the cache
            // does not hold.
            let committed = entry.reusable_prefix_len(&entry.tokens);
            if committed == 0 {
                continue;
            }
            let name = std::ffi::OsString::from(format!("prefix-{index}.bin"));
            let path = dir.join(&name);
            let temp = dir.join(format!("prefix-{index}.tmp"));
            let mut blob = Vec::new();
            blob.extend_from_slice(PREFIX_FILE_MAGIC);
            let fp = fingerprint.as_bytes();
            blob.extend_from_slice(&(fp.len() as u32).to_le_bytes());
            blob.extend_from_slice(fp);
            blob.extend_from_slice(&(committed as u32).to_le_bytes());
            for token in &entry.tokens[..committed] {
                blob.extend_from_slice(&token.to_le_bytes());
            }
            blob.extend_from_slice(&entry.cache.to_bytes());
            std::fs::write(&temp, &blob)?;
            std::fs::rename(&temp, &path)?;
            keep.push(name);
            written += 1;
        }
        // Anything left from a larger previous snapshot would otherwise be
        // reloaded forever.
        if let Ok(read) = std::fs::read_dir(dir) {
            for stale in read.flatten() {
                if !keep.contains(&stale.file_name()) {
                    let _ = std::fs::remove_file(stale.path());
                }
            }
        }
        Ok(written)
    }

    /// Reloads a snapshot written by [`Self::save_to`], skipping any file that
    /// is missing, truncated, or was written for a different model.
    ///
    /// Every failure here is silent and costs a cold start, which is the
    /// status quo. A prefix cache is an optimisation; it must never be the
    /// reason a server refuses to come up.
    pub fn load_from(&self, dir: &Path, fingerprint: &str) -> usize {
        let Ok(read) = std::fs::read_dir(dir) else {
            return 0;
        };
        let mut paths: Vec<_> = read.flatten().map(|e| e.path()).collect();
        paths.sort();
        let mut loaded = 0usize;
        for path in paths {
            let Ok(blob) = std::fs::read(&path) else {
                continue;
            };
            let Some(entry) = decode_entry(&blob, fingerprint) else {
                continue;
            };
            self.store(entry.tokens, entry.cache);
            loaded += 1;
        }
        loaded
    }
}

/// Marks a file as this pool's, so a stray file in the directory is skipped
/// rather than parsed as a truncated entry.
const PREFIX_FILE_MAGIC: &[u8] = b"ORGUPFX1";

/// Parses one entry, or `None` for anything that is not exactly one written by
/// this build for this model.
fn decode_entry(blob: &[u8], fingerprint: &str) -> Option<CachedPrefill> {
    let mut at = 0usize;
    let mut take = |n: usize| -> Option<&[u8]> {
        let out = blob.get(at..at + n)?;
        at += n;
        Some(out)
    };
    if take(PREFIX_FILE_MAGIC.len())? != PREFIX_FILE_MAGIC {
        return None;
    }
    let fp_len = u32::from_le_bytes(take(4)?.try_into().ok()?) as usize;
    if take(fp_len)? != fingerprint.as_bytes() {
        // A cache built for another model would match on token ids and answer
        // from the wrong state — the exact silent failure this whole design is
        // shaped to avoid.
        return None;
    }
    let n_tokens = u32::from_le_bytes(take(4)?.try_into().ok()?) as usize;
    let mut tokens = Vec::with_capacity(n_tokens);
    for _ in 0..n_tokens {
        tokens.push(u32::from_le_bytes(take(4)?.try_into().ok()?));
    }
    let cache = KvCache::from_bytes(&blob[at..]).ok()?;
    Some(CachedPrefill { tokens, cache })
}

fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cache already "committed" up through `len` positions (matching
    /// what a real finished request leaves behind) — `take_best_match`
    /// bounds its own matches by exactly this, not by an entry's
    /// `tokens.len()`, so a test cache with `len == 0` (a freshly built,
    /// never-pushed-to one) would make every match trivially empty
    /// regardless of what tokens are compared.
    fn cache(n_layer: usize, capacity: usize, kv_dim: usize, len: usize) -> KvCache {
        let mut c = KvCache::new(n_layer, capacity, kv_dim);
        for layer in &mut c.layers {
            for _ in 0..len {
                layer.push(&vec![0.0; kv_dim], &vec![0.0; kv_dim]);
            }
        }
        c
    }

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "orangu-prefix-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// The point of the whole task: a conversation survives a restart, so
    /// turn two does not re-prefill turn one — which on a streaming model
    /// means not re-reading every expert that turn touched.
    #[test]
    fn a_saved_pool_still_serves_its_prefix_after_a_reload() {
        let dir = temp_dir("roundtrip");
        let tokens = vec![7u32, 8, 9, 10];

        let saved = PrefixCache::new(4);
        saved.store(tokens.clone(), cache(2, 8, 4, 4));
        assert_eq!(saved.save_to(&dir, "fp-abc").unwrap(), 1);

        let reloaded = PrefixCache::new(4);
        assert_eq!(reloaded.load_from(&dir, "fp-abc"), 1);
        let (prefix_len, entry) = reloaded
            .take_best_match(&[7, 8, 9, 11])
            .expect("the reloaded entry matches");
        assert_eq!(prefix_len, 3);
        assert_eq!(entry.tokens[..3], tokens[..3]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// **The failure this format exists to prevent.** A KV cache from another
    /// model would match on token ids and answer from a state that belongs to
    /// a different model — not corrupt data, *plausible* data, which is the
    /// worse kind. The fingerprint must refuse it.
    #[test]
    fn a_snapshot_from_another_model_is_refused_not_misread() {
        let dir = temp_dir("fingerprint");
        let saved = PrefixCache::new(4);
        saved.store(vec![1, 2, 3], cache(2, 8, 4, 3));
        saved.save_to(&dir, "model-A").unwrap();

        let reloaded = PrefixCache::new(4);
        assert_eq!(
            reloaded.load_from(&dir, "model-B"),
            0,
            "another model's cache was loaded"
        );
        assert!(reloaded.take_best_match(&[1, 2, 3]).is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Anything that is not one of these files — a stray file, a truncated
    /// write — costs a cold start and nothing else.
    #[test]
    fn rubbish_in_the_directory_is_skipped_rather_than_parsed() {
        let dir = temp_dir("rubbish");
        std::fs::write(dir.join("not-ours.bin"), b"hello").unwrap();
        std::fs::write(dir.join("prefix-9.bin"), b"ORGUPFX1truncated").unwrap();

        let reloaded = PrefixCache::new(4);
        assert_eq!(reloaded.load_from(&dir, "fp"), 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A shrinking pool must not leave older, larger snapshots behind for the
    /// next start to reload alongside the current ones.
    #[test]
    fn a_smaller_snapshot_sweeps_away_the_larger_one_it_replaces() {
        let dir = temp_dir("sweep");
        let big = PrefixCache::new(4);
        big.store(vec![1, 2, 3], cache(2, 8, 4, 3));
        big.store(vec![4, 5, 6], cache(2, 8, 4, 3));
        assert_eq!(big.save_to(&dir, "fp").unwrap(), 2);

        let small = PrefixCache::new(4);
        small.store(vec![7, 8, 9], cache(2, 8, 4, 3));
        assert_eq!(small.save_to(&dir, "fp").unwrap(), 1);

        let reloaded = PrefixCache::new(4);
        assert_eq!(reloaded.load_from(&dir, "fp"), 1, "a stale file survived");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `tokens` can run one ahead of what the cache actually committed. A
    /// saved entry that claimed the extra token would, on reload, promise a
    /// position it cannot serve.
    #[test]
    fn only_the_committed_prefix_is_written() {
        let dir = temp_dir("committed");
        let saved = PrefixCache::new(4);
        // Four token ids, but only three positions committed.
        saved.store(vec![1, 2, 3, 4], cache(2, 8, 4, 3));
        saved.save_to(&dir, "fp").unwrap();

        let reloaded = PrefixCache::new(4);
        reloaded.load_from(&dir, "fp");
        let (_, entry) = reloaded.take_best_match(&[1, 2, 3, 4]).expect("matches");
        assert_eq!(
            entry.tokens,
            vec![1, 2, 3],
            "an uncommitted token was saved"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn take_best_match_prefers_the_longest_shared_prefix() {
        let pool = PrefixCache::new(4);
        pool.store(vec![1, 2, 3], cache(1, 8, 4, 3));
        pool.store(vec![1, 2, 3, 4, 5], cache(1, 8, 4, 5));
        pool.store(vec![9, 9, 9], cache(1, 8, 4, 3));

        let (prefix_len, entry) = pool.take_best_match(&[1, 2, 3, 4, 9]).unwrap();
        assert_eq!(prefix_len, 4);
        assert_eq!(entry.tokens, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn take_best_match_returns_none_without_any_shared_prefix() {
        let pool = PrefixCache::new(4);
        pool.store(vec![1, 2, 3], cache(1, 8, 4, 3));
        assert!(pool.take_best_match(&[9, 9, 9]).is_none());
    }

    #[test]
    fn take_best_match_removes_the_returned_entry() {
        let pool = PrefixCache::new(4);
        pool.store(vec![1, 2, 3], cache(1, 8, 4, 3));
        assert!(pool.take_best_match(&[1, 2, 3, 4]).is_some());
        assert!(pool.take_best_match(&[1, 2, 3, 4]).is_none());
    }

    #[test]
    fn take_best_match_never_exceeds_the_cache_s_own_committed_length() {
        // `tokens.len() == 3` but only 2 positions actually made it into
        // the cache — `engine::generate::run`'s decode loop can end this
        // way (see `Self::take_best_match`'s own doc comment). A new
        // prompt matching all 3 tokens must still only reuse 2.
        let pool = PrefixCache::new(4);
        pool.store(vec![1, 2, 3], cache(1, 8, 4, 2));
        let (prefix_len, _) = pool.take_best_match(&[1, 2, 3, 4]).unwrap();
        assert_eq!(prefix_len, 2);
    }

    #[test]
    fn a_cross_layer_kv_donor_s_permanently_empty_slot_does_not_block_matching_or_copying() {
        // Mirrors `engine::arch::gemma`'s cross-layer KV-donor layers: one
        // array slot (index 0 here, standing in for a donor layer whose
        // writes always redirect to another layer's own slot instead)
        // stays at `len == 0` forever, while the other slot (index 1, a
        // real owning layer) reflects the model's actual progress. Both
        // `take_best_match` (bounding by the *maximum* len across layers)
        // and `KvCache::copy_prefix_from` (a no-op on a `len == 0` source
        // layer, not a panic) must treat this as a normal 3-token cache,
        // not an empty one.
        let mut donor_cache = KvCache::new_with_dims(8, &[4, 4]);
        for _ in 0..3 {
            donor_cache.layers[1].push(&[0.0; 4], &[0.0; 4]);
        }
        assert_eq!(donor_cache.layers[0].len, 0);
        assert_eq!(donor_cache.layers[1].len, 3);

        let pool = PrefixCache::new(4);
        pool.store(vec![1, 2, 3], donor_cache);
        let (prefix_len, entry) = pool.take_best_match(&[1, 2, 3, 4]).unwrap();
        assert_eq!(prefix_len, 3);

        let mut dst = KvCache::new_with_dims(8, &[4, 4]);
        dst.copy_prefix_from(&entry.cache, prefix_len);
        assert_eq!(dst.layers[0].len, 0, "donor slot must stay untouched");
        assert_eq!(
            dst.layers[1].len, 3,
            "the real owning layer must be fully copied"
        );
    }

    #[test]
    fn store_evicts_the_oldest_entry_once_full() {
        let pool = PrefixCache::new(2);
        pool.store(vec![1], cache(1, 8, 4, 1));
        pool.store(vec![2], cache(1, 8, 4, 1));
        pool.store(vec![3], cache(1, 8, 4, 1));

        assert!(pool.take_best_match(&[1, 9]).is_none());
        assert!(pool.take_best_match(&[2, 9]).is_some());
    }

    #[test]
    fn disabled_pool_never_stores_or_matches() {
        let pool = PrefixCache::new(0);
        pool.store(vec![1, 2, 3], cache(1, 8, 4, 3));
        assert!(pool.take_best_match(&[1, 2, 3]).is_none());
    }

    #[test]
    fn a_mixed_recurrent_cache_only_matches_on_its_full_length() {
        let pool = PrefixCache::new(4);
        let mut mixed = KvCache::new_mixed(8, &[4], &[(2, 3, 1, 2)]);
        for layer in &mut mixed.layers {
            for _ in 0..3 {
                layer.push(&[0.0; 4], &[0.0; 4]);
            }
        }
        pool.store(vec![1, 2, 3], mixed);

        // A strictly longer new prompt (append-only) still matches in full.
        let (prefix_len, entry) = pool.take_best_match(&[1, 2, 3, 4]).unwrap();
        assert_eq!(prefix_len, 3);
        assert_eq!(entry.tokens, vec![1, 2, 3]);
    }

    #[test]
    fn a_mixed_recurrent_cache_is_skipped_on_a_partial_match() {
        let pool = PrefixCache::new(4);
        let mut mixed = KvCache::new_mixed(8, &[4], &[(2, 3, 1, 2)]);
        for layer in &mut mixed.layers {
            for _ in 0..3 {
                layer.push(&[0.0; 4], &[0.0; 4]);
            }
        }
        pool.store(vec![1, 2, 3], mixed);

        // Only the first two of three tokens match — recurrent state can't
        // be rewound to that shorter prefix, so this entry must be skipped
        // rather than returned with prefix_len == 2.
        assert!(pool.take_best_match(&[1, 2, 9]).is_none());
    }
}
