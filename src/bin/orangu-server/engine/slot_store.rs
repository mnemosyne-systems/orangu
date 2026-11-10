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

//! Durable, per-slot KV-cache persistence — the receiving end of the
//! `POST /slots/{id}?action=save|restore` endpoints, and orangu-server's
//! answer to llama.cpp's `--slot-save-path`.
//!
//! Unlike llama.cpp, an orangu-server slot is a *concurrency permit*
//! (`engine::scheduler::SlotPool`), not a long-lived owner of one KV cache.
//! A completed request's cache normally lives on only inside the in-memory
//! `engine::prefix_cache` pool (opt-in, RAM-only, bounded, cross-slot). This
//! module adds the missing durable layer: each slot remembers the
//! `(tokens, KvCache)` of the last request that ran on it, so that
//!
//!   * `save` serializes that snapshot to
//!     `~/.orangu/server/<fingerprint>/slots/<filename>`, and
//!   * `restore` loads it back so the slot's *next* request reuses the
//!     prefix instead of reprefilling — surviving tab switches under cache
//!     pressure, server restarts, and coordinator-driven model swaps, none
//!     of which the RAM-only prefix pool covers.
//!
//! `<fingerprint>` is a hash of the model's identity and KV structure (see
//! [`SlotStore::fingerprint`]): a snapshot saved for one model can never be
//! silently restored into a different one — a mismatch resolves to a
//! different directory *and* is rejected by the in-file fingerprint check.
//!
//! Reuse goes through the exact same `KvCache::copy_prefix_from` path and
//! `CachedPrefill::reusable_prefix_len` matching rules as the prefix pool,
//! so a restored prefix is subject to the same committed-length and
//! recurrent-state guarantees. A slot is exclusive to one in-flight request
//! at a time (the `SlotGuard`), so a slot's retained cache is never raced
//! and can be *borrowed* for reuse rather than removed.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, SystemTime};

use anyhow::{Result, bail};
use sha2::{Digest, Sha256};

use super::kv_cache::KvCache;
use super::prefix_cache::CachedPrefill;

/// How many positions more than a slot's own conversation another slot's
/// must match before it is copied from ([`SlotStore::reuse_into`]): below
/// this the copy — and on a tree, a message to every worker — is not worth
/// what it saves.
pub const SHARE_MIN: usize = 64;

const SLOT_FILE_MAGIC: &[u8] = b"ORGUSLOT";
const SLOT_FILE_VERSION: u32 = 1;

/// Per-slot durable KV-cache persistence. `retained[i]` holds the last
/// completed `(tokens, cache)` for slot `i`, updated at the end of every
/// request that ran on that slot (see [`SlotStore::retain`]) and read both
/// for live per-slot prefix reuse ([`SlotStore::reuse_into`]) and for
/// [`SlotStore::save`].
pub struct SlotStore {
    /// `~/.orangu/server/<fingerprint>/slots/` — created lazily on the first
    /// successful save, never at startup.
    dir: PathBuf,
    /// Model-identity hash embedded in every saved file and checked on
    /// restore. Also the `<fingerprint>` directory component of `dir`.
    fingerprint: String,
    retained: Vec<Mutex<Option<CachedPrefill>>>,
}

/// A retained-cache slot's guard, recovering from a poisoned lock.
///
/// `lock().unwrap()` was wrong here in a way that turned one bad request into
/// a dead server: a panic anywhere under this lock poisons it, and every later
/// request — on *every* slot's code path through this module — then panics on
/// the `unwrap` instead of serving. A caught panic that leaves the process
/// alive but unable to answer anything is worse than the crash it replaced.
///
/// This is a cache. The worst a poisoned entry can be is stale or
/// half-written, and the caller's contract already allows it to be absent, so
/// recovering the guard and letting the entry be overwritten or read as a
/// miss is both sound and the smallest possible blast radius.
fn retained(
    cell: &Mutex<Option<CachedPrefill>>,
) -> std::sync::MutexGuard<'_, Option<CachedPrefill>> {
    cell.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// [`retained`] for a slot that may be busy: `None` rather than waiting.
fn try_retained(
    cell: &Mutex<Option<CachedPrefill>>,
) -> Option<std::sync::MutexGuard<'_, Option<CachedPrefill>>> {
    match cell.try_lock() {
        Ok(guard) => Some(guard),
        Err(std::sync::TryLockError::Poisoned(poisoned)) => Some(poisoned.into_inner()),
        Err(std::sync::TryLockError::WouldBlock) => None,
    }
}

/// How many of `prompt`'s positions `entry` may give: its matching prefix,
/// leaving at least one prompt token for a real forward pass. For a
/// recurrent (SSM / gated-delta-net) architecture, whose state can only
/// carry over whole, an exact full-prompt match cannot both carry the state
/// and leave a fresh token, so it gives nothing — mirroring the
/// all-or-nothing rule `CachedPrefill::reusable_prefix_len` enforces.
fn usable_len(entry: &CachedPrefill, prompt: &[u32]) -> usize {
    let raw = entry.reusable_prefix_len(prompt);
    if raw == 0 {
        0
    } else if entry.cache.recurrent.is_empty() {
        raw.min(prompt.len().saturating_sub(1))
    } else if raw < prompt.len() {
        raw
    } else {
        0
    }
}

/// Whether reusing `len` positions of a conversation a tree's top-level node
/// went on with alone (`KvCache::whole`) saves too little: the rest of
/// the prompt would run here alone, and a tree reads a prompt faster than
/// its top-level node — about 1.45 times on the board measured — so less
/// than half the prompt reused is better sent through the tree whole.
fn too_little_to_go_on_alone(src: &KvCache, dst: &KvCache, len: usize, prompt: &[u32]) -> bool {
    src.whole && dst.remote.is_some() && len * 2 < prompt.len()
}

impl SlotStore {
    /// A stable hash of everything that must match for a saved KV cache to be
    /// safely reusable: the architecture, the model label (`general.name` /
    /// resolved spec id — distinguishes two same-architecture models), and
    /// the KV structure tag (layer count, per-layer `kv_dim`, recurrent
    /// specs). Returned as lowercase hex, suitable as a path component.
    pub fn fingerprint(architecture: &str, model_label: &str, structure_tag: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(architecture.as_bytes());
        hasher.update([0]);
        hasher.update(model_label.as_bytes());
        hasher.update([0]);
        hasher.update(structure_tag);
        let digest = hasher.finalize();
        digest.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Builds a store for `n_slots` slots under
    /// `~/.orangu/server/<fingerprint>/slots/`. `None` when the home
    /// directory can't be resolved — persistence is an optimization only, so
    /// a missing `$HOME` just means "run without it," never "fail to start."
    pub fn new(n_slots: usize, fingerprint: String) -> Option<Self> {
        let dir = home::home_dir()?
            .join(".orangu/server")
            .join(&fingerprint)
            .join("slots");
        Some(Self::at(dir, fingerprint, n_slots))
    }

    /// Builds a store rooted at an explicit `dir` — used by [`Self::new`]
    /// (with the resolved `~/.orangu/server/<fingerprint>/slots` path) and by
    /// tests (with a scratch directory).
    pub(crate) fn at(dir: PathBuf, fingerprint: String, n_slots: usize) -> Self {
        Self {
            dir,
            fingerprint,
            retained: (0..n_slots).map(|_| Mutex::new(None)).collect(),
        }
    }

    /// Records the just-finished `(tokens, cache)` as slot `slot_id`'s
    /// retained snapshot, replacing any previous one. A no-op for an
    /// out-of-range `slot_id` (never happens for a real `SlotGuard`).
    pub fn retain(&self, slot_id: usize, tokens: Vec<u32>, cache: KvCache) {
        if let Some(cell) = self.retained.get(slot_id) {
            *retained(cell) = Some(CachedPrefill { tokens, cache });
        }
    }

    /// [`Self::retain`] for background work (`GenerateRequest::background`):
    /// fills a slot that retains nothing and never replaces a retained entry.
    pub fn fill_if_empty(&self, slot_id: usize, tokens: Vec<u32>, cache: KvCache) {
        if let Some(cell) = self.retained.get(slot_id) {
            let mut held = retained(cell);
            if held.is_none() {
                *held = Some(CachedPrefill { tokens, cache });
            }
        }
    }

    /// Drops every retained snapshot that holds pages of the shared pool and
    /// is not in use; whether any did. Its pages were published when its
    /// request ended, so they stay findable as cached pages and serve the
    /// slot's next turn unless the pool reclaims them first — what a
    /// sequence short of pages asks for before giving its own back.
    pub fn release_paged(&self) -> bool {
        let mut released = false;
        for cell in &self.retained {
            if let Some(mut held) = try_retained(cell)
                && held.as_ref().is_some_and(|entry| entry.cache.is_paged())
            {
                *held = None;
                released = true;
            }
        }
        released
    }

    /// Drops every slot's retained snapshot: a `[workers]` node switching
    /// between its tree and its model alone, whose caches do not
    /// serve each other — one holds its rows on the workers.
    pub fn clear(&self) {
        for cell in &self.retained {
            *retained(cell) = None;
        }
    }

    /// Copies as much of slot `slot_id`'s retained-cache prefix into `dst`
    /// (this request's fresh cache) as safely matches `prompt`, returning the
    /// number of reused positions (`0` if nothing matched or nothing is
    /// retained). Borrows — never removes — the retained entry: the slot is
    /// exclusive to this one in-flight request, so no other request can race
    /// to extend it.
    ///
    /// Always leaves at least one prompt token for a real forward pass. For a
    /// recurrent (SSM / gated-delta-net) architecture, whose state can only
    /// carry over whole (never a rewound prefix), an exact full-prompt match
    /// can't both carry the full state *and* leave a fresh token, so reuse is
    /// skipped there rather than applied partially — mirroring the
    /// all-or-nothing rule `CachedPrefill::reusable_prefix_len` already
    /// enforces.
    ///
    /// On a model spread over workers, the rows of their layers are taken
    /// over rather than copied ([`KvCache::take_remote_prefix`]), and the
    /// snapshot is dropped: what is left of it could not serve again, and
    /// this request's own cache replaces it when it ends.
    ///
    /// **In a tree of workers, another slot's conversation serves when it
    /// matches clearly more** ([`SHARE_MIN`] positions past this slot's own
    /// match) — the same system prompt under a different conversation, most
    /// often. A server alone keeps to its own slot. Its rows
    /// are copied and its snapshot kept, since it belongs to that slot: on a
    /// model spread over workers, they copy their own rows
    /// ([`KvCache::fork_remote_prefix`]), and a tree that cannot is not
    /// shared from. Other slots are only looked at when free to, one at a
    /// time, so two requests sharing from each other's slots never wait on
    /// each other.
    pub fn reuse_into(&self, slot_id: usize, prompt: &[u32], dst: &mut KvCache) -> usize {
        let Some(cell) = self.retained.get(slot_id) else {
            return 0;
        };
        // Only in a tree of workers: a server alone reuses its own slot's
        // conversation and nothing else, as it always has, and goes straight
        // to it below.
        let spread = dst.remote.is_some();
        let own = match spread {
            true => retained(cell)
                .as_ref()
                .map_or(0, |entry| usable_len(entry, prompt)),
            false => 0,
        };
        let mut best: Option<(usize, usize)> = None;
        for (other, cell) in self.retained.iter().enumerate() {
            if other == slot_id || !spread {
                continue;
            }
            let Some(guard) = try_retained(cell) else {
                continue;
            };
            let len = guard.as_ref().map_or(0, |entry| usable_len(entry, prompt));
            if len >= own + SHARE_MIN && best.is_none_or(|(_, longest)| len > longest) {
                best = Some((other, len));
            }
        }
        if let Some((other, _)) = best {
            let shared = self.share_from(other, prompt, dst);
            if shared > 0 {
                log::debug!(
                    "orangu-server: [slot {slot_id}] {shared} prompt positions shared from slot \
                     {other}"
                );
                return shared;
            }
        }
        let mut guard = retained(cell);
        let Some(entry) = guard.as_mut() else {
            return 0;
        };
        let len = usable_len(entry, prompt);
        if len == 0 || too_little_to_go_on_alone(&entry.cache, dst, len, prompt) {
            return 0;
        }
        let remote = dst.remote.is_some();
        if !dst.take_remote_prefix(&mut entry.cache, len) {
            return 0;
        }
        dst.copy_prefix_from(&entry.cache, len);
        if remote {
            *guard = None;
        }
        len
    }

    /// [`Self::reuse_into`] for a paged `dst`, which is filled by pushing
    /// rows (`KvCache::push_prefix_from`): this slot's own retained or
    /// restored conversation, as far as it matches `prompt`. `0` when nothing
    /// matches or `dst` cannot be filled that way.
    pub fn reuse_into_paged(&self, slot_id: usize, prompt: &[u32], dst: &mut KvCache) -> usize {
        if dst.remote.is_some() {
            return 0;
        }
        let Some(cell) = self.retained.get(slot_id) else {
            return 0;
        };
        let guard = retained(cell);
        let Some(entry) = guard.as_ref() else {
            return 0;
        };
        let len = usable_len(entry, prompt).min(entry.cache.host_committed_len());
        if len == 0 || !dst.push_prefix_from(&entry.cache, len) {
            return 0;
        }
        len
    }

    /// Copies slot `other`'s matching prefix into `dst`, keeping `other`'s
    /// snapshot: see [`Self::reuse_into`]. `0` when that slot is busy, no
    /// longer matches, or its workers cannot copy.
    fn share_from(&self, other: usize, prompt: &[u32], dst: &mut KvCache) -> usize {
        let Some(guard) = self.retained.get(other).and_then(try_retained) else {
            return 0;
        };
        let Some(entry) = guard.as_ref() else {
            return 0;
        };
        let len = usable_len(entry, prompt);
        if len == 0
            || too_little_to_go_on_alone(&entry.cache, dst, len, prompt)
            || !dst.fork_remote_prefix(&entry.cache, len)
        {
            return 0;
        }
        dst.copy_prefix_from(&entry.cache, len);
        len
    }

    /// Serializes slot `slot_id`'s retained cache to `<dir>/<filename>` and
    /// returns how many token positions were saved. `0` (and no file written)
    /// when the slot has nothing retained — the client treats that as a
    /// successful no-op, same as llama.cpp saving an empty slot. `Err` only
    /// for an unsafe `filename` or a filesystem failure.
    pub fn save(&self, slot_id: usize, filename: &str) -> Result<usize> {
        let path = self.resolve(filename)?;
        let Some(cell) = self.retained.get(slot_id) else {
            bail!("slot {slot_id} out of range");
        };
        let guard = retained(cell);
        let Some(entry) = guard.as_ref() else {
            return Ok(0);
        };
        // A file holds the rows this process has. A model spread over
        // workers keeps most of them there, and a file without them would
        // restore into a server running the whole model as a cache with
        // layers missing.
        if entry.cache.remote.is_some() {
            bail!("slot {slot_id} is spread over [workers]; its rows cannot be saved");
        }
        let bytes = encode_file(&self.fingerprint, entry);
        // What the file actually carries, which is not `committed_len`: the
        // fused GPU decode path leaves the cache's `len` ahead of its host
        // buffers, and `KvCache::to_bytes` writes only the host-resident
        // rows. Reporting the larger number told the client it had persisted
        // a generated tail that is not in the file.
        let saved = entry.cache.host_committed_len();
        drop(guard);
        write_atomic(&path, &bytes)?;
        Ok(saved)
    }

    /// Loads `<dir>/<filename>` into slot `slot_id`'s retained cache so the
    /// slot's next request reuses it, returning the restored token count.
    /// A missing, corrupt, or fingerprint-mismatched file resolves to `0`
    /// (nothing restored, normal prefill next request) rather than an error,
    /// so a stale or incompatible sidecar never poisons the endpoint for the
    /// client. `Err` only for an unsafe `filename`.
    pub fn restore(&self, slot_id: usize, filename: &str) -> Result<usize> {
        let path = self.resolve(filename)?;
        if self.retained.get(slot_id).is_none() {
            bail!("slot {slot_id} out of range");
        }
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(0),
            Err(err) => bail!("reading {}: {err}", path.display()),
        };
        let (tokens, cache) = match decode_file(&self.fingerprint, &bytes) {
            Ok(v) => v,
            Err(err) => {
                // Incompatible/corrupt on-disk state is not a client error —
                // fall back to a full prefill instead of failing the restore.
                log::warn!(
                    "orangu-server: ignoring unusable slot file {}: {err}",
                    path.display()
                );
                return Ok(0);
            }
        };
        let restored = cache.committed_len();
        // `get` re-checked above, so this index is in range.
        *retained(&self.retained[slot_id]) = Some(CachedPrefill { tokens, cache });
        Ok(restored)
    }

    /// [`Self::restore`] for every slot from `<prefix>-<slot>.bin`, as the
    /// coordinator names the slots it saves before a swap: how many token
    /// positions each slot got back. A slot with no file, or an unusable one,
    /// gets nothing and the next request prefills as it would have.
    pub fn restore_all(&self, prefix: &str) -> Vec<usize> {
        (0..self.retained.len())
            .map(|slot| {
                self.restore(slot, &format!("{prefix}-{slot}.bin"))
                    .unwrap_or_else(|err| {
                        log::warn!("orangu-server: slot {slot} not restored: {err}");
                        0
                    })
            })
            .collect()
    }

    /// Writes the paged pool's published pages to `<dir>/<filename>`: every
    /// conversation the pool still holds, not only the slots' last ones —
    /// what a swap would otherwise throw away with the process. In use first,
    /// then newest first, until `cap_bytes` of rows. Returns the pages
    /// written. A pool whose host pages are compact is not saved.
    pub fn save_pool(
        &self,
        pool: &super::kv_pool::KvPool,
        index: &super::prefix_index::PrefixIndex,
        filename: &str,
        cap_bytes: u64,
    ) -> Result<usize> {
        let path = self.resolve(filename)?;
        if pool.is_compact() {
            return Ok(0);
        }
        let layers = pool.layers().len();
        let mut out = Vec::new();
        out.extend_from_slice(POOL_FILE_MAGIC);
        out.extend_from_slice(&POOL_FILE_VERSION.to_le_bytes());
        out.extend_from_slice(&(self.fingerprint.len() as u32).to_le_bytes());
        out.extend_from_slice(self.fingerprint.as_bytes());
        out.extend_from_slice(&(pool.page_tokens() as u32).to_le_bytes());
        out.extend_from_slice(&(layers as u32).to_le_bytes());
        let count_at = out.len();
        out.extend_from_slice(&0u32.to_le_bytes());
        let mut written = 0u32;
        for (page, tag) in pool.published_recent_first() {
            let Some(tokens) = index.tokens_of(tag) else {
                continue;
            };
            if out.len() as u64 >= cap_bytes {
                break;
            }
            out.extend_from_slice(&tag.to_le_bytes());
            for t in &tokens {
                out.extend_from_slice(&t.to_le_bytes());
            }
            for layer in 0..layers {
                for rows in [pool.page_k(layer, page), pool.page_v(layer, page)] {
                    out.extend_from_slice(&(rows.len() as u32).to_le_bytes());
                    out.extend_from_slice(bytemuck::cast_slice(rows));
                }
            }
            written += 1;
        }
        out[count_at..count_at + 4].copy_from_slice(&written.to_le_bytes());
        write_atomic(&path, &out)?;
        Ok(written as usize)
    }

    /// Reads pages [`Self::save_pool`] wrote back into the pool and the
    /// index, each published as its own request would have left it, so the
    /// conversations they held are found again by their tokens. A missing,
    /// foreign or damaged file restores nothing; a page the pool already holds
    /// or has no room for is skipped. Returns the pages restored.
    pub fn restore_pool(
        &self,
        pool: &super::kv_pool::KvPool,
        index: &super::prefix_index::PrefixIndex,
        filename: &str,
    ) -> Result<usize> {
        let path = self.resolve(filename)?;
        let Ok(bytes) = std::fs::read(&path) else {
            return Ok(0);
        };
        Ok(decode_pool(&bytes, &self.fingerprint, pool, index).unwrap_or(0))
    }

    /// Resolves `filename` to a path inside `dir`, rejecting anything that is
    /// not a single safe filename component — the only defense between a
    /// client-supplied name and `fs::read`/`fs::write`, so path traversal
    /// (`..`, absolute paths, separators) must be caught here.
    fn resolve(&self, filename: &str) -> Result<PathBuf> {
        if !is_safe_filename(filename) {
            bail!("unsafe slot filename {filename:?}");
        }
        Ok(self.dir.join(filename))
    }
}

const POOL_FILE_MAGIC: &[u8] = b"ORGUPOOL";
const POOL_FILE_VERSION: u32 = 1;

/// [`SlotStore::restore_pool`]'s reader: `None` for a file that is not this
/// model's or does not parse to its end.
fn decode_pool(
    bytes: &[u8],
    fingerprint: &str,
    pool: &super::kv_pool::KvPool,
    index: &super::prefix_index::PrefixIndex,
) -> Option<usize> {
    let mut at = 0usize;
    let mut take = |n: usize| -> Option<&[u8]> {
        let slice = bytes.get(at..at + n)?;
        at += n;
        Some(slice)
    };
    let u32_of = |b: &[u8]| u32::from_le_bytes(b.try_into().unwrap());
    if take(POOL_FILE_MAGIC.len())? != POOL_FILE_MAGIC || u32_of(take(4)?) != POOL_FILE_VERSION {
        return None;
    }
    let fp_len = u32_of(take(4)?) as usize;
    if take(fp_len)? != fingerprint.as_bytes() {
        return None;
    }
    let page_tokens = u32_of(take(4)?) as usize;
    let layers = u32_of(take(4)?) as usize;
    if page_tokens != pool.page_tokens() || layers != pool.layers().len() || pool.is_compact() {
        return None;
    }
    let count = u32_of(take(4)?) as usize;
    let mut restored = 0;
    for _ in 0..count {
        let tag = u64::from_le_bytes(take(8)?.try_into().unwrap());
        let tokens: Vec<u32> = take(page_tokens * 4)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect();
        let mut rows: Vec<(Vec<f32>, Vec<f32>)> = Vec::with_capacity(layers);
        for _ in 0..layers {
            let mut side = || -> Option<Vec<f32>> {
                let n = u32_of(take(4)?) as usize;
                Some(
                    take(n * 4)?
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|b| f32::from_le_bytes(*b))
                        .collect(),
                )
            };
            let k = side()?;
            let v = side()?;
            rows.push((k, v));
        }
        if pool.holds(tag) {
            continue;
        }
        let Ok(acquired) = pool.acquire(&[tag]) else {
            break;
        };
        let page = acquired[0].page;
        if !acquired[0].hit {
            for (layer, (k, v)) in rows.iter().enumerate() {
                if !k.is_empty() {
                    pool.fill(layer, page, k, v);
                }
            }
            pool.seal(page);
            pool.refill_device(page);
            index.remember(tag, &tokens);
            restored += 1;
        }
        pool.release(&[page]);
    }
    Some(restored)
}

/// A safe filename is a non-empty single component of `[A-Za-z0-9._-]` that
/// is neither `.` nor `..` — matching what the client actually sends
/// (`<uuid>.bin`) while forbidding separators, parent refs, and absolute
/// paths.
fn is_safe_filename(name: &str) -> bool {
    if name.is_empty() || name == "." || name == ".." {
        return false;
    }
    name.chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

/// `[MAGIC][version u32][fp_len u32][fp][n_tokens u32][tokens…][KvCache bytes]`,
/// all little-endian. The cache bytes are `KvCache::to_bytes`'s own
/// self-describing format and run to the end of the file.
fn encode_file(fingerprint: &str, entry: &CachedPrefill) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(SLOT_FILE_MAGIC);
    out.extend_from_slice(&SLOT_FILE_VERSION.to_le_bytes());
    out.extend_from_slice(&(fingerprint.len() as u32).to_le_bytes());
    out.extend_from_slice(fingerprint.as_bytes());
    out.extend_from_slice(&(entry.tokens.len() as u32).to_le_bytes());
    for &t in &entry.tokens {
        out.extend_from_slice(&t.to_le_bytes());
    }
    out.extend_from_slice(&entry.cache.to_bytes());
    out
}

fn decode_file(expected_fingerprint: &str, bytes: &[u8]) -> Result<(Vec<u32>, KvCache)> {
    let mut pos = 0usize;
    let mut take = |n: usize| -> Result<&[u8]> {
        let end = pos
            .checked_add(n)
            .filter(|&e| e <= bytes.len())
            .ok_or_else(|| anyhow::anyhow!("unexpected end of slot file"))?;
        let slice = &bytes[pos..end];
        pos = end;
        Ok(slice)
    };
    if take(SLOT_FILE_MAGIC.len())? != SLOT_FILE_MAGIC {
        bail!("not an orangu slot file (bad magic)");
    }
    let version = read_u32(take(4)?);
    if version != SLOT_FILE_VERSION {
        bail!("unsupported slot file version {version}");
    }
    let fp_len = read_u32(take(4)?) as usize;
    let fp = std::str::from_utf8(take(fp_len)?).map_err(|_| anyhow::anyhow!("bad fingerprint"))?;
    if fp != expected_fingerprint {
        bail!("slot file was saved for a different model");
    }
    let n_tokens = read_u32(take(4)?) as usize;
    let token_bytes = take(
        n_tokens
            .checked_mul(4)
            .ok_or_else(|| anyhow::anyhow!("token count overflows"))?,
    )?;
    let tokens: Vec<u32> = token_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| read_u32(word))
        .collect();
    let cache = KvCache::from_bytes(&bytes[pos..])?;
    Ok((tokens, cache))
}

fn read_u32(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

/// Writes `data` to `path` atomically (temp file, then rename over the real
/// path), creating the parent directory on demand — the same crash-safe
/// pattern `engine::backend::vulkan`'s pipeline cache uses, so a save
/// interrupted midway can never leave a truncated slot file behind.
fn write_atomic(path: &Path, data: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, data)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// What a [`sweep_stale_slot_files`] pass removed.
pub struct SlotSweep {
    pub removed: usize,
    pub bytes: u64,
}

/// `~/.orangu/server` — the shared root under which both `web::sessions`'
/// `sessions/` directory and every model's own `<fingerprint>/slots/`
/// directory live. `None` when the home directory can't be resolved.
fn server_root() -> Option<PathBuf> {
    Some(home::home_dir()?.join(".orangu/server"))
}

/// Deletes persisted slot files (`<fingerprint>/slots/*.bin` under
/// `~/.orangu/server`) whose last-modified time is older than `max_age`, plus
/// any leftover `*.tmp` from an interrupted save, and removes a `slots/`
/// directory left empty afterward. Best-effort: a filesystem error on one
/// entry is skipped, never propagated.
///
/// Age is the only staleness signal used, deliberately: a slot file is named
/// by the *client's* session id (not the server's own `sessions/<uuid>`), so
/// the server can't tell whether the client session still exists — and the
/// file is a pure reprefill-avoidance cache anyway, so deleting one that's
/// still wanted costs only a one-time prefill, never data. Called
/// unconditionally by `orangu-server prune`, the same way it sweeps empty
/// sessions.
pub fn sweep_stale_slot_files(max_age: Duration) -> SlotSweep {
    let Some(root) = server_root() else {
        return SlotSweep {
            removed: 0,
            bytes: 0,
        };
    };
    sweep_stale_slot_files_in(&root, max_age, SystemTime::now())
}

/// The root- and clock-injected core of [`sweep_stale_slot_files`], so tests
/// can drive it against a scratch directory with a controlled `now` (a `now`
/// far in the future makes freshly written files read as stale, without
/// needing to backdate their mtimes).
fn sweep_stale_slot_files_in(root: &Path, max_age: Duration, now: SystemTime) -> SlotSweep {
    let mut sweep = SlotSweep {
        removed: 0,
        bytes: 0,
    };
    let Ok(dirs) = std::fs::read_dir(root) else {
        return sweep;
    };
    for dir in dirs.flatten() {
        // Only `<fingerprint>/slots/` is a slot directory — `sessions/` and the
        // Vulkan pipeline cache's per-adapter dirs have no `slots/` child and
        // are skipped by the failing `read_dir` below.
        let slots_dir = dir.path().join("slots");
        let Ok(files) = std::fs::read_dir(&slots_dir) else {
            continue;
        };
        let mut any_remaining = false;
        for file in files.flatten() {
            let path = file.path();
            let ext = path.extension().and_then(|e| e.to_str());
            let meta = file.metadata().ok();
            let old_enough = meta
                .as_ref()
                .and_then(|m| m.modified().ok())
                .and_then(|mtime| now.duration_since(mtime).ok())
                .is_some_and(|age| age >= max_age);
            // A stale `.bin`, or a stale `.tmp` (an interrupted save's
            // leftover), is removable; a *fresh* `.tmp` might be an in-flight
            // save mid-rename, so the age gate protects it. Anything else stays
            // and blocks the rmdir below.
            let removable = old_enough && matches!(ext, Some("bin") | Some("tmp"));
            if removable {
                let size = meta.as_ref().map(|m| m.len()).unwrap_or(0);
                if std::fs::remove_file(&path).is_ok() {
                    sweep.removed += 1;
                    sweep.bytes += size;
                    continue;
                }
            }
            any_remaining = true;
        }
        if !any_remaining {
            let _ = std::fs::remove_dir(&slots_dir);
        }
    }
    sweep
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::kv_cache::KvCache;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A scratch directory under the system temp dir, unique per call and
    /// removed on drop — keeps slot-file tests off the real `$HOME`.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            static COUNTER: AtomicUsize = AtomicUsize::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let dir =
                std::env::temp_dir().join(format!("orangu-slot-test-{}-{n}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A cache whose layers each hold `len` committed positions of `kv_dim`
    /// floats, filled with distinct values so a round-trip is verifiable.
    fn filled_cache(n_layer: usize, kv_dim: usize, len: usize) -> KvCache {
        let mut c = KvCache::new(n_layer, len.max(1), kv_dim);
        for (li, layer) in c.layers.iter_mut().enumerate() {
            for pos in 0..len {
                let base = (li * 1000 + pos * 10) as f32;
                let k: Vec<f32> = (0..kv_dim).map(|d| base + d as f32).collect();
                let v: Vec<f32> = (0..kv_dim).map(|d| base - d as f32).collect();
                layer.push(&k, &v);
            }
        }
        c
    }

    fn store(dir: &std::path::Path, n_slots: usize) -> SlotStore {
        SlotStore::at(dir.to_path_buf(), "fptest".to_string(), n_slots)
    }

    /// Background work fills an empty slot and never displaces a client's
    /// conversation.
    #[test]
    fn background_work_never_replaces_what_a_slot_retains() {
        let dir = std::env::temp_dir().join(format!("orangu-slots-bg-{}", std::process::id()));
        let store = store(&dir, 1);
        store.fill_if_empty(0, vec![9, 9, 9, 9], filled_cache(1, 4, 4));
        let mut dst = KvCache::new(1, 8, 4);
        assert_eq!(
            store.reuse_into(0, &[9, 9, 9, 9, 1], &mut dst),
            4,
            "an empty slot takes it"
        );
        store.retain(0, vec![1, 2, 3, 4], filled_cache(1, 4, 4));
        store.fill_if_empty(0, vec![9, 9, 9, 9], filled_cache(1, 4, 4));
        let mut dst = KvCache::new(1, 8, 4);
        assert_eq!(
            store.reuse_into(0, &[1, 2, 3, 4, 5], &mut dst),
            4,
            "the client's conversation was displaced"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn kv_cache_bytes_round_trip_preserves_committed_positions() {
        let cache = filled_cache(3, 4, 5);
        let bytes = cache.to_bytes();
        let restored = KvCache::from_bytes(&bytes).unwrap();
        assert_eq!(restored.layers.len(), 3);
        assert_eq!(restored.committed_len(), 5);
        // A restored cache is a valid copy_prefix_from source: copying its
        // full length into a fresh cache reproduces the same K/V bytes.
        let mut dst = KvCache::new(3, 8, 4);
        dst.copy_prefix_from(&restored, 5);
        assert_eq!(dst.to_bytes(), cache.to_bytes());
    }

    #[test]
    fn from_bytes_rejects_truncated_input() {
        let bytes = filled_cache(2, 4, 3).to_bytes();
        assert!(KvCache::from_bytes(&bytes[..bytes.len() - 4]).is_err());
        assert!(KvCache::from_bytes(b"not a real blob").is_err());
    }

    #[test]
    fn save_then_restore_round_trips_through_disk() {
        let tmp = TempDir::new();
        let s = store(&tmp.0, 2);
        s.retain(1, vec![1, 2, 3, 4, 5], filled_cache(2, 4, 5));
        let n_saved = s.save(1, "session.bin").unwrap();
        assert_eq!(n_saved, 5);

        // A fresh store (as if the server restarted) restores from the file.
        let s2 = store(&tmp.0, 2);
        let n_restored = s2.restore(0, "session.bin").unwrap();
        assert_eq!(n_restored, 5);

        // The restored slot now serves the saved prefix for a matching prompt.
        let mut dst = KvCache::new(2, 16, 4);
        let reused = s2.reuse_into(0, &[1, 2, 3, 4, 5, 9], &mut dst);
        assert_eq!(reused, 5);
    }

    #[test]
    fn reuse_leaves_at_least_one_token_and_matches_prefix() {
        let tmp = TempDir::new();
        let s = store(&tmp.0, 1);
        s.retain(0, vec![1, 2, 3, 4], filled_cache(1, 4, 4));

        // Exact full-prompt match: capped to keep one real forward token.
        let mut dst = KvCache::new(1, 8, 4);
        assert_eq!(s.reuse_into(0, &[1, 2, 3, 4], &mut dst), 3);

        // Longer prompt sharing the whole retained prefix reuses all of it.
        let mut dst = KvCache::new(1, 8, 4);
        assert_eq!(s.reuse_into(0, &[1, 2, 3, 4, 5], &mut dst), 4);

        // No shared prefix: nothing reused.
        let mut dst = KvCache::new(1, 8, 4);
        assert_eq!(s.reuse_into(0, &[9, 9, 9], &mut dst), 0);
    }

    /// A server alone never copies another slot's conversation, however
    /// much of the prompt it matches — its own slot's, as always. (In a
    /// tree of workers it does: `workers::pipeline`'s
    /// `another_slot_s_prefix_is_copied_on_every_worker`.)
    #[test]
    fn a_server_alone_keeps_to_its_own_slot() {
        let tmp = TempDir::new();
        let s = store(&tmp.0, 2);
        let shared: Vec<u32> = (1..=100).collect();
        s.retain(1, shared.clone(), filled_cache(2, 4, 100));
        let mut prompt = shared[..90].to_vec();
        prompt.push(999);
        let mut dst = KvCache::new(2, 128, 4);
        assert_eq!(s.reuse_into(0, &prompt, &mut dst), 0);
        s.retain(0, shared[..60].to_vec(), filled_cache(2, 4, 60));
        let mut own = KvCache::new(2, 128, 4);
        assert_eq!(s.reuse_into(0, &prompt, &mut own), 60);
    }

    #[test]
    fn restore_of_missing_or_foreign_file_is_a_no_op_not_an_error() {
        let tmp = TempDir::new();
        let s = store(&tmp.0, 1);
        // Missing file.
        assert_eq!(s.restore(0, "nope.bin").unwrap(), 0);

        // A file saved under a different fingerprint must not be restored.
        let other = SlotStore::at(tmp.0.clone(), "different-model".to_string(), 1);
        other.retain(0, vec![1, 2, 3], filled_cache(1, 4, 3));
        other.save(0, "x.bin").unwrap();
        assert_eq!(s.restore(0, "x.bin").unwrap(), 0);
    }

    #[test]
    fn unsafe_filenames_are_rejected() {
        assert!(is_safe_filename("session.bin"));
        assert!(is_safe_filename("a-b_c.9"));
        assert!(!is_safe_filename(""));
        assert!(!is_safe_filename("."));
        assert!(!is_safe_filename(".."));
        assert!(!is_safe_filename("../escape"));
        assert!(!is_safe_filename("a/b"));
        assert!(!is_safe_filename("/abs"));
        assert!(!is_safe_filename("a\\b"));

        let tmp = TempDir::new();
        let s = store(&tmp.0, 1);
        s.retain(0, vec![1], filled_cache(1, 4, 1));
        assert!(s.save(0, "../escape.bin").is_err());
        assert!(s.restore(0, "../escape.bin").is_err());
    }

    #[test]
    fn fingerprint_separates_models() {
        let a = SlotStore::fingerprint("llama", "model-a", &[1, 2, 3]);
        let b = SlotStore::fingerprint("llama", "model-b", &[1, 2, 3]);
        let c = SlotStore::fingerprint("gemma", "model-a", &[1, 2, 3]);
        let d = SlotStore::fingerprint("llama", "model-a", &[9, 9, 9]);
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);
        // Stable for identical inputs.
        assert_eq!(a, SlotStore::fingerprint("llama", "model-a", &[1, 2, 3]));
    }

    #[test]
    fn sweep_removes_only_stale_slot_files_and_tidies_empty_dirs() {
        let tmp = TempDir::new();
        let fp_a = tmp.0.join("aaa").join("slots");
        std::fs::create_dir_all(&fp_a).unwrap();
        std::fs::write(fp_a.join("s1.bin"), b"1234567890").unwrap();
        std::fs::write(fp_a.join("s2.bin"), b"xx").unwrap();
        std::fs::write(fp_a.join("half.tmp"), b"y").unwrap();
        // A Vulkan-pipeline-cache-shaped sibling: `<key>/cache.bin`, no
        // `slots/` child — must never be treated as a slot dir.
        let vk = tmp.0.join("adapterkey");
        std::fs::create_dir_all(&vk).unwrap();
        std::fs::write(vk.join("cache.bin"), b"pipeline").unwrap();

        // A `now` far in the future makes every file read as older than the
        // 1s max_age, so `fp_a` is fully swept (a real prune uses the actual
        // clock and a 30-day threshold). Its stale `.bin`s and leftover `.tmp`
        // all go; the emptied `slots/` dir is removed; the pipeline cache is
        // untouched.
        let future = SystemTime::now() + Duration::from_secs(10_000);
        let swept = sweep_stale_slot_files_in(&tmp.0, Duration::from_secs(1), future);
        assert_eq!(swept.removed, 3);
        assert_eq!(swept.bytes, 10 + 2 + 1);
        assert!(!fp_a.exists());
        assert!(vk.join("cache.bin").exists());

        // A fresh file under the real clock (age ≈ 0 < a large max_age) is
        // left alone, and its `slots/` dir is not removed.
        let fp_b = tmp.0.join("bbb").join("slots");
        std::fs::create_dir_all(&fp_b).unwrap();
        std::fs::write(fp_b.join("keep.bin"), b"z").unwrap();
        let swept =
            sweep_stale_slot_files_in(&tmp.0, Duration::from_secs(10_000), SystemTime::now());
        assert_eq!(swept.removed, 0);
        assert!(fp_b.join("keep.bin").exists());
    }

    #[test]
    fn save_of_empty_slot_writes_nothing_and_reports_zero() {
        let tmp = TempDir::new();
        let s = store(&tmp.0, 1);
        assert_eq!(s.save(0, "empty.bin").unwrap(), 0);
        assert!(!tmp.0.join("empty.bin").exists());
    }

    /// The pool's published pages survive a save and a restore into a fresh
    /// pool: the same tag, the same tokens, the same rows, found again; a
    /// file of another model's restores nothing.
    #[test]
    fn the_pools_pages_survive_a_save_and_a_restore() {
        use crate::engine::kv_pool::{KvPool, LayerGeometry};
        use crate::engine::prefix_index::PrefixIndex;
        let geometry = || {
            vec![LayerGeometry {
                kv_dim: 4,
                stride: 1,
                ring: None,
            }]
        };
        let dir = tempfile::tempdir().unwrap();
        let store = SlotStore::at(dir.path().to_path_buf(), "fp".into(), 1);
        let (pool, index) = (KvPool::new(4, 2, geometry()), PrefixIndex::new(2));
        let tag = 0xABCD_u64;
        let page = pool.acquire(&[tag]).unwrap()[0].page;
        let (k, v) = (vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![-1.0; 8]);
        pool.fill(0, page, &k, &v);
        pool.seal(page);
        index.remember(tag, &[11, 12]);
        pool.release(&[page]);
        assert_eq!(store.save_pool(&pool, &index, "p.bin", 1 << 20).unwrap(), 1);

        let (fresh, fresh_index) = (KvPool::new(4, 2, geometry()), PrefixIndex::new(2));
        assert_eq!(
            store.restore_pool(&fresh, &fresh_index, "p.bin").unwrap(),
            1
        );
        assert!(fresh.holds(tag));
        assert_eq!(fresh_index.tokens_of(tag), Some(vec![11, 12]));
        let back = fresh.acquire(&[tag]).unwrap()[0];
        assert!(back.hit);
        assert_eq!(fresh.page_k(0, back.page), &k[..]);
        assert_eq!(fresh.page_v(0, back.page), &v[..]);

        let other = SlotStore::at(dir.path().to_path_buf(), "other".into(), 1);
        let (third, third_index) = (KvPool::new(4, 2, geometry()), PrefixIndex::new(2));
        assert_eq!(
            other.restore_pool(&third, &third_index, "p.bin").unwrap(),
            0
        );
        assert!(
            other
                .restore_pool(&third, &third_index, "../p.bin")
                .is_err()
        );
    }
}
