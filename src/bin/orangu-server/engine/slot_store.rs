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
    fn at(dir: PathBuf, fingerprint: String, n_slots: usize) -> Self {
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
            *cell.lock().unwrap() = Some(CachedPrefill { tokens, cache });
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
    pub fn reuse_into(&self, slot_id: usize, prompt: &[u32], dst: &mut KvCache) -> usize {
        let Some(cell) = self.retained.get(slot_id) else {
            return 0;
        };
        let guard = cell.lock().unwrap();
        let Some(entry) = guard.as_ref() else {
            return 0;
        };
        let raw = entry.reusable_prefix_len(prompt);
        if raw == 0 {
            return 0;
        }
        let len = if entry.cache.recurrent.is_empty() {
            raw.min(prompt.len().saturating_sub(1))
        } else if raw < prompt.len() {
            raw
        } else {
            0
        };
        if len == 0 {
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
        let guard = cell.lock().unwrap();
        let Some(entry) = guard.as_ref() else {
            return Ok(0);
        };
        let bytes = encode_file(&self.fingerprint, entry);
        let saved = entry.cache.committed_len();
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
                eprintln!(
                    "orangu-server: ignoring unusable slot file {}: {err}",
                    path.display()
                );
                return Ok(0);
            }
        };
        let restored = cache.committed_len();
        // `get` re-checked above, so this index is in range.
        *self.retained[slot_id].lock().unwrap() = Some(CachedPrefill { tokens, cache });
        Ok(restored)
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
    let tokens: Vec<u32> = token_bytes.chunks_exact(4).map(read_u32).collect();
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
}
