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

//! Weights the picture pipelines re-encode at load, kept on disk.
//!
//! A VAE published as `bf16` safetensors, and a `BF16` vision projector,
//! are encoded at load to `Q6_K` and to the per-row `int8` copy the CPU's
//! tile kernel reads ([`super::vae::conv_matrix`], [`super::vae::row_matrix`]).
//! That is the same work every start, and seconds of it per model, so each
//! result is written to `~/.orangu/cache/encoded/` the first time and read
//! back after.
//!
//! An entry is named by a hash of everything its bytes depend on: the
//! input rows themselves, their shape, the kind of encoding, and this
//! build. A changed file, a different model or a new build can therefore
//! never be served another's bytes; they only miss. Each build keeps its own
//! directory, and only the two most recently used are kept. A cache that
//! cannot be read or written is a slower start, never a failure.
//! `ORANGU_IMAGE_ENCODED_CACHE=0` turns it off.

use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// Bumped whenever an entry's layout, or the meaning of a key, changes.
const FORMAT: u32 = 1;

/// How many builds' directories are kept: this one and the last other one
/// used, so two builds started in turn (an A/B) do not evict each other.
const BUILDS_KEPT: usize = 2;

static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);
static HIT_BYTES: AtomicU64 = AtomicU64::new(0);

/// This build's directory, or `None` when the cache is off or cannot be made.
fn dir() -> Option<&'static Path> {
    static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
    DIR.get_or_init(|| {
        if !crate::engine::env::flag_on_unless_disabled("ORANGU_IMAGE_ENCODED_CACHE") {
            return None;
        }
        let root = home::home_dir()?
            .join(".orangu")
            .join("cache")
            .join("encoded");
        let build = hex(&Sha256::digest(orangu::build_info::id().as_bytes())[..8]);
        let dir = root.join(build);
        std::fs::create_dir_all(&dir).ok()?;
        // Marks this build's directory as the most recently used.
        let _ = std::fs::File::create(dir.join(".used"));
        prune(&root, &dir);
        Some(dir)
    })
    .as_deref()
}

/// Removes every build directory but the [`BUILDS_KEPT`] most recently used.
fn prune(root: &Path, current: &Path) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    let mut others: Vec<(std::time::SystemTime, PathBuf)> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir() && p != current)
        .map(|p| {
            let used = std::fs::metadata(p.join(".used"))
                .and_then(|m| m.modified())
                .unwrap_or(std::time::UNIX_EPOCH);
            (used, p)
        })
        .collect();
    others.sort_by_key(|(used, _)| std::cmp::Reverse(*used));
    for (_, stale) in others.into_iter().skip(BUILDS_KEPT - 1) {
        let _ = std::fs::remove_dir_all(stale);
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The entry name for encoding `rows` (shape `dims`) as `kind`.
fn key(kind: &str, rows: &[f32], dims: &[usize]) -> String {
    // Hashed in parallel pieces, then the pieces' digests in order: a
    // vision tower is gigabytes of `f32` once widened.
    // Small pieces: most of a VAE's tensors are under a million values, and
    // a piece is hashed on one core.
    let pieces: Vec<Vec<u8>> = rows
        .par_chunks(1 << 14)
        .map(|chunk| Sha256::digest(bytemuck::cast_slice::<f32, u8>(chunk)).to_vec())
        .collect();
    let mut h = Sha256::new();
    h.update(FORMAT.to_le_bytes());
    h.update(kind.as_bytes());
    h.update((rows.len() as u64).to_le_bytes());
    for d in dims {
        h.update((*d as u64).to_le_bytes());
    }
    for piece in &pieces {
        h.update(piece);
    }
    format!("{}.{kind}", hex(&h.finalize()[..16]))
}

/// The bytes stored for `(kind, rows, dims)`, made by `make` and stored the
/// first time. `valid` checks what was read back — a torn or foreign file
/// is encoded again rather than trusted.
fn cached(
    kind: &str,
    rows: &[f32],
    dims: &[usize],
    valid: impl Fn(&[u8]) -> bool,
    make: impl FnOnce() -> Vec<u8>,
) -> Vec<u8> {
    let Some(dir) = dir() else {
        return make();
    };
    let path = dir.join(key(kind, rows, dims));
    if let Ok(bytes) = std::fs::read(&path)
        && valid(&bytes)
    {
        HITS.fetch_add(1, Ordering::Relaxed);
        HIT_BYTES.fetch_add(bytes.len() as u64, Ordering::Relaxed);
        return bytes;
    }
    MISSES.fetch_add(1, Ordering::Relaxed);
    let bytes = make();
    // Written aside and renamed, so a reader never sees half an entry.
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    if std::fs::write(&tmp, &bytes).is_ok() {
        let _ = std::fs::rename(&tmp, &path);
    } else {
        let _ = std::fs::remove_file(&tmp);
    }
    bytes
}

/// `Q6_K` rows: `rows` is what the encoding reads, `dims` its shape.
pub(crate) fn q6_k(
    rows: &[f32],
    dims: &[usize],
    expected_len: usize,
    make: impl FnOnce() -> Vec<u8>,
) -> Vec<u8> {
    cached("q6k", rows, dims, |b| b.len() == expected_len, make)
}

/// The per-row `int8` copy of `rows`.
pub(crate) fn row_i8(
    rows: &[f32],
    dims: &[usize],
    make: impl FnOnce() -> crate::engine::vecdot::RowI8,
) -> crate::engine::vecdot::RowI8 {
    let mut made = None;
    let bytes = cached(
        "rowi8",
        rows,
        dims,
        |b| crate::engine::vecdot::RowI8::from_bytes(b).is_some(),
        || {
            let m = make();
            let bytes = m.to_bytes();
            made = Some(m);
            bytes
        },
    );
    match made {
        Some(m) => m,
        None => {
            crate::engine::vecdot::RowI8::from_bytes(&bytes).expect("validated when it was read")
        }
    }
}

/// What the cache did since the last call: `(read back, encoded, bytes read)`.
pub fn take_counts() -> (u64, u64, u64) {
    (
        HITS.swap(0, Ordering::Relaxed),
        MISSES.swap(0, Ordering::Relaxed),
        HIT_BYTES.swap(0, Ordering::Relaxed),
    )
}

#[cfg(test)]
mod tests {
    /// The key moves with every input it depends on, and only with them.
    #[test]
    fn the_key_follows_the_rows_the_shape_and_the_kind() {
        let rows: Vec<f32> = (0..1000).map(|i| i as f32 * 0.5).collect();
        let base = super::key("q6k", &rows, &[10, 100]);
        assert_eq!(base, super::key("q6k", &rows, &[10, 100]));
        let mut changed = rows.clone();
        changed[999] += 1.0;
        assert_ne!(base, super::key("q6k", &changed, &[10, 100]));
        assert_ne!(base, super::key("q6k", &rows, &[20, 50]));
        assert_ne!(base, super::key("rowi8", &rows, &[10, 100]));
    }
}
