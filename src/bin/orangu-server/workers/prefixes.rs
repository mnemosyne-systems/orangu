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

//! A tree's prefix pool, saved per node: every node of a tree writes the
//! rows a pool entry holds on its own layers to its own disk, under a key the
//! top-level node chose ([`super::protocol::Message::Save`]), and opens a
//! session from them on a later claim ([`super::protocol::Message::Load`]).
//!
//! Each node keeps its files under
//! `~/.orangu/server/<model header hash>/tree-prefixes/<node name>/`: one
//! `<key>.bin` per entry. The top-level node keeps its own layers' files and a
//! `manifest.json` — every entry's key and length, and the split of the tree
//! the entries were saved under — in `lead/` beside them.
//!
//! A file says what it holds: the model, the layers, the cache's shape, the
//! context it was made for and the positions it holds. Every one of them is
//! checked before the rows are used; a file that differs in any is refused,
//! never read as the rows of something else.

use super::protocol::SavedPrefix;
use crate::engine::kv_cache::KvCache;
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use std::io::Read;
use std::ops::Range;
use std::path::{Path, PathBuf};

/// Marks a file as one of these, so a stray file is skipped rather than read.
const MAGIC: &[u8; 8] = b"ORGUTPX1";

/// The layout of a file. A file of another version is refused.
const VERSION: u32 = 1;

/// The top-level node's record of what it saved.
const MANIFEST: &str = "manifest.json";

/// A new key: 128 random bits, as 32 lowercase hex digits.
pub fn new_key() -> String {
    format!("{:032x}", rand::random::<u128>())
}

/// Whether `key` is one [`new_key`] could have made: what a node checks
/// before it names a file after a key its parent sent.
pub fn valid_key(key: &str) -> bool {
    key.len() == 32 && key.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// What a saved file must agree with to be loaded.
pub struct Expect<'a> {
    /// The layers the loading node runs itself.
    pub layers: Range<usize>,
    /// [`KvCache::structure_tag`] of a cache for those layers.
    pub shape: &'a [u8],
    /// The context the session is opened with.
    pub n_ctx: usize,
    /// The positions the session is opened with.
    pub len: usize,
}

/// A file's header: everything but the rows.
struct Header {
    layers: Range<usize>,
    shape: Vec<u8>,
    n_ctx: usize,
    len: usize,
    saved_at: u64,
    tokens: Vec<u32>,
}

/// One node's saved prefixes on its disk.
pub struct PrefixFiles {
    dir: PathBuf,
    header_hash: [u8; 32],
}

/// The split of a tree its entries were saved under, and the entries: what a
/// top-level node plans and loads by on a later claim.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    /// The context every entry's sessions on the workers were made for.
    pub n_ctx: usize,
    /// The layers the top-level node ran itself.
    pub local: (usize, usize),
    /// The top-level node's budget for layers then.
    #[serde(default)]
    pub local_budget: u64,
    /// Every worker that took a share, in the order the layers pass through
    /// them: its name and its share, its subtree included.
    pub workers: Vec<SavedNode>,
    /// Every node of the tree and the layers it ran itself, as the plan had it.
    pub plan: Vec<SavedNode>,
    /// The entries, oldest first.
    pub entries: Vec<SavedEntry>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SavedNode {
    /// The node's name in plans, `host:port`: what stays the same across its
    /// restarts.
    pub name: String,
    pub layers: (usize, usize),
    /// Its budget for layers then, as it reported it; `0` in the plan's list.
    #[serde(default)]
    pub budget: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SavedEntry {
    pub key: String,
    /// The positions every node holds of it.
    pub len: usize,
    /// The capacity of this node's cache of it, as the request that made it
    /// sized it.
    #[serde(default)]
    pub capacity: usize,
}

impl Manifest {
    /// The layers the top-level node ran itself.
    pub fn local(&self) -> Range<usize> {
        self.local.0..self.local.1
    }
}

impl SavedNode {
    pub fn layers(&self) -> Range<usize> {
        self.layers.0..self.layers.1
    }
}

impl PrefixFiles {
    /// The files of the node named `node` serving the model whose
    /// [`super::protocol::ModelIdentity::header_hash`] is `header_hash`, under
    /// `root` (`~/.orangu/server`).
    pub fn under(root: &Path, header_hash: [u8; 32], node: &str) -> Self {
        let model: String = header_hash.iter().map(|b| format!("{b:02x}")).collect();
        let node: String = node
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        Self {
            dir: root.join(model).join("tree-prefixes").join(node),
            header_hash,
        }
    }

    #[cfg(test)]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn lead_dir(&self) -> PathBuf {
        self.dir.join("lead")
    }

    /// Where `key`'s file is, as a worker's (`lead` false) or the top-level
    /// node's own. `None` for a key that is not one.
    fn path(&self, key: &str, lead: bool) -> Option<PathBuf> {
        if !valid_key(key) {
            return None;
        }
        let dir = if lead {
            self.lead_dir()
        } else {
            self.dir.clone()
        };
        Some(dir.join(format!("{key}.bin")))
    }

    /// Writes the first `len` positions `cache` holds on `layers`, and the
    /// `tokens` they are of (none on a worker), under `key`; returns the bytes
    /// written. Through a temporary file and a rename: a crash leaves the
    /// file whole or absent.
    #[allow(clippy::too_many_arguments)]
    pub fn write(
        &self,
        key: &str,
        lead: bool,
        layers: Range<usize>,
        n_ctx: usize,
        len: usize,
        tokens: &[u32],
        cache: &KvCache,
    ) -> Result<u64> {
        let path = self
            .path(key, lead)
            .ok_or_else(|| anyhow::anyhow!("{key:?} is not a key"))?;
        let shape = cache.structure_tag();
        let mut blob = Vec::new();
        blob.extend_from_slice(MAGIC);
        blob.extend_from_slice(&VERSION.to_le_bytes());
        blob.extend_from_slice(&self.header_hash);
        for value in [layers.start, layers.end, shape.len()] {
            blob.extend_from_slice(&(value as u32).to_le_bytes());
        }
        blob.extend_from_slice(&shape);
        blob.extend_from_slice(&(n_ctx as u32).to_le_bytes());
        blob.extend_from_slice(&(len as u32).to_le_bytes());
        let saved_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64);
        blob.extend_from_slice(&saved_at.to_le_bytes());
        blob.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
        for token in tokens {
            blob.extend_from_slice(&token.to_le_bytes());
        }
        blob.extend_from_slice(&cache.to_bytes());
        write_atomic(&path, &blob)?;
        Ok(blob.len() as u64)
    }

    /// Reads `key`'s file, checking it field by field against this node's
    /// model and `expect`: its tokens and its rows. An error says which field
    /// differs, or that the file is missing or damaged.
    pub fn read(&self, key: &str, lead: bool, expect: &Expect) -> Result<(Vec<u32>, KvCache)> {
        let path = self
            .path(key, lead)
            .ok_or_else(|| anyhow::anyhow!("{key:?} is not a key"))?;
        let bytes =
            std::fs::read(&path).with_context(|| format!("no prefix {key} on this node"))?;
        let mut reader = bytes.as_slice();
        let header = self
            .header(&mut reader)
            .with_context(|| format!("prefix {key}"))?;
        ensure!(
            header.layers == expect.layers,
            "prefix {key} holds layers {:?}, this node runs {:?}",
            header.layers,
            expect.layers
        );
        ensure!(
            header.shape == expect.shape,
            "prefix {key} was saved from a cache of another shape"
        );
        ensure!(
            header.n_ctx == expect.n_ctx,
            "prefix {key} was saved for a context of {}, the session is {}",
            header.n_ctx,
            expect.n_ctx
        );
        ensure!(
            header.len >= expect.len,
            "prefix {key} holds {} positions, not {}",
            header.len,
            expect.len
        );
        let cache =
            KvCache::from_bytes(reader).with_context(|| format!("prefix {key} is damaged"))?;
        ensure!(
            Some(dims_of(&cache)) == dims_of_tag(expect.shape),
            "prefix {key} is damaged: its rows are of another shape than its header says"
        );
        let held = if cache.layers.iter().all(|l| l.kv_dim() == 0) {
            header.len
        } else {
            cache.host_committed_len()
        };
        ensure!(
            held >= expect.len,
            "prefix {key} is damaged: it holds {held} positions, its header says {}",
            header.len
        );
        // A recurrent state is of every position it saw: only all of them.
        ensure!(
            cache.recurrent.is_empty() || (header.len == expect.len && held == expect.len),
            "prefix {key} holds a state of {} positions, not {}",
            header.len,
            expect.len
        );
        Ok((header.tokens, cache))
    }

    /// Reads a header from `reader`, refusing a file of another kind, layout
    /// or model.
    fn header(&self, reader: &mut impl Read) -> Result<Header> {
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic).context("damaged")?;
        ensure!(&magic == MAGIC, "not a saved prefix");
        let version = read_u32(reader)?;
        ensure!(
            version == VERSION,
            "a file of layout {version}, not {VERSION}"
        );
        let mut hash = [0u8; 32];
        reader.read_exact(&mut hash).context("damaged")?;
        ensure!(hash == self.header_hash, "saved from another model");
        let start = read_u32(reader)? as usize;
        let end = read_u32(reader)? as usize;
        let shape_len = read_u32(reader)? as usize;
        ensure!(shape_len <= 1 << 20, "damaged");
        let mut shape = vec![0u8; shape_len];
        reader.read_exact(&mut shape).context("damaged")?;
        let n_ctx = read_u32(reader)? as usize;
        let len = read_u32(reader)? as usize;
        let mut saved_at = [0u8; 8];
        reader.read_exact(&mut saved_at).context("damaged")?;
        let n_tokens = read_u32(reader)? as usize;
        ensure!(n_tokens <= 1 << 26, "damaged");
        let mut tokens = Vec::with_capacity(n_tokens.min(1 << 16));
        for _ in 0..n_tokens {
            tokens.push(read_u32(reader)?);
        }
        if start > end {
            bail!("damaged");
        }
        Ok(Header {
            layers: start..end,
            shape,
            n_ctx,
            len,
            saved_at: u64::from_le_bytes(saved_at),
            tokens,
        })
    }

    /// Deletes `key`'s file, if there is one.
    pub fn remove(&self, key: &str, lead: bool) {
        if let Some(path) = self.path(key, lead) {
            let _ = std::fs::remove_file(path);
        }
    }

    /// Every file this node holds as a worker, with its header and size,
    /// oldest first. A file that is not one, or of another model, is left
    /// out.
    fn entries(&self) -> Vec<(String, Header, u64)> {
        let Ok(read) = std::fs::read_dir(&self.dir) else {
            return Vec::new();
        };
        let mut out: Vec<(String, Header, u64)> = read
            .flatten()
            .filter_map(|entry| {
                let name = entry.file_name().into_string().ok()?;
                let key = name.strip_suffix(".bin")?.to_string();
                if !valid_key(&key) {
                    return None;
                }
                let file = std::fs::File::open(entry.path()).ok()?;
                let size = file.metadata().ok()?.len();
                let header = self.header(&mut std::io::BufReader::new(file)).ok()?;
                Some((key, header, size))
            })
            .collect();
        out.sort_by(|a, b| a.1.saved_at.cmp(&b.1.saved_at).then(a.0.cmp(&b.0)));
        out
    }

    /// The prefixes this node holds as a worker: what its handshake lists.
    pub fn list(&self) -> Vec<SavedPrefix> {
        self.entries()
            .into_iter()
            .map(|(key, header, _)| SavedPrefix {
                key,
                layer_start: header.layers.start as u32,
                layer_end: header.layers.end as u32,
                len: header.len as u32,
            })
            .collect()
    }

    /// How many prefixes this node holds as a worker, and their bytes.
    pub fn usage(&self) -> (usize, u64) {
        self.entries()
            .iter()
            .fold((0, 0), |(n, bytes), (_, _, size)| (n + 1, bytes + size))
    }

    /// Deletes this node's oldest files as a worker past the newest `keep`.
    pub fn keep_newest(&self, keep: usize) {
        let entries = self.entries();
        let over = entries.len().saturating_sub(keep);
        for (key, _, _) in entries.into_iter().take(over) {
            log::info!(
                "orangu-server: workers: saved prefix {key} deleted, the oldest of more than {keep}"
            );
            self.remove(&key, false);
        }
    }

    /// The top-level node's manifest, if it has one it can read.
    pub fn manifest(&self) -> Option<Manifest> {
        let text = std::fs::read(self.lead_dir().join(MANIFEST)).ok()?;
        serde_json::from_slice(&text).ok()
    }

    /// Writes the top-level node's manifest, or — with no entries — deletes it
    /// and every file of its own.
    pub fn write_manifest(&self, manifest: &Manifest) -> Result<()> {
        if manifest.entries.is_empty() {
            self.clear_lead();
            return Ok(());
        }
        let text = serde_json::to_vec_pretty(manifest)?;
        write_atomic(&self.lead_dir().join(MANIFEST), &text)
    }

    /// Deletes the top-level node's manifest and its own files.
    pub fn clear_lead(&self) {
        let _ = std::fs::remove_dir_all(self.lead_dir());
    }

    /// Drops `key` from the top-level node's manifest and deletes its own file
    /// of it.
    pub fn forget_lead(&self, key: &str) {
        self.remove(key, true);
        if let Some(mut manifest) = self.manifest() {
            let before = manifest.entries.len();
            manifest.entries.retain(|e| e.key != key);
            if manifest.entries.len() != before
                && let Err(error) = self.write_manifest(&manifest)
            {
                log::warn!("orangu-server: workers: the saved prefixes' manifest: {error:#}");
            }
        }
    }
}

/// Each layer's `kv_dim` and the count of recurrent states: the part of a
/// cache's shape its rows carry.
fn dims_of(cache: &KvCache) -> (Vec<u32>, usize) {
    (
        cache.layers.iter().map(|l| l.kv_dim() as u32).collect(),
        cache.recurrent.len(),
    )
}

/// [`dims_of`] as [`KvCache::structure_tag`] wrote it.
fn dims_of_tag(tag: &[u8]) -> Option<(Vec<u32>, usize)> {
    let mut words = tag
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w));
    let n_layer = words.next()? as usize;
    let mut dims = Vec::with_capacity(n_layer.min(4096));
    for _ in 0..n_layer {
        dims.push(words.next()?);
        // The stride.
        words.next()?;
    }
    Some((dims, words.next()? as usize))
}

fn read_u32(reader: &mut impl Read) -> Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes).context("damaged")?;
    Ok(u32::from_le_bytes(bytes))
}

fn write_atomic(path: &Path, data: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, data)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn files(tag: &str) -> PrefixFiles {
        let root = std::env::temp_dir().join(format!(
            "orangu-tree-prefixes-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        PrefixFiles::under(&root, [7; 32], "w1:8400")
    }

    fn cache(len: usize) -> KvCache {
        let mut cache = KvCache::new_with_dims(16, &[0, 4, 4, 0]);
        for layer in &mut cache.layers[1..3] {
            for i in 0..len {
                layer.push(&[i as f32; 4], &[-(i as f32); 4]);
            }
        }
        cache
    }

    fn expect(shape: &[u8], len: usize) -> Expect<'_> {
        Expect {
            layers: 1..3,
            shape,
            n_ctx: 16,
            len,
        }
    }

    #[test]
    fn a_key_is_32_hex_digits_and_nothing_else() {
        let key = new_key();
        assert!(valid_key(&key), "{key}");
        assert_ne!(key, new_key());
        for bad in [
            "",
            "../etc/passwd",
            &key[..31],
            &key.to_uppercase(),
            "g".repeat(32).as_str(),
        ] {
            if bad != key {
                assert!(!valid_key(bad), "{bad}");
            }
        }
    }

    #[test]
    fn a_saved_prefix_reads_back_as_it_was() {
        let files = files("roundtrip");
        let key = new_key();
        let saved = cache(5);
        let shape = saved.structure_tag();
        files.write(&key, false, 1..3, 16, 5, &[], &saved).unwrap();
        let (tokens, loaded) = files.read(&key, false, &expect(&shape, 4)).unwrap();
        assert!(tokens.is_empty());
        assert_eq!(loaded.to_bytes(), saved.to_bytes());
        assert_eq!(
            files.list(),
            vec![SavedPrefix {
                key: key.clone(),
                layer_start: 1,
                layer_end: 3,
                len: 5,
            }]
        );
        assert_eq!(files.usage().0, 1);
        files.remove(&key, false);
        assert!(files.list().is_empty());
    }

    /// Each field is checked: a file that differs in any is refused, and says
    /// why.
    #[test]
    fn a_file_that_does_not_match_is_refused() {
        let files = files("mismatch");
        let key = new_key();
        let saved = cache(5);
        let shape = saved.structure_tag();
        files.write(&key, false, 1..3, 16, 5, &[], &saved).unwrap();
        let refused = |expect: Expect| match files.read(&key, false, &expect) {
            Ok(_) => panic!("read"),
            Err(error) => error.to_string(),
        };
        assert!(
            refused(Expect {
                layers: 0..3,
                ..expect(&shape, 5)
            })
            .contains("layers")
        );
        assert!(
            refused(Expect {
                n_ctx: 32,
                ..expect(&shape, 5)
            })
            .contains("context")
        );
        assert!(refused(expect(&shape, 6)).contains("positions"));
        let other = KvCache::new_with_dims(16, &[0, 8, 8, 0]).structure_tag();
        assert!(refused(expect(&other, 5)).contains("shape"));
        let foreign = PrefixFiles {
            dir: files.dir.clone(),
            header_hash: [8; 32],
        };
        let Err(error) = foreign.read(&key, false, &expect(&shape, 5)) else {
            panic!("another model's file was read");
        };
        assert!(format!("{error:#}").contains("another model"), "{error:#}");
        assert!(
            foreign.list().is_empty(),
            "another model's files are not listed"
        );
        assert!(files.read(&new_key(), false, &expect(&shape, 5)).is_err());
        assert!(files.read("../x", false, &expect(&shape, 5)).is_err());
    }

    #[test]
    fn a_damaged_file_is_refused() {
        let files = files("damaged");
        let key = new_key();
        let saved = cache(5);
        let shape = saved.structure_tag();
        files.write(&key, false, 1..3, 16, 5, &[], &saved).unwrap();
        let path = files.path(&key, false).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, &bytes[..bytes.len() - 7]).unwrap();
        assert!(files.read(&key, false, &expect(&shape, 5)).is_err());
        std::fs::write(&path, &bytes[..20]).unwrap();
        assert!(files.read(&key, false, &expect(&shape, 5)).is_err());
        assert!(files.list().is_empty(), "a damaged header is not listed");
    }

    #[test]
    fn the_oldest_files_go_past_the_newest_few() {
        let files = files("cap");
        let keys: Vec<String> = (0..6).map(|_| new_key()).collect();
        for key in &keys {
            files
                .write(key, false, 1..3, 16, 2, &[], &cache(2))
                .unwrap();
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        files.keep_newest(4);
        let left: Vec<String> = files.list().into_iter().map(|s| s.key).collect();
        assert_eq!(left, keys[2..].to_vec());
    }

    #[test]
    fn the_manifest_and_the_lead_files_go_together() {
        let files = files("manifest");
        let key = new_key();
        files
            .write(&key, true, 0..1, 16, 3, &[1, 2, 3], &cache(3))
            .unwrap();
        let manifest = Manifest {
            n_ctx: 16,
            local: (0, 1),
            local_budget: 1 << 20,
            workers: vec![SavedNode {
                name: "w1:8400".to_string(),
                layers: (1, 4),
                budget: 1 << 30,
            }],
            plan: Vec::new(),
            entries: vec![SavedEntry {
                key: key.clone(),
                len: 3,
                capacity: 16,
            }],
        };
        files.write_manifest(&manifest).unwrap();
        assert_eq!(files.manifest(), Some(manifest));
        assert!(
            files.list().is_empty(),
            "the lead's files are not a worker's"
        );
        files.forget_lead(&key);
        assert_eq!(files.manifest(), None);
        assert!(!files.lead_dir().exists());
    }
}
