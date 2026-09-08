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

//! An on-disk cache of blocks compiled for the NPU.
//!
//! Compiling a block costs seconds and running it costs milliseconds, so a
//! serving path cannot compile. It also *must* not: compiling and executing
//! load two copies of the same vendor driver and corrupt each other's symbol
//! bindings, which [`crate::npu_ort`] documents at length. The two have to be
//! separate processes, and something has to carry artifacts between them.
//!
//! This is that something. One process compiles into the cache; another opens
//! it and runs. The cache is keyed by the model it came from, so pointing a
//! server at a different or rebuilt GGUF misses rather than silently running
//! another model's weights.

use std::io;
use std::path::{Path, PathBuf};

use crate::gguf::GgufFile;
use crate::npu_ort::CompiledGatedFfn;

/// Length of [`crate::npu_ort::FFN_MAGIC`], for the staleness check.
///
/// **Derived, not written down.** This was a hand-kept `16` and the magic
/// has a version number on the end of it, so bumping that number to two
/// digits left this reading one byte short: the comparison could then never
/// match, `contains` reported every cached block stale, and the cache
/// silently stopped working. Taken from the constant it describes, that
/// cannot happen again.
const FFN_MAGIC_LEN: usize = crate::npu_ort::FFN_MAGIC.len();

/// Which compiled block is wanted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockKey<'a> {
    /// The model's fingerprint — see [`fingerprint`].
    pub model: &'a str,
    /// The block's tensor-name prefix, such as `v.blk.0`.
    pub block: &'a str,
    /// The token count the block was compiled for. A graph has one static
    /// shape, so this is part of its identity rather than a detail.
    pub tokens: usize,
    /// A digest of the calibration the block was compiled against, or `0`
    /// when it was compiled without any.
    ///
    /// **Part of the identity, because the calibration decides what the
    /// artifact computes.** A block's activation quantizer takes its range
    /// from calibration data, so the same weights at the same width
    /// compiled against two different captures are two different programs —
    /// one of which, on the other's input, returns an output with no signal
    /// in it. Without this in the key, re-capturing a model silently reused
    /// blocks built for the capture before it.
    pub calibration: u64,
}

impl BlockKey<'_> {
    /// The file name this key maps to.
    ///
    /// The block name goes in verbatim apart from characters that cannot
    /// appear in a path component; tensor prefixes are already
    /// `v.blk.0`-shaped, so the result stays readable — someone looking at
    /// the directory should be able to tell what is in it.
    fn file_name(&self) -> String {
        let block: String = self
            .block
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '.' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        // Zero stays out of the name, so every artifact compiled before
        // calibration existed keeps the name it has and no cache is thrown
        // away by the upgrade itself.
        match self.calibration {
            0 => format!("{}-{block}-{}.npu", self.model, self.tokens),
            digest => format!("{}-{block}-{}-c{digest:016x}.npu", self.model, self.tokens),
        }
    }
}

/// Identifies a GGUF by its shape rather than its path.
///
/// Hashes the tensor table — every name, type, shape and offset — plus where
/// the data starts. Two files with the same fingerprint hold the same
/// tensors at the same places, which is exactly the question a cache needs
/// answered; a copy or a rename matches, an edit or a requantization does
/// not. Reading it costs nothing extra because the table is already parsed.
pub fn fingerprint(gguf: &GgufFile) -> String {
    // FNV-1a, 64-bit. Not a security property: this only has to notice that
    // a model changed, and a wrong answer means a cache miss.
    let mut hash: u64 = 0xcbf5_2913_2984_2225;
    let mut eat = |bytes: &[u8]| {
        for byte in bytes {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(0x1000_0000_01b3);
        }
    };
    eat(&gguf.data_offset.to_le_bytes());
    eat(&(gguf.tensors.len() as u64).to_le_bytes());
    for tensor in &gguf.tensors {
        eat(tensor.name.as_bytes());
        eat(&tensor.ggml_type.to_le_bytes());
        eat(&tensor.offset.to_le_bytes());
        for dim in &tensor.dims {
            eat(&dim.to_le_bytes());
        }
    }
    format!("{hash:016x}")
}

/// A directory of compiled blocks.
pub struct NpuCache {
    dir: PathBuf,
}

impl NpuCache {
    /// Opens a cache directory, creating it if it does not exist.
    pub fn open(dir: impl Into<PathBuf>) -> io::Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    /// `~/.orangu/npu`, beside the rest of orangu's state.
    /// Where compiled blocks live: `~/.orangu/npu`.
    pub fn default_dir() -> PathBuf {
        home::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".orangu")
            .join("npu")
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn path(&self, key: BlockKey<'_>) -> PathBuf {
        self.dir.join(key.file_name())
    }

    /// Whether this block is already compiled, so a compile pass can skip it.
    pub fn contains(&self, key: BlockKey<'_>) -> bool {
        // The file *and* a header this build can read. A format change
        // otherwise leaves a cache that is present enough to skip
        // recompiling and stale enough to fail every load — which is what
        // happened when the artifact gained an activation tag: every block
        // reported a corrupt artifact at startup, on every start, and
        // nothing regenerated them because the files were right there.
        //
        // Reads the magic only, so this stays a directory-entry-and-a-header
        // check rather than a parse of every cached block.
        let Ok(mut file) = std::fs::File::open(self.path(key)) else {
            return false;
        };
        let mut magic = [0u8; FFN_MAGIC_LEN];
        use std::io::Read;
        file.read_exact(&mut magic).is_ok() && magic == *crate::npu_ort::FFN_MAGIC
    }

    /// Whether the file under `key` begins with `magic`.
    ///
    /// The same header-only check [`Self::contains`] makes, for an artifact
    /// kind that identifies itself differently.
    fn has_magic(&self, key: BlockKey<'_>, magic: &[u8]) -> bool {
        let Ok(mut file) = std::fs::File::open(self.path(key)) else {
            return false;
        };
        let mut head = vec![0u8; magic.len()];
        use std::io::Read;
        file.read_exact(&mut head).is_ok() && head == magic
    }

    /// Writes a compiled block.
    ///
    /// Written to a temporary name and renamed into place, so an interrupted
    /// compile leaves no half-written artifact for a later run to load as if
    /// it were whole.
    pub fn store(&self, key: BlockKey<'_>, compiled: &CompiledGatedFfn) -> io::Result<PathBuf> {
        self.store_bytes(key, &compiled.to_bytes())
    }

    /// [`Self::store`] for an artifact already serialized.
    pub fn store_bytes(&self, key: BlockKey<'_>, bytes: &[u8]) -> io::Result<PathBuf> {
        let final_path = self.path(key);
        let temporary = final_path.with_extension("npu.partial");
        std::fs::write(&temporary, bytes)?;
        std::fs::rename(&temporary, &final_path)?;
        Ok(final_path)
    }

    /// Reads a compiled block back, or `None` if it is not cached.
    ///
    /// A cached file that will not parse is an error rather than a miss: a
    /// miss invites a silent fallback, where a corrupt artifact is something
    /// the operator should hear about.
    /// A compiled projection set, for the attention `Q`/`K`/`V` triple.
    ///
    /// A second reader rather than a generic one, because the two artifact
    /// kinds have different magics and the caller always knows which it
    /// asked for — a key naming `attn.blk.3` is never a feed-forward block.
    pub fn load_projections(
        &self,
        key: BlockKey<'_>,
    ) -> io::Result<Option<crate::npu_ort::CompiledProjections>> {
        let path = self.path(key);
        if !path.is_file() {
            return Ok(None);
        }
        let bytes = std::fs::read(&path)?;
        crate::npu_ort::CompiledProjections::from_bytes(&bytes)
            .map(Some)
            .map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("{}: {e}", path.display()),
                )
            })
    }

    /// Writes a compiled projection set.
    pub fn store_projections(
        &self,
        key: BlockKey<'_>,
        compiled: &crate::npu_ort::CompiledProjections,
    ) -> io::Result<PathBuf> {
        self.store_bytes(key, &compiled.to_bytes())
    }

    /// Whether a compiled projection set is already cached under this key.
    pub fn contains_projections(&self, key: BlockKey<'_>) -> bool {
        self.has_magic(key, crate::npu_ort::PROJ_MAGIC)
    }

    pub fn load(&self, key: BlockKey<'_>) -> io::Result<Option<CompiledGatedFfn>> {
        let path = self.path(key);
        if !path.is_file() {
            return Ok(None);
        }
        let bytes = std::fs::read(&path)?;
        CompiledGatedFfn::from_bytes(&bytes).map(Some).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}: {e}", path.display()),
            )
        })
    }

    /// Every artifact in the cache, as `(file name, bytes)`, newest last.
    pub fn entries(&self) -> io::Result<Vec<(String, u64)>> {
        let mut out = Vec::new();
        for entry in std::fs::read_dir(&self.dir)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.ends_with(".npu") {
                out.push((name, entry.metadata()?.len()));
            }
        }
        out.sort();
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("orangu-npu-cache-{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    /// A key names one block of one model at one shape, and reads back as
    /// a file name that says so.
    /// Two captures of the same model are two different programs, and the
    /// cache has to tell them apart: a block compiled against one returns
    /// nothing useful on the other's input.
    #[test]
    fn a_different_calibration_is_a_different_cache_entry() {
        let of = |calibration| {
            BlockKey {
                model: "abc123",
                block: "blk.0",
                tokens: 16,
                calibration,
            }
            .file_name()
        };
        assert_ne!(of(1), of(2));
        // And an artifact from before calibration existed keeps its name,
        // so introducing this threw no cache away.
        assert_eq!(of(0), "abc123-blk.0-16.npu");
    }

    #[test]
    fn a_key_names_the_block_it_stands_for() {
        let key = BlockKey {
            model: "0123456789abcdef",
            block: "v.blk.12",
            tokens: 196,
            calibration: 0,
        };
        assert_eq!(key.file_name(), "0123456789abcdef-v.blk.12-196.npu");

        // Anything that cannot be a path component is replaced rather than
        // escaping into the directory.
        let awkward = BlockKey {
            block: "../../etc/passwd",
            ..key
        };
        assert!(!awkward.file_name().contains('/'));
    }

    /// The token count is part of a block's identity, because a graph is
    /// compiled for exactly one shape.
    #[test]
    fn the_token_count_is_part_of_the_key() {
        let base = BlockKey {
            model: "aaaa",
            block: "v.blk.0",
            tokens: 196,
            calibration: 0,
        };
        let other = BlockKey {
            tokens: 128,
            ..base
        };
        assert_ne!(base.file_name(), other.file_name());
    }

    #[test]
    fn a_stored_block_reads_back_and_a_missing_one_is_absent() {
        let cache = NpuCache::open(scratch("roundtrip")).expect("open");
        let key = BlockKey {
            model: "feedface",
            block: "v.blk.0",
            tokens: 8,
            calibration: 0,
        };
        assert!(!cache.contains(key));
        assert!(cache.load(key).expect("load").is_none());

        let compiled = CompiledGatedFfn::from_bytes(&sample_ffn()).expect("sample");
        cache.store(key, &compiled).expect("store");

        assert!(cache.contains(key));
        let back = cache.load(key).expect("load").expect("present");
        assert_eq!(back.binary_len(), compiled.binary_len());
        assert_eq!(back.in_features(), compiled.in_features());
        assert_eq!(back.max_tokens(), compiled.max_tokens());

        assert_eq!(cache.entries().expect("entries").len(), 1);
        let _ = std::fs::remove_dir_all(cache.dir());
    }

    /// A cached file that is not an artifact is an error, not a miss —
    /// a miss would quietly recompile or quietly fall back.
    #[test]
    fn a_corrupt_artifact_is_reported() {
        let cache = NpuCache::open(scratch("corrupt")).expect("open");
        let key = BlockKey {
            model: "deadbeef",
            block: "v.blk.1",
            tokens: 4,
            calibration: 0,
        };
        std::fs::write(cache.path(key), b"not an artifact").expect("write");
        assert!(cache.load(key).is_err());
        let _ = std::fs::remove_dir_all(cache.dir());
    }

    /// A model whose tensors moved is a different model.
    #[test]
    fn the_fingerprint_follows_the_tensor_table() {
        use crate::gguf::TensorInfo;
        let make = |offset: u64| GgufFile {
            version: 3,
            metadata: Vec::new(),
            tensors: vec![TensorInfo {
                name: "w".into(),
                dims: vec![2, 2],
                ggml_type: 0,
                offset,
            }],
            alignment: 32,
            data_offset: 64,
        };
        assert_eq!(fingerprint(&make(0)), fingerprint(&make(0)));
        assert_ne!(fingerprint(&make(0)), fingerprint(&make(32)));
    }

    /// The smallest thing `CompiledLinear::from_bytes` accepts.
    ///
    /// Hand-rolled, and it has to be: `CompiledLinear` has no public
    /// constructor by design — a compiled block comes from the compiler.
    /// That means this **does** drift when the format changes, and it did:
    /// the header gained an input count and the magic went to `-2`, and this
    /// went on claiming it could not drift while failing, then the artifact
    /// gained per-channel smoothing scales and the magic went to `-3`. If a
    /// field is added there again, it must be added here too.
    fn sample_artifact() -> Vec<u8> {
        // **The constant, not a copy of it.** This fixture has now drifted
        // twice behind a magic bump and failed as a corrupt artifact both
        // times, which tests nothing anyone wanted tested.
        let mut bytes = crate::npu_ort::ARTIFACT_MAGIC.to_vec();
        // `k`, `n`, `max_tokens`, then the graph's input count.
        for value in [4u32, 4, 8, 1] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for _ in 0..2 {
            bytes.extend_from_slice(&1.0f32.to_le_bytes());
            bytes.extend_from_slice(&0u32.to_le_bytes());
        }
        // How many per-channel smoothing scales follow. None here — and the
        // third time this fixture has drifted, which is what the note above
        // is for.
        bytes.extend_from_slice(&0u32.to_le_bytes());
        let payload = b"\x7fELF a compiled block";
        bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        bytes.extend_from_slice(payload);
        bytes
    }

    /// The three-projection form a cache entry actually holds: the magic,
    /// then each part length-prefixed. Same hand-rolled caveat as
    /// [`sample_artifact`].
    fn sample_ffn() -> Vec<u8> {
        let part = sample_artifact();
        let mut bytes = crate::npu_ort::FFN_MAGIC.to_vec();
        // The host activation's tag: 0 is GELU. Added when the compiler
        // learned SwiGLU for the llama family.
        bytes.push(0);
        for _ in 0..3 {
            bytes.extend_from_slice(&(part.len() as u32).to_le_bytes());
            bytes.extend_from_slice(&part);
        }
        bytes
    }
}
