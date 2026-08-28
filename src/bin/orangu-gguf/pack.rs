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

//! Corpus text to a token stream on disk.
//!
//! The packed file is the training set: little-endian `u32` ids, one flat
//! sequence, with an end-of-document token between documents. Training
//! samples windows out of it at random offsets, so a window may straddle
//! that boundary — which is deliberate. It is the only way the model ever
//! sees the token that ends a document, and therefore the only way it
//! learns to emit one.
//!
//! Files are deduplicated by content as they are packed. A corpus assembled
//! from public repositories contains the same vendored header, licence
//! text, and generated file dozens of times over; training on those copies
//! teaches nothing except to memorize them.

use anyhow::{Context, Result};
use rayon::prelude::*;
use rustc_hash::FxHashSet;
use std::{
    fs::File,
    hash::{Hash, Hasher},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

use crate::corpus;
use crate::vocab::Encoder;

/// What packing produced.
#[derive(Debug, Default, Clone)]
pub struct PackReport {
    pub documents: usize,
    pub duplicates: usize,
    pub unreadable: usize,
    pub tokens: u64,
    pub bytes: u64,
}

/// Files read at a time. Large enough that the parallel encode is worth its
/// setup, small enough that a corpus of any size stays within a bounded
/// amount of memory.
const CHUNK: usize = 512;

/// Encodes every file into `out`, returning what went in.
///
/// The stream is written to a temporary file and renamed into place at the
/// end. A packed file is reused by later runs on the strength of existing,
/// so an interrupted pack that left a truncated one behind would be picked
/// up as a complete corpus and quietly train on a fraction of it.
pub fn pack(
    files: &[PathBuf],
    encoder: &Encoder<'_>,
    out: &Path,
    progress: &dyn Fn(&PackReport, usize),
) -> Result<PackReport> {
    let temporary = out.with_extension("tmp");
    let file =
        File::create(&temporary).with_context(|| format!("creating {}", temporary.display()))?;
    let mut writer = BufWriter::with_capacity(1 << 20, file);
    let mut report = PackReport::default();
    let mut seen: FxHashSet<u64> = FxHashSet::default();
    let eos = encoder.eos();

    for (n, chunk) in files.chunks(CHUNK).enumerate() {
        let encoded: Vec<Option<(u64, usize, Vec<u32>)>> = chunk
            .par_iter()
            .map(|path| {
                let text = corpus::read_document(path)?;
                let mut hasher = rustc_hash::FxHasher::default();
                text.hash(&mut hasher);
                Some((hasher.finish(), text.len(), encoder.encode(&text)))
            })
            .collect();

        let mut buffer: Vec<u8> = Vec::new();
        for entry in encoded {
            let Some((hash, len, ids)) = entry else {
                report.unreadable += 1;
                continue;
            };
            if !seen.insert(hash) {
                report.duplicates += 1;
                continue;
            }
            report.documents += 1;
            report.bytes += len as u64;
            report.tokens += ids.len() as u64 + 1;
            buffer.reserve((ids.len() + 1) * 4);
            for id in ids {
                buffer.extend_from_slice(&id.to_le_bytes());
            }
            buffer.extend_from_slice(&eos.to_le_bytes());
        }
        writer.write_all(&buffer)?;
        progress(&report, (n + 1) * CHUNK);
    }
    writer.flush()?;
    drop(writer);
    std::fs::rename(&temporary, out)
        .with_context(|| format!("renaming {} into place", temporary.display()))?;
    Ok(report)
}

/// Reads up to `max_bytes` of corpus text for tokenizer training, spread
/// evenly across the file list rather than taken from the front — the front
/// of a sorted list is one repository, and a vocabulary trained on one
/// repository is a vocabulary for one repository.
pub fn sample(files: &[PathBuf], max_bytes: u64) -> Vec<String> {
    if files.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut taken = 0u64;
    // A stride that visits the whole list, coprime-ish with its length so
    // consecutive picks are not from the same directory.
    let stride = (files.len() / 997).max(1);
    for start in 0..stride {
        let mut i = start;
        while i < files.len() {
            if taken >= max_bytes {
                return out;
            }
            if let Some(text) = corpus::read_document(&files[i]) {
                taken += text.len() as u64;
                out.push(text);
            }
            i += stride;
        }
    }
    out
}

/// The packed token stream, memory-mapped for training.
pub struct Tokens {
    map: memmap2::Mmap,
}

impl Tokens {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        // Safety: the packed file is written once by this tool and only
        // read afterwards; nothing truncates it while training runs.
        let map = unsafe { memmap2::Mmap::map(&file) }
            .with_context(|| format!("mapping {}", path.display()))?;
        Ok(Tokens { map })
    }

    pub fn len(&self) -> usize {
        self.map.len() / 4
    }

    /// The `count` ids starting at `offset`.
    pub fn window(&self, offset: usize, count: usize) -> Vec<u32> {
        let bytes = &self.map[offset * 4..(offset + count) * 4];
        bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab;

    /// An interrupted pack must not leave something a later run mistakes
    /// for a finished corpus.
    #[test]
    fn packing_lands_atomically() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.rs"), "fn main() {}").unwrap();
        let files = vec![dir.path().join("a.rs")];
        let vocab =
            vocab::train(std::iter::once("fn main() {}".to_string()), 300, &|_, _| {}).unwrap();
        let encoder = vocab.encoder().unwrap();
        let out = dir.path().join("tokens.bin");

        pack(&files, &encoder, &out, &|_, _| {
            // Mid-pack, the destination does not exist yet.
            assert!(
                !out.exists(),
                "a partial pack was visible at its final name"
            );
        })
        .unwrap();
        assert!(out.exists());
        assert!(!out.with_extension("tmp").exists());
    }

    #[test]
    fn packs_deduplicates_and_terminates_each_document() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.rs"), "fn main() {}").unwrap();
        std::fs::write(dir.path().join("b.rs"), "fn main() {}").unwrap();
        std::fs::write(dir.path().join("c.rs"), "let x = 1;").unwrap();

        let files = vec![
            dir.path().join("a.rs"),
            dir.path().join("b.rs"),
            dir.path().join("c.rs"),
        ];
        let vocab = vocab::train(
            files
                .iter()
                .filter_map(|f| corpus::read_document(f))
                .collect::<Vec<_>>()
                .into_iter(),
            400,
            &|_, _| {},
        )
        .unwrap();
        let encoder = vocab.encoder().unwrap();

        let out = dir.path().join("tokens.bin");
        let report = pack(&files, &encoder, &out, &|_, _| {}).unwrap();
        assert_eq!(report.documents, 2, "the identical file is packed once");
        assert_eq!(report.duplicates, 1);

        let tokens = Tokens::open(&out).unwrap();
        assert_eq!(tokens.len() as u64, report.tokens);
        let all = tokens.window(0, tokens.len());
        assert_eq!(*all.last().unwrap(), encoder.eos());
        assert_eq!(
            all.iter().filter(|&&t| t == encoder.eos()).count(),
            2,
            "one end-of-document token per document"
        );
    }
}
