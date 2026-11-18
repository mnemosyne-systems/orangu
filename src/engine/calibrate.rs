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

//! Measuring an importance matrix: a calibration text is run through a model
//! on the host, chunk by chunk, and what `engine::importance` collected is
//! written in the GGUF importance-matrix format.
//!
//! The text is tokenized once and cut into fixed chunks; each chunk runs
//! from position 0 in a fresh cache, with the model's BOS token in its first
//! place when the model takes one, and a leftover shorter than a chunk is
//! not run.

use std::{collections::BTreeMap, path::Path, sync::Arc, time::Instant};

use crate::gguf::{GgufFile, GgufValue};
use crate::gguf_write::{self, TensorPlan};
use anyhow::{Context, Result, bail};

use crate::engine::{
    self,
    backend::{Backend, CpuBackend},
    importance::{self, Entry, Recording},
    loader::LoadedModel,
    tokenizer::Tokenizer,
};

/// Measures `model_path`'s importance matrix over `text` in chunks of
/// `chunk` tokens (at most `chunks` of them) and writes it to `output`.
///
/// Each chunk is measured from nothing and the chunks' accumulators are
/// added up in chunk order ([`fold`]), so the matrix does not depend on
/// where or in what order the chunks ran.
pub fn run(
    model_path: &Path,
    text: &Path,
    output: &Path,
    chunk: usize,
    chunks: Option<usize>,
) -> Result<()> {
    let parts = text_chunks(model_path, text, chunk, chunks)?;
    let measurer = Measurer::open(model_path, chunk)?;
    let total = parts.len();
    let started = Instant::now();
    let mut stats = BTreeMap::new();
    for (i, part) in parts.iter().enumerate() {
        fold(
            &mut stats,
            measurer
                .measure(part)
                .with_context(|| format!("running chunk {}", i + 1))?,
        );
        let elapsed = started.elapsed().as_secs_f64();
        let left = elapsed / (i + 1) as f64 * (total - i - 1) as f64;
        eprintln!(
            "chunk {}/{total}: {:.1} s, about {:.0} s left",
            i + 1,
            elapsed / (i + 1) as f64,
            left
        );
    }
    finish(output, &stats, text, total, chunk)
}

/// The whole `chunk`-token chunks of `text` as `model_path` tokenizes it,
/// at most `chunks` of them, each starting with the model's BOS where it
/// takes one.
pub fn text_chunks(
    model_path: &Path,
    text: &Path,
    chunk: usize,
    chunks: Option<usize>,
) -> Result<Vec<Vec<u32>>> {
    if chunk < 2 {
        bail!("a chunk is at least 2 tokens");
    }
    let body =
        std::fs::read_to_string(text).with_context(|| format!("reading {}", text.display()))?;
    let gguf =
        GgufFile::open(model_path).with_context(|| format!("reading {}", model_path.display()))?;
    let tokenizer = Tokenizer::from_gguf(&gguf).context("building the tokenizer")?;
    let add_bos = tokenizer.wants_bos();
    let tokens = tokenizer.encode(&body, add_bos);
    let available = tokens.len() / chunk;
    if available == 0 {
        bail!(
            "{} is {} tokens, less than one {chunk}-token chunk",
            text.display(),
            tokens.len()
        );
    }
    let total = chunks.map_or(available, |n| n.min(available));
    eprintln!(
        "{}: {} tokens, {total} chunks of {chunk}",
        text.display(),
        tokens.len()
    );
    Ok(tokens
        .chunks_exact(chunk)
        .take(total)
        .map(|part| {
            let mut part = part.to_vec();
            if add_bos && let Some(bos) = tokenizer.bos_token {
                part[0] = bos;
            }
            part
        })
        .collect())
}

/// A model loaded to measure chunks of a calibration text, on the host.
pub struct Measurer {
    model: Arc<dyn engine::arch::ModelForward>,
    chunk: usize,
}

impl Measurer {
    pub fn open(model_path: &Path, chunk: usize) -> Result<Self> {
        let loaded = LoadedModel::open(model_path)
            .with_context(|| format!("loading {}", model_path.display()))?;
        importance::enable(&loaded);
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend);
        let model = crate::engine::build::model(&loaded, &backend)?;
        // Whatever an earlier model of this process left is not this one's.
        importance::take();
        Ok(Self { model, chunk })
    }

    /// One chunk's accumulators, measured from nothing in a fresh cache.
    pub fn measure(&self, part: &[u32]) -> Result<BTreeMap<String, Entry>> {
        let mut cache = engine::arch::ModelForward::new_kv_cache(self.model.as_ref(), self.chunk);
        {
            let _recording = Recording::start();
            self.model.forward(&mut cache, part, 0, 0)?;
        }
        Ok(importance::take())
    }
}

/// Adds one chunk's accumulators into `total`. Folding chunks in the same
/// order gives the same bits wherever each chunk was measured.
pub fn fold(total: &mut BTreeMap<String, Entry>, chunk: BTreeMap<String, Entry>) {
    for (name, entry) in chunk {
        match total.get_mut(&name) {
            Some(sum) if sum.columns == entry.columns && sum.counts.len() == entry.counts.len() => {
                for (a, b) in sum.sums.iter_mut().zip(&entry.sums) {
                    *a += b;
                }
                for (a, b) in sum.counts.iter_mut().zip(&entry.counts) {
                    *a += b;
                }
            }
            Some(_) => {}
            None => {
                total.insert(name, entry);
            }
        }
    }
}

/// Reports what `stats` covers and writes it to `output`.
pub fn finish(
    output: &Path,
    stats: &BTreeMap<String, Entry>,
    text: &Path,
    chunks: usize,
    chunk: usize,
) -> Result<()> {
    let (_, unattributed) = importance::snapshot();
    let expected = importance::collectable();
    let missing: Vec<&String> = expected
        .iter()
        .filter(|n| !stats.contains_key(*n))
        .collect();
    eprintln!(
        "collected {} of {} weight matrices",
        stats.len(),
        expected.len()
    );
    if !missing.is_empty() {
        eprintln!("not reached by a matmul: {}", list(&missing));
    }
    for ((in_dim, out_dim), calls) in &unattributed {
        eprintln!("unnamed weights {in_dim}×{out_dim}: {calls} matmuls");
    }
    if stats.is_empty() {
        bail!("no matmul was collected");
    }
    let dataset = text.file_name().map_or_else(
        || text.display().to_string(),
        |n| n.to_string_lossy().into_owned(),
    );
    write(output, stats, &dataset, chunks as u32, chunk as u32)
}

fn list(names: &[&String]) -> String {
    const SHOWN: usize = 12;
    let mut out = names
        .iter()
        .take(SHOWN)
        .map(|s| s.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    if names.len() > SHOWN {
        out.push_str(&format!(" and {} more", names.len() - SHOWN));
    }
    out
}

/// The counts written for `name`: one per expert for a routed tensor
/// (`*_exps`), whose experts each see their own tokens; one for any other,
/// whose matrices (an attention tensor's heads, say) all see every row.
fn counts(name: &str, entry: &Entry) -> Vec<u64> {
    let first = entry.counts.first().copied().unwrap_or(0);
    if !name.contains("_exps") && entry.counts.iter().all(|&c| c == first) {
        vec![first]
    } else {
        entry.counts.clone()
    }
}

/// Writes `stats` as a GGUF importance matrix: per tensor
/// `<name>.in_sum2` (`columns × matrices`) and `<name>.counts`
/// (`1 × matrices`, or `1` where [`counts`] gives one), in name order.
pub fn write(
    path: &Path,
    stats: &std::collections::BTreeMap<String, Entry>,
    dataset: &str,
    chunks: u32,
    chunk_size: u32,
) -> Result<()> {
    let metadata = vec![
        (
            "general.type".to_string(),
            GgufValue::String("imatrix".into()),
        ),
        (
            "imatrix.datasets".to_string(),
            GgufValue::Array(vec![GgufValue::String(dataset.into())]),
        ),
        ("imatrix.chunk_count".to_string(), GgufValue::U32(chunks)),
        ("imatrix.chunk_size".to_string(), GgufValue::U32(chunk_size)),
    ];
    let mut plans = Vec::with_capacity(stats.len() * 2);
    for (name, entry) in stats {
        let matrices = entry.counts.len() as u64;
        let counted = counts(name, entry).len() as u64;
        plans.push(TensorPlan {
            name: format!("{name}.in_sum2"),
            dims: if counted == 1 {
                vec![entry.columns as u64 * matrices]
            } else {
                vec![entry.columns as u64, matrices]
            },
            ggml_type: crate::quantize::GGML_TYPE_F32,
        });
        plans.push(TensorPlan {
            name: format!("{name}.counts"),
            dims: if counted == 1 {
                vec![1]
            } else {
                vec![1, counted]
            },
            ggml_type: crate::quantize::GGML_TYPE_F32,
        });
    }
    gguf_write::write(
        path,
        &metadata,
        &plans,
        |plan| {
            if let Some(name) = plan.name.strip_suffix(".in_sum2") {
                Ok(stats[name].sums.iter().map(|&s| s as f32).collect())
            } else {
                let name = plan.name.strip_suffix(".counts").unwrap_or(&plan.name);
                Ok(counts(name, &stats[name])
                    .iter()
                    .map(|&c| c as f32)
                    .collect())
            }
        },
        &|_, _, _| {},
    )
    .with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}
