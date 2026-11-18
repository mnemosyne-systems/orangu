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

//! The KL divergence of an encoding of a model from its source, measured
//! here: the model reads a text on the cores, once as the source and once
//! per encoding, and every position's next-token distribution is compared.
//!
//! The encodings are held in memory (`LoadedModel::replace_tensor`), so
//! nothing is written to measure one. The text is cut into chunks read from
//! the start with a fresh context, the BOS token first where the model takes
//! one, as an importance matrix is measured.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use half::f16;
use orangu::engine::{
    backend::{Backend, CpuBackend},
    loader::LoadedModel,
    tokenizer::Tokenizer,
};

/// Tokens in one chunk.
pub const CHUNK: usize = 512;

/// The text the model reads: whole chunks of it.
pub struct Text {
    chunks: Vec<Vec<u32>>,
}

impl Text {
    /// The last `count` whole chunks of `path`, tokenized for the model at
    /// `model`: the end of a calibration text, which an importance matrix
    /// measured on the whole text has seen, so that the chunks read here
    /// are the same for every encoding compared.
    pub fn open(model: &Path, path: &Path, count: usize) -> Result<Self> {
        let body =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let gguf = orangu::gguf::GgufFile::open(model)?;
        let tokenizer = Tokenizer::from_gguf(&gguf).context("building the tokenizer")?;
        let add_bos = tokenizer.wants_bos();
        let tokens = tokenizer.encode(&body, add_bos);
        let available = tokens.len() / CHUNK;
        if available == 0 {
            bail!(
                "{} is {} tokens, less than one {CHUNK}-token chunk",
                path.display(),
                tokens.len()
            );
        }
        let skip = available.saturating_sub(count);
        let chunks = tokens
            .as_chunks::<CHUNK>()
            .0
            .iter()
            .skip(skip)
            .map(|c| {
                let mut c = c.to_vec();
                if add_bos && let Some(bos) = tokenizer.bos_token {
                    c[0] = bos;
                }
                c
            })
            .collect();
        Ok(Self { chunks })
    }

    pub fn positions(&self) -> usize {
        self.chunks.len() * CHUNK
    }
}

/// Every position's log-probabilities under the source model.
pub struct Reference {
    rows: Vec<Vec<f16>>,
}

/// Every position's logits of `loaded` reading `text`, handed to `each`.
fn read(loaded: &LoadedModel, text: &Text, mut each: impl FnMut(usize, &[f32])) -> Result<()> {
    let backend: Arc<dyn Backend> = Arc::new(CpuBackend);
    let model = orangu::engine::build::model(loaded, &backend)?;
    let mut at = 0;
    for chunk in &text.chunks {
        let mut cache = model.new_kv_cache(CHUNK);
        match model.forward_all_logits(&mut cache, chunk, 0, 0) {
            Ok(rows) => {
                for row in rows {
                    each(at, &row);
                    at += 1;
                }
            }
            // An architecture without the multi-position pass reads the
            // chunk a token at a time.
            Err(_) => {
                let mut cache = model.new_kv_cache(CHUNK);
                for (pos, &token) in chunk.iter().enumerate() {
                    let row = model.forward(&mut cache, &[token], pos, 0)?;
                    each(at, &row);
                    at += 1;
                }
            }
        }
    }
    Ok(())
}

fn log_softmax(row: &[f32]) -> Vec<f64> {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = row.iter().map(|&v| (v as f64 - max).exp()).sum();
    let log_sum = max + sum.ln();
    row.iter().map(|&v| v as f64 - log_sum).collect()
}

/// The source's distributions over `text`.
pub fn reference(loaded: &LoadedModel, text: &Text) -> Result<Reference> {
    let mut rows = vec![Vec::new(); text.positions()];
    read(loaded, text, |at, row| {
        rows[at] = log_softmax(row).into_iter().map(f16::from_f64).collect();
    })?;
    Ok(Reference { rows })
}

/// The mean KL divergence of `loaded`'s distributions over `text` from the
/// source's, in nats.
pub fn kl(loaded: &LoadedModel, text: &Text, reference: &Reference) -> Result<f64> {
    let mut total = 0.0;
    let mut n = 0usize;
    read(loaded, text, |at, row| {
        let q = log_softmax(row);
        let p = &reference.rows[at];
        let mut sum = 0.0;
        for (lp, lq) in p.iter().zip(&q) {
            let lp = lp.to_f64();
            if lp > -30.0 {
                sum += lp.exp() * (lp - lq);
            }
        }
        total += sum.max(0.0);
        n += 1;
    })?;
    Ok(total / n.max(1) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_probabilities_sum_to_one() {
        let lp = log_softmax(&[1.0, 2.0, 3.0, -50.0]);
        let total: f64 = lp.iter().map(|v| v.exp()).sum();
        assert!((total - 1.0).abs() < 1e-12);
        assert!(lp[2] > lp[1] && lp[1] > lp[0]);
    }
}
