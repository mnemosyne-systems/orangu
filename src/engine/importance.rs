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

//! Importance statistics: per input column of every weight matrix, the
//! summed square of the activations that reached it over a calibration text,
//! and how many rows were counted (`orangu-server imatrix`).
//!
//! A quantizer below about three bits uses them to choose which columns'
//! errors to accept. They are collected on the host path: the CPU backend's
//! matmul entry points and the routed-expert projection call [`record`]
//! with the weight bytes they are about to read, and the weight is named by
//! the address those bytes live at. Every tensor's address range is known
//! once the file is loaded ([`enable`]); a weight an architecture copies
//! into a buffer of its own at load (a row gather, a de-interleave) is
//! registered under its source's name when the copy is made ([`alias`]).
//!
//! For a stacked-expert tensor the statistics are per expert, over the rows
//! routed to that expert only. Sums are `f64` and every matrix's rows are
//! added in the order the forward pass produced them, so the result does not
//! depend on the thread count.
//!
//! Collection is off unless [`enable`] was called, and then only while
//! [`Recording`] is held: a matmul pays one atomic load when it is off.

use std::{
    collections::{BTreeMap, HashMap},
    sync::{
        Arc, Mutex, OnceLock, RwLock,
        atomic::{AtomicBool, Ordering},
    },
};

use crate::engine::loader::{LoadedModel, QuantMatrix};

/// One tensor's bytes and how they divide into matrices.
#[derive(Clone)]
struct Span {
    start: usize,
    end: usize,
    name: Arc<str>,
    /// Bytes per matrix: the whole tensor for a 2-D weight, one expert for
    /// a stacked one.
    stride: usize,
    /// Whether a matmul against it is collected. Only the blocks' weights
    /// are; the output head and the embedding table are not.
    collected: bool,
}

/// One tensor's accumulators.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Entry {
    /// Input columns per matrix.
    pub columns: usize,
    /// `columns × matrices` summed squares, matrix-major.
    pub sums: Vec<f64>,
    /// Rows counted, per matrix.
    pub counts: Vec<u64>,
}

struct Collector {
    spans: RwLock<Vec<Span>>,
    stats: Mutex<HashMap<Arc<str>, Entry>>,
    /// Matmuls whose weights lie in no known tensor, by `(in_dim, out_dim)`.
    unattributed: Mutex<BTreeMap<(usize, usize), u64>>,
    recording: AtomicBool,
}

static COLLECTOR: OnceLock<Collector> = OnceLock::new();

/// The tensors whose matmuls are collected: a block's 2-D or stacked
/// weight, other than a convolution kernel or a grouped norm.
fn is_collected(name: &str, dims: &[u64]) -> bool {
    name.starts_with("blk.")
        && name.ends_with(".weight")
        && !name.ends_with("_conv1d.weight")
        && !name.ends_with("_norm.weight")
        && matches!(dims.len(), 2 | 3)
}

/// Turns collection on for `model`'s tensors. Call it after the file is
/// loaded and before the architecture is built, so the copies it makes are
/// registered ([`alias`]).
pub fn enable(model: &LoadedModel) {
    let mut spans: Vec<Span> = model
        .tensor_spans()
        .filter(|&(_, _, len, _)| len > 0)
        .map(|(name, start, len, dims)| {
            let matrices = if dims.len() == 3 { dims[2] as usize } else { 1 };
            Span {
                start,
                end: start + len,
                name: Arc::from(name),
                stride: len / matrices.max(1),
                collected: is_collected(name, dims),
            }
        })
        .collect();
    spans.sort_by_key(|s| s.start);
    let collector = COLLECTOR.get_or_init(|| Collector {
        spans: RwLock::new(Vec::new()),
        stats: Mutex::new(HashMap::new()),
        unattributed: Mutex::new(BTreeMap::new()),
        recording: AtomicBool::new(false),
    });
    *collector.spans.write().unwrap() = spans;
}

/// Whether a matmul should call [`record`] now.
#[inline]
pub fn recording() -> bool {
    COLLECTOR
        .get()
        .is_some_and(|c| c.recording.load(Ordering::Relaxed))
}

/// Collection is on while this is held.
pub struct Recording(());

impl Recording {
    /// Starts recording; `None` when [`enable`] was never called.
    pub fn start() -> Option<Self> {
        let collector = COLLECTOR.get()?;
        collector.recording.store(true, Ordering::Relaxed);
        Some(Recording(()))
    }
}

impl Drop for Recording {
    fn drop(&mut self) {
        if let Some(collector) = COLLECTOR.get() {
            collector.recording.store(false, Ordering::Relaxed);
        }
    }
}

/// Registers `copy` (`len` bytes from `copy_start`) under the name of the
/// tensor holding `source`. The copy keeps its source's matrix stride, so a
/// copy of a whole stacked tensor still divides per expert.
pub fn alias(source: usize, copy_start: usize, len: usize) {
    let Some(collector) = COLLECTOR.get() else {
        return;
    };
    let mut spans = collector.spans.write().unwrap();
    let Some(found) = find(&spans, source).cloned() else {
        return;
    };
    let span = Span {
        start: copy_start,
        end: copy_start + len,
        stride: found.stride.min(len.max(1)),
        ..found
    };
    let at = spans.partition_point(|s| s.start < copy_start);
    spans.insert(at, span);
}

fn find(spans: &[Span], address: usize) -> Option<&Span> {
    let at = spans.partition_point(|s| s.start <= address);
    let span = spans.get(at.checked_sub(1)?)?;
    (address < span.end).then_some(span)
}

/// A dense matmul: `n_tokens` rows of `w.in_dim` inputs in `x`.
pub fn record_matmul(w: &QuantMatrix, x: &[f32], n_tokens: usize) {
    let in_dim = w.in_dim;
    if in_dim == 0 || w.out_dim == 0 {
        return;
    }
    let rows = x.chunks_exact(in_dim).take(n_tokens);
    record(w.raw_bytes().as_ptr() as usize, in_dim, w.out_dim, rows);
}

/// The weights starting at `address` multiplied `rows`, each `in_dim`
/// inputs wide.
pub fn record<'a>(
    address: usize,
    in_dim: usize,
    out_dim: usize,
    rows: impl IntoIterator<Item = &'a [f32]>,
) {
    let Some(collector) = COLLECTOR.get() else {
        return;
    };
    let span = find(&collector.spans.read().unwrap(), address).cloned();
    let Some(span) = span else {
        *collector
            .unattributed
            .lock()
            .unwrap()
            .entry((in_dim, out_dim))
            .or_default() += 1;
        return;
    };
    if !span.collected {
        return;
    }
    let matrices = (span.end - span.start) / span.stride.max(1);
    let matrix = (address - span.start) / span.stride.max(1);
    let mut stats = collector.stats.lock().unwrap();
    let entry = stats.entry(span.name.clone()).or_insert_with(|| Entry {
        columns: in_dim,
        sums: vec![0.0; in_dim * matrices],
        counts: vec![0; matrices],
    });
    if entry.columns != in_dim || matrix >= entry.counts.len() {
        drop(stats);
        *collector
            .unattributed
            .lock()
            .unwrap()
            .entry((in_dim, out_dim))
            .or_default() += 1;
        return;
    }
    let sums = &mut entry.sums[matrix * in_dim..(matrix + 1) * in_dim];
    let mut counted = 0u64;
    for row in rows {
        for (sum, &v) in sums.iter_mut().zip(row) {
            *sum += f64::from(v) * f64::from(v);
        }
        counted += 1;
    }
    entry.counts[matrix] += counted;
}

/// What was collected: per tensor name, its accumulators; and the shapes of
/// matmuls whose weights were not found, with how often each ran.
pub fn snapshot() -> (BTreeMap<String, Entry>, BTreeMap<(usize, usize), u64>) {
    let Some(collector) = COLLECTOR.get() else {
        return Default::default();
    };
    let stats = collector
        .stats
        .lock()
        .unwrap()
        .iter()
        .map(|(name, entry)| (name.to_string(), entry.clone()))
        .collect();
    let unattributed = collector.unattributed.lock().unwrap().clone();
    (stats, unattributed)
}

/// What was collected since the last take, which starts the next from
/// nothing: one chunk's accumulators, measured alone.
pub fn take() -> BTreeMap<String, Entry> {
    let Some(collector) = COLLECTOR.get() else {
        return BTreeMap::new();
    };
    collector
        .stats
        .lock()
        .unwrap()
        .drain()
        .map(|(name, entry)| (name.to_string(), entry))
        .collect()
}

/// The names of every weight a matmul would be collected for.
pub fn collectable() -> Vec<String> {
    let Some(collector) = COLLECTOR.get() else {
        return Vec::new();
    };
    let mut names: Vec<String> = collector
        .spans
        .read()
        .unwrap()
        .iter()
        .filter(|s| s.collected)
        .map(|s| s.name.to_string())
        .collect();
    names.sort();
    names.dedup();
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(start: usize, end: usize, name: &str, stride: usize) -> Span {
        Span {
            start,
            end,
            name: Arc::from(name),
            stride,
            collected: true,
        }
    }

    #[test]
    fn an_address_finds_the_tensor_holding_it() {
        let spans = vec![span(100, 200, "a", 100), span(200, 260, "b", 20)];
        assert!(find(&spans, 99).is_none());
        assert_eq!(&*find(&spans, 100).unwrap().name, "a");
        assert_eq!(&*find(&spans, 199).unwrap().name, "a");
        assert_eq!(&*find(&spans, 240).unwrap().name, "b");
        assert!(find(&spans, 260).is_none());
    }

    #[test]
    fn the_collected_weights_are_the_blocks_matrices() {
        assert!(is_collected("blk.0.attn_q.weight", &[64, 64]));
        assert!(is_collected("blk.3.ffn_up_exps.weight", &[64, 32, 8]));
        assert!(!is_collected("blk.0.attn_norm.weight", &[64]));
        assert!(!is_collected("blk.0.ssm_conv1d.weight", &[4, 64]));
        assert!(!is_collected("blk.0.ssm_norm.weight", &[64, 8]));
        assert!(!is_collected("output.weight", &[64, 100]));
        assert!(!is_collected("token_embd.weight", &[64, 100]));
    }
}
