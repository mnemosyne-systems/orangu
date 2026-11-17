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

//! How fast a node runs its layers.
//!
//! The nodes of a tree may be anything — a GPU, cores of one kind or
//! another, prompts on an NPU — so a node times the model's first layers
//! itself, at start, through the path its layers take in a tree
//! (`ModelForward::forward_layers` on its own backend): one decode step and
//! one 128-token prompt chunk, over two ranges so that what a forward costs
//! whatever its layers cancels out. Each is reported as the gigabytes of layer
//! weights it gets through a second, which a parent divides a range by
//! (`super::plan::by_speed`), since a layer's cost follows its bytes.

use super::plan;
use crate::engine::arch::ModelForward;
use anyhow::Result;
use std::ops::Range;
use std::time::{Duration, Instant};

/// The prompt chunk timed: the parts a tree's prompts travel in.
pub const PROMPT_TOKENS: usize = 128;

/// Decode steps timed, after two that are not.
const DECODE_STEPS: usize = 16;

/// Times each range is timed, the two ranges in turn: the fastest of them is
/// what counts, so a device still clocking up, or another process's burst,
/// does not.
const ROUNDS: usize = 3;

/// The least part of the longer range's time the layers between the cuts
/// must take for the difference to rate them. Below it the difference is
/// noise divided by little (117 GB/s once, from a GPU that does 8), and the
/// longer range is rated whole instead.
const MIN_SHARE: f64 = 0.15;

/// A node's measured speed: gigabytes of layer weights a second.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Speed {
    pub decode: f32,
    pub prompt: f32,
    /// What a decode step and a prompt chunk cost whatever their layers, in
    /// milliseconds: the shorter range's time less its layers at the rate.
    /// `0` when a model is too small for two cuts to tell.
    pub decode_fixed_ms: f32,
    pub prompt_fixed_ms: f32,
}

/// The most weights the longer of the two timed ranges takes: what a node
/// reads in at start to time itself.
const MEASURE_BYTES: u64 = 1 << 30;

/// Times the model's first layers through a prompt chunk and decode steps,
/// on two ranges from layer 0 — the first cut the model allows from the
/// second layer on, and a longer one (up to a quarter of the layers and
/// [`MEASURE_BYTES`]) — and rates the layers between the two cuts by the
/// difference. A forward's fixed costs, whatever its range (Gemma 4's
/// per-layer inputs for every layer, a device's submission and readback),
/// cancel out: timing a short range alone made them the rate, and those
/// costs differ from backend to backend. A model too
/// small for two cuts is rated over the one. Each range runs once untimed
/// first: the first run reads the weights in, uploads them and builds a
/// device's pipelines. Returns the speed, the layers rated and how long it
/// all took.
pub fn measure(
    model: &dyn ModelForward,
    layer_bytes: &[u64],
) -> Result<(Speed, Range<usize>, Duration)> {
    let started = Instant::now();
    let (short, long) = cuts(model, layer_bytes);
    let fastest = |a: (Duration, Duration), b: (Duration, Duration)| (a.0.min(b.0), a.1.min(b.1));
    let slowest = (Duration::MAX, Duration::MAX);
    let (mut short_times, mut long_times) = (slowest, slowest);
    for _ in 0..ROUNDS {
        short_times = fastest(short_times, time_range(model, 0..short)?);
        if let Some(long) = long {
            long_times = fastest(long_times, time_range(model, 0..long)?);
        }
    }
    let (short_prompt, short_decode) = short_times;
    let (layers, prompt, decode, rated_apart) = match long {
        Some(long) => {
            let (long_prompt, long_decode) = long_times;
            let between = |long: Duration, short: Duration| {
                long.checked_sub(short)
                    .filter(|d| d.as_secs_f64() >= MIN_SHARE * long.as_secs_f64())
            };
            match (
                between(long_prompt, short_prompt),
                between(long_decode, short_decode),
            ) {
                (Some(p), Some(d)) => (short..long, p, d, true),
                _ => (0..long, long_prompt, long_decode, false),
            }
        }
        None => (0..short, short_prompt, short_decode, false),
    };
    let bytes = plan::bytes_of(&layers, layer_bytes) as f64;
    let rate = |d: Duration| (bytes / d.as_secs_f64().max(1e-9) / 1e9) as f32;
    // The short range at the rate, and what its time holds beyond that.
    let short_bytes = plan::bytes_of(&(0..short), layer_bytes) as f64;
    let fixed = |time: Duration, per: Duration| {
        if !rated_apart {
            return 0.0;
        }
        let layers_ms = short_bytes / bytes * per.as_secs_f64() * 1e3;
        (time.as_secs_f64() * 1e3 - layers_ms).max(0.0) as f32
    };
    let speed = Speed {
        decode: rate(decode),
        prompt: rate(prompt),
        decode_fixed_ms: fixed(short_decode, decode),
        prompt_fixed_ms: fixed(short_prompt, prompt),
    };
    Ok((speed, layers, started.elapsed()))
}

/// The ends of the two ranges [`measure`] times from layer 0: the first cut
/// the model allows from the second layer on, and the longest cut past it
/// within a quarter of the layers and [`MEASURE_BYTES`] — none when the
/// model is too small for a second.
fn cuts(model: &dyn ModelForward, layer_bytes: &[u64]) -> (usize, Option<usize>) {
    let n = model.config().n_layer;
    let cut = |at: usize| at == n || model.split_allowed(at);
    let short = (2.min(n)..=n).find(|at| cut(*at)).unwrap_or(n);
    let long = (short + 1..=(n / 4).max(short + 1).min(n))
        .rev()
        .find(|at| cut(*at) && plan::bytes_of(&(0..*at), layer_bytes) <= MEASURE_BYTES);
    (short, long)
}

/// Runs `layers` once, untimed — a 128-token prompt chunk and a decode
/// step — so that the first request does not pay for reading the weights
/// in, uploading them or building a device's pipelines: a node does it
/// when a plan gives it layers. Returns how long it took.
pub fn warm(model: &dyn ModelForward, layers: Range<usize>) -> Result<Duration> {
    let started = Instant::now();
    let mut cache = model.new_kv_cache_for_layers(layers.clone(), PROMPT_TOKENS + 1);
    let vocab = model.config().n_vocab.clamp(2, 1000) as u32;
    let tokens: Vec<u32> = (0..PROMPT_TOKENS as u32)
        .map(|i| 1 + i * 7 % (vocab - 1))
        .collect();
    let hidden = model.embed(&tokens)?;
    model.forward_layers(&mut cache, hidden, &tokens, layers.clone(), 0)?;
    let hidden = model.embed(&tokens[..1])?;
    model.forward_layers(&mut cache, hidden, &tokens[..1], layers, PROMPT_TOKENS)?;
    Ok(started.elapsed())
}

/// One 128-token prompt chunk's time and one decode step's (the fastest of
/// [`DECODE_STEPS`]) through `layers`, each after an untimed run.
fn time_range(model: &dyn ModelForward, layers: Range<usize>) -> Result<(Duration, Duration)> {
    let mut cache =
        model.new_kv_cache_for_layers(layers.clone(), 2 * PROMPT_TOKENS + 2 + DECODE_STEPS);
    let vocab = model.config().n_vocab.clamp(2, 1000) as u32;
    let tokens: Vec<u32> = (0..PROMPT_TOKENS as u32)
        .map(|i| 1 + i * 7 % (vocab - 1))
        .collect();
    let mut run = |tokens: &[u32], pos: usize| -> Result<Duration> {
        let hidden = model.embed(tokens)?;
        let at = Instant::now();
        model.forward_layers(&mut cache, hidden, tokens, layers.clone(), pos)?;
        Ok(at.elapsed())
    };
    run(&tokens, 0)?;
    let prompt = run(&tokens, PROMPT_TOKENS)?;
    let mut pos = 2 * PROMPT_TOKENS;
    for _ in 0..2 {
        run(&tokens[..1], pos)?;
        pos += 1;
    }
    let mut decode = Duration::MAX;
    for _ in 0..DECODE_STEPS {
        decode = decode.min(run(&tokens[..1], pos)?);
        pos += 1;
    }
    Ok((prompt, decode))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workers::pipeline::fixture::{self, Variant};

    fn layer_bytes(model: &dyn ModelForward) -> Vec<u64> {
        let loaded = fixture::loaded(Variant::default());
        let n = model.config().n_layer;
        let mut layer_bytes = vec![0u64; n];
        for (name, bytes) in loaded.tensor_sizes() {
            if let Some(il) = crate::engine::loader::block_index(name).filter(|il| *il < n) {
                layer_bytes[il] += bytes;
            }
        }
        layer_bytes
    }

    /// The first cut is layer 2 and the longer range ends at layer 3; with
    /// every layer past the first over the byte budget, there is no second.
    #[test]
    fn the_timed_ranges_end_at_the_first_and_the_longest_cut() {
        let model = fixture::model();
        let bytes = layer_bytes(model.as_ref());
        assert_eq!(cuts(model.as_ref(), &bytes), (2, Some(3)));
        let heavy = vec![MEASURE_BYTES; bytes.len()];
        assert_eq!(cuts(model.as_ref(), &heavy), (2, None));
    }

    /// A node rates the layers between the two cuts — or, when its clock
    /// cannot tell a layer from noise (the longer range timed no slower),
    /// the whole longer range.
    #[test]
    fn a_range_can_be_warmed() {
        let model = fixture::model();
        assert!(warm(model.as_ref(), 1..3).is_ok());
        assert!(warm(model.as_ref(), 0..model.config().n_layer).is_ok());
    }

    #[test]
    fn a_node_measures_its_first_layers() {
        let model = fixture::model();
        let (speed, layers, _) = measure(model.as_ref(), &layer_bytes(model.as_ref())).unwrap();
        assert!(layers == (2..3) || layers == (0..3), "{layers:?}");
        assert!(speed.decode > 0.0 && speed.prompt > 0.0, "{speed:?}");
    }
}
