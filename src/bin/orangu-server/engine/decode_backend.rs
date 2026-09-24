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

//! Where a token is decoded, measured rather than assumed — the decode half
//! of what `engine::prefill_backend` does for prompts.
//!
//! `backend = auto` takes a GPU whenever there is one. That is right on a
//! machine whose GPU out-decodes its cores, and wrong on one with a weak
//! integrated GPU beside fast cores — and until this was measured nothing
//! told the two apart. So under `auto`, once the model is built on the
//! device, it is also built on the CPU backend (a view of the same mapped
//! weights, not a copy) and a few real one-token decode steps are timed on
//! each: the model's own forward pass, at the width a served token takes.
//! The device keeps the model unless the cores are at least
//! [`MIN_GAIN`] faster — at parity the device is the better place, since it
//! leaves the cores to the prompts and to everything else (`doc/PERF-ALL.md`,
//! task 7).
//!
//! `ORANGU_DECODE_PROBE=0` skips the measurement and keeps the device, as
//! before; an explicit `backend` is never second-guessed.

use std::sync::Arc;
use std::time::Instant;

use crate::engine::arch::ModelForward;

/// How much faster a token must be on the cores for decode to leave the
/// device.
pub const MIN_GAIN: f64 = 1.10;

/// Steps timed per backend, after [`WARMUP`] untimed ones (a device's first
/// steps build pipelines and lift its clock).
const STEPS: usize = 8;
const WARMUP: usize = 4;

/// Whether the probe may run — `ORANGU_DECODE_PROBE=0` turns it off.
pub fn enabled() -> bool {
    crate::engine::env::flag_on_unless_disabled("ORANGU_DECODE_PROBE")
}

/// Seconds per one-token decode step of `model`: the median of [`STEPS`]
/// consecutive steps of one sequence, after [`WARMUP`] — each a real
/// `forward` of one token at the next position, as a served token is.
/// `None` when the model refuses a step.
pub fn seconds_per_token(model: &Arc<dyn ModelForward>) -> Option<f64> {
    seconds_per_token_after(model, WARMUP, STEPS)
}

/// [`seconds_per_token`] with its own counts: the median of `steps` timed
/// steps after `warmup` untimed ones — a longer warm-up for a measurement
/// that has to let a device's clock governor settle at the duty cycle
/// decode gives it (`engine::head_split`).
pub fn seconds_per_token_after(
    model: &Arc<dyn ModelForward>,
    warmup: usize,
    steps: usize,
) -> Option<f64> {
    seconds_per_token_unless(model, warmup, steps, f64::INFINITY)
}

/// [`seconds_per_token_after`] that stops early once two timed steps have
/// each taken longer than `give_up_above` seconds — the answer to "is this
/// faster?" is already no — and returns the median of the steps it timed.
pub fn seconds_per_token_unless(
    model: &Arc<dyn ModelForward>,
    warmup: usize,
    steps: usize,
    give_up_above: f64,
) -> Option<f64> {
    let (warmup_steps, timed) = (warmup, steps.max(1));
    let mut cache = model.new_kv_cache(warmup_steps + timed + 1);
    // A token every vocabulary has; its value does not change the cost.
    let token = 1u32.min(model.config().n_vocab.saturating_sub(1) as u32);
    let mut times = Vec::with_capacity(timed);
    for pos in 0..warmup_steps + timed {
        let started = Instant::now();
        model.forward(&mut cache, &[token], pos, 0).ok()?;
        if pos >= warmup_steps {
            times.push(started.elapsed().as_secs_f64());
            if times.iter().filter(|&&t| t > give_up_above).count() >= 2 {
                break;
            }
        }
    }
    times.sort_by(f64::total_cmp);
    Some(times[times.len() / 2])
}

/// [`MIN_GAIN`], or `ORANGU_DECODE_PROBE_MIN_GAIN` for a measurement — a
/// value below 1 makes a slower CPU "win", which is how the switch itself
/// is exercised on a machine whose device is the faster.
pub fn min_gain() -> f64 {
    std::env::var("ORANGU_DECODE_PROBE_MIN_GAIN")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|g| g.is_finite() && *g > 0.0)
        .unwrap_or(MIN_GAIN)
}

/// The measurement's verdict: `true` when the cores should decode.
pub fn cpu_wins(device: f64, cpu: f64) -> bool {
    cpu_wins_by(device, cpu, min_gain())
}

fn cpu_wins_by(device: f64, cpu: f64, gain: f64) -> bool {
    device > cpu * gain
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_device_keeps_decode_unless_the_cores_are_clearly_faster() {
        // 70 ms on the device against 80 on the cores: the device.
        assert!(!cpu_wins_by(0.070, 0.080, MIN_GAIN));
        // Level, and within the margin: still the device.
        assert!(!cpu_wins_by(0.070, 0.070, MIN_GAIN));
        assert!(!cpu_wins_by(0.075, 0.070, MIN_GAIN));
        // A weak integrated GPU at 120 ms against 80 on the cores.
        assert!(cpu_wins_by(0.120, 0.080, MIN_GAIN));
    }
}
