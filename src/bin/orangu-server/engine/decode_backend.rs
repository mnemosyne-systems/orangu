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
//! leaves the cores to the prompts and to everything else.
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

/// The most warm-up steps taken while waiting for a device's clock to
/// settle — see [`seconds_per_token_unless`]. Bounded so a backend whose
/// step time never stops falling cannot hold up the server's start.
const WARMUP_CAP: usize = 24;

/// How much faster a warm-up step must be than the best one so far to count
/// as "still ramping". Two consecutive steps that do not beat the best by
/// this much end the warm-up.
const SETTLE: f64 = 0.02;

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
    let cap = warmup_steps.max(WARMUP_CAP);
    let mut cache = model.new_kv_cache(cap + timed + 1);
    // A token every vocabulary has; its value does not change the cost.
    let token = 1u32.min(model.config().n_vocab.saturating_sub(1) as u32);
    // **Warm until the step time stops falling, not for a fixed count.** A
    // discrete card idles at a low clock and ramps under the probe's own
    // load, so a fixed warm-up times whatever the governor happened to have
    // reached: on one machine this probe read 49.4 ms and 55.8 ms for the
    // same model on consecutive starts of the same binary, a 13% spread
    // against cores that read 45.7 and 47.0 — and since the decision turns
    // on [`MIN_GAIN`], which sits inside that spread, the same machine
    // placed the same model differently from one start to the next.
    //
    // Two steps that fail to beat the best seen by [`SETTLE`] end it. The
    // floor is the caller's `warmup`, so a backend that is already warm
    // pays nothing extra, and [`WARMUP_CAP`] bounds the case where the
    // times never settle.
    let mut best = f64::INFINITY;
    let mut steady = 0usize;
    let mut pos = 0usize;
    while pos < cap {
        let started = Instant::now();
        model.forward(&mut cache, &[token], pos, 0).ok()?;
        let took = started.elapsed().as_secs_f64();
        pos += 1;
        if pos >= warmup_steps {
            if took < best * (1.0 - SETTLE) {
                steady = 0;
            } else {
                steady += 1;
                if steady >= 2 {
                    break;
                }
            }
        }
        best = best.min(took);
    }
    let mut times = Vec::with_capacity(timed);
    for pos in pos..pos + timed {
        let started = Instant::now();
        model.forward(&mut cache, &[token], pos, 0).ok()?;
        times.push(started.elapsed().as_secs_f64());
        if times.iter().filter(|&&t| t > give_up_above).count() >= 2 {
            break;
        }
    }
    times.sort_by(f64::total_cmp);
    Some(times[times.len() / 2])
}

/// [`MIN_GAIN`], or `ORANGU_DECODE_PROBE_MIN_GAIN` for a measurement — a
/// value below 1 makes a slower CPU "win", which is how the switch itself
/// is exercised on a machine whose device is the faster.
pub fn min_gain() -> f64 {
    crate::engine::backend::tuning_profile::env_var("ORANGU_DECODE_PROBE_MIN_GAIN")
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

    /// The warm-up ends when two consecutive steps stop beating the best
    /// seen, and not before the caller's floor — the rule that keeps a
    /// ramping clock out of the timed window. Asserted on the rule itself
    /// rather than through a model, because there is no backend in a unit
    /// test whose clock ramps.
    ///
    /// The case this exists for, measured: the same model on consecutive
    /// starts of the same binary read 49.4 ms and 55.8 ms on the device
    /// against 45.7 and 47.0 on the cores, so `MIN_GAIN` fell inside the
    /// spread and the placement flipped between starts.
    #[test]
    fn the_warm_up_runs_until_the_step_time_stops_falling() {
        // A ramp that settles: each step beats the last until step 5.
        let ramp = [100.0, 80.0, 64.0, 55.0, 54.5, 54.4, 54.4, 54.4];
        let settled_at = |times: &[f64], warmup: usize| {
            let (mut best, mut steady) = (f64::INFINITY, 0usize);
            for (i, &t) in times.iter().enumerate() {
                let pos = i + 1;
                if pos >= warmup {
                    if t < best * (1.0 - SETTLE) {
                        steady = 0;
                    } else {
                        steady += 1;
                        if steady >= 2 {
                            return pos;
                        }
                    }
                }
                best = best.min(t);
            }
            times.len()
        };
        // 54.5 is within 2% of 55.0 and 54.4 of 54.5: two steady steps.
        assert_eq!(settled_at(&ramp, WARMUP), 6);
        // A backend already warm pays only the caller's floor plus the two
        // steps it takes to observe that nothing is improving.
        let flat = [50.0; 8];
        assert_eq!(settled_at(&flat, WARMUP), WARMUP + 1);
        // One that never settles is bounded by the cap, not by this rule.
        let falling: Vec<f64> = (0..WARMUP_CAP)
            .map(|i| 100.0 * 0.5f64.powi(i as i32))
            .collect();
        assert_eq!(settled_at(&falling, WARMUP), falling.len());
    }

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
