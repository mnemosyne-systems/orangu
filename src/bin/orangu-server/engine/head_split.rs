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

//! The output head shared between the cores and an idle GPU, where the two
//! together read memory faster than the cores alone.
//!
//! The head is the one matmul of a decode step that is both large and
//! alone: `gemma-4-E2B`'s is the tied 262 144 × 1536 `Q4_K` embedding,
//! 226 MB read once per token. When decode runs on the cores
//! (`engine::decode_backend`) it reads those bytes at the rate the cores
//! can pull from memory — on the CIX P1, 35 GB/s against a measured
//! ceiling of ~40 for the whole CPU cluster — so no CPU kernel makes it
//! faster. But that ceiling is the cluster's, not the memory's: with the
//! Mali decoding at the same time the cores still read at 37–38 GB/s, so
//! the two together reach ~60 (`doc/PERF-ALL.md`, task 14).
//!
//! So when decode left a GPU idle, a fresh backend on that GPU is offered
//! the head's last rows, and each token's head runs on both at once: the
//! device part on a thread of its own, the cores' part in the request's
//! pool, the two halves concatenated. Two measurements decide it at
//! start-up, both on the model's own head:
//!
//! 1. **The share** — the head alone, back to back, on the cores and then
//!    split at each of [`SHARES`]; the fastest share is the candidate.
//! 2. **The verdict** — whole one-token decode steps with the candidate
//!    split and without it (the latter usually `engine::decode_backend`'s
//!    own measurement, just taken). The split is kept only when it makes a
//!    *step* [`MIN_GAIN`] faster; its steps stop being timed as soon as two
//!    are [`GIVE_UP`] slower.
//!
//! The second is the one that matters, because a GPU's clock is a governor's
//! decision about the work it has been seeing. Back to back, the head keeps
//! the Mali busy and its governor (`simple_ondemand`, sampling every
//! 100 ms) clocks it to 1 GHz: the head took 4.8 ms split, 8.3 on the cores.
//! In a decode step the device's part is a few milliseconds in sixty, the
//! governor holds it at 72 MHz, and the same split made the head 30 ms and
//! decode 17 → 10 tok/s. Only a measurement with decode's own duty cycle
//! sees that — and on a machine whose GPU stays clocked it would see the
//! gain instead.
//!
//! When the split is kept the device holds only the head's rows (at most
//! [`SHARES`]' largest share of them), never the rest of the model; when it
//! is not, the device backend is dropped again.
//!
//! The weights are the same bytes; the device's rows are dotted against its
//! own `int8` quantization of the activation, as a GPU decode always was,
//! so their logits can differ from the cores' in the last bits.
//!
//! `ORANGU_HEAD_SPLIT=0` keeps the head on the cores and skips the
//! measurement; `ORANGU_HEAD_SPLIT_SHARE=<0..1>` fixes the device's share
//! and keeps the split whatever the verdict — for measuring it.

use std::sync::{Arc, RwLock};
use std::time::Instant;

use crate::engine::arch::ModelForward;
use crate::engine::backend::Backend;
use crate::engine::loader::QuantMatrix;

/// How much faster a decode step must be with the split to keep it: the
/// device thread and its submission are overhead a near-tie does not repay.
const MIN_GAIN: f64 = 1.05;

/// The device shares tried, largest first — the first upload is the largest
/// share's rows, and every smaller share's rows lie inside it, so the device
/// binds them there rather than uploading again.
const SHARES: [f64; 6] = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25];

/// Row counts are kept to a multiple of this, so a share's first row sits on
/// the device's buffer-offset alignment inside the largest share's upload.
const ROW_ALIGN: usize = 256;

/// Untimed decode steps before the split's arm of the verdict: long enough
/// (~0.35 s at this board's ~55 ms a step) for a clock governor sampling
/// every 100 ms to see a few windows of decode's duty cycle.
const STEP_WARMUP: usize = 6;

/// The split's steps stop being timed once two of them are this much slower
/// than a step without it: the verdict is already no.
const GIVE_UP: f64 = 1.25;

/// Timed decode steps per arm of the verdict.
const STEP_TIMED: usize = 8;

struct Split {
    device: Arc<dyn Backend>,
    /// The head this split was measured on, by [`QuantMatrix::cache_key`].
    head: (usize, usize),
    /// Rows `0..cpu_rows` on the cores, the rest on the device.
    cpu_rows: usize,
}

/// The installed split, if any — set while the verdict measures it, and
/// kept or cleared by the verdict.
static SPLIT: RwLock<Option<Arc<Split>>> = RwLock::new(None);

/// Whether `ORANGU_HEAD_SPLIT` leaves this on. Default on.
pub fn enabled() -> bool {
    crate::engine::env::flag_on_unless_disabled("ORANGU_HEAD_SPLIT")
}

fn forced_share() -> Option<f64> {
    std::env::var("ORANGU_HEAD_SPLIT_SHARE")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|s| (0.0..1.0).contains(s))
}

fn installed() -> Option<Arc<Split>> {
    SPLIT.read().ok().and_then(|s| s.clone())
}

fn install(split: Option<Split>) {
    if let Ok(mut current) = SPLIT.write() {
        *current = split.map(Arc::new);
    }
}

/// One token's head, `x · wᵀ` over every row of `w`, on `cpu` — or split
/// with the device when a split is installed for this head.
pub fn matmul(cpu: &dyn Backend, x: &[f32], w: &QuantMatrix) -> Vec<f32> {
    match installed() {
        Some(split) if split.head == w.cache_key() && cpu.is_cpu() => {
            run(cpu, &*split.device, x, w, split.cpu_rows)
        }
        _ => cpu.matmul(x, 1, w),
    }
}

/// Rows `0..cpu_rows` on `cpu`, the rest on `device`, at the same time.
fn run(
    cpu: &dyn Backend,
    device: &dyn Backend,
    x: &[f32],
    w: &QuantMatrix,
    cpu_rows: usize,
) -> Vec<f32> {
    let ours = w.rows(0, cpu_rows);
    let theirs = w.rows(cpu_rows, w.out_dim - cpu_rows);
    std::thread::scope(|s| {
        // Started first, so the submission is on its way while the cores work.
        let device_part = s.spawn(|| device.matmul(x, 1, &theirs));
        let mut out = cpu.matmul(x, 1, &ours);
        out.extend(
            device_part
                .join()
                .expect("head split: the device thread panicked"),
        );
        out
    })
}

/// The median of `n` timed runs of `f`, after two untimed ones.
fn median_seconds(n: usize, mut f: impl FnMut()) -> f64 {
    f();
    f();
    let mut times: Vec<f64> = (0..n)
        .map(|_| {
            let started = Instant::now();
            f();
            started.elapsed().as_secs_f64()
        })
        .collect();
    times.sort_by(f64::total_cmp);
    times[n / 2]
}

/// The cores' rows for a device share of `share`, aligned.
fn cpu_rows_for(out_dim: usize, share: f64) -> usize {
    let device = ((out_dim as f64 * share) as usize) / ROW_ALIGN * ROW_ALIGN;
    out_dim - device.min(out_dim)
}

/// Chooses a device share for `model`'s head `w` and keeps the split only
/// when whole decode steps are faster with it (see the module comment),
/// reporting the decision as an `[adapt]` line. Called inside the request
/// pool decode will run in; `cpu` is the model's backend, and `without` a
/// one-token step's seconds on it without the split, when already measured.
pub fn choose(
    cpu: &dyn Backend,
    device: Arc<dyn Backend>,
    w: &QuantMatrix,
    model: &Arc<dyn ModelForward>,
    without: Option<f64>,
) {
    let started = Instant::now();
    let out_dim = w.out_dim;
    // 1. The share, on the head alone. A fixed, non-degenerate activation:
    // its values do not change the cost.
    let x: Vec<f32> = (0..w.in_dim)
        .map(|i| ((i as f32 * 0.618_034).fract() - 0.5) * 2.0)
        .collect();
    let alone = median_seconds(7, || {
        std::hint::black_box(cpu.matmul(&x, 1, w));
    });
    let forced = forced_share();
    let shares: Vec<f64> = match forced {
        Some(share) => vec![share],
        None => SHARES.to_vec(),
    };
    let mut best: Option<(f64, usize, f64)> = None;
    for share in shares {
        let cpu_rows = cpu_rows_for(out_dim, share);
        if cpu_rows == 0 || cpu_rows == out_dim {
            continue;
        }
        let t = median_seconds(7, || {
            std::hint::black_box(run(cpu, &*device, &x, w, cpu_rows));
        });
        if best.is_none_or(|(b, _, _)| t < b) {
            best = Some((t, cpu_rows, share));
        }
    }
    let Some((split_s, cpu_rows, share)) = best else {
        return;
    };
    let head = format!(
        "the head alone takes {:.2} ms on the cores, {:.2} ms back to back with {:.0}% of its \
         rows on the device",
        alone * 1e3,
        split_s * 1e3,
        share * 100.0
    );
    let split = || Split {
        device: device.clone(),
        head: w.cache_key(),
        cpu_rows,
    };
    // 2. The verdict, on whole decode steps at decode's own duty cycle,
    // against the step time decode's own probe measured without the split
    // just before (`without`), or measured here when there is none.
    install(None);
    let without = without.or_else(|| {
        crate::engine::decode_backend::seconds_per_token_after(model, STEP_WARMUP, STEP_TIMED)
    });
    let Some(without) = without else {
        return;
    };
    install(Some(split()));
    let Some(with) = crate::engine::decode_backend::seconds_per_token_unless(
        model,
        STEP_WARMUP,
        STEP_TIMED,
        without * GIVE_UP,
    ) else {
        install(None);
        return;
    };
    let because = format!(
        "a one-token decode step takes {:.1} ms with the head on the cores, {:.1} ms with {} of \
         its {out_dim} rows on the device ({head}; measured in {:.1} s)",
        without * 1e3,
        with * 1e3,
        out_dim - cpu_rows,
        started.elapsed().as_secs_f64()
    );
    if forced.is_some() || without > with * MIN_GAIN {
        crate::engine::adapt::note(
            "output head",
            format!(
                "{} rows on the cores, {} on the device, at once",
                cpu_rows,
                out_dim - cpu_rows
            ),
            because,
        );
    } else {
        install(None);
        crate::engine::adapt::note(
            "output head",
            "on the cores; the device released",
            format!("{because}; the split must make a step {MIN_GAIN}× faster"),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_share_leaves_the_cores_an_aligned_remainder() {
        let rows = cpu_rows_for(262_144, 0.35);
        assert_eq!((262_144 - rows) % ROW_ALIGN, 0);
        assert!((262_144 - rows) as f64 <= 262_144.0 * 0.35);
        // Every smaller share's device rows lie inside the largest share's.
        for share in SHARES {
            assert!(cpu_rows_for(262_144, share) >= cpu_rows_for(262_144, SHARES[0]));
        }
        // A head too small for one aligned block stays on the cores.
        assert_eq!(cpu_rows_for(100, 0.5), 100);
    }
}
