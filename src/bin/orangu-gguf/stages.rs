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

//! Where a training step's *wall clock* goes, stage by stage.
//!
//! A sampling profiler answers a different question from the one this
//! answers, and the difference is the whole reason this exists. `perf`
//! counts where the CPUs are; a stage that runs on one thread while
//! fifteen sit idle costs a sixteenth of the samples it costs in seconds.
//! Every kernel here is called from inside a step, one after another, so
//! timing them on the calling thread — around the parallel region, not
//! inside it — measures exactly the seconds the step is made of, and the
//! total comes back within a percent of the step itself.
//!
//! Off unless `ORANGU_GGUF_STAGES` is set to something other than `0`, and
//! the check is one relaxed atomic load, so a disabled build of the
//! instrument costs a predictable branch per kernel call and nothing else.
//!
//! ```sh
//! ORANGU_GGUF_STAGES=1 orangu-gguf manifest.json
//! ```
//!
//! prints a table at the end of the run: seconds in each stage, its share,
//! and how many times it was entered.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

/// One line of the report. Ordered as a step runs, so the table reads as a
/// forward pass followed by a backward one.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Embed,
    RmsNorm,
    MatmulQkv,
    Rope,
    Attention,
    MatmulAttnOut,
    MatmulSwiglu,
    MatmulFfnDown,
    Logits,
    CrossEntropy,
    SwigluBackward,
    AttentionBackward,
    RmsNormBackward,
    MatmulDx,
    MatmulDw,
    QkPreNorm,
    EmbedBackward,
    GradientZero,
    GradientNorm,
    Optimizer,
}

impl Stage {
    pub const ALL: &'static [Stage] = &[
        Stage::Embed,
        Stage::RmsNorm,
        Stage::MatmulQkv,
        Stage::Rope,
        Stage::Attention,
        Stage::MatmulAttnOut,
        Stage::MatmulSwiglu,
        Stage::MatmulFfnDown,
        Stage::Logits,
        Stage::CrossEntropy,
        Stage::SwigluBackward,
        Stage::AttentionBackward,
        Stage::RmsNormBackward,
        Stage::MatmulDx,
        Stage::MatmulDw,
        Stage::QkPreNorm,
        Stage::EmbedBackward,
        Stage::GradientZero,
        Stage::GradientNorm,
        Stage::Optimizer,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Stage::Embed => "embed",
            Stage::RmsNorm => "rmsnorm",
            Stage::MatmulQkv => "matmul qkv",
            Stage::Rope => "rope",
            Stage::Attention => "attention",
            Stage::MatmulAttnOut => "matmul attn_out",
            Stage::MatmulSwiglu => "matmul swiglu",
            Stage::MatmulFfnDown => "matmul ffn_down",
            Stage::Logits => "logits",
            Stage::CrossEntropy => "cross entropy",
            Stage::SwigluBackward => "swiglu backward",
            Stage::AttentionBackward => "attention backward",
            Stage::RmsNormBackward => "rmsnorm backward",
            Stage::MatmulDx => "matmul dx",
            Stage::MatmulDw => "matmul dw",
            Stage::QkPreNorm => "qk pre-norm",
            Stage::EmbedBackward => "embed backward",
            Stage::GradientZero => "gradient zero",
            Stage::GradientNorm => "gradient norm",
            Stage::Optimizer => "optimizer",
        }
    }
}

const COUNT: usize = 20;

static NANOS: [AtomicU64; COUNT] = [const { AtomicU64::new(0) }; COUNT];
static CALLS: [AtomicU64; COUNT] = [const { AtomicU64::new(0) }; COUNT];
static ENABLED: AtomicBool = AtomicBool::new(false);

/// Reads the environment once, at startup. Called before any timing so the
/// hot path never touches the environment or a lock.
pub fn init() {
    let on = match std::env::var("ORANGU_GGUF_STAGES") {
        Ok(value) => value != "0",
        Err(_) => false,
    };
    ENABLED.store(on, Ordering::Relaxed);
}

pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Runs `body`, charging its elapsed time to `stage`.
///
/// Elapsed, not CPU: the point is to catch a stage that is slow *because*
/// it is narrow, and CPU time cannot see that at all.
#[inline]
pub fn time<T>(stage: Stage, body: impl FnOnce() -> T) -> T {
    if !enabled() {
        return body();
    }
    let started = Instant::now();
    let out = body();
    let elapsed = started.elapsed().as_nanos() as u64;
    NANOS[stage as usize].fetch_add(elapsed, Ordering::Relaxed);
    CALLS[stage as usize].fetch_add(1, Ordering::Relaxed);
    out
}

/// Forgets everything measured so far, so a report covers the steps that
/// were asked for rather than the warm-up as well.
pub fn reset() {
    for i in 0..COUNT {
        NANOS[i].store(0, Ordering::Relaxed);
        CALLS[i].store(0, Ordering::Relaxed);
    }
}

/// The table, or nothing at all when the instrument is off.
pub fn report(wall: f64) -> String {
    if !enabled() {
        return String::new();
    }
    let mut rows: Vec<(Stage, f64, u64)> = Stage::ALL
        .iter()
        .map(|&s| {
            (
                s,
                NANOS[s as usize].load(Ordering::Relaxed) as f64 / 1e9,
                CALLS[s as usize].load(Ordering::Relaxed),
            )
        })
        .filter(|(_, seconds, _)| *seconds > 0.0)
        .collect();
    rows.sort_by(|a, b| b.1.total_cmp(&a.1));

    let measured: f64 = rows.iter().map(|(_, s, _)| *s).sum();
    let mut out = String::new();
    out.push_str("\nstage                seconds   share    calls      per call\n");
    for (stage, seconds, calls) in &rows {
        let per = if *calls > 0 {
            seconds * 1e3 / *calls as f64
        } else {
            0.0
        };
        out.push_str(&format!(
            "{:<18} {:>9.2}  {:>5.1}%  {:>7}  {:>8.3} ms\n",
            stage.name(),
            seconds,
            100.0 * seconds / wall.max(f64::MIN_POSITIVE),
            calls,
            per
        ));
    }
    out.push_str(&format!(
        "{:<18} {:>9.2}  {:>5.1}%\n",
        "measured",
        measured,
        100.0 * measured / wall.max(f64::MIN_POSITIVE)
    ));
    // What the stages did not account for: the gap is corpus reading,
    // allocation outside a timed region, and the report itself. A large one
    // means a stage is missing, not that the run was idle.
    out.push_str(&format!(
        "{:<18} {:>9.2}  {:>5.1}%\n",
        "unattributed",
        wall - measured,
        100.0 * (wall - measured) / wall.max(f64::MIN_POSITIVE)
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every variant has a name and a slot, and `ALL` is the whole set —
    /// a variant missing from `ALL` would be timed and never reported.
    #[test]
    fn every_stage_is_in_the_table() {
        assert_eq!(Stage::ALL.len(), COUNT);
        for (i, stage) in Stage::ALL.iter().enumerate() {
            assert_eq!(*stage as usize, i, "{} is out of order", stage.name());
            assert!(!stage.name().is_empty());
        }
    }

    /// Disabled is the default, and a disabled instrument still runs the
    /// body exactly once.
    #[test]
    fn the_body_runs_whether_or_not_the_instrument_is_on() {
        let mut count = 0;
        time(Stage::Embed, || count += 1);
        ENABLED.store(true, Ordering::Relaxed);
        time(Stage::Embed, || count += 1);
        ENABLED.store(false, Ordering::Relaxed);
        assert_eq!(count, 2);
    }
}
