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
//! and how many times it was entered. It also prints a line as each phase
//! of a run ends — setup, every step, the checkpoint, the model file — with
//! the time since the run started, how many cores were busy since the line
//! before, and the memory in use, so a run on another machine can be read
//! from its log alone.

use std::cell::Cell;
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
    Ple,
    PleBackward,
    HcMix,
    HcMixBackward,
    HcCombine,
    HcCombineBackward,
    FullAttention,
    FullAttentionBackward,
    Indexer,
    IndexerBackward,
    DeltaRule,
    DeltaRuleBackward,
    Moe,
    MoeBackward,
    OutputHead,
    OutputHeadBackward,
    GradientFold,
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
        Stage::Ple,
        Stage::PleBackward,
        Stage::HcMix,
        Stage::HcMixBackward,
        Stage::HcCombine,
        Stage::HcCombineBackward,
        Stage::FullAttention,
        Stage::FullAttentionBackward,
        Stage::Indexer,
        Stage::IndexerBackward,
        Stage::DeltaRule,
        Stage::DeltaRuleBackward,
        Stage::Moe,
        Stage::MoeBackward,
        Stage::OutputHead,
        Stage::OutputHeadBackward,
        Stage::GradientFold,
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
            Stage::Ple => "ple",
            Stage::PleBackward => "ple backward",
            Stage::HcMix => "hc mix",
            Stage::HcMixBackward => "hc mix backward",
            Stage::HcCombine => "hc combine",
            Stage::HcCombineBackward => "hc combine bwd",
            Stage::FullAttention => "full attention",
            Stage::FullAttentionBackward => "full attn backward",
            Stage::Indexer => "indexer",
            Stage::IndexerBackward => "indexer backward",
            Stage::DeltaRule => "delta rule",
            Stage::DeltaRuleBackward => "delta rule bwd",
            Stage::Moe => "moe",
            Stage::MoeBackward => "moe backward",
            Stage::OutputHead => "output head",
            Stage::OutputHeadBackward => "output head bwd",
            Stage::GradientFold => "gradient fold",
        }
    }
}

const COUNT: usize = 37;

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

/// When the run started, and the process's CPU seconds at the last phase
/// line and when it was printed.
struct Clock {
    start: Instant,
    last: Instant,
    cpu: f64,
}

static CLOCK: std::sync::Mutex<Option<Clock>> = std::sync::Mutex::new(None);

/// The process's user and system CPU seconds.
fn cpu_seconds() -> Option<f64> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    // The command name may hold spaces; the fields count from after it.
    let fields: Vec<&str> = stat.rsplit_once(')')?.1.split_whitespace().collect();
    let utime: f64 = fields.get(11)?.parse().ok()?;
    let stime: f64 = fields.get(12)?.parse().ok()?;
    // SAFETY: sysconf has no preconditions.
    let ticks = unsafe { libc::sysconf(libc::_SC_CLK_TCK) }.max(1) as f64;
    Some((utime + stime) / ticks)
}

/// A `/proc/meminfo` or `/proc/self/status` field, in bytes.
fn kib_field(path: &str, key: &str) -> Option<u64> {
    let text = std::fs::read_to_string(path).ok()?;
    let line = text.lines().find(|l| l.starts_with(key))?;
    Some(line.split_whitespace().nth(1)?.parse::<u64>().ok()? * 1024)
}

fn gib(bytes: Option<u64>) -> String {
    bytes.map_or_else(
        || "?".into(),
        |b| format!("{:.1} GiB", b as f64 / (1u64 << 30) as f64),
    )
}

/// One line about the machine, at the start of a run.
pub fn machine() {
    if !enabled() {
        return;
    }
    let mut simd = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        for (name, on) in [
            ("sse4.2", std::arch::is_x86_feature_detected!("sse4.2")),
            ("avx2", std::arch::is_x86_feature_detected!("avx2")),
            ("fma", std::arch::is_x86_feature_detected!("fma")),
            ("avx512f", std::arch::is_x86_feature_detected!("avx512f")),
        ] {
            if on {
                simd.push(name);
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        for (name, on) in [
            ("neon", std::arch::is_aarch64_feature_detected!("neon")),
            (
                "dotprod",
                std::arch::is_aarch64_feature_detected!("dotprod"),
            ),
            ("fp16", std::arch::is_aarch64_feature_detected!("fp16")),
            ("sve", std::arch::is_aarch64_feature_detected!("sve")),
        ] {
            if on {
                simd.push(name);
            }
        }
    }
    let mut built = Vec::new();
    for (name, on) in [
        ("avx2", cfg!(target_feature = "avx2")),
        ("fma", cfg!(target_feature = "fma")),
        ("neon", cfg!(target_feature = "neon")),
    ] {
        if on {
            built.push(name);
        }
    }
    println!(
        "machine: {} with {} cores, {} threads in the pool; cpu has {}; built for {}; memory {} of which {} available",
        std::env::consts::ARCH,
        std::thread::available_parallelism().map_or(0, |n| n.get()),
        rayon::current_num_threads(),
        simd.join(" "),
        if built.is_empty() {
            "the baseline".to_string()
        } else {
            built.join(" ")
        },
        gib(kib_field("/proc/meminfo", "MemTotal:")),
        gib(kib_field("/proc/meminfo", "MemAvailable:")),
    );
}

/// Prints `[seconds] what` with the cores busy since the last such line and
/// the memory in use, when the instrument is on.
pub fn phase(what: &str) {
    if !enabled() {
        return;
    }
    let now = Instant::now();
    let cpu = cpu_seconds().unwrap_or(0.0);
    let mut clock = CLOCK.lock().unwrap_or_else(|e| e.into_inner());
    let c = clock.get_or_insert(Clock {
        start: now,
        last: now,
        cpu,
    });
    let wall = now.duration_since(c.last).as_secs_f64();
    let busy = if wall > 0.0 {
        (cpu - c.cpu) / wall
    } else {
        0.0
    };
    println!(
        "[{:>9.1}s] {what}  ({wall:.1} s, {busy:.1} cores busy, resident {}, available {})",
        now.duration_since(c.start).as_secs_f64(),
        gib(kib_field("/proc/self/status", "VmRSS:")),
        gib(kib_field("/proc/meminfo", "MemAvailable:")),
    );
    c.last = now;
    c.cpu = cpu;
}

thread_local! {
    /// Time charged so far to stages run inside the current one, on this
    /// thread.
    static INNER: Cell<u64> = const { Cell::new(0) };
}

/// Runs `body`, charging its elapsed time to `stage`.
///
/// Elapsed, not CPU: the point is to catch a stage that is slow *because*
/// it is narrow, and CPU time cannot see that at all. A stage timed inside
/// another on the same thread is charged to itself only, and taken out of
/// the one around it, so the stages add up to the pass however they nest.
#[inline]
pub fn time<T>(stage: Stage, body: impl FnOnce() -> T) -> T {
    if !enabled() {
        return body();
    }
    let outer = INNER.with(|inner| inner.replace(0));
    let started = Instant::now();
    let out = body();
    let elapsed = started.elapsed().as_nanos() as u64;
    let nested = INNER.with(|inner| inner.replace(outer + elapsed));
    NANOS[stage as usize].fetch_add(elapsed.saturating_sub(nested), Ordering::Relaxed);
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
