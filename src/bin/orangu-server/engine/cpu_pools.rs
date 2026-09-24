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

//! Which cores a request's CPU work runs on, chosen by measurement — the
//! worker pool per phase that `hardware::big_cores` left as the open task.
//!
//! On a machine with big and little cores, rayon's default is one worker
//! per core, and a compute-bound step split evenly across them finishes
//! when the slowest core does. Whether that costs or pays is the
//! workload's and the machine's question: on the CIX P1 (8 × A720 + 4 ×
//! A520) the little cores once helped a prompt, and after `doc/PERF-ALL.md`
//! tasks 3, 12 and 13 they no longer do (prompts 231 tok/s at 2048 tokens
//! on twelve workers, 229 on the eight big cores), while they cost the CPU
//! backend's decode a third of its rate (8.2 tok/s on twelve, 12.6 on the
//! eight — task 9).
//!
//! So beside the global pool — every core, what the loader and everything
//! outside a request use — a machine with two kinds of core gets a second
//! pool pinned to the big ones, and each phase is timed in both once the
//! model is loaded: prompts on the model's widest FFN GEMM at a prompt
//! chunk's width, decode (where the cores decode) on the model's own
//! one-token steps. A request then runs in the pool of its phase — the
//! decode pool when the cores decode, else the prompts pool; one pool per
//! request, since a prompt and its answer share the request's thread.
//!
//! Not built when `threads` (or `--threads`, `ORANGU_THREADS`) sizes the
//! pool, or `ORANGU_EXPERT_BIG_CORES=1` pins it: those are decisions, and a
//! measurement does not overrule them. `ORANGU_CPU_POOLS=0` turns it off.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, Ordering};

/// Which pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pool {
    /// The global pool: every core.
    All,
    /// The big cores only, pinned.
    Big,
}

impl Pool {
    fn code(self) -> u8 {
        match self {
            Self::All => 0,
            Self::Big => 1,
        }
    }

    fn from_code(code: u8) -> Self {
        if code == 1 { Self::Big } else { Self::All }
    }
}

/// The every-core pool a request runs in when it does not run on the big
/// cores — the same size as the global one, but a pool a request can run
/// *inside*: a request on a tokio thread hands each of a decode step's
/// hundreds of parallel regions to the global pool from outside (queue,
/// wake, latch), and one on a pool worker splits them from its own deque.
/// On the CIX P1 that alone took the CPU backend's decode from 12.6 to
/// 17.5 tok/s on the same eight cores (`doc/PERF-ALL.md`, task 9).
static ALL: OnceLock<rayon::ThreadPool> = OnceLock::new();

/// The big-core pool, when this machine has one.
static BIG: OnceLock<(rayon::ThreadPool, usize)> = OnceLock::new();

/// The pool each phase measured faster in.
static PROMPTS: AtomicU8 = AtomicU8::new(0);
static DECODE: AtomicU8 = AtomicU8::new(0);

/// Whether a request's decode runs on the cores (`main`'s decode decision).
static DECODE_ON_CPU: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// How much faster the big cores must be to take a phase: a little, since
/// fewer workers also leave the little cores to everything else.
const BIG_WITHIN: f64 = 1.02;

/// Builds the request pools — called by `main` when the global pool is the
/// default one: every core (`all` workers), and the big cores when this
/// machine's cores are of two kinds. `pin` restricts a worker to the cores
/// given.
pub fn init(all: usize, big_cores: Option<Vec<usize>>, pin: fn(&[usize])) {
    if !crate::engine::env::flag_on_unless_disabled("ORANGU_CPU_POOLS") {
        return;
    }
    if let Ok(pool) = rayon::ThreadPoolBuilder::new()
        .num_threads(all)
        .thread_name(|i| format!("orangu-cpu-{i}"))
        .build()
    {
        let _ = ALL.set(pool);
    }
    let Some(big_cores) = big_cores.filter(|c| !c.is_empty()) else {
        return;
    };
    let n = big_cores.len();
    let built = rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .thread_name(|i| format!("orangu-big-{i}"))
        .start_handler(move |_| pin(&big_cores))
        .build();
    if let Ok(pool) = built {
        let _ = BIG.set((pool, n));
    }
}

/// Runs `f` inside `pool`: the big cores when that pool exists, else every
/// core's request pool, else here.
pub fn in_pool<T: Send>(pool: Pool, f: impl FnOnce() -> T + Send) -> T {
    match (pool, BIG.get(), ALL.get()) {
        (Pool::Big, Some((big, _)), _) => big.install(f),
        (_, _, Some(all)) => all.install(f),
        _ => f(),
    }
}

/// Times `time` in each pool and keeps the faster for `phase`, reporting it
/// as an `[adapt]` line. `None` from either timing keeps every core.
fn choose(phase: &'static str, what: &str, mut time: impl FnMut(Pool) -> Option<f64>) -> Pool {
    let Some((_, n_big)) = BIG.get() else {
        return Pool::All;
    };
    let all_threads = ALL
        .get()
        .map_or_else(rayon::current_num_threads, |p| p.current_num_threads());
    let (Some(all), Some(big)) = (time(Pool::All), time(Pool::Big)) else {
        return Pool::All;
    };
    let chosen = if big <= all * BIG_WITHIN {
        Pool::Big
    } else {
        Pool::All
    };
    crate::engine::adapt::note(
        phase,
        match chosen {
            Pool::Big => format!("the {n_big} big cores"),
            Pool::All => format!("all {all_threads} cores"),
        },
        format!(
            "{what} takes {:.1} ms on all {all_threads} cores, {:.1} ms on the {n_big} big ones",
            all * 1e3,
            big * 1e3
        ),
    );
    chosen
}

/// Decides the prompts pool: `time` runs a prompt-shaped workload once and
/// returns its seconds (the caller warms and repeats inside it).
pub fn choose_prompts(what: &str, time: impl FnMut(Pool) -> Option<f64>) {
    PROMPTS.store(
        choose("cpu pool (prompts)", what, time).code(),
        Ordering::Relaxed,
    );
}

/// Decides the decode pool, where the cores decode — or, called with the
/// two timings `main`'s decode decision already took, records them.
pub fn choose_decode(what: &str, time: impl FnMut(Pool) -> Option<f64>) {
    DECODE.store(
        choose("cpu pool (decode)", what, time).code(),
        Ordering::Relaxed,
    );
}

/// Whether there is a big-core pool to choose at all.
pub fn available() -> bool {
    BIG.get().is_some()
}

/// Records where a request's decode runs, so [`for_request`] picks its pool.
pub fn set_decode_on_cpu(on_cpu: bool) {
    DECODE_ON_CPU.store(on_cpu, Ordering::Relaxed);
}

/// Runs a whole request — its prompt and its answer — in its pool: the
/// decode pool when the cores decode, else the prompts pool.
pub fn for_request<T: Send>(f: impl FnOnce() -> T + Send) -> T {
    let pool = if DECODE_ON_CPU.load(Ordering::Relaxed) {
        Pool::from_code(DECODE.load(Ordering::Relaxed))
    } else {
        Pool::from_code(PROMPTS.load(Ordering::Relaxed))
    };
    in_pool(pool, f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn without_a_big_pool_everything_runs_here() {
        // No `init` in the test process: `Big` falls back to the caller.
        let here = std::thread::current().id();
        assert_eq!(in_pool(Pool::Big, || std::thread::current().id()), here);
        assert_eq!(for_request(|| 7), 7);
    }
}
