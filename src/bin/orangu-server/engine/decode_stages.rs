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

//! Wall-clock time per forward-pass stage, drained by `GET /decode-stages`.
//!
//! A CPU profile answers "which function burned the cycles". On a model whose
//! forward pass alternates between a device submission, a wide `rayon` fan-out
//! and a stretch of single-threaded scalar arithmetic, that is the wrong
//! question: a stage running alone on one core for a fifth of the token costs
//! a fifth of the token, while the same cycle count spread over sixteen cores
//! costs a sixteenth — and a sampling profile reports the two identically.
//! These counters measure the **elapsed** time of each stage on the thread
//! that runs it, which is what a token's latency is actually made of.
//!
//! They are the counterpart to [`super::moe_stats`], and follow it exactly:
//! relaxed atomics, **drain-on-read** (read once to discard the warmup, run
//! the workload, read again to get precisely that window), and off unless
//! asked for.
//!
//! **Off by default; opt in with `ORANGU_DECODE_STAGES=1`.** Two
//! `Instant::now()` calls and two relaxed atomic adds per stage entry is not
//! free at the rate a decode step enters them, and a measurement tool that
//! perturbs the thing it measures whenever it is linked in is worse than no
//! tool. `ORANGU_DECODE_STAGES=0` is a real control arm, per
//! [`super::env::flag_on`].
//!
//! **What is generic and what is not.** [`Stage::Forward`],
//! [`Stage::Attn`], [`Stage::FfnRouter`] and [`Stage::FfnRouted`] are timed
//! inside code every architecture goes through — the generation loop, the
//! shared attention kernels, and the shared mixture-of-experts helpers — so
//! they are reported for *any* model without that architecture's file
//! knowing this module exists. The rest are timed at the one call site that
//! can identify them, which is per-architecture; an architecture that has not
//! been given them reports zero for those stages and the whole of its pass
//! under `other`. That is the intended shape, not a gap: `other` is always
//! `forward` minus what was attributed, so a breakdown is never silently
//! incomplete.
//!
//! **Stages nest and can overlap.** [`Stage::Forward`] is the only parent and
//! spans one whole pass; every other stage is a disjoint sibling inside it,
//! so `other` is `forward` less the rest. The one exception is
//! [`Stage::FfnRouted`] and [`Stage::FfnShared`], which run concurrently when
//! an FFN overlaps its two branches — their sum can then exceed the elapsed
//! time they shared, which is what overlapping them was for. Percentages are
//! reported against the pass, never against each other.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// One measured span of a forward pass.
///
/// Ordered as the pass runs them, because that is the order they are reported
/// in and a breakdown that reads out of order is harder to follow than one
/// that does not.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// One whole forward pass, from the generation loop. The parent of every
    /// other stage, and the denominator every percentage is taken against.
    /// Timed generically, so every architecture reports it.
    Forward,
    /// Gathering the input rows out of the token-embedding matrix.
    Embed,
    /// A recurrent block's batched input projections — q/k/v, the output
    /// gate, and beta/alpha.
    RecurrentProject,
    /// A recurrent block's per-head state update: the causal convolution, the
    /// L2 norms, the delta rule itself, and the gated output norm. Scalar and
    /// sequential by construction — this is the stage whose elapsed time and
    /// whose cycle count differ most.
    RecurrentDelta,
    /// A recurrent block's output projection.
    RecurrentOut,
    /// Scaled dot-product attention itself — the shared kernels, host and
    /// device, that every attention architecture calls. Timed generically.
    /// The surrounding q/k/v and output projections are *not* in here.
    Attn,
    /// The router matmul in front of a mixture-of-experts FFN. Timed
    /// generically, in the shared helper every MoE architecture routes
    /// through.
    FfnRouter,
    /// Turning router logits into a selection: the softmax, the top-k, and
    /// the renormalization. Separate from [`Self::FfnRouter`] because they
    /// answer different questions — one is bytes read, the other is
    /// arithmetic over `n_expert` numbers — and on a model with 256 experts
    /// the second was four times the first.
    FfnSelect,
    /// Evaluating the routed experts. Timed generically, in the shared
    /// helpers every MoE architecture evaluates through — both the host path
    /// and the batched device one.
    FfnRouted,
    /// Evaluating the gated shared expert.
    FfnShared,
    /// Summing the shared and routed contributions into the block's output.
    FfnCombine,
    /// A dense (non-MoE) feed-forward block — gate, up, the activation and
    /// down — timed where an architecture routes it as one unit
    /// (`qwen_hybrid::DenseFfn`), so a device backend's one-submission FFN
    /// has a line of its own.
    FfnDense,
    /// The attention block's input side: the q/k/v projections. On a model
    /// whose weights sit under the host-matmul crossover these run on the
    /// CPU, and at decode that is most of what the host does before it
    /// waits for attention.
    AttnProject,
    /// Waiting for attention's own submission and bringing its output
    /// home — the readback a layer pays when the work after attention is
    /// not in the same submission. The ceiling on what fusing that work in
    /// could remove.
    AttnRead,
    /// The attention block's output side, as the architecture submits it:
    /// the `wo` projection, the residual add and the feed-forward norm.
    /// One device submission per layer where those are fused.
    AttnOut,
    /// The final norm and the output projection over the vocabulary.
    Head,
    /// Recording one token's layer loop into a device submission
    /// (`qwen_hybrid::Trunk::forward_token_on_device`) — host time only,
    /// before anything runs.
    DeviceRecord,
    /// That submission's submit, the wait for the device and the residual
    /// stream's readback.
    DeviceSubmit,
}

impl Stage {
    /// Every stage, in report order.
    pub const ALL: [Stage; 18] = [
        Stage::Forward,
        Stage::Embed,
        Stage::RecurrentProject,
        Stage::RecurrentDelta,
        Stage::RecurrentOut,
        Stage::Attn,
        Stage::FfnRouter,
        Stage::FfnSelect,
        Stage::FfnRouted,
        Stage::FfnShared,
        Stage::FfnCombine,
        Stage::FfnDense,
        Stage::AttnProject,
        Stage::AttnRead,
        Stage::AttnOut,
        Stage::Head,
        Stage::DeviceRecord,
        Stage::DeviceSubmit,
    ];

    /// The name this stage is reported under. Dotted, so a reader can see the
    /// nesting without the report having to draw it.
    pub fn name(self) -> &'static str {
        match self {
            Stage::Forward => "forward",
            Stage::Embed => "embed",
            Stage::RecurrentProject => "recurrent.project",
            Stage::RecurrentDelta => "recurrent.delta",
            Stage::RecurrentOut => "recurrent.out",
            Stage::Attn => "attn",
            Stage::FfnRouter => "ffn.router",
            Stage::FfnSelect => "ffn.select",
            Stage::FfnRouted => "ffn.routed",
            Stage::FfnShared => "ffn.shared",
            Stage::FfnCombine => "ffn.combine",
            Stage::FfnDense => "ffn.dense",
            Stage::AttnProject => "attn.project",
            Stage::AttnRead => "attn.read",
            Stage::AttnOut => "attn.out",
            Stage::Head => "head",
            Stage::DeviceRecord => "device.record",
            Stage::DeviceSubmit => "device.submit",
        }
    }

    /// Whether this stage contains the others, and so must not be added to
    /// their total. Only [`Stage::Forward`] is a parent; naming the
    /// relationship here rather than in the reporting code keeps the two from
    /// drifting.
    pub fn is_parent(self) -> bool {
        self == Stage::Forward
    }

    fn index(self) -> usize {
        match self {
            Stage::Forward => 0,
            Stage::Embed => 1,
            Stage::RecurrentProject => 2,
            Stage::RecurrentDelta => 3,
            Stage::RecurrentOut => 4,
            Stage::Attn => 5,
            Stage::FfnRouter => 6,
            Stage::FfnSelect => 7,
            Stage::FfnRouted => 8,
            Stage::FfnShared => 9,
            Stage::FfnCombine => 10,
            Stage::FfnDense => 11,
            Stage::AttnProject => 12,
            Stage::AttnRead => 13,
            Stage::AttnOut => 14,
            Stage::Head => 15,
            Stage::DeviceRecord => 16,
            Stage::DeviceSubmit => 17,
        }
    }
}

/// Nanoseconds and entry counts per stage. Relaxed throughout: these are
/// statistics, and no other memory is published through them.
static NANOS: [AtomicU64; Stage::ALL.len()] = [const { AtomicU64::new(0) }; Stage::ALL.len()];
static CALLS: [AtomicU64; Stage::ALL.len()] = [const { AtomicU64::new(0) }; Stage::ALL.len()];
/// Device submissions charged to each stage — see [`record_submission`].
static SUBMITS: [AtomicU64; Stage::ALL.len()] = [const { AtomicU64::new(0) }; Stage::ALL.len()];

thread_local! {
    /// The innermost stage this thread is inside, so a device submission can
    /// be charged to it. `None` outside any scope.
    ///
    /// Thread-local because a submission is made by whichever thread is
    /// running the stage, and there is no handle threaded from one to the
    /// other. That bounds what this can see: a submission issued from a
    /// worker a stage fanned out onto is charged to nothing, because the
    /// worker never entered the scope. Every stage that submits today does so
    /// on the thread that opened it, and a stage whose submissions go missing
    /// shows up as a count of zero next to a non-zero time rather than as a
    /// wrong attribution.
    static CURRENT: std::cell::Cell<Option<Stage>> = const { std::cell::Cell::new(None) };
}

/// Forward passes that have finished since the last drain, so a window's
/// per-stage totals can be divided into a per-pass cost without the reader
/// having to know how many tokens the benchmark asked for. Incremented by
/// [`Stage::Forward`]'s own scope, so it cannot drift from the stage it
/// divides.
static PASSES: AtomicU64 = AtomicU64::new(0);

/// Whether the counters are being kept, read once.
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| super::env::flag_on("ORANGU_DECODE_STAGES"))
}

/// Times `f` as `stage`, or just runs it when the counters are off.
///
/// Written as a wrapper rather than an RAII guard so that the disabled path is
/// a single predictable branch around an untouched call, with no timer
/// constructed and nothing to drop.
pub fn scope<T>(stage: Stage, f: impl FnOnce() -> T) -> T {
    if !enabled() {
        return f();
    }
    let start = Instant::now();
    // Restored rather than cleared: stages nest, and a submission made after
    // an inner one returns still belongs to the outer stage.
    let outer = CURRENT.with(|c| c.replace(Some(stage)));
    let out = f();
    CURRENT.with(|c| c.set(outer));
    let elapsed = start.elapsed().as_nanos() as u64;
    NANOS[stage.index()].fetch_add(elapsed, Ordering::Relaxed);
    CALLS[stage.index()].fetch_add(1, Ordering::Relaxed);
    out
}

/// [`scope`] as a guard, for a stage that spans statements rather than one
/// expression — a loop in the middle of a function, where wrapping the
/// whole thing in a closure would mean restructuring the code around the
/// measurement.
///
/// The stage closes when the guard drops, so a guard must be dropped where
/// the stage really ends and not merely at the end of the function: a stage
/// that runs long swallows its siblings, and the breakdown stops summing to
/// the pass.
pub fn enter(stage: Stage) -> Option<StageGuard> {
    if !enabled() {
        return None;
    }
    Some(StageGuard {
        stage,
        start: Instant::now(),
        outer: CURRENT.with(|c| c.replace(Some(stage))),
    })
}

/// See [`enter`].
pub struct StageGuard {
    stage: Stage,
    start: Instant,
    outer: Option<Stage>,
}

impl Drop for StageGuard {
    fn drop(&mut self) {
        CURRENT.with(|c| c.set(self.outer));
        let elapsed = self.start.elapsed().as_nanos() as u64;
        NANOS[self.stage.index()].fetch_add(elapsed, Ordering::Relaxed);
        CALLS[self.stage.index()].fetch_add(1, Ordering::Relaxed);
    }
}

/// Charges one device submission to whichever stage this thread is inside.
///
/// Called from the backend's `queue.submit` sites, so the breakdown can say
/// not just where a token's time went but **how many round trips each stage
/// cost** — which is the number a batching change moves and a timing alone
/// cannot separate from the arithmetic.
pub fn record_submission() {
    ALL_SUBMITS.fetch_add(1, Ordering::Relaxed);
    if !enabled() {
        return;
    }
    if let Some(stage) = CURRENT.with(|c| c.get()) {
        SUBMITS[stage.index()].fetch_add(1, Ordering::Relaxed);
    }
}

/// Every submission the backends have recorded since the process started,
/// instrument or no instrument — the prefill chunk sizer reads it before
/// and after a chunk to learn how many round trips the chunk was, which is
/// what turns a chunk's wall time into an estimate of its longest single
/// submission (see `generate::budget_elapsed`). A lower bound: a submit path
/// that does not call [`record_submission`] is simply not counted, which
/// errs toward the longer estimate.
static ALL_SUBMITS: AtomicU64 = AtomicU64::new(0);

pub fn submissions_so_far() -> u64 {
    ALL_SUBMITS.load(Ordering::Relaxed)
}

/// Times one whole **decode** forward pass, and counts it.
///
/// The generation loop's own wrapper: this is what makes the breakdown
/// generic. Every architecture is driven through here, so `forward` — the
/// denominator — is measured whether or not that architecture's file has been
/// given any of the finer stages.
pub fn pass<T>(f: impl FnOnce() -> T) -> T {
    pass_of(1, f)
}

/// [`pass`] for a pass of known width. **A pass wider than one token leaves
/// these counters exactly as it found them.**
///
/// A request measured at depth prefills its prompt before it generates
/// anything, and those chunks are forward passes through this same wrapper.
/// Counted in, a multi-second prefill is averaged into the per-pass
/// breakdown of a 90 ms decode step: measured on a mixture at depth 1024,
/// the window reported **251.5 ms a pass against a token that took 92.3** —
/// nearly two thirds of the breakdown belonging to work the reader is not
/// looking at, distributed across the stages in prefill's proportions rather
/// than decode's.
///
/// So a wide pass snapshots every counter on the way in and restores it on
/// the way out. Subtracting the delta rather than suppressing the writes is
/// what makes it complete: a stage that ran on a `rayon` worker inside the
/// pass is still removed, where a flag on the calling thread would have
/// missed it.
///
/// **The one thing this cannot separate is a genuinely concurrent decode**
/// on another request, whose contribution is subtracted along with the
/// prefill's. That is a measurement run with one stream, which is what a
/// breakdown is read from anyway.
pub fn pass_of<T>(n_tokens: usize, f: impl FnOnce() -> T) -> T {
    if !enabled() {
        return f();
    }
    if n_tokens > 1 {
        let before = CounterSnapshot::take();
        let out = scope(Stage::Forward, f);
        before.restore();
        return out;
    }
    let out = scope(Stage::Forward, f);
    PASSES.fetch_add(1, Ordering::Relaxed);
    out
}

/// Every counter as it stood, so a pass whose contribution is not wanted can
/// put them back — see [`pass_of`].
struct CounterSnapshot {
    nanos: Vec<u64>,
    calls: Vec<u64>,
    submits: Vec<u64>,
    all_submits: u64,
}

impl CounterSnapshot {
    fn take() -> Self {
        Self {
            nanos: NANOS.iter().map(|v| v.load(Ordering::Relaxed)).collect(),
            calls: CALLS.iter().map(|v| v.load(Ordering::Relaxed)).collect(),
            submits: SUBMITS.iter().map(|v| v.load(Ordering::Relaxed)).collect(),
            all_submits: ALL_SUBMITS.load(Ordering::Relaxed),
        }
    }

    fn restore(self) {
        for (slot, was) in NANOS.iter().zip(self.nanos) {
            slot.store(was, Ordering::Relaxed);
        }
        for (slot, was) in CALLS.iter().zip(self.calls) {
            slot.store(was, Ordering::Relaxed);
        }
        for (slot, was) in SUBMITS.iter().zip(self.submits) {
            slot.store(was, Ordering::Relaxed);
        }
        ALL_SUBMITS.store(self.all_submits, Ordering::Relaxed);
    }
}

/// One stage's window.
pub struct StageTotal {
    pub name: &'static str,
    pub nanos: u64,
    pub calls: u64,
    /// Device submissions made inside this stage, on the thread that opened
    /// it — see [`record_submission`].
    pub submits: u64,
    /// Whether this stage contains the others — see [`Stage::is_parent`].
    pub parent: bool,
}

/// The window since the last drain, and **reset**.
///
/// `passes` is zero when nothing ran, which is how a caller tells "not
/// measured" from "measured and took no time" — the same rule `/gpu-timings`
/// and `/moe-stats` report their windows by.
pub fn take() -> (u64, Vec<StageTotal>) {
    let passes = PASSES.swap(0, Ordering::Relaxed);
    let totals = Stage::ALL
        .iter()
        .map(|&stage| StageTotal {
            name: stage.name(),
            nanos: NANOS[stage.index()].swap(0, Ordering::Relaxed),
            calls: CALLS[stage.index()].swap(0, Ordering::Relaxed),
            submits: SUBMITS[stage.index()].swap(0, Ordering::Relaxed),
            parent: stage.is_parent(),
        })
        .collect();
    (passes, totals)
}

/// The window as JSON, for `GET /decode-stages`.
pub fn take_json() -> serde_json::Value {
    let enabled = enabled();
    let (passes, totals) = take();
    serde_json::json!({
        "enabled": enabled,
        "passes": passes,
        "stages": totals
            .iter()
            .map(|t| serde_json::json!({
                "name": t.name,
                "ms": t.nanos as f64 / 1e6,
                "calls": t.calls,
                "submits": t.submits,
                "parent": t.parent,
            }))
            .collect::<Vec<_>>(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A wide pass restores every counter it found, which is what keeps a
    /// request's prompt out of the breakdown of its decode steps. Asserted
    /// on the mechanism rather than through `pass_of`, because the counters
    /// are off unless the process was started with the variable set and a
    /// test that quietly returns when they are is a test that never fails.
    ///
    /// The case this exists for, measured: at depth 1024 the window
    /// reported 251.5 ms a pass against a token that took 92.3, the excess
    /// being a multi-second prefill spread over 64 decode steps.
    #[test]
    fn a_snapshot_restores_every_counter_it_found() {
        let attn = Stage::Attn.index();
        let head = Stage::Head.index();
        // Whatever a neighbouring test left behind is the starting state;
        // this asserts a delta, so it does not care what that is.
        NANOS[attn].store(1_000, Ordering::Relaxed);
        CALLS[attn].store(7, Ordering::Relaxed);
        SUBMITS[head].store(3, Ordering::Relaxed);
        ALL_SUBMITS.store(11, Ordering::Relaxed);

        let before = CounterSnapshot::take();
        // What a prefill pass would have added.
        NANOS[attn].fetch_add(5_000_000, Ordering::Relaxed);
        CALLS[attn].fetch_add(48, Ordering::Relaxed);
        SUBMITS[head].fetch_add(96, Ordering::Relaxed);
        ALL_SUBMITS.fetch_add(96, Ordering::Relaxed);
        before.restore();

        assert_eq!(NANOS[attn].load(Ordering::Relaxed), 1_000);
        assert_eq!(CALLS[attn].load(Ordering::Relaxed), 7);
        assert_eq!(SUBMITS[head].load(Ordering::Relaxed), 3);
        assert_eq!(ALL_SUBMITS.load(Ordering::Relaxed), 11);
    }

    /// Every stage must have a distinct index and a distinct name, or two of
    /// them would silently share a counter — a bug that would look like one
    /// stage being unexpectedly expensive and the other unexpectedly free.
    #[test]
    fn stage_indices_and_names_are_distinct() {
        let mut indices: Vec<usize> = Stage::ALL.iter().map(|s| s.index()).collect();
        indices.sort_unstable();
        indices.dedup();
        assert_eq!(indices.len(), Stage::ALL.len());
        assert_eq!(indices.last(), Some(&(Stage::ALL.len() - 1)));

        let mut names: Vec<&str> = Stage::ALL.iter().map(|s| s.name()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), Stage::ALL.len());
    }

    /// Exactly one stage is a parent, and it is the one the reporting divides
    /// by. If a second were added without the reporting learning about it,
    /// the siblings' total would double-count.
    #[test]
    fn exactly_one_stage_is_a_parent() {
        assert_eq!(Stage::ALL.iter().filter(|s| s.is_parent()).count(), 1);
        assert!(Stage::Forward.is_parent());
    }

    /// The disabled path must return the closure's value and leave the
    /// counters alone — a diagnostic that costs something when it is off is
    /// one nobody can afford to leave linked in.
    #[test]
    fn a_disabled_scope_still_returns_the_value() {
        // Whether the process has the flag set is not this test's business;
        // what it asserts is true either way.
        assert_eq!(scope(Stage::Head, || 7), 7);
    }
}
