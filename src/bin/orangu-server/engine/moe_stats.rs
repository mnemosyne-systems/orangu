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

//! Process-wide mixture-of-experts counters, drained by `GET /moe-stats`.
//!
//! Throughput alone cannot score work on a MoE model that does not fit in
//! RAM. A change that removes redundant expert dequantization shows up as
//! tok/s on a model small enough to sit in the page cache and as *nothing* on
//! one large enough that the disk hides it; a prefetch that improves decode
//! while fetching three times the bytes it uses looks, from tok/s, exactly
//! like one that guessed right. These counters exist so the mechanism can be
//! measured directly rather than inferred from the outcome.
//!
//! The two that carry the most information are a pair:
//!
//! - **`bytes_dequantized`** — expert weight bytes this window actually read
//!   and dequantized, as the implementation reports through
//!   [`LayerRecorder::loaded_once_per_distinct_expert`].
//! - **`bytes_unique`** — the same window's *union*: each expert counted once
//!   per layer call, however many of the batch's tokens routed to it.
//!
//! `bytes_unique` is the floor, and `union_ratio` is the first over the
//! second — `1.0` when nothing is read twice. It is deliberately **not**
//! `visits / distinct`: those are equal only for an implementation that
//! evaluates one token at a time, and a ratio derived from the routing could
//! never report an implementation that stopped doing that. The routing's own
//! redundancy — the ceiling on what grouping can save, which no
//! implementation moves — is `selection_ratio`.
//!
//! **Cost.** One `vec![false; n_expert]` and a handful of relaxed atomic adds
//! per *layer call*, plus two integer operations per `(token, expert)` pair.
//! Nothing here runs per row: the row and byte totals are multiplied out at
//! commit time from counts the recorder already holds. A MoE layer call does
//! thousands of `in_dim`-long dot products, so this is not measurable against
//! it — but "not measurable" is a claim, and BIG.md's M1 acceptance requires
//! showing it against an uninstrumented build rather than asserting it here.
//!
//! Counters are **drain-on-read**, exactly like `VulkanBackend::take_timings`
//! behind `/gpu-timings`: read once to discard the warmup, run the workload,
//! read again to get precisely that window.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::engine::loader::ExpertQuantMatrix;

/// Every counter, as one process-wide block. Relaxed ordering throughout:
/// these are statistics, and no other memory is published through them.
struct Counters {
    layer_calls: AtomicU64,
    token_slots: AtomicU64,
    visits: AtomicU64,
    distinct: AtomicU64,
    rows_dequantized: AtomicU64,
    bytes_dequantized: AtomicU64,
    bytes_unique: AtomicU64,
    budget_dropped: AtomicU64,
    budget_rescued: AtomicU64,
}

static COUNTERS: Counters = Counters {
    layer_calls: AtomicU64::new(0),
    token_slots: AtomicU64::new(0),
    visits: AtomicU64::new(0),
    distinct: AtomicU64::new(0),
    rows_dequantized: AtomicU64::new(0),
    bytes_dequantized: AtomicU64::new(0),
    bytes_unique: AtomicU64::new(0),
    budget_dropped: AtomicU64::new(0),
    budget_rescued: AtomicU64::new(0),
};

/// Records an expert-budget trim: selections dropped, and positions rescued
/// from having nothing routed at all.
///
/// `rescued` is not a curiosity. A position left with no routed expert
/// receives only the shared one, and that wrong hidden state enters the KV
/// cache — so a nonzero count means the budget is set tight enough to be
/// fighting the guard that prevents it, and is the number to watch before any
/// quality claim.
pub fn record_budget(dropped: u64, rescued: u64) {
    COUNTERS
        .budget_dropped
        .fetch_add(dropped, Ordering::Relaxed);
    COUNTERS
        .budget_rescued
        .fetch_add(rescued, Ordering::Relaxed);
}

/// The major-fault count `/proc/self/stat` reported at the previous drain, so
/// each drain can report the faults taken *in its own window* as well as the
/// process total. Zero until the first drain.
static LAST_MAJOR_FAULTS: AtomicU64 = AtomicU64::new(0);
static LAST_READ_BYTES: AtomicU64 = AtomicU64::new(0);

/// One MoE layer call's routing, accumulated as the caller routes and flushed
/// to the process counters once.
///
/// The caller reports *selections* — one [`Self::select`] per `(token,
/// expert)` pair, in whatever order it routes them — and the recorder derives
/// everything else. `rows_per_expert` and `bytes_per_expert` describe one
/// expert's full set of matrices (gate, up and down, or a fused gate/up plus
/// down), so a single selection accounts for all of that expert's weights.
///
/// Dropping a recorder without [`Self::commit`] discards it. That is
/// deliberate: a layer that returns early on an error should not contribute a
/// partial sample.
pub struct LayerRecorder {
    /// Which experts this layer call has selected at least once. Indexed by
    /// expert id; sized from the weight tensor, so an id outside it is a
    /// routing bug rather than something to grow for.
    seen: Vec<bool>,
    visits: u64,
    distinct: u64,
    /// Expert weight sets this layer call actually read and dequantized —
    /// reported by the implementation through
    /// [`Self::loaded_once_per_distinct_expert`], never derived from `visits`.
    ///
    /// The distinction is the whole point of the batch-union work. An
    /// implementation that evaluates experts one token at a time loads one
    /// set per *selection*; one that groups the batch by expert loads one per
    /// *distinct* expert; one that groups within sub-batches lands in
    /// between. A counter computed as `visits x bytes` would report all three
    /// identically — and would have reported the change that removed the
    /// redundancy as having removed nothing.
    loads: u64,
    rows_per_expert: u64,
    bytes_per_expert: u64,
}

impl LayerRecorder {
    /// A recorder for a layer whose routed experts live in `tensors` — the
    /// per-expert matrices one selection reads, which is `[gate, up, down]`
    /// on most architectures and `[gate_up, down]` where the gate and up
    /// projections are fused into one tensor (`gemma-4-26B-A4B`).
    ///
    /// Deriving the per-expert row and byte costs from the tensors
    /// themselves, rather than letting each architecture pass its own
    /// arithmetic, is the point: six call sites computing the same product by
    /// hand is six chances for one of them to be quietly wrong in a way only
    /// a cross-architecture comparison would ever reveal.
    pub fn for_tensors(tensors: &[&ExpertQuantMatrix]) -> Self {
        Self::new(
            tensors.first().map_or(0, |t| t.n_expert),
            tensors.iter().map(|t| t.out_dim as u64).sum(),
            tensors.iter().map(|t| t.expert_bytes()).sum(),
        )
    }

    pub fn new(n_expert: usize, rows_per_expert: u64, bytes_per_expert: u64) -> Self {
        Self {
            seen: vec![false; n_expert],
            visits: 0,
            distinct: 0,
            loads: 0,
            rows_per_expert,
            bytes_per_expert,
        }
    }

    /// Records that one token routed to `expert`.
    ///
    /// An out-of-range id still counts as a visit — the work happened, and
    /// the caller is about to read those weights — but cannot join the union,
    /// since there is no slot to mark. It is not an error here: this is
    /// instrumentation, and it must never be the reason a forward pass
    /// panics.
    #[inline]
    pub fn select(&mut self, expert: usize) {
        self.visits += 1;
        if let Some(seen) = self.seen.get_mut(expert)
            && !*seen
        {
            *seen = true;
            self.distinct += 1;
        }
    }

    /// Records that this layer call read each expert's weights exactly once,
    /// however many of the batch's tokens selected it — what
    /// `arch::evaluate_routed_experts` does.
    ///
    /// Derived from the recorder's own union rather than from a count the
    /// caller passes in: the two must agree, and the way to guarantee they do
    /// is to have one number, not two that a future edit could separate.
    pub fn loaded_once_per_distinct_expert(&mut self) {
        self.loads = self.distinct;
    }

    /// Adds this layer call to the process counters. `n_tokens` is the number
    /// of positions the call covered, whether or not each one routed
    /// anywhere.
    pub fn commit(self, n_tokens: usize) {
        // A layer that routed experts and reported loading none is an
        // instrumented call site that forgot `loaded` — which would report
        // zero bytes dequantized, i.e. an engine that read no weights at all.
        // Loud in a debug build; in release the zero is its own evidence.
        debug_assert!(
            self.visits == 0 || self.loads > 0,
            "a MoE layer routed {} experts but reported loading none",
            self.visits
        );
        let c = &COUNTERS;
        c.layer_calls.fetch_add(1, Ordering::Relaxed);
        c.token_slots.fetch_add(n_tokens as u64, Ordering::Relaxed);
        c.visits.fetch_add(self.visits, Ordering::Relaxed);
        c.distinct.fetch_add(self.distinct, Ordering::Relaxed);
        c.rows_dequantized
            .fetch_add(self.loads * self.rows_per_expert, Ordering::Relaxed);
        c.bytes_dequantized
            .fetch_add(self.loads * self.bytes_per_expert, Ordering::Relaxed);
        c.bytes_unique
            .fetch_add(self.distinct * self.bytes_per_expert, Ordering::Relaxed);
    }
}

/// One drained window.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MoeStats {
    /// MoE layer calls in the window. **Zero means no MoE layer ran** — a
    /// dense model, or an empty window — not "the layers moved no bytes".
    pub layer_calls: u64,
    /// Summed `n_tokens` over those calls (positions × MoE layers).
    pub token_slots: u64,
    /// `(token, expert)` pairs routed.
    pub visits: u64,
    /// Summed per-layer-call distinct expert counts — the union size, added
    /// up across calls rather than across the whole window, since an expert
    /// is only shared within one call's batch.
    pub distinct: u64,
    /// Expert weight rows dequantized (`visits` × rows per expert).
    pub rows_dequantized: u64,
    /// Expert weight bytes read and dequantized.
    pub bytes_dequantized: u64,
    /// The same bytes counted once per expert per layer call — what a
    /// batch-union implementation would move instead.
    pub bytes_unique: u64,
    /// `(token, expert)` selections the expert budget removed.
    pub budget_dropped: u64,
    /// Positions the budget would have left with nothing routed, and which
    /// were given their best expert back. Nonzero means the budget is tight
    /// enough to be fighting its own safety guard.
    pub budget_rescued: u64,
}

impl MoeStats {
    /// How many times the average expert's weights were read within one layer
    /// call, against reading each distinct expert once. `1.0` means no
    /// redundancy left; `None` when nothing was routed.
    ///
    /// Measured (`bytes_dequantized`) over the floor (`bytes_unique`), **not**
    /// `visits / distinct`. The two are the same number for an implementation
    /// that evaluates one token at a time, and only the first one moves when
    /// that stops being true — see [`LayerRecorder::loaded`]. The routing's
    /// own redundancy, which no implementation changes, is
    /// [`Self::selection_ratio`].
    pub fn union_ratio(&self) -> Option<f64> {
        (self.bytes_unique > 0).then(|| self.bytes_dequantized as f64 / self.bytes_unique as f64)
    }

    /// `visits / distinct` — how many times the average expert was *selected*
    /// within one layer call. A property of the routing and the batch size,
    /// so it is the ceiling on what grouping by expert can save, and it does
    /// not move when an implementation reaches that ceiling.
    pub fn selection_ratio(&self) -> Option<f64> {
        (self.distinct > 0).then(|| self.visits as f64 / self.distinct as f64)
    }

    /// Experts routed per position per MoE layer — `expert_used_count` when
    /// every position routes a full selection.
    pub fn experts_per_token(&self) -> Option<f64> {
        (self.token_slots > 0).then(|| self.visits as f64 / self.token_slots as f64)
    }

    /// Distinct experts touched by the average layer call. On a decode step
    /// this is at most `expert_used_count`; during prefill it grows with the
    /// batch until it saturates at `expert_count`.
    pub fn distinct_per_layer_call(&self) -> Option<f64> {
        (self.layer_calls > 0).then(|| self.distinct as f64 / self.layer_calls as f64)
    }

    pub fn to_json(self) -> serde_json::Value {
        serde_json::json!({
            "layer_calls": self.layer_calls,
            "token_slots": self.token_slots,
            "visits": self.visits,
            "distinct": self.distinct,
            "rows_dequantized": self.rows_dequantized,
            "bytes_dequantized": self.bytes_dequantized,
            "bytes_unique": self.bytes_unique,
            "budget_dropped": self.budget_dropped,
            "budget_rescued": self.budget_rescued,
            "union_ratio": self.union_ratio(),
            "selection_ratio": self.selection_ratio(),
            "experts_per_token": self.experts_per_token(),
            "distinct_per_layer_call": self.distinct_per_layer_call(),
        })
    }
}

/// Reads and resets every counter.
pub fn take() -> MoeStats {
    let c = &COUNTERS;
    MoeStats {
        layer_calls: c.layer_calls.swap(0, Ordering::Relaxed),
        token_slots: c.token_slots.swap(0, Ordering::Relaxed),
        visits: c.visits.swap(0, Ordering::Relaxed),
        distinct: c.distinct.swap(0, Ordering::Relaxed),
        rows_dequantized: c.rows_dequantized.swap(0, Ordering::Relaxed),
        bytes_dequantized: c.bytes_dequantized.swap(0, Ordering::Relaxed),
        bytes_unique: c.bytes_unique.swap(0, Ordering::Relaxed),
        budget_dropped: c.budget_dropped.swap(0, Ordering::Relaxed),
        budget_rescued: c.budget_rescued.swap(0, Ordering::Relaxed),
    }
}

/// What the kernel says this process has paid for its memory.
///
/// `major_faults` is the only honest signal that weights came from the disk
/// rather than the page cache — a streaming model's real cost, and one no
/// throughput number contains. `peak_rss_kb` is `VmHWM`, a high-water mark
/// the kernel does not let a process reset, so it is reported as-is and read
/// as "the most this process has ever held", not "the most it held during
/// this window".
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ProcessMemory {
    pub major_faults: u64,
    /// Major faults since the previous [`take_process_memory`] call — the
    /// window's own, which is what a bracketed benchmark wants.
    pub major_faults_window: u64,
    pub minor_faults: u64,
    pub rss_kb: u64,
    pub peak_rss_kb: u64,
    /// Bytes this process pulled through the block layer, from
    /// `/proc/self/io`. A page-cache hit does not count and a major fault
    /// does, so this is what a run actually cost the disk.
    ///
    /// **It is the numerator of every I/O claim in `DISK.md`, and until now
    /// it lived outside the engine.** Every one of those measurements came
    /// from an ad-hoc script reading `/proc/<pid>/io` beside the server,
    /// which meant finding the pid and keeping exactly one server alive —
    /// three contaminated runs are recorded there as the price of getting
    /// that wrong. Served from inside the process, a benchmark can bracket
    /// its own window without a pid, a second process, or root.
    ///
    /// `0` where `/proc/self/io` is unreadable. That is a weaker convention
    /// than `resident_bytes`'s `None`, and deliberate: this whole struct is
    /// already `None` on a platform without `/proc`, so the only way to get
    /// here with no value is a permission failure on Linux, where zero and
    /// unknown are equally uninformative.
    pub read_bytes: u64,
    /// Bytes read since the previous [`take_process_memory`] call. The same
    /// window as `major_faults_window`, and the one figure on a streamed
    /// model that is a **count** rather than a rate — which is why it
    /// survives the drive drift that invalidates timings on this hardware.
    pub read_bytes_window: u64,
}

impl ProcessMemory {
    pub fn to_json(self) -> serde_json::Value {
        serde_json::json!({
            "major_faults": self.major_faults,
            "major_faults_window": self.major_faults_window,
            "minor_faults": self.minor_faults,
            "rss_kb": self.rss_kb,
            "peak_rss_kb": self.peak_rss_kb,
            "read_bytes": self.read_bytes,
            "read_bytes_window": self.read_bytes_window,
        })
    }
}

/// Current fault and RSS figures, with the major-fault delta since the last
/// call. `None` where `/proc` is not available or does not parse — reported
/// as absent rather than as zeros, so "this platform cannot tell you" is not
/// mistaken for "nothing went to disk".
pub fn take_process_memory() -> Option<ProcessMemory> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    let (minor_faults, major_faults) = parse_faults(&stat)?;
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let previous = LAST_MAJOR_FAULTS.swap(major_faults, Ordering::Relaxed);
    let read_bytes = std::fs::read_to_string("/proc/self/io")
        .ok()
        .and_then(|io| parse_read_bytes(&io))
        .unwrap_or(0);
    let previous_read = LAST_READ_BYTES.swap(read_bytes, Ordering::Relaxed);
    Some(ProcessMemory {
        major_faults,
        major_faults_window: major_faults.saturating_sub(previous),
        minor_faults,
        rss_kb: parse_status_kb(&status, "VmRSS:").unwrap_or(0),
        peak_rss_kb: parse_status_kb(&status, "VmHWM:").unwrap_or(0),
        read_bytes,
        read_bytes_window: read_bytes.saturating_sub(previous_read),
    })
}

/// `read_bytes` from a `/proc/<pid>/io` block.
///
/// Named-key lookup rather than a field index, because this file has no fixed
/// ordering guarantee and carries a `rchar` line whose name is a prefix of
/// nothing but whose *value* is wildly different — `rchar` counts every byte
/// read through a syscall including page-cache hits, which is the number this
/// is specifically not.
fn parse_read_bytes(io: &str) -> Option<u64> {
    io.lines()
        .find_map(|line| line.strip_prefix("read_bytes:"))
        .and_then(|value| value.trim().parse().ok())
}

/// `(minflt, majflt)` from a `/proc/<pid>/stat` line.
///
/// Fields are split *after the last `)`* rather than on whitespace from the
/// start: field 2 is the executable name in parentheses and may itself
/// contain spaces and parentheses, so a plain `split_whitespace().nth(11)`
/// silently reads the wrong number for any binary whose name has a space in
/// it — a wrong count that still looks like a plausible count.
fn parse_faults(stat: &str) -> Option<(u64, u64)> {
    let after_comm = &stat[stat.rfind(')')? + 1..];
    // `after_comm` starts at field 3 (state), so field 10 (minflt) is index 7
    // and field 12 (majflt) is index 9.
    let mut fields = after_comm.split_whitespace().skip(7);
    let minor = fields.next()?.parse().ok()?;
    let major = fields.nth(1)?.parse().ok()?;
    Some((minor, major))
}

/// The kibibyte value of a `/proc/self/status` line such as `VmHWM:  1234 kB`.
fn parse_status_kb(status: &str, key: &str) -> Option<u64> {
    status
        .lines()
        .find(|line| line.starts_with(key))
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|value| value.parse().ok())
}

#[cfg(test)]
mod tests {

    /// `rchar` is the trap: it sits in the same file, counts every byte read
    /// through a syscall *including page-cache hits*, and is typically an
    /// order of magnitude larger. A field-index parse or a sloppy prefix
    /// match picks it up and every I/O claim built on it is wrong in the
    /// direction that looks like a discovery.
    #[test]
    fn read_bytes_is_not_rchar() {
        let io = "rchar: 1627049701\n\
                  wchar: 4096\n\
                  syscr: 199034\n\
                  syscw: 12\n\
                  read_bytes: 20725362688\n\
                  write_bytes: 0\n\
                  cancelled_write_bytes: 0\n";
        assert_eq!(parse_read_bytes(io), Some(20_725_362_688));
    }

    #[test]
    fn an_io_block_without_read_bytes_reports_nothing() {
        assert_eq!(parse_read_bytes("rchar: 5\nwchar: 6\n"), None);
        assert_eq!(parse_read_bytes(""), None);
        assert_eq!(parse_read_bytes("read_bytes: not-a-number\n"), None);
    }
    use super::*;

    /// The counters are process-wide, so tests that drain them cannot run
    /// concurrently with each other. Rust runs a test binary's tests in
    /// parallel threads by default, and two draining tests interleaving would
    /// each see the other's numbers — a flake that reads as a counting bug.
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Takes the serial lock, ignoring poisoning. One failing test must not
    /// turn every other draining test into a `PoisonError` — three failures
    /// where there is one bug is a worse report than one.
    fn serial() -> std::sync::MutexGuard<'static, ()> {
        SERIAL.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The union is what a batch-union implementation would move; the visit
    /// count is what today's per-token loop actually moves. The whole point
    /// of the pair is that they differ, so check both against a selection
    /// whose answer is obvious by inspection.
    #[test]
    fn distinct_counts_each_expert_once_however_many_tokens_routed_to_it() {
        let mut rec = LayerRecorder::new(8, 10, 100);
        // Three tokens, two experts each: {0,1}, {1,2}, {0,1}.
        for expert in [0, 1, 1, 2, 0, 1] {
            rec.select(expert);
        }
        assert_eq!(rec.visits, 6);
        assert_eq!(rec.distinct, 3, "experts 0, 1 and 2 were touched");
    }

    /// The work counters follow what the implementation says it *loaded*, not
    /// what the router *selected*. Two tokens routing to the same expert cost
    /// one weight read, and the counters must say so — this is the whole
    /// reason `loads` is a field rather than `visits` multiplied out.
    #[test]
    fn the_work_counters_follow_the_loads_not_the_selections() {
        let _guard = serial();
        let _ = take();

        let mut rec = LayerRecorder::new(4, 10, 100);
        for expert in [0, 1, 0] {
            rec.select(expert);
        }
        rec.loaded_once_per_distinct_expert();
        rec.commit(2);

        let stats = take();
        assert_eq!(stats.layer_calls, 1);
        assert_eq!(stats.token_slots, 2);
        assert_eq!(stats.visits, 3, "three (token, expert) selections");
        assert_eq!(stats.distinct, 2);
        assert_eq!(stats.rows_dequantized, 20, "2 loads x 10 rows, not 3");
        assert_eq!(stats.bytes_dequantized, 200, "2 loads x 100 bytes, not 3");
        assert_eq!(stats.bytes_unique, 200, "2 distinct x 100 bytes");
        // At the floor: everything read was read once.
        assert_eq!(stats.union_ratio(), Some(1.0));
        // The routing's own redundancy is unchanged by any of this.
        assert_eq!(stats.selection_ratio(), Some(1.5));
        assert_eq!(stats.experts_per_token(), Some(1.5));
        assert_eq!(stats.distinct_per_layer_call(), Some(2.0));
    }

    /// `union_ratio` has to be able to report redundancy, or it cannot score
    /// its removal. A recorder told it loaded once per selection — what every
    /// architecture did before the batch union — must report the selections'
    /// full redundancy.
    #[test]
    fn a_loader_that_reads_once_per_selection_reports_the_redundancy() {
        let _guard = serial();
        let _ = take();

        let mut rec = LayerRecorder::new(4, 10, 100);
        for expert in [0, 1, 0] {
            rec.select(expert);
        }
        // Not `loaded_once_per_distinct_expert`: three reads for three
        // selections, the pre-batch-union behaviour.
        rec.loads = rec.visits;
        rec.commit(2);

        let stats = take();
        assert_eq!(stats.rows_dequantized, 30, "3 loads x 10 rows");
        assert_eq!(stats.bytes_dequantized, 300);
        assert_eq!(stats.bytes_unique, 200);
        assert_eq!(stats.union_ratio(), Some(1.5), "50% wasted re-reads");
        assert_eq!(stats.selection_ratio(), Some(1.5));
    }

    /// Drain-on-read is the whole contract a bracketed benchmark rests on: if
    /// a second read returned the first window's numbers again, every "after"
    /// measurement would include its "before".
    #[test]
    fn taking_the_counters_resets_them() {
        let _guard = serial();
        let _ = take();

        let mut rec = LayerRecorder::new(4, 1, 1);
        rec.select(0);
        rec.loaded_once_per_distinct_expert();
        rec.commit(1);
        assert_eq!(take().visits, 1);
        assert_eq!(take(), MoeStats::default());
    }

    /// A layer that returns early must not contribute a partial sample.
    #[test]
    fn a_dropped_recorder_contributes_nothing() {
        let _guard = serial();
        let _ = take();

        let mut rec = LayerRecorder::new(4, 1, 1);
        rec.select(0);
        drop(rec);
        assert_eq!(take(), MoeStats::default());
    }

    /// Instrumentation must never be the reason a forward pass panics.
    #[test]
    fn an_out_of_range_expert_counts_as_work_without_panicking() {
        let mut rec = LayerRecorder::new(2, 1, 1);
        rec.select(9);
        assert_eq!(rec.visits, 1);
        assert_eq!(rec.distinct, 0, "it cannot join a union it has no slot in");
    }

    /// Ratios describe nothing when nothing was routed, and must say so
    /// rather than divide by zero into a `NaN` that serializes as `null`
    /// anyway but arrives via a floating-point accident.
    #[test]
    fn an_empty_window_reports_no_ratios() {
        let stats = MoeStats::default();
        assert_eq!(stats.union_ratio(), None);
        assert_eq!(stats.experts_per_token(), None);
        assert_eq!(stats.distinct_per_layer_call(), None);
    }

    /// The field this exists to get right: a `comm` containing both a space
    /// and a closing parenthesis, which `split_whitespace` alone gets wrong
    /// while still returning a plausible-looking number.
    #[test]
    fn faults_are_read_past_an_awkward_executable_name() {
        let stat = "42 (my server) (x) S 1 42 42 0 -1 4194304 111 0 222 0 \
                    13 14 15 16 17 18 19 20";
        assert_eq!(parse_faults(stat), Some((111, 222)));
    }

    #[test]
    fn faults_from_an_ordinary_stat_line() {
        let stat = "7 (orangu-server) S 1 7 7 0 -1 4194560 9012 0 34 0 1 2 3 4 20 0 5 0";
        assert_eq!(parse_faults(stat), Some((9012, 34)));
    }

    #[test]
    fn a_truncated_stat_line_reports_nothing_rather_than_guessing() {
        assert_eq!(parse_faults("42 (x) S 1 2 3"), None);
        assert_eq!(parse_faults("no parenthesis here"), None);
    }

    #[test]
    fn status_values_are_read_in_kibibytes() {
        let status = "Name:\torangu-server\nVmRSS:\t  123456 kB\nVmHWM:\t  234567 kB\n";
        assert_eq!(parse_status_kb(status, "VmRSS:"), Some(123_456));
        assert_eq!(parse_status_kb(status, "VmHWM:"), Some(234_567));
        assert_eq!(parse_status_kb(status, "VmNope:"), None);
    }

    /// The process block is only meaningful on Linux; everywhere else it must
    /// be absent rather than a row of zeros.
    #[test]
    #[cfg(target_os = "linux")]
    fn the_running_process_reports_its_own_memory() {
        let memory = take_process_memory().expect("/proc/self is readable on Linux");
        assert!(memory.rss_kb > 0);
        assert!(memory.peak_rss_kb >= memory.rss_kb);
    }
}
