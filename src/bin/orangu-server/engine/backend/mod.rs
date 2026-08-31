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

//! The seam a GPU backend plugs into without touching the model code in
//! `engine::arch`: everything the forward pass needs from "the thing that
//! actually multiplies matrices" is this one trait.
//!
//! Implementors: `CpuBackend` (scalar with runtime AVX2 dispatch, always
//! available); `VulkanBackend` (compute shaders via `wgpu`, the most
//! mature GPU backend — real fused attention/RoPE/layer submissions,
//! verified against real AMD hardware, see its own module doc); `metal`'s
//! `MetalBackend` (that *same* engine and the same WGSL kernels, brought
//! up on `wgpu`'s Metal backend for Apple GPUs — not a reimplementation,
//! see its module doc); `cuda`'s
//! `CudaBackend` and `opencl`'s `OpenClBackend` (both dlopen their vendor
//! library at runtime, same as `wgpu`, so both are always compiled in);
//! `rocm`'s `RocmBackend` (behind the `rocm` Cargo feature, off by
//! default — the one exception, since `cubecl-hip-sys` hard-links a vendor
//! library at *build* time, see that module's own doc comment for why).
//! `CudaBackend`/`OpenClBackend`/`RocmBackend` are each a real but
//! smaller-scoped `matmul`-only implementation — see their module docs for
//! exactly what's ported and what isn't; `MetalBackend` is the one that
//! is at full parity, because it is the same code.
//!
//! Earlier revisions of this file claimed AMD GPUs are reached only through
//! `VulkanBackend` (Mesa/RADV implements Vulkan on AMD hardware directly)
//! and that there was no separate ROCm/HIP backend. That's still true for
//! Vulkan/RADV as a *path* to AMD hardware — it's real, verified, and the
//! default `auto` backend selection still prefers it — but `rocm::
//! RocmBackend` now also exists as a genuine, separate HIP-based backend
//! for when it's specifically asked for (`backend = rocm`).

pub mod cpu;
pub mod cuda;
pub mod device;
pub mod metal;
pub mod multi;
pub mod opencl;
#[cfg(feature = "rocm")]
pub mod rocm;
pub mod vendor_shaders;
pub mod vulkan;
pub mod vulkan_replay;
pub mod vulkan_shaders;

pub use cpu::CpuBackend;
pub use cuda::CudaBackend;
pub use device::{DeviceCandidate, DeviceError, DeviceErrorKind, DeviceRequest};
pub use metal::MetalBackend;
pub use multi::MultiDeviceBackend;
pub use opencl::OpenClBackend;
#[cfg(feature = "rocm")]
pub use rocm::RocmBackend;
pub use vulkan::VulkanBackend;

use super::loader::QuantMatrix;

/// One tuning knob's value from the environment, **saying so when it is
/// rejected** rather than quietly falling back.
///
/// The silence is what makes this worth a helper. A benchmark sweep sets the
/// variable, the backend declines the value and runs the default, and the
/// sweep records a duplicate of the default *under the rejected value's
/// name* — two identical configurations reported as two distinct points,
/// which reads as "this knob does nothing in that range" rather than "that
/// value was never tried". It costs a whole sweep to notice, and only if
/// someone reads the startup tuning line closely enough to see the value did
/// not change. That happened, on the first sweep run through this code.
///
/// Same precedent as `VulkanBackend::try_init`'s `ORANGU_GPU_TIMESTAMPS`
/// warning and `vulkan_shaders::coop_geom`'s malformed-spec one: a knob that
/// silently does nothing is worse than one that is absent, because it invites
/// the reader to conclude the thing it controls does not matter.
///
/// Lives here, above both `vulkan` and `vulkan_shaders`, because the knobs
/// that need it are split across the two — the ones that pick a dispatch
/// threshold and the ones that get baked into generated WGSL.
pub(crate) fn env_tuning_value<T>(
    var: &str,
    default: T,
    expected: &str,
    valid: impl Fn(T) -> bool,
) -> T
where
    T: std::str::FromStr + std::fmt::Display + Copy,
{
    let Ok(raw) = std::env::var(var) else {
        return default;
    };
    match raw.trim().parse::<T>() {
        Ok(v) if valid(v) => v,
        // Both arms, not just the parse failure: `ORANGU_NORM_WG=32` parses
        // perfectly and is still not a value that kernel has, and that is the
        // case a sweep actually hits.
        _ => {
            eprintln!(
                "orangu-server: {var}={raw:?} is not {expected} — using {default}. \
                 This run measures the default, not the value you asked for."
            );
            default
        }
    }
}

/// Weight bytes below which a **decode** matmul is faster on the host than on
/// the device, from `ORANGU_HOST_MATMUL_KIB`. `0` switches the rule off, and
/// is the control arm a sweep needs.
///
/// A device matmul costs a large fixed amount before it computes anything —
/// building a command buffer, handing it to the driver, waiting for the device
/// to pick it up — and that cost is several times the host's. So the two
/// backends' costs are two lines, `fixed + bytes / bandwidth`, and they cross:
/// above the crossing the device's bandwidth wins, below it the device's fixed
/// cost does. A decode step is nothing but small matmuls, so on a model whose
/// per-layer tensors are small this is not a rounding error — it was most of
/// the token.
///
/// **Measure it on the machine that will run it.** `decode_matvec_
/// fixed_cost_gpu_versus_cpu` (`engine::backend::vulkan::decode_matvec_probe`)
/// prints the crossing directly, and the default here is a measured value on
/// one card — a starting point rather than an answer, exactly as
/// `attention::min_gpu_tokens` and the coop-tile threshold are.
pub(crate) fn host_matmul_threshold_bytes() -> usize {
    static BYTES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *BYTES.get_or_init(|| {
        env_tuning_value(
            "ORANGU_HOST_MATMUL_KIB",
            HOST_MATMUL_KIB_DEFAULT,
            "a size in KiB (0 disables the rule)",
            |kib| kib <= 1024 * 1024,
        ) * 1024
    })
}

/// The default for [`host_matmul_threshold_bytes`], in KiB.
///
/// The probe puts the crossing between 5.3 and 7.7 MiB on this project's card,
/// and an end-to-end sweep of this constant agrees with it — which is the
/// point, since the two measure different things and only one of them is a
/// prediction:
///
/// | KiB | decode |
/// | --: | --: |
/// | 0 (rule off) | 9.72 tok/s |
/// | 4096 | 9.40 |
/// | 6144 | **10.02–10.18** |
/// | 8192 | 10.11 |
/// | 20480 | 9.60 |
///
/// The curve is **flat across the crossing and steep outside it**: 6 and 8 MiB
/// are within 1% of each other, while 4 MiB gives up 6% by leaving a 4.5 MiB
/// projection on the device and 20 MiB gives up 5% by taking a 17.7 MiB one
/// off it. So the default sits inside the measured crossing rather than at
/// either edge — being a megabyte wrong in either direction costs nothing,
/// and being several megabytes wrong costs about as much as the rule is worth.
///
/// Swept with `--delay`, and that is not incidental: an earlier sweep of the
/// same values on a card that had been benchmarked for an hour reported the
/// rate rising all the way to 20 MiB, which is the opposite conclusion. Heat,
/// not the threshold.
#[cfg(not(test))]
const HOST_MATMUL_KIB_DEFAULT: usize = 6144;

/// **Zero under `cfg(test)`**, which is not a detail: this module's sibling
/// tests cross-check the device's kernels against `CpuBackend`, and every test
/// weight is small. Left at the production default they would route to the
/// host, compare the host against itself, and pass while testing nothing — a
/// cross-check that shares the component under test proves nothing. Six of
/// them said so out loud when this landed.
///
/// Only the *default* changes. An explicit `ORANGU_HOST_MATMUL_KIB` still
/// wins, so a real-model test can ask for the production rule.
#[cfg(test)]
const HOST_MATMUL_KIB_DEFAULT: usize = 0;

/// Whether a batch of matmuls is small enough to be worth running on the host
/// instead of the device — see [`host_matmul_threshold_bytes`].
///
/// Two conditions beyond the size, both of them about the comparison staying
/// the one that was measured:
///
/// - **One token.** The crossing was measured at `n_tokens == 1`, which is
///   what a decode step is. A wider batch reads the same weights and does more
///   arithmetic with them, which is the regime the device was built for, and
///   the fixed cost it is being charged is unchanged — so the crossing moves
///   away, and this rule has no business guessing how far.
/// - **The whole batch together.** The fixed cost is per submission, not per
///   op, so a batch of eight small matmuls pays it once. What is compared
///   against the threshold is therefore the batch's total, not any one op's:
///   splitting a batch up to send half of it to the host would pay the fixed
///   cost anyway and lose the amortization that made it cheap.
pub(crate) fn prefers_host_matmul(ops: &[MatmulOp<'_>]) -> bool {
    prefers_host_matmul_with(host_matmul_threshold_bytes(), ops)
}

/// The fewest output rows a thread-pool task will take, from
/// `ORANGU_MATMUL_MIN_ROWS`.
///
/// A decode matmul fans out over its output rows, and the natural expression
/// of that is one row per item — which leaves `rayon` free to split until each
/// task is a single dot product. On this model a routed expert's row is
/// 288–1152 bytes, so a task is a few hundred nanoseconds of arithmetic
/// wrapped in a job that has to be created, possibly stolen, and joined.
///
/// The cost is not hypothetical and it grew: once
/// [`host_matmul_threshold_bytes`] moved the small matmuls to the host, a
/// decode step's parallel regions multiplied, and **40% of all CPU samples
/// were pool machinery — `find_work`, `steal`, `join_context`, and 24% in
/// `sched_yield` alone**. Those are cores spinning to discover there is
/// nothing to steal.
///
/// `with_min_len` is a floor, not a chunk size: `rayon` still splits down to it
/// and no further, so a wide projection keeps every worker and a narrow one
/// stops manufacturing jobs smaller than the overhead of having one.
///
/// **A flat floor, and it was worth measuring rather than reasoning about.**
/// The obvious refinement is to scale it with the projection and the pool —
/// `out_dim / threads`, so a narrow projection never leaves a worker idle.
/// That was tried, and it is **6.7% slower than no floor at all**, while the
/// flat 64 below is **3.7% faster** than no floor; both against the same
/// control, three matched pairs each, every pair agreeing.
///
/// The reason is that this function cannot see the parallelism *around* it. A
/// routed expert's projection already sits inside an eight-wide fan-out over
/// experts, so eight tasks per projection is sixty-four in flight, not eight.
/// Scaling the floor down to "fill the pool from this call alone" manufactures
/// work the pool did not need.
///
/// Measured on `Ornith-1.5-35B-A3B`: a floor of 1 is the worst setting in
/// every sweep, 32 and 64 are the best and are not separable from each other
/// here (two adjacent points at one setting differed by 14%), 256 is worse
/// again.
///
/// **The risk this leaves.** On a much wider pool, a narrow projection with no
/// outer fan-out around it would get few tasks and leave workers idle. That
/// case was not measurable here, which is what the environment variable is
/// for — this is a machine-dependent constant like every other threshold in
/// this module.
///
/// **Bit-identical at any value.** Each output row is computed and written
/// independently, so this changes which worker does what and nothing else.
/// `ORANGU_MATMUL_MIN_ROWS=1` restores one-row-per-task splitting and is the
/// control arm.
pub(crate) fn matmul_min_rows() -> usize {
    static ROWS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ROWS.get_or_init(|| {
        env_tuning_value(
            "ORANGU_MATMUL_MIN_ROWS",
            MATMUL_MIN_ROWS_DEFAULT,
            "a row count of at least 1 (1 disables the floor)",
            |n| n >= 1,
        )
    })
}

/// The default for [`matmul_min_rows`].
const MATMUL_MIN_ROWS_DEFAULT: usize = 64;

/// The same decision from the weights alone.
///
/// For a caller that has to know **where a matmul will run before it builds
/// the call** — which is what deciding whether two branches are on different
/// processors requires, and that decision has to be made before either branch
/// starts. See `arch::moe_overlap`.
pub(crate) fn weights_prefer_host_matmul(n_tokens: usize, weights: &[&QuantMatrix]) -> bool {
    let threshold = host_matmul_threshold_bytes();
    if threshold == 0 || weights.is_empty() || n_tokens != 1 {
        return false;
    }
    let mut total = 0usize;
    for w in weights {
        total = total.saturating_add(w.row_bytes().saturating_mul(w.out_dim));
        if total >= threshold {
            return false;
        }
    }
    true
}

/// [`prefers_host_matmul`] against an explicit threshold.
///
/// Split out so the decision can be tested: the threshold itself is read from
/// the environment once per process, which a test cannot vary and must not
/// depend on.
fn prefers_host_matmul_with(threshold: usize, ops: &[MatmulOp<'_>]) -> bool {
    if threshold == 0 || ops.is_empty() {
        return false;
    }
    if ops.iter().any(|op| op.n_tokens != 1) {
        return false;
    }
    let mut total = 0usize;
    for op in ops {
        total = total.saturating_add(op.w.row_bytes().saturating_mul(op.w.out_dim));
        if total >= threshold {
            return false;
        }
    }
    true
}

/// One `matmul` call's operands, for [`Backend::matmul_batch`] — a slice of
/// these describes several matmuls that don't depend on each other's
/// results (e.g. a transformer layer's Q/K/V projections, all reading the
/// same normed input) and so can be issued together.
pub struct MatmulOp<'a> {
    pub x: &'a [f32],
    pub n_tokens: usize,
    pub w: &'a QuantMatrix,
}

/// The shared token cap for one multi-token backend phase, across every
/// backend. `wgpu` backends may clamp it upward to their own kernel
/// crossover where needed, but this is the one default policy the model
/// code and the non-`wgpu` backends all share.
pub(crate) const MAX_MULTI_TOKEN_PHASE_TOKENS_DEFAULT: usize = 256;

/// `ORANGU_MAX_TOKENS_PER_SUBMISSION`, shared above every backend-specific
/// implementation.
///
/// The name comes from the original Vulkan-only tuning, but the intent is
/// backend-agnostic now: one bound on how many prompt tokens any single
/// multi-token backend phase should process at once unless a backend has a
/// stricter floor of its own.
pub(crate) fn max_multi_token_phase_tokens() -> usize {
    static MAX: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MAX.get_or_init(|| {
        std::env::var("ORANGU_MAX_TOKENS_PER_SUBMISSION")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|n| *n > 0)
            .unwrap_or(MAX_MULTI_TOKEN_PHASE_TOKENS_DEFAULT)
    })
}

/// Runs one prefill-style matmul op in token stripes when it exceeds the
/// shared multi-token phase cap, concatenating the per-stripe outputs back to
/// `[n_tokens, out_dim]`.
pub(crate) fn guarded_matmul_op(
    op: &MatmulOp<'_>,
    mut run: impl FnMut(&[f32], usize, &QuantMatrix) -> Vec<f32>,
) -> Vec<f32> {
    let max_tokens = max_multi_token_phase_tokens();
    if op.n_tokens <= max_tokens {
        return run(op.x, op.n_tokens, op.w);
    }

    let in_dim = op.w.in_dim;
    let mut out = Vec::with_capacity(op.n_tokens * op.w.out_dim);
    let mut start = 0usize;
    while start < op.n_tokens {
        let end = (start + max_tokens).min(op.n_tokens);
        out.extend(run(&op.x[start * in_dim..end * in_dim], end - start, op.w));
        start = end;
    }
    out
}

/// [`guarded_matmul_op`]'s out-parameter form, for callers that own the
/// destination buffer. The striped path also reuses **one** stripe buffer
/// across the whole loop rather than taking a fresh `Vec` per stripe.
pub(crate) fn guarded_matmul_op_into(
    out: &mut Vec<f32>,
    op: &MatmulOp<'_>,
    mut run: impl FnMut(&mut Vec<f32>, &[f32], usize, &QuantMatrix),
) {
    let max_tokens = max_multi_token_phase_tokens();
    if op.n_tokens <= max_tokens {
        run(out, op.x, op.n_tokens, op.w);
        return;
    }

    let in_dim = op.w.in_dim;
    out.clear();
    out.reserve(op.n_tokens * op.w.out_dim);
    let mut stripe = Vec::new();
    let mut start = 0usize;
    while start < op.n_tokens {
        let end = (start + max_tokens).min(op.n_tokens);
        run(
            &mut stripe,
            &op.x[start * in_dim..end * in_dim],
            end - start,
            op.w,
        );
        out.extend_from_slice(&stripe);
        start = end;
    }
}

/// How many bytes of *transient* device memory one batched submission may
/// hold at once — the per-op `x` uploads and `y` outputs, not the weights,
/// which are cached and shared.
///
/// A batched `matmul_batch` exists to stop synchronizing once per op, and the
/// way it does that is to keep every op's buffers alive until the last kernel
/// has been launched. That trades synchronizations for peak memory, and
/// without a bound the trade is unbounded: `evaluate_routed_experts_batched_
/// views` can hand this function two ops per routed expert group, which on a
/// wide MoE layer is a few hundred ops at once. A card that was serving the
/// model fine op-by-op would start failing to allocate.
///
/// 256 MiB is chosen to be small next to any device this backend runs on and
/// large next to a single layer's activations, so the common case is one
/// chunk and the pathological case still completes.
pub(crate) const BATCH_DEVICE_BUDGET_BYTES: usize = 256 << 20;

/// One op's token stripe: `ops[op]` rows `start..start + n_tokens`.
///
/// [`guarded_matmul_op`] applies the shared multi-token phase cap by running
/// a long op in stripes; a *batched* backend needs the same cap but has to
/// know the stripes up front, because it launches every stripe before reading
/// any of them back. This is that decomposition, made explicit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BatchStripe {
    pub op: usize,
    pub start: usize,
    pub n_tokens: usize,
}

/// Every stripe of `ops`, in order, and the chunk boundaries a backend should
/// synchronize at.
///
/// Returns `(stripes, chunks)`, where each `chunks[i]` indexes into `stripes`
/// and names a group that fits inside [`BATCH_DEVICE_BUDGET_BYTES`]. A
/// backend uploads and launches one chunk's stripes, then reads that chunk
/// back — one synchronization point per chunk instead of one per op.
///
/// A single stripe larger than the budget gets its own chunk rather than
/// being split further: the phase cap has already bounded its token count,
/// and refusing to run it would be worse than exceeding a self-imposed
/// budget by one op.
pub(crate) fn plan_batch(
    ops: &[MatmulOp<'_>],
    budget_bytes: usize,
) -> (Vec<BatchStripe>, Vec<std::ops::Range<usize>>) {
    let max_tokens = max_multi_token_phase_tokens();
    let mut stripes = Vec::with_capacity(ops.len());
    for (op_index, op) in ops.iter().enumerate() {
        let mut start = 0usize;
        // An op with no tokens still gets one empty stripe, so every op has
        // at least one output to concatenate and the caller's result vector
        // lines up with `ops` positionally.
        loop {
            let n_tokens = max_tokens.min(op.n_tokens.saturating_sub(start));
            stripes.push(BatchStripe {
                op: op_index,
                start,
                n_tokens,
            });
            start += n_tokens.max(1);
            if start >= op.n_tokens {
                break;
            }
        }
    }

    let bytes_of = |s: &BatchStripe| -> usize {
        let op = &ops[s.op];
        let f32_bytes = std::mem::size_of::<f32>();
        s.n_tokens * (op.w.in_dim + op.w.out_dim) * f32_bytes
    };
    let mut chunks = Vec::new();
    let mut chunk_start = 0usize;
    let mut chunk_bytes = 0usize;
    for (i, stripe) in stripes.iter().enumerate() {
        let bytes = bytes_of(stripe);
        if i > chunk_start && chunk_bytes + bytes > budget_bytes {
            chunks.push(chunk_start..i);
            chunk_start = i;
            chunk_bytes = 0;
        }
        chunk_bytes += bytes;
    }
    if chunk_start < stripes.len() {
        chunks.push(chunk_start..stripes.len());
    }
    (stripes, chunks)
}

/// [`Backend::reduced_surface`]'s answer for the three vendor backends that
/// implement [`Backend::matmul`] and nothing else.
///
/// One constant rather than one string per backend, because the limitation is
/// one fact about three modules and three copies of it would be three things
/// to keep in step. `CudaBackend`, `RocmBackend` and `OpenClBackend` are the
/// same scope by construction — each is a port of the same
/// `MAIN_REDUCE_SUFFIX` dispatch into a different kernel language — so a
/// wording that had drifted between them would be describing a difference
/// that does not exist.
pub(crate) const MATMUL_ONLY_SURFACE: &str =
    "matmul only - no fused layer chain, GPU attention or GPU sampling";

pub trait Backend: Send + Sync {
    /// Whether one submission to this backend is subject to a driver
    /// timeout — a limit on how long a *single* forward pass may run before
    /// the device is reset out from under it.
    ///
    /// `engine::generate`'s prefill chunker exists to stay inside that limit,
    /// and it sizes each chunk by dividing the last chunk's wall time by its
    /// token count. That quotient is a per-token rate only when the model is
    /// resident. When the weights are streamed from disk each pass pays a
    /// largely *fixed* cost — the model is read once per pass whatever the
    /// pass contains — so a narrow chunk reports a huge apparent rate, the
    /// sizer narrows the next one, and the fixed cost is paid again over
    /// fewer tokens. Measured, that spiral turned one pass and 6.4 GiB into
    /// 39 passes and 134 GiB.
    ///
    /// **The default is `true`, and it has to be.** Neither
    /// [`Backend::as_wgpu`] nor `ModelForward::vulkan_backend` can answer this
    /// question: both are `None` for the CPU backend *and* for the CUDA,
    /// OpenCL and ROCm ones, which very much do reset. A backend that says
    /// nothing therefore keeps the chunker exactly as it is today; only one
    /// that can prove it has no device to lose overrides this.
    fn has_submission_timeout(&self) -> bool {
        true
    }

    /// `y[t, o] = sum_i x[t, i] * w.row(o)[i]` — `x` is `[n_tokens,
    /// w.in_dim]`, `y` is `[n_tokens, w.out_dim]`. `w`'s rows are
    /// dequantized on demand, not pre-materialized.
    fn matmul(&self, x: &[f32], n_tokens: usize, w: &QuantMatrix) -> Vec<f32>;

    /// [`Backend::matmul`] into a caller-owned buffer, so a projection run
    /// once per layer allocates once per *forward pass* instead.
    ///
    /// The default is the allocating form, which makes this a strictly
    /// additive method: every backend keeps working without implementing
    /// it, and every call site can move over independently. Only a backend
    /// that can genuinely write into a supplied buffer — `CpuBackend`, which
    /// builds the result on the host — gains anything by overriding it.
    ///
    /// **Why it is worth having.** A projection's output at prefill widths
    /// is larger than the hidden state, not smaller: an FFN buffer is
    /// `n_tokens * n_ff`, allocated and freed once per layer per forward
    /// pass. Reusing one buffer across layers removes that bookkeeping, and
    /// in `guarded_matmul_op_into`'s striped path it also collapses one
    /// allocation per stripe into one for the whole loop.
    ///
    /// It is **not** an `mmap`-traffic argument, though it looks like one:
    /// `ORANGU_PREFILL_BATCH` caps a forward pass at 512 tokens and
    /// `MAX_MULTI_TOKEN_PHASE_TOKENS_DEFAULT` stripes a batched op at 64, so
    /// the largest buffer any single call produces stays under the 32 MiB
    /// ceiling `glibc`'s adaptive `mmap` threshold can reach, and is
    /// therefore recyclable from the heap. Raise either knob and that stops
    /// being true.
    ///
    /// `out` is resized, not cleared: the kernels write every element, so
    /// zeroing first would be a wasted pass. Callers must not assume
    /// anything about `out`'s contents on entry.
    fn matmul_into(&self, out: &mut Vec<f32>, x: &[f32], n_tokens: usize, w: &QuantMatrix) {
        *out = self.matmul(x, n_tokens, w);
    }

    /// [`Backend::matmul_decode`]'s out-parameter form — see
    /// [`Backend::matmul_into`], and [`Backend::matmul_decode`] for why
    /// decode needs its own entry point at all.
    fn matmul_decode_into(&self, out: &mut Vec<f32>, x: &[f32], n_tokens: usize, w: &QuantMatrix) {
        *out = self.matmul_decode(x, n_tokens, w);
    }

    /// Runs several *independent* matmuls (no result of one feeds another
    /// — see [`MatmulOp`]) as a batch, returning results in the same
    /// order. The default implementation also enforces the shared multi-token
    /// phase cap by splitting a long op into token stripes before calling
    /// `matmul`; only a backend that actually benefits from batching (a GPU
    /// backend, which can submit one command buffer and block on it once
    /// instead of once per op) needs to override it. `CpuBackend` doesn't:
    /// its `matmul` is already parallelized internally and has no per-call
    /// dispatch/sync overhead to amortize.
    fn matmul_batch(&self, ops: &[MatmulOp<'_>]) -> Vec<Vec<f32>> {
        ops.iter()
            .map(|op| guarded_matmul_op(op, |x, n_tokens, w| self.matmul(x, n_tokens, w)))
            .collect()
    }

    /// The same product as [`Backend::matmul`], for a **decode** batch:
    /// `x`'s rows are one per *sequence* rather than one per position in a
    /// prompt.
    ///
    /// It exists because `n_tokens > 1` is not, on its own, a safe signal
    /// to switch kernels. `CpuBackend` picks its kernel from `n_tokens`,
    /// and the multi-token (GEMM) kernels are *not* bit-identical to the
    /// single-token (GEMV) one — they sum in a different order, and the
    /// K-quant GEMM quantizes activations one scale per 256 elements
    /// against the GEMV's one per 32. Both stay inside `engine::vecdot`'s
    /// error budget, but routing a decode batch through them would make a
    /// sequence's logits depend on how many *other* sequences happened to
    /// be decoding alongside it in that window — the same token, decoded
    /// alone or in a busy batch, would come out differently. Prefill has
    /// no such twin to agree with and keeps the faster GEMM.
    ///
    /// The default is `matmul` itself, which is right for every backend
    /// whose kernels don't vary with `n_tokens`.
    fn matmul_decode(&self, x: &[f32], n_tokens: usize, w: &QuantMatrix) -> Vec<f32> {
        self.matmul(x, n_tokens, w)
    }

    /// [`Backend::matmul_batch`] into caller-owned buffers.
    ///
    /// `outs` is resized to `ops.len()`; each element is then filled as
    /// [`Backend::matmul_into`] would fill it. The default hands the whole
    /// thing to `matmul_batch` and replaces `outs` wholesale, which keeps
    /// every backend that overrode `matmul_batch` for *submission* batching
    /// — every GPU backend — behaving exactly as it did.
    fn matmul_batch_into(&self, outs: &mut Vec<Vec<f32>>, ops: &[MatmulOp<'_>]) {
        *outs = self.matmul_batch(ops);
    }

    /// [`Backend::matmul_batch`]'s decode counterpart — see
    /// [`Backend::matmul_decode`] for why decode needs its own entry point.
    /// Defaults to `matmul_batch` so a backend that overrode *that* for
    /// submission batching keeps it here too.
    fn matmul_batch_decode(&self, ops: &[MatmulOp<'_>]) -> Vec<Vec<f32>> {
        self.matmul_batch(ops)
    }

    /// What a forward pass on this backend is made of, when that is **less
    /// than the engine's full GPU path** — one short phrase for the startup
    /// banner and `/props`, or `None` when there is nothing to say.
    ///
    /// `None` for [`CpuBackend`], which is not missing out on a faster path
    /// it could have taken, and `None` for the `wgpu` backends, which *are*
    /// the full path. `Some` for `CudaBackend`, `RocmBackend` and
    /// `OpenClBackend`, each of which implements [`Backend::matmul`] and
    /// nothing else: no fused whole-layer submission, no GPU-resident
    /// attention/RoPE/norm, no GPU-side sampling, no device-resident KV, and
    /// one dispatch strategy where the `wgpu` backends pick between several.
    ///
    /// It exists because the banner's `Kernels` row comes from
    /// [`Backend::as_wgpu`] and so prints *nothing at all* on those three.
    /// A number taken from a `--device cuda:0` run then looks like a number
    /// about this engine's architecture, when it is a number about the one
    /// path that backend has. An absent row reads as "nothing to report";
    /// this makes it read as what it is.
    ///
    /// A phrase rather than a warning: choosing one of these is a legitimate
    /// thing to do (it is what `--device` is for, and on a machine whose
    /// Vulkan driver is broken it is the only thing that works), so this
    /// belongs beside the device it describes and not in a `Note`.
    fn reduced_surface(&self) -> Option<&'static str> {
        None
    }

    /// Downcast hook for the GPU-specific fast paths that don't fit this
    /// trait's backend-agnostic shape: `VulkanBackend::
    /// fused_post_attention` chains a whole gemma4 sub-layer's matmuls and
    /// elementwise/norm ops into a single GPU submission, which needs
    /// `VulkanBackend`'s own buffer-cache internals, not just `matmul`/
    /// `matmul_batch`. `CpuBackend` has no round-trip cost to amortize
    /// there, so it keeps the default `None` and callers fall back to the
    /// ordinary step-by-step path.
    ///
    /// Named for `wgpu`, not for Vulkan, because two backends answer
    /// `Some` here: `VulkanBackend` itself and [`MetalBackend`], which is
    /// the same `wgpu` engine and the same WGSL kernels brought up on
    /// Metal (see `metal`'s module doc). Everything reached through this
    /// hook is therefore live on Apple GPUs too — a `Some` here means "a
    /// `wgpu` device with orangu's fused pipelines on it", not "Vulkan".
    /// Whether this backend *is* the host — so a caller weighing whether two
    /// pieces of work will run on different processors can tell.
    ///
    /// Not the same question as `as_wgpu().is_none()`: a split model's
    /// multi-device wrapper answers `None` there by design and is emphatically
    /// not the host. Overridden only by `CpuBackend`.
    fn is_host(&self) -> bool {
        false
    }

    fn as_wgpu(&self) -> Option<&VulkanBackend> {
        None
    }

    /// [`Backend::as_wgpu`] for work scoped to **one layer**, on the device
    /// that layer's weights were placed on.
    ///
    /// `device` is always read off a weight this call is about
    /// (`QuantMatrix::device`), never tracked separately, so it cannot
    /// disagree with where `matmul` will send the same layer's operands —
    /// there is one map, and it lives on the weights.
    ///
    /// This is what lets a *split* model keep the fused, GPU-resident
    /// per-layer paths. `as_wgpu` has to answer `None` on a split, because
    /// the things that reach for it without a layer in hand — the
    /// whole-step decode submission, GPU sampling, the logits readback —
    /// span layers and so span devices. A per-layer fused chain does not:
    /// it takes host input, returns host output, and touches only that
    /// layer's weights and that layer's KV. Answering `None` for those too
    /// would put attention and every norm back on the CPU for no reason.
    ///
    /// Every weight of one layer is on one device (`engine::placement`
    /// assigns whole layers), which is the invariant that makes a single
    /// answer per layer correct.
    fn as_wgpu_on(&self, _device: usize) -> Option<&VulkanBackend> {
        self.as_wgpu()
    }

    /// Whether this backend has a kernel for `ggml_type`.
    ///
    /// `CpuBackend` goes straight through `quant::dequantize`, so it covers
    /// everything `quant::supports_type` does and keeps this default. Every
    /// GPU backend has to compile a kernel per type and covers less.
    ///
    /// **How much less is pinned by a test, not by this comment** —
    /// [`tests::the_gpu_type_gap_is_exactly_what_this_comment_says`] asserts
    /// the set differences and fails if any of them moves. Read the
    /// assertions, not this paragraph, if the two ever disagree. As of that
    /// test: `VulkanBackend` lacks only the three lowest-bit `IQ1_*` ids that
    /// are not ggml's own (`quant`'s 64..=66); the vendor backends
    /// (`CudaBackend`/`RocmBackend`/`OpenClBackend`, which share one shader
    /// source) lack those and `MXFP4`. The vendor list is a strict subset of
    /// Vulkan's, and the test holds it that way on purpose — Vulkan is the
    /// backend with hardware to verify a kernel on, so a new one lands there
    /// first.
    ///
    /// This exists so that gap is reported by
    /// [`unsupported_tensor_types`] as a startup error naming the type and
    /// the backend, instead of surfacing as a panic from inside `matmul`
    /// partway through the first request — which is what it did while every
    /// GPU backend was assumed to be at parity with the CPU path.
    ///
    /// This paragraph used to name a different gap — that `VulkanBackend`
    /// lacked the three lowest-bit `IQ*` types and that the vendor backends
    /// lacked every codebook-indexed one. Both stopped being true when the
    /// `IQ` kernels landed, and neither the compiler nor a test noticed,
    /// because a claim about which kernels exist is a fact about two `const`
    /// arrays and nothing was comparing them. It stayed wrong long enough to
    /// mislead a reader into scoping work that was already done. The test
    /// named above is the actual fix; this rewrite is only the half of it a
    /// reader sees.
    fn supports_type(&self, _ggml_type: u32) -> bool {
        true
    }
}

/// Every distinct tensor type in `tensors` that `backend` has no kernel
/// for, as type names sorted for a stable message — empty when the backend
/// can run every tensor.
///
/// Takes `(name, ggml_type)` pairs rather than a `GgufFile` so the caller
/// can pass `LoadedModel::tensor_types`, which spans every shard; a split
/// model's `GgufFile` is only shard 1, and a type that appears solely in a
/// later shard would slip through. Reports the whole set rather than the
/// first hit because the pairs arrive in hash order, and because a mixed
/// low-bit file usually carries more than one type a backend lacks —
/// naming all of them means one restart, not one per type.
///
/// Reads the tensor *directory* only, never the data, so this is as cheap
/// as reading the header. Runs after `quant::supports_type`'s own check
/// (`loader::model_load_support`), so a type reported here is one this
/// build *can* read on the CPU but not on the selected device.
///
/// Some MoE tensors stay on the CPU today:
/// - `ExpertQuantMatrix` stacks are dequantized and dotted directly.
/// - Qwen shared-expert matrices can fall back to `CpuBackend` when the
///   selected backend lacks a quant kernel.
///
/// Rejecting a GPU backend for a type that appears only in those tensors
/// would therefore be a false negative at startup.
pub fn unsupported_tensor_types<'a>(
    tensors: impl Iterator<Item = (&'a str, u32)>,
    backend: &dyn Backend,
) -> Vec<String> {
    let mut names: Vec<String> = tensors
        .filter(|(name, ggml_type)| !is_cpu_only_tensor(name) && !backend.supports_type(*ggml_type))
        .map(|(_, ggml_type)| orangu::gguf::ggml_type_name(ggml_type))
        .collect();
    names.sort_unstable();
    names.dedup();
    names
}

/// Splits `tensors` (name, stored bytes) into what a GPU backend uploads
/// and what stays in host memory — `(device_bytes, host_bytes)`.
///
/// The same [`is_cpu_only_tensor`] rule [`unsupported_tensor_types`] uses,
/// and for the same reason: a routed expert's weights never reach the
/// device, so counting them against a card's VRAM would overstate the
/// footprint of every MoE model by most of its file size.
///
/// This is an upper bound on the device side, not a prediction. A tensor is
/// uploaded on first use, so a layer no request ever reaches is never
/// uploaded at all — but every layer of a served model is reached by the
/// first token, so the bound is tight in practice and loose only for a
/// model that is loaded and never used.
pub fn device_resident_split<'a>(tensors: impl Iterator<Item = (&'a str, u64)>) -> (u64, u64) {
    let mut device = 0u64;
    let mut host = 0u64;
    for (name, bytes) in tensors {
        if is_cpu_only_tensor(name) {
            host += bytes;
        } else {
            device += bytes;
        }
    }
    (device, host)
}

pub(crate) fn is_cpu_only_tensor(name: &str) -> bool {
    name.ends_with(".ffn_gate_exps.weight")
        || name.ends_with(".ffn_up_exps.weight")
        || name.ends_with(".ffn_down_exps.weight")
        || name.ends_with(".ffn_gate_up_exps.weight")
        || name.ends_with(".ffn_gate_shexp.weight")
        || name.ends_with(".ffn_up_shexp.weight")
        || name.ends_with(".ffn_down_shexp.weight")
        || name.ends_with(".ffn_gate_tid2eid.weight")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::loader::test_quant_matrix;
    use crate::engine::quant::{GGML_TYPE_F32, GGML_TYPE_IQ1_S, GGML_TYPE_IQ4_NL, GGML_TYPE_MXFP4};
    use std::sync::Mutex;

    /// A `[in, out]` F32 weight of exactly `bytes` bytes, for the routing
    /// tests — only its size matters to the decision under test.
    fn weight_of_bytes(bytes: usize) -> crate::engine::loader::QuantMatrix {
        let out_dim = bytes / 4;
        test_quant_matrix(&vec![0u8; bytes], GGML_TYPE_F32, 1, out_dim)
    }

    /// The decision is about the **batch**, not any one op in it.
    ///
    /// This is the load-bearing part of the rule and the easy thing to get
    /// wrong. A device pays its fixed cost once per submission, so eight small
    /// matmuls issued together pay it once between them — pricing each op on
    /// its own would send a batch to the host that the device would have won,
    /// and would do it precisely when batching had already made the device
    /// cheap.
    #[test]
    fn the_host_matmul_rule_weighs_the_whole_batch_not_one_op() {
        let w = weight_of_bytes(4096);
        let op = || MatmulOp {
            x: &[],
            n_tokens: 1,
            w: &w,
        };
        // One op, comfortably under.
        assert!(prefers_host_matmul_with(8192, &[op()]));
        // Two of the same op are not: together they cross.
        assert!(!prefers_host_matmul_with(8192, &[op(), op()]));
        assert!(prefers_host_matmul_with(16384, &[op(), op()]));
    }

    /// The crossing was measured at one token, so the rule applies at one
    /// token. A wider batch reads the same weights and does more arithmetic
    /// with them — the regime the device exists for — while the fixed cost it
    /// is charged does not change, so the crossing moves and this rule has no
    /// business guessing how far.
    #[test]
    fn the_host_matmul_rule_is_decode_only() {
        let w = weight_of_bytes(1024);
        let one = MatmulOp {
            x: &[],
            n_tokens: 1,
            w: &w,
        };
        let two = MatmulOp {
            x: &[],
            n_tokens: 2,
            w: &w,
        };
        assert!(prefers_host_matmul_with(8192, &[one]));
        assert!(!prefers_host_matmul_with(8192, &[two]));
    }

    /// `ORANGU_HOST_MATMUL_KIB=0` has to be a real control arm — a sweep that
    /// cannot switch the feature off measures it against itself. See
    /// `engine::env::flag_on`'s own note on why zero must mean off.
    #[test]
    fn a_zero_threshold_switches_the_host_matmul_rule_off() {
        let w = weight_of_bytes(4);
        let op = MatmulOp {
            x: &[],
            n_tokens: 1,
            w: &w,
        };
        assert!(prefers_host_matmul_with(1024, &[op]));
        let op = MatmulOp {
            x: &[],
            n_tokens: 1,
            w: &w,
        };
        assert!(!prefers_host_matmul_with(0, &[op]));
    }

    /// The fan-out floor is a floor, and `1` has to switch it off.
    ///
    /// `with_min_len(0)` panics, and `1` is the original one-row-per-task
    /// splitting — the control arm every sweep of this value is scored
    /// against.
    #[test]
    fn the_matmul_floor_is_at_least_one_row() {
        assert!(matmul_min_rows() >= 1);
    }

    /// The two forms of the host-routing decision must agree.
    ///
    /// `weights_prefer_host_matmul` exists so a caller can ask *before* it
    /// builds the call — deciding whether two branches will share a processor
    /// has to happen before either starts. Two predicates that can disagree
    /// would put a mixture-of-experts layer's branches on the same pool while
    /// the join still believed they were on different ones, which is the exact
    /// failure the join condition was rewritten to avoid.
    #[test]
    fn the_two_host_matmul_predicates_agree() {
        let w = weight_of_bytes(4096);
        for (n_tokens, threshold) in [(1usize, 8192usize), (1, 4096), (1, 0), (2, 8192)] {
            let ops = [MatmulOp {
                x: &[],
                n_tokens,
                w: &w,
            }];
            // `weights_prefer_host_matmul` reads the live threshold, so only
            // the cases where that is what `threshold` is can be compared
            // directly; the rest check the shape of the rule.
            let by_ops = prefers_host_matmul_with(threshold, &ops);
            let by_weights =
                threshold != 0 && n_tokens == 1 && w.row_bytes() * w.out_dim < threshold;
            assert_eq!(
                by_ops, by_weights,
                "n_tokens {n_tokens}, threshold {threshold}"
            );
        }
        // And the weights-only form itself, against the live threshold.
        assert!(!weights_prefer_host_matmul(2, &[&w]));
        assert!(!weights_prefer_host_matmul(1, &[]));
    }

    /// An empty batch is not a batch the host should be handed: it is nothing
    /// at all, and the device path returns for it immediately.
    #[test]
    fn an_empty_batch_never_prefers_the_host() {
        assert!(!prefers_host_matmul_with(usize::MAX, &[]));
    }

    /// **The claim `Backend::supports_type`'s doc makes, asserted.**
    ///
    /// Written because that comment made a *different* claim for a while after
    /// it stopped being true, and nothing caught it: which kernels exist is a
    /// fact about two `const` arrays, and no test compared them to each other
    /// or to the CPU path. A stale comment of that shape is not harmless —
    /// this one was read as evidence that the `IQ` family had no GPU kernels
    /// and used to scope work that was already finished.
    ///
    /// So the comment is no longer the record. This is. Adding a kernel is
    /// meant to fail this test: update the expected set here and the prose in
    /// `supports_type` and `arch::gpu_project_expert` in the same commit,
    /// which is exactly the coupling that was missing.
    ///
    /// The CPU side is enumerated through `quant::supports_type` rather than
    /// from a second hand-written list, so a type added to `quant` shows up
    /// here without anyone remembering to add it. The repacked ids are
    /// excluded on the loader's own account: it unpacks them back into the
    /// type they were built from and nothing downstream of it ever sees them
    /// (`quant`'s own comment above `GGML_TYPE_Q4_0_4_4`), so a backend is
    /// never asked about one and a "gap" there would be fiction.
    #[test]
    fn the_gpu_type_gap_is_exactly_what_this_comment_says() {
        use crate::engine::quant;

        // Loader-internal repacks: unpacked before any backend sees them.
        const REPACKS: &[u32] = &[
            quant::GGML_TYPE_Q4_0_4_4,
            quant::GGML_TYPE_Q4_0_4_8,
            quant::GGML_TYPE_Q4_0_8_8,
            quant::GGML_TYPE_IQ4_NL_4_4,
            quant::GGML_TYPE_IQ4_NL_4_8,
            quant::GGML_TYPE_IQ4_NL_8_8,
        ];
        // `I32` is an index/metadata type, never a weight a matmul kernel is
        // asked for, so it is not part of the quantization coverage question.
        const NOT_A_WEIGHT: &[u32] = &[quant::GGML_TYPE_I32];

        // Every id `quant` can read, swept rather than listed — the highest
        // id in the tree is 66 and the sweep is deliberately well past it, so
        // a new type lands in this set on its own.
        let cpu: Vec<u32> = (0..=127u32)
            .filter(|&t| quant::supports_type(t))
            .filter(|t| !REPACKS.contains(t) && !NOT_A_WEIGHT.contains(t))
            .collect();

        let vulkan = super::vulkan::SUPPORTED_TYPES;
        let vendor = super::vendor_shaders::SUPPORTED_TYPES;

        // The vendor backends (CUDA, ROCm, OpenCL) render from one shared
        // source, so their coverage is one list. It is a strict *subset* of
        // Vulkan's, which is the shape to assert: Vulkan is the backend with
        // hardware to verify a kernel on, so it is where a new one lands
        // first, and a vendor list that grew a type Vulkan does not have
        // would be a kernel nobody has ever run.
        let vendor_only: Vec<String> = vendor
            .iter()
            .filter(|t| !vulkan.contains(t))
            .map(|&t| orangu::gguf::ggml_type_name(t))
            .collect();
        assert!(
            vendor_only.is_empty(),
            "the vendor backends claim kernels Vulkan does not have: {vendor_only:?} — \
             these are the three backends documented as unverified on real hardware, \
             so a type reaching them first has been tested nowhere"
        );

        let vulkan_only: Vec<String> = vulkan
            .iter()
            .filter(|t| !vendor.contains(t))
            .map(|&t| orangu::gguf::ggml_type_name(t))
            .collect();
        assert_eq!(
            vulkan_only,
            ["MXFP4"],
            "the set of types Vulkan has a kernel for and the vendor backends do \
             not has changed; update `Backend::supports_type`'s doc to match"
        );

        let gap = |list: &[u32]| -> Vec<String> {
            cpu.iter()
                .filter(|t| !list.contains(t))
                .map(|&t| orangu::gguf::ggml_type_name(t))
                .collect()
        };

        assert_eq!(
            gap(vulkan),
            ["IQ1_XS", "IQ1_XXS", "IQ1_XXXS"],
            "the set of types the CPU path reads and Vulkan has no kernel for has \
             changed; update `Backend::supports_type` and \
             `arch::gpu_project_expert`'s doc comments to match"
        );
        assert_eq!(
            gap(vendor),
            ["MXFP4", "IQ1_XS", "IQ1_XXS", "IQ1_XXXS"],
            "the set of types the CPU path reads and the vendor backends have no \
             kernel for has changed; update `Backend::supports_type`'s doc to match"
        );

        // The half the previous comment got backwards, stated positively so a
        // future reader does not have to re-derive it from the list above.
        for &iq in &[
            quant::GGML_TYPE_IQ1_S,
            quant::GGML_TYPE_IQ1_M,
            quant::GGML_TYPE_IQ2_XXS,
            quant::GGML_TYPE_IQ2_XS,
            quant::GGML_TYPE_IQ2_S,
            quant::GGML_TYPE_IQ3_XXS,
            quant::GGML_TYPE_IQ3_S,
            quant::GGML_TYPE_IQ4_NL,
            quant::GGML_TYPE_IQ4_XS,
        ] {
            assert!(
                vulkan.contains(&iq),
                "{} has no GPU kernel; the doc comments claiming the IQ family \
                 is covered are wrong again",
                orangu::gguf::ggml_type_name(iq)
            );
        }
    }

    /// `plan_batch`'s stripes must reconstruct exactly the rows
    /// `guarded_matmul_op` would have produced, in the same order — a
    /// batched backend that dropped or reordered a stripe would produce a
    /// result that is subtly wrong per token rather than obviously wrong.
    #[test]
    fn every_op_is_covered_by_its_stripes_in_row_order() {
        let bytes = vec![0u8; 4 * 8 * 4];
        let w = test_quant_matrix(&bytes, GGML_TYPE_F32, 4, 8);
        let max = max_multi_token_phase_tokens();
        // One op under the cap, one exactly at it, one well over.
        let counts = [1usize, max, max * 2 + 3];
        let xs: Vec<Vec<f32>> = counts.iter().map(|&n| vec![0.0; n * 4]).collect();
        let ops: Vec<MatmulOp<'_>> = counts
            .iter()
            .zip(&xs)
            .map(|(&n_tokens, x)| MatmulOp { x, n_tokens, w: &w })
            .collect();

        let (stripes, _) = plan_batch(&ops, BATCH_DEVICE_BUDGET_BYTES);
        for (op_index, &n_tokens) in counts.iter().enumerate() {
            let mine: Vec<&BatchStripe> = stripes.iter().filter(|s| s.op == op_index).collect();
            // Contiguous from 0, no gaps and no overlap.
            let mut next = 0usize;
            for stripe in &mine {
                assert_eq!(
                    stripe.start, next,
                    "op {op_index} stripes are not contiguous"
                );
                assert!(stripe.n_tokens <= max, "stripe exceeds the phase cap");
                next += stripe.n_tokens;
            }
            assert_eq!(next, n_tokens, "op {op_index} lost or gained rows");
        }
        // Ops keep their order, so a caller can concatenate per op.
        let order: Vec<usize> = stripes.iter().map(|s| s.op).collect();
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(order, sorted, "stripes are not grouped by op in order");
    }

    /// Chunks are what a backend synchronizes at, so one must never be
    /// empty — an empty chunk is a submission with nothing in it — and
    /// together they must cover every stripe exactly once.
    #[test]
    fn chunks_partition_the_stripes_and_respect_the_budget() {
        let bytes = vec![0u8; 4 * 8 * 4];
        let w = test_quant_matrix(&bytes, GGML_TYPE_F32, 4, 8);
        let x = vec![0.0f32; 4 * 4];
        let ops: Vec<MatmulOp<'_>> = (0..16)
            .map(|_| MatmulOp {
                x: &x,
                n_tokens: 4,
                w: &w,
            })
            .collect();

        // A budget that fits ~2 stripes: 4 tokens x (4 + 8) dims x 4 bytes.
        let per_stripe = 4 * (4 + 8) * std::mem::size_of::<f32>();
        let (stripes, chunks) = plan_batch(&ops, per_stripe * 2);
        assert!(chunks.len() > 1, "a tight budget should split the batch");
        let mut covered = 0usize;
        for chunk in &chunks {
            assert!(
                !chunk.is_empty(),
                "an empty chunk is a submission with no work"
            );
            assert_eq!(chunk.start, covered, "chunks are not contiguous");
            covered = chunk.end;
            let bytes: usize = stripes[chunk.clone()]
                .iter()
                .map(|s| s.n_tokens * (4 + 8) * std::mem::size_of::<f32>())
                .sum();
            assert!(
                bytes <= per_stripe * 2 || chunk.len() == 1,
                "chunk of {} stripes holds {bytes} B over a {} B budget",
                chunk.len(),
                per_stripe * 2
            );
        }
        assert_eq!(covered, stripes.len(), "chunks do not cover every stripe");
    }

    /// A single stripe over the budget gets its own chunk rather than being
    /// dropped or split below the phase cap. Refusing to run it would be a
    /// worse answer than exceeding a budget this module set itself.
    #[test]
    fn one_oversized_stripe_still_gets_a_chunk() {
        let bytes = vec![0u8; 4 * 8 * 4];
        let w = test_quant_matrix(&bytes, GGML_TYPE_F32, 4, 8);
        let x = vec![0.0f32; 4 * 4];
        let ops = [MatmulOp {
            x: &x,
            n_tokens: 4,
            w: &w,
        }];
        let (stripes, chunks) = plan_batch(&ops, 1);
        assert_eq!(stripes.len(), 1);
        assert_eq!(chunks, vec![0..1]);
    }

    /// A backend that accepts everything except the listed types, standing
    /// in for a real GPU backend's `SUPPORTED_TYPES` gap without needing a
    /// device.
    struct Picky<'a>(&'a [u32]);

    impl Backend for Picky<'_> {
        fn matmul(&self, _x: &[f32], _n_tokens: usize, _w: &QuantMatrix) -> Vec<f32> {
            unreachable!("supports_type is the only thing under test here")
        }
        fn supports_type(&self, ggml_type: u32) -> bool {
            !self.0.contains(&ggml_type)
        }
    }

    /// A backend missing two types must name both, deduped and sorted, and
    /// must not name the ones it does support. Sorted so the message doesn't
    /// change between runs — `LoadedModel::tensor_types` iterates a
    /// `HashMap`.
    #[test]
    fn unsupported_tensor_types_reports_every_missing_type_once() {
        let tensors = [
            ("output_norm.weight", GGML_TYPE_F32),
            ("blk.0.attn_k.weight", GGML_TYPE_IQ4_NL),
            ("blk.0.attn_q.weight", GGML_TYPE_IQ1_S),
            ("blk.1.attn_k.weight", GGML_TYPE_IQ4_NL),
        ];
        let found = unsupported_tensor_types(
            tensors.iter().copied(),
            &Picky(&[GGML_TYPE_IQ4_NL, GGML_TYPE_IQ1_S]),
        );
        assert_eq!(found, vec!["IQ1_S".to_string(), "IQ4_NL".to_string()]);
    }

    #[test]
    fn unsupported_tensor_types_ignores_cpu_only_moe_tensors() {
        let tensors = [
            ("blk.0.ffn_gate_exps.weight", GGML_TYPE_MXFP4),
            ("blk.0.ffn_up_exps.weight", GGML_TYPE_MXFP4),
            ("blk.0.ffn_down_exps.weight", GGML_TYPE_MXFP4),
            ("blk.0.ffn_gate_up_exps.weight", GGML_TYPE_MXFP4),
            ("blk.0.ffn_gate_shexp.weight", GGML_TYPE_MXFP4),
            ("blk.0.ffn_up_shexp.weight", GGML_TYPE_MXFP4),
            ("blk.0.ffn_down_shexp.weight", GGML_TYPE_MXFP4),
        ];
        let found = unsupported_tensor_types(tensors.iter().copied(), &Picky(&[GGML_TYPE_MXFP4]));
        assert!(found.is_empty(), "{found:?}");
    }

    struct RecordingBackend {
        calls: Mutex<Vec<usize>>,
    }

    impl Backend for RecordingBackend {
        fn matmul(&self, _x: &[f32], n_tokens: usize, w: &QuantMatrix) -> Vec<f32> {
            self.calls.lock().unwrap().push(n_tokens);
            vec![0.0; n_tokens * w.out_dim]
        }
    }

    #[test]
    fn default_matmul_batch_stripes_a_long_op_for_any_backend() {
        let backend = RecordingBackend {
            calls: Mutex::new(Vec::new()),
        };
        let w = test_quant_matrix(&[0; 4 * 3 * 2], GGML_TYPE_F32, 3, 2);
        let n_tokens = MAX_MULTI_TOKEN_PHASE_TOKENS_DEFAULT + 6;
        let x = vec![0.0; n_tokens * w.in_dim];

        let ys = backend.matmul_batch(&[MatmulOp {
            x: &x,
            n_tokens,
            w: &w,
        }]);

        assert_eq!(ys.len(), 1);
        assert_eq!(ys[0].len(), n_tokens * w.out_dim);
        assert_eq!(
            *backend.calls.lock().unwrap(),
            vec![MAX_MULTI_TOKEN_PHASE_TOKENS_DEFAULT, 6]
        );
    }

    /// `CpuBackend` keeps the permissive default, so a file it can decode
    /// never trips this check — which is what keeps the startup gate from
    /// rejecting models that previously ran.
    #[test]
    fn unsupported_tensor_types_passes_a_backend_that_covers_everything() {
        let tensors = [
            ("blk.0.attn_k.weight", GGML_TYPE_IQ4_NL),
            ("blk.0.attn_q.weight", GGML_TYPE_IQ1_S),
        ];
        assert!(unsupported_tensor_types(tensors.iter().copied(), &CpuBackend).is_empty());
    }
}
