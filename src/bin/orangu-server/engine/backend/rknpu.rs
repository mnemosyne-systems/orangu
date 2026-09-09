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

//! Rockchip RKNPU backend, via `librknnrt`'s `rknn_matmul_*` API.
//!
//! Same scope as `engine::backend::cuda` and `engine::backend::opencl`:
//! [`Backend::matmul`] and nothing else, with `CpuBackend` behind it for
//! everything this device cannot take. Like both of those it `dlopen`s its
//! vendor library at runtime, so a build made here still runs on a machine
//! that has never heard of an NPU.
//!
//! # Why this exists when `crate::npu` said it couldn't
//!
//! `orangu::npu`'s NOE path drives graphs compiled ahead of time and has no
//! per-operation entry point, which is why `backend = npu` used to be an
//! error. RKNPU's runtime is a different shape: `rknn_matmul_create` /
//! `rknn_matmul_run` is exactly the per-call `C = A × B` seam
//! [`Backend::matmul`] needs, so on a Rockchip board the NPU can be a real
//! backend rather than an inventory line. The two stacks stay entirely
//! separate — see [`orangu::npu::NpuStack`].
//!
//! # Wide calls only
//!
//! **This device is for prefill.** Its fixed cost per call is around 0.4 ms,
//! so at one token it manages 20 GFLOP/s against 1190 at 512 — and the CPU,
//! which reads the `Q4_K` weight straight out of the mapped file rather than a
//! requantized copy, wins a narrow call outright. So anything under
//! [`min_tokens`] tokens goes to `CpuBackend`, and [`Backend::matmul_decode`]
//! is overridden to send every decode there regardless of width.
//!
//! Both, not one: the override is the semantically right statement and on its
//! own it was not enough — a decode step reaches `matmul` too, and with only
//! the override in place `backend = npu` still decoded at 2.15 tok/s against
//! `cpu`'s 5.27. The width rule is what actually closed that, to 5.12.
//!
//! What that buys, end to end on an RK3588 against `backend = cpu`: decode
//! 5.12 tok/s against 5.27, prefill **21.2 against 16.4**. The device is worth
//! having here, and only for the wide half of the work.
//!
//! # What runs on the device
//!
//! `int8 × int8 → int32` by default, with a symmetric scale per output channel
//! on the weight and **one per token** on the activations, all recombined on
//! the host — see [`Mode`] for the throughput that makes int8 the default and
//! [`quantize_activations`] for why per-token granularity is not optional.
//! `ORANGU_NPU_MODE=fp16` selects `float16 × float16 → float32` instead, which
//! needs no requantization of a GGUF weight at all and is the fallback if int8
//! ever costs visible quality. Normal layouts throughout.
//!
//! A weight is only offloaded when the device can take its shape — `K` a
//! multiple of [`K_ALIGN_ELEMS`], `N` a multiple of [`Mode::n_align`] — when
//! the residency budget has room for it, *and* when one matmul through it
//! agrees with a host reference ([`Weight::agrees_with_host`]). Anything else
//! is answered by `CpuBackend`, per weight, decided once and remembered.
//! **This is the load-bearing design decision**: "some weights on the NPU and
//! the rest on the CPU" is the normal case here rather than a degraded one,
//! and a `QuantMatrix` whose `out_dim` is not a multiple of 32 — an attention
//! projection on a model with an awkward head count, say — simply never
//! reaches the device.
//!
//! The arithmetic check is there because the vendor header and the vendor
//! runtime disagree about those limits in both directions; see
//! [`K_ALIGN_ELEMS`].
//!
//! # Precision: what the device owes, and what quantization costs
//!
//! Two different questions, and conflating them wastes a day. Ask them
//! separately.
//!
//! **The device computes what it is asked to, essentially exactly.** Measured
//! against a host reference reproducing the same arithmetic — fp16 operands
//! accumulated in `f64` — the worst relative error over an output column is
//! ~1e-6, at `K` from 256 to 11008. So it accumulates in fp32, and
//! [`Weight::agrees_with_host`] holds it to 1e-3 and means it.
//!
//! **Quantizing the operands is the real cost, and it is this backend's
//! choice.** Against an `f32`-operand reference the gap reaches 3% on
//! synthetic quantized blocks, nearly all of it cancellation: those blocks
//! have random per-block scales, which drives `sum |terms| / |sum terms|` as
//! high as 560, and any rounding is amplified by exactly that factor.
//!
//! That is also why every cross-check here references the active mode's own
//! arithmetic rather than `CpuBackend::matmul_dequant`. Comparing a quantized
//! matmul against an f32 one and calling the difference an error measures the
//! choice, not the implementation — and it costs real time: two cross-checks
//! failed that way before the reference was fixed, and the "failure" was
//! entirely in the reference.
//!
//! What the quantization costs in practice is a question only the whole model
//! answers, and it has bitten once already: with a single activation scale per
//! *tensor*, this model stopped being able to count to twenty. Per token, it
//! produces output identical to the CPU's. That is the level at which an
//! accuracy change here has to be checked.
//!
//! # Cost of residency
//!
//! An int8 copy of a `Q4_K` weight is about 1.8× the bytes of the original
//! (fp16 is 3.5×), and the device stops handing out memory near 2 GiB across
//! all contexts. That is the whole reason [`Residency`] exists, why
//! [`NPU_WEIGHTS_GB_DEFAULT`] leaves half the ceiling free, and why halving
//! the bytes per weight matters twice — it doubles how much of a model can be
//! resident at all. Weights are taken first-come until the budget is spent,
//! deliberately not an LRU: a transformer touches every weight once per token,
//! and an LRU over that access pattern is a thrash generator.

use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_void};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use half::f16;
use half::slice::HalfFloatSliceExt;
use libloading::Library;
use rayon::prelude::*;

use crate::engine::loader::QuantMatrix;

use super::device::{DeviceCandidate, DeviceClass};
use super::{Backend, CpuBackend, MatmulOp};

/// `RKNN_SUCC`. Every entry point returns this on success and a negative
/// error code otherwise; none of the distinctions between those codes
/// changes what this module does, which is decline the weight.
const RKNN_SUCC: c_int = 0;

/// `RKNN_TENSOR_FLOAT16` / `RKNN_TENSOR_FLOAT32` from `rknn_api.h`, used
/// only to check that the runtime agrees with us about what it was asked
/// for before any pointer is handed to it.
const RKNN_TENSOR_FLOAT32: c_int = 0;
const RKNN_TENSOR_FLOAT16: c_int = 1;
const RKNN_TENSOR_INT8: c_int = 2;
const RKNN_TENSOR_INT32: c_int = 6;

/// The two `rknn_matmul_type` values this platform supports and this module
/// uses — see [`Mode`].
const RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32: c_int = 1;
const RKNN_INT8_MM_INT8_TO_INT32: c_int = 2;

/// The largest magnitude a symmetric int8 quantization uses. 127 and not 128,
/// so negating a quantized value cannot overflow.
const INT8_MAX: f32 = 127.0;

const RKNN_MAX_NAME_LEN: usize = 256;
const RKNN_MAX_DIMS: usize = 16;

/// Which arithmetic the device is asked for.
///
/// **`Int8` is the default, and the reason is throughput.** Measured on an
/// RK3588 with `librknnrt` 2.3.0, steady-state, bound once and re-run:
///
/// | mode | M=1 | M=64 | M=512 |
/// |---|---|---|---|
/// | fp16 × fp16 → fp32 | 10 GFLOP/s | 265 | 266 |
/// | int8 × int8 → int32 | 20 GFLOP/s | **926** | **1190** |
///
/// 4.5× at the widths that matter, and half the bytes per weight — which
/// matters twice over, because the device tops out near 2 GiB (see
/// [`NPU_WEIGHTS_GB_DEFAULT`]) so halving a weight doubles how much of a
/// model can be resident at all.
///
/// The platform's other modes are not options here: `fp16 × int8 → fp32` and
/// every int4 variant are rejected outright by this runtime as `unsupported
/// matmul dtype ... in this platform`, so the mixed-precision path that would
/// have given int8's density with fp16's activations does not exist on
/// RK3588. [`Fp16`](Mode::Fp16) stays reachable through `ORANGU_NPU_MODE` as
/// the accuracy-safe option, since it is the one that needs no requantization
/// of a GGUF weight at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// `int8 × int8 → int32`, with a symmetric scale per output channel for
    /// the weight and one per tensor for the activations, both applied on the
    /// host. The device does integer arithmetic and nothing else, which is
    /// why no `rknn_matmul_set_quant_params` call appears anywhere here: the
    /// scales never need to cross into the runtime.
    Int8,
    /// `float16 × float16 → float32`.
    Fp16,
}

impl Mode {
    fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "int8" | "i8" => Some(Self::Int8),
            "fp16" | "f16" | "float16" => Some(Self::Fp16),
            _ => None,
        }
    }

    fn matmul_type(self) -> c_int {
        match self {
            Self::Int8 => RKNN_INT8_MM_INT8_TO_INT32,
            Self::Fp16 => RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32,
        }
    }

    /// What the runtime must say `A`, `B` and `C` are, so a driver that
    /// answered with something else is caught before a pointer reaches it.
    fn tensor_kinds(self) -> (c_int, c_int, c_int) {
        match self {
            Self::Int8 => (RKNN_TENSOR_INT8, RKNN_TENSOR_INT8, RKNN_TENSOR_INT32),
            Self::Fp16 => (
                RKNN_TENSOR_FLOAT16,
                RKNN_TENSOR_FLOAT16,
                RKNN_TENSOR_FLOAT32,
            ),
        }
    }

    /// Bytes per element of `A`/`B`. `C` is four in both modes.
    fn element_bytes(self) -> usize {
        match self {
            Self::Int8 => 1,
            Self::Fp16 => 2,
        }
    }

    /// The alignment this mode's `K` must satisfy, in elements. 32 for both,
    /// measured — see [`K_ALIGN_ELEMS`].
    fn k_align(self) -> usize {
        K_ALIGN_ELEMS
    }

    /// The alignment this mode's `N` must satisfy, in elements. **Not the
    /// same for the two modes**, measured: int8 needs 32 and fp16 needs 16.
    fn n_align(self) -> usize {
        match self {
            Self::Int8 => 32,
            Self::Fp16 => 16,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Int8 => "int8",
            Self::Fp16 => "fp16",
        }
    }
}

/// Which mode this process uses, from `ORANGU_NPU_MODE`.
fn mode() -> Mode {
    static MODE: std::sync::OnceLock<Mode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        let Ok(raw) = std::env::var("ORANGU_NPU_MODE") else {
            return Mode::Int8;
        };
        Mode::parse(&raw).unwrap_or_else(|| {
            eprintln!(
                "orangu-server: ORANGU_NPU_MODE={raw:?} is not int8 or fp16 — using int8. \
                 This run measures int8, not the value you asked for."
            );
            Mode::Int8
        })
    })
}

/// The device's shape alignment, in elements, **measured on real hardware
/// rather than read off the header**.
///
/// `rknn_matmul_api.h` says `K` must be "aligned with 32byte" and `N` with
/// "16byte" for fp16, which reads as 16 and 8 elements. It is not what the
/// runtime enforces: `librknnrt` 2.3.0 on an RK3588 rejects every `K` that is
/// not a multiple of **32 elements** (`matmul K:16 must be align with 32!`),
/// and for `N` wants **16** in fp16 and **32** in int8. Swept from 8 to 128
/// in steps of 8, in both modes — the accepted set is exactly those
/// multiples.
///
/// The header's `K max: k <= 10240` is in the same category and is *not*
/// enforced either: `K` of 11008 and 14336 both create and both compute
/// correctly. So there is no maximum here. Nothing is taken on trust
/// instead — every weight is proved against a host reference before it is
/// used, which is a stronger guarantee than either number. See
/// [`Weight::agrees_with_host`].
/// `K`'s alignment, the same in both modes. `N`'s is not — it is 32 for int8
/// and 16 for fp16 — so it lives on [`Mode::n_align`] rather than here.
const K_ALIGN_ELEMS: usize = 32;

/// Token counts a resident weight is prepared to answer, smallest first.
///
/// One context per weight serves every one of them through
/// `rknn_matmul_create_dynamic_shape`, which is the whole reason the ladder
/// exists: a context is where the fp16 copy of the weight lives, so a
/// context per (weight, token count) would hold that copy once per rung and
/// spend the residency budget several times over on the same weight.
///
/// Powers of two, so a call is padded to at most twice its width — the
/// wasted rows are real device work, and a coarser ladder spends more of it
/// than the dispatch it saves. `1` is exact, which is the one that matters
/// most: it is every decode step.
const TOKEN_RUNGS: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512];

/// How many output columns [`Weight::agrees_with_host`] checks.
const VERIFY_COLUMNS: usize = 64;

/// Fewest tokens in a call worth sending to the device, from
/// `ORANGU_NPU_MIN_TOKENS`. `0` sends everything.
///
/// **A width threshold, not a phase test, and that is deliberate.** Overriding
/// [`Backend::matmul_decode`] is the semantically right thing and it is not
/// sufficient: measured, `backend = npu` still decoded at 2.15 tok/s against
/// `cpu`'s 5.27 with that override in place, because a decode step reaches
/// `matmul` too — whatever the engine happens to route through the
/// phase-agnostic entry point arrives here looking like any other call. A rule
/// on the only thing this method can actually see — how many tokens it was
/// given — catches every such path without this module having to know which
/// they are.
///
/// The default comes from the device's own curve. At `K = N = 2048` it manages
/// 20 GFLOP/s at one token, 926 at 64 and 1190 at 512: the fixed cost per call
/// is around 0.4 ms, so a narrow call is nearly all overhead and the CPU —
/// which reads the `Q4_K` weight directly instead of a requantized copy — wins
/// outright. The same shape of argument, and the same conclusion, as
/// `backend::host_matmul_threshold_bytes`.
fn min_tokens() -> usize {
    static TOKENS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *TOKENS.get_or_init(|| {
        super::env_tuning_value(
            "ORANGU_NPU_MIN_TOKENS",
            NPU_MIN_TOKENS_DEFAULT,
            "a token count (0 sends every call to the device)",
            |tokens: usize| tokens <= 512,
        )
    })
}

/// The default for [`min_tokens`].
const NPU_MIN_TOKENS_DEFAULT: usize = 16;

/// How many bytes of fp16 weights may be made resident on the device, from
/// `ORANGU_NPU_WEIGHTS_GB`. See the module doc on why there is a budget at
/// all.
///
/// The default is deliberately a fraction of a small board's RAM rather
/// than a fraction of the model: this memory is DMA memory held for the
/// life of the process, and it competes with the mapped GGUF that the CPU
/// backend is still reading for every weight this one declined.
fn budget_bytes() -> u64 {
    static BYTES: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *BYTES.get_or_init(|| {
        let gb = super::env_tuning_value(
            "ORANGU_NPU_WEIGHTS_GB",
            NPU_WEIGHTS_GB_DEFAULT,
            "a size in GiB (0 keeps every weight on the CPU)",
            |gb: f64| (0.0..=8.0).contains(&gb),
        );
        (gb * (1u64 << 30) as f64) as u64
    })
}

/// The default for [`budget_bytes`], in GiB.
///
/// **Headroom under a measured hard ceiling, not a guess.** The device stops
/// handing out memory at about 2 GiB: a loop allocating 4 MiB at a time is
/// refused at 2048 MiB from one context, and 254 contexts holding a weight
/// plus scratch each are refused at 2034 MiB. It is a limit on the IOMMU
/// domain every context here shares, and it is not the `4GB` the header
/// attributes to a domain.
///
/// The default leaves headroom under that rather than claiming it, because
/// running out is not a soft landing: the allocation that fails is as likely
/// to be a scratch buffer for an already-resident weight as a new weight, and
/// the first version of this defaulted to the whole 2 GiB and spent the first
/// forward pass emitting hundreds of driver errors.
///
/// **How much of the model is resident is the single biggest lever on what
/// this backend is worth**, because everything it cannot hold runs on the CPU
/// exactly as before. Measured on an RK3588 prefilling a fresh 1976-token
/// prompt through `gemma-4-E2B-it` at `Q4_K_M`, against `backend = cpu`'s
/// 16.4 tok/s:
///
/// | budget | prefill | vs CPU |
/// |---|---|---|
/// | 0 (control — every matmul on the CPU) | 16.3 tok/s | 0.99× |
/// | 1.0 GiB | 19.4 tok/s | 1.18× |
/// | 1.5 GiB | 21.2 tok/s | 1.29× |
/// | 1.75 GiB | 22.2 tok/s | 1.35× |
///
/// 1.5 is the default and not 1.75: the curve is still climbing at the top of
/// that table, so the limit here is the device's 2 GiB and not diminishing
/// returns, and the quarter-gibibyte between them is what a model with a
/// vocabulary-sized output projection needs for scratch at a wide rung. Raise
/// it with `ORANGU_NPU_WEIGHTS_GB` on a board where it pays; running out is
/// handled, and says so once.
const NPU_WEIGHTS_GB_DEFAULT: f64 = 1.5;

/// Every byte this backend has on the device, against the budget.
///
/// **Scratch counts.** The first version of this accounted only the fp16
/// weights, and the per-call `A`/`C` buffers — which are per *weight*, since
/// each weight owns the context they belong to, and which at a prefill width
/// run to tens of MiB each — were invisible to it. On a 35-layer model that
/// is gigabytes of unaccounted device memory, and it is what actually hit the
/// ceiling.
struct Residency {
    used: Mutex<u64>,
    budget: u64,
    /// Set the first time the *device* refuses an allocation, as opposed to
    /// the budget refusing one. Sticky, because the only thing that frees
    /// device memory here is a weight being dropped, and weights are held for
    /// the life of the process: once it is full it stays full, and retrying
    /// costs a context and a whole-matrix dequantize per weight to learn
    /// nothing.
    exhausted: AtomicBool,
}

impl Residency {
    fn new(budget: u64) -> Self {
        Self {
            used: Mutex::new(0),
            budget,
            exhausted: AtomicBool::new(false),
        }
    }

    /// Claims `bytes` if the budget has room, and says whether it did.
    ///
    /// Claimed *before* the allocation it pays for, so two threads arriving
    /// together cannot both see room for memory only one of them can have.
    fn try_reserve(&self, bytes: u64) -> bool {
        if self.exhausted.load(Ordering::Relaxed) {
            return false;
        }
        let mut used = self.used.lock().expect("rknpu residency poisoned");
        if *used + bytes > self.budget {
            return false;
        }
        *used += bytes;
        true
    }

    fn release(&self, bytes: u64) {
        let mut used = self.used.lock().expect("rknpu residency poisoned");
        *used = used.saturating_sub(bytes);
    }

    /// Records that the device itself said no, and says so once.
    ///
    /// Worth a line because it is the one condition here an operator can act
    /// on, and because its symptom without one is "the NPU stopped helping
    /// part way through the model" with nothing in the log.
    fn note_device_refusal(&self) {
        if !self.exhausted.swap(true, Ordering::Relaxed) {
            eprintln!(
                "orangu-server: [npu] the device refused more memory; the weights \
                 already on it stay there and the rest of the model runs on the CPU. \
                 This device tops out near 2 GiB across all contexts — lower \
                 ORANGU_NPU_WEIGHTS_GB if this is costing more than it buys."
            );
        }
    }

    fn is_exhausted(&self) -> bool {
        self.exhausted.load(Ordering::Relaxed)
    }

    /// **Tests only** — see [`RknpuBackend::resident_bytes`], which is the
    /// only caller and says why it earns its place.
    #[cfg(test)]
    fn used(&self) -> u64 {
        *self.used.lock().expect("rknpu residency poisoned")
    }
}

// ---------------------------------------------------------------------------
// FFI
//
// Every struct below is `rknn_api.h`/`rknn_matmul_api.h` verbatim, and the
// four whose layout this code depends on were checked against the headers
// compiled by the system's own toolchain: `rknn_matmul_info` 64 bytes,
// `rknn_matmul_tensor_attr` 332, `rknn_matmul_io_attr` 996,
// `rknn_tensor_mem` 40. `assert_ffi_struct_sizes` below pins those numbers
// so a field added here in the wrong place is a test failure rather than a
// pointer handed to the driver at the wrong offset.
// ---------------------------------------------------------------------------

/// `rknn_context`, which is `uint64_t` on every 64-bit target — the header
/// narrows it to `uint32_t` only under `__arm__`, which this is not.
type RknnContext = u64;

#[repr(C)]
#[derive(Clone, Copy)]
struct MatmulInfo {
    m: i32,
    k: i32,
    n: i32,
    kind: c_int,
    b_layout: i16,
    b_quant_type: i16,
    ac_layout: i16,
    ac_quant_type: i16,
    iommu_domain_id: i32,
    group_size: i16,
    reserved: [i8; 34],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MatmulTensorAttr {
    name: [c_char; RKNN_MAX_NAME_LEN],
    n_dims: u32,
    dims: [u32; RKNN_MAX_DIMS],
    size: u32,
    kind: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MatmulIoAttr {
    a: MatmulTensorAttr,
    b: MatmulTensorAttr,
    c: MatmulTensorAttr,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MatmulShape {
    m: i32,
    k: i32,
    n: i32,
}

#[repr(C)]
struct TensorMem {
    virt_addr: *mut c_void,
    phys_addr: u64,
    fd: i32,
    offset: i32,
    size: u32,
    flags: u32,
    priv_data: *mut c_void,
}

/// The `rknn_matmul_*` entry points this backend calls, resolved out of an
/// open `librknnrt`.
///
/// `Copy`, so a weight can hold its own copy rather than borrowing the
/// backend — a resident weight has to be able to destroy its context and
/// its memory in `Drop`, and threading a lifetime for that through
/// `Backend`'s object-safe trait is not worth what it buys.
#[derive(Clone, Copy)]
struct Api {
    create_dynamic_shape: unsafe extern "C" fn(
        *mut RknnContext,
        *mut MatmulInfo,
        c_int,
        *mut MatmulShape,
        *mut MatmulIoAttr,
    ) -> c_int,
    set_dynamic_shape: unsafe extern "C" fn(RknnContext, *mut MatmulShape) -> c_int,
    set_io_mem: unsafe extern "C" fn(RknnContext, *mut TensorMem, *mut MatmulTensorAttr) -> c_int,
    run: unsafe extern "C" fn(RknnContext) -> c_int,
    destroy: unsafe extern "C" fn(RknnContext) -> c_int,
    create_mem: unsafe extern "C" fn(RknnContext, u32) -> *mut TensorMem,
    destroy_mem: unsafe extern "C" fn(RknnContext, *mut TensorMem) -> c_int,
}

impl Api {
    /// Resolves every entry point, or `None` for a library that opens and
    /// isn't this API — the same all-or-nothing contract `orangu::npu`'s
    /// `Noe::load` has, and for the same reason.
    fn load(library: &Library) -> Option<Self> {
        // SAFETY: each signature here matches its declaration in
        // `rknn_matmul_api.h`/`rknn_api.h`, which are C and exported
        // unmangled.
        unsafe {
            Some(Self {
                create_dynamic_shape: *library.get(b"rknn_matmul_create_dynamic_shape\0").ok()?,
                set_dynamic_shape: *library.get(b"rknn_matmul_set_dynamic_shape\0").ok()?,
                set_io_mem: *library.get(b"rknn_matmul_set_io_mem\0").ok()?,
                run: *library.get(b"rknn_matmul_run\0").ok()?,
                destroy: *library.get(b"rknn_matmul_destroy\0").ok()?,
                create_mem: *library.get(b"rknn_create_mem\0").ok()?,
                destroy_mem: *library.get(b"rknn_destroy_mem\0").ok()?,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Resident weights
// ---------------------------------------------------------------------------

/// `QuantMatrix::cache_key()`'s return type, named for the same reason
/// `opencl.rs` names it.
type WeightCacheKey = (usize, usize);

/// The device memory bound to one rung of [`TOKEN_RUNGS`].
struct Io {
    a: *mut TensorMem,
    c: *mut TensorMem,
}

/// Everything one offloaded weight owns on the device.
///
/// The `Mutex` is the device lock as well as the state lock: selecting a
/// shape, binding memory and running are one transaction on a context that
/// the vendor runtime does not document as thread-safe, and separating the
/// two locks would only create a way to interleave them.
struct Weight {
    /// Kept alive so the fn pointers in [`Self::api`] outlive every context
    /// and every allocation below, whatever order fields are dropped in.
    _library: Arc<Library>,
    api: Api,
    ctx: RknnContext,
    n: usize,
    mode: Mode,
    /// One symmetric scale per output channel, for [`Mode::Int8`]; empty for
    /// [`Mode::Fp16`], which needs none.
    ///
    /// **Per channel and not per tensor.** A GGUF K-quant carries a scale
    /// every 32-256 elements, and collapsing a whole matrix onto one scale
    /// throws away the dynamic range between its rows — `crate::npu_ort`
    /// reached the same conclusion from the other direction and documents the
    /// experiments behind it. One per row is the finest this device's integer
    /// path can use without `rknn_matmul_set_quant_params`, and it costs a
    /// multiply per output element on the host.
    scales: Vec<f32>,
    /// One per [`TOKEN_RUNGS`] entry, from `create_dynamic_shape`.
    attrs: Vec<MatmulIoAttr>,
    state: Mutex<State>,
    residency: Arc<Residency>,
    /// What this weight has claimed from [`Self::residency`], so `Drop` can
    /// hand back exactly that and no more. Grows as rungs are allocated.
    reserved: Mutex<u64>,
}

struct State {
    /// The weight's own fp16 copy, bound once and left bound.
    b: *mut TensorMem,
    /// Lazily allocated: a model that only ever decodes never pays for the
    /// prefill rungs.
    io: Vec<Option<Io>>,
    /// Which rung's memory is currently bound to the context, so a steady
    /// stream of same-width calls rebinds nothing. Rebinding is not free —
    /// it is where the runtime converts `B` into its native layout.
    bound: Option<usize>,
}

// SAFETY: every pointer below is device memory this `Weight` owns
// exclusively, and every use of one happens under `state`'s lock. Nothing
// here is shared with another `Weight` or with the runtime outside a call.
unsafe impl Send for Weight {}
unsafe impl Sync for Weight {}

impl Drop for Weight {
    fn drop(&mut self) {
        let state = self.state.get_mut().expect("rknpu weight state poisoned");
        // SAFETY: every pointer was returned by `create_mem` on this
        // context and has not been destroyed yet; the context is destroyed
        // last, after everything allocated from it.
        unsafe {
            for io in state.io.iter().flatten() {
                (self.api.destroy_mem)(self.ctx, io.a);
                (self.api.destroy_mem)(self.ctx, io.c);
            }
            (self.api.destroy_mem)(self.ctx, state.b);
            (self.api.destroy)(self.ctx);
        }
        let reserved = *self.reserved.get_mut().expect("rknpu reservation poisoned");
        self.residency.release(reserved);
    }
}

impl Weight {
    /// Builds the device-side copy of `w`, or `None` if the device refuses
    /// any step of it.
    ///
    /// Never an error and never a panic, because there is always a correct
    /// answer available without the device: the caller keeps this weight on
    /// `CpuBackend`.
    fn build(
        library: &Arc<Library>,
        api: Api,
        residency: &Arc<Residency>,
        mode: Mode,
        k: usize,
        n: usize,
        rows: &(impl Fn(usize) -> Vec<f32> + Sync),
    ) -> Option<Self> {
        let mut info = MatmulInfo {
            m: TOKEN_RUNGS[0] as i32,
            k: k as i32,
            n: n as i32,
            kind: mode.matmul_type(),
            // Normal layouts throughout. The native layouts are faster and
            // are a different, undocumented-per-part packing for each of
            // `A`, `B` and `C`; the runtime converts from normal itself, and
            // for `B` it does so once, at bind time, which is where the cost
            // belongs.
            b_layout: 0,
            // Per-layer, which here means "the runtime applies no scaling at
            // all": in [`Mode::Int8`] the device returns raw `int32`
            // accumulations and every scale is applied on the host. Asking
            // for per-channel would hand the runtime a quantization story it
            // would then need `rknn_matmul_set_quant_params` to be told, for
            // no gain — and measured, it is 8% *slower* at M=512.
            b_quant_type: 0,
            ac_layout: 0,
            ac_quant_type: 0,
            iommu_domain_id: 0,
            group_size: 0,
            reserved: [0; 34],
        };
        let mut shapes: Vec<MatmulShape> = TOKEN_RUNGS
            .iter()
            .map(|&m| MatmulShape {
                m: m as i32,
                k: k as i32,
                n: n as i32,
            })
            .collect();
        let mut attrs = vec![
            MatmulIoAttr {
                a: zeroed_attr(),
                b: zeroed_attr(),
                c: zeroed_attr(),
            };
            TOKEN_RUNGS.len()
        ];

        let mut ctx: RknnContext = 0;
        // SAFETY: `shapes` and `attrs` are both `TOKEN_RUNGS.len()` long,
        // which is the count passed, and both outlive the call.
        let status = unsafe {
            (api.create_dynamic_shape)(
                &mut ctx,
                &mut info,
                TOKEN_RUNGS.len() as c_int,
                shapes.as_mut_ptr(),
                attrs.as_mut_ptr(),
            )
        };
        if status != RKNN_SUCC || ctx == 0 {
            return None;
        }

        // Built before anything else is allocated, so a context that turns
        // out to describe something other than what was asked for is
        // destroyed here rather than used. A driver that answered with a
        // different element type would otherwise be handed an fp16 buffer
        // to read as something else.
        let weight = Self {
            _library: library.clone(),
            api,
            ctx,
            n,
            mode,
            scales: Vec::new(),
            attrs,
            state: Mutex::new(State {
                b: std::ptr::null_mut(),
                io: (0..TOKEN_RUNGS.len()).map(|_| None).collect(),
                bound: None,
            }),
            residency: residency.clone(),
            reserved: Mutex::new(0),
        };
        let (a_kind, b_kind, c_kind) = mode.tensor_kinds();
        let element = mode.element_bytes();
        if !weight.attrs.iter().enumerate().all(|(rung, attr)| {
            attr.a.kind == a_kind
                && attr.b.kind == b_kind
                && attr.c.kind == c_kind
                && attr.a.size as usize >= TOKEN_RUNGS[rung] * k * element
                && attr.c.size as usize >= TOKEN_RUNGS[rung] * n * 4
                && attr.b.size as usize >= k * n * element
        }) {
            return None;
        }

        let b_size = weight.attrs[0].b.size;
        if !weight.claim(u64::from(b_size)) {
            return None;
        }
        // SAFETY: the context was created above and is live until this
        // `Weight` drops.
        let b = unsafe { (api.create_mem)(ctx, b_size) };
        if b.is_null() {
            residency.note_device_refusal();
            return None;
        }
        // SAFETY: `create_mem` returned non-null, so `virt_addr` is a
        // mapping of `b_size` bytes owned by this context.
        let b_addr = unsafe { (*b).virt_addr };
        if b_addr.is_null() {
            // SAFETY: `b` came from this context's `create_mem`.
            unsafe { (api.destroy_mem)(ctx, b) };
            return None;
        }
        // SAFETY: `b_addr` is `b_size` bytes, and `k * n * element` was
        // checked against `attr.b.size` above.
        let b_bytes =
            unsafe { std::slice::from_raw_parts_mut(b_addr.cast::<u8>(), k * n * element) };
        let mut weight = weight;
        weight.scales = write_b(b_bytes, mode, k, n, rows);

        weight.state.lock().expect("rknpu weight state poisoned").b = b;

        // **Proved before it is used, every weight, not just the first.**
        //
        // This is what lets the shape gate be alignment and nothing else.
        // The vendor header states a maximum `K` that the runtime does not
        // enforce, and a byte alignment that is not the element alignment it
        // actually applies — so the written contract and the implemented one
        // differ, in both directions, on the one device this has been run
        // on. Checking the arithmetic is a stronger guarantee than either
        // document, and the only one that would catch a `K` at which this
        // particular runtime goes quietly wrong.
        //
        // Same principle as `VulkanBackend`'s per-kernel cross-checks: the
        // failure being guarded against is not a crash, it is a model that
        // answers slightly worse with no error anywhere.
        if !weight.agrees_with_host(k, rows) {
            return None;
        }
        Some(weight)
    }

    /// Takes `bytes` from the shared budget and remembers it as this
    /// weight's, so `Drop` returns exactly what was taken.
    fn claim(&self, bytes: u64) -> bool {
        if !self.residency.try_reserve(bytes) {
            return false;
        }
        *self.reserved.lock().expect("rknpu reservation poisoned") += bytes;
        true
    }

    /// Hands back a claim whose allocation then failed.
    fn release(&self, bytes: u64) {
        *self.reserved.lock().expect("rknpu reservation poisoned") -= bytes;
        self.residency.release(bytes);
    }

    /// Whether one matmul through this weight matches a host reference.
    ///
    /// A **sample** of output columns, not all of them: the device computes
    /// every column either way, and a host reference over all of them costs
    /// `K × N` multiply-adds — on a vocabulary-sized `N` that is seconds of
    /// CPU per weight at load. [`VERIFY_COLUMNS`] evenly spaced columns catch
    /// a systematically wrong result, which is the only kind a matmul unit
    /// has.
    ///
    /// The activations are mixed in sign and magnitude rather than all ones,
    /// because an all-ones input passes against a device that returns a
    /// column sum for every element.
    fn agrees_with_host(&self, k: usize, rows: &(impl Fn(usize) -> Vec<f32> + Sync)) -> bool {
        let x: Vec<f32> = (0..k).map(|i| ((i % 17) as f32 - 8.0) * 0.0625).collect();
        let mut y = Vec::new();
        if !self.run_into(&mut y, &x, 1, k) || y.len() != self.n {
            return false;
        }

        // The reference reproduces this mode's own arithmetic — fp16 operands,
        // or int8 operands with the two symmetric scales reapplied — rather
        // than the f32 product. Comparing against f32 would charge the device
        // for the backend's precision choice and make the tolerance a guess;
        // what is under test here is whether the device multiplies what it was
        // given, which it should do to within rounding.
        // One token, so its scale is the whole of `quantize_activations`'s
        // answer for this input.
        let a_scale = int8_scale(&x);
        let a_reciprocal = 1.0 / a_scale;
        let step = (self.n / VERIFY_COLUMNS).max(1);
        (0..self.n).step_by(step).all(|column| {
            let row = rows(column);
            let want: f32 = match self.mode {
                Mode::Fp16 => row
                    .iter()
                    .zip(&x)
                    .map(|(w, x)| f32::from(f16::from_f32(*w)) * f32::from(f16::from_f32(*x)))
                    .sum(),
                Mode::Int8 => {
                    let w_scale = self.scales[column];
                    let w_reciprocal = 1.0 / w_scale;
                    let acc: i32 = row
                        .iter()
                        .zip(&x)
                        .map(|(w, x)| {
                            i32::from(quantize_i8(*w, w_reciprocal))
                                * i32::from(quantize_i8(*x, a_reciprocal))
                        })
                        .sum();
                    acc as f32 * a_scale * w_scale
                }
            };
            (y[column] - want).abs() < 1e-3 * want.abs().max(1.0)
        })
    }

    /// `y = x × W` on the device into `out`, or `false` when the device
    /// declined and the caller should use the CPU.
    ///
    /// An out-parameter rather than a returned `Vec`, because this is the hot
    /// path: a prefill stripe's output is `n_tokens × N` floats — hundreds of
    /// kilobytes — and returning it by value allocated and freed that on every
    /// matmul of every layer. [`Backend::matmul_into`] exists precisely so a
    /// caller can keep one buffer across a whole forward pass, and this is the
    /// half of that bargain the backend has to hold up.
    ///
    /// `out` is overwritten, not appended to, and is left in an unspecified
    /// state when this returns `false` — every caller either uses it or has the
    /// CPU overwrite it.
    fn run_into(&self, out: &mut Vec<f32>, x: &[f32], n_tokens: usize, k: usize) -> bool {
        let Some(rung) = TOKEN_RUNGS.iter().position(|&m| m >= n_tokens) else {
            return false;
        };
        let m = TOKEN_RUNGS[rung];
        let mut state = self.state.lock().expect("rknpu weight state poisoned");

        if state.io[rung].is_none() {
            let attr = &self.attrs[rung];
            // **Scratch is charged for, like the weight is.** `A` and `C`
            // belong to this weight's context, so they are per weight, and at
            // a prefill width they are tens of MiB each — on a model with
            // hundreds of weights they are the larger half of what this
            // backend puts on the device. A rung whose scratch will not fit
            // is declined, and the call goes to the CPU at that width while
            // the widths already allocated keep working.
            if !self.claim(u64::from(attr.a.size) + u64::from(attr.c.size)) {
                return false;
            }
            // SAFETY: the context is live for as long as `self` is.
            let a = unsafe { (self.api.create_mem)(self.ctx, attr.a.size) };
            // SAFETY: as above.
            let c = unsafe { (self.api.create_mem)(self.ctx, attr.c.size) };
            if a.is_null() || c.is_null() {
                self.residency.note_device_refusal();
                self.release(u64::from(attr.a.size) + u64::from(attr.c.size));
                // SAFETY: each pointer, if non-null, came from this
                // context's `create_mem` and has not been destroyed.
                unsafe {
                    if !a.is_null() {
                        (self.api.destroy_mem)(self.ctx, a);
                    }
                    if !c.is_null() {
                        (self.api.destroy_mem)(self.ctx, c);
                    }
                }
                return false;
            }
            // `A` is zeroed once, here, and never again: every call writes
            // rows `0..n_tokens` and `n_tokens <= m`, so the padding rows
            // keep whatever is put in them now. Leaving them as whatever the
            // allocator returned would make a padded call's own rows correct
            // and still have the device read uninitialized memory for the
            // rest.
            //
            // **`C` is deliberately not zeroed, and must not be.** This
            // memory is cacheable and the device writes it by DMA, so a CPU
            // write to `C` leaves dirty cache lines that win over what the
            // device puts in DRAM — the readback returns the CPU's zeros.
            // Measured on an RK3588: zeroing `C` here makes results wrong
            // (a 4096×64 fp16 matmul came back 0.141 off, and a smaller one
            // came back entirely zero), and `rknn_mem_sync(FROM_DEVICE)`
            // does **not** repair it. Not zeroing it is also not a gap: the
            // device writes all `m * n` outputs every run, and only rows
            // `0..n_tokens` are ever read.
            //
            // SAFETY: `a` is a non-null mapping of `attr.a.size` bytes.
            unsafe {
                std::ptr::write_bytes((*a).virt_addr.cast::<u8>(), 0, attr.a.size as usize);
            }
            state.io[rung] = Some(Io { a, c });
        }

        let io = state.io[rung].as_ref().expect("just allocated").a;
        let io_c = state.io[rung].as_ref().expect("just allocated").c;

        // Only rows `0..n_tokens` are written; the rest were zeroed once at
        // allocation and stay zero. See that allocation for why `C` is not
        // treated the same way.
        let a_scales = match self.mode {
            Mode::Fp16 => {
                // SAFETY: `io` is this rung's `A`, at least `m * k * 2` bytes.
                let a =
                    unsafe { std::slice::from_raw_parts_mut((*io).virt_addr.cast::<f16>(), m * k) };
                // `half`'s own slice conversion rather than a loop of
                // `f16::from_f32`: it is the vectorized path, and on aarch64
                // that is the hardware instruction. This runs over `M * K`
                // elements on every matmul, so it is hot — at a prefill width
                // it is millions of conversions per call.
                a[..n_tokens * k].convert_from_f32_slice(x);
                Vec::new()
            }
            Mode::Int8 => {
                // SAFETY: `io` is this rung's `A`, at least `m * k` bytes.
                let a =
                    unsafe { std::slice::from_raw_parts_mut((*io).virt_addr.cast::<i8>(), m * k) };
                quantize_activations(&mut a[..n_tokens * k], x, n_tokens, k)
            }
        };

        if state.bound != Some(rung) {
            let mut shape = MatmulShape {
                m: m as i32,
                k: k as i32,
                n: self.n as i32,
            };
            let mut attr = self.attrs[rung];
            let b = state.b;
            // SAFETY: every pointer belongs to this context; `attr` is a
            // copy of the descriptor the runtime itself produced for this
            // rung, and lives across all four calls.
            let ok = unsafe {
                (self.api.set_dynamic_shape)(self.ctx, &mut shape) == RKNN_SUCC
                    && (self.api.set_io_mem)(self.ctx, io, &mut attr.a) == RKNN_SUCC
                    && (self.api.set_io_mem)(self.ctx, b, &mut attr.b) == RKNN_SUCC
                    && (self.api.set_io_mem)(self.ctx, io_c, &mut attr.c) == RKNN_SUCC
            };
            if !ok {
                state.bound = None;
                return false;
            }
            state.bound = Some(rung);
        }

        // No `rknn_mem_sync` either side of this, which is worth saying
        // because the zero-copy pattern invites it. Measured on an RK3588:
        // rewriting `A` in place and re-running with no rebind and no sync
        // is exact over repeated iterations, and syncing does not change any
        // result. It is not free — a flush spans `M * K * 2` bytes, which at
        // a prefill width is megabytes of cache maintenance per matmul — so
        // calling it would be paying for nothing. See `write_activations`
        // for the one coherency hazard that is real here, which sync does
        // *not* fix.
        //
        // SAFETY: the context is live and every input is bound.
        if unsafe { (self.api.run)(self.ctx) } != RKNN_SUCC {
            return false;
        }

        out.clear();
        out.reserve(n_tokens * self.n);
        match self.mode {
            Mode::Fp16 => {
                // SAFETY: `io_c` is this rung's `C`, at least `m * n * 4` bytes.
                let c = unsafe {
                    std::slice::from_raw_parts((*io_c).virt_addr.cast::<f32>(), m * self.n)
                };
                out.extend_from_slice(&c[..n_tokens * self.n]);
            }
            Mode::Int8 => {
                // SAFETY: as above, read as the `int32` the runtime said `C`
                // is in this mode.
                let c = unsafe {
                    std::slice::from_raw_parts((*io_c).virt_addr.cast::<i32>(), m * self.n)
                };
                // `y = C * a_scale[token] * w_scale[channel]` — the two
                // symmetric scales recombined, one multiply per output element,
                // with the token's own scale hoisted out of the inner loop.
                for (token, a_scale) in a_scales.iter().enumerate() {
                    let row = &c[token * self.n..(token + 1) * self.n];
                    out.extend(
                        row.iter()
                            .zip(&self.scales)
                            .map(|(acc, w_scale)| *acc as f32 * a_scale * *w_scale),
                    );
                }
            }
        }
        true
    }
}

fn zeroed_attr() -> MatmulTensorAttr {
    MatmulTensorAttr {
        name: [0; RKNN_MAX_NAME_LEN],
        n_dims: 0,
        dims: [0; RKNN_MAX_DIMS],
        size: 0,
        kind: 0,
    }
}

/// Writes `n` rows of `k` weights into `b` in the device's `(K, N)` order,
/// returning the per-output-channel scales [`Mode::Int8`] needs (empty for
/// [`Mode::Fp16`]).
///
/// The transpose is the whole reason this is not two lines. A `QuantMatrix`
/// is `out_dim` rows of `in_dim` weights — `(N, K)` — and `B` is `(K, N)`, so
/// every source row becomes a strided column. `rows` rather than the matrix
/// itself so the bring-up self-check in [`device_answers_correctly`] goes
/// through this same code.
///
/// Parallelized over source rows, which is also what makes the per-channel
/// scale cheap: a row's scale is a property of that row alone, so the worker
/// that dequantizes it computes it, with nothing shared and no second pass.
fn write_b(
    b: &mut [u8],
    mode: Mode,
    k: usize,
    n: usize,
    rows: &(impl Fn(usize) -> Vec<f32> + Sync),
) -> Vec<f32> {
    debug_assert_eq!(b.len(), k * n * mode.element_bytes());

    /// A raw pointer that may cross into a `rayon` worker. See the SAFETY
    /// note at each write below for why the aliasing is sound.
    struct Column(*mut u8);
    // SAFETY: the pointer is only ever offset to indices owned by the
    // sending thread's own row, and the target outlives the scope.
    unsafe impl Send for Column {}
    unsafe impl Sync for Column {}

    let base = Column(b.as_mut_ptr());
    (0..n)
        .into_par_iter()
        .map(|row| {
            let base = &base;
            let values = rows(row);
            debug_assert_eq!(values.len(), k);
            match mode {
                Mode::Fp16 => {
                    for (i, value) in values.iter().enumerate() {
                        // SAFETY: `row < n` and `i < k`, so the element index
                        // `i * n + row` is below `k * n`, and `b` holds that
                        // many `f16`. Two iterations write one index only if
                        // they share both `i` and `row`, and `row` is unique
                        // per task — so no two workers touch one element.
                        unsafe {
                            base.0
                                .cast::<f16>()
                                .add(i * n + row)
                                .write(f16::from_f32(*value))
                        };
                    }
                    0.0
                }
                Mode::Int8 => {
                    // Symmetric, so there is no zero point to carry and a zero
                    // weight stays exactly zero.
                    let scale = int8_scale(&values);
                    let reciprocal = 1.0 / scale;
                    for (i, value) in values.iter().enumerate() {
                        let q = quantize_i8(*value, reciprocal);
                        // SAFETY: as the fp16 arm above, with one byte per
                        // element instead of two.
                        unsafe { base.0.cast::<i8>().add(i * n + row).write(q) };
                    }
                    scale
                }
            }
        })
        .collect()
}

/// The symmetric int8 scale for a set of values: the one that maps their
/// largest magnitude onto [`INT8_MAX`].
///
/// A set that is entirely zero gets a scale of 1 rather than 0, so the reverse
/// multiply is always defined.
#[inline]
fn int8_scale(values: &[f32]) -> f32 {
    let peak = values.iter().fold(0f32, |m, v| m.max(v.abs()));
    if peak > 0.0 { peak / INT8_MAX } else { 1.0 }
}

/// The **one** definition of symmetric int8 quantization in this module, taking
/// the *reciprocal* of the scale.
///
/// Shared by every site that quantizes or predicts a quantization — the weight
/// writer, the activation path, the per-weight proof, and the tests'
/// reference. That is not tidiness: the first version had the weight writer
/// multiply by the reciprocal and the proof divide by the scale, which differ
/// in the last bit, and a value sitting on a rounding boundary then quantized
/// to a different integer in the two places. Two of the cross-checks failed
/// on it — one weight declined outright because the device "disagreed" with a
/// reference that had quantized it differently.
///
/// The reciprocal rather than a division because this runs over `M * K`
/// activations on every matmul; keeping it the reciprocal everywhere is what
/// lets the hot path be fast *and* bit-identical to the references.
#[inline]
fn quantize_i8(value: f32, reciprocal: f32) -> i8 {
    (value * reciprocal).round().clamp(-INT8_MAX, INT8_MAX) as i8
}

/// Quantizes `x` into `a` as symmetric int8, **one scale per token**,
/// returning those scales.
///
/// Per token and not per tensor, and the difference is not subtle. The device
/// only offers per-channel quantization for `B`, so `A`'s granularity is
/// entirely the host's choice — and a prefill chunk's rows are different
/// positions in a prompt with genuinely different magnitudes, so one scale
/// across all of them sizes every row by the worst row in the batch. Measured
/// end to end, a per-tensor scale was enough to turn "count from one to
/// twenty" into `20`: the prompt's own forward pass came out degraded, which
/// poisoned everything generated after it. Per token costs nothing — the
/// rescale below was already a multiply per output element — and it is exact
/// for `M = 1`.
///
/// Outliers *within* a token are the remaining risk, and they are the same
/// risk `crate::npu_ort` documents and smooths for the Zhouyi path. The lever
/// left here is the device's per-group weight quantization; until that is
/// built, `ORANGU_NPU_MODE=fp16` is one environment variable away.
fn quantize_activations(a: &mut [i8], x: &[f32], n_tokens: usize, k: usize) -> Vec<f32> {
    debug_assert_eq!(x.len(), n_tokens * k);
    debug_assert!(a.len() >= n_tokens * k);
    x.chunks_exact(k)
        .zip(a.chunks_exact_mut(k))
        .map(|(row, out)| {
            let scale = int8_scale(row);
            let reciprocal = 1.0 / scale;
            for (dst, src) in out.iter_mut().zip(row) {
                *dst = quantize_i8(*src, reciprocal);
            }
            scale
        })
        .collect()
}

// ---------------------------------------------------------------------------
// The backend
// ---------------------------------------------------------------------------

pub struct RknpuBackend {
    library: Arc<Library>,
    api: Api,
    /// `None` for a weight this device declined, so the decision — which
    /// costs a context creation and a dequantize — is made once per weight
    /// and not once per call.
    weights: Mutex<HashMap<WeightCacheKey, Option<Arc<Weight>>>>,
    residency: Arc<Residency>,
    mode: Mode,
    min_tokens: usize,
    /// Whether a run that had a resident weight has already fallen back to
    /// the CPU and said so. Once is informative; once per matmul is a log
    /// flood in the middle of a decode loop.
    warned: AtomicBool,
    cpu: CpuBackend,
    /// The device's own name — for the startup banner. Which library
    /// answered is reported by [`Self::try_init_index`] as it binds it,
    /// rather than kept here: the banner's label has room for one of the
    /// two, and the other is worth a line of its own.
    pub device_name: String,
}

impl RknpuBackend {
    /// The machine's RKNPU as a device list — one entry or none.
    ///
    /// Both halves have to hold: `orangu::npu` must see the kernel driver
    /// *and* the RKNN runtime must load. Either alone is a machine where
    /// this backend cannot run a single matmul, and offering it as a
    /// candidate would turn the `auto` chain's silent "try the next one"
    /// into a hard failure at bring-up.
    pub fn devices() -> Vec<DeviceCandidate> {
        let Some(info) = rknpu_info() else {
            return Vec::new();
        };
        if !device_node_is_openable(&info) {
            return Vec::new();
        }
        let Some((library, api, runtime)) = open_runtime() else {
            return Vec::new();
        };
        // The same proof `try_init_index` demands, for the same reason: a
        // candidate this list offers and bring-up then refuses is a hard
        // error in `select_backend`, not a fall-through to the next backend.
        // The two must not be able to disagree.
        if !device_answers_correctly(&library, api, mode()) {
            return Vec::new();
        }
        vec![DeviceCandidate {
            index: 0,
            name: format!("{} {}", info.vendor, info.target),
            class: DeviceClass::Npu,
            // The NPU has no memory of its own: it reads the same DRAM the
            // CPU does, through an IOMMU. Reporting a number here would
            // describe a pool that doesn't exist.
            vram_total_bytes: None,
            id: None,
            driver: info.driver.clone().or_else(|| {
                runtime
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
            }),
        }]
    }

    /// **Tests only** — see `OpenClBackend::try_init`. `select_backend`
    /// goes through [`Self::devices`] and [`Self::try_init_index`].
    #[cfg(test)]
    pub fn try_init() -> Option<Self> {
        Self::try_init_index(0)
    }

    /// Binds the RKNN runtime, or `None` on a machine that has no RKNPU,
    /// no runtime library, or a library that isn't this API.
    ///
    /// `index` exists to match every other backend's shape; the device list
    /// is one long, so anything but `0` is a device this machine doesn't
    /// have.
    pub fn try_init_index(index: usize) -> Option<Self> {
        if index != 0 {
            return None;
        }
        let info = rknpu_info()?;
        if !device_node_is_openable(&info) {
            return None;
        }
        let (library, api, runtime) = open_runtime()?;
        // **Before this backend is offered to anything.** Everything up to
        // here is filesystem evidence — a driver bound in sysfs, a library
        // that opens — and none of it says the device can be reached, let
        // alone that it computes correctly. On this project's own board it
        // could not: the render node is `root:render 0660` and a server not
        // in that group gets every `rknn_matmul_create` refused. Without
        // this check `auto` selected "NPU" there and then ran every single
        // matmul on the CPU behind a one-line warning, which is the worst
        // of the available outcomes — slower than `backend = cpu` and
        // reported as something else.
        //
        // Same reasoning as `VulkanBackend`'s kernel cross-checks (see
        // **The GPU is checked before it is trusted** in the manual): the
        // cost is one tiny matmul at start-up, and it is the only thing
        // standing between a wrong device and a quietly wrong run.
        let mode = mode();
        if !device_answers_correctly(&library, api, mode) {
            return None;
        }
        let budget = budget_bytes();
        let device_name = format!("{} {}", info.vendor, info.target);
        // Said at bring-up, because the two facts an operator needs about
        // this backend are not in the banner's one-word label and cannot be
        // worked out from it: which library is driving the device, and how
        // much of the model is even eligible to land on it. A budget of 0 is
        // the case most worth saying out loud — the backend is selected, and
        // every matmul runs on the CPU.
        eprintln!(
            "orangu-server: [npu] {device_name} via {} [{} weights, budget {}, \
             CPU below {} tokens]",
            runtime.display(),
            mode.label(),
            orangu::format::format_bytes(budget),
            min_tokens(),
        );
        Some(Self {
            library,
            api,
            weights: Mutex::new(HashMap::new()),
            residency: Arc::new(Residency::new(budget)),
            mode,
            min_tokens: min_tokens(),
            warned: AtomicBool::new(false),
            cpu: CpuBackend,
            device_name,
        })
    }

    /// How many bytes of fp16 weights are currently resident.
    ///
    /// **Tests only**, and it earns its place there: every cross-check in
    /// this module would pass just as well with the device declining each
    /// weight and `CpuBackend` quietly answering, so one test has to be able
    /// to assert that an offload actually happened.
    #[cfg(test)]
    pub fn resident_bytes(&self) -> u64 {
        self.residency.used()
    }

    /// Whether the device can take this shape at all, before anything is
    /// dequantized or allocated.
    ///
    /// Alignment only — see [`K_ALIGN_ELEMS`] for why there is no maximum
    /// `K` here and [`Weight::agrees_with_host`] for what stands in its
    /// place. A shape this admits can still be refused afterwards, by the
    /// residency budget or by the arithmetic check; this is the cheap test
    /// that runs first.
    fn shape_is_supported(mode: Mode, w: &QuantMatrix) -> bool {
        w.in_dim.is_multiple_of(mode.k_align())
            && w.out_dim.is_multiple_of(mode.n_align())
            && w.in_dim > 0
            && w.out_dim > 0
    }

    /// This weight's device-side copy, building it on first sight, or
    /// `None` for one that stays on the CPU.
    fn weight(&self, w: &QuantMatrix) -> Option<Arc<Weight>> {
        let key = w.cache_key();
        if let Some(existing) = self
            .weights
            .lock()
            .expect("rknpu weight cache poisoned")
            .get(&key)
        {
            return existing.clone();
        }

        let built = self.build_weight(w);
        self.weights
            .lock()
            .expect("rknpu weight cache poisoned")
            .insert(key, built.clone());
        built
    }

    /// [`Self::weight`]'s miss path, split out so the cache lock is not
    /// held across a dequantize of the whole matrix.
    fn build_weight(&self, w: &QuantMatrix) -> Option<Arc<Weight>> {
        if !Self::shape_is_supported(self.mode, w) {
            return None;
        }
        // The cheap refusal first: once the device is full there is nothing to
        // be learned by dequantizing a whole matrix to find out again.
        if self.residency.is_exhausted() {
            return None;
        }
        Weight::build(
            &self.library,
            self.api,
            &self.residency,
            self.mode,
            w.in_dim,
            w.out_dim,
            &|row| w.row(row),
        )
        .map(Arc::new)
    }

    /// Says once, and only once, that the device declined a weight it had
    /// already accepted — which is a real fault (a driver reset, memory
    /// pressure) rather than the ordinary "this shape stays on the CPU".
    fn warn_fallback(&self) {
        if !self.warned.swap(true, Ordering::Relaxed) {
            eprintln!(
                "orangu-server: [npu] a resident weight failed on the device; \
                 this matmul and any like it are running on the CPU instead"
            );
        }
    }
}

impl Backend for RknpuBackend {
    /// See [`Backend::reduced_surface`] — this backend implements
    /// [`Backend::matmul`] and nothing else.
    fn reduced_surface(&self) -> Option<&'static str> {
        Some(super::MATMUL_ONLY_SURFACE)
    }

    /// **Yes — this backend is the host, for everything but a wide matmul.**
    ///
    /// Narrow calls and every decode go to `CpuBackend` by design, and the
    /// weights the device never accepted go there too, so on a real model the
    /// overwhelming majority of this backend's work happens on the CPU. The
    /// one caller, `arch::moe_overlap`, is deciding whether overlapping host
    /// work with device work can pay; answering "device" would have it plan an
    /// overlap against a device that is not going to be running most of this.
    fn is_host(&self) -> bool {
        self.cpu.is_host()
    }

    /// **No, and this is worth 27% of prefill.**
    ///
    /// The default is `true`, and the trait is right to make it so: a backend
    /// that cannot prove it has no device to lose must keep the adaptive
    /// chunker. This one can prove it. The property the chunker protects
    /// against is a *single submission* whose duration grows with the chunk —
    /// which is what a Vulkan command buffer holding a whole forward pass is.
    /// Here a submission is one `rknn_matmul_run` over at most
    /// `TOKEN_RUNGS`'s widest rung; a long prefill pass is thousands of short
    /// independent submissions, and no individual one gets longer because the
    /// chunk did. There is nothing for a watchdog to catch.
    ///
    /// Taking the default cost real throughput, not theory: measured, this
    /// backend with a **zero** weight budget — delegating every single matmul
    /// to `CpuBackend` and touching the device not at all — prefilled at 12.0
    /// tok/s against `backend = cpu`'s 16.4 on the same fresh prompt. The
    /// adaptive policy was the difference.
    fn has_submission_timeout(&self) -> bool {
        self.cpu.has_submission_timeout()
    }

    /// Every type, because the device never sees a quantized block: a
    /// weight is dequantized through `QuantMatrix::row` on the host and
    /// made resident as fp16, so whatever `row` can decode this backend can
    /// run — and whatever it declines falls to `CpuBackend`, which decodes
    /// the same set.
    fn supports_type(&self, _ggml_type: u32) -> bool {
        true
    }

    fn matmul(&self, x: &[f32], n_tokens: usize, w: &QuantMatrix) -> Vec<f32> {
        let mut out = Vec::new();
        self.matmul_into(&mut out, x, n_tokens, w);
        out
    }

    /// The form every hot call site uses, and the one this backend actually
    /// implements — see [`Weight::run_into`] for the allocation it saves and
    /// why that is worth an override here when `CudaBackend` and
    /// `OpenClBackend` both leave it to the default.
    fn matmul_into(&self, out: &mut Vec<f32>, x: &[f32], n_tokens: usize, w: &QuantMatrix) {
        // Narrow calls never reach the device, whichever entry point they came
        // through — see [`min_tokens`], which is where the measurement behind
        // this lives. Before `self.weight`, so a model that only ever decodes
        // does not pay for a residency it would never use.
        if n_tokens < self.min_tokens {
            self.cpu.matmul_into(out, x, n_tokens, w);
            return;
        }
        let Some(weight) = self.weight(w) else {
            self.cpu.matmul_into(out, x, n_tokens, w);
            return;
        };
        if !weight.run_into(out, x, n_tokens, w.in_dim) {
            self.warn_fallback();
            self.cpu.matmul_into(out, x, n_tokens, w);
        }
    }

    /// **Decode runs on the CPU.** Measured on an RK3588 serving
    /// `gemma-4-E2B-it` at `Q4_K_M`, all three producing identical text:
    ///
    /// | backend | decode | prefill (fresh prompt) |
    /// |---|---|---|
    /// | `cpu` | 5.27 tok/s | 16.4 tok/s |
    /// | `npu` (this module) | 5.12 tok/s | **21.2 tok/s** |
    /// | `npu`, decode on the device | 2.27 tok/s | — |
    /// | `opencl` (Mali-G610) | 0.82 tok/s | 9.3 tok/s |
    ///
    /// Decode is bandwidth-bound and this device is on the wrong side of it.
    /// Its operands are quantized copies, so a weight it holds is read at 1
    /// byte per element where `CpuBackend` reads the `Q4_K` original at 0.56;
    /// its fixed cost is ~0.4 ms per call against a decode step that is
    /// nothing but small matmuls; and the ~2 GiB ceiling means much of the
    /// model is not resident anyway, so a decode step would pay dispatch for
    /// part of the model and the CPU's own bandwidth for the rest. Sending it
    /// to the CPU outright is 2.3× faster than that.
    ///
    /// Prefill keeps the device, and wins: it is compute-bound, the fixed cost
    /// amortizes over a wide call, and int8 at `M = 512` is 1190 GFLOP/s
    /// against eight Cortex-A55s managing roughly 65.
    ///
    /// **Measure prefill on a prompt the server has not seen.** An earlier
    /// version of this table said 147 tok/s for both backends, because the
    /// benchmark warmed up on the same text it then timed and was reading the
    /// prefix cache — a number that would have required ~590 GFLOP/s from the
    /// CPU, about eight times what those cores can do. It made a real 1.29×
    /// speed-up look like a 0.83× regression.
    ///
    /// This is what [`Backend::matmul_decode`] is for, and the split is safe in
    /// the way that method cares about: `CpuBackend`'s decode kernel does not
    /// vary with `n_tokens`, so a sequence's logits do not depend on how many
    /// others were decoding beside it. It is also not sufficient on its own —
    /// see [`min_tokens`].
    fn matmul_decode(&self, x: &[f32], n_tokens: usize, w: &QuantMatrix) -> Vec<f32> {
        self.cpu.matmul_decode(x, n_tokens, w)
    }

    /// [`Self::matmul_decode`]'s out-parameter form. Forwarded rather than
    /// left to the default, which would route through `matmul_decode` and lose
    /// `CpuBackend`'s own buffer reuse.
    fn matmul_decode_into(&self, out: &mut Vec<f32>, x: &[f32], n_tokens: usize, w: &QuantMatrix) {
        self.cpu.matmul_decode_into(out, x, n_tokens, w);
    }

    /// The batched form of [`Self::matmul_decode`], and it has to be
    /// overridden too: the default would fall through `matmul_batch` to
    /// `matmul`, which is the device — so without this, a batched decode would
    /// quietly take the slow path the method above exists to avoid.
    fn matmul_batch_decode(&self, ops: &[MatmulOp<'_>]) -> Vec<Vec<f32>> {
        self.cpu.matmul_batch_decode(ops)
    }

    /// A batch whose every op is too narrow for the device is handed to
    /// `CpuBackend` whole, rather than one op at a time through this type.
    ///
    /// Not a micro-optimization: `CpuBackend::matmul_batch_into` reuses the
    /// caller's inner buffers across the batch and across layers, and routing
    /// the batch through the default here would rebuild that decision per op.
    /// A batch with any wide op falls to the default, which sends each op to
    /// [`Self::matmul_into`] and so lets the device have the ones it can take.
    fn matmul_batch_into(&self, outs: &mut Vec<Vec<f32>>, ops: &[MatmulOp<'_>]) {
        if ops.iter().all(|op| op.n_tokens < self.min_tokens) {
            self.cpu.matmul_batch_into(outs, ops);
            return;
        }
        *outs = self.matmul_batch(ops);
    }
}

/// Whether the device can be reached and computes correctly, checked with
/// one matmul at the smallest shape its alignment rules admit.
///
/// [`Weight::build`] already proves every weight it builds, so this is that
/// same proof on a throwaway weight — which makes it the *real* path rather
/// than a shortcut: the context creation, the fp16 transpose, the shape
/// selection, the binding and the readback are all the code that will serve
/// the model. A check that bypassed any of them would pass on a machine
/// where that step is the broken one.
fn device_answers_correctly(library: &Arc<Library>, api: Api, mode: Mode) -> bool {
    let k = mode.k_align();
    let rows = |row: usize| -> Vec<f32> {
        (0..k)
            .map(|i| ((i + 2 * row) % 7) as f32 * 0.25 - 0.75)
            .collect()
    };
    // Its own budget, not the backend's: this weight is built before there is
    // a backend, it is a few kilobytes, and it is dropped before the first
    // real one.
    let residency = Arc::new(Residency::new(1 << 20));
    Weight::build(library, api, &residency, mode, k, mode.n_align(), &rows).is_some()
}

/// Whether this NPU's device node can actually be opened — checked **before**
/// `librknnrt` is loaded.
///
/// `rknn_matmul_create` fails cleanly without it, so unlike the OpenCL
/// counterpart this guards noise rather than a crash: a server run by a user
/// outside the `render` group printed
///
/// ```text
/// E RKNN: failed to open rknpu module, need to insmod rknpu dirver!
/// E RKNN: failed to open rknn device!
/// ```
///
/// on every single start, from inside a C library, where it reads as a broken
/// installation rather than as a permission the operator has not granted. The
/// node is a file; asking the filesystem is cheaper than a `dlopen` and a
/// context, and it is silent.
///
/// A probe that cannot name a node gets the benefit of the doubt and goes on
/// to try the runtime — the check is here to skip work that is known to fail,
/// not to become a second gate on what counts as an NPU.
fn device_node_is_openable(info: &orangu::npu::NpuInfo) -> bool {
    let Some(node) = &info.render_node else {
        return true;
    };
    std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(node)
        .is_ok()
}

/// The machine's RKNPU, or `None` — [`orangu::npu`]'s inventory probe,
/// narrowed to the one stack this backend drives.
fn rknpu_info() -> Option<orangu::npu::NpuInfo> {
    let info = orangu::npu::detect_npu_inventory()?;
    (info.stack == orangu::npu::NpuStack::Rknpu).then_some(info)
}

/// Opens `librknnrt` and resolves the API, returning the path that answered.
///
/// **Where** to look is `orangu::npu`'s answer, not this module's — the same
/// function the `orangu-server system` report uses, so the library named there
/// is the library bound here. See [`orangu::npu::rknn_runtime`]. The bare
/// soname is kept as a fallback for a machine that has the runtime on the
/// loader path but not at any of the paths that search stats.
fn open_runtime() -> Option<(Arc<Library>, Api, PathBuf)> {
    let candidates = orangu::npu::rknn_runtime()
        .into_iter()
        .chain([PathBuf::from("librknnrt.so")]);
    for candidate in candidates {
        // SAFETY: `dlopen` runs the library's initializers, which is
        // unavoidable for any runtime-loaded driver and is what the vendor
        // stack expects.
        let Ok(library) = (unsafe { Library::new(&candidate) }) else {
            continue;
        };
        if let Some(api) = Api::load(&library) {
            return Some((Arc::new(library), api, candidate));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::backend::probe_blocks::build_block;
    use crate::engine::loader::test_quant_matrix;
    use crate::engine::quant::{GGML_TYPE_F32, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0};

    /// The four struct layouts this module hands to the driver, pinned to the
    /// sizes the vendor headers compile to on this target (checked with the
    /// system toolchain against `rknn_api.h`/`rknn_matmul_api.h`).
    ///
    /// A size that drifts means a field was added, reordered or mistyped
    /// here, and the failure mode of that is not a wrong answer — it is the
    /// driver reading a pointer out of the middle of another field.
    #[test]
    fn ffi_structs_match_the_vendor_headers() {
        assert_eq!(std::mem::size_of::<MatmulInfo>(), 64);
        assert_eq!(std::mem::size_of::<MatmulTensorAttr>(), 332);
        assert_eq!(std::mem::size_of::<MatmulIoAttr>(), 996);
        assert_eq!(std::mem::size_of::<TensorMem>(), 40);
    }

    /// One `RknpuBackend`, shared across this module — the same pattern
    /// `opencl::tests::shared_opencl` uses, and for the same reason: each init
    /// `dlopen`s the vendor runtime and claims the device. Returns `None` on
    /// every machine without an RKNPU, where each test below skips rather than
    /// fails.
    fn shared_rknpu() -> Option<&'static RknpuBackend> {
        static RKNPU: std::sync::OnceLock<Option<RknpuBackend>> = std::sync::OnceLock::new();
        RKNPU.get_or_init(RknpuBackend::try_init).as_ref()
    }

    /// Dimensions every mode accepts, so one set of test shapes exercises
    /// whichever one `ORANGU_NPU_MODE` selected: `K` a multiple of 32 and `N`
    /// of 32, int8's rules being the stricter of the two.
    const TEST_N: usize = 32;

    fn quant_matrix(ggml_type: u32, in_dim: usize, out_dim: usize) -> QuantMatrix {
        let block_elems = crate::engine::quant::block_layout(ggml_type)
            .expect("a type under cross-check has a block layout")
            .1;
        assert!(in_dim.is_multiple_of(block_elems));
        let mut seed = 0x5eed_1234_9abc_def0u64
            ^ (ggml_type as u64) << 32
            ^ (in_dim as u64) << 16
            ^ out_dim as u64;
        let bytes: Vec<u8> = (0..out_dim * (in_dim / block_elems))
            .flat_map(|_| build_block(ggml_type, &mut seed))
            .collect();
        test_quant_matrix(&bytes, ggml_type, in_dim, out_dim)
    }

    /// `y = x × W` reproducing **the active mode's own arithmetic** on the
    /// host: fp16 operands, or int8 operands with a symmetric scale per output
    /// channel and one per activation tensor, recombined.
    ///
    /// Deliberately *not* `CpuBackend::matmul_dequant`, which the Vulkan, CUDA
    /// and OpenCL cross-checks use: those backends keep f32 operands, so that
    /// reference is right for them and wrong here. Using it measured the
    /// backend's precision *choice* — up to 4% on these synthetic blocks,
    /// nearly all of it cancellation — and said nothing about whether the
    /// device multiplies correctly. See the module doc.
    fn mode_reference(mode: Mode, x: &[f32], n_tokens: usize, w: &QuantMatrix) -> Vec<f32> {
        let rows: Vec<Vec<f32>> = (0..w.out_dim).map(|o| w.row(o)).collect();
        let mut y = vec![0f32; n_tokens * w.out_dim];
        for t in 0..n_tokens {
            let xt = &x[t * w.in_dim..(t + 1) * w.in_dim];
            let a_scale = int8_scale(xt);
            let a_reciprocal = 1.0 / a_scale;
            for (o, row) in rows.iter().enumerate() {
                y[t * w.out_dim + o] = match mode {
                    Mode::Fp16 => row
                        .iter()
                        .zip(xt)
                        .map(|(wv, xv)| {
                            f64::from(f32::from(f16::from_f32(*wv)))
                                * f64::from(f32::from(f16::from_f32(*xv)))
                        })
                        .sum::<f64>() as f32,
                    Mode::Int8 => {
                        let w_scale = int8_scale(row);
                        let w_reciprocal = 1.0 / w_scale;
                        let acc: i32 = row
                            .iter()
                            .zip(xt)
                            .map(|(wv, xv)| {
                                i32::from(quantize_i8(*wv, w_reciprocal))
                                    * i32::from(quantize_i8(*xv, a_reciprocal))
                            })
                            .sum();
                        acc as f32 * a_scale * w_scale
                    }
                };
            }
        }
        y
    }

    /// Cross-checks the device against [`mode_reference`].
    ///
    /// The tolerance is 1e-3 and that is not slack: measured, the device
    /// agrees with this reference to about 1e-6, so anything it catches is a
    /// real defect — a layout error, a stale binding, a padded row leaking in,
    /// a scale applied to the wrong channel — rather than arithmetic noise.
    ///
    /// Exercises `matmul`, not `matmul_decode`: decode is routed to the CPU by
    /// design (see [`RknpuBackend::matmul_decode`]), so calling it here would
    /// test `CpuBackend` and skip this device entirely.
    /// `matmul` with the width threshold out of the way, so a narrow call
    /// still reaches the device.
    ///
    /// The threshold ([`min_tokens`]) is a performance rule, and every
    /// correctness test here is about shapes a real model has at *decode*
    /// width — one token, the case where a padding or binding bug is easiest
    /// to see. Without this the cross-checks would all be testing
    /// `CpuBackend`.
    fn on_device(npu: &RknpuBackend, x: &[f32], n_tokens: usize, w: &QuantMatrix) -> Vec<f32> {
        let weight = npu.weight(w).expect("this shape should be offloaded");
        let mut out = Vec::new();
        assert!(
            weight.run_into(&mut out, x, n_tokens, w.in_dim),
            "the device should accept this width"
        );
        out
    }

    fn cross_check(ggml_type: u32, in_dim: usize, out_dim: usize, n_tokens: usize) {
        let Some(npu) = shared_rknpu() else {
            return;
        };
        let w = quant_matrix(ggml_type, in_dim, out_dim);
        let x: Vec<f32> = (0..n_tokens * in_dim)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
            .collect();

        let expected = mode_reference(npu.mode, &x, n_tokens, &w);
        let actual = on_device(npu, &x, n_tokens, &w);
        assert_eq!(expected.len(), actual.len());
        for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
            assert!(
                (e - a).abs() < 1e-3 * e.abs().max(1.0),
                "index {i}: expected {e}, got {a} (mode {}, ggml_type {ggml_type}, \
                 {in_dim}x{out_dim}, n_tokens {n_tokens})",
                npu.mode.label()
            );
        }
    }

    #[test]
    fn matmul_matches_the_host_for_f32() {
        cross_check(GGML_TYPE_F32, 64, TEST_N, 1);
    }

    #[test]
    fn matmul_matches_the_host_for_q8_0() {
        cross_check(GGML_TYPE_Q8_0, 256, TEST_N, 1);
    }

    #[test]
    fn matmul_matches_the_host_for_q4_k() {
        cross_check(GGML_TYPE_Q4_K, 256, TEST_N, 1);
    }

    #[test]
    fn matmul_matches_the_host_for_q6_k() {
        cross_check(GGML_TYPE_Q6_K, 256, TEST_N, 1);
    }

    /// Several tokens at once, including counts that are not rungs — the
    /// padded path — and one that is.
    #[test]
    fn matmul_handles_token_counts_on_and_off_the_rungs() {
        cross_check(GGML_TYPE_Q4_K, 256, TEST_N, 1);
        cross_check(GGML_TYPE_Q4_K, 256, TEST_N, 3);
        cross_check(GGML_TYPE_Q4_K, 256, TEST_N, 8);
        cross_check(GGML_TYPE_Q4_K, 256, TEST_N, 17);
    }

    /// **`K` past the header's documented maximum is not excluded.** The
    /// header says `k <= 10240`; the runtime creates and computes 11008
    /// correctly, and a transformer's `ffn_down` lives exactly there, so
    /// excluding it on the strength of the comment would leave a third of
    /// every FFN on the CPU for no reason. What makes that safe is the
    /// per-weight arithmetic check, which this exercises on a real device.
    #[test]
    fn a_k_past_the_documented_maximum_still_agrees_with_the_host() {
        cross_check(GGML_TYPE_Q4_K, 11008, TEST_N, 1);
    }

    /// **The padded rows must not leak into a later, narrower call.** A wide
    /// call writes rows the next narrow one does not, so if the device read
    /// `A` beyond `n_tokens` the narrow result would depend on what ran before
    /// it. Same weight, same backend, decreasing widths.
    #[test]
    fn a_narrow_call_after_a_wide_one_is_unaffected_by_it() {
        let Some(npu) = shared_rknpu() else {
            return;
        };
        let w = quant_matrix(GGML_TYPE_Q4_K, 256, TEST_N);
        let wide: Vec<f32> = (0..8 * 256)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.2)
            .collect();
        let narrow = wide[..256].to_vec();

        let alone = on_device(npu, &narrow, 1, &w);
        let _ = on_device(npu, &wide, 8, &w);
        let after = on_device(npu, &narrow, 1, &w);

        assert_eq!(alone, after);
    }

    /// **The steady state, which is where coherency bugs live.** A prefill
    /// chunk writes new activations into the same device buffer and re-runs
    /// without rebinding anything, so every call after the first depends on
    /// the device seeing a host write to memory it already holds. One call
    /// proves nothing about that; a sequence with different inputs does.
    #[test]
    fn repeated_calls_on_one_weight_each_see_their_own_activations() {
        let Some(npu) = shared_rknpu() else {
            return;
        };
        let w = quant_matrix(GGML_TYPE_Q4_K, 256, TEST_N);
        for step in 0..6 {
            let x: Vec<f32> = (0..256)
                .map(|i| (((i + step * 5) % 13) as f32 - 6.0) * 0.1)
                .collect();
            let expected = mode_reference(npu.mode, &x, 1, &w);
            let actual = on_device(npu, &x, 1, &w);
            for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
                assert!(
                    (e - a).abs() < 1e-3 * e.abs().max(1.0),
                    "step {step}, index {i}: expected {e}, got {a}"
                );
            }
        }
    }

    /// **How far the backend's quantization moves the answer, as a test rather
    /// than a comment.** Measured against the f32 reference the other backends
    /// use, on synthetic blocks whose random per-block scales make cancellation
    /// about as bad as it gets. A tripwire, not a tight bound: if this fails,
    /// the chosen mode has stopped being acceptable for this path.
    #[test]
    fn quantized_operands_stay_within_a_documented_distance_of_the_f32_answer() {
        let Some(npu) = shared_rknpu() else {
            return;
        };
        let w = quant_matrix(GGML_TYPE_Q4_K, 256, TEST_N);
        let x: Vec<f32> = (0..256).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();

        let f32_answer = CpuBackend.matmul_dequant(&x, 1, &w);
        let device = on_device(npu, &x, 1, &w);
        let worst = f32_answer
            .iter()
            .zip(&device)
            .map(|(e, a)| (e - a).abs() / e.abs().max(1.0))
            .fold(0f32, f32::max);
        assert!(
            worst < 0.5,
            "{} operands are {worst} away from the f32 answer — further than \
             this path has ever measured",
            npu.mode.label()
        );
    }

    /// A shape the device's own limits exclude must be answered correctly
    /// anyway — on the CPU — rather than declined or wrong.
    #[test]
    fn an_unsupported_shape_still_produces_the_cpu_answer() {
        let Some(npu) = shared_rknpu() else {
            return;
        };
        // `out_dim` of 4 is a multiple of neither mode's `N` alignment.
        let w = quant_matrix(GGML_TYPE_Q4_K, 256, 4);
        assert!(!RknpuBackend::shape_is_supported(npu.mode, &w));
        let x: Vec<f32> = (0..256).map(|i| ((i % 7) as f32 - 3.0) * 0.3).collect();

        assert_eq!(npu.matmul(&x, 1, &w), CpuBackend.matmul(&x, 1, &w));
    }

    /// **Decode does not touch the device.** The split is the difference
    /// between this backend being a 2.3× slowdown and not, so it is asserted
    /// rather than left to the call graph: a decode call must return exactly
    /// what `CpuBackend` returns, bit for bit, which it cannot if any part of
    /// it went through fp16 or int8.
    #[test]
    fn decode_is_answered_by_the_cpu_bit_for_bit() {
        let Some(npu) = shared_rknpu() else {
            return;
        };
        let w = quant_matrix(GGML_TYPE_Q4_K, 256, TEST_N);
        let x: Vec<f32> = (0..256).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();

        assert_eq!(
            npu.matmul_decode(&x, 1, &w),
            CpuBackend.matmul_decode(&x, 1, &w)
        );
    }

    /// The shape gate is arithmetic and testable without a device, which
    /// matters because it is what decides which of a real model's weights
    /// reach the NPU at all.
    ///
    /// The alignments asserted here are the ones the runtime *enforces*, which
    /// are not the ones its header documents — see [`K_ALIGN_ELEMS`] — and
    /// `N`'s differs between the two modes, which is why it is per mode.
    #[test]
    fn the_shape_gate_is_the_alignment_the_runtime_actually_enforces() {
        for mode in [Mode::Int8, Mode::Fp16] {
            assert!(RknpuBackend::shape_is_supported(
                mode,
                &quant_matrix(GGML_TYPE_F32, 64, 32)
            ));

            // K a multiple of 16 but not 32 — accepted by the documented rule
            // and rejected by the real one, in both modes.
            assert!(!RknpuBackend::shape_is_supported(
                mode,
                &quant_matrix(GGML_TYPE_F32, 48, 32)
            ));
        }

        // N of 16: fine for fp16, too narrow for int8.
        let narrow = quant_matrix(GGML_TYPE_F32, 64, 16);
        assert!(RknpuBackend::shape_is_supported(Mode::Fp16, &narrow));
        assert!(!RknpuBackend::shape_is_supported(Mode::Int8, &narrow));
    }

    /// `ORANGU_NPU_MODE` accepts the spellings an operator would reach for,
    /// and nothing else — an unrecognized value must not silently select a
    /// mode, which is what `mode()` warns about.
    #[test]
    fn the_mode_knob_parses_only_real_modes() {
        assert_eq!(Mode::parse("int8"), Some(Mode::Int8));
        assert_eq!(Mode::parse(" I8 "), Some(Mode::Int8));
        assert_eq!(Mode::parse("fp16"), Some(Mode::Fp16));
        assert_eq!(Mode::parse("float16"), Some(Mode::Fp16));
        assert_eq!(Mode::parse("int4"), None);
        assert_eq!(Mode::parse(""), None);
    }

    /// **Something has to actually land on the device.** Every cross-check
    /// above would also pass with the weight silently declined and
    /// `CpuBackend` answering, which is precisely the failure this backend is
    /// most likely to have — so one test asserts the offload happened, and
    /// that int8 really is half the bytes.
    #[test]
    fn a_supported_weight_becomes_resident_on_the_device() {
        let Some(npu) = shared_rknpu() else {
            return;
        };
        let w = quant_matrix(GGML_TYPE_Q4_K, 256, 64);
        assert!(RknpuBackend::shape_is_supported(npu.mode, &w));
        let x: Vec<f32> = (0..256).map(|i| ((i % 7) as f32 - 3.0) * 0.3).collect();
        let _ = on_device(npu, &x, 1, &w);

        let weight_bytes = (256 * 64 * npu.mode.element_bytes()) as u64;
        assert!(
            npu.resident_bytes() >= weight_bytes,
            "this weight's {} copy should be resident, got {} bytes",
            npu.mode.label(),
            npu.resident_bytes()
        );
    }

    /// **A device the process cannot open is not offered.** The node is
    /// `root:render` on a stock image, so this is the ordinary state of a
    /// server run by a user outside that group — and probing anyway made
    /// `librknnrt` write two lines of its own to stderr on every start.
    ///
    /// Asserted against this machine's actual answer rather than a fixture,
    /// because that is the coupling that matters: whatever the node says, the
    /// device list has to agree with it.
    #[test]
    fn an_unopenable_device_node_is_not_offered_as_a_device() {
        let Some(info) = rknpu_info() else {
            return; // No RKNPU here; nothing to agree with.
        };
        let Some(node) = &info.render_node else {
            return; // No node named; the probe gets the benefit of the doubt.
        };
        let openable = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(node)
            .is_ok();
        assert_eq!(
            device_node_is_openable(&info),
            openable,
            "the guard must reflect whether {} can actually be opened",
            node.display()
        );
        if !openable {
            assert!(
                RknpuBackend::devices().is_empty(),
                "a device this process cannot open must not be offered"
            );
        }
    }

    /// A probe that names no node must not be blocked by the guard — it is
    /// there to skip work known to fail, not to become a second definition of
    /// what counts as an NPU.
    #[test]
    fn a_device_with_no_named_node_is_still_tried() {
        let info = orangu::npu::NpuInfo {
            stack: orangu::npu::NpuStack::Rknpu,
            render_node: None,
            vendor: "Rockchip".to_string(),
            target: "RK3588 RKNPU".to_string(),
            partitions: None,
            clusters: None,
            cores: 3,
            driver: Some("RKNPU".to_string()),
            runtime: None,
        };
        assert!(device_node_is_openable(&info));
    }

    /// A machine with no RKNPU must report no device rather than one that
    /// cannot be brought up — the `auto` chain reads an empty list as "try the
    /// next backend" and a listed-but-dead device as a hard failure.
    #[test]
    fn the_device_list_and_the_backend_agree_about_whether_there_is_one() {
        assert_eq!(
            RknpuBackend::devices().is_empty(),
            shared_rknpu().is_none(),
            "devices() and try_init() must not disagree about this machine"
        );
    }
}
