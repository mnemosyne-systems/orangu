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

//! Fused quantized-weight × int8-activation dot products — the same trick
//! ggml's `ggml_vec_dot_q*_q8_K` family uses, and the reason llama.cpp is
//! several times faster than a dequantize-then-`f32`-dot engine on a CPU.
//!
//! The `engine::backend::cpu` path this replaces did, per output row per
//! token: `QuantMatrix::row` → `quant::dequantize` → a fresh `Vec<f32>` of
//! `in_dim` floats → a scalar `f32` dot. A `perf` profile of a decode step
//! on a Raspberry Pi 4 (Cortex-A72, `Qwen2.5-Coder-0.5B` `Q4_K_M`) put
//! **95%** of all CPU time inside that one closure: 48% in `row`/
//! `dequantize`, 42% in the dot, and ~8% in `malloc`/`free` churning the
//! per-row `Vec`. Attention, RoPE, norms and sampling together were 1.75%.
//!
//! What this module does instead: quantize the *activations* to `int8`
//! once per matmul call (not once per row), then dot them straight against
//! the still-quantized weight bytes with integer NEON. That removes the
//! dequantize, removes the allocation, and replaces scalar `f32` multiplies
//! with 16-wide `int8` ones.
//!
//! Accuracy: quantizing activations to `int8` is lossy, exactly as it is in
//! llama.cpp — that is the accepted tradeoff of this kernel family, not an
//! oversight. Blocks are 32 elements with their own `f32` scale, so the
//! relative error stays around 1e-3 (see the tests, which check every
//! kernel against the `quant::dequantize` reference).
//!
//! Only the five types that actually carry weight in real GGUF releases get
//! a fused kernel — `Q8_0`, `Q5_0`, `Q4_K`, `Q6_K`, `IQ4_NL`. Anything else
//! returns `false` from [`supports`] and the caller keeps its old
//! dequantize path, so this is strictly an addition: no existing type
//! changes behavior.

use super::iq_grids::KVALUES_IQ4NL;
use super::quant::{
    GGML_TYPE_IQ4_NL, GGML_TYPE_Q4_K, GGML_TYPE_Q5_0, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0,
    get_scale_min_k4, read_f16,
};

/// Elements per activation-quantization block — one shared `f32` scale per
/// 32 activations, matching `Q8_0`/`Q5_0`/`Q4_K`'s own 32-element sub-block
/// granularity.
pub const ACT_BLOCK: usize = 32;

/// Elements per *scale group*, the unit both [`UnpackedRow`] and the dot loop
/// work in. 16 rather than 32 because `Q6_K` carries one scale per 16
/// weights; every other supported type simply repeats its scale across the
/// two groups of its 32-element block. Choosing the finest granularity any
/// type needs is what lets one dot loop serve all of them.
pub const GROUP: usize = 16;

/// Activations quantized to `int8`, as parallel arrays.
pub struct ActQ8 {
    /// `int8` activations, one per element.
    pub q: Vec<i8>,
    /// Per-[`ACT_BLOCK`] scale: the original `f32` is `q * d`.
    pub d: Vec<f32>,
    /// Per-[`GROUP`] `sum(q)`, for the asymmetric-min correction `Q4_K`
    /// needs. Per group rather than per block so the dot loop can stay
    /// uniform; a block's sum is just the sum of its two groups'.
    pub sums: Vec<i32>,
}

/// Quantizes `x` to `int8`. `x.len()` must be a multiple of [`ACT_BLOCK`] —
/// guaranteed by [`supports`] before this is ever called.
pub fn quantize_act(x: &[f32]) -> ActQ8 {
    debug_assert_eq!(x.len() % ACT_BLOCK, 0);
    let mut q = vec![0i8; x.len()];
    let mut d = Vec::with_capacity(x.len() / ACT_BLOCK);
    let mut sums = Vec::with_capacity(x.len() / GROUP);
    for (b, chunk) in x.chunks_exact(ACT_BLOCK).enumerate() {
        let amax = chunk.iter().fold(0f32, |m, v| m.max(v.abs()));
        // A block of exact zeros (real: a masked or unused tail) has no
        // scale; leave it 0 so `q * d` reproduces 0 rather than NaN.
        let scale = amax / 127.0;
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        let out = &mut q[b * ACT_BLOCK..(b + 1) * ACT_BLOCK];
        // Walk the block one GROUP at a time with the sum in a register and
        // one `push` per group. Indexing a `sums[i / GROUP]` slot per element
        // instead costs a read-modify-write on every activation and measurably
        // slowed decode (2.05 -> 1.98 tok/s on the reference Pi 4).
        for (half, group) in out.chunks_exact_mut(GROUP).enumerate() {
            let src = &chunk[half * GROUP..(half + 1) * GROUP];
            let mut sum = 0i32;
            for (slot, &v) in group.iter_mut().zip(src) {
                // `round` then clamp: `amax * inv` is exactly 127, and negative
                // values reach -127 only, never -128, so the i8 cast is safe.
                let qi = (v * inv).round().clamp(-127.0, 127.0) as i8;
                *slot = qi;
                sum += qi as i32;
            }
            sums.push(sum);
        }
        d.push(scale);
    }
    ActQ8 { q, d, sums }
}

/// Whether a fused kernel exists for `ggml_type` at this `in_dim`. The
/// `in_dim` check is not paranoia: a GGUF row is only guaranteed to be a
/// whole number of blocks, and `Q4_K`/`Q6_K` need 256 | `in_dim` while
/// `Q8_0`/`Q5_0`/`IQ4_NL` need only 32 — a `Qwen2.5-0.5B` is exactly the
/// case that mixes both (`embedding_length` 896 is 28 blocks of 32 but not
/// a multiple of 256, which is why its layer weights are `Q5_0`/`IQ4_NL`
/// and only its `256`-divisible `ffn_down` is a K-quant).
pub fn supports(ggml_type: u32, in_dim: usize) -> bool {
    match ggml_type {
        GGML_TYPE_Q8_0 | GGML_TYPE_Q5_0 | GGML_TYPE_IQ4_NL => in_dim.is_multiple_of(32),
        GGML_TYPE_Q4_K | GGML_TYPE_Q6_K => in_dim.is_multiple_of(256),
        _ => false,
    }
}

/// One weight row, unpacked from its quantized bytes into plain `int8` plus
/// per-[`GROUP`] scale metadata, in a form that is identical for every
/// supported `ggml_type`. Every type's value reduces to
///
/// ```text
/// weight[i] = scale[i / GROUP] * q[i]   -   min[i / GROUP]
/// ```
///
/// with `min` zero for the symmetric types (`Q8_0`, `Q5_0`, `Q6_K` — the
/// latter two fold their `-16`/`-32` bias directly into the `int8` value, so
/// no correction term survives). Only `Q4_K` is genuinely asymmetric.
///
/// This exists so a row is unpacked **once per matmul rather than once per
/// (row, token)**. During decode (`n_tokens == 1`) that is a wash, but for
/// prefill it removes an entire re-unpack per extra token — the reason
/// prefill lagged decode badly relative to llama.cpp.
///
/// Reuse one of these across rows via [`UnpackedRow::new`] +
/// [`unpack_row`]; the buffers are resized only when a wider row appears.
pub struct UnpackedRow {
    /// `int8` weights, one per element.
    q: Vec<i8>,
    /// Per-[`GROUP`] multiplier applied to that group's integer dot.
    scale: Vec<f32>,
    /// Per-[`GROUP`] term subtracted, weighted by the group's activation sum.
    min: Vec<f32>,
    /// Whether any entry of `min` is nonzero. Lets the dot loop skip the
    /// correction entirely for symmetric types, which are 92% of a typical
    /// `Q4_K_M` model's weight bytes.
    has_min: bool,
    /// Whether `scale`/`min` are indexed per 32-element block rather than per
    /// [`GROUP`]. True for every type whose scale is uniform across a block —
    /// `Q8_0`, `Q5_0` and `Q4_K`, together ~90% of a `Q4_K_M` model. Only
    /// `Q6_K` genuinely varies per 16. Halves both the horizontal reductions
    /// and the `f32` work in the dot loop, worth ~20% on the prefill GEMM.
    per32: bool,
}

impl UnpackedRow {
    pub fn new() -> Self {
        Self {
            q: Vec::new(),
            scale: Vec::new(),
            min: Vec::new(),
            has_min: false,
            per32: false,
        }
    }

    /// Sizes the buffers for `in_dim` elements at the given scale stride
    /// (32 for the uniform-scale types, [`GROUP`] for `Q6_K`). Always resizes
    /// both, so switching a reused buffer between types cannot leave a stale
    /// tail.
    fn resize_for(&mut self, in_dim: usize, stride: usize) {
        let n = in_dim / stride;
        self.q.resize(in_dim, 0);
        self.q.truncate(in_dim);
        self.scale.resize(n, 0.0);
        self.scale.truncate(n);
        self.min.resize(n, 0.0);
        self.min.truncate(n);
        self.per32 = stride == 32;
    }
}

impl Default for UnpackedRow {
    fn default() -> Self {
        Self::new()
    }
}

/// Unpacks one weight row (`QuantMatrix::row_bytes()` long) into `out`.
/// `ggml_type`/`in_dim` must have passed [`supports`].
pub fn unpack_row(ggml_type: u32, row: &[u8], in_dim: usize, out: &mut UnpackedRow) {
    // Only `Q6_K` needs per-GROUP scales; everything else is uniform per block.
    let stride = if ggml_type == GGML_TYPE_Q6_K {
        GROUP
    } else {
        32
    };
    out.resize_for(in_dim, stride);
    match ggml_type {
        GGML_TYPE_Q8_0 => unpack_q8_0(row, out),
        GGML_TYPE_Q5_0 => unpack_q5_0(row, out),
        GGML_TYPE_IQ4_NL => unpack_iq4_nl(row, out),
        GGML_TYPE_Q4_K => unpack_q4_k(row, out),
        GGML_TYPE_Q6_K => unpack_q6_k(row, out),
        // Unreachable via `supports`, but a wrong answer here would be a
        // silently corrupt forward pass, so make it loud instead.
        other => panic!("vecdot::unpack_row called for unsupported ggml_type {other}"),
    }
}

/// `sum_i weight[i] * act[i]` for an already-unpacked row and a single token.
///
/// Production always goes through `dot_unpacked_multi`, which handles any
/// token count including one; this is kept as the reference that the tiled
/// path is tested against.
#[cfg(test)]
fn dot_unpacked(w: &UnpackedRow, act: &ActQ8) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if have_dotprod() {
        return dot_unpacked_impl::<ISA_DOTPROD>(w, act);
    }
    dot_unpacked_impl::<ISA_BASELINE>(w, act)
}

fn dot_unpacked_impl<const ISA: u8>(w: &UnpackedRow, act: &ActQ8) -> f32 {
    debug_assert_eq!(w.q.len(), act.q.len());
    let mut total = 0f32;
    if w.per32 {
        for b in 0..w.scale.len() {
            let isum = dot32::<ISA>(&w.q[b * 32..], &act.q[b * 32..]);
            total += if w.has_min {
                act.d[b] * (w.scale[b] * isum as f32 - w.min[b] * block_sum(act, b) as f32)
            } else {
                act.d[b] * w.scale[b] * isum as f32
            };
        }
    } else {
        for g in 0..w.scale.len() {
            let isum = dot16::<ISA>(&w.q[g * GROUP..], &act.q[g * GROUP..]);
            // `d` is per ACT_BLOCK, i.e. one scale shared by two groups.
            total += act.d[g * GROUP / ACT_BLOCK]
                * (w.scale[g] * isum as f32 - w.min[g] * act.sums[g] as f32);
        }
    }
    total
}

/// One-shot form of [`unpack_row`] + [`dot_unpacked`], allocating its own
/// buffer. Test-only: production callers either reuse a buffer across rows
/// (prefill) or take the fused [`dot_row`] path (decode), so shipping this
/// would just be an allocation waiting to be used by mistake.
#[cfg(test)]
fn dot_via_unpack(ggml_type: u32, row: &[u8], act: &ActQ8, in_dim: usize) -> f32 {
    let mut w = UnpackedRow::new();
    unpack_row(ggml_type, row, in_dim, &mut w);
    dot_unpacked(&w, act)
}

/// Which SIMD kernel a monomorphized dot loop should use.
///
/// A plain `u8` rather than an enum because const generics accept only
/// integral types. Threaded through `dot16`/`dot32` and every kernel so the
/// choice is resolved **at compile time inside a `#[target_feature]`
/// wrapper**, which is the whole point: Rust will not inline a
/// `#[target_feature]` function into a caller that lacks the feature, so a
/// leaf kernel selected by a runtime `if` becomes a real `call` per
/// 16-element block and loses more than the instruction gains. Selecting the
/// ISA once per row, inside a wrapper that already declares the feature, lets
/// the whole chain collapse into straight-line SIMD.
pub(crate) const ISA_BASELINE: u8 = 0;
/// aarch64 ARMv8.2 `sdot`.
#[cfg(target_arch = "aarch64")]
pub(crate) const ISA_DOTPROD: u8 = 1;
/// x86-64 AVX2.
#[cfg(target_arch = "x86_64")]
pub(crate) const ISA_AVX2: u8 = 2;
/// x86-64 AVX-512 VNNI (`vpdpbusd`) via the 128-bit `vl` form.
#[cfg(target_arch = "x86_64")]
pub(crate) const ISA_VNNI: u8 = 3;

/// `sum_i w[i] * x[i]` over the first **16** `int8` pairs of each slice.
///
/// 16 is the natural unit: it is one 128-bit vector, and it is also `Q6_K`'s
/// scale granularity, so every kernel here is either one call (`Q6_K` half)
/// or two ([`dot32`]).
///
/// Both slices must be at least 16 long; callers pass fixed-size buffers.
#[inline(always)]
fn dot16<const ISA: u8>(w: &[i8], x: &[i8]) -> i32 {
    debug_assert!(w.len() >= 16 && x.len() >= 16);
    #[cfg(target_arch = "aarch64")]
    {
        if ISA == ISA_DOTPROD {
            // Safety: only reachable from a path `have_dotprod()` approved.
            return unsafe { dot16_sdot(w, x) };
        }
        // Safety: operands are >= 16 bytes; NEON is baseline on aarch64.
        return unsafe { dot16_neon(w, x) };
    }
    #[cfg(target_arch = "x86_64")]
    {
        if ISA == ISA_VNNI {
            // Safety: only reachable from `dot_*_vnni`, which declares the
            // features and was approved by `have_vnni()`.
            return unsafe { dot16_vnni(w, x) };
        }
        if ISA == ISA_AVX2 {
            // Safety: only reachable from `dot_*_avx2`, which declares avx2.
            return unsafe { dot16_avx2(w, x) };
        }
        // ISA_BASELINE on x86: SSE4.1 if the CPU has it. Runtime-checked
        // because this path is also what a pre-SSE4.1 CPU reaches.
        if is_x86_feature_detected!("sse4.1") {
            // Safety: guarded by the runtime feature check above.
            return unsafe { dot16_sse41(w, x) };
        }
    }
    #[allow(unreachable_code)]
    dot16_scalar(w, x)
}

/// Whether the ARMv8.2 `dotprod` (`sdot`) kernel may be used.
///
/// Two conditions, both required. First, the CPU must advertise `dotprod`.
/// Second — and this is not belt-and-braces — [`dot16_sdot`] must actually
/// agree with [`dot16_neon`] on known inputs. The reference platform for this
/// code is a Cortex-A72 (ARMv8.0), which has no `sdot` and no way to emulate
/// it, so that kernel's inline assembly has never been *executed* during
/// development. Rather than ship it on trust, it is validated once on the
/// machine that will run it, and quietly declined if it misbehaves.
///
/// Resolved once per process and then checked once per weight row — never per
/// 16-element block — before being baked into a monomorphized kernel via
/// `dot16`'s `DOTPROD` const parameter, so the inner loop carries no feature
/// test and no branch.
#[cfg(target_arch = "aarch64")]
fn have_dotprod() -> bool {
    use std::sync::OnceLock;
    static USABLE: OnceLock<bool> = OnceLock::new();
    *USABLE.get_or_init(|| {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return false;
        }
        // Cases chosen to catch the plausible failure modes of the asm
        // constraints: sign handling on both operands, the i8 extremes, and a
        // non-trivial mix so a dropped lane or a missing accumulate shows up.
        const CASES: [([i8; 16], [i8; 16]); 4] = [
            ([1; 16], [1; 16]),
            ([-1; 16], [1; 16]),
            (
                [
                    127, -128, 127, -128, 1, -1, 0, 64, -64, 32, -32, 16, -16, 8, -8, 2,
                ],
                [1; 16],
            ),
            (
                [
                    3, -5, 7, -11, 13, -17, 19, -23, 29, -31, 37, -41, 43, -47, 53, -59,
                ],
                [
                    2, 4, -6, 8, -10, 12, 14, -16, 18, -20, 22, 24, -26, 28, 30, -32,
                ],
            ),
        ];
        CASES.iter().all(|(w, x)| {
            // Safety: `dotprod` was just detected; both slices are 16 bytes.
            let via_sdot = unsafe { dot16_sdot(w, x) };
            // Safety: NEON is baseline on aarch64; both slices are 16 bytes.
            let via_neon = unsafe { dot16_neon(w, x) };
            via_sdot == via_neon
        })
    })
}

/// Whether the AVX-512 VNNI kernel may be used.
///
/// Same two-part contract as [`have_dotprod`]: the CPU must advertise the
/// features, **and** [`dot16_vnni`] must agree with the AVX2 kernel on known
/// inputs. `vpdpbusd` multiplies *unsigned* by *signed* bytes, so the signed
/// weights are passed as `|w|` against `sign(w) * x`; that rearrangement is
/// the part worth verifying, and no x86 machine was available while it was
/// written. A mismatch quietly falls back to AVX2.
#[cfg(target_arch = "x86_64")]
fn have_vnni() -> bool {
    use std::sync::OnceLock;
    static USABLE: OnceLock<bool> = OnceLock::new();
    *USABLE.get_or_init(|| {
        if !is_x86_feature_detected!("avx512vnni")
            || !is_x86_feature_detected!("avx512vl")
            || !is_x86_feature_detected!("avx2")
            || !is_x86_feature_detected!("ssse3")
        {
            return false;
        }
        const CASES: [([i8; 16], [i8; 16]); 4] = [
            ([1; 16], [1; 16]),
            ([-1; 16], [1; 16]),
            (
                [
                    127, -128, 127, -128, 1, -1, 0, 64, -64, 32, -32, 16, -16, 8, -8, 2,
                ],
                [1; 16],
            ),
            (
                [
                    3, -5, 7, -11, 13, -17, 19, -23, 29, -31, 37, -41, 43, -47, 53, -59,
                ],
                [
                    2, 4, -6, 8, -10, 12, 14, -16, 18, -20, 22, 24, -26, 28, 30, -32,
                ],
            ),
        ];
        CASES.iter().all(|(w, x)| {
            // Safety: the features above were just detected; slices are 16 bytes.
            let a = unsafe { dot16_vnni(w, x) };
            let b = unsafe { dot16_avx2(w, x) };
            a == b
        })
    })
}

/// `sdot`: one instruction accumulates four `int8` products into each of four
/// `i32` lanes, so 16 pairs take a single `sdot` instead of the two
/// `vmull_s8` widenings plus two `vpadalq_s16` accumulations the ARMv8.0 path
/// needs.
///
/// Written as inline assembly rather than `vdotq_s32` because that intrinsic
/// is still gated behind the unstable `stdarch_neon_dotprod` feature; this
/// crate builds on stable. The encoding is fixed and unambiguous —
/// `sdot <Vd>.4s, <Vn>.16b, <Vm>.16b` — and `have_dotprod` cross-checks the
/// result against the baseline kernel at startup, so a mistake in the operand
/// constraints degrades to the ARMv8.0 path instead of corrupting a forward
/// pass.
///
/// Absent on the Cortex-A72 reference platform, so this is verified to compile
/// and to emit `sdot`, but **not executed** there — see `doc/PERFORMANCE.md`.
///
/// Deliberately *not* `#[target_feature(enable = "dotprod")]`. Rust refuses to
/// inline a `#[target_feature]` function into a caller that lacks the feature,
/// so the attribute would turn every 16-element dot into a real `bl` call and
/// throw away more than `sdot` gains — `objdump` showed exactly one `sdot`
/// behind a call, rather than one inlined per block. `asm!` emits the
/// instruction regardless of declared features, and `have_dotprod` guarantees
/// at runtime that the CPU can execute it, so the attribute buys nothing here.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dot16_sdot(w: &[i8], x: &[i8]) -> i32 {
    use std::arch::aarch64::*;
    unsafe {
        let wv = vld1q_s8(w.as_ptr());
        let xv = vld1q_s8(x.as_ptr());
        let mut acc = vdupq_n_s32(0);
        std::arch::asm!(
            // The assembler still gates the mnemonic even though `asm!` itself
            // does not check target features, so enable it locally. This is why
            // the block can stay free of `#[target_feature]` and still inline.
            ".arch_extension dotprod",
            "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
            acc = inout(vreg) acc,
            a = in(vreg) wv,
            b = in(vreg) xv,
            options(pure, nomem, nostack)
        );
        vaddvq_s32(acc)
    }
}

/// `sum_i w[i] * x[i]` over 32 `int8` pairs.
///
/// Not simply two [`dot16`] calls: that reduces to a scalar twice, and a
/// horizontal reduction is multi-cycle and serializing on every ISA here.
/// These keep a vector accumulator across both halves and reduce **once**.
/// Every supported type except `Q6_K` shares one scale across a 32-element
/// block, so this is the natural unit for them — worth ~20% on the prefill
/// GEMM. Kept per-ISA so AVX2, SSE4.1 and AVX-512 all get the single-reduction
/// form, not just NEON.
///
/// * **aarch64 (NEON)** — `vmull_s8` widens to `i16` lanes (max
///   |127*127| = 16129, so no overflow) and `vpadalq_s16` accumulates
///   pairwise into `i32`. This is deliberately *not* `vdotq_s32`: a
///   Cortex-A72 has NEON but not `dotprod`/`i8mm`/SVE, and the widening
///   chain is exactly what ggml falls back to on such a core. NEON is
///   baseline on aarch64, so no runtime check is needed.
/// * **x86_64 (AVX-512 VNNI / AVX2 / SSE4.1)** — chosen by runtime feature
///   detection, never assumed from the compile-time baseline, matching how
///   `engine::tensor::dot` already dispatches. `vpdpbusd` (VNNI) does the
///   whole thing in one instruction; AVX2/SSE4.1 sign-extend to `i16` and
///   use `madd_epi16`, which is exact for this range.
/// * **anything else** — a scalar loop, which LLVM autovectorizes
///   acceptably for `i8`->`i32` since the accumulation is integer and
///   therefore reassociable (unlike the `f32` sum this whole module
///   replaces).
#[inline(always)]
fn dot32<const ISA: u8>(w: &[i8], x: &[i8]) -> i32 {
    debug_assert!(w.len() >= 32 && x.len() >= 32);
    #[cfg(target_arch = "aarch64")]
    {
        if ISA == ISA_DOTPROD {
            // Safety: only reachable from a path `have_dotprod()` approved.
            return unsafe { dot32_sdot(w, x) };
        }
        // Safety: operands are >= 32 bytes; NEON is baseline on aarch64.
        return unsafe { dot32_neon(w, x) };
    }
    #[cfg(target_arch = "x86_64")]
    {
        if ISA == ISA_VNNI {
            // Safety: reachable only from a `#[target_feature]` wrapper that
            // declares the features, approved by `have_vnni()`.
            return unsafe { dot32_vnni(w, x) };
        }
        if ISA == ISA_AVX2 {
            // Safety: reachable only from a wrapper that declares avx2.
            return unsafe { dot32_avx2(w, x) };
        }
        if is_x86_feature_detected!("sse4.1") {
            // Safety: guarded by the runtime feature check above.
            return unsafe { dot32_sse41(w, x) };
        }
    }
    #[allow(unreachable_code)]
    {
        dot16_scalar(w, x) + dot16_scalar(&w[16..], &x[16..])
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dot32_neon(w: &[i8], x: &[i8]) -> i32 {
    use std::arch::aarch64::*;
    unsafe {
        let mut a = vdupq_n_s32(0);
        for h in 0..2 {
            let wv = vld1q_s8(w.as_ptr().add(h * 16));
            let xv = vld1q_s8(x.as_ptr().add(h * 16));
            a = vpadalq_s16(a, vmull_s8(vget_low_s8(wv), vget_low_s8(xv)));
            a = vpadalq_s16(a, vmull_s8(vget_high_s8(wv), vget_high_s8(xv)));
        }
        vaddvq_s32(a)
    }
}

/// See [`dot16_sdot`] for why this is `asm!` rather than `vdotq_s32`, and why
/// it deliberately carries no `#[target_feature]`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dot32_sdot(w: &[i8], x: &[i8]) -> i32 {
    use std::arch::aarch64::*;
    unsafe {
        let mut acc = vdupq_n_s32(0);
        for h in 0..2 {
            let wv = vld1q_s8(w.as_ptr().add(h * 16));
            let xv = vld1q_s8(x.as_ptr().add(h * 16));
            std::arch::asm!(
                ".arch_extension dotprod",
                "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
                acc = inout(vreg) acc,
                a = in(vreg) wv,
                b = in(vreg) xv,
                options(pure, nomem, nostack)
            );
        }
        vaddvq_s32(acc)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot32_avx2(w: &[i8], x: &[i8]) -> i32 {
    use std::arch::x86_64::*;
    unsafe {
        let mut acc = _mm256_setzero_si256();
        for h in 0..2 {
            let wv = _mm256_cvtepi8_epi16(_mm_loadu_si128(w.as_ptr().add(h * 16) as *const _));
            let xv = _mm256_cvtepi8_epi16(_mm_loadu_si128(x.as_ptr().add(h * 16) as *const _));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(wv, xv));
        }
        hsum_epi32_avx2(acc)
    }
}

/// Two `vpdpbusd` into one accumulator — see [`dot16_vnni`] for the
/// unsigned/signed rearrangement.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512vnni,avx512vl,ssse3")]
unsafe fn dot32_vnni(w: &[i8], x: &[i8]) -> i32 {
    use std::arch::x86_64::*;
    unsafe {
        let mut acc = _mm_setzero_si128();
        for h in 0..2 {
            let wv = _mm_loadu_si128(w.as_ptr().add(h * 16) as *const _);
            let xv = _mm_loadu_si128(x.as_ptr().add(h * 16) as *const _);
            acc = _mm_dpbusd_epi32(acc, _mm_abs_epi8(wv), _mm_sign_epi8(xv, wv));
        }
        let mut t = _mm_add_epi32(acc, _mm_shuffle_epi32(acc, 0b01_00_11_10));
        t = _mm_add_epi32(t, _mm_shuffle_epi32(t, 0b00_00_00_01));
        _mm_cvtsi128_si32(t)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn dot32_sse41(w: &[i8], x: &[i8]) -> i32 {
    use std::arch::x86_64::*;
    unsafe {
        let mut acc = _mm_setzero_si128();
        for q in 0..4 {
            let wv = _mm_cvtepi8_epi16(_mm_loadl_epi64(w.as_ptr().add(q * 8) as *const _));
            let xv = _mm_cvtepi8_epi16(_mm_loadl_epi64(x.as_ptr().add(q * 8) as *const _));
            acc = _mm_add_epi32(acc, _mm_madd_epi16(wv, xv));
        }
        let mut t = _mm_add_epi32(acc, _mm_shuffle_epi32(acc, 0b01_00_11_10));
        t = _mm_add_epi32(t, _mm_shuffle_epi32(t, 0b00_00_00_01));
        _mm_cvtsi128_si32(t)
    }
}

#[inline(always)]
fn dot16_scalar(w: &[i8], x: &[i8]) -> i32 {
    let mut sum = 0i32;
    for i in 0..16 {
        sum += w[i] as i32 * x[i] as i32;
    }
    sum
}

/// Baseline NEON: `vmull_s8` widens 8 pairs to `i16` (max |127*127| = 16129,
/// so no overflow) and `vpadalq_s16` accumulates pairwise into `i32`. This is
/// what an ARMv8.0 core such as the Cortex-A72 reference platform uses, and is
/// the same widening chain ggml falls back to there.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dot16_neon(w: &[i8], x: &[i8]) -> i32 {
    use std::arch::aarch64::*;
    unsafe {
        let wv = vld1q_s8(w.as_ptr());
        let xv = vld1q_s8(x.as_ptr());
        let mut acc = vpadalq_s16(vdupq_n_s32(0), vmull_s8(vget_low_s8(wv), vget_low_s8(xv)));
        acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(wv), vget_high_s8(xv)));
        vaddvq_s32(acc)
    }
}

/// AVX2: sign-extend 16 bytes to `i16` in one 256-bit register, then
/// `madd_epi16`. `maddubs_epi16` would need one operand unsigned (an
/// abs/sign dance); the widening form is exact and simpler.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot16_avx2(w: &[i8], x: &[i8]) -> i32 {
    use std::arch::x86_64::*;
    unsafe {
        let wv = _mm256_cvtepi8_epi16(_mm_loadu_si128(w.as_ptr() as *const _));
        let xv = _mm256_cvtepi8_epi16(_mm_loadu_si128(x.as_ptr() as *const _));
        hsum_epi32_avx2(_mm256_madd_epi16(wv, xv))
    }
}

/// Horizontal sum of eight `i32` lanes. No `unsafe` body: every intrinsic
/// here operates register-to-register, so enabling the target feature is the
/// only precondition — unlike the loads in the kernels above.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn hsum_epi32_avx2(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    let mut s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b00_00_00_01));
    _mm_cvtsi128_si32(s)
}

/// AVX-512 VNNI (via the 128-bit `vl` form): `vpdpbusd` multiplies
/// *unsigned* bytes by *signed* bytes and accumulates into `i32` in a single
/// instruction. Weights here are signed, so pass `|w|` against `x` with
/// `w`'s sign folded in — the standard ggml arrangement, exact because
/// `|w| * sign(w)*x == w * x`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512vnni,avx512vl,ssse3")]
unsafe fn dot16_vnni(w: &[i8], x: &[i8]) -> i32 {
    use std::arch::x86_64::*;
    unsafe {
        let wv = _mm_loadu_si128(w.as_ptr() as *const _);
        let xv = _mm_loadu_si128(x.as_ptr() as *const _);
        let acc = _mm_dpbusd_epi32(_mm_setzero_si128(), _mm_abs_epi8(wv), _mm_sign_epi8(xv, wv));
        let mut s = _mm_add_epi32(acc, _mm_shuffle_epi32(acc, 0b01_00_11_10));
        s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b00_00_00_01));
        _mm_cvtsi128_si32(s)
    }
}

/// SSE4.1: same widening approach as AVX2, eight pairs at a time.
/// `_mm_cvtepi8_epi16` is the SSE4.1 instruction this needs.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn dot16_sse41(w: &[i8], x: &[i8]) -> i32 {
    use std::arch::x86_64::*;
    unsafe {
        let mut acc = _mm_setzero_si128();
        for half in 0..2 {
            let wv = _mm_cvtepi8_epi16(_mm_loadl_epi64(w.as_ptr().add(half * 8) as *const _));
            let xv = _mm_cvtepi8_epi16(_mm_loadl_epi64(x.as_ptr().add(half * 8) as *const _));
            acc = _mm_add_epi32(acc, _mm_madd_epi16(wv, xv));
        }
        let mut s = _mm_add_epi32(acc, _mm_shuffle_epi32(acc, 0b01_00_11_10));
        s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b00_00_00_01));
        _mm_cvtsi128_si32(s)
    }
}

// ------------------------------------------- shared unpack primitives
//
// The bit layouts below are implemented exactly once and called from both the
// GEMV kernels (`dot_q*`) and the GEMM unpackers (`unpack_q*`), which
// previously each carried their own copy.
//
// Only `Q5_0` and `Q6_K` get hand-written NEON. `Q4_K`'s unpack is a plain
// `(byte >> shift) & 0x0F` map that LLVM already auto-vectorizes — measured on
// the reference Pi 4, its unpack costs ~1.2 ms per 512x4864 pass against
// `Q5_0`'s 3.5 ms and `Q6_K`'s 6.3 ms, so intrinsics there would add unsafe
// code for no gain.

/// Unpacks an `IQ4_NL` block's 32 weights into `w`: nibble `j`'s low half is
/// element `j` and its high half element `16 + j`, each selecting one of the
/// 16 non-uniformly spaced levels in [`KVALUES_IQ4NL`].
///
/// The table is `i8` already and spans `-127..=113`, so — as with `Q5_0`'s
/// `-16` bias — the whole value folds into the `int8` weight and no
/// correction term survives. Scalar rather than NEON for the same reason
/// `Q4_K`'s unpack is: this is a 16-entry lookup LLVM turns into a
/// register-resident table, not a bit-shuffle worth intrinsics.
#[inline(always)]
fn unpack_block_iq4_nl(qs: &[u8], w: &mut [i8; 32]) {
    for j in 0..16 {
        w[j] = KVALUES_IQ4NL[(qs[j] & 0x0F) as usize];
        w[j + 16] = KVALUES_IQ4NL[(qs[j] >> 4) as usize];
    }
}

/// Unpacks a `Q5_0` block's 32 weights into `w`: a nibble from `qs` plus a
/// 5th bit from `qh`, biased by -16 so the result fits a signed `int8`.
#[inline(always)]
fn unpack_block_q5_0(qh: u32, qs: &[u8], w: &mut [i8; 32]) {
    #[cfg(target_arch = "aarch64")]
    {
        // Safety: `qs` is >= 16 bytes (a Q5_0 block's payload) and `w` is
        // exactly 32; NEON is baseline on aarch64.
        unsafe { unpack_block_q5_0_neon(qh, qs, w) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for j in 0..16 {
            let hi_lo = ((qh >> j) << 4) & 0x10;
            let hi_hi = (qh >> (j + 12)) & 0x10;
            w[j] = (((qs[j] & 0x0F) as u32 | hi_lo) as i32 - 16) as i8;
            w[16 + j] = (((qs[j] >> 4) as u32 | hi_hi) as i32 - 16) as i8;
        }
    }
}

/// The scalar loop's per-lane `qh >> j` is what defeats autovectorization
/// here. NEON does it with one variable-shift: `vshlq_u8` shifts right when
/// the amount is negative, so lane `j` can receive bit `j` of a broadcast
/// `qh` byte.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn unpack_block_q5_0_neon(qh: u32, qs: &[u8], w: &mut [i8; 32]) {
    use std::arch::aarch64::*;
    const SH: [i8; 16] = [0, -1, -2, -3, -4, -5, -6, -7, 0, -1, -2, -3, -4, -5, -6, -7];
    unsafe {
        let sh = vld1q_s8(SH.as_ptr());
        let one = vdupq_n_u8(1);
        let bias = vdupq_n_s8(16);
        let v = vld1q_u8(qs.as_ptr());
        // The low half takes `qh` bits 0..15 and the high half bits 16..31;
        // within each, lanes 0..7 come from one byte and 8..15 from the next.
        let src_lo = vcombine_u8(vdup_n_u8(qh as u8), vdup_n_u8((qh >> 8) as u8));
        let src_hi = vcombine_u8(vdup_n_u8((qh >> 16) as u8), vdup_n_u8((qh >> 24) as u8));
        let fifth_lo = vshlq_n_u8(vandq_u8(vshlq_u8(src_lo, sh), one), 4);
        let fifth_hi = vshlq_n_u8(vandq_u8(vshlq_u8(src_hi, sh), one), 4);
        let lo = vorrq_u8(vandq_u8(v, vdupq_n_u8(0x0F)), fifth_lo);
        let hi = vorrq_u8(vshrq_n_u8(v, 4), fifth_hi);
        vst1q_s8(w.as_mut_ptr(), vsubq_s8(vreinterpretq_s8_u8(lo), bias));
        vst1q_s8(
            w.as_mut_ptr().add(16),
            vsubq_s8(vreinterpretq_s8_u8(hi), bias),
        );
    }
}

/// `Q6_K`'s four 32-element runs per 128-element half, as
/// `(ql byte offset within the half, qh bit shift, take the high nibble?)`.
/// Mirrors the run order in `quant::dequantize_q6_k`.
const Q6K_RUNS: [(usize, i32, bool); 4] =
    [(0, 0, false), (32, 2, false), (0, 4, true), (32, 6, true)];

/// Unpacks one `Q6_K` run's 32 weights into `w`: a nibble from `ql` plus a bit
/// pair from `qh` forming a 6-bit value, biased by -32 into a signed `int8`.
#[inline(always)]
fn unpack_q6k_run(ql: &[u8], qh: &[u8], hshift: i32, high_nib: bool, w: &mut [i8; 32]) {
    #[cfg(target_arch = "aarch64")]
    {
        // Safety: both slices are >= 32 bytes and `w` is exactly 32; NEON is
        // baseline on aarch64.
        unsafe { unpack_q6k_run_neon(ql, qh, hshift, high_nib, w) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for l in 0..32 {
            let nib = if high_nib { ql[l] >> 4 } else { ql[l] & 0x0F };
            w[l] = (nib | (((qh[l] >> hshift) & 3) << 4)) as i8 - 32;
        }
    }
}

/// `hshift` stays a runtime value rather than a const generic: `vshlq_u8`'s
/// negative-amount right shift handles 0/2/4/6 with one code path, and
/// `#[inline(always)]` at literal call sites lets LLVM fold it anyway. A
/// `vshrq_n_u8` would need a compile-time constant *and* a special case,
/// since an immediate shift of 0 is not encodable.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn unpack_q6k_run_neon(ql: &[u8], qh: &[u8], hshift: i32, high_nib: bool, w: &mut [i8; 32]) {
    use std::arch::aarch64::*;
    unsafe {
        let three = vdupq_n_u8(3);
        let bias = vdupq_n_s8(32);
        let nibmask = vdupq_n_u8(0x0F);
        let shv = vdupq_n_s8(-(hshift as i8));
        for k in 0..2 {
            let qlv = vld1q_u8(ql.as_ptr().add(k * 16));
            let qhv = vld1q_u8(qh.as_ptr().add(k * 16));
            let nib = if high_nib {
                vshrq_n_u8(qlv, 4)
            } else {
                vandq_u8(qlv, nibmask)
            };
            let hb = vandq_u8(vshlq_u8(qhv, shv), three);
            let q = vorrq_u8(nib, vshlq_n_u8(hb, 4));
            vst1q_s8(
                w.as_mut_ptr().add(k * 16),
                vsubq_s8(vreinterpretq_s8_u8(q), bias),
            );
        }
    }
}

// ---------------------------------------------------------------- Q8_0

/// `block_q8_0`: `{ d: f16, qs: [i8; 32] }` — already `int8` on disk, so this
/// is a straight copy plus one scale per 32 weights, duplicated across the
/// block's two groups.
fn unpack_q8_0(row: &[u8], out: &mut UnpackedRow) {
    const BLOCK_BYTES: usize = 2 + 32;
    out.has_min = false;
    for (b, block) in row.chunks_exact(BLOCK_BYTES).enumerate() {
        let dw = read_f16(block, 0);
        out.q[b * 32..b * 32 + 32].copy_from_slice(bytemuck::cast_slice(&block[2..]));
        out.scale[b] = dw;
        out.min[b] = 0.0;
    }
}

// ---------------------------------------------------------------- Q5_0

/// `block_q5_0`: `{ d: f16, qh: u32, qs: [u8; 16] }`, 32 elements. Element
/// `j` (0..16) is `qs[j]`'s low nibble plus bit `j` of `qh` as a 5th bit;
/// element `16+j` is the high nibble plus bit `j+16`. ggml biases by -16, and
/// `q5 - 16` lands in `-16..=15`, so it folds straight into a signed `int8`
/// weight and no correction term survives.
fn unpack_q5_0(row: &[u8], out: &mut UnpackedRow) {
    const BLOCK_BYTES: usize = 2 + 4 + 16;
    out.has_min = false;
    for (b, block) in row.chunks_exact(BLOCK_BYTES).enumerate() {
        let dw = read_f16(block, 0);
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        let w: &mut [i8; 32] = (&mut out.q[b * 32..b * 32 + 32]).try_into().unwrap();
        unpack_block_q5_0(qh, &block[6..], w);
        out.scale[b] = dw;
        out.min[b] = 0.0;
    }
}

// -------------------------------------------------------------- IQ4_NL

/// `block_iq4_nl`: `{ d: f16, qs: [u8; 16] }`, 32 elements — `Q4_0`'s block
/// shape with a codebook lookup in place of the `- 8`. Symmetric like
/// `Q5_0`: one scale per 32, no `min` term.
fn unpack_iq4_nl(row: &[u8], out: &mut UnpackedRow) {
    const BLOCK_BYTES: usize = 2 + 16;
    out.has_min = false;
    for (b, block) in row.chunks_exact(BLOCK_BYTES).enumerate() {
        let dw = read_f16(block, 0);
        let w: &mut [i8; 32] = (&mut out.q[b * 32..b * 32 + 32]).try_into().unwrap();
        unpack_block_iq4_nl(&block[2..], w);
        out.scale[b] = dw;
        out.min[b] = 0.0;
    }
}

// ---------------------------------------------------------------- Q4_K

/// `block_q4_K`: `{ d: f16, dmin: f16, scales: [u8; 12], qs: [u8; 128] }`,
/// 256 elements as eight 32-element sub-blocks. The only *asymmetric* type
/// here — `value = d*sc*q - dmin*m` with `q` unsigned `0..=15` — so the
/// `-dmin*m` part cannot fold into the `int8` weight and is carried in
/// `min`, to be applied against the group's activation sum.
///
/// Sub-block order matches `quant::dequantize_q4_k`: for each of the four
/// 32-byte groups of `qs`, the low nibbles are one sub-block and the high
/// nibbles the next.
fn unpack_q4_k(row: &[u8], out: &mut UnpackedRow) {
    const BLOCK_BYTES: usize = 2 + 2 + 12 + 128;
    out.has_min = true;
    let mut sb = 0usize; // 32-element sub-block index across the whole row
    for block in row.chunks_exact(BLOCK_BYTES) {
        let d = read_f16(block, 0);
        let dmin = read_f16(block, 2);
        let scales = &block[4..16];
        let qs = &block[16..];
        for g in 0..4 {
            let bytes = &qs[g * 32..g * 32 + 32];
            for (half, shift) in [(0usize, 0u32), (1, 4)] {
                let (sc, m) = get_scale_min_k4(g * 2 + half, scales);
                let w = &mut out.q[(sb + half) * 32..(sb + half) * 32 + 32];
                for (j, &byte) in bytes.iter().enumerate() {
                    w[j] = ((byte >> shift) & 0x0F) as i8;
                }
                // One scale per 32-element sub-block.
                out.scale[sb + half] = d * sc as f32;
                out.min[sb + half] = dmin * m as f32;
            }
            sb += 2;
        }
    }
}

// ---------------------------------------------------------------- Q6_K

/// `block_q6_K`: `{ ql: [u8; 128], qh: [u8; 64], scales: [i8; 16], d: f16 }`,
/// 256 elements. Per `quant::dequantize_q6_k` the scale for element `e` is
/// `scales[e / 16]` — i.e. genuinely per-[`GROUP`], which is why [`GROUP`] is
/// 16 — and `value = d * scales[e/16] * (q - 32)` with `q` a 6-bit `0..=63`.
/// `q - 32` lands in `-32..=31`, so like `Q5_0` the bias folds into the
/// `int8` weight and no correction term survives.
fn unpack_q6_k(row: &[u8], out: &mut UnpackedRow) {
    const BLOCK_BYTES: usize = 128 + 64 + 16 + 2;
    out.has_min = false;
    for (sblk, block) in row.chunks_exact(BLOCK_BYTES).enumerate() {
        let ql = &block[0..128];
        let qh = &block[128..192];
        let sc = &block[192..208];
        let d = read_f16(block, 208);
        let base = sblk * 256;
        for h in 0..2 {
            let qh_run = &qh[h * 32..h * 32 + 32];
            for (run, &(ql_add, hshift, high)) in Q6K_RUNS.iter().enumerate() {
                let e0 = base + h * 128 + run * 32;
                let ql_run = &ql[h * 64 + ql_add..h * 64 + ql_add + 32];
                let w: &mut [i8; 32] = (&mut out.q[e0..e0 + 32]).try_into().unwrap();
                unpack_q6k_run(ql_run, qh_run, hshift, high, w);
                // This 32-element run spans two scale groups of 16.
                let gi = e0 / GROUP;
                out.scale[gi] = d * sc[gi % 16] as i8 as f32;
                out.scale[gi + 1] = d * sc[(gi + 1) % 16] as i8 as f32;
                out.min[gi] = 0.0;
                out.min[gi + 1] = 0.0;
            }
        }
    }
}

/// Dots one unpacked row against **several tokens at once**, writing one
/// result per token — the prefill (GEMM) entry point.
///
/// The obvious shape, calling the single-token path once per token, walks the whole
/// weight row again for every token. An unpacked row is small enough to stay
/// in L1 (4.8 KiB at `in_dim` 4864), so that is not a bandwidth problem — but
/// it does reload the same weight vector from L1 on every token and leaves a
/// single dependent accumulator chain. Tiling over tokens loads each weight
/// vector once and runs [`TOKEN_TILE`] independent chains, which is worth
/// **+34%** at `in_dim` 896 and +6% at 4864 on the reference platform
/// (4 threads, 64 tokens). 896 is the width of every matmul in a
/// `Qwen2.5-0.5B` except `ffn_down`.
///
/// Note this is the opposite shape to blocking over weight *rows*, which was
/// measured to be a loss for decode — that multiplies concurrent weight
/// streams, whereas this reduces weight reads and tiles over activations,
/// which are small and cache-resident.
pub fn dot_unpacked_multi(w: &UnpackedRow, acts: &[ActQ8], out: &mut [f32]) {
    debug_assert_eq!(acts.len(), out.len());
    #[cfg(target_arch = "aarch64")]
    if have_dotprod() {
        return dot_unpacked_multi_impl::<ISA_DOTPROD>(w, acts, out);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if have_vnni() {
            // Safety: see `dot_row`.
            return unsafe { dot_unpacked_multi_vnni(w, acts, out) };
        }
        if is_x86_feature_detected!("avx2") {
            // Safety: guarded by the runtime feature check above.
            return unsafe { dot_unpacked_multi_avx2(w, acts, out) };
        }
    }
    dot_unpacked_multi_impl::<ISA_BASELINE>(w, acts, out)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512vnni,avx512vl,avx2,ssse3")]
unsafe fn dot_unpacked_multi_vnni(w: &UnpackedRow, acts: &[ActQ8], out: &mut [f32]) {
    dot_unpacked_multi_impl::<ISA_VNNI>(w, acts, out)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_unpacked_multi_avx2(w: &UnpackedRow, acts: &[ActQ8], out: &mut [f32]) {
    dot_unpacked_multi_impl::<ISA_AVX2>(w, acts, out)
}

/// Like [`dot_unpacked_multi`] but for **two weight rows at once**, sharing
/// each token's activation load between them.
///
/// The register tile becomes 2 rows x [`TOKEN_TILE`] tokens: an activation
/// vector is loaded once and used against both rows, and there are twice as
/// many independent accumulator chains. Measured +14% to +28% over the
/// one-row form across every shape and thread count tried.
///
/// This is worth contrasting with the *decode* case, where blocking over
/// weight rows was measured to be a clear loss (see `doc/PERFORMANCE.md`).
/// The difference is what gets reused: in GEMV there is one activation vector
/// and N rows, so extra rows only multiply the weight streams; in GEMM the
/// activations are reused across rows and the kernel is issue-bound rather
/// than stream-bound, so the extra ILP pays.
///
/// Both rows must have been unpacked from the same `ggml_type` — they share
/// the `per32`/`has_min` layout decisions.
pub fn dot_unpacked_pair(
    w0: &UnpackedRow,
    w1: &UnpackedRow,
    acts: &[ActQ8],
    out0: &mut [f32],
    out1: &mut [f32],
) {
    debug_assert_eq!(acts.len(), out0.len());
    debug_assert_eq!(acts.len(), out1.len());
    debug_assert_eq!(w0.per32, w1.per32);
    debug_assert_eq!(w0.has_min, w1.has_min);
    #[cfg(target_arch = "aarch64")]
    if have_dotprod() {
        return dot_unpacked_pair_impl::<ISA_DOTPROD>(w0, w1, acts, out0, out1);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if have_vnni() {
            // Safety: see `dot_row`.
            return unsafe { dot_unpacked_pair_vnni(w0, w1, acts, out0, out1) };
        }
        if is_x86_feature_detected!("avx2") {
            // Safety: guarded by the runtime feature check above.
            return unsafe { dot_unpacked_pair_avx2(w0, w1, acts, out0, out1) };
        }
    }
    dot_unpacked_pair_impl::<ISA_BASELINE>(w0, w1, acts, out0, out1)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512vnni,avx512vl,avx2,ssse3")]
unsafe fn dot_unpacked_pair_vnni(
    w0: &UnpackedRow,
    w1: &UnpackedRow,
    acts: &[ActQ8],
    out0: &mut [f32],
    out1: &mut [f32],
) {
    dot_unpacked_pair_impl::<ISA_VNNI>(w0, w1, acts, out0, out1)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_unpacked_pair_avx2(
    w0: &UnpackedRow,
    w1: &UnpackedRow,
    acts: &[ActQ8],
    out0: &mut [f32],
    out1: &mut [f32],
) {
    dot_unpacked_pair_impl::<ISA_AVX2>(w0, w1, acts, out0, out1)
}

/// Arithmetic here is parenthesised to match [`dot_unpacked_multi`] term for
/// term, so the two produce bit-identical results and the tests can compare
/// them for exact equality.
fn dot_unpacked_pair_impl<const ISA: u8>(
    w0: &UnpackedRow,
    w1: &UnpackedRow,
    acts: &[ActQ8],
    out0: &mut [f32],
    out1: &mut [f32],
) {
    let n = acts.len();
    let mut t0 = 0;
    while t0 + TOKEN_TILE <= n {
        let mut a0 = [0f32; TOKEN_TILE];
        let mut a1 = [0f32; TOKEN_TILE];
        if w0.per32 {
            for b in 0..w0.scale.len() {
                let (s0, s1) = (w0.scale[b], w1.scale[b]);
                let (q0, q1) = (&w0.q[b * 32..], &w1.q[b * 32..]);
                if w0.has_min {
                    let (m0, m1) = (w0.min[b], w1.min[b]);
                    for k in 0..TOKEN_TILE {
                        let a = &acts[t0 + k];
                        let xq = &a.q[b * 32..];
                        let (d, bs) = (a.d[b], block_sum(a, b) as f32);
                        a0[k] += d * (s0 * dot32::<ISA>(q0, xq) as f32 - m0 * bs);
                        a1[k] += d * (s1 * dot32::<ISA>(q1, xq) as f32 - m1 * bs);
                    }
                } else {
                    for k in 0..TOKEN_TILE {
                        let a = &acts[t0 + k];
                        let xq = &a.q[b * 32..];
                        let d = a.d[b];
                        a0[k] += d * s0 * dot32::<ISA>(q0, xq) as f32;
                        a1[k] += d * s1 * dot32::<ISA>(q1, xq) as f32;
                    }
                }
            }
        } else {
            for g in 0..w0.scale.len() {
                let (s0, s1) = (w0.scale[g], w1.scale[g]);
                let (q0, q1) = (&w0.q[g * GROUP..], &w1.q[g * GROUP..]);
                let b = g * GROUP / ACT_BLOCK;
                for k in 0..TOKEN_TILE {
                    let a = &acts[t0 + k];
                    let xq = &a.q[g * GROUP..];
                    let d = a.d[b];
                    a0[k] += d * (s0 * dot16::<ISA>(q0, xq) as f32);
                    a1[k] += d * (s1 * dot16::<ISA>(q1, xq) as f32);
                }
            }
        }
        out0[t0..t0 + TOKEN_TILE].copy_from_slice(&a0);
        out1[t0..t0 + TOKEN_TILE].copy_from_slice(&a1);
        t0 += TOKEN_TILE;
    }
    for t in t0..n {
        out0[t] = dot_unpacked_impl::<ISA>(w0, &acts[t]);
        out1[t] = dot_unpacked_impl::<ISA>(w1, &acts[t]);
    }
}

/// Tokens accumulated per pass over the weight row. Four keeps the
/// accumulators and the tile's activation slices comfortably within L1
/// alongside the weight row.
pub const TOKEN_TILE: usize = 4;

fn dot_unpacked_multi_impl<const ISA: u8>(w: &UnpackedRow, acts: &[ActQ8], out: &mut [f32]) {
    // The per-GROUP branch below omits the min correction: `Q6_K` is the only
    // type with per-16 scales and it is symmetric. A future asymmetric per-16
    // type would need that branch extended, so fail loudly rather than
    // silently drop the term.
    debug_assert!(
        w.per32 || !w.has_min,
        "a per-GROUP type with a min correction needs the other branch"
    );
    let n = acts.len();
    let mut t0 = 0;
    while t0 + TOKEN_TILE <= n {
        let mut acc = [0f32; TOKEN_TILE];
        if w.per32 {
            for b in 0..w.scale.len() {
                let wq = &w.q[b * 32..];
                let sc = w.scale[b];
                if w.has_min {
                    let mn = w.min[b];
                    for (k, slot) in acc.iter_mut().enumerate() {
                        let a = &acts[t0 + k];
                        let isum = dot32::<ISA>(wq, &a.q[b * 32..]);
                        *slot += a.d[b] * (sc * isum as f32 - mn * block_sum(a, b) as f32);
                    }
                } else {
                    for (k, slot) in acc.iter_mut().enumerate() {
                        let a = &acts[t0 + k];
                        *slot += a.d[b] * sc * dot32::<ISA>(wq, &a.q[b * 32..]) as f32;
                    }
                }
            }
        } else {
            for g in 0..w.scale.len() {
                let wq = &w.q[g * GROUP..];
                let sc = w.scale[g];
                let b = g * GROUP / ACT_BLOCK;
                for (k, slot) in acc.iter_mut().enumerate() {
                    let a = &acts[t0 + k];
                    // Parenthesised to match `dot_unpacked_impl`'s association
                    // exactly: `d * (sc * isum)`, not `(d * sc) * isum`. Same
                    // maths, different rounding — the tests compare the two
                    // paths for bit equality, which is what caught this.
                    let isum = dot16::<ISA>(wq, &a.q[g * GROUP..]);
                    *slot += a.d[b] * (sc * isum as f32);
                }
            }
        }
        out[t0..t0 + TOKEN_TILE].copy_from_slice(&acc);
        t0 += TOKEN_TILE;
    }
    // Tail: fewer than a full tile left.
    for t in t0..n {
        out[t] = dot_unpacked_impl::<ISA>(w, &acts[t]);
    }
}

// =====================================================================
// Single-token (GEMV) path: fuse unpack and dot, per block.
//
// The `unpack_row` + `dot_unpacked` pair above wins when several tokens
// amortize one unpack. With a single token it *loses*: materializing a whole
// row to memory and reading it straight back costs more than the unpack it
// saves, and the per-`GROUP` dot loop pays twice the `f32` scalar work of a
// per-block one. Measured on the reference Pi 4, decode went 2.05 -> 1.46
// tok/s when the row-at-a-time path was used for `n_tokens == 1`.
//
// So decode keeps these fused kernels, which never spill an unpacked row to
// memory — the 32 weights live in a stack array that stays L1/register
// resident for the one dot that consumes them. `engine::backend::cpu`
// picks between the two on `n_tokens`, the same GEMV/GEMM split any BLAS
// makes.
// =====================================================================

/// `sum_i weight_row[i] * act[i]` for a single token, straight from the
/// quantized bytes. `ggml_type`/`in_dim` must have passed [`supports`].
pub fn dot_row(ggml_type: u32, row: &[u8], act: &ActQ8) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if have_dotprod() {
        return dot_row_impl::<ISA_DOTPROD>(ggml_type, row, act);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if have_vnni() {
            // Safety: `have_vnni` verified both the feature bits and the
            // kernel's output against AVX2.
            return unsafe { dot_row_vnni(ggml_type, row, act) };
        }
        if is_x86_feature_detected!("avx2") {
            // Safety: guarded by the runtime feature check above.
            return unsafe { dot_row_avx2(ggml_type, row, act) };
        }
    }
    dot_row_impl::<ISA_BASELINE>(ggml_type, row, act)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512vnni,avx512vl,avx2,ssse3")]
unsafe fn dot_row_vnni(ggml_type: u32, row: &[u8], act: &ActQ8) -> f32 {
    dot_row_impl::<ISA_VNNI>(ggml_type, row, act)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_row_avx2(ggml_type: u32, row: &[u8], act: &ActQ8) -> f32 {
    dot_row_impl::<ISA_AVX2>(ggml_type, row, act)
}

fn dot_row_impl<const ISA: u8>(ggml_type: u32, row: &[u8], act: &ActQ8) -> f32 {
    match ggml_type {
        GGML_TYPE_Q8_0 => dot_q8_0::<ISA>(row, act),
        GGML_TYPE_Q5_0 => dot_q5_0::<ISA>(row, act),
        GGML_TYPE_IQ4_NL => dot_iq4_nl::<ISA>(row, act),
        GGML_TYPE_Q4_K => dot_q4_k::<ISA>(row, act),
        GGML_TYPE_Q6_K => dot_q6_k::<ISA>(row, act),
        other => panic!("vecdot::dot_row called for unsupported ggml_type {other}"),
    }
}

/// Activation sum over the 32-element block `b`, from the per-[`GROUP`] sums.
#[inline(always)]
fn block_sum(act: &ActQ8, b: usize) -> i32 {
    act.sums[b * 2] + act.sums[b * 2 + 1]
}

/// `block_q8_0`: `{ d: f16, qs: [i8; 32] }` — already `int8`, no unpack at all.
fn dot_q8_0<const ISA: u8>(row: &[u8], act: &ActQ8) -> f32 {
    const BLOCK_BYTES: usize = 2 + 32;
    let mut total = 0f32;
    for (b, block) in row.chunks_exact(BLOCK_BYTES).enumerate() {
        let dw = read_f16(block, 0);
        let w: &[i8] = bytemuck::cast_slice(&block[2..]);
        total += dw * act.d[b] * dot32::<ISA>(w, &act.q[b * 32..]) as f32;
    }
    total
}

/// `block_iq4_nl`, 32 elements: low nibbles then high nibbles, each a
/// codebook index rather than a signed integer.
fn dot_iq4_nl<const ISA: u8>(row: &[u8], act: &ActQ8) -> f32 {
    const BLOCK_BYTES: usize = 2 + 16;
    let mut total = 0f32;
    let mut w = [0i8; 32];
    for (b, block) in row.chunks_exact(BLOCK_BYTES).enumerate() {
        let dw = read_f16(block, 0);
        unpack_block_iq4_nl(&block[2..], &mut w);
        total += dw * act.d[b] * dot32::<ISA>(&w, &act.q[b * 32..]) as f32;
    }
    total
}

/// `block_q5_0`, 32 elements: low nibbles then high nibbles, each gaining a
/// 5th bit from `qh`, biased by -16 into a signed `int8`.
fn dot_q5_0<const ISA: u8>(row: &[u8], act: &ActQ8) -> f32 {
    const BLOCK_BYTES: usize = 2 + 4 + 16;
    let mut total = 0f32;
    let mut w = [0i8; 32];
    for (b, block) in row.chunks_exact(BLOCK_BYTES).enumerate() {
        let dw = read_f16(block, 0);
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        unpack_block_q5_0(qh, &block[6..], &mut w);
        total += dw * act.d[b] * dot32::<ISA>(&w, &act.q[b * 32..]) as f32;
    }
    total
}

/// `block_q4_K`, 256 elements as eight 32-element sub-blocks. Asymmetric:
/// `value = d*sc*q - dmin*m`, so the min term is applied against the block's
/// activation sum rather than folded into the weight.
fn dot_q4_k<const ISA: u8>(row: &[u8], act: &ActQ8) -> f32 {
    const BLOCK_BYTES: usize = 2 + 2 + 12 + 128;
    let mut total = 0f32;
    let mut sb = 0usize;
    for block in row.chunks_exact(BLOCK_BYTES) {
        let d = read_f16(block, 0);
        let dmin = read_f16(block, 2);
        let scales = &block[4..16];
        let qs = &block[16..];
        for g in 0..4 {
            let bytes = &qs[g * 32..g * 32 + 32];
            for (half, shift) in [(0usize, 0u32), (1, 4)] {
                let (sc, m) = get_scale_min_k4(g * 2 + half, scales);
                let b = sb + half;
                let mut w = [0i8; 32];
                for (j, &byte) in bytes.iter().enumerate() {
                    w[j] = ((byte >> shift) & 0x0F) as i8;
                }
                let isum = dot32::<ISA>(&w, &act.q[b * 32..]);
                total += act.d[b]
                    * (d * sc as f32 * isum as f32 - dmin * m as f32 * block_sum(act, b) as f32);
            }
            sb += 2;
        }
    }
    total
}

/// `block_q6_K`, 256 elements, one scale per 16 — so each 32-element run needs
/// two separate 16-lane dots. `q - 32` folds into the signed `int8` weight.
fn dot_q6_k<const ISA: u8>(row: &[u8], act: &ActQ8) -> f32 {
    const BLOCK_BYTES: usize = 128 + 64 + 16 + 2;
    let mut total = 0f32;
    let mut blk = 0usize;
    let mut w = [0i8; 32];
    for block in row.chunks_exact(BLOCK_BYTES) {
        let ql = &block[0..128];
        let qh = &block[128..192];
        let sc = &block[192..208];
        let d = read_f16(block, 208);
        for h in 0..2 {
            let qh_run = &qh[h * 32..h * 32 + 32];
            for &(ql_add, hshift, high) in Q6K_RUNS.iter() {
                let ql_run = &ql[h * 64 + ql_add..h * 64 + ql_add + 32];
                unpack_q6k_run(ql_run, qh_run, hshift, high, &mut w);
                let qx = &act.q[blk * 32..];
                let s0 = sc[(2 * blk) % 16] as i8 as f32;
                let s1 = sc[(2 * blk + 1) % 16] as i8 as f32;
                total += act.d[blk]
                    * d
                    * (s0 * dot16::<ISA>(&w, qx) as f32
                        + s1 * dot16::<ISA>(&w[16..], &qx[16..]) as f32);
                blk += 1;
            }
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::quant;

    /// Deterministic pseudo-random bytes — a fixed LCG rather than a real
    /// RNG so a failure is reproducible.
    fn pseudo_bytes(n: usize, seed: u32) -> Vec<u8> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 16) as u8
            })
            .collect()
    }

    fn activations(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i * 37 % 23) as f32 - 11.0) * 0.031)
            .collect()
    }

    /// Every fused kernel must agree with `dequantize` + an exact `f32` dot
    /// to within the error `int8` activation quantization can introduce.
    /// The bound is relative to the *magnitude of the terms* rather than to
    /// the (cancellation-prone) result, which is the meaningful measure for
    /// a dot product of mixed-sign values.
    fn check(ggml_type: u32, in_dim: usize, seed: u32) {
        let (block_bytes, block_elems) = match ggml_type {
            GGML_TYPE_Q8_0 => (34, 32),
            GGML_TYPE_Q5_0 => (22, 32),
            GGML_TYPE_IQ4_NL => (18, 32),
            GGML_TYPE_Q4_K => (144, 256),
            GGML_TYPE_Q6_K => (210, 256),
            other => panic!("unhandled {other}"),
        };
        let row_bytes = in_dim / block_elems * block_bytes;
        let mut row = pseudo_bytes(row_bytes, seed);
        // Random bytes in an `f16` scale field decode to Inf/NaN often
        // enough to make this test meaningless, so pin every block's scale
        // to a representative finite value (0.125, and 0.0625 for `dmin`)
        // and leave the quant payload random.
        for block in row.chunks_exact_mut(block_bytes) {
            match ggml_type {
                GGML_TYPE_Q8_0 | GGML_TYPE_Q5_0 | GGML_TYPE_IQ4_NL => {
                    block[0..2].copy_from_slice(&[0x00, 0x30])
                }
                GGML_TYPE_Q4_K => {
                    block[0..2].copy_from_slice(&[0x00, 0x30]);
                    block[2..4].copy_from_slice(&[0x00, 0x2c]);
                }
                GGML_TYPE_Q6_K => block[208..210].copy_from_slice(&[0x00, 0x30]),
                other => panic!("unhandled {other}"),
            }
        }
        let x = activations(in_dim);

        let reference: f32 = quant::dequantize(ggml_type, &row, in_dim)
            .unwrap()
            .iter()
            .zip(&x)
            .map(|(w, v)| w * v)
            .sum();
        let scale: f32 = quant::dequantize(ggml_type, &row, in_dim)
            .unwrap()
            .iter()
            .zip(&x)
            .map(|(w, v)| (w * v).abs())
            .sum();

        assert!(
            supports(ggml_type, in_dim),
            "type {ggml_type} in_dim {in_dim}"
        );
        let act = quantize_act(&x);

        // Both production paths must land within the int8-activation error
        // budget: `dot_row` (decode/GEMV) and unpack+dot (prefill/GEMM).
        let via_row = dot_row(ggml_type, &row, &act);
        let via_unpack = dot_via_unpack(ggml_type, &row, &act, in_dim);
        for (label, got) in [("dot_row", via_row), ("unpack", via_unpack)] {
            let err = (got - reference).abs();
            assert!(
                err <= 0.01 * scale.max(1e-6),
                "{label}: type {ggml_type} in_dim {in_dim}: got {got}, want {reference} \
                 (err {err}, term-magnitude {scale})"
            );
        }

        // And they must agree with *each other* far more tightly than with the
        // reference — they quantize activations identically, so the only
        // difference is `f32` summation order. A real divergence here means one
        // of the two unpack implementations has drifted from the other.
        assert!(
            (via_row - via_unpack).abs() <= 1e-3 * scale.max(1e-6),
            "type {ggml_type} in_dim {in_dim}: dot_row {via_row} vs unpack {via_unpack}"
        );
    }

    /// The tiled multi-token path must agree with calling the single-token
    /// path once per token. Both quantize identically and sum in the same
    /// order per token, so this is an exact-equality check, not a tolerance —
    /// any difference means the tiling got an index wrong.
    #[test]
    fn tiled_multi_token_matches_one_call_per_token() {
        let in_dim = 896;
        // Deliberately not a multiple of TOKEN_TILE, to exercise the tail.
        for n_tokens in [1usize, 3, 4, 7, 8, 17] {
            for ggml_type in [GGML_TYPE_Q8_0, GGML_TYPE_Q5_0, GGML_TYPE_IQ4_NL] {
                let block_bytes = match ggml_type {
                    GGML_TYPE_Q8_0 => 34,
                    GGML_TYPE_Q5_0 => 22,
                    _ => 18,
                };
                let mut row = pseudo_bytes(in_dim / 32 * block_bytes, 5);
                for block in row.chunks_exact_mut(block_bytes) {
                    block[0..2].copy_from_slice(&[0x00, 0x30]);
                }
                let acts: Vec<ActQ8> = (0..n_tokens)
                    .map(|t| {
                        let x: Vec<f32> = (0..in_dim)
                            .map(|i| (((i * 37 + t * 11) % 23) as f32 - 11.0) * 0.031)
                            .collect();
                        quantize_act(&x)
                    })
                    .collect();
                let mut w = UnpackedRow::new();
                unpack_row(ggml_type, &row, in_dim, &mut w);

                let want: Vec<f32> = acts.iter().map(|a| dot_unpacked(&w, a)).collect();
                let mut got = vec![0f32; n_tokens];
                dot_unpacked_multi(&w, &acts, &mut got);
                assert_eq!(got, want, "type {ggml_type}, {n_tokens} tokens");
            }
        }
    }

    /// A buffer reused across rows must not leak state from the previous row.
    #[test]
    fn unpacked_row_buffer_is_safe_to_reuse() {
        let in_dim = 4864;
        let act = quantize_act(&activations(in_dim));
        let mut scratch = UnpackedRow::new();
        for ggml_type in [
            GGML_TYPE_Q8_0,
            GGML_TYPE_Q5_0,
            GGML_TYPE_IQ4_NL,
            GGML_TYPE_Q4_K,
            GGML_TYPE_Q6_K,
        ] {
            let (block_bytes, block_elems) = match ggml_type {
                GGML_TYPE_Q8_0 => (34, 32),
                GGML_TYPE_Q5_0 => (22, 32),
                GGML_TYPE_IQ4_NL => (18, 32),
                GGML_TYPE_Q4_K => (144, 256),
                _ => (210, 256),
            };
            let mut rows = Vec::new();
            for seed in [3u32, 11] {
                let mut row = pseudo_bytes(in_dim / block_elems * block_bytes, seed);
                for block in row.chunks_exact_mut(block_bytes) {
                    match ggml_type {
                        GGML_TYPE_Q8_0 | GGML_TYPE_Q5_0 | GGML_TYPE_IQ4_NL => {
                            block[0..2].copy_from_slice(&[0x00, 0x30])
                        }
                        GGML_TYPE_Q4_K => {
                            block[0..2].copy_from_slice(&[0x00, 0x30]);
                            block[2..4].copy_from_slice(&[0x00, 0x2c]);
                        }
                        _ => block[208..210].copy_from_slice(&[0x00, 0x30]),
                    }
                }
                rows.push(row);
            }
            // Fresh buffer for row 1.
            let mut fresh = UnpackedRow::new();
            unpack_row(ggml_type, &rows[1], in_dim, &mut fresh);
            let want = dot_unpacked(&fresh, &act);
            // Same row through a buffer that just held a *different* row, and
            // a different type before that.
            unpack_row(ggml_type, &rows[0], in_dim, &mut scratch);
            unpack_row(ggml_type, &rows[1], in_dim, &mut scratch);
            let got = dot_unpacked(&scratch, &act);
            assert_eq!(got, want, "stale state for ggml_type {ggml_type}");
        }
    }

    #[test]
    fn q8_0_matches_dequantize_reference() {
        for seed in [1, 7, 99] {
            check(GGML_TYPE_Q8_0, 896, seed);
            check(GGML_TYPE_Q8_0, 4864, seed);
        }
    }

    #[test]
    fn q5_0_matches_dequantize_reference() {
        for seed in [1, 7, 99] {
            check(GGML_TYPE_Q5_0, 896, seed);
            check(GGML_TYPE_Q5_0, 4864, seed);
        }
    }

    /// 896 and 4864 are `Qwen2.5-Coder-0.5B`'s own two row widths, and 896
    /// is the one that matters: it is not a multiple of 256, which is why
    /// its rows carry `IQ4_NL` rather than the `Q2_K`/`Q3_K` the file name
    /// advertises.
    #[test]
    fn iq4_nl_matches_dequantize_reference() {
        for seed in [1, 7, 99] {
            check(GGML_TYPE_IQ4_NL, 896, seed);
            check(GGML_TYPE_IQ4_NL, 4864, seed);
        }
    }

    #[test]
    fn q4_k_matches_dequantize_reference() {
        for seed in [1, 7, 99] {
            check(GGML_TYPE_Q4_K, 256, seed);
            check(GGML_TYPE_Q4_K, 4864, seed);
        }
    }

    #[test]
    fn q6_k_matches_dequantize_reference() {
        for seed in [1, 7, 99] {
            check(GGML_TYPE_Q6_K, 256, seed);
            check(GGML_TYPE_Q6_K, 4864, seed);
        }
    }

    #[test]
    fn supports_rejects_rows_that_are_not_a_whole_number_of_blocks() {
        // A `Qwen2.5-0.5B`'s 896-wide rows are 28 blocks of 32 but not a
        // multiple of 256 — the exact case that must not take a K-quant
        // kernel.
        assert!(supports(GGML_TYPE_Q5_0, 896));
        assert!(supports(GGML_TYPE_Q8_0, 896));
        assert!(supports(GGML_TYPE_IQ4_NL, 896));
        assert!(!supports(GGML_TYPE_IQ4_NL, 24));
        assert!(!supports(GGML_TYPE_Q4_K, 896));
        assert!(!supports(GGML_TYPE_Q6_K, 896));
        // And types with no fused kernel stay on the old path regardless.
        assert!(!supports(quant::GGML_TYPE_F32, 1024));
        assert!(!supports(quant::GGML_TYPE_Q5_K, 1024));
    }

    #[test]
    fn quantize_act_handles_an_all_zero_block_without_nan() {
        let a = quantize_act(&vec![0f32; 64]);
        assert!(a.d.iter().all(|d| *d == 0.0));
        assert!(a.q.iter().all(|q| *q == 0));
        assert!(a.sums.iter().all(|s| *s == 0));
    }
}
