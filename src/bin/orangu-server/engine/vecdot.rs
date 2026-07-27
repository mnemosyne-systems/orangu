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
    GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS, GGML_TYPE_Q4_K, GGML_TYPE_Q5_0, GGML_TYPE_Q6_K,
    GGML_TYPE_Q8_0, get_scale_min_k4, read_f16,
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
        GGML_TYPE_Q4_K | GGML_TYPE_Q6_K | GGML_TYPE_IQ4_XS => in_dim.is_multiple_of(256),
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
        GGML_TYPE_IQ4_XS => unpack_iq4_xs(row, out),
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
    #[cfg(target_arch = "aarch64")]
    {
        // Safety: NEON (`asimd`) is architecturally mandatory on aarch64, so
        // this needs no runtime check — unlike `dotprod`/`i8mm`.
        return unsafe { unpack_block_iq4_nl_neon(qs, w) };
    }
    #[cfg(target_arch = "x86_64")]
    {
        // Checked per block rather than per row. `is_x86_feature_detected!`
        // compiles to a relaxed load of a cached bitmask plus a
        // perfectly-predicted branch, which is far below the ~60 scalar ops
        // it saves.
        if is_x86_feature_detected!("ssse3") {
            // Safety: guarded by the runtime feature check above.
            return unsafe { unpack_block_iq4_nl_ssse3(qs, w) };
        }
    }
    #[allow(unreachable_code)]
    unpack_block_iq4_nl_scalar(qs, w)
}

/// The reference form, and the fallback where no byte-shuffle exists.
#[inline(always)]
fn unpack_block_iq4_nl_scalar(qs: &[u8], w: &mut [i8; 32]) {
    for j in 0..16 {
        w[j] = KVALUES_IQ4NL[(qs[j] & 0x0F) as usize];
        w[j + 16] = KVALUES_IQ4NL[(qs[j] >> 4) as usize];
    }
}

/// `vqtbl1q_s8` is a 16-entry byte table lookup across a whole vector —
/// exactly the shape of [`KVALUES_IQ4NL`], which is 16 `i8` by construction.
///
/// This replaces ~64 scalar operations with 8 instructions, and it is not a
/// micro-optimization: a decode profile of `SmolLM2-360M-IQ4_XS` put **79.3%**
/// of all time in the scalar form (52.1% in `ld1 {v.b}[n]` lane inserts,
/// 27.2% in the `ldrb` table reads) against **0.4%** in the actual
/// multiply-accumulate. The kernel was not computing, it was gathering bytes.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn unpack_block_iq4_nl_neon(qs: &[u8], w: &mut [i8; 32]) {
    use std::arch::aarch64::*;
    unsafe {
        let table = vld1q_s8(KVALUES_IQ4NL.as_ptr());
        let packed = vld1q_u8(qs.as_ptr());
        // Low nibbles land in elements 0..16, high nibbles in 16..32 — the
        // layout both `IQ4_NL` and `IQ4_XS` blocks use.
        let lo = vandq_u8(packed, vdupq_n_u8(0x0F));
        let hi = vshrq_n_u8::<4>(packed);
        vst1q_s8(w.as_mut_ptr(), vqtbl1q_s8(table, lo));
        vst1q_s8(w.as_mut_ptr().add(16), vqtbl1q_s8(table, hi));
    }
}

/// `pshufb` is x86's byte-table shuffle and behaves like `vqtbl1q_s8` for
/// indices below 16 (it zeroes a lane only when the index's high bit is set,
/// which a 4-bit nibble never has). Same 8-instruction shape as the NEON
/// path above.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
unsafe fn unpack_block_iq4_nl_ssse3(qs: &[u8], w: &mut [i8; 32]) {
    use std::arch::x86_64::*;
    unsafe {
        let table = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i);
        let packed = _mm_loadu_si128(qs.as_ptr() as *const __m128i);
        let lo = _mm_and_si128(packed, _mm_set1_epi8(0x0F));
        // No 8-bit shift on x86: shift 16-bit lanes and mask the borrowed bits.
        let hi = _mm_and_si128(_mm_srli_epi16::<4>(packed), _mm_set1_epi8(0x0F));
        _mm_storeu_si128(w.as_mut_ptr() as *mut __m128i, _mm_shuffle_epi8(table, lo));
        _mm_storeu_si128(
            w.as_mut_ptr().add(16) as *mut __m128i,
            _mm_shuffle_epi8(table, hi),
        );
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
/// `IQ4_XS` into the generic [`UnpackedRow`] form — per-32 `f32` scales,
/// no min.
///
/// Not reachable through [`supports`] today: `IQ4_XS` requires `256 | in_dim`
/// there, which is exactly [`supports_k`]'s condition, so prefill always
/// routes to [`dot_k_pair`] instead. It exists because `unpack_row`'s
/// fallthrough is a `panic!`, and a gating change that silently turned a
/// missing arm into a crash would be a poor trade for the dozen lines this
/// costs.
fn unpack_iq4_xs(row: &[u8], out: &mut UnpackedRow) {
    const BLOCK_BYTES: usize = 2 + 2 + SUPER_BLOCK / 64 + SUPER_BLOCK / 2;
    out.has_min = false;
    for (s, block) in row.chunks_exact(BLOCK_BYTES).enumerate() {
        let d = read_f16(block, 0);
        let scales_h = u16::from_le_bytes([block[2], block[3]]);
        let scales_l = &block[4..8];
        let qs = &block[8..BLOCK_BYTES];
        for ib in 0..SUBS {
            let b = s * SUBS + ib;
            let low = (scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F;
            let high = ((scales_h >> (2 * ib)) & 3) as u8;
            let ls = ((low | (high << 4)) as i32) - 32;
            let w: &mut [i8; 32] = (&mut out.q[b * 32..b * 32 + 32]).try_into().unwrap();
            unpack_block_iq4_nl(&qs[ib * 16..], w);
            out.scale[b] = d * ls as f32;
            out.min[b] = 0.0;
        }
    }
}

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
// K-quant prefill GEMM: `q8_K`-style activations, tile-interleaved.
//
// A second GEMM path used only for `Q4_K`/`Q6_K` with several tokens. Those
// two types are 100% of a `Llama-3.2-1B-Instruct-Q4_K_M` (58% `Q4_K`, 42%
// `Q6_K`) — `embedding_length` 2048 is a multiple of 256, so unlike
// `Qwen2.5-0.5B` nothing falls back to `Q5_0` — and on that model the generic
// path above reached only 61% of llama.cpp's prefill. A `perf annotate` of a
// real prefill put ~80% of all time in `dot_unpacked_pair_impl`'s
// `per32 && has_min` branch and showed three distinct problems, which this
// path fixes together:
//
// 1. **Pointer chasing.** `acts: &[ActQ8]` is a slice of structs of three
//    `Vec`s, so the inner loop reloads three data pointers plus their lengths
//    for every (block, token) — the 72-byte `ActQ8` stride appears literally
//    as `ldr x16, [x26], #72`, and one activation load alone carried 21.6% of
//    the profile. [`ActQ8K`] is instead one flat allocation with the tile's
//    four tokens interleaved per block, so the whole loop is one sequential
//    stream off a single base pointer.
// 2. **`f32` bookkeeping every 32 elements.** The generic path converts to
//    `f32` and does a multiply/subtract/FMA per 32-element sub-block. ggml
//    carries a separate `block_q8_K` activation format with one scale per 256
//    precisely so it can keep `scale * partial` in `i32` across a whole
//    super-block and touch `f32` once per 256 — see
//    `ggml_vec_dot_q4_K_q8_K`'s NEON branch, whose entire per-super-block
//    `f32` work is `sumf += d * (sumi1 + sumi2)`. This does the same.
// 3. **A slow scalar tail.** `n_tokens % TOKEN_TILE` tokens went through the
//    untiled `dot_unpacked_impl`; at a 79-token prompt that was 4.2% of
//    prefill. Here the token count is padded up to a whole tile with zero
//    activations, which contribute nothing, so every token is tiled.
//
// `Q4_K`'s asymmetric `-dmin*m` correction is computed in its own pass over
// the row rather than inside the dot loop. It reads only two small `i16`
// arrays that stay in L1 and touches neither the weight quants nor the
// activations, and hoisting it out frees four registers in the hot loop —
// worth +10% on `ffn_down` at 4 threads.
//
// Measured on the reference Pi 4 at 4 threads, 64 tokens, against the generic
// path at the real Llama-3.2-1B shapes: `wq`/`wo` +52%, `gate`/`up` +45%,
// `ffn_down` +61% (`Q6_K`) and +84% (`Q4_K`), `attn_v` +47%.
//
// `Q8_0`/`Q5_0` deliberately keep the generic path: they have no super-block
// to accumulate across, so only point 1 would apply to them, and the
// `Qwen2.5-0.5B` numbers that path was tuned against are already at 89% of
// llama.cpp.
// =====================================================================

/// Elements per K-quant super-block.
pub const SUPER_BLOCK: usize = 256;
/// 32-element sub-blocks per super-block.
const SUBS: usize = SUPER_BLOCK / 32;
/// Scale groups per `Q6_K` super-block (one per 16 elements).
const Q6K_GROUPS: usize = SUPER_BLOCK / GROUP;

/// Whether [`ActQ8K`] and [`dot_k_pair`] handle `ggml_type` at this `in_dim`.
///
/// Narrower than [`supports`] on purpose: this path exists for the two K-quant
/// types, which are the only ones with a 256-element super-block to accumulate
/// across.
pub fn supports_k(ggml_type: u32, in_dim: usize) -> bool {
    matches!(
        ggml_type,
        GGML_TYPE_Q4_K | GGML_TYPE_Q6_K | GGML_TYPE_IQ4_XS
    ) && in_dim.is_multiple_of(SUPER_BLOCK)
}

/// Activations quantized to `int8` with **one `f32` scale per 256 elements**
/// (ggml's `block_q8_K`) and laid out interleaved by token tile.
///
/// The coarser scale is what makes integer accumulation across a super-block
/// possible, and is the same tradeoff ggml makes for K-quant weights — it uses
/// `block_q8_K` for exactly these types and the finer `block_q8_0`/`q8_1` for
/// the rest. Accuracy stays inside the same budget [`ActQ8`] is held to; see
/// the tests.
pub struct ActQ8K {
    n_tokens: usize,
    /// `in_dim / 32`
    n_block: usize,
    /// `in_dim / 256`
    n_super: usize,
    /// `n_tokens.div_ceil(TOKEN_TILE)`
    n_tile: usize,
    /// `[tile][block][token][32]` — the tile's four tokens contiguous per
    /// block, so the dot loop walks one stream instead of four.
    q: Vec<i8>,
    /// `[tile][super][token]`
    d: Vec<f32>,
    /// `[tile][super][token][8]` — `sum(q)` per 32-element sub-block, for
    /// `Q4_K`'s min term. Grouped eight-at-a-time per (super-block, token) so
    /// the correction is one short contiguous dot in its own pass.
    bsum: Vec<i16>,
}

impl ActQ8K {
    /// `x` is `[n_tokens][in_dim]`, token-major. `in_dim` must be a multiple
    /// of [`SUPER_BLOCK`] — guaranteed by [`supports_k`].
    pub fn quantize(x: &[f32], in_dim: usize, n_tokens: usize) -> Self {
        debug_assert_eq!(in_dim % SUPER_BLOCK, 0);
        debug_assert_eq!(x.len(), n_tokens * in_dim);
        let n_block = in_dim / 32;
        let n_super = in_dim / SUPER_BLOCK;
        let n_tile = n_tokens.div_ceil(TOKEN_TILE);
        // The padding tokens of the last tile stay zero: a zero activation
        // block has a zero scale and contributes nothing, and their outputs
        // are never read.
        let mut q = vec![0i8; n_tile * n_block * TOKEN_TILE * 32];
        let mut d = vec![0f32; n_tile * n_super * TOKEN_TILE];
        let mut bsum = vec![0i16; n_tile * n_super * TOKEN_TILE * SUBS];
        for tl in 0..n_tile {
            for k in 0..TOKEN_TILE {
                let t = tl * TOKEN_TILE + k;
                if t >= n_tokens {
                    break;
                }
                let row = &x[t * in_dim..(t + 1) * in_dim];
                for s in 0..n_super {
                    let chunk = &row[s * SUPER_BLOCK..(s + 1) * SUPER_BLOCK];
                    let amax = chunk.iter().fold(0f32, |m, v| m.max(v.abs()));
                    let scale = amax / 127.0;
                    let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
                    d[(tl * n_super + s) * TOKEN_TILE + k] = scale;
                    for j in 0..SUBS {
                        let b = s * SUBS + j;
                        let src = &chunk[j * 32..(j + 1) * 32];
                        let dst = &mut q[((tl * n_block + b) * TOKEN_TILE + k) * 32..][..32];
                        let mut sum = 0i32;
                        for (slot, &v) in dst.iter_mut().zip(src) {
                            // `round` then clamp, as in `quantize_act`: the
                            // extreme is exactly ±127, never -128.
                            let qi = (v * inv).round().clamp(-127.0, 127.0) as i8;
                            *slot = qi;
                            sum += qi as i32;
                        }
                        // |sum| <= 32 * 127 = 4064, so `i16` is ample.
                        bsum[((tl * n_super + s) * TOKEN_TILE + k) * SUBS + j] = sum as i16;
                    }
                }
            }
        }
        Self {
            n_tokens,
            n_block,
            n_super,
            n_tile,
            q,
            d,
            bsum,
        }
    }
}

/// One unpacked K-quant weight row, with the scales left as **integers**.
///
/// That is the difference from [`UnpackedRow`], and the whole point: only the
/// per-super-block `d`/`dmin` are `f32`, so the dot loop can fold each
/// sub-block's scale into an `i32` running sum and convert once per 256
/// elements instead of once per 32.
pub struct KRow {
    /// `int8` weights. `Q4_K` keeps its unsigned `0..=15`; `Q6_K` folds its
    /// `-32` bias in, giving `-32..=31`.
    q: Vec<i8>,
    /// Integer scale — per 32 elements for `Q4_K`, per [`GROUP`] for `Q6_K`.
    sc: Vec<i16>,
    /// Integer min per 32 elements. `Q4_K` only; empty otherwise.
    mins: Vec<i16>,
    /// Per-super-block `f32` scale.
    d: Vec<f32>,
    /// Per-super-block `f32` min scale. `Q4_K` only.
    dmin: Vec<f32>,
    /// Which loop shape this row needs. These differ structurally, not just
    /// by constant, so this selects a kernel rather than parameterizing one.
    kind: KKind,
}

/// The three K-quant shapes [`KRow`] can hold.
///
/// Note what the discriminating properties actually are — scale granularity
/// and symmetry — rather than the type names: `Q4_K` and `IQ4_XS` share a
/// loop and differ only in whether the min pass runs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum KKind {
    /// Per-32 integer scales *and* a per-32 integer min. The only asymmetric
    /// K-quant, so the only one that needs the correction pass.
    Q4K,
    /// Per-[`GROUP`] (16) signed scales, symmetric.
    Q6K,
    /// Per-32 signed scales, symmetric — `Q4_K`'s loop with the min pass
    /// skipped, which is exactly how it is implemented.
    Iq4Xs,
}

impl KRow {
    pub fn new() -> Self {
        Self {
            q: Vec::new(),
            sc: Vec::new(),
            mins: Vec::new(),
            d: Vec::new(),
            dmin: Vec::new(),
            // Overwritten by `resize_for` before any read; the buffers are
            // empty until then.
            kind: KKind::Q4K,
        }
    }

    fn resize_for(&mut self, in_dim: usize, kind: KKind) {
        let n_super = in_dim / SUPER_BLOCK;
        // Only `Q6_K` carries scales per 16; the other two are per 32. Only
        // `Q4_K` carries mins at all.
        let n_sc = if kind == KKind::Q6K {
            in_dim / GROUP
        } else {
            in_dim / 32
        };
        let q4k = kind == KKind::Q4K;
        let n_min = if q4k { in_dim / 32 } else { 0 };
        // Resize *and* truncate both ways, so a buffer reused across tensors
        // of different types or widths cannot keep a stale tail.
        self.q.resize(in_dim, 0);
        self.q.truncate(in_dim);
        self.sc.resize(n_sc, 0);
        self.sc.truncate(n_sc);
        self.mins.resize(n_min, 0);
        self.mins.truncate(n_min);
        self.d.resize(n_super, 0.0);
        self.d.truncate(n_super);
        self.dmin.resize(if q4k { n_super } else { 0 }, 0.0);
        self.dmin.truncate(if q4k { n_super } else { 0 });
        self.kind = kind;
    }
}

impl Default for KRow {
    fn default() -> Self {
        Self::new()
    }
}

/// Unpacks one weight row into `out`. `ggml_type`/`in_dim` must have passed
/// [`supports_k`].
pub fn unpack_k_row(ggml_type: u32, row: &[u8], in_dim: usize, out: &mut KRow) {
    match ggml_type {
        GGML_TYPE_Q4_K => {
            out.resize_for(in_dim, KKind::Q4K);
            unpack_k_q4_k(row, out);
        }
        GGML_TYPE_Q6_K => {
            out.resize_for(in_dim, KKind::Q6K);
            unpack_k_q6_k(row, out);
        }
        GGML_TYPE_IQ4_XS => {
            out.resize_for(in_dim, KKind::Iq4Xs);
            unpack_k_iq4_xs(row, out);
        }
        other => panic!("vecdot::unpack_k_row called for unsupported ggml_type {other}"),
    }
}

/// Same bit layout as [`unpack_q4_k`], but `sc`/`m` stay integers and `d`/
/// `dmin` are kept once per super-block rather than multiplied in.
fn unpack_k_q4_k(row: &[u8], out: &mut KRow) {
    const BLOCK_BYTES: usize = 2 + 2 + 12 + 128;
    let mut sb = 0usize;
    for (s, block) in row.chunks_exact(BLOCK_BYTES).enumerate() {
        out.d[s] = read_f16(block, 0);
        out.dmin[s] = read_f16(block, 2);
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
                out.sc[sb + half] = sc as i16;
                out.mins[sb + half] = m as i16;
            }
            sb += 2;
        }
    }
}

/// Same bit layout as [`unpack_q6_k`], with the `i8` scales kept as integers.
fn unpack_k_q6_k(row: &[u8], out: &mut KRow) {
    const BLOCK_BYTES: usize = 128 + 64 + 16 + 2;
    for (s, block) in row.chunks_exact(BLOCK_BYTES).enumerate() {
        let ql = &block[0..128];
        let qh = &block[128..192];
        let sc = &block[192..208];
        out.d[s] = read_f16(block, 208);
        let base = s * SUPER_BLOCK;
        for h in 0..2 {
            let qh_run = &qh[h * 32..h * 32 + 32];
            for (run, &(ql_add, hshift, high)) in Q6K_RUNS.iter().enumerate() {
                let e0 = base + h * 128 + run * 32;
                let ql_run = &ql[h * 64 + ql_add..h * 64 + ql_add + 32];
                let w: &mut [i8; 32] = (&mut out.q[e0..e0 + 32]).try_into().unwrap();
                unpack_q6k_run(ql_run, qh_run, hshift, high, w);
                // This 32-element run spans two scale groups of 16. `scales`
                // is `int8_t` in ggml's struct — a negative scale must stay
                // negative, hence the `as i8` before widening.
                let gi = e0 / GROUP;
                out.sc[gi] = sc[gi % 16] as i8 as i16;
                out.sc[gi + 1] = sc[(gi + 1) % 16] as i8 as i16;
            }
        }
    }
}

/// Dots **two** unpacked K-quant rows against every token, sharing each
/// activation load between them — the prefill entry point for
/// `Q4_K`/`Q6_K`/`IQ4_XS`.
///
/// `out0`/`out1` are `n_tokens` long; the tile padding is dropped on the way
/// out.
pub fn dot_k_pair(w0: &KRow, w1: &KRow, a: &ActQ8K, out0: &mut [f32], out1: &mut [f32]) {
    debug_assert_eq!(w0.kind, w1.kind);
    debug_assert_eq!(out0.len(), a.n_tokens);
    debug_assert_eq!(out1.len(), a.n_tokens);
    #[cfg(target_arch = "aarch64")]
    if have_dotprod() {
        return dot_k_pair_impl::<ISA_DOTPROD>(w0, w1, a, out0, out1);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if have_vnni() {
            // Safety: `have_vnni` verified the features and the kernel's
            // output against AVX2.
            return unsafe { dot_k_pair_vnni(w0, w1, a, out0, out1) };
        }
        if is_x86_feature_detected!("avx2") {
            // Safety: guarded by the runtime feature check above.
            return unsafe { dot_k_pair_avx2(w0, w1, a, out0, out1) };
        }
    }
    dot_k_pair_impl::<ISA_BASELINE>(w0, w1, a, out0, out1)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512vnni,avx512vl,avx2,ssse3")]
unsafe fn dot_k_pair_vnni(w0: &KRow, w1: &KRow, a: &ActQ8K, out0: &mut [f32], out1: &mut [f32]) {
    dot_k_pair_impl::<ISA_VNNI>(w0, w1, a, out0, out1)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_k_pair_avx2(w0: &KRow, w1: &KRow, a: &ActQ8K, out0: &mut [f32], out1: &mut [f32]) {
    dot_k_pair_impl::<ISA_AVX2>(w0, w1, a, out0, out1)
}

fn dot_k_pair_impl<const ISA: u8>(
    w0: &KRow,
    w1: &KRow,
    a: &ActQ8K,
    out0: &mut [f32],
    out1: &mut [f32],
) {
    match w0.kind {
        KKind::Q4K => gemm_k_q4_k::<ISA>(w0, w1, a, out0, out1),
        KKind::Q6K => gemm_k_q6_k::<ISA>(w0, w1, a, out0, out1),
        KKind::Iq4Xs => gemm_k_iq4_xs::<ISA>(w0, w1, a, out0, out1),
    }
}

/// One row against every token — the trailing row when `out_dim` is odd.
///
/// Runs the pair kernel with the row supplied twice and the duplicate results
/// discarded. `out_dim` is even for every tensor in every model this has been
/// run against, so this is a correctness path, not a hot one; giving it its
/// own copy of the loop would double the kernel surface to save nothing
/// measurable.
pub fn dot_k_multi(w: &KRow, a: &ActQ8K, out: &mut [f32], scratch: &mut Vec<f32>) {
    scratch.clear();
    scratch.resize(out.len(), 0.0);
    dot_k_pair(w, w, a, out, scratch);
}

fn gemm_k_q4_k<const ISA: u8>(
    w0: &KRow,
    w1: &KRow,
    a: &ActQ8K,
    out0: &mut [f32],
    out1: &mut [f32],
) {
    for tl in 0..a.n_tile {
        let mut acc0 = [0f32; TOKEN_TILE];
        let mut acc1 = [0f32; TOKEN_TILE];
        let qtile = &a.q[tl * a.n_block * TOKEN_TILE * 32..];
        let dtile = &a.d[tl * a.n_super * TOKEN_TILE..];
        let btile = &a.bsum[tl * a.n_super * TOKEN_TILE * SUBS..];

        // Pass 1: the asymmetric `-dmin*m` correction, `sum_j m[j]*bsum[j]`
        // per (super-block, token). Reads ~1 KiB of `i16` and never touches
        // the weight quants or activations, so it stays in L1 and leaves the
        // dot loop below four registers freer.
        for s in 0..a.n_super {
            let ad = &dtile[s * TOKEN_TILE..];
            let m0 = &w0.mins[s * SUBS..];
            let m1 = &w1.mins[s * SUBS..];
            let (dm0, dm1) = (w0.dmin[s], w1.dmin[s]);
            for k in 0..TOKEN_TILE {
                let bs = &btile[(s * TOKEN_TILE + k) * SUBS..];
                let mut i0 = 0i32;
                let mut i1 = 0i32;
                for j in 0..SUBS {
                    let b = bs[j] as i32;
                    i0 += m0[j] as i32 * b;
                    i1 += m1[j] as i32 * b;
                }
                acc0[k] -= ad[k] * dm0 * i0 as f32;
                acc1[k] -= ad[k] * dm1 * i1 as f32;
            }
        }

        // Pass 2: the symmetric dot, shared with `IQ4_XS`.
        accumulate_sym32::<ISA>(w0, w1, a, qtile, dtile, &mut acc0, &mut acc1);
        store_tile(tl, &acc0, out0);
        store_tile(tl, &acc1, out1);
    }
}

/// The per-32 symmetric dot: `sum_s d * sum_j sc[j] * (q[j] . x[j])`, one
/// `f32` conversion per super-block.
///
/// Shared verbatim by [`gemm_k_q4_k`] (as its second pass, after the min
/// correction) and [`gemm_k_iq4_xs`] (which has no min term). Factored out
/// rather than copied so the two cannot drift, and so `Q4_K` — the
/// best-measured path here — keeps bit-identical arithmetic.
///
/// Integer range: `Q4_K`'s `sc` is `0..=63` against unsigned `0..=15` quants,
/// `IQ4_XS`'s is `-32..=31` against `KVALUES_IQ4NL`'s `-127..=113`. The larger
/// case is |sc| 32 * 32 lanes * 127 * 127 ~ 1.65e7 per sub-block, so eight of
/// them stay far inside `i32`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn accumulate_sym32<const ISA: u8>(
    w0: &KRow,
    w1: &KRow,
    a: &ActQ8K,
    qtile: &[i8],
    dtile: &[f32],
    acc0: &mut [f32; TOKEN_TILE],
    acc1: &mut [f32; TOKEN_TILE],
) {
    for s in 0..a.n_super {
        let mut isum0 = [0i32; TOKEN_TILE];
        let mut isum1 = [0i32; TOKEN_TILE];
        for j in 0..SUBS {
            let b = s * SUBS + j;
            let sc0 = w0.sc[b] as i32;
            let sc1 = w1.sc[b] as i32;
            let q0 = &w0.q[b * 32..];
            let q1 = &w1.q[b * 32..];
            let xq = &qtile[b * TOKEN_TILE * 32..];
            for k in 0..TOKEN_TILE {
                let x = &xq[k * 32..];
                isum0[k] += sc0 * dot32::<ISA>(q0, x);
                isum1[k] += sc1 * dot32::<ISA>(q1, x);
            }
        }
        let (d0, d1) = (w0.d[s], w1.d[s]);
        let ad = &dtile[s * TOKEN_TILE..];
        for k in 0..TOKEN_TILE {
            acc0[k] += ad[k] * (d0 * isum0[k] as f32);
            acc1[k] += ad[k] * (d1 * isum1[k] as f32);
        }
    }
}

/// `IQ4_XS` prefill: [`gemm_k_q4_k`] without the min pass. Its scales are
/// per-32 and signed, and its quants already carry the `KVALUES_IQ4NL` levels
/// as `int8`, so nothing is left to correct.
fn gemm_k_iq4_xs<const ISA: u8>(
    w0: &KRow,
    w1: &KRow,
    a: &ActQ8K,
    out0: &mut [f32],
    out1: &mut [f32],
) {
    for tl in 0..a.n_tile {
        let mut acc0 = [0f32; TOKEN_TILE];
        let mut acc1 = [0f32; TOKEN_TILE];
        let qtile = &a.q[tl * a.n_block * TOKEN_TILE * 32..];
        let dtile = &a.d[tl * a.n_super * TOKEN_TILE..];
        accumulate_sym32::<ISA>(w0, w1, a, qtile, dtile, &mut acc0, &mut acc1);
        store_tile(tl, &acc0, out0);
        store_tile(tl, &acc1, out1);
    }
}

/// Unpacks an `IQ4_XS` row: one `f16` per 256, a 6-bit scale per 32 split
/// across `scales_l`/`scales_h` and biased by -32, and 4-bit indices into
/// [`KVALUES_IQ4NL`] laid out exactly as `IQ4_NL`'s — low nibbles to elements
/// 0..16, high nibbles to 16..32 — so [`unpack_block_iq4_nl`] serves both.
///
/// Scales stay **integers** here, as with `Q4_K`/`Q6_K`: that is what lets the
/// dot accumulate in `i32` across a whole super-block.
fn unpack_k_iq4_xs(row: &[u8], out: &mut KRow) {
    const BLOCK_BYTES: usize = 2 + 2 + SUPER_BLOCK / 64 + SUPER_BLOCK / 2;
    for (s, block) in row.chunks_exact(BLOCK_BYTES).enumerate() {
        out.d[s] = read_f16(block, 0);
        let scales_h = u16::from_le_bytes([block[2], block[3]]);
        let scales_l = &block[4..8];
        let qs = &block[8..BLOCK_BYTES];
        for ib in 0..SUBS {
            let low = (scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F;
            let high = ((scales_h >> (2 * ib)) & 3) as u8;
            out.sc[s * SUBS + ib] = ((low | (high << 4)) as i16) - 32;
            let w: &mut [i8; 32] = (&mut out.q[(s * SUBS + ib) * 32..][..32])
                .try_into()
                .unwrap();
            unpack_block_iq4_nl(&qs[ib * 16..], w);
        }
    }
}

fn gemm_k_q6_k<const ISA: u8>(
    w0: &KRow,
    w1: &KRow,
    a: &ActQ8K,
    out0: &mut [f32],
    out1: &mut [f32],
) {
    for tl in 0..a.n_tile {
        let mut acc0 = [0f32; TOKEN_TILE];
        let mut acc1 = [0f32; TOKEN_TILE];
        let qtile = &a.q[tl * a.n_block * TOKEN_TILE * 32..];
        let dtile = &a.d[tl * a.n_super * TOKEN_TILE..];
        for s in 0..a.n_super {
            let mut isum0 = [0i32; TOKEN_TILE];
            let mut isum1 = [0i32; TOKEN_TILE];
            // `Q6_K` is symmetric — the `-32` bias is folded into the `int8`
            // weight — so there is no min pass, but its scales genuinely vary
            // per 16, hence [`GROUP`]-wide steps. |sc| <= 127 and a group
            // partial is at most 16*32*127 = 65024, so 16 of them stay inside
            // `i32`.
            for g in 0..Q6K_GROUPS {
                let e = s * SUPER_BLOCK + g * GROUP;
                let sc0 = w0.sc[s * Q6K_GROUPS + g] as i32;
                let sc1 = w1.sc[s * Q6K_GROUPS + g] as i32;
                let q0 = &w0.q[e..];
                let q1 = &w1.q[e..];
                // Element `e` sits at offset `e % 32` inside block `e / 32`,
                // and `GROUP` divides 32, so the 16 activations are contiguous.
                let xq = &qtile[(e / 32) * TOKEN_TILE * 32..];
                let off = e % 32;
                for k in 0..TOKEN_TILE {
                    let x = &xq[k * 32 + off..];
                    isum0[k] += sc0 * dot16::<ISA>(q0, x);
                    isum1[k] += sc1 * dot16::<ISA>(q1, x);
                }
            }
            let (d0, d1) = (w0.d[s], w1.d[s]);
            let ad = &dtile[s * TOKEN_TILE..];
            for k in 0..TOKEN_TILE {
                acc0[k] += ad[k] * (d0 * isum0[k] as f32);
                acc1[k] += ad[k] * (d1 * isum1[k] as f32);
            }
        }
        store_tile(tl, &acc0, out0);
        store_tile(tl, &acc1, out1);
    }
}

/// Writes one tile's accumulators, dropping the padded tail tokens.
#[inline]
fn store_tile(tl: usize, acc: &[f32; TOKEN_TILE], out: &mut [f32]) {
    let base = tl * TOKEN_TILE;
    let n = TOKEN_TILE.min(out.len().saturating_sub(base));
    out[base..base + n].copy_from_slice(&acc[..n]);
}

// =====================================================================
// Flat-activation prefill GEMM for the symmetric per-32 types.
//
// The K-quant GEMM above fixed three defects at once; this applies the first
// of them — the activation *layout* — to the path K-quants do not take.
//
// A `perf` profile of `Q8_0` prefill (600-token prompt, DWARF) put 74.8% of
// all time in `dot_unpacked_pair`, and inside it **74.9% in loads** against
// 12.4% in the actual arithmetic (`smull` + `sadalp`). The kernel was starved,
// not busy. Two load patterns accounted for it:
//
// * `ldp q26, q18, [x14, #-16]` (12.3% + 10.7% + 7.2% + 6.1%) — the `q`
//   vectors, reached through a `&[ActQ8]` of structs-of-`Vec`s. Each token is
//   a separate heap allocation 72 bytes of stride away, so four tokens are
//   four independent pointer chases the prefetcher cannot follow.
// * `ldr s17, [x13, x12, lsl #2]` (5.2% + 4.6% + 4.6% + 4.1% + 3.7%, ~22%
//   total) — **scalar** 32-bit loads of `a.d[b]`, the per-block activation
//   scale, fetched one at a time from four different `Vec<f32>`s.
//
// Flattening to `[tile][block][token][32]` for `q` and `[tile][block][token]`
// for `d` turns the first into four sequential streams and collapses the
// second into a single 16-byte load of four contiguous scales.
//
// Unlike the K path this does **not** widen the activation scale: `Q8_0` and
// `Q5_0` carry one weight scale per 32, so `d_act * d_w * sum` genuinely has
// to be accumulated in `f32` every block — exactly as ggml's own
// `ggml_vec_dot_q8_0_q8_0` does. This is a pure layout change, and the
// numbers it produces are bit-identical to the path it replaces.
//
// Why only the symmetric per-32 types: after the K GEMM landed, they are the
// only ones that reach here. [`supports`] and [`supports_k`] impose the *same*
// 256-divisibility on `Q4_K`/`Q6_K`, so any K-quant that gets through
// `supports` also satisfies `supports_k` and is routed away before this. That
// leaves `Q8_0`, `Q5_0` and `IQ4_NL` — all `per32`, all symmetric — so there
// is no `has_min` correction and no per-`GROUP` branch to carry.
//
// `IQ4_NL` qualifies for exactly the reason the other two do, despite its
// non-uniform quantization levels: `unpack_iq4_nl` resolves the 16-entry
// `KVALUES_IQ4NL` table into plain `int8` weights and leaves `has_min` false,
// so by the time a row reaches this kernel it is indistinguishable in shape
// from a `Q8_0` one. Anything that unpacks to per-32 scales and no min term
// belongs here; the test below is what pins that down rather than the type
// list itself.

/// Whether the flat-activation GEMM covers this type. Deliberately narrower
/// than [`supports`]: the per-`GROUP` (`Q6_K`) and asymmetric (`Q4_K`) shapes
/// are handled by the K-quant GEMM, and a type that satisfied neither would
/// fall back to [`dot_unpacked_pair`] rather than being handled wrongly here.
pub fn supports_flat(ggml_type: u32, in_dim: usize) -> bool {
    matches!(
        ggml_type,
        GGML_TYPE_Q8_0 | GGML_TYPE_Q5_0 | GGML_TYPE_IQ4_NL
    ) && in_dim.is_multiple_of(ACT_BLOCK)
}

/// Activations for [`dot_flat_pair`], laid out so a token tile is contiguous.
///
/// Same idea as [`ActQ8K`] but with the scale granularity the symmetric
/// per-32 types actually have: one `f32` per (block, token) rather than one
/// per (super-block, token), and no `bsum` — nothing here needs a min
/// correction.
pub struct ActQ8Flat {
    n_tokens: usize,
    n_block: usize,
    n_tile: usize,
    /// `[tile][block][token][32]`.
    q: Vec<i8>,
    /// `[tile][block][token]` — the four scales a tile needs for a given
    /// block are adjacent, which is the entire point of this struct.
    d: Vec<f32>,
}

impl ActQ8Flat {
    /// `x` is `[n_tokens][in_dim]`, token-major. `in_dim` must be a multiple
    /// of [`ACT_BLOCK`] — guaranteed by [`supports_flat`].
    pub fn quantize(x: &[f32], in_dim: usize, n_tokens: usize) -> Self {
        debug_assert_eq!(in_dim % ACT_BLOCK, 0);
        debug_assert_eq!(x.len(), n_tokens * in_dim);
        let n_block = in_dim / ACT_BLOCK;
        let n_tile = n_tokens.div_ceil(TOKEN_TILE);
        // Padding tokens in the last tile stay zero: a zero block has a zero
        // scale and contributes nothing, and their outputs are never read.
        // `k_gemm_result_is_independent_of_the_batch_size`'s analogue below
        // pins that down.
        let mut q = vec![0i8; n_tile * n_block * TOKEN_TILE * ACT_BLOCK];
        let mut d = vec![0f32; n_tile * n_block * TOKEN_TILE];
        for tl in 0..n_tile {
            for k in 0..TOKEN_TILE {
                let t = tl * TOKEN_TILE + k;
                if t >= n_tokens {
                    break;
                }
                let row = &x[t * in_dim..(t + 1) * in_dim];
                for b in 0..n_block {
                    let chunk = &row[b * ACT_BLOCK..(b + 1) * ACT_BLOCK];
                    let amax = chunk.iter().fold(0f32, |m, v| m.max(v.abs()));
                    let scale = amax / 127.0;
                    let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
                    d[(tl * n_block + b) * TOKEN_TILE + k] = scale;
                    let dst =
                        &mut q[((tl * n_block + b) * TOKEN_TILE + k) * ACT_BLOCK..][..ACT_BLOCK];
                    for (slot, &v) in dst.iter_mut().zip(chunk) {
                        // `round` then clamp, as in `quantize_act`: the
                        // extreme is exactly ±127, never -128.
                        *slot = (v * inv).round().clamp(-127.0, 127.0) as i8;
                    }
                }
            }
        }
        Self {
            n_tokens,
            n_block,
            n_tile,
            q,
            d,
        }
    }
}

/// Two weight rows against every token, over [`ActQ8Flat`]. The `UnpackedRow`
/// side is unchanged — only the activations were relaid out.
///
/// Both rows must be `per32` and symmetric; [`supports_flat`] is what
/// guarantees it at the call site.
pub fn dot_flat_pair(
    w0: &UnpackedRow,
    w1: &UnpackedRow,
    a: &ActQ8Flat,
    out0: &mut [f32],
    out1: &mut [f32],
) {
    debug_assert_eq!(a.n_tokens, out0.len());
    debug_assert_eq!(a.n_tokens, out1.len());
    debug_assert!(w0.per32 && w1.per32);
    debug_assert!(!w0.has_min && !w1.has_min);
    #[cfg(target_arch = "aarch64")]
    if have_dotprod() {
        return dot_flat_pair_impl::<ISA_DOTPROD>(w0, w1, a, out0, out1);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if have_vnni() {
            // Safety: see `dot_row`.
            return unsafe { dot_flat_pair_vnni(w0, w1, a, out0, out1) };
        }
        if is_x86_feature_detected!("avx2") {
            // Safety: guarded by the runtime feature check above.
            return unsafe { dot_flat_pair_avx2(w0, w1, a, out0, out1) };
        }
    }
    dot_flat_pair_impl::<ISA_BASELINE>(w0, w1, a, out0, out1)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512vnni,avx512vl,avx2,ssse3")]
unsafe fn dot_flat_pair_vnni(
    w0: &UnpackedRow,
    w1: &UnpackedRow,
    a: &ActQ8Flat,
    out0: &mut [f32],
    out1: &mut [f32],
) {
    dot_flat_pair_impl::<ISA_VNNI>(w0, w1, a, out0, out1)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_flat_pair_avx2(
    w0: &UnpackedRow,
    w1: &UnpackedRow,
    a: &ActQ8Flat,
    out0: &mut [f32],
    out1: &mut [f32],
) {
    dot_flat_pair_impl::<ISA_AVX2>(w0, w1, a, out0, out1)
}

/// One weight row. Runs the pair kernel against itself and discards the
/// duplicate column — the trailing row of an odd `out_dim` only, so the
/// wasted half is at most one row per matmul.
pub fn dot_flat_multi(w: &UnpackedRow, a: &ActQ8Flat, out: &mut [f32], scratch: &mut Vec<f32>) {
    scratch.clear();
    scratch.resize(out.len(), 0.0);
    dot_flat_pair(w, w, a, out, scratch);
}

/// Parenthesised to match [`dot_unpacked_pair_impl`]'s `per32`/no-min branch
/// term for term, so both produce bit-identical results and the tests can
/// compare them for exact equality.
fn dot_flat_pair_impl<const ISA: u8>(
    w0: &UnpackedRow,
    w1: &UnpackedRow,
    a: &ActQ8Flat,
    out0: &mut [f32],
    out1: &mut [f32],
) {
    for tl in 0..a.n_tile {
        let mut acc0 = [0f32; TOKEN_TILE];
        let mut acc1 = [0f32; TOKEN_TILE];
        // One tile's activations are one contiguous run: the block loop walks
        // it forwards and never revisits, which is what the old layout could
        // not do.
        let qtile = &a.q[tl * a.n_block * TOKEN_TILE * ACT_BLOCK..];
        let dtile = &a.d[tl * a.n_block * TOKEN_TILE..];
        for b in 0..a.n_block {
            let (s0, s1) = (w0.scale[b], w1.scale[b]);
            let (q0, q1) = (&w0.q[b * ACT_BLOCK..], &w1.q[b * ACT_BLOCK..]);
            let xq = &qtile[b * TOKEN_TILE * ACT_BLOCK..];
            // Four adjacent `f32` — one vector load, not four scalar ones.
            let ad = &dtile[b * TOKEN_TILE..];
            for k in 0..TOKEN_TILE {
                let x = &xq[k * ACT_BLOCK..];
                let d = ad[k];
                acc0[k] += d * s0 * dot32::<ISA>(q0, x) as f32;
                acc1[k] += d * s1 * dot32::<ISA>(q1, x) as f32;
            }
        }
        store_tile(tl, &acc0, out0);
        store_tile(tl, &acc1, out1);
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
        GGML_TYPE_IQ4_XS => dot_iq4_xs::<ISA>(row, act),
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
/// `IQ4_XS` decode. One `dot32` per sub-block rather than `Q6_K`'s two
/// `dot16`, because the scale is uniform across all 32 — the same reason
/// `UnpackedRow::per32` exists.
fn dot_iq4_xs<const ISA: u8>(row: &[u8], act: &ActQ8) -> f32 {
    const BLOCK_BYTES: usize = 2 + 2 + SUPER_BLOCK / 64 + SUPER_BLOCK / 2;
    let mut total = 0f32;
    let mut blk = 0usize;
    let mut w = [0i8; 32];
    for block in row.chunks_exact(BLOCK_BYTES) {
        let d = read_f16(block, 0);
        let scales_h = u16::from_le_bytes([block[2], block[3]]);
        let scales_l = &block[4..8];
        let qs = &block[8..BLOCK_BYTES];
        for ib in 0..SUBS {
            let low = (scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F;
            let high = ((scales_h >> (2 * ib)) & 3) as u8;
            let ls = ((low | (high << 4)) as i32) - 32;
            unpack_block_iq4_nl(&qs[ib * 16..], &mut w);
            total += act.d[blk] * d * (ls as f32) * dot32::<ISA>(&w, &act.q[blk * 32..]) as f32;
            blk += 1;
        }
    }
    total
}

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
            GGML_TYPE_IQ4_XS => (136, 256),
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
                GGML_TYPE_IQ4_XS => block[0..2].copy_from_slice(&[0x00, 0x30]),
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

    /// The vectorized nibble->level lookup must agree with the scalar
    /// reference for **every** byte value, not just the ones a fixture
    /// happens to produce: all 256 inputs are enumerated, which covers each
    /// of the 16 table entries in both nibble positions.
    #[test]
    fn iq4_nl_block_unpack_is_exact_for_every_byte_value() {
        for base in 0..=255u8 {
            let qs: Vec<u8> = (0..16u8)
                .map(|j| base.wrapping_add(j.wrapping_mul(17)))
                .collect();
            let mut want = [0i8; 32];
            let mut got = [0i8; 32];
            unpack_block_iq4_nl_scalar(&qs, &mut want);
            unpack_block_iq4_nl(&qs, &mut got);
            assert_eq!(got, want, "qs = {qs:?}");
        }
        // And the trivial case that exercises every table entry in order.
        let qs: Vec<u8> = (0..16u8).map(|j| j | (15 - j) << 4).collect();
        let mut want = [0i8; 32];
        let mut got = [0i8; 32];
        unpack_block_iq4_nl_scalar(&qs, &mut want);
        unpack_block_iq4_nl(&qs, &mut got);
        assert_eq!(got, want);
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

    /// The decode half of `IQ4_XS`. Widths are multiples of 256 because that
    /// is what `supports` requires of it — 4864 already is one (19 blocks),
    /// so it is the same row width the `IQ4_NL` test above uses.
    #[test]
    fn iq4_xs_matches_dequantize_reference() {
        for seed in [1, 7, 99] {
            check(GGML_TYPE_IQ4_XS, 256, seed);
            check(GGML_TYPE_IQ4_XS, 2048, seed);
            check(GGML_TYPE_IQ4_XS, 4864, seed);
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

    // ------------------------------------------- K-quant prefill GEMM

    /// A weight matrix of `out_dim` K-quant rows with finite `f16` scales,
    /// plus `n_tokens` rows of activations.
    fn k_fixture(
        ggml_type: u32,
        in_dim: usize,
        out_dim: usize,
        n_tokens: usize,
        seed: u32,
    ) -> (Vec<u8>, Vec<f32>, usize) {
        let block_bytes = match ggml_type {
            GGML_TYPE_Q4_K => 144,
            GGML_TYPE_IQ4_XS => 136, // 2 + 2 + 4 + 128
            _ => 210,                // Q6_K
        };
        let row_bytes = in_dim / 256 * block_bytes;
        let mut raw = pseudo_bytes(row_bytes * out_dim, seed);
        for block in raw.chunks_exact_mut(block_bytes) {
            match ggml_type {
                GGML_TYPE_Q4_K => {
                    block[0..2].copy_from_slice(&[0x00, 0x30]); // d = 0.125
                    block[2..4].copy_from_slice(&[0x00, 0x2c]); // dmin = 0.0625
                }
                // `d` only; the 6-bit scales stay pseudo-random, which is
                // what exercises the `scales_l`/`scales_h` split and the -32
                // bias (so a sub-block scale can legitimately be negative).
                GGML_TYPE_IQ4_XS => block[0..2].copy_from_slice(&[0x00, 0x30]),
                _ => block[208..210].copy_from_slice(&[0x00, 0x30]),
            }
        }
        let x: Vec<f32> = (0..in_dim * n_tokens)
            .map(|i| ((i * 37 % 23) as f32 - 11.0) * 0.031)
            .collect();
        (raw, x, row_bytes)
    }

    /// Runs the K-quant GEMM the way `CpuBackend::matmul_k_gemm` does, and
    /// returns it transposed as `[out_dim][n_tokens]`.
    fn k_gemm(
        ggml_type: u32,
        raw: &[u8],
        row_bytes: usize,
        in_dim: usize,
        out_dim: usize,
        x: &[f32],
        n_tokens: usize,
    ) -> Vec<f32> {
        let acts = ActQ8K::quantize(x, in_dim, n_tokens);
        let mut yt = vec![0f32; out_dim * n_tokens];
        let (mut s0, mut s1) = (KRow::new(), KRow::new());
        let mut scratch = Vec::new();
        for (pair, dst) in yt.chunks_mut(n_tokens * 2).enumerate() {
            let o0 = pair * 2;
            let row = |o: usize| &raw[o * row_bytes..(o + 1) * row_bytes];
            unpack_k_row(ggml_type, row(o0), in_dim, &mut s0);
            if dst.len() == n_tokens * 2 {
                unpack_k_row(ggml_type, row(o0 + 1), in_dim, &mut s1);
                let (d0, d1) = dst.split_at_mut(n_tokens);
                dot_k_pair(&s0, &s1, &acts, d0, d1);
            } else {
                dot_k_multi(&s0, &acts, dst, &mut scratch);
            }
        }
        yt
    }

    /// The K-quant GEMM must land inside the same `int8`-activation error
    /// budget the generic path is held to. Its activation scale is coarser —
    /// one per 256 rather than one per 32, matching ggml's `block_q8_K` — so
    /// this is the check that the coarser scale stays acceptable, not just
    /// that the indexing is right.
    #[test]
    fn k_gemm_matches_dequantize_reference() {
        for ggml_type in [GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ4_XS] {
            for in_dim in [256usize, 2048, 8192] {
                // 5 rows and 7 tokens: odd both ways, so the trailing-row and
                // tile-padding paths both run.
                let (out_dim, n_tokens) = (5usize, 7usize);
                let (raw, x, row_bytes) = k_fixture(ggml_type, in_dim, out_dim, n_tokens, 4242);
                let got = k_gemm(ggml_type, &raw, row_bytes, in_dim, out_dim, &x, n_tokens);
                for o in 0..out_dim {
                    let w = quant::dequantize(
                        ggml_type,
                        &raw[o * row_bytes..(o + 1) * row_bytes],
                        in_dim,
                    )
                    .unwrap();
                    for t in 0..n_tokens {
                        let xs = &x[t * in_dim..(t + 1) * in_dim];
                        let reference: f32 = w.iter().zip(xs).map(|(a, b)| a * b).sum();
                        let magnitude: f32 = w.iter().zip(xs).map(|(a, b)| (a * b).abs()).sum();
                        let err = (got[o * n_tokens + t] - reference).abs();
                        assert!(
                            err <= 0.01 * magnitude.max(1e-6),
                            "type {ggml_type} in_dim {in_dim} row {o} token {t}: \
                             got {} want {reference} (err {err}, term-magnitude {magnitude})",
                            got[o * n_tokens + t]
                        );
                    }
                }
            }
        }
    }

    /// Tile padding must not leak into the answer: a token's result has to be
    /// the same whatever else is batched alongside it. This is the check that
    /// the zero-padded tail tokens really contribute nothing, and that
    /// `store_tile` drops them rather than writing past `out`.
    #[test]
    fn k_gemm_result_is_independent_of_the_batch_size() {
        let in_dim = 512;
        let out_dim = 4;
        for ggml_type in [GGML_TYPE_Q4_K, GGML_TYPE_Q6_K] {
            let (raw, x, row_bytes) = k_fixture(ggml_type, in_dim, out_dim, 8, 77);
            let full = k_gemm(ggml_type, &raw, row_bytes, in_dim, out_dim, &x, 8);
            for n in [2usize, 3, 5, 7] {
                let part = k_gemm(
                    ggml_type,
                    &raw,
                    row_bytes,
                    in_dim,
                    out_dim,
                    &x[..n * in_dim],
                    n,
                );
                for o in 0..out_dim {
                    for t in 0..n {
                        assert_eq!(
                            part[o * n + t],
                            full[o * 8 + t],
                            "type {ggml_type}, {n} of 8 tokens, row {o} token {t}"
                        );
                    }
                }
            }
        }
    }

    /// A `KRow` reused across rows — and across *types*, which changes every
    /// buffer's length — must not keep a stale tail.
    #[test]
    fn k_row_buffer_is_safe_to_reuse() {
        let in_dim = 512;
        let n_tokens = 4;
        let mut scratch = KRow::new();
        for ggml_type in [
            GGML_TYPE_Q4_K,
            GGML_TYPE_Q6_K,
            GGML_TYPE_IQ4_XS,
            GGML_TYPE_Q4_K,
            GGML_TYPE_IQ4_XS,
            GGML_TYPE_Q6_K,
        ] {
            let (raw, x, row_bytes) = k_fixture(ggml_type, in_dim, 2, n_tokens, 31);
            let acts = ActQ8K::quantize(&x, in_dim, n_tokens);
            let mut fresh = KRow::new();
            unpack_k_row(ggml_type, &raw[row_bytes..], in_dim, &mut fresh);
            unpack_k_row(ggml_type, &raw[..row_bytes], in_dim, &mut scratch);
            unpack_k_row(ggml_type, &raw[row_bytes..], in_dim, &mut scratch);

            let (mut want, mut got) = (vec![0f32; n_tokens], vec![0f32; n_tokens]);
            let mut sp = Vec::new();
            dot_k_multi(&fresh, &acts, &mut want, &mut sp);
            dot_k_multi(&scratch, &acts, &mut got, &mut sp);
            assert_eq!(got, want, "stale state for ggml_type {ggml_type}");
        }
    }

    #[test]
    fn supports_k_covers_only_the_k_quants() {
        assert!(supports_k(GGML_TYPE_Q4_K, 2048));
        assert!(supports_k(GGML_TYPE_Q6_K, 8192));
        // `IQ4_XS` is a 256-element super-block type like the other two, and
        // must NOT be claimed by `supports_flat` — its scales are per-32 but
        // it accumulates per-256, which is the K-quant kernel's shape.
        assert!(supports_k(GGML_TYPE_IQ4_XS, 2048));
        assert!(!supports_flat(GGML_TYPE_IQ4_XS, 2048));
        // 896 is not a multiple of 256 — a `Qwen2.5-0.5B` row.
        assert!(!supports_k(GGML_TYPE_Q4_K, 896));
        // `Q8_0`/`Q5_0` have no super-block to accumulate across and keep the
        // generic GEMM.
        assert!(!supports_k(GGML_TYPE_Q8_0, 2048));
        assert!(!supports_k(GGML_TYPE_Q5_0, 2048));
    }

    /// Weight rows for the flat GEMM's types, with valid `f16` scales.
    /// Random bytes would give NaN/inf scales, so the scale field of every
    /// block is overwritten.
    fn flat_fixture(
        ggml_type: u32,
        in_dim: usize,
        out_dim: usize,
        n_tokens: usize,
        seed: u32,
    ) -> (Vec<u8>, Vec<f32>, usize) {
        let block_bytes = match ggml_type {
            GGML_TYPE_Q8_0 => 34,   // 2 + 32
            GGML_TYPE_IQ4_NL => 18, // 2 + 16
            _ => 22,                // Q5_0: 2 + 4 + 16
        };
        let row_bytes = in_dim / 32 * block_bytes;
        let mut raw = pseudo_bytes(row_bytes * out_dim, seed);
        for block in raw.chunks_exact_mut(block_bytes) {
            block[0..2].copy_from_slice(&[0x00, 0x30]); // d = 0.125
        }
        let x: Vec<f32> = (0..in_dim * n_tokens)
            .map(|i| ((i * 37 % 23) as f32 - 11.0) * 0.031)
            .collect();
        (raw, x, row_bytes)
    }

    /// Drives the flat GEMM the way `cpu::matmul_flat_gemm` does.
    fn flat_gemm(
        ggml_type: u32,
        raw: &[u8],
        row_bytes: usize,
        in_dim: usize,
        out_dim: usize,
        x: &[f32],
        n_tokens: usize,
    ) -> Vec<f32> {
        let acts = ActQ8Flat::quantize(x, in_dim, n_tokens);
        let mut yt = vec![0f32; out_dim * n_tokens];
        let (mut s0, mut s1) = (UnpackedRow::new(), UnpackedRow::new());
        let mut scratch = Vec::new();
        for (pair, dst) in yt.chunks_mut(n_tokens * 2).enumerate() {
            let o0 = pair * 2;
            let row = |o: usize| &raw[o * row_bytes..(o + 1) * row_bytes];
            unpack_row(ggml_type, row(o0), in_dim, &mut s0);
            if dst.len() == n_tokens * 2 {
                unpack_row(ggml_type, row(o0 + 1), in_dim, &mut s1);
                let (d0, d1) = dst.split_at_mut(n_tokens);
                dot_flat_pair(&s0, &s1, &acts, d0, d1);
            } else {
                dot_flat_multi(&s0, &acts, dst, &mut scratch);
            }
        }
        yt
    }

    /// Same, through the path this replaces.
    fn generic_gemm(
        ggml_type: u32,
        raw: &[u8],
        row_bytes: usize,
        in_dim: usize,
        out_dim: usize,
        x: &[f32],
        n_tokens: usize,
    ) -> Vec<f32> {
        let acts: Vec<ActQ8> = (0..n_tokens)
            .map(|t| quantize_act(&x[t * in_dim..(t + 1) * in_dim]))
            .collect();
        let mut yt = vec![0f32; out_dim * n_tokens];
        let (mut s0, mut s1) = (UnpackedRow::new(), UnpackedRow::new());
        for (pair, dst) in yt.chunks_mut(n_tokens * 2).enumerate() {
            let o0 = pair * 2;
            let row = |o: usize| &raw[o * row_bytes..(o + 1) * row_bytes];
            unpack_row(ggml_type, row(o0), in_dim, &mut s0);
            if dst.len() == n_tokens * 2 {
                unpack_row(ggml_type, row(o0 + 1), in_dim, &mut s1);
                let (d0, d1) = dst.split_at_mut(n_tokens);
                dot_unpacked_pair(&s0, &s1, &acts, d0, d1);
            } else {
                dot_unpacked_multi(&s0, &acts, dst);
            }
        }
        yt
    }

    /// The strong claim: relaying the activations out changed *where* values
    /// are read from, not the arithmetic or the order of it. Exact equality,
    /// not a tolerance — anything else means the two paths disagree and the
    /// weaker reference test below could hide it inside its 1% budget.
    ///
    /// `n_tokens` 8 is a whole number of tiles and 7 is not, so the
    /// tile-padding path is compared too (the old code handles that tail with
    /// a separate scalar loop, this one with zero-padding).
    #[test]
    fn flat_gemm_is_bit_identical_to_the_generic_gemm() {
        for ggml_type in [GGML_TYPE_Q8_0, GGML_TYPE_Q5_0, GGML_TYPE_IQ4_NL] {
            for in_dim in [32usize, 896, 4864] {
                for n_tokens in [2usize, 7, 8] {
                    let out_dim = 5; // odd, so the trailing-row path runs
                    let (raw, x, row_bytes) =
                        flat_fixture(ggml_type, in_dim, out_dim, n_tokens, 909);
                    let flat = flat_gemm(ggml_type, &raw, row_bytes, in_dim, out_dim, &x, n_tokens);
                    let generic =
                        generic_gemm(ggml_type, &raw, row_bytes, in_dim, out_dim, &x, n_tokens);
                    assert_eq!(
                        flat, generic,
                        "type {ggml_type} in_dim {in_dim} n_tokens {n_tokens}"
                    );
                }
            }
        }
    }

    #[test]
    fn flat_gemm_matches_dequantize_reference() {
        for ggml_type in [GGML_TYPE_Q8_0, GGML_TYPE_Q5_0, GGML_TYPE_IQ4_NL] {
            for in_dim in [896usize, 4864] {
                let (out_dim, n_tokens) = (5usize, 7usize);
                let (raw, x, row_bytes) = flat_fixture(ggml_type, in_dim, out_dim, n_tokens, 4242);
                let got = flat_gemm(ggml_type, &raw, row_bytes, in_dim, out_dim, &x, n_tokens);
                for o in 0..out_dim {
                    let w = quant::dequantize(
                        ggml_type,
                        &raw[o * row_bytes..(o + 1) * row_bytes],
                        in_dim,
                    )
                    .unwrap();
                    for t in 0..n_tokens {
                        let xs = &x[t * in_dim..(t + 1) * in_dim];
                        let reference: f32 = w.iter().zip(xs).map(|(a, b)| a * b).sum();
                        let magnitude: f32 = w.iter().zip(xs).map(|(a, b)| (a * b).abs()).sum();
                        let err = (got[o * n_tokens + t] - reference).abs();
                        assert!(
                            err <= 0.01 * magnitude.max(1e-6),
                            "type {ggml_type} in_dim {in_dim} row {o} token {t}: \
                             got {} want {reference} (err {err}, term-magnitude {magnitude})",
                            got[o * n_tokens + t]
                        );
                    }
                }
            }
        }
    }

    /// The zero-padded tokens of a partial tile must not leak into any real
    /// token's result: batching 2, 3, 5 or 7 tokens has to give exactly what
    /// batching 8 gives for those same tokens.
    #[test]
    fn flat_gemm_result_is_independent_of_the_batch_size() {
        let (ggml_type, in_dim, out_dim) = (GGML_TYPE_Q8_0, 896usize, 4usize);
        let (raw, x, row_bytes) = flat_fixture(ggml_type, in_dim, out_dim, 8, 31337);
        let full = flat_gemm(ggml_type, &raw, row_bytes, in_dim, out_dim, &x, 8);
        for n in [2usize, 3, 5, 7] {
            let got = flat_gemm(
                ggml_type,
                &raw,
                row_bytes,
                in_dim,
                out_dim,
                &x[..n * in_dim],
                n,
            );
            for o in 0..out_dim {
                for t in 0..n {
                    assert_eq!(
                        got[o * n + t],
                        full[o * 8 + t],
                        "n_tokens {n} row {o} token {t}"
                    );
                }
            }
        }
    }

    #[test]
    fn supports_flat_covers_only_the_symmetric_per32_types() {
        assert!(supports_flat(GGML_TYPE_Q8_0, 896));
        assert!(supports_flat(GGML_TYPE_Q5_0, 4864));
        // `IQ4_NL` unpacks its 16-level table into plain `int8` with no min
        // term, so it is the same shape as the other two by the time the
        // kernel sees it. Omitting it here would not be wrong, just slow —
        // it would fall back to the generic GEMM this path exists to beat.
        assert!(supports_flat(GGML_TYPE_IQ4_NL, 896));
        // The K-quants are routed to `dot_k_pair` before this is consulted,
        // and this kernel has no min correction and no per-GROUP branch, so
        // it must decline them rather than compute them wrongly.
        assert!(!supports_flat(GGML_TYPE_Q4_K, 2048));
        assert!(!supports_flat(GGML_TYPE_Q6_K, 2048));
        // A row that is not a whole number of 32-element blocks.
        assert!(!supports_flat(GGML_TYPE_Q8_0, 100));
    }
}
