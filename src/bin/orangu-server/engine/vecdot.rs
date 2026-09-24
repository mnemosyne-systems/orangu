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
#[cfg(target_arch = "aarch64")]
use super::quant::unpack_block_pq2_0_neon;
use super::quant::{
    GGML_TYPE_BF16, GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS,
    GGML_TYPE_PQ2_0, GGML_TYPE_PTQ1_0, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_K, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0,
    PQ2_0_BLOCK_BYTES, PTQ1_0_BLOCK_BYTES, QK_PRISM, get_scale_min_k4, read_f16,
    unpack_block_pq2_0, unpack_block_ptq1_0, unpack_q3_k_scales,
};
use rayon::prelude::*;

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
    for (b, chunk) in x.as_chunks::<ACT_BLOCK>().0.iter().enumerate() {
        let out = &mut q[b * ACT_BLOCK..(b + 1) * ACT_BLOCK];
        d.push(quantize_block::<true>(chunk, out, &mut sums));
    }
    ActQ8 { q, d, sums }
}

/// The portable reference for [`quantize_act`], and the definition of what
/// its vector kernel must reproduce bit for bit. Also the live path on every
/// target without the AVX2 floor.
#[cfg_attr(not(test), allow(dead_code))]
pub fn quantize_act_scalar(x: &[f32]) -> ActQ8 {
    debug_assert_eq!(x.len() % ACT_BLOCK, 0);
    let mut q = vec![0i8; x.len()];
    let mut d = Vec::with_capacity(x.len() / ACT_BLOCK);
    let mut sums = Vec::with_capacity(x.len() / GROUP);
    for (b, chunk) in x.as_chunks::<ACT_BLOCK>().0.iter().enumerate() {
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
        for (half, group) in out.as_chunks_mut::<GROUP>().0.iter_mut().enumerate() {
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

/// Quantizes one [`ACT_BLOCK`] of activations to `int8`, writing them into
/// `out` and returning the block's scale — **the** activation-quantization
/// kernel, shared by [`quantize_act`] and [`ActQ8Flat::quantize`], which
/// differ only in where their outputs land.
///
/// Three details carry bit-identity with the scalar form, and each is a
/// deliberate choice rather than the obvious one:
///
/// * **The `abs`-max fold** goes lane-parallel where the scalar is
///   sequential. Sound because `max` is associative and commutative — but
///   only away from `NaN`. `wide`'s `max` is `rhs.is_nan().select(self,
///   max(self, rhs))`, and `max_m256` itself returns its right operand when
///   either is unordered, so the pair ignores `NaN` on *either* side exactly
///   as `f32::max` does. Operand order here is therefore not load-bearing;
///   the `NaN`-blindness is. `fast_max` would not do: it lets a `NaN` reach
///   the accumulator, and while the scalar horizontal reduce happens to
///   discard it again, that is an accident of the reduce rather than a
///   property of the fold.
/// * **`round`.** `wide` 1.6.1's `f32x8::round` is ties-*away*-from-zero
///   (`|x| + 0.5`, truncate, with the `0.5.next_down()` correction and a
///   `8388608.0` bounds mask that passes large values, infinities and `NaN`
///   through unchanged). That is precisely `f32::round`, not the
///   `roundps`-native ties-to-even, so this is exact rather than merely
///   close. `round_ties_even` here would silently change every logit.
/// * **`NaN` to zero, before the clamp.** Scalar `f32::clamp` propagates
///   `NaN` and the subsequent `as i8` saturating cast turns it into `0`. No
///   vector min/max pair reproduces that, so the `NaN` lanes are zeroed
///   explicitly and the clamp can then use the branch-free `fast_*` forms.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
fn quantize_block<const SUMS: bool>(chunk: &[f32], out: &mut [i8], sums: &mut Vec<i32>) -> f32 {
    use wide::{f32x8, i32x8};

    const LANES: usize = 8;
    const VECS: usize = ACT_BLOCK / LANES;
    debug_assert_eq!(chunk.len(), ACT_BLOCK);
    debug_assert_eq!(out.len(), ACT_BLOCK);

    let load = |s: &[f32], i: usize| -> f32x8 {
        let mut lane = [0f32; LANES];
        lane.copy_from_slice(&s[i..i + LANES]);
        f32x8::from(lane)
    };
    let v: [f32x8; VECS] = std::array::from_fn(|k| load(chunk, k * LANES));

    let mut amax = f32x8::ZERO;
    for vk in &v {
        amax = amax.max(vk.abs());
    }
    let amax = amax.to_array().iter().fold(0f32, |m, &a| m.max(a));

    let scale = amax / 127.0;
    let inv = f32x8::from(if scale > 0.0 { 1.0 / scale } else { 0.0 });
    let lo = f32x8::from(-127.0);
    let hi = f32x8::from(127.0);

    // `SUMS` is a const parameter rather than a second function: the two
    // callers want the same arithmetic and differ only in whether they need
    // the per-[`GROUP`] sums, and monomorphizing keeps the accumulation
    // inside the vector loop for the caller that does — while leaving the
    // other's code free of it entirely.
    let mut sum = i32x8::ZERO;
    for (k, vk) in v.iter().enumerate() {
        let y = (*vk * inv).round();
        let y = y.is_nan().select(f32x8::ZERO, y);
        let qi = y.fast_max(lo).fast_min(hi).trunc_int();
        if SUMS {
            sum += qi;
        }
        for (slot, w) in out[k * LANES..(k + 1) * LANES]
            .iter_mut()
            .zip(qi.to_array())
        {
            *slot = w as i8;
        }
        // One [`GROUP`] is two lanes' worth, so a sum lands every other
        // vector. Integers, so the reduction order is free.
        if SUMS && (k * LANES + LANES).is_multiple_of(GROUP) {
            sums.push(sum.to_array().iter().sum());
            sum = i32x8::ZERO;
        }
    }
    scale
}

/// [`quantize_block`] on every target without the AVX2 floor, and the
/// definition the vector form reproduces bit for bit.
#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
)))]
#[inline(always)]
fn quantize_block<const SUMS: bool>(chunk: &[f32], out: &mut [i8], sums: &mut Vec<i32>) -> f32 {
    debug_assert_eq!(chunk.len(), ACT_BLOCK);
    debug_assert_eq!(out.len(), ACT_BLOCK);
    let amax = chunk.iter().fold(0f32, |m, v| m.max(v.abs()));
    let scale = amax / 127.0;
    let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
    for (slot, &v) in out.iter_mut().zip(chunk) {
        *slot = (v * inv).round().clamp(-127.0, 127.0) as i8;
    }
    if SUMS {
        for group in out.as_chunks::<GROUP>().0 {
            sums.push(group.iter().map(|&v| v as i32).sum());
        }
    }
    scale
}

/// Whether a fused kernel exists for `ggml_type` at this `in_dim`. The
/// `in_dim` check is not paranoia: a GGUF row is only guaranteed to be a
/// whole number of blocks, and `Q4_K`/`Q5_K`/`Q6_K` need 256 | `in_dim` while
/// `Q8_0`/`Q5_0`/`IQ4_NL` need only 32 — a `Qwen2.5-0.5B` is exactly the
/// case that mixes both (`embedding_length` 896 is 28 blocks of 32 but not
/// a multiple of 256, which is why its layer weights are `Q5_0`/`IQ4_NL`
/// and only its `256`-divisible `ffn_down` is a K-quant).
///
/// Every quantized type these models actually use is now here. The float
/// types (`F32`, `F16`, `BF16`) have no int8 weight form and take
/// [`supports_float`] / [`dot_row_f32`] instead; anything left over falls to
/// `CpuBackend::matmul_dequant`, which costs far more than its share of a
/// model's weights suggests — `Q5_K` was 13.9% of `Phi-4-mini`'s bytes and
/// 48.9% of its decode time before it was added here.
///
/// # Adding a type
///
/// There are **five** places a type must be registered on the CPU path, and
/// they are separate dispatches rather than one table — declaring support in
/// [`supports`] without adding the matching arm elsewhere compiles fine and
/// panics at the first matmul:
///
/// 1. [`supports`] — gates the fused path at all.
/// 2. [`dot_row_impl`] — decode (`n_tokens == 1`), straight from the bytes.
/// 3. [`unpack_row`] — the generic prefill GEMM's unpack.
/// 4. [`supports_k`] + [`unpack_k_row`] — only if it is a 256-element
///    super-block type that should take the K-quant GEMM.
/// 5. [`supports_flat`] — only if it is symmetric with per-32 scales.
///
/// 4 and 5 are mutually exclusive, and a type may need neither. Adding
/// `Q5_K` needed 1-4; step 2 was missed on the first attempt and the
/// `q5_k_matches_dequantize_reference` test is what caught it.
pub fn supports(ggml_type: u32, in_dim: usize) -> bool {
    match ggml_type {
        GGML_TYPE_Q8_0 | GGML_TYPE_Q5_0 | GGML_TYPE_Q4_0 | GGML_TYPE_Q5_1 | GGML_TYPE_IQ4_NL => {
            in_dim.is_multiple_of(32)
        }
        GGML_TYPE_Q2_K | GGML_TYPE_Q3_K | GGML_TYPE_Q4_K | GGML_TYPE_Q5_K | GGML_TYPE_Q6_K
        | GGML_TYPE_IQ4_XS => in_dim.is_multiple_of(256),
        GGML_TYPE_PQ2_0 | GGML_TYPE_PTQ1_0 => in_dim.is_multiple_of(QK_PRISM),
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
    // `Q6_K`, `Q3_K` and `Q2_K` carry a scale per 16; everything else is
    // uniform across a 32-element block.
    let stride = if matches!(ggml_type, GGML_TYPE_Q6_K | GGML_TYPE_Q3_K | GGML_TYPE_Q2_K) {
        GROUP
    } else {
        32
    };
    out.resize_for(in_dim, stride);
    match ggml_type {
        GGML_TYPE_Q8_0 => unpack_q8_0(row, out),
        GGML_TYPE_Q5_0 => unpack_q5_0(row, out),
        GGML_TYPE_Q4_0 => unpack_q4_0(row, out),
        GGML_TYPE_Q5_1 => unpack_q5_1(row, out),
        GGML_TYPE_IQ4_NL => unpack_iq4_nl(row, out),
        GGML_TYPE_IQ4_XS => unpack_iq4_xs(row, out),
        GGML_TYPE_Q2_K => unpack_q2_k(row, out),
        GGML_TYPE_Q3_K => unpack_q3_k(row, out),
        GGML_TYPE_Q4_K => unpack_q4_k(row, out),
        GGML_TYPE_Q5_K => unpack_q5_k(row, out),
        GGML_TYPE_Q6_K => unpack_q6_k(row, out),
        GGML_TYPE_PQ2_0 => unpack_pq2_0(row, out),
        GGML_TYPE_PTQ1_0 => unpack_ptq1_0(row, out),
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

/// `#[inline]` for the reason the `ISA_*` docs give: this is called from
/// inside `#[target_feature]` wrappers, and Rust will not inline a
/// `#[target_feature]` leaf into a caller that lacks the feature. Left
/// out-of-line, the `ISA_VNNI` monomorphization could not fold its
/// `dot16`/`dot32` calls in and carried **15 real `call`s** — one per block
/// — where the AVX2 one had none, because AVX2 is a compile-time baseline
/// here and needs no wrapper to cross.
#[inline(always)]
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
/// `Q4_0` is `Q5_0` with no high-bit plane and a `-8` zero point, and `Q5_1`
/// is `Q5_0`'s bit extraction with no zero point at all (its offset is the
/// per-block `m`, applied in `f32` against the activation sum instead). So all
/// three share one unpack, parameterized by `qh` and `bias`.
///
/// `bias` stays a runtime argument rather than a const generic for the reason
/// given on [`unpack_q6k_run_neon`]: every call site passes a literal and
/// `#[inline(always)]` lets LLVM fold it, without a second monomorphization.
#[inline(always)]
fn unpack_block_q5_bits(qh: u32, qs: &[u8], bias: i8, w: &mut [i8; 32]) {
    #[cfg(target_arch = "aarch64")]
    {
        // Safety: `qs` is >= 16 bytes and `w` is exactly 32; NEON is
        // baseline on aarch64.
        unsafe { unpack_block_q5_bits_neon(qh, qs, bias, w) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        // The detection is cached by the standard library — one relaxed
        // load per block against the sixteen scalar iterations it replaces.
        // Callers that already know their ISA use
        // [`unpack_block_q5_bits_isa`] and pay nothing.
        if is_x86_feature_detected!("avx2") && q5_vector_unpack_on() {
            // Safety: guarded by the runtime feature check; `qs` is >= 16
            // bytes and `w` is exactly 32.
            return unsafe { unpack_block_q5_bits_avx2(qh, qs, bias, w) };
        }
        unpack_block_q5_bits_scalar(qh, qs, bias, w)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        unpack_block_q5_bits_scalar(qh, qs, bias, w)
    }
}

/// [`unpack_block_q5_bits`] for a caller that has already chosen its ISA —
/// the monomorphized dot kernels — so the choice costs nothing per block.
#[inline(always)]
fn unpack_block_q5_bits_isa<const ISA: u8>(qh: u32, qs: &[u8], bias: i8, w: &mut [i8; 32]) {
    #[cfg(target_arch = "x86_64")]
    if (ISA == ISA_AVX2 || ISA == ISA_VNNI) && q5_vector_unpack_on() {
        // Safety: reachable only from a wrapper that declares avx2; `qs` is
        // >= 16 bytes and `w` is exactly 32.
        return unsafe { unpack_block_q5_bits_avx2(qh, qs, bias, w) };
    }
    unpack_block_q5_bits(qh, qs, bias, w)
}

/// `ORANGU_Q5_UNPACK_VECTOR=0` keeps the scalar bit loop for the `Q5_0`,
/// `Q5_1` and `Q4_0` unpack on x86-64 — the control arm for the AVX2 form.
/// On unless `0`; read once, so the branch it guards costs a cached load
/// per block.
#[cfg(target_arch = "x86_64")]
fn q5_vector_unpack_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::engine::env::flag_on_unless_disabled("ORANGU_Q5_UNPACK_VECTOR"))
}

/// The portable form of the `Q5` unpack, and the reference the vector forms
/// are tested against.
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn unpack_block_q5_bits_scalar(qh: u32, qs: &[u8], bias: i8, w: &mut [i8; 32]) {
    for j in 0..16 {
        let hi_lo = ((qh >> j) << 4) & 0x10;
        let hi_hi = (qh >> (j + 12)) & 0x10;
        w[j] = (((qs[j] & 0x0F) as u32 | hi_lo) as i32 - bias as i32) as i8;
        w[16 + j] = (((qs[j] >> 4) as u32 | hi_hi) as i32 - bias as i32) as i8;
    }
}

/// The AVX2 form: the scalar loop's per-lane `qh >> j` becomes one byte
/// shuffle and one compare. `qh` is broadcast to every dword, a shuffle
/// gives lane `j` the byte holding its bit (`j / 8`, in each 128-bit half
/// from that half's own two bytes), a compare against the lane's own bit
/// mask turns the bit into `0xFF` or `0`, and masking that to `0x10` is the
/// fifth bit — thirty-two lanes in eight instructions.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn unpack_block_q5_bits_avx2(qh: u32, qs: &[u8], bias: i8, w: &mut [i8; 32]) {
    use std::arch::x86_64::*;
    debug_assert!(qs.len() >= 16);
    unsafe {
        let v = _mm_loadu_si128(qs.as_ptr() as *const __m128i);
        let nibble = _mm_set1_epi8(0x0F);
        let lo = _mm_and_si128(v, nibble);
        let hi = _mm_and_si128(_mm_srli_epi16::<4>(v), nibble);
        // Elements 0..16 are the low nibbles, 16..32 the high — the two
        // halves of one 256-bit vector.
        let nibbles = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi);
        let bits = _mm256_set1_epi32(qh as i32);
        // Lane `j` reads byte `j / 8` of `qh`: bytes 0 and 1 in the low
        // half, 2 and 3 in the high half (`shuffle_epi8` indexes within its
        // own 128-bit half, where the broadcast has put every byte).
        let byte_of_lane = _mm256_setr_epi8(
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
            3, 3, 3,
        );
        let bytes = _mm256_shuffle_epi8(bits, byte_of_lane);
        let bit_of_lane = _mm256_setr_epi8(
            1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64,
            -128, 1, 2, 4, 8, 16, 32, 64, -128,
        );
        let set = _mm256_cmpeq_epi8(_mm256_and_si256(bytes, bit_of_lane), bit_of_lane);
        let fifth = _mm256_and_si256(set, _mm256_set1_epi8(0x10));
        let out = _mm256_sub_epi8(_mm256_or_si256(nibbles, fifth), _mm256_set1_epi8(bias));
        _mm256_storeu_si256(w.as_mut_ptr() as *mut __m256i, out);
    }
}

/// `block_q4_0`: `{ d: f16, qs: [u8; 16] }`, 32 elements — the same nibble
/// layout as `Q5_0` with no `qh` and a `-8` zero point folded into the
/// `int8`, so it is symmetric and carries no `min` term.
#[inline(always)]
fn unpack_block_q4_0(qs: &[u8], w: &mut [i8; 32]) {
    unpack_block_q5_bits(0, qs, 8, w);
}

#[inline(always)]
fn unpack_block_q4_0_isa<const ISA: u8>(qs: &[u8], w: &mut [i8; 32]) {
    unpack_block_q5_bits_isa::<ISA>(0, qs, 8, w);
}

/// `block_q5_1`: `{ d: f16, m: f16, qh: [u8; 4], qs: [u8; 16] }`, 32
/// elements. Identical bit extraction to `Q5_0`, but the value is
/// `d*q + m` rather than `d*(q - 16)` — asymmetric, so `m` survives as a
/// correction term rather than folding into the weight.
#[inline(always)]
fn unpack_block_q5_1(qh: u32, qs: &[u8], w: &mut [i8; 32]) {
    unpack_block_q5_bits(qh, qs, 0, w);
}

#[inline(always)]
fn unpack_block_q5_1_isa<const ISA: u8>(qh: u32, qs: &[u8], w: &mut [i8; 32]) {
    unpack_block_q5_bits_isa::<ISA>(qh, qs, 0, w);
}

#[inline(always)]
fn unpack_block_q5_0(qh: u32, qs: &[u8], w: &mut [i8; 32]) {
    unpack_block_q5_bits(qh, qs, 16, w);
}

#[inline(always)]
fn unpack_block_q5_0_isa<const ISA: u8>(qh: u32, qs: &[u8], w: &mut [i8; 32]) {
    unpack_block_q5_bits_isa::<ISA>(qh, qs, 16, w);
}

/// The scalar loop's per-lane `qh >> j` is what defeats autovectorization
/// here. NEON does it with one variable-shift: `vshlq_u8` shifts right when
/// the amount is negative, so lane `j` can receive bit `j` of a broadcast
/// `qh` byte.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn unpack_block_q5_bits_neon(qh: u32, qs: &[u8], bias_v: i8, w: &mut [i8; 32]) {
    use std::arch::aarch64::*;
    const SH: [i8; 16] = [0, -1, -2, -3, -4, -5, -6, -7, 0, -1, -2, -3, -4, -5, -6, -7];
    unsafe {
        let sh = vld1q_s8(SH.as_ptr());
        let one = vdupq_n_u8(1);
        let bias = vdupq_n_s8(bias_v);
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

// =====================================================================
// `Q2_K` and `Q3_K`: the two-bit family.
//
// Both store their quants two bits at a time in a 64-byte `qs` array that is
// read **four times over** at shifts 0/2/4/6, and both carry one scale per
// **16** elements. `Q3_K` adds a third bit from a separate `hmask` plane;
// `Q2_K` adds a per-16 min instead.
//
// `quant::dequantize_q{2,3}_k` walk that as a `(q_off, shift, half)` triple.
// Rewritten as eight 32-element runs it collapses: run `p` covers elements
// `p*32 .. p*32+32` — exactly activation block `p` — reading the contiguous
// `qs[(p/4)*32 ..][..32]` at shift `2*(p%4)`, under scale sub-blocks `2p` and
// `2p+1`. For `Q3_K` the `hmask` bit works out to precisely `1 << p`.
//
// That equivalence is not obvious from the reference loops and is what lets
// both types reuse the existing per-`GROUP` machinery unchanged, so it is
// checked directly against `quant::dequantize` in the tests below.

/// `(qs byte offset, low-bit shift)` for each of the eight 32-element runs of
/// a `Q2_K`/`Q3_K` super-block, in element order. `Q3_K`'s `hmask` bit for run
/// `p` is `1 << p`.
const Q2K_RUNS: [(usize, u32); 8] = [
    (0, 0),
    (0, 2),
    (0, 4),
    (0, 6),
    (32, 0),
    (32, 2),
    (32, 4),
    (32, 6),
];

/// `Q2_K` bytes per super-block: `scales[16] + qs[64] + d + dmin`.
const Q2K_BLOCK_BYTES: usize = 16 + 64 + 2 + 2;
/// `Q3_K` bytes per super-block: `hmask[32] + qs[64] + scales[12] + d`.
const Q3K_BLOCK_BYTES: usize = 32 + 64 + 12 + 2;

/// One `Q2_K` run: 32 unsigned two-bit quants, `0..=3`. The per-16 min that
/// makes the type asymmetric lives in the scales, not here.
#[inline(always)]
fn unpack_q2k_run(qs: &[u8], shift: u32, w: &mut [i8; 32]) {
    for (j, slot) in w.iter_mut().enumerate() {
        *slot = ((qs[j] >> shift) & 3) as i8;
    }
}

/// One `Q3_K` run: 32 three-bit quants biased into `-4..=3`.
///
/// The third bit is stored **inverted** — a set `hmask` bit means "do *not*
/// subtract 4" — which is why the bias applies when the bit is clear. Getting
/// that backwards produces a model that still runs and still emits fluent
/// text, so it is covered by an exact round-trip test rather than by reading
/// the output.
#[inline(always)]
fn unpack_q3k_run(qs: &[u8], hmask: &[u8], shift: u32, bit: u8, w: &mut [i8; 32]) {
    for (j, slot) in w.iter_mut().enumerate() {
        let lo = ((qs[j] >> shift) & 3) as i8;
        *slot = lo - if hmask[j] & bit != 0 { 0 } else { 4 };
    }
}

/// `Q2_K` into the per-[`GROUP`] [`UnpackedRow`] shape: `scale[g] * q - min[g]`
/// with `scale = d * (sc & 0xF)` and `min = dmin * (sc >> 4)`.
///
/// The only asymmetric type here besides `Q4_K`, and the only one that is
/// asymmetric *per 16* — which is exactly why it gets no `supports_k` arm; see
/// [`supports_k`].
fn unpack_q2_k(row: &[u8], out: &mut UnpackedRow) {
    out.has_min = true;
    for (s, block) in row.as_chunks::<Q2K_BLOCK_BYTES>().0.iter().enumerate() {
        let scales = &block[0..16];
        let qs = &block[16..80];
        let d = read_f16(block, 80);
        let dmin = read_f16(block, 82);
        let base = s * SUPER_BLOCK;
        for (p, &(off, shift)) in Q2K_RUNS.iter().enumerate() {
            let e0 = base + p * 32;
            let w: &mut [i8; 32] = (&mut out.q[e0..e0 + 32]).try_into().unwrap();
            unpack_q2k_run(&qs[off..off + 32], shift, w);
            let gi = e0 / GROUP;
            for half in 0..2 {
                let sc = scales[2 * p + half];
                out.scale[gi + half] = d * (sc & 0xF) as f32;
                out.min[gi + half] = dmin * (sc >> 4) as f32;
            }
        }
    }
}

/// `Q3_K` into the per-[`GROUP`] [`UnpackedRow`] shape. Symmetric — the `-4`
/// bias is folded into the `int8` weight — so `min` is zero throughout and the
/// dot loop's correction term vanishes.
fn unpack_q3_k(row: &[u8], out: &mut UnpackedRow) {
    out.has_min = false;
    for (s, block) in row.as_chunks::<Q3K_BLOCK_BYTES>().0.iter().enumerate() {
        let hmask = &block[0..32];
        let qs = &block[32..96];
        let sc = unpack_q3_k_scales(&block[96..108]);
        let d = read_f16(block, 108);
        let base = s * SUPER_BLOCK;
        for (p, &(off, shift)) in Q2K_RUNS.iter().enumerate() {
            let e0 = base + p * 32;
            let w: &mut [i8; 32] = (&mut out.q[e0..e0 + 32]).try_into().unwrap();
            unpack_q3k_run(&qs[off..off + 32], hmask, shift, 1 << p, w);
            let gi = e0 / GROUP;
            for half in 0..2 {
                out.scale[gi + half] = d * (sc[2 * p + half] - 32) as f32;
                out.min[gi + half] = 0.0;
            }
        }
    }
}

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
    for (b, block) in row.as_chunks::<BLOCK_BYTES>().0.iter().enumerate() {
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
    for (b, block) in row.as_chunks::<BLOCK_BYTES>().0.iter().enumerate() {
        let dw = read_f16(block, 0);
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        let w: &mut [i8; 32] = (&mut out.q[b * 32..b * 32 + 32]).try_into().unwrap();
        unpack_block_q5_0(qh, &block[6..], w);
        out.scale[b] = dw;
        out.min[b] = 0.0;
    }
}

// ---------------------------------------------------------------- Q4_0

/// `Q4_0` into the generic [`UnpackedRow`] form. Symmetric like `Q5_0` — the
/// `-8` zero point folds into the `int8` weight — so no `min` survives and it
/// also qualifies for the flat-activation GEMM.
fn unpack_q4_0(row: &[u8], out: &mut UnpackedRow) {
    const BLOCK_BYTES: usize = 2 + 16;
    out.has_min = false;
    for (b, block) in row.as_chunks::<BLOCK_BYTES>().0.iter().enumerate() {
        let dw = read_f16(block, 0);
        let w: &mut [i8; 32] = (&mut out.q[b * 32..b * 32 + 32]).try_into().unwrap();
        unpack_block_q4_0(&block[2..], w);
        out.scale[b] = dw;
        out.min[b] = 0.0;
    }
}

// ---------------------------------------------------------------- PQ2_0 / PTQ1_0

/// `PQ2_0` into the generic [`UnpackedRow`] form: the block's one scale
/// repeated across its four 32-groups, no `min`. Ternary values are already
/// `int8`-sized, so this is the bit unpack and nothing else.
fn unpack_pq2_0(row: &[u8], out: &mut UnpackedRow) {
    out.has_min = false;
    for (b, block) in row.as_chunks::<PQ2_0_BLOCK_BYTES>().0.iter().enumerate() {
        let dw = read_f16(block, 0);
        let w: &mut [i8; QK_PRISM] = (&mut out.q[b * QK_PRISM..(b + 1) * QK_PRISM])
            .try_into()
            .unwrap();
        unpack_block_pq2_0(&block[2..], w);
        prism_scales(out, b, dw);
    }
}

/// `PTQ1_0` into the generic [`UnpackedRow`] form — as [`unpack_pq2_0`],
/// with `TQ1_0`'s trit unpack and the scale read from the block's tail.
fn unpack_ptq1_0(row: &[u8], out: &mut UnpackedRow) {
    const QS: usize = PTQ1_0_BLOCK_BYTES - 4;
    out.has_min = false;
    for (b, block) in row.as_chunks::<PTQ1_0_BLOCK_BYTES>().0.iter().enumerate() {
        let dw = read_f16(block, QS + 2);
        let w: &mut [i8; QK_PRISM] = (&mut out.q[b * QK_PRISM..(b + 1) * QK_PRISM])
            .try_into()
            .unwrap();
        unpack_block_ptq1_0(&block[..QS], &block[QS..QS + 2], w);
        prism_scales(out, b, dw);
    }
}

/// One Prism block's scale, written to each of the four per-32 slots it
/// covers — what lets a per-128 type ride the per-32 kernels unchanged.
#[inline(always)]
fn prism_scales(out: &mut UnpackedRow, b: usize, dw: f32) {
    for s in 0..QK_PRISM / 32 {
        out.scale[b * (QK_PRISM / 32) + s] = dw;
        out.min[b * (QK_PRISM / 32) + s] = 0.0;
    }
}

// ---------------------------------------------------------------- Q5_1

/// `Q5_1` into the generic [`UnpackedRow`] form.
///
/// The one type here whose offset is **added**: ggml's value is `d*q + m`
/// while [`UnpackedRow`] is defined as `scale*q - min`, so `min` is `-m`.
/// Getting that sign wrong would not crash — it would bias every weight in
/// the tensor by `2m` and read as a mildly degraded model.
fn unpack_q5_1(row: &[u8], out: &mut UnpackedRow) {
    const BLOCK_BYTES: usize = 2 + 2 + 4 + 16;
    out.has_min = true;
    for (b, block) in row.as_chunks::<BLOCK_BYTES>().0.iter().enumerate() {
        let dw = read_f16(block, 0);
        let m = read_f16(block, 2);
        let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        let w: &mut [i8; 32] = (&mut out.q[b * 32..b * 32 + 32]).try_into().unwrap();
        unpack_block_q5_1(qh, &block[8..], w);
        out.scale[b] = dw;
        out.min[b] = -m;
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
    for (s, block) in row.as_chunks::<BLOCK_BYTES>().0.iter().enumerate() {
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
    for (b, block) in row.as_chunks::<BLOCK_BYTES>().0.iter().enumerate() {
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
    for block in row.as_chunks::<BLOCK_BYTES>().0 {
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

// ---------------------------------------------------------------- Q5_K

/// `block_q5_K`: `{ d: f16, dmin: f16, scales: [u8; 12], qh: [u8; 32],
/// qs: [u8; 128] }`, 256 elements.
///
/// Byte-for-byte [`unpack_q4_k`]'s block with a 32-byte high-bit plane
/// inserted before `qs` — the same relationship `Q5_0` has to `Q4_0`. The
/// packed 6-bit `sc`/`m` pairs are identical and go through the same
/// [`get_scale_min_k4`], so the value is `sc * (nibble + 16*hi) - dmin*m`
/// and the shape is unchanged: per-32 scale, per-32 min, asymmetric.
///
/// `qh[l]` holds one bit per 32-element sub-block, `l` indexing the *byte*
/// within the run exactly as `qs` does. Sub-block `sb` of the super-block
/// reads bit `sb % 8`: the low nibbles of run `g` take bit `2g`, the high
/// nibbles bit `2g + 1` (ggml's `u1`/`u2`, each shifted left by 2 per run).
///
/// The result is `0..=31`, which still fits `i8`, so nothing downstream
/// widens — this is why `Q5_K` needs no kernel of its own.
fn unpack_q5_k(row: &[u8], out: &mut UnpackedRow) {
    const BLOCK_BYTES: usize = 2 + 2 + 12 + 32 + 128;
    out.has_min = true;
    let mut sb = 0usize; // 32-element sub-block index across the whole row
    for block in row.as_chunks::<BLOCK_BYTES>().0 {
        let d = read_f16(block, 0);
        let dmin = read_f16(block, 2);
        let scales = &block[4..16];
        let qh = &block[16..48];
        let qs = &block[48..];
        for g in 0..4 {
            let bytes = &qs[g * 32..g * 32 + 32];
            for (half, shift) in [(0usize, 0u32), (1, 4)] {
                let (sc, m) = get_scale_min_k4(g * 2 + half, scales);
                let hi_mask = 1u8 << (g * 2 + half);
                let w = &mut out.q[(sb + half) * 32..(sb + half) * 32 + 32];
                for (j, &byte) in bytes.iter().enumerate() {
                    let hi = if qh[j] & hi_mask != 0 { 16 } else { 0 };
                    w[j] = (((byte >> shift) & 0x0F) + hi) as i8;
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
    for (sblk, block) in row.as_chunks::<BLOCK_BYTES>().0.iter().enumerate() {
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
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && tiled_dot_on() {
            // Safety: both features checked at runtime.
            return unsafe { dot_unpacked_multi_avx2_tiled(w, acts, out) };
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

/// `ORANGU_EXPERT_DOT_TILED=0` keeps the per-block reduction
/// ([`dot_unpacked_multi_avx2`]) — the control arm for the register-resident
/// tile below. On unless `0`.
#[cfg(target_arch = "x86_64")]
fn tiled_dot_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::engine::env::flag_on_unless_disabled("ORANGU_EXPERT_DOT_TILED"))
}

/// The AVX2 form of [`dot_unpacked_multi_impl`] with the reduction kept in
/// registers: the tiled loop above called a 32-wide dot per (block, token)
/// that ended in a **horizontal sum** — six shuffle-and-add instructions to
/// fold eight lanes into one integer — and then a scalar epilogue, so of the
/// dozen instructions a block cost per token, four were the multiply-adds.
/// A prefill of a routed-expert model spends 87% of ten cores here.
///
/// Here the weight block is widened **once** per tile of tokens, each
/// token's eight `i32` partial sums are converted to `f32` and folded into a
/// per-token vector accumulator with one FMA against the block's `d × scale`,
/// the asymmetric-min correction runs as a scalar beside it, and the eight
/// lanes are reduced **once per row** at the end. Same integer sums, so the
/// products are exact; the floating additions happen in a different order
/// from the single-token path, which is why the cross-check against it is a
/// tolerance rather than equality.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_unpacked_multi_avx2_tiled(w: &UnpackedRow, acts: &[ActQ8], out: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = acts.len();
    let mut t0 = 0;
    // SAFETY: every pointer below is within the lengths the unpack and the
    // quantizer established (`w.q` is `in_dim` bytes, each `a.q` too, the
    // scale and sum tables per block/group of it).
    unsafe {
        while t0 + TOKEN_TILE <= n {
            let mut accf = [_mm256_setzero_ps(); TOKEN_TILE];
            let mut mins = [0f32; TOKEN_TILE];
            if w.per32 {
                for b in 0..w.scale.len() {
                    let wp = w.q.as_ptr().add(b * 32);
                    let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp as *const _));
                    let w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(16) as *const _));
                    let sc = w.scale[b];
                    let mn = if w.has_min { w.min[b] } else { 0.0 };
                    for k in 0..TOKEN_TILE {
                        let a = &acts[t0 + k];
                        let xp = a.q.as_ptr().add(b * 32);
                        let x0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(xp as *const _));
                        let x1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(xp.add(16) as *const _));
                        let sum =
                            _mm256_add_epi32(_mm256_madd_epi16(w0, x0), _mm256_madd_epi16(w1, x1));
                        let d = a.d[b];
                        accf[k] = _mm256_fmadd_ps(
                            _mm256_cvtepi32_ps(sum),
                            _mm256_set1_ps(d * sc),
                            accf[k],
                        );
                        if w.has_min {
                            mins[k] += d * mn * block_sum(a, b) as f32;
                        }
                    }
                }
            } else {
                for g in 0..w.scale.len() {
                    let wp = w.q.as_ptr().add(g * GROUP);
                    let wv = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp as *const _));
                    let sc = w.scale[g];
                    let mn = if w.has_min { w.min[g] } else { 0.0 };
                    let b = g * GROUP / ACT_BLOCK;
                    for k in 0..TOKEN_TILE {
                        let a = &acts[t0 + k];
                        let xv = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                            a.q.as_ptr().add(g * GROUP) as *const _
                        ));
                        let sum = _mm256_madd_epi16(wv, xv);
                        let d = a.d[b];
                        accf[k] = _mm256_fmadd_ps(
                            _mm256_cvtepi32_ps(sum),
                            _mm256_set1_ps(d * sc),
                            accf[k],
                        );
                        if w.has_min {
                            mins[k] += d * mn * a.sums[g] as f32;
                        }
                    }
                }
            }
            for k in 0..TOKEN_TILE {
                out[t0 + k] = hsum_ps_avx2(accf[k]) - mins[k];
            }
            t0 += TOKEN_TILE;
        }
    }
    for t in t0..n {
        out[t] = dot_unpacked_impl::<ISA_AVX2>(w, &acts[t]);
    }
}

/// Horizontal sum of eight `f32` lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn hsum_ps_avx2(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0b01));
    _mm_cvtss_f32(s)
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
                // `Q2_K` is the first per-[`GROUP`] type with a min. While
                // `Q6_K` and `Q3_K` were the only ones, both symmetric, this
                // branch could drop the correction — and did, which produced a
                // model that emitted nothing but newlines. `act.sums` is
                // per-GROUP, so the term is indexed by `g`, not by `b`.
                if w0.has_min {
                    let (m0, m1) = (w0.min[g], w1.min[g]);
                    for k in 0..TOKEN_TILE {
                        let a = &acts[t0 + k];
                        let xq = &a.q[g * GROUP..];
                        let (d, gs) = (a.d[b], a.sums[g] as f32);
                        a0[k] += d * (s0 * dot16::<ISA>(q0, xq) as f32 - m0 * gs);
                        a1[k] += d * (s1 * dot16::<ISA>(q1, xq) as f32 - m1 * gs);
                    }
                } else {
                    for k in 0..TOKEN_TILE {
                        let a = &acts[t0 + k];
                        let xq = &a.q[g * GROUP..];
                        let d = a.d[b];
                        a0[k] += d * (s0 * dot16::<ISA>(q0, xq) as f32);
                        a1[k] += d * (s1 * dot16::<ISA>(q1, xq) as f32);
                    }
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
                // See the same branch in `dot_unpacked_pair_impl`: the min is
                // per-GROUP for `Q2_K`, so it is indexed by `g`. This used to
                // carry a `debug_assert!` that a per-16 asymmetric type would
                // need the branch extended; `Q2_K` is that type, and the
                // assertion never fired because release builds disable it.
                //
                // Split rather than folded into one expression with a zero
                // min: `Q3_K` and `Q6_K` come through here on every prefill,
                // and `- 0.0 * sums` would cost them a multiply and a subtract
                // per group for a term that is always zero.
                if w.has_min {
                    let mn = w.min[g];
                    for (k, slot) in acc.iter_mut().enumerate() {
                        let a = &acts[t0 + k];
                        let isum = dot16::<ISA>(wq, &a.q[g * GROUP..]);
                        *slot += a.d[b] * (sc * isum as f32 - mn * a.sums[g] as f32);
                    }
                } else {
                    for (k, slot) in acc.iter_mut().enumerate() {
                        let a = &acts[t0 + k];
                        // Parenthesised to match `dot_unpacked_impl`'s
                        // association exactly: `d * (sc * isum)`, not
                        // `(d * sc) * isum`. Same maths, different rounding —
                        // the tests compare the two paths for bit equality,
                        // which is what caught this.
                        let isum = dot16::<ISA>(wq, &a.q[g * GROUP..]);
                        *slot += a.d[b] * (sc * isum as f32);
                    }
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
/// Note `Q2_K` is deliberately absent while [`supports`] accepts it.
///
/// Its min is per **16**, and [`gemm_k_q4_k`]'s correction pass reads
/// [`ActQ8K::bsum`], which is per **32** — so a `Q2_K` arm would need a second
/// sum array built for every K-quant matmul, including the ones that would
/// never read it. Without an arm here it simply takes the generic
/// [`dot_unpacked_pair`] prefill path instead, which is still fused; what it
/// gives up is the per-256 integer accumulation, not the kernel. `Q2_K` is
/// 11.5% of a `Q2_K` file against `Q3_K`'s 77.2%, so that trade is measured
/// before it is paid for — see `doc/PERF-TINY.md`.
pub fn supports_k(ggml_type: u32, in_dim: usize) -> bool {
    // The Prism ternary types have no super-block of their own, but two of
    // their 128-blocks make one: `unpack_k_prism` turns the pair of `f16`
    // scales into one `f32` base and two 15-bit integer scales, which is
    // exactly the `IQ4_XS` shape (per-32 signed integer scale, no min).
    matches!(
        ggml_type,
        GGML_TYPE_Q3_K
            | GGML_TYPE_Q4_K
            | GGML_TYPE_Q5_K
            | GGML_TYPE_Q6_K
            | GGML_TYPE_IQ4_XS
            | GGML_TYPE_PQ2_0
            | GGML_TYPE_PTQ1_0
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
        // Same loop shape as `Q4_K` — per-32 integer scale and min — so it
        // shares the kind and therefore `dot_k_pair` unchanged.
        GGML_TYPE_Q5_K => {
            out.resize_for(in_dim, KKind::Q4K);
            unpack_k_q5_k(row, out);
        }
        // `Q3_K` *is* `Q6_K`'s shape — a signed integer scale per 16 with no
        // min — so it fills a [`KKind::Q6K`] row and takes `gemm_k_q6_k`
        // unchanged. The quants are narrower (`-4..=3` against `-32..=31`),
        // which only makes the accumulator bound looser.
        GGML_TYPE_Q3_K => {
            out.resize_for(in_dim, KKind::Q6K);
            unpack_k_q3_k(row, out);
        }
        GGML_TYPE_Q6_K => {
            out.resize_for(in_dim, KKind::Q6K);
            unpack_k_q6_k(row, out);
        }
        GGML_TYPE_IQ4_XS => {
            out.resize_for(in_dim, KKind::Iq4Xs);
            unpack_k_iq4_xs(row, out);
        }
        GGML_TYPE_PQ2_0 => {
            out.resize_for(in_dim, KKind::Iq4Xs);
            unpack_k_prism::<false>(row, out);
        }
        GGML_TYPE_PTQ1_0 => {
            out.resize_for(in_dim, KKind::Iq4Xs);
            unpack_k_prism::<true>(row, out);
        }
        other => panic!("vecdot::unpack_k_row called for unsupported ggml_type {other}"),
    }
}

/// The largest integer scale [`unpack_k_prism`] hands out. 15 bits rather
/// than `i16`'s 16 so the super-block accumulation stays inside `i32` at
/// the format's extreme: `16383 · |q| ≤ 2 · |x| ≤ 127 · 256` is 1.07e9,
/// under half of `i32::MAX`; a full 16 bits would be over it.
const PRISM_K_SCALE_MAX: i32 = 16383;

/// Unpacks a `PQ2_0` (`TRITS = false`) or `PTQ1_0` (`TRITS = true`) row into
/// [`KKind::Iq4Xs`] form: per super-block the larger of its two blocks'
/// `f16` scales, divided by [`PRISM_K_SCALE_MAX`], is the `f32` base, and
/// each block's scale is rounded to an integer multiple of it, written to
/// its four per-32 slots.
///
/// The rounding is the one place this path is not exact: a block's scale
/// lands within `1 / (2 · 16383)` of itself, a relative error of 3e-5 —
/// against the 1e-3 the int8 activation quantization already introduces,
/// and far tighter than the 2% the k-row test allows. What the coarser
/// activation scale buys is the reason the K GEMM exists: the whole
/// super-block accumulates in `i32` and converts once.
fn unpack_k_prism<const TRITS: bool>(row: &[u8], out: &mut KRow) {
    let block_bytes = if TRITS {
        PTQ1_0_BLOCK_BYTES
    } else {
        PQ2_0_BLOCK_BYTES
    };
    let per_super = SUPER_BLOCK / QK_PRISM;
    for (s, pair) in row.chunks_exact(per_super * block_bytes).enumerate() {
        let mut scales = [0f32; SUPER_BLOCK / QK_PRISM];
        for (h, block) in pair.chunks_exact(block_bytes).enumerate() {
            let w: &mut [i8; QK_PRISM] = (&mut out.q[(s * per_super + h) * QK_PRISM..][..QK_PRISM])
                .try_into()
                .unwrap();
            scales[h] = if TRITS {
                const QS: usize = PTQ1_0_BLOCK_BYTES - 4;
                unpack_block_ptq1_0(&block[..QS], &block[QS..QS + 2], w);
                read_f16(block, QS + 2)
            } else {
                unpack_block_pq2_0(&block[2..], w);
                read_f16(block, 0)
            };
        }
        let amax = scales.iter().fold(0f32, |m, v| m.max(v.abs()));
        let base = amax / PRISM_K_SCALE_MAX as f32;
        let inv = if base > 0.0 { 1.0 / base } else { 0.0 };
        out.d[s] = base;
        for (h, &scale) in scales.iter().enumerate() {
            let sc = (scale * inv).round() as i32;
            debug_assert!(sc.abs() <= PRISM_K_SCALE_MAX);
            for sub in 0..QK_PRISM / 32 {
                out.sc[s * SUBS + h * (QK_PRISM / 32) + sub] = sc as i16;
            }
        }
    }
}

/// Same bit layout as [`unpack_q4_k`], but `sc`/`m` stay integers and `d`/
/// `dmin` are kept once per super-block rather than multiplied in.
fn unpack_k_q4_k(row: &[u8], out: &mut KRow) {
    const BLOCK_BYTES: usize = 2 + 2 + 12 + 128;
    let mut sb = 0usize;
    for (s, block) in row.as_chunks::<BLOCK_BYTES>().0.iter().enumerate() {
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

/// Same bit layout as [`unpack_q5_k`], but `sc`/`m` stay integers and
/// `d`/`dmin` are kept once per super-block — i.e. [`unpack_k_q4_k`] with the
/// high-bit plane applied. Fills a [`KKind::Q4K`] row, because that *is* the
/// shape: per-32 integer scale, per-32 integer min, asymmetric.
fn unpack_k_q5_k(row: &[u8], out: &mut KRow) {
    const BLOCK_BYTES: usize = 2 + 2 + 12 + 32 + 128;
    let mut sb = 0usize;
    for (s, block) in row.as_chunks::<BLOCK_BYTES>().0.iter().enumerate() {
        out.d[s] = read_f16(block, 0);
        out.dmin[s] = read_f16(block, 2);
        let scales = &block[4..16];
        let qh = &block[16..48];
        let qs = &block[48..];
        for g in 0..4 {
            let bytes = &qs[g * 32..g * 32 + 32];
            for (half, shift) in [(0usize, 0u32), (1, 4)] {
                let (sc, m) = get_scale_min_k4(g * 2 + half, scales);
                let hi_mask = 1u8 << (g * 2 + half);
                let w = &mut out.q[(sb + half) * 32..(sb + half) * 32 + 32];
                for (j, &byte) in bytes.iter().enumerate() {
                    let hi = if qh[j] & hi_mask != 0 { 16 } else { 0 };
                    w[j] = (((byte >> shift) & 0x0F) + hi) as i8;
                }
                out.sc[sb + half] = sc as i16;
                out.mins[sb + half] = m as i16;
            }
            sb += 2;
        }
    }
}

/// Same bit layout as [`unpack_q3_k`], with the six-bit scales kept as
/// integers — `sc - 32`, i.e. `-32..=31`, matching what [`gemm_k_q6_k`]
/// expects of a [`KKind::Q6K`] row.
///
/// Accumulator bound, the thing that has to hold for the integer path to be
/// legal: `|sc| <= 32`, `|q| <= 4`, `|x| <= 127`, so a 16-element group
/// partial is at most `16*4*127 = 8128`, and 16 groups of 32 lanes reach
/// ~4.2e6 — two orders of magnitude inside `i32`, and looser than `Q6_K`'s
/// own, which the same kernel already carries.
fn unpack_k_q3_k(row: &[u8], out: &mut KRow) {
    for (s, block) in row.as_chunks::<Q3K_BLOCK_BYTES>().0.iter().enumerate() {
        let hmask = &block[0..32];
        let qs = &block[32..96];
        let sc = unpack_q3_k_scales(&block[96..108]);
        out.d[s] = read_f16(block, 108);
        let base = s * SUPER_BLOCK;
        for (p, &(off, shift)) in Q2K_RUNS.iter().enumerate() {
            let e0 = base + p * 32;
            let w: &mut [i8; 32] = (&mut out.q[e0..e0 + 32]).try_into().unwrap();
            unpack_q3k_run(&qs[off..off + 32], hmask, shift, 1 << p, w);
            let gi = e0 / GROUP;
            out.sc[gi] = (sc[2 * p] - 32) as i16;
            out.sc[gi + 1] = (sc[2 * p + 1] - 32) as i16;
        }
    }
}

/// Same bit layout as [`unpack_q6_k`], with the `i8` scales kept as integers.
fn unpack_k_q6_k(row: &[u8], out: &mut KRow) {
    const BLOCK_BYTES: usize = 128 + 64 + 16 + 2;
    for (s, block) in row.as_chunks::<BLOCK_BYTES>().0.iter().enumerate() {
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

// =====================================================================
// The `i8mm` prefill GEMM: four rows x four tokens per tile, `smmla`.
//
// `dot_k_pair` above is the right shape for a machine without `i8mm` and the
// wrong one for a machine with it. Its unit of work is a 32-element dot of
// one row against one token: two `sdot` and then a **horizontal reduction**
// (`addv`) to get a scalar out, followed by a scalar multiply by the
// sub-block's scale. Per (row, token, 32 elements) that is two useful
// instructions and three or four that only move data sideways, and the
// reduction serialises on every core. Profiled on a Qwen-Image transformer
// pass (256 tokens, `Q4_K`): 80% of all samples in that kernel, at about
// 80 G MAC/s across twelve cores — a tenth of what the cores can do.
//
// `smmla` (ARMv8.6 `i8mm`) multiplies a 2x8 `int8` matrix by an 8x2 one and
// accumulates the 2x2 result into four `i32` lanes — 32 multiply-adds per
// instruction, twice `sdot`'s, and it never leaves the vector unit: the
// per-sub-block scale becomes one vector multiply-accumulate (`mla`) into a
// running super-block sum, and the super-block's `f32` step is three vector
// ops. Nothing is reduced horizontally until the tile is stored.
//
// Tile: 4 weight rows x 4 tokens = two row pairs x two token pairs, four
// `smmla` accumulators. Per 8-element chunk that is 4 `smmla` against 8
// 64-bit loads (each vector is two rows' or two tokens' 8 bytes side by
// side). The activations stay in `ActQ8K`'s existing layout — a token's 32
// bytes contiguous — and the weights in `KRow`'s, so every other kernel is
// untouched; the pairing is done by the loads (`ld1 {v.d}[1]`), which costs
// nothing the loads were not already paying.
//
// **Bit-identical to `dot_k_pair`.** The integer sums are exact and the
// same; the `f32` steps are the same operations in the same order per lane
// (`acc += ad * (d * isum)`, multiply, multiply, add — Rust never contracts
// to an FMA), and the `Q4_K` min correction runs the same scalar code first.
// The test asserts equality, not tolerance.
// =====================================================================

/// Rows per [`dot_k_rows`] quad — the `smmla` tile height.
pub const ROW_QUAD: usize = 4;

/// Whether [`dot_k_rows`] may be used: the CPU has `i8mm`, `smmla` agrees
/// with a scalar 2x2 product on known inputs (the same trust-but-verify
/// contract as [`have_dotprod`] — the instruction is inline assembly and
/// was written on a machine that has it, but a constraint mistake must
/// degrade to `sdot`, not corrupt a picture), and `ORANGU_EXPERT_I8MM` is
/// not set to `0`/`off` — the switch that lets a before/after be measured on
/// one binary.
#[cfg(target_arch = "aarch64")]
pub fn have_i8mm() -> bool {
    use std::sync::OnceLock;
    static USABLE: OnceLock<bool> = OnceLock::new();
    *USABLE.get_or_init(|| {
        if !std::arch::is_aarch64_feature_detected!("i8mm") {
            return false;
        }
        if !crate::engine::env::flag_on_unless_disabled("ORANGU_EXPERT_I8MM") {
            return false;
        }
        // Two 2x8 matrices with signs, extremes and a mix, checked against
        // the scalar definition of the 2x2 product.
        const A: [i8; 16] = [
            3, -5, 7, -11, 13, -17, 19, -23, 127, -128, 1, -1, 0, 64, -64, 32,
        ];
        const B: [i8; 16] = [
            2, 4, -6, 8, -10, 12, 14, -16, -128, 127, 29, -31, 37, -41, 43, -47,
        ];
        let mut expect = [0i32; 4];
        for (i, row) in A.as_chunks::<8>().0.iter().enumerate() {
            for (j, col) in B.as_chunks::<8>().0.iter().enumerate() {
                expect[i * 2 + j] = row
                    .iter()
                    .zip(col)
                    .map(|(&a, &b)| a as i32 * b as i32)
                    .sum();
            }
        }
        // Safety: `i8mm` was just detected; both arrays are 16 bytes.
        let got = unsafe {
            use std::arch::aarch64::*;
            let acc = mmla_2x8(vdupq_n_s32(0), vld1q_s8(A.as_ptr()), vld1q_s8(B.as_ptr()));
            let mut out = [0i32; 4];
            vst1q_s32(out.as_mut_ptr(), acc);
            out
        };
        got == expect
    })
}

#[cfg(not(target_arch = "aarch64"))]
pub fn have_i8mm() -> bool {
    false
}

/// `smmla`: `acc += a (2x8, rows in each 8-byte half) . b^T (2x8, rows in
/// each half)`, lanes `[a0.b0, a0.b1, a1.b0, a1.b1]`.
///
/// Inline assembly for the same reason as [`dot16_sdot`]: the intrinsic is
/// unstable, the encoding is fixed, and `asm!` inlines where a
/// `#[target_feature]` function would not. [`have_i8mm`] verifies the
/// operand order at startup.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn mmla_2x8(
    acc: std::arch::aarch64::int32x4_t,
    a: std::arch::aarch64::int8x16_t,
    b: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    let mut acc = acc;
    unsafe {
        std::arch::asm!(
            ".arch_extension i8mm",
            "smmla {acc:v}.4s, {a:v}.16b, {b:v}.16b",
            acc = inout(vreg) acc,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack)
        );
    }
    acc
}

/// `int8` rows laid out for `smmla`, two at a time: for each 8-element chunk
/// the pair's two rows sit side by side (16 bytes), which is the operand
/// `smmla` takes. Each row keeps its own scale (`max |x| / 127`). An odd
/// row count is padded with a zero row. What the picture transformers'
/// attention scores are computed from — see `image::transformer`.
pub struct PairedI8 {
    pub data: Vec<i8>,
    pub scales: Vec<f32>,
    pub dim: usize,
    pub pairs: usize,
}

impl PairedI8 {
    /// `n` rows of `dim` (a multiple of 8), row `i` read through `row(i)`.
    pub fn quantize<'a>(n: usize, dim: usize, row: impl Fn(usize) -> &'a [f32]) -> Self {
        debug_assert!(dim.is_multiple_of(8));
        let pairs = n.div_ceil(2);
        let mut data = vec![0i8; pairs * 2 * dim];
        let mut scales = vec![0f32; pairs * 2];
        for (i, slot) in scales.iter_mut().enumerate().take(n) {
            let r = &row(i)[..dim];
            let max = r.iter().fold(0f32, |m, v| m.max(v.abs()));
            let scale = max / 127.0;
            let inv = if max > 0.0 { 1.0 / scale } else { 0.0 };
            *slot = scale;
            let (pair, half) = (i / 2, i % 2);
            let base = pair * 2 * dim;
            for (c, chunk) in r.as_chunks::<8>().0.iter().enumerate() {
                let dst = &mut data[base + c * 16 + half * 8..][..8];
                for (d, v) in dst.iter_mut().zip(chunk) {
                    *d = (v * inv).round().clamp(-127.0, 127.0) as i8;
                }
            }
        }
        Self {
            data,
            scales,
            dim,
            pairs,
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn pair(&self, p: usize) -> &[i8] {
        &self.data[p * 2 * self.dim..(p + 1) * 2 * self.dim]
    }
}

/// Integer dot products of four query rows — query pairs `qa` and `qb` of
/// `q` — against every key row of `k`: `out[r][j] = q_r · k_j` as `i32`
/// (`out` rows at least `2 * k.pairs` long). Needs [`have_i8mm`] on
/// `aarch64`; elsewhere, or without it, the scalar definition.
pub fn i8_scores_4rows(q: &PairedI8, qa: usize, qb: usize, k: &PairedI8, out: [&mut [i32]; 4]) {
    i8_scores_4rows_in(q, qa, qb, k, 0..k.pairs, out)
}

/// [`i8_scores_4rows`] against key pairs `pairs` of `k` only: `out[r][j]`
/// is the score against key `2 * pairs.start + j`. What a query block whose
/// windows cover part of the keys asks for — a sliding-window layer's
/// prompt attention (`attention_mixed`).
pub fn i8_scores_4rows_in(
    q: &PairedI8,
    qa: usize,
    qb: usize,
    k: &PairedI8,
    pairs: std::ops::Range<usize>,
    out: [&mut [i32]; 4],
) {
    debug_assert_eq!(q.dim, k.dim);
    debug_assert!(pairs.end <= k.pairs);
    let dim = q.dim;
    let p0 = pairs.start;
    let n_pairs = pairs.len();
    #[cfg(target_arch = "aarch64")]
    if have_i8mm() {
        use std::arch::aarch64::*;
        let (a, b) = (q.pair(qa), q.pair(qb));
        let [o0, o1, o2, o3] = out;
        // Four key pairs at a time: eight independent `smmla` chains (two
        // ran at the instruction's latency, sixteen deep), the query rows
        // loaded once for all four, and each row's eight scores stored as
        // two vectors — a 64-bit zip of the pairs' lanes.
        let quads = n_pairs / 4;
        for p4 in 0..quads {
            // Safety: `i8mm` was verified; every load is 16 bytes inside a
            // `2 * dim` pair, `dim` a multiple of 8; the stores write
            // `o*[8 * p4..8 * p4 + 8]`, inside `2 * n_pairs`.
            unsafe {
                let kp: [*const i8; 4] = std::array::from_fn(|j| k.pair(p0 + 4 * p4 + j).as_ptr());
                let mut x = [vdupq_n_s32(0); 4];
                let mut y = [vdupq_n_s32(0); 4];
                let mut c = 0;
                while c < 2 * dim {
                    let av = vld1q_s8(a.as_ptr().add(c));
                    let bv = vld1q_s8(b.as_ptr().add(c));
                    for j in 0..4 {
                        let kv = vld1q_s8(kp[j].add(c));
                        x[j] = mmla_2x8(x[j], av, kv);
                        y[j] = mmla_2x8(y[j], bv, kv);
                    }
                    c += 16;
                }
                // Lanes per pair: [row0.key0, row0.key1, row1.key0, row1.key1].
                let rows = |v: &[int32x4_t; 4], lo: bool| -> [int32x4_t; 2] {
                    let z = |p: int32x4_t, q: int32x4_t| {
                        let (p, q) = (vreinterpretq_s64_s32(p), vreinterpretq_s64_s32(q));
                        vreinterpretq_s32_s64(if lo {
                            vzip1q_s64(p, q)
                        } else {
                            vzip2q_s64(p, q)
                        })
                    };
                    [z(v[0], v[1]), z(v[2], v[3])]
                };
                for (o, v, lo) in [
                    (&mut *o0, &x, true),
                    (&mut *o1, &x, false),
                    (&mut *o2, &y, true),
                    (&mut *o3, &y, false),
                ] {
                    let [h0, h1] = rows(v, lo);
                    vst1q_s32(o.as_mut_ptr().add(8 * p4), h0);
                    vst1q_s32(o.as_mut_ptr().add(8 * p4 + 4), h1);
                }
            }
        }
        for p in 4 * quads..n_pairs {
            let kp = k.pair(p0 + p);
            // Safety: `i8mm` was verified; every load is 16 bytes inside a
            // `2 * dim` pair, `dim` a multiple of 8.
            let (x, y) = unsafe {
                let mut x = vdupq_n_s32(0);
                let mut y = vdupq_n_s32(0);
                let mut c = 0;
                while c < 2 * dim {
                    let kv = vld1q_s8(kp.as_ptr().add(c));
                    x = mmla_2x8(x, vld1q_s8(a.as_ptr().add(c)), kv);
                    y = mmla_2x8(y, vld1q_s8(b.as_ptr().add(c)), kv);
                    c += 16;
                }
                let mut xs = [0i32; 4];
                let mut ys = [0i32; 4];
                vst1q_s32(xs.as_mut_ptr(), x);
                vst1q_s32(ys.as_mut_ptr(), y);
                (xs, ys)
            };
            // Lanes: [row0.key0, row0.key1, row1.key0, row1.key1].
            o0[2 * p] = x[0];
            o0[2 * p + 1] = x[1];
            o1[2 * p] = x[2];
            o1[2 * p + 1] = x[3];
            o2[2 * p] = y[0];
            o2[2 * p + 1] = y[1];
            o3[2 * p] = y[2];
            o3[2 * p + 1] = y[3];
        }
        return;
    }
    let unpaired = |m: &PairedI8, row: usize, c: usize| -> i32 {
        m.data[(row / 2) * 2 * dim + (c / 8) * 16 + (row % 2) * 8 + c % 8] as i32
    };
    let rows = [2 * qa, 2 * qa + 1, 2 * qb, 2 * qb + 1];
    for (o, &r) in out.into_iter().zip(&rows) {
        for (j, slot) in o.iter_mut().enumerate().take(2 * n_pairs) {
            *slot = (0..dim)
                .map(|c| unpaired(q, r, c) * unpaired(k, 2 * p0 + j, c))
                .sum();
        }
    }
}

/// How many rows [`dot_k_rows`] is handed per rayon task — a multiple of
/// [`ROW_QUAD`]. More rows per task means each activation tile is read from
/// L2 once and reused across more row quads while it sits in L1; fewer
/// means less unpacked weight per task competing for that same L1.
/// `ORANGU_EXPERT_K_ROWS` overrides the default for a measurement; it is
/// rounded down to a multiple of four and floored at four.
pub fn k_rows_per_task() -> usize {
    static ROWS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ROWS.get_or_init(|| {
        let rows = std::env::var("ORANGU_EXPERT_K_ROWS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(K_ROWS_PER_TASK_DEFAULT);
        (rows / ROW_QUAD * ROW_QUAD).max(ROW_QUAD)
    })
}

/// The default for [`k_rows_per_task`]: measured on the board this was
/// written on (see `doc/PERF-IMAGE.md`).
const K_ROWS_PER_TASK_DEFAULT: usize = 16;

/// The widest `int8` matmul kernel this process will actually dispatch to,
/// for the startup `[cpu]` line — the answer to "what can this machine
/// use", as opposed to what it advertises: each name here is behind the
/// same runtime check and self-test that gates the kernel itself, so a
/// core that claims a feature whose instruction misbehaves is reported as
/// the level it really runs at.
pub fn int8_kernel_label() -> &'static str {
    #[cfg(target_arch = "aarch64")]
    {
        if have_i8mm() {
            return "i8mm (smmla)";
        }
        if have_dotprod() {
            return "dotprod (sdot)";
        }
        return "NEON";
    }
    #[cfg(target_arch = "x86_64")]
    {
        if have_vnni() {
            return "AVX-512 VNNI";
        }
        if is_x86_feature_detected!("avx2") {
            return "AVX2";
        }
        if is_x86_feature_detected!("sse4.1") {
            return "SSE4.1";
        }
        return "scalar";
    }
    #[allow(unreachable_code)]
    "scalar"
}

/// Tokens per [`ActQ8Mm`] tile: four token pairs, so one weight vector is
/// reused across eight tokens — 6 loads per 8 `smmla` in the inner loop.
pub const MM_TILE: usize = 8;

/// Activations quantized like [`ActQ8K`] — `int8`, one `f32` scale per 256,
/// per-32 sums for `Q4_K`'s min term — but laid out for `smmla`: in tiles of
/// [`MM_TILE`] tokens, and within each 32-element block as
/// `[token pair][8-element chunk][the pair's two tokens][8]`, so the 16 bytes
/// an `smmla` wants (two tokens' same 8 elements) are one aligned load.
///
/// A separate type rather than a flag on [`ActQ8K`], so the kernels that
/// read the other layout cannot be handed this one by mistake.
pub struct ActQ8Mm {
    n_tokens: usize,
    n_block: usize,
    n_super: usize,
    /// `n_tokens.div_ceil(MM_TILE)`
    n_tile: usize,
    /// `[tile][block][pair][chunk][2][8]`
    q: Vec<i8>,
    /// `[tile][super][token]`
    d: Vec<f32>,
    /// `[tile][super][token][8]` — `sum(q)` per 32-element sub-block.
    bsum: Vec<i16>,
}

impl ActQ8Mm {
    /// `x` is `[n_tokens][in_dim]`, token-major; `in_dim` a multiple of
    /// [`SUPER_BLOCK`]. The padding tokens of the last tile stay zero and
    /// contribute nothing; their outputs are never read.
    pub fn quantize(x: &[f32], in_dim: usize, n_tokens: usize) -> Self {
        debug_assert_eq!(x.len(), n_tokens * in_dim);
        Self::build(
            in_dim,
            n_tokens,
            |t| &x[t * in_dim..(t + 1) * in_dim],
            |_, _| {},
        )
    }

    /// [`Self::quantize`] over rows produced on demand: `fill(t, row)`
    /// writes token `t`'s `in_dim` values into a scratch row that is
    /// quantized straight away. What lets a convolution's `im2col` be
    /// gathered into a 4 KiB buffer per token and never exist as a whole
    /// — the full-resolution band of a VAE decode was 268 MiB of `f32`
    /// written and read back for every convolution.
    pub fn quantize_with(
        in_dim: usize,
        n_tokens: usize,
        fill: impl Fn(usize, &mut [f32]) + Sync,
    ) -> Self {
        Self::build(in_dim, n_tokens, |_| &[], fill)
    }

    /// The tile loop behind both: `direct(t)` is token `t`'s row when it
    /// already exists (empty otherwise), `fill(t, scratch)` produces it
    /// when it does not.
    fn build<'a>(
        in_dim: usize,
        n_tokens: usize,
        direct: impl Fn(usize) -> &'a [f32] + Sync,
        fill: impl Fn(usize, &mut [f32]) + Sync,
    ) -> Self {
        debug_assert_eq!(in_dim % SUPER_BLOCK, 0);
        let n_block = in_dim / 32;
        let n_super = in_dim / SUPER_BLOCK;
        let n_tile = n_tokens.div_ceil(MM_TILE);
        let mut q = vec![0i8; n_tile * n_block * MM_TILE * 32];
        let mut d = vec![0f32; n_tile * n_super * MM_TILE];
        let mut bsum = vec![0i16; n_tile * n_super * MM_TILE * SUBS];
        // One rayon task per tile: a tile's eight tokens own a contiguous
        // slice of each of the three arrays, so the tiles quantize in
        // parallel with no sharing. Serial, this was a quarter of a
        // 256-token matmul's wall time while eleven threads slept.
        q.par_chunks_mut(n_block * MM_TILE * 32)
            .zip(d.par_chunks_mut(n_super * MM_TILE))
            .zip(bsum.par_chunks_mut(n_super * MM_TILE * SUBS))
            .enumerate()
            .for_each_init(Vec::new, |scratch: &mut Vec<f32>, (tl, ((q, d), bsum))| {
                for k in 0..MM_TILE {
                    let t = tl * MM_TILE + k;
                    if t >= n_tokens {
                        break;
                    }
                    let (tp, half) = (k / 2, k % 2);
                    let existing = direct(t);
                    let row: &[f32] = if existing.is_empty() {
                        scratch.resize(in_dim, 0.0);
                        fill(t, scratch);
                        scratch
                    } else {
                        existing
                    };
                    for s in 0..n_super {
                        let chunk = &row[s * SUPER_BLOCK..(s + 1) * SUPER_BLOCK];
                        let amax = abs_max(chunk);
                        let scale = amax / 127.0;
                        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
                        d[s * MM_TILE + k] = scale;
                        for j in 0..SUBS {
                            let b = s * SUBS + j;
                            let src = &chunk[j * 32..(j + 1) * 32];
                            let block = &mut q[b * MM_TILE * 32..][..MM_TILE * 32];
                            let mut sum = 0i32;
                            for c in 0..4 {
                                let dst: &mut [i8; 8] = (&mut block
                                    [((tp * 4 + c) * 2 + half) * 8..][..8])
                                    .try_into()
                                    .unwrap();
                                sum += quantize_8(&src[c * 8..c * 8 + 8], inv, dst);
                            }
                            bsum[(s * MM_TILE + k) * SUBS + j] = sum as i16;
                        }
                    }
                }
            });
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

/// The largest magnitude in `x`, four lanes at a time on NEON.
#[inline]
fn abs_max(x: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    // Safety: NEON is baseline on aarch64; the loop stays within `x`.
    unsafe {
        use std::arch::aarch64::*;
        let mut m = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + 4 <= x.len() {
            m = vmaxq_f32(m, vabsq_f32(vld1q_f32(x.as_ptr().add(i))));
            i += 4;
        }
        let mut max = vmaxvq_f32(m);
        for v in &x[i..] {
            max = max.max(v.abs());
        }
        max
    }
    #[cfg(not(target_arch = "aarch64"))]
    x.iter().fold(0f32, |m, v| m.max(v.abs()))
}

/// Eight values times `inv`, rounded half away from zero and clamped to
/// `±127`, into `dst`; returns their sum. The same rounding as
/// `f32::round` — `vcvtaq_s32_f32` is "round to nearest, ties away" — so
/// the vector and scalar forms produce identical bytes. The quantizer was
/// a tenth of a VAE decode's samples as a scalar loop with a computed
/// scatter per element.
#[inline]
fn quantize_8(src: &[f32], inv: f32, dst: &mut [i8; 8]) -> i32 {
    debug_assert_eq!(src.len(), 8);
    #[cfg(target_arch = "aarch64")]
    // Safety: NEON is baseline on aarch64; `src` is eight floats.
    unsafe {
        use std::arch::aarch64::*;
        let invv = vdupq_n_f32(inv);
        let lo = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(src.as_ptr()), invv));
        let hi = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(src.as_ptr().add(4)), invv));
        let limit = vdupq_n_s32(127);
        let lo = vmaxq_s32(vminq_s32(lo, limit), vnegq_s32(limit));
        let hi = vmaxq_s32(vminq_s32(hi, limit), vnegq_s32(limit));
        let narrow = vmovn_s16(vcombine_s16(vmovn_s32(lo), vmovn_s32(hi)));
        vst1_s8(dst.as_mut_ptr(), narrow);
        vaddvq_s32(vaddq_s32(lo, hi))
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut sum = 0i32;
        for (slot, &v) in dst.iter_mut().zip(src) {
            let qi = (v * inv).round().clamp(-127.0, 127.0) as i8;
            *slot = qi;
            sum += qi as i32;
        }
        sum
    }
}

/// `w.len()` rows — a multiple of [`ROW_QUAD`] — against every token: the
/// `i8mm` prefill kernel. `out[i]` is row `i`'s `n_tokens`-long output.
/// Only callable when [`have_i8mm`] is true.
///
/// Same arithmetic as [`dot_k_pair`], to the bit: the integer sums are
/// exact and identical, the `f32` steps are the same operations in the same
/// order per (row, token), and `Q4_K`'s min correction is applied first as
/// there. Only the register shape differs.
#[cfg_attr(not(test), allow(dead_code))]
pub fn dot_k_rows(w: &[&KRow], a: &ActQ8Mm, out: &mut [&mut [f32]]) {
    dot_k_rows_tiles(w, a, 0..a.n_tile, out)
}

/// [`dot_k_rows`] over the token tiles `tiles` only — `out[i]` is still
/// row `i`'s full `n_tokens`-long output, of which the tiles' tokens are
/// written. What lets a wide, short matmul (a VAE convolution: 96 rows
/// against 65,536 pixels) be split over tokens as well as rows.
#[cfg(target_arch = "aarch64")]
pub fn dot_k_rows_tiles(
    w: &[&KRow],
    a: &ActQ8Mm,
    tiles: std::ops::Range<usize>,
    out: &mut [&mut [f32]],
) {
    debug_assert!(!w.is_empty() && w.len().is_multiple_of(ROW_QUAD));
    debug_assert_eq!(w.len(), out.len());
    debug_assert!(w.iter().all(|r| r.kind == w[0].kind));
    debug_assert!(out.iter().all(|o| o.len() == a.n_tokens));
    debug_assert!(tiles.end <= a.n_tile);
    debug_assert!(have_i8mm());
    // Safety: `have_i8mm` verified the instruction and the caller's
    // contract requires it.
    unsafe {
        match w[0].kind {
            KKind::Q4K => gemm_k_rows_mmla::<true, false>(w, a, tiles, out),
            KKind::Iq4Xs => gemm_k_rows_mmla::<false, false>(w, a, tiles, out),
            KKind::Q6K => gemm_k_rows_mmla::<false, true>(w, a, tiles, out),
        }
    }
}

/// [`dot_k_rows_tiles`] off `aarch64`: the same product over the same
/// [`ActQ8Mm`] layout in plain integer arithmetic — [`dot_k_rows_portable`].
/// The callers prefer the pair kernels where [`have_i8mm`] is false (their
/// `AVX2`/`VNNI` forms are the tuned path there), so this is the complete
/// definition rather than the fast one.
#[cfg(not(target_arch = "aarch64"))]
pub fn dot_k_rows_tiles(
    w: &[&KRow],
    a: &ActQ8Mm,
    tiles: std::ops::Range<usize>,
    out: &mut [&mut [f32]],
) {
    dot_k_rows_portable(w, a, tiles, out)
}

/// The `i8mm` tile product in portable Rust: every row of `w` against
/// every token of the tiles, `i32` sums per scale group folded by the
/// group's integer scale, then `acc += ad * (d * isum)` per super-block —
/// the pair kernel's expression, operation for operation, so the result is
/// bit-identical to [`dot_k_pair`]'s and to `smmla`'s (the test on an
/// `i8mm` machine asserts equality). Written once for every architecture:
/// what a machine without `i8mm` runs when handed this layout, and what
/// the `smmla` kernel is checked against.
// On `aarch64` the check is its only caller.
#[cfg_attr(all(target_arch = "aarch64", not(test)), allow(dead_code))]
pub fn dot_k_rows_portable(
    rows: &[&KRow],
    a: &ActQ8Mm,
    tiles: std::ops::Range<usize>,
    out: &mut [&mut [f32]],
) {
    debug_assert!(!rows.is_empty());
    debug_assert_eq!(rows.len(), out.len());
    debug_assert!(rows.iter().all(|r| r.kind == rows[0].kind));
    debug_assert!(out.iter().all(|o| o.len() == a.n_tokens));
    debug_assert!(tiles.end <= a.n_tile);
    let kind = rows[0].kind;
    let mins = kind == KKind::Q4K;
    // A scale group is 32 elements, or 16 for `Q6_K`.
    let group = if kind == KKind::Q6K { 16 } else { 32 };
    for tl in tiles {
        let qtile = &a.q[tl * a.n_block * MM_TILE * 32..][..a.n_block * MM_TILE * 32];
        let dtile = &a.d[tl * a.n_super * MM_TILE..][..a.n_super * MM_TILE];
        let btile = &a.bsum[tl * a.n_super * MM_TILE * SUBS..][..a.n_super * MM_TILE * SUBS];
        for k in 0..MM_TILE {
            let t = tl * MM_TILE + k;
            if t >= a.n_tokens {
                break;
            }
            let (tp, half) = (k / 2, k % 2);
            // Token `k`'s 32 bytes of block `b`, in element order, out of
            // the pair-interleaved tile.
            let token_block = |b: usize, e: usize| -> i32 {
                let c = e / 8;
                qtile[b * MM_TILE * 32 + ((tp * 4 + c) * 2 + half) * 8 + e % 8] as i32
            };
            for (row, o) in rows.iter().zip(out.iter_mut()) {
                let mut acc = 0f32;
                if mins {
                    for s in 0..a.n_super {
                        let ad = dtile[s * MM_TILE + k];
                        let mut i = 0i32;
                        for j in 0..SUBS {
                            i += row.mins[s * SUBS + j] as i32
                                * btile[(s * MM_TILE + k) * SUBS + j] as i32;
                        }
                        acc -= ad * row.dmin[s] * i as f32;
                    }
                }
                for s in 0..a.n_super {
                    let mut isum = 0i32;
                    for j in 0..SUBS {
                        let b = s * SUBS + j;
                        let wq = &row.q[b * 32..b * 32 + 32];
                        for g in 0..(32 / group) {
                            let dot: i32 = (g * group..(g + 1) * group)
                                .map(|e| wq[e] as i32 * token_block(b, e))
                                .sum();
                            isum += row.sc[b * (32 / group) + g] as i32 * dot;
                        }
                    }
                    let ad = dtile[s * MM_TILE + k];
                    acc += ad * (row.d[s] * isum as f32);
                }
                o[t] = acc;
            }
        }
    }
}

/// The number of [`MM_TILE`]-token tiles `a` holds — for a caller splitting
/// [`dot_k_rows_tiles`] over tokens.
impl ActQ8Mm {
    pub fn n_tiles(&self) -> usize {
        self.n_tile
    }
}

/// The rows' weights re-laid for the inner loop, built once per call:
/// per row pair, per 32-element block, four 16-byte vectors of the two
/// rows' same 8 elements — one load per `smmla` operand — and the scale
/// vectors `[sc0, sc0, sc1, sc1]` per scale group, ready to `mla`.
#[cfg(target_arch = "aarch64")]
struct PairedRows {
    /// `[pair][block][chunk][2][8]`
    q: Vec<i8>,
    /// `[pair][group]` — a group is 32 elements, or 16 for `Q6_K`.
    sc: Vec<[i32; 4]>,
    /// `[pair][super]` — `[d0, d0, d1, d1]`.
    d: Vec<[f32; 4]>,
}

#[cfg(target_arch = "aarch64")]
impl PairedRows {
    fn build(rows: &[&KRow], n_block: usize, n_super: usize, sc16: bool) -> Self {
        let n_pair = rows.len() / 2;
        let groups = if sc16 { n_block * 2 } else { n_block };
        let mut q = vec![0i8; n_pair * n_block * 64];
        let mut sc = vec![[0i32; 4]; n_pair * groups];
        let mut d = vec![[0f32; 4]; n_pair * n_super];
        for (p, [r0, r1]) in rows.as_chunks::<2>().0.iter().enumerate() {
            let dst = &mut q[p * n_block * 64..][..n_block * 64];
            debug_assert!(r0.q.len() >= n_block * 32 && r1.q.len() >= n_block * 32);
            // Each 8-byte chunk of the two rows side by side: a 64-bit
            // zip of two 16-byte loads, two chunks a store.
            // Safety: NEON is baseline on aarch64; block `b` reads
            // `r*.q[32b..32b + 32]` and writes `dst[64b..64b + 64]`.
            unsafe {
                use std::arch::aarch64::*;
                for b in 0..n_block {
                    let (a, c) = (r0.q.as_ptr().add(b * 32), r1.q.as_ptr().add(b * 32));
                    let o = dst.as_mut_ptr().add(b * 64);
                    for h in 0..2 {
                        let x0 = vreinterpretq_u64_s8(vld1q_s8(a.add(h * 16)));
                        let x1 = vreinterpretq_u64_s8(vld1q_s8(c.add(h * 16)));
                        vst1q_s8(o.add(h * 32), vreinterpretq_s8_u64(vzip1q_u64(x0, x1)));
                        vst1q_s8(o.add(h * 32 + 16), vreinterpretq_s8_u64(vzip2q_u64(x0, x1)));
                    }
                }
            }
            for g in 0..groups {
                let (s0, s1) = (r0.sc[g] as i32, r1.sc[g] as i32);
                sc[p * groups + g] = [s0, s0, s1, s1];
            }
            for s in 0..n_super {
                d[p * n_super + s] = [r0.d[s], r0.d[s], r1.d[s], r1.d[s]];
            }
        }
        Self { q, sc, d }
    }
}

/// The tile loop. `MINS` runs `Q4_K`'s asymmetric correction first;
/// `SC16` reads a scale per 16 elements (`Q6_K`) rather than per 32.
///
/// Per 8-element chunk: two weight vectors (a row pair each), four
/// activation vectors (a token pair each), eight `smmla` into eight 2x2
/// accumulators. Per scale group the accumulators fold into the super-block
/// sums with one `mla` each; per super-block, three vector ops take them to
/// `f32`. Everything stays in lanes until the tile is stored.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
unsafe fn gemm_k_rows_mmla<const MINS: bool, const SC16: bool>(
    rows: &[&KRow],
    a: &ActQ8Mm,
    tiles: std::ops::Range<usize>,
    out: &mut [&mut [f32]],
) {
    use std::arch::aarch64::*;
    let paired = PairedRows::build(rows, a.n_block, a.n_super, SC16);
    let groups_per_block = if SC16 { 2 } else { 1 };
    let n_groups = a.n_block * groups_per_block;
    let n_quad = rows.len() / ROW_QUAD;
    unsafe {
        // Each super-block's activation scales in the accumulators' lane
        // order, `[ad[2tp], ad[2tp+1], ad[2tp], ad[2tp+1]]` per token pair:
        // built once a tile for all its row quads rather than lane by lane
        // in every quad's epilogue.
        let mut adl = vec![[vdupq_n_f32(0.0); 4]; a.n_super];
        // For `Q4_K`: each row pair's mins per super-block as `smmla`'s row
        // operand (`[r0 m0..8, r1 m0..8]`, all in `0..64`) with its
        // `[dmin0, dmin0, dmin1, dmin1]`; and a tile's sub-block sums per
        // token pair split for it, low seven bits and the rest.
        let mins: Vec<([i8; 16], [f32; 4])> = if MINS {
            (0..rows.len() / 2)
                .flat_map(|p| (0..a.n_super).map(move |s| (p, s)))
                .map(|(p, s)| {
                    let (r0, r1) = (rows[2 * p], rows[2 * p + 1]);
                    let mut m = [0i8; 16];
                    for j in 0..SUBS {
                        m[j] = r0.mins[s * SUBS + j] as i8;
                        m[8 + j] = r1.mins[s * SUBS + j] as i8;
                    }
                    (m, [r0.dmin[s], r0.dmin[s], r1.dmin[s], r1.dmin[s]])
                })
                .collect()
        } else {
            Vec::new()
        };
        let mut bsplit =
            vec![([vdupq_n_s8(0); 4], [vdupq_n_s8(0); 4]); if MINS { a.n_super } else { 0 }];
        for tl in tiles {
            let qtile = a.q.as_ptr().add(tl * a.n_block * MM_TILE * 32);
            let dtile = &a.d[tl * a.n_super * MM_TILE..][..a.n_super * MM_TILE];
            for (s, lanes) in adl.iter_mut().enumerate() {
                let ad = &dtile[s * MM_TILE..];
                for (tp, v) in lanes.iter_mut().enumerate() {
                    let l = [ad[tp * 2], ad[tp * 2 + 1], ad[tp * 2], ad[tp * 2 + 1]];
                    *v = vld1q_f32(l.as_ptr());
                }
            }
            if MINS {
                let btile = &a.bsum[tl * a.n_super * MM_TILE * SUBS..];
                for (s, (lo, hi)) in bsplit.iter_mut().enumerate() {
                    for tp in 0..4 {
                        let b0 = vld1q_s16(btile.as_ptr().add((s * MM_TILE + 2 * tp) * SUBS));
                        let b1 = vld1q_s16(btile.as_ptr().add((s * MM_TILE + 2 * tp + 1) * SUBS));
                        let mask = vdupq_n_s16(0x7F);
                        lo[tp] = vcombine_s8(
                            vmovn_s16(vandq_s16(b0, mask)),
                            vmovn_s16(vandq_s16(b1, mask)),
                        );
                        hi[tp] = vcombine_s8(
                            vmovn_s16(vshrq_n_s16::<7>(b0)),
                            vmovn_s16(vshrq_n_s16::<7>(b1)),
                        );
                    }
                }
            }
            for quad in 0..n_quad {
                // Pass 1: `Q4_K`'s `-dmin*m` correction per (row, token),
                // `sum_j m[j]*bsum[j]` over the eight sub-blocks of each
                // super-block — itself a 2 × 8 by 8 × 2 product per row
                // pair and token pair, so `smmla` takes it: the mins as the
                // row operand, the sub-block sums split into their low
                // seven bits and the rest (two `smmla`, recombined exactly
                // in `i32`). Lanes `[r0t0, r0t1, r1t0, r1t1]` as the
                // accumulators', and per lane the scalar kernel's `f32`
                // expression in its order: `acc -= (ad * dm) * sum`.
                let pair0 = quad * 2;
                let mut facc = [[vdupq_n_f32(0.0); 4]; 2];
                if MINS {
                    for s in 0..a.n_super {
                        let (lo, hi) = &bsplit[s];
                        for (rp, facc_rp) in facc.iter_mut().enumerate() {
                            let (mv, dml) = &mins[(pair0 + rp) * a.n_super + s];
                            let mv = vld1q_s8(mv.as_ptr());
                            let dml = vld1q_f32(dml.as_ptr());
                            for (tp, f) in facc_rp.iter_mut().enumerate() {
                                let il = mmla_2x8(vdupq_n_s32(0), mv, lo[tp]);
                                let ih = mmla_2x8(vdupq_n_s32(0), mv, hi[tp]);
                                let sum = vaddq_s32(il, vshlq_n_s32::<7>(ih));
                                let corr =
                                    vmulq_f32(vmulq_f32(adl[s][tp], dml), vcvtq_f32_s32(sum));
                                *f = vsubq_f32(*f, corr);
                            }
                        }
                    }
                }
                let mut acc = [[0f32; MM_TILE]; ROW_QUAD];

                let wq0 = paired.q.as_ptr().add(pair0 * a.n_block * 64);
                let wq1 = paired.q.as_ptr().add((pair0 + 1) * a.n_block * 64);
                let sc0 = paired.sc.as_ptr().add(pair0 * n_groups);
                let sc1 = paired.sc.as_ptr().add((pair0 + 1) * n_groups);

                // Pass 2: the symmetric dot.
                for (s, adl_s) in adl.iter().enumerate() {
                    let mut isum = [[vdupq_n_s32(0); 4]; 2];
                    for j in 0..SUBS {
                        let b = s * SUBS + j;
                        let x = qtile.add(b * MM_TILE * 32);
                        let w0 = wq0.add(b * 64);
                        let w1 = wq1.add(b * 64);
                        let mut part = [[vdupq_n_s32(0); 4]; 2];
                        for h in 0..2 {
                            for c in (h * 2)..(h * 2 + 2) {
                                let wv0 = vld1q_s8(w0.add(c * 16));
                                let wv1 = vld1q_s8(w1.add(c * 16));
                                let [p0, p1] = &mut part;
                                for (tp, (a0, a1)) in p0.iter_mut().zip(p1.iter_mut()).enumerate() {
                                    let xv = vld1q_s8(x.add(tp * 64 + c * 16));
                                    *a0 = mmla_2x8(*a0, wv0, xv);
                                    *a1 = mmla_2x8(*a1, wv1, xv);
                                }
                            }
                            if SC16 || h == 1 {
                                let g = if SC16 { b * 2 + h } else { b };
                                let scv = [
                                    vld1q_s32((*sc0.add(g)).as_ptr()),
                                    vld1q_s32((*sc1.add(g)).as_ptr()),
                                ];
                                for rp in 0..2 {
                                    for tp in 0..4 {
                                        isum[rp][tp] =
                                            vmlaq_s32(isum[rp][tp], part[rp][tp], scv[rp]);
                                        part[rp][tp] = vdupq_n_s32(0);
                                    }
                                }
                            }
                        }
                    }
                    // `acc += ad * (d * isum)`, per lane — the scalar
                    // kernel's expression, operation for operation.
                    for rp in 0..2 {
                        let dv = vld1q_f32(paired.d[(pair0 + rp) * a.n_super + s].as_ptr());
                        for tp in 0..4 {
                            let adv = adl_s[tp];
                            let f = vcvtq_f32_s32(isum[rp][tp]);
                            facc[rp][tp] =
                                vaddq_f32(facc[rp][tp], vmulq_f32(adv, vmulq_f32(dv, f)));
                        }
                    }
                }

                // Back to `[row][token]` and out, dropping the padded tail.
                for rp in 0..2 {
                    for tp in 0..4 {
                        let mut lanes = [0f32; 4];
                        vst1q_f32(lanes.as_mut_ptr(), facc[rp][tp]);
                        acc[rp * 2][tp * 2] = lanes[0];
                        acc[rp * 2][tp * 2 + 1] = lanes[1];
                        acc[rp * 2 + 1][tp * 2] = lanes[2];
                        acc[rp * 2 + 1][tp * 2 + 1] = lanes[3];
                    }
                }
                for (r, row_acc) in acc.iter().enumerate() {
                    let dst = &mut out[quad * ROW_QUAD + r];
                    let base = tl * MM_TILE;
                    let n = MM_TILE.min(dst.len().saturating_sub(base));
                    dst[base..base + n].copy_from_slice(&row_acc[..n]);
                }
            }
        }
    }
}

/// Whether [`bf16_tiles`] may use `bfmmla` (ARMv8.6 `bf16`): the CPU says
/// so and the instruction agrees with the scalar definition on known
/// operands — the [`have_i8mm`] contract.
#[cfg(target_arch = "aarch64")]
pub fn have_bf16mm() -> bool {
    use std::sync::OnceLock;
    static USABLE: OnceLock<bool> = OnceLock::new();
    *USABLE.get_or_init(|| {
        if !std::arch::is_aarch64_feature_detected!("bf16") {
            return false;
        }
        let a: [f32; 8] = [1.0, -2.0, 0.5, 3.0, 4.0, 0.25, -1.0, 2.0];
        let b: [f32; 8] = [2.0, 1.0, -4.0, 0.5, -1.0, 3.0, 2.0, 0.75];
        let (ab, bb) = (a.map(to_bf16), b.map(to_bf16));
        let mut expect = [0f32; 4];
        for i in 0..2 {
            for j in 0..2 {
                expect[i * 2 + j] = (0..4).map(|k| a[i * 4 + k] * b[j * 4 + k]).sum();
            }
        }
        // Safety: `bf16` was detected; both arrays are 16 bytes.
        let got = unsafe {
            use std::arch::aarch64::*;
            let acc = bfmmla(
                vdupq_n_f32(0.0),
                vld1q_u16(ab.as_ptr()),
                vld1q_u16(bb.as_ptr()),
            );
            let mut out = [0f32; 4];
            vst1q_f32(out.as_mut_ptr(), acc);
            out
        };
        got == expect
    })
}

#[cfg(not(target_arch = "aarch64"))]
pub fn have_bf16mm() -> bool {
    false
}

/// `bfmmla`: `acc += a (2x4 bf16, rows in each 8-byte half) . b^T (2x4)`,
/// lanes `[a0.b0, a0.b1, a1.b0, a1.b1]`, products summed in `f32`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn bfmmla(
    acc: std::arch::aarch64::float32x4_t,
    a: std::arch::aarch64::uint16x8_t,
    b: std::arch::aarch64::uint16x8_t,
) -> std::arch::aarch64::float32x4_t {
    let mut acc = acc;
    unsafe {
        std::arch::asm!(
            ".arch_extension bf16",
            "bfmmla {acc:v}.4s, {a:v}.8h, {b:v}.8h",
            acc = inout(vreg) acc,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack)
        );
    }
    acc
}

/// `f32` to `bf16`, rounded to nearest even (NaN is not expected here).
#[inline]
pub fn to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16
}

#[inline]
fn from_bf16(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}

/// `e^x` for `x ≤ 0` to the precision a `bf16` probability keeps: the
/// same range reduction as `tensor::exp_neon`, a cubic for `e^r` (relative
/// error ~6e-4 against `bf16`'s 4e-3 rounding) and only the lower clamp —
/// a softmax row's shifted scores are never positive.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn exp_for_bf16(x: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    unsafe {
        let x = vmaxq_f32(x, vdupq_n_f32(-87.0));
        let n = vrndnq_f32(vmulq_n_f32(x, std::f32::consts::LOG2_E));
        let r = vfmaq_n_f32(x, n, -0.693_359_4);
        let r = vfmaq_n_f32(r, n, 2.121_944_4e-4);
        let p = vfmaq_n_f32(vdupq_n_f32(0.5), r, 1.0 / 6.0);
        let p = vfmaq_f32(vdupq_n_f32(1.0), r, p);
        let p = vfmaq_f32(vdupq_n_f32(1.0), r, p);
        let scale = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(
            vcvtq_s32_f32(n),
            vdupq_n_s32(127),
        )));
        vmulq_f32(p, scale)
    }
}

/// Four `f32` to `bf16`, rounded to nearest even — one `bfcvtn`.
///
/// # Safety
/// The `bf16` extension ([`have_bf16mm`]).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn bfcvtn(v: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::uint16x4_t {
    let out: std::arch::aarch64::uint16x4_t;
    unsafe {
        std::arch::asm!(
            ".arch_extension bf16",
            "bfcvtn {o:v}.4h, {i:v}.4s",
            o = out(vreg) out,
            i = in(vreg) v,
            options(pure, nomem, nostack)
        );
    }
    out
}

/// Rows of `bf16` for [`bf16_tiles`], two at a time: for each chunk of four
/// `k`, a pair's two rows side by side (`[pair][chunk][2][4]`) — `bfmmla`'s
/// operand. Rows are padded to a multiple of eight and `k` to four, with
/// zeros.
pub struct PackedBf16 {
    data: Vec<u16>,
    /// Padded to a multiple of 8.
    pub rows: usize,
    /// `k` padded to a multiple of 4, over 4.
    chunks: usize,
}

impl PackedBf16 {
    /// `n_rows` rows of `k` values, element `(r, i)` read through `get`.
    pub fn pack(n_rows: usize, k: usize, get: impl Fn(usize, usize) -> f32) -> Self {
        let rows = n_rows.div_ceil(8) * 8;
        let chunks = k.div_ceil(4);
        let mut data = vec![0u16; rows * chunks * 4];
        for r in 0..n_rows {
            let (pair, half) = (r / 2, r % 2);
            let base = pair * chunks * 8;
            for i in 0..k {
                data[base + (i / 4) * 8 + half * 4 + i % 4] = to_bf16(get(r, i));
            }
        }
        Self { data, rows, chunks }
    }

    /// The transpose of `k` source rows of `n_rows` values each — row `j`
    /// is `src[j * stride..][..n_rows]` — so that packed row `r` is column
    /// `r` of the source: a head's values `[keys][head_dim]` read in place,
    /// row by row, rather than a strided element at a time.
    pub fn pack_transposed(n_rows: usize, k: usize, src: &[f32], stride: usize) -> Self {
        let rows = n_rows.div_ceil(8) * 8;
        let chunks = k.div_ceil(4);
        let mut data = vec![0u16; rows * chunks * 4];
        for j in 0..k {
            let row = &src[j * stride..j * stride + n_rows];
            let at = (j / 4) * 8 + j % 4;
            for (r, pair) in row.chunks(2).enumerate() {
                let base = r * chunks * 8 + at;
                data[base] = to_bf16(pair[0]);
                if let Some(&v) = pair.get(1) {
                    data[base + 4] = to_bf16(v);
                }
            }
        }
        Self { data, rows, chunks }
    }

    /// Row `r` as `e^(values − shift)` (the rest of the row zeros), returning
    /// the sum of the exponentials — a softmax row's exponentials, their
    /// denominator and its `bf16` packing in one pass, the rounding
    /// [`to_bf16`]'s, four lanes at a time.
    pub fn set_row_exp(&mut self, r: usize, values: &[f32], shift: f32) -> f32 {
        let (pair, half) = (r / 2, r % 2);
        let base = pair * self.chunks * 8 + half * 4;
        debug_assert!(r < self.rows && values.len() <= self.chunks * 4);
        let row = &mut self.data[base..base + (self.chunks - 1) * 8 + 4];
        let whole = values.len() / 4;
        #[cfg(not(target_arch = "aarch64"))]
        let mut sum = 0f32;
        #[cfg(target_arch = "aarch64")]
        // Safety: NEON is baseline on aarch64; chunk `c < whole` reads
        // `values[4c..4c + 4]` and writes `row[8c..8c + 4]`, both in bounds.
        let mut sum = unsafe {
            use std::arch::aarch64::*;
            let s = vdupq_n_f32(shift);
            let round = vdupq_n_u32(0x7FFF);
            let fast = have_bf16mm();
            let one = |c: usize, acc: &mut float32x4_t, row: &mut [u16]| {
                let x = vsubq_f32(vld1q_f32(values.as_ptr().add(c * 4)), s);
                if fast {
                    let e = exp_for_bf16(x);
                    *acc = vaddq_f32(*acc, e);
                    vst1_u16(row.as_mut_ptr().add(c * 8), bfcvtn(e));
                    return;
                }
                let e = crate::engine::tensor::exp_neon(x);
                *acc = vaddq_f32(*acc, e);
                let bits = vreinterpretq_u32_f32(e);
                let odd = vandq_u32(vshrq_n_u32::<16>(bits), vdupq_n_u32(1));
                let rounded = vaddq_u32(vaddq_u32(bits, round), odd);
                vst1_u16(row.as_mut_ptr().add(c * 8), vshrn_n_u32::<16>(rounded));
            };
            // Four chunks an iteration: each exponential is one long
            // dependent chain, and four of them side by side keep the
            // pipes busy where one ran at its latency.
            let mut acc = [vdupq_n_f32(0.0); 4];
            let mut c = 0;
            while c + 4 <= whole {
                for (l, a) in acc.iter_mut().enumerate() {
                    one(c + l, a, row);
                }
                c += 4;
            }
            while c < whole {
                one(c, &mut acc[0], row);
                c += 1;
            }
            vaddvq_f32(vaddq_f32(
                vaddq_f32(acc[0], acc[1]),
                vaddq_f32(acc[2], acc[3]),
            ))
        };
        #[cfg(not(target_arch = "aarch64"))]
        for c in 0..whole {
            for e in 0..4 {
                let v = (values[c * 4 + e] - shift).exp();
                sum += v;
                row[c * 8 + e] = to_bf16(v);
            }
        }
        for c in whole..self.chunks {
            for e in 0..4 {
                row[c * 8 + e] = match values.get(c * 4 + e) {
                    Some(&x) => {
                        let v = (x - shift).exp();
                        sum += v;
                        to_bf16(v)
                    }
                    None => 0,
                };
            }
        }
        sum
    }

    /// Room for `n_rows` rows of `k`, all zero, to be filled by
    /// [`PackedBf16::set_row`] — kept across calls of one shape.
    pub fn zeroed(n_rows: usize, k: usize) -> Self {
        let (rows, chunks) = (n_rows.div_ceil(8) * 8, k.div_ceil(4));
        Self {
            data: vec![0u16; rows * chunks * 4],
            rows,
            chunks,
        }
    }

    /// Row `r` from `values` (at most `k` long; the rest of the row, its
    /// padding included, zeros) — a softmax row written straight into
    /// `bfmmla`'s layout.
    pub fn set_row(&mut self, r: usize, values: &[f32]) {
        let (pair, half) = (r / 2, r % 2);
        let base = pair * self.chunks * 8 + half * 4;
        debug_assert!(r < self.rows && values.len() <= self.chunks * 4);
        let row = &mut self.data[base..base + (self.chunks - 1) * 8 + 4];
        let whole = values.len() / 4;
        for (c, v) in values.as_chunks::<4>().0.iter().enumerate() {
            let slot = &mut row[c * 8..c * 8 + 4];
            for (s, &x) in slot.iter_mut().zip(v) {
                *s = to_bf16(x);
            }
        }
        for c in whole..self.chunks {
            for e in 0..4 {
                let i = c * 4 + e;
                row[c * 8 + e] = values.get(i).map_or(0, |&x| to_bf16(x));
            }
        }
    }
}

/// `out[i][j] = a_i · b_j` for every row of `a` and of `b` (both packed
/// over the same `k`), `out` row-major `[a.rows][b.rows]` — 8 × 8 tiles of
/// sixteen `bfmmla` per eight loads, `f32` sums. The value product of the
/// picture transformers' attention (`doc/PERF-IMAGE.md`, task 5b): `a` the
/// block's probabilities, `b` a head's values transposed.
pub fn bf16_tiles(a: &PackedBf16, b: &PackedBf16, out: &mut [f32]) {
    debug_assert_eq!(a.chunks, b.chunks);
    bf16_tiles_at(a, b, 0, out)
}

/// [`bf16_tiles`] with `a` packed over part of `b`'s `k`: `a`'s chunk `c`
/// meets `b`'s chunk `b_chunk + c` — a query block's probabilities over the
/// keys its windows cover, against a head's values packed once over all of
/// them (`attention_mixed`).
pub fn bf16_tiles_at(a: &PackedBf16, b: &PackedBf16, b_chunk: usize, out: &mut [f32]) {
    debug_assert!(b_chunk + a.chunks <= b.chunks);
    debug_assert_eq!(out.len(), a.rows * b.rows);
    #[cfg(target_arch = "aarch64")]
    if have_bf16mm() {
        // Safety: `bf16` was verified.
        unsafe { bf16_tiles_mmla(a, b, b_chunk, out) };
        return;
    }
    bf16_tiles_portable(a, b, b_chunk, out)
}

/// The definition [`bf16_tiles_mmla`] is held to.
pub fn bf16_tiles_portable(a: &PackedBf16, b: &PackedBf16, b_chunk: usize, out: &mut [f32]) {
    let at = |m: &PackedBf16, r: usize, c: usize, i: usize| -> f32 {
        from_bf16(m.data[(r / 2) * m.chunks * 8 + c * 8 + (r % 2) * 4 + i])
    };
    for i in 0..a.rows {
        for j in 0..b.rows {
            let mut acc = 0f32;
            for c in 0..a.chunks {
                for e in 0..4 {
                    acc += at(a, i, c, e) * at(b, j, b_chunk + c, e);
                }
            }
            out[i * b.rows + j] = acc;
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn bf16_tiles_mmla(a: &PackedBf16, b: &PackedBf16, b_chunk: usize, out: &mut [f32]) {
    use std::arch::aarch64::*;
    let chunks = a.chunks;
    let b_chunks = b.chunks;
    unsafe {
        for it in 0..a.rows / 8 {
            for jt in 0..b.rows / 8 {
                let mut acc = [[vdupq_n_f32(0.0); 4]; 4];
                let ap = a.data.as_ptr().add(it * 4 * chunks * 8);
                let bp = b.data.as_ptr().add(jt * 4 * b_chunks * 8 + b_chunk * 8);
                for c in 0..chunks {
                    let av: [uint16x8_t; 4] =
                        std::array::from_fn(|p| vld1q_u16(ap.add(p * chunks * 8 + c * 8)));
                    let bv: [uint16x8_t; 4] =
                        std::array::from_fn(|p| vld1q_u16(bp.add(p * b_chunks * 8 + c * 8)));
                    for (ap_, row) in av.iter().zip(acc.iter_mut()) {
                        for (bp_, cell) in bv.iter().zip(row.iter_mut()) {
                            *cell = bfmmla(*cell, *ap_, *bp_);
                        }
                    }
                }
                for (pi, row) in acc.iter().enumerate() {
                    for (pj, &v4) in row.iter().enumerate() {
                        let mut lanes = [0f32; 4];
                        vst1q_f32(lanes.as_mut_ptr(), v4);
                        for (lane, v) in lanes.iter().enumerate() {
                            let i = it * 8 + pi * 2 + lane / 2;
                            let j = jt * 8 + pj * 2 + lane % 2;
                            out[i * b.rows + j] = *v;
                        }
                    }
                }
            }
        }
    }
}

/// A weight matrix in `bf16`, packed once for [`matmul_bf16`]: its rows
/// are [`bf16_tiles`]'s `b` operand, so a prompt's activations, packed per
/// task, meet them on the 8 × 8 `bfmmla` tile with `f32` sums.
///
/// For weights the file stores unquantized (`F32`, `F16`, `BF16`), where an
/// `int8` copy would add a quantization the file does not have: a `BF16`
/// matrix is exact here, an `F32` one rounds each weight to 8 bits of
/// mantissa (`doc/PERF-ALL.md`, task 13).
pub struct Bf16Weights {
    packed: PackedBf16,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl Bf16Weights {
    /// `out_dim` rows of `in_dim`, row `o` produced by `row(o)` in `f32`.
    pub fn from_rows(
        out_dim: usize,
        in_dim: usize,
        row: impl Fn(usize) -> Vec<f32> + Sync,
    ) -> Self {
        let rows: Vec<Vec<f32>> = (0..out_dim).into_par_iter().map(&row).collect();
        let packed = PackedBf16::pack(out_dim, in_dim, |r, i| rows[r][i]);
        Self {
            packed,
            in_dim,
            out_dim,
        }
    }

    /// Resident bytes.
    pub fn bytes(&self) -> usize {
        self.packed.data.len() * 2
    }
}

/// Tokens per [`matmul_bf16`] task — two 8-row `bfmmla` tiles of them.
const BF16_TOKEN_BLOCK: usize = 16;

/// Weight tiles (8 rows each) per [`matmul_bf16`] task.
const BF16_ROW_TILES: usize = 8;

/// `x · wᵀ` for `x` `[n_tokens][in_dim]` on `bfmmla`: the activations
/// rounded to `bf16` per task, `f32` sums, `[n_tokens][out_dim]` out.
/// Tasks are token blocks × groups of weight tiles, so a narrow matrix (256
/// rows) against a chunk of tokens and a wide one (8960 rows) both spread
/// over the pool. Needs [`have_bf16mm`].
pub fn matmul_bf16(x: &[f32], n_tokens: usize, w: &Bf16Weights) -> Vec<f32> {
    debug_assert_eq!(x.len(), n_tokens * w.in_dim);
    debug_assert!(have_bf16mm());
    let (in_dim, out_dim) = (w.in_dim, w.out_dim);
    let mut out = vec![0f32; n_tokens * out_dim];
    let n_tb = n_tokens.div_ceil(BF16_TOKEN_BLOCK);
    let n_jt = w.packed.rows / 8;
    let n_jg = n_jt.div_ceil(BF16_ROW_TILES);
    #[derive(Clone, Copy)]
    struct Sink(*mut f32);
    unsafe impl Send for Sink {}
    unsafe impl Sync for Sink {}
    impl Sink {
        fn at(self, offset: usize) -> *mut f32 {
            // Safety: the caller writes inside `out`.
            unsafe { self.0.add(offset) }
        }
    }
    let sink = Sink(out.as_mut_ptr());
    (0..n_tb * n_jg).into_par_iter().for_each_init(
        || (usize::MAX, PackedBf16::zeroed(BF16_TOKEN_BLOCK, in_dim)),
        |(held, a), task| {
            let (tb, jg) = (task / n_jg, task % n_jg);
            let t0 = tb * BF16_TOKEN_BLOCK;
            let nt = BF16_TOKEN_BLOCK.min(n_tokens - t0);
            // Consecutive tasks of a worker share a token block: packed once.
            if *held != tb {
                for r in 0..BF16_TOKEN_BLOCK {
                    if r < nt {
                        a.set_row(r, &x[(t0 + r) * in_dim..(t0 + r + 1) * in_dim]);
                    } else {
                        a.set_row(r, &[]);
                    }
                }
                *held = tb;
            }
            let jts = jg * BF16_ROW_TILES..((jg + 1) * BF16_ROW_TILES).min(n_jt);
            let mut tile = [0f32; 64];
            for it in 0..a.rows / 8 {
                for jt in jts.clone() {
                    bf16_tile(a, &w.packed, it, jt, &mut tile);
                    for (i, row) in tile.chunks(8).enumerate() {
                        let t = it * 8 + i;
                        if t >= nt {
                            break;
                        }
                        for (j, v) in row.iter().enumerate() {
                            let o = jt * 8 + j;
                            if o < out_dim {
                                // Safety: (t0 + t, o) is this task's alone.
                                unsafe { *sink.at((t0 + t) * out_dim + o) = *v };
                            }
                        }
                    }
                }
            }
        },
    );
    out
}

/// One 8 × 8 tile of `a · bᵀ`: `a`'s rows `8 it ..`, `b`'s rows `8 jt ..`,
/// row-major into `out`.
fn bf16_tile(a: &PackedBf16, b: &PackedBf16, it: usize, jt: usize, out: &mut [f32; 64]) {
    debug_assert_eq!(a.chunks, b.chunks);
    #[cfg(target_arch = "aarch64")]
    if have_bf16mm() {
        use std::arch::aarch64::*;
        let chunks = a.chunks;
        // Safety: `bf16` was verified; both tiles are inside their packs.
        unsafe {
            let mut acc = [[vdupq_n_f32(0.0); 4]; 4];
            let ap = a.data.as_ptr().add(it * 4 * chunks * 8);
            let bp = b.data.as_ptr().add(jt * 4 * chunks * 8);
            for c in 0..chunks {
                let av: [uint16x8_t; 4] =
                    std::array::from_fn(|p| vld1q_u16(ap.add(p * chunks * 8 + c * 8)));
                let bv: [uint16x8_t; 4] =
                    std::array::from_fn(|p| vld1q_u16(bp.add(p * chunks * 8 + c * 8)));
                for (ap_, row) in av.iter().zip(acc.iter_mut()) {
                    for (bp_, cell) in bv.iter().zip(row.iter_mut()) {
                        *cell = bfmmla(*cell, *ap_, *bp_);
                    }
                }
            }
            for (pi, row) in acc.iter().enumerate() {
                for (pj, &v4) in row.iter().enumerate() {
                    let mut lanes = [0f32; 4];
                    vst1q_f32(lanes.as_mut_ptr(), v4);
                    for (lane, v) in lanes.iter().enumerate() {
                        out[(pi * 2 + lane / 2) * 8 + pj * 2 + lane % 2] = *v;
                    }
                }
            }
        }
        return;
    }
    let at = |m: &PackedBf16, r: usize, c: usize, i: usize| -> f32 {
        from_bf16(m.data[(r / 2) * m.chunks * 8 + c * 8 + (r % 2) * 4 + i])
    };
    for i in 0..8 {
        for j in 0..8 {
            let mut acc = 0f32;
            for c in 0..a.chunks {
                for e in 0..4 {
                    acc += at(a, it * 8 + i, c, e) * at(b, jt * 8 + j, c, e);
                }
            }
            out[i * 8 + j] = acc;
        }
    }
}

/// Rows per [`RowI8`] group — the 8 × 8 `smmla` tile's height.
pub const ROW_OCT: usize = 8;

/// Weights as `int8` with **one `f32` scale per row**, packed eight rows at
/// a time in [`ActQ8Mm`]'s block layout (`[group][block][pair][chunk][2][8]`,
/// a pair's two rows side by side per 8-element chunk) — what
/// [`matmul_rowi8`] multiplies (`doc/PERF-IMAGE.md`, task 10).
///
/// Unlike a K-quant row there is no per-block weight scale, so the tile
/// accumulates in `i32` over a whole 256-element super-block and folds into
/// `f32` once per super-block with the *activation's* scale; the activations
/// keep their per-256 scale, which is what confines an outlier channel.
/// About a byte a weight, resident: built at load from any weight type.
pub struct RowI8 {
    pub in_dim: usize,
    pub out_dim: usize,
    groups: usize,
    q: Vec<i8>,
    /// `[groups * 8]`, the padding rows' 1.0.
    scale: Vec<f32>,
}

impl RowI8 {
    /// `out_dim` rows of `in_dim` (a multiple of [`SUPER_BLOCK`]), row `o`
    /// produced by `row(o)` in `f32`.
    pub fn quantize(out_dim: usize, in_dim: usize, row: impl Fn(usize) -> Vec<f32> + Sync) -> Self {
        assert!(
            in_dim.is_multiple_of(SUPER_BLOCK),
            "RowI8 rows are whole super-blocks"
        );
        let groups = out_dim.div_ceil(ROW_OCT);
        let n_block = in_dim / 32;
        let mut q = vec![0i8; groups * n_block * ROW_OCT * 32];
        let mut scale = vec![1f32; groups * ROW_OCT];
        q.par_chunks_mut(n_block * ROW_OCT * 32)
            .zip(scale.par_chunks_mut(ROW_OCT))
            .enumerate()
            .for_each(|(g, (qg, sg))| {
                for (r, slot) in sg.iter_mut().enumerate() {
                    let o = g * ROW_OCT + r;
                    if o >= out_dim {
                        break;
                    }
                    let values = row(o);
                    debug_assert_eq!(values.len(), in_dim);
                    let max = values.iter().fold(0f32, |m, v| m.max(v.abs()));
                    let s = if max > 0.0 { max / 127.0 } else { 1.0 };
                    *slot = s;
                    let inv = 1.0 / s;
                    let (rp, half) = (r / 2, r % 2);
                    for b in 0..n_block {
                        for c in 0..4 {
                            let dst =
                                &mut qg[b * ROW_OCT * 32 + ((rp * 4 + c) * 2 + half) * 8..][..8];
                            for (d, v) in dst.iter_mut().zip(&values[b * 32 + c * 8..][..8]) {
                                *d = (v * inv).round().clamp(-127.0, 127.0) as i8;
                            }
                        }
                    }
                }
            });
        Self {
            in_dim,
            out_dim,
            groups,
            q,
            scale,
        }
    }

    /// Resident bytes.
    pub fn bytes(&self) -> usize {
        self.q.len() + self.scale.len() * 4
    }

    fn group(&self, g: usize) -> &[i8] {
        let n = self.in_dim / 32 * ROW_OCT * 32;
        &self.q[g * n..(g + 1) * n]
    }
}

/// `x · wᵀ` for `x` `[n_tokens][in_dim]`: the activations quantized per
/// 256 ([`ActQ8Mm`]), then row groups × token blocks in parallel, each an
/// 8 × 8 tile product ([`rowi8_tiles`]). `[n_tokens][out_dim]` out.
pub fn matmul_rowi8(x: &[f32], n_tokens: usize, w: &RowI8) -> Vec<f32> {
    debug_assert_eq!(x.len(), n_tokens * w.in_dim);
    let acts = ActQ8Mm::quantize(x, w.in_dim, n_tokens);
    matmul_rowi8_acts(&acts, n_tokens, w)
}

/// The weight bytes a [`matmul_rowi8_acts`] task keeps hot while an
/// activation tile passes over them: L2-sized, so each 8-token tile is read
/// from memory once per chunk of row groups rather than once per group.
/// Measured best of 128 KB–1.5 MB at a 1024² step's three shapes.
const ROWI8_CHUNK_BYTES: usize = 384 << 10;

/// [`matmul_rowi8`] on activations already quantized.
///
/// Tasks are chunks of row groups × blocks of token tiles; inside one, each
/// 8-token tile (`in_dim` bytes a token, L1) meets every group of the chunk
/// (`ROWI8_CHUNK_BYTES`, L2) before the next tile is read. One task per row
/// group streamed every activation once per group — at a 1024² step's 4,096
/// × 4,096, 16 MB read 512 times a linear.
pub fn matmul_rowi8_acts(acts: &ActQ8Mm, n_tokens: usize, w: &RowI8) -> Vec<f32> {
    let out_dim = w.out_dim;
    let mut out = vec![0f32; n_tokens * out_dim];
    let n_tiles = acts.n_tiles();
    let chunk = (ROWI8_CHUNK_BYTES / (w.in_dim * ROW_OCT)).clamp(1, w.groups);
    let n_chunks = w.groups.div_ceil(chunk);
    // Token blocks only when the chunks alone cannot occupy the pool.
    let wanted = 4 * rayon::current_num_threads();
    let n_blocks = if n_chunks >= wanted {
        1
    } else {
        wanted.div_ceil(n_chunks).min(n_tiles).max(1)
    };
    let tiles_per_block = n_tiles.div_ceil(n_blocks).max(1);
    let n_blocks = n_tiles.div_ceil(tiles_per_block);
    #[derive(Clone, Copy)]
    struct Sink(*mut f32);
    unsafe impl Send for Sink {}
    unsafe impl Sync for Sink {}
    let sink = Sink(out.as_mut_ptr());
    (0..n_chunks * n_blocks).into_par_iter().for_each_init(
        || vec![0f32; ROW_OCT * n_tokens],
        move |yt, task| {
            let sink = sink;
            let (c, b) = (task / n_blocks, task % n_blocks);
            let groups = c * chunk..((c + 1) * chunk).min(w.groups);
            let tiles = b * tiles_per_block..((b + 1) * tiles_per_block).min(n_tiles);
            let mut rows: Vec<&mut [f32]> = yt.chunks_mut(n_tokens).collect();
            for tl in tiles {
                let (t0, t1) = (tl * MM_TILE, ((tl + 1) * MM_TILE).min(n_tokens));
                for g in groups.clone() {
                    rowi8_tiles(w, g, acts, tl..tl + 1, &mut rows);
                    let n_rows = ROW_OCT.min(out_dim - g * ROW_OCT);
                    for t in t0..t1 {
                        for (r, row) in rows.iter().enumerate().take(n_rows) {
                            // Safety: task `task` alone writes columns of
                            // its groups for token rows of its tiles, in a
                            // buffer sized `n_tokens * out_dim` that
                            // outlives the parallel loop.
                            unsafe {
                                *sink.0.add(t * out_dim + g * ROW_OCT + r) = row[t];
                            }
                        }
                    }
                }
            }
        },
    );
    out
}

/// Group `g`'s eight rows against the tokens of `tiles`: `yt[r][t]`, each of
/// the eight slices `n_tokens` long (tokens outside `tiles` untouched).
pub fn rowi8_tiles(
    w: &RowI8,
    g: usize,
    a: &ActQ8Mm,
    tiles: std::ops::Range<usize>,
    yt: &mut [&mut [f32]],
) {
    debug_assert_eq!(yt.len(), ROW_OCT);
    #[cfg(target_arch = "aarch64")]
    if have_i8mm() {
        // Safety: `i8mm` was verified.
        unsafe { rowi8_tiles_mmla(w, g, a, tiles, yt) };
        return;
    }
    rowi8_tiles_portable(w, g, a, tiles, yt)
}

/// The definition [`rowi8_tiles_mmla`] is held to: `i32` sums per
/// super-block, folded with the activation's scale, times the row's.
pub fn rowi8_tiles_portable(
    w: &RowI8,
    g: usize,
    a: &ActQ8Mm,
    tiles: std::ops::Range<usize>,
    yt: &mut [&mut [f32]],
) {
    let wq = w.group(g);
    for tl in tiles {
        let qtile = &a.q[tl * a.n_block * MM_TILE * 32..][..a.n_block * MM_TILE * 32];
        let dtile = &a.d[tl * a.n_super * MM_TILE..][..a.n_super * MM_TILE];
        for k in 0..MM_TILE {
            let t = tl * MM_TILE + k;
            if t >= a.n_tokens {
                break;
            }
            let (tp, th) = (k / 2, k % 2);
            for (r, row_out) in yt.iter_mut().enumerate() {
                let (rp, rh) = (r / 2, r % 2);
                let mut acc = 0f32;
                for s in 0..a.n_super {
                    let mut isum = 0i32;
                    for b in s * SUBS..(s + 1) * SUBS {
                        for c in 0..4 {
                            let wo = b * ROW_OCT * 32 + ((rp * 4 + c) * 2 + rh) * 8;
                            let xo = b * MM_TILE * 32 + ((tp * 4 + c) * 2 + th) * 8;
                            for e in 0..8 {
                                isum += wq[wo + e] as i32 * qtile[xo + e] as i32;
                            }
                        }
                    }
                    acc += isum as f32 * dtile[s * MM_TILE + k];
                }
                row_out[t] = acc * w.scale[g * ROW_OCT + r];
            }
        }
    }
}

/// The 8 × 8 tile: per 8-element chunk, four row pairs and four token pairs
/// loaded and sixteen `smmla` issued into sixteen `i32` accumulators (the
/// K-quant tile is 4 × 8: eight per six loads, and a scale per sub-block).
/// One `f32` fold per 256 elements.
#[cfg(target_arch = "aarch64")]
unsafe fn rowi8_tiles_mmla(
    w: &RowI8,
    g: usize,
    a: &ActQ8Mm,
    tiles: std::ops::Range<usize>,
    yt: &mut [&mut [f32]],
) {
    use std::arch::aarch64::*;
    let wq = w.group(g).as_ptr();
    let scale = &w.scale[g * ROW_OCT..(g + 1) * ROW_OCT];
    unsafe {
        for tl in tiles {
            let qtile = a.q.as_ptr().add(tl * a.n_block * MM_TILE * 32);
            let dtile = &a.d[tl * a.n_super * MM_TILE..][..a.n_super * MM_TILE];
            // `[row pair][token pair]`, lanes `[r0t0, r0t1, r1t0, r1t1]`.
            let mut facc = [[vdupq_n_f32(0.0); 4]; 4];
            for s in 0..a.n_super {
                let mut acc = [[vdupq_n_s32(0); 4]; 4];
                for b in s * SUBS..(s + 1) * SUBS {
                    let x = qtile.add(b * MM_TILE * 32);
                    let wb = wq.add(b * ROW_OCT * 32);
                    for c in 0..4 {
                        let wv: [int8x16_t; 4] =
                            std::array::from_fn(|rp| vld1q_s8(wb.add(rp * 64 + c * 16)));
                        let xv: [int8x16_t; 4] =
                            std::array::from_fn(|tp| vld1q_s8(x.add(tp * 64 + c * 16)));
                        for rp in 0..4 {
                            for tp in 0..4 {
                                acc[rp][tp] = mmla_2x8(acc[rp][tp], wv[rp], xv[tp]);
                            }
                        }
                    }
                }
                let ad = &dtile[s * MM_TILE..s * MM_TILE + MM_TILE];
                for tp in 0..4 {
                    let adl = [ad[tp * 2], ad[tp * 2 + 1], ad[tp * 2], ad[tp * 2 + 1]];
                    let adv = vld1q_f32(adl.as_ptr());
                    for rp in 0..4 {
                        facc[rp][tp] = vfmaq_f32(facc[rp][tp], vcvtq_f32_s32(acc[rp][tp]), adv);
                    }
                }
            }
            let base = tl * MM_TILE;
            let n = MM_TILE.min(a.n_tokens.saturating_sub(base));
            for (rp, row) in facc.iter().enumerate() {
                for (tp, &v4) in row.iter().enumerate() {
                    let mut lanes = [0f32; 4];
                    vst1q_f32(lanes.as_mut_ptr(), v4);
                    for (lane, v) in lanes.iter().enumerate() {
                        let (r, t) = (rp * 2 + lane / 2, tp * 2 + lane % 2);
                        if t < n {
                            yt[r][base + t] = v * scale[r];
                        }
                    }
                }
            }
        }
    }
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
    for (s, block) in row.as_chunks::<BLOCK_BYTES>().0.iter().enumerate() {
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
    // `PQ2_0`/`PTQ1_0` qualify the way `IQ4_NL` does: their scale is per 128
    // rather than per 32, but `unpack_pq2_0`/`unpack_ptq1_0` repeat it across
    // the block's four 32-groups, and a ternary weight has no min term. In
    // `UnpackedRow` form they are indistinguishable from `Q8_0`.
    matches!(
        ggml_type,
        GGML_TYPE_Q8_0
            | GGML_TYPE_Q5_0
            | GGML_TYPE_Q4_0
            | GGML_TYPE_IQ4_NL
            | GGML_TYPE_PQ2_0
            | GGML_TYPE_PTQ1_0
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
        // This layout has no per-`GROUP` sums — `supports_flat` is exactly
        // the symmetric types, whose kernels need none. The `SUMS = false`
        // monomorphization never touches it.
        let mut no_sums = Vec::new();
        for tl in 0..n_tile {
            for k in 0..TOKEN_TILE {
                let t = tl * TOKEN_TILE + k;
                if t >= n_tokens {
                    break;
                }
                let row = &x[t * in_dim..(t + 1) * in_dim];
                for b in 0..n_block {
                    let chunk = &row[b * ACT_BLOCK..(b + 1) * ACT_BLOCK];
                    let dst =
                        &mut q[((tl * n_block + b) * TOKEN_TILE + k) * ACT_BLOCK..][..ACT_BLOCK];
                    // The same kernel `quantize_act` uses — this loop was an
                    // open-coded copy of it, and the copy was the reason
                    // LLVM widened this one to 128 bits while the other got
                    // 256. See `quantize_block`.
                    d[(tl * n_block + b) * TOKEN_TILE + k] =
                        quantize_block::<false>(chunk, dst, &mut no_sums);
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
    // The row-wide AVX2 kernels for the two legacy types a decode reads
    // most — a tied `Q8_0` output head, a QAT file's `q4_0` experts — see
    // `dot_k_row_impl` for why the per-32 form below is slower.
    #[cfg(target_arch = "x86_64")]
    if ISA == ISA_AVX2 || ISA == ISA_VNNI {
        // Safety: both ISAs imply AVX2, checked at run time by the callers
        // that instantiate them.
        match ggml_type {
            GGML_TYPE_Q8_0 => return unsafe { dot_q8_0_avx2(row, act) },
            GGML_TYPE_Q4_0 => return unsafe { dot_q4_0_avx2(row, act) },
            GGML_TYPE_Q5_0 => return unsafe { dot_q5_x_avx2::<false>(row, act) },
            GGML_TYPE_Q5_1 => return unsafe { dot_q5_x_avx2::<true>(row, act) },
            _ => {}
        }
    }
    match ggml_type {
        GGML_TYPE_Q8_0 => dot_q8_0::<ISA>(row, act),
        GGML_TYPE_Q5_0 => dot_q5_0::<ISA>(row, act),
        GGML_TYPE_Q4_0 => dot_q4_0::<ISA>(row, act),
        GGML_TYPE_Q5_1 => dot_q5_1::<ISA>(row, act),
        GGML_TYPE_IQ4_NL => dot_iq4_nl::<ISA>(row, act),
        GGML_TYPE_Q2_K => dot_q2_k::<ISA>(row, act),
        GGML_TYPE_Q3_K => dot_q3_k::<ISA>(row, act),
        GGML_TYPE_Q4_K => dot_q4_k::<ISA>(row, act),
        GGML_TYPE_Q5_K => dot_q5_k::<ISA>(row, act),
        GGML_TYPE_Q6_K => dot_q6_k::<ISA>(row, act),
        GGML_TYPE_IQ4_XS => dot_iq4_xs::<ISA>(row, act),
        GGML_TYPE_PQ2_0 => dot_pq2_0::<ISA>(row, act),
        GGML_TYPE_PTQ1_0 => dot_ptq1_0::<ISA>(row, act),
        other => panic!("vecdot::dot_row called for unsupported ggml_type {other}"),
    }
}

/// Activation sum over the 32-element block `b`, from the per-[`GROUP`] sums.
#[inline(always)]
fn block_sum(act: &ActQ8, b: usize) -> i32 {
    act.sums[b * 2] + act.sums[b * 2 + 1]
}

// =====================================================================
// Float weights: widen and FMA, no activation quantization.
//
// `F32`, `F16` and `BF16` are the types this module never had a kernel for,
// and they are a different problem from the quantized ones. There is no
// int8 form of the weights to dot against, and inventing one would quantize
// weights that the file stores in full precision — an accuracy change ggml
// does not make. So these keep `f32` arithmetic and fix the two things that
// were actually wrong with the fallback:
//
// 1. `matmul_dequant` widens each row into a **freshly allocated
//    `Vec<f32>`** (`QuantMatrix::row` -> `quant::dequantize`). Same defect
//    the fused path removed long ago; it was never removed here.
// 2. `tensor::dot` has an AVX2 path and **no NEON path at all**, so on
//    aarch64 it is `dot_scalar` — one accumulator, which makes the loop a
//    serial dependency chain of `in_dim` FMAs at ~4 cycles of latency each.
//
// Both go away by widening one 16-element block at a time into a stack
// array and accumulating into eight lanes, which is two independent 4-wide
// chains. Nothing is materialized to memory and nothing is allocated.
//
// This is why `BF16` was the last unfused type: it is 0.9-1.1% of the two
// `gemma-4` files (`per_layer_model_proj`), and those were the lowest fused
// decode ratios in the sweep.
// =====================================================================

/// Elements widened per iteration of the float dot. 16 covers two 4-wide
/// NEON registers' worth of accumulators with a chain depth of two.
const FLOAT_BLOCK: usize = 16;

/// Whether [`dot_row_f32`] has a kernel for `ggml_type`.
///
/// Unlike [`supports`] there is no `in_dim` condition: these types have no
/// block structure, so any row length works, tail included.
///
/// This is deliberately *not* folded into [`supports`]. That one promises a
/// kernel taking [`ActQ8`], and these take `&[f32]` — a caller that confused
/// the two would compile and then quantize activations it must not quantize.
pub fn supports_float(ggml_type: u32) -> bool {
    matches!(ggml_type, GGML_TYPE_F32 | GGML_TYPE_F16 | GGML_TYPE_BF16)
}

/// `sum_i weight_row[i] * x[i]` for one token, straight from the float
/// bytes. `ggml_type` must have passed [`supports_float`].
///
/// Activations stay `f32`: unlike the quantized kernels above, this adds no
/// error of its own beyond `f32` summation order.
pub fn dot_row_f32(ggml_type: u32, row: &[u8], x: &[f32]) -> f32 {
    match ggml_type {
        GGML_TYPE_F32 => dot_float::<FKIND_F32>(row, x),
        GGML_TYPE_F16 => dot_float::<FKIND_F16>(row, x),
        GGML_TYPE_BF16 => dot_float::<FKIND_BF16>(row, x),
        other => panic!("vecdot::dot_row_f32 called for unsupported ggml_type {other}"),
    }
}

const FKIND_F32: u8 = 0;
const FKIND_F16: u8 = 1;
const FKIND_BF16: u8 = 2;

/// Bytes per weight for a float kind.
#[inline(always)]
const fn float_stride<const KIND: u8>() -> usize {
    if KIND == FKIND_F32 { 4 } else { 2 }
}

/// One weight, widened to `f32`.
///
/// `BF16` is the cheap one — it is the *top* 16 bits of the `f32`, so
/// widening is a shift with no exponent rebiasing, which is why ggml can
/// convert it with a byte shuffle.
#[inline(always)]
fn widen_float<const KIND: u8>(row: &[u8], i: usize) -> f32 {
    let at = i * float_stride::<KIND>();
    match KIND {
        FKIND_F32 => f32::from_le_bytes([row[at], row[at + 1], row[at + 2], row[at + 3]]),
        FKIND_BF16 => f32::from_bits((u16::from_le_bytes([row[at], row[at + 1]]) as u32) << 16),
        _ => read_f16(row, at),
    }
}

/// Both operands are walked with [`slice::chunks_exact`] so every index the
/// inner loops perform is provably in bounds and LLVM will vectorize them.
/// Indexing `row`/`x` against a separately-derived length instead costs a
/// compare-and-branch per element and blocks the vectorizer outright — the
/// mistake is easy to make and invisible without `objdump`; `tensor::dot`
/// documents the same trap after hitting it.
fn dot_float<const KIND: u8>(row: &[u8], x: &[f32]) -> f32 {
    const LANES: usize = 8;
    let stride = float_stride::<KIND>();
    debug_assert!(row.len() >= x.len() * stride);

    // Eight lanes, not one: two independent 4-wide chains cover the FMA
    // latency a single accumulator would serialize on.
    let mut acc = [0f32; LANES];
    let mut w = [0f32; FLOAT_BLOCK];

    let mut rows = row[..x.len() * stride].chunks_exact(FLOAT_BLOCK * stride);
    // Paired with `rows` above through `by_ref`, and the tail is read from
    // `ChunksExact::remainder` — neither exists on `as_chunks`.
    #[allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]
    let mut xs = x.chunks_exact(FLOAT_BLOCK);
    for (rb, xb) in rows.by_ref().zip(xs.by_ref()) {
        for (j, slot) in w.iter_mut().enumerate() {
            *slot = widen_float::<KIND>(rb, j);
        }
        for half in 0..FLOAT_BLOCK / LANES {
            for lane in 0..LANES {
                acc[lane] += w[half * LANES + lane] * xb[half * LANES + lane];
            }
        }
    }

    let mut total: f32 = acc.iter().sum();
    // Tail: these types carry no block structure, so `in_dim` need not be a
    // multiple of anything and a row can end mid-block.
    let rb = rows.remainder();
    for (j, xv) in xs.remainder().iter().enumerate() {
        total += widen_float::<KIND>(rb, j) * xv;
    }
    total
}

/// Widens one float row (`F32`/`F16`/`BF16` bytes) to `f32`, for
/// [`gemm_f32_rows`]'s tile — done once per row per task, not per token.
pub fn widen_float_row(ggml_type: u32, row: &[u8], in_dim: usize, out: &mut Vec<f32>) {
    out.clear();
    out.reserve(in_dim);
    match ggml_type {
        GGML_TYPE_F32 => out.extend((0..in_dim).map(|i| widen_float::<FKIND_F32>(row, i))),
        GGML_TYPE_F16 => out.extend((0..in_dim).map(|i| widen_float::<FKIND_F16>(row, i))),
        GGML_TYPE_BF16 => out.extend((0..in_dim).map(|i| widen_float::<FKIND_BF16>(row, i))),
        other => panic!("vecdot::widen_float_row called for unsupported ggml_type {other}"),
    }
}

/// Rows per [`gemm_f32_rows`] tile.
pub const F32_ROWS: usize = 4;
/// Tokens per [`gemm_f32_rows`] tile: 4 × 6 accumulators plus four weight
/// vectors and one activation vector is 29 of the 32 NEON registers — the
/// largest tile that does not spill. (4 × 8 needs 37 and spilled every
/// accumulator: 79 vector stores per 32 multiply-adds.) The portable tile
/// keeps the same shape: 24 eight-lane accumulators is within `AVX2`'s
/// sixteen registers only with spills, but the tile's shape is what the
/// callers' splits are sized to, and the same shape keeps them one code.
const F32_TOKENS: usize = 6;

/// Four `f32` rows against every token — the prefill kernel for the float
/// types, which have no `int8` form. `x` is `[n_tokens][in_dim]`
/// token-major; `out[r]` is row `r`'s `n_tokens`-long output.
///
/// [`dot_row_f32`] takes one row against one token and is load-bound: two
/// loads per multiply-add, and the row re-read from cache for every token.
/// A Qwen-Image VAE decode — 3×3 convolutions as `im2col` matmuls of 96 to
/// 384 output rows against tens of thousands of pixels — ran it at under
/// 2 G MAC/s per core. This holds a 4-row × 6-token tile of accumulators in
/// registers and walks `in_dim` four lanes at a time: 10 loads per 96
/// multiply-adds, each `vfmaq_f32` independent of the others, the lane sums
/// reduced once at the end. The tile is a separate function with every
/// loop bound a constant, which is what keeps the accumulators in registers
/// — a tile whose token count is a runtime value indexes them through
/// memory.
///
/// The `f32` result differs from [`dot_row_f32`]'s only in summation order
/// (four interleaved partial sums against eight), which is `f32` rounding
/// noise — the same seeded picture came out byte-identical through both.
pub fn gemm_f32_rows(w: [&[f32]; F32_ROWS], x: &[f32], in_dim: usize, out: [&mut [f32]; F32_ROWS]) {
    let n_tokens = out[0].len();
    debug_assert!(w.iter().all(|r| r.len() >= in_dim));
    debug_assert!(x.len() >= n_tokens * in_dim);
    debug_assert!(out.iter().all(|o| o.len() == n_tokens));
    let full = n_tokens / F32_TOKENS * F32_TOKENS;
    let mut t0 = 0;
    while t0 < full {
        // Safety: the tile reads `in_dim` floats from each of the four rows
        // and from tokens `t0..t0 + F32_TOKENS`, all inside the slices
        // checked above.
        #[cfg(target_arch = "aarch64")]
        let tile = unsafe { f32_tile(&w, x.as_ptr().add(t0 * in_dim), in_dim) };
        #[cfg(not(target_arch = "aarch64"))]
        let tile = f32_tile_portable(&w, &x[t0 * in_dim..], in_dim);
        for (r, row) in tile.iter().enumerate() {
            out[r][t0..t0 + F32_TOKENS].copy_from_slice(row);
        }
        t0 += F32_TOKENS;
    }
    // The last, short tile: one token at a time, four rows.
    for t in full..n_tokens {
        let xt = &x[t * in_dim..(t + 1) * in_dim];
        for r in 0..F32_ROWS {
            out[r][t] = dot_f32_slices(&w[r][..in_dim], xt);
        }
    }
}

/// One 4 × [`F32_TOKENS`] tile of [`gemm_f32_rows`]. `x` points at the
/// first of the tile's tokens, each `in_dim` floats long.
// Index loops rather than iterators, deliberately: with every bound a
// constant, LLVM unrolls them and keeps all 24 accumulators in registers
// (checked in the disassembly: 24 `fmla` per 10 loads and no spill); the
// iterator forms clippy prefers were not tested to do the same.
#[allow(clippy::needless_range_loop)]
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn f32_tile(
    w: &[&[f32]; F32_ROWS],
    x: *const f32,
    in_dim: usize,
) -> [[f32; F32_TOKENS]; F32_ROWS] {
    use std::arch::aarch64::*;
    unsafe {
        let k4 = in_dim / 4 * 4;
        let mut acc = [[vdupq_n_f32(0.0); F32_TOKENS]; F32_ROWS];
        let mut k = 0;
        while k < k4 {
            let w0 = vld1q_f32(w[0].as_ptr().add(k));
            let w1 = vld1q_f32(w[1].as_ptr().add(k));
            let w2 = vld1q_f32(w[2].as_ptr().add(k));
            let w3 = vld1q_f32(w[3].as_ptr().add(k));
            for t in 0..F32_TOKENS {
                let xv = vld1q_f32(x.add(t * in_dim + k));
                acc[0][t] = vfmaq_f32(acc[0][t], w0, xv);
                acc[1][t] = vfmaq_f32(acc[1][t], w1, xv);
                acc[2][t] = vfmaq_f32(acc[2][t], w2, xv);
                acc[3][t] = vfmaq_f32(acc[3][t], w3, xv);
            }
            k += 4;
        }
        let mut out = [[0f32; F32_TOKENS]; F32_ROWS];
        for t in 0..F32_TOKENS {
            for r in 0..F32_ROWS {
                let mut sum = vaddvq_f32(acc[r][t]);
                for kk in k4..in_dim {
                    sum += w[r][kk] * *x.add(t * in_dim + kk);
                }
                out[r][t] = sum;
            }
        }
        out
    }
}

/// The portable 4 × [`F32_TOKENS`] tile: the same 24 accumulators as
/// [`f32_tile`], each eight lanes wide as plain arrays, which LLVM keeps in
/// vector registers where the target has them (`AVX2` is the `x86_64`
/// build's baseline). Every loop bound is a constant for the same reason as
/// the NEON tile's.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[allow(clippy::needless_range_loop)]
#[inline(always)]
fn f32_tile_portable(
    w: &[&[f32]; F32_ROWS],
    x: &[f32],
    in_dim: usize,
) -> [[f32; F32_TOKENS]; F32_ROWS] {
    const LANES: usize = 8;
    let k8 = in_dim / LANES * LANES;
    let mut acc = [[[0f32; LANES]; F32_TOKENS]; F32_ROWS];
    let mut k = 0;
    while k < k8 {
        let wv: [&[f32]; F32_ROWS] = [
            &w[0][k..k + LANES],
            &w[1][k..k + LANES],
            &w[2][k..k + LANES],
            &w[3][k..k + LANES],
        ];
        for t in 0..F32_TOKENS {
            let xv = &x[t * in_dim + k..t * in_dim + k + LANES];
            for r in 0..F32_ROWS {
                for l in 0..LANES {
                    acc[r][t][l] += wv[r][l] * xv[l];
                }
            }
        }
        k += LANES;
    }
    let mut out = [[0f32; F32_TOKENS]; F32_ROWS];
    for t in 0..F32_TOKENS {
        for r in 0..F32_ROWS {
            let mut sum = acc[r][t].iter().sum::<f32>();
            for kk in k8..in_dim {
                sum += w[r][kk] * x[t * in_dim + kk];
            }
            out[r][t] = sum;
        }
    }
    out
}

/// `sum_i w[i] * x[i]` over two `f32` slices of equal length, four lanes at
/// a time — the short-tile tail of [`gemm_f32_rows`].
#[cfg(target_arch = "aarch64")]
pub fn dot_f32_slices(w: &[f32], x: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    debug_assert_eq!(w.len(), x.len());
    let k4 = w.len() / 4 * 4;
    // Safety: NEON is baseline on aarch64; `k` stays below `k4 <= len`.
    let mut sum = unsafe {
        let mut acc = vdupq_n_f32(0.0);
        let mut k = 0;
        while k < k4 {
            acc = vfmaq_f32(
                acc,
                vld1q_f32(w.as_ptr().add(k)),
                vld1q_f32(x.as_ptr().add(k)),
            );
            k += 4;
        }
        vaddvq_f32(acc)
    };
    for kk in k4..w.len() {
        sum += w[kk] * x[kk];
    }
    sum
}

/// [`dot_f32_slices`] off `aarch64`: eight independent lane sums, which
/// LLVM vectorizes where the target can.
#[cfg(not(target_arch = "aarch64"))]
pub fn dot_f32_slices(w: &[f32], x: &[f32]) -> f32 {
    debug_assert_eq!(w.len(), x.len());
    let mut acc = [0f32; 8];
    let (wc, wr) = w.as_chunks::<8>();
    let (xc, xr) = x.as_chunks::<8>();
    for (wv, xv) in wc.iter().zip(xc) {
        for l in 0..8 {
            acc[l] += wv[l] * xv[l];
        }
    }
    let mut sum = acc.iter().sum::<f32>();
    for (a, b) in wr.iter().zip(xr) {
        sum += a * b;
    }
    sum
}

/// `block_q8_0`: `{ d: f16, qs: [i8; 32] }` — already `int8`, no unpack at all.
/// `Q8_0` against a q8 row, a 32-block per pass without a horizontal sum:
/// the block's `int8` weights against the activations with the
/// **weight's** sign folded into the activation (`vpmaddubsw` of `|w|`
/// and `sign(w)·x`) — that way round because a stored byte can be `-128`,
/// whose negation is itself, while a quantized activation never is —
/// widened to eight `i32` lanes by a `vpmaddwd` against ones, scaled by
/// the block's two `f32` scales into eight float lanes that are summed
/// once at the end of the row.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_q8_0_avx2(row: &[u8], act: &ActQ8) -> f32 {
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 2 + 32;
    unsafe {
        let ones = _mm256_set1_epi16(1);
        let mut accf = _mm256_setzero_ps();
        for (b, block) in row.as_chunks::<BLOCK_BYTES>().0.iter().enumerate() {
            let dw = read_f16(block, 0);
            let w = _mm256_loadu_si256(block.as_ptr().add(2) as *const __m256i);
            let x = _mm256_loadu_si256(act.q.as_ptr().add(b * 32) as *const __m256i);
            let p = _mm256_maddubs_epi16(_mm256_abs_epi8(w), _mm256_sign_epi8(x, w));
            let sum = _mm256_madd_epi16(p, ones);
            accf = _mm256_fmadd_ps(_mm256_set1_ps(dw * act.d[b]), _mm256_cvtepi32_ps(sum), accf);
        }
        hsum_ps_avx2(accf)
    }
}

/// `Q4_0`: [`dot_q8_0_avx2`] with the block's 32 weights unpacked from 16
/// bytes — the low nibbles are the first sixteen elements, the high the
/// second — less 8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_q4_0_avx2(row: &[u8], act: &ActQ8) -> f32 {
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 2 + 16;
    unsafe {
        let ones = _mm256_set1_epi16(1);
        let m4 = _mm256_set1_epi8(0x0F);
        let eight = _mm256_set1_epi8(8);
        let mut accf = _mm256_setzero_ps();
        for (b, block) in row.as_chunks::<BLOCK_BYTES>().0.iter().enumerate() {
            let dw = read_f16(block, 0);
            let q = _mm_loadu_si128(block.as_ptr().add(2) as *const __m128i);
            // Low nibbles in the low lane, high nibbles in the high lane.
            let both = _mm256_set_m128i(_mm_srli_epi16(q, 4), q);
            let w = _mm256_sub_epi8(_mm256_and_si256(both, m4), eight);
            let x = _mm256_loadu_si256(act.q.as_ptr().add(b * 32) as *const __m256i);
            let p = _mm256_maddubs_epi16(_mm256_abs_epi8(x), _mm256_sign_epi8(w, x));
            let sum = _mm256_madd_epi16(p, ones);
            accf = _mm256_fmadd_ps(_mm256_set1_ps(dw * act.d[b]), _mm256_cvtepi32_ps(sum), accf);
        }
        hsum_ps_avx2(accf)
    }
}

/// `Q5_0` and `Q5_1` (`WITH_MIN`) a 32-block per pass: the nibbles with
/// the fifth bit set from `qh` as [`unpack_block_q5_bits_avx2`] sets it,
/// left **unsigned** (0..31) so the multiply is a plain `vpmaddubsw`
/// against the activation, and the zero point applied afterwards as a
/// scalar against the block's activation sum — `-16` for `Q5_0`, the
/// block's `m` for `Q5_1` — which is exact because both are constant
/// across the block. Eight float lanes across the row, one sum at the end.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_q5_x_avx2<const WITH_MIN: bool>(row: &[u8], act: &ActQ8) -> f32 {
    use std::arch::x86_64::*;
    let block_bytes = if WITH_MIN { 2 + 2 + 4 + 16 } else { 2 + 4 + 16 };
    unsafe {
        let ones = _mm256_set1_epi16(1);
        let nibble = _mm_set1_epi8(0x0F);
        let byte_of_lane = _mm256_setr_epi8(
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
            3, 3, 3,
        );
        let bit_of_lane = _mm256_setr_epi8(
            1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64,
            -128, 1, 2, 4, 8, 16, 32, 64, -128,
        );
        let sixteen = _mm256_set1_epi8(0x10);
        let mut accf = _mm256_setzero_ps();
        let mut total_min = 0f32;
        for (b, block) in row.chunks_exact(block_bytes).enumerate() {
            let dw = read_f16(block, 0);
            let (m, at) = if WITH_MIN {
                (read_f16(block, 2), 4)
            } else {
                (-16.0 * dw, 2)
            };
            let qh = u32::from_le_bytes([block[at], block[at + 1], block[at + 2], block[at + 3]]);
            let v = _mm_loadu_si128(block.as_ptr().add(at + 4) as *const __m128i);
            let lo = _mm_and_si128(v, nibble);
            let hi = _mm_and_si128(_mm_srli_epi16::<4>(v), nibble);
            let nibbles = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi);
            let bits = _mm256_set1_epi32(qh as i32);
            let bytes = _mm256_shuffle_epi8(bits, byte_of_lane);
            let set = _mm256_cmpeq_epi8(_mm256_and_si256(bytes, bit_of_lane), bit_of_lane);
            let w = _mm256_or_si256(nibbles, _mm256_and_si256(set, sixteen));
            let x = _mm256_loadu_si256(act.q.as_ptr().add(b * 32) as *const __m256i);
            let p = _mm256_maddubs_epi16(w, x);
            let sum = _mm256_madd_epi16(p, ones);
            accf = _mm256_fmadd_ps(_mm256_set1_ps(dw * act.d[b]), _mm256_cvtepi32_ps(sum), accf);
            total_min += act.d[b] * m * block_sum(act, b) as f32;
        }
        hsum_ps_avx2(accf) + total_min
    }
}

fn dot_q8_0<const ISA: u8>(row: &[u8], act: &ActQ8) -> f32 {
    const BLOCK_BYTES: usize = 2 + 32;
    let mut total = 0f32;
    for (b, block) in row.as_chunks::<BLOCK_BYTES>().0.iter().enumerate() {
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
    for (b, block) in row.as_chunks::<BLOCK_BYTES>().0.iter().enumerate() {
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
    for (b, block) in row.as_chunks::<BLOCK_BYTES>().0.iter().enumerate() {
        let dw = read_f16(block, 0);
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        unpack_block_q5_0_isa::<ISA>(qh, &block[6..], &mut w);
        total += dw * act.d[b] * dot32::<ISA>(&w, &act.q[b * 32..]) as f32;
    }
    total
}

/// `block_q4_0`, 32 elements: low nibbles then high nibbles, biased by -8
/// into a signed `int8`. Symmetric, so no correction term.
fn dot_q4_0<const ISA: u8>(row: &[u8], act: &ActQ8) -> f32 {
    const BLOCK_BYTES: usize = 2 + 16;
    let mut total = 0f32;
    let mut w = [0i8; 32];
    for (b, block) in row.as_chunks::<BLOCK_BYTES>().0.iter().enumerate() {
        let dw = read_f16(block, 0);
        unpack_block_q4_0_isa::<ISA>(&block[2..], &mut w);
        total += dw * act.d[b] * dot32::<ISA>(&w, &act.q[b * 32..]) as f32;
    }
    total
}

/// `block_pq2_0`, 128 elements: one `f16` scale over four 32-element
/// activation blocks. Symmetric (`-1..=2`), so the four integer dots are
/// scaled by their own activation scale and summed under the one weight
/// scale — the same shape as [`dot_q4_0`] with the weight scale hoisted
/// across four blocks. Unpacking goes through `quant::unpack_block_pq2_0`
/// so the bit layout is written once.
fn dot_pq2_0<const ISA: u8>(row: &[u8], act: &ActQ8) -> f32 {
    let mut total = 0f32;
    let mut w = [0i8; QK_PRISM];
    for (b, block) in row.as_chunks::<PQ2_0_BLOCK_BYTES>().0.iter().enumerate() {
        let dw = read_f16(block, 0);
        unpack_block_pq2_0(&block[2..], &mut w);
        total += dw * prism_block_dot::<ISA>(&w, act, b);
    }
    total
}

/// `block_ptq1_0`, 128 elements: `TQ1_0`'s trit packing with the scale
/// last. Same accumulation as [`dot_pq2_0`]; only the unpack differs.
fn dot_ptq1_0<const ISA: u8>(row: &[u8], act: &ActQ8) -> f32 {
    const QS: usize = PTQ1_0_BLOCK_BYTES - 4;
    let mut total = 0f32;
    let mut w = [0i8; QK_PRISM];
    for (b, block) in row.as_chunks::<PTQ1_0_BLOCK_BYTES>().0.iter().enumerate() {
        let dw = read_f16(block, QS + 2);
        unpack_block_ptq1_0(&block[..QS], &block[QS..QS + 2], &mut w);
        total += dw * prism_block_dot::<ISA>(&w, act, b);
    }
    total
}

/// `sum_j d_act[j] · dot32(w[j], act[j])` over the four activation blocks
/// that make up Prism block `b` — the per-128 weight scale is the caller's.
#[inline(always)]
fn prism_block_dot<const ISA: u8>(w: &[i8; QK_PRISM], act: &ActQ8, b: usize) -> f32 {
    let mut sum = 0f32;
    for s in 0..QK_PRISM / ACT_BLOCK {
        let ab = b * (QK_PRISM / ACT_BLOCK) + s;
        sum += act.d[ab] * dot32::<ISA>(&w[s * ACT_BLOCK..], &act.q[ab * ACT_BLOCK..]) as f32;
    }
    sum
}

/// `block_q5_1`, 32 elements: `Q5_0`'s bit layout with `value = d*q + m`.
/// The `+m` is applied against the block's activation sum, the same way
/// [`dot_q4_k`] applies `-dmin*m` — note the opposite sign.
fn dot_q5_1<const ISA: u8>(row: &[u8], act: &ActQ8) -> f32 {
    const BLOCK_BYTES: usize = 2 + 2 + 4 + 16;
    let mut total = 0f32;
    let mut w = [0i8; 32];
    for (b, block) in row.as_chunks::<BLOCK_BYTES>().0.iter().enumerate() {
        let dw = read_f16(block, 0);
        let m = read_f16(block, 2);
        let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        unpack_block_q5_1_isa::<ISA>(qh, &block[8..], &mut w);
        let isum = dot32::<ISA>(&w, &act.q[b * 32..]);
        total += act.d[b] * (dw * isum as f32 + m * block_sum(act, b) as f32);
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
    for block in row.as_chunks::<BLOCK_BYTES>().0 {
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

/// `block_q5_K`, 256 elements — [`dot_q4_k`] with the 5th bit taken from the
/// `qh` plane. Same asymmetric form (`value = d*sc*q - dmin*m`), so the min
/// term is likewise applied against the block's activation sum rather than
/// folded into the weight; see [`unpack_q5_k`] for the bit mapping.
fn dot_q5_k<const ISA: u8>(row: &[u8], act: &ActQ8) -> f32 {
    const BLOCK_BYTES: usize = 2 + 2 + 12 + 32 + 128;
    let mut total = 0f32;
    let mut sb = 0usize;
    for block in row.as_chunks::<BLOCK_BYTES>().0 {
        let d = read_f16(block, 0);
        let dmin = read_f16(block, 2);
        let scales = &block[4..16];
        let qh = &block[16..48];
        let qs = &block[48..];
        for g in 0..4 {
            let bytes = &qs[g * 32..g * 32 + 32];
            for (half, shift) in [(0usize, 0u32), (1, 4)] {
                let (sc, m) = get_scale_min_k4(g * 2 + half, scales);
                let hi_mask = 1u8 << (g * 2 + half);
                let b = sb + half;
                let mut w = [0i8; 32];
                for (j, &byte) in bytes.iter().enumerate() {
                    let hi = if qh[j] & hi_mask != 0 { 16 } else { 0 };
                    w[j] = (((byte >> shift) & 0x0F) + hi) as i8;
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
    for block in row.as_chunks::<BLOCK_BYTES>().0 {
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

// =====================================================================
// K-quant decode with a per-super-block activation scale.
//
// [`ActQ8`] carries an `f32` scale per **32** activations. For a per-16
// K-quant that forces the accumulator out of the integer domain eight times
// per super-block, because each 32-element block has a different scale and
// each 16-element group has a different weight scale.
//
// ggml does not pay this: `block_q8_K` has a single `d` per **256**, so the
// whole super-block accumulates in `i32` — a weight scale is an integer, so
// `sc * partial` stays integral — and converts once. orangu already has that
// activation shape as [`ActQ8K`] and already uses it for the K-quant *prefill*
// GEMM; decode never got it.
//
// Measured in isolation with `doc/perf/tiny-kernel` (512 rows, in_dim 2048,
// single thread, best of 7): **1.24x for `Q3_K` and 1.45x for `Q2_K`** against
// the per-32 form. Two other candidates were measured there and rejected
// first — a NEON unpack (1.02x/1.06x: LLVM already vectorizes the scalar
// loops, as it does for `Q4_K`) and removing the per-16 horizontal reduction
// (a further 1.07x/1.08x). The activation format is the one that matters.

/// Activations for the K-quant decode path: `int8` with **one scale per
/// [`SUPER_BLOCK`]**, plus per-[`GROUP`] sums for the asymmetric min.
///
/// Flat rather than tiled by token, unlike [`ActQ8K`] — decode is one token, so
/// there is nothing to interleave.
pub struct ActQ8KRow {
    q: Vec<i8>,
    /// Per-[`SUPER_BLOCK`].
    d: Vec<f32>,
    /// Per-[`GROUP`] `sum(q)`, for `Q2_K`'s min term.
    sums: Vec<i32>,
    /// Per-[`QK_PRISM`] `sum(q)` — the `-1` bias of a ternary block folded
    /// into one subtraction. Summed once here rather than from `sums` per
    /// (row, block), which was a fifth of the Prism decode kernel's time.
    /// Read only by the `sdot` row kernels; elsewhere the block dot unpacks
    /// the offset weights and needs no sum.
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    prism_sums: Vec<i32>,
}

/// Quantizes one token's activations for [`dot_k_row`]. `x.len()` must be a
/// multiple of [`SUPER_BLOCK`] — guaranteed by [`supports_k_row`].
pub fn quantize_act_k_row(x: &[f32]) -> ActQ8KRow {
    debug_assert_eq!(x.len() % SUPER_BLOCK, 0);
    let n_super = x.len() / SUPER_BLOCK;
    let mut q = vec![0i8; x.len()];
    let mut d = vec![0f32; n_super];
    let mut sums = vec![0i32; x.len() / GROUP];
    let quantize_super = |chunk: &[f32], q: &mut [i8], d: &mut f32, sums: &mut [i32]| {
        let amax = chunk.iter().fold(0f32, |m, v| m.max(v.abs()));
        // A super-block of exact zeros has no scale; leave it 0 so `q * d`
        // reproduces 0 rather than NaN. Same contract as `quantize_act`.
        let scale = amax / 127.0;
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        for (g, group) in chunk.as_chunks::<GROUP>().0.iter().enumerate() {
            let mut sum = 0i32;
            for (i, &v) in group.iter().enumerate() {
                let qi = (v * inv).round().clamp(-127.0, 127.0) as i8;
                q[g * GROUP + i] = qi;
                sum += qi as i32;
            }
            sums[g] = sum;
        }
        *d = scale;
    };
    // Serial on purpose: this is one decode row, and fanning its
    // super-blocks across the pool was measured to cost more in fork-join
    // than the ~30 µs it spread (27B `PTQ1_0`, 8 threads).
    for (s, chunk) in x.as_chunks::<SUPER_BLOCK>().0.iter().enumerate() {
        quantize_super(
            chunk,
            &mut q[s * SUPER_BLOCK..(s + 1) * SUPER_BLOCK],
            &mut d[s],
            &mut sums[s * (SUPER_BLOCK / GROUP)..(s + 1) * (SUPER_BLOCK / GROUP)],
        );
    }
    let prism_sums = sums
        .as_chunks::<{ QK_PRISM / GROUP }>()
        .0
        .iter()
        .map(|g| g.iter().sum())
        .collect();
    ActQ8KRow {
        q,
        d,
        sums,
        prism_sums,
    }
}

/// Whether [`dot_k_row`] handles this type.
///
/// **Every 256-element super-block type — a strict superset of
/// [`supports_k`].** The extra member is `Q2_K`, and the asymmetry is real
/// rather than an oversight: `Q2_K`'s min is per 16, which the prefill GEMM
/// cannot serve because [`ActQ8K::bsum`] is per 32, but which decode computes
/// directly from [`ActQ8KRow::sums`]. So `Q2_K` gets the integer accumulation
/// at decode and not at prefill.
///
/// This began as `Q2_K`/`Q3_K` only, so the change could be measured on four
/// files before altering the arithmetic of every K-quant model in the
/// reference set; the extension to the other four was then measured and kept.
///
/// **The bound that has to hold for each type** is that a whole super-block's
/// integer accumulation stays inside `i32`. Worst case per type, at `|x| <=
/// 127`:
///
/// | type | scale | quant | per super-block |
/// |---|---:|---:|---:|
/// | `Q2_K` | 15 | 3 | 1.5e6 |
/// | `Q3_K` | 32 | 4 | 4.2e6 |
/// | `Q4_K` | 63 | 15 | 3.1e7 |
/// | `Q5_K` | 63 | 31 | 6.3e7 |
/// | `Q6_K` | 127 | 32 | 1.4e8 |
/// | `IQ4_XS` | 32 | 127 | 1.4e8 |
///
/// All are at least an order of magnitude inside `i32::MAX` (2.1e9), the
/// tightest being `Q6_K` and `IQ4_XS` at ~15x margin.
pub fn supports_k_row(ggml_type: u32, in_dim: usize) -> bool {
    // The Prism ternary types are not super-block types, but they take this
    // path for the same reason the K-quants do: one weight scale per 128 and
    // one activation scale per 256 let a whole block accumulate in `i32` —
    // and, on `sdot` hardware, let the trit planes be dotted straight from
    // the packed bytes with no unpacked row in between. Bound: `|w| <= 2`,
    // `|x| <= 127`, 128 terms — 3.3e4 per block, nowhere near `i32`.
    matches!(
        ggml_type,
        GGML_TYPE_Q2_K
            | GGML_TYPE_Q3_K
            | GGML_TYPE_Q4_K
            | GGML_TYPE_Q5_K
            | GGML_TYPE_Q6_K
            | GGML_TYPE_IQ4_XS
            | GGML_TYPE_PQ2_0
            | GGML_TYPE_PTQ1_0
    ) && in_dim.is_multiple_of(SUPER_BLOCK)
}

/// Activation sum over the 32-element block `b`, from the per-[`GROUP`] sums.
#[inline(always)]
fn k_row_block_sum(act: &ActQ8KRow, b: usize) -> i32 {
    act.sums[b * 2] + act.sums[b * 2 + 1]
}

/// Whether the super-block-wide `sdot` row kernels for `Q4_K`/`Q6_K` are
/// used — `ORANGU_SDOT_K_ROWS=0` keeps the per-32 generic form, for
/// measuring the difference. Default on.
#[cfg(target_arch = "aarch64")]
fn sdot_k_rows_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| super::env::flag_on_unless_disabled("ORANGU_SDOT_K_ROWS"))
}

/// Decode dot for the two-bit family, accumulating across the whole
/// super-block in `i32`.
pub fn dot_k_row(ggml_type: u32, row: &[u8], act: &ActQ8KRow) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if have_dotprod() {
        return dot_k_row_impl::<ISA_DOTPROD>(ggml_type, row, act);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if have_vnni() {
            // Safety: `have_vnni` verified the features.
            return unsafe { dot_k_row_vnni(ggml_type, row, act) };
        }
        if is_x86_feature_detected!("avx2") {
            // Safety: guarded by the runtime feature check above.
            return unsafe { dot_k_row_avx2(ggml_type, row, act) };
        }
    }
    dot_k_row_impl::<ISA_BASELINE>(ggml_type, row, act)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512vnni,avx512vl,avx2,ssse3")]
unsafe fn dot_k_row_vnni(ggml_type: u32, row: &[u8], act: &ActQ8KRow) -> f32 {
    dot_k_row_impl::<ISA_VNNI>(ggml_type, row, act)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_k_row_avx2(ggml_type: u32, row: &[u8], act: &ActQ8KRow) -> f32 {
    dot_k_row_impl::<ISA_AVX2>(ggml_type, row, act)
}

fn dot_k_row_impl<const ISA: u8>(ggml_type: u32, row: &[u8], act: &ActQ8KRow) -> f32 {
    // The three types a `Q4_K_M` file is made of take the super-block-wide
    // AVX2 kernels: the per-32 unpack-then-dot form below costs a horizontal
    // sum and a scalar nibble loop per 32 elements, and measured on a 12B
    // model's host layers it read weights at 24 GB/s where the block-wide
    // form reads at the memory's own rate.
    #[cfg(target_arch = "x86_64")]
    if ISA == ISA_AVX2 || ISA == ISA_VNNI {
        // Safety: both ISAs imply AVX2, and the callers that instantiate
        // them checked it at run time.
        match ggml_type {
            GGML_TYPE_Q4_K => return unsafe { dot_k_row_q4_k_avx2(row, act) },
            GGML_TYPE_Q5_K => return unsafe { dot_k_row_q5_k_avx2(row, act) },
            GGML_TYPE_Q6_K => return unsafe { dot_k_row_q6_k_avx2(row, act) },
            _ => {}
        }
    }
    // The two types a `Q4_K_M` file's layers are made of take the
    // super-block-wide `sdot` kernels, the aarch64 counterpart of the AVX2
    // ones above.
    #[cfg(target_arch = "aarch64")]
    if ISA == ISA_DOTPROD && sdot_k_rows_enabled() {
        // Safety: `ISA_DOTPROD` is instantiated only after `have_dotprod`.
        match ggml_type {
            GGML_TYPE_Q4_K => return unsafe { dot_k_row_q4_k_sdot(row, act) },
            GGML_TYPE_Q6_K => return unsafe { dot_k_row_q6_k_sdot(row, act) },
            _ => {}
        }
    }
    match ggml_type {
        GGML_TYPE_Q2_K => dot_k_row_q2_k::<ISA>(row, act),
        GGML_TYPE_Q3_K => dot_k_row_q3_k::<ISA>(row, act),
        GGML_TYPE_Q4_K => dot_k_row_q4_k::<ISA>(row, act),
        GGML_TYPE_Q5_K => dot_k_row_q5_k::<ISA>(row, act),
        GGML_TYPE_Q6_K => dot_k_row_q6_k::<ISA>(row, act),
        GGML_TYPE_IQ4_XS => dot_k_row_iq4_xs::<ISA>(row, act),
        GGML_TYPE_PQ2_0 => dot_k_row_prism::<ISA, false>(row, act),
        GGML_TYPE_PTQ1_0 => dot_k_row_prism::<ISA, true>(row, act),
        other => panic!("vecdot::dot_k_row called for unsupported ggml_type {other}"),
    }
}

/// `Q4_K` against a q8 row, one **super-block per pass** in AVX2: the four
/// 32-byte quant runs are split into their low and high nibbles as bytes
/// (unsigned, 0..15), each multiplied against its 32 activations with
/// `vpmaddubsw` (u8 × i8 pairs into i16 — at most 2 × 15 × 127, no
/// saturation), widened by its 6-bit scale with `vpmaddwd`, and summed in
/// one eight-lane `i32` accumulator across the super-block. One horizontal
/// sum per super-block instead of one per 32 elements, and no unpacked
/// `i8` array between the bytes and the multiply. The min term is the
/// scalar it always was: eight scales times eight precomputed activation
/// sums.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_k_row_q4_k_avx2(row: &[u8], act: &ActQ8KRow) -> f32 {
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 2 + 2 + 12 + 128;
    unsafe {
        let m4 = _mm256_set1_epi8(0x0F);
        // The dot as eight float lanes across the whole row, summed once at
        // the end; the min term as a scalar beside it.
        let mut accf = _mm256_setzero_ps();
        let mut total_min = 0f32;
        for (s, block) in row.as_chunks::<BLOCK_BYTES>().0.iter().enumerate() {
            let d = read_f16(block, 0);
            let dmin = read_f16(block, 2);
            let scales = &block[4..16];
            let qs = block.as_ptr().add(16);
            let x = act.q.as_ptr().add(s * SUPER_BLOCK);
            let mut sc = [0i16; 8];
            let mut imin = 0i32;
            for (j, sc_j) in sc.iter_mut().enumerate() {
                let (a, m) = get_scale_min_k4(j, scales);
                *sc_j = a as i16;
                imin += m as i32 * k_row_block_sum(act, s * SUBS + j);
            }
            let mut acc = _mm256_setzero_si256();
            for g in 0..4 {
                let q = _mm256_loadu_si256(qs.add(g * 32) as *const __m256i);
                let lo = _mm256_and_si256(q, m4);
                let hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
                let x0 = _mm256_loadu_si256(x.add(g * 64) as *const __m256i);
                let x1 = _mm256_loadu_si256(x.add(g * 64 + 32) as *const __m256i);
                let p0 = _mm256_maddubs_epi16(lo, x0);
                let p1 = _mm256_maddubs_epi16(hi, x1);
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p0, _mm256_set1_epi16(sc[2 * g])));
                acc =
                    _mm256_add_epi32(acc, _mm256_madd_epi16(p1, _mm256_set1_epi16(sc[2 * g + 1])));
            }
            accf = _mm256_fmadd_ps(_mm256_set1_ps(act.d[s] * d), _mm256_cvtepi32_ps(acc), accf);
            total_min += act.d[s] * dmin * imin as f32;
        }
        hsum_ps_avx2(accf) - total_min
    }
}

/// `Q5_K`: [`dot_k_row_q4_k_avx2`] with the fifth bit or-ed in from the
/// `qh` plane — bit `2g + half` of each `qh` byte, moved to bit 4.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_k_row_q5_k_avx2(row: &[u8], act: &ActQ8KRow) -> f32 {
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 2 + 2 + 12 + 32 + 128;
    unsafe {
        let m4 = _mm256_set1_epi8(0x0F);
        let one = _mm256_set1_epi8(1);
        let mut accf = _mm256_setzero_ps();
        let mut total_min = 0f32;
        for (s, block) in row.as_chunks::<BLOCK_BYTES>().0.iter().enumerate() {
            let d = read_f16(block, 0);
            let dmin = read_f16(block, 2);
            let scales = &block[4..16];
            let qh = _mm256_loadu_si256(block.as_ptr().add(16) as *const __m256i);
            let qs = block.as_ptr().add(48);
            let x = act.q.as_ptr().add(s * SUPER_BLOCK);
            let mut sc = [0i16; 8];
            let mut imin = 0i32;
            for (j, sc_j) in sc.iter_mut().enumerate() {
                let (a, m) = get_scale_min_k4(j, scales);
                *sc_j = a as i16;
                imin += m as i32 * k_row_block_sum(act, s * SUBS + j);
            }
            let mut acc = _mm256_setzero_si256();
            for g in 0..4 {
                let q = _mm256_loadu_si256(qs.add(g * 32) as *const __m256i);
                // The fifth bits of the two halves: bits 2g and 2g+1 of qh,
                // each brought to bit 4 (a byte shift right then left, as
                // 16-bit shifts on a masked value).
                let h0 = _mm256_slli_epi16(
                    _mm256_and_si256(_mm256_srl_epi16(qh, _mm_cvtsi32_si128((2 * g) as i32)), one),
                    4,
                );
                let h1 = _mm256_slli_epi16(
                    _mm256_and_si256(
                        _mm256_srl_epi16(qh, _mm_cvtsi32_si128((2 * g + 1) as i32)),
                        one,
                    ),
                    4,
                );
                let lo = _mm256_or_si256(_mm256_and_si256(q, m4), h0);
                let hi = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q, 4), m4), h1);
                let x0 = _mm256_loadu_si256(x.add(g * 64) as *const __m256i);
                let x1 = _mm256_loadu_si256(x.add(g * 64 + 32) as *const __m256i);
                let p0 = _mm256_maddubs_epi16(lo, x0);
                let p1 = _mm256_maddubs_epi16(hi, x1);
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p0, _mm256_set1_epi16(sc[2 * g])));
                acc =
                    _mm256_add_epi32(acc, _mm256_madd_epi16(p1, _mm256_set1_epi16(sc[2 * g + 1])));
            }
            accf = _mm256_fmadd_ps(_mm256_set1_ps(act.d[s] * d), _mm256_cvtepi32_ps(acc), accf);
            total_min += act.d[s] * dmin * imin as f32;
        }
        hsum_ps_avx2(accf) - total_min
    }
}

/// `Q6_K` one super-block per pass in AVX2. Each of the eight 32-element
/// runs is the low or high nibble of a `ql` half joined with two bits of
/// `qh`, less 32 — signed, so the multiply is the `vpmaddubsw` of the
/// *activation's* magnitude against the weight with the activation's sign
/// folded in (`|x| · sign(x)·w == x·w`), the ggml arrangement. The scales
/// are per 16, so each run's i16 pair sums are widened by two scales in
/// one `vpmaddwd` against a lane-interleaved scale vector.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_k_row_q6_k_avx2(row: &[u8], act: &ActQ8KRow) -> f32 {
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 128 + 64 + 16 + 2;
    unsafe {
        let m4 = _mm256_set1_epi8(0x0F);
        let m3 = _mm256_set1_epi8(3);
        let bias = _mm256_set1_epi8(32);
        let mut accf = _mm256_setzero_ps();
        for (s, block) in row.as_chunks::<BLOCK_BYTES>().0.iter().enumerate() {
            let ql = block.as_ptr();
            let qh = block.as_ptr().add(128);
            let sc = &block[192..208];
            let d = read_f16(block, 208);
            let x = act.q.as_ptr().add(s * SUPER_BLOCK);
            let mut acc = _mm256_setzero_si256();
            let mut r = 0usize;
            for h in 0..2 {
                let qhv = _mm256_loadu_si256(qh.add(h * 32) as *const __m256i);
                for &(ql_add, hshift, high) in Q6K_RUNS.iter() {
                    let qlv = _mm256_loadu_si256(ql.add(h * 64 + ql_add) as *const __m256i);
                    let nib = if high {
                        _mm256_and_si256(_mm256_srli_epi16(qlv, 4), m4)
                    } else {
                        _mm256_and_si256(qlv, m4)
                    };
                    let hb = _mm256_slli_epi16(
                        _mm256_and_si256(_mm256_srl_epi16(qhv, _mm_cvtsi32_si128(hshift)), m3),
                        4,
                    );
                    let w = _mm256_sub_epi8(_mm256_or_si256(nib, hb), bias);
                    let xv = _mm256_loadu_si256(x.add(r * 32) as *const __m256i);
                    // u8 × i8 with the sign on the weight side.
                    let p = _mm256_maddubs_epi16(_mm256_abs_epi8(xv), _mm256_sign_epi8(w, xv));
                    // Sixteen i16 pair sums: lanes 0..7 belong to the run's
                    // first 16 elements, 8..15 to its second, so the scale
                    // vector is the two scales over those halves.
                    let s0 = sc[2 * r] as i8 as i16;
                    let s1 = sc[2 * r + 1] as i8 as i16;
                    let scv = _mm256_setr_epi16(
                        s0, s0, s0, s0, s0, s0, s0, s0, s1, s1, s1, s1, s1, s1, s1, s1,
                    );
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p, scv));
                    r += 1;
                }
            }
            accf = _mm256_fmadd_ps(_mm256_set1_ps(act.d[s] * d), _mm256_cvtepi32_ps(acc), accf);
        }
        hsum_ps_avx2(accf)
    }
}

/// Whether [`dot_k_row_pair`] has a two-row kernel for `ggml_type` — the
/// Prism ternary types, whose decode is instruction-bound enough that
/// sharing every activation load and the per-block scalar tail between two
/// rows is measurable.
pub fn supports_k_row_pair(ggml_type: u32, in_dim: usize) -> bool {
    matches!(ggml_type, GGML_TYPE_PQ2_0 | GGML_TYPE_PTQ1_0) && supports_k_row(ggml_type, in_dim)
}

/// Two rows against one token — [`dot_k_row`] for `row0` and `row1` at
/// once, the activations loaded once for both. `ggml_type` must have passed
/// [`supports_k_row_pair`].
pub fn dot_k_row_pair(ggml_type: u32, row0: &[u8], row1: &[u8], act: &ActQ8KRow) -> (f32, f32) {
    #[cfg(target_arch = "aarch64")]
    if have_dotprod() {
        return dot_k_row_pair_impl::<ISA_DOTPROD>(ggml_type, row0, row1, act);
    }
    dot_k_row_pair_impl::<ISA_BASELINE>(ggml_type, row0, row1, act)
}

fn dot_k_row_pair_impl<const ISA: u8>(
    ggml_type: u32,
    row0: &[u8],
    row1: &[u8],
    act: &ActQ8KRow,
) -> (f32, f32) {
    match ggml_type {
        GGML_TYPE_PQ2_0 => dot_k_row_prism_pair::<ISA, false>(row0, row1, act),
        GGML_TYPE_PTQ1_0 => dot_k_row_prism_pair::<ISA, true>(row0, row1, act),
        other => panic!("vecdot::dot_k_row_pair called for unsupported ggml_type {other}"),
    }
}

/// [`dot_k_row_prism`] over two rows. A kernel that shared the activation
/// loads between the rows was measured at +3% per core over two plain
/// calls (9.0 → 9.3 G weights/s on a Cortex-A720) and dropped; what the
/// two-row *task* buys is scheduling granularity, which `cpu.rs` keeps.
fn dot_k_row_prism_pair<const ISA: u8, const TRITS: bool>(
    row0: &[u8],
    row1: &[u8],
    act: &ActQ8KRow,
) -> (f32, f32) {
    (
        dot_k_row_prism::<ISA, TRITS>(row0, act),
        dot_k_row_prism::<ISA, TRITS>(row1, act),
    )
}

/// `PQ2_0` (`TRITS = false`) and `PTQ1_0` (`TRITS = true`) decode: two
/// 128-element blocks per activation super-block, each dotted in `i32`
/// and scaled once, so the `f32` work is two multiplies per 256 weights
/// where [`dot_pq2_0`]/[`dot_ptq1_0`] pay eight and reduce four times.
///
/// Measured on the 27B `PTQ1_0` file (CIX P1, `orangu-bench --depths 0`),
/// this and the fused `sdot` unpack below took decode from 1.30 to 1.58
/// tok/s, on top of the 0.19 -> 1.30 the NEON unpack gave; the per-32 path
/// stays as the fallback for a width that is not a whole number of
/// super-blocks.
fn dot_k_row_prism<const ISA: u8, const TRITS: bool>(row: &[u8], act: &ActQ8KRow) -> f32 {
    let block_bytes = if TRITS {
        PTQ1_0_BLOCK_BYTES
    } else {
        PQ2_0_BLOCK_BYTES
    };
    #[cfg(target_arch = "aarch64")]
    if ISA == ISA_DOTPROD {
        // Safety: only reachable from a path `have_dotprod()` approved.
        return unsafe {
            if TRITS {
                dot_k_row_ptq1_0_sdot(row, act)
            } else {
                dot_k_row_pq2_0_sdot(row, act)
            }
        };
    }
    let mut total = 0f32;
    for (s, pair) in row.chunks_exact(2 * block_bytes).enumerate() {
        let mut sum = 0f32;
        for (h, block) in pair.chunks_exact(block_bytes).enumerate() {
            let b = 2 * s + h;
            let x = &act.q[b * QK_PRISM..(b + 1) * QK_PRISM];
            let (dw, isum) = if TRITS {
                const QS: usize = PTQ1_0_BLOCK_BYTES - 4;
                (
                    read_f16(block, QS + 2),
                    ptq1_0_block_isum::<ISA>(&block[..QS], &block[QS..QS + 2], x, act, b),
                )
            } else {
                (read_f16(block, 0), pq2_0_block_isum::<ISA>(&block[2..], x))
            };
            sum += dw * isum as f32;
        }
        total += act.d[s] * sum;
    }
    total
}

/// `sum_i w[i] · x[i]` over one `PQ2_0` block, in `i32`.
#[inline(always)]
fn pq2_0_block_isum<const ISA: u8>(qs: &[u8], x: &[i8]) -> i32 {
    let mut w = [0i8; QK_PRISM];
    unpack_block_pq2_0(qs, &mut w);
    #[cfg(target_arch = "aarch64")]
    if ISA == ISA_DOTPROD {
        // Safety: only reachable from a path `have_dotprod()` approved.
        return unsafe { sdot_128(&w, x) };
    }
    (0..QK_PRISM / 32)
        .map(|s| dot32::<ISA>(&w[s * 32..], &x[s * 32..]))
        .sum()
}

/// `sum_i w[i] · x[i]` over one `PTQ1_0` block, in `i32`. `b` is the
/// block's index into `act`, for the activation sums the fused path needs.
#[inline(always)]
fn ptq1_0_block_isum<const ISA: u8>(
    qs: &[u8],
    qh: &[u8],
    x: &[i8],
    act: &ActQ8KRow,
    b: usize,
) -> i32 {
    #[cfg(target_arch = "aarch64")]
    if ISA == ISA_DOTPROD {
        // Safety: only reachable from a path `have_dotprod()` approved.
        return unsafe { ptq1_0_block_isum_sdot(qs, qh, x, act.prism_sums[b]) };
    }
    let _ = (act, b);
    let mut w = [0i8; QK_PRISM];
    unpack_block_ptq1_0(qs, qh, &mut w);
    (0..QK_PRISM / 32)
        .map(|s| dot32::<ISA>(&w[s * 32..], &x[s * 32..]))
        .sum()
}

/// 128 `int8` pairs into one `i32`, two independent `sdot` chains, one
/// horizontal reduction.
///
/// Intrinsics under `#[target_feature]` rather than the `asm!` block
/// [`dot32_sdot`] uses: an `asm!` is a scheduling barrier, and with one
/// accumulator that made the eight `sdot`s of a block a serial chain at
/// the instruction's latency. Two chains, and the compiler free to
/// interleave the trit extraction with them, is what the pair kernel
/// below was written for and what this shares with it.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[inline]
unsafe fn sdot_128(w: &[i8; QK_PRISM], x: &[i8]) -> i32 {
    use std::arch::aarch64::*;
    unsafe {
        let mut acc0 = vdupq_n_s32(0);
        let mut acc1 = vdupq_n_s32(0);
        for h in (0..QK_PRISM / 16).step_by(2) {
            acc0 = vdotq_s32(
                acc0,
                vld1q_s8(w.as_ptr().add(16 * h)),
                vld1q_s8(x.as_ptr().add(16 * h)),
            );
            acc1 = vdotq_s32(
                acc1,
                vld1q_s8(w.as_ptr().add(16 * h + 16)),
                vld1q_s8(x.as_ptr().add(16 * h + 16)),
            );
        }
        vaddvq_s32(vaddq_s32(acc0, acc1))
    }
}

/// The `PTQ1_0` block dot with no unpacked row at all: each trit plane of
/// the 16-byte run *is* sixteen consecutive elements (`16n..16n+16`), so
/// it is dotted against those activations as soon as it is extracted; the
/// 8-byte run's planes pair up into the next three vectors the same way
/// (`80..96`, `96..112`, and the last plane with the eight `qh` values at
/// `112..128`), and `qh` itself is one 8-lane multiply of `[h0 h1 h0 h1 …]`
/// by `[1 1 3 3 9 9 27 27]`. The trits are dotted as `0..=2` and the
/// block's `-1` bias is one subtraction of the activation sum at the end,
/// which is what `x_sum` is for.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn ptq1_0_block_isum_sdot(qs: &[u8], qh: &[u8], x: &[i8], x_sum: i32) -> i32 {
    use std::arch::aarch64::*;
    unsafe {
        let planes = ptq1_0_planes(qs, qh);
        let mut acc0 = vdupq_n_s32(0);
        let mut acc1 = vdupq_n_s32(0);
        for (n, pair) in planes.as_chunks::<2>().0.iter().enumerate() {
            acc0 = vdotq_s32(acc0, pair[0], vld1q_s8(x.as_ptr().add(32 * n)));
            acc1 = vdotq_s32(acc1, pair[1], vld1q_s8(x.as_ptr().add(32 * n + 16)));
        }
        vaddvq_s32(vaddq_s32(acc0, acc1)) - x_sum
    }
}

/// The whole `PTQ1_0` row on `sdot` hardware: one `#[target_feature]`
/// function, so [`ptq1_0_super_isum_sdot`] inlines into the super-block
/// loop instead of being called once per 256 weights with sliced
/// arguments, and the per-block activation sums come precomputed from
/// [`ActQ8KRow::prism_sums`]. The wrapper this replaced was a fifth of the
/// decode profile by itself.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn dot_k_row_ptq1_0_sdot(row: &[u8], act: &ActQ8KRow) -> f32 {
    const QS: usize = PTQ1_0_BLOCK_BYTES - 4;
    let mut total = 0f32;
    for (s, pair) in row
        .as_chunks::<{ 2 * PTQ1_0_BLOCK_BYTES }>()
        .0
        .iter()
        .enumerate()
    {
        let (block0, block1) = pair.split_at(PTQ1_0_BLOCK_BYTES);
        let (isum0, isum1) = unsafe {
            ptq1_0_super_isum_sdot(
                block0,
                block1,
                &act.q[s * SUPER_BLOCK..(s + 1) * SUPER_BLOCK],
                act.prism_sums[2 * s],
                act.prism_sums[2 * s + 1],
            )
        };
        total += act.d[s]
            * (read_f16(block0, QS + 2) * isum0 as f32 + read_f16(block1, QS + 2) * isum1 as f32);
    }
    total
}

/// The `PQ2_0` counterpart of [`dot_k_row_ptq1_0_sdot`]: the NEON field
/// unpack and [`sdot_128`] inlined into one loop, so the unpacked block
/// never leaves registers.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn dot_k_row_pq2_0_sdot(row: &[u8], act: &ActQ8KRow) -> f32 {
    let mut total = 0f32;
    let mut w = [0i8; QK_PRISM];
    for (s, pair) in row
        .as_chunks::<{ 2 * PQ2_0_BLOCK_BYTES }>()
        .0
        .iter()
        .enumerate()
    {
        let mut sum = 0f32;
        for (h, block) in pair.as_chunks::<PQ2_0_BLOCK_BYTES>().0.iter().enumerate() {
            let b = 2 * s + h;
            unsafe {
                unpack_block_pq2_0_neon(&block[2..], &mut w);
                sum += read_f16(block, 0)
                    * sdot_128(&w, &act.q[b * QK_PRISM..(b + 1) * QK_PRISM]) as f32;
            }
        }
        total += act.d[s] * sum;
    }
    total
}

/// Both `PTQ1_0` blocks of one activation super-block, dotted in `i32` —
/// [`ptq1_0_block_isum_sdot`] with the two blocks' half-width work fused
/// into full vectors: each 8-byte run's trit plane, and each block's `qh`,
/// is extracted for both blocks in one 16-lane operation, and `vdotq`
/// keeps the two blocks apart by itself — its lanes 0–1 sum the low eight
/// bytes (block 0) and lanes 2–3 the high eight (block 1), so one
/// accumulator carries both partial sums until the end. That removes
/// 23 half-width operations per 256 weights against the one-block kernel.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[inline]
unsafe fn ptq1_0_super_isum_sdot(
    block0: &[u8],
    block1: &[u8],
    x: &[i8],
    x_sum0: i32,
    x_sum1: i32,
) -> (i32, i32) {
    use std::arch::aarch64::*;
    const QS: usize = PTQ1_0_BLOCK_BYTES - 4;
    unsafe {
        let trits16 = |v: uint8x16_t| -> int8x16_t {
            vreinterpretq_s8_u8(vshrq_n_u8::<6>(vhaddq_u8(v, vshrq_n_u8::<1>(v))))
        };
        let x0 = x.as_ptr();
        let x1 = x.as_ptr().add(QK_PRISM);
        // The 16-byte runs: five planes each, own accumulators.
        let run0 = vld1q_u8(block0.as_ptr());
        let run1 = vld1q_u8(block1.as_ptr());
        let mut acc0 = vdotq_s32(vdupq_n_s32(0), trits16(run0), vld1q_s8(x0));
        let mut acc1 = vdotq_s32(vdupq_n_s32(0), trits16(run1), vld1q_s8(x1));
        for (n, &p) in [3u8, 9, 27, 81].iter().enumerate() {
            let pv = vdupq_n_u8(p);
            acc0 = vdotq_s32(
                acc0,
                trits16(vmulq_u8(run0, pv)),
                vld1q_s8(x0.add(16 * (n + 1))),
            );
            acc1 = vdotq_s32(
                acc1,
                trits16(vmulq_u8(run1, pv)),
                vld1q_s8(x1.add(16 * (n + 1))),
            );
        }
        // The 8-byte runs, side by side: low half block 0, high half block 1.
        let tail = vcombine_u8(
            vld1_u8(block0.as_ptr().add(16)),
            vld1_u8(block1.as_ptr().add(16)),
        );
        let xt =
            |off: usize| -> int8x16_t { vcombine_s8(vld1_s8(x0.add(off)), vld1_s8(x1.add(off))) };
        let mut acc = vdotq_s32(vdupq_n_s32(0), trits16(tail), xt(80));
        for (n, &p) in [3u8, 9, 27, 81].iter().enumerate() {
            acc = vdotq_s32(
                acc,
                trits16(vmulq_u8(tail, vdupq_n_u8(p))),
                xt(80 + 8 * (n + 1)),
            );
        }
        // `qh`: `[h0 h1 h0 h1 …]` of each block times `[1 1 3 3 9 9 27 27]`.
        let hpow: [u8; 16] = [1, 1, 3, 3, 9, 9, 27, 27, 1, 1, 3, 3, 9, 9, 27, 27];
        let hv = vcombine_u16(
            vdup_n_u16(u16::from_le_bytes([block0[QS], block0[QS + 1]])),
            vdup_n_u16(u16::from_le_bytes([block1[QS], block1[QS + 1]])),
        );
        acc = vdotq_s32(
            acc,
            trits16(vmulq_u8(vreinterpretq_u8_u16(hv), vld1q_u8(hpow.as_ptr()))),
            xt(120),
        );
        let lo = vget_low_s32(acc);
        let hi = vget_high_s32(acc);
        (
            vaddvq_s32(acc0) + vaddv_s32(lo) - x_sum0,
            vaddvq_s32(acc1) + vaddv_s32(hi) - x_sum1,
        )
    }
}

/// The eight 16-lane trit vectors of one `PTQ1_0` block, in element order
/// and as `0..=2` — the shared unpack of the one- and two-row `sdot`
/// kernels. Kept in registers by the caller; nothing is stored.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn ptq1_0_planes(qs: &[u8], qh: &[u8]) -> [std::arch::aarch64::int8x16_t; 8] {
    use std::arch::aarch64::*;
    const POW3: [u8; 5] = [1, 3, 9, 27, 81];
    unsafe {
        let trits16 = |v: uint8x16_t| -> int8x16_t {
            vreinterpretq_s8_u8(vshrq_n_u8::<6>(vhaddq_u8(v, vshrq_n_u8::<1>(v))))
        };
        let trits8 = |v: uint8x8_t| -> int8x8_t {
            vreinterpret_s8_u8(vshr_n_u8::<6>(vhadd_u8(v, vshr_n_u8::<1>(v))))
        };
        let run16 = vld1q_u8(qs.as_ptr());
        let run8 = vld1_u8(qs.as_ptr().add(16));
        let plane = |n: usize| -> int8x8_t {
            if n == 0 {
                trits8(run8)
            } else {
                trits8(vmul_u8(run8, vdup_n_u8(POW3[n])))
            }
        };
        let hpow: [u8; 8] = [1, 1, 3, 3, 9, 9, 27, 27];
        let hq = trits8(vmul_u8(
            vreinterpret_u8_u16(vdup_n_u16(u16::from_le_bytes([qh[0], qh[1]]))),
            vld1_u8(hpow.as_ptr()),
        ));
        [
            trits16(run16),
            trits16(vmulq_u8(run16, vdupq_n_u8(3))),
            trits16(vmulq_u8(run16, vdupq_n_u8(9))),
            trits16(vmulq_u8(run16, vdupq_n_u8(27))),
            trits16(vmulq_u8(run16, vdupq_n_u8(81))),
            vcombine_s8(plane(0), plane(1)),
            vcombine_s8(plane(2), plane(3)),
            vcombine_s8(plane(4), hq),
        ]
    }
}

/// `Q4_K`: [`dot_q4_k`]'s unpack, with both the dot and the asymmetric min
/// correction accumulated across the super-block in `i32`.
///
/// The min term collapses the same way the dot does — `sum_b m_b * bsum_b` is
/// integral — so an asymmetric type saves *two* float conversions per 32
/// rather than one.
fn dot_k_row_q4_k<const ISA: u8>(row: &[u8], act: &ActQ8KRow) -> f32 {
    const BLOCK_BYTES: usize = 2 + 2 + 12 + 128;
    let mut total = 0f32;
    let mut w = [0i8; 32];
    for (s, block) in row.as_chunks::<BLOCK_BYTES>().0.iter().enumerate() {
        let d = read_f16(block, 0);
        let dmin = read_f16(block, 2);
        let scales = &block[4..16];
        let qs = &block[16..];
        let sb = s * SUBS;
        let (mut isum, mut imin) = (0i32, 0i32);
        for g in 0..4 {
            let bytes = &qs[g * 32..g * 32 + 32];
            for (half, shift) in [(0usize, 0u32), (1, 4)] {
                let (sc, m) = get_scale_min_k4(g * 2 + half, scales);
                let b = sb + g * 2 + half;
                for (j, &byte) in bytes.iter().enumerate() {
                    w[j] = ((byte >> shift) & 0x0F) as i8;
                }
                isum += sc as i32 * dot32::<ISA>(&w, &act.q[b * 32..]);
                imin += m as i32 * k_row_block_sum(act, b);
            }
        }
        total += act.d[s] * (d * isum as f32 - dmin * imin as f32);
    }
    total
}

/// `Q5_K`: [`dot_k_row_q4_k`] with the fifth bit taken from the `qh` plane.
fn dot_k_row_q5_k<const ISA: u8>(row: &[u8], act: &ActQ8KRow) -> f32 {
    const BLOCK_BYTES: usize = 2 + 2 + 12 + 32 + 128;
    let mut total = 0f32;
    let mut w = [0i8; 32];
    for (s, block) in row.as_chunks::<BLOCK_BYTES>().0.iter().enumerate() {
        let d = read_f16(block, 0);
        let dmin = read_f16(block, 2);
        let scales = &block[4..16];
        let qh = &block[16..48];
        let qs = &block[48..];
        let sb = s * SUBS;
        let (mut isum, mut imin) = (0i32, 0i32);
        for g in 0..4 {
            let bytes = &qs[g * 32..g * 32 + 32];
            for (half, shift) in [(0usize, 0u32), (1, 4)] {
                let (sc, m) = get_scale_min_k4(g * 2 + half, scales);
                let hi_mask = 1u8 << (g * 2 + half);
                let b = sb + g * 2 + half;
                for (j, &byte) in bytes.iter().enumerate() {
                    let hi = if qh[j] & hi_mask != 0 { 16 } else { 0 };
                    w[j] = (((byte >> shift) & 0x0F) + hi) as i8;
                }
                isum += sc as i32 * dot32::<ISA>(&w, &act.q[b * 32..]);
                imin += m as i32 * k_row_block_sum(act, b);
            }
        }
        total += act.d[s] * (d * isum as f32 - dmin * imin as f32);
    }
    total
}

/// `Q6_K`: symmetric, so no min pass — but its scales are per 16, so each
/// 32-element run contributes two separately-scaled 16-lane dots to the same
/// integer accumulator.
fn dot_k_row_q6_k<const ISA: u8>(row: &[u8], act: &ActQ8KRow) -> f32 {
    const BLOCK_BYTES: usize = 128 + 64 + 16 + 2;
    let mut total = 0f32;
    let mut w = [0i8; 32];
    for (s, block) in row.as_chunks::<BLOCK_BYTES>().0.iter().enumerate() {
        let ql = &block[0..128];
        let qh = &block[128..192];
        let sc = &block[192..208];
        let d = read_f16(block, 208);
        let base = s * SUPER_BLOCK;
        let mut isum = 0i32;
        let mut r = 0usize;
        for h in 0..2 {
            let qh_run = &qh[h * 32..h * 32 + 32];
            for &(ql_add, hshift, high) in Q6K_RUNS.iter() {
                let ql_run = &ql[h * 64 + ql_add..h * 64 + ql_add + 32];
                unpack_q6k_run(ql_run, qh_run, hshift, high, &mut w);
                let x = &act.q[base + r * 32..];
                isum += sc[2 * r] as i8 as i32 * dot16::<ISA>(&w, x)
                    + sc[2 * r + 1] as i8 as i32 * dot16::<ISA>(&w[16..], &x[16..]);
                r += 1;
            }
        }
        total += act.d[s] * d * isum as f32;
    }
    total
}

/// [`dot_k_row_q4_k`] on `sdot` hardware, one **super-block per pass**: the
/// 128 quant bytes split into low and high nibbles in registers, each
/// 16-byte half dotted against its activations (two independent `sdot`s per
/// 32-element sub-block), the pair's four lanes scaled by the sub-block's
/// 6-bit scale with one `mla`, and one horizontal sum per super-block — no
/// unpacked `i8` array and no reduction per 32 elements. The integer sums
/// are the generic kernel's exactly and the float expression is the same,
/// so the result is bit-identical (`sdot_k_rows_match_the_generic_kernels`).
///
/// Why it exists: the generic form ran at 5.0 GB/s of weights on one A720,
/// a quarter of what one core can read, so eight cores could not reach the
/// memory's rate however they were scheduled (`doc/PERF-ALL.md`, task 15).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn dot_k_row_q4_k_sdot(row: &[u8], act: &ActQ8KRow) -> f32 {
    use std::arch::aarch64::*;
    const BLOCK_BYTES: usize = 2 + 2 + 12 + 128;
    let mut total = 0f32;
    unsafe {
        let m4 = vdupq_n_u8(0x0F);
        let zero = vdupq_n_s32(0);
        for (s, block) in row.as_chunks::<BLOCK_BYTES>().0.iter().enumerate() {
            let d = read_f16(block, 0);
            let dmin = read_f16(block, 2);
            let scales = &block[4..16];
            let qs = block.as_ptr().add(16);
            let x = act.q.as_ptr().add(s * SUPER_BLOCK);
            let sb = s * SUBS;
            let mut sc = [0i32; SUBS];
            let mut imin = 0i32;
            for (j, sc_j) in sc.iter_mut().enumerate() {
                let (a, m) = get_scale_min_k4(j, scales);
                *sc_j = a as i32;
                imin += m as i32 * k_row_block_sum(act, sb + j);
            }
            let mut acc0 = zero;
            let mut acc1 = zero;
            for g in 0..4 {
                let q0 = vld1q_u8(qs.add(g * 32));
                let q1 = vld1q_u8(qs.add(g * 32 + 16));
                let lo0 = vreinterpretq_s8_u8(vandq_u8(q0, m4));
                let lo1 = vreinterpretq_s8_u8(vandq_u8(q1, m4));
                let hi0 = vreinterpretq_s8_u8(vshrq_n_u8(q0, 4));
                let hi1 = vreinterpretq_s8_u8(vshrq_n_u8(q1, 4));
                let xg = x.add(g * 64);
                let plo = vdotq_s32(
                    vdotq_s32(zero, lo0, vld1q_s8(xg)),
                    lo1,
                    vld1q_s8(xg.add(16)),
                );
                let phi = vdotq_s32(
                    vdotq_s32(zero, hi0, vld1q_s8(xg.add(32))),
                    hi1,
                    vld1q_s8(xg.add(48)),
                );
                acc0 = vmlaq_n_s32(acc0, plo, sc[2 * g]);
                acc1 = vmlaq_n_s32(acc1, phi, sc[2 * g + 1]);
            }
            let isum = vaddvq_s32(vaddq_s32(acc0, acc1));
            total += act.d[s] * (d * isum as f32 - dmin * imin as f32);
        }
    }
    total
}

/// [`dot_k_row_q6_k`] on `sdot` hardware, the same way as
/// [`dot_k_row_q4_k_sdot`]: each 32-element run's six-bit quants assembled
/// in registers from its `ql` nibbles and `qh` bit pair, and each 16-lane
/// half — `Q6_K` scales per 16 — dotted and scaled into the super-block's
/// accumulator. Bit-identical to the generic kernel.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn dot_k_row_q6_k_sdot(row: &[u8], act: &ActQ8KRow) -> f32 {
    use std::arch::aarch64::*;
    const BLOCK_BYTES: usize = 128 + 64 + 16 + 2;
    let mut total = 0f32;
    unsafe {
        let m4 = vdupq_n_u8(0x0F);
        let three = vdupq_n_u8(3);
        let bias = vdupq_n_s8(32);
        let zero = vdupq_n_s32(0);
        // One run's 16 lanes: the nibble, the two high bits above it, less 32.
        let six = |nib: uint8x16_t, hb: uint8x16_t| {
            vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(nib, vshlq_n_u8(hb, 4))), bias)
        };
        for (s, block) in row.as_chunks::<BLOCK_BYTES>().0.iter().enumerate() {
            let ql = block.as_ptr();
            let qh = block.as_ptr().add(128);
            let sc = &block[192..208];
            let d = read_f16(block, 208);
            let x = act.q.as_ptr().add(s * SUPER_BLOCK);
            let mut acc0 = zero;
            let mut acc1 = zero;
            for h in 0..2 {
                let l0 = vld1q_u8(ql.add(h * 64));
                let l1 = vld1q_u8(ql.add(h * 64 + 16));
                let l2 = vld1q_u8(ql.add(h * 64 + 32));
                let l3 = vld1q_u8(ql.add(h * 64 + 48));
                let h0 = vld1q_u8(qh.add(h * 32));
                let h1 = vld1q_u8(qh.add(h * 32 + 16));
                // The four runs of `Q6K_RUNS`, in element order.
                let runs = [
                    (
                        six(vandq_u8(l0, m4), vandq_u8(h0, three)),
                        six(vandq_u8(l1, m4), vandq_u8(h1, three)),
                    ),
                    (
                        six(vandq_u8(l2, m4), vandq_u8(vshrq_n_u8(h0, 2), three)),
                        six(vandq_u8(l3, m4), vandq_u8(vshrq_n_u8(h1, 2), three)),
                    ),
                    (
                        six(vshrq_n_u8(l0, 4), vandq_u8(vshrq_n_u8(h0, 4), three)),
                        six(vshrq_n_u8(l1, 4), vandq_u8(vshrq_n_u8(h1, 4), three)),
                    ),
                    (
                        six(vshrq_n_u8(l2, 4), vshrq_n_u8(h0, 6)),
                        six(vshrq_n_u8(l3, 4), vshrq_n_u8(h1, 6)),
                    ),
                ];
                for (k, (w0, w1)) in runs.into_iter().enumerate() {
                    let r = h * 4 + k;
                    let xr = x.add(r * 32);
                    let p0 = vdotq_s32(zero, w0, vld1q_s8(xr));
                    let p1 = vdotq_s32(zero, w1, vld1q_s8(xr.add(16)));
                    acc0 = vmlaq_n_s32(acc0, p0, sc[2 * r] as i8 as i32);
                    acc1 = vmlaq_n_s32(acc1, p1, sc[2 * r + 1] as i8 as i32);
                }
            }
            let isum = vaddvq_s32(vaddq_s32(acc0, acc1));
            total += act.d[s] * d * isum as f32;
        }
    }
    total
}

/// `IQ4_XS`: symmetric with a uniform scale across each 32, so one `dot32` per
/// sub-block feeds the accumulator directly.
fn dot_k_row_iq4_xs<const ISA: u8>(row: &[u8], act: &ActQ8KRow) -> f32 {
    const BLOCK_BYTES: usize = 2 + 2 + SUPER_BLOCK / 64 + SUPER_BLOCK / 2;
    let mut total = 0f32;
    let mut w = [0i8; 32];
    for (s, block) in row.as_chunks::<BLOCK_BYTES>().0.iter().enumerate() {
        let d = read_f16(block, 0);
        let scales_h = u16::from_le_bytes([block[2], block[3]]);
        let scales_l = &block[4..8];
        let qs = &block[8..BLOCK_BYTES];
        let base = s * SUPER_BLOCK;
        let mut isum = 0i32;
        for ib in 0..SUBS {
            let low = (scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F;
            let high = ((scales_h >> (2 * ib)) & 3) as u8;
            let ls = ((low | (high << 4)) as i32) - 32;
            unpack_block_iq4_nl(&qs[ib * 16..], &mut w);
            isum += ls * dot32::<ISA>(&w, &act.q[base + ib * 32..]);
        }
        total += act.d[s] * d * isum as f32;
    }
    total
}

/// `Q3_K`: `sum_s d_s * act_d_s * sum_g sc_g * (q_g . x_g)`, the inner sum
/// entirely in `i32`.
///
/// Accumulator bound: `|sc| <= 32`, `|q| <= 4`, `|x| <= 127`, 256 elements —
/// at most `32 * 4 * 127 * 256 ~ 4.2e6`, two orders inside `i32`.
fn dot_k_row_q3_k<const ISA: u8>(row: &[u8], act: &ActQ8KRow) -> f32 {
    let mut total = 0f32;
    let mut w = [0i8; 32];
    for (s, block) in row.as_chunks::<Q3K_BLOCK_BYTES>().0.iter().enumerate() {
        let hmask = &block[0..32];
        let qs = &block[32..96];
        let sc = unpack_q3_k_scales(&block[96..108]);
        let d = read_f16(block, 108);
        let base = s * SUPER_BLOCK;
        let mut isum = 0i32;
        for (p, &(off, shift)) in Q2K_RUNS.iter().enumerate() {
            unpack_q3k_run(&qs[off..off + 32], hmask, shift, 1 << p, &mut w);
            let x = &act.q[base + p * 32..];
            isum += (sc[2 * p] - 32) * dot16::<ISA>(&w, x)
                + (sc[2 * p + 1] - 32) * dot16::<ISA>(&w[16..], &x[16..]);
        }
        total += act.d[s] * d * isum as f32;
    }
    total
}

/// `Q2_K`: as above, and the min term collapses the same way —
/// `dmin * sum_g m_g * bsum_g` accumulates in `i32` and converts once.
fn dot_k_row_q2_k<const ISA: u8>(row: &[u8], act: &ActQ8KRow) -> f32 {
    let mut total = 0f32;
    let mut w = [0i8; 32];
    for (s, block) in row.as_chunks::<Q2K_BLOCK_BYTES>().0.iter().enumerate() {
        let scales = &block[0..16];
        let qs = &block[16..80];
        let d = read_f16(block, 80);
        let dmin = read_f16(block, 82);
        let base = s * SUPER_BLOCK;
        let (mut isum, mut imin) = (0i32, 0i32);
        for (p, &(off, shift)) in Q2K_RUNS.iter().enumerate() {
            unpack_q2k_run(&qs[off..off + 32], shift, &mut w);
            let x = &act.q[base + p * 32..];
            let (a, b) = (scales[2 * p], scales[2 * p + 1]);
            isum += (a & 0xF) as i32 * dot16::<ISA>(&w, x)
                + (b & 0xF) as i32 * dot16::<ISA>(&w[16..], &x[16..]);
            let g = s * (SUPER_BLOCK / GROUP) + p * 2;
            imin += (a >> 4) as i32 * act.sums[g] + (b >> 4) as i32 * act.sums[g + 1];
        }
        total += act.d[s] * (d * isum as f32 - dmin * imin as f32);
    }
    total
}

/// Fused `Q2_K` decode: `sum_g (d*sc_g) * (q_g . x_g) - (dmin*m_g) * sum(x_g)`,
/// over 16-element groups.
///
/// The min term is why this cannot share [`dot_q3_k`]'s loop: it needs the
/// activation *sums* per [`GROUP`], which [`ActQ8`] already carries for
/// exactly this reason.
fn dot_q2_k<const ISA: u8>(row: &[u8], act: &ActQ8) -> f32 {
    let mut total = 0f32;
    let mut blk = 0usize;
    let mut w = [0i8; 32];
    for block in row.as_chunks::<Q2K_BLOCK_BYTES>().0 {
        let scales = &block[0..16];
        let qs = &block[16..80];
        let d = read_f16(block, 80);
        let dmin = read_f16(block, 82);
        for (p, &(off, shift)) in Q2K_RUNS.iter().enumerate() {
            unpack_q2k_run(&qs[off..off + 32], shift, &mut w);
            let qx = &act.q[blk * 32..];
            let g = blk * 2;
            let (a, b) = (scales[2 * p], scales[2 * p + 1]);
            total += act.d[blk]
                * (d * ((a & 0xF) as f32 * dot16::<ISA>(&w, qx) as f32
                    + (b & 0xF) as f32 * dot16::<ISA>(&w[16..], &qx[16..]) as f32)
                    - dmin
                        * ((a >> 4) as f32 * act.sums[g] as f32
                            + (b >> 4) as f32 * act.sums[g + 1] as f32));
            blk += 1;
        }
    }
    total
}

/// Fused `Q3_K` decode. Symmetric, so this is [`dot_q6_k`]'s shape at a
/// different unpack: two 16-element dots per 32-element run, one shared `f32`
/// conversion, no min pass.
fn dot_q3_k<const ISA: u8>(row: &[u8], act: &ActQ8) -> f32 {
    let mut total = 0f32;
    let mut blk = 0usize;
    let mut w = [0i8; 32];
    for block in row.as_chunks::<Q3K_BLOCK_BYTES>().0 {
        let hmask = &block[0..32];
        let qs = &block[32..96];
        let sc = unpack_q3_k_scales(&block[96..108]);
        let d = read_f16(block, 108);
        for (p, &(off, shift)) in Q2K_RUNS.iter().enumerate() {
            unpack_q3k_run(&qs[off..off + 32], hmask, shift, 1 << p, &mut w);
            let qx = &act.q[blk * 32..];
            let s0 = (sc[2 * p] - 32) as f32;
            let s1 = (sc[2 * p + 1] - 32) as f32;
            total += act.d[blk]
                * d
                * (s0 * dot16::<ISA>(&w, qx) as f32
                    + s1 * dot16::<ISA>(&w[16..], &qx[16..]) as f32);
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
    for block in row.as_chunks::<BLOCK_BYTES>().0 {
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
            GGML_TYPE_Q4_0 => (18, 32),
            GGML_TYPE_Q5_1 => (24, 32),
            GGML_TYPE_IQ4_NL => (18, 32),
            GGML_TYPE_Q2_K => (84, 256),
            GGML_TYPE_Q3_K => (110, 256),
            GGML_TYPE_Q4_K => (144, 256),
            GGML_TYPE_Q5_K => (176, 256),
            GGML_TYPE_Q6_K => (210, 256),
            GGML_TYPE_IQ4_XS => (136, 256),
            GGML_TYPE_PQ2_0 => (34, 128),
            GGML_TYPE_PTQ1_0 => (28, 128),
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
                GGML_TYPE_Q8_0 | GGML_TYPE_Q5_0 | GGML_TYPE_Q4_0 | GGML_TYPE_IQ4_NL => {
                    block[0..2].copy_from_slice(&[0x00, 0x30])
                }
                // `Q5_1` carries `d` then `m`, both `f16`.
                GGML_TYPE_Q5_1 => {
                    block[0..2].copy_from_slice(&[0x00, 0x30]);
                    block[2..4].copy_from_slice(&[0x00, 0x2c]);
                }
                // `Q5_K` shares `Q4_K`'s leading `d`/`dmin` pair.
                GGML_TYPE_Q4_K | GGML_TYPE_Q5_K => {
                    block[0..2].copy_from_slice(&[0x00, 0x30]);
                    block[2..4].copy_from_slice(&[0x00, 0x2c]);
                }
                // `Q2_K` carries `d` then `dmin` at the *end* of the block.
                GGML_TYPE_Q2_K => {
                    block[80..82].copy_from_slice(&[0x00, 0x30]);
                    block[82..84].copy_from_slice(&[0x00, 0x2c]);
                }
                // `Q3_K` is symmetric: one trailing `d`, no `dmin`.
                GGML_TYPE_Q3_K => block[108..110].copy_from_slice(&[0x00, 0x30]),
                GGML_TYPE_Q6_K => block[208..210].copy_from_slice(&[0x00, 0x30]),
                GGML_TYPE_IQ4_XS => block[0..2].copy_from_slice(&[0x00, 0x30]),
                GGML_TYPE_PQ2_0 => block[0..2].copy_from_slice(&[0x00, 0x30]),
                // `PTQ1_0` is the other type with a trailing `d`.
                GGML_TYPE_PTQ1_0 => block[26..28].copy_from_slice(&[0x00, 0x30]),
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

    /// A float weight row of `in_dim` elements, from pseudo-random bytes with
    /// every exponent pinned to 0 so the values land in `±[1, 2)`.
    ///
    /// Random bytes in a float field decode to Inf/NaN often enough to make a
    /// dot-product comparison meaningless — the same reason [`check`] pins the
    /// `f16` scale of each quantized block. Only the exponent is forced; sign
    /// and mantissa stay random.
    fn float_row(ggml_type: u32, in_dim: usize, seed: u32) -> Vec<u8> {
        let stride = if ggml_type == GGML_TYPE_F32 { 4 } else { 2 };
        let mut row = pseudo_bytes(in_dim * stride, seed);
        for e in row.chunks_exact_mut(stride) {
            match ggml_type {
                GGML_TYPE_F32 => {
                    let bits = u32::from_le_bytes([e[0], e[1], e[2], e[3]]);
                    let pinned = (bits & 0x807F_FFFF) | (127 << 23);
                    e.copy_from_slice(&pinned.to_le_bytes());
                }
                // `f16`: sign 1, exponent 5 (bias 15), mantissa 10.
                GGML_TYPE_F16 => {
                    let bits = u16::from_le_bytes([e[0], e[1]]);
                    let pinned = (bits & 0x83FF) | (15 << 10);
                    e.copy_from_slice(&pinned.to_le_bytes());
                }
                // `bf16`: sign 1, exponent 8 (bias 127), mantissa 7.
                GGML_TYPE_BF16 => {
                    let bits = u16::from_le_bytes([e[0], e[1]]);
                    let pinned = (bits & 0x807F) | (127 << 7);
                    e.copy_from_slice(&pinned.to_le_bytes());
                }
                other => panic!("unhandled {other}"),
            }
        }
        row
    }

    /// Unlike the quantized kernels, the float path quantizes nothing, so it
    /// must agree with `dequantize` + an exact `f32` dot to *summation-order*
    /// precision — three orders of magnitude tighter than [`check`]'s budget.
    fn check_float(ggml_type: u32, in_dim: usize, seed: u32) {
        let row = float_row(ggml_type, in_dim, seed);
        let x = activations(in_dim);
        let widened = quant::dequantize(ggml_type, &row, in_dim).unwrap();
        let reference: f32 = widened.iter().zip(&x).map(|(w, v)| w * v).sum();
        let scale: f32 = widened.iter().zip(&x).map(|(w, v)| (w * v).abs()).sum();

        assert!(supports_float(ggml_type), "type {ggml_type}");
        let got = dot_row_f32(ggml_type, &row, &x);
        let err = (got - reference).abs();
        assert!(
            err <= 1e-5 * scale.max(1e-6),
            "type {ggml_type} in_dim {in_dim}: got {got}, want {reference} \
             (err {err}, term-magnitude {scale})"
        );
    }

    #[test]
    fn float_kernels_match_dequantize_reference() {
        for &t in &[GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16] {
            for &n in &[16, 32, 256, 2048] {
                check_float(t, n, 0x51D5 + t);
            }
        }
    }

    /// `F32`/`F16`/`BF16` rows carry no block structure, so `in_dim` need not
    /// be a multiple of anything — including of `FLOAT_BLOCK`, which the main
    /// loop widens 16 at a time. A row shorter than one block, or ending
    /// mid-block, must still be exact.
    #[test]
    fn float_dot_handles_rows_that_are_not_a_whole_block() {
        for &t in &[GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16] {
            for &n in &[1, 7, 15, 17, 31, 33, 100] {
                check_float(t, n, 0x2C0F + t + n as u32);
            }
        }
    }

    /// Widening `bf16` is a pure shift — the top 16 bits of the `f32` — so it
    /// is exact for every bit pattern, with no rounding to hide a mistake.
    /// A wrong shift or a byte-order slip would show up here and nowhere else.
    #[test]
    fn bf16_widening_is_the_top_half_of_the_f32() {
        for raw in [0x0000u16, 0x3F80, 0xBF80, 0x4049, 0x7F7F, 0x8000] {
            let bytes = raw.to_le_bytes();
            let got = widen_float::<FKIND_BF16>(&bytes, 0);
            assert_eq!(got.to_bits() >> 16, raw as u32, "bf16 {raw:#06x}");
        }
    }

    #[test]
    fn supports_float_covers_only_the_float_types() {
        for &t in &[GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16] {
            assert!(supports_float(t), "type {t}");
        }
        // The quantized types have an `int8` weight form and must keep taking
        // `supports`/`dot_row`; routing one here would silently drop its
        // scales, since `dot_row_f32` reads the bytes as raw floats.
        for &t in &[
            GGML_TYPE_Q8_0,
            GGML_TYPE_Q5_0,
            GGML_TYPE_Q4_0,
            GGML_TYPE_Q5_1,
            GGML_TYPE_IQ4_NL,
            GGML_TYPE_Q2_K,
            GGML_TYPE_Q3_K,
            GGML_TYPE_Q4_K,
            GGML_TYPE_Q5_K,
            GGML_TYPE_Q6_K,
            GGML_TYPE_IQ4_XS,
        ] {
            assert!(!supports_float(t), "type {t}");
            assert!(!supports(t, 0) || !supports_float(t), "type {t}");
        }
    }

    /// Block geometry and pinned `f16` scale fields for a synthetic row of any
    /// supported type. Random bytes in an `f16` scale decode to Inf/NaN often
    /// enough to make a comparison meaningless, so every fixture pins them and
    /// leaves the quant payload random.
    fn fixture_row(ggml_type: u32, in_dim: usize, seed: u32) -> Vec<u8> {
        let (block_bytes, block_elems) = match ggml_type {
            GGML_TYPE_Q8_0 => (34, 32),
            GGML_TYPE_Q5_0 => (22, 32),
            GGML_TYPE_Q4_0 => (18, 32),
            GGML_TYPE_Q5_1 => (24, 32),
            GGML_TYPE_IQ4_NL => (18, 32),
            GGML_TYPE_Q2_K => (84, 256),
            GGML_TYPE_Q3_K => (110, 256),
            GGML_TYPE_Q4_K => (144, 256),
            GGML_TYPE_Q5_K => (176, 256),
            GGML_TYPE_Q6_K => (210, 256),
            GGML_TYPE_IQ4_XS => (136, 256),
            GGML_TYPE_PQ2_0 => (34, 128),
            GGML_TYPE_PTQ1_0 => (28, 128),
            other => panic!("unhandled {other}"),
        };
        let mut row = pseudo_bytes(in_dim / block_elems * block_bytes, seed);
        for block in row.chunks_exact_mut(block_bytes) {
            match ggml_type {
                GGML_TYPE_Q8_0 | GGML_TYPE_Q5_0 | GGML_TYPE_Q4_0 | GGML_TYPE_IQ4_NL
                | GGML_TYPE_IQ4_XS => block[0..2].copy_from_slice(&[0x00, 0x30]),
                GGML_TYPE_Q5_1 | GGML_TYPE_Q4_K | GGML_TYPE_Q5_K => {
                    block[0..2].copy_from_slice(&[0x00, 0x30]);
                    block[2..4].copy_from_slice(&[0x00, 0x2c]);
                }
                GGML_TYPE_Q2_K => {
                    block[80..82].copy_from_slice(&[0x00, 0x30]);
                    block[82..84].copy_from_slice(&[0x00, 0x2c]);
                }
                GGML_TYPE_Q3_K => block[108..110].copy_from_slice(&[0x00, 0x30]),
                GGML_TYPE_Q6_K => block[208..210].copy_from_slice(&[0x00, 0x30]),
                GGML_TYPE_PQ2_0 => block[0..2].copy_from_slice(&[0x00, 0x30]),
                GGML_TYPE_PTQ1_0 => block[26..28].copy_from_slice(&[0x00, 0x30]),
                other => panic!("unhandled {other}"),
            }
        }
        row
    }

    /// Every prefill kernel must agree with the single-token path, for **every**
    /// supported type.
    ///
    /// This test exists because the one below it did not do that, and a real
    /// bug went out through the gap. `dot_unpacked_pair` is what prefill
    /// actually runs, and its per-[`GROUP`] branch dropped the min correction
    /// outright — correct while `Q6_K` was the only per-16 type, since it is
    /// symmetric, and silently wrong the moment `Q2_K` arrived. The older test
    /// covered only `dot_unpacked_multi` and only `Q8_0`/`Q5_0`/`IQ4_NL`, all
    /// per-32, so nothing in the suite executed the broken branch. What caught
    /// it was a whole-model generation that produced nothing but newlines.
    ///
    /// Both kernels, every type, and token counts either side of
    /// [`TOKEN_TILE`] so the tiled body and the scalar tail both run.
    #[test]
    fn prefill_kernels_match_the_single_token_path_for_every_type() {
        // A multiple of 256, so the 256-element super-block types are legal.
        let in_dim = 2048;
        for n_tokens in [1usize, 3, 4, 7, 8, 17] {
            for ggml_type in [
                GGML_TYPE_Q8_0,
                GGML_TYPE_Q5_0,
                GGML_TYPE_Q4_0,
                GGML_TYPE_Q5_1,
                GGML_TYPE_IQ4_NL,
                GGML_TYPE_IQ4_XS,
                GGML_TYPE_Q2_K,
                GGML_TYPE_Q3_K,
                GGML_TYPE_Q4_K,
                GGML_TYPE_Q5_K,
                GGML_TYPE_Q6_K,
            ] {
                let row = fixture_row(ggml_type, in_dim, 5);
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
                // A tolerance, not equality: the tiled AVX2 form keeps its
                // partial sums in eight lanes and reduces once per row, so
                // its floating additions run in a different order from the
                // single-token path's. The integer sums are the same.
                for (t, (g, e)) in got.iter().zip(want.iter()).enumerate() {
                    assert!(
                        (g - e).abs() <= 1e-4 * e.abs().max(1.0),
                        "dot_unpacked_multi disagrees: type {ggml_type}, {n_tokens} tokens, \
                         token {t}: {g} vs {e}"
                    );
                }

                // The same row down both lanes: each must reproduce the
                // single-token result, which also pins the two lanes together.
                let mut g0 = vec![0f32; n_tokens];
                let mut g1 = vec![0f32; n_tokens];
                dot_unpacked_pair(&w, &w, &acts, &mut g0, &mut g1);
                assert_eq!(
                    g0, want,
                    "dot_unpacked_pair lane 0 disagrees: type {ggml_type}, {n_tokens} tokens"
                );
                assert_eq!(
                    g1, want,
                    "dot_unpacked_pair lane 1 disagrees: type {ggml_type}, {n_tokens} tokens"
                );
            }
        }
    }

    /// The tiled multi-token path must agree with calling the single-token
    /// path once per token. Both quantize identically and form the same
    /// integer sums; the AVX2 tile then folds them into eight `f32` lanes
    /// and reduces once per row, a different order of floating additions
    /// from the single-token path's per-block scalar, so this is a tight
    /// tolerance rather than equality — an index error shows as a difference
    /// of order one, not 1e-5.
    ///
    /// Kept alongside the every-type test above for its `in_dim` of 896, which
    /// is *not* a multiple of 256 — the width that forced `Qwen2.5-0.5B` onto
    /// `IQ4_NL`, and one no super-block type can be tested at.
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
                for (t, (g, e)) in got.iter().zip(want.iter()).enumerate() {
                    assert!(
                        (g - e).abs() <= 1e-4 * e.abs().max(1.0),
                        "type {ggml_type}, {n_tokens} tokens, token {t}: {g} vs {e}"
                    );
                }
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

    /// The vector `Q5` unpack against the scalar reference, bit for bit,
    /// over every single-bit `qh` (each lane's fifth bit set alone — the
    /// case a wrong shuffle index or bit mask gets wrong for exactly one
    /// lane), a spread of dense `qh` patterns, and all three biases the
    /// wrappers use.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn q5_block_unpack_avx2_is_exact_for_every_bit_and_bias() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let qs: Vec<u8> = (0..16u8)
            .map(|j| j.wrapping_mul(37).wrapping_add(11))
            .collect();
        let mut patterns: Vec<u32> = (0..32).map(|b| 1u32 << b).collect();
        patterns.extend([
            0,
            u32::MAX,
            0x8000_0001,
            0xDEAD_BEEF,
            0x0F0F_F0F0,
            0xAAAA_5555,
        ]);
        for &qh in &patterns {
            for bias in [0i8, 8, 16] {
                let mut want = [0i8; 32];
                let mut got = [0i8; 32];
                unpack_block_q5_bits_scalar(qh, &qs, bias, &mut want);
                unsafe { unpack_block_q5_bits_avx2(qh, &qs, bias, &mut got) };
                assert_eq!(got, want, "qh = {qh:#010x}, bias = {bias}");
            }
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
    fn q2_k_matches_dequantize_reference() {
        for seed in [1, 7, 99] {
            check(GGML_TYPE_Q2_K, 256, seed);
            check(GGML_TYPE_Q2_K, 2048, seed);
            check(GGML_TYPE_Q2_K, 5632, seed);
        }
    }

    #[test]
    fn q3_k_matches_dequantize_reference() {
        for seed in [1, 7, 99] {
            check(GGML_TYPE_Q3_K, 256, seed);
            check(GGML_TYPE_Q3_K, 2048, seed);
            // TinyLlama's `ffn_down` width, the one this was written for.
            check(GGML_TYPE_Q3_K, 5632, seed);
        }
    }

    /// `Q3_K` stores its third bit **inverted**: a set `hmask` bit means "do
    /// not subtract 4". Reading that sense backwards is the realistic failure
    /// mode — it yields plausible weights of the wrong value everywhere, and a
    /// model that still emits fluent text — so pin both poles directly rather
    /// than trusting the round-trip test to notice.
    ///
    /// Also asserts the K-path agrees, since `Q3_K` reaches `gemm_k_q6_k`
    /// through a *different* unpack than the GEMV path uses.
    #[test]
    fn q3_k_unpack_applies_the_inverted_high_bit() {
        let block_for = |hmask: u8| {
            let mut b = vec![hmask; 32]; // hmask
            b.extend_from_slice(&[0x00u8; 64]); // qs: every 2-bit field zero
            b.extend_from_slice(&[0x20u8; 12]); // scales, value irrelevant here
            b.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes()); // d
            b
        };
        // hmask clear -> subtract 4; hmask set -> subtract nothing.
        for (hmask, want) in [(0x00u8, -4i8), (0xFFu8, 0i8)] {
            let block = block_for(hmask);
            assert_eq!(block.len(), Q3K_BLOCK_BYTES);

            let mut row = UnpackedRow::new();
            unpack_row(GGML_TYPE_Q3_K, &block, 256, &mut row);
            assert!(
                row.q.iter().all(|&q| q == want),
                "hmask {hmask:#04x}: expected every weight {want}; got {:?}",
                &row.q[..16]
            );

            let mut krow = KRow::new();
            unpack_k_row(GGML_TYPE_Q3_K, &block, 256, &mut krow);
            assert!(
                krow.q.iter().all(|&q| q == want),
                "hmask {hmask:#04x}: K-path disagrees with the GEMV path; got {:?}",
                &krow.q[..16]
            );
            // `Q3_K` is `Q6_K`'s shape, not a fourth one.
            assert_eq!(krow.kind, KKind::Q6K);
        }
    }

    /// The eight-run rewrite of the `(q_off, shift, half)` walk is the one
    /// piece of these two kernels that is derived rather than transcribed, so
    /// check the element *ordering* against `quant::dequantize` exactly —
    /// scales pinned to 1 so any mismatch is the walk, not rounding.
    #[test]
    fn two_bit_family_run_walk_matches_the_reference_ordering() {
        for (ggml_type, bytes) in [(GGML_TYPE_Q2_K, 84), (GGML_TYPE_Q3_K, 110)] {
            let mut block = pseudo_bytes(bytes, 42);
            match ggml_type {
                GGML_TYPE_Q2_K => {
                    block[80..82].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
                    block[82..84].copy_from_slice(&half::f16::from_f32(0.0).to_le_bytes());
                }
                _ => block[108..110].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes()),
            }
            let want = quant::dequantize(ggml_type, &block, 256).unwrap();

            let mut row = UnpackedRow::new();
            unpack_row(ggml_type, &block, 256, &mut row);
            for (i, &w) in want.iter().enumerate().take(256) {
                let g = i / GROUP;
                let got = row.scale[g] * row.q[i] as f32 - row.min[g];
                assert!(
                    (got - w).abs() <= 1e-3 * w.abs().max(1.0),
                    "type {ggml_type} element {i}: got {got}, want {w}"
                );
            }
        }
    }

    /// The per-super-block decode path must agree with `dequantize` + an exact
    /// `f32` dot, to the same budget every other fused kernel is held to.
    ///
    /// A tolerance rather than equality against `dot_row`: one activation scale
    /// per 256 is coarser than one per 32, so this is deliberately a *different*
    /// approximation — the same one ggml makes for K-quants (`block_q8_K`), and
    /// the one orangu's K-quant prefill GEMM already makes. What must hold is
    /// that it stays inside the int8-activation error budget, not that it
    /// reproduces the finer path bit for bit.
    #[test]
    fn k_row_decode_matches_dequantize_reference() {
        for ggml_type in [
            GGML_TYPE_Q2_K,
            GGML_TYPE_Q3_K,
            GGML_TYPE_Q4_K,
            GGML_TYPE_Q5_K,
            GGML_TYPE_Q6_K,
            GGML_TYPE_IQ4_XS,
            GGML_TYPE_PQ2_0,
            GGML_TYPE_PTQ1_0,
        ] {
            for in_dim in [256usize, 2048, 5632] {
                for seed in [1u32, 7, 99] {
                    let row = fixture_row(ggml_type, in_dim, seed);
                    let x = activations(in_dim);
                    let deq = quant::dequantize(ggml_type, &row, in_dim).unwrap();
                    let reference: f32 = deq.iter().zip(&x).map(|(w, v)| w * v).sum();
                    let scale: f32 = deq.iter().zip(&x).map(|(w, v)| (w * v).abs()).sum();

                    assert!(supports_k_row(ggml_type, in_dim));
                    let act = quantize_act_k_row(&x);
                    let got = dot_k_row(ggml_type, &row, &act);
                    let err = (got - reference).abs();
                    assert!(
                        err <= 0.02 * scale.max(1e-6),
                        "type {ggml_type} in_dim {in_dim}: got {got}, want {reference} \
                         (err {err}, term-magnitude {scale})"
                    );
                }
            }
        }
    }

    /// The super-block-wide `sdot` kernels must reproduce the generic ones
    /// exactly: the same integer sums, the same float expression.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn sdot_k_rows_match_the_generic_kernels() {
        if !have_dotprod() {
            return;
        }
        for in_dim in [256usize, 1536, 6144] {
            for seed in [1u32, 7, 99, 1234] {
                let x = activations(in_dim);
                let act = quantize_act_k_row(&x);
                let row = fixture_row(GGML_TYPE_Q4_K, in_dim, seed);
                assert_eq!(
                    unsafe { dot_k_row_q4_k_sdot(&row, &act) }.to_bits(),
                    dot_k_row_q4_k::<ISA_DOTPROD>(&row, &act).to_bits(),
                    "Q4_K in_dim {in_dim} seed {seed}"
                );
                let row = fixture_row(GGML_TYPE_Q6_K, in_dim, seed);
                assert_eq!(
                    unsafe { dot_k_row_q6_k_sdot(&row, &act) }.to_bits(),
                    dot_k_row_q6_k::<ISA_DOTPROD>(&row, &act).to_bits(),
                    "Q6_K in_dim {in_dim} seed {seed}"
                );
            }
        }
    }

    /// What the decode row dot reads weights at, against what the memory
    /// gives a plain sum: one thread and every thread, on a `Q4_K` matrix
    /// too large for any cache. The number that says whether the kernel or
    /// the memory bounds a host layer.
    ///
    /// `cargo test --release _scratch_measure_k_row_bandwidth -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn _scratch_measure_k_row_bandwidth() {
        use rayon::prelude::*;
        let in_dim = 3840usize;
        let out_dim = 2048usize;
        let n_mats = 16usize; // 16 × 11 MiB, well past the caches
        let mats: Vec<Vec<u8>> = (0..n_mats)
            .map(|m| {
                let mut v = Vec::with_capacity(out_dim * in_dim * 144 / 256);
                for r in 0..out_dim {
                    v.extend(fixture_row(
                        GGML_TYPE_Q4_K,
                        in_dim,
                        (m * out_dim + r) as u32,
                    ));
                }
                v
            })
            .collect();
        let row_bytes = in_dim * 144 / 256;
        let x = activations(in_dim);
        let act = quantize_act_k_row(&x);
        let bytes = (n_mats * out_dim * row_bytes) as f64;
        let mut out = vec![0f32; out_dim];
        for threads in [1usize, 4, 8, 16] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let mut best = f64::MAX;
            for _ in 0..3 {
                let t = std::time::Instant::now();
                for m in &mats {
                    pool.install(|| {
                        out.par_chunks_mut(1)
                            .with_min_len(64)
                            .enumerate()
                            .for_each(|(r, o)| {
                                o[0] = dot_k_row(
                                    GGML_TYPE_Q4_K,
                                    &m[r * row_bytes..(r + 1) * row_bytes],
                                    &act,
                                );
                            });
                    });
                }
                best = best.min(t.elapsed().as_secs_f64());
            }
            eprintln!(
                "  dot_k_row Q4_K, {threads:2} threads: {:6.1} GB/s",
                bytes / 1e9 / best
            );
            // The memory's own rate: a plain u64 sum over the same bytes.
            let mut best = f64::MAX;
            for _ in 0..3 {
                let t = std::time::Instant::now();
                let mut total = 0u64;
                for m in &mats {
                    total = total.wrapping_add(pool.install(|| {
                        m.par_chunks(64 * 1024)
                            .map(|c| {
                                c.as_chunks::<8>()
                                    .0
                                    .iter()
                                    .map(|b| u64::from_le_bytes(*b))
                                    .fold(0u64, u64::wrapping_add)
                            })
                            .reduce(|| 0, u64::wrapping_add)
                    }));
                }
                std::hint::black_box(total);
                best = best.min(t.elapsed().as_secs_f64());
            }
            eprintln!(
                "  plain sum,      {threads:2} threads: {:6.1} GB/s",
                bytes / 1e9 / best
            );
        }
        std::hint::black_box(&out);
    }

    /// The super-block-wide AVX2 kernels compute the same integers as the
    /// per-32 form they replace — the same products, the same scales — and
    /// differ from it only in the order the super-blocks' floats are
    /// summed (eight lanes across the row, one sum at the end), so they
    /// agree to float reassociation: a wrong nibble half or a wrong scale
    /// lane is off by whole terms, orders of magnitude past this bound.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn k_row_block_kernels_match_the_baseline_exactly() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        for ggml_type in [GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K] {
            for in_dim in [256usize, 2048, 5632] {
                for seed in [1u32, 7, 99, 1234] {
                    let row = fixture_row(ggml_type, in_dim, seed);
                    let x = activations(in_dim);
                    let act = quantize_act_k_row(&x);
                    let baseline = dot_k_row_impl::<ISA_BASELINE>(ggml_type, &row, &act);
                    // Safety: AVX2 checked above.
                    let block = unsafe { dot_k_row_avx2(ggml_type, &row, &act) };
                    // Against the terms' magnitude, not the result's: a dot
                    // of random signs is a small difference of large sums,
                    // and reassociation moves it by a part in 10^5 of those.
                    let deq = quant::dequantize(ggml_type, &row, in_dim).unwrap();
                    let scale: f32 = deq.iter().zip(&x).map(|(w, v)| (w * v).abs()).sum();
                    assert!(
                        (block - baseline).abs() <= 1e-5 * scale.max(1e-3),
                        "type {ggml_type} in_dim {in_dim} seed {seed}: block {block} vs baseline {baseline} (terms {scale})"
                    );
                }
            }
        }
    }

    /// The row-wide AVX2 kernels for `Q8_0` and `Q4_0` against the per-32
    /// baseline, to reassociation.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn row_block_kernels_match_the_baseline() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        for ggml_type in [
            GGML_TYPE_Q8_0,
            GGML_TYPE_Q4_0,
            GGML_TYPE_Q5_0,
            GGML_TYPE_Q5_1,
        ] {
            for in_dim in [32usize, 2048, 3840] {
                for seed in [1u32, 7, 99, 1234] {
                    let row = fixture_row(ggml_type, in_dim, seed);
                    let x = activations(in_dim);
                    let act = quantize_act(&x);
                    let baseline = dot_row_impl::<ISA_BASELINE>(ggml_type, &row, &act);
                    // Safety: AVX2 checked above.
                    let block = unsafe { dot_row_avx2(ggml_type, &row, &act) };
                    let deq = quant::dequantize(ggml_type, &row, in_dim).unwrap();
                    let scale: f32 = deq.iter().zip(&x).map(|(w, v)| (w * v).abs()).sum();
                    assert!(
                        (block - baseline).abs() <= 1e-5 * scale.max(1e-3),
                        "type {ggml_type} in_dim {in_dim} seed {seed}: block {block} vs baseline {baseline} (terms {scale})"
                    );
                }
            }
        }
    }

    /// The fused `sdot` block dots — trit planes dotted straight from the
    /// packed bytes, the `-1` bias folded into one activation-sum
    /// subtraction — are integer arithmetic, so they must equal the
    /// unpack-then-dot form exactly, block for block, over random payloads
    /// and random activations.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn fused_prism_block_dots_equal_the_unpacked_form_exactly() {
        if !have_dotprod() {
            eprintln!("no dotprod on this machine; nothing to compare");
            return;
        }
        for seed in 0..64u32 {
            let bytes = pseudo_bytes(32, seed * 7 + 1);
            let x: Vec<f32> = pseudo_bytes(256, seed * 13 + 5)
                .iter()
                .map(|&b| (b as f32 - 128.0) / 37.0)
                .collect();
            let act = quantize_act_k_row(&x);
            let xq = &act.q[..QK_PRISM];
            let mut w = [0i8; QK_PRISM];

            unpack_block_ptq1_0(&bytes[..24], &bytes[24..26], &mut w);
            let want: i32 = (0..4)
                .map(|s| dot32::<ISA_BASELINE>(&w[s * 32..], &xq[s * 32..]))
                .sum();
            let got = ptq1_0_block_isum::<ISA_DOTPROD>(&bytes[..24], &bytes[24..26], xq, &act, 0);
            assert_eq!(got, want, "ptq1_0 seed {seed}");

            unpack_block_pq2_0(&bytes, &mut w);
            let want: i32 = (0..4)
                .map(|s| dot32::<ISA_BASELINE>(&w[s * 32..], &xq[s * 32..]))
                .sum();
            let got = pq2_0_block_isum::<ISA_DOTPROD>(&bytes, xq);
            assert_eq!(got, want, "pq2_0 seed {seed}");
        }
        // And the whole row through the super-block kernel against the
        // baseline ISA's unpack-then-dot: integer sums either way, the same
        // float operations after, so equal to the bit.
        for seed in 0..16u32 {
            for ggml_type in [GGML_TYPE_PTQ1_0, GGML_TYPE_PQ2_0] {
                let row = fixture_row(ggml_type, 2048, seed);
                let act = quantize_act_k_row(&activations(2048));
                let got = dot_k_row_impl::<ISA_DOTPROD>(ggml_type, &row, &act);
                let want = dot_k_row_impl::<ISA_BASELINE>(ggml_type, &row, &act);
                assert_eq!(got, want, "type {ggml_type} seed {seed}");
            }
        }
    }

    /// The two-row decode kernel is the one-row kernel's arithmetic with the
    /// activations loaded once, so its two results must equal the one-row
    /// results bit for bit — for both types, at a width with several
    /// super-blocks, over random rows.
    #[test]
    fn the_row_pair_decode_kernel_equals_two_single_rows_exactly() {
        for ggml_type in [GGML_TYPE_PQ2_0, GGML_TYPE_PTQ1_0] {
            for seed in [3u32, 11, 42] {
                let in_dim = 1024;
                let row0 = fixture_row(ggml_type, in_dim, seed);
                let row1 = fixture_row(ggml_type, in_dim, seed + 100);
                let act = quantize_act_k_row(&activations(in_dim));
                assert!(supports_k_row_pair(ggml_type, in_dim));
                let (got0, got1) = dot_k_row_pair(ggml_type, &row0, &row1, &act);
                assert_eq!(
                    got0,
                    dot_k_row(ggml_type, &row0, &act),
                    "type {ggml_type} row 0"
                );
                assert_eq!(
                    got1,
                    dot_k_row(ggml_type, &row1, &act),
                    "type {ggml_type} row 1"
                );
            }
        }
        assert!(!supports_k_row_pair(GGML_TYPE_Q4_K, 1024));
    }

    /// Single-thread rate of the Prism decode kernels, in G weights/s —
    /// the number the multi-threaded sweep cannot separate from scheduling.
    /// `cargo test --profile release-with-debug --bin orangu-server
    /// prism_decode_kernel_rate -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn prism_decode_kernel_rate() {
        let in_dim = 5120;
        let rows = 2048;
        for (name, ggml_type) in [("PTQ1_0", GGML_TYPE_PTQ1_0), ("PQ2_0", GGML_TYPE_PQ2_0)] {
            let row_bytes = fixture_row(ggml_type, in_dim, 1).len();
            let raw: Vec<u8> = (0..rows)
                .flat_map(|r| fixture_row(ggml_type, in_dim, r as u32))
                .collect();
            let act = quantize_act_k_row(&activations(in_dim));
            let act8 = quantize_act(&activations(in_dim));
            for (label, pair) in [("k_row", false), ("k_row_pair", true), ("dot_row", false)] {
                let mut best = f64::MAX;
                let mut sink = 0f32;
                for _ in 0..5 {
                    let t = std::time::Instant::now();
                    if label == "dot_row" {
                        for r in 0..rows {
                            sink +=
                                dot_row(ggml_type, &raw[r * row_bytes..(r + 1) * row_bytes], &act8);
                        }
                    } else if pair {
                        for r in (0..rows).step_by(2) {
                            let (a, b) = dot_k_row_pair(
                                ggml_type,
                                &raw[r * row_bytes..(r + 1) * row_bytes],
                                &raw[(r + 1) * row_bytes..(r + 2) * row_bytes],
                                &act,
                            );
                            sink += a + b;
                        }
                    } else {
                        for r in 0..rows {
                            sink += dot_k_row(
                                ggml_type,
                                &raw[r * row_bytes..(r + 1) * row_bytes],
                                &act,
                            );
                        }
                    }
                    best = best.min(t.elapsed().as_secs_f64());
                }
                let gw = (rows * in_dim) as f64 / best / 1e9;
                let gb = (rows * row_bytes) as f64 / best / 1e9;
                eprintln!("{name:7} {label:11} {gw:6.2} G weights/s  {gb:5.2} GB/s  (sink {sink})");
            }
        }
    }

    /// `supports_k_row` must be a **superset** of `supports_k`, with `Q2_K` the
    /// only difference.
    ///
    /// A type in `supports_k` but missing here would be a K-quant silently left
    /// on the per-32 activation scale — invisible, because it is only slower,
    /// never wrong. A type here but missing from `dot_k_row`'s match is a panic
    /// on the first token. And `Q2_K` is in decode but not prefill on purpose:
    /// its min is per 16 and the prefill GEMM's `bsum` is per 32.
    #[test]
    fn supports_k_row_is_a_superset_of_supports_k() {
        for t in [
            GGML_TYPE_Q2_K,
            GGML_TYPE_Q3_K,
            GGML_TYPE_Q4_K,
            GGML_TYPE_Q5_K,
            GGML_TYPE_Q6_K,
            GGML_TYPE_IQ4_XS,
        ] {
            assert!(supports_k_row(t, 2048), "type {t}");
            // Decode must cover everything prefill's K-GEMM does.
            assert!(
                !supports_k(t, 2048) || supports_k_row(t, 2048),
                "type {t} takes the K GEMM at prefill but not the k-row path at decode"
            );
        }
        // The one deliberate asymmetry, pinned so it cannot drift silently.
        assert!(supports_k_row(GGML_TYPE_Q2_K, 2048));
        assert!(!supports_k(GGML_TYPE_Q2_K, 2048));
        // The Prism pair: decode through the k-row path at a whole number of
        // super-blocks, the flat GEMM at prefill, and the per-32 path at a
        // width that is only a whole number of their own blocks.
        for t in [GGML_TYPE_PQ2_0, GGML_TYPE_PTQ1_0] {
            assert!(supports_k_row(t, 2048), "type {t}");
            assert!(!supports_k_row(t, 128 * 3), "type {t}");
            assert!(supports_k(t, 2048), "type {t}");
            assert!(!supports_k(t, 128 * 3), "type {t}");
            assert!(supports_flat(t, 128 * 3), "type {t}");
        }
        // Not super-block types: they have no 256-element accumulation to make.
        for t in [
            GGML_TYPE_Q8_0,
            GGML_TYPE_Q5_0,
            GGML_TYPE_Q4_0,
            GGML_TYPE_Q5_1,
            GGML_TYPE_IQ4_NL,
        ] {
            assert!(!supports_k_row(t, 2048), "type {t}");
            assert!(!supports_k(t, 2048), "type {t}");
        }
        // 896 is not a whole number of super-blocks.
        assert!(!supports_k_row(GGML_TYPE_Q3_K, 896));
        assert!(!supports_k_row(GGML_TYPE_Q4_K, 896));
    }

    /// The per-super-block accumulator must not overflow `i32` for any
    /// supported type, at the worst case the format allows.
    ///
    /// Checked by construction rather than by fixture, because a random row
    /// will not produce the extreme and the failure would be a silent wrap
    /// rather than a panic in release. The table in `supports_k_row`'s doc
    /// comment is what this pins.
    #[test]
    fn k_row_accumulator_bounds_stay_inside_i32() {
        // (type, |max scale|, |max quant|)
        for (t, sc, q) in [
            (GGML_TYPE_Q2_K, 15i64, 3i64),
            (GGML_TYPE_Q3_K, 32, 4),
            (GGML_TYPE_Q4_K, 63, 15),
            (GGML_TYPE_Q5_K, 63, 31),
            (GGML_TYPE_Q6_K, 127, 32),
            (GGML_TYPE_IQ4_XS, 32, 127),
        ] {
            // Every element of a super-block at its extreme, against |x| = 127.
            let worst = sc * q * 127 * SUPER_BLOCK as i64;
            assert!(
                worst < i32::MAX as i64,
                "type {t}: worst-case accumulator {worst} exceeds i32::MAX"
            );
            // And keep a real margin, not a hair's breadth.
            assert!(
                worst < i32::MAX as i64 / 8,
                "type {t}: worst-case accumulator {worst} leaves under 8x margin"
            );
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
    fn q5_k_matches_dequantize_reference() {
        for seed in [1, 7, 99] {
            check(GGML_TYPE_Q5_K, 256, seed);
            check(GGML_TYPE_Q5_K, 3072, seed);
            check(GGML_TYPE_Q5_K, 4864, seed);
        }
    }

    /// The high-bit plane is the only thing separating `Q5_K` from `Q4_K`, and
    /// getting its bit-to-sub-block mapping wrong is the realistic failure
    /// mode: it would still produce plausible values, just the wrong ones, in
    /// six of eight sub-blocks. Pin it directly — every `qh` bit set and every
    /// nibble zero means every unpacked weight must be exactly 16.
    #[test]
    fn q5_k_unpack_applies_the_high_bit_to_every_sub_block() {
        let mut block = Vec::new();
        block.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes()); // d
        block.extend_from_slice(&half::f16::from_f32(0.0).to_le_bytes()); // dmin
        block.extend_from_slice(&[1u8, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]); // sc=1, m=0
        block.extend_from_slice(&[0xFFu8; 32]); // qh: every high bit set
        block.extend_from_slice(&[0x00u8; 128]); // qs: every nibble 0

        let mut row = UnpackedRow::new();
        unpack_row(GGML_TYPE_Q5_K, &block, 256, &mut row);
        assert!(
            row.q.iter().all(|&q| q == 16),
            "every weight should be the bare high bit; got {:?}",
            &row.q[..16]
        );

        let mut krow = KRow::new();
        unpack_k_row(GGML_TYPE_Q5_K, &block, 256, &mut krow);
        assert!(
            krow.q.iter().all(|&q| q == 16),
            "K-path disagrees with the GEMV path; got {:?}",
            &krow.q[..16]
        );
        assert_eq!(krow.kind, KKind::Q4K);
    }

    #[test]
    fn q6_k_matches_dequantize_reference() {
        for seed in [1, 7, 99] {
            check(GGML_TYPE_Q6_K, 256, seed);
            check(GGML_TYPE_Q6_K, 4864, seed);
        }
    }

    #[test]
    fn q4_0_matches_dequantize_reference() {
        for seed in [1, 7, 99] {
            check(GGML_TYPE_Q4_0, 32, seed);
            check(GGML_TYPE_Q4_0, 896, seed);
            check(GGML_TYPE_Q4_0, 4864, seed);
        }
    }

    /// The two Prism ternary types: 128-element blocks, so the widths are
    /// multiples of 128 that are *not* multiples of 256 (`Ternary-Bonsai-2-
    /// 27B`'s own 5120 is the first). Random payload bytes exercise every
    /// trit pattern `PTQ1_0`'s multiply-and-shift decode can meet, including
    /// bytes above 242 that a real encoder never writes.
    #[test]
    fn pq2_0_matches_dequantize_reference() {
        for seed in [1, 7, 99] {
            check(GGML_TYPE_PQ2_0, 128, seed);
            check(GGML_TYPE_PQ2_0, 5120, seed);
            check(GGML_TYPE_PQ2_0, 17408, seed);
        }
    }

    #[test]
    fn ptq1_0_matches_dequantize_reference() {
        for seed in [1, 7, 99] {
            check(GGML_TYPE_PTQ1_0, 128, seed);
            check(GGML_TYPE_PTQ1_0, 5120, seed);
            check(GGML_TYPE_PTQ1_0, 17408, seed);
        }
    }

    #[test]
    fn q5_1_matches_dequantize_reference() {
        for seed in [1, 7, 99] {
            check(GGML_TYPE_Q5_1, 32, seed);
            check(GGML_TYPE_Q5_1, 896, seed);
            check(GGML_TYPE_Q5_1, 4864, seed);
        }
    }

    /// `Q5_1`'s offset is **added** (`d*q + m`) where every other asymmetric
    /// type here subtracts it, and [`UnpackedRow`] is defined as
    /// `scale*q - min`. A sign slip would not crash or produce garbage — it
    /// would bias every weight by `2m` and read as a slightly worse model,
    /// which is the failure mode this whole document warns about. Pin it:
    /// `d = 0` makes the quantized term vanish, so every weight is exactly
    /// `m`.
    #[test]
    fn q5_1_min_term_is_added_not_subtracted() {
        let mut block = Vec::new();
        block.extend_from_slice(&half::f16::from_f32(0.0).to_le_bytes()); // d
        block.extend_from_slice(&half::f16::from_f32(2.5).to_le_bytes()); // m
        block.extend_from_slice(&[0u8; 4]); // qh
        block.extend_from_slice(&[0u8; 16]); // qs

        let reference = quant::dequantize(GGML_TYPE_Q5_1, &block, 32).unwrap();
        assert!(
            reference.iter().all(|&v| (v - 2.5).abs() < 1e-6),
            "{reference:?}"
        );

        let mut row = UnpackedRow::new();
        unpack_row(GGML_TYPE_Q5_1, &block, 32, &mut row);
        assert!(row.has_min);
        // `scale*q - min` with q = 0 must give +2.5, so `min` is -2.5.
        assert!((row.min[0] + 2.5).abs() < 1e-6, "min was {}", row.min[0]);
    }

    /// `Q4_0` is `Q5_0` with no high-bit plane and a -8 zero point; both go
    /// through one unpack, so the risk is the shared path applying the wrong
    /// bias. All-zero nibbles at `d = 1.0` must be exactly -8 (`Q5_0`'s
    /// answer would be -16).
    #[test]
    fn q4_0_uses_its_own_zero_point_not_q5_0s() {
        let mut block = Vec::new();
        block.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        block.extend_from_slice(&[0u8; 16]);

        let mut row = UnpackedRow::new();
        unpack_row(GGML_TYPE_Q4_0, &block, 32, &mut row);
        assert!(!row.has_min);
        assert!(row.q.iter().all(|&q| q == -8), "got {:?}", &row.q[..8]);
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
        assert!(!supports(GGML_TYPE_Q5_K, 896));
        // `Q4_0`/`Q5_1` are per-32 like `Q5_0`, so 896 is fine for them.
        assert!(supports(GGML_TYPE_Q4_0, 896));
        assert!(supports(GGML_TYPE_Q5_1, 896));
        assert!(!supports(GGML_TYPE_Q4_0, 24));
        // And types with no fused kernel stay on the old path regardless.
        assert!(!supports(quant::GGML_TYPE_F32, 1024));
        assert!(!supports(quant::GGML_TYPE_BF16, 1024));
    }

    #[test]
    fn quantize_act_handles_an_all_zero_block_without_nan() {
        let a = quantize_act(&vec![0f32; 64]);
        assert!(a.d.iter().all(|d| *d == 0.0));
        assert!(a.q.iter().all(|q| *q == 0));
        assert!(a.sums.iter().all(|s| *s == 0));
    }

    /// Blocks chosen for the places [`quantize_act`]'s vector kernel and
    /// [`quantize_act_scalar`] could legally disagree, rather than for
    /// coverage: `round`'s tie direction, the `0.5.next_down()` case that
    /// the `|x| + 0.5` formulation gets wrong without a correction, `NaN`
    /// in the `abs`-max fold, `NaN` and infinity through the clamp, and the
    /// saturating `as i8` cast. Every block's `amax` is `254.0` where the
    /// tie behaviour is under test, which makes `inv` exactly `0.5` so the
    /// scaled values land on representable halves.
    fn quantize_act_adversarial_blocks() -> Vec<f32> {
        let mut x = Vec::new();
        let mut block = |v: &[f32]| {
            assert_eq!(v.len(), ACT_BLOCK);
            x.extend_from_slice(v);
        };

        block(&[0.0; ACT_BLOCK]);

        // inv == 0.5 exactly. 1.0 -> 0.5 and -1.0 -> -0.5 are ties, and
        // `f32::round` takes them away from zero, which `roundps`'s native
        // ties-to-even would not.
        let mut ties = [254.0f32; ACT_BLOCK];
        for (i, slot) in ties.iter_mut().enumerate().skip(1) {
            *slot = match i % 8 {
                1 => 1.0,                       // 0.5  -> 1
                2 => -1.0,                      // -0.5 -> -1
                3 => 3.0,                       // 1.5  -> 2
                4 => -3.0,                      // -1.5 -> -2
                5 => 2.0 * 0.5f32.next_down(),  // just under 0.5 -> 0
                6 => -2.0 * 0.5f32.next_down(), // just over -0.5 -> 0
                _ => 5.0,                       // 2.5 -> 3
            };
        }
        block(&ties);

        // Denormals and the smallest normal, against a huge `amax`, so `inv`
        // underflows the products to zero.
        let mut tiny = [f32::MIN_POSITIVE; ACT_BLOCK];
        tiny[0] = 1.0e30;
        tiny[7] = -f32::from_bits(1); // smallest denormal
        tiny[9] = f32::from_bits(1);
        block(&tiny);

        // `NaN` must be skipped by the `abs`-max fold, leaving `amax` at
        // 4.0, and must itself quantize to 0 through the clamp and cast.
        let mut nans = [4.0f32; ACT_BLOCK];
        nans[0] = f32::NAN;
        nans[3] = -f32::NAN;
        nans[31] = f32::NAN;
        block(&nans);

        // An infinity makes `scale` infinite and `inv` zero, so every finite
        // lane scales to 0 and the infinity itself becomes `inf * 0` = `NaN`.
        let mut infs = [1.0f32; ACT_BLOCK];
        infs[5] = f32::INFINITY;
        infs[6] = f32::NEG_INFINITY;
        block(&infs);

        // The block maximum is negative, and is the *only* lane of that
        // magnitude. Without this, dropping the `abs` from the max fold is
        // undetectable: every other block here happens to carry its peak
        // magnitude as a positive value too.
        let mut neg = [0.25f32; ACT_BLOCK];
        neg[13] = -100.0;
        neg[20] = -60.0;
        block(&neg);

        // A plain spread, plus lanes that land on exactly +-127.
        let mut lcg = 0x1234_5678u32;
        let mut spread = [0f32; ACT_BLOCK];
        for slot in spread.iter_mut() {
            lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *slot = (lcg >> 8) as f32 / 8_388_608.0 - 0.5;
        }
        spread[0] = 1.0;
        spread[1] = -1.0;
        block(&spread);

        x
    }

    #[test]
    fn quantize_act_vector_kernel_is_bit_identical_to_its_scalar_reference() {
        let x = quantize_act_adversarial_blocks();
        let want = quantize_act_scalar(&x);
        let got = quantize_act(&x);

        assert_eq!(got.q, want.q, "int8 activations differ");
        assert_eq!(got.sums, want.sums, "per-group sums differ");
        // Bit patterns, not values: `d` carries infinities and the sign of
        // zero, and `==` would call two different scales equal.
        let bits = |v: &[f32]| v.iter().map(|s| s.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&got.d), bits(&want.d), "block scales differ");
    }

    /// `quantize_block`'s `SUMS` parameter must change *only* whether the
    /// per-[`GROUP`] sums are produced — never the scale and never a single
    /// quantized byte. `ActQ8Flat::quantize` takes the `false`
    /// monomorphization and `quantize_act` the `true` one, so a divergence
    /// would silently give the two activation layouts different numbers from
    /// the same input.
    #[test]
    fn quantize_block_is_the_same_quantization_with_and_without_sums() {
        let x = quantize_act_adversarial_blocks();
        for chunk in x.as_chunks::<ACT_BLOCK>().0 {
            let mut with = [0i8; ACT_BLOCK];
            let mut without = [0i8; ACT_BLOCK];
            let mut sums = Vec::new();
            let mut unused = Vec::new();
            let d_with = quantize_block::<true>(chunk, &mut with, &mut sums);
            let d_without = quantize_block::<false>(chunk, &mut without, &mut unused);
            assert_eq!(with, without, "SUMS changed the quantized bytes");
            assert_eq!(
                d_with.to_bits(),
                d_without.to_bits(),
                "SUMS changed the scale"
            );
            assert!(unused.is_empty(), "SUMS = false produced sums anyway");
            // And the sums it does produce are the sums of what it wrote.
            let want: Vec<i32> = with
                .as_chunks::<GROUP>()
                .0
                .iter()
                .map(|g| g.iter().map(|&v| v as i32).sum())
                .collect();
            assert_eq!(sums, want, "the per-group sums do not match the bytes");
        }
    }

    #[test]
    fn quantize_act_rounds_ties_away_from_zero_like_f32_round() {
        // Guards the kernel against `round_ties_even`, which is what a
        // native `roundps` would give and which passes a random-input test
        // roughly always. `inv` is exactly 0.5 here, so 1.0 and 3.0 scale to
        // 0.5 and 1.5 — the two ties that split the modes apart.
        let mut x = [254.0f32; ACT_BLOCK];
        x[1] = 1.0;
        x[2] = 3.0;
        x[3] = -1.0;
        x[4] = -3.0;
        let a = quantize_act(&x);
        assert_eq!(a.q[1], 1, "0.5 must round away from zero, not to even");
        assert_eq!(a.q[2], 2, "1.5 must round away from zero");
        assert_eq!(a.q[3], -1, "-0.5 must round away from zero");
        assert_eq!(a.q[4], -2, "-1.5 must round away from zero");
    }

    /// The tiled float kernel is `dot_row_f32` in a different summation
    /// order: every output within `f32` rounding of the per-token dot, at an
    /// `in_dim` that is not a multiple of four (a scalar tail), a token
    /// count that is not a multiple of eight (a short last tile), and rows
    /// stored as `F16` and `BF16` as well as `F32`.
    #[test]
    fn tiled_float_gemm_matches_the_per_token_dot() {
        for (ggml_type, stride) in [
            (GGML_TYPE_F32, 4usize),
            (GGML_TYPE_F16, 2),
            (GGML_TYPE_BF16, 2),
        ] {
            let (in_dim, n_tokens) = (867usize, 21usize);
            let value = |i: usize| ((i * 31 % 17) as f32 - 8.0) * 0.125;
            let mut raw = Vec::new();
            for o in 0..F32_ROWS {
                for i in 0..in_dim {
                    let v = value(o * in_dim + i);
                    match ggml_type {
                        GGML_TYPE_F32 => raw.extend_from_slice(&v.to_le_bytes()),
                        GGML_TYPE_F16 => {
                            raw.extend_from_slice(&half::f16::from_f32(v).to_le_bytes())
                        }
                        _ => raw.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes()),
                    }
                }
            }
            let row_bytes = in_dim * stride;
            let x: Vec<f32> = (0..n_tokens * in_dim).map(|i| value(i + 5) * 0.5).collect();
            let mut rows: Vec<Vec<f32>> = vec![Vec::new(); F32_ROWS];
            for (o, row) in rows.iter_mut().enumerate() {
                widen_float_row(
                    ggml_type,
                    &raw[o * row_bytes..(o + 1) * row_bytes],
                    in_dim,
                    row,
                );
            }
            let mut got = vec![0f32; F32_ROWS * n_tokens];
            {
                let (g0, rest) = got.split_at_mut(n_tokens);
                let (g1, rest) = rest.split_at_mut(n_tokens);
                let (g2, g3) = rest.split_at_mut(n_tokens);
                gemm_f32_rows(
                    [&rows[0], &rows[1], &rows[2], &rows[3]],
                    &x,
                    in_dim,
                    [g0, g1, g2, g3],
                );
            }
            for o in 0..F32_ROWS {
                for t in 0..n_tokens {
                    let want = dot_row_f32(
                        ggml_type,
                        &raw[o * row_bytes..(o + 1) * row_bytes],
                        &x[t * in_dim..(t + 1) * in_dim],
                    );
                    let have = got[o * n_tokens + t];
                    assert!(
                        (have - want).abs() <= 1e-4 * want.abs().max(1.0),
                        "type {ggml_type} row {o} token {t}: {have} vs {want}"
                    );
                }
            }
            // The portable tile — what `gemm_f32_rows` runs off `aarch64` —
            // is the same product in another summation order: checked on
            // every architecture, against the same per-token dot.
            let wr: [&[f32]; F32_ROWS] = [&rows[0], &rows[1], &rows[2], &rows[3]];
            let tile = f32_tile_portable(&wr, &x, in_dim);
            for o in 0..F32_ROWS {
                for t in 0..F32_TOKENS {
                    let want = dot_row_f32(
                        ggml_type,
                        &raw[o * row_bytes..(o + 1) * row_bytes],
                        &x[t * in_dim..(t + 1) * in_dim],
                    );
                    assert!(
                        (tile[o][t] - want).abs() <= 1e-4 * want.abs().max(1.0),
                        "portable tile, type {ggml_type} row {o} token {t}: {} vs {want}",
                        tile[o][t]
                    );
                }
            }
        }
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
        let (block_bytes, block_elems) = match ggml_type {
            GGML_TYPE_Q4_K => (144, 256),
            GGML_TYPE_IQ4_XS => (136, 256), // 2 + 2 + 4 + 128
            GGML_TYPE_PQ2_0 => (34, 128),
            GGML_TYPE_PTQ1_0 => (28, 128),
            _ => (210, 256), // Q6_K
        };
        let row_bytes = in_dim / block_elems * block_bytes;
        let mut raw = pseudo_bytes(row_bytes * out_dim, seed);
        for (i, block) in raw.chunks_exact_mut(block_bytes).enumerate() {
            match ggml_type {
                GGML_TYPE_Q4_K => {
                    block[0..2].copy_from_slice(&[0x00, 0x30]); // d = 0.125
                    block[2..4].copy_from_slice(&[0x00, 0x2c]); // dmin = 0.0625
                }
                // `d` only; the 6-bit scales stay pseudo-random, which is
                // what exercises the `scales_l`/`scales_h` split and the -32
                // bias (so a sub-block scale can legitimately be negative).
                GGML_TYPE_IQ4_XS => block[0..2].copy_from_slice(&[0x00, 0x30]),
                // A scale that *differs* between the two blocks of every
                // super-block (and is negative now and then), so the
                // base-plus-integer encoding is exercised rather than hidden
                // by a uniform 0.125.
                GGML_TYPE_PQ2_0 | GGML_TYPE_PTQ1_0 => {
                    let scale =
                        0.125 * (1.0 + (i % 7) as f32 / 8.0) * if i % 5 == 3 { -1.0 } else { 1.0 };
                    let at = if ggml_type == GGML_TYPE_PQ2_0 { 0 } else { 26 };
                    block[at..at + 2].copy_from_slice(&half::f16::from_f32(scale).to_le_bytes());
                }
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

    /// The `i8mm` four-row kernel is the pair kernel's arithmetic in a
    /// different register shape: same integer sums, same `f32` operations in
    /// the same order per lane. So it must agree **to the bit** — on every
    /// kind, with a trailing tile of padded tokens, and against rows that
    /// carry negative scales (`IQ4_XS`/`Q6_K`) and a min pass (`Q4_K`).
    /// The portable form of the same tile product (`dot_k_rows_portable`,
    /// what `dot_k_rows_tiles` is off `aarch64`) is held to the same bit —
    /// on every architecture; the `smmla` form where the core has `i8mm`.
    #[test]
    fn i8mm_quad_kernel_is_bit_identical_to_the_pair_kernel() {
        let smmla = have_i8mm();
        if !smmla {
            eprintln!("no i8mm on this core; the portable form alone is checked");
        }
        for ggml_type in [GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ4_XS] {
            for (in_dim, n_tokens) in [(256usize, 7usize), (2048, 4), (1024, 13)] {
                // Two quads, so the per-task row loop runs more than once.
                let out_dim = 2 * ROW_QUAD;
                let (raw, x, row_bytes) = k_fixture(ggml_type, in_dim, out_dim, n_tokens, 777);
                let acts = ActQ8K::quantize(&x, in_dim, n_tokens);
                let mut rows: Vec<KRow> = (0..out_dim).map(|_| KRow::new()).collect();
                for (o, row) in rows.iter_mut().enumerate() {
                    unpack_k_row(
                        ggml_type,
                        &raw[o * row_bytes..(o + 1) * row_bytes],
                        in_dim,
                        row,
                    );
                }
                let mut expect = vec![0f32; out_dim * n_tokens];
                for (pair, dst) in expect.chunks_mut(2 * n_tokens).enumerate() {
                    let (e0, e1) = dst.split_at_mut(n_tokens);
                    dot_k_pair(&rows[pair * 2], &rows[pair * 2 + 1], &acts, e0, e1);
                }
                let mm = ActQ8Mm::quantize(&x, in_dim, n_tokens);
                let refs: Vec<&KRow> = rows.iter().collect();
                let mut portable = vec![0f32; out_dim * n_tokens];
                {
                    let mut outs: Vec<&mut [f32]> = portable.chunks_mut(n_tokens).collect();
                    dot_k_rows_portable(&refs, &mm, 0..mm.n_tiles(), &mut outs);
                }
                assert!(
                    portable
                        .iter()
                        .zip(&expect)
                        .all(|(g, e)| g.to_bits() == e.to_bits()),
                    "type {ggml_type} in_dim {in_dim} tokens {n_tokens}: portable {portable:?} != pair {expect:?}"
                );
                if smmla {
                    let mut got = vec![0f32; out_dim * n_tokens];
                    {
                        let mut outs: Vec<&mut [f32]> = got.chunks_mut(n_tokens).collect();
                        dot_k_rows(&refs, &mm, &mut outs);
                    }
                    assert!(
                        got.iter()
                            .zip(&expect)
                            .all(|(g, e)| g.to_bits() == e.to_bits()),
                        "type {ggml_type} in_dim {in_dim} tokens {n_tokens}: quad {got:?} != pair {expect:?}"
                    );
                }
                // And not trivially: the rows produce something.
                assert!(expect.iter().any(|v| *v != 0.0));
            }
        }
    }

    /// The K-quant GEMM must land inside the same `int8`-activation error
    /// budget the generic path is held to. Its activation scale is coarser —
    /// one per 256 rather than one per 32, matching ggml's `block_q8_K` — so
    /// this is the check that the coarser scale stays acceptable, not just
    /// that the indexing is right.
    #[test]
    fn k_gemm_matches_dequantize_reference() {
        for ggml_type in [
            GGML_TYPE_Q4_K,
            GGML_TYPE_Q6_K,
            GGML_TYPE_IQ4_XS,
            GGML_TYPE_PQ2_0,
            GGML_TYPE_PTQ1_0,
        ] {
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
        for ggml_type in [GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_PTQ1_0] {
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
        // `Q5_K` is `Q4_K` plus a high-bit plane: same per-32 scale-and-min
        // shape, so it takes the same kernel and must never reach the
        // symmetric flat GEMM.
        assert!(supports_k(GGML_TYPE_Q5_K, 2048));
        assert!(!supports_flat(GGML_TYPE_Q5_K, 2048));
        assert!(!supports_k(GGML_TYPE_Q5_K, 896));
        // `Q3_K` is `Q6_K`'s shape — per-16 signed scale, symmetric — so it
        // takes the same kernel.
        assert!(supports_k(GGML_TYPE_Q3_K, 2048));
        assert!(!supports_flat(GGML_TYPE_Q3_K, 2048));
        // `Q2_K` deliberately does NOT: its min is per-16 while the min pass
        // reads a per-32 `bsum`. It must still be fused for decode, and it
        // must still be refused here, or the K GEMM would silently drop the
        // min term and produce a subtly wrong forward pass.
        assert!(supports(GGML_TYPE_Q2_K, 2048));
        assert!(!supports_k(GGML_TYPE_Q2_K, 2048));
        assert!(!supports_flat(GGML_TYPE_Q2_K, 2048));
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
        let (block_bytes, block_elems, d_at) = match ggml_type {
            GGML_TYPE_Q8_0 => (34, 32, 0),     // 2 + 32
            GGML_TYPE_IQ4_NL => (18, 32, 0),   // 2 + 16
            GGML_TYPE_PQ2_0 => (34, 128, 0),   // 2 + 32
            GGML_TYPE_PTQ1_0 => (28, 128, 26), // 24 + 2 + 2, scale last
            _ => (22, 32, 0),                  // Q5_0: 2 + 4 + 16
        };
        let row_bytes = in_dim / block_elems * block_bytes;
        let mut raw = pseudo_bytes(row_bytes * out_dim, seed);
        for block in raw.chunks_exact_mut(block_bytes) {
            block[d_at..d_at + 2].copy_from_slice(&[0x00, 0x30]); // d = 0.125
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
    /// are read from, not the arithmetic. The generic path's trailing odd
    /// row goes through the AVX2 tile, which reduces its eight lanes once per
    /// row rather than per block, so the comparison is a tight tolerance
    /// (1e-4) rather than equality — still far inside what an indexing
    /// difference would produce, and far tighter than the reference test
    /// below's 1% budget.
    ///
    /// `n_tokens` 8 is a whole number of tiles and 7 is not, so the
    /// tile-padding path is compared too (the old code handles that tail with
    /// a separate scalar loop, this one with zero-padding).
    #[test]
    fn flat_gemm_is_bit_identical_to_the_generic_gemm() {
        for ggml_type in [GGML_TYPE_Q8_0, GGML_TYPE_Q5_0, GGML_TYPE_IQ4_NL] {
            for in_dim in [32usize, 896, 4864] {
                flat_vs_generic(ggml_type, in_dim);
            }
        }
        // The Prism types block at 128; `5120` is the served model's own
        // width.
        for ggml_type in [GGML_TYPE_PQ2_0, GGML_TYPE_PTQ1_0] {
            for in_dim in [128usize, 5120] {
                flat_vs_generic(ggml_type, in_dim);
            }
        }
    }

    fn flat_vs_generic(ggml_type: u32, in_dim: usize) {
        for n_tokens in [2usize, 7, 8] {
            let out_dim = 5; // odd, so the trailing-row path runs
            let (raw, x, row_bytes) = flat_fixture(ggml_type, in_dim, out_dim, n_tokens, 909);
            let flat = flat_gemm(ggml_type, &raw, row_bytes, in_dim, out_dim, &x, n_tokens);
            let generic = generic_gemm(ggml_type, &raw, row_bytes, in_dim, out_dim, &x, n_tokens);
            assert_eq!(flat.len(), generic.len());
            for (i, (f, g)) in flat.iter().zip(generic.iter()).enumerate() {
                assert!(
                    (f - g).abs() <= 1e-4 * g.abs().max(1.0),
                    "type {ggml_type} in_dim {in_dim} n_tokens {n_tokens} at {i}: {f} vs {g}"
                );
            }
        }
    }

    #[test]
    fn flat_gemm_matches_dequantize_reference() {
        for (ggml_type, widths) in [
            (GGML_TYPE_Q8_0, [896usize, 4864]),
            (GGML_TYPE_Q5_0, [896, 4864]),
            (GGML_TYPE_IQ4_NL, [896, 4864]),
            (GGML_TYPE_PQ2_0, [128, 5120]),
            (GGML_TYPE_PTQ1_0, [128, 5120]),
        ] {
            for in_dim in widths {
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
        // `Q4_0` is symmetric per-32 and qualifies; `Q5_1` looks identical
        // on the wire but carries a per-block `m`, which this kernel has no
        // term for — claiming it here would silently drop the offset from
        // every weight.
        assert!(supports_flat(GGML_TYPE_Q4_0, 896));
        assert!(!supports_flat(GGML_TYPE_Q5_1, 896));
        // The K-quants are routed to `dot_k_pair` before this is consulted,
        // and this kernel has no min correction and no per-GROUP branch, so
        // it must decline them rather than compute them wrongly.
        assert!(!supports_flat(GGML_TYPE_Q4_K, 2048));
        assert!(!supports_flat(GGML_TYPE_Q6_K, 2048));
        // A row that is not a whole number of 32-element blocks.
        assert!(!supports_flat(GGML_TYPE_Q8_0, 100));
    }
}

#[cfg(test)]
mod rowi8_tests {
    use super::*;

    fn data(n: usize, salt: u64) -> Vec<f32> {
        let mut s = 0x9E37_79B9_7F4A_7C15u64 ^ salt;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0
            })
            .collect()
    }

    /// The `smmla` tile and its portable twin agree, and both track the exact
    /// `f32` product to the quantization — with three activation channels at
    /// 300× the rest, the case one scale per token (`rten-gemm`) cannot
    /// carry. Awkward shapes: rows not a multiple of eight, tokens not of
    /// the tile.
    #[test]
    fn the_rowi8_product_tracks_the_f32_one_with_outliers() {
        // Eight 256-element super-blocks, two of them carrying the outliers:
        // the activations' scale is per super-block (as for every K-quant
        // product), so an outlier's block is coarse and the others are not.
        let (in_dim, out_dim, n) = (2048usize, 21usize, 13usize);
        let w = data(in_dim * out_dim, 1);
        let mut x = data(n * in_dim, 2);
        for t in 0..n {
            for c in [7usize, 300, 511] {
                x[t * in_dim + c] *= 300.0;
            }
        }
        let rows = RowI8::quantize(out_dim, in_dim, |o| {
            w[o * in_dim..(o + 1) * in_dim].to_vec()
        });
        let got = matmul_rowi8(&x, n, &rows);
        let acts = ActQ8Mm::quantize(&x, in_dim, n);
        let mut portable = vec![0f32; n * out_dim];
        for g in 0..rows.groups {
            let mut yt = vec![0f32; ROW_OCT * n];
            {
                let mut slices: Vec<&mut [f32]> = yt.chunks_mut(n).collect();
                rowi8_tiles_portable(&rows, g, &acts, 0..acts.n_tiles(), &mut slices);
            }
            for r in 0..ROW_OCT.min(out_dim - g * ROW_OCT) {
                for t in 0..n {
                    portable[t * out_dim + g * ROW_OCT + r] = yt[r * n + t];
                }
            }
        }
        // The activations as `ActQ8Mm` quantizes them — per 256, the part of
        // the error every K-quant product shares — so what is measured
        // against it is this format's own: the weights' per-row `int8`.
        let mut xq = x.clone();
        for row in xq.chunks_mut(SUPER_BLOCK) {
            let scale = row.iter().fold(0f32, |m, v| m.max(v.abs())) / 127.0;
            for v in row.iter_mut() {
                *v = if scale > 0.0 {
                    (*v / scale).round().clamp(-127.0, 127.0) * scale
                } else {
                    0.0
                };
            }
        }
        let (mut se, mut ss, mut se_full, mut worst_twin) = (0f64, 0f64, 0f64, 0f32);
        for t in 0..n {
            for o in 0..out_dim {
                let dot = |a: &[f32]| -> f32 {
                    (0..in_dim)
                        .map(|k| a[t * in_dim + k] * w[o * in_dim + k])
                        .sum()
                };
                let (want_q, want) = (dot(&xq), dot(&x));
                let g = got[t * out_dim + o];
                se += ((g - want_q) as f64).powi(2);
                ss += (want_q as f64).powi(2);
                se_full += ((g - want) as f64).powi(2);
                let p = portable[t * out_dim + o];
                worst_twin = worst_twin.max((g - p).abs() / p.abs().max(1.0));
            }
        }
        let rel = (se / ss).sqrt();
        let rel_full = (se_full / ss).sqrt();
        assert!(worst_twin < 1e-4, "tile vs portable {worst_twin}");
        assert!(rel < 0.01, "the weights' own relative RMS error {rel}");
        assert!(rel_full < 0.05, "against exact f32 {rel_full}");
    }

    /// `matmul_rowi8` at a Qwen-Image 2.1 step's shapes, against
    /// `k_gemm_throughput`'s `Q4_K` numbers:
    ///
    /// ```text
    /// ORANGU_BENCH_SHAPE=4096,24576,4096 cargo test --profile release-with-debug \
    ///     --bin orangu-server rowi8_throughput -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore]
    fn rowi8_throughput() {
        let (in_dim, out_dim, n_tokens) = std::env::var("ORANGU_BENCH_SHAPE")
            .ok()
            .and_then(|v| {
                let mut it = v.split(',').map(|p| p.trim().parse::<usize>().ok());
                Some((it.next()??, it.next()??, it.next()??))
            })
            .unwrap_or((4096, 4096, 4096));
        let w = data(in_dim * out_dim, 3);
        let rows = RowI8::quantize(out_dim, in_dim, |o| {
            w[o * in_dim..(o + 1) * in_dim].to_vec()
        });
        drop(w);
        let x = data(n_tokens * in_dim, 4);
        let _ = matmul_rowi8(&x, n_tokens, &rows);
        let reps = 3;
        let started = std::time::Instant::now();
        for _ in 0..reps {
            let _ = matmul_rowi8(&x, n_tokens, &rows);
        }
        let secs = started.elapsed().as_secs_f64() / reps as f64;
        eprintln!(
            "rowi8 {in_dim}x{out_dim} x {n_tokens} tokens: {:.1} ms, {:.1} G MAC/s",
            secs * 1e3,
            (in_dim * out_dim * n_tokens) as f64 / secs / 1e9
        );
    }
}

#[cfg(test)]
mod i8_scores_tests {
    use super::*;

    /// The four-rows score kernel is the exact integer product of the
    /// quantized rows, over a key count that leaves both whole groups of
    /// four pairs and a tail.
    #[test]
    fn i8_scores_4rows_is_the_exact_integer_product() {
        let dim = 24;
        let rows: Vec<Vec<f32>> = (0..5)
            .map(|r| {
                (0..dim)
                    .map(|c| ((r * 31 + c * 7) % 23) as f32 - 11.0)
                    .collect()
            })
            .collect();
        let keys: Vec<Vec<f32>> = (0..14)
            .map(|r| {
                (0..dim)
                    .map(|c| ((r * 13 + c * 5) % 19) as f32 - 9.0)
                    .collect()
            })
            .collect();
        let q = PairedI8::quantize(rows.len(), dim, |i| &rows[i]);
        let k = PairedI8::quantize(keys.len(), dim, |i| &keys[i]);
        let at = |m: &PairedI8, row: usize, c: usize| -> i32 {
            m.data[(row / 2) * 2 * dim + (c / 8) * 16 + (row % 2) * 8 + c % 8] as i32
        };
        let mut out = vec![vec![0i32; 2 * k.pairs]; 4];
        let [o0, o1, o2, o3] = &mut out[..] else {
            unreachable!()
        };
        i8_scores_4rows(&q, 0, 1, &k, [o0, o1, o2, o3]);
        for (r, row) in [0, 1, 2, 3].into_iter().zip(&out) {
            for (j, &got) in row.iter().enumerate().take(keys.len()) {
                let want: i32 = (0..dim).map(|c| at(&q, r, c) * at(&k, j, c)).sum();
                assert_eq!(got, want, "row {r} key {j}");
            }
        }
    }
}

#[cfg(test)]
mod bf16_tests {
    use super::*;

    /// `matmul_bf16` against the `f32` product: token counts and row counts
    /// that are not whole tiles or whole tasks, the output token-major.
    /// Within `bf16` rounding of both operands (8 bits of mantissa each).
    #[test]
    fn matmul_bf16_matches_the_f32_product() {
        if !have_bf16mm() {
            return;
        }
        for &(n_tokens, in_dim, out_dim) in
            &[(1usize, 64usize, 8usize), (17, 256, 37), (40, 1536, 300)]
        {
            let x: Vec<f32> = (0..n_tokens * in_dim)
                .map(|i| ((i * 31 % 97) as f32 - 48.0) / 48.0)
                .collect();
            let w: Vec<f32> = (0..out_dim * in_dim)
                .map(|i| ((i * 17 % 89) as f32 - 44.0) / 44.0)
                .collect();
            let packed = Bf16Weights::from_rows(out_dim, in_dim, |o| {
                w[o * in_dim..(o + 1) * in_dim].to_vec()
            });
            let got = matmul_bf16(&x, n_tokens, &packed);
            assert_eq!(got.len(), n_tokens * out_dim);
            for t in 0..n_tokens {
                for o in 0..out_dim {
                    let xs = &x[t * in_dim..(t + 1) * in_dim];
                    let ws = &w[o * in_dim..(o + 1) * in_dim];
                    let want: f32 = xs.iter().zip(ws).map(|(a, b)| a * b).sum();
                    let scale: f32 = xs.iter().zip(ws).map(|(a, b)| (a * b).abs()).sum();
                    let g = got[t * out_dim + o];
                    assert!(
                        (g - want).abs() <= 1e-2 * scale.max(1.0),
                        "{n_tokens}x{in_dim}x{out_dim} token {t} row {o}: {g} against {want}"
                    );
                }
            }
        }
    }

    /// The `bfmmla` tiles match their portable twin exactly (both sum
    /// `bf16` products in `f32`; the order differs only within a lane's
    /// four) to rounding, and the exact `f32` product to `bf16`'s eight bits.
    #[test]
    fn bf16_tiles_match_the_twin_and_the_f32_product() {
        let (n_a, n_b, k) = (13usize, 11usize, 37usize);
        let a: Vec<f32> = (0..n_a * k)
            .map(|i| ((i * 7 % 23) as f32 - 11.0) * 0.05)
            .collect();
        let b: Vec<f32> = (0..n_b * k)
            .map(|i| ((i * 5 % 19) as f32 - 9.0) * 0.07)
            .collect();
        let pa = PackedBf16::pack(n_a, k, |r, i| a[r * k + i]);
        let pb = PackedBf16::pack(n_b, k, |r, i| b[r * k + i]);
        let mut got = vec![0f32; pa.rows * pb.rows];
        bf16_tiles(&pa, &pb, &mut got);
        let mut twin = vec![0f32; pa.rows * pb.rows];
        bf16_tiles_portable(&pa, &pb, 0, &mut twin);
        for i in 0..n_a {
            for j in 0..n_b {
                let (g, t) = (got[i * pb.rows + j], twin[i * pb.rows + j]);
                assert!(
                    (g - t).abs() <= 1e-4 * t.abs().max(1.0),
                    "{i},{j}: {g} vs {t}"
                );
                let exact: f32 = (0..k).map(|e| a[i * k + e] * b[j * k + e]).sum();
                assert!(
                    (g - exact).abs() <= 0.02 * exact.abs().max(0.5),
                    "{i},{j}: {g} vs {exact}"
                );
            }
        }
        eprintln!("bfmmla in use: {}", have_bf16mm());
    }

    /// `pack_transposed` is `pack` of the transpose, strided source and
    /// odd sizes included.
    #[test]
    fn pack_transposed_is_pack_of_the_transpose() {
        let (n_rows, k, stride) = (13usize, 11usize, 20usize);
        let src: Vec<f32> = (0..k * stride).map(|i| (i as f32 * 0.21).cos()).collect();
        let want = PackedBf16::pack(n_rows, k, |r, j| src[j * stride + r]);
        let got = PackedBf16::pack_transposed(n_rows, k, &src, stride);
        assert_eq!(got.data, want.data);
    }

    /// `set_row_exp` is `set_row` of the exponentials, and returns their
    /// sum.
    #[test]
    fn set_row_exp_is_set_row_of_the_exponentials() {
        let k = 37usize;
        let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.7).sin() * 3.0).collect();
        for len in [k, k - 1, k - 3, 4, 1] {
            let max = x[..len].iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut want = PackedBf16::zeroed(3, k);
            let mut e: Vec<f32> = x[..len].to_vec();
            let sum = crate::engine::tensor::exp_shifted_sum(&mut e, max);
            want.set_row(1, &e);
            let mut got = PackedBf16::pack(3, k, |_, _| 5.0);
            got.set_row(0, &[0.0; 0]);
            got.set_row(2, &[0.0; 0]);
            let got_sum = got.set_row_exp(1, &x[..len], max);
            // The body is `exp_for_bf16`'s cubic where `bf16` is available:
            // ~6e-4 relative, far inside the rounding it goes through.
            assert!(
                (got_sum - sum).abs() <= 1e-3 * sum,
                "{len}: {got_sum} vs {sum}"
            );
            for (i, (&a, &b)) in got.data.iter().zip(&want.data).enumerate() {
                // The tail's exponentials are `libm`'s, the body's a
                // polynomial's: two `bf16` units apart at most.
                assert!(a.abs_diff(b) <= 2, "{len} at {i}: {a:#x} vs {b:#x}");
            }
        }
    }

    /// Issue rates of `bfmmla` against `smmla` (sixteen independent
    /// accumulators, registers only) on the calling core:
    ///
    /// ```text
    /// taskset -c 0 <test binary> mmla_issue_rates --ignored --nocapture
    /// ```
    #[test]
    #[ignore]
    #[cfg(target_arch = "aarch64")]
    fn mmla_issue_rates() {
        use std::arch::aarch64::*;
        let reps = 20_000_000u64;
        unsafe {
            let mut f = [vdupq_n_f32(0.0); 16];
            let a = vdupq_n_u16(0x3f80);
            let t = std::time::Instant::now();
            for _ in 0..reps {
                for x in f.iter_mut() {
                    *x = bfmmla(*x, a, a);
                }
            }
            let bf = t.elapsed().as_secs_f64();
            std::hint::black_box(&f);
            let mut s = [vdupq_n_s32(0); 16];
            let b = vdupq_n_s8(1);
            let t = std::time::Instant::now();
            for _ in 0..reps {
                for x in s.iter_mut() {
                    *x = mmla_2x8(*x, b, b);
                }
            }
            let sm = t.elapsed().as_secs_f64();
            std::hint::black_box(&s);
            let n = (reps * 16) as f64;
            eprintln!(
                "bfmmla {:.2} G/s ({:.1} G MAC/s), smmla {:.2} G/s ({:.1} G MAC/s)",
                n / bf / 1e9,
                16.0 * n / bf / 1e9,
                n / sm / 1e9,
                32.0 * n / sm / 1e9
            );
        }
    }

    /// A row written with `set_row` over whatever was there is the row
    /// `pack` makes, shorter rows padded with zeros.
    #[test]
    fn set_row_is_pack_row_by_row() {
        let (rows, k) = (9usize, 37usize);
        let a: Vec<f32> = (0..rows * k).map(|i| (i as f32 * 0.37).sin()).collect();
        let lens = |r: usize| k - (r * 5) % 13;
        let want = PackedBf16::pack(rows, k, |r, i| if i < lens(r) { a[r * k + i] } else { 0.0 });
        let mut got = PackedBf16::pack(rows, k, |_, _| 7.0);
        for r in 0..rows {
            got.set_row(r, &a[r * k..r * k + lens(r)]);
        }
        assert_eq!(got.data, want.data);
    }
}
