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

//! The compute kernels the three **vendor** backends compile at startup -
//! `engine::backend::cuda`, `engine::backend::rocm` and
//! `engine::backend::opencl` - written once and rendered per dialect.
//!
//! `engine::backend::vulkan_shaders` is the model: a shared I/O prelude, a
//! per-quantization *middle*, and an algorithm *suffix*, assembled so that
//! "a new decode algorithm costs one suffix, and a new quantization costs one
//! middle". The three vendor backends had that same three-part shape, but by
//! **copy** - the CUDA and HIP sources were byte-identical to each other and
//! the OpenCL one differed only in how C spells its types. Adding a
//! quantization meant writing the same arithmetic three times and keeping
//! three copies in step; nobody did, which is part of why these backends
//! cover 12 `ggml_type`s against `vulkan`'s 22.
//!
//! So the arithmetic lives here once, in a neutral spelling, and a
//! [`Dialect`] renders it. The difference between CUDA/HIP and OpenCL is
//! entirely a token table: `unsigned int` against `uint`, `__shared__`
//! against `__local`, `__syncthreads()` against
//! `barrier(CLK_LOCAL_MEM_FENCE)`. There is no algorithmic difference
//! between the three backends and there never was.
//!
//! **This module is testable where the backends are not.** Rendering a
//! kernel needs no device, no driver and no vendor SDK, so the tests below
//! run on every machine - including this project's own, which has none of
//! the three runtimes. They cannot tell you the kernel is *fast*; they can
//! tell you it exists for every type, that no placeholder leaked into the
//! emitted source, and that CUDA and HIP still agree.
//!
//! What they cannot check is the arithmetic. That rests on the same footing
//! it always did: each middle is a transcription of `engine::quant`'s
//! dequantizer for that type, and the `matmul_matches_cpu_backend_for_*`
//! cross-checks in each backend's own test module are what actually prove
//! it - on hardware this project does not have.

/// A C-family dialect one of the vendor backends compiles.
///
/// [`Dialect::Cuda`] and [`Dialect::Hip`] render **identically** - HIP-C is
/// CUDA-C for everything these kernels use, which is why the two backends'
/// sources were byte-identical before this module existed. They are still two
/// variants rather than one so that a future divergence has somewhere to go,
/// and `cuda_and_hip_render_identically` holds them together until then.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Dialect {
    Cuda,
    #[cfg_attr(
        not(feature = "rocm"),
        allow(dead_code, reason = "the rocm backend is feature-gated")
    )]
    Hip,
    OpenCl,
}

/// The entry point every rendered kernel declares.
pub(crate) const KERNEL_NAME: &str = "matmul_reduce";

/// The `ggml_type`s these backends have a kernel for.
///
/// Deliberately a *subset* of what `engine::quant` reads on the CPU path and
/// of what `engine::backend::vulkan` covers on the GPU: the remaining `IQ*`
/// types index lattice codebooks that need their own uploaded buffer, which
/// `vulkan` has (`IQ_GRID_PRELUDE`) and these backends do not. `IQ4_NL` is
/// the exception - a 16-entry level table small enough to inline into the
/// kernel source. Anything absent is rejected by `Backend::supports_type` at
/// startup rather than reaching `matmul`.
///
/// One list rather than three identical copies: a type added here is added to
/// all three backends at once, which is the whole point of this module.
pub(crate) const SUPPORTED_TYPES: &[u32] = &[
    crate::engine::quant::GGML_TYPE_F32,
    crate::engine::quant::GGML_TYPE_F16,
    crate::engine::quant::GGML_TYPE_BF16,
    crate::engine::quant::GGML_TYPE_Q4_0,
    crate::engine::quant::GGML_TYPE_Q4_1,
    crate::engine::quant::GGML_TYPE_Q5_0,
    crate::engine::quant::GGML_TYPE_Q5_1,
    crate::engine::quant::GGML_TYPE_Q8_0,
    crate::engine::quant::GGML_TYPE_Q4_K,
    crate::engine::quant::GGML_TYPE_Q5_K,
    crate::engine::quant::GGML_TYPE_Q6_K,
    crate::engine::quant::GGML_TYPE_Q2_K,
    crate::engine::quant::GGML_TYPE_Q3_K,
    crate::engine::quant::GGML_TYPE_IQ1_S,
    crate::engine::quant::GGML_TYPE_IQ1_M,
    crate::engine::quant::GGML_TYPE_IQ2_XXS,
    crate::engine::quant::GGML_TYPE_IQ2_XS,
    crate::engine::quant::GGML_TYPE_IQ2_S,
    crate::engine::quant::GGML_TYPE_IQ3_XXS,
    crate::engine::quant::GGML_TYPE_IQ3_S,
    crate::engine::quant::GGML_TYPE_IQ4_NL,
    crate::engine::quant::GGML_TYPE_IQ4_XS,
];

/// `(placeholder, replacement)` for `dialect`.
///
/// Order does **not** matter, and that is a property of the placeholder
/// spelling rather than of this list: every token is delimited at both ends
/// (`@CONST@`, not `CONST`), so no token is a substring of another — the
/// nearest pair, `@CONST@` and `@CONSTARR_I8@`, differ at the character
/// where the first one closes. An undelimited spelling would have made
/// ordering load-bearing and a reordering silently corrupting; this one
/// cannot be got wrong that way.
///
/// What *can* go wrong is a token used in the neutral source and missing
/// from one dialect's table, which reaches the vendor compiler as a syntax
/// error on a machine this project cannot reproduce.
/// `no_placeholder_survives_rendering` is the guard for that.
fn tokens(dialect: Dialect) -> &'static [(&'static str, &'static str)] {
    match dialect {
        // HIP-C is CUDA-C for everything here. See [`Dialect`].
        Dialect::Cuda | Dialect::Hip => &[
            ("@CONSTARR_I8@", "__device__ const signed char"),
            ("@ASF_OPEN@", "__int_as_float((int)"),
            ("@GROUPID@", "blockIdx.x"),
            ("@LOCALID@", "threadIdx.x"),
            ("@BARRIER@", "__syncthreads()"),
            ("@KERNEL@", "extern \"C\" __global__ void"),
            ("@SHARED@", "__shared__"),
            ("@GF32C@", "const float *"),
            ("@GU32C@", "const unsigned int *"),
            ("@CONST@", "const"),
            ("@LOCAL@", "local"),
            ("@GF32@", "float *"),
            ("@DEV@", "extern \"C\" __device__"),
            ("@GU8@", "const unsigned char *"),
            ("@U32@", "unsigned int"),
            ("@U16@", "unsigned short"),
            ("@I8@", "signed char"),
            ("@U8@", "unsigned char"),
        ],
        Dialect::OpenCl => &[
            ("@CONSTARR_I8@", "constant char"),
            ("@ASF_OPEN@", "as_float("),
            ("@GROUPID@", "get_group_id(0)"),
            ("@LOCALID@", "get_local_id(0)"),
            ("@BARRIER@", "barrier(CLK_LOCAL_MEM_FENCE)"),
            ("@KERNEL@", "__kernel void"),
            ("@SHARED@", "__local"),
            ("@GF32C@", "__global const float *"),
            ("@GU32C@", "__global const uint *"),
            ("@CONST@", "constant"),
            // `local` is a reserved word in OpenCL C, so the variable the
            // CUDA source calls `local` has to be called something else.
            ("@LOCAL@", "local_id"),
            ("@GF32@", "__global float *"),
            ("@DEV@", "inline"),
            ("@GU8@", "__global const uchar *"),
            ("@U32@", "uint"),
            ("@U16@", "ushort"),
            ("@I8@", "char"),
            ("@U8@", "uchar"),
        ],
    }
}

/// The codebook accessors the lattice `IQ*` types need, appended to the
/// prelude only for those types — the same split
/// `vulkan_shaders::needs_iq_grids` makes, and for the same reason: a kernel
/// should declare only what it reads.
///
/// The word offsets are formatted in from `engine::iq_grids::packed`, which
/// is also what fills the buffer, so a table that grows cannot leave these
/// pointing at the old place.
///
/// Every kernel takes the codebook pointer whether or not it reads one. That
/// keeps a single launch path in all three backends — one argument list, one
/// bound buffer — at the cost of an unused parameter on most kernels, which
/// is nothing. The alternative was a second entry-point signature and a
/// branch at every launch site.
fn iq_grid_prelude() -> String {
    use crate::engine::iq_grids::packed;
    format!(
        r#"
@CONST@ @U32@ IQ2XS_GRID_OFF = {iq2xs}u;
@CONST@ @U32@ IQ2S_GRID_OFF = {iq2s}u;
@CONST@ @U32@ IQ3XXS_GRID_OFF = {iq3xxs}u;
@CONST@ @U32@ IQ3S_GRID_OFF = {iq3s}u;
@CONST@ @U32@ KSIGNS_OFF = {ksigns}u;
@CONST@ @U32@ IQ2XXS_GRID_OFF = {iq2xxs}u;
@CONST@ @U32@ IQ1S_GRID_OFF = {iq1s}u;

// Byte `j` (0..8) of the 8-element lattice point `idx` in an `iq2*` grid,
// which stores two `u32` words per entry.
@DEV@ @U32@ iq_grid8(@GU32C@g, @U32@ base, @U32@ idx, @U32@ j) {{
    @U32@ word = g[base + idx * 2u + (j >> 2)];
    return (word >> ((j & 3u) * 8u)) & 0xFFu;
}}

// Byte `j` (0..4) of the 4-element lattice point `idx` in an `iq3*` grid,
// one `u32` word per entry.
@DEV@ @U32@ iq_grid4(@GU32C@g, @U32@ base, @U32@ idx, @U32@ j) {{
    @U32@ word = g[base + idx];
    return (word >> ((j & 3u) * 8u)) & 0xFFu;
}}

// `ksigns_iq2xs[i]`: the 8 sign bits a 7-bit sign field expands to.
@DEV@ @U32@ iq_ksigns(@GU32C@g, @U32@ i) {{
    return (g[KSIGNS_OFF + (i >> 2)] >> ((i & 3u) * 8u)) & 0xFFu;
}}

// Byte `j` (0..8) of an `iq1*` lattice point, sign-extended from the
// `int8_t` it is stored as. The `iq1*` grids carry **signed** values and no
// sign field at all, unlike every `iq2*`/`iq3*` grid above.
@DEV@ float iq_grid8_signed(@GU32C@g, @U32@ base, @U32@ idx, @U32@ j) {{
    @U32@ b = iq_grid8(g, base, idx, j);
    int v = (int)b;
    if (v >= 128) {{
        v = v - 256;
    }}
    return (float)v;
}}

// `kmask_iq2xs[j]` is `1 << j`, so the sign of element `j` is bit `j`.
@DEV@ float iq_sign(@U32@ signs, @U32@ j) {{
    return ((signs & (1u << j)) != 0u) ? -1.0f : 1.0f;
}}

// The `+/-` offset every `iq1*` weight carries on top of its codebook value.
@CONST@ float IQ1_DELTA = {delta}f;
"#,
        iq2xs = packed::IQ2XS_GRID_OFF,
        iq2s = packed::IQ2S_GRID_OFF,
        iq3xxs = packed::IQ3XXS_GRID_OFF,
        iq3s = packed::IQ3S_GRID_OFF,
        ksigns = packed::KSIGNS_OFF,
        iq2xxs = packed::IQ2XXS_GRID_OFF,
        iq1s = packed::IQ1S_GRID_OFF,
        delta = packed::IQ1_DELTA,
    )
}

/// Whether `ggml_type`'s dequantizer reads the lattice codebooks, and so
/// needs [`iq_grid_prelude`]. `IQ4_NL` and `IQ4_XS` do **not**: their
/// 16-entry level table is small enough to carry in the shared prelude.
fn needs_iq_grids(ggml_type: u32) -> bool {
    use crate::engine::quant::*;
    matches!(
        ggml_type,
        t if t == GGML_TYPE_IQ1_S
            || t == GGML_TYPE_IQ1_M
            || t == GGML_TYPE_IQ2_XXS
            || t == GGML_TYPE_IQ2_XS
            || t == GGML_TYPE_IQ2_S
            || t == GGML_TYPE_IQ3_XXS
            || t == GGML_TYPE_IQ3_S
    )
}

/// Substitutes every placeholder in `neutral` for `dialect`.
fn render(neutral: &str, dialect: Dialect) -> String {
    let mut out = neutral.to_string();
    for (placeholder, replacement) in tokens(dialect) {
        out = out.replace(placeholder, replacement);
    }
    out
}

/// The complete, compile-ready kernel source for `ggml_type` in `dialect`, or
/// `None` if there is no middle for that type.
///
/// Prelude, then that type's dequantizer, then the reduction - the same
/// three-part assembly `vulkan_shaders::shader_source_reduce` uses, and the
/// same order the three backends composed by hand before this module.
pub(crate) fn kernel_source(dialect: Dialect, ggml_type: u32) -> Option<String> {
    // The block-hoisted pair when this type has one, unless opted out.
    // `ORANGU_NO_BLOCK_HOISTED` is deliberately the *same* variable
    // `engine::backend::vulkan` reads for the same choice: it is one question
    // — "decode a block header once per block, or once per element?" — and a
    // second spelling of it would be a second thing to remember.
    let prelude = if needs_iq_grids(ggml_type) {
        format!("{NEUTRAL_PRELUDE}\n{}", iq_grid_prelude())
    } else {
        NEUTRAL_PRELUDE.to_string()
    };
    if !crate::engine::env::flag_on("ORANGU_NO_BLOCK_HOISTED")
        && let Some(block) = neutral_block_middle(ggml_type)
    {
        return Some(render(
            &format!("{prelude}\n{block}\n{NEUTRAL_MAIN_BLOCK_HOISTED}"),
            dialect,
        ));
    }
    let middle = neutral_middle(ggml_type)?;
    Some(render(
        &format!("{prelude}\n{middle}\n{NEUTRAL_MAIN_REDUCE}"),
        dialect,
    ))
}

/// `ggml_type`'s `block_dot`, in the neutral spelling, or `None` for a type
/// that has no block-hoisted kernel and stays on the element-wise path.
///
/// The 32-element legacy family only. The K-quants are a deliberate
/// omission, not an oversight: `vulkan` does not give them a block-hoisted
/// kernel either — they have their own tuned `*-light` kernels there, which
/// these backends have no port of. So a `Q4_K` model, which is most models,
/// still decodes element-wise here. That is the largest remaining gap on
/// these backends and it is recorded in `PARITY.md`, not hidden in a match
/// arm.
///
/// `Q2_K` and `Q3_K` are the other omission, and a smaller one: both are
/// 256-element blocks whose per-sub-block scales need unpacking that has no
/// counterpart in the legacy family's single header, so neither is the
/// mechanical transcription the six below are.
fn neutral_block_middle(ggml_type: u32) -> Option<&'static str> {
    use crate::engine::quant::*;
    Some(match ggml_type {
        t if t == GGML_TYPE_Q4_0 => GGML_TYPE_Q4_0_BLOCK_MIDDLE,
        t if t == GGML_TYPE_Q4_1 => GGML_TYPE_Q4_1_BLOCK_MIDDLE,
        t if t == GGML_TYPE_Q5_0 => GGML_TYPE_Q5_0_BLOCK_MIDDLE,
        t if t == GGML_TYPE_Q5_1 => GGML_TYPE_Q5_1_BLOCK_MIDDLE,
        t if t == GGML_TYPE_Q8_0 => GGML_TYPE_Q8_0_BLOCK_MIDDLE,
        t if t == GGML_TYPE_IQ1_S => GGML_TYPE_IQ1_S_BLOCK_MIDDLE,
        t if t == GGML_TYPE_IQ1_M => GGML_TYPE_IQ1_M_BLOCK_MIDDLE,
        t if t == GGML_TYPE_IQ2_XXS => GGML_TYPE_IQ2_XXS_BLOCK_MIDDLE,
        t if t == GGML_TYPE_IQ2_XS => GGML_TYPE_IQ2_XS_BLOCK_MIDDLE,
        t if t == GGML_TYPE_IQ2_S => GGML_TYPE_IQ2_S_BLOCK_MIDDLE,
        t if t == GGML_TYPE_IQ3_XXS => GGML_TYPE_IQ3_XXS_BLOCK_MIDDLE,
        t if t == GGML_TYPE_IQ3_S => GGML_TYPE_IQ3_S_BLOCK_MIDDLE,
        t if t == GGML_TYPE_IQ4_XS => GGML_TYPE_IQ4_XS_BLOCK_MIDDLE,
        t if t == GGML_TYPE_IQ4_NL => GGML_TYPE_IQ4_NL_BLOCK_MIDDLE,
        t if t == GGML_TYPE_Q2_K => GGML_TYPE_Q2_K_BLOCK_MIDDLE,
        t if t == GGML_TYPE_Q3_K => GGML_TYPE_Q3_K_BLOCK_MIDDLE,
        t if t == GGML_TYPE_Q4_K => GGML_TYPE_Q4_K_BLOCK_MIDDLE,
        t if t == GGML_TYPE_Q5_K => GGML_TYPE_Q5_K_BLOCK_MIDDLE,
        t if t == GGML_TYPE_Q6_K => GGML_TYPE_Q6_K_BLOCK_MIDDLE,
        _ => return None,
    })
}

/// `ggml_type`'s `dequant_element`, in the neutral spelling.
fn neutral_middle(ggml_type: u32) -> Option<&'static str> {
    use crate::engine::quant::*;
    Some(match ggml_type {
        t if t == GGML_TYPE_F32 => GGML_TYPE_F32_MIDDLE,
        t if t == GGML_TYPE_F16 => GGML_TYPE_F16_MIDDLE,
        t if t == GGML_TYPE_BF16 => GGML_TYPE_BF16_MIDDLE,
        t if t == GGML_TYPE_Q4_0 => GGML_TYPE_Q4_0_MIDDLE,
        t if t == GGML_TYPE_Q4_1 => GGML_TYPE_Q4_1_MIDDLE,
        t if t == GGML_TYPE_Q5_0 => GGML_TYPE_Q5_0_MIDDLE,
        t if t == GGML_TYPE_Q5_1 => GGML_TYPE_Q5_1_MIDDLE,
        t if t == GGML_TYPE_Q8_0 => GGML_TYPE_Q8_0_MIDDLE,
        t if t == GGML_TYPE_Q4_K => GGML_TYPE_Q4_K_MIDDLE,
        t if t == GGML_TYPE_Q5_K => GGML_TYPE_Q5_K_MIDDLE,
        t if t == GGML_TYPE_Q6_K => GGML_TYPE_Q6_K_MIDDLE,
        t if t == GGML_TYPE_Q2_K => GGML_TYPE_Q2_K_MIDDLE,
        t if t == GGML_TYPE_Q3_K => GGML_TYPE_Q3_K_MIDDLE,
        t if t == GGML_TYPE_IQ1_S => GGML_TYPE_IQ1_S_MIDDLE,
        t if t == GGML_TYPE_IQ1_M => GGML_TYPE_IQ1_M_MIDDLE,
        t if t == GGML_TYPE_IQ2_XXS => GGML_TYPE_IQ2_XXS_MIDDLE,
        t if t == GGML_TYPE_IQ2_XS => GGML_TYPE_IQ2_XS_MIDDLE,
        t if t == GGML_TYPE_IQ2_S => GGML_TYPE_IQ2_S_MIDDLE,
        t if t == GGML_TYPE_IQ3_XXS => GGML_TYPE_IQ3_XXS_MIDDLE,
        t if t == GGML_TYPE_IQ3_S => GGML_TYPE_IQ3_S_MIDDLE,
        t if t == GGML_TYPE_IQ4_NL => GGML_TYPE_IQ4_NL_MIDDLE,
        t if t == GGML_TYPE_IQ4_XS => GGML_TYPE_IQ4_XS_MIDDLE,
        _ => return None,
    })
}

/// Shared by every type's kernel: the manual (no vendor intrinsic) IEEE-754
/// binary16 and bfloat16 decoders, and ggml's `get_scale_min_k4`. Written
/// without `__half`/`half` on purpose - the same source has to compile in
/// three dialects, and only one of them has that type.
const NEUTRAL_PRELUDE: &str = r#"
@DEV@ float orangu_half_to_float(@U16@ h) {
    @U32@ sign = ((@U32@)(h & 0x8000u)) << 16;
    @U32@ exp = (h >> 10) & 0x1Fu;
    @U32@ mant = h & 0x3FFu;
    @U32@ bits;
    if (exp == 0u) {
        if (mant == 0u) {
            bits = sign;
        } else {
            int e = -1;
            do {
                mant <<= 1;
                e++;
            } while ((mant & 0x400u) == 0u);
            mant &= 0x3FFu;
            bits = sign | ((@U32@)(127 - 15 - e) << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
    }
    return @ASF_OPEN@bits);
}

// bfloat16 -> f32: the top 16 bits of an f32, left-shifted into place —
// mirrors `quant::dequantize`'s `GGML_TYPE_BF16` arm exactly.
@DEV@ float orangu_bf16_to_float(@U16@ h) {
    @U32@ bits = ((@U32@)h) << 16;
    return @ASF_OPEN@bits);
}

// The 16 non-uniformly spaced levels an `IQ4_NL`/`IQ4_XS` nibble selects
// between — `engine::iq_grids::KVALUES_IQ4NL`, transcribed. Small enough to
// carry in every kernel, unlike the `IQ*` lattice codebooks; three middles
// read it (`IQ4_NL` element-wise and block-hoisted, and `IQ4_XS`), which is
// why it is here rather than three times over.
@CONSTARR_I8@ ORANGU_KVALUES_IQ4NL[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
@DEV@ float orangu_iq4_kvalue(@U32@ i) {
    return (float)ORANGU_KVALUES_IQ4NL[i];
}

// ggml's `get_scale_min_k4`: unpacks the 6-bit scale and 6-bit min for
// sub-block `j` (0..8) of a Q4_K/Q5_K super-block's 12-byte `scales` region
// starting at byte `base`. Mirrors `quant::get_scale_min_k4` exactly.
@DEV@ void orangu_get_scale_min_k4(
    @GU8@w, @U32@ base, @U32@ j,
    @U32@ *sc, @U32@ *m) {
    if (j < 4u) {
        *sc = w[base + j] & 63u;
        *m = w[base + j + 4u] & 63u;
    } else {
        *sc = (w[base + j + 4u] & 0xFu) | ((w[base + j - 4u] >> 6) << 4);
        *m = (w[base + j + 4u] >> 4) | ((w[base + j] >> 6) << 4);
    }
}
"#;

/// The compute entry point: one workgroup per (output-row group of 4, token)
/// pair, all 64 lanes splitting `in_dim` grid-stride style and reducing their
/// partial dot products in shared memory.
///
/// A direct port of `vulkan_shaders`'s `MAIN_REDUCE_SUFFIX`, and the *only*
/// dispatch strategy these backends have - `vulkan`'s cooperative, tiled and
/// block-hoisted variants have no counterpart here yet. It calls
/// `dequant_element` once **per element**, which for a K-quant re-decodes
/// that block's header for every one of its 256 elements. That is what a
/// block-hoisted suffix beside this one would fix; see `PARITY.md` C1.
const NEUTRAL_MAIN_REDUCE: &str = r#"
@KERNEL@ matmul_reduce(
    @GU8@weights,
    @GF32C@x,
    @GF32@y,
    @U32@ in_dim,
    @U32@ out_dim,
    @U32@ n_tokens,
    @U32@ row_bytes,
    @GU32C@iq_grids) {
    @SHARED@ float partial_sums[256];

    @U32@ n_row_groups = (out_dim + 3u) / 4u;
    @U32@ flat = @GROUPID@;
    if (flat >= n_row_groups * n_tokens) {
        return;
    }
    @U32@ rg = flat / n_tokens;
    @U32@ t = flat % n_tokens;
    @U32@ o0 = rg * 4u;
    @U32@ o1 = o0 + 1u;
    @U32@ o2 = o0 + 2u;
    @U32@ o3 = o0 + 3u;
    @U32@ @LOCAL@ = @LOCALID@;
    @U32@ x_base = t * in_dim;

    float partial0 = 0.0f;
    float partial1 = 0.0f;
    float partial2 = 0.0f;
    float partial3 = 0.0f;
    for (@U32@ k = @LOCAL@; k < in_dim; k += 64u) {
        @U32@ block_idx = k / BLOCK_ELEMS;
        @U32@ local_k = k % BLOCK_ELEMS;
        @U32@ block_off = block_idx * BLOCK_BYTES;
        float xv = x[x_base + k];
        partial0 += dequant_element(weights, iq_grids, o0 * row_bytes + block_off, local_k) * xv;
        if (o1 < out_dim) {
            partial1 += dequant_element(weights, iq_grids, o1 * row_bytes + block_off, local_k) * xv;
        }
        if (o2 < out_dim) {
            partial2 += dequant_element(weights, iq_grids, o2 * row_bytes + block_off, local_k) * xv;
        }
        if (o3 < out_dim) {
            partial3 += dequant_element(weights, iq_grids, o3 * row_bytes + block_off, local_k) * xv;
        }
    }

    partial_sums[@LOCAL@] = partial0;
    partial_sums[64u + @LOCAL@] = partial1;
    partial_sums[128u + @LOCAL@] = partial2;
    partial_sums[192u + @LOCAL@] = partial3;
    @BARRIER@;
    for (@U32@ stride = 32u; stride > 0u; stride /= 2u) {
        if (@LOCAL@ < stride) {
            partial_sums[@LOCAL@] += partial_sums[@LOCAL@ + stride];
            partial_sums[64u + @LOCAL@] += partial_sums[64u + @LOCAL@ + stride];
            partial_sums[128u + @LOCAL@] += partial_sums[128u + @LOCAL@ + stride];
            partial_sums[192u + @LOCAL@] += partial_sums[192u + @LOCAL@ + stride];
        }
        @BARRIER@;
    }
    if (@LOCAL@ == 0u) {
        y[t * out_dim + o0] = partial_sums[0];
        if (o1 < out_dim) {
            y[t * out_dim + o1] = partial_sums[64u];
        }
        if (o2 < out_dim) {
            y[t * out_dim + o2] = partial_sums[128u];
        }
        if (o3 < out_dim) {
            y[t * out_dim + o3] = partial_sums[192u];
        }
    }
}
"#;

/// The **block-hoisted** compute entry point: same workgroup geometry and
/// same reduction as [`NEUTRAL_MAIN_REDUCE`], but the inner loop walks whole
/// *blocks* through a per-type `block_dot` instead of individual elements
/// through `dequant_element`.
///
/// The point is the block header. `dequant_element` re-decodes a block's
/// scale — and for the `f16` types, through a hand-written IEEE-754 decoder,
/// because no vendor intrinsic is portable across three dialects — once per
/// element. `block_dot` decodes it once per call, and a call covers
/// `BLOCK_ELEMS / LANES_PER_BLOCK` elements.
///
/// **`LANES_PER_BLOCK` adjacent lanes share one block**, each taking a
/// contiguous slice of its bytes. Giving a lane a whole block to itself is
/// the obvious shape and it is the wrong one: `vulkan_shaders`'s own
/// `block_hoisted_suffix` measured that at **34% slower than the element-wise
/// path it was meant to beat**, because sixty-four lanes then read
/// sixty-four different byte runs at once and losing coalescing costs more
/// than the saved header decodes are worth. This is the shape that measured
/// faster there; whether it does here is unmeasured, for want of any of the
/// three runtimes.
///
/// Whole blocks only, with no tail: a quantized tensor is stored in whole
/// blocks, so `in_dim` is always a multiple of `BLOCK_ELEMS`. The
/// element-wise path needs no such assumption because its stride is the
/// workgroup rather than the block.
const NEUTRAL_MAIN_BLOCK_HOISTED: &str = r#"
@KERNEL@ matmul_reduce(
    @GU8@weights,
    @GF32C@x,
    @GF32@y,
    @U32@ in_dim,
    @U32@ out_dim,
    @U32@ n_tokens,
    @U32@ row_bytes,
    @GU32C@iq_grids) {
    @SHARED@ float partial_sums[256];

    @U32@ n_row_groups = (out_dim + 3u) / 4u;
    @U32@ flat = @GROUPID@;
    if (flat >= n_row_groups * n_tokens) {
        return;
    }
    @U32@ rg = flat / n_tokens;
    @U32@ t = flat % n_tokens;
    @U32@ o0 = rg * 4u;
    @U32@ o1 = o0 + 1u;
    @U32@ o2 = o0 + 2u;
    @U32@ o3 = o0 + 3u;
    @U32@ @LOCAL@ = @LOCALID@;
    @U32@ x_base = t * in_dim;

    float partial0 = 0.0f;
    float partial1 = 0.0f;
    float partial2 = 0.0f;
    float partial3 = 0.0f;

    @U32@ sub = @LOCAL@ % LANES_PER_BLOCK;
    @U32@ slot = @LOCAL@ / LANES_PER_BLOCK;
    @U32@ blocks_in_flight = 64u / LANES_PER_BLOCK;
    @U32@ n_blocks = in_dim / BLOCK_ELEMS;
    for (@U32@ b = slot; b < n_blocks; b += blocks_in_flight) {
        @U32@ block_off = b * BLOCK_BYTES;
        @U32@ x_off = x_base + b * BLOCK_ELEMS;
        partial0 += block_dot(weights, iq_grids, x, o0 * row_bytes + block_off, x_off, sub);
        if (o1 < out_dim) {
            partial1 += block_dot(weights, iq_grids, x, o1 * row_bytes + block_off, x_off, sub);
        }
        if (o2 < out_dim) {
            partial2 += block_dot(weights, iq_grids, x, o2 * row_bytes + block_off, x_off, sub);
        }
        if (o3 < out_dim) {
            partial3 += block_dot(weights, iq_grids, x, o3 * row_bytes + block_off, x_off, sub);
        }
    }

    partial_sums[@LOCAL@] = partial0;
    partial_sums[64u + @LOCAL@] = partial1;
    partial_sums[128u + @LOCAL@] = partial2;
    partial_sums[192u + @LOCAL@] = partial3;
    @BARRIER@;
    for (@U32@ stride = 32u; stride > 0u; stride /= 2u) {
        if (@LOCAL@ < stride) {
            partial_sums[@LOCAL@] += partial_sums[@LOCAL@ + stride];
            partial_sums[64u + @LOCAL@] += partial_sums[64u + @LOCAL@ + stride];
            partial_sums[128u + @LOCAL@] += partial_sums[128u + @LOCAL@ + stride];
            partial_sums[192u + @LOCAL@] += partial_sums[192u + @LOCAL@ + stride];
        }
        @BARRIER@;
    }
    if (@LOCAL@ == 0u) {
        y[t * out_dim + o0] = partial_sums[0];
        if (o1 < out_dim) {
            y[t * out_dim + o1] = partial_sums[64u];
        }
        if (o2 < out_dim) {
            y[t * out_dim + o2] = partial_sums[128u];
        }
        if (o3 < out_dim) {
            y[t * out_dim + o3] = partial_sums[192u];
        }
    }
}
"#;

/// `block_dot` for `GGML_TYPE_Q4_0` — [`NEUTRAL_MAIN_BLOCK_HOISTED`]'s per-type half,
/// transcribed from `vulkan_shaders`'s `GGML_TYPE_Q4_0_BLOCK_MIDDLE`, which is the
/// version measured on hardware.
///
/// This lane's contribution to one output row from one block: the header is
/// decoded once here, where `dequant_element` decodes it once per element.
const GGML_TYPE_Q4_0_BLOCK_MIDDLE: &str = r#"
@CONST@ @U32@ BLOCK_BYTES = 18u;
@CONST@ @U32@ BLOCK_ELEMS = 32u;
@CONST@ @U32@ LANES_PER_BLOCK = 8u;
@DEV@ float block_dot(@GU8@w, @GU32C@g, @GF32C@x, @U32@ byte_offset, @U32@ x_off, @U32@ sub) {
    float d = orangu_half_to_float((@U16@)w[byte_offset] | ((@U16@)w[byte_offset + 1] << 8));
    float acc = 0.0f;
    for (@U32@ m = 0u; m < 2u; m += 1u) {
        @U32@ j = sub * 2u + m;
        @U32@ byte = (@U32@)w[byte_offset + 2u + j];
        acc += ((float)((int)(byte & 0xFu) - 8) * d) * x[x_off + j];
        acc += ((float)((int)(byte >> 4) - 8) * d) * x[x_off + 16u + j];
    }
    return acc;
}
"#;

/// `block_dot` for `GGML_TYPE_Q4_1` — [`NEUTRAL_MAIN_BLOCK_HOISTED`]'s per-type half,
/// transcribed from `vulkan_shaders`'s `GGML_TYPE_Q4_1_BLOCK_MIDDLE`, which is the
/// version measured on hardware.
///
/// This lane's contribution to one output row from one block: the header is
/// decoded once here, where `dequant_element` decodes it once per element.
const GGML_TYPE_Q4_1_BLOCK_MIDDLE: &str = r#"
@CONST@ @U32@ BLOCK_BYTES = 20u;
@CONST@ @U32@ BLOCK_ELEMS = 32u;
@CONST@ @U32@ LANES_PER_BLOCK = 8u;
@DEV@ float block_dot(@GU8@w, @GU32C@g, @GF32C@x, @U32@ byte_offset, @U32@ x_off, @U32@ sub) {
    float d = orangu_half_to_float((@U16@)w[byte_offset] | ((@U16@)w[byte_offset + 1] << 8));
    float mn = orangu_half_to_float((@U16@)w[byte_offset + 2] | ((@U16@)w[byte_offset + 3] << 8));
    float acc = 0.0f;
    for (@U32@ m = 0u; m < 2u; m += 1u) {
        @U32@ j = sub * 2u + m;
        @U32@ byte = (@U32@)w[byte_offset + 4u + j];
        acc += ((float)(byte & 0xFu) * d + mn) * x[x_off + j];
        acc += ((float)(byte >> 4) * d + mn) * x[x_off + 16u + j];
    }
    return acc;
}
"#;

/// `block_dot` for `GGML_TYPE_Q5_0` — [`NEUTRAL_MAIN_BLOCK_HOISTED`]'s per-type half,
/// transcribed from `vulkan_shaders`'s `GGML_TYPE_Q5_0_BLOCK_MIDDLE`, which is the
/// version measured on hardware.
///
/// This lane's contribution to one output row from one block: the header is
/// decoded once here, where `dequant_element` decodes it once per element.
const GGML_TYPE_Q5_0_BLOCK_MIDDLE: &str = r#"
@CONST@ @U32@ BLOCK_BYTES = 22u;
@CONST@ @U32@ BLOCK_ELEMS = 32u;
@CONST@ @U32@ LANES_PER_BLOCK = 8u;
@DEV@ float block_dot(@GU8@w, @GU32C@g, @GF32C@x, @U32@ byte_offset, @U32@ x_off, @U32@ sub) {
    float d = orangu_half_to_float((@U16@)w[byte_offset] | ((@U16@)w[byte_offset + 1] << 8));
    @U32@ qh = (@U32@)w[byte_offset + 2] | ((@U32@)w[byte_offset + 3] << 8)
        | ((@U32@)w[byte_offset + 4] << 16) | ((@U32@)w[byte_offset + 5] << 24);
    float acc = 0.0f;
    for (@U32@ m = 0u; m < 2u; m += 1u) {
        @U32@ j = sub * 2u + m;
        @U32@ byte = (@U32@)w[byte_offset + 6u + j];
        @U32@ xh0 = ((qh >> j) << 4) & 0x10u;
        @U32@ xh1 = (qh >> (j + 12u)) & 0x10u;
        acc += ((float)((int)((byte & 0xFu) | xh0) - 16) * d) * x[x_off + j];
        acc += ((float)((int)((byte >> 4) | xh1) - 16) * d) * x[x_off + 16u + j];
    }
    return acc;
}
"#;

/// `block_dot` for `GGML_TYPE_Q5_1` — [`NEUTRAL_MAIN_BLOCK_HOISTED`]'s per-type half,
/// transcribed from `vulkan_shaders`'s `GGML_TYPE_Q5_1_BLOCK_MIDDLE`, which is the
/// version measured on hardware.
///
/// This lane's contribution to one output row from one block: the header is
/// decoded once here, where `dequant_element` decodes it once per element.
const GGML_TYPE_Q5_1_BLOCK_MIDDLE: &str = r#"
@CONST@ @U32@ BLOCK_BYTES = 24u;
@CONST@ @U32@ BLOCK_ELEMS = 32u;
@CONST@ @U32@ LANES_PER_BLOCK = 8u;
@DEV@ float block_dot(@GU8@w, @GU32C@g, @GF32C@x, @U32@ byte_offset, @U32@ x_off, @U32@ sub) {
    float d = orangu_half_to_float((@U16@)w[byte_offset] | ((@U16@)w[byte_offset + 1] << 8));
    float mn = orangu_half_to_float((@U16@)w[byte_offset + 2] | ((@U16@)w[byte_offset + 3] << 8));
    @U32@ qh = (@U32@)w[byte_offset + 4] | ((@U32@)w[byte_offset + 5] << 8)
        | ((@U32@)w[byte_offset + 6] << 16) | ((@U32@)w[byte_offset + 7] << 24);
    float acc = 0.0f;
    for (@U32@ m = 0u; m < 2u; m += 1u) {
        @U32@ j = sub * 2u + m;
        @U32@ byte = (@U32@)w[byte_offset + 8u + j];
        @U32@ xh0 = ((qh >> j) << 4) & 0x10u;
        @U32@ xh1 = (qh >> (j + 12u)) & 0x10u;
        acc += ((float)((byte & 0xFu) | xh0) * d + mn) * x[x_off + j];
        acc += ((float)((byte >> 4) | xh1) * d + mn) * x[x_off + 16u + j];
    }
    return acc;
}
"#;

/// `block_dot` for `GGML_TYPE_Q8_0` — [`NEUTRAL_MAIN_BLOCK_HOISTED`]'s per-type half,
/// transcribed from `vulkan_shaders`'s `GGML_TYPE_Q8_0_BLOCK_MIDDLE`, which is the
/// version measured on hardware.
///
/// This lane's contribution to one output row from one block: the header is
/// decoded once here, where `dequant_element` decodes it once per element.
const GGML_TYPE_Q8_0_BLOCK_MIDDLE: &str = r#"
@CONST@ @U32@ BLOCK_BYTES = 34u;
@CONST@ @U32@ BLOCK_ELEMS = 32u;
@CONST@ @U32@ LANES_PER_BLOCK = 8u;
@DEV@ float block_dot(@GU8@w, @GU32C@g, @GF32C@x, @U32@ byte_offset, @U32@ x_off, @U32@ sub) {
    float d = orangu_half_to_float((@U16@)w[byte_offset] | ((@U16@)w[byte_offset + 1] << 8));
    float acc = 0.0f;
    for (@U32@ m = 0u; m < 4u; m += 1u) {
        @U32@ j = sub * 4u + m;
        int v = (int)w[byte_offset + 2u + j];
        if (v >= 128) {
            v = v - 256;
        }
        acc += ((float)v * d) * x[x_off + j];
    }
    return acc;
}
"#;

/// `block_dot` for `GGML_TYPE_IQ4_NL` — [`NEUTRAL_MAIN_BLOCK_HOISTED`]'s per-type half,
/// transcribed from `vulkan_shaders`'s `GGML_TYPE_IQ4_NL_BLOCK_MIDDLE`, which is the
/// version measured on hardware.
///
/// This lane's contribution to one output row from one block: the header is
/// decoded once here, where `dequant_element` decodes it once per element.
const GGML_TYPE_IQ4_NL_BLOCK_MIDDLE: &str = r#"
@CONST@ @U32@ BLOCK_BYTES = 18u;
@CONST@ @U32@ BLOCK_ELEMS = 32u;
@CONST@ @U32@ LANES_PER_BLOCK = 8u;
@DEV@ float block_dot(@GU8@w, @GU32C@g, @GF32C@x, @U32@ byte_offset, @U32@ x_off, @U32@ sub) {
    float d = orangu_half_to_float((@U16@)w[byte_offset] | ((@U16@)w[byte_offset + 1] << 8));
    float acc = 0.0f;
    for (@U32@ m = 0u; m < 2u; m += 1u) {
        @U32@ j = sub * 2u + m;
        @U32@ byte = (@U32@)w[byte_offset + 2u + j];
        acc += (d * orangu_iq4_kvalue(byte & 0xFu)) * x[x_off + j];
        acc += (d * orangu_iq4_kvalue(byte >> 4)) * x[x_off + 16u + j];
    }
    return acc;
}
"#;

/// `block_dot` for `Q2_K` — transcribed from `vulkan_shaders`'s `Q2_K_BLOCK_MIDDLE`,
/// which is the version measured on hardware.
///
/// `LANES_PER_BLOCK` is 16 for every 256-element block: a lane owns the
/// contiguous run `[sub * 16, sub * 16 + 16)`, and four blocks are in flight
/// across the workgroup.
const GGML_TYPE_Q2_K_BLOCK_MIDDLE: &str = r#"
@CONST@ @U32@ BLOCK_BYTES = 84u;
@CONST@ @U32@ BLOCK_ELEMS = 256u;
@CONST@ @U32@ LANES_PER_BLOCK = 16u;
@DEV@ float block_dot(@GU8@w, @GU32C@g, @GF32C@x, @U32@ byte_offset, @U32@ x_off, @U32@ sub) {
    @U32@ scales_off = byte_offset;
    @U32@ qs_off = byte_offset + 16u;
    float d = orangu_half_to_float((@U16@)w[byte_offset + 80] | ((@U16@)w[byte_offset + 81] << 8));
    float dmin = orangu_half_to_float((@U16@)w[byte_offset + 82] | ((@U16@)w[byte_offset + 83] << 8));
    @U32@ n = sub / 8u;
    @U32@ s = (sub % 8u) / 2u;
    @U32@ h = sub % 2u;
    @U32@ sc = (@U32@)w[scales_off + n * 8u + s * 2u + h];
    float dl = d * (float)(sc & 0xFu);
    float ml = dmin * (float)(sc >> 4);
    @U32@ base = qs_off + n * 32u + h * 16u;
    @U32@ x_lane = x_off + sub * 16u;
    float acc = 0.0f;
    for (@U32@ l = 0u; l < 16u; l += 1u) {
        @U32@ byte = (@U32@)w[base + l];
        acc += (dl * (float)((byte >> (2u * s)) & 3u) - ml) * x[x_lane + l];
    }
    return acc;
}
"#;

/// `block_dot` for `Q3_K` — transcribed from `vulkan_shaders`'s `Q3_K_BLOCK_MIDDLE`,
/// which is the version measured on hardware.
///
/// `LANES_PER_BLOCK` is 16 for every 256-element block: a lane owns the
/// contiguous run `[sub * 16, sub * 16 + 16)`, and four blocks are in flight
/// across the workgroup.
const GGML_TYPE_Q3_K_BLOCK_MIDDLE: &str = r#"
@CONST@ @U32@ BLOCK_BYTES = 110u;
@CONST@ @U32@ BLOCK_ELEMS = 256u;
@CONST@ @U32@ LANES_PER_BLOCK = 16u;
@DEV@ @U32@ orangu_q3k_scale(@GU8@w, @U32@ base, @U32@ i) {
    @U32@ low;
    if (i < 8u) {
        low = w[base + i] & 0xFu;
    } else {
        low = w[base + i - 8u] >> 4;
    }
    @U32@ high = (w[base + 8u + (i % 4u)] >> (2u * (i / 4u))) & 3u;
    return low | (high << 4);
}
@DEV@ float block_dot(@GU8@w, @GU32C@g, @GF32C@x, @U32@ byte_offset, @U32@ x_off, @U32@ sub) {
    @U32@ hmask_off = byte_offset;
    @U32@ qs_off = byte_offset + 32u;
    @U32@ scales_off = byte_offset + 96u;
    float d_all = orangu_half_to_float((@U16@)w[byte_offset + 108] | ((@U16@)w[byte_offset + 109] << 8));
    @U32@ n = sub / 8u;
    @U32@ s = (sub % 8u) / 2u;
    @U32@ h = sub % 2u;
    @U32@ m = 1u << (n * 4u + s);
    float dl = d_all * (float)((int)orangu_q3k_scale(w, scales_off, n * 8u + s * 2u + h) - 32);
    @U32@ x_lane = x_off + sub * 16u;
    float acc = 0.0f;
    for (@U32@ l = 0u; l < 16u; l += 1u) {
        @U32@ idx = h * 16u + l;
        int hi = 4;
        if ((w[hmask_off + idx] & m) != 0u) {
            hi = 0;
        }
        @U32@ q = ((@U32@)w[qs_off + n * 32u + idx] >> (2u * s)) & 3u;
        acc += (dl * (float)((int)q - hi)) * x[x_lane + l];
    }
    return acc;
}
"#;

/// `block_dot` for `Q4_K` — the header hoisted out of `GGML_TYPE_Q4_K_MIDDLE`.
///
/// `vulkan_shaders` has no `Q4_K_BLOCK_MIDDLE` to transcribe: it gives the
/// K-quants a tuned `*-light` kernel instead, which these backends have no
/// port of. So this is derived from the element-wise version above rather
/// than from a measured one.
///
/// Every term except the quant byte is uniform across a lane's sixteen
/// elements — the 64-group `n`, which 32-half it falls in, and therefore the
/// `get_scale_min_k4` pair — because 16 divides both 32 and 64. That is what
/// makes the hoist exact rather than approximate: the lane's run cannot
/// straddle a scale boundary.
///
/// `LANES_PER_BLOCK` is 16 for every 256-element block: a lane owns the
/// contiguous run `[sub * 16, sub * 16 + 16)`, and four blocks are in flight
/// across the workgroup.
const GGML_TYPE_Q4_K_BLOCK_MIDDLE: &str = r#"
@CONST@ @U32@ BLOCK_BYTES = 144u;
@CONST@ @U32@ BLOCK_ELEMS = 256u;
@CONST@ @U32@ LANES_PER_BLOCK = 16u;
@DEV@ float block_dot(@GU8@w, @GU32C@g, @GF32C@x, @U32@ byte_offset, @U32@ x_off, @U32@ sub) {
    float d = orangu_half_to_float((@U16@)w[byte_offset] | ((@U16@)w[byte_offset + 1] << 8));
    float dmin = orangu_half_to_float((@U16@)w[byte_offset + 2] | ((@U16@)w[byte_offset + 3] << 8));
    @U32@ scales_off = byte_offset + 4u;
    @U32@ qs_off = byte_offset + 16u;
    @U32@ n = sub / 4u;
    @U32@ half_hi = (sub % 4u) / 2u;
    @U32@ lane_off = (sub % 2u) * 16u;
    @U32@ q_base = qs_off + n * 32u + lane_off;
    @U32@ sc, mn;
    orangu_get_scale_min_k4(w, scales_off, n * 2u + half_hi, &sc, &mn);
    float dl = d * (float)sc;
    float ml = dmin * (float)mn;
    @U32@ x_lane = x_off + sub * 16u;
    float acc = 0.0f;
    for (@U32@ l = 0u; l < 16u; l += 1u) {
        @U32@ byte = (@U32@)w[q_base + l];
        @U32@ q = (half_hi == 0u) ? (byte & 0xFu) : (byte >> 4);
        acc += (dl * (float)q - ml) * x[x_lane + l];
    }
    return acc;
}
"#;

/// `block_dot` for `Q5_K` — the header hoisted out of `GGML_TYPE_Q5_K_MIDDLE`.
///
/// `Q4_K` plus a fifth bit: a 32-byte `qh` plane between the scales and the
/// quants, and `+16` on a nibble whose `qh` bit is set. The bit selected
/// depends only on the 64-group and which 32-half, both uniform across a
/// lane, so the mask is hoisted with the scales. Derived, not transcribed —
/// see `GGML_TYPE_Q4_K_BLOCK_MIDDLE`.
///
/// `LANES_PER_BLOCK` is 16 for every 256-element block: a lane owns the
/// contiguous run `[sub * 16, sub * 16 + 16)`, and four blocks are in flight
/// across the workgroup.
const GGML_TYPE_Q5_K_BLOCK_MIDDLE: &str = r#"
@CONST@ @U32@ BLOCK_BYTES = 176u;
@CONST@ @U32@ BLOCK_ELEMS = 256u;
@CONST@ @U32@ LANES_PER_BLOCK = 16u;
@DEV@ float block_dot(@GU8@w, @GU32C@g, @GF32C@x, @U32@ byte_offset, @U32@ x_off, @U32@ sub) {
    float d = orangu_half_to_float((@U16@)w[byte_offset] | ((@U16@)w[byte_offset + 1] << 8));
    float dmin = orangu_half_to_float((@U16@)w[byte_offset + 2] | ((@U16@)w[byte_offset + 3] << 8));
    @U32@ scales_off = byte_offset + 4u;
    @U32@ qh_off = byte_offset + 16u;
    @U32@ qs_off = byte_offset + 48u;
    @U32@ n = sub / 4u;
    @U32@ half_hi = (sub % 4u) / 2u;
    @U32@ lane_off = (sub % 2u) * 16u;
    @U32@ q_base = qs_off + n * 32u + lane_off;
    @U32@ qh_base = qh_off + lane_off;
    @U32@ hmask = (half_hi == 0u) ? (1u << (2u * n)) : (2u << (2u * n));
    @U32@ sc, mn;
    orangu_get_scale_min_k4(w, scales_off, n * 2u + half_hi, &sc, &mn);
    float dl = d * (float)sc;
    float ml = dmin * (float)mn;
    @U32@ x_lane = x_off + sub * 16u;
    float acc = 0.0f;
    for (@U32@ l = 0u; l < 16u; l += 1u) {
        @U32@ byte = (@U32@)w[q_base + l];
        @U32@ q = (half_hi == 0u) ? (byte & 0xFu) : (byte >> 4);
        int hi_bit = ((@U32@)w[qh_base + l] & hmask) != 0u ? 16 : 0;
        acc += (dl * (float)((int)q + hi_bit) - ml) * x[x_lane + l];
    }
    return acc;
}
"#;

/// `block_dot` for `Q6_K` — the header hoisted out of `GGML_TYPE_Q6_K_MIDDLE`.
///
/// The 128-group, which of the four quant planes, and therefore the signed
/// 8-bit scale are all uniform across a lane, so `d * sc` is computed once
/// per block instead of once per element. Derived, not transcribed — see
/// `GGML_TYPE_Q4_K_BLOCK_MIDDLE`.
///
/// `LANES_PER_BLOCK` is 16 for every 256-element block: a lane owns the
/// contiguous run `[sub * 16, sub * 16 + 16)`, and four blocks are in flight
/// across the workgroup.
const GGML_TYPE_Q6_K_BLOCK_MIDDLE: &str = r#"
@CONST@ @U32@ BLOCK_BYTES = 210u;
@CONST@ @U32@ BLOCK_ELEMS = 256u;
@CONST@ @U32@ LANES_PER_BLOCK = 16u;
@DEV@ float block_dot(@GU8@w, @GU32C@g, @GF32C@x, @U32@ byte_offset, @U32@ x_off, @U32@ sub) {
    @U32@ ql_off = byte_offset;
    @U32@ qh_off = byte_offset + 128u;
    @U32@ sc_off = byte_offset + 192u;
    float d = orangu_half_to_float((@U16@)w[byte_offset + 208] | ((@U16@)w[byte_offset + 209] << 8));
    @U32@ idx = sub / 8u;
    @U32@ which_q = (sub % 8u) / 2u;
    @U32@ lane_off = (sub % 2u) * 16u;
    @U32@ ql_base = ql_off + idx * 64u + lane_off;
    @U32@ qh_base = qh_off + idx * 32u + lane_off;
    @I8@ sc = (@I8@)w[sc_off + idx * 8u + (sub % 2u) + which_q * 2u];
    float dl = d * (float)sc;
    @U32@ x_lane = x_off + sub * 16u;
    float acc = 0.0f;
    for (@U32@ l = 0u; l < 16u; l += 1u) {
        @U32@ ql_l = (@U32@)w[ql_base + l];
        @U32@ ql_l32 = (@U32@)w[ql_base + l + 32u];
        @U32@ qh_l = (@U32@)w[qh_base + l];
        int q;
        if (which_q == 0u) {
            q = (int)((ql_l & 0xFu) | ((qh_l & 3u) << 4)) - 32;
        } else if (which_q == 1u) {
            q = (int)((ql_l32 & 0xFu) | (((qh_l >> 2) & 3u) << 4)) - 32;
        } else if (which_q == 2u) {
            q = (int)((ql_l >> 4) | (((qh_l >> 4) & 3u) << 4)) - 32;
        } else {
            q = (int)((ql_l32 >> 4) | (((qh_l >> 6) & 3u) << 4)) - 32;
        }
        acc += (dl * (float)q) * x[x_lane + l];
    }
    return acc;
}
"#;

/// `dequant_element` for `IQ4_XS` — transcribed from `vulkan_shaders`'s
/// `IQ4_XS_COOP_MIDDLE`, which is the version measured on hardware.
///
/// `IQ4_NL`'s 16-level codebook with K-quant-style block structure: 256
/// elements, a 6-bit scale per 32-element sub-block split across a 2-byte
/// high field and a 4-byte low field. The **only** `IQ*` type besides
/// `IQ4_NL` that needs no lattice codebook, which is why it lands here ahead
/// of the other seven.
const GGML_TYPE_IQ4_XS_MIDDLE: &str = r#"
@CONST@ @U32@ BLOCK_BYTES = 136u;
@CONST@ @U32@ BLOCK_ELEMS = 256u;
@DEV@ float dequant_element(@GU8@w, @GU32C@g, @U32@ byte_offset, @U32@ k) {
    float d = orangu_half_to_float((@U16@)w[byte_offset] | ((@U16@)w[byte_offset + 1] << 8));
    @U32@ scales_h = (@U32@)w[byte_offset + 2] | ((@U32@)w[byte_offset + 3] << 8);
    @U32@ scales_l_off = byte_offset + 4u;
    @U32@ qs_off = byte_offset + 8u;
    @U32@ ib = k / 32u;
    @U32@ r = k % 32u;
    @U32@ low = ((@U32@)w[scales_l_off + ib / 2u] >> (4u * (ib % 2u))) & 0xFu;
    @U32@ high = (scales_h >> (2u * ib)) & 3u;
    float dl = d * (float)((int)(low | (high << 4)) - 32);
    @U32@ byte = (@U32@)w[qs_off + 16u * ib + (r % 16u)];
    @U32@ nib = (r < 16u) ? (byte & 0xFu) : (byte >> 4);
    return dl * orangu_iq4_kvalue(nib);
}
"#;

/// `dequant_element` for `IQ2_XXS` — transcribed from `vulkan_shaders`'s
/// `IQ2_XXS_COOP_MIDDLE`, which is the version measured on hardware.
///
/// An 8-element lattice point per 8 weights, indexed out of
/// `IQ2XXS_GRID`, with a 7-bit sign field expanded through `ksigns_iq2xs`
/// and a 4-bit per-32 scale packed into the top nibble of `aux1`.
const GGML_TYPE_IQ2_XXS_MIDDLE: &str = r#"
@CONST@ @U32@ BLOCK_BYTES = 66u;
@CONST@ @U32@ BLOCK_ELEMS = 256u;
@DEV@ float dequant_element(@GU8@w, @GU32C@g, @U32@ byte_offset, @U32@ k) {
    float d = orangu_half_to_float((@U16@)w[byte_offset] | ((@U16@)w[byte_offset + 1] << 8));
    @U32@ qs_off = byte_offset + 2u;
    @U32@ ib32 = k / 32u;
    @U32@ l = (k % 32u) / 8u;
    @U32@ j = k % 8u;
    @U32@ base = qs_off + 8u * ib32;
    @U32@ aux0 = (@U32@)w[base] | ((@U32@)w[base + 1u] << 8)
        | ((@U32@)w[base + 2u] << 16) | ((@U32@)w[base + 3u] << 24);
    @U32@ aux1 = (@U32@)w[base + 4u] | ((@U32@)w[base + 5u] << 8)
        | ((@U32@)w[base + 6u] << 16) | ((@U32@)w[base + 7u] << 24);
    float db = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;
    @U32@ idx = (aux0 >> (8u * l)) & 0xFFu;
    @U32@ signs = iq_ksigns(g, (aux1 >> (7u * l)) & 127u);
    @U32@ gv = iq_grid8(g, IQ2XXS_GRID_OFF, idx, j);
    return db * (float)gv * iq_sign(signs, j);
}
"#;

/// `dequant_element` for `IQ2_XS` — transcribed from `vulkan_shaders`'s
/// `IQ2_XS_COOP_MIDDLE`, which is the version measured on hardware.
///
/// `IQ2_XXS` with the lattice index and sign field packed into one 16-bit
/// word per 8 weights, and the scale in its own per-32 nibble.
const GGML_TYPE_IQ2_XS_MIDDLE: &str = r#"
@CONST@ @U32@ BLOCK_BYTES = 74u;
@CONST@ @U32@ BLOCK_ELEMS = 256u;
@DEV@ float dequant_element(@GU8@w, @GU32C@g, @U32@ byte_offset, @U32@ k) {
    float d = orangu_half_to_float((@U16@)w[byte_offset] | ((@U16@)w[byte_offset + 1] << 8));
    @U32@ qs_off = byte_offset + 2u;
    @U32@ scales_off = byte_offset + 66u;
    @U32@ ib32 = k / 32u;
    @U32@ l = (k % 32u) / 8u;
    @U32@ j = k % 8u;
    @U32@ qo = qs_off + 2u * (4u * ib32 + l);
    @U32@ q = (@U32@)w[qo] | ((@U32@)w[qo + 1u] << 8);
    @U32@ sc = ((@U32@)w[scales_off + ib32] >> (4u * (l / 2u))) & 0xFu;
    float db = d * (0.5f + (float)sc) * 0.25f;
    @U32@ gv = iq_grid8(g, IQ2XS_GRID_OFF, q & 511u, j);
    return db * (float)gv * iq_sign(iq_ksigns(g, q >> 9), j);
}
"#;

/// `dequant_element` for `IQ2_S` — transcribed from `vulkan_shaders`'s
/// `IQ2_S_COOP_MIDDLE`, which is the version measured on hardware.
///
/// `IQ2_XS`'s lattice with the index widened by two bits from a per-32
/// `qh` byte, and an explicit sign byte per 8 weights instead of the
/// packed 7-bit field.
const GGML_TYPE_IQ2_S_MIDDLE: &str = r#"
@CONST@ @U32@ BLOCK_BYTES = 82u;
@CONST@ @U32@ BLOCK_ELEMS = 256u;
@DEV@ float dequant_element(@GU8@w, @GU32C@g, @U32@ byte_offset, @U32@ k) {
    float d = orangu_half_to_float((@U16@)w[byte_offset] | ((@U16@)w[byte_offset + 1] << 8));
    @U32@ qs_off = byte_offset + 2u;
    @U32@ qh_off = byte_offset + 66u;
    @U32@ scales_off = byte_offset + 74u;
    @U32@ signs_off = qs_off + 32u;
    @U32@ ib32 = k / 32u;
    @U32@ l = (k % 32u) / 8u;
    @U32@ j = k % 8u;
    @U32@ qh = (@U32@)w[qh_off + ib32];
    @U32@ idx = (@U32@)w[qs_off + 4u * ib32 + l] | ((qh << (8u - 2u * l)) & 0x300u);
    @U32@ sc = ((@U32@)w[scales_off + ib32] >> (4u * (l / 2u))) & 0xFu;
    float db = d * (0.5f + (float)sc) * 0.25f;
    @U32@ gv = iq_grid8(g, IQ2S_GRID_OFF, idx, j);
    return db * (float)gv * iq_sign((@U32@)w[signs_off + 4u * ib32 + l], j);
}
"#;

/// `dequant_element` for `IQ3_XXS` — transcribed from `vulkan_shaders`'s
/// `IQ3_XXS_COOP_MIDDLE`, which is the version measured on hardware.
///
/// A 4-element lattice point per 4 weights out of `IQ3XXS_GRID`, with the
/// same packed 7-bit sign field and top-nibble scale as `IQ2_XXS`.
const GGML_TYPE_IQ3_XXS_MIDDLE: &str = r#"
@CONST@ @U32@ BLOCK_BYTES = 98u;
@CONST@ @U32@ BLOCK_ELEMS = 256u;
@DEV@ float dequant_element(@GU8@w, @GU32C@g, @U32@ byte_offset, @U32@ k) {
    float d = orangu_half_to_float((@U16@)w[byte_offset] | ((@U16@)w[byte_offset + 1] << 8));
    @U32@ qs_off = byte_offset + 2u;
    @U32@ aux_off = qs_off + 64u;
    @U32@ ib32 = k / 32u;
    @U32@ l = (k % 32u) / 8u;
    @U32@ j = k % 8u;
    @U32@ ao = aux_off + 4u * ib32;
    @U32@ aux32 = (@U32@)w[ao] | ((@U32@)w[ao + 1u] << 8)
        | ((@U32@)w[ao + 2u] << 16) | ((@U32@)w[ao + 3u] << 24);
    float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
    @U32@ signs = iq_ksigns(g, (aux32 >> (7u * l)) & 127u);
    @U32@ idx = (@U32@)w[qs_off + 8u * ib32 + 2u * l + (j >> 2)];
    @U32@ gv = iq_grid4(g, IQ3XXS_GRID_OFF, idx, j & 3u);
    return db * (float)gv * iq_sign(signs, j);
}
"#;

/// `dequant_element` for `IQ3_S` — transcribed from `vulkan_shaders`'s
/// `IQ3_S_COOP_MIDDLE`, which is the version measured on hardware.
///
/// `IQ3_XXS`'s lattice with the index widened by one bit from a per-32
/// `qh` byte, an explicit sign byte per 8 weights, and an odd-integer
/// scale (`1 + 2*sc`) rather than the `0.5 + sc` form.
const GGML_TYPE_IQ3_S_MIDDLE: &str = r#"
@CONST@ @U32@ BLOCK_BYTES = 110u;
@CONST@ @U32@ BLOCK_ELEMS = 256u;
@DEV@ float dequant_element(@GU8@w, @GU32C@g, @U32@ byte_offset, @U32@ k) {
    float d = orangu_half_to_float((@U16@)w[byte_offset] | ((@U16@)w[byte_offset + 1] << 8));
    @U32@ qs_off = byte_offset + 2u;
    @U32@ qh_off = byte_offset + 66u;
    @U32@ signs_off = byte_offset + 74u;
    @U32@ scales_off = byte_offset + 106u;
    @U32@ ib32 = k / 32u;
    @U32@ l = (k % 32u) / 8u;
    @U32@ j = k % 8u;
    @U32@ sc = ((@U32@)w[scales_off + ib32 / 2u] >> (4u * (ib32 % 2u))) & 0xFu;
    float db = d * (float)(1u + 2u * sc);
    @U32@ hb = (@U32@)w[qh_off + ib32];
    @U32@ half_j = j >> 2;
    @U32@ idx = (@U32@)w[qs_off + 8u * ib32 + 2u * l + half_j]
        | ((hb << (8u - 2u * l - half_j)) & 256u);
    @U32@ gv = iq_grid4(g, IQ3S_GRID_OFF, idx, j & 3u);
    return db * (float)gv * iq_sign((@U32@)w[signs_off + 4u * ib32 + l], j);
}
"#;

/// `dequant_element` for `IQ1_S` — transcribed from `vulkan_shaders`'s
/// `IQ1_S_COOP_MIDDLE`, which is the version measured on hardware.
///
/// A **signed** 8-element lattice point out of `IQ1S_GRID` — the `iq1*`
/// grids carry signed values and no sign field at all — plus a per-block
/// `delta` of `+/-0.125` selected by the top bit of `qh`.
const GGML_TYPE_IQ1_S_MIDDLE: &str = r#"
@CONST@ @U32@ BLOCK_BYTES = 50u;
@CONST@ @U32@ BLOCK_ELEMS = 256u;
@DEV@ float dequant_element(@GU8@w, @GU32C@g, @U32@ byte_offset, @U32@ k) {
    float d = orangu_half_to_float((@U16@)w[byte_offset] | ((@U16@)w[byte_offset + 1] << 8));
    @U32@ qs_off = byte_offset + 2u;
    @U32@ qh_off = byte_offset + 34u;
    @U32@ ib = k / 32u;
    @U32@ l = (k % 32u) / 8u;
    @U32@ j = k % 8u;
    @U32@ qh = (@U32@)w[qh_off + 2u * ib] | ((@U32@)w[qh_off + 2u * ib + 1u] << 8);
    float dl = d * (float)(2u * ((qh >> 12) & 7u) + 1u);
    float delta = ((qh & 0x8000u) != 0u) ? -IQ1_DELTA : IQ1_DELTA;
    @U32@ idx = (@U32@)w[qs_off + 4u * ib + l] | (((qh >> (3u * l)) & 7u) << 8);
    return dl * (iq_grid8_signed(g, IQ1S_GRID_OFF, idx, j) + delta);
}
"#;

/// `dequant_element` for `IQ1_M` — transcribed from `vulkan_shaders`'s
/// `IQ1_M_COOP_MIDDLE`, which is the version measured on hardware.
///
/// `IQ1_S`'s grid with no `f16` header at all: the block scale is
/// reassembled from one nibble of each of the four 16-bit scale words, and
/// each 16 weights get their own 3-bit sub-scale and sign bit.
const GGML_TYPE_IQ1_M_MIDDLE: &str = r#"
@CONST@ @U32@ BLOCK_BYTES = 56u;
@CONST@ @U32@ BLOCK_ELEMS = 256u;
@DEV@ @U32@ orangu_iq1m_scale_u16(@GU8@w, @U32@ scales_off, @U32@ i) {
    return (@U32@)w[scales_off + 2u * i] | ((@U32@)w[scales_off + 2u * i + 1u] << 8);
}
@DEV@ float dequant_element(@GU8@w, @GU32C@g, @U32@ byte_offset, @U32@ k) {
    @U32@ qs_off = byte_offset;
    @U32@ qh_off = byte_offset + 32u;
    @U32@ scales_off = byte_offset + 48u;
    @U32@ s0 = orangu_iq1m_scale_u16(w, scales_off, 0u);
    @U32@ s1 = orangu_iq1m_scale_u16(w, scales_off, 1u);
    @U32@ s2 = orangu_iq1m_scale_u16(w, scales_off, 2u);
    @U32@ s3 = orangu_iq1m_scale_u16(w, scales_off, 3u);
    @U32@ packed = (s0 >> 12) | ((s1 >> 8) & 0x00F0u) | ((s2 >> 4) & 0x0F00u) | (s3 & 0xF000u);
    float d = orangu_half_to_float((@U16@)packed);
    @U32@ ib = k / 32u;
    @U32@ l = (k % 32u) / 8u;
    @U32@ j = k % 8u;
    @U32@ s = s0;
    if (ib / 2u == 1u) {
        s = s1;
    } else if (ib / 2u == 2u) {
        s = s2;
    } else if (ib / 2u == 3u) {
        s = s3;
    }
    @U32@ shift = 6u * (ib % 2u);
    @U32@ sub = (l >= 2u) ? (shift + 3u) : shift;
    float dl = d * (float)(2u * ((s >> sub) & 7u) + 1u);
    @U32@ qhb = (l >= 2u) ? (@U32@)w[qh_off + 2u * ib + 1u] : (@U32@)w[qh_off + 2u * ib];
    @U32@ hshift = ((l % 2u) == 1u) ? 4u : 8u;
    @U32@ bit = ((l % 2u) == 1u) ? 0x80u : 0x08u;
    @U32@ idx = (@U32@)w[qs_off + 4u * ib + l] | ((qhb << hshift) & 0x700u);
    float delta = ((qhb & bit) != 0u) ? -IQ1_DELTA : IQ1_DELTA;
    return dl * (iq_grid8_signed(g, IQ1S_GRID_OFF, idx, j) + delta);
}
"#;

/// `block_dot` for `IQ2_XXS` — the per-lane invariants hoisted out of
/// `GGML_TYPE_IQ2_XXS_MIDDLE`.
///
/// A lane owns `[sub * 16, sub * 16 + 16)`, which is **two** 8-element
/// lattice points. `ib32` is uniform across the run and `l` takes exactly
/// two values, so the block header, the scale and the per-32 fields are
/// read once per lane instead of once per element, and the sign field is
/// expanded twice instead of sixteen times.
const GGML_TYPE_IQ2_XXS_BLOCK_MIDDLE: &str = r#"
@CONST@ @U32@ BLOCK_BYTES = 66u;
@CONST@ @U32@ BLOCK_ELEMS = 256u;
@CONST@ @U32@ LANES_PER_BLOCK = 16u;
@DEV@ float block_dot(@GU8@w, @GU32C@g, @GF32C@x, @U32@ byte_offset, @U32@ x_off, @U32@ sub) {
    float d = orangu_half_to_float((@U16@)w[byte_offset] | ((@U16@)w[byte_offset + 1] << 8));
    @U32@ qs_off = byte_offset + 2u;
    @U32@ ib32 = sub / 2u;
    @U32@ base = qs_off + 8u * ib32;
    @U32@ aux0 = (@U32@)w[base] | ((@U32@)w[base + 1u] << 8)
        | ((@U32@)w[base + 2u] << 16) | ((@U32@)w[base + 3u] << 24);
    @U32@ aux1 = (@U32@)w[base + 4u] | ((@U32@)w[base + 5u] << 8)
        | ((@U32@)w[base + 6u] << 16) | ((@U32@)w[base + 7u] << 24);
    float db = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;
    @U32@ x_lane = x_off + sub * 16u;
    float acc = 0.0f;
    for (@U32@ m = 0u; m < 2u; m += 1u) {
        @U32@ l = (sub % 2u) * 2u + m;
        @U32@ idx = (aux0 >> (8u * l)) & 0xFFu;
        @U32@ signs = iq_ksigns(g, (aux1 >> (7u * l)) & 127u);
        for (@U32@ j = 0u; j < 8u; j += 1u) {
            @U32@ gv = iq_grid8(g, IQ2XXS_GRID_OFF, idx, j);
            acc += db * (float)gv * iq_sign(signs, j) * x[x_lane + m * 8u + j];
        }
    }
    return acc;
}
"#;

/// `block_dot` for `IQ2_XS` — the per-lane invariants hoisted out of
/// `GGML_TYPE_IQ2_XS_MIDDLE`.
///
/// A lane owns `[sub * 16, sub * 16 + 16)`, which is **two** 8-element
/// lattice points. `ib32` is uniform across the run and `l` takes exactly
/// two values, so the block header, the scale and the per-32 fields are
/// read once per lane instead of once per element, and the sign field is
/// expanded twice instead of sixteen times.
const GGML_TYPE_IQ2_XS_BLOCK_MIDDLE: &str = r#"
@CONST@ @U32@ BLOCK_BYTES = 74u;
@CONST@ @U32@ BLOCK_ELEMS = 256u;
@CONST@ @U32@ LANES_PER_BLOCK = 16u;
@DEV@ float block_dot(@GU8@w, @GU32C@g, @GF32C@x, @U32@ byte_offset, @U32@ x_off, @U32@ sub) {
    float d = orangu_half_to_float((@U16@)w[byte_offset] | ((@U16@)w[byte_offset + 1] << 8));
    @U32@ qs_off = byte_offset + 2u;
    @U32@ scales_off = byte_offset + 66u;
    @U32@ ib32 = sub / 2u;
    @U32@ sc = ((@U32@)w[scales_off + ib32] >> (4u * (sub % 2u))) & 0xFu;
    float db = d * (0.5f + (float)sc) * 0.25f;
    @U32@ x_lane = x_off + sub * 16u;
    float acc = 0.0f;
    for (@U32@ m = 0u; m < 2u; m += 1u) {
        @U32@ l = (sub % 2u) * 2u + m;
        @U32@ qo = qs_off + 2u * (4u * ib32 + l);
        @U32@ q = (@U32@)w[qo] | ((@U32@)w[qo + 1u] << 8);
        @U32@ signs = iq_ksigns(g, q >> 9);
        for (@U32@ j = 0u; j < 8u; j += 1u) {
            @U32@ gv = iq_grid8(g, IQ2XS_GRID_OFF, q & 511u, j);
            acc += db * (float)gv * iq_sign(signs, j) * x[x_lane + m * 8u + j];
        }
    }
    return acc;
}
"#;

/// `block_dot` for `IQ2_S` — the per-lane invariants hoisted out of
/// `GGML_TYPE_IQ2_S_MIDDLE`.
///
/// A lane owns `[sub * 16, sub * 16 + 16)`, which is **two** 8-element
/// lattice points. `ib32` is uniform across the run and `l` takes exactly
/// two values, so the block header, the scale and the per-32 fields are
/// read once per lane instead of once per element, and the sign field is
/// expanded twice instead of sixteen times.
const GGML_TYPE_IQ2_S_BLOCK_MIDDLE: &str = r#"
@CONST@ @U32@ BLOCK_BYTES = 82u;
@CONST@ @U32@ BLOCK_ELEMS = 256u;
@CONST@ @U32@ LANES_PER_BLOCK = 16u;
@DEV@ float block_dot(@GU8@w, @GU32C@g, @GF32C@x, @U32@ byte_offset, @U32@ x_off, @U32@ sub) {
    float d = orangu_half_to_float((@U16@)w[byte_offset] | ((@U16@)w[byte_offset + 1] << 8));
    @U32@ qs_off = byte_offset + 2u;
    @U32@ qh_off = byte_offset + 66u;
    @U32@ scales_off = byte_offset + 74u;
    @U32@ signs_off = qs_off + 32u;
    @U32@ ib32 = sub / 2u;
    @U32@ qh = (@U32@)w[qh_off + ib32];
    @U32@ sc = ((@U32@)w[scales_off + ib32] >> (4u * (sub % 2u))) & 0xFu;
    float db = d * (0.5f + (float)sc) * 0.25f;
    @U32@ x_lane = x_off + sub * 16u;
    float acc = 0.0f;
    for (@U32@ m = 0u; m < 2u; m += 1u) {
        @U32@ l = (sub % 2u) * 2u + m;
        @U32@ idx = (@U32@)w[qs_off + 4u * ib32 + l] | ((qh << (8u - 2u * l)) & 0x300u);
        @U32@ signs = (@U32@)w[signs_off + 4u * ib32 + l];
        for (@U32@ j = 0u; j < 8u; j += 1u) {
            @U32@ gv = iq_grid8(g, IQ2S_GRID_OFF, idx, j);
            acc += db * (float)gv * iq_sign(signs, j) * x[x_lane + m * 8u + j];
        }
    }
    return acc;
}
"#;

/// `block_dot` for `IQ3_XXS` — the per-lane invariants hoisted out of
/// `GGML_TYPE_IQ3_XXS_MIDDLE`.
///
/// A lane owns `[sub * 16, sub * 16 + 16)`, which is **two** 8-element
/// lattice points. `ib32` is uniform across the run and `l` takes exactly
/// two values, so the block header, the scale and the per-32 fields are
/// read once per lane instead of once per element, and the sign field is
/// expanded twice instead of sixteen times.
const GGML_TYPE_IQ3_XXS_BLOCK_MIDDLE: &str = r#"
@CONST@ @U32@ BLOCK_BYTES = 98u;
@CONST@ @U32@ BLOCK_ELEMS = 256u;
@CONST@ @U32@ LANES_PER_BLOCK = 16u;
@DEV@ float block_dot(@GU8@w, @GU32C@g, @GF32C@x, @U32@ byte_offset, @U32@ x_off, @U32@ sub) {
    float d = orangu_half_to_float((@U16@)w[byte_offset] | ((@U16@)w[byte_offset + 1] << 8));
    @U32@ qs_off = byte_offset + 2u;
    @U32@ aux_off = qs_off + 64u;
    @U32@ ib32 = sub / 2u;
    @U32@ ao = aux_off + 4u * ib32;
    @U32@ aux32 = (@U32@)w[ao] | ((@U32@)w[ao + 1u] << 8)
        | ((@U32@)w[ao + 2u] << 16) | ((@U32@)w[ao + 3u] << 24);
    float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
    @U32@ x_lane = x_off + sub * 16u;
    float acc = 0.0f;
    for (@U32@ m = 0u; m < 2u; m += 1u) {
        @U32@ l = (sub % 2u) * 2u + m;
        @U32@ signs = iq_ksigns(g, (aux32 >> (7u * l)) & 127u);
        @U32@ qbase = qs_off + 8u * ib32 + 2u * l;
        for (@U32@ j = 0u; j < 8u; j += 1u) {
            @U32@ idx = (@U32@)w[qbase + (j >> 2)];
            @U32@ gv = iq_grid4(g, IQ3XXS_GRID_OFF, idx, j & 3u);
            acc += db * (float)gv * iq_sign(signs, j) * x[x_lane + m * 8u + j];
        }
    }
    return acc;
}
"#;

/// `block_dot` for `IQ3_S` — the per-lane invariants hoisted out of
/// `GGML_TYPE_IQ3_S_MIDDLE`.
///
/// A lane owns `[sub * 16, sub * 16 + 16)`, which is **two** 8-element
/// lattice points. `ib32` is uniform across the run and `l` takes exactly
/// two values, so the block header, the scale and the per-32 fields are
/// read once per lane instead of once per element, and the sign field is
/// expanded twice instead of sixteen times.
const GGML_TYPE_IQ3_S_BLOCK_MIDDLE: &str = r#"
@CONST@ @U32@ BLOCK_BYTES = 110u;
@CONST@ @U32@ BLOCK_ELEMS = 256u;
@CONST@ @U32@ LANES_PER_BLOCK = 16u;
@DEV@ float block_dot(@GU8@w, @GU32C@g, @GF32C@x, @U32@ byte_offset, @U32@ x_off, @U32@ sub) {
    float d = orangu_half_to_float((@U16@)w[byte_offset] | ((@U16@)w[byte_offset + 1] << 8));
    @U32@ qs_off = byte_offset + 2u;
    @U32@ qh_off = byte_offset + 66u;
    @U32@ signs_off = byte_offset + 74u;
    @U32@ scales_off = byte_offset + 106u;
    @U32@ ib32 = sub / 2u;
    @U32@ sc = ((@U32@)w[scales_off + ib32 / 2u] >> (4u * (ib32 % 2u))) & 0xFu;
    float db = d * (float)(1u + 2u * sc);
    @U32@ hb = (@U32@)w[qh_off + ib32];
    @U32@ x_lane = x_off + sub * 16u;
    float acc = 0.0f;
    for (@U32@ m = 0u; m < 2u; m += 1u) {
        @U32@ l = (sub % 2u) * 2u + m;
        @U32@ signs = (@U32@)w[signs_off + 4u * ib32 + l];
        @U32@ qbase = qs_off + 8u * ib32 + 2u * l;
        for (@U32@ j = 0u; j < 8u; j += 1u) {
            @U32@ half_j = j >> 2;
            @U32@ idx = (@U32@)w[qbase + half_j] | ((hb << (8u - 2u * l - half_j)) & 256u);
            @U32@ gv = iq_grid4(g, IQ3S_GRID_OFF, idx, j & 3u);
            acc += db * (float)gv * iq_sign(signs, j) * x[x_lane + m * 8u + j];
        }
    }
    return acc;
}
"#;

/// `block_dot` for `IQ1_S` — the per-lane invariants hoisted out of
/// `GGML_TYPE_IQ1_S_MIDDLE`.
///
/// A lane owns `[sub * 16, sub * 16 + 16)`, which is **two** 8-element
/// lattice points. `ib32` is uniform across the run and `l` takes exactly
/// two values, so the block header, the scale and the per-32 fields are
/// read once per lane instead of once per element, and the sign field is
/// expanded twice instead of sixteen times.
const GGML_TYPE_IQ1_S_BLOCK_MIDDLE: &str = r#"
@CONST@ @U32@ BLOCK_BYTES = 50u;
@CONST@ @U32@ BLOCK_ELEMS = 256u;
@CONST@ @U32@ LANES_PER_BLOCK = 16u;
@DEV@ float block_dot(@GU8@w, @GU32C@g, @GF32C@x, @U32@ byte_offset, @U32@ x_off, @U32@ sub) {
    float d = orangu_half_to_float((@U16@)w[byte_offset] | ((@U16@)w[byte_offset + 1] << 8));
    @U32@ qs_off = byte_offset + 2u;
    @U32@ qh_off = byte_offset + 34u;
    @U32@ ib = sub / 2u;
    @U32@ qh = (@U32@)w[qh_off + 2u * ib] | ((@U32@)w[qh_off + 2u * ib + 1u] << 8);
    float dl = d * (float)(2u * ((qh >> 12) & 7u) + 1u);
    float delta = ((qh & 0x8000u) != 0u) ? -IQ1_DELTA : IQ1_DELTA;
    @U32@ x_lane = x_off + sub * 16u;
    float acc = 0.0f;
    for (@U32@ m = 0u; m < 2u; m += 1u) {
        @U32@ l = (sub % 2u) * 2u + m;
        @U32@ idx = (@U32@)w[qs_off + 4u * ib + l] | (((qh >> (3u * l)) & 7u) << 8);
        for (@U32@ j = 0u; j < 8u; j += 1u) {
            acc += dl * (iq_grid8_signed(g, IQ1S_GRID_OFF, idx, j) + delta)
                * x[x_lane + m * 8u + j];
        }
    }
    return acc;
}
"#;

/// `block_dot` for `IQ1_M` — the per-lane invariants hoisted out of
/// `GGML_TYPE_IQ1_M_MIDDLE`.
///
/// A lane owns `[sub * 16, sub * 16 + 16)`, which is **two** 8-element
/// lattice points. `ib32` is uniform across the run and `l` takes exactly
/// two values, so the block header, the scale and the per-32 fields are
/// read once per lane instead of once per element, and the sign field is
/// expanded twice instead of sixteen times.
///
/// `l >= 2` is uniform across a lane here — it is `sub % 2` — so the
/// sub-scale shift and which `qh` byte to read hoist with the rest, and
/// only the `+/-` delta and the lattice index vary between the lane's two
/// points.
const GGML_TYPE_IQ1_M_BLOCK_MIDDLE: &str = r#"
@CONST@ @U32@ BLOCK_BYTES = 56u;
@CONST@ @U32@ BLOCK_ELEMS = 256u;
@CONST@ @U32@ LANES_PER_BLOCK = 16u;
@DEV@ @U32@ orangu_iq1m_scale_u16_bd(@GU8@w, @U32@ scales_off, @U32@ i) {
    return (@U32@)w[scales_off + 2u * i] | ((@U32@)w[scales_off + 2u * i + 1u] << 8);
}
@DEV@ float block_dot(@GU8@w, @GU32C@g, @GF32C@x, @U32@ byte_offset, @U32@ x_off, @U32@ sub) {
    @U32@ qs_off = byte_offset;
    @U32@ qh_off = byte_offset + 32u;
    @U32@ scales_off = byte_offset + 48u;
    @U32@ s0 = orangu_iq1m_scale_u16_bd(w, scales_off, 0u);
    @U32@ s1 = orangu_iq1m_scale_u16_bd(w, scales_off, 1u);
    @U32@ s2 = orangu_iq1m_scale_u16_bd(w, scales_off, 2u);
    @U32@ s3 = orangu_iq1m_scale_u16_bd(w, scales_off, 3u);
    @U32@ packed = (s0 >> 12) | ((s1 >> 8) & 0x00F0u) | ((s2 >> 4) & 0x0F00u) | (s3 & 0xF000u);
    float d = orangu_half_to_float((@U16@)packed);
    @U32@ ib = sub / 2u;
    @U32@ s = s0;
    if (ib / 2u == 1u) {
        s = s1;
    } else if (ib / 2u == 2u) {
        s = s2;
    } else if (ib / 2u == 3u) {
        s = s3;
    }
    @U32@ hi = sub % 2u;
    @U32@ shift = 6u * (ib % 2u) + hi * 3u;
    float dl = d * (float)(2u * ((s >> shift) & 7u) + 1u);
    @U32@ qhb = (@U32@)w[qh_off + 2u * ib + hi];
    @U32@ x_lane = x_off + sub * 16u;
    float acc = 0.0f;
    for (@U32@ m = 0u; m < 2u; m += 1u) {
        @U32@ l = hi * 2u + m;
        @U32@ hshift = ((l % 2u) == 1u) ? 4u : 8u;
        @U32@ bit = ((l % 2u) == 1u) ? 0x80u : 0x08u;
        @U32@ idx = (@U32@)w[qs_off + 4u * ib + l] | ((qhb << hshift) & 0x700u);
        float delta = ((qhb & bit) != 0u) ? -IQ1_DELTA : IQ1_DELTA;
        for (@U32@ j = 0u; j < 8u; j += 1u) {
            acc += dl * (iq_grid8_signed(g, IQ1S_GRID_OFF, idx, j) + delta)
                * x[x_lane + m * 8u + j];
        }
    }
    return acc;
}
"#;

/// `block_dot` for `IQ4_XS` — the per-lane invariants hoisted out of
/// `GGML_TYPE_IQ4_XS_MIDDLE`.
///
/// No lattice here, so no two-point split: `ib` and the nibble half are
/// both uniform across a lane's sixteen elements, leaving a flat loop over
/// sixteen bytes with the scale already unpacked.
const GGML_TYPE_IQ4_XS_BLOCK_MIDDLE: &str = r#"
@CONST@ @U32@ BLOCK_BYTES = 136u;
@CONST@ @U32@ BLOCK_ELEMS = 256u;
@CONST@ @U32@ LANES_PER_BLOCK = 16u;
@DEV@ float block_dot(@GU8@w, @GU32C@g, @GF32C@x, @U32@ byte_offset, @U32@ x_off, @U32@ sub) {
    float d = orangu_half_to_float((@U16@)w[byte_offset] | ((@U16@)w[byte_offset + 1] << 8));
    @U32@ scales_h = (@U32@)w[byte_offset + 2] | ((@U32@)w[byte_offset + 3] << 8);
    @U32@ scales_l_off = byte_offset + 4u;
    @U32@ qs_off = byte_offset + 8u;
    @U32@ ib = sub / 2u;
    @U32@ hi = sub % 2u;
    @U32@ low = ((@U32@)w[scales_l_off + ib / 2u] >> (4u * (ib % 2u))) & 0xFu;
    @U32@ high = (scales_h >> (2u * ib)) & 3u;
    float dl = d * (float)((int)(low | (high << 4)) - 32);
    @U32@ qbase = qs_off + 16u * ib;
    @U32@ x_lane = x_off + sub * 16u;
    float acc = 0.0f;
    for (@U32@ i = 0u; i < 16u; i += 1u) {
        @U32@ byte = (@U32@)w[qbase + i];
        @U32@ nib = (hi == 0u) ? (byte & 0xFu) : (byte >> 4);
        acc += dl * orangu_iq4_kvalue(nib) * x[x_lane + i];
    }
    return acc;
}
"#;

/// `dequant_element` for `F32` - a transcription of
/// `engine::quant`'s dequantizer for the same type.
const GGML_TYPE_F32_MIDDLE: &str = r#"
@CONST@ @U32@ BLOCK_BYTES = 4u;
@CONST@ @U32@ BLOCK_ELEMS = 1u;
@DEV@ float dequant_element(@GU8@w, @GU32C@g, @U32@ byte_offset, @U32@ k) {
    @U32@ bits = (@U32@)w[byte_offset] | ((@U32@)w[byte_offset + 1] << 8)
        | ((@U32@)w[byte_offset + 2] << 16) | ((@U32@)w[byte_offset + 3] << 24);
    return @ASF_OPEN@bits);
}
"#;

/// `dequant_element` for `F16` - a transcription of
/// `engine::quant`'s dequantizer for the same type.
const GGML_TYPE_F16_MIDDLE: &str = r#"
@CONST@ @U32@ BLOCK_BYTES = 2u;
@CONST@ @U32@ BLOCK_ELEMS = 1u;
@DEV@ float dequant_element(@GU8@w, @GU32C@g, @U32@ byte_offset, @U32@ k) {
    @U16@ bits = (@U16@)w[byte_offset] | ((@U16@)w[byte_offset + 1] << 8);
    return orangu_half_to_float(bits);
}
"#;

/// `dequant_element` for `BF16` - a transcription of
/// `engine::quant`'s dequantizer for the same type.
const GGML_TYPE_BF16_MIDDLE: &str = r#"
@CONST@ @U32@ BLOCK_BYTES = 2u;
@CONST@ @U32@ BLOCK_ELEMS = 1u;
@DEV@ float dequant_element(@GU8@w, @GU32C@g, @U32@ byte_offset, @U32@ k) {
    @U16@ bits = (@U16@)w[byte_offset] | ((@U16@)w[byte_offset + 1] << 8);
    return orangu_bf16_to_float(bits);
}
"#;

/// `dequant_element` for `Q4_0` - a transcription of
/// `engine::quant`'s dequantizer for the same type.
const GGML_TYPE_Q4_0_MIDDLE: &str = r#"
@CONST@ @U32@ BLOCK_BYTES = 18u;
@CONST@ @U32@ BLOCK_ELEMS = 32u;
@DEV@ float dequant_element(@GU8@w, @GU32C@g, @U32@ byte_offset, @U32@ k) {
    float d = orangu_half_to_float((@U16@)w[byte_offset] | ((@U16@)w[byte_offset + 1] << 8));
    if (k < 16u) {
        @U8@ byte = w[byte_offset + 2u + k];
        return ((float)((int)(byte & 0xFu) - 8)) * d;
    }
    @U8@ byte = w[byte_offset + 2u + (k - 16u)];
    return ((float)((int)(byte >> 4) - 8)) * d;
}
"#;

/// `dequant_element` for `Q4_1` — a transcription of
/// `engine::quant::dequantize_q4_1`.
///
/// `Q4_0`'s nibble split with a stored minimum in place of the fixed `-8`
/// bias: `block_q4_1` is `{ d: f16, m: f16, qs: [u8; 16] }`, and a weight is
/// `q * d + m` rather than `(q - 8) * d`. Nothing here needs a codebook or an
/// extra binding, which is why its absence from these backends was an
/// omission rather than a decision — see `PARITY.md` C3.
const GGML_TYPE_Q4_1_MIDDLE: &str = r#"
@CONST@ @U32@ BLOCK_BYTES = 20u;
@CONST@ @U32@ BLOCK_ELEMS = 32u;
@DEV@ float dequant_element(@GU8@w, @GU32C@g, @U32@ byte_offset, @U32@ k) {
    float d = orangu_half_to_float((@U16@)w[byte_offset] | ((@U16@)w[byte_offset + 1] << 8));
    float m = orangu_half_to_float((@U16@)w[byte_offset + 2] | ((@U16@)w[byte_offset + 3] << 8));
    if (k < 16u) {
        @U8@ byte = w[byte_offset + 4u + k];
        return ((float)(byte & 0xFu)) * d + m;
    }
    @U8@ byte = w[byte_offset + 4u + (k - 16u)];
    return ((float)(byte >> 4)) * d + m;
}
"#;

/// `dequant_element` for `Q5_1` — a transcription of
/// `engine::quant::dequantize_q5_1`.
///
/// `Q5_0`'s fifth bit, packed across a 32-bit `qh`, with `Q4_1`'s stored
/// minimum in place of the fixed `-16` bias: `block_q5_1` is
/// `{ d: f16, m: f16, qh: [u8; 4], qs: [u8; 16] }`. The `qh` bit for the
/// low nibble of `j` is bit `j`; for the high nibble it is bit `j + 16`,
/// which the shift by `j + 12` followed by the `0x10` mask selects — the
/// same arithmetic `Q5_0` uses, and the same the CPU path does.
const GGML_TYPE_Q5_1_MIDDLE: &str = r#"
@CONST@ @U32@ BLOCK_BYTES = 24u;
@CONST@ @U32@ BLOCK_ELEMS = 32u;
@DEV@ float dequant_element(@GU8@w, @GU32C@g, @U32@ byte_offset, @U32@ k) {
    float d = orangu_half_to_float((@U16@)w[byte_offset] | ((@U16@)w[byte_offset + 1] << 8));
    float m = orangu_half_to_float((@U16@)w[byte_offset + 2] | ((@U16@)w[byte_offset + 3] << 8));
    @U32@ qh = (@U32@)w[byte_offset + 4] | ((@U32@)w[byte_offset + 5] << 8)
        | ((@U32@)w[byte_offset + 6] << 16) | ((@U32@)w[byte_offset + 7] << 24);
    if (k < 16u) {
        @U8@ byte = w[byte_offset + 8u + k];
        @U32@ xh0 = ((qh >> k) << 4) & 0x10u;
        return ((float)((byte & 0xFu) | xh0)) * d + m;
    }
    @U32@ j = k - 16u;
    @U8@ byte = w[byte_offset + 8u + j];
    @U32@ xh1 = (qh >> (j + 12u)) & 0x10u;
    return ((float)((byte >> 4) | xh1)) * d + m;
}
"#;

/// `dequant_element` for `Q5_0` - a transcription of
/// `engine::quant`'s dequantizer for the same type.
const GGML_TYPE_Q5_0_MIDDLE: &str = r#"
@CONST@ @U32@ BLOCK_BYTES = 22u;
@CONST@ @U32@ BLOCK_ELEMS = 32u;
@DEV@ float dequant_element(@GU8@w, @GU32C@g, @U32@ byte_offset, @U32@ k) {
    float d = orangu_half_to_float((@U16@)w[byte_offset] | ((@U16@)w[byte_offset + 1] << 8));
    @U32@ qh = (@U32@)w[byte_offset + 2] | ((@U32@)w[byte_offset + 3] << 8)
        | ((@U32@)w[byte_offset + 4] << 16) | ((@U32@)w[byte_offset + 5] << 24);
    if (k < 16u) {
        @U8@ byte = w[byte_offset + 6u + k];
        @U32@ xh0 = ((qh >> k) << 4) & 0x10u;
        return ((float)((int)((byte & 0xFu) | xh0) - 16)) * d;
    }
    @U32@ j = k - 16u;
    @U8@ byte = w[byte_offset + 6u + j];
    @U32@ xh1 = (qh >> (j + 12u)) & 0x10u;
    return ((float)((int)((byte >> 4) | xh1) - 16)) * d;
}
"#;

/// `dequant_element` for `Q8_0` - a transcription of
/// `engine::quant`'s dequantizer for the same type.
const GGML_TYPE_Q8_0_MIDDLE: &str = r#"
@CONST@ @U32@ BLOCK_BYTES = 34u;
@CONST@ @U32@ BLOCK_ELEMS = 32u;
@DEV@ float dequant_element(@GU8@w, @GU32C@g, @U32@ byte_offset, @U32@ k) {
    float d = orangu_half_to_float((@U16@)w[byte_offset] | ((@U16@)w[byte_offset + 1] << 8));
    @I8@ q = (@I8@)w[byte_offset + 2u + k];
    return ((float)q) * d;
}
"#;

/// `dequant_element` for `Q4_K` - a transcription of
/// `engine::quant`'s dequantizer for the same type.
const GGML_TYPE_Q4_K_MIDDLE: &str = r#"
@CONST@ @U32@ BLOCK_BYTES = 144u;
@CONST@ @U32@ BLOCK_ELEMS = 256u;
@DEV@ float dequant_element(@GU8@w, @GU32C@g, @U32@ byte_offset, @U32@ k) {
    float d = orangu_half_to_float((@U16@)w[byte_offset] | ((@U16@)w[byte_offset + 1] << 8));
    float dmin = orangu_half_to_float((@U16@)w[byte_offset + 2] | ((@U16@)w[byte_offset + 3] << 8));
    @U32@ scales_off = byte_offset + 4u;
    @U32@ qs_off = byte_offset + 16u;
    @U32@ q_offset = (k / 64u) * 64u;
    @U32@ local_in_group = k % 64u;
    @U32@ is_base = (q_offset / 64u) * 2u;
    @U32@ q_base = qs_off + q_offset / 2u;
    @U32@ sc, m;
    if (local_in_group < 32u) {
        @U8@ byte = w[q_base + local_in_group];
        orangu_get_scale_min_k4(w, scales_off, is_base, &sc, &m);
        float d1 = d * (float)sc;
        float m1 = dmin * (float)m;
        return d1 * (float)(byte & 0xFu) - m1;
    }
    @U32@ l = local_in_group - 32u;
    @U8@ byte = w[q_base + l];
    orangu_get_scale_min_k4(w, scales_off, is_base + 1u, &sc, &m);
    float d2 = d * (float)sc;
    float m2 = dmin * (float)m;
    return d2 * (float)(byte >> 4) - m2;
}
"#;

/// `dequant_element` for `Q5_K` - a transcription of
/// `engine::quant`'s dequantizer for the same type.
const GGML_TYPE_Q5_K_MIDDLE: &str = r#"
@CONST@ @U32@ BLOCK_BYTES = 176u;
@CONST@ @U32@ BLOCK_ELEMS = 256u;
@DEV@ float dequant_element(@GU8@w, @GU32C@g, @U32@ byte_offset, @U32@ k) {
    float d = orangu_half_to_float((@U16@)w[byte_offset] | ((@U16@)w[byte_offset + 1] << 8));
    float dmin = orangu_half_to_float((@U16@)w[byte_offset + 2] | ((@U16@)w[byte_offset + 3] << 8));
    @U32@ scales_off = byte_offset + 4u;
    @U32@ qh_off = byte_offset + 16u;
    @U32@ qs_off = byte_offset + 48u;
    @U32@ q_offset = (k / 64u) * 64u;
    @U32@ idx = q_offset / 64u;
    @U32@ local_in_group = k % 64u;
    @U32@ is_base = idx * 2u;
    @U32@ ql_offset = idx * 32u;
    @U32@ u1 = 1u << (2u * idx);
    @U32@ u2 = 2u << (2u * idx);
    @U32@ sc, m;
    if (local_in_group < 32u) {
        @U32@ l = local_in_group;
        @U8@ byte = w[qs_off + ql_offset + l];
        @U8@ qhbyte = w[qh_off + l];
        int hi_bit = (qhbyte & u1) != 0u ? 16 : 0;
        orangu_get_scale_min_k4(w, scales_off, is_base, &sc, &m);
        float d1 = d * (float)sc;
        float m1 = dmin * (float)m;
        return d1 * (float)((int)(byte & 0xFu) + hi_bit) - m1;
    }
    @U32@ l = local_in_group - 32u;
    @U8@ byte = w[qs_off + ql_offset + l];
    @U8@ qhbyte = w[qh_off + l];
    int hi_bit = (qhbyte & u2) != 0u ? 16 : 0;
    orangu_get_scale_min_k4(w, scales_off, is_base + 1u, &sc, &m);
    float d2 = d * (float)sc;
    float m2 = dmin * (float)m;
    return d2 * (float)((int)(byte >> 4) + hi_bit) - m2;
}
"#;

/// `dequant_element` for `Q6_K` - a transcription of
/// `engine::quant`'s dequantizer for the same type.
const GGML_TYPE_Q6_K_MIDDLE: &str = r#"
@CONST@ @U32@ BLOCK_BYTES = 210u;
@CONST@ @U32@ BLOCK_ELEMS = 256u;
@DEV@ float dequant_element(@GU8@w, @GU32C@g, @U32@ byte_offset, @U32@ k) {
    @U32@ ql_off = byte_offset;
    @U32@ qh_off = byte_offset + 128u;
    @U32@ sc_off = byte_offset + 192u;
    float d = orangu_half_to_float((@U16@)w[byte_offset + 208] | ((@U16@)w[byte_offset + 209] << 8));
    @U32@ y_off = (k / 128u) * 128u;
    @U32@ idx = y_off / 128u;
    @U32@ local_in_group = k % 128u;
    @U32@ which_q = local_in_group / 32u;
    @U32@ l = local_in_group % 32u;
    @U32@ ql_o = idx * 64u;
    @U32@ qh_o = idx * 32u;
    @U32@ sc_o = idx * 8u;
    @U32@ is = l / 16u;
    @U8@ ql_l = w[ql_off + ql_o + l];
    @U8@ ql_l32 = w[ql_off + ql_o + l + 32u];
    @U8@ qh_l = w[qh_off + qh_o + l];
    int q;
    @U32@ sc_idx;
    if (which_q == 0u) {
        q = (int)((ql_l & 0xFu) | ((qh_l & 3u) << 4)) - 32;
        sc_idx = is;
    } else if (which_q == 1u) {
        q = (int)((ql_l32 & 0xFu) | (((qh_l >> 2) & 3u) << 4)) - 32;
        sc_idx = is + 2u;
    } else if (which_q == 2u) {
        q = (int)((ql_l >> 4) | (((qh_l >> 4) & 3u) << 4)) - 32;
        sc_idx = is + 4u;
    } else {
        q = (int)((ql_l32 >> 4) | (((qh_l >> 6) & 3u) << 4)) - 32;
        sc_idx = is + 6u;
    }
    @I8@ sc = (@I8@)w[sc_off + sc_o + sc_idx];
    return d * (float)sc * (float)q;
}
"#;

/// `dequant_element` for `Q2_K` - a transcription of
/// `engine::quant`'s dequantizer for the same type.
const GGML_TYPE_Q2_K_MIDDLE: &str = r#"
@CONST@ @U32@ BLOCK_BYTES = 84u;
@CONST@ @U32@ BLOCK_ELEMS = 256u;
@DEV@ float dequant_element(@GU8@w, @GU32C@g, @U32@ byte_offset, @U32@ k) {
    @U32@ scales_off = byte_offset;
    @U32@ qs_off = byte_offset + 16u;
    float d = orangu_half_to_float((@U16@)w[byte_offset + 80] | ((@U16@)w[byte_offset + 81] << 8));
    float dmin = orangu_half_to_float((@U16@)w[byte_offset + 82] | ((@U16@)w[byte_offset + 83] << 8));
    @U32@ n = k / 128u;
    @U32@ r = k % 128u;
    @U32@ s = r / 32u;
    @U32@ h = (r % 32u) / 16u;
    @U32@ l = r % 16u;
    @U8@ sc = w[scales_off + n * 8u + s * 2u + h];
    float dl = d * (float)(sc & 0xFu);
    float ml = dmin * (float)(sc >> 4);
    @U8@ byte = w[qs_off + n * 32u + h * 16u + l];
    return dl * (float)((byte >> (2u * s)) & 3u) - ml;
}
"#;

/// `dequant_element` for `Q3_K` - a transcription of
/// `engine::quant`'s dequantizer for the same type.
const GGML_TYPE_Q3_K_MIDDLE: &str = r#"
@CONST@ @U32@ BLOCK_BYTES = 110u;
@CONST@ @U32@ BLOCK_ELEMS = 256u;
// Q3_K's `i`th 6-bit sub-block scale (0..16), still biased by 32, out of the
// 12 bytes at `base`. Mirrors `quant::unpack_q3_k_scales` for one index.
@DEV@ @U32@ orangu_q3k_scale(@GU8@w, @U32@ base, @U32@ i) {
    @U32@ low;
    if (i < 8u) {
        low = w[base + i] & 0xFu;
    } else {
        low = w[base + i - 8u] >> 4;
    }
    @U32@ high = (w[base + 8u + (i % 4u)] >> (2u * (i / 4u))) & 3u;
    return low | (high << 4);
}
@DEV@ float dequant_element(@GU8@w, @GU32C@g, @U32@ byte_offset, @U32@ k) {
    @U32@ hmask_off = byte_offset;
    @U32@ qs_off = byte_offset + 32u;
    @U32@ scales_off = byte_offset + 96u;
    float d_all = orangu_half_to_float((@U16@)w[byte_offset + 108] | ((@U16@)w[byte_offset + 109] << 8));
    @U32@ n = k / 128u;
    @U32@ r = k % 128u;
    @U32@ s = r / 32u;
    @U32@ h = (r % 32u) / 16u;
    @U32@ l = r % 16u;
    @U32@ idx = h * 16u + l;
    @U32@ m = 1u << (n * 4u + s);
    float dl = d_all * (float)((int)orangu_q3k_scale(w, scales_off, n * 8u + s * 2u + h) - 32);
    int hi = 4;
    if ((w[hmask_off + idx] & m) != 0u) {
        hi = 0;
    }
    @U32@ q = (w[qs_off + n * 32u + idx] >> (2u * s)) & 3u;
    return dl * (float)((int)q - hi);
}
"#;

/// `dequant_element` for `IQ4_NL` - a transcription of
/// `engine::quant`'s dequantizer for the same type.
const GGML_TYPE_IQ4_NL_MIDDLE: &str = r#"
@CONST@ @U32@ BLOCK_BYTES = 18u;
@CONST@ @U32@ BLOCK_ELEMS = 32u;
@DEV@ float dequant_element(@GU8@w, @GU32C@g, @U32@ byte_offset, @U32@ k) {
    float d = orangu_half_to_float((@U16@)w[byte_offset] | ((@U16@)w[byte_offset + 1] << 8));
    @U8@ byte = w[byte_offset + 2u + (k % 16u)];
    @U32@ nib = (k < 16u) ? (@U32@)(byte & 0xFu) : (@U32@)(byte >> 4);
    return d * orangu_iq4_kvalue(nib);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// Every dialect, so a test that only ever ran the CUDA table would not
    /// be the one that caught an OpenCL typo.
    const DIALECTS: [Dialect; 3] = [Dialect::Cuda, Dialect::Hip, Dialect::OpenCl];

    /// The gap this module closes: `SUPPORTED_TYPES` and the middles were
    /// separate lists in three files, so a type could be advertised by one
    /// and implemented by none. `Backend::supports_type` answers from the
    /// first, and `matmul` panics on a miss in the second.
    #[test]
    fn every_supported_type_renders_in_every_dialect() {
        for dialect in DIALECTS {
            for &ggml_type in SUPPORTED_TYPES {
                let source = kernel_source(dialect, ggml_type).unwrap_or_else(|| {
                    panic!("{dialect:?} has no kernel for advertised type {ggml_type}")
                });
                // Exactly one of the two algorithms, never both and never
                // neither: a kernel with no dequantizer at all computes
                // nothing, and one with both has had a middle spliced in
                // beside a `block_dot` that shadows it.
                let element_wise = source.contains("float dequant_element(");
                let block_hoisted = source.contains("float block_dot(");
                assert!(
                    element_wise != block_hoisted,
                    "{dialect:?} type {ggml_type}: element-wise {element_wise}, \
                     block-hoisted {block_hoisted} — expected exactly one"
                );
                // And the entry point has to call whichever one it got.
                let called = if block_hoisted {
                    "block_dot(weights, iq_grids, x,"
                } else {
                    "dequant_element(weights, iq_grids,"
                };
                assert!(
                    source.contains(called),
                    "{dialect:?} type {ggml_type} never calls {called}"
                );
                assert!(
                    source.contains(KERNEL_NAME),
                    "{dialect:?} type {ggml_type} rendered no `{KERNEL_NAME}` entry point"
                );
            }
        }
    }

    /// A placeholder that reaches a vendor compiler is a syntax error in a
    /// language this project cannot compile here, surfacing as a startup
    /// failure on somebody else's machine. Cheaper to catch it as text.
    ///
    /// The failure it is really for: adding a quantization whose middle uses
    /// a token one dialect's table does not define. `@` is not legal in any
    /// of these dialects, so its presence in rendered output is unambiguous.
    /// Confirmed to fail, naming the dialect and the offending lines, when
    /// `@U16@` is dropped from the OpenCL table.
    #[test]
    fn no_placeholder_survives_rendering() {
        for dialect in DIALECTS {
            for &ggml_type in SUPPORTED_TYPES {
                let source = kernel_source(dialect, ggml_type).expect("supported type");
                assert!(
                    !source.contains('@'),
                    "{dialect:?} type {ggml_type} kept a placeholder:\n{}",
                    source
                        .lines()
                        .filter(|l| l.contains('@'))
                        .collect::<Vec<_>>()
                        .join("\n")
                );
            }
        }
    }

    /// HIP-C is CUDA-C for everything these kernels use, and the two
    /// backends' sources were byte-identical before this module existed.
    /// They are two `Dialect` variants so a future divergence has somewhere
    /// to go; until one arrives, this is what stops them drifting by
    /// accident.
    #[test]
    fn cuda_and_hip_render_identically() {
        for &ggml_type in SUPPORTED_TYPES {
            assert_eq!(
                kernel_source(Dialect::Cuda, ggml_type),
                kernel_source(Dialect::Hip, ggml_type),
                "type {ggml_type} rendered differently for CUDA and HIP"
            );
        }
    }

    /// The dialects must differ *somewhere*, or the token table is not being
    /// applied at all and every backend is compiling CUDA-C. A test suite
    /// where every dialect agrees would pass just as happily with `render`
    /// returning its input.
    #[test]
    fn opencl_is_actually_rendered_differently() {
        let ty = SUPPORTED_TYPES[0];
        let cuda = kernel_source(Dialect::Cuda, ty).expect("supported type");
        let opencl = kernel_source(Dialect::OpenCl, ty).expect("supported type");
        assert_ne!(cuda, opencl);
        assert!(opencl.contains("__kernel void"), "no OpenCL entry point");
        assert!(opencl.contains("get_local_id(0)"), "no OpenCL lane id");
        assert!(
            !opencl.contains("threadIdx"),
            "CUDA lane id leaked into OpenCL"
        );
        assert!(cuda.contains("__global__"), "no CUDA entry point");
        assert!(
            !cuda.contains("get_local_id"),
            "OpenCL lane id leaked into CUDA"
        );
    }

    /// `local` is a reserved word in OpenCL C. The CUDA source names a
    /// variable that, and a rendering that let the name through would fail
    /// to compile on every OpenCL device — which is exactly the class of
    /// failure nobody here can reproduce.
    #[test]
    fn opencl_does_not_declare_a_variable_named_local() {
        let source = kernel_source(Dialect::OpenCl, SUPPORTED_TYPES[0]).expect("supported type");
        assert!(
            !source.contains("uint local ") && !source.contains("uint local="),
            "`local` is a reserved word in OpenCL C"
        );
        assert!(
            source.contains("local_id"),
            "the renamed lane id is missing"
        );
    }

    /// The arithmetic of the two quantizations this pass added, checked
    /// against `engine::quant`'s own dequantizer for the same types.
    ///
    /// **What this does and does not prove.** The kernels are C for three
    /// devices none of which exist on this machine, so nothing here compiles
    /// or runs them. What it runs is a Rust mirror of
    /// `GGML_TYPE_Q4_1_MIDDLE`/`GGML_TYPE_Q5_1_MIDDLE`, written
    /// statement-for-statement from the C below it, against the CPU
    /// dequantizer that is this project's ground truth for both types. A
    /// misread of the block layout — a wrong offset, a wrong mask, the `qh`
    /// bit for the high nibble taken from the wrong place — fails here.
    ///
    /// What escapes it is a transcription slip *between* the mirror and the
    /// C string, since one author wrote both. That residual is what the
    /// `matmul_matches_cpu_backend_for_q4_1`/`_q5_1` cross-checks in each
    /// backend's own test module are for, and they need hardware this
    /// project does not have. Stated rather than glossed: this is the
    /// strongest check available here, not a substitute for that one.
    #[test]
    fn the_added_quantizations_match_the_cpu_dequantizer() {
        use crate::engine::quant::{GGML_TYPE_IQ4_XS, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1, dequantize};

        // `dequant_element`, transcribed from the C. Kept deliberately
        // literal — same names, same order, same casts — so the two can be
        // read side by side.
        fn half(w: &[u8], off: usize) -> f32 {
            f32::from(half::f16::from_le_bytes([w[off], w[off + 1]]))
        }
        fn q4_1(w: &[u8], byte_offset: usize, k: u32) -> f32 {
            let d = half(w, byte_offset);
            let m = half(w, byte_offset + 2);
            if k < 16 {
                let byte = w[byte_offset + 4 + k as usize];
                return (byte & 0xF) as f32 * d + m;
            }
            let byte = w[byte_offset + 4 + (k - 16) as usize];
            (byte >> 4) as f32 * d + m
        }
        fn iq4_xs(w: &[u8], byte_offset: usize, k: u32) -> f32 {
            const KVALUES: [i8; 16] = [
                -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
            ];
            let d = half(w, byte_offset);
            let scales_h = u32::from(w[byte_offset + 2]) | (u32::from(w[byte_offset + 3]) << 8);
            let scales_l_off = byte_offset + 4;
            let qs_off = byte_offset + 8;
            let (ib, r) = (k / 32, k % 32);
            let low = (u32::from(w[scales_l_off + (ib / 2) as usize]) >> (4 * (ib % 2))) & 0xF;
            let high = (scales_h >> (2 * ib)) & 3;
            let dl = d * ((low | (high << 4)) as i32 - 32) as f32;
            let byte = u32::from(w[qs_off + (16 * ib + (r % 16)) as usize]);
            let nib = if r < 16 { byte & 0xF } else { byte >> 4 };
            dl * f32::from(KVALUES[nib as usize])
        }
        fn q5_1(w: &[u8], byte_offset: usize, k: u32) -> f32 {
            let d = half(w, byte_offset);
            let m = half(w, byte_offset + 2);
            let qh = u32::from(w[byte_offset + 4])
                | (u32::from(w[byte_offset + 5]) << 8)
                | (u32::from(w[byte_offset + 6]) << 16)
                | (u32::from(w[byte_offset + 7]) << 24);
            if k < 16 {
                let byte = w[byte_offset + 8 + k as usize];
                let xh0 = ((qh >> k) << 4) & 0x10;
                return (u32::from(byte & 0xF) | xh0) as f32 * d + m;
            }
            let j = k - 16;
            let byte = w[byte_offset + 8 + j as usize];
            let xh1 = (qh >> (j + 12)) & 0x10;
            (u32::from(byte >> 4) | xh1) as f32 * d + m
        }

        // Deterministic pseudo-random blocks — the same xorshift the
        // backends' own cross-check tests use, so a failure is reproducible.
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut byte = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed & 0xFF) as u8
        };
        for (ggml_type, block_bytes, elem, kernel) in [
            (
                GGML_TYPE_Q4_1,
                20usize,
                32usize,
                &q4_1 as &dyn Fn(&[u8], usize, u32) -> f32,
            ),
            (GGML_TYPE_Q5_1, 24, 32, &q5_1),
            (GGML_TYPE_IQ4_XS, 136, 256, &iq4_xs),
        ] {
            const BLOCKS: usize = 8;
            let mut bytes: Vec<u8> = (0..block_bytes * BLOCKS).map(|_| byte()).collect();
            // Both types carry two `f16` block headers, `d` and `m`, and four
            // random bytes are perfectly capable of spelling a NaN or an
            // infinity — at which point every element of that block is NaN
            // and the comparison below tests nothing but `NaN != NaN`. The
            // payload stays random, because the payload is where a wrong
            // mask or a wrong `qh` bit lives; only the scales are pinned.
            for b in 0..BLOCKS {
                let d = half::f16::from_f32(0.05 + 0.01 * b as f32);
                bytes[b * block_bytes..b * block_bytes + 2].copy_from_slice(&d.to_le_bytes());
                // `IQ4_XS` keeps its packed scale-high field in bytes 2..4,
                // not a second `f16`; pinning those would flatten every
                // sub-block scale in the block to the same value and stop the
                // test exercising the unpacking at all.
                if ggml_type != GGML_TYPE_IQ4_XS {
                    let m = half::f16::from_f32(-0.4 + 0.1 * b as f32);
                    bytes[b * block_bytes + 2..b * block_bytes + 4]
                        .copy_from_slice(&m.to_le_bytes());
                }
            }
            let expected = dequantize(ggml_type, &bytes, elem * BLOCKS).expect("cpu dequantizer");
            for b in 0..BLOCKS {
                for k in 0..elem {
                    let got = kernel(&bytes, b * block_bytes, k as u32);
                    let want = expected[b * elem + k];
                    assert!(
                        (got - want).abs() <= want.abs() * 1e-6 + 1e-6,
                        "type {ggml_type} block {b} element {k}: kernel {got}, cpu {want}"
                    );
                }
            }
        }
    }

    /// Every `block_dot` against `engine::quant`'s dequantizer for the same
    /// type, summed the way the kernel sums it.
    ///
    /// Same standing as `the_added_quantizations_match_the_cpu_dequantizer`,
    /// and the same stated limit: this runs a Rust mirror of the C, not the
    /// C. It catches a wrong offset, a wrong mask, a wrong `qh` bit or a
    /// wrong lane-to-element mapping — the `sub`-indexed layout is new here
    /// and is exactly the kind of thing that is easy to get subtly wrong —
    /// and it cannot catch a slip between the mirror and the kernel text.
    ///
    /// The check is a **whole block's dot product**, accumulated over all
    /// `LANES_PER_BLOCK` lanes, against the same dot product computed from
    /// the dequantized weights. That is the property the kernel actually
    /// needs: not that any one lane is right, but that the lanes together
    /// cover every element of the block exactly once. A mapping that
    /// double-counted one element and dropped another would pass a
    /// per-element check and fail this one.
    #[test]
    fn every_block_dot_covers_its_block_exactly_once() {
        use crate::engine::quant::{
            GGML_TYPE_IQ4_NL, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
            GGML_TYPE_Q4_K, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
            GGML_TYPE_Q8_0, block_layout, dequantize,
        };

        const KVALUES: [i8; 16] = [
            -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
        ];
        fn half(w: &[u8], off: usize) -> f32 {
            f32::from(half::f16::from_le_bytes([w[off], w[off + 1]]))
        }
        // Each of these is `block_dot` transcribed statement for statement.
        // `sub` is the lane's index within its block.
        fn q4_0(w: &[u8], x: &[f32], bo: usize, xo: usize, sub: u32) -> f32 {
            let d = half(w, bo);
            let mut acc = 0.0;
            for m in 0..2u32 {
                let j = (sub * 2 + m) as usize;
                let byte = u32::from(w[bo + 2 + j]);
                acc += ((byte & 0xF) as i32 - 8) as f32 * d * x[xo + j];
                acc += ((byte >> 4) as i32 - 8) as f32 * d * x[xo + 16 + j];
            }
            acc
        }
        fn q4_1(w: &[u8], x: &[f32], bo: usize, xo: usize, sub: u32) -> f32 {
            let (d, mn) = (half(w, bo), half(w, bo + 2));
            let mut acc = 0.0;
            for m in 0..2u32 {
                let j = (sub * 2 + m) as usize;
                let byte = u32::from(w[bo + 4 + j]);
                acc += ((byte & 0xF) as f32 * d + mn) * x[xo + j];
                acc += ((byte >> 4) as f32 * d + mn) * x[xo + 16 + j];
            }
            acc
        }
        fn q5_0(w: &[u8], x: &[f32], bo: usize, xo: usize, sub: u32) -> f32 {
            let d = half(w, bo);
            let qh = u32::from_le_bytes([w[bo + 2], w[bo + 3], w[bo + 4], w[bo + 5]]);
            let mut acc = 0.0;
            for m in 0..2u32 {
                let j = sub * 2 + m;
                let byte = u32::from(w[bo + 6 + j as usize]);
                let xh0 = ((qh >> j) << 4) & 0x10;
                let xh1 = (qh >> (j + 12)) & 0x10;
                acc += (((byte & 0xF) | xh0) as i32 - 16) as f32 * d * x[xo + j as usize];
                acc += (((byte >> 4) | xh1) as i32 - 16) as f32 * d * x[xo + 16 + j as usize];
            }
            acc
        }
        fn q5_1(w: &[u8], x: &[f32], bo: usize, xo: usize, sub: u32) -> f32 {
            let (d, mn) = (half(w, bo), half(w, bo + 2));
            let qh = u32::from_le_bytes([w[bo + 4], w[bo + 5], w[bo + 6], w[bo + 7]]);
            let mut acc = 0.0;
            for m in 0..2u32 {
                let j = sub * 2 + m;
                let byte = u32::from(w[bo + 8 + j as usize]);
                let xh0 = ((qh >> j) << 4) & 0x10;
                let xh1 = (qh >> (j + 12)) & 0x10;
                acc += (((byte & 0xF) | xh0) as f32 * d + mn) * x[xo + j as usize];
                acc += (((byte >> 4) | xh1) as f32 * d + mn) * x[xo + 16 + j as usize];
            }
            acc
        }
        fn q8_0(w: &[u8], x: &[f32], bo: usize, xo: usize, sub: u32) -> f32 {
            let d = half(w, bo);
            let mut acc = 0.0;
            for m in 0..4u32 {
                let j = (sub * 4 + m) as usize;
                let mut v = i32::from(w[bo + 2 + j]);
                if v >= 128 {
                    v -= 256;
                }
                acc += v as f32 * d * x[xo + j];
            }
            acc
        }
        fn iq4_nl(w: &[u8], x: &[f32], bo: usize, xo: usize, sub: u32) -> f32 {
            let d = half(w, bo);
            let mut acc = 0.0;
            for m in 0..2u32 {
                let j = (sub * 2 + m) as usize;
                let byte = u32::from(w[bo + 2 + j]);
                acc += d * f32::from(KVALUES[(byte & 0xF) as usize]) * x[xo + j];
                acc += d * f32::from(KVALUES[(byte >> 4) as usize]) * x[xo + 16 + j];
            }
            acc
        }

        fn q2_k(w: &[u8], x: &[f32], bo: usize, xo: usize, sub: u32) -> f32 {
            let (scales_off, qs_off) = (bo, bo + 16);
            let d = half(w, bo + 80);
            let dmin = half(w, bo + 82);
            let (n, s, h) = (sub / 8, (sub % 8) / 2, sub % 2);
            let sc = u32::from(w[scales_off + (n * 8 + s * 2 + h) as usize]);
            let dl = d * (sc & 0xF) as f32;
            let ml = dmin * (sc >> 4) as f32;
            let base = qs_off + (n * 32 + h * 16) as usize;
            let x_lane = xo + (sub * 16) as usize;
            let mut acc = 0.0;
            for l in 0..16usize {
                let byte = u32::from(w[base + l]);
                acc += (dl * ((byte >> (2 * s)) & 3) as f32 - ml) * x[x_lane + l];
            }
            acc
        }
        fn q3k_scale(w: &[u8], base: usize, i: u32) -> u32 {
            let low = if i < 8 {
                u32::from(w[base + i as usize]) & 0xF
            } else {
                u32::from(w[base + (i - 8) as usize]) >> 4
            };
            let high = (u32::from(w[base + 8 + (i % 4) as usize]) >> (2 * (i / 4))) & 3;
            low | (high << 4)
        }
        fn q3_k(w: &[u8], x: &[f32], bo: usize, xo: usize, sub: u32) -> f32 {
            let (hmask_off, qs_off, scales_off) = (bo, bo + 32, bo + 96);
            let d_all = half(w, bo + 108);
            let (n, s, h) = (sub / 8, (sub % 8) / 2, sub % 2);
            let m = 1u32 << (n * 4 + s);
            let dl = d_all * (q3k_scale(w, scales_off, n * 8 + s * 2 + h) as i32 - 32) as f32;
            let x_lane = xo + (sub * 16) as usize;
            let mut acc = 0.0;
            for l in 0..16u32 {
                let idx = (h * 16 + l) as usize;
                let hi = if u32::from(w[hmask_off + idx]) & m != 0 {
                    0
                } else {
                    4
                };
                let q = (u32::from(w[qs_off + (n * 32) as usize + idx]) >> (2 * s)) & 3;
                acc += dl * (q as i32 - hi) as f32 * x[x_lane + l as usize];
            }
            acc
        }
        fn q4_k(w: &[u8], x: &[f32], bo: usize, xo: usize, sub: u32) -> f32 {
            let (d, dmin) = (half(w, bo), half(w, bo + 2));
            let (scales_off, qs_off) = (bo + 4, bo + 16);
            let n = sub / 4;
            let half_hi = (sub % 4) / 2;
            let lane_off = (sub % 2) * 16;
            let q_base = qs_off + (n * 32 + lane_off) as usize;
            let (sc, mn) = crate::engine::quant::get_scale_min_k4(
                (n * 2 + half_hi) as usize,
                &w[scales_off..],
            );
            let dl = d * f32::from(sc);
            let ml = dmin * f32::from(mn);
            let x_lane = xo + (sub * 16) as usize;
            let mut acc = 0.0;
            for l in 0..16usize {
                let byte = u32::from(w[q_base + l]);
                let q = if half_hi == 0 { byte & 0xF } else { byte >> 4 };
                acc += (dl * q as f32 - ml) * x[x_lane + l];
            }
            acc
        }
        fn q5_k(w: &[u8], x: &[f32], bo: usize, xo: usize, sub: u32) -> f32 {
            let (d, dmin) = (half(w, bo), half(w, bo + 2));
            let (scales_off, qh_off, qs_off) = (bo + 4, bo + 16, bo + 48);
            let n = sub / 4;
            let half_hi = (sub % 4) / 2;
            let lane_off = (sub % 2) * 16;
            let q_base = qs_off + (n * 32 + lane_off) as usize;
            let qh_base = qh_off + lane_off as usize;
            let hmask = if half_hi == 0 {
                1u32 << (2 * n)
            } else {
                2u32 << (2 * n)
            };
            let (sc, mn) = crate::engine::quant::get_scale_min_k4(
                (n * 2 + half_hi) as usize,
                &w[scales_off..],
            );
            let dl = d * f32::from(sc);
            let ml = dmin * f32::from(mn);
            let x_lane = xo + (sub * 16) as usize;
            let mut acc = 0.0;
            for l in 0..16usize {
                let byte = u32::from(w[q_base + l]);
                let q = if half_hi == 0 { byte & 0xF } else { byte >> 4 };
                let hi_bit = if u32::from(w[qh_base + l]) & hmask != 0 {
                    16
                } else {
                    0
                };
                acc += (dl * (q as i32 + hi_bit) as f32 - ml) * x[x_lane + l];
            }
            acc
        }
        fn q6_k(w: &[u8], x: &[f32], bo: usize, xo: usize, sub: u32) -> f32 {
            let (ql_off, qh_off, sc_off) = (bo, bo + 128, bo + 192);
            let d = half(w, bo + 208);
            let idx = sub / 8;
            let which_q = (sub % 8) / 2;
            let lane_off = (sub % 2) * 16;
            let ql_base = ql_off + (idx * 64 + lane_off) as usize;
            let qh_base = qh_off + (idx * 32 + lane_off) as usize;
            let sc = w[sc_off + (idx * 8 + (sub % 2) + which_q * 2) as usize] as i8;
            let dl = d * f32::from(sc);
            let x_lane = xo + (sub * 16) as usize;
            let mut acc = 0.0;
            for l in 0..16usize {
                let ql_l = u32::from(w[ql_base + l]);
                let ql_l32 = u32::from(w[ql_base + l + 32]);
                let qh_l = u32::from(w[qh_base + l]);
                let q = match which_q {
                    0 => ((ql_l & 0xF) | ((qh_l & 3) << 4)) as i32 - 32,
                    1 => ((ql_l32 & 0xF) | (((qh_l >> 2) & 3) << 4)) as i32 - 32,
                    2 => ((ql_l >> 4) | (((qh_l >> 4) & 3) << 4)) as i32 - 32,
                    _ => ((ql_l32 >> 4) | (((qh_l >> 6) & 3) << 4)) as i32 - 32,
                };
                acc += dl * q as f32 * x[x_lane + l];
            }
            acc
        }

        type BlockDot = fn(&[u8], &[f32], usize, usize, u32) -> f32;
        // `LANES_PER_BLOCK` per type: 8 for the 32-element legacy family,
        // 16 for the 256-element K-quants.
        let cases: [(u32, BlockDot, u32); 11] = [
            (GGML_TYPE_Q4_0, q4_0, 8),
            (GGML_TYPE_Q4_1, q4_1, 8),
            (GGML_TYPE_Q5_0, q5_0, 8),
            (GGML_TYPE_Q5_1, q5_1, 8),
            (GGML_TYPE_Q8_0, q8_0, 8),
            (GGML_TYPE_IQ4_NL, iq4_nl, 8),
            (GGML_TYPE_Q2_K, q2_k, 16),
            (GGML_TYPE_Q3_K, q3_k, 16),
            (GGML_TYPE_Q4_K, q4_k, 16),
            (GGML_TYPE_Q5_K, q5_k, 16),
            (GGML_TYPE_Q6_K, q6_k, 16),
        ];

        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut byte = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed & 0xFF) as u8
        };
        for (ggml_type, block_dot, lanes_per_block) in cases {
            let (block_bytes, elems) = block_layout(ggml_type).expect("a covered type");
            const BLOCKS: usize = 6;
            let mut w: Vec<u8> = (0..block_bytes * BLOCKS).map(|_| byte()).collect();
            // Finite scales, random payload — see
            // `the_added_quantizations_match_the_cpu_dequantizer` for why.
            // Where each type keeps `d`, and `dmin` when it has one. Random
            // bytes can spell an f16 NaN, and one NaN scale makes a whole
            // block NaN and the comparison vacuous; the payload stays random,
            // because that is where a wrong mask or lane mapping lives.
            let (d_at, dmin_at) = match ggml_type {
                t if t == GGML_TYPE_Q4_1 || t == GGML_TYPE_Q5_1 => (0, Some(2)),
                t if t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K => (0, Some(2)),
                t if t == GGML_TYPE_Q2_K => (80, Some(82)),
                t if t == GGML_TYPE_Q3_K => (108, None),
                t if t == GGML_TYPE_Q6_K => (208, None),
                _ => (0, None),
            };
            for b in 0..BLOCKS {
                let d = half::f16::from_f32(0.05 + 0.01 * b as f32);
                bytes_at(&mut w, b * block_bytes + d_at, d);
                if let Some(off) = dmin_at {
                    let m = half::f16::from_f32(0.4 - 0.1 * b as f32);
                    bytes_at(&mut w, b * block_bytes + off, m);
                }
            }
            let dequantized = dequantize(ggml_type, &w, elems * BLOCKS).expect("cpu dequantizer");
            let x: Vec<f32> = (0..elems * BLOCKS)
                .map(|i| ((i % 17) as f32 - 8.0) * 0.125)
                .collect();

            for b in 0..BLOCKS {
                // Every lane of the block, summed — what the kernel's
                // `partial` accumulators add up to across the workgroup.
                let got: f32 = (0..lanes_per_block)
                    .map(|sub| block_dot(&w, &x, b * block_bytes, b * elems, sub))
                    .sum();
                let want: f32 = (0..elems)
                    .map(|k| dequantized[b * elems + k] * x[b * elems + k])
                    .sum();
                assert!(
                    (got - want).abs() <= want.abs() * 1e-4 + 1e-4,
                    "type {ggml_type} block {b}: block_dot sum {got}, cpu {want}"
                );
            }
        }
    }

    /// Writes an `f16` into `w` at `off`, little-endian.
    fn bytes_at(w: &mut [u8], off: usize, v: half::f16) {
        w[off..off + 2].copy_from_slice(&v.to_le_bytes());
    }

    /// Every lattice `IQ*` dequantizer against `engine::quant`'s own, with
    /// the codebook read out of the very buffer the backends upload.
    ///
    /// These seven are the most intricate kernels in the module — a lattice
    /// index assembled from two or three fields, a sign pattern that is
    /// sometimes a packed 7-bit code through `ksigns_iq2xs` and sometimes an
    /// explicit byte, and a scale that is sometimes `0.5 + sc`, sometimes
    /// `1 + 2*sc`, and for `IQ1_M` four nibbles gathered out of four
    /// different 16-bit words. Transcribing them blind without checking the
    /// arithmetic would be indefensible.
    ///
    /// Same standing and same limit as the other two mirror tests: this runs
    /// a Rust transcription of the C, not the C. It reads the codebook
    /// through `packed::words()` at the offsets `iq_grid_prelude` formats in,
    /// so an offset that drifted from the packing fails here too.
    #[test]
    fn every_iq_dequantizer_matches_the_cpu_dequantizer() {
        use crate::engine::iq_grids::packed;
        use crate::engine::quant::{
            GGML_TYPE_IQ1_M, GGML_TYPE_IQ1_S, GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_XXS,
            GGML_TYPE_IQ3_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_XS, block_layout, dequantize,
        };

        let g = packed::words();
        // The codebook accessors, transcribed from `iq_grid_prelude`.
        let iq_grid8 = |base: u32, idx: u32, j: u32| -> u32 {
            let word = g[(base + idx * 2 + (j >> 2)) as usize];
            (word >> ((j & 3) * 8)) & 0xFF
        };
        let iq_grid4 = |base: u32, idx: u32, j: u32| -> u32 {
            (g[(base + idx) as usize] >> ((j & 3) * 8)) & 0xFF
        };
        let iq_ksigns = |i: u32| -> u32 {
            (g[(packed::KSIGNS_OFF + (i >> 2)) as usize] >> ((i & 3) * 8)) & 0xFF
        };
        let iq_grid8_signed = |base: u32, idx: u32, j: u32| -> f32 {
            let b = iq_grid8(base, idx, j) as i32;
            (if b >= 128 { b - 256 } else { b }) as f32
        };
        let iq_sign =
            |signs: u32, j: u32| -> f32 { if signs & (1 << j) != 0 { -1.0 } else { 1.0 } };
        let half = |w: &[u8], off: usize| -> f32 {
            f32::from(half::f16::from_le_bytes([w[off], w[off + 1]]))
        };
        let u16le =
            |w: &[u8], off: usize| -> u32 { u32::from(w[off]) | (u32::from(w[off + 1]) << 8) };

        let iq2_xxs = |w: &[u8], bo: usize, k: u32| -> f32 {
            let d = half(w, bo);
            let qs_off = bo + 2;
            let (ib32, l, j) = (k / 32, (k % 32) / 8, k % 8);
            let base = qs_off + (8 * ib32) as usize;
            let aux0 = u32::from_le_bytes([w[base], w[base + 1], w[base + 2], w[base + 3]]);
            let aux1 = u32::from_le_bytes([w[base + 4], w[base + 5], w[base + 6], w[base + 7]]);
            let db = d * (0.5 + (aux1 >> 28) as f32) * 0.25;
            let idx = (aux0 >> (8 * l)) & 0xFF;
            let signs = iq_ksigns((aux1 >> (7 * l)) & 127);
            db * iq_grid8(packed::IQ2XXS_GRID_OFF, idx, j) as f32 * iq_sign(signs, j)
        };
        let iq2_xs = |w: &[u8], bo: usize, k: u32| -> f32 {
            let d = half(w, bo);
            let (qs_off, scales_off) = (bo + 2, bo + 66);
            let (ib32, l, j) = (k / 32, (k % 32) / 8, k % 8);
            let qo = qs_off + (2 * (4 * ib32 + l)) as usize;
            let q = u16le(w, qo);
            let sc = (u32::from(w[scales_off + ib32 as usize]) >> (4 * (l / 2))) & 0xF;
            let db = d * (0.5 + sc as f32) * 0.25;
            db * iq_grid8(packed::IQ2XS_GRID_OFF, q & 511, j) as f32 * iq_sign(iq_ksigns(q >> 9), j)
        };
        let iq2_s = |w: &[u8], bo: usize, k: u32| -> f32 {
            let d = half(w, bo);
            let (qs_off, qh_off, scales_off) = (bo + 2, bo + 66, bo + 74);
            let signs_off = qs_off + 32;
            let (ib32, l, j) = (k / 32, (k % 32) / 8, k % 8);
            let qh = u32::from(w[qh_off + ib32 as usize]);
            let idx =
                u32::from(w[qs_off + (4 * ib32 + l) as usize]) | ((qh << (8 - 2 * l)) & 0x300);
            let sc = (u32::from(w[scales_off + ib32 as usize]) >> (4 * (l / 2))) & 0xF;
            let db = d * (0.5 + sc as f32) * 0.25;
            db * iq_grid8(packed::IQ2S_GRID_OFF, idx, j) as f32
                * iq_sign(u32::from(w[signs_off + (4 * ib32 + l) as usize]), j)
        };
        let iq3_xxs = |w: &[u8], bo: usize, k: u32| -> f32 {
            let d = half(w, bo);
            let qs_off = bo + 2;
            let aux_off = qs_off + 64;
            let (ib32, l, j) = (k / 32, (k % 32) / 8, k % 8);
            let ao = aux_off + (4 * ib32) as usize;
            let aux32 = u32::from_le_bytes([w[ao], w[ao + 1], w[ao + 2], w[ao + 3]]);
            let db = d * (0.5 + (aux32 >> 28) as f32) * 0.5;
            let signs = iq_ksigns((aux32 >> (7 * l)) & 127);
            let idx = u32::from(w[qs_off + (8 * ib32 + 2 * l + (j >> 2)) as usize]);
            db * iq_grid4(packed::IQ3XXS_GRID_OFF, idx, j & 3) as f32 * iq_sign(signs, j)
        };
        let iq3_s = |w: &[u8], bo: usize, k: u32| -> f32 {
            let d = half(w, bo);
            let (qs_off, qh_off, signs_off, scales_off) = (bo + 2, bo + 66, bo + 74, bo + 106);
            let (ib32, l, j) = (k / 32, (k % 32) / 8, k % 8);
            let sc = (u32::from(w[scales_off + (ib32 / 2) as usize]) >> (4 * (ib32 % 2))) & 0xF;
            let db = d * (1 + 2 * sc) as f32;
            let hb = u32::from(w[qh_off + ib32 as usize]);
            let half_j = j >> 2;
            let idx = u32::from(w[qs_off + (8 * ib32 + 2 * l + half_j) as usize])
                | ((hb << (8 - 2 * l - half_j)) & 256);
            db * iq_grid4(packed::IQ3S_GRID_OFF, idx, j & 3) as f32
                * iq_sign(u32::from(w[signs_off + (4 * ib32 + l) as usize]), j)
        };
        let iq1_s = |w: &[u8], bo: usize, k: u32| -> f32 {
            let d = half(w, bo);
            let (qs_off, qh_off) = (bo + 2, bo + 34);
            let (ib, l, j) = (k / 32, (k % 32) / 8, k % 8);
            let qh = u16le(w, qh_off + (2 * ib) as usize);
            let dl = d * (2 * ((qh >> 12) & 7) + 1) as f32;
            let delta = if qh & 0x8000 != 0 {
                -packed::IQ1_DELTA
            } else {
                packed::IQ1_DELTA
            };
            let idx = u32::from(w[qs_off + (4 * ib + l) as usize]) | (((qh >> (3 * l)) & 7) << 8);
            dl * (iq_grid8_signed(packed::IQ1S_GRID_OFF, idx, j) + delta)
        };
        let iq1_m = |w: &[u8], bo: usize, k: u32| -> f32 {
            let (qs_off, qh_off, scales_off) = (bo, bo + 32, bo + 48);
            let s: [u32; 4] = std::array::from_fn(|i| u16le(w, scales_off + 2 * i));
            let packed_d =
                (s[0] >> 12) | ((s[1] >> 8) & 0x00F0) | ((s[2] >> 4) & 0x0F00) | (s[3] & 0xF000);
            let d = f32::from(half::f16::from_bits(packed_d as u16));
            let (ib, l, j) = (k / 32, (k % 32) / 8, k % 8);
            let sw = s[(ib / 2) as usize];
            let shift = 6 * (ib % 2);
            let sub = if l >= 2 { shift + 3 } else { shift };
            let dl = d * (2 * ((sw >> sub) & 7) + 1) as f32;
            let qhb = if l >= 2 {
                u32::from(w[qh_off + (2 * ib + 1) as usize])
            } else {
                u32::from(w[qh_off + (2 * ib) as usize])
            };
            let (hshift, bit) = if l % 2 == 1 { (4, 0x80) } else { (8, 0x08) };
            let idx = u32::from(w[qs_off + (4 * ib + l) as usize]) | ((qhb << hshift) & 0x700);
            let delta = if qhb & bit != 0 {
                -packed::IQ1_DELTA
            } else {
                packed::IQ1_DELTA
            };
            dl * (iq_grid8_signed(packed::IQ1S_GRID_OFF, idx, j) + delta)
        };

        let iq4_xs_elem = |w: &[u8], bo: usize, k: u32| -> f32 {
            const KVALUES: [i8; 16] = [
                -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
            ];
            let d = half(w, bo);
            let scales_h = u16le(w, bo + 2);
            let (scales_l_off, qs_off) = (bo + 4, bo + 8);
            let (ib, r) = (k / 32, k % 32);
            let low = (u32::from(w[scales_l_off + (ib / 2) as usize]) >> (4 * (ib % 2))) & 0xF;
            let high = (scales_h >> (2 * ib)) & 3;
            let dl = d * ((low | (high << 4)) as i32 - 32) as f32;
            let byte = u32::from(w[qs_off + (16 * ib + (r % 16)) as usize]);
            let nib = if r < 16 { byte & 0xF } else { byte >> 4 };
            dl * f32::from(KVALUES[nib as usize])
        };

        type Deq<'a> = &'a dyn Fn(&[u8], usize, u32) -> f32;
        let cases: [(u32, Deq<'_>); 8] = [
            (GGML_TYPE_IQ2_XXS, &iq2_xxs),
            (GGML_TYPE_IQ2_XS, &iq2_xs),
            (GGML_TYPE_IQ2_S, &iq2_s),
            (GGML_TYPE_IQ3_XXS, &iq3_xxs),
            (GGML_TYPE_IQ3_S, &iq3_s),
            (GGML_TYPE_IQ1_S, &iq1_s),
            (GGML_TYPE_IQ1_M, &iq1_m),
            (GGML_TYPE_IQ4_XS, &iq4_xs_elem),
        ];

        let mut seed = 0xD1B5_4A32_D192_ED03u64;
        let mut byte = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed & 0xFF) as u8
        };
        // The hoisted `block_dot` form, transcribed the same way. A lane owns
        // `[sub*16, sub*16+16)` = two lattice points; `ib32` is uniform
        // across it and `l` takes exactly two values, which is what makes
        // the hoist legal. Checked as a whole-block sum over all sixteen
        // lanes, so a mapping that covered one element twice and another not
        // at all fails here where a per-element check would not.
        let bd_iq2_xxs = |w: &[u8], x: &[f32], bo: usize, xo: usize, sub: u32| -> f32 {
            let d = half(w, bo);
            let qs_off = bo + 2;
            let ib32 = sub / 2;
            let base = qs_off + (8 * ib32) as usize;
            let aux0 = u32::from_le_bytes([w[base], w[base + 1], w[base + 2], w[base + 3]]);
            let aux1 = u32::from_le_bytes([w[base + 4], w[base + 5], w[base + 6], w[base + 7]]);
            let db = d * (0.5 + (aux1 >> 28) as f32) * 0.25;
            let x_lane = xo + (sub * 16) as usize;
            let mut acc = 0.0;
            for m in 0..2u32 {
                let l = (sub % 2) * 2 + m;
                let idx = (aux0 >> (8 * l)) & 0xFF;
                let signs = iq_ksigns((aux1 >> (7 * l)) & 127);
                for j in 0..8u32 {
                    acc += db
                        * iq_grid8(packed::IQ2XXS_GRID_OFF, idx, j) as f32
                        * iq_sign(signs, j)
                        * x[x_lane + (m * 8 + j) as usize];
                }
            }
            acc
        };
        let bd_iq2_xs = |w: &[u8], x: &[f32], bo: usize, xo: usize, sub: u32| -> f32 {
            let d = half(w, bo);
            let (qs_off, scales_off) = (bo + 2, bo + 66);
            let ib32 = sub / 2;
            let sc = (u32::from(w[scales_off + ib32 as usize]) >> (4 * (sub % 2))) & 0xF;
            let db = d * (0.5 + sc as f32) * 0.25;
            let x_lane = xo + (sub * 16) as usize;
            let mut acc = 0.0;
            for m in 0..2u32 {
                let l = (sub % 2) * 2 + m;
                let qo = qs_off + (2 * (4 * ib32 + l)) as usize;
                let q = u16le(w, qo);
                let signs = iq_ksigns(q >> 9);
                for j in 0..8u32 {
                    acc += db
                        * iq_grid8(packed::IQ2XS_GRID_OFF, q & 511, j) as f32
                        * iq_sign(signs, j)
                        * x[x_lane + (m * 8 + j) as usize];
                }
            }
            acc
        };
        let bd_iq2_s = |w: &[u8], x: &[f32], bo: usize, xo: usize, sub: u32| -> f32 {
            let d = half(w, bo);
            let (qs_off, qh_off, scales_off) = (bo + 2, bo + 66, bo + 74);
            let signs_off = qs_off + 32;
            let ib32 = sub / 2;
            let qh = u32::from(w[qh_off + ib32 as usize]);
            let sc = (u32::from(w[scales_off + ib32 as usize]) >> (4 * (sub % 2))) & 0xF;
            let db = d * (0.5 + sc as f32) * 0.25;
            let x_lane = xo + (sub * 16) as usize;
            let mut acc = 0.0;
            for m in 0..2u32 {
                let l = (sub % 2) * 2 + m;
                let idx =
                    u32::from(w[qs_off + (4 * ib32 + l) as usize]) | ((qh << (8 - 2 * l)) & 0x300);
                let signs = u32::from(w[signs_off + (4 * ib32 + l) as usize]);
                for j in 0..8u32 {
                    acc += db
                        * iq_grid8(packed::IQ2S_GRID_OFF, idx, j) as f32
                        * iq_sign(signs, j)
                        * x[x_lane + (m * 8 + j) as usize];
                }
            }
            acc
        };
        let bd_iq3_xxs = |w: &[u8], x: &[f32], bo: usize, xo: usize, sub: u32| -> f32 {
            let d = half(w, bo);
            let qs_off = bo + 2;
            let aux_off = qs_off + 64;
            let ib32 = sub / 2;
            let ao = aux_off + (4 * ib32) as usize;
            let aux32 = u32::from_le_bytes([w[ao], w[ao + 1], w[ao + 2], w[ao + 3]]);
            let db = d * (0.5 + (aux32 >> 28) as f32) * 0.5;
            let x_lane = xo + (sub * 16) as usize;
            let mut acc = 0.0;
            for m in 0..2u32 {
                let l = (sub % 2) * 2 + m;
                let signs = iq_ksigns((aux32 >> (7 * l)) & 127);
                let qbase = qs_off + (8 * ib32 + 2 * l) as usize;
                for j in 0..8u32 {
                    let idx = u32::from(w[qbase + (j >> 2) as usize]);
                    acc += db
                        * iq_grid4(packed::IQ3XXS_GRID_OFF, idx, j & 3) as f32
                        * iq_sign(signs, j)
                        * x[x_lane + (m * 8 + j) as usize];
                }
            }
            acc
        };
        let bd_iq3_s = |w: &[u8], x: &[f32], bo: usize, xo: usize, sub: u32| -> f32 {
            let d = half(w, bo);
            let (qs_off, qh_off, signs_off, scales_off) = (bo + 2, bo + 66, bo + 74, bo + 106);
            let ib32 = sub / 2;
            let sc = (u32::from(w[scales_off + (ib32 / 2) as usize]) >> (4 * (ib32 % 2))) & 0xF;
            let db = d * (1 + 2 * sc) as f32;
            let hb = u32::from(w[qh_off + ib32 as usize]);
            let x_lane = xo + (sub * 16) as usize;
            let mut acc = 0.0;
            for m in 0..2u32 {
                let l = (sub % 2) * 2 + m;
                let signs = u32::from(w[signs_off + (4 * ib32 + l) as usize]);
                let qbase = qs_off + (8 * ib32 + 2 * l) as usize;
                for j in 0..8u32 {
                    let half_j = j >> 2;
                    let idx = u32::from(w[qbase + half_j as usize])
                        | ((hb << (8 - 2 * l - half_j)) & 256);
                    acc += db
                        * iq_grid4(packed::IQ3S_GRID_OFF, idx, j & 3) as f32
                        * iq_sign(signs, j)
                        * x[x_lane + (m * 8 + j) as usize];
                }
            }
            acc
        };
        let bd_iq1_s = |w: &[u8], x: &[f32], bo: usize, xo: usize, sub: u32| -> f32 {
            let d = half(w, bo);
            let (qs_off, qh_off) = (bo + 2, bo + 34);
            let ib = sub / 2;
            let qh = u16le(w, qh_off + (2 * ib) as usize);
            let dl = d * (2 * ((qh >> 12) & 7) + 1) as f32;
            let delta = if qh & 0x8000 != 0 {
                -packed::IQ1_DELTA
            } else {
                packed::IQ1_DELTA
            };
            let x_lane = xo + (sub * 16) as usize;
            let mut acc = 0.0;
            for m in 0..2u32 {
                let l = (sub % 2) * 2 + m;
                let idx =
                    u32::from(w[qs_off + (4 * ib + l) as usize]) | (((qh >> (3 * l)) & 7) << 8);
                for j in 0..8u32 {
                    acc += dl
                        * (iq_grid8_signed(packed::IQ1S_GRID_OFF, idx, j) + delta)
                        * x[x_lane + (m * 8 + j) as usize];
                }
            }
            acc
        };
        let bd_iq1_m = |w: &[u8], x: &[f32], bo: usize, xo: usize, sub: u32| -> f32 {
            let (qs_off, qh_off, scales_off) = (bo, bo + 32, bo + 48);
            let sw: [u32; 4] = std::array::from_fn(|i| u16le(w, scales_off + 2 * i));
            let packed_d = (sw[0] >> 12)
                | ((sw[1] >> 8) & 0x00F0)
                | ((sw[2] >> 4) & 0x0F00)
                | (sw[3] & 0xF000);
            let d = f32::from(half::f16::from_bits(packed_d as u16));
            let ib = sub / 2;
            let s = sw[(ib / 2) as usize];
            let hi = sub % 2;
            let shift = 6 * (ib % 2) + hi * 3;
            let dl = d * (2 * ((s >> shift) & 7) + 1) as f32;
            let qhb = u32::from(w[qh_off + (2 * ib + hi) as usize]);
            let x_lane = xo + (sub * 16) as usize;
            let mut acc = 0.0;
            for m in 0..2u32 {
                let l = hi * 2 + m;
                let (hshift, bit) = if l % 2 == 1 { (4, 0x80) } else { (8, 0x08) };
                let idx = u32::from(w[qs_off + (4 * ib + l) as usize]) | ((qhb << hshift) & 0x700);
                let delta = if qhb & bit != 0 {
                    -packed::IQ1_DELTA
                } else {
                    packed::IQ1_DELTA
                };
                for j in 0..8u32 {
                    acc += dl
                        * (iq_grid8_signed(packed::IQ1S_GRID_OFF, idx, j) + delta)
                        * x[x_lane + (m * 8 + j) as usize];
                }
            }
            acc
        };
        let bd_iq4_xs = |w: &[u8], x: &[f32], bo: usize, xo: usize, sub: u32| -> f32 {
            const KVALUES: [i8; 16] = [
                -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
            ];
            let d = half(w, bo);
            let scales_h = u16le(w, bo + 2);
            let (scales_l_off, qs_off) = (bo + 4, bo + 8);
            let (ib, hi) = (sub / 2, sub % 2);
            let low = (u32::from(w[scales_l_off + (ib / 2) as usize]) >> (4 * (ib % 2))) & 0xF;
            let high = (scales_h >> (2 * ib)) & 3;
            let dl = d * ((low | (high << 4)) as i32 - 32) as f32;
            let qbase = qs_off + (16 * ib) as usize;
            let x_lane = xo + (sub * 16) as usize;
            let mut acc = 0.0;
            for i in 0..16usize {
                let byte = u32::from(w[qbase + i]);
                let nib = if hi == 0 { byte & 0xF } else { byte >> 4 };
                acc += dl * f32::from(KVALUES[nib as usize]) * x[x_lane + i];
            }
            acc
        };

        type BlockDot<'a> = &'a dyn Fn(&[u8], &[f32], usize, usize, u32) -> f32;
        let block_dots: [(u32, BlockDot<'_>); 8] = [
            (GGML_TYPE_IQ2_XXS, &bd_iq2_xxs),
            (GGML_TYPE_IQ2_XS, &bd_iq2_xs),
            (GGML_TYPE_IQ2_S, &bd_iq2_s),
            (GGML_TYPE_IQ3_XXS, &bd_iq3_xxs),
            (GGML_TYPE_IQ3_S, &bd_iq3_s),
            (GGML_TYPE_IQ1_S, &bd_iq1_s),
            (GGML_TYPE_IQ1_M, &bd_iq1_m),
            (GGML_TYPE_IQ4_XS, &bd_iq4_xs),
        ];

        for (ggml_type, deq) in cases {
            let (block_bytes, elems) = block_layout(ggml_type).expect("an IQ type");
            const BLOCKS: usize = 4;
            let mut w: Vec<u8> = (0..block_bytes * BLOCKS).map(|_| byte()).collect();
            for b in 0..BLOCKS {
                let d = half::f16::from_f32(0.05 + 0.01 * b as f32);
                if ggml_type == GGML_TYPE_IQ1_M {
                    // No `f16` header: the block scale is four nibbles, one
                    // from the top of each 16-bit scale word. Pin those four
                    // and leave the low bits — the per-16 sub-scales and the
                    // sign bits — random, which is the part worth testing.
                    let bits = d.to_bits();
                    for i in 0..4 {
                        let off = b * block_bytes + 48 + 2 * i + 1;
                        let nib = ((bits >> (4 * i)) & 0xF) as u8;
                        w[off] = (w[off] & 0x0F) | (nib << 4);
                    }
                } else {
                    w[b * block_bytes..b * block_bytes + 2].copy_from_slice(&d.to_le_bytes());
                }
            }
            let expected = dequantize(ggml_type, &w, elems * BLOCKS).expect("cpu dequantizer");
            for b in 0..BLOCKS {
                for k in 0..elems {
                    let got = deq(&w, b * block_bytes, k as u32);
                    let want = expected[b * elems + k];
                    assert!(
                        (got - want).abs() <= want.abs() * 1e-5 + 1e-5,
                        "type {ggml_type} block {b} element {k}: kernel {got}, cpu {want}"
                    );
                }
            }

            // And the hoisted form, as a whole-block dot summed over lanes.
            let bd = block_dots
                .iter()
                .find(|(t, _)| *t == ggml_type)
                .map(|(_, f)| f)
                .expect("every IQ type has a block_dot");
            let x: Vec<f32> = (0..elems * BLOCKS)
                .map(|i| ((i % 23) as f32 - 11.0) * 0.0625)
                .collect();
            for b in 0..BLOCKS {
                let got: f32 = (0..16)
                    .map(|sub| bd(&w, &x, b * block_bytes, b * elems, sub))
                    .sum();
                let want: f32 = (0..elems)
                    .map(|k| expected[b * elems + k] * x[b * elems + k])
                    .sum();
                assert!(
                    (got - want).abs() <= want.abs() * 1e-4 + 1e-4,
                    "type {ggml_type} block {b}: block_dot sum {got}, cpu {want}"
                );
            }
        }
    }

    /// A type with no middle answers `None` rather than rendering a kernel
    /// with no `dequant_element` in it, which would fail at compile time on
    /// the device instead of at startup here.
    #[test]
    fn an_unsupported_type_has_no_kernel() {
        // `MXFP4` is read by `engine::quant` on the CPU and has no middle
        // here — the shape of every type this module does not cover.
        for dialect in DIALECTS {
            assert!(kernel_source(dialect, crate::engine::quant::GGML_TYPE_MXFP4).is_none());
        }
    }
}
