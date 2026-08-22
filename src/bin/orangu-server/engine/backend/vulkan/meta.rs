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
//! The uniform-buffer POD structs the compute shaders read.
//!
//! Every type here is `#[repr(C)]` and `bytemuck::Pod` because it is written
//! straight into a `wgpu` uniform buffer and read back by WGSL as a struct
//! declaration in `vulkan_shaders`. The two declarations are the same layout
//! written twice, in two languages, with nothing checking them against each
//! other at compile time — which is why they are collected here rather than
//! sitting beside the code that fills them: a field added on one side and not
//! the other is a silent misread of every field after it, and the two lists
//! being adjacent is the cheapest defence available.

/// `Meta` in `vulkan_shaders::PRELUDE` — `#[repr(C)]` so its layout matches
/// WGSL's `struct Meta { in_dim: u32, out_dim: u32, n_tokens: u32,
/// row_bytes: u32 }` field-for-field.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct Meta {
    pub(super) in_dim: u32,
    pub(super) out_dim: u32,
    pub(super) n_tokens: u32,
    pub(super) row_bytes: u32,
}

/// `ElemMeta` in `vulkan_shaders::ELEM_META` — `#[repr(C)]` so its layout
/// matches WGSL's `struct ElemMeta { len: u32, aux: u32, extra: f32,
/// out_scale: f32 }` field-for-field. `extra` is `eps` for the RMSNorm
/// pipeline, the multiplier for the scale pipeline, and unused (left `0.0`) for
/// add/mul/gelu. `out_scale` is `layer_output_scale` for the
/// `rmsnorm_add_scale` pipeline only (see
/// `vulkan_shaders::shader_source_rmsnorm_add_scale`); left `0.0` and unread by
/// every other shader.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ElemMeta {
    pub(super) len: u32,
    /// A second integer whose meaning is the shader's: the KV-cast shader
    /// reads it as a destination `offset`, the bias-add shader as the `row`
    /// width to broadcast along. `0` and unread everywhere else. Shared rather
    /// than grown into per-shader structs because the binding layout is what
    /// costs, and these three agree on everything except this word.
    pub(super) aux: u32,
    pub(super) extra: f32,
    pub(super) out_scale: f32,
}

/// `AttnMeta` in `vulkan_shaders::ATTENTION_SHADER` — `#[repr(C)]` so its
/// layout matches WGSL's `struct AttnMeta` field-for-field. The last four
/// fields are read only by the multi-query (prefill) variant of the kernel,
/// which derives each query's own window from them; the single-query variant
/// takes `window_start`/`n_pos` as given and leaves them zero.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct AttnMeta {
    pub(super) n_head: u32,
    pub(super) n_head_kv: u32,
    pub(super) head_dim: u32,
    pub(super) window_start: u32,
    pub(super) n_pos: u32,
    pub(super) capacity: u32,
    pub(super) scale: f32,
    pub(super) start_pos: u32,
    pub(super) n_query: u32,
    pub(super) n_swa: u32,
    pub(super) causal: u32,
    pub(super) _pad: u32,
}

/// `AttnSplitMeta` in `vulkan_shaders::ATTENTION_SPLIT_SHADER_TEMPLATE` —
/// `#[repr(C)]` so its layout matches WGSL's `struct AttnSplitMeta {
/// n_head: u32, n_head_kv: u32, head_dim: u32, window_start: u32, n_pos:
/// u32, k_num: u32, scale: f32, _pad: u32 }` field-for-field. Almost
/// `AttnMeta`'s own shape, `capacity` swapped for `k_num` — split-k phase
/// 1 doesn't need `capacity` (it never reads past `n_pos`, unlike the
/// un-split kernel which doesn't either — `capacity` is otherwise unused
/// dead weight in `AttnMeta` too, kept there only for layout stability
/// with `probs_scratch`-era code).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct AttnSplitMeta {
    pub(super) n_head: u32,
    pub(super) n_head_kv: u32,
    pub(super) head_dim: u32,
    pub(super) window_start: u32,
    pub(super) n_pos: u32,
    pub(super) k_num: u32,
    pub(super) scale: f32,
    pub(super) _pad: u32,
}

/// `AttnReduceMeta` in `vulkan_shaders::ATTENTION_SPLIT_REDUCE_SHADER` —
/// `#[repr(C)]` so its layout matches WGSL's `struct AttnReduceMeta {
/// head_dim: u32, k_num: u32, _pad0: u32, _pad1: u32 }` field-for-field.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct AttnReduceMeta {
    pub(super) head_dim: u32,
    pub(super) k_num: u32,
    pub(super) _pad0: u32,
    pub(super) _pad1: u32,
}

/// The YaRN tail both [`RopeMeta`] and [`FusedNormRopeMeta`] carry — five
/// `f32`s plus padding to a 16-byte multiple, matching the fields
/// `vulkan_shaders::ROPE_YARN_WGSL` documents in each WGSL struct.
///
/// Nested here rather than spelled out twice because a rope kernel that
/// disagrees with its fused twin is silent: same shapes, same magnitudes,
/// wrong angle.
///
/// `#[repr(C)]` with only 4-byte members, so nesting it costs no padding and
/// the bytes land exactly where the inlined WGSL fields expect them.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RopeYarn {
    freq_scale: f32,
    ext_factor: f32,
    /// ggml's `mscale`, already folded with `attn_factor`.
    mscale: f32,
    corr_lo: f32,
    corr_hi: f32,
    _pad: [u32; 3],
}

impl RopeYarn {
    /// The unscaled rope every non-YaRN model uses: `freq_scale * theta` with
    /// `freq_scale == 1.0` and `sin/cos * 1.0`, both exact in IEEE 754, so a
    /// shader carrying this is bit-identical to one with no YaRN terms at all.
    pub const IDENTITY: Self = Self {
        freq_scale: 1.0,
        ext_factor: 0.0,
        mscale: 1.0,
        corr_lo: 0.0,
        corr_hi: 0.0,
        _pad: [0; 3],
    };

    /// The GPU form of a CPU [`crate::engine::tensor::RopeParams`], taking the
    /// derived constants from [`crate::engine::tensor::RopeParams::yarn_terms`]
    /// rather than recomputing them.
    pub fn from_params(params: &crate::engine::tensor::RopeParams) -> Self {
        let (corr_lo, corr_hi, mscale) = params.yarn_terms();
        Self {
            freq_scale: params.freq_scale,
            ext_factor: params.ext_factor,
            mscale,
            corr_lo,
            corr_hi,
            _pad: [0; 3],
        }
    }
}

/// `RopeMeta` in `vulkan_shaders::ROPE_SHADER` — `#[repr(C)]` so its
/// layout matches WGSL's `struct RopeMeta { n_head: u32, head_dim: u32,
/// rope_dim: u32, pos: u32, freq_base: f32, n_tokens: u32, pairing: u32,
/// _pad2: u32, <RopeYarn> }` field-for-field. `n_head` is heads *per token*
/// and `n_tokens` the rows in the batch; `pos` is the first row's position.
/// `layout` is `0` for NEOX pairing and `1` for NORM — see
/// [`rope_layout_code`].
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct RopeMeta {
    pub(super) n_head: u32,
    pub(super) head_dim: u32,
    pub(super) rope_dim: u32,
    pub(super) pos: u32,
    pub(super) freq_base: f32,
    pub(super) n_tokens: u32,
    /// `0` = NEOX, `1` = NORM. Named `pairing` rather than `layout` because
    /// WGSL reserves the latter.
    pub(super) pairing: u32,
    pub(super) _pad2: u32,
    pub(super) yarn: RopeYarn,
}

/// The `RopeMeta::layout` code for a CPU-side [`crate::engine::tensor::
/// RopeLayout`], so the two descriptions of one convention live next to each
/// other instead of as a bare `0`/`1` at each call site.
pub fn rope_layout_code(layout: crate::engine::tensor::RopeLayout) -> u32 {
    match layout {
        crate::engine::tensor::RopeLayout::Neox => 0,
        crate::engine::tensor::RopeLayout::Norm => 1,
    }
}

/// `PerHeadNormMeta` in `vulkan_shaders::PERHEAD_RMSNORM_SHADER`/
/// `PERHEAD_RMSNORM_WEIGHTLESS_SHADER` — `#[repr(C)]` so its layout
/// matches WGSL's `struct PerHeadNormMeta { n_head: u32, head_dim: u32,
/// eps: f32, _pad: u32 }` field-for-field.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct PerHeadNormMeta {
    pub(super) n_head: u32,
    pub(super) head_dim: u32,
    pub(super) eps: f32,
    pub(super) _pad: u32,
}

/// `FusedNormRopeMeta` in `vulkan_shaders::FUSED_NORM_ROPE_SHADER` —
/// `#[repr(C)]` so its layout matches WGSL's `struct FusedNormRopeMeta {
/// n_head: u32, head_dim: u32, rope_dim: u32, pos: u32, freq_base: f32,
/// eps: f32, _pad0: u32, _pad1: u32, <RopeYarn> }` field-for-field. The union
/// of `RopeMeta`'s and `PerHeadNormMeta`'s own fields (`n_head` is common to
/// both, so this has one copy, not two).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct FusedNormRopeMeta {
    pub(super) n_head: u32,
    pub(super) head_dim: u32,
    pub(super) rope_dim: u32,
    pub(super) pos: u32,
    pub(super) freq_base: f32,
    pub(super) eps: f32,
    /// `0` = NEOX, `1` = NORM — see [`rope_layout_code`]. Named to match
    /// `RopeMeta::pairing`; WGSL reserves `layout`.
    pub(super) pairing: u32,
    pub(super) _pad1: u32,
    pub(super) yarn: RopeYarn,
}

/// `SampleMeta` in `vulkan_shaders::ARGMAX_PENALTY_SHADER` —
/// `#[repr(C)]` so its layout matches WGSL's `struct SampleMeta {
/// n_vocab: u32, n_recent: u32, repeat_penalty: f32, logit_softcap: f32 }`
/// field-for-field.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct SampleMeta {
    pub(super) n_vocab: u32,
    pub(super) n_recent: u32,
    pub(super) repeat_penalty: f32,
    /// Gemma final-logit softcap `cap` (`cap * tanh(v / cap)`), applied to
    /// every logit by the softcap phase *before* the repeat-penalty phase —
    /// so the GPU sample path reproduces the CPU order softcap → penalty →
    /// argmax exactly. `0.0` means "no softcap" (the softcap phase is not
    /// dispatched, so the value is never read in that case).
    pub(super) logit_softcap: f32,
}

/// `ArgmaxSplitMeta` in `vulkan_shaders::ARGMAX_SPLIT_SHADER` —
/// `#[repr(C)]` so its layout matches WGSL's `struct ArgmaxSplitMeta {
/// n_vocab: u32, n_split: u32, _pad0: u32, _pad1: u32 }` field-for-field.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ArgmaxSplitMeta {
    pub(super) n_vocab: u32,
    pub(super) n_split: u32,
    pub(super) _pad0: u32,
    pub(super) _pad1: u32,
}

/// Per-slot GPU-sample scratch, cached across decode steps so
/// `VulkanBackend::record_argmax_sample` allocates **nothing** on the hot
/// path — every decode token reuses the same buffers and bind groups,
/// re-writing only the three per-token uniforms (logits, recent-token list,
/// `SampleMeta`) via `queue.write_buffer`. Keyed by `batch_slot` (like
/// `op_cache`) so concurrently-decoding sequences never share the same
/// `out_buf`/`logits_buf`. Every field is `wgpu`'s `Arc`-backed handle, so a
/// caller clones the few it needs out from under the cache lock and records
/// without holding it. The `split_meta`/`reduce_meta`/`partial_*` buffers
/// aren't stored explicitly — the `split_bind_group`/`reduce_bind_group`
/// keep them alive, and their contents are constant for a given `n_vocab`
/// (written once at build). Rebuilt when `n_vocab` changes or a call needs a
/// larger recent-token window than the cached `recent_cap`.
pub(super) struct ArgmaxSampleResources {
    pub(super) n_vocab: usize,
    pub(super) recent_cap: usize,
    pub(super) logits_buf: wgpu::Buffer,
    pub(super) recent_buf: wgpu::Buffer,
    pub(super) out_buf: wgpu::Buffer,
    pub(super) sample_meta_buf: wgpu::Buffer,
    pub(super) penalty_bind_group: wgpu::BindGroup,
    pub(super) split_bind_group: wgpu::BindGroup,
    pub(super) reduce_bind_group: wgpu::BindGroup,
}
