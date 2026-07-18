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

//! A GPU `Backend` via `wgpu`'s Vulkan backend (pure Rust — `wgpu` dlopens
//! the Vulkan loader at runtime through `ash`, no Vulkan SDK needed to
//! *build* `orangu-server`, only a working Vulkan driver to *run* it on a
//! GPU). A conformant Vulkan driver implements Vulkan directly on the GPU,
//! so `VulkanBackend` already reaches a wide range of GPUs (AMD included)
//! without any vendor-specific code — a genuine HIP/ROCm backend also
//! exists (`engine::backend::rocm`) for AMD hardware that needs it
//! specifically, but this path needs nothing beyond a working Vulkan
//! driver.
//!
//! Each supported `ggml_type` gets two compute pipelines — `pipelines`
//! (`vulkan_shaders::shader_source_reduce`, one workgroup per `(row,
//! token)` pair reducing across all 64 threads, used for small `n_tokens`
//! like decode) and `pipelines_coop` (`vulkan_shaders::shader_source_coop`,
//! one workgroup per row shared across many tokens, used for a long
//! prompt's prefill) — dequantizing a weight row's raw bytes and
//! dot-producting it against the input activations directly on the GPU.
//! The weight tensor is uploaded once (still quantized) and cached for the
//! model's lifetime (`weight_cache`, keyed by [`QuantMatrix::cache_key`]),
//! never re-uploaded on later `matmul` calls for the same tensor.
//!
//! The activation/output/readback/uniform buffers and the bind group tying
//! them (plus a weight buffer) together are *also* cached now, keyed by
//! `(weight tensor identity, n_tokens)` (`op_cache`) — every decode step
//! re-issues the exact same shape of call against the exact same weight
//! tensors (`n_tokens == 1`, every layer, every token), so after the first
//! token these calls hit the cache and skip buffer/bind-group creation
//! entirely, down to just an activation upload + dispatch + readback.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};

use super::vulkan_shaders;
use super::{Backend, MatmulOp};
use crate::engine::loader::QuantMatrix;

/// Bind group layout shared by every type's pipeline: `weights` (storage,
/// read-only, the raw quantized bytes), `x` (storage, read-only, the
/// input activations), `y` (storage, read-write, the output), `meta`
/// (uniform, the shapes — see `vulkan_shaders::PRELUDE`'s `Meta` struct).
fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let storage = |read_only: bool| wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only },
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("orangu-server matmul bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: storage(true),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: storage(true),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: storage(false),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

/// `Meta` in `vulkan_shaders::PRELUDE` — `#[repr(C)]` so its layout matches
/// WGSL's `struct Meta { in_dim: u32, out_dim: u32, n_tokens: u32,
/// row_bytes: u32 }` field-for-field.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Meta {
    in_dim: u32,
    out_dim: u32,
    n_tokens: u32,
    row_bytes: u32,
}

/// `ElemMeta` in `vulkan_shaders::ELEM_META` — `#[repr(C)]` so its layout
/// matches WGSL's `struct ElemMeta { len: u32, _pad0: u32, extra: f32,
/// _pad1: u32 }` field-for-field. `extra` is `eps` for the RMSNorm pipeline,
/// the multiplier for the scale pipeline, and unused (left `0.0`) for add/
/// mul/gelu.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ElemMeta {
    len: u32,
    _pad0: u32,
    extra: f32,
    _pad1: u32,
}

/// `AttnMeta` in `vulkan_shaders::ATTENTION_SHADER` — `#[repr(C)]` so its
/// layout matches WGSL's `struct AttnMeta { n_head: u32, n_head_kv: u32,
/// head_dim: u32, window_start: u32, n_pos: u32, capacity: u32, scale:
/// f32, _pad: u32 }` field-for-field.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AttnMeta {
    n_head: u32,
    n_head_kv: u32,
    head_dim: u32,
    window_start: u32,
    n_pos: u32,
    capacity: u32,
    scale: f32,
    _pad: u32,
}

/// `RopeMeta` in `vulkan_shaders::ROPE_SHADER` — `#[repr(C)]` so its
/// layout matches WGSL's `struct RopeMeta { n_head: u32, head_dim: u32,
/// rope_dim: u32, pos: u32, freq_base: f32, _pad0: u32, _pad1: u32,
/// _pad2: u32 }` field-for-field.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RopeMeta {
    n_head: u32,
    head_dim: u32,
    rope_dim: u32,
    pos: u32,
    freq_base: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// `PerHeadNormMeta` in `vulkan_shaders::PERHEAD_RMSNORM_SHADER`/
/// `PERHEAD_RMSNORM_WEIGHTLESS_SHADER` — `#[repr(C)]` so its layout
/// matches WGSL's `struct PerHeadNormMeta { n_head: u32, head_dim: u32,
/// eps: f32, _pad: u32 }` field-for-field.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PerHeadNormMeta {
    n_head: u32,
    head_dim: u32,
    eps: f32,
    _pad: u32,
}

/// `SampleMeta` in `vulkan_shaders::ARGMAX_SAMPLE_SHADER` — `#[repr(C)]`
/// so its layout matches WGSL's `struct SampleMeta { n_vocab: u32,
/// n_recent: u32, repeat_penalty: f32, _pad: u32 }` field-for-field.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SampleMeta {
    n_vocab: u32,
    n_recent: u32,
    repeat_penalty: f32,
    _pad: u32,
}

/// Bind group layout for the binary elementwise/norm shaders (`add`, `mul`,
/// `rmsnorm`): two read-only storage buffers, one read-write storage
/// buffer, one uniform — `rmsnorm`'s `(x, weight, y, meta)` happens to have
/// the exact same binding shape as `add`/`mul`'s `(a, b, y, meta)`, so all
/// three share one layout and one pipeline layout.
fn elem4_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let storage = |read_only: bool| wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only },
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("orangu-server elem4 bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: storage(true),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: storage(true),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: storage(false),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

/// Bind group layout for the unary elementwise shaders (`gelu`, `scale`):
/// one read-only storage buffer, one read-write storage buffer, one
/// uniform.
fn elem3_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let storage = |read_only: bool| wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only },
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("orangu-server elem3 bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: storage(true),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: storage(false),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

/// Bind group layout for `perhead_rmsnorm_weightless_pipeline` (V's
/// weightless norm): one read-write storage buffer, one uniform — no
/// weight vector, unlike [`elem3_bind_group_layout`]'s shape.
fn elem2_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("orangu-server elem2 bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

/// Bind group layout for `attn_pipeline`: `aq` (storage, read-only, this
/// token's query vectors), `k_cache`/`v_cache` (storage, read-only, the
/// GPU-resident KV cache mirror — see `engine::kv_cache::GpuLayerCache`),
/// `probs_scratch` (storage, read-write, softmax working memory),
/// `aout` (storage, read-write, the attention output), `am` (uniform,
/// shapes/position) — see `vulkan_shaders::shader_source_attention`.
fn attn_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let storage = |read_only: bool| wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only },
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    let entry = |binding: u32, ty: wgpu::BindingType| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty,
        count: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("orangu-server attention bind group layout"),
        entries: &[
            entry(0, storage(true)),
            entry(1, storage(true)),
            entry(2, storage(true)),
            entry(3, storage(false)),
            entry(4, storage(false)),
            entry(
                5,
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            ),
        ],
    })
}

/// Bind group layout for `vulkan_shaders::ARGMAX_SAMPLE_SHADER` —
/// `logits` (storage, read-write: mutated in place by the repeat-penalty
/// step), `recent_tokens` (storage, read-only), `out_token` (storage,
/// read-write, one `u32`), `meta` (uniform).
fn argmax_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let storage = |read_only: bool| wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only },
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    let entry = |binding: u32, ty: wgpu::BindingType| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty,
        count: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("orangu-server argmax sample bind group layout"),
        entries: &[
            entry(0, storage(false)),
            entry(1, storage(true)),
            entry(2, storage(false)),
            entry(
                3,
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            ),
        ],
    })
}

/// `~/.orangu/server/<key>/cache.bin` — a persistent, on-disk pipeline
/// cache. `key` is `wgpu::util::pipeline_cache_key`'s output
/// (vendor/device-derived, so a cache built
/// for one GPU is never handed to a different one), one directory per
/// adapter rather than a flat file, matching `web::sessions::sessions_dir`'s
/// own "one identifier, one directory" shape rather than introducing a
/// second, differently-shaped convention. `None` if the home directory
/// can't be resolved — this cache is a startup-time optimization only,
/// never required for correctness, so a missing `$HOME` just means "skip
/// the cache," not "fail to start."
fn pipeline_cache_file_path(key: &str) -> Option<PathBuf> {
    Some(
        home::home_dir()?
            .join(".orangu/server")
            .join(key)
            .join("cache.bin"),
    )
}

/// Writes `data` to `path` atomically (temp file, then rename over the
/// real path) — `wgpu::PipelineCache`'s own doc comment recommends exactly
/// this so a crash or concurrent write mid-save can never leave a
/// truncated, half-written cache file for the next startup to try to load.
fn save_pipeline_cache(path: &std::path::Path, data: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("bin.tmp");
    std::fs::write(&tmp, data)?;
    std::fs::rename(&tmp, path)
}

/// `(weight cache_key.0, weight cache_key.1, in_dim, out_dim, row_bytes,
/// n_tokens)`. The shape fields beyond the raw `cache_key()` address pair
/// are redundant in production — `LoadedModel`'s `mmap` lives for the
/// whole server process, so a `(ptr, start)` pair is never reused for a
/// different tensor — but they make a same-address collision impossible
/// to *silently* misuse even so: a stale entry with a different shape
/// simply misses the cache and gets rebuilt correctly-sized, rather than
/// having its (wrong-sized) buffers reused. Address reuse can genuinely
/// happen for short-lived `QuantMatrix`es backed by a temp-file `mmap`
/// that gets unmapped and then coincidentally remapped at the same
/// address (exactly what the cross-check tests do, sharing one
/// `VulkanBackend`/`op_cache` — see `tests::shared_vulkan`) — this was
/// caught by, not just anticipated for, that exact scenario.
type OpCacheKey = (usize, usize, u32, usize, usize, usize, usize);

/// `(cache_key.0, cache_key.1, ggml_type, raw_bytes().len())`.
type WeightCacheKey = (usize, usize, u32, usize);

pub struct VulkanBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    bind_group_layout: wgpu::BindGroupLayout,
    pipelines: HashMap<u32, wgpu::ComputePipeline>,
    /// The workgroup-cooperative variant of each type's pipeline (see
    /// `vulkan_shaders::shader_source_coop`) — dispatched instead of
    /// `pipelines`' entry when `n_tokens` is large enough that sharing a
    /// dequantized block across many tokens' threads beats each token
    /// dequantizing it independently (`COOP_MIN_N_TOKENS`).
    pipelines_coop: HashMap<u32, wgpu::ComputePipeline>,
    /// The opt-in (`ORANGU_TILED_PREFILL=1`, see `Self::tiled_prefill`)
    /// tiled-GEMM alternative to `pipelines_coop` —
    /// `vulkan_shaders::shader_source_coop_tiled`.
    /// Built unconditionally alongside `pipelines_coop` (compiling an
    /// unused pipeline costs a little startup time, not correctness), so
    /// `Self::use_tiled_coop` only ever has to choose which already-built
    /// map to read from, never whether one exists.
    pipelines_coop_tiled: HashMap<u32, wgpu::ComputePipeline>,
    /// `ggml_type`/byte length are redundant in production (see
    /// `OpCacheKey`'s doc comment for why: a real model's `mmap` address
    /// is never reused for a different tensor), but the same
    /// short-lived-mmap address-reuse collision `OpCacheKey` guards
    /// against applies here too, and this closes it the same way — a
    /// stale entry with a different type or byte length simply misses the
    /// cache instead of being silently reused. Byte length alone isn't
    /// enough: two types can share an identical row layout (`F16`/`BF16`
    /// are both 2 bytes/element, 1 element/block) while decoding those
    /// bytes completely differently — exactly the pair that first caught
    /// this needing `ggml_type` too, not just a length check.
    weight_cache: Mutex<HashMap<WeightCacheKey, Arc<wgpu::Buffer>>>,
    /// Each entry is individually locked (not the whole map) for the
    /// duration of the op that uses it, so unrelated ops (different weight
    /// tensors, or the same tensor at a different `n_tokens`) never
    /// contend with each other — only two calls that would race on the
    /// exact same buffers do, which is also the only case where
    /// serializing them is actually required for correctness.
    op_cache: Mutex<HashMap<OpCacheKey, Arc<Mutex<CachedOpResources>>>>,
    /// The adapter's own description (the driver's GPU name string) — for
    /// the startup banner.
    pub adapter_name: String,
    /// Bind group layout shared by `add_pipeline`/`mul_pipeline`/
    /// `rmsnorm_pipeline` — see `elem4_bind_group_layout`.
    elem4_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group layout shared by `gelu_pipeline`/`scale_pipeline` — see
    /// `elem3_bind_group_layout`.
    elem3_bind_group_layout: wgpu::BindGroupLayout,
    add_pipeline: wgpu::ComputePipeline,
    mul_pipeline: wgpu::ComputePipeline,
    gelu_pipeline: wgpu::ComputePipeline,
    scale_pipeline: wgpu::ComputePipeline,
    rmsnorm_pipeline: wgpu::ComputePipeline,
    /// Every buffer/bind group `fused_post_attention` needs *except* the
    /// three that genuinely change contents every call (`wo`'s own
    /// `x_buffer`, already covered by `op_cache`; the residual snapshot;
    /// PLE's per-layer slice) is built once per layer and reused forever
    /// after, the same reuse discipline `op_cache` already applies to plain
    /// matmul calls — bind group and buffer *creation* has real driver
    /// cost, so paying it once per layer instead of once per token matters
    /// for decode, where this whole chain runs every single token.
    fused_cache: Mutex<HashMap<FusedCacheKey, Arc<FusedResources>>>,
    /// Bind group layout for `attn_pipeline` — see `attn_bind_group_layout`.
    attn_bind_group_layout: wgpu::BindGroupLayout,
    /// GPU-resident causal attention for decode (`n_tokens == 1`) — see
    /// `vulkan_shaders::shader_source_attention` and `Self::gpu_attention`.
    attn_pipeline: wgpu::ComputePipeline,
    /// Total `queue.submit` calls across this backend's lifetime — a
    /// decode-step-scoped delta of this (see `Self::submission_count`)
    /// reflects how many GPU round trips a decode step makes.
    submission_count: std::sync::atomic::AtomicU64,
    /// GPU RoPE (`vulkan_shaders::shader_source_rope`) — reuses
    /// `elem3_bind_group_layout` (its bindings match that shape).
    rope_pipeline: wgpu::ComputePipeline,
    /// Per-head weighted RMSNorm (Q-norm/K-norm) — also reuses
    /// `elem3_bind_group_layout`.
    perhead_rmsnorm_pipeline: wgpu::ComputePipeline,
    /// Bind group layout for `perhead_rmsnorm_weightless_pipeline` (V's
    /// norm) — see `elem2_bind_group_layout`.
    elem2_bind_group_layout: wgpu::BindGroupLayout,
    /// Per-head weightless RMSNorm (V's norm).
    perhead_rmsnorm_weightless_pipeline: wgpu::ComputePipeline,
    /// Every bind group `fused_attention` needs that *doesn't* touch a
    /// per-request KV cache buffer (Q-norm, Q-RoPE, K-norm, V's norm,
    /// K-RoPE — all of which operate on `op_cache`-cached, model-scoped
    /// weight-projection output buffers, not anything request-specific),
    /// built once per layer and reused forever after — the same reuse
    /// discipline `fused_cache` already applies to `fused_post_attention`.
    /// The one piece of `fused_attention`'s resources that *does* touch
    /// per-request state (the attention dispatch itself, since it binds
    /// this request's own KV-cache mirror) is cached instead on the
    /// request-owned `LayerCache` itself
    /// (`engine::kv_cache::LayerCache::attn_dispatch`) — a
    /// `VulkanBackend`-level cache keyed only by weight-tensor identity
    /// would otherwise have no way to avoid reusing one request's KV
    /// buffers for a different request's attention dispatch.
    fused_attn_layer_cache: Mutex<HashMap<FusedAttnLayerCacheKey, Arc<FusedAttnLayerResources>>>,
    /// `fused_layer`'s own resources (the pre-attention norm's buffers/
    /// bind group, plus the residual-stream and readback buffers) —
    /// model-scoped like `fused_attn_layer_cache`, keyed the same
    /// shape-plus-identity way (`FusedLayerCacheKey`).
    fused_layer_cache: Mutex<HashMap<FusedLayerCacheKey, Arc<FusedLayerResources>>>,
    /// Scoped to its most clearly
    /// justified single piece: the per-request KV mirror
    /// (`engine::kv_cache::LayerCache::sync_gpu`) stored as `f16` instead
    /// of `f32`. **`false` unless both** the adapter supports native WGSL
    /// `f16` **and** `ORANGU_KV_F16=1` is set — see `Self::try_init`'s own
    /// comment at this flag's construction site. Only the KV mirror is
    /// converted; f16 dequant/dot math in the matmul kernels, f16
    /// activation buffers between fused-chain stages, and f16 elementwise/
    /// norm/softmax are **not** part of this flag (those
    /// pieces are weight/dequant-bandwidth-dominated already, or need a much
    /// larger f16-vs-f32 dual-kernel surface than the KV mirror alone).
    /// Every KV-mirror-touching call site (`sync_gpu`, the KV-cache write
    /// in `record_fused_attention`, the attention shader itself) branches
    /// on this flag; the `f32` path is the *original*, already-verified,
    /// still-default code.
    kv_f16: bool,
    /// Casts a freshly RoPE'd/normed `f32` key or value row into the
    /// `f16`-stored KV mirror on write — see `Self::kv_f16`. `None` when
    /// `kv_f16` is `false`; every call site checks `kv_f16` (equivalently,
    /// this being `Some`) before using it.
    kv_cast_pipeline: Option<wgpu::ComputePipeline>,
    /// The reduce
    /// kernel's `f32`-dequant + `f32`-dot inner loop, replaced with a
    /// `Q4_K`-only packed `vec2<f16>` dot —
    /// `vulkan_shaders::shader_source_reduce_q4k_packed_f16`. **`false`
    /// unless both** the adapter supports native WGSL `f16` **and**
    /// `ORANGU_PACKED_DOT=1` is set. A probe (`requires packed_4x8_
    /// integer_dot_product;`/`dot4I8Packed` *and* `vec2<f16>` `dot()`)
    /// confirms both variants actually compile and run
    /// correctly on the adapter before this is enabled. See `Self::try_init`'s
    /// own comment at this flag's construction site.
    packed_dot_f16: bool,
    /// `Q4_K`'s packed-`f16` reduce pipeline — see `Self::packed_dot_f16`.
    /// `None` unless `packed_dot_f16` is `true`.
    q4_k_packed_f16_pipeline: Option<wgpu::ComputePipeline>,
    /// Wide vectorized weight loads: the
    /// reduce kernel's byte-wise `read_u8` weight reads replaced with
    /// `vec4<u32>` (16-byte) loads — `vulkan_shaders::
    /// shader_source_reduce_wide_load`, one pipeline per supported
    /// `ggml_type` (see `Self::wide_load_pipelines`). **`false` unless
    /// `ORANGU_WIDE_LOAD=1` is set.** Correctness-verified against
    /// `CpuBackend` (bit-for-bit for every type — unlike `packed_dot_f16`,
    /// this kernel does the exact same arithmetic as the scalar path, just
    /// sourced from a wider load, so no precision loss to tolerate).
    /// No `SHADER_F16` gate, unlike `kv_f16`/`packed_dot_f16`
    /// — every wide-load kernel's own `f16_to_f32` uses the same
    /// core-WGSL `unpack2x16float` builtin the regular scalar kernels
    /// already use unconditionally, not the native `f16` type those two
    /// flags need `wgpu::Features::SHADER_F16` for.
    wide_load: bool,
    /// One wide-load reduce pipeline per supported `ggml_type` — see
    /// `Self::wide_load`. Empty unless `wide_load` is `true`.
    wide_load_pipelines: HashMap<u32, wgpu::ComputePipeline>,
    /// Wide loads combined with the packed-`f16` dot above —
    /// `vulkan_shaders::shader_source_reduce_q4k_wide_packed_f16`.
    /// `Q4_K`-only, like the packed-dot kernel itself. `Some` only when
    /// **both** `wide_load` and `packed_dot_f16`
    /// are `true` (so `ORANGU_WIDE_LOAD=1 ORANGU_PACKED_DOT=1` together
    /// select this combined kernel, not either flag alone) — see `Self::
    /// pipeline_for`'s own precedence for how the three `Q4_K` reduce
    /// kernels (this one, wide-load-alone, packed-alone) are chosen among.
    ///
    /// Correctness-verified; kept available as a selectable combination
    /// rather than a default (same precedent as `kv_f16`/`gpu_sample`/
    /// `ORANGU_BATCH_DECODE`).
    wide_packed_pipeline: Option<wgpu::ComputePipeline>,
    /// Memory-level-parallelism decode kernel for `Q4_K`
    /// (`vulkan_shaders::shader_source_reduce_q4k_wide_unroll`).
    /// Restructures the reduce inner loop to iterate
    /// whole 256-element super-blocks, issuing several independent
    /// activation/qs-byte loads (and loading each block's header once, not
    /// per element) before the dependent dot, so more memory requests are in
    /// flight per lane on this latency-bound weight stream. Builds on the
    /// wide-load (`array<vec4<u32>>`) binding but with a different loop
    /// shape than `shader_source_reduce_wide_load`'s `MAIN_REDUCE_SUFFIX`
    /// reuse. **On by default (`true`); opt out with `ORANGU_NO_MLP_UNROLL=1`.**
    /// Pure `f32`, so it cross-checks **bit-for-bit** against `CpuBackend`.
    /// Takes precedence over `wide_load`/`packed_dot_f16` for
    /// the `Q4_K` decode (`n_tokens < COOP_MIN_N_TOKENS`) path (see
    /// `Self::selects_wide_unroll`/`pipeline_for`).
    wide_unroll: bool,
    /// The memory-level-parallelism reduce pipelines, one per K-quant type
    /// the block-unroll covers (`Q4_K`/`Q5_K`/`Q6_K` — see
    /// `vulkan_shaders::shader_source_reduce_wide_unroll`). Empty unless
    /// `wide_unroll` is `true`. All three use the identical block-unroll
    /// mechanism and ship default as the same strict, bit-for-bit-verified
    /// improvement.
    wide_unroll_pipelines: HashMap<u32, wgpu::ComputePipeline>,
    /// `Q4_K` block-unroll combined with the packed-`f16` dot
    /// (`vulkan_shaders::shader_source_reduce_q4k_wide_unroll_packed_f16`).
    /// `Some` only when `wide_unroll` **and**
    /// `packed_dot_f16` are both on (`ORANGU_PACKED_DOT=1`), selected ahead
    /// of the scalar unroll for `Q4_K` decode. Since decode is memory-bound
    /// once the block-unroll is in place, halving the multiply-accumulate
    /// count does not change the memory structure. Kept reachable behind the
    /// env var (the same precedent as `wide_packed_pipeline`), not promoted
    /// to default.
    q4_k_unroll_packed_pipeline: Option<wgpu::ComputePipeline>,
    /// The prefill cooperative
    /// path's tiled-GEMM alternative (`pipelines_coop_tiled`,
    /// `vulkan_shaders::shader_source_coop_tiled`), gated behind
    /// `ORANGU_TILED_PREFILL=1` for the same reason `kv_f16`/
    /// `packed_dot_f16` are opt-in: correctness-verified against
    /// `CpuBackend` (`matmul_matches_cpu_backend_cooperative_path_*`, run
    /// with the env var set) and ships opt-in. No
    /// `SHADER_F16`/adapter-feature gate, unlike `kv_f16`/`packed_dot_f16`
    /// — the tiled kernel only uses plain `f32` shared memory.
    tiled_prefill: bool,
    /// Bind group layout for `argmax_sample_pipeline` — see
    /// `argmax_bind_group_layout`.
    argmax_bind_group_layout: wgpu::BindGroupLayout,
    /// See `Self::record_argmax_sample`.
    argmax_sample_pipeline: wgpu::ComputePipeline,
    /// GPU-resident greedy sampling.
    /// **Off by default.** `record_argmax_sample`'s reduction dispatches a
    /// *single* 64-thread workgroup over the full `[n_vocab]` logits buffer,
    /// which underuses the GPU's parallelism; a version that splits the
    /// reduction across many workgroups (a two-level reduction: per-
    /// workgroup partial max, then a small final merge) is not implemented
    /// here. `Self::
    /// forward_maybe_sampling`'s GPU fast path is skipped entirely unless
    /// this is `true`, whatever `greedy_sample` says — the caller still
    /// gets `ForwardOutcome::Logits`, exactly as if no fast path existed.
    gpu_sample: bool,
}

/// `(wq.cache_key().0, wq.cache_key().1, n_head, n_head_kv, head_dim,
/// has_kv, owns_v)` — the same defensive shape-plus-identity pattern
/// `FusedCacheKey`/`OpCacheKey` use.
type FusedAttnLayerCacheKey = (usize, usize, usize, usize, usize, bool, bool);

/// `(wo.cache_key().0, wo.cache_key().1, ffn_gate.out_dim, PLE's
/// per_layer_dim (0 if this layer has no PLE), layer_output_scale.is_some())`
/// — `wo`'s identity alone would be enough in production (see
/// `OpCacheKey`'s doc comment for why), but the extra shape/feature fields
/// make a stale entry miss the cache instead of being silently reused with
/// the wrong shape or wrong optional stages, the same defensive pattern
/// `OpCacheKey`/`WeightCacheKey` already use.
type FusedCacheKey = (usize, usize, usize, usize, bool);

/// `(wq.cache_key().0, wq.cache_key().1, n_embd, eps.to_bits(),
/// attn_norm.as_ptr() as usize)` — the same defensive shape-plus-identity
/// pattern `FusedAttnLayerCacheKey`/`FusedCacheKey` use, extended one field
/// further than a first pass at this key had it. `n_embd` and `eps` are the
/// two shape/config values `build_fused_layer_resources` bakes into
/// `FusedLayerResources`'s buffers and bind group at build time (`eps`
/// goes into the meta buffer once and is never rewritten per call, unlike
/// `pos` in the attention dispatch's own meta buffer) — but `attn_norm`'s
/// own *contents* are baked in too (`attn_norm_w = self.upload_new(attn_
/// norm)`, uploaded once, never refreshed), and nothing about its shape or
/// `eps` distinguishes *which* layer's `attn_norm` that was. A bare
/// `(ptr, start, n_embd, eps.to_bits())` key missed exactly that: two
/// different test-local layers sharing `n_embd`/`eps` (a common case — RMS
/// epsilon is almost always the same constant across a whole model, and
/// small synthetic test shapes routinely reuse round `n_embd` values) can
/// still collide if `wq`'s own address happens to be reused (a real,
/// reproducible risk for the short-lived test-local buffers `engine::
/// backend::vulkan::tests` builds against one shared `VulkanBackend` — see
/// `OpCacheKey`'s doc comment), silently returning a cache entry built
/// with a *different* layer's `attn_norm` weights baked in — caught by,
/// not just anticipated for, exactly that scenario
/// (`fused_layer_matches_cpu_reference_full_layer_with_ple` failing when
/// run immediately after `fused_layer_kv_donor_matches_cpu_reference_
/// many_steps` under `cargo test -- --test-threads=1`, both of which use
/// `n_embd = 24` and `eps = 1e-6`). `attn_norm`'s own pointer closes that
/// gap the same way `QuantMatrix::cache_key()`'s pointer closes it for
/// `wq` itself. `eps` is compared via `to_bits()` since `f32` isn't
/// `Eq`/`Hash`; bit-identical equality is exactly what "was this cache
/// entry built for this exact call's `eps`" needs, not a tolerance-based
/// comparison.
type FusedLayerCacheKey = (usize, usize, usize, u32, usize);

/// One op's GPU-side resources, reused across every call that shares its
/// `(weight, n_tokens)` cache key rather than rebuilt each time. Only
/// `x_buffer`'s *contents* change between reuses (a fresh `write_buffer`
/// per call — the activations themselves differ every time); everything
/// else (`bind_group`, buffer identities/sizes, `workgroups`) is fixed for
/// the key's whole lifetime, since `in_dim`/`out_dim`/`n_tokens`/
/// `row_bytes` are all fixed by it.
struct CachedOpResources {
    bind_group: wgpu::BindGroup,
    x_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    output_len: u64,
    workgroups: (u32, u32, u32),
}

/// Below this many tokens, the regular per-`(row, token)` dispatch (one
/// thread per output element, full occupancy) beats the cooperative
/// dispatch (one workgroup per row, only `n_tokens` of its 64 threads
/// active) — 64 is where the cooperative path's workgroups are first
/// fully occupied too, so it's the natural crossover: below it, the
/// occupancy loss isn't repaid by the redundant-dequant savings; at or
/// above it, both the occupancy and the dequant-sharing favor the
/// cooperative path.
const COOP_MIN_N_TOKENS: usize = 64;

/// How many output rows every reduce/block-unroll kernel
/// (`vulkan_shaders::shader_source_reduce`/`shader_source_reduce_wide_load`/
/// `shader_source_reduce_wide_unroll`/`..._q4k_wide_unroll_packed_f16`, plus
/// their subgroup-reduce variants) computes per workgroup — reading each
/// `x[k]` once and reusing it across all `REDUCE_N_ROWS` rows' dot products,
/// rather than one workgroup per row. Passed straight into every one of
/// those shader-source generators, which unroll their `partial0..partialN`
/// accumulators and dispatch-relevant index math for exactly this many rows
/// (`vulkan_shaders::main_reduce_suffix`/`unroll_suffix`), so this single
/// value now drives both the WGSL row count and
/// [`VulkanBackend::build_op_resources`]'s dispatch-count computation — no
/// second hardcoded copy to drift out of sync with this one.
///
/// Swept against 2, 8, and 16 with a real same-session A/B (3 rounds,
/// alternating, warmup + a measured 128-token greedy generation each): `4`
/// won every round outright, with the same ordering (`4 > 2 > 8 > 16`) in
/// all three. Larger values add per-workgroup register/shared-memory
/// pressure and dispatch fewer, larger workgroups for the same output
/// dimension without buying enough additional memory-level parallelism to
/// pay for it; smaller values dispatch more workgroups that each reuse
/// `x[k]` across fewer rows, so the *same* activation elements get re-read
/// from memory more times in total across the whole dispatch. `4` sits at
/// the point those two costs balance for this backend's decode-time output
/// dimensions. Kept correctness-verified at every one of those swept
/// values too (every existing cross-check test passed under each, not just
/// the shipped default), confirming the generator in `vulkan_shaders` is
/// genuinely parameterized rather than only accidentally correct at `4`.
const REDUCE_N_ROWS: usize = 4;

/// The `ggml_type`s a shader exists for — kept in one place so
/// construction (build every pipeline up front) and the `matmul` dispatch
/// (look one up) can't drift apart.
const SUPPORTED_TYPES: &[u32] = &[
    crate::engine::quant::GGML_TYPE_F32,
    crate::engine::quant::GGML_TYPE_F16,
    crate::engine::quant::GGML_TYPE_BF16,
    crate::engine::quant::GGML_TYPE_Q4_0,
    crate::engine::quant::GGML_TYPE_Q5_0,
    crate::engine::quant::GGML_TYPE_Q8_0,
    crate::engine::quant::GGML_TYPE_Q4_K,
    crate::engine::quant::GGML_TYPE_Q5_K,
    crate::engine::quant::GGML_TYPE_Q6_K,
];

impl VulkanBackend {
    /// Looks for a usable Vulkan adapter and builds every quant type's
    /// compute pipeline up front. Returns `None` (never panics) if no
    /// Vulkan driver is present, or device/pipeline creation otherwise
    /// fails — callers fall back to `CpuBackend` in that case.
    pub fn try_init() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            ..Default::default()
        }))
        .ok()?;
        let info = adapter.get_info();
        let adapter_name = format!("{} ({:?})", info.name, info.backend);

        // Request the adapter's own limits rather than wgpu's conservative
        // portable defaults (128MiB storage buffers) — a model's larger
        // weight matrices (embedding tables, the output projection) are
        // routinely bigger than that.
        let limits = adapter.limits();
        // An `f16`-stored KV
        // mirror (see `Self::kv_f16`'s own doc comment for what else that
        // idea does and does not cover), gated **off by default** and only
        // enabled by `ORANGU_KV_F16=1`, even on adapters that support
        // `wgpu::Features::SHADER_F16`. Built and cross-checked (`Self::
        // kv_cast_pipeline`, `engine::kv_cache`'s `f16` upload path). The
        // KV-cache write's extra compute-shader dispatch (replacing what
        // was a plain `copy_buffer_to_buffer`) trades off against the
        // halved KV-read memory traffic, so whether it helps depends on
        // context length; left available and opt-in rather than made the
        // default.
        // Requested whenever the hardware supports it, independent of
        // whether either `f16`-gated opt-in below (`kv_f16`,
        // `packed_dot_f16`) is actually turned on — a device that never
        // requested this feature can't retroactively use it, so both
        // opt-ins need it present at device-creation time even if only one
        // (or neither) ends up used.
        let supports_f16 = adapter.features().contains(wgpu::Features::SHADER_F16);
        let kv_f16 = supports_f16 && std::env::var_os("ORANGU_KV_F16").is_some();
        // See `Self::
        // packed_dot_f16`'s own doc comment.
        let packed_dot_f16 = supports_f16 && std::env::var_os("ORANGU_PACKED_DOT").is_some();
        // See `Self::
        // tiled_prefill`'s own doc comment for why this stays opt-in.
        let tiled_prefill = std::env::var_os("ORANGU_TILED_PREFILL").is_some();
        // See `Self::wide_load`'s own
        // doc comment. No `supports_f16` gate, unlike `kv_f16`/
        // `packed_dot_f16` above — these kernels only use core-WGSL
        // `unpack2x16float`, available on every adapter.
        let wide_load = std::env::var_os("ORANGU_WIDE_LOAD").is_some();
        // See `Self::wide_unroll`'s own doc comment. **On by default** (opt
        // *out* with `ORANGU_NO_MLP_UNROLL=1`), unlike every other kernel
        // toggle in this file: it's a strict, bit-for-bit-correct, hardware-
        // *general* memory-level-parallelism restructuring of the `Q4_K`
        // decode reduce loop. No `supports_f16`
        // gate — the arithmetic is all `f32`, only `unpack2x16float` (core
        // WGSL) touches half-floats.
        let wide_unroll = std::env::var_os("ORANGU_NO_MLP_UNROLL").is_none();
        // See `Self::gpu_sample`'s own doc comment for why this stays
        // opt-in.
        let gpu_sample = std::env::var_os("ORANGU_GPU_SAMPLE").is_some();
        // `subgroupAdd`/`subgroupMax` hardware reductions in place
        // of the classic 6-round `workgroupBarrier` pairwise-tree
        // reductions — selects the subgroup-reduce shader source for the
        // decode reduce/block-unroll kernels, RMSNorm (both variants), and
        // the attention softmax's per-tile max/sum, all built just below.
        // Correctness-verified (bit-for-bit-within-tolerance against
        // `CpuBackend`, same as every other kernel here) and generalized to
        // *any* subgroup size, not hardcoded to assume the subgroup spans
        // the whole 64-thread workgroup: `subgroupAdd`/`subgroupMax` first
        // reduce within each subgroup, then a short sequential combine over
        // `num_subgroups` (workgroup-uniform, read via the `num_subgroups`/
        // `subgroup_id` builtins) partials — see `wide_unroll`'s own
        // comment for why that generality matters here (`wgpu`'s adapter
        // info only reports subgroup size as a range, not a fixed value).
        // Despite that, a real same-session A/B (several cycles, alternating
        // on/off, warmup + a measured multi-token greedy generation each)
        // measured this a **real, reproducible regression** end-to-end, not
        // a wash. Barrier count was never actually the decode bottleneck —
        // `q4_k_unroll_packed_pipeline`'s own A/B (see its doc comment)
        // already showed decode is memory-bound, not ALU/barrier-bound once
        // `wide_unroll` is in place, so removing barriers here had nothing
        // to buy, and the extra per-lane builtin plumbing this adds (four
        // separate `subgroupAdd` calls per workgroup, one per row) was pure
        // overhead. **Off by default; opt in with `ORANGU_SUBGROUP=1`** —
        // kept available as an honest negative result, the same precedent
        // as `kv_f16`/`gpu_sample`/`ORANGU_BATCH_DECODE`, not deleted, since
        // a different adapter/driver's `subgroupAdd` lowering could still
        // make this pay off.
        // Not stored as a field — every kernel it affects is a straight
        // shader-source swap at pipeline-build time (same buffer layouts,
        // same dispatch shapes either way), so unlike `wide_load`/
        // `packed_dot_f16`/`kv_f16` (which each leave multiple pipeline
        // variants coexisting, chosen between per-call) there is nothing
        // left for any call site to branch on afterward.
        let supports_subgroup = adapter.features().contains(wgpu::Features::SUBGROUP);
        let subgroup_reduce = supports_subgroup && std::env::var_os("ORANGU_SUBGROUP").is_some();
        // A persistent, on-disk pipeline
        // cache — `wgpu::util::pipeline_cache_key` returns
        // `Some` only for the Vulkan backend (the only one this project
        // ever requests, `wgpu::Backends::VULKAN` above) and only encodes
        // the info needed to keep a cache from one GPU/driver from being
        // loaded on an incompatible one; the driver itself does a further,
        // stricter validation of the blob it's handed (see `Self::
        // try_init`'s own comment where the cache is actually created).
        let supports_pipeline_cache = adapter.features().contains(wgpu::Features::PIPELINE_CACHE);
        let pipeline_cache_key = wgpu::util::pipeline_cache_key(&info);
        let mut required_features = if supports_f16 {
            wgpu::Features::SHADER_F16
        } else {
            wgpu::Features::empty()
        };
        if supports_pipeline_cache && pipeline_cache_key.is_some() {
            required_features |= wgpu::Features::PIPELINE_CACHE;
        }
        if supports_subgroup {
            required_features |= wgpu::Features::SUBGROUP;
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("orangu-server"),
            required_features,
            required_limits: limits,
            experimental_features: Default::default(),
            memory_hints: Default::default(),
            trace: Default::default(),
        }))
        .ok()?;

        // Loaded (if present and this adapter supports `PIPELINE_CACHE`)
        // before any pipeline is built, so every `build_pipeline`/
        // `build_elem_pipeline` call below can pass it through and benefit
        // — cuts cold-start shader-compile time on repeat runs against the
        // same GPU/driver, the one thing this cache is for (it has no
        // effect on decode/prefill throughput itself).
        let pipeline_cache_path = pipeline_cache_key
            .as_deref()
            .and_then(pipeline_cache_file_path);
        let pipeline_cache = pipeline_cache_path.as_deref().map(|path| {
            let existing = std::fs::read(path).ok();
            // SAFETY: `existing`, if `Some`, only ever contains bytes this
            // exact process previously wrote via this same `PipelineCache`
            // type's own `get_data()` (see the save at the end of this
            // function) — the one file `pipeline_cache_path` ever names is
            // never touched by anything else. `fallback: true` means a
            // corrupt or genuinely incompatible blob (a different driver
            // version, say) is discarded and a fresh cache used instead,
            // rather than erroring `try_init` out entirely.
            unsafe {
                device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                    label: Some("orangu-server pipeline cache"),
                    data: existing.as_deref(),
                    fallback: true,
                })
            }
        });

        let bind_group_layout = bind_group_layout(&device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("orangu-server matmul pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let build_pipeline = |source: String| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("orangu-server matmul shader"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("orangu-server matmul pipeline"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: pipeline_cache.as_ref(),
            })
        };

        let mut pipelines = HashMap::with_capacity(SUPPORTED_TYPES.len());
        let mut pipelines_coop = HashMap::with_capacity(SUPPORTED_TYPES.len());
        let mut pipelines_coop_tiled = HashMap::with_capacity(SUPPORTED_TYPES.len());
        for &ggml_type in SUPPORTED_TYPES {
            pipelines.insert(
                ggml_type,
                build_pipeline(vulkan_shaders::shader_source_reduce(
                    ggml_type,
                    REDUCE_N_ROWS,
                    subgroup_reduce,
                )?),
            );
            pipelines_coop.insert(
                ggml_type,
                build_pipeline(vulkan_shaders::shader_source_coop(ggml_type)?),
            );
            pipelines_coop_tiled.insert(
                ggml_type,
                build_pipeline(vulkan_shaders::shader_source_coop_tiled(ggml_type)?),
            );
        }

        let elem4_bind_group_layout = elem4_bind_group_layout(&device);
        let elem3_bind_group_layout = elem3_bind_group_layout(&device);
        let elem4_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("orangu-server elem4 pipeline layout"),
                bind_group_layouts: &[Some(&elem4_bind_group_layout)],
                immediate_size: 0,
            });
        let elem3_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("orangu-server elem3 pipeline layout"),
                bind_group_layouts: &[Some(&elem3_bind_group_layout)],
                immediate_size: 0,
            });
        let build_elem_pipeline = |layout: &wgpu::PipelineLayout, source: String| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("orangu-server elem shader"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("orangu-server elem pipeline"),
                layout: Some(layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: pipeline_cache.as_ref(),
            })
        };
        let add_pipeline =
            build_elem_pipeline(&elem4_pipeline_layout, vulkan_shaders::shader_source_add());
        let mul_pipeline =
            build_elem_pipeline(&elem4_pipeline_layout, vulkan_shaders::shader_source_mul());
        let rmsnorm_pipeline = build_elem_pipeline(
            &elem4_pipeline_layout,
            vulkan_shaders::shader_source_rmsnorm(subgroup_reduce),
        );
        let gelu_pipeline =
            build_elem_pipeline(&elem3_pipeline_layout, vulkan_shaders::shader_source_gelu());
        let scale_pipeline = build_elem_pipeline(
            &elem3_pipeline_layout,
            vulkan_shaders::shader_source_scale(),
        );

        let attn_bind_group_layout = attn_bind_group_layout(&device);
        let attn_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("orangu-server attention pipeline layout"),
            bind_group_layouts: &[Some(&attn_bind_group_layout)],
            immediate_size: 0,
        });
        let attn_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("orangu-server attention shader"),
            source: wgpu::ShaderSource::Wgsl(
                vulkan_shaders::shader_source_attention(kv_f16, subgroup_reduce).into(),
            ),
        });
        let attn_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("orangu-server attention pipeline"),
            layout: Some(&attn_pipeline_layout),
            module: &attn_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: pipeline_cache.as_ref(),
        });

        // RoPE and per-head weighted RMSNorm both have the same binding
        // shape as `gelu`/`scale` (read-only storage, read-write storage,
        // uniform — see each shader's own doc comment), so they reuse
        // `elem3_pipeline_layout` rather than needing their own.
        let rope_pipeline =
            build_elem_pipeline(&elem3_pipeline_layout, vulkan_shaders::shader_source_rope());
        let perhead_rmsnorm_pipeline = build_elem_pipeline(
            &elem3_pipeline_layout,
            vulkan_shaders::shader_source_perhead_rmsnorm(subgroup_reduce),
        );

        let elem2_bind_group_layout = elem2_bind_group_layout(&device);
        let elem2_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("orangu-server elem2 pipeline layout"),
                bind_group_layouts: &[Some(&elem2_bind_group_layout)],
                immediate_size: 0,
            });
        let perhead_rmsnorm_weightless_pipeline = build_elem_pipeline(
            &elem2_pipeline_layout,
            vulkan_shaders::shader_source_perhead_rmsnorm_weightless(subgroup_reduce),
        );

        // Casts a freshly
        // RoPE'd/normed `f32` key or value row into the `f16`-stored KV
        // mirror on write (see `Self::kv_f16`'s doc comment). Reuses
        // `elem3_pipeline_layout`/`elem3_bind_group` — same three-binding
        // shape (read-only source, read-write destination, uniform meta)
        // as `rope_pipeline`/`perhead_rmsnorm_pipeline` above, just with a
        // `f16`-typed destination. `None` when the adapter lacks
        // `SHADER_F16` — every call site checks `kv_f16` first.
        let kv_cast_pipeline = kv_f16.then(|| {
            build_elem_pipeline(
                &elem3_pipeline_layout,
                vulkan_shaders::shader_source_kv_cast(),
            )
        });

        // See
        // `Self::packed_dot_f16`'s own doc comment. Reuses the plain
        // matmul `bind_group_layout`/`pipeline_layout` (same 4-binding
        // shape as every other reduce/coop pipeline: weights, x, y,
        // params), not `elem3_pipeline_layout` — `build_pipeline`, not
        // `build_elem_pipeline`.
        let q4_k_packed_f16_pipeline = packed_dot_f16
            .then(|| build_pipeline(vulkan_shaders::shader_source_reduce_q4k_packed_f16()));

        // See `Self::wide_load`'s own
        // doc comment. Same `bind_group_layout`/`pipeline_layout` reuse as
        // `q4_k_packed_f16_pipeline` above: `wgpu::BindGroupLayoutEntry`
        // has no WGSL element-type field (`bind_group_layout`'s own
        // `storage(true)` closure only declares "read-only storage
        // buffer," nothing about `array<u32>` vs. `array<vec4<u32>>`), so
        // a shader is free to reinterpret binding 0's bytes however its
        // own WGSL module declares — confirmed by this actually compiling
        // and cross-checking correctly against `CpuBackend`, not just
        // assumed from reading the wgpu API. One pipeline per type this
        // step covers (same set as `SUPPORTED_TYPES` — every type
        // `shader_source_reduce_wide_load` has a kernel for), built eagerly
        // like every other pipeline in this function, not lazily on first
        // use.
        let wide_load_pipelines: HashMap<u32, wgpu::ComputePipeline> = if wide_load {
            SUPPORTED_TYPES
                .iter()
                .filter_map(|&ggml_type| {
                    let source = vulkan_shaders::shader_source_reduce_wide_load(
                        ggml_type,
                        REDUCE_N_ROWS,
                        subgroup_reduce,
                    )?;
                    Some((ggml_type, build_pipeline(source)))
                })
                .collect()
        } else {
            HashMap::new()
        };

        // See `Self::
        // wide_packed_pipeline`'s own doc comment. `packed_dot_f16`
        // already encodes the `supports_f16` gate the packed-`f16`
        // arithmetic needs, so gating on it here (rather than re-checking
        // `supports_f16` separately) is both correct and sufficient.
        let wide_packed_pipeline = (wide_load && packed_dot_f16)
            .then(|| build_pipeline(vulkan_shaders::shader_source_reduce_q4k_wide_packed_f16()));

        // See `Self::wide_unroll`'s own doc comment. One pipeline per K-quant
        // type the block-unroll covers (`Q4_K`/`Q5_K`/`Q6_K`), built eagerly
        // like every other pipeline.
        let wide_unroll_pipelines: HashMap<u32, wgpu::ComputePipeline> = if wide_unroll {
            SUPPORTED_TYPES
                .iter()
                .filter_map(|&ggml_type| {
                    let source = vulkan_shaders::shader_source_reduce_wide_unroll(
                        ggml_type,
                        REDUCE_N_ROWS,
                        subgroup_reduce,
                    )?;
                    Some((ggml_type, build_pipeline(source)))
                })
                .collect()
        } else {
            HashMap::new()
        };

        // See `Self::q4_k_unroll_packed_pipeline`'s own doc comment. Built
        // only when both the block-unroll (default) and the packed-`f16` dot
        // (`ORANGU_PACKED_DOT=1`, which already encodes the `supports_f16`
        // gate) are on.
        let q4_k_unroll_packed_pipeline = (wide_unroll && packed_dot_f16).then(|| {
            build_pipeline(
                vulkan_shaders::shader_source_reduce_q4k_wide_unroll_packed_f16(
                    REDUCE_N_ROWS,
                    subgroup_reduce,
                ),
            )
        });

        // See `Self::record_argmax_sample`'s own doc comment.
        let argmax_bind_group_layout = argmax_bind_group_layout(&device);
        let argmax_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("orangu-server argmax sample pipeline layout"),
                bind_group_layouts: &[Some(&argmax_bind_group_layout)],
                immediate_size: 0,
            });
        let argmax_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("orangu-server argmax sample shader"),
            source: wgpu::ShaderSource::Wgsl(vulkan_shaders::shader_source_argmax_sample().into()),
        });
        let argmax_sample_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("orangu-server argmax sample pipeline"),
                layout: Some(&argmax_pipeline_layout),
                module: &argmax_module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: pipeline_cache.as_ref(),
            });

        // Persist whatever the driver accumulated across every
        // `build_pipeline`/`build_elem_pipeline`/`attn_pipeline` call
        // above — every pipeline this backend will ever build happens
        // eagerly right here in `try_init`, none lazily later, so one save
        // at the end covers the whole cache. Best-effort: a failed write
        // only costs a slower cold start next time, never correctness, so
        // it's logged and swallowed rather than failing `try_init` itself.
        if let (Some(cache), Some(path)) = (&pipeline_cache, &pipeline_cache_path)
            && let Some(data) = cache.get_data()
            && let Err(e) = save_pipeline_cache(path, &data)
        {
            eprintln!(
                "orangu-server: warning: failed to save GPU pipeline cache to {}: {e}",
                path.display()
            );
        }

        Some(Self {
            device,
            queue,
            bind_group_layout,
            pipelines,
            pipelines_coop,
            pipelines_coop_tiled,
            weight_cache: Mutex::new(HashMap::new()),
            op_cache: Mutex::new(HashMap::new()),
            adapter_name,
            elem4_bind_group_layout,
            elem3_bind_group_layout,
            add_pipeline,
            mul_pipeline,
            gelu_pipeline,
            scale_pipeline,
            rmsnorm_pipeline,
            fused_cache: Mutex::new(HashMap::new()),
            attn_bind_group_layout,
            attn_pipeline,
            submission_count: std::sync::atomic::AtomicU64::new(0),
            rope_pipeline,
            perhead_rmsnorm_pipeline,
            elem2_bind_group_layout,
            perhead_rmsnorm_weightless_pipeline,
            fused_attn_layer_cache: Mutex::new(HashMap::new()),
            fused_layer_cache: Mutex::new(HashMap::new()),
            kv_f16,
            kv_cast_pipeline,
            packed_dot_f16,
            q4_k_packed_f16_pipeline,
            wide_load,
            wide_load_pipelines,
            wide_packed_pipeline,
            wide_unroll,
            wide_unroll_pipelines,
            q4_k_unroll_packed_pipeline,
            tiled_prefill,
            argmax_bind_group_layout,
            argmax_sample_pipeline,
            gpu_sample,
        })
    }

    /// Total `queue.submit` calls this backend has made so far — read
    /// before and after a decode step (`GemmaModel::forward` does this
    /// when `ORANGU_GPU_TRACE` is set) to count round trips per token
    /// directly.
    pub fn submission_count(&self) -> u64 {
        self.submission_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Returns this matrix's raw bytes already uploaded to the GPU,
    /// uploading (and caching) them first if this is the first `matmul`
    /// call to ever use this tensor.
    fn weight_buffer(&self, w: &QuantMatrix) -> Arc<wgpu::Buffer> {
        let (ptr, start) = w.cache_key();
        let bytes = w.raw_bytes();
        let key = (ptr, start, w.ggml_type(), bytes.len());
        let mut cache = self.weight_cache.lock().expect("weight cache poisoned");
        if let Some(buffer) = cache.get(&key) {
            return buffer.clone();
        }
        // Pad to a multiple of 16: `array<u32>`-bound kernels (the default
        // scalar reduce/coop pipelines) only ever needed a multiple of 4,
        // but the wide-load kernels bind this
        // same buffer as `array<vec4<u32>>` (16-byte elements) — WGPU
        // derives that array's runtime length from the *bound region's*
        // byte size divided by 16, rounded down, so a buffer padded only
        // to a multiple of 4 can leave the tensor's own real trailing
        // bytes outside that array's bounds whenever `row_bytes * out_dim`
        // isn't itself a multiple of 16 (`Q6_K`'s 210-byte blocks, and
        // `Q4_0`/`Q5_0`/`Q8_0` depending on `out_dim`, all routinely
        // aren't — only `Q4_K`/`Q5_K`'s 144-/176-byte blocks always are).
        // Padded unconditionally, not just when `Self::wide_load` is on:
        // this buffer is cached and shared across whichever kernel calls
        // into it later, and the extra padding (at most 15 bytes per
        // tensor) is immaterial next to real model tensor sizes. The
        // padding itself is never addressed by any kernel — every read
        // stays within the tensor's own real byte range — it only needs
        // to exist so the buffer's *allocated* size covers both binding
        // types' alignment requirements.
        let padded_len = (bytes.len() as u64).next_multiple_of(16);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("orangu-server weight"),
            size: padded_len,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buffer, 0, bytes);
        let buffer = Arc::new(buffer);
        cache.insert(key, buffer.clone());
        buffer
    }

    /// Splits `total` compute-shader workgroups across up to two dispatch
    /// dimensions — Vulkan caps a single dimension at 65535 workgroups,
    /// which a large `out_dim * n_tokens` (a long prompt's prefill against
    /// a wide vocabulary) can exceed.
    fn workgroup_dims(total: u32) -> (u32, u32, u32) {
        const MAX: u32 = 65535;
        if total <= MAX {
            (total.max(1), 1, 1)
        } else {
            let y = total.div_ceil(MAX);
            (MAX, y, 1)
        }
    }
}

impl Backend for VulkanBackend {
    fn matmul(&self, x: &[f32], n_tokens: usize, w: &QuantMatrix) -> Vec<f32> {
        let op = MatmulOp { x, n_tokens, w };
        self.matmul_batch(std::slice::from_ref(&op))
            .pop()
            .expect("matmul_batch returns exactly one result per input op")
    }

    /// Records every op's dispatch into one command encoder and submits
    /// them together, then blocks once (not once per op) for the whole
    /// batch to finish before reading every result back, so `ops.len()` GPU
    /// round-trips collapse into one. Each op's
    /// buffers/bind group are looked up (or, on first use, built) from
    /// `op_cache` rather than created fresh: after a
    /// tensor's first `matmul`/`matmul_batch` call at a given `n_tokens`,
    /// every later call at that same shape skips buffer/bind-group
    /// creation and only uploads the new activations.
    fn matmul_batch(&self, ops: &[MatmulOp<'_>]) -> Vec<Vec<f32>> {
        if ops.is_empty() {
            return Vec::new();
        }

        let pipelines: Vec<&wgpu::ComputePipeline> = ops
            .iter()
            .map(|op| self.pipeline_for(op.w.ggml_type(), op.n_tokens))
            .collect();

        // Two ops in the same batch can never legitimately share a cache
        // key: every real call site batches *independent* projections of
        // one input (e.g. a layer's Q/K/V), which are always different
        // weight tensors. If that ever changed, locking the same entry's
        // `Mutex` twice on this thread below would deadlock silently
        // instead of failing loudly — so check for it explicitly instead.
        let mut seen_keys = HashSet::with_capacity(ops.len());
        for op in ops {
            let (ptr, start) = op.w.cache_key();
            assert!(
                seen_keys.insert((ptr, start, op.n_tokens)),
                "matmul_batch called with the same (weight, n_tokens) op twice in one batch \
                 — would deadlock locking its cached resources twice"
            );
        }

        let entries: Vec<Arc<Mutex<CachedOpResources>>> =
            ops.iter().map(|op| self.op_entry(op)).collect();
        let guards: Vec<MutexGuard<'_, CachedOpResources>> = entries
            .iter()
            .map(|entry| entry.lock().expect("op cache entry poisoned"))
            .collect();

        for (op, guard) in ops.iter().zip(guards.iter()) {
            self.queue
                .write_buffer(&guard.x_buffer, 0, bytemuck::cast_slice(op.x));
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("orangu-server matmul batch encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("orangu-server matmul batch pass"),
                timestamp_writes: None,
            });
            for (pipeline, guard) in pipelines.iter().zip(guards.iter()) {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &guard.bind_group, &[]);
                let (wx, wy, wz) = guard.workgroups;
                pass.dispatch_workgroups(wx, wy, wz);
            }
        }
        for guard in &guards {
            encoder.copy_buffer_to_buffer(
                &guard.output_buffer,
                0,
                &guard.readback_buffer,
                0,
                guard.output_len,
            );
        }
        self.queue.submit(Some(encoder.finish()));
        self.submission_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Every readback buffer's `map_async` is fired before the single
        // `poll(Wait)` below — that one poll drains every callback bound to
        // this submission, not just one buffer's, which is what turns
        // `ops.len()` waits into one.
        for guard in &guards {
            guard
                .readback_buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, |result| {
                    result.expect("mapping a matmul batch readback buffer failed");
                });
        }
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("polling the device for the matmul batch readback failed");

        guards
            .iter()
            .map(|guard| {
                let slice = guard.readback_buffer.slice(..);
                let data = slice.get_mapped_range().expect(
                    "matmul batch readback buffer was not mapped after a successful \
                     map_async + poll",
                );
                let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
                drop(data);
                guard.readback_buffer.unmap();
                result
            })
            .collect()
    }

    fn as_vulkan(&self) -> Option<&VulkanBackend> {
        Some(self)
    }
}

impl VulkanBackend {
    /// Whether `n_tokens` both takes the cooperative-path (prefill) shape
    /// *and* `ORANGU_TILED_PREFILL=1` is set — the single decision point
    /// `pipeline_for` (which pipeline map to read from) and
    /// `build_op_resources` (how many workgroups that pipeline needs) both
    /// call, rather than each independently re-deriving it. Two independent
    /// copies of "which coop kernel is this" is exactly the drift that
    /// caused a real dispatch-count bug found while adding the packed-dot
    /// kernel — one call site's condition changed, the other's didn't.
    fn use_tiled_coop(&self, n_tokens: usize) -> bool {
        n_tokens >= COOP_MIN_N_TOKENS && self.tiled_prefill
    }

    /// Whether the memory-level-parallelism block-unroll kernel applies to
    /// this `(ggml_type, n_tokens)`: any K-quant (`Q4_K`/`Q5_K`/`Q6_K` — the
    /// types `wide_unroll_pipelines` covers) at a decode shape (`n_tokens <
    /// COOP_MIN_N_TOKENS`). Used by both `pipeline_for` (to select it) and
    /// `build_op_resources` (to keep the `REDUCE_N_ROWS`-batched dispatch
    /// count rather than the one-row-per-workgroup one), so the two can
    /// never disagree about which kernel is running.
    fn selects_wide_unroll(&self, ggml_type: u32, n_tokens: usize) -> bool {
        self.wide_unroll
            && n_tokens < COOP_MIN_N_TOKENS
            && self.wide_unroll_pipelines.contains_key(&ggml_type)
    }

    fn pipeline_for(&self, ggml_type: u32, n_tokens: usize) -> &wgpu::ComputePipeline {
        // Checked *before* every other decode special case below, so the
        // default-on memory-level-parallelism block-unroll (opt out
        // with `ORANGU_NO_MLP_UNROLL=1`) takes precedence over `wide_load`/
        // `packed_dot_f16` for the K-quant reduce path — it builds on the
        // wide-load binding and restructures the loop on top of it.
        // First its `Q4_K` packed-`f16` variant, when
        // `ORANGU_PACKED_DOT=1` is also on; then the scalar block-unroll for
        // whichever K-quant this is. Both are `REDUCE_N_ROWS`-batched, so
        // they need no dispatch-count special case in `build_op_resources`
        // (unlike the packed kernels' one-row-per-workgroup shape), and
        // `selects_wide_unroll` guards the one-row exclusion there for both.
        if self.wide_unroll
            && self.packed_dot_f16
            && ggml_type == crate::engine::quant::GGML_TYPE_Q4_K
            && n_tokens < COOP_MIN_N_TOKENS
            && let Some(pipeline) = &self.q4_k_unroll_packed_pipeline
        {
            return pipeline;
        }
        if self.selects_wide_unroll(ggml_type, n_tokens)
            && let Some(pipeline) = self.wide_unroll_pipelines.get(&ggml_type)
        {
            return pipeline;
        }
        // Checked *first*, so
        // `ORANGU_WIDE_LOAD=1 ORANGU_PACKED_DOT=1` together select the
        // combined kernel rather than either alone. `Q4_K`-only, like
        // the packed-dot kernel below; every other type/shape falls through
        // to the plain wide-load check below.
        if self.wide_load
            && self.packed_dot_f16
            && ggml_type == crate::engine::quant::GGML_TYPE_Q4_K
            && n_tokens < COOP_MIN_N_TOKENS
            && let Some(pipeline) = &self.wide_packed_pipeline
        {
            return pipeline;
        }
        // Checked *before* the
        // packed-`f16` dot below, so setting `ORANGU_WIDE_LOAD=1` alone
        // (without `ORANGU_PACKED_DOT=1`) is enough to select this kernel by
        // itself. Unlike the packed-dot kernel (`Q4_K`-only), this covers every type
        // `wide_load_pipelines` has an entry for — built for the whole
        // `SUPPORTED_TYPES` set when `wide_load` is on (see `Self::
        // try_init`).
        if self.wide_load
            && n_tokens < COOP_MIN_N_TOKENS
            && let Some(pipeline) = self.wide_load_pipelines.get(&ggml_type)
        {
            return pipeline;
        }
        // Only the
        // `Q4_K` reduce (decode, `n_tokens < COOP_MIN_N_TOKENS`) path has a
        // packed-`f16` variant; every other type/shape falls through to
        // the regular `pipelines`/`pipelines_coop`/`pipelines_coop_tiled`
        // lookup below unchanged.
        if self.packed_dot_f16
            && ggml_type == crate::engine::quant::GGML_TYPE_Q4_K
            && n_tokens < COOP_MIN_N_TOKENS
            && let Some(pipeline) = &self.q4_k_packed_f16_pipeline
        {
            return pipeline;
        }
        let map = if self.use_tiled_coop(n_tokens) {
            &self.pipelines_coop_tiled
        } else if n_tokens >= COOP_MIN_N_TOKENS {
            &self.pipelines_coop
        } else {
            &self.pipelines
        };
        // Load-time validation (`quant::dequantize`, exercised for every
        // tensor when the model loads) already rejects any `ggml_type`
        // this build can't dequantize at all, so every `QuantMatrix`
        // reaching here has a matching CPU dequantizer — this can only
        // mean a *GPU* shader gap for an otherwise supported type, which
        // doesn't exist today (see `SUPPORTED_TYPES`), but fail loudly
        // rather than silently returning zeros if that ever changes.
        map.get(&ggml_type).unwrap_or_else(|| {
            panic!(
                "VulkanBackend has no compute shader for ggml_type {ggml_type} — this is a \
                 bug, not a data problem (engine::quant already validated this tensor at load \
                 time)"
            )
        })
    }

    /// Returns the cached GPU resources for `op`'s `(weight, n_tokens)`
    /// shape, building (and caching) them first on a cache miss.
    fn op_entry(&self, op: &MatmulOp<'_>) -> Arc<Mutex<CachedOpResources>> {
        let (ptr, start) = op.w.cache_key();
        let key: OpCacheKey = (
            ptr,
            start,
            op.w.ggml_type(),
            op.w.in_dim,
            op.w.out_dim,
            op.w.row_bytes(),
            op.n_tokens,
        );
        {
            let cache = self.op_cache.lock().expect("op cache poisoned");
            if let Some(entry) = cache.get(&key) {
                return entry.clone();
            }
        }
        // Built outside the lock — buffer/bind-group creation doesn't need
        // exclusive access to the cache, only the final insert does. If
        // two threads race to build the same brand-new key, both build a
        // (functionally identical) copy and whichever inserts second just
        // has its own copy dropped; no correctness issue, only a rare,
        // one-time bit of redundant setup work.
        let resources = self.build_op_resources(op);
        let entry = Arc::new(Mutex::new(resources));
        let mut cache = self.op_cache.lock().expect("op cache poisoned");
        cache.entry(key).or_insert(entry).clone()
    }

    /// Builds one op's activation/output/readback/uniform buffers and the
    /// bind group tying them (plus the tensor's cached weight buffer)
    /// together — everything needed to dispatch against this `(weight,
    /// n_tokens)` shape, minus the activation data itself (written fresh
    /// by the caller on every reuse, since that's the one thing that
    /// actually changes call to call).
    fn build_op_resources(&self, op: &MatmulOp<'_>) -> CachedOpResources {
        let &MatmulOp { x, n_tokens, w } = op;
        let in_dim = w.in_dim;
        let out_dim = w.out_dim;
        debug_assert_eq!(x.len(), n_tokens * in_dim);

        let weight_buffer = self.weight_buffer(w);

        let x_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("orangu-server activations"),
            size: (x.len() as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let output_len = (n_tokens * out_dim) as u64 * 4;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("orangu-server matmul output"),
            size: output_len,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("orangu-server matmul readback"),
            size: output_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // `in_dim`/`out_dim`/`n_tokens`/`row_bytes` are all fixed for this
        // cache key's whole lifetime, so — unlike `x_buffer` — this never
        // needs writing again after this first (and only) time.
        let meta = Meta {
            in_dim: in_dim as u32,
            out_dim: out_dim as u32,
            n_tokens: n_tokens as u32,
            row_bytes: w.row_bytes() as u32,
        };
        let meta_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("orangu-server matmul meta"),
            size: std::mem::size_of::<Meta>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&meta_buffer, 0, bytemuck::bytes_of(&meta));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("orangu-server matmul bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: weight_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: meta_buffer.as_entire_binding(),
                },
            ],
        });

        // The default cooperative shader dispatches one workgroup *per
        // output row* (it internally loops over tiles of up to 64 tokens
        // each — see `vulkan_shaders::MAIN_COOP_SUFFIX`). Its opt-in tiled
        // alternative (`Self::use_tiled_coop`, `ORANGU_TILED_PREFILL=1`)
        // instead dispatches one
        // workgroup per `(row-tile, token-tile)` pair, using the exact same
        // `COOP_TILE_ROWS`/`COOP_TILE_TOKENS` constants the shader itself
        // is templated with, not hand-kept-in-sync duplicates (see those
        // constants' own doc comment for why: duplicating this exact kind
        // of "shader tiling assumption" as a separate literal here is what
        // caused a real dispatch-count bug). The reduction shader
        // dispatches one workgroup per `(REDUCE_N_ROWS-row group, token)`
        // pair (its 64 threads split each of those rows' elements and
        // reduce all of them together — see `vulkan_shaders::
        // MAIN_REDUCE_SUFFIX` and `REDUCE_N_ROWS`'s own doc comment),
        // `ceil(out_dim / REDUCE_N_ROWS) * n_tokens` groups, not `out_dim *
        // n_tokens` one-row-per-workgroup pairs — *except* the
        // packed-`f16` `Q4_K` kernel (`Self::pipeline_for`'s own special
        // case), which is **not** `REDUCE_N_ROWS`-batched (still one row
        // per workgroup), so it
        // needs `out_dim * n_tokens` workgroups — using the batched count
        // for it here would silently under-dispatch and leave `out_dim -
        // ceil(out_dim / REDUCE_N_ROWS)` rows' worth of output at whatever
        // the (zero-initialized) buffer already held, exactly the bug a
        // real cross-check test against `CpuBackend` caught. The
        // wide-load kernels (`Self::pipeline_for`'s other special case)
        // do *not* share this one-row-per-workgroup shape —
        // `shader_source_reduce_wide_load` reuses `MAIN_REDUCE_SUFFIX`
        // verbatim (see its own doc comment for why), so they dispatch
        // exactly like the regular reduce kernel and need no branch here.
        // The default-on block-unroll kernel (opt out with
        // `ORANGU_NO_MLP_UNROLL=1`) takes precedence over `packed_dot_f16`
        // in `pipeline_for`
        // for `Q4_K` decode and is `REDUCE_N_ROWS`-batched (not one row per
        // workgroup), so exclude that case here — otherwise setting both env
        // vars would under-dispatch the unroll kernel and leave most rows
        // unwritten, the exact class of bug the packed kernel's own
        // one-row-per-workgroup special case was added to avoid.
        let one_row_per_workgroup = self.packed_dot_f16
            && !self.selects_wide_unroll(w.ggml_type(), n_tokens)
            && w.ggml_type() == crate::engine::quant::GGML_TYPE_Q4_K
            && n_tokens < COOP_MIN_N_TOKENS;
        let total_workgroups = if self.use_tiled_coop(n_tokens) {
            let row_tiles = out_dim.div_ceil(vulkan_shaders::COOP_TILE_ROWS as usize);
            let token_tiles = n_tokens.div_ceil(vulkan_shaders::COOP_TILE_TOKENS as usize);
            (row_tiles * token_tiles) as u32
        } else if n_tokens >= COOP_MIN_N_TOKENS {
            out_dim as u32
        } else if one_row_per_workgroup {
            (out_dim * n_tokens) as u32
        } else {
            (out_dim.div_ceil(REDUCE_N_ROWS) * n_tokens) as u32
        };
        let workgroups = Self::workgroup_dims(total_workgroups);

        CachedOpResources {
            bind_group,
            x_buffer,
            output_buffer,
            readback_buffer,
            output_len,
            workgroups,
        }
    }
}

/// One optional per-layer-embedding (PLE) sub-chain within
/// [`FusedPostAttentionInput`] — see `engine::arch::gemma`'s module doc for
/// what PLE is. Fusable for the same reason the rest of the chain is: gate
/// -> GELU -> multiply by `per_layer_slice` -> proj -> RMSNorm -> residual
/// add has no CPU-only (attention) step anywhere in it.
pub struct FusedPle<'a> {
    pub gate_w: &'a QuantMatrix,
    pub proj_w: &'a QuantMatrix,
    pub post_norm: &'a [f32],
    /// This token/layer's slice of the precomputed per-layer-embedding
    /// input. `Cpu` for the CPU-orchestrated path (`GemmaModel::
    /// compute_per_layer_inputs`'s output, sliced per layer on the Rust
    /// side); `Gpu(buf, il * per_layer_dim)` for the decode full-forward-
    /// fusion path, where `buf` is `VulkanBackend::record_ple_projection`'s
    /// `[n_layer, per_layer_dim]` output and every layer reads its own
    /// slice out of the *same* buffer with no copy in between.
    pub per_layer_slice: GpuInput<'a>,
    /// `per_layer_slice`'s length — `GpuInput` doesn't carry one itself
    /// (unlike a plain `&[f32]`), so this has to travel alongside it.
    pub per_layer_dim: usize,
}

/// [`VulkanBackend::record_ple_projection`]'s parameters.
pub struct PleProjectionInput<'a> {
    /// The scaled token-embedding row(s) this forward pass started from
    /// (`GemmaModel::forward`'s `x`, before any layer touches it) —
    /// `Cpu` for every caller today (the embedding lookup has no GPU
    /// buffer of its own), but `GpuInput` rather than a plain slice for
    /// the same reason every other fused entry point takes one.
    pub x: GpuInput<'a>,
    /// `per_layer_model_proj` — `in_dim == n_embd`, `out_dim == n_layer *
    /// per_layer`.
    pub proj_w: &'a QuantMatrix,
    /// `per_layer_proj_norm`, `[per_layer]` — the *same* weight applied to
    /// every layer's row, mirroring `compute_per_layer_inputs`'s
    /// `tensor::rmsnorm_inplace(&mut proj, per_layer_proj_norm, n_tokens *
    /// n_layer, per_layer, eps)` call exactly.
    pub proj_norm: &'a [f32],
    /// `GemmaModel::gather_per_layer_tok_embd`'s output, `[n_layer *
    /// per_layer]` — already gathered and tok-embedding-scaled CPU-side.
    pub gathered: &'a [f32],
    pub n_layer: usize,
    pub per_layer: usize,
    pub eps: f32,
}

/// [`VulkanBackend::record_argmax_sample`]'s parameters.
pub struct GpuArgmaxSampleInput<'a> {
    /// `[n_vocab]` — typically the GPU-resident `lm_head` output
    /// `record_full_matmul` just produced (`GpuInput::Gpu`), so this
    /// whole kernel and the matmul that fed it stay in the same
    /// submission; `Cpu` also accepted, used by this module's own
    /// standalone cross-check test.
    pub logits: GpuInput<'a>,
    pub n_vocab: usize,
    /// Already trimmed to the sampler's `repeat_last_n` window by the
    /// caller (`engine::arch::GreedySampleParams`'s own doc comment) —
    /// applied to `logits` in exactly this order, matching `engine::
    /// sampling::apply_repeat_penalty`'s compounding-on-repeats behavior.
    pub recent_tokens: &'a [u32],
    pub repeat_penalty: f32,
}

/// Either CPU data (uploaded via `queue.write_buffer`) or an
/// already-GPU-resident buffer (copied via `copy_buffer_to_buffer`) — lets
/// the standalone `fused_attention`/`fused_post_attention` entry points
/// (used by their own cross-check tests — `GemmaModel::forward` calls
/// `fused_layer`, which uses `Gpu` internally to chain them with no
/// readback in between) keep taking plain CPU slices via `Cpu`. `Copy`
/// since both variants are themselves just references.
///
/// `Gpu`'s second field is an element offset into the buffer — `0` for
/// every use before the PLE-fusion path was added, which needs
/// it to let each layer's `FusedPle` read its own `per_layer`-wide slice
/// out of one shared `[n_layer, per_layer]` buffer (`VulkanBackend::
/// record_ple_projection`'s output) without a separate copy per layer.
#[derive(Clone, Copy)]
pub enum GpuInput<'a> {
    #[allow(dead_code)]
    Cpu(&'a [f32]),
    Gpu(&'a wgpu::Buffer, usize),
}

/// Everything [`VulkanBackend::fused_post_attention`] needs: a gemma4
/// layer's `wo` through the *next* layer's normed input has no CPU-only
/// step in it anywhere (attention itself, the one CPU-only step in a
/// layer, has already happened by the time `attn_out` is known) — so the
/// whole chain runs as one GPU submission instead of the 3-5 separate
/// `matmul`/`matmul_batch` round trips (`wo`, `gate`/`up`, `down`, and
/// PLE's `gate`/`proj` when present) the step-by-step CPU-orchestrated path
/// pays.
pub struct FusedPostAttentionInput<'a> {
    /// Attention's output, `[n_embd]`.
    pub attn_out: GpuInput<'a>,
    /// `x` from before this sub-layer's residual adds.
    pub residual: GpuInput<'a>,
    pub wo: &'a QuantMatrix,
    pub attn_post_norm: &'a [f32],
    pub ffn_norm: &'a [f32],
    pub ffn_gate: &'a QuantMatrix,
    pub ffn_up: &'a QuantMatrix,
    pub ffn_down: &'a QuantMatrix,
    pub ffn_post_norm: &'a [f32],
    pub eps: f32,
    pub ple: Option<FusedPle<'a>>,
    pub layer_output_scale: Option<f32>,
}

/// [`VulkanBackend::gpu_attention`]'s parameters, grouped into one struct
/// rather than an 8-argument function signature.
///
/// `gpu_attention` itself, and the standalone GPU RoPE/per-head-norm
/// primitives near it (`gpu_rope`, `gpu_perhead_rmsnorm`,
/// `gpu_perhead_rmsnorm_weightless`), are no longer called from
/// `GemmaModel::forward`'s hot path — `fused_attention` supersedes all
/// four by chaining the same work into one submission instead of running each as its own
/// upload/dispatch/readback. They're kept, `#[allow(dead_code)]`ed
/// rather than deleted, because each still has its own cross-check test
/// against the CPU reference — isolated coverage that narrows down
/// *which* piece broke if `fused_attention` ever regresses, and doubles
/// as living documentation of each primitive's exact semantics.
#[allow(dead_code)]
pub struct GpuAttentionInput<'a> {
    /// `[n_head, head_dim]`, already Q-normed and RoPE'd.
    pub q: &'a [f32],
    /// This token's `k`/`v` must already be pushed here before calling —
    /// same ordering the CPU attention path requires.
    pub cache: &'a mut crate::engine::kv_cache::LayerCache,
    pub pos: usize,
    pub window_start: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub scale: f32,
}

/// [`VulkanBackend::gpu_rope`]'s parameters — see [`GpuAttentionInput`]'s
/// doc comment for why this (and `gpu_rope` itself) is `#[allow(dead_code)]`.
#[allow(dead_code)]
pub struct GpuRopeInput<'a> {
    pub x: &'a [f32],
    pub n_head: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub pos: usize,
    pub freq_base: f32,
    pub freq_factors: Option<&'a [f32]>,
}

/// This layer's own K/V projection, when it has one (`layer.has_kv`) —
/// see [`FusedAttnInput::kv`].
pub struct FusedAttnProjection<'a> {
    pub wk: &'a QuantMatrix,
    /// `[head_dim]` — the same vector for every KV head.
    pub k_norm: &'a [f32],
    /// `None` when this layer doesn't own its own V projection (V is a
    /// copy of K's post-norm output instead — see `fused_attention`'s
    /// doc comment for why that's still correct here).
    pub wv: Option<&'a QuantMatrix>,
}

/// [`VulkanBackend::fused_attention`]'s parameters.
pub struct FusedAttnInput<'a> {
    /// `attn_norm`'s output, `[n_embd]`.
    pub normed: GpuInput<'a>,
    pub wq: &'a QuantMatrix,
    /// `[head_dim]` — the same vector for every Q head.
    pub q_norm: &'a [f32],
    /// `Some` iff this layer owns a KV cache of its own (`layer.has_kv`);
    /// `None` for gemma4's cross-layer KV-donor layers, which skip the
    /// whole K/V projection/norm/RoPE/write sub-chain and read straight
    /// from `cache` (already fully up to date from an earlier layer in
    /// the same forward pass).
    pub kv: Option<FusedAttnProjection<'a>>,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub rope_freq_base: f32,
    /// Gemma4's proportional-RoPE divisor, full-attention layers only —
    /// `None` for SWA layers, matching the CPU path exactly.
    pub freq_factors: Option<&'a [f32]>,
    pub eps: f32,
    pub pos: usize,
    pub window_start: usize,
    pub scale: f32,
    pub cache: &'a mut crate::engine::kv_cache::LayerCache,
}

/// [`VulkanBackend::fused_layer`]'s parameters — the union of
/// [`FusedAttnInput`] and [`FusedPostAttentionInput`], minus the three
/// fields (`normed`, `attn_out`, `residual`) `fused_layer` computes/
/// threads through internally as GPU buffers, plus `x` (the residual
/// stream this layer starts from) and `attn_norm` (the pre-attention norm
/// weight, the one piece neither of the two chains it wraps already
/// covered).
pub struct FusedLayerInput<'a> {
    /// This layer's residual-stream input, `[n_embd]` — `Cpu` for the
    /// first layer of a forward pass (the embedding row), `Gpu` for every
    /// later layer when [`VulkanBackend::record_fused_layer`] chains one
    /// layer's output straight into the next's input with no CPU round
    /// trip.
    pub x: GpuInput<'a>,
    pub attn_norm: &'a [f32],
    pub wq: &'a QuantMatrix,
    pub q_norm: &'a [f32],
    pub kv: Option<FusedAttnProjection<'a>>,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub rope_freq_base: f32,
    pub freq_factors: Option<&'a [f32]>,
    pub eps: f32,
    pub pos: usize,
    pub window_start: usize,
    pub scale: f32,
    pub cache: &'a mut crate::engine::kv_cache::LayerCache,
    pub wo: &'a QuantMatrix,
    pub attn_post_norm: &'a [f32],
    pub ffn_norm: &'a [f32],
    pub ffn_gate: &'a QuantMatrix,
    pub ffn_up: &'a QuantMatrix,
    pub ffn_down: &'a QuantMatrix,
    pub ffn_post_norm: &'a [f32],
    pub ple: Option<FusedPle<'a>>,
    pub layer_output_scale: Option<f32>,
}

impl VulkanBackend {
    fn scratch_buffer(&self, len_f32: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("orangu-server fused scratch"),
            size: (len_f32 as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn upload_new(&self, data: &[f32]) -> wgpu::Buffer {
        let buffer = self.scratch_buffer(data.len());
        self.queue
            .write_buffer(&buffer, 0, bytemuck::cast_slice(data));
        buffer
    }

    /// Like `upload_new`, but for `u32` data — `Self::record_argmax_
    /// sample`'s `recent_tokens` buffer. `scratch_buffer`'s sizing is
    /// element-count-based assuming 4-byte elements, true of `u32` just as
    /// much as `f32`, so it's reused as-is rather than duplicated.
    fn upload_new_u32(&self, data: &[u32]) -> wgpu::Buffer {
        let buffer = self.scratch_buffer(data.len());
        self.queue
            .write_buffer(&buffer, 0, bytemuck::cast_slice(data));
        buffer
    }

    fn elem_meta_buffer(&self, len: u32, extra: f32) -> wgpu::Buffer {
        let meta = ElemMeta {
            len,
            _pad0: 0,
            extra,
            _pad1: 0,
        };
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("orangu-server elem meta"),
            size: std::mem::size_of::<ElemMeta>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&buffer, 0, bytemuck::bytes_of(&meta));
        buffer
    }

    /// `vulkan_shaders::shader_source_kv_cast`'s `CastMeta` — byte-for-byte
    /// the same layout as `ElemMeta` (see that shader's own doc comment
    /// for why this reuses `ElemMeta`'s Rust struct too, just naming its
    /// second field `offset` here instead of `_pad0`).
    fn cast_meta_buffer(&self, len: u32, offset: u32) -> wgpu::Buffer {
        let meta = ElemMeta {
            len,
            _pad0: offset,
            extra: 0.0,
            _pad1: 0,
        };
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("orangu-server kv cast meta"),
            size: std::mem::size_of::<ElemMeta>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&buffer, 0, bytemuck::bytes_of(&meta));
        buffer
    }

    fn elem4_bind_group(
        &self,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        y: &wgpu::Buffer,
        meta: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("orangu-server elem4 bind group"),
            layout: &self.elem4_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: y.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: meta.as_entire_binding(),
                },
            ],
        })
    }

    fn elem3_bind_group(
        &self,
        x: &wgpu::Buffer,
        y: &wgpu::Buffer,
        meta: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("orangu-server elem3 bind group"),
            layout: &self.elem3_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: y.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: meta.as_entire_binding(),
                },
            ],
        })
    }

    fn elem2_bind_group(&self, x: &wgpu::Buffer, meta: &wgpu::Buffer) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("orangu-server elem2 bind group"),
            layout: &self.elem2_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: meta.as_entire_binding(),
                },
            ],
        })
    }

    /// Finishes `encoder`, submits it, copies `src`'s first `len_f32`
    /// elements into a fresh readback buffer, and blocks until they're
    /// available — the same submit/poll/map/read sequence every GPU
    /// method here ends with, factored out once it started repeating a
    /// fourth time (`gpu_rope`/`gpu_perhead_rmsnorm`/`gpu_perhead_rmsnorm_
    /// weightless` — all standalone, cross-check-test-only entry points;
    /// `matmul_batch`/`fused_post_attention`/`gpu_attention` predate this
    /// and aren't worth the churn of switching over). `pub` since
    /// `GemmaModel::forward`
    /// also uses this directly to finish its own hand-assembled
    /// full-forward-pass encoder.
    pub fn submit_and_readback(
        &self,
        mut encoder: wgpu::CommandEncoder,
        src: &wgpu::Buffer,
        len_f32: usize,
    ) -> Vec<f32> {
        let byte_len = (len_f32 as u64) * 4;
        let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("orangu-server generic readback"),
            size: byte_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(src, 0, &readback_buffer, 0, byte_len);
        self.queue.submit(Some(encoder.finish()));
        self.submission_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        readback_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, |result| {
                result.expect("mapping a generic readback buffer failed");
            });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("polling for a generic readback failed");
        let data = readback_buffer
            .slice(..)
            .get_mapped_range()
            .expect("generic readback buffer was not mapped after a successful map_async + poll");
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        readback_buffer.unmap();
        result
    }

    /// Like `submit_and_readback`, but for a single `u32` — `Self::
    /// record_argmax_sample`'s output buffer: this is the whole point of
    /// that kernel, reading back 4 bytes instead of the `[n_vocab]` logits
    /// vector `submit_and_readback` would otherwise need.
    pub fn submit_and_readback_u32(
        &self,
        mut encoder: wgpu::CommandEncoder,
        src: &wgpu::Buffer,
    ) -> u32 {
        let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("orangu-server u32 readback"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(src, 0, &readback_buffer, 0, 4);
        self.queue.submit(Some(encoder.finish()));
        self.submission_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        readback_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, |result| {
                result.expect("mapping a u32 readback buffer failed");
            });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("polling for a u32 readback failed");
        let data = readback_buffer
            .slice(..)
            .get_mapped_range()
            .expect("u32 readback buffer was not mapped after a successful map_async + poll");
        let result: u32 = bytemuck::cast_slice::<u8, u32>(&data)[0];
        drop(data);
        readback_buffer.unmap();
        result
    }

    /// Puts `src` into `dst`: a `queue.write_buffer` for CPU data, or a
    /// `copy_buffer_to_buffer` recorded into `encoder` for an
    /// already-GPU-resident buffer — the one seam `fused_layer` needs to
    /// feed one fused chain's output straight into the next with no
    /// readback, while the standalone `fused_attention`/
    /// `fused_post_attention` entry points keep working unchanged with
    /// plain CPU slices.
    fn upload_or_copy(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        dst: &wgpu::Buffer,
        src: GpuInput<'_>,
        len_f32: usize,
    ) {
        match src {
            GpuInput::Cpu(data) => {
                debug_assert_eq!(data.len(), len_f32);
                self.queue.write_buffer(dst, 0, bytemuck::cast_slice(data));
            }
            GpuInput::Gpu(buf, offset) => {
                encoder.copy_buffer_to_buffer(
                    buf,
                    (offset as u64) * 4,
                    dst,
                    0,
                    (len_f32 as u64) * 4,
                );
            }
        }
    }

    /// GPU RoPE (`vulkan_shaders::shader_source_rope`) — cross-checked
    /// against `tensor::rope_apply_scaled_inplace`. `x` is `[n_head,
    /// head_dim]`; `freq_factors`, when given, is `[rope_dim/2]` (Gemma4's
    /// proportional-RoPE divisor for full-attention layers only — SWA
    /// layers pass `None`, matching the CPU path exactly).
    #[allow(dead_code)]
    pub fn gpu_rope(&self, input: GpuRopeInput<'_>) -> Vec<f32> {
        let GpuRopeInput {
            x,
            n_head,
            head_dim,
            rope_dim,
            pos,
            freq_base,
            freq_factors,
        } = input;
        debug_assert_eq!(x.len(), n_head * head_dim);
        let half = rope_dim / 2;
        let ff_owned;
        let ff: &[f32] = match freq_factors {
            Some(ff) => ff,
            None => {
                ff_owned = vec![1.0f32; half];
                &ff_owned
            }
        };
        debug_assert_eq!(ff.len(), half);

        let x_buf = self.upload_new(x);
        let ff_buf = self.upload_new(ff);
        let meta = RopeMeta {
            n_head: n_head as u32,
            head_dim: head_dim as u32,
            rope_dim: rope_dim as u32,
            pos: pos as u32,
            freq_base,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let meta_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("orangu-server rope meta"),
            size: std::mem::size_of::<RopeMeta>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&meta_buf, 0, bytemuck::bytes_of(&meta));
        let bind_group = self.elem3_bind_group(&ff_buf, &x_buf, &meta_buf);

        let total = (n_head * half) as u32;
        let wg = total.div_ceil(64).max(1);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("orangu-server rope encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("orangu-server rope pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.rope_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(wg, 1, 1);
        }
        self.submit_and_readback(encoder, &x_buf, x.len())
    }

    /// GPU per-head weighted RMSNorm (Q-norm/K-norm) — cross-checked
    /// against `tensor::rmsnorm_inplace(x, weight, n_head, head_dim,
    /// eps)`. `weight` (`[head_dim]`) is shared across every head.
    #[allow(dead_code)]
    pub fn gpu_perhead_rmsnorm(
        &self,
        x: &[f32],
        weight: &[f32],
        n_head: usize,
        head_dim: usize,
        eps: f32,
    ) -> Vec<f32> {
        debug_assert_eq!(x.len(), n_head * head_dim);
        debug_assert_eq!(weight.len(), head_dim);

        let x_buf = self.upload_new(x);
        let w_buf = self.upload_new(weight);
        let meta = PerHeadNormMeta {
            n_head: n_head as u32,
            head_dim: head_dim as u32,
            eps,
            _pad: 0,
        };
        let meta_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("orangu-server perhead norm meta"),
            size: std::mem::size_of::<PerHeadNormMeta>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&meta_buf, 0, bytemuck::bytes_of(&meta));
        let bind_group = self.elem3_bind_group(&w_buf, &x_buf, &meta_buf);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("orangu-server perhead norm encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("orangu-server perhead norm pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.perhead_rmsnorm_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(n_head as u32, 1, 1);
        }
        self.submit_and_readback(encoder, &x_buf, x.len())
    }

    /// GPU per-head weightless RMSNorm (V's norm) — cross-checked against
    /// `rmsnorm_weightless_inplace(x, n_head, head_dim, eps)`.
    #[allow(dead_code)]
    pub fn gpu_perhead_rmsnorm_weightless(
        &self,
        x: &[f32],
        n_head: usize,
        head_dim: usize,
        eps: f32,
    ) -> Vec<f32> {
        debug_assert_eq!(x.len(), n_head * head_dim);

        let x_buf = self.upload_new(x);
        let meta = PerHeadNormMeta {
            n_head: n_head as u32,
            head_dim: head_dim as u32,
            eps,
            _pad: 0,
        };
        let meta_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("orangu-server perhead weightless norm meta"),
            size: std::mem::size_of::<PerHeadNormMeta>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&meta_buf, 0, bytemuck::bytes_of(&meta));
        let bind_group = self.elem2_bind_group(&x_buf, &meta_buf);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("orangu-server perhead weightless norm encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("orangu-server perhead weightless norm pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.perhead_rmsnorm_weightless_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(n_head as u32, 1, 1);
        }
        self.submit_and_readback(encoder, &x_buf, x.len())
    }

    fn perhead_norm_meta_buffer(&self, n_head: u32, head_dim: u32, eps: f32) -> wgpu::Buffer {
        let meta = PerHeadNormMeta {
            n_head,
            head_dim,
            eps,
            _pad: 0,
        };
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("orangu-server perhead norm meta"),
            size: std::mem::size_of::<PerHeadNormMeta>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buf, 0, bytemuck::bytes_of(&meta));
        buf
    }

    fn rope_meta_buffer(
        &self,
        n_head: u32,
        head_dim: u32,
        rope_dim: u32,
        pos: u32,
        freq_base: f32,
    ) -> wgpu::Buffer {
        let meta = RopeMeta {
            n_head,
            head_dim,
            rope_dim,
            pos,
            freq_base,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("orangu-server rope meta (fused)"),
            size: std::mem::size_of::<RopeMeta>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buf, 0, bytemuck::bytes_of(&meta));
        buf
    }

    /// One matmul's op-cache entry, primed with a correctly-sized (but
    /// content-irrelevant — this call only ever uses the entry's buffers as
    /// a GPU-GPU copy target, never `op.x`'s CPU contents) dummy operand.
    /// Only used by [`Self::fused_post_attention`], where every matmul
    /// input is itself GPU-resident data produced earlier in the same
    /// chain, not CPU data `matmul_batch`'s normal callers would pass.
    fn op_entry_for(&self, w: &QuantMatrix) -> Arc<Mutex<CachedOpResources>> {
        let dummy = vec![0f32; w.in_dim];
        self.op_entry(&MatmulOp {
            x: &dummy,
            n_tokens: 1,
            w,
        })
    }

    /// Records `w`'s matmul dispatch (`x -> y`, `n_tokens == 1`) into
    /// `pass`, using `entry`'s already-cached bind group/workgroup count.
    fn record_matmul<'p>(
        &'p self,
        pass: &mut wgpu::ComputePass<'p>,
        w: &QuantMatrix,
        entry: &'p CachedOpResources,
    ) {
        pass.set_pipeline(self.pipeline_for(w.ggml_type(), 1));
        pass.set_bind_group(0, &entry.bind_group, &[]);
        let (wx, wy, wz) = entry.workgroups;
        pass.dispatch_workgroups(wx, wy, wz);
    }

    /// PLE's own gate -> GELU -> multiply -> proj -> RMSNorm -> residual-add
    /// sub-chain's cached buffers/bind groups — see [`FusedResources`].
    /// `per_layer_buf` is the one field here whose *contents* change every
    /// call (like `FusedResources::residual_buf`); everything else is
    /// fixed once this layer's shapes are known.
    fn build_ple_resources(
        &self,
        ple: &FusedPle<'_>,
        n_embd: usize,
        meta_embd_eps: &wgpu::Buffer,
        x2: &wgpu::Buffer,
        ple_gate_g: &CachedOpResources,
        ple_proj_g: &CachedOpResources,
    ) -> FusedPleResources {
        let per_layer_dim = ple.per_layer_dim;
        debug_assert_eq!(ple.gate_w.out_dim, per_layer_dim);
        debug_assert_eq!(ple.proj_w.in_dim, per_layer_dim);
        debug_assert_eq!(ple.proj_w.out_dim, n_embd);

        let post_norm_w = self.upload_new(ple.post_norm);
        let per_layer_buf = self.scratch_buffer(per_layer_dim);
        let gelu_out = self.scratch_buffer(per_layer_dim);
        let mulled = self.scratch_buffer(per_layer_dim);
        let normed = self.scratch_buffer(n_embd);
        let x3 = self.scratch_buffer(n_embd);
        let meta_plain = self.elem_meta_buffer(per_layer_dim as u32, 0.0);
        let wg = (per_layer_dim as u32).div_ceil(64);

        let bg_gelu = self.elem3_bind_group(&ple_gate_g.output_buffer, &gelu_out, &meta_plain);
        let bg_mul = self.elem4_bind_group(&gelu_out, &per_layer_buf, &mulled, &meta_plain);
        let bg_post_norm = self.elem4_bind_group(
            &ple_proj_g.output_buffer,
            &post_norm_w,
            &normed,
            meta_embd_eps,
        );
        let bg_add = self.elem4_bind_group(&normed, x2, &x3, meta_embd_eps);

        FusedPleResources {
            per_layer_buf,
            gelu_out,
            mulled,
            normed,
            x3,
            bg_gelu,
            bg_mul,
            bg_post_norm,
            bg_add,
            wg,
        }
    }

    /// Builds (never cached itself — callers cache the whole
    /// [`FusedResources`] this returns) every buffer/bind group
    /// `fused_post_attention` needs beyond what `op_cache` already
    /// provides for the matmul steps.
    fn build_fused_resources(
        &self,
        input: &FusedPostAttentionInput<'_>,
        wo_g: &CachedOpResources,
        gate_g: &CachedOpResources,
        up_g: &CachedOpResources,
        down_g: &CachedOpResources,
        ple_g: Option<(&CachedOpResources, &CachedOpResources)>,
    ) -> FusedResources {
        let n_embd = input.wo.out_dim;
        let ffn_len = input.ffn_gate.out_dim;

        let residual_buf = self.scratch_buffer(n_embd);
        let attn_post_norm_w = self.upload_new(input.attn_post_norm);
        let ffn_norm_w = self.upload_new(input.ffn_norm);
        let ffn_post_norm_w = self.upload_new(input.ffn_post_norm);

        let normed1 = self.scratch_buffer(n_embd);
        let x1 = self.scratch_buffer(n_embd);
        let ffn_normed = self.scratch_buffer(n_embd);
        let gelu_out = self.scratch_buffer(ffn_len);
        let mulled = self.scratch_buffer(ffn_len);
        let normed2 = self.scratch_buffer(n_embd);
        let x2 = self.scratch_buffer(n_embd);

        let meta_embd_eps = self.elem_meta_buffer(n_embd as u32, input.eps);
        let meta_embd_plain = self.elem_meta_buffer(n_embd as u32, 0.0);
        let meta_ffn_plain = self.elem_meta_buffer(ffn_len as u32, 0.0);

        let bg_attn_post_norm = self.elem4_bind_group(
            &wo_g.output_buffer,
            &attn_post_norm_w,
            &normed1,
            &meta_embd_eps,
        );
        let bg_add1 = self.elem4_bind_group(&normed1, &residual_buf, &x1, &meta_embd_plain);
        let bg_ffn_norm = self.elem4_bind_group(&x1, &ffn_norm_w, &ffn_normed, &meta_embd_eps);
        let bg_gelu = self.elem3_bind_group(&gate_g.output_buffer, &gelu_out, &meta_ffn_plain);
        let bg_mul =
            self.elem4_bind_group(&gelu_out, &up_g.output_buffer, &mulled, &meta_ffn_plain);
        let bg_ffn_post_norm = self.elem4_bind_group(
            &down_g.output_buffer,
            &ffn_post_norm_w,
            &normed2,
            &meta_embd_eps,
        );
        let bg_add2 = self.elem4_bind_group(&normed2, &x1, &x2, &meta_embd_plain);

        let ple = if let (Some(ple), Some((ple_gate_g, ple_proj_g))) = (&input.ple, ple_g) {
            Some(self.build_ple_resources(ple, n_embd, &meta_embd_eps, &x2, ple_gate_g, ple_proj_g))
        } else {
            None
        };

        let scale = if let Some(s) = input.layer_output_scale {
            let scaled = self.scratch_buffer(n_embd);
            let meta_buf = self.elem_meta_buffer(n_embd as u32, s);
            let source: &wgpu::Buffer = ple.as_ref().map_or(&x2, |p| &p.x3);
            let bg = self.elem3_bind_group(source, &scaled, &meta_buf);
            Some(FusedScaleResources { scaled, bg })
        } else {
            None
        };

        FusedResources {
            residual_buf,
            normed1,
            x1,
            ffn_normed,
            gelu_out,
            mulled,
            normed2,
            x2,
            bg_attn_post_norm,
            bg_add1,
            bg_ffn_norm,
            bg_gelu,
            bg_mul,
            bg_ffn_post_norm,
            bg_add2,
            ple,
            scale,
            embd_wg: (n_embd as u32).div_ceil(64),
            ffn_wg: (ffn_len as u32).div_ceil(64),
        }
    }

    /// Returns this layer's cached [`FusedResources`], building (and
    /// caching) them first on a cache miss — the same reuse-after-first-
    /// call shape `op_entry`/`weight_buffer` already follow.
    fn fused_entry_for(
        &self,
        input: &FusedPostAttentionInput<'_>,
        wo_g: &CachedOpResources,
        gate_g: &CachedOpResources,
        up_g: &CachedOpResources,
        down_g: &CachedOpResources,
        ple_g: Option<(&CachedOpResources, &CachedOpResources)>,
    ) -> Arc<FusedResources> {
        let (ptr, start) = input.wo.cache_key();
        let per_layer_dim = input.ple.as_ref().map_or(0, |p| p.per_layer_dim);
        let key: FusedCacheKey = (
            ptr,
            start,
            input.ffn_gate.out_dim,
            per_layer_dim,
            input.layer_output_scale.is_some(),
        );
        {
            let cache = self.fused_cache.lock().expect("fused cache poisoned");
            if let Some(entry) = cache.get(&key) {
                return entry.clone();
            }
        }
        let resources =
            Arc::new(self.build_fused_resources(input, wo_g, gate_g, up_g, down_g, ple_g));
        let mut cache = self.fused_cache.lock().expect("fused cache poisoned");
        cache.entry(key).or_insert(resources).clone()
    }

    /// Records `wo` through this layer's `layer_output_scale` into
    /// `encoder` (does **not** submit) and returns the GPU buffer holding
    /// the layer's final `[n_embd]` result — the recording half of
    /// [`Self::fused_post_attention`], split out so [`Self::fused_layer`]
    /// can chain it after
    /// [`Self::record_fused_attention`] in one submission instead of two.
    fn record_fused_post_attention(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: FusedPostAttentionInput<'_>,
    ) -> wgpu::Buffer {
        let n_embd = input.wo.out_dim;
        let ffn_len = input.ffn_gate.out_dim;
        debug_assert_eq!(input.ffn_up.out_dim, ffn_len);
        debug_assert_eq!(input.ffn_down.in_dim, ffn_len);
        debug_assert_eq!(input.ffn_down.out_dim, n_embd);

        let wo_entry = self.op_entry_for(input.wo);
        let gate_entry = self.op_entry_for(input.ffn_gate);
        let up_entry = self.op_entry_for(input.ffn_up);
        let down_entry = self.op_entry_for(input.ffn_down);
        let ple_entries = input
            .ple
            .as_ref()
            .map(|ple| (self.op_entry_for(ple.gate_w), self.op_entry_for(ple.proj_w)));

        let wo_g = wo_entry.lock().expect("op cache entry poisoned");
        let gate_g = gate_entry.lock().expect("op cache entry poisoned");
        let up_g = up_entry.lock().expect("op cache entry poisoned");
        let down_g = down_entry.lock().expect("op cache entry poisoned");
        let ple_g = ple_entries.as_ref().map(|(g, p)| {
            (
                g.lock().expect("op cache entry poisoned"),
                p.lock().expect("op cache entry poisoned"),
            )
        });
        let ple_g_refs = ple_g.as_ref().map(|(g, p)| (&**g, &**p));

        let res = self.fused_entry_for(&input, &wo_g, &gate_g, &up_g, &down_g, ple_g_refs);

        // The three genuinely-per-call inputs — everything else in `res`
        // was uploaded once when this layer's entry was first built.
        // `attn_out` is `wo.in_dim`-sized (`n_head * head_dim`, attention's
        // own output width) — *not* `n_embd`, which only coincides with it
        // when a model happens to pick `head_dim` so `n_head * head_dim ==
        // n_embd` (true of the small synthetic shapes every test here used
        // until this was caught, false of the real `E2B` model: `n_head=8`,
        // `head_dim=512` makes `wo.in_dim = 4096`, `n_embd = 1536`). Using
        // `n_embd` here copied too many bytes out of a too-small source
        // buffer — silently corrupting `wo`'s input immediately after the
        // very first attention call once the two dims genuinely differ.
        self.upload_or_copy(encoder, &wo_g.x_buffer, input.attn_out, input.wo.in_dim);
        self.upload_or_copy(encoder, &res.residual_buf, input.residual, n_embd);
        if let (Some(ple), Some(pres)) = (&input.ple, &res.ple) {
            self.upload_or_copy(
                encoder,
                &pres.per_layer_buf,
                ple.per_layer_slice,
                ple.per_layer_dim,
            );
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("orangu-server fused wo+norms pass"),
                timestamp_writes: None,
            });
            self.record_matmul(&mut pass, input.wo, &wo_g);

            pass.set_pipeline(&self.rmsnorm_pipeline);
            pass.set_bind_group(0, &res.bg_attn_post_norm, &[]);
            pass.dispatch_workgroups(1, 1, 1);

            pass.set_pipeline(&self.add_pipeline);
            pass.set_bind_group(0, &res.bg_add1, &[]);
            pass.dispatch_workgroups(res.embd_wg, 1, 1);

            pass.set_pipeline(&self.rmsnorm_pipeline);
            pass.set_bind_group(0, &res.bg_ffn_norm, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&res.ffn_normed, 0, &gate_g.x_buffer, 0, (n_embd as u64) * 4);
        encoder.copy_buffer_to_buffer(&res.ffn_normed, 0, &up_g.x_buffer, 0, (n_embd as u64) * 4);

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("orangu-server fused ffn pass"),
                timestamp_writes: None,
            });
            self.record_matmul(&mut pass, input.ffn_gate, &gate_g);
            self.record_matmul(&mut pass, input.ffn_up, &up_g);

            pass.set_pipeline(&self.gelu_pipeline);
            pass.set_bind_group(0, &res.bg_gelu, &[]);
            pass.dispatch_workgroups(res.ffn_wg, 1, 1);

            pass.set_pipeline(&self.mul_pipeline);
            pass.set_bind_group(0, &res.bg_mul, &[]);
            pass.dispatch_workgroups(res.ffn_wg, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&res.mulled, 0, &down_g.x_buffer, 0, (ffn_len as u64) * 4);

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("orangu-server fused down+norms pass"),
                timestamp_writes: None,
            });
            self.record_matmul(&mut pass, input.ffn_down, &down_g);

            pass.set_pipeline(&self.rmsnorm_pipeline);
            pass.set_bind_group(0, &res.bg_ffn_post_norm, &[]);
            pass.dispatch_workgroups(1, 1, 1);

            pass.set_pipeline(&self.add_pipeline);
            pass.set_bind_group(0, &res.bg_add2, &[]);
            pass.dispatch_workgroups(res.embd_wg, 1, 1);
        }

        let mut final_buf = &res.x2;
        if let (Some(ple_input), Some(pres), Some((ple_gate_g, ple_proj_g))) =
            (&input.ple, &res.ple, &ple_g)
        {
            encoder.copy_buffer_to_buffer(
                final_buf,
                0,
                &ple_gate_g.x_buffer,
                0,
                (n_embd as u64) * 4,
            );

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("orangu-server fused ple gate pass"),
                    timestamp_writes: None,
                });
                self.record_matmul(&mut pass, ple_input.gate_w, ple_gate_g);

                pass.set_pipeline(&self.gelu_pipeline);
                pass.set_bind_group(0, &pres.bg_gelu, &[]);
                pass.dispatch_workgroups(pres.wg, 1, 1);

                pass.set_pipeline(&self.mul_pipeline);
                pass.set_bind_group(0, &pres.bg_mul, &[]);
                pass.dispatch_workgroups(pres.wg, 1, 1);
            }
            let per_layer_dim = ple_input.per_layer_dim;
            encoder.copy_buffer_to_buffer(
                &pres.mulled,
                0,
                &ple_proj_g.x_buffer,
                0,
                (per_layer_dim as u64) * 4,
            );

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("orangu-server fused ple proj pass"),
                    timestamp_writes: None,
                });
                self.record_matmul(&mut pass, ple_input.proj_w, ple_proj_g);

                pass.set_pipeline(&self.rmsnorm_pipeline);
                pass.set_bind_group(0, &pres.bg_post_norm, &[]);
                pass.dispatch_workgroups(1, 1, 1);

                pass.set_pipeline(&self.add_pipeline);
                pass.set_bind_group(0, &pres.bg_add, &[]);
                pass.dispatch_workgroups(res.embd_wg, 1, 1);
            }
            final_buf = &pres.x3;
        }

        if let Some(sres) = &res.scale {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("orangu-server fused scale pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.scale_pipeline);
                pass.set_bind_group(0, &sres.bg, &[]);
                pass.dispatch_workgroups(res.embd_wg, 1, 1);
            }
            final_buf = &sres.scaled;
        }

        final_buf.clone()
    }

    /// Fuses a gemma4 layer's post-attention chain — `wo`, `attn_post_norm`,
    /// the residual add, `ffn_norm`, `gate`/`up`, GELU, the gate*up
    /// multiply, `down`, `ffn_post_norm`, the second residual add, and
    /// (when present) PLE's own gate/proj/norm/residual-add sub-chain and
    /// `layer_output_scale` — into one command encoder and one submission,
    /// reading the final `[n_embd]` result back exactly once. Every matmul
    /// step reuses the same `op_cache`-cached buffers/bind groups
    /// `matmul`/`matmul_batch` do, and every elementwise/norm step's own
    /// buffers/bind groups are themselves cached per layer
    /// (`fused_entry_for`/`FusedResources`) after the first call — so a
    /// decode step after the first token only pays for three small
    /// uploads (`attn_out`, the residual snapshot, PLE's per-layer slice)
    /// plus the dispatches themselves, not fresh buffer/bind-group
    /// creation every time. Only valid for `n_tokens == 1` (decode) — see
    /// `GemmaModel::forward`'s call site for why prefill doesn't use this
    /// path.
    ///
    /// This is now also the recording half of [`Self::fused_layer`] — see
    /// [`Self::record_fused_post_attention`]. As a standalone entry point
    /// it's only used by this module's own cross-check tests now
    /// (`GemmaModel::forward` calls `fused_layer`), so it uses the
    /// generic `submit_and_readback` (a fresh readback buffer every call)
    /// rather than the cached-readback-buffer optimization it used to
    /// have — correctness-critical test coverage, not a hot path, so that
    /// trade is fine here.
    #[allow(dead_code)]
    pub fn fused_post_attention(&self, input: FusedPostAttentionInput<'_>) -> Vec<f32> {
        let n_embd = input.wo.out_dim;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("orangu-server fused post-attention (standalone) encoder"),
            });
        let final_buf = self.record_fused_post_attention(&mut encoder, input);
        self.submit_and_readback(encoder, &final_buf, n_embd)
    }

    /// GPU-resident causal attention for one decode step (`n_tokens == 1`)
    /// against `cache`'s GPU-resident mirror (built/synced lazily —
    /// `LayerCache::sync_gpu`), replacing the CPU attention loop
    /// `GemmaModel::forward` otherwise runs. `pos` is this token's
    /// absolute position (already pushed into `cache` by the caller, same
    /// as the CPU path requires); `window_start` is `0` for full attention
    /// or the SWA window's start for a sliding-window layer. `q` is
    /// `[n_head, head_dim]`, already Q-normed and RoPE'd by the caller
    /// (still CPU-side — moving those onto the GPU too remains a further,
    /// separate step, not attempted here). Returns `[n_head,
    /// head_dim]`, matching the CPU path's `attn_out` exactly so callers
    /// don't need to branch on shape.
    #[allow(dead_code)]
    pub fn gpu_attention(&self, input: GpuAttentionInput<'_>) -> Vec<f32> {
        let GpuAttentionInput {
            q,
            cache,
            pos,
            window_start,
            n_head,
            n_head_kv,
            head_dim,
            scale,
        } = input;
        debug_assert_eq!(q.len(), n_head * head_dim);
        debug_assert!(window_start <= pos);
        let capacity = cache.capacity();
        let (k_buf, v_buf, probs_buf) =
            cache.sync_gpu(&self.device, &self.queue, n_head, self.kv_f16);

        let q_buf = self.upload_new(q);
        let out_buf = self.scratch_buffer(n_head * head_dim);
        let meta = AttnMeta {
            n_head: n_head as u32,
            n_head_kv: n_head_kv as u32,
            head_dim: head_dim as u32,
            window_start: window_start as u32,
            n_pos: (pos - window_start + 1) as u32,
            capacity: capacity as u32,
            scale,
            _pad: 0,
        };
        let meta_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("orangu-server attention meta"),
            size: std::mem::size_of::<AttnMeta>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&meta_buf, 0, bytemuck::bytes_of(&meta));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("orangu-server attention bind group"),
            layout: &self.attn_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: q_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: k_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: v_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: probs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: meta_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("orangu-server attention encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("orangu-server attention pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.attn_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(n_head as u32, 1, 1);
        }
        let readback_len = (n_head * head_dim) as u64 * 4;
        let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("orangu-server attention readback"),
            size: readback_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&out_buf, 0, &readback_buffer, 0, readback_len);

        self.queue.submit(Some(encoder.finish()));
        self.submission_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        readback_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, |result| {
                result.expect("mapping the attention readback buffer failed");
            });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("polling the device for the attention readback failed");
        let data = readback_buffer
            .slice(..)
            .get_mapped_range()
            .expect("attention readback buffer was not mapped after a successful map_async + poll");
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        readback_buffer.unmap();
        result
    }

    /// Builds (never cached itself — callers cache the whole
    /// [`FusedAttnLayerResources`] this returns) every bind group
    /// `fused_attention` needs that doesn't touch a per-request KV-cache
    /// buffer.
    fn build_fused_attn_layer_resources(
        &self,
        input: &FusedAttnInput<'_>,
        wq_g: &CachedOpResources,
        wk_g: Option<&CachedOpResources>,
        wv_g: Option<&CachedOpResources>,
    ) -> FusedAttnLayerResources {
        let half = input.rope_dim / 2;
        let ff_owned;
        let ff: &[f32] = match input.freq_factors {
            Some(f) => f,
            None => {
                ff_owned = vec![1.0f32; half];
                &ff_owned
            }
        };

        let q_norm_w = self.upload_new(input.q_norm);
        let q_norm_meta =
            self.perhead_norm_meta_buffer(input.n_head as u32, input.head_dim as u32, input.eps);
        let q_norm_bg = self.elem3_bind_group(&q_norm_w, &wq_g.output_buffer, &q_norm_meta);
        let q_norm_wg = input.n_head as u32;

        let q_ff = self.upload_new(ff);
        let q_rope_meta_buf = self.rope_meta_buffer(
            input.n_head as u32,
            input.head_dim as u32,
            input.rope_dim as u32,
            input.pos as u32,
            input.rope_freq_base,
        );
        let q_rope_bg = self.elem3_bind_group(&q_ff, &wq_g.output_buffer, &q_rope_meta_buf);
        let q_rope_wg = ((input.n_head * half) as u32).div_ceil(64).max(1);

        let kv = if let (Some(proj), Some(wk_guard)) = (&input.kv, wk_g) {
            let k_norm_w = self.upload_new(proj.k_norm);
            let k_norm_meta = self.perhead_norm_meta_buffer(
                input.n_head_kv as u32,
                input.head_dim as u32,
                input.eps,
            );
            let k_norm_bg = self.elem3_bind_group(&k_norm_w, &wk_guard.output_buffer, &k_norm_meta);

            let owns_v = wv_g.is_some();
            let v_scratch = if owns_v {
                None
            } else {
                Some(self.scratch_buffer(input.n_head_kv * input.head_dim))
            };
            let v_norm_meta = self.perhead_norm_meta_buffer(
                input.n_head_kv as u32,
                input.head_dim as u32,
                input.eps,
            );
            let v_target: &wgpu::Buffer = match wv_g {
                Some(g) => &g.output_buffer,
                None => v_scratch.as_ref().unwrap(),
            };
            let v_norm_bg = self.elem2_bind_group(v_target, &v_norm_meta);

            let k_ff = self.upload_new(ff);
            let k_rope_meta_buf = self.rope_meta_buffer(
                input.n_head_kv as u32,
                input.head_dim as u32,
                input.rope_dim as u32,
                input.pos as u32,
                input.rope_freq_base,
            );
            let k_rope_bg = self.elem3_bind_group(&k_ff, &wk_guard.output_buffer, &k_rope_meta_buf);

            Some(FusedAttnKvLayerResources {
                k_norm_bg,
                k_norm_wg: input.n_head_kv as u32,
                v_scratch,
                v_norm_bg,
                v_norm_wg: input.n_head_kv as u32,
                k_rope_bg,
                k_rope_meta_buf,
                k_rope_wg: ((input.n_head_kv * half) as u32).div_ceil(64).max(1),
            })
        } else {
            None
        };

        FusedAttnLayerResources {
            q_norm_bg,
            q_norm_wg,
            q_rope_bg,
            q_rope_meta_buf,
            q_rope_wg,
            kv,
        }
    }

    /// Returns this layer's cached [`FusedAttnLayerResources`], building
    /// (and caching) them first on a cache miss.
    fn fused_attn_layer_entry_for(
        &self,
        input: &FusedAttnInput<'_>,
        wq_g: &CachedOpResources,
        wk_g: Option<&CachedOpResources>,
        wv_g: Option<&CachedOpResources>,
    ) -> Arc<FusedAttnLayerResources> {
        let (ptr, start) = input.wq.cache_key();
        let key: FusedAttnLayerCacheKey = (
            ptr,
            start,
            input.n_head,
            input.n_head_kv,
            input.head_dim,
            input.kv.is_some(),
            input.kv.as_ref().is_some_and(|p| p.wv.is_some()),
        );
        {
            let cache = self
                .fused_attn_layer_cache
                .lock()
                .expect("fused attn layer cache poisoned");
            if let Some(entry) = cache.get(&key) {
                return entry.clone();
            }
        }
        let resources = Arc::new(self.build_fused_attn_layer_resources(input, wq_g, wk_g, wv_g));
        let mut cache = self
            .fused_attn_layer_cache
            .lock()
            .expect("fused attn layer cache poisoned");
        cache.entry(key).or_insert(resources).clone()
    }

    /// Records the QKV-projection→Q-norm→Q-RoPE→K-norm→V-norm→K-RoPE→
    /// KV-cache-write→attention chain into `encoder` (does **not**
    /// submit) and returns the GPU buffer holding `attn_out` — the
    /// recording half of [`Self::fused_attention`], split out so
    /// [`Self::fused_layer`]
    /// can chain it with the pre-attention norm and the post-attention
    /// chain in one submission instead of three. See `fused_attention`'s
    /// own doc comment for the full ordering rationale — identical here,
    /// just without the final `queue.submit`/map/read.
    fn record_fused_attention(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: FusedAttnInput<'_>,
    ) -> wgpu::Buffer {
        let kv_dim = input.n_head_kv * input.head_dim;
        let n_embd = input.wq.in_dim;

        let wq_entry = self.op_entry_for(input.wq);
        let wk_entry = input.kv.as_ref().map(|p| self.op_entry_for(p.wk));
        let wv_entry = input
            .kv
            .as_ref()
            .and_then(|p| p.wv)
            .map(|w| self.op_entry_for(w));

        let wq_g = wq_entry.lock().expect("op cache entry poisoned");
        let wk_g = wk_entry
            .as_ref()
            .map(|e| e.lock().expect("op cache entry poisoned"));
        let wv_g = wv_entry
            .as_ref()
            .map(|e| e.lock().expect("op cache entry poisoned"));

        self.upload_or_copy(encoder, &wq_g.x_buffer, input.normed, n_embd);
        if let Some(g) = &wk_g {
            self.upload_or_copy(encoder, &g.x_buffer, input.normed, n_embd);
        }
        if let Some(g) = &wv_g {
            self.upload_or_copy(encoder, &g.x_buffer, input.normed, n_embd);
        }

        let layer =
            self.fused_attn_layer_entry_for(&input, &wq_g, wk_g.as_deref(), wv_g.as_deref());

        // Consume `input` now — nothing above needs the whole struct
        // again, so the rest of this function works with owned fields
        // directly instead of going back through `input.*`.
        let FusedAttnInput {
            normed: _,
            wq,
            q_norm: _,
            kv,
            n_head,
            n_head_kv,
            head_dim,
            rope_dim,
            rope_freq_base,
            freq_factors: _,
            eps: _,
            pos,
            window_start,
            scale,
            cache,
        } = input;

        // `pos` is the one field in these cached meta buffers that
        // genuinely changes every call.
        self.queue.write_buffer(
            &layer.q_rope_meta_buf,
            0,
            bytemuck::bytes_of(&RopeMeta {
                n_head: n_head as u32,
                head_dim: head_dim as u32,
                rope_dim: rope_dim as u32,
                pos: pos as u32,
                freq_base: rope_freq_base,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            }),
        );
        if let Some(kv_res) = &layer.kv {
            self.queue.write_buffer(
                &kv_res.k_rope_meta_buf,
                0,
                bytemuck::bytes_of(&RopeMeta {
                    n_head: n_head_kv as u32,
                    head_dim: head_dim as u32,
                    rope_dim: rope_dim as u32,
                    pos: pos as u32,
                    freq_base: rope_freq_base,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                }),
            );
        }

        // Captured *before* anything below runs — this call (if `kv` is
        // present) will append exactly one position, at this index.
        let write_pos = cache.len;
        let capacity = cache.capacity();
        let (k_buf, v_buf, probs_buf) =
            cache.sync_gpu(&self.device, &self.queue, n_head, self.kv_f16);
        // Cheap `Arc`-backed clones — turns these into values `cache` is
        // no longer borrowed by, so `cache.attn_dispatch()`/
        // `set_attn_dispatch` below (which need their own access to
        // `cache`) don't fight this borrow.
        let k_buf = k_buf.clone();
        let v_buf = v_buf.clone();
        let probs_buf = probs_buf.clone();

        // The calling layer's own `wq` identity — see `GpuAttnDispatch`'s
        // doc comment for why a cross-layer KV-donor layer must not reuse
        // the owning layer's cached dispatch (it binds a *different*
        // layer's Q output).
        let wq_key = wq.cache_key();
        if cache.attn_dispatch(wq_key).is_none() {
            let out_buf = self.scratch_buffer(n_head * head_dim);
            let meta_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("orangu-server fused attention meta"),
                size: std::mem::size_of::<AttnMeta>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("orangu-server fused attention bind group"),
                layout: &self.attn_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wq_g.output_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: k_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: v_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: probs_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: out_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: meta_buf.as_entire_binding(),
                    },
                ],
            });
            let readback_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("orangu-server fused attention readback"),
                size: (n_head * head_dim) as u64 * 4,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            // This layer's own
            // K/V-cast dispatches into the `f16` KV mirror, built once
            // here alongside everything else that references this specific
            // `LayerCache`'s `k_buf`/`v_buf`. `self.kv_cast_pipeline` is
            // only `Some` when `self.kv_f16` is, so this whole block is a
            // no-op cost-wise on adapters without native `f16`.
            let k_cast = if let (true, Some(wk_guard), Some(_)) = (self.kv_f16, &wk_g, &layer.kv) {
                let meta_buf = self.cast_meta_buffer(kv_dim as u32, 0);
                Some(crate::engine::kv_cache::KvCastDispatch {
                    bind_group: self.elem3_bind_group(&wk_guard.output_buffer, &k_buf, &meta_buf),
                    meta_buf,
                })
            } else {
                None
            };
            let v_cast = if let (true, Some(kv_res)) = (self.kv_f16, &layer.kv) {
                let v_source: Option<&wgpu::Buffer> = match &wv_g {
                    Some(g) => Some(&g.output_buffer),
                    None => kv_res.v_scratch.as_ref(),
                };
                v_source.map(|src| {
                    let meta_buf = self.cast_meta_buffer(kv_dim as u32, 0);
                    crate::engine::kv_cache::KvCastDispatch {
                        bind_group: self.elem3_bind_group(src, &v_buf, &meta_buf),
                        meta_buf,
                    }
                })
            } else {
                None
            };
            cache.set_attn_dispatch(
                wq_key,
                crate::engine::kv_cache::GpuAttnDispatch {
                    bind_group,
                    out_buf,
                    meta_buf,
                    readback_buf,
                    k_cast,
                    v_cast,
                },
            );
        }
        let dispatch = cache
            .attn_dispatch(wq_key)
            .expect("attn_dispatch was just built above");
        self.queue.write_buffer(
            &dispatch.meta_buf,
            0,
            bytemuck::bytes_of(&AttnMeta {
                n_head: n_head as u32,
                n_head_kv: n_head_kv as u32,
                head_dim: head_dim as u32,
                window_start: window_start as u32,
                n_pos: (pos - window_start + 1) as u32,
                capacity: capacity as u32,
                scale,
                _pad: 0,
            }),
        );

        // Pass A: QKV matmuls, Q-norm, Q-RoPE, K-norm (independent of the
        // v_scratch copy below, so all safe in one pass).
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("orangu-server fused attention qkv+qnorm+qrope+knorm pass"),
                timestamp_writes: None,
            });
            self.record_matmul(&mut pass, wq, &wq_g);
            if let (Some(proj), Some(wk_guard)) = (&kv, &wk_g) {
                self.record_matmul(&mut pass, proj.wk, wk_guard);
            }
            if let (Some(proj), Some(wv_guard)) = (&kv, &wv_g)
                && proj.wv.is_some()
            {
                self.record_matmul(&mut pass, proj.wv.unwrap(), wv_guard);
            }

            pass.set_pipeline(&self.perhead_rmsnorm_pipeline);
            pass.set_bind_group(0, &layer.q_norm_bg, &[]);
            pass.dispatch_workgroups(layer.q_norm_wg, 1, 1);

            pass.set_pipeline(&self.rope_pipeline);
            pass.set_bind_group(0, &layer.q_rope_bg, &[]);
            pass.dispatch_workgroups(layer.q_rope_wg, 1, 1);

            if let Some(kv_res) = &layer.kv {
                pass.set_pipeline(&self.perhead_rmsnorm_pipeline);
                pass.set_bind_group(0, &kv_res.k_norm_bg, &[]);
                pass.dispatch_workgroups(kv_res.k_norm_wg, 1, 1);
            }
        }

        // V = copy of K's (post-norm) output, only when this layer
        // doesn't own its own V projection — must happen between passes:
        // it needs K-norm (pass A) already done, and K's RoPE (pass B)
        // must not have run yet (V never gets RoPE'd).
        if let (Some(wk_guard), Some(kv_res)) = (&wk_g, &layer.kv)
            && let Some(scratch) = &kv_res.v_scratch
        {
            encoder.copy_buffer_to_buffer(
                &wk_guard.output_buffer,
                0,
                scratch,
                0,
                (kv_dim as u64) * 4,
            );
        }

        // Pass B: V's weightless norm, K's RoPE.
        if let Some(kv_res) = &layer.kv {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("orangu-server fused attention vnorm+krope pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.perhead_rmsnorm_weightless_pipeline);
            pass.set_bind_group(0, &kv_res.v_norm_bg, &[]);
            pass.dispatch_workgroups(kv_res.v_norm_wg, 1, 1);

            pass.set_pipeline(&self.rope_pipeline);
            pass.set_bind_group(0, &kv_res.k_rope_bg, &[]);
            pass.dispatch_workgroups(kv_res.k_rope_wg, 1, 1);
        }

        // KV-cache write: this token's (fully processed) key/value into
        // the GPU-resident mirror at `write_pos` — must happen after K's
        // RoPE and before attention reads the cache. `f16` KV mirror: a straight
        // `copy_buffer_to_buffer` would reinterpret bytes, not convert
        // values, since the source (K/V projection output) is always
        // `f32` — so this dispatches the cast shader instead when
        // `self.kv_f16`, using the two cast dispatches built alongside
        // `dispatch` itself above. The `f32` path is a plain byte copy.
        if let (Some(wk_guard), Some(kv_res)) = (&wk_g, &layer.kv) {
            if self.kv_f16 {
                let offset = (write_pos * kv_dim) as u32;
                let cast_meta = ElemMeta {
                    len: kv_dim as u32,
                    _pad0: offset,
                    extra: 0.0,
                    _pad1: 0,
                };
                let k_cast = dispatch
                    .k_cast
                    .as_ref()
                    .expect("kv_f16 but no k_cast dispatch built");
                let v_cast = dispatch
                    .v_cast
                    .as_ref()
                    .expect("kv_f16 but no v_cast dispatch built");
                self.queue
                    .write_buffer(&k_cast.meta_buf, 0, bytemuck::bytes_of(&cast_meta));
                self.queue
                    .write_buffer(&v_cast.meta_buf, 0, bytemuck::bytes_of(&cast_meta));
                let wg = (kv_dim as u32).div_ceil(64);
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("orangu-server kv cast pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(
                    self.kv_cast_pipeline
                        .as_ref()
                        .expect("kv_f16 but no kv_cast_pipeline"),
                );
                pass.set_bind_group(0, &k_cast.bind_group, &[]);
                pass.dispatch_workgroups(wg, 1, 1);
                pass.set_bind_group(0, &v_cast.bind_group, &[]);
                pass.dispatch_workgroups(wg, 1, 1);
            } else {
                let byte_offset = (write_pos * kv_dim * 4) as u64;
                let byte_len = (kv_dim as u64) * 4;
                encoder.copy_buffer_to_buffer(
                    &wk_guard.output_buffer,
                    0,
                    &k_buf,
                    byte_offset,
                    byte_len,
                );
                let v_source: &wgpu::Buffer = match &wv_g {
                    Some(g) => &g.output_buffer,
                    None => kv_res.v_scratch.as_ref().unwrap(),
                };
                encoder.copy_buffer_to_buffer(v_source, 0, &v_buf, byte_offset, byte_len);
            }
        }

        // Pass C: attention, now that the cache includes this token's own
        // key/value.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("orangu-server fused attention pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.attn_pipeline);
            pass.set_bind_group(0, &dispatch.bind_group, &[]);
            pass.dispatch_workgroups(n_head as u32, 1, 1);
        }

        let out_buf = dispatch.out_buf.clone();
        if kv.is_some() {
            cache.advance_gpu_only();
        }
        out_buf
    }

    /// Chains the QKV
    /// projection's *already-GPU-resident* output straight into Q-norm,
    /// RoPE, K-norm, V's norm, K's RoPE, the KV-cache write, and
    /// attention — all inside **one command encoder**, reading back only
    /// `attn_out`. Replaces what today costs two separate submissions
    /// (`matmul_batch` for QKV, then `gpu_attention`) with one, and moves
    /// Q/K-norm + RoPE + V's norm off the CPU entirely for the decode
    /// path.
    ///
    /// Ordering matches `GemmaModel::forward`'s CPU reference exactly
    /// (see its own comments): Q-norm and Q-RoPE run independently of the
    /// K/V side; K-norm runs, *then* (only when this layer doesn't own
    /// its own V projection) V is a **copy of K's output already after
    /// K-norm** — a `copy_buffer_to_buffer` into `v_scratch`, since that's
    /// GPU-resident data now, not a CPU `Vec` to `.clone()` — *then* V's
    /// weightless norm runs on whichever buffer V ends up in, and *only
    /// then* does K get RoPE'd (V never does). The KV-cache write
    /// (`LayerCache::advance_gpu_only`) happens after K's RoPE, and
    /// attention's own dispatch happens after the KV-cache write, so it
    /// sees the current token's own key/value too — the same dependency
    /// chain the CPU path's statement order encodes, just as encoder
    /// barriers instead of Rust statement order.
    ///
    /// Cross-layer KV-donor layers (`input.kv: None`) skip the entire K/V
    /// sub-chain and the cache write — `cache` is already fully up to
    /// date from an earlier layer in the same forward pass, the same
    /// `cache_index` indirection the CPU path uses.
    ///
    /// Every bind group here is built once per layer and reused forever
    /// after (`fused_attn_layer_cache` for everything model-scoped,
    /// `LayerCache::attn_dispatch` for the one piece that touches this
    /// request's own KV-cache buffers) — a first, uncached version
    /// rebuilt them fresh on every call, paying bind group/buffer
    /// *creation* cost per token, the
    /// exact failure mode `fused_post_attention` hit first (see its own
    /// doc comment). `GemmaModel::forward` calls this method, not
    /// `gpu_attention`.
    ///
    /// This is now also the recording half of [`Self::fused_layer`] — see
    /// [`Self::record_fused_attention`]. As a standalone entry point it's
    /// only used by this module's own cross-check tests now (`GemmaModel
    /// ::forward` calls `fused_layer`), so it uses the generic
    /// `submit_and_readback` (a fresh readback buffer every call) rather
    /// than the cached-readback-buffer optimization it used to have —
    /// correctness-critical test coverage, not a hot path, so that
    /// trade is fine here.
    #[allow(dead_code)]
    pub fn fused_attention(&self, input: FusedAttnInput<'_>) -> Vec<f32> {
        let n_head = input.n_head;
        let head_dim = input.head_dim;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("orangu-server fused attention (standalone) encoder"),
            });
        let out_buf = self.record_fused_attention(&mut encoder, input);
        self.submit_and_readback(encoder, &out_buf, n_head * head_dim)
    }

    /// Builds (never cached itself — callers cache the whole
    /// [`FusedLayerResources`] this returns) `fused_layer`'s own
    /// buffers/bind group: the residual-stream buffer, the pre-attention
    /// norm's output buffer and bind group, and the final readback
    /// buffer.
    fn build_fused_layer_resources(
        &self,
        n_embd: usize,
        attn_norm: &[f32],
        eps: f32,
    ) -> FusedLayerResources {
        let x_buf = self.scratch_buffer(n_embd);
        let normed_buf = self.scratch_buffer(n_embd);
        let attn_norm_w = self.upload_new(attn_norm);
        let meta = self.elem_meta_buffer(n_embd as u32, eps);
        let attn_norm_bg = self.elem4_bind_group(&x_buf, &attn_norm_w, &normed_buf, &meta);
        FusedLayerResources {
            x_buf,
            normed_buf,
            attn_norm_bg,
        }
    }

    /// Returns this layer's cached [`FusedLayerResources`], building (and
    /// caching) them first on a cache miss, keyed by `wq`'s identity plus
    /// the shape/config values that identity's cached resources were built
    /// for — see [`FusedLayerCacheKey`]'s own doc comment for why the bare
    /// identity used to be, and no longer is, enough.
    fn fused_layer_entry_for(
        &self,
        wq: &QuantMatrix,
        n_embd: usize,
        attn_norm: &[f32],
        eps: f32,
    ) -> Arc<FusedLayerResources> {
        let (ptr, start) = wq.cache_key();
        let key: FusedLayerCacheKey = (
            ptr,
            start,
            n_embd,
            eps.to_bits(),
            attn_norm.as_ptr() as usize,
        );
        {
            let cache = self
                .fused_layer_cache
                .lock()
                .expect("fused layer cache poisoned");
            if let Some(entry) = cache.get(&key) {
                return entry.clone();
            }
        }
        let resources = Arc::new(self.build_fused_layer_resources(n_embd, attn_norm, eps));
        let mut cache = self
            .fused_layer_cache
            .lock()
            .expect("fused layer cache poisoned");
        cache.entry(key).or_insert(resources).clone()
    }

    /// Records the pre-attention `attn_norm`, the whole QKV/RoPE/norm/
    /// KV-write/attention chain ([`Self::record_fused_attention`]), and
    /// the whole post-attention `wo`/FFN/PLE chain
    /// ([`Self::record_fused_post_attention`]) into `encoder` (does
    /// **not** submit) and returns the GPU buffer holding this layer's
    /// final `[n_embd]` output — the recording half of
    /// [`Self::fused_layer`], split out
    /// so a whole *forward pass* can chain every layer, plus
    /// `output_norm` and `lm_head`, into **one** encoder/submission
    /// instead of one per layer.
    ///
    /// `input.x` is uploaded/copied into a cached, per-layer buffer
    /// (`fused_layer_entry_for`/`FusedLayerResources`) that also backs
    /// the pre-attention norm's bind group and `record_fused_post_
    /// attention`'s `residual` input — the *same* GPU buffer serves both
    /// roles, exactly matching the CPU reference's `x` being read once
    /// for `attn_norm` and again (unmodified) for the residual add after
    /// `wo`. `pub` so `GemmaModel::forward` can call it once per layer
    /// while driving its own full-forward-pass encoder directly.
    pub fn record_fused_layer(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: FusedLayerInput<'_>,
    ) -> wgpu::Buffer {
        let n_embd = input.wq.in_dim;
        let layer_res = self.fused_layer_entry_for(input.wq, n_embd, input.attn_norm, input.eps);

        let FusedLayerInput {
            x,
            attn_norm: _,
            wq,
            q_norm,
            kv,
            n_head,
            n_head_kv,
            head_dim,
            rope_dim,
            rope_freq_base,
            freq_factors,
            eps,
            pos,
            window_start,
            scale,
            cache,
            wo,
            attn_post_norm,
            ffn_norm,
            ffn_gate,
            ffn_up,
            ffn_down,
            ffn_post_norm,
            ple,
            layer_output_scale,
        } = input;

        self.upload_or_copy(encoder, &layer_res.x_buf, x, n_embd);

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("orangu-server fused layer attn_norm pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.rmsnorm_pipeline);
            pass.set_bind_group(0, &layer_res.attn_norm_bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        let attn_out_buf = self.record_fused_attention(
            encoder,
            FusedAttnInput {
                normed: GpuInput::Gpu(&layer_res.normed_buf, 0),
                wq,
                q_norm,
                kv,
                n_head,
                n_head_kv,
                head_dim,
                rope_dim,
                rope_freq_base,
                freq_factors,
                eps,
                pos,
                window_start,
                scale,
                cache,
            },
        );

        self.record_fused_post_attention(
            encoder,
            FusedPostAttentionInput {
                attn_out: GpuInput::Gpu(&attn_out_buf, 0),
                residual: GpuInput::Gpu(&layer_res.x_buf, 0),
                wo,
                attn_post_norm,
                ffn_norm,
                ffn_gate,
                ffn_up,
                ffn_down,
                ffn_post_norm,
                eps,
                ple,
                layer_output_scale,
            },
        )
    }

    /// The whole layer in one
    /// command encoder, one submission, one readback. This is
    /// also the recording half of a full forward pass
    /// ([`Self::record_fused_layer`]) — see that method's doc comment. As
    /// a standalone entry point it's only used by this module's own
    /// cross-check tests now (`GemmaModel::forward` chains
    /// `record_fused_layer` across every layer itself), so it uses the
    /// generic [`Self::submit_and_readback`] rather than a dedicated
    /// cached readback buffer.
    #[allow(dead_code)]
    pub fn fused_layer(&self, input: FusedLayerInput<'_>) -> Vec<f32> {
        let n_embd = input.wq.in_dim;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("orangu-server fused layer (standalone) encoder"),
            });
        let final_buf = self.record_fused_layer(&mut encoder, input);
        self.submit_and_readback(encoder, &final_buf, n_embd)
    }

    /// Starts a fresh, empty command encoder — the one piece of GPU state
    /// `GemmaModel::forward` (a different module) needs to drive its own
    /// full-forward-pass recording (`record_fused_layer` × every layer,
    /// `record_output_norm`,
    /// `record_full_matmul` for `lm_head`, all chained into **one**
    /// encoder it submits itself via [`Self::submit_and_readback`]).
    pub fn new_encoder(&self, label: &str) -> wgpu::CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) })
    }

    /// Records a single RMSNorm dispatch into `encoder` (does **not**
    /// submit) — `output_norm`, the one norm in a gemma4 forward pass that
    /// isn't already part of `fused_layer`'s per-layer chain. Builds a
    /// fresh weight buffer/bind group every call rather than caching one:
    /// unlike every per-layer resource cache in this file, this runs
    /// *once* per decode step (not once per layer), so the cost this
    /// avoids caching is already negligible next to the 35-layer chain
    /// around it — and skipping a cache sidesteps the exact kind of
    /// stale-buffer-identity bug two per-layer caches already hit earlier
    /// in this project (see `fused_post_attention`/`fused_attention`'s own
    /// history). `pub` for the same reason as `submit_and_readback`.
    pub fn record_output_norm(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        x: GpuInput<'_>,
        weight: &[f32],
        eps: f32,
        n_embd: usize,
    ) -> wgpu::Buffer {
        let x_buf = self.scratch_buffer(n_embd);
        self.upload_or_copy(encoder, &x_buf, x, n_embd);
        let weight_buf = self.upload_new(weight);
        let out_buf = self.scratch_buffer(n_embd);
        let meta = self.elem_meta_buffer(n_embd as u32, eps);
        let bg = self.elem4_bind_group(&x_buf, &weight_buf, &out_buf, &meta);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("orangu-server output_norm pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.rmsnorm_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        drop(pass);
        out_buf
    }

    /// Records one full `x -> w` matmul (`n_tokens == 1`) into `encoder`
    /// (does **not** submit), using the same `op_cache`-backed
    /// buffers/bind group every other matmul call reuses, and returns the
    /// GPU buffer holding the result — the one piece `record_fused_layer`
    /// doesn't already expose standalone, needed for `lm_head` in a fully
    /// GPU-resident forward pass.
    /// `pub` for the same reason as `submit_and_readback`.
    pub fn record_full_matmul(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        x: GpuInput<'_>,
        w: &QuantMatrix,
    ) -> wgpu::Buffer {
        let entry = self.op_entry_for(w);
        let g = entry.lock().expect("op cache entry poisoned");
        self.upload_or_copy(encoder, &g.x_buffer, x, w.in_dim);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("orangu-server record_full_matmul pass"),
                timestamp_writes: None,
            });
            self.record_matmul(&mut pass, w, &g);
        }
        g.output_buffer.clone()
    }

    /// Records gemma4's per-layer-embedding (PLE) *input* projection —
    /// `GemmaModel::compute_per_layer_inputs`'s GPU-fused equivalent — into
    /// `encoder` (does **not** submit), returning a `[n_layer, per_layer]`
    /// GPU buffer every layer's `FusedPle::per_layer_slice` then reads its
    /// own slice out of via `GpuInput::Gpu(buf, il * per_layer)`, no copy
    /// needed. Before this, the
    /// projection ran as `Backend::matmul`'s own separate submit-and-wait,
    /// the "2nd" of decode's two GPU round trips; folding it into the same
    /// encoder as the rest of the forward pass (`GemmaModel::forward`'s
    /// `n_tokens == 1` branch) takes that back down to one. Only used by
    /// that branch — prefill and the CPU backend still call
    /// `compute_per_layer_inputs` directly, unaffected by this.
    ///
    /// Mirrors `compute_per_layer_inputs`'s three steps exactly, just with
    /// each one a GPU dispatch instead of a CPU loop: scale by `1/sqrt
    /// (n_embd)` (`scale_pipeline`), RMSNorm each of `n_layer` independent
    /// `per_layer`-wide rows against the *one* shared `proj_norm` weight
    /// (`perhead_rmsnorm_pipeline` — the exact shape this kernel already
    /// exists for, just with "head" relabeled "layer"), add the gathered
    /// per-layer token embedding (`add_pipeline`), scale by `1/sqrt(2)`
    /// (`scale_pipeline` again). `gathered` itself (`GemmaModel::
    /// gather_per_layer_tok_embd` — a tiny embedding-table row copy) stays
    /// a plain CPU compute + upload: cheap enough that a dedicated GPU
    /// gather kernel would only add complexity, not speed.
    pub fn record_ple_projection(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: PleProjectionInput<'_>,
    ) -> wgpu::Buffer {
        let PleProjectionInput {
            x,
            proj_w,
            proj_norm,
            gathered,
            n_layer,
            per_layer,
            eps,
        } = input;
        let total = n_layer * per_layer;
        debug_assert_eq!(proj_w.out_dim, total);
        debug_assert_eq!(proj_norm.len(), per_layer);
        debug_assert_eq!(gathered.len(), total);

        let proj_entry = self.op_entry_for(proj_w);
        let proj_g = proj_entry.lock().expect("op cache entry poisoned");
        self.upload_or_copy(encoder, &proj_g.x_buffer, x, proj_w.in_dim);

        let projection_scale = 1.0 / (proj_w.in_dim as f32).sqrt();
        let input_scale = 1.0 / 2f32.sqrt();

        let scaled = self.scratch_buffer(total);
        let norm_w = self.upload_new(proj_norm);
        let gathered_buf = self.upload_new(gathered);
        let summed = self.scratch_buffer(total);
        let final_buf = self.scratch_buffer(total);

        let meta_scale1 = self.elem_meta_buffer(total as u32, projection_scale);
        let norm_meta = PerHeadNormMeta {
            n_head: n_layer as u32,
            head_dim: per_layer as u32,
            eps,
            _pad: 0,
        };
        let norm_meta_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("orangu-server ple projection norm meta"),
            size: std::mem::size_of::<PerHeadNormMeta>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&norm_meta_buf, 0, bytemuck::bytes_of(&norm_meta));
        let meta_add = self.elem_meta_buffer(total as u32, 0.0);
        let meta_scale2 = self.elem_meta_buffer(total as u32, input_scale);

        let bg_scale1 = self.elem3_bind_group(&proj_g.output_buffer, &scaled, &meta_scale1);
        // Same (weight, x-in-place, meta) role assignment as
        // `gpu_perhead_rmsnorm`'s own `elem3_bind_group` call — RMSNorm
        // writes back into `scaled` itself.
        let bg_norm = self.elem3_bind_group(&norm_w, &scaled, &norm_meta_buf);
        let bg_add = self.elem4_bind_group(&scaled, &gathered_buf, &summed, &meta_add);
        let bg_scale2 = self.elem3_bind_group(&summed, &final_buf, &meta_scale2);

        let wg = (total as u32).div_ceil(64);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("orangu-server ple projection pass"),
                timestamp_writes: None,
            });
            self.record_matmul(&mut pass, proj_w, &proj_g);

            pass.set_pipeline(&self.scale_pipeline);
            pass.set_bind_group(0, &bg_scale1, &[]);
            pass.dispatch_workgroups(wg, 1, 1);

            pass.set_pipeline(&self.perhead_rmsnorm_pipeline);
            pass.set_bind_group(0, &bg_norm, &[]);
            pass.dispatch_workgroups(n_layer as u32, 1, 1);

            pass.set_pipeline(&self.add_pipeline);
            pass.set_bind_group(0, &bg_add, &[]);
            pass.dispatch_workgroups(wg, 1, 1);

            pass.set_pipeline(&self.scale_pipeline);
            pass.set_bind_group(0, &bg_scale2, &[]);
            pass.dispatch_workgroups(wg, 1, 1);
        }
        final_buf
    }

    /// `true` iff `ORANGU_GPU_SAMPLE=1` was set at startup — see `Self::
    /// gpu_sample`'s own doc comment for why this stays opt-in. Callers
    /// (`GemmaModel::forward_maybe_sampling`) check this before attempting
    /// the GPU-argmax fast path at all.
    pub fn gpu_sample(&self) -> bool {
        self.gpu_sample
    }

    /// Records greedy (argmax) sampling with repeat penalty into `encoder`
    /// (does **not** submit) — `vulkan_shaders::ARGMAX_SAMPLE_SHADER`.
    /// Returns a 1-`u32` buffer holding the winning token id; read it back
    /// with `Self::submit_and_readback_u32`. Not cached the way per-layer
    /// matmul resources are — this runs once per decode step (not once
    /// per layer), so the buffer/bind-group creation cost this would save
    /// is already negligible next to the 35-layer chain feeding it.
    pub fn record_argmax_sample(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: GpuArgmaxSampleInput<'_>,
    ) -> wgpu::Buffer {
        let GpuArgmaxSampleInput {
            logits,
            n_vocab,
            recent_tokens,
            repeat_penalty,
        } = input;

        let logits_buf = self.scratch_buffer(n_vocab);
        self.upload_or_copy(encoder, &logits_buf, logits, n_vocab);
        // A zero-length storage buffer is invalid in WGSL/Vulkan, so
        // always upload at least one element — `meta.n_recent = 0` (set
        // below) means the shader's repeat-penalty loop never reads it in
        // that case, so the padding element's value is never observed.
        let recent_buf = if recent_tokens.is_empty() {
            self.upload_new_u32(&[0])
        } else {
            self.upload_new_u32(recent_tokens)
        };
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("orangu-server argmax sample output"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let meta = SampleMeta {
            n_vocab: n_vocab as u32,
            n_recent: recent_tokens.len() as u32,
            repeat_penalty,
            _pad: 0,
        };
        let meta_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("orangu-server argmax sample meta"),
            size: std::mem::size_of::<SampleMeta>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&meta_buf, 0, bytemuck::bytes_of(&meta));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("orangu-server argmax sample bind group"),
            layout: &self.argmax_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: logits_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: recent_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: meta_buf.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("orangu-server argmax sample pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.argmax_sample_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        drop(pass);

        out_buf
    }
}

/// PLE's own gate/GELU/multiply/proj/RMSNorm/residual-add sub-chain's
/// cached resources — see `VulkanBackend::build_ple_resources`.
struct FusedPleResources {
    /// Rewritten every call (`fused_post_attention`) — everything else
    /// here is fixed once built.
    per_layer_buf: wgpu::Buffer,
    /// Only ever read by `bg_mul` (built once, at cache-build time) — kept
    /// as a field purely to keep the underlying GPU buffer alive for as
    /// long as `bg_mul` references it, not because Rust code reads it
    /// again later.
    #[allow(dead_code)]
    gelu_out: wgpu::Buffer,
    mulled: wgpu::Buffer,
    /// Same reasoning as `gelu_out` above — read only by `bg_add`.
    #[allow(dead_code)]
    normed: wgpu::Buffer,
    x3: wgpu::Buffer,
    bg_gelu: wgpu::BindGroup,
    bg_mul: wgpu::BindGroup,
    bg_post_norm: wgpu::BindGroup,
    bg_add: wgpu::BindGroup,
    wg: u32,
}

/// `layer_output_scale`'s single-shader sub-chain's cached resources.
struct FusedScaleResources {
    scaled: wgpu::Buffer,
    bg: wgpu::BindGroup,
}

/// One gemma4 layer's cached `fused_attention` resources that *don't*
/// touch any per-request KV-cache buffer — see `VulkanBackend::
/// fused_attn_layer_cache`'s doc comment for why the attention dispatch
/// itself is cached elsewhere (on the request-owned `LayerCache`)
/// instead of here. `q_rope_meta_buf`'s contents (`pos`) are the only
/// thing here that change call to call; everything else, including every
/// bind group, is fixed once built.
struct FusedAttnLayerResources {
    q_norm_bg: wgpu::BindGroup,
    q_norm_wg: u32,
    q_rope_bg: wgpu::BindGroup,
    q_rope_meta_buf: wgpu::Buffer,
    q_rope_wg: u32,
    kv: Option<FusedAttnKvLayerResources>,
}

/// [`FusedAttnLayerResources::kv`] — only present for layers that own
/// their own K/V projection.
struct FusedAttnKvLayerResources {
    k_norm_bg: wgpu::BindGroup,
    k_norm_wg: u32,
    /// `Some` only when this layer doesn't own its own V projection (V
    /// is a copy of K's post-norm output instead).
    v_scratch: Option<wgpu::Buffer>,
    v_norm_bg: wgpu::BindGroup,
    v_norm_wg: u32,
    k_rope_bg: wgpu::BindGroup,
    k_rope_meta_buf: wgpu::Buffer,
    k_rope_wg: u32,
}

/// One gemma4 layer's cached [`VulkanBackend::fused_layer`] resources —
/// the pre-attention norm's own buffers/bind group, plus the residual
/// stream buffer shared across the whole fused chain. `x_buf`'s contents
/// are rewritten every call (this layer's current residual stream, or —
/// when chained by [`VulkanBackend::record_fused_layer`] from a previous
/// layer's GPU output — copied in-place with no CPU round trip);
/// everything else, including the bind group, is fixed once built.
struct FusedLayerResources {
    x_buf: wgpu::Buffer,
    normed_buf: wgpu::Buffer,
    attn_norm_bg: wgpu::BindGroup,
}

/// One gemma4 layer's cached `fused_post_attention` resources — built once
/// (`VulkanBackend::build_fused_resources`) and reused by every later call
/// for the same layer, the same way `CachedOpResources` is reused by plain
/// matmul calls. Only `residual_buf`'s and (when present) `FusedPleResources
/// ::per_layer_buf`'s *contents* change call to call (rewritten via
/// `queue.write_buffer`, same buffer object every time) — everything else,
/// including every bind group, is fixed for this layer's whole lifetime.
struct FusedResources {
    residual_buf: wgpu::Buffer,
    /// Only ever read by bind groups built once at cache-build time
    /// (`bg_add1`/`bg_ffn_norm` for `normed1`, `bg_ffn_norm`/`bg_add2` for
    /// `x1`, `bg_ffn_post_norm` for `normed2`) — kept as fields purely to
    /// keep the underlying GPU buffers alive for as long as those bind
    /// groups reference them, not because Rust code reads them again.
    #[allow(dead_code)]
    normed1: wgpu::Buffer,
    #[allow(dead_code)]
    x1: wgpu::Buffer,
    ffn_normed: wgpu::Buffer,
    #[allow(dead_code)]
    gelu_out: wgpu::Buffer,
    mulled: wgpu::Buffer,
    #[allow(dead_code)]
    normed2: wgpu::Buffer,
    x2: wgpu::Buffer,
    bg_attn_post_norm: wgpu::BindGroup,
    bg_add1: wgpu::BindGroup,
    bg_ffn_norm: wgpu::BindGroup,
    bg_gelu: wgpu::BindGroup,
    bg_mul: wgpu::BindGroup,
    bg_ffn_post_norm: wgpu::BindGroup,
    bg_add2: wgpu::BindGroup,
    ple: Option<FusedPleResources>,
    scale: Option<FusedScaleResources>,
    embd_wg: u32,
    ffn_wg: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::backend::CpuBackend;
    use crate::engine::loader::test_quant_matrix;
    use crate::engine::quant::{
        GGML_TYPE_BF16, GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q4_K,
        GGML_TYPE_Q5_0, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0,
    };
    use std::sync::OnceLock;

    /// One `VulkanBackend` shared by every test in this module, rather
    /// than each test creating (and racing to create) its own. This
    /// matches how the real server actually uses `VulkanBackend` — exactly
    /// one instance, built once at startup, called concurrently by however
    /// many slots are configured (see `main.rs::select_backend`) — and
    /// sidesteps a real, reproducible crash that has nothing to do with
    /// this backend's own logic: creating *multiple separate* `wgpu::
    /// Instance`/`Device` objects concurrently from different threads
    /// (`cargo test`'s default parallelism, one `VulkanBackend::try_init()`
    /// per test, was doing exactly that) intermittently SIGSEGVs
    /// somewhere below wgpu in the GPU driver stack —
    /// confirmed by a dedicated stress test (`stress_single_backend_
    /// concurrent_threads`, still below) hammering one shared instance
    /// from 8 threads at once with zero failures across many runs, while
    /// `cargo test`'s many-separate-instances pattern crashed
    /// intermittently. Concurrent *use* of one Vulkan device is safe (and
    /// is what this pool now proves); concurrent *creation* of several was
    /// not — and was never something the real server does anyway.
    fn shared_vulkan() -> Option<&'static VulkanBackend> {
        static BACKEND: OnceLock<Option<VulkanBackend>> = OnceLock::new();
        BACKEND.get_or_init(VulkanBackend::try_init).as_ref()
    }

    fn next_byte(seed: &mut u64) -> u8 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*seed >> 33) as u8
    }

    fn next_bytes(seed: &mut u64, n: usize) -> Vec<u8> {
        (0..n).map(|_| next_byte(seed)).collect()
    }

    /// A small positive value, bounded well away from zero, infinity, and
    /// subnormals — safe to use for every type's `d`/`dmin` scale field
    /// (and the whole value, for `F32`/`F16`/`BF16`) without risking a NaN
    /// or Inf poisoning the dot product on either backend.
    fn next_bounded_f32(seed: &mut u64) -> f32 {
        0.05 + (next_byte(seed) as f32 / 255.0) * 1.95
    }

    fn f16_bytes(v: f32) -> [u8; 2] {
        half::f16::from_f32(v).to_le_bytes()
    }

    /// Builds one block's raw bytes for `ggml_type`, matching the exact
    /// layout `engine::quant::dequantize` reads. Scale/whole-value float
    /// fields are bounded (see `next_bounded_f32`); every other field
    /// (quant nibbles, high-bit packs, K-quant scale bytes) is safe with
    /// arbitrary bits since it's read back as a plain integer, never
    /// reinterpreted as a float.
    fn build_block(ggml_type: u32, seed: &mut u64) -> Vec<u8> {
        let mut out = Vec::new();
        match ggml_type {
            t if t == GGML_TYPE_F32 => {
                out.extend_from_slice(&next_bounded_f32(seed).to_le_bytes());
            }
            t if t == GGML_TYPE_F16 => {
                out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
            }
            t if t == GGML_TYPE_BF16 => {
                let bits = (next_bounded_f32(seed).to_bits() >> 16) as u16;
                out.extend_from_slice(&bits.to_le_bytes());
            }
            t if t == GGML_TYPE_Q4_0 => {
                out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
                out.extend(next_bytes(seed, 16));
            }
            t if t == GGML_TYPE_Q5_0 => {
                out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
                out.extend(next_bytes(seed, 4));
                out.extend(next_bytes(seed, 16));
            }
            t if t == GGML_TYPE_Q8_0 => {
                out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
                out.extend(next_bytes(seed, 32));
            }
            t if t == GGML_TYPE_Q4_K => {
                out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
                out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
                out.extend(next_bytes(seed, 12));
                out.extend(next_bytes(seed, 128));
            }
            t if t == GGML_TYPE_Q5_K => {
                out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
                out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
                out.extend(next_bytes(seed, 12));
                out.extend(next_bytes(seed, 32));
                out.extend(next_bytes(seed, 128));
            }
            t if t == GGML_TYPE_Q6_K => {
                out.extend(next_bytes(seed, 128));
                out.extend(next_bytes(seed, 64));
                out.extend(next_bytes(seed, 16));
                out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
            }
            other => panic!("build_block: unhandled ggml_type {other}"),
        }
        out
    }

    fn block_elems(ggml_type: u32) -> usize {
        match ggml_type {
            t if t == GGML_TYPE_F32 || t == GGML_TYPE_F16 || t == GGML_TYPE_BF16 => 1,
            t if t == GGML_TYPE_Q4_0 || t == GGML_TYPE_Q5_0 || t == GGML_TYPE_Q8_0 => 32,
            _ => 256,
        }
    }

    /// Cross-checks `VulkanBackend::matmul` against `CpuBackend::matmul`
    /// (already known-correct, see `engine::quant`'s own unit tests) for
    /// `ggml_type`, over random-but-valid quantized data and random
    /// activations — the only real way to verify the WGSL dequant/dot
    /// translation is bit-for-bit faithful to its Rust counterpart, short
    /// of reading GPU assembly. Skips (rather than fails) when no Vulkan
    /// adapter is available, e.g. in a CI container with no GPU.
    fn cross_check(ggml_type: u32, in_dim: usize, out_dim: usize) {
        cross_check_n_tokens(ggml_type, in_dim, out_dim, 3);
    }

    /// Like `cross_check`, but with an explicit `n_tokens` — used with a
    /// value `>= COOP_MIN_N_TOKENS` to exercise the workgroup-cooperative
    /// dispatch path (`VulkanBackend::pipeline_for`/`vulkan_shaders::
    /// shader_source_coop`), which `cross_check`'s fixed `n_tokens = 3`
    /// never reaches.
    fn cross_check_n_tokens(ggml_type: u32, in_dim: usize, out_dim: usize, n_tokens: usize) {
        let Some(vulkan) = shared_vulkan() else {
            eprintln!("skipping: no Vulkan adapter available in this environment");
            return;
        };

        let elems = block_elems(ggml_type);
        assert!(in_dim % elems == 0, "in_dim must be a multiple of {elems}");
        let n_blocks_per_row = in_dim / elems;

        let mut seed = 0xC0FFEE_u64;
        let mut bytes = Vec::new();
        for _ in 0..out_dim {
            for _ in 0..n_blocks_per_row {
                bytes.extend(build_block(ggml_type, &mut seed));
            }
        }
        let w = test_quant_matrix(&bytes, ggml_type, in_dim, out_dim);

        let mut x = vec![0f32; n_tokens * in_dim];
        for v in x.iter_mut() {
            let b = next_byte(&mut seed);
            *v = (b as f32 - 128.0) / 64.0;
        }

        let cpu_out = CpuBackend.matmul(&x, n_tokens, &w);
        let gpu_out = vulkan.matmul(&x, n_tokens, &w);

        // When `ORANGU_PACKED_DOT
        // =1` is set, `Q4_K` at reduce-path shapes (`n_tokens <
        // COOP_MIN_N_TOKENS`) goes through the packed-`f16` dot kernel
        // instead of the scalar `f32` one, needing the same kind of
        // widened, still-bug-catching tolerance the `f16` KV mirror
        // above did. Every other type/shape combination is untouched by that
        // flag and keeps the tight tolerance.
        let packed =
            vulkan.packed_dot_f16 && ggml_type == GGML_TYPE_Q4_K && n_tokens < COOP_MIN_N_TOKENS;
        let tol_factor = if packed { 6e-2 } else { 1e-2 };

        assert_eq!(cpu_out.len(), gpu_out.len());
        for (i, (a, b)) in cpu_out.iter().zip(gpu_out.iter()).enumerate() {
            let tol = tol_factor * a.abs().max(1.0);
            assert!(
                (a - b).abs() <= tol,
                "ggml_type {ggml_type}: mismatch at flat index {i}: cpu={a} gpu={b}"
            );
        }
    }

    #[test]
    fn matmul_matches_cpu_backend_for_f32() {
        cross_check(GGML_TYPE_F32, 64, 17);
    }

    #[test]
    fn matmul_matches_cpu_backend_for_f16() {
        cross_check(GGML_TYPE_F16, 64, 17);
    }

    #[test]
    fn matmul_matches_cpu_backend_for_bf16() {
        cross_check(GGML_TYPE_BF16, 64, 17);
    }

    #[test]
    fn matmul_matches_cpu_backend_for_q4_0() {
        cross_check(GGML_TYPE_Q4_0, 64, 17);
    }

    #[test]
    fn matmul_matches_cpu_backend_for_q5_0() {
        cross_check(GGML_TYPE_Q5_0, 64, 17);
    }

    #[test]
    fn matmul_matches_cpu_backend_for_q8_0() {
        cross_check(GGML_TYPE_Q8_0, 64, 17);
    }

    #[test]
    fn matmul_matches_cpu_backend_for_q4_k() {
        cross_check(GGML_TYPE_Q4_K, 512, 5);
    }

    /// A larger `Q4_K` reduce-path shape than the 512×5 above: `in_dim =
    /// 1536` (6 super-blocks) and `out_dim = 40` (10 full `REDUCE_N_ROWS`
    /// row groups) with `n_tokens > 1`. The 512×5 case has only one full
    /// row group plus a partial one and a single multi-block row; this
    /// exercises the multi-block, multi-full-group, multi-token path that
    /// the block-unroll kernel
    /// (`shader_source_reduce_q4k_wide_unroll`) is built around — the block-
    /// unroll is on by default (opt out with `ORANGU_NO_MLP_UNROLL=1`), so
    /// this cross-checks its kernel bit-for-bit against
    /// `CpuBackend`, just as `ORANGU_WIDE_LOAD=1` exercises the wide-load
    /// kernel through these same shared cross-checks. (Harmless and
    /// tight-tolerance for every other config too.)
    #[test]
    fn matmul_matches_cpu_backend_for_q4_k_multi_group() {
        cross_check_n_tokens(GGML_TYPE_Q4_K, 1536, 40, 3);
    }

    /// The `Q5_K` and `Q6_K` counterparts of the multi-group `Q4_K` test:
    /// same 1536×40 (multi-block, multi-full-4-row-group, multi-token) shape
    /// that the block-unroll kernels
    /// (`shader_source_reduce_q5k_wide_unroll`/`..._q6k_...`) are built
    /// around — cross-checked bit-for-bit against `CpuBackend` on the real
    /// GPU. These exercise the unroll path by default now (it's on unless
    /// `ORANGU_NO_MLP_UNROLL=1`); `Q6_K`'s 2×128 geometry in particular
    /// makes its own kernel the one most worth a dedicated multi-block test.
    #[test]
    fn matmul_matches_cpu_backend_for_q5_k_multi_group() {
        cross_check_n_tokens(GGML_TYPE_Q5_K, 1536, 40, 3);
    }

    #[test]
    fn matmul_matches_cpu_backend_for_q6_k_multi_group() {
        cross_check_n_tokens(GGML_TYPE_Q6_K, 1536, 40, 3);
    }

    #[test]
    fn matmul_matches_cpu_backend_for_q5_k() {
        cross_check(GGML_TYPE_Q5_K, 512, 5);
    }

    #[test]
    fn matmul_matches_cpu_backend_for_q6_k() {
        cross_check(GGML_TYPE_Q6_K, 512, 5);
    }

    /// `n_tokens = 130` (> 64, so this needs 3 tiles of the cooperative
    /// shader's internal token-tiling loop — 64 + 64 + a final, only
    /// partially-active tile of 2 — not just the first) against every
    /// type, exercising `shader_source_coop`/`MAIN_COOP_SUFFIX` for real:
    /// `cross_check`'s own `n_tokens = 3` never reaches it.
    #[test]
    fn matmul_matches_cpu_backend_cooperative_path_f32() {
        cross_check_n_tokens(GGML_TYPE_F32, 64, 17, 130);
    }

    #[test]
    fn matmul_matches_cpu_backend_cooperative_path_f16() {
        cross_check_n_tokens(GGML_TYPE_F16, 64, 17, 130);
    }

    #[test]
    fn matmul_matches_cpu_backend_cooperative_path_bf16() {
        cross_check_n_tokens(GGML_TYPE_BF16, 64, 17, 130);
    }

    #[test]
    fn matmul_matches_cpu_backend_cooperative_path_q4_0() {
        cross_check_n_tokens(GGML_TYPE_Q4_0, 64, 17, 130);
    }

    #[test]
    fn matmul_matches_cpu_backend_cooperative_path_q5_0() {
        cross_check_n_tokens(GGML_TYPE_Q5_0, 64, 17, 130);
    }

    #[test]
    fn matmul_matches_cpu_backend_cooperative_path_q8_0() {
        cross_check_n_tokens(GGML_TYPE_Q8_0, 64, 17, 130);
    }

    #[test]
    fn matmul_matches_cpu_backend_cooperative_path_q4_k() {
        cross_check_n_tokens(GGML_TYPE_Q4_K, 512, 5, 130);
    }

    #[test]
    fn matmul_matches_cpu_backend_cooperative_path_q5_k() {
        cross_check_n_tokens(GGML_TYPE_Q5_K, 512, 5, 130);
    }

    #[test]
    fn matmul_matches_cpu_backend_cooperative_path_q6_k() {
        cross_check_n_tokens(GGML_TYPE_Q6_K, 512, 5, 130);
    }

    /// Every other cooperative-
    /// path test above uses `out_dim <= 17`, which never exceeds
    /// `vulkan_shaders::COOP_TILE_ROWS` (16) and so never exercises more
    /// than one *row* tile of the tiled GEMM's `(row-tile, token-tile)`
    /// dispatch grid — only the token-tile boundary (already covered by
    /// `n_tokens = 130`, 3 token tiles) was ever genuinely multi-tile.
    /// `out_dim = 40` (3 row tiles: 0..16, 16..32, 32..40 — the last only
    /// partially full) combined with `n_tokens = 130` (3 token tiles) and
    /// `in_dim = 768` (24 `COOP_CHUNK`-sized K-streaming iterations, vs.
    /// `Q4_K`'s native 3 super-blocks) exercises row-tile, token-tile, and
    /// K-chunk boundaries all at once, for the one type (`Q4_K`) this
    /// project's real model actually uses.
    #[test]
    fn matmul_matches_cpu_backend_cooperative_path_multi_row_tile_q4_k() {
        cross_check_n_tokens(GGML_TYPE_Q4_K, 768, 40, 130);
    }

    /// The actual batching path (`matmul_batch` with more than one op,
    /// mirroring a transformer layer's independent Q/K/V projections: same
    /// `x`, three different weight matrices, of two different quant
    /// types) — one submission, one poll, must still return each op's
    /// individually-correct result in the same order.
    #[test]
    fn matmul_batch_matches_sequential_cpu_matmuls() {
        let Some(vulkan) = shared_vulkan() else {
            eprintln!("skipping: no Vulkan adapter available in this environment");
            return;
        };

        // 256 (not 64) so the Q4_K op below (block size 256) is valid too,
        // alongside F16 and Q8_0 (block sizes 1 and 32, both divisors of
        // 256) — a mismatch here silently built a zero-length row for the
        // K-type op the first time this test was written, caught only by
        // the length assertions below.
        let in_dim = 256;
        let mut seed = 0xBADF00D_u64;
        let build = |ggml_type: u32, out_dim: usize, seed: &mut u64| {
            let elems = block_elems(ggml_type);
            let n_blocks_per_row = in_dim / elems;
            let mut bytes = Vec::new();
            for _ in 0..out_dim {
                for _ in 0..n_blocks_per_row {
                    bytes.extend(build_block(ggml_type, seed));
                }
            }
            test_quant_matrix(&bytes, ggml_type, in_dim, out_dim)
        };
        let wq = build(GGML_TYPE_Q4_K, 11, &mut seed);
        let wk = build(GGML_TYPE_F16, 7, &mut seed);
        let wv = build(GGML_TYPE_Q8_0, 9, &mut seed);

        let n_tokens = 2;
        let mut x = vec![0f32; n_tokens * in_dim];
        for v in x.iter_mut() {
            *v = (next_byte(&mut seed) as f32 - 128.0) / 64.0;
        }

        let expected_q = CpuBackend.matmul(&x, n_tokens, &wq);
        let expected_k = CpuBackend.matmul(&x, n_tokens, &wk);
        let expected_v = CpuBackend.matmul(&x, n_tokens, &wv);

        let mut batch = vulkan.matmul_batch(&[
            MatmulOp {
                x: &x,
                n_tokens,
                w: &wq,
            },
            MatmulOp {
                x: &x,
                n_tokens,
                w: &wk,
            },
            MatmulOp {
                x: &x,
                n_tokens,
                w: &wv,
            },
        ]);
        assert_eq!(batch.len(), 3);
        let got_v = batch.pop().unwrap();
        let got_k = batch.pop().unwrap();
        let got_q = batch.pop().unwrap();

        for (name, expected, got) in [
            ("q", &expected_q, &got_q),
            ("k", &expected_k, &got_k),
            ("v", &expected_v, &got_v),
        ] {
            assert_eq!(expected.len(), got.len(), "{name}: length mismatch");
            // "q" (`Q4_K`, this
            // test's only reduce-path-shaped op, `n_tokens = 2 <
            // COOP_MIN_N_TOKENS`) goes through the packed-`f16` dot kernel
            // instead of the scalar `f32` one when `ORANGU_PACKED_DOT=1`,
            // which needs the same kind of widened, still-bug-catching
            // tolerance the `f16` KV mirror did; "k"/"v" (`F16`/
            // `Q8_0`) are untouched by that flag and keep the tight
            // tolerance.
            let tol_factor = if name == "q" { 6e-2 } else { 1e-2 };
            for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
                let tol = tol_factor * a.abs().max(1.0);
                assert!(
                    (a - b).abs() <= tol,
                    "{name}: mismatch at index {i}: cpu={a} gpu(batched)={b}"
                );
            }
        }
    }

    /// Permanent regression test: one `VulkanBackend`, many OS threads
    /// hammering it concurrently (the shape real `slots > 1` usage takes).
    /// Written to check whether the intermittent SIGSEGV seen under
    /// `cargo test`'s default parallelism (many *separate*
    /// `VulkanBackend`/`Device` instances created concurrently across
    /// threads) also reproduces for a *single* shared instance, which is
    /// the actually-relevant production scenario — it doesn't (confirmed
    /// across many runs while diagnosing that bug), so this stays as a
    /// standing guard against a regression there.
    #[test]
    fn stress_single_backend_concurrent_threads() {
        let Some(vulkan) = shared_vulkan() else {
            eprintln!("skipping: no Vulkan adapter available in this environment");
            return;
        };

        let in_dim = 256;
        let mut seed = 0x5EED_u64;
        let build = |ggml_type: u32, out_dim: usize, seed: &mut u64| {
            let elems = block_elems(ggml_type);
            let n_blocks_per_row = in_dim / elems;
            let mut bytes = Vec::new();
            for _ in 0..out_dim {
                for _ in 0..n_blocks_per_row {
                    bytes.extend(build_block(ggml_type, seed));
                }
            }
            test_quant_matrix(&bytes, ggml_type, in_dim, out_dim)
        };
        let weights: Vec<Arc<QuantMatrix>> = [
            GGML_TYPE_Q4_K,
            GGML_TYPE_F16,
            GGML_TYPE_Q8_0,
            GGML_TYPE_Q4_0,
        ]
        .iter()
        .map(|&t| Arc::new(build(t, 11, &mut seed)))
        .collect();

        let mut handles = Vec::new();
        for thread_id in 0..8u64 {
            let weights = weights.clone();
            handles.push(std::thread::spawn(move || {
                let mut seed = 0x1000_u64 + thread_id;
                for _ in 0..40 {
                    let n_tokens = 1 + (next_byte(&mut seed) as usize % 4);
                    let w = &weights[next_byte(&mut seed) as usize % weights.len()];
                    let mut x = vec![0f32; n_tokens * in_dim];
                    for v in x.iter_mut() {
                        *v = (next_byte(&mut seed) as f32 - 128.0) / 64.0;
                    }
                    let _ = vulkan.matmul(&x, n_tokens, w);
                }
            }));
        }
        for h in handles {
            h.join().expect("stress thread panicked");
        }
    }

    /// Cross-checks `fused_post_attention` against the exact same sequence
    /// of `CpuBackend`/`engine::tensor` calls `GemmaModel::forward` makes
    /// today (see `gemma.rs` lines around `let mut attn_proj = self.backend.
    /// matmul(&attn_out, ...)` through `layer_output_scale`) — the only
    /// real way to verify the fused GPU chain (wo -> attn_post_norm ->
    /// residual add -> ffn_norm -> gate/up -> GELU -> mul -> down ->
    /// ffn_post_norm -> residual add -> PLE -> layer_output_scale)
    /// reproduces that reference bit-for-bit (within float tolerance),
    /// including the PLE branch, which the real E2B model actually has.
    #[test]
    fn fused_post_attention_matches_cpu_reference_with_ple() {
        let Some(vulkan) = shared_vulkan() else {
            eprintln!("skipping: no Vulkan adapter available in this environment");
            return;
        };

        let n_embd = 64;
        let ffn_len = 32;
        let per_layer_dim = 16;
        let eps = 1e-6;
        let layer_output_scale = 1.0 / (2.0f32).sqrt();

        let mut seed = 0x5EA1ED_u64;
        let build = |ggml_type: u32, in_dim: usize, out_dim: usize, seed: &mut u64| {
            let elems = block_elems(ggml_type);
            let n_blocks_per_row = in_dim / elems;
            let mut bytes = Vec::new();
            for _ in 0..out_dim {
                for _ in 0..n_blocks_per_row {
                    bytes.extend(build_block(ggml_type, seed));
                }
            }
            test_quant_matrix(&bytes, ggml_type, in_dim, out_dim)
        };

        let wo = build(GGML_TYPE_F32, n_embd, n_embd, &mut seed);
        let ffn_gate = build(GGML_TYPE_F32, n_embd, ffn_len, &mut seed);
        let ffn_up = build(GGML_TYPE_F32, n_embd, ffn_len, &mut seed);
        let ffn_down = build(GGML_TYPE_F32, ffn_len, n_embd, &mut seed);
        let ple_gate_w = build(GGML_TYPE_F32, n_embd, per_layer_dim, &mut seed);
        let ple_proj_w = build(GGML_TYPE_F32, per_layer_dim, n_embd, &mut seed);

        let rand_vec = |len: usize, seed: &mut u64| -> Vec<f32> {
            (0..len)
                .map(|_| (next_byte(seed) as f32 - 128.0) / 64.0)
                .collect()
        };
        let attn_out = rand_vec(n_embd, &mut seed);
        let residual = rand_vec(n_embd, &mut seed);
        let attn_post_norm = rand_vec(n_embd, &mut seed);
        let ffn_norm = rand_vec(n_embd, &mut seed);
        let ffn_post_norm = rand_vec(n_embd, &mut seed);
        let ple_post_norm = rand_vec(n_embd, &mut seed);
        let per_layer_slice = rand_vec(per_layer_dim, &mut seed);

        // Reference: the exact CPU sequence `GemmaModel::forward` runs for
        // this part of a layer.
        let mut attn_proj = CpuBackend.matmul(&attn_out, 1, &wo);
        crate::engine::tensor::rmsnorm_inplace(&mut attn_proj, &attn_post_norm, 1, n_embd, eps);
        let mut x = residual.clone();
        crate::engine::tensor::add_inplace(&mut x, &attn_proj);
        let attn_out_residual = x.clone();

        let mut ffn_normed = x.clone();
        crate::engine::tensor::rmsnorm_inplace(&mut ffn_normed, &ffn_norm, 1, n_embd, eps);
        let mut gate = CpuBackend.matmul(&ffn_normed, 1, &ffn_gate);
        let up = CpuBackend.matmul(&ffn_normed, 1, &ffn_up);
        for g in gate.iter_mut() {
            *g = crate::engine::tensor::gelu(*g);
        }
        crate::engine::tensor::mul_inplace(&mut gate, &up);
        let mut ffn_out = CpuBackend.matmul(&gate, 1, &ffn_down);
        crate::engine::tensor::rmsnorm_inplace(&mut ffn_out, &ffn_post_norm, 1, n_embd, eps);
        x = attn_out_residual;
        crate::engine::tensor::add_inplace(&mut x, &ffn_out);

        let pe_in = x.clone();
        let mut g = CpuBackend.matmul(&x, 1, &ple_gate_w);
        for v in g.iter_mut() {
            *v = crate::engine::tensor::gelu(*v);
        }
        crate::engine::tensor::mul_inplace(&mut g, &per_layer_slice);
        let mut proj = CpuBackend.matmul(&g, 1, &ple_proj_w);
        crate::engine::tensor::rmsnorm_inplace(&mut proj, &ple_post_norm, 1, n_embd, eps);
        x = pe_in;
        crate::engine::tensor::add_inplace(&mut x, &proj);

        for v in x.iter_mut() {
            *v *= layer_output_scale;
        }
        let expected = x;

        let got = vulkan.fused_post_attention(FusedPostAttentionInput {
            attn_out: GpuInput::Cpu(&attn_out),
            residual: GpuInput::Cpu(&residual),
            wo: &wo,
            attn_post_norm: &attn_post_norm,
            ffn_norm: &ffn_norm,
            ffn_gate: &ffn_gate,
            ffn_up: &ffn_up,
            ffn_down: &ffn_down,
            ffn_post_norm: &ffn_post_norm,
            eps,
            ple: Some(FusedPle {
                gate_w: &ple_gate_w,
                proj_w: &ple_proj_w,
                post_norm: &ple_post_norm,
                per_layer_slice: GpuInput::Cpu(&per_layer_slice),
                per_layer_dim: per_layer_slice.len(),
            }),
            layer_output_scale: Some(layer_output_scale),
        });

        assert_eq!(expected.len(), got.len());
        for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
            let tol = 3e-2 * a.abs().max(1.0);
            assert!(
                (a - b).abs() <= tol,
                "mismatch at index {i}: cpu={a} gpu(fused)={b}"
            );
        }
    }

    /// Like the test above but without PLE and without
    /// `layer_output_scale` — covers the (also real) gemma4 layer shape
    /// that has neither, so both `Option`s stay exercised as `None`, not
    /// just `Some`.
    #[test]
    fn fused_post_attention_matches_cpu_reference_without_ple() {
        let Some(vulkan) = shared_vulkan() else {
            eprintln!("skipping: no Vulkan adapter available in this environment");
            return;
        };

        let n_embd = 64;
        let ffn_len = 32;
        let eps = 1e-6;

        let mut seed = 0xFACADE_u64;
        let build = |ggml_type: u32, in_dim: usize, out_dim: usize, seed: &mut u64| {
            let elems = block_elems(ggml_type);
            let n_blocks_per_row = in_dim / elems;
            let mut bytes = Vec::new();
            for _ in 0..out_dim {
                for _ in 0..n_blocks_per_row {
                    bytes.extend(build_block(ggml_type, seed));
                }
            }
            test_quant_matrix(&bytes, ggml_type, in_dim, out_dim)
        };

        let wo = build(GGML_TYPE_F32, n_embd, n_embd, &mut seed);
        let ffn_gate = build(GGML_TYPE_F32, n_embd, ffn_len, &mut seed);
        let ffn_up = build(GGML_TYPE_F32, n_embd, ffn_len, &mut seed);
        let ffn_down = build(GGML_TYPE_F32, ffn_len, n_embd, &mut seed);

        let rand_vec = |len: usize, seed: &mut u64| -> Vec<f32> {
            (0..len)
                .map(|_| (next_byte(seed) as f32 - 128.0) / 64.0)
                .collect()
        };
        let attn_out = rand_vec(n_embd, &mut seed);
        let residual = rand_vec(n_embd, &mut seed);
        let attn_post_norm = rand_vec(n_embd, &mut seed);
        let ffn_norm = rand_vec(n_embd, &mut seed);
        let ffn_post_norm = rand_vec(n_embd, &mut seed);

        let mut attn_proj = CpuBackend.matmul(&attn_out, 1, &wo);
        crate::engine::tensor::rmsnorm_inplace(&mut attn_proj, &attn_post_norm, 1, n_embd, eps);
        let mut x = residual.clone();
        crate::engine::tensor::add_inplace(&mut x, &attn_proj);
        let attn_out_residual = x.clone();

        let mut ffn_normed = x.clone();
        crate::engine::tensor::rmsnorm_inplace(&mut ffn_normed, &ffn_norm, 1, n_embd, eps);
        let mut gate = CpuBackend.matmul(&ffn_normed, 1, &ffn_gate);
        let up = CpuBackend.matmul(&ffn_normed, 1, &ffn_up);
        for g in gate.iter_mut() {
            *g = crate::engine::tensor::gelu(*g);
        }
        crate::engine::tensor::mul_inplace(&mut gate, &up);
        let mut ffn_out = CpuBackend.matmul(&gate, 1, &ffn_down);
        crate::engine::tensor::rmsnorm_inplace(&mut ffn_out, &ffn_post_norm, 1, n_embd, eps);
        x = attn_out_residual;
        crate::engine::tensor::add_inplace(&mut x, &ffn_out);
        let expected = x;

        let got = vulkan.fused_post_attention(FusedPostAttentionInput {
            attn_out: GpuInput::Cpu(&attn_out),
            residual: GpuInput::Cpu(&residual),
            wo: &wo,
            attn_post_norm: &attn_post_norm,
            ffn_norm: &ffn_norm,
            ffn_gate: &ffn_gate,
            ffn_up: &ffn_up,
            ffn_down: &ffn_down,
            ffn_post_norm: &ffn_post_norm,
            eps,
            ple: None,
            layer_output_scale: None,
        });

        assert_eq!(expected.len(), got.len());
        for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
            let tol = 3e-2 * a.abs().max(1.0);
            assert!(
                (a - b).abs() <= tol,
                "mismatch at index {i}: cpu={a} gpu(fused)={b}"
            );
        }
    }

    /// `fused_post_attention` caches every buffer/bind group it can reuse
    /// across calls for the *same* layer (`FusedResources`, built once,
    /// looked up by `wo`'s tensor identity on every later call) — a real
    /// risk that reuse introduces: forgetting to rewrite some buffer that
    /// should change every call, so a second call for the same layer
    /// silently reuses the *first* call's data instead of its own. Calls
    /// `fused_post_attention` twice for the same weight tensors with two
    /// different, unrelated sets of `attn_out`/`residual`/PLE inputs and
    /// checks both results independently against the CPU reference — a
    /// caching bug would make the second call's result match the first
    /// call's expected output (or some stale mix) rather than its own.
    #[test]
    fn fused_post_attention_repeated_calls_use_fresh_data_not_cached_data() {
        let Some(vulkan) = shared_vulkan() else {
            eprintln!("skipping: no Vulkan adapter available in this environment");
            return;
        };

        let n_embd = 64;
        let ffn_len = 32;
        let per_layer_dim = 16;
        let eps = 1e-6;

        let mut seed = 0xCACEDCAC_u64;
        let build = |ggml_type: u32, in_dim: usize, out_dim: usize, seed: &mut u64| {
            let elems = block_elems(ggml_type);
            let n_blocks_per_row = in_dim / elems;
            let mut bytes = Vec::new();
            for _ in 0..out_dim {
                for _ in 0..n_blocks_per_row {
                    bytes.extend(build_block(ggml_type, seed));
                }
            }
            test_quant_matrix(&bytes, ggml_type, in_dim, out_dim)
        };

        let wo = build(GGML_TYPE_F32, n_embd, n_embd, &mut seed);
        let ffn_gate = build(GGML_TYPE_F32, n_embd, ffn_len, &mut seed);
        let ffn_up = build(GGML_TYPE_F32, n_embd, ffn_len, &mut seed);
        let ffn_down = build(GGML_TYPE_F32, ffn_len, n_embd, &mut seed);
        let ple_gate_w = build(GGML_TYPE_F32, n_embd, per_layer_dim, &mut seed);
        let ple_proj_w = build(GGML_TYPE_F32, per_layer_dim, n_embd, &mut seed);

        let rand_vec = |len: usize, seed: &mut u64| -> Vec<f32> {
            (0..len)
                .map(|_| (next_byte(seed) as f32 - 128.0) / 64.0)
                .collect()
        };
        let attn_post_norm = rand_vec(n_embd, &mut seed);
        let ffn_norm = rand_vec(n_embd, &mut seed);
        let ffn_post_norm = rand_vec(n_embd, &mut seed);
        let ple_post_norm = rand_vec(n_embd, &mut seed);
        let layer_output_scale = 1.0 / (2.0f32).sqrt();

        let cpu_reference = |attn_out: &[f32],
                             residual: &[f32],
                             per_layer_slice: &[f32]|
         -> Vec<f32> {
            let mut attn_proj = CpuBackend.matmul(attn_out, 1, &wo);
            crate::engine::tensor::rmsnorm_inplace(&mut attn_proj, &attn_post_norm, 1, n_embd, eps);
            let mut x = residual.to_vec();
            crate::engine::tensor::add_inplace(&mut x, &attn_proj);
            let attn_out_residual = x.clone();

            let mut ffn_normed = x.clone();
            crate::engine::tensor::rmsnorm_inplace(&mut ffn_normed, &ffn_norm, 1, n_embd, eps);
            let mut gate = CpuBackend.matmul(&ffn_normed, 1, &ffn_gate);
            let up = CpuBackend.matmul(&ffn_normed, 1, &ffn_up);
            for g in gate.iter_mut() {
                *g = crate::engine::tensor::gelu(*g);
            }
            crate::engine::tensor::mul_inplace(&mut gate, &up);
            let mut ffn_out = CpuBackend.matmul(&gate, 1, &ffn_down);
            crate::engine::tensor::rmsnorm_inplace(&mut ffn_out, &ffn_post_norm, 1, n_embd, eps);
            x = attn_out_residual;
            crate::engine::tensor::add_inplace(&mut x, &ffn_out);

            let pe_in = x.clone();
            let mut g = CpuBackend.matmul(&x, 1, &ple_gate_w);
            for v in g.iter_mut() {
                *v = crate::engine::tensor::gelu(*v);
            }
            crate::engine::tensor::mul_inplace(&mut g, per_layer_slice);
            let mut proj = CpuBackend.matmul(&g, 1, &ple_proj_w);
            crate::engine::tensor::rmsnorm_inplace(&mut proj, &ple_post_norm, 1, n_embd, eps);
            x = pe_in;
            crate::engine::tensor::add_inplace(&mut x, &proj);

            for v in x.iter_mut() {
                *v *= layer_output_scale;
            }
            x
        };

        for call in 0..2 {
            let attn_out = rand_vec(n_embd, &mut seed);
            let residual = rand_vec(n_embd, &mut seed);
            let per_layer_slice = rand_vec(per_layer_dim, &mut seed);

            let expected = cpu_reference(&attn_out, &residual, &per_layer_slice);
            let got = vulkan.fused_post_attention(FusedPostAttentionInput {
                attn_out: GpuInput::Cpu(&attn_out),
                residual: GpuInput::Cpu(&residual),
                wo: &wo,
                attn_post_norm: &attn_post_norm,
                ffn_norm: &ffn_norm,
                ffn_gate: &ffn_gate,
                ffn_up: &ffn_up,
                ffn_down: &ffn_down,
                ffn_post_norm: &ffn_post_norm,
                eps,
                ple: Some(FusedPle {
                    gate_w: &ple_gate_w,
                    proj_w: &ple_proj_w,
                    post_norm: &ple_post_norm,
                    per_layer_slice: GpuInput::Cpu(&per_layer_slice),
                    per_layer_dim: per_layer_slice.len(),
                }),
                layer_output_scale: Some(layer_output_scale),
            });

            assert_eq!(expected.len(), got.len());
            for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
                let tol = 3e-2 * a.abs().max(1.0);
                assert!(
                    (a - b).abs() <= tol,
                    "call {call}: mismatch at index {i}: cpu={a} gpu(fused)={b}"
                );
            }
        }
    }

    /// Cross-checks `gpu_attention` against the exact CPU attention loop
    /// `GemmaModel::forward` runs (per-head dot products against the
    /// cached keys in the causal window, softmax, weighted value sum) —
    /// GQA (`n_head_kv < n_head`), a KV cache with several positions
    /// already pushed, and full (non-windowed) attention.
    #[test]
    fn gpu_attention_matches_cpu_reference_full_window() {
        let Some(vulkan) = shared_vulkan() else {
            eprintln!("skipping: no Vulkan adapter available in this environment");
            return;
        };

        let n_head = 4;
        let n_head_kv = 2;
        let head_dim = 8;
        let group_size = n_head / n_head_kv;
        let kv_dim = n_head_kv * head_dim;
        let capacity = 16;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut seed = 0xA77E17_u64;
        let mut kv_cache = crate::engine::kv_cache::KvCache::new_with_dims(capacity, &[kv_dim]);
        let n_positions = 5; // positions 0..=4
        for _ in 0..n_positions {
            let k: Vec<f32> = (0..kv_dim)
                .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
                .collect();
            let v: Vec<f32> = (0..kv_dim)
                .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
                .collect();
            kv_cache.layers[0].push(&k, &v);
        }
        let pos = n_positions - 1;
        let window_start = 0;

        let q: Vec<f32> = (0..n_head * head_dim)
            .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
            .collect();

        let mut expected = vec![0f32; n_head * head_dim];
        for h in 0..n_head {
            let kv_head = h / group_size;
            let qh = &q[h * head_dim..(h + 1) * head_dim];
            let mut scores = Vec::with_capacity(pos + 1 - window_start);
            for p in window_start..=pos {
                let kh = kv_cache.layers[0].key_at(p, kv_head, head_dim);
                scores.push(crate::engine::tensor::dot(qh, kh) * scale);
            }
            crate::engine::tensor::softmax_inplace(&mut scores);
            let out = &mut expected[h * head_dim..(h + 1) * head_dim];
            for (offset, &weight) in scores.iter().enumerate() {
                let p = window_start + offset;
                let vh = kv_cache.layers[0].value_at(p, kv_head, head_dim);
                for (o, vi) in out.iter_mut().zip(vh.iter()) {
                    *o += weight * vi;
                }
            }
        }

        let got = vulkan.gpu_attention(GpuAttentionInput {
            q: &q,
            cache: &mut kv_cache.layers[0],
            pos,
            window_start,
            n_head,
            n_head_kv,
            head_dim,
            scale,
        });

        assert_eq!(expected.len(), got.len());
        for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
            let tol = 6e-2 * a.abs().max(1.0);
            assert!(
                (a - b).abs() <= tol,
                "mismatch at index {i}: cpu={a} gpu={b}"
            );
        }
    }

    /// Like the above, but with a nonzero `window_start` (sliding-window
    /// attention) and multiple sequential decode-style calls — each
    /// pushing one new position and re-running attention, the same
    /// prefill-then-decode shape a real request takes, verifying
    /// `LayerCache::sync_gpu`'s incremental upload stays correct across
    /// several calls, not just a single one.
    #[test]
    fn gpu_attention_matches_cpu_reference_sliding_window_across_multiple_steps() {
        let Some(vulkan) = shared_vulkan() else {
            eprintln!("skipping: no Vulkan adapter available in this environment");
            return;
        };

        let n_head = 2;
        let n_head_kv = 1;
        let head_dim = 6;
        let group_size = n_head / n_head_kv;
        let kv_dim = n_head_kv * head_dim;
        let capacity = 16;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let n_swa = 3usize;

        let mut seed = 0x51D1E5_u64;
        let mut kv_cache = crate::engine::kv_cache::KvCache::new_with_dims(capacity, &[kv_dim]);

        for pos in 0..8usize {
            let k: Vec<f32> = (0..kv_dim)
                .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
                .collect();
            let v: Vec<f32> = (0..kv_dim)
                .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
                .collect();
            kv_cache.layers[0].push(&k, &v);

            let window_start = pos.saturating_sub(n_swa - 1);
            let q: Vec<f32> = (0..n_head * head_dim)
                .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
                .collect();

            let mut expected = vec![0f32; n_head * head_dim];
            for h in 0..n_head {
                let kv_head = h / group_size;
                let qh = &q[h * head_dim..(h + 1) * head_dim];
                let mut scores = Vec::with_capacity(pos + 1 - window_start);
                for p in window_start..=pos {
                    let kh = kv_cache.layers[0].key_at(p, kv_head, head_dim);
                    scores.push(crate::engine::tensor::dot(qh, kh) * scale);
                }
                crate::engine::tensor::softmax_inplace(&mut scores);
                let out = &mut expected[h * head_dim..(h + 1) * head_dim];
                for (offset, &weight) in scores.iter().enumerate() {
                    let p = window_start + offset;
                    let vh = kv_cache.layers[0].value_at(p, kv_head, head_dim);
                    for (o, vi) in out.iter_mut().zip(vh.iter()) {
                        *o += weight * vi;
                    }
                }
            }

            let got = vulkan.gpu_attention(GpuAttentionInput {
                q: &q,
                cache: &mut kv_cache.layers[0],
                pos,
                window_start,
                n_head,
                n_head_kv,
                head_dim,
                scale,
            });

            assert_eq!(expected.len(), got.len());
            for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
                let tol = 6e-2 * a.abs().max(1.0);
                assert!(
                    (a - b).abs() <= tol,
                    "pos {pos}: mismatch at index {i}: cpu={a} gpu={b}"
                );
            }
        }
    }

    /// Every other attention
    /// cross-check test here uses `n_pos <= 8`, which never exercises the
    /// online-softmax kernel's multi-*tile* path at all (`TILE = 64`
    /// positions; `n_pos <= 64` is a single tile, no cross-tile merge ever
    /// runs). This test pushes 150 positions and checks attention at
    /// `n_pos = 150` (3 tiles: 64 + 64 + 22, the last only
    /// partially full) and, separately, a sliding window
    /// (`window_start = 50`, `n_pos = 100`, 2 tiles) so the tile-boundary
    /// bookkeeping (`tile_len < 64` on the last tile; `window_start` not
    /// aligned to a tile boundary) is exercised too, not just the common
    /// case where every position happens to fit in one tile.
    #[test]
    fn gpu_attention_matches_cpu_reference_many_positions_multi_tile() {
        let Some(vulkan) = shared_vulkan() else {
            eprintln!("skipping: no Vulkan adapter available in this environment");
            return;
        };

        let n_head = 4;
        let n_head_kv = 2;
        let head_dim = 8;
        let group_size = n_head / n_head_kv;
        let kv_dim = n_head_kv * head_dim;
        let capacity = 200;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut seed = 0x7117E5_u64;
        let mut kv_cache = crate::engine::kv_cache::KvCache::new_with_dims(capacity, &[kv_dim]);
        let n_positions = 150;
        for _ in 0..n_positions {
            let k: Vec<f32> = (0..kv_dim)
                .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
                .collect();
            let v: Vec<f32> = (0..kv_dim)
                .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
                .collect();
            kv_cache.layers[0].push(&k, &v);
        }
        let pos = n_positions - 1;

        for window_start in [0usize, 50] {
            let q: Vec<f32> = (0..n_head * head_dim)
                .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
                .collect();

            let mut expected = vec![0f32; n_head * head_dim];
            for h in 0..n_head {
                let kv_head = h / group_size;
                let qh = &q[h * head_dim..(h + 1) * head_dim];
                let mut scores = Vec::with_capacity(pos + 1 - window_start);
                for p in window_start..=pos {
                    let kh = kv_cache.layers[0].key_at(p, kv_head, head_dim);
                    scores.push(crate::engine::tensor::dot(qh, kh) * scale);
                }
                crate::engine::tensor::softmax_inplace(&mut scores);
                let out = &mut expected[h * head_dim..(h + 1) * head_dim];
                for (offset, &weight) in scores.iter().enumerate() {
                    let p = window_start + offset;
                    let vh = kv_cache.layers[0].value_at(p, kv_head, head_dim);
                    for (o, vi) in out.iter_mut().zip(vh.iter()) {
                        *o += weight * vi;
                    }
                }
            }

            let got = vulkan.gpu_attention(GpuAttentionInput {
                q: &q,
                cache: &mut kv_cache.layers[0],
                pos,
                window_start,
                n_head,
                n_head_kv,
                head_dim,
                scale,
            });

            assert_eq!(expected.len(), got.len());
            for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
                let tol = 6e-2 * a.abs().max(1.0);
                assert!(
                    (a - b).abs() <= tol,
                    "window_start {window_start}: mismatch at index {i}: cpu={a} gpu={b}"
                );
            }
        }
    }

    /// Cross-checks `gpu_rope` against `tensor::rope_apply_scaled_inplace`
    /// — no `freq_factors` (the common case: SWA layers, and every layer
    /// in models without Gemma4's proportional-RoPE tensor).
    #[test]
    fn gpu_rope_matches_cpu_reference_without_freq_factors() {
        let Some(vulkan) = shared_vulkan() else {
            eprintln!("skipping: no Vulkan adapter available in this environment");
            return;
        };

        let n_head = 4;
        let head_dim = 8;
        let rope_dim = 8;
        let pos = 17;
        let freq_base = 10000.0;

        let mut seed = 0x20BE20BE_u64;
        let x: Vec<f32> = (0..n_head * head_dim)
            .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
            .collect();

        let mut expected = x.clone();
        crate::engine::tensor::rope_apply_scaled_inplace(
            &mut expected,
            n_head,
            head_dim,
            rope_dim,
            pos,
            freq_base,
            None,
        );

        let got = vulkan.gpu_rope(GpuRopeInput {
            x: &x,
            n_head,
            head_dim,
            rope_dim,
            pos,
            freq_base,
            freq_factors: None,
        });

        assert_eq!(expected.len(), got.len());
        for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
            let tol = 3e-3 * a.abs().max(1.0);
            assert!(
                (a - b).abs() <= tol,
                "mismatch at index {i}: cpu={a} gpu={b}"
            );
        }
    }

    /// Like the above, but with `freq_factors` set (Gemma4's proportional
    /// RoPE, full-attention layers) and a partial-rope shape (`head_dim >
    /// rope_dim`, so the tail of each head must pass through untouched).
    #[test]
    fn gpu_rope_matches_cpu_reference_with_freq_factors_and_partial_rope() {
        let Some(vulkan) = shared_vulkan() else {
            eprintln!("skipping: no Vulkan adapter available in this environment");
            return;
        };

        let n_head = 3;
        let head_dim = 10;
        let rope_dim = 6; // < head_dim: elements [6, 10) must stay unchanged
        let pos = 5;
        let freq_base = 1_000_000.0;

        let mut seed = 0xFACE0FF_u64;
        let x: Vec<f32> = (0..n_head * head_dim)
            .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
            .collect();
        let freq_factors: Vec<f32> = (0..rope_dim / 2)
            .map(|_| 1.0 + next_byte(&mut seed) as f32 / 255.0)
            .collect();

        let mut expected = x.clone();
        crate::engine::tensor::rope_apply_scaled_inplace(
            &mut expected,
            n_head,
            head_dim,
            rope_dim,
            pos,
            freq_base,
            Some(&freq_factors),
        );

        let got = vulkan.gpu_rope(GpuRopeInput {
            x: &x,
            n_head,
            head_dim,
            rope_dim,
            pos,
            freq_base,
            freq_factors: Some(&freq_factors),
        });

        assert_eq!(expected.len(), got.len());
        for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
            let tol = 3e-3 * a.abs().max(1.0);
            assert!(
                (a - b).abs() <= tol,
                "mismatch at index {i}: cpu={a} gpu={b}"
            );
        }
    }

    /// Cross-checks `gpu_perhead_rmsnorm` (Q-norm/K-norm) against
    /// `tensor::rmsnorm_inplace` treating the input as `n_head` independent
    /// `head_dim`-length rows sharing one weight vector — exactly how
    /// `GemmaModel::forward` calls it today.
    #[test]
    fn gpu_perhead_rmsnorm_matches_cpu_reference() {
        let Some(vulkan) = shared_vulkan() else {
            eprintln!("skipping: no Vulkan adapter available in this environment");
            return;
        };

        let n_head = 5;
        let head_dim = 16;
        let eps = 1e-6;

        let mut seed = 0xB00B00_u64;
        let x: Vec<f32> = (0..n_head * head_dim)
            .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
            .collect();
        let weight: Vec<f32> = (0..head_dim)
            .map(|_| (next_byte(&mut seed) as f32) / 128.0)
            .collect();

        let mut expected = x.clone();
        crate::engine::tensor::rmsnorm_inplace(&mut expected, &weight, n_head, head_dim, eps);

        let got = vulkan.gpu_perhead_rmsnorm(&x, &weight, n_head, head_dim, eps);

        assert_eq!(expected.len(), got.len());
        for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
            let tol = 3e-3 * a.abs().max(1.0);
            assert!(
                (a - b).abs() <= tol,
                "mismatch at index {i}: cpu={a} gpu={b}"
            );
        }
    }

    /// Cross-checks
    /// `VulkanBackend::record_ple_projection` against the same three-step
    /// math `GemmaModel::compute_per_layer_inputs` performs on the CPU
    /// (project, scale, per-layer RMSNorm against one shared weight, add
    /// the already-gathered token embedding, scale again), at `n_tokens ==
    /// 1` — the only shape the decode full-forward-fusion path this feeds
    /// ever uses.
    #[test]
    fn record_ple_projection_matches_cpu_reference() {
        let Some(vulkan) = shared_vulkan() else {
            eprintln!("skipping: no Vulkan adapter available in this environment");
            return;
        };

        let n_embd = 20;
        let n_layer = 5;
        let per_layer = 8;
        let eps = 1e-6;
        let total = n_layer * per_layer;

        let mut seed = 0xFEEDFACE_u64;
        let mut bytes = Vec::new();
        for _ in 0..total {
            for _ in 0..n_embd {
                bytes.extend(build_block(GGML_TYPE_F32, &mut seed));
            }
        }
        let proj_w = test_quant_matrix(&bytes, GGML_TYPE_F32, n_embd, total);

        let rand_vec = |len: usize, seed: &mut u64| -> Vec<f32> {
            (0..len)
                .map(|_| (next_byte(seed) as f32 - 128.0) / 64.0)
                .collect()
        };
        let x = rand_vec(n_embd, &mut seed);
        let proj_norm = rand_vec(per_layer, &mut seed);
        let gathered = rand_vec(total, &mut seed);

        // CPU reference, matching `GemmaModel::compute_per_layer_inputs`'s
        // projection/scale/norm/residual stages (the gather is `gathered`,
        // already done).
        let mut expected = CpuBackend.matmul(&x, 1, &proj_w);
        let projection_scale = 1.0 / (n_embd as f32).sqrt();
        for v in expected.iter_mut() {
            *v *= projection_scale;
        }
        crate::engine::tensor::rmsnorm_inplace(&mut expected, &proj_norm, n_layer, per_layer, eps);
        crate::engine::tensor::add_inplace(&mut expected, &gathered);
        let input_scale = 1.0 / 2f32.sqrt();
        for v in expected.iter_mut() {
            *v *= input_scale;
        }

        let mut encoder = vulkan.new_encoder("test ple projection encoder");
        let buf = vulkan.record_ple_projection(
            &mut encoder,
            PleProjectionInput {
                x: GpuInput::Cpu(&x),
                proj_w: &proj_w,
                proj_norm: &proj_norm,
                gathered: &gathered,
                n_layer,
                per_layer,
                eps,
            },
        );
        let got = vulkan.submit_and_readback(encoder, &buf, total);

        assert_eq!(expected.len(), got.len());
        for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
            let tol = 3e-3 * a.abs().max(1.0);
            assert!(
                (a - b).abs() <= tol,
                "mismatch at index {i}: cpu={a} gpu={b}"
            );
        }
    }

    /// Cross-checks `VulkanBackend::record_argmax_sample` against the same
    /// repeat-penalty-then-argmax math `engine::sampling`'s own
    /// `apply_repeat_penalty`/`argmax` perform on the CPU (reimplemented
    /// inline here since neither is `pub`, the same reason
    /// `gpu_perhead_rmsnorm_weightless_matches_cpu_reference` below
    /// reimplements its own CPU reference rather than importing one).
    /// Uses continuous (not byte-quantized) random logits deliberately:
    /// this kernel's tie-breaking doesn't match `Iterator::max_by`'s "last
    /// element wins" rule (see `ARGMAX_SAMPLE_SHADER`'s own doc comment
    /// for why matching it exactly was never worth the complexity), and
    /// byte-quantized values collide often enough at real vocab sizes to
    /// make ties a real test hazard, not just a theoretical one.
    #[test]
    fn record_argmax_sample_matches_cpu_reference() {
        let Some(vulkan) = shared_vulkan() else {
            eprintln!("skipping: no Vulkan adapter available in this environment");
            return;
        };

        let n_vocab = 2000usize;
        let repeat_penalty = 1.3f32;
        let mut seed = 0xA6C7A5_u64;
        let next_f32 = |seed: &mut u64| -> f32 {
            let a = next_byte(seed) as u32;
            let b = next_byte(seed) as u32;
            let c = next_byte(seed) as u32;
            let d = next_byte(seed) as u32;
            let bits = (a << 24) | (b << 16) | (c << 8) | d;
            (bits as f64 / u32::MAX as f64) as f32 * 8.0 - 4.0
        };

        // Empty, single, several-distinct, and a deliberate repeat (to
        // exercise the compounding-penalty behavior on a token that
        // appears twice in the recent window).
        let recent_cases: Vec<Vec<u32>> = vec![
            vec![],
            vec![7],
            vec![3, 900, 1500, 42],
            vec![3, 900, 3, 1500],
        ];

        for recent_tokens in recent_cases {
            let logits: Vec<f32> = (0..n_vocab).map(|_| next_f32(&mut seed)).collect();

            let mut expected_logits = logits.clone();
            for &tok in &recent_tokens {
                if let Some(v) = expected_logits.get_mut(tok as usize) {
                    *v = if *v > 0.0 {
                        *v / repeat_penalty
                    } else {
                        *v * repeat_penalty
                    };
                }
            }
            let expected = expected_logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as u32)
                .unwrap_or(0);

            let mut encoder = vulkan.new_encoder("test argmax sample encoder");
            let buf = vulkan.record_argmax_sample(
                &mut encoder,
                GpuArgmaxSampleInput {
                    logits: GpuInput::Cpu(&logits),
                    n_vocab,
                    recent_tokens: &recent_tokens,
                    repeat_penalty,
                },
            );
            let got = vulkan.submit_and_readback_u32(encoder, &buf);

            assert_eq!(
                expected, got,
                "recent_tokens={recent_tokens:?}: cpu argmax={expected} gpu argmax={got}"
            );
        }
    }

    /// Cross-checks `gpu_perhead_rmsnorm_weightless` (V's norm) against
    /// the same weightless-RMSNorm formula `GemmaModel`'s private
    /// `rmsnorm_weightless_inplace` uses (mean-of-squares, no learned
    /// scale) — replicated inline here since that helper isn't `pub`.
    #[test]
    fn gpu_perhead_rmsnorm_weightless_matches_cpu_reference() {
        let Some(vulkan) = shared_vulkan() else {
            eprintln!("skipping: no Vulkan adapter available in this environment");
            return;
        };

        let n_head = 4;
        let head_dim = 12;
        let eps = 1e-6;

        let mut seed = 0x5CA1AB1E_u64;
        let x: Vec<f32> = (0..n_head * head_dim)
            .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
            .collect();

        let mut expected = x.clone();
        for row in expected.chunks_mut(head_dim) {
            let mean_sq: f32 = row.iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
            let scale = 1.0 / (mean_sq + eps).sqrt();
            for v in row.iter_mut() {
                *v *= scale;
            }
        }

        let got = vulkan.gpu_perhead_rmsnorm_weightless(&x, n_head, head_dim, eps);

        assert_eq!(expected.len(), got.len());
        for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
            let tol = 3e-3 * a.abs().max(1.0);
            assert!(
                (a - b).abs() <= tol,
                "mismatch at index {i}: cpu={a} gpu={b}"
            );
        }
    }

    /// Cross-checks `fused_attention` against the exact sequence
    /// `GemmaModel::forward` runs on the CPU for a `has_kv` layer that
    /// *owns* its V projection: `matmul_batch(Q,K,V)` -> Q-norm -> Q-RoPE
    /// -> K-norm -> V's weightless norm -> K-RoPE -> cache push ->
    /// attention. Also verifies the KV-cache mirror actually advanced
    /// (`cache.len`) and that a *second* call (simulating the next
    /// decode step) still matches, since `fused_attention` writes
    /// directly into the GPU cache rather than going through `push`.
    #[test]
    fn fused_attention_matches_cpu_reference_owns_v() {
        let Some(vulkan) = shared_vulkan() else {
            eprintln!("skipping: no Vulkan adapter available in this environment");
            return;
        };

        let n_embd = 32;
        let n_head = 4;
        let n_head_kv = 2;
        let head_dim = 8;
        let rope_dim = 8;
        let group_size = n_head / n_head_kv;
        let kv_dim = n_head_kv * head_dim;
        let capacity = 16;
        let eps = 1e-6;
        let rope_freq_base = 10000.0;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut seed = 0xA770C4E5_u64;
        let build = |in_dim: usize, out_dim: usize, seed: &mut u64| {
            let mut bytes = Vec::new();
            for _ in 0..out_dim {
                for _ in 0..in_dim {
                    bytes.extend(build_block(GGML_TYPE_F32, seed));
                }
            }
            test_quant_matrix(&bytes, GGML_TYPE_F32, in_dim, out_dim)
        };
        let wq = build(n_embd, n_head * head_dim, &mut seed);
        let wk = build(n_embd, kv_dim, &mut seed);
        let wv = build(n_embd, kv_dim, &mut seed);

        let rand_vec = |len: usize, seed: &mut u64| -> Vec<f32> {
            (0..len)
                .map(|_| (next_byte(seed) as f32 - 128.0) / 64.0)
                .collect()
        };
        let q_norm = rand_vec(head_dim, &mut seed);
        let k_norm = rand_vec(head_dim, &mut seed);

        // Pre-seed the cache with a few earlier positions (as if a
        // multi-token prefill already ran), so this decode step's
        // attention has real history to attend over. `reference_cache` is
        // a *separate* CPU-only cache the test itself keeps in sync via
        // `push` (real data at every position) — `kv_cache`, the one
        // actually fed to `fused_attention`, only ever advances via
        // `advance_gpu_only` after the first call, which deliberately
        // leaves its own CPU-side vecs unpopulated (see that method's doc
        // comment), so it can't be reused as a second-step reference.
        let mut kv_cache = crate::engine::kv_cache::KvCache::new_with_dims(capacity, &[kv_dim]);
        let mut reference_cache =
            crate::engine::kv_cache::KvCache::new_with_dims(capacity, &[kv_dim]);
        for _ in 0..3 {
            let k: Vec<f32> = rand_vec(kv_dim, &mut seed);
            let v: Vec<f32> = rand_vec(kv_dim, &mut seed);
            kv_cache.layers[0].push(&k, &v);
            reference_cache.layers[0].push(&k, &v);
        }

        for step in 0..2 {
            let pos = kv_cache.layers[0].len;
            let window_start = 0;
            let normed = rand_vec(n_embd, &mut seed);

            // CPU reference, matching `GemmaModel::forward`'s statement
            // order exactly.
            let mut q = CpuBackend.matmul(&normed, 1, &wq);
            crate::engine::tensor::rmsnorm_inplace(&mut q, &q_norm, n_head, head_dim, eps);
            crate::engine::tensor::rope_apply_scaled_inplace(
                &mut q,
                n_head,
                head_dim,
                rope_dim,
                pos,
                rope_freq_base,
                None,
            );
            let mut k = CpuBackend.matmul(&normed, 1, &wk);
            crate::engine::tensor::rmsnorm_inplace(&mut k, &k_norm, n_head_kv, head_dim, eps);
            let mut v = CpuBackend.matmul(&normed, 1, &wv);
            for row in v.chunks_mut(head_dim) {
                let mean_sq: f32 = row.iter().map(|x| x * x).sum::<f32>() / head_dim as f32;
                let s = 1.0 / (mean_sq + eps).sqrt();
                for x in row.iter_mut() {
                    *x *= s;
                }
            }
            crate::engine::tensor::rope_apply_scaled_inplace(
                &mut k,
                n_head_kv,
                head_dim,
                rope_dim,
                pos,
                rope_freq_base,
                None,
            );

            reference_cache.layers[0].push(&k, &v);

            let mut expected = vec![0f32; n_head * head_dim];
            for h in 0..n_head {
                let kv_head = h / group_size;
                let qh = &q[h * head_dim..(h + 1) * head_dim];
                let mut scores = Vec::with_capacity(pos + 1 - window_start);
                for p in window_start..=pos {
                    let kh = reference_cache.layers[0].key_at(p, kv_head, head_dim);
                    scores.push(crate::engine::tensor::dot(qh, kh) * scale);
                }
                crate::engine::tensor::softmax_inplace(&mut scores);
                let out = &mut expected[h * head_dim..(h + 1) * head_dim];
                for (offset, &weight) in scores.iter().enumerate() {
                    let p = window_start + offset;
                    let vh = reference_cache.layers[0].value_at(p, kv_head, head_dim);
                    for (o, vi) in out.iter_mut().zip(vh.iter()) {
                        *o += weight * vi;
                    }
                }
            }

            let got = vulkan.fused_attention(FusedAttnInput {
                normed: GpuInput::Cpu(&normed),
                wq: &wq,
                q_norm: &q_norm,
                kv: Some(FusedAttnProjection {
                    wk: &wk,
                    k_norm: &k_norm,
                    wv: Some(&wv),
                }),
                n_head,
                n_head_kv,
                head_dim,
                rope_dim,
                rope_freq_base,
                freq_factors: None,
                eps,
                pos,
                window_start,
                scale,
                cache: &mut kv_cache.layers[0],
            });

            assert_eq!(expected.len(), got.len());
            for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
                let tol = 6e-2 * a.abs().max(1.0);
                assert!(
                    (a - b).abs() <= tol,
                    "step {step}: mismatch at index {i}: cpu={a} gpu={b}"
                );
            }
            assert_eq!(
                kv_cache.layers[0].len,
                pos + 1,
                "cache should have advanced by one"
            );
        }
    }

    /// Like the above, but for a layer that does *not* own its V
    /// projection (`wv: None`, so V is a copy of K's post-norm output —
    /// the CPU reference's `k.clone()` branch) and *with* `freq_factors`
    /// (Gemma4's proportional RoPE), exercising the other side of both
    /// branches the first test doesn't reach.
    #[test]
    fn fused_attention_matches_cpu_reference_shared_v_with_freq_factors() {
        let Some(vulkan) = shared_vulkan() else {
            eprintln!("skipping: no Vulkan adapter available in this environment");
            return;
        };

        let n_embd = 24;
        let n_head = 4;
        let n_head_kv = 1;
        let head_dim = 6;
        let rope_dim = 6;
        let group_size = n_head / n_head_kv;
        let kv_dim = n_head_kv * head_dim;
        let capacity = 16;
        let eps = 1e-6;
        let rope_freq_base = 500000.0;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut seed = 0x5BA4E5_u64;
        let build = |in_dim: usize, out_dim: usize, seed: &mut u64| {
            let mut bytes = Vec::new();
            for _ in 0..out_dim {
                for _ in 0..in_dim {
                    bytes.extend(build_block(GGML_TYPE_F32, seed));
                }
            }
            test_quant_matrix(&bytes, GGML_TYPE_F32, in_dim, out_dim)
        };
        let wq = build(n_embd, n_head * head_dim, &mut seed);
        let wk = build(n_embd, kv_dim, &mut seed);

        let rand_vec = |len: usize, seed: &mut u64| -> Vec<f32> {
            (0..len)
                .map(|_| (next_byte(seed) as f32 - 128.0) / 64.0)
                .collect()
        };
        let q_norm = rand_vec(head_dim, &mut seed);
        let k_norm = rand_vec(head_dim, &mut seed);
        let freq_factors = rand_vec(rope_dim / 2, &mut seed)
            .iter()
            .map(|v| 1.0 + v.abs())
            .collect::<Vec<f32>>();

        let mut kv_cache = crate::engine::kv_cache::KvCache::new_with_dims(capacity, &[kv_dim]);
        for _ in 0..2 {
            let k: Vec<f32> = rand_vec(kv_dim, &mut seed);
            let v: Vec<f32> = rand_vec(kv_dim, &mut seed);
            kv_cache.layers[0].push(&k, &v);
        }

        let pos = kv_cache.layers[0].len;
        let window_start = 0;
        let normed = rand_vec(n_embd, &mut seed);

        let mut q = CpuBackend.matmul(&normed, 1, &wq);
        crate::engine::tensor::rmsnorm_inplace(&mut q, &q_norm, n_head, head_dim, eps);
        crate::engine::tensor::rope_apply_scaled_inplace(
            &mut q,
            n_head,
            head_dim,
            rope_dim,
            pos,
            rope_freq_base,
            Some(&freq_factors),
        );
        let mut k = CpuBackend.matmul(&normed, 1, &wk);
        crate::engine::tensor::rmsnorm_inplace(&mut k, &k_norm, n_head_kv, head_dim, eps);
        let mut v = k.clone();
        for row in v.chunks_mut(head_dim) {
            let mean_sq: f32 = row.iter().map(|x| x * x).sum::<f32>() / head_dim as f32;
            let s = 1.0 / (mean_sq + eps).sqrt();
            for x in row.iter_mut() {
                *x *= s;
            }
        }
        crate::engine::tensor::rope_apply_scaled_inplace(
            &mut k,
            n_head_kv,
            head_dim,
            rope_dim,
            pos,
            rope_freq_base,
            Some(&freq_factors),
        );

        let mut cpu_cache = kv_cache.layers[0].clone_for_test();
        cpu_cache.push(&k, &v);

        let mut expected = vec![0f32; n_head * head_dim];
        for h in 0..n_head {
            let kv_head = h / group_size;
            let qh = &q[h * head_dim..(h + 1) * head_dim];
            let mut scores = Vec::with_capacity(pos + 1 - window_start);
            for p in window_start..=pos {
                let kh = cpu_cache.key_at(p, kv_head, head_dim);
                scores.push(crate::engine::tensor::dot(qh, kh) * scale);
            }
            crate::engine::tensor::softmax_inplace(&mut scores);
            let out = &mut expected[h * head_dim..(h + 1) * head_dim];
            for (offset, &weight) in scores.iter().enumerate() {
                let p = window_start + offset;
                let vh = cpu_cache.value_at(p, kv_head, head_dim);
                for (o, vi) in out.iter_mut().zip(vh.iter()) {
                    *o += weight * vi;
                }
            }
        }

        let got = vulkan.fused_attention(FusedAttnInput {
            normed: GpuInput::Cpu(&normed),
            wq: &wq,
            q_norm: &q_norm,
            kv: Some(FusedAttnProjection {
                wk: &wk,
                k_norm: &k_norm,
                wv: None,
            }),
            n_head,
            n_head_kv,
            head_dim,
            rope_dim,
            rope_freq_base,
            freq_factors: Some(&freq_factors),
            eps,
            pos,
            window_start,
            scale,
            cache: &mut kv_cache.layers[0],
        });

        assert_eq!(expected.len(), got.len());
        for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
            let tol = 6e-2 * a.abs().max(1.0);
            assert!(
                (a - b).abs() <= tol,
                "mismatch at index {i}: cpu={a} gpu={b}"
            );
        }
    }

    /// Regression test for a real bug caught only by a real end-to-end
    /// request against the actual `E2B` model, not by any of the other
    /// synthetic `fused_attention` tests above: Gemma4's cross-layer
    /// KV-donor layers share *one* `LayerCache` across two layers with
    /// *different* `wq` tensors, and the first version of `LayerCache`'s
    /// cached attention dispatch (`Option<GpuAttnDispatch>`, one slot per
    /// cache) let the *second* layer's call silently reuse the *first*
    /// layer's cached bind group — which binds the first layer's own Q
    /// output buffer, not the second's. Every other test here only ever
    /// calls `fused_attention` with one `wq` per `LayerCache`, so none of
    /// them could have caught this. This test calls it twice against the
    /// *same* `LayerCache` with two distinct `wq`s/`q_norm`s (so a
    /// mix-up produces a detectably wrong `expected`) and checks both
    /// results independently.
    #[test]
    fn fused_attention_two_layers_sharing_one_kv_cache_stay_independent() {
        let Some(vulkan) = shared_vulkan() else {
            eprintln!("skipping: no Vulkan adapter available in this environment");
            return;
        };

        let n_embd = 24;
        let n_head = 4;
        let n_head_kv = 1;
        let head_dim = 6;
        let rope_dim = 6;
        let group_size = n_head / n_head_kv;
        let kv_dim = n_head_kv * head_dim;
        let capacity = 16;
        let eps = 1e-6;
        let rope_freq_base = 10000.0;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut seed = 0xD04202_u64;
        let build = |in_dim: usize, out_dim: usize, seed: &mut u64| {
            let mut bytes = Vec::new();
            for _ in 0..out_dim {
                for _ in 0..in_dim {
                    bytes.extend(build_block(GGML_TYPE_F32, seed));
                }
            }
            test_quant_matrix(&bytes, GGML_TYPE_F32, in_dim, out_dim)
        };
        let rand_vec = |len: usize, seed: &mut u64| -> Vec<f32> {
            (0..len)
                .map(|_| (next_byte(seed) as f32 - 128.0) / 64.0)
                .collect()
        };

        // The donor layer's K/V, and the one KV cache both layers share.
        let wk = build(n_embd, kv_dim, &mut seed);
        let k_norm = rand_vec(head_dim, &mut seed);
        let mut kv_cache = crate::engine::kv_cache::KvCache::new_with_dims(capacity, &[kv_dim]);

        // Compute the expected attention output for a single query `q`
        // (already normed/RoPE'd) against a cache that has exactly one
        // position pushed, matching the CPU reference loop shape.
        let expected_attn = |q: &[f32], reference: &crate::engine::kv_cache::KvCache| -> Vec<f32> {
            let mut out = vec![0f32; n_head * head_dim];
            for h in 0..n_head {
                let kv_head = h / group_size;
                let qh = &q[h * head_dim..(h + 1) * head_dim];
                let mut scores = vec![
                    crate::engine::tensor::dot(
                        qh,
                        reference.layers[0].key_at(0, kv_head, head_dim),
                    ) * scale,
                ];
                crate::engine::tensor::softmax_inplace(&mut scores);
                let vh = reference.layers[0].value_at(0, kv_head, head_dim);
                for (o, vi) in out[h * head_dim..(h + 1) * head_dim]
                    .iter_mut()
                    .zip(vh.iter())
                {
                    *o += scores[0] * vi;
                }
            }
            out
        };
        let cpu_q = |wq: &QuantMatrix, q_norm: &[f32], normed: &[f32], pos: usize| -> Vec<f32> {
            let mut q = CpuBackend.matmul(normed, 1, wq);
            crate::engine::tensor::rmsnorm_inplace(&mut q, q_norm, n_head, head_dim, eps);
            crate::engine::tensor::rope_apply_scaled_inplace(
                &mut q,
                n_head,
                head_dim,
                rope_dim,
                pos,
                rope_freq_base,
                None,
            );
            q
        };

        // Layer A (the donor): its call builds the cache's very first
        // `attn_dispatch` entry, keyed by its own `wq`. `wv: None` (this
        // layer doesn't own a V projection either), so the real K/V the
        // cache ends up with must follow the same rule
        // `fused_attention`/the CPU reference use: V is a copy of K's
        // *post-norm* output, weightless-normed on top, K then RoPE'd
        // (V never is) — not two independent random vectors, or this
        // test's own reference cache wouldn't match what `fused_attention`
        // actually wrote.
        let normed_a = rand_vec(n_embd, &mut seed);
        let wq_a = build(n_embd, n_head * head_dim, &mut seed);
        let q_norm_a = rand_vec(head_dim, &mut seed);
        let mut k_a = CpuBackend.matmul(&normed_a, 1, &wk);
        crate::engine::tensor::rmsnorm_inplace(&mut k_a, &k_norm, n_head_kv, head_dim, eps);
        let mut v_a = k_a.clone();
        for row in v_a.chunks_mut(head_dim) {
            let mean_sq: f32 = row.iter().map(|x| x * x).sum::<f32>() / head_dim as f32;
            let s = 1.0 / (mean_sq + eps).sqrt();
            for x in row.iter_mut() {
                *x *= s;
            }
        }
        crate::engine::tensor::rope_apply_scaled_inplace(
            &mut k_a,
            n_head_kv,
            head_dim,
            rope_dim,
            0,
            rope_freq_base,
            None,
        );
        let mut reference_cache =
            crate::engine::kv_cache::KvCache::new_with_dims(capacity, &[kv_dim]);
        reference_cache.layers[0].push(&k_a, &v_a);

        let q_a = cpu_q(&wq_a, &q_norm_a, &normed_a, 0);
        let expected_a = expected_attn(&q_a, &reference_cache);
        let got_a = vulkan.fused_attention(FusedAttnInput {
            normed: GpuInput::Cpu(&normed_a),
            wq: &wq_a,
            q_norm: &q_norm_a,
            kv: Some(FusedAttnProjection {
                wk: &wk,
                k_norm: &k_norm,
                wv: None,
            }),
            n_head,
            n_head_kv,
            head_dim,
            rope_dim,
            rope_freq_base,
            freq_factors: None,
            eps,
            pos: 0,
            window_start: 0,
            scale,
            cache: &mut kv_cache.layers[0],
        });
        assert_eq!(expected_a.len(), got_a.len());
        for (i, (a, b)) in expected_a.iter().zip(got_a.iter()).enumerate() {
            let tol = 6e-2 * a.abs().max(1.0);
            assert!(
                (a - b).abs() <= tol,
                "layer A: mismatch at index {i}: cpu={a} gpu={b}"
            );
        }

        // Layer B: a KV donor of layer A (`kv: None`), with its own,
        // *different* `wq`/`q_norm`, reading attention from the *same*
        // `LayerCache`. Same position deliberately, to isolate the `wq`
        // mix-up specifically. If the bug were still present, this call
        // would silently reuse layer A's cached bind group (layer A's Q,
        // not layer B's).
        let normed_b = rand_vec(n_embd, &mut seed);
        let wq_b = build(n_embd, n_head * head_dim, &mut seed);
        let q_norm_b = rand_vec(head_dim, &mut seed);

        let q_b = cpu_q(&wq_b, &q_norm_b, &normed_b, 0);
        let expected_b = expected_attn(&q_b, &reference_cache);
        let got_b = vulkan.fused_attention(FusedAttnInput {
            normed: GpuInput::Cpu(&normed_b),
            wq: &wq_b,
            q_norm: &q_norm_b,
            kv: None,
            n_head,
            n_head_kv,
            head_dim,
            rope_dim,
            rope_freq_base,
            freq_factors: None,
            eps,
            pos: 0,
            window_start: 0,
            scale,
            cache: &mut kv_cache.layers[0],
        });
        assert_eq!(expected_b.len(), got_b.len());
        for (i, (a, b)) in expected_b.iter().zip(got_b.iter()).enumerate() {
            let tol = 6e-2 * a.abs().max(1.0);
            assert!(
                (a - b).abs() <= tol,
                "layer B (donor read): mismatch at index {i}: cpu={a} gpu={b} \
                 — if this fails, `LayerCache::attn_dispatch` is reusing layer A's bind group"
            );
        }
    }

    /// Cross-checks `fused_layer` — the whole `attn_norm -> QKV/RoPE/
    /// norm/KV-write/attention -> wo/FFN/PLE/scale` chain in one
    /// submission — against the exact sequence `GemmaModel::forward`
    /// runs on the CPU, end to end for one full layer (owns its own V
    /// projection, has PLE, has `layer_output_scale` — the shape the
    /// real `E2B` model actually uses). Also runs it twice against the
    /// same `LayerCache` (simulating two decode steps) to catch any
    /// staleness in the per-layer caches this introduces.
    #[test]
    fn fused_layer_matches_cpu_reference_full_layer_with_ple() {
        let Some(vulkan) = shared_vulkan() else {
            eprintln!("skipping: no Vulkan adapter available in this environment");
            return;
        };

        let n_embd = 24;
        let n_head = 4;
        let n_head_kv = 2;
        let head_dim = 6;
        let rope_dim = 6;
        let ffn_len = 16;
        let per_layer_dim = 8;
        let group_size = n_head / n_head_kv;
        let kv_dim = n_head_kv * head_dim;
        let capacity = 128;
        let eps = 1e-6;
        let rope_freq_base = 10000.0;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let layer_output_scale = 1.0 / (2.0f32).sqrt();

        let mut seed = 0xFEED1AE4_u64;
        let build = |in_dim: usize, out_dim: usize, seed: &mut u64| {
            let mut bytes = Vec::new();
            for _ in 0..out_dim {
                for _ in 0..in_dim {
                    bytes.extend(build_block(GGML_TYPE_F32, seed));
                }
            }
            test_quant_matrix(&bytes, GGML_TYPE_F32, in_dim, out_dim)
        };
        let rand_vec = |len: usize, seed: &mut u64| -> Vec<f32> {
            (0..len)
                .map(|_| (next_byte(seed) as f32 - 128.0) / 64.0)
                .collect()
        };

        let attn_norm = rand_vec(n_embd, &mut seed);
        let wq = build(n_embd, n_head * head_dim, &mut seed);
        let q_norm = rand_vec(head_dim, &mut seed);
        let wk = build(n_embd, kv_dim, &mut seed);
        let k_norm = rand_vec(head_dim, &mut seed);
        let wv = build(n_embd, kv_dim, &mut seed);
        let wo = build(n_head * head_dim, n_embd, &mut seed);
        let attn_post_norm = rand_vec(n_embd, &mut seed);
        let ffn_norm = rand_vec(n_embd, &mut seed);
        let ffn_gate = build(n_embd, ffn_len, &mut seed);
        let ffn_up = build(n_embd, ffn_len, &mut seed);
        let ffn_down = build(ffn_len, n_embd, &mut seed);
        let ffn_post_norm = rand_vec(n_embd, &mut seed);
        let ple_gate_w = build(n_embd, per_layer_dim, &mut seed);
        let ple_proj_w = build(per_layer_dim, n_embd, &mut seed);
        let ple_post_norm = rand_vec(n_embd, &mut seed);

        let mut kv_cache = crate::engine::kv_cache::KvCache::new_with_dims(capacity, &[kv_dim]);
        let mut reference_cache =
            crate::engine::kv_cache::KvCache::new_with_dims(capacity, &[kv_dim]);
        for _ in 0..3 {
            let k: Vec<f32> = rand_vec(kv_dim, &mut seed);
            let v: Vec<f32> = rand_vec(kv_dim, &mut seed);
            kv_cache.layers[0].push(&k, &v);
            reference_cache.layers[0].push(&k, &v);
        }

        for step in 0..40 {
            let pos = kv_cache.layers[0].len;
            let window_start = 0;
            let x = rand_vec(n_embd, &mut seed);
            let per_layer_slice = rand_vec(per_layer_dim, &mut seed);

            // CPU reference, matching `GemmaModel::forward`'s statement
            // order exactly.
            let mut normed = x.clone();
            crate::engine::tensor::rmsnorm_inplace(&mut normed, &attn_norm, 1, n_embd, eps);

            let mut q = CpuBackend.matmul(&normed, 1, &wq);
            crate::engine::tensor::rmsnorm_inplace(&mut q, &q_norm, n_head, head_dim, eps);
            crate::engine::tensor::rope_apply_scaled_inplace(
                &mut q,
                n_head,
                head_dim,
                rope_dim,
                pos,
                rope_freq_base,
                None,
            );
            let mut k = CpuBackend.matmul(&normed, 1, &wk);
            crate::engine::tensor::rmsnorm_inplace(&mut k, &k_norm, n_head_kv, head_dim, eps);
            let mut v = CpuBackend.matmul(&normed, 1, &wv);
            for row in v.chunks_mut(head_dim) {
                let mean_sq: f32 = row.iter().map(|x| x * x).sum::<f32>() / head_dim as f32;
                let s = 1.0 / (mean_sq + eps).sqrt();
                for x in row.iter_mut() {
                    *x *= s;
                }
            }
            crate::engine::tensor::rope_apply_scaled_inplace(
                &mut k,
                n_head_kv,
                head_dim,
                rope_dim,
                pos,
                rope_freq_base,
                None,
            );
            reference_cache.layers[0].push(&k, &v);

            let mut attn_out = vec![0f32; n_head * head_dim];
            for h in 0..n_head {
                let kv_head = h / group_size;
                let qh = &q[h * head_dim..(h + 1) * head_dim];
                let mut scores = Vec::with_capacity(pos + 1 - window_start);
                for p in window_start..=pos {
                    let kh = reference_cache.layers[0].key_at(p, kv_head, head_dim);
                    scores.push(crate::engine::tensor::dot(qh, kh) * scale);
                }
                crate::engine::tensor::softmax_inplace(&mut scores);
                let out = &mut attn_out[h * head_dim..(h + 1) * head_dim];
                for (offset, &weight) in scores.iter().enumerate() {
                    let p = window_start + offset;
                    let vh = reference_cache.layers[0].value_at(p, kv_head, head_dim);
                    for (o, vi) in out.iter_mut().zip(vh.iter()) {
                        *o += weight * vi;
                    }
                }
            }

            let mut attn_proj = CpuBackend.matmul(&attn_out, 1, &wo);
            crate::engine::tensor::rmsnorm_inplace(&mut attn_proj, &attn_post_norm, 1, n_embd, eps);
            let mut xr = x.clone();
            crate::engine::tensor::add_inplace(&mut xr, &attn_proj);
            let attn_out_residual = xr.clone();

            let mut ffn_normed = xr.clone();
            crate::engine::tensor::rmsnorm_inplace(&mut ffn_normed, &ffn_norm, 1, n_embd, eps);
            let mut gate = CpuBackend.matmul(&ffn_normed, 1, &ffn_gate);
            let up = CpuBackend.matmul(&ffn_normed, 1, &ffn_up);
            for g in gate.iter_mut() {
                *g = crate::engine::tensor::gelu(*g);
            }
            crate::engine::tensor::mul_inplace(&mut gate, &up);
            let mut ffn_out = CpuBackend.matmul(&gate, 1, &ffn_down);
            crate::engine::tensor::rmsnorm_inplace(&mut ffn_out, &ffn_post_norm, 1, n_embd, eps);
            xr = attn_out_residual;
            crate::engine::tensor::add_inplace(&mut xr, &ffn_out);

            let pe_in = xr.clone();
            let mut g = CpuBackend.matmul(&xr, 1, &ple_gate_w);
            for v in g.iter_mut() {
                *v = crate::engine::tensor::gelu(*v);
            }
            crate::engine::tensor::mul_inplace(&mut g, &per_layer_slice);
            let mut proj = CpuBackend.matmul(&g, 1, &ple_proj_w);
            crate::engine::tensor::rmsnorm_inplace(&mut proj, &ple_post_norm, 1, n_embd, eps);
            xr = pe_in;
            crate::engine::tensor::add_inplace(&mut xr, &proj);

            for v in xr.iter_mut() {
                *v *= layer_output_scale;
            }
            let expected = xr;

            let got = vulkan.fused_layer(FusedLayerInput {
                x: GpuInput::Cpu(&x),
                attn_norm: &attn_norm,
                wq: &wq,
                q_norm: &q_norm,
                kv: Some(FusedAttnProjection {
                    wk: &wk,
                    k_norm: &k_norm,
                    wv: Some(&wv),
                }),
                n_head,
                n_head_kv,
                head_dim,
                rope_dim,
                rope_freq_base,
                freq_factors: None,
                eps,
                pos,
                window_start,
                scale,
                cache: &mut kv_cache.layers[0],
                wo: &wo,
                attn_post_norm: &attn_post_norm,
                ffn_norm: &ffn_norm,
                ffn_gate: &ffn_gate,
                ffn_up: &ffn_up,
                ffn_down: &ffn_down,
                ffn_post_norm: &ffn_post_norm,
                ple: Some(FusedPle {
                    gate_w: &ple_gate_w,
                    proj_w: &ple_proj_w,
                    post_norm: &ple_post_norm,
                    per_layer_slice: GpuInput::Cpu(&per_layer_slice),
                    per_layer_dim: per_layer_slice.len(),
                }),
                layer_output_scale: Some(layer_output_scale),
            });

            assert_eq!(expected.len(), got.len());
            for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
                let tol = 1e-1 * a.abs().max(1.0);
                assert!(
                    (a - b).abs() <= tol,
                    "step {step}: mismatch at index {i}: cpu={a} gpu={b}"
                );
            }
        }
    }

    /// Cross-checks `fused_layer` against two layers that share one
    /// `LayerCache` (an owner and a cross-layer KV-donor, gemma4's real
    /// pattern — see `fused_attention_two_layers_sharing_one_kv_cache_stay_
    /// independent`) across *many* sequential decode steps, calling
    /// `fused_layer` for both layers every step exactly as `GemmaModel::
    /// forward` does (owner first, so the donor's attention this step sees
    /// the owner's just-pushed key/value). Every other `fused_layer` test
    /// only exercises one `wq`/`LayerCache` pair at a time; the real
    /// end-to-end bug this is chasing (correct at ~5 decode tokens,
    /// degenerate by ~60) only ever showed up on the real `E2B` model,
    /// which mixes owner and donor layers sharing caches — this test tries
    /// to reproduce that same shape synthetically, far cheaper than a full
    /// HTTP round trip per bisection step.
    #[test]
    fn fused_layer_kv_donor_matches_cpu_reference_many_steps() {
        let Some(vulkan) = shared_vulkan() else {
            eprintln!("skipping: no Vulkan adapter available in this environment");
            return;
        };

        let n_embd = 24;
        let n_head = 2;
        let n_head_kv = 1;
        let head_dim = 6;
        let rope_dim = 6;
        let ffn_len = 16;
        let kv_dim = n_head_kv * head_dim;
        let capacity = 128;
        let eps = 1e-6;
        let rope_freq_base = 10000.0;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut seed = 0xD042_025E_ED00_u64;
        let build = |in_dim: usize, out_dim: usize, seed: &mut u64| {
            let mut bytes = Vec::new();
            for _ in 0..out_dim {
                for _ in 0..in_dim {
                    bytes.extend(build_block(GGML_TYPE_F32, seed));
                }
            }
            test_quant_matrix(&bytes, GGML_TYPE_F32, in_dim, out_dim)
        };
        let rand_vec = |len: usize, seed: &mut u64| -> Vec<f32> {
            (0..len)
                .map(|_| (next_byte(seed) as f32 - 128.0) / 64.0)
                .collect()
        };

        struct LayerWeights {
            attn_norm: Vec<f32>,
            wq: QuantMatrix,
            q_norm: Vec<f32>,
            wo: QuantMatrix,
            attn_post_norm: Vec<f32>,
            ffn_norm: Vec<f32>,
            ffn_gate: QuantMatrix,
            ffn_up: QuantMatrix,
            ffn_down: QuantMatrix,
            ffn_post_norm: Vec<f32>,
        }
        let build_layer = |seed: &mut u64| LayerWeights {
            attn_norm: rand_vec(n_embd, seed),
            wq: build(n_embd, n_head * head_dim, seed),
            q_norm: rand_vec(head_dim, seed),
            wo: build(n_head * head_dim, n_embd, seed),
            attn_post_norm: rand_vec(n_embd, seed),
            ffn_norm: rand_vec(n_embd, seed),
            ffn_gate: build(n_embd, ffn_len, seed),
            ffn_up: build(n_embd, ffn_len, seed),
            ffn_down: build(ffn_len, n_embd, seed),
            ffn_post_norm: rand_vec(n_embd, seed),
        };

        // Layer 0 owns K/V; layer 1 is its cross-layer KV donor
        // (`kv: None`), sharing layer 0's `LayerCache` exactly like
        // gemma4's real donor layers do.
        let l0 = build_layer(&mut seed);
        let wk = build(n_embd, kv_dim, &mut seed);
        let k_norm = rand_vec(head_dim, &mut seed);
        let wv = build(n_embd, kv_dim, &mut seed);
        let l1 = build_layer(&mut seed);

        let mut kv_cache = crate::engine::kv_cache::KvCache::new_with_dims(capacity, &[kv_dim]);
        let mut reference_cache =
            crate::engine::kv_cache::KvCache::new_with_dims(capacity, &[kv_dim]);
        for _ in 0..35 {
            let k: Vec<f32> = rand_vec(kv_dim, &mut seed);
            let v: Vec<f32> = rand_vec(kv_dim, &mut seed);
            kv_cache.layers[0].push(&k, &v);
            reference_cache.layers[0].push(&k, &v);
        }

        // Runs one layer's CPU reference chain (attn_norm -> QKV/RoPE ->
        // attention -> wo/FFN, no PLE/scale), matching `GemmaModel::
        // forward`'s statement order. `kv` is `Some((wk, k_norm, wv))` for
        // the owner (pushes into `reference_cache`), `None` for the donor
        // (reads `reference_cache` without pushing).
        #[allow(clippy::too_many_arguments)]
        fn cpu_layer_reference(
            x: &[f32],
            l: &LayerWeights,
            kv: Option<(&QuantMatrix, &[f32], &QuantMatrix)>,
            n_head: usize,
            n_head_kv: usize,
            head_dim: usize,
            rope_dim: usize,
            rope_freq_base: f32,
            eps: f32,
            pos: usize,
            scale: f32,
            reference_cache: &mut crate::engine::kv_cache::KvCache,
        ) -> Vec<f32> {
            let group_size = n_head / n_head_kv;
            let n_embd = x.len();
            let mut normed = x.to_vec();
            crate::engine::tensor::rmsnorm_inplace(&mut normed, &l.attn_norm, 1, n_embd, eps);

            let mut q = CpuBackend.matmul(&normed, 1, &l.wq);
            crate::engine::tensor::rmsnorm_inplace(&mut q, &l.q_norm, n_head, head_dim, eps);
            crate::engine::tensor::rope_apply_scaled_inplace(
                &mut q,
                n_head,
                head_dim,
                rope_dim,
                pos,
                rope_freq_base,
                None,
            );

            if let Some((wk, k_norm, wv)) = kv {
                let mut k = CpuBackend.matmul(&normed, 1, wk);
                crate::engine::tensor::rmsnorm_inplace(&mut k, k_norm, n_head_kv, head_dim, eps);
                let mut v = CpuBackend.matmul(&normed, 1, wv);
                for row in v.chunks_mut(head_dim) {
                    let mean_sq: f32 = row.iter().map(|x| x * x).sum::<f32>() / head_dim as f32;
                    let s = 1.0 / (mean_sq + eps).sqrt();
                    for x in row.iter_mut() {
                        *x *= s;
                    }
                }
                crate::engine::tensor::rope_apply_scaled_inplace(
                    &mut k,
                    n_head_kv,
                    head_dim,
                    rope_dim,
                    pos,
                    rope_freq_base,
                    None,
                );
                reference_cache.layers[0].push(&k, &v);
            }

            let mut attn_out = vec![0f32; n_head * head_dim];
            for h in 0..n_head {
                let kv_head = h / group_size;
                let qh = &q[h * head_dim..(h + 1) * head_dim];
                let mut scores = Vec::with_capacity(pos + 1);
                for p in 0..=pos {
                    let kh = reference_cache.layers[0].key_at(p, kv_head, head_dim);
                    scores.push(crate::engine::tensor::dot(qh, kh) * scale);
                }
                crate::engine::tensor::softmax_inplace(&mut scores);
                let out = &mut attn_out[h * head_dim..(h + 1) * head_dim];
                for (p, &weight) in scores.iter().enumerate() {
                    let vh = reference_cache.layers[0].value_at(p, kv_head, head_dim);
                    for (o, vi) in out.iter_mut().zip(vh.iter()) {
                        *o += weight * vi;
                    }
                }
            }

            let mut attn_proj = CpuBackend.matmul(&attn_out, 1, &l.wo);
            crate::engine::tensor::rmsnorm_inplace(
                &mut attn_proj,
                &l.attn_post_norm,
                1,
                n_embd,
                eps,
            );
            let mut xr = x.to_vec();
            crate::engine::tensor::add_inplace(&mut xr, &attn_proj);
            let attn_out_residual = xr.clone();

            let mut ffn_normed = xr.clone();
            crate::engine::tensor::rmsnorm_inplace(&mut ffn_normed, &l.ffn_norm, 1, n_embd, eps);
            let mut gate = CpuBackend.matmul(&ffn_normed, 1, &l.ffn_gate);
            let up = CpuBackend.matmul(&ffn_normed, 1, &l.ffn_up);
            for g in gate.iter_mut() {
                *g = crate::engine::tensor::gelu(*g);
            }
            crate::engine::tensor::mul_inplace(&mut gate, &up);
            let mut ffn_out = CpuBackend.matmul(&gate, 1, &l.ffn_down);
            crate::engine::tensor::rmsnorm_inplace(&mut ffn_out, &l.ffn_post_norm, 1, n_embd, eps);
            xr = attn_out_residual;
            crate::engine::tensor::add_inplace(&mut xr, &ffn_out);
            xr
        }

        for step in 0..60 {
            let pos = kv_cache.layers[0].len;
            let x0 = rand_vec(n_embd, &mut seed);
            let x1 = rand_vec(n_embd, &mut seed);

            let expected0 = cpu_layer_reference(
                &x0,
                &l0,
                Some((&wk, &k_norm, &wv)),
                n_head,
                n_head_kv,
                head_dim,
                rope_dim,
                rope_freq_base,
                eps,
                pos,
                scale,
                &mut reference_cache,
            );
            let expected1 = cpu_layer_reference(
                &x1,
                &l1,
                None,
                n_head,
                n_head_kv,
                head_dim,
                rope_dim,
                rope_freq_base,
                eps,
                pos,
                scale,
                &mut reference_cache,
            );

            let got0 = vulkan.fused_layer(FusedLayerInput {
                x: GpuInput::Cpu(&x0),
                attn_norm: &l0.attn_norm,
                wq: &l0.wq,
                q_norm: &l0.q_norm,
                kv: Some(FusedAttnProjection {
                    wk: &wk,
                    k_norm: &k_norm,
                    wv: Some(&wv),
                }),
                n_head,
                n_head_kv,
                head_dim,
                rope_dim,
                rope_freq_base,
                freq_factors: None,
                eps,
                pos,
                window_start: 0,
                scale,
                cache: &mut kv_cache.layers[0],
                wo: &l0.wo,
                attn_post_norm: &l0.attn_post_norm,
                ffn_norm: &l0.ffn_norm,
                ffn_gate: &l0.ffn_gate,
                ffn_up: &l0.ffn_up,
                ffn_down: &l0.ffn_down,
                ffn_post_norm: &l0.ffn_post_norm,
                ple: None,
                layer_output_scale: None,
            });
            assert_eq!(expected0.len(), got0.len());
            for (i, (a, b)) in expected0.iter().zip(got0.iter()).enumerate() {
                let tol = 1e-1 * a.abs().max(1.0);
                assert!(
                    (a - b).abs() <= tol,
                    "step {step}, layer 0 (owner): mismatch at index {i}: cpu={a} gpu={b}"
                );
            }

            let got1 = vulkan.fused_layer(FusedLayerInput {
                x: GpuInput::Cpu(&x1),
                attn_norm: &l1.attn_norm,
                wq: &l1.wq,
                q_norm: &l1.q_norm,
                kv: None,
                n_head,
                n_head_kv,
                head_dim,
                rope_dim,
                rope_freq_base,
                freq_factors: None,
                eps,
                pos,
                window_start: 0,
                scale,
                cache: &mut kv_cache.layers[0],
                wo: &l1.wo,
                attn_post_norm: &l1.attn_post_norm,
                ffn_norm: &l1.ffn_norm,
                ffn_gate: &l1.ffn_gate,
                ffn_up: &l1.ffn_up,
                ffn_down: &l1.ffn_down,
                ffn_post_norm: &l1.ffn_post_norm,
                ple: None,
                layer_output_scale: None,
            });
            assert_eq!(expected1.len(), got1.len());
            for (i, (a, b)) in expected1.iter().zip(got1.iter()).enumerate() {
                let tol = 1e-1 * a.abs().max(1.0);
                assert!(
                    (a - b).abs() <= tol,
                    "step {step}, layer 1 (donor): mismatch at index {i}: cpu={a} gpu={b}"
                );
            }
        }
    }
}
