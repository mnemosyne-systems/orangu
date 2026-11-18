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

//! The bind-group layouts every compute pipeline is built against.
//!
//! One function per binding *shape*, not per kernel: `rmsnorm`'s
//! `(x, weight, y, meta)` has the same shape as `add`/`mul`'s
//! `(a, b, y, meta)`, so three kernels share one layout and one pipeline
//! layout rather than declaring the same thing three times. Grouped here
//! because the shapes are what the WGSL `@group(0) @binding(n)` declarations
//! have to agree with, and a mismatch is a validation error at pipeline
//! creation with a message that names a binding index and nothing else.

/// Bind group layout shared by every type's pipeline: `weights` (storage,
/// read-only, the raw quantized bytes), `x` (storage, read-only, the
/// input activations), `y` (storage, read-write, the output), `meta`
/// (uniform, the shapes — see `vulkan_shaders::PRELUDE`'s `Meta` struct).
/// [`bind_group_layout`] plus binding 5: the token and row tables of an
/// **indexed** integer-dot GEMM (`vulkan_shaders::shader_source_mmq_indexed`),
/// which multiplies a stack of expert matrices by the rows each expert was
/// routed in one dispatch.
pub(super) fn indexed_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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
        label: Some("orangu-server indexed matmul bind group layout"),
        entries: &[
            entry(0, storage(true)),
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
            entry(4, storage(true)),
            entry(5, storage(true)),
        ],
    })
}

pub(super) fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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
            // The `IQ*` codebooks (`vulkan_shaders::IQ_GRID_PRELUDE`). Only
            // the five `IQ*` types' shaders declare this binding; it is in
            // the shared layout regardless because a bind group layout may
            // carry entries a shader never reads — only the reverse is
            // rejected — and one layout for every matmul pipeline is worth
            // more than saving a ~15 KiB binding on the other eleven.
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: storage(true),
                count: None,
            },
        ],
    })
}

/// Bind group layout for the binary elementwise/norm shaders (`add`, `mul`,
/// `rmsnorm`): two read-only storage buffers, one read-write storage
/// buffer, one uniform — `rmsnorm`'s `(x, weight, y, meta)` happens to have
/// the exact same binding shape as `add`/`mul`'s `(a, b, y, meta)`, so all
/// three share one layout and one pipeline layout.
pub(super) fn elem4_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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

/// Bind group layout for `rmsnorm_add_pipeline` — RMSNorm fused with an
/// immediately-following residual add, in one dispatch: `x`/`weight` (both
/// read-only storage, RMSNorm's own inputs), `residual` (read-only
/// storage, the value added after normalizing), `y` (read-write storage,
/// the final post-add output), `meta` (uniform). See `shader_source_
/// rmsnorm_add`'s own doc comment for why this exists as a fifth binding
/// rather than reusing `elem4_bind_group_layout`.
pub(super) fn elem5_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let storage = |read_only: bool| wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only },
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("orangu-server elem5 bind group layout"),
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
                ty: storage(true),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: storage(false),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
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
pub(super) fn elem3_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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
pub(super) fn elem2_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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
pub(super) fn attn_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    attn_layout(device, false)
}

/// The same, plus binding 6 — the block table a paged attention kernel reads to
/// turn a cached position into a row (`vulkan_shaders::KvPaging::Paged`).
///
/// A separate layout rather than one with an optional entry, because a bind
/// group layout is part of a pipeline's identity: a kernel compiled without the
/// binding cannot be given one, and a kernel compiled with it must always be.
/// Two layouts make that a compile-time fact about which pipeline is in use,
/// where an "optional" binding would make it a run-time hope.
pub(super) fn attn_paged_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    attn_layout(device, true)
}

fn attn_layout(device: &wgpu::Device, paged: bool) -> wgpu::BindGroupLayout {
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
        ]
        .iter()
        .copied()
        .chain(paged.then(|| entry(6, storage(true))))
        .collect::<Vec<_>>(),
    })
}

/// Bind group layout for `vulkan_shaders::ARGMAX_PENALTY_SHADER` —
/// `logits` (storage, read-write: mutated in place by the repeat-penalty
/// step), `recent_tokens` (storage, read-only), `out_token` (storage,
/// read-write, one `u32`, unused by this phase but still bound — same
/// layout the whole `record_argmax_sample` chain shares this bind group
/// for), `meta` (uniform).
pub(super) fn argmax_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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

/// Bind group layout for `vulkan_shaders::ARGMAX_SPLIT_SHADER` — `logits`
/// (storage, read-only: the penalty phase already ran), `partial_val`/
/// `partial_idx` (storage, read-write — each of `ARGMAX_SPLIT_N`
/// workgroups writes its own slot), `meta` (uniform). Distinct from
/// `argmax_bind_group_layout` above: binding 1 is read-write here
/// (`partial_val`, an output), read-only there (`recent_tokens`, an
/// input) — the two shapes don't coincide the way `elem4_bind_group_
/// layout` happens to fit the merge phase (see `ARGMAX_REDUCE_SHADER_
/// BODY`'s own doc comment).
pub(super) fn argmax_split_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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
        label: Some("orangu-server argmax split bind group layout"),
        entries: &[
            entry(0, storage(true)),
            entry(1, storage(false)),
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

/// Bind group layout for `vulkan_shaders::KV_EPILOGUE_SHADER`: the key and
/// value projections (read-write, processed in place), the K norm weight
/// and the RoPE divisors (read-only), the key and value regions of the KV
/// mirror (read-write), and the meta.
pub(super) fn kv_epilogue_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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
        label: Some("orangu-server kv epilogue bind group layout"),
        entries: &[
            entry(0, storage(false)),
            entry(1, storage(false)),
            entry(2, storage(true)),
            entry(3, storage(true)),
            entry(4, storage(false)),
            entry(5, storage(false)),
            entry(
                6,
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            ),
        ],
    })
}

/// Bind group layout for `vulkan_shaders::shader_source_rmsnorm_add_norm_wide`:
/// `x`, `w1`, `residual`, `w2` read-only, `y1` and `y2` read-write, the
/// meta, and the 8-bit form of `y2` (`q8`, written when the meta's `aux`
/// asks for it).
pub(super) fn norm_pair_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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
        label: Some("orangu-server norm pair bind group layout"),
        entries: &[
            entry(0, storage(true)),
            entry(1, storage(true)),
            entry(2, storage(true)),
            entry(3, storage(true)),
            entry(4, storage(false)),
            entry(5, storage(false)),
            entry(
                6,
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            ),
            entry(7, storage(false)),
        ],
    })
}

/// Bind group layout for `add_rmsnorm_wide_pipeline` — the pre-norm pair
/// `x_out = x + residual; y = rmsnorm(x_out) · weight` in one dispatch:
/// `x`, `weight`, `residual` read-only, `x_out` and `y` read-write,
/// `meta` uniform. See `shader_source_add_rmsnorm_wide`.
pub(super) fn elem6_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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
        label: Some("orangu-server elem6 bind group layout"),
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

/// Bind group layout for `record_recurrent_rows`'s two kernels:
/// `elem6_bind_group_layout`'s six, and a seventh read-write binding for the
/// state behind each row (`VulkanStateMirror::row_states`). See
/// `vulkan_shaders::shader_source_gated_delta_rows`.
pub(super) fn recurrent_rows_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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
        label: Some("orangu-server recurrent rows bind group layout"),
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
            entry(6, storage(false)),
        ],
    })
}
