//! Raw-Vulkan decode-submit path — persistent command-buffer reuse.
//!
//! # Why this exists
//!
//! orangu records its whole decode graph into fresh `wgpu` command encoders
//! every token and submits them. RADV then rebuilds and re-validates the
//! entire buffer-object (BO) list on every `queue.submit()` — the single
//! biggest CPU bucket the decode loop spends that llama.cpp does not, because
//! llama reuses recorded command buffers and RADV caches the BO list.
//!
//! `wgpu` has no command-buffer-reuse primitive, and its `VkDescriptorSet`s /
//! `VkPipeline`s are hidden behind `wgpu-core`'s private hub. But everything a
//! *standalone* raw-Vulkan compute path needs is reachable through stock
//! `wgpu 30`'s public `as_hal` surface:
//!
//! - [`wgpu::Device::as_hal`] → the raw `ash::Device`, `VkQueue`,
//!   `VkPhysicalDevice`, queue-family index, and `ash::Instance`.
//! - [`wgpu::Buffer::as_hal`] → the raw `VkBuffer` behind any `wgpu` buffer
//!   (weights, KV, activation arenas), so this path *shares* orangu's existing
//!   GPU-resident buffers rather than copying them.
//! - [`wgpu::naga`] (re-exported, `spv-out` enabled by the `vulkan` feature)
//!   compiles orangu's own WGSL to SPIR-V, so this path builds its own
//!   `VkPipeline`/`VkDescriptorSet`s with no `wgpu-core` fork.
//!
//! The persistent command buffer records the decode graph once; per token only
//! the host-visible uniform contents (`pos`/`n_pos`) change and the same
//! `VkCommandBuffer` is resubmitted via raw `vkQueueSubmit` — no `wgpu` submit
//! on the hot path, so RADV never rebuilds the BO list.
//!
//! This module is the foundation: raw-context bootstrap from `wgpu`, WGSL→SPIR-V
//! compilation, and the pipeline / descriptor / command-buffer primitives, each
//! proven byte-identical against a reference before the decode graph is built
//! on top of them.

#![allow(dead_code)]

use std::ffi::CString;

use ash::vk;

/// The `wgpu-hal` API marker for the Vulkan backend — `as_hal::<Vulkan>()`
/// yields the `wgpu_hal::vulkan::*` types this module reaches into.
type Vulkan = wgpu::hal::api::Vulkan;

/// An owned handle onto the raw Vulkan objects underneath a `wgpu::Device`.
///
/// `ash::Device`/`ash::Instance` are cheap clonable handle+fn-pointer bundles
/// and `VkQueue`/`VkPhysicalDevice` are `Copy`, so we clone everything out of
/// the transient `as_hal` guard once and drop the guard — the `wgpu::Device`
/// keeps the underlying `VkDevice` alive for the whole session.
pub struct ReplayContext {
    pub instance: ash::Instance,
    pub device: ash::Device,
    pub phys: vk::PhysicalDevice,
    pub queue: vk::Queue,
    pub queue_family_index: u32,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    /// The device's Vulkan API version — drives the SPIR-V `lang_version` and
    /// workgroup-zero-init mode so the raw path's `naga`-compiled pipelines are
    /// byte-for-byte the SPIR-V wgpu-hal would emit for the same WGSL (see
    /// [`compile_wgsl_to_spirv`]), keeping greedy output bit-identical.
    api_version: u32,
}

impl ReplayContext {
    /// Bootstraps a raw context from a live `wgpu::Device`/`wgpu::Queue`.
    ///
    /// Returns `None` if the device is not the Vulkan backend (e.g. a test
    /// runner on a non-Vulkan adapter), so callers gate the replay path on it.
    ///
    /// # Safety
    /// The returned handles alias `wgpu`'s. The caller must keep the source
    /// `wgpu::Device` alive for the lifetime of this context, and must not
    /// submit to `queue` concurrently with `wgpu` (both require external
    /// synchronization on the `VkQueue`).
    pub unsafe fn from_wgpu(device: &wgpu::Device) -> Option<ReplayContext> {
        let hal = unsafe { device.as_hal::<Vulkan>()? };
        let instance = hal.shared_instance().raw_instance().clone();
        let phys = hal.raw_physical_device();
        let ash_device = hal.raw_device().clone();
        let queue = hal.raw_queue();
        let queue_family_index = hal.queue_family_index();
        let mem_props = unsafe { instance.get_physical_device_memory_properties(phys) };
        let api_version = unsafe { instance.get_physical_device_properties(phys) }.api_version;
        Some(ReplayContext {
            instance,
            device: ash_device,
            phys,
            queue,
            queue_family_index,
            mem_props,
            api_version,
        })
    }

    /// Picks a memory type index satisfying `type_bits` (from a buffer's
    /// `MemoryRequirements`) and containing all of `flags`.
    fn find_memory_type(&self, type_bits: u32, flags: vk::MemoryPropertyFlags) -> Option<u32> {
        (0..self.mem_props.memory_type_count).find(|&i| {
            let supported = type_bits & (1 << i) != 0;
            let props = self.mem_props.memory_types[i as usize].property_flags;
            supported && props.contains(flags)
        })
    }
}

/// Compiles WGSL source to SPIR-V words via `wgpu`'s bundled `naga`.
///
/// The entry point must be named `main`. `@group(0) @binding(n)` in the WGSL
/// becomes descriptor set 0, binding `n` — matching the single-set layout this
/// module builds per pipeline.
pub fn compile_wgsl_to_spirv(src: &str, api_version: u32) -> Result<Vec<u32>, String> {
    use wgpu::naga;
    let module = naga::front::wgsl::parse_str(src).map_err(|e| format!("wgsl parse: {e:?}"))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| format!("wgsl validate: {e:?}"))?;
    // Drive naga's SPIR-V writer the *exact* way `wgpu-hal`'s Vulkan backend does
    // (`wgpu_hal::vulkan::adapter`'s `spv::Options`), so this path compiles the
    // same WGSL to the same SPIR-V wgpu already runs — which keeps the raw
    // replay's greedy output bit-identical to the wgpu path (a mismatched
    // `lang_version`/zero-init produces a different reduction encoding whose
    // <1e-5 rounding differences flip rare near-ties, e.g. an EOS decision).
    //  - `lang_version`: the max SPIR-V version for the device's Vulkan version,
    //    exactly as wgpu picks it. SPIR-V 1.0 (naga's default) is also *invalid*
    //    for `var<workgroup>` arrays (VUID-StandaloneSpirv-None-10684).
    //  - `zero_initialize_workgroup_memory`: `Native` once the device supports
    //    it (Vulkan 1.3 promotes `VK_KHR_zero_initialize_workgroup_memory`),
    //    else `Polyfill` — matching wgpu.
    //  - `bounds_check_policies`: `Restrict` (clamp), as wgpu uses, so an
    //    out-of-range index is a defined clamp, not a GPU fault.
    let lang_version = match api_version {
        v if v >= vk::API_VERSION_1_3 => (1, 6),
        v if v >= vk::API_VERSION_1_2 => (1, 5),
        v if v >= vk::API_VERSION_1_1 => (1, 3),
        _ => (1, 0),
    };
    let zero_init = if api_version >= vk::API_VERSION_1_3 {
        naga::back::spv::ZeroInitializeWorkgroupMemoryMode::Native
    } else {
        naga::back::spv::ZeroInitializeWorkgroupMemoryMode::Polyfill
    };
    let options = naga::back::spv::Options {
        lang_version,
        bounds_check_policies: naga::proc::BoundsCheckPolicies {
            index: naga::proc::BoundsCheckPolicy::Restrict,
            buffer: naga::proc::BoundsCheckPolicy::Restrict,
            image_load: naga::proc::BoundsCheckPolicy::Restrict,
            binding_array: naga::proc::BoundsCheckPolicy::Unchecked,
        },
        zero_initialize_workgroup_memory: zero_init,
        force_loop_bounding: true,
        ..Default::default()
    };
    naga::back::spv::write_vec(&module, &info, &options, None)
        .map_err(|e| format!("spv write: {e:?}"))
}

/// A raw host-visible, coherent, persistently-mapped buffer we own outright —
/// used for the per-token uniforms (`pos`/`n_pos`) so an update is a plain
/// `memcpy` with no staging and no `wgpu` submit.
pub struct MappedBuffer {
    pub buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    ptr: *mut u8,
    pub size: vk::DeviceSize,
}

impl MappedBuffer {
    /// # Safety
    /// `ctx` must outlive the returned buffer; destroy via [`Self::destroy`].
    pub unsafe fn new(
        ctx: &ReplayContext,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> Result<MappedBuffer, String> {
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { ctx.device.create_buffer(&info, None) }
            .map_err(|e| format!("create_buffer: {e}"))?;
        let req = unsafe { ctx.device.get_buffer_memory_requirements(buffer) };
        let mem_type = ctx
            .find_memory_type(
                req.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .ok_or("no host-visible|coherent memory type")?;
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(mem_type);
        let memory = unsafe { ctx.device.allocate_memory(&alloc, None) }
            .map_err(|e| format!("allocate_memory: {e}"))?;
        unsafe { ctx.device.bind_buffer_memory(buffer, memory, 0) }
            .map_err(|e| format!("bind_buffer_memory: {e}"))?;
        let ptr = unsafe {
            ctx.device
                .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
        }
        .map_err(|e| format!("map_memory: {e}"))? as *mut u8;
        Ok(MappedBuffer {
            buffer,
            memory,
            ptr,
            size,
        })
    }

    /// Copies `bytes` into the mapped region at offset 0. Coherent memory, so
    /// the write is visible to the device at the next queue submit with no
    /// explicit flush.
    ///
    /// # Safety
    /// `bytes.len()` must be `<= self.size`.
    pub unsafe fn write(&self, bytes: &[u8]) {
        debug_assert!(bytes.len() as vk::DeviceSize <= self.size);
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.ptr, bytes.len());
        }
    }

    /// Copies `bytes` into the mapped region at `offset` — patches one field of
    /// a per-token uniform without rewriting the whole struct.
    ///
    /// # Safety
    /// `offset + bytes.len()` must be `<= self.size`.
    pub unsafe fn write_at(&self, offset: usize, bytes: &[u8]) {
        debug_assert!((offset + bytes.len()) as vk::DeviceSize <= self.size);
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.ptr.add(offset), bytes.len());
        }
    }

    /// Reads `len` bytes back out of the mapped region (host-visible buffers
    /// used as compute outputs in tests).
    ///
    /// # Safety
    /// `len` must be `<= self.size`.
    pub unsafe fn read(&self, len: usize) -> Vec<u8> {
        let mut out = vec![0u8; len];
        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr, out.as_mut_ptr(), len);
        }
        out
    }

    /// # Safety
    /// Must not be in use by any in-flight submission.
    pub unsafe fn destroy(self, ctx: &ReplayContext) {
        unsafe {
            ctx.device.unmap_memory(self.memory);
            ctx.device.destroy_buffer(self.buffer, None);
            ctx.device.free_memory(self.memory, None);
        }
    }
}

/// A compute pipeline plus the single descriptor-set-0 layout it binds. Owns
/// the `VkShaderModule` (kept for the pipeline's lifetime; safe to destroy
/// after creation but we hold it to be explicit).
pub struct ComputeProgram {
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub set_layout: vk::DescriptorSetLayout,
    module: vk::ShaderModule,
}

/// One entry in a pipeline's descriptor-set-0 layout.
#[derive(Clone, Copy)]
pub struct BindingSpec {
    pub binding: u32,
    pub kind: vk::DescriptorType,
}

impl ComputeProgram {
    /// Builds a compute pipeline from WGSL and an explicit set-0 binding
    /// layout (which must match the shader's `@binding` decorations).
    ///
    /// # Safety
    /// `ctx` must outlive the program; destroy via [`Self::destroy`].
    pub unsafe fn from_wgsl(
        ctx: &ReplayContext,
        wgsl: &str,
        bindings: &[BindingSpec],
    ) -> Result<ComputeProgram, String> {
        let spirv = compile_wgsl_to_spirv(wgsl, ctx.api_version)?;
        unsafe { Self::from_spirv(ctx, &spirv, bindings) }
    }

    /// # Safety
    /// `ctx` must outlive the program; destroy via [`Self::destroy`].
    pub unsafe fn from_spirv(
        ctx: &ReplayContext,
        spirv: &[u32],
        bindings: &[BindingSpec],
    ) -> Result<ComputeProgram, String> {
        let module_info = vk::ShaderModuleCreateInfo::default().code(spirv);
        let module = unsafe { ctx.device.create_shader_module(&module_info, None) }
            .map_err(|e| format!("create_shader_module: {e}"))?;

        let layout_bindings: Vec<vk::DescriptorSetLayoutBinding> = bindings
            .iter()
            .map(|b| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b.binding)
                    .descriptor_type(b.kind)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let set_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&layout_bindings);
        let set_layout = unsafe {
            ctx.device
                .create_descriptor_set_layout(&set_layout_info, None)
        }
        .map_err(|e| format!("create_descriptor_set_layout: {e}"))?;

        let set_layouts = [set_layout];
        let pl_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        let pipeline_layout = unsafe { ctx.device.create_pipeline_layout(&pl_info, None) }
            .map_err(|e| format!("create_pipeline_layout: {e}"))?;

        let entry = CString::new("main").unwrap();
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(&entry);
        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pipeline_layout);
        let pipeline = unsafe {
            ctx.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
        }
        .map_err(|(_, e)| format!("create_compute_pipelines: {e}"))?[0];

        Ok(ComputeProgram {
            pipeline,
            pipeline_layout,
            set_layout,
            module,
        })
    }

    /// # Safety
    /// Must not be in use by any in-flight submission.
    pub unsafe fn destroy(self, ctx: &ReplayContext) {
        unsafe {
            ctx.device.destroy_pipeline(self.pipeline, None);
            ctx.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            ctx.device
                .destroy_descriptor_set_layout(self.set_layout, None);
            ctx.device.destroy_shader_module(self.module, None);
        }
    }
}

/// One binding in a replay op's descriptor set 0 — a raw `VkBuffer` sub-range.
/// The buffer is either an orangu wgpu buffer (via `Buffer::as_hal` →
/// `raw_handle`) or one of the replay path's own [`MappedBuffer`]s; `offset`
/// carries the `BindSrc::Slice` base for KV/arena sub-ranges.
#[derive(Clone, Copy)]
pub struct OpBinding {
    pub binding: u32,
    pub kind: vk::DescriptorType,
    pub buffer: vk::Buffer,
    pub offset: vk::DeviceSize,
    pub size: vk::DeviceSize,
}

/// One recorded dispatch: which program, the buffers it binds, and its
/// workgroup grid. Mirrors a single `pass.set_pipeline` + `set_bind_group` +
/// `dispatch_workgroups` in the wgpu path.
#[derive(Clone)]
pub struct ReplayOp {
    pub program: usize,
    pub bindings: Vec<OpBinding>,
    pub groups: [u32; 3],
}

/// One recorded step of the decode graph: either a compute dispatch or a
/// buffer-to-buffer copy. orangu's fused decode path moves activations between
/// per-op arena buffers with `copy_buffer_to_buffer` (residual snapshot,
/// normed→QKV inputs, attn-out→wo input, ffn_normed→gate/up inputs), so the
/// replay records those same copies inline — their offsets are static, so they
/// replay unchanged every token.
#[derive(Clone)]
pub enum ReplayStep {
    Dispatch(ReplayOp),
    Copy {
        src: vk::Buffer,
        src_offset: vk::DeviceSize,
        dst: vk::Buffer,
        dst_offset: vk::DeviceSize,
        size: vk::DeviceSize,
    },
}

/// A recorded decode graph replayed from a persistent `VkCommandBuffer`.
///
/// The op sequence is recorded **once** into `cmd_buffer` (descriptor sets
/// pointing at the same buffers every token, per-token state living in
/// host-visible uniforms bound into those sets). Each token,
/// [`Self::run_token`] just resubmits the same command buffer — so RADV never
/// rebuilds the BO list, the entire point of the exercise.
///
/// Barriers between ops are conservative full compute→compute memory barriers.
/// Decode is CPU-bound (the GPU sits idle waiting on the CPU), so
/// over-synchronizing costs nothing measurable while guaranteeing correctness.
pub struct ReplayGraph {
    descriptor_pool: vk::DescriptorPool,
    cmd_pool: vk::CommandPool,
    cmd_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    /// Host-visible per-token uniform buffers (from `from_capture`) and the
    /// fields to patch each token. Empty for graphs built via `build`/
    /// `build_steps` directly. See [`Self::update_per_token`].
    per_token: Vec<(MappedBuffer, Vec<PerTokenField>)>,
}

// SAFETY: the raw handles / mapped pointers a `ReplayGraph` (and its
// `MappedBuffer`s) own are only ever touched from the single decode thread that
// built it — the decode loop drives one sequence's replay serially, never
// sharing a graph across threads. Storing it behind a `Mutex` on the model
// (which must be `Send`/`Sync`) needs these markers; the `Mutex` also enforces
// the "no concurrent access" invariant the raw `VkQueue`/mapped memory require.
unsafe impl Send for ReplayGraph {}
unsafe impl Sync for ReplayGraph {}
// SAFETY: same rationale — `MappedBuffer`'s `*mut u8` is host-coherent GPU
// memory written only from the owning decode thread.
unsafe impl Send for MappedBuffer {}
unsafe impl Sync for MappedBuffer {}
// SAFETY: `ReplayContext` holds `ash` handles (`ash::Device`/`Instance` are
// themselves `Send`/`Sync`) plus `Copy` Vulkan handles.
unsafe impl Send for ReplayContext {}
unsafe impl Sync for ReplayContext {}
// SAFETY: `ComputeProgram` holds only `Copy` Vulkan handles.
unsafe impl Send for ComputeProgram {}
unsafe impl Sync for ComputeProgram {}

impl ReplayGraph {
    /// Records a dispatch-only op list. Convenience wrapper over
    /// [`Self::build_steps`] for graphs with no inline copies.
    ///
    /// # Safety
    /// `ctx`, `programs`, and every buffer referenced by `ops` must outlive the
    /// graph. Destroy via [`Self::destroy`] before any of them.
    pub unsafe fn build(
        ctx: &ReplayContext,
        programs: &[ComputeProgram],
        ops: &[ReplayOp],
    ) -> Result<ReplayGraph, String> {
        let steps: Vec<ReplayStep> = ops.iter().cloned().map(ReplayStep::Dispatch).collect();
        unsafe { Self::build_steps(ctx, programs, &steps) }
    }

    /// Records `steps` (dispatches + inline copies) into one persistent command
    /// buffer: a descriptor set per dispatch, a conservative full memory barrier
    /// between every step, and `cmd_copy_buffer` for copy steps.
    ///
    /// # Safety
    /// `ctx`, `programs`, and every buffer referenced by `steps` must outlive
    /// the graph. Destroy via [`Self::destroy`] before any of them.
    pub unsafe fn build_steps(
        ctx: &ReplayContext,
        programs: &[ComputeProgram],
        steps: &[ReplayStep],
    ) -> Result<ReplayGraph, String> {
        unsafe {
            // Size the descriptor pool from the dispatch steps' binding kinds.
            let mut n_storage = 0u32;
            let mut n_uniform = 0u32;
            let mut n_dispatch = 0u32;
            for step in steps {
                if let ReplayStep::Dispatch(op) = step {
                    n_dispatch += 1;
                    for b in &op.bindings {
                        match b.kind {
                            vk::DescriptorType::STORAGE_BUFFER => n_storage += 1,
                            vk::DescriptorType::UNIFORM_BUFFER => n_uniform += 1,
                            _ => {
                                return Err(format!("unsupported descriptor type {:?}", b.kind));
                            }
                        }
                    }
                }
            }
            let mut pool_sizes = Vec::new();
            if n_storage > 0 {
                pool_sizes.push(
                    vk::DescriptorPoolSize::default()
                        .ty(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(n_storage),
                );
            }
            if n_uniform > 0 {
                pool_sizes.push(
                    vk::DescriptorPoolSize::default()
                        .ty(vk::DescriptorType::UNIFORM_BUFFER)
                        .descriptor_count(n_uniform),
                );
            }
            let descriptor_pool = ctx
                .device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(n_dispatch.max(1))
                        .pool_sizes(&pool_sizes),
                    None,
                )
                .map_err(|e| format!("create_descriptor_pool: {e}"))?;

            // One descriptor set per dispatch step (aligned to `steps` by index;
            // `None` for copy steps).
            let mut step_sets: Vec<Option<vk::DescriptorSet>> = Vec::with_capacity(steps.len());
            for step in steps {
                let ReplayStep::Dispatch(op) = step else {
                    step_sets.push(None);
                    continue;
                };
                let layout = [programs[op.program].set_layout];
                let set = ctx
                    .device
                    .allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default()
                            .descriptor_pool(descriptor_pool)
                            .set_layouts(&layout),
                    )
                    .map_err(|e| format!("allocate_descriptor_sets: {e}"))?[0];
                // `infos` must stay alive across `update_descriptor_sets`, so
                // build them all first, then reference each by slice.
                let infos: Vec<vk::DescriptorBufferInfo> = op
                    .bindings
                    .iter()
                    .map(|b| {
                        vk::DescriptorBufferInfo::default()
                            .buffer(b.buffer)
                            .offset(b.offset)
                            .range(b.size)
                    })
                    .collect();
                let writes: Vec<vk::WriteDescriptorSet> = op
                    .bindings
                    .iter()
                    .enumerate()
                    .map(|(i, b)| {
                        vk::WriteDescriptorSet::default()
                            .dst_set(set)
                            .dst_binding(b.binding)
                            .descriptor_type(b.kind)
                            .buffer_info(std::slice::from_ref(&infos[i]))
                    })
                    .collect();
                ctx.device.update_descriptor_sets(&writes, &[]);
                step_sets.push(Some(set));
            }

            // Record the persistent command buffer.
            let cmd_pool = ctx
                .device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(ctx.queue_family_index),
                    None,
                )
                .map_err(|e| format!("create_command_pool: {e}"))?;
            let cmd_buffer = ctx
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(cmd_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .map_err(|e| format!("allocate_command_buffers: {e}"))?[0];

            ctx.device
                .begin_command_buffer(cmd_buffer, &vk::CommandBufferBeginInfo::default())
                .map_err(|e| format!("begin_command_buffer: {e}"))?;
            // Entry barrier: make the transfer write from *before* this
            // submission visible to the first shader read — the decode loop
            // uploads each token's input embedding into a captured buffer via a
            // separate `wgpu` transfer submit just before `run_token`, and
            // without this the first `attn_norm` could read the previous token's
            // input. (Per-token uniforms are host-coherent, already visible at
            // submit — no `HOST` stage here, which RADV can device-lost on.)
            {
                let entry = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ);
                ctx.device.cmd_pipeline_barrier(
                    cmd_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[entry],
                    &[],
                    &[],
                );
            }
            for (i, step) in steps.iter().enumerate() {
                if i > 0 {
                    // Conservative full barrier covering both compute and copy:
                    // any step may read what any prior dispatch or copy wrote.
                    let barrier = vk::MemoryBarrier::default()
                        .src_access_mask(
                            vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE,
                        )
                        .dst_access_mask(
                            vk::AccessFlags::SHADER_READ
                                | vk::AccessFlags::SHADER_WRITE
                                | vk::AccessFlags::TRANSFER_READ
                                | vk::AccessFlags::TRANSFER_WRITE,
                        );
                    ctx.device.cmd_pipeline_barrier(
                        cmd_buffer,
                        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[barrier],
                        &[],
                        &[],
                    );
                }
                match step {
                    ReplayStep::Dispatch(op) => {
                        let prog = &programs[op.program];
                        ctx.device.cmd_bind_pipeline(
                            cmd_buffer,
                            vk::PipelineBindPoint::COMPUTE,
                            prog.pipeline,
                        );
                        ctx.device.cmd_bind_descriptor_sets(
                            cmd_buffer,
                            vk::PipelineBindPoint::COMPUTE,
                            prog.pipeline_layout,
                            0,
                            &[step_sets[i].expect("dispatch step has a set")],
                            &[],
                        );
                        ctx.device.cmd_dispatch(
                            cmd_buffer,
                            op.groups[0],
                            op.groups[1],
                            op.groups[2],
                        );
                    }
                    ReplayStep::Copy {
                        src,
                        src_offset,
                        dst,
                        dst_offset,
                        size,
                    } => {
                        let region = vk::BufferCopy::default()
                            .src_offset(*src_offset)
                            .dst_offset(*dst_offset)
                            .size(*size);
                        ctx.device
                            .cmd_copy_buffer(cmd_buffer, *src, *dst, &[region]);
                    }
                }
            }
            // Final barrier: make every write available/visible to whatever
            // reads the outputs next — a subsequent `wgpu` readback of the logits
            // on the production path, or a host-mapped read in the cross-checks.
            let final_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
            ctx.device.cmd_pipeline_barrier(
                cmd_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[final_barrier],
                &[],
                &[],
            );
            ctx.device
                .end_command_buffer(cmd_buffer)
                .map_err(|e| format!("end_command_buffer: {e}"))?;

            let fence = ctx
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(|e| format!("create_fence: {e}"))?;

            Ok(ReplayGraph {
                descriptor_pool,
                cmd_pool,
                cmd_buffer,
                fence,
                per_token: Vec::new(),
            })
        }
    }

    /// Patches every per-token uniform for the current token — a `memcpy` into
    /// each host-visible buffer (coherent memory), no wgpu submit. Call before
    /// [`Self::run_token`]. `n_pos` = attended positions, `window_start` = SWA
    /// window base (0 for full-attention layers).
    ///
    /// # Safety
    /// The graph's per-token buffers must not be in use by an in-flight submit.
    pub unsafe fn update_per_token(&self, pos: u32) {
        for (buf, fields) in &self.per_token {
            for f in fields {
                f.apply(buf, pos);
            }
        }
    }

    /// Resubmits the recorded command buffer and blocks until it completes.
    /// The caller must have already written this token's host-visible uniforms
    /// (the descriptor sets already point at them).
    ///
    /// # Safety
    /// No other submission to `ctx.queue` may be in flight concurrently
    /// (external `VkQueue` synchronization).
    pub unsafe fn run_token(&self, ctx: &ReplayContext) -> Result<(), String> {
        unsafe {
            // The raw `VkQueue` is shared with wgpu, which requires external
            // synchronization: no other submission may be in flight when we
            // submit. The caller's per-token `wgpu` transfer (the embedding
            // upload) is polled to completion first, but `device_wait_idle`
            // guarantees the queue is fully drained of any wgpu work before this
            // raw submit — sharing a `VkQueue` across the two submitters
            // otherwise races and faults the driver non-deterministically.
            ctx.device
                .device_wait_idle()
                .map_err(|e| format!("device_wait_idle (pre): {e}"))?;
            ctx.device
                .reset_fences(&[self.fence])
                .map_err(|e| format!("reset_fences: {e}"))?;
            let cbs = [self.cmd_buffer];
            ctx.device
                .queue_submit(
                    ctx.queue,
                    &[vk::SubmitInfo::default().command_buffers(&cbs)],
                    self.fence,
                )
                .map_err(|e| format!("queue_submit: {e}"))?;
            ctx.device
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .map_err(|e| format!("wait_for_fences: {e}"))?;
            Ok(())
        }
    }

    /// # Safety
    /// Must not be in use by any in-flight submission.
    pub unsafe fn destroy(self, ctx: &ReplayContext) {
        unsafe {
            ctx.device.destroy_fence(self.fence, None);
            ctx.device.destroy_command_pool(self.cmd_pool, None);
            ctx.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            for (buf, _) in self.per_token {
                buf.destroy(ctx);
            }
        }
    }

    /// Validation layer for the captured op-list: checks every binding/copy
    /// range fits its `wgpu::Buffer`, and that per-token uniform bindings are
    /// large enough for their patched fields. Returns a precise `Err` (step
    /// index, binding, offset/size vs buffer size) instead of letting an
    /// out-of-bounds descriptor reach the GPU as a silent fault.
    fn validate_capture(steps: &[CaptureStep]) -> Result<(), String> {
        let fits = |what: &str, buf: &wgpu::Buffer, offset: u64, size: u64| -> Result<(), String> {
            let cap = buf.size();
            if offset + size > cap {
                Err(format!(
                    "{what}: range [{offset}, {}) exceeds buffer size {cap}",
                    offset + size
                ))
            } else if size == 0 {
                Err(format!("{what}: zero-size binding"))
            } else {
                Ok(())
            }
        };
        for (i, step) in steps.iter().enumerate() {
            match step {
                CaptureStep::Dispatch {
                    bindings,
                    per_token,
                    groups,
                    ..
                } => {
                    if groups.contains(&0) {
                        return Err(format!(
                            "step {i}: dispatch has a zero workgroup dim {groups:?}"
                        ));
                    }
                    for b in bindings {
                        fits(
                            &format!("step {i} binding {}", b.binding),
                            &b.buffer,
                            b.offset,
                            b.size,
                        )?;
                    }
                    for p in per_token {
                        let need = p
                            .fields
                            .iter()
                            .map(|f| f.byte_offset() + 4)
                            .max()
                            .unwrap_or(0);
                        if p.init_bytes.len() < need {
                            return Err(format!(
                                "step {i} per-token binding {}: uniform is {} bytes but a field writes at {need}",
                                p.binding,
                                p.init_bytes.len()
                            ));
                        }
                    }
                }
                CaptureStep::Copy {
                    src,
                    src_offset,
                    dst,
                    dst_offset,
                    size,
                } => {
                    fits(&format!("step {i} copy src"), src, *src_offset, *size)?;
                    fits(&format!("step {i} copy dst"), dst, *dst_offset, *size)?;
                }
                CaptureStep::HostInput {
                    buffer,
                    offset,
                    size,
                    ..
                } => {
                    fits(&format!("step {i} host input"), buffer, *offset, *size)?;
                }
            }
        }
        Ok(())
    }

    /// Builds a graph directly from a [`CaptureStep`] list emitted by orangu's
    /// real decode recording (see `VulkanBackend`'s `decode_capture`). Compiles
    /// one pipeline per distinct kernel WGSL (deduped), resolves each captured
    /// `wgpu::Buffer` to its raw `VkBuffer` via `as_hal`, and records the graph.
    /// Returns the owned programs alongside the graph — the caller must keep
    /// both (and the source buffers) alive until [`Self::destroy`].
    ///
    /// # Safety
    /// Every `wgpu::Buffer` referenced by `steps` must outlive the returned
    /// graph and programs.
    pub unsafe fn from_capture(
        ctx: &ReplayContext,
        steps: &[CaptureStep],
    ) -> Result<(ReplayGraph, Vec<ComputeProgram>), String> {
        // Validation pass — a lightweight "validation layer" for the raw path
        // (no Vulkan validation layer is assumed installed). Every captured
        // binding/copy references a real `wgpu::Buffer` whose size we know, so
        // check every `offset + size` fits before recording — a violation would
        // otherwise be a silent GPU out-of-bounds / segfault deep in the driver,
        // impossible to attribute to a dispatch. Turns that into a precise error.
        Self::validate_capture(steps)?;
        unsafe {
            let mut program_wgsl: Vec<String> = Vec::new();
            let mut programs: Vec<ComputeProgram> = Vec::new();
            let mut replay_steps: Vec<ReplayStep> = Vec::with_capacity(steps.len());
            // Host-visible per-token uniforms, in emission order; attached to the
            // graph so `update_per_token` can patch them each token.
            let mut per_token: Vec<(MappedBuffer, Vec<PerTokenField>)> = Vec::new();

            let raw = |b: &wgpu::Buffer| -> Result<vk::Buffer, String> {
                b.as_hal::<Vulkan>()
                    .map(|h| h.raw_handle())
                    .ok_or_else(|| "buffer is not a Vulkan buffer".to_string())
            };

            for step in steps {
                match step {
                    CaptureStep::Dispatch {
                        wgsl,
                        bindings,
                        per_token: pt,
                        groups,
                    } => {
                        // Program binding layout = static bindings + per-token
                        // uniforms (which are always uniform buffers).
                        let idx = match program_wgsl.iter().position(|w| w == wgsl) {
                            Some(i) => i,
                            None => {
                                let mut specs: Vec<BindingSpec> = bindings
                                    .iter()
                                    .map(|b| BindingSpec {
                                        binding: b.binding,
                                        kind: b.kind.vk(),
                                    })
                                    .collect();
                                for p in pt {
                                    specs.push(BindingSpec {
                                        binding: p.binding,
                                        kind: vk::DescriptorType::UNIFORM_BUFFER,
                                    });
                                }
                                let prog = ComputeProgram::from_wgsl(ctx, wgsl, &specs)?;
                                programs.push(prog);
                                program_wgsl.push(wgsl.clone());
                                programs.len() - 1
                            }
                        };
                        let mut op_bindings: Vec<OpBinding> = bindings
                            .iter()
                            .map(|b| {
                                Ok(OpBinding {
                                    binding: b.binding,
                                    kind: b.kind.vk(),
                                    buffer: raw(&b.buffer)?,
                                    offset: b.offset,
                                    size: b.size,
                                })
                            })
                            .collect::<Result<_, String>>()?;
                        // Create a host-visible buffer for each per-token uniform,
                        // initialise it, and bind it.
                        for p in pt {
                            let buf = MappedBuffer::new(
                                ctx,
                                p.init_bytes.len() as vk::DeviceSize,
                                vk::BufferUsageFlags::UNIFORM_BUFFER,
                            )?;
                            buf.write(&p.init_bytes);
                            op_bindings.push(OpBinding {
                                binding: p.binding,
                                kind: vk::DescriptorType::UNIFORM_BUFFER,
                                buffer: buf.buffer,
                                offset: 0,
                                size: p.init_bytes.len() as vk::DeviceSize,
                            });
                            per_token.push((buf, p.fields.clone()));
                        }
                        replay_steps.push(ReplayStep::Dispatch(ReplayOp {
                            program: idx,
                            bindings: op_bindings,
                            groups: *groups,
                        }));
                    }
                    CaptureStep::Copy {
                        src,
                        src_offset,
                        dst,
                        dst_offset,
                        size,
                    } => {
                        replay_steps.push(ReplayStep::Copy {
                            src: raw(src)?,
                            src_offset: *src_offset,
                            dst: raw(dst)?,
                            dst_offset: *dst_offset,
                            size: *size,
                        });
                    }
                    // Per-token host uploads are replayed from the CPU each token
                    // (see the decode loop), not recorded into the command buffer.
                    CaptureStep::HostInput { .. } => {}
                }
            }

            let mut graph = Self::build_steps(ctx, &programs, &replay_steps)?;
            graph.per_token = per_token;
            Ok((graph, programs))
        }
    }
}

/// Whether a captured binding is a storage or uniform buffer — the backend-side
/// capture (`vulkan.rs`) uses this rather than depending on `ash` directly.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DescriptorKind {
    Storage,
    Uniform,
}

impl DescriptorKind {
    fn vk(self) -> vk::DescriptorType {
        match self {
            DescriptorKind::Storage => vk::DescriptorType::STORAGE_BUFFER,
            DescriptorKind::Uniform => vk::DescriptorType::UNIFORM_BUFFER,
        }
    }
}

/// One binding of a captured dispatch, as emitted by orangu's recording — a
/// `wgpu::Buffer` sub-range (resolved to a raw `VkBuffer` at graph-build time).
#[derive(Clone)]
pub struct CaptureBinding {
    pub binding: u32,
    pub kind: DescriptorKind,
    pub buffer: wgpu::Buffer,
    pub offset: u64,
    pub size: u64,
}

/// A single u32 field of a per-token uniform, patched each token by
/// [`ReplayGraph::update_per_token`] from `(pos, n_pos, window_start)`. The byte
/// offsets are into the uniform's `#[repr(C)]` struct (e.g.
/// `FusedNormRopeMeta.pos` at 12, `AttnSplitMeta.n_pos` at 16, `ElemMeta._pad0`
/// at 4).
#[derive(Clone, Copy)]
pub enum PerTokenField {
    /// u32 current position `pos`.
    Pos { byte_offset: usize },
    /// u32 number of attended positions. `window` is the layer's sliding-window
    /// size (`None` = full attention): `n_pos = min(pos+1, window)`.
    NPos {
        byte_offset: usize,
        window: Option<u32>,
    },
    /// u32 sliding-window start = `pos + 1 - n_pos` (0 for full attention).
    WindowStart {
        byte_offset: usize,
        window: Option<u32>,
    },
    /// u32 KV element write offset `pos * kv_dim` (the KV-cast destination).
    KvWriteOffset { byte_offset: usize, kv_dim: u32 },
}

impl PerTokenField {
    fn byte_offset(self) -> usize {
        match self {
            PerTokenField::Pos { byte_offset }
            | PerTokenField::NPos { byte_offset, .. }
            | PerTokenField::WindowStart { byte_offset, .. }
            | PerTokenField::KvWriteOffset { byte_offset, .. } => byte_offset,
        }
    }

    fn apply(self, buf: &MappedBuffer, pos: u32) {
        let n_pos = |window: Option<u32>| window.map_or(pos + 1, |w| (pos + 1).min(w));
        let (off, val) = match self {
            PerTokenField::Pos { byte_offset } => (byte_offset, pos),
            PerTokenField::NPos {
                byte_offset,
                window,
            } => (byte_offset, n_pos(window)),
            PerTokenField::WindowStart {
                byte_offset,
                window,
            } => (byte_offset, pos + 1 - n_pos(window)),
            PerTokenField::KvWriteOffset {
                byte_offset,
                kv_dim,
            } => (byte_offset, pos * kv_dim),
        };
        unsafe { buf.write_at(off, &val.to_ne_bytes()) };
    }
}

/// A per-token uniform binding of a captured dispatch — the metas whose only
/// per-token change is a position/offset field (Q/K norm+RoPE `pos`, KV-cast
/// `write_pos`, split attention `n_pos`). [`ReplayGraph::from_capture`] creates
/// a host-visible buffer initialised to `init_bytes` and binds it; each token
/// `update_per_token` rewrites `fields` in place (a `memcpy`, no wgpu submit).
#[derive(Clone)]
pub struct PerTokenBinding {
    pub binding: u32,
    pub init_bytes: Vec<u8>,
    pub fields: Vec<PerTokenField>,
}

impl CaptureStep {
    /// The `wgpu::Buffer` + offset bound at index `i` of a dispatch step — the
    /// decode loop uses this to find the captured input (`steps[0]` binding 0 =
    /// the first layer's residual buffer) it must write each token. Panics on a
    /// copy step or out-of-range index.
    pub fn dispatch_binding(&self, i: usize) -> (wgpu::Buffer, u64) {
        match self {
            CaptureStep::Dispatch { bindings, .. } => {
                (bindings[i].buffer.clone(), bindings[i].offset)
            }
            CaptureStep::Copy { .. } => panic!("dispatch_binding on a copy step"),
            CaptureStep::HostInput { .. } => panic!("dispatch_binding on a host-input step"),
        }
    }
}

/// One step of a captured decode graph — a dispatch (carrying the kernel's WGSL
/// so [`ReplayGraph::from_capture`] can compile and dedup pipelines, plus any
/// per-token uniform bindings) or an inter-buffer copy. Emitted by orangu's real
/// recording into the backend's `decode_capture` sink, so the op sequence is
/// never re-derived by hand.
#[derive(Clone)]
pub enum CaptureStep {
    Dispatch {
        wgsl: String,
        bindings: Vec<CaptureBinding>,
        per_token: Vec<PerTokenBinding>,
        groups: [u32; 3],
    },
    Copy {
        src: wgpu::Buffer,
        src_offset: u64,
        dst: wgpu::Buffer,
        dst_offset: u64,
        size: u64,
    },
    /// A per-token host→device input the wgpu path uploads via
    /// `queue.write_buffer` (`GpuInput::Cpu`) — not a GPU op, so it records
    /// nothing into the persistent command buffer. Instead the decode loop
    /// rewrites `buffer[offset..]` from the CPU each token (the new token's
    /// embedding / gathered per-layer embeddings), then a bare
    /// `queue.submit(empty)` + the command buffer's entry barrier makes the
    /// transfer visible to the raw compute submit. `from_capture` collects
    /// these (via [`host_inputs`]) rather than recording them.
    HostInput {
        tag: HostInputTag,
        buffer: wgpu::Buffer,
        offset: u64,
        size: u64,
    },
}

/// Which per-token CPU value a [`CaptureStep::HostInput`] target consumes — all
/// buffers with the same tag receive the same bytes each token.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HostInputTag {
    /// The scaled token embedding (`tok_embeddings.row(t) * sqrt(n_embd)`) —
    /// the layer-0 input and, for PLE models, the PLE projection's `x` input.
    EmbeddingX,
    /// The gathered per-layer token embeddings (`[n_layer * per_layer]`) — the
    /// PLE projection's second per-token input.
    Gathered,
}

/// Every [`CaptureStep::HostInput`] in a captured op-list, in emission order —
/// the decode loop writes each token's fresh bytes into these buffers before
/// resubmitting. See [`CaptureStep::HostInput`].
pub fn host_inputs(steps: &[CaptureStep]) -> Vec<(HostInputTag, wgpu::Buffer, u64, u64)> {
    steps
        .iter()
        .filter_map(|s| match s {
            CaptureStep::HostInput {
                tag,
                buffer,
                offset,
                size,
            } => Some((*tag, buffer.clone(), *offset, *size)),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The single process-wide test `VulkanBackend`, or `None` (skip) if no
    /// Vulkan adapter is present. Reuses the *same* `wgpu::Device` the rest of
    /// the GPU tests use — creating a second `wgpu::Instance` here would
    /// reintroduce the multi-instance SIGSEGV `vulkan::shared_test_backend`
    /// exists to avoid.
    fn shared_backend() -> Option<&'static super::super::vulkan::VulkanBackend> {
        super::super::vulkan::shared_test_backend()
    }

    /// End-to-end proof of the raw path: build a `out[i] = in[i] * 2.0`
    /// pipeline from WGSL, run it on host-visible buffers via a raw command
    /// buffer + fence, and check the result — exercising SPIR-V compile,
    /// descriptor layout/pool/set, command recording, submit, and fence.
    #[test]
    fn raw_compute_doubles_input() {
        let Some(backend) = shared_backend() else {
            eprintln!("skipping: no Vulkan adapter");
            return;
        };
        let device = backend.wgpu_device();

        unsafe {
            let ctx = ReplayContext::from_wgpu(device).expect("vulkan context");

            const N: usize = 256;
            let wgsl = r#"
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < arrayLength(&src)) {
        dst[i] = src[i] * 2.0;
    }
}
"#;
            let program = ComputeProgram::from_wgsl(
                &ctx,
                wgsl,
                &[
                    BindingSpec {
                        binding: 0,
                        kind: vk::DescriptorType::STORAGE_BUFFER,
                    },
                    BindingSpec {
                        binding: 1,
                        kind: vk::DescriptorType::STORAGE_BUFFER,
                    },
                ],
            )
            .expect("program");

            let bytes = (N * 4) as vk::DeviceSize;
            let src = MappedBuffer::new(&ctx, bytes, vk::BufferUsageFlags::STORAGE_BUFFER)
                .expect("src buf");
            let dst = MappedBuffer::new(&ctx, bytes, vk::BufferUsageFlags::STORAGE_BUFFER)
                .expect("dst buf");
            let input: Vec<f32> = (0..N).map(|i| i as f32 + 0.5).collect();
            src.write(bytemuck::cast_slice(&input));

            // Descriptor pool + set, pointed at src/dst.
            let pool_sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(2)];
            let pool_info = vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&pool_sizes);
            let pool = ctx
                .device
                .create_descriptor_pool(&pool_info, None)
                .expect("pool");
            let set_layouts = [program.set_layout];
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&set_layouts);
            let set = ctx
                .device
                .allocate_descriptor_sets(&alloc_info)
                .expect("set")[0];
            let src_info = [vk::DescriptorBufferInfo::default()
                .buffer(src.buffer)
                .offset(0)
                .range(bytes)];
            let dst_info = [vk::DescriptorBufferInfo::default()
                .buffer(dst.buffer)
                .offset(0)
                .range(bytes)];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&src_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&dst_info),
            ];
            ctx.device.update_descriptor_sets(&writes, &[]);

            // Command pool + buffer.
            let cp_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(ctx.queue_family_index)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            let cmd_pool = ctx
                .device
                .create_command_pool(&cp_info, None)
                .expect("cmd pool");
            let cb_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(cmd_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cb = ctx.device.allocate_command_buffers(&cb_info).expect("cb")[0];

            ctx.device
                .begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default())
                .unwrap();
            ctx.device
                .cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, program.pipeline);
            ctx.device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                program.pipeline_layout,
                0,
                &[set],
                &[],
            );
            ctx.device.cmd_dispatch(cb, (N as u32).div_ceil(64), 1, 1);
            ctx.device.end_command_buffer(cb).unwrap();

            let fence = ctx
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .unwrap();
            let cbs = [cb];
            let submit = vk::SubmitInfo::default().command_buffers(&cbs);
            ctx.device
                .queue_submit(ctx.queue, &[submit], fence)
                .unwrap();
            ctx.device
                .wait_for_fences(&[fence], true, u64::MAX)
                .unwrap();

            let out_bytes = dst.read(N * 4);
            let out: &[f32] = bytemuck::cast_slice(&out_bytes);
            for i in 0..N {
                assert!(
                    (out[i] - input[i] * 2.0).abs() < 1e-6,
                    "idx {i}: {} != {}",
                    out[i],
                    input[i] * 2.0
                );
            }

            // Teardown.
            ctx.device.destroy_fence(fence, None);
            ctx.device.destroy_command_pool(cmd_pool, None);
            ctx.device.destroy_descriptor_pool(pool, None);
            src.destroy(&ctx);
            dst.destroy(&ctx);
            program.destroy(&ctx);
        }
    }

    /// Extracts the raw `VkBuffer` behind a `wgpu::Buffer` via the public
    /// `as_hal` surface. The handle is `Copy`; the `wgpu::Buffer` must outlive
    /// its use (the underlying `VkBuffer` lives as long as the wgpu buffer).
    unsafe fn raw_vk_buffer(buf: &wgpu::Buffer) -> vk::Buffer {
        unsafe { buf.as_hal::<Vulkan>().expect("vulkan buffer").raw_handle() }
    }

    /// The keystone: run orangu's **real** `rmsnorm` WGSL through the raw path,
    /// binding orangu's **actual** wgpu-created `x`/`weight` buffers (reached via
    /// the public `Buffer::as_hal`) and a host-visible `ElemMeta` uniform, and
    /// check byte-for-byte against a CPU reference. Proves the two claims the
    /// full decode-graph migration rests on: (1) a `wgpu` buffer can back a
    /// descriptor in a raw pipeline, and (2) orangu's own shader source runs
    /// identically off the raw submit path.
    #[test]
    fn raw_path_runs_orangu_rmsnorm_on_wgpu_buffers() {
        let Some(backend) = shared_backend() else {
            eprintln!("skipping: no Vulkan adapter");
            return;
        };
        let device = backend.wgpu_device();
        let queue = backend.wgpu_queue();

        const N: usize = 320;
        const WG: usize = 128;
        let eps = 1e-6f32;
        let x: Vec<f32> = (0..N).map(|i| ((i as f32) * 0.017).sin()).collect();
        let weight: Vec<f32> = (0..N).map(|i| 1.0 + ((i as f32) * 0.003)).collect();

        // CPU reference matching RMSNORM_SHADER_BODY_TEMPLATE exactly.
        let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / N as f32;
        let scale = 1.0 / (mean_sq + eps).sqrt();
        let reference: Vec<f32> = (0..N).map(|i| x[i] * scale * weight[i]).collect();

        // orangu's real wgpu buffers for the two storage inputs.
        let x_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("x"),
            size: (N * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let w_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("weight"),
            size: (N * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&x_buf, 0, bytemuck::cast_slice(&x));
        queue.write_buffer(&w_buf, 0, bytemuck::cast_slice(&weight));
        // Flush the staged uploads (they land on the next submit) and wait, so
        // the contents are resident before the raw submit reads them.
        queue.submit(std::iter::empty());
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");

        unsafe {
            let ctx = ReplayContext::from_wgpu(device).expect("vulkan context");

            let wgsl = super::super::vulkan_shaders::shader_source_rmsnorm(false, WG);
            let program = ComputeProgram::from_wgsl(
                &ctx,
                &wgsl,
                &[
                    // elem4 layout: x(ro), weight(ro), y(rw), meta(uniform).
                    BindingSpec {
                        binding: 0,
                        kind: vk::DescriptorType::STORAGE_BUFFER,
                    },
                    BindingSpec {
                        binding: 1,
                        kind: vk::DescriptorType::STORAGE_BUFFER,
                    },
                    BindingSpec {
                        binding: 2,
                        kind: vk::DescriptorType::STORAGE_BUFFER,
                    },
                    BindingSpec {
                        binding: 3,
                        kind: vk::DescriptorType::UNIFORM_BUFFER,
                    },
                ],
            )
            .expect("program");

            // Host-visible output + meta uniform we own outright.
            let y = MappedBuffer::new(&ctx, (N * 4) as u64, vk::BufferUsageFlags::STORAGE_BUFFER)
                .expect("y");
            let meta =
                MappedBuffer::new(&ctx, 16, vk::BufferUsageFlags::UNIFORM_BUFFER).expect("meta");
            // ElemMeta { len: u32, _pad0: u32, extra: f32, _pad1: u32 }.
            let mut meta_bytes = [0u8; 16];
            meta_bytes[0..4].copy_from_slice(&(N as u32).to_ne_bytes());
            meta_bytes[8..12].copy_from_slice(&eps.to_ne_bytes());
            meta.write(&meta_bytes);

            // Raw handles for orangu's wgpu input buffers.
            let x_raw = raw_vk_buffer(&x_buf);
            let w_raw = raw_vk_buffer(&w_buf);

            // Descriptor pool + set wired to the four buffers.
            let pool_sizes = [
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(3),
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(1),
            ];
            let pool = ctx
                .device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(1)
                        .pool_sizes(&pool_sizes),
                    None,
                )
                .expect("pool");
            let set_layouts = [program.set_layout];
            let set = ctx
                .device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(pool)
                        .set_layouts(&set_layouts),
                )
                .expect("set")[0];
            let bytes = (N * 4) as vk::DeviceSize;
            let x_info = [vk::DescriptorBufferInfo::default()
                .buffer(x_raw)
                .offset(0)
                .range(bytes)];
            let w_info = [vk::DescriptorBufferInfo::default()
                .buffer(w_raw)
                .offset(0)
                .range(bytes)];
            let y_info = [vk::DescriptorBufferInfo::default()
                .buffer(y.buffer)
                .offset(0)
                .range(bytes)];
            let m_info = [vk::DescriptorBufferInfo::default()
                .buffer(meta.buffer)
                .offset(0)
                .range(16)];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&x_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&w_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&y_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(3)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&m_info),
            ];
            ctx.device.update_descriptor_sets(&writes, &[]);

            let cmd_pool = ctx
                .device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(ctx.queue_family_index),
                    None,
                )
                .expect("cmd pool");
            let cb = ctx
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(cmd_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .expect("cb")[0];
            ctx.device
                .begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default())
                .unwrap();
            ctx.device
                .cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, program.pipeline);
            ctx.device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                program.pipeline_layout,
                0,
                &[set],
                &[],
            );
            // rmsnorm is one workgroup grid-striding the whole row.
            ctx.device.cmd_dispatch(cb, 1, 1, 1);
            // Make the shader write host-visible for the mapped read-back.
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ);
            ctx.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
            ctx.device.end_command_buffer(cb).unwrap();

            let fence = ctx
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .unwrap();
            let cbs = [cb];
            ctx.device
                .queue_submit(
                    ctx.queue,
                    &[vk::SubmitInfo::default().command_buffers(&cbs)],
                    fence,
                )
                .unwrap();
            ctx.device
                .wait_for_fences(&[fence], true, u64::MAX)
                .unwrap();

            let out_bytes = y.read(N * 4);
            let out: &[f32] = bytemuck::cast_slice(&out_bytes);
            let mut max_err = 0.0f32;
            for i in 0..N {
                max_err = max_err.max((out[i] - reference[i]).abs());
            }
            assert!(max_err < 1e-5, "rmsnorm raw-path max abs err {max_err}");

            ctx.device.destroy_fence(fence, None);
            ctx.device.destroy_command_pool(cmd_pool, None);
            ctx.device.destroy_descriptor_pool(pool, None);
            y.destroy(&ctx);
            meta.destroy(&ctx);
            program.destroy(&ctx);
        }
    }

    /// Proves the [`ReplayGraph`] engine end-to-end: a two-op chain
    /// (`tmp = in*k`, then `out = tmp + k`) recorded once into a persistent
    /// command buffer, with an inter-op barrier, and **resubmitted twice** with
    /// `k` changed via a host-visible-uniform `memcpy` between submits. The
    /// output must track `k` on the second run with no re-recording — exactly
    /// the per-token decode pattern (same command buffer, only uniforms change).
    #[test]
    fn replay_graph_reuses_command_buffer_across_tokens() {
        let Some(backend) = shared_backend() else {
            eprintln!("skipping: no Vulkan adapter");
            return;
        };
        let device = backend.wgpu_device();

        const N: usize = 512;
        // Two kernels sharing the same 3-binding layout (src, dst, uniform k).
        let mul_wgsl = r#"
struct U { k: f32, _p0: f32, _p1: f32, _p2: f32 }
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> u: U;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < arrayLength(&src)) { dst[i] = src[i] * u.k; }
}
"#;
        let add_wgsl = r#"
struct U { k: f32, _p0: f32, _p1: f32, _p2: f32 }
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> u: U;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < arrayLength(&src)) { dst[i] = src[i] + u.k; }
}
"#;
        let bindings = [
            BindingSpec {
                binding: 0,
                kind: vk::DescriptorType::STORAGE_BUFFER,
            },
            BindingSpec {
                binding: 1,
                kind: vk::DescriptorType::STORAGE_BUFFER,
            },
            BindingSpec {
                binding: 2,
                kind: vk::DescriptorType::UNIFORM_BUFFER,
            },
        ];

        unsafe {
            let ctx = ReplayContext::from_wgpu(device).expect("vulkan context");
            let mul = ComputeProgram::from_wgsl(&ctx, mul_wgsl, &bindings).expect("mul");
            let add = ComputeProgram::from_wgsl(&ctx, add_wgsl, &bindings).expect("add");
            let programs = [mul, add];

            let bytes = (N * 4) as vk::DeviceSize;
            let input: Vec<f32> = (0..N).map(|i| (i as f32) * 0.25 - 3.0).collect();
            let in_buf =
                MappedBuffer::new(&ctx, bytes, vk::BufferUsageFlags::STORAGE_BUFFER).expect("in");
            let tmp_buf =
                MappedBuffer::new(&ctx, bytes, vk::BufferUsageFlags::STORAGE_BUFFER).expect("tmp");
            let out_buf =
                MappedBuffer::new(&ctx, bytes, vk::BufferUsageFlags::STORAGE_BUFFER).expect("out");
            let k_buf =
                MappedBuffer::new(&ctx, 16, vk::BufferUsageFlags::UNIFORM_BUFFER).expect("k");
            in_buf.write(bytemuck::cast_slice(&input));

            let groups = [(N as u32).div_ceil(64), 1, 1];
            let ops = vec![
                // op0: tmp = in * k
                ReplayOp {
                    program: 0,
                    groups,
                    bindings: vec![
                        OpBinding {
                            binding: 0,
                            kind: vk::DescriptorType::STORAGE_BUFFER,
                            buffer: in_buf.buffer,
                            offset: 0,
                            size: bytes,
                        },
                        OpBinding {
                            binding: 1,
                            kind: vk::DescriptorType::STORAGE_BUFFER,
                            buffer: tmp_buf.buffer,
                            offset: 0,
                            size: bytes,
                        },
                        OpBinding {
                            binding: 2,
                            kind: vk::DescriptorType::UNIFORM_BUFFER,
                            buffer: k_buf.buffer,
                            offset: 0,
                            size: 16,
                        },
                    ],
                },
                // op1: out = tmp + k  (reads op0's output → needs the barrier)
                ReplayOp {
                    program: 1,
                    groups,
                    bindings: vec![
                        OpBinding {
                            binding: 0,
                            kind: vk::DescriptorType::STORAGE_BUFFER,
                            buffer: tmp_buf.buffer,
                            offset: 0,
                            size: bytes,
                        },
                        OpBinding {
                            binding: 1,
                            kind: vk::DescriptorType::STORAGE_BUFFER,
                            buffer: out_buf.buffer,
                            offset: 0,
                            size: bytes,
                        },
                        OpBinding {
                            binding: 2,
                            kind: vk::DescriptorType::UNIFORM_BUFFER,
                            buffer: k_buf.buffer,
                            offset: 0,
                            size: 16,
                        },
                    ],
                },
            ];

            let graph = ReplayGraph::build(&ctx, &programs, &ops).expect("graph");

            // Two "tokens" with different k, same recorded command buffer.
            for &k in &[3.0f32, 5.0f32] {
                let mut kb = [0u8; 16];
                kb[0..4].copy_from_slice(&k.to_ne_bytes());
                k_buf.write(&kb);
                graph.run_token(&ctx).expect("run");
                let out_bytes = out_buf.read(N * 4);
                let out: &[f32] = bytemuck::cast_slice(&out_bytes);
                let mut max_err = 0.0f32;
                for i in 0..N {
                    let want = input[i] * k + k;
                    max_err = max_err.max((out[i] - want).abs());
                }
                assert!(max_err < 1e-4, "k={k}: replay chain max abs err {max_err}");
            }

            graph.destroy(&ctx);
            in_buf.destroy(&ctx);
            tmp_buf.destroy(&ctx);
            out_buf.destroy(&ctx);
            k_buf.destroy(&ctx);
            let [mul, add] = programs;
            mul.destroy(&ctx);
            add.destroy(&ctx);
        }
    }

    /// The first **real capture**: build one layer's actual cached
    /// `attn_norm` resources (`VulkanBackend::build_fused_layer_resources`),
    /// then drive that dispatch through the `ReplayGraph` — binding orangu's
    /// *own* arena `x_buf` at its `BindSrc::Slice` offset, its real uploaded
    /// `attn_norm` weight, and its real `ElemMeta` uniform, all reached via
    /// `Buffer::as_hal` — and check byte-identical against a CPU reference.
    ///
    /// Proves the capture step end-to-end on genuine orangu cached buffers,
    /// including the non-zero sub-range offset an arena chunk carries. This is
    /// the seam that scales to the whole per-layer chain.
    #[test]
    fn replay_captures_real_attn_norm_dispatch() {
        let Some(backend) = shared_backend() else {
            eprintln!("skipping: no Vulkan adapter");
            return;
        };
        let device = backend.wgpu_device();
        let queue = backend.wgpu_queue();

        const N: usize = 2048; // gemma-E2B n_embd
        const WG: usize = 128;
        let eps = 1e-6f32;
        let attn_norm: Vec<f32> = (0..N).map(|i| 1.0 + ((i % 17) as f32) * 0.01).collect();
        let x: Vec<f32> = (0..N).map(|i| ((i as f32) * 0.001).cos() * 0.5).collect();

        // CPU reference (RMSNORM_SHADER_BODY_TEMPLATE).
        let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / N as f32;
        let scale = 1.0 / (mean_sq + eps).sqrt();
        let reference: Vec<f32> = (0..N).map(|i| x[i] * scale * attn_norm[i]).collect();

        // Build orangu's real cached attn_norm buffers (weight + meta uploaded
        // inside), then upload our known `x` into the arena chunk at its offset.
        let cap = backend.test_attn_norm_buffers(N, &attn_norm, eps);
        queue.write_buffer(&cap.x_buf, cap.x_off, bytemuck::cast_slice(&x));
        // Flush every staged upload (x here + weight/meta from the build above)
        // so they're resident before the raw submit reads them.
        queue.submit(std::iter::empty());
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");

        unsafe {
            let ctx = ReplayContext::from_wgpu(device).expect("vulkan context");
            let wgsl = super::super::vulkan_shaders::shader_source_rmsnorm(false, WG);
            let program = ComputeProgram::from_wgsl(
                &ctx,
                &wgsl,
                &[
                    BindingSpec {
                        binding: 0,
                        kind: vk::DescriptorType::STORAGE_BUFFER,
                    },
                    BindingSpec {
                        binding: 1,
                        kind: vk::DescriptorType::STORAGE_BUFFER,
                    },
                    BindingSpec {
                        binding: 2,
                        kind: vk::DescriptorType::STORAGE_BUFFER,
                    },
                    BindingSpec {
                        binding: 3,
                        kind: vk::DescriptorType::UNIFORM_BUFFER,
                    },
                ],
            )
            .expect("program");
            let programs = [program];

            let out = MappedBuffer::new(&ctx, (N * 4) as u64, vk::BufferUsageFlags::STORAGE_BUFFER)
                .expect("out");

            // Raw handles for orangu's real cached buffers.
            let x_raw = raw_vk_buffer(&cap.x_buf);
            let w_raw = raw_vk_buffer(&cap.weight);
            let m_raw = raw_vk_buffer(&cap.meta);

            let ops = vec![ReplayOp {
                program: 0,
                groups: [1, 1, 1], // rmsnorm: one workgroup grid-strides the row
                bindings: vec![
                    // x bound at the arena chunk's real Slice offset.
                    OpBinding {
                        binding: 0,
                        kind: vk::DescriptorType::STORAGE_BUFFER,
                        buffer: x_raw,
                        offset: cap.x_off,
                        size: cap.n_embd_bytes,
                    },
                    OpBinding {
                        binding: 1,
                        kind: vk::DescriptorType::STORAGE_BUFFER,
                        buffer: w_raw,
                        offset: 0,
                        size: cap.n_embd_bytes,
                    },
                    OpBinding {
                        binding: 2,
                        kind: vk::DescriptorType::STORAGE_BUFFER,
                        buffer: out.buffer,
                        offset: 0,
                        size: (N * 4) as u64,
                    },
                    OpBinding {
                        binding: 3,
                        kind: vk::DescriptorType::UNIFORM_BUFFER,
                        buffer: m_raw,
                        offset: 0,
                        size: 16,
                    },
                ],
            }];

            let graph = ReplayGraph::build(&ctx, &programs, &ops).expect("graph");
            graph.run_token(&ctx).expect("run");

            let out_bytes = out.read(N * 4);
            let got: &[f32] = bytemuck::cast_slice(&out_bytes);
            let mut max_err = 0.0f32;
            for i in 0..N {
                max_err = max_err.max((got[i] - reference[i]).abs());
            }
            assert!(max_err < 1e-5, "captured attn_norm max abs err {max_err}");

            graph.destroy(&ctx);
            out.destroy(&ctx);
            let [program] = programs;
            program.destroy(&ctx);
        }
    }

    /// Capture of the **matmul** dispatch — the one that binds the
    /// `WeightArena` chunk that dominates the per-token BO list. Places a real
    /// F32 weight into orangu's genuine weight arena (`test_weight_buffer` →
    /// `WeightArena`), then runs a matmul through the `ReplayGraph` using
    /// orangu's own base reduce kernel (`shader_source_reduce`, at orangu's
    /// `reduce_n_rows`/`subgroup_reduce`), binding that real weight chunk at its
    /// arena offset (binding 0) plus host-visible `x`/`out`/`Meta`. Checked
    /// against a CPU F32 matmul. F32 weights keep the reference trivial and the
    /// block layout raw, isolating the new claim: **orangu's weight buffer binds
    /// into the raw path and computes correctly**.
    #[test]
    fn replay_captures_real_matmul_dispatch() {
        let Some(backend) = shared_backend() else {
            eprintln!("skipping: no Vulkan adapter");
            return;
        };
        let device = backend.wgpu_device();
        let queue = backend.wgpu_queue();

        const IN_DIM: usize = 256;
        const OUT_DIM: usize = 64;
        const N_TOKENS: usize = 1;

        // Deterministic F32 weight [out_dim, in_dim] and activation [in_dim].
        let weight: Vec<f32> = (0..OUT_DIM * IN_DIM)
            .map(|i| (((i * 2654435761) % 1009) as f32 / 1009.0) - 0.5)
            .collect();
        let x: Vec<f32> = (0..IN_DIM).map(|i| ((i as f32) * 0.013).sin()).collect();

        // Reference: plain F32 matmul out[o] = Σ_e W[o,e]·x[e].
        let reference: Vec<f32> = (0..OUT_DIM)
            .map(|o| {
                (0..IN_DIM)
                    .map(|e| weight[o * IN_DIM + e] * x[e])
                    .sum::<f32>()
            })
            .collect();

        // Real orangu weight arena residency + the base reduce kernel config.
        let w = super::super::super::loader::test_quant_matrix(
            bytemuck::cast_slice(&weight),
            crate::engine::quant::GGML_TYPE_F32,
            IN_DIM,
            OUT_DIM,
        );
        let (weight_chunk, weight_off, weight_size) = backend.test_weight_buffer(&w);
        let (n_rows, subgroup) = backend.test_reduce_config();
        let wgsl = super::super::vulkan_shaders::shader_source_reduce(
            crate::engine::quant::GGML_TYPE_F32,
            n_rows,
            subgroup,
        )
        .expect("f32 reduce kernel");

        // Flush the weight upload (write_buffer staged inside test_weight_buffer)
        // so it's resident before the raw submit reads it.
        queue.submit(std::iter::empty());
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");

        unsafe {
            let ctx = ReplayContext::from_wgpu(device).expect("vulkan context");
            let program = ComputeProgram::from_wgsl(
                &ctx,
                &wgsl,
                &[
                    // matmul bind_group_layout: weights(0), x(1), y(2), meta(3).
                    BindingSpec {
                        binding: 0,
                        kind: vk::DescriptorType::STORAGE_BUFFER,
                    },
                    BindingSpec {
                        binding: 1,
                        kind: vk::DescriptorType::STORAGE_BUFFER,
                    },
                    BindingSpec {
                        binding: 2,
                        kind: vk::DescriptorType::STORAGE_BUFFER,
                    },
                    BindingSpec {
                        binding: 3,
                        kind: vk::DescriptorType::UNIFORM_BUFFER,
                    },
                ],
            )
            .expect("program");
            let programs = [program];

            let x_buf = MappedBuffer::new(
                &ctx,
                (IN_DIM * 4) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            )
            .expect("x");
            x_buf.write(bytemuck::cast_slice(&x));
            let out = MappedBuffer::new(
                &ctx,
                (OUT_DIM * 4) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            )
            .expect("out");
            // Meta { in_dim, out_dim, n_tokens, row_bytes } — same #[repr(C)] as
            // the wgpu path's `Meta`.
            let meta =
                MappedBuffer::new(&ctx, 16, vk::BufferUsageFlags::UNIFORM_BUFFER).expect("meta");
            let mut mb = [0u8; 16];
            mb[0..4].copy_from_slice(&(IN_DIM as u32).to_ne_bytes());
            mb[4..8].copy_from_slice(&(OUT_DIM as u32).to_ne_bytes());
            mb[8..12].copy_from_slice(&(N_TOKENS as u32).to_ne_bytes());
            mb[12..16].copy_from_slice(&(w.row_bytes() as u32).to_ne_bytes());
            meta.write(&mb);

            let w_raw = raw_vk_buffer(&weight_chunk);

            // Dispatch geometry matches build_op_resources: workgroup_dims(
            // ceil(out_dim / reduce_n_rows) * n_tokens).
            let total = ((OUT_DIM.div_ceil(n_rows)) * N_TOKENS) as u32;
            let groups = if total <= 65535 {
                [total.max(1), 1, 1]
            } else {
                [65535, total.div_ceil(65535), 1]
            };

            let ops = vec![ReplayOp {
                program: 0,
                groups,
                bindings: vec![
                    // The real weight-arena chunk at its arena offset.
                    OpBinding {
                        binding: 0,
                        kind: vk::DescriptorType::STORAGE_BUFFER,
                        buffer: w_raw,
                        offset: weight_off,
                        size: weight_size,
                    },
                    OpBinding {
                        binding: 1,
                        kind: vk::DescriptorType::STORAGE_BUFFER,
                        buffer: x_buf.buffer,
                        offset: 0,
                        size: (IN_DIM * 4) as u64,
                    },
                    OpBinding {
                        binding: 2,
                        kind: vk::DescriptorType::STORAGE_BUFFER,
                        buffer: out.buffer,
                        offset: 0,
                        size: (OUT_DIM * 4) as u64,
                    },
                    OpBinding {
                        binding: 3,
                        kind: vk::DescriptorType::UNIFORM_BUFFER,
                        buffer: meta.buffer,
                        offset: 0,
                        size: 16,
                    },
                ],
            }];

            let graph = ReplayGraph::build(&ctx, &programs, &ops).expect("graph");
            graph.run_token(&ctx).expect("run");

            let out_bytes = out.read(OUT_DIM * 4);
            let got: &[f32] = bytemuck::cast_slice(&out_bytes);
            let mut max_err = 0.0f32;
            for o in 0..OUT_DIM {
                max_err = max_err.max((got[o] - reference[o]).abs());
            }
            assert!(max_err < 1e-3, "captured matmul max abs err {max_err}");

            graph.destroy(&ctx);
            x_buf.destroy(&ctx);
            out.destroy(&ctx);
            meta.destroy(&ctx);
            let [program] = programs;
            program.destroy(&ctx);
        }
    }

    /// Capture of the **split-k attention** — the last and most complex decode
    /// dispatch shape: a two-phase chain (phase-1 writes softmax partials →
    /// barrier → phase-2 reduce) with the 6-binding attention layout, binding
    /// orangu's **real per-layer KV buffer** sub-ranges (`k_off/k_size`,
    /// `v_off/v_size` from `LayerCache::sync_gpu` → `GpuKvRefs`) reached via
    /// `Buffer::as_hal`. Runs orangu's own `shader_source_attention_split` +
    /// `_reduce` kernels through the `ReplayGraph` and checks against a CPU
    /// attention reference. Proves the real KV-buffer sub-range binding and the
    /// two-op-with-barrier attention path end-to-end.
    #[test]
    fn replay_captures_real_split_attention() {
        use crate::engine::kv_cache::KvCache;

        let Some(backend) = shared_backend() else {
            eprintln!("skipping: no Vulkan adapter");
            return;
        };
        let device = backend.wgpu_device();
        let queue = backend.wgpu_queue();
        let (kv_storage, subgroup, attn_gqa, _flash, k_num) = backend.test_attn_config();
        if attn_gqa {
            eprintln!("skipping: ORANGU_ATTN_GQA changes the kernel/dispatch");
            return;
        }

        let n_head = 4usize;
        let n_head_kv = 2usize;
        let head_dim = 8usize;
        let group_size = n_head / n_head_kv;
        let kv_dim = n_head_kv * head_dim;
        let capacity = 64usize;
        let n_positions = 37usize;
        let pos = n_positions - 1;
        let window_start = 0usize;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Deterministic KV + Q.
        let mut s = 0x59717u64;
        let mut nb = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) & 0xFF) as f32
        };
        let mut kv_cache = KvCache::new_with_dims(capacity, &[kv_dim]);
        for _ in 0..n_positions {
            let k: Vec<f32> = (0..kv_dim).map(|_| (nb() - 128.0) / 64.0).collect();
            let v: Vec<f32> = (0..kv_dim).map(|_| (nb() - 128.0) / 64.0).collect();
            kv_cache.layers[0].push(&k, &v);
        }
        let q: Vec<f32> = (0..n_head * head_dim)
            .map(|_| (nb() - 128.0) / 64.0)
            .collect();

        // CPU attention reference.
        let mut reference = vec![0f32; n_head * head_dim];
        for h in 0..n_head {
            let kv_head = h / group_size;
            let qh = &q[h * head_dim..(h + 1) * head_dim];
            let mut scores = Vec::with_capacity(pos + 1 - window_start);
            for p in window_start..=pos {
                let kh = kv_cache.layers[0].key_at(p, kv_head, head_dim);
                scores.push(crate::engine::tensor::dot(qh, kh) * scale);
            }
            crate::engine::tensor::softmax_inplace(&mut scores);
            let out = &mut reference[h * head_dim..(h + 1) * head_dim];
            for (offset, &w) in scores.iter().enumerate() {
                let vh = kv_cache.layers[0].value_at(window_start + offset, kv_head, head_dim);
                for (o, vi) in out.iter_mut().zip(vh.iter()) {
                    *o += w * vi;
                }
            }
        }

        // Real KV buffer residency (uploads staged inside sync_gpu).
        let kv_refs = kv_cache.layers[0].sync_gpu(device, queue, n_head, kv_storage);
        queue.submit(std::iter::empty());
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");

        let split_wgsl = super::super::vulkan_shaders::shader_source_attention_split(
            kv_storage,
            subgroup,
            head_dim as u32,
        );
        let reduce_wgsl = super::super::vulkan_shaders::shader_source_attention_split_reduce();

        unsafe {
            let ctx = ReplayContext::from_wgpu(device).expect("vulkan context");
            // attn_bind_group_layout: q(0), k(1), v(2), partial_ml(3),
            // partial_acc(4), meta(5) — 5 storage + 1 uniform.
            let storage = vk::DescriptorType::STORAGE_BUFFER;
            let uniform = vk::DescriptorType::UNIFORM_BUFFER;
            let split_prog = ComputeProgram::from_wgsl(
                &ctx,
                &split_wgsl,
                &[
                    BindingSpec {
                        binding: 0,
                        kind: storage,
                    },
                    BindingSpec {
                        binding: 1,
                        kind: storage,
                    },
                    BindingSpec {
                        binding: 2,
                        kind: storage,
                    },
                    BindingSpec {
                        binding: 3,
                        kind: storage,
                    },
                    BindingSpec {
                        binding: 4,
                        kind: storage,
                    },
                    BindingSpec {
                        binding: 5,
                        kind: uniform,
                    },
                ],
            )
            .expect("split program");
            // elem4 reduce: partial_ml(0), partial_acc(1), out(2), meta(3).
            let reduce_prog = ComputeProgram::from_wgsl(
                &ctx,
                &reduce_wgsl,
                &[
                    BindingSpec {
                        binding: 0,
                        kind: storage,
                    },
                    BindingSpec {
                        binding: 1,
                        kind: storage,
                    },
                    BindingSpec {
                        binding: 2,
                        kind: storage,
                    },
                    BindingSpec {
                        binding: 3,
                        kind: uniform,
                    },
                ],
            )
            .expect("reduce program");
            let programs = [split_prog, reduce_prog];

            let q_buf = MappedBuffer::new(
                &ctx,
                (q.len() * 4) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            )
            .expect("q");
            q_buf.write(bytemuck::cast_slice(&q));
            let ml_len = n_head * k_num as usize * 2;
            let acc_len = n_head * k_num as usize * head_dim;
            let partial_ml = MappedBuffer::new(
                &ctx,
                (ml_len * 4) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            )
            .expect("ml");
            let partial_acc = MappedBuffer::new(
                &ctx,
                (acc_len * 4) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            )
            .expect("acc");
            let out = MappedBuffer::new(
                &ctx,
                (n_head * head_dim * 4) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            )
            .expect("out");

            // AttnSplitMeta {n_head,n_head_kv,head_dim,window_start,n_pos,k_num,scale,_pad}.
            let split_meta = MappedBuffer::new(&ctx, 32, vk::BufferUsageFlags::UNIFORM_BUFFER)
                .expect("split meta");
            let mut sm = [0u8; 32];
            sm[0..4].copy_from_slice(&(n_head as u32).to_ne_bytes());
            sm[4..8].copy_from_slice(&(n_head_kv as u32).to_ne_bytes());
            sm[8..12].copy_from_slice(&(head_dim as u32).to_ne_bytes());
            sm[12..16].copy_from_slice(&(window_start as u32).to_ne_bytes());
            sm[16..20].copy_from_slice(&((pos - window_start + 1) as u32).to_ne_bytes());
            sm[20..24].copy_from_slice(&k_num.to_ne_bytes());
            sm[24..28].copy_from_slice(&scale.to_ne_bytes());
            split_meta.write(&sm);

            // AttnReduceMeta {head_dim,k_num,_pad0,_pad1}.
            let reduce_meta = MappedBuffer::new(&ctx, 16, vk::BufferUsageFlags::UNIFORM_BUFFER)
                .expect("reduce meta");
            let mut rm = [0u8; 16];
            rm[0..4].copy_from_slice(&(head_dim as u32).to_ne_bytes());
            rm[4..8].copy_from_slice(&k_num.to_ne_bytes());
            reduce_meta.write(&rm);

            let kv_raw = raw_vk_buffer(&kv_refs.buffer);

            let ops = vec![
                // phase 1: split attention → partials.
                ReplayOp {
                    program: 0,
                    groups: [n_head as u32, k_num, 1],
                    bindings: vec![
                        OpBinding {
                            binding: 0,
                            kind: storage,
                            buffer: q_buf.buffer,
                            offset: 0,
                            size: (q.len() * 4) as u64,
                        },
                        OpBinding {
                            binding: 1,
                            kind: storage,
                            buffer: kv_raw,
                            offset: kv_refs.k_off,
                            size: kv_refs.k_size,
                        },
                        OpBinding {
                            binding: 2,
                            kind: storage,
                            buffer: kv_raw,
                            offset: kv_refs.v_off,
                            size: kv_refs.v_size,
                        },
                        OpBinding {
                            binding: 3,
                            kind: storage,
                            buffer: partial_ml.buffer,
                            offset: 0,
                            size: (ml_len * 4) as u64,
                        },
                        OpBinding {
                            binding: 4,
                            kind: storage,
                            buffer: partial_acc.buffer,
                            offset: 0,
                            size: (acc_len * 4) as u64,
                        },
                        OpBinding {
                            binding: 5,
                            kind: uniform,
                            buffer: split_meta.buffer,
                            offset: 0,
                            size: 32,
                        },
                    ],
                },
                // phase 2: reduce partials → out.
                ReplayOp {
                    program: 1,
                    groups: [n_head as u32, 1, 1],
                    bindings: vec![
                        OpBinding {
                            binding: 0,
                            kind: storage,
                            buffer: partial_ml.buffer,
                            offset: 0,
                            size: (ml_len * 4) as u64,
                        },
                        OpBinding {
                            binding: 1,
                            kind: storage,
                            buffer: partial_acc.buffer,
                            offset: 0,
                            size: (acc_len * 4) as u64,
                        },
                        OpBinding {
                            binding: 2,
                            kind: storage,
                            buffer: out.buffer,
                            offset: 0,
                            size: (n_head * head_dim * 4) as u64,
                        },
                        OpBinding {
                            binding: 3,
                            kind: uniform,
                            buffer: reduce_meta.buffer,
                            offset: 0,
                            size: 16,
                        },
                    ],
                },
            ];

            let graph = ReplayGraph::build(&ctx, &programs, &ops).expect("graph");
            graph.run_token(&ctx).expect("run");

            let out_bytes = out.read(n_head * head_dim * 4);
            let got: &[f32] = bytemuck::cast_slice(&out_bytes);
            let mut max_err = 0.0f32;
            for i in 0..n_head * head_dim {
                let tol = 6e-2 * reference[i].abs().max(1.0);
                max_err = max_err.max((got[i] - reference[i]).abs() - tol).max(0.0);
            }
            assert!(
                max_err <= 0.0,
                "captured split attention exceeded tolerance by {max_err}"
            );

            graph.destroy(&ctx);
            q_buf.destroy(&ctx);
            partial_ml.destroy(&ctx);
            partial_acc.destroy(&ctx);
            out.destroy(&ctx);
            split_meta.destroy(&ctx);
            reduce_meta.destroy(&ctx);
            let [split_prog, reduce_prog] = programs;
            split_prog.destroy(&ctx);
            reduce_prog.destroy(&ctx);
        }
    }

    /// Orchestration milestone: a real **FFN-shaped 4-op chain** —
    /// `gate = Wgate·x`, `g = gelu(gate)`, `up = Wup·x`, `prod = g * up` —
    /// assembled into **one** persistent command buffer binding **two** real
    /// `WeightArena` buffers plus host-visible intermediates, then **reused
    /// across two tokens** with only `x` changed by `memcpy`. This is the exact
    /// decode shape (weights fixed in the reused BO list, activation the only
    /// per-token change) at the granularity of a layer sub-chain: multiple real
    /// matmuls + elementwise kernels chained with barriers, no re-recording.
    /// Checked against a CPU reference each token.
    #[test]
    fn replay_orchestrates_ffn_chain_across_tokens() {
        let Some(backend) = shared_backend() else {
            eprintln!("skipping: no Vulkan adapter");
            return;
        };
        let device = backend.wgpu_device();
        let queue = backend.wgpu_queue();

        const IN_DIM: usize = 256;
        const FFN: usize = 96;
        let f32_ty = crate::engine::quant::GGML_TYPE_F32;

        let gate_w: Vec<f32> = (0..FFN * IN_DIM)
            .map(|i| (((i * 40503) % 1013) as f32 / 1013.0) - 0.5)
            .collect();
        let up_w: Vec<f32> = (0..FFN * IN_DIM)
            .map(|i| (((i * 15485863) % 997) as f32 / 997.0) - 0.5)
            .collect();
        let two_x: [Vec<f32>; 2] = [
            (0..IN_DIM).map(|i| ((i as f32) * 0.011).sin()).collect(),
            (0..IN_DIM)
                .map(|i| ((i as f32) * 0.023).cos() * 0.7)
                .collect(),
        ];

        let gelu = |v: f32| 0.5 * v * (1.0 + (0.7978846 * v * (1.0 + 0.044715 * v * v)).tanh());
        let cpu_ffn = |x: &[f32]| -> Vec<f32> {
            (0..FFN)
                .map(|o| {
                    let gate: f32 = (0..IN_DIM).map(|e| gate_w[o * IN_DIM + e] * x[e]).sum();
                    let up: f32 = (0..IN_DIM).map(|e| up_w[o * IN_DIM + e] * x[e]).sum();
                    gelu(gate) * up
                })
                .collect()
        };

        let gate_qm = super::super::super::loader::test_quant_matrix(
            bytemuck::cast_slice(&gate_w),
            f32_ty,
            IN_DIM,
            FFN,
        );
        let up_qm = super::super::super::loader::test_quant_matrix(
            bytemuck::cast_slice(&up_w),
            f32_ty,
            IN_DIM,
            FFN,
        );
        let (gate_chunk, gate_off, gate_size) = backend.test_weight_buffer(&gate_qm);
        let (up_chunk, up_off, up_size) = backend.test_weight_buffer(&up_qm);
        let (n_rows, subgroup) = backend.test_reduce_config();
        let reduce_wgsl =
            super::super::vulkan_shaders::shader_source_reduce(f32_ty, n_rows, subgroup)
                .expect("reduce kernel");
        let gelu_wgsl = super::super::vulkan_shaders::shader_source_gelu();
        let mul_wgsl = super::super::vulkan_shaders::shader_source_mul();

        queue.submit(std::iter::empty());
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");

        let storage = vk::DescriptorType::STORAGE_BUFFER;
        let uniform = vk::DescriptorType::UNIFORM_BUFFER;
        unsafe {
            let ctx = ReplayContext::from_wgpu(device).expect("vulkan context");
            let matmul_layout = [
                BindingSpec {
                    binding: 0,
                    kind: storage,
                },
                BindingSpec {
                    binding: 1,
                    kind: storage,
                },
                BindingSpec {
                    binding: 2,
                    kind: storage,
                },
                BindingSpec {
                    binding: 3,
                    kind: uniform,
                },
            ];
            let reduce_prog =
                ComputeProgram::from_wgsl(&ctx, &reduce_wgsl, &matmul_layout).expect("reduce");
            let gelu_prog = ComputeProgram::from_wgsl(
                &ctx,
                &gelu_wgsl,
                &[
                    BindingSpec {
                        binding: 0,
                        kind: storage,
                    },
                    BindingSpec {
                        binding: 1,
                        kind: storage,
                    },
                    BindingSpec {
                        binding: 2,
                        kind: uniform,
                    },
                ],
            )
            .expect("gelu");
            let mul_prog = ComputeProgram::from_wgsl(&ctx, &mul_wgsl, &matmul_layout).expect("mul");
            let programs = [reduce_prog, gelu_prog, mul_prog];

            let mk = |n: usize, u: vk::BufferUsageFlags| {
                MappedBuffer::new(&ctx, (n * 4) as u64, u).expect("buf")
            };
            let x_buf = mk(IN_DIM, vk::BufferUsageFlags::STORAGE_BUFFER);
            let gate_buf = mk(FFN, vk::BufferUsageFlags::STORAGE_BUFFER);
            let g_buf = mk(FFN, vk::BufferUsageFlags::STORAGE_BUFFER);
            let up_buf = mk(FFN, vk::BufferUsageFlags::STORAGE_BUFFER);
            let prod = mk(FFN, vk::BufferUsageFlags::STORAGE_BUFFER);

            // matmul Meta {in_dim, out_dim, n_tokens, row_bytes}.
            let mm_meta =
                MappedBuffer::new(&ctx, 16, vk::BufferUsageFlags::UNIFORM_BUFFER).expect("mm meta");
            let mut mmb = [0u8; 16];
            mmb[0..4].copy_from_slice(&(IN_DIM as u32).to_ne_bytes());
            mmb[4..8].copy_from_slice(&(FFN as u32).to_ne_bytes());
            mmb[8..12].copy_from_slice(&1u32.to_ne_bytes());
            mmb[12..16].copy_from_slice(&(gate_qm.row_bytes() as u32).to_ne_bytes());
            mm_meta.write(&mmb);
            // ElemMeta {len, _, _, _} for gelu/mul over FFN elements.
            let elem_meta = MappedBuffer::new(&ctx, 16, vk::BufferUsageFlags::UNIFORM_BUFFER)
                .expect("elem meta");
            let mut emb = [0u8; 16];
            emb[0..4].copy_from_slice(&(FFN as u32).to_ne_bytes());
            elem_meta.write(&emb);

            let gate_raw = raw_vk_buffer(&gate_chunk);
            let up_raw = raw_vk_buffer(&up_chunk);
            let mm_total = (FFN.div_ceil(n_rows)) as u32;
            let mm_groups = [mm_total.max(1), 1, 1];
            let elem_groups = [(FFN as u32).div_ceil(64), 1, 1];
            let sb = |buf: vk::Buffer, off: u64, sz: u64, b: u32, k| OpBinding {
                binding: b,
                kind: k,
                buffer: buf,
                offset: off,
                size: sz,
            };
            let ffn4 = (FFN * 4) as u64;
            let ops = vec![
                // gate = Wgate · x
                ReplayOp {
                    program: 0,
                    groups: mm_groups,
                    bindings: vec![
                        sb(gate_raw, gate_off, gate_size, 0, storage),
                        sb(x_buf.buffer, 0, (IN_DIM * 4) as u64, 1, storage),
                        sb(gate_buf.buffer, 0, ffn4, 2, storage),
                        sb(mm_meta.buffer, 0, 16, 3, uniform),
                    ],
                },
                // g = gelu(gate)
                ReplayOp {
                    program: 1,
                    groups: elem_groups,
                    bindings: vec![
                        sb(gate_buf.buffer, 0, ffn4, 0, storage),
                        sb(g_buf.buffer, 0, ffn4, 1, storage),
                        sb(elem_meta.buffer, 0, 16, 2, uniform),
                    ],
                },
                // up = Wup · x
                ReplayOp {
                    program: 0,
                    groups: mm_groups,
                    bindings: vec![
                        sb(up_raw, up_off, up_size, 0, storage),
                        sb(x_buf.buffer, 0, (IN_DIM * 4) as u64, 1, storage),
                        sb(up_buf.buffer, 0, ffn4, 2, storage),
                        sb(mm_meta.buffer, 0, 16, 3, uniform),
                    ],
                },
                // prod = g * up
                ReplayOp {
                    program: 2,
                    groups: elem_groups,
                    bindings: vec![
                        sb(g_buf.buffer, 0, ffn4, 0, storage),
                        sb(up_buf.buffer, 0, ffn4, 1, storage),
                        sb(prod.buffer, 0, ffn4, 2, storage),
                        sb(elem_meta.buffer, 0, 16, 3, uniform),
                    ],
                },
            ];

            let graph = ReplayGraph::build(&ctx, &programs, &ops).expect("graph");

            for x in &two_x {
                x_buf.write(bytemuck::cast_slice(x));
                graph.run_token(&ctx).expect("run");
                let out_bytes = prod.read(FFN * 4);
                let got: &[f32] = bytemuck::cast_slice(&out_bytes);
                let reference = cpu_ffn(x);
                let mut max_err = 0.0f32;
                for o in 0..FFN {
                    max_err = max_err.max((got[o] - reference[o]).abs());
                }
                assert!(max_err < 2e-3, "ffn chain max abs err {max_err}");
            }

            graph.destroy(&ctx);
            x_buf.destroy(&ctx);
            gate_buf.destroy(&ctx);
            g_buf.destroy(&ctx);
            up_buf.destroy(&ctx);
            prod.destroy(&ctx);
            mm_meta.destroy(&ctx);
            elem_meta.destroy(&ctx);
            let [reduce_prog, gelu_prog, mul_prog] = programs;
            reduce_prog.destroy(&ctx);
            gelu_prog.destroy(&ctx);
            mul_prog.destroy(&ctx);
        }
    }

    /// Proves inline `ReplayStep::Copy` support: a `copy_buffer_to_buffer`
    /// (`src → mid`) followed by a dispatch reading `mid` (`out = mid * k`),
    /// recorded once and reused across two tokens with `src`/`k` changed by
    /// `memcpy`. orangu's fused decode path threads activations between arena
    /// buffers with exactly such copies (normed→QKV inputs, attn-out→wo input,
    /// ffn_normed→gate/up), so the graph must record and reuse them.
    #[test]
    fn replay_graph_copy_step_feeds_dispatch() {
        let Some(backend) = shared_backend() else {
            eprintln!("skipping: no Vulkan adapter");
            return;
        };
        let device = backend.wgpu_device();

        const N: usize = 320;
        let mul_wgsl = r#"
struct U { k: f32, _p0: f32, _p1: f32, _p2: f32 }
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> u: U;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < arrayLength(&src)) { dst[i] = src[i] * u.k; }
}
"#;
        let storage = vk::DescriptorType::STORAGE_BUFFER;
        let uniform = vk::DescriptorType::UNIFORM_BUFFER;
        unsafe {
            let ctx = ReplayContext::from_wgpu(device).expect("vulkan context");
            let mul = ComputeProgram::from_wgsl(
                &ctx,
                mul_wgsl,
                &[
                    BindingSpec {
                        binding: 0,
                        kind: storage,
                    },
                    BindingSpec {
                        binding: 1,
                        kind: storage,
                    },
                    BindingSpec {
                        binding: 2,
                        kind: uniform,
                    },
                ],
            )
            .expect("mul");
            let programs = [mul];

            let bytes = (N * 4) as vk::DeviceSize;
            let src = MappedBuffer::new(
                &ctx,
                bytes,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            )
            .expect("src");
            let mid = MappedBuffer::new(
                &ctx,
                bytes,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .expect("mid");
            let out =
                MappedBuffer::new(&ctx, bytes, vk::BufferUsageFlags::STORAGE_BUFFER).expect("out");
            let k_buf =
                MappedBuffer::new(&ctx, 16, vk::BufferUsageFlags::UNIFORM_BUFFER).expect("k");

            let steps = vec![
                ReplayStep::Copy {
                    src: src.buffer,
                    src_offset: 0,
                    dst: mid.buffer,
                    dst_offset: 0,
                    size: bytes,
                },
                ReplayStep::Dispatch(ReplayOp {
                    program: 0,
                    groups: [(N as u32).div_ceil(64), 1, 1],
                    bindings: vec![
                        OpBinding {
                            binding: 0,
                            kind: storage,
                            buffer: mid.buffer,
                            offset: 0,
                            size: bytes,
                        },
                        OpBinding {
                            binding: 1,
                            kind: storage,
                            buffer: out.buffer,
                            offset: 0,
                            size: bytes,
                        },
                        OpBinding {
                            binding: 2,
                            kind: uniform,
                            buffer: k_buf.buffer,
                            offset: 0,
                            size: 16,
                        },
                    ],
                }),
            ];
            let graph = ReplayGraph::build_steps(&ctx, &programs, &steps).expect("graph");

            for &(scale, k) in &[(1.0f32, 3.0f32), (0.5f32, 7.0f32)] {
                let input: Vec<f32> = (0..N).map(|i| (i as f32) * scale - 5.0).collect();
                src.write(bytemuck::cast_slice(&input));
                let mut kb = [0u8; 16];
                kb[0..4].copy_from_slice(&k.to_ne_bytes());
                k_buf.write(&kb);
                graph.run_token(&ctx).expect("run");
                let got_bytes = out.read(N * 4);
                let got: &[f32] = bytemuck::cast_slice(&got_bytes);
                let mut max_err = 0.0f32;
                for i in 0..N {
                    max_err = max_err.max((got[i] - input[i] * k).abs());
                }
                assert!(max_err < 1e-4, "copy+dispatch k={k} max abs err {max_err}");
            }

            graph.destroy(&ctx);
            src.destroy(&ctx);
            mid.destroy(&ctx);
            out.destroy(&ctx);
            k_buf.destroy(&ctx);
            let [mul] = programs;
            mul.destroy(&ctx);
        }
    }

    /// Real multi-dispatch capture with **inline copies through orangu's own
    /// arena buffers** — the crux of the full-layer assembly. Chains the first
    /// three real layer steps: `attn_norm` (into the fused-layer `normed_buf`) →
    /// `copy(normed → wq.x_buffer)` → `wq` projection (into the op's
    /// `output_buffer`), then copies the output back to a host-visible buffer to
    /// read it. Every arena buffer + offset is orangu's genuine cached one
    /// (`build_fused_layer_resources`, `op_entry_for`, `WeightArena`), reached
    /// via `Buffer::as_hal`; checked against a CPU `rmsnorm`-then-matmul
    /// reference. Proves the copy-chained buffer hand-off the 20-dispatch layer
    /// is built from.
    #[test]
    fn replay_captures_attn_norm_copy_projection_chain() {
        let Some(backend) = shared_backend() else {
            eprintln!("skipping: no Vulkan adapter");
            return;
        };
        let device = backend.wgpu_device();
        let queue = backend.wgpu_queue();

        const N_EMBD: usize = 256;
        const OUT_DIM: usize = 64; // n_head * head_dim
        const WG: usize = 128;
        let eps = 1e-6f32;
        let f32_ty = crate::engine::quant::GGML_TYPE_F32;

        let attn_norm: Vec<f32> = (0..N_EMBD)
            .map(|i| 1.0 + ((i % 13) as f32) * 0.01)
            .collect();
        let wq: Vec<f32> = (0..OUT_DIM * N_EMBD)
            .map(|i| (((i * 2246822519usize) % 1021) as f32 / 1021.0) - 0.5)
            .collect();
        let x: Vec<f32> = (0..N_EMBD)
            .map(|i| ((i as f32) * 0.017).sin() * 0.6)
            .collect();

        // CPU reference: normed = rmsnorm(x, attn_norm, eps); q = Wq · normed.
        let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / N_EMBD as f32;
        let scale = 1.0 / (mean_sq + eps).sqrt();
        let normed: Vec<f32> = (0..N_EMBD).map(|i| x[i] * scale * attn_norm[i]).collect();
        let reference: Vec<f32> = (0..OUT_DIM)
            .map(|o| (0..N_EMBD).map(|e| wq[o * N_EMBD + e] * normed[e]).sum())
            .collect();

        // Genuine orangu resources.
        let cap = backend.test_attn_norm_buffers(N_EMBD, &attn_norm, eps);
        let wq_qm = super::super::super::loader::test_quant_matrix(
            bytemuck::cast_slice(&wq),
            f32_ty,
            N_EMBD,
            OUT_DIM,
        );
        let op = backend.test_op_buffers(&wq_qm);
        let (wq_chunk, wq_w_off, wq_w_size) = backend.test_weight_buffer(&wq_qm);
        let (n_rows, subgroup) = backend.test_reduce_config();
        let reduce_wgsl =
            super::super::vulkan_shaders::shader_source_reduce(f32_ty, n_rows, subgroup)
                .expect("reduce kernel");
        let rmsnorm_wgsl = super::super::vulkan_shaders::shader_source_rmsnorm(false, WG);

        // Upload x into the real fused-layer x_buf; flush all staged uploads.
        queue.write_buffer(&cap.x_buf, cap.x_off, bytemuck::cast_slice(&x));
        queue.submit(std::iter::empty());
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");

        let storage = vk::DescriptorType::STORAGE_BUFFER;
        let uniform = vk::DescriptorType::UNIFORM_BUFFER;
        unsafe {
            let ctx = ReplayContext::from_wgpu(device).expect("vulkan context");
            let elem4 = [
                BindingSpec {
                    binding: 0,
                    kind: storage,
                },
                BindingSpec {
                    binding: 1,
                    kind: storage,
                },
                BindingSpec {
                    binding: 2,
                    kind: storage,
                },
                BindingSpec {
                    binding: 3,
                    kind: uniform,
                },
            ];
            let norm_prog = ComputeProgram::from_wgsl(&ctx, &rmsnorm_wgsl, &elem4).expect("norm");
            let mm_prog = ComputeProgram::from_wgsl(&ctx, &reduce_wgsl, &elem4).expect("mm");
            let programs = [norm_prog, mm_prog];

            // Host-visible result (copy target) + matmul Meta (rebuilt host-side).
            let result = MappedBuffer::new(
                &ctx,
                (OUT_DIM * 4) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .expect("result");
            let mm_meta =
                MappedBuffer::new(&ctx, 16, vk::BufferUsageFlags::UNIFORM_BUFFER).expect("mm meta");
            let mut mmb = [0u8; 16];
            mmb[0..4].copy_from_slice(&(N_EMBD as u32).to_ne_bytes());
            mmb[4..8].copy_from_slice(&(OUT_DIM as u32).to_ne_bytes());
            mmb[8..12].copy_from_slice(&1u32.to_ne_bytes());
            mmb[12..16].copy_from_slice(&(wq_qm.row_bytes() as u32).to_ne_bytes());
            mm_meta.write(&mmb);

            let x_raw = raw_vk_buffer(&cap.x_buf);
            let w_raw = raw_vk_buffer(&cap.weight);
            let m_raw = raw_vk_buffer(&cap.meta);
            let normed_raw = raw_vk_buffer(&cap.normed_buf);
            let opx_raw = raw_vk_buffer(&op.x_buffer);
            let opy_raw = raw_vk_buffer(&op.output_buffer);
            let wqw_raw = raw_vk_buffer(&wq_chunk);

            let n_embd_bytes = cap.n_embd_bytes;
            let sb = |buf, off, sz, b, k| OpBinding {
                binding: b,
                kind: k,
                buffer: buf,
                offset: off,
                size: sz,
            };
            let steps = vec![
                // 1. attn_norm → normed_buf
                ReplayStep::Dispatch(ReplayOp {
                    program: 0,
                    groups: [1, 1, 1],
                    bindings: vec![
                        sb(x_raw, cap.x_off, n_embd_bytes, 0, storage),
                        sb(w_raw, 0, n_embd_bytes, 1, storage),
                        sb(normed_raw, cap.normed_off, n_embd_bytes, 2, storage),
                        sb(m_raw, 0, 16, 3, uniform),
                    ],
                }),
                // 2. copy normed_buf → wq.x_buffer (arena → arena)
                ReplayStep::Copy {
                    src: normed_raw,
                    src_offset: cap.normed_off,
                    dst: opx_raw,
                    dst_offset: op.x_offset,
                    size: n_embd_bytes,
                },
                // 3. wq projection → op.output_buffer
                ReplayStep::Dispatch(ReplayOp {
                    program: 1,
                    groups: [op.workgroups.0, op.workgroups.1, op.workgroups.2],
                    bindings: vec![
                        sb(wqw_raw, wq_w_off, wq_w_size, 0, storage),
                        sb(opx_raw, op.x_offset, n_embd_bytes, 1, storage),
                        sb(opy_raw, op.output_offset, op.output_len, 2, storage),
                        sb(mm_meta.buffer, 0, 16, 3, uniform),
                    ],
                }),
                // 4. copy op.output_buffer → host-visible result
                ReplayStep::Copy {
                    src: opy_raw,
                    src_offset: op.output_offset,
                    dst: result.buffer,
                    dst_offset: 0,
                    size: (OUT_DIM * 4) as u64,
                },
            ];

            let graph = ReplayGraph::build_steps(&ctx, &programs, &steps).expect("graph");
            graph.run_token(&ctx).expect("run");

            let out_bytes = result.read(OUT_DIM * 4);
            let got: &[f32] = bytemuck::cast_slice(&out_bytes);
            let mut max_err = 0.0f32;
            for o in 0..OUT_DIM {
                max_err = max_err.max((got[o] - reference[o]).abs());
            }
            assert!(
                max_err < 2e-3,
                "attn_norm→copy→proj chain max abs err {max_err}"
            );

            graph.destroy(&ctx);
            result.destroy(&ctx);
            mm_meta.destroy(&ctx);
            let [norm_prog, mm_prog] = programs;
            norm_prog.destroy(&ctx);
            mm_prog.destroy(&ctx);
        }
    }

    /// The production capture path end-to-end: enable `begin_decode_capture`,
    /// run orangu's **real** `fused_layer` recording (which now emits a
    /// `CaptureStep` for its `attn_norm` dispatch), take the captured steps, and
    /// rebuild them via `ReplayGraph::from_capture`. Then feed a **fresh** input
    /// into the captured `x_buf` and replay — the recomputed `normed_buf` must
    /// match `rmsnorm(fresh_x)`, proving the capture emitted from the genuine
    /// recording replays correctly (not leftover wgpu state, since the input
    /// differs from the one `fused_layer` ran).
    #[test]
    fn decode_capture_from_real_recording_replays_q_projection() {
        use super::super::vulkan::{FusedAttnProjection, FusedLayerInput, GpuInput};

        let Some(backend) = shared_backend() else {
            eprintln!("skipping: no Vulkan adapter");
            return;
        };
        let device = backend.wgpu_device();
        let queue = backend.wgpu_queue();

        let n_embd = 64usize;
        let n_head = 4usize;
        let n_head_kv = 2usize;
        let head_dim = 8usize;
        let rope_dim = 8usize;
        let ffn_len = 32usize;
        let kv_dim = n_head_kv * head_dim;
        let eps = 1e-6f32;
        let f32_ty = crate::engine::quant::GGML_TYPE_F32;

        let mut s = 0x1234_5678u64;
        let mut rnd = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) & 0xFFFF) as f32 / 65535.0) - 0.5
        };
        let vecf = |n: usize, r: &mut dyn FnMut() -> f32| (0..n).map(|_| r()).collect::<Vec<f32>>();
        let mat = |ind: usize, outd: usize, r: &mut dyn FnMut() -> f32| {
            let w = vecf(ind * outd, r);
            super::super::super::loader::test_quant_matrix(
                bytemuck::cast_slice(&w),
                f32_ty,
                ind,
                outd,
            )
        };

        let attn_norm = vecf(n_embd, &mut rnd);
        // Keep wq's raw values for the Q-projection reference.
        let wq_raw = vecf(n_head * head_dim * n_embd, &mut rnd);
        let wq = super::super::super::loader::test_quant_matrix(
            bytemuck::cast_slice(&wq_raw),
            f32_ty,
            n_embd,
            n_head * head_dim,
        );
        let q_norm = vecf(head_dim, &mut rnd);
        let wk = mat(n_embd, kv_dim, &mut rnd);
        let k_norm = vecf(head_dim, &mut rnd);
        let wv_raw = vecf(kv_dim * n_embd, &mut rnd);
        let wv = super::super::super::loader::test_quant_matrix(
            bytemuck::cast_slice(&wv_raw),
            f32_ty,
            n_embd,
            kv_dim,
        );
        let wo = mat(n_head * head_dim, n_embd, &mut rnd);
        let attn_post_norm = vecf(n_embd, &mut rnd);
        let ffn_norm = vecf(n_embd, &mut rnd);
        let ffn_gate = mat(n_embd, ffn_len, &mut rnd);
        let ffn_up = mat(n_embd, ffn_len, &mut rnd);
        let ffn_down = mat(ffn_len, n_embd, &mut rnd);
        let ffn_post_norm = vecf(n_embd, &mut rnd);
        let x0 = vecf(n_embd, &mut rnd);

        let mut kv_cache = crate::engine::kv_cache::KvCache::new_with_dims(64, &[kv_dim]);

        // Capture the real recording of one layer.
        backend.begin_decode_capture();
        let _ = backend.fused_layer(FusedLayerInput {
            x: GpuInput::Cpu(&x0),
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
            rope_freq_base: 10000.0,
            freq_factors: None,
            eps,
            pos: 0,
            window_start: 0,
            window: None,
            scale: 1.0 / (head_dim as f32).sqrt(),
            cache: &mut kv_cache.layers[0],
            wo: &wo,
            attn_post_norm: &attn_post_norm,
            ffn_norm: &ffn_norm,
            ffn_gate: &ffn_gate,
            ffn_up: &ffn_up,
            ffn_down: &ffn_down,
            ffn_post_norm: &ffn_post_norm,
            ple: None,
            layer_output_scale: None,
            batch_slot: 0,
            attn_ts: None,
        });
        // Drop the per-token `HostInput` marker(s) — the layer-0 embedding
        // upload emits one, but it records no GPU op (the decode loop rewrites
        // it from the CPU), so the dispatch/copy index sequence below is defined
        // over the GPU-op steps only.
        let steps: Vec<CaptureStep> = backend
            .take_decode_capture()
            .expect("capture was begun")
            .into_iter()
            .filter(|s| !matches!(s, CaptureStep::HostInput { .. }))
            .collect();
        // Wired sites now capture, in order (the Q/K/V matmuls read the
        // attn-norm output directly on the shared-input default, so the three
        // `normed → {q,k,v}.x` copies are gone from the capture):
        // [attn_norm, q_mm, k_mm, v_mm, q_norm_rope, k_norm_rope, v_norm, ...].
        // KV-cast + attention sit after v_norm and aren't captured yet.
        assert!(
            steps.len() >= 7,
            "expected attn_norm + qkv + q/k norm+rope + v_norm"
        );
        let CaptureStep::Dispatch { bindings: nb, .. } = &steps[0] else {
            panic!("step 0 should be attn_norm dispatch");
        };
        let x_buf = nb[0].buffer.clone();
        let x_off = nb[0].offset;
        let out_dim = n_head * head_dim;
        let CaptureStep::Dispatch { bindings: qb, .. } = &steps[1] else {
            panic!("step 1 should be the q projection matmul");
        };
        let (q_out, q_out_off) = (qb[2].buffer.clone(), qb[2].offset);
        let CaptureStep::Dispatch { bindings: vb, .. } = &steps[3] else {
            panic!("step 3 should be the v projection matmul");
        };
        let (v_out, v_out_off) = (vb[2].buffer.clone(), vb[2].offset);

        // Replay at a nonzero position to exercise the per-token `pos` override
        // (capture-time pos was 0). Fresh input into x_buf.
        let replay_pos = 3usize;
        let freq_base = 10000.0f32;
        let x1: Vec<f32> = (0..n_embd)
            .map(|i| ((i as f32) * 0.031).cos() * 0.4)
            .collect();
        let mean_sq = x1.iter().map(|v| v * v).sum::<f32>() / n_embd as f32;
        let sc = 1.0 / (mean_sq + eps).sqrt();
        let normed: Vec<f32> = (0..n_embd).map(|i| x1[i] * sc * attn_norm[i]).collect();

        // Q reference: RoPE(per-head weighted rmsnorm(Wq·normed, q_norm), pos).
        let mut q_ref: Vec<f32> = (0..out_dim)
            .map(|o| {
                (0..n_embd)
                    .map(|e| wq_raw[o * n_embd + e] * normed[e])
                    .sum()
            })
            .collect();
        for h in 0..n_head {
            let head = &mut q_ref[h * head_dim..(h + 1) * head_dim];
            let ms: f32 = head.iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
            let s = 1.0 / (ms + eps).sqrt();
            for (i, v) in head.iter_mut().enumerate() {
                *v = *v * s * q_norm[i];
            }
        }
        crate::engine::tensor::rope_apply_scaled_inplace(
            &mut q_ref, n_head, head_dim, rope_dim, replay_pos, freq_base, None,
        );

        // V reference: per-head weightless rmsnorm(Wv·normed) (no RoPE on V).
        let mut v_ref: Vec<f32> = (0..kv_dim)
            .map(|o| {
                (0..n_embd)
                    .map(|e| wv_raw[o * n_embd + e] * normed[e])
                    .sum()
            })
            .collect();
        for h in 0..n_head_kv {
            let head = &mut v_ref[h * head_dim..(h + 1) * head_dim];
            let ms: f32 = head.iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
            let s = 1.0 / (ms + eps).sqrt();
            for v in head.iter_mut() {
                *v *= s;
            }
        }
        queue.write_buffer(&x_buf, x_off, bytemuck::cast_slice(&x1));
        queue.submit(std::iter::empty());
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");

        unsafe {
            let ctx = ReplayContext::from_wgpu(device).expect("vulkan context");
            // Replay the prefix up to and including V-norm (index 9).
            let (graph, programs) =
                ReplayGraph::from_capture(&ctx, &steps[0..7]).expect("from_capture");
            graph.update_per_token(replay_pos as u32);
            graph.run_token(&ctx).expect("run");

            // Read a device-local buffer region back via wgpu (the final barrier
            // makes the raw-submit writes visible to these transfers).
            let read_buf = |buf: &wgpu::Buffer, off: u64, len: usize| -> Vec<f32> {
                let rb = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("readback"),
                    size: (len * 4) as u64,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("readback enc"),
                });
                enc.copy_buffer_to_buffer(buf, off, &rb, 0, (len * 4) as u64);
                queue.submit(Some(enc.finish()));
                rb.slice(..).map_async(wgpu::MapMode::Read, |_| {});
                device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .expect("poll");
                let out = {
                    let view = rb.slice(..).get_mapped_range().expect("mapped range");
                    bytemuck::cast_slice::<u8, f32>(&view).to_vec()
                };
                rb.unmap();
                out
            };

            let got_q = read_buf(&q_out, q_out_off, out_dim);
            let got_v = read_buf(&v_out, v_out_off, kv_dim);
            let q_err = (0..out_dim)
                .map(|i| (got_q[i] - q_ref[i]).abs())
                .fold(0.0f32, f32::max);
            let v_err = (0..kv_dim)
                .map(|i| (got_v[i] - v_ref[i]).abs())
                .fold(0.0f32, f32::max);
            assert!(q_err < 2e-3, "captured Q-proj replay max abs err {q_err}");
            assert!(v_err < 2e-3, "captured V-norm replay max abs err {v_err}");

            graph.destroy(&ctx);
            for p in programs {
                p.destroy(&ctx);
            }

            // --- Attention half: replay through KV-cast + split-k attention
            // (steps 0..11, the shared-input path having dropped the 3 Q/K/V
            // copies) at pos=0,
            // single-position. With one attended position, softmax is 1.0 and the
            // attention output equals V[0] — the F16-cast, weightless per-head
            // rmsnorm of Wv·normed. This exercises the KV-cast (f32→f16 write at
            // KvWriteOffset 0), the split-k phase-1 read of the KV region, and the
            // reduce.
            assert!(
                steps.len() >= 11,
                "expected attention split+reduce captured"
            );
            for (i, want_split) in [(9usize, true), (10usize, false)] {
                let CaptureStep::Dispatch { per_token, .. } = &steps[i] else {
                    panic!("step {i} should be an attention dispatch");
                };
                // split has a per-token AttnSplitMeta; reduce's meta is static.
                assert_eq!(per_token[0].fields.is_empty(), !want_split);
            }
            let CaptureStep::Dispatch { bindings: rb, .. } = &steps[10] else {
                panic!("step 10 should be the attention reduce");
            };
            let (attn_out, attn_out_off) = (rb[2].buffer.clone(), rb[2].offset);

            let x2: Vec<f32> = (0..n_embd)
                .map(|i| ((i as f32) * 0.019).sin() * 0.5)
                .collect();
            let ms2 = x2.iter().map(|v| v * v).sum::<f32>() / n_embd as f32;
            let sc2 = 1.0 / (ms2 + eps).sqrt();
            let normed2: Vec<f32> = (0..n_embd).map(|i| x2[i] * sc2 * attn_norm[i]).collect();
            let mut vref2: Vec<f32> = (0..kv_dim)
                .map(|o| {
                    (0..n_embd)
                        .map(|e| wv_raw[o * n_embd + e] * normed2[e])
                        .sum()
                })
                .collect();
            for h in 0..n_head_kv {
                let head = &mut vref2[h * head_dim..(h + 1) * head_dim];
                let ms: f32 = head.iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
                let s = 1.0 / (ms + eps).sqrt();
                for v in head.iter_mut() {
                    *v *= s;
                }
            }
            queue.write_buffer(&x_buf, x_off, bytemuck::cast_slice(&x2));
            queue.submit(std::iter::empty());
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("poll");

            let ctx = ReplayContext::from_wgpu(device).expect("vulkan context");
            let (agraph, aprograms) =
                ReplayGraph::from_capture(&ctx, &steps[0..11]).expect("from_capture");
            agraph.update_per_token(0); // pos=0
            agraph.run_token(&ctx).expect("run");

            let read_buf2 = |buf: &wgpu::Buffer, off: u64, len: usize| -> Vec<f32> {
                let rb = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("attn readback"),
                    size: (len * 4) as u64,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("attn readback enc"),
                });
                enc.copy_buffer_to_buffer(buf, off, &rb, 0, (len * 4) as u64);
                queue.submit(Some(enc.finish()));
                rb.slice(..).map_async(wgpu::MapMode::Read, |_| {});
                device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .expect("poll");
                let out = {
                    let view = rb.slice(..).get_mapped_range().expect("mapped range");
                    bytemuck::cast_slice::<u8, f32>(&view).to_vec()
                };
                rb.unmap();
                out
            };
            let got_attn = read_buf2(&attn_out, attn_out_off, out_dim);
            let group = n_head / n_head_kv;
            let mut attn_err = 0.0f32;
            for h in 0..n_head {
                let kvh = h / group;
                for i in 0..head_dim {
                    let want = vref2[kvh * head_dim + i];
                    attn_err = attn_err.max((got_attn[h * head_dim + i] - want).abs());
                }
            }
            // F16-stored V → looser tolerance.
            assert!(
                attn_err < 2e-2,
                "captured attention replay max abs err {attn_err}"
            );

            agraph.destroy(&ctx);
            for p in aprograms {
                p.destroy(&ctx);
            }

            // --- Whole layer: replay EVERY captured step at pos=0 and compare
            // the layer output (x2, the last step's binding 3) against orangu's
            // own `fused_layer` for the same input. This is the full-layer
            // cross-check — every dispatch/copy/per-token uniform, end to end.
            let x_ref: Vec<f32> = (0..n_embd)
                .map(|i| ((i as f32) * 0.0071).sin() * 0.3 + 0.05)
                .collect();
            let mut ref_cache = crate::engine::kv_cache::KvCache::new_with_dims(64, &[kv_dim]);
            let reference = backend.fused_layer(FusedLayerInput {
                x: GpuInput::Cpu(&x_ref),
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
                rope_freq_base: 10000.0,
                freq_factors: None,
                eps,
                pos: 0,
                window_start: 0,
                window: None,
                scale: 1.0 / (head_dim as f32).sqrt(),
                cache: &mut ref_cache.layers[0],
                wo: &wo,
                attn_post_norm: &attn_post_norm,
                ffn_norm: &ffn_norm,
                ffn_gate: &ffn_gate,
                ffn_up: &ffn_up,
                ffn_down: &ffn_down,
                ffn_post_norm: &ffn_post_norm,
                ple: None,
                layer_output_scale: None,
                batch_slot: 0,
                attn_ts: None,
            });
            // Last captured step is ffn_post_norm_add (elem5); binding 3 = x2.
            let CaptureStep::Dispatch { bindings: lb, .. } = &steps[steps.len() - 1] else {
                panic!("last step should be a dispatch (ffn_post_norm_add)");
            };
            let (x2_buf, x2_off) = (lb[3].buffer.clone(), lb[3].offset);

            queue.write_buffer(&x_buf, x_off, bytemuck::cast_slice(&x_ref));
            queue.submit(std::iter::empty());
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("poll");
            let ctx = ReplayContext::from_wgpu(device).expect("vulkan context");
            let (fgraph, fprograms) =
                ReplayGraph::from_capture(&ctx, &steps).expect("from_capture");
            fgraph.update_per_token(0);
            fgraph.run_token(&ctx).expect("run");
            let got_layer = read_buf2(&x2_buf, x2_off, n_embd);
            let mut layer_err = 0.0f32;
            for i in 0..n_embd {
                layer_err = layer_err.max((got_layer[i] - reference[i]).abs());
            }
            assert!(
                layer_err < 2e-2,
                "captured full-layer replay max abs err {layer_err}"
            );

            fgraph.destroy(&ctx);
            for p in fprograms {
                p.destroy(&ctx);
            }
        }
    }

    /// Proves the per-token uniform mechanism through `from_capture`: a captured
    /// dispatch writes `f32(m.pos)` from a per-token uniform whose `pos` field
    /// `update_per_token` rewrites each token. The same recorded command buffer,
    /// resubmitted, must reflect the new `pos` — exactly how norm+RoPE / KV-cast
    /// / attention get their per-token position without any wgpu submit.
    #[test]
    fn per_token_uniform_updates_across_tokens() {
        let Some(backend) = shared_backend() else {
            eprintln!("skipping: no Vulkan adapter");
            return;
        };
        let device = backend.wgpu_device();
        let queue = backend.wgpu_queue();

        // out[0] = f32(m.pos); the uniform mirrors FusedNormRopeMeta's layout
        // enough that `pos` sits at byte 12 (n_head, head_dim, rope_dim, pos).
        let wgsl = r#"
struct M { n_head: u32, head_dim: u32, rope_dim: u32, pos: u32,
           freq_base: f32, eps: f32, p0: u32, p1: u32 }
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<uniform> m: M;
@compute @workgroup_size(1)
fn main() { out[0] = f32(m.pos); }
"#;
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pt out"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Realize the buffer's backing allocation before as_hal reads its handle.
        queue.write_buffer(&out_buf, 0, &[0u8; 16]);
        queue.submit(std::iter::empty());
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");

        let steps = vec![CaptureStep::Dispatch {
            wgsl: wgsl.to_string(),
            bindings: vec![CaptureBinding {
                binding: 0,
                kind: DescriptorKind::Storage,
                buffer: out_buf.clone(),
                offset: 0,
                size: 16,
            }],
            per_token: vec![PerTokenBinding {
                binding: 1,
                init_bytes: vec![0u8; 32],
                fields: vec![PerTokenField::Pos { byte_offset: 12 }],
            }],
            groups: [1, 1, 1],
        }];

        unsafe {
            let ctx = ReplayContext::from_wgpu(device).expect("vulkan context");
            let (graph, programs) = ReplayGraph::from_capture(&ctx, &steps).expect("from_capture");

            for &pos in &[5u32, 9u32, 3u32] {
                graph.update_per_token(pos);
                graph.run_token(&ctx).expect("run");

                let rb = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("pt rb"),
                    size: 16,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pt rb enc"),
                });
                enc.copy_buffer_to_buffer(&out_buf, 0, &rb, 0, 16);
                queue.submit(Some(enc.finish()));
                rb.slice(..).map_async(wgpu::MapMode::Read, |_| {});
                device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .expect("poll");
                let got = {
                    let view = rb.slice(..).get_mapped_range().expect("mapped range");
                    bytemuck::cast_slice::<u8, f32>(&view)[0]
                };
                rb.unmap();
                assert_eq!(got, pos as f32, "per-token pos={pos} not reflected");
            }

            graph.destroy(&ctx);
            for p in programs {
                p.destroy(&ctx);
            }
        }
    }
}
