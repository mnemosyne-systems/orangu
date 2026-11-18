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

//! **Holding the card's clock across the host's turns.** A decode step on
//! a model whose experts live in host memory alternates between the card
//! (the attention or recurrent projections, the shared expert) and the
//! host (the routed experts); each host turn leaves the card idle for a
//! few milliseconds, its governor parks the clock, and every device call
//! after a host turn runs at a fraction of its rate. Measured on a hybrid
//! model: more than half of a token's time in device calls that would
//! take a tenth of it warm.
//!
//! A spin dispatch on the queue the work uses cannot help — the queue is
//! in order, so the spin sits in front of the work it would warm. This
//! module opens a **second logical device** on the same card through the
//! raw API, with one queue from a compute-only family where the card has
//! one (its own hardware ring, scheduled beside the graphics ring rather
//! than in its order), and runs a one-workgroup spin on it for as long as
//! a step is armed. One compute unit's power, no ordering with anything.
//!
//! Compiled per target like the replay module: Apple targets have no
//! Vulkan and get a holder that does nothing.

#[cfg(not(target_vendor = "apple"))]
mod imp {
    use ash::vk;
    use std::ffi::CString;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Condvar, Mutex};

    type Vulkan = wgpu::hal::api::Vulkan;

    /// One spin submission: the loop count is the card's own time, and
    /// the count is resubmitted while armed, so it only has to be short
    /// enough that disarming takes effect within a millisecond or so.
    const SPIN_ITERATIONS: u32 = 20_000;

    const SPIN_WGSL: &str = r#"
struct SpinParams { iterations: u32 }
@group(0) @binding(0) var<storage, read_write> sink: array<f32>;
@group(0) @binding(1) var<uniform> params: SpinParams;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var acc: f32 = sink[gid.x % 16u];
    var i: u32 = 0u;
    loop {
        if (i >= params.iterations) { break; }
        acc = fma(acc, 1.000001, 0.5);
        i = i + 1u;
    }
    if (acc == 12345.678) { sink[gid.x % 16u] = acc; }
}
"#;

    struct Raw {
        device: ash::Device,
        queue: vk::Queue,
        pool: vk::CommandPool,
        cmd: vk::CommandBuffer,
        fence: vk::Fence,
        pipeline: vk::Pipeline,
        pipeline_layout: vk::PipelineLayout,
        set_layout: vk::DescriptorSetLayout,
        descriptor_pool: vk::DescriptorPool,
        module: vk::ShaderModule,
        buffers: [(vk::Buffer, vk::DeviceMemory); 2],
    }

    // The raw handles are used from the holder's own thread only.
    unsafe impl Send for Raw {}

    impl Drop for Raw {
        fn drop(&mut self) {
            unsafe {
                let _ = self.device.device_wait_idle();
                self.device.destroy_fence(self.fence, None);
                self.device.destroy_command_pool(self.pool, None);
                self.device.destroy_pipeline(self.pipeline, None);
                self.device
                    .destroy_pipeline_layout(self.pipeline_layout, None);
                self.device
                    .destroy_descriptor_pool(self.descriptor_pool, None);
                self.device
                    .destroy_descriptor_set_layout(self.set_layout, None);
                self.device.destroy_shader_module(self.module, None);
                for (buffer, memory) in self.buffers {
                    self.device.destroy_buffer(buffer, None);
                    self.device.free_memory(memory, None);
                }
                self.device.destroy_device(None);
            }
        }
    }

    pub struct ClockHold {
        armed: Arc<(Mutex<bool>, Condvar)>,
        stop: Arc<AtomicBool>,
        thread: Option<std::thread::JoinHandle<()>>,
    }

    impl ClockHold {
        /// Opens the second device and starts its thread, parked until
        /// [`Self::arm`]. `None` where the device is not Vulkan, or the
        /// raw setup fails (which is reported and is not an error of the
        /// model's).
        pub fn new(device: &wgpu::Device) -> Option<Self> {
            let raw = match unsafe { Self::open(device) } {
                Ok(raw) => raw,
                Err(e) => {
                    log::info!("orangu-server: [vulkan] no clock holder: {e}");
                    return None;
                }
            };
            let armed = Arc::new((Mutex::new(false), Condvar::new()));
            let stop = Arc::new(AtomicBool::new(false));
            let thread = {
                let armed = armed.clone();
                let stop = stop.clone();
                std::thread::Builder::new()
                    .name("orangu-clock-hold".into())
                    .spawn(move || Self::run(raw, &armed, &stop))
                    .ok()?
            };
            Some(Self {
                armed,
                stop,
                thread: Some(thread),
            })
        }

        pub fn arm(&self) {
            let (lock, cv) = &*self.armed;
            *lock.lock().unwrap_or_else(|p| p.into_inner()) = true;
            cv.notify_one();
        }

        pub fn disarm(&self) {
            let (lock, _) = &*self.armed;
            *lock.lock().unwrap_or_else(|p| p.into_inner()) = false;
        }

        fn run(raw: Raw, armed: &(Mutex<bool>, Condvar), stop: &AtomicBool) {
            let (lock, cv) = armed;
            while !stop.load(Ordering::Relaxed) {
                {
                    let mut on = lock.lock().unwrap_or_else(|p| p.into_inner());
                    while !*on && !stop.load(Ordering::Relaxed) {
                        on = cv.wait(on).unwrap_or_else(|p| p.into_inner());
                    }
                }
                if stop.load(Ordering::Relaxed) {
                    break;
                }
                if unsafe { Self::spin_once(&raw) }.is_err() {
                    // A failed submission is a holder that has stopped
                    // holding, not a step that has failed.
                    break;
                }
            }
            drop(raw);
        }

        unsafe fn spin_once(raw: &Raw) -> Result<(), vk::Result> {
            unsafe {
                raw.device.reset_fences(&[raw.fence])?;
                let cmds = [raw.cmd];
                let submit = vk::SubmitInfo::default().command_buffers(&cmds);
                raw.device.queue_submit(raw.queue, &[submit], raw.fence)?;
                raw.device.wait_for_fences(&[raw.fence], true, u64::MAX)?;
            }
            Ok(())
        }

        unsafe fn open(device: &wgpu::Device) -> Result<Raw, String> {
            let hal = unsafe { device.as_hal::<Vulkan>() }.ok_or("not a Vulkan device")?;
            let instance = hal.shared_instance().raw_instance().clone();
            let phys = hal.raw_physical_device();
            let api_version = unsafe { instance.get_physical_device_properties(phys) }.api_version;
            drop(hal);

            // A compute-only family where there is one — its own hardware
            // ring — else the first family with compute.
            let families = unsafe { instance.get_physical_device_queue_family_properties(phys) };
            let compute_only = families.iter().position(|f| {
                f.queue_flags.contains(vk::QueueFlags::COMPUTE)
                    && !f.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                    && f.queue_count > 0
            });
            let family = compute_only
                .or_else(|| {
                    families
                        .iter()
                        .position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
                })
                .ok_or("no compute queue family")? as u32;
            let priorities = [0.0f32];
            let queue_info = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(family)
                .queue_priorities(&priorities);
            let queue_infos = [queue_info];
            let device_info = vk::DeviceCreateInfo::default().queue_create_infos(&queue_infos);
            let raw_device = unsafe { instance.create_device(phys, &device_info, None) }
                .map_err(|e| format!("create_device: {e}"))?;
            let queue = unsafe { raw_device.get_device_queue(family, 0) };

            let spirv = super::super::vulkan_replay::compile_wgsl_to_spirv(SPIN_WGSL, api_version)?;
            let module_info = vk::ShaderModuleCreateInfo::default().code(&spirv);
            let module = unsafe { raw_device.create_shader_module(&module_info, None) }
                .map_err(|e| format!("create_shader_module: {e}"))?;
            let bindings = [
                vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            ];
            let set_layout = unsafe {
                raw_device.create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
            }
            .map_err(|e| format!("create_descriptor_set_layout: {e}"))?;
            let set_layouts = [set_layout];
            let pipeline_layout = unsafe {
                raw_device.create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts),
                    None,
                )
            }
            .map_err(|e| format!("create_pipeline_layout: {e}"))?;
            let entry = CString::new("main").expect("a literal");
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(&entry);
            let pipeline = unsafe {
                raw_device.create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(stage)
                        .layout(pipeline_layout)],
                    None,
                )
            }
            .map_err(|(_, e)| format!("create_compute_pipelines: {e}"))?[0];

            // Two tiny host-visible buffers: the sink and the meta.
            let mem_props = unsafe { instance.get_physical_device_memory_properties(phys) };
            let make = |size: u64,
                        usage: vk::BufferUsageFlags|
             -> Result<(vk::Buffer, vk::DeviceMemory), String> {
                let buffer = unsafe {
                    raw_device.create_buffer(
                        &vk::BufferCreateInfo::default()
                            .size(size)
                            .usage(usage)
                            .sharing_mode(vk::SharingMode::EXCLUSIVE),
                        None,
                    )
                }
                .map_err(|e| format!("create_buffer: {e}"))?;
                let req = unsafe { raw_device.get_buffer_memory_requirements(buffer) };
                let wanted =
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
                let index = (0..mem_props.memory_type_count)
                    .find(|&i| {
                        req.memory_type_bits & (1 << i) != 0
                            && mem_props.memory_types[i as usize]
                                .property_flags
                                .contains(wanted)
                    })
                    .ok_or("no host-visible memory type")?;
                let memory = unsafe {
                    raw_device.allocate_memory(
                        &vk::MemoryAllocateInfo::default()
                            .allocation_size(req.size)
                            .memory_type_index(index),
                        None,
                    )
                }
                .map_err(|e| format!("allocate_memory: {e}"))?;
                unsafe { raw_device.bind_buffer_memory(buffer, memory, 0) }
                    .map_err(|e| format!("bind_buffer_memory: {e}"))?;
                Ok((buffer, memory))
            };
            let sink = make(64, vk::BufferUsageFlags::STORAGE_BUFFER)?;
            let meta = make(16, vk::BufferUsageFlags::UNIFORM_BUFFER)?;
            unsafe {
                let p = raw_device
                    .map_memory(meta.1, 0, 16, vk::MemoryMapFlags::empty())
                    .map_err(|e| format!("map_memory: {e}"))?;
                (p as *mut u32).write(SPIN_ITERATIONS);
                raw_device.unmap_memory(meta.1);
            }

            let pool_sizes = [
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1),
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(1),
            ];
            let descriptor_pool = unsafe {
                raw_device.create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(1)
                        .pool_sizes(&pool_sizes),
                    None,
                )
            }
            .map_err(|e| format!("create_descriptor_pool: {e}"))?;
            let set = unsafe {
                raw_device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(&set_layouts),
                )
            }
            .map_err(|e| format!("allocate_descriptor_sets: {e}"))?[0];
            let sink_info = [vk::DescriptorBufferInfo::default()
                .buffer(sink.0)
                .offset(0)
                .range(64)];
            let meta_info = [vk::DescriptorBufferInfo::default()
                .buffer(meta.0)
                .offset(0)
                .range(16)];
            unsafe {
                raw_device.update_descriptor_sets(
                    &[
                        vk::WriteDescriptorSet::default()
                            .dst_set(set)
                            .dst_binding(0)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&sink_info),
                        vk::WriteDescriptorSet::default()
                            .dst_set(set)
                            .dst_binding(1)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                            .buffer_info(&meta_info),
                    ],
                    &[],
                );
            }

            let pool = unsafe {
                raw_device.create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(family),
                    None,
                )
            }
            .map_err(|e| format!("create_command_pool: {e}"))?;
            let cmd = unsafe {
                raw_device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
            }
            .map_err(|e| format!("allocate_command_buffers: {e}"))?[0];
            unsafe {
                raw_device
                    .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
                    .map_err(|e| format!("begin_command_buffer: {e}"))?;
                raw_device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
                raw_device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline_layout,
                    0,
                    &[set],
                    &[],
                );
                raw_device.cmd_dispatch(cmd, 1, 1, 1);
                raw_device
                    .end_command_buffer(cmd)
                    .map_err(|e| format!("end_command_buffer: {e}"))?;
            }
            let fence = unsafe { raw_device.create_fence(&vk::FenceCreateInfo::default(), None) }
                .map_err(|e| format!("create_fence: {e}"))?;
            log::info!(
                "orangu-server: [vulkan] clock holder on queue family {family}{}",
                if compute_only.is_some() {
                    " (compute-only)"
                } else {
                    ""
                }
            );
            Ok(Raw {
                device: raw_device,
                queue,
                pool,
                cmd,
                fence,
                pipeline,
                pipeline_layout,
                set_layout,
                descriptor_pool,
                module,
                buffers: [sink, meta],
            })
        }
    }

    impl Drop for ClockHold {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
            self.arm();
            if let Some(t) = self.thread.take() {
                let _ = t.join();
            }
        }
    }
}

#[cfg(target_vendor = "apple")]
mod imp {
    pub struct ClockHold;
    impl ClockHold {
        pub fn new(_device: &wgpu::Device) -> Option<Self> {
            None
        }
        pub fn arm(&self) {}
        pub fn disarm(&self) {}
    }
}

pub use imp::ClockHold;
