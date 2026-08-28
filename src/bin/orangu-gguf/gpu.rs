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

//! The matmul on a GPU, when there is one worth using.
//!
//! **Device order is fixed: discrete, then integrated, then the CPU.** A
//! discrete card has its own memory and its own bandwidth; an integrated
//! one shares both with the process that is feeding it, so it is the
//! second choice rather than the first; and the CPU path is always there,
//! because a machine with no usable adapter still has to train.
//!
//! What is *not* obvious, and is why this module starts with a
//! measurement rather than an integration: a dispatch costs tens of
//! microseconds before it computes anything, and moving a matrix across
//! PCIe costs more than that again. A small matmul loses to sixteen CPU
//! cores no matter how fast the card is. [`Gpu::matmul`] is deliberately
//! the naive shape — upload, dispatch, read back — so that the benchmark
//! measures the *whole* cost of using a GPU this way, which is the number
//! that decides whether a resident-weight implementation is worth
//! building.

use anyhow::{Context, Result};
use std::borrow::Cow;

/// What a candidate device is, in the order this tool wants them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Class {
    /// Its own memory, its own bandwidth. First choice.
    Discrete,
    /// Shares system memory with the process feeding it. Second.
    Integrated,
    /// A software or virtual adapter — slower than the CPU path it would
    /// be standing in for, so never chosen.
    Other,
}

#[derive(Debug, Clone)]
pub struct Device {
    /// Position in the driver's own enumeration — not this list's order.
    #[allow(dead_code)]
    pub index: usize,
    pub name: String,
    pub class: Class,
}

fn classify(kind: wgpu::DeviceType) -> Class {
    match kind {
        wgpu::DeviceType::DiscreteGpu => Class::Discrete,
        wgpu::DeviceType::IntegratedGpu => Class::Integrated,
        _ => Class::Other,
    }
}

fn instance() -> wgpu::Instance {
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    })
}

/// Every adapter, in this tool's preference order: discrete first,
/// integrated next, everything else last and never used.
pub fn devices() -> Vec<Device> {
    let instance = instance();
    let mut found: Vec<Device> =
        pollster::block_on(instance.enumerate_adapters(
            wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12,
        ))
        .iter()
        .enumerate()
        .map(|(index, adapter)| {
            let info = adapter.get_info();
            Device {
                index,
                name: info.name,
                class: classify(info.device_type),
            }
        })
        .collect();
    // Stable within a class, so the order the driver reports still decides
    // between two cards of the same kind.
    found.sort_by_key(|d| d.class);
    found
}

/// A device and the one pipeline this module runs on it.
///
/// Not on the training path — see the module comment and `GGUF.md`'s T6
/// for the measurement that keeps it off. It is kept, and kept tested,
/// because the device policy above is a requirement and because the
/// numbers a real implementation has to beat came from exactly this code.
#[allow(dead_code)]
pub struct Gpu {
    pub device: Device,
    gpu: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

/// `y[t, n] = x[t, k] . w[n, k]`, one thread per output element.
///
/// No tiling and no shared memory: this is the shape that answers "is a
/// GPU worth it here at all", and a tiled kernel would only be worth
/// writing once that answer is yes.
const MATMUL_WGSL: &str = r#"
struct Meta { t: u32, k: u32, n: u32, span: u32 };

@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read>       w: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform>          dims: Meta;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // A dispatch dimension caps at 65535 groups, which a 512x8192 output
    // exceeds, so the grid is two-dimensional and `span` is how wide one
    // row of it is.
    let idx = gid.y * dims.span + gid.x;
    if (idx >= dims.t * dims.n) { return; }
    let row = idx / dims.n;
    let col = idx % dims.n;
    var sum = 0.0;
    for (var i = 0u; i < dims.k; i = i + 1u) {
        sum = sum + x[row * dims.k + i] * w[col * dims.k + i];
    }
    y[idx] = sum;
}
"#;

#[allow(dead_code)]
impl Gpu {
    /// Brings up the best available device, or `None` when there is no
    /// adapter this tool will use — in which case the caller trains on the
    /// CPU, which is not a failure.
    pub fn best() -> Option<Gpu> {
        devices()
            .into_iter()
            .find(|d| d.class != Class::Other)
            .and_then(|device| Gpu::open(device).ok())
    }

    pub fn open(device: Device) -> Result<Gpu> {
        let instance = instance();
        let adapters = pollster::block_on(instance.enumerate_adapters(
            wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12,
        ));
        let adapter = adapters
            .get(device.index)
            .context("the adapter went away between enumeration and use")?;

        let (gpu, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("orangu-gguf"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .context("creating the compute device")?;

        let shader = gpu.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matmul"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(MATMUL_WGSL)),
        });
        let pipeline = gpu.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("matmul"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let layout = pipeline.get_bind_group_layout(0);
        Ok(Gpu {
            device,
            gpu,
            queue,
            pipeline,
            layout,
        })
    }

    /// `y[t, n] = x[t, k] . w[n, k]`, uploading both operands and reading
    /// the result back.
    pub fn matmul(&self, y: &mut [f32], x: &[f32], w: &[f32], t: usize, k: usize, n: usize) {
        use wgpu::util::DeviceExt;

        let storage = wgpu::BufferUsages::STORAGE;
        let x_buffer = self
            .gpu
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("x"),
                contents: bytemuck::cast_slice(x),
                usage: storage,
            });
        let w_buffer = self
            .gpu
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("w"),
                contents: bytemuck::cast_slice(w),
                usage: storage,
            });
        let y_bytes = (t * n * 4) as u64;
        let y_buffer = self.gpu.create_buffer(&wgpu::BufferDescriptor {
            label: Some("y"),
            size: y_bytes,
            usage: storage | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = self.gpu.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: y_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        // Workgroups needed, split over two dimensions because one is
        // capped at 65535.
        const GROUP: u32 = 64;
        const MAX_GROUPS: u32 = 65535;
        let groups = ((t * n) as u32).div_ceil(GROUP);
        let groups_x = groups.min(MAX_GROUPS);
        let groups_y = groups.div_ceil(groups_x);
        let meta = [t as u32, k as u32, n as u32, groups_x * GROUP];
        let meta_buffer = self
            .gpu
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("meta"),
                contents: bytemuck::cast_slice(&meta),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind = self.gpu.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: w_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: y_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: meta_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .gpu
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        }
        encoder.copy_buffer_to_buffer(&y_buffer, 0, &readback, 0, y_bytes);
        self.queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.gpu.poll(wgpu::PollType::wait_indefinitely());
        {
            let view = slice
                .get_mapped_range()
                .expect("the readback buffer was mapped above");
            y.copy_from_slice(bytemuck::cast_slice(&view));
        }
        readback.unmap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The order is the requirement: discrete before integrated, and a
    /// software adapter never chosen.
    #[test]
    fn devices_come_back_in_preference_order() {
        let found = devices();
        let mut classes: Vec<Class> = found.iter().map(|d| d.class).collect();
        let sorted = {
            let mut c = classes.clone();
            c.sort();
            c
        };
        assert_eq!(classes, sorted, "devices are not in preference order");
        classes.dedup();
        for device in &found {
            println!("  [{}] {:?}  {}", device.index, device.class, device.name);
        }
        if let Some(best) = found.iter().find(|d| d.class != Class::Other) {
            assert!(matches!(best.class, Class::Discrete | Class::Integrated));
        }
    }

    /// Is a GPU worth it at these shapes? Run with
    /// `cargo test --release --bin orangu-gguf gpu_versus_cpu -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn gpu_versus_cpu() {
        let Some(gpu) = Gpu::best() else {
            println!("no usable adapter");
            return;
        };
        println!("device: {} ({:?})\n", gpu.device.name, gpu.device.class);
        println!(
            "{:>26}  {:>10}  {:>10}  {:>8}",
            "shape (t x k x n)", "cpu", "gpu", "ratio"
        );

        // The smoke model's four matmul shapes at a 512-token sequence,
        // then two `2b` shapes for scale.
        for (label, t, k, n) in [
            ("smoke attn", 512, 256, 256),
            ("smoke ffn up", 512, 256, 688),
            ("smoke ffn down", 512, 688, 256),
            ("smoke head", 512, 256, 4096),
            // Nothing larger: a single naive dispatch over a `2b`-shaped
            // output runs long enough to trip the driver's watchdog, and a
            // lost device takes the whole process with it. Bounding a
            // dispatch is a requirement of any real implementation, the
            // same way the inference side chunks its prefill.
            ("2b attn", 256, 2048, 2048),
        ] {
            let x: Vec<f32> = (0..t * k).map(|i| (i % 13) as f32 * 0.01).collect();
            let w: Vec<f32> = (0..n * k).map(|i| (i % 7) as f32 * 0.02).collect();
            let mut y = vec![0f32; t * n];
            let flop = 2.0 * (t * k * n) as f64;

            // Warm both paths before timing either.
            crate::model::matmul(&mut y, &x, &w, t, k, n);
            gpu.matmul(&mut y, &x, &w, t, k, n);

            let start = std::time::Instant::now();
            for _ in 0..5 {
                crate::model::matmul(&mut y, &x, &w, t, k, n);
            }
            let cpu = start.elapsed().as_secs_f64() / 5.0;

            let start = std::time::Instant::now();
            for _ in 0..5 {
                gpu.matmul(&mut y, &x, &w, t, k, n);
            }
            let gpu_time = start.elapsed().as_secs_f64() / 5.0;

            println!(
                "{label:>14} {t:4}x{k:4}x{n:5}  {:>7.1} GF/s  {:>7.1} GF/s  {:>7.2}x",
                flop / cpu / 1e9,
                flop / gpu_time / 1e9,
                cpu / gpu_time
            );
        }
    }

    /// What the GPU computes has to be what the CPU computes.
    #[test]
    fn a_gpu_matmul_matches_the_cpu() {
        let Some(gpu) = Gpu::best() else {
            println!("no usable adapter; skipping");
            return;
        };
        let (t, k, n) = (8usize, 64usize, 16usize);
        let x: Vec<f32> = (0..t * k).map(|i| (i % 13) as f32 * 0.1 - 0.6).collect();
        let w: Vec<f32> = (0..n * k).map(|i| (i % 7) as f32 * 0.2 - 0.7).collect();

        let mut want = vec![0f32; t * n];
        crate::model::matmul(&mut want, &x, &w, t, k, n);
        let mut got = vec![0f32; t * n];
        gpu.matmul(&mut got, &x, &w, t, k, n);

        for (i, (a, b)) in want.iter().zip(got.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "element {i}: cpu {a}, {} {b}",
                gpu.device.name
            );
        }
    }
}
