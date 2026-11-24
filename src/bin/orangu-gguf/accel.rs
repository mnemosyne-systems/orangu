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

//! Training's matrix products on a graphics card.
//!
//! One tiled kernel computes `C[m, n] = A[m, p] · B[p, n]` with either
//! operand read through a pair of strides, so a layer's forward
//! (`x · wᵀ`), input gradient (`dy · w`) and weight gradient (`dyᵀ · x`)
//! are all one pipeline. A workgroup owns a 64 × 64 block of `C`, sixteen
//! by sixteen threads of four by four each, and walks the shared dimension
//! sixteen terms at a time through workgroup memory.
//!
//! Buffers are kept per device and reused: an upload is a queue write into
//! a buffer that is already there, and a result comes back through one
//! mapped staging buffer.

use crate::gemm::View;
use crate::gpu::{self, Class};
use anyhow::{Context, Result};
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Bumped each time the optimizer changes the weights, so a copy kept on
/// the card from before is written again before it is read.
static GENERATION: AtomicU64 = AtomicU64::new(0);

/// The weights changed: every copy on a card is out of date.
pub fn weights_changed() {
    GENERATION.fetch_add(1, Ordering::Relaxed);
}

/// A weight matrix kept on the card, and the generation it holds.
struct Kept {
    buffer: wgpu::Buffer,
    /// Held while the copy is checked and written, so no product is
    /// submitted against a copy another thread has stamped but not yet
    /// queued.
    generation: Mutex<u64>,
}

/// How much of the card the kept weights may take, in bytes;
/// `ORANGU_GGUF_CARD_RESIDENT_MIB` moves it.
fn resident_budget() -> u64 {
    std::env::var("ORANGU_GGUF_CARD_RESIDENT_MIB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map_or(2 << 30, |mib| mib << 20)
}

const GEMM_WGSL: &str = r#"
struct Dims {
    n: u32, add: u32, pad1: u32, pad2: u32,
    a_row: u32, a_col: u32, b_row: u32, b_col: u32,
};

// One product of a dispatch: where its A, B and C start, its rows, and
// the terms it sums.
struct Group {
    a: u32, rows: u32, b: u32, c: u32,
    depth: u32, pad0: u32, pad1: u32, pad2: u32,
};

@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform>          dims: Dims;
@group(0) @binding(4) var<storage, read>  groups: array<Group>;

const TILE: u32 = 64u;
const DEPTH: u32 = 16u;

// [term][row] and [term][column], so the inner loop reads four
// consecutive floats of each.
var<workgroup> ta: array<f32, 1024>;
var<workgroup> tb: array<f32, 1024>;
var<workgroup> mine: Group;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
    @builtin(local_invocation_index) index: u32,
) {
    if (index == 0u) { mine = groups[group.z]; }
    // The same for every thread of the workgroup, so the loop around the
    // barriers below is too.
    let g = workgroupUniformLoad(&mine);
    let row0 = group.y * TILE;
    let col0 = group.x * TILE;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    var first = 0u;
    loop {
        if (first >= g.depth || row0 >= g.rows) { break; }
        // 64 rows x 16 terms of A and 16 terms x 64 columns of B: four
        // floats a thread each, in whichever order walks memory forwards.
        for (var e = 0u; e < 4u; e = e + 1u) {
            let flat = index + e * 256u;
            var r: u32; var q: u32;
            if (dims.a_col == 1u) { r = flat / DEPTH; q = flat % DEPTH; }
            else { r = flat % TILE; q = flat / TILE; }
            let row = row0 + r;
            let term = first + q;
            var v = 0.0;
            if (row < g.rows && term < g.depth) { v = a[g.a + row * dims.a_row + term * dims.a_col]; }
            ta[q * 64u + r] = v;

            var j: u32; var s: u32;
            if (dims.b_row == 1u) { j = flat / DEPTH; s = flat % DEPTH; }
            else { j = flat % TILE; s = flat / TILE; }
            let col = col0 + j;
            let term_b = first + s;
            var u = 0.0;
            if (col < dims.n && term_b < g.depth) { u = b[g.b + term_b * dims.b_row + col * dims.b_col]; }
            tb[s * 64u + j] = u;
        }
        workgroupBarrier();
        for (var q = 0u; q < DEPTH; q = q + 1u) {
            let base_a = q * 64u + local.y * 4u;
            let av = vec4<f32>(ta[base_a], ta[base_a + 1u], ta[base_a + 2u], ta[base_a + 3u]);
            let base_b = q * 64u + local.x * 4u;
            let bv = vec4<f32>(tb[base_b], tb[base_b + 1u], tb[base_b + 2u], tb[base_b + 3u]);
            acc0 = fma(vec4<f32>(av.x), bv, acc0);
            acc1 = fma(vec4<f32>(av.y), bv, acc1);
            acc2 = fma(vec4<f32>(av.z), bv, acc2);
            acc3 = fma(vec4<f32>(av.w), bv, acc3);
        }
        workgroupBarrier();
        first = first + DEPTH;
    }
    let row = row0 + local.y * 4u;
    let col = col0 + local.x * 4u;
    var accs = array<vec4<f32>, 4>(acc0, acc1, acc2, acc3);
    for (var i = 0u; i < 4u; i = i + 1u) {
        if (row + i >= g.rows) { break; }
        for (var j = 0u; j < 4u; j = j + 1u) {
            if (col + j < dims.n) {
                let at = g.c + (row + i) * dims.n + col + j;
                if (dims.add == 1u) { c[at] = c[at] + accs[i][j]; } else { c[at] = accs[i][j]; }
            }
        }
    }
}
"#;

/// The same product without workgroup memory: each thread reads its four
/// rows of `a` and four columns of `b` from the buffers and sums a 4x4
/// block. A GPU whose workgroup memory is ordinary cached memory pays for
/// the staging and the barriers of [`GEMM_WGSL`] and saves no reads.
const GEMM_DIRECT_WGSL: &str = r#"
struct Dims {
    n: u32, add: u32, pad1: u32, pad2: u32,
    a_row: u32, a_col: u32, b_row: u32, b_col: u32,
};

struct Group {
    a: u32, rows: u32, b: u32, c: u32,
    depth: u32, pad0: u32, pad1: u32, pad2: u32,
};

@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform>          dims: Dims;
@group(0) @binding(4) var<storage, read>  groups: array<Group>;

const TILE: u32 = 32u;

@compute @workgroup_size(8, 8)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let g = groups[group.z];
    let row = group.y * TILE + local.y * 4u;
    let col = group.x * TILE + local.x * 4u;
    if (row >= g.rows || col >= dims.n) { return; }
    // Rows and columns past the edge read the last one and are not stored.
    let last_row = g.rows - 1u;
    let last_col = dims.n - 1u;
    let pa = vec4<u32>(row, min(row + 1u, last_row), min(row + 2u, last_row), min(row + 3u, last_row)) * dims.a_row + vec4<u32>(g.a);
    let pb = vec4<u32>(col, min(col + 1u, last_col), min(col + 2u, last_col), min(col + 3u, last_col)) * dims.b_col + vec4<u32>(g.b);
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    for (var q = 0u; q < g.depth; q = q + 1u) {
        let ta = pa + vec4<u32>(q * dims.a_col);
        let tb = pb + vec4<u32>(q * dims.b_row);
        let bv = vec4<f32>(b[tb.x], b[tb.y], b[tb.z], b[tb.w]);
        acc0 = fma(vec4<f32>(a[ta.x]), bv, acc0);
        acc1 = fma(vec4<f32>(a[ta.y]), bv, acc1);
        acc2 = fma(vec4<f32>(a[ta.z]), bv, acc2);
        acc3 = fma(vec4<f32>(a[ta.w]), bv, acc3);
    }
    var accs = array<vec4<f32>, 4>(acc0, acc1, acc2, acc3);
    for (var i = 0u; i < 4u; i = i + 1u) {
        if (row + i >= g.rows) { break; }
        for (var j = 0u; j < 4u; j = j + 1u) {
            if (col + j < dims.n) {
                let at = g.c + (row + i) * dims.n + col + j;
                if (dims.add == 1u) { c[at] = c[at] + accs[i][j]; } else { c[at] = accs[i][j]; }
            }
        }
    }
}
"#;

/// A product kernel: its pipeline, its bindings and the output tile one
/// workgroup covers.
struct Kernel {
    name: &'static str,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    tile: usize,
}

/// The kernels a card runs products with: name, source and the output tile
/// one workgroup covers. The first is used unless the probe picks another.
const KERNELS: [(&str, &str, usize); 2] =
    [("tiled", GEMM_WGSL, 64), ("direct", GEMM_DIRECT_WGSL, 32)];

/// A buffer kept for reuse, grown when a call needs more.
struct Slot {
    buffer: wgpu::Buffer,
    size: u64,
}

/// The reusable buffers of one call shape: the two operands, the result,
/// the uniform and the staging buffer the result comes back through.
#[derive(Default)]
struct Buffers {
    a: Option<Slot>,
    b: Option<Slot>,
    c: Option<Slot>,
    back: Option<Slot>,
    groups: Option<Slot>,
    dims: Option<wgpu::Buffer>,
}

pub struct Accel {
    pub device: gpu::Device,
    gpu: wgpu::Device,
    queue: wgpu::Queue,
    kernels: Vec<Kernel>,
    /// Which of `kernels` products run with.
    kernel: std::sync::atomic::AtomicUsize,
    /// Free sets of buffers; a call takes one and gives it back, so calls
    /// from several threads never share one.
    free: Mutex<Vec<Buffers>>,
    /// Weight matrices kept between calls, by where they lie in memory.
    kept: Mutex<HashMap<(usize, usize), Arc<Kept>>>,
    kept_bytes: AtomicU64,
    budget: u64,
}

fn slot(gpu: &wgpu::Device, slot: &mut Option<Slot>, size: u64, usage: wgpu::BufferUsages) {
    if slot.as_ref().is_some_and(|s| s.size >= size) {
        return;
    }
    let size = size.next_power_of_two().max(256);
    *slot = Some(Slot {
        buffer: gpu.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage,
            mapped_at_creation: false,
        }),
        size,
    });
}

impl Accel {
    /// The first card of `class`, if there is one and it comes up.
    pub fn open(class: Class) -> Option<Accel> {
        let device = gpu::devices().into_iter().find(|d| d.class == class)?;
        Accel::on(device).ok()
    }

    fn on(device: gpu::Device) -> Result<Accel> {
        let instance = gpu::instance();
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
        let kernels = KERNELS
            .iter()
            .map(|&(name, source, tile)| {
                let shader = gpu.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(name),
                    source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
                });
                let pipeline = gpu.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(name),
                    layout: None,
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });
                let layout = pipeline.get_bind_group_layout(0);
                Kernel {
                    name,
                    pipeline,
                    layout,
                    tile,
                }
            })
            .collect();
        Ok(Accel {
            device,
            gpu,
            queue,
            kernels,
            kernel: std::sync::atomic::AtomicUsize::new(0),
            free: Mutex::new(Vec::new()),
            kept: Mutex::new(HashMap::new()),
            kept_bytes: AtomicU64::new(0),
            budget: resident_budget(),
        })
    }

    /// `b`'s copy on the card, written when the weights changed since,
    /// or `None` when the budget is spent.
    fn keep(&self, data: &[f32]) -> Option<Arc<Kept>> {
        let key = (data.as_ptr() as usize, data.len());
        let now = GENERATION.load(Ordering::Relaxed);
        let bytes = (data.len() * 4) as u64;
        let kept = {
            let mut map = self.kept.lock().unwrap_or_else(|e| e.into_inner());
            match map.get(&key) {
                Some(k) => Arc::clone(k),
                None => {
                    if self.kept_bytes.load(Ordering::Relaxed) + bytes > self.budget {
                        return None;
                    }
                    self.kept_bytes.fetch_add(bytes, Ordering::Relaxed);
                    let k = Arc::new(Kept {
                        buffer: self.gpu.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("kept weights"),
                            size: bytes.max(4),
                            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        }),
                        generation: Mutex::new(u64::MAX),
                    });
                    map.insert(key, Arc::clone(&k));
                    k
                }
            }
        };
        {
            let mut generation = kept.generation.lock().unwrap_or_else(|e| e.into_inner());
            if *generation != now {
                self.queue
                    .write_buffer(&kept.buffer, 0, bytemuck::cast_slice(data));
                *generation = now;
            }
        }
        Some(kept)
    }

    /// `c[m, n] (+)= a · b`, `a` and `b` read through their strides; `b`
    /// kept on the card between calls when `keep_b` — a weight matrix,
    /// which only changes when the optimizer runs.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(
        &self,
        c: &mut [f32],
        a: View<'_>,
        b: View<'_>,
        m: usize,
        n: usize,
        p: usize,
        add: bool,
        keep_b: bool,
    ) {
        if m == 0 || n == 0 {
            return;
        }
        let span =
            |v: &View<'_>, rows: usize, cols: usize| (rows - 1) * v.row + (cols - 1) * v.col + 1;
        let a = View {
            data: &a.data[..if p == 0 { 1 } else { span(&a, m, p) }],
            ..a
        };
        let b = View {
            data: &b.data[..if p == 0 { 1 } else { span(&b, p, n) }],
            ..b
        };
        let one = Part {
            a: 0,
            rows: m,
            b: 0,
            c: 0,
            depth: p,
        };
        self.grouped(&mut c[..m * n], a, b, n, &[one], add, keep_b);
    }

    /// Several products in one dispatch, all `n` columns wide with the same
    /// strides: product `i` reads `a` from `parts[i].a`, `b` from
    /// `parts[i].b`, sums `parts[i].depth` terms over `parts[i].rows` rows,
    /// and writes `c` from `parts[i].c`. The whole of `a`, `b` and `c` goes
    /// up and comes back.
    #[allow(clippy::too_many_arguments)]
    pub fn grouped(
        &self,
        c: &mut [f32],
        a: View<'_>,
        b: View<'_>,
        n: usize,
        parts: &[Part],
        add: bool,
        keep_b: bool,
    ) {
        self.grouped_with(c.len(), a, b, n, parts, keep_b, &mut |i, got| {
            let span = parts[i].c..parts[i].c + got.len();
            if add {
                for (o, v) in c[span].iter_mut().zip(got) {
                    *o += v;
                }
            } else {
                c[span].copy_from_slice(got);
            }
        });
    }

    /// [`Self::grouped`] handing each product's `rows × n` result to
    /// `deliver` instead of writing one `c`: the products' destinations
    /// need not be one slice.
    #[allow(clippy::too_many_arguments)]
    pub fn grouped_with(
        &self,
        c_len: usize,
        a: View<'_>,
        b: View<'_>,
        n: usize,
        parts: &[Part],
        keep_b: bool,
        deliver: &mut dyn FnMut(usize, &[f32]),
    ) {
        let tall = parts.iter().map(|g| g.rows).max().unwrap_or(0);
        if tall == 0 || n == 0 || c_len == 0 {
            return;
        }
        let mut set = self
            .free
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pop()
            .unwrap_or_default();
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let c_bytes = (c_len * 4) as u64;
        slot(&self.gpu, &mut set.a, (a.data.len() * 4) as u64, storage);
        slot(&self.gpu, &mut set.b, (b.data.len() * 4) as u64, storage);
        slot(
            &self.gpu,
            &mut set.groups,
            (parts.len() * 32) as u64,
            storage,
        );
        slot(
            &self.gpu,
            &mut set.c,
            c_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        slot(
            &self.gpu,
            &mut set.back,
            c_bytes,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        let dims = set.dims.get_or_insert_with(|| {
            self.gpu.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 32,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        let (sa, sb, sc, sback, sgroups) = (
            set.a.as_ref().expect("made above"),
            set.b.as_ref().expect("made above"),
            set.c.as_ref().expect("made above"),
            set.back.as_ref().expect("made above"),
            set.groups.as_ref().expect("made above"),
        );
        let meta: [u32; 8] = [
            n as u32,
            0,
            0,
            0,
            a.row as u32,
            a.col as u32,
            b.row as u32,
            b.col as u32,
        ];
        let table: Vec<u32> = parts
            .iter()
            .flat_map(|g| {
                [
                    g.a as u32,
                    g.rows as u32,
                    g.b as u32,
                    g.c as u32,
                    g.depth as u32,
                    0,
                    0,
                    0,
                ]
            })
            .collect();
        self.queue
            .write_buffer(dims, 0, bytemuck::cast_slice(&meta));
        self.queue
            .write_buffer(&sgroups.buffer, 0, bytemuck::cast_slice(&table));
        self.queue
            .write_buffer(&sa.buffer, 0, bytemuck::cast_slice(a.data));
        let kept = if keep_b { self.keep(b.data) } else { None };
        if kept.is_none() {
            self.queue
                .write_buffer(&sb.buffer, 0, bytemuck::cast_slice(b.data));
        }
        let b_buffer = kept.as_ref().map_or(&sb.buffer, |k| &k.buffer);
        let kernel = &self.kernels[self.kernel.load(Ordering::Relaxed)];
        let bind = self.gpu.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &kernel.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sa.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: sc.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dims.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: sgroups.buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .gpu
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&kernel.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(
                n.div_ceil(kernel.tile) as u32,
                tall.div_ceil(kernel.tile) as u32,
                parts.len() as u32,
            );
        }
        encoder.copy_buffer_to_buffer(&sc.buffer, 0, &sback.buffer, 0, c_bytes);
        self.queue.submit(Some(encoder.finish()));
        let slice = sback.buffer.slice(..c_bytes);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.gpu.poll(wgpu::PollType::wait_indefinitely());
        {
            let view = slice
                .get_mapped_range()
                .expect("the staging buffer was mapped above");
            let got: &[f32] = bytemuck::cast_slice(&view);
            // Only what the products wrote: the rest is not theirs.
            for (i, g) in parts.iter().enumerate() {
                deliver(i, &got[g.c..g.c + g.rows * n]);
            }
        }
        sback.buffer.unmap();
        self.free
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(set);
    }
}

/// One product of [`Accel::grouped`]: offsets in floats.
#[derive(Clone, Copy, Debug)]
pub struct Part {
    pub a: usize,
    pub rows: usize,
    pub b: usize,
    pub c: usize,
    pub depth: usize,
}

/// The card training's products go to, chosen once: the discrete card;
/// without one, the integrated GPU when it runs a product faster than the
/// processor does; or none. `ORANGU_GGUF_DEVICE=cpu` keeps everything on
/// the processor.
pub fn card() -> Option<&'static Accel> {
    static CARD: std::sync::OnceLock<Option<Accel>> = std::sync::OnceLock::new();
    CARD.get_or_init(|| {
        if std::env::var("ORANGU_GGUF_DEVICE").as_deref() == Ok("cpu") {
            return None;
        }
        if let Some(card) = Accel::open(Class::Discrete) {
            return Some(card);
        }
        let devices = gpu::devices();
        let integrated = devices.iter().find(|d| d.class == Class::Integrated);
        let accel = match integrated.map(|d| Accel::on(d.clone())) {
            Some(Ok(accel)) => Some(accel),
            Some(Err(e)) => {
                eprintln!("orangu-gguf: the integrated GPU does not come up: {e:#}");
                return None;
            }
            None => None,
        };
        let Some(accel) = accel else {
            let seen: Vec<String> = devices
                .iter()
                .map(|d| format!("{} ({:?})", d.name, d.class))
                .collect();
            eprintln!(
                "orangu-gguf: no discrete or integrated GPU comes up; adapters seen: {}",
                if seen.is_empty() {
                    "none".to_string()
                } else {
                    seen.join(", ")
                }
            );
            return None;
        };
        let probe = rates(&accel);
        let (best, gpu) = probe
            .kernels
            .iter()
            .copied()
            .enumerate()
            .fold((0, 0.0), |b, (i, r)| if r > b.1 { (i, r) } else { b });
        accel.kernel.store(best, Ordering::Relaxed);
        let cpu = probe.cpu;
        let used = gpu > cpu;
        let kernels: Vec<String> = accel
            .kernels
            .iter()
            .zip(&probe.kernels)
            .map(|(k, r)| format!("{r:.0} GFLOP/s {}", k.name))
            .collect();
        eprintln!(
            "orangu-gguf: the integrated GPU {} runs a {}x{}x{} product at {}, {:.2} ms a call \
             fixed; the processor at {cpu:.0}; {}",
            accel.device.name,
            PROBE.0,
            PROBE.1,
            PROBE.2,
            kernels.join(", "),
            probe.fixed_ms,
            if used {
                "products go to it"
            } else {
                "it is not used"
            }
        );
        used.then_some(accel)
    })
    .as_ref()
}

/// The product [`rates`] times: the size of training's larger ones.
const PROBE: (usize, usize, usize) = (1024, 768, 512);

/// What [`rates`] measures.
struct Probe {
    /// GFLOP/s of each of the card's kernels.
    kernels: Vec<f64>,
    /// GFLOP/s of the processor.
    cpu: f64,
    /// Milliseconds an 8x8x8 product takes on the card: the cost of a call.
    fixed_ms: f64,
}

/// GFLOP/s of each of the card's kernels and of the processor on [`PROBE`],
/// the operands sent to the card each time, after one untimed run of each;
/// and the time of a product too small to cost anything but the call.
fn rates(accel: &Accel) -> Probe {
    let (m, n, p) = PROBE;
    let a: Vec<f32> = (0..m * p).map(|i| (i as f32 * 0.37).sin()).collect();
    let b: Vec<f32> = (0..p * n).map(|i| (i as f32 * 0.91).sin()).collect();
    let (av, bv) = (
        View {
            data: &a,
            row: p,
            col: 1,
        },
        View {
            data: &b,
            row: n,
            col: 1,
        },
    );
    let mut c = vec![0f32; m * n];
    let flop = 2.0 * (m * n * p) as f64;
    let mut rate = |run: &mut dyn FnMut(&mut [f32])| {
        run(&mut c);
        let started = std::time::Instant::now();
        for _ in 0..3 {
            run(&mut c);
        }
        3.0 * flop / started.elapsed().as_secs_f64() / 1e9
    };
    let mut kernels = Vec::new();
    for i in 0..accel.kernels.len() {
        accel.kernel.store(i, Ordering::Relaxed);
        kernels.push(rate(&mut |c| accel.gemm(c, av, bv, m, n, p, false, false)));
    }
    let cpu = rate(&mut |c| crate::gemm::library(c, av, bv, m, n, p, false));
    let small = View {
        data: &a,
        row: 8,
        col: 1,
    };
    let mut tiny = [0f32; 64];
    accel.gemm(&mut tiny, small, small, 8, 8, 8, false, false);
    let started = std::time::Instant::now();
    for _ in 0..8 {
        accel.gemm(&mut tiny, small, small, 8, 8, 8, false, false);
    }
    let fixed_ms = started.elapsed().as_secs_f64() * 1e3 / 8.0;
    Probe {
        kernels,
        cpu,
        fixed_ms,
    }
}

/// Work one call divides between the card and the processor.
#[derive(Clone, Copy)]
pub enum Split {
    ExpertsForward,
    ExpertsBackward,
}

/// Rows a second each side has managed on each kind of divided work.
static RATES: Mutex<[(f64, f64); 2]> = Mutex::new([(0.0, 0.0); 2]);

/// The card's share of `split`'s rows: in proportion to the two sides'
/// measured rates, half until both have been measured, and never so small
/// that a side stops being measured.
pub fn card_share(split: Split) -> f64 {
    let (card, cpu) = RATES.lock().unwrap_or_else(|e| e.into_inner())[split as usize];
    if card <= 0.0 || cpu <= 0.0 {
        return 0.5;
    }
    (card / (card + cpu)).clamp(0.1, 0.9)
}

/// What one divided call measured on each side.
pub fn measured(split: Split, card: (usize, f64), cpu: (usize, f64)) {
    let mut rates = RATES.lock().unwrap_or_else(|e| e.into_inner());
    let slot = &mut rates[split as usize];
    for ((rows, seconds), rate) in [(card, &mut slot.0), (cpu, &mut slot.1)] {
        if rows > 0 && seconds > 0.0 {
            let now = rows as f64 / seconds;
            *rate = if *rate > 0.0 {
                0.7 * *rate + 0.3 * now
            } else {
                now
            };
        }
    }
}

/// Threads named so send every product to the card, or none.
#[cfg(test)]
pub const FORCE_CARD: &str = "accel-card";
#[cfg(test)]
pub const FORCE_CPU: &str = "accel-cpu";

/// Products of at least this many floating-point operations go to the
/// card; `ORANGU_GGUF_CARD_MIN_MFLOP` moves the line.
const CARD_MIN_MFLOP: u64 = 500;

/// Output sides narrower than this stay on the processor: a thin product
/// leaves most of a workgroup's tile empty.
const CARD_MIN_SIDE: usize = 128;

/// The card, when a `[m, n]` product over `p` terms is worth sending to it.
pub fn for_product(m: usize, n: usize, p: usize) -> Option<&'static Accel> {
    // A test runs a model in a pool of its own and names where its products
    // go, without touching the tests beside it.
    #[cfg(test)]
    if let Some(name) = std::thread::current().name() {
        if name.starts_with(FORCE_CARD) {
            return card();
        }
        if name.starts_with(FORCE_CPU) {
            return None;
        }
    }
    static MIN: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let min = *MIN.get_or_init(|| {
        std::env::var("ORANGU_GGUF_CARD_MIN_MFLOP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(CARD_MIN_MFLOP)
    });
    if m < CARD_MIN_SIDE || n < CARD_MIN_SIDE || (2 * m * n * p) as u64 / 1_000_000 < min {
        return None;
    }
    card()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemm_wgsl_is_valid() {
        for (_, source, _) in KERNELS {
            valid(source);
        }
    }

    fn valid(source: &str) {
        let module = naga::front::wgsl::parse_str(source).expect("valid WGSL");
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("validates");
        let mut out = String::new();
        let options = naga::back::msl::Options {
            lang_version: (2, 4),
            ..Default::default()
        };
        let pipeline = naga::back::msl::PipelineOptions::default();
        let mut writer = naga::back::msl::Writer::new(&mut out);
        writer
            .write(&module, &info, &options, &pipeline)
            .expect("Metal translation");
    }

    fn wave(n: usize, k: f32) -> Vec<f32> {
        (0..n).map(|i| (i as f32 * k).sin()).collect()
    }

    /// The card's product is the CPU's, to rounding, in both stride orders
    /// and across block edges.
    #[test]
    fn the_card_product_is_the_cpus() {
        let Some(accel) = Accel::open(Class::Discrete).or_else(|| Accel::open(Class::Integrated))
        else {
            println!("no card; skipping");
            return;
        };
        let shapes = [(1, 1, 1), (7, 70, 19), (130, 65, 300)];
        for (kernel, &(m, n, p)) in
            (0..accel.kernels.len()).flat_map(|k| shapes.iter().map(move |s| (k, s)))
        {
            accel.kernel.store(kernel, Ordering::Relaxed);
            let name = accel.kernels[kernel].name;
            let a = wave(m * p, 0.37);
            let b = wave(p * n, 0.91);
            for (av, bv) in [
                (
                    View {
                        data: &a,
                        row: p,
                        col: 1,
                    },
                    View {
                        data: &b,
                        row: n,
                        col: 1,
                    },
                ),
                (
                    View {
                        data: &a,
                        row: 1,
                        col: m,
                    },
                    View {
                        data: &b,
                        row: 1,
                        col: p,
                    },
                ),
            ] {
                let mut want = wave(m * n, 0.13);
                let mut got = want.clone();
                crate::gemm::library(&mut want, av, bv, m, n, p, true);
                accel.gemm(&mut got, av, bv, m, n, p, true, false);
                for (i, (x, y)) in want.iter().zip(&got).enumerate() {
                    assert!(
                        (x - y).abs() <= 1e-3 * x.abs().max(1.0),
                        "{name} {m}x{n}x{p} at {i}: {x} vs {y}"
                    );
                }
            }
        }
    }

    /// A kept weight is written again once the weights change, and many
    /// threads reaching a new one at once all read it written.
    #[test]
    fn a_kept_weight_follows_the_optimizer() {
        let Some(accel) = Accel::open(Class::Discrete).or_else(|| Accel::open(Class::Integrated))
        else {
            println!("no card; skipping");
            return;
        };
        let (m, n, p) = (70, 130, 40);
        let a = wave(m * p, 0.37);
        let mut w = wave(n * p, 0.91);
        let product = |w: &[f32], keep: bool| {
            let mut c = vec![0.0; m * n];
            let av = View {
                data: &a,
                row: p,
                col: 1,
            };
            let bv = View {
                data: w,
                row: 1,
                col: p,
            };
            accel.gemm(&mut c, av, bv, m, n, p, false, keep);
            c
        };
        weights_changed();
        let first: Vec<Vec<f32>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8).map(|_| scope.spawn(|| product(&w, true))).collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let plain = product(&w, false);
        for c in &first {
            assert_eq!(c, &plain);
        }
        for v in w.iter_mut() {
            *v *= -0.5;
        }
        weights_changed();
        assert_eq!(product(&w, true), product(&w, false));
    }

    /// The card against the CPU at the training shapes, transfers
    /// included. Run with
    /// `cargo test --release --bin orangu-gguf card_versus_cpu -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn card_versus_cpu() {
        for class in [Class::Discrete, Class::Integrated] {
            let Some(accel) = Accel::open(class) else {
                continue;
            };
            println!("{} ({class:?})", accel.device.name);
            for (name, t, k, n) in [
                ("expert gate/up", 256, 768, 512),
                ("shared gate/up", 1024, 768, 512),
                ("attn q joint  ", 1024, 768, 1536),
                ("delta qkv     ", 1024, 768, 2048),
                ("hc down       ", 1024, 1536, 64),
                ("output head   ", 1024, 768, 8192),
            ] {
                let x = wave(t * k, 0.37);
                let w = wave(n * k, 0.91);
                let dy = wave(t * n, 0.53);
                let flop = 2.0 * (t * k * n) as f64;
                let time = |f: &mut dyn FnMut()| {
                    f();
                    let start = std::time::Instant::now();
                    for _ in 0..10 {
                        f();
                    }
                    flop / (start.elapsed().as_secs_f64() / 10.0) / 1e9
                };
                let mut y = vec![0f32; t * n];
                let mut dx = vec![0f32; t * k];
                let mut dw = vec![0f32; n * k];
                let fwd = (
                    View {
                        data: &x,
                        row: k,
                        col: 1,
                    },
                    View {
                        data: &w,
                        row: 1,
                        col: k,
                    },
                );
                let bdx = (
                    View {
                        data: &dy,
                        row: n,
                        col: 1,
                    },
                    View {
                        data: &w,
                        row: k,
                        col: 1,
                    },
                );
                let bdw = (
                    View {
                        data: &dy,
                        row: 1,
                        col: n,
                    },
                    View {
                        data: &x,
                        row: k,
                        col: 1,
                    },
                );
                let card = [
                    time(&mut || accel.gemm(&mut y, fwd.0, fwd.1, t, n, k, false, true)),
                    time(&mut || accel.gemm(&mut dx, bdx.0, bdx.1, t, k, n, true, true)),
                    time(&mut || accel.gemm(&mut dw, bdw.0, bdw.1, n, k, t, true, false)),
                ];
                let cpu = [
                    time(&mut || crate::model::matmul(&mut y, &x, &w, t, k, n)),
                    time(&mut || crate::model::matmul_add_dx(&mut dx, &dy, &w, t, k, n)),
                    time(&mut || crate::model::matmul_add_dw(&mut dw, &dy, &x, t, k, n)),
                ];
                println!(
                    "  {name} card fwd {:>6.0} dx {:>6.0} dw {:>6.0}   cpu fwd {:>6.0} dx {:>6.0} dw {:>6.0} GFLOP/s",
                    card[0], card[1], card[2], cpu[0], cpu[1], cpu[2]
                );
            }
        }
    }
}
