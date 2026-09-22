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

use super::*;
use crate::engine::backend::CpuBackend;
use crate::engine::kv_cache::strided_dims;
use crate::engine::loader::test_quant_matrix;
use crate::engine::quant::{
    GGML_TYPE_BF16, GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_IQ1_M, GGML_TYPE_IQ1_S,
    GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ3_XXS,
    GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS, GGML_TYPE_MXFP4, GGML_TYPE_PQ2_0, GGML_TYPE_PTQ1_0,
    GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q4_K, GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0,
};

/// The RMSNorm width rule must reproduce every width it was measured at,
/// and stay inside the set of kernels that actually get compiled.
///
/// The three measured points are the whole justification for the rule
/// existing at all — a single constant cannot be right at both 960 and
/// 3072 — so a change that stopped reproducing them would have removed the
/// reason for the complexity while keeping the complexity.
#[test]
fn the_norm_width_rule_reproduces_every_measured_best() {
    // Measured: decode, 5 reps, two context depths, against a fixed 128.
    for (n_embd, want) in [(960usize, 128usize), (2048, 256), (3072, 256)] {
        assert_eq!(
            norm_wg_for(n_embd),
            want,
            "n_embd {n_embd} measured best at {want}"
        );
    }
    // Extrapolations of the mechanism, not of the data — but they still
    // have to name a kernel that exists, and stay monotone in width.
    let mut last = 0;
    for n_embd in [64usize, 256, 512, 1024, 4096, 16384] {
        let wg = norm_wg_for(n_embd);
        assert!(NORM_WGS.contains(&wg), "n_embd {n_embd} chose {wg}");
        assert!(wg >= last, "wider rows must not choose a narrower kernel");
        last = wg;
        assert!(norm_wg_index(n_embd) < NORM_WGS.len());
    }
}

/// An out-of-range tuning value must fall back to the default **and be
/// rejected out loud**.
///
/// The silent half is the one that costs something. A sweep sets
/// `ORANGU_NORM_WG=32`, the server runs 128, and the benchmark records a
/// second copy of the default under the name `32` — two identical
/// configurations reported as two distinct points, which reads as "this
/// knob does nothing down there" rather than "that value was never tried".
/// That happened, on the first sweep run through this code.
///
/// The variable is set and removed inside this one test rather than
/// through the real `norm_wg`/`reduce_n_rows`, which memoize in a
/// `OnceLock` and so can only be observed once per process.
#[test]
fn a_rejected_tuning_value_falls_back_and_says_so() {
    const VAR: &str = "ORANGU_TEST_TUNING_VALUE";
    let read = || {
        super::super::env_tuning_value(VAR, 128usize, "one of 64, 128, 256", |n| {
            matches!(n, 64 | 128 | 256)
        })
    };
    // SAFETY: this variable is named for this test and read by nothing
    // else; no other thread in the binary looks at it.
    unsafe { std::env::remove_var(VAR) };
    assert_eq!(read(), 128, "unset means the default");
    unsafe { std::env::set_var(VAR, "256") };
    assert_eq!(read(), 256, "an accepted value is used");
    unsafe { std::env::set_var(VAR, " 64 ") };
    assert_eq!(read(), 64, "surrounding whitespace is not a rejection");
    // Parses fine, is not a value the kernel has — the case a sweep hits.
    unsafe { std::env::set_var(VAR, "32") };
    assert_eq!(read(), 128, "an out-of-range value falls back");
    unsafe { std::env::set_var(VAR, "banana") };
    assert_eq!(read(), 128, "an unparseable value falls back");
    unsafe { std::env::remove_var(VAR) };
}

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
///
/// Not necessarily a *Vulkan* device: [`shared_test_backend`] asks for
/// whichever `wgpu` API the platform has, which is Metal on Apple. Every
/// cross-check below is written against `wgpu`, not against Vulkan, so
/// they are the Metal backend's correctness tests too — see
/// `engine::backend::metal`.
fn shared_vulkan() -> Option<&'static VulkanBackend> {
    // Delegates to the module-level single shared backend so this module's
    // tests and `vulkan_replay`'s share one `wgpu::Device` across the whole
    // test binary (one instance, never several concurrently).
    super::shared_test_backend()
}

/// What one GPU submit-and-wait round trip costs on this device.
///
/// The number that decides whether a decode step can be split at the FFN
/// boundary. Decode records all 42 layers into **one** submission today;
/// handing the FFN to the NPU means breaking the encoder at every layer, so
/// the split trades 41 extra round trips for the difference between the
/// GPU's 7.1 ms a layer and the NPU's 3.36. If a round trip costs much more
/// than 3.6 ms the trade is a loss before it starts.
///
/// Deliberately trivial work — a one-element copy — so what it measures is
/// the submit/fence/map path and not a kernel.
#[test]
#[ignore = "diagnostic; prints a measurement"]
fn _scratch_measure_submit_roundtrip() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    let src = vulkan.upload_for_test(&[1.0f32; 4]);
    let mut samples = Vec::new();
    for _ in 0..50 {
        let started = std::time::Instant::now();
        let encoder = vulkan.new_encoder("roundtrip probe");
        let out = vulkan.submit_and_readback_for_test(encoder, &src, 4);
        std::hint::black_box(&out);
        samples.push(started.elapsed().as_secs_f64());
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eprintln!(
        "  submit+readback round trip: median {:.3} ms, min {:.3} ms, max {:.3} ms",
        samples[samples.len() / 2] * 1000.0,
        samples[0] * 1000.0,
        samples[samples.len() - 1] * 1000.0,
    );
}

/// What streaming bandwidth this device actually delivers to a compute
/// shader — the number every "why is the GEMV only at N GB/s" question is
/// implicitly compared against, and which had been taken from the card's
/// spec sheet (224 GB/s) rather than measured.
///
/// A trivial kernel: one `vec4<f32>` load per thread over a large buffer,
/// summed so nothing is dead-code eliminated, nothing written back beyond a
/// single value. No dequantization, no shared memory, perfectly coalesced.
/// That is the ceiling this hardware offers, and no matmul can beat it.
#[test]
#[ignore]
fn _scratch_measure_streaming_bandwidth() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    // Same kernel at three load widths: 16 B (`vec4<f32>`), 8 B and 4 B.
    // The GEMV reads quantized blocks a dword at a time, so if the memory
    // system is request-rate-bound rather than byte-bound this is where it
    // shows.
    const SRC_TMPL: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<LOADT>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> n: vec4<u32>;

@compute @workgroup_size(WGSIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var acc: f32 = 0.0;
    var i: u32 = START;
    loop {
        if (i >= n.x) { break; }
        SUMEXPR
        i = i + n.y;
    }
    // Never true; keeps `acc` live without writing every thread.
    if (acc == 12345.678) { dst[gid.x] = acc; }
}
"#;
    let device = &vulkan.device;
    let bytes: u64 = 512 * 1024 * 1024;
    for (label, loadt, sumexpr, width, wg, scatter) in [
        (
            "vec4 coalesced, wg256",
            "vec4<f32>",
            "let v = src[i]; acc = acc + v.x + v.y + v.z + v.w;",
            16u64,
            256u32,
            false,
        ),
        (
            "vec4 coalesced, wg 64",
            "vec4<f32>",
            "let v = src[i]; acc = acc + v.x + v.y + v.z + v.w;",
            16,
            64,
            false,
        ),
        (
            "vec4 coalesced, wg 32",
            "vec4<f32>",
            "let v = src[i]; acc = acc + v.x + v.y + v.z + v.w;",
            16,
            32,
            false,
        ),
        (
            "f32  coalesced, wg 32",
            "f32",
            "acc = acc + src[i];",
            4,
            32,
            false,
        ),
        (
            "f32  scattered, wg 32",
            "f32",
            "acc = acc + src[i];",
            4,
            32,
            true,
        ),
        (
            "f32  scattered, wg256",
            "f32",
            "acc = acc + src[i];",
            4,
            256,
            true,
        ),
        // The decode GEMV's question, asked so the answer is unambiguous:
        // when four lanes of a wave request the *same* 16-byte line, does
        // the hardware merge them into one transaction or fetch it four
        // times? Both rows below cover the **same distinct bytes**; the
        // shared one needs four times as many loop steps to do it. If
        // merging works they take the same wall time; if not, the shared
        // one takes ~4x.
        (
            "distinct lines  (1 lane : 1 line)",
            "vec4<f32>",
            "let v = src[i]; acc = acc + v.x + v.y + v.z + v.w;",
            16,
            256,
            false,
        ),
        (
            "shared lines    (4 lanes : 1 line)",
            "vec4<f32>",
            "let v = src[i / 4u]; acc = acc + v.x + v.y + v.z + v.w;",
            16,
            256,
            false,
        ),
    ] {
        let wgsl = SRC_TMPL
            .replace("LOADT", loadt)
            .replace("SUMEXPR", sumexpr)
            .replace("WGSIZE", &wg.to_string());
        let wgsl = wgsl.replace("START", if scatter { "gid.x * 36u" } else { "gid.x" });
        // `shared` divides the index by 4, so a step advances the distinct
        // frontier by only a quarter as much; give it 4x the elements to
        // walk so both rows cover the same bytes.
        let shared = label.starts_with("shared");
        let wgsl = wgsl.as_str();
        let vec4s = (bytes / width) as u32 * if shared { 4 } else { 1 };
        let src = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bw src"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let dst = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bw dst"),
            size: 1024,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        // 64 workgroups/CU worth of threads, each grid-striding the buffer.
        let threads: u32 = wg * 22 * 8;
        let meta = [vec4s, threads, 0u32, 0u32];
        let meta_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bw meta"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        vulkan
            .queue
            .write_buffer(&meta_buf, 0, bytemuck::cast_slice(&meta));
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bw"),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bw"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bw"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: src.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dst.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: meta_buf.as_entire_binding(),
                },
            ],
        });
        let groups = threads / wg;
        let mut best = 0.0f64;
        for _round in 0..5 {
            let t0 = std::time::Instant::now();
            let mut enc = vulkan.new_encoder("bw");
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("bw"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            vulkan.queue.submit(Some(enc.finish()));
            device.poll(wgpu::PollType::wait_indefinitely()).ok();
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            let gbs = (bytes as f64) / (ms / 1000.0) / 1e9;
            if gbs > best {
                best = gbs;
            }
        }
        eprintln!("  {label}: {best:.0} GB/s");
    }
}

/// Compiles **one** WGSL file into **one** compute pipeline, on a device
/// that has nothing else on it — so `RADV_DEBUG=shaders` emits exactly one
/// disassembly block and it is unambiguously the kernel you asked for.
///
/// This is the ISA-archaeology tool. A full server run compiles ~58
/// pipelines and the driver labels none of them, so the disassembly can
/// only be attributed by guessing from LDS size or code length — which is
/// how a load-count comparison between two shapes of the same kernel went
/// unanswerable. `ORANGU_DUMP_SHADERS` already writes every generated
/// kernel's WGSL; this reads one back and compiles it alone.
///
/// ```sh
/// ORANGU_DUMP_SHADERS=/tmp/wgsl orangu-server <model>   # once, to get the files
/// ORANGU_ISA_WGSL=/tmp/wgsl/q4k_matmul_light.wgsl RADV_DEBUG=shaders,shaderstats \
///   cargo test --release --bin orangu-server -- --ignored isa_compile_one_shader \
///   --nocapture 2>&1 | tee /tmp/one.isa
/// ```
///
/// The pipeline takes `layout: None` so wgpu derives the bind group layout
/// from the shader itself — the dumped WGSL is self-contained, and nothing
/// here needs to match the real backend's layouts.
#[test]
#[ignore]
fn isa_compile_one_shader() {
    let Ok(path) = std::env::var("ORANGU_ISA_WGSL") else {
        eprintln!("set ORANGU_ISA_WGSL=<path to a .wgsl> — see this test's docs");
        return;
    };
    let src = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{path}: {e}");
            return;
        }
    };
    // Vulkan specifically, not `test_backends()`: this test exists to
    // read back RADV's ACO disassembly via `RADV_DEBUG`, which no other
    // driver produces. Hence its own skip message rather than
    // `NO_GPU_SKIP` — on a Mac the honest answer is "wrong driver", not
    // "no adapter".
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let Some(adapter) =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()
    else {
        eprintln!("skipping: no Vulkan adapter available in this environment");
        return;
    };
    // Ask for `SHADER_F16` when present so an `enable f16;` kernel compiles
    // too; everything else stays at the defaults.
    let features = adapter.features() & wgpu::Features::SHADER_F16;
    let (device, _queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("orangu-server isa isolate"),
        required_features: features,
        ..Default::default()
    }))
    .expect("request_device");
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("orangu-server isa isolate module"),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("orangu-server isa isolate pipeline"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    // Force the driver to have actually compiled it before the process can
    // exit, so the disassembly is emitted.
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    drop(pipeline);
    eprintln!("compiled {path} as the only pipeline on this device");
}

/// Scratch measurement — NOT a correctness test, deleted once the
/// number is recorded.
/// Duplicates `gpu_attention`'s exact body (same pipeline, same bind
/// group layout, same `n_head`-workgroup dispatch shape) but wraps its
/// one compute pass with GPU-timestamp `timestamp_writes` instead of
/// `None`, to measure the `attn_pipeline` dispatch's own GPU execution
/// time in isolation — via hardware timer, not CPU wall-clock, so
/// submission/poll overhead doesn't confound the number. Real
/// gemma4-E2B full-attention-layer shape (`n_head=8`, `n_head_kv=1`,
/// `head_dim=512`, confirmed via `orangu-server show`) and a
/// context length matching the range used elsewhere in this module's
/// scratch measurements.
#[test]
#[ignore]
fn _scratch_measure_attention_dispatch_cost() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };

    let n_head = 8;
    let n_head_kv = 1;
    let head_dim = 512;
    let kv_dim = n_head_kv * head_dim;
    let capacity = 64;
    let n_positions = 32;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut seed = 0xA77E17_u64;
    let mut kv_cache = crate::engine::kv_cache::KvCache::new_with_dims(capacity, &[kv_dim]);
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
    let cache = &mut kv_cache.layers[0];

    let cap = cache.capacity();
    let kv_refs = cache.sync_gpu(&vulkan.device, &vulkan.queue, n_head, vulkan.kv_storage);
    let q_buf = vulkan.upload_new(&q);
    let out_buf = vulkan.scratch_buffer(n_head * head_dim);
    let meta = AttnMeta {
        n_head: n_head as u32,
        n_head_kv: n_head_kv as u32,
        head_dim: head_dim as u32,
        window_start: window_start as u32,
        n_pos: (pos - window_start + 1) as u32,
        capacity: cap as u32,
        scale,
        start_pos: 0,
        n_query: 0,
        n_swa: 0,
        causal: 0,
        kv_page_base: 0,
        kv_page_tokens: 0,
        kv_ring_rows: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let meta_buf = vulkan.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scratch attention meta"),
        size: std::mem::size_of::<AttnMeta>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    vulkan
        .queue
        .write_buffer(&meta_buf, 0, bytemuck::bytes_of(&meta));
    let bind_group = vulkan.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scratch attention bind group"),
        layout: &vulkan.attn_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: q_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: BindSrc::Slice(&kv_refs.buffer, kv_refs.k_off, kv_refs.k_size).resource(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: BindSrc::Slice(&kv_refs.buffer, kv_refs.v_off, kv_refs.v_size).resource(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: kv_refs.probs.as_entire_binding(),
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

    let query_set = vulkan.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("scratch timestamps"),
        ty: wgpu::QueryType::Timestamp,
        count: 2,
    });
    let resolve_buf = vulkan.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scratch timestamp resolve"),
        size: 16,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback_buf = vulkan.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scratch timestamp readback"),
        size: 16,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Run several times, in separate submissions (matching how one
    // decode step's attention dispatch is one among many separate
    // GPU-side passes, not a tight synthetic loop), and report the
    // minimum — the same "min, not mean" instinct as a microbenchmark,
    // to reduce first-touch/driver-side noise across runs.
    let mut samples = Vec::new();
    for _ in 0..20 {
        let mut encoder = vulkan
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("scratch attention encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("scratch attention pass"),
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: &query_set,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                }),
            });
            pass.set_pipeline(&vulkan.attn_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(n_head as u32, 1, 1);
        }
        encoder.resolve_query_set(&query_set, 0..2, &resolve_buf, 0);
        encoder.copy_buffer_to_buffer(&resolve_buf, 0, &readback_buf, 0, 16);
        vulkan.queue.submit(Some(encoder.finish()));
        readback_buf
            .slice(..)
            .map_async(wgpu::MapMode::Read, |r| r.expect("map failed"));
        vulkan
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll failed");
        let data = readback_buf
            .slice(..)
            .get_mapped_range()
            .expect("readback buffer was not mapped after a successful map_async + poll");
        let ticks: Vec<u64> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        readback_buf.unmap();
        let ns_per_tick = vulkan.queue.get_timestamp_period() as f64;
        let ms = (ticks[1].saturating_sub(ticks[0])) as f64 * ns_per_tick / 1_000_000.0;
        samples.push(ms);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eprintln!(
        "orangu-server: [scratch] attn_pipeline dispatch (n_head={n_head}, n_head_kv={n_head_kv}, \
             head_dim={head_dim}, n_positions={n_positions}): min={:.4}ms median={:.4}ms max={:.4}ms samples={samples:?}",
        samples[0],
        samples[samples.len() / 2],
        samples[samples.len() - 1],
    );
}

/// Scratch measurement — NOT a correctness test, kept `#[ignore]`d as
/// reusable tuning infrastructure the same way the attention scratch
/// benchmark above was.
/// Isolates the FFN block's elementwise `gelu` + `mul` dispatch pair
/// (`record_fused_post_attention`'s "fused ffn pass" —
/// `gelu_pipeline` then `mul_pipeline`, each `ffn_len.div_ceil(64)`
/// workgroups) at E2B's real `ffn_len = 6144`
/// (`gemma4.feed_forward_length`, confirmed via `orangu-server show`)
/// — the next thing worth checking before writing a GEGLU-fusion
/// shader, exactly the way attention was measured before rewriting
/// it. Deliberately excludes
/// the gate/up matmuls that share the same compute pass in
/// production (`vulkan.rs:3031-3046`) — those are expected-expensive
/// GEMMs, not the "many small dispatches" mechanism this measurement
/// is auditing.
#[test]
#[ignore]
fn _scratch_measure_ffn_elementwise_dispatch_cost() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };

    let ffn_len = 6144usize;
    let mut seed = 0xF44E17_u64;
    let gate: Vec<f32> = (0..ffn_len)
        .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
        .collect();
    let up: Vec<f32> = (0..ffn_len)
        .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
        .collect();

    let gate_buf = vulkan.upload_new(&gate);
    let up_buf = vulkan.upload_new(&up);
    let gelu_out = vulkan.scratch_buffer(ffn_len);
    let mulled = vulkan.scratch_buffer(ffn_len);
    let meta = vulkan.elem_meta_buffer(ffn_len as u32, 0.0);
    let bg_gelu = vulkan.elem3_bind_group(&gate_buf, &gelu_out, &meta);
    let bg_mul = vulkan.elem4_bind_group(&gelu_out, &up_buf, &mulled, &meta);
    let ffn_wg = (ffn_len as u32).div_ceil(64);

    let query_set = vulkan.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("scratch timestamps"),
        ty: wgpu::QueryType::Timestamp,
        count: 2,
    });
    let resolve_buf = vulkan.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scratch timestamp resolve"),
        size: 16,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback_buf = vulkan.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scratch timestamp readback"),
        size: 16,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut samples = Vec::new();
    for _ in 0..20 {
        let mut encoder = vulkan
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("scratch ffn elementwise encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("scratch ffn elementwise pass"),
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: &query_set,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                }),
            });
            pass.set_pipeline(&vulkan.gelu_pipeline);
            pass.set_bind_group(0, &bg_gelu, &[]);
            pass.dispatch_workgroups(ffn_wg, 1, 1);
            pass.set_pipeline(&vulkan.mul_pipeline);
            pass.set_bind_group(0, &bg_mul, &[]);
            pass.dispatch_workgroups(ffn_wg, 1, 1);
        }
        encoder.resolve_query_set(&query_set, 0..2, &resolve_buf, 0);
        encoder.copy_buffer_to_buffer(&resolve_buf, 0, &readback_buf, 0, 16);
        vulkan.queue.submit(Some(encoder.finish()));
        readback_buf
            .slice(..)
            .map_async(wgpu::MapMode::Read, |r| r.expect("map failed"));
        vulkan
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll failed");
        let data = readback_buf
            .slice(..)
            .get_mapped_range()
            .expect("readback buffer was not mapped after a successful map_async + poll");
        let ticks: Vec<u64> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        readback_buf.unmap();
        let ns_per_tick = vulkan.queue.get_timestamp_period() as f64;
        let ms = (ticks[1].saturating_sub(ticks[0])) as f64 * ns_per_tick / 1_000_000.0;
        samples.push(ms);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eprintln!(
        "orangu-server: [scratch] gelu_pipeline+mul_pipeline dispatch pair (ffn_len={ffn_len}): \
             min={:.4}ms median={:.4}ms max={:.4}ms samples={samples:?}",
        samples[0],
        samples[samples.len() / 2],
        samples[samples.len() - 1],
    );
}

/// Isolated GPU time (min of 20 samples, same methodology as
/// [`Self::_scratch_measure_attention_dispatch_cost`]) of the split-k
/// attention pipeline pair (`attn_split_pipeline` +
/// `attn_split_reduce_pipeline`) at one `k_num`, E2B's real
/// full-attention-layer shape otherwise.
fn measure_split_k_dispatch_ms(vulkan: &VulkanBackend, k_num: u32) -> f64 {
    let n_head = 8usize;
    let n_head_kv = 1usize;
    let head_dim = 512usize;
    let kv_dim = n_head_kv * head_dim;
    let capacity = 64;
    let n_positions = 32;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut seed = 0x53717717_u64 ^ (k_num as u64);
    let mut kv_cache = crate::engine::kv_cache::KvCache::new_with_dims(capacity, &[kv_dim]);
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
    let cache = &mut kv_cache.layers[0];

    let kv_refs = cache.sync_gpu(&vulkan.device, &vulkan.queue, n_head, vulkan.kv_storage);
    let q_buf = vulkan.upload_new(&q);
    let out_buf = vulkan.scratch_buffer(n_head * head_dim);
    let partial_ml = vulkan.scratch_buffer(n_head * k_num as usize * 2);
    let partial_acc = vulkan.scratch_buffer(n_head * k_num as usize * head_dim);

    let split_meta = AttnSplitMeta {
        n_head: n_head as u32,
        n_head_kv: n_head_kv as u32,
        head_dim: head_dim as u32,
        window_start: window_start as u32,
        n_pos: (pos - window_start + 1) as u32,
        k_num,
        scale,
        kv_page_base: 0,
        kv_page_tokens: 0,
        kv_ring_rows: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let split_meta_buf = vulkan.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scratch attention split meta"),
        size: std::mem::size_of::<AttnSplitMeta>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    vulkan
        .queue
        .write_buffer(&split_meta_buf, 0, bytemuck::bytes_of(&split_meta));
    let split_bind_group = vulkan.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scratch attention split bind group"),
        layout: &vulkan.attn_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: q_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: BindSrc::Slice(&kv_refs.buffer, kv_refs.k_off, kv_refs.k_size).resource(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: BindSrc::Slice(&kv_refs.buffer, kv_refs.v_off, kv_refs.v_size).resource(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: partial_ml.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: partial_acc.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: split_meta_buf.as_entire_binding(),
            },
        ],
    });

    let reduce_meta = AttnReduceMeta {
        head_dim: head_dim as u32,
        k_num,
        _pad0: 0,
        _pad1: 0,
    };
    let reduce_meta_buf = vulkan.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scratch attention split reduce meta"),
        size: std::mem::size_of::<AttnReduceMeta>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    vulkan
        .queue
        .write_buffer(&reduce_meta_buf, 0, bytemuck::bytes_of(&reduce_meta));
    let reduce_bind_group =
        vulkan.elem4_bind_group(&partial_ml, &partial_acc, &out_buf, &reduce_meta_buf);

    let query_set = vulkan.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("scratch timestamps"),
        ty: wgpu::QueryType::Timestamp,
        count: 2,
    });
    let resolve_buf = vulkan.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scratch timestamp resolve"),
        size: 16,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback_buf = vulkan.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scratch timestamp readback"),
        size: 16,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut samples = Vec::new();
    for _ in 0..20 {
        let mut encoder = vulkan
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("scratch split-k encoder"),
            });
        let split_pipeline = vulkan.attn_split_pipeline_for(
            head_dim,
            1,
            crate::engine::backend::vulkan_shaders::KvPaging::Contiguous,
        );
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("scratch split-k pass"),
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: &query_set,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                }),
            });
            pass.set_pipeline(&split_pipeline);
            pass.set_bind_group(0, &split_bind_group, &[]);
            pass.dispatch_workgroups(n_head as u32, k_num, 1);
            pass.set_pipeline(&vulkan.attn_split_reduce_pipeline);
            pass.set_bind_group(0, &reduce_bind_group, &[]);
            pass.dispatch_workgroups(n_head as u32, 1, 1);
        }
        encoder.resolve_query_set(&query_set, 0..2, &resolve_buf, 0);
        encoder.copy_buffer_to_buffer(&resolve_buf, 0, &readback_buf, 0, 16);
        vulkan.queue.submit(Some(encoder.finish()));
        readback_buf
            .slice(..)
            .map_async(wgpu::MapMode::Read, |r| r.expect("map failed"));
        vulkan
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll failed");
        let data = readback_buf
            .slice(..)
            .get_mapped_range()
            .expect("readback buffer was not mapped after a successful map_async + poll");
        let ticks: Vec<u64> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        readback_buf.unmap();
        let ns_per_tick = vulkan.queue.get_timestamp_period() as f64;
        let ms = (ticks[1].saturating_sub(ticks[0])) as f64 * ns_per_tick / 1_000_000.0;
        samples.push(ms);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[0]
}

/// Sweeps `ATTN_SPLIT_K` candidates — a cheaper, lower-risk follow-up
/// than a new dispatch-count audit, since `ATTN_SPLIT_K` was picked
/// as `4` as "a starting point," explicitly unswept. NOT a
/// correctness test, kept `#[ignore]`d as reusable tuning
/// infrastructure.
#[test]
#[ignore]
fn _scratch_sweep_attn_split_k() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    for k_num in [1u32, 2, 4, 8, 16] {
        let ms = measure_split_k_dispatch_ms(vulkan, k_num);
        eprintln!("orangu-server: [scratch] split-k dispatch pair (k_num={k_num}): min={ms:.4}ms");
    }
}

/// Isolated GPU time (min of 20 samples, same methodology as every
/// other `_scratch_measure_*` here) of one `rmsnorm_pipeline`
/// dispatch at E2B's real `n_embd = 1536`, comparing three shader
/// variants: the default 6-round `workgroupBarrier` tree reduction, the
/// existing 64-wide `subgroupAdd` reduction (the one an earlier
/// same-session A/B measured as a real regression end-to-end), and a
/// new 32-wide `subgroupAdd` variant matching a common 32-lane
/// subgroup width, which (if the adapter's actual subgroup size is
/// 32) lets each workgroup fit in exactly one subgroup, skipping the
/// cross-subgroup merge the 64-wide variant always pays.
fn measure_rmsnorm_variant_ms(vulkan: &VulkanBackend, source: String) -> f64 {
    let n_embd = 1536usize;
    let mut seed = 0x2181717_u64;
    let x: Vec<f32> = (0..n_embd)
        .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
        .collect();
    let weight: Vec<f32> = (0..n_embd)
        .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
        .collect();

    let x_buf = vulkan.upload_new(&x);
    let weight_buf = vulkan.upload_new(&weight);
    let y_buf = vulkan.scratch_buffer(n_embd);
    let meta = vulkan.elem_meta_buffer(n_embd as u32, 1e-6);
    let bg = vulkan.elem4_bind_group(&x_buf, &weight_buf, &y_buf, &meta);

    // `VulkanBackend` only keeps the bind-group *layout* around after
    // `try_init` (every production pipeline sharing it was already
    // built); rebuild the matching pipeline layout locally rather than
    // adding a field solely for this scratch benchmark's own use.
    let pipeline_layout = vulkan
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scratch elem4 pipeline layout"),
            bind_group_layouts: &[Some(&vulkan.elem4_bind_group_layout)],
            immediate_size: 0,
        });
    let module = vulkan
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scratch rmsnorm variant shader"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let pipeline = vulkan
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("scratch rmsnorm variant pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    let query_set = vulkan.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("scratch timestamps"),
        ty: wgpu::QueryType::Timestamp,
        count: 2,
    });
    let resolve_buf = vulkan.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scratch timestamp resolve"),
        size: 16,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback_buf = vulkan.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scratch timestamp readback"),
        size: 16,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut samples = Vec::new();
    for _ in 0..20 {
        let mut encoder = vulkan
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("scratch rmsnorm variant encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("scratch rmsnorm variant pass"),
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: &query_set,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                }),
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.resolve_query_set(&query_set, 0..2, &resolve_buf, 0);
        encoder.copy_buffer_to_buffer(&resolve_buf, 0, &readback_buf, 0, 16);
        vulkan.queue.submit(Some(encoder.finish()));
        readback_buf
            .slice(..)
            .map_async(wgpu::MapMode::Read, |r| r.expect("map failed"));
        vulkan
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll failed");
        let data = readback_buf
            .slice(..)
            .get_mapped_range()
            .expect("readback buffer was not mapped after a successful map_async + poll");
        let ticks: Vec<u64> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        readback_buf.unmap();
        let ns_per_tick = vulkan.queue.get_timestamp_period() as f64;
        let ms = (ticks[1].saturating_sub(ticks[0])) as f64 * ns_per_tick / 1_000_000.0;
        samples.push(ms);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[0]
}

/// NOT a correctness test,
/// kept `#[ignore]`d as reusable tuning infrastructure like every
/// other `_scratch_*` benchmark here. Requires `wgpu::Features::
/// SUBGROUP`; skips (not fails) without it, same as every other
/// subgroup-gated path in this file.
#[test]
#[ignore]
fn _scratch_measure_rmsnorm_workgroup_size_and_subgroup() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    if !vulkan.device.features().contains(wgpu::Features::SUBGROUP) {
        eprintln!("skipping: adapter does not support wgpu::Features::SUBGROUP");
        return;
    }

    let variants: [(&str, String); 3] = [
        (
            "default (wg64, tree-reduce)",
            vulkan_shaders::shader_source_rmsnorm(false, 64),
        ),
        (
            "subgroup wg64 (existing, previously measured as a regression)",
            vulkan_shaders::shader_source_rmsnorm(true, 64),
        ),
        (
            "subgroup wg32 (new candidate)",
            vulkan_shaders::shader_source_rmsnorm_subgroup_wg(32),
        ),
    ];
    for (label, source) in variants {
        let ms = measure_rmsnorm_variant_ms(vulkan, source);
        eprintln!("orangu-server: [scratch] rmsnorm {label}: min={ms:.4}ms");
    }
}

/// The single-workgroup argmax reduction `record_argmax_sample` used
/// before the split-reduction fix — reconstructed here, not
/// reachable from production code anymore, purely so
/// `_scratch_measure_argmax_dispatch_cost` has a real "before" to
/// compare the fix against, the same before/after shape used
/// elsewhere in this module's split-k measurement (there via `git
/// stash`; here inline, since the old shader was simple enough to
/// keep as a literal instead of round-tripping through git).
const OLD_ARGMAX_SAMPLE_SHADER: &str = r#"
struct SampleMeta {
    n_vocab: u32,
    n_recent: u32,
    repeat_penalty: f32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> logits: array<f32>;
@group(0) @binding(1) var<storage, read> recent_tokens: array<u32>;
@group(0) @binding(2) var<storage, read_write> out_token: array<u32>;
@group(0) @binding(3) var<uniform> sample_meta: SampleMeta;

var<workgroup> best_val: array<f32, 64>;
var<workgroup> best_idx: array<u32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let local = lid.x;

    if (local == 0u) {
        var i: u32 = 0u;
        loop {
            if (i >= sample_meta.n_recent) {
                break;
            }
            let tok = recent_tokens[i];
            if (tok < sample_meta.n_vocab) {
                let v = logits[tok];
                if (v > 0.0) {
                    logits[tok] = v / sample_meta.repeat_penalty;
                } else {
                    logits[tok] = v * sample_meta.repeat_penalty;
                }
            }
            i = i + 1u;
        }
    }
    workgroupBarrier();

    var my_best_val: f32 = -3.4028235e38;
    var my_best_idx: u32 = 0u;
    var k: u32 = local;
    loop {
        if (k >= sample_meta.n_vocab) {
            break;
        }
        let v = logits[k];
        if (v > my_best_val) {
            my_best_val = v;
            my_best_idx = k;
        }
        k = k + 64u;
    }
    best_val[local] = my_best_val;
    best_idx[local] = my_best_idx;
    workgroupBarrier();

    var stride: u32 = 32u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (local < stride && best_val[local + stride] > best_val[local]) {
            best_val[local] = best_val[local + stride];
            best_idx[local] = best_idx[local + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (local == 0u) {
        out_token[0] = best_idx[0];
    }
}
"#;

/// Isolated GPU time (min of 20 samples) of the pre-item-9
/// single-workgroup argmax reduction, at real `n_vocab`.
fn measure_argmax_old_ms(vulkan: &VulkanBackend, n_vocab: usize) -> f64 {
    let mut seed = 0xA126A5_u64;
    let logits: Vec<f32> = (0..n_vocab)
        .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
        .collect();
    let logits_buf = vulkan.upload_new(&logits);
    let recent_buf = vulkan.upload_new_u32(&[0]);
    let out_buf = vulkan.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scratch argmax old output"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let meta = SampleMeta {
        n_vocab: n_vocab as u32,
        n_recent: 0,
        repeat_penalty: 1.0,
        logit_softcap: 0.0,
    };
    let meta_buf = vulkan.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scratch argmax old meta"),
        size: std::mem::size_of::<SampleMeta>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    vulkan
        .queue
        .write_buffer(&meta_buf, 0, bytemuck::bytes_of(&meta));
    let bind_group = vulkan.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scratch argmax old bind group"),
        layout: &vulkan.argmax_bind_group_layout,
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

    let pipeline_layout = vulkan
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scratch argmax old pipeline layout"),
            bind_group_layouts: &[Some(&vulkan.argmax_bind_group_layout)],
            immediate_size: 0,
        });
    let module = vulkan
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scratch argmax old shader"),
            source: wgpu::ShaderSource::Wgsl(OLD_ARGMAX_SAMPLE_SHADER.into()),
        });
    let pipeline = vulkan
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("scratch argmax old pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    measure_one_pass_ms(vulkan, |pass| {
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    })
}

/// Isolated GPU time (min of 20 samples) of the fixed, three-
/// dispatch split argmax reduction (the exact same pipelines/bind
/// groups `record_argmax_sample` builds), at real `n_vocab`.
fn measure_argmax_new_ms(vulkan: &VulkanBackend, n_vocab: usize) -> f64 {
    let mut seed = 0xA126A5_u64;
    let logits: Vec<f32> = (0..n_vocab)
        .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
        .collect();
    let logits_buf = vulkan.upload_new(&logits);
    let recent_buf = vulkan.upload_new_u32(&[0]);
    let out_buf = vulkan.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scratch argmax new output"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let sample_meta = SampleMeta {
        n_vocab: n_vocab as u32,
        n_recent: 0,
        repeat_penalty: 1.0,
        logit_softcap: 0.0,
    };
    let sample_meta_buf = vulkan.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scratch argmax new sample meta"),
        size: std::mem::size_of::<SampleMeta>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    vulkan
        .queue
        .write_buffer(&sample_meta_buf, 0, bytemuck::bytes_of(&sample_meta));
    let penalty_bind_group = vulkan.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scratch argmax new penalty bind group"),
        layout: &vulkan.argmax_bind_group_layout,
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
                resource: sample_meta_buf.as_entire_binding(),
            },
        ],
    });

    let n_split = ARGMAX_SPLIT_N;
    let partial_val = vulkan.scratch_buffer(n_split as usize);
    let partial_idx = vulkan.scratch_buffer(n_split as usize);
    let split_meta = ArgmaxSplitMeta {
        n_vocab: n_vocab as u32,
        n_split,
        _pad0: 0,
        _pad1: 0,
    };
    let split_meta_buf = vulkan.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scratch argmax new split meta"),
        size: std::mem::size_of::<ArgmaxSplitMeta>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    vulkan
        .queue
        .write_buffer(&split_meta_buf, 0, bytemuck::bytes_of(&split_meta));
    let split_bind_group = vulkan.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scratch argmax new split bind group"),
        layout: &vulkan.argmax_split_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: logits_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: partial_val.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: partial_idx.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: split_meta_buf.as_entire_binding(),
            },
        ],
    });
    let reduce_meta_buf = vulkan.elem_meta_buffer(n_split, 0.0);
    let reduce_bind_group =
        vulkan.elem4_bind_group(&partial_val, &partial_idx, &out_buf, &reduce_meta_buf);

    measure_one_pass_ms(vulkan, |pass| {
        pass.set_pipeline(&vulkan.argmax_penalty_pipeline);
        pass.set_bind_group(0, &penalty_bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        pass.set_pipeline(&vulkan.argmax_split_pipeline);
        pass.set_bind_group(0, &split_bind_group, &[]);
        pass.dispatch_workgroups(n_split, 1, 1);
        pass.set_pipeline(&vulkan.argmax_reduce_pipeline);
        pass.set_bind_group(0, &reduce_bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    })
}

/// Shared min-of-20-samples GPU-timestamp harness — `record` sets up
/// pipeline/bind-group/dispatch calls inside one timestamped compute
/// pass; everything around it (query set, resolve/readback buffers,
/// submission loop) is the same boilerplate every `_scratch_measure_*`
/// benchmark in this file already repeats.
fn measure_one_pass_ms(vulkan: &VulkanBackend, record: impl Fn(&mut wgpu::ComputePass<'_>)) -> f64 {
    let query_set = vulkan.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("scratch timestamps"),
        ty: wgpu::QueryType::Timestamp,
        count: 2,
    });
    let resolve_buf = vulkan.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scratch timestamp resolve"),
        size: 16,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback_buf = vulkan.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scratch timestamp readback"),
        size: 16,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut samples = Vec::new();
    for _ in 0..20 {
        let mut encoder = vulkan
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("scratch measure_one_pass encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("scratch measure_one_pass pass"),
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: &query_set,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                }),
            });
            record(&mut pass);
        }
        encoder.resolve_query_set(&query_set, 0..2, &resolve_buf, 0);
        encoder.copy_buffer_to_buffer(&resolve_buf, 0, &readback_buf, 0, 16);
        vulkan.queue.submit(Some(encoder.finish()));
        readback_buf
            .slice(..)
            .map_async(wgpu::MapMode::Read, |r| r.expect("map failed"));
        vulkan
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll failed");
        let data = readback_buf
            .slice(..)
            .get_mapped_range()
            .expect("readback buffer was not mapped after a successful map_async + poll");
        let ticks: Vec<u64> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        readback_buf.unmap();
        let ns_per_tick = vulkan.queue.get_timestamp_period() as f64;
        let ms = (ticks[1].saturating_sub(ticks[0])) as f64 * ns_per_tick / 1_000_000.0;
        samples.push(ms);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[0]
}

/// NOT a correctness test,
/// kept `#[ignore]`d as reusable tuning infrastructure like every
/// other `_scratch_*` benchmark here. E2B's real `n_vocab = 262144`.
#[test]
#[ignore]
fn _scratch_measure_argmax_dispatch_cost() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    let n_vocab = 262144usize;
    let old_ms = measure_argmax_old_ms(vulkan, n_vocab);
    let new_ms = measure_argmax_new_ms(vulkan, n_vocab);
    eprintln!(
        "orangu-server: [scratch] argmax dispatch (n_vocab={n_vocab}): \
             old (single workgroup)={old_ms:.4}ms new (split, ARGMAX_SPLIT_N={ARGMAX_SPLIT_N})={new_ms:.4}ms"
    );
}

// The weight fixture is `engine::backend::probe_blocks` — shared with
// the other backends' cross-checks. See that module's doc comment.
use crate::engine::backend::probe_blocks::{build_block, next_byte};

/// `IQ4_NL` is in the 32 arm, not the 256 default: it is the one `IQ*`
/// type that blocks at 32, so the otherwise-safe "`IQ*` means `QK_K`"
/// reading would build every fixture row at 8× the right length here.
fn block_elems(ggml_type: u32) -> usize {
    match ggml_type {
        t if t == GGML_TYPE_F32 || t == GGML_TYPE_F16 || t == GGML_TYPE_BF16 => 1,
        t if t == GGML_TYPE_Q4_0
            || t == GGML_TYPE_Q4_1
            || t == GGML_TYPE_Q5_0
            || t == GGML_TYPE_Q5_1
            || t == GGML_TYPE_Q8_0
            || t == GGML_TYPE_IQ4_NL
            || t == GGML_TYPE_MXFP4 =>
        {
            32
        }
        t if t == GGML_TYPE_PQ2_0 || t == GGML_TYPE_PTQ1_0 => 128,
        _ => 256,
    }
}

/// Wall-clock of one `matmul` call (min of `samples` runs, after a
/// warm-up, the same min-of-N methodology as the other scratch
/// measurements here) plus the arithmetic rate it implies. `matmul`
/// blocks on its own `poll(wait_indefinitely())`, so this is that
/// submission's GPU time.
fn measure_matmul_gflops(
    vulkan: &VulkanBackend,
    ggml_type: u32,
    in_dim: usize,
    out_dim: usize,
    n_tokens: usize,
    samples: usize,
) -> (f64, f64) {
    let elems = block_elems(ggml_type);
    let mut seed = 0xB0BB1E_u64;
    let mut bytes = Vec::new();
    for _ in 0..out_dim {
        for _ in 0..(in_dim / elems) {
            bytes.extend(build_block(ggml_type, &mut seed));
        }
    }
    let w = test_quant_matrix(&bytes, ggml_type, in_dim, out_dim);
    let x: Vec<f32> = (0..n_tokens * in_dim)
        .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
        .collect();

    let _ = vulkan.matmul(&x, n_tokens, &w);
    let mut best = f64::MAX;
    for _ in 0..samples {
        let start = std::time::Instant::now();
        let _ = vulkan.matmul(&x, n_tokens, &w);
        best = best.min(start.elapsed().as_secs_f64());
    }
    let flops = 2.0 * (n_tokens * in_dim * out_dim) as f64;
    (best * 1000.0, flops / best / 1e9)
}

/// Scratch measurement — NOT a correctness test, kept `#[ignore]`d as
/// reusable tuning infrastructure like the other `_scratch_*` entries
/// here. Times the two GEMMs that dominate prefill at E2B's real
/// shapes (`n_embd = 1536`, `feed_forward_length = 6144`, fused
/// gate+up so `out_dim = 12288`) at the prefill submission's token
/// chunk (`MAX_MATMUL_TOKENS_PER_SUBMISSION`), which is what a whole
/// prompt's cost is built out of.
#[test]
#[ignore]
fn _scratch_measure_prefill_gemm() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    // Same output size (so upload/readback/submission cost is identical)
    // at halved and quartered K: if the time tracks K, the call is
    // compute-bound and the kernel is what matters; if it barely moves,
    // the per-call data movement dominates and the kernel is not the
    // thing to tune.
    for k in if std::env::var("ORANGU_SCRATCH_BONSAI").is_ok() {
        vec![]
    } else {
        vec![1536usize, 768, 384]
    } {
        let (ms, gflops) = measure_matmul_gflops(vulkan, GGML_TYPE_Q4_K, k, 12288, 128, 10);
        eprintln!(
            "orangu-server: [scratch] k-sweep in_dim={k} out_dim=12288 n_tokens=128: \
                 min={ms:.2}ms ({gflops:.1} GFLOP/s)"
        );
    }
    // `ORANGU_SCRATCH_BONSAI=1`: the 27B ternary model's FFN shapes and
    // types instead, through `matmul` — whichever GEMM the batch path
    // picks (the integer-dot one where the type has it).
    let bonsai = std::env::var("ORANGU_SCRATCH_BONSAI").is_ok();
    let shapes: Vec<(&str, usize, usize)> = if bonsai {
        vec![("gate", 5120, 17408), ("down", 17408, 5120)]
    } else {
        vec![
            ("gate_up", 1536, 12288),
            ("ffn_down", 6144, 1536),
            ("qkv", 1536, 1024),
        ]
    };
    let types: Vec<(&str, u32)> = if bonsai {
        vec![
            ("ptq1_0", GGML_TYPE_PTQ1_0),
            ("pq2_0", GGML_TYPE_PQ2_0),
            ("q4_k", GGML_TYPE_Q4_K),
            ("q8_0", GGML_TYPE_Q8_0),
        ]
    } else {
        vec![("q4_k", GGML_TYPE_Q4_K), ("f16", GGML_TYPE_F16)]
    };
    for (label, in_dim, out_dim) in shapes {
        for n_tokens in [64usize, 128] {
            for &(type_label, ggml_type) in &types {
                let (ms, gflops) =
                    measure_matmul_gflops(vulkan, ggml_type, in_dim, out_dim, n_tokens, 10);
                eprintln!(
                    "orangu-server: [scratch] {label} {type_label} {in_dim}x{out_dim} \
                         n_tokens={n_tokens}: min={ms:.2}ms ({gflops:.1} GFLOP/s)"
                );
            }
        }
    }
}

/// Cross-checks `VulkanBackend::matmul` against
/// `CpuBackend::matmul_dequant` (already known-correct, see
/// `engine::quant`'s own unit tests) for `ggml_type`, over
/// random-but-valid quantized data and random activations — the only
/// real way to verify the WGSL dequant/dot translation is bit-for-bit
/// faithful to its Rust counterpart, short of reading GPU assembly.
/// Skips (rather than fails) when no Vulkan adapter is available, e.g.
/// in a CI container with no GPU.
///
/// `matmul_dequant`, not `matmul`: the latter now prefers the fused
/// `int8`-activation path for `Q8_0`/`Q5_0`/`Q4_K`/`Q6_K`, whose
/// quantization loss on this test's adversarial random-uniform data is
/// several % — far above the tolerance below, and not something the GPU
/// kernel (which keeps activations in `f32`) should be reproducing.
/// Runs `wgsl` (one workgroup of `threads`, entry point `main`, one
/// `read_write` storage buffer at `@binding(0)`) and reads back
/// `out_len` floats. A minimal harness for kernels that test a single
/// language/hardware behaviour rather than any of orangu's own math —
/// nothing here touches the backend's layouts or caches.
fn run_probe_kernel(vulkan: &VulkanBackend, wgsl: &str, threads: u32, out_len: usize) -> Vec<f32> {
    let device = &vulkan.device;
    let bytes = (out_len * 4) as u64;
    let out = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe out"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe readback"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("probe"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("probe"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("probe"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: out.as_entire_binding(),
        }],
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("probe"),
    });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("probe"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    enc.copy_buffer_to_buffer(&out, 0, &read, 0, bytes);
    vulkan.queue.submit(Some(enc.finish()));
    let slice = read.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let data = slice.get_mapped_range().expect("map probe readback");
    let values: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    read.unmap();
    let _ = threads;
    values
}

/// **The invariant `vulkan_shaders::coop_vec4_tiles` has to satisfy**, and
/// the language-level probe it rests on.
///
/// Many threads each write one *component* of a shared `vec4<f32>`,
/// barrier, then read the completed `vec4`s back — exactly what
/// `store_w`/`store_x` do to fill `tile_w`/`tile_x` in the tiled prefill
/// GEMM, and the one pattern that kernel uses which no other kernel does.
/// If that does not land as a 4-byte store, four threads read-modify-write
/// the same 16 bytes and three of every four values are lost.
///
/// This exists because the tiled GEMM was the *only* thing that broke when
/// this WGSL was first run through Metal: every scalar-shared-memory
/// kernel (the whole reduce/GEMV family, every quant type) agreed with the
/// CPU backend, while every test routing through the tiled path disagreed
/// or produced `NaN`, including plain `f32` with no dequant in play. On
/// CI's Apple Paravirtual device this probe returned
/// `[1, 0, 0, 0, 5, 0, 0, 0, …]` — one surviving component per vector,
/// precisely the predicted clobber — while RADV returns all 64 values.
///
/// Asserted **one-directionally**: a backend using `vec4` tiles must pass
/// the probe. The converse is deliberately not required, so a driver that
/// later gains component-granular stores does not fail this test merely
/// for still being on the (correct, slightly slower) scalar form until
/// someone measures the switch.
#[test]
fn coop_tiles_are_vec4_only_where_component_stores_work() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    // 64 threads, 64 f32 slots = 16 shared vec4s. Thread `t` writes slot
    // `t` (vec4 `t >> 2`, component `t & 3`) — so four different threads
    // write four components of each vec4, which is the racy case if the
    // store is not component-granular.
    const SRC: &str = r#"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;

var<workgroup> tile: array<vec4<f32>, 16>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    tile[t >> 2u][t & 3u] = f32(t) + 1.0;
    workgroupBarrier();
    // Read back whole vec4s, the way the tiled GEMM's register block does.
    if (t < 16u) {
        let v = tile[t];
        out[t * 4u + 0u] = v.x;
        out[t * 4u + 1u] = v.y;
        out[t * 4u + 2u] = v.z;
        out[t * 4u + 3u] = v.w;
    }
}
"#;
    let got = run_probe_kernel(vulkan, SRC, 64, 64);
    let want: Vec<f32> = (0..64).map(|i| i as f32 + 1.0).collect();
    let component_stores_work = got == want;
    // Only `tile_w` is asserted against this probe. `tile_x`'s fill was
    // rearranged so every thread writes a whole vector (`store_x4`), which
    // this probe says nothing about — its control twin,
    // `shared_vec4_whole_stores_survive_a_barrier`, is the one that covers it,
    // and `tile_x_vec4_fill_writes_whole_vectors` is what holds the
    // generated fill to actually being that shape.
    if vulkan_shaders::coop_vec4_tiles(vulkan.wgpu_backend()).w {
        assert!(
            component_stores_work,
            "{} builds the tiled GEMM's weight tile as vec4, but \
                 component-wise stores into shared vec4 memory do not survive a \
                 barrier there — every tiled-path result on this device is \
                 wrong. Got {got:?}",
            vulkan.adapter_name
        );
    } else if component_stores_work {
        // Not a failure — the scalar form is always correct — but worth
        // saying, since it is the one measurement that would justify
        // turning `vec4` weight tiles on for this backend.
        eprintln!(
            "note: {} passes the component-store probe but is on a scalar \
                 weight tile; vec4 tile_w may be worth measuring here",
            vulkan.adapter_name
        );
    }
}

/// [`VulkanBackend::tuning_report`] must name a kernel for every
/// [`SUPPORTED_TYPES`] entry at both probe shapes, and must agree with
/// [`VulkanBackend::pipeline_for`] about which one.
///
/// The agreement half is what gives the report its value. A report that
/// merely *described* the selection would be a second copy of
/// `pipeline_for`'s branch ladder, and a stale copy is worse than no
/// report at all: it would send a reader hunting for a regression in a
/// kernel that never ran. `pipeline_for` delegating to
/// `pipeline_for_named` makes them the same code, and this holds them to
/// it by pointer identity — the named pipeline must be the *same object*
/// the dispatch path would have bound.
///
/// The completeness half catches the other failure: a type whose
/// selection falls through every branch to a map that has no entry for it
/// would panic inside `matmul` on the first request. Building the report
/// walks every type at startup, so this test finds that gap on a device
/// rather than a user finding it mid-generation.
#[test]
fn tuning_report_names_the_kernel_the_dispatch_would_use() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    let report = vulkan.tuning_report();
    eprintln!(
        "features: {}\nflags: {}",
        report["features"], report["flags"]
    );
    for (shape, n_tokens) in [("decode", 1), ("prefill", vulkan.coop_min_n_tokens)] {
        let named = report["kernels"][shape]
            .as_object()
            .unwrap_or_else(|| panic!("tuning_report has no {shape} kernel map"));
        assert_eq!(
            named.len(),
            SUPPORTED_TYPES.len(),
            "{shape} kernel map covers {} of {} supported types",
            named.len(),
            SUPPORTED_TYPES.len()
        );
        for &ty in SUPPORTED_TYPES {
            let name = orangu::gguf::ggml_type_name(ty);
            let reported = named[&name].as_str().expect("kernel name is a string");
            let (dispatched, from_selector) = vulkan.pipeline_for_named(ty, 4096, n_tokens);
            assert_eq!(
                reported, from_selector,
                "{name} {shape}: report and selector disagree"
            );
            assert!(
                std::ptr::eq(dispatched, vulkan.pipeline_for(ty, 4096, n_tokens)),
                "{name} {shape}: pipeline_for_named named {reported} but \
                     pipeline_for would bind a different pipeline"
            );
        }
    }
    // The banner line is built from the same selector, so it can't
    // disagree either — but it is what a reader actually sees, so prove
    // it is populated rather than an empty format string.
    let summary = vulkan.tuning_summary_for(&[GGML_TYPE_Q4_K, GGML_TYPE_Q6_K]);
    assert!(
        summary.contains("q4_k") && summary.contains("kv "),
        "tuning_summary_for is not the banner line it claims to be: {summary}"
    );
    // A model carrying none of the types this backend has a pipeline for
    // still has to produce a line rather than a panic or an empty
    // prefix — that is a CPU-only file whose banner still reports the
    // GPU's other settings.
    let none = vulkan.tuning_summary_for(&[]);
    assert!(
        none.starts_with("none · kv "),
        "an empty type list should still yield a banner: {none}"
    );
}

/// The control for [`coop_tiles_are_vec4_only_where_component_stores_work`]:
/// the same shared tile filled by *whole*-`vec4` stores, one thread per
/// vector. It passes on Metal, where the component version does not —
/// which is what pins the fault to the component store specifically
/// rather than to shared `vec4` memory, the barrier, or the readback.
///
/// It also rules out the tempting cheaper fix. Whole-`vec4` stores work
/// everywhere, but the tiled kernel cannot use them without changing its
/// tile layout: `store_w`'s four consecutive slots are four consecutive
/// *rows* at one `k`, while the fill deliberately gives one thread `RUN`
/// consecutive `k` of one row so a quantized block's scale/min hoist once
/// per run. Hence the scalar tile rather than a restructured fill.
#[test]
fn shared_vec4_whole_stores_survive_a_barrier() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    const SRC: &str = r#"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;

var<workgroup> tile: array<vec4<f32>, 16>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    if (t < 16u) {
        let b = f32(t * 4u);
        tile[t] = vec4<f32>(b + 1.0, b + 2.0, b + 3.0, b + 4.0);
    }
    workgroupBarrier();
    if (t < 16u) {
        let v = tile[t];
        out[t * 4u + 0u] = v.x;
        out[t * 4u + 1u] = v.y;
        out[t * 4u + 2u] = v.z;
        out[t * 4u + 3u] = v.w;
    }
}
"#;
    let got = run_probe_kernel(vulkan, SRC, 64, 64);
    let want: Vec<f32> = (0..64).map(|i| i as f32 + 1.0).collect();
    assert_eq!(got, want, "whole-vec4 shared stores are broken too");
}

/// The GELU kernels must stay finite for large-but-finite input.
///
/// `GELU_SHADER_BODY`'s cubic reaches `tanh(3.6e7)` at `|v| = 1000`.
/// Where `tanh` is a saturating hardware instruction that is harmless;
/// where it is lowered to `(exp(2x) - 1) / (exp(2x) + 1)` — `wgpu`'s
/// Metal backend — `exp` overflows and the result is `NaN`. That is what
/// made every GELU-path fused cross-check fail on Metal while every
/// SwiGLU one passed, and it is why the shader clamps the argument.
///
/// Checked against the CPU `gelu` this is a port of, so the test says
/// "the two implementations still agree at the extremes" rather than
/// merely "not NaN" — a clamp that was too tight would fail here too.
#[test]
fn gelu_kernel_stays_finite_at_large_inputs() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    // Spans the range a badly-scaled activation can reach, both signs,
    // well past where the cubic overflows a naive `tanh`.
    const SRC: &str = r#"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    // -1e5 .. 1e5, dense near zero and reaching far past saturation.
    let v = (f32(t) - 32.0) * 3125.0;
    let sqrt_2_over_pi = 0.7978846;
    let coef_a = 0.044715;
    out[t] = 0.5 * v * (1.0 + tanh(clamp(sqrt_2_over_pi * v * (1.0 + coef_a * v * v), -20.0, 20.0)));
}
"#;
    let got = run_probe_kernel(vulkan, SRC, 64, 64);
    for (t, g) in got.iter().enumerate() {
        let v = (t as f32 - 32.0) * 3125.0;
        let want = crate::engine::tensor::gelu(v);
        assert!(
            g.is_finite(),
            "gelu({v}) came back {g} on {} — the tanh argument overflowed",
            vulkan.adapter_name
        );
        let tol = 1e-3 * want.abs().max(1.0);
        assert!(
            (g - want).abs() <= tol,
            "gelu({v}): gpu={g} cpu={want} on {}",
            vulkan.adapter_name
        );
    }
}

fn cross_check(ggml_type: u32, in_dim: usize, out_dim: usize) {
    cross_check_n_tokens(ggml_type, in_dim, out_dim, 3);
}

/// Like `cross_check`, but with an explicit `n_tokens` — used with a
/// value `>= COOP_MIN_N_TOKENS` to exercise the workgroup-cooperative
/// dispatch path (`VulkanBackend::pipeline_for`/`vulkan_shaders::
/// shader_source_coop`), which `cross_check`'s fixed `n_tokens = 3`
/// never reaches.
fn cross_check_n_tokens(ggml_type: u32, in_dim: usize, out_dim: usize, n_tokens: usize) {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };

    let elems = block_elems(ggml_type);
    assert!(
        in_dim.is_multiple_of(elems),
        "in_dim must be a multiple of {elems}"
    );
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
        // Not a plain `byte / 64`: those are dyadic, and an element that is
        // exactly half its block's absmax quantizes to an exact `.5` tie
        // that the device and the CPU reference may round apart by a level
        // (Mali-G720: `-63.5` to `-64`, the CPU's `-63.499996` to `-63`) —
        // one level of a large `Q2_K` weight was 35% of a small output. A
        // second byte's worth of jitter, far below the quantization step,
        // keeps the values off the ties.
        let jitter = next_byte(&mut seed) as f32 / 256.0 * 1e-3;
        *v = (b as f32 - 128.0) / 64.0 + jitter;
    }

    if vulkan.ternary_idot_for(&w, n_tokens) {
        // Off the `1/64` grid: on it, an element at exactly half the block
        // maximum quantizes to the tie `±63.5`, and whether the device's
        // reciprocal lands a hair above or below decides the rounding — a
        // difference of one activation LSB that says nothing about the
        // kernel. Real activations are never on a grid.
        for (i, v) in x.iter_mut().enumerate() {
            *v += ((i * 7919) % 101) as f32 * 1e-4;
        }
    }
    let cpu_out = CpuBackend.matmul_dequant(&x, n_tokens, &w);
    // The float kernels: this is a tight check of the tiled GEMM against
    // the dequantized product, and the integer-dot kernel the generic
    // batch path takes at these widths is a different (8-bit activation)
    // computation with its own test, `mmq_q4k_gemm_matches_the_cpu_product`.
    let gpu_out = vulkan.without_prefill_mmq(|| vulkan.matmul(&x, n_tokens, &w));

    // The reference the GPU is checked against. The MMVQ path
    // (`ORANGU_Q4K_MMVQ`) quantizes the activation to q8, so comparing it to
    // the full-precision `cpu_out` conflates two things: whether the kernel
    // is *correct* (it should compute the q8-quantized matmul exactly), and
    // the inherent q8 rounding loss (which on this test's adversarial
    // random-uniform data is several % — real model activations quantize
    // far better). Verify the kernel against the **q8-quantized reference**
    // (float-dequant weights × the same quantized x) at a tight tolerance,
    // isolating kernel correctness from the expected quantization loss —
    // this is what llama.cpp's own q8 mat-vec cross-checks effectively do.
    // The same applies to the sub-`Q4_K` types' integer-dot decode kernels
    // (`decode_mmvq`) — when the op reaches the device at all: a small
    // one-token weight goes to the host matmul and is float there.
    let on_integer_dot = vulkan.mmvq_pipeline_for(&w, n_tokens).is_some()
        && !crate::engine::backend::prefers_host_matmul(&[MatmulOp {
            x: &x,
            n_tokens,
            w: &w,
        }]);
    let (reference, tol_factor) = if on_integer_dot {
        let wdq = crate::engine::quant::dequantize(ggml_type, &bytes, out_dim * in_dim).unwrap();
        let q8 = quantize_activation_q8(&x);
        let mut qx = vec![0f32; x.len()];
        for (blk, chunk) in q8.as_chunks::<10>().0.iter().enumerate() {
            let d = f32::from_bits(chunk[0]);
            for i in 0..32 {
                let byte = ((chunk[2 + i / 4] >> (8 * (i % 4))) & 0xFF) as u8 as i8;
                qx[blk * 32 + i] = d * byte as f32;
            }
        }
        let mut reference = vec![0f32; n_tokens * out_dim];
        for t in 0..n_tokens {
            for o in 0..out_dim {
                let mut s = 0f32;
                for e in 0..in_dim {
                    s += wdq[o * in_dim + e] * qx[t * in_dim + e];
                }
                reference[t * out_dim + o] = s;
            }
        }
        (reference, 1e-2)
    } else if vulkan.ternary_idot_for(&w, n_tokens) {
        // The integer-dot ternary kernel quantizes the activations to `int8`
        // per 128-block inside the kernel (`pack4x8snorm(x / amax)`); the
        // reference is the dequantized weights against exactly that
        // rounding, so the check is of the kernel, not of the quantization.
        let wdq = crate::engine::quant::dequantize(ggml_type, &bytes, out_dim * in_dim).unwrap();
        let mut qx = vec![0f32; x.len()];
        for (blk, chunk) in x.chunks(128).enumerate() {
            let amax = chunk.iter().fold(0f32, |m, v| m.max(v.abs()));
            let inv = if amax > 0.0 { 1.0 / amax } else { 0.0 };
            for (i, &v) in chunk.iter().enumerate() {
                // `PackSnorm4x8` rounds to nearest; this test's `x` grid
                // lands on exact halves often enough that `floor(0.5 + v)`
                // (WGSL's wording) is measurably a different rounding.
                let q = (127.0 * (v * inv).clamp(-1.0, 1.0)).round_ties_even();
                qx[blk * 128 + i] = q * (amax / 127.0);
            }
        }
        let mut reference = vec![0f32; n_tokens * out_dim];
        for t in 0..n_tokens {
            for o in 0..out_dim {
                let mut s = 0f32;
                for e in 0..in_dim {
                    s += wdq[o * in_dim + e] * qx[t * in_dim + e];
                }
                reference[t * out_dim + o] = s;
            }
        }
        (reference, 1e-2)
    } else {
        // `packed_dot_f16` (`ORANGU_PACKED_DOT`) also widens the dot to an
        // `f16` accumulate, needing the loose tolerance vs `cpu_out`.
        let packed = vulkan.packed_dot_f16
            && ggml_type == GGML_TYPE_Q4_K
            && n_tokens < vulkan.coop_min_n_tokens;
        (cpu_out.clone(), if packed { 6e-2 } else { 1e-2 })
    };

    assert_eq!(reference.len(), gpu_out.len());
    if std::env::var("ORANGU_CROSS_CHECK_STATS").is_ok() {
        let stats = |name: &str, r: &[f32]| {
            let (mut worst, mut sum) = (0f32, 0f32);
            for (a, b) in r.iter().zip(gpu_out.iter()) {
                let e = (a - b).abs() / a.abs().max(1.0);
                worst = worst.max(e);
                sum += e;
            }
            eprintln!(
                "{ggml_type} [{in_dim} x {out_dim}] x {n_tokens}: gpu vs {name}: worst {worst:.4} mean {:.5}",
                sum / r.len() as f32
            );
        };
        stats("reference", &reference);
        stats("cpu float", &cpu_out);
        let bad: Vec<String> = reference
            .iter()
            .zip(gpu_out.iter())
            .enumerate()
            .filter(|(_, (a, b))| (*a - *b).abs() > 1e-3 * a.abs().max(1.0))
            .take(24)
            .map(|(i, (a, b))| format!("{i}(t{} o{}): {a:.4} vs {b:.4}", i / out_dim, i % out_dim))
            .collect();
        eprintln!("  off: {}", bad.join(", "));
        if n_tokens == 1 {
            // Project the row errors onto each element's weight column — a
            // mis-quantized element shows as its LSB count.
            let wdq =
                crate::engine::quant::dequantize(ggml_type, &bytes, out_dim * in_dim).unwrap();
            for (blk, chunk) in x.chunks(128).enumerate() {
                let amax = chunk.iter().fold(0f32, |m, v| m.max(v.abs()));
                let step = amax / 127.0;
                let mut found = Vec::new();
                for e in 0..128 {
                    let (mut num, mut den) = (0f64, 0f64);
                    for o in 0..out_dim {
                        let w = wdq[o * in_dim + blk * 128 + e] as f64;
                        num += (reference[o] - gpu_out[o]) as f64 * w;
                        den += w * w;
                    }
                    let c = if den > 0.0 {
                        num / den / step as f64
                    } else {
                        0.0
                    };
                    if c.abs() > 0.3 {
                        let sc = 127.0 * chunk[e] / amax;
                        found.push(format!("e{e}: {c:+.2} LSB (x={} sc={sc:.4})", chunk[e]));
                    }
                }
                if !found.is_empty() {
                    eprintln!(
                        "  block {blk} amax {amax} (element {}): {}",
                        chunk.iter().position(|v| v.abs() == amax).unwrap(),
                        found.join(", ")
                    );
                }
            }
        }
    }
    for (i, (a, b)) in reference.iter().zip(gpu_out.iter()).enumerate() {
        let tol = tol_factor * a.abs().max(1.0);
        assert!(
            (a - b).abs() <= tol,
            "ggml_type {ggml_type}: mismatch at flat index {i}: ref={a} gpu={b}"
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

/// The `_1` legacy quants, which store a per-block minimum rather than
/// assuming a symmetric range around a fixed offset.
#[test]
fn matmul_matches_cpu_backend_for_q4_1() {
    cross_check(GGML_TYPE_Q4_1, 64, 17);
}

#[test]
fn matmul_matches_cpu_backend_for_q5_1() {
    cross_check(GGML_TYPE_Q5_1, 64, 17);
}

#[test]
fn matmul_matches_cpu_backend_for_q8_0() {
    cross_check(GGML_TYPE_Q8_0, 64, 17);
}

/// The legacy quants at a **model-shaped** `in_dim`, which is what
/// actually exercises the block-hoisted decode kernel.
///
/// `cross_check`'s usual `in_dim = 64` is two 32-element blocks. The
/// block-hoisted path assigns one block per lane and strides by the
/// workgroup, so at two blocks exactly two of sixty-four lanes do any
/// work and the loop never goes round twice — every indexing mistake
/// that depends on the stride, the second iteration, or a lane whose
/// first block is already past the end is invisible. 2816 is this
/// hardware's `gemma-4-26B-A4B` `n_embd` and 88 blocks: lanes 0..23 run
/// twice, 24..63 once, and the tail condition is live.
///
/// `out_dim = 7` is deliberately not a multiple of `REDUCE_N_ROWS`, so
/// the `o{i} < params.out_dim` guards are exercised too.
#[test]
fn matmul_matches_cpu_backend_for_q4_0_model_shaped() {
    cross_check(GGML_TYPE_Q4_0, 2816, 7);
}

#[test]
fn matmul_matches_cpu_backend_for_q4_1_model_shaped() {
    cross_check(GGML_TYPE_Q4_1, 2816, 7);
}

#[test]
fn matmul_matches_cpu_backend_for_q8_0_model_shaped() {
    cross_check(GGML_TYPE_Q8_0, 2816, 7);
}

#[test]
fn matmul_matches_cpu_backend_for_q4_k() {
    cross_check(GGML_TYPE_Q4_K, 512, 5);
}

/// The word-reading `block_dot`s on the **decode** path — and under
/// `ORANGU_DECODE_MMVQ=1` the integer-dot ones, against the q8-quantized
/// reference: one token, a
/// model-shaped width, and a weight large enough that the host-matmul rule
/// leaves it on the device — `cross_check`'s three tokens take the
/// thin-tile kernel and a small one-token weight goes to the CPU (both
/// proved by breaking the kernel under them: they kept passing). The
/// pipeline the backend picks is asserted to be the block-hoisted one.
/// Enough blocks for every lane to go round the block loop more than
/// once, an `out_dim` that is not a multiple of the row batch, and for the
/// 110/82-byte types both word parities of a block start.
fn cross_check_decode(ggml_type: u32, in_dim: usize) {
    let (block_bytes, block_elems) = crate::engine::quant::block_layout(ggml_type).unwrap();
    let row_bytes = in_dim / block_elems * block_bytes;
    // Above the host-matmul threshold, and wide enough for the four-row
    // pipelines (`BLOCK_HOISTED_WIDE_MIN_OUT`); the `+ 7` leaves a partial
    // row group.
    let out_dim = crate::engine::backend::host_matmul_threshold_bytes()
        .div_ceil(row_bytes)
        .max(2048)
        + 7;
    {
        let Some(vulkan) = shared_vulkan() else {
            return;
        };
        let name = vulkan.pipeline_for_named(ggml_type, in_dim, 1).1;
        assert!(
            matches!(name, "kq-light" | "block-hoisted"),
            "type {ggml_type} at one token: {name}"
        );
        if vulkan.decode_mmvq {
            let mut seed = 0x1D07_u64;
            let block = build_block(ggml_type, &mut seed);
            let w = test_quant_matrix(&block.repeat(in_dim / block_elems), ggml_type, in_dim, 1);
            assert!(
                vulkan.mmvq_pipeline_for(&w, 1).is_some(),
                "type {ggml_type}: no integer-dot decode pipeline"
            );
        }
    }
    cross_check_n_tokens(ggml_type, in_dim, out_dim, 1);
}

/// Prism's ternary pair at the 27B's width, on the integer dot where the
/// device has it (`decode_mmvq`, `shader_source_ternary_i8`).
#[test]
fn decode_matvec_matches_cpu_backend_for_the_ternary_types_model_shaped() {
    cross_check_decode(GGML_TYPE_PTQ1_0, 5120);
    cross_check_decode(GGML_TYPE_PQ2_0, 5120);
}

#[test]
fn decode_matvec_matches_cpu_backend_for_q2_k_model_shaped() {
    cross_check_decode(GGML_TYPE_Q2_K, 1536);
    cross_check_decode(GGML_TYPE_Q2_K, 6144);
}

#[test]
fn decode_matvec_matches_cpu_backend_for_q3_k_model_shaped() {
    cross_check_decode(GGML_TYPE_Q3_K, 1536);
    cross_check_decode(GGML_TYPE_Q3_K, 6144);
}

#[test]
fn decode_matvec_matches_cpu_backend_for_iq4_xs_model_shaped() {
    cross_check_decode(GGML_TYPE_IQ4_XS, 1536);
}

#[test]
fn decode_matvec_matches_cpu_backend_for_iq3_s_model_shaped() {
    cross_check_decode(GGML_TYPE_IQ3_S, 1536);
    cross_check_decode(GGML_TYPE_IQ3_S, 6144);
}

#[test]
fn decode_matvec_matches_cpu_backend_for_iq2_s_model_shaped() {
    cross_check_decode(GGML_TYPE_IQ2_S, 1536);
    cross_check_decode(GGML_TYPE_IQ2_S, 6144);
}

/// The legacy types on the integer dot (`decode_mmvq`), and on their
/// block-hoisted float form otherwise.
#[test]
fn decode_matvec_matches_cpu_backend_for_the_legacy_types_model_shaped() {
    for ggml_type in [
        GGML_TYPE_Q4_0,
        GGML_TYPE_Q4_1,
        GGML_TYPE_Q5_0,
        GGML_TYPE_Q5_1,
        GGML_TYPE_Q8_0,
    ] {
        cross_check_decode(ggml_type, 1536);
        cross_check_decode(ggml_type, 6144);
    }
}

/// A prefill-width batch routes to *every* expert in a layer, which on
/// this model is 408 MiB against a 256 MiB region. Grouping is what keeps
/// that inside the region: each group is dispatched separately and so
/// preceded by its own rewind.
///
/// Getting it wrong is not a crash — `weight_buffer_streamed` returns
/// `None` for whatever no longer fits and those weights go to the
/// permanent arena instead, which never evicts. So an overflowing group
/// still computes the right answer while quietly reinstating the
/// residency cap streaming exists to remove.
#[test]
fn a_streamed_batch_is_split_into_groups_that_fit_the_region() {
    // 34 raw bytes each (one `Q8_0` block), placed on 256-byte
    // boundaries, so the budget below is exactly three weights wide.
    const ALIGN: u64 = 256;
    let bytes: Vec<Vec<u8>> = (0..5)
        .map(|_| {
            let mut seed = 0xA11CE_u64;
            build_block(GGML_TYPE_Q8_0, &mut seed)
        })
        .collect();
    let mats: Vec<QuantMatrix> = bytes
        .iter()
        .map(|b| test_quant_matrix(b, GGML_TYPE_Q8_0, 32, 1))
        .collect();
    let x = vec![0.5f32; 32];
    fn op<'a>(x: &'a [f32], w: &'a QuantMatrix) -> MatmulOp<'a> {
        MatmulOp { x, n_tokens: 1, w }
    }

    let five: Vec<MatmulOp<'_>> = mats.iter().map(|w| op(&x, w)).collect();
    let groups = stream_groups(&five, 3 * ALIGN, ALIGN);
    assert_eq!(
        groups.iter().map(|(g, _)| g.len()).collect::<Vec<_>>(),
        vec![3, 2],
        "five weights into a three-wide region is 3 + 2"
    );
    // The reported size is what `reserve_stream_space` rewinds against,
    // so a group that under-reports its own bytes would overflow the
    // region it was just told it fits in.
    assert_eq!(
        groups.iter().map(|&(_, b)| b).collect::<Vec<_>>(),
        vec![3 * ALIGN, 2 * ALIGN]
    );
    // Whatever the split, every op has to be dispatched exactly once and
    // in order — the caller concatenates the groups' results positionally.
    let flattened: Vec<*const QuantMatrix> = groups
        .iter()
        .flat_map(|(g, _)| g.iter().map(|op| std::ptr::from_ref(op.w)))
        .collect();
    let want: Vec<*const QuantMatrix> = five.iter().map(|op| std::ptr::from_ref(op.w)).collect();
    assert_eq!(flattened, want);

    // One tensor named twice is uploaded once per epoch (a fused gate/up
    // pair is two row ranges of one expert), so counting it twice would
    // split a batch that fits.
    let repeated = vec![
        op(&x, &mats[0]),
        op(&x, &mats[0]),
        op(&x, &mats[1]),
        op(&x, &mats[2]),
    ];
    let groups = stream_groups(&repeated, 3 * ALIGN, ALIGN);
    assert_eq!(
        groups.len(),
        1,
        "three distinct weights fit a three-wide region however often they are named"
    );

    // A budget below a single weight: each gets its own group rather than
    // one overflowing group, and the batch still runs (that weight takes
    // the permanent-arena fallback).
    let groups = stream_groups(&five, 1, ALIGN);
    assert_eq!(groups.len(), 5);
    assert!(groups.iter().all(|(g, _)| g.len() == 1));
}

/// `n` distinct `Q4_0` weights of one shape, plus activations — the
/// shape a batch of routed experts has.
/// The tolerance for one output element of a matmul checked against the
/// float reference: a fraction of the element where the op runs in float,
/// a fraction of the *largest* output where it runs on the integer dot —
/// an 8-bit activation moves an output by a fraction of the terms that
/// formed it, and an element the terms cancel to can be small.
fn matmul_tolerance(
    vulkan: &VulkanBackend,
    w: &QuantMatrix,
    n_tokens: usize,
    want: &[f32],
) -> impl Fn(f32) -> f32 {
    let integer_dot = vulkan.mmvq_pipeline_for(w, n_tokens).is_some();
    let scale = want.iter().fold(1.0f32, |m, v| m.max(v.abs()));
    move |a: f32| {
        if integer_dot {
            2e-2 * scale
        } else {
            1e-2 * a.abs().max(1.0)
        }
    }
}

fn streamed_expert_fixture(
    n: usize,
    in_dim: usize,
    out_dim: usize,
    n_tokens: usize,
) -> (Vec<Vec<u8>>, Vec<f32>) {
    let mut seed = 0x5EED_D0DE_u64;
    let blocks = out_dim * (in_dim / block_elems(GGML_TYPE_Q4_0));
    let weights = (0..n)
        .map(|_| {
            (0..blocks)
                .flat_map(|_| build_block(GGML_TYPE_Q4_0, &mut seed))
                .collect()
        })
        .collect();
    let x = (0..n_tokens * in_dim)
        .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
        .collect();
    (weights, x)
}

/// Two different weights of the **same shape**, streamed one call after
/// the other.
///
/// Streamed ops share one cache entry per shape — that is what stops the
/// path allocating fresh scratch for every expert of every call — and the
/// entry's bind group pins the region its weights sat in, which the next
/// call rewinds. So the second call has to follow its weight
/// (`rebind_weight`) or it computes the *first* expert's answer against
/// the second's activations: a plausible matmul of the wrong tensor, with
/// nothing about it to notice.
#[test]
fn a_second_streamed_weight_of_one_shape_reuses_the_entry_and_still_computes_its_own() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    // A shape of this test's own: the whole point is that streamed
    // entries are keyed by shape, so a shape shared with another test
    // would have that test's entries counted here — and the tests in this
    // binary share one backend and run concurrently.
    const IN_DIM: usize = 512;
    const OUT_DIM: usize = 4;
    const N_TOKENS: usize = 3;

    let (bytes, x) = streamed_expert_fixture(2, IN_DIM, OUT_DIM, N_TOKENS);
    let mats: Vec<QuantMatrix> = bytes
        .iter()
        .map(|b| test_quant_matrix(b, GGML_TYPE_Q4_0, IN_DIM, OUT_DIM))
        .collect();
    let want: Vec<Vec<f32>> = mats
        .iter()
        .map(|w| CpuBackend.matmul_dequant(&x, N_TOKENS, w))
        .collect();
    // Without this the test would pass on a backend that ignored the
    // second weight entirely — the exact bug it exists to catch.
    assert!(
        want[0]
            .iter()
            .zip(&want[1])
            .any(|(a, b)| (a - b).abs() > 1e-2),
        "the two fixtures must give different results for this test to mean anything"
    );

    fn op<'a>(x: &'a [f32], n_tokens: usize, w: &'a QuantMatrix) -> MatmulOp<'a> {
        MatmulOp { x, n_tokens, w }
    }
    // Streamed entries only (`waddr == 0`), and only this shape's.
    let entries = || {
        vulkan
            .op_cache
            .lock()
            .expect("op cache poisoned")
            .keys()
            .filter(|k| k.0 == 0 && k.3 == IN_DIM && k.4 == OUT_DIM)
            .count()
    };

    let first = vulkan.matmul_batch_streamed(&[op(&x, N_TOKENS, &mats[0])]);
    let after_first = entries();
    let second = vulkan.matmul_batch_streamed(&[op(&x, N_TOKENS, &mats[1])]);
    let after_second = entries();

    for (got, want, w) in [
        (&first[0], &want[0], &mats[0]),
        (&second[0], &want[1], &mats[1]),
    ] {
        assert_eq!(got.len(), want.len());
        let tol = matmul_tolerance(vulkan, w, N_TOKENS, want);
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= tol(*w),
                "element {i}: gpu={g} cpu={w} on {}",
                vulkan.adapter_name
            );
        }
    }
    assert_eq!(
        after_first, after_second,
        "a second streamed weight of the same shape must rebind the first's \
             entry, not allocate its own"
    );
}

/// Two streamed experts of the same shape **in one batch**, which are
/// live at the same instant and so cannot share an entry.
///
/// The shape key is what makes that a live hazard: they differ only in
/// the `region_slot` the batch assigns by position. Get that wrong and
/// both dispatches write the same output region, so one expert's result
/// silently becomes the other's.
#[test]
fn two_streamed_experts_of_one_shape_in_a_batch_keep_their_own_outputs() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    // Not the shape the entry-reuse test above uses — it counts entries
    // by shape, and these tests share one backend.
    const IN_DIM: usize = 640;
    const OUT_DIM: usize = 5;
    const N_TOKENS: usize = 3;

    let (bytes, x) = streamed_expert_fixture(2, IN_DIM, OUT_DIM, N_TOKENS);
    let mats: Vec<QuantMatrix> = bytes
        .iter()
        .map(|b| test_quant_matrix(b, GGML_TYPE_Q4_0, IN_DIM, OUT_DIM))
        .collect();
    let want: Vec<Vec<f32>> = mats
        .iter()
        .map(|w| CpuBackend.matmul_dequant(&x, N_TOKENS, w))
        .collect();
    assert!(
        want[0]
            .iter()
            .zip(&want[1])
            .any(|(a, b)| (a - b).abs() > 1e-2),
        "the two fixtures must give different results for this test to mean anything"
    );

    let ops: Vec<MatmulOp<'_>> = mats
        .iter()
        .map(|w| MatmulOp {
            x: &x,
            n_tokens: N_TOKENS,
            w,
        })
        .collect();
    let got = vulkan.matmul_batch_streamed(&ops);

    for (e, (got, want)) in got.iter().zip(&want).enumerate() {
        let tol = matmul_tolerance(vulkan, ops[e].w, N_TOKENS, want);
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= tol(*w),
                "expert {e} element {i}: gpu={g} cpu={w} on {}",
                vulkan.adapter_name
            );
        }
    }
}

/// A streamed batch **wider than one submission's stripe** — the shape a
/// host-resident dense layer's projections take when a split model's
/// prefill sends them to the card — matches the CPU product, and its
/// weights cross the bus once: every stripe after the first finds them
/// resident.
///
/// Model-shaped rows, the two weight types a `Q4_K_M` layer has, and a
/// width that is not a stripe multiple, so the last stripe is a ragged
/// tail. The arena counters say what the bus saw: `uploads` grows by the
/// number of distinct weights, `hits` by one per op per extra stripe.
#[test]
fn a_streamed_batch_wider_than_a_stripe_matches_the_cpu_backend_and_uploads_once() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    const IN_DIM: usize = 3840;
    const OUT_DIM: usize = 256;
    let n_tokens = crate::engine::backend::vulkan::max_matmul_tokens_per_submission() * 2 + 77;
    let stripes =
        n_tokens.div_ceil(crate::engine::backend::vulkan::max_matmul_tokens_per_submission());

    let mut seed = 0x57A1_7ED5_u64;
    let mut mats = Vec::new();
    let mut raws = Vec::new();
    for ggml_type in [GGML_TYPE_Q4_K, GGML_TYPE_Q6_K] {
        let blocks = OUT_DIM * (IN_DIM / block_elems(ggml_type));
        let bytes: Vec<u8> = (0..blocks)
            .flat_map(|_| build_block(ggml_type, &mut seed))
            .collect();
        mats.push(test_quant_matrix(&bytes, ggml_type, IN_DIM, OUT_DIM));
        raws.push((ggml_type, bytes));
    }
    let x: Vec<f32> = (0..n_tokens * IN_DIM)
        .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
        .collect();
    let ops: Vec<MatmulOp<'_>> = mats
        .iter()
        .map(|w| MatmulOp { x: &x, n_tokens, w })
        .collect();

    let before = vulkan
        .stream_upload_rate()
        .unwrap_or((0, 0.0, 0, 0, 0, 0, 0));
    let got = vulkan.matmul_batch_streamed(&ops);
    let after = vulkan.stream_upload_rate().expect("the call streamed");
    assert_eq!(
        after.2 - before.2,
        mats.len() as u64,
        "each distinct weight uploads once for the whole call"
    );
    assert_eq!(
        after.4 - before.4,
        (mats.len() * (stripes - 1)) as u64,
        "every stripe after the first finds its weights resident"
    );

    // The integer-dot GEMM's own bound (`mmq_q4k_gemm_matches_the_cpu_product`):
    // the error against a token row's magnitude, since the activation is
    // 8-bit per block and a wrong stripe would be off by whole rows.
    for (e, (got, op)) in got.iter().zip(&ops).enumerate() {
        assert_eq!(
            got.len(),
            n_tokens * OUT_DIM,
            "op {e} has every token's row"
        );
        let want = CpuBackend.matmul_dequant(&x, n_tokens, op.w);
        for t in 0..n_tokens {
            let row = &want[t * OUT_DIM..(t + 1) * OUT_DIM];
            let mag = row.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-3);
            for o in 0..OUT_DIM {
                let i = t * OUT_DIM + o;
                assert!(
                    (got[i] - want[i]).abs() / mag <= 2e-2,
                    "op {e} token {t} row {o}: gpu={} cpu={} (row magnitude {mag}) on {}",
                    got[i],
                    want[i],
                    vulkan.adapter_name
                );
            }
        }
    }
}

/// The split backend sends a **host layer's** wide projection to the card
/// and leaves everything else where it was: a weight tagged for the host
/// device streams at prefill width and comes back within the integer-dot
/// GEMM's bound; the same weight at one token, and at any width through
/// the decode entry points, is bit for bit the CPU's own product.
///
/// The card is device 0 and the CPU device 1, the way a `device_split =
/// cpu` plan lays them out; the weight is stamped for device 1 the way
/// `LoadedModel::matrix` stamps a host layer's.
#[test]
fn the_split_backend_streams_a_host_layers_wide_projection_and_nothing_else() {
    use crate::engine::backend::Backend;
    use crate::engine::backend::multi::MultiDeviceBackend;
    use std::sync::Arc;

    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    if crate::engine::backend::multi::host_stream_min_tokens() == 0 {
        eprintln!("host-layer streaming is off (ORANGU_HOST_STREAM_TOKENS=0) — nothing to check");
        return;
    }
    /// The shared test card as a `dyn Backend` the split wrapper can own.
    struct Card(&'static VulkanBackend);
    impl Backend for Card {
        fn matmul(&self, x: &[f32], n_tokens: usize, w: &QuantMatrix) -> Vec<f32> {
            self.0.matmul(x, n_tokens, w)
        }
        fn matmul_batch(&self, ops: &[MatmulOp<'_>]) -> Vec<Vec<f32>> {
            self.0.matmul_batch(ops)
        }
        fn as_wgpu(&self) -> Option<&VulkanBackend> {
            Some(self.0)
        }
        fn supports_type(&self, ggml_type: u32) -> bool {
            self.0.supports_type(ggml_type)
        }
    }
    let split = MultiDeviceBackend::new(vec![
        Arc::new(Card(vulkan)) as Arc<dyn Backend>,
        Arc::new(CpuBackend) as Arc<dyn Backend>,
    ]);

    const IN_DIM: usize = 1536;
    const OUT_DIM: usize = 256;
    let mut seed = 0x0057_5EA7_u64;
    let blocks = OUT_DIM * (IN_DIM / block_elems(GGML_TYPE_Q4_K));
    let bytes: Vec<u8> = (0..blocks)
        .flat_map(|_| build_block(GGML_TYPE_Q4_K, &mut seed))
        .collect();
    let mut w = test_quant_matrix(&bytes, GGML_TYPE_Q4_K, IN_DIM, OUT_DIM);
    w.set_device(1);
    let wide = crate::engine::backend::multi::host_stream_min_tokens().max(64);
    let x: Vec<f32> = (0..wide * IN_DIM)
        .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
        .collect();

    // Prefill width: streamed, so it is the card's integer-dot product —
    // the bus saw an upload, and the result meets that kernel's bound.
    let before = vulkan.stream_upload_rate().map_or(0, |r| r.2);
    let got = split.matmul(&x, wide, &w);
    let after = vulkan.stream_upload_rate().map_or(0, |r| r.2);
    assert_eq!(after - before, 1, "the wide host-layer op streamed");
    let want = CpuBackend.matmul_dequant(&x, wide, &w);
    for t in 0..wide {
        let row = &want[t * OUT_DIM..(t + 1) * OUT_DIM];
        let mag = row.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-3);
        for o in 0..OUT_DIM {
            let i = t * OUT_DIM + o;
            assert!(
                (got[i] - want[i]).abs() / mag <= 2e-2,
                "token {t} row {o}: streamed {} vs cpu {} (row magnitude {mag})",
                got[i],
                want[i]
            );
        }
    }
    let batched = split.matmul_batch(&[
        MatmulOp {
            x: &x,
            n_tokens: wide,
            w: &w,
        },
        MatmulOp {
            x: &x,
            n_tokens: wide,
            w: &w,
        },
    ]);
    assert_eq!(batched.len(), 2);
    assert_eq!(
        batched[0], got,
        "a batch streams the same way a single op does"
    );
    assert_eq!(batched[1], got);

    // One token, and decode at any width: the host's own product, exactly.
    let one = split.matmul(&x[..IN_DIM], 1, &w);
    assert_eq!(one, CpuBackend.matmul(&x[..IN_DIM], 1, &w));
    let before = vulkan.stream_upload_rate().map_or(0, |r| r.2);
    let decoded = split.matmul_decode(&x, wide, &w);
    assert_eq!(decoded, CpuBackend.matmul_decode(&x, wide, &w));
    let decoded = split.matmul_batch_decode(&[MatmulOp {
        x: &x,
        n_tokens: wide,
        w: &w,
    }]);
    assert_eq!(decoded[0], CpuBackend.matmul_decode(&x, wide, &w));
    let after = vulkan.stream_upload_rate().map_or(0, |r| r.2);
    assert_eq!(after, before, "neither one token nor decode streamed");
}

/// The indexed expert GEMM (`matmul_experts`) against the per-expert CPU
/// product: a stack of experts, each multiplied by the rows routed to it,
/// as the two halves of a fused gate/up tensor (`Q4_K`, the 64-row tile)
/// and a down projection (`Q8_0`, the 128-row tile, once at a row width
/// that is not whole super-blocks). Routing is uneven on
/// purpose — an expert with more tokens than a token tile, one with a
/// single token, and one routed nothing — so the table's tiles, padding
/// and output bases are all exercised.
#[test]
fn the_indexed_expert_gemm_matches_the_per_expert_cpu_product() {
    use crate::engine::backend::vulkan::ExpertOp;
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    const N_EXPERT: usize = 6;
    const N_TOKENS: usize = 300;
    let mut seed = 0x1DE7_E4A7_u64;
    // (type, in_dim, rows per expert, (first_row, n_rows) per op)
    type Case = (u32, usize, usize, Vec<(usize, usize)>);
    let cases: [Case; 3] = [
        (GGML_TYPE_Q4_K, 512, 256, vec![(0, 128), (128, 128)]),
        (
            crate::engine::quant::GGML_TYPE_Q8_0,
            256,
            256,
            vec![(0, 256)],
        ),
        // A row that is not whole super-blocks: 22 sub-blocks, the shape
        // of a 704-wide expert down projection.
        (
            crate::engine::quant::GGML_TYPE_Q8_0,
            704,
            256,
            vec![(0, 256)],
        ),
    ];
    for (ggml_type, in_dim, rows_per_expert, projections) in cases {
        let blocks = N_EXPERT * rows_per_expert * (in_dim / block_elems(ggml_type));
        let bytes: Vec<u8> = (0..blocks)
            .flat_map(|_| build_block(ggml_type, &mut seed))
            .collect();
        let stack = test_quant_matrix(&bytes, ggml_type, in_dim, N_EXPERT * rows_per_expert);
        let x: Vec<f32> = (0..N_TOKENS * in_dim)
            .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
            .collect();
        // Expert 3 gets nothing, expert 0 gets most tokens (over a tile),
        // expert 5 exactly one; the rest share what is left.
        let mut groups: Vec<(usize, Vec<usize>)> = vec![
            (0, Vec::new()),
            (1, Vec::new()),
            (2, Vec::new()),
            (4, Vec::new()),
            (5, vec![7]),
        ];
        for t in 0..N_TOKENS {
            let g = match t % 5 {
                0..=2 => 0,
                3 => 1,
                _ => {
                    if t % 10 == 4 {
                        2
                    } else {
                        3
                    }
                }
            };
            groups[g].1.push(t);
        }
        assert!(
            groups[0].1.len() > 128,
            "expert 0 spans more than one token tile"
        );
        let ops: Vec<ExpertOp<'_>> = projections
            .iter()
            .map(|&(first_row, n_rows)| ExpertOp {
                stack: &stack,
                rows_per_expert,
                first_row,
                n_rows,
                scale: None,
            })
            .collect();
        if !vulkan.serves_experts(&ops) {
            // A device without the integer-dot kernels (no accelerated
            // packed dot, or `ORANGU_PREFILL_MMQ=0`) has no indexed GEMM
            // either; the routed experts stay on the host there.
            eprintln!("type {ggml_type}: no indexed kernel on this device — nothing to check");
            return;
        }
        let got = vulkan.matmul_experts(&x, N_TOKENS, in_dim, &ops, &groups, &[]);
        assert_eq!(got.len(), ops.len());
        for (op_i, (op, per_group)) in ops.iter().zip(&got).enumerate() {
            assert_eq!(per_group.len(), groups.len());
            for ((expert, tokens), result) in groups.iter().zip(per_group) {
                let w = stack.rows(expert * rows_per_expert + op.first_row, op.n_rows);
                let mut gathered = Vec::with_capacity(tokens.len() * in_dim);
                for &t in tokens {
                    gathered.extend_from_slice(&x[t * in_dim..(t + 1) * in_dim]);
                }
                let want = CpuBackend.matmul_dequant(&gathered, tokens.len(), &w);
                assert_eq!(result.len(), want.len(), "op {op_i} expert {expert}");
                for t in 0..tokens.len() {
                    let row = &want[t * op.n_rows..(t + 1) * op.n_rows];
                    let mag = row.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-3);
                    for o in 0..op.n_rows {
                        let i = t * op.n_rows + o;
                        assert!(
                            (result[i] - want[i]).abs() / mag <= 2e-2,
                            "type {ggml_type} op {op_i} expert {expert} member {t} row {o}: \
                             indexed {} vs cpu {} (row magnitude {mag})",
                            result[i],
                            want[i]
                        );
                    }
                }
            }
        }
    }
}

/// The grouped expert GEMM on a real layer's stacks, phase by phase: the
/// first call uploads the stack, the second finds it resident, so the
/// difference is the upload and the second call is quantize + GEMM +
/// readback. Random routing of 8 experts per token over `ORANGU_PROBE_TOKENS`
/// (512) tokens, the shape of a prefill chunk.
///
/// `ORANGU_PROBE_GGUF=/path/to/moe.gguf cargo test _scratch_measure_expert_gemm -- --ignored --nocapture`
#[test]
#[ignore = "needs a real GGUF; run with --ignored"]
fn _scratch_measure_expert_gemm() {
    use crate::engine::backend::vulkan::ExpertOp;
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        return;
    };
    let path = std::env::var("ORANGU_PROBE_GGUF").expect("set ORANGU_PROBE_GGUF");
    let n_tokens: usize = std::env::var("ORANGU_PROBE_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    let model =
        crate::engine::loader::LoadedModel::open(std::path::Path::new(&path)).expect("open gguf");
    let gate_up = model
        .expert_matrix("blk.0.ffn_gate_up_exps.weight")
        .expect("a fused gate/up stack");
    let down = model
        .expert_matrix("blk.0.ffn_down_exps.weight")
        .expect("a down stack");
    let n_expert = gate_up.n_expert;
    let n_ff = gate_up.out_dim / 2;
    let mut seed = 0xE0E0_7EA5_u64;
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); n_expert];
    for t in 0..n_tokens {
        let mut picked = std::collections::HashSet::new();
        while picked.len() < 8 {
            let e =
                (next_byte(&mut seed) as usize * 256 + next_byte(&mut seed) as usize) % n_expert;
            if picked.insert(e) {
                members[e].push(t);
            }
        }
    }
    let groups: Vec<(usize, Vec<usize>)> = members
        .into_iter()
        .enumerate()
        .filter(|(_, m)| !m.is_empty())
        .collect();
    let total: usize = groups.iter().map(|(_, m)| m.len()).sum();
    let x: Vec<f32> = (0..n_tokens * gate_up.in_dim)
        .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 512.0)
        .collect();
    let gu_stack = gate_up.stack_matrix();
    let down_stack = down.stack_matrix();
    let gu_ops = [
        ExpertOp {
            stack: &gu_stack,
            rows_per_expert: gate_up.out_dim,
            first_row: 0,
            n_rows: n_ff,
            scale: None,
        },
        ExpertOp {
            stack: &gu_stack,
            rows_per_expert: gate_up.out_dim,
            first_row: n_ff,
            n_rows: n_ff,
            scale: None,
        },
    ];
    let x2: Vec<f32> = (0..total * down.in_dim)
        .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 512.0)
        .collect();
    let mut ranges = Vec::new();
    let mut at = 0;
    for (e, m) in &groups {
        ranges.push((*e, (at..at + m.len()).collect::<Vec<_>>()));
        at += m.len();
    }
    let down_ops = [ExpertOp {
        stack: &down_stack,
        rows_per_expert: down.out_dim,
        first_row: 0,
        n_rows: down.out_dim,
        scale: None,
    }];
    assert!(vulkan.serves_experts(&gu_ops) && vulkan.serves_experts(&down_ops));
    eprintln!(
        "  {n_expert} experts, {} groups, {total} rows; gate/up {} MiB, down {} MiB",
        groups.len(),
        gu_stack.raw_bytes().len() >> 20,
        down_stack.raw_bytes().len() >> 20
    );
    for round in 0..3 {
        let t = std::time::Instant::now();
        let _ = vulkan.matmul_experts(
            &x,
            n_tokens,
            gate_up.in_dim,
            &gu_ops,
            &groups,
            &[&down_stack],
        );
        let gu_ms = t.elapsed().as_secs_f64() * 1000.0;
        let t = std::time::Instant::now();
        let _ = vulkan.matmul_experts(&x2, total, down.in_dim, &down_ops, &ranges, &[&gu_stack]);
        let down_ms = t.elapsed().as_secs_f64() * 1000.0;
        // Again with both stacks resident: the region holds one at a time
        // when they exceed it together, so this is only resident when the
        // region is large enough for both.
        eprintln!("  round {round}: gate/up {gu_ms:.1} ms, down {down_ms:.1} ms");
    }
    // The whole routed feed-forward, with device stamps when
    // `ORANGU_GPU_TIMESTAMPS=ops` is set — the per-dispatch times to put
    // beside an in-situ `ops.sh` run.
    let down_op = ExpertOp {
        stack: &down_stack,
        rows_per_expert: down.out_dim,
        first_row: 0,
        n_rows: down.out_dim,
        scale: None,
    };
    for round in 0..3 {
        vulkan.begin_op_span();
        let t = std::time::Instant::now();
        let _ = vulkan.moe_ffn_experts(
            &x,
            n_tokens,
            gate_up.in_dim,
            Some(&gu_ops[0]),
            &gu_ops[1],
            &down_op,
            &groups,
            MoeActivation::Geglu,
            None,
            &[&gu_stack, &down_stack],
        );
        eprintln!(
            "  fused ffn round {round}: {:.1} ms",
            t.elapsed().as_secs_f64() * 1000.0
        );
        vulkan.finish_op_span(round);
    }
    // The GEMM alone: the same stack twice in a row is a hit.
    for _ in 0..2 {
        let t = std::time::Instant::now();
        let _ = vulkan.matmul_experts(&x, n_tokens, gate_up.in_dim, &gu_ops, &groups, &[]);
        eprintln!(
            "  gate/up resident: {:.1} ms",
            t.elapsed().as_secs_f64() * 1000.0
        );
    }
    for _ in 0..2 {
        let t = std::time::Instant::now();
        let _ = vulkan.matmul_experts(&x2, total, down.in_dim, &down_ops, &ranges, &[]);
        eprintln!(
            "  down resident: {:.1} ms",
            t.elapsed().as_secs_f64() * 1000.0
        );
    }
}

/// The one-submission routed feed-forward (`moe_ffn_experts`) against the
/// same computation done by parts on the CPU: per expert, gate and up from
/// the dequantized stack, `gelu(gate) * up`, then down — the shape of a
/// fused gate/up tensor (`Q4_K`) and a `Q8_0` down stack whose row is the
/// activation's width. Two 8-bit activation quantizations sit between
/// the reference and the card, so the bound is wider than the single
/// GEMM's: one row in 334 of this data lands 4.5% off, and the same row is
/// 4.5% off through the two-call form (`ORANGU_TEST_DUMP=1` prints both),
/// so that is the activation's rounding and not the chain's.
#[test]
fn the_fused_routed_ffn_matches_the_cpu_computation_by_parts() {
    use crate::engine::backend::vulkan::ExpertOp;
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    const N_EXPERT: usize = 5;
    const N_TOKENS: usize = 200;
    const N_EMBD: usize = 512;
    const N_FF: usize = 128;
    let mut seed = 0xF0F0_FEED_u64;
    let gu_blocks = N_EXPERT * 2 * N_FF * (N_EMBD / block_elems(GGML_TYPE_Q4_K));
    let gu_bytes: Vec<u8> = (0..gu_blocks)
        .flat_map(|_| build_block(GGML_TYPE_Q4_K, &mut seed))
        .collect();
    let gu_stack = test_quant_matrix(&gu_bytes, GGML_TYPE_Q4_K, N_EMBD, N_EXPERT * 2 * N_FF);
    let q8_0 = crate::engine::quant::GGML_TYPE_Q8_0;
    let dn_blocks = N_EXPERT * N_EMBD * (N_FF / block_elems(q8_0));
    let dn_bytes: Vec<u8> = (0..dn_blocks)
        .flat_map(|_| build_block(q8_0, &mut seed))
        .collect();
    let dn_stack = test_quant_matrix(&dn_bytes, q8_0, N_FF, N_EXPERT * N_EMBD);
    // Small activations: the random blocks' scales put a projection of a
    // unit-scale row far out on the GELU's flat sides, where the product's
    // 8-bit rounding is a step rather than a rounding.
    let x: Vec<f32> = (0..N_TOKENS * N_EMBD)
        .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 65536.0)
        .collect();
    let mut groups: Vec<(usize, Vec<usize>)> = (0..N_EXPERT).map(|e| (e, Vec::new())).collect();
    for t in 0..N_TOKENS {
        groups[t % 3].1.push(t);
        groups[3 + (t % 2)].1.push(t);
    }
    let gate = ExpertOp {
        stack: &gu_stack,
        rows_per_expert: 2 * N_FF,
        first_row: 0,
        n_rows: N_FF,
        scale: None,
    };
    let up = ExpertOp {
        stack: &gu_stack,
        rows_per_expert: 2 * N_FF,
        first_row: N_FF,
        n_rows: N_FF,
        scale: None,
    };
    let down = ExpertOp {
        stack: &dn_stack,
        rows_per_expert: N_EMBD,
        first_row: 0,
        n_rows: N_EMBD,
        scale: None,
    };
    if [&gate, &up, &down]
        .into_iter()
        .any(|op| !vulkan.serves_experts(std::slice::from_ref(op)))
    {
        eprintln!("no indexed kernel on this device — nothing to check");
        return;
    }
    let got = vulkan.moe_ffn_experts(
        &x,
        N_TOKENS,
        N_EMBD,
        Some(&gate),
        &up,
        &down,
        &groups,
        MoeActivation::Geglu,
        None,
        &[],
    );
    assert_eq!(got.len(), groups.len());

    // Packed into stripes of two experts (the region cut to just over two
    // experts' gate/up rows): the same rows, group for group, as the
    // stack placed whole — the stripes differ only in where the rows sit.
    if vulkan.packs_experts() {
        let two = 2 * (2 * N_FF * gu_stack.row_bytes()) as u64 + 3 * (256 + 16);
        let striped = vulkan.with_expert_stripe_bytes(two, || {
            vulkan.moe_ffn_experts(
                &x,
                N_TOKENS,
                N_EMBD,
                Some(&gate),
                &up,
                &down,
                &groups,
                MoeActivation::Geglu,
                None,
                &[],
            )
        });
        assert_eq!(striped.len(), got.len());
        for (g, (a, b)) in got.iter().zip(&striped).enumerate() {
            assert_eq!(a.len(), b.len(), "group {g}");
            for (i, (p, q)) in a.iter().zip(b).enumerate() {
                assert!(
                    (p - q).abs() <= 1e-5 * p.abs().max(1e-3),
                    "group {g} element {i}: whole {p} vs striped {q}"
                );
            }
        }
    } else {
        eprintln!("no staging pair on this adapter; the striped form is not exercised");
    }

    // Per-expert output scales on gate and up, applied by the GEMMs as
    // they store: against the by-parts reference with the same scales.
    let gate_scale = [0.5f32, 1.5, 0.25, 2.0, 1.0];
    let up_scale = [1.0f32, 0.75, 1.25, 0.5, 2.0];
    let scaled = vulkan.moe_ffn_experts(
        &x,
        N_TOKENS,
        N_EMBD,
        Some(&ExpertOp {
            scale: Some(&gate_scale),
            ..gate
        }),
        &ExpertOp {
            scale: Some(&up_scale),
            ..up
        },
        &down,
        &groups,
        MoeActivation::Geglu,
        None,
        &[],
    );
    for ((expert, tokens), result) in groups.iter().zip(&scaled) {
        let mut gathered = Vec::with_capacity(tokens.len() * N_EMBD);
        for &t in tokens {
            gathered.extend_from_slice(&x[t * N_EMBD..(t + 1) * N_EMBD]);
        }
        let wg = gu_stack.rows(expert * 2 * N_FF, N_FF);
        let wu = gu_stack.rows(expert * 2 * N_FF + N_FF, N_FF);
        let wd = dn_stack.rows(expert * N_EMBD, N_EMBD);
        let mut g = CpuBackend.matmul_dequant(&gathered, tokens.len(), &wg);
        let mut u = CpuBackend.matmul_dequant(&gathered, tokens.len(), &wu);
        g.iter_mut().for_each(|v| *v *= gate_scale[*expert]);
        u.iter_mut().for_each(|v| *v *= up_scale[*expert]);
        crate::engine::tensor::gelu_inplace(&mut g);
        crate::engine::tensor::mul_inplace(&mut g, &u);
        let want = CpuBackend.matmul_dequant(&g, tokens.len(), &wd);
        for t in 0..tokens.len() {
            let row = &want[t * N_EMBD..(t + 1) * N_EMBD];
            let mag = row.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-3);
            for o in 0..N_EMBD {
                let i = t * N_EMBD + o;
                assert!(
                    (result[i] - want[i]).abs() / mag <= 5e-2,
                    "scaled: expert {expert} member {t} row {o}: fused {} vs cpu {} (row magnitude {mag})",
                    result[i],
                    want[i]
                );
            }
        }
    }

    // The same with the rows combined on the card: every token's two
    // picks weighted 0.25 and 0.75, against the weighted sum of the rows
    // just returned.
    let k = 2;
    let mut table = vec![0u32; 2 * N_TOKENS * k];
    let mut base = 0usize;
    for (g, (_, tokens)) in groups.iter().enumerate() {
        for (m, &t) in tokens.iter().enumerate() {
            let rank = if g < 3 { 0 } else { 1 };
            table[t * k + rank] = (base + m) as u32;
            table[N_TOKENS * k + t * k + rank] =
                if rank == 0 { 0.25f32 } else { 0.75f32 }.to_bits();
        }
        base += tokens.len();
    }
    let combine = crate::engine::backend::vulkan::MoeCombine {
        table,
        n_tokens: N_TOKENS,
        k,
    };
    let combined = vulkan
        .moe_ffn_experts(
            &x,
            N_TOKENS,
            N_EMBD,
            Some(&gate),
            &up,
            &down,
            &groups,
            MoeActivation::Geglu,
            Some(&combine),
            &[],
        )
        .pop()
        .expect("the combined rows");
    assert_eq!(combined.len(), N_TOKENS * N_EMBD);
    let mut want = vec![0f32; N_TOKENS * N_EMBD];
    for (g, (_, tokens)) in groups.iter().enumerate() {
        let w = if g < 3 { 0.25 } else { 0.75 };
        for (m, &t) in tokens.iter().enumerate() {
            for e in 0..N_EMBD {
                want[t * N_EMBD + e] += w * got[g][m * N_EMBD + e];
            }
        }
    }
    for t in 0..N_TOKENS {
        let row = &want[t * N_EMBD..(t + 1) * N_EMBD];
        let mag = row.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-3);
        for e in 0..N_EMBD {
            let i = t * N_EMBD + e;
            assert!(
                (combined[i] - want[i]).abs() / mag <= 1e-3,
                "token {t} element {e}: combined {} vs summed rows {}",
                combined[i],
                want[i]
            );
        }
    }
    if std::env::var_os("ORANGU_TEST_DUMP").is_some() {
        // The two-call form of the same computation, for telling the
        // fused chain's own error from the 8-bit activation's.
        let gu = vulkan.matmul_experts(&x, N_TOKENS, N_EMBD, &[gate, up], &groups, &[]);
        let mut h = Vec::new();
        let mut ranges = Vec::new();
        let mut at = 0;
        for (gi, (e, tokens)) in groups.iter().enumerate() {
            let mut g = gu[0][gi].clone();
            crate::engine::tensor::gelu_inplace(&mut g);
            crate::engine::tensor::mul_inplace(&mut g, &gu[1][gi]);
            h.extend_from_slice(&g);
            ranges.push((*e, (at..at + tokens.len()).collect::<Vec<_>>()));
            at += tokens.len();
        }
        let total = at;
        let two = vulkan
            .matmul_experts(&h, total, N_FF, &[down], &ranges, &[])
            .pop()
            .unwrap();
        for (gi, (expert, tokens)) in groups.iter().enumerate() {
            for t in 0..tokens.len() {
                let a = &got[gi][t * N_EMBD..(t + 1) * N_EMBD];
                let b = &two[gi][t * N_EMBD..(t + 1) * N_EMBD];
                let mag = b.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-3);
                let worst = a
                    .iter()
                    .zip(b)
                    .map(|(x, y)| (x - y).abs() / mag)
                    .fold(0.0f32, f32::max);
                if worst > 1e-2 {
                    eprintln!("fused vs two-call: expert {expert} member {t}: worst {worst:.3e}");
                }
            }
        }
    }
    for ((expert, tokens), result) in groups.iter().zip(&got) {
        let mut gathered = Vec::with_capacity(tokens.len() * N_EMBD);
        for &t in tokens {
            gathered.extend_from_slice(&x[t * N_EMBD..(t + 1) * N_EMBD]);
        }
        let wg = gu_stack.rows(expert * 2 * N_FF, N_FF);
        let wu = gu_stack.rows(expert * 2 * N_FF + N_FF, N_FF);
        let wd = dn_stack.rows(expert * N_EMBD, N_EMBD);
        let mut g = CpuBackend.matmul_dequant(&gathered, tokens.len(), &wg);
        let u = CpuBackend.matmul_dequant(&gathered, tokens.len(), &wu);
        crate::engine::tensor::gelu_inplace(&mut g);
        crate::engine::tensor::mul_inplace(&mut g, &u);
        let want = CpuBackend.matmul_dequant(&g, tokens.len(), &wd);
        assert_eq!(result.len(), want.len(), "expert {expert}");
        if std::env::var_os("ORANGU_TEST_DUMP").is_some() {
            for t in 0..tokens.len() {
                let row = &want[t * N_EMBD..(t + 1) * N_EMBD];
                let mag = row.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-3);
                let worst = (0..N_EMBD)
                    .map(|o| (result[t * N_EMBD + o] - want[t * N_EMBD + o]).abs() / mag)
                    .fold(0.0f32, f32::max);
                eprintln!(
                    "expert {expert} member {t} (token {}): worst {worst:.3e}",
                    tokens[t]
                );
            }
        }
        for t in 0..tokens.len() {
            let row = &want[t * N_EMBD..(t + 1) * N_EMBD];
            let mag = row.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-3);
            for o in 0..N_EMBD {
                let i = t * N_EMBD + o;
                assert!(
                    (result[i] - want[i]).abs() / mag <= 5e-2,
                    "expert {expert} member {t} row {o}: fused {} vs cpu {} (row magnitude {mag})",
                    result[i],
                    want[i]
                );
            }
        }
    }
}

/// What one host-to-device upload of an expert stack costs by each route
/// this backend could take, so the streaming path's upload can be judged
/// against the bus rather than against itself: `queue.write_buffer` (the
/// staging belt: one memcpy on the calling thread, then a copy the driver
/// enqueues ahead of the next submission), a persistently mapped staging
/// buffer filled by every core and copied by one `copy_buffer_to_buffer`,
/// and the same with the fill left out (the copy alone, which is the DMA
/// engine's rate over the link).
#[test]
#[ignore]
fn _scratch_measure_upload_routes() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    use rayon::prelude::*;
    let device = &vulkan.device;
    let bytes: u64 = 256 * 1024 * 1024;
    let src: Vec<u8> = (0..bytes).map(|i| (i % 251) as u8).collect();
    let dst = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe dst"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let wait = |label: &str, t: std::time::Instant| {
        vulkan.poll_blocking_with(wgpu::PollType::wait_indefinitely(), "upload probe");
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "  {label:<44} {ms:8.1} ms  {:6.2} GB/s",
            bytes as f64 / 1e9 / (ms / 1000.0)
        );
    };
    for round in 0..3 {
        eprintln!(" round {round}");
        // 1. write_buffer: one call for the whole stack.
        let t = std::time::Instant::now();
        vulkan.queue.write_buffer(&dst, 0, &src);
        let stage_ms = t.elapsed().as_secs_f64() * 1000.0;
        vulkan.queue.submit(std::iter::empty());
        wait(&format!("write_buffer (staging {stage_ms:.1} ms)"), t);

        // 2. write_buffer in 4 MiB pieces from four threads.
        let t = std::time::Instant::now();
        let piece = 4 * 1024 * 1024;
        src.par_chunks(piece).enumerate().for_each(|(i, chunk)| {
            vulkan.queue.write_buffer(&dst, (i * piece) as u64, chunk);
        });
        let stage_ms = t.elapsed().as_secs_f64() * 1000.0;
        vulkan.queue.submit(std::iter::empty());
        wait(
            &format!(
                "write_buffer x{} threads (staging {stage_ms:.1} ms)",
                rayon::current_num_threads()
            ),
            t,
        );

        // 3. Mapped staging buffer, filled in parallel, one copy.
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("probe staging"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });
        let t = std::time::Instant::now();
        // A write-only mapped range splits into pieces one per thread. The
        // piece type is not `Send` for an unsized element (its bound is on
        // `T: Sized`), so the pieces cross threads in a wrapper: each piece
        // is a disjoint range of one mapping, written by one thread.
        struct Piece<'a>(wgpu::WriteOnly<'a, [u8]>);
        unsafe impl Send for Piece<'_> {}
        let fill = |view: wgpu::WriteOnly<'_, [u8]>| {
            let mut pieces = Vec::new();
            let mut rest = view;
            while rest.len() > piece {
                let (head, tail) = rest.split_at(piece);
                pieces.push(Piece(head));
                rest = tail;
            }
            pieces.push(Piece(rest));
            pieces
                .into_par_iter()
                .zip(src.par_chunks(piece))
                .for_each(|(mut d, s)| d.0.copy_from_slice(s));
        };
        {
            let mut view = staging.slice(..).get_mapped_range_mut().expect("mapped");
            fill(view.slice(..));
        }
        staging.unmap();
        let fill_ms = t.elapsed().as_secs_f64() * 1000.0;
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&staging, 0, &dst, 0, bytes);
        vulkan.queue.submit([encoder.finish()]);
        wait(
            &format!("mapped staging, parallel fill ({fill_ms:.1} ms) + copy"),
            t,
        );

        // 4. The copy alone, staging already filled.
        let t = std::time::Instant::now();
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&staging, 0, &dst, 0, bytes);
        vulkan.queue.submit([encoder.finish()]);
        wait("copy_buffer_to_buffer alone", t);

        // 5. Re-map and fill again (the reuse case: map_async + wait).
        let t = std::time::Instant::now();
        let mw = super::MapWait::new();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Write, mw.callback());
        vulkan.wait_mapped(&mw, "probe remap");
        {
            let mut view = staging.slice(..).get_mapped_range_mut().expect("mapped");
            fill(view.slice(..));
        }
        staging.unmap();
        let fill_ms = t.elapsed().as_secs_f64() * 1000.0;
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&staging, 0, &dst, 0, bytes);
        vulkan.queue.submit([encoder.finish()]);
        wait(
            &format!("remapped staging, parallel fill ({fill_ms:.1} ms) + copy"),
            t,
        );
    }
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

/// `Q5_K` at the shape a model actually has, because the light kernel is
/// where an indexing mistake would hide.
///
/// The other `Q5_K` checks run `in_dim` 512 or 1536 against `out_dim` 5 or
/// 40 — two or six super-blocks per row, and fewer output rows than one
/// workgroup covers. The light kernel's thread mapping folds `tid` through
/// `itid`/`il`/`ir`/`v_im`/`v_in` into byte offsets within a 176-byte
/// block, and its block loop strides by two; an error in either could
/// still come out right on a couple of blocks and a handful of rows and be
/// wrong on 8 blocks × 2048 rows, which is what an `ffn_down` is. So:
/// `in_dim = 2048` (8 super-blocks), `out_dim = 2048`.
///
/// Prompted by the measurement rather than by suspicion — the kernel came
/// out 78% faster than the block-unroll it replaced, which is a large
/// enough jump to be worth ruling out "it is fast because it is skipping
/// something" before believing it.
#[test]
fn matmul_matches_cpu_backend_for_q5_k_model_shaped() {
    cross_check_n_tokens(GGML_TYPE_Q5_K, 2048, 2048, 1);
}

/// The `Q6_K` twin, and it carries more weight than the `Q5_K` one: this
/// format's 210-byte block is **not 4-byte aligned**, so the light kernel
/// reads every weight word through the two-load unaligned path and the
/// alignment *alternates with the block index*. A kernel that got that
/// wrong would still be right on the even blocks. `in_dim = 2048` is eight
/// super-blocks per row, so both parities are exercised many times over,
/// and 2048 output rows put every one of the 16 thread mappings against
/// every parity.
#[test]
fn matmul_matches_cpu_backend_for_q6_k_model_shaped() {
    cross_check_n_tokens(GGML_TYPE_Q6_K, 2048, 2048, 1);
}

#[test]
fn matmul_matches_cpu_backend_for_q5_k() {
    cross_check(GGML_TYPE_Q5_K, 512, 5);
}

#[test]
fn matmul_matches_cpu_backend_for_q6_k() {
    cross_check(GGML_TYPE_Q6_K, 512, 5);
}

/// A weight tensor whose byte length is not a multiple of 4.
///
/// `Q6_K` blocks are 210 bytes, so a single-super-block row against an odd
/// row count gives 1,050 bytes — even, and not a multiple of 4. Uploading
/// that is rejected outright (`Copy size N does not respect
/// COPY_BUFFER_ALIGNMENT`), which is a *panic*, not a wrong number, and it
/// takes the whole server down on the first token. The shapes above all
/// happen to land on 4 and so never saw it; a vocabulary with an odd token
/// count and a `Q6_K` `output.weight` does, and nothing about that file is
/// malformed.
#[test]
fn matmul_uploads_a_weight_whose_length_is_not_a_multiple_of_four() {
    assert_eq!(210 * 5 % 4, 2, "this shape has to be the awkward one");
    cross_check(GGML_TYPE_Q6_K, 256, 5);
}

/// `n_tokens = 130` (> 64, so this needs 3 tiles of the cooperative
/// path's internal token-tiling loop — 64 + 64 + a final, only
/// partially-active tile of 2 — not just the first) against every
/// type, exercising whichever cooperative-path kernel `VulkanBackend::
/// tiled_prefill` currently selects (`shader_source_coop_tiled`/
/// `MAIN_COOP_TILED_SUFFIX` by default; `shader_source_coop`/
/// `MAIN_COOP_SUFFIX` under `ORANGU_NO_TILED_PREFILL=1` — `shared_
/// vulkan`'s one-`VulkanBackend`-per-process design means a given test
/// run only ever exercises one of the two, whichever the environment
/// selected at first construction) for real: `cross_check`'s own
/// `n_tokens = 3` never reaches either.
/// The K-quant and `IQ*` types added for mixed "dynamic" releases, on
/// the per-token reduce path.
///
/// `engine::quant`'s own fixture already holds the CPU dequantizers to
/// ggml bit-for-bit, so the CPU side of this comparison is ground truth
/// rather than a second opinion — what these check is the WGSL
/// restatement of each `dequant_element` as a function of `k`, which is
/// where a K-quant's four-pass `shift` walk or an `IQ*` type's
/// index/sign/scale decomposition can go wrong independently of the
/// Rust.
#[test]
fn matmul_matches_cpu_backend_for_q2_k() {
    cross_check(GGML_TYPE_Q2_K, 512, 5);
}

#[test]
fn matmul_matches_cpu_backend_for_q3_k() {
    cross_check(GGML_TYPE_Q3_K, 512, 5);
}

#[test]
fn matmul_matches_cpu_backend_for_iq2_xs() {
    cross_check(GGML_TYPE_IQ2_XS, 512, 5);
}

/// The three types this backend gained a shader for at once, and the
/// reason the `IQ*` codebook buffer grew from ~15 KiB to ~33 KiB: they
/// are what a `UD`-style 2-bit release is actually made of
/// (`unsloth/Qwen3.8-27B-GGUF:IQ2_XXS` is 96 `IQ1_M` tensors and 48
/// `IQ2_XXS` ones), and until they existed such a file could not use a
/// GPU at all — `engine::backend::unsupported_tensor_types` rejected the
/// whole model up front.
///
/// `IQ1_S` and `IQ1_M` are the only quantizations here whose codebook
/// values are **signed** and which carry no sign field, so a `±delta` on
/// each weight is the whole per-group freedom. Reading the grid byte as
/// unsigned — the shape every `iq2*`/`iq3*` shader above uses — is a
/// mistake that stays well formed and produces plausible output, which
/// is what this catches.
#[test]
fn matmul_matches_cpu_backend_for_iq2_xxs() {
    cross_check(GGML_TYPE_IQ2_XXS, 512, 5);
}

#[test]
fn matmul_matches_cpu_backend_for_iq1_s() {
    cross_check(GGML_TYPE_IQ1_S, 512, 5);
}

#[test]
fn matmul_matches_cpu_backend_for_iq1_m() {
    cross_check(GGML_TYPE_IQ1_M, 512, 5);
}

#[test]
fn matmul_matches_cpu_backend_for_iq2_s() {
    cross_check(GGML_TYPE_IQ2_S, 512, 5);
}

#[test]
fn matmul_matches_cpu_backend_for_iq3_xxs() {
    cross_check(GGML_TYPE_IQ3_XXS, 512, 5);
}

#[test]
fn matmul_matches_cpu_backend_for_iq3_s() {
    cross_check(GGML_TYPE_IQ3_S, 512, 5);
}

#[test]
fn matmul_matches_cpu_backend_for_iq4_xs() {
    cross_check(GGML_TYPE_IQ4_XS, 512, 5);
}

/// `in_dim = 896`, not the 512 its siblings use, because 896 is
/// deliberately *not* a multiple of 256: a row this wide is the reason
/// `IQ4_NL` appears in these files at all (upstream substitutes it where
/// a K-quant's 256-element block won't divide the row), so it is the
/// shape a wrong `QK_K` assumption anywhere in the block-offset math
/// would fail on and 512 would not.
#[test]
fn matmul_matches_cpu_backend_for_iq4_nl() {
    cross_check(GGML_TYPE_IQ4_NL, 896, 5);
}

/// `MXFP4` on the decode (block-hoisted) path.
///
/// `in_dim = 896` for `IQ4_NL`'s reason above — not a multiple of 256, so a
/// stray `QK_K` assumption fails here — and because `MXFP4`'s block is the
/// only **odd** byte size in the tree (17). A kernel that assumed blocks
/// were 4-aligned would read every block after the first at a shifted
/// offset, which is a wrong answer rather than a crash: `read_u8` peels
/// bytes out of an `array<u32>` and will happily return the wrong one.
#[test]
fn matmul_matches_cpu_backend_for_mxfp4() {
    cross_check(GGML_TYPE_MXFP4, 896, 5);
}

/// Prism's two ternary types on the decode path — `in_dim = 5120`, the
/// served 27B's own width, is a multiple of 128 but **not** of 256, and
/// `PQ2_0`'s 34-byte block puts every other block at a half-word offset
/// while `PTQ1_0`'s 28-byte one is always aligned; both readers are
/// exercised. `out_dim = 5` keeps a trailing row for the multi-row
/// block-hoisted kernel's tail.
#[test]
fn matmul_matches_cpu_backend_for_pq2_0() {
    cross_check(GGML_TYPE_PQ2_0, 5120, 5);
    cross_check(GGML_TYPE_PQ2_0, 128, 3);
    // Three blocks: a row of 102 bytes, so odd rows start at a half word.
    cross_check(GGML_TYPE_PQ2_0, 384, 2048);
    cross_check(GGML_TYPE_PQ2_0, 2048, 2055);
    // Wide enough for the four-row block-hoisted variant.
    cross_check(GGML_TYPE_PQ2_0, 1024, 2048);
}

#[test]
fn matmul_matches_cpu_backend_for_ptq1_0() {
    cross_check(GGML_TYPE_PTQ1_0, 5120, 5);
    cross_check(GGML_TYPE_PTQ1_0, 128, 3);
    // Wide enough for the four-row kernel — on this device the integer-dot
    // one. `896` is seven blocks: the last block iteration has a masked
    // slot, and the `qh` lane's spare activation loads once reached into
    // the next block's elements and skewed the block maximum here.
    cross_check(GGML_TYPE_PTQ1_0, 1024, 2048);
    cross_check_n_tokens(GGML_TYPE_PTQ1_0, 896, 2048, 1);
    // A row count that is not a multiple of the workgroup's rows (the
    // start-up self-check's own wide case): the tail workgroup's rows.
    cross_check(GGML_TYPE_PTQ1_0, 2048, 2055);
}

/// A word-reading type through the generic batched matmul at a width that
/// takes the four-row block-hoisted kernel. `matmul_batch_dispatch_streamed`
/// used to bind the one-row pipeline to a grid sized for four rows, and
/// every row from `out_dim / 4` up came back zero — for every words type,
/// on every architecture that does not go through a fused chain. `Q2_K` is
/// the one that has been in the tree longest; the `PQ2_0`/`PTQ1_0` checks
/// above cover the same shape.
#[test]
fn a_wide_words_type_projection_is_computed_in_full_by_the_batched_matmul() {
    cross_check_n_tokens(GGML_TYPE_Q2_K, 1024, 2048, 1);
    cross_check(GGML_TYPE_Q2_K, 1024, 2048);
}

/// The cooperative-tiled (prefill) path over the same two types.
#[test]
fn matmul_matches_cpu_backend_cooperative_path_pq2_0() {
    cross_check_n_tokens(GGML_TYPE_PQ2_0, 1024, 64, 130);
}

#[test]
fn matmul_matches_cpu_backend_cooperative_path_ptq1_0() {
    cross_check_n_tokens(GGML_TYPE_PTQ1_0, 1024, 64, 130);
}

/// The `e8m0` exponent decode across its whole range.
///
/// `cross_check`'s generated blocks deliberately bound the exponent near
/// unity so a dot product does not overflow, which means they never reach the
/// ends of the range. So this checks every code directly against
/// `quant::dequantize` — one block per code, comparing the dequantized
/// weights themselves rather than a matmul over them.
///
/// # Codes 0 and 1 are exempt, and the exemption is the finding
///
/// `MXFP4` is the only type here whose scale can be **subnormal in f32**:
/// code 0 decodes to `2^-128` and code 1 to `2^-127`, both below f32's
/// smallest normal `2^-126`. The device flushes subnormals to zero, the host
/// does not, and they disagree — measured, not assumed: every other code
/// (2..=255) is bit-exact on both paths, and only these two differ.
///
/// This is a property of the hardware, not a transcription error —
/// `mxfp4_scale` produces the same *bits* the host does; what differs is what
/// arithmetic on those bits then yields. It is left as a divergence rather
/// than papered over on either side, because the alternatives are worse:
/// flushing on the host would corrupt the reference every other test is
/// judged against, and no portable knob asks a device to keep subnormals.
///
/// It cannot affect a generated token. The largest weight either code can
/// express is `2^-127 * 12`, about `7e-38`. Against a dot product over
/// hundreds of `O(1)` terms, that is ~30 orders of magnitude below the f32
/// epsilon of the running sum — it is not a small contribution, it is one
/// that cannot change any bit of the result it is added to. A block scaled
/// this way encodes weights that are zero in every sense that matters, which
/// is what such a code means in the format.
#[test]
fn mxfp4_scale_decode_matches_the_host_over_every_exponent_code() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    // One row per exponent code. The nibbles are fixed to the identity
    // pattern `j | (j << 4)`, so every codebook entry appears in every row
    // and a wrong scale shows up on all 32 elements rather than some.
    let mut bytes = Vec::new();
    for e in 0..=255u16 {
        bytes.push(e as u8);
        if e <= 252 {
            bytes.extend((0..16u8).map(|j| j | (j << 4)));
        } else {
            // **The top three codes carry only the entries worth ±1, and
            // that is the whole portability story.**
            //
            // The codebook reaches 12 and the scale is `2^(e - 128)`, so from
            // code 253 up a full-codebook row's largest product passes
            // `f32::MAX` and the row holds `±inf`. Such a row cannot be read
            // out by a one-hot activation at all: the reference sums
            // `±inf * 0.0` over the 31 elements the activation zeroes, which
            // is `NaN` under IEEE and simply vanishes under the fast-math a
            // Metal backend compiles with by default, leaving the true
            // element behind. Both are defensible, which is precisely what
            // makes such a row useless for deciding whether the *scale*
            // decoded correctly.
            //
            // So these rows carry `KVALUES_MXFP4[1]` and `[9]`, exactly `1`
            // and `-1`, whose product with even `2^127` stays finite. All 256
            // exponent codes are still checked, and nothing in this test
            // depends on how a device treats `inf * 0`.
            bytes.extend((0..16u8).map(|_| 0x91u8));
        }
    }
    let w = test_quant_matrix(&bytes, GGML_TYPE_MXFP4, 32, 256);
    // A one-hot activation reads out element `k` of each row unchanged, so
    // the matmul is a dequantization with the accumulation removed — no
    // summation order to reconcile, and a mismatch names the element.
    for k in 0..32usize {
        let mut x = vec![0f32; 32];
        x[k] = 1.0;
        let gpu = vulkan.matmul(&x, 1, &w);
        let cpu = CpuBackend.matmul_dequant(&x, 1, &w);
        for (e, (g, c)) in gpu.iter().zip(&cpu).enumerate() {
            if e < 2 {
                // The subnormal pair. This used to assert flush-to-zero
                // outright, so that a device which *does* keep subnormals
                // would fail here and the claim get revisited rather than
                // quietly become untrue. That happened: a Mali-G720 returns
                // the subnormal. Vulkan leaves denormal handling to the
                // implementation unless `shaderDenormFlushToZeroFloat32` or
                // `shaderDenormPreserveFloat32` is requested, and this
                // backend requests neither — so *both* answers are correct
                // and the test now accepts either. What it still refuses is
                // a third answer: some other number entirely.
                assert!(
                    *g == 0.0 || (g - c).abs() <= 1e-30,
                    "exponent code {e} is subnormal, so element {k} must come back either \
                     flushed to zero or equal to the host's {c:e} — got {g:e}"
                );
                // Element 1's codebook entry is exactly 1, so the host value
                // there *is* the scale — the one place to check that the
                // scale itself is subnormal. Not checked at other `k`: the
                // codebook multiplies up to 12, which lifts some products
                // back into the normal range even from a subnormal scale
                // (code 1, element 2 is `1.175e-38`, exactly `f32::MIN_
                // POSITIVE`). The device still returns zero for those,
                // because it flushed `d` before the multiply.
                if k == 1 {
                    assert!(
                        c.abs() < f32::MIN_POSITIVE,
                        "exponent code {e} was expected to decode to a subnormal \
                         scale on the host, but it is {c:e}"
                    );
                }
                continue;
            }
            // Every product is finite by construction now, so there is no
            // device-dependent case to allow and nothing to skip.
            assert!(
                c.is_finite(),
                "exponent code {e}, element {k}: reference is {c:e}, but these \
                 rows are built so that every product stays finite"
            );
            assert!(
                g == c || (g - c).abs() <= c.abs() * 1e-6,
                "exponent code {e}, element {k}: gpu {g:e} != cpu {c:e}"
            );
        }
    }
}

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
fn matmul_matches_cpu_backend_cooperative_path_q4_1() {
    cross_check_n_tokens(GGML_TYPE_Q4_1, 64, 17, 130);
}

#[test]
fn matmul_matches_cpu_backend_cooperative_path_q5_1() {
    cross_check_n_tokens(GGML_TYPE_Q5_1, 64, 17, 130);
}

#[test]
fn matmul_matches_cpu_backend_cooperative_path_q8_0() {
    cross_check_n_tokens(GGML_TYPE_Q8_0, 64, 17, 130);
}

#[test]
fn matmul_matches_cpu_backend_cooperative_path_q4_k() {
    cross_check_n_tokens(GGML_TYPE_Q4_K, 512, 5, 130);
}

/// The same seven types through the cooperative/tiled kernel, whose
/// `fill_w_run` reaches `dequant_element` via the generic per-element
/// fallback rather than one of the `Q4_K`/`Q5_K`/`Q6_K` specializations
/// — a different call path over the same function, and the one the model
/// actually takes during prefill.
#[test]
fn matmul_matches_cpu_backend_cooperative_path_q2_k() {
    cross_check_n_tokens(GGML_TYPE_Q2_K, 512, 5, 130);
}

#[test]
fn matmul_matches_cpu_backend_cooperative_path_q3_k() {
    cross_check_n_tokens(GGML_TYPE_Q3_K, 512, 5, 130);
}

#[test]
fn matmul_matches_cpu_backend_cooperative_path_iq2_xs() {
    cross_check_n_tokens(GGML_TYPE_IQ2_XS, 512, 5, 130);
}

#[test]
fn matmul_matches_cpu_backend_cooperative_path_iq2_s() {
    cross_check_n_tokens(GGML_TYPE_IQ2_S, 512, 5, 130);
}

#[test]
fn matmul_matches_cpu_backend_cooperative_path_iq3_xxs() {
    cross_check_n_tokens(GGML_TYPE_IQ3_XXS, 512, 5, 130);
}

/// The three new types on the *prefill* side too. Their per-token
/// (`block_dot`) and prefill (`dequant_element`) restatements of the same
/// layout are written separately and can disagree with each other while
/// each looks right on its own, so both are checked against the same CPU
/// ground truth.
#[test]
fn matmul_matches_cpu_backend_cooperative_path_iq2_xxs() {
    cross_check_n_tokens(GGML_TYPE_IQ2_XXS, 512, 5, 130);
}

#[test]
fn matmul_matches_cpu_backend_cooperative_path_iq1_s() {
    cross_check_n_tokens(GGML_TYPE_IQ1_S, 512, 5, 130);
}

#[test]
fn matmul_matches_cpu_backend_cooperative_path_iq1_m() {
    cross_check_n_tokens(GGML_TYPE_IQ1_M, 512, 5, 130);
}

#[test]
fn matmul_matches_cpu_backend_cooperative_path_iq3_s() {
    cross_check_n_tokens(GGML_TYPE_IQ3_S, 512, 5, 130);
}

#[test]
fn matmul_matches_cpu_backend_cooperative_path_iq4_xs() {
    cross_check_n_tokens(GGML_TYPE_IQ4_XS, 512, 5, 130);
}

/// The cooperative-tiled path over an `IQ4_NL` weight at a real row
/// width, with `out_dim` past one 32-row output tile so the staged
/// `fill_w_run` is exercised on more than a partial tile.
#[test]
fn matmul_matches_cpu_backend_cooperative_path_iq4_nl() {
    cross_check_n_tokens(GGML_TYPE_IQ4_NL, 896, 64, 130);
}

/// The cooperative-tiled (prefill) path over an `MXFP4` weight — the same
/// shape as the `IQ4_NL` case above, which is the type `MXFP4` shares its
/// nibble layout with.
#[test]
fn matmul_matches_cpu_backend_cooperative_path_mxfp4() {
    cross_check_n_tokens(GGML_TYPE_MXFP4, 896, 64, 130);
}

/// The cooperative tiled kernel past its own `COOP_TILE_ROWS = 32` output
/// tile. Every other cooperative cross-check uses `out_dim = 5`, so none of
/// them exercises more than a partial first row-tile — while the real
/// model's `out_dim` is 1536–6144.
#[test]
fn matmul_matches_cpu_backend_cooperative_path_multi_row_tile() {
    cross_check_n_tokens(GGML_TYPE_Q4_K, 512, 64, 130);
}

/// Replays **real GGUF weights** through the tiled GEMM and diffs against
/// the CPU backend. The synthetic cross-checks above are exact at every
/// shape, quant type and token count the model uses, yet the model itself
/// returns garbage above the tiled crossover — so the remaining variable is
/// the weight data, and this is the one experiment that settles it.
///
/// Result (2026-07-25, gemma-4-E2B-it-Q4_K_M): **60 real tensors across 12
/// layers, all exact** — `gpu_absum / cpu_absum = 1.0000`, worst relative
/// error ~1e-4, nothing over 1e-2. So the weight data is not the variable,
/// and neither is a warm shared backend with hundreds of cached ops.
///
/// It also covers the `IQ*` family on real weights, which is the only
/// place they are checked outside synthetic blocks. `unsloth/
/// Qwen3.8-27B-GGUF:IQ2_XXS` (`IQ1_M`, `IQ2_S`, `IQ2_XXS`, `IQ2_XS`,
/// `IQ3_XXS`, `IQ4_XS`, `Q2_K`, `Q3_K`) and its `Q3_K_XL` sibling
/// (`IQ4_XS`, `IQ3_S`, `Q5_K`, `Q3_K`): **42 tensors each, both widths,
/// 0 bad.**
///
/// `#[ignore]` because it needs the model file: run with
/// `ORANGU_PROBE_GGUF=/path/to/model.gguf cargo test real_gguf_weights -- --ignored --nocapture`.
#[test]
#[ignore = "needs a real GGUF; run with --ignored"]
fn real_gguf_weights_match_the_cpu_backend() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        return;
    };
    let path = std::env::var("ORANGU_PROBE_GGUF").expect("set ORANGU_PROBE_GGUF");
    let model =
        crate::engine::loader::LoadedModel::open(std::path::Path::new(&path)).expect("open gguf");
    let mut seed = 0x1234_5678_u64;
    let mut names: Vec<String> = Vec::new();
    for l in 0..12 {
        for t in ["attn_q", "attn_output", "ffn_gate", "ffn_up", "ffn_down"] {
            names.push(format!("blk.{l}.{t}.weight"));
        }
    }
    let mut bad = 0usize;
    for name in names.iter().map(String::as_str) {
        let Ok(w) = model.matrix(name) else {
            eprintln!("  {name}: not present");
            continue;
        };
        // Both sides of the `coop_min_n_tokens` crossover. `91` is the
        // tiled prefill path this test was written for; `1` is the decode
        // matmul-vec, which is a *different kernel per quantization*
        // (`pipeline_for_named`) and so a separate restatement of the
        // same block layout — the two can disagree with each other while
        // each looks right on synthetic blocks.
        for nt in [1usize, 91usize] {
            let mut x = vec![0f32; nt * w.in_dim];
            for v in x.iter_mut() {
                *v = (next_byte(&mut seed) as f32 - 128.0) / 512.0;
            }
            let cpu = CpuBackend.matmul_dequant(&x, nt, &w);
            let gpu = vulkan.matmul(&x, nt, &w);
            let ca: f64 = cpu.iter().map(|v| v.abs() as f64).sum();
            let ga: f64 = gpu.iter().map(|v| v.abs() as f64).sum();
            let mut worst = 0f32;
            let mut over = 0usize;
            for (a, b) in cpu.iter().zip(gpu.iter()) {
                let rel = (a - b).abs() / a.abs().max(1e-3);
                if rel > worst {
                    worst = rel;
                }
                if rel > 1e-2 {
                    over += 1;
                }
            }
            if over > 0 || !(0.999..1.001).contains(&(ga / ca)) {
                bad += 1;
                eprintln!(
                    "  BAD {name} type {} {}x{} nt={nt}: ratio {:.4} worst_rel {worst:.3e} over_1e-2 {over}/{}",
                    w.ggml_type(),
                    w.in_dim,
                    w.out_dim,
                    ga / ca,
                    gpu.len()
                );
            }
        }
    }
    eprintln!("  checked {} tensors, {bad} bad", names.len());
}

/// A host layer's projections on **real weights and real activations**, the
/// streamed card product beside the CPU's own, each against the float
/// reference — so a split model's streamed prefill can be judged by the
/// error it adds rather than by whether a greedy continuation still agrees
/// with the host's (on a 12B model it stops agreeing after eight tokens
/// either way, and that says nothing about which arm is closer to the
/// truth).
///
/// `ORANGU_PROBE_GGUF` names the model, `ORANGU_PROBE_LAYER` the layer
/// (default 18, the first host layer of the 12B split on a 4 GiB card) and
/// `ORANGU_PROBE_ACT` a row-major `f32` activation file for that layer's
/// width, such as an `ORANGU_NPU_DUMP_ACT` capture's `blk.<layer>.<n>.f32`
/// (random rows when unset) — the feed-forward's input stands in for the
/// attention's, a normed residual of the same width. Reported per projection: the relative RMS
/// error of each kernel against the dequantized product, and the two
/// kernels' distance from each other.
///
/// `ORANGU_PROBE_GGUF=... ORANGU_PROBE_ACT=... cargo test real_gguf_host_layer_streamed -- --ignored --nocapture`
#[test]
#[ignore = "needs a real GGUF; run with --ignored"]
fn real_gguf_host_layer_streamed_matches_the_cpu_kernel() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        return;
    };
    let path = std::env::var("ORANGU_PROBE_GGUF").expect("set ORANGU_PROBE_GGUF");
    let layer: usize = std::env::var("ORANGU_PROBE_LAYER")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(18);
    let model =
        crate::engine::loader::LoadedModel::open(std::path::Path::new(&path)).expect("open gguf");
    let rel_rms = |a: &[f32], b: &[f32]| -> f64 {
        let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
        let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
        (num / den.max(1e-30)).sqrt()
    };
    let mut seed = 0x9A7E_5EED_u64;
    for name in ["attn_q", "attn_k", "attn_v", "ffn_gate", "ffn_up"] {
        let w = model
            .matrix(&format!("blk.{layer}.{name}.weight"))
            .expect("the layer's projection");
        let x: Vec<f32> = match std::env::var("ORANGU_PROBE_ACT") {
            Ok(file) => {
                let bytes = std::fs::read(&file).expect("read the activation file");
                let all: Vec<f32> = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect();
                let rows = (all.len() / w.in_dim).min(512);
                all[..rows * w.in_dim].to_vec()
            }
            _ => (0..512 * w.in_dim)
                .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 512.0)
                .collect(),
        };
        let n_tokens = x.len() / w.in_dim;
        let reference = CpuBackend.matmul_dequant(&x, n_tokens, &w);
        let cpu = CpuBackend.matmul(&x, n_tokens, &w);
        let streamed = vulkan
            .matmul_batch_streamed(&[MatmulOp {
                x: &x,
                n_tokens,
                w: &w,
            }])
            .pop()
            .expect("one result");
        eprintln!(
            "  blk.{layer}.{name} type {} {}x{} at {n_tokens} tokens: cpu {:.3e}  streamed {:.3e}  cpu-vs-streamed {:.3e}",
            w.ggml_type(),
            w.in_dim,
            w.out_dim,
            rel_rms(&cpu, &reference),
            rel_rms(&streamed, &reference),
            rel_rms(&streamed, &cpu)
        );
    }
}

/// The same real-weight cross-check for **stacked routed-expert** tensors,
/// which [`real_gguf_weights_match_the_cpu_backend`] cannot reach.
///
/// That test walks `blk.N.ffn_gate.weight`-style names through
/// `LoadedModel::matrix`. A mixture-of-experts file has no such tensor: its
/// experts live in one stacked `blk.N.ffn_gate_exps.weight` that only
/// `LoadedModel::expert_matrix` can open, and a single expert's rows are a
/// sub-range of it. So every expert weight in the tree has been checked on
/// synthetic blocks and none on real ones.
///
/// This matters most for the quantizations that *only* appear on expert
/// tensors. `MXFP4` is the case in point — the files carrying it store it as
/// `ffn_{gate,down,up}_exps` (routed) or `ffn_{gate,up}_shexp` (shared), and
/// both are exactly the tensors `arch::gpu_project_expert` and
/// `arch::matmul_host_fallback` gate on `Backend::supports_type`. Before a
/// kernel exists they go to the host; after one does, this is what says the
/// kernel is right on the weights a model actually ships.
///
/// Probes expert 0 of each stack it finds, at both sides of the prefill
/// crossover, and reports the quantization so a run's output says which
/// types were actually exercised rather than implying all of them.
///
/// `#[ignore]` because it needs the model file: run with
/// `ORANGU_PROBE_GGUF=/path/to/moe.gguf cargo test real_gguf_expert_weights -- --ignored --nocapture`.
#[test]
#[ignore = "needs a real MoE GGUF; run with --ignored"]
fn real_gguf_expert_weights_match_the_cpu_backend() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        return;
    };
    let path = std::env::var("ORANGU_PROBE_GGUF").expect("set ORANGU_PROBE_GGUF");
    let model =
        crate::engine::loader::LoadedModel::open(std::path::Path::new(&path)).expect("open gguf");
    let mut seed = 0x0FEE_1234_u64;
    let mut checked = 0usize;
    let mut bad = 0usize;
    let mut seen: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for l in 0..12 {
        for t in [
            "ffn_gate_exps",
            "ffn_up_exps",
            "ffn_down_exps",
            "ffn_gate_shexp",
            "ffn_up_shexp",
            "ffn_down_shexp",
        ] {
            let name = format!("blk.{l}.{t}.weight");
            // A shared expert is an ordinary matrix; a routed stack is not.
            // Try the stack first and fall back, so one loop covers both.
            let w = match model.expert_matrix(&name) {
                Ok(stack) => stack.expert_matrix(0),
                Err(_) => match model.matrix(&name) {
                    Ok(m) => m,
                    Err(_) => continue,
                },
            };
            let ty = orangu::gguf::ggml_type_name(w.ggml_type());
            *seen.entry(ty.clone()).or_default() += 1;
            checked += 1;
            if !vulkan.supports_type(w.ggml_type()) {
                eprintln!("  {name}: {ty} has no GPU kernel, skipped");
                continue;
            }
            for nt in [1usize, 91usize] {
                let mut x = vec![0f32; nt * w.in_dim];
                for v in x.iter_mut() {
                    *v = (next_byte(&mut seed) as f32 - 128.0) / 512.0;
                }
                let cpu = CpuBackend.matmul_dequant(&x, nt, &w);
                let gpu = vulkan.matmul(&x, nt, &w);
                let ca: f64 = cpu.iter().map(|v| v.abs() as f64).sum();
                let ga: f64 = gpu.iter().map(|v| v.abs() as f64).sum();
                let mut worst = 0f32;
                let mut over = 0usize;
                for (a, b) in cpu.iter().zip(gpu.iter()) {
                    let rel = (a - b).abs() / a.abs().max(1e-3);
                    if rel > worst {
                        worst = rel;
                    }
                    if rel > 1e-2 {
                        over += 1;
                    }
                }
                if over > 0 || !(0.999..1.001).contains(&(ga / ca)) {
                    bad += 1;
                    eprintln!(
                        "  BAD {name} {ty} {}x{} nt={nt}: ratio {:.4} worst_rel {worst:.3e} over_1e-2 {over}/{}",
                        w.in_dim,
                        w.out_dim,
                        ga / ca,
                        gpu.len()
                    );
                }
            }
        }
    }
    eprintln!("  checked {checked} expert tensors, {bad} bad");
    for (ty, n) in &seen {
        eprintln!("    {ty}: {n}");
    }
    assert_eq!(bad, 0, "expert weights disagree between GPU and CPU");
}

/// `matmul_batch` against per-op `matmul`, **elementwise**, at the real
/// Q/K/V shapes and a prefill width above the tiled crossover.
///
/// This is the last GPU matmul path in a prefill layer without an isolated
/// cross-check at model dimensions: nothing but the Q/K/V projections uses
/// it, and it is the only one with its own striping, staging-buffer fan-out
/// and per-stripe result assembly. Compared elementwise on purpose — an
/// earlier checksum comparison over 186,368 values was flat for whatever is
/// actually wrong, which is how this path came to be set aside.
#[test]
#[ignore = "needs a real GGUF; run with --ignored"]
fn matmul_batch_matches_per_op_matmul_on_real_weights() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        return;
    };
    let path = std::env::var("ORANGU_PROBE_GGUF").expect("set ORANGU_PROBE_GGUF");
    let model =
        crate::engine::loader::LoadedModel::open(std::path::Path::new(&path)).expect("open gguf");
    // Confirmed at the time of writing: this backend really does take the
    // tiled path at these widths (`use_tiled_coop(91) == true`), so a pass
    // here is evidence about the tiled kernel and not about a fallback.
    assert!(
        vulkan.use_tiled_coop(91),
        "expected the tiled path at 91 tokens"
    );
    let mut seed = 0x51DE_51DE_u64;
    for l in [0usize, 1, 4] {
        let wq = model.matrix(&format!("blk.{l}.attn_q.weight")).expect("wq");
        let wk = model.matrix(&format!("blk.{l}.attn_k.weight")).expect("wk");
        let wv = model.matrix(&format!("blk.{l}.attn_v.weight")).expect("wv");
        for nt in [91usize, 128, 200] {
            let mut x = vec![0f32; nt * wq.in_dim];
            for v in x.iter_mut() {
                *v = (next_byte(&mut seed) as f32 - 128.0) / 512.0;
            }
            let ops = vec![
                MatmulOp {
                    x: &x,
                    n_tokens: nt,
                    w: &wq,
                },
                MatmulOp {
                    x: &x,
                    n_tokens: nt,
                    w: &wk,
                },
                MatmulOp {
                    x: &x,
                    n_tokens: nt,
                    w: &wv,
                },
            ];
            let batched = vulkan.matmul_batch(&ops);
            for (i, (w, got)) in [&wq, &wk, &wv].iter().zip(batched.iter()).enumerate() {
                let single = vulkan.matmul(&x, nt, w);
                assert_eq!(single.len(), got.len(), "layer {l} op {i} nt {nt}: length");
                let mut worst = 0f32;
                let mut worst_at = 0usize;
                for (n, (a, b)) in single.iter().zip(got.iter()).enumerate() {
                    let d = (a - b).abs();
                    if d > worst {
                        worst = d;
                        worst_at = n;
                    }
                }
                assert!(
                    worst == 0.0,
                    "layer {l} op {i} nt {nt}: matmul_batch differs from matmul by {worst} \
                         at flat index {worst_at} (row {}, col {}) — same kernel, same weights, \
                         so this is the batching path",
                    worst_at / w.out_dim,
                    worst_at % w.out_dim
                );
            }
        }
    }
}

/// A **recorded** matmul (`record_matmul`, what every fused chain uses)
/// against `Backend::matmul` (what every other cross-check uses), on the
/// same weights and activations.
///
/// This is the comparison whose absence let a live correctness bug survive
/// a whole session of testing. `record_matmul` selected its pipeline with a
/// hardcoded `n_tokens = 1` while dispatching the grid the cached entry had
/// sized for the *real* token count; above `COOP_MIN_TOKENS` that pairs the
/// decode reduce kernel with the tiled kernel's much smaller grid, so most
/// of the output was never written. Nothing caught it, because:
///
/// - `Backend::matmul` and `matmul_batch` compute their own dispatch and
///   never go through `record_matmul`, so they were genuinely exact;
/// - the fused-chain cross-checks compare a fused recording against an
///   unfused sequence **built from the same cached entries**, so both sides
///   took the same mismatched pairing and agreed with each other.
///
/// A reference that shares the suspect component cannot see the fault. This
/// one crosses the boundary: the recorded path against the non-recorded one.
///
/// The token counts deliberately straddle `COOP_MIN_TOKENS` (64), since the
/// two paths only disagree above it.
fn cross_check_recorded_matmul(ggml_type: u32, in_dim: usize, out_dim: usize, n_tokens: usize) {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    if vulkan.q4_k_mmvq {
        eprintln!("skipping: ORANGU_Q4K_MMVQ records a different dispatch");
        return;
    }
    let elems = block_elems(ggml_type);
    let mut seed = 0x2ECD_2ECD_u64;
    let mut bytes = Vec::new();
    for _ in 0..out_dim {
        for _ in 0..(in_dim / elems) {
            bytes.extend(build_block(ggml_type, &mut seed));
        }
    }
    let w = test_quant_matrix(&bytes, ggml_type, in_dim, out_dim);
    let x: Vec<f32> = (0..n_tokens * in_dim)
        .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
        .collect();

    // The recorded form dispatches the float kernel, so the backend call
    // it is held exact against must too.
    let expected = vulkan.without_prefill_mmq(|| vulkan.matmul(&x, n_tokens, &w));

    // The recorded path, driven exactly as a fused chain drives it.
    let op = MatmulOp {
        x: &x,
        n_tokens,
        w: &w,
    };
    let entry = vulkan.op_entry(&op, 0);
    let g = entry.lock().expect("op cache entry poisoned");
    vulkan
        .queue
        .write_buffer(&g.x_buffer, g.x_offset, bytemuck::cast_slice(&x));
    let out_len = n_tokens * out_dim;
    let staging = vulkan.scratch_buffer(out_len);
    let mut encoder = vulkan.new_encoder("orangu-server recorded matmul cross-check");
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("orangu-server recorded matmul cross-check pass"),
            timestamp_writes: None,
        });
        vulkan.record_matmul(&mut pass, &w, &g);
    }
    encoder.copy_buffer_to_buffer(
        &g.output_buffer,
        g.output_offset,
        &staging,
        0,
        (out_len as u64) * 4,
    );
    let got = vulkan.submit_and_readback(encoder, &staging, 0, out_len);
    drop(g);

    assert_eq!(expected.len(), got.len());
    let mut worst = 0f32;
    let mut worst_at = 0usize;
    for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
        let d = (a - b).abs();
        if d > worst {
            worst = d;
            worst_at = i;
        }
    }
    // Same kernel, same weights, same activations — the two paths differ
    // only in which buffers they use, so this is exact, not approximate.
    assert!(
        worst == 0.0,
        "ggml_type {ggml_type} {in_dim}x{out_dim} n_tokens {n_tokens}: recorded matmul \
             differs from Backend::matmul by {worst} at flat index {worst_at} \
             (token {}, out {}) — the recorded dispatch and the kernel it binds disagree",
        worst_at / out_dim,
        worst_at % out_dim
    );
}

#[test]
fn recorded_matmul_matches_backend_matmul_decode_width() {
    cross_check_recorded_matmul(GGML_TYPE_Q4_K, 512, 128, 1);
}

/// Concurrent prefill matmuls on *different* weights of the same shape
/// must not corrupt each other — the property
/// [`VulkanBackend::prefill_region_guard`] exists to provide.
///
/// With `ORANGU_POOL_PREFILL_REGIONS`, two different weights of one shape
/// deliberately share an arena region, and their op-cache entries are
/// different mutexes — so nothing but that guard serialises them. The
/// failure mode is silent: `queue.write_buffer` applies at the next
/// submit, so one request's activations can land in front of another's
/// dispatch and produce plausible, wrong numbers.
///
/// Runs at both a striped and an unstriped width; see the comment inside
/// for why only one of them can actually race.
///
/// A race is not guaranteed to reproduce on any one pass, so this runs many
/// rounds on several threads. Verified to fail with the guard removed.
#[test]
fn concurrent_prefill_matmuls_on_same_shaped_weights_do_not_corrupt_each_other() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    if vulkan.q4_k_mmvq {
        eprintln!("skipping: ORANGU_Q4K_MMVQ takes a different dispatch");
        return;
    }
    const THREADS: usize = 4;
    const ROUNDS: usize = 6;
    let (in_dim, out_dim) = (256usize, 128usize);
    let elems = block_elems(GGML_TYPE_Q4_K);
    // Both prefill entry points, because only one of them can actually
    // race. `matmul_batch_striped` (above the stripe width) fills its
    // inputs with encoder-recorded copies, which the queue orders for
    // free; `matmul_batch_dispatch` (at or below it) uses
    // `queue.write_buffer`, which lands at the next submit and is what the
    // guard has to serialise. Testing only the striped width passes even
    // with the guard removed — it was checked.
    for n_tokens in [
        max_matmul_tokens_per_submission() - 32,
        max_matmul_tokens_per_submission() + 32,
    ] {
        // Distinct weights and distinct activations per thread, so any
        // cross-talk shows up as a wrong value rather than a coincidence.
        let mut cases = Vec::new();
        for t in 0..THREADS {
            let mut seed = 0x9E37_79B9_u64.wrapping_add(t as u64 * 0x1234_5678);
            let mut bytes = Vec::new();
            for _ in 0..out_dim {
                for _ in 0..(in_dim / elems) {
                    bytes.extend(build_block(GGML_TYPE_Q4_K, &mut seed));
                }
            }
            let w = test_quant_matrix(&bytes, GGML_TYPE_Q4_K, in_dim, out_dim);
            let x: Vec<f32> = (0..n_tokens * in_dim)
                .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
                .collect();
            cases.push((w, x));
        }
        // Reference values, computed one at a time with nothing else running.
        let expected: Vec<Vec<f32>> = cases
            .iter()
            .map(|(w, x)| vulkan.matmul(x, n_tokens, w))
            .collect();

        std::thread::scope(|scope| {
            for (t, ((w, x), want)) in cases.iter().zip(expected.iter()).enumerate() {
                scope.spawn(move || {
                    for round in 0..ROUNDS {
                        let got = vulkan.matmul(x, n_tokens, w);
                        assert_eq!(
                            &got, want,
                            "thread {t} round {round}: a concurrent matmul on a different \
                             weight of the same shape changed this one's result — pooled \
                             regions are not being serialised"
                        );
                    }
                });
            }
        });
    }
}

/// Runs `run` over `cases` once, sequentially, to establish what each
/// case's answer is; then runs it again with one thread per case, several
/// rounds each, asserting every result is unchanged.
///
/// Shared by the pooled-region concurrency tests. Each case owns its own
/// same-shaped-but-different weights, so a region shared between two of
/// them shows up as a wrong number rather than as a coincidence.
fn assert_concurrent_agrees<C: Sync>(cases: &[C], run: impl Fn(&C) -> Vec<f32> + Sync, what: &str) {
    const ROUNDS: usize = 6;
    let expected: Vec<Vec<f32>> = cases.iter().map(&run).collect();
    let run = &run;
    std::thread::scope(|scope| {
        for (t, (case, want)) in cases.iter().zip(expected.iter()).enumerate() {
            scope.spawn(move || {
                for round in 0..ROUNDS {
                    let got = run(case);
                    assert_eq!(
                        got.len(),
                        want.len(),
                        "{what}: thread {t} round {round} length changed"
                    );
                    if let Some(i) = got.iter().zip(want.iter()).position(|(a, b)| a != b) {
                        panic!(
                            "{what}: thread {t} round {round} differs at {i} \
                                 ({} vs {}) — a concurrent call on same-shaped weights \
                                 changed this one's result, so pooled regions are not \
                                 being serialised",
                            got[i], want[i]
                        );
                    }
                }
            });
        }
    });
}

/// Random `Q4_K` weights of the given shape, for the concurrency tests.
fn concurrency_weight(in_dim: usize, out_dim: usize, seed: &mut u64) -> QuantMatrix {
    let mut bytes = Vec::new();
    for _ in 0..out_dim {
        for _ in 0..(in_dim / 256) {
            bytes.extend(build_block(GGML_TYPE_Q4_K, seed));
        }
    }
    test_quant_matrix(&bytes, GGML_TYPE_Q4_K, in_dim, out_dim)
}

fn concurrency_vec(n: usize, seed: &mut u64) -> Vec<f32> {
    (0..n)
        .map(|_| (next_byte(seed) as f32 - 128.0) / 64.0)
        .collect()
}

/// `fused_ffn_prefill` under concurrency — see
/// [`VulkanBackend::prefill_region_guard`].
#[test]
fn concurrent_fused_ffn_prefills_do_not_corrupt_each_other() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    if vulkan.q4_k_mmvq {
        eprintln!("skipping: ORANGU_Q4K_MMVQ selects the unfused fallback path");
        return;
    }
    let (n_embd, ffn_len, n_tokens) = (256usize, 512usize, 96usize);
    struct Case {
        gate: QuantMatrix,
        up: QuantMatrix,
        down: QuantMatrix,
        x: Vec<f32>,
    }
    let cases: Vec<Case> = (0..4)
        .map(|t| {
            let mut seed = 0xFFA1_u64.wrapping_add(t * 0x9E37_79B9);
            Case {
                gate: concurrency_weight(n_embd, ffn_len, &mut seed),
                up: concurrency_weight(n_embd, ffn_len, &mut seed),
                down: concurrency_weight(ffn_len, n_embd, &mut seed),
                x: concurrency_vec(n_tokens * n_embd, &mut seed),
            }
        })
        .collect();
    assert_concurrent_agrees(
        &cases,
        |c| {
            vulkan
                .fused_ffn_prefill(
                    &c.x,
                    n_tokens,
                    &c.gate,
                    &c.up,
                    &c.down,
                    crate::engine::backend::vulkan::FfnActivation::Geglu,
                    None,
                )
                .expect("fused FFN available without MMVQ")
        },
        "fused FFN prefill",
    );
}

/// `fused_post_attention_prefill` under concurrency — the recorder that is
/// two thirds of prefill, and the one with the most pooled ops (four).
#[test]
fn concurrent_fused_post_attention_prefills_do_not_corrupt_each_other() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    if vulkan.q4_k_mmvq {
        eprintln!("skipping: ORANGU_Q4K_MMVQ selects the unfused fallback path");
        return;
    }
    let (n_embd, attn_dim, ffn_len, n_tokens) = (256usize, 256usize, 512usize, 96usize);
    let eps = 1e-6f32;
    struct Case {
        wo: QuantMatrix,
        gate: QuantMatrix,
        up: QuantMatrix,
        down: QuantMatrix,
        attn_out: Vec<f32>,
        residual: Vec<f32>,
        attn_post_norm: Vec<f32>,
        ffn_norm: Vec<f32>,
        ffn_post_norm: Vec<f32>,
    }
    let cases: Vec<Case> = (0..4)
        .map(|t| {
            let mut seed = 0x50DA_u64.wrapping_add(t * 0x9E37_79B9);
            let norm = |n: usize, seed: &mut u64| -> Vec<f32> {
                concurrency_vec(n, seed)
                    .iter()
                    .map(|v| 1.0 + v * 0.1)
                    .collect()
            };
            Case {
                wo: concurrency_weight(attn_dim, n_embd, &mut seed),
                gate: concurrency_weight(n_embd, ffn_len, &mut seed),
                up: concurrency_weight(n_embd, ffn_len, &mut seed),
                down: concurrency_weight(ffn_len, n_embd, &mut seed),
                attn_out: concurrency_vec(n_tokens * attn_dim, &mut seed),
                residual: concurrency_vec(n_tokens * n_embd, &mut seed),
                attn_post_norm: norm(n_embd, &mut seed),
                ffn_norm: norm(n_embd, &mut seed),
                ffn_post_norm: norm(n_embd, &mut seed),
            }
        })
        .collect();
    assert_concurrent_agrees(
        &cases,
        |c| {
            vulkan
                .fused_post_attention_prefill(
                    AttnOutSrc::Host(&c.attn_out),
                    &c.residual,
                    n_tokens,
                    &c.wo,
                    Some(&c.attn_post_norm),
                    &c.ffn_norm,
                    &c.gate,
                    &c.up,
                    &c.down,
                    Some(&c.ffn_post_norm),
                    eps,
                    FfnActivation::Geglu,
                )
                .expect("fused post-attention available without MMVQ")
        },
        "fused post-attention prefill",
    );
}

/// `fused_attention_prefill` under concurrency — the last of the four
/// recorders that reach the pooled regions.
///
/// Each call builds its own `KvCache`, since the recorder mutates it; the
/// cache is per-request in production for the same reason, so this matches
/// how it is actually used rather than sharing one between threads.
#[test]
fn concurrent_fused_attention_prefills_do_not_corrupt_each_other() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    if vulkan.q4_k_mmvq {
        eprintln!("skipping: ORANGU_Q4K_MMVQ selects the unfused fallback path");
        return;
    }
    let (n_embd, n_head, n_head_kv, head_dim) = (256usize, 4usize, 2usize, 64usize);
    let (rope_dim, rope_freq_base, eps) = (64usize, 10000.0f32, 1e-6f32);
    let n_tokens = 96usize;
    let kv_dim = n_head_kv * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let capacity = n_tokens + 8;

    struct Case {
        wq: QuantMatrix,
        wk: QuantMatrix,
        wv: QuantMatrix,
        normed: Vec<f32>,
        q_norm: Vec<f32>,
        k_norm: Vec<f32>,
    }
    let cases: Vec<Case> = (0..4)
        .map(|t| {
            let mut seed = 0x0FA5_7ADD_u64.wrapping_add(t * 0x9E37_79B9);
            let norm = |n: usize, seed: &mut u64| -> Vec<f32> {
                concurrency_vec(n, seed)
                    .iter()
                    .map(|v| 1.0 + v * 0.1)
                    .collect()
            };
            Case {
                wq: concurrency_weight(n_embd, n_head * head_dim, &mut seed),
                wk: concurrency_weight(n_embd, kv_dim, &mut seed),
                wv: concurrency_weight(n_embd, kv_dim, &mut seed),
                normed: concurrency_vec(n_tokens * n_embd, &mut seed),
                q_norm: norm(head_dim, &mut seed),
                k_norm: norm(head_dim, &mut seed),
            }
        })
        .collect();

    assert_concurrent_agrees(
        &cases,
        |c| {
            let mut cache = crate::engine::kv_cache::KvCache::new_with_dims(capacity, &[kv_dim]);
            vulkan
                .fused_attention_prefill(
                    FusedAttnPrefillInput {
                        x_gpu: None,
                        attn_norm: None,
                        yarn: RopeYarn::IDENTITY,
                        q_bias: None,
                        pairing: crate::engine::tensor::RopeLayout::Neox,
                        normalize_v: true,
                        attn_gate: None,
                        normed: &c.normed,
                        n_tokens,
                        start_pos: 0,
                        wq: &c.wq,
                        q_norm: Some(&c.q_norm),
                        kv: Some(FusedAttnPrefillKv {
                            k_bias: None,
                            v_bias: None,
                            wk: &c.wk,
                            k_norm: Some(&c.k_norm),
                            wv: Some(&c.wv),
                        }),
                        n_head,
                        n_head_kv,
                        head_dim,
                        rope_dim,
                        rope_freq_base,
                        freq_factors: None,
                        eps,
                        n_swa: 0,
                        causal: true,
                        scale,
                        want_attn_out_host: true,
                    },
                    &mut cache.layers[0],
                )
                .expect("fused prefill attention available without MMVQ")
                .attn_out
        },
        "fused attention prefill",
    );
}

/// The same property for a *fused* recorder rather than the plain matmul
/// path — `fused_ple_prefill` uploads its activations with
/// `queue.write_buffer` too, so pooled regions need the same guard there.
///
/// Kept separate from the matmul version because the fused recorders reach
/// the pool by a different route (`op_entry_at` with a `ROLE_*` base
/// instead of `ROLE_BATCH + i`), and it is the routes that need covering,
/// not the pool.
#[test]
fn concurrent_fused_ple_prefills_do_not_corrupt_each_other() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    if vulkan.q4_k_mmvq {
        eprintln!("skipping: ORANGU_Q4K_MMVQ selects the unfused fallback path");
        return;
    }
    const THREADS: usize = 4;
    const ROUNDS: usize = 6;
    let (n_embd, per_layer_dim) = (256usize, 256usize);
    let n_tokens = 96;

    let mut cases = Vec::new();
    for t in 0..THREADS {
        let mut seed = 0x0BAD_F00D_u64.wrapping_add(t as u64 * 0x9E37_79B9);
        let build = |in_dim: usize, out_dim: usize, seed: &mut u64| {
            let mut bytes = Vec::new();
            for _ in 0..out_dim {
                for _ in 0..(in_dim / 256) {
                    bytes.extend(build_block(GGML_TYPE_Q4_K, seed));
                }
            }
            test_quant_matrix(&bytes, GGML_TYPE_Q4_K, in_dim, out_dim)
        };
        let gate = build(n_embd, per_layer_dim, &mut seed);
        let proj = build(per_layer_dim, n_embd, &mut seed);
        let x: Vec<f32> = (0..n_tokens * n_embd)
            .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
            .collect();
        let per_layer: Vec<f32> = (0..n_tokens * per_layer_dim)
            .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
            .collect();
        cases.push((gate, proj, x, per_layer));
    }
    let expected: Vec<Vec<f32>> = cases
        .iter()
        .map(|(gate, proj, x, per_layer)| {
            vulkan
                .fused_ple_prefill(x, n_tokens, gate, proj, per_layer)
                .expect("fused PLE returned None on a supported path")
        })
        .collect();

    std::thread::scope(|scope| {
        for (t, ((gate, proj, x, per_layer), want)) in cases.iter().zip(expected.iter()).enumerate()
        {
            scope.spawn(move || {
                for round in 0..ROUNDS {
                    let got = vulkan
                        .fused_ple_prefill(x, n_tokens, gate, proj, per_layer)
                        .expect("fused PLE returned None on a supported path");
                    assert_eq!(
                        &got, want,
                        "thread {t} round {round}: a concurrent fused PLE on \
                             same-shaped weights changed this one's result"
                    );
                }
            });
        }
    });
}

/// Two `n_tokens` widths of the *same* weight must not share an activation
/// region — `n_tokens` in [`OpCacheKey`] is load-bearing for **safety**,
/// not only for sizing, and this pins that.
///
/// The prefill recorders pass `batch_slot: 0` unconditionally while the
/// decode chain passes the real slot id through `op_entry_for` (which
/// hardcodes `n_tokens: 1`). So for slot `0` the *only* thing separating a
/// prefill op from a decode op on the same weight is the token count. Drop
/// it from the key — as an arena-footprint change is tempting to do, since
/// a narrow region is a subset of a wide one — and the two collide.
///
/// That collision is not benign. `queue.write_buffer` takes effect when it
/// is called, not in encoder order, and the decode chain releases its
/// entry guard after *recording*, before the submission runs. This is the
/// same failure that was already found and fixed once here by threading
/// `slot_id` into the decode keys: no panic, no wrong shape, just wrong
/// numbers under concurrency.
///
/// Written as a data test rather than a race reproduction on purpose — a
/// scheduling race reproduces unreliably, whereas "these two regions
/// overlap" is decidable and fails every time.
#[test]
fn a_decode_width_and_a_prefill_width_never_share_an_activation_region() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    let (in_dim, out_dim) = (512usize, 128usize);
    let elems = block_elems(GGML_TYPE_Q4_K);
    let mut seed = 0x51A7_51A7_u64;
    let mut bytes = Vec::new();
    for _ in 0..out_dim {
        for _ in 0..(in_dim / elems) {
            bytes.extend(build_block(GGML_TYPE_Q4_K, &mut seed));
        }
    }
    let w = test_quant_matrix(&bytes, GGML_TYPE_Q4_K, in_dim, out_dim);

    // Each guard is dropped before the next is taken. If a future change
    // merges the two widths into one cache entry, holding both at once
    // would *deadlock* on the same mutex — this test must fail, not hang.
    let region = |n_tokens: usize| {
        let x = vec![0f32; n_tokens * in_dim];
        let entry = vulkan.op_entry(
            &MatmulOp {
                x: &x,
                n_tokens,
                w: &w,
            },
            0,
        );
        let g = entry.lock().expect("op cache entry poisoned");
        (g.x_buffer.clone(), g.x_offset)
    };
    let (buf_decode, off_decode) = region(1);
    let (buf_prefill, off_prefill) = region(128);

    // Write a distinct pattern through each width's region, decode first.
    // `queue.write_buffer` takes effect at the next submission, in call
    // order, so the prefill write lands second — exactly the ordering that
    // makes a shared region clobber the decode activations.
    let decode_pattern: Vec<f32> = (0..in_dim).map(|i| 1.0 + i as f32).collect();
    let prefill_pattern: Vec<f32> = vec![-7.0; 128 * in_dim];
    vulkan.queue.write_buffer(
        &buf_decode,
        off_decode,
        bytemuck::cast_slice(&decode_pattern),
    );
    vulkan.queue.write_buffer(
        &buf_prefill,
        off_prefill,
        bytemuck::cast_slice(&prefill_pattern),
    );

    let staging = vulkan.scratch_buffer(in_dim);
    let mut encoder = vulkan.new_encoder("orangu-server op region isolation check");
    encoder.copy_buffer_to_buffer(&buf_decode, off_decode, &staging, 0, (in_dim as u64) * 4);
    let got = vulkan.submit_and_readback(encoder, &staging, 0, in_dim);

    assert_eq!(
        got, decode_pattern,
        "the prefill-width write landed on top of the decode-width activations — the two \
             widths of one weight share a region. `n_tokens` must stay in OpCacheKey (or the \
             prefill path must thread the real slot id) or concurrent requests corrupt each \
             other silently"
    );
}

/// Below the tiled crossover.
#[test]
fn recorded_matmul_matches_backend_matmul_below_crossover() {
    cross_check_recorded_matmul(GGML_TYPE_Q4_K, 512, 128, 32);
}

/// Above it — the regime where the grid and the kernel used to disagree.
#[test]
fn recorded_matmul_matches_backend_matmul_above_crossover() {
    cross_check_recorded_matmul(GGML_TYPE_Q4_K, 512, 128, 91);
}

/// Above it, at model-shaped dimensions and past one whole token tile.
#[test]
fn recorded_matmul_matches_backend_matmul_model_shaped() {
    cross_check_recorded_matmul(GGML_TYPE_Q4_K, 1536, 6144, 130);
}

/// A `Q6_K` weight, which `Q4_K_M` really does mix in (`ffn_down`).
#[test]
fn recorded_matmul_matches_backend_matmul_q6_k() {
    cross_check_recorded_matmul(GGML_TYPE_Q6_K, 6144, 1536, 91);
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
/// `vulkan_shaders::COOP_TILE_ROWS` and so never exercises more
/// than one *row* tile of the tiled GEMM's `(row-tile, token-tile)`
/// dispatch grid — only the token-tile boundary (already covered by
/// `n_tokens = 130`, 3 token tiles) was ever genuinely multi-tile.
/// `out_dim = 80` (3 row tiles at `COOP_TILE_ROWS = 32`: 0..32, 32..64,
/// 64..80 — the last only partially full, and the partial one not
/// aligned to the kernel's `REG_ROWS` register block either) combined
/// with `n_tokens = 130` (3 token tiles) and `in_dim = 768` (24
/// `COOP_CHUNK`-sized K-streaming iterations, vs. `Q4_K`'s native 3
/// super-blocks) exercises row-tile, token-tile, and K-chunk boundaries
/// all at once, for the one type (`Q4_K`) this project's real model
/// actually uses.
#[test]
fn matmul_matches_cpu_backend_cooperative_path_multi_row_tile_q4_k() {
    cross_check_n_tokens(GGML_TYPE_Q4_K, 768, 80, 130);
}

/// The tiled GEMM's weight staging dequantizes a **run** of consecutive
/// `k` from one row at a time, hoisting out everything the run shares —
/// for the K-quants the block scale and the 32-wide sub-block scale/min
/// pair. That is only correct while a run stays inside one sub-block, and
/// which sub-block a run lands in is a function of `k`'s position within
/// the 256-element super-block.
///
/// `in_dim = 1536` is six whole super-blocks — 48 K-chunks, every
/// sub-block index 0..8 and (for `Q6_K`) every `which_q` 0..4 visited many
/// times — where the pre-existing checks at `in_dim = 512` reach far fewer
/// of those positions. `out_dim = 100` is a partial last row-tile
/// (3 × 32 + 4) so the run fill's bounds-checked edge path runs too, and
/// `n_tokens = 130` clears three token tiles.
///
/// The reference is `CpuBackend`, which shares no code with the kernel;
/// `recorded_matmul_matches_backend_matmul_model_shaped` cannot serve here
/// because both of its sides run this same shader.

#[test]
fn tiled_dequant_run_matches_cpu_backend_q4_k() {
    cross_check_n_tokens(GGML_TYPE_Q4_K, 1536, 100, 130);
}

/// Whether the tuned `Q4_K` kernel cares where in the weight arena its
/// tensor happens to land.
///
/// A `Q4_K` block is 144 bytes, which is not a multiple of 16, so a tensor
/// that follows another in the arena starts at an offset the shader has to
/// handle rather than assume away. The startup probe measures a matrix that
/// is the *first* thing uploaded, and passes; a real model's tensors are
/// packed behind others, and that model answers in word salad. This asks
/// the question directly.
#[test]
#[ignore = "reports on the local device"]
fn does_arena_offset_change_the_q4_k_answer() {
    use crate::engine::backend::Backend;
    let Some(gpu) = crate::engine::backend::vulkan::VulkanBackend::try_init() else {
        println!("  no GPU");
        return;
    };
    let ty = crate::engine::quant::GGML_TYPE_Q4_K;
    let (in_dim, out_dim) = (2048usize, 256usize);
    let build = |seed: &mut u64| {
        let bytes: Vec<u8> = (0..in_dim * out_dim / 256)
            .flat_map(|_| crate::engine::backend::probe_blocks::build_block(ty, seed))
            .collect();
        crate::engine::loader::probe_quant_matrix(bytes, ty, in_dim, out_dim)
    };
    let x: Vec<f32> = (0..in_dim)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.05)
        .collect();
    let worst = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(a, b)| (a - b).abs() / a.abs().max(1.0))
            .fold(0.0f32, f32::max)
            * 100.0
    };
    println!(
        "  kernel selected: {}",
        gpu.selected_kernel_name(ty, in_dim, 1)
    );
    for fillers in [0usize, 1, 2, 3] {
        // Push the tensor under test further into the arena, one uploaded
        // matrix at a time.
        let mut seed = 0xF11E_2222u64;
        for _ in 0..fillers {
            let filler = build(&mut seed);
            let _ = gpu.matmul(&x, 1, &filler);
        }
        let mut seed = 0x51ed_1234u64;
        let weights = build(&mut seed);
        let exact = crate::engine::backend::CpuBackend.matmul_dequant(&x, 1, &weights);
        let mine = gpu.matmul(&x, 1, &weights);
        println!(
            "  after {fillers} filler tensor(s): gpu vs exact {:>9.1}%",
            worst(&exact, &mine)
        );
    }
}

/// How much of the self-check's reported disagreement is the *check*.
#[test]
fn how_far_is_the_cpus_own_matmul_from_its_dequant_reference() {
    use crate::engine::backend::Backend;
    use crate::engine::quant::block_layout;
    for (name, ty) in [
        ("Q4_K", crate::engine::quant::GGML_TYPE_Q4_K),
        ("Q5_1", crate::engine::quant::GGML_TYPE_Q5_1),
        ("Q2_K", crate::engine::quant::GGML_TYPE_Q2_K),
        ("Q8_0", crate::engine::quant::GGML_TYPE_Q8_0),
        ("Q6_K", crate::engine::quant::GGML_TYPE_Q6_K),
    ] {
        const IN_DIM: usize = 512;
        const OUT_DIM: usize = 4;
        let Some((_, block_elems)) = block_layout(ty) else {
            continue;
        };
        if !IN_DIM.is_multiple_of(block_elems) {
            continue;
        }
        let mut seed = 0x51ed_1234u64;
        let bytes: Vec<u8> = (0..IN_DIM * OUT_DIM / block_elems)
            .flat_map(|_| crate::engine::backend::probe_blocks::build_block(ty, &mut seed))
            .collect();
        let weights = crate::engine::loader::probe_quant_matrix(bytes, ty, IN_DIM, OUT_DIM);
        let x: Vec<f32> = (0..IN_DIM)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.05)
            .collect();
        let reference = crate::engine::backend::CpuBackend.matmul_dequant(&x, 1, &weights);
        let mine = crate::engine::backend::CpuBackend.matmul(&x, 1, &weights);
        let worst = reference
            .iter()
            .zip(&mine)
            .map(|(a, b)| (a - b).abs() / a.abs().max(1.0))
            .fold(0.0f32, f32::max);
        println!(
            "  {name:<5} cpu matmul vs cpu matmul_dequant: {:.1}%",
            worst * 100.0
        );
    }
}

#[test]
fn tiled_dequant_run_matches_cpu_backend_q5_k() {
    cross_check_n_tokens(GGML_TYPE_Q5_K, 1536, 100, 130);
}

/// `Q6_K` is what `Q4_K_M` really mixes in for `ffn_down`/`attn_v`, and it
/// hoists the most per run: the 8-bit scale, the `which_q` selector, and
/// both nibble shifts.
#[test]
fn tiled_dequant_run_matches_cpu_backend_q6_k() {
    cross_check_n_tokens(GGML_TYPE_Q6_K, 1536, 100, 130);
}

/// An `in_dim` that is not a multiple of `COOP_CHUNK`, which only a type
/// with a block smaller than the chunk can produce. The last K-chunk is
/// then partial and every thread's staging run takes the bounds-checked
/// path — the one place the fast run fill is bypassed entirely, and
/// otherwise unreached by any test here (every other `in_dim` in this file
/// is a multiple of 32).
#[test]
fn tiled_dequant_run_handles_a_partial_k_chunk() {
    cross_check_n_tokens(GGML_TYPE_F32, 70, 40, 130);
}

/// The actual batching path (`matmul_batch` with more than one op,
/// mirroring a transformer layer's independent Q/K/V projections: same
/// `x`, three different weight matrices, of two different quant
/// types) — one submission, one poll, must still return each op's
/// individually-correct result in the same order.
#[test]
fn matmul_batch_matches_sequential_cpu_matmuls() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
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

    let expected_q = CpuBackend.matmul_dequant(&x, n_tokens, &wq);
    let expected_k = CpuBackend.matmul_dequant(&x, n_tokens, &wk);
    let expected_v = CpuBackend.matmul_dequant(&x, n_tokens, &wv);

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
        // "v" (`Q8_0`) runs on the integer dot where the device has it.
        let w = match name {
            "q" => &wq,
            "k" => &wk,
            _ => &wv,
        };
        let integer_dot = matmul_tolerance(vulkan, w, n_tokens, expected);
        for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
            let tol = (tol_factor * a.abs().max(1.0)).max(integer_dot(*a));
            assert!(
                (a - b).abs() <= tol,
                "{name}: mismatch at index {i}: cpu={a} gpu(batched)={b}"
            );
        }
    }
}

/// `n_tokens = 300` deliberately spans three of `Backend::matmul_batch`'s
/// own token-range stripes (`MAX_MATMUL_TOKENS_PER_SUBMISSION = 128`:
/// 0..128, 128..256, 256..300 — the last only partially full), so this
/// exercises the chunking wrapper itself, not just the shapes it calls
/// into: results from several separate stripe submissions must
/// concatenate back into the exact same `[n_tokens, out_dim]` a single
/// unsplit call would have produced, for a batch of independent ops
/// (mirroring a real prefill layer's own Q/K/V projections) sharing one
/// `x` and one `n_tokens` — the shape this feature exists for.
#[test]
fn matmul_batch_matches_cpu_backend_across_multiple_token_stripes() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };

    let in_dim = 256;
    let mut seed = 0x57121E5_u64;
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

    let n_tokens = 300;
    let mut x = vec![0f32; n_tokens * in_dim];
    for v in x.iter_mut() {
        *v = (next_byte(&mut seed) as f32 - 128.0) / 64.0;
    }

    let expected_q = CpuBackend.matmul_dequant(&x, n_tokens, &wq);
    let expected_k = CpuBackend.matmul_dequant(&x, n_tokens, &wk);

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
    ]);
    assert_eq!(batch.len(), 2);
    let got_k = batch.pop().unwrap();
    let got_q = batch.pop().unwrap();

    for (name, expected, got) in [("q", &expected_q, &got_q), ("k", &expected_k, &got_k)] {
        assert_eq!(
            expected.len(),
            got.len(),
            "{name}: length mismatch — stripes didn't concatenate to the full n_tokens"
        );
        let tol_factor = if name == "q" { 6e-2 } else { 1e-2 };
        let out_dim = expected.len() / n_tokens;
        for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
            let tol = tol_factor * a.abs().max(1.0);
            assert!(
                (a - b).abs() <= tol,
                "{name}: mismatch at index {i} (token {}, dim {}): cpu={a} gpu={b}",
                i / out_dim,
                i % out_dim
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
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
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
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
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
    let mut attn_proj = CpuBackend.matmul_dequant(&attn_out, 1, &wo);
    crate::engine::tensor::rmsnorm_inplace(&mut attn_proj, &attn_post_norm, 1, n_embd, eps);
    let mut x = residual.clone();
    crate::engine::tensor::add_inplace(&mut x, &attn_proj);
    let attn_out_residual = x.clone();

    let mut ffn_normed = x.clone();
    crate::engine::tensor::rmsnorm_inplace(&mut ffn_normed, &ffn_norm, 1, n_embd, eps);
    let mut gate = CpuBackend.matmul_dequant(&ffn_normed, 1, &ffn_gate);
    let up = CpuBackend.matmul_dequant(&ffn_normed, 1, &ffn_up);
    for g in gate.iter_mut() {
        *g = crate::engine::tensor::gelu(*g);
    }
    crate::engine::tensor::mul_inplace(&mut gate, &up);
    let mut ffn_out = CpuBackend.matmul_dequant(&gate, 1, &ffn_down);
    crate::engine::tensor::rmsnorm_inplace(&mut ffn_out, &ffn_post_norm, 1, n_embd, eps);
    x = attn_out_residual;
    crate::engine::tensor::add_inplace(&mut x, &ffn_out);

    let pe_in = x.clone();
    let mut g = CpuBackend.matmul_dequant(&x, 1, &ple_gate_w);
    for v in g.iter_mut() {
        *v = crate::engine::tensor::gelu(*v);
    }
    crate::engine::tensor::mul_inplace(&mut g, &per_layer_slice);
    let mut proj = CpuBackend.matmul_dequant(&g, 1, &ple_proj_w);
    crate::engine::tensor::rmsnorm_inplace(&mut proj, &ple_post_norm, 1, n_embd, eps);
    x = pe_in;
    crate::engine::tensor::add_inplace(&mut x, &proj);

    for v in x.iter_mut() {
        *v *= layer_output_scale;
    }
    let expected = x;

    let got = vulkan.fused_post_attention(FusedPostAttentionInput {
        stop_at_ffn_norm: false,
        activation: FfnActivation::Geglu,
        attn_out: GpuInput::Cpu(&attn_out),
        residual: GpuInput::Cpu(&residual),
        wo: &wo,
        attn_post_norm: Some(&attn_post_norm),
        ffn_norm: &ffn_norm,
        ffn_gate: &ffn_gate,
        ffn_up: &ffn_up,
        ffn_gate_up: None,
        ffn_down: &ffn_down,
        ffn_post_norm: Some(&ffn_post_norm),
        eps,
        post_norm_eps: None,
        ple: Some(FusedPle {
            gate_w: &ple_gate_w,
            proj_w: &ple_proj_w,
            post_norm: &ple_post_norm,
            per_layer_slice: GpuInput::Cpu(&per_layer_slice),
            per_layer_dim: per_layer_slice.len(),
        }),
        layer_output_scale: Some(layer_output_scale),
        batch_slot: 0,
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
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
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

    let mut attn_proj = CpuBackend.matmul_dequant(&attn_out, 1, &wo);
    crate::engine::tensor::rmsnorm_inplace(&mut attn_proj, &attn_post_norm, 1, n_embd, eps);
    let mut x = residual.clone();
    crate::engine::tensor::add_inplace(&mut x, &attn_proj);
    let attn_out_residual = x.clone();

    let mut ffn_normed = x.clone();
    crate::engine::tensor::rmsnorm_inplace(&mut ffn_normed, &ffn_norm, 1, n_embd, eps);
    let mut gate = CpuBackend.matmul_dequant(&ffn_normed, 1, &ffn_gate);
    let up = CpuBackend.matmul_dequant(&ffn_normed, 1, &ffn_up);
    for g in gate.iter_mut() {
        *g = crate::engine::tensor::gelu(*g);
    }
    crate::engine::tensor::mul_inplace(&mut gate, &up);
    let mut ffn_out = CpuBackend.matmul_dequant(&gate, 1, &ffn_down);
    crate::engine::tensor::rmsnorm_inplace(&mut ffn_out, &ffn_post_norm, 1, n_embd, eps);
    x = attn_out_residual;
    crate::engine::tensor::add_inplace(&mut x, &ffn_out);
    let expected = x;

    let got = vulkan.fused_post_attention(FusedPostAttentionInput {
        stop_at_ffn_norm: false,
        activation: FfnActivation::Geglu,
        attn_out: GpuInput::Cpu(&attn_out),
        residual: GpuInput::Cpu(&residual),
        wo: &wo,
        attn_post_norm: Some(&attn_post_norm),
        ffn_norm: &ffn_norm,
        ffn_gate: &ffn_gate,
        ffn_up: &ffn_up,
        ffn_gate_up: None,
        ffn_down: &ffn_down,
        ffn_post_norm: Some(&ffn_post_norm),
        eps,
        post_norm_eps: None,
        ple: None,
        layer_output_scale: None,
        batch_slot: 0,
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
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
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

    let cpu_reference = |attn_out: &[f32], residual: &[f32], per_layer_slice: &[f32]| -> Vec<f32> {
        let mut attn_proj = CpuBackend.matmul_dequant(attn_out, 1, &wo);
        crate::engine::tensor::rmsnorm_inplace(&mut attn_proj, &attn_post_norm, 1, n_embd, eps);
        let mut x = residual.to_vec();
        crate::engine::tensor::add_inplace(&mut x, &attn_proj);
        let attn_out_residual = x.clone();

        let mut ffn_normed = x.clone();
        crate::engine::tensor::rmsnorm_inplace(&mut ffn_normed, &ffn_norm, 1, n_embd, eps);
        let mut gate = CpuBackend.matmul_dequant(&ffn_normed, 1, &ffn_gate);
        let up = CpuBackend.matmul_dequant(&ffn_normed, 1, &ffn_up);
        for g in gate.iter_mut() {
            *g = crate::engine::tensor::gelu(*g);
        }
        crate::engine::tensor::mul_inplace(&mut gate, &up);
        let mut ffn_out = CpuBackend.matmul_dequant(&gate, 1, &ffn_down);
        crate::engine::tensor::rmsnorm_inplace(&mut ffn_out, &ffn_post_norm, 1, n_embd, eps);
        x = attn_out_residual;
        crate::engine::tensor::add_inplace(&mut x, &ffn_out);

        let pe_in = x.clone();
        let mut g = CpuBackend.matmul_dequant(&x, 1, &ple_gate_w);
        for v in g.iter_mut() {
            *v = crate::engine::tensor::gelu(*v);
        }
        crate::engine::tensor::mul_inplace(&mut g, per_layer_slice);
        let mut proj = CpuBackend.matmul_dequant(&g, 1, &ple_proj_w);
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
            stop_at_ffn_norm: false,
            activation: FfnActivation::Geglu,
            attn_out: GpuInput::Cpu(&attn_out),
            residual: GpuInput::Cpu(&residual),
            wo: &wo,
            attn_post_norm: Some(&attn_post_norm),
            ffn_norm: &ffn_norm,
            ffn_gate: &ffn_gate,
            ffn_up: &ffn_up,
            ffn_gate_up: None,
            ffn_down: &ffn_down,
            ffn_post_norm: Some(&ffn_post_norm),
            eps,
            post_norm_eps: None,
            ple: Some(FusedPle {
                gate_w: &ple_gate_w,
                proj_w: &ple_proj_w,
                post_norm: &ple_post_norm,
                per_layer_slice: GpuInput::Cpu(&per_layer_slice),
                per_layer_dim: per_layer_slice.len(),
            }),
            layer_output_scale: Some(layer_output_scale),
            batch_slot: 0,
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

/// Same cross-check as [`gpu_attention_matches_cpu_reference_full_window`]
/// below, but with `head_dim = 32` so `kv_dim` (`n_head_kv * head_dim`)
/// is a multiple of 32 — the one shape `KvStorage::Q8_0`'s block
/// format requires (see its own doc comment). Every other cross-check
/// test in this module uses smaller, non-block-aligned dims and so
/// only ever exercises whichever of `F32`/`F16` `Self::kv_storage`
/// picked at `shared_vulkan()`'s construction; this one is run twice
/// by hand — once under the ambient default, once with
/// `ORANGU_KV_Q8_0=1` set before the test binary starts — to check
/// the quantize-on-write shader and the attention shader's
/// dequant-on-read path against each other and against this same CPU
/// reference. The tolerance is wider than the other cross-check tests'
/// here specifically to give `Q8_0`'s lossy 8-bit quantization (versus
/// `F16`'s much smaller rounding error) room to differ from the exact
/// CPU result.
#[test]
fn gpu_attention_matches_cpu_reference_kv_dim_32() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };

    let n_head = 4;
    let n_head_kv = 2;
    let head_dim = 32;
    let group_size = n_head / n_head_kv;
    let kv_dim = n_head_kv * head_dim;
    let capacity = 16;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut seed = 0x008A_0D1D_u64;
    let mut kv_cache = crate::engine::kv_cache::KvCache::new_with_dims(capacity, &[kv_dim]);
    let n_positions = 5;
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
        let tol = 1.5e-1 * a.abs().max(1.0);
        assert!(
            (a - b).abs() <= tol,
            "mismatch at index {i}: cpu={a} gpu={b}"
        );
    }
}

/// Cross-checks `gpu_attention` against the exact CPU attention loop
/// `GemmaModel::forward` runs (per-head dot products against the
/// cached keys in the causal window, softmax, weighted value sum) —
/// GQA (`n_head_kv < n_head`), a KV cache with several positions
/// already pushed, and full (non-windowed) attention.
#[test]
fn gpu_attention_matches_cpu_reference_full_window() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
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
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
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
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
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

/// Cross-checks `gpu_attention_split` (the split-k
/// phase-1 + reduce phase-2 pipeline pair) against the same CPU
/// reference loop the `gpu_attention` tests above use. `n_positions =
/// 37` deliberately doesn't divide evenly by `ATTN_SPLIT_K = 4`
/// (37 = 9+9+9+10), exercising the uneven-remainder split-range
/// bookkeeping in `ATTENTION_SPLIT_SHADER_TEMPLATE`, not just the
/// tidy multiple-of-k_num case.
/// The same guarantee for **prefill**, which is a different kernel family from
/// the decode split path and reads the cache through its own address
/// computation.
///
/// Same construction as the decode cross-check and for the same reason: the
/// sequence lives in the upper half of a pool twice its size, so a kernel that
/// ignored the block table reads the lower half rather than a permutation of
/// its own pages. A multi-token prefill also exercises the per-query window
/// derivation, which the single-query decode path does not.
#[test]
fn paged_prefill_matches_the_contiguous_kernel() {
    paged_prefill_agrees_at(4, 8);
}

/// The same at a realistic context. A 32-position prefill exercises one tile
/// and one page-table lookup per query; a sixteen-hundred-position one
/// exercises the window derivation, many tiles, and a block table long enough
/// that an off-by-one in its base or stride has somewhere to go wrong.
///
/// Written because the small case passed while the engine diverged end-to-end
/// at ~1600 positions, which is exactly the gap a small fixture leaves.
#[test]
fn paged_prefill_matches_the_contiguous_kernel_at_a_real_context() {
    paged_prefill_agrees_at(16, 100);
}

/// **A context that is not a whole number of pages.**
///
/// Every fixture above used `page * pages` positions exactly, so the sequence
/// always ended on a page boundary and there was never a partial tail. A real
/// prompt almost never does: 1638 tokens at 16 to a page leaves 6 in a page
/// that has not been sealed. Those positions are what a dispatch has to be able
/// to read, and an exact-multiple fixture cannot ask about them.
#[test]
fn paged_prefill_matches_the_contiguous_kernel_with_a_partial_tail() {
    paged_prefill_agrees_at_len(16, 100, 16 * 100 - 10);
}

fn paged_prefill_agrees_at(page: usize, pages: usize) {
    paged_prefill_agrees_at_len(page, pages, page * pages);
}

fn paged_prefill_agrees_at_len(page: usize, pages: usize, positions: usize) {
    use crate::engine::kv_pool::{KvPool, LayerGeometry, Policy};

    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    let (page_tokens, n_pages) = (page, pages);
    // Twice the sequence, plus the two this test then holds back on purpose:
    // a sequence is admitted against pool room now, so a pool that is exactly
    // twice the sequence and half held is one page short of promising it.
    let pool_pages: usize = n_pages * 2 + 2;
    const N_HEAD: usize = 4;
    const N_HEAD_KV: usize = 2;
    const HEAD_DIM: usize = 64;
    let kv_dim = N_HEAD_KV * HEAD_DIM;
    let n_positions = positions;

    let mut seed = 0xBEEF_4321_u64;
    let mut rows_k: Vec<Vec<f32>> = Vec::new();
    let mut rows_v: Vec<Vec<f32>> = Vec::new();
    for _ in 0..n_positions {
        rows_k.push(
            (0..kv_dim)
                .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
                .collect(),
        );
        rows_v.push(
            (0..kv_dim)
                .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
                .collect(),
        );
    }
    // Every cached position is a query, which is the prefill shape.
    let n_tokens = n_positions;
    let q: Vec<f32> = (0..n_tokens * N_HEAD * HEAD_DIM)
        .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
        .collect();
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut plain = crate::engine::kv_cache::KvCache::new(1, n_positions + 1, kv_dim);
    for i in 0..n_positions {
        plain.layers[0].push(&rows_k[i], &rows_v[i]);
    }
    let want = vulkan.gpu_attention_prefill(
        &q,
        &mut plain.layers[0],
        0,
        n_tokens,
        N_HEAD,
        N_HEAD_KV,
        HEAD_DIM,
        0,
        true,
        scale,
    );

    // The same rows, reached only through a block table.
    let mut pool = KvPool::with_policy(
        pool_pages,
        page_tokens,
        vec![LayerGeometry {
            kv_dim,
            stride: 1,
            ring: None,
        }],
        Policy::Lru,
    );
    let (device, queue) = vulkan.device_and_queue();
    // Table room for every page the pool has, twice over — a sequence reserves
    // `pages_for(capacity)` entries up front, and a table too small to hold
    // that makes the sequence fall back to the mirror silently. The assertion
    // below catches it, but sizing it right is what makes the test a test.
    assert!(pool.attach_device(device, vulkan.kv_storage(), pool_pages * 4));
    let pool = std::sync::Arc::new(pool);

    // **Hold the low pages so the sequence cannot land on them.**
    //
    // The pool hands out never-used pages from the bottom, so a cache created
    // against an empty pool gets 0, 1, 2 … — which is exactly the identity
    // mapping. A kernel ignoring the block table would then read the right rows
    // by accident and the test would pass while proving nothing; it did, until
    // this was added. Holding the low half forces the sequence into the upper
    // half, where only the table can find it.
    let held = pool.alloc(n_pages).expect("pool has room");
    for &physical in &held {
        let junk: Vec<f32> = (0..page_tokens * kv_dim)
            .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 32.0)
            .collect();
        pool.fill_device(queue, 0, physical, &junk, &junk);
    }

    let mut paged = crate::engine::kv_cache::KvCache::new_with_strided_dims(
        n_positions + page_tokens,
        &strided_dims(&pool),
    )
    .try_into_paged(pool.clone())
    .unwrap_or_else(|_| panic!("test pool has room"));
    for i in 0..n_positions {
        paged.layers[0].push(&rows_k[i], &rows_v[i]);
    }
    paged.commit_pages();
    // Without this the test can silently compare the fallback against itself:
    // `gpu_attention_prefill` uses the per-request mirror whenever the cache
    // cannot supply pages, and that mirror is materialized *from* the pages, so
    // it is correct no matter what the block table says.
    assert!(
        paged.layers[0].paged_device_refs(queue).is_some(),
        "the paged path was not taken; this test would be comparing the \
         mirrored fallback with itself"
    );

    let got = vulkan.gpu_attention_prefill(
        &q,
        &mut paged.layers[0],
        0,
        n_tokens,
        N_HEAD,
        N_HEAD_KV,
        HEAD_DIM,
        0,
        true,
        scale,
    );

    assert_eq!(got.len(), want.len());
    let mut worst = 0f32;
    for (a, b) in want.iter().zip(got.iter()) {
        worst = worst.max((a - b).abs() / a.abs().max(1e-3));
    }
    assert!(
        worst < 1e-3,
        "paged prefill disagrees with the contiguous kernel (worst relative \
         error {worst:.3e})"
    );
    pool.release(&held);
}

/// **The paged kernel against the contiguous one, through a table that maps
/// away from where the kernel would otherwise look.**
///
/// The same keys and values, the same query, the same window — once in a
/// contiguous per-request cache and once in pool pages the block table has to
/// be consulted to find.
///
/// # A shuffle alone proves nothing, which took a mutant to discover
///
/// The first version of this test put the sequence in pages `0..8` and reversed
/// the table. A kernel that ignored the table entirely **passed it**, at a
/// relative error of `4e-5`.
///
/// The reason is that attention over a full window is *permutation-invariant*:
/// softmax normalizes, and the weighted sum of values does not depend on the
/// order the pairs are visited. Reversing the pages changes the order the
/// kernel reads them and not the set, so the answer is the same to within
/// floating-point reassociation. The test was measuring reassociation noise and
/// calling it agreement.
///
/// So the sequence is placed in the *upper* half of a pool twice its size, and
/// the lower half is filled with different data. Now a kernel that computes
/// `p * n_head_kv * head_dim` reads the lower half — a different set of pairs,
/// not a permutation of the same one — and the answers separate.
#[test]
fn paged_attention_matches_the_contiguous_kernel_through_a_shuffled_table() {
    use crate::engine::kv_pool::{KvPool, LayerGeometry, Policy};

    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    // One page per 4 positions. The sequence needs `PAGES`; the pool is twice
    // that, and the sequence lives in the upper half — see the note above on
    // why a permutation within its own pages would not discriminate.
    const PAGE: usize = 4;
    const PAGES: usize = 8;
    const POOL_PAGES: usize = 16;
    const N_HEAD: usize = 4;
    const N_HEAD_KV: usize = 2;
    const HEAD_DIM: usize = 64;
    let kv_dim = N_HEAD_KV * HEAD_DIM;
    let n_positions = PAGE * PAGES;

    let mut seed = 0x5EED_1234_u64;
    let mut rows_k: Vec<Vec<f32>> = Vec::new();
    let mut rows_v: Vec<Vec<f32>> = Vec::new();
    for _ in 0..n_positions {
        rows_k.push(
            (0..kv_dim)
                .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
                .collect(),
        );
        rows_v.push(
            (0..kv_dim)
                .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
                .collect(),
        );
    }
    let q: Vec<f32> = (0..N_HEAD * HEAD_DIM)
        .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
        .collect();
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Contiguous reference, through the kernel that already has cross-checks.
    let mut kv_cache = crate::engine::kv_cache::KvCache::new(1, n_positions + 1, kv_dim);
    for i in 0..n_positions {
        kv_cache.layers[0].push(&rows_k[i], &rows_v[i]);
    }
    let want = vulkan.gpu_attention_split(crate::engine::backend::vulkan::GpuAttentionInput {
        q: &q,
        cache: &mut kv_cache.layers[0],
        pos: n_positions - 1,
        window_start: 0,
        n_head: N_HEAD,
        n_head_kv: N_HEAD_KV,
        head_dim: HEAD_DIM,
        scale,
    });

    // Paged: logical page i lives in physical page (PAGES - 1 - i).
    let mut pool = KvPool::with_policy(
        POOL_PAGES,
        PAGE,
        vec![LayerGeometry {
            kv_dim,
            stride: 1,
            ring: None,
        }],
        Policy::Lru,
    );
    let (device, queue) = vulkan.device_and_queue();
    assert!(pool.attach_device(device, vulkan.kv_storage(), 64));

    // Decoy data where an unmapped kernel would look.
    for physical in 0..PAGES {
        let junk: Vec<f32> = (0..PAGE * kv_dim)
            .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 32.0)
            .collect();
        pool.fill_device(queue, 0, physical as u32, &junk, &junk);
    }

    // The sequence: upper half, and reversed within it so order is exercised
    // too.
    let table: Vec<u32> = (0..PAGES).map(|i| (POOL_PAGES - 1 - i) as u32).collect();
    for (logical, &physical) in table.iter().enumerate() {
        let mut k = Vec::new();
        let mut v = Vec::new();
        for r in 0..PAGE {
            k.extend_from_slice(&rows_k[logical * PAGE + r]);
            v.extend_from_slice(&rows_v[logical * PAGE + r]);
        }
        pool.fill_device(queue, 0, physical, &k, &v);
    }
    pool.write_table(queue, 0, &table);

    let got = vulkan.gpu_attention_split_paged(
        &q,
        &pool,
        0,
        0,
        n_positions - 1,
        0,
        N_HEAD,
        N_HEAD_KV,
        HEAD_DIM,
        scale,
    );

    assert_eq!(got.len(), want.len());
    let mut worst = 0f32;
    for (a, b) in want.iter().zip(got.iter()) {
        worst = worst.max((a - b).abs() / a.abs().max(1e-3));
    }
    assert!(
        worst < 1e-3,
        "paged attention disagrees with the contiguous kernel (worst relative \
         error {worst:.3e}); the block table is not being followed"
    );
}

#[test]
fn gpu_attention_split_matches_cpu_reference() {
    cross_check_gpu_attention_split(4, 2, 8);
}

/// The same check at a `head_dim` that is a multiple of 32, which is what
/// selects the **cooperative** phase-1 kernel — the decode default. The
/// case above uses `head_dim = 8` and therefore silently exercises the
/// classic kernel instead, leaving the one that actually runs in
/// production uncovered. 256 and 512 are the model's own two head_dims.
#[test]
fn gpu_attention_split_coop_matches_cpu_reference_head_dim_256() {
    cross_check_gpu_attention_split(4, 2, 256);
}

#[test]
fn gpu_attention_split_coop_matches_cpu_reference_head_dim_512() {
    cross_check_gpu_attention_split(8, 1, 512);
}

fn cross_check_gpu_attention_split(n_head: usize, n_head_kv: usize, head_dim: usize) {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };

    let group_size = n_head / n_head_kv;
    let kv_dim = n_head_kv * head_dim;
    let capacity = 64;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut seed = 0x59717_u64;
    let mut kv_cache = crate::engine::kv_cache::KvCache::new_with_dims(capacity, &[kv_dim]);
    let n_positions = 37;
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

    let got = vulkan.gpu_attention_split(GpuAttentionInput {
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

/// `n_positions = 2 < ATTN_SPLIT_K = 4` — most of the `k_num` split
/// workgroups get an *empty* `[split_start, split_end)` range. Checks
/// that phase 1 leaves those workgroups' partial state as a proper
/// softmax identity (`m = -inf`, `l = 0`, `acc = 0`) and phase 2's
/// merge correctly ignores them, rather than corrupting the result
/// with e.g. uninitialized-buffer garbage or a `NaN` from `0/0`.
#[test]
fn gpu_attention_split_matches_cpu_reference_fewer_positions_than_splits() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };

    let n_head = 2;
    let n_head_kv = 1;
    let head_dim = 8;
    let group_size = n_head / n_head_kv;
    let kv_dim = n_head_kv * head_dim;
    let capacity = 16;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut seed = 0xF0F0F_u64;
    let mut kv_cache = crate::engine::kv_cache::KvCache::new_with_dims(capacity, &[kv_dim]);
    let n_positions = 2;
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

    let got = vulkan.gpu_attention_split(GpuAttentionInput {
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

/// Cross-checks `gpu_rope` against `tensor::rope_apply_scaled_inplace`
/// — no `freq_factors` (the common case: SWA layers, and every layer
/// in models without Gemma4's proportional-RoPE tensor).
#[test]
fn gpu_rope_matches_cpu_reference_without_freq_factors() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
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
        yarn: RopeYarn::IDENTITY,
        x: &x,
        n_head,
        head_dim,
        rope_dim,
        pos,
        freq_base,
        freq_factors: None,
        layout: crate::engine::tensor::RopeLayout::Neox,
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
/// The `llama`/`mistral` pairing. A separate test rather than a parameter
/// on the existing ones because the failure it guards is not a tolerance
/// question: NEOX and NORM rotate *different pairs of elements*, so the
/// wrong one is not a slightly wrong answer, it is a different tensor. The
/// reference is `engine::tensor`'s own CPU RoPE, which shares nothing with
/// the shader.
#[test]
fn gpu_rope_matches_cpu_reference_with_norm_pairing() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    let (n_head, head_dim, rope_dim, pos, freq_base) = (4usize, 32usize, 32usize, 7usize, 1e4f32);
    let mut st = 0x1234_5678u64;
    let mut rand_vec = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| {
                st = st.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                ((st >> 33) as f32 / 2f32.powi(31)) - 1.0
            })
            .collect()
    };
    let x = rand_vec(n_head * head_dim);

    let params = crate::engine::tensor::RopeParams {
        rope_dim,
        freq_base,
        layout: crate::engine::tensor::RopeLayout::Norm,
        ..crate::engine::tensor::RopeParams::default()
    };
    let mut expected = x.clone();
    crate::engine::tensor::rope_apply_params_inplace(
        &mut expected,
        n_head,
        head_dim,
        pos,
        None,
        &params,
    );

    let got = vulkan.gpu_rope(GpuRopeInput {
        yarn: RopeYarn::IDENTITY,
        x: &x,
        n_head,
        head_dim,
        rope_dim,
        pos,
        freq_base,
        freq_factors: None,
        layout: crate::engine::tensor::RopeLayout::Norm,
    });

    assert_eq!(got.len(), expected.len());
    for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
        assert!(
            (a - b).abs() <= 1e-4 * a.abs().max(1.0),
            "mismatch at {i}: cpu={a} gpu={b}"
        );
    }
    // And the two layouts must not agree with each other, or this test
    // would pass against a shader that ignored `layout` entirely.
    let neox = vulkan.gpu_rope(GpuRopeInput {
        yarn: RopeYarn::IDENTITY,
        x: &x,
        n_head,
        head_dim,
        rope_dim,
        pos,
        freq_base,
        freq_factors: None,
        layout: crate::engine::tensor::RopeLayout::Neox,
    });
    assert!(
        neox.iter()
            .zip(got.iter())
            .any(|(a, b)| (a - b).abs() > 1e-3),
        "NEOX and NORM produced the same tensor — `pairing` is being ignored"
    );
}

/// Ministral-3-3B's own RoPE hyperparameters, read from the checkpoint:
/// `scaling.type = yarn`, `factor = 16`, `beta_fast/slow = 32/1`,
/// `original_context_length = 16384`, `freq_base = 1e6`, `head_dim = 128`,
/// NORM pairing.
fn ministral_yarn_params() -> crate::engine::tensor::RopeParams {
    crate::engine::tensor::RopeParams {
        rope_dim: 128,
        freq_base: 1.0e6,
        freq_scale: 1.0 / 16.0,
        ext_factor: 1.0,
        attn_factor: 1.0 / (1.0 + 0.1 * 16.0f32.ln()),
        beta_fast: 32.0,
        beta_slow: 1.0,
        n_ctx_orig: 16384,
        layout: crate::engine::tensor::RopeLayout::Norm,
    }
}

/// RoPE's angle is `pos * freq`, and `sin`/`cos` of a large argument lose
/// precision differently on the CPU and on the GPU. This measures that
/// divergence rather than asserting it away, because it bounds how strict
/// every other RoPE cross-check in this file can be — and it is a property
/// of plain RoPE, present long before any scaling was added.
///
/// The bound is loose on purpose: it exists to catch a *change* in the
/// characteristic, not to pin a specific adapter's `sin` implementation.
#[test]
fn gpu_rope_argument_reduction_diverges_with_position() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    let params = crate::engine::tensor::RopeParams {
        rope_dim: 128,
        freq_base: 1.0e6,
        layout: crate::engine::tensor::RopeLayout::Norm,
        ..crate::engine::tensor::RopeParams::default()
    };
    let (n_head, head_dim) = (4usize, 128usize);
    let x: Vec<f32> = (0..n_head * head_dim)
        .map(|i| ((i % 17) as f32 / 17.0) - 0.5)
        .collect();
    // Position, and the error budget it justifies.
    // The bound at each position. The first was 1e-6, which measured fine on
    // the adapter it was written against and pins that adapter's `sin` —
    // exactly what the comment above says this test is not for. A Mali-G720
    // lands at 1.1e-5 for `pos = 8`, well inside what Vulkan guarantees for
    // a transcendental (it guarantees very little), so the budget is the
    // measured value with room, and the divergence this test is named for is
    // still visible across the three.
    for (pos, bound) in [(8usize, 5e-5f32), (1_000, 1e-4), (20_000, 4e-3)] {
        let mut expected = x.clone();
        crate::engine::tensor::rope_apply_params_inplace(
            &mut expected,
            n_head,
            head_dim,
            pos,
            None,
            &params,
        );
        let got = vulkan.gpu_rope(GpuRopeInput {
            yarn: RopeYarn::IDENTITY,
            x: &x,
            n_head,
            head_dim,
            rope_dim: params.rope_dim,
            pos,
            freq_base: params.freq_base,
            freq_factors: None,
            layout: params.layout,
        });
        let err = expected
            .iter()
            .zip(got.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(err < bound, "pos={pos}: max abs err {err} exceeds {bound}");
    }
}

#[test]
fn gpu_rope_matches_cpu_reference_with_yarn_scaling() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };

    let params = ministral_yarn_params();
    // `pos` is squeezed from both sides. Too large and `sin`/`cos` of a
    // big argument diverges between CPU and GPU by more than the tolerance
    // below, for reasons that have nothing to do with scaling — measured in
    // `gpu_rope_argument_reduction_diverges_with_position`. Too small and
    // YaRN's own effect shrinks with it (every angle is proportional to
    // `pos`) until the "and they must differ" guard cannot see it: at
    // `pos = 8` the largest divergence across the whole ramp band is under
    // 1e-2. 512 clears both by an order of magnitude.
    let (n_head, head_dim, pos) = (4usize, 128usize, 512usize);
    let mut st = 0x0BAD_F00Du64;
    let mut rand_vec = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| {
                st = st.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                ((st >> 33) as f32 / 2f32.powi(31)) - 1.0
            })
            .collect()
    };
    let x = rand_vec(n_head * head_dim);

    let mut expected = x.clone();
    crate::engine::tensor::rope_apply_params_inplace(
        &mut expected,
        n_head,
        head_dim,
        pos,
        None,
        &params,
    );

    let got = vulkan.gpu_rope(GpuRopeInput {
        yarn: RopeYarn::from_params(&params),
        x: &x,
        n_head,
        head_dim,
        rope_dim: params.rope_dim,
        pos,
        freq_base: params.freq_base,
        freq_factors: None,
        layout: params.layout,
    });

    assert_eq!(got.len(), expected.len());
    for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
        assert!(
            (a - b).abs() <= 1e-4 * a.abs().max(1.0),
            "mismatch at {i}: cpu={a} gpu={b}"
        );
    }
    // The guard that makes the above mean something: with the identity
    // tail the shader computes the *unscaled* rope, and at `pos` well past
    // `n_ctx_orig` the two must be nowhere near each other. Without this a
    // shader that dropped every YaRN term would still pass — the failure
    // mode this whole change exists to avoid.
    let unscaled = vulkan.gpu_rope(GpuRopeInput {
        yarn: RopeYarn::IDENTITY,
        x: &x,
        n_head,
        head_dim,
        rope_dim: params.rope_dim,
        pos,
        freq_base: params.freq_base,
        freq_factors: None,
        layout: params.layout,
    });
    assert!(
        unscaled
            .iter()
            .zip(got.iter())
            .any(|(a, b)| (a - b).abs() > 1e-2),
        "YaRN and unscaled RoPE produced the same tensor — the terms are being ignored"
    );
}

#[test]
fn gpu_fused_norm_rope_matches_cpu_reference_with_yarn_scaling() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };

    let params = ministral_yarn_params();
    // `pos` chosen the same way, and for the same two reasons, as the
    // `gpu_rope` YaRN test above.
    let (n_head, head_dim, pos, eps) = (4usize, 128usize, 512usize, 1e-5f32);
    let mut st = 0xFEED_BEEFu64;
    let mut rand_vec = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| {
                st = st.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                ((st >> 33) as f32 / 2f32.powi(31)) - 1.0
            })
            .collect()
    };
    let x = rand_vec(n_head * head_dim);
    let weight = rand_vec(head_dim);

    let mut expected = x.clone();
    crate::engine::tensor::rmsnorm_inplace(&mut expected, &weight, n_head, head_dim, eps);
    crate::engine::tensor::rope_apply_params_inplace(
        &mut expected,
        n_head,
        head_dim,
        pos,
        None,
        &params,
    );

    let got = vulkan.gpu_fused_norm_rope(GpuFusedNormRopeInput {
        yarn: RopeYarn::from_params(&params),
        x: &x,
        weight: &weight,
        n_tokens: 1,
        n_head,
        head_dim,
        rope_dim: params.rope_dim,
        pos,
        freq_base: params.freq_base,
        freq_factors: None,
        eps,
        pairing: params.layout,
    });

    assert_eq!(got.len(), expected.len());
    for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
        assert!(
            (a - b).abs() <= 1e-4 * a.abs().max(1.0),
            "mismatch at {i}: cpu={a} gpu={b}"
        );
    }
    // The fused kernel is a *different* kernel from `gpu_rope`, so it needs
    // its own proof that the terms reach it — see `ROPE_YARN_WGSL`.
    let unscaled = vulkan.gpu_fused_norm_rope(GpuFusedNormRopeInput {
        yarn: RopeYarn::IDENTITY,
        x: &x,
        weight: &weight,
        n_tokens: 1,
        n_head,
        head_dim,
        rope_dim: params.rope_dim,
        pos,
        freq_base: params.freq_base,
        freq_factors: None,
        eps,
        pairing: params.layout,
    });
    assert!(
        unscaled
            .iter()
            .zip(got.iter())
            .any(|(a, b)| (a - b).abs() > 1e-2),
        "YaRN and unscaled RoPE produced the same tensor — the terms are being ignored"
    );
}

#[test]
fn gpu_rope_matches_cpu_reference_with_freq_factors_and_partial_rope() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
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
        yarn: RopeYarn::IDENTITY,
        x: &x,
        n_head,
        head_dim,
        rope_dim,
        pos,
        freq_base,
        freq_factors: Some(&freq_factors),
        layout: crate::engine::tensor::RopeLayout::Neox,
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
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
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

/// Cross-checks `gpu_fused_norm_rope` against calling `tensor::
/// rmsnorm_inplace` then `tensor::rope_apply_scaled_inplace` on the
/// result — the same two CPU references `gpu_perhead_rmsnorm_matches_
/// cpu_reference`/`gpu_rope_matches_cpu_reference_without_freq_
/// factors` each check individually, run back to back, since that's
/// exactly what the fused dispatch replaces. No `freq_factors` (SWA
/// layers, and every layer in models without Gemma4's proportional
/// RoPE).
#[test]
fn gpu_fused_norm_rope_matches_cpu_reference_without_freq_factors() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };

    let n_head = 5;
    let head_dim = 16;
    let rope_dim = 16;
    let pos = 17;
    let freq_base = 10000.0;
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
    crate::engine::tensor::rope_apply_scaled_inplace(
        &mut expected,
        n_head,
        head_dim,
        rope_dim,
        pos,
        freq_base,
        None,
    );

    let got = vulkan.gpu_fused_norm_rope(GpuFusedNormRopeInput {
        yarn: RopeYarn::IDENTITY,
        pairing: crate::engine::tensor::RopeLayout::Neox,
        x: &x,
        weight: &weight,
        n_tokens: 1,
        n_head,
        head_dim,
        rope_dim,
        pos,
        freq_base,
        freq_factors: None,
        eps,
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
/// rope_dim`, so the tail of each head must pass through the norm's
/// output untouched by the rotation) — the exact shape `record_fused_
/// attention` dispatches for E2B's full-attention layers.
#[test]
fn gpu_fused_norm_rope_matches_cpu_reference_with_freq_factors_and_partial_rope() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };

    let n_head = 3;
    let head_dim = 10;
    let rope_dim = 6; // < head_dim: elements [6, 10) must stay unchanged
    let pos = 5;
    let freq_base = 1_000_000.0;
    let eps = 1e-6;

    let mut seed = 0xFACE0FF_u64;
    let x: Vec<f32> = (0..n_head * head_dim)
        .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
        .collect();
    let weight: Vec<f32> = (0..head_dim)
        .map(|_| (next_byte(&mut seed) as f32) / 128.0)
        .collect();
    let freq_factors: Vec<f32> = (0..rope_dim / 2)
        .map(|_| 1.0 + next_byte(&mut seed) as f32 / 255.0)
        .collect();

    let mut expected = x.clone();
    crate::engine::tensor::rmsnorm_inplace(&mut expected, &weight, n_head, head_dim, eps);
    crate::engine::tensor::rope_apply_scaled_inplace(
        &mut expected,
        n_head,
        head_dim,
        rope_dim,
        pos,
        freq_base,
        Some(&freq_factors),
    );

    let got = vulkan.gpu_fused_norm_rope(GpuFusedNormRopeInput {
        yarn: RopeYarn::IDENTITY,
        pairing: crate::engine::tensor::RopeLayout::Neox,
        x: &x,
        weight: &weight,
        n_tokens: 1,
        n_head,
        head_dim,
        rope_dim,
        pos,
        freq_base,
        freq_factors: Some(&freq_factors),
        eps,
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

/// A whole prefill batch in one dispatch, against the exact CPU sequence
/// `GemmaModel::run_layers_cpu` runs for Q: one `rmsnorm_inplace` over
/// `n_tokens * n_head` rows, then a per-token `rope_apply_scaled_inplace`
/// at position `start_pos + t`.
///
/// The per-token position is the whole point — every row taking `pos + t`
/// rather than one shared `pos` is what the token dimension exists for, and
/// a `start_pos > 0` makes sure the offset is applied rather than `t` being
/// used as the position outright. `rope_dim < head_dim` and `freq_factors`
/// are both set so the batch case covers the partial-rope tail and the
/// proportional-RoPE divisor at the same time, since the full-attention
/// layers this will serve have both.
/// The `llama`/`mistral` pairing through the **fused** norm+rope kernel.
///
/// A separate test from `gpu_rope_matches_cpu_reference_with_norm_pairing`
/// because these are two different shaders: `ROPE_SHADER` is the standalone
/// one, `FUSED_NORM_ROPE_SHADER` is what the QKV fusion actually reaches.
/// Teaching one the convention and assuming the other would have left the
/// path that matters silently NEOX — which is how this was nearly shipped.
#[test]
fn gpu_fused_norm_rope_matches_cpu_reference_with_norm_pairing() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    let (n_head, head_dim, rope_dim, pos, freq_base, eps) =
        (3usize, 32usize, 32usize, 5usize, 1e4f32, 1e-6f32);
    let mut st = 0x9E37_79B9u64;
    let mut rand_vec = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| {
                st = st.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                ((st >> 33) as f32 / 2f32.powi(31)) - 1.0
            })
            .collect()
    };
    let x = rand_vec(n_head * head_dim);
    let weight: Vec<f32> = rand_vec(head_dim).iter().map(|v| 1.0 + v * 0.1).collect();

    // Reference: the CPU per-head RMSNorm followed by the CPU NORM-pairing
    // RoPE — two functions that share nothing with this shader.
    let mut expected = x.clone();
    crate::engine::tensor::rmsnorm_inplace(&mut expected, &weight, n_head, head_dim, eps);
    crate::engine::tensor::rope_apply_params_inplace(
        &mut expected,
        n_head,
        head_dim,
        pos,
        None,
        &crate::engine::tensor::RopeParams {
            rope_dim,
            freq_base,
            layout: crate::engine::tensor::RopeLayout::Norm,
            ..crate::engine::tensor::RopeParams::default()
        },
    );

    let mk = |pairing| GpuFusedNormRopeInput {
        yarn: RopeYarn::IDENTITY,
        x: &x,
        weight: &weight,
        n_tokens: 1,
        n_head,
        head_dim,
        rope_dim,
        pos,
        freq_base,
        freq_factors: None,
        eps,
        pairing,
    };
    let got = vulkan.gpu_fused_norm_rope(mk(crate::engine::tensor::RopeLayout::Norm));
    assert_eq!(got.len(), expected.len());
    for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
        assert!(
            (a - b).abs() <= 2e-4 * a.abs().max(1.0),
            "mismatch at {i}: cpu={a} gpu={b}"
        );
    }
    // And the two conventions must differ, or this passes against a shader
    // that ignores `pairing`.
    let neox = vulkan.gpu_fused_norm_rope(mk(crate::engine::tensor::RopeLayout::Neox));
    assert!(
        neox.iter()
            .zip(got.iter())
            .any(|(a, b)| (a - b).abs() > 1e-3),
        "NEOX and NORM produced the same tensor — `pairing` is being ignored"
    );
}

#[test]
fn gpu_fused_norm_rope_matches_cpu_reference_over_a_token_batch() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };

    let n_tokens = 7;
    let n_head = 3;
    let head_dim = 10;
    let rope_dim = 6;
    let start_pos = 5;
    let freq_base = 1_000_000.0;
    let eps = 1e-6;

    let mut seed = 0x5EED_1234_u64;
    let x: Vec<f32> = (0..n_tokens * n_head * head_dim)
        .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
        .collect();
    let weight: Vec<f32> = (0..head_dim)
        .map(|_| (next_byte(&mut seed) as f32) / 128.0)
        .collect();
    let freq_factors: Vec<f32> = (0..rope_dim / 2)
        .map(|_| 1.0 + next_byte(&mut seed) as f32 / 255.0)
        .collect();

    let mut expected = x.clone();
    crate::engine::tensor::rmsnorm_inplace(
        &mut expected,
        &weight,
        n_tokens * n_head,
        head_dim,
        eps,
    );
    for t in 0..n_tokens {
        let row = &mut expected[t * n_head * head_dim..(t + 1) * n_head * head_dim];
        crate::engine::tensor::rope_apply_scaled_inplace(
            row,
            n_head,
            head_dim,
            rope_dim,
            start_pos + t,
            freq_base,
            Some(&freq_factors),
        );
    }

    let got = vulkan.gpu_fused_norm_rope(GpuFusedNormRopeInput {
        yarn: RopeYarn::IDENTITY,
        pairing: crate::engine::tensor::RopeLayout::Neox,
        x: &x,
        weight: &weight,
        n_tokens,
        n_head,
        head_dim,
        rope_dim,
        pos: start_pos,
        freq_base,
        freq_factors: Some(&freq_factors),
        eps,
    });

    assert_eq!(expected.len(), got.len());
    for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
        let tol = 3e-3 * a.abs().max(1.0);
        assert!(
            (a - b).abs() <= tol,
            "mismatch at token {} head {} index {}: cpu={a} gpu={b}",
            i / (n_head * head_dim),
            (i / head_dim) % n_head,
            i % head_dim
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
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
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
    let mut expected = CpuBackend.matmul_dequant(&x, 1, &proj_w);
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
        0,
    );
    let got = vulkan.submit_and_readback(encoder, &buf, 0, total);

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
/// element wins" rule (see `ARGMAX_PENALTY_SHADER`'s own doc comment
/// for why matching it exactly was never worth the complexity), and
/// byte-quantized values collide often enough at real vocab sizes to
/// make ties a real test hazard, not just a theoretical one.
#[test]
fn record_argmax_sample_matches_cpu_reference() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
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

    // `None` exercises the no-softcap fast path (softcap phase skipped);
    // `Some(30.0)` (Gemma-2/4's `final_logit_softcapping`) exercises the
    // softcap phase and, crucially, its interaction with the
    // value-dependent repeat penalty — the whole reason softcap must run
    // *before* the penalty rather than being skipped as a monotonic no-op.
    for logit_softcap in [None, Some(30.0f32)] {
        for recent_tokens in &recent_cases {
            let logits: Vec<f32> = (0..n_vocab).map(|_| next_f32(&mut seed)).collect();

            // CPU reference: softcap (if any) → penalty → argmax, the same
            // order the GPU phases run in.
            let mut expected_logits = logits.clone();
            if let Some(cap) = logit_softcap {
                for v in expected_logits.iter_mut() {
                    *v = cap * (*v / cap).tanh();
                }
            }
            for &tok in recent_tokens {
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
                    recent_tokens,
                    repeat_penalty,
                    logit_softcap,
                },
                0,
            );
            let got = vulkan.submit_and_readback_u32(encoder, &buf);

            assert_eq!(
                expected, got,
                "softcap={logit_softcap:?} recent_tokens={recent_tokens:?}: \
                     cpu argmax={expected} gpu argmax={got}"
            );
        }
    }
}

/// Like `record_argmax_sample_matches_cpu_reference` above, but at a
/// vocabulary size (`300_000`, close to real `E2B`'s 262144) both
/// large enough that every one of `ARGMAX_SPLIT_N`'s workgroups has
/// real work (unlike the smaller test above, which also exercises the
/// opposite — mostly-empty — case) and not a multiple of `ARGMAX_
/// SPLIT_N * 64`, so the split shader's global-stride loop bounds are
/// exercised on an uneven remainder too. The winning logit is planted
/// at a handful of different positions across different split ranges
/// (not just position 0) so the test can't pass by accident if
/// `partial_val`/`partial_idx` ever got swapped or misindexed between
/// the split and merge phases.
#[test]
fn record_argmax_sample_matches_cpu_reference_at_a_large_uneven_vocab() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };

    let n_vocab = 300_000usize;
    let repeat_penalty = 1.1f32;
    let mut seed = 0x900D_u64;

    // Positions in three different split workgroups (`ARGMAX_SPLIT_N
    // == 256`, so workgroup boundaries land roughly every ~1172
    // elements of `n_vocab / 256`) plus one right at the very end, to
    // cover the uneven-remainder tail.
    for winner in [5usize, 100_000, 210_777, n_vocab - 1] {
        let mut logits = vec![0f32; n_vocab];
        for v in logits.iter_mut() {
            *v = ((next_byte(&mut seed) as f32 - 128.0) / 64.0).min(3.9);
        }
        logits[winner] = 4.0; // strictly greater than every other value above

        let mut encoder = vulkan.new_encoder("test argmax sample large encoder");
        let buf = vulkan.record_argmax_sample(
            &mut encoder,
            GpuArgmaxSampleInput {
                logits: GpuInput::Cpu(&logits),
                n_vocab,
                recent_tokens: &[],
                repeat_penalty,
                logit_softcap: None,
            },
            // A distinct sample-cache slot from the other argmax test, so
            // the two running in parallel on the shared backend don't
            // thrash one slot-0 entry between their different `n_vocab`s.
            1,
        );
        let got = vulkan.submit_and_readback_u32(encoder, &buf);

        assert_eq!(
            winner as u32, got,
            "n_vocab={n_vocab}: expected winner at {winner}, gpu argmax={got}"
        );
    }
}

/// Cross-checks `gpu_perhead_rmsnorm_weightless` (V's norm) against
/// the same weightless-RMSNorm formula `GemmaModel`'s private
/// `rmsnorm_weightless_inplace` uses (mean-of-squares, no learned
/// scale) — replicated inline here since that helper isn't `pub`.
#[test]
fn gpu_perhead_rmsnorm_weightless_matches_cpu_reference() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
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
    cross_check_fused_attention_owns_v(32, 4, 2, 8, 8);
}

/// The served model's two attention shapes — eight query heads over one
/// KV head, 256 wide on the sliding-window layers and 512 wide with a
/// 512-wide RoPE on the full-attention ones. The small shape above fits in
/// one `vec4` per lane of the wide per-head kernels; these fill both slots
/// (`vulkan_shaders::HEAD_WIDE_MAX_DIM`) and run the rotation loop more
/// than once per lane.
#[test]
fn fused_attention_matches_cpu_reference_at_the_sliding_window_shape() {
    cross_check_fused_attention_owns_v(64, 8, 1, 256, 256);
}

#[test]
fn fused_attention_matches_cpu_reference_at_the_full_attention_shape() {
    cross_check_fused_attention_owns_v(64, 8, 1, 512, 512);
}

fn cross_check_fused_attention_owns_v(
    n_embd: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    rope_dim: usize,
) {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };

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
    let mut reference_cache = crate::engine::kv_cache::KvCache::new_with_dims(capacity, &[kv_dim]);
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
        let mut q = CpuBackend.matmul_dequant(&normed, 1, &wq);
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
        let mut k = CpuBackend.matmul_dequant(&normed, 1, &wk);
        crate::engine::tensor::rmsnorm_inplace(&mut k, &k_norm, n_head_kv, head_dim, eps);
        let mut v = CpuBackend.matmul_dequant(&normed, 1, &wv);
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
            projections_ready: false,
            yarn: RopeYarn::IDENTITY,
            normalize_v: true,
            attn_gate: None,
            q_bias: None,
            pairing: crate::engine::tensor::RopeLayout::Neox,
            normed: GpuInput::Cpu(&normed),
            normed_q8: None,
            wq: &wq,
            q_norm: Some(&q_norm),
            kv: Some(FusedAttnProjection {
                k_bias: None,
                v_bias: None,
                wk: &wk,
                k_norm: Some(&k_norm),
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
            window: None,
            scale,
            cache: &mut kv_cache.layers[0],
            batch_slot: 0,
            attn_ts: None,
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

/// Same cross-check as the one above, but with `head_dim = 32` so
/// `kv_dim` is a multiple of 32 — the shape `KvStorage::Q8_0`'s block
/// format requires (every other `fused_attention`/`fused_layer` test
/// in this module uses a smaller, non-block-aligned `kv_dim`, so only
/// this one is meaningful to re-run with `ORANGU_KV_Q8_0=1` set before
/// the test binary starts). This exercises `record_fused_attention`'s
/// actual per-decode-step KV-cache write path (the quantize-on-write
/// dispatch, not just `gpu_attention`'s simpler standalone entry
/// point), across two sequential steps so the write offset advances
/// past the first block too. Wider tolerance than the sibling test,
/// same reasoning as `gpu_attention_matches_cpu_reference_kv_dim_32`.
#[test]
fn fused_attention_matches_cpu_reference_kv_dim_32() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };

    let n_embd = 32;
    let n_head = 4;
    let n_head_kv = 2;
    let head_dim = 32;
    let rope_dim = 32;
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

    let mut kv_cache = crate::engine::kv_cache::KvCache::new_with_dims(capacity, &[kv_dim]);
    let mut reference_cache = crate::engine::kv_cache::KvCache::new_with_dims(capacity, &[kv_dim]);
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

        let mut q = CpuBackend.matmul_dequant(&normed, 1, &wq);
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
        let mut k = CpuBackend.matmul_dequant(&normed, 1, &wk);
        crate::engine::tensor::rmsnorm_inplace(&mut k, &k_norm, n_head_kv, head_dim, eps);
        let mut v = CpuBackend.matmul_dequant(&normed, 1, &wv);
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
            projections_ready: false,
            yarn: RopeYarn::IDENTITY,
            normalize_v: true,
            attn_gate: None,
            q_bias: None,
            pairing: crate::engine::tensor::RopeLayout::Neox,
            normed: GpuInput::Cpu(&normed),
            normed_q8: None,
            wq: &wq,
            q_norm: Some(&q_norm),
            kv: Some(FusedAttnProjection {
                k_bias: None,
                v_bias: None,
                wk: &wk,
                k_norm: Some(&k_norm),
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
            window: None,
            scale,
            cache: &mut kv_cache.layers[0],
            batch_slot: 0,
            attn_ts: None,
        });

        assert_eq!(expected.len(), got.len());
        for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
            let tol = 1.5e-1 * a.abs().max(1.0);
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
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
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

    let mut q = CpuBackend.matmul_dequant(&normed, 1, &wq);
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
    let mut k = CpuBackend.matmul_dequant(&normed, 1, &wk);
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
        projections_ready: false,
        yarn: RopeYarn::IDENTITY,
        normalize_v: true,
        attn_gate: None,
        q_bias: None,
        pairing: crate::engine::tensor::RopeLayout::Neox,
        normed: GpuInput::Cpu(&normed),
        normed_q8: None,
        wq: &wq,
        q_norm: Some(&q_norm),
        kv: Some(FusedAttnProjection {
            k_bias: None,
            v_bias: None,
            wk: &wk,
            k_norm: Some(&k_norm),
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
        window: None,
        scale,
        cache: &mut kv_cache.layers[0],
        batch_slot: 0,
        attn_ts: None,
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
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
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
                crate::engine::tensor::dot(qh, reference.layers[0].key_at(0, kv_head, head_dim))
                    * scale,
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
        let mut q = CpuBackend.matmul_dequant(normed, 1, wq);
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
    let mut k_a = CpuBackend.matmul_dequant(&normed_a, 1, &wk);
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
    let mut reference_cache = crate::engine::kv_cache::KvCache::new_with_dims(capacity, &[kv_dim]);
    reference_cache.layers[0].push(&k_a, &v_a);

    let q_a = cpu_q(&wq_a, &q_norm_a, &normed_a, 0);
    let expected_a = expected_attn(&q_a, &reference_cache);
    let got_a = vulkan.fused_attention(FusedAttnInput {
        projections_ready: false,
        yarn: RopeYarn::IDENTITY,
        normalize_v: true,
        attn_gate: None,
        q_bias: None,
        pairing: crate::engine::tensor::RopeLayout::Neox,
        normed: GpuInput::Cpu(&normed_a),
        normed_q8: None,
        wq: &wq_a,
        q_norm: Some(&q_norm_a),
        kv: Some(FusedAttnProjection {
            k_bias: None,
            v_bias: None,
            wk: &wk,
            k_norm: Some(&k_norm),
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
        window: None,
        scale,
        cache: &mut kv_cache.layers[0],
        batch_slot: 0,
        attn_ts: None,
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
        projections_ready: false,
        yarn: RopeYarn::IDENTITY,
        normalize_v: true,
        attn_gate: None,
        q_bias: None,
        pairing: crate::engine::tensor::RopeLayout::Neox,
        normed: GpuInput::Cpu(&normed_b),
        normed_q8: None,
        wq: &wq_b,
        q_norm: Some(&q_norm_b),
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
        window: None,
        scale,
        cache: &mut kv_cache.layers[0],
        batch_slot: 0,
        attn_ts: None,
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

/// Cross-checks `fused_ffn_prefill` — a prefill layer's gate/up/GEGLU/
/// down block in one submission — against the exact sequence it replaces:
/// the same two GPU matmuls, `gelu` and `mul` on the CPU, then the same
/// GPU down matmul. Comparing against *that* rather than against a pure
/// CPU reference isolates what the fusion changed (where the intermediate
/// lives, and how many submissions carry it) from what the matmul kernels
/// themselves do, which their own cross-checks already cover.
///
/// Run at two token counts on purpose: 3 is below the cooperative
/// dispatch's `COOP_MIN_N_TOKENS` crossover, 192 is above it *and* past
/// `MAX_MATMUL_TOKENS_PER_SUBMISSION`, so the fused recorder is exercised
/// against both matmul kernels and against the token-range chunking. 192
/// splits into 128 + 64 with no stripe padding, deliberately: these
/// weights are random `Q4_K` blocks with random `f16` scales, which drive
/// the projections to ~1e12, and at that magnitude GELU acts as a step —
/// two kernels differing in the last bits produce wildly different
/// outputs. Padding switches the tail to a different matmul kernel, so it
/// is verified where the arithmetic is well-conditioned instead, by
/// `padding_a_stripe_leaves_its_real_rows_unchanged`.
fn cross_check_fused_ffn_prefill(n_tokens: usize) {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    if vulkan.q4_k_mmvq {
        eprintln!("skipping: ORANGU_Q4K_MMVQ selects the unfused fallback path");
        return;
    }

    let (n_embd, ffn_len) = (256usize, 512usize);
    let mut seed = 0xFFA1_u64;
    let mut build = |in_dim: usize, out_dim: usize| {
        let mut bytes = Vec::new();
        for _ in 0..out_dim {
            for _ in 0..(in_dim / 256) {
                bytes.extend(build_block(GGML_TYPE_Q4_K, &mut seed));
            }
        }
        test_quant_matrix(&bytes, GGML_TYPE_Q4_K, in_dim, out_dim)
    };
    let gate = build(n_embd, ffn_len);
    let up = build(n_embd, ffn_len);
    let down = build(ffn_len, n_embd);

    let x: Vec<f32> = (0..n_tokens * n_embd)
        .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
        .collect();

    // The unfused sequence, step for step.
    let mut expected = vulkan.matmul(&x, n_tokens, &gate);
    let up_out = vulkan.matmul(&x, n_tokens, &up);
    crate::engine::tensor::gelu_inplace(&mut expected);
    crate::engine::tensor::mul_inplace(&mut expected, &up_out);
    let expected = vulkan.matmul(&expected, n_tokens, &down);

    let got = vulkan
        .fused_ffn_prefill(
            &x,
            n_tokens,
            &gate,
            &up,
            &down,
            crate::engine::backend::vulkan::FfnActivation::Geglu,
            None,
        )
        .expect("fused path available without MMVQ");

    assert_eq!(got.len(), expected.len());
    assert_eq!(got.len(), n_tokens * n_embd);
    for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
        // The GPU GEGLU kernel and the CPU `gelu` are separate
        // implementations of the same function, so this is a closeness
        // check, not bit-equality — the same tolerance shape the fused
        // decode-layer cross-check uses.
        let tol = 6e-2 * a.abs().max(1.0);
        assert!(
            (a - b).abs() <= tol,
            "n_tokens={n_tokens}: mismatch at {i}: unfused={a} fused={b}"
        );
    }
}

/// The PLE counterpart of [`cross_check_fused_ffn_prefill`]: same
/// fusion shape, but the multiply's second operand is model input rather
/// than a second projection, so it also pins down that the per-layer
/// block is uploaded and indexed per token the same way the unfused
/// per-token `mul_inplace` loop reads it.
fn cross_check_fused_ple_prefill(n_tokens: usize) {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    if vulkan.q4_k_mmvq {
        eprintln!("skipping: ORANGU_Q4K_MMVQ selects the unfused fallback path");
        return;
    }

    let (n_embd, per_layer_dim) = (256usize, 256usize);
    let mut seed = 0x9E11_u64;
    let mut build = |in_dim: usize, out_dim: usize| {
        let mut bytes = Vec::new();
        for _ in 0..out_dim {
            for _ in 0..(in_dim / 256) {
                bytes.extend(build_block(GGML_TYPE_Q4_K, &mut seed));
            }
        }
        test_quant_matrix(&bytes, GGML_TYPE_Q4_K, in_dim, out_dim)
    };
    let gate = build(n_embd, per_layer_dim);
    let proj = build(per_layer_dim, n_embd);

    let x: Vec<f32> = (0..n_tokens * n_embd)
        .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
        .collect();
    let per_layer: Vec<f32> = (0..n_tokens * per_layer_dim)
        .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
        .collect();

    let mut g = vulkan.matmul(&x, n_tokens, &gate);
    crate::engine::tensor::gelu_inplace(&mut g);
    for t in 0..n_tokens {
        let slice = &per_layer[t * per_layer_dim..(t + 1) * per_layer_dim];
        crate::engine::tensor::mul_inplace(
            &mut g[t * per_layer_dim..(t + 1) * per_layer_dim],
            slice,
        );
    }
    let expected = vulkan.matmul(&g, n_tokens, &proj);

    let got = vulkan
        .fused_ple_prefill(&x, n_tokens, &gate, &proj, &per_layer)
        .expect("fused path available without MMVQ");

    assert_eq!(got.len(), expected.len());
    for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
        let tol = 6e-2 * a.abs().max(1.0);
        assert!(
            (a - b).abs() <= tol,
            "n_tokens={n_tokens}: mismatch at {i}: unfused={a} fused={b}"
        );
    }
}

#[test]
fn fused_ple_prefill_matches_the_unfused_sequence_small() {
    cross_check_fused_ple_prefill(3);
}

#[test]
fn fused_ple_prefill_matches_the_unfused_sequence_multi_chunk() {
    // At this width the unfused sequence's matmuls take the integer-dot
    // kernel through the generic batch path, and the chain quantizes its
    // intermediates at different points than the CPU sequence does; on
    // synthetic weights with outputs in the billions that is not a 6%
    // comparison. The chain's structure is what this checks, on the float
    // kernels; the integer-dot kernel has its own test.
    if let Some(vulkan) = shared_vulkan() {
        vulkan.without_prefill_mmq(|| cross_check_fused_ple_prefill(192));
    }
}

/// Cross-checks `fused_post_attention_prefill` — `wo`, the attention
/// residual, the FFN norm, gate/up/GEGLU/down, and the FFN residual in one
/// submission — against the exact CPU-orchestrated sequence it replaces,
/// step for step with the same GPU matmuls and the same CPU norms and
/// adds in between. Both residual paths matter here: `x1` feeds the FFN
/// norm *and* the final add, so a chain that overwrote it would still look
/// right for one of the two.
/// Cross-checks `fused_attention_prefill` against the exact unfused
/// sequence it replaces — `matmul_batch` for Q/K/V, the CPU's per-head
/// norms and RoPE, the per-token `LayerCache::push`, then
/// `gpu_attention_prefill` — using the same GPU matmul and attention
/// kernels, with only the norms/RoPE/cache-write moving.
///
/// `owns_v` picks between the two K stagings, and getting that wrong is
/// silent: with `owns_v == false`, V must be a copy of K taken *after* its
/// norm and *before* its RoPE, so the fused K kernel cannot be used.
///
/// Also checks the K/V rows the GPU hands back for the host mirror against
/// what the CPU path pushed, since those are what slot save serializes.
fn cross_check_fused_attention_prefill(n_tokens: usize, owns_v: bool, start_pos: usize) {
    cross_check_fused_attention_prefill_paged(n_tokens, owns_v, start_pos, false);
}

/// The chain's **deferred** form (`fused_attention_prefill_deferred` +
/// `fill_kv_rows`), against the same unfused reference: attention's output
/// stays on the device and is read back for the comparison, and the K/V
/// rows reach the host only through the fill — so what this checks is that
/// the cache ends up holding, row for row, what the in-order path pushes.
fn cross_check_fused_attention_prefill_deferred(n_tokens: usize, start_pos: usize, paged: bool) {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    vulkan.without_prefill_mmq(|| {
        cross_check_fused_attention_prefill_paged_float_mode(
            vulkan, n_tokens, true, start_pos, paged, true,
        )
    });
}

/// `paged` runs the fused recorder against a cache backed by the page pool
/// instead of a per-request mirror, comparing against the same unfused
/// reference. The reference is deliberately *not* paged: a paged reference
/// would share the component under test.
fn cross_check_fused_attention_prefill_paged(
    n_tokens: usize,
    owns_v: bool,
    start_pos: usize,
    paged: bool,
) {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    // An exact comparison against the float sequence: the integer-dot GEMM
    // is checked on its own (`mmq_q4k_gemm_matches_the_cpu_product`).
    vulkan.without_prefill_mmq(|| {
        cross_check_fused_attention_prefill_paged_float(vulkan, n_tokens, owns_v, start_pos, paged)
    });
}

fn cross_check_fused_attention_prefill_paged_float(
    vulkan: &VulkanBackend,
    n_tokens: usize,
    owns_v: bool,
    start_pos: usize,
    paged: bool,
) {
    cross_check_fused_attention_prefill_paged_float_mode(
        vulkan, n_tokens, owns_v, start_pos, paged, false,
    )
}

fn cross_check_fused_attention_prefill_paged_float_mode(
    vulkan: &VulkanBackend,
    n_tokens: usize,
    owns_v: bool,
    start_pos: usize,
    paged: bool,
    deferred: bool,
) {
    if vulkan.q4_k_mmvq {
        eprintln!("skipping: ORANGU_Q4K_MMVQ selects the unfused fallback path");
        return;
    }

    let (n_embd, n_head, n_head_kv, head_dim) = (256usize, 4usize, 2usize, 64usize);
    let (rope_dim, rope_freq_base, eps) = (64usize, 10000.0f32, 1e-6f32);
    let kv_dim = n_head_kv * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut seed = 0x0FA5_7ADD_u64;
    let mut build = |in_dim: usize, out_dim: usize| {
        let mut bytes = Vec::new();
        for _ in 0..out_dim {
            for _ in 0..(in_dim / 256) {
                bytes.extend(build_block(GGML_TYPE_Q4_K, &mut seed));
            }
        }
        test_quant_matrix(&bytes, GGML_TYPE_Q4_K, in_dim, out_dim)
    };
    let wq = build(n_embd, n_head * head_dim);
    let wk = build(n_embd, kv_dim);
    let wv = owns_v.then(|| build(n_embd, kv_dim));

    let mut rand_vec = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
            .collect()
    };
    let normed = rand_vec(n_tokens * n_embd);
    let q_norm: Vec<f32> = rand_vec(head_dim).iter().map(|v| 1.0 + v * 0.1).collect();
    let k_norm: Vec<f32> = rand_vec(head_dim).iter().map(|v| 1.0 + v * 0.1).collect();

    let capacity = start_pos + n_tokens + 8;
    let prior: Vec<(Vec<f32>, Vec<f32>)> = (0..start_pos)
        .map(|_| (rand_vec(kv_dim), rand_vec(kv_dim)))
        .collect();

    // ---- the unfused sequence ----
    let mut ref_cache = crate::engine::kv_cache::KvCache::new_with_dims(capacity, &[kv_dim]);
    for (k, v) in &prior {
        ref_cache.layers[0].push(k, v);
    }
    let mut ops = vec![
        MatmulOp {
            x: &normed,
            n_tokens,
            w: &wq,
        },
        MatmulOp {
            x: &normed,
            n_tokens,
            w: &wk,
        },
    ];
    if let Some(wv) = &wv {
        ops.push(MatmulOp {
            x: &normed,
            n_tokens,
            w: wv,
        });
    }
    let mut results = vulkan.matmul_batch(&ops).into_iter();
    let mut q = results.next().unwrap();
    crate::engine::tensor::rmsnorm_inplace(&mut q, &q_norm, n_tokens * n_head, head_dim, eps);
    for t in 0..n_tokens {
        crate::engine::tensor::rope_apply_scaled_inplace(
            &mut q[t * n_head * head_dim..(t + 1) * n_head * head_dim],
            n_head,
            head_dim,
            rope_dim,
            start_pos + t,
            rope_freq_base,
            None,
        );
    }
    let mut k = results.next().unwrap();
    crate::engine::tensor::rmsnorm_inplace(&mut k, &k_norm, n_tokens * n_head_kv, head_dim, eps);
    let mut v = match results.next() {
        Some(v) => v,
        None => k.clone(),
    };
    crate::engine::arch::gemma::rmsnorm_weightless_inplace(
        &mut v,
        n_tokens * n_head_kv,
        head_dim,
        eps,
    );
    for t in 0..n_tokens {
        crate::engine::tensor::rope_apply_scaled_inplace(
            &mut k[t * kv_dim..(t + 1) * kv_dim],
            n_head_kv,
            head_dim,
            rope_dim,
            start_pos + t,
            rope_freq_base,
            None,
        );
    }
    for t in 0..n_tokens {
        ref_cache.layers[0].push(
            &k[t * kv_dim..(t + 1) * kv_dim],
            &v[t * kv_dim..(t + 1) * kv_dim],
        );
    }
    let expected = vulkan.gpu_attention_prefill(
        &q,
        &mut ref_cache.layers[0],
        start_pos,
        n_tokens,
        n_head,
        n_head_kv,
        head_dim,
        0,
        true,
        scale,
    );

    // ---- the fused recorder ----
    // Held across the call so the pool outlives the cache that borrows it.
    let mut held_pool = None;
    let mut cache = if paged {
        use crate::engine::kv_pool::{KvPool, LayerGeometry, Policy};
        const PAGE: usize = 8;
        let pool_pages = capacity.div_ceil(PAGE) * 4;
        let mut pool = KvPool::with_policy(
            pool_pages,
            PAGE,
            vec![LayerGeometry {
                kv_dim,
                stride: 1,
                ring: None,
            }],
            Policy::Lru,
        );
        let (device, queue) = vulkan.device_and_queue();
        assert!(pool.attach_device(device, vulkan.kv_storage(), pool_pages * 4));
        let pool = std::sync::Arc::new(pool);
        // **The sequence's pages must be neither low nor adjacent.**
        //
        // Holding the low half alone is not enough here, and the difference is
        // what a mutation found: the pool hands out pages in ascending order,
        // so a sequence gets a *consecutive* run, and a write that failed to
        // split at a page boundary ran straight through into the next page —
        // which is exactly where those rows belonged anyway. The test passed
        // while proving nothing about the split.
        //
        // Taking everything and giving back only alternate pages leaves the
        // sequence with a run that is both high and non-adjacent, so writing
        // past a page lands in a held page full of junk and both the write
        // split and the block table have to be right.
        let all = pool.alloc(pool_pages).expect("pool has room");
        let (given, held): (Vec<u32>, Vec<u32>) = all.iter().partition(|p| !(*p).is_multiple_of(2));
        pool.release(&given);
        for &physical in &held {
            let junk: Vec<f32> = (0..PAGE * kv_dim)
                .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 32.0)
                .collect();
            pool.fill_device(queue, 0, physical, &junk, &junk);
        }
        held_pool = Some((pool.clone(), held));
        crate::engine::kv_cache::KvCache::new_with_strided_dims(capacity, &strided_dims(&pool))
            .try_into_paged(pool)
            .unwrap_or_else(|_| panic!("test pool has room"))
    } else {
        crate::engine::kv_cache::KvCache::new_with_dims(capacity, &[kv_dim])
    };
    for (pk, pv) in &prior {
        cache.layers[0].push(pk, pv);
    }
    if paged {
        cache.commit_pages();
    }
    let input = FusedAttnPrefillInput {
        x_gpu: None,
        attn_norm: None,
        yarn: RopeYarn::IDENTITY,
        q_bias: None,
        pairing: crate::engine::tensor::RopeLayout::Neox,
        normalize_v: true,
        attn_gate: None,
        normed: &normed,
        n_tokens,
        start_pos,
        wq: &wq,
        q_norm: Some(&q_norm),
        kv: Some(FusedAttnPrefillKv {
            k_bias: None,
            v_bias: None,
            wk: &wk,
            k_norm: Some(&k_norm),
            wv: wv.as_ref(),
        }),
        n_head,
        n_head_kv,
        head_dim,
        rope_dim,
        rope_freq_base,
        freq_factors: None,
        eps,
        n_swa: 0,
        causal: true,
        scale,
        want_attn_out_host: !deferred,
    };
    let got = if deferred {
        let mut stage = vulkan.kv_readback_stage((n_tokens * kv_dim * 2 * 4) as u64);
        let (mut out, pending) = vulkan
            .fused_attention_prefill_deferred(input, &mut cache.layers[0], &mut stage)
            .expect("fused prefill attention returned None on a supported path");
        assert!(
            !pending.is_empty(),
            "a KV-owning layer leaves rows in flight"
        );
        assert_eq!(
            cache.layers[0].pending_rows(),
            n_tokens,
            "every position is counted before its rows arrive"
        );
        vulkan.end_prefill_group();
        vulkan.fill_kv_rows(stage, vec![(0, pending)], &mut cache);
        assert_eq!(cache.layers[0].pending_rows(), 0);
        out.attn_out = vulkan.readback_rows(&out.attn_out_buf, n_tokens * n_head * head_dim);
        for pos in start_pos..start_pos + n_tokens {
            let (gk, gv) = cache.layers[0].host_row(pos);
            out.k_rows.extend_from_slice(gk);
            out.v_rows.extend_from_slice(gv);
        }
        out
    } else {
        vulkan
            .fused_attention_prefill(input, &mut cache.layers[0])
            .expect("fused prefill attention returned None on a supported path")
    };

    let cmp = |label: &str, a: &[f32], b: &[f32]| {
        assert_eq!(a.len(), b.len(), "{label}: length");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() <= 3e-3 * x.abs().max(1.0),
                "{label} mismatch at {i} (n_tokens={n_tokens} owns_v={owns_v}): \
                     unfused={x} fused={y}"
            );
        }
    };
    if paged {
        assert!(
            cache.layers[0].is_pool_backed(),
            "the paged prefill path was not taken; this would be comparing the \
             mirrored fallback with itself"
        );
    }
    cmp("attn_out", &expected, &got.attn_out);
    // The host mirror the fused path leaves behind must match what the CPU
    // path pushed — this is what slot save serializes.
    cmp("k_rows", &k, &got.k_rows);
    cmp("v_rows", &v, &got.v_rows);
    assert_eq!(cache.layers[0].len, ref_cache.layers[0].len);
    if let Some((pool, held)) = held_pool {
        drop(cache);
        pool.release(&held);
    }
}

/// A **ring mirror** (`LayerCache::set_mirror_ring`) against the plain
/// one: the same rows, the same queries, the same kernels — only the row a
/// position lives at differs — so every output must be bit-identical. The
/// ring is sized so it wraps several times over the test: the fused prefill
/// chain writing chunks through the wrap, the single-query kernel reading
/// a window that straddles it, and the host-path prefill kernel over a
/// window that does too. A ring that never wrapped would prove nothing.
#[test]
fn ring_mirror_matches_the_plain_mirror_across_the_wrap() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    if vulkan.q4_k_mmvq {
        eprintln!("skipping: ORANGU_Q4K_MMVQ selects the unfused fallback path");
        return;
    }
    let (n_embd, n_head, n_head_kv, head_dim) = (256usize, 4usize, 2usize, 64usize);
    let (rope_dim, rope_freq_base, eps) = (64usize, 10000.0f32, 1e-6f32);
    let kv_dim = n_head_kv * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let n_swa = 5usize;
    let chunk = 8usize;
    let capacity = 96usize;
    let mut seed = 0x0051_D1E5_u64;
    let mut build = |in_dim: usize, out_dim: usize| {
        let mut bytes = Vec::new();
        for _ in 0..out_dim {
            for _ in 0..(in_dim / 256) {
                bytes.extend(build_block(GGML_TYPE_Q4_K, &mut seed));
            }
        }
        test_quant_matrix(&bytes, GGML_TYPE_Q4_K, in_dim, out_dim)
    };
    let wq = build(n_embd, n_head * head_dim);
    let wk = build(n_embd, kv_dim);
    let wv = build(n_embd, kv_dim);
    let mut rand_vec = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
            .collect()
    };
    let q_norm: Vec<f32> = rand_vec(head_dim).iter().map(|v| 1.0 + v * 0.1).collect();
    let k_norm: Vec<f32> = rand_vec(head_dim).iter().map(|v| 1.0 + v * 0.1).collect();

    let mut plain = crate::engine::kv_cache::KvCache::new_with_dims(capacity, &[kv_dim]);
    let mut ring = crate::engine::kv_cache::KvCache::new_with_dims(capacity, &[kv_dim]);
    ring.set_mirror_ring(0, n_swa, chunk);
    // 5 + 8 rounds up to 16 rows: the ring wraps every 16 positions.
    assert_eq!(ring.layers[0].ring_rows(), Some(16));
    assert_eq!(ring.max_positions_per_call(), Some(16 - n_swa));
    // The same ring layer served by a page pool: its rows go to the pool's
    // host pages (for sharing and slot save) and its device copy stays this
    // request's ring — the pool allocates it no device pages.
    let mut probe = crate::engine::kv_cache::KvCache::new_with_dims(1, &[kv_dim]);
    probe.set_mirror_ring(0, n_swa, chunk);
    let geom = crate::engine::kv_pool::LayerGeometry::of(&probe);
    assert_eq!(geom[0].ring, Some(16));
    let pool_pages = 64;
    let mut pool = crate::engine::kv_pool::KvPool::with_policy(
        pool_pages,
        8,
        geom.clone(),
        crate::engine::kv_pool::Policy::Lru,
    );
    let (device, _) = vulkan.device_and_queue();
    assert!(pool.attach_device(device, vulkan.kv_storage(), pool_pages * 4));
    assert_eq!(
        crate::engine::kv_pool::device_page_bytes(&geom, 8, vulkan.kv_storage()),
        0,
        "a ring layer costs the pool no device bytes"
    );
    let pool = std::sync::Arc::new(pool);
    let mut paged_ring = crate::engine::kv_cache::KvCache::new_with_dims(capacity, &[kv_dim]);
    paged_ring.set_mirror_ring(0, n_swa, chunk);
    let mut paged_ring = paged_ring
        .try_into_paged(pool.clone())
        .unwrap_or_else(|_| panic!("test pool has room"));
    assert_eq!(paged_ring.layers[0].ring_rows(), Some(16));

    let exact = |label: &str, a: &[f32], b: &[f32]| {
        assert_eq!(a.len(), b.len(), "{label}: length");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(x == y, "{label}: plain={x} ring={y} at {i}");
        }
    };

    // Chunks through the fused chain, the window bounded as a gemma
    // sliding-window layer's is.
    let mut pos = 0usize;
    for chunk_index in 0..5 {
        let n_tokens = if chunk_index % 2 == 0 { chunk } else { 3 };
        let normed = rand_vec(n_tokens * n_embd);
        let input = |cache: &mut crate::engine::kv_cache::KvCache| {
            let input = FusedAttnPrefillInput {
                x_gpu: None,
                attn_norm: None,
                yarn: RopeYarn::IDENTITY,
                q_bias: None,
                pairing: crate::engine::tensor::RopeLayout::Neox,
                normalize_v: true,
                attn_gate: None,
                normed: &normed,
                n_tokens,
                start_pos: pos,
                wq: &wq,
                q_norm: Some(&q_norm),
                kv: Some(FusedAttnPrefillKv {
                    k_bias: None,
                    v_bias: None,
                    wk: &wk,
                    k_norm: Some(&k_norm),
                    wv: Some(&wv),
                }),
                n_head,
                n_head_kv,
                head_dim,
                rope_dim,
                rope_freq_base,
                freq_factors: None,
                eps,
                n_swa,
                causal: true,
                scale,
                want_attn_out_host: true,
            };
            vulkan
                .fused_attention_prefill(input, &mut cache.layers[0])
                .expect("fused prefill attention returned None on a supported path")
        };
        let a = input(&mut plain);
        let b = input(&mut ring);
        let c = input(&mut paged_ring);
        paged_ring.commit_pages();
        exact(
            &format!("chunk {chunk_index} attn_out"),
            &a.attn_out,
            &b.attn_out,
        );
        exact(&format!("chunk {chunk_index} k_rows"), &a.k_rows, &b.k_rows);
        exact(
            &format!("chunk {chunk_index} paged attn_out"),
            &a.attn_out,
            &c.attn_out,
        );
        exact(
            &format!("chunk {chunk_index} paged k_rows"),
            &a.k_rows,
            &c.k_rows,
        );
        pos += n_tokens;
        assert_eq!(plain.layers[0].len, pos);
        assert_eq!(ring.layers[0].len, pos);
        assert_eq!(paged_ring.layers[0].len, pos);
    }
    assert!(
        !paged_ring.layers[0].is_pool_backed(),
        "a ring layer's device copy is never the pool's"
    );

    // Single queries at the wrap: the window straddles rows 15|0.
    for _ in 0..6 {
        let k = rand_vec(kv_dim);
        let v = rand_vec(kv_dim);
        plain.layers[0].push(&k, &v);
        ring.layers[0].push(&k, &v);
        paged_ring.layers[0].push(&k, &v);
        let q = rand_vec(n_head * head_dim);
        let window_start = pos.saturating_sub(n_swa - 1);
        let run = |cache: &mut crate::engine::kv_cache::KvCache| {
            vulkan.gpu_attention(GpuAttentionInput {
                q: &q,
                cache: &mut cache.layers[0],
                pos,
                window_start,
                n_head,
                n_head_kv,
                head_dim,
                scale,
            })
        };
        let a = run(&mut plain);
        let b = run(&mut ring);
        let c = run(&mut paged_ring);
        exact(&format!("single query at {pos}"), &a, &b);
        exact(&format!("paged single query at {pos}"), &a, &c);
        pos += 1;
    }

    // The host-path prefill kernel over rows pushed from the host.
    let n_tokens = 7;
    let mut q = Vec::new();
    for _ in 0..n_tokens {
        let k = rand_vec(kv_dim);
        let v = rand_vec(kv_dim);
        plain.layers[0].push(&k, &v);
        ring.layers[0].push(&k, &v);
        paged_ring.layers[0].push(&k, &v);
        q.extend(rand_vec(n_head * head_dim));
    }
    let run = |cache: &mut crate::engine::kv_cache::KvCache| {
        vulkan.gpu_attention_prefill(
            &q,
            &mut cache.layers[0],
            pos,
            n_tokens,
            n_head,
            n_head_kv,
            head_dim,
            n_swa,
            true,
            scale,
        )
    };
    let a = run(&mut plain);
    let b = run(&mut ring);
    let c = run(&mut paged_ring);
    exact("host-path prefill", &a, &b);
    exact("paged host-path prefill", &a, &c);
}

/// The `llama`/`mistral` shape through the same fused chain: **no** per-head
/// Q or K norm, and NORM rope pairing rather than NEOX. Both differ from
/// gemma's, both are load-bearing, and neither is visible from the
/// signature — which is why this is its own case rather than a parameter
/// tweak of the gemma one.
///
/// The reference is built from `engine::tensor`'s CPU RoPE and the CPU
/// attention, sharing no kernel with the chain under test.
fn cross_check_fused_attention_prefill_no_norms(n_tokens: usize, start_pos: usize) {
    cross_check_fused_attention_prefill_shaped(n_tokens, start_pos, "");
}

/// `biases` selects which projection biases to give the layer — `""` for
/// none, `"qkv"` for Qwen2's shape. Split per-projection because that is
/// how the bug was found: Q and K agree with the reference and V does not.
fn cross_check_fused_attention_prefill_shaped(n_tokens: usize, start_pos: usize, biases: &str) {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    vulkan.without_prefill_mmq(|| {
        cross_check_fused_attention_prefill_shaped_float(vulkan, n_tokens, start_pos, biases)
    });
}

fn cross_check_fused_attention_prefill_shaped_float(
    vulkan: &VulkanBackend,
    n_tokens: usize,
    start_pos: usize,
    biases: &str,
) {
    if vulkan.q4_k_mmvq {
        eprintln!("skipping: ORANGU_Q4K_MMVQ selects the unfused fallback path");
        return;
    }

    let (n_embd, n_head, n_head_kv, head_dim) = (256usize, 4usize, 2usize, 64usize);
    let (rope_dim, rope_freq_base, eps) = (64usize, 10000.0f32, 1e-6f32);
    let kv_dim = n_head_kv * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut seed = 0x0FA5_7ADD_u64;
    let mut build = |in_dim: usize, out_dim: usize| {
        let mut bytes = Vec::new();
        for _ in 0..out_dim {
            for _ in 0..(in_dim / 256) {
                bytes.extend(build_block(GGML_TYPE_Q4_K, &mut seed));
            }
        }
        test_quant_matrix(&bytes, GGML_TYPE_Q4_K, in_dim, out_dim)
    };
    let wq = build(n_embd, n_head * head_dim);
    let wk = build(n_embd, kv_dim);
    let wv = build(n_embd, kv_dim);
    // `g` in `biases`: a sigmoid gate on attention's output from its own
    // projection; `n`: no rotation (`rope_dim: 0`) — the muse-glimmer
    // full-attention layer's two departures from the llama shape.
    let gated = biases.contains('g');
    let rotate = !biases.contains('n');
    let w_gate = gated.then(|| build(n_embd, n_head * head_dim));
    let mut rand_vec = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
            .collect()
    };
    // Scaled so the *projections* land near unit magnitude, which is where
    // a real layer's do — its input has just been through `attn_norm`.
    // Left unscaled, Q and K reach the hundreds, attention scores reach
    // 1e6, and softmax becomes a hard argmax: the two paths then agree on
    // every token until a 1-ULP difference flips which position wins, and
    // the test reports a "mismatch" that is really a knife-edge. The fix is
    // to condition the input, not to loosen the assertion.
    let normed: Vec<f32> = rand_vec(n_tokens * n_embd)
        .into_iter()
        .map(|v| v * 0.02)
        .collect();

    let rope = crate::engine::tensor::RopeParams {
        rope_dim,
        freq_base: rope_freq_base,
        layout: crate::engine::tensor::RopeLayout::Norm,
        ..crate::engine::tensor::RopeParams::default()
    };

    // Scaled to the same magnitude the *projections* land at, for the
    // reason the `normed` comment above gives. A real Q/K bias is
    // comparable to its projection; left at `rand_vec`'s full range these
    // are several times larger, and since one bias vector is added to
    // every token, it puts a large constant into every q.k score. Softmax
    // then becomes a hard argmax and a 1-ULP difference flips which
    // position wins — a knife-edge that shows up as a huge output diff at
    // some token counts and not others. Condition the input, not the
    // assertion.
    let mut bias_vec =
        |n: usize| -> Vec<f32> { rand_vec(n).into_iter().map(|v| v * 0.25).collect() };
    let (q_bias, k_bias, v_bias) = (
        biases.contains('q').then(|| bias_vec(n_head * head_dim)),
        biases.contains('k').then(|| bias_vec(kv_dim)),
        biases.contains('v').then(|| bias_vec(kv_dim)),
    );

    // Reference: project, add biases, RoPE (no norms), fill the cache, attend.
    let mut q = vulkan.matmul(&normed, n_tokens, &wq);
    let mut k = vulkan.matmul(&normed, n_tokens, &wk);
    let mut v = vulkan.matmul(&normed, n_tokens, &wv);
    if let Some(b) = &q_bias {
        crate::engine::tensor::add_bias_per_row(&mut q, b, n_tokens);
    }
    if let Some(b) = &k_bias {
        crate::engine::tensor::add_bias_per_row(&mut k, b, n_tokens);
    }
    if let Some(b) = &v_bias {
        crate::engine::tensor::add_bias_per_row(&mut v, b, n_tokens);
    }
    let mut cache_ref =
        crate::engine::kv_cache::KvCache::new_with_dims(start_pos + n_tokens + 8, &[kv_dim]);
    for _ in 0..start_pos {
        cache_ref.layers[0].push(&vec![0.0; kv_dim], &vec![0.0; kv_dim]);
    }
    for t in 0..n_tokens {
        if rotate {
            crate::engine::tensor::rope_apply_params_inplace(
                &mut q[t * n_head * head_dim..(t + 1) * n_head * head_dim],
                n_head,
                head_dim,
                start_pos + t,
                None,
                &rope,
            );
            crate::engine::tensor::rope_apply_params_inplace(
                &mut k[t * kv_dim..(t + 1) * kv_dim],
                n_head_kv,
                head_dim,
                start_pos + t,
                None,
                &rope,
            );
        }
        cache_ref.layers[0].push(
            &k[t * kv_dim..(t + 1) * kv_dim],
            &v[t * kv_dim..(t + 1) * kv_dim],
        );
    }
    let mut expected = vec![0f32; n_tokens * n_head * head_dim];
    crate::engine::attention::multi_head_attention(
        &mut expected,
        &q,
        &cache_ref.layers[0],
        n_head,
        n_head / n_head_kv,
        head_dim,
        scale,
        |t| (0, start_pos + t),
    );
    if let Some(w) = &w_gate {
        let gate = vulkan.matmul(&normed, n_tokens, w);
        for (o, g) in expected.iter_mut().zip(gate.iter()) {
            *o *= crate::engine::tensor::sigmoid(*g);
        }
    }

    let mut cache =
        crate::engine::kv_cache::KvCache::new_with_dims(start_pos + n_tokens + 8, &[kv_dim]);
    for _ in 0..start_pos {
        cache.layers[0].push(&vec![0.0; kv_dim], &vec![0.0; kv_dim]);
    }
    let out = vulkan
        .fused_attention_prefill(
            FusedAttnPrefillInput {
                x_gpu: None,
                attn_norm: None,
                yarn: RopeYarn::IDENTITY,
                // The whole point of this helper: the reference above
                // applies these, so the fused call has to be given them.
                // They were `None` here while the reference was biased,
                // which is what made three of these cases "fail" — the
                // comparison was biased-reference against unbiased-fused,
                // and the diff it reported was the bias itself.
                q_bias: q_bias.as_deref(),
                pairing: crate::engine::tensor::RopeLayout::Norm,
                normalize_v: false,
                attn_gate: w_gate.as_ref(),
                normed: &normed,
                n_tokens,
                start_pos,
                wq: &wq,
                q_norm: None,
                kv: Some(FusedAttnPrefillKv {
                    k_bias: k_bias.as_deref(),
                    v_bias: v_bias.as_deref(),
                    wk: &wk,
                    k_norm: None,
                    wv: Some(&wv),
                }),
                n_head,
                n_head_kv,
                head_dim,
                rope_dim: if rotate { rope_dim } else { 0 },
                rope_freq_base,
                freq_factors: None,
                eps,
                n_swa: 0,
                causal: true,
                scale,
                want_attn_out_host: true,
            },
            &mut cache.layers[0],
        )
        .expect("fused path available without MMVQ");

    // The K and V rows the chain wrote, checked *before* attention gets a
    // chance to hide them. A K or V bias only reaches `attn_out` through
    // the softmax, which at small token counts barely moves the weighted
    // average — dropping the K and V bias dispatches entirely still passed
    // the 9-token attention check. These do not: they compare the rows
    // themselves, so a missing bias is a direct mismatch.
    let check = |label: &str, want: &[f32], got: &[f32]| {
        assert_eq!(want.len(), got.len(), "n_tokens={n_tokens}: {label} length");
        for (i, (a, b)) in want.iter().zip(got.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 6e-2 * a.abs().max(1.0),
                "n_tokens={n_tokens} start_pos={start_pos} biases={biases:?}: \
                     {label} mismatch at {i}: unfused={a} fused={b}"
            );
        }
    };
    check("k_rows", &k, &out.k_rows);
    check("v_rows", &v, &out.v_rows);

    assert_eq!(out.attn_out.len(), expected.len());
    for (i, (a, b)) in expected.iter().zip(out.attn_out.iter()).enumerate() {
        assert!(
            (a - b).abs() <= 6e-2 * a.abs().max(1.0),
            "n_tokens={n_tokens} start_pos={start_pos}: mismatch at {i}: \
                 unfused={a} fused={b}"
        );
    }
}

/// This failed by **~5700×** when first written (`unfused=-2095.5` against
/// `fused=-0.369`), with the Q and K norms already made optional. The ratio
/// was the RMS of a V row: the chain still applied gemma's per-head
/// weightless norm to V, which the llama family does not. Three
/// conventions, not two, and only the third was invisible from the
/// signature — every shape matched throughout.
#[test]
fn fused_attention_prefill_matches_the_unfused_sequence_without_norms() {
    cross_check_fused_attention_prefill_no_norms(7, 0);
}

/// See [`fused_attention_prefill_matches_the_unfused_sequence_without_norms`].
/// The muse-glimmer full-attention layer through the prefill chain: a
/// sigmoid gate from its own projection on attention's output, and no
/// rotation at all (`rope_dim: 0`) — against the CPU sequence with the
/// gate applied on the host. Past one stripe too, so the gate's own
/// pooled region and the in-place gating are exercised per stripe.
#[test]
fn fused_attention_prefill_gated_unrotated_matches_the_unfused_sequence() {
    cross_check_fused_attention_prefill_shaped(40, 0, "gn");
    cross_check_fused_attention_prefill_shaped(300, 16, "gn");
}

#[test]
fn fused_attention_prefill_without_norms_matches_at_a_nonzero_start_pos() {
    cross_check_fused_attention_prefill_no_norms(9, 5);
}

/// Past `MAX_MATMUL_TOKENS_PER_SUBMISSION` (128) the chain *stripes*, and
/// that path had its own bug: the per-stripe recursion restated `pairing`
/// and `normalize_v` instead of inheriting them through
/// `..input.reborrow()`, so a striped prompt silently ran a different
/// configuration from an unstriped one. Llama-3.2-3B answered 6- and
/// 47-token prompts correctly and returned token soup at 207.
///
/// The 7- and 9-token cases above never reached it. This one does.
#[test]
fn fused_attention_prefill_without_norms_matches_across_a_stripe_boundary() {
    cross_check_fused_attention_prefill_no_norms(192, 0);
}

/// Qwen2's shape: a bias on all three projections.
///
/// The K and V biases are caught by the `k_rows`/`v_rows` checks rather
/// than by `attn_out` — verified by deleting each dispatch in turn.
#[test]
fn fused_attention_prefill_with_qkv_biases_matches_the_unfused_sequence() {
    cross_check_fused_attention_prefill_shaped(9, 0, "qkv");
}

/// The same three biases past `MAX_MATMUL_TOKENS_PER_SUBMISSION`, so the
/// per-stripe recursion has to carry them. It does — they are per-row
/// constants and need no slicing — but this was believed for a while to be
/// broken, so it is pinned.
#[test]
fn fused_attention_prefill_with_qkv_biases_matches_across_a_stripe_boundary() {
    cross_check_fused_attention_prefill_shaped(192, 0, "qkv");
}

/// V's bias alone — the case that was `#[ignore]`d as "V's bias never
/// reaches attention at all". It reaches it correctly; the helper was
/// building a *biased* reference and then calling the fused path with
/// `v_bias: None`, so the difference it reported was the bias itself.
#[test]
fn fused_attention_prefill_with_a_v_bias_matches_the_unfused_sequence() {
    cross_check_fused_attention_prefill_shaped(9, 0, "v");
}

#[test]
fn fused_attention_prefill_matches_the_unfused_sequence_own_v() {
    cross_check_fused_attention_prefill(6, true, 3);
}

/// The layer without its own V projection — the K norm/copy/V-norm/K-RoPE
/// ordering, which no other test reaches.
#[test]
fn fused_attention_prefill_matches_the_unfused_sequence_shared_v() {
    cross_check_fused_attention_prefill(6, false, 3);
}

/// Past the cooperative-dispatch crossover, and at `start_pos = 0`.
#[test]
fn fused_attention_prefill_matches_the_unfused_sequence_wide() {
    cross_check_fused_attention_prefill(96, true, 0);
}

/// Past `MAX_MATMUL_TOKENS_PER_SUBMISSION`, so the batch **stripes** —
/// every shorter case above fits in one. Production prefills are far past
/// this, so the striped path is the one that actually runs.
#[test]
fn fused_attention_prefill_matches_the_unfused_sequence_striped() {
    cross_check_fused_attention_prefill(160, true, 0);
}

/// A chunk that runs **past the mirror the cache had**: 16 positions on the
/// host size the per-request mirror at its 256-row floor, and the chunk
/// then writes 300 more. Sized for one row ahead, the write ran off the
/// end of the K region into V, and the positions were only right again
/// once the *next* call re-uploaded them from a host copy that had waited
/// for them — a 1,563-token prompt on the contiguous path answered
/// differently from the unfused sequence because of it. The mirror is
/// sized for the rows about to be written now (`sync_gpu_ahead`).
#[test]
fn fused_attention_prefill_matches_the_unfused_sequence_past_the_mirror() {
    cross_check_fused_attention_prefill(300, true, 16);
}

/// One token deep in its context through the prefill chain — a decode
/// step of a model whose layers step one by one — takes the split
/// ("flash-decode") kernel over the window rather than the prefill
/// kernels' one workgroup per query tile: at a thousand positions the
/// latter was 0.9 ms a layer. Checked past the split's threshold, at a
/// window that is not a whole number of its chunks.
#[test]
fn fused_attention_prefill_one_deep_token_takes_the_split_kernel() {
    cross_check_fused_attention_prefill(1, true, 700);
    cross_check_fused_attention_prefill(1, false, 333);
}

/// The deferred form on the contiguous mirror, past its floor as above,
/// and on the page pool: the rows counted first and filled after the
/// chunk must be, row for row, what the in-order path pushed.
#[test]
fn fused_attention_prefill_deferred_fills_the_rows_it_counted() {
    cross_check_fused_attention_prefill_deferred(300, 16, false);
}

#[test]
fn fused_attention_prefill_deferred_fills_the_rows_it_counted_paged() {
    cross_check_fused_attention_prefill_deferred(160, 0, true);
}

/// A cross-layer **KV-donor** layer (`kv: None`): it projects Q only and
/// attends against a cache an earlier layer already filled, skipping the
/// whole K/V sub-chain and the cache write. Gemma4 has these, so the
/// end-to-end path exercises it, but the sub-chain being skipped rather
/// than run is a distinct branch worth pinning down on its own — and it
/// must leave the cache's length untouched.
#[test]
fn fused_attention_prefill_matches_the_unfused_sequence_kv_donor() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    if vulkan.q4_k_mmvq {
        eprintln!("skipping: ORANGU_Q4K_MMVQ selects the unfused fallback path");
        return;
    }

    let (n_embd, n_head, n_head_kv, head_dim, n_tokens) = (256usize, 4, 2, 64, 6);
    let (rope_dim, rope_freq_base, eps) = (64usize, 10000.0f32, 1e-6f32);
    let kv_dim = n_head_kv * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let start_pos = 4usize;
    let mut seed = 0x0D0D_0E11_u64;
    let wq = {
        let mut bytes = Vec::new();
        for _ in 0..(n_head * head_dim) {
            for _ in 0..(n_embd / 256) {
                bytes.extend(build_block(GGML_TYPE_Q4_K, &mut seed));
            }
        }
        test_quant_matrix(&bytes, GGML_TYPE_Q4_K, n_embd, n_head * head_dim)
    };
    let mut rand_vec = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
            .collect()
    };
    let normed = rand_vec(n_tokens * n_embd);
    let q_norm: Vec<f32> = rand_vec(head_dim).iter().map(|v| 1.0 + v * 0.1).collect();
    // The donor cache already holds every position this batch attends to.
    let filled: Vec<(Vec<f32>, Vec<f32>)> = (0..start_pos + n_tokens)
        .map(|_| (rand_vec(kv_dim), rand_vec(kv_dim)))
        .collect();
    let capacity = start_pos + n_tokens + 8;

    let fill = |cache: &mut crate::engine::kv_cache::KvCache| {
        for (k, v) in &filled {
            cache.layers[0].push(k, v);
        }
    };
    let mut ref_cache = crate::engine::kv_cache::KvCache::new_with_dims(capacity, &[kv_dim]);
    fill(&mut ref_cache);
    let mut q = vulkan.matmul(&normed, n_tokens, &wq);
    crate::engine::tensor::rmsnorm_inplace(&mut q, &q_norm, n_tokens * n_head, head_dim, eps);
    for t in 0..n_tokens {
        crate::engine::tensor::rope_apply_scaled_inplace(
            &mut q[t * n_head * head_dim..(t + 1) * n_head * head_dim],
            n_head,
            head_dim,
            rope_dim,
            start_pos + t,
            rope_freq_base,
            None,
        );
    }
    let expected = vulkan.gpu_attention_prefill(
        &q,
        &mut ref_cache.layers[0],
        start_pos,
        n_tokens,
        n_head,
        n_head_kv,
        head_dim,
        0,
        true,
        scale,
    );

    let mut cache = crate::engine::kv_cache::KvCache::new_with_dims(capacity, &[kv_dim]);
    fill(&mut cache);
    let before_len = cache.layers[0].len;
    let got = vulkan
        .fused_attention_prefill(
            FusedAttnPrefillInput {
                x_gpu: None,
                attn_norm: None,
                yarn: RopeYarn::IDENTITY,
                q_bias: None,
                pairing: crate::engine::tensor::RopeLayout::Neox,
                normalize_v: true,
                attn_gate: None,
                normed: &normed,
                n_tokens,
                start_pos,
                wq: &wq,
                q_norm: Some(&q_norm),
                kv: None,
                n_head,
                n_head_kv,
                head_dim,
                rope_dim,
                rope_freq_base,
                freq_factors: None,
                eps,
                n_swa: 0,
                causal: true,
                scale,
                want_attn_out_host: true,
            },
            &mut cache.layers[0],
        )
        .expect("fused prefill attention returned None for a KV-donor layer");

    assert!(got.k_rows.is_empty() && got.v_rows.is_empty());
    assert_eq!(
        cache.layers[0].len, before_len,
        "a donor layer must not append cache positions"
    );
    assert_eq!(expected.len(), got.attn_out.len());
    for (i, (a, b)) in expected.iter().zip(got.attn_out.iter()).enumerate() {
        assert!(
            (a - b).abs() <= 3e-3 * a.abs().max(1.0),
            "mismatch at {i}: unfused={a} fused={b}"
        );
    }
}

/// Handing attention's output to the post-attention chain **on the GPU**
/// must produce exactly what handing it through host memory does. This is
/// the pairing that removes the largest transfer in a prefill layer — a
/// readback of `[n_tokens, n_head, head_dim]` immediately followed by an
/// upload of the same block — so it is worth pinning the two against each
/// other rather than only against the CPU reference each already has.
///
/// Run at 160 tokens so both halves stripe, and their striping differs: the
/// attention half must not pad (padded rows would enter the KV cache as
/// real positions) while the post-attention half does pad. A GPU source is
/// sliced by byte offset instead, and this is what catches that going
/// wrong.
#[test]
fn fused_post_attention_prefill_gpu_source_matches_the_host_source() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    if vulkan.q4_k_mmvq {
        eprintln!("skipping: ORANGU_Q4K_MMVQ selects the unfused fallback path");
        return;
    }

    let (n_embd, n_head, n_head_kv, head_dim, n_tokens) = (256usize, 4, 2, 64, 160);
    let (rope_dim, ffn_len, eps) = (64usize, 512usize, 1e-6f32);
    let kv_dim = n_head_kv * head_dim;
    let attn_dim = n_head * head_dim;
    let mut seed = 0x9A17_C0DE_u64;
    let mut build = |in_dim: usize, out_dim: usize| {
        let mut bytes = Vec::new();
        for _ in 0..out_dim {
            for _ in 0..(in_dim / 256) {
                bytes.extend(build_block(GGML_TYPE_Q4_K, &mut seed));
            }
        }
        test_quant_matrix(&bytes, GGML_TYPE_Q4_K, in_dim, out_dim)
    };
    let wq = build(n_embd, attn_dim);
    let wk = build(n_embd, kv_dim);
    let wv = build(n_embd, kv_dim);
    let wo = build(attn_dim, n_embd);
    let gate = build(n_embd, ffn_len);
    let up = build(n_embd, ffn_len);
    let down = build(ffn_len, n_embd);
    let mut rand_vec = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
            .collect()
    };
    let normed = rand_vec(n_tokens * n_embd);
    let residual = rand_vec(n_tokens * n_embd);
    let q_norm: Vec<f32> = rand_vec(head_dim).iter().map(|v| 1.0 + v * 0.1).collect();
    let k_norm: Vec<f32> = rand_vec(head_dim).iter().map(|v| 1.0 + v * 0.1).collect();
    let n1: Vec<f32> = rand_vec(n_embd).iter().map(|v| 1.0 + v * 0.1).collect();
    let n2: Vec<f32> = rand_vec(n_embd).iter().map(|v| 1.0 + v * 0.1).collect();
    let n3: Vec<f32> = rand_vec(n_embd).iter().map(|v| 1.0 + v * 0.1).collect();

    let attn_input = |want_host: bool| FusedAttnPrefillInput {
        x_gpu: None,
        attn_norm: None,
        yarn: RopeYarn::IDENTITY,
        q_bias: None,
        pairing: crate::engine::tensor::RopeLayout::Neox,
        normalize_v: true,
        attn_gate: None,
        normed: &normed,
        n_tokens,
        start_pos: 0,
        wq: &wq,
        q_norm: Some(&q_norm),
        kv: Some(FusedAttnPrefillKv {
            k_bias: None,
            v_bias: None,
            wk: &wk,
            k_norm: Some(&k_norm),
            wv: Some(&wv),
        }),
        n_head,
        n_head_kv,
        head_dim,
        rope_dim,
        rope_freq_base: 10000.0,
        freq_factors: None,
        eps,
        n_swa: 0,
        causal: true,
        scale: 1.0 / (head_dim as f32).sqrt(),
        want_attn_out_host: want_host,
    };
    let post = |src: AttnOutSrc<'_>| {
        vulkan
            .fused_post_attention_prefill(
                src,
                &residual,
                n_tokens,
                &wo,
                Some(&n1),
                &n2,
                &gate,
                &up,
                &down,
                Some(&n3),
                eps,
                FfnActivation::Geglu,
            )
            .expect("fused post-attention returned None on a supported path")
    };

    // One attention result, consumed both ways, so the only variable is the
    // handoff itself.
    let mut c1 = crate::engine::kv_cache::KvCache::new_with_dims(n_tokens + 8, &[kv_dim]);
    let a1 = vulkan
        .fused_attention_prefill(attn_input(true), &mut c1.layers[0])
        .expect("fused prefill attention returned None");
    let via_host = post(AttnOutSrc::Host(&a1.attn_out));
    let via_gpu = post(AttnOutSrc::Gpu(&a1.attn_out_buf, 0, n_tokens));

    // NOTE: a second run with `want_attn_out_host: false` was found to
    // produce K rows differing from this one by ~0.1% from stripe 1
    // onward. That is a difference between two independent runs, not
    // between the two handoffs, so it is not what this test is for — but
    // it is unexplained and worth chasing: identical inputs through
    // identical kernels should be bit-identical.
    assert_eq!(via_host.len(), via_gpu.len());
    for (i, (a, b)) in via_host.iter().zip(via_gpu.iter()).enumerate() {
        assert!(
            (a - b).abs() <= 1e-4 * a.abs().max(1.0),
            "mismatch at {i}: host-source={a} gpu-source={b}"
        );
    }
}

/// The fused post-attention chain at the **model's own dimensions** and a
/// token count above the tiled-GEMM crossover. The existing cross-checks
/// use `n_embd = 256 / ffn_len = 512`; the model runs 1536 / 6144, and a
/// 91-token prompt is where real output goes wrong while every isolated
/// matmul at these same shapes is exact.
#[test]
fn fused_post_attention_prefill_matches_the_unfused_sequence_model_shaped() {
    cross_check_fused_post_attention_prefill_dims(91, 1536, 2048, 6144);
}

fn cross_check_fused_post_attention_prefill(n_tokens: usize) {
    cross_check_fused_post_attention_prefill_dims(n_tokens, 256, 512, 512);
}

fn cross_check_fused_post_attention_prefill_dims(
    n_tokens: usize,
    n_embd: usize,
    attn_dim: usize,
    ffn_len: usize,
) {
    cross_check_fused_post_attention_shaped(n_tokens, n_embd, attn_dim, ffn_len, true);
}

/// `gemma_shaped` picks which of the two architectures' chains is under
/// test: gemma has a post-norm on both residual adds and a GEGLU gate;
/// Llama/Qwen2/Mistral/Phi have neither post-norm and a SwiGLU gate. Both
/// go through the *same* fused function, and both are compared against a
/// sequence built from unfused `matmul` calls and CPU tensor ops — a
/// reference that shares no kernel with the fused chain (LESSONS §1).
#[allow(clippy::too_many_arguments)]
fn cross_check_fused_post_attention_shaped(
    n_tokens: usize,
    n_embd: usize,
    attn_dim: usize,
    ffn_len: usize,
    gemma_shaped: bool,
) {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    vulkan.without_prefill_mmq(|| {
        cross_check_fused_post_attention_shaped_float(
            vulkan,
            n_tokens,
            n_embd,
            attn_dim,
            ffn_len,
            gemma_shaped,
        )
    });
}

fn cross_check_fused_post_attention_shaped_float(
    vulkan: &VulkanBackend,
    n_tokens: usize,
    n_embd: usize,
    attn_dim: usize,
    ffn_len: usize,
    gemma_shaped: bool,
) {
    if vulkan.q4_k_mmvq {
        eprintln!("skipping: ORANGU_Q4K_MMVQ selects the unfused fallback path");
        return;
    }

    let eps = 1e-6f32;
    let mut seed = 0x50DA_u64;
    let mut build = |in_dim: usize, out_dim: usize| {
        let mut bytes = Vec::new();
        for _ in 0..out_dim {
            for _ in 0..(in_dim / 256) {
                bytes.extend(build_block(GGML_TYPE_Q4_K, &mut seed));
            }
        }
        test_quant_matrix(&bytes, GGML_TYPE_Q4_K, in_dim, out_dim)
    };
    let wo = build(attn_dim, n_embd);
    let gate = build(n_embd, ffn_len);
    let up = build(n_embd, ffn_len);
    let down = build(ffn_len, n_embd);

    let mut rand_vec = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
            .collect()
    };
    let attn_out = rand_vec(n_tokens * attn_dim);
    let residual = rand_vec(n_tokens * n_embd);
    // Norm weights near 1, as a trained model's are — random ones make the
    // comparison dominated by whichever path rounds first.
    let attn_post_norm: Vec<f32> = rand_vec(n_embd).iter().map(|v| 1.0 + v * 0.1).collect();
    let ffn_norm: Vec<f32> = rand_vec(n_embd).iter().map(|v| 1.0 + v * 0.1).collect();
    let ffn_post_norm: Vec<f32> = rand_vec(n_embd).iter().map(|v| 1.0 + v * 0.1).collect();

    // The unfused sequence.
    let mut x1 = vulkan.matmul(&attn_out, n_tokens, &wo);
    if gemma_shaped {
        crate::engine::tensor::rmsnorm_inplace(&mut x1, &attn_post_norm, n_tokens, n_embd, eps);
    }
    crate::engine::tensor::add_inplace(&mut x1, &residual);
    let mut ffn_normed = x1.clone();
    crate::engine::tensor::rmsnorm_inplace(&mut ffn_normed, &ffn_norm, n_tokens, n_embd, eps);
    let mut g = vulkan.matmul(&ffn_normed, n_tokens, &gate);
    let u = vulkan.matmul(&ffn_normed, n_tokens, &up);
    if gemma_shaped {
        crate::engine::tensor::gelu_inplace(&mut g);
    } else {
        for v in g.iter_mut() {
            *v = crate::engine::tensor::silu(*v);
        }
    }
    crate::engine::tensor::mul_inplace(&mut g, &u);
    let mut ffn_out = vulkan.matmul(&g, n_tokens, &down);
    if gemma_shaped {
        crate::engine::tensor::rmsnorm_inplace(&mut ffn_out, &ffn_post_norm, n_tokens, n_embd, eps);
    }
    crate::engine::tensor::add_inplace(&mut ffn_out, &x1);
    let expected = ffn_out;

    let got = vulkan
        .fused_post_attention_prefill(
            AttnOutSrc::Host(&attn_out),
            &residual,
            n_tokens,
            &wo,
            gemma_shaped.then_some(attn_post_norm.as_slice()),
            &ffn_norm,
            &gate,
            &up,
            &down,
            gemma_shaped.then_some(ffn_post_norm.as_slice()),
            eps,
            if gemma_shaped {
                FfnActivation::Geglu
            } else {
                FfnActivation::Swiglu
            },
        )
        .expect("fused path available without MMVQ");

    assert_eq!(got.len(), expected.len());
    assert_eq!(got.len(), n_tokens * n_embd);
    for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
        let tol = 6e-2 * a.abs().max(1.0);
        assert!(
            (a - b).abs() <= tol,
            "n_tokens={n_tokens}: mismatch at {i}: unfused={a} fused={b}"
        );
    }

    // The same chain with a layer output scale folded into its last step
    // — the form a dense gemma layer without a per-layer-embedding stage
    // takes on the device-resident stream — against the sequence scaled
    // on the host.
    if gemma_shaped {
        let scale = 0.65f32;
        let got = vulkan
            .fused_post_attention_prefill_rows(
                AttnOutSrc::Host(&attn_out),
                AttnOutSrc::Host(&residual),
                n_tokens,
                &wo,
                Some(attn_post_norm.as_slice()),
                &ffn_norm,
                &gate,
                &up,
                &down,
                Some(ffn_post_norm.as_slice()),
                eps,
                FfnActivation::Geglu,
                None,
                Some(scale),
                None,
            )
            .expect("the scaled form exists whenever the post-norm does");
        assert_eq!(got.len(), expected.len());
        for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
            let a = a * scale;
            let tol = 6e-2 * a.abs().max(1.0);
            assert!(
                (a - b).abs() <= tol,
                "n_tokens={n_tokens}: scaled mismatch at {i}: unfused={a} fused={b}"
            );
        }
    }
}

#[test]
fn fused_post_attention_prefill_matches_the_unfused_sequence_small() {
    cross_check_fused_post_attention_prefill(3);
}

#[test]
fn fused_post_attention_prefill_matches_the_unfused_sequence_multi_chunk() {
    cross_check_fused_post_attention_prefill(192);
}

/// The Llama/Qwen2/Mistral/Phi shape: SwiGLU, and **no** post-norm on
/// either residual add. Both differences are load-bearing — a post-norm
/// left in place normalizes a tensor that must not be normalized, and GEGLU
/// against SwiGLU is a different function entirely — so this is a separate
/// case rather than a variation of the gemma one.
#[test]
fn fused_post_attention_prefill_matches_the_unfused_sequence_swiglu_no_post_norms() {
    cross_check_fused_post_attention_shaped(3, 256, 512, 512, false);
}

#[test]
fn fused_post_attention_prefill_swiglu_matches_across_a_stripe_boundary() {
    cross_check_fused_post_attention_shaped(192, 256, 512, 512, false);
}

#[test]
fn fused_post_attention_prefill_swiglu_matches_at_model_shaped_dims() {
    cross_check_fused_post_attention_shaped(91, 1536, 2048, 6144, false);
}

/// The MoE head chain (`moe_head_rows`) against the same computation as
/// the separate device calls the host path makes: the output projection,
/// post-norm and residual for `x1`; the router's weightless norm, scale
/// and projection for the logits; the shared MLP with its post-norm. At
/// one token (the decode shape) and a few, on float kernels so the two
/// sides differ only by fusion.
#[test]
fn the_moe_head_chain_matches_the_separate_calls() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    if vulkan.q4_k_mmvq {
        return;
    }
    vulkan.without_prefill_mmq(|| {
        let (n_embd, attn_dim, ffn_len, n_expert) = (256usize, 512usize, 512usize, 8usize);
        let eps = 1e-6f32;
        let mut seed = 0x0E0E_u64;
        let mut build = |in_dim: usize, out_dim: usize| {
            let mut bytes = Vec::new();
            for _ in 0..out_dim {
                for _ in 0..(in_dim / 256) {
                    bytes.extend(build_block(GGML_TYPE_Q4_K, &mut seed));
                }
            }
            test_quant_matrix(&bytes, GGML_TYPE_Q4_K, in_dim, out_dim)
        };
        let wo = build(attn_dim, n_embd);
        let gate_inp = build(n_embd, n_expert);
        let gate = build(n_embd, ffn_len);
        let up = build(n_embd, ffn_len);
        let down = build(ffn_len, n_embd);
        let mut rand_vec = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
                .collect()
        };
        for n_tokens in [1usize, 3] {
            let attn_out = rand_vec(n_tokens * attn_dim);
            let residual = rand_vec(n_tokens * n_embd);
            let attn_post_norm: Vec<f32> = rand_vec(n_embd).iter().map(|v| 1.0 + v * 0.1).collect();
            let ffn_norm: Vec<f32> = rand_vec(n_embd).iter().map(|v| 1.0 + v * 0.1).collect();
            let post_norm_1: Vec<f32> = rand_vec(n_embd).iter().map(|v| 1.0 + v * 0.1).collect();
            let gate_inp_scale: Vec<f32> =
                rand_vec(n_embd).iter().map(|v| 1.0 + v * 0.05).collect();
            let scale = 1.0 / (n_embd as f32).sqrt();
            let router_weight: Vec<f32> = gate_inp_scale.iter().map(|s| s * scale).collect();

            // The separate calls.
            let mut x1 = vulkan.matmul(&attn_out, n_tokens, &wo);
            crate::engine::tensor::rmsnorm_inplace(&mut x1, &attn_post_norm, n_tokens, n_embd, eps);
            crate::engine::tensor::add_inplace(&mut x1, &residual);
            let mut tmp = x1.clone();
            crate::engine::arch::gemma::rmsnorm_weightless_inplace(&mut tmp, n_tokens, n_embd, eps);
            for row in tmp.chunks_mut(n_embd) {
                for (v, s) in row.iter_mut().zip(&gate_inp_scale) {
                    *v *= scale * s;
                }
            }
            let logits = vulkan.matmul(&tmp, n_tokens, &gate_inp);
            let mut normed = x1.clone();
            crate::engine::tensor::rmsnorm_inplace(&mut normed, &ffn_norm, n_tokens, n_embd, eps);
            let mut g = vulkan.matmul(&normed, n_tokens, &gate);
            let u = vulkan.matmul(&normed, n_tokens, &up);
            crate::engine::tensor::gelu_inplace(&mut g);
            crate::engine::tensor::mul_inplace(&mut g, &u);
            let mut shared = vulkan.matmul(&g, n_tokens, &down);
            crate::engine::tensor::rmsnorm_inplace(
                &mut shared,
                &post_norm_1,
                n_tokens,
                n_embd,
                eps,
            );

            let (got_x1, got_logits, pending) = vulkan
                .moe_head_rows(
                    AttnOutSrc::Host(&attn_out),
                    &residual,
                    n_tokens,
                    &wo,
                    &attn_post_norm,
                    &router_weight,
                    &gate_inp,
                    &ffn_norm,
                    &gate,
                    &up,
                    &down,
                    &post_norm_1,
                    eps,
                )
                .expect("the head chain runs on float kernels");
            let got_shared = vulkan.finish_rows(pending);
            for (name, got, want) in [
                ("x1", &got_x1, &x1),
                ("logits", &got_logits, &logits),
                ("shared", &got_shared, &shared),
            ] {
                assert_eq!(got.len(), want.len(), "{name} at {n_tokens} tokens");
                for (i, (a, b)) in want.iter().zip(got).enumerate() {
                    assert!(
                        (a - b).abs() <= 6e-2 * a.abs().max(1.0),
                        "{name} at {n_tokens} tokens, element {i}: separate {a} vs chain {b}"
                    );
                }
            }
        }
    });
}

/// Zero-padding a prefill stripe up to [`padded_stripe_len`] must not
/// change the rows that were really there — the padded rows are dispatched
/// work whose results get sliced off, and nothing else may move.
///
/// Checked on the projection itself rather than through a fused chain
/// because padding crosses a kernel boundary (a 2-token stripe takes the
/// per-token reduce path, a 64-token one the cooperative tiled path), and
/// the two agree to within normal float reassociation — which a GELU fed
/// ~1e12 inputs, as this module's random `Q4_K` test weights produce,
/// would then amplify without bound. This is the property that actually
/// matters; the fused chains' own cross-checks cover the rest.
#[test]
fn padding_a_stripe_leaves_its_real_rows_unchanged() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    let (in_dim, out_dim) = (256usize, 512usize);
    let mut seed = 0x9AD_u64;
    let mut bytes = Vec::new();
    for _ in 0..out_dim {
        bytes.extend(build_block(GGML_TYPE_Q4_K, &mut seed));
    }
    let w = test_quant_matrix(&bytes, GGML_TYPE_Q4_K, in_dim, out_dim);

    let real_rows = 2usize;
    let x: Vec<f32> = (0..real_rows * in_dim)
        .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
        .collect();
    let padded = padded_stripe_len(real_rows, max_matmul_tokens_per_submission() * 2);
    assert!(padded > real_rows, "this shape is supposed to pad");
    let mut x_padded = x.clone();
    x_padded.resize(padded * in_dim, 0.0);

    // Float kernels on both sides: the padded width is one the integer-dot
    // kernel would take and the real width is not, and this checks the
    // stripe padding, not the two kernels against each other.
    let (unpadded, widened) = vulkan.without_prefill_mmq(|| {
        (
            vulkan.matmul(&x, real_rows, &w),
            vulkan.matmul(&x_padded, padded, &w),
        )
    });
    for i in 0..real_rows * out_dim {
        let (a, b) = (unpadded[i], widened[i]);
        let rel = (a - b).abs() / a.abs().max(1.0);
        assert!(
            rel <= 1e-3,
            "padding moved element {i} of row {}: unpadded={a} padded={b} (rel {rel:e})",
            i / out_dim
        );
    }
}

/// Cross-checks `gpu_attention_prefill` against a direct transcription of
/// the CPU attention loop `GemmaModel::run_layers_cpu` runs, for all three
/// window shapes the model can ask for. The GPU kernel derives each
/// query's window itself from `start_pos + t`, so this is really checking
/// that its in-shader rule and `GemmaModel::attention_window` agree — the
/// one place the two implementations could silently diverge.
///
/// `start_pos > 0` on purpose: a prompt continuing an existing
/// conversation attends over cache positions that precede its own first
/// token, which a test starting at zero would never exercise.
fn cross_check_gpu_attention_prefill(n_swa: usize, causal: bool, start_pos: usize) {
    cross_check_gpu_attention_prefill_shaped(n_swa, causal, start_pos, 4, 2, 32);
}

fn cross_check_gpu_attention_prefill_shaped(
    n_swa: usize,
    causal: bool,
    start_pos: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
) {
    cross_check_gpu_attention_prefill_sized(
        n_swa, causal, start_pos, n_head, n_head_kv, head_dim, 9,
    )
}

#[allow(clippy::too_many_arguments)]
fn cross_check_gpu_attention_prefill_sized(
    n_swa: usize,
    causal: bool,
    start_pos: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    n_tokens: usize,
) {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    let kv_dim = n_head_kv * head_dim;
    let capacity = (start_pos + n_tokens + 8).max(64);
    let scale = 0.125f32;

    let mut seed = 0xA77Eu64;
    let mut rand_vec = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
            .collect()
    };

    let mut kv_cache = crate::engine::kv_cache::KvCache::new_with_dims(capacity, &[kv_dim]);
    // Everything the prompt attends to: `start_pos` earlier positions plus
    // this batch's own tokens, all pushed before attention runs.
    for _ in 0..(start_pos + n_tokens) {
        let k = rand_vec(kv_dim);
        let v = rand_vec(kv_dim);
        kv_cache.layers[0].push(&k, &v);
    }
    let q = rand_vec(n_tokens * n_head * head_dim);

    // The CPU reference, transcribed from `run_layers_cpu`.
    let group_size = n_head / n_head_kv;
    let mut expected = vec![0f32; n_tokens * n_head * head_dim];
    for t in 0..n_tokens {
        let pos = start_pos + t;
        let (window_start, window_end) = if !causal {
            if n_swa > 0 {
                let half = n_swa / 2;
                (pos.saturating_sub(half), (pos + half).min(n_tokens - 1))
            } else {
                (0, n_tokens - 1)
            }
        } else if n_swa > 0 {
            (pos.saturating_sub(n_swa - 1), pos)
        } else {
            (0, pos)
        };
        for h in 0..n_head {
            let kv_head = h / group_size;
            let qh = &q[(t * n_head + h) * head_dim..(t * n_head + h + 1) * head_dim];
            let mut scores = Vec::new();
            for p in window_start..=window_end {
                let kh = kv_cache.layers[0].key_at(p, kv_head, head_dim);
                scores.push(crate::engine::tensor::dot(qh, kh) * scale);
            }
            crate::engine::tensor::softmax_inplace(&mut scores);
            let out = &mut expected[(t * n_head + h) * head_dim..(t * n_head + h + 1) * head_dim];
            for (offset, &weight) in scores.iter().enumerate() {
                let vh = kv_cache.layers[0].value_at(window_start + offset, kv_head, head_dim);
                for (o, vi) in out.iter_mut().zip(vh.iter()) {
                    *o += weight * vi;
                }
            }
        }
    }

    let got = vulkan.gpu_attention_prefill(
        &q,
        &mut kv_cache.layers[0],
        start_pos,
        n_tokens,
        n_head,
        n_head_kv,
        head_dim,
        n_swa,
        causal,
        scale,
    );

    assert_eq!(got.len(), expected.len());
    for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
        assert!(
            (a - b).abs() <= 2e-3 * a.abs().max(1.0),
            "n_swa={n_swa} causal={causal} start_pos={start_pos}: \
                 mismatch at token {} head {}: cpu={a} gpu={b}",
            i / (n_head * head_dim),
            (i / head_dim) % n_head
        );
    }
}

#[test]
fn gpu_attention_prefill_matches_cpu_reference_causal() {
    cross_check_gpu_attention_prefill(0, true, 5);
}

#[test]
fn gpu_attention_prefill_matches_cpu_reference_sliding_window() {
    cross_check_gpu_attention_prefill(4, true, 5);
}

#[test]
fn gpu_attention_prefill_matches_cpu_reference_non_causal() {
    cross_check_gpu_attention_prefill(0, false, 0);
}

/// The GQA prefill kernel at the two head_dims a Gemma-shaped model asks
/// for, both with the model's own MQA ratio (`n_head_kv == 1`, so one KV
/// head feeds all eight query heads — the maximum sharing the kernel can
/// do, and the case where getting the `(kv_head, slice)` split of the
/// dispatch's `x` wrong would be silent).
///
/// `256` and `512` sit on opposite sides of the register budget in
/// [`vulkan_shaders::gqa_prefill_heads_per_workgroup`]: the first fits the
/// whole group in one workgroup, the second is split across several, so
/// between them they cover both the `slices == 1` and `slices > 1` paths.
#[test]
fn gpu_attention_prefill_gqa_matches_cpu_reference_single_slice() {
    cross_check_gpu_attention_prefill_shaped(0, true, 5, 8, 1, 256);
}

#[test]
fn gpu_attention_prefill_gqa_matches_cpu_reference_multi_slice() {
    cross_check_gpu_attention_prefill_shaped(0, true, 5, 8, 1, 512);
}

#[test]
fn gpu_attention_prefill_gqa_matches_cpu_reference_sliding_window() {
    cross_check_gpu_attention_prefill_shaped(4, true, 5, 8, 1, 256);
}

#[test]
fn gpu_attention_prefill_gqa_matches_cpu_reference_non_causal() {
    cross_check_gpu_attention_prefill_shaped(0, false, 0, 8, 1, 512);
}

/// `gpu_attention_prefill` reuses one pair of device buffers across every
/// call, grown to the largest request so far and never shrunk — the fix
/// for it exhausting VRAM on a long prompt. The hazard that introduces is
/// a *smaller* call afterwards reading whatever the larger one left
/// behind: the buffers are then longer than the shapes bound to them, and
/// nothing about that is visible in a single-call test.
///
/// So: a wide batch first, then a narrow one, through the same (shared)
/// backend, each checked against the CPU reference. The order matters and
/// is the whole point — reversed, this passes without exercising anything.
#[test]
fn gpu_attention_prefill_is_correct_after_a_larger_batch_on_the_same_backend() {
    cross_check_gpu_attention_prefill_sized(0, true, 5, 8, 1, 256, 96);
    cross_check_gpu_attention_prefill_sized(0, true, 5, 8, 1, 256, 3);
    // A narrower head_dim after a wider one, so the reused buffer is
    // oversized on both axes at once.
    cross_check_gpu_attention_prefill_sized(0, true, 5, 8, 1, 512, 64);
    cross_check_gpu_attention_prefill_sized(4, true, 5, 8, 1, 256, 5);
}

/// The tiled kernel over many blocks: a window several blocks wide that
/// starts and ends inside blocks, a tile count that does not divide the
/// token count, a deep `start_pos`, and the narrower head_dim that takes
/// eight rows per thread.
#[test]
fn gpu_attention_prefill_tiled_long_sliding_window() {
    cross_check_gpu_attention_prefill_sized(64, true, 300, 8, 2, 256, 203);
}

#[test]
fn gpu_attention_prefill_tiled_deep_causal_narrow_head() {
    cross_check_gpu_attention_prefill_sized(0, true, 1000, 4, 1, 128, 100);
    cross_check_gpu_attention_prefill_sized(40, false, 0, 4, 2, 128, 77);
}

/// `n_head_kv == n_head` leaves nothing to share, so this is the one shape
/// that still reaches the ungrouped cooperative kernel — the other prefill
/// cross-checks all have a group and take the GQA path.
#[test]
fn gpu_attention_prefill_ungrouped_matches_cpu_reference() {
    cross_check_gpu_attention_prefill_shaped(0, true, 5, 4, 4, 256);
}

/// A group split over several workgroups must still cover every query head
/// exactly once: `heads` divides `group`, and the dispatch derives its `x`
/// extent as `n_head / heads`.
#[test]
fn gqa_prefill_heads_divide_the_group() {
    for group in [1u32, 2, 4, 8, 16] {
        for head_dim in [32u32, 64, 128, 256, 512, 1024] {
            let heads = vulkan_shaders::gqa_prefill_heads_per_workgroup(head_dim, group);
            assert!(heads >= 1 && heads <= group);
            assert_eq!(group % heads, 0, "group={group} head_dim={head_dim}");
        }
    }
}

#[test]
fn fused_ffn_prefill_matches_the_unfused_sequence_small() {
    cross_check_fused_ffn_prefill(3);
}

/// How this device lays a 64-lane workgroup over subgroups, and what
/// `subgroupShuffleXor` by 1, 2 and 4 returns per lane — the assumption the
/// ternary kernel's per-block maximum rests on (eight adjacent lanes in one
/// subgroup, xor-shuffles staying among them).
/// `cargo test --profile release-with-debug --bin orangu-server
/// subgroup_layout_on_this_device -- --ignored --nocapture`.
#[test]
#[ignore]
fn subgroup_layout_on_this_device() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    const SRC: &str = r#"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_lane: u32,
    @builtin(subgroup_id) sg_id: u32,
    @builtin(subgroup_size) sg_size: u32,
) {
    let t = lid.x;
    var it: u32 = 0u;
    var a: f32 = 0.0;
    var b: f32 = 0.0;
    var c: f32 = 0.0;
    loop {
        if (it >= 2u) { break; }
        let v = f32(t) + f32(it) * 100.0;
        a = subgroupShuffleXor(v, 1u);
        b = subgroupShuffleXor(v, 2u);
        c = subgroupShuffleXor(v, 4u);
        it = it + 1u;
    }
    // The ternary kernel's own chain: a max over eight adjacent lanes.
    var mx: f32 = 0.0;
    it = 0u;
    loop {
        if (it >= 3u) { break; }
        let a0 = f32((t * 37u + it * 11u) % 64u);
        let a1 = max(a0, subgroupShuffleXor(a0, 1u));
        let a2 = max(a1, subgroupShuffleXor(a1, 2u));
        let a3 = max(a2, subgroupShuffleXor(a2, 4u));
        mx = mx + a3 * 1.0;
        it = it + 1u;
    }
    out[t * 4u] = f32(sg_id) * 100.0 + f32(sg_lane) + f32(sg_size) * 10000.0;
    out[t * 4u + 1u] = a;
    out[t * 4u + 2u] = b;
    out[t * 4u + 3u] = mx;
}
"#;
    let got = run_probe_kernel(vulkan, SRC, 64, 256);
    for t in 0..64 {
        let meta = got[t * 4] as u32;
        let want: usize = (0..3)
            .map(|it| {
                (0..8)
                    .map(|l| ((t / 8 * 8 + l) * 37 + it * 11) % 64)
                    .max()
                    .unwrap()
            })
            .sum();
        eprintln!(
            "lane {t:2}: subgroup {} lane {:2} (size {}) | xor1 {} xor2 {} | 8-lane max chain {} (want {want}){}",
            (meta / 100) % 100,
            meta % 100,
            meta / 10000,
            got[t * 4 + 1] - 100.0,
            got[t * 4 + 2] - 100.0,
            got[t * 4 + 3],
            if got[t * 4 + 3] as usize == want {
                ""
            } else {
                "  <-- WRONG"
            }
        );
    }
}

/// What `pack4x8snorm` rounds to on this device around a tie: 64 values
/// `v = (63.5 + k · 0.004) / 127`, `k = -32..32`, packed and read back.
/// `cargo test --profile release-with-debug --bin orangu-server
/// pack4x8snorm_rounding_on_this_device -- --ignored --nocapture`.
#[test]
#[ignore]
fn pack4x8snorm_rounding_on_this_device() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    const SRC: &str = r#"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    let v = (63.5 + (f32(t) - 32.0) * 0.004) / 127.0;
    let p = pack4x8snorm(vec4<f32>(v, -v, v * 0.5, 1.0));
    out[t] = f32(i32(p << 24u) >> 24u);
}
"#;
    let got = run_probe_kernel(vulkan, SRC, 64, 64);
    for (t, g) in got.iter().enumerate() {
        let v = (63.5 + (t as f32 - 32.0) * 0.004) / 127.0;
        eprintln!("v*127 = {:9.4} -> {g}", v * 127.0);
    }
}

/// The matmul kernel's device time read two ways: the pass timestamps
/// (`matmul_kernel_us_tokens`) against the wall clock of the whole
/// submission for 1, 4, 16 and 64 back-to-back dispatches — the slope of
/// the wall clock over the dispatch count is the kernel's real time per
/// dispatch, whatever the timestamps say, and the intercept is the
/// submission's fixed cost. Random weights, not a constant byte, so the
/// reading is what a model's layer sees.
/// `cargo test --profile release-with-debug --bin orangu-server
/// matmul_kernel_time_by_wall_clock -- --ignored --nocapture`.
#[test]
#[ignore]
fn matmul_kernel_time_by_wall_clock() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    let (in_dim, out_dim) = (5120usize, 17408usize);
    // `ORANGU_KERNEL_PROBE_TOKENS` reads the prefill kernel at that batch.
    let n_tokens: usize = std::env::var("ORANGU_KERNEL_PROBE_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    for &ggml_type in &[
        GGML_TYPE_PTQ1_0,
        GGML_TYPE_PQ2_0,
        GGML_TYPE_Q8_0,
        GGML_TYPE_Q4_K,
    ] {
        let mut seed = 0xC0FFEE_u64;
        let mut bytes = Vec::new();
        for _ in 0..out_dim {
            for _ in 0..(in_dim
                / crate::engine::quant::block_layout(ggml_type)
                    .expect("layout")
                    .1)
            {
                bytes.extend(build_block(ggml_type, &mut seed));
            }
        }
        // `ORANGU_KERNEL_PROBE_FILL=const` reads the kernel on a constant
        // byte instead (what the format sweep does).
        if std::env::var("ORANGU_KERNEL_PROBE_FILL").as_deref() == Ok("const") {
            bytes.fill(0x42);
        }
        let mib = bytes.len() as f64 / (1024.0 * 1024.0);
        let w = test_quant_matrix(&bytes, ggml_type, in_dim, out_dim);
        let x: Vec<f32> = (0..in_dim * n_tokens)
            .map(|i| ((i * 37 % 23) as f32 - 11.0) * 0.031)
            .collect();
        let mut name = "";
        for &reps in &[1u32, 4, 16, 64] {
            let mut best_wall = f64::MAX;
            let mut ts = 0.0;
            for _ in 0..3 {
                let t = std::time::Instant::now();
                let (us, n) = vulkan
                    .matmul_kernel_us_tokens(&x, n_tokens, &w, reps)
                    .expect("timestamps");
                name = n;
                let wall = t.elapsed().as_secs_f64() * 1e6;
                if wall < best_wall {
                    best_wall = wall;
                    ts = us;
                }
            }
            eprintln!(
                "  {} [{in_dim} x {out_dim}] x {n_tokens} {mib:.1} MiB ({name}) reps {reps:3}: wall {:9.0} us ({:7.0} us/dispatch, {:5.1} GB/s)   timestamps {ts:7.0} us/dispatch ({:5.1} GB/s, {:5.1} G weights/s)",
                orangu::gguf::ggml_type_name(ggml_type),
                best_wall,
                best_wall / f64::from(reps),
                mib * 1.048576 * f64::from(reps) / best_wall * 1e3,
                mib * 1.048576 / ts * 1e3,
                (in_dim * out_dim * n_tokens) as f64 / ts * 1e-3,
            );
        }
    }
}

/// The cost of a *dependent* dispatch on this device: `k` back-to-back
/// `add` dispatches over one small buffer (each reads what the last
/// wrote) in one submission, against the same `k` over `n` independent
/// buffers. If the dependent chain costs milliseconds a dispatch where the
/// independent set does not, the device (or its driver) serialises on a
/// per-dispatch latency that no amount of submission fusion removes.
/// `cargo test --profile release-with-debug --bin orangu-server
/// dependent_dispatch_latency -- --ignored --nocapture`.
#[test]
#[ignore]
fn dependent_dispatch_latency() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    for &elems in &[64usize, 17408, 1 << 20] {
        let buf = |label: &str| {
            vulkan.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (elems as u64) * 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let a = buf("probe a");
        let b = buf("probe b");
        let c = buf("probe c");
        let meta = vulkan.elem_meta_buffer(elems as u32, 0.0);
        // dependent: c = a + b, then b = a + c, alternating — each dispatch
        // reads what the previous one wrote.
        let bg_ab_c = vulkan.elem4_bind_group(&a, &b, &c, &meta);
        let bg_ac_b = vulkan.elem4_bind_group(&a, &c, &b, &meta);
        for &k in &[1usize, 16, 64, 256] {
            let mut best = f64::MAX;
            for _ in 0..3 {
                let mut encoder = vulkan.new_encoder("probe");
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                    for i in 0..k {
                        pass.set_pipeline(&vulkan.add_pipeline);
                        pass.set_bind_group(0, if i % 2 == 0 { &bg_ab_c } else { &bg_ac_b }, &[]);
                        pass.dispatch_workgroups(vulkan.strided_workgroups(elems), 1, 1);
                    }
                }
                let t = std::time::Instant::now();
                let _ = vulkan.submit_and_readback(encoder, &b, 0, 4);
                best = best.min(t.elapsed().as_secs_f64() * 1e3);
            }
            eprintln!(
                "elems {elems:8}: {k:4} dependent add dispatches: {best:8.2} ms  ({:.3} ms each)",
                best / k as f64
            );
        }
    }
}

/// Wall time of one decode-shaped fused FFN call at the 27B's shape, beside
/// the three matmuls issued separately — where a fused submission's time
/// goes. `cargo test --profile release-with-debug --bin orangu-server
/// fused_ffn_decode_call_cost -- --ignored --nocapture`.
#[test]
#[ignore]
fn fused_ffn_decode_call_cost() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    let (n_embd, ffn_len) = (5120usize, 17408usize);
    let mut seed = 0xFFF1_u64;
    let build = |in_dim: usize, out_dim: usize, seed: &mut u64| {
        let mut bytes = Vec::new();
        for _ in 0..out_dim {
            for _ in 0..(in_dim / 128) {
                bytes.extend(build_block(GGML_TYPE_PTQ1_0, seed));
            }
        }
        test_quant_matrix(&bytes, GGML_TYPE_PTQ1_0, in_dim, out_dim)
    };
    let gate = build(n_embd, ffn_len, &mut seed);
    let up = build(n_embd, ffn_len, &mut seed);
    let down = build(ffn_len, n_embd, &mut seed);
    let x: Vec<f32> = (0..n_embd)
        .map(|i| ((i * 37 % 23) as f32 - 11.0) * 0.031)
        .collect();
    let act = crate::engine::backend::vulkan::FfnActivation::Swiglu;
    // The Mali's devfreq clock (`simple_ondemand`, 72 MHz – 1 GHz on the
    // CIX P1): a decode loop that round-trips per projection looks idle to
    // the governor, so the clock it settles at is part of the measurement.
    // A sampler thread reads it every millisecond while a section runs and
    // the section reports the time-weighted histogram beside its min /
    // median / max call time.
    let cur_freq_path = std::fs::read_dir("/sys/class/devfreq")
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .find(|p| {
            std::fs::read_to_string(p.join("device/uevent"))
                .is_ok_and(|s| s.contains("DRIVER=mali"))
        })
        .map(|p| p.join("cur_freq"));
    let read_mhz = |path: &std::path::Path| -> Option<u64> {
        std::fs::read_to_string(path)
            .ok()?
            .trim()
            .parse::<u64>()
            .ok()
            .map(|hz| hz / 1_000_000)
    };
    let reps_env: usize = std::env::var("ORANGU_FFN_COST_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);
    let time = |label: &str, reps: usize, f: &mut dyn FnMut()| {
        f();
        let reps = reps.max(reps_env);
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let sampler = cur_freq_path.clone().map(|path| {
            let stop = stop.clone();
            std::thread::spawn(move || {
                let mut hist = std::collections::BTreeMap::<u64, u32>::new();
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    if let Some(mhz) = read_mhz(&path) {
                        *hist.entry(mhz).or_default() += 1;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                hist
            })
        });
        let mut calls = Vec::with_capacity(reps);
        for _ in 0..reps {
            let t = std::time::Instant::now();
            f();
            calls.push(t.elapsed().as_secs_f64() * 1e3);
        }
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        let hist = sampler.and_then(|h| h.join().ok()).unwrap_or_default();
        let total: u32 = hist.values().sum::<u32>().max(1);
        let clocks: Vec<String> = hist
            .iter()
            .map(|(mhz, n)| format!("{mhz} MHz {:.0}%", 100.0 * f64::from(*n) / f64::from(total)))
            .collect();
        let mean = calls.iter().sum::<f64>() / reps as f64;
        calls.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eprintln!(
            "{label:36} mean {mean:6.2} ms  min {:6.2}  median {:6.2}  max {:6.2}   clock: {}",
            calls[0],
            calls[reps / 2],
            calls[reps - 1],
            clocks.join(", ")
        );
    };
    time("fused_ffn_prefill (decode, 1 token)", 10, &mut || {
        vulkan
            .fused_ffn_prefill(&x, 1, &gate, &up, &down, act, None)
            .unwrap();
    });
    {
        let mut call = 0f64;
        let mut drop_ms = 0f64;
        for _ in 0..10 {
            let t = std::time::Instant::now();
            let r = vulkan.fused_ffn_prefill(&x, 1, &gate, &up, &down, act, None);
            call += t.elapsed().as_secs_f64() * 1e3;
            let t = std::time::Instant::now();
            drop(r);
            drop_ms += t.elapsed().as_secs_f64() * 1e3;
        }
        eprintln!(
            "fused call {:.2} ms, result drop {:.2} ms",
            call / 10.0,
            drop_ms / 10.0
        );
    }
    time("three separate matmul calls", 10, &mut || {
        let g = vulkan.matmul(&x, 1, &gate);
        let _ = vulkan.matmul(&x, 1, &up);
        let _ = vulkan.matmul(&g[..ffn_len], 1, &down);
    });
    time("one matmul, gate shape", 10, &mut || {
        vulkan.matmul(&x, 1, &gate);
    });
    time("one matmul, down shape", 10, &mut || {
        vulkan.matmul(&x[..1].repeat(ffn_len), 1, &down);
    });
    time("matmul_batch gate+up", 10, &mut || {
        use crate::engine::backend::MatmulOp;
        vulkan.matmul_batch(&[
            MatmulOp {
                x: &x,
                n_tokens: 1,
                w: &gate,
            },
            MatmulOp {
                x: &x,
                n_tokens: 1,
                w: &up,
            },
        ]);
    });
}

/// The device side of a recurrent layer's decode step — the delta rule,
/// the gated norm, the `gdn_v_grouped` regrouping, the fold and `ssm_out`
/// as one submission — against the host sequence step for step
/// (`delta_head_step`, `rmsnorm_inplace`, the gate, `Rotation::apply`, the
/// matmul), on both the returned output and the state it leaves behind,
/// over two consecutive tokens so a state error would compound. Three
/// shapes: no fold, a fold with the head permutation and identity signs,
/// and one with explicit signs; and both output gates.
#[test]
fn fused_recurrent_tail_matches_the_host_delta_rule_and_projection() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    if vulkan.q4_k_mmvq {
        eprintln!("skipping: ORANGU_Q4K_MMVQ selects the unfused fallback path");
        return;
    }
    use crate::engine::arch::qwen_hybrid::delta_head_step;
    use crate::engine::backend::vulkan::GatedDeltaInput;
    use crate::engine::hadamard::HadamardFold;
    use crate::engine::tensor;
    use orangu::gguf::GgufValue;

    let (n_k, n_v, hd) = (2usize, 8usize, 128usize);
    let value_dim = n_v * hd;
    let n_embd = 256usize;
    let eps = 1e-6f32;
    let mut seed = 0xDE17A_u64;
    let mut bytes = Vec::new();
    for _ in 0..n_embd {
        for _ in 0..(value_dim / 256) {
            bytes.extend(build_block(GGML_TYPE_Q4_K, &mut seed));
        }
    }
    let ssm_out = test_quant_matrix(&bytes, GGML_TYPE_Q4_K, value_dim, n_embd);
    let mut rnd = |scale: f32| (next_byte(&mut seed) as f32 - 128.0) / 128.0 * scale;
    let ssm_norm: Vec<f32> = (0..hd).map(|_| 1.0 + rnd(0.3)).collect();

    let fold_metadata = |sign_mode: &str| -> Vec<(String, GgufValue)> {
        let mut m = vec![
            ("prism.hadamard.version", GgufValue::U32(1)),
            ("prism.hadamard.block_size", GgufValue::U32(1024)),
            (
                "prism.hadamard.transform",
                GgufValue::String("normalized-sylvester-walsh-hadamard".into()),
            ),
            (
                "prism.hadamard.axis",
                GgufValue::String("input-last-dimension".into()),
            ),
            (
                "prism.hadamard.sign_mode",
                GgufValue::String(sign_mode.into()),
            ),
            (
                "prism.hadamard.weight_names",
                GgufValue::Array(vec![GgufValue::String("blk.0.ssm_out.weight".into())]),
            ),
            ("prism.hadamard.gdn_v_grouped", GgufValue::Bool(true)),
        ];
        if sign_mode == "explicit" {
            m.push((
                "prism.hadamard.sign_widths",
                GgufValue::Array(vec![GgufValue::I32(value_dim as i32)]),
            ));
            m.push((
                "prism.hadamard.sign_values",
                GgufValue::Array(
                    (0..value_dim)
                        .map(|i| GgufValue::I32(if (i * 11 + i / 7) % 3 == 1 { -1 } else { 1 }))
                        .collect(),
                ),
            ));
        }
        m.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
    };
    let rotations: Vec<Option<crate::engine::hadamard::Rotation>> = vec![
        None,
        Some(
            HadamardFold::from_metadata(&fold_metadata("identity"))
                .unwrap()
                .unwrap()
                .input_rotation("blk.0.ssm_out.weight", value_dim, Some((n_v, n_k)))
                .unwrap()
                .unwrap(),
        ),
        Some(
            HadamardFold::from_metadata(&fold_metadata("explicit"))
                .unwrap()
                .unwrap()
                .input_rotation("blk.0.ssm_out.weight", value_dim, Some((n_v, n_k)))
                .unwrap()
                .unwrap(),
        ),
    ];

    for (case, rot) in rotations.iter().enumerate() {
        for sigmoid_gate in [false, true] {
            let mut host_state: Vec<f32> = (0..n_v * hd * hd).map(|_| rnd(0.05)).collect();
            // The device side as the cache hands it over: the host copy
            // current before the first token (uploaded), the device's
            // after it (kept), and downloaded here only to compare.
            let mut dev_state = host_state.clone();
            let mut dev_conv: Vec<f32> = Vec::new();
            let mut mirror: Option<Box<dyn crate::engine::kv_cache::DeviceStateMirror>> = None;
            let mut fresh = crate::engine::kv_cache::Fresh::Host;
            for token in 0..2 {
                let q: Vec<f32> = (0..n_k * hd).map(|_| rnd(0.2)).collect();
                let k: Vec<f32> = (0..n_k * hd).map(|_| rnd(0.2)).collect();
                let v: Vec<f32> = (0..n_v * hd).map(|_| rnd(1.0)).collect();
                let beta: Vec<f32> = (0..n_v).map(|_| 0.5 + rnd(0.4)).collect();
                let decay: Vec<f32> = (0..n_v).map(|_| 0.9 + rnd(0.09)).collect();
                let z: Vec<f32> = (0..n_v * hd).map(|_| rnd(2.0)).collect();

                // Host: the delta rule per head, gated norm, gate.
                let mut attn = vec![0f32; value_dim];
                let mut scratch = vec![0f32; 2 * hd];
                for vh in 0..n_v {
                    let kh = vh % n_k;
                    let (sk, d) = scratch.split_at_mut(hd);
                    let out = &mut attn[vh * hd..(vh + 1) * hd];
                    delta_head_step(
                        &mut host_state[vh * hd * hd..(vh + 1) * hd * hd],
                        &q[kh * hd..(kh + 1) * hd],
                        &k[kh * hd..(kh + 1) * hd],
                        &v[vh * hd..(vh + 1) * hd],
                        beta[vh],
                        decay[vh],
                        out,
                        sk,
                        d,
                    );
                    tensor::rmsnorm_inplace(out, &ssm_norm, 1, hd, eps);
                    for (o, &zv) in out.iter_mut().zip(&z[vh * hd..(vh + 1) * hd]) {
                        *o *= if sigmoid_gate {
                            tensor::sigmoid(zv)
                        } else {
                            tensor::silu(zv)
                        };
                    }
                }
                if let Some(rot) = rot {
                    rot.apply(&mut attn, value_dim);
                }
                let expected = vulkan.matmul(&attn, 1, &ssm_out);

                let got = vulkan
                    .fused_recurrent_tail(GatedDeltaInput {
                        q: &q,
                        k: &k,
                        v: &v,
                        beta: &beta,
                        decay: &decay,
                        z: &z,
                        state: crate::engine::kv_cache::DeviceStateAccess {
                            host: &mut dev_state,
                            conv: &mut dev_conv,
                            mirror: &mut mirror,
                            fresh: &mut fresh,
                        },
                        ssm_norm: &ssm_norm,
                        eps,
                        sigmoid_gate,
                        n_k,
                        n_v,
                        head_dim: hd,
                        out_rotation: rot.as_ref(),
                        ssm_out: &ssm_out,
                        batch_slot: 0,
                    })
                    .expect("the device takes this shape");
                assert_eq!(got.len(), n_embd);
                for (i, (a, b)) in expected.iter().zip(&got).enumerate() {
                    let tol = 6e-2 * a.abs().max(1.0);
                    assert!(
                        (a - b).abs() <= tol,
                        "case {case} sigmoid={sigmoid_gate} token {token}: output {i}: host={a} device={b}"
                    );
                }
                assert_eq!(fresh, crate::engine::kv_cache::Fresh::Device);
                let mut downloaded = vec![0f32; dev_state.len()];
                mirror
                    .as_ref()
                    .expect("a mirror after a step")
                    .download(&mut downloaded, &mut []);
                for (i, (a, b)) in host_state.iter().zip(&downloaded).enumerate() {
                    assert!(
                        (a - b).abs() <= 1e-4 * a.abs().max(1e-2),
                        "case {case} sigmoid={sigmoid_gate} token {token}: state {i}: host={a} device={b}"
                    );
                }
            }
        }
    }
}

/// The q8-input ternary kernel (`shader_source_ternary_i8`) against the
/// dequantized product on main's per-32 q8 activation, and its device time
/// beside the f32-input kernel's, at the FFN gate shape.
/// `cargo test --profile release-with-debug --bin orangu-server
/// ternary_i8_kernel -- --nocapture` (the timing part needs `--ignored`).
fn ternary_i8_check(ggml_type: u32, in_dim: usize, out_dim: usize, n_tokens: usize, time: bool) {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    let mut seed = 0x1D07_u64;
    let mut bytes = Vec::new();
    for _ in 0..out_dim {
        for _ in 0..(in_dim / 128) {
            bytes.extend(build_block(ggml_type, &mut seed));
        }
    }
    let w = test_quant_matrix(&bytes, ggml_type, in_dim, out_dim);
    let x: Vec<f32> = (0..in_dim * n_tokens)
        .map(|i| (next_byte(&mut seed) as f32 - 128.0) / 64.0 + ((i * 7919) % 101) as f32 * 1e-4)
        .collect();
    let q8 = super::quantize_activation_q8(&x);
    // The dequantized product against exactly the q8 rounding.
    let wdq = crate::engine::quant::dequantize(ggml_type, &bytes, out_dim * in_dim).unwrap();
    let mut qx = vec![0f32; x.len()];
    for (blk, chunk) in q8.as_chunks::<10>().0.iter().enumerate() {
        let d = f32::from_bits(chunk[0]);
        for i in 0..32 {
            let byte = ((chunk[2 + i / 4] >> (8 * (i % 4))) & 0xFF) as u8 as i8;
            qx[blk * 32 + i] = d * byte as f32;
        }
    }
    let mut expected = vec![0f32; n_tokens * out_dim];
    for t in 0..n_tokens {
        for o in 0..out_dim {
            let mut s = 0f32;
            for e in 0..in_dim {
                s += wdq[o * in_dim + e] * qx[t * in_dim + e];
            }
            expected[t * out_dim + o] = s;
        }
    }
    let env_usize = |name: &str, default: usize| -> usize {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let (rows, groups) = (
        env_usize("ORANGU_TERNARY_IDOT_ROWS", 8),
        env_usize("ORANGU_TERNARY_IDOT_GROUPS", 4),
    );
    let mut src = crate::engine::backend::vulkan_shaders::shader_source_ternary_i8(
        ggml_type, rows, groups, true,
    )
    .unwrap();
    // Scratch, timing only: the weight word as one aligned load (wrong
    // for every other block), the bound of removing the unaligned read.
    match std::env::var("ORANGU_O_EXPERIMENT").as_deref() {
        Ok("aligned") => {
            src = src.replace(
                "let word = read_u32_at(byte_off);",
                "let word = weights[byte_off >> 2u];",
            );
        }
        Ok("noweights") => {
            src = src
                .replace(
                    "let word = read_u32_at(byte_off);",
                    "let word = byte_off * 0x01010101u;",
                )
                .replace(
                    "let dw = f16_to_f32(read_u16_even(byte_off - 2u - 4u * sub));",
                    "let dw = f32(byte_off & 15u);",
                );
        }
        Ok("noact") => {
            for j in 0..4 {
                src = src.replace(
                    &format!("let a{j} = q8_word(eb + e{j}) & zero;"),
                    &format!("let a{j} = (eb * {}u + e{j}) & zero;", j + 3),
                );
            }
            src = src.replace("let dx = q8_scale(eb + e0);", "let dx = f32(eb & 7u);");
        }
        Ok("skeleton") => {
            src = src
                .replace(
                    "let word = read_u32_at(byte_off);",
                    "let word = byte_off * 0x01010101u;",
                )
                .replace(
                    "let dw = f16_to_f32(read_u16_even(byte_off - 2u - 4u * sub));",
                    "let dw = f32(byte_off & 15u);",
                );
            for j in 0..4 {
                src = src.replace(
                    &format!("let a{j} = q8_word(eb + e{j}) & zero;"),
                    &format!("let a{j} = (eb * {}u + e{j}) & zero;", j + 3),
                );
            }
            src = src.replace("let dx = q8_scale(eb + e0);", "let dx = f32(eb & 7u);");
        }
        Ok("pretransposed") => {
            // The activation words used as the transposed words (timing
            // only): what a pre-transposed layout would save.
            for k in 0..4 {
                let sh = 8 * k;
                src = src.replace(
                    &format!("        let xq{k} = ((a0 >> {sh}u) & 0xFFu) | (((a1 >> {sh}u) & 0xFFu) << 8u) | (((a2 >> {sh}u) & 0xFFu) << 16u) | (((a3 >> {sh}u) & 0xFFu) << 24u);\n"),
                    &format!("        let xq{k} = a{k};\n"),
                );
            }
        }
        Ok("noscale") => {
            // The per-row scale as a constant (timing only): what the
            // `f16` read and convert per row per block cost.
            src = src.replace(
                "let dw = f16_to_f32(read_u16_even(byte_off - 2u - 4u * sub));",
                "let dw = 1.0;",
            );
        }
        Ok("nodots") => {
            for k in 1..4 {
                src = src.replace(
                    &format!(
                        "            isum = isum + dot4I8Packed((word >> {}u) & fm, xq{k});\n",
                        2 * k
                    ),
                    "",
                );
            }
        }
        _ => {}
    }
    let experiment = std::env::var_os("ORANGU_O_EXPERIMENT").is_some();
    let module = vulkan
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("probe ternary i8"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
    let layout = vulkan
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&vulkan.bind_group_layout)],
            immediate_size: 0,
        });
    let pipeline = vulkan
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("probe ternary i8"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
    let q8_buf = vulkan.upload_new_u32(&q8);
    let out = vulkan.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe out"),
        size: (n_tokens * out_dim * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let meta = vulkan.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe meta"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    vulkan.queue.write_buffer(
        &meta,
        0,
        bytemuck::cast_slice(&[
            in_dim as u32,
            out_dim as u32,
            n_tokens as u32,
            w.row_bytes() as u32,
        ]),
    );
    let (wchunk, woff, wsize) = vulkan.weight_buffer(&w);
    let bg = vulkan.matmul_bind_group(
        "probe ternary i8",
        (&wchunk, woff, wsize),
        BindSrc::Whole(&q8_buf),
        BindSrc::Whole(&out),
        BindSrc::Whole(&meta),
    );
    let workgroups = (out_dim.div_ceil(rows * groups) * n_tokens) as u32;
    let mut encoder = vulkan.new_encoder("probe");
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    let got = vulkan.submit_and_readback(encoder, &out, 0, n_tokens * out_dim);
    let mut worst = 0f32;
    for (i, (a, b)) in expected.iter().zip(&got).enumerate() {
        let tol = 1e-2 * a.abs().max(1.0);
        worst = worst.max((a - b).abs() / a.abs().max(1.0));
        assert!(
            experiment || (a - b).abs() <= tol,
            "type {ggml_type} [{in_dim} x {out_dim}] x {n_tokens}: output {i}: ref={a} gpu={b}"
        );
    }
    eprintln!("type {ggml_type} [{in_dim} x {out_dim}] x {n_tokens}: worst rel {worst:.5}");
    if time {
        // The best of a dozen bursts: the first are read at whatever clock
        // the device idles at, and it takes a while to hold the top one.
        let us = (0..12)
            .map(|_| {
                vulkan
                    .dispatch_kernel_us(&pipeline, &bg, (workgroups, 1, 1), 32)
                    .expect("timestamps")
            })
            .fold(f64::MAX, f64::min);
        let mib = bytes.len() as f64 / (1024.0 * 1024.0);
        eprintln!(
            "  i8 kernel {us:.0} us ({:.1} GB/s)",
            mib * 1.048576 / us * 1e3
        );
    }
}

/// The two-phase `PQ2_0` kernel (`shader_source_ternary_t`) over the
/// transposed activation layout its quantize kernel writes
/// (`shader_source_quantize_ternary_t`), against the dequantized product
/// at exactly the per-128 q8 rounding; timed with `time`.
fn ternary_t_check(in_dim: usize, out_dim: usize, n_tokens: usize, time: bool) {
    ternary_t_check_kind(in_dim, out_dim, n_tokens, time, false);
}

/// `octet`: the octet kernel (`shader_source_ternary_o`) over its own
/// layout instead of the two-phase one.
///
/// Both kernels are written for a 16-lane subgroup (`shader_source_
/// ternary_t`'s doc): a lane's block index, a subgroup's row slice and the
/// octet kernel's sixteen-`vec4`-plus-four row walk all count to 16. On a
/// wider subgroup half the rows come back zero and the rest sum blocks
/// twice — wrong by construction, as `attn_coop` is on a narrower one — so
/// the check runs only where the width is exactly 16, and says so.
fn ternary_t_check_kind(in_dim: usize, out_dim: usize, n_tokens: usize, time: bool, octet: bool) {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    if vulkan.subgroup_lanes != (16, 16) {
        eprintln!(
            "skipping: the 16-lane ternary probe kernels on subgroups of {}..={} lanes",
            vulkan.subgroup_lanes.0, vulkan.subgroup_lanes.1
        );
        return;
    }
    let ggml_type = GGML_TYPE_PQ2_0;
    let mut seed = 0x1D07_u64;
    let mut bytes = Vec::new();
    for _ in 0..out_dim {
        for _ in 0..(in_dim / 128) {
            bytes.extend(build_block(ggml_type, &mut seed));
        }
    }
    let w = test_quant_matrix(&bytes, ggml_type, in_dim, out_dim);
    let x: Vec<f32> = (0..in_dim * n_tokens)
        .map(|i| (next_byte(&mut seed) as f32 - 128.0) / 64.0 + ((i * 7919) % 101) as f32 * 1e-4)
        .collect();
    // The host's per-128 rounding, for the reference.
    let mut qx = vec![0f32; x.len()];
    for (blk, chunk) in x.as_chunks::<128>().0.iter().enumerate() {
        let amax = chunk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        for (i, &v) in chunk.iter().enumerate() {
            qx[blk * 128 + i] = d * (v * id).round().clamp(-127.0, 127.0);
        }
    }
    let wdq = crate::engine::quant::dequantize(ggml_type, &bytes, out_dim * in_dim).unwrap();
    let mut expected = vec![0f32; n_tokens * out_dim];
    for t in 0..n_tokens {
        for o in 0..out_dim {
            let mut s = 0f32;
            for e in 0..in_dim {
                s += wdq[o * in_dim + e] * qx[t * in_dim + e];
            }
            expected[t * out_dim + o] = s;
        }
    }
    let build = |label: &str, source: String, layout: &wgpu::BindGroupLayout| {
        let module = vulkan
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let pl = vulkan
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(layout)],
                immediate_size: 0,
            });
        vulkan
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
    };
    use crate::engine::backend::vulkan_shaders as sh;
    let words = if octet {
        sh::TERNARY_O_WORDS
    } else {
        sh::TERNARY_T_WORDS
    } as usize;
    let quantize = build(
        "probe quantize ternary t",
        if octet {
            sh::shader_source_quantize_ternary_o()
        } else {
            sh::shader_source_quantize_ternary_t()
        },
        &vulkan.elem3_bind_group_layout,
    );
    let mut kernel_src = if octet {
        sh::shader_source_ternary_o()
    } else {
        sh::shader_source_ternary_t()
    };
    // Scratch experiments on the octet kernel's source, timing only.
    match std::env::var("ORANGU_O_EXPERIMENT").as_deref() {
        Ok("noshuffle") => {
            kernel_src = kernel_src
                .replace("subgroupShuffle(mine, sa / 4u)", "mine")
                .replace("subgroupShuffle(mine, sb / 4u)", "mine")
                .replace("subgroupShuffle(mine, 14u)", "mine");
        }
        Ok("nodyn+noshuffle") => {
            kernel_src = kernel_src
                .replace("subgroupShuffle(mine, sa / 4u)", "mine")
                .replace("subgroupShuffle(mine, sb / 4u)", "mine")
                .replace("subgroupShuffle(mine, 14u)", "mine")
                .replace("let mine = v[l / 4u];", "let mine = v[0];");
            for r in 0..8 {
                kernel_src = kernel_src.replace(
                    &format!("weights[row{r} + wbase + 16u][l & 3u]"),
                    &format!("weights[row{r} + wbase + 16u][0]"),
                );
            }
        }
        Ok("fewdots") => {
            // Only field 0's dot per word: the ALU share.
            kernel_src = kernel_src
                .replace("subgroupShuffle(mine, sa / 4u)", "mine")
                .replace("subgroupShuffle(mine, sb / 4u)", "mine")
                .replace("subgroupShuffle(mine, 14u)", "mine");
            for m in 0..4 {
                for k in 1..4 {
                    kernel_src = kernel_src.replace(
                        &format!("            s{m} = s{m} + dot4I8Packed((v[{m}] >> {}u) & fm, x{m}_{k});\n", 2 * k),
                        "",
                    );
                }
            }
        }
        Ok("noextra") => {
            kernel_src = kernel_src
                .replace("subgroupShuffle(mine, sa / 4u)", "mine")
                .replace("subgroupShuffle(mine, sb / 4u)", "mine")
                .replace("subgroupShuffle(mine, 14u)", "mine");
            for k in 0..4 {
                let src = if k == 0 {
                    "ve & fm".to_string()
                } else {
                    format!("(ve >> {}u) & fm", 2 * k)
                };
                kernel_src = kernel_src.replace(
                    &format!("            se = se + dot4I8Packed({src}, xe_{k});\n"),
                    "",
                );
            }
        }
        Ok("scalar4") => {
            // The weights as `array<u32>`, four loads a lane.
            kernel_src = kernel_src
                .replace("subgroupShuffle(mine, sa / 4u)", "mine")
                .replace("subgroupShuffle(mine, sb / 4u)", "mine")
                .replace("subgroupShuffle(mine, 14u)", "mine")
                .replace(
                    "var<storage, read> weights: array<vec4<u32>>;",
                    "var<storage, read> weights: array<u32>;",
                );
            for r in 0..8 {
                kernel_src = kernel_src
                    .replace(
                        &format!("let v = weights[row{r} + wbase + l];"),
                        &format!("let wb = (row{r} + wbase + l) * 4u; let v = vec4<u32>(weights[wb], weights[wb + 1u], weights[wb + 2u], weights[wb + 3u]);"),
                    )
                    .replace(
                        &format!("weights[row{r} + wbase + 16u][l & 3u]"),
                        &format!("weights[(row{r} + wbase + 16u) * 4u + (l & 3u)]"),
                    );
            }
        }
        Ok("noweights") => {
            kernel_src = kernel_src
                .replace("subgroupShuffle(mine, sa / 4u)", "mine")
                .replace("subgroupShuffle(mine, sb / 4u)", "mine")
                .replace("subgroupShuffle(mine, 14u)", "mine");
            for r in 0..8 {
                kernel_src = kernel_src
                    .replace(
                        &format!("let v = weights[row{r} + wbase + l];"),
                        &format!("let v = vec4<u32>(row{r} * 0x01010101u + oct, l * 0x01010101u, oct * 0x01010101u, row{r} ^ l);"),
                    )
                    .replace(
                        &format!("let ve = weights[row{r} + wbase + 16u][l & 3u];"),
                        &format!("let ve = row{r} ^ oct;"),
                    );
            }
        }
        Ok("noact") => {
            kernel_src = kernel_src
                .replace("subgroupShuffle(mine, sa / 4u)", "mine")
                .replace("subgroupShuffle(mine, sb / 4u)", "mine")
                .replace("subgroupShuffle(mine, 14u)", "mine");
            for m in 0..4 {
                for k in 0..4 {
                    kernel_src = kernel_src.replace(
                        &format!("let x{m}_{k} = q8x[xb{m}_ + {k}u];"),
                        &format!("let x{m}_{k} = xb{m}_ * {}u + {k}u;", m + 1),
                    );
                }
                kernel_src = kernel_src.replace(
                    &format!("let xs{m} = bitcast<i32>(q8x[xb{m}_ + 4u]);"),
                    &format!("let xs{m} = i32(xb{m}_ & 7u);"),
                );
            }
            for k in 0..4 {
                kernel_src = kernel_src.replace(
                    &format!("let xe_{k} = select(0u, q8x[xbe + {k}u], extra);"),
                    &format!("let xe_{k} = select(0u, xbe + {k}u, extra);"),
                );
            }
            kernel_src = kernel_src.replace(
                "let xse = select(0, bitcast<i32>(q8x[xbe + 4u]), extra);",
                "let xse = select(0, i32(xbe & 3u), extra);",
            );
        }
        Ok("nodyn") => {
            kernel_src = kernel_src
                .replace("let mine = v[l / 4u];", "let mine = v[0];")
                .replace(
                    "weights[row{r} + wbase + 16u][l & 3u]",
                    "weights[row{r} + wbase + 16u][0]",
                );
            for r in 0..8 {
                kernel_src = kernel_src.replace(
                    &format!("weights[row{r} + wbase + 16u][l & 3u]"),
                    &format!("weights[row{r} + wbase + 16u][0]"),
                );
            }
        }
        _ => {}
    }
    let kernel = build("probe ternary t", kernel_src, &vulkan.bind_group_layout);
    let n_blocks = n_tokens * in_dim / 128;
    let x_buf = vulkan.upload_new(&x);
    let tq8 = vulkan.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe tq8"),
        size: (n_blocks as u64) * (words as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let qmeta = vulkan.cast_meta_buffer((n_tokens * in_dim) as u32, 0);
    let qbg = vulkan.elem3_bind_group(&x_buf, &tq8, &qmeta);
    let out = vulkan.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe out"),
        size: (n_tokens * out_dim * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let meta = vulkan.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe meta"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    vulkan.queue.write_buffer(
        &meta,
        0,
        bytemuck::cast_slice(&[
            in_dim as u32,
            out_dim as u32,
            n_tokens as u32,
            w.row_bytes() as u32,
        ]),
    );
    let (wchunk, woff, wsize) = vulkan.weight_buffer(&w);
    let bg = vulkan.matmul_bind_group(
        "probe ternary t",
        (&wchunk, woff, wsize),
        BindSrc::Whole(&tq8),
        BindSrc::Whole(&out),
        BindSrc::Whole(&meta),
    );
    let workgroups = (out_dim.div_ceil(32) * n_tokens) as u32;
    let mut encoder = vulkan.new_encoder("probe");
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&quantize);
        pass.set_bind_group(0, &qbg, &[]);
        pass.dispatch_workgroups(n_blocks as u32, 1, 1);
        pass.set_pipeline(&kernel);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    let got = vulkan.submit_and_readback(encoder, &out, 0, n_tokens * out_dim);
    // The layout itself, against the host's construction of it.
    let encoder = vulkan.new_encoder("probe tq8 readback");
    let got_t: Vec<u32> = vulkan
        .submit_and_readback(encoder, &tq8, 0, n_blocks * words)
        .iter()
        .map(|v| v.to_bits())
        .collect();
    for (blk, chunk) in x.as_chunks::<128>().0.iter().enumerate() {
        let amax = chunk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let at = blk * words;
        // The device's own scale (its division may differ by an ulp), so
        // the words compare exactly.
        let d = f32::from_bits(got_t[at]);
        assert!(
            (d - amax / 127.0).abs() <= 1e-6 * amax.max(1e-6),
            "block {blk} scale {d} vs {}",
            amax / 127.0
        );
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        let q: Vec<i32> = chunk
            .iter()
            .map(|&v| (v * id).round().clamp(-127.0, 127.0) as i32)
            .collect();
        assert_eq!(
            got_t[at + 1] as i32,
            q.iter().sum::<i32>(),
            "block {blk} sum"
        );
        for phase in 0..2usize {
            for i in 0..9usize {
                for k in 0..4usize {
                    let mut word = 0u32;
                    for tt in 0..4usize {
                        let j = (4 * i + tt) as i32 - if phase == 1 { 4 } else { 2 };
                        let qv = if (0..32).contains(&j) {
                            q[4 * j as usize + k]
                        } else {
                            0
                        };
                        word |= ((qv as u8) as u32) << (8 * tt);
                    }
                    let slot = if octet {
                        at + 4 + 45 * phase + 5 * i + k
                    } else {
                        at + 4 + 36 * phase + 4 * i + k
                    };
                    assert_eq!(
                        got_t[slot], word,
                        "block {blk} phase {phase} word {i} field {k}"
                    );
                }
                if octet {
                    let mut gsum = 0i32;
                    for k in 0..4usize {
                        for tt in 0..4usize {
                            let j = (4 * i + tt) as i32 - if phase == 1 { 4 } else { 2 };
                            if (0..32).contains(&j) {
                                gsum += q[4 * j as usize + k];
                            }
                        }
                    }
                    assert_eq!(
                        got_t[at + 4 + 45 * phase + 5 * i + 4] as i32,
                        gsum,
                        "block {blk} phase {phase} group {i} sum"
                    );
                }
            }
        }
    }
    let mut worst = 0f32;
    let experiment = std::env::var_os("ORANGU_O_EXPERIMENT").is_some();
    for (i, (a, b)) in expected.iter().zip(&got).enumerate() {
        let tol = 1e-2 * a.abs().max(1.0);
        worst = worst.max((a - b).abs() / a.abs().max(1.0));
        assert!(
            experiment || (a - b).abs() <= tol,
            "ternary {} [{in_dim} x {out_dim}] x {n_tokens}: output {i}: ref={a} gpu={b}",
            if octet { "o" } else { "t" }
        );
    }
    eprintln!(
        "ternary {} [{in_dim} x {out_dim}] x {n_tokens}: worst rel {worst:.5}",
        if octet { "o" } else { "t" }
    );
    if time {
        let us = (0..12)
            .map(|_| {
                vulkan
                    .dispatch_kernel_us(&kernel, &bg, (workgroups, 1, 1), 32)
                    .expect("timestamps")
            })
            .fold(f64::MAX, f64::min);
        let mib = bytes.len() as f64 / (1024.0 * 1024.0);
        eprintln!(
            "  ternary t kernel {us:.0} us ({:.1} GB/s)",
            mib * 1.048576 / us * 1e3
        );
        let qus = (0..6)
            .map(|_| {
                vulkan
                    .dispatch_kernel_us(&quantize, &qbg, (n_blocks as u32, 1, 1), 32)
                    .expect("timestamps")
            })
            .fold(f64::MAX, f64::min);
        eprintln!("  quantize t {qus:.0} us");
    }
}

/// Widths with an even block count only: an odd count puts odd rows two
/// bytes off the phase pattern (`shader_source_ternary_t`'s doc), and the
/// kernel is not offered for them.
#[test]
fn ternary_t_kernel_matches_the_dequantized_product() {
    ternary_t_check(1024, 2048, 3, false);
    ternary_t_check(768, 2055, 2, false);
    ternary_t_check(5120, 40, 1, false);
    ternary_t_check(5120, 33, 2, false);
    ternary_t_check(2304, 65, 1, false);
}

/// `cargo test --profile release-with-debug --bin orangu-server ternary_t_kernel_time -- --ignored --nocapture`
#[test]
#[ignore]
fn ternary_t_kernel_time() {
    for (in_dim, out_dim) in [(5120, 17408), (17408, 5120), (5120, 10240), (6144, 5120)] {
        ternary_t_check(in_dim, out_dim, 1, true);
    }
}

/// Block counts that are multiples of eight only.
#[test]
fn ternary_o_kernel_matches_the_dequantized_product() {
    ternary_t_check_kind(1024, 2048, 3, false, true);
    ternary_t_check_kind(2048, 2055, 2, false, true);
    ternary_t_check_kind(5120, 40, 1, false, true);
    ternary_t_check_kind(5120, 33, 2, false, true);
}

/// `cargo test --profile release-with-debug --bin orangu-server ternary_o_kernel_time -- --ignored --nocapture`
#[test]
#[ignore]
fn ternary_o_kernel_time() {
    for (in_dim, out_dim) in [(5120, 17408), (17408, 5120), (5120, 10240), (6144, 5120)] {
        ternary_t_check_kind(in_dim, out_dim, 1, true, true);
    }
}

#[test]
fn ternary_i8_kernel_matches_the_dequantized_product() {
    for ggml_type in [GGML_TYPE_PTQ1_0, GGML_TYPE_PQ2_0] {
        ternary_i8_check(ggml_type, 1024, 2048, 3, false);
        ternary_i8_check(ggml_type, 896, 2055, 2, false);
        ternary_i8_check(ggml_type, 5120, 40, 1, false);
    }
}

#[test]
#[ignore]
fn ternary_i8_kernel_time() {
    // `ORANGU_TERNARY_PROBE_SHAPES=in:out,in:out` replaces the FFN shapes.
    let shapes: Vec<(usize, usize)> = std::env::var("ORANGU_TERNARY_PROBE_SHAPES")
        .ok()
        .map(|v| {
            v.split(',')
                .filter_map(|s| {
                    let (i, o) = s.split_once(':')?;
                    Some((i.parse().ok()?, o.parse().ok()?))
                })
                .collect()
        })
        .unwrap_or_else(|| vec![(5120, 17408), (17408, 5120)]);
    for ggml_type in [GGML_TYPE_PTQ1_0, GGML_TYPE_PQ2_0] {
        for &(in_dim, out_dim) in &shapes {
            ternary_i8_check(ggml_type, in_dim, out_dim, 1, true);
        }
    }
}

/// The device's read bandwidth as a decode kernel sees it: every thread
/// of a large grid streams `vec4<u32>` loads from a 256 MiB buffer and
/// folds them into one word, best of several 8-repetition readings.
/// The ceiling a weight kernel's GB/s is judged against on this board.
///
/// `cargo test --release --bin orangu-server device_read_bandwidth_probe -- --ignored --nocapture`
#[test]
#[ignore]
fn device_read_bandwidth_probe() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    let words = 64usize << 20; // 256 MiB
    let buffer = vulkan.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe read bandwidth"),
        size: (words * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let out = vulkan.scratch_buffer(1 << 16);
    let meta = vulkan.elem_meta_buffer_aux((words / 4) as u32, 0);
    for (label, per_thread) in [("16 vec4 a thread", 16u32), ("64 vec4 a thread", 64)] {
        let src = format!(
            r#"
struct ElemMeta {{ len: u32, aux: u32, extra: f32, out_scale: f32 }}
@group(0) @binding(0) var<storage, read> x: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> y: array<u32>;
@group(0) @binding(2) var<uniform> em: ElemMeta;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let threads = nwg.x * 256u;
    var acc = vec4<u32>(0u);
    var i = gid.x;
    for (var k: u32 = 0u; k < {per_thread}u; k = k + 1u) {{
        acc = acc ^ x[i];
        i = i + threads;
    }}
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x9E3779B9u) {{ y[gid.x % 1024u] = 1u; }}
}}
"#
        );
        let module = vulkan
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("probe read bandwidth"),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
        let pipeline = vulkan
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("probe read bandwidth"),
                layout: Some(&vulkan.elem3_pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let bg = vulkan.elem3_bind_group(&buffer, &out, &meta);
        let workgroups = (words / 4 / per_thread as usize / 256) as u32;
        let us = (0..5)
            .map(|_| {
                vulkan
                    .dispatch_kernel_us(&pipeline, &bg, (workgroups, 1, 1), 8)
                    .expect("timestamps")
            })
            .fold(f64::INFINITY, f64::min);
        eprintln!(
            "read bandwidth, {label} ({workgroups} workgroups): {us:.0} us for 256 MiB = {:.1} GB/s",
            (words * 4) as f64 / us / 1e3
        );
    }
}

/// Prints the generated ternary kernel, for reading.
#[test]
#[ignore]
fn dump_ternary_kernel_source() {
    let src = crate::engine::backend::vulkan_shaders::shader_source_ternary_idot(
        GGML_TYPE_PTQ1_0,
        2,
        2,
        true,
    )
    .unwrap();
    eprintln!("{src}");
}

/// The start-up self-check's own probe cases through the ternary kernel
/// against the dequantized product, per case — what
/// `decode_kernel_agrees` sees, reproduced where it can be looked at.
/// `cargo test --profile release-with-debug --bin orangu-server
/// ternary_kernel_on_the_probe_cases -- --ignored --nocapture`.
#[test]
#[ignore]
fn ternary_kernel_on_the_probe_cases() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    for ggml_type in [GGML_TYPE_PTQ1_0, GGML_TYPE_PQ2_0] {
        let cases = vulkan.kernel_probe_cases(ggml_type).expect("cases");
        for (case, (x, n, w)) in cases.iter().enumerate() {
            let got = vulkan.matmul(x, *n, w);
            let exact = CpuBackend.matmul_dequant(x, *n, w);
            let scale = exact.iter().fold(1f32, |m, v| m.max(v.abs()));
            let (mut worst, mut at) = (0f32, 0usize);
            for (i, (a, b)) in got.iter().zip(&exact).enumerate() {
                if (a - b).abs() > worst {
                    worst = (a - b).abs();
                    at = i;
                }
            }
            eprintln!(
                "type {ggml_type} case {case} [{} x {}] x {n} ({}): worst {worst:.4} of {scale:.4} at {at} (t{} o{}): got {} exact {}",
                w.in_dim,
                w.out_dim,
                vulkan.ternary_idot_for(w, *n),
                at / w.out_dim,
                at % w.out_dim,
                got[at],
                exact[at]
            );
        }
    }
}

/// The gated-delta kernel's device time at the 27B's recurrent shape
/// (48 value heads of 128, 16 key heads): what one recurrent layer's
/// state step costs on its own, before any projection.
/// `cargo test --profile release-with-debug --bin orangu-server
/// gated_delta_kernel_time -- --ignored --nocapture`.
#[test]
#[ignore]
fn gated_delta_kernel_time() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    let n_v: usize = std::env::var("ORANGU_DELTA_PROBE_NV")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(48);
    let (n_k, hd) = (16usize, 128usize);
    let state_len = n_v * hd * hd;
    let inputs_len = 2 * n_k * hd + n_v * hd + 2 * n_v + n_v * hd;
    let tail_len = n_v * hd;
    let inputs_off = state_len + tail_len;
    let total = inputs_off + inputs_len;
    let mut seed = 0xD3174_u64;
    let mut rnd = |scale: f32| (next_byte(&mut seed) as f32 - 128.0) / 128.0 * scale;
    let mut scratch: Vec<f32> = (0..total).map(|_| rnd(0.1)).collect();
    for h in 0..n_v {
        scratch[inputs_off + 2 * n_k * hd + n_v * hd + h] = 0.5; // beta
        scratch[inputs_off + 2 * n_k * hd + n_v * hd + n_v + h] = 0.95; // decay
    }
    let scratch_buf = vulkan.upload_new(&scratch);
    let norm = vulkan.upload_new(&vec![1.0f32; hd]);
    let meta = vulkan.elem_meta_buffer_aux_extra(inputs_off as u32, 2, 1e-6);
    let bg = vulkan.elem4_bind_group(&norm, &norm, &scratch_buf, &meta);
    let module = vulkan
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("probe gated delta"),
            source: wgpu::ShaderSource::Wgsl(
                crate::engine::backend::vulkan_shaders::shader_source_gated_delta(
                    hd as u32, n_k as u32, n_v as u32, false,
                )
                .into(),
            ),
        });
    let pipeline = vulkan
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("probe gated delta"),
            layout: Some(&vulkan.elem4_pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
    let build = |label: &str, source: String| {
        let module = vulkan
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        vulkan
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&vulkan.elem4_pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
    };
    // Best of a few readings at 32 repetitions: the clock ramps during
    // the first.
    let best = |steps: &[super::ProbeDispatch<'_>]| {
        (0..4)
            .map(|_| vulkan.dispatch_sequence_us(steps, 32).expect("timestamps"))
            .fold(f64::INFINITY, f64::min)
    };
    let one = best(&[(&pipeline, &[&bg], (n_v as u32, 1, 1))]);
    eprintln!("gated delta [{n_v} x {hd} x {hd}] one workgroup per head: {one:.0} us");
    let norm = build(
        "probe gated delta norm",
        crate::engine::backend::vulkan_shaders::shader_source_gated_delta_norm(
            hd as u32, n_k as u32, n_v as u32, false,
        ),
    );
    for cols in [128u32, 64, 32, 16] {
        let split = build(
            "probe gated delta split",
            crate::engine::backend::vulkan_shaders::shader_source_gated_delta_split(
                hd as u32, n_k as u32, n_v as u32, cols,
            ),
        );
        let workgroups = n_v as u32 * hd as u32 / cols;
        let us = best(&[
            (&split, &[&bg], (workgroups, 1, 1)),
            (&norm, &[&bg], (n_v as u32, 1, 1)),
        ]);
        let alone = best(&[(&split, &[&bg], (workgroups, 1, 1))]);
        eprintln!(
            "gated delta [{n_v} x {hd} x {hd}] {cols} columns a workgroup ({workgroups} workgroups) + norm: {us:.0} us (the split alone {alone:.0})"
        );
    }
    let one = best(&[(&pipeline, &[&bg], (n_v as u32, 1, 1))]);
    eprintln!("gated delta [{n_v} x {hd} x {hd}] one workgroup per head, again: {one:.0} us");
}

/// The whole full-attention sub-layer on the device
/// (`fused_attention_layer`) against the host sequence: the four
/// projections, the per-head norms, RoPE, the cache write, attention over
/// a pre-seeded history, the sigmoid gate, the output fold and `wo` — over
/// three consecutive tokens, with and without the fold. Over the weight
/// types the model family ships: `Q4_K` reads the float input (or the
/// norm's q8, on main's i8 kernels), the ternary types the per-token q8
/// the recorder quantizes for them.
#[test]
fn fused_attention_layer_matches_the_host_sequence() {
    fused_attention_layer_check(GGML_TYPE_Q4_K);
}

#[test]
fn fused_attention_layer_matches_the_host_sequence_for_ptq1_0() {
    fused_attention_layer_check(GGML_TYPE_PTQ1_0);
}

#[test]
fn fused_attention_layer_matches_the_host_sequence_for_pq2_0() {
    fused_attention_layer_check(GGML_TYPE_PQ2_0);
}

fn fused_attention_layer_check(ggml_type: u32) {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    if vulkan.q4_k_mmvq {
        eprintln!("skipping: ORANGU_Q4K_MMVQ selects the unfused fallback path");
        return;
    }
    use crate::engine::backend::vulkan::FusedAttentionLayerInput;
    use crate::engine::hadamard::HadamardFold;
    use crate::engine::tensor;
    use orangu::gguf::GgufValue;

    let (n_head, n_head_kv, head_dim, rope_dim) = (8usize, 2usize, 128usize, 64usize);
    let attn_width = n_head * head_dim;
    let kv_dim = n_head_kv * head_dim;
    let n_embd = 1024usize;
    let eps = 1e-6f32;
    let rope_freq_base = 10000.0f32;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut seed = 0xA77E_u64;
    let matrix = |in_dim: usize, out_dim: usize, seed: &mut u64| {
        let mut bytes = Vec::new();
        for _ in 0..out_dim {
            for _ in 0..(in_dim / block_elems(ggml_type)) {
                bytes.extend(build_block(ggml_type, seed));
            }
        }
        test_quant_matrix(&bytes, ggml_type, in_dim, out_dim)
    };
    let wq = matrix(n_embd, attn_width, &mut seed);
    let wgate = matrix(n_embd, attn_width, &mut seed);
    let wk = matrix(n_embd, kv_dim, &mut seed);
    let wv = matrix(n_embd, kv_dim, &mut seed);
    let wo = matrix(attn_width, n_embd, &mut seed);
    let mut rnd = |scale: f32| (next_byte(&mut seed) as f32 - 128.0) / 128.0 * scale;
    let q_norm: Vec<f32> = (0..head_dim).map(|_| 1.0 + rnd(0.3)).collect();
    let k_norm: Vec<f32> = (0..head_dim).map(|_| 1.0 + rnd(0.3)).collect();
    let fold = HadamardFold::from_metadata(
        &[
            ("prism.hadamard.version", GgufValue::U32(1)),
            ("prism.hadamard.block_size", GgufValue::U32(1024)),
            (
                "prism.hadamard.transform",
                GgufValue::String("normalized-sylvester-walsh-hadamard".into()),
            ),
            (
                "prism.hadamard.axis",
                GgufValue::String("input-last-dimension".into()),
            ),
            (
                "prism.hadamard.sign_mode",
                GgufValue::String("identity".into()),
            ),
            (
                "prism.hadamard.weight_names",
                GgufValue::Array(vec![GgufValue::String("blk.0.attn_output.weight".into())]),
            ),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect::<Vec<_>>(),
    )
    .unwrap()
    .unwrap();
    let o_rot = fold
        .input_rotation("blk.0.attn_output.weight", attn_width, None)
        .unwrap()
        .unwrap();

    for (case, o_rot) in [None, Some(&o_rot)].into_iter().enumerate() {
        let capacity = 16;
        let mut kv_cache = crate::engine::kv_cache::KvCache::new_with_dims(capacity, &[kv_dim]);
        let mut reference_cache =
            crate::engine::kv_cache::KvCache::new_with_dims(capacity, &[kv_dim]);
        for _ in 0..3 {
            let k: Vec<f32> = (0..kv_dim).map(|_| rnd(1.0)).collect();
            let v: Vec<f32> = (0..kv_dim).map(|_| rnd(1.0)).collect();
            kv_cache.layers[0].push(&k, &v);
            reference_cache.layers[0].push(&k, &v);
        }
        for token in 0..3 {
            let pos = kv_cache.layers[0].len;
            let normed: Vec<f32> = (0..n_embd).map(|_| rnd(1.0)).collect();

            // Host, through the same device matmul so the weights read
            // identically.
            let mut q = vulkan.matmul(&normed, 1, &wq);
            let gate = vulkan.matmul(&normed, 1, &wgate);
            let mut k = vulkan.matmul(&normed, 1, &wk);
            let v = vulkan.matmul(&normed, 1, &wv);
            tensor::rmsnorm_inplace(&mut q, &q_norm, n_head, head_dim, eps);
            tensor::rope_apply_inplace(&mut q, n_head, head_dim, rope_dim, pos, rope_freq_base);
            tensor::rmsnorm_inplace(&mut k, &k_norm, n_head_kv, head_dim, eps);
            tensor::rope_apply_inplace(&mut k, n_head_kv, head_dim, rope_dim, pos, rope_freq_base);
            reference_cache.layers[0].push(&k, &v);
            let group = n_head / n_head_kv;
            let mut attn = vec![0f32; attn_width];
            for h in 0..n_head {
                let kv_head = h / group;
                let qh = &q[h * head_dim..(h + 1) * head_dim];
                let mut scores: Vec<f32> = (0..=pos)
                    .map(|p| {
                        tensor::dot(qh, reference_cache.layers[0].key_at(p, kv_head, head_dim))
                            * scale
                    })
                    .collect();
                tensor::softmax_inplace(&mut scores);
                let out = &mut attn[h * head_dim..(h + 1) * head_dim];
                for (p, &w) in scores.iter().enumerate() {
                    let vh = reference_cache.layers[0].value_at(p, kv_head, head_dim);
                    for (o, vi) in out.iter_mut().zip(vh) {
                        *o += w * vi;
                    }
                }
            }
            for (o, &g) in attn.iter_mut().zip(&gate) {
                *o *= tensor::sigmoid(g);
            }
            if let Some(rot) = o_rot {
                rot.apply(&mut attn, attn_width);
            }
            // `wo` from the dequantized weights: the batched `matmul` may
            // quantize its activations to 8 bits per op on this device
            // (`decode_mmvq`), which at the magnitudes a random `Q4_K`
            // layer produces is a several-percent difference of its own,
            // while the chain's `wo` reads the `f32` output in place.
            let expected = CpuBackend.matmul_dequant(&attn, 1, &wo);

            let got = vulkan
                .fused_attention_layer(FusedAttentionLayerInput {
                    normed: &normed,
                    wq: &wq,
                    wgate: &wgate,
                    wk: &wk,
                    wv: &wv,
                    q_norm: &q_norm,
                    k_norm: &k_norm,
                    n_head,
                    n_head_kv,
                    head_dim,
                    rope_dim,
                    rope_freq_base,
                    eps,
                    pos,
                    o_rotation: o_rot,
                    wo: &wo,
                    cache: &mut kv_cache.layers[0],
                    batch_slot: 0,
                })
                .expect("the device takes this shape");
            assert_eq!(got.len(), n_embd);
            assert_eq!(kv_cache.layers[0].len, pos + 1, "the cache advanced");
            // Against the row's scale, not each element's: a random `Q4_K`
            // layer's outputs are large and a near-cancelling element
            // among them differs between two summation orders by far more
            // than its own size.
            let scale_out = expected.iter().fold(0f32, |m, v| m.max(v.abs()));
            for (i, (a, b)) in expected.iter().zip(&got).enumerate() {
                assert!(
                    (a - b).abs() <= 2e-2 * scale_out,
                    "case {case} token {token}: output {i}: host={a} device={b} (row scale {scale_out})"
                );
            }
        }
    }
}

/// The fused fold + quantize kernel (`shader_source_hadamard_q8`) against
/// the host: the rotated row in place and its per-32 q8, from the row
/// itself and from `silu(a) · b` (the FFN's form), with explicit signs
/// and without.
#[test]
fn hadamard_q8_kernel_matches_the_host() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    use crate::engine::hadamard::HadamardFold;
    use orangu::gguf::GgufValue;
    let width = 5120usize;
    let fold_metadata = |sign_mode: &str| -> Vec<(String, GgufValue)> {
        let mut m = vec![
            ("prism.hadamard.version", GgufValue::U32(1)),
            ("prism.hadamard.block_size", GgufValue::U32(1024)),
            (
                "prism.hadamard.transform",
                GgufValue::String("normalized-sylvester-walsh-hadamard".into()),
            ),
            (
                "prism.hadamard.axis",
                GgufValue::String("input-last-dimension".into()),
            ),
            (
                "prism.hadamard.sign_mode",
                GgufValue::String(sign_mode.into()),
            ),
            (
                "prism.hadamard.weight_names",
                GgufValue::Array(vec![GgufValue::String("blk.0.ffn_down.weight".into())]),
            ),
        ];
        if sign_mode == "explicit" {
            m.push((
                "prism.hadamard.sign_widths",
                GgufValue::Array(vec![GgufValue::I32(width as i32)]),
            ));
            m.push((
                "prism.hadamard.sign_values",
                GgufValue::Array(
                    (0..width)
                        .map(|i| GgufValue::I32(if (i * 13 + i / 5) % 3 == 1 { -1 } else { 1 }))
                        .collect(),
                ),
            ));
        }
        m.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
    };
    let mut seed = 0x4AD4_u64;
    for sign_mode in ["identity", "explicit"] {
        let rot = HadamardFold::from_metadata(&fold_metadata(sign_mode))
            .unwrap()
            .unwrap()
            .input_rotation("blk.0.ffn_down.weight", width, None)
            .unwrap()
            .unwrap();
        for silu_mul in [false, true] {
            let mut rnd = |scale: f32| (next_byte(&mut seed) as f32 - 128.0) / 128.0 * scale;
            let a: Vec<f32> = (0..width).map(|_| rnd(2.0)).collect();
            let b: Vec<f32> = (0..width).map(|_| rnd(1.0)).collect();
            let x: Vec<f32> = (0..width).map(|_| rnd(1.0)).collect();
            let mut expected: Vec<f32> = if silu_mul {
                a.iter()
                    .zip(&b)
                    .map(|(&g, &u)| g / (1.0 + (-g).exp()) * u)
                    .collect()
            } else {
                x.clone()
            };
            rot.apply(&mut expected, width);
            let expected_q8 = super::quantize_activation_q8(&expected);

            let a_buf = vulkan.upload_new(&a);
            let b_buf = vulkan.upload_new(&b);
            let x_buf = vulkan.upload_new(&x);
            let q8_buf = vulkan.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("test q8"),
                size: (width / 32 * 10 * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let (pipeline, bg, workgroups) = vulkan.hadamard_q8_dispatch(
                &rot,
                silu_mul.then_some((BindSrc::from(&a_buf), BindSrc::from(&b_buf))),
                BindSrc::from(&x_buf),
                BindSrc::from(&q8_buf),
                width,
            );
            let mut encoder = vulkan.new_encoder("test hadamard q8");
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("test hadamard q8"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            let got = vulkan.submit_and_readback(encoder, &x_buf, 0, width);
            let encoder = vulkan.new_encoder("test hadamard q8 readback");
            let got_q8: Vec<u32> = vulkan
                .submit_and_readback(encoder, &q8_buf, 0, width / 32 * 10)
                .iter()
                .map(|v| v.to_bits())
                .collect();
            for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
                assert!(
                    (g - e).abs() <= 1e-4 * e.abs().max(1.0),
                    "{sign_mode} silu_mul={silu_mul}: element {i}: {g} vs {e}"
                );
            }
            for blk in 0..width / 32 {
                let (g, e) = (
                    &got_q8[blk * 10..blk * 10 + 10],
                    &expected_q8[blk * 10..blk * 10 + 10],
                );
                let (gd, ed) = (f32::from_bits(g[0]), f32::from_bits(e[0]));
                assert!(
                    (gd - ed).abs() <= 1e-5 * ed.abs().max(1e-6),
                    "{sign_mode} silu_mul={silu_mul}: block {blk} scale {gd} vs {ed}"
                );
                for w in 2..10 {
                    for k in 0..4 {
                        let gq = ((g[w] >> (8 * k)) & 0xFF) as u8 as i8;
                        let eq = ((e[w] >> (8 * k)) & 0xFF) as u8 as i8;
                        assert!(
                            (i32::from(gq) - i32::from(eq)).abs() <= 1,
                            "{sign_mode} silu_mul={silu_mul}: block {blk} word {w} byte {k}: {gq} vs {eq}"
                        );
                    }
                }
            }
        }
    }
}

/// The decode chain's fused residual add + norm
/// (`hybrid_record_add_norm`) against the host: the stream after the add,
/// and the norm of it, at the 27B's width and at a width past the wide
/// norm's straight-line slots.
#[test]
fn hybrid_add_norm_matches_the_host() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    use crate::engine::tensor;
    let eps = 1e-6f32;
    let mut seed = 0xADD0_u64;
    for n_embd in [5120usize, 9216] {
        let mut rnd = |scale: f32| (next_byte(&mut seed) as f32 - 128.0) / 128.0 * scale;
        let x: Vec<f32> = (0..n_embd).map(|_| rnd(1.0)).collect();
        let out: Vec<f32> = (0..n_embd).map(|_| rnd(0.5)).collect();
        let weight: Vec<f32> = (0..n_embd).map(|_| 1.0 + rnd(0.3)).collect();
        let out_buf = vulkan.upload_new(&out);
        let mut tok = vulkan.hybrid_decode_begin(&x, eps);
        let res = tok.res.clone();
        vulkan.hybrid_record_add_norm(&mut tok, (&out_buf, 0), &weight);
        let side = tok.side;
        let stream = vulkan.hybrid_decode_finish(tok, 0);
        assert_eq!(side, 1, "the add lands on the other residual buffer");
        let encoder = vulkan.new_encoder("test add+norm readback");
        let normed = vulkan.submit_and_readback(encoder, &res.n, 0, n_embd);

        let mut expected_stream: Vec<f32> = x.iter().zip(&out).map(|(a, b)| a + b).collect();
        for (i, (a, b)) in stream.iter().zip(&expected_stream).enumerate() {
            assert!(
                (a - b).abs() <= 1e-6,
                "width {n_embd} stream {i}: {a} vs {b}"
            );
        }
        tensor::rmsnorm_inplace(&mut expected_stream, &weight, 1, n_embd, eps);
        for (i, (a, b)) in normed.iter().zip(&expected_stream).enumerate() {
            assert!(
                (a - b).abs() <= 1e-4 * b.abs().max(1.0),
                "width {n_embd} normed {i}: {a} vs {b}"
            );
        }
    }
}

/// The whole recurrent sub-layer on the device (`fused_recurrent_layer`)
/// against the host sequence step for step — the input fold, the four
/// projections, the conv step over a rolling history, SiLU, the L2 norms
/// and the query scale, `beta`, `decay`, then the delta rule, gated norm,
/// output fold and `ssm_out` — over three consecutive tokens, on both the
/// output and the state and history the device keeps. With and without
/// the input fold (`blk.0.attn_qkv.weight`, identity signs), the output
/// fold with the head permutation, and both output gates. Over the
/// weight types as the attention layer's test.
#[test]
fn fused_recurrent_layer_matches_the_host_sequence() {
    fused_recurrent_layer_check(GGML_TYPE_Q4_K);
}

#[test]
fn fused_recurrent_layer_matches_the_host_sequence_for_ptq1_0() {
    fused_recurrent_layer_check(GGML_TYPE_PTQ1_0);
}

#[test]
fn fused_recurrent_layer_matches_the_host_sequence_for_pq2_0() {
    fused_recurrent_layer_check(GGML_TYPE_PQ2_0);
}

fn fused_recurrent_layer_check(ggml_type: u32) {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    if vulkan.q4_k_mmvq {
        eprintln!("skipping: ORANGU_Q4K_MMVQ selects the unfused fallback path");
        return;
    }
    use crate::engine::arch::qwen_hybrid::delta_head_step;
    use crate::engine::backend::vulkan::RecurrentLayerInput;
    use crate::engine::hadamard::HadamardFold;
    use crate::engine::kv_cache::{DeviceStateAccess, DeviceStateMirror, Fresh};
    use crate::engine::tensor;
    use orangu::gguf::GgufValue;

    let (n_k, n_v, hd) = (2usize, 8usize, 128usize);
    let key_dim = n_k * hd;
    let value_dim = n_v * hd;
    let conv_channels = 2 * key_dim + value_dim;
    let d_conv = 4usize;
    let n_embd = 1024usize;
    let eps = 1e-6f32;
    let mut seed = 0x5EC0D_u64;
    let matrix = |in_dim: usize, out_dim: usize, seed: &mut u64| {
        let mut bytes = Vec::new();
        for _ in 0..out_dim {
            for _ in 0..(in_dim / block_elems(ggml_type)) {
                bytes.extend(build_block(ggml_type, seed));
            }
        }
        test_quant_matrix(&bytes, ggml_type, in_dim, out_dim)
    };
    let wqkv = matrix(n_embd, conv_channels, &mut seed);
    let wgate = matrix(n_embd, value_dim, &mut seed);
    let wbeta = matrix(n_embd, n_v, &mut seed);
    let walpha = matrix(n_embd, n_v, &mut seed);
    let ssm_out = matrix(value_dim, n_embd, &mut seed);
    let mut rnd = |scale: f32| (next_byte(&mut seed) as f32 - 128.0) / 128.0 * scale;
    let ssm_norm: Vec<f32> = (0..hd).map(|_| 1.0 + rnd(0.3)).collect();
    let conv_kernel: Vec<f32> = (0..conv_channels * d_conv).map(|_| rnd(0.5)).collect();
    let dt_bias: Vec<f32> = (0..n_v).map(|_| rnd(1.0)).collect();
    let ssm_a: Vec<f32> = (0..n_v).map(|_| -0.5 - rnd(0.4).abs()).collect();

    let fold_metadata = |names: &[&str], grouped: bool| -> Vec<(String, GgufValue)> {
        vec![
            ("prism.hadamard.version", GgufValue::U32(1)),
            ("prism.hadamard.block_size", GgufValue::U32(1024)),
            (
                "prism.hadamard.transform",
                GgufValue::String("normalized-sylvester-walsh-hadamard".into()),
            ),
            (
                "prism.hadamard.axis",
                GgufValue::String("input-last-dimension".into()),
            ),
            (
                "prism.hadamard.sign_mode",
                GgufValue::String("identity".into()),
            ),
            (
                "prism.hadamard.weight_names",
                GgufValue::Array(
                    names
                        .iter()
                        .map(|n| GgufValue::String((*n).into()))
                        .collect(),
                ),
            ),
            ("prism.hadamard.gdn_v_grouped", GgufValue::Bool(grouped)),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect()
    };
    let fold = HadamardFold::from_metadata(&fold_metadata(
        &[
            "blk.0.attn_qkv.weight",
            "blk.0.attn_gate.weight",
            "blk.0.ssm_beta.weight",
            "blk.0.ssm_alpha.weight",
            "blk.0.ssm_out.weight",
        ],
        true,
    ))
    .unwrap()
    .unwrap();
    let in_rot = fold
        .input_rotation("blk.0.attn_qkv.weight", n_embd, None)
        .unwrap()
        .unwrap();
    let out_rot = fold
        .input_rotation("blk.0.ssm_out.weight", value_dim, Some((n_v, n_k)))
        .unwrap()
        .unwrap();

    for (case, (in_rot, ba_rot, out_rot)) in [
        (None, None, None),
        (Some(&in_rot), Some(&in_rot), Some(&out_rot)),
        // The served file: QKV and the gate folded, beta/alpha not.
        (Some(&in_rot), None, Some(&out_rot)),
    ]
    .into_iter()
    .enumerate()
    {
        for sigmoid_gate in [false, true] {
            let mut host_state: Vec<f32> = (0..n_v * hd * hd).map(|_| rnd(0.05)).collect();
            let mut host_conv: Vec<f32> = (0..conv_channels * (d_conv - 1))
                .map(|_| rnd(0.5))
                .collect();
            let mut dev_state = host_state.clone();
            let mut dev_conv = host_conv.clone();
            let mut mirror: Option<Box<dyn DeviceStateMirror>> = None;
            let mut fresh = Fresh::Host;
            for token in 0..3 {
                let normed: Vec<f32> = (0..n_embd).map(|_| rnd(1.0)).collect();

                // Host: the fold, the projections (through the same device
                // matmul, so the weights read identically), the conv step,
                // the norms, the delta rule.
                let mut x = normed.clone();
                if let Some(rot) = in_rot {
                    rot.apply(&mut x, n_embd);
                }
                let mixed = vulkan.matmul(&x, 1, &wqkv);
                let z = vulkan.matmul(&x, 1, &wgate);
                let x_ba = if ba_rot.is_some() { &x } else { &normed };
                let b = vulkan.matmul(x_ba, 1, &wbeta);
                let a = vulkan.matmul(x_ba, 1, &walpha);
                let hw = d_conv - 1;
                let mut conv = vec![0f32; conv_channels];
                for c in 0..conv_channels {
                    let mut sum = 0f32;
                    for t in 0..hw {
                        sum += host_conv[c * hw + t] * conv_kernel[c * d_conv + t];
                    }
                    sum += mixed[c] * conv_kernel[c * d_conv + hw];
                    conv[c] = tensor::silu(sum);
                    host_conv.copy_within(c * hw + 1..c * hw + hw, c * hw);
                    host_conv[c * hw + hw - 1] = mixed[c];
                }
                let (q, rest) = conv.split_at_mut(key_dim);
                let (k, v) = rest.split_at_mut(key_dim);
                for h in 0..n_k {
                    tensor::l2_norm_inplace(&mut q[h * hd..(h + 1) * hd], eps);
                    tensor::l2_norm_inplace(&mut k[h * hd..(h + 1) * hd], eps);
                }
                let q_scale = 1.0 / (hd as f32).sqrt();
                for qv in q.iter_mut() {
                    *qv *= q_scale;
                }
                let beta: Vec<f32> = b.iter().map(|&x| tensor::sigmoid(x)).collect();
                let decay: Vec<f32> = (0..n_v)
                    .map(|h| (tensor::softplus(a[h] + dt_bias[h]) * ssm_a[h]).exp())
                    .collect();
                let mut attn = vec![0f32; value_dim];
                let mut scratch = vec![0f32; 2 * hd];
                for vh in 0..n_v {
                    let kh = vh % n_k;
                    let (sk, d) = scratch.split_at_mut(hd);
                    let out = &mut attn[vh * hd..(vh + 1) * hd];
                    delta_head_step(
                        &mut host_state[vh * hd * hd..(vh + 1) * hd * hd],
                        &q[kh * hd..(kh + 1) * hd],
                        &k[kh * hd..(kh + 1) * hd],
                        &v[vh * hd..(vh + 1) * hd],
                        beta[vh],
                        decay[vh],
                        out,
                        sk,
                        d,
                    );
                    tensor::rmsnorm_inplace(out, &ssm_norm, 1, hd, eps);
                    for (o, &zv) in out.iter_mut().zip(&z[vh * hd..(vh + 1) * hd]) {
                        *o *= if sigmoid_gate {
                            tensor::sigmoid(zv)
                        } else {
                            tensor::silu(zv)
                        };
                    }
                }
                if let Some(rot) = out_rot {
                    rot.apply(&mut attn, value_dim);
                }
                let expected = vulkan.matmul(&attn, 1, &ssm_out);

                let got = vulkan
                    .fused_recurrent_layer(RecurrentLayerInput {
                        normed: &normed,
                        qkv_rotation: in_rot,
                        ba_rotation: ba_rot,
                        wqkv: &wqkv,
                        wgate: &wgate,
                        wbeta: &wbeta,
                        walpha: &walpha,
                        conv_kernel: &conv_kernel,
                        d_conv,
                        dt_bias: &dt_bias,
                        ssm_a: &ssm_a,
                        state: DeviceStateAccess {
                            host: &mut dev_state,
                            conv: &mut dev_conv,
                            mirror: &mut mirror,
                            fresh: &mut fresh,
                        },
                        ssm_norm: &ssm_norm,
                        eps,
                        sigmoid_gate,
                        n_k,
                        n_v,
                        head_dim: hd,
                        out_rotation: out_rot,
                        ssm_out: &ssm_out,
                        batch_slot: 0,
                    })
                    .expect("the device takes this shape");
                assert_eq!(got.len(), n_embd);
                assert_eq!(fresh, Fresh::Device);
                for (i, (a, b)) in expected.iter().zip(&got).enumerate() {
                    let tol = 6e-2 * a.abs().max(1.0);
                    assert!(
                        (a - b).abs() <= tol,
                        "case {case} sigmoid={sigmoid_gate} token {token}: output {i}: host={a} device={b}"
                    );
                }
                let mut down_state = vec![0f32; dev_state.len()];
                let mut down_conv = vec![0f32; dev_conv.len()];
                mirror
                    .as_ref()
                    .expect("a mirror after a step")
                    .download(&mut down_state, &mut down_conv);
                // The device's inputs to the delta rule differ from the
                // host's by the rounding of a Hadamard butterfly and a
                // projection each, and the state carries that forward.
                for (i, (a, b)) in host_conv.iter().zip(&down_conv).enumerate() {
                    assert!(
                        (a - b).abs() <= 1e-2 * a.abs().max(1e-2),
                        "case {case} sigmoid={sigmoid_gate} token {token}: history {i}: host={a} device={b}"
                    );
                }
                for (i, (a, b)) in host_state.iter().zip(&down_state).enumerate() {
                    assert!(
                        (a - b).abs() <= 1e-2 * a.abs().max(1e-2),
                        "case {case} sigmoid={sigmoid_gate} token {token}: state {i}: host={a} device={b}"
                    );
                }
            }
        }
    }
}

/// The SwiGLU form with a Hadamard-folded `down`: the intermediate is
/// rotated on the device before the down projection, and must match the
/// CPU sequence that rotates it with `hadamard::Rotation::apply` — at one
/// token (decode) and at several (prefill), in identity and explicit sign
/// mode, and with a fold this build refuses (a permuting rotation) giving
/// `None` rather than a wrong answer.
#[test]
fn fused_ffn_with_a_folded_down_projection_matches_the_host_rotation() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    if vulkan.q4_k_mmvq {
        eprintln!("skipping: ORANGU_Q4K_MMVQ selects the unfused fallback path");
        return;
    }
    use crate::engine::hadamard::HadamardFold;
    use orangu::gguf::GgufValue;

    let (n_embd, ffn_len) = (256usize, 2048usize);
    let mut seed = 0xFADE_u64;
    let mut build = |in_dim: usize, out_dim: usize| {
        let mut bytes = Vec::new();
        for _ in 0..out_dim {
            for _ in 0..(in_dim / 256) {
                bytes.extend(build_block(GGML_TYPE_Q4_K, &mut seed));
            }
        }
        test_quant_matrix(&bytes, GGML_TYPE_Q4_K, in_dim, out_dim)
    };
    let gate = build(n_embd, ffn_len);
    let up = build(n_embd, ffn_len);
    let down = build(ffn_len, n_embd);

    for sign_mode in ["identity", "explicit"] {
        let mut metadata = vec![
            ("prism.hadamard.version", GgufValue::U32(1)),
            ("prism.hadamard.block_size", GgufValue::U32(1024)),
            (
                "prism.hadamard.transform",
                GgufValue::String("normalized-sylvester-walsh-hadamard".into()),
            ),
            (
                "prism.hadamard.axis",
                GgufValue::String("input-last-dimension".into()),
            ),
            (
                "prism.hadamard.sign_mode",
                GgufValue::String(sign_mode.into()),
            ),
            (
                "prism.hadamard.weight_names",
                GgufValue::Array(vec![GgufValue::String("blk.0.ffn_down.weight".into())]),
            ),
        ];
        if sign_mode == "explicit" {
            metadata.push((
                "prism.hadamard.sign_widths",
                GgufValue::Array(vec![GgufValue::I32(ffn_len as i32)]),
            ));
            metadata.push((
                "prism.hadamard.sign_values",
                GgufValue::Array(
                    (0..ffn_len)
                        .map(|i| GgufValue::I32(if (i * 7 + i / 13) % 3 == 0 { -1 } else { 1 }))
                        .collect(),
                ),
            ));
        }
        let metadata: Vec<(String, GgufValue)> = metadata
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        let fold = HadamardFold::from_metadata(&metadata).unwrap().unwrap();
        let rot = fold
            .input_rotation("blk.0.ffn_down.weight", ffn_len, None)
            .unwrap()
            .unwrap();

        for n_tokens in [1usize, 5] {
            let x: Vec<f32> = (0..n_tokens * n_embd)
                .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
                .collect();
            // The host sequence: gate/up, SiLU·up, the rotation, down.
            let mut expected = vulkan.matmul(&x, n_tokens, &gate);
            let up_out = vulkan.matmul(&x, n_tokens, &up);
            for (g, u) in expected.iter_mut().zip(&up_out) {
                *g = crate::engine::tensor::silu(*g) * u;
            }
            rot.apply(&mut expected, ffn_len);
            let expected = vulkan.matmul(&expected, n_tokens, &down);

            let got = vulkan
                .fused_ffn_prefill(
                    &x,
                    n_tokens,
                    &gate,
                    &up,
                    &down,
                    crate::engine::backend::vulkan::FfnActivation::Swiglu,
                    Some(&rot),
                )
                .expect("fused path available without MMVQ");
            assert_eq!(got.len(), n_tokens * n_embd);
            for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
                let tol = 6e-2 * a.abs().max(1.0);
                assert!(
                    (a - b).abs() <= tol,
                    "{sign_mode} n_tokens={n_tokens}: mismatch at {i}: host={a} device={b}"
                );
            }
        }
    }

    // A rotation with a head permutation is not something the kernel does.
    let metadata: Vec<(String, GgufValue)> = vec![
        ("prism.hadamard.version", GgufValue::U32(1)),
        ("prism.hadamard.block_size", GgufValue::U32(1024)),
        (
            "prism.hadamard.transform",
            GgufValue::String("normalized-sylvester-walsh-hadamard".into()),
        ),
        (
            "prism.hadamard.axis",
            GgufValue::String("input-last-dimension".into()),
        ),
        (
            "prism.hadamard.sign_mode",
            GgufValue::String("identity".into()),
        ),
        (
            "prism.hadamard.weight_names",
            GgufValue::Array(vec![GgufValue::String("blk.0.ssm_out.weight".into())]),
        ),
        ("prism.hadamard.gdn_v_grouped", GgufValue::Bool(true)),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    let fold = HadamardFold::from_metadata(&metadata).unwrap().unwrap();
    let permuting = fold
        .input_rotation("blk.0.ssm_out.weight", ffn_len, Some((16, 4)))
        .unwrap()
        .unwrap();
    let x = vec![0.5f32; n_embd];
    assert!(
        vulkan
            .fused_ffn_prefill(
                &x,
                1,
                &gate,
                &up,
                &down,
                crate::engine::backend::vulkan::FfnActivation::Swiglu,
                Some(&permuting),
            )
            .is_none()
    );
}

#[test]
fn fused_ffn_prefill_matches_the_unfused_sequence_multi_chunk() {
    // At this width the unfused sequence's matmuls take the integer-dot
    // kernel through the generic batch path, and the chain quantizes its
    // intermediates at different points than the CPU sequence does; on
    // synthetic weights with outputs in the billions that is not a 6%
    // comparison. The chain's structure is what this checks, on the float
    // kernels; the integer-dot kernel has its own test.
    if let Some(vulkan) = shared_vulkan() {
        vulkan.without_prefill_mmq(|| cross_check_fused_ffn_prefill(192));
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
/// The **llama-family** shape of a fused decode layer, against the
/// step-by-step CPU sequence: no per-head Q/K norms, no post-norm on either
/// residual, SwiGLU rather than GEGLU, NORM rather than NEOX rope pairing,
/// and optionally Q/K/V projection biases.
///
/// Every one of those is a branch the decode chain grew for this family and
/// that **nothing executed** until this test. The existing full-layer check
/// above takes the other arm of each: gemma passes `Some(...)` for every
/// norm and `None` for every bias, so "gemma is byte-identical" — the check
/// each of those six changes was landed against — proves only that the new
/// arms do not disturb the old path. It never enters one. See LESSONS §64;
/// the first caller of these arms produced token soup, and this is the test
/// that should have existed before it.
fn cross_check_fused_layer_llama_shaped(biases: &str) {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };

    let n_embd = 24;
    let n_head = 4;
    let n_head_kv = 2;
    let head_dim = 6;
    let rope_dim = 6;
    let ffn_len = 16;
    let kv_dim = n_head_kv * head_dim;
    let capacity = 64;
    let eps = 1e-6;
    let rope_freq_base = 10000.0;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let pairing = crate::engine::tensor::RopeLayout::Norm;
    let rope = crate::engine::tensor::RopeParams {
        rope_dim,
        freq_base: rope_freq_base,
        layout: pairing,
        ..crate::engine::tensor::RopeParams::default()
    };

    let mut seed = 0x11A3_A0DE_u64;
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
    let wk = build(n_embd, kv_dim, &mut seed);
    let wv = build(n_embd, kv_dim, &mut seed);
    let wo = build(n_head * head_dim, n_embd, &mut seed);
    let ffn_norm = rand_vec(n_embd, &mut seed);
    let ffn_gate = build(n_embd, ffn_len, &mut seed);
    let ffn_up = build(n_embd, ffn_len, &mut seed);
    let ffn_down = build(ffn_len, n_embd, &mut seed);
    // Scaled the way the prefill bias check scales its own, for the reason
    // that check documents: an unscaled bias dominates the projections and
    // turns softmax into a hard argmax, and the mismatch that reports is a
    // knife-edge rather than a defect.
    let bias_vec = |len: usize, seed: &mut u64| -> Vec<f32> {
        rand_vec(len, seed).into_iter().map(|v| v * 0.25).collect()
    };
    let q_bias = biases
        .contains('q')
        .then(|| bias_vec(n_head * head_dim, &mut seed));
    let k_bias = biases.contains('k').then(|| bias_vec(kv_dim, &mut seed));
    let v_bias = biases.contains('v').then(|| bias_vec(kv_dim, &mut seed));

    let mut kv_cache = crate::engine::kv_cache::KvCache::new_with_dims(capacity, &[kv_dim]);
    let mut reference_cache = crate::engine::kv_cache::KvCache::new_with_dims(capacity, &[kv_dim]);
    for _ in 0..3 {
        let k = rand_vec(kv_dim, &mut seed);
        let v = rand_vec(kv_dim, &mut seed);
        kv_cache.layers[0].push(&k, &v);
        reference_cache.layers[0].push(&k, &v);
    }

    for step in 0..8 {
        let pos = kv_cache.layers[0].len;
        let x = rand_vec(n_embd, &mut seed);

        // CPU reference: exactly `LlamaModel::run_layers`' statement order.
        let mut normed = x.clone();
        crate::engine::tensor::rmsnorm_inplace(&mut normed, &attn_norm, 1, n_embd, eps);

        let mut q = CpuBackend.matmul_dequant(&normed, 1, &wq);
        let mut k = CpuBackend.matmul_dequant(&normed, 1, &wk);
        let mut v = CpuBackend.matmul_dequant(&normed, 1, &wv);
        if let Some(b) = &q_bias {
            crate::engine::tensor::add_bias_per_row(&mut q, b, 1);
        }
        if let Some(b) = &k_bias {
            crate::engine::tensor::add_bias_per_row(&mut k, b, 1);
        }
        if let Some(b) = &v_bias {
            crate::engine::tensor::add_bias_per_row(&mut v, b, 1);
        }
        crate::engine::tensor::rope_apply_params_inplace(
            &mut q, n_head, head_dim, pos, None, &rope,
        );
        crate::engine::tensor::rope_apply_params_inplace(
            &mut k, n_head_kv, head_dim, pos, None, &rope,
        );
        reference_cache.layers[0].push(&k, &v);

        let mut attn = vec![0f32; n_head * head_dim];
        crate::engine::attention::multi_head_attention(
            &mut attn,
            &q,
            &reference_cache.layers[0],
            n_head,
            n_head / n_head_kv,
            head_dim,
            scale,
            |_| (0, pos),
        );

        // No post-norm on either residual — a plain add, not a norm with
        // weights of one.
        let mut xr = x.clone();
        let attn_proj = CpuBackend.matmul_dequant(&attn, 1, &wo);
        crate::engine::tensor::add_inplace(&mut xr, &attn_proj);

        let mut normed2 = xr.clone();
        crate::engine::tensor::rmsnorm_inplace(&mut normed2, &ffn_norm, 1, n_embd, eps);
        let gate = CpuBackend.matmul_dequant(&normed2, 1, &ffn_gate);
        let up = CpuBackend.matmul_dequant(&normed2, 1, &ffn_up);
        let mut act: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
            .collect();
        act = CpuBackend.matmul_dequant(&act, 1, &ffn_down);
        crate::engine::tensor::add_inplace(&mut xr, &act);
        let expected = xr;

        let got = vulkan.fused_layer(FusedLayerInput {
            stop_at_ffn_norm: false,
            yarn: RopeYarn::IDENTITY,
            normalize_v: false, // llama does not normalize V
            attn_gate: None,
            q_bias: q_bias.as_deref(),
            pairing,
            activation: FfnActivation::Swiglu,
            x: GpuInput::Cpu(&x),
            attn_norm: &attn_norm,
            wq: &wq,
            q_norm: None,
            kv: Some(FusedAttnProjection {
                k_bias: k_bias.as_deref(),
                v_bias: v_bias.as_deref(),
                wk: &wk,
                k_norm: None,
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
            window: None,
            scale,
            cache: &mut kv_cache.layers[0],
            wo: &wo,
            attn_post_norm: None,
            ffn_norm: &ffn_norm,
            ffn_gate: &ffn_gate,
            ffn_up: &ffn_up,
            ffn_gate_up: None,
            ffn_down: &ffn_down,
            ffn_post_norm: None,
            ple: None,
            layer_output_scale: None,
            post_norm_eps: None,
            batch_slot: 0,
            attn_ts: None,
        });

        assert_eq!(got.len(), expected.len());
        for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 6e-2 * a.abs().max(1.0),
                "biases={biases:?} step={step} pos={pos}: mismatch at {i}: \
                     cpu={a} fused={b}"
            );
        }
    }
}

/// The **muse-glimmer** shape of a fused decode layer, against the
/// step-by-step CPU sequence `MuseModel::run_layers` runs: per-head Q/K
/// norms, **no rotation** on a full-attention layer (`rope_dim: 0`), a
/// sigmoid gate on attention's output projected from the same normed
/// input, sandwich post-norms with **their own epsilon**, SwiGLU. Three
/// things the chain learned for this architecture, each load-bearing:
/// a rotated head, an ungated output or the wrong epsilon on the
/// post-norms each fails this at the first position.
#[test]
fn fused_layer_muse_shaped_matches_cpu_reference() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };

    let n_embd = 24;
    let n_head = 4;
    let n_head_kv = 2;
    let head_dim = 6;
    let ffn_len = 16;
    let kv_dim = n_head_kv * head_dim;
    let capacity = 64;
    let eps = 1e-6;
    let post_eps = 0.5;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut seed = 0x0A5E_0A5E_u64;
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
    let wk = build(n_embd, kv_dim, &mut seed);
    let wv = build(n_embd, kv_dim, &mut seed);
    let w_gate = build(n_embd, n_head * head_dim, &mut seed);
    let q_norm = rand_vec(head_dim, &mut seed);
    let k_norm = rand_vec(head_dim, &mut seed);
    let wo = build(n_head * head_dim, n_embd, &mut seed);
    let attn_post_norm = rand_vec(n_embd, &mut seed);
    let ffn_norm = rand_vec(n_embd, &mut seed);
    let ffn_gate = build(n_embd, ffn_len, &mut seed);
    let ffn_up = build(n_embd, ffn_len, &mut seed);
    let ffn_down = build(ffn_len, n_embd, &mut seed);
    let ffn_post_norm = rand_vec(n_embd, &mut seed);

    let mut kv_cache = crate::engine::kv_cache::KvCache::new_with_dims(capacity, &[kv_dim]);
    let mut reference_cache = crate::engine::kv_cache::KvCache::new_with_dims(capacity, &[kv_dim]);
    for _ in 0..3 {
        let k = rand_vec(kv_dim, &mut seed);
        let v = rand_vec(kv_dim, &mut seed);
        kv_cache.layers[0].push(&k, &v);
        reference_cache.layers[0].push(&k, &v);
    }

    for step in 0..6 {
        let pos = kv_cache.layers[0].len;
        let x = rand_vec(n_embd, &mut seed);

        let mut normed = x.clone();
        crate::engine::tensor::rmsnorm_inplace(&mut normed, &attn_norm, 1, n_embd, eps);
        let mut q = CpuBackend.matmul_dequant(&normed, 1, &wq);
        let mut k = CpuBackend.matmul_dequant(&normed, 1, &wk);
        let v = CpuBackend.matmul_dequant(&normed, 1, &wv);
        let gate = CpuBackend.matmul_dequant(&normed, 1, &w_gate);
        crate::engine::tensor::rmsnorm_inplace(&mut q, &q_norm, n_head, head_dim, eps);
        crate::engine::tensor::rmsnorm_inplace(&mut k, &k_norm, n_head_kv, head_dim, eps);
        // No rotation: a full-attention layer of this architecture; and no
        // norm on V, as on the llama family.
        reference_cache.layers[0].push(&k, &v);

        let mut attn = vec![0f32; n_head * head_dim];
        crate::engine::attention::multi_head_attention(
            &mut attn,
            &q,
            &reference_cache.layers[0],
            n_head,
            n_head / n_head_kv,
            head_dim,
            scale,
            |_| (0, pos),
        );
        for (o, g) in attn.iter_mut().zip(gate.iter()) {
            *o *= crate::engine::tensor::sigmoid(*g);
        }

        let mut xr = x.clone();
        let mut attn_proj = CpuBackend.matmul_dequant(&attn, 1, &wo);
        crate::engine::tensor::rmsnorm_inplace(
            &mut attn_proj,
            &attn_post_norm,
            1,
            n_embd,
            post_eps,
        );
        crate::engine::tensor::add_inplace(&mut xr, &attn_proj);

        let mut normed2 = xr.clone();
        crate::engine::tensor::rmsnorm_inplace(&mut normed2, &ffn_norm, 1, n_embd, eps);
        let g = CpuBackend.matmul_dequant(&normed2, 1, &ffn_gate);
        let u = CpuBackend.matmul_dequant(&normed2, 1, &ffn_up);
        let act: Vec<f32> = g
            .iter()
            .zip(u.iter())
            .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
            .collect();
        let mut ffn_out = CpuBackend.matmul_dequant(&act, 1, &ffn_down);
        crate::engine::tensor::rmsnorm_inplace(&mut ffn_out, &ffn_post_norm, 1, n_embd, post_eps);
        crate::engine::tensor::add_inplace(&mut xr, &ffn_out);
        let expected = xr;

        let got = vulkan.fused_layer(FusedLayerInput {
            stop_at_ffn_norm: false,
            yarn: RopeYarn::IDENTITY,
            normalize_v: false,
            attn_gate: Some(&w_gate),
            q_bias: None,
            pairing: crate::engine::tensor::RopeLayout::Neox,
            activation: FfnActivation::Swiglu,
            x: GpuInput::Cpu(&x),
            attn_norm: &attn_norm,
            wq: &wq,
            q_norm: Some(&q_norm),
            kv: Some(FusedAttnProjection {
                k_bias: None,
                v_bias: None,
                wk: &wk,
                k_norm: Some(&k_norm),
                wv: Some(&wv),
            }),
            n_head,
            n_head_kv,
            head_dim,
            rope_dim: 0,
            rope_freq_base: 10000.0,
            freq_factors: None,
            eps,
            pos,
            window_start: 0,
            window: None,
            scale,
            cache: &mut kv_cache.layers[0],
            wo: &wo,
            attn_post_norm: Some(&attn_post_norm),
            ffn_norm: &ffn_norm,
            ffn_gate: &ffn_gate,
            ffn_up: &ffn_up,
            ffn_gate_up: None,
            ffn_down: &ffn_down,
            ffn_post_norm: Some(&ffn_post_norm),
            ple: None,
            layer_output_scale: None,
            post_norm_eps: Some(post_eps),
            batch_slot: 0,
            attn_ts: None,
        });

        assert_eq!(got.len(), expected.len());
        for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 6e-2 * a.abs().max(1.0),
                "step={step} pos={pos}: mismatch at {i}: cpu={a} fused={b}"
            );
        }
    }
}

/// The **decode** attention half on the llama shape — no per-head Q/K
/// norms, NORM pairing — against a step-by-step CPU sequence.
///
/// **This test's own reference is not yet right, so nothing may be
/// concluded from its failure.** Run it with the *gemma* shape — real
/// per-head Q/K norms, at `pos = 3` with a populated cache — and it still
/// fails, while `fused_layer_matches_cpu_reference_full_layer_with_ple`
/// exercises that exact configuration through the full layer and passes. A
/// check that rejects a configuration known to be correct is measuring its
/// own reference, not the engine.
///
/// So the earlier reading — "the defect is in the attention half, and the
/// `RopeOnly` stages are where it lives" — is **withdrawn**. What survives
/// is only what a *passing* test established:
/// `fused_post_attention_decode_matches_prefill_on_the_llama_shape` shows
/// the post-attention half correct for this shape, so items 4 and 5 are
/// clear. Items 1, 2, 3 and 6 are unattributed again.
///
/// Fixing this test is the next step, and it comes before any further
/// attribution. The likely suspects in the reference, none yet checked:
/// whether `fused_attention`'s standalone wrapper returns the same quantity
/// this computes; whether the KV cast (`kv_storage` may be `F16`/`Q8_0`)
/// makes an exact-`f32` reference the wrong comparison; and whether the
/// window closure here matches what the chain derives internally.
///
/// The `pos = 0` empty-cache case is a separate matter and probably a real
/// engine bug: there the reference is not in doubt, because attention over
/// one position is V by definition, and the fused path returns a constant
/// 3.013x less than V for *both* shapes. Production never reaches it — a
/// decode step always follows a prefill — so it is filed, not urgent.
#[test]
fn fused_attention_decode_matches_cpu_on_the_llama_shape() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    let n_embd = 24;
    let n_head = 4;
    let n_head_kv = 2;
    let head_dim = 6;
    let rope_dim = 6;
    let kv_dim = n_head_kv * head_dim;
    let eps = 1e-6;
    let rope_freq_base = 10000.0;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let pairing = crate::engine::tensor::RopeLayout::Norm;
    let rope = crate::engine::tensor::RopeParams {
        rope_dim,
        freq_base: rope_freq_base,
        layout: pairing,
        ..crate::engine::tensor::RopeParams::default()
    };
    let mut seed = 0xA77E_4711_u64;
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
    let wq = build(n_embd, n_head * head_dim, &mut seed);
    let wk = build(n_embd, kv_dim, &mut seed);
    let wv = build(n_embd, kv_dim, &mut seed);
    let attn_norm = rand_vec(n_embd, &mut seed);

    let mut kv_cache = crate::engine::kv_cache::KvCache::new_with_dims(64, &[kv_dim]);
    let mut ref_cache = crate::engine::kv_cache::KvCache::new_with_dims(64, &[kv_dim]);
    for _ in 0..3 {
        let k = rand_vec(kv_dim, &mut seed);
        let v = rand_vec(kv_dim, &mut seed);
        kv_cache.layers[0].push(&k, &v);
        ref_cache.layers[0].push(&k, &v);
    }
    let x = rand_vec(n_embd, &mut seed);
    let pos = kv_cache.layers[0].len;

    let mut normed = x.clone();
    crate::engine::tensor::rmsnorm_inplace(&mut normed, &attn_norm, 1, n_embd, eps);
    let mut q = CpuBackend.matmul_dequant(&normed, 1, &wq);
    let mut k = CpuBackend.matmul_dequant(&normed, 1, &wk);
    let v = CpuBackend.matmul_dequant(&normed, 1, &wv);
    crate::engine::tensor::rope_apply_params_inplace(&mut q, n_head, head_dim, pos, None, &rope);
    crate::engine::tensor::rope_apply_params_inplace(&mut k, n_head_kv, head_dim, pos, None, &rope);
    ref_cache.layers[0].push(&k, &v);
    let mut expected = vec![0f32; n_head * head_dim];
    crate::engine::attention::multi_head_attention(
        &mut expected,
        &q,
        &ref_cache.layers[0],
        n_head,
        n_head / n_head_kv,
        head_dim,
        scale,
        |_| (0, pos),
    );

    let got = vulkan.fused_attention(FusedAttnInput {
        projections_ready: false,
        yarn: RopeYarn::IDENTITY,
        normalize_v: false, // llama does not normalize V
        attn_gate: None,
        q_bias: None,
        pairing,
        normed: GpuInput::Cpu(&normed),
        normed_q8: None,
        wq: &wq,
        q_norm: None,
        kv: Some(FusedAttnProjection {
            k_bias: None,
            v_bias: None,
            wk: &wk,
            k_norm: None,
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
        window: None,
        scale,
        cache: &mut kv_cache.layers[0],
        batch_slot: 0,
        attn_ts: None,
    });

    assert_eq!(got.len(), expected.len());
    for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
        assert!(
            (a - b).abs() <= 6e-2 * a.abs().max(1.0),
            "mismatch at {i}: cpu={a} fused={b}"
        );
    }
}

/// The **decode** post-attention chain against the **prefill** one, on the
/// llama shape, at one token.
///
/// The tightest possible repro for the `fused_layer_llama_shaped_*` failure:
/// the prefill chain has taken optional post-norms and a SwiGLU switch since
/// G2's first increment and is what `llama.rs` runs in production, so it is
/// a known-good reference for exactly this configuration. Anything the
/// decode chain does differently here is the decode chain's bug, with no
/// attention, no RoPE and no KV cache in the way.
#[test]
fn fused_post_attention_decode_matches_prefill_on_the_llama_shape() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    let n_embd = 24;
    let ffn_len = 16;
    let eps = 1e-6;
    let mut seed = 0x5EED_B00C_u64;
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
    let wo = build(n_embd, n_embd, &mut seed);
    let ffn_norm = rand_vec(n_embd, &mut seed);
    let ffn_gate = build(n_embd, ffn_len, &mut seed);
    let ffn_up = build(n_embd, ffn_len, &mut seed);
    let ffn_down = build(ffn_len, n_embd, &mut seed);
    let attn_out = rand_vec(n_embd, &mut seed);
    let residual = rand_vec(n_embd, &mut seed);

    let reference = vulkan
        .fused_post_attention_prefill(
            AttnOutSrc::Host(&attn_out),
            &residual,
            1,
            &wo,
            None,
            &ffn_norm,
            &ffn_gate,
            &ffn_up,
            &ffn_down,
            None,
            eps,
            FfnActivation::Swiglu,
        )
        .expect("the prefill chain handles this shape");

    let got = vulkan.fused_post_attention(FusedPostAttentionInput {
        stop_at_ffn_norm: false,
        activation: FfnActivation::Swiglu,
        attn_out: GpuInput::Cpu(&attn_out),
        residual: GpuInput::Cpu(&residual),
        wo: &wo,
        attn_post_norm: None,
        ffn_norm: &ffn_norm,
        ffn_gate: &ffn_gate,
        ffn_up: &ffn_up,
        ffn_gate_up: None,
        ffn_down: &ffn_down,
        ffn_post_norm: None,
        eps,
        post_norm_eps: None,
        ple: None,
        layer_output_scale: None,
        batch_slot: 0,
    });

    assert_eq!(got.len(), reference.len());
    for (i, (a, b)) in reference.iter().zip(got.iter()).enumerate() {
        assert!(
            (a - b).abs() <= 1e-3 * a.abs().max(1.0),
            "mismatch at {i}: prefill={a} decode={b}"
        );
    }
}

/// **Failing and `#[ignore]`d — this is the specification, not a passing
/// check.** It is the test that should have existed before the decode chain
/// grew its llama-family arms; written after the first caller of those arms
/// produced token soup (LESSONS §64).
///
/// **Localised to the attention half**, by checking each half against its
/// own reference rather than by varying the configuration:
///
/// - `fused_post_attention_decode_matches_prefill_on_the_llama_shape`
///   **passes** — the decode post-attention chain agrees with the prefill
///   one, which has taken optional post-norms and a SwiGLU switch since
///   G2's first increment and is what `llama.rs` runs in production. So the
///   no-post-norm residual arms and the activation switch are *correct*.
/// - `fused_attention_decode_matches_cpu_on_the_llama_shape` **fails**. The
///   defect is in the attention half, which is where the two `RopeOnly`
///   stages live.
///
/// An earlier bisection of this test pointed at the residual arms instead —
/// restoring the post-norms dropped the error from 45.0 to 0.20 — and that
/// was **wrong**. A post-norm *normalizes away* the magnitude of whatever
/// reaches it, so restoring one masks an upstream error rather than
/// implicating its own absence. Varying a configuration only localises a
/// defect when the varied step cannot hide the others.
///
/// Checked and excluded: the `add` shader's contract (`y[i] = a[i] + b[i]`
/// over `elem4`, guarded by `em.len`) matches the bind group's
/// `(a, b, y, meta)` order; the RoPE pairing is irrelevant (the failure is
/// byte-identical under NEOX); and it is not resource-cache reuse (the
/// failure reproduces with this test running alone on one thread).
#[test]
fn fused_layer_llama_shaped_matches_cpu_reference() {
    cross_check_fused_layer_llama_shaped("");
}

/// See [`fused_layer_llama_shaped_matches_cpu_reference`] — same defect,
/// with Qwen2's projection biases on top. Kept separate so that when the
/// residual arm is fixed this says whether the bias arm is also right,
/// rather than the two failing as one.
#[test]
fn fused_layer_llama_shaped_with_qkv_biases_matches_cpu_reference() {
    cross_check_fused_layer_llama_shaped("qkv");
}

#[test]
fn fused_layer_matches_cpu_reference_full_layer_with_ple() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
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
    let mut reference_cache = crate::engine::kv_cache::KvCache::new_with_dims(capacity, &[kv_dim]);
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

        let mut q = CpuBackend.matmul_dequant(&normed, 1, &wq);
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
        let mut k = CpuBackend.matmul_dequant(&normed, 1, &wk);
        crate::engine::tensor::rmsnorm_inplace(&mut k, &k_norm, n_head_kv, head_dim, eps);
        let mut v = CpuBackend.matmul_dequant(&normed, 1, &wv);
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

        let mut attn_proj = CpuBackend.matmul_dequant(&attn_out, 1, &wo);
        crate::engine::tensor::rmsnorm_inplace(&mut attn_proj, &attn_post_norm, 1, n_embd, eps);
        let mut xr = x.clone();
        crate::engine::tensor::add_inplace(&mut xr, &attn_proj);
        let attn_out_residual = xr.clone();

        let mut ffn_normed = xr.clone();
        crate::engine::tensor::rmsnorm_inplace(&mut ffn_normed, &ffn_norm, 1, n_embd, eps);
        let mut gate = CpuBackend.matmul_dequant(&ffn_normed, 1, &ffn_gate);
        let up = CpuBackend.matmul_dequant(&ffn_normed, 1, &ffn_up);
        for g in gate.iter_mut() {
            *g = crate::engine::tensor::gelu(*g);
        }
        crate::engine::tensor::mul_inplace(&mut gate, &up);
        let mut ffn_out = CpuBackend.matmul_dequant(&gate, 1, &ffn_down);
        crate::engine::tensor::rmsnorm_inplace(&mut ffn_out, &ffn_post_norm, 1, n_embd, eps);
        xr = attn_out_residual;
        crate::engine::tensor::add_inplace(&mut xr, &ffn_out);

        let pe_in = xr.clone();
        let mut g = CpuBackend.matmul_dequant(&xr, 1, &ple_gate_w);
        for v in g.iter_mut() {
            *v = crate::engine::tensor::gelu(*v);
        }
        crate::engine::tensor::mul_inplace(&mut g, &per_layer_slice);
        let mut proj = CpuBackend.matmul_dequant(&g, 1, &ple_proj_w);
        crate::engine::tensor::rmsnorm_inplace(&mut proj, &ple_post_norm, 1, n_embd, eps);
        xr = pe_in;
        crate::engine::tensor::add_inplace(&mut xr, &proj);

        for v in xr.iter_mut() {
            *v *= layer_output_scale;
        }
        let expected = xr;

        let got = vulkan.fused_layer(FusedLayerInput {
            stop_at_ffn_norm: false,
            yarn: RopeYarn::IDENTITY,
            normalize_v: true,
            attn_gate: None,
            q_bias: None,
            pairing: crate::engine::tensor::RopeLayout::Neox,
            activation: FfnActivation::Geglu,
            x: GpuInput::Cpu(&x),
            attn_norm: &attn_norm,
            wq: &wq,
            q_norm: Some(&q_norm),
            kv: Some(FusedAttnProjection {
                k_bias: None,
                v_bias: None,
                wk: &wk,
                k_norm: Some(&k_norm),
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
            window: None,
            scale,
            cache: &mut kv_cache.layers[0],
            wo: &wo,
            attn_post_norm: Some(&attn_post_norm),
            ffn_norm: &ffn_norm,
            ffn_gate: &ffn_gate,
            ffn_up: &ffn_up,
            ffn_gate_up: None,
            ffn_down: &ffn_down,
            ffn_post_norm: Some(&ffn_post_norm),
            ple: Some(FusedPle {
                gate_w: &ple_gate_w,
                proj_w: &ple_proj_w,
                post_norm: &ple_post_norm,
                per_layer_slice: GpuInput::Cpu(&per_layer_slice),
                per_layer_dim: per_layer_slice.len(),
            }),
            layer_output_scale: Some(layer_output_scale),
            post_norm_eps: None,
            batch_slot: 0,
            attn_ts: None,
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
/// The full fused decode layer at a model shape with word-reading types on
/// every projection — the path where the attention norm and the FFN's
/// producers write the 8-bit activations the integer-dot kernels read
/// (`decode_mmvq`), and the float word-reading kernels otherwise — against
/// the CPU reference over three steps. A wrong-input projection here is
/// what the greedy check caught after Round 10's first wiring.
#[test]
fn fused_layer_model_shaped_on_the_word_reading_types_matches_cpu_reference() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };

    let n_embd = 1536;
    let n_head = 8;
    let n_head_kv = 1;
    let head_dim = 256;
    let rope_dim = 256;
    let ffn_len = 6144;
    let per_layer_dim = 256;
    let group_size = n_head / n_head_kv;
    let kv_dim = n_head_kv * head_dim;
    let capacity = 128;
    let eps = 1e-6;
    let rope_freq_base = 10000.0;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let layer_output_scale = 1.0 / (2.0f32).sqrt();

    let mut seed = 0x0DD1AE4_u64;
    let build = |ggml_type: u32, in_dim: usize, out_dim: usize, seed: &mut u64| {
        let elems = block_elems(ggml_type);
        let mut bytes = Vec::new();
        for _ in 0..out_dim * (in_dim / elems) {
            bytes.extend(build_block(ggml_type, seed));
        }
        test_quant_matrix(&bytes, ggml_type, in_dim, out_dim)
    };
    let rand_vec = |len: usize, seed: &mut u64| -> Vec<f32> {
        (0..len)
            .map(|_| (next_byte(seed) as f32 - 128.0) / 256.0)
            .collect()
    };
    let near_one = |len: usize, seed: &mut u64| -> Vec<f32> {
        rand_vec(len, seed).iter().map(|v| 1.0 + v * 0.1).collect()
    };

    let attn_norm = near_one(n_embd, &mut seed);
    let wq = build(GGML_TYPE_Q2_K, n_embd, n_head * head_dim, &mut seed);
    let q_norm = near_one(head_dim, &mut seed);
    let wk = build(GGML_TYPE_Q3_K, n_embd, kv_dim, &mut seed);
    let k_norm = near_one(head_dim, &mut seed);
    let wv = build(GGML_TYPE_IQ3_S, n_embd, kv_dim, &mut seed);
    let wo = build(GGML_TYPE_Q4_K, n_head * head_dim, n_embd, &mut seed);
    let attn_post_norm = near_one(n_embd, &mut seed);
    let ffn_norm = near_one(n_embd, &mut seed);
    let ffn_gate = build(GGML_TYPE_IQ2_S, n_embd, ffn_len, &mut seed);
    let ffn_up = build(GGML_TYPE_Q2_K, n_embd, ffn_len, &mut seed);
    let ffn_down = build(GGML_TYPE_Q3_K, ffn_len, n_embd, &mut seed);
    let ffn_post_norm = near_one(n_embd, &mut seed);
    let ple_gate_w = build(GGML_TYPE_F32, n_embd, per_layer_dim, &mut seed);
    let ple_proj_w = build(GGML_TYPE_F32, per_layer_dim, n_embd, &mut seed);
    let ple_post_norm = near_one(n_embd, &mut seed);

    let mut kv_cache = crate::engine::kv_cache::KvCache::new_with_dims(capacity, &[kv_dim]);
    let mut reference_cache = crate::engine::kv_cache::KvCache::new_with_dims(capacity, &[kv_dim]);
    for _ in 0..3 {
        let k: Vec<f32> = rand_vec(kv_dim, &mut seed);
        let v: Vec<f32> = rand_vec(kv_dim, &mut seed);
        kv_cache.layers[0].push(&k, &v);
        reference_cache.layers[0].push(&k, &v);
    }

    for step in 0..3 {
        let pos = kv_cache.layers[0].len;
        let window_start = 0;
        let x = rand_vec(n_embd, &mut seed);
        let per_layer_slice = rand_vec(per_layer_dim, &mut seed);

        // CPU reference, matching `GemmaModel::forward`'s statement
        // order exactly.
        let mut normed = x.clone();
        crate::engine::tensor::rmsnorm_inplace(&mut normed, &attn_norm, 1, n_embd, eps);

        let mut q = CpuBackend.matmul_dequant(&normed, 1, &wq);
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
        let mut k = CpuBackend.matmul_dequant(&normed, 1, &wk);
        crate::engine::tensor::rmsnorm_inplace(&mut k, &k_norm, n_head_kv, head_dim, eps);
        let mut v = CpuBackend.matmul_dequant(&normed, 1, &wv);
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

        let mut attn_proj = CpuBackend.matmul_dequant(&attn_out, 1, &wo);
        crate::engine::tensor::rmsnorm_inplace(&mut attn_proj, &attn_post_norm, 1, n_embd, eps);
        let mut xr = x.clone();
        crate::engine::tensor::add_inplace(&mut xr, &attn_proj);
        let attn_out_residual = xr.clone();

        let mut ffn_normed = xr.clone();
        crate::engine::tensor::rmsnorm_inplace(&mut ffn_normed, &ffn_norm, 1, n_embd, eps);
        let mut gate = CpuBackend.matmul_dequant(&ffn_normed, 1, &ffn_gate);
        let up = CpuBackend.matmul_dequant(&ffn_normed, 1, &ffn_up);
        for g in gate.iter_mut() {
            *g = crate::engine::tensor::gelu(*g);
        }
        crate::engine::tensor::mul_inplace(&mut gate, &up);
        let mut ffn_out = CpuBackend.matmul_dequant(&gate, 1, &ffn_down);
        crate::engine::tensor::rmsnorm_inplace(&mut ffn_out, &ffn_post_norm, 1, n_embd, eps);
        xr = attn_out_residual;
        crate::engine::tensor::add_inplace(&mut xr, &ffn_out);

        let pe_in = xr.clone();
        let mut g = CpuBackend.matmul_dequant(&xr, 1, &ple_gate_w);
        for v in g.iter_mut() {
            *v = crate::engine::tensor::gelu(*v);
        }
        crate::engine::tensor::mul_inplace(&mut g, &per_layer_slice);
        let mut proj = CpuBackend.matmul_dequant(&g, 1, &ple_proj_w);
        crate::engine::tensor::rmsnorm_inplace(&mut proj, &ple_post_norm, 1, n_embd, eps);
        xr = pe_in;
        crate::engine::tensor::add_inplace(&mut xr, &proj);

        for v in xr.iter_mut() {
            *v *= layer_output_scale;
        }
        let expected = xr;

        let got = vulkan.fused_layer(FusedLayerInput {
            stop_at_ffn_norm: false,
            yarn: RopeYarn::IDENTITY,
            normalize_v: true,
            attn_gate: None,
            q_bias: None,
            pairing: crate::engine::tensor::RopeLayout::Neox,
            activation: FfnActivation::Geglu,
            x: GpuInput::Cpu(&x),
            attn_norm: &attn_norm,
            wq: &wq,
            q_norm: Some(&q_norm),
            kv: Some(FusedAttnProjection {
                k_bias: None,
                v_bias: None,
                wk: &wk,
                k_norm: Some(&k_norm),
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
            window: None,
            scale,
            cache: &mut kv_cache.layers[0],
            wo: &wo,
            attn_post_norm: Some(&attn_post_norm),
            ffn_norm: &ffn_norm,
            ffn_gate: &ffn_gate,
            ffn_up: &ffn_up,
            ffn_gate_up: None,
            ffn_down: &ffn_down,
            ffn_post_norm: Some(&ffn_post_norm),
            ple: Some(FusedPle {
                gate_w: &ple_gate_w,
                proj_w: &ple_proj_w,
                post_norm: &ple_post_norm,
                per_layer_slice: GpuInput::Cpu(&per_layer_slice),
                per_layer_dim: per_layer_slice.len(),
            }),
            layer_output_scale: Some(layer_output_scale),
            post_norm_eps: None,
            batch_slot: 0,
            attn_ts: None,
        });

        // The projections' activations are 8-bit on this path, so the
        // bound is a fraction of the output's magnitude, not of each
        // element's.
        assert_eq!(expected.len(), got.len());
        let magnitude = expected.iter().fold(1.0f32, |m, v| m.max(v.abs()));
        for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 5e-2 * magnitude,
                "step {step}: mismatch at index {i}: cpu={a} gpu={b} (magnitude {magnitude})"
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
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
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
    let mut reference_cache = crate::engine::kv_cache::KvCache::new_with_dims(capacity, &[kv_dim]);
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

        let mut q = CpuBackend.matmul_dequant(&normed, 1, &l.wq);
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
            let mut k = CpuBackend.matmul_dequant(&normed, 1, wk);
            crate::engine::tensor::rmsnorm_inplace(&mut k, k_norm, n_head_kv, head_dim, eps);
            let mut v = CpuBackend.matmul_dequant(&normed, 1, wv);
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

        let mut attn_proj = CpuBackend.matmul_dequant(&attn_out, 1, &l.wo);
        crate::engine::tensor::rmsnorm_inplace(&mut attn_proj, &l.attn_post_norm, 1, n_embd, eps);
        let mut xr = x.to_vec();
        crate::engine::tensor::add_inplace(&mut xr, &attn_proj);
        let attn_out_residual = xr.clone();

        let mut ffn_normed = xr.clone();
        crate::engine::tensor::rmsnorm_inplace(&mut ffn_normed, &l.ffn_norm, 1, n_embd, eps);
        let mut gate = CpuBackend.matmul_dequant(&ffn_normed, 1, &l.ffn_gate);
        let up = CpuBackend.matmul_dequant(&ffn_normed, 1, &l.ffn_up);
        for g in gate.iter_mut() {
            *g = crate::engine::tensor::gelu(*g);
        }
        crate::engine::tensor::mul_inplace(&mut gate, &up);
        let mut ffn_out = CpuBackend.matmul_dequant(&gate, 1, &l.ffn_down);
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
            stop_at_ffn_norm: false,
            yarn: RopeYarn::IDENTITY,
            normalize_v: true,
            attn_gate: None,
            q_bias: None,
            pairing: crate::engine::tensor::RopeLayout::Neox,
            activation: FfnActivation::Geglu,
            x: GpuInput::Cpu(&x0),
            attn_norm: &l0.attn_norm,
            wq: &l0.wq,
            q_norm: Some(&l0.q_norm),
            kv: Some(FusedAttnProjection {
                k_bias: None,
                v_bias: None,
                wk: &wk,
                k_norm: Some(&k_norm),
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
            window: None,
            scale,
            cache: &mut kv_cache.layers[0],
            wo: &l0.wo,
            attn_post_norm: Some(&l0.attn_post_norm),
            ffn_norm: &l0.ffn_norm,
            ffn_gate: &l0.ffn_gate,
            ffn_up: &l0.ffn_up,
            ffn_gate_up: None,
            ffn_down: &l0.ffn_down,
            ffn_post_norm: Some(&l0.ffn_post_norm),
            ple: None,
            layer_output_scale: None,
            post_norm_eps: None,
            batch_slot: 0,
            attn_ts: None,
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
            stop_at_ffn_norm: false,
            yarn: RopeYarn::IDENTITY,
            normalize_v: true,
            attn_gate: None,
            q_bias: None,
            pairing: crate::engine::tensor::RopeLayout::Neox,
            activation: FfnActivation::Geglu,
            x: GpuInput::Cpu(&x1),
            attn_norm: &l1.attn_norm,
            wq: &l1.wq,
            q_norm: Some(&l1.q_norm),
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
            window: None,
            scale,
            cache: &mut kv_cache.layers[0],
            wo: &l1.wo,
            attn_post_norm: Some(&l1.attn_post_norm),
            ffn_norm: &l1.ffn_norm,
            ffn_gate: &l1.ffn_gate,
            ffn_up: &l1.ffn_up,
            ffn_gate_up: None,
            ffn_down: &l1.ffn_down,
            ffn_post_norm: Some(&l1.ffn_post_norm),
            ple: None,
            layer_output_scale: None,
            post_norm_eps: None,
            batch_slot: 0,
            attn_ts: None,
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

/// The stripe bound is pure arithmetic over `(limit, row_elems)`, so it is
/// tested as arithmetic rather than by finding a device that overflows.
///
/// Mirrors `VulkanBackend::max_stripe_tokens_for`, which cannot be called
/// without a device.
fn stripe_bound(limit: usize, row_elems: usize) -> usize {
    if row_elems == 0 {
        return usize::MAX;
    }
    (limit.saturating_mul(64) / row_elems).max(1)
}

/// Every `(device limit, n_ff)` pair the bound is asked for must produce a
/// stripe whose flat dispatch fits — including the boundary, where an
/// off-by-one puts the count one workgroup over and panics `wgpu`.
#[test]
fn the_stripe_bound_never_lets_a_flat_dispatch_overflow() {
    // 65535 is what this project's hardware reports; the others cover the
    // adapters that report more or fewer.
    for limit in [1024usize, 65535, 65536, 2_147_483_647] {
        // Real `n_ff` values across the architectures in `engine::arch`,
        // plus the awkward non-power-of-two ones real files actually carry.
        for row_elems in [
            1usize, 63, 64, 65, 2048, 5376, 8192, 11008, 12288, 14336, 16384, 17408, 18432, 32768,
        ] {
            let n = stripe_bound(limit, row_elems);
            assert!(
                n >= 1,
                "limit={limit} n_ff={row_elems}: stripe must allow a token"
            );
            let groups = (n * row_elems).div_ceil(64);
            assert!(
                groups <= limit,
                "limit={limit} n_ff={row_elems}: stripe {n} dispatches {groups} workgroups"
            );
            // And it must be the *largest* such stripe — a bound that is
            // needlessly small costs throughput silently.
            let next = ((n + 1) * row_elems).div_ceil(64);
            assert!(
                next > limit || n >= usize::MAX / 2,
                "limit={limit} n_ff={row_elems}: stripe {n} is smaller than it needs to be"
            );
        }
    }
}

/// The case that actually crashed: 512 tokens at `n_ff = 12288` on a device
/// reporting the usual 65,535 limit asked for 96,192 workgroups.
#[test]
fn the_reported_prefill_overflow_is_bounded_away() {
    let limit = 65535usize;
    let n_ff = 12288usize;
    assert!(
        (512 * n_ff).div_ceil(64) > limit,
        "the reported configuration must still be over the limit"
    );
    let n = stripe_bound(limit, n_ff);
    assert_eq!(n, 341, "the widest safe stripe for this model");
    assert!((n * n_ff).div_ceil(64) <= limit);
}

/// A model whose `n_ff` makes the configured default itself unsafe must be
/// clamped below it rather than trusted — 16384 lands exactly on the boundary.
#[test]
fn a_wide_ffn_clamps_below_the_configured_default() {
    let limit = 65535usize;
    let bound = stripe_bound(limit, 16384);
    assert_eq!(bound, 255, "256 would dispatch 65536, one over");
    assert!(
        bound < crate::engine::backend::MAX_MULTI_TOKEN_PHASE_TOKENS_DEFAULT,
        "the device bound has to win over the configured default"
    );
}

/// A zero row width is not a capacity finding and must not clamp anything to
/// death — it means the caller has nothing to dispatch over.
#[test]
fn a_zero_row_width_does_not_clamp() {
    assert_eq!(stripe_bound(65535, 0), usize::MAX);
}

/// **The differential test the paged fused decode path has to pass.**
///
/// `fused_attention` is the decode step: it computes this token's key and
/// value on the device, writes them into the cache, and reads the whole
/// window back — all in one submission. Paging it moves both halves at once,
/// which is what makes it worth a test of its own rather than trusting the
/// prefill one. The write now lands in a pool page instead of a per-request
/// mirror, and if the destination row and the row the kernel reads back
/// disagree, attention answers from whatever that page held before.
///
/// The low pages are held and filled with junk for the reason
/// `paged_prefill_matches_contiguous`'s own comment gives: the pool hands out
/// never-used pages from the bottom, so a fresh sequence lands on the identity
/// mapping and a kernel ignoring the block table reads the right rows by
/// accident.
///
/// Nine positions over four-token pages, so the run crosses two page
/// boundaries — the case where the fused path has to take a page the host
/// side never asked for, because there is no tail to upload at a boundary.
#[test]
fn paged_fused_decode_matches_cpu_reference() {
    use crate::engine::kv_pool::{KvPool, LayerGeometry, Policy};
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };

    let n_embd = 32;
    let n_head = 4;
    let n_head_kv = 2;
    let head_dim = 8;
    let rope_dim = 8;
    let group_size = n_head / n_head_kv;
    let kv_dim = n_head_kv * head_dim;
    let page_tokens = 4;
    // Half of these are held back below (see the junk fill), so this is twice
    // what the sequence needs plus room to be promised it.
    let pool_pages = 24;
    let capacity = 32;
    let eps = 1e-6;
    let rope_freq_base = 10000.0;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut seed = 0x5E9D_1C0B_u64;
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

    let mut pool = KvPool::with_policy(
        pool_pages,
        page_tokens,
        vec![LayerGeometry {
            kv_dim,
            stride: 1,
            ring: None,
        }],
        Policy::Lru,
    );
    let (device, queue) = vulkan.device_and_queue();
    assert!(pool.attach_device(device, vulkan.kv_storage(), pool_pages * 4));
    let pool = std::sync::Arc::new(pool);
    // Alternate pages, so the sequence's run is neither low nor adjacent —
    // see the same fixture in `cross_check_fused_attention_prefill_paged` for
    // why adjacency alone lets a broken address computation pass.
    let all = pool.alloc(pool_pages).expect("pool has room");
    let (given, held): (Vec<u32>, Vec<u32>) = all.iter().partition(|p| !(*p).is_multiple_of(2));
    pool.release(&given);
    for &physical in &held {
        let junk: Vec<f32> = (0..page_tokens * kv_dim)
            .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 32.0)
            .collect();
        pool.fill_device(queue, 0, physical, &junk, &junk);
    }

    let mut kv_cache =
        crate::engine::kv_cache::KvCache::new_with_strided_dims(capacity, &strided_dims(&pool))
            .try_into_paged(pool.clone())
            .unwrap_or_else(|_| panic!("test pool has room"));
    let mut reference_cache = crate::engine::kv_cache::KvCache::new_with_dims(capacity, &[kv_dim]);
    for _ in 0..3 {
        let k: Vec<f32> = rand_vec(kv_dim, &mut seed);
        let v: Vec<f32> = rand_vec(kv_dim, &mut seed);
        kv_cache.layers[0].push(&k, &v);
        reference_cache.layers[0].push(&k, &v);
    }
    kv_cache.commit_pages();

    for step in 0..6 {
        let pos = kv_cache.layers[0].len;
        let window_start = 0;
        let normed = rand_vec(n_embd, &mut seed);

        let mut q = CpuBackend.matmul_dequant(&normed, 1, &wq);
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
        let mut k = CpuBackend.matmul_dequant(&normed, 1, &wk);
        crate::engine::tensor::rmsnorm_inplace(&mut k, &k_norm, n_head_kv, head_dim, eps);
        let mut v = CpuBackend.matmul_dequant(&normed, 1, &wv);
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
            projections_ready: false,
            yarn: RopeYarn::IDENTITY,
            normalize_v: true,
            attn_gate: None,
            q_bias: None,
            pairing: crate::engine::tensor::RopeLayout::Neox,
            normed: GpuInput::Cpu(&normed),
            normed_q8: None,
            wq: &wq,
            q_norm: Some(&q_norm),
            kv: Some(FusedAttnProjection {
                k_bias: None,
                v_bias: None,
                wk: &wk,
                k_norm: Some(&k_norm),
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
            window: None,
            scale,
            cache: &mut kv_cache.layers[0],
            batch_slot: 0,
            attn_ts: None,
        });

        // Without this the test compares the mirrored fallback with itself:
        // the mirror is built from the same pages, so it is right either way.
        assert!(
            kv_cache.layers[0].is_pool_backed(),
            "step {step}: the paged fused path was not taken"
        );
        assert_eq!(expected.len(), got.len());
        for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
            let tol = 6e-2 * a.abs().max(1.0);
            assert!(
                (a - b).abs() <= tol,
                "step {step}: mismatch at index {i}: cpu={a} gpu={b}"
            );
        }
        assert_eq!(kv_cache.layers[0].len, pos + 1);
    }
    pool.release(&held);
}

/// **The differential test the paged fused prefill has to pass.**
///
/// The prefill writes a whole range of positions at once, so unlike the decode
/// step its write is not one row but a run per page it crosses. `start_pos` of
/// 5 against 8-token pages puts the range's start part way into a page and its
/// end part way into another, which is the case where every run has a
/// different source offset, destination page and length.
#[test]
fn paged_fused_prefill_matches_unfused_reference() {
    // One token: the degenerate single-run case, mid-page.
    cross_check_fused_attention_prefill_paged(1, true, 5, true);
    // Crossing one boundary.
    cross_check_fused_attention_prefill_paged(6, true, 5, true);
    // Spanning several whole pages plus partial ends, both V arrangements.
    cross_check_fused_attention_prefill_paged(21, true, 5, true);
    cross_check_fused_attention_prefill_paged(21, false, 5, true);
    // Starting exactly on a page boundary — no leading partial run.
    cross_check_fused_attention_prefill_paged(17, true, 8, true);
}

/// One deep token on the page pool takes the split kernel through the
/// table (`fused_attention_prefill_one_deep_token_takes_the_split_kernel`
/// is the contiguous form): past the split's threshold, a window that is
/// not a whole number of pages or of split chunks, both V arrangements.
#[test]
fn paged_fused_prefill_one_deep_token_takes_the_split_kernel() {
    cross_check_fused_attention_prefill_paged(1, true, 700, true);
    cross_check_fused_attention_prefill_paged(1, false, 333, true);
}

/// **Evidence that the matmul kernels are not where this device goes wrong.**
///
/// `VulkanBackend::decode_kernel_agrees` is the startup probe that decides
/// whether the tuned decode kernels can be trusted. On this machine it says
/// yes while the served model produces word salad, so the obvious reading is
/// that the probe is too narrow. This sweeps every dimension it could be too
/// narrow in — type, `in_dim`, `out_dim`, `n_tokens`, `matmul` against
/// `matmul_batch` — against `matmul_dequant`, the full-precision reference
/// (`matmul` rounds activations to `int8`, so it cannot referee this).
///
/// Everything agrees, to 1.2e-3 at worst. That is the finding: the fault is
/// not reachable through `Backend::matmul`/`matmul_batch` at any shape, so no
/// amount of widening the probe would have caught it, and a kernel-level
/// check is the wrong shape of check. See `disable_miscompiled_decode_kernels`.
#[test]
#[ignore = "diagnostic; prints a table"]
fn _scratch_decode_probe_shape_sweep() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    // Which kernel this backend actually selects, so a sweep that quietly
    // ran on the reference path cannot be mistaken for evidence.
    for (label, ty) in [
        ("q4_k", GGML_TYPE_Q4_K),
        ("q5_k", GGML_TYPE_Q5_K),
        ("q6_k", GGML_TYPE_Q6_K),
    ] {
        eprintln!(
            "  selected for {label}: {} (wide_unroll={})",
            vulkan.selected_kernel_name(ty, 2048, 1),
            vulkan.wide_unroll_on(),
        );
    }
    eprintln!("  type    in_dim out_dim   fixture     max rel err");
    for (label, ty) in [
        ("q4_k", GGML_TYPE_Q4_K),
        ("q5_k", GGML_TYPE_Q5_K),
        ("q6_k", GGML_TYPE_Q6_K),
    ] {
        let elems = block_elems(ty);
        for &(in_dim, out_dim) in &[(512usize, 512usize), (2048, 2048)] {
            for spacers in [0usize, 1, 2, 3] {
                // Push the probe weight off offset 0 in its chunk buffer. A single
                // uploaded tensor lands at the start of a fresh chunk; in a served
                // model every weight but the first sits at some arbitrary offset,
                // and the wide-load kernels bind `array<vec4<u32>>` views of it.
                for spacer in 0..spacers {
                    let n_blocks = 512 / elems * (3 + spacer);
                    let mut seed = 0xD00D_u64 + spacer as u64;
                    let bytes: Vec<u8> = (0..n_blocks)
                        .flat_map(|_| build_block(ty, &mut seed))
                        .collect();
                    let sw = crate::engine::loader::probe_quant_matrix(bytes, ty, 512, 3 + spacer);
                    let sx: Vec<f32> = (0..512).map(|i| (i % 7) as f32 * 0.01).collect();
                    let _ = vulkan.matmul(&sx, 1, &sw);
                }
                for n_tokens in [1usize, 8] {
                    let n_blocks = in_dim / elems * out_dim;
                    let mut seed = 0xC0FFEE_u64;
                    let bytes: Vec<u8> = (0..n_blocks)
                        .flat_map(|_| build_block(ty, &mut seed))
                        .collect();
                    let w = crate::engine::loader::probe_quant_matrix(bytes, ty, in_dim, out_dim);
                    let x: Vec<f32> = (0..n_tokens * in_dim)
                        .map(|i| ((i % 13) as f32 - 6.0) * 0.05)
                        .collect();
                    let want = CpuBackend.matmul_dequant(&x, n_tokens, &w);
                    // Two ops in one batch — `matmul_batch`, not `matmul`. The
                    // gate/up pair of every FFN is issued exactly this way.
                    let ops = [
                        crate::engine::backend::MatmulOp {
                            x: &x,
                            n_tokens,
                            w: &w,
                        },
                        crate::engine::backend::MatmulOp {
                            x: &x,
                            n_tokens,
                            w: &w,
                        },
                    ];
                    let got = vulkan.matmul_batch(&ops).swap_remove(0);
                    let err = want
                        .iter()
                        .zip(&got)
                        .map(|(a, b)| (a - b).abs() / a.abs().max(1.0))
                        .fold(0.0f32, f32::max);
                    eprintln!(
                        "  {label:6}  {in_dim:6} {out_dim:7}  tok {n_tokens:3}  spacers {spacers}   {err:.6}"
                    );
                }
            }
        }
    }
}

/// The wide whole-row norms (`vec4`, straight-line loads) must produce what
/// the scalar grid-stride kernels produce — at a model width, at a width
/// that overflows the straight-line slots into the tail loop, and at one
/// too narrow to fill the workgroup — for all three members of the family.
///
/// Compared kernel against kernel rather than against a CPU reference:
/// the scalar kernel is already held to the reference by the fused-chain
/// tests, and what this rewrite changed is the load structure, not the
/// arithmetic, so the two should agree to rounding.
#[test]
fn wide_norms_match_the_scalar_norms() {
    let _gpu_lock = gpu_test_lock();
    let Some(gpu) = shared_test_backend() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    let tail_width = vulkan_shaders::NORM_WIDE_WG * vulkan_shaders::NORM_WIDE_SLOTS * 4 + 512;
    for n_embd in [1536usize, 256, 960, tail_width] {
        let x: Vec<f32> = (0..n_embd)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.03)
            .collect();
        let w: Vec<f32> = (0..n_embd).map(|i| 0.5 + (i % 7) as f32 * 0.1).collect();
        let r: Vec<f32> = (0..n_embd).map(|i| (i % 13) as f32 * 0.02 - 0.1).collect();
        let eps = 1e-6;
        let out_scale = 0.75;
        for kind in ["norm", "norm_add", "norm_add_scale"] {
            let run = |wide: bool| -> Vec<f32> {
                let xb = gpu.upload_new(&x);
                let wb = gpu.upload_new(&w);
                let rb = gpu.upload_new(&r);
                let yb = gpu.upload_new(&vec![0.0f32; n_embd]);
                let bytes = (n_embd as u64) * 4;
                let scalar_index = norm_wg_index(n_embd);
                let mut encoder = gpu.new_encoder("wide norm parity");
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                    match kind {
                        "norm" => {
                            let meta = gpu.elem_meta_buffer(n_embd as u32, eps);
                            let bg = gpu.elem4_bind_group(
                                BindSrc::Slice(&xb, 0, bytes),
                                &wb,
                                BindSrc::Slice(&yb, 0, bytes),
                                &meta,
                            );
                            pass.set_pipeline(if wide {
                                &gpu.rmsnorm_wide_pipeline
                            } else {
                                &gpu.rmsnorm_pipeline[scalar_index]
                            });
                            pass.set_bind_group(0, &bg, &[]);
                            pass.dispatch_workgroups(1, 1, 1);
                        }
                        "norm_add" => {
                            let meta = gpu.elem_meta_buffer(n_embd as u32, eps);
                            let bg = gpu.elem5_bind_group(
                                BindSrc::Slice(&xb, 0, bytes),
                                &wb,
                                BindSrc::Slice(&rb, 0, bytes),
                                BindSrc::Slice(&yb, 0, bytes),
                                &meta,
                            );
                            pass.set_pipeline(if wide {
                                &gpu.rmsnorm_add_wide_pipeline
                            } else {
                                &gpu.rmsnorm_add_pipeline[scalar_index]
                            });
                            pass.set_bind_group(0, &bg, &[]);
                            pass.dispatch_workgroups(1, 1, 1);
                        }
                        _ => {
                            let meta = gpu.elem_meta_buffer_scaled(n_embd as u32, eps, out_scale);
                            let bg = gpu.elem5_bind_group(
                                BindSrc::Slice(&xb, 0, bytes),
                                &wb,
                                BindSrc::Slice(&rb, 0, bytes),
                                BindSrc::Slice(&yb, 0, bytes),
                                &meta,
                            );
                            pass.set_pipeline(if wide {
                                &gpu.rmsnorm_add_scale_wide_pipeline
                            } else {
                                &gpu.rmsnorm_add_scale_pipeline[scalar_index]
                            });
                            pass.set_bind_group(0, &bg, &[]);
                            pass.dispatch_workgroups(1, 1, 1);
                        }
                    }
                }
                gpu.submit_and_readback_for_test(encoder, &yb, n_embd)
            };
            let scalar = run(false);
            let wide = run(true);
            // A CPU rendering of the same formula, so a shared mistake in the
            // two kernels cannot pass.
            let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / n_embd as f32;
            let scale = 1.0 / (mean_sq + eps).sqrt();
            for i in 0..n_embd {
                let want = match kind {
                    "norm" => x[i] * scale * w[i],
                    "norm_add" => x[i] * scale * w[i] + r[i],
                    _ => (x[i] * scale * w[i] + r[i]) * out_scale,
                };
                assert!(
                    (wide[i] - scalar[i]).abs() <= 1e-5 * (1.0 + scalar[i].abs()),
                    "{kind} width {n_embd} at {i}: wide {} vs scalar {}",
                    wide[i],
                    scalar[i]
                );
                assert!(
                    (wide[i] - want).abs() <= 1e-4 * (1.0 + want.abs()),
                    "{kind} width {n_embd} at {i}: wide {} vs cpu {want}",
                    wide[i]
                );
            }
        }
    }
}

/// The integer-dot prefill GEMM against the CPU's dequantized product, at
/// the served model's FFN shape (`1536 → 6144`, cut to 512 rows) and its
/// down-projection width (`6144 → 512`), one and two token tiles wide. Not
/// bit-exact — the activation is quantized to 8 bits per 32-block — so the
/// bound is on the error relative to the row's magnitude, the same kind the
/// decode MMVQ kernel meets; a wrong scale, sub-block or nibble half is off
/// by whole multiples, not by a rounding.
#[test]
fn mmq_q4k_gemm_matches_the_cpu_product() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    let mut seed = 0x51EE_D4A7_u64;
    for (ggml_type, in_dim, out_dim, n_tokens) in [
        (GGML_TYPE_Q4_K, 1536usize, 512usize, 64usize),
        (GGML_TYPE_Q4_K, 6144, 512, 128),
        (GGML_TYPE_Q4_K, 1536, 256, 90),
        // Wide enough for the 128 × 128 kernel (`mmq_kernel_for`): the
        // FFN gate shape at two token tiles, and a down projection whose
        // last token tile is partial.
        (GGML_TYPE_Q4_K, 1536, 6144, 256),
        (GGML_TYPE_Q4_K, 6144, 1536, 200),
        // `Q6_K`: 210-byte blocks, so the odd blocks of a row start two
        // bytes into a word, and a row of `1536` is not 16-byte aligned.
        (crate::engine::quant::GGML_TYPE_Q6_K, 6144, 512, 128),
        (crate::engine::quant::GGML_TYPE_Q6_K, 1536, 256, 90),
        // And the wide `Q6_K` tile, on the down-projection shape with a
        // partial last token tile and on an unaligned row width.
        (crate::engine::quant::GGML_TYPE_Q6_K, 6144, 1536, 200),
        (crate::engine::quant::GGML_TYPE_Q6_K, 1536, 6144, 256),
        // The byte-unpacked kernels, one shape each that takes the tall
        // tile and one the half tile with a partial token tile; the
        // 18/22/34-byte blocks exercise both word alignments.
        (crate::engine::quant::GGML_TYPE_Q5_K, 1536, 6144, 256),
        (crate::engine::quant::GGML_TYPE_Q5_K, 6144, 1536, 200),
        (crate::engine::quant::GGML_TYPE_Q8_0, 1536, 6144, 256),
        (crate::engine::quant::GGML_TYPE_Q8_0, 6144, 1536, 200),
        (crate::engine::quant::GGML_TYPE_Q4_0, 1536, 6144, 200),
        (crate::engine::quant::GGML_TYPE_Q4_1, 1536, 6144, 200),
        (crate::engine::quant::GGML_TYPE_Q5_0, 1536, 6144, 200),
        (crate::engine::quant::GGML_TYPE_Q5_1, 1536, 6144, 200),
        // A single-head K/V width, which no wide or half tile serves: the
        // byte-unpacked types take the narrow 64 × 32 shape there, as
        // `Q4_K` does, with a partial token tile and both block parities.
        (crate::engine::quant::GGML_TYPE_Q4_1, 1536, 256, 90),
        (crate::engine::quant::GGML_TYPE_Q4_0, 1536, 256, 90),
        // Rows that are not whole super-blocks (960 = 3.75 × 256), which
        // the byte-unpacked kernels take in sub-block pairs: the last
        // super-block of the quantized activation is partial, and a
        // 1020-byte `Q8_0` row puts odd rows two bytes into a word.
        (crate::engine::quant::GGML_TYPE_Q8_0, 960, 2560, 200),
        (crate::engine::quant::GGML_TYPE_Q8_0, 960, 320, 90),
        (crate::engine::quant::GGML_TYPE_Q4_0, 960, 960, 128),
        (crate::engine::quant::GGML_TYPE_Q5_1, 960, 2560, 200),
        (crate::engine::quant::GGML_TYPE_Q8_0, 1536, 256, 40),
        (crate::engine::quant::GGML_TYPE_Q5_K, 1536, 256, 90),
        (crate::engine::quant::GGML_TYPE_Q3_K, 1536, 256, 90),
        (crate::engine::quant::GGML_TYPE_IQ2_S, 1536, 256, 90),
        // The sixteen-value scale types take the dot per half; `Q3_K`'s
        // 110-byte block puts odd blocks two bytes into a word.
        (crate::engine::quant::GGML_TYPE_Q2_K, 1536, 6144, 256),
        (crate::engine::quant::GGML_TYPE_Q2_K, 6144, 1536, 200),
        (crate::engine::quant::GGML_TYPE_Q3_K, 1536, 6144, 256),
        (crate::engine::quant::GGML_TYPE_Q3_K, 6144, 1536, 200),
        (crate::engine::quant::GGML_TYPE_IQ4_XS, 1536, 6144, 256),
        (crate::engine::quant::GGML_TYPE_IQ4_XS, 6144, 1536, 200),
        // The lattice types: codebook lookups with per-value signs, `IQ2_S`
        // with its scales per half; both have blocks of two-byte parity.
        (crate::engine::quant::GGML_TYPE_IQ3_S, 1536, 6144, 256),
        (crate::engine::quant::GGML_TYPE_IQ3_S, 6144, 1536, 200),
        (crate::engine::quant::GGML_TYPE_IQ2_S, 1536, 6144, 256),
        (crate::engine::quant::GGML_TYPE_IQ2_S, 6144, 1536, 200),
        // A 48-layer model's own shapes: an odd super-block count on the
        // row (`3840 = 15 × 256`), the global layers' Q and single-head
        // K/V widths, and the down projection's `15360`-wide row.
        (GGML_TYPE_Q4_K, 3840, 8192, 128),
        (GGML_TYPE_Q4_K, 3840, 512, 128),
        (GGML_TYPE_Q4_K, 3840, 15360, 128),
        (crate::engine::quant::GGML_TYPE_Q6_K, 15360, 3840, 128),
        // Rows the device pads off the memory interleave
        // (`VulkanBackend::device_row_stride`): a stride that is a multiple
        // of 512 bytes gets one 256-byte gap per row, so every kernel that
        // walks these rows must take its stride from the meta rather than
        // from the type and the width. `Q4_1` at 12288 (7680 = 512 · 15) is
        // the shape that found this, `Q5_1` at 12288 (9216) and `Q8_0` at
        // 8192 (8704) are its twins, and `Q4_K` at 8192 (4608) is a
        // super-block type at the same fault. Each is also a row the CPU
        // reference still reads unpadded, which is the half of the check
        // that bites.
        (crate::engine::quant::GGML_TYPE_Q4_1, 12288, 1536, 200),
        (crate::engine::quant::GGML_TYPE_Q4_1, 12288, 256, 90),
        (crate::engine::quant::GGML_TYPE_Q5_1, 12288, 1536, 128),
        (crate::engine::quant::GGML_TYPE_Q8_0, 8192, 1536, 200),
        (GGML_TYPE_Q4_K, 8192, 1536, 200),
    ] {
        let mut bytes = Vec::new();
        let (_, block_elems) =
            crate::engine::quant::block_layout(ggml_type).expect("a quantized type");
        for _ in 0..out_dim * (in_dim / block_elems) {
            bytes.extend(build_block(ggml_type, &mut seed));
        }
        let w = test_quant_matrix(&bytes, ggml_type, in_dim, out_dim);
        let x: Vec<f32> = (0..n_tokens * in_dim)
            .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
            .collect();
        let Some(got) = vulkan.mmq_matmul_for_test(&x, n_tokens, &w) else {
            eprintln!("mmq kernel not built (ORANGU_PREFILL_MMQ=0?) — nothing to check");
            return;
        };
        let want = CpuBackend.matmul_dequant(&x, n_tokens, &w);
        assert_eq!(got.len(), want.len());
        let mut worst = 0.0f32;
        for t in 0..n_tokens {
            let row = &want[t * out_dim..(t + 1) * out_dim];
            let mag = row.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-3);
            for o in 0..out_dim {
                let i = t * out_dim + o;
                let err = (got[i] - want[i]).abs() / mag;
                worst = worst.max(err);
                assert!(
                    err <= 2e-2,
                    "[type {ggml_type}: {in_dim}x{out_dim}] token {t} row {o}: mmq {} vs cpu {} (row magnitude {mag})",
                    got[i],
                    want[i]
                );
            }
        }
        eprintln!(
            "[type {ggml_type}: {in_dim}x{out_dim}] {n_tokens} tokens: worst relative error {worst:.2e}"
        );
    }
}

/// The generic batch path — `Backend::matmul_batch`, what every
/// CPU-orchestrated architecture's prefill GEMM goes through — takes the
/// integer-dot kernel for the ops it accepts, and the result is **bit for
/// bit** what the kernel produces on its own (same quantize, same kernel,
/// same data): a wiring check, not a tolerance one. Three ops sharing one
/// input (a layer's Q/K/V shape) share one quantized copy; a fourth op at
/// a width the kernel does not tile stays on the float kernel in the same
/// batch, so the two can mix. The CPU product is the sanity bound at the
/// kernel test's own tolerance.
#[test]
fn the_generic_batch_takes_the_integer_dot_kernel() {
    let _gpu_lock = gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    let mut seed = 0x5EED_5EED_u64;
    let (in_dim, out_dim, n_tokens) = (1536usize, 256usize, 90usize);
    let mut weights = Vec::new();
    for ggml_type in [
        GGML_TYPE_Q4_K,
        GGML_TYPE_Q4_K,
        crate::engine::quant::GGML_TYPE_Q6_K,
    ] {
        let mut bytes = Vec::new();
        for _ in 0..out_dim * (in_dim / 256) {
            bytes.extend(build_block(ggml_type, &mut seed));
        }
        weights.push(test_quant_matrix(&bytes, ggml_type, in_dim, out_dim));
    }
    let x: Vec<f32> = (0..n_tokens * in_dim)
        .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 64.0)
        .collect();
    let ops: Vec<MatmulOp<'_>> = weights
        .iter()
        .map(|w| MatmulOp { x: &x, n_tokens, w })
        .collect();
    let got = vulkan.matmul_batch(&ops);
    for (w, got) in weights.iter().zip(&got) {
        let Some(alone) = vulkan.mmq_matmul_for_test(&x, n_tokens, w) else {
            eprintln!("mmq kernel not built (ORANGU_PREFILL_MMQ=0?) — nothing to check");
            return;
        };
        assert_eq!(got.len(), alone.len());
        assert!(
            got == &alone,
            "type {}: the batch path's result is not the kernel's own",
            w.ggml_type()
        );
        let want = CpuBackend.matmul_dequant(&x, n_tokens, w);
        for t in 0..n_tokens {
            let row = &want[t * out_dim..(t + 1) * out_dim];
            let mag = row.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-3);
            for o in 0..out_dim {
                let i = t * out_dim + o;
                assert!(
                    (got[i] - want[i]).abs() / mag <= 2e-2,
                    "type {}: token {t} row {o}: batch {} vs cpu {}",
                    w.ggml_type(),
                    got[i],
                    want[i]
                );
            }
        }
    }

    // A width the kernel does not tile (out_dim 80) beside one it does,
    // in one batch: the float kernel's exact result for the one, the
    // integer-dot kernel's for the other.
    let mut bytes = Vec::new();
    for _ in 0..80 * (in_dim / 256) {
        bytes.extend(build_block(GGML_TYPE_Q4_K, &mut seed));
    }
    let narrow = test_quant_matrix(&bytes, GGML_TYPE_Q4_K, in_dim, 80);
    let mixed = vulkan.matmul_batch(&[
        MatmulOp {
            x: &x,
            n_tokens,
            w: &weights[0],
        },
        MatmulOp {
            x: &x,
            n_tokens,
            w: &narrow,
        },
    ]);
    assert!(
        mixed[0] == got[0],
        "the mixed batch changed the integer-dot op's result"
    );
    let narrow_float = vulkan.without_prefill_mmq(|| vulkan.matmul(&x, n_tokens, &narrow));
    assert!(
        mixed[1] == narrow_float,
        "the op the kernel does not take must stay on the float kernel, exactly"
    );
}

/// The row-strided wide norms must match the rolled rows kernels row for
/// row: three rows of a width inside the straight-line slots, with the
/// weight shared across rows and each row's own residual.
#[test]
fn wide_row_norms_match_the_rolled_row_norms() {
    let _gpu_lock = gpu_test_lock();
    let Some(gpu) = shared_test_backend() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    let rows = 3usize;
    for n_embd in [1536usize, 256] {
        let x: Vec<f32> = (0..rows * n_embd)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.03)
            .collect();
        let w: Vec<f32> = (0..n_embd).map(|i| 0.5 + (i % 7) as f32 * 0.1).collect();
        let r: Vec<f32> = (0..rows * n_embd)
            .map(|i| (i % 13) as f32 * 0.02 - 0.1)
            .collect();
        let eps = 1e-6;
        let out_scale = 0.75;
        let bytes = (rows * n_embd) as u64 * 4;
        for kind in ["norm", "norm_add", "norm_add_scale"] {
            let run = |wide: bool| -> Vec<f32> {
                let xb = gpu.upload_new(&x);
                let wb = gpu.upload_new(&w);
                let rb = gpu.upload_new(&r);
                let yb = gpu.upload_new(&vec![0.0f32; rows * n_embd]);
                let mut encoder = gpu.new_encoder("wide row norm parity");
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                    match kind {
                        "norm" => {
                            let meta = gpu.elem_meta_buffer(n_embd as u32, eps);
                            let bg = gpu.elem4_bind_group(
                                BindSrc::Slice(&xb, 0, bytes),
                                &wb,
                                BindSrc::Slice(&yb, 0, bytes),
                                &meta,
                            );
                            pass.set_pipeline(if wide {
                                &gpu.rmsnorm_rows_wide_pipeline
                            } else {
                                &gpu.rmsnorm_rows_pipeline
                            });
                            pass.set_bind_group(0, &bg, &[]);
                        }
                        "norm_add" => {
                            let meta = gpu.elem_meta_buffer(n_embd as u32, eps);
                            let bg = gpu.elem5_bind_group(
                                BindSrc::Slice(&xb, 0, bytes),
                                &wb,
                                BindSrc::Slice(&rb, 0, bytes),
                                BindSrc::Slice(&yb, 0, bytes),
                                &meta,
                            );
                            pass.set_pipeline(if wide {
                                &gpu.rmsnorm_add_rows_wide_pipeline
                            } else {
                                &gpu.rmsnorm_add_rows_pipeline
                            });
                            pass.set_bind_group(0, &bg, &[]);
                        }
                        _ => {
                            let meta = gpu.elem_meta_buffer_scaled(n_embd as u32, eps, out_scale);
                            let bg = gpu.elem5_bind_group(
                                BindSrc::Slice(&xb, 0, bytes),
                                &wb,
                                BindSrc::Slice(&rb, 0, bytes),
                                BindSrc::Slice(&yb, 0, bytes),
                                &meta,
                            );
                            pass.set_pipeline(if wide {
                                &gpu.rmsnorm_add_scale_rows_wide_pipeline
                            } else {
                                &gpu.rmsnorm_add_scale_rows_pipeline
                            });
                            pass.set_bind_group(0, &bg, &[]);
                        }
                    }
                    pass.dispatch_workgroups(rows as u32, 1, 1);
                }
                gpu.submit_and_readback_for_test(encoder, &yb, rows * n_embd)
            };
            let rolled = run(false);
            let wide = run(true);
            for row in 0..rows {
                let xr = &x[row * n_embd..(row + 1) * n_embd];
                let mean_sq = xr.iter().map(|v| v * v).sum::<f32>() / n_embd as f32;
                let scale = 1.0 / (mean_sq + eps).sqrt();
                for i in 0..n_embd {
                    let j = row * n_embd + i;
                    let want = match kind {
                        "norm" => xr[i] * scale * w[i],
                        "norm_add" => xr[i] * scale * w[i] + r[j],
                        _ => (xr[i] * scale * w[i] + r[j]) * out_scale,
                    };
                    assert!(
                        (wide[j] - rolled[j]).abs() <= 1e-5 * (1.0 + rolled[j].abs()),
                        "{kind} width {n_embd} row {row} at {i}: wide {} vs rolled {}",
                        wide[j],
                        rolled[j]
                    );
                    assert!(
                        (wide[j] - want).abs() <= 1e-4 * (1.0 + want.abs()),
                        "{kind} width {n_embd} row {row} at {i}: wide {} vs cpu {want}",
                        wide[j]
                    );
                }
            }
        }
    }
}

/// The norm pair — post-norm+residual, then the FFN norm over that result —
/// must leave both outputs as the two scalar dispatches would: `y1` is the
/// FFN's residual and `y2` its input, so both are read downstream. Widths
/// inside the straight-line slots and one past them (the rolled tail, which
/// re-reads its own `y1`).
#[test]
fn the_norm_pair_matches_the_two_scalar_norms() {
    let _gpu_lock = gpu_test_lock();
    let Some(gpu) = shared_test_backend() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    let tail_width = vulkan_shaders::NORM_WIDE_WG * vulkan_shaders::NORM_WIDE_SLOTS * 4 + 512;
    for n_embd in [1536usize, 256, tail_width] {
        let x: Vec<f32> = (0..n_embd)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.03)
            .collect();
        let w1: Vec<f32> = (0..n_embd).map(|i| 0.5 + (i % 7) as f32 * 0.1).collect();
        let w2: Vec<f32> = (0..n_embd).map(|i| 1.5 - (i % 5) as f32 * 0.2).collect();
        let r: Vec<f32> = (0..n_embd).map(|i| (i % 13) as f32 * 0.02 - 0.1).collect();
        let eps = 1e-6;
        let bytes = (n_embd as u64) * 4;
        let scalar_index = norm_wg_index(n_embd);
        let xb = gpu.upload_new(&x);
        let w1b = gpu.upload_new(&w1);
        let w2b = gpu.upload_new(&w2);
        let rb = gpu.upload_new(&r);
        let meta = gpu.elem_meta_buffer(n_embd as u32, eps);

        // The two scalar dispatches.
        let y1s = gpu.upload_new(&vec![0.0f32; n_embd]);
        let y2s = gpu.upload_new(&vec![0.0f32; n_embd]);
        let mut encoder = gpu.new_encoder("norm pair parity: scalar");
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            let bg = gpu.elem5_bind_group(
                BindSrc::Slice(&xb, 0, bytes),
                &w1b,
                BindSrc::Slice(&rb, 0, bytes),
                BindSrc::Slice(&y1s, 0, bytes),
                &meta,
            );
            pass.set_pipeline(&gpu.rmsnorm_add_pipeline[scalar_index]);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
            let bg = gpu.elem4_bind_group(
                BindSrc::Slice(&y1s, 0, bytes),
                &w2b,
                BindSrc::Slice(&y2s, 0, bytes),
                &meta,
            );
            pass.set_pipeline(&gpu.rmsnorm_pipeline[scalar_index]);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        let y1_scalar = gpu.submit_and_readback_for_test(encoder, &y1s, n_embd);
        let y2_scalar = gpu.readback_for_test(&y2s, n_embd);

        // The pair.
        let y1p = gpu.upload_new(&vec![0.0f32; n_embd]);
        let y2p = gpu.upload_new(&vec![0.0f32; n_embd]);
        // With the q8 epilogue asked for (`aux = 1`), on the rows the
        // slots cover whole; its output is checked below against the CPU
        // quantizer of the pair's own `y2`.
        let q8_words = n_embd / 32 * 10;
        let q8b = gpu.upload_new(&vec![0.0f32; q8_words]);
        let quantize_here =
            n_embd / 4 <= vulkan_shaders::NORM_WIDE_SLOTS * vulkan_shaders::NORM_WIDE_WG;
        let meta_q8 = gpu.elem_meta_buffer_aux_extra(n_embd as u32, u32::from(quantize_here), eps);
        let mut encoder = gpu.new_encoder("norm pair parity: pair");
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            let bg = gpu.norm_pair_bind_group(
                BindSrc::Slice(&xb, 0, bytes),
                &w1b,
                BindSrc::Slice(&rb, 0, bytes),
                &w2b,
                BindSrc::Slice(&y1p, 0, bytes),
                BindSrc::Slice(&y2p, 0, bytes),
                &meta_q8,
                &q8b,
            );
            pass.set_pipeline(&gpu.rmsnorm_add_norm_wide_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        let y1_pair = gpu.submit_and_readback_for_test(encoder, &y1p, n_embd);
        let y2_pair = gpu.readback_for_test(&y2p, n_embd);
        if quantize_here {
            let got: Vec<u32> = gpu
                .readback_for_test(&q8b, q8_words)
                .into_iter()
                .map(f32::to_bits)
                .collect();
            let want = quantize_activation_q8(&y2_pair);
            assert_eq!(got.len(), want.len());
            for (blk, (g, w)) in got.chunks(10).zip(want.chunks(10)).enumerate() {
                let (gd, wd) = (f32::from_bits(g[0]), f32::from_bits(w[0]));
                assert!(
                    (gd - wd).abs() <= 1e-6 * (1.0 + wd.abs()),
                    "width {n_embd} q8 block {blk}: scale {gd} vs {wd}"
                );
                assert_eq!(g[1], w[1], "width {n_embd} q8 block {blk}: quant sum");
                assert_eq!(&g[2..], &w[2..], "width {n_embd} q8 block {blk}: quants");
            }
        }

        // And the formula on the CPU, so a shared mistake cannot pass.
        let s1 = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() / n_embd as f32 + eps).sqrt();
        let y1: Vec<f32> = (0..n_embd).map(|i| x[i] * s1 * w1[i] + r[i]).collect();
        let s2 = 1.0 / (y1.iter().map(|v| v * v).sum::<f32>() / n_embd as f32 + eps).sqrt();
        for i in 0..n_embd {
            let want2 = y1[i] * s2 * w2[i];
            assert!(
                (y1_pair[i] - y1_scalar[i]).abs() <= 1e-5 * (1.0 + y1_scalar[i].abs()),
                "width {n_embd} y1 at {i}: pair {} vs scalar {}",
                y1_pair[i],
                y1_scalar[i]
            );
            assert!(
                (y2_pair[i] - y2_scalar[i]).abs() <= 1e-5 * (1.0 + y2_scalar[i].abs()),
                "width {n_embd} y2 at {i}: pair {} vs scalar {}",
                y2_pair[i],
                y2_scalar[i]
            );
            assert!(
                (y1_pair[i] - y1[i]).abs() <= 1e-4 * (1.0 + y1[i].abs()),
                "width {n_embd} y1 at {i}: pair {} vs cpu {}",
                y1_pair[i],
                y1[i]
            );
            assert!(
                (y2_pair[i] - want2).abs() <= 1e-4 * (1.0 + want2.abs()),
                "width {n_embd} y2 at {i}: pair {} vs cpu {want2}",
                y2_pair[i]
            );
        }
    }
}

/// The device top-k must return exactly the `k` largest penalized logits,
/// largest first, at a real vocabulary size — with the winners planted
/// across different slices (so a slice's local top-k and the merge are both
/// exercised), the penalty applied to some of them (so the ordering is the
/// *penalized* one), and a `k` that is neither 1 nor the cap.
#[test]
fn record_topk_sample_returns_the_k_largest_penalized_logits() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    let n_vocab = 262_144usize;
    let mut seed = 0x70CC_u64;
    let mut logits = vec![0f32; n_vocab];
    for v in logits.iter_mut() {
        *v = (next_byte(&mut seed) as f32 - 128.0) / 64.0;
    }
    // Planted winners across slices; two of them are "recent" and get
    // penalized below others.
    let planted: [(usize, f32); 6] = [
        (7, 9.0),
        (4_100, 8.5),
        (131_072, 8.0),
        (200_001, 7.5),
        (262_143, 7.0),
        (65_536, 6.5),
    ];
    for &(i, v) in &planted {
        logits[i] = v;
    }
    let recent: [u32; 2] = [7, 200_001];
    let repeat_penalty = 1.5f32;

    // The host's answer over the same rules.
    let mut penalized = logits.clone();
    for &t in &recent {
        let v = penalized[t as usize];
        penalized[t as usize] = if v > 0.0 {
            v / repeat_penalty
        } else {
            v * repeat_penalty
        };
    }
    let k = 40u32;
    let mut want: Vec<(u32, f32)> = penalized
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as u32, v))
        .collect();
    want.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    want.truncate(k as usize);

    let mut encoder = vulkan.new_encoder("test topk sample encoder");
    let buf = vulkan.record_topk_sample(
        &mut encoder,
        GpuArgmaxSampleInput {
            logits: GpuInput::Cpu(&logits),
            n_vocab,
            recent_tokens: &recent,
            repeat_penalty,
            logit_softcap: None,
        },
        k,
        2,
    );
    let got = vulkan.submit_and_readback_topk(encoder, &buf, k);
    assert_eq!(got.len(), k as usize);
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(g.0, w.0, "candidate {i}: gpu {g:?} vs host {w:?}");
        assert!(
            (g.1 - w.1).abs() <= 1e-5,
            "candidate {i}: gpu {g:?} vs host {w:?}"
        );
    }
    // The penalized planted tokens sit where the penalty put them: 7 (9.0
    // -> 6.0) below 65_536 (6.5), and 200_001 (7.5 -> 5.0) below both.
    let pos = |t: u32| {
        got.iter()
            .position(|c| c.0 == t)
            .expect("planted token in the top k")
    };
    assert!(pos(65_536) < pos(7));
    assert!(pos(7) < pos(200_001));
}

/// The device form of gemma4's per-layer-embedding inputs
/// (`ple_inputs_prefill`) against the same arithmetic on the host — the
/// projection scaled, RMS-normed per (token, layer) row, added to the
/// gathered rows, scaled — read back layer by layer at the layer-major,
/// capacity-strided layout the stage consumes. A token count that is not
/// a whole stripe, so the per-layer stride is exercised.
#[test]
fn ple_inputs_on_the_device_match_the_host_computation() {
    let _gpu_lock = gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    use crate::engine::backend::probe_blocks::next_byte;
    let mut seed = 0x9E37_79B9_u64;
    let (n_embd, n_layer, per_layer, n_tokens) = (512usize, 6usize, 256usize, 300usize);
    let total = n_layer * per_layer;
    // BF16, as the served model stores its projection.
    // Irregular values — nothing a 16-bit intermediate could hold exactly,
    // so a kernel that rounds its operands shows it here.
    let mut irregular = |scale: f32| {
        let a = next_byte(&mut seed) as f32;
        let b = next_byte(&mut seed) as f32;
        ((a * 0.37 + b * 0.011).sin() * 1.7 + (b - 128.0) / 300.0) * scale
    };
    let bf16: Vec<u8> = (0..total * n_embd)
        .flat_map(|_| (irregular(0.05).to_bits() >> 16).to_le_bytes())
        .collect();
    let proj_w = test_quant_matrix(&bf16, crate::engine::quant::GGML_TYPE_BF16, n_embd, total);
    let x: Vec<f32> = (0..n_tokens * n_embd).map(|_| irregular(2.0)).collect();
    let gathered: Vec<f32> = (0..n_tokens * total).map(|_| irregular(4.0)).collect();
    let norm_w: Vec<f32> = (0..per_layer).map(|_| 0.5 + irregular(0.2).abs()).collect();
    let eps = 1e-6;

    let Some(dev) = vulkan.ple_inputs_prefill(
        &x, n_tokens, &proj_w, &norm_w, &gathered, n_layer, per_layer, eps,
    ) else {
        eprintln!("device per-layer inputs declined — nothing to check");
        return;
    };
    // The chain was parked in no group, so it was submitted; the readback
    // orders after it.
    let cap = prefill_rows_capacity(n_tokens);
    let got = vulkan.readback_rows(&dev, n_layer * cap * per_layer);

    // The host arithmetic, on the same (8-bit-activation) GEMM the device
    // ran when it applies, else the float one — either way through the
    // backend, so only the norm/add/scale are compared here.
    let mut proj = CpuBackend.matmul_dequant(&x, n_tokens, &proj_w);
    {
        // How far the device GEMM itself is from the exact product, for
        // the record: the bound below has to hold over it.
        let dev_proj = vulkan.matmul(&x, n_tokens, &proj_w);
        let worst = proj
            .iter()
            .zip(&dev_proj)
            .map(|(a, b)| (a - b).abs() / (1.0 + a.abs()))
            .fold(0.0f32, f32::max);
        eprintln!("device projection vs exact: worst relative error {worst:.2e}");
    }
    let proj_scale = 1.0 / (n_embd as f32).sqrt();
    for v in proj.iter_mut() {
        *v *= proj_scale;
    }
    crate::engine::tensor::rmsnorm_inplace(&mut proj, &norm_w, n_tokens * n_layer, per_layer, eps);
    crate::engine::tensor::add_inplace(&mut proj, &gathered);
    let in_scale = 1.0 / 2f32.sqrt();
    let mut worst = 0.0f32;
    for t in 0..n_tokens {
        for il in 0..n_layer {
            for c in 0..per_layer {
                let want = proj[(t * n_layer + il) * per_layer + c] * in_scale;
                let have = got[(il * cap + t) * per_layer + c];
                let err = (have - want).abs() / (1.0 + want.abs());
                worst = worst.max(err);
                assert!(
                    err <= 1e-2,
                    "token {t} layer {il} elem {c}: device {have} vs host {want}"
                );
            }
        }
    }
    eprintln!("per-layer inputs on the device: worst relative error {worst:.2e}");
}

/// A wide projection of a word-reading type takes the four-row block-hoisted
/// pipeline at decode, a narrow one and a prefill-width batch do not.
#[test]
fn wide_projections_take_the_four_row_block_hoisted_pipeline() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    let mut seed = 0xB10C_u64;
    let block = build_block(GGML_TYPE_Q2_K, &mut seed);
    let wide = test_quant_matrix(&block.repeat(6 * 2048), GGML_TYPE_Q2_K, 1536, 2048);
    let narrow = test_quant_matrix(&block.repeat(6 * 256), GGML_TYPE_Q2_K, 1536, 256);
    assert!(vulkan.block_hoisted_wide_for(&wide, 1).is_some());
    assert_eq!(vulkan.decode_rows_per_workgroup(&wide, 1), 4);
    assert!(vulkan.block_hoisted_wide_for(&narrow, 1).is_none());
    assert!(vulkan.block_hoisted_wide_for(&wide, 64).is_none());
}

/// The activation+multiply with the q8 epilogue: the product itself
/// matches the plain fused kernel's, and the q8 rows match the CPU
/// quantizer of that product block for block — at a width that leaves a
/// partial last workgroup.
#[test]
fn the_activation_mul_q8_matches_the_plain_kernel_and_the_cpu_quantizer() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(gpu) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    for silu in [false, true] {
        let len = 6144usize + 32;
        let a: Vec<f32> = (0..len).map(|i| ((i % 23) as f32 - 11.0) * 0.3).collect();
        let b: Vec<f32> = (0..len).map(|i| ((i % 17) as f32 - 8.0) * 0.25).collect();
        let bytes = (len as u64) * 4;
        let ab = gpu.upload_new(&a);
        let bb = gpu.upload_new(&b);
        let y = gpu.upload_new(&vec![0.0f32; len]);
        let q8_words = len / 32 * 10;
        let q8 = gpu.upload_new(&vec![0.0f32; q8_words]);
        let meta = gpu.elem_meta_buffer(len as u32, 0.0);
        let mut encoder = gpu.new_encoder("activation q8");
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            let bg = gpu.norm_pair_bind_group(
                BindSrc::Slice(&ab, 0, bytes),
                &gpu.placeholder_ro,
                BindSrc::Slice(&bb, 0, bytes),
                &gpu.placeholder_ro,
                BindSrc::Slice(&y, 0, bytes),
                BindSrc::Slice(&q8, 0, (q8_words as u64) * 4),
                &meta,
                &gpu.q8_placeholder,
            );
            pass.set_pipeline(if silu {
                &gpu.silu_mul_q8_pipeline
            } else {
                &gpu.gelu_mul_q8_pipeline
            });
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((len as u32).div_ceil(64), 1, 1);
        }
        let product = gpu.submit_and_readback_for_test(encoder, &y, len);
        let got: Vec<u32> = gpu
            .readback_for_test(&q8, q8_words)
            .into_iter()
            .map(f32::to_bits)
            .collect();
        for i in 0..len {
            let g = if silu {
                a[i] / (1.0 + (-a[i]).exp())
            } else {
                0.5 * a[i]
                    * (1.0
                        + (0.7978846f32 * a[i] * (1.0 + 0.044715 * a[i] * a[i]))
                            .clamp(-20.0, 20.0)
                            .tanh())
            };
            let want = g * b[i];
            assert!(
                (product[i] - want).abs() <= 1e-4 * (1.0 + want.abs()),
                "silu={silu} product at {i}: {} vs {want}",
                product[i]
            );
        }
        // WGSL's `round` is half-to-even and the CPU quantizer's rounds half
        // away from zero, so a quant on an exact tie may differ by one, and
        // the block's sum by as many as differ.
        let want = quantize_activation_q8(&product);
        for (blk, (g, w)) in got.chunks(10).zip(want.chunks(10)).enumerate() {
            let (gd, wd) = (f32::from_bits(g[0]), f32::from_bits(w[0]));
            assert!(
                (gd - wd).abs() <= 1e-6 * (1.0 + wd.abs()),
                "silu={silu} q8 block {blk}: scale {gd} vs {wd}"
            );
            let bytes = |word: u32| (0..4).map(move |k| ((word >> (8 * k)) & 0xFF) as u8 as i8);
            let mut ties = 0i32;
            for (gw, ww) in g[2..].iter().zip(&w[2..]) {
                for (gq, wq) in bytes(*gw).zip(bytes(*ww)) {
                    let d = (i32::from(gq) - i32::from(wq)).abs();
                    assert!(d <= 1, "silu={silu} q8 block {blk}: quant {gq} vs {wq}");
                    ties += d;
                }
            }
            let ds = (g[1] as i32 - w[1] as i32).abs();
            assert!(
                ds <= ties,
                "silu={silu} q8 block {blk}: quant sum off by {ds}"
            );
        }
    }
}

/// The fused decode chain at a model shape with word-reading types on the
/// FFN — the case that takes the producer-quantized integer-dot path
/// (`FusedResources::ffn_q8`/`down_q8`) when `decode_mmvq` is on, and the
/// float word-reading kernels otherwise. Against the CPU reference at the
/// tolerance an 8-bit activation needs. The greedy check caught the first
/// wiring of this path producing nothing but separators while every
/// kernel's own test passed; this is the test that should have.
#[test]
fn fused_post_attention_decode_model_shaped_on_the_word_reading_types() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    let n_embd = 1536;
    let ffn_len = 6144;
    let eps = 1e-6;
    let mut seed = 0x0DD_u64;
    let build = |ggml_type: u32, in_dim: usize, out_dim: usize, seed: &mut u64| {
        let elems = block_elems(ggml_type);
        let mut bytes = Vec::new();
        for _ in 0..out_dim * (in_dim / elems) {
            bytes.extend(build_block(ggml_type, seed));
        }
        test_quant_matrix(&bytes, ggml_type, in_dim, out_dim)
    };
    let wo = build(GGML_TYPE_Q4_K, n_embd, n_embd, &mut seed);
    let ffn_gate = build(GGML_TYPE_Q2_K, n_embd, ffn_len, &mut seed);
    let ffn_up = build(GGML_TYPE_IQ3_S, n_embd, ffn_len, &mut seed);
    let ffn_down = build(GGML_TYPE_Q3_K, ffn_len, n_embd, &mut seed);
    let rand_vec = |len: usize, seed: &mut u64| -> Vec<f32> {
        (0..len)
            .map(|_| (next_byte(seed) as f32 - 128.0) / 256.0)
            .collect()
    };
    let attn_out = rand_vec(n_embd, &mut seed);
    let residual = rand_vec(n_embd, &mut seed);
    let near_one = |seed: &mut u64| -> Vec<f32> {
        rand_vec(n_embd, seed)
            .iter()
            .map(|v| 1.0 + v * 0.1)
            .collect()
    };
    let attn_post_norm = near_one(&mut seed);
    let ffn_norm = near_one(&mut seed);
    let ffn_post_norm = near_one(&mut seed);

    let mut attn_proj = CpuBackend.matmul_dequant(&attn_out, 1, &wo);
    crate::engine::tensor::rmsnorm_inplace(&mut attn_proj, &attn_post_norm, 1, n_embd, eps);
    let mut x = residual.clone();
    crate::engine::tensor::add_inplace(&mut x, &attn_proj);
    let x1 = x.clone();
    let mut ffn_normed = x.clone();
    crate::engine::tensor::rmsnorm_inplace(&mut ffn_normed, &ffn_norm, 1, n_embd, eps);
    let mut gate = CpuBackend.matmul_dequant(&ffn_normed, 1, &ffn_gate);
    let up = CpuBackend.matmul_dequant(&ffn_normed, 1, &ffn_up);
    for g in gate.iter_mut() {
        *g = crate::engine::tensor::gelu(*g);
    }
    crate::engine::tensor::mul_inplace(&mut gate, &up);
    let mut ffn_out = CpuBackend.matmul_dequant(&gate, 1, &ffn_down);
    crate::engine::tensor::rmsnorm_inplace(&mut ffn_out, &ffn_post_norm, 1, n_embd, eps);
    x = x1;
    crate::engine::tensor::add_inplace(&mut x, &ffn_out);
    let expected = x;

    let got = vulkan.fused_post_attention(FusedPostAttentionInput {
        stop_at_ffn_norm: false,
        activation: FfnActivation::Geglu,
        attn_out: GpuInput::Cpu(&attn_out),
        residual: GpuInput::Cpu(&residual),
        wo: &wo,
        attn_post_norm: Some(&attn_post_norm),
        ffn_norm: &ffn_norm,
        ffn_gate: &ffn_gate,
        ffn_up: &ffn_up,
        ffn_gate_up: None,
        ffn_down: &ffn_down,
        ffn_post_norm: Some(&ffn_post_norm),
        eps,
        post_norm_eps: None,
        ple: None,
        layer_output_scale: None,
        batch_slot: 0,
    });
    assert_eq!(expected.len(), got.len());
    let scale = expected.iter().fold(1.0f32, |m, v| m.max(v.abs()));
    for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
        assert!(
            (a - b).abs() <= 3e-2 * scale,
            "mismatch at index {i}: cpu={a} gpu(fused)={b} (scale {scale})"
        );
    }
}

/// A few tokens (under the coop threshold, under the thin-tile width) on
/// the integer-dot path through the CPU-orchestrated batch — the shape a
/// prompt's last short chunk takes.
#[test]
fn decode_matvec_integer_dot_a_few_tokens() {
    for n_tokens in [2usize, 3, 5] {
        cross_check_n_tokens(GGML_TYPE_Q2_K, 1536, 2055, n_tokens);
        cross_check_n_tokens(GGML_TYPE_Q3_K, 1536, 2055, n_tokens);
    }
}

/// The gate and up projections as one dispatch (`ffn_gate_up`, a
/// `QuantMatrix::adjacent_pair` view) against the two-dispatch form and
/// the CPU reference, at a llama-family shape: SwiGLU, no post-norms, and
/// a `Q8_0` pair so the integer-dot form of the pair runs where the
/// device has it. The halves are views of one tensor, which is exactly
/// what a checkpoint that stores `ffn_gate` then `ffn_up` hands the arch.
#[test]
fn fused_post_attention_gate_up_pair_matches_the_two_dispatches() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };

    let n_embd = 960;
    let ffn_len = 2560;
    let eps = 1e-5;
    let mut seed = 0x6A7E_u64;
    // `Q8_0` blocks at one small scale: random scales put the activations
    // in the billions, where the 8-bit activation rounding of the
    // integer-dot path is all a float reference can see.
    let build = |in_dim: usize, out_dim: usize, seed: &mut u64| {
        let scale = half::f16::from_f32(1.0 / 64.0).to_le_bytes();
        let mut bytes = Vec::new();
        for _ in 0..out_dim * (in_dim / 32) {
            bytes.extend_from_slice(&scale);
            bytes.extend((0..32).map(|_| next_byte(seed)));
        }
        test_quant_matrix(&bytes, GGML_TYPE_Q8_0, in_dim, out_dim)
    };
    let wo = build(n_embd, n_embd, &mut seed);
    // One tensor holding gate's rows then up's; the halves are row views.
    let gate_up = build(n_embd, 2 * ffn_len, &mut seed);
    let ffn_gate = gate_up.rows(0, ffn_len);
    let ffn_up = gate_up.rows(ffn_len, ffn_len);
    let pair = QuantMatrix::adjacent_pair(&ffn_gate, &ffn_up).expect("the halves are adjacent");
    assert_eq!(pair.cache_key(), gate_up.cache_key());
    let ffn_down = build(ffn_len, n_embd, &mut seed);
    let rand_vec = |len: usize, seed: &mut u64| -> Vec<f32> {
        (0..len)
            .map(|_| (next_byte(seed) as f32 - 128.0) / 64.0)
            .collect()
    };
    let attn_out = rand_vec(n_embd, &mut seed);
    let residual = rand_vec(n_embd, &mut seed);
    let ffn_norm = rand_vec(n_embd, &mut seed);

    let attn_proj = CpuBackend.matmul_dequant(&attn_out, 1, &wo);
    let mut x = residual.clone();
    crate::engine::tensor::add_inplace(&mut x, &attn_proj);
    let mut ffn_normed = x.clone();
    crate::engine::tensor::rmsnorm_inplace(&mut ffn_normed, &ffn_norm, 1, n_embd, eps);
    let mut gate = CpuBackend.matmul_dequant(&ffn_normed, 1, &ffn_gate);
    let up = CpuBackend.matmul_dequant(&ffn_normed, 1, &ffn_up);
    for g in gate.iter_mut() {
        *g = crate::engine::tensor::silu(*g);
    }
    crate::engine::tensor::mul_inplace(&mut gate, &up);
    let ffn_out = CpuBackend.matmul_dequant(&gate, 1, &ffn_down);
    crate::engine::tensor::add_inplace(&mut x, &ffn_out);
    let expected = x;

    let run = |gate_up: Option<&QuantMatrix>| {
        vulkan.fused_post_attention(FusedPostAttentionInput {
            stop_at_ffn_norm: false,
            activation: FfnActivation::Swiglu,
            attn_out: GpuInput::Cpu(&attn_out),
            residual: GpuInput::Cpu(&residual),
            wo: &wo,
            attn_post_norm: None,
            ffn_norm: &ffn_norm,
            ffn_gate: &ffn_gate,
            ffn_up: &ffn_up,
            ffn_gate_up: gate_up,
            ffn_down: &ffn_down,
            ffn_post_norm: None,
            eps,
            post_norm_eps: None,
            ple: None,
            layer_output_scale: None,
            // Its own slot: the two-dispatch entry for this `wo` is keyed
            // by slot, and the pair must not inherit resources built
            // without it.
            batch_slot: gate_up.map_or(0, |_| 1),
        })
    };
    let paired = run(Some(&pair));
    let split = run(None);
    assert_eq!(expected.len(), paired.len());
    // The two device forms are the same rows through the same kernel and
    // must agree exactly; against the float reference both carry the
    // 8-bit activation rounding of the integer-dot path, which is of the
    // row's scale, not the element's — an element near zero is a
    // cancellation of terms that size.
    let scale = expected.iter().fold(0f32, |m, v| m.max(v.abs()));
    for (i, ((a, b), c)) in expected.iter().zip(&paired).zip(&split).enumerate() {
        assert_eq!(b, c, "pair vs split at {i}");
        let tol = 2e-2 * scale;
        assert!((a - b).abs() <= tol, "pair vs cpu at {i}: cpu={a} pair={b}");
    }
}

/// The one-submission routed feed-forward without a gate projection
/// (`MoeActivation::ReluSquared` — `down(relu(up(x))²)`, the shape of a
/// `nemotron_h_moe` expert) against the same computation by parts on the
/// CPU, with the rows combined on the card. The up stack alone crosses
/// the bus and the activation reads one projection; everything else is
/// the gated form's, which is what the test is for.
#[test]
fn the_fused_routed_ffn_without_a_gate_matches_the_cpu_computation() {
    use crate::engine::backend::vulkan::{ExpertOp, MoeCombine};
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    const N_EXPERT: usize = 5;
    const N_TOKENS: usize = 200;
    const N_EMBD: usize = 512;
    const N_FF: usize = 128;
    let mut seed = 0x05EE_D0FF_u64;
    let up_blocks = N_EXPERT * N_FF * (N_EMBD / block_elems(GGML_TYPE_Q4_K));
    let up_bytes: Vec<u8> = (0..up_blocks)
        .flat_map(|_| build_block(GGML_TYPE_Q4_K, &mut seed))
        .collect();
    let up_stack = test_quant_matrix(&up_bytes, GGML_TYPE_Q4_K, N_EMBD, N_EXPERT * N_FF);
    let q8_0 = crate::engine::quant::GGML_TYPE_Q8_0;
    let dn_blocks = N_EXPERT * N_EMBD * (N_FF / block_elems(q8_0));
    let dn_bytes: Vec<u8> = (0..dn_blocks)
        .flat_map(|_| build_block(q8_0, &mut seed))
        .collect();
    let dn_stack = test_quant_matrix(&dn_bytes, q8_0, N_FF, N_EXPERT * N_EMBD);
    let x: Vec<f32> = (0..N_TOKENS * N_EMBD)
        .map(|_| (next_byte(&mut seed) as f32 - 128.0) / 65536.0)
        .collect();
    let mut groups: Vec<(usize, Vec<usize>)> = (0..N_EXPERT).map(|e| (e, Vec::new())).collect();
    for t in 0..N_TOKENS {
        groups[t % 3].1.push(t);
        groups[3 + (t % 2)].1.push(t);
    }
    let up = ExpertOp {
        stack: &up_stack,
        rows_per_expert: N_FF,
        first_row: 0,
        n_rows: N_FF,
        scale: None,
    };
    let down = ExpertOp {
        stack: &dn_stack,
        rows_per_expert: N_EMBD,
        first_row: 0,
        n_rows: N_EMBD,
        scale: None,
    };
    if [&up, &down]
        .into_iter()
        .any(|op| !vulkan.serves_experts(std::slice::from_ref(op)))
    {
        eprintln!("no indexed kernel on this device — nothing to check");
        return;
    }
    let got = vulkan.moe_ffn_experts(
        &x,
        N_TOKENS,
        N_EMBD,
        None,
        &up,
        &down,
        &groups,
        MoeActivation::ReluSquared,
        None,
        &[],
    );
    assert_eq!(got.len(), groups.len());
    for ((expert, tokens), result) in groups.iter().zip(&got) {
        let mut gathered = Vec::with_capacity(tokens.len() * N_EMBD);
        for &t in tokens {
            gathered.extend_from_slice(&x[t * N_EMBD..(t + 1) * N_EMBD]);
        }
        let wu = up_stack.rows(expert * N_FF, N_FF);
        let wd = dn_stack.rows(expert * N_EMBD, N_EMBD);
        let mut h = CpuBackend.matmul_dequant(&gathered, tokens.len(), &wu);
        for v in h.iter_mut() {
            let r = v.max(0.0);
            *v = r * r;
        }
        let want = CpuBackend.matmul_dequant(&h, tokens.len(), &wd);
        for t in 0..tokens.len() {
            let row = &want[t * N_EMBD..(t + 1) * N_EMBD];
            let mag = row.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-3);
            for o in 0..N_EMBD {
                let i = t * N_EMBD + o;
                assert!(
                    (result[i] - want[i]).abs() / mag <= 5e-2,
                    "expert {expert} member {t} row {o}: fused {} vs cpu {} (row magnitude {mag})",
                    result[i],
                    want[i]
                );
            }
        }
    }

    // Combined on the card: two picks per token, weighted 0.25 and 0.75.
    let k = 2;
    let mut table = vec![0u32; 2 * N_TOKENS * k];
    let mut base = 0usize;
    for (g, (_, tokens)) in groups.iter().enumerate() {
        for (m, &t) in tokens.iter().enumerate() {
            let rank = if g < 3 { 0 } else { 1 };
            table[t * k + rank] = (base + m) as u32;
            table[N_TOKENS * k + t * k + rank] =
                if rank == 0 { 0.25f32 } else { 0.75f32 }.to_bits();
        }
        base += tokens.len();
    }
    let combine = MoeCombine {
        table,
        n_tokens: N_TOKENS,
        k,
    };
    let combined = vulkan
        .moe_ffn_experts(
            &x,
            N_TOKENS,
            N_EMBD,
            None,
            &up,
            &down,
            &groups,
            MoeActivation::ReluSquared,
            Some(&combine),
            &[],
        )
        .pop()
        .expect("the combined rows");
    let mut want = vec![0f32; N_TOKENS * N_EMBD];
    for (g, (_, tokens)) in groups.iter().enumerate() {
        let w = if g < 3 { 0.25 } else { 0.75 };
        for (m, &t) in tokens.iter().enumerate() {
            for e in 0..N_EMBD {
                want[t * N_EMBD + e] += w * got[g][m * N_EMBD + e];
            }
        }
    }
    for t in 0..N_TOKENS {
        let row = &want[t * N_EMBD..(t + 1) * N_EMBD];
        let mag = row.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-3);
        for e in 0..N_EMBD {
            let i = t * N_EMBD + e;
            assert!(
                (combined[i] - want[i]).abs() / mag <= 1e-3,
                "token {t} element {e}: combined {} vs summed rows {}",
                combined[i],
                want[i]
            );
        }
    }
}

/// **Every integer-dot GEMM pipeline, built on this platform's own
/// backend.**
///
/// The kernels are generated as WGSL and translated by `wgpu` into SPIR-V
/// here, Metal on Apple and DXIL on Windows — and a kernel the SPIR-V path
/// accepts can still be refused by another. One was: an `M2 Max` rejected
/// the byte-unpacked GEMM because the Metal translation declared the same
/// polyfill temporary twice in a block, and since these pipelines are built
/// on first *use*, the refusal arrived on a user's first prompt rather than
/// at startup.
///
/// `vulkan_shaders::msl_tests` checks the translation itself and needs no
/// device; this builds them, which is the half only a device can answer.
/// Skips itself where there is no adapter.
#[test]
fn every_integer_dot_pipeline_builds() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    let Some(built) = vulkan.build_every_mmq_pipeline_for_test() else {
        eprintln!(
            "this adapter has no accelerated integer dot, so the whole family \
             is switched off and there is nothing to build"
        );
        return;
    };
    eprintln!("built {built} integer-dot GEMM pipelines on this backend");
    assert!(
        built > 0,
        "the family is switched on and yet not one kernel built — every one \
         of them was refused by this platform's shader compiler"
    );
}
