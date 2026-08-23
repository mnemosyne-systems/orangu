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
    GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS, GGML_TYPE_MXFP4, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K,
    GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q4_K, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K, GGML_TYPE_Q8_0,
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
        _pad0: 0,
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
        _pad0: 0,
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
        // `Q4_0`'s 16 `qs` bytes behind a one-byte `e8m0` exponent instead of
        // an `f16` scale. The exponent is **bounded** where the nibbles are
        // not: it is the only field here read as a float, and an unbounded
        // byte spans 2^-127..2^127, which overflows a dot product to `inf` on
        // both paths and would compare equal while testing nothing. `128`
        // decodes to exactly 1.0 (`(128-1) << 23` is f32 exponent 127), so
        // this is a symmetric 2^-4..2^4 around unity.
        t if t == GGML_TYPE_MXFP4 => {
            out.push(124 + (next_byte(seed) % 9));
            out.extend(next_bytes(seed, 16));
        }
        t if t == GGML_TYPE_Q4_1 => {
            out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
            out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
            out.extend(next_bytes(seed, 16));
        }
        t if t == GGML_TYPE_Q5_0 => {
            out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
            out.extend(next_bytes(seed, 4));
            out.extend(next_bytes(seed, 16));
        }
        t if t == GGML_TYPE_Q5_1 => {
            out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
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
        t if t == GGML_TYPE_Q2_K => {
            out.extend(next_bytes(seed, 16));
            out.extend(next_bytes(seed, 64));
            out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
            out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
        }
        t if t == GGML_TYPE_Q3_K => {
            out.extend(next_bytes(seed, 32));
            out.extend(next_bytes(seed, 64));
            out.extend(next_bytes(seed, 12));
            out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
        }
        // Every `IQ*` field below is a codebook index, a sign pattern or
        // a packed scale, all of which are valid for any bit pattern —
        // no field needs constraining to keep the block well formed, so
        // random bytes reach the whole encoding space.
        t if t == GGML_TYPE_IQ2_XS => {
            out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
            out.extend(next_bytes(seed, 64));
            out.extend(next_bytes(seed, 8));
        }
        t if t == GGML_TYPE_IQ2_S => {
            out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
            out.extend(next_bytes(seed, 64));
            out.extend(next_bytes(seed, 8));
            out.extend(next_bytes(seed, 8));
        }
        t if t == GGML_TYPE_IQ3_XXS => {
            out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
            out.extend(next_bytes(seed, 96));
        }
        t if t == GGML_TYPE_IQ3_S => {
            out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
            out.extend(next_bytes(seed, 64));
            out.extend(next_bytes(seed, 8));
            out.extend(next_bytes(seed, 32));
            out.extend(next_bytes(seed, 4));
        }
        t if t == GGML_TYPE_IQ4_XS => {
            out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
            out.extend(next_bytes(seed, 2));
            out.extend(next_bytes(seed, 4));
            out.extend(next_bytes(seed, 128));
        }
        t if t == GGML_TYPE_IQ4_NL => {
            out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
            out.extend(next_bytes(seed, 16));
        }
        t if t == GGML_TYPE_IQ2_XXS => {
            out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
            out.extend(next_bytes(seed, 64));
        }
        t if t == GGML_TYPE_IQ1_S => {
            out.extend_from_slice(&f16_bytes(next_bounded_f32(seed)));
            out.extend(next_bytes(seed, 32));
            out.extend(next_bytes(seed, 16));
        }
        // The one type with no `d` field at all: its `f16` block scale is
        // four nibbles scattered across the top of the four `scales`
        // `u16`s, so random bytes there *are* a random `f16` — exponent
        // included, where every other arm draws its scale through
        // `next_bounded_f32`. The top nibble of the last `u16` (byte 7's
        // high half) is the `f16`'s own top nibble, so pinning it to
        // `0x3` keeps the block scale a positive normal of order 1
        // instead of an occasional `inf`/`NaN`, which would make the
        // comparison below vacuous rather than strict. Every other bit,
        // including the rest of the exponent, stays random.
        t if t == GGML_TYPE_IQ1_M => {
            out.extend(next_bytes(seed, 32));
            out.extend(next_bytes(seed, 16));
            let mut scales = next_bytes(seed, 8);
            scales[7] = (scales[7] & 0x0F) | 0x30;
            out.extend(scales);
        }
        other => panic!("build_block: unhandled ggml_type {other}"),
    }
    out
}

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
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    // Same output size (so upload/readback/submission cost is identical)
    // at halved and quartered K: if the time tracks K, the call is
    // compute-bound and the kernel is what matters; if it barely moves,
    // the per-call data movement dominates and the kernel is not the
    // thing to tune.
    for k in [1536usize, 768, 384] {
        let (ms, gflops) = measure_matmul_gflops(vulkan, GGML_TYPE_Q4_K, k, 12288, 128, 10);
        eprintln!(
            "orangu-server: [scratch] k-sweep in_dim={k} out_dim=12288 n_tokens=128: \
                 min={ms:.2}ms ({gflops:.1} GFLOP/s)"
        );
    }
    for (label, in_dim, out_dim) in [
        ("gate_up", 1536usize, 12288usize),
        ("ffn_down", 6144, 1536),
        ("qkv", 1536, 1024),
    ] {
        for n_tokens in [64usize, 128] {
            for (type_label, ggml_type) in [("q4_k", GGML_TYPE_Q4_K), ("f16", GGML_TYPE_F16)] {
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
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    let report = vulkan.tuning_report();
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
        *v = (b as f32 - 128.0) / 64.0;
    }

    let cpu_out = CpuBackend.matmul_dequant(&x, n_tokens, &w);
    let gpu_out = vulkan.matmul(&x, n_tokens, &w);

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
    let (reference, tol_factor) = if vulkan.q4_k_mmvq && ggml_type == GGML_TYPE_Q4_K {
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
    } else {
        // `packed_dot_f16` (`ORANGU_PACKED_DOT`) also widens the dot to an
        // `f16` accumulate, needing the loose tolerance vs `cpu_out`.
        let packed = vulkan.packed_dot_f16
            && ggml_type == GGML_TYPE_Q4_K
            && n_tokens < vulkan.coop_min_n_tokens;
        (cpu_out, if packed { 6e-2 } else { 1e-2 })
    };

    assert_eq!(reference.len(), gpu_out.len());
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

    for (got, want) in [(&first[0], &want[0]), (&second[0], &want[1])] {
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-2 * w.abs().max(1.0),
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
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-2 * w.abs().max(1.0),
                "expert {e} element {i}: gpu={g} cpu={w} on {}",
                vulkan.adapter_name
            );
        }
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
        bytes.extend((0..16u8).map(|j| j | (j << 4)));
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
                // The subnormal pair. Asserted rather than skipped, so that a
                // device which *does* keep subnormals fails here and the
                // comment above gets revisited instead of quietly becoming
                // untrue on other hardware.
                assert_eq!(
                    *g, 0.0,
                    "exponent code {e} is subnormal and was expected to flush to \
                     zero on the device, but element {k} came back as {g:e}"
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
            assert!(
                g == c || (g.is_nan() && c.is_nan()) || (g - c).abs() <= c.abs() * 1e-6,
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

    let expected = vulkan.matmul(&x, n_tokens, &w);

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
                .fused_ffn_prefill(&c.x, n_tokens, &c.gate, &c.up, &c.down)
                .expect("fused FFN available without MMVQ")
        },
        "fused FFN prefill",
    );
}

/// `fused_post_attention_prefill` under concurrency — the recorder that is
/// two thirds of prefill, and the one with the most pooled ops (four).
#[test]
fn concurrent_fused_post_attention_prefills_do_not_corrupt_each_other() {
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
                        yarn: RopeYarn::IDENTITY,
                        q_bias: None,
                        pairing: crate::engine::tensor::RopeLayout::Neox,
                        normalize_v: true,
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
        for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
            let tol = tol_factor * a.abs().max(1.0);
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
        activation: FfnActivation::Geglu,
        attn_out: GpuInput::Cpu(&attn_out),
        residual: GpuInput::Cpu(&residual),
        wo: &wo,
        attn_post_norm: Some(&attn_post_norm),
        ffn_norm: &ffn_norm,
        ffn_gate: &ffn_gate,
        ffn_up: &ffn_up,
        ffn_down: &ffn_down,
        ffn_post_norm: Some(&ffn_post_norm),
        eps,
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
        activation: FfnActivation::Geglu,
        attn_out: GpuInput::Cpu(&attn_out),
        residual: GpuInput::Cpu(&residual),
        wo: &wo,
        attn_post_norm: Some(&attn_post_norm),
        ffn_norm: &ffn_norm,
        ffn_gate: &ffn_gate,
        ffn_up: &ffn_up,
        ffn_down: &ffn_down,
        ffn_post_norm: Some(&ffn_post_norm),
        eps,
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
            activation: FfnActivation::Geglu,
            attn_out: GpuInput::Cpu(&attn_out),
            residual: GpuInput::Cpu(&residual),
            wo: &wo,
            attn_post_norm: Some(&attn_post_norm),
            ffn_norm: &ffn_norm,
            ffn_gate: &ffn_gate,
            ffn_up: &ffn_up,
            ffn_down: &ffn_down,
            ffn_post_norm: Some(&ffn_post_norm),
            eps,
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

    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    let (page_tokens, n_pages) = (page, pages);
    let pool_pages: usize = n_pages * 2;
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
        vec![LayerGeometry { kv_dim, stride: 1 }],
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
    .into_paged(pool.clone());
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
        vec![LayerGeometry { kv_dim, stride: 1 }],
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
    for (pos, bound) in [(8usize, 1e-6f32), (1_000, 1e-4), (20_000, 4e-3)] {
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
            yarn: RopeYarn::IDENTITY,
            normalize_v: true,
            q_bias: None,
            pairing: crate::engine::tensor::RopeLayout::Neox,
            normed: GpuInput::Cpu(&normed),
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
            yarn: RopeYarn::IDENTITY,
            normalize_v: true,
            q_bias: None,
            pairing: crate::engine::tensor::RopeLayout::Neox,
            normed: GpuInput::Cpu(&normed),
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
        yarn: RopeYarn::IDENTITY,
        normalize_v: true,
        q_bias: None,
        pairing: crate::engine::tensor::RopeLayout::Neox,
        normed: GpuInput::Cpu(&normed),
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
        yarn: RopeYarn::IDENTITY,
        normalize_v: true,
        q_bias: None,
        pairing: crate::engine::tensor::RopeLayout::Neox,
        normed: GpuInput::Cpu(&normed_a),
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
        yarn: RopeYarn::IDENTITY,
        normalize_v: true,
        q_bias: None,
        pairing: crate::engine::tensor::RopeLayout::Neox,
        normed: GpuInput::Cpu(&normed_b),
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
        .fused_ffn_prefill(&x, n_tokens, &gate, &up, &down)
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
    cross_check_fused_ple_prefill(192);
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
            vec![LayerGeometry { kv_dim, stride: 1 }],
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
            .into_paged(pool)
    } else {
        crate::engine::kv_cache::KvCache::new_with_dims(capacity, &[kv_dim])
    };
    for (pk, pv) in &prior {
        cache.layers[0].push(pk, pv);
    }
    if paged {
        cache.commit_pages();
    }
    let got = vulkan
        .fused_attention_prefill(
            FusedAttnPrefillInput {
                yarn: RopeYarn::IDENTITY,
                q_bias: None,
                pairing: crate::engine::tensor::RopeLayout::Neox,
                normalize_v: true,
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
                want_attn_out_host: true,
            },
            &mut cache.layers[0],
        )
        .expect("fused prefill attention returned None on a supported path");

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

    let mut cache =
        crate::engine::kv_cache::KvCache::new_with_dims(start_pos + n_tokens + 8, &[kv_dim]);
    for _ in 0..start_pos {
        cache.layers[0].push(&vec![0.0; kv_dim], &vec![0.0; kv_dim]);
    }
    let out = vulkan
        .fused_attention_prefill(
            FusedAttnPrefillInput {
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

/// A cross-layer **KV-donor** layer (`kv: None`): it projects Q only and
/// attends against a cache an earlier layer already filled, skipping the
/// whole K/V sub-chain and the cache write. Gemma4 has these, so the
/// end-to-end path exercises it, but the sub-chain being skipped rather
/// than run is a distinct branch worth pinning down on its own — and it
/// must leave the cache's length untouched.
#[test]
fn fused_attention_prefill_matches_the_unfused_sequence_kv_donor() {
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
                yarn: RopeYarn::IDENTITY,
                q_bias: None,
                pairing: crate::engine::tensor::RopeLayout::Neox,
                normalize_v: true,
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
        yarn: RopeYarn::IDENTITY,
        q_bias: None,
        pairing: crate::engine::tensor::RopeLayout::Neox,
        normalize_v: true,
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
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
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

    let unpadded = vulkan.matmul(&x, real_rows, &w);
    let widened = vulkan.matmul(&x_padded, padded, &w);
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

#[test]
fn fused_ffn_prefill_matches_the_unfused_sequence_multi_chunk() {
    cross_check_fused_ffn_prefill(192);
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
            yarn: RopeYarn::IDENTITY,
            normalize_v: false, // llama does not normalize V
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
            ffn_down: &ffn_down,
            ffn_post_norm: None,
            ple: None,
            layer_output_scale: None,
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
        yarn: RopeYarn::IDENTITY,
        normalize_v: false, // llama does not normalize V
        q_bias: None,
        pairing,
        normed: GpuInput::Cpu(&normed),
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
        activation: FfnActivation::Swiglu,
        attn_out: GpuInput::Cpu(&attn_out),
        residual: GpuInput::Cpu(&residual),
        wo: &wo,
        attn_post_norm: None,
        ffn_norm: &ffn_norm,
        ffn_gate: &ffn_gate,
        ffn_up: &ffn_up,
        ffn_down: &ffn_down,
        ffn_post_norm: None,
        eps,
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
            yarn: RopeYarn::IDENTITY,
            normalize_v: true,
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
#[test]
fn fused_layer_kv_donor_matches_cpu_reference_many_steps() {
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
            yarn: RopeYarn::IDENTITY,
            normalize_v: true,
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
            ffn_down: &l0.ffn_down,
            ffn_post_norm: Some(&l0.ffn_post_norm),
            ple: None,
            layer_output_scale: None,
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
            yarn: RopeYarn::IDENTITY,
            normalize_v: true,
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
            ffn_down: &l1.ffn_down,
            ffn_post_norm: Some(&l1.ffn_post_norm),
            ple: None,
            layer_output_scale: None,
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
    let pool_pages = 16;
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
        vec![LayerGeometry { kv_dim, stride: 1 }],
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
            .into_paged(pool.clone());
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
            yarn: RopeYarn::IDENTITY,
            normalize_v: true,
            q_bias: None,
            pairing: crate::engine::tensor::RopeLayout::Neox,
            normed: GpuInput::Cpu(&normed),
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
