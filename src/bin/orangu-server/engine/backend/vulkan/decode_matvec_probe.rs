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

//! **Why is a small matmul slow on the GPU?** Measures the fixed cost of one
//! decode-shaped `matmul` round trip, and how much of it a submission is.
//!
//! A decode step is a chain of `[1, in] x [in, out]` products — one activation
//! row against one weight matrix, dozens of times per layer. Each is a
//! separate call, so each pays whatever a call costs whether or not there is
//! any arithmetic worth paying it for. That cost is invisible in a throughput
//! number and invisible in a CPU profile (the thread is parked in the driver);
//! the only way to see it is to hold the arithmetic still and vary the number
//! of calls.
//!
//! Three sweeps, all on one weight shape family so the arithmetic is the only
//! thing that moves:
//!
//! 1. **Size** — the same call over growing weights. A straight line through
//!    these points is `fixed + bytes / bandwidth`; the intercept is what one
//!    call costs with nothing to compute, and where it crosses the CPU's own
//!    line is the size below which the device is the wrong place to run.
//! 2. **Batch** — `n` independent products issued as `n` calls, then as one
//!    `matmul_batch`. The ratio says how much of the fixed cost is *per
//!    submission* (and so removable by batching) rather than per dispatch.
//! 3. **Depth** — `n` products issued back-to-back without reading any of them
//!    back, against the same `n` read back one at a time. The difference is
//!    the round trip: submit, fence, wake, map.
//!
//! Nothing here is a benchmark of the model. It is the unit cost the model's
//! own numbers are made of, and it is reported per call in microseconds so it
//! can be multiplied by a token's call count directly.
//!
//! **Two clocks.** The wall-clock columns are the call as the host sees it,
//! fixed cost included; the `kernel us` column is the dispatch alone, from
//! the device's own timestamps around a burst of back-to-back dispatches
//! (`VulkanBackend::dispatch_kernel_us`). At a decode shape the two differ
//! by more than the kernel itself, and a short burst is read at whatever
//! clock the device idles at — `ORANGU_SWEEP_KERNEL_REPS` lengthens it.
//! A kernel change is judged by the second clock, never the first.
//!
//! `cargo test --release --bin orangu-server decode_matvec -- --ignored --nocapture`

use super::*;
use crate::engine::backend::{Backend, CpuBackend, MatmulOp};
use crate::engine::loader::test_quant_matrix;
use crate::engine::quant::GGML_TYPE_Q4_K;

/// Q4_K packs 256 weights into 144 bytes.
const Q4_K_BLOCK: usize = 256;
const Q4_K_BYTES: usize = 144;

fn weight(in_dim: usize, out_dim: usize) -> crate::engine::loader::QuantMatrix {
    let bytes = vec![0x42u8; in_dim * out_dim / Q4_K_BLOCK * Q4_K_BYTES];
    test_quant_matrix(&bytes, GGML_TYPE_Q4_K, in_dim, out_dim)
}

fn mib(in_dim: usize, out_dim: usize) -> f64 {
    (in_dim * out_dim / Q4_K_BLOCK * Q4_K_BYTES) as f64 / (1024.0 * 1024.0)
}

/// Median of `reps` timings, in microseconds. Median rather than mean: a
/// single scheduler hiccup in a hundred-microsecond measurement moves a mean
/// by more than the effect being measured.
fn median_us(reps: usize, mut run: impl FnMut()) -> f64 {
    run();
    let mut samples: Vec<f64> = (0..reps)
        .map(|_| {
            let t = std::time::Instant::now();
            run();
            t.elapsed().as_secs_f64() * 1e6
        })
        .collect();
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

/// A least-squares fit of `us = fixed + mib * slope`, reported as the
/// intercept (microseconds per call with nothing to compute) and the implied
/// bandwidth.
fn fit(points: &[(f64, f64)]) -> (f64, f64) {
    let n = points.len() as f64;
    let sx: f64 = points.iter().map(|p| p.0).sum();
    let sy: f64 = points.iter().map(|p| p.1).sum();
    let sxx: f64 = points.iter().map(|p| p.0 * p.0).sum();
    let sxy: f64 = points.iter().map(|p| p.0 * p.1).sum();
    let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    (sy / n - slope * sx / n, slope)
}

#[test]
#[ignore]
fn decode_matvec_fixed_cost_gpu_versus_cpu() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(gpu) = shared_test_backend() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    let cpu = CpuBackend;
    let reps = 21;

    // One activation row — a decode step, which is the whole point: at
    // `n_tokens == 1` there is nothing to amortize a call over.
    println!("\n== 1. size sweep: [1 x in] x [in x out] Q4_K, one call ==");
    println!(
        "{:>7} {:>7} {:>9} {:>11} {:>11} {:>9}",
        "in", "out", "MiB", "GPU us", "CPU us", "GPU/CPU"
    );
    let in_dim = 4096usize;
    let mut gpu_points = Vec::new();
    let mut cpu_points = Vec::new();
    for &out_dim in &[512usize, 1024, 2048, 4096, 8192, 16384, 32768] {
        let w = weight(in_dim, out_dim);
        let x = vec![0.05f32; in_dim];
        let size = mib(in_dim, out_dim);
        let gpu_us = median_us(reps, || {
            let _ = gpu.matmul(&x, 1, &w);
        });
        let cpu_us = median_us(reps, || {
            let _ = cpu.matmul(&x, 1, &w);
        });
        gpu_points.push((size, gpu_us));
        cpu_points.push((size, cpu_us));
        println!(
            "{in_dim:>7} {out_dim:>7} {size:>9.2} {gpu_us:>11.1} {cpu_us:>11.1} {:>8.2}x",
            gpu_us / cpu_us.max(1e-9)
        );
    }
    let (gpu_fixed, gpu_slope) = fit(&gpu_points);
    let (cpu_fixed, cpu_slope) = fit(&cpu_points);
    println!(
        "  GPU  fixed {gpu_fixed:.1} us/call  +  {:.1} GiB/s",
        1e6 / gpu_slope / 1024.0
    );
    println!(
        "  CPU  fixed {cpu_fixed:.1} us/call  +  {:.1} GiB/s",
        1e6 / cpu_slope / 1024.0
    );
    // Where the two lines cross: below this many MiB per call the device is
    // the slower place to run, however fast it is once it is running.
    let crossover = (gpu_fixed - cpu_fixed) / (cpu_slope - gpu_slope);
    if crossover.is_finite() && crossover > 0.0 {
        println!("  crossover at {crossover:.2} MiB per call — smaller than this, the CPU wins");
    }

    // Same arithmetic, issued two ways. `matmul_batch` is the backend's own
    // one-submission form, so the difference between the two rows is exactly
    // what a submission costs.
    println!("\n== 2. batching: 8 independent [1 x 4096] x [4096 x 2048] products ==");
    let w = weight(in_dim, 2048);
    let x = vec![0.05f32; in_dim];
    let ops: Vec<MatmulOp<'_>> = (0..8)
        .map(|_| MatmulOp {
            x: &x,
            n_tokens: 1,
            w: &w,
        })
        .collect();
    let separate = median_us(reps, || {
        for _ in 0..8 {
            let _ = gpu.matmul(&x, 1, &w);
        }
    });
    let batched = median_us(reps, || {
        let _ = gpu.matmul_batch(&ops);
    });
    println!(
        "  8 separate calls   {separate:>9.1} us  ({:.1} us each)",
        separate / 8.0
    );
    println!(
        "  1 matmul_batch     {batched:>9.1} us  ({:.1} us each)",
        batched / 8.0
    );
    println!(
        "  batching saves     {:>9.1} us per call — that share of the fixed cost is per-submission",
        (separate - batched) / 8.0
    );

    // The CPU's own comparison for the same eight, so the batched GPU number
    // has something to be better or worse than.
    let cpu_eight = median_us(reps, || {
        for _ in 0..8 {
            let _ = cpu.matmul(&x, 1, &w);
        }
    });
    println!(
        "  8 on the CPU       {cpu_eight:>9.1} us  ({:.1} us each)",
        cpu_eight / 8.0
    );

    // The same eight products' worth of weight bytes as **one** product with
    // eight times the output width: one submission and one result, where the
    // batch above is one submission and eight results. The gap between the two
    // is what reading each result back separately costs, and it is the part of
    // the fixed cost batching cannot reach.
    println!("\n== 3. readback: the same bytes, 8 results against 1 ==");
    let tall = weight(in_dim, 2048 * 8);
    let one_result = median_us(reps, || {
        let _ = gpu.matmul(&x, 1, &tall);
    });
    println!("  1 submission, 8 results  {batched:>9.1} us");
    println!("  1 submission, 1 result   {one_result:>9.1} us");
    println!(
        "  reading 8 results back separately costs {:.1} us ({:.1} us each)",
        batched - one_result,
        (batched - one_result) / 8.0
    );

    println!(
        "\n  a decode step of this model issues ~200 such calls; at the fixed cost above that is\n  {:.1} ms per token before any arithmetic.",
        200.0 * gpu_fixed / 1000.0
    );

    // A real batch, not a synthetic one. `qwen35moe`'s recurrent block projects
    // its input through four weights at once — a wide Q6_K q/k/v, a Q4_K output
    // gate, and two `[n_embd, 32]` slivers for beta and alpha — and after the
    // host-routing rule it is the largest thing still on the device. The
    // question this answers is whether a batch of four *unequal* ops costs what
    // its bytes say it should, or whether the two slivers are being charged a
    // dispatch each for nothing.
    println!("\n== 4. a real mixed batch: one recurrent block's projections ==");
    let wide = weight(2048, 8192); // q/k/v, 13.12 MiB at Q4_K's rate here
    let gate = weight(2048, 4096); // output gate, 4.50 MiB
    let beta = weight(2048, 32); // 0.04 MiB
    let alpha = weight(2048, 32);
    let xr = vec![0.05f32; 2048];
    fn op<'a>(x: &'a [f32], w: &'a crate::engine::loader::QuantMatrix) -> MatmulOp<'a> {
        MatmulOp { x, n_tokens: 1, w }
    }
    let bytes = mib(2048, 8192) + mib(2048, 4096) + 2.0 * mib(2048, 32);
    let all_four = median_us(reps, || {
        let _ = gpu.matmul_batch(&[
            op(&xr, &wide),
            op(&xr, &gate),
            op(&xr, &beta),
            op(&xr, &alpha),
        ]);
    });
    let big_two = median_us(reps, || {
        let _ = gpu.matmul_batch(&[op(&xr, &wide), op(&xr, &gate)]);
    });
    let slivers = median_us(reps, || {
        let _ = gpu.matmul_batch(&[op(&xr, &beta), op(&xr, &alpha)]);
    });
    let on_cpu = median_us(reps, || {
        for w in [&wide, &gate, &beta, &alpha] {
            let _ = cpu.matmul(&xr, 1, w);
        }
    });
    println!(
        "  all four, one submission  {all_four:>9.1} us  ({bytes:.2} MiB → {:.1} GiB/s)",
        bytes / 1024.0 / (all_four / 1e6)
    );
    println!("  the two large ones only   {big_two:>9.1} us");
    println!("  the two slivers only      {slivers:>9.1} us");
    println!(
        "  the slivers cost {:.1} us inside the batch — {:.1}% of it, for {:.2}% of its bytes",
        all_four - big_two,
        100.0 * (all_four - big_two) / all_four,
        100.0 * 2.0 * mib(2048, 32) / bytes,
    );
    println!("  all four on the CPU       {on_cpu:>9.1} us");

    // The router. `ffn_gate_inp` is stored **F32** — 2 MiB a layer, forty
    // layers, to produce 256 numbers that only decide which experts run, and
    // it is the largest full-precision tensor in an otherwise Q4_K model.
    // Re-quantizing a weight the file stores in full precision is an accuracy
    // change this engine does not make, so the question is not what a smaller
    // type would buy but whether the float kernel is already reading memory as
    // fast as memory can be read. The `f32` line below is that ceiling: the
    // same bytes, the same arithmetic, no widening step.
    println!("\n== 5. the router: [1 x 2048] x [2048 x 256] ==");
    const R_IN: usize = 2048;
    const R_OUT: usize = 256;
    let f32_bytes = vec![0x42u8; R_IN * R_OUT * 4];
    let w_f32 = test_quant_matrix(&f32_bytes, crate::engine::quant::GGML_TYPE_F32, R_IN, R_OUT);
    let q8_bytes = vec![0x42u8; R_IN * R_OUT / 32 * 34];
    let w_q8 = test_quant_matrix(&q8_bytes, crate::engine::quant::GGML_TYPE_Q8_0, R_IN, R_OUT);
    let xr = vec![0.05f32; R_IN];
    let weights: &[f32] = bytemuck::cast_slice(&f32_bytes);

    let f32_mib = (R_IN * R_OUT * 4) as f64 / (1024.0 * 1024.0);
    let q8_mib = (R_IN * R_OUT / 32 * 34) as f64 / (1024.0 * 1024.0);

    let as_stored = median_us(reps, || {
        let _ = cpu.matmul(&xr, 1, &w_f32);
    });
    let as_q8 = median_us(reps, || {
        let _ = cpu.matmul(&xr, 1, &w_q8);
    });
    // The ceiling: the identical multiply-adds over the identical bytes, with
    // the row already typed as `f32`. Single-threaded on purpose — what it
    // bounds is the kernel, not the fan-out.
    let ceiling = median_us(reps, || {
        let mut out = vec![0f32; R_OUT];
        for (o, slot) in out.iter_mut().enumerate() {
            let row = &weights[o * R_IN..(o + 1) * R_IN];
            let mut acc = [0f32; 8];
            // `as_chunks` rather than `chunks_exact`: both slices are exact
            // multiples of the lane count, so there is no remainder to read
            // and the typed form is what clippy asks for at a constant width.
            for (rb, xb) in row.as_chunks::<8>().0.iter().zip(xr.as_chunks::<8>().0) {
                for lane in 0..8 {
                    acc[lane] += rb[lane] * xb[lane];
                }
            }
            *slot = acc.iter().sum();
        }
        std::hint::black_box(out);
    });

    let bw = |mib: f64, us: f64| mib / 1024.0 / (us / 1e6);
    println!(
        "  F32, as the file stores it  {as_stored:>9.1} us  ({f32_mib:.2} MiB → {:.1} GiB/s)",
        bw(f32_mib, as_stored)
    );
    println!(
        "  same shape at Q8_0          {as_q8:>9.1} us  ({q8_mib:.2} MiB → {:.1} GiB/s)",
        bw(q8_mib, as_q8)
    );
    println!(
        "  f32 ceiling, 1 thread       {ceiling:>9.1} us  ({f32_mib:.2} MiB → {:.1} GiB/s)",
        bw(f32_mib, ceiling)
    );
    println!(
        "  forty layers a token: {:.2} ms as stored, {:.2} ms at the ceiling",
        40.0 * as_stored / 1000.0,
        40.0 * ceiling / 1000.0
    );
}

/// **Does the device's bandwidth actually win at the top end?**
/// [`host_matmul_threshold_bytes`](crate::engine::backend::host_matmul_threshold_bytes)
/// assumes it does — that host and device costs are two lines which cross
/// once, so above the crossing the device is the right place forever. That
/// holds only if the device reads a weight faster than the host does, and
/// whether it *does* turns out to depend on the format the weight is stored
/// in.
///
/// So: one shape, every format a real file uses, both engines. The shape is
/// the vocabulary projection of Llama-3.2-1B (`2048 -> 128256`), which is the
/// largest single matmul in a decode step and the one with the most to lose.
///
/// `cargo test --release --bin orangu-server decode_matvec_format -- --ignored --nocapture`
/// The formats [`decode_matvec_format_sweep_gpu_versus_cpu`] compares, named
/// once so the measuring loop and the reporting loop cannot drift apart.
///
/// `F16`, `Q4_K` and `Q6_K` are controls: the first has no block structure
/// to get wrong, the other two have their own tuned kernels and never reach
/// `block_hoisted_suffix`. A change to the block-hoisted family should move
/// the middle of this list and leave the ends alone.
const SWEEP_FORMATS: &[(&str, u32)] = &[
    // `F32` is not a format a weight is *stored* in for size, but real files
    // carry it for small per-layer tensors — gemma 4's per-layer-embedding
    // gate and projection are F32, 3 MiB a layer — and a decode step reads
    // every one of them. Its rate is the one that says what those cost.
    ("F32", crate::engine::quant::GGML_TYPE_F32),
    ("F16", crate::engine::quant::GGML_TYPE_F16),
    ("Q4_0", crate::engine::quant::GGML_TYPE_Q4_0),
    ("Q4_1", crate::engine::quant::GGML_TYPE_Q4_1),
    ("Q5_0", crate::engine::quant::GGML_TYPE_Q5_0),
    ("Q5_1", crate::engine::quant::GGML_TYPE_Q5_1),
    ("Q8_0", crate::engine::quant::GGML_TYPE_Q8_0),
    ("Q2_K", crate::engine::quant::GGML_TYPE_Q2_K),
    ("Q3_K", crate::engine::quant::GGML_TYPE_Q3_K),
    ("IQ4_XS", crate::engine::quant::GGML_TYPE_IQ4_XS),
    ("IQ3_S", crate::engine::quant::GGML_TYPE_IQ3_S),
    ("IQ2_S", crate::engine::quant::GGML_TYPE_IQ2_S),
    ("PQ2_0", crate::engine::quant::GGML_TYPE_PQ2_0),
    ("PTQ1_0", crate::engine::quant::GGML_TYPE_PTQ1_0),
    ("Q6_K", crate::engine::quant::GGML_TYPE_Q6_K),
    ("Q4_K", GGML_TYPE_Q4_K),
];

#[test]
#[ignore]
fn decode_matvec_format_sweep_gpu_versus_cpu() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(gpu) = shared_test_backend() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    let cpu = CpuBackend;
    // Twenty-one, not five. Capping the shape (above) put a single
    // measurement at a few milliseconds, and five of those leave a median
    // that still swings by 3x run to run — `F16` read 4.2 and 13.8 GB/s in
    // consecutive runs of the *same* binary, which is wider than any kernel
    // difference worth reporting. The whole sweep is ~5 s at five reps, so
    // there is room to spend here, and the invariant formats are how the
    // noise floor is checked: if they disagree between arms, the comparison
    // is void.
    let reps = 21;
    // `32768`, not a real vocabulary's `128256`. Eight formats at the full
    // width is 1.77 GiB of device memory across the run, and this board
    // charges ~0.35 s per MiB of *cumulative* allocation at process exit
    // (`teardown_probe_holds_gpu_memory`) — the full-width version of this
    // test finished its work in 13 s and then sat in `Z` state for 620 s,
    // holding a Vulkan context that slowed the next run to a crawl. A
    // quarter of the width is ~440 MiB for the whole sweep, under the point
    // where the stall begins, and every weight here is still far larger than
    // any cache, which is all a bandwidth measurement needs.
    // `ORANGU_SWEEP_SHAPE=in,out` puts the sweep at another shape — the one
    // the reference engine's own op benchmark uses (`14336,4096`), so the two
    // kernels can be read at the same bytes, or a model's FFN (`1536,6144`).
    let (in_dim, out_dim) = std::env::var("ORANGU_SWEEP_SHAPE")
        .ok()
        .and_then(|v| {
            let (a, b) = v.split_once(',')?;
            Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
        })
        .unwrap_or((2048usize, 32768usize));
    let x = vec![0.05f32; in_dim];

    println!("\n== one shape [1 x {in_dim}] x [{in_dim} x {out_dim}], every format ==");
    // `kernel us` is the dispatch alone, from the device's own timestamps
    // around a run of back-to-back dispatches (`matmul_kernel_us`); `GPU us`
    // is the whole call as the host sees it. At a decode shape the two differ
    // by the call's fixed cost, which is what a kernel change must not be
    // read through.
    println!(
        "{:>8} {:>9} {:>11} {:>11} {:>9} {:>11} {:>11} {:>9}",
        "type", "MiB", "GPU us", "CPU us", "GPU GB/s", "CPU GB/s", "kernel us", "kern GB/s"
    );
    // Two passes over the whole list, each format keeping its better pass.
    // Within a format `median_us` already defends against a single hiccup,
    // but the formats are measured in sequence and the board drifts *across*
    // that sequence — whoever runs last is charged for the heat of everyone
    // before it. One pass had `Q4_K` at half its own rate purely for being
    // at the end of the list, which reads exactly like a kernel regression
    // and is not one.
    // `ORANGU_SWEEP_ONLY=Q4_0` measures one format and nothing else.
    //
    // Comparing two builds of a *kernel* through the full list does not work
    // on this board: the unchanged formats, which ought to be identical
    // between the two, swing by 0.65-1.63x between runs, because each
    // measurement is now a couple of milliseconds and whatever the GPU is
    // doing across a run moves more than any kernel change does. One format
    // per process removes that entirely — every run measures the same thing
    // from the same starting state.
    let only = std::env::var("ORANGU_SWEEP_ONLY").ok();
    // Dispatches per timed pass for the kernel column. A short burst is read
    // at whatever clock the device idles at — at the default's old value
    // (the 21 of `reps`) a 1536-row dispatch read 45–110 µs whatever its
    // width, the card never leaving its idle clock, and the same dispatches
    // read 34–60 at 64 — so the burst is long enough to be read at the
    // clock a decode runs at. `ORANGU_SWEEP_KERNEL_REPS` overrides it.
    let kernel_reps: u32 = std::env::var("ORANGU_SWEEP_KERNEL_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let mut best: std::collections::HashMap<&str, (f64, f64, f64)> =
        std::collections::HashMap::new();
    for _ in 0..2 {
        for &(label, ggml_type) in SWEEP_FORMATS {
            if only.as_deref().is_some_and(|want| want != label) {
                continue;
            }
            let Some((bytes_per_block, block)) = crate::engine::quant::block_layout(ggml_type)
            else {
                continue;
            };
            let bytes = in_dim * out_dim / block * bytes_per_block;
            let w = crate::engine::loader::test_quant_matrix(
                &vec![0x42u8; bytes],
                ggml_type,
                in_dim,
                out_dim,
            );
            let gpu_us = median_us(reps, || {
                let _ = gpu.matmul(&x, 1, &w);
            });
            let cpu_us = median_us(reps, || {
                let _ = cpu.matmul(&x, 1, &w);
            });
            // The median of `reps` runs of `reps` dispatches: one warm-up
            // dispatch already happened in `gpu.matmul` above, so every
            // dispatch here reads a resident weight.
            let kernel_us = {
                let mut runs: Vec<f64> = (0..reps)
                    .filter_map(|_| gpu.matmul_kernel_us(&x, &w, kernel_reps))
                    .collect();
                runs.sort_by(|a, b| a.total_cmp(b));
                runs.get(runs.len() / 2).copied().unwrap_or(f64::NAN)
            };
            let seen = best.entry(label).or_insert((f64::MAX, f64::MAX, f64::MAX));
            seen.0 = seen.0.min(gpu_us);
            seen.1 = seen.1.min(cpu_us);
            seen.2 = seen.2.min(kernel_us);
        }
    }
    for &(label, ggml_type) in SWEEP_FORMATS {
        let Some((bytes_per_block, block)) = crate::engine::quant::block_layout(ggml_type) else {
            continue;
        };
        let Some(&(gpu_us, cpu_us, kernel_us)) = best.get(label) else {
            continue;
        };
        let bytes = in_dim * out_dim / block * bytes_per_block;
        let gbs = |us: f64| bytes as f64 / (us * 1e3);
        println!(
            "{label:>8} {:>9.1} {gpu_us:>11.0} {cpu_us:>11.0} {:>9.1} {:>11.1} {kernel_us:>11.1} {:>9.1}",
            bytes as f64 / (1024.0 * 1024.0),
            gbs(gpu_us),
            gbs(cpu_us),
            gbs(kernel_us)
        );
    }
}

/// **How long does this device take to release GPU memory?** Allocates
/// `ORANGU_TD_MIB` megabytes of weights, touches them once, and exits.
///
/// Not a benchmark — a controlled input for measuring *teardown*. A process
/// that has allocated GPU memory on this board finishes its work, exits, and
/// then sits in `Z` state for minutes while a Mali kernel thread
/// (`mali-mem-purge`, `mali-event-hand`, both blocked at `blk_mq_get_tag`)
/// releases it. It is not reachable from here: the process has already run
/// to completion and the memory goes back when the kernel closes the
/// `/dev/mali` fd, so no shutdown path of ours executes. What this probe is
/// for is knowing *when* it happens, which turns out to be a sharp line.
///
/// **One allocation, varying its size** (`ORANGU_TD_MIB`):
///
/// | held | teardown |
/// | --: | --: |
/// | 64 MiB | 0 s |
/// | 256 MiB | 0 s |
/// | 384 MiB | 0 s |
/// | 512 MiB | 0 s |
/// | 768 MiB | minutes |
/// | 1024 MiB | 364 s |
///
/// **Peak fixed, total varied** (`ORANGU_TD_REPS`), which is the experiment
/// that matters, because the first table alone would have produced the wrong
/// rule: four 256 MiB allocations with the cache released between them —
/// 256 MiB ever live, 1 GiB asked for — takes **449 s**, like the single
/// gigabyte and unlike the single 256 MiB.
///
/// So the cost tracks the memory a process has **ever asked the driver for**,
/// not what it holds at exit. That is why
/// `decode_matvec_format_sweep_gpu_versus_cpu` stalls for 620 s while
/// peaking at 501 MiB: it allocates 1.9 GiB across its run. And it is why
/// `VulkanBackend::release_weight_cache` does not help — `wgpu`
/// suballocates, so releasing returns memory to `wgpu`, never to the driver.
///
/// **The practical rule for a GPU test: keep the total under half a
/// gigabyte, or accept that the process will hang for minutes after it
/// passes** — and note that a test binary is one process, so it is the whole
/// suite's total that counts, not one test's.
///
/// Two other explanations were measured and rejected: freeing the buffers
/// during the run (above), and a 3.1 GB dirty-page backlog on the
/// USB-attached root disk — removing it entirely, to `Dirty: 184 kB` and
/// zero I/O pressure, left teardown at 620 s.
///
/// `ORANGU_TD_MIB=512 cargo test --release --bin orangu-server teardown_probe -- --ignored`
#[test]
#[ignore]
fn teardown_probe_holds_gpu_memory() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(gpu) = shared_test_backend() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    let mib: usize = std::env::var("ORANGU_TD_MIB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);
    // Q4_K rows of a fixed width, as many as it takes to reach `mib`.
    let in_dim = 4096usize;
    let row_bytes = in_dim / Q4_K_BLOCK * Q4_K_BYTES;
    let out_dim = (mib * 1024 * 1024).div_ceil(row_bytes);
    // `ORANGU_TD_REPS` distinct weights of that size, one after another,
    // with the cache released between them. Peak residency stays at one
    // weight while the *total* the process has asked the driver for grows —
    // which is the question a single allocation cannot answer: 512 MiB in
    // one piece tears down instantly, yet the format sweep peaks at 501 MiB
    // and hangs for 620 s after allocating 1.9 GiB across its run.
    let reps: usize = std::env::var("ORANGU_TD_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let x = vec![0.05f32; in_dim];
    let at = std::time::Instant::now();
    let mut first = 0.0f32;
    for rep in 0..reps {
        // A different byte per rep, so `weight_buffer`'s cache key differs
        // and each one is a fresh device allocation rather than a hit.
        let bytes = vec![0x42u8.wrapping_add(rep as u8); out_dim * row_bytes];
        let w = crate::engine::loader::test_quant_matrix(&bytes, GGML_TYPE_Q4_K, in_dim, out_dim);
        let y = gpu.matmul(&x, 1, &w);
        first = y.first().copied().unwrap_or(0.0);
        if reps > 1 {
            gpu.release_weight_cache();
        }
    }
    println!(
        "held {:.0} MiB x {reps} ({:.0} MiB total) in {:.0} ms, first output {first:.3}",
        (out_dim * row_bytes) as f64 / (1024.0 * 1024.0),
        (out_dim * row_bytes * reps) as f64 / (1024.0 * 1024.0),
        at.elapsed().as_secs_f64() * 1e3,
    );
}

/// **What does one whole-row norm cost as a decode step pays it?** A layer
/// runs several single-workgroup RMSNorm-family dispatches over the model
/// width, each a barrier-bounded dispatch of its own; this reads each at the
/// device's clock, back to back, so a rewrite of the kernel can be judged
/// against the floor a trivial dispatch already costs.
///
/// `ORANGU_NORM_PROBE_WIDTH=1536` picks the row width.
///
/// `cargo test --release --bin orangu-server norm_kernel_probe -- --ignored --nocapture`
#[test]
#[ignore]
fn norm_kernel_probe() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(gpu) = shared_test_backend() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    let n_embd: usize = std::env::var("ORANGU_NORM_PROBE_WIDTH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1536);
    // `ORANGU_NORM_PROBE_REPS` for a device that loses itself under two
    // thousand passes in one submission (`ORANGU_PROBE_PASS_PER_DISPATCH=1`
    // on the Mali-G720: 200 is fine).
    let reps: u32 = std::env::var("ORANGU_NORM_PROBE_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2000);
    println!("\n== whole-row norm dispatches at width {n_embd}, {reps} back to back ==");
    for which in ["norm", "norm_add"] {
        let mut runs: Vec<f64> = (0..21)
            .filter_map(|_| gpu.norm_kernel_us(which, n_embd, reps))
            .collect();
        runs.sort_by(|a, b| a.total_cmp(b));
        match runs.get(runs.len() / 2) {
            Some(us) => println!("{which:>10} {us:>8.1} us"),
            None => println!("{which:>10} (no timestamp query)"),
        }
    }
}

/// **What does one `queue.submit` cost the host?** Times an empty
/// submission, then one carrying a single trivial dispatch — the floor a
/// decode step pays per submission before any of its own commands.
///
/// `cargo test --release --bin orangu-server submit_cost_probe -- --ignored --nocapture`
#[test]
#[ignore]
fn submit_cost_probe() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(gpu) = shared_test_backend() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    // Warm: the first submissions pay one-time setup.
    for _ in 0..5 {
        let encoder = gpu.new_encoder("warm");
        gpu.queue.submit(Some(encoder.finish()));
    }
    gpu.poll_blocking("submit probe warm-up");
    let reps = 200;
    let t = std::time::Instant::now();
    for _ in 0..reps {
        let encoder = gpu.new_encoder("empty");
        gpu.queue.submit(Some(encoder.finish()));
    }
    let empty_us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
    gpu.poll_blocking("submit probe");
    let t = std::time::Instant::now();
    for _ in 0..reps {
        let mut encoder = gpu.new_encoder("finish only");
        {
            let _pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
        }
        let _ = encoder.finish();
    }
    let finish_us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
    println!("\n== queue.submit, host time per call ==");
    println!("{:>28} {empty_us:>8.1} us", "empty submission");
    println!("{:>28} {finish_us:>8.1} us", "record+finish, no submit");
}

/// The prefill GEMM on the device clock, at the served model's FFN shapes
/// and the chunk widths a prefill dispatches — `dispatch_kernel_us` around a
/// burst of the kernel `pipeline_for_named` picks for that width, so what is
/// printed is the kernel and nothing else: no upload, no submission, no
/// readback. `GFLOP/s` counts two per multiply-add. `ORANGU_SWEEP_SHAPE=in,out`
/// moves the shape; `ORANGU_SWEEP_TOKENS=64,128,...` the widths.
///
/// `cargo test --release --bin orangu-server prefill_gemm_probe -- --ignored --nocapture`
#[test]
#[ignore]
fn prefill_gemm_probe() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(gpu) = shared_test_backend() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    let shapes: Vec<(usize, usize)> = match std::env::var("ORANGU_SWEEP_SHAPE") {
        Ok(v) => {
            let (a, b) = v.split_once(',').expect("ORANGU_SWEEP_SHAPE=in,out");
            vec![(a.trim().parse().unwrap(), b.trim().parse().unwrap())]
        }
        Err(_) => vec![(1536, 6144), (6144, 1536), (2048, 1536), (1536, 2048)],
    };
    let widths: Vec<usize> = std::env::var("ORANGU_SWEEP_TOKENS")
        .ok()
        .map(|v| v.split(',').map(|t| t.trim().parse().unwrap()).collect())
        .unwrap_or_else(|| vec![32, 64, 128, 256, 512]);
    let reps: u32 = std::env::var("ORANGU_SWEEP_KERNEL_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    // `ORANGU_SWEEP_ROTATE=N`: the integer-dot kernel over N distinct weight
    // matrices in turn, one per repetition — a layer loop's cache
    // behaviour rather than one matrix's.
    let rotate: usize = std::env::var("ORANGU_SWEEP_ROTATE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1)
        .max(1);
    for &(in_dim, out_dim) in &shapes {
        // `ORANGU_SWEEP_TYPES=Q4_K,Q8_0,...` picks the weight types; the
        // default is the pair every K-quant file's FFN is made of plus the
        // float baseline.
        let types: Vec<u32> = match std::env::var("ORANGU_SWEEP_TYPES") {
            Ok(list) => list
                .split(',')
                .map(|t| quant_type_named(t.trim()).unwrap_or_else(|| panic!("unknown type {t}")))
                .collect(),
            Err(_) => vec![
                crate::engine::quant::GGML_TYPE_Q4_K,
                crate::engine::quant::GGML_TYPE_Q6_K,
                crate::engine::quant::GGML_TYPE_F32,
            ],
        };
        for &ggml_type in &types {
            let w = weight_of(ggml_type, in_dim, out_dim);
            println!(
                "\n== [{{n}} x {in_dim}] x [{in_dim} x {out_dim}] {} ==",
                quant_name(ggml_type)
            );
            println!(
                "{:>6} {:>22} {:>11} {:>10} {:>10}",
                "tokens", "kernel", "kernel us", "GFLOP/s", "us/token"
            );
            // `ORANGU_SWEEP_SKIP_FLOAT=1`: the integer-dot kernel alone,
            // without the float baseline at the same shape. The baseline is
            // the slower kernel by several times, and at a wide shape
            // (`12288 × 1536` at 512 tokens) a burst of it overruns the
            // device's watchdog and takes the whole device down with it —
            // so a sweep that only wants the integer-dot kernels has to be
            // able to say so.
            let skip_float = crate::engine::env::flag_on("ORANGU_SWEEP_SKIP_FLOAT");
            for &n in &widths {
                let x = vec![0.05f32; in_dim * n];
                let flops = 2.0 * (n * in_dim * out_dim) as f64;
                let baseline = if skip_float {
                    None
                } else {
                    match gpu.matmul_kernel_us_tokens(&x, n, &w, reps) {
                        Some(v) => Some(v),
                        None => {
                            println!("  (no timestamp query on this adapter)");
                            return;
                        }
                    }
                };
                if let Some((us, name)) = baseline {
                    // `ORANGU_SWEEP_GAP_MS=<ms>`: the same dispatch, one per
                    // submission, with the device left idle that long between
                    // them — a prefill's own rhythm, where every chain waits for
                    // a readback before the next is recorded. The difference
                    // between this column and the burst is what idling costs the
                    // clock.
                    let gapped = std::env::var("ORANGU_SWEEP_GAP_MS")
                        .ok()
                        .and_then(|v| v.parse::<u64>().ok())
                        .map(|gap| {
                            let mut samples: Vec<f64> = (0..9)
                                .filter_map(|_| {
                                    std::thread::sleep(std::time::Duration::from_millis(gap));
                                    gpu.matmul_kernel_us_tokens(&x, n, &w, 1).map(|(us, _)| us)
                                })
                                .collect();
                            samples.sort_by(|a, b| a.total_cmp(b));
                            samples[samples.len() / 2]
                        });
                    println!(
                        "{n:>6} {name:>22} {us:>11.1} {:>10.0} {:>10.2}{}",
                        flops / us / 1e3,
                        us / n as f64,
                        gapped.map_or(String::new(), |g| format!(
                            "   gapped {g:>9.1} us ({:.0} GFLOP/s)",
                            flops / g / 1e3
                        ))
                    );
                }
                // The integer-dot GEMM at the same shape, where it applies.
                let rotation: Vec<_> = (0..rotate)
                    .map(|i| weight_of_seeded(ggml_type, in_dim, out_dim, 0x42 + i as u8))
                    .collect();
                if let Some((us, name)) =
                    gpu.mmq_kernel_us_tokens_rotating(&x, n, &rotation, reps.max(rotate as u32))
                {
                    println!(
                        "{n:>6} {name:>22} {us:>11.1} {:>10.0} {:>10.2}{}",
                        flops / us / 1e3,
                        us / n as f64,
                        if rotate > 1 {
                            format!("   (over {rotate} matrices)")
                        } else {
                            String::new()
                        }
                    );
                }
            }
        }
    }
}

fn weight_of(ggml_type: u32, in_dim: usize, out_dim: usize) -> crate::engine::loader::QuantMatrix {
    weight_of_seeded(ggml_type, in_dim, out_dim, 0x42)
}

fn weight_of_seeded(
    ggml_type: u32,
    in_dim: usize,
    out_dim: usize,
    fill: u8,
) -> crate::engine::loader::QuantMatrix {
    let (bytes_per_block, block) =
        crate::engine::quant::block_layout(ggml_type).expect("a probed format has a block layout");
    let bytes = in_dim * out_dim / block * bytes_per_block;
    test_quant_matrix(&vec![fill; bytes], ggml_type, in_dim, out_dim)
}

const PROBE_TYPES: &[(&str, u32)] = &[
    ("Q4_K", crate::engine::quant::GGML_TYPE_Q4_K),
    ("Q5_K", crate::engine::quant::GGML_TYPE_Q5_K),
    ("Q6_K", crate::engine::quant::GGML_TYPE_Q6_K),
    ("Q8_0", crate::engine::quant::GGML_TYPE_Q8_0),
    ("Q4_0", crate::engine::quant::GGML_TYPE_Q4_0),
    ("Q4_1", crate::engine::quant::GGML_TYPE_Q4_1),
    ("Q5_0", crate::engine::quant::GGML_TYPE_Q5_0),
    ("Q5_1", crate::engine::quant::GGML_TYPE_Q5_1),
    ("Q2_K", crate::engine::quant::GGML_TYPE_Q2_K),
    ("Q3_K", crate::engine::quant::GGML_TYPE_Q3_K),
    ("IQ4_XS", crate::engine::quant::GGML_TYPE_IQ4_XS),
    ("IQ3_S", crate::engine::quant::GGML_TYPE_IQ3_S),
    ("IQ2_S", crate::engine::quant::GGML_TYPE_IQ2_S),
    ("F32", crate::engine::quant::GGML_TYPE_F32),
];

fn quant_name(ggml_type: u32) -> &'static str {
    PROBE_TYPES
        .iter()
        .find(|(_, t)| *t == ggml_type)
        .map_or("?", |(n, _)| n)
}

fn quant_type_named(name: &str) -> Option<u32> {
    PROBE_TYPES
        .iter()
        .find(|(n, _)| n.eq_ignore_ascii_case(name))
        .map(|(_, t)| *t)
}

/// The instruction rates the GEMM kernels are built on, measured in
/// isolation: a register-only loop of `f32` fused multiply-adds against the
/// same loop of packed 8-bit dots, each 4096 deep per thread, over enough
/// workgroups to fill the device. Reports the device's achieved rate for
/// each and their ratio — what an integer-dot kernel can hope for over a
/// float one on this card, before memory enters.
///
/// `cargo test --release --bin orangu-server alu_rate_probe -- --ignored --nocapture`
#[test]
#[ignore]
fn alu_rate_probe() {
    let _gpu_lock = super::gpu_test_lock();
    let Some(gpu) = shared_test_backend() else {
        eprintln!("{NO_GPU_SKIP}");
        return;
    };
    let body = |op: &str| -> String {
        format!(
            r#"
struct ElemMeta {{ len: u32, aux: u32, extra: f32, out_scale: f32 }}
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> em: ElemMeta;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    var a0: f32 = x[lid.x]; var a1: f32 = x[lid.x + 1u]; var a2: f32 = x[lid.x + 2u]; var a3: f32 = x[lid.x + 3u];
    var i0: i32 = i32(gid.x); var i1: i32 = i0 + 1; var i2: i32 = i0 + 2; var i3: i32 = i0 + 3;
    let w: u32 = gid.x * 0x01010101u + 0x03020100u;
    let v: u32 = gid.x ^ 0x7F3F1F0Fu;
    let m: f32 = x[4u] * 0.999;
    var k: u32 = 0u;
    loop {{
        if (k >= 1024u) {{ break; }}
        {op}
        k = k + 1u;
    }}
    y[gid.x] = a0 + a1 + a2 + a3 + f32(i0 + i1 + i2 + i3);
}}
"#
        )
    };
    let fma = body(
        "a0 = fma(a0, m, 1.0); a1 = fma(a1, m, 1.0); a2 = fma(a2, m, 1.0); a3 = fma(a3, m, 1.0);",
    );
    let dot = body(
        "i0 = dot4I8Packed(w, v ^ u32(i0)) + i0; i1 = dot4I8Packed(w, v ^ u32(i1)) + i1; i2 = dot4I8Packed(w, v ^ u32(i2)) + i2; i3 = dot4I8Packed(w, v ^ u32(i3)) + i3;",
    );
    let workgroups = 22 * 16;
    let threads = (workgroups * 256) as f64;
    let ops = threads * 4096.0;
    let reps = 10;
    let Some(fma_us) = gpu.adhoc_kernel_us(fma, workgroups, reps) else {
        println!("  (no timestamp query on this adapter)");
        return;
    };
    let dot_us = gpu.adhoc_kernel_us(dot, workgroups, reps).unwrap();
    println!(
        "\n  f32 fma:  {fma_us:>9.1} us  {:>8.0} GFLOP/s (2 per fma)",
        2.0 * ops / fma_us / 1e3
    );
    println!(
        "  int8 dot4: {dot_us:>8.1} us  {:>8.0} GMAC/s (4 per dot), {:.2}x the fma rate per instruction",
        4.0 * ops / dot_us / 1e3,
        fma_us / dot_us
    );
}
