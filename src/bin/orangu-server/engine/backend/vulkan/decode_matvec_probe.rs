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
