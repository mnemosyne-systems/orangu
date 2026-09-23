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

//! **Is the `rten-gemm` crate's host matmul faster than this engine's?**
//!
//! A packed, cache-blocked GEMM is a different algorithm from the row dots
//! this engine runs: it copies both operands into tiles sized for the cache
//! before multiplying, which costs a pass over the data and buys locality.
//! Which one wins depends on the shape and on how the weight is stored, so
//! the question is not answerable in general — only at the shapes a forward
//! pass actually runs.
//!
//! Two comparisons, because the engine's weights are quantized and the
//! crate's GEMM is not:
//!
//! 1. **`F32` weights** — the honest like-for-like. Both read the same
//!    numbers in the same layout; only the multiply differs.
//! 2. **`Q4_K` weights** — the engine multiplies the quantized bytes
//!    directly, so a packed GEMM has to dequantize first. The dequantize is
//!    reported as its own column, because a caller who did this once at load
//!    would pay it once and one who did it per call would pay it every time,
//!    and those are different answers.
//!
//! `cargo test --release --bin orangu-server rten_gemm -- --ignored --nocapture`

#![cfg(test)]

use std::time::Instant;

use rten_gemm::{GemmExecutor, GemmInputA, GemmInputB, GemmOptions};
use rten_tensor::NdTensorView;

use crate::engine::backend::{Backend, CpuBackend};
use crate::engine::loader::{QuantMatrix, test_quant_matrix};
use crate::engine::quant::{GGML_TYPE_F32, GGML_TYPE_Q4_K};

/// One shape a forward pass actually runs, named for where it comes from.
struct Shape {
    what: &'static str,
    /// Rows of activation — tokens in the batch.
    m: usize,
    /// The contraction — `n_embd`, or the feed-forward width.
    k: usize,
    /// Outputs per token.
    n: usize,
}

/// Shapes taken from the mixtures this engine serves: an expert projection
/// at a narrow and at a wide prefill chunk, a dense feed-forward branch, an
/// attention output projection, and one decode step's mat-vec — the shape
/// with no arithmetic to amortize anything over.
const SHAPES: &[Shape] = &[
    Shape {
        what: "expert proj, narrow chunk",
        m: 256,
        k: 2048,
        n: 768,
    },
    Shape {
        what: "expert proj, wide chunk",
        m: 2048,
        k: 2048,
        n: 768,
    },
    Shape {
        what: "dense ffn gate, prefill",
        m: 512,
        k: 2048,
        n: 6144,
    },
    Shape {
        what: "attention output, prefill",
        m: 512,
        k: 4096,
        n: 2048,
    },
    Shape {
        what: "decode mat-vec",
        m: 1,
        k: 2048,
        n: 2048,
    },
    // The mixture router: `F32` by choice, one call per layer per chunk,
    // and the widest `F32` matmul a mixture prefill runs.
    Shape {
        what: "moe router, wide chunk",
        m: 2048,
        k: 2048,
        n: 128,
    },
    // The output-width sweep. A router is the narrowest product a forward
    // pass runs, and narrow output is where this engine's row grouping has
    // the fewest groups to spread across the machine — so if the two
    // kernels cross anywhere, they cross here. Where they cross is what
    // decides which shapes are worth handing over.
    Shape {
        what: "narrow out, n=32",
        m: 2048,
        k: 2048,
        n: 32,
    },
    Shape {
        what: "narrow out, n=64",
        m: 2048,
        k: 2048,
        n: 64,
    },
    Shape {
        what: "narrow out, n=256",
        m: 2048,
        k: 2048,
        n: 256,
    },
    Shape {
        what: "narrow out, n=512",
        m: 2048,
        k: 2048,
        n: 512,
    },
];

/// Repetitions per point. Enough that the median is not one scheduling
/// accident, few enough that the whole table is seconds.
const REPS: usize = 7;

/// Bytes of a `Q4_K` row: 256 weights to 144 bytes.
fn q4k_bytes(in_dim: usize, out_dim: usize) -> usize {
    in_dim.div_ceil(256) * 144 * out_dim
}

/// A weight matrix of `out_dim` rows, filled with a fixed pattern — the
/// values do not matter to a timing, only that they are not degenerate.
fn weights(ggml_type: u32, in_dim: usize, out_dim: usize) -> QuantMatrix {
    let len = match ggml_type {
        GGML_TYPE_F32 => in_dim * out_dim * 4,
        _ => q4k_bytes(in_dim, out_dim),
    };
    // **Normal floats, not a byte pattern.** Bytes reinterpreted as `f32`
    // are overwhelmingly denormal, and denormal arithmetic on x86 runs at a
    // fraction of the normal rate — which would be measured as a property
    // of whichever kernel is being timed. The quantized types cannot have
    // this problem (their values come back through a scale), so a byte
    // pattern there is harmless and only the float case is built by value.
    let bytes: Vec<u8> = if ggml_type == GGML_TYPE_F32 {
        (0..in_dim * out_dim)
            .flat_map(|i| (((i % 17) as f32 - 8.0) * 0.125).to_le_bytes())
            .collect()
    } else {
        (0..len).map(|i| (i % 251) as u8).collect()
    };
    test_quant_matrix(&bytes, ggml_type, in_dim, out_dim)
}

/// The same weight as `[k, n]` floats — what a packed GEMM needs, and the
/// transpose of how this engine stores it (one row per output).
fn transposed_f32(w: &QuantMatrix, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; k * n];
    for col in 0..n {
        for (i, &v) in w.row(col).iter().enumerate() {
            out[i * n + col] = v;
        }
    }
    out
}

/// Median of `REPS` runs of `f`, in milliseconds.
fn median_ms(mut f: impl FnMut()) -> f64 {
    // One untimed run so neither side is charged for its first-touch page
    // faults or a cold weight.
    f();
    let mut times: Vec<f64> = (0..REPS)
        .map(|_| {
            let at = Instant::now();
            f();
            at.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

fn gflops(shape: &Shape, ms: f64) -> f64 {
    2.0 * shape.m as f64 * shape.k as f64 * shape.n as f64 / (ms * 1e6)
}

#[test]
#[ignore]
fn rten_gemm_against_this_engines_host_matmul() {
    let cpu = CpuBackend;
    let gemm = GemmExecutor::<f32, f32, f32>::default();
    println!("rten-gemm kernel: {}", gemm.kernel_name());
    println!(
        "{:<28} {:>5} {:>5} {:>6}  {:>10} {:>10} {:>10} {:>8}",
        "shape", "m", "k", "n", "ours ms", "rten ms", "dequant ms", "rten x"
    );
    for ggml_type in [GGML_TYPE_F32, GGML_TYPE_Q4_K] {
        let name = if ggml_type == GGML_TYPE_F32 {
            "F32"
        } else {
            "Q4_K"
        };
        println!("--- {name} weights");
        for shape in SHAPES {
            let w = weights(ggml_type, shape.k, shape.n);
            let x: Vec<f32> = (0..shape.m * shape.k)
                .map(|i| (i % 17) as f32 * 0.01 - 0.08)
                .collect();
            // `matmul_into`, not `matmul`: the allocating form would
            // charge our side for a fresh `[m, n]` buffer every call —
            // 12 MiB on the widest shape here — where the crate writes
            // into one the caller already owns. Both sides now reuse.
            let mut ours_out: Vec<f32> = Vec::new();
            let ours = median_ms(|| {
                cpu.matmul_into(&mut ours_out, &x, shape.m, &w);
                std::hint::black_box(&ours_out);
            });
            let bt = transposed_f32(&w, shape.k, shape.n);
            let mut out = vec![0f32; shape.m * shape.n];
            let theirs = median_ms(|| {
                let a = NdTensorView::from_data([shape.m, shape.k], x.as_slice());
                let b = NdTensorView::from_data([shape.k, shape.n], bt.as_slice());
                gemm.gemm(
                    &mut out,
                    GemmInputA::Unpacked(a),
                    GemmInputB::Unpacked(b),
                    GemmOptions::default(),
                )
                .expect("shapes agree");
                std::hint::black_box(&out);
            });
            // **On this engine's own layout, with no copy.** A weight is
            // stored as `[out_dim, in_dim]` — one row per output — which is
            // the transpose of the `[k, n]` a GEMM wants. If the crate's
            // packing reads a strided view as fast as a contiguous one,
            // adopting it costs nothing but the call; if it does not, it
            // costs a transpose of the whole weight.
            if ggml_type == GGML_TYPE_F32 {
                let raw = w.raw_bytes();
                // Safety: `f32` has no invalid bit patterns and only a fully
                // aligned run is taken, as `matmul_float_into` does.
                let (head, floats, _) = unsafe { raw.align_to::<f32>() };
                if head.is_empty() && floats.len() >= shape.k * shape.n {
                    let view =
                        NdTensorView::from_data([shape.n, shape.k], &floats[..shape.k * shape.n]);
                    let transposed = view.transposed();
                    let strided = median_ms(|| {
                        gemm.gemm(
                            &mut out,
                            GemmInputA::Unpacked(NdTensorView::from_data(
                                [shape.m, shape.k],
                                x.as_slice(),
                            )),
                            GemmInputB::Unpacked(transposed),
                            GemmOptions::default(),
                        )
                        .expect("shapes agree");
                        std::hint::black_box(&out);
                    });
                    println!(
                        "{:<28} {:>5} {:>5} {:>6}  {:>10} {:>10.3} {:>10} {:>8.2}  (no-copy transposed view, {:.0} GFLOP/s)",
                        "  .. our layout, no copy",
                        shape.m,
                        shape.k,
                        shape.n,
                        "",
                        strided,
                        "",
                        ours / strided,
                        gflops(shape, strided),
                    );
                }
            }
            // What a caller would add if it dequantized per call rather than
            // once at load. Zero for `F32`, where nothing is unpacked.
            let dequant = if ggml_type == GGML_TYPE_F32 {
                0.0
            } else {
                median_ms(|| {
                    std::hint::black_box(transposed_f32(&w, shape.k, shape.n));
                })
            };
            println!(
                "{:<28} {:>5} {:>5} {:>6}  {:>10.3} {:>10.3} {:>10.3} {:>8.2}  ({:.0} vs {:.0} GFLOP/s)",
                shape.what,
                shape.m,
                shape.k,
                shape.n,
                ours,
                theirs,
                dequant,
                ours / theirs,
                gflops(shape, ours),
                gflops(shape, theirs),
            );
        }
    }
}
