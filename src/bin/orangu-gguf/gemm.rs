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

//! The matrix products through `rten_gemm`: `C[m, n] (+)= A[m, p] · B[p, n]`,
//! either operand read through any pair of strides, so the training
//! kernels' forward (`x · wᵀ`), input gradient (`dy · w`) and weight
//! gradient (`dyᵀ · x`) are each one call. `rten_gemm` picks the widest
//! vector kernel the processor has (AVX-512, AVX2 or NEON).

/// A matrix read through two strides: element `(i, j)` is at
/// `data[i * row + j * col]`.
#[derive(Clone, Copy)]
pub struct View<'a> {
    pub data: &'a [f32],
    pub row: usize,
    pub col: usize,
}

/// `c[m, n]` (row stride `n`) `+= a · b` when `add`, `= a · b` otherwise,
/// through `rten_gemm`, which picks the widest vector kernel the processor
/// has. Its depth blocking depends on the shape alone, so a sum's order
/// does not depend on the thread count.
#[allow(clippy::too_many_arguments)]
pub fn library(c: &mut [f32], a: View<'_>, b: View<'_>, m: usize, n: usize, p: usize, add: bool) {
    use rten_gemm::{GemmInputA, GemmInputB, GemmOptions};
    use rten_tensor::NdTensorView;
    thread_local! {
        static GEMM: rten_gemm::GemmExecutor<f32, f32, f32> = rten_gemm::GemmExecutor::default();
    }
    let span = |v: &View<'_>, rows: usize, cols: usize| {
        if rows == 0 || cols == 0 {
            0
        } else {
            (rows - 1) * v.row + (cols - 1) * v.col + 1
        }
    };
    let a = NdTensorView::from_data_with_strides([m, p], &a.data[..span(&a, m, p)], [a.row, a.col])
        .expect("A's strides fit its data");
    let b = NdTensorView::from_data_with_strides([p, n], &b.data[..span(&b, p, n)], [b.row, b.col])
        .expect("B's strides fit its data");
    GEMM.with(|gemm| {
        gemm.gemm(
            &mut c[..m * n],
            GemmInputA::Unpacked(a),
            GemmInputB::Unpacked(b),
            GemmOptions {
                beta: if add { 1.0 } else { 0.0 },
                ..GemmOptions::default()
            },
        )
        .expect("the shapes agree")
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both stride orders, with and without accumulating, against the plain
    /// loop.
    #[test]
    fn the_library_product_is_the_plain_loop() {
        let wave =
            |n: usize, k: f32| -> Vec<f32> { (0..n).map(|i| (i as f32 * k).sin()).collect() };
        for &(m, n, p) in &[(1, 1, 1), (7, 17, 3), (100, 140, 600)] {
            let a = wave(m * p, 0.37);
            let b = wave(p * n, 0.91);
            for (a_view, b_view) in [
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
                for add in [false, true] {
                    let seed = wave(m * n, 0.13);
                    let mut fast = seed.clone();
                    library(&mut fast, a_view, b_view, m, n, p, add);
                    for i in 0..m {
                        for j in 0..n {
                            let mut s = if add { seed[i * n + j] as f64 } else { 0.0 };
                            for q in 0..p {
                                s += a_view.data[i * a_view.row + q * a_view.col] as f64
                                    * b_view.data[q * b_view.row + j * b_view.col] as f64;
                            }
                            let got = fast[i * n + j] as f64;
                            assert!(
                                (got - s).abs() <= 1e-4 * s.abs().max(1.0),
                                "{m}x{n}x{p} ({i}, {j})"
                            );
                        }
                    }
                }
            }
        }
    }
}
