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

//! The always-available backend: scalar dot products with runtime AVX2
//! dispatch (`engine::tensor::dot`), parallelized across output rows via
//! `rayon`. Also the fallback when no Vulkan-capable adapter is found.

use rayon::prelude::*;

use crate::engine::loader::QuantMatrix;
use crate::engine::tensor;

use super::Backend;

#[derive(Default)]
pub struct CpuBackend;

impl Backend for CpuBackend {
    fn matmul(&self, x: &[f32], n_tokens: usize, w: &QuantMatrix) -> Vec<f32> {
        let in_dim = w.in_dim;
        let out_dim = w.out_dim;
        debug_assert_eq!(x.len(), n_tokens * in_dim);
        let mut y = vec![0f32; n_tokens * out_dim];
        // Parallelize over output rows (typically far more of these than
        // tokens) so each weight row is dequantized exactly once and reused
        // across every token, rather than once per (token, row) pair.
        let columns: Vec<(usize, Vec<f32>)> = (0..out_dim)
            .into_par_iter()
            .map(|o| {
                let wo = w.row(o);
                let column: Vec<f32> = (0..n_tokens)
                    .map(|t| tensor::dot(&x[t * in_dim..(t + 1) * in_dim], &wo))
                    .collect();
                (o, column)
            })
            .collect();
        for (o, column) in columns {
            for (t, value) in column.into_iter().enumerate() {
                y[t * out_dim + o] = value;
            }
        }
        y
    }
}
