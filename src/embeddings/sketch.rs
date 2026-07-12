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

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EllipsoidSketch {
    pub centroid: Vec<f32>,
    pub principal_axis: Vec<f32>,
    pub axis_radius: f32,
    pub residual_radius: f32,
}

#[derive(Debug, Clone)]
pub struct SketchMatch {
    pub inside: bool,
    pub density: f32,
}

impl EllipsoidSketch {
    /// Compute a rank-1 ellipsoid sketch from a set of chunk vectors using Power Iteration.
    pub fn compute(vectors: &[&[f32]]) -> Self {
        if vectors.is_empty() {
            return Self {
                centroid: Vec::new(),
                principal_axis: Vec::new(),
                axis_radius: 0.0,
                residual_radius: 0.0,
            };
        }

        let dim = vectors[0].len();
        let mut centroid = vec![0.0; dim];
        for v in vectors {
            for (i, &val) in v.iter().enumerate() {
                centroid[i] += val;
            }
        }
        let n = vectors.len() as f32;
        for c in &mut centroid {
            *c /= n;
        }

        if vectors.len() == 1 {
            return Self {
                centroid,
                principal_axis: vec![0.0; dim],
                axis_radius: 0.0,
                residual_radius: 0.0,
            };
        }

        // Center the vectors for PCA
        let centered: Vec<Vec<f32>> = vectors
            .iter()
            .map(|v| v.iter().zip(&centroid).map(|(a, b)| a - b).collect())
            .collect();

        // Power iteration to find the principal axis (largest eigenvector of X^T X)
        let mut p = vec![1.0 / (dim as f32).sqrt(); dim];
        for _ in 0..10 {
            let mut next_p = vec![0.0; dim];
            for x in &centered {
                let mut dot = 0.0;
                for (a, b) in x.iter().zip(&p) {
                    dot += a * b;
                }
                for (i, &val) in x.iter().enumerate() {
                    next_p[i] += dot * val;
                }
            }
            let mut mag = 0.0;
            for &val in &next_p {
                mag += val * val;
            }
            mag = mag.sqrt();
            if mag > 1e-6 {
                for val in &mut next_p {
                    *val /= mag;
                }
            }
            p = next_p;
        }

        let principal_axis = p;

        // Compute radii
        let mut max_axis_proj = 0.0f32;
        let mut max_residual = 0.0f32;

        for x in &centered {
            let mut proj_len = 0.0;
            for (a, b) in x.iter().zip(&principal_axis) {
                proj_len += a * b;
            }
            if proj_len.abs() > max_axis_proj {
                max_axis_proj = proj_len.abs();
            }

            let mut residual_sq = 0.0;
            for (i, &val) in x.iter().enumerate() {
                let res = val - proj_len * principal_axis[i];
                residual_sq += res * res;
            }
            let res_len = residual_sq.sqrt();
            if res_len > max_residual {
                max_residual = res_len;
            }
        }

        Self {
            centroid,
            principal_axis,
            axis_radius: max_axis_proj,
            residual_radius: max_residual,
        }
    }

    /// Check if a query vector is within the sketch.
    /// Returns a SketchMatch with a density score.
    pub fn matches(&self, query: &[f32], threshold: f32) -> SketchMatch {
        if query.len() != self.centroid.len() || self.centroid.is_empty() {
            return SketchMatch {
                inside: false,
                density: 0.0,
            };
        }

        let mut nq = 0.0;
        for q in query {
            nq += q * q;
        }
        let q_norm = if nq > 0.0 { nq.sqrt() } else { 1.0 };

        let mut q_dot_c = 0.0;
        let mut q_dot_u = 0.0;

        for (i, &q_val) in query.iter().enumerate() {
            let q = q_val / q_norm;
            q_dot_c += q * self.centroid[i];
            if !self.principal_axis.is_empty() {
                q_dot_u += q * self.principal_axis[i];
            }
        }

        // The query's projection onto the residual space has length sqrt(1 - (q_dot_u)^2).
        // Due to floating point inaccuracy, clamp to >= 0.0.
        let q_res_len = (1.0 - q_dot_u * q_dot_u).max(0.0).sqrt();

        // The maximum possible dot product with any vector in this ellipsoid:
        // max_sim = (q • c) + axis_radius * |q • u| + residual_radius * ||q_res||
        let max_sim = q_dot_c + self.axis_radius * q_dot_u.abs() + self.residual_radius * q_res_len;

        SketchMatch {
            inside: max_sim >= threshold,
            density: max_sim,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_sketch_power_iteration() {
        let v1 = vec![1.0, 2.0];
        let v2 = vec![3.0, 4.0];
        let v3 = vec![5.0, 6.0];
        let vectors = vec![v1.as_slice(), v2.as_slice(), v3.as_slice()];

        let sketch = EllipsoidSketch::compute(&vectors);

        // Centroid should be (3.0, 4.0)
        assert_eq!(sketch.centroid, vec![3.0, 4.0]);
        // Principal axis should be normalized (1, 1) direction
        let expected_axis = 1.0 / (2.0f32).sqrt();
        assert!((sketch.principal_axis[0].abs() - expected_axis).abs() < 1e-4);
        assert!((sketch.principal_axis[1].abs() - expected_axis).abs() < 1e-4);
    }
}
