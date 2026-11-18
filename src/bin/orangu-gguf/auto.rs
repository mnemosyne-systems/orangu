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

//! A type per tensor, chosen for size and speed: the error table.
//!
//! What a type costs a tensor is measured, not assumed: the tensor's rows
//! are encoded at each candidate type, read back, and the difference is
//! weighed by the importance matrix — how much each input column matters to
//! the layer's output on the calibration text. Divided by the tensor's own
//! weighted energy, that is the share of the layer's output signal the type
//! loses, which compares across tensors of any size or scale.
//!
//! A large tensor is measured on a sample of its rows. The sample is
//! deterministic and spread evenly over the rows, so every expert of a
//! stacked tensor is in it and the same inputs give the same table.

use orangu::quantize::{self as q, block_size};
use rayon::prelude::*;

/// The types a speed-first choice starts from and spends upgrades on: each
/// has the tuned decode kernel, the integer-dot prefill GEMM and an AVX2
/// row dot. Smallest first.
pub const FAST: [u32; 3] = [q::GGML_TYPE_Q4_K, q::GGML_TYPE_Q5_K, q::GGML_TYPE_Q6_K];

/// The types below [`FAST`] a size cap can reach for: smaller, and slower
/// per byte on the card. Smallest first.
pub const SMALL: [u32; 3] = [q::GGML_TYPE_Q2_K, q::GGML_TYPE_Q3_K, q::GGML_TYPE_IQ4_XS];

/// For rows a 256-wide block does not divide: the 32-wide types. Smallest
/// first.
pub const NARROW: [u32; 4] = [
    q::GGML_TYPE_Q4_0,
    q::GGML_TYPE_Q5_0,
    q::GGML_TYPE_Q5_1,
    q::GGML_TYPE_Q8_0,
];

/// Rows measured per tensor at most. Enough that a tensor's estimate is
/// within a few percent of measuring every row.
pub const SAMPLE_ROWS: usize = 512;

/// The candidate types for a tensor whose rows are `ncols` long, smallest
/// first: the [`SMALL`] and [`FAST`] types where a 256-wide block divides
/// the row, the [`NARROW`] ones where it does not.
pub fn candidates(ncols: usize) -> Vec<u32> {
    let wide: Vec<u32> = SMALL.iter().chain(FAST.iter()).copied().collect();
    if ncols.is_multiple_of(block_size(q::GGML_TYPE_Q4_K)) {
        let mut all = wide;
        all.push(q::GGML_TYPE_Q8_0);
        all
    } else {
        NARROW
            .iter()
            .copied()
            .filter(|&t| ncols.is_multiple_of(block_size(t)))
            .collect()
    }
}

/// One candidate type's cost to one tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct Cost {
    pub ggml_type: u32,
    /// Stored size of the whole tensor at this type.
    pub bytes: u64,
    /// Weighted squared error over weighted energy, over the sampled rows.
    pub error: f64,
}

/// The rows a tensor of `rows` rows is measured on: all of them up to
/// `budget`, otherwise one row from each of `budget` equal stretches, at a
/// position a fixed hash of the stretch picks. A fixed stride would fall in
/// step with a tensor's own layout — the rotary pairs of a key projection
/// alternate row by row — and measure one kind of row only.
pub fn sample_rows(rows: usize, budget: usize) -> Vec<usize> {
    if rows <= budget {
        return (0..rows).collect();
    }
    (0..budget)
        .map(|i| {
            let start = i * rows / budget;
            let end = (i + 1) * rows / budget;
            let hash = (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32;
            start + (hash as usize) % (end - start)
        })
        .collect()
}

/// The importances a row of a tensor answers to: its expert's, for a
/// stacked tensor whose rows split evenly between its experts.
fn row_weights(
    importance: Option<&[f32]>,
    total_rows: usize,
    ncols: usize,
    r: usize,
) -> Option<&[f32]> {
    importance.map(|w| {
        let experts = (w.len() / ncols.max(1)).max(1);
        let rows_per_expert = (total_rows / experts).max(1);
        let e = (r / rows_per_expert).min(experts - 1);
        &w[e * ncols..(e + 1) * ncols]
    })
}

/// Per row of `rows`: the weighted squared error of encoding it at
/// `ggml_type`, and its weighted energy.
fn row_terms(
    ggml_type: u32,
    values: &[f32],
    ncols: usize,
    importance: Option<&[f32]>,
    rows: &[usize],
) -> Vec<(f64, f64)> {
    let total_rows = values.len() / ncols.max(1);
    rows.par_iter()
        .map(|&r| {
            let row = &values[r * ncols..(r + 1) * ncols];
            let weights = row_weights(importance, total_rows, ncols, r);
            let bytes = match weights {
                Some(w) => q::encode_weighted(ggml_type, row, ncols, w),
                None => q::encode(ggml_type, row, ncols),
            };
            let back = orangu::dequantize::dequantize(ggml_type, &bytes, ncols)
                .expect("every candidate type reads back");
            let (mut error, mut energy) = (0.0f64, 0.0f64);
            for (j, (&v, &b)) in row.iter().zip(&back).enumerate() {
                let w = weights.map_or(1.0, |w| f64::from(w[j]));
                let d = f64::from(v) - f64::from(b);
                error += w * d * d;
                energy += w * f64::from(v) * f64::from(v);
            }
            (error, energy)
        })
        .collect()
}

/// Error over energy, with nothing to lose counted as nothing lost.
fn ratio(error: f64, energy: f64) -> f64 {
    if energy > 0.0 {
        error / energy
    } else if error > 0.0 {
        f64::INFINITY
    } else {
        0.0
    }
}

/// [`Cost::error`] of encoding `values` (`ncols` to a row) at `ggml_type`,
/// measured on every row of `rows`. `importance` holds one weight per
/// column, per expert for a stacked tensor whose rows split evenly between
/// its experts; without one every column counts the same.
pub fn relative_error(
    ggml_type: u32,
    values: &[f32],
    ncols: usize,
    importance: Option<&[f32]>,
    rows: &[usize],
) -> f64 {
    // Summed in row order: the same table whatever the threads did.
    let (error, energy) = row_terms(ggml_type, values, ncols, importance, rows)
        .iter()
        .fold((0.0, 0.0), |(a, b), &(e, s)| (a + e, b + s));
    ratio(error, energy)
}

/// What a row's error is expected to follow, without encoding it: per
/// 32-wide group, the square of its largest magnitude — every type's step
/// is set by it — times the importance the group carries. And the row's
/// weighted energy.
fn row_proxy(row: &[f32], weights: Option<&[f32]>) -> (f64, f64) {
    let (mut proxy, mut energy) = (0.0f64, 0.0f64);
    for (g, group) in row.chunks(32).enumerate() {
        let mut amax = 0.0f32;
        let mut weight = 0.0f64;
        for (j, &v) in group.iter().enumerate() {
            amax = amax.max(v.abs());
            let w = weights.map_or(1.0, |w| f64::from(w[32 * g + j]));
            weight += w;
            energy += w * f64::from(v) * f64::from(v);
        }
        proxy += f64::from(amax) * f64::from(amax) * weight;
    }
    (proxy, energy)
}

/// [`Cost::error`] over a whole tensor, encoding at most `budget` rows.
///
/// A tensor's error is carried by a few rows — the ones holding its
/// outliers — so a plain sample of rows can miss most of it. Every row is
/// first given its [`row_proxy`], which costs one pass and no encoding. The
/// quarter of the budget with the largest proxies is measured outright; the
/// rest of the budget samples the remaining rows, and their error is scaled
/// up by how their proxies compare with the remaining rows' — a ratio that
/// varies far less from row to row than the error does.
pub fn estimated_error(
    ggml_type: u32,
    values: &[f32],
    ncols: usize,
    importance: Option<&[f32]>,
    budget: usize,
) -> f64 {
    let rows = values.len() / ncols.max(1);
    if rows <= budget {
        let all: Vec<usize> = (0..rows).collect();
        return relative_error(ggml_type, values, ncols, importance, &all);
    }
    let proxies: Vec<(f64, f64)> = (0..rows)
        .into_par_iter()
        .map(|r| {
            row_proxy(
                &values[r * ncols..(r + 1) * ncols],
                row_weights(importance, rows, ncols, r),
            )
        })
        .collect();
    let energy: f64 = proxies.iter().map(|p| p.1).sum();
    let mut by_proxy: Vec<usize> = (0..rows).collect();
    by_proxy.sort_by(|&a, &b| proxies[b].0.total_cmp(&proxies[a].0).then(a.cmp(&b)));
    let heavy = budget / 4;
    let mut rest = by_proxy[heavy..].to_vec();
    rest.sort_unstable();
    let picked: Vec<usize> = sample_rows(rest.len(), budget - heavy)
        .into_iter()
        .map(|i| rest[i])
        .collect();
    let error_of = |rows: &[usize]| -> f64 {
        row_terms(ggml_type, values, ncols, importance, rows)
            .iter()
            .map(|t| t.0)
            .sum()
    };
    let heavy_error = error_of(&by_proxy[..heavy]);
    let picked_proxy: f64 = picked.iter().map(|&r| proxies[r].0).sum();
    let rest_proxy: f64 = rest.iter().map(|&r| proxies[r].0).sum();
    let rest_error = if picked_proxy > 0.0 {
        error_of(&picked) * rest_proxy / picked_proxy
    } else {
        0.0
    };
    ratio(heavy_error + rest_error, energy)
}

/// Every candidate's [`Cost`] to one tensor of `ncols`-long rows, encoding
/// at most [`SAMPLE_ROWS`] of them. Smallest type first.
pub fn costs(values: &[f32], ncols: usize, importance: Option<&[f32]>) -> Vec<Cost> {
    let rows = values.len() / ncols.max(1);
    candidates(ncols)
        .into_iter()
        .map(|ggml_type| Cost {
            ggml_type,
            bytes: (q::row_bytes(ggml_type, ncols) * rows) as u64,
            error: estimated_error(ggml_type, values, ncols, importance, SAMPLE_ROWS),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic weights with a few large ones, like a trained layer's.
    fn weights(rows: usize, ncols: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (0..rows * ncols)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                if state & 0xff == 0 { u * 8.0 } else { u }
            })
            .collect()
    }

    /// More bits must lose less: the table is only worth anything if it
    /// orders the types the way their sizes do.
    #[test]
    fn more_bits_lose_less() {
        let ncols = 512;
        let values = weights(64, ncols, 1);
        let costs = costs(&values, ncols, None);
        let order = [
            q::GGML_TYPE_Q2_K,
            q::GGML_TYPE_Q3_K,
            q::GGML_TYPE_Q4_K,
            q::GGML_TYPE_Q5_K,
            q::GGML_TYPE_Q6_K,
            q::GGML_TYPE_Q8_0,
        ];
        let error = |t: u32| costs.iter().find(|c| c.ggml_type == t).unwrap().error;
        for pair in order.windows(2) {
            assert!(
                error(pair[0]) > error(pair[1]),
                "{} {} vs {} {}",
                q::type_name(pair[0]),
                error(pair[0]),
                q::type_name(pair[1]),
                error(pair[1])
            );
        }
        assert!(error(q::GGML_TYPE_Q8_0) < 1e-3);
    }

    /// A column the matrix says nothing reaches costs nothing, whatever
    /// its weights: two tensors that differ only there measure the same.
    #[test]
    fn unimportant_columns_cost_nothing() {
        let ncols = 512;
        let mut a = weights(16, ncols, 2);
        let mut b = a.clone();
        let other = weights(16, ncols, 3);
        // Each row's second block differs, and no importance reaches it.
        for r in 0..16 {
            a[r * ncols + 256..(r + 1) * ncols]
                .copy_from_slice(&other[r * ncols + 256..(r + 1) * ncols]);
            for v in &mut b[r * ncols + 256..(r + 1) * ncols] {
                *v *= 3.0;
            }
        }
        let mut importance = vec![1.0f32; ncols];
        importance[256..].fill(0.0);
        let rows: Vec<usize> = (0..16).collect();
        for t in FAST {
            let ea = relative_error(t, &a, ncols, Some(&importance), &rows);
            let eb = relative_error(t, &b, ncols, Some(&importance), &rows);
            assert_eq!(ea, eb, "{}", q::type_name(t));
        }
    }

    /// A stacked tensor's rows each take their own expert's importances.
    #[test]
    fn each_expert_reads_its_own_importances() {
        let ncols = 256;
        let values = weights(8, ncols, 4);
        let ones = vec![1.0f32; ncols];
        let mut second = vec![1.0f32; ncols];
        second[..128].fill(0.0);
        let stacked: Vec<f32> = ones.iter().chain(second.iter()).copied().collect();
        let t = q::GGML_TYPE_Q4_K;
        let first_half: Vec<usize> = (0..4).collect();
        let second_half: Vec<usize> = (4..8).collect();
        assert_eq!(
            relative_error(t, &values, ncols, Some(&stacked), &first_half),
            relative_error(t, &values, ncols, Some(&ones), &first_half)
        );
        assert_eq!(
            relative_error(t, &values, ncols, Some(&stacked), &second_half),
            relative_error(t, &values[4 * ncols..], ncols, Some(&second), &first_half)
        );
    }

    /// The sample is spread over the whole tensor and stands in for it.
    #[test]
    fn a_sample_stands_in_for_every_row() {
        assert_eq!(sample_rows(4, 8), vec![0, 1, 2, 3]);
        let rows = sample_rows(1000, 10);
        assert_eq!(rows.len(), 10);
        for (i, &r) in rows.iter().enumerate() {
            assert!(
                (i * 100..(i + 1) * 100).contains(&r),
                "row {r} for stretch {i}"
            );
        }
        assert_eq!(rows, sample_rows(1000, 10), "the same rows every time");

        let ncols = 256;
        let values = weights(4096, ncols, 5);
        let all: Vec<usize> = (0..4096).collect();
        for t in [q::GGML_TYPE_Q4_K, q::GGML_TYPE_Q6_K] {
            let full = relative_error(t, &values, ncols, None, &all);
            let sampled = estimated_error(t, &values, ncols, None, SAMPLE_ROWS);
            assert!(
                (sampled - full).abs() <= 0.05 * full,
                "{}: sampled {sampled} vs full {full}",
                q::type_name(t)
            );
        }
    }

    /// On a real model the sample stands in for the whole tensor too: every
    /// block matrix of Qwen3-0.6B at Q4_K and Q6_K, weighted by its own
    /// calibration matrix. Needs `orangu-converted/Qwen3-0.6B-BF16.gguf` and
    /// its `.calib.imatrix.gguf` (`orangu-gguf -m Qwen/Qwen3-0.6B
    /// --calibration TEXT` writes both).
    #[test]
    #[ignore]
    fn a_sample_stands_in_for_every_row_of_a_real_model() {
        use orangu::gguf::GgufFile;
        let dir = crate::hf::models_dir()
            .expect("a models directory")
            .join("orangu-converted");
        let path = dir.join("Qwen3-0.6B-BF16.gguf");
        let matrix = crate::imatrix::load(&dir.join("Qwen3-0.6B-BF16.calib.imatrix.gguf")).unwrap();
        let file = GgufFile::open(&path).unwrap();
        let handle = std::fs::File::open(&path).unwrap();
        // Safety: read-only, and nothing writes the file meanwhile.
        let map = unsafe { memmap2::Mmap::map(&handle) }.unwrap();
        let started = std::time::Instant::now();
        let (mut tensors, mut worst) = (0, 0.0f64);
        for t in &file.tensors {
            let Some(importance) = matrix.get(&t.name) else {
                continue;
            };
            let elements = t.element_count() as usize;
            let ncols = t.dims[0] as usize;
            let start = (file.data_offset + t.offset) as usize;
            let size = orangu::dequantize::tensor_byte_size(t.ggml_type, elements as u64).unwrap();
            let values = orangu::dequantize::dequantize(
                t.ggml_type,
                &map[start..start + size as usize],
                elements,
            )
            .unwrap();
            let rows = elements / ncols;
            let all: Vec<usize> = (0..rows).collect();
            for ty in [q::GGML_TYPE_Q4_K, q::GGML_TYPE_Q6_K] {
                let full = relative_error(ty, &values, ncols, Some(importance), &all);
                let sampled = estimated_error(ty, &values, ncols, Some(importance), SAMPLE_ROWS);
                let off = (sampled - full).abs() / full;
                worst = worst.max(off);
                assert!(
                    off <= 0.05,
                    "{} {}: sampled {sampled} vs full {full}",
                    t.name,
                    q::type_name(ty)
                );
            }
            tensors += 1;
        }
        eprintln!(
            "{tensors} tensors, worst sample {:.2}% off, {:.1} s",
            worst * 100.0,
            started.elapsed().as_secs_f64()
        );
        assert_eq!(tensors, 196);
    }

    /// Rows a 256-wide block does not divide get the 32-wide types only.
    #[test]
    fn narrow_rows_get_the_narrow_types() {
        assert_eq!(candidates(896), NARROW.to_vec());
        assert!(candidates(1024).contains(&q::GGML_TYPE_Q4_K));
        assert!(!candidates(1024).contains(&q::GGML_TYPE_Q4_0));
    }
}
