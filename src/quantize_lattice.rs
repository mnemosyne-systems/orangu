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

//! The lattice I-quants: groups of 4 or 8 magnitudes stored as one index
//! into a fixed codebook of points, with the signs kept apart.
//!
//! A codebook holds only some of the points a group's levels could form.
//! Fitting a group therefore means rounding each value to a level and,
//! when the result is not in the codebook, choosing among its nearest
//! codebook points the one with the smallest weighted error. The nearest
//! points of every off-codebook position are worked out once per type
//! ([`Lattice`]): every codebook point ordered by squared distance and
//! then by index, keeping the first few distinct distances.

use std::sync::OnceLock;

use half::f16;
use rayon::prelude::*;

use crate::iq_grids::{IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID};
use crate::quantize::{QK_K, fit_positive_scale, nearest_even};

/// A codebook and, for every position off it, its nearest points.
pub(crate) struct Lattice {
    /// Values per group.
    dims: usize,
    /// Bits per level in a position's index.
    bits: usize,
    /// Each codebook point as `2·level + 1` per value.
    points: Vec<[i8; 8]>,
    /// Per position: its codebook index, or `-(start + 1)` of its
    /// neighbour list in `neighbours`.
    map: Vec<i32>,
    /// Neighbour lists, each a count followed by that many indices.
    neighbours: Vec<u16>,
}

impl Lattice {
    /// Builds the lattice for `points`, with `positions` positions and
    /// each off-codebook position's points within `shells` distinct
    /// distances.
    fn build(
        points: Vec<[i8; 8]>,
        dims: usize,
        bits: usize,
        positions: usize,
        shells: usize,
    ) -> Self {
        let mask = (1usize << bits) - 1;
        let mut map = vec![-1i32; positions];
        for (i, point) in points.iter().enumerate() {
            let mut u = 0usize;
            for (k, &p) in point.iter().take(dims).enumerate() {
                u |= (((p - 1) / 2) as usize) << (bits * k);
            }
            map[u] = i as i32;
        }
        let lists: Vec<Option<Vec<u16>>> = (0..positions)
            .into_par_iter()
            .map(|i| {
                if map[i] >= 0 {
                    return None;
                }
                let position: [i32; 8] = std::array::from_fn(|k| {
                    if k < dims {
                        2 * ((i >> (bits * k)) & mask) as i32 + 1
                    } else {
                        0
                    }
                });
                let mut by_distance: Vec<(i32, u16)> = points
                    .iter()
                    .enumerate()
                    .map(|(j, p)| {
                        let d2 = (0..dims)
                            .map(|k| (p[k] as i32 - position[k]).pow(2))
                            .sum::<i32>();
                        (d2, j as u16)
                    })
                    .collect();
                by_distance.sort_unstable();
                let mut list = Vec::new();
                let (mut shell, mut have) = (by_distance[0].0, 1);
                for &(d2, j) in &by_distance {
                    if d2 > shell {
                        if have == shells {
                            break;
                        }
                        shell = d2;
                        have += 1;
                    }
                    list.push(j);
                }
                Some(list)
            })
            .collect();
        let mut neighbours = Vec::new();
        for (i, list) in lists.into_iter().enumerate() {
            if let Some(list) = list {
                map[i] = -(neighbours.len() as i32 + 1);
                neighbours.push(list.len() as u16);
                neighbours.extend(list);
            }
        }
        Lattice {
            dims,
            bits,
            points,
            map,
            neighbours,
        }
    }

    /// The position `levels` spell.
    fn position(&self, levels: &[i8]) -> usize {
        levels
            .iter()
            .take(self.dims)
            .enumerate()
            .fold(0, |u, (k, &l)| u | ((l as usize) << (self.bits * k)))
    }

    /// The codebook index of `levels`, or the nearest codebook point by
    /// weighted error at `scale` — which then replaces `levels`. Returns
    /// the index and whether `levels` was already on the codebook.
    fn snap(
        &self,
        levels: &mut [i8],
        values: &[f32],
        weights: &[f32],
        scale: f32,
    ) -> (usize, bool) {
        let u = self.position(levels);
        let found = self.map[u];
        if found >= 0 {
            return (found as usize, true);
        }
        let start = (-found - 1) as usize;
        let count = self.neighbours[start] as usize;
        let mut best = f32::MAX;
        let mut index = usize::MAX;
        for &j in &self.neighbours[start + 1..start + 1 + count] {
            let point = &self.points[j as usize];
            let mut d2 = 0.0f32;
            for i in 0..self.dims {
                let diff = scale * point[i] as f32 - values[i];
                d2 += weights[i] * diff * diff;
            }
            if d2 < best {
                best = d2;
                index = j as usize;
            }
        }
        for (l, &p) in levels.iter_mut().zip(&self.points[index]).take(self.dims) {
            *l = (p - 1) / 2;
        }
        (index, false)
    }
}

/// Builds the lattice `ggml_type` uses, if any, so the parallel encoders
/// that follow never wait on one another for it: a rayon worker blocked
/// on a lattice another worker is building in parallel would never be
/// given the work that finishes it.
pub(crate) fn prepare(ggml_type: u32) {
    use crate::quantize::*;
    match ggml_type {
        GGML_TYPE_IQ1_S | GGML_TYPE_IQ1_M => {
            iq1s_lattice();
        }
        GGML_TYPE_IQ2_XXS => {
            iq2xxs_lattice();
        }
        GGML_TYPE_IQ2_XS => {
            iq2xs_lattice();
        }
        GGML_TYPE_IQ2_S => {
            iq2s_lattice();
        }
        GGML_TYPE_IQ3_XXS => {
            iq3xxs_lattice();
        }
        GGML_TYPE_IQ3_S => {
            iq3s_lattice();
        }
        _ => {}
    }
}

/// A lattice type's block encoder: one 256-value block, fitted to the
/// column importances when given them.
pub(crate) type Encoder = fn(&[f32], Option<&[f32]>, &mut Vec<u8>);

/// The encoder for `ggml_type`, if it is a lattice type.
pub(crate) fn encoder(ggml_type: u32) -> Option<Encoder> {
    use crate::quantize::*;
    Some(match ggml_type {
        GGML_TYPE_IQ1_S => quantize_iq1_s,
        GGML_TYPE_IQ1_M => quantize_iq1_m,
        GGML_TYPE_IQ2_XXS => quantize_iq2_xxs,
        GGML_TYPE_IQ2_XS => quantize_iq2_xs,
        GGML_TYPE_IQ2_S => quantize_iq2_s,
        GGML_TYPE_IQ3_XXS => quantize_iq3_xxs,
        GGML_TYPE_IQ3_S => quantize_iq3_s,
        _ => return None,
    })
}

/// Builds a lattice from a codebook whose values are given by rank in
/// `alphabet` (level `l` is `alphabet[l]`).
fn from_alphabet(
    grid: impl Iterator<Item = u64>,
    dims: usize,
    alphabet: &[u8],
    bits: usize,
    positions: usize,
    shells: usize,
) -> Lattice {
    let points = grid
        .map(|g| {
            std::array::from_fn(|k| {
                if k >= dims {
                    return 0;
                }
                let m = (g >> (8 * k)) as u8;
                let level = alphabet
                    .iter()
                    .position(|&v| v == m)
                    .expect("a codebook value");
                2 * level as i8 + 1
            })
        })
        .collect();
    Lattice::build(points, dims, bits, positions, shells)
}

/// The magnitudes of the 2-bit codebooks, by level.
const TWO_BIT: [u8; 3] = [8, 25, 43];

fn iq2xxs_lattice() -> &'static Lattice {
    static LATTICE: OnceLock<Lattice> = OnceLock::new();
    LATTICE.get_or_init(|| from_alphabet(IQ2XXS_GRID.iter().copied(), 8, &TWO_BIT, 2, 43_692, 2))
}

fn iq2xs_lattice() -> &'static Lattice {
    static LATTICE: OnceLock<Lattice> = OnceLock::new();
    LATTICE.get_or_init(|| from_alphabet(IQ2XS_GRID.iter().copied(), 8, &TWO_BIT, 2, 43_692, 2))
}

fn iq2s_lattice() -> &'static Lattice {
    static LATTICE: OnceLock<Lattice> = OnceLock::new();
    LATTICE.get_or_init(|| from_alphabet(IQ2S_GRID.iter().copied(), 8, &TWO_BIT, 2, 43_692, 1))
}

fn iq3xxs_lattice() -> &'static Lattice {
    static LATTICE: OnceLock<Lattice> = OnceLock::new();
    LATTICE.get_or_init(|| {
        const ALPHABET: [u8; 8] = [4, 12, 20, 28, 36, 44, 52, 62];
        from_alphabet(
            IQ3XXS_GRID.iter().map(|&g| g as u64),
            4,
            &ALPHABET,
            3,
            4096,
            2,
        )
    })
}

fn iq3s_lattice() -> &'static Lattice {
    static LATTICE: OnceLock<Lattice> = OnceLock::new();
    LATTICE.get_or_init(|| {
        const ALPHABET: [u8; 8] = [1, 3, 5, 7, 9, 11, 13, 15];
        from_alphabet(
            IQ3S_GRID.iter().map(|&g| g as u64),
            4,
            &ALPHABET,
            3,
            4096,
            3,
        )
    })
}

/// The 1-bit codebook of `IQ1_S` and `IQ1_M`: values −1, 0 and 1.
fn iq1s_lattice() -> &'static Lattice {
    static LATTICE: OnceLock<Lattice> = OnceLock::new();
    LATTICE.get_or_init(|| {
        const ALPHABET: [u8; 3] = [0xff, 0x00, 0x01];
        from_alphabet(IQ1S_GRID.iter().copied(), 8, &ALPHABET, 2, 43_692, 3)
    })
}

/// How the groups are refitted once the scan has chosen a scale.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Refit {
    /// Only the groups that were off the lattice.
    OffLattice,
    /// Every group, when any was off the lattice.
    AllWhenAnyOff,
    /// Every group, always.
    Always,
}

/// What differs between the 2- and 3-bit lattice types' fits.
struct Shape {
    /// Values per sub-block, each with its own scale.
    block: usize,
    /// Levels per value.
    max_level: i32,
    /// The scan: `id = (2·max_level − 1 + step·is) / range` for
    /// `is` in `−tries..=tries`.
    step: f32,
    tries: i32,
    refit: Refit,
    /// Whether a group's signs are forced to even parity (seven sign bits
    /// stored), by flipping the value an error costs least on.
    even_signs: bool,
    /// The scan starts from a positive-scale fit rather than the largest
    /// magnitude.
    start_from_fit: bool,
    /// Below this largest magnitude a sub-block is left empty.
    too_small: fn(f32) -> bool,
    /// Without an importance matrix, a value's weight from the block's
    /// spread `sigma2` and itself; `None` when the type requires a matrix.
    weight: fn(f32, f32) -> f32,
    /// Whether the block's spread is taken twice (`2·Σx²/256`).
    double_spread: bool,
}

/// The fit of one super-block: per sub-block its scale, per group its
/// codebook index, per group its sign byte, and which sub-blocks were too
/// small to fit.
struct SubBlockFit {
    scales: Vec<f32>,
    indices: Vec<usize>,
    signs: Vec<u8>,
    skipped: Vec<bool>,
}

fn fit_sub_blocks(
    x: &[f32; QK_K],
    importance: Option<&[f32]>,
    lattice: &Lattice,
    shape: &Shape,
) -> SubBlockFit {
    let dims = lattice.dims;
    let block = shape.block;
    let groups = block / dims;
    let kmax = shape.max_level;
    let top = (2 * kmax - 1) as f32;
    let mut sum = 0.0f32;
    for v in x {
        sum += v * v;
    }
    let sigma2 = if shape.double_spread { 2.0 * sum } else { sum } / QK_K as f32;
    let mut out = SubBlockFit {
        scales: vec![0.0; QK_K / block],
        indices: vec![0; QK_K / dims],
        signs: vec![0; QK_K / 8],
        skipped: vec![false; QK_K / block],
    };
    for ib in 0..QK_K / block {
        let xb = &x[block * ib..block * (ib + 1)];
        let weight: Vec<f32> = xb
            .iter()
            .enumerate()
            .map(|(i, &v)| match importance {
                Some(q) => q[block * ib + i] * (sigma2 + v * v).sqrt(),
                None => (shape.weight)(sigma2, v),
            })
            .collect();
        let waux: Vec<f32> = weight.iter().map(|w| w.sqrt()).collect();
        let mut xval = vec![0.0f32; block];
        let mut signs = vec![0u8; block / 8];
        for (k, s) in signs.iter_mut().enumerate() {
            let mut flips = 0;
            for i in 0..8 {
                let v = xb[8 * k + i];
                if v >= 0.0 {
                    xval[8 * k + i] = v;
                } else {
                    xval[8 * k + i] = -v;
                    flips += 1;
                    *s |= 1 << i;
                }
            }
            if shape.even_signs {
                if flips % 2 == 1 {
                    let cost = |i: usize| weight[8 * k + i] * xb[8 * k + i] * xb[8 * k + i];
                    let mut imin = 0;
                    let mut min = cost(0);
                    for i in 1..8 {
                        let c = cost(i);
                        if c < min {
                            min = c;
                            imin = i;
                        }
                    }
                    xval[8 * k + imin] = -xval[8 * k + imin];
                    *s ^= 1 << imin;
                }
                *s &= 127;
            }
        }
        let mut max = xval[0];
        for &v in &xval[1..] {
            max = max.max(v);
        }
        if (shape.too_small)(max) {
            out.skipped[ib] = true;
            continue;
        }
        let mut levels = vec![0i8; block];
        let mut indices = vec![0usize; groups];
        let (mut scale, range) = if shape.start_from_fit {
            let mut ignored = vec![0u8; block];
            let fitted = fit_positive_scale(&xval, kmax + 1, &mut ignored, &weight);
            let range = fitted * kmax as f32;
            if range <= 0.0 {
                out.skipped[ib] = true;
                continue;
            }
            (fitted, range)
        } else {
            (max / top, max)
        };
        let mut best = 0.0f32;
        let mut on_lattice = vec![shape.refit != Refit::AllWhenAnyOff; groups];
        let mut aux = vec![0i8; block];
        let mut aux_on = vec![true; groups];
        let mut aux_indices = vec![0usize; groups];
        for is in -shape.tries..=shape.tries {
            let id = (top + is as f32 * shape.step) / range;
            let this_scale = 1.0 / id;
            for k in 0..groups {
                for i in 0..dims {
                    let l = nearest_even(0.5 * (id * xval[dims * k + i] - 1.0));
                    aux[dims * k + i] = l.clamp(0, kmax - 1) as i8;
                }
                let at = dims * k..dims * (k + 1);
                let (index, on) = lattice.snap(
                    &mut aux[at.clone()],
                    &xval[at.clone()],
                    &waux[at],
                    this_scale,
                );
                aux_on[k] = on;
                aux_indices[k] = index;
            }
            let (mut sumqx, mut sumq2) = (0.0f32, 0.0f32);
            for i in 0..block {
                let w = weight[i];
                let q = (2 * aux[i] as i32 + 1) as f32;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                scale = sumqx / sumq2;
                best = scale * sumqx;
                levels.copy_from_slice(&aux);
                on_lattice.copy_from_slice(&aux_on);
                indices.copy_from_slice(&aux_indices);
            }
        }
        let refit = match shape.refit {
            Refit::Always => scale > 0.0,
            _ => on_lattice.iter().any(|&on| !on) && scale > 0.0,
        };
        if refit {
            let id = 1.0 / scale;
            for k in 0..groups {
                if shape.refit == Refit::OffLattice && on_lattice[k] {
                    continue;
                }
                let at = dims * k..dims * (k + 1);
                for i in 0..dims {
                    let l = nearest_even(0.5 * (id * xval[dims * k + i] - 1.0));
                    levels[dims * k + i] = l.clamp(0, kmax - 1) as i8;
                }
                let (index, _) =
                    lattice.snap(&mut levels[at.clone()], &xval[at.clone()], &waux[at], scale);
                indices[k] = index;
            }
            let (mut sumqx, mut sumq2) = (0.0f32, 0.0f32);
            for i in 0..block {
                let w = weight[i];
                let q = (2 * levels[i] as i32 + 1) as f32;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if sumq2 > 0.0 {
                scale = sumqx / sumq2;
            }
        }
        if scale < 0.0 {
            scale = -scale;
            for s in &mut signs {
                *s = if shape.even_signs { !*s & 127 } else { !*s };
            }
        }
        // Every group's index, from the levels as they now stand.
        for k in 0..groups {
            let u = lattice.position(&levels[dims * k..dims * (k + 1)]);
            let found = lattice.map[u];
            assert!(found >= 0, "a fitted group off the lattice");
            out.indices[ib * groups + k] = found as usize;
        }
        out.signs[ib * block / 8..(ib + 1) * block / 8].copy_from_slice(&signs);
        out.scales[ib] = scale;
    }
    out
}

/// The 4-bit sub-block scales against `max / 31`; returns that divisor
/// and the codes.
fn scale_codes(scales: &[f32]) -> (f32, Vec<u8>) {
    let max_scale = scales.iter().fold(0.0f32, |m, &s| m.max(s));
    if max_scale == 0.0 {
        return (0.0, vec![0; scales.len()]);
    }
    let d = max_scale / 31.0;
    let id = 1.0 / d;
    let codes = scales
        .iter()
        .map(|&s| nearest_even(0.5 * (id * s - 1.0)).clamp(0, 15) as u8)
        .collect();
    (d, codes)
}

/// A copy of one super-block, zero-filled past a short tail.
fn super_block(block: &[f32]) -> [f32; QK_K] {
    let mut x = [0.0f32; QK_K];
    let n = block.len().min(QK_K);
    x[..n].copy_from_slice(&block[..n]);
    x
}

/// `IQ2_XXS`: 256 values as eight 32-wide sub-blocks; each 8-value group a
/// byte index into a 256-point codebook, seven sign bits per group (the
/// eighth is the parity), and a 4-bit scale per sub-block.
pub(crate) fn quantize_iq2_xxs(block: &[f32], importance: Option<&[f32]>, out: &mut Vec<u8>) {
    let x = super_block(block);
    let shape = Shape {
        block: 32,
        max_level: 3,
        step: 0.1,
        tries: 6,
        refit: Refit::Always,
        even_signs: true,
        start_from_fit: true,
        too_small: |max| max < 1e-15,
        weight: |_, v| v * v,
        double_spread: false,
    };
    let fit = fit_sub_blocks(&x, importance, iq2xxs_lattice(), &shape);
    let (d, codes) = scale_codes(&fit.scales);
    let mut q2 = [0u32; QK_K / 16];
    if d > 0.0 {
        for ib in 0..QK_K / 32 {
            if fit.skipped[ib] {
                q2[2 * ib + 1] |= (codes[ib] as u32) << 28;
                continue;
            }
            for k in 0..4 {
                q2[2 * ib] |= (fit.indices[4 * ib + k] as u32) << (8 * k);
                q2[2 * ib + 1] |= (fit.signs[4 * ib + k] as u32) << (7 * k);
            }
            q2[2 * ib + 1] |= (codes[ib] as u32) << 28;
        }
    }
    out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
    for w in q2 {
        out.extend_from_slice(&w.to_le_bytes());
    }
}

/// `IQ2_XS`: 256 values as sixteen 16-wide sub-blocks; each 8-value group a
/// 9-bit index into a 512-point codebook with seven sign bits beside it,
/// and a 4-bit scale per sub-block.
pub(crate) fn quantize_iq2_xs(block: &[f32], importance: Option<&[f32]>, out: &mut Vec<u8>) {
    let x = super_block(block);
    let shape = Shape {
        block: 16,
        max_level: 3,
        step: 0.1,
        tries: 9,
        refit: Refit::OffLattice,
        even_signs: true,
        start_from_fit: false,
        too_small: |max| max < 1e-15,
        weight: |_, v| v * v,
        double_spread: false,
    };
    let fit = fit_sub_blocks(&x, importance, iq2xs_lattice(), &shape);
    let (d, codes) = scale_codes(&fit.scales);
    let mut q2 = [0u16; QK_K / 8];
    let mut scales = [0u8; QK_K / 32];
    if d > 0.0 {
        for ib in 0..QK_K / 16 {
            if !fit.skipped[ib] {
                for k in 0..2 {
                    q2[2 * ib + k] =
                        fit.indices[2 * ib + k] as u16 | ((fit.signs[2 * ib + k] as u16) << 9);
                }
            }
            scales[ib / 2] |= codes[ib] << (4 * (ib % 2));
        }
    }
    out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
    for w in q2 {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out.extend_from_slice(&scales);
}

/// `IQ2_S`: 256 values as sixteen 16-wide sub-blocks with 4-bit scales,
/// each 8-value group a 10-bit index into a 1024-point codebook of three
/// magnitudes, and a sign byte per group.
pub(crate) fn quantize_iq2_s(block: &[f32], importance: Option<&[f32]>, out: &mut Vec<u8>) {
    let x = super_block(block);
    let shape = Shape {
        block: 16,
        max_level: 3,
        step: 0.1,
        tries: 9,
        refit: Refit::OffLattice,
        even_signs: false,
        start_from_fit: false,
        too_small: |max| max < 1e-8,
        weight: |sigma2, v| 0.25 * sigma2 + v * v,
        double_spread: true,
    };
    let fit = fit_sub_blocks(&x, importance, iq2s_lattice(), &shape);
    let (d, codes) = scale_codes(&fit.scales);
    let mut qs = [0u8; QK_K / 4];
    let mut qh = [0u8; QK_K / 32];
    for ib in 0..QK_K / 16 {
        if fit.skipped[ib] {
            continue;
        }
        for k in 0..2 {
            let i8 = 2 * ib + k;
            let index = fit.indices[i8];
            qs[i8] = (index & 255) as u8;
            qh[i8 / 4] |= ((index >> 8) as u8) << (2 * (i8 % 4));
            qs[QK_K / 8 + i8] = fit.signs[i8];
        }
    }
    let mut scales = [0u8; QK_K / 32];
    for (ib, &c) in codes.iter().enumerate() {
        scales[ib / 2] |= c << (4 * (ib % 2));
    }
    out.extend_from_slice(&f16::from_f32(d * 0.9875).to_le_bytes());
    out.extend_from_slice(&qs);
    out.extend_from_slice(&qh);
    out.extend_from_slice(&scales);
}

/// `IQ3_XXS`: 256 values as eight 32-wide sub-blocks; each 4-value group a
/// byte index into a 256-point codebook, and per sub-block 28 sign bits
/// (seven per eight values) and a 4-bit scale in one word.
pub(crate) fn quantize_iq3_xxs(block: &[f32], importance: Option<&[f32]>, out: &mut Vec<u8>) {
    let x = super_block(block);
    let shape = Shape {
        block: 32,
        max_level: 8,
        step: 0.2,
        tries: 15,
        refit: Refit::OffLattice,
        even_signs: true,
        start_from_fit: false,
        too_small: |max| max < 1e-8,
        weight: |_, v| v * v,
        double_spread: true,
    };
    let fit = fit_sub_blocks(&x, importance, iq3xxs_lattice(), &shape);
    let (d, codes) = scale_codes(&fit.scales);
    let mut qs = [0u8; QK_K / 4];
    let mut words = [0u32; QK_K / 32];
    if d > 0.0 {
        for ib in 0..QK_K / 32 {
            if !fit.skipped[ib] {
                for k in 0..8 {
                    qs[8 * ib + k] = fit.indices[8 * ib + k] as u8;
                }
                for k in 0..4 {
                    words[ib] |= (fit.signs[4 * ib + k] as u32) << (7 * k);
                }
            }
            words[ib] |= (codes[ib] as u32) << 28;
        }
    }
    out.extend_from_slice(&f16::from_f32(d * 1.0125).to_le_bytes());
    out.extend_from_slice(&qs);
    for w in words {
        out.extend_from_slice(&w.to_le_bytes());
    }
}

/// `IQ3_S`: 256 values as eight 32-wide sub-blocks with 4-bit scales, each
/// 4-value group a 9-bit index into a 512-point codebook of eight
/// magnitudes, and the signs as bits.
///
/// A sub-block of exact zeros writes no indices or signs and does not
/// advance past them, so the next sub-block's land in its place; readers
/// expect files laid out that way.
pub(crate) fn quantize_iq3_s(block: &[f32], importance: Option<&[f32]>, out: &mut Vec<u8>) {
    let x = super_block(block);
    let shape = Shape {
        block: 32,
        max_level: 8,
        step: 0.2,
        tries: 9,
        refit: Refit::AllWhenAnyOff,
        even_signs: false,
        start_from_fit: false,
        too_small: |max| max == 0.0,
        weight: |_, v| v * v,
        double_spread: true,
    };
    let fit = fit_sub_blocks(&x, importance, iq3s_lattice(), &shape);
    let (d, codes) = scale_codes(&fit.scales);
    let mut qs = [0u8; QK_K / 4];
    let mut qh = [0u8; QK_K / 32];
    let mut signs = [0u8; QK_K / 8];
    let (mut qs_at, mut signs_at) = (0usize, 0usize);
    for ib in 0..QK_K / 32 {
        if fit.skipped[ib] {
            continue;
        }
        for k in 0..8 {
            let g = ib * 8 + k;
            let index = fit.indices[g];
            qs[qs_at + k] = (index & 255) as u8;
            qh[g / 8] |= ((index >> 8) as u8) << (g % 8);
        }
        qs_at += 8;
        signs[signs_at..signs_at + 4].copy_from_slice(&fit.signs[ib * 4..ib * 4 + 4]);
        signs_at += 4;
    }
    let mut scales = [0u8; QK_K / 64];
    for (ib, &c) in codes.iter().enumerate() {
        scales[ib / 2] |= c << (4 * (ib % 2));
    }
    out.extend_from_slice(&f16::from_f32(d * 1.033).to_le_bytes());
    out.extend_from_slice(&qs);
    out.extend_from_slice(&qh);
    out.extend_from_slice(&signs);
    out.extend_from_slice(&scales);
}

/// The 1-bit types' three values, shifted by `±delta` per group.
const IQ1_DELTA: f32 = 0.125;
const SHIFTED_UP: [f32; 3] = [-1.0 + IQ1_DELTA, IQ1_DELTA, 1.0 + IQ1_DELTA];
const SHIFTED_DOWN: [f32; 3] = [-1.0 - IQ1_DELTA, -IQ1_DELTA, 1.0 - IQ1_DELTA];

/// `values` sorted ascending, stably, with their positions.
fn sorted(values: &[f32]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| {
        values[a]
            .partial_cmp(&values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    order
}

/// The nearest codebook point to `levels` by weighted error with the
/// shifted values `shifted` at `scale`, when `levels` is off the codebook.
/// Returns the index; `levels` becomes that point's.
fn snap_1bit(
    lattice: &Lattice,
    levels: &mut [i8],
    values: &[f32],
    weights: &[f32],
    scale: f32,
    shifted: &[f32; 3],
) -> (usize, bool) {
    let u = lattice.position(levels);
    let found = lattice.map[u];
    if found >= 0 {
        return (found as usize, true);
    }
    let start = (-found - 1) as usize;
    let count = lattice.neighbours[start] as usize;
    let mut best = f32::MAX;
    let mut index = usize::MAX;
    for &j in &lattice.neighbours[start + 1..start + 1 + count] {
        let point = &lattice.points[j as usize];
        let mut d2 = 0.0f32;
        for i in 0..8 {
            let q = shifted[((point[i] - 1) / 2) as usize];
            let diff = scale * q - values[i];
            d2 += weights[i] * diff * diff;
        }
        if d2 < best {
            best = d2;
            index = j as usize;
        }
    }
    for (l, &p) in levels.iter_mut().zip(&lattice.points[index]).take(8) {
        *l = (p - 1) / 2;
    }
    (index, false)
}

/// The weighted least-squares scale of `indices`' points against `x`, each
/// group with its own shift.
fn refit_1bit(
    lattice: &Lattice,
    indices: &[usize],
    x: &[f32],
    weight: &[f32],
    shift_of: impl Fn(usize) -> &'static [f32; 3],
) -> Option<f32> {
    let (mut sumqx, mut sumq2) = (0.0f32, 0.0f32);
    for (k, &index) in indices.iter().enumerate() {
        let point = &lattice.points[index];
        let shifted = shift_of(k);
        for j in 0..8 {
            let w = weight[8 * k + j];
            let q = shifted[((point[j] - 1) / 2) as usize];
            sumqx += w * q * x[8 * k + j];
            sumq2 += w * q * q;
        }
    }
    (sumqx > 0.0 && sumq2 > 0.0).then(|| sumqx / sumq2)
}

/// Each value's column importance times its size against the block's
/// spread, or its square without a matrix.
fn weights_1bit(x: &[f32; QK_K], importance: Option<&[f32]>, at: usize, n: usize) -> Vec<f32> {
    let mut sum = 0.0f32;
    for v in x {
        sum += v * v;
    }
    let sigma2 = 2.0 * sum / QK_K as f32;
    (at..at + n)
        .map(|i| match importance {
            Some(q) => q[i] * (sigma2 + x[i] * x[i]).sqrt(),
            None => x[i] * x[i],
        })
        .collect()
}

/// `IQ1_S`: 256 values as eight 32-wide sub-blocks; each 8-value group an
/// 11-bit index into a 2048-point codebook of −1, 0 and 1, the whole
/// sub-block shifted up or down by an eighth, and a 3-bit scale.
///
/// With three values a sub-block's best fit is found exactly: sorted, every
/// pair of boundaries splitting it into the three values is scored with
/// running sums, for both shifts.
pub(crate) fn quantize_iq1_s(block: &[f32], importance: Option<&[f32]>, out: &mut Vec<u8>) {
    const BLOCK: usize = 32;
    let x = super_block(block);
    let lattice = iq1s_lattice();
    let mut qs = [0u8; QK_K / 8];
    let mut qh = [0u16; QK_K / 32];
    let mut scales = [0.0f32; QK_K / BLOCK];
    let mut shifts = [1i8; QK_K / BLOCK];
    let mut max_scale = 0.0f32;
    for ib in 0..QK_K / BLOCK {
        let xb = &x[BLOCK * ib..BLOCK * (ib + 1)];
        let weight = weights_1bit(&x, importance, BLOCK * ib, BLOCK);
        let mut max = xb[0].abs();
        for v in &xb[1..] {
            max = max.max(v.abs());
        }
        if max < 1e-12 {
            continue;
        }
        let order = sorted(xb);
        let mut sumx = [0.0f32; BLOCK + 1];
        let mut sumw = [0.0f32; BLOCK + 1];
        for (j, &i) in order.iter().enumerate() {
            sumx[j + 1] = sumx[j] + weight[i] * xb[i];
            sumw[j + 1] = sumw[j] + weight[i];
        }
        let mut best_score = -f32::MAX;
        let mut scale = max;
        let (mut b1, mut b2, mut shift) = (usize::MAX, usize::MAX, 0i8);
        for i1 in 0..=BLOCK {
            for i2 in i1..=BLOCK {
                for (s, v) in [(1i8, &SHIFTED_UP), (-1i8, &SHIFTED_DOWN)] {
                    let sumqx = (sumx[i1] - sumx[0]) * v[0]
                        + (sumx[i2] - sumx[i1]) * v[1]
                        + (sumx[BLOCK] - sumx[i2]) * v[2];
                    let sumq2 = (sumw[i1] - sumw[0]) * v[0] * v[0]
                        + (sumw[i2] - sumw[i1]) * v[1] * v[1]
                        + (sumw[BLOCK] - sumw[i2]) * v[2] * v[2];
                    if sumq2 > 0.0 && sumqx * sumqx > best_score * sumq2 {
                        scale = sumqx / sumq2;
                        best_score = scale * sumqx;
                        (b1, b2, shift) = (i1, i2, s);
                    }
                }
            }
        }
        if b1 == usize::MAX || shift == 0 {
            continue;
        }
        let mut levels = [0i8; BLOCK];
        for (j, &i) in order.iter().enumerate() {
            levels[i] = if j < b1 {
                0
            } else if j < b2 {
                1
            } else {
                2
            };
        }
        if scale < 0.0 {
            for l in &mut levels {
                *l = 2 - *l;
            }
            scale = -scale;
            shift = -shift;
        }
        let shifted: &'static [f32; 3] = if shift == 1 {
            &SHIFTED_UP
        } else {
            &SHIFTED_DOWN
        };
        let mut indices = [0usize; BLOCK / 8];
        let mut all_on = true;
        for (k, slot) in indices.iter_mut().enumerate() {
            let at = 8 * k..8 * (k + 1);
            let (index, on) = snap_1bit(
                lattice,
                &mut levels[at.clone()],
                &xb[at.clone()],
                &weight[at],
                scale,
                shifted,
            );
            all_on &= on;
            *slot = index;
        }
        if !all_on && let Some(s) = refit_1bit(lattice, &indices, xb, &weight, |_| shifted) {
            scale = s;
        }
        let mut h = 0u16;
        for (k, &index) in indices.iter().enumerate() {
            qs[(BLOCK / 8) * ib + k] = (index & 255) as u8;
            h |= ((index >> 8) as u16) << (3 * k);
        }
        qh[ib] = h;
        scales[ib] = scale;
        shifts[ib] = shift;
        max_scale = max_scale.max(scale);
    }
    let mut d = 0.0f32;
    if max_scale > 0.0 {
        d = max_scale / 15.0;
        let id = 1.0 / d;
        for ib in 0..QK_K / BLOCK {
            let mut l = nearest_even(0.5 * (id * scales[ib] - 1.0)).clamp(0, 7) as u16;
            if shifts[ib] == -1 {
                l |= 8;
            }
            qh[ib] |= l << 12;
        }
        d *= 1.125;
    }
    out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
    out.extend_from_slice(&qs);
    for h in qh {
        out.extend_from_slice(&h.to_le_bytes());
    }
}

/// `IQ1_M`: [`quantize_iq1_s`] on sixteen 16-wide sub-blocks, each half of
/// a sub-block with its own shift, 3-bit scales, and the super-block scale
/// refitted to the chosen points and spread across the scale bits.
pub(crate) fn quantize_iq1_m(block: &[f32], importance: Option<&[f32]>, out: &mut Vec<u8>) {
    const BLOCK: usize = 16;
    const MASKS: [u8; 4] = [0x00, 0x80, 0x08, 0x88];
    let x = super_block(block);
    let lattice = iq1s_lattice();
    let mut qs = [0u8; QK_K / 8];
    let mut qh = [0u8; QK_K / 16];
    let mut sc = [0u16; 4];
    let mut scales = [0.0f32; QK_K / BLOCK];
    let mut shifts = [0usize; QK_K / BLOCK];
    let mut max_scale = 0.0f32;
    // The shift of half `k` of a sub-block with sign pattern `pattern`.
    let shift_of = |pattern: usize, k: usize| -> &'static [f32; 3] {
        let up = if k == 0 {
            pattern < 2
        } else {
            pattern.is_multiple_of(2)
        };
        if up { &SHIFTED_UP } else { &SHIFTED_DOWN }
    };
    for ib in 0..QK_K / BLOCK {
        let xb = &x[BLOCK * ib..BLOCK * (ib + 1)];
        let weight = weights_1bit(&x, importance, BLOCK * ib, BLOCK);
        let mut max = xb[0].abs();
        for v in &xb[1..] {
            max = max.max(v.abs());
        }
        if max < 1e-7 {
            continue;
        }
        let order = sorted(xb);
        let mut px = [[0.0f32; BLOCK + 1]; 2];
        let mut pw = [[0.0f32; BLOCK + 1]; 2];
        for (j, &i) in order.iter().enumerate() {
            let h = usize::from(i >= BLOCK / 2);
            px[h][j + 1] = px[h][j] + weight[i] * xb[i];
            px[1 - h][j + 1] = px[1 - h][j];
            pw[h][j + 1] = pw[h][j] + weight[i];
            pw[1 - h][j + 1] = pw[1 - h][j];
        }
        let txs = [px[0][BLOCK], px[1][BLOCK]];
        let tws = [pw[0][BLOCK], pw[1][BLOCK]];
        let (p, m) = (&SHIFTED_UP, &SHIFTED_DOWN);
        let (p2, m2) = (p[2], m[2]);
        let cp1 = p[0] * p[0] - p[1] * p[1];
        let cp2 = p[1] * p[1] - p[2] * p[2];
        let cm1 = m[0] * m[0] - m[1] * m[1];
        let cm2 = m[1] * m[1] - m[2] * m[2];
        let mut best_score = -f32::MAX;
        let mut scale = max;
        let (mut b1, mut b2, mut best_k) = (usize::MAX, usize::MAX, usize::MAX);
        for i1 in 0..=BLOCK {
            for i2 in i1..=BLOCK {
                let (mut qx_p, mut qx_m, mut q2_p, mut q2_m) =
                    ([0.0f32; 2], [0.0f32; 2], [0.0f32; 2], [0.0f32; 2]);
                for h in 0..2 {
                    let sx = px[h][i1] + px[h][i2];
                    qx_p[h] = txs[h] * p2 - sx;
                    qx_m[h] = txs[h] * m2 - sx;
                    q2_p[h] = tws[h] * p2 * p2 + pw[h][i1] * cp1 + pw[h][i2] * cp2;
                    q2_m[h] = tws[h] * m2 * m2 + pw[h][i1] * cm1 + pw[h][i2] * cm2;
                }
                let sumqx = [
                    qx_p[0] + qx_p[1],
                    qx_p[0] + qx_m[1],
                    qx_m[0] + qx_p[1],
                    qx_m[0] + qx_m[1],
                ];
                let sumq2 = [
                    q2_p[0] + q2_p[1],
                    q2_p[0] + q2_m[1],
                    q2_m[0] + q2_p[1],
                    q2_m[0] + q2_m[1],
                ];
                for k in 0..4 {
                    if sumq2[k] > 0.0 && sumqx[k] * sumqx[k] > best_score * sumq2[k] {
                        scale = sumqx[k] / sumq2[k];
                        best_score = scale * sumqx[k];
                        (b1, b2, best_k) = (i1, i2, k);
                    }
                }
            }
        }
        if b1 == usize::MAX {
            continue;
        }
        let mut levels = [0i8; BLOCK];
        for (j, &i) in order.iter().enumerate() {
            levels[i] = if j < b1 {
                0
            } else if j < b2 {
                1
            } else {
                2
            };
        }
        if scale < 0.0 {
            for l in &mut levels {
                *l = 2 - *l;
            }
            scale = -scale;
            best_k = 3 - best_k;
        }
        let mut indices = [0usize; 2];
        let mut all_on = true;
        for (k, slot) in indices.iter_mut().enumerate() {
            let at = 8 * k..8 * (k + 1);
            let (index, on) = snap_1bit(
                lattice,
                &mut levels[at.clone()],
                &xb[at.clone()],
                &weight[at],
                scale,
                shift_of(best_k, k),
            );
            all_on &= on;
            *slot = index;
        }
        if !all_on
            && let Some(s) = refit_1bit(lattice, &indices, xb, &weight, |k| shift_of(best_k, k))
        {
            scale = s;
        }
        qs[2 * ib] = (indices[0] & 255) as u8;
        qs[2 * ib + 1] = (indices[1] & 255) as u8;
        qh[ib] = ((indices[0] >> 8) | ((indices[1] >> 8) << 4)) as u8;
        scales[ib] = scale;
        shifts[ib] = best_k;
        max_scale = max_scale.max(scale);
    }
    if max_scale > 0.0 {
        let mut d = max_scale / 15.0;
        let id = 1.0 / d;
        let (mut sumqx, mut sumq2) = (0.0f32, 0.0f32);
        for ib in 0..QK_K / BLOCK {
            let l = nearest_even(0.5 * (id * scales[ib] - 1.0)).clamp(0, 7);
            sc[ib / 4] |= (l as u16) << (3 * (ib % 4));
            qh[ib] |= MASKS[shifts[ib]];
            let xb = &x[BLOCK * ib..BLOCK * (ib + 1)];
            let weight = weights_1bit(&x, importance, BLOCK * ib, BLOCK);
            for k in 0..2 {
                let shifted = shift_of(shifts[ib], k);
                let index = qs[2 * ib + k] as usize | (((qh[ib] as usize) << (8 - 4 * k)) & 0x700);
                let point = &lattice.points[index];
                for j in 0..8 {
                    let w = weight[8 * k + j];
                    let q = shifted[((point[j] - 1) / 2) as usize] * (2 * l + 1) as f32;
                    sumqx += w * q * xb[8 * k + j];
                    sumq2 += w * q * q;
                }
            }
        }
        if sumq2 > 0.0 {
            d = sumqx / sumq2;
        }
        let bits = f16::from_f32(d * 1.1125).to_bits();
        sc[0] |= (bits & 0x000f) << 12;
        sc[1] |= (bits & 0x00f0) << 8;
        sc[2] |= (bits & 0x0f00) << 4;
        sc[3] |= bits & 0xf000;
    }
    out.extend_from_slice(&qs);
    out.extend_from_slice(&qh);
    for s in sc {
        out.extend_from_slice(&s.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The derived lattices index their points the way the stored
    /// codebooks do: the first entries of each, as level positions.
    #[test]
    fn the_lattices_index_the_codebooks_in_order() {
        let iq2 = iq2s_lattice();
        let positions: Vec<usize> = iq2
            .points
            .iter()
            .take(8)
            .map(|p| {
                let levels: Vec<i8> = p.iter().map(|&v| (v - 1) / 2).collect();
                iq2.position(&levels)
            })
            .collect();
        assert_eq!(positions, [0, 2, 5, 8, 10, 17, 20, 22]);
        let iq3 = iq3s_lattice();
        let positions: Vec<usize> = iq3
            .points
            .iter()
            .take(8)
            .map(|p| {
                let levels: Vec<i8> = p.iter().take(4).map(|&v| (v - 1) / 2).collect();
                iq3.position(&levels)
            })
            .collect();
        assert_eq!(positions, [0, 1, 2, 5, 7, 8, 9, 10]);
        // Every position off the codebook has somewhere to go.
        for lattice in [iq2, iq3] {
            for &m in &lattice.map {
                if m < 0 {
                    assert!(lattice.neighbours[(-m - 1) as usize] > 0);
                }
            }
        }
    }
}
