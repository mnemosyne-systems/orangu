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

//! Prompt attention on the CPU as tiles — a block of queries against a
//! block of keys, with the softmax carried across key blocks online (the
//! "flash attention" shape).
//!
//! [`crate::engine::attention::multi_head_attention`] takes one query at a
//! time and walks its whole window, so a prompt of `T` tokens reads every
//! cached K and V row about `T` times, and each read feeds one query's
//! dot products. At 2048 tokens on `gemma-4-E2B` that was a quarter of the
//! prefill (`doc/PERF-ALL.md`, task 3). Here a task owns `TILE_Q`
//! consecutive queries and every head of one K/V group — `TILE_Q × group`
//! rows sharing the same keys — and each K and V row is loaded once per
//! four rows into registers (`qk_4x4`, `pv_4x16`) rather than once per
//! query head.
//!
//! **Not bit-identical to the one-query loop**: the exponentials are taken
//! against a running maximum and rescaled as it grows, and the sums are in
//! a different order. Within float reassociation of it, which the tests
//! below hold it to; the dispatcher (`attention::attention`) uses it only
//! for multi-token passes wide enough to gain, and `ORANGU_ATTENTION_TILED=0`
//! turns it off.

use rayon::prelude::*;

use crate::engine::kv_cache::LayerCache;

/// Keys per step of the online softmax: 64 rows of K (or V) at
/// `head_dim = 256` are 64 KiB, the size of the A720's L1 data cache — the
/// block is read once from L2 per query tile and then from L1 for every
/// row group.
const TILE_K: usize = 64;

/// The most queries one task takes. More rows per task means fewer passes
/// over K and V; fewer means more tasks for the pool — see [`tile_q`].
const MAX_TILE_Q: usize = 16;

/// Whether the tiled kernel may take a call — `ORANGU_ATTENTION_TILED=0`
/// turns it off (the one-query loop is then the only CPU kernel, as before).
pub fn enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| crate::engine::env::flag_on_unless_disabled("ORANGU_ATTENTION_TILED"))
}

/// Multi-token passes at least this wide take the tiled kernel. Below it a
/// pass has too few query tiles to spread over the pool, and the one-query
/// loop, which parallelises per token, keeps every core busy instead.
/// `ORANGU_ATTENTION_TILED_MIN_TOKENS` overrides it for a sweep.
pub fn min_tokens() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("ORANGU_ATTENTION_TILED_MIN_TOKENS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(32)
    })
}

/// Whether [`attention_tiled`] handles this shape: the value kernel works
/// in sixteen-float column blocks.
pub fn supports(head_dim: usize) -> bool {
    head_dim > 0 && head_dim.is_multiple_of(16)
}

/// Queries per task: as many as [`MAX_TILE_Q`], but no more than keep two
/// tasks per worker thread, and a multiple of four.
fn tile_q(n_tokens: usize) -> usize {
    let threads = rayon::current_num_threads().max(1);
    let per_task = n_tokens.div_ceil(2 * threads);
    (per_task.div_ceil(4) * 4).clamp(4, MAX_TILE_Q)
}

/// Per-task scratch, reused across the tasks a worker runs.
#[derive(Default)]
struct Scratch {
    /// Scaled scores, then probabilities: `[rows, TILE_K]`.
    s: Vec<f32>,
    /// Running output: `[rows, head_dim]`.
    o: Vec<f32>,
    /// Running maximum per row.
    m: Vec<f32>,
    /// Running sum of exponentials per row.
    l: Vec<f32>,
    /// Each row's window, `(start, end)` inclusive; `end < start` is empty.
    win: Vec<(usize, usize)>,
}

/// Multi-head attention for `n_tokens` queries against `cache`, into `out`
/// (`[n_tokens, n_head * head_dim]`, fully written) — the same contract as
/// [`crate::engine::attention::multi_head_attention`], computed in tiles.
#[allow(clippy::too_many_arguments)]
pub fn attention_tiled(
    out: &mut [f32],
    q: &[f32],
    cache: &LayerCache,
    n_head: usize,
    group_size: usize,
    head_dim: usize,
    scale: f32,
    window: impl Fn(usize) -> (usize, usize) + Sync,
) {
    debug_assert_eq!(out.len(), q.len());
    debug_assert!(supports(head_dim));
    let row = n_head * head_dim;
    debug_assert!(row > 0 && out.len().is_multiple_of(row));
    let n_tokens = out.len() / row;
    let n_kv = n_head.div_ceil(group_size);
    let tq = tile_q(n_tokens);

    out.par_chunks_mut(tq * row).enumerate().for_each_init(
        Scratch::default,
        |scratch, (block, out_block)| {
            let t0 = block * tq;
            let n_q = out_block.len() / row;
            for kv_head in 0..n_kv {
                let h0 = kv_head * group_size;
                let heads = ((kv_head + 1) * group_size).min(n_head) - h0;
                tile(
                    scratch, out_block, q, cache, t0, n_q, h0, heads, kv_head, row, head_dim,
                    scale, &window,
                );
            }
        },
    );
}

/// One task: queries `t0 .. t0 + n_q`, heads `h0 .. h0 + heads` (one K/V
/// head's group). Rows are token-major — row `r` is token `r / heads`,
/// head `h0 + r % heads` — so the rows of one token share a window.
#[allow(clippy::too_many_arguments)]
fn tile(
    sc: &mut Scratch,
    out_block: &mut [f32],
    q: &[f32],
    cache: &LayerCache,
    t0: usize,
    n_q: usize,
    h0: usize,
    heads: usize,
    kv_head: usize,
    row: usize,
    head_dim: usize,
    scale: f32,
    window: &(impl Fn(usize) -> (usize, usize) + Sync),
) {
    let rows = n_q * heads;
    sc.s.resize(rows * TILE_K, 0.0);
    sc.o.clear();
    sc.o.resize(rows * head_dim, 0.0);
    sc.m.clear();
    sc.m.resize(rows, f32::NEG_INFINITY);
    sc.l.clear();
    sc.l.resize(rows, 0.0);
    sc.win.clear();
    let (mut k_lo, mut k_hi) = (usize::MAX, 0usize);
    for i in 0..n_q {
        let (ws, we) = window(t0 + i);
        if we >= ws {
            k_lo = k_lo.min(ws);
            k_hi = k_hi.max(we + 1);
        }
        sc.win.push((ws, we));
    }
    // The query row `r` reads, where it lives in `q`.
    let q_row = |r: usize| {
        let (i, hi) = (r / heads, r % heads);
        let at = (t0 + i) * row + (h0 + hi) * head_dim;
        &q[at..at + head_dim]
    };

    let mut k0 = k_lo;
    while k0 < k_hi {
        let kt = (k_hi - k0).min(TILE_K);
        // Scores for every row against this block's keys, scaled.
        let s = &mut sc.s;
        let mut r = 0;
        while r + 4 <= rows {
            let qs = [q_row(r), q_row(r + 1), q_row(r + 2), q_row(r + 3)];
            scores_rows(s, r, qs, cache, k0, kt, kv_head, head_dim, scale);
            r += 4;
        }
        while r < rows {
            let qr = q_row(r);
            for j in 0..kt {
                let k = cache.key_at(k0 + j, kv_head, head_dim);
                s[r * TILE_K + j] = crate::engine::tensor::dot(qr, k) * scale;
            }
            r += 1;
        }
        // Online softmax per row: mask to the row's window, fold the
        // block's maximum into the running one, rescale what is carried.
        for r in 0..rows {
            let (ws, we) = sc.win[r / heads];
            let sr = &mut s[r * TILE_K..r * TILE_K + kt];
            // The block's positions this row may see: `lo..hi` of `0..kt`.
            let lo = ws.saturating_sub(k0).min(kt);
            let hi = if we < ws {
                0
            } else {
                (we + 1).saturating_sub(k0).min(kt)
            };
            if lo >= hi {
                // Nothing here for this row: its probabilities are zero and
                // its carried state is unchanged.
                sr.fill(0.0);
                continue;
            }
            let block_max = sr[lo..hi].iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let m_new = sc.m[r].max(block_max);
            let correction = (sc.m[r] - m_new).exp();
            let mut sum = 0f32;
            for (j, v) in sr.iter_mut().enumerate() {
                *v = if (lo..hi).contains(&j) {
                    let p = (*v - m_new).exp();
                    sum += p;
                    p
                } else {
                    0.0
                };
            }
            sc.l[r] = sc.l[r] * correction + sum;
            sc.m[r] = m_new;
            if correction != 1.0 {
                sc.o[r * head_dim..(r + 1) * head_dim]
                    .iter_mut()
                    .for_each(|v| *v *= correction);
            }
        }
        // Output += probabilities · V for this block.
        let mut r = 0;
        while r + 4 <= rows {
            values_rows(&mut sc.o, &sc.s, r, 4, cache, k0, kt, kv_head, head_dim);
            r += 4;
        }
        if r < rows {
            values_rows(
                &mut sc.o,
                &sc.s,
                r,
                rows - r,
                cache,
                k0,
                kt,
                kv_head,
                head_dim,
            );
        }
        k0 += kt;
    }

    // Normalize into the caller's rows; an empty window stays zero.
    for r in 0..rows {
        let (i, hi) = (r / heads, r % heads);
        let at = i * row + (h0 + hi) * head_dim;
        let dst = &mut out_block[at..at + head_dim];
        let l = sc.l[r];
        if l > 0.0 {
            let inv = 1.0 / l;
            for (d, o) in dst.iter_mut().zip(&sc.o[r * head_dim..(r + 1) * head_dim]) {
                *d = o * inv;
            }
        } else {
            dst.fill(0.0);
        }
    }
}

/// Four rows' scaled scores against keys `k0 .. k0 + kt`, into `s`'s rows
/// `r .. r + 4`: four keys at a time through [`qk_4x4`], the rest one at a
/// time.
#[allow(clippy::too_many_arguments)]
fn scores_rows(
    s: &mut [f32],
    r: usize,
    qs: [&[f32]; 4],
    cache: &LayerCache,
    k0: usize,
    kt: usize,
    kv_head: usize,
    head_dim: usize,
    scale: f32,
) {
    let mut j = 0;
    while j + 4 <= kt {
        let ks: [&[f32]; 4] = std::array::from_fn(|c| cache.key_at(k0 + j + c, kv_head, head_dim));
        let got = qk_4x4(qs, ks);
        for (a, got_a) in got.iter().enumerate() {
            for (c, v) in got_a.iter().enumerate() {
                s[(r + a) * TILE_K + j + c] = v * scale;
            }
        }
        j += 4;
    }
    while j < kt {
        let k = cache.key_at(k0 + j, kv_head, head_dim);
        for (a, qa) in qs.iter().enumerate() {
            s[(r + a) * TILE_K + j] = crate::engine::tensor::dot(qa, k) * scale;
        }
        j += 1;
    }
}

/// `o`'s rows `r .. r + n` (`n ≤ 4`) += `s`'s probabilities for keys
/// `k0 .. k0 + kt` times their V rows, sixteen columns at a time.
#[allow(clippy::too_many_arguments)]
fn values_rows(
    o: &mut [f32],
    s: &[f32],
    r: usize,
    n: usize,
    cache: &LayerCache,
    k0: usize,
    kt: usize,
    kv_head: usize,
    head_dim: usize,
) {
    let mut col = 0;
    while col < head_dim {
        pv_4x16(o, s, r, n, cache, k0, kt, kv_head, head_dim, col);
        col += 16;
    }
}

/// Sixteen dot products — four query rows against four keys — with each
/// operand loaded once per step.
fn qk_4x4(qs: [&[f32]; 4], ks: [&[f32]; 4]) -> [[f32; 4]; 4] {
    #[cfg(target_arch = "aarch64")]
    {
        qk_4x4_neon(qs, ks)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut acc = [[0f32; 4]; 4];
        for (a, qa) in qs.iter().enumerate() {
            for (c, kc) in ks.iter().enumerate() {
                acc[a][c] = crate::engine::tensor::dot(qa, kc);
            }
        }
        acc
    }
}

#[cfg(target_arch = "aarch64")]
fn qk_4x4_neon(qs: [&[f32]; 4], ks: [&[f32]; 4]) -> [[f32; 4]; 4] {
    use std::arch::aarch64::*;
    let n = qs[0].len();
    debug_assert!(qs.iter().chain(ks.iter()).all(|v| v.len() == n));
    debug_assert!(n.is_multiple_of(4));
    // Safety: every load is at `i + 4 <= n`, and each operand is `n` long;
    // NEON is unconditionally present on this target.
    unsafe {
        let mut acc = [[vdupq_n_f32(0.0); 4]; 4];
        let mut i = 0;
        while i + 4 <= n {
            let k = [
                vld1q_f32(ks[0].as_ptr().add(i)),
                vld1q_f32(ks[1].as_ptr().add(i)),
                vld1q_f32(ks[2].as_ptr().add(i)),
                vld1q_f32(ks[3].as_ptr().add(i)),
            ];
            for a in 0..4 {
                let qv = vld1q_f32(qs[a].as_ptr().add(i));
                for c in 0..4 {
                    acc[a][c] = vfmaq_f32(acc[a][c], qv, k[c]);
                }
            }
            i += 4;
        }
        let mut out = [[0f32; 4]; 4];
        for a in 0..4 {
            for c in 0..4 {
                out[a][c] = vaddvq_f32(acc[a][c]);
            }
        }
        out
    }
}

/// `o[r + a][col .. col + 16] += Σ_j s[r + a][j] · V[k0 + j][col .. col + 16]`
/// for `a < n`: the sixteen columns of up to four rows held in registers
/// across the whole key block, each V row's sixteen columns loaded once.
#[allow(clippy::too_many_arguments)]
fn pv_4x16(
    o: &mut [f32],
    s: &[f32],
    r: usize,
    n: usize,
    cache: &LayerCache,
    k0: usize,
    kt: usize,
    kv_head: usize,
    head_dim: usize,
    col: usize,
) {
    debug_assert!(n <= 4 && col + 16 <= head_dim);
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        // Safety: `o` holds `rows * head_dim` and `r + n <= rows`; `s` holds
        // `rows * TILE_K` and `j < kt <= TILE_K`; each V row is `head_dim`
        // long and `col + 16 <= head_dim`. NEON is always present here.
        unsafe {
            let mut acc = [[vdupq_n_f32(0.0); 4]; 4];
            for (a, regs) in acc.iter_mut().enumerate().take(n) {
                let p = o.as_ptr().add((r + a) * head_dim + col);
                for (v, reg) in regs.iter_mut().enumerate() {
                    *reg = vld1q_f32(p.add(4 * v));
                }
            }
            for j in 0..kt {
                let vp = cache.value_at(k0 + j, kv_head, head_dim).as_ptr().add(col);
                let v = [
                    vld1q_f32(vp),
                    vld1q_f32(vp.add(4)),
                    vld1q_f32(vp.add(8)),
                    vld1q_f32(vp.add(12)),
                ];
                for (a, regs) in acc.iter_mut().enumerate().take(n) {
                    let w = *s.get_unchecked((r + a) * TILE_K + j);
                    for (reg, vc) in regs.iter_mut().zip(v) {
                        *reg = vfmaq_n_f32(*reg, vc, w);
                    }
                }
            }
            for (a, regs) in acc.iter().enumerate().take(n) {
                let p = o.as_mut_ptr().add((r + a) * head_dim + col);
                for (v, reg) in regs.iter().enumerate() {
                    vst1q_f32(p.add(4 * v), *reg);
                }
            }
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    for j in 0..kt {
        let v = &cache.value_at(k0 + j, kv_head, head_dim)[col..col + 16];
        for a in 0..n {
            let w = s[(r + a) * TILE_K + j];
            let dst = &mut o[(r + a) * head_dim + col..(r + a) * head_dim + col + 16];
            for (d, x) in dst.iter_mut().zip(v) {
                *d += w * x;
            }
        }
    }
}

/// Rows per task of [`attention_mixed`]: query tokens times the heads of
/// one K/V group. Thirty-two rows of 256 keys' scores and probabilities are
/// 48 KiB, and the task's value tile `32 × head_dim` another 32–64 KiB.
const MIXED_ROWS: usize = 32;

/// Whether [`attention_mixed`] may take a call: the CPU has `i8mm` and
/// `bf16`, the head width is a multiple of eight, and
/// `ORANGU_ATTENTION_MIXED=0` has not turned it off (the `f32` tiles, or
/// the one-query loop, then take the call).
pub fn mixed_available(head_dim: usize) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    head_dim.is_multiple_of(8)
        && *ON.get_or_init(|| {
            crate::engine::vecdot::have_i8mm()
                && crate::engine::vecdot::have_bf16mm()
                && crate::engine::env::flag_on_unless_disabled("ORANGU_ATTENTION_MIXED")
        })
}

/// A pointer tasks write disjoint rectangles of `out` through: every
/// (token, head) row belongs to exactly one task.
#[derive(Clone, Copy)]
struct Sink(*mut f32);
// Safety: see the type — no two tasks write the same element.
unsafe impl Send for Sink {}
unsafe impl Sync for Sink {}

impl Sink {
    /// The row pointer, through a method so a closure captures the whole
    /// `Sink` (which is `Send`) rather than its raw-pointer field.
    fn at(self, offset: usize) -> *mut f32 {
        // Safety: the caller writes inside `out`; see the type.
        unsafe { self.0.add(offset) }
    }
}

/// `n` rows of `dim` as [`PairedI8`], the pairs quantized in parallel —
/// the same layout and rounding as `PairedI8::quantize`, which runs one
/// row at a time and was measurable against a whole prompt's keys.
fn paired_i8_par(n: usize, dim: usize, rows: &[f32]) -> crate::engine::vecdot::PairedI8 {
    let pairs = n.div_ceil(2);
    let mut data = vec![0i8; pairs * 2 * dim];
    let mut scales = vec![0f32; pairs * 2];
    data.par_chunks_mut(2 * dim)
        .zip(scales.par_chunks_mut(2))
        .enumerate()
        .for_each(|(pair, (dst, sc))| {
            for half in 0..2 {
                let i = 2 * pair + half;
                if i >= n {
                    break;
                }
                let r = &rows[i * dim..(i + 1) * dim];
                let max = r.iter().fold(0f32, |m, v| m.max(v.abs()));
                let scale = max / 127.0;
                let inv = if max > 0.0 { 1.0 / scale } else { 0.0 };
                sc[half] = scale;
                for (c, chunk) in r.as_chunks::<8>().0.iter().enumerate() {
                    let d = &mut dst[c * 16 + half * 8..][..8];
                    for (d, v) in d.iter_mut().zip(chunk) {
                        *d = (v * inv).round().clamp(-127.0, 127.0) as i8;
                    }
                }
            }
        });
    crate::engine::vecdot::PairedI8 {
        data,
        scales,
        dim,
        pairs,
    }
}

/// Prompt attention with the scores in `int8` and the value product in
/// `bf16` — the SageAttention recipe the picture transformers already use
/// (`image::transformer::step_attention`), for grouped heads, a paged
/// cache and sliding windows.
///
/// Per call, each K/V head's keys over the pass's window union are taken
/// less their per-channel mean (which moves every score of a query by the
/// same amount, so no softmax changes, and centres the rows for `int8`)
/// and quantized per row for `smmla`; its values are packed transposed in
/// `bf16` for `bfmmla`. Each task then takes [`MIXED_ROWS`] rows — query
/// tokens × the group's heads, which share keys — against only the keys
/// their windows cover: `smmla` scores, an `f32` softmax written straight
/// into `bfmmla`'s operand (`PackedBf16::set_row_exp`), and one
/// `bf16_tiles_at` for the value product.
///
/// Within `int8`/`bf16` rounding of the `f32` loop, not float
/// reassociation — the tests hold it to that, and the end-to-end check in
/// `doc/PERF-ALL.md` (task 3) to the model's answers.
#[allow(clippy::too_many_arguments)]
pub fn attention_mixed(
    out: &mut [f32],
    q: &[f32],
    cache: &LayerCache,
    n_head: usize,
    group_size: usize,
    head_dim: usize,
    scale: f32,
    window: impl Fn(usize) -> (usize, usize) + Sync,
) {
    use crate::engine::vecdot::{PackedBf16, bf16_tiles_at, i8_scores_4rows_in};
    debug_assert_eq!(out.len(), q.len());
    debug_assert!(head_dim.is_multiple_of(8));
    let row = n_head * head_dim;
    let n_tokens = out.len() / row;
    let n_kv = n_head.div_ceil(group_size);
    let wins: Vec<(usize, usize)> = (0..n_tokens).map(&window).collect();
    let (k_lo, k_hi) = wins
        .iter()
        .filter(|(ws, we)| we >= ws)
        .fold((usize::MAX, 0), |(lo, hi), &(ws, we)| {
            (lo.min(ws), hi.max(we + 1))
        });
    if k_lo >= k_hi {
        out.fill(0.0);
        return;
    }
    // Keys from an 8-aligned base, padded to a whole 8 with zero rows, so
    // every block's range below is whole pairs and whole `bf16` chunks.
    let base = k_lo / 8 * 8;
    let n = k_hi - base;
    let n_pad = n.div_ceil(8) * 8;

    let heads: Vec<(crate::engine::vecdot::PairedI8, PackedBf16)> = (0..n_kv)
        .map(|kv_head| {
            let mut keys = vec![0f32; n_pad * head_dim];
            let mut vals = vec![0f32; n_pad * head_dim];
            keys.par_chunks_mut(head_dim)
                .zip(vals.par_chunks_mut(head_dim))
                .take(n)
                .enumerate()
                .for_each(|(i, (k, v))| {
                    k.copy_from_slice(cache.key_at(base + i, kv_head, head_dim));
                    v.copy_from_slice(cache.value_at(base + i, kv_head, head_dim));
                });
            let mut mean = vec![0f32; head_dim];
            for r in keys[..n * head_dim].chunks(head_dim) {
                for (m, v) in mean.iter_mut().zip(r) {
                    *m += v;
                }
            }
            let inv_n = 1.0 / n as f32;
            mean.iter_mut().for_each(|m| *m *= inv_n);
            keys[..n * head_dim].par_chunks_mut(head_dim).for_each(|r| {
                for (v, m) in r.iter_mut().zip(&mean) {
                    *v -= m;
                }
            });
            (
                paired_i8_par(n_pad, head_dim, &keys),
                PackedBf16::pack_transposed(head_dim, n_pad, &vals, head_dim),
            )
        })
        .collect();

    let tokens_per_task = (MIXED_ROWS / group_size.min(n_head)).max(1);
    let n_blocks = n_tokens.div_ceil(tokens_per_task);
    let sink = Sink(out.as_mut_ptr());
    let zeros = vec![0f32; head_dim];
    (0..n_kv * n_blocks).into_par_iter().for_each_init(
        || (Vec::<f32>::new(), Vec::<i32>::new(), Vec::<f32>::new()),
        |(scores, ints, tile), task| {
            let (kv_head, b) = (task / n_blocks, task % n_blocks);
            let (kq, vt) = &heads[kv_head];
            let t0 = b * tokens_per_task;
            let nt = tokens_per_task.min(n_tokens - t0);
            let h0 = kv_head * group_size;
            let g = ((kv_head + 1) * group_size).min(n_head) - h0;
            let rows = nt * g;
            let write = |r: usize, values: &mut dyn FnMut(usize) -> f32| {
                let (t, h) = (t0 + r / g, h0 + r % g);
                // Safety: row (t, h) is this task's alone, and inside `out`.
                unsafe {
                    let dst = sink.at(t * row + h * head_dim);
                    for d in 0..head_dim {
                        *dst.add(d) = values(d);
                    }
                }
            };
            let (lo, hi) = wins[t0..t0 + nt]
                .iter()
                .filter(|(ws, we)| we >= ws)
                .fold((usize::MAX, 0), |(lo, hi), &(ws, we)| {
                    (lo.min(ws), hi.max(we + 1))
                });
            if lo >= hi {
                for r in 0..rows {
                    write(r, &mut |_| 0.0);
                }
                return;
            }
            let c_lo = (lo - base) / 8 * 8;
            let c_hi = ((hi - base).div_ceil(8) * 8).min(n_pad);
            let nb = c_hi - c_lo;

            // The block's queries in `int8`, padded to a whole quad with
            // zero rows whose scores are never read.
            let padded = rows.div_ceil(4) * 4;
            let q_row = |r: usize| -> &[f32] {
                if r >= rows {
                    return &zeros;
                }
                let (t, h) = (t0 + r / g, h0 + r % g);
                &q[t * row + h * head_dim..t * row + (h + 1) * head_dim]
            };
            let qs = crate::engine::vecdot::PairedI8::quantize(padded, head_dim, q_row);
            scores.resize(nb, 0.0);
            ints.resize(4 * nb, 0);
            let mut probs = PackedBf16::zeroed(rows, nb);
            let mut inv = vec![0f32; rows];
            for quad in 0..padded / 4 {
                let (i0, rest) = ints.split_at_mut(nb);
                let (i1, rest) = rest.split_at_mut(nb);
                let (i2, i3) = rest.split_at_mut(nb);
                i8_scores_4rows_in(
                    &qs,
                    2 * quad,
                    2 * quad + 1,
                    kq,
                    c_lo / 2..c_hi / 2,
                    [i0, i1, i2, i3],
                );
                for r in 0..4 {
                    let ri = quad * 4 + r;
                    if ri >= rows {
                        break;
                    }
                    let (ws, we) = wins[t0 + ri / g];
                    let sq = qs.scales[ri] * scale;
                    let mut max = f32::NEG_INFINITY;
                    for (j, s) in scores.iter_mut().enumerate() {
                        let pos = base + c_lo + j;
                        *s = if pos >= ws && pos <= we {
                            let v = ints[r * nb + j] as f32 * sq * kq.scales[c_lo + j];
                            max = max.max(v);
                            v
                        } else {
                            f32::NEG_INFINITY
                        };
                    }
                    if max == f32::NEG_INFINITY {
                        // An empty window: the row stays zero.
                        continue;
                    }
                    inv[ri] = 1.0 / probs.set_row_exp(ri, scores, max);
                }
            }
            tile.resize(probs.rows * vt.rows, 0.0);
            bf16_tiles_at(&probs, vt, c_lo / 4, tile);
            for (r, &w) in inv.iter().enumerate().take(rows) {
                let src = &tile[r * vt.rows..r * vt.rows + head_dim];
                write(r, &mut |d| src[d] * w);
            }
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::attention::multi_head_attention;

    fn build(n_pos: usize, kv_dim: usize, seed: &mut u32) -> crate::engine::kv_cache::KvCache {
        let mut next = || {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 17;
            *seed ^= *seed << 5;
            (*seed as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let mut cache = crate::engine::kv_cache::KvCache::new(1, n_pos, kv_dim);
        for _ in 0..n_pos {
            let k: Vec<f32> = (0..kv_dim).map(|_| next()).collect();
            let v: Vec<f32> = (0..kv_dim).map(|_| next()).collect();
            cache.layers[0].push(&k, &v);
        }
        cache
    }

    /// The tiled kernel against the one-query loop, to float
    /// reassociation: grouped and ungrouped heads, a group that does not
    /// divide the heads, a pass that is not a whole number of query tiles
    /// or key blocks, a prefix already cached (`start_pos`), sliding
    /// windows narrower and wider than a key block, and a window that is
    /// empty for every query.
    #[test]
    fn tiled_attention_matches_the_one_query_loop() {
        type Win = fn(usize, usize, usize) -> (usize, usize);
        let causal: Win = |start, _w, t| (0, start + t);
        let sliding: Win = |start, w, t| ((start + t + 1).saturating_sub(w), start + t);
        let empty: Win = |_, _, _| (5, 4);
        for &(n_head, n_head_kv) in &[(8usize, 1usize), (8, 2), (4, 4), (6, 4), (1, 1)] {
            for &head_dim in &[16usize, 64, 256] {
                for &(start_pos, n_tokens) in &[(0usize, 1usize), (0, 37), (100, 70), (0, 200)] {
                    for (name, win, w) in [
                        ("causal", causal, 0usize),
                        ("sliding 20", sliding, 20),
                        ("sliding 150", sliding, 150),
                        ("empty", empty, 0),
                    ] {
                        let group_size = n_head.div_ceil(n_head_kv);
                        let n_kv = n_head.div_ceil(group_size);
                        let mut seed = 0xBEEF_u32 ^ (n_head * 131 + head_dim * 7 + n_tokens) as u32;
                        let kv = build(start_pos + n_tokens, n_kv * head_dim, &mut seed);
                        let cache = &kv.layers[0];
                        let row = n_head * head_dim;
                        let q: Vec<f32> = (0..n_tokens * row)
                            .map(|i| ((i * 37 % 29) as f32 - 14.0) / 7.0)
                            .collect();
                        let window = |t: usize| win(start_pos, w, t);
                        let scale = 1.0 / (head_dim as f32).sqrt();

                        let mut want = vec![0f32; n_tokens * row];
                        multi_head_attention(
                            &mut want, &q, cache, n_head, group_size, head_dim, scale, window,
                        );
                        let mut got = vec![9f32; n_tokens * row];
                        attention_tiled(
                            &mut got, &q, cache, n_head, group_size, head_dim, scale, window,
                        );
                        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                            assert!(
                                (g - w).abs() <= 2e-5 * (1.0 + w.abs()),
                                "{n_head}/{n_head_kv} heads, head_dim {head_dim}, \
                                 {start_pos}+{n_tokens} tokens, {name}: element {i} \
                                 tiled {g} against {w}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The `int8`/`bf16` kernel against the `f32` loop: every output within
    /// a few hundredths of the same weighted mean of values in `[-1, 1]`,
    /// and the mean error a few thousandths — the rounding of an `int8`
    /// score and a `bf16` probability, not a wrong key or a lost window.
    /// Same shapes as above, windows included.
    #[test]
    fn mixed_attention_matches_the_one_query_loop_to_its_rounding() {
        if !mixed_available(16) {
            return;
        }
        type Win = fn(usize, usize, usize) -> (usize, usize);
        let causal: Win = |start, _w, t| (0, start + t);
        let sliding: Win = |start, w, t| ((start + t + 1).saturating_sub(w), start + t);
        let empty: Win = |_, _, _| (5, 4);
        for &(n_head, n_head_kv) in &[(8usize, 1usize), (8, 2), (4, 4), (6, 4), (1, 1)] {
            for &head_dim in &[16usize, 64, 256] {
                for &(start_pos, n_tokens) in &[(0usize, 1usize), (3, 37), (100, 70), (0, 200)] {
                    for (name, win, w) in [
                        ("causal", causal, 0usize),
                        ("sliding 20", sliding, 20),
                        ("sliding 150", sliding, 150),
                        ("empty", empty, 0),
                    ] {
                        let group_size = n_head.div_ceil(n_head_kv);
                        let n_kv = n_head.div_ceil(group_size);
                        let mut seed = 0xFACE_u32 ^ (n_head * 131 + head_dim * 7 + n_tokens) as u32;
                        let kv = build(start_pos + n_tokens, n_kv * head_dim, &mut seed);
                        let cache = &kv.layers[0];
                        let row = n_head * head_dim;
                        let q: Vec<f32> = (0..n_tokens * row)
                            .map(|i| ((i * 37 % 29) as f32 - 14.0) / 7.0)
                            .collect();
                        let window = |t: usize| win(start_pos, w, t);
                        let scale = 1.0 / (head_dim as f32).sqrt();

                        let mut want = vec![0f32; n_tokens * row];
                        multi_head_attention(
                            &mut want, &q, cache, n_head, group_size, head_dim, scale, window,
                        );
                        let mut got = vec![9f32; n_tokens * row];
                        attention_mixed(
                            &mut got, &q, cache, n_head, group_size, head_dim, scale, window,
                        );
                        let mut total = 0f64;
                        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                            assert!(
                                (g - w).abs() <= 0.05,
                                "{n_head}/{n_head_kv} heads, head_dim {head_dim}, \
                                 {start_pos}+{n_tokens} tokens, {name}: element {i} \
                                 mixed {g} against {w}"
                            );
                            total += (g - w).abs() as f64;
                        }
                        let mean = total / got.len() as f64;
                        assert!(
                            mean <= 0.005,
                            "{n_head}/{n_head_kv} heads, head_dim {head_dim}, \
                             {start_pos}+{n_tokens} tokens, {name}: mean error {mean}"
                        );
                    }
                }
            }
        }
    }

    /// Both kernels at `gemma-4-E2B`'s full-attention shape (8 heads, one
    /// K/V head, `head_dim` 512) and its sliding one (256, window 512), a
    /// 2048-token pass over an empty cache. Prints the times; run with
    /// `--ignored --nocapture`.
    #[test]
    #[ignore = "a timing, not a check"]
    fn tiled_attention_timing() {
        for &(head_dim, n_swa) in &[(512usize, 0usize), (256, 512)] {
            for &n_tokens in &[512usize, 2048] {
                let mut seed = 7u32;
                let kv = build(n_tokens, head_dim, &mut seed);
                let cache = &kv.layers[0];
                let (n_head, group_size) = (8, 8);
                let row = n_head * head_dim;
                let q: Vec<f32> = (0..n_tokens * row)
                    .map(|i| ((i % 19) as f32 - 9.0) / 9.0)
                    .collect();
                let window = |t: usize| {
                    if n_swa == 0 {
                        (0, t)
                    } else {
                        ((t + 1).saturating_sub(n_swa), t)
                    }
                };
                let scale = 1.0 / (head_dim as f32).sqrt();
                let mut out = vec![0f32; n_tokens * row];
                let time = |f: &mut dyn FnMut()| {
                    f();
                    (0..3)
                        .map(|_| {
                            let t = std::time::Instant::now();
                            f();
                            t.elapsed().as_secs_f64() * 1e3
                        })
                        .fold(f64::INFINITY, f64::min)
                };
                let one = time(&mut || {
                    multi_head_attention(
                        &mut out, &q, cache, n_head, group_size, head_dim, scale, window,
                    )
                });
                let tiled = time(&mut || {
                    attention_tiled(
                        &mut out, &q, cache, n_head, group_size, head_dim, scale, window,
                    )
                });
                let mixed = if mixed_available(head_dim) {
                    time(&mut || {
                        attention_mixed(
                            &mut out, &q, cache, n_head, group_size, head_dim, scale, window,
                        )
                    })
                } else {
                    f64::NAN
                };
                println!(
                    "head_dim {head_dim} window {} tokens {n_tokens}: one-query {one:.1} ms, \
                     tiled {tiled:.1} ms ({:.2}x), mixed {mixed:.1} ms ({:.2}x)",
                    if n_swa == 0 {
                        "causal".to_string()
                    } else {
                        n_swa.to_string()
                    },
                    one / tiled,
                    one / mixed
                );
            }
        }
    }
}
