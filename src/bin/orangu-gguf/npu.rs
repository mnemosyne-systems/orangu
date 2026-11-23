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

//! Training's matrix products on a Rockchip NPU.
//!
//! The device multiplies fp16 operands and accumulates in fp32. Each operand
//! is scaled by a power of two that puts its largest magnitude just under
//! fp16's top, so gradients keep their small values, and the product is
//! scaled back exactly. `K` and `N` are padded with zeros to the device's
//! alignment.
//!
//! Every shape is timed on both sides: its first calls go to the NPU and
//! then to the processor, after which each call goes to whichever was
//! faster, with the other tried again now and then. A shape the device
//! declines stays on the processor.
//!
//! `ORANGU_GGUF_DEVICE=cpu` keeps every product on the processor;
//! `ORANGU_GGUF_NPU_MIN_MFLOP` sets the smallest product offered to the NPU.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

use half::f16;
use orangu::engine::backend::rknpu::{Dense, NO_RKNPU};
use rayon::prelude::*;

use crate::gemm::View;

/// Smallest product, in MFLOP, offered to the NPU.
const NPU_MIN_MFLOP: u64 = 50;

/// Calls each side gets before the faster one is chosen.
const TRIALS: u32 = 3;

/// Every this many calls the slower side is timed again.
const RETRY: u32 = 64;

/// Tile edge of the operand packing.
const TILE: usize = 32;

/// The machine's NPU, opened once.
pub fn npu() -> Option<&'static Dense> {
    static NPU: std::sync::OnceLock<Option<Dense>> = std::sync::OnceLock::new();
    NPU.get_or_init(|| {
        if std::env::var("ORANGU_GGUF_DEVICE").as_deref() == Ok("cpu") {
            return None;
        }
        match Dense::try_open() {
            Ok(dense) => Some(dense),
            Err(why) => {
                if why != NO_RKNPU {
                    eprintln!("orangu-gguf: the NPU is not used: {why}");
                }
                None
            }
        }
    })
    .as_ref()
}

/// Where one product runs.
pub enum Choice {
    Npu(&'static Dense),
    /// On the processor; the timer, when held to the end of the product,
    /// records how long the processor took.
    Cpu(Option<Timer>),
}

/// One shape's record.
#[derive(Default, Clone, Copy)]
struct Record {
    npu: Side,
    cpu: Side,
    calls: u32,
    declined: bool,
}

#[derive(Default, Clone, Copy)]
struct Side {
    seconds: f64,
    runs: u32,
}

impl Side {
    fn add(&mut self, seconds: f64) {
        self.runs += 1;
        // The first run of a shape on the NPU makes its contexts; it is not
        // counted.
        self.seconds = if self.runs <= 2 {
            seconds
        } else {
            0.7 * self.seconds + 0.3 * seconds
        };
    }
}

type Key = (usize, usize, usize);

static RECORDS: Mutex<Option<HashMap<Key, Record>>> = Mutex::new(None);

fn with_record<R>(key: Key, f: impl FnOnce(&mut Record) -> R) -> R {
    let mut records = RECORDS.lock().expect("npu records poisoned");
    f(records
        .get_or_insert_with(HashMap::new)
        .entry(key)
        .or_default())
}

/// Times a product on the processor until dropped.
pub struct Timer {
    key: Key,
    start: Instant,
}

impl Drop for Timer {
    fn drop(&mut self) {
        let seconds = self.start.elapsed().as_secs_f64();
        with_record(self.key, |r| r.cpu.add(seconds));
    }
}

/// Where `c[m, n] = a[m, p] · b[p, n]` runs.
pub fn for_product(m: usize, n: usize, p: usize) -> Choice {
    let Some(device) = npu() else {
        return Choice::Cpu(None);
    };
    static MIN: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let min = *MIN.get_or_init(|| {
        std::env::var("ORANGU_GGUF_NPU_MIN_MFLOP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(NPU_MIN_MFLOP)
    });
    if ((2 * m * n * p) as u64) / 1_000_000 < min {
        return Choice::Cpu(None);
    }
    let key = (m, n, p);
    let on_npu = with_record(key, |r| {
        if r.declined {
            return false;
        }
        r.calls += 1;
        if r.npu.runs < TRIALS {
            return true;
        }
        if r.cpu.runs < TRIALS {
            return false;
        }
        let npu_faster = r.npu.seconds < r.cpu.seconds;
        // Now and then the slower side runs, so a change in either is seen.
        if r.calls % RETRY == 0 {
            !npu_faster
        } else {
            npu_faster
        }
    });
    if on_npu {
        Choice::Npu(device)
    } else {
        Choice::Cpu(Some(Timer {
            key,
            start: Instant::now(),
        }))
    }
}

/// `c[m, n] (+)= a[m, p] · b[p, n]` on `device`, or `false` when the device
/// declined and the caller computes it.
#[allow(clippy::too_many_arguments)]
pub fn gemm(
    device: &Dense,
    c: &mut [f32],
    a: View<'_>,
    b: View<'_>,
    m: usize,
    n: usize,
    p: usize,
    add: bool,
) -> bool {
    let start = Instant::now();
    let done = product(c, a, b, m, n, p, add, |m, k, n, fill, read| {
        device.product(m, k, n, fill, read)
    });
    let seconds = start.elapsed().as_secs_f64();
    with_record((m, n, p), |r| {
        if done {
            r.npu.add(seconds);
        } else {
            r.declined = true;
        }
    });
    done
}

/// The packing, scaling and unpacking around one device product, with the
/// device passed in so the tests can stand in for it.
#[allow(clippy::too_many_arguments)]
fn product(
    c: &mut [f32],
    a: View<'_>,
    b: View<'_>,
    m: usize,
    n: usize,
    p: usize,
    add: bool,
    device: impl FnOnce(
        usize,
        usize,
        usize,
        &mut dyn FnMut(&mut [f16], &mut [f16]),
        &mut dyn FnMut(&[f32]),
    ) -> bool,
) -> bool {
    let k_pad = p.next_multiple_of(Dense::K_ALIGN);
    let n_pad = n.next_multiple_of(Dense::N_ALIGN);
    let a_scale = scale(a, m, p);
    let b_scale = scale(b, p, n);
    let back = 1.0 / (a_scale * b_scale);
    let mut fill = |a16: &mut [f16], b16: &mut [f16]| {
        rayon::join(
            || pack(a16, k_pad, a, m, p, a_scale),
            || pack(b16, n_pad, b, p, n, b_scale),
        );
    };
    let mut read = |c16: &[f32]| {
        c[..m * n]
            .par_chunks_mut(n)
            .zip(c16.par_chunks(n_pad))
            .for_each(|(out, got)| {
                if add {
                    for (o, g) in out.iter_mut().zip(got) {
                        *o += g * back;
                    }
                } else {
                    for (o, g) in out.iter_mut().zip(got) {
                        *o = g * back;
                    }
                }
            });
    };
    device(m, k_pad, n_pad, &mut fill, &mut read)
}

/// The power of two that brings `v`'s largest magnitude to between 2¹⁴ and
/// 2¹⁵, inside fp16's range and clear of its smallest values.
fn scale(v: View<'_>, rows: usize, cols: usize) -> f32 {
    let peak = (0..rows)
        .into_par_iter()
        .map(|i| {
            (0..cols)
                .map(|j| v.data[i * v.row + j * v.col].abs())
                .fold(0f32, f32::max)
        })
        .reduce(|| 0.0, f32::max);
    if !(peak.is_finite() && peak > 0.0) {
        return 1.0;
    }
    let exponent = 14 - peak.log2().floor() as i32;
    2f32.powi(exponent.clamp(-100, 100))
}

/// Writes `v` (`rows × cols`) times `scale` into `dst` with row stride `ld`,
/// zeros in the padding, in tiles so a transposed read stays in cache.
fn pack(dst: &mut [f16], ld: usize, v: View<'_>, rows: usize, cols: usize, scale: f32) {
    let used = rows * ld;
    dst[used..].fill(f16::ZERO);
    dst[..used]
        .par_chunks_mut(ld * TILE)
        .enumerate()
        .for_each(|(band, out)| {
            let first = band * TILE;
            let height = out.len() / ld;
            for row in out.chunks_mut(ld) {
                row[cols..].fill(f16::ZERO);
            }
            for j0 in (0..cols).step_by(TILE) {
                let width = TILE.min(cols - j0);
                for di in 0..height {
                    let i = first + di;
                    let row = &mut out[di * ld + j0..di * ld + j0 + width];
                    if v.col == 1 {
                        let src = &v.data[i * v.row + j0..i * v.row + j0 + width];
                        for (d, s) in row.iter_mut().zip(src) {
                            *d = f16::from_f32(s * scale);
                        }
                    } else {
                        for (dj, d) in row.iter_mut().enumerate() {
                            *d = f16::from_f32(v.data[i * v.row + (j0 + dj) * v.col] * scale);
                        }
                    }
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wave(n: usize, k: f32, size: f32) -> Vec<f32> {
        (0..n).map(|i| (i as f32 * k).sin() * size).collect()
    }

    /// A stand-in for the device: the fp16 operands it is handed, multiplied
    /// in f64.
    fn host(
        m: usize,
        k: usize,
        n: usize,
        fill: &mut dyn FnMut(&mut [f16], &mut [f16]),
        read: &mut dyn FnMut(&[f32]),
    ) -> bool {
        assert_eq!(k % Dense::K_ALIGN, 0);
        assert_eq!(n % Dense::N_ALIGN, 0);
        let mut a = vec![f16::from_f32(f32::NAN); m * k];
        let mut b = vec![f16::from_f32(f32::NAN); k * n];
        fill(&mut a, &mut b);
        let mut c = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let s: f64 = (0..k)
                    .map(|q| {
                        f64::from(f32::from(a[i * k + q])) * f64::from(f32::from(b[q * n + j]))
                    })
                    .sum();
                c[i * n + j] = s as f32;
            }
        }
        read(&c);
        true
    }

    /// Padding, both stride orders, accumulation and scaling of values far
    /// below fp16's smallest normal all come back as the f32 product to
    /// fp16 rounding.
    #[test]
    fn the_npu_product_is_the_plain_loop() {
        for &(m, n, p, size) in &[
            (1, 1, 1, 1.0),
            (7, 17, 33, 1e-7),
            (40, 70, 100, 3e3),
            (33, 16, 64, 1.0),
        ] {
            let a = wave(m * p, 0.37, size);
            let b = wave(p * n, 0.91, 1.0);
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
                    let seed = wave(m * n, 0.13, size);
                    let mut got = seed.clone();
                    assert!(product(&mut got, a_view, b_view, m, n, p, add, host));
                    for i in 0..m {
                        for j in 0..n {
                            let mut want = if add { f64::from(seed[i * n + j]) } else { 0.0 };
                            let mut magnitude = 0f64;
                            for q in 0..p {
                                let t = f64::from(a_view.data[i * a_view.row + q * a_view.col])
                                    * f64::from(b_view.data[q * b_view.row + j * b_view.col]);
                                want += t;
                                magnitude += t.abs();
                            }
                            let got = f64::from(got[i * n + j]);
                            assert!(
                                (got - want).abs() <= 2e-3 * magnitude + 1e-6 * want.abs(),
                                "{m}x{n}x{p} size {size} ({i}, {j}): {got} vs {want}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// On a machine with an NPU, the device itself against the plain loop.
    #[test]
    fn the_device_product_is_the_plain_loop() {
        let Some(device) = npu() else {
            println!(
                "no NPU ({}); skipping",
                Dense::try_open().err().unwrap_or_default()
            );
            return;
        };
        let (m, n, p) = (64, 96, 200);
        let a = wave(m * p, 0.37, 0.01);
        let b = wave(p * n, 0.91, 1.0);
        let mut got = vec![0f32; m * n];
        let a_view = View {
            data: &a,
            row: p,
            col: 1,
        };
        let b_view = View {
            data: &b,
            row: n,
            col: 1,
        };
        assert!(gemm(device, &mut got, a_view, b_view, m, n, p, false));
        for i in 0..m {
            for j in 0..n {
                let (want, magnitude) = (0..p).fold((0f64, 0f64), |(s, g), q| {
                    let t = f64::from(a[i * p + q]) * f64::from(b[q * n + j]);
                    (s + t, g + t.abs())
                });
                assert!((f64::from(got[i * n + j]) - want).abs() <= 2e-3 * magnitude);
            }
        }
    }
}
