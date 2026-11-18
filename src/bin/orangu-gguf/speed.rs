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

//! A type per tensor, chosen for size and speed: the speed table.
//!
//! What a type costs a tensor in time is measured on the machine, through
//! the engine's own kernels, at the model's own shapes: one token's matrix
//! product (a decode step) and one prompt chunk's (prefill), on every
//! device the model can run on. The kernels differ by type far more than
//! the bytes do, so a smaller type can be the slower one.
//!
//! A card is timed on its own clock (timestamps around a long burst of
//! dispatches, after an untimed burst that brings the clock up), the cores
//! on the wall clock. Both cycle through copies of the weight that together
//! outgrow the caches, so each product reads its weight from memory as a
//! model's next layer does. Answers are kept per device, driver and build
//! for a week.

#[cfg(test)]
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use orangu::engine::backend::{Backend, CpuBackend, vulkan::VulkanBackend};
use orangu::engine::loader::QuantMatrix;
use orangu::quantize as q;

/// Rows in one prefill product: a prompt chunk's width.
pub const PREFILL_TOKENS: usize = 512;

/// How many bytes of copies a shape is timed over: past every cache.
const OUTRUN_BYTES: usize = 256 << 20;

/// The most weight one product is timed over. A larger weight is timed on
/// its first rows and scaled up by its row count: its rows are independent,
/// and one product of a whole output head at prompt width can outlast what
/// a device allows a submission.
const TIMED_BYTES: usize = 64 << 20;

/// How long one timed burst lasts.
const BURST: Duration = Duration::from_millis(40);

/// Timed bursts per answer; the median is kept.
const RUNS: usize = 5;

/// A weight's shape: `in_dim` columns, `out_dim` rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Shape {
    pub in_dim: usize,
    pub out_dim: usize,
}

/// One type's time at one shape on one device, on each kernel family: the
/// float kernels first, then the integer dot. A card chooses its family for
/// decode and for prompts once per model, by timing the whole model both
/// ways, so both are kept; where a device has one family, both are its.
#[derive(Clone, Debug, PartialEq)]
pub struct Rate {
    /// One token's product, in microseconds.
    pub decode_us: [f64; 2],
    /// One [`PREFILL_TOKENS`]-row product, in microseconds.
    pub prefill_us: [f64; 2],
    /// The kernels that ran.
    pub kernels: String,
}

/// A device a model can run on.
pub enum Device {
    Gpu {
        name: String,
        backend: Box<VulkanBackend>,
        /// The card's own memory, where the driver reports it.
        memory: Option<u64>,
    },
    Cpu {
        name: String,
    },
}

impl Device {
    pub fn name(&self) -> &str {
        match self {
            Device::Gpu { name, .. } | Device::Cpu { name } => name,
        }
    }
}

/// Every device on this machine, best first: the cards the server would
/// rank, then the cores.
pub fn devices() -> Vec<Device> {
    use orangu::engine::backend::device;
    let backends = wgpu::Backends::VULKAN;
    let candidates = VulkanBackend::devices(backends);
    let mut out: Vec<Device> = device::preference_order(&candidates)
        .into_iter()
        .map(|i| &candidates[i])
        .filter(|c| !c.class.is_software())
        .filter_map(|c| {
            let backend = VulkanBackend::try_init_selected(backends, &[c.index])?;
            let driver = c.driver.as_deref().unwrap_or("");
            Some(Device::Gpu {
                name: format!("{} [{driver}]", c.name),
                backend: Box::new(backend),
                memory: c.vram_total_bytes,
            })
        })
        .collect();
    out.push(Device::Cpu {
        name: format!("cpu x{}", rayon::current_num_threads()),
    });
    out
}

/// `copies` matrices of `ggml_type` at `shape`, each over its own bytes.
/// The rows are one encoded row of weight-like values, repeated.
fn weights(ggml_type: u32, shape: Shape, copies: usize) -> Vec<QuantMatrix> {
    let row: Vec<f32> = (0..shape.in_dim)
        .map(|i| {
            let h = (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            ((h >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.08
        })
        .collect();
    let encoded = q::encode(ggml_type, &row, shape.in_dim);
    let bytes: Vec<u8> = encoded
        .iter()
        .copied()
        .cycle()
        .take(encoded.len() * shape.out_dim)
        .collect();
    (0..copies)
        .map(|_| {
            QuantMatrix::from_encoded_rows(
                bytes.clone(),
                ggml_type,
                shape.in_dim,
                shape.out_dim,
                None,
            )
        })
        .collect()
}

/// How many of `out_dim` rows of `row_bytes` [`measure`] times: all of them
/// up to [`TIMED_BYTES`], else as many as fit, in whole 256-row runs.
fn timed_rows(row_bytes: usize, out_dim: usize) -> usize {
    if row_bytes * out_dim <= TIMED_BYTES {
        return out_dim;
    }
    (TIMED_BYTES / row_bytes.max(1) / 256 * 256).clamp(256, out_dim)
}

/// How many copies of a `bytes`-sized weight outgrow the caches.
fn copies_for(bytes: usize) -> usize {
    OUTRUN_BYTES.div_ceil(bytes.max(1)).clamp(2, 64)
}

/// The median wall time of one `run` over `n` items, in microseconds, the
/// items cycled; bursts of about [`BURST`] after an untimed one.
fn wall_us(n: usize, mut run: impl FnMut(usize)) -> f64 {
    let started = Instant::now();
    for i in 0..n {
        run(i);
    }
    let once = started.elapsed().as_secs_f64() / n as f64;
    let reps = ((BURST.as_secs_f64() / once.max(1e-9)) as usize).max(n);
    let burst = |run: &mut dyn FnMut(usize)| {
        let started = Instant::now();
        for i in 0..reps {
            run(i % n);
        }
        started.elapsed().as_secs_f64() * 1e6 / reps as f64
    };
    burst(&mut run);
    let mut samples: Vec<f64> = (0..RUNS).map(|_| burst(&mut run)).collect();
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

/// Measures `ggml_type` at `shape` on `device`, or `None` when the device
/// cannot time it.
pub fn measure(device: &Device, ggml_type: u32, shape: Shape) -> Option<Rate> {
    let row_bytes = q::row_bytes(ggml_type, shape.in_dim);
    let rows = timed_rows(row_bytes, shape.out_dim);
    let scale = shape.out_dim as f64 / rows as f64;
    let timed = Shape {
        in_dim: shape.in_dim,
        out_dim: rows,
    };
    let ws = weights(ggml_type, timed, copies_for(row_bytes * rows));
    let rate = match device {
        Device::Gpu { backend, .. } => {
            let mut decode_us = [0.0; 2];
            let mut prefill_us = [0.0; 2];
            let mut kernels = Vec::new();
            for (family, on) in [false, true].into_iter().enumerate() {
                backend.set_integer_dot_decode(on);
                backend.set_integer_dot_prefill(on);
                let (d, dk) = backend.matmul_rate_us(&ws, 1, BURST, RUNS)?;
                let (p, pk) = backend.matmul_rate_us(&ws, PREFILL_TOKENS, BURST, RUNS)?;
                decode_us[family] = d;
                prefill_us[family] = p;
                for k in [dk, pk] {
                    if !kernels.contains(&k) {
                        kernels.push(k);
                    }
                }
            }
            Some(Rate {
                decode_us,
                prefill_us,
                kernels: kernels.join(", "),
            })
        }
        Device::Cpu { .. } => {
            let one: Vec<f32> = vec![0.03; shape.in_dim];
            let chunk: Vec<f32> = vec![0.03; shape.in_dim * PREFILL_TOKENS];
            let decode_us = wall_us(ws.len(), |i| {
                std::hint::black_box(CpuBackend.matmul_decode(&one, 1, &ws[i]));
            });
            let prefill_us = wall_us(ws.len().min(2), |i| {
                std::hint::black_box(CpuBackend.matmul(&chunk, PREFILL_TOKENS, &ws[i]));
            });
            Some(Rate {
                decode_us: [decode_us; 2],
                prefill_us: [prefill_us; 2],
                kernels: "cpu".into(),
            })
        }
    };
    rate.map(|r| Rate {
        decode_us: r.decode_us.map(|us| us * scale),
        prefill_us: r.prefill_us.map(|us| us * scale),
        kernels: r.kernels,
    })
}

/// What a kept answer depends on: the device and its driver, this build,
/// and the shape and type.
fn cache_key(device: &Device, ggml_type: u32, shape: Shape) -> String {
    format!(
        "{}|{}|{}|{}x{}",
        device.name(),
        orangu::build_info::id(),
        ggml_type,
        shape.in_dim,
        shape.out_dim
    )
}

const CACHE_WHAT: &str = "gguf speed";

/// [`measure`], answered from the kept table where it can be.
pub fn rate(device: &Device, ggml_type: u32, shape: Shape) -> Option<Rate> {
    use orangu::engine::probe_cache;
    let key = cache_key(device, ggml_type, shape);
    if let Some(v) = probe_cache::value(CACHE_WHAT, &key) {
        let pair = |v: &serde_json::Value| Some([v[0].as_f64()?, v[1].as_f64()?]);
        let rate = (|| {
            Some(Rate {
                decode_us: pair(&v["decode_us"])?,
                prefill_us: pair(&v["prefill_us"])?,
                kernels: v["kernels"].as_str()?.to_string(),
            })
        })();
        if rate.is_some() {
            return rate;
        }
    }
    let rate = measure(device, ggml_type, shape)?;
    probe_cache::keep_value(
        CACHE_WHAT,
        &key,
        serde_json::json!({
            "decode_us": rate.decode_us,
            "prefill_us": rate.prefill_us,
            "kernels": rate.kernels,
        }),
    );
    Some(rate)
}

/// The matrix products one token of a model runs, as (type, shape): every
/// two-dimensional weight but the token embedding, which a step reads one
/// row of — unless no `output` follows it, when it is the output head too.
#[cfg(test)]
pub fn products(tensors: &[orangu::gguf::TensorInfo]) -> Vec<(u32, Shape)> {
    let has_output = tensors.iter().any(|t| t.name == "output.weight");
    tensors
        .iter()
        .filter(|t| t.dims.len() == 2 && t.name.ends_with(".weight"))
        .filter(|t| t.name != "token_embd.weight" || !has_output)
        .filter(|t| {
            t.name.starts_with("blk.")
                || t.name.starts_with("output")
                || t.name == "token_embd.weight"
        })
        .filter(|t| t.dims[1] > 1)
        .map(|t| {
            (
                t.ggml_type,
                Shape {
                    in_dim: t.dims[0] as usize,
                    out_dim: t.dims[1] as usize,
                },
            )
        })
        .collect()
}

/// One token's predicted product time on a device, in milliseconds, and
/// one prefill chunk's per token: the sum over `products` of the table's
/// rates, each on the kernel family faster for the whole set, as the device
/// would choose it. `None` when the table lacks one of them.
#[cfg(test)]
pub fn predict(
    table: &BTreeMap<(u32, Shape), Rate>,
    products: &[(u32, Shape)],
) -> Option<(f64, f64)> {
    let mut decode = [0.0; 2];
    let mut prefill = [0.0; 2];
    for key in products {
        let rate = table.get(key)?;
        for family in 0..2 {
            decode[family] += rate.decode_us[family];
            prefill[family] += rate.prefill_us[family];
        }
    }
    let decode = decode[0].min(decode[1]);
    let prefill = prefill[0].min(prefill[1]);
    Some((decode / 1e3, prefill / 1e3 / PREFILL_TOKENS as f64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_large_weight_is_timed_on_its_first_rows() {
        assert_eq!(timed_rows(2520, 3072), 3072);
        assert_eq!(timed_rows(2520, 200_064), 26_624);
        assert_eq!(timed_rows(1 << 30, 4096), 256);
    }

    #[test]
    fn copies_outgrow_the_caches_within_bounds() {
        assert_eq!(copies_for(1 << 30), 2);
        assert_eq!(copies_for(1 << 10), 64);
        assert_eq!(copies_for(16 << 20), 16);
    }

    #[test]
    fn the_cores_time_a_product() {
        let device = Device::Cpu { name: "cpu".into() };
        let shape = Shape {
            in_dim: 256,
            out_dim: 64,
        };
        let rate = measure(&device, q::GGML_TYPE_Q4_K, shape).unwrap();
        assert!(rate.decode_us[0] > 0.0 && rate.prefill_us[0] > rate.decode_us[0]);
    }

    fn tensor(name: &str, dims: &[u64], ggml_type: u32) -> orangu::gguf::TensorInfo {
        orangu::gguf::TensorInfo {
            name: name.into(),
            dims: dims.to_vec(),
            ggml_type,
            offset: 0,
        }
    }

    #[test]
    fn a_token_runs_every_matrix_but_the_embedding() {
        let k = q::GGML_TYPE_Q4_K;
        let mut tensors = vec![
            tensor("token_embd.weight", &[256, 1000], q::GGML_TYPE_Q6_K),
            tensor("blk.0.attn_q.weight", &[256, 512], k),
            tensor("blk.0.attn_norm.weight", &[256], 0),
            tensor("output.weight", &[256, 1000], q::GGML_TYPE_Q6_K),
        ];
        let shapes =
            |p: Vec<(u32, Shape)>| p.into_iter().map(|(_, s)| s.out_dim).collect::<Vec<_>>();
        assert_eq!(shapes(products(&tensors)), [512, 1000]);
        tensors.pop();
        assert_eq!(shapes(products(&tensors)), [1000, 512]);
    }

    /// On this machine's devices, the speed table of a real model's shapes
    /// and its prediction of one token, for a run of `orangu-bench` on the
    /// same file to be held against. `ORANGU_GGUF` names the file.
    #[test]
    #[ignore = "needs ORANGU_GGUF; times every device"]
    fn predicts_a_real_model() {
        let path = std::env::var("ORANGU_GGUF").expect("ORANGU_GGUF");
        let file = orangu::gguf::GgufFile::open(std::path::Path::new(&path)).unwrap();
        let products = products(&file.tensors);
        let mut wanted: Vec<(u32, Shape)> = products.clone();
        wanted.sort();
        wanted.dedup();
        for device in devices() {
            let started = Instant::now();
            let mut table = BTreeMap::new();
            for &(ggml_type, shape) in &wanted {
                if let Some(rate) = rate(&device, ggml_type, shape) {
                    println!(
                        "  {:>8} {:>6}x{:<6} decode {:>8.1}/{:<8.1} us  prefill {:>9.1}/{:<9.1} us  {}",
                        q::type_name(ggml_type),
                        shape.in_dim,
                        shape.out_dim,
                        rate.decode_us[0],
                        rate.decode_us[1],
                        rate.prefill_us[0],
                        rate.prefill_us[1],
                        rate.kernels
                    );
                    table.insert((ggml_type, shape), rate);
                }
            }
            match predict(&table, &products) {
                Some((decode, prefill)) => println!(
                    "{}: {} products, decode {decode:.2} ms/token, prefill {prefill:.3} ms/token ({:.1} s)",
                    device.name(),
                    products.len(),
                    started.elapsed().as_secs_f64()
                ),
                None => println!("{}: not every type measured", device.name()),
            }
        }
    }
}
