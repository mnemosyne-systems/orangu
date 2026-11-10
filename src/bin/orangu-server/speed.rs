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

//! How fast this machine's own kernels read weights, for choosing a model.
//!
//! A decode step reads every weight the token touches once, so its rate is
//! close to the rate the decode kernel reads weights at, divided by the bytes
//! a token touches. The first is a property of this machine — its memory, its
//! card, its driver — and no table of hardware names gets it right; the
//! second is a property of the model. So the rate is measured, briefly, on
//! the kernels the server would run: a one-token `Q4_K` product over a
//! matrix far larger than any cache, on the cores and on the first card,
//! each until a deadline. The figure is kept per device and driver
//! (`~/.orangu/server/tuning/speed-<key>.json`) and measured again only when
//! the key changes, so `suggest` pays for it once.
//!
//! What it estimates is the best case: the weights alone, at depth zero, on
//! one stream. Attention at depth, host turns and a model that spills its
//! card are slower; a model whose weights the card cannot hold is priced at
//! the cores' rate.

use std::time::{Duration, Instant};

use crate::engine::backend::{Backend, CpuBackend};
use crate::engine::loader::QuantMatrix;

/// Where a rate was measured, and how fast weights were read there.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Rate {
    /// The device's name, or `cpu`.
    pub device: String,
    /// Weight bytes a second the one-token decode kernel reads.
    pub bytes_per_second: f64,
}

/// The weights one probe call reads: 256 MiB of `Q4_K` rows, far past any
/// cache, and enough that a call's fixed cost (its submission and readback)
/// is a small share of it.
const PROBE_IN: usize = 4096;
const PROBE_OUT: usize = 256 * 1024 * 1024 / (PROBE_IN / 256 * 144);

/// How long each device is measured for.
const PROBE_TIME: Duration = Duration::from_millis(1500);

/// A `Q4_K` matrix of `out` rows of `in_dim`, with scales that make every
/// weight a small finite value.
fn probe_matrix(in_dim: usize, out: usize) -> QuantMatrix {
    const BLOCK: usize = 144;
    let blocks = in_dim / 256 * out;
    let mut bytes = vec![0u8; blocks * BLOCK];
    for (i, block) in bytes.as_chunks_mut::<BLOCK>().0.iter_mut().enumerate() {
        // d = 0.001, dmin = 0 as f16; the rest a cheap pattern.
        block[0..2].copy_from_slice(&half::f16::from_f32(0.001).to_le_bytes());
        for (j, b) in block[4..].iter_mut().enumerate() {
            *b = ((i * 31 + j * 7) % 251) as u8;
        }
    }
    QuantMatrix::from_encoded_rows(
        bytes,
        crate::engine::quant::GGML_TYPE_Q4_K,
        in_dim,
        out,
        None,
    )
}

/// Weight bytes a second `backend`'s one-token product reads, measured until
/// `time` has passed (and at least three calls after a warm-up).
pub fn measure(backend: &dyn Backend, time: Duration) -> f64 {
    let w = probe_matrix(PROBE_IN, PROBE_OUT);
    let x = vec![0.5f32; PROBE_IN];
    let _ = backend.matmul_decode(&x, 1, &w);
    let started = Instant::now();
    let mut calls = 0u32;
    while calls < 3 || started.elapsed() < time {
        std::hint::black_box(backend.matmul_decode(&x, 1, &w));
        calls += 1;
    }
    let bytes = (PROBE_IN / 256 * 144 * PROBE_OUT) as f64 * f64::from(calls);
    bytes / started.elapsed().as_secs_f64()
}

/// Where the rates are kept, by the device key they were measured under.
fn cache_path(key: &str) -> Option<std::path::PathBuf> {
    crate::engine::backend::tuning_profile::dir().map(|d| d.join(format!("speed-{key}.json")))
}

/// The cores' rate and, where the server would use one, the first card's:
/// from the cache when it was measured under this device key, else measured
/// now and kept.
pub fn rates() -> Vec<Rate> {
    let cpu = CpuBackend;
    let gpu = first_card();
    let key = match (&gpu, crate::engine::backend::tuning_profile::device_key()) {
        (Some(_), Some(key)) => format!("{key}+{}", host_key()),
        _ => host_key(),
    };
    if let Some(path) = cache_path(&key)
        && let Ok(text) = std::fs::read_to_string(&path)
        && let Ok(rates) = serde_json::from_str::<Vec<Rate>>(&text)
        && !rates.is_empty()
    {
        return rates;
    }
    let mut rates = vec![Rate {
        device: "cpu".to_string(),
        bytes_per_second: measure(&cpu, PROBE_TIME),
    }];
    if let Some(gpu) = &gpu {
        rates.push(Rate {
            device: gpu.adapter_name.clone(),
            bytes_per_second: measure(gpu, PROBE_TIME),
        });
    }
    if let Some(path) = cache_path(&key) {
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let _ = std::fs::write(
            &path,
            serde_json::to_string_pretty(&rates).unwrap_or_default(),
        );
    }
    rates
}

/// The card the server would choose by itself (`device` unset), brought up
/// the way it brings it up; `None` without one.
fn first_card() -> Option<crate::engine::backend::VulkanBackend> {
    use crate::engine::backend::{VulkanBackend, device};
    let backends = wgpu::Backends::VULKAN;
    let candidates = VulkanBackend::devices(backends);
    let selected = device::select_all(&candidates, &device::DeviceRequest::Auto).ok()?;
    let indices: Vec<usize> = selected.iter().map(|&p| candidates[p].index).collect();
    VulkanBackend::try_init_selected(backends, &indices)
}

/// The cores, named for the cache: the architecture and the thread count.
fn host_key() -> String {
    format!(
        "cpu-{}-{}",
        std::env::consts::ARCH,
        std::thread::available_parallelism().map_or(0, std::num::NonZeroUsize::get)
    )
}

/// Weight bytes one token of a catalogue model reads: its file, or the
/// share of it the parameters a token reads are — a mixture of experts'
/// active ones (`…-A3B`), or a gemma-4 `E` model's effective ones (`…-E2B`),
/// whose per-layer embeddings are most of the file and a few rows of them a
/// token.
pub fn bytes_per_token(entry: &orangu::model_catalogue::CatalogueEntry) -> u64 {
    let name = entry.repo.rsplit('/').next().unwrap_or(entry.repo);
    let active = name
        .split('-')
        .find_map(|part| {
            part.strip_prefix('A')
                .or_else(|| part.strip_prefix('E'))
                .and_then(|n| n.strip_suffix('B'))
                .and_then(|n| n.parse::<f64>().ok())
        })
        .filter(|&a| a > 0.0 && a < entry.params_billion);
    match active {
        Some(active) => (entry.file_bytes as f64 * active / entry.params_billion) as u64,
        None => entry.file_bytes,
    }
}

/// A decode rate's best case: tokens a second at `rate` bytes a second.
pub fn tokens_per_second(bytes_per_token: u64, rate: f64) -> f64 {
    if bytes_per_token == 0 {
        return 0.0;
    }
    rate / bytes_per_token as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A mixture of experts reads its active share; a dense model all of it.
    #[test]
    fn a_token_reads_the_active_share_of_a_mixture() {
        let moe = orangu::model_catalogue::entry_for("unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF")
            .unwrap();
        let share = bytes_per_token(moe) as f64 / moe.file_bytes as f64;
        assert!((share - 3.0 / 30.5).abs() < 1e-6, "{share}");
        let effective = orangu::model_catalogue::entry_for("unsloth/gemma-4-E2B-it-GGUF").unwrap();
        let share = bytes_per_token(effective) as f64 / effective.file_bytes as f64;
        assert!(
            (share - 2.0 / effective.params_billion).abs() < 1e-6,
            "{share}"
        );
        let dense = orangu::model_catalogue::entry_for("unsloth/Qwen3-4B-GGUF").unwrap();
        assert_eq!(bytes_per_token(dense), dense.file_bytes);
        assert!((tokens_per_second(2_000_000_000, 40e9) - 20.0).abs() < 1e-9);
        assert_eq!(tokens_per_second(0, 40e9), 0.0);
    }

    /// The probe reads its matrix on the cores at a finite, positive rate.
    #[test]
    fn the_cores_read_at_a_measurable_rate() {
        let rate = measure(&CpuBackend, Duration::from_millis(50));
        assert!(rate.is_finite() && rate > 1e6, "{rate}");
    }
}
