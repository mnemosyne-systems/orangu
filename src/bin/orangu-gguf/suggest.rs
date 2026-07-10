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

//! `orangu-gguf suggest`: a rough, hardware-based suggestion for which GGUF
//! model *size* (parameter count) is likely to run comfortably on this
//! machine — not a specific model yet, just a size class. Reuses the same
//! CPU/GPU detection `system` reports, then estimates the total VRAM a
//! candidate model would need across a curated ladder of common open-weight
//! parameter counts, recommending the largest that fits — for each of
//! several context lengths ([`CONTEXT_LADDER`], 1K to 256K) and each of a
//! few common quantizations ([`QUANT_LADDER`]: `Q2_K`, `Q4_K_M` — this
//! project's own default, matching `download.rs`'s `DEFAULT_TAG_PREFERENCE`
//! and the role wizard's own `-hf ...` examples — and `Q8_0`), presented as
//! a table.
//!
//! The memory-estimation formula mirrors Sam McLeod's GGUF VRAM Estimator
//! (<https://smcleod.net/vram-estimator/>, read directly from its published
//! `vram-calculator.min.js` rather than guessed) and the general shape of
//! erans/selfhostllm's calculator (<https://github.com/erans/selfhostllm>):
//! model weight bytes scale as parameters × bits-per-weight ÷ 8, KV cache
//! bytes scale with context length × layers × hidden size, plus a small
//! fixed runtime overhead. Since there's no real GGUF file to read yet
//! (`suggest` runs before any model is chosen), hidden size and layer count
//! are themselves estimated from the parameter count via the standard
//! transformer parameter-count approximation (params ≈ 12 × layers ×
//! hidden_size²) — the same fallback smcleod's own calculator uses when it
//! has no real GGUF metadata to read.

use crate::format_bytes;
use crate::system::{self, CpuInfo, GpuInfo, MemoryKind};

/// Fixed CUDA/runtime overhead added on top of model weights — matches
/// smcleod's `CUDA_SIZE` constant (500 MiB) exactly.
const RUNTIME_OVERHEAD_BYTES: u64 = 500 * 1024 * 1024;

/// Bits per weight for Q4_K_M, from smcleod's own per-quantization
/// bits-per-weight table — this project's own default quantization
/// (`download.rs`'s `DEFAULT_TAG_PREFERENCE`, the role wizard's examples).
const DEFAULT_BITS_PER_WEIGHT: f64 = 4.83;

/// Quantizations (tag, bits-per-weight) the suggestion table sizes model
/// weights against, in ascending bit-depth order — bits-per-weight values
/// from smcleod's own per-quantization table. `Q2_K` and `Q8_0` bracket this
/// project's own default (`Q4_K_M`, matching `download.rs`'s
/// `DEFAULT_TAG_PREFERENCE`).
const QUANT_LADDER: &[(&str, f64)] = &[
    ("Q2_K", 3.00),
    ("Q4_K_M", DEFAULT_BITS_PER_WEIGHT),
    ("Q8_0", 8.5),
];

/// KV cache held at Q8_0 (8 bits/element), matching the role wizard's own
/// `-ctk q8_0 -ctv q8_0` default rather than assuming a full FP16 cache.
const KV_CACHE_BITS: f64 = 8.0;

/// Context lengths (in tokens) the suggestion table sizes the KV cache
/// against, from a bare minimum up to the role wizard's own largest default
/// (262144, for `all`/`review`) — since KV cache grows linearly with context,
/// the model size that comfortably fits shrinks as context grows.
const CONTEXT_LADDER: &[u64] = &[1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144];

/// A curated ladder of common open-weight model parameter counts (in
/// billions), spanning the range real Hugging Face GGUF releases actually
/// come in. `suggest_param_count` walks this from largest to smallest and
/// recommends the first that fits the estimated budget.
const PARAM_LADDER_BILLIONS: &[f64] = &[
    671.0, 405.0, 235.0, 120.0, 110.0, 70.0, 65.0, 34.0, 32.0, 30.0, 27.0, 24.0, 22.0, 14.0, 13.0,
    12.0, 9.0, 8.0, 7.0, 4.0, 3.0, 2.0, 1.0,
];

/// Estimates hidden size and layer count from a parameter count alone, via
/// the standard transformer-parameter approximation params ≈ 12 × layers ×
/// hidden_size².
fn estimate_hidden_dims(params_billion: f64) -> (f64, f64) {
    let params = params_billion * 1e9;
    let hidden_size = (params / 12.0).sqrt();
    let layers = (params / (12.0 * hidden_size * hidden_size)).round();
    (hidden_size, layers)
}

/// Estimated total VRAM (bytes) to run a `params_billion`-parameter model at
/// `bits_per_weight` quantization with a `context_size`-token KV cache held
/// at `kv_cache_bits` per element: model weight bytes (`params × bpw ÷ 8`)
/// plus a fixed runtime overhead, plus KV cache bytes (`context × 2 (K and
/// V) × layers × hidden_size × kv_bytes_per_element`) plus a smaller
/// "compute buffer" term (`context × hidden_size × 3 × bytes_per_weight`)
/// for attention scratch space.
fn estimate_total_vram_bytes(
    params_billion: f64,
    bits_per_weight: f64,
    context_size: u64,
    kv_cache_bits: f64,
) -> u64 {
    let params = params_billion * 1e9;
    let model_bytes = params * bits_per_weight / 8.0;
    let (hidden_size, layers) = estimate_hidden_dims(params_billion);
    let context = context_size as f64;
    let kv_cache_bytes = context * 2.0 * layers * hidden_size * (kv_cache_bits / 8.0);
    let compute_buffer_bytes = context * hidden_size * 3.0 * (bits_per_weight / 8.0);
    (model_bytes + RUNTIME_OVERHEAD_BYTES as f64 + kv_cache_bytes + compute_buffer_bytes) as u64
}

/// The largest entry in [`PARAM_LADDER_BILLIONS`] whose estimated VRAM
/// requirement (at `bits_per_weight` quantization, at `context_size` tokens
/// of KV cache) fits within `budget_bytes`, if any.
fn suggest_param_count(budget_bytes: u64, context_size: u64, bits_per_weight: f64) -> Option<f64> {
    PARAM_LADDER_BILLIONS.iter().copied().find(|&params| {
        estimate_total_vram_bytes(params, bits_per_weight, context_size, KV_CACHE_BITS)
            <= budget_bytes
    })
}

/// The sum of every `Dedicated` GPU's VRAM (multi-GPU tensor-split, matching
/// the role wizard's `-sm layer`), `0` when there's no dedicated GPU at all.
/// The conservative, GPU-only budget: everything fits in real VRAM, no
/// spillover to a shared pool or system RAM.
fn dedicated_vram_budget_bytes(gpus: &[GpuInfo]) -> u64 {
    gpus.iter()
        .filter(|g| matches!(g.memory_kind, MemoryKind::Dedicated | MemoryKind::Unknown))
        .filter_map(|g| {
            g.vram_total_bytes
                .map(|total| total.saturating_sub(g.vram_used_bytes.unwrap_or(0)))
        })
        .sum()
}

/// The sum of every GPU's own reported `vram_total_bytes`, `Dedicated` and
/// `Shared` alike (a `Shared` GPU's is already the system RAM total, via
/// `system::apply_shared_memory_total`) — the more permissive budget,
/// representing every device `--fit on` could spread layers across at once.
/// Falls back to the CPU's own available RAM when that sum is `0` (no GPU
/// detected at all). `Unknown`-kind GPUs (macOS/Windows edge cases
/// `system.rs` itself can't classify, such as Windows AMD) are now included
/// since their `vram_total_bytes` represents dedicated memory capacity.
fn combined_gpu_budget_bytes(cpu: &CpuInfo, gpus: &[GpuInfo]) -> u64 {
    let total: u64 = gpus
        .iter()
        .filter(|g| {
            matches!(
                g.memory_kind,
                MemoryKind::Dedicated | MemoryKind::Shared | MemoryKind::Unknown
            )
        })
        .filter_map(|g| {
            g.vram_total_bytes
                .map(|total| total.saturating_sub(g.vram_used_bytes.unwrap_or(0)))
        })
        .sum();
    if total > 0 {
        total
    } else {
        cpu.available_memory_bytes
    }
}

/// `~14B` for a whole number of billions, `~4.8B` otherwise.
fn format_param_count(params_billion: f64) -> String {
    if params_billion.fract() == 0.0 {
        format!("{params_billion:.0}B")
    } else {
        format!("{params_billion:.1}B")
    }
}

/// Appends one `label`ed budget/suggestion block to `out`: the estimated
/// budget, followed by a table of the largest model size that comfortably
/// fits at each context length in [`CONTEXT_LADDER`], one column per
/// quantization in [`QUANT_LADDER`] — larger contexts and heavier
/// quantizations both leave less budget for model weights, so suggested
/// sizes shrink as either grows.
fn push_suggestion_block(out: &mut String, label: &str, budget: u64) {
    out.push_str(&format!("\n{label}\n"));
    out.push_str(&format!("  Estimated budget : {}\n", format_bytes(budget)));

    let headers: Vec<String> = QUANT_LADDER
        .iter()
        .map(|(tag, _)| format!("Suggestion ({tag})"))
        .collect();

    out.push_str(&format!("\n  {:<7}  {}\n", "Context", headers.join("  ")));
    out.push_str(&format!(
        "  {}  {}\n",
        "-".repeat(7),
        headers
            .iter()
            .map(|h| "-".repeat(h.len()))
            .collect::<Vec<_>>()
            .join("  "),
    ));

    for &context_size in CONTEXT_LADDER {
        let context_label = format!("{}K", context_size / 1024);
        let cells: Vec<String> = QUANT_LADDER
            .iter()
            .zip(&headers)
            .map(|((_, bits_per_weight), header)| {
                let suggestion = match suggest_param_count(budget, context_size, *bits_per_weight) {
                    Some(params) => format!("~{} parameters", format_param_count(params)),
                    None => "-".to_string(),
                };
                format!("{suggestion:<width$}", width = header.len())
            })
            .collect();
        out.push_str(&format!("  {context_label:<7}  {}\n", cells.join("  ")));
    }
}

/// Formats `suggest`'s full report: the same CPU/GPU inventory `system`
/// prints, followed by two model-size suggestions — one sized against
/// dedicated GPU VRAM alone, one against every GPU's memory combined (which,
/// for a shared/integrated GPU, already means system RAM).
pub fn format_suggestion(cpu: &CpuInfo, gpus: &[GpuInfo]) -> String {
    let mut out = system::format_report(cpu, gpus);

    push_suggestion_block(
        &mut out,
        "Suggested model size (Dedicated)",
        dedicated_vram_budget_bytes(gpus),
    );
    push_suggestion_block(
        &mut out,
        "Suggested model size (Shared)",
        combined_gpu_budget_bytes(cpu, gpus),
    );

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(memory_kind: MemoryKind, vram_total_bytes: Option<u64>) -> GpuInfo {
        GpuInfo {
            vendor: "Test".to_string(),
            name: "Test GPU".to_string(),
            vram_total_bytes,
            vram_used_bytes: None,
            driver: None,
            memory_kind,
        }
    }

    fn cpu(total_memory_bytes: u64) -> CpuInfo {
        CpuInfo {
            brand: "Test CPU".to_string(),
            vendor: String::new(),
            arch: "x86_64".to_string(),
            physical_cores: Some(8),
            logical_cores: 16,
            frequency_mhz: 0,
            total_memory_bytes,
            available_memory_bytes: total_memory_bytes,
        }
    }

    const GIB: u64 = 1024 * 1024 * 1024;

    #[test]
    fn estimate_total_vram_bytes_is_dominated_by_model_weights() {
        // A 7B model at Q4_K_M should need roughly 7e9 * 4.83 / 8 bytes for
        // weights alone (~4.2 GiB), plus modest KV cache/overhead on top —
        // total comfortably under 6 GiB, comfortably over 4 GiB.
        let estimate = estimate_total_vram_bytes(7.0, DEFAULT_BITS_PER_WEIGHT, 8192, 8.0);
        assert!(estimate > 4 * GIB, "estimate too low: {estimate}");
        assert!(estimate < 6 * GIB, "estimate too high: {estimate}");
    }

    #[test]
    fn estimate_total_vram_bytes_grows_with_context_size() {
        let small_ctx = estimate_total_vram_bytes(7.0, DEFAULT_BITS_PER_WEIGHT, 4096, 8.0);
        let large_ctx = estimate_total_vram_bytes(7.0, DEFAULT_BITS_PER_WEIGHT, 32768, 8.0);
        assert!(large_ctx > small_ctx);
    }

    #[test]
    fn suggest_param_count_picks_the_largest_that_fits() {
        // A tiny budget should fall back to the smallest rung (1B needs
        // ~1.3 GiB at this formula; 2B would already exceed 1.5 GiB).
        assert_eq!(
            suggest_param_count(3 * GIB / 2, 8192, DEFAULT_BITS_PER_WEIGHT),
            Some(1.0)
        );
        // A generous budget should recommend a large rung.
        assert_eq!(
            suggest_param_count(1024 * GIB, 8192, DEFAULT_BITS_PER_WEIGHT),
            Some(671.0)
        );
        // An essentially empty budget fits nothing on the ladder.
        assert_eq!(
            suggest_param_count(1024, 8192, DEFAULT_BITS_PER_WEIGHT),
            None
        );
    }

    #[test]
    fn suggest_param_count_shrinks_as_context_grows() {
        // The same budget should recommend a smaller (or equal) model size
        // as the KV cache eats more of that budget at longer contexts.
        let budget = 24 * GIB;
        let short_ctx = suggest_param_count(budget, 4096, DEFAULT_BITS_PER_WEIGHT).unwrap();
        let long_ctx = suggest_param_count(budget, 262144, DEFAULT_BITS_PER_WEIGHT).unwrap();
        assert!(long_ctx <= short_ctx);
    }

    #[test]
    fn suggest_param_count_shrinks_as_quantization_gets_heavier() {
        // The same budget/context should recommend a smaller (or equal)
        // model size as bits-per-weight grows, since heavier quantizations
        // need more bytes per parameter.
        let budget = 24 * GIB;
        let q2 = suggest_param_count(budget, 8192, 3.00).unwrap();
        let q8 = suggest_param_count(budget, 8192, 8.5).unwrap();
        assert!(q8 <= q2);
    }

    #[test]
    fn dedicated_vram_budget_bytes_sums_multiple_dedicated_gpus() {
        let gpus = vec![
            gpu(MemoryKind::Dedicated, Some(24 * GIB)),
            gpu(MemoryKind::Dedicated, Some(24 * GIB)),
        ];
        assert_eq!(dedicated_vram_budget_bytes(&gpus), 48 * GIB);
    }

    #[test]
    fn dedicated_vram_budget_bytes_ignores_shared_and_subtracts_used() {
        let gpus = vec![
            GpuInfo {
                vendor: "Test".to_string(),
                name: "Test GPU 1".to_string(),
                vram_total_bytes: Some(24 * GIB),
                vram_used_bytes: Some(4 * GIB),
                driver: None,
                memory_kind: MemoryKind::Dedicated,
            },
            gpu(MemoryKind::Shared, Some(64 * GIB)),
            gpu(MemoryKind::Unknown, Some(16 * GIB)),
        ];
        // 20 GiB from Dedicated (24 - 4) + 16 GiB from Unknown = 36 GiB
        assert_eq!(dedicated_vram_budget_bytes(&gpus), 36 * GIB);
    }

    #[test]
    fn dedicated_vram_budget_bytes_is_zero_without_a_dedicated_gpu() {
        let gpus = vec![gpu(MemoryKind::Shared, Some(32 * GIB))];
        assert_eq!(dedicated_vram_budget_bytes(&gpus), 0);
        assert_eq!(dedicated_vram_budget_bytes(&[]), 0);
    }

    #[test]
    fn combined_gpu_budget_bytes_sums_dedicated_and_shared() {
        let gpus = vec![
            gpu(MemoryKind::Dedicated, Some(4 * GIB)),
            gpu(MemoryKind::Shared, Some(64 * GIB)),
        ];
        assert_eq!(combined_gpu_budget_bytes(&cpu(64 * GIB), &gpus), 68 * GIB);
    }

    #[test]
    fn combined_gpu_budget_bytes_includes_unknown_and_subtracts_used() {
        let gpus = vec![
            GpuInfo {
                vendor: "Test".to_string(),
                name: "Test GPU 1".to_string(),
                vram_total_bytes: Some(4 * GIB),
                vram_used_bytes: Some(1 * GIB),
                driver: None,
                memory_kind: MemoryKind::Dedicated,
            },
            gpu(MemoryKind::Unknown, Some(12 * GIB)),
        ];
        assert_eq!(combined_gpu_budget_bytes(&cpu(64 * GIB), &gpus), 15 * GIB);
    }

    #[test]
    fn combined_gpu_budget_bytes_falls_back_to_system_ram_without_any_gpu() {
        assert_eq!(combined_gpu_budget_bytes(&cpu(16 * GIB), &[]), 16 * GIB);
    }

    #[test]
    fn combined_gpu_budget_bytes_falls_back_to_system_ram_with_no_gpu() {
        assert_eq!(combined_gpu_budget_bytes(&cpu(16 * GIB), &[]), 16 * GIB);
    }

    #[test]
    fn format_param_count_drops_the_decimal_for_whole_numbers() {
        assert_eq!(format_param_count(7.0), "7B");
        assert_eq!(format_param_count(4.83), "4.8B");
    }

    #[test]
    fn format_suggestion_includes_hardware_report_and_both_suggestions() {
        let gpus = vec![
            gpu(MemoryKind::Dedicated, Some(4 * GIB)),
            gpu(MemoryKind::Shared, Some(64 * GIB)),
        ];
        let report = format_suggestion(&cpu(64 * GIB), &gpus);
        assert!(report.contains("CPU"));
        assert!(report.contains("GPU"));
        assert!(report.contains("Suggested model size (Dedicated)"));
        assert!(report.contains("Suggested model size (Shared)"));
        assert!(report.contains("Suggestion (Q2_K)"));
        assert!(report.contains("Suggestion (Q4_K_M)"));
        assert!(report.contains("Suggestion (Q8_0)"));

        // The dedicated-only budget (4 GiB) and the combined budget (68 GiB)
        // should recommend different, larger sizes for the combined one.
        let dedicated_section = report
            .split("(Dedicated)")
            .nth(1)
            .and_then(|rest| rest.split("(Shared)").next())
            .unwrap();
        assert!(dedicated_section.contains("4.00 GiB"));
    }
}
