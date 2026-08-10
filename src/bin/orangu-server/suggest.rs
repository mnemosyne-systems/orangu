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

//! `orangu-server suggest`: a rough, hardware-based suggestion for which
//! GGUF model *size* (parameter count) is likely to run comfortably on this
//! machine — not a specific model yet, just a size class. Reuses the same
//! CPU/GPU detection `system` reports, then estimates the total VRAM a
//! candidate model would need across a curated ladder of common open-weight
//! parameter counts, recommending the largest that fits — for each of
//! several context lengths ([`CONTEXT_LADDER`], 1K to 256K) and each of a
//! few common quantizations ([`QUANT_LADDER`]: `Q2_K`, `Q4_K_M` — this
//! project's own default, matching `orangu::model_download`'s
//! `DEFAULT_TAG_PREFERENCE` — and `Q8_0`), presented as a table.
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

use orangu::format::format_bytes;
use orangu::hardware::{self as system, CpuInfo, GpuInfo, MemoryKind};
use orangu::os::OsInfo;

/// Fixed CUDA/runtime overhead added on top of model weights — matches
/// smcleod's `CUDA_SIZE` constant (500 MiB) exactly.
const RUNTIME_OVERHEAD_BYTES: u64 = 500 * 1024 * 1024;

/// Bits per weight for Q4_K_M, from smcleod's own per-quantization
/// bits-per-weight table — this project's own default quantization
/// (`orangu::model_download`'s `DEFAULT_TAG_PREFERENCE`).
const DEFAULT_BITS_PER_WEIGHT: f64 = 4.83;

/// Quantizations (tag, bits-per-weight) the suggestion table sizes model
/// weights against, in ascending bit-depth order — bits-per-weight values
/// from smcleod's own per-quantization table. `Q2_K` and `Q8_0` bracket this
/// project's own default (`Q4_K_M`, matching `orangu::model_download`'s
/// `DEFAULT_TAG_PREFERENCE`).
const QUANT_LADDER: &[(&str, f64)] = &[
    ("Q2_K", 3.00),
    ("Q4_K_M", DEFAULT_BITS_PER_WEIGHT),
    ("Q8_0", 8.5),
];

/// KV cache held at Q8_0 (8 bits/element) rather than a full FP16 cache —
/// matches how `orangu-server` itself stores it (see `engine::kv_cache`).
const KV_CACHE_BITS: f64 = 8.0;

/// Context lengths (in tokens) the suggestion table sizes the KV cache
/// against, from a bare minimum up to a generous long-context ceiling —
/// since KV cache grows linearly with context, the model size that
/// comfortably fits shrinks as context grows.
const CONTEXT_LADDER: &[u64] = &[1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144];

/// A curated ladder of common open-weight model parameter counts (in
/// billions), spanning the range real Hugging Face GGUF releases actually
/// come in. `suggest_param_count` walks this from largest to smallest and
/// recommends the first that fits the estimated budget.
const PARAM_LADDER_BILLIONS: &[f64] = &[
    671.0, 405.0, 235.0, 120.0, 110.0, 70.0, 65.0, 34.0, 32.0, 30.0, 27.0, 24.0, 22.0, 14.0, 13.0,
    12.0, 9.0, 8.0, 7.0, 4.0, 3.0, 2.0, 1.0,
];

/// Estimates hidden size and layer count from a parameter count alone.
///
/// The standard transformer approximation (params ≈ 12 × layers ×
/// hidden_size²) is one equation with two unknowns, so the split is
/// underdetermined: this resolves it by putting everything into the hidden
/// size (`hidden = √(params ÷ 12)`), which makes `layers` work out to
/// exactly 1 — by construction, not by accident. The KV-cache estimate
/// built on it therefore scales as context × √params, which tracks modern
/// GQA-era models well (their per-layer KV width shrinks as depth grows,
/// so total KV grows sublinearly in parameters), and matches what smcleod's
/// own calculator falls back to without real GGUF metadata to read.
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

/// On Windows, `system::windows_memory_kind` classifies *any* AMD adapter as
/// `Unknown` — it can't tell an integrated APU's small BIOS-reserved
/// carve-out from a real discrete Radeon card's VRAM by name alone (that
/// distinction only exists in DXGI, unreachable from a plain WMI query). An
/// `Unknown` GPU's own `vram_total_bytes` is trusted as real dedicated VRAM
/// only above this threshold, chosen well above the few-hundred-MiB to
/// low-GiB carve-out a typical integrated GPU reports and comfortably below
/// any real discrete card's VRAM. Below it, the GPU is treated the same as
/// a `Shared` one: no dedicated capacity of its own, real ceiling is system
/// RAM. Linux and macOS need no such guess — `linux_memory_kind` and
/// `macos_memory_kind` already classify every GPU as `Dedicated` or `Shared`
/// directly from a reliable per-platform signal.
#[cfg(target_os = "windows")]
const WINDOWS_UNKNOWN_DEDICATED_THRESHOLD_BYTES: u64 = 1024 * 1024 * 1024;

/// Whether `gpu` counts as having real, hard-ceiling dedicated VRAM for the
/// conservative budget. Only `Dedicated`-kind on Linux/macOS, where
/// `MemoryKind` is already reliably known; on Windows, an `Unknown`-kind GPU
/// (see [`WINDOWS_UNKNOWN_DEDICATED_THRESHOLD_BYTES`]) above the threshold
/// counts too.
#[cfg(not(target_os = "windows"))]
fn is_dedicated_for_budget(gpu: &GpuInfo) -> bool {
    gpu.memory_kind == MemoryKind::Dedicated
}

#[cfg(target_os = "windows")]
fn is_dedicated_for_budget(gpu: &GpuInfo) -> bool {
    gpu.memory_kind == MemoryKind::Dedicated
        || (gpu.memory_kind == MemoryKind::Unknown
            && gpu.vram_total_bytes.unwrap_or(0) >= WINDOWS_UNKNOWN_DEDICATED_THRESHOLD_BYTES)
}

/// Whether `gpu`'s own memory pool counts as a candidate for the
/// whole-machine budget: `Dedicated` or `Shared` on Linux/macOS; on Windows,
/// also an `Unknown`-kind GPU above
/// [`WINDOWS_UNKNOWN_DEDICATED_THRESHOLD_BYTES`] (a below-threshold one is
/// excluded here too — like a genuine `Shared` GPU, its real ceiling is
/// system RAM, which [`total_budget_bytes`] already considers on its own).
#[cfg(not(target_os = "windows"))]
fn is_total_budget_eligible(gpu: &GpuInfo) -> bool {
    matches!(gpu.memory_kind, MemoryKind::Dedicated | MemoryKind::Shared)
}

#[cfg(target_os = "windows")]
fn is_total_budget_eligible(gpu: &GpuInfo) -> bool {
    matches!(gpu.memory_kind, MemoryKind::Dedicated | MemoryKind::Shared)
        || (gpu.memory_kind == MemoryKind::Unknown
            && gpu.vram_total_bytes.unwrap_or(0) >= WINDOWS_UNKNOWN_DEDICATED_THRESHOLD_BYTES)
}

/// The largest dedicated GPU's `vram_total_bytes`, `0` when there's no
/// dedicated GPU at all (see [`is_dedicated_for_budget`] for what counts as
/// dedicated per platform). The conservative, GPU-only budget: everything
/// fits in real VRAM, no spillover to a shared pool or system RAM.
///
/// The largest card, not the sum of every card: a model is loaded onto one
/// device, and nothing here splits a model's tensors across two of them, so
/// two 24 GiB cards run the same models one 24 GiB card does — summing them
/// would budget 48 GiB of VRAM that no single run can address. Same reason
/// [`total_budget_bytes`] takes a maximum.
///
/// Deliberately *not* reduced by `vram_used_bytes`: `suggest` estimates the
/// hardware's own capability (`suggest.rs`'s module doc — "likely to run
/// comfortably on this machine", picked before any model is chosen), not how
/// much happens to be free right now. Whatever else is transiently using
/// VRAM when `suggest` runs (a compositor, a browser, an already-running
/// `llama-server`) shouldn't shrink a hardware-based estimate.
fn dedicated_vram_budget_bytes(gpus: &[GpuInfo]) -> u64 {
    largest_dedicated_gpu(gpus)
        .and_then(|gpu| gpu.vram_total_bytes)
        .unwrap_or(0)
}

/// The GPU [`dedicated_vram_budget_bytes`] took its number from — so the
/// report can name it rather than leaving the reader to work out which of
/// the machine's cards a bare byte count refers to.
fn largest_dedicated_gpu(gpus: &[GpuInfo]) -> Option<&GpuInfo> {
    gpus.iter()
        .filter(|g| is_dedicated_for_budget(g) && g.vram_total_bytes.is_some())
        .max_by_key(|g| g.vram_total_bytes.unwrap_or(0))
}

/// The GPU [`total_budget_bytes`] took its number from, or `None` when the
/// winner was system RAM.
fn largest_total_pool<'a>(cpu: &CpuInfo, gpus: &'a [GpuInfo]) -> Option<&'a GpuInfo> {
    gpus.iter()
        .filter(|g| is_total_budget_eligible(g))
        .filter(|g| {
            g.vram_total_bytes
                .is_some_and(|v| v > cpu.total_memory_bytes)
        })
        .max_by_key(|g| g.vram_total_bytes.unwrap_or(0))
}

/// The largest single memory pool a model can actually be run in on this
/// machine: the biggest budget-eligible GPU's own `vram_total_bytes` (see
/// [`is_total_budget_eligible`]), or the CPU's total RAM, whichever is
/// larger. System RAM is always a candidate — the CPU backend needs no GPU
/// at all — which is also what covers a machine with no GPU detected, and a
/// `Shared` GPU (whose `vram_total_bytes` is already the system RAM total,
/// via `system::apply_shared_memory_total`).
///
/// The largest pool, deliberately *not* the sum of every pool. A model runs
/// on exactly one backend — `main.rs`'s `select_backend` picks one, and
/// there is no partial-offload split of layers between a GPU and the CPU —
/// so no single run can ever draw on a dedicated card's VRAM *and* system
/// RAM together. Summing them would put a budget on the table that nothing
/// on this machine can reach: on a 4 GiB discrete card beside 62 GiB of RAM,
/// a 66 GiB sum recommends a model that neither pool can hold.
///
/// Dedicated VRAM is one of the candidates here, so this budget is never
/// smaller than [`dedicated_vram_budget_bytes`]'s — the dedicated table is
/// the *fast* subset of this one, not a separate machine. Like that budget,
/// deliberately not reduced by currently-used memory — see its doc for why —
/// so it stays a hardware ceiling rather than a promise: the RAM in it is
/// the same RAM the OS and everything else on the machine live in.
fn total_budget_bytes(cpu: &CpuInfo, gpus: &[GpuInfo]) -> u64 {
    gpus.iter()
        .filter(|g| is_total_budget_eligible(g))
        .filter_map(|g| g.vram_total_bytes)
        .max()
        .unwrap_or(0)
        .max(cpu.total_memory_bytes)
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
fn push_suggestion_block(out: &mut String, label: &str, budget: u64, source: Option<&str>) {
    out.push_str(&format!("\n{label}\n"));
    // Which pool the number came from, not just its size. On a machine with
    // three GPUs — the ordinary laptop with a discrete card, an iGPU, and a
    // software rasterizer all reporting memory — "3.98 GiB" alone is a
    // number the reader has to guess the provenance of, and the plausible
    // wrong guess (the iGPU, which reports the whole system RAM) is off by
    // an order of magnitude.
    match source {
        Some(source) => out.push_str(&format!(
            "  Estimated budget : {} ({source})\n",
            format_bytes(budget)
        )),
        None => out.push_str(&format!("  Estimated budget : {}\n", format_bytes(budget))),
    }

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

/// Formats `suggest`'s full report: the same OS/CPU/GPU inventory `system`
/// prints, followed by two model-size suggestions — one sized against
/// dedicated GPU VRAM alone (the fast path), one against the largest memory
/// pool on the machine, GPU or system RAM (see [`total_budget_bytes`]).
pub fn format_suggestion(os: &OsInfo, cpu: &CpuInfo, gpus: &[GpuInfo]) -> String {
    let mut out = system::format_report(os, cpu, gpus);

    // A "Dedicated" budget of 0 (no GPU with real, hard-ceiling VRAM at
    // all) would just print an estimated budget of 0 B and a table of
    // nothing but "-" — noise, not a suggestion — so the whole block is
    // skipped rather than shown empty.
    if let Some(gpu) = largest_dedicated_gpu(gpus) {
        push_suggestion_block(
            &mut out,
            "Suggested model size (Dedicated)",
            dedicated_vram_budget_bytes(gpus),
            Some(&gpu.name),
        );
    }
    let total = total_budget_bytes(cpu, gpus);
    push_suggestion_block(
        &mut out,
        "Suggested model size (Total)",
        total,
        Some(
            &largest_total_pool(cpu, gpus)
                .map(|gpu| gpu.name.clone())
                .unwrap_or_else(|| "system RAM".to_string()),
        ),
    );

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(memory_kind: MemoryKind, vram_total_bytes: Option<u64>) -> GpuInfo {
        named_gpu("Test GPU", memory_kind, vram_total_bytes)
    }

    fn named_gpu(name: &str, memory_kind: MemoryKind, vram_total_bytes: Option<u64>) -> GpuInfo {
        GpuInfo {
            vendor: "Test".to_string(),
            name: name.to_string(),
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
            features: system::CpuFeatures {
                sse4_2: false,
                avx2: false,
                avx512f: false,
            },
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

    /// The budget names the card it came from. On a machine with a
    /// discrete card *and* an iGPU reporting the whole system RAM, a bare
    /// byte count is a number whose most plausible misreading is off by an
    /// order of magnitude.
    #[test]
    fn the_dedicated_budget_names_the_card_it_came_from() {
        let gpus = vec![
            named_gpu("Small Card", MemoryKind::Dedicated, Some(4 * GIB)),
            named_gpu("Big Card", MemoryKind::Dedicated, Some(24 * GIB)),
            named_gpu("iGPU", MemoryKind::Shared, Some(64 * GIB)),
        ];
        let report = format_suggestion(&orangu::os::detect(), &cpu(64 * GIB), &gpus);
        assert!(report.contains("24.00 GiB (Big Card)"), "{report}");
        // The `Total` block's winner here is system RAM, tied with the
        // iGPU's shared pool — and a tie must not be reported as the GPU,
        // since the pool really is the same RAM.
        assert!(report.contains("64.00 GiB (system RAM)"), "{report}");
    }

    /// A card genuinely bigger than system RAM is what the `Total` budget
    /// is drawn from, and saying so is the point of the label.
    #[test]
    fn the_total_budget_names_a_gpu_when_the_gpu_is_the_largest_pool() {
        let gpus = vec![named_gpu("Big Card", MemoryKind::Dedicated, Some(80 * GIB))];
        let report = format_suggestion(&orangu::os::detect(), &cpu(32 * GIB), &gpus);
        assert!(report.contains("80.00 GiB (Big Card)"), "{report}");
    }

    #[test]
    fn dedicated_vram_budget_bytes_takes_the_largest_of_several_gpus() {
        // Two cards run the models one card runs — nothing splits a model's
        // tensors across devices, so the budget is 24 GiB, not 48.
        let gpus = vec![
            gpu(MemoryKind::Dedicated, Some(24 * GIB)),
            gpu(MemoryKind::Dedicated, Some(24 * GIB)),
        ];
        assert_eq!(dedicated_vram_budget_bytes(&gpus), 24 * GIB);

        // ...and it's the biggest one that counts, not the first found.
        let mixed = vec![
            gpu(MemoryKind::Dedicated, Some(8 * GIB)),
            gpu(MemoryKind::Dedicated, Some(24 * GIB)),
        ];
        assert_eq!(dedicated_vram_budget_bytes(&mixed), 24 * GIB);
    }

    #[test]
    fn dedicated_vram_budget_bytes_ignores_currently_used_vram() {
        // A hardware-capability estimate shouldn't shrink just because
        // something else happens to be using VRAM right now.
        let gpus = vec![GpuInfo {
            vendor: "Test".to_string(),
            name: "Test GPU 1".to_string(),
            vram_total_bytes: Some(24 * GIB),
            vram_used_bytes: Some(4 * GIB),
            driver: None,
            memory_kind: MemoryKind::Dedicated,
        }];
        assert_eq!(dedicated_vram_budget_bytes(&gpus), 24 * GIB);
    }

    #[cfg(not(target_os = "windows"))]
    #[test]
    fn dedicated_vram_budget_bytes_ignores_shared_and_unknown() {
        let gpus = vec![
            gpu(MemoryKind::Dedicated, Some(24 * GIB)),
            gpu(MemoryKind::Shared, Some(64 * GIB)),
            gpu(MemoryKind::Unknown, Some(999 * GIB)),
        ];
        assert_eq!(dedicated_vram_budget_bytes(&gpus), 24 * GIB);
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn dedicated_vram_budget_bytes_trusts_unknown_above_threshold_on_windows() {
        // The `Unknown` card is the larger of the two, so it can only be the
        // answer if it counted as dedicated at all.
        let gpus = vec![
            gpu(MemoryKind::Dedicated, Some(4 * GIB)),
            gpu(MemoryKind::Unknown, Some(16 * GIB)),
        ];
        assert_eq!(dedicated_vram_budget_bytes(&gpus), 16 * GIB);
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn dedicated_vram_budget_bytes_ignores_unknown_below_threshold_on_windows() {
        // Nothing else in the list, so a below-threshold `Unknown` GPU
        // counting would show up as a non-zero budget.
        let gpus = vec![gpu(
            MemoryKind::Unknown,
            Some(WINDOWS_UNKNOWN_DEDICATED_THRESHOLD_BYTES - 1),
        )];
        assert_eq!(dedicated_vram_budget_bytes(&gpus), 0);
    }

    #[test]
    fn dedicated_vram_budget_bytes_is_zero_without_a_dedicated_gpu() {
        let gpus = vec![gpu(MemoryKind::Shared, Some(32 * GIB))];
        assert_eq!(dedicated_vram_budget_bytes(&gpus), 0);
        assert_eq!(dedicated_vram_budget_bytes(&[]), 0);
    }

    /// The whole point of the total budget: a model runs on one backend, so
    /// the budget is the largest pool it could run in — never dedicated VRAM
    /// *plus* the system RAM beside it, a figure no single run could reach.
    #[test]
    fn total_budget_bytes_takes_the_largest_pool_not_their_sum() {
        let gpus = vec![
            gpu(MemoryKind::Dedicated, Some(4 * GIB)),
            gpu(MemoryKind::Shared, Some(64 * GIB)),
        ];
        assert_eq!(total_budget_bytes(&cpu(64 * GIB), &gpus), 64 * GIB);
    }

    /// This machine's own shape, and the case that made the sum visibly
    /// wrong: a small discrete card beside a lot of system RAM. The budget
    /// is the RAM, not RAM + VRAM.
    #[test]
    fn total_budget_bytes_never_exceeds_the_machines_own_memory() {
        let gpus = vec![
            gpu(MemoryKind::Dedicated, Some(4 * GIB)),
            gpu(MemoryKind::Shared, Some(62 * GIB)),
        ];
        let budget = total_budget_bytes(&cpu(62 * GIB), &gpus);
        assert_eq!(budget, 62 * GIB);
        assert!(budget < 62 * GIB + 4 * GIB);
    }

    /// A dedicated card bigger than the machine's RAM wins instead — the
    /// dedicated budget is a subset of this one, never larger than it.
    #[test]
    fn total_budget_bytes_prefers_a_dedicated_gpu_larger_than_system_ram() {
        let gpus = vec![gpu(MemoryKind::Dedicated, Some(24 * GIB))];
        assert_eq!(total_budget_bytes(&cpu(16 * GIB), &gpus), 24 * GIB);
        assert!(total_budget_bytes(&cpu(16 * GIB), &gpus) >= dedicated_vram_budget_bytes(&gpus));
    }

    #[test]
    fn total_budget_bytes_ignores_currently_used_vram() {
        let gpus = vec![GpuInfo {
            vendor: "Test".to_string(),
            name: "Test GPU 1".to_string(),
            vram_total_bytes: Some(80 * GIB),
            vram_used_bytes: Some(GIB),
            driver: None,
            memory_kind: MemoryKind::Dedicated,
        }];
        assert_eq!(total_budget_bytes(&cpu(64 * GIB), &gpus), 80 * GIB);
    }

    #[cfg(not(target_os = "windows"))]
    #[test]
    fn total_budget_bytes_ignores_unknown_gpus() {
        let gpus = vec![
            gpu(MemoryKind::Dedicated, Some(4 * GIB)),
            gpu(MemoryKind::Unknown, Some(999 * GIB)),
        ];
        assert_eq!(total_budget_bytes(&cpu(64 * GIB), &gpus), 64 * GIB);
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn total_budget_bytes_includes_unknown_above_threshold_on_windows() {
        let gpus = vec![
            gpu(MemoryKind::Dedicated, Some(4 * GIB)),
            gpu(MemoryKind::Unknown, Some(80 * GIB)),
        ];
        assert_eq!(total_budget_bytes(&cpu(64 * GIB), &gpus), 80 * GIB);
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn total_budget_bytes_falls_back_to_system_ram_below_threshold_on_windows() {
        // A below-threshold `Unknown` GPU (likely an integrated APU's small
        // BIOS carve-out) shouldn't count at face value — its real ceiling
        // is system RAM, same as a genuine `Shared` GPU.
        let gpus = vec![gpu(
            MemoryKind::Unknown,
            Some(WINDOWS_UNKNOWN_DEDICATED_THRESHOLD_BYTES - 1),
        )];
        assert_eq!(total_budget_bytes(&cpu(16 * GIB), &gpus), 16 * GIB);
    }

    #[test]
    fn total_budget_bytes_falls_back_to_system_ram_without_any_gpu() {
        assert_eq!(total_budget_bytes(&cpu(16 * GIB), &[]), 16 * GIB);
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
        let report = format_suggestion(&orangu::os::detect(), &cpu(64 * GIB), &gpus);
        assert!(report.contains("CPU"));
        assert!(report.contains("GPU"));
        assert!(report.contains("Suggested model size (Dedicated)"));
        assert!(report.contains("Suggested model size (Total)"));
        assert!(report.contains("Suggestion (Q2_K)"));
        assert!(report.contains("Suggestion (Q4_K_M)"));
        assert!(report.contains("Suggestion (Q8_0)"));

        // Each table states the budget it was sized against: 4 GiB of real
        // VRAM for the dedicated one, and 64 GiB — the largest single pool,
        // not the 68 GiB the two of them add up to — for the total.
        let dedicated_section = report
            .split("(Dedicated)")
            .nth(1)
            .and_then(|rest| rest.split("(Total)").next())
            .unwrap();
        assert!(
            dedicated_section.contains("4.00 GiB"),
            "{dedicated_section}"
        );
        let total_section = report.split("(Total)").nth(1).unwrap();
        assert!(
            total_section.contains("Estimated budget : 64.00 GiB"),
            "{total_section}"
        );
    }

    #[test]
    fn format_suggestion_omits_the_dedicated_table_without_a_dedicated_gpu() {
        for (case, gpus) in [
            ("shared-only", vec![gpu(MemoryKind::Shared, Some(64 * GIB))]),
            ("no gpus", Vec::new()),
        ] {
            let report = format_suggestion(&orangu::os::detect(), &cpu(64 * GIB), &gpus);
            assert!(
                !report.contains("Suggested model size (Dedicated)"),
                "unexpected Dedicated table for case: {case}"
            );
            assert!(report.contains("Suggested model size (Total)"));
        }
    }

    /// On a machine with no GPU at all, `format_report` leaves the GPU
    /// section out entirely — the suggestion block that follows still starts
    /// with its own blank line, so the report doesn't run together.
    #[test]
    fn format_suggestion_has_no_gpu_section_without_a_gpu() {
        let report = format_suggestion(&orangu::os::detect(), &cpu(64 * GIB), &[]);
        assert!(!report.contains("GPU"), "unexpected GPU section:\n{report}");
        assert!(
            report.contains("\n\nSuggested model size (Total)\n"),
            "report:\n{report}"
        );
    }
}
