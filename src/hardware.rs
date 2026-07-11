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

//! Machine hardware inventory: the CPU (model, core counts, frequency,
//! system RAM) and any GPUs (vendor, model, VRAM) — the two things that
//! decide how a GGUF model can actually be run (how many layers fit in
//! VRAM, how much has to fall back to CPU/RAM).
//!
//! GPU detection has no single cross-platform API, so it layers several
//! best-effort sources: `nvidia-smi` for NVIDIA (installed alongside any
//! NVIDIA driver, Linux or Windows), Linux's `/sys/class/drm` for everything
//! else on Linux (AMD, Intel, and any other PCI display device), and native
//! OS tools (`system_profiler` / PowerShell's `Win32_VideoController`) on
//! macOS and Windows. A card that isn't recognized by any source simply
//! doesn't show up — this is inventory, not a hard dependency of anything
//! else `orangu-gguf` does.

use crate::format::format_bytes;
use std::process::Command;
use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};

pub struct CpuInfo {
    pub brand: String,
    pub vendor: String,
    pub arch: String,
    pub physical_cores: Option<usize>,
    pub logical_cores: usize,
    pub frequency_mhz: u64,
    pub total_memory_bytes: u64,
    pub available_memory_bytes: u64,
    pub features: CpuFeatures,
}

/// SIMD instruction sets that llama.cpp's CPU backend probes for at startup
/// to pick its matmul kernels. Detected via CPUID at run time (not compile
/// time) so a binary built on one machine reports accurately on whatever
/// machine it actually runs on — the two can easily differ.
pub struct CpuFeatures {
    pub sse4_2: bool,
    pub avx2: bool,
    pub avx512f: bool,
}

/// Runs the actual CPUID checks. Only meaningful on x86/x86_64: the feature
/// names themselves (`is_x86_feature_detected!`) don't exist on other
/// architectures, so ARM/RISC-V etc. simply report all three as absent.
fn detect_cpu_features() -> CpuFeatures {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        CpuFeatures {
            sse4_2: is_x86_feature_detected!("sse4.2"),
            avx2: is_x86_feature_detected!("avx2"),
            avx512f: is_x86_feature_detected!("avx512f"),
        }
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        CpuFeatures {
            sse4_2: false,
            avx2: false,
            avx512f: false,
        }
    }
}

pub struct GpuInfo {
    pub vendor: String,
    pub name: String,
    pub vram_total_bytes: Option<u64>,
    pub vram_used_bytes: Option<u64>,
    pub driver: Option<String>,
    pub memory_kind: MemoryKind,
}

/// Whether a GPU's reported memory is physically dedicated VRAM chips or an
/// integrated GPU/APU's carve-out of, or unified architecture over, system
/// RAM — the two behave very differently for offloading model layers (a
/// dedicated card's VRAM is a hard capacity limit; shared memory instead
/// competes with the CPU for the same RAM pool). Best-effort and derived
/// differently per detection source — see each `detect_*_gpus` function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryKind {
    Dedicated,
    Shared,
    /// No source strong enough to tell either way was available. Only ever
    /// constructed on macOS/Windows, whose detection is `cfg`'d out on other
    /// build targets — hence the blanket `allow` rather than a per-target one.
    #[allow(dead_code)]
    Unknown,
}

impl MemoryKind {
    fn label(self) -> &'static str {
        match self {
            MemoryKind::Dedicated => "Dedicated",
            MemoryKind::Shared => "Shared",
            MemoryKind::Unknown => "Unknown",
        }
    }
}

pub fn detect_cpu() -> CpuInfo {
    let mut sys = System::new_with_specifics(
        RefreshKind::nothing()
            .with_cpu(CpuRefreshKind::everything())
            .with_memory(MemoryRefreshKind::everything()),
    );
    sys.refresh_cpu_all();
    sys.refresh_memory();

    let cpus = sys.cpus();
    let brand = cpus
        .first()
        .map(|c| c.brand().trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());
    let vendor = cpus
        .first()
        .map(|c| c.vendor_id().trim().to_string())
        .unwrap_or_default();
    let frequency_mhz = cpus.iter().map(|c| c.frequency()).max().unwrap_or(0);

    CpuInfo {
        brand,
        vendor,
        arch: System::cpu_arch(),
        physical_cores: System::physical_core_count(),
        logical_cores: cpus.len(),
        frequency_mhz,
        total_memory_bytes: sys.total_memory(),
        available_memory_bytes: sys.available_memory(),
        features: detect_cpu_features(),
    }
}

/// `total_memory_bytes` is the system's total RAM (`CpuInfo::total_memory_bytes`,
/// so callers don't pay for a second `sysinfo` query) — every `Shared` GPU's
/// `vram_total_bytes` is set to it, overriding whatever a platform's own
/// query returned. A shared GPU has no VRAM capacity of its own to report:
/// an APU's tiny BIOS-reserved carve-out (`mem_info_vram_total` on Linux, as
/// little as a few hundred MiB) drastically understates what it can
/// actually draw on, and Intel/Windows sources often report nothing at all.
/// System RAM is the real ceiling on how much such a GPU can use, so it's
/// the only figure worth showing as its total.
pub fn detect_gpus(total_memory_bytes: u64) -> Vec<GpuInfo> {
    let mut gpus = detect_nvidia_gpus();

    #[cfg(target_os = "linux")]
    gpus.extend(detect_linux_sysfs_gpus());

    #[cfg(target_os = "macos")]
    gpus.extend(detect_macos_gpus());

    // Like the Linux sysfs scan, skip the NVIDIA adapters WMI reports when
    // `nvidia-smi` already listed them above with better data (real VRAM
    // use, and totals beyond `AdapterRAM`'s 32-bit cap) — without this the
    // same card shows up twice. When `nvidia-smi` isn't available (it found
    // nothing), WMI is the only source, so its NVIDIA entries are kept.
    #[cfg(target_os = "windows")]
    {
        let have_nvidia_smi = !gpus.is_empty();
        gpus.extend(
            detect_windows_gpus()
                .into_iter()
                .filter(|gpu| !have_nvidia_smi || !gpu.name.to_lowercase().contains("nvidia")),
        );
    }

    apply_shared_memory_total(&mut gpus, total_memory_bytes);
    gpus
}

fn apply_shared_memory_total(gpus: &mut [GpuInfo], total_memory_bytes: u64) {
    for gpu in gpus {
        if gpu.memory_kind == MemoryKind::Shared {
            gpu.vram_total_bytes = Some(total_memory_bytes);
        }
    }
}

/// Runs `nvidia-smi`'s CSV query mode, the one interface guaranteed to exist
/// wherever an NVIDIA driver is installed (Linux or Windows) regardless of
/// which GPU backend (CUDA, Vulkan, ...) llama.cpp itself ends up using.
/// Returns an empty list — not an error — when the binary is absent or
/// fails, since "no NVIDIA GPU" is the common case this is probing for.
fn detect_nvidia_gpus() -> Vec<GpuInfo> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.used,driver_version",
            "--format=csv,noheader,nounits",
        ])
        .output();

    let Ok(output) = output else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }

    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| {
            let fields: Vec<&str> = line.split(',').map(|f| f.trim()).collect();
            let [name, mem_total, mem_used, driver] = fields.as_slice() else {
                return None;
            };
            Some(GpuInfo {
                vendor: "NVIDIA".to_string(),
                name: name.to_string(),
                vram_total_bytes: mem_total.parse::<u64>().ok().map(|mib| mib * 1024 * 1024),
                vram_used_bytes: mem_used.parse::<u64>().ok().map(|mib| mib * 1024 * 1024),
                driver: Some(driver.to_string()),
                // No consumer NVIDIA GPU is anything but a discrete card
                // with its own dedicated VRAM.
                memory_kind: MemoryKind::Dedicated,
            })
        })
        .collect()
}

/// Enumerates display devices via `/sys/class/drm/card*/device`, the kernel
/// interface every Linux GPU driver exposes regardless of vendor. NVIDIA
/// devices are skipped here: `nvidia-smi` already reported them above (with
/// VRAM figures this path can't get anyway — `mem_info_vram_total` is an
/// amdgpu-specific attribute), so including them too would double-list every
/// NVIDIA card.
#[cfg(target_os = "linux")]
fn detect_linux_sysfs_gpus() -> Vec<GpuInfo> {
    const NVIDIA_VENDOR_ID: u32 = 0x10de;

    let Ok(entries) = std::fs::read_dir("/sys/class/drm") else {
        return Vec::new();
    };

    let mut gpus = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for entry in entries.flatten() {
        let file_name = entry.file_name();
        let file_name = file_name.to_string_lossy();
        // Only bare `cardN` directories name a device; `cardN-DP-1` etc. name
        // a connector on that same device and would otherwise double-list it.
        if !file_name.starts_with("card") || file_name.contains('-') {
            continue;
        }

        let device_dir = entry.path().join("device");
        let Some(vendor_id) = read_hex_file(&device_dir.join("vendor")) else {
            continue;
        };
        if vendor_id == NVIDIA_VENDOR_ID || !seen.insert(device_dir.clone()) {
            continue;
        }
        let Some(device_id) = read_hex_file(&device_dir.join("device")) else {
            continue;
        };

        let vendor = pci_vendor_name(vendor_id);
        let name = pci_device_name(vendor_id, device_id)
            .unwrap_or_else(|| format!("{vendor} GPU [{vendor_id:04x}:{device_id:04x}]"));

        gpus.push(GpuInfo {
            vendor,
            name,
            vram_total_bytes: read_u64_file(&device_dir.join("mem_info_vram_total")),
            vram_used_bytes: read_u64_file(&device_dir.join("mem_info_vram_used")),
            driver: std::fs::read_link(device_dir.join("driver"))
                .ok()
                .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned())),
            memory_kind: linux_memory_kind(device_dir.join("mem_info_vram_vendor").is_file()),
        });
    }
    gpus
}

/// Distinguishes a genuine dedicated card from an integrated GPU/APU on
/// Linux by whether the `amdgpu` driver exposes `mem_info_vram_vendor` (the
/// VRAM chip manufacturer, e.g. `samsung`/`hynix`) for this device.
/// Verified directly against real hardware carrying both: a discrete AMD
/// card (Navi 14) has this file; that same machine's integrated AMD APU
/// (Renoir) — which still reports a `mem_info_vram_total` for its
/// BIOS-reserved carve-out of system RAM — does not, since there is no
/// separate memory chip to name. Devices with neither `mem_info_vram_*`
/// attribute at all (Intel's `i915` driver, almost always integrated; a
/// rare discrete Intel Arc card would be misclassified here, since its
/// local-memory sysfs interface isn't read) default to `Shared` too.
#[cfg(target_os = "linux")]
fn linux_memory_kind(has_vram_vendor_file: bool) -> MemoryKind {
    if has_vram_vendor_file {
        MemoryKind::Dedicated
    } else {
        MemoryKind::Shared
    }
}

#[cfg(target_os = "linux")]
fn read_hex_file(path: &std::path::Path) -> Option<u32> {
    let content = std::fs::read_to_string(path).ok()?;
    u32::from_str_radix(content.trim().trim_start_matches("0x"), 16).ok()
}

#[cfg(target_os = "linux")]
fn read_u64_file(path: &std::path::Path) -> Option<u64> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

#[cfg(target_os = "linux")]
fn pci_vendor_name(vendor_id: u32) -> String {
    match vendor_id {
        0x1002 => "AMD".to_string(),
        0x10de => "NVIDIA".to_string(),
        0x8086 => "Intel".to_string(),
        0x1414 => "Microsoft".to_string(),
        other => format!("Vendor {other:04x}"),
    }
}

/// Looks up a device's marketing name in the system's `pci.ids` database
/// (shipped by the `hwdata` package on Fedora/RHEL, `pciutils` elsewhere),
/// the same file `lspci` itself reads. Returns `None` — falling back to the
/// raw vendor:device id — when the file isn't installed rather than failing.
#[cfg(target_os = "linux")]
fn pci_device_name(vendor_id: u32, device_id: u32) -> Option<String> {
    static PCI_IDS: std::sync::OnceLock<std::collections::HashMap<(u32, u32), String>> =
        std::sync::OnceLock::new();

    let table = PCI_IDS.get_or_init(load_pci_ids);
    table.get(&(vendor_id, device_id)).cloned()
}

#[cfg(target_os = "linux")]
fn load_pci_ids() -> std::collections::HashMap<(u32, u32), String> {
    const CANDIDATE_PATHS: &[&str] = &[
        "/usr/share/hwdata/pci.ids",
        "/usr/share/misc/pci.ids",
        "/usr/share/pci.ids",
    ];

    let mut table = std::collections::HashMap::new();
    let Some(contents) = CANDIDATE_PATHS
        .iter()
        .find_map(|path| std::fs::read_to_string(path).ok())
    else {
        return table;
    };

    let mut current_vendor: Option<u32> = None;
    for line in contents.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        // Vendor lines start in column 0; device lines are indented by one
        // tab; subsystem lines by two tabs (skipped — not needed here).
        if !line.starts_with('\t') {
            let mut parts = line.splitn(2, char::is_whitespace);
            let id = parts.next().unwrap_or_default();
            current_vendor = u32::from_str_radix(id, 16).ok();
        } else if !line.starts_with("\t\t")
            && let Some(vendor_id) = current_vendor
        {
            let rest = line.trim_start_matches('\t');
            let mut parts = rest.splitn(2, char::is_whitespace);
            if let (Some(id), Some(name)) = (parts.next(), parts.next())
                && let Ok(device_id) = u32::from_str_radix(id, 16)
            {
                table.insert((vendor_id, device_id), name.trim().to_string());
            }
        }
    }
    table
}

#[cfg(target_os = "macos")]
fn detect_macos_gpus() -> Vec<GpuInfo> {
    let output = Command::new("system_profiler")
        .args(["SPDisplaysDataType", "-json"])
        .output();
    let Ok(output) = output else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    let Ok(json) = serde_json::from_slice::<serde_json::Value>(&output.stdout) else {
        return Vec::new();
    };
    let Some(displays) = json.get("SPDisplaysDataType").and_then(|v| v.as_array()) else {
        return Vec::new();
    };

    displays
        .iter()
        .map(|entry| {
            let name = entry
                .get("sppci_model")
                .or_else(|| entry.get("_name"))
                .and_then(|v| v.as_str())
                .unwrap_or("Apple GPU")
                .to_string();
            // Only the dedicated key's value is worth parsing as a real
            // VRAM figure — a `Shared` entry gets `vram_total_bytes`
            // overridden to system RAM by `detect_gpus` regardless of
            // whatever `spdisplays_vram_shared`'s own value looks like.
            let vram_total_bytes = entry
                .get("spdisplays_vram")
                .and_then(|v| v.as_str())
                .and_then(parse_size_string);
            GpuInfo {
                vendor: "Apple".to_string(),
                name,
                vram_total_bytes,
                vram_used_bytes: None,
                driver: None,
                memory_kind: macos_memory_kind(entry),
            }
        })
        .collect()
}

/// `system_profiler`'s own two keys already say which kind of memory this
/// is: `spdisplays_vram` names a real dedicated-VRAM figure (a discrete
/// card, e.g. an eGPU or an older Mac Pro/MacBook Pro), while
/// `spdisplays_vram_shared` marks Apple Silicon's unified-memory
/// architecture or an older Intel Mac's integrated graphics — either way,
/// memory shared with the CPU rather than a separate pool.
#[cfg(target_os = "macos")]
fn macos_memory_kind(entry: &serde_json::Value) -> MemoryKind {
    if entry.get("spdisplays_vram").is_some() {
        MemoryKind::Dedicated
    } else if entry.get("spdisplays_vram_shared").is_some() {
        MemoryKind::Shared
    } else {
        MemoryKind::Unknown
    }
}

/// Parses `system_profiler`-style human sizes like `"8 GB"` or `"1536 MB"`
/// into bytes.
#[cfg(target_os = "macos")]
fn parse_size_string(value: &str) -> Option<u64> {
    let value = value.trim();
    let (number, unit) = value.split_once(' ')?;
    let number: f64 = number.parse().ok()?;
    let multiplier = match unit.to_uppercase().as_str() {
        "GB" | "GIB" => 1024 * 1024 * 1024,
        "MB" | "MIB" => 1024 * 1024,
        "KB" | "KIB" => 1024,
        _ => return None,
    };
    Some((number * multiplier as f64) as u64)
}

#[cfg(target_os = "windows")]
fn detect_windows_gpus() -> Vec<GpuInfo> {
    let output = Command::new("powershell")
        .args([
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,DriverVersion | ConvertTo-Json",
        ])
        .output();
    let Ok(output) = output else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    let Ok(json) = serde_json::from_slice::<serde_json::Value>(&output.stdout) else {
        return Vec::new();
    };
    // A single result comes back as a bare object, not a one-element array.
    let entries: Vec<serde_json::Value> = match json {
        serde_json::Value::Array(items) => items,
        other @ serde_json::Value::Object(_) => vec![other],
        _ => Vec::new(),
    };

    entries
        .into_iter()
        .map(|entry| GpuInfo {
            vendor: "".to_string(),
            name: entry
                .get("Name")
                .and_then(|v| v.as_str())
                .unwrap_or("GPU")
                .to_string(),
            // WMI's AdapterRAM is a 32-bit field and is well known to
            // misreport (often as 0 or a wrapped value) for cards with more
            // than ~4 GiB of VRAM; still the best zero-dependency source
            // available on Windows.
            vram_total_bytes: entry
                .get("AdapterRAM")
                .and_then(|v| v.as_u64())
                .filter(|&b| b > 0),
            vram_used_bytes: None,
            driver: entry
                .get("DriverVersion")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            memory_kind: windows_memory_kind(
                entry.get("Name").and_then(|v| v.as_str()).unwrap_or(""),
            ),
        })
        .collect()
}

/// `Win32_VideoController` has no dedicated/shared field of its own (that
/// distinction lives in DXGI's `DXGI_ADAPTER_DESC`, which a WMI/PowerShell
/// query can't reach without a real helper binary), so this falls back to
/// guessing from the adapter name string: NVIDIA never ships an integrated
/// GPU, and Intel's line is overwhelmingly integrated (`UHD`/`Iris`/`Iris
/// Xe`) with discrete Arc cards as the rare exception this misses. AMD is
/// left `Unknown` outright — its Windows driver names an APU's integrated
/// GPU and a discrete Radeon card too similarly (e.g. plain "AMD Radeon(TM)
/// Graphics" for either) to guess reliably from the name alone.
#[cfg(target_os = "windows")]
fn windows_memory_kind(name: &str) -> MemoryKind {
    let lower = name.to_lowercase();
    if lower.contains("nvidia") {
        MemoryKind::Dedicated
    } else if lower.contains("intel") && !lower.contains("arc") {
        MemoryKind::Shared
    } else {
        MemoryKind::Unknown
    }
}

fn yes_no(value: bool) -> &'static str {
    if value { "Yes" } else { "No" }
}

pub fn format_report(cpu: &CpuInfo, gpus: &[GpuInfo]) -> String {
    let mut out = String::new();
    out.push_str("CPU\n");
    out.push_str(&format!("  Model            : {}\n", cpu.brand));
    if !cpu.vendor.is_empty() {
        out.push_str(&format!("  Vendor           : {}\n", cpu.vendor));
    }
    out.push_str(&format!("  Architecture     : {}\n", cpu.arch));
    out.push_str(&format!(
        "  Physical cores   : {}\n",
        cpu.physical_cores
            .map(|c| c.to_string())
            .unwrap_or_else(|| "unknown".to_string())
    ));
    out.push_str(&format!("  Logical cores    : {}\n", cpu.logical_cores));
    if cpu.frequency_mhz > 0 {
        out.push_str(&format!(
            "  Frequency        : {:.2} GHz\n",
            cpu.frequency_mhz as f64 / 1000.0
        ));
    }
    out.push_str(&format!(
        "  Memory total     : {}\n",
        format_bytes(cpu.total_memory_bytes)
    ));
    out.push_str(&format!(
        "  Memory available : {}\n",
        format_bytes(cpu.available_memory_bytes)
    ));
    out.push_str(&format!(
        "  SSE4.2           : {}\n",
        yes_no(cpu.features.sse4_2)
    ));
    out.push_str(&format!(
        "  AVX2             : {}\n",
        yes_no(cpu.features.avx2)
    ));
    out.push_str(&format!(
        "  AVX512           : {}\n",
        yes_no(cpu.features.avx512f)
    ));

    out.push_str("\nGPU\n");
    if gpus.is_empty() {
        out.push_str("  No GPU detected\n");
    }
    for (index, gpu) in gpus.iter().enumerate() {
        if index > 0 {
            out.push('\n');
        }
        // Skip the vendor prefix when the device name already leads with it
        // (nvidia-smi reports "NVIDIA GeForce ..."), so the line doesn't
        // read "NVIDIA NVIDIA GeForce ...".
        let redundant_vendor = gpu.vendor.is_empty()
            || gpu
                .name
                .to_lowercase()
                .starts_with(&gpu.vendor.to_lowercase());
        out.push_str(&format!(
            "  [{index}] {}{}\n",
            if redundant_vendor {
                String::new()
            } else {
                format!("{} ", gpu.vendor)
            },
            gpu.name
        ));
        out.push_str(&format!(
            "      Memory type  : {}\n",
            gpu.memory_kind.label()
        ));
        out.push_str(&format!(
            "      VRAM total   : {}\n",
            gpu.vram_total_bytes
                .map(format_bytes)
                .unwrap_or_else(|| "n/a".to_string())
        ));
        if let Some(used) = gpu.vram_used_bytes {
            out.push_str(&format!("      VRAM used    : {}\n", format_bytes(used)));
        }
        if let Some(driver) = &gpu.driver {
            out.push_str(&format!("      Driver       : {driver}\n"));
        }
    }

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

    #[test]
    fn apply_shared_memory_total_overrides_only_shared_gpus() {
        let mut gpus = vec![
            gpu(MemoryKind::Dedicated, Some(4 * 1024 * 1024 * 1024)),
            gpu(MemoryKind::Shared, Some(512 * 1024 * 1024)),
            gpu(MemoryKind::Unknown, None),
        ];
        let system_ram = 64 * 1024 * 1024 * 1024;

        apply_shared_memory_total(&mut gpus, system_ram);

        assert_eq!(gpus[0].vram_total_bytes, Some(4 * 1024 * 1024 * 1024));
        assert_eq!(gpus[1].vram_total_bytes, Some(system_ram));
        assert_eq!(gpus[2].vram_total_bytes, None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_memory_kind_follows_vram_vendor_file_presence() {
        // Verified against real hardware carrying both a discrete AMD card
        // (Navi 14, has `mem_info_vram_vendor`) and its integrated AMD APU
        // (Renoir, doesn't) on the same machine.
        assert_eq!(linux_memory_kind(true), MemoryKind::Dedicated);
        assert_eq!(linux_memory_kind(false), MemoryKind::Shared);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_memory_kind_prefers_dedicated_key_then_shared_then_unknown() {
        let dedicated = serde_json::json!({"spdisplays_vram": "8 GB"});
        assert_eq!(macos_memory_kind(&dedicated), MemoryKind::Dedicated);

        let shared = serde_json::json!({"spdisplays_vram_shared": "spdisplays_unified"});
        assert_eq!(macos_memory_kind(&shared), MemoryKind::Shared);

        let neither = serde_json::json!({"_name": "Some GPU"});
        assert_eq!(macos_memory_kind(&neither), MemoryKind::Unknown);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn parse_size_string_handles_common_units() {
        assert_eq!(parse_size_string("8 GB"), Some(8 * 1024 * 1024 * 1024));
        assert_eq!(parse_size_string("1536 MB"), Some(1536 * 1024 * 1024));
        assert_eq!(parse_size_string("not a size"), None);
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn windows_memory_kind_guesses_from_the_adapter_name() {
        assert_eq!(
            windows_memory_kind("NVIDIA GeForce RTX 4090"),
            MemoryKind::Dedicated
        );
        assert_eq!(
            windows_memory_kind("Intel(R) Iris(R) Xe Graphics"),
            MemoryKind::Shared
        );
        assert_eq!(
            windows_memory_kind("Intel(R) Arc(R) A770"),
            MemoryKind::Unknown
        );
        assert_eq!(
            windows_memory_kind("AMD Radeon(TM) Graphics"),
            MemoryKind::Unknown
        );
    }
}
