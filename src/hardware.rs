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
//! VRAM, how much has to fall back to CPU/RAM). The OS the machine runs
//! lives in [`crate::os`]; [`format_report`] prints both as one report.
//!
//! GPU detection has no single cross-platform API, so it layers several
//! best-effort sources: `nvidia-smi` for NVIDIA (installed alongside any
//! NVIDIA driver, Linux or Windows), Linux's `/sys/class/drm` for everything
//! else on Linux (AMD, Intel, and any other PCI display device), and native
//! OS tools (`system_profiler` / PowerShell's `Win32_VideoController`) on
//! macOS and Windows. A card that isn't recognized by any source simply
//! doesn't show up — this is inventory, not a hard dependency of anything
//! else `orangu-server` does.

use crate::format::format_bytes;
use crate::os::OsInfo;
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

/// Where the machine is drawing power from right now.
///
/// Worth detecting because it is the one environmental fact that silently
/// changes what the same model on the same machine will do: on battery, both
/// the CPU governor and the GPU power state drop under the platform's own
/// power management, and a sustained decode loop is exactly the workload
/// those are tuned to suppress.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum PowerSource {
    /// Mains — or a machine with no battery at all, which is the same thing
    /// for every decision made from this.
    Mains,
    /// Running down a battery.
    Battery,
    /// No source on this platform could say. Distinct from `Mains` on
    /// purpose: "we did not find out" must not read as "you are plugged in".
    ///
    /// The default, so a `PowerInfo` nobody filled in claims nothing.
    #[default]
    Unknown,
}

/// One temperature sensor, as `sysinfo` reports it.
#[derive(Clone, Debug, PartialEq)]
pub struct ThermalInfo {
    /// The sensor's own name — chip and channel, e.g. `k10temp Tctl`. Left
    /// exactly as the platform gives it: these are searchable strings, and
    /// prettifying them would only make them harder to look up.
    pub label: String,
    pub celsius: f32,
    /// The temperature at which the platform says this component halts or
    /// throttles hard, when it declares one. Most sensors do not.
    pub critical_celsius: Option<f32>,
}

impl ThermalInfo {
    /// How far this sensor is from its own critical threshold, as a fraction
    /// — `1.0` being at it. `None` when the platform declares no threshold,
    /// which is the common case and is not the same as "plenty of room".
    pub fn critical_fraction(&self) -> Option<f32> {
        self.critical_celsius
            .filter(|c| *c > 0.0)
            .map(|c| self.celsius / c)
    }
}

/// Power source, battery charge, and temperature sensors.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PowerInfo {
    pub source: PowerSource,
    /// Battery charge, 0–100, when the machine has a battery that reports
    /// one. `None` on a desktop, and on a laptop whose platform will not say.
    pub battery_percent: Option<u8>,
    /// Sensors that reported a real reading, **warmest first**. Empty where
    /// the platform exposes none, which is normal in a container and on some
    /// virtualised hosts.
    pub thermals: Vec<ThermalInfo>,
}

impl PowerInfo {
    /// The warmest sensor, or `None` where nothing reported.
    pub fn warmest(&self) -> Option<&ThermalInfo> {
        self.thermals.first()
    }
}

/// Detects power source, battery charge, and temperature sensors.
///
/// **Temperatures come from `sysinfo`; the battery does not.** `sysinfo`
/// exposes components and their temperatures on every platform this targets,
/// which is worth having rather than writing three sysfs/SMC/WMI readers by
/// hand — but it has no battery or AC-line API at all, at any version, so
/// that half is per-platform below.
pub fn detect_power() -> PowerInfo {
    let mut thermals: Vec<ThermalInfo> = sysinfo::Components::new_with_refreshed_list()
        .list()
        .iter()
        .filter_map(|component| {
            // A sensor present but not reporting comes back `None`, and one
            // reporting exactly 0 °C is a sensor that is not wired up rather
            // than a component at freezing — the integrated GPU's memory
            // channel does this. Neither belongs in a report about heat.
            let celsius = component.temperature()?;
            (celsius > 0.0).then(|| ThermalInfo {
                label: component.label().to_string(),
                celsius,
                critical_celsius: component.critical().filter(|c| c.is_finite() && *c > 0.0),
            })
        })
        .collect();
    // Warmest first: the only sensor anybody acts on is the one closest to
    // its limit, and a machine can report a dozen.
    thermals.sort_by(|a, b| {
        b.celsius
            .partial_cmp(&a.celsius)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let (source, battery_percent) = detect_power_source();
    PowerInfo {
        source,
        battery_percent,
        thermals,
    }
}

/// Linux: `/sys/class/power_supply`, the same interface every desktop
/// battery indicator reads.
///
/// A `Mains` supply that is `online` settles it outright — that is the AC
/// adapter reporting itself connected, and it is authoritative whatever the
/// battery says. Only with no mains supply online does a battery's own
/// `status` decide, because a laptop that is plugged in but has a full
/// battery reports `Not charging`, which must not read as "on battery".
#[cfg(target_os = "linux")]
fn detect_power_source() -> (PowerSource, Option<u8>) {
    let Ok(entries) = std::fs::read_dir("/sys/class/power_supply") else {
        return (PowerSource::Unknown, None);
    };
    let read = |dir: &std::path::Path, name: &str| -> Option<String> {
        std::fs::read_to_string(dir.join(name))
            .ok()
            .map(|s| s.trim().to_string())
    };

    let mut mains_online = false;
    let mut battery_percent = None;
    let mut battery_discharging = false;

    let mut dirs: Vec<_> = entries.flatten().map(|e| e.path()).collect();
    dirs.sort();
    for dir in dirs {
        match read(&dir, "type").as_deref() {
            Some("Mains") => mains_online |= read(&dir, "online").as_deref() == Some("1"),
            Some("Battery") => {
                battery_percent = battery_percent.or_else(|| {
                    read(&dir, "capacity").and_then(|c| c.parse::<u8>().ok().map(|p| p.min(100)))
                });
                battery_discharging |= read(&dir, "status").as_deref() == Some("Discharging");
            }
            _ => {}
        }
    }

    (
        classify_power_source(mains_online, battery_discharging),
        battery_percent,
    )
}

/// Turns "is an AC line connected" and "is a battery draining" into a
/// [`PowerSource`].
///
/// Split out from the `sysfs` walk above because the walk is unremarkable and
/// this is where the one real trap is: a laptop that is plugged in with a full
/// battery reports its battery `status` as **`Not charging`**, not
/// `Charging`. Deciding from the battery alone would therefore have to treat
/// `Not charging` as mains and `Discharging` as battery, which reads
/// backwards and is easy to get inverted. Asking the AC line first removes
/// the question — an adapter reporting itself online is authoritative,
/// whatever the battery is doing.
///
/// A machine with neither an AC line nor a draining battery is a desktop, a
/// server, or a container, and `Mains` is the right answer for all three:
/// every caller uses this to ask "is my power about to run out", and there
/// the answer is no.
///
/// Compiled into the tests on every platform, not just the one that calls it,
/// so the rule is checked wherever the suite runs.
#[cfg(any(target_os = "linux", test))]
fn classify_power_source(mains_online: bool, battery_discharging: bool) -> PowerSource {
    if !mains_online && battery_discharging {
        PowerSource::Battery
    } else {
        PowerSource::Mains
    }
}

/// macOS: `pmset -g batt`, whose first line names the source
/// (`'AC Power'` / `'Battery Power'`) and whose second carries the
/// percentage. Parsed rather than read from IOKit for the same reason
/// `detect_macos_gpus` shells out to `system_profiler`: no extra dependency,
/// and the tool is present on every install.
#[cfg(target_os = "macos")]
fn detect_power_source() -> (PowerSource, Option<u8>) {
    let Ok(output) = std::process::Command::new("pmset")
        .args(["-g", "batt"])
        .output()
    else {
        return (PowerSource::Unknown, None);
    };
    let text = String::from_utf8_lossy(&output.stdout);
    let source = if text.contains("'AC Power'") {
        PowerSource::Mains
    } else if text.contains("'Battery Power'") {
        PowerSource::Battery
    } else {
        // No battery at all — a desktop Mac prints no source line.
        PowerSource::Mains
    };
    let percent = text
        .split_whitespace()
        .find_map(|word| word.strip_suffix("%;").or_else(|| word.strip_suffix('%')))
        .and_then(|n| n.parse::<u8>().ok())
        .map(|p| p.min(100));
    (source, percent)
}

/// Windows: `GetSystemPowerStatus`, declared here rather than pulled in.
///
/// The rest of this module reaches Windows through PowerShell because WMI is
/// the only way to the data it wants. This is not that: `GetSystemPowerStatus`
/// is a single `kernel32` call filling a six-field struct, and it is *the*
/// canonical answer to "am I on mains" — going through PowerShell instead
/// would spawn a process costing several hundred milliseconds, on the startup
/// path, to learn one byte. The declaration is three lines and the ABI has
/// been fixed since Windows 95, so there is nothing here a crate would carry
/// for us.
#[cfg(target_os = "windows")]
fn detect_power_source() -> (PowerSource, Option<u8>) {
    #[repr(C)]
    #[derive(Default)]
    struct SystemPowerStatus {
        ac_line_status: u8,
        battery_flag: u8,
        battery_life_percent: u8,
        system_status_flag: u8,
        battery_life_time: u32,
        battery_full_life_time: u32,
    }

    unsafe extern "system" {
        fn GetSystemPowerStatus(status: *mut SystemPowerStatus) -> i32;
    }

    let mut status = SystemPowerStatus::default();
    // SAFETY: `GetSystemPowerStatus` writes at most `sizeof(SYSTEM_POWER_STATUS)`
    // bytes into the pointer it is given, and `status` is exactly that layout,
    // owned here, and outlives the call.
    if unsafe { GetSystemPowerStatus(&mut status) } == 0 {
        return (PowerSource::Unknown, None);
    }

    // 255 means the percentage is unknown; anything above 100 is not a
    // percentage either way.
    let percent = (status.battery_life_percent <= 100).then_some(status.battery_life_percent);
    // Bit 7 of `battery_flag` is "no system battery" — a desktop, which is
    // `Mains` for every purpose here whatever the AC line says.
    if status.battery_flag & 128 != 0 {
        return (PowerSource::Mains, None);
    }
    let source = match status.ac_line_status {
        0 => PowerSource::Battery,
        1 => PowerSource::Mains,
        // 255, documented as "unknown status".
        _ => PowerSource::Unknown,
    };
    (source, percent)
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn detect_power_source() -> (PowerSource, Option<u8>) {
    (PowerSource::Unknown, None)
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

/// `spdisplays_vram` means dedicated, `spdisplays_vram_shared` means
/// unified/shared. Some Apple Silicon machines omit both keys (confirmed
/// on an M5 Pro), so `aarch64` treats that as `Shared` too: every Apple
/// Silicon Mac is unified memory, no discrete-VRAM model exists. Intel
/// Macs (`x86_64`) keep the old `Unknown` fallback, since that
/// architecture shipped both integrated and discrete GPUs.
#[cfg(target_os = "macos")]
fn macos_memory_kind(entry: &serde_json::Value) -> MemoryKind {
    if entry.get("spdisplays_vram").is_some() {
        MemoryKind::Dedicated
    } else if entry.get("spdisplays_vram_shared").is_some() || cfg!(target_arch = "aarch64") {
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

/// The CPU's scaling frequency governor, capitalized for display
/// (`Performance`, `Powersave`, `Schedutil`, …), or `None` on a platform or
/// a kernel that does not expose one.
///
/// A *state* rather than an advisory, and reported as one: the server prints
/// it on its startup banner alongside the rest of what it resolved, because
/// the governor decides whether a core holds its clock through the bursty
/// CPU work between GPU submissions — which is most of what decode latency
/// is. `Performance` is the answer that makes a throughput number
/// comparable; anything else is worth seeing *before* reading one, not only
/// when something is already wrong.
///
/// Changing it is `sudo cpupower frequency-set -g performance`, which this
/// process cannot do for itself: the file is root-owned `sysfs`.
#[cfg(target_os = "linux")]
pub fn cpu_governor() -> Option<String> {
    let raw =
        std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").ok()?;
    let raw = raw.trim();
    let mut chars = raw.chars();
    let first = chars.next()?;
    Some(first.to_uppercase().collect::<String>() + chars.as_str())
}

#[cfg(not(target_os = "linux"))]
pub fn cpu_governor() -> Option<String> {
    None
}

/// How close to its critical threshold a sensor has to be before the report
/// says anything.
///
/// Set where it is because the point is to explain *throughput*, not to warn
/// about hardware. Silicon throttles well before it halts, so a component at
/// nine tenths of the temperature at which the platform would shut it down is
/// already losing clock, and that is the thing a reader watching slow tokens
/// needs told. Below it, a warm machine is just a machine doing work.
const THERMAL_ADVISORY_FRACTION: f32 = 0.9;

/// Machine state that will hold throughput down, in words a reader can act
/// on. Empty when there is nothing to say, which is the normal case on a
/// plugged-in machine that is not already hot.
///
/// These are the only advisories left, and they are *conditions*: they hold
/// on every platform, and neither has a command as an answer — one is
/// answered by a cable and the other by airflow. The **settings** that used
/// to print beside them do not, because both have somewhere better to be: a
/// scaling CPU governor is [`cpu_governor`], a banner row with a value on
/// every start, and an AMD card left at `power_dpm_force_performance_level
/// = auto` is documented rather than warned about, since a machine with
/// several cards printed a line per card on every start to say the same
/// thing about each. See `doc/SERVER.md`.
pub fn power_advisories(power: &PowerInfo) -> Vec<String> {
    let mut out = Vec::new();

    if power.source == PowerSource::Battery {
        let charge = power
            .battery_percent
            .map(|p| format!(" ({p}% remaining)"))
            .unwrap_or_default();
        out.push(format!(
            "Running on battery{charge}. Sustained decode is exactly the workload platform power \
             management clocks down, so throughput here is not what this machine does on mains — \
             and a long generation will empty the battery. Plug in before measuring anything."
        ));
    }

    // Only the warmest sensor, however many are close. A reader acts on the
    // hottest component; listing five is a wall of text saying one thing.
    if let Some(hot) = power.thermals.iter().find(|t| {
        t.critical_fraction()
            .is_some_and(|f| f >= THERMAL_ADVISORY_FRACTION)
    }) {
        out.push(format!(
            "{} is at {:.1} °C, against a critical {:.1} °C, before any work has started. \
             Sustained decode will thermally throttle from here, which reads as the engine being \
             slow rather than as the machine being hot.",
            hot.label,
            hot.celsius,
            hot.critical_celsius.unwrap_or_default(),
        ));
    }

    out
}

/// The whole `orangu-server system` report: the OS section (formatted by
/// [`crate::os::format_section`], which owns everything OS-level), then this
/// module's CPU and GPU inventory. The OS comes first because it frames how
/// the two below it should be read — a container memory limit or a WSL2
/// kernel says more about what a model will do on this machine than its core
/// count does.
/// How many sensors the report lists.
///
/// A laptop reports nine and a server can report dozens, nearly all of them
/// telling the reader nothing — an idle NVMe controller is not why decode is
/// slow. Three is enough to show the CPU, the GPU and whatever is unexpectedly
/// above both, which is the whole diagnostic value on offer.
const SENSORS_REPORTED: usize = 3;

/// The `POWER` section: where the machine is drawing from, and how hot it is
/// before any work starts.
///
/// Skipped entirely when there is nothing to say — no sensors and an unknown
/// source, which is the normal state inside a container. An empty heading is
/// worse than no heading.
fn format_power_section(power: &PowerInfo) -> String {
    if power.thermals.is_empty() && power.source == PowerSource::Unknown {
        return String::new();
    }
    let mut out = String::from("\nPOWER\n");
    let source = match power.source {
        PowerSource::Mains => "Mains".to_string(),
        PowerSource::Battery => "Battery".to_string(),
        // Named as the non-answer it is, so nobody reads a blank as "mains".
        PowerSource::Unknown => "Unknown".to_string(),
    };
    let charge = power
        .battery_percent
        .map(|p| format!(" (battery {p}%)"))
        .unwrap_or_default();
    out.push_str(&format!("  Source           : {source}{charge}\n"));

    for sensor in power.thermals.iter().take(SENSORS_REPORTED) {
        // The critical figure only when the platform declares one — printing
        // "critical n/a" on every line would bury the sensors that have one.
        let critical = sensor
            .critical_celsius
            .map(|c| format!(" (critical {c:.1} °C)"))
            .unwrap_or_default();
        out.push_str(&format!(
            "  {:<17}: {:.1} °C{critical}\n",
            sensor.label, sensor.celsius
        ));
    }
    out
}

pub fn format_report(os: &OsInfo, cpu: &CpuInfo, gpus: &[GpuInfo], power: &PowerInfo) -> String {
    let mut out = crate::os::format_section(os);
    out.push('\n');
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

    out.push_str(&format_power_section(power));

    // No GPU at all means no GPU section: a heading over a single "none
    // found" line is two lines of report saying nothing the reader can act
    // on, and it's the CPU inventory above that matters on such a machine.
    if gpus.is_empty() {
        return out;
    }

    out.push_str("\nGPU\n");
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

    /// The governor is a banner *value*, so it is capitalized for display
    /// and reported whatever it says — including `Performance`, which the
    /// advisory it replaced could only communicate by staying silent.
    ///
    /// On a machine with no `cpufreq` (every non-Linux target, and some
    /// virtualized Linux ones) there is no row at all rather than a made-up
    /// one.
    #[test]
    fn the_cpu_governor_is_reported_as_a_capitalized_value_or_not_at_all() {
        // `None` on a box with no cpufreq, where the banner prints no row
        // and there is nothing to assert.
        if let Some(governor) = cpu_governor() {
            assert!(!governor.is_empty());
            assert!(
                governor.starts_with(|c: char| c.is_uppercase()),
                "governor {governor:?} should be capitalized for the banner"
            );
            assert!(
                !governor.ends_with('\n'),
                "governor {governor:?} still carries sysfs's trailing newline"
            );
        }
    }

    /// Every remaining advisory is a *condition* — something no command
    /// answers. The machine settings that used to print beside them are a
    /// banner row and a manual page now, so an advisory naming one would be
    /// saying it twice.
    ///
    /// Checked against a machine that is both on battery and hot, so the
    /// two conditions really are produced and this is not asserting over an
    /// empty list.
    #[test]
    fn advisories_are_conditions_only_and_never_repeat_a_machine_setting() {
        let power = PowerInfo {
            source: PowerSource::Battery,
            battery_percent: Some(31),
            thermals: vec![ThermalInfo {
                label: "k10temp Tctl".to_string(),
                celsius: 95.0,
                critical_celsius: Some(100.0),
            }],
        };
        let advisories = power_advisories(&power);
        assert_eq!(advisories.len(), 2, "{advisories:?}");
        for advisory in &advisories {
            assert!(
                !advisory.contains("frequency governor") && !advisory.contains("power level"),
                "a machine setting leaked back into an advisory: {advisory}"
            );
        }
    }

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

    fn cpu() -> CpuInfo {
        CpuInfo {
            brand: "Test CPU".to_string(),
            vendor: "TestVendor".to_string(),
            arch: "x86_64".to_string(),
            physical_cores: Some(8),
            logical_cores: 16,
            frequency_mhz: 4200,
            total_memory_bytes: 64 * 1024 * 1024 * 1024,
            available_memory_bytes: 32 * 1024 * 1024 * 1024,
            features: CpuFeatures {
                sse4_2: true,
                avx2: true,
                avx512f: false,
            },
        }
    }

    /// A machine with no GPU gets no GPU section at all — not a heading over
    /// a "none found" line. The OS and CPU inventories are the whole report
    /// there, so it also ends without a trailing blank line.
    #[test]
    fn the_gpu_section_is_omitted_when_no_gpu_was_detected() {
        let report = format_report(&crate::os::detect(), &cpu(), &[], &PowerInfo::default());
        assert!(
            !report.contains("\nGPU\n"),
            "unexpected GPU section:\n{report}"
        );
        assert!(report.contains("\nCPU\n"), "report:\n{report}");
        assert!(
            report.ends_with("AVX512           : No\n"),
            "report:\n{report}"
        );
    }

    /// The OS section leads the report — what this machine runs frames how
    /// the CPU and GPU inventories under it should be read.
    #[test]
    fn the_os_section_comes_first() {
        let report = format_report(&crate::os::detect(), &cpu(), &[], &PowerInfo::default());
        assert!(report.starts_with("OS\n"), "report:\n{report}");
        assert!(
            report.find("\nCPU\n") < report.find("\nGPU\n").or(Some(usize::MAX)),
            "report:\n{report}"
        );
    }

    /// The section is still there — heading, blank line before it, and one
    /// indexed block per device — as soon as there's a device to report.
    #[test]
    fn the_gpu_section_is_kept_when_a_gpu_was_detected() {
        let report = format_report(
            &crate::os::detect(),
            &cpu(),
            &[gpu(MemoryKind::Dedicated, Some(4 * 1024 * 1024 * 1024))],
            &PowerInfo::default(),
        );
        assert!(report.contains("\nGPU\n"), "report:\n{report}");
        assert!(report.contains("[0] Test GPU"), "report:\n{report}");
        assert!(
            report.contains("VRAM total   : 4.00 GiB"),
            "report:\n{report}"
        );
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
    fn macos_memory_kind_prefers_dedicated_key_then_shared_key() {
        let dedicated = serde_json::json!({"spdisplays_vram": "8 GB"});
        assert_eq!(macos_memory_kind(&dedicated), MemoryKind::Dedicated);

        let shared = serde_json::json!({"spdisplays_vram_shared": "spdisplays_unified"});
        assert_eq!(macos_memory_kind(&shared), MemoryKind::Shared);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn macos_memory_kind_assumes_shared_on_apple_silicon_without_either_key() {
        let neither = serde_json::json!({"_name": "Apple GPU"});
        assert_eq!(macos_memory_kind(&neither), MemoryKind::Shared);
    }

    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    #[test]
    fn macos_memory_kind_is_unknown_without_either_key_on_intel() {
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

    fn sensor(label: &str, celsius: f32, critical: Option<f32>) -> ThermalInfo {
        ThermalInfo {
            label: label.to_string(),
            celsius,
            critical_celsius: critical,
        }
    }

    /// The one case worth pinning: a laptop that is plugged in with a full
    /// battery reports `Not charging`, not `Charging`.
    ///
    /// Deciding from the battery alone means reading `Not charging` as
    /// "on mains", which is the sort of double negative that gets inverted
    /// and would have every desk-bound laptop told it was running down a
    /// battery. Asking the AC line first is what avoids it, so this asserts
    /// the AC line wins in every combination.
    #[test]
    fn a_connected_adapter_settles_the_power_source_whatever_the_battery_says() {
        // Plugged in: full ("Not charging"), charging, and even the odd case
        // of draining under too weak a charger.
        assert_eq!(classify_power_source(true, false), PowerSource::Mains);
        assert_eq!(classify_power_source(true, true), PowerSource::Mains);
        // Unplugged and draining is the only battery answer.
        assert_eq!(classify_power_source(false, true), PowerSource::Battery);
        // No AC line and nothing draining: a desktop, a server, or a
        // container. Nothing is going to run out, which is what every caller
        // means by mains.
        assert_eq!(classify_power_source(false, false), PowerSource::Mains);
    }

    /// Being on battery is worth interrupting for; being on mains is not.
    #[test]
    fn only_battery_power_raises_an_advisory() {
        let on_battery = PowerInfo {
            source: PowerSource::Battery,
            battery_percent: Some(64),
            thermals: Vec::new(),
        };
        let notes = power_advisories(&on_battery);
        assert_eq!(notes.len(), 1, "{notes:?}");
        assert!(notes[0].contains("battery"), "{notes:?}");
        assert!(notes[0].contains("64%"), "{notes:?}");

        assert!(
            power_advisories(&PowerInfo {
                source: PowerSource::Mains,
                battery_percent: Some(64),
                thermals: Vec::new(),
            })
            .is_empty()
        );
        // An unknown source must not be reported as a problem: not finding
        // out is not the same as finding something wrong.
        assert!(power_advisories(&PowerInfo::default()).is_empty());
    }

    /// A warm machine is a machine doing work; a machine near its own
    /// shutdown threshold is about to lose clock. Only the second is worth a
    /// line, and only against a threshold the platform actually declared.
    #[test]
    fn a_thermal_advisory_needs_a_declared_critical_and_real_proximity() {
        let near = PowerInfo {
            source: PowerSource::Mains,
            battery_percent: None,
            thermals: vec![sensor("gpu junction", 95.0, Some(100.0))],
        };
        let notes = power_advisories(&near);
        assert_eq!(notes.len(), 1, "{notes:?}");
        assert!(notes[0].contains("gpu junction"), "{notes:?}");
        assert!(notes[0].contains("throttle"), "{notes:?}");

        // Warm, but with room — no note.
        assert!(
            power_advisories(&PowerInfo {
                thermals: vec![sensor("gpu junction", 80.0, Some(100.0))],
                source: PowerSource::Mains,
                battery_percent: None,
            })
            .is_empty()
        );
        // Hot, but the platform declared no threshold, so there is no
        // proximity to claim. Most sensors are in this state, and inventing a
        // fixed limit for them would fire on machines that are perfectly fine.
        assert!(
            power_advisories(&PowerInfo {
                thermals: vec![sensor("cpu package", 95.0, None)],
                source: PowerSource::Mains,
                battery_percent: None,
            })
            .is_empty()
        );
    }

    /// However many sensors are close, the reader gets the hottest one. A
    /// machine can report a dozen and they all say the same thing.
    #[test]
    fn the_thermal_advisory_names_one_sensor_not_every_hot_one() {
        let notes = power_advisories(&PowerInfo {
            source: PowerSource::Mains,
            battery_percent: None,
            thermals: vec![
                sensor("hottest", 99.0, Some(100.0)),
                sensor("also hot", 96.0, Some(100.0)),
                sensor("hot too", 95.0, Some(100.0)),
            ],
        });
        assert_eq!(notes.len(), 1, "{notes:?}");
        assert!(notes[0].contains("hottest"), "{notes:?}");
    }

    /// The section reports what it found and disappears when it found
    /// nothing — an empty `POWER` heading is worse than no heading, and a
    /// container reports neither a power source nor a sensor.
    #[test]
    fn the_power_section_is_omitted_when_nothing_is_known() {
        assert!(format_power_section(&PowerInfo::default()).is_empty());

        let section = format_power_section(&PowerInfo {
            source: PowerSource::Battery,
            battery_percent: Some(98),
            thermals: vec![
                sensor("a", 90.0, Some(100.0)),
                sensor("b", 80.0, None),
                sensor("c", 70.0, None),
                sensor("d", 60.0, None),
            ],
        });
        assert!(
            section.contains("Source           : Battery (battery 98%)"),
            "{section}"
        );
        assert!(section.contains("(critical 100.0 °C)"), "{section}");
        // Capped, and it is the coolest that gets dropped.
        assert!(section.contains("a "), "{section}");
        assert!(!section.contains("\n  d "), "{section}");
    }

    /// An unknown source is printed as unknown rather than left blank, so a
    /// reader never takes a missing line for "you are plugged in".
    #[test]
    fn an_unknown_power_source_is_named_rather_than_omitted() {
        let section = format_power_section(&PowerInfo {
            source: PowerSource::Unknown,
            battery_percent: None,
            thermals: vec![sensor("a", 40.0, None)],
        });
        assert!(section.contains("Source           : Unknown"), "{section}");
    }

    /// Whatever this machine is, detection must not panic and must not
    /// report a temperature that is obviously not one.
    #[test]
    fn detect_power_returns_something_sane_on_this_machine() {
        let power = detect_power();
        for sensor in &power.thermals {
            assert!(
                sensor.celsius > 0.0 && sensor.celsius < 200.0,
                "implausible sensor: {sensor:?}"
            );
        }
        // Warmest first, which is what makes `warmest` and the advisory the
        // same sensor.
        for pair in power.thermals.windows(2) {
            assert!(pair[0].celsius >= pair[1].celsius, "{:?}", power.thermals);
        }
        if let Some(percent) = power.battery_percent {
            assert!(percent <= 100);
        }
    }
}
