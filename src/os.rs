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

//! Operating-system inventory: what the machine runs, how long it's been
//! running it, and the OS-level settings that decide how a model server
//! actually behaves there — swap, huge-page policy, page size, the
//! open-file limit.
//!
//! Printed as the first section of [`crate::hardware::format_report`] (the
//! `orangu-server system` report, and the same report every attached server
//! prints at startup), ahead of the CPU and GPU sections: which OS this is
//! frames how everything below it should be read — a `Shared` GPU on a
//! 16 KiB-page Apple Silicon Mac behaves nothing like one on x86 Linux.
//!
//! Everything here is read through Rust APIs — [`sysinfo`] for the portable
//! fields, `libc` for the POSIX ones, plain file reads for the Linux
//! `procfs`/`sysfs` ones — never by shelling out to `uname`/`sysctl`/
//! PowerShell. Every field is best-effort: anything a
//! platform can't answer is [`None`] and simply doesn't get a line, rather
//! than being guessed at or reported as an error.

use crate::format::format_bytes;
use std::path::{Path, PathBuf};
use sysinfo::{MemoryRefreshKind, RefreshKind, System};

/// One machine's OS-level inventory. Field availability differs per platform
/// — see [`detect`].
pub struct OsInfo {
    /// The OS/distribution name: `Fedora Linux`, `Darwin`, `Windows`.
    pub name: String,
    /// The OS/distribution version: `44`, `15.1`, `11`.
    pub version: Option<String>,
    /// The verbose name, which on macOS/Windows carries what `name`/`version`
    /// can't: the release codename (`MacOS 15.1 Sequoia`) or the edition
    /// (`Windows 11 Pro`).
    pub long_version: Option<String>,
    /// Kernel release (`7.1.3-200.fc44.x86_64`) — on Windows, the build
    /// number.
    pub kernel: Option<String>,
    /// `/etc/os-release`'s `ID` on Linux (`fedora`, `debian`, ...), and
    /// `darwin`/`windows` elsewhere.
    pub distribution_id: String,
    pub hostname: Option<String>,
    /// The physical machine's vendor and product name, from SMBIOS/DMI on
    /// Linux and `hw.model` on macOS. Unavailable on Windows.
    pub machine: Option<String>,
    pub uptime_seconds: u64,
    /// 1/5/15-minute run-queue averages. [`None`] on Windows, which has no
    /// equivalent metric.
    pub load_average: Option<LoadAverage>,
    pub swap_total_bytes: u64,
    pub swap_used_bytes: u64,
    /// Linux's transparent-hugepage policy (`always`/`madvise`/`never`) —
    /// the currently selected one, not the whole menu.
    pub transparent_hugepages: Option<String>,
    pub page_size_bytes: Option<u64>,
    /// `RLIMIT_NOFILE` as (soft, hard); [`None`] on Windows, which has no
    /// process-wide descriptor limit to report.
    pub open_files_limit: Option<(u64, u64)>,
    /// The target this binary was *built* for (`x86_64-linux-gnu`), which
    /// isn't always the machine it's running on: an `x86_64` build on an
    /// `aarch64` Mac is running under Rosetta, and a `gnu` build won't run
    /// on a musl-only host at all.
    pub build_target: String,
    /// The models directory and the disk it sits on, when one is known —
    /// [`detect`] can't find it out on its own (it's a configured path, not
    /// a machine property), so callers that have a config fill this in with
    /// [`detect_model_storage`] and the rest leave it [`None`].
    pub models: Option<ModelStorage>,
}

pub struct LoadAverage {
    pub one: f64,
    pub five: f64,
    pub fifteen: f64,
}

/// How much room the downloaded models take, and how much is left for the
/// next one — the two numbers that decide whether a `download` will fit,
/// reported for the configured `[orangu-server].models` directory.
pub struct ModelStorage {
    /// The models directory itself, so the two sizes below say what they're
    /// about: a machine can have several, one per config.
    pub path: PathBuf,
    /// Bytes occupied by everything under [`path`](Self::path) — not just
    /// the `.gguf` files `list` counts, since a half-finished download or a
    /// repo's metadata takes the same disk from the same filesystem.
    pub used_bytes: u64,
    /// Free space on the filesystem holding the models directory, from
    /// [`available_space`] — [`None`] where that can't be answered.
    pub available_bytes: Option<u64>,
}

/// Measures `dir` as a models directory: what it holds and what's left
/// beside it. [`None`] when `dir` isn't a directory at all — a models path
/// that was configured but never created has nothing to say about disk use,
/// and reporting `0 B` used would read as "empty" rather than "not there".
pub fn detect_model_storage(dir: &Path) -> Option<ModelStorage> {
    if !dir.is_dir() {
        return None;
    }
    Some(ModelStorage {
        path: dir.to_path_buf(),
        used_bytes: directory_size(dir),
        available_bytes: available_space(dir),
    })
}

/// Bytes on disk under `dir`, counting each physical file exactly once —
/// which a naive walk wouldn't: a model cache names one downloaded blob from
/// every snapshot revision that shares it, so symlinks are not followed
/// (their targets are already visited under `blobs/`), and hard links to one
/// blob are collapsed by `(device, inode)`. Unreadable entries are skipped
/// rather than failing the walk: an under-reported total is worth more here
/// than no total at all.
fn directory_size(dir: &Path) -> u64 {
    #[cfg(unix)]
    let mut seen = std::collections::HashSet::new();
    let mut total = 0u64;

    for entry in walkdir::WalkDir::new(dir)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_file())
    {
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            if !seen.insert((metadata.dev(), metadata.ino())) {
                continue;
            }
        }
        total = total.saturating_add(metadata.len());
    }

    total
}

/// Collects everything [`OsInfo`] can hold on this platform. Portable fields
/// come from `sysinfo`; the rest is filled in per-platform below and left
/// [`None`] where a platform has no answer.
pub fn detect() -> OsInfo {
    let mut sys = System::new_with_specifics(
        RefreshKind::nothing().with_memory(MemoryRefreshKind::everything()),
    );
    sys.refresh_memory();

    OsInfo {
        name: System::name().unwrap_or_else(|| std::env::consts::OS.to_string()),
        version: System::os_version(),
        long_version: System::long_os_version(),
        kernel: System::kernel_version(),
        distribution_id: System::distribution_id(),
        hostname: System::host_name(),
        machine: detect_machine(),
        uptime_seconds: System::uptime(),
        load_average: detect_load_average(),
        swap_total_bytes: sys.total_swap(),
        swap_used_bytes: sys.used_swap(),
        transparent_hugepages: detect_transparent_hugepages(),
        page_size_bytes: detect_page_size(),
        open_files_limit: detect_open_files_limit(),
        build_target: build_target(),
        models: None,
    }
}

/// `sysinfo`'s load average is documented as not working on Windows (it
/// returns zeros there), so that platform reports nothing rather than three
/// convincing-looking `0.00`s.
#[cfg(not(target_os = "windows"))]
fn detect_load_average() -> Option<LoadAverage> {
    let avg = System::load_average();
    Some(LoadAverage {
        one: avg.one,
        five: avg.five,
        fifteen: avg.fifteen,
    })
}

#[cfg(target_os = "windows")]
fn detect_load_average() -> Option<LoadAverage> {
    None
}

/// The build's own target triple, reassembled from the `cfg` values the
/// compiler exposes — there's no constant holding the triple itself, and
/// `std::env::consts::ARCH`/`OS` plus the target env is the same information.
fn build_target() -> String {
    let env = match () {
        _ if cfg!(target_env = "gnu") => "-gnu",
        _ if cfg!(target_env = "musl") => "-musl",
        _ if cfg!(target_env = "msvc") => "-msvc",
        _ => "",
    };
    format!("{}-{}{env}", std::env::consts::ARCH, std::env::consts::OS)
}

// ---------------------------------------------------------------- POSIX ---

/// `sysconf(_SC_PAGESIZE)`, the page size the kernel actually maps with —
/// 4 KiB on most machines, but 16 KiB on Apple Silicon and configurable to
/// 64 KiB on ARM64 Linux, which changes how a memory-mapped model file is
/// paged in.
#[cfg(unix)]
fn detect_page_size() -> Option<u64> {
    let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    (size > 0).then_some(size as u64)
}

#[cfg(not(unix))]
fn detect_page_size() -> Option<u64> {
    None
}

/// `RLIMIT_NOFILE`, the cap on concurrently open descriptors: a server
/// holding a listener, every accepted connection, and one mapping per model
/// shard runs into this one first. Windows has no process-wide equivalent
/// to report.
#[cfg(unix)]
// `rlim_t` is `u64` on Linux and macOS, where the casts below are the no-ops
// clippy sees, but it isn't everywhere `#[cfg(unix)]` covers (it's `i64` on
// the BSDs, `c_ulong` on some 32-bit targets) — so the casts stay.
#[allow(clippy::unnecessary_cast)]
fn detect_open_files_limit() -> Option<(u64, u64)> {
    let mut limit = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    let rc = unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) };
    (rc == 0).then_some((limit.rlim_cur as u64, limit.rlim_max as u64))
}

#[cfg(not(unix))]
fn detect_open_files_limit() -> Option<(u64, u64)> {
    None
}

// ---------------------------------------------------------------- Linux ---

/// SMBIOS/DMI's vendor and product strings, the same pair `dmidecode` prints
/// — world-readable in `sysfs`, unlike the serial/UUID attributes next to
/// them (root-only, and not read here). Absent on machines with no DMI at
/// all, e.g. most ARM boards.
#[cfg(target_os = "linux")]
fn detect_machine() -> Option<String> {
    let vendor = read_trimmed("/sys/class/dmi/id/sys_vendor");
    let product = read_trimmed("/sys/class/dmi/id/product_name");
    machine_label(vendor.as_deref(), product.as_deref())
}

/// Joins the vendor and product strings, dropping the vendor when the
/// product name already starts with it (`LENOVO` + `LENOVO ThinkPad` would
/// otherwise read twice) and tolerating either half being missing.
#[cfg(any(target_os = "linux", test))]
fn machine_label(vendor: Option<&str>, product: Option<&str>) -> Option<String> {
    match (vendor, product) {
        (Some(vendor), Some(product)) => {
            if product.to_lowercase().starts_with(&vendor.to_lowercase()) {
                Some(product.to_string())
            } else {
                Some(format!("{vendor} {product}"))
            }
        }
        (Some(single), None) | (None, Some(single)) => Some(single.to_string()),
        (None, None) => None,
    }
}

/// The kernel's transparent-hugepage policy, which decides whether a large
/// model's mapped weights get 2 MiB pages (fewer TLB misses per token) or
/// 4 KiB ones. The file lists every mode with the active one in brackets —
/// `always [madvise] never` — so only the bracketed one is reported.
#[cfg(target_os = "linux")]
fn detect_transparent_hugepages() -> Option<String> {
    let contents = read_trimmed("/sys/kernel/mm/transparent_hugepage/enabled")?;
    parse_selected_option(&contents).map(str::to_string)
}

#[cfg(any(target_os = "linux", test))]
fn parse_selected_option(contents: &str) -> Option<&str> {
    contents
        .split_whitespace()
        .find_map(|token| token.strip_prefix('[')?.strip_suffix(']'))
}

#[cfg(target_os = "linux")]
fn read_trimmed(path: &str) -> Option<String> {
    let contents = std::fs::read_to_string(path).ok()?;
    let trimmed = contents.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

// ---------------------------------------------------------------- macOS ---

/// `hw.model` is the Mac's model identifier (`Mac16,6`, `MacBookPro18,3`) —
/// the string Apple's own support pages are keyed by.
#[cfg(target_os = "macos")]
fn detect_machine() -> Option<String> {
    sysctl_string("hw.model")
}

/// `sysctlbyname` for a string value, in the usual two calls: one with a
/// null buffer to learn the length, one to fill a buffer of that length.
#[cfg(target_os = "macos")]
fn sysctl_string(name: &str) -> Option<String> {
    let name = std::ffi::CString::new(name).ok()?;
    let mut length: libc::size_t = 0;
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            std::ptr::null_mut(),
            &mut length,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 || length == 0 {
        return None;
    }

    let mut buffer = vec![0u8; length];
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            buffer.as_mut_ptr().cast(),
            &mut length,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 {
        return None;
    }

    buffer.truncate(length);
    // The value comes back NUL-terminated; `String` shouldn't carry that.
    while buffer.last() == Some(&0) {
        buffer.pop();
    }
    let value = String::from_utf8(buffer).ok()?;
    let value = value.trim().to_string();
    (!value.is_empty()).then_some(value)
}

// -------------------------------------------------- other platforms ---

/// Windows has no machine vendor/product a Rust API reaches without a WMI
/// or registry round-trip, so it reports none rather than guessing.
/// Everything else in the report — name, edition, build, hostname, uptime,
/// swap — `sysinfo` provides on Windows just as it does elsewhere.
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn detect_machine() -> Option<String> {
    None
}

/// Transparent hugepages are a Linux concept; no other platform has a
/// policy to report.
#[cfg(not(target_os = "linux"))]
fn detect_transparent_hugepages() -> Option<String> {
    None
}

/// Bytes free on the filesystem holding `path`, as available to *this* user
/// — `f_bavail`, not `f_bfree`, so the root-only reserve (5% of an ext4 by
/// default) isn't counted as room a download can use.
///
/// `None` when the question can't be answered: `path` doesn't exist, the
/// syscall fails, or the platform has no `statvfs` (Windows). A caller that
/// checks free space before a large download has to treat that as "don't
/// know" and carry on, not as "no room".
#[cfg(unix)]
pub fn available_space(path: &Path) -> Option<u64> {
    use std::os::unix::ffi::OsStrExt;

    let c_path = std::ffi::CString::new(path.as_os_str().as_bytes()).ok()?;
    // SAFETY: `c_path` is a valid NUL-terminated string that outlives the
    // call, and `stat` is a live, correctly-sized `statvfs` this initializes
    // before reading.
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::statvfs(c_path.as_ptr(), &mut stat) };
    if rc != 0 {
        return None;
    }
    // `f_frsize` is the fragment size the block counts are in; it's `f_bsize`
    // on Linux but not everywhere, and 0 on some filesystems, where the
    // block size is the one to use.
    let block = if stat.f_frsize > 0 {
        stat.f_frsize
    } else {
        stat.f_bsize
    };
    Some(stat.f_bavail as u64 * block as u64)
}

#[cfg(not(unix))]
pub fn available_space(_path: &Path) -> Option<u64> {
    None
}

#[cfg(all(test, unix))]
mod space_tests {
    /// A real filesystem, asked about through a path that exists: the answer
    /// has to be a number, and a plausible one — a temp dir on a machine
    /// that can build this has room for something, and not more than an
    /// exabyte.
    #[test]
    fn available_space_answers_for_a_real_directory() {
        let dir = tempfile::tempdir().expect("temp dir");

        let free = super::available_space(dir.path()).expect("statvfs");

        assert!(free > 0, "a usable temp dir with no free space at all");
        assert!(free < 1 << 60, "implausible free space: {free}");
    }

    /// A path that isn't there can't be measured, and says so rather than
    /// claiming zero — which a caller would read as "no room".
    #[test]
    fn a_missing_path_is_unknown_rather_than_zero() {
        let dir = tempfile::tempdir().expect("temp dir");

        assert_eq!(super::available_space(&dir.path().join("nope")), None);
    }

    /// The models total is the bytes actually on the disk: a cached repo's
    /// one blob, counted once, however many snapshot revisions name it —
    /// which is the layout every Hugging Face download produces.
    #[test]
    fn a_blob_named_from_two_snapshots_counts_once() {
        let dir = tempfile::tempdir().expect("temp dir");
        let blobs = dir.path().join("blobs");
        std::fs::create_dir_all(&blobs).expect("blobs dir");
        let blob = blobs.join("sha256-abc");
        std::fs::write(&blob, vec![0u8; 4096]).expect("blob");
        for revision in ["rev1", "rev2"] {
            let snapshot = dir.path().join("snapshots").join(revision);
            std::fs::create_dir_all(&snapshot).expect("snapshot dir");
            std::os::unix::fs::symlink(&blob, snapshot.join("model.gguf")).expect("symlink");
        }

        let storage = super::detect_model_storage(dir.path()).expect("models dir");

        assert_eq!(storage.used_bytes, 4096);
        assert_eq!(storage.path, dir.path());
        assert!(storage.available_bytes.is_some_and(|free| free > 0));
    }

    /// A configured models directory that was never created has no disk use
    /// to report — and reports none, rather than an empty-looking `0 B`.
    #[test]
    fn a_models_directory_that_does_not_exist_is_not_reported() {
        let dir = tempfile::tempdir().expect("temp dir");

        assert!(super::detect_model_storage(&dir.path().join("nope")).is_none());
    }
}

// ------------------------------------------------------------ reporting ---

/// Formats the `OS` section of the `system` report: one `label : value`
/// line per field the platform could answer, in the same column layout the
/// CPU and GPU sections use. Fields that came back [`None`] are left out
/// entirely — a line saying `unknown` is a line the reader has to parse to
/// learn nothing.
pub fn format_section(os: &OsInfo) -> String {
    let mut out = String::from("OS\n");
    let mut field = |label: &str, value: &str| {
        out.push_str(&format!("  {label:<17}: {value}\n"));
    };

    field("Name", &os.name);
    if let Some(version) = &os.version {
        field("Version", version);
    }
    // Skipped when it's only the two fields above run together, which is all
    // it is on Linux; on macOS/Windows it adds the codename or edition.
    if let Some(long_version) = &os.long_version
        && !is_redundant_long_version(long_version, &os.name, os.version.as_deref())
    {
        field("Full name", long_version);
    }
    if let Some(kernel) = &os.kernel {
        field("Kernel", kernel);
    }
    // `windows`/`darwin` next to a `Name` of `Windows`/`Darwin` says nothing
    // twice; a distribution `ID` of `fedora` next to `Fedora Linux` is the
    // form other tooling keys on, so it stays.
    if !os.distribution_id.eq_ignore_ascii_case(&os.name) {
        field("Distribution", &os.distribution_id);
    }
    if let Some(machine) = &os.machine {
        field("Machine", machine);
    }
    if let Some(hostname) = &os.hostname {
        field("Hostname", hostname);
    }
    field("Uptime", &format_uptime(os.uptime_seconds));
    if let Some(load) = &os.load_average {
        field(
            "Load average",
            &format!("{:.2}, {:.2}, {:.2}", load.one, load.five, load.fifteen),
        );
    }
    // A machine with no swap configured at all gets no swap lines rather
    // than two zeroes.
    if os.swap_total_bytes > 0 {
        field("Swap total", &format_bytes(os.swap_total_bytes));
        field("Swap used", &format_bytes(os.swap_used_bytes));
    }
    if let Some(hugepages) = &os.transparent_hugepages {
        field("Huge pages", hugepages);
    }
    if let Some(page_size) = os.page_size_bytes {
        field("Page size", &format_bytes(page_size));
    }
    if let Some((soft, hard)) = os.open_files_limit {
        field(
            "Open files",
            &format!("{} (max {})", format_limit(soft), format_limit(hard)),
        );
    }
    // Only where a models directory is known — `suggest` and the report a
    // machine with no config prints have no directory to measure, and say
    // nothing about disk rather than measuring some default that isn't in
    // use. The free-space line is the filesystem's, not the directory's:
    // it's what the *next* download has to fit into.
    if let Some(models) = &os.models {
        field("Models", &models.path.display().to_string());
        field("Models used", &format_bytes(models.used_bytes));
        if let Some(available) = models.available_bytes {
            field("Models free", &format_bytes(available));
        }
    }
    field("Built for", &os.build_target);

    out
}

/// `RLIM_INFINITY` comes back as `u64::MAX`; printing that number is worse
/// than printing what it means.
fn format_limit(limit: u64) -> String {
    if limit == u64::MAX {
        "unlimited".to_string()
    } else {
        limit.to_string()
    }
}

/// Whether `long_version` is just `name` and `version` rearranged, with
/// nothing else in it. Compared word by word — ignoring the punctuation
/// `sysinfo` wraps its Linux answer in — so `Linux (Fedora Linux 44)` counts
/// as redundant against a name of `Fedora Linux` and a version of `44`,
/// while `MacOS 15.1 Sequoia` and `Windows 11 Pro` don't: the codename and
/// the edition are exactly the extra words this is looking for.
fn is_redundant_long_version(long_version: &str, name: &str, version: Option<&str>) -> bool {
    fn words(text: &str) -> impl Iterator<Item = String> {
        text.split_whitespace()
            .map(|word| {
                word.trim_matches(|c: char| !c.is_alphanumeric() && c != '.')
                    .to_lowercase()
            })
            .filter(|word| !word.is_empty())
    }

    // "Linux" is the kernel every distribution runs on; `sysinfo` leads its
    // long version with it whether or not the distribution's own name says
    // so, and it's no more informative there than it is in `Name`.
    let known: Vec<String> = words(name)
        .chain(words(version.unwrap_or_default()))
        .chain(std::iter::once("linux".to_string()))
        .collect();
    words(long_version).all(|word| known.contains(&word))
}

/// Uptime as the two largest units that carry information: `12d 04h`,
/// `4h 12m`, `12m 05s`, `45s`. A server's uptime is read for scale ("days,
/// not minutes"), never for precision.
fn format_uptime(seconds: u64) -> String {
    let days = seconds / 86_400;
    let hours = (seconds % 86_400) / 3_600;
    let minutes = (seconds % 3_600) / 60;
    let secs = seconds % 60;

    if days > 0 {
        format!("{days}d {hours:02}h")
    } else if hours > 0 {
        format!("{hours}h {minutes:02}m")
    } else if minutes > 0 {
        format!("{minutes}m {secs:02}s")
    } else {
        format!("{secs}s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn os() -> OsInfo {
        OsInfo {
            name: "Fedora Linux".to_string(),
            version: Some("44".to_string()),
            long_version: Some("Linux 44 Fedora Linux".to_string()),
            kernel: Some("7.1.3-200.fc44.x86_64".to_string()),
            distribution_id: "fedora".to_string(),
            hostname: Some("orangu".to_string()),
            machine: Some("LENOVO ThinkPad X1".to_string()),
            uptime_seconds: 3 * 86_400 + 4 * 3_600,
            load_average: Some(LoadAverage {
                one: 1.2,
                five: 0.98,
                fifteen: 0.75,
            }),
            swap_total_bytes: 8 * 1024 * 1024 * 1024,
            swap_used_bytes: 1024 * 1024 * 1024,
            transparent_hugepages: Some("madvise".to_string()),
            page_size_bytes: Some(4096),
            open_files_limit: Some((1024, 524_288)),
            build_target: "x86_64-linux-gnu".to_string(),
            models: Some(ModelStorage {
                path: PathBuf::from("/srv/models"),
                used_bytes: 42 * 1024 * 1024 * 1024,
                available_bytes: Some(128 * 1024 * 1024 * 1024),
            }),
        }
    }

    /// The section's labels line up in the same column the CPU/GPU sections
    /// use — 17 characters of label, then `: `.
    #[test]
    fn every_field_line_shares_one_column_layout() {
        let section = format_section(&os());
        for line in section.lines().skip(1) {
            assert_eq!(
                line.find(": "),
                Some(19),
                "misaligned value column in {line:?}"
            );
        }
    }

    #[test]
    fn known_fields_are_reported() {
        let section = format_section(&os());
        assert!(section.starts_with("OS\n"), "section:\n{section}");
        assert!(section.contains("Name             : Fedora Linux\n"));
        assert!(section.contains("Kernel           : 7.1.3-200.fc44.x86_64\n"));
        assert!(section.contains("Distribution     : fedora\n"));
        assert!(section.contains("Machine          : LENOVO ThinkPad X1\n"));
        assert!(section.contains("Uptime           : 3d 04h\n"));
        assert!(section.contains("Load average     : 1.20, 0.98, 0.75\n"));
        assert!(section.contains("Swap total       : 8.00 GiB\n"));
        assert!(section.contains("Huge pages       : madvise\n"));
        assert!(section.contains("Page size        : 4.00 KiB\n"));
        assert!(section.contains("Open files       : 1024 (max 524288)\n"));
        assert!(section.contains("Models           : /srv/models\n"));
        assert!(section.contains("Models used      : 42.00 GiB\n"));
        assert!(section.contains("Models free      : 128.00 GiB\n"));
        assert!(section.contains("Built for        : x86_64-linux-gnu\n"));
    }

    /// A models directory on a filesystem that can't be asked for its free
    /// space still reports what it holds — the two numbers are measured
    /// separately, and one being unavailable doesn't hide the other.
    #[test]
    fn models_are_reported_even_when_free_space_is_unknown() {
        let mut os = os();
        os.models = Some(ModelStorage {
            path: PathBuf::from("/srv/models"),
            used_bytes: 1024 * 1024 * 1024,
            available_bytes: None,
        });

        let section = format_section(&os);

        assert!(
            section.contains("Models used      : 1.00 GiB\n"),
            "{section}"
        );
        assert!(!section.contains("Models free"), "{section}");
    }

    /// Every optional field is genuinely optional: a platform that answers
    /// none of them still gets a section, made only of the lines it could
    /// fill.
    #[test]
    fn unavailable_fields_get_no_line_at_all() {
        let mut os = os();
        os.version = None;
        os.long_version = None;
        os.kernel = None;
        os.hostname = None;
        os.machine = None;
        os.load_average = None;
        os.swap_total_bytes = 0;
        os.transparent_hugepages = None;
        os.page_size_bytes = None;
        os.open_files_limit = None;
        os.models = None;

        let section = format_section(&os);

        for absent in [
            "Version",
            "Full name",
            "Kernel",
            "Machine",
            "Hostname",
            "Load average",
            "Swap",
            "Huge pages",
            "Page size",
            "Open files",
            "Models",
        ] {
            assert!(!section.contains(absent), "{absent} in section:\n{section}");
        }
        assert!(section.contains("Name             : Fedora Linux\n"));
        assert!(
            section.contains("Uptime           : "),
            "section:\n{section}"
        );
    }

    /// `RLIMIT_NOFILE`'s hard limit is very often `RLIM_INFINITY`.
    #[test]
    fn an_infinite_open_file_limit_is_named_not_numbered() {
        let mut os = os();
        os.open_files_limit = Some((1024, u64::MAX));
        assert!(
            format_section(&os).contains("Open files       : 1024 (max unlimited)\n"),
            "section:\n{}",
            format_section(&os)
        );
    }

    /// Linux's long version is its name and version rearranged — no line.
    /// macOS's carries the codename and Windows' the edition — both print.
    #[test]
    fn the_long_version_line_is_kept_only_when_it_adds_a_word() {
        // Exactly what `sysinfo` answers on this project's own build box,
        // punctuation included.
        assert!(is_redundant_long_version(
            "Linux (Fedora Linux 44)",
            "Fedora Linux",
            Some("44")
        ));
        assert!(is_redundant_long_version(
            "Linux (Ubuntu 24.04)",
            "Ubuntu",
            Some("24.04")
        ));
        assert!(!is_redundant_long_version(
            "MacOS 15.1 Sequoia",
            "Darwin",
            Some("15.1")
        ));
        assert!(!is_redundant_long_version(
            "Windows 11 Pro",
            "Windows",
            Some("11")
        ));
    }

    #[test]
    fn uptime_is_formatted_as_its_two_largest_units() {
        assert_eq!(format_uptime(45), "45s");
        assert_eq!(format_uptime(12 * 60 + 5), "12m 05s");
        assert_eq!(format_uptime(4 * 3_600 + 12 * 60), "4h 12m");
        assert_eq!(format_uptime(12 * 86_400 + 4 * 3_600 + 12 * 60), "12d 04h");
    }

    #[test]
    fn the_selected_bracketed_option_is_the_one_reported() {
        assert_eq!(
            parse_selected_option("always [madvise] never"),
            Some("madvise")
        );
        assert_eq!(
            parse_selected_option("[always] madvise never"),
            Some("always")
        );
        assert_eq!(parse_selected_option("always madvise never"), None);
    }

    /// The detection path itself has to work on whatever machine the suite
    /// runs on — it reads real files and `libc`, so this is the only place
    /// those paths get exercised.
    #[test]
    fn detection_answers_the_fields_this_platform_has() {
        let os = detect();
        assert!(!os.name.is_empty());
        assert!(!os.build_target.is_empty());
        assert!(!format_section(&os).is_empty());

        #[cfg(unix)]
        {
            assert!(
                os.page_size_bytes
                    .is_some_and(|size| size.is_power_of_two())
            );
            assert!(os.open_files_limit.is_some_and(|(soft, hard)| soft <= hard));
        }
    }
}
