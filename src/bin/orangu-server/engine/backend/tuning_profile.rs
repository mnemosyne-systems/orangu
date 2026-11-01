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

//! A tuning profile per device and driver.
//!
//! The kernels' parameters are compiled-in defaults that an environment
//! variable can override ([`super::env_tuning_value`]). A profile is the same
//! overrides kept on disk for one device: what a sweep found best on this
//! card, applied on every start without anyone exporting anything.
//!
//! Precedence is the environment, then the profile, then the default — so a
//! sweep that sets a variable still measures the value it set.
//!
//! The file is `~/.orangu/server/tuning/<vendor>-<device>-<driver>.json`:
//!
//! ```json
//! { "values": { "ORANGU_REDUCE_N_ROWS": "2", "ORANGU_COOP_MIN_TOKENS": "32" } }
//! ```
//!
//! A device another API drives (CUDA, ROCm, OpenCL) has no PCI ids to hand,
//! so its file is named by the API, the device's name and the driver's
//! version instead: `cuda-nvidia_geforce_rtx_4090-12040.json`.
//!
//! The driver is part of the name because a driver update changes which
//! choice is fastest; a profile from the previous driver is not applied to
//! the next one. `ORANGU_TUNING_PROFILE=0` ignores the profile.
//!
//! The knobs are process-wide and read once, so one profile applies: the
//! first device's. A second device with a profile of its own is told so at
//! start-up rather than half-applied.
//!
//! Every tuning variable reads through [`var`], [`env_var`] or [`var_os`] —
//! the boolean flags too, through `engine::env` — so a profile can set any
//! of them. A variable read before the first device is installed sees the
//! environment alone.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;

/// What was loaded, once per process.
struct Profile {
    path: PathBuf,
    values: HashMap<String, String>,
}

static PROFILE: OnceLock<Option<Profile>> = OnceLock::new();

/// The device's identity as [`file_name`] spells it, without the extension —
/// what anything kept per device is keyed by.
static DEVICE_KEY: OnceLock<String> = OnceLock::new();

#[derive(serde::Deserialize)]
struct File {
    #[serde(default)]
    values: HashMap<String, serde_json::Value>,
}

/// The profile's file name for a device: vendor and device ids, and the
/// driver's name and version reduced to a file-name-safe slug.
pub fn file_name(vendor: u32, device: u32, driver: &str, driver_info: &str) -> String {
    let slug = slug(&format!("{driver}-{driver_info}"));
    format!("{vendor:04x}-{device:04x}-{slug}.json")
}

/// The profile's file name for a device of another API: the API, the
/// device's name and the driver's version, each a file-name-safe slug.
pub fn named_file_name(api: &str, device: &str, driver: &str) -> String {
    format!("{}-{}-{}.json", slug(api), slug(device), slug(driver))
}

fn slug(text: &str) -> String {
    let slug: String = text
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();
    slug.trim_matches('_').to_string()
}

/// Where profiles live.
pub fn dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".orangu/server/tuning"))
}

/// Loads the profile for this adapter. Only the first call counts: knobs are
/// read once, and a second device's profile arriving later could not change
/// them consistently.
pub fn install(info: &wgpu::AdapterInfo) {
    install_file(file_name(
        info.vendor,
        info.device,
        &info.driver,
        &info.driver_info,
    ));
}

/// [`install`] for a device of another API — see [`named_file_name`].
pub fn install_named(api: &str, device: &str, driver: &str) {
    install_file(named_file_name(api, device, driver));
}

fn install_file(name: String) {
    let key = name.trim_end_matches(".json").to_string();
    let first = DEVICE_KEY.get_or_init(|| key.clone());
    if *first != key {
        // A second device: the knobs were read for the first one's.
        if disabled() {
            return;
        }
        if let Some(path) = dir().map(|d| d.join(&name)).filter(|p| p.exists()) {
            log::warn!(
                "orangu-server: tuning profile {} is not applied: the tuning variables are \
                 read once for the whole process, and the first device's profile ({}) is \
                 the one in use",
                path.display(),
                loaded_path().map_or_else(|| "none".to_string(), |p| p.display().to_string())
            );
        }
        return;
    }
    PROFILE.get_or_init(|| {
        if disabled() {
            return None;
        }
        load(dir()?.join(&name))
    });
}

fn disabled() -> bool {
    std::env::var("ORANGU_TUNING_PROFILE").is_ok_and(|v| v.trim() == "0")
}

fn load(path: PathBuf) -> Option<Profile> {
    let text = std::fs::read_to_string(&path).ok()?;
    let file: File = match serde_json::from_str(&text) {
        Ok(file) => file,
        Err(err) => {
            log::warn!(
                "orangu-server: tuning profile {} is not valid ({err}); ignoring it",
                path.display()
            );
            return None;
        }
    };
    let values: HashMap<String, String> = file
        .values
        .into_iter()
        .map(|(k, v)| {
            let v = match v {
                serde_json::Value::String(s) => s,
                other => other.to_string(),
            };
            (k, v)
        })
        .collect();
    log::info!(
        "orangu-server: tuning profile {} — {} value(s)",
        path.display(),
        values.len()
    );
    Some(Profile { path, values })
}

/// The first device's key — see [`DEVICE_KEY`].
pub fn device_key() -> Option<&'static str> {
    DEVICE_KEY.get().map(String::as_str)
}

/// `var` from the environment, else from the profile.
pub fn var(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .or_else(|| value(name).map(str::to_string))
}

/// [`var`] in the shape of [`std::env::var`], for a reader written against
/// it: `NotPresent` when neither the environment nor the profile sets it.
pub fn env_var(name: &str) -> Result<String, std::env::VarError> {
    match std::env::var(name) {
        Err(std::env::VarError::NotPresent) => value(name)
            .map(str::to_string)
            .ok_or(std::env::VarError::NotPresent),
        other => other,
    }
}

/// [`var`] in the shape of [`std::env::var_os`].
pub fn var_os(name: &str) -> Option<std::ffi::OsString> {
    std::env::var_os(name).or_else(|| value(name).map(std::ffi::OsString::from))
}

/// `name` from the profile alone.
pub fn value(name: &str) -> Option<&'static str> {
    PROFILE
        .get()?
        .as_ref()?
        .values
        .get(name)
        .map(String::as_str)
}

/// The loaded profile's path, for the startup report and `/props`.
pub fn loaded_path() -> Option<&'static std::path::Path> {
    Some(PROFILE.get()?.as_ref()?.path.as_path())
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_file_name_carries_the_driver_version() {
        let a = super::file_name(0x1002, 0x7340, "radv", "Mesa 25.2.1");
        let b = super::file_name(0x1002, 0x7340, "radv", "Mesa 25.3.0");
        assert_eq!(a, "1002-7340-radv_mesa_25.2.1.json");
        assert_ne!(a, b, "a driver update must not reuse the old profile");
    }

    #[test]
    fn another_apis_file_name_carries_its_device_and_driver() {
        assert_eq!(
            super::named_file_name("cuda", "NVIDIA GeForce RTX 4090", "12040"),
            "cuda-nvidia_geforce_rtx_4090-12040.json"
        );
        assert_ne!(
            super::named_file_name("cuda", "NVIDIA GeForce RTX 4090", "12040"),
            super::named_file_name("cuda", "NVIDIA GeForce RTX 4090", "12050")
        );
    }

    /// Every `ORANGU_*` read under `engine::backend` goes through the
    /// profile's readers, so a profile can set it — except the variables
    /// below, which select a test, a probe or a diagnostic rather than tune a
    /// kernel.
    #[test]
    fn every_backend_tuning_variable_reads_through_the_profile() {
        const NOT_TUNING: &[&str] = &[
            "ORANGU_TEST_",
            "ORANGU_BENCH_",
            "ORANGU_SWEEP_",
            "ORANGU_TD_",
            "ORANGU_NORM_PROBE_",
            "ORANGU_MSL_",
            "ORANGU_DUMP_SHADERS",
            "ORANGU_PROBE_TRACE",
            "ORANGU_VK_NO_SELFCHECK",
            "ORANGU_GPU_TIMESTAMPS",
            "ORANGU_CPU_TIMESTAMPS",
            "ORANGU_TUNING_PROFILE",
            "ORANGU_NPU_MODE",
            "ORANGU_Q2K_STUB",
        ];
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/bin/orangu-server/engine/backend");
        let mut files = Vec::new();
        let mut dirs = vec![root];
        while let Some(dir) = dirs.pop() {
            for entry in std::fs::read_dir(&dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    dirs.push(path);
                } else if path.extension().is_some_and(|e| e == "rs")
                    // Test code selects tests and probes, not tunings.
                    && path.file_name().is_some_and(|n| n != "tests.rs")
                {
                    files.push(path);
                }
            }
        }
        let mut direct = Vec::new();
        for file in files {
            let text = std::fs::read_to_string(&file).unwrap();
            for (n, line) in text.lines().enumerate() {
                if line.trim() == "mod tests {" {
                    break;
                }
                for reader in ["std::env::var(\"", "std::env::var_os(\""] {
                    let Some(at) = line.find(reader) else {
                        continue;
                    };
                    let name: String = line[at + reader.len()..]
                        .chars()
                        .take_while(|c| *c != '"')
                        .collect();
                    if name.starts_with("ORANGU_")
                        && !NOT_TUNING.iter().any(|p| name.starts_with(p))
                    {
                        direct.push(format!("{}:{} {name}", file.display(), n + 1));
                    }
                }
            }
        }
        assert!(
            direct.is_empty(),
            "read with std::env, so a tuning profile cannot set them — use \
             tuning_profile::var/env_var/var_os, or env_tuning_value:\n{}",
            direct.join("\n")
        );
    }

    #[test]
    fn a_profile_reads_strings_and_numbers() {
        let dir = std::env::temp_dir().join(format!("orangu-tuning-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("p.json");
        std::fs::write(&path, r#"{"values":{"A":"2","B":32}}"#).unwrap();
        let p = super::load(path).unwrap();
        assert_eq!(p.values["A"], "2");
        assert_eq!(p.values["B"], "32");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
