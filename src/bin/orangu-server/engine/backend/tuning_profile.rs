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
//! Every device's profile is loaded. While a device is being brought up —
//! between [`install`] and the end of its scope — [`var`] reads that
//! device's profile, so the knobs a backend reads into its own instance
//! follow its own card on a split model. A knob kept process-wide (a
//! `static` read once) takes the profile of the device that first read it:
//! the first device's, in practice.
//!
//! Every tuning variable reads through [`var`], [`env_var`] or [`var_os`] —
//! the boolean flags too, through `engine::env` — so a profile can set any
//! of them. A variable read before the first device is installed sees the
//! environment alone; when the profile then sets it, start-up says so.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

/// One device's loaded profile.
struct Profile {
    path: PathBuf,
    values: HashMap<String, String>,
}

/// Every installed device's profile by [`file_name`] key, `None` where the
/// device has none. Leaked: a handful per process, alive for all of it.
static PROFILES: Mutex<Vec<(String, Option<&'static Profile>)>> = Mutex::new(Vec::new());

/// The first device's profile — what a read outside any device's bring-up
/// sees.
static PROFILE: OnceLock<Option<&'static Profile>> = OnceLock::new();

/// The device's identity as [`file_name`] spells it, without the extension —
/// what anything kept per device is keyed by.
static DEVICE_KEY: OnceLock<String> = OnceLock::new();

/// Names read before the first profile was installed — see [`value`].
static EARLY: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Whether reads are still recorded into [`EARLY`]: until the first profile
/// is installed, or start-up says it is over ([`settled`]) — so a process
/// that never installs one (the CPU backend) pays nothing per read after
/// start-up.
static RECORDING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Start-up is over: later reads are not recorded as early ones.
pub fn settled() {
    RECORDING.store(false, std::sync::atomic::Ordering::Relaxed);
}

thread_local! {
    /// The device being brought up on this thread, if any — see [`Scope`].
    static CURRENT: std::cell::Cell<Option<Option<&'static Profile>>> =
        const { std::cell::Cell::new(None) };
}

/// While alive, [`var`] on this thread reads the device's profile that
/// [`install`] returned it for. Dropped at the end of the device's bring-up.
#[must_use = "the device's profile applies only while the scope is alive"]
pub struct Scope {
    previous: Option<Option<&'static Profile>>,
}

impl Drop for Scope {
    fn drop(&mut self) {
        CURRENT.with(|c| c.set(self.previous));
    }
}

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

/// Loads the profile for this adapter, and applies it on this thread until
/// the returned scope ends.
pub fn install(info: &wgpu::AdapterInfo) -> Scope {
    install_file(file_name(
        info.vendor,
        info.device,
        &info.driver,
        &info.driver_info,
    ))
}

/// [`install`] for a device of another API — see [`named_file_name`].
pub fn install_named(api: &str, device: &str, driver: &str) -> Scope {
    install_file(named_file_name(api, device, driver))
}

fn install_file(name: String) -> Scope {
    let key = name.trim_end_matches(".json").to_string();
    let profile = {
        let mut profiles = PROFILES.lock().unwrap_or_else(|e| e.into_inner());
        match profiles.iter().find(|(k, _)| *k == key) {
            Some((_, profile)) => *profile,
            None => {
                let profile = (!disabled())
                    .then(|| dir().and_then(|d| load(d.join(&name))))
                    .flatten()
                    .map(|p| &*Box::leak(Box::new(p)));
                profiles.push((key.clone(), profile));
                profile
            }
        }
    };
    let first = DEVICE_KEY.get_or_init(|| key.clone());
    if *first == key {
        PROFILE.get_or_init(|| {
            if let Some(profile) = profile {
                warn_early_reads(profile);
            }
            settled();
            profile
        });
    } else if let Some(profile) = profile {
        log::info!(
            "orangu-server: tuning profile {} applies to this device's own knobs; \
             process-wide ones keep the first device's ({})",
            profile.path.display(),
            loaded_path().map_or_else(|| "none".to_string(), |p| p.display().to_string())
        );
    }
    Scope {
        previous: CURRENT.with(|c| c.replace(Some(profile))),
    }
}

/// Says which of `profile`'s values were read before it was installed, and
/// so are not in effect.
fn warn_early_reads(profile: &Profile) {
    let early = EARLY.lock().unwrap_or_else(|e| e.into_inner());
    let missed = early_and_set(&early, profile);
    if !missed.is_empty() {
        log::warn!(
            "orangu-server: tuning profile {} sets {} — read before the profile was \
             installed, so this run uses the environment or the default for them",
            profile.path.display(),
            missed.join(", ")
        );
    }
}

/// The names in `early` that `profile` sets, in order, once each.
fn early_and_set(early: &[String], profile: &Profile) -> Vec<String> {
    let mut missed: Vec<String> = early
        .iter()
        .filter(|n| profile.values.contains_key(n.as_str()))
        .cloned()
        .collect();
    missed.dedup();
    missed
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

/// `name` from the profile alone: the device being brought up on this
/// thread, else the first device's.
pub fn value(name: &str) -> Option<&'static str> {
    let profile = match CURRENT.with(std::cell::Cell::get) {
        Some(current) => current,
        None => match PROFILE.get() {
            Some(first) => *first,
            None => {
                if RECORDING.load(std::sync::atomic::Ordering::Relaxed) {
                    let mut early = EARLY.lock().unwrap_or_else(|e| e.into_inner());
                    if !early.iter().any(|n| n == name) {
                        early.push(name.to_string());
                    }
                }
                return None;
            }
        },
    };
    profile?.values.get(name).map(String::as_str)
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

    /// Every `ORANGU_*` read under `engine` goes through the
    /// profile's readers, so a profile can set it — except the variables
    /// below, which select a test, a probe or a diagnostic rather than tune a
    /// kernel.
    #[test]
    fn every_engine_tuning_variable_reads_through_the_profile() {
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
            // Instruments: they report on a run rather than tune it.
            "ORANGU_DECODE_STAGES",
            "ORANGU_GPU_TRACE",
            "ORANGU_MOE_TIME",
            "ORANGU_NPU_DUMP_ACT",
            "ORANGU_EXPERT_USAGE",
            // A quality cap the configuration also sets, not a kernel choice.
            "ORANGU_IMAGE_REFERENCE",
        ];
        let root =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/bin/orangu-server/engine");
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
                if line.trim_start().starts_with("//") {
                    continue;
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

    fn leaked(values: &[(&str, &str)]) -> &'static super::Profile {
        Box::leak(Box::new(super::Profile {
            path: "test.json".into(),
            values: values
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        }))
    }

    /// While a device is brought up, its own profile answers on that thread
    /// alone, and the answer ends with the scope.
    #[test]
    fn a_devices_profile_answers_during_its_bring_up() {
        let second = leaked(&[("ORANGU_SCOPE_TEST_KNOB", "7")]);
        assert_eq!(super::value("ORANGU_SCOPE_TEST_KNOB"), None);
        {
            let _scope = super::Scope {
                previous: super::CURRENT.with(|c| c.replace(Some(Some(second)))),
            };
            assert_eq!(super::value("ORANGU_SCOPE_TEST_KNOB"), Some("7"));
            let elsewhere = std::thread::spawn(|| super::value("ORANGU_SCOPE_TEST_KNOB"));
            assert_eq!(elsewhere.join().unwrap(), None);
        }
        assert_eq!(super::value("ORANGU_SCOPE_TEST_KNOB"), None);
    }

    /// A knob read before the profile arrived is named when the profile
    /// sets it, and only then.
    #[test]
    fn a_knob_read_too_early_is_named() {
        let profile = leaked(&[("A", "1"), ("C", "3")]);
        let early = ["A".to_string(), "B".to_string(), "C".to_string()];
        assert_eq!(super::early_and_set(&early, profile), ["A", "C"]);
        assert!(super::early_and_set(&["B".to_string()], profile).is_empty());
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
