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

//! Start-up probe answers, kept across starts.
//!
//! Where a token is decoded is measured at every start
//! (`engine::decode_backend`): the model is built on the cores as well and a
//! few decode steps are timed on each — seconds of start-up for a model of a
//! few billion parameters. The answer depends on the device and its driver,
//! the model file, this build's kernels and the thread count, so it is kept
//! under exactly those in `~/.orangu/server/tuning/probes.json` and used
//! again while it is younger than [`MAX_AGE`]; past that it is measured
//! afresh, so a machine whose conditions drift is re-checked.
//!
//! Start-up choices between kernel families made by timing them (whether
//! the decode kernels take the integer dot) are kept the same way, under
//! the same key, and so are the choices made for the whole run — which
//! cores a prompt runs on, whether prompts run on the CPU backend, and the
//! prefill chunk sizer's cost line — under [`model_key`], which names the
//! cores where there is no device, so a run on the cores alone keeps its
//! answers too.
//!
//! The model is named by its content, not its path: a file moved or copied
//! keeps its answers, and one replaced at the same path (another
//! quantization, a re-download) does not inherit them.
//!
//! `ORANGU_PROBE_CACHE=0` measures every time.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// How long a kept answer is trusted.
pub const MAX_AGE: Duration = Duration::from_secs(7 * 24 * 3600);

/// One decode-placement measurement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecodeProbe {
    /// Seconds per one-token decode step on the device.
    pub device_s: f64,
    /// The same on the cores.
    pub cpu_s: f64,
}

fn enabled() -> bool {
    crate::engine::env::flag_on_unless_disabled("ORANGU_PROBE_CACHE")
}

fn file() -> Option<PathBuf> {
    crate::engine::backend::tuning_profile::dir().map(|d| d.join("probes.json"))
}

/// Bytes read from each end of a model file for its fingerprint: the head
/// holds the whole tensor table, the tail the last tensor's data.
const FINGERPRINT_BYTES: u64 = 64 * 1024;

/// A model file's content fingerprint: its length and a hash of its first
/// and last [`FINGERPRINT_BYTES`]. Kept per path, length and modification
/// time, so a file is read once a run and again only once it changes.
fn fingerprint(model: &Path) -> Option<u64> {
    use std::collections::HashMap;
    use std::hash::Hasher;
    use std::io::{Read, Seek, SeekFrom};
    type Seen = HashMap<PathBuf, (u64, SystemTime, u64)>;
    static SEEN: std::sync::Mutex<Option<Seen>> = std::sync::Mutex::new(None);
    let meta = std::fs::metadata(model).ok()?;
    let (len, mtime) = (meta.len(), meta.modified().ok()?);
    let mut seen = SEEN.lock().unwrap_or_else(|p| p.into_inner());
    if let Some(&(l, m, fp)) = seen.as_ref().and_then(|s| s.get(model))
        && l == len
        && m == mtime
    {
        return Some(fp);
    }
    let mut file = std::fs::File::open(model).ok()?;
    let mut hasher = rustc_hash::FxHasher::default();
    hasher.write_u64(len);
    let mut buf = vec![0u8; FINGERPRINT_BYTES.min(len) as usize];
    file.read_exact(&mut buf).ok()?;
    hasher.write(&buf);
    if len > FINGERPRINT_BYTES {
        let tail = FINGERPRINT_BYTES.min(len - FINGERPRINT_BYTES);
        file.seek(SeekFrom::End(-(tail as i64))).ok()?;
        buf.resize(tail as usize, 0);
        file.read_exact(&mut buf).ok()?;
        hasher.write(&buf);
    }
    let fp = hasher.finish();
    seen.get_or_insert_with(HashMap::new)
        .insert(model.to_path_buf(), (len, mtime, fp));
    Some(fp)
}

/// What the answer for `model` on `device` depends on, as one string — or
/// `None` when the file cannot be read.
pub fn key(device: &str, model: &Path) -> Option<String> {
    let fp = fingerprint(model)?;
    // The binary's own time as well as its version: two builds of one commit
    // can carry different kernels.
    let exe = std::env::current_exe()
        .and_then(std::fs::metadata)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_secs());
    Some(format!(
        "{device}|{fp:016x}|{} {exe}|{}",
        orangu::build_info::id(),
        rayon::current_num_threads()
    ))
}

/// The cores, named for a run with no device: the architecture and, where
/// the system says, the processor's model name.
fn host_key() -> String {
    let name = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split_once(':'))
                .map(|(_, v)| v.trim().to_string())
        })
        .unwrap_or_default();
    let slug: String = format!("cpu-{}-{name}", std::env::consts::ARCH)
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();
    slug.trim_end_matches('_').to_string()
}

/// The model this process serves, for [`model_key`].
static MODEL: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

/// Names the model this process serves — once, before it is built.
pub fn set_model(path: &Path) {
    let _ = MODEL.set(path.to_path_buf());
}

/// The key for the whole run's choices: the served model on this process's
/// first device ([`crate::engine::backend::tuning_profile::device_key`]), or
/// on the cores when there is none. Computed on first use, which comes after
/// the device is chosen.
pub fn model_key() -> Option<&'static str> {
    static KEY: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    KEY.get_or_init(|| {
        let model = MODEL.get()?;
        let device = crate::engine::backend::tuning_profile::device_key()
            .map_or_else(host_key, str::to_string);
        key(&device, model)
    })
    .as_deref()
}

/// A kept value `what` for `key`, as it was written, if there is a young
/// enough one.
pub fn value(what: &str, key: &str) -> Option<serde_json::Value> {
    if !enabled() {
        return None;
    }
    let map = read(&file()?);
    let entry = map.get(&choice_key(what, key))?;
    let when = entry["when"].as_u64()?;
    if now().saturating_sub(when) > MAX_AGE.as_secs() {
        return None;
    }
    Some(entry["value"].clone())
}

/// Keeps `value` as `what` for `key`.
pub fn keep_value(what: &str, key: &str, value: serde_json::Value) {
    if !enabled() {
        return;
    }
    let Some(path) = file() else {
        return;
    };
    let at = now();
    let entry = serde_json::json!({"value": value, "when": at});
    if let Err(err) = put_value_in(&path, &choice_key(what, key), entry, at) {
        log::warn!(
            "orangu-server: could not keep the {what} in {}: {err}",
            path.display()
        );
    }
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn read(path: &Path) -> serde_json::Map<String, serde_json::Value> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .and_then(|v| match v {
            serde_json::Value::Object(m) => Some(m),
            _ => None,
        })
        .unwrap_or_default()
}

fn get_in(path: &Path, key: &str, at: u64) -> Option<DecodeProbe> {
    let map = read(path);
    let entry = map.get(key)?;
    let when = entry["when"].as_u64()?;
    if at.saturating_sub(when) > MAX_AGE.as_secs() {
        return None;
    }
    Some(DecodeProbe {
        device_s: entry["device_s"].as_f64()?,
        cpu_s: entry["cpu_s"].as_f64()?,
    })
}

fn put_in(path: &Path, key: &str, probe: DecodeProbe, at: u64) -> std::io::Result<()> {
    put_value_in(
        path,
        key,
        serde_json::json!({"device_s": probe.device_s, "cpu_s": probe.cpu_s, "when": at}),
        at,
    )
}

/// Writes `value` under `key`, keeping every other young entry.
///
/// Several servers on one host (a coordinator's residents) share the file,
/// so the read, the change and the write happen under an exclusive lock on
/// a file beside it: without it two servers writing at once each read the
/// map before the other's entry was in it, and the last rename dropped the
/// first one's. The new map is written to a temporary file of this writer's
/// own and renamed over the old, so a reader never sees half a file.
fn put_value_in(path: &Path, key: &str, value: serde_json::Value, at: u64) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(path.with_extension("json.lock"))?;
    lock.lock()?;
    let mut map = read(path);
    // Entries past their age only take room.
    map.retain(|_, v| {
        v["when"]
            .as_u64()
            .is_some_and(|w| at.saturating_sub(w) <= MAX_AGE.as_secs())
    });
    map.insert(key.to_string(), value);
    static WRITES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let tmp = path.with_extension(format!(
        "json.{}.{}.tmp",
        std::process::id(),
        WRITES.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let written = std::fs::write(&tmp, serde_json::Value::Object(map).to_string())
        .and_then(|()| std::fs::rename(&tmp, path));
    if written.is_err() {
        std::fs::remove_file(&tmp).ok();
    }
    // The lock is released when `lock` closes, after the rename.
    written
}

/// The key a kept on/off choice named `what` is stored under, beside the
/// decode probe of the same `key`.
fn choice_key(what: &str, key: &str) -> String {
    format!("{what}|{key}")
}

fn get_choice_in(path: &Path, key: &str, at: u64) -> Option<bool> {
    let map = read(path);
    let entry = map.get(key)?;
    let when = entry["when"].as_u64()?;
    if at.saturating_sub(when) > MAX_AGE.as_secs() {
        return None;
    }
    entry["on"].as_bool()
}

/// A kept on/off choice `what` for `key` — whether a kernel family won its
/// start-up timing, say — if there is a young enough one.
pub fn choice(what: &str, key: &str) -> Option<bool> {
    if !enabled() {
        return None;
    }
    get_choice_in(&file()?, &choice_key(what, key), now())
}

/// Keeps the choice `what` for `key`, with the two timings that made it.
pub fn keep_choice(what: &str, key: &str, on: bool, on_s: f64, off_s: f64) {
    if !enabled() {
        return;
    }
    let Some(path) = file() else {
        return;
    };
    let at = now();
    let value = serde_json::json!({"on": on, "on_s": on_s, "off_s": off_s, "when": at});
    if let Err(err) = put_value_in(&path, &choice_key(what, key), value, at) {
        log::warn!(
            "orangu-server: could not keep the {what} choice in {}: {err}",
            path.display()
        );
    }
}

/// A kept count `what` for `key` — the row count a kernel won its
/// start-up timing at, say — if there is a young enough one.
pub fn count(what: &str, key: &str) -> Option<usize> {
    if !enabled() {
        return None;
    }
    get_count_in(&file()?, &choice_key(what, key), now())
}

fn get_count_in(path: &Path, key: &str, at: u64) -> Option<usize> {
    let map = read(path);
    let entry = map.get(key)?;
    let when = entry["when"].as_u64()?;
    if at.saturating_sub(when) > MAX_AGE.as_secs() {
        return None;
    }
    usize::try_from(entry["count"].as_u64()?).ok()
}

/// Keeps the count `what` for `key`, with each candidate's timing.
pub fn keep_count(what: &str, key: &str, count: usize, timings: &[(usize, f64)]) {
    if !enabled() {
        return;
    }
    let Some(path) = file() else {
        return;
    };
    let at = now();
    let seconds: serde_json::Map<String, serde_json::Value> = timings
        .iter()
        .map(|&(c, s)| (c.to_string(), serde_json::json!(s)))
        .collect();
    let value = serde_json::json!({"count": count, "seconds": seconds, "when": at});
    if let Err(err) = put_value_in(&path, &choice_key(what, key), value, at) {
        log::warn!(
            "orangu-server: could not keep the {what} count in {}: {err}",
            path.display()
        );
    }
}

/// The kept decode probe for `key`, if there is a young enough one.
pub fn decode_probe(key: &str) -> Option<DecodeProbe> {
    if !enabled() {
        return None;
    }
    get_in(&file()?, key, now())
}

/// Keeps `probe` for `key`. A failure to write costs the next start a
/// measurement and nothing else, so it is logged and dropped.
pub fn keep_decode_probe(key: &str, probe: DecodeProbe) {
    if !enabled() {
        return;
    }
    let Some(path) = file() else {
        return;
    };
    if let Err(err) = put_in(&path, key, probe, now()) {
        log::warn!(
            "orangu-server: could not keep the decode probe in {}: {err}",
            path.display()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_kept_probe_is_read_back_until_it_ages_out() {
        let dir = std::env::temp_dir().join(format!("orangu-probes-{}", std::process::id()));
        let path = dir.join("probes.json");
        let p = DecodeProbe {
            device_s: 0.0153,
            cpu_s: 0.0641,
        };
        put_in(&path, "k", p, 1000).unwrap();
        assert_eq!(get_in(&path, "k", 1000 + 60), Some(p));
        assert_eq!(get_in(&path, "other", 1000), None);
        assert_eq!(get_in(&path, "k", 1000 + MAX_AGE.as_secs() + 1), None);
        // A later write drops the aged entry and keeps the new one.
        put_in(&path, "k2", p, 1000 + MAX_AGE.as_secs() + 1).unwrap();
        assert_eq!(read(&path).len(), 1);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_kept_choice_sits_beside_the_probe_of_its_key() {
        let dir = std::env::temp_dir().join(format!("orangu-choices-{}", std::process::id()));
        let path = dir.join("probes.json");
        let p = DecodeProbe {
            device_s: 0.02,
            cpu_s: 0.05,
        };
        put_in(&path, "k", p, 1000).unwrap();
        let value = serde_json::json!({"on": false, "on_s": 0.03, "off_s": 0.02, "when": 1000});
        put_value_in(&path, &choice_key("i8", "k"), value, 1000).unwrap();
        assert_eq!(get_in(&path, "k", 1000), Some(p));
        assert_eq!(
            get_choice_in(&path, &choice_key("i8", "k"), 1000),
            Some(false)
        );
        assert_eq!(get_choice_in(&path, &choice_key("i8", "other"), 1000), None);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_kept_count_reads_back_and_a_choice_is_not_one() {
        let dir = std::env::temp_dir().join(format!("orangu-counts-{}", std::process::id()));
        let path = dir.join("probes.json");
        let value =
            serde_json::json!({"count": 4, "seconds": {"0": 0.03, "4": 0.02}, "when": 1000});
        put_value_in(&path, &choice_key("q4k-short", "k"), value, 1000).unwrap();
        let value = serde_json::json!({"on": true, "on_s": 0.02, "off_s": 0.03, "when": 1000});
        put_value_in(&path, &choice_key("mmq", "k"), value, 1000).unwrap();
        assert_eq!(
            get_count_in(&path, &choice_key("q4k-short", "k"), 1000),
            Some(4)
        );
        assert_eq!(get_count_in(&path, &choice_key("mmq", "k"), 1000), None);
        assert_eq!(
            get_count_in(
                &path,
                &choice_key("q4k-short", "k"),
                1000 + MAX_AGE.as_secs() + 1
            ),
            None
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Writers running at once — threads here, servers on one host in use;
    /// the lock is on the file, so both are kept apart the same way — each
    /// keep their entry.
    #[test]
    fn writers_at_once_lose_no_entry() {
        let dir = std::env::temp_dir().join(format!("orangu-writers-{}", std::process::id()));
        let path = dir.join("probes.json");
        const WRITERS: usize = 8;
        const EACH: usize = 25;
        std::thread::scope(|s| {
            for w in 0..WRITERS {
                let path = &path;
                s.spawn(move || {
                    for i in 0..EACH {
                        let p = DecodeProbe {
                            device_s: w as f64,
                            cpu_s: i as f64,
                        };
                        put_in(path, &format!("{w}-{i}"), p, 1000).unwrap();
                    }
                });
            }
        });
        assert_eq!(read(&path).len(), WRITERS * EACH);
        let left: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .filter(|n| n.ends_with(".tmp"))
            .collect();
        assert!(left.is_empty(), "temporary files left behind: {left:?}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_key_names_the_file_and_the_build() {
        let me = std::env::current_exe().unwrap();
        let k = key("1002-7340-radv", &me).unwrap();
        assert!(k.starts_with("1002-7340-radv|"));
        assert!(k.contains(orangu::build_info::id()));
        assert!(key("d", Path::new("/no/such/file")).is_none());
    }

    /// A copy of a file at another path is the same model; the same path
    /// holding other bytes is not.
    #[test]
    fn the_key_names_the_content_not_the_path() {
        let dir = std::env::temp_dir().join(format!("orangu-fp-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (a, b) = (dir.join("a.gguf"), dir.join("b.gguf"));
        let bytes: Vec<u8> = (0..200_000u32).map(|i| (i * 31 % 251) as u8).collect();
        std::fs::write(&a, &bytes).unwrap();
        std::fs::write(&b, &bytes).unwrap();
        assert_eq!(key("d", &a), key("d", &b));
        // Same length, a different last byte: a different model.
        let mut other = bytes.clone();
        *other.last_mut().unwrap() ^= 1;
        let c = dir.join("c.gguf");
        std::fs::write(&c, &other).unwrap();
        assert_ne!(key("d", &a), key("d", &c));
        // A short file is read whole.
        let short = dir.join("short.gguf");
        std::fs::write(&short, b"GGUF").unwrap();
        assert!(key("d", &short).is_some());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_host_key_names_the_cores() {
        let k = host_key();
        assert!(k.starts_with(&format!("cpu_{}", std::env::consts::ARCH)));
        assert!(
            k.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        );
    }
}
