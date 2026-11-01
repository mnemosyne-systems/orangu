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

//! `--write-profile`: a sweep's winner kept in the server's tuning profile.
//!
//! The server reads `~/.orangu/server/tuning/<device key>.json` at start-up
//! (`engine::backend::tuning_profile`), values under `"values"`. A sweep of
//! one variable between two values, judged by its verdict, writes the
//! candidate there when it beat the control at some point and lost at none;
//! anything else leaves the file as it was.

use std::path::{Path, PathBuf};

/// The device key a server reports at `/props` `gpu.tuning.key`.
pub fn device_key(props: &serde_json::Value) -> Option<String> {
    props
        .get("gpu")?
        .get("tuning")?
        .get("key")?
        .as_str()
        .map(str::to_string)
}

/// The profile file for `key`, under `home`.
pub fn path_under(home: &Path, key: &str) -> PathBuf {
    home.join(".orangu/server/tuning")
        .join(format!("{key}.json"))
}

/// `contents` (a profile, or `None` for a new one) with `var` set to
/// `value`, or removed when `value` is empty — an empty value is the
/// variable left unset, which is the compiled-in default. Every other value
/// and field is kept.
pub fn with_value(contents: Option<&str>, var: &str, value: &str) -> anyhow::Result<String> {
    let mut doc: serde_json::Value = match contents {
        Some(text) => serde_json::from_str(text)
            .map_err(|e| anyhow::anyhow!("the profile is not valid JSON ({e})"))?,
        None => serde_json::json!({}),
    };
    let object = doc
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("the profile is not a JSON object"))?;
    let values = object
        .entry("values")
        .or_insert_with(|| serde_json::json!({}))
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("the profile's \"values\" is not an object"))?;
    if value.is_empty() {
        values.remove(var);
    } else {
        values.insert(var.to_string(), serde_json::json!(value));
    }
    Ok(serde_json::to_string_pretty(&doc)? + "\n")
}

/// Writes `var = value` into the profile at `path`, through a temporary file
/// beside it.
pub fn write(path: &Path, var: &str, value: &str) -> anyhow::Result<()> {
    let existing = match std::fs::read_to_string(path) {
        Ok(text) => Some(text),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(anyhow::anyhow!("reading {}: {e}", path.display())),
    };
    let updated = with_value(existing.as_deref(), var, value)
        .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension(format!("json.{}.tmp", std::process::id()));
    std::fs::write(&tmp, updated)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_value_is_set_beside_the_others_and_unset_removes_it() {
        let text = r#"{"values":{"ORANGU_A":"1"},"note":"kept"}"#;
        let out = with_value(Some(text), "ORANGU_B", "32").unwrap();
        let doc: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(doc["values"]["ORANGU_A"], "1");
        assert_eq!(doc["values"]["ORANGU_B"], "32");
        assert_eq!(doc["note"], "kept");
        let out = with_value(Some(&out), "ORANGU_A", "").unwrap();
        let doc: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert!(doc["values"].get("ORANGU_A").is_none());
        assert_eq!(doc["values"]["ORANGU_B"], "32");
    }

    #[test]
    fn a_new_profile_holds_the_one_value() {
        let out = with_value(None, "ORANGU_REDUCE_N_ROWS", "2").unwrap();
        let doc: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(
            doc,
            serde_json::json!({"values": {"ORANGU_REDUCE_N_ROWS": "2"}})
        );
    }

    #[test]
    fn a_profile_that_is_not_one_is_refused() {
        assert!(with_value(Some("not json"), "A", "1").is_err());
        assert!(with_value(Some("[1]"), "A", "1").is_err());
        assert!(with_value(Some(r#"{"values":3}"#), "A", "1").is_err());
    }

    #[test]
    fn the_key_is_read_from_props() {
        let props = serde_json::json!({"gpu": {"tuning": {"key": "1002-7340-radv"}}});
        assert_eq!(device_key(&props).as_deref(), Some("1002-7340-radv"));
        assert_eq!(device_key(&serde_json::json!({"gpu": null})), None);
        assert_eq!(
            path_under(Path::new("/h"), "k"),
            Path::new("/h/.orangu/server/tuning/k.json")
        );
    }
}
