// Copyright (C) 2026 The orangu community
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

//! Persistent model inventory metadata.
//!
//! Model weights live in the configured models directory, which may be
//! anywhere on disk. The small pieces of information that cannot be
//! recovered by scanning those weights live in `~/.orangu/models`: when a
//! model was downloaded and when it was last successfully loaded.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    fs::OpenOptions,
    path::{Path, PathBuf},
    thread,
    time::{SystemTime, UNIX_EPOCH},
};

const VERSION: u32 = 1;

#[derive(Debug, Default, Deserialize, Serialize)]
struct Registry {
    #[serde(default = "current_version")]
    version: u32,
    #[serde(default)]
    models: Vec<ModelRecord>,
}

fn current_version() -> u32 {
    VERSION
}

/// One locally available model. Timestamps are Unix seconds so the file is
/// portable and remains straightforward for other tools to consume.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelRecord {
    pub model: String,
    pub path: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub downloaded_at: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_used: Option<u64>,
}

/// Records a completed download. Re-running `download` for a model already
/// on disk does not pretend it was newly downloaded.
pub fn record_download(model: &str, path: &Path) -> Result<()> {
    update(model, path, Update::Downloaded)
}

/// Records a model only after its weights have been successfully opened.
pub fn record_used(model: &str, path: &Path) -> Result<()> {
    update(model, path, Update::Used)
}

/// Drops records for model files that have been deleted.
pub fn forget(paths: &[PathBuf]) -> Result<()> {
    let Some(file) = registry_path() else {
        return Ok(());
    };
    if !file.is_file() {
        return Ok(());
    }
    let _lock = RegistryLock::acquire(&file)?;
    let mut registry = read_registry(&file)?;
    let paths: Vec<PathBuf> = paths.iter().map(|path| normalized(path)).collect();
    let before = registry.models.len();
    registry
        .models
        .retain(|record| !paths.contains(&normalized(&record.path)));
    if registry.models.len() == before {
        return Ok(());
    }
    write_registry(&file, &registry)
}

/// Returns one use time per group, reading `~/.orangu/models` only once.
/// Each item in `groups` is the complete set of shard paths for one row.
pub fn last_used_for<'a>(groups: impl IntoIterator<Item = &'a [PathBuf]>) -> Vec<Option<u64>> {
    let registry = registry_path()
        .and_then(|path| read_registry(&path).ok())
        .unwrap_or_default();
    groups
        .into_iter()
        .map(|paths| {
            paths
                .iter()
                .filter_map(|path| {
                    let path = normalized(path);
                    registry
                        .models
                        .iter()
                        .find(|record| normalized(&record.path) == path)
                        .and_then(|record| record.last_used)
                })
                .max()
        })
        .collect()
}

enum Update {
    Downloaded,
    Used,
}

fn update(model: &str, path: &Path, update: Update) -> Result<()> {
    let Some(file) = registry_path() else {
        return Ok(());
    };
    update_at(&file, model, path, update)
}

fn update_at(file: &Path, model: &str, path: &Path, update: Update) -> Result<()> {
    let _lock = RegistryLock::acquire(file)?;
    let mut registry = read_registry(file)?;
    let path = normalized(path);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let index = registry
        .models
        .iter()
        .position(|record| normalized(&record.path) == path);
    let record = match index {
        Some(index) => &mut registry.models[index],
        None => {
            registry.models.push(ModelRecord {
                model: model.to_string(),
                path: path.clone(),
                downloaded_at: None,
                last_used: None,
            });
            registry
                .models
                .last_mut()
                .expect("a record was just inserted")
        }
    };
    record.model = model.to_string();
    record.path = path;
    match update {
        Update::Downloaded => {
            record.downloaded_at.get_or_insert(now);
        }
        Update::Used => record.last_used = Some(now),
    }

    write_registry(file, &registry)
}

fn registry_path() -> Option<PathBuf> {
    home::home_dir().map(|home| home.join(".orangu").join("models"))
}

fn read_registry(path: &Path) -> Result<Registry> {
    match std::fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .with_context(|| format!("failed to parse {}", path.display())),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(Registry {
            version: VERSION,
            models: Vec::new(),
        }),
        Err(err) => Err(err).with_context(|| format!("failed to read {}", path.display())),
    }
}

fn write_registry(path: &Path, registry: &Registry) -> Result<()> {
    let parent = path.parent().expect("the registry path has a parent");
    std::fs::create_dir_all(parent)
        .with_context(|| format!("failed to create {}", parent.display()))?;
    let mut json = serde_json::to_string_pretty(registry).context("serializing model registry")?;
    json.push('\n');
    let temporary = path.with_extension("tmp");
    std::fs::write(&temporary, json)
        .with_context(|| format!("failed to write {}", temporary.display()))?;
    match std::fs::rename(&temporary, path) {
        Ok(()) => Ok(()),
        // Unix replaces the destination atomically. Windows does not; the
        // writer lock still prevents two updates from interleaving there.
        Err(_) if path.is_file() => {
            std::fs::remove_file(path)
                .with_context(|| format!("failed to replace {}", path.display()))?;
            std::fs::rename(&temporary, path)
                .with_context(|| format!("failed to replace {}", path.display()))
        }
        Err(err) => Err(err).with_context(|| format!("failed to replace {}", path.display())),
    }
}

fn normalized(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// A tiny cross-process lock around the registry's read-modify-write cycle.
/// Several coordinator-managed servers can start at the same time; without
/// this, the last writer would silently discard the other model's update.
struct RegistryLock {
    path: PathBuf,
}

impl RegistryLock {
    fn acquire(registry: &Path) -> Result<Self> {
        let parent = registry.parent().expect("the registry path has a parent");
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
        let path = registry.with_extension("lock");
        for _ in 0..200 {
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(_) => return Ok(Self { path }),
                Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
                    let stale = std::fs::metadata(&path)
                        .and_then(|metadata| metadata.modified())
                        .ok()
                        .and_then(|modified| modified.elapsed().ok())
                        .is_some_and(|age| age.as_secs() >= 30);
                    if stale {
                        let _ = std::fs::remove_file(&path);
                    } else {
                        thread::sleep(std::time::Duration::from_millis(10));
                    }
                }
                Err(err) => {
                    return Err(err)
                        .with_context(|| format!("failed to lock {}", registry.display()));
                }
            }
        }
        anyhow::bail!("timed out locking {}", registry.display())
    }
}

impl Drop for RegistryLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn download_and_use_are_kept_in_the_models_file() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("model.gguf");
        std::fs::write(&model, b"GGUF").unwrap();
        let registry = dir.path().join("models");

        update_at(&registry, "owner/model:Q4_K_M", &model, Update::Downloaded).unwrap();
        update_at(&registry, "owner/model:Q4_K_M", &model, Update::Used).unwrap();

        let saved = read_registry(&registry).unwrap();
        assert_eq!(saved.version, VERSION);
        assert_eq!(saved.models.len(), 1);
        assert_eq!(saved.models[0].model, "owner/model:Q4_K_M");
        assert_eq!(saved.models[0].path, model.canonicalize().unwrap());
        assert!(saved.models[0].downloaded_at.is_some());
        assert!(saved.models[0].last_used.is_some());
    }

    #[test]
    fn using_a_manually_installed_model_creates_a_record_without_a_download_time() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("manual.gguf");
        std::fs::write(&model, b"GGUF").unwrap();
        let registry = dir.path().join("models");

        update_at(&registry, "manual", &model, Update::Used).unwrap();

        let saved = read_registry(&registry).unwrap();
        assert_eq!(saved.models.len(), 1);
        assert_eq!(saved.models[0].downloaded_at, None);
        assert!(saved.models[0].last_used.is_some());
    }

    #[test]
    fn downloading_an_existing_model_preserves_its_original_time() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("model.gguf");
        std::fs::write(&model, b"GGUF").unwrap();
        let registry = dir.path().join("models");

        update_at(&registry, "owner/model", &model, Update::Downloaded).unwrap();
        let first = read_registry(&registry).unwrap().models[0].downloaded_at;
        update_at(&registry, "owner/model", &model, Update::Downloaded).unwrap();

        assert_eq!(
            read_registry(&registry).unwrap().models[0].downloaded_at,
            first
        );
    }
}
