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

//! The on-disk `wgpu::PipelineCache`, which turns a multi-second shader
//! compile at startup into a file read.
//!
//! One directory per adapter, keyed by `wgpu::util::pipeline_cache_key`, so a
//! cache built for one GPU is never handed to another. Writes go through a
//! temporary file and a rename, because a crash mid-write would otherwise
//! leave a truncated cache for the next startup to try to load.
//!
//! **It also makes `RADV_DEBUG=shaderstats` report nothing**, which costs an
//! afternoon if you do not know it: a cached pipeline is never compiled, so
//! ACO never runs and never prints. Clear
//! `~/.orangu/server/wgpu_pipeline_cache_*` before any ISA measurement.

use std::path::PathBuf;

/// `~/.orangu/server/<key>/cache.bin` — a persistent, on-disk pipeline
/// cache. `key` is `wgpu::util::pipeline_cache_key`'s output
/// (vendor/device-derived, so a cache built
/// for one GPU is never handed to a different one), one directory per
/// adapter rather than a flat file, matching `web::sessions::sessions_dir`'s
/// own "one identifier, one directory" shape rather than introducing a
/// second, differently-shaped convention. `None` if the home directory
/// can't be resolved — this cache is a startup-time optimization only,
/// never required for correctness, so a missing `$HOME` just means "skip
/// the cache," not "fail to start."
pub(super) fn pipeline_cache_file_path(key: &str) -> Option<PathBuf> {
    Some(
        home::home_dir()?
            .join(".orangu/server")
            .join(key)
            .join("cache.bin"),
    )
}

/// Writes `data` to `path` atomically (temp file, then rename over the
/// real path) — `wgpu::PipelineCache`'s own doc comment recommends exactly
/// this so a crash or concurrent write mid-save can never leave a
/// truncated, half-written cache file for the next startup to try to load.
pub(super) fn save_pipeline_cache(path: &std::path::Path, data: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("bin.tmp");
    std::fs::write(&tmp, data)?;
    std::fs::rename(&tmp, path)
}
