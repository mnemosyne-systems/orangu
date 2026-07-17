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

//! Recursively discovers `.gguf` files under the configured `models`
//! directory and summarizes each one for the `list` subcommand. Uses the
//! same lightweight [`crate::gguf::GgufFile`] reader `show` uses — it never
//! touches tensor data, so scanning a directory of multi-gigabyte model
//! files stays fast.

use crate::format::format_bytes;
use crate::gguf::{GgufFile, ggml_type_name};
use anyhow::{Context, Result};
use std::{
    collections::{BTreeMap, HashMap},
    path::{Path, PathBuf},
};

pub struct ModelSummary {
    pub path: PathBuf,
    pub size_bytes: u64,
    /// Element counts per `ggml_type`, empty when `error` is set.
    pub type_totals: HashMap<u32, u128>,
    /// Set instead of `type_totals` when the file's header couldn't be
    /// parsed (truncated download, not actually a GGUF file, ...) — reported
    /// per-file rather than aborting the whole scan.
    pub error: Option<String>,
}

/// Recursively scans `dir` for `.gguf` files (case-insensitive extension),
/// returning one summary per unique model, sorted by path. Two kinds of
/// non-models are deliberately excluded so only real, distinct models are
/// counted and listed:
///
/// - **Duplicate underlying files.** A model cache (Hugging Face's hub
///   cache in particular) can reference the exact same downloaded bytes
///   from more than one directory — most commonly two snapshot revisions of
///   one repo whose ref moved without the file's content changing, where
///   the cache reuses (symlinks to) the already-downloaded blob rather than
///   fetching it again. Resolving each candidate to its real, symlink-free
///   path and keeping only the first occurrence collapses these back down
///   to one entry per physical file.
/// - **Multimodal projector ("mmproj") sidecars.** These accompany a base
///   model rather than standing in for one; see
///   [`GgufFile::is_clip_projector`].
pub fn scan_models_dir(dir: &Path) -> Result<Vec<ModelSummary>> {
    if !dir.is_dir() {
        anyhow::bail!("models directory {} does not exist", dir.display());
    }

    // Model caches (Hugging Face's hub cache in particular) store the actual
    // file under `blobs/` and name it via a symlink under `snapshots/<rev>/`;
    // without `follow_links`, `entry.file_type().is_file()` reports the
    // symlink itself (never `true`) and every such model would be silently
    // skipped instead of listed.
    let mut paths: Vec<PathBuf> = walkdir::WalkDir::new(dir)
        .follow_links(true)
        .into_iter()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().is_file())
        .map(walkdir::DirEntry::into_path)
        .filter(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        })
        .collect();
    paths.sort();

    let mut seen_targets = std::collections::HashSet::new();
    let mut summaries = Vec::new();
    for path in paths {
        let real_path = std::fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
        if !seen_targets.insert(real_path) {
            continue;
        }

        let size_bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        match GgufFile::open(&path) {
            Ok(gguf) => {
                if gguf.is_clip_projector() {
                    continue;
                }
                summaries.push(ModelSummary {
                    path,
                    size_bytes,
                    type_totals: gguf.type_element_totals(),
                    error: None,
                });
            }
            Err(err) => summaries.push(ModelSummary {
                path,
                size_bytes,
                type_totals: HashMap::new(),
                error: Some(err.to_string()),
            }),
        }
    }

    Ok(summaries)
}

/// Resolves a `show` target that names a file directly: used as-is if it
/// names an existing file (relative to the current directory or absolute),
/// otherwise resolved against the configured models directory — so
/// `orangu-server show my-model.gguf` works without repeating the full path.
fn resolve_model_path(models_dir: &Path, requested: &str) -> Result<PathBuf> {
    let direct = PathBuf::from(requested);
    if direct.is_file() {
        return Ok(direct);
    }
    let under_models = models_dir.join(requested);
    if under_models.is_file() {
        return Ok(under_models);
    }
    anyhow::bail!(
        "'{requested}' was not found as a file or under the models directory {}",
        models_dir.display()
    )
}

/// Resolves whatever `show` was given: a direct/bare file path (checked
/// first — no directory scan needed, so the common case of passing a path
/// stays instant), an `NR` from `list`'s first column, or a `MODEL` name
/// from its second. `list`'s numbering and grouping are recomputed here
/// (`orangu-server` keeps no state between runs), so `NR` is only meaningful
/// as of the current directory contents — matching `list`'s exact sort
/// order is what keeps it stable between one `list` call and the next.
pub fn resolve_show_target(models_dir: &Path, requested: &str) -> Result<PathBuf> {
    if let Ok(path) = resolve_model_path(models_dir, requested) {
        return Ok(path);
    }

    let models = scan_models_dir(models_dir)?;
    let groups = group_models(&models);

    if let Ok(nr) = requested.parse::<usize>() {
        return nr
            .checked_sub(1)
            .and_then(|index| groups.get(index))
            .map(|group| group.representative_path.clone())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "no model with NR {nr} ({} model(s) found under {}; run 'orangu-server list' to see them)",
                    groups.len(),
                    models_dir.display()
                )
            });
    }

    groups
        .iter()
        .find(|group| group.label == requested)
        .map(|group| group.representative_path.clone())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "'{requested}' was not found as a file, an NR, or a MODEL name; run 'orangu-server list' to see valid values"
            )
        })
}

/// Resolves a model the caller named — a direct/bare file path, an `NR`/
/// `MODEL` label already present under `models_dir` (exactly like
/// [`resolve_show_target`]), or a `<user>/<model>[:quant]` Hugging Face
/// repo — to a local `.gguf` path, **fetching it from the Hub first** when
/// it names a repo not already cached under `models_dir`. This is what lets
/// `orangu-server <spec>` start straight from a bare model reference (the
/// same one `orangu-server download <spec>` would fetch explicitly) with no
/// separate download step.
pub fn resolve_or_fetch_model(models_dir: &Path, requested: &str) -> Result<PathBuf> {
    if let Ok(path) = resolve_show_target(models_dir, requested) {
        return Ok(path);
    }
    crate::model_download::download_model(models_dir, requested)
        .with_context(|| format!("'{requested}' was not found locally and could not be fetched"))
}

/// Resolves whatever `delete` was given to a full [`ModelGroup`] — every
/// shard, not just one file — so a multi-shard model is always deleted
/// atomically regardless of which shard's path happened to be named.
/// Unlike [`resolve_show_target`], this always scans and groups first (no
/// scan-free fast path for a direct file argument): even a plain path needs
/// the full grouping to know whether it names one shard of a larger group.
///
/// Resolution order matches `resolve_show_target`: a direct/relative/
/// absolute path or a bare name under `models_dir` first (returning that
/// file's whole group when it belongs to one, or a synthetic single-file
/// group when it doesn't — e.g. an mmproj sidecar, which `group_models`
/// deliberately excludes from every real group); then an `NR` from `list`'s
/// first column; then a `MODEL` name from its second.
pub fn resolve_delete_target(models_dir: &Path, requested: &str) -> Result<ModelGroup> {
    let models = scan_models_dir(models_dir)?;
    let groups = group_models(&models);

    if let Ok(path) = resolve_model_path(models_dir, requested) {
        if let Some(group) = groups.into_iter().find(|g| g.paths.contains(&path)) {
            return Ok(group);
        }
        let size_bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        return Ok(ModelGroup {
            label: path.display().to_string(),
            size_bytes,
            quantization: None,
            errors: Vec::new(),
            representative_path: path.clone(),
            paths: vec![path],
        });
    }

    if let Ok(nr) = requested.parse::<usize>() {
        let count = groups.len();
        return nr
            .checked_sub(1)
            .and_then(|index| groups.into_iter().nth(index))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "no model with NR {nr} ({count} model(s) found under {}; run 'orangu-server list' to see them)",
                    models_dir.display()
                )
            });
    }

    groups
        .into_iter()
        .find(|group| group.label == requested)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "'{requested}' was not found as a file, an NR, or a MODEL name; run 'orangu-server list' to see valid values"
            )
        })
}

/// Deletes every path in `group` from disk. When a path is a Hugging Face
/// hub-cache symlink (`models--<user>--<model>/snapshots/<rev>/<file>`,
/// pointing into that same repo's `blobs/`), its target blob is deleted too
/// — but only when no other snapshot left in that repo still points at it:
/// a repo's ref can move without a file's content changing, in which case
/// the cache reuses (symlinks to), rather than re-fetches, the
/// already-downloaded blob (`scan_models_dir`'s own dedup logic collapses
/// that pair down to one listed file, so the *other* snapshot's symlink —
/// not part of `group`, since it was never listed — must not be left
/// dangling). Empty snapshot/model directories left behind are removed
/// too, walking up from each deleted path but never past `models_dir`
/// itself, which is left alone regardless of what remains inside it.
pub fn delete_model(models_dir: &Path, group: &ModelGroup) -> Result<()> {
    for path in &group.paths {
        let blob_target = std::fs::symlink_metadata(path)
            .ok()
            .filter(std::fs::Metadata::is_symlink)
            .and_then(|_| std::fs::canonicalize(path).ok());

        std::fs::remove_file(path)
            .with_context(|| format!("failed to delete {}", path.display()))?;

        if let Some(blob) = blob_target
            && let Some(repo_root) = hf_repo_root_from_path(path)
            && blob.starts_with(repo_root.join("blobs"))
            && !blob_still_referenced(&repo_root, &blob)
            && std::fs::remove_file(&blob).is_ok()
        {
            // `blob` sits under a sibling `blobs/` directory, not under
            // `path`'s own `snapshots/...` chain, so it needs its own
            // upward sweep — otherwise a now-empty `blobs/` (and, once
            // both it and `snapshots/` are gone, the whole repo directory)
            // would survive even though nothing is left inside it.
            remove_empty_ancestors(&blob, models_dir);
        }

        remove_empty_ancestors(path, models_dir);
    }
    Ok(())
}

/// The Hugging Face hub-cache repo root a path lives under
/// (`models--<user>--<model>`, the directory [`hf_repo_id_from_path`]
/// decodes the id from), or `None` outside that layout. Checks every
/// ancestor, not just the immediate parent, for the same reason
/// `hf_repo_id_from_path` does — a file sits under `snapshots/<rev>/`,
/// sometimes with a further per-quant subfolder.
fn hf_repo_root_from_path(path: &Path) -> Option<PathBuf> {
    for ancestor in path.parent()?.ancestors() {
        let name = ancestor.file_name()?.to_str()?;
        if name.starts_with("models--") {
            return Some(ancestor.to_path_buf());
        }
    }
    None
}

/// Whether any symlink still left under `repo_root`'s own `snapshots/`
/// resolves to `blob` — scoped to just this one repo (blobs are already
/// repo-scoped by construction, nested under `models--<user>--<model>/
/// blobs/`, so a blob from one repo can never collide with another's) and
/// checked *after* the symlink being deleted is already gone, so it
/// answers "does anything else still need this blob".
fn blob_still_referenced(repo_root: &Path, blob: &Path) -> bool {
    let snapshots = repo_root.join("snapshots");
    if !snapshots.is_dir() {
        return false;
    }
    walkdir::WalkDir::new(&snapshots)
        .follow_links(false)
        .into_iter()
        .filter_map(|entry| entry.ok())
        .any(|entry| {
            std::fs::canonicalize(entry.path())
                .map(|resolved| resolved == blob)
                .unwrap_or(false)
        })
}

/// Removes `path`'s parent directory, and each ancestor above it in turn,
/// as long as it's empty — stopping the moment one isn't, or at `stop_at`
/// (never removed itself, whatever's left inside it), so deleting a
/// model's last shard also cleans up the now-empty `snapshots/<rev>/` (and,
/// if that was the repo's only snapshot, `models--<user>--<model>/` itself)
/// rather than leaving empty directories behind.
fn remove_empty_ancestors(path: &Path, stop_at: &Path) {
    let mut dir = path.parent();
    while let Some(d) = dir {
        if d == stop_at || !d.starts_with(stop_at) {
            break;
        }
        match std::fs::read_dir(d) {
            Ok(mut entries) => {
                if entries.next().is_some() || std::fs::remove_dir(d).is_err() {
                    break;
                }
                dir = d.parent();
            }
            Err(_) => break,
        }
    }
}

/// One row of the `list` output: a model, collapsed from every shard file
/// that makes it up.
#[derive(Debug)]
pub struct ModelGroup {
    pub label: String,
    pub size_bytes: u64,
    pub quantization: Option<String>,
    /// Parse errors from any shard in this group; a non-empty list is shown
    /// instead of `quantization`/`size_bytes`.
    pub errors: Vec<String>,
    /// The first shard's path — the one `show` opens for this group, since
    /// GGUF metadata for a multi-shard model lives entirely in shard 1.
    pub representative_path: PathBuf,
    /// Every shard file that makes up this model, in the same sorted order
    /// `representative_path` (the first of them) was chosen from — what
    /// `delete_model` actually removes, so a multi-shard model is deleted
    /// atomically rather than leaving orphaned shards behind.
    pub paths: Vec<PathBuf>,
}

/// Collapses a multi-part model's shard files (`name-00001-of-00004.gguf`,
/// `name-00002-of-00004.gguf`, ...) into a single [`ModelGroup`]: one entry
/// per model rather than one per shard, with `size_bytes` summed across
/// shards and `quantization` picked from the combined element counts of
/// every shard's tensors (a single shard's own tensors are only part of the
/// whole model — see [`crate::gguf::GgufFile::type_element_totals`]).
/// Grouping is keyed by (parent directory, shard-suffix-stripped file stem),
/// so two files that merely share a name in different directories (e.g. two
/// Hugging Face cache snapshots of the same release) are kept separate.
///
/// `label` is the exact string to hand to llama.cpp's `-hf`/`--hf-repo`
/// (`<user>/<model>[:quant]`) when the file lives under a Hugging Face hub
/// cache directory (`models--<user>--<model>/...`, the layout `-hf` itself
/// downloads into) — otherwise it falls back to the shard-stripped filename,
/// since there's no repo to recommend.
pub fn group_models(models: &[ModelSummary]) -> Vec<ModelGroup> {
    struct Accumulator {
        representative_path: PathBuf,
        paths: Vec<PathBuf>,
        shard_label: String,
        size_bytes: u64,
        type_totals: HashMap<u32, u128>,
        errors: Vec<String>,
    }

    let mut groups: BTreeMap<(PathBuf, String), Accumulator> = BTreeMap::new();
    for model in models {
        let parent = model
            .path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_default();
        let stem = model
            .path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        let shard_label = shard_group_label(stem).to_string();

        let acc = groups
            .entry((parent, shard_label.clone()))
            .or_insert_with(|| Accumulator {
                representative_path: model.path.clone(),
                paths: Vec::new(),
                shard_label,
                size_bytes: 0,
                type_totals: HashMap::new(),
                errors: Vec::new(),
            });
        acc.paths.push(model.path.clone());
        acc.size_bytes += model.size_bytes;
        match &model.error {
            Some(error) => acc.errors.push(error.clone()),
            None => {
                for (ty, count) in &model.type_totals {
                    *acc.type_totals.entry(*ty).or_default() += count;
                }
            }
        }
    }

    let mut result: Vec<ModelGroup> = groups
        .into_values()
        .map(|acc| {
            let label = match hf_repo_id_from_path(&acc.representative_path) {
                Some(repo) => match hf_tag_from_label(&acc.shard_label) {
                    Some(tag) => format!("{repo}:{tag}"),
                    None => repo,
                },
                None => acc.shard_label,
            };
            ModelGroup {
                label,
                size_bytes: acc.size_bytes,
                quantization: acc
                    .type_totals
                    .into_iter()
                    .max_by_key(|(_, total)| *total)
                    .map(|(ty, _)| ggml_type_name(ty)),
                errors: acc.errors,
                representative_path: acc.representative_path,
                paths: acc.paths,
            }
        })
        .collect();
    result.sort_by(|a, b| a.label.cmp(&b.label));
    result
}

/// Strips a trailing GGUF shard suffix (`-NNNNN-of-NNNNN`, per the [naming
/// convention](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md#gguf-naming-convention):
/// exactly 5 zero-padded digits on each side) from a file stem, so every
/// shard of one model reduces to the same group label. Returns `stem`
/// unchanged when it has no such suffix. Mirrors llama.cpp's own
/// `get_gguf_split_info` in `common/download.cpp`.
fn shard_group_label(stem: &str) -> &str {
    static SHARD_SUFFIX: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let pattern = SHARD_SUFFIX.get_or_init(|| regex::Regex::new(r"-\d{5}-of-\d{5}$").unwrap());
    match pattern.find(stem) {
        Some(m) => &stem[..m.start()],
        None => stem,
    }
}

/// Recovers the Hugging Face `user/model` repo id from a path under a hub
/// cache directory, whose top-level model folders are always named
/// `models--<user>--<model>` (the layout `-hf`/`--hf-repo` itself downloads
/// into — see llama.cpp's README: "models downloaded with `-hf` are now
/// stored in the standard Hugging Face cache directory"). Checks every
/// ancestor directory, not just the immediate parent, since a repo's GGUF
/// files are nested under `snapshots/<revision>/` (and sometimes a further
/// per-quant subfolder). Returns `None` when no ancestor matches — a plain
/// models directory with no hub-cache structure has no repo id to recover.
fn hf_repo_id_from_path(path: &Path) -> Option<String> {
    for ancestor in path.parent()?.ancestors() {
        let name = ancestor.file_name()?.to_str()?;
        if let Some(rest) = name.strip_prefix("models--") {
            return Some(match rest.split_once("--") {
                Some((user, model)) => format!("{user}/{model}"),
                None => rest.to_string(),
            });
        }
    }
    None
}

/// Extracts the quantization tag llama.cpp's `-hf user/model:TAG` expects,
/// from a shard-suffix-stripped file stem — the trailing run of
/// alphanumeric/underscore characters after the *last* `-` or `.` in the
/// name (e.g. `Llama-3.2-3B-Instruct-Q4_K_M` -> `Q4_K_M`). Mirrors
/// llama.cpp's own tag regex (`common/download.cpp`'s `get_gguf_split_info`:
/// `[-.]([A-Z0-9_]+)$`) exactly, so the tag shown is one llama.cpp itself
/// would recognize — not [`crate::gguf::GgufFile::type_element_totals`]'s
/// coarser ggml-type-based `quantization` label, which can't distinguish
/// e.g. `Q4_K_S` from `Q4_K_M` (both use the `Q4_K` ggml type for most
/// tensors).
fn hf_tag_from_label(label: &str) -> Option<String> {
    let separator = label.rfind(['-', '.'])?;
    let candidate = &label[separator + 1..];
    (!candidate.is_empty()
        && candidate
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_'))
    .then(|| candidate.to_uppercase())
}

pub fn format_list(models: &[ModelSummary], base: &Path) -> String {
    if models.is_empty() {
        return format!("No .gguf files found under {}\n", base.display());
    }

    let groups = group_models(models);

    let nr_width = groups.len().to_string().len().max("NR".len());
    let model_width = groups
        .iter()
        .map(|g| g.label.len())
        .max()
        .unwrap_or(0)
        .max("MODEL".len());
    let quant_width = groups
        .iter()
        .map(|g| g.quantization.as_deref().unwrap_or("-").len())
        .max()
        .unwrap_or(0)
        .max("QUANT".len());

    let mut out = String::new();
    out.push_str(&format!(
        "{:>nr_width$}  {:<model_width$}  {:<quant_width$}  SIZE\n",
        "NR", "MODEL", "QUANT"
    ));
    for (index, group) in groups.iter().enumerate() {
        let nr = index + 1;
        if !group.errors.is_empty() {
            out.push_str(&format!(
                "{nr:>nr_width$}  {:<model_width$}  error: {}\n",
                group.label,
                group.errors.join("; ")
            ));
            continue;
        }
        out.push_str(&format!(
            "{nr:>nr_width$}  {:<model_width$}  {:<quant_width$}  {}\n",
            group.label,
            group.quantization.as_deref().unwrap_or("-"),
            format_bytes(group.size_bytes)
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Writes a minimal GGUF file with one metadata key and, optionally, one
    /// tensor — enough to exercise quantization aggregation across shards.
    fn write_minimal_gguf(path: &Path, architecture: &str, tensor: Option<(u32, u64)>) {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&(tensor.is_some() as u64).to_le_bytes()); // tensor_count
        buf.extend_from_slice(&1u64.to_le_bytes()); // metadata_kv_count

        let key = "general.architecture";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key.as_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes()); // STRING
        buf.extend_from_slice(&(architecture.len() as u64).to_le_bytes());
        buf.extend_from_slice(architecture.as_bytes());

        if let Some((ggml_type, element_count)) = tensor {
            let name = "weight";
            buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
            buf.extend_from_slice(name.as_bytes());
            buf.extend_from_slice(&1u32.to_le_bytes()); // n_dims
            buf.extend_from_slice(&element_count.to_le_bytes());
            buf.extend_from_slice(&ggml_type.to_le_bytes());
            buf.extend_from_slice(&0u64.to_le_bytes()); // offset
        }

        std::fs::File::create(path)
            .unwrap()
            .write_all(&buf)
            .unwrap();
    }

    #[test]
    fn scans_nested_gguf_files_and_ignores_others() {
        let dir = tempfile::tempdir().unwrap();
        write_minimal_gguf(&dir.path().join("a.gguf"), "llama", None);
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        write_minimal_gguf(&dir.path().join("sub/b.GGUF"), "qwen2", None);
        std::fs::write(dir.path().join("readme.txt"), "not a model").unwrap();

        let models = scan_models_dir(dir.path()).unwrap();
        assert_eq!(models.len(), 2);
    }

    #[test]
    fn excludes_clip_projector_sidecars_from_the_scan() {
        let dir = tempfile::tempdir().unwrap();
        write_minimal_gguf(&dir.path().join("model.gguf"), "llama", None);
        write_minimal_gguf(&dir.path().join("mmproj-model.gguf"), "clip", None);

        let models = scan_models_dir(dir.path()).unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].path, dir.path().join("model.gguf"));
    }

    #[cfg(unix)]
    #[test]
    fn collapses_symlinks_to_the_same_underlying_file_into_one_model() {
        // Mirrors the Hugging Face hub cache: two `snapshots/<rev>/` folders
        // (here, `rev1`/`rev2`) can both symlink to the exact same blob when
        // a repo's ref moved without the file's content changing.
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("blobs")).unwrap();
        let blob = dir.path().join("blobs/abc123");
        write_minimal_gguf(&blob, "llama", None);
        std::fs::create_dir(dir.path().join("rev1")).unwrap();
        std::fs::create_dir(dir.path().join("rev2")).unwrap();
        std::os::unix::fs::symlink(&blob, dir.path().join("rev1/model.gguf")).unwrap();
        std::os::unix::fs::symlink(&blob, dir.path().join("rev2/model.gguf")).unwrap();

        let models = scan_models_dir(dir.path()).unwrap();
        assert_eq!(models.len(), 1);
    }

    #[test]
    fn reports_parse_errors_per_file_without_aborting_scan() {
        let dir = tempfile::tempdir().unwrap();
        write_minimal_gguf(&dir.path().join("good.gguf"), "llama", None);
        std::fs::write(dir.path().join("bad.gguf"), b"not a real gguf file").unwrap();

        let models = scan_models_dir(dir.path()).unwrap();
        assert_eq!(models.len(), 2);
        let bad = models.iter().find(|m| m.error.is_some()).unwrap();
        assert!(bad.error.as_ref().unwrap().contains("GGUF"));
    }

    #[test]
    fn resolves_direct_path_before_models_dir() {
        let dir = tempfile::tempdir().unwrap();
        write_minimal_gguf(&dir.path().join("model.gguf"), "llama", None);

        let resolved = resolve_model_path(
            dir.path(),
            &dir.path().join("model.gguf").display().to_string(),
        )
        .unwrap();
        assert_eq!(resolved, dir.path().join("model.gguf"));
    }

    #[test]
    fn resolves_bare_name_under_models_dir() {
        let dir = tempfile::tempdir().unwrap();
        write_minimal_gguf(&dir.path().join("model.gguf"), "llama", None);

        let resolved = resolve_model_path(dir.path(), "model.gguf").unwrap();
        assert_eq!(resolved, dir.path().join("model.gguf"));
    }

    #[test]
    fn errors_when_neither_path_exists() {
        let dir = tempfile::tempdir().unwrap();
        let err = resolve_model_path(dir.path(), "missing.gguf").unwrap_err();
        assert!(err.to_string().contains("missing.gguf"));
    }

    #[test]
    fn resolve_show_target_accepts_an_nr_from_list() {
        let dir = tempfile::tempdir().unwrap();
        write_minimal_gguf(&dir.path().join("a.gguf"), "llama", None);
        write_minimal_gguf(&dir.path().join("b.gguf"), "llama", None);

        // group_models sorts by label, so "a" is NR 1 and "b" is NR 2.
        let resolved = resolve_show_target(dir.path(), "1").unwrap();
        assert_eq!(resolved, dir.path().join("a.gguf"));
        let resolved = resolve_show_target(dir.path(), "2").unwrap();
        assert_eq!(resolved, dir.path().join("b.gguf"));
    }

    #[test]
    fn resolve_show_target_accepts_a_model_label() {
        let dir = tempfile::tempdir().unwrap();
        let repo_dir = dir
            .path()
            .join("models--bartowski--Llama-3.2-3B-Instruct-GGUF/snapshots/rev1");
        std::fs::create_dir_all(&repo_dir).unwrap();
        let file = repo_dir.join("Llama-3.2-3B-Instruct-Q4_K_M.gguf");
        write_minimal_gguf(&file, "llama", None);

        let resolved =
            resolve_show_target(dir.path(), "bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M").unwrap();
        assert_eq!(resolved, file);
    }

    #[test]
    fn resolve_show_target_rejects_an_out_of_range_nr() {
        let dir = tempfile::tempdir().unwrap();
        write_minimal_gguf(&dir.path().join("a.gguf"), "llama", None);

        let err = resolve_show_target(dir.path(), "5").unwrap_err();
        assert!(err.to_string().contains("no model with NR 5"), "{err}");
    }

    #[test]
    fn resolve_show_target_rejects_an_unknown_model_label() {
        let dir = tempfile::tempdir().unwrap();
        write_minimal_gguf(&dir.path().join("a.gguf"), "llama", None);

        let err = resolve_show_target(dir.path(), "no/such-model:Q4_K_M").unwrap_err();
        assert!(err.to_string().contains("was not found"), "{err}");
    }

    #[test]
    fn shard_group_label_strips_well_formed_shard_suffix_only() {
        assert_eq!(
            shard_group_label("Qwen3-Coder-Next-Q4_K_M-00001-of-00004"),
            "Qwen3-Coder-Next-Q4_K_M"
        );
        // Not a valid shard suffix (not 5 digits) — left untouched.
        assert_eq!(shard_group_label("model-1-of-4"), "model-1-of-4");
        // No shard suffix at all.
        assert_eq!(shard_group_label("model-Q4_K_M"), "model-Q4_K_M");
    }

    #[test]
    fn groups_multi_part_shards_into_one_model_summing_size_and_quantization() {
        let dir = tempfile::tempdir().unwrap();
        // Q4_K (type 12) dominates by element count even though the F32
        // (type 0) tensor lives in its own shard.
        write_minimal_gguf(
            &dir.path().join("model-00001-of-00002.gguf"),
            "llama",
            Some((0, 8)),
        );
        write_minimal_gguf(
            &dir.path().join("model-00002-of-00002.gguf"),
            "llama",
            Some((12, 4096)),
        );

        let models = scan_models_dir(dir.path()).unwrap();
        let groups = group_models(&models);

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].label, "model");
        assert_eq!(
            groups[0].size_bytes,
            models[0].size_bytes + models[1].size_bytes
        );
        assert_eq!(groups[0].quantization.as_deref(), Some("Q4_K"));
    }

    #[test]
    fn same_named_files_in_different_directories_are_not_merged() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("rev1")).unwrap();
        std::fs::create_dir(dir.path().join("rev2")).unwrap();
        write_minimal_gguf(&dir.path().join("rev1/model.gguf"), "llama", None);
        write_minimal_gguf(&dir.path().join("rev2/model.gguf"), "llama", None);

        let models = scan_models_dir(dir.path()).unwrap();
        let groups = group_models(&models);

        assert_eq!(groups.len(), 2);
        assert!(groups.iter().all(|g| g.label == "model"));
    }

    #[test]
    fn hf_repo_id_from_path_decodes_hub_cache_directory() {
        let path = Path::new(
            "/mnt/models/models--unsloth--Qwen3-Coder-Next-GGUF/snapshots/abc123/Qwen3-Coder-Next-Q4_K_M/Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf",
        );
        assert_eq!(
            hf_repo_id_from_path(path).as_deref(),
            Some("unsloth/Qwen3-Coder-Next-GGUF")
        );
    }

    #[test]
    fn hf_repo_id_from_path_returns_none_outside_a_hub_cache() {
        let path = Path::new("/mnt/models/my-own-model.gguf");
        assert_eq!(hf_repo_id_from_path(path), None);
    }

    #[test]
    fn hf_repo_id_from_path_handles_an_org_less_repo_name() {
        let path = Path::new("/mnt/models/models--gpt2/snapshots/abc/model.gguf");
        assert_eq!(hf_repo_id_from_path(path).as_deref(), Some("gpt2"));
    }

    #[test]
    fn hf_tag_from_label_extracts_trailing_quant_tag() {
        assert_eq!(
            hf_tag_from_label("Llama-3.2-3B-Instruct-Q4_K_M").as_deref(),
            Some("Q4_K_M")
        );
        assert_eq!(
            hf_tag_from_label("mmproj-gemma-4-12B-it-bf16").as_deref(),
            Some("BF16")
        );
        assert_eq!(
            hf_tag_from_label("GLM-5.2-UD-Q2_K_XL").as_deref(),
            Some("Q2_K_XL")
        );
    }

    #[test]
    fn hf_tag_from_label_returns_none_without_a_recognizable_tag() {
        // No separator at all.
        assert_eq!(hf_tag_from_label("model"), None);
    }

    #[test]
    fn group_models_formats_hf_repo_and_tag_for_hub_cache_files() {
        let dir = tempfile::tempdir().unwrap();
        let repo_dir = dir
            .path()
            .join("models--bartowski--Llama-3.2-3B-Instruct-GGUF/snapshots/rev1");
        std::fs::create_dir_all(&repo_dir).unwrap();
        write_minimal_gguf(
            &repo_dir.join("Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
            "llama",
            None,
        );

        let models = scan_models_dir(dir.path()).unwrap();
        let groups = group_models(&models);

        assert_eq!(groups.len(), 1);
        assert_eq!(
            groups[0].label,
            "bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M"
        );
    }

    #[test]
    fn format_list_numbers_models_starting_from_one() {
        let dir = tempfile::tempdir().unwrap();
        write_minimal_gguf(&dir.path().join("a.gguf"), "llama", None);
        write_minimal_gguf(&dir.path().join("b.gguf"), "llama", None);

        let models = scan_models_dir(dir.path()).unwrap();
        let output = format_list(&models, dir.path());

        let mut lines = output.lines();
        assert_eq!(lines.next().unwrap().split_whitespace().next(), Some("NR"));
        assert!(lines.next().unwrap().trim_start().starts_with("1  "));
        assert!(lines.next().unwrap().trim_start().starts_with("2  "));
    }

    #[test]
    fn resolve_delete_target_by_nr_returns_every_shard() {
        let dir = tempfile::tempdir().unwrap();
        write_minimal_gguf(&dir.path().join("model-00001-of-00002.gguf"), "llama", None);
        write_minimal_gguf(&dir.path().join("model-00002-of-00002.gguf"), "llama", None);

        let group = resolve_delete_target(dir.path(), "1").unwrap();
        assert_eq!(group.paths.len(), 2);
    }

    #[test]
    fn resolve_delete_target_by_model_label() {
        let dir = tempfile::tempdir().unwrap();
        let repo_dir = dir
            .path()
            .join("models--bartowski--Llama-3.2-3B-Instruct-GGUF/snapshots/rev1");
        std::fs::create_dir_all(&repo_dir).unwrap();
        write_minimal_gguf(
            &repo_dir.join("Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
            "llama",
            None,
        );

        let group =
            resolve_delete_target(dir.path(), "bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M")
                .unwrap();
        assert_eq!(group.paths.len(), 1);
    }

    #[test]
    fn resolve_delete_target_by_direct_path_returns_the_whole_group() {
        let dir = tempfile::tempdir().unwrap();
        let shard1 = dir.path().join("model-00001-of-00002.gguf");
        let shard2 = dir.path().join("model-00002-of-00002.gguf");
        write_minimal_gguf(&shard1, "llama", None);
        write_minimal_gguf(&shard2, "llama", None);

        // Naming just one shard's own path should still resolve (and later
        // delete) the whole group, not that one file alone.
        let group = resolve_delete_target(dir.path(), &shard2.display().to_string()).unwrap();
        assert_eq!(group.paths.len(), 2);
    }

    #[test]
    fn resolve_delete_target_falls_back_to_a_synthetic_single_file_group_for_an_mmproj_sidecar() {
        let dir = tempfile::tempdir().unwrap();
        write_minimal_gguf(&dir.path().join("mmproj-model.gguf"), "clip", None);

        // mmproj sidecars are excluded from every real group (see
        // `excludes_clip_projector_sidecars_from_the_scan`), but `delete`
        // should still be able to name and remove one directly.
        let group = resolve_delete_target(dir.path(), "mmproj-model.gguf").unwrap();
        assert_eq!(group.paths, vec![dir.path().join("mmproj-model.gguf")]);
    }

    #[test]
    fn resolve_delete_target_rejects_an_out_of_range_nr() {
        let dir = tempfile::tempdir().unwrap();
        write_minimal_gguf(&dir.path().join("a.gguf"), "llama", None);

        let err = resolve_delete_target(dir.path(), "5").unwrap_err();
        assert!(err.to_string().contains("no model with NR 5"), "{err}");
    }

    #[test]
    fn resolve_delete_target_rejects_an_unknown_model_label() {
        let dir = tempfile::tempdir().unwrap();
        write_minimal_gguf(&dir.path().join("a.gguf"), "llama", None);

        let err = resolve_delete_target(dir.path(), "no/such-model:Q4_K_M").unwrap_err();
        assert!(err.to_string().contains("was not found"), "{err}");
    }

    #[test]
    fn delete_model_removes_every_shard() {
        let dir = tempfile::tempdir().unwrap();
        let shard1 = dir.path().join("model-00001-of-00002.gguf");
        let shard2 = dir.path().join("model-00002-of-00002.gguf");
        write_minimal_gguf(&shard1, "llama", None);
        write_minimal_gguf(&shard2, "llama", None);

        let models = scan_models_dir(dir.path()).unwrap();
        let groups = group_models(&models);
        assert_eq!(groups.len(), 1);

        delete_model(dir.path(), &groups[0]).unwrap();

        assert!(!shard1.exists());
        assert!(!shard2.exists());
    }

    #[test]
    fn delete_model_removes_now_empty_ancestor_directories_but_not_models_dir() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("sub/nested");
        std::fs::create_dir_all(&nested).unwrap();
        let file = nested.join("model.gguf");
        write_minimal_gguf(&file, "llama", None);

        let models = scan_models_dir(dir.path()).unwrap();
        let groups = group_models(&models);
        assert_eq!(groups.len(), 1);

        delete_model(dir.path(), &groups[0]).unwrap();

        assert!(!file.exists());
        assert!(!nested.exists());
        assert!(!dir.path().join("sub").exists());
        assert!(dir.path().exists());
    }

    #[cfg(unix)]
    #[test]
    fn delete_model_prunes_the_whole_repo_tree_when_it_was_the_only_model_left() {
        // A blob's own `blobs/` directory sits *beside* the symlink's
        // `snapshots/<rev>/` chain, not inside it — cleaning up only the
        // latter would leave a hollowed-out `blobs/` (and the whole repo
        // directory, since it'd still contain that leftover `blobs/`)
        // behind even after the blob itself was reclaimed.
        let dir = tempfile::tempdir().unwrap();
        let repo = dir.path().join("models--org--solo");
        std::fs::create_dir_all(repo.join("blobs")).unwrap();
        std::fs::create_dir_all(repo.join("snapshots/rev1")).unwrap();

        let blob = repo.join("blobs/only");
        write_minimal_gguf(&blob, "llama", None);
        std::os::unix::fs::symlink(&blob, repo.join("snapshots/rev1/model.gguf")).unwrap();

        let models = scan_models_dir(dir.path()).unwrap();
        let groups = group_models(&models);
        assert_eq!(groups.len(), 1);

        delete_model(dir.path(), &groups[0]).unwrap();

        assert!(!repo.exists(), "the whole repo directory should be gone");
        assert!(dir.path().exists());
    }

    #[cfg(unix)]
    #[test]
    fn delete_model_reclaims_an_unreferenced_blob_but_keeps_one_still_in_use() {
        // Mirrors a real Hugging Face hub cache: `blob_a` is referenced from
        // two snapshot revisions (a moved ref reusing already-downloaded
        // content — `scan_models_dir`'s own dedup collapses that pair down
        // to one listed file, so only `rev1`'s symlink is ever part of a
        // group), while `blob_b` has exactly one reference.
        let dir = tempfile::tempdir().unwrap();
        let repo = dir.path().join("models--org--model");
        std::fs::create_dir_all(repo.join("blobs")).unwrap();
        std::fs::create_dir_all(repo.join("snapshots/rev1")).unwrap();
        std::fs::create_dir_all(repo.join("snapshots/rev2")).unwrap();

        let blob_a = repo.join("blobs/aaa");
        let blob_b = repo.join("blobs/bbb");
        write_minimal_gguf(&blob_a, "llama", None);
        write_minimal_gguf(&blob_b, "llama", None);

        std::os::unix::fs::symlink(&blob_a, repo.join("snapshots/rev1/model-A.gguf")).unwrap();
        std::os::unix::fs::symlink(&blob_a, repo.join("snapshots/rev2/model-A.gguf")).unwrap();
        std::os::unix::fs::symlink(&blob_b, repo.join("snapshots/rev1/model-B.gguf")).unwrap();

        let models = scan_models_dir(dir.path()).unwrap();
        let groups = group_models(&models);
        assert_eq!(groups.len(), 2);

        for group in &groups {
            delete_model(dir.path(), group).unwrap();
        }

        assert!(!repo.join("snapshots/rev1/model-A.gguf").exists());
        assert!(
            blob_a.exists(),
            "blob_a is still referenced from rev2 and must survive"
        );
        assert!(
            !blob_b.exists(),
            "blob_b had no other reference and should have been reclaimed"
        );
    }
}
