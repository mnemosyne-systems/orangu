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
        .find(|group| group.matches_label(requested))
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

/// Resolves what a *load* was given — a path, a bare name, an `NR`, or a
/// `MODEL` label — to the file to open **and** the spec that should stand as
/// the loaded model's id.
///
/// The second half is why this exists rather than [`resolve_or_fetch_model`]
/// alone. A caller that names a model by `NR` (the web console's model
/// manager clicks a row, and a row is a position) would otherwise end up
/// with `"7"` as the model id every response's `model` field reports, the
/// slot-store fingerprint is built from, and — since loading a model
/// re-executes the server with that spec in `argv` — the process is
/// restarted on. A number that means nothing outside the listing it came
/// from, and a different model as soon as one is added. Taking the resolved
/// group's own `MODEL` label instead gives the same id a
/// `orangu-server <repo>` start would have produced for the same file.
///
/// A spec that names nothing on disk falls through to
/// [`resolve_or_fetch_model`] — a Hugging Face repo to fetch first — and
/// keeps the spec as written, exactly as the CLI does.
pub fn resolve_load_target(models_dir: &Path, requested: &str) -> Result<(PathBuf, String)> {
    let Ok(group) = resolve_delete_target(models_dir, requested) else {
        return Ok((
            resolve_or_fetch_model(models_dir, requested)?,
            requested.to_string(),
        ));
    };

    // The bare `MODEL` label is the right id whenever it is unambiguous —
    // it is exactly what a `orangu-server <repo>` start would have produced
    // for this file. It is *not* unambiguous when a repo has more than one
    // quantization on disk: those rows all print the same bare label, and
    // resolving it takes whichever comes first. Handing that back would
    // resolve row 4 to row 3's file, which for a caller that is about to
    // restart the server on it means silently loading a different model
    // than the one asked for.
    //
    // So disambiguate, but only then: `<repo>:<quant>` is a spelling
    // `ModelGroup::matches_label` already accepts, so it resolves straight
    // back to this exact group.
    let label = match ambiguous_label(models_dir, &group) {
        true => group
            .quantization
            .as_ref()
            .and_then(|quant| group.hf_repo.as_ref().map(|repo| format!("{repo}:{quant}")))
            .unwrap_or(group.label),
        false => group.label,
    };
    Ok((group.representative_path, label))
}

/// Whether more than one model under `models_dir` answers to `group`'s own
/// `MODEL` label — two quantizations of one repo, most commonly.
fn ambiguous_label(models_dir: &Path, group: &ModelGroup) -> bool {
    let Ok(models) = scan_models_dir(models_dir) else {
        return false;
    };
    group_models(&models)
        .iter()
        .filter(|other| other.matches_label(&group.label))
        .count()
        > 1
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
        let hf_repo = hf_repo_id_from_path(&path);
        let local_commit = hf_local_commit_from_path(&path);
        return Ok(ModelGroup {
            label: path.display().to_string(),
            size_bytes,
            quantization: None,
            errors: Vec::new(),
            representative_path: path.clone(),
            paths: vec![path],
            hf_repo,
            local_commit,
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
        .find(|group| group.matches_label(requested))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "'{requested}' was not found as a file, an NR, or a MODEL name; run 'orangu-server list' to see valid values"
            )
        })
}

/// Resolves whatever `refresh` was given to the full [`ModelGroup`] to
/// delete and download again, the same way [`resolve_delete_target`] does —
/// a direct/relative/absolute path or a bare name under `models_dir` first,
/// then an `NR` from `list`'s first column, then a `MODEL` name from its
/// second — with two deliberate differences: an ambiguous `MODEL` name is an
/// error rather than a first-match, and a companion mmproj sidecar (which
/// `delete` happily removes as a synthetic one-file group) is refused,
/// since `download` only ever fetches one alongside its base model.
///
/// Two quantizations of one repo share a `MODEL` cell (`QUANT` is what tells
/// them apart), so a bare repo id can name more than one row. `delete` takes
/// the first and spells out in its confirmation line which quantization that
/// was; `refresh` can't lean on that, because the copy it deletes is the one
/// it then re-downloads — silently picking a row would refresh the wrong
/// quantization *and* leave the one the user meant untouched. So it says
/// which quantizations are on disk and asks for one, `<repo>:<quant>`.
pub fn resolve_refresh_target(models_dir: &Path, requested: &str) -> Result<ModelGroup> {
    let models = scan_models_dir(models_dir)?;
    let mut groups = group_models(&models);

    if let Ok(path) = resolve_model_path(models_dir, requested) {
        if let Some(index) = groups.iter().position(|g| g.paths.contains(&path)) {
            return Ok(groups.swap_remove(index));
        }
        // A file that resolves but belongs to no group is a companion
        // sidecar — an mmproj projector, which `scan_models_dir` deliberately
        // keeps out of the listing. `delete` synthesizes a one-file group so
        // it can still be removed on its own; `refresh` can't do the same,
        // since `download` only ever fetches a sidecar *alongside* the base
        // model it belongs to.
        anyhow::bail!(
            "'{requested}' is a companion file (mmproj), not a model of its own; refresh the model it was downloaded with instead"
        );
    }

    if let Ok(nr) = requested.parse::<usize>() {
        let count = groups.len();
        return nr
            .checked_sub(1)
            .filter(|index| *index < groups.len())
            .map(|index| groups.swap_remove(index))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "no model with NR {nr} ({count} model(s) found under {}; run 'orangu-server list' to see them)",
                    models_dir.display()
                )
            });
    }

    let matches: Vec<usize> = groups
        .iter()
        .enumerate()
        .filter(|(_, group)| group.matches_label(requested))
        .map(|(index, _)| index)
        .collect();
    match matches.as_slice() {
        [] => anyhow::bail!(
            "'{requested}' was not found as a file, an NR, or a MODEL name; run 'orangu-server list' to see valid values"
        ),
        [index] => Ok(groups.swap_remove(*index)),
        indices => {
            let quants: Vec<&str> = indices
                .iter()
                .map(|index| groups[*index].quantization.as_deref().unwrap_or("-"))
                .collect();
            // Naming the quantization only disambiguates when the rows
            // actually differ by one. Two snapshots of the same repo at the
            // same quantization (or a request that already carries a
            // `:quant`) can only be told apart by their `NR`.
            let distinct = quants
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len();
            let hint = if distinct == indices.len() && !requested.contains(':') {
                format!(
                    "name the quantization too — '{requested}:{}' — or use an NR from 'orangu-server list'",
                    quants[0]
                )
            } else {
                "use an NR from 'orangu-server list' to name one".to_string()
            };
            anyhow::bail!(
                "'{requested}' names {} models on disk ({}); {hint}",
                indices.len(),
                quants.join(", "),
            )
        }
    }
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
    // Resolve symlinks while they still exist: registry records use the
    // canonical blob path, which cannot be recovered after the snapshot
    // link and its final blob have both been removed.
    let registry_paths: Vec<PathBuf> = group
        .paths
        .iter()
        .map(|path| std::fs::canonicalize(path).unwrap_or_else(|_| path.clone()))
        .collect();
    for path in &group.paths {
        let blob_target = std::fs::symlink_metadata(path)
            .ok()
            .filter(std::fs::Metadata::is_symlink)
            .and_then(|_| std::fs::canonicalize(path).ok());

        std::fs::remove_file(path)
            .with_context(|| format!("failed to delete {}", path.display()))?;

        // `blob_target` came from `canonicalize`, so every path it is
        // compared against has to be canonical too. On macOS the temporary
        // and home directories reach the filesystem through a symlink
        // (`/var` → `/private/var`), so a canonical blob path and an
        // uncanonicalized repo root disagree on their prefix and none of
        // this ran — the blob survived its last symlink and the space was
        // never reclaimed. Linux has no such symlink, which is why that
        // only ever showed up off Linux.
        let canonical = |p: &Path| std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
        if let Some(blob) = blob_target
            && let Some(repo_root) = hf_repo_root_from_path(path)
            && blob.starts_with(canonical(&repo_root).join("blobs"))
            && !blob_still_referenced(&repo_root, &blob)
            && std::fs::remove_file(&blob).is_ok()
        {
            // `blob` sits under a sibling `blobs/` directory, not under
            // `path`'s own `snapshots/...` chain, so it needs its own
            // upward sweep — otherwise a now-empty `blobs/` (and, once
            // both it and `snapshots/` are gone, the whole repo directory)
            // would survive even though nothing is left inside it.
            //
            // Swept against a canonical `models_dir` for the same reason
            // the prefix test above uses one: `blob` is canonical, and the
            // sweep stops the moment an ancestor stops matching `stop_at`.
            remove_empty_ancestors(&blob, &canonical(models_dir));
        }

        remove_empty_ancestors(path, models_dir);
    }
    if let Err(err) = crate::model_registry::forget(&registry_paths) {
        eprintln!("warning: could not update ~/.orangu/models: {err:#}");
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

fn hf_snapshot_relative_path(path: &Path) -> Option<String> {
    let components: Vec<&str> = path
        .components()
        .map(|c| c.as_os_str().to_str())
        .collect::<Option<Vec<_>>>()?;
    let snapshots = components.iter().position(|c| *c == "snapshots")?;
    let start = snapshots + 2;
    (start < components.len()).then(|| components[start..].join("/"))
}

fn hf_blob_oid_from_path(path: &Path) -> Option<String> {
    std::fs::read_link(path)
        .ok()
        .and_then(|target| {
            target
                .file_name()
                .and_then(|name| name.to_str())
                .map(str::to_string)
        })
        .or_else(|| {
            std::fs::canonicalize(path).ok().and_then(|target| {
                target
                    .file_name()
                    .and_then(|name| name.to_str())
                    .map(str::to_string)
            })
        })
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
    /// The `QUANT` column: the quantization scheme this model was produced
    /// with, taken from its own filename tag ([`quant_tag_from_label`]) when
    /// it carries one, and only otherwise from the ggml type most of its
    /// tensor elements are stored as. See [`group_models`] for why the name
    /// wins.
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
    /// The Hugging Face `user/model` repo id this group was downloaded from,
    /// when it lives under a hub-cache directory — the same id [`label`]'s
    /// `:quant` tag is appended to. `None` for a model outside that layout,
    /// which has no repo to check for updates against.
    ///
    /// [`label`]: ModelGroup::label
    pub hf_repo: Option<String>,
    /// The commit sha this group was downloaded at — the `snapshots/<sha>/`
    /// directory name its files sit under. Compared against the Hub's live
    /// `main` commit to decide whether `list` marks this row `(Refresh)`.
    pub local_commit: Option<String>,
}

impl ModelGroup {
    /// Whether `requested` names this group's `MODEL` column. Accepts the
    /// label as printed *and* the fully-tagged `<repo>:<quant>` spelling —
    /// which is what the same row printed before [`group_models`] began
    /// dropping a `:TAG` that only repeats the `QUANT` column, and so what a
    /// `model =` config value, a shell alias, or a script written against an
    /// older listing still says. Keep this in step with whatever
    /// `group_models` does to `label`: every spelling ever printed has to go
    /// on resolving locally, or a saved config silently turns into a
    /// re-download.
    ///
    /// It is also the only way to name one particular quantization of a repo
    /// that has several on disk: those rows all print the same bare `MODEL`,
    /// so a bare request matches whichever comes first (an `NR` from `list`
    /// picks a row exactly, too).
    pub fn matches_label(&self, requested: &str) -> bool {
        if self.label == requested {
            return true;
        }
        match (&self.hf_repo, &self.quantization) {
            (Some(repo), Some(quant)) => requested == format!("{repo}:{quant}"),
            _ => false,
        }
    }

    /// Whether this group's downloadable files differ from the latest repo
    /// tree. Compared per row rather than per repo, so a commit that touched
    /// only another quantization does not mark this row `(Refresh)`. Always
    /// `false` for a model outside the hub-cache layout and for a repo the
    /// lookup did not reach - an unreachable Hub means "unknown", never
    /// "behind".
    ///
    /// Both what `list` marks `(Refresh)` and what `refresh` leaves
    /// un-greyed, so the two always agree on which rows are worth acting on.
    ///
    /// [`local_commit`]: ModelGroup::local_commit
    pub fn is_behind(
        &self,
        latest_updates: &HashMap<String, crate::model_download::RepoUpdateInfo>,
    ) -> bool {
        let Some(repo) = self.hf_repo.as_deref() else {
            return false;
        };
        let Some(update) = latest_updates.get(repo) else {
            return false;
        };
        let stem = self
            .representative_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        let tag = hf_tag_from_label(shard_group_label(stem));
        let Ok(latest_files) =
            crate::model_download::select_files_to_download(&update.files, tag.as_deref())
        else {
            return false;
        };
        if latest_files.len() != self.paths.len() {
            return true;
        }
        let latest_by_path: HashMap<&str, &str> = latest_files
            .iter()
            .map(|file| (file.path.as_str(), file.oid.as_str()))
            .collect();
        self.paths.iter().any(|path| {
            let Some(relative) = hf_snapshot_relative_path(path) else {
                return true;
            };
            let Some(local_oid) = hf_blob_oid_from_path(path) else {
                return true;
            };
            match latest_by_path.get(relative.as_str()) {
                Some(oid) => **oid != local_oid,
                None => true,
            }
        })
    }

    /// The `<user>/<model>[:tag]` spec that downloads exactly this group's
    /// file(s) again — what `refresh` hands
    /// [`crate::model_download::download_model`] after deleting the local
    /// copy. `None` for a model outside the hub-cache layout, which has no
    /// repo to re-download from.
    ///
    /// The tag is the one this model's own filename carries, read back off
    /// the file rather than taken from [`quantization`]: `quantization` (the
    /// `QUANT` column) falls back to a ggml type counted out of the tensors
    /// for a file whose name says nothing, and that type names no file in
    /// the repo — `download` would reject it as an unknown quant instead of
    /// re-fetching the model that is actually here.
    ///
    /// [`quantization`]: ModelGroup::quantization
    pub fn download_spec(&self) -> Option<String> {
        let repo = self.hf_repo.as_deref()?;
        let stem = self
            .representative_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        Some(match hf_tag_from_label(shard_group_label(stem)) {
            Some(tag) => format!("{repo}:{tag}"),
            None => repo.to_string(),
        })
    }
}

/// Collapses a multi-part model's shard files (`name-00001-of-00004.gguf`,
/// `name-00002-of-00004.gguf`, ...) into a single [`ModelGroup`]: one entry
/// per model rather than one per shard, with `size_bytes` summed across
/// shards.
/// Grouping is keyed by (parent directory, shard-suffix-stripped file stem),
/// so two files that merely share a name in different directories (e.g. two
/// Hugging Face cache snapshots of the same release) are kept separate.
///
/// `quantization` is the scheme named by the file itself
/// ([`quant_tag_from_label`]) whenever it carries one, and only falls back to
/// the ggml type most of the model's tensor *elements* are stored as (summed
/// across every shard — a single shard's tensors are only part of the whole
/// model, see [`crate::gguf::GgufFile::type_element_totals`]) for a file
/// whose name says nothing. The name has to win: a mixed scheme is *defined*
/// by storing some tensors at a heavier type than its name, so the dominant
/// ggml type of a genuine `Q4_K_M` model can legitimately come out `Q5_K` or
/// `Q6_K` — a `QUANT` column contradicting the `MODEL` label right next to
/// it, for a model that is exactly what its name says.
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
            let hf_repo = hf_repo_id_from_path(&acc.representative_path);
            let quant_tag = quant_tag_from_label(&acc.shard_label);
            let label = match &hf_repo {
                Some(repo) => match hf_tag_from_label(&acc.shard_label) {
                    Some(tag) => format!("{repo}:{tag}"),
                    None => repo.clone(),
                },
                None => acc.shard_label,
            };
            let local_commit = hf_local_commit_from_path(&acc.representative_path);
            ModelGroup {
                label,
                size_bytes: acc.size_bytes,
                quantization: quant_tag.or_else(|| {
                    acc.type_totals
                        .into_iter()
                        .max_by_key(|(_, total)| *total)
                        .map(|(ty, _)| ggml_type_name(ty))
                }),
                errors: acc.errors,
                representative_path: acc.representative_path,
                paths: acc.paths,
                hf_repo,
                local_commit,
            }
        })
        .collect();

    // `list` prints the quantization in its own `QUANT` column, so carrying it
    // in `MODEL` too just makes the widest column wider for no added
    // information — drop the `:TAG` suffix when it says exactly what `QUANT`
    // already does. Two quantizations of one repo therefore share a `MODEL`
    // cell and are told apart by their `QUANT` cells; either spelling still
    // resolves, and the tagged one is what names a specific quantization —
    // see [`ModelGroup::matches_label`].
    for group in &mut result {
        let (Some(repo), Some(quant)) = (&group.hf_repo, &group.quantization) else {
            continue;
        };
        if group.label == format!("{repo}:{quant}") {
            group.label = repo.clone();
        }
    }

    // Stable, so two rows left sharing a label keep the order their (parent
    // directory, file stem) key gave them — `NR` and "first match wins"
    // resolution stay the same between one `list` and the next.
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

pub fn hf_repo_for_path(path: &Path) -> Option<String> {
    hf_repo_id_from_path(path)
}

/// The commit sha a Hugging Face hub-cache path was downloaded at: the name
/// of the `snapshots/<commit>/...` directory a file sits under — the same
/// sha [`crate::model_download::download_model`] names that directory after
/// and records in `refs/main`. Checks every ancestor the same way
/// [`hf_repo_id_from_path`] does, since a file can sit a further per-quant
/// subfolder below `snapshots/<commit>/`. `None` outside that layout, or for
/// a path directly under `models--<user>--<model>/` with no `snapshots`
/// ancestor at all.
fn hf_local_commit_from_path(path: &Path) -> Option<String> {
    let mut child: Option<&str> = None;
    for ancestor in path.parent()?.ancestors() {
        let name = ancestor.file_name()?.to_str()?;
        if name == "snapshots" {
            return child.map(str::to_string);
        }
        child = Some(name);
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

/// The quantization scheme a file names itself for: [`hf_tag_from_label`]'s
/// trailing tag, but only when it actually reads as a quantization —
/// [`hf_tag_from_label`] happily returns `IT` for `gemma-4-E2B-it`, or `1B`
/// for `TinyLlama-1.1B`, which are fine as `-hf` tags to try but must never
/// be shown in `list`'s `QUANT` column. Everything else falls back to the
/// ggml type counted out of the tensors (see [`group_models`]).
fn quant_tag_from_label(label: &str) -> Option<String> {
    let tag = hf_tag_from_label(label)?;
    is_quant_tag(&tag).then_some(tag)
}

/// The quantization scheme to show for one already-resolved model file, by
/// the same rule [`group_models`] fills `list`'s `QUANT` column with: the tag
/// the filename carries whenever it reads as a quantization
/// ([`quant_tag_from_label`]), falling back to the ggml type most of the
/// file's tensor *elements* are stored as.
///
/// Unlike [`group_models`], the fallback sees a single file rather than every
/// shard of a multi-part model, so for a sharded model it reports the
/// dominant type of the shard handed in. Only that fallback is affected — a
/// file whose name carries a quantization tag (the overwhelmingly common
/// case) reads its scheme off the name either way.
pub fn quantization_for_file(path: &Path, gguf: &GgufFile) -> Option<String> {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    quant_tag_from_label(shard_group_label(stem)).or_else(|| {
        gguf.type_element_totals()
            .into_iter()
            .max_by_key(|(_, total)| *total)
            .map(|(ty, _)| ggml_type_name(ty))
    })
}

/// Whether an already-uppercased tag names a ggml quantization: the float
/// types spelled out, or one of the `Q`/`IQ`/`TQ` families — a digit-led
/// bit-width followed by any number of `_`-separated variant parts (`Q4_0`,
/// `Q6_K`, `Q4_K_M`, `IQ2_XXS`, `IQ4_NL`, `TQ1_0`, and unsloth's own
/// `Q4_K_XL`). Deliberately a shape check rather than a fixed list: new
/// quantizations appear regularly, and a name-shaped tag this build's
/// [`ggml_type_name`] doesn't know yet is still the right thing to print for
/// a file that carries it.
fn is_quant_tag(tag: &str) -> bool {
    static QUANT_TAG: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let pattern = QUANT_TAG.get_or_init(|| {
        regex::Regex::new(r"^(?:F16|F32|BF16|MXFP\d+|(?:Q|IQ|TQ)\d+(?:_[A-Z0-9]+)*)$").unwrap()
    });
    pattern.is_match(tag)
}

/// Whether this build can load a model's architecture, plus the
/// architecture id it was read from — one entry per [`ModelGroup`], aligned
/// by index, for the `SUPPORTED` column [`format_groups`] renders. It's
/// populated by `orangu-server` (which owns the architecture resolver), not
/// here, so the lib stays free of the engine's arch tables.
#[derive(Debug, Clone, Default)]
pub struct ModelSupport {
    /// `general.architecture` read from the group's representative file, or
    /// `None` if that file couldn't be opened or lacks the key.
    pub architecture: Option<String>,
    /// Whether this build recognises (and can load) that architecture.
    pub supported: bool,
    /// The name of a tensor quantization this build has no dequantizer for
    /// (e.g. `TQ1_0`), when the group carries one — checked across *every*
    /// shard, since a split model's later shards can introduce a type shard
    /// 1 never uses. `None` when every tensor type is readable.
    ///
    /// Kept separate from [`ModelSupport::supported`] so the `SUPPORTED`
    /// cell can say *which* of the two reasons applies: an architecture
    /// this build doesn't implement is a different problem from a
    /// quantization it can't decode, and only the latter is fixed by
    /// downloading a different file of the same model.
    pub unsupported_quant: Option<String>,
}

impl ModelSupport {
    /// Whether this build can actually load the model — a recognised
    /// architecture *and* no unreadable tensor type.
    pub fn loadable(&self) -> bool {
        self.supported && self.unsupported_quant.is_none()
    }

    /// The `SUPPORTED` cell text, e.g. `Yes (llama)`, `No (glm-dsa)`, or
    /// `No (llama, TQ1_0)` when the architecture is fine but a tensor type
    /// isn't. Public because the web console's model manager draws the same
    /// column, and drawing it from the same function is what keeps the two
    /// from ever disagreeing about the same file.
    pub fn cell(&self) -> String {
        let arch = self.architecture.as_deref().unwrap_or("unknown");
        match (&self.unsupported_quant, self.supported) {
            (Some(quant), true) => format!("No ({arch}, {quant})"),
            (Some(quant), false) => format!("No ({arch}, {quant})"),
            (None, true) => format!("Yes ({arch})"),
            (None, false) => format!("No ({arch})"),
        }
    }
}

#[cfg(test)]
mod support_cell_tests {
    use super::ModelSupport;

    fn support(arch: &str, supported: bool, quant: Option<&str>) -> ModelSupport {
        ModelSupport {
            architecture: Some(arch.to_string()),
            supported,
            unsupported_quant: quant.map(str::to_string),
        }
    }

    /// The distinction the cell exists to draw: a readable architecture in
    /// an unreadable quantization is *not* the same as an unimplemented
    /// architecture, and only the first is worth re-downloading a different
    /// file for. Before the quant check existed the first case rendered as
    /// a plain `Yes (llama)` and then failed at load.
    #[test]
    fn cell_distinguishes_a_bad_quant_from_a_bad_architecture() {
        assert_eq!(support("llama", true, None).cell(), "Yes (llama)");
        assert_eq!(
            support("llama", true, Some("TQ1_0")).cell(),
            "No (llama, TQ1_0)"
        );
        assert_eq!(support("glm-dsa", false, None).cell(), "No (glm-dsa)");
    }

    /// `loadable()` is what greys the row and gates the pickers, so it has
    /// to fail on *either* reason, not just the architecture.
    #[test]
    fn loadable_requires_both_a_known_arch_and_a_readable_quant() {
        assert!(support("llama", true, None).loadable());
        assert!(!support("llama", true, Some("TQ1_0")).loadable());
        assert!(!support("glm-dsa", false, None).loadable());
    }
}

/// Dim/grey ANSI SGR codes, used to visually deprioritize a row the caller's
/// [`Dimming`] mode says isn't what this command is about — visible but
/// greyed, not hidden. Only emitted when the caller asks for a mode other
/// than [`Dimming::Off`] (i.e. writing to a terminal), so the plain `list`
/// output that the shell-completion scripts parse by column stays
/// escape-free.
const ANSI_DIM: &str = "\x1b[2m";
const ANSI_RESET: &str = "\x1b[0m";

/// Which rows [`format_groups`] greys out — the same table, deprioritizing
/// whatever the command asking for it can't act on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dimming {
    /// No escapes at all: every row plain. What a caller writing to a pipe
    /// or a file passes, since the shell-completion scripts parse `list`'s
    /// output by column and must never see an ANSI sequence.
    Off,
    /// `list`/`show`/`delete`: grey the rows whose architecture this build
    /// can't load. Needs a non-empty `support` slice to have any effect.
    Unsupported,
    /// `refresh`: grey the rows already at their latest downloadable file
    /// set, so only the `NR`s worth refreshing - the ones marked
    /// `(Refresh)` - stand out. Needs a non-empty `latest_updates` map to
    /// have any effect.
    UpToDate,
}

/// The `list` table for every `.gguf` model found, with no Hugging Face
/// update check and no `SUPPORTED` column — the plain, fully offline
/// rendering used by callers (tests) that don't need either. `list` itself
/// calls [`format_groups`] directly so it can pass `latest_updates` and the
/// per-group [`ModelSupport`].
pub fn format_list(models: &[ModelSummary], base: &Path) -> String {
    if models.is_empty() {
        return format!("No .gguf files found under {}\n", base.display());
    }
    format_groups(
        &group_models(models),
        base,
        &HashMap::new(),
        &[],
        Dimming::Off,
    )
}

/// Renders the `list` table from already-grouped models. `latest_updates`
/// maps each [`ModelGroup::hf_repo`] id to the latest downloadable file
/// state the Hub reports for that repo. A row gets a trailing `(Refresh)`
/// marker, appended after `SIZE`, exactly when the latest file set or blob
/// oids for that row differ from what is on disk. The marker sits after
/// `SIZE` rather than inside `MODEL` so a consumer that reads `list`'s
/// output by column position (e.g. the shell completion scripts, which only
/// read `NR`/`MODEL`) is unaffected.
///
/// `support`, when non-empty, adds a trailing `SUPPORTED` column reading
/// `Yes (<arch>)`/`No (<arch>)` per row (aligned to `groups` by index — see
/// [`ModelSupport`]); an empty slice omits the column entirely. `dim` picks
/// which rows are greyed (see [`Dimming`]); callers pass anything but
/// [`Dimming::Off`] only when writing to a terminal, so piped output (what
/// the completion scripts parse) never carries ANSI escapes.
pub fn format_groups(
    groups: &[ModelGroup],
    base: &Path,
    latest_updates: &HashMap<String, crate::model_download::RepoUpdateInfo>,
    support: &[ModelSupport],
    dim: Dimming,
) -> String {
    format_groups_with_last_used(groups, base, latest_updates, support, dim, None)
}

/// [`format_groups`] with the persistent `LAST_USED` column requested by
/// `orangu-server list`. Other model pickers keep their compact table by
/// passing through [`format_groups`] instead.
pub fn format_groups_with_last_used(
    groups: &[ModelGroup],
    base: &Path,
    latest_updates: &HashMap<String, crate::model_download::RepoUpdateInfo>,
    support: &[ModelSupport],
    dim: Dimming,
    last_used: Option<&[Option<u64>]>,
) -> String {
    let order: Vec<usize> = (0..groups.len()).collect();
    format_groups_with_last_used_in_order(
        groups,
        base,
        latest_updates,
        support,
        dim,
        last_used,
        &order,
    )
}

/// [`format_groups_with_last_used`] with an explicit display order. Entries
/// in `order` are indices into the canonical `groups` slice; the printed `NR`
/// remains that original index plus one, and aligned metadata is read by that
/// same index.
pub fn format_groups_with_last_used_in_order(
    groups: &[ModelGroup],
    base: &Path,
    latest_updates: &HashMap<String, crate::model_download::RepoUpdateInfo>,
    support: &[ModelSupport],
    dim: Dimming,
    last_used: Option<&[Option<u64>]>,
    order: &[usize],
) -> String {
    if groups.is_empty() {
        return format!("No .gguf files found under {}\n", base.display());
    }

    let show_support = !support.is_empty();
    let show_last_used = last_used.is_some();

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
    // SIZE needs a fixed width whenever another column follows it. Error
    // rows carry no size, so they don't factor into the width.
    let size_width = if show_support || show_last_used {
        groups
            .iter()
            .filter(|g| g.errors.is_empty())
            .map(|g| format_bytes(g.size_bytes).len())
            .max()
            .unwrap_or(0)
            .max("SIZE".len())
    } else {
        0
    };

    let mut out = String::new();
    if show_support && show_last_used {
        out.push_str(&format!(
            "{:>nr_width$}  {:<model_width$}  {:<quant_width$}  {:<size_width$}  {:<16}  SUPPORTED\n",
            "NR", "MODEL", "QUANT", "SIZE", "LAST_USED"
        ));
    } else if show_support {
        out.push_str(&format!(
            "{:>nr_width$}  {:<model_width$}  {:<quant_width$}  {:<size_width$}  SUPPORTED\n",
            "NR", "MODEL", "QUANT", "SIZE"
        ));
    } else if show_last_used {
        out.push_str(&format!(
            "{:>nr_width$}  {:<model_width$}  {:<quant_width$}  {:<size_width$}  LAST_USED\n",
            "NR", "MODEL", "QUANT", "SIZE"
        ));
    } else {
        out.push_str(&format!(
            "{:>nr_width$}  {:<model_width$}  {:<quant_width$}  SIZE\n",
            "NR", "MODEL", "QUANT"
        ));
    }
    for &index in order {
        let group = &groups[index];
        let nr = index + 1;
        let refresh = group.is_behind(latest_updates);
        if !group.errors.is_empty() {
            // An error row carries neither `SIZE` nor `SUPPORTED`, so there's
            // no `(Refresh)` marker to hang off the end of it — but `refresh`
            // still greys it when it isn't behind, since the row is only
            // interesting to that command when re-downloading would replace
            // the file that failed to parse.
            let row = format!(
                "{nr:>nr_width$}  {:<model_width$}  error: {}",
                group.label,
                group.errors.join("; ")
            );
            if dim == Dimming::UpToDate && !refresh {
                out.push_str(&format!("{ANSI_DIM}{row}{ANSI_RESET}\n"));
            } else {
                out.push_str(&row);
                out.push('\n');
            }
            continue;
        }
        let refresh_suffix = if refresh { " (Refresh)" } else { "" };
        let used = last_used
            .and_then(|values| values.get(index).copied().flatten())
            .map(format_last_used)
            .unwrap_or_else(|| "Never".to_string());
        let row = if show_support && show_last_used {
            format!(
                "{nr:>nr_width$}  {:<model_width$}  {:<quant_width$}  {:<size_width$}  {used:<16}  {}{refresh_suffix}",
                group.label,
                group.quantization.as_deref().unwrap_or("-"),
                format_bytes(group.size_bytes),
                support[index].cell(),
            )
        } else if show_support {
            format!(
                "{nr:>nr_width$}  {:<model_width$}  {:<quant_width$}  {:<size_width$}  {}{refresh_suffix}",
                group.label,
                group.quantization.as_deref().unwrap_or("-"),
                format_bytes(group.size_bytes),
                support[index].cell(),
            )
        } else if show_last_used {
            format!(
                "{nr:>nr_width$}  {:<model_width$}  {:<quant_width$}  {:<size_width$}  {used}{refresh_suffix}",
                group.label,
                group.quantization.as_deref().unwrap_or("-"),
                format_bytes(group.size_bytes),
            )
        } else {
            format!(
                "{nr:>nr_width$}  {:<model_width$}  {:<quant_width$}  {}{refresh_suffix}",
                group.label,
                group.quantization.as_deref().unwrap_or("-"),
                format_bytes(group.size_bytes),
            )
        };
        let dimmed = match dim {
            Dimming::Off => false,
            Dimming::Unsupported => show_support && !support[index].loadable(),
            Dimming::UpToDate => !refresh,
        };
        if dimmed {
            out.push_str(&format!("{ANSI_DIM}{row}{ANSI_RESET}\n"));
        } else {
            out.push_str(&row);
            out.push('\n');
        }
    }
    out
}

fn format_last_used(timestamp: u64) -> String {
    use chrono::{DateTime, Local};

    DateTime::from_timestamp(timestamp as i64, 0)
        .map(|time| {
            time.with_timezone(&Local)
                .format("%Y-%m-%d %H:%M")
                .to_string()
        })
        .unwrap_or_else(|| "Never".to_string())
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

    /// A hub-cache layout: the blob under `blobs/<oid>`, named by a symlink
    /// under `snapshots/<rev>/`. The oid a row's local commit is read from is
    /// the symlink's target name, so every test built on this fixture — and
    /// the fixture itself — is unix-only.
    #[cfg(unix)]
    fn write_cached_minimal_gguf(snapshot_path: &Path, architecture: &str, oid: &str) {
        let repo_root = snapshot_path
            .ancestors()
            .find(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("models--"))
            })
            .unwrap();
        let blob = repo_root.join("blobs").join(oid);
        std::fs::create_dir_all(blob.parent().unwrap()).unwrap();
        std::fs::create_dir_all(snapshot_path.parent().unwrap()).unwrap();
        write_minimal_gguf(&blob, architecture, None);
        std::os::unix::fs::symlink(&blob, snapshot_path).unwrap();
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

    /// The web console's model manager names a row by its `NR` — a
    /// *position*, useless as a model id, and worse still as the `argv` the
    /// server is about to be re-executed with. `resolve_load_target` turns
    /// it back into the same `MODEL` label a `orangu-server <repo>` start
    /// would have produced for the same file.
    #[test]
    fn resolve_load_target_labels_an_nr_with_the_models_own_name() {
        let dir = tempfile::tempdir().unwrap();
        write_hub_cache_gguf(
            dir.path(),
            "bartowski/Llama-3.2-3B-Instruct-GGUF",
            "rev1",
            "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        );

        let (path, label) = resolve_load_target(dir.path(), "1").unwrap();

        assert_eq!(label, "bartowski/Llama-3.2-3B-Instruct-GGUF");
        assert!(
            path.ends_with("Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
            "{path:?}"
        );
        // And the label it produced resolves back to the same file — which
        // is what the re-executed process will have to do with it.
        assert_eq!(resolve_load_target(dir.path(), &label).unwrap().0, path);
    }

    /// An `NR` names one row exactly — including which quantization of a repo
    /// that has several on disk, where the bare `MODEL` label they all share
    /// would take whichever came first.
    #[test]
    fn resolve_load_target_picks_the_exact_row_an_nr_names() {
        let dir = tempfile::tempdir().unwrap();
        let repo = "bartowski/Llama-3.2-3B-Instruct-GGUF";
        write_hub_cache_gguf(dir.path(), repo, "rev1", "Llama-3.2-3B-Instruct-IQ3_M.gguf");
        write_hub_cache_gguf(
            dir.path(),
            repo,
            "rev1",
            "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        );

        let groups = group_models(&scan_models_dir(dir.path()).unwrap());
        assert_eq!(groups.len(), 2, "both quantizations should be listed");

        for (index, group) in groups.iter().enumerate() {
            let (path, _) = resolve_load_target(dir.path(), &(index + 1).to_string()).unwrap();
            assert_eq!(path, group.representative_path);
        }
    }

    /// Two quantizations of one repo print the same bare `MODEL`, so that
    /// label resolves to whichever comes first. A caller about to *restart
    /// the server* on the label it is handed back must not be given one that
    /// resolves to a different file than the row it asked for.
    #[test]
    fn resolve_load_target_disambiguates_a_label_two_rows_share() {
        let dir = tempfile::tempdir().unwrap();
        let repo = "bartowski/Llama-3.2-3B-Instruct-GGUF";
        write_hub_cache_gguf(dir.path(), repo, "rev1", "Llama-3.2-3B-Instruct-IQ3_M.gguf");
        write_hub_cache_gguf(
            dir.path(),
            repo,
            "rev1",
            "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        );

        // Every row's label must resolve back to that row's own file.
        for nr in ["1", "2"] {
            let (path, label) = resolve_load_target(dir.path(), nr).unwrap();
            assert!(
                label.contains(':'),
                "NR {nr} needs a quant to be unambiguous: {label}"
            );
            assert_eq!(
                resolve_load_target(dir.path(), &label).unwrap().0,
                path,
                "'{label}' must resolve back to the file NR {nr} named"
            );
        }

        // And the two rows must not collapse onto the same file.
        assert_ne!(
            resolve_load_target(dir.path(), "1").unwrap().0,
            resolve_load_target(dir.path(), "2").unwrap().0
        );
    }

    /// A spec naming nothing on disk is left exactly as written — it is a
    /// repo to fetch, and the fetch is what decides whether it exists.
    #[test]
    fn resolve_load_target_keeps_an_unknown_spec_verbatim() {
        let dir = tempfile::tempdir().unwrap();
        write_minimal_gguf(&dir.path().join("a.gguf"), "llama", None);

        // No network in a test, so this can only fail — but it must fail as a
        // *download* of the spec as typed, not by resolving it to the one
        // model that happens to be here.
        let err = resolve_load_target(dir.path(), "no/such-repo:Q4_K_M").unwrap_err();
        assert!(
            err.to_string().contains("no/such-repo:Q4_K_M"),
            "should have tried to fetch the spec as written: {err:#}"
        );
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

    /// The reported `QUANT` is what the file calls itself, not the ggml type
    /// its tensors mostly use: a mixed scheme stores part of the model at a
    /// heavier type by definition, so a real `Q4_K_M` model whose dominant
    /// type is `Q5_K` (type 13) must still read `Q4_K_M` — anything else
    /// contradicts the `MODEL` label beside it.
    #[test]
    fn quantization_prefers_the_name_the_file_carries_over_its_dominant_type() {
        let dir = tempfile::tempdir().unwrap();
        write_minimal_gguf(
            &dir.path().join("gemma-4-E2B-it-Q4_K_M.gguf"),
            "llama",
            Some((13, 4096)),
        );

        let groups = group_models(&scan_models_dir(dir.path()).unwrap());
        assert_eq!(groups[0].quantization.as_deref(), Some("Q4_K_M"));
    }

    /// One resolved file — what `orangu-server`'s startup banner reports —
    /// reads its quantization by the same two rules `list`'s `QUANT` column
    /// does: the name's tag when it carries one, the dominant ggml type
    /// otherwise. A shard suffix is stripped first, so any shard of a
    /// multi-part model answers with the model's own tag.
    #[test]
    fn quantization_for_file_reads_the_name_tag_then_the_dominant_type() {
        let dir = tempfile::tempdir().unwrap();
        let tagged = dir.path().join("gemma-4-E2B-it-Q4_K_M.gguf");
        write_minimal_gguf(&tagged, "gemma4", Some((13, 4096)));
        let sharded = dir.path().join("gemma-4-E2B-it-Q4_K_M-00002-of-00003.gguf");
        write_minimal_gguf(&sharded, "gemma4", Some((13, 4096)));
        let untagged = dir.path().join("gemma-4-E2B-it.gguf");
        write_minimal_gguf(&untagged, "gemma4", Some((13, 4096)));

        let quant = |path: &Path| {
            quantization_for_file(path, &GgufFile::open(path).unwrap()).unwrap_or_default()
        };
        assert_eq!(quant(&tagged), "Q4_K_M");
        assert_eq!(quant(&sharded), "Q4_K_M");
        assert_eq!(quant(&untagged), "Q5_K");
    }

    /// A lowercase tag counts too — plenty of published GGUFs spell it that
    /// way — and is shown uppercased, like the `:quant` label is.
    #[test]
    fn quantization_accepts_a_lowercase_name_tag() {
        let dir = tempfile::tempdir().unwrap();
        write_minimal_gguf(
            &dir.path().join("qwen2.5-0.5b-instruct-q4_k_m.gguf"),
            "qwen2",
            Some((13, 4096)),
        );

        let groups = group_models(&scan_models_dir(dir.path()).unwrap());
        assert_eq!(groups[0].quantization.as_deref(), Some("Q4_K_M"));
    }

    /// A name whose trailing token isn't a quantization at all (`-it` here)
    /// must not be shown as one: those fall back to the ggml type counted
    /// out of the tensors, which is all such a file says about itself.
    #[test]
    fn quantization_falls_back_to_the_dominant_type_without_a_name_tag() {
        let dir = tempfile::tempdir().unwrap();
        write_minimal_gguf(
            &dir.path().join("gemma-4-E2B-it.gguf"),
            "llama",
            Some((13, 4096)),
        );
        write_minimal_gguf(
            &dir.path().join("TinyLlama-1.1B.gguf"),
            "llama",
            Some((0, 8)),
        );

        let groups = group_models(&scan_models_dir(dir.path()).unwrap());
        let quants: Vec<_> = groups
            .iter()
            .map(|group| (group.label.as_str(), group.quantization.as_deref()))
            .collect();
        assert_eq!(
            quants,
            vec![
                ("TinyLlama-1.1B", Some("F32")),
                ("gemma-4-E2B-it", Some("Q5_K")),
            ]
        );
    }

    /// The tag shapes that count as a quantization, and the near-misses that
    /// don't — `hf_tag_from_label` still returns those to try as `-hf` tags,
    /// they just can't be printed as this model's quantization.
    #[test]
    fn quant_tag_from_label_accepts_only_quantization_shaped_tags() {
        for label in [
            "model-Q4_0",
            "model-Q6_K",
            "model-Q4_K_M",
            "model-UD-Q4_K_XL",
            "model-IQ2_XXS",
            "model-IQ4_NL",
            "model-TQ1_0",
            "model.F16",
            "model-BF16",
            "model-MXFP4",
            "model-q8_0",
        ] {
            assert!(
                quant_tag_from_label(label).is_some(),
                "should be a quantization: {label}"
            );
        }
        for label in [
            "gemma-4-E2B-it",
            "TinyLlama-1.1B",
            "model-Instruct",
            "Meta-Llama-3-8B",
            "model",
        ] {
            assert_eq!(
                quant_tag_from_label(label),
                None,
                "should not be a quantization: {label}"
            );
        }
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

    /// Writes `file` into a hub-cache layout for `repo` under `dir`, the way
    /// `-hf` itself lays a download out: `models--<user>--<model>/snapshots/
    /// <rev>/<file>`.
    fn write_hub_cache_gguf(dir: &Path, repo: &str, rev: &str, file: &str) {
        let repo_dir = dir
            .join(format!("models--{}", repo.replace('/', "--")))
            .join("snapshots")
            .join(rev);
        std::fs::create_dir_all(&repo_dir).unwrap();
        write_minimal_gguf(&repo_dir.join(file), "llama", None);
    }

    /// `MODEL` drops a `:TAG` that only repeats the `QUANT` column — but the
    /// tagged spelling it used to print (and that saved `model =` values still
    /// carry) has to keep resolving, or a config silently turns into a
    /// re-download.
    #[test]
    fn group_models_drops_a_quant_tag_the_quant_column_already_shows() {
        let dir = tempfile::tempdir().unwrap();
        write_hub_cache_gguf(
            dir.path(),
            "bartowski/Llama-3.2-3B-Instruct-GGUF",
            "rev1",
            "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        );

        let groups = group_models(&scan_models_dir(dir.path()).unwrap());

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].label, "bartowski/Llama-3.2-3B-Instruct-GGUF");
        assert_eq!(groups[0].quantization.as_deref(), Some("Q4_K_M"));
        assert_eq!(
            groups[0].hf_repo.as_deref(),
            Some("bartowski/Llama-3.2-3B-Instruct-GGUF")
        );
        assert_eq!(groups[0].local_commit.as_deref(), Some("rev1"));
        assert!(groups[0].matches_label("bartowski/Llama-3.2-3B-Instruct-GGUF"));
        assert!(groups[0].matches_label("bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M"));
        assert!(!groups[0].matches_label("bartowski/Llama-3.2-3B-Instruct-GGUF:Q8_0"));
    }

    /// Two quantizations of one repo share a `MODEL` cell — `QUANT` is what
    /// tells them apart — but each still resolves individually by its tagged
    /// spelling, which is the only way (besides `NR`) to name one of them.
    #[test]
    fn two_quants_of_one_repo_share_a_label_and_differ_by_quant() {
        let dir = tempfile::tempdir().unwrap();
        for file in [
            "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            "Llama-3.2-3B-Instruct-Q8_0.gguf",
        ] {
            write_hub_cache_gguf(
                dir.path(),
                "bartowski/Llama-3.2-3B-Instruct-GGUF",
                "rev1",
                file,
            );
        }

        let groups = group_models(&scan_models_dir(dir.path()).unwrap());

        assert_eq!(
            groups
                .iter()
                .map(|group| (group.label.as_str(), group.quantization.as_deref()))
                .collect::<Vec<_>>(),
            vec![
                ("bartowski/Llama-3.2-3B-Instruct-GGUF", Some("Q4_K_M")),
                ("bartowski/Llama-3.2-3B-Instruct-GGUF", Some("Q8_0")),
            ]
        );

        let q4 =
            resolve_show_target(dir.path(), "bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M").unwrap();
        let q8 =
            resolve_show_target(dir.path(), "bartowski/Llama-3.2-3B-Instruct-GGUF:Q8_0").unwrap();
        assert!(q4.ends_with("Llama-3.2-3B-Instruct-Q4_K_M.gguf"));
        assert!(q8.ends_with("Llama-3.2-3B-Instruct-Q8_0.gguf"));

        // A bare request is genuinely ambiguous: it takes the first row, the
        // same one `NR` 1 names, deterministically between runs.
        let bare = resolve_show_target(dir.path(), "bartowski/Llama-3.2-3B-Instruct-GGUF").unwrap();
        assert_eq!(bare, q4);
        assert_eq!(resolve_show_target(dir.path(), "1").unwrap(), q4);

        // `delete` resolves the same way, and returns the whole group so the
        // confirmation names that one quantization's files only.
        let target =
            resolve_delete_target(dir.path(), "bartowski/Llama-3.2-3B-Instruct-GGUF:Q8_0").unwrap();
        assert_eq!(target.quantization.as_deref(), Some("Q8_0"));
        assert_eq!(target.paths, vec![q8]);
    }

    /// The stripped `MODEL` string and the tagged one both resolve through
    /// `show`'s own resolver — to the same file.
    #[test]
    fn resolve_show_target_accepts_a_model_label_with_or_without_its_quant_tag() {
        let dir = tempfile::tempdir().unwrap();
        write_hub_cache_gguf(
            dir.path(),
            "bartowski/Llama-3.2-3B-Instruct-GGUF",
            "rev1",
            "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        );

        let stripped =
            resolve_show_target(dir.path(), "bartowski/Llama-3.2-3B-Instruct-GGUF").unwrap();
        let tagged =
            resolve_show_target(dir.path(), "bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M").unwrap();
        assert_eq!(stripped, tagged);
        assert!(stripped.ends_with("Llama-3.2-3B-Instruct-Q4_K_M.gguf"));
    }

    #[test]
    fn group_models_leaves_hf_repo_and_local_commit_none_outside_a_hub_cache() {
        let dir = tempfile::tempdir().unwrap();
        write_minimal_gguf(&dir.path().join("plain.gguf"), "llama", None);

        let models = scan_models_dir(dir.path()).unwrap();
        let groups = group_models(&models);

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].hf_repo, None);
        assert_eq!(groups[0].local_commit, None);
    }

    #[test]
    fn last_used_column_shows_never_until_a_model_has_been_loaded() {
        let dir = tempfile::tempdir().unwrap();
        write_minimal_gguf(&dir.path().join("plain.gguf"), "llama", None);
        let groups = group_models(&scan_models_dir(dir.path()).unwrap());

        let output = format_groups_with_last_used(
            &groups,
            dir.path(),
            &HashMap::new(),
            &[],
            Dimming::Off,
            Some(&[None]),
        );

        assert!(output.lines().next().unwrap().contains("LAST_USED"));
        assert!(output.lines().nth(1).unwrap().ends_with("Never"));

        let support = [ModelSupport {
            architecture: Some("llama".to_string()),
            supported: true,
            unsupported_quant: None,
        }];
        let output = format_groups_with_last_used(
            &groups,
            dir.path(),
            &HashMap::new(),
            &support,
            Dimming::Off,
            Some(&[None]),
        );
        let header = output.lines().next().unwrap();
        assert!(header.find("LAST_USED") < header.find("SUPPORTED"));
        let row = output.lines().nth(1).unwrap();
        assert!(row.contains("Never"));
        assert!(row.ends_with("Yes (llama)"));
    }

    #[test]
    fn explicit_display_order_keeps_original_numbers_and_metadata_alignment() {
        let dir = tempfile::tempdir().unwrap();
        write_minimal_gguf(&dir.path().join("a.gguf"), "llama", None);
        write_minimal_gguf(&dir.path().join("b.gguf"), "llama", None);
        let groups = group_models(&scan_models_dir(dir.path()).unwrap());

        let output = format_groups_with_last_used_in_order(
            &groups,
            dir.path(),
            &HashMap::new(),
            &[],
            Dimming::Off,
            Some(&[None, Some(0)]),
            &[1, 0],
        );
        let rows: Vec<&str> = output.lines().skip(1).collect();

        assert!(rows[0].trim_start().starts_with("2  b "), "{}", rows[0]);
        assert!(!rows[0].ends_with("Never"), "{}", rows[0]);
        assert!(rows[1].trim_start().starts_with("1  a "), "{}", rows[1]);
        assert!(rows[1].ends_with("Never"), "{}", rows[1]);
    }

    #[cfg(unix)]
    #[test]
    fn format_groups_marks_a_row_whose_local_commit_is_behind() {
        let dir = tempfile::tempdir().unwrap();
        let snapshot = dir
            .path()
            .join("models--bartowski--Llama-3.2-3B-Instruct-GGUF/snapshots/rev1");
        write_cached_minimal_gguf(
            &snapshot.join("Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
            "llama",
            "blob-1",
        );
        write_minimal_gguf(&dir.path().join("plain.gguf"), "llama", None);

        let models = scan_models_dir(dir.path()).unwrap();
        let groups = group_models(&models);
        let mut latest_updates = HashMap::new();
        latest_updates.insert(
            "bartowski/Llama-3.2-3B-Instruct-GGUF".to_string(),
            crate::model_download::RepoUpdateInfo {
                commit: "rev2".to_string(),
                files: vec![crate::model_download::RepoFile {
                    path: "Llama-3.2-3B-Instruct-Q4_K_M.gguf".to_string(),
                    oid: "blob-2".to_string(),
                    size: 1,
                }],
            },
        );

        let output = format_groups(&groups, dir.path(), &latest_updates, &[], Dimming::Off);

        let mut lines = output.lines().skip(1); // header
        assert!(lines.next().unwrap().ends_with("(Refresh)"));
        assert!(!lines.next().unwrap().contains("(Refresh)"));
    }

    #[cfg(unix)]
    #[test]
    fn format_groups_does_not_mark_a_row_already_at_the_latest_commit() {
        let dir = tempfile::tempdir().unwrap();
        let snapshot = dir
            .path()
            .join("models--bartowski--Llama-3.2-3B-Instruct-GGUF/snapshots/rev1");
        write_cached_minimal_gguf(
            &snapshot.join("Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
            "llama",
            "blob-1",
        );

        let models = scan_models_dir(dir.path()).unwrap();
        let groups = group_models(&models);
        let mut latest_updates = HashMap::new();
        latest_updates.insert(
            "bartowski/Llama-3.2-3B-Instruct-GGUF".to_string(),
            crate::model_download::RepoUpdateInfo {
                commit: "rev1".to_string(),
                files: vec![crate::model_download::RepoFile {
                    path: "Llama-3.2-3B-Instruct-Q4_K_M.gguf".to_string(),
                    oid: "blob-1".to_string(),
                    size: 1,
                }],
            },
        );

        let output = format_groups(&groups, dir.path(), &latest_updates, &[], Dimming::Off);

        assert!(!output.lines().nth(1).unwrap().contains("(Refresh)"));
    }

    #[cfg(unix)]
    #[test]
    fn format_groups_does_not_mark_a_row_when_only_the_repo_commit_changed() {
        let dir = tempfile::tempdir().unwrap();
        let snapshot = dir
            .path()
            .join("models--bartowski--Llama-3.2-3B-Instruct-GGUF/snapshots/rev1");
        write_cached_minimal_gguf(
            &snapshot.join("Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
            "llama",
            "blob-1",
        );

        let models = scan_models_dir(dir.path()).unwrap();
        let groups = group_models(&models);
        let mut latest_updates = HashMap::new();
        latest_updates.insert(
            "bartowski/Llama-3.2-3B-Instruct-GGUF".to_string(),
            crate::model_download::RepoUpdateInfo {
                commit: "rev2".to_string(),
                files: vec![crate::model_download::RepoFile {
                    path: "Llama-3.2-3B-Instruct-Q4_K_M.gguf".to_string(),
                    oid: "blob-1".to_string(),
                    size: 1,
                }],
            },
        );

        let output = format_groups(&groups, dir.path(), &latest_updates, &[], Dimming::Off);
        assert!(
            !output.lines().nth(1).unwrap().contains("(Refresh)"),
            "{output}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn format_groups_only_marks_the_row_actually_behind_when_a_repo_has_two_local_commits() {
        // Two `:quant` rows of the same repo, cached at different commits —
        // the exact scenario `check_for_updates`/`latest_commits` dedupes by
        // repo id for (one Hub lookup covers both rows), so this pins that a
        // stale sibling row doesn't also mark an already-current one.
        let dir = tempfile::tempdir().unwrap();
        let old_dir = dir
            .path()
            .join("models--bartowski--Llama-3.2-3B-Instruct-GGUF/snapshots/rev1");
        write_cached_minimal_gguf(
            &old_dir.join("Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
            "llama",
            "blob-1",
        );
        let current_dir = dir
            .path()
            .join("models--bartowski--Llama-3.2-3B-Instruct-GGUF/snapshots/rev2");
        write_cached_minimal_gguf(
            &current_dir.join("Llama-3.2-3B-Instruct-Q8_0.gguf"),
            "llama",
            "blob-2",
        );

        let models = scan_models_dir(dir.path()).unwrap();
        let groups = group_models(&models);
        let mut latest_updates = HashMap::new();
        latest_updates.insert(
            "bartowski/Llama-3.2-3B-Instruct-GGUF".to_string(),
            crate::model_download::RepoUpdateInfo {
                commit: "rev2".to_string(),
                files: vec![
                    crate::model_download::RepoFile {
                        path: "Llama-3.2-3B-Instruct-Q4_K_M.gguf".to_string(),
                        oid: "blob-9".to_string(),
                        size: 1,
                    },
                    crate::model_download::RepoFile {
                        path: "Llama-3.2-3B-Instruct-Q8_0.gguf".to_string(),
                        oid: "blob-2".to_string(),
                        size: 1,
                    },
                ],
            },
        );

        let output = format_groups(&groups, dir.path(), &latest_updates, &[], Dimming::Off);

        let mut lines = output.lines().skip(1); // header
        let q4 = lines.next().unwrap(); // Q4_K_M, sorted before Q8_0
        let q8 = lines.next().unwrap();
        assert!(q4.contains("Q4_K_M"));
        assert!(q4.ends_with("(Refresh)"));
        assert!(q8.contains("Q8_0"));
        assert!(!q8.contains("(Refresh)"));
    }

    /// The marker is one space off the column before it, like every other
    /// column separator on the row — not the two it was first written with.
    #[cfg(unix)]
    #[test]
    fn the_refresh_marker_is_one_space_off_the_preceding_column() {
        let dir = tempfile::tempdir().unwrap();
        let snapshot = dir
            .path()
            .join("models--bartowski--Llama-3.2-3B-Instruct-GGUF/snapshots/rev1");
        write_cached_minimal_gguf(
            &snapshot.join("Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
            "llama",
            "blob-1",
        );

        let models = scan_models_dir(dir.path()).unwrap();
        let groups = group_models(&models);
        let mut latest_updates = HashMap::new();
        latest_updates.insert(
            "bartowski/Llama-3.2-3B-Instruct-GGUF".to_string(),
            crate::model_download::RepoUpdateInfo {
                commit: "rev2".to_string(),
                files: vec![crate::model_download::RepoFile {
                    path: "Llama-3.2-3B-Instruct-Q4_K_M.gguf".to_string(),
                    oid: "blob-2".to_string(),
                    size: 1,
                }],
            },
        );

        let output = format_groups(&groups, dir.path(), &latest_updates, &[], Dimming::Off);

        let row = output.lines().nth(1).unwrap();
        assert!(row.ends_with(" (Refresh)"), "unexpected row: {row:?}");
        assert!(!row.ends_with("  (Refresh)"), "unexpected row: {row:?}");
    }

    /// `refresh`'s table: the rows worth acting on stay plain, everything
    /// already current is greyed — the inverse of what `list` greys.
    #[cfg(unix)]
    #[test]
    fn up_to_date_dimming_greys_every_row_that_is_not_behind() {
        let dir = tempfile::tempdir().unwrap();
        let stale = dir
            .path()
            .join("models--bartowski--Llama-3.2-3B-Instruct-GGUF/snapshots/rev1");
        write_cached_minimal_gguf(
            &stale.join("Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
            "llama",
            "blob-1",
        );
        let current = dir
            .path()
            .join("models--ggml-org--gemma-4-12B-it-GGUF/snapshots/rev9");
        write_cached_minimal_gguf(
            &current.join("gemma-4-12B-it-Q4_K_M.gguf"),
            "gemma4",
            "blob-9",
        );

        let models = scan_models_dir(dir.path()).unwrap();
        let groups = group_models(&models);
        let latest_updates = HashMap::from([
            (
                "bartowski/Llama-3.2-3B-Instruct-GGUF".to_string(),
                crate::model_download::RepoUpdateInfo {
                    commit: "rev2".to_string(),
                    files: vec![crate::model_download::RepoFile {
                        path: "Llama-3.2-3B-Instruct-Q4_K_M.gguf".to_string(),
                        oid: "blob-2".to_string(),
                        size: 1,
                    }],
                },
            ),
            (
                "ggml-org/gemma-4-12B-it-GGUF".to_string(),
                crate::model_download::RepoUpdateInfo {
                    commit: "rev9".to_string(),
                    files: vec![crate::model_download::RepoFile {
                        path: "gemma-4-12B-it-Q4_K_M.gguf".to_string(),
                        oid: "blob-9".to_string(),
                        size: 1,
                    }],
                },
            ),
        ]);

        let output = format_groups(&groups, dir.path(), &latest_updates, &[], Dimming::UpToDate);

        let mut lines = output.lines().skip(1); // header
        let stale_row = lines.next().unwrap(); // bartowski, sorted first
        let current_row = lines.next().unwrap();
        assert!(stale_row.contains("(Refresh)"));
        assert!(!stale_row.contains(ANSI_DIM), "stale row was greyed");
        assert!(!current_row.contains("(Refresh)"));
        assert!(current_row.starts_with(ANSI_DIM), "current row was plain");
    }

    /// Two quantizations of one repo share a `MODEL` cell. `delete` takes the
    /// first; `refresh` must not, since it deletes what it then re-downloads.
    #[test]
    fn refresh_rejects_a_model_name_naming_more_than_one_quantization() {
        let dir = tempfile::tempdir().unwrap();
        let snapshot = dir
            .path()
            .join("models--bartowski--Llama-3.2-3B-Instruct-GGUF/snapshots/rev1");
        std::fs::create_dir_all(&snapshot).unwrap();
        write_minimal_gguf(
            &snapshot.join("Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
            "llama",
            None,
        );
        write_minimal_gguf(
            &snapshot.join("Llama-3.2-3B-Instruct-Q8_0.gguf"),
            "llama",
            None,
        );

        let err = resolve_refresh_target(dir.path(), "bartowski/Llama-3.2-3B-Instruct-GGUF")
            .expect_err("ambiguous");
        let message = err.to_string();
        assert!(message.contains("Q4_K_M"), "unexpected error: {message}");
        assert!(message.contains("Q8_0"), "unexpected error: {message}");
        assert!(
            message.contains("bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M"),
            "unexpected error: {message}"
        );

        // The tagged spelling names exactly one of them, and resolves.
        let group = resolve_refresh_target(dir.path(), "bartowski/Llama-3.2-3B-Instruct-GGUF:Q8_0")
            .unwrap();
        assert_eq!(group.quantization.as_deref(), Some("Q8_0"));
        // As does the `NR` `list` printed beside either row.
        assert_eq!(
            resolve_refresh_target(dir.path(), "1")
                .unwrap()
                .quantization
                .as_deref(),
            Some("Q4_K_M")
        );
    }

    /// A companion sidecar is in no group: `delete` synthesizes a one-file
    /// group for it, but there's no such thing as downloading one on its own.
    #[test]
    fn refresh_rejects_an_mmproj_sidecar_named_directly() {
        let dir = tempfile::tempdir().unwrap();
        let snapshot = dir
            .path()
            .join("models--ggml-org--gemma-4-12B-it-GGUF/snapshots/rev1");
        std::fs::create_dir_all(&snapshot).unwrap();
        write_minimal_gguf(&snapshot.join("gemma-4-12B-it-Q4_K_M.gguf"), "gemma4", None);
        // `clip` is what marks it a projector — see
        // `excludes_clip_projector_sidecars_from_the_scan`.
        let sidecar = snapshot.join("mmproj-gemma-4-12B-it-F16.gguf");
        write_minimal_gguf(&sidecar, "clip", None);

        let err = resolve_refresh_target(dir.path(), sidecar.to_str().unwrap())
            .expect_err("companion file");
        assert!(
            err.to_string().contains("companion file"),
            "unexpected error: {err:#}"
        );
    }

    /// The re-download spec carries the tag the file's *name* has, which is
    /// what `download` matches against the repo listing — never the `QUANT`
    /// column's ggml-type fallback, which names no file in the repo.
    #[test]
    fn the_download_spec_uses_the_filename_tag_not_the_quant_fallback() {
        let dir = tempfile::tempdir().unwrap();
        let snapshot = dir
            .path()
            .join("models--unsloth--gemma-4-E2B-it-GGUF/snapshots/rev1");
        std::fs::create_dir_all(&snapshot).unwrap();
        write_minimal_gguf(&snapshot.join("gemma-4-E2B-it-Q4_K_M.gguf"), "gemma4", None);
        // No quantization tag in the name at all: `QUANT` shows whatever the
        // tensors say (nothing here), while the spec keeps the `-hf` tag the
        // name does carry — the one that picked this file the first time.
        write_minimal_gguf(&snapshot.join("gemma-4-E2B-it.gguf"), "gemma4", None);

        let models = scan_models_dir(dir.path()).unwrap();
        let groups = group_models(&models);

        let tagged = groups
            .iter()
            .find(|g| g.quantization.as_deref() == Some("Q4_K_M"))
            .expect("tagged row");
        assert_eq!(
            tagged.download_spec().as_deref(),
            Some("unsloth/gemma-4-E2B-it-GGUF:Q4_K_M")
        );

        let untagged = groups
            .iter()
            .find(|g| g.representative_path.ends_with("gemma-4-E2B-it.gguf"))
            .expect("untagged row");
        assert_eq!(
            untagged.download_spec().as_deref(),
            Some("unsloth/gemma-4-E2B-it-GGUF:IT")
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
