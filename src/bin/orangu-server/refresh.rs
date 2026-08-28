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

//! `orangu-server refresh`: re-downloads a model whose Hugging Face repo has
//! moved on since it was fetched — `delete` followed by `download` of the
//! same spec, as one step, so a `(Refresh)` marker in `list` has a command
//! that acts on it.
//!
//! Nothing is deleted until the Hub has confirmed that the local files are
//! actually behind: the repo's file hashes are compared against the ones on
//! disk (the same comparison `list` marks `(Refresh)` from), and a model
//! already at the latest revision is a no-op. Offline — or with the repo
//! unreachable for any other reason — refresh does nothing at all rather
//! than guessing, since deleting a model it then cannot download again
//! would destroy it.
//!
//! When a refresh does go ahead, the local copy goes first and the download
//! follows, rather than the other way round: the point of a refresh is that
//! the repo's files have changed, so the new revision is a full second copy
//! on disk, not a cheap blob-sharing snapshot. Deleting first means a 17 GiB
//! model needs 17 GiB of free space to refresh, not 34. The cost is that an
//! interrupted download leaves the model missing rather than stale, and
//! `download`'s own resume (a `.part` file picked up on the next run) is
//! what recovers it.
//!
//! With `--all`, every model known to be behind is refreshed in one run.
//! With no argument it prints `list`'s table with every already-current row
//! greyed ([`orangu::model_spec::Dimming::UpToDate`]) and prompts for an
//! `NR`, so the rows worth refreshing are the only ones standing out. When
//! every reachable repo is already current, it exits without opening the
//! picker.

use crate::{check_for_updates, dimming, model_support};
use anyhow::{Context, Result, anyhow, bail};
use orangu::{
    model_download::RepoUpdateInfo,
    model_spec::{Dimming, ModelGroup},
};
use std::{collections::HashMap, io::Write, path::Path};

/// What `refresh` should do with a model, once the Hub has been asked about
/// its repo. Decided before anything is deleted — see [`plan`].
#[derive(Debug, PartialEq, Eq)]
enum Plan {
    /// The repo's files differ from the local ones: delete and download.
    Refresh,
    /// The local files are already the repo's latest: do nothing.
    UpToDate,
    /// The repo could not be reached, so "behind" is unknown: do nothing.
    Unreachable,
}

/// Whether `group` is worth deleting and downloading again, given whatever
/// [`check_for_updates`] came back with.
///
/// A repo missing from `updates` is one the lookup didn't reach — offline,
/// a timeout, a gated repo without `HF_TOKEN`. That is deliberately *not*
/// folded into [`Plan::UpToDate`]: `is_behind` answers `false` for both "the
/// hashes match" and "there is nothing to compare against", and only the
/// first of those is a reason to tell the user their model is current.
fn plan(group: &ModelGroup, updates: &HashMap<String, RepoUpdateInfo>) -> Plan {
    match group.hf_repo.as_deref() {
        Some(repo) if updates.contains_key(repo) => {
            if group.is_behind(updates) {
                Plan::Refresh
            } else {
                Plan::UpToDate
            }
        }
        _ => Plan::Unreachable,
    }
}

/// The subset `refresh --all` can prove needs work. Repositories that could
/// not be reached are deliberately absent: unknown is not stale.
fn refresh_targets<'a>(
    groups: &'a [ModelGroup],
    updates: &HashMap<String, RepoUpdateInfo>,
) -> Vec<&'a ModelGroup> {
    groups
        .iter()
        .filter(|group| plan(group, updates) == Plan::Refresh)
        .collect()
}

/// Deletes one model and downloads it again, when the Hub says its files
/// have changed. `model` resolves the same way `delete`'s argument does,
/// except that a `MODEL` name matching more than one row is an error rather
/// than a first-match (see [`orangu::model_spec::resolve_refresh_target`]);
/// omitting it picks one interactively. `all` refreshes every model that the
/// same Hub check says is behind.
pub fn run(models_dir: &Path, model: Option<String>, all: bool, _yes: bool) -> Result<()> {
    if all {
        debug_assert!(model.is_none(), "clap makes MODEL conflict with --all");
        return refresh_all(models_dir);
    }

    // The interactive path has already asked the Hub about every repo on
    // disk, so it hands its answer over rather than making the same lookup
    // twice; an explicit argument asks about that one repo only.
    let (group, known_updates) = match model {
        Some(spec) => (
            orangu::model_spec::resolve_refresh_target(models_dir, &spec)?,
            None,
        ),
        None => match select_model_to_refresh(models_dir)? {
            Some(picked) => (picked.0, Some(picked.1)),
            None => return Ok(()),
        },
    };

    refresh_group(
        models_dir,
        &group,
        &known_updates.unwrap_or_else(|| check_for_updates(std::slice::from_ref(&group))),
    )?;
    Ok(())
}

/// Refreshes every model whose remote file hashes differ from its local
/// blobs. The Hub is queried once for all distinct repositories, exactly as
/// for `list`; a repository that could not be checked is reported and left
/// untouched.
fn refresh_all(models_dir: &Path) -> Result<()> {
    let models = orangu::model_spec::scan_models_dir(models_dir)
        .with_context(|| format!("scanning {}", models_dir.display()))?;
    let groups = orangu::model_spec::group_models(&models);
    if groups.is_empty() {
        bail!("no .gguf models found under {}", models_dir.display());
    }

    let updates = check_for_updates(&groups);
    let targets = refresh_targets(&groups, &updates);
    let unreachable = groups
        .iter()
        .filter(|group| {
            group
                .hf_repo
                .as_ref()
                .is_some_and(|repo| !updates.contains_key(repo))
        })
        .count();

    if targets.is_empty() {
        if groups.iter().all(|group| group.hf_repo.is_none()) {
            println!(
                "No model under {} was downloaded from Hugging Face, so there is nothing to refresh.",
                models_dir.display()
            );
        } else if unreachable > 0 {
            println!(
                "No reachable model needs refreshing; {unreachable} model(s) could not be checked and nothing was changed."
            );
        } else {
            println!("Every model is already at its repo's latest revision; nothing to do.");
        }
        return Ok(());
    }

    println!("Refreshing {} model(s).", targets.len());
    if unreachable > 0 {
        println!("Skipping {unreachable} model(s) whose repositories could not be reached.");
    }
    for group in &targets {
        refresh_group(models_dir, group, &updates)
            .with_context(|| format!("refreshing '{}'", group.label))?;
    }
    println!("Refreshed {} model(s).", targets.len());
    Ok(())
}

/// Applies a previously computed plan to one group.
fn refresh_group(
    models_dir: &Path,
    group: &ModelGroup,
    updates: &HashMap<String, RepoUpdateInfo>,
) -> Result<()> {
    // A model outside the Hugging Face hub-cache layout — a `.gguf` copied
    // in by hand, say — has no repo to download it again from, so there's
    // nothing to refresh *to*. Bail before deleting anything: this is the
    // one case where going ahead would destroy a model with no way back.
    let spec = group.download_spec().ok_or_else(|| {
        anyhow!(
            "'{}' was not downloaded from Hugging Face (no models--<user>--<model> cache directory), so there is no repo to refresh it from",
            group.label
        )
    })?;

    match plan(group, updates) {
        Plan::Refresh => {}
        Plan::UpToDate => {
            println!(
                "'{}' is already at its repo's latest revision; nothing to do.",
                group.label
            );
            return Ok(());
        }
        Plan::Unreachable => {
            println!(
                "Could not reach Hugging Face for '{spec}', so there is no way to tell whether '{}' is behind its repo; nothing was changed.",
                group.label
            );
            return Ok(());
        }
    }

    let plural = if group.paths.len() == 1 { "" } else { "s" };
    orangu::model_spec::delete_model(models_dir, group)?;
    println!(
        "Deleted '{}' ({} file{plural}, {})",
        group.label,
        group.paths.len(),
        orangu::format::format_bytes(group.size_bytes),
    );

    let path = orangu::model_download::download_model(models_dir, &spec)
        .with_context(|| format!("'{spec}' was deleted but could not be downloaded again"))?;
    println!("Downloaded to {}", path.display());
    Ok(())
}

/// Prints `list`'s table — greying every row that is *not* behind its repo,
/// the inverse of what `list` itself greys — and prompts for an `NR`, for
/// `refresh` invoked with no model argument. The Hub lookup is the same one
/// `list` does, so the un-greyed rows are exactly the `(Refresh)` ones, and
/// it is returned alongside the chosen group so [`run`] can decide on it
/// without asking again.
///
/// Returns `None` — with a line saying why — when there is nothing to pick:
/// no repo could be reached (offline, so no row is *known* to be behind and
/// a pick could only end in a no-op), or every reachable repo is current.
fn select_model_to_refresh(
    models_dir: &Path,
) -> Result<Option<(ModelGroup, HashMap<String, RepoUpdateInfo>)>> {
    let models = orangu::model_spec::scan_models_dir(models_dir)
        .with_context(|| format!("scanning {}", models_dir.display()))?;
    let mut groups = orangu::model_spec::group_models(&models);
    if groups.is_empty() {
        bail!("no .gguf models found under {}", models_dir.display());
    }
    let latest_commits = check_for_updates(&groups);
    if latest_commits.is_empty() {
        if groups.iter().any(|group| group.hf_repo.is_some()) {
            println!(
                "Could not reach Hugging Face, so there is no way to tell which models are behind their repos; nothing was changed."
            );
        } else {
            println!(
                "No model under {} was downloaded from Hugging Face, so there is nothing to refresh.",
                models_dir.display()
            );
        }
        return Ok(None);
    }
    if !groups.iter().any(|group| group.is_behind(&latest_commits)) {
        println!("Every model is already at its repo's latest revision; nothing to do.");
        return Ok(None);
    }

    print!(
        "{}",
        orangu::model_spec::format_groups(
            &groups,
            models_dir,
            &latest_commits,
            &model_support(&groups),
            dimming(Dimming::UpToDate),
        )
    );
    print!("\nSelect a model to refresh (NR): ");
    std::io::stdout().flush().ok();
    let mut input = String::new();
    std::io::stdin()
        .read_line(&mut input)
        .context("failed to read model selection")?;
    let nr: usize = input
        .trim()
        .parse()
        .with_context(|| format!("'{}' is not a number", input.trim()))?;
    let count = groups.len();
    nr.checked_sub(1)
        .filter(|index| *index < count)
        .map(|index| Some((groups.swap_remove(index), latest_commits)))
        .ok_or_else(|| anyhow!("no model with NR {nr} ({count} model(s) listed)"))
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use orangu::model_download::RepoFile;
    use std::path::PathBuf;

    /// A hub-cache `snapshots/rev1/<name>` symlink into `blobs/<oid>` — the
    /// layout `is_behind` reads the local hash back out of. The file content
    /// is irrelevant here: the group is built by hand rather than scanned,
    /// so nothing parses it as GGUF.
    fn cached(root: &Path, repo: &str, name: &str, oid: &str) -> PathBuf {
        let repo_root = root.join(format!("models--{}", repo.replace('/', "--")));
        let blob = repo_root.join("blobs").join(oid);
        let snapshot = repo_root.join("snapshots/rev1").join(name);
        std::fs::create_dir_all(blob.parent().unwrap()).unwrap();
        std::fs::create_dir_all(snapshot.parent().unwrap()).unwrap();
        std::fs::write(&blob, b"gguf").unwrap();
        std::os::unix::fs::symlink(&blob, &snapshot).unwrap();
        snapshot
    }

    fn group(repo: &str, path: PathBuf) -> ModelGroup {
        ModelGroup {
            label: repo.to_string(),
            size_bytes: 4,
            quantization: Some("Q4_K_M".to_string()),
            errors: Vec::new(),
            representative_path: path.clone(),
            paths: vec![path],
            hf_repo: Some(repo.to_string()),
            local_commit: Some("rev1".to_string()),
        }
    }

    fn updates(repo: &str, name: &str, oid: &str) -> HashMap<String, RepoUpdateInfo> {
        HashMap::from([(
            repo.to_string(),
            RepoUpdateInfo {
                commit: "rev2".to_string(),
                files: vec![RepoFile {
                    path: name.to_string(),
                    oid: oid.to_string(),
                    size: 4,
                }],
            },
        )])
    }

    const REPO: &str = "bartowski/Llama-3.2-3B-Instruct-GGUF";
    const FILE: &str = "Llama-3.2-3B-Instruct-Q4_K_M.gguf";

    #[test]
    fn a_model_whose_blob_hash_matches_the_repo_is_a_no_op() {
        let dir = tempfile::tempdir().unwrap();
        let group = group(REPO, cached(dir.path(), REPO, FILE, "blob-1"));

        assert_eq!(plan(&group, &updates(REPO, FILE, "blob-1")), Plan::UpToDate);
    }

    #[test]
    fn a_model_whose_blob_hash_differs_is_refreshed() {
        let dir = tempfile::tempdir().unwrap();
        let group = group(REPO, cached(dir.path(), REPO, FILE, "blob-1"));

        assert_eq!(plan(&group, &updates(REPO, FILE, "blob-2")), Plan::Refresh);
    }

    #[test]
    fn an_unreachable_hub_refreshes_nothing() {
        // The distinction that keeps an offline run from deleting anything:
        // `is_behind` says `false` here exactly as it does for a current
        // model, so a plan built on that alone would report "up to date"
        // about a repo it never managed to ask.
        let dir = tempfile::tempdir().unwrap();
        let group = group(REPO, cached(dir.path(), REPO, FILE, "blob-1"));

        assert_eq!(plan(&group, &HashMap::new()), Plan::Unreachable);
        assert!(!group.is_behind(&HashMap::new()));
    }

    #[test]
    fn one_reachable_repo_does_not_speak_for_another() {
        // The interactive path passes the whole map, so a group whose own
        // repo timed out must not be read as current just because a
        // different repo's lookup succeeded.
        let dir = tempfile::tempdir().unwrap();
        let group = group(REPO, cached(dir.path(), REPO, FILE, "blob-1"));

        let other = updates("unsloth/Qwen3-Coder-Next-GGUF", FILE, "blob-1");
        assert_eq!(plan(&group, &other), Plan::Unreachable);
    }

    #[test]
    fn refresh_all_selects_only_models_proven_stale() {
        let dir = tempfile::tempdir().unwrap();
        let stale_repo = "owner/stale-GGUF";
        let current_repo = "owner/current-GGUF";
        let unreachable_repo = "owner/unreachable-GGUF";
        let groups = vec![
            group(stale_repo, cached(dir.path(), stale_repo, FILE, "old-blob")),
            group(
                current_repo,
                cached(dir.path(), current_repo, FILE, "current-blob"),
            ),
            group(
                unreachable_repo,
                cached(dir.path(), unreachable_repo, FILE, "unknown-blob"),
            ),
        ];
        let mut known = updates(stale_repo, FILE, "new-blob");
        known.extend(updates(current_repo, FILE, "current-blob"));

        let targets = refresh_targets(&groups, &known);

        assert_eq!(
            targets
                .iter()
                .map(|group| group.label.as_str())
                .collect::<Vec<_>>(),
            vec![stale_repo]
        );
    }

    #[test]
    fn a_model_outside_the_hub_cache_is_never_planned_for_refresh() {
        let mut group = group(REPO, PathBuf::from("/models/hand-copied.gguf"));
        group.hf_repo = None;

        assert_eq!(
            plan(&group, &updates(REPO, FILE, "blob-2")),
            Plan::Unreachable
        );
    }
}
