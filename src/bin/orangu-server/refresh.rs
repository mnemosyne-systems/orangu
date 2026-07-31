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
//! The local copy goes first and the download follows, rather than the other
//! way round: the point of a refresh is that the repo's files have changed,
//! so the new revision is a full second copy on disk, not a cheap
//! blob-sharing snapshot. Deleting first means a 17 GiB model needs 17 GiB
//! of free space to refresh, not 34. The cost is that an interrupted
//! download leaves the model missing rather than stale — which is why the
//! confirmation line says so, and why `download`'s own resume (a `.part`
//! file picked up on the next run) is what recovers it.
//!
//! With no argument it prints `list`'s table with every already-current row
//! greyed ([`orangu::model_spec::Dimming::UpToDate`]) and prompts for an
//! `NR`, so the rows worth refreshing are the only ones standing out.

use crate::{check_for_updates, confirm, dimming, model_support};
use anyhow::{Context, Result, anyhow, bail};
use orangu::model_spec::{Dimming, ModelGroup};
use std::{io::Write, path::Path};

/// Deletes one model and downloads it again. `model` resolves the same way
/// `delete`'s argument does, except that a `MODEL` name matching more than
/// one row is an error rather than a first-match (see
/// [`orangu::model_spec::resolve_refresh_target`]); omitting it picks one
/// interactively.
pub fn run(models_dir: &Path, model: Option<String>, yes: bool) -> Result<()> {
    let group = match model {
        Some(spec) => orangu::model_spec::resolve_refresh_target(models_dir, &spec)?,
        None => select_model_to_refresh(models_dir)?,
    };

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

    let plural = if group.paths.len() == 1 { "" } else { "s" };
    if !yes {
        let confirmed = confirm(&format!(
            "Refresh '{spec}' ({} file{plural}, {})? The local copy is deleted first, then downloaded again. [y/N]: ",
            group.paths.len(),
            orangu::format::format_bytes(group.size_bytes),
        ))?;
        if !confirmed {
            println!("Aborted. Nothing deleted.");
            return Ok(());
        }
    }

    orangu::model_spec::delete_model(models_dir, &group)?;
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
/// `list` does, so the un-greyed rows are exactly the `(Refresh)` ones.
///
/// An unreachable Hub greys nothing (no row is *known* to be behind), which
/// reads correctly: with no update information, no row is more worth picking
/// than another.
fn select_model_to_refresh(models_dir: &Path) -> Result<ModelGroup> {
    let models = orangu::model_spec::scan_models_dir(models_dir)
        .with_context(|| format!("scanning {}", models_dir.display()))?;
    let mut groups = orangu::model_spec::group_models(&models);
    if groups.is_empty() {
        bail!("no .gguf models found under {}", models_dir.display());
    }
    let latest_commits = check_for_updates(&groups);
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
    let behind = groups
        .iter()
        .filter(|group| group.is_behind(&latest_commits))
        .count();
    if behind == 0 {
        println!("\nEvery model is at its repository's latest revision.");
    }

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
        .map(|index| groups.swap_remove(index))
        .ok_or_else(|| anyhow!("no model with NR {nr} ({count} model(s) listed)"))
}
