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

use anyhow::{Context, Result, anyhow};
use std::path::{Path, PathBuf};

use super::*;

pub fn rebase_output(workspace: &Path, forge: Forge) -> Result<String> {
    let repo_root = discover_git_root(workspace)
        .ok_or_else(|| anyhow!("rebase is only available inside a Git repository"))?;
    if let Some(output) = try_gh_rebase(&repo_root, forge)? {
        return Ok(output);
    }
    git_rebase_main(&repo_root)
}

pub fn try_gh_rebase(repo_root: &Path, forge: Forge) -> Result<Option<String>> {
    // Only GitHub's `gh` exposes the default branch directly; for GitLab we
    // fall through to the git-based detection in `git_rebase_main`.
    if forge != Forge::GitHub {
        return Ok(None);
    }
    let branch_output = match std::process::Command::new("gh")
        .args([
            "repo",
            "view",
            "--json",
            "defaultBranchRef",
            "--jq",
            ".defaultBranchRef.name",
        ])
        .current_dir(repo_root)
        .output()
    {
        Ok(output) => output,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err).context("failed to run gh"),
    };
    if !branch_output.status.success() {
        return Ok(None);
    }
    let default_branch = String::from_utf8_lossy(&branch_output.stdout)
        .trim()
        .to_string();
    if default_branch.is_empty() {
        return Ok(None);
    }
    git_rebase_onto(repo_root, &default_branch).map(Some)
}

pub fn git_rebase_main(repo_root: &Path) -> Result<String> {
    for branch in ["main", "master"] {
        let check = std::process::Command::new("git")
            .arg("-C")
            .arg(repo_root)
            .args(["ls-remote", "--heads", "origin", branch])
            .output()
            .context("failed to run git ls-remote")?;
        if check.status.success() && !check.stdout.is_empty() {
            return git_rebase_onto(repo_root, branch);
        }
    }
    Err(anyhow!(
        "could not determine the default branch (tried main and master)"
    ))
}

pub fn git_rebase_onto(repo_root: &Path, branch: &str) -> Result<String> {
    let fetch = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["fetch", "origin", branch])
        .output()
        .context("failed to run git fetch")?;
    if !fetch.status.success() {
        let stderr = String::from_utf8_lossy(&fetch.stderr).trim().to_string();
        return Err(anyhow!(
            "git fetch failed{}",
            if stderr.is_empty() {
                String::new()
            } else {
                format!(": {stderr}")
            }
        ));
    }
    let rebase = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["rebase", &format!("origin/{branch}")])
        .output()
        .context("failed to run git rebase")?;
    if !rebase.status.success() {
        let stderr = String::from_utf8_lossy(&rebase.stderr).trim().to_string();
        let stdout = String::from_utf8_lossy(&rebase.stdout).trim().to_string();
        let detail = [stdout, stderr]
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        return Err(anyhow!(
            "git rebase failed{}",
            if detail.is_empty() {
                String::new()
            } else {
                format!(": {detail}")
            }
        ));
    }
    let stdout = String::from_utf8_lossy(&rebase.stdout).trim().to_string();
    Ok(if stdout.is_empty() {
        format!("Rebased onto origin/{branch}")
    } else {
        stdout
    })
}

/// Determine the repository's default branch name (e.g. `main`), preferring
/// `gh` (GitHub only), then `origin/HEAD`, then the first of `main`/`master`
/// present on origin. GitLab relies on the git-based detection.
fn git_default_branch(repo_root: &Path, forge: Forge) -> Option<String> {
    if forge == Forge::GitHub
        && let Ok(output) = std::process::Command::new("gh")
            .args([
                "repo",
                "view",
                "--json",
                "defaultBranchRef",
                "--jq",
                ".defaultBranchRef.name",
            ])
            .current_dir(repo_root)
            .output()
        && output.status.success()
    {
        let name = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !name.is_empty() {
            return Some(name);
        }
    }

    if let Ok(output) = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["symbolic-ref", "--short", "refs/remotes/origin/HEAD"])
        .output()
        && output.status.success()
    {
        let name = String::from_utf8_lossy(&output.stdout)
            .trim()
            .trim_start_matches("origin/")
            .to_string();
        if !name.is_empty() {
            return Some(name);
        }
    }

    for branch in ["main", "master"] {
        if let Ok(output) = std::process::Command::new("git")
            .arg("-C")
            .arg(repo_root)
            .args(["ls-remote", "--heads", "origin", branch])
            .output()
            && output.status.success()
            && !output.stdout.is_empty()
        {
            return Some(branch.to_string());
        }
    }

    None
}

/// Fast-forward the local default branch (main/master) to match `origin`,
/// without ever creating a merge commit, rebasing, or touching a feature
/// branch's working tree. Returns `Ok(None)` when there is nothing to do (no
/// `origin` remote or no detectable default branch), `Ok(Some(msg))` on a
/// successful sync, and `Err` if a sync was attempted but failed.
pub fn sync_default_branch(workspace: &Path, forge: Forge) -> Result<Option<String>> {
    let Some(repo_root) = discover_git_root(workspace) else {
        return Ok(None);
    };

    // Nothing to sync without an `origin` remote.
    let remotes = std::process::Command::new("git")
        .arg("-C")
        .arg(&repo_root)
        .arg("remote")
        .output()
        .context("failed to run git remote")?;
    if !remotes.status.success()
        || !String::from_utf8_lossy(&remotes.stdout)
            .lines()
            .any(|remote| remote.trim() == "origin")
    {
        return Ok(None);
    }

    let Some(default) = git_default_branch(&repo_root, forge) else {
        return Ok(None);
    };

    let on_default = workspace_branch_name(&repo_root).as_deref() == Some(default.as_str());
    let output = if on_default {
        // On the default branch: fast-forward the working tree.
        std::process::Command::new("git")
            .arg("-C")
            .arg(&repo_root)
            .args(["pull", "--ff-only", "origin", &default])
            .output()
            .context("failed to run git pull")?
    } else {
        // On another branch: fast-forward the local default ref in place.
        std::process::Command::new("git")
            .arg("-C")
            .arg(&repo_root)
            .args(["fetch", "origin", &format!("{default}:{default}")])
            .output()
            .context("failed to run git fetch")?
    };

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(anyhow!(
            "{}",
            if stderr.is_empty() {
                format!("could not sync {default}")
            } else {
                stderr
            }
        ));
    }

    Ok(Some(format!("Synced {default} with origin")))
}

pub fn merge_output(workspace: &Path, branch: &str, forge: Forge) -> Result<String> {
    let repo_root = discover_git_root(workspace)
        .ok_or_else(|| anyhow!("merge is only available inside a Git repository"))?;
    let is_local = git_local_branch_names(&repo_root)
        .iter()
        .any(|b| b == branch);
    if !is_local && let Some(output) = try_gh_merge(&repo_root, branch, forge)? {
        return Ok(output);
    }
    git_merge(&repo_root, branch)
}

pub fn try_gh_merge(repo_root: &Path, branch: &str, forge: Forge) -> Result<Option<String>> {
    let cli = forge.cli();
    // GitHub: `gh pr merge --merge BRANCH`. GitLab: `glab mr merge BRANCH --yes`
    // (positional source branch, `--yes` skips the confirmation prompt).
    let args: Vec<&str> = match forge {
        Forge::GitHub => vec!["pr", "merge", "--merge", branch],
        Forge::GitLab => vec!["mr", "merge", branch, "--yes"],
    };
    let output = match std::process::Command::new(cli)
        .args(&args)
        .current_dir(repo_root)
        .output()
    {
        Ok(output) => output,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err).context(format!("failed to run {cli}")),
    };
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let request = if forge == Forge::GitHub { "pr" } else { "mr" };
        return Err(anyhow!(
            "{cli} {request} merge failed{}",
            if stderr.is_empty() {
                String::new()
            } else {
                format!(": {stderr}")
            }
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let mut combined = stdout;
    if !stderr.is_empty() {
        if !combined.is_empty() {
            combined.push('\n');
        }
        combined.push_str(&stderr);
    }
    Ok(Some(if combined.is_empty() {
        format!("Merged branch '{branch}'")
    } else {
        combined
    }))
}

pub fn git_merge(repo_root: &Path, branch: &str) -> Result<String> {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["merge", "--ff", branch])
        .output()
        .context("failed to run git merge")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let detail = [stdout, stderr]
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        return Err(anyhow!(
            "git merge failed{}",
            if detail.is_empty() {
                String::new()
            } else {
                format!(": {detail}")
            }
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok(if stdout.is_empty() {
        format!("Merged '{branch}'")
    } else {
        stdout
    })
}

pub fn git_checkout(repo_root: &Path, target: &str) -> Result<String> {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["checkout", target])
        .output()
        .context("failed to run git checkout")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let detail = [stdout, stderr]
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        return Err(anyhow!(
            "git checkout failed{}",
            if detail.is_empty() {
                String::new()
            } else {
                format!(": {detail}")
            }
        ));
    }
    Ok(format!("Switched to '{target}'"))
}

pub fn add_file_output(workspace: &Path, path: &str) -> Result<String> {
    let repo_root = discover_git_root(workspace)
        .ok_or_else(|| anyhow!("add_file is only available inside a Git repository"))?;
    if let Some(output) = try_gh_add_file(&repo_root, path)? {
        return Ok(output);
    }
    git_add_file(&repo_root, path)
}

pub fn try_gh_add_file(_repo_root: &Path, _path: &str) -> Result<Option<String>> {
    Ok(None)
}

pub fn git_add_file(repo_root: &Path, path: &str) -> Result<String> {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["add", path])
        .output()
        .context("failed to run git add")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(anyhow!(
            "git add failed{}",
            if stderr.is_empty() {
                String::new()
            } else {
                format!(": {stderr}")
            }
        ));
    }
    Ok(format!("Staged '{path}'"))
}

pub fn remove_file_output(workspace: &Path, path: &str) -> Result<String> {
    let repo_root = discover_git_root(workspace)
        .ok_or_else(|| anyhow!("remove_file is only available inside a Git repository"))?;
    if let Some(output) = try_gh_remove_file(&repo_root, path)? {
        return Ok(output);
    }
    git_remove_file(&repo_root, path)
}

pub fn try_gh_remove_file(_repo_root: &Path, _path: &str) -> Result<Option<String>> {
    Ok(None)
}

pub fn git_remove_file(repo_root: &Path, path: &str) -> Result<String> {
    let mut args = vec!["rm"];
    if repo_root.join(path.trim_end_matches('/')).is_dir() {
        args.push("-r");
    }
    args.push(path);
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(&args)
        .output()
        .context("failed to run git rm")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(anyhow!(
            "git rm failed{}",
            if stderr.is_empty() {
                String::new()
            } else {
                format!(": {stderr}")
            }
        ));
    }
    Ok(format!("Removed '{path}'"))
}

pub fn move_file_output(workspace: &Path, source: &str, destination: &str) -> Result<String> {
    let repo_root = discover_git_root(workspace)
        .ok_or_else(|| anyhow!("move_file is only available inside a Git repository"))?;
    if let Some(output) = try_gh_move_file(&repo_root, source, destination)? {
        return Ok(output);
    }
    git_move_file(&repo_root, source, destination)
}

pub fn try_gh_move_file(
    _repo_root: &Path,
    _source: &str,
    _destination: &str,
) -> Result<Option<String>> {
    Ok(None)
}

pub fn git_move_file(repo_root: &Path, source: &str, destination: &str) -> Result<String> {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["mv", source, destination])
        .output()
        .context("failed to run git mv")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(anyhow!(
            "git mv failed{}",
            if stderr.is_empty() {
                String::new()
            } else {
                format!(": {stderr}")
            }
        ));
    }
    Ok(format!("Moved '{source}' to '{destination}'"))
}

pub fn cherry_pick_output(workspace: &Path, commit: &str) -> Result<String> {
    let repo_root = discover_git_root(workspace)
        .ok_or_else(|| anyhow!("cherry_pick is only available inside a Git repository"))?;
    if let Some(output) = try_gh_cherry_pick(&repo_root, commit)? {
        return Ok(output);
    }
    git_cherry_pick(&repo_root, commit)
}

pub fn try_gh_cherry_pick(_repo_root: &Path, _commit: &str) -> Result<Option<String>> {
    Ok(None)
}

pub fn git_cherry_pick(repo_root: &Path, commit: &str) -> Result<String> {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["cherry-pick", commit])
        .output()
        .context("failed to run git cherry-pick")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let detail = [stdout, stderr]
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        return Err(anyhow!(
            "git cherry-pick failed{}",
            if detail.is_empty() {
                String::new()
            } else {
                format!(": {detail}")
            }
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok(if stdout.is_empty() {
        format!("Cherry-picked {commit}")
    } else {
        stdout
    })
}

pub fn commit_output(workspace: &Path, message: &str) -> Result<String> {
    let repo_root = discover_git_root(workspace)
        .ok_or_else(|| anyhow!("commit is only available inside a Git repository"))?;
    if let Some(output) = try_gh_commit(&repo_root, message)? {
        return Ok(output);
    }
    git_commit(&repo_root, message)
}

pub fn try_gh_commit(_repo_root: &Path, _message: &str) -> Result<Option<String>> {
    Ok(None)
}

pub fn git_commit(repo_root: &Path, message: &str) -> Result<String> {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["commit", "-a", "-m", message])
        .output()
        .context("failed to run git commit")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let detail = [stdout, stderr]
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        return Err(anyhow!(
            "git commit failed{}",
            if detail.is_empty() {
                String::new()
            } else {
                format!(": {detail}")
            }
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok(if stdout.is_empty() {
        format!("Committed: {message}")
    } else {
        stdout
    })
}

pub fn amend_output(workspace: &Path, message: &str) -> Result<String> {
    let repo_root = discover_git_root(workspace)
        .ok_or_else(|| anyhow!("amend is only available inside a Git repository"))?;
    if let Some(output) = try_gh_amend(&repo_root, message)? {
        return Ok(output);
    }
    git_amend(&repo_root, message)
}

pub fn try_gh_amend(_repo_root: &Path, _message: &str) -> Result<Option<String>> {
    Ok(None)
}

pub fn git_amend(repo_root: &Path, message: &str) -> Result<String> {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["commit", "--amend", "-m", message])
        .output()
        .context("failed to run git commit --amend")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let detail = [stdout, stderr]
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        return Err(anyhow!(
            "git commit --amend failed{}",
            if detail.is_empty() {
                String::new()
            } else {
                format!(": {detail}")
            }
        ));
    }
    Ok(String::new())
}

pub fn push_output(workspace: &Path, force: bool) -> Result<Option<String>> {
    let repo_root = discover_git_root(workspace)
        .ok_or_else(|| anyhow!("push is only available inside a Git repository"))?;
    if try_gh_push(&repo_root, force)?.is_none() {
        git_push(&repo_root, force)?;
    }
    Ok(pr_sync_advice(&repo_root))
}

pub fn try_gh_push(_repo_root: &Path, _force: bool) -> Result<Option<String>> {
    Ok(None)
}

pub fn git_push(repo_root: &Path, force: bool) -> Result<String> {
    let branch = git_current_branch(repo_root)?;
    if force && is_protected_branch(&branch) {
        return Err(anyhow!(
            "force push is not allowed on the '{}' branch",
            branch
        ));
    }
    // Require the branch to be rebased on top of main/master before pushing, so
    // we never push a branch that is behind its base. Protected branches are
    // their own base, and a missing base branch (e.g. no main/master) is not
    // grounds to block the push.
    if !is_protected_branch(&branch)
        && let Ok(base_ref) = git_find_base_ref(repo_root)
    {
        let behind = git_commit_count(repo_root, &format!("HEAD..{base_ref}"))?;
        if behind > 0 {
            return Err(anyhow!(
                "branch '{branch}' is {behind} commit{} behind {base_ref}; \
                 rebase on {base_ref} before pushing (run /rebase)",
                if behind == 1 { "" } else { "s" }
            ));
        }
    }
    let mut command = std::process::Command::new("git");
    command.arg("-C").arg(repo_root).arg("push");
    if force {
        command.arg("-f");
    }
    command.args(["origin", &branch]);
    let output = command.output().context("failed to run git push")?;
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    if !output.status.success() {
        let detail = [&stdout, &stderr]
            .iter()
            .filter(|s| !s.is_empty())
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        return Err(anyhow!(
            "git push failed{}",
            if detail.is_empty() {
                String::new()
            } else {
                format!(": {detail}")
            }
        ));
    }
    let combined = [stdout, stderr]
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    Ok(if combined.is_empty() {
        format!("Pushed '{branch}' to origin")
    } else {
        combined
    })
}

pub fn init_repo_output(workspace: &Path) -> Result<String> {
    if let Some(output) = try_gh_init_repo(workspace)? {
        return Ok(output);
    }
    git_init(workspace)
}

pub fn try_gh_init_repo(_workspace: &Path) -> Result<Option<String>> {
    Ok(None)
}

pub fn git_init(workspace: &Path) -> Result<String> {
    let output = std::process::Command::new("git")
        .arg("init")
        .current_dir(workspace)
        .output()
        .context("failed to run git init")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(anyhow!(
            "git init failed{}",
            if stderr.is_empty() {
                String::new()
            } else {
                format!(": {stderr}")
            }
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok(if stdout.is_empty() {
        format!("Initialized Git repository in {}", workspace.display())
    } else {
        stdout
    })
}

pub fn squash_output(workspace: &Path) -> Result<String> {
    let repo_root = discover_git_root(workspace)
        .ok_or_else(|| anyhow!("squash is only available inside a Git repository"))?;
    if let Some(output) = try_gh_squash(&repo_root)? {
        return Ok(output);
    }
    git_squash(&repo_root)
}

pub fn try_gh_squash(_repo_root: &Path) -> Result<Option<String>> {
    Ok(None)
}

pub fn git_squash(repo_root: &Path) -> Result<String> {
    let current = git_current_branch(repo_root)?;
    if is_protected_branch(&current) {
        return Err(anyhow!("squash is not allowed on the '{}' branch", current));
    }

    let base_ref = git_find_base_ref(repo_root)?;

    let merge_base_output = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["merge-base", "HEAD", &base_ref])
        .output()
        .context("failed to run git merge-base")?;
    if !merge_base_output.status.success() {
        return Err(anyhow!("could not find merge base with {base_ref}"));
    }
    let merge_base = String::from_utf8_lossy(&merge_base_output.stdout)
        .trim()
        .to_string();

    let count_output = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["rev-list", "--count", &format!("{merge_base}..HEAD")])
        .output()
        .context("failed to run git rev-list")?;
    let count: usize = String::from_utf8_lossy(&count_output.stdout)
        .trim()
        .parse()
        .unwrap_or(0);

    if count == 0 {
        return Err(anyhow!("no commits to squash on current branch"));
    }
    if count == 1 {
        return Err(anyhow!("nothing to squash: branch has only one commit"));
    }

    let oldest_hash_output = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args([
            "log",
            "--format=%H",
            "--reverse",
            &format!("{merge_base}..HEAD"),
        ])
        .output()
        .context("failed to run git log")?;
    let oldest_hash = String::from_utf8_lossy(&oldest_hash_output.stdout)
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .to_string();

    let message_output = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["log", "-1", "--format=%B", &oldest_hash])
        .output()
        .context("failed to run git log")?;
    let first_message = String::from_utf8_lossy(&message_output.stdout)
        .trim()
        .to_string();

    let reset = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["reset", "--soft", &merge_base])
        .output()
        .context("failed to run git reset --soft")?;
    if !reset.status.success() {
        let stderr = String::from_utf8_lossy(&reset.stderr).trim().to_string();
        return Err(anyhow!(
            "git reset --soft failed{}",
            if stderr.is_empty() {
                String::new()
            } else {
                format!(": {stderr}")
            }
        ));
    }

    let commit = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["commit", "-m", &first_message])
        .output()
        .context("failed to run git commit")?;
    if !commit.status.success() {
        let stderr = String::from_utf8_lossy(&commit.stderr).trim().to_string();
        let stdout = String::from_utf8_lossy(&commit.stdout).trim().to_string();
        let detail = [stdout, stderr]
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        return Err(anyhow!(
            "git commit failed{}",
            if detail.is_empty() {
                String::new()
            } else {
                format!(": {detail}")
            }
        ));
    }

    Ok(format!("Squashed {count} commits into '{current}'"))
}

pub fn git_find_base_ref(repo_root: &Path) -> Result<String> {
    for branch in ["origin/main", "origin/master"] {
        let check = std::process::Command::new("git")
            .arg("-C")
            .arg(repo_root)
            .args(["rev-parse", "--verify", branch])
            .output()
            .context("failed to run git rev-parse")?;
        if check.status.success() {
            return Ok(branch.to_string());
        }
    }
    for branch in ["main", "master"] {
        let check = std::process::Command::new("git")
            .arg("-C")
            .arg(repo_root)
            .args(["rev-parse", "--verify", branch])
            .output()
            .context("failed to run git rev-parse")?;
        if check.status.success() {
            return Ok(branch.to_string());
        }
    }
    Err(anyhow!(
        "could not find base branch (tried origin/main, origin/master, main, master)"
    ))
}

pub fn stash_output(workspace: &Path) -> Result<String> {
    let repo_root = discover_git_root(workspace)
        .ok_or_else(|| anyhow!("stash is only available inside a Git repository"))?;
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(&repo_root)
        .args(["stash", "push"])
        .output()
        .context("failed to run git stash push")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(anyhow!(
            "git stash push failed{}",
            if stderr.is_empty() {
                String::new()
            } else {
                format!(": {stderr}")
            }
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok(if stdout.is_empty() {
        "Changes stashed".to_string()
    } else {
        stdout
    })
}

pub fn stash_pop_output(workspace: &Path) -> Result<String> {
    let repo_root = discover_git_root(workspace)
        .ok_or_else(|| anyhow!("stash is only available inside a Git repository"))?;
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(&repo_root)
        .args(["stash", "pop"])
        .output()
        .context("failed to run git stash pop")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(anyhow!(
            "git stash pop failed{}",
            if stderr.is_empty() {
                String::new()
            } else {
                format!(": {stderr}")
            }
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok(if stdout.is_empty() {
        "Stash applied and dropped".to_string()
    } else {
        stdout
    })
}

pub fn stash_list_output(workspace: &Path) -> Result<String> {
    let repo_root = discover_git_root(workspace)
        .ok_or_else(|| anyhow!("stash is only available inside a Git repository"))?;
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(&repo_root)
        .args(["stash", "list"])
        .output()
        .context("failed to run git stash list")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(anyhow!(
            "git stash list failed{}",
            if stderr.is_empty() {
                String::new()
            } else {
                format!(": {stderr}")
            }
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok(if stdout.is_empty() {
        "No stashes found".to_string()
    } else {
        stdout
    })
}

pub fn stash_drop_output(workspace: &Path) -> Result<String> {
    let repo_root = discover_git_root(workspace)
        .ok_or_else(|| anyhow!("stash is only available inside a Git repository"))?;
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(&repo_root)
        .args(["stash", "drop"])
        .output()
        .context("failed to run git stash drop")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(anyhow!(
            "git stash drop failed{}",
            if stderr.is_empty() {
                String::new()
            } else {
                format!(": {stderr}")
            }
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok(if stdout.is_empty() {
        "Stash dropped".to_string()
    } else {
        stdout
    })
}

/// Run `git -C <repo_root> bisect <args...>` and return its trimmed stdout.
///
/// On a non-zero exit the combined stdout/stderr is surfaced as the error so
/// git's own message (e.g. "You need to start by 'git bisect start'") reaches
/// the user. On success, an empty stdout is replaced with `fallback` so every
/// subcommand has something to print.
fn run_bisect(repo_root: &Path, args: &[&str], fallback: &str) -> Result<String> {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .arg("bisect")
        .args(args)
        .output()
        .context("failed to run git bisect")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let detail = [stdout, stderr]
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        return Err(anyhow!(
            "git bisect {} failed{}",
            args.first().copied().unwrap_or(""),
            if detail.is_empty() {
                String::new()
            } else {
                format!(": {detail}")
            }
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok(if stdout.is_empty() {
        fallback.to_string()
    } else {
        stdout
    })
}

/// `/bisect start [<args>]` — begin a session. `args` is the optional rev
/// specification (e.g. `<bad> <good>`), split on whitespace and forwarded to
/// `git bisect start`.
pub fn bisect_start_output(workspace: &Path, args: Option<&str>) -> Result<String> {
    let repo_root = discover_git_root(workspace)
        .ok_or_else(|| anyhow!("bisect is only available inside a Git repository"))?;
    let mut cmd_args = vec!["start"];
    let args_owned;
    let extra: Vec<&str>;
    if let Some(a) = args {
        args_owned = a.to_string();
        extra = args_owned.split_whitespace().collect();
        cmd_args.extend_from_slice(&extra);
    }
    run_bisect(&repo_root, &cmd_args, "Bisect session started")
}

/// `/bisect good [<commit>]` — mark a commit (default `HEAD`) as good.
pub fn bisect_good_output(workspace: &Path, commit: Option<&str>) -> Result<String> {
    let repo_root = discover_git_root(workspace)
        .ok_or_else(|| anyhow!("bisect is only available inside a Git repository"))?;
    let mut args = vec!["good"];
    if let Some(c) = commit {
        args.push(c);
    }
    run_bisect(&repo_root, &args, "Marked as good")
}

/// `/bisect bad [<commit>]` — mark a commit (default `HEAD`) as bad.
pub fn bisect_bad_output(workspace: &Path, commit: Option<&str>) -> Result<String> {
    let repo_root = discover_git_root(workspace)
        .ok_or_else(|| anyhow!("bisect is only available inside a Git repository"))?;
    let mut args = vec!["bad"];
    if let Some(c) = commit {
        args.push(c);
    }
    run_bisect(&repo_root, &args, "Marked as bad")
}

/// `/bisect skip [<commit>]` — skip a commit (default `HEAD`) that cannot be
/// tested.
pub fn bisect_skip_output(workspace: &Path, commit: Option<&str>) -> Result<String> {
    let repo_root = discover_git_root(workspace)
        .ok_or_else(|| anyhow!("bisect is only available inside a Git repository"))?;
    let mut args = vec!["skip"];
    if let Some(c) = commit {
        args.push(c);
    }
    run_bisect(&repo_root, &args, "Commit skipped")
}

/// `/bisect reset` — end the session and return to the original branch.
pub fn bisect_reset_output(workspace: &Path) -> Result<String> {
    let repo_root = discover_git_root(workspace)
        .ok_or_else(|| anyhow!("bisect is only available inside a Git repository"))?;
    run_bisect(&repo_root, &["reset"], "Bisect session ended")
}

/// `/bisect log` — print the commits marked so far in the session.
pub fn bisect_log_output(workspace: &Path) -> Result<String> {
    let repo_root = discover_git_root(workspace)
        .ok_or_else(|| anyhow!("bisect is only available inside a Git repository"))?;
    run_bisect(&repo_root, &["log"], "Bisect log is empty")
}

/// Returns the path of the git directory for `repo_root`, using
/// `git rev-parse --git-dir` so that worktrees and non-standard `GIT_DIR`
/// locations are handled correctly.
fn git_dir(repo_root: &Path) -> Option<PathBuf> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["rev-parse", "--git-dir"])
        .output()
        .ok()
        .filter(|o| o.status.success())?;
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() {
        return None;
    }
    let p = Path::new(&s);
    Some(if p.is_absolute() {
        p.to_path_buf()
    } else {
        repo_root.join(p)
    })
}

/// `/bisect status` (and a bare `/bisect`) — report the session state.
///
/// A session is detected by the presence of `BISECT_START` in the git
/// directory (resolved via [`git_dir`] so worktrees are handled). When no
/// session is active a friendly hint is returned instead of letting
/// `git bisect log` fail; otherwise the bisect log is shown.
pub fn bisect_status_output(workspace: &Path) -> Result<String> {
    let repo_root = discover_git_root(workspace)
        .ok_or_else(|| anyhow!("bisect is only available inside a Git repository"))?;
    let git_dir =
        git_dir(&repo_root).ok_or_else(|| anyhow!("could not determine git directory"))?;
    if !git_dir.join("BISECT_START").exists() {
        return Ok("No bisect session in progress. Use /bisect start to begin.".to_string());
    }
    run_bisect(&repo_root, &["log"], "Bisect session in progress")
}

pub fn branch_list_output(workspace: &Path) -> Result<String> {
    let repo_root = discover_git_root(workspace)
        .ok_or_else(|| anyhow!("branch is only available inside a Git repository"))?;
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(&repo_root)
        .args(["branch"])
        .output()
        .context("failed to run git branch")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(anyhow!(
            "git branch failed{}",
            if stderr.is_empty() {
                String::new()
            } else {
                format!(": {stderr}")
            }
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok(if stdout.is_empty() {
        "No local branches found".to_string()
    } else {
        stdout
    })
}

pub fn branch_list_all_output(workspace: &Path) -> Result<String> {
    let repo_root = discover_git_root(workspace)
        .ok_or_else(|| anyhow!("branch is only available inside a Git repository"))?;
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(&repo_root)
        .args(["branch", "-a"])
        .output()
        .context("failed to run git branch -a")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(anyhow!(
            "git branch -a failed{}",
            if stderr.is_empty() {
                String::new()
            } else {
                format!(": {stderr}")
            }
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok(if stdout.is_empty() {
        "No branches found".to_string()
    } else {
        stdout
    })
}

pub fn branch_create_output(workspace: &Path, name: &str) -> Result<String> {
    let repo_root = discover_git_root(workspace)
        .ok_or_else(|| anyhow!("branch is only available inside a Git repository"))?;
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(&repo_root)
        .args(["checkout", "-b", name])
        .output()
        .context("failed to run git checkout -b")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(anyhow!(
            "git checkout -b failed{}",
            if stderr.is_empty() {
                String::new()
            } else {
                format!(": {stderr}")
            }
        ));
    }
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    Ok(if stderr.is_empty() {
        format!("Switched to a new branch '{name}'")
    } else {
        stderr
    })
}

pub fn branch_rename_output(workspace: &Path, new_name: &str) -> Result<String> {
    let repo_root = discover_git_root(workspace)
        .ok_or_else(|| anyhow!("branch is only available inside a Git repository"))?;
    let current = git_current_branch(&repo_root)?;
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(&repo_root)
        .args(["branch", "-m", new_name])
        .output()
        .context("failed to run git branch -m")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(anyhow!(
            "git branch -m failed{}",
            if stderr.is_empty() {
                String::new()
            } else {
                format!(": {stderr}")
            }
        ));
    }
    Ok(format!("Renamed branch '{current}' to '{new_name}'"))
}

pub fn branch_delete_output(workspace: &Path, name: &str) -> Result<String> {
    let repo_root = discover_git_root(workspace)
        .ok_or_else(|| anyhow!("branch is only available inside a Git repository"))?;
    if is_protected_branch(name) {
        return Err(anyhow!("deleting the '{}' branch is not allowed", name));
    }
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(&repo_root)
        .args(["branch", "-D", name])
        .output()
        .context("failed to run git branch -D")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(anyhow!(
            "git branch -D failed{}",
            if stderr.is_empty() {
                String::new()
            } else {
                format!(": {stderr}")
            }
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok(if stdout.is_empty() {
        format!("Deleted branch '{name}'")
    } else {
        stdout
    })
}

pub fn restore_output(workspace: &Path, path: &str, staged: bool) -> Result<String> {
    let repo_root = discover_git_root(workspace)
        .ok_or_else(|| anyhow!("restore is only available inside a Git repository"))?;
    let args: &[&str] = if staged {
        &["restore", "--staged", path]
    } else {
        &["restore", path]
    };
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(&repo_root)
        .args(args)
        .output()
        .context("failed to run git restore")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(anyhow!(
            "git restore failed{}",
            if stderr.is_empty() {
                String::new()
            } else {
                format!(": {stderr}")
            }
        ));
    }
    Ok(if staged {
        format!("Unstaged '{path}'")
    } else {
        format!("Restored '{path}'")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::process_env_lock;
    use tempfile::tempdir;

    #[test]
    fn force_push_blocked_on_protected_branches() {
        assert!(is_protected_branch("main"));
        assert!(is_protected_branch("master"));
        assert!(!is_protected_branch("feature/my-branch"));
        assert!(!is_protected_branch("develop"));
    }

    #[test]
    fn push_blocked_when_branch_behind_base() {
        let _env_lock = process_env_lock().lock().unwrap_or_else(|p| p.into_inner());
        let workspace = tempdir().expect("workspace");
        let home = tempdir().expect("home");
        let _home_guard = EnvVarGuard::set_path("HOME", home.path());
        init_git_for_test(workspace.path());

        let git = |args: &[&str]| {
            std::process::Command::new("git")
                .args(args)
                .current_dir(workspace.path())
                .status()
                .expect("git command");
        };
        let commit = |name: &str, content: &str, msg: &str| {
            std::fs::write(workspace.path().join(name), content).expect("write");
            git(&["add", "."]);
            git(&["commit", "-m", msg]);
        };

        git(&["checkout", "-B", "main"]);
        commit("base.txt", "base\n", "Base commit");

        // Feature branch, then advance main so the branch falls behind.
        git(&["checkout", "-b", "feature/push-test"]);
        commit("feature.txt", "feat\n", "Feature commit");
        git(&["checkout", "main"]);
        commit("base.txt", "base2\n", "Base advances");
        git(&["checkout", "feature/push-test"]);

        let result = git_push(workspace.path(), false);
        assert!(result.is_err(), "push should be blocked when behind base");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("behind main") && msg.contains("/rebase"),
            "error should explain the branch is behind and to rebase: {msg}"
        );
    }

    #[test]
    fn init_repo_creates_git_repository() {
        let workspace = tempdir().expect("workspace");
        assert!(!workspace.path().join(".git").exists());
        let result = init_repo_output(workspace.path());
        assert!(result.is_ok(), "init_repo_output failed: {:?}", result);
        assert!(workspace.path().join(".git").exists());
    }

    #[test]
    fn bisect_requires_git_repository() {
        // A throwaway directory that is not a git repository: every bisect
        // entry point reports it clearly rather than shelling out to git.
        let dir = tempdir().expect("workspace");
        for result in [
            bisect_status_output(dir.path()),
            bisect_log_output(dir.path()),
            bisect_reset_output(dir.path()),
            bisect_good_output(dir.path(), None),
        ] {
            let err = result.expect_err("expected an error outside a repo");
            assert!(
                err.to_string().contains("Git repository"),
                "unexpected error: {err}"
            );
        }
    }

    #[test]
    fn bisect_status_reports_no_session() {
        let workspace = tempdir().expect("workspace");
        std::process::Command::new("git")
            .arg("init")
            .current_dir(workspace.path())
            .output()
            .expect("git init");
        let status = bisect_status_output(workspace.path()).expect("status");
        assert!(
            status.contains("No bisect session"),
            "expected the no-session hint, got: {status}"
        );
    }

    #[test]
    fn bisect_start_status_and_reset_flow() {
        let _env_lock = process_env_lock().lock().unwrap_or_else(|p| p.into_inner());
        let workspace = tempdir().expect("workspace");
        let home = tempdir().expect("home");
        let _home_guard = EnvVarGuard::set_path("HOME", home.path());
        init_git_for_test(workspace.path());

        let git = |args: &[&str]| {
            std::process::Command::new("git")
                .args(args)
                .current_dir(workspace.path())
                .status()
                .expect("git command");
        };
        let commit = |name: &str, content: &str, msg: &str| {
            std::fs::write(workspace.path().join(name), content).expect("write");
            git(&["add", "."]);
            git(&["commit", "-m", msg]);
        };
        git(&["checkout", "-B", "main"]);
        commit("a.txt", "1\n", "first");
        commit("a.txt", "2\n", "second");
        commit("a.txt", "3\n", "third");

        let bisect_start = workspace.path().join(".git").join("BISECT_START");

        // No session yet.
        assert!(!bisect_start.exists());
        let status = bisect_status_output(workspace.path()).expect("status");
        assert!(status.contains("No bisect session"), "got: {status}");

        // Starting a session creates BISECT_START, and status reflects it.
        bisect_start_output(workspace.path(), None).expect("start");
        assert!(bisect_start.exists());
        let status = bisect_status_output(workspace.path()).expect("status");
        assert!(
            !status.contains("No bisect session"),
            "session should be active: {status}"
        );

        // Resetting ends the session.
        bisect_reset_output(workspace.path()).expect("reset");
        assert!(!bisect_start.exists());
        let status = bisect_status_output(workspace.path()).expect("status");
        assert!(
            status.contains("No bisect session"),
            "session should be cleared: {status}"
        );
    }

    #[test]
    fn bisect_good_bad_and_skip_mark_commits() {
        let _env_lock = process_env_lock().lock().unwrap_or_else(|p| p.into_inner());
        let workspace = tempdir().expect("workspace");
        let home = tempdir().expect("home");
        let _home_guard = EnvVarGuard::set_path("HOME", home.path());
        init_git_for_test(workspace.path());

        let git = |args: &[&str]| {
            std::process::Command::new("git")
                .args(args)
                .current_dir(workspace.path())
                .output()
                .expect("git command")
        };
        let rev_parse = |spec: &str| {
            String::from_utf8_lossy(&git(&["rev-parse", spec]).stdout)
                .trim()
                .to_string()
        };
        let commit = |name: &str, content: &str, msg: &str| {
            std::fs::write(workspace.path().join(name), content).expect("write");
            git(&["add", "."]);
            git(&["commit", "-m", msg]);
        };
        git(&["checkout", "-B", "main"]);
        // Five commits so the suspect range stays non-trivial: marking the
        // tips good/bad leaves several commits to test, and skipping one keeps
        // the session running instead of converging immediately.
        for n in 1..=5 {
            commit("a.txt", &format!("{n}\n"), &format!("commit {n}"));
        }
        let first = rev_parse("HEAD~4");
        let last = rev_parse("HEAD");

        bisect_start_output(workspace.path(), None).expect("start");

        // Marking an explicit bad tip then an explicit good base narrows the
        // search; git checks out a commit between them and the session lives on.
        bisect_bad_output(workspace.path(), Some(&last)).expect("bad <last>");
        bisect_good_output(workspace.path(), Some(&first)).expect("good <first>");

        // Skipping the currently checked-out commit (HEAD default) is accepted
        // and the session remains active with commits still to test.
        bisect_skip_output(workspace.path(), None).expect("skip");

        // The log records the good/bad marks we made.
        let log = bisect_log_output(workspace.path()).expect("log");
        assert!(
            log.contains("good") && log.contains("bad"),
            "log should record the good and bad marks: {log}"
        );

        bisect_reset_output(workspace.path()).expect("reset");
        let bisect_start = workspace.path().join(".git").join("BISECT_START");
        assert!(!bisect_start.exists(), "reset should clear the session");
    }

    #[test]
    fn delete_branch_blocked_on_protected_branches() {
        let workspace = tempdir().expect("workspace");
        std::process::Command::new("git")
            .arg("init")
            .current_dir(workspace.path())
            .output()
            .expect("git init");
        for branch in ["main", "master"] {
            let result = branch_delete_output(workspace.path(), branch);
            assert!(result.is_err(), "should block deletion of '{branch}'");
            let msg = result.unwrap_err().to_string();
            assert!(
                msg.contains(branch),
                "error should mention branch name: {msg}"
            );
        }
    }

    #[test]
    fn squash_blocked_on_protected_branches() {
        let _env_lock = process_env_lock().lock().unwrap_or_else(|p| p.into_inner());
        let workspace = tempdir().expect("workspace");
        let home = tempdir().expect("home");
        let _home_guard = EnvVarGuard::set_path("HOME", home.path());
        init_git_for_test(workspace.path());
        // Ensure we are on a protected branch
        std::process::Command::new("git")
            .args(["checkout", "-B", "main"])
            .current_dir(workspace.path())
            .status()
            .expect("git checkout -B main");
        std::fs::write(workspace.path().join("file.txt"), "first\n").expect("write");
        std::process::Command::new("git")
            .args(["add", "."])
            .current_dir(workspace.path())
            .status()
            .expect("git add");
        std::process::Command::new("git")
            .args(["commit", "-m", "Initial commit"])
            .current_dir(workspace.path())
            .status()
            .expect("git commit");
        let result = squash_output(workspace.path());
        assert!(result.is_err(), "squash should fail on protected branch");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("not allowed"),
            "error should mention 'not allowed': {msg}"
        );
    }

    #[test]
    fn squash_combines_multiple_commits() {
        let _env_lock = process_env_lock().lock().unwrap_or_else(|p| p.into_inner());
        let workspace = tempdir().expect("workspace");
        let home = tempdir().expect("home");
        let _home_guard = EnvVarGuard::set_path("HOME", home.path());
        init_git_for_test(workspace.path());

        // Ensure the base branch is named "main"
        std::process::Command::new("git")
            .args(["checkout", "-B", "main"])
            .current_dir(workspace.path())
            .status()
            .expect("git checkout -B main");

        // Commit on main as the base
        std::fs::write(workspace.path().join("base.txt"), "base\n").expect("write base");
        std::process::Command::new("git")
            .args(["add", "."])
            .current_dir(workspace.path())
            .status()
            .expect("git add");
        std::process::Command::new("git")
            .args(["commit", "-m", "Base commit"])
            .current_dir(workspace.path())
            .status()
            .expect("git commit base");

        // Create a feature branch off main
        std::process::Command::new("git")
            .args(["checkout", "-b", "feature/squash-test"])
            .current_dir(workspace.path())
            .status()
            .expect("git checkout -b");

        // Add two commits on the feature branch
        std::fs::write(workspace.path().join("feature.txt"), "feat1\n").expect("write feat1");
        std::process::Command::new("git")
            .args(["add", "."])
            .current_dir(workspace.path())
            .status()
            .expect("git add feat1");
        std::process::Command::new("git")
            .args(["commit", "-m", "First feature commit"])
            .current_dir(workspace.path())
            .status()
            .expect("git commit feat1");

        std::fs::write(workspace.path().join("feature.txt"), "feat2\n").expect("write feat2");
        std::process::Command::new("git")
            .args(["add", "."])
            .current_dir(workspace.path())
            .status()
            .expect("git add feat2");
        std::process::Command::new("git")
            .args(["commit", "-m", "Second feature commit"])
            .current_dir(workspace.path())
            .status()
            .expect("git commit feat2");

        let result = squash_output(workspace.path());
        assert!(result.is_ok(), "squash failed: {:?}", result);
        let msg = result.unwrap();
        assert!(
            msg.contains("Squashed 2 commits"),
            "unexpected message: {msg}"
        );

        // After squash, only one commit on the branch
        let count_output = std::process::Command::new("git")
            .args(["rev-list", "--count", "main..HEAD"])
            .current_dir(workspace.path())
            .output()
            .expect("git rev-list");
        let count: usize = String::from_utf8_lossy(&count_output.stdout)
            .trim()
            .parse()
            .unwrap_or(99);
        assert_eq!(count, 1, "expected 1 commit after squash, got {count}");
    }

    #[test]
    fn amend_rewrites_last_commit_message() {
        let _env_lock = process_env_lock().lock().unwrap_or_else(|p| p.into_inner());
        let workspace = tempdir().expect("workspace");
        let home = tempdir().expect("home");
        let _home_guard = EnvVarGuard::set_path("HOME", home.path());
        init_git_for_test(workspace.path());

        std::fs::write(workspace.path().join("file.txt"), "content\n").expect("write");
        std::process::Command::new("git")
            .args(["add", "."])
            .current_dir(workspace.path())
            .status()
            .expect("git add");
        std::process::Command::new("git")
            .args(["commit", "-m", "Original message"])
            .current_dir(workspace.path())
            .status()
            .expect("git commit");

        let result = amend_output(workspace.path(), "[#42] Amended message");
        assert!(result.is_ok(), "amend failed: {:?}", result);

        let log = std::process::Command::new("git")
            .args(["log", "-1", "--format=%s"])
            .current_dir(workspace.path())
            .output()
            .expect("git log");
        let subject = String::from_utf8_lossy(&log.stdout).trim().to_string();
        assert_eq!(subject, "[#42] Amended message");
    }

    #[test]
    fn sync_default_branch_fast_forwards_local_main() {
        let _env_lock = process_env_lock().lock().unwrap_or_else(|p| p.into_inner());
        let home = tempdir().expect("home");
        let _home_guard = EnvVarGuard::set_path("HOME", home.path());

        // A bare "origin" plus an author clone that pushes commits to it.
        let origin = tempdir().expect("origin");
        git_run(origin.path(), &["init", "--bare", "-b", "main"]);

        let author = tempdir().expect("author");
        git_run(author.path(), &["init", "-b", "main"]);
        git_run(author.path(), &["config", "user.name", "Author"]);
        git_run(
            author.path(),
            &["config", "user.email", "author@example.com"],
        );
        git_run(
            author.path(),
            &["remote", "add", "origin", origin.path().to_str().unwrap()],
        );
        std::fs::write(author.path().join("f.txt"), "1\n").expect("write");
        git_run(author.path(), &["add", "."]);
        git_run(author.path(), &["commit", "-m", "base"]);
        git_run(author.path(), &["push", "-u", "origin", "main"]);

        // The consumer clones origin and sits on main.
        let consumer = tempdir().expect("consumer");
        git_run(
            home.path(),
            &[
                "clone",
                origin.path().to_str().unwrap(),
                consumer.path().to_str().unwrap(),
            ],
        );
        git_run(consumer.path(), &["config", "user.name", "Consumer"]);
        git_run(
            consumer.path(),
            &["config", "user.email", "consumer@example.com"],
        );
        assert_eq!(rev_count(consumer.path(), "HEAD"), 1);

        // Author advances main on origin.
        std::fs::write(author.path().join("f.txt"), "2\n").expect("write");
        git_run(author.path(), &["commit", "-am", "second"]);
        git_run(author.path(), &["push", "origin", "main"]);

        // On main, sync fast-forwards the working tree.
        let message = sync_default_branch(consumer.path(), Forge::GitHub).expect("sync");
        assert_eq!(message.as_deref(), Some("Synced main with origin"));
        assert_eq!(rev_count(consumer.path(), "HEAD"), 2);

        // On a feature branch, sync still fast-forwards the local main ref.
        git_run(consumer.path(), &["checkout", "-b", "feature"]);
        std::fs::write(author.path().join("f.txt"), "3\n").expect("write");
        git_run(author.path(), &["commit", "-am", "third"]);
        git_run(author.path(), &["push", "origin", "main"]);

        sync_default_branch(consumer.path(), Forge::GitHub).expect("sync on feature");
        assert_eq!(rev_count(consumer.path(), "main"), 3);
        assert_eq!(
            workspace_branch_name(consumer.path()).as_deref(),
            Some("feature"),
            "current branch is untouched",
        );
    }

    #[test]
    fn sync_default_branch_without_origin_is_a_no_op() {
        let _env_lock = process_env_lock().lock().unwrap_or_else(|p| p.into_inner());
        let workspace = tempdir().expect("workspace");
        let home = tempdir().expect("home");
        let _home_guard = EnvVarGuard::set_path("HOME", home.path());
        init_git_for_test(workspace.path());

        // No origin remote configured ⇒ nothing to sync, no error.
        assert_eq!(
            sync_default_branch(workspace.path(), Forge::GitHub).unwrap(),
            None
        );
    }
}
