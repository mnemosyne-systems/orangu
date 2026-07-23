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

use ignore::gitignore::{Gitignore, GitignoreBuilder};
use std::path::Path;

use super::*;
use crate::commands::{shell_words, strip_ascii_prefix};
use crate::git::{
    discover_git_root, git_branch_names, git_local_branch_names, git_remote_names, git_tag_names,
};

pub fn checkout_completion_candidates(
    prefix: &str,
    workspace: &Path,
) -> Option<(usize, Vec<String>)> {
    let (start, token, switch_form) = if let Some(rest) = prefix.strip_prefix("/branch ") {
        ("/branch ".len(), rest, false)
    } else if let Some(rest) = prefix.strip_prefix("/checkout ") {
        ("/checkout ".len(), rest, false)
    } else if let Some(rest) = strip_ascii_prefix(prefix, "git checkout ") {
        (prefix.len() - rest.len(), rest, false)
    } else if let Some(rest) = strip_ascii_prefix(prefix, "checkout ") {
        (prefix.len() - rest.len(), rest, false)
    } else if let Some(rest) = strip_ascii_prefix(prefix, "switch to branch ") {
        // Checked before the shorter `switch to ` so `switch to branch m` keeps
        // `m` as the token rather than treating `branch m` as the branch prefix.
        (prefix.len() - rest.len(), rest, true)
    } else {
        let rest = strip_ascii_prefix(prefix, "switch to ")?;
        (prefix.len() - rest.len(), rest, true)
    };

    let mut candidates: Vec<String> = discover_git_root(workspace)
        .map(|root| {
            let mut refs = git_branch_names(&root);
            if switch_form {
                refs.extend(git_tag_names(&root));
                refs.sort();
                refs.dedup();
            }
            refs
        })
        .unwrap_or_default()
        .into_iter()
        .filter(|r| r.starts_with(token))
        .collect();

    if !switch_form {
        for file in file_completion_candidates(token, workspace) {
            if !candidates.contains(&file) {
                candidates.push(file);
            }
        }
    }

    Some((start, candidates))
}

/// Tab/ghost completion for the `/branch -…` flag forms. The bare `/branch
/// <name>` switch form is owned by [`checkout_completion_candidates`] (which
/// offers branches and files); this handles only the flagged forms so they are
/// not swallowed by that switch completion.
///
/// - `/branch -` offers the flag names `-a`, `--all`, `-b`, `-d`, `-m`;
/// - `/branch -d <name>` completes the deletable (non-protected) local
///   branches, matching `/delete <name>`;
/// - `/branch -b <name>` / `-m <name>` create or rename to a brand-new name, so
///   they are recognised but offer nothing.
///
/// Returns `None` for the switch form and for anything that is not `/branch`.
pub fn branch_completion_candidates(
    prefix: &str,
    workspace: &Path,
) -> Option<(usize, Vec<String>)> {
    let rest = prefix.strip_prefix("/branch ")?;
    if !rest.starts_with('-') {
        return None;
    }
    if let Some(name) = rest.strip_prefix("-d ") {
        let candidates = discover_git_root(workspace)
            .map(|root| git_local_branch_names(&root))
            .unwrap_or_default()
            .into_iter()
            .filter(|b| !crate::git::is_protected_branch(b) && b.starts_with(name))
            .collect();
        return Some(("/branch -d ".len(), candidates));
    }
    if rest.starts_with("-b ") || rest.starts_with("-m ") {
        return Some((prefix.len(), Vec::new()));
    }
    let candidates = ["-a", "--all", "-b", "-d", "-m"]
        .into_iter()
        .filter(|flag| flag.starts_with(rest))
        .map(str::to_string)
        .collect();
    Some(("/branch ".len(), candidates))
}

/// Tab/ghost completion for `/restore` and its natural-language aliases
/// (`restore `, `git restore `): the working-tree files to restore.
/// - the bare `<file>` argument completes the modified (unstaged) files;
/// - `/restore --staged <file>` / `-S <file>` completes the staged files to
///   unstage (the flags only apply to the slash form, mirroring the parser);
/// - `/restore -` offers the `--staged` / `-S` flag names.
///
/// Returns `None` when `prefix` is not a restore command.
pub fn restore_completion_candidates(
    prefix: &str,
    workspace: &Path,
) -> Option<(usize, Vec<String>)> {
    let (start, rest, allow_flags) = if let Some(rest) = prefix.strip_prefix("/restore ") {
        ("/restore ".len(), rest, true)
    } else if let Some(rest) = strip_ascii_prefix(prefix, "git restore ") {
        (prefix.len() - rest.len(), rest, false)
    } else {
        let rest = strip_ascii_prefix(prefix, "restore ")?;
        (prefix.len() - rest.len(), rest, false)
    };

    if allow_flags {
        for flag in ["--staged ", "-S "] {
            if let Some(file) = rest.strip_prefix(flag) {
                let candidates = discover_git_root(workspace)
                    .map(|root| git_modified_candidates(&root, file, true))
                    .unwrap_or_default();
                return Some((start + flag.len(), candidates));
            }
        }
        if rest.starts_with('-') {
            let candidates = ["--staged", "-S"]
                .into_iter()
                .filter(|flag| flag.starts_with(rest))
                .map(str::to_string)
                .collect();
            return Some((start, candidates));
        }
    }

    let candidates = discover_git_root(workspace)
        .map(|root| git_modified_candidates(&root, rest, false))
        .unwrap_or_default();
    Some((start, candidates))
}

/// Tab/ghost completion for the `/push` flag: `/push -` (or a bare `/push `)
/// offers `--force`, `-f`, and the bare `force` keyword the parser also accepts.
/// Slash-only — the natural-language push phrases (`push force`, `push --force`,
/// …) are complete bindings the natural-language ghost already covers.
pub fn push_completion_candidates(prefix: &str) -> Option<(usize, Vec<String>)> {
    let rest = prefix.strip_prefix("/push ")?;
    let candidates = ["--force", "-f", "force"]
        .into_iter()
        .filter(|flag| flag.starts_with(rest))
        .map(str::to_string)
        .collect();
    Some(("/push ".len(), candidates))
}

/// Tab/ghost completion for the `/stash` subcommand: `/stash ` offers `pop`,
/// `list`, `drop`, and the explicit `push`. Slash-only — the natural-language
/// stash phrases (`stash pop`, `git stash list`, …) are complete bindings the
/// natural-language ghost already covers.
pub fn stash_completion_candidates(prefix: &str) -> Option<(usize, Vec<String>)> {
    let rest = prefix.strip_prefix("/stash ")?;
    let candidates = ["pop", "list", "drop", "push"]
        .into_iter()
        .filter(|sub| sub.starts_with(rest))
        .map(str::to_string)
        .collect();
    Some(("/stash ".len(), candidates))
}

/// The modified files in the repository whose path starts with `token`, as
/// reported by `git diff --name-only` — the unstaged working-tree changes, or
/// the staged changes when `staged` is set (`--cached`). Drives `/restore`
/// completion; an empty list outside a repository or on failure.
pub fn git_modified_candidates(repo_root: &Path, token: &str, staged: bool) -> Vec<String> {
    let mut args = vec!["diff", "--name-only"];
    if staged {
        args.push("--cached");
    }
    let Ok(output) = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(&args)
        .output()
    else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    let mut files: Vec<String> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|line| !line.is_empty() && line.starts_with(token))
        .map(str::to_string)
        .collect();
    files.sort();
    files.dedup();
    files
}

pub fn create_file_completion_candidates(
    prefix: &str,
    workspace: &Path,
) -> Option<(usize, Vec<String>)> {
    let (start, token) = if let Some(rest) = prefix.strip_prefix("/create_file ") {
        ("/create_file ".len(), rest)
    } else if let Some(rest) = strip_ascii_prefix(prefix, "git add ") {
        (prefix.len() - rest.len(), rest)
    } else if let Some(rest) = strip_ascii_prefix(prefix, "add file ") {
        (prefix.len() - rest.len(), rest)
    } else {
        let rest = strip_ascii_prefix(prefix, "add ")?;
        (prefix.len() - rest.len(), rest)
    };

    let candidates = discover_git_root(workspace)
        .map(|root| git_untracked_candidates(&root, token))
        .unwrap_or_default();

    Some((start, candidates))
}

pub fn delete_file_completion_candidates(
    prefix: &str,
    workspace: &Path,
) -> Option<(usize, Vec<String>)> {
    let (start, token) = if let Some(rest) = prefix.strip_prefix("/delete_file ") {
        ("/delete_file ".len(), rest)
    } else if let Some(rest) = strip_ascii_prefix(prefix, "git rm ") {
        (prefix.len() - rest.len(), rest)
    } else if let Some(rest) = strip_ascii_prefix(prefix, "delete file ") {
        (prefix.len() - rest.len(), rest)
    } else if let Some(rest) = strip_ascii_prefix(prefix, "remove file ") {
        (prefix.len() - rest.len(), rest)
    } else {
        let rest = strip_ascii_prefix(prefix, "remove ")?;
        (prefix.len() - rest.len(), rest)
    };

    let candidates = discover_git_root(workspace)
        .map(|root| git_tracked_candidates(&root, token))
        .unwrap_or_default();

    Some((start, candidates))
}

pub fn move_file_completion_candidates(
    prefix: &str,
    workspace: &Path,
) -> Option<(usize, Vec<String>)> {
    let (cmd_len, args) = if let Some(rest) = prefix.strip_prefix("/move_file ") {
        ("/move_file ".len(), rest)
    } else if let Some(rest) = strip_ascii_prefix(prefix, "git mv ") {
        (prefix.len() - rest.len(), rest)
    } else if let Some(rest) = strip_ascii_prefix(prefix, "move file ") {
        (prefix.len() - rest.len(), rest)
    } else {
        let rest = strip_ascii_prefix(prefix, "move ")?;
        (prefix.len() - rest.len(), rest)
    };

    let (token_start, token) = last_shell_token(args);
    let previous = args[..token_start].trim_end();
    let previous_count = if previous.is_empty() {
        0
    } else {
        shell_words(previous).unwrap_or_default().len()
    };

    let absolute_start = cmd_len + token_start;
    let candidates = if previous_count == 0 {
        discover_git_root(workspace)
            .map(|root| git_tracked_candidates(&root, token))
            .unwrap_or_default()
    } else {
        file_completion_candidates(token, workspace)
    };

    Some((absolute_start, candidates))
}

pub fn cherry_pick_completion_candidates(
    prefix: &str,
    workspace: &Path,
) -> Option<(usize, Vec<String>)> {
    let (cmd_len, token) = if let Some(rest) = prefix.strip_prefix("/cherry_pick ") {
        ("/cherry_pick ".len(), rest)
    } else if let Some(rest) = strip_ascii_prefix(prefix, "git cherry-pick ") {
        (prefix.len() - rest.len(), rest)
    } else if let Some(rest) = strip_ascii_prefix(prefix, "cherry-pick ") {
        (prefix.len() - rest.len(), rest)
    } else {
        let rest = strip_ascii_prefix(prefix, "cherry pick ")?;
        (prefix.len() - rest.len(), rest)
    };
    let token = token.trim_start();
    let candidates = discover_git_root(workspace)
        .map(|root| git_commit_hashes(&root, token))
        .unwrap_or_default();
    Some((cmd_len, candidates))
}

/// Tab/ghost completion for `/show <commit>` (and its natural-language forms
/// `git show ` / `show commit `): the abbreviated hashes of the latest 25
/// commits reachable from the local `HEAD`, filtered by the typed token. The
/// most recent commit is offered first, so it previews as the inline ghost.
/// Returns `None` when the input is not a show command, leaving the
/// slash-command list to complete the command name itself.
pub fn show_completion_candidates(prefix: &str, workspace: &Path) -> Option<(usize, Vec<String>)> {
    let (start, token) = if let Some(rest) = prefix.strip_prefix("/show ") {
        ("/show ".len(), rest)
    } else if let Some(rest) = strip_ascii_prefix(prefix, "git show ") {
        (prefix.len() - rest.len(), rest)
    } else {
        let rest = strip_ascii_prefix(prefix, "show commit ")?;
        (prefix.len() - rest.len(), rest)
    };
    let token = token.trim_start();
    let candidates = discover_git_root(workspace)
        .map(|root| git_recent_commit_hashes(&root, token))
        .unwrap_or_default();
    Some((start, candidates))
}

/// The abbreviated hashes of the latest 25 commits reachable from the local
/// `HEAD`, in newest-first order, whose hash starts with `token`. Unlike
/// [`git_commit_hashes`] (which walks `origin/main`/`main` for cherry-pick), this
/// walks the current branch locally so `/show` completes commits that are not
/// yet pushed. Returns an empty list outside a repository or on failure.
pub fn git_recent_commit_hashes(repo_root: &Path, token: &str) -> Vec<String> {
    let Ok(output) = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args([
            "log",
            "--abbrev-commit",
            "--format=%h",
            "--max-count=25",
            "HEAD",
        ])
        .output()
    else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .filter(|h| !h.is_empty() && h.starts_with(token))
        .map(str::to_string)
        .collect()
}

/// Completion candidates for `/fetch <remote>` (and its natural-language forms
/// `fetch ` / `git fetch `): the configured remotes whose name starts with the
/// typed token, in [`git_remote_names`] order so the default — `origin` floated
/// to the front — is offered first and previewed as the inline ghost. Returns
/// `None` when the input is not a fetch command, leaving the slash-command list
/// to complete the command name itself.
pub fn fetch_completion_candidates(prefix: &str, workspace: &Path) -> Option<(usize, Vec<String>)> {
    let (start, token) = if let Some(rest) = prefix.strip_prefix("/fetch ") {
        ("/fetch ".len(), rest)
    } else if let Some(rest) = strip_ascii_prefix(prefix, "git fetch ") {
        (prefix.len() - rest.len(), rest)
    } else {
        let rest = strip_ascii_prefix(prefix, "fetch ")?;
        (prefix.len() - rest.len(), rest)
    };

    let candidates = discover_git_root(workspace)
        .map(|root| git_remote_names(&root))
        .unwrap_or_default()
        .into_iter()
        .filter(|remote| remote.starts_with(token))
        .collect();

    Some((start, candidates))
}

/// Completion candidates for `/rebase <target>` (and its natural-language forms
/// `rebase ` / `git rebase `): the rebase target, offered in priority order —
/// local branches first (from `git branch`), then the configured remotes (from
/// `git remote`, `origin` floated to the front), then the remote-tracking
/// branches (e.g. `origin/main`). The first local branch is previewed as the
/// inline ghost. Returns `None` when the input is not a rebase command, leaving
/// the slash-command list to complete the command name itself.
pub fn rebase_completion_candidates(
    prefix: &str,
    workspace: &Path,
) -> Option<(usize, Vec<String>)> {
    let (start, token) = if let Some(rest) = prefix.strip_prefix("/rebase ") {
        ("/rebase ".len(), rest)
    } else if let Some(rest) = strip_ascii_prefix(prefix, "git rebase ") {
        (prefix.len() - rest.len(), rest)
    } else {
        let rest = strip_ascii_prefix(prefix, "rebase ")?;
        (prefix.len() - rest.len(), rest)
    };

    let candidates = discover_git_root(workspace)
        .map(|root| rebase_target_candidates(&root))
        .unwrap_or_default()
        .into_iter()
        .filter(|target| target.starts_with(token))
        .collect();

    Some((start, candidates))
}

/// The rebase targets in offer order — local branches, then remotes, then
/// remote-tracking branches — deduplicated while preserving that order. A
/// remote-tracking branch is any `git branch --all` ref that is not also a local
/// branch (e.g. `origin/main`); local branches and bare remote names come first
/// so they are preferred when the user has only typed a short prefix.
fn rebase_target_candidates(repo_root: &Path) -> Vec<String> {
    let local = git_local_branch_names(repo_root);
    let local_set: std::collections::HashSet<&str> = local.iter().map(String::as_str).collect();
    let remote_branches: Vec<String> = git_branch_names(repo_root)
        .into_iter()
        .filter(|branch| !local_set.contains(branch.as_str()))
        .collect();
    let remotes = git_remote_names(repo_root);

    let mut seen = std::collections::HashSet::new();
    local
        .into_iter()
        .chain(remotes)
        .chain(remote_branches)
        .filter(|candidate| seen.insert(candidate.clone()))
        .collect()
}

pub fn diff_completion_prefix(prefix: &str) -> Option<(usize, &str)> {
    if let Some(branch) = prefix.strip_prefix("/diff ") {
        return Some(("/diff ".len(), branch));
    }
    for command_prefix in ["diff against ", "show diff against ", "git diff "] {
        if let Some(branch) = strip_ascii_prefix(prefix, command_prefix) {
            return Some((prefix.len() - branch.len(), branch));
        }
    }
    None
}

pub fn merge_completion_prefix(prefix: &str) -> Option<(usize, &str)> {
    if let Some(branch) = prefix.strip_prefix("/merge ") {
        return Some(("/merge ".len(), branch));
    }
    for command_prefix in ["git merge ", "merge "] {
        if let Some(branch) = strip_ascii_prefix(prefix, command_prefix) {
            return Some((prefix.len() - branch.len(), branch));
        }
    }
    None
}

pub fn delete_branch_completion_prefix(prefix: &str) -> Option<(usize, &str)> {
    if let Some(branch) = prefix.strip_prefix("/delete ") {
        return Some(("/delete ".len(), branch));
    }
    for command_prefix in ["git branch -D ", "delete branch ", "delete "] {
        if let Some(branch) = strip_ascii_prefix(prefix, command_prefix) {
            return Some((prefix.len() - branch.len(), branch));
        }
    }
    None
}

pub fn git_untracked_candidates(repo_root: &Path, token: &str) -> Vec<String> {
    let Ok(output) = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["ls-files", "--others", "--exclude-standard", "--directory"])
        .output()
    else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    let mut dirs = Vec::new();
    let mut files = Vec::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        if line.is_empty() || !line.starts_with(token) {
            continue;
        }
        if line.ends_with('/') {
            dirs.push(line.to_string());
        } else {
            files.push(line.to_string());
        }
    }
    dirs.sort();
    files.sort();
    dirs.extend(files);
    dirs
}

pub fn git_tracked_candidates(repo_root: &Path, token: &str) -> Vec<String> {
    let Ok(output) = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["ls-files"])
        .output()
    else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    let mut dirs = std::collections::BTreeSet::new();
    let mut files = Vec::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        if line.is_empty() || !line.starts_with(token) {
            continue;
        }
        let rest = &line[token.len()..];
        if let Some(slash) = rest.find('/') {
            dirs.insert(format!("{}{}/", token, &rest[..slash]));
        } else {
            files.push(line.to_string());
        }
    }
    let mut result: Vec<String> = dirs.into_iter().collect();
    files.sort();
    result.extend(files);
    result
}

pub fn git_commit_hashes(repo_root: &Path, token: &str) -> Vec<String> {
    for branch in ["origin/main", "origin/master", "main", "master"] {
        let check = std::process::Command::new("git")
            .arg("-C")
            .arg(repo_root)
            .args(["rev-parse", "--verify", branch])
            .output();
        if !matches!(check, Ok(ref o) if o.status.success()) {
            continue;
        }
        let Ok(output) = std::process::Command::new("git")
            .arg("-C")
            .arg(repo_root)
            .args(["log", "--abbrev-commit", "--format=%h", branch])
            .output()
        else {
            continue;
        };
        if !output.status.success() {
            continue;
        }
        let hashes: Vec<String> = String::from_utf8_lossy(&output.stdout)
            .lines()
            .map(str::trim)
            .filter(|h| !h.is_empty() && h.starts_with(token))
            .take(50)
            .map(str::to_string)
            .collect();
        if !hashes.is_empty() || token.is_empty() {
            return hashes;
        }
    }
    Vec::new()
}

pub fn workspace_gitignore(workspace: &Path) -> Option<Gitignore> {
    let ignore_root = discover_git_root(workspace).unwrap_or_else(|| workspace.to_path_buf());
    let mut builder = GitignoreBuilder::new(&ignore_root);
    let root_gitignore_path = ignore_root.join(".gitignore");
    if root_gitignore_path.is_file() {
        builder.add(root_gitignore_path);
    }
    let workspace_gitignore_path = workspace.join(".gitignore");
    if workspace != ignore_root && workspace_gitignore_path.is_file() {
        builder.add(workspace_gitignore_path);
    }
    builder.build().ok()
}

pub fn should_include_completion_path(
    workspace: &Path,
    path: &Path,
    is_dir: bool,
    gitignore: Option<&Gitignore>,
) -> bool {
    let Ok(relative) = path.strip_prefix(workspace) else {
        return false;
    };

    if gitignore.is_some_and(|matcher| {
        matcher
            .matched_path_or_any_parents(path, is_dir)
            .is_ignore()
    }) {
        return false;
    }

    if relative.as_os_str().is_empty() {
        return true;
    }

    let relative = relative.to_string_lossy().replace('\\', "/");
    !(relative == ".git"
        || relative.starts_with(".git/")
        || relative == "build"
        || relative.starts_with("build/")
        || relative == "target"
        || relative.starts_with("target/"))
}
