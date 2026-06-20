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
use std::{fs, path::Path};

use super::*;
use crate::commands::current_terminal_width;
use crate::render::{
    ANSI_BOLD_OFF, ANSI_BOLD_ON, ANSI_FG_LIGHT_GREEN, ANSI_FG_LIGHT_RED, ANSI_FG_RESET,
    ANSI_FG_SUBTLE,
};

pub fn git_branch_names(repo_root: &Path) -> Vec<String> {
    let Ok(output) = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["branch", "--all", "--format=%(refname:short)"])
        .output()
    else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    let mut branches: Vec<String> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && *l != "HEAD" && !l.ends_with("/HEAD"))
        .map(str::to_string)
        .collect();
    branches.sort();
    branches.dedup();
    branches
}

pub fn git_local_branch_names(repo_root: &Path) -> Vec<String> {
    let Ok(output) = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["branch", "--format=%(refname:short)"])
        .output()
    else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    let mut branches: Vec<String> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect();
    branches.sort();
    branches.dedup();
    branches
}

pub fn git_tag_names(repo_root: &Path) -> Vec<String> {
    let Ok(output) = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["tag"])
        .output()
    else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    let mut tags: Vec<String> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect();
    tags.sort();
    tags.dedup();
    tags
}

/// The names of the configured remotes (`git remote`), in `git`'s own listing
/// order (alphabetical) with `origin` floated to the front when present, so the
/// conventional default for a fetch is offered first. Returns an empty vector
/// when there are no remotes or the command fails.
pub fn git_remote_names(repo_root: &Path) -> Vec<String> {
    let Ok(output) = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .arg("remote")
        .output()
    else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    let mut remotes: Vec<String> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect();
    // Float `origin` to the front so it is the default for `/fetch`; `git remote`
    // already emits the rest in alphabetical order.
    if let Some(index) = remotes.iter().position(|remote| remote == "origin")
        && index != 0
    {
        let origin = remotes.remove(index);
        remotes.insert(0, origin);
    }
    remotes
}

/// The repository's name taken from its `origin` remote URL — the final path
/// segment with any trailing `.git` removed (so `git@host:owner/orangu.git` and
/// `https://host/owner/orangu` both yield `orangu`). Returns `None` when there
/// is no `origin` remote, the command fails, or the URL has no usable segment;
/// callers fall back to the directory name. This is what names an export, so the
/// PDF carries the repository's name even when it was cloned into a differently
/// named directory.
pub fn git_repository_name(repo_root: &Path) -> Option<String> {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["remote", "get-url", "origin"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let url = String::from_utf8_lossy(&output.stdout);
    repository_name_from_url(url.trim())
}

/// Extract the repository name from a remote URL: drop any trailing slashes and
/// `.git` suffix, then take the segment after the last `/` or `:`.
fn repository_name_from_url(url: &str) -> Option<String> {
    let trimmed = url.trim_end_matches('/');
    let trimmed = trimmed.strip_suffix(".git").unwrap_or(trimmed);
    let name = trimmed.rsplit(['/', ':']).next().unwrap_or(trimmed);
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
}

pub fn git_current_branch(repo_root: &Path) -> Result<String> {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["branch", "--show-current"])
        .output()
        .context("failed to run git branch")?;
    if !output.status.success() {
        return Err(anyhow!("failed to determine current branch"));
    }
    let branch = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if branch.is_empty() {
        return Err(anyhow!(
            "could not determine current branch (detached HEAD?)"
        ));
    }
    Ok(branch)
}

pub fn is_protected_branch(branch: &str) -> bool {
    matches!(branch, "main" | "master")
}

pub fn workspace_is_not_git(_workspace: &Path) -> Result<String> {
    Err(anyhow!("diff is only available inside a Git repository"))
}

pub fn git_show_file_content(workspace: &Path, file_path: &Path, rev: &str) -> Result<String> {
    let repo_root = discover_git_root(workspace)
        .ok_or_else(|| anyhow!("git show is only available inside a Git repository"))?;
    let relative = file_path.strip_prefix(&repo_root).unwrap_or(file_path);
    let spec = format!("{rev}:{}", relative.display());
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(&repo_root)
        .args(["show", &spec])
        .output()
        .context("failed to run git show")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(anyhow!(
            "git show failed{}",
            if stderr.is_empty() {
                String::new()
            } else {
                format!(": {stderr}")
            }
        ));
    }
    String::from_utf8(output.stdout).context("git show output was not UTF-8")
}

pub fn git_file_commit_hashes(repo_root: &Path, relative_path: &Path) -> Vec<String> {
    let Ok(output) = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["log", "--follow", "--format=%h", "--"])
        .arg(relative_path)
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
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect()
}

pub fn list_workspace_files_tree(workspace: &Path) -> Result<String> {
    let mut lines = vec![workspace.display().to_string()];
    append_workspace_tree(workspace, "", &mut lines)?;
    Ok(lines.join("\n"))
}

pub fn append_workspace_tree(
    directory: &Path,
    prefix: &str,
    lines: &mut Vec<String>,
) -> Result<()> {
    let mut entries = fs::read_dir(directory)
        .with_context(|| format!("failed to read {}", directory.display()))?
        .collect::<std::result::Result<Vec<_>, _>>()
        .with_context(|| format!("failed to read {}", directory.display()))?;
    entries.retain(|entry| should_include_listed_path(&entry.file_name(), &entry.path()));
    entries.sort_by(|left, right| {
        compare_tree_entries(
            &left.file_name(),
            &left.path(),
            &right.file_name(),
            &right.path(),
        )
    });
    let total_entries = entries.len();

    for (index, entry) in entries.into_iter().enumerate() {
        let path = entry.path();
        let is_dir = path.is_dir();
        let name = entry.file_name().to_string_lossy().to_string();
        let branch = if index + 1 == total_entries {
            "└── "
        } else {
            "├── "
        };
        lines.push(format!("{prefix}{branch}{name}"));
        if is_dir {
            let next_prefix = if index + 1 == total_entries {
                format!("{prefix}    ")
            } else {
                format!("{prefix}│   ")
            };
            append_workspace_tree(&path, &next_prefix, lines)?;
        }
    }

    Ok(())
}

pub fn should_include_listed_path(file_name: &std::ffi::OsStr, path: &Path) -> bool {
    !(path.is_dir() && matches!(file_name.to_str(), Some(".git" | "build" | "target")))
}

pub fn compare_tree_entries(
    left_name: &std::ffi::OsStr,
    left_path: &Path,
    right_name: &std::ffi::OsStr,
    right_path: &Path,
) -> std::cmp::Ordering {
    left_path
        .is_file()
        .cmp(&right_path.is_file())
        .then_with(|| {
            left_name
                .to_string_lossy()
                .to_lowercase()
                .cmp(&right_name.to_string_lossy().to_lowercase())
        })
}

pub fn grep_output(workspace: &Path, pattern: &str) -> Result<String> {
    let repo_root = discover_git_root(workspace)
        .ok_or_else(|| anyhow!("grep is only available inside a Git repository"))?;
    let terminal_width = current_terminal_width();
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(&repo_root)
        .args(["grep", pattern])
        .env("COLUMNS", terminal_width.to_string())
        .output()
        .context("failed to run git grep")?;
    if output.status.code() == Some(1) {
        return Ok(format!("No matches for '{pattern}'."));
    }
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(anyhow!(
            "git grep failed{}",
            if stderr.is_empty() {
                String::new()
            } else {
                format!(": {stderr}")
            }
        ));
    }
    if let Some(pager_command) = configured_git_pager(&repo_root, "grep")? {
        run_git_diff_pager(&repo_root, &pager_command, &output.stdout, terminal_width)
    } else {
        String::from_utf8(output.stdout).context("git grep output was not UTF-8")
    }
}

pub fn status_output(workspace: &Path) -> Result<String> {
    let repo_root = discover_git_root(workspace)
        .ok_or_else(|| anyhow!("status is only available inside a Git repository"))?;
    if let Some(output) = try_gh_status(&repo_root)? {
        return Ok(output);
    }
    git_status(&repo_root)
}

pub fn try_gh_status(_repo_root: &Path) -> Result<Option<String>> {
    Ok(None)
}

pub fn git_status(repo_root: &Path) -> Result<String> {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["status", "--branch", "--short"])
        .output()
        .context("failed to run git status")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(anyhow!(
            "git status failed{}",
            if stderr.is_empty() {
                String::new()
            } else {
                format!(": {stderr}")
            }
        ));
    }
    let raw = String::from_utf8_lossy(&output.stdout);
    let colored = colorize_git_status(&raw);
    if colored.trim().is_empty() {
        Ok("Nothing to commit, working tree clean.".to_string())
    } else {
        Ok(colored)
    }
}

pub fn colorize_git_status(raw: &str) -> String {
    let mut result = String::new();
    for line in raw.lines() {
        if line.starts_with("## ") {
            result.push_str(ANSI_FG_SUBTLE);
            result.push_str(line);
            result.push_str(ANSI_FG_RESET);
        } else if line.len() >= 2 {
            let x = line.as_bytes()[0] as char;
            let y = line.as_bytes()[1] as char;
            let color = status_entry_color(x, y);
            let display_char = if x != ' ' { x } else { y };
            let path = line.get(3..).unwrap_or("");
            result.push_str(color);
            result.push(display_char);
            result.push(' ');
            result.push_str(path);
            if !color.is_empty() {
                result.push_str(ANSI_FG_RESET);
            }
        } else {
            result.push_str(line);
        }
        result.push('\n');
    }
    result.trim_end_matches('\n').to_string()
}

pub fn status_entry_color(x: char, y: char) -> &'static str {
    if x == 'D' || y == 'D' {
        return ANSI_FG_LIGHT_RED;
    }
    if x == 'A' || x == '?' {
        return ANSI_FG_LIGHT_GREEN;
    }
    ""
}

pub fn log_output(workspace: &Path, count: Option<u64>) -> Result<String> {
    let repo_root = discover_git_root(workspace)
        .ok_or_else(|| anyhow!("log is only available inside a Git repository"))?;
    if let Some(output) = try_gh_log(&repo_root)? {
        return Ok(output);
    }
    git_log(&repo_root, count)
}

pub fn try_gh_log(_repo_root: &Path) -> Result<Option<String>> {
    Ok(None)
}

pub fn git_log(repo_root: &Path, count: Option<u64>) -> Result<String> {
    let has_lg = std::process::Command::new("git")
        .args(["config", "--global", "--get", "alias.lg"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    let mut command = std::process::Command::new("git");
    command.arg("-C").arg(repo_root);
    command.args(["-c", "color.ui=always"]);
    if has_lg {
        command.arg("lg");
    } else {
        command.args([
            "log",
            "--color=always",
            "--graph",
            "--oneline",
            "--decorate",
        ]);
    }
    if let Some(count) = count {
        command.arg(format!("--max-count={count}"));
    }

    let output = command.output().context("failed to run git log")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(anyhow!(
            "git log failed{}",
            if stderr.is_empty() {
                String::new()
            } else {
                format!(": {stderr}")
            }
        ));
    }
    let log = String::from_utf8_lossy(&output.stdout).to_string();
    let summary = pending_changes_summary(repo_root);
    if log.trim().is_empty() {
        Ok(format!("No commits yet.\n{summary}"))
    } else {
        Ok(format!("{}\n{summary}", log.trim_end_matches('\n')))
    }
}

/// Counts uncommitted (tracked) and untracked changes in the working tree and
/// renders a one-line, highlighted summary to append to `/log` output. Returns
/// an empty string if the status check fails, so the log itself is never lost.
pub fn pending_changes_summary(repo_root: &Path) -> String {
    let (total, untracked) = match count_pending_changes(repo_root) {
        Ok(counts) => counts,
        Err(_) => return String::new(),
    };
    if total == 0 {
        return format!("{ANSI_FG_SUBTLE}● Working tree clean{ANSI_FG_RESET}");
    }
    let tracked = total - untracked;
    let mut parts = Vec::new();
    if tracked > 0 {
        parts.push(format!("{tracked} uncommitted"));
    }
    if untracked > 0 {
        parts.push(format!("{untracked} untracked"));
    }
    format!(
        "{ANSI_BOLD_ON}{ANSI_FG_LIGHT_RED}● {} change{} ({}){ANSI_FG_RESET}{ANSI_BOLD_OFF}",
        total,
        if total == 1 { "" } else { "s" },
        parts.join(", "),
    )
}

/// Returns `(total, untracked)` counts of changed paths in the working tree
/// via `git status --porcelain`. Each porcelain line is one path.
pub fn count_pending_changes(repo_root: &Path) -> Result<(usize, usize)> {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["status", "--porcelain"])
        .output()
        .context("failed to run git status")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(anyhow!(
            "git status failed{}",
            if stderr.is_empty() {
                String::new()
            } else {
                format!(": {stderr}")
            }
        ));
    }
    let raw = String::from_utf8_lossy(&output.stdout);
    let mut total = 0;
    let mut untracked = 0;
    for line in raw.lines() {
        if line.trim().is_empty() {
            continue;
        }
        total += 1;
        if line.starts_with("??") {
            untracked += 1;
        }
    }
    Ok((total, untracked))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn colorizes_git_status_output() {
        use crate::render::{ANSI_FG_LIGHT_GREEN, ANSI_FG_LIGHT_RED};
        let raw = "## main...origin/main\nA  new_file.rs\n M modified.rs\nD  deleted.rs\n?? untracked.txt\n";
        let colored = colorize_git_status(raw);
        assert!(colored.contains("## main...origin/main"));
        // Each entry renders as a single status char followed by a space and the path.
        // ANSI codes may precede the char, so check without a leading newline.
        assert!(colored.contains("A new_file.rs"));
        assert!(colored.contains("M modified.rs"));
        assert!(colored.contains("D deleted.rs"));
        assert!(colored.contains("? untracked.txt"));
        let green_start = colored
            .find(ANSI_FG_LIGHT_GREEN)
            .expect("green color present");
        assert!(colored[green_start..].contains("new_file.rs"));
        let red_start = colored.find(ANSI_FG_LIGHT_RED).expect("red color present");
        assert!(colored[red_start..].contains("deleted.rs"));
        let mod_idx = colored.find("modified.rs").expect("modified.rs present");
        let before_mod = &colored[..mod_idx];
        assert!(!before_mod.ends_with(ANSI_FG_LIGHT_RED));
        assert!(!before_mod.ends_with(ANSI_FG_LIGHT_GREEN));
        assert!(colored.contains("untracked.txt"));
        let green_positions: Vec<_> = colored.match_indices(ANSI_FG_LIGHT_GREEN).collect();
        assert!(green_positions.len() >= 2);
    }

    #[test]
    fn git_remote_names_floats_origin_first_then_alphabetical() {
        let workspace = tempdir().expect("workspace");
        init_git_for_test(workspace.path());

        // No remotes configured yet.
        assert!(git_remote_names(workspace.path()).is_empty());

        // Added out of order; `git remote` lists alphabetically and we float
        // `origin` to the front so it is the default for /fetch and /rebase.
        for (name, url) in [
            ("upstream", "https://example.com/upstream.git"),
            ("origin", "https://example.com/origin.git"),
            ("fork", "https://example.com/fork.git"),
        ] {
            git_run(workspace.path(), &["remote", "add", name, url]);
        }
        assert_eq!(
            git_remote_names(workspace.path()),
            vec!["origin", "fork", "upstream"]
        );
    }

    #[test]
    fn repository_name_from_url_handles_common_forms() {
        for (url, expected) in [
            ("https://github.com/owner/orangu.git", "orangu"),
            ("git@github.com:owner/orangu.git", "orangu"),
            ("https://github.com/owner/orangu", "orangu"),
            ("https://github.com/owner/orangu/", "orangu"),
            ("/home/user/official/orangu", "orangu"),
        ] {
            assert_eq!(
                repository_name_from_url(url).as_deref(),
                Some(expected),
                "url {url}"
            );
        }
        assert_eq!(repository_name_from_url(""), None);
    }

    #[test]
    fn git_repository_name_reads_origin_not_directory() {
        // The repo lives in a directory named `official` but its `origin` points
        // at `orangu`; the export name follows the remote, not the directory.
        let parent = tempdir().expect("parent");
        let workspace = parent.path().join("official");
        std::fs::create_dir(&workspace).expect("workspace dir");
        init_git_for_test(&workspace);
        git_run(
            &workspace,
            &[
                "remote",
                "add",
                "origin",
                "https://example.com/owner/orangu.git",
            ],
        );
        assert_eq!(git_repository_name(&workspace).as_deref(), Some("orangu"));
    }

    #[test]
    fn git_repository_name_is_none_without_origin() {
        let workspace = tempdir().expect("workspace");
        init_git_for_test(workspace.path());
        assert_eq!(git_repository_name(workspace.path()), None);
    }

    #[test]
    fn git_remote_names_leaves_alphabetical_order_without_origin() {
        let workspace = tempdir().expect("workspace");
        init_git_for_test(workspace.path());
        for (name, url) in [
            ("upstream", "https://example.com/upstream.git"),
            ("fork", "https://example.com/fork.git"),
        ] {
            git_run(workspace.path(), &["remote", "add", name, url]);
        }
        // No `origin` to special-case, so git's alphabetical order stands.
        assert_eq!(git_remote_names(workspace.path()), vec!["fork", "upstream"]);
    }

    #[test]
    fn counts_uncommitted_and_untracked_changes() {
        let workspace = tempdir().expect("workspace");
        init_git_for_test(workspace.path());

        // Clean tree: no pending changes.
        std::fs::write(workspace.path().join("tracked.txt"), "first\n").expect("write");
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
        assert_eq!(
            count_pending_changes(workspace.path()).expect("count"),
            (0, 0)
        );
        assert!(pending_changes_summary(workspace.path()).contains("Working tree clean"));

        // Modify a tracked file and add an untracked one.
        std::fs::write(workspace.path().join("tracked.txt"), "changed\n").expect("write");
        std::fs::write(workspace.path().join("untracked.txt"), "new\n").expect("write");
        assert_eq!(
            count_pending_changes(workspace.path()).expect("count"),
            (2, 1)
        );
        let summary = pending_changes_summary(workspace.path());
        assert!(summary.contains("2 changes"), "summary: {summary}");
        assert!(summary.contains("1 uncommitted"), "summary: {summary}");
        assert!(summary.contains("1 untracked"), "summary: {summary}");
    }
}
