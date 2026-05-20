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
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};

use crossterm::terminal::{disable_raw_mode, enable_raw_mode};

use super::commands::{current_terminal_width, shell_words};
use super::render::{ANSI_FG_LIGHT_GREEN, ANSI_FG_LIGHT_RED, ANSI_FG_RESET, ANSI_FG_SUBTLE};
use orangu::tools::resolve_workspace_path;

struct RawModePauseGuard;

impl RawModePauseGuard {
    fn new() -> Result<Self> {
        disable_raw_mode()?;
        Ok(Self)
    }
}

impl Drop for RawModePauseGuard {
    fn drop(&mut self) {
        let _ = enable_raw_mode();
    }
}

pub fn discover_git_root(workspace: &Path) -> Option<PathBuf> {
    discover_git_repository(workspace).map(|(root, _)| root)
}

pub fn discover_git_dir(workspace: &Path) -> Option<PathBuf> {
    discover_git_repository(workspace).map(|(_, git_dir)| git_dir)
}

pub fn discover_git_repository(workspace: &Path) -> Option<(PathBuf, PathBuf)> {
    for ancestor in workspace.ancestors() {
        let git_entry = ancestor.join(".git");
        if git_entry.is_dir() {
            return Some((ancestor.to_path_buf(), git_entry));
        }
        if git_entry.is_file() {
            let gitdir = fs::read_to_string(&git_entry).ok()?;
            let relative = gitdir.trim().strip_prefix("gitdir: ")?.trim();
            let path = Path::new(relative);
            let git_dir = if path.is_absolute() {
                path.to_path_buf()
            } else {
                ancestor.join(path)
            };
            return Some((ancestor.to_path_buf(), git_dir));
        }
    }
    None
}

pub fn workspace_branch_name(workspace: &Path) -> Option<String> {
    let git_dir = discover_git_dir(workspace)?;
    let head = fs::read_to_string(git_dir.join("HEAD")).ok()?;
    let reference = head.trim().strip_prefix("ref: ")?;
    reference.strip_prefix("refs/heads/").map(ToOwned::to_owned)
}

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

pub fn open_in_editor(workspace: &Path, raw_path: &str) -> Result<()> {
    let editor = std::env::var("EDITOR").context("EDITOR is not set")?;
    let editor_parts = shell_words(&editor)?;
    let path = resolve_workspace_path(workspace, raw_path)?;
    let (program, args) = editor_parts
        .split_first()
        .ok_or_else(|| anyhow!("EDITOR is empty"))?;

    let _raw_mode_pause_guard = RawModePauseGuard::new()?;
    let _child = std::process::Command::new(program)
        .args(args)
        .arg(&path)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .with_context(|| format!("failed to launch editor '{}'", editor))?;

    Ok(())
}

pub fn git_workspace_diff(workspace: &Path) -> Result<String> {
    let Some(repo_root) = discover_git_root(workspace) else {
        return workspace_is_not_git(workspace);
    };
    let terminal_width = current_terminal_width();
    let workspace_pathspec = workspace
        .strip_prefix(&repo_root)
        .ok()
        .filter(|path| !path.as_os_str().is_empty());

    let mut command = std::process::Command::new("git");
    command
        .arg("-C")
        .arg(&repo_root)
        .arg("diff")
        .arg("--color=always");
    command.env("COLUMNS", terminal_width.to_string());
    if let Some(pathspec) = workspace_pathspec {
        command.arg("--").arg(pathspec);
    }

    let output = command.output().context("failed to run git diff")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(anyhow!(
            "git diff failed{}",
            if stderr.is_empty() {
                String::new()
            } else {
                format!(": {stderr}")
            }
        ));
    }

    let diff = if let Some(pager_command) = configured_git_diff_pager(&repo_root)? {
        run_git_diff_pager(&repo_root, &pager_command, &output.stdout, terminal_width)?
    } else {
        String::from_utf8_lossy(&output.stdout).to_string()
    };
    if diff.trim().is_empty() {
        Ok("No changes against the current branch.".to_string())
    } else {
        Ok(diff)
    }
}

pub fn git_diff_against_branch(workspace: &Path, branch: &str) -> Result<String> {
    let Some(repo_root) = discover_git_root(workspace) else {
        return workspace_is_not_git(workspace);
    };
    let terminal_width = current_terminal_width();

    let mut command = std::process::Command::new("git");
    command
        .arg("-C")
        .arg(&repo_root)
        .arg("diff")
        .arg("--color=always")
        .arg(format!("{branch}...HEAD"));
    command.env("COLUMNS", terminal_width.to_string());

    let output = command.output().context("failed to run git diff")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(anyhow!(
            "git diff failed{}",
            if stderr.is_empty() {
                String::new()
            } else {
                format!(": {stderr}")
            }
        ));
    }

    let diff = if let Some(pager_command) = configured_git_diff_pager(&repo_root)? {
        run_git_diff_pager(&repo_root, &pager_command, &output.stdout, terminal_width)?
    } else {
        String::from_utf8_lossy(&output.stdout).to_string()
    };
    if diff.trim().is_empty() {
        Ok(format!("No changes against {branch}."))
    } else {
        Ok(diff)
    }
}

pub fn configured_git_diff_pager(repo_root: &Path) -> Result<Option<String>> {
    for key in ["pager.diff", "core.pager"] {
        let output = std::process::Command::new("git")
            .arg("-C")
            .arg(repo_root)
            .args(["config", "--get", key])
            .output()
            .with_context(|| format!("failed to read git config key {key}"))?;
        if !output.status.success() {
            continue;
        }
        let value = String::from_utf8(output.stdout)
            .with_context(|| format!("git config key {key} was not valid UTF-8"))?;
        let value = value.trim();
        if value.is_empty() || looks_like_interactive_pager(value) {
            continue;
        }
        return Ok(Some(value.to_string()));
    }

    Ok(None)
}

pub fn looks_like_interactive_pager(command: &str) -> bool {
    let first = shell_words(command)
        .ok()
        .and_then(|parts| parts.into_iter().next())
        .unwrap_or_else(|| command.trim().to_string());
    let first = Path::new(&first)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or(first.as_str());
    matches!(first, "less" | "more" | "most" | "lv")
}

pub fn with_explicit_pager_width(command: &str, terminal_width: usize) -> String {
    let Ok(parts) = shell_words(command) else {
        return command.to_string();
    };
    let Some(first) = parts.first() else {
        return command.to_string();
    };
    let executable = Path::new(first)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or(first.as_str());
    if executable != "delta"
        || parts
            .iter()
            .any(|part| part == "--width" || part.starts_with("--width="))
    {
        return command.to_string();
    }

    format!("{command} --width={terminal_width}")
}

pub fn run_git_diff_pager(
    repo_root: &Path,
    pager_command: &str,
    diff: &[u8],
    terminal_width: usize,
) -> Result<String> {
    let pager_command = with_explicit_pager_width(pager_command, terminal_width);
    let mut pager = std::process::Command::new("sh")
        .arg("-lc")
        .arg(&pager_command)
        .current_dir(repo_root)
        .env("COLUMNS", terminal_width.to_string())
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .with_context(|| format!("failed to launch configured git pager '{pager_command}'"))?;

    if let Some(mut stdin) = pager.stdin.take() {
        stdin
            .write_all(diff)
            .with_context(|| format!("failed to write diff to git pager '{pager_command}'"))?;
    }

    let output = pager
        .wait_with_output()
        .with_context(|| format!("failed to read output from git pager '{pager_command}'"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(anyhow!(
            "git pager failed{}",
            if stderr.is_empty() {
                String::new()
            } else {
                format!(": {stderr}")
            }
        ));
    }

    String::from_utf8(output.stdout).context("git pager output was not UTF-8")
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

pub fn log_output(workspace: &Path) -> Result<String> {
    let repo_root = discover_git_root(workspace)
        .ok_or_else(|| anyhow!("log is only available inside a Git repository"))?;
    if let Some(output) = try_gh_log(&repo_root)? {
        return Ok(output);
    }
    git_log(&repo_root)
}

pub fn try_gh_log(_repo_root: &Path) -> Result<Option<String>> {
    Ok(None)
}

pub fn git_log(repo_root: &Path) -> Result<String> {
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
    if log.trim().is_empty() {
        Ok("No commits yet.".to_string())
    } else {
        Ok(log)
    }
}

pub fn pull_request_output(workspace: &Path, pr_number: u64) -> Result<String> {
    let repo_root = discover_git_root(workspace)
        .ok_or_else(|| anyhow!("pull is only available inside a Git repository"))?;
    if let Some(output) = try_gh_pr_checkout(&repo_root, pr_number)? {
        return Ok(output);
    }
    git_pr_checkout(&repo_root, pr_number)
}

pub fn try_gh_pr_checkout(repo_root: &Path, pr_number: u64) -> Result<Option<String>> {
    let output = match std::process::Command::new("gh")
        .args(["pr", "checkout", &pr_number.to_string()])
        .current_dir(repo_root)
        .output()
    {
        Ok(output) => output,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err).context("failed to run gh"),
    };
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(anyhow!(
            "gh pr checkout failed{}",
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
        format!("Checked out pull request #{pr_number}")
    } else {
        combined
    }))
}

pub fn git_pr_checkout(repo_root: &Path, pr_number: u64) -> Result<String> {
    let branch = format!("pr-{pr_number}");
    let fetch = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args([
            "fetch",
            "origin",
            "--force",
            &format!("pull/{pr_number}/head:{branch}"),
        ])
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
    let checkout = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["checkout", &branch])
        .output()
        .context("failed to run git checkout")?;
    if !checkout.status.success() {
        let stderr = String::from_utf8_lossy(&checkout.stderr).trim().to_string();
        return Err(anyhow!(
            "git checkout failed{}",
            if stderr.is_empty() {
                String::new()
            } else {
                format!(": {stderr}")
            }
        ));
    }
    let mut parts = Vec::new();
    let fetch_stderr = String::from_utf8_lossy(&fetch.stderr).trim().to_string();
    if !fetch_stderr.is_empty() {
        parts.push(fetch_stderr);
    }
    let checkout_stderr = String::from_utf8_lossy(&checkout.stderr).trim().to_string();
    if !checkout_stderr.is_empty() {
        parts.push(checkout_stderr);
    }
    Ok(if parts.is_empty() {
        format!("Switched to branch '{branch}'")
    } else {
        parts.join("\n")
    })
}

pub fn rebase_output(workspace: &Path) -> Result<String> {
    let repo_root = discover_git_root(workspace)
        .ok_or_else(|| anyhow!("rebase is only available inside a Git repository"))?;
    if let Some(output) = try_gh_rebase(&repo_root)? {
        return Ok(output);
    }
    git_rebase_main(&repo_root)
}

pub fn try_gh_rebase(repo_root: &Path) -> Result<Option<String>> {
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

pub fn merge_output(workspace: &Path, branch: &str) -> Result<String> {
    let repo_root = discover_git_root(workspace)
        .ok_or_else(|| anyhow!("merge is only available inside a Git repository"))?;
    if let Some(output) = try_gh_merge(&repo_root, branch)? {
        return Ok(output);
    }
    git_merge(&repo_root, branch)
}

pub fn try_gh_merge(repo_root: &Path, branch: &str) -> Result<Option<String>> {
    let output = match std::process::Command::new("gh")
        .args(["pr", "merge", "--merge", branch])
        .current_dir(repo_root)
        .output()
    {
        Ok(output) => output,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err).context("failed to run gh"),
    };
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(anyhow!(
            "gh pr merge failed{}",
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
        .args(["merge", branch])
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

pub fn checkout_output(workspace: &Path, target: &str) -> Result<String> {
    let repo_root = discover_git_root(workspace)
        .ok_or_else(|| anyhow!("checkout is only available inside a Git repository"))?;
    if let Some(output) = try_gh_checkout(&repo_root, target)? {
        return Ok(output);
    }
    git_checkout(&repo_root, target)
}

pub fn try_gh_checkout(_repo_root: &Path, _target: &str) -> Result<Option<String>> {
    Ok(None)
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

pub fn push_output(workspace: &Path, force: bool) -> Result<String> {
    let repo_root = discover_git_root(workspace)
        .ok_or_else(|| anyhow!("push is only available inside a Git repository"))?;
    if let Some(output) = try_gh_push(&repo_root, force)? {
        return Ok(output);
    }
    git_push(&repo_root, force)
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

fn git_find_base_ref(repo_root: &Path) -> Result<String> {
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

pub fn delete_branch_output(workspace: &Path, branch: &str) -> Result<String> {
    let repo_root = discover_git_root(workspace)
        .ok_or_else(|| anyhow!("delete is only available inside a Git repository"))?;
    if is_protected_branch(branch) {
        return Err(anyhow!("deleting the '{}' branch is not allowed", branch));
    }
    if let Some(output) = try_gh_delete_branch(&repo_root, branch)? {
        return Ok(output);
    }
    git_delete_branch(&repo_root, branch)
}

pub fn try_gh_delete_branch(_repo_root: &Path, _branch: &str) -> Result<Option<String>> {
    Ok(None)
}

pub fn git_delete_branch(repo_root: &Path, branch: &str) -> Result<String> {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["branch", "-D", branch])
        .output()
        .context("failed to run git branch -D")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let detail = [stdout, stderr]
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        return Err(anyhow!(
            "git branch -D failed{}",
            if detail.is_empty() {
                String::new()
            } else {
                format!(": {detail}")
            }
        ));
    }
    Ok(format!("Deleted branch '{branch}'"))
}

/// Test helper: initialize a git repo with test user config.
#[cfg(test)]
pub fn init_git_for_test(workspace: &Path) {
    assert!(
        std::process::Command::new("git")
            .args(["init", "--quiet"])
            .current_dir(workspace)
            .status()
            .expect("git init")
            .success()
    );
    assert!(
        std::process::Command::new("git")
            .args(["config", "user.name", "Orangu Tests"])
            .current_dir(workspace)
            .status()
            .expect("git config name")
            .success()
    );
    assert!(
        std::process::Command::new("git")
            .args(["config", "user.email", "tests@example.com"])
            .current_dir(workspace)
            .status()
            .expect("git config email")
            .success()
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::process_env_lock;
    use tempfile::tempdir;

    struct EnvVarGuard {
        key: &'static str,
        original: Option<std::ffi::OsString>,
    }

    impl EnvVarGuard {
        fn set_path(key: &'static str, value: &Path) -> Self {
            let original = std::env::var_os(key);
            unsafe {
                std::env::set_var(key, value);
            }
            Self { key, original }
        }

        fn set_value(key: &'static str, value: &str) -> Self {
            let original = std::env::var_os(key);
            unsafe {
                std::env::set_var(key, value);
            }
            Self { key, original }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            unsafe {
                match &self.original {
                    Some(value) => std::env::set_var(self.key, value),
                    None => std::env::remove_var(self.key),
                }
            }
        }
    }

    #[test]
    fn colorizes_git_status_output() {
        use super::super::render::{ANSI_FG_LIGHT_GREEN, ANSI_FG_LIGHT_RED};
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
    fn force_push_blocked_on_protected_branches() {
        assert!(is_protected_branch("main"));
        assert!(is_protected_branch("master"));
        assert!(!is_protected_branch("feature/my-branch"));
        assert!(!is_protected_branch("develop"));
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
    fn delete_branch_blocked_on_protected_branches() {
        let workspace = tempdir().expect("workspace");
        std::process::Command::new("git")
            .arg("init")
            .current_dir(workspace.path())
            .output()
            .expect("git init");
        for branch in ["main", "master"] {
            let result = delete_branch_output(workspace.path(), branch);
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
    fn discovers_git_branch_name_from_workspace() {
        let workspace = tempdir().expect("workspace");
        std::fs::create_dir(workspace.path().join(".git")).expect("git dir");
        std::fs::write(workspace.path().join(".git/HEAD"), "ref: refs/heads/main\n").expect("head");

        assert_eq!(
            workspace_branch_name(workspace.path()).as_deref(),
            Some("main")
        );
        assert_eq!(
            discover_git_root(workspace.path()).as_deref(),
            Some(workspace.path())
        );
        assert_eq!(
            discover_git_dir(workspace.path()).as_deref(),
            Some(workspace.path().join(".git").as_path())
        );
    }

    #[test]
    fn git_workspace_diff_is_colorized_and_unified() {
        let _env_lock = process_env_lock().lock().unwrap_or_else(|p| p.into_inner());
        let workspace = tempdir().expect("workspace");
        let home = tempdir().expect("home");
        let _home_guard = EnvVarGuard::set_path("HOME", home.path());
        init_git_for_test(workspace.path());
        std::fs::write(workspace.path().join("README.md"), "one\ntwo\nthree\n")
            .expect("write file");
        assert!(
            std::process::Command::new("git")
                .args(["add", "README.md"])
                .current_dir(workspace.path())
                .status()
                .expect("git add")
                .success()
        );
        assert!(
            std::process::Command::new("git")
                .args(["commit", "-m", "initial"])
                .current_dir(workspace.path())
                .status()
                .expect("git commit")
                .success()
        );

        std::fs::write(
            workspace.path().join("README.md"),
            "one\nchanged\nthree\nfour\n",
        )
        .expect("update file");

        let diff = git_workspace_diff(workspace.path()).expect("git diff");
        assert!(diff.contains("\u{1b}["));
        assert!(diff.contains("@@"));
        assert!(diff.contains("diff --git"));
        assert!(diff.contains("changed"));
        assert!(diff.contains("four"));
    }

    #[test]
    fn git_workspace_diff_honors_global_gitconfig() {
        let _env_lock = process_env_lock().lock().unwrap_or_else(|p| p.into_inner());
        let workspace = tempdir().expect("workspace");
        init_git_for_test(workspace.path());
        std::fs::write(workspace.path().join("README.md"), "one\ntwo\nthree\n")
            .expect("write file");
        assert!(
            std::process::Command::new("git")
                .args(["add", "README.md"])
                .current_dir(workspace.path())
                .status()
                .expect("git add")
                .success()
        );
        assert!(
            std::process::Command::new("git")
                .args(["commit", "-m", "initial"])
                .current_dir(workspace.path())
                .status()
                .expect("git commit")
                .success()
        );

        std::fs::write(
            workspace.path().join("README.md"),
            "one\nchanged\nthree\nfour\n",
        )
        .expect("update file");

        let home = tempdir().expect("home");
        std::fs::write(
            home.path().join(".gitconfig"),
            "[diff]\n\tnoprefix = true\n",
        )
        .expect("gitconfig");
        let _home_guard = EnvVarGuard::set_path("HOME", home.path());

        let diff = git_workspace_diff(workspace.path()).expect("git diff");
        assert!(diff.contains("diff --git README.md README.md"));
        assert!(diff.contains("--- README.md"));
        assert!(diff.contains("+++ README.md"));
        assert!(!diff.contains("diff --git a/README.md b/README.md"));
    }

    #[test]
    fn git_workspace_diff_uses_configured_noninteractive_pager() {
        let _env_lock = process_env_lock().lock().unwrap_or_else(|p| p.into_inner());
        let workspace = tempdir().expect("workspace");
        init_git_for_test(workspace.path());
        std::fs::write(workspace.path().join("README.md"), "one\ntwo\nthree\n")
            .expect("write file");
        assert!(
            std::process::Command::new("git")
                .args(["add", "README.md"])
                .current_dir(workspace.path())
                .status()
                .expect("git add")
                .success()
        );
        assert!(
            std::process::Command::new("git")
                .args(["commit", "-m", "initial"])
                .current_dir(workspace.path())
                .status()
                .expect("git commit")
                .success()
        );

        std::fs::write(
            workspace.path().join("README.md"),
            "one\nchanged\nthree\nfour\n",
        )
        .expect("update file");

        let home = tempdir().expect("home");
        let pager = home.path().join("pager.sh");
        std::fs::write(
            &pager,
            "#!/bin/sh\nprintf 'PAGER-START WIDTH=%s\\n' \"$COLUMNS\"\ncat\nprintf 'PAGER-END\\n'\n",
        )
        .expect("pager script");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut permissions = std::fs::metadata(&pager)
                .expect("pager metadata")
                .permissions();
            permissions.set_mode(0o755);
            std::fs::set_permissions(&pager, permissions).expect("pager permissions");
        }
        std::fs::write(
            home.path().join(".gitconfig"),
            format!("[core]\n\tpager = {}\n", pager.display()),
        )
        .expect("gitconfig");
        let _home_guard = EnvVarGuard::set_path("HOME", home.path());
        let _columns_guard = EnvVarGuard::set_value("COLUMNS", "123");

        let diff = git_workspace_diff(workspace.path()).expect("git diff");
        assert!(diff.contains("PAGER-START WIDTH="));
        assert!(diff.contains("diff --git"));
        assert!(diff.ends_with("PAGER-END\n"));
    }

    #[test]
    fn adds_explicit_width_to_delta_pager_command() {
        assert_eq!(
            with_explicit_pager_width("delta --side-by-side", 123),
            "delta --side-by-side --width=123"
        );
        assert_eq!(
            with_explicit_pager_width("/usr/bin/delta --width=90 --side-by-side", 123),
            "/usr/bin/delta --width=90 --side-by-side"
        );
        assert_eq!(with_explicit_pager_width("less -FRX", 123), "less -FRX");
    }
}
