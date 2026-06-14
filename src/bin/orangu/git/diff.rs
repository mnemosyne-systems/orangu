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
use std::{io::Write, path::Path};

use super::*;
use crate::commands::{current_terminal_width, shell_words};

pub fn git_workspace_diff(workspace: &Path) -> Result<String> {
    let Some(repo_root) = discover_git_root(workspace) else {
        return workspace_is_not_git(workspace);
    };
    let terminal_width = current_terminal_width();
    let workspace_pathspec = workspace
        .strip_prefix(&repo_root)
        .ok()
        .filter(|path| !path.as_os_str().is_empty());

    let mut command = colorized_git_diff_command(&repo_root);
    if let Some(pathspec) = workspace_pathspec {
        command.arg("--").arg(pathspec);
    }

    let diff = render_git_diff(&repo_root, command, terminal_width)?;
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

    let mut command = colorized_git_diff_command(&repo_root);
    command.arg(format!("{branch}...HEAD"));

    let diff = render_git_diff(&repo_root, command, terminal_width)?;
    if diff.trim().is_empty() {
        Ok(format!("No changes against {branch}."))
    } else {
        Ok(diff)
    }
}

/// Start a `git -C <root> diff --color=always` command; callers append the
/// range and pathspec they need.
fn colorized_git_diff_command(repo_root: &Path) -> std::process::Command {
    let mut command = std::process::Command::new("git");
    command
        .arg("-C")
        .arg(repo_root)
        .arg("diff")
        .arg("--color=always");
    command
}

/// Run a prepared `git diff` command and render it exactly as the `/diff` tool
/// does: pipe the colorized output through the configured non-interactive git
/// pager (e.g. `delta`) when one is set, otherwise return the raw colorized
/// diff.
fn render_git_diff(
    repo_root: &Path,
    mut command: std::process::Command,
    terminal_width: usize,
) -> Result<String> {
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

    if let Some(pager_command) = configured_git_diff_pager(repo_root)? {
        run_git_diff_pager(repo_root, &pager_command, &output.stdout, terminal_width)
    } else {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }
}

/// The combined diff for `/review`: local (uncommitted) changes plus the
/// changes committed on the current branch, measured against the merge base
/// with the default branch (main/master). Returns the colorized diff split
/// into lines, plus the location of each file's section within those lines.
pub struct ReviewDiff {
    pub base_label: String,
    pub files: Vec<ReviewFileDiff>,
}

pub struct ReviewFileDiff {
    pub path: String,
    /// Colorized diff lines for display (configured pager applied).
    pub lines: Vec<String>,
    /// Plain unified diff (no color/pager) suitable for sending to the LLM.
    pub patch: String,
}

pub fn collect_review_diff(workspace: &Path) -> Result<ReviewDiff> {
    let repo_root = discover_git_root(workspace)
        .ok_or_else(|| anyhow!("review is only available inside a Git repository"))?;
    let base_ref = git_find_base_ref(&repo_root)?;

    // Diff against the merge base so we show what this branch adds (committed
    // and uncommitted) without reverse-diffing commits the base has but we do
    // not. Fall back to the base ref itself if no merge base can be found.
    let merge_base = git_merge_base(&repo_root, &base_ref).unwrap_or_else(|| base_ref.clone());

    // The changed files, in git's diff order.
    let names: Vec<String> = run_git_diff_capture(&repo_root, &["--name-only", &merge_base])?
        .lines()
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .collect();

    // Render each file's diff separately through the same pipeline as the
    // `/diff` tool (configured pager included). Each file keeps its own lines so
    // the review view can show just the selected file's diff.
    let terminal_width = current_terminal_width();
    let mut files: Vec<ReviewFileDiff> = Vec::new();
    for path in names {
        let mut command = colorized_git_diff_command(&repo_root);
        command.arg(&merge_base).arg("--").arg(&path);
        let rendered = render_git_diff(&repo_root, command, terminal_width)?;

        let mut lines: Vec<String> = rendered
            .split('\n')
            .map(|line| line.strip_suffix('\r').unwrap_or(line).to_string())
            .collect();
        // Drop a trailing empty line produced by the final newline.
        if matches!(lines.last(), Some(last) if last.is_empty()) {
            lines.pop();
        }
        if lines.is_empty() {
            continue;
        }

        // A plain (uncolored, unpaged) patch for the file, for the LLM prompt.
        let patch = run_git_diff_capture(&repo_root, &[&merge_base, "--", &path])?;

        files.push(ReviewFileDiff { path, lines, patch });
    }

    Ok(ReviewDiff {
        base_label: base_ref,
        files,
    })
}

/// The repo-relative paths changed on the current branch against the default
/// branch's merge base — the files a single-file `/auto_review` can target on a
/// non-default branch, and the candidates its Tab completion offers there.
/// Returns an empty list outside a repository or on any git error, so
/// completion degrades quietly.
pub fn review_changed_paths(workspace: &Path) -> Vec<String> {
    let Some(repo_root) = discover_git_root(workspace) else {
        return Vec::new();
    };
    let Ok(base_ref) = git_find_base_ref(&repo_root) else {
        return Vec::new();
    };
    let merge_base = git_merge_base(&repo_root, &base_ref).unwrap_or(base_ref);
    run_git_diff_capture(&repo_root, &["--name-only", &merge_base])
        .map(|output| {
            output
                .lines()
                .map(str::trim)
                .filter(|name| !name.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn git_merge_base(repo_root: &Path, base_ref: &str) -> Option<String> {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["merge-base", base_ref, "HEAD"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let merge_base = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if merge_base.is_empty() {
        None
    } else {
        Some(merge_base)
    }
}

fn run_git_diff_capture(repo_root: &Path, args: &[&str]) -> Result<String> {
    let mut command = std::process::Command::new("git");
    command.arg("-C").arg(repo_root).arg("diff");
    command.args(args);
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
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

/// Look up the non-interactive pager configured for a specific git subcommand.
/// Checks `pager.<subcommand>` first, then falls back to `core.pager`.
/// Returns `None` when no pager is configured or the configured pager is an
/// interactive one (less, more, …) that cannot be used non-interactively.
pub fn configured_git_pager(repo_root: &Path, subcommand: &str) -> Result<Option<String>> {
    let pager_key = format!("pager.{subcommand}");
    for key in [pager_key.as_str(), "core.pager"] {
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

pub fn configured_git_diff_pager(repo_root: &Path) -> Result<Option<String>> {
    configured_git_pager(repo_root, "diff")
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
        .arg("-c")
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::process_env_lock;
    use tempfile::tempdir;

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

    #[test]
    fn collect_review_diff_reports_files_and_local_changes() {
        let _env_lock = process_env_lock().lock().unwrap_or_else(|p| p.into_inner());
        let workspace = tempdir().expect("workspace");
        let home = tempdir().expect("home");
        let _home_guard = EnvVarGuard::set_path("HOME", home.path());
        init_git_for_test(workspace.path());

        // Base commit on main.
        std::process::Command::new("git")
            .args(["checkout", "-B", "main"])
            .current_dir(workspace.path())
            .status()
            .expect("git checkout -B main");
        std::fs::write(workspace.path().join("base.txt"), "base\n").expect("write base");
        std::process::Command::new("git")
            .args(["add", "."])
            .current_dir(workspace.path())
            .status()
            .expect("git add base");
        std::process::Command::new("git")
            .args(["commit", "-m", "Base commit"])
            .current_dir(workspace.path())
            .status()
            .expect("git commit base");

        // Feature branch with a committed change.
        std::process::Command::new("git")
            .args(["checkout", "-b", "feature/review-test"])
            .current_dir(workspace.path())
            .status()
            .expect("git checkout -b feature");
        std::fs::write(workspace.path().join("committed.txt"), "committed\n")
            .expect("write committed");
        std::process::Command::new("git")
            .args(["add", "."])
            .current_dir(workspace.path())
            .status()
            .expect("git add committed");
        std::process::Command::new("git")
            .args(["commit", "-m", "Add committed file"])
            .current_dir(workspace.path())
            .status()
            .expect("git commit committed");

        // Uncommitted local change.
        std::fs::write(workspace.path().join("local.txt"), "local\n").expect("write local");
        std::process::Command::new("git")
            .args(["add", "."])
            .current_dir(workspace.path())
            .status()
            .expect("git add local");

        let review = collect_review_diff(workspace.path()).expect("collect review");
        assert_eq!(review.base_label, "main");

        let paths: Vec<&str> = review.files.iter().map(|f| f.path.as_str()).collect();
        assert!(
            paths.contains(&"committed.txt"),
            "expected committed file in {paths:?}"
        );
        assert!(
            paths.contains(&"local.txt"),
            "expected local change in {paths:?}"
        );

        // Each file carries its own diff. With no configured pager (the test
        // environment), each block starts with the colorized `diff --git`
        // header for that file.
        for file in &review.files {
            assert!(!file.lines.is_empty(), "no diff lines for {}", file.path);
            assert!(
                file.lines[0].contains("diff --git"),
                "first line for {} is not a header: {}",
                file.path,
                file.lines[0]
            );
        }
    }
}
