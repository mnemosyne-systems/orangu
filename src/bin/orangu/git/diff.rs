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

/// Show a single commit (`git show <commit>`), defaulting to `HEAD` when no
/// commit is given. The output is a commit header followed by its diff, rendered
/// like the `/diff` tool: colorized and piped through the configured
/// non-interactive `show` pager (falling back to `core.pager`, e.g. `delta`)
/// when one is set.
pub fn show_output(workspace: &Path, commit: Option<&str>) -> Result<String> {
    let Some(repo_root) = discover_git_root(workspace) else {
        return workspace_is_not_git(workspace);
    };
    let terminal_width = current_terminal_width();

    let mut command = std::process::Command::new("git");
    command
        .arg("-C")
        .arg(&repo_root)
        .arg("show")
        .arg("--color=always");
    if let Some(commit) = commit {
        command.arg(commit);
    }
    command.env("COLUMNS", terminal_width.to_string());

    let output = command.output().context("failed to run git show")?;
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

    if let Some(pager_command) = configured_git_pager(&repo_root, "show")? {
        run_git_diff_pager(&repo_root, &pager_command, &output.stdout, terminal_width)
    } else {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
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

/// The lines a non-default branch adds versus its merge base with main/master,
/// grouped by workspace-relative file. Used by `/duplicates` to restrict the
/// analysis to the functions a branch introduces or changes.
pub struct BranchChanges {
    /// The base ref the diff was taken against (e.g. `origin/main`).
    pub base: String,
    /// Each changed file under the workspace, with the 1-based inclusive line
    /// ranges added on the branch (new-file coordinates).
    pub files: Vec<(std::path::PathBuf, Vec<(usize, usize)>)>,
}

/// Compute the [`BranchChanges`] for `workspace`, or `None` when the analysis
/// should run against the whole project instead: outside a Git repository, when
/// no base branch (main/master) is found, when `HEAD` is detached, or when the
/// current branch *is* the default branch. Any git error also yields `None`.
pub fn branch_added_lines(workspace: &Path) -> Option<BranchChanges> {
    let repo_root = discover_git_root(workspace)?;
    let base_ref = git_find_base_ref(&repo_root).ok()?;
    let current = workspace_branch_name(workspace)?;
    if current.is_empty() {
        return None;
    }
    // `origin/main` -> `main`; on the default branch we want the whole project.
    let default_name = base_ref.rsplit('/').next().unwrap_or(&base_ref);
    if current == default_name {
        return None;
    }

    let merge_base = git_merge_base(&repo_root, &base_ref).unwrap_or_else(|| base_ref.clone());
    // `--unified=0` so each hunk's `+` range is exactly the added lines.
    let diff = run_git_diff_capture(&repo_root, &["--unified=0", &merge_base]).ok()?;

    let mut files: Vec<(std::path::PathBuf, Vec<(usize, usize)>)> = Vec::new();
    let mut tracked = false;
    for line in diff.lines() {
        if let Some(path) = line.strip_prefix("+++ b/") {
            tracked = match repo_relative_under_workspace(&repo_root, workspace, path.trim()) {
                Some(relative) => {
                    files.push((relative, Vec::new()));
                    true
                }
                None => false,
            };
        } else if line.starts_with("+++ ") {
            // `+++ /dev/null` (a deletion) or an unparseable header.
            tracked = false;
        } else if let Some(hunk) = line.strip_prefix("@@ ")
            && tracked
            && let Some(range) = parse_added_range(hunk)
            && let Some((_, ranges)) = files.last_mut()
        {
            ranges.push(range);
        }
    }
    files.retain(|(_, ranges)| !ranges.is_empty());
    // A branch that adds nothing (e.g. rebased onto the base with no commits on
    // top and a clean tree) has nothing new to analyse, so fall back to a
    // whole-project report rather than an empty patch one.
    if files.is_empty() {
        return None;
    }
    Some(BranchChanges {
        base: base_ref,
        files,
    })
}

/// Map a repository-relative diff path to a workspace-relative path, or `None`
/// when the file lies outside the scanned workspace (so it is ignored).
fn repo_relative_under_workspace(
    repo_root: &Path,
    workspace: &Path,
    repo_relative: &str,
) -> Option<std::path::PathBuf> {
    repo_root
        .join(repo_relative)
        .strip_prefix(workspace)
        .ok()
        .map(Path::to_path_buf)
}

/// Parse a unified-diff hunk header's `+` side (the text after `@@ `) into the
/// 1-based inclusive line range it adds, e.g. `-1,0 +5,3 @@` -> `(5, 7)`. `None`
/// for a pure deletion (`+c,0`) or an unparseable header.
fn parse_added_range(hunk: &str) -> Option<(usize, usize)> {
    let plus = hunk
        .split_whitespace()
        .find(|token| token.starts_with('+'))?;
    let numbers = &plus[1..];
    let (start, count) = match numbers.split_once(',') {
        Some((start, count)) => (start.parse().ok()?, count.parse().ok()?),
        None => (numbers.parse().ok()?, 1usize),
    };
    if count == 0 {
        return None;
    }
    Some((start, start + count - 1))
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
    if executable != "delta" {
        return command.to_string();
    }
    let has_width = parts
        .iter()
        .any(|part| part == "--width" || part.starts_with("--width="));
    let has_mode = parts
        .iter()
        .any(|part| part == "--light" || part == "--dark");

    let mut modified = command.to_string();
    if !has_width {
        modified = format!("{modified} --width={terminal_width}");
    }
    if !has_mode {
        let mode = if orangu::tui::Theme::is_dark() {
            "--dark"
        } else {
            "--light"
        };
        modified = format!("{modified} {mode}");
    }

    modified
}

pub fn run_git_diff_pager(
    repo_root: &Path,
    pager_command: &str,
    diff: &[u8],
    terminal_width: usize,
) -> Result<String> {
    let pager_command = with_explicit_pager_width(pager_command, terminal_width);

    let bat_theme = if orangu::tui::Theme::is_dark() {
        "base16-ocean.dark"
    } else {
        "base16-ocean.light"
    };

    let mut pager = std::process::Command::new("sh")
        .arg("-c")
        .arg(&pager_command)
        .current_dir(repo_root)
        .env("COLUMNS", terminal_width.to_string())
        .env("BAT_THEME", bat_theme)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .with_context(|| format!("failed to launch configured git pager '{pager_command}'"))?;

    // Feed the diff on a separate thread so the pager's stdout (and stderr) can
    // be drained concurrently. Writing the whole diff up front and only reading
    // stdout afterwards deadlocks once the pager's output fills the OS pipe
    // buffer (~64 KiB): the pager blocks writing stdout while we block writing
    // stdin. `delta` with side-by-side/line-numbers easily exceeds that on a
    // large file, which hung `/review`.
    let writer = pager.stdin.take().map(|mut stdin| {
        let diff = diff.to_vec();
        std::thread::spawn(move || stdin.write_all(&diff))
    });

    let output = pager
        .wait_with_output()
        .with_context(|| format!("failed to read output from git pager '{pager_command}'"))?;

    // The pager has exited, so the writer thread has unblocked. Surface a write
    // failure only when the pager itself reported success — otherwise its
    // stderr (handled below) is the more informative error, and a broken pipe
    // from the pager exiting early is expected.
    if let Some(writer) = writer {
        match writer.join() {
            Ok(Ok(())) => {}
            Ok(Err(err)) if output.status.success() => {
                return Err(err).with_context(|| {
                    format!("failed to write diff to git pager '{pager_command}'")
                });
            }
            Ok(Err(_)) => {}
            Err(_) => return Err(anyhow!("git pager writer thread panicked")),
        }
    }

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
    fn parse_added_range_reads_the_plus_side() {
        // `+c,d` -> c..=c+d-1; a bare `+c` is one line; `+c,0` (pure deletion)
        // and malformed headers yield None.
        assert_eq!(parse_added_range("-1,0 +5,3 @@ fn f()"), Some((5, 7)));
        assert_eq!(parse_added_range("-10 +12 @@"), Some((12, 12)));
        assert_eq!(parse_added_range("-3,2 +4,0 @@"), None);
        assert_eq!(parse_added_range("nonsense"), None);
    }

    #[test]
    fn branch_added_lines_falls_back_when_branch_adds_nothing() {
        let _env_lock = process_env_lock().lock().unwrap_or_else(|p| p.into_inner());
        let workspace = tempdir().expect("workspace");
        let home = tempdir().expect("home");
        let _home_guard = EnvVarGuard::set_path("HOME", home.path());
        init_git_for_test(workspace.path());

        // A base commit on main.
        git_run(workspace.path(), &["checkout", "-B", "main"]);
        std::fs::write(
            workspace.path().join("lib.rs"),
            "fn alpha() -> i32 {\n    let mut t = 0;\n    t += 1;\n    t\n}\n",
        )
        .expect("write base");
        git_run(workspace.path(), &["add", "."]);
        git_run(workspace.path(), &["commit", "-m", "base"]);

        // A branch with no commits on top of main adds nothing — so the scan
        // should run against the whole project (`None`), not as a patch.
        git_run(workspace.path(), &["checkout", "-b", "feature/x"]);
        assert!(branch_added_lines(workspace.path()).is_none());

        // Once the branch commits a new function, it has added lines again.
        std::fs::write(
            workspace.path().join("lib.rs"),
            "fn alpha() -> i32 {\n    let mut t = 0;\n    t += 1;\n    t\n}\n\n\
             fn beta() -> i32 {\n    let mut s = 0;\n    s += 2;\n    s\n}\n",
        )
        .expect("write change");
        git_run(workspace.path(), &["add", "."]);
        git_run(workspace.path(), &["commit", "-m", "add beta"]);
        let changes = branch_added_lines(workspace.path()).expect("branch changes");
        assert_eq!(changes.base, "main");
        assert!(
            changes
                .files
                .iter()
                .any(|(path, _)| path == std::path::Path::new("lib.rs"))
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
    fn large_diff_through_a_pager_does_not_deadlock() {
        // Regression: the pager's stdin used to be written in full before its
        // stdout was read. A diff that makes the pager emit more than the OS
        // pipe buffer (~64 KiB) before we finish writing deadlocked both ends,
        // which hung `/review`. The writer now runs on its own thread. This
        // test would hang (not just fail) without the fix.
        let _env_lock = process_env_lock().lock().unwrap_or_else(|p| p.into_inner());
        let workspace = tempdir().expect("workspace");
        let home = tempdir().expect("home");
        let _home_guard = EnvVarGuard::set_path("HOME", home.path());
        // `cat` is a non-interactive pass-through pager, so the pager's output
        // is as large as the (large) diff fed to it — well past the pipe buffer.
        std::fs::write(home.path().join(".gitconfig"), "[core]\n\tpager = cat\n")
            .expect("gitconfig");
        init_git_for_test(workspace.path());

        // A file large enough that its diff dwarfs the pipe buffer.
        let original: String = (0..8000).map(|n| format!("original line {n}\n")).collect();
        std::fs::write(workspace.path().join("big.txt"), &original).expect("write big file");
        assert!(
            std::process::Command::new("git")
                .args(["add", "big.txt"])
                .current_dir(workspace.path())
                .status()
                .expect("git add")
                .success()
        );
        assert!(
            std::process::Command::new("git")
                .args(["commit", "-m", "add big file"])
                .current_dir(workspace.path())
                .status()
                .expect("git commit")
                .success()
        );
        let changed: String = (0..8000).map(|n| format!("changed line {n}\n")).collect();
        std::fs::write(workspace.path().join("big.txt"), &changed).expect("update big file");

        let diff = git_workspace_diff(workspace.path()).expect("git diff");
        // The whole diff made it through the pager (both removed and added
        // lines), so nothing was lost to a stalled pipe.
        assert!(diff.contains("original line 0"));
        assert!(diff.contains("changed line 7999"));
    }

    #[test]
    fn adds_explicit_width_to_delta_pager_command() {
        assert_eq!(
            with_explicit_pager_width("delta --side-by-side", 123),
            "delta --side-by-side --width=123 --dark"
        );
        assert_eq!(
            with_explicit_pager_width("/usr/bin/delta --width=90 --side-by-side", 123),
            "/usr/bin/delta --width=90 --side-by-side --dark"
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
