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
    process::Stdio,
};

use super::commands::{current_terminal_width, shell_words};
use super::render::{ANSI_FG_LIGHT_GREEN, ANSI_FG_LIGHT_RED, ANSI_FG_RESET, ANSI_FG_SUBTLE};
use orangu::tools::resolve_workspace_path;

/// A code-hosting platform whose CLI orangu drives for pull/merge-request and
/// issue operations. `gh` for GitHub, `glab` for GitLab.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Forge {
    GitHub,
    GitLab,
}

impl Forge {
    /// Resolve a `[orangu].platform` value into a forge. Unknown values fall
    /// back to GitHub, matching the configuration default.
    pub fn from_platform(platform: &str) -> Forge {
        match platform.trim().to_lowercase().as_str() {
            "gitlab" | "glab" => Forge::GitLab,
            _ => Forge::GitHub,
        }
    }

    /// The command-line tool that talks to this forge.
    pub fn cli(self) -> &'static str {
        match self {
            Forge::GitHub => "gh",
            Forge::GitLab => "glab",
        }
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

/// Resolve the `$EDITOR` invocation for a workspace file into the program, its
/// leading arguments, and the absolute target path.
pub fn resolve_editor_command(
    workspace: &Path,
    raw_path: &str,
) -> Result<(String, Vec<String>, PathBuf)> {
    let editor = std::env::var("EDITOR").context("EDITOR is not set")?;
    let editor_parts = shell_words(&editor)?;
    let path = resolve_workspace_path(workspace, raw_path)?;
    let (program, args) = editor_parts
        .split_first()
        .ok_or_else(|| anyhow!("EDITOR is empty"))?;
    Ok((program.clone(), args.to_vec(), path))
}

/// Whether `$EDITOR` is a terminal editor that needs its own terminal window
/// (vim, nano, `emacs -nw`, …) as opposed to a GUI editor that opens its own
/// window (code, gvim, plain emacs, …).
pub fn editor_needs_terminal(program: &str, args: &[String]) -> bool {
    let name = Path::new(program)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(program);
    let name = name
        .strip_suffix(".exe")
        .unwrap_or(name)
        .to_ascii_lowercase();

    const TERMINAL_EDITORS: &[&str] = &[
        "vi", "vim", "nvim", "nvi", "elvis", "vis", "nano", "pico", "micro", "helix", "hx", "kak",
        "kakoune", "joe", "jed", "ne", "mg", "ed",
    ];
    if TERMINAL_EDITORS.contains(&name.as_str()) {
        return true;
    }
    // emacs/emacsclient open a GUI window by default and only run in the
    // terminal when explicitly asked to.
    if name == "emacs" || name == "emacsclient" {
        return args
            .iter()
            .any(|arg| matches!(arg.as_str(), "-nw" | "-t" | "--tty" | "--no-window-system"));
    }
    false
}

/// Whether an executable named `name` exists on `$PATH`.
fn binary_on_path(name: &str) -> bool {
    let Ok(path) = std::env::var("PATH") else {
        return false;
    };
    std::env::split_paths(&path).any(|dir| dir.join(name).is_file())
}

/// The command prefix used to launch a terminal editor in its own window, e.g.
/// `["gnome-terminal", "--"]`. Uses the configured `terminal` command (a full
/// command, with flags) when non-empty, otherwise auto-detects a known emulator
/// on `$PATH`. The editor command is appended as separate arguments, and these
/// emulators close the window when that command exits.
fn terminal_launcher(configured: &str) -> Option<Vec<String>> {
    if !configured.trim().is_empty() {
        let parts = shell_words(configured).ok()?;
        if !parts.is_empty() {
            return Some(parts);
        }
    }

    // (binary, arguments that introduce the command to run)
    const TERMINALS: &[(&str, &[&str])] = &[
        ("x-terminal-emulator", &["-e"]),
        ("ptyxis", &["--"]),
        ("gnome-terminal", &["--"]),
        ("konsole", &["-e"]),
        ("kitty", &[]),
        ("alacritty", &["-e"]),
        ("wezterm", &["start", "--"]),
        ("foot", &[]),
        ("xfce4-terminal", &["-x"]),
        ("terminator", &["-x"]),
        ("st", &["-e"]),
        ("urxvt", &["-e"]),
        ("rxvt", &["-e"]),
        ("xterm", &["-e"]),
    ];
    for (binary, prefix) in TERMINALS {
        if binary_on_path(binary) {
            let mut launcher = vec![(*binary).to_string()];
            launcher.extend(prefix.iter().map(|arg| (*arg).to_string()));
            return Some(launcher);
        }
    }
    None
}

/// Build the full argument vector for opening `raw_path` in `$EDITOR`. Terminal
/// editors are wrapped in a terminal emulator (the configured `terminal`, or an
/// auto-detected one) so they get their own window; GUI editors are launched
/// directly.
fn editor_launch_argv(workspace: &Path, raw_path: &str, terminal: &str) -> Result<Vec<String>> {
    let (program, args, path) = resolve_editor_command(workspace, raw_path)?;

    let mut argv = Vec::new();
    if editor_needs_terminal(&program, &args) {
        let launcher = terminal_launcher(terminal).ok_or_else(|| {
            anyhow!(
                "no terminal emulator found to open '{program}' in a new window; \
                 set the `terminal` option in orangu.conf (e.g. \"xterm -e\") or use a GUI editor"
            )
        })?;
        argv.extend(launcher);
    }
    argv.push(program);
    argv.extend(args);
    argv.push(path.to_string_lossy().into_owned());
    Ok(argv)
}

/// Open `$EDITOR` on a workspace file in a separate window so orangu stays
/// usable. Terminal editors (vim, nano, `emacs -nw`, …) open in a new terminal
/// window (the configured `terminal` command, or an auto-detected emulator) that
/// closes when the editor exits; GUI editors open their own window. The process
/// is detached and not waited on.
pub fn open_in_editor(workspace: &Path, raw_path: &str, terminal: &str) -> Result<()> {
    let argv = editor_launch_argv(workspace, raw_path, terminal)?;
    let (program, args) = argv
        .split_first()
        .ok_or_else(|| anyhow!("editor command is empty"))?;

    std::process::Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .with_context(|| format!("failed to launch editor '{program}'"))?;
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

pub fn pull_request_output(workspace: &Path, pr_number: u64, forge: Forge) -> Result<Option<String>> {
    let repo_root = discover_git_root(workspace)
        .ok_or_else(|| anyhow!("pull is only available inside a Git repository"))?;
    if try_forge_pr_checkout(&repo_root, pr_number, forge)?.is_none() {
        git_pr_checkout(&repo_root, pr_number)?;
    }
    Ok(pr_sync_advice(&repo_root))
}

/// Build the rebase/squash hint lines for the checked-out branch relative to the
/// default base branch. Each entry is one line such as
/// "branch is 2 commits behind origin/main; run /rebase". Returns an empty `Vec`
/// when the branch is up to date with at most a single commit ahead.
fn pr_sync_notes(repo_root: &Path, base_ref: &str) -> Result<Vec<String>> {
    // Commits on the base branch the branch has not yet incorporated.
    let behind = git_commit_count(repo_root, &format!("HEAD..{base_ref}"))?;
    // Commits the branch adds on top of the merge base.
    let ahead = git_commit_count(repo_root, &format!("{base_ref}..HEAD"))?;

    let mut notes = Vec::new();
    if behind > 0 {
        notes.push(format!(
            "branch is {behind} commit{} behind {base_ref}; run /rebase",
            if behind == 1 { "" } else { "s" }
        ));
    }
    if ahead > 1 {
        notes.push(format!("{ahead} commits ahead of {base_ref}; run /squash"));
    }
    Ok(notes)
}

/// Report whether the checked-out branch would benefit from a rebase and/or
/// squash against the default base branch before it can be merged. Returns
/// `None` when the branch is already up to date with a single commit, or when
/// the base branch cannot be determined.
fn pr_sync_advice(repo_root: &Path) -> Option<String> {
    let base_ref = git_find_base_ref(repo_root).ok()?;
    let notes = pr_sync_notes(repo_root, &base_ref).ok()?;
    if notes.is_empty() {
        return None;
    }
    Some(format!(
        "This pull request needs attention:\n- {}",
        notes.join("\n- ")
    ))
}

pub fn try_forge_pr_checkout(
    repo_root: &Path,
    pr_number: u64,
    forge: Forge,
) -> Result<Option<String>> {
    let cli = forge.cli();
    // `gh pr checkout N` and `glab mr checkout N` are spelled the same way
    // apart from the request noun.
    let request = match forge {
        Forge::GitHub => "pr",
        Forge::GitLab => "mr",
    };
    let output = match std::process::Command::new(cli)
        .args([request, "checkout", &pr_number.to_string()])
        .current_dir(repo_root)
        .output()
    {
        Ok(output) => output,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err).context(format!("failed to run {cli}")),
    };
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(anyhow!(
            "{cli} {request} checkout failed{}",
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

pub fn comment_output(
    workspace: &Path,
    issue_number: u64,
    body: &str,
    forge: Forge,
) -> Result<String> {
    let repo_root = discover_git_root(workspace)
        .ok_or_else(|| anyhow!("comment is only available inside a Git repository"))?;
    let cli = forge.cli();
    let number = issue_number.to_string();
    // GitHub: `gh issue comment N --body B`. GitLab: `glab issue note N --message B`.
    let args: Vec<&str> = match forge {
        Forge::GitHub => vec!["issue", "comment", &number, "--body", body],
        Forge::GitLab => vec!["issue", "note", &number, "--message", body],
    };
    let output = match std::process::Command::new(cli)
        .args(&args)
        .current_dir(&repo_root)
        .output()
    {
        Ok(output) => output,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return Err(anyhow!("comment requires the {cli} CLI to be installed"));
        }
        Err(err) => return Err(err).context(format!("failed to run {cli}")),
    };
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(anyhow!(
            "{cli} issue comment failed{}",
            if stderr.is_empty() {
                String::new()
            } else {
                format!(": {stderr}")
            }
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok(if stdout.is_empty() {
        format!("Added comment on issue #{issue_number}")
    } else {
        stdout
    })
}

pub fn create_pull_request_output(
    workspace: &Path,
    auto_rebase: bool,
    auto_squash: bool,
    forge: Forge,
) -> Result<String> {
    let repo_root = discover_git_root(workspace)
        .ok_or_else(|| anyhow!("pull_request is only available inside a Git repository"))?;
    let current = git_current_branch(&repo_root)?;
    if is_protected_branch(&current) {
        return Err(anyhow!(
            "cannot create a pull request from the '{}' branch",
            current
        ));
    }
    let base_ref = git_find_base_ref(&repo_root)?;
    let ahead = git_commit_count(&repo_root, &format!("{base_ref}..HEAD"))?;
    if ahead == 0 {
        return Err(anyhow!(
            "no commits ahead of {base_ref}; make at least one commit before opening a pull request"
        ));
    }
    // Apply any configured auto-fixes before re-checking the branch state.
    let behind = git_commit_count(&repo_root, &format!("HEAD..{base_ref}"))?;
    if behind > 0 && auto_rebase {
        rebase_output(workspace, forge)?;
    }
    let ahead = git_commit_count(&repo_root, &format!("{base_ref}..HEAD"))?;
    if ahead > 1 && auto_squash {
        squash_output(workspace)?;
    }
    // Anything still outstanding blocks PR creation with the shared hint.
    let notes = pr_sync_notes(&repo_root, &base_ref)?;
    if !notes.is_empty() {
        return Err(anyhow!(
            "This pull request needs attention:\n- {}",
            notes.join("\n- ")
        ));
    }
    try_forge_create_pr(&repo_root, &current, &base_ref, forge)
}

fn git_commit_count(repo_root: &Path, range: &str) -> Result<usize> {
    Ok(String::from_utf8_lossy(
        &std::process::Command::new("git")
            .arg("-C")
            .arg(repo_root)
            .args(["rev-list", "--count", range])
            .output()
            .context("failed to run git rev-list")?
            .stdout,
    )
    .trim()
    .parse()
    .unwrap_or(0))
}

pub fn try_forge_create_pr(
    repo_root: &Path,
    branch: &str,
    base_ref: &str,
    forge: Forge,
) -> Result<String> {
    let full_message = String::from_utf8_lossy(
        &std::process::Command::new("git")
            .arg("-C")
            .arg(repo_root)
            .args(["log", "-1", "--format=%B"])
            .output()
            .context("failed to run git log")?
            .stdout,
    )
    .trim()
    .to_string();
    let (title, body) = match full_message.split_once('\n') {
        Some((subject, rest)) => (subject.trim().to_string(), rest.trim().to_string()),
        None => (full_message.clone(), String::new()),
    };
    if title.is_empty() {
        return Err(anyhow!(
            "commit message is empty; cannot derive a pull request title"
        ));
    }
    let push = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["push", "--set-upstream", "origin", branch])
        .output()
        .context("failed to run git push")?;
    if !push.status.success() {
        let stderr = String::from_utf8_lossy(&push.stderr).trim().to_string();
        return Err(anyhow!(
            "git push failed{}",
            if stderr.is_empty() {
                String::new()
            } else {
                format!(": {stderr}")
            }
        ));
    }
    let base = base_ref.trim_start_matches("origin/");
    let cli = forge.cli();
    // GitHub: `gh pr create --title T --body B --base BASE`.
    // GitLab: `glab mr create --title T --description B --source-branch BRANCH
    //          --target-branch BASE --yes` (`--yes` skips the interactive prompt).
    let args: Vec<&str> = match forge {
        Forge::GitHub => vec![
            "pr", "create", "--title", &title, "--body", &body, "--base", base,
        ],
        Forge::GitLab => vec![
            "mr",
            "create",
            "--title",
            &title,
            "--description",
            &body,
            "--source-branch",
            branch,
            "--target-branch",
            base,
            "--yes",
        ],
    };
    let output = match std::process::Command::new(cli)
        .args(&args)
        .current_dir(repo_root)
        .output()
    {
        Ok(output) => output,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return Err(anyhow!(
                "pull_request requires the {cli} CLI to be installed"
            ));
        }
        Err(err) => return Err(err).context(format!("failed to run {cli}")),
    };
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let request = if forge == Forge::GitHub { "pr" } else { "mr" };
        return Err(anyhow!(
            "{cli} {request} create failed{}",
            if stderr.is_empty() {
                String::new()
            } else {
                format!(": {stderr}")
            }
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok(if stdout.is_empty() {
        format!("Created pull request from '{branch}'")
    } else {
        stdout
    })
}

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
    fn pr_sync_advice_flags_rebase_and_squash() {
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

        // Feature branch with two commits (squash needed).
        git(&["checkout", "-b", "pr-1"]);
        commit("feature.txt", "feat1\n", "First feature commit");
        commit("feature.txt", "feat2\n", "Second feature commit");

        // Advance main so the branch is also behind (rebase needed).
        git(&["checkout", "main"]);
        commit("base.txt", "base2\n", "Base advances");
        git(&["checkout", "pr-1"]);

        let advice = pr_sync_advice(workspace.path()).expect("advice expected");
        assert!(advice.contains("run /rebase"), "missing rebase: {advice}");
        assert!(advice.contains("run /squash"), "missing squash: {advice}");
        assert!(advice.contains("1 commit behind main"), "behind text: {advice}");
        assert!(advice.contains("2 commits ahead of main"), "ahead text: {advice}");
    }

    #[test]
    fn pr_sync_advice_silent_when_up_to_date() {
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

        git(&["checkout", "-B", "main"]);
        std::fs::write(workspace.path().join("base.txt"), "base\n").expect("write");
        git(&["add", "."]);
        git(&["commit", "-m", "Base commit"]);

        // Single commit, fully up to date with main: no advice.
        git(&["checkout", "-b", "pr-2"]);
        std::fs::write(workspace.path().join("feature.txt"), "feat\n").expect("write");
        git(&["add", "."]);
        git(&["commit", "-m", "Lone feature commit"]);

        assert!(pr_sync_advice(workspace.path()).is_none());
    }

    #[test]
    fn create_pull_request_blocks_with_combined_hint() {
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

        git(&["checkout", "-b", "feature/pr"]);
        commit("feature.txt", "feat1\n", "First feature commit");
        commit("feature.txt", "feat2\n", "Second feature commit");

        git(&["checkout", "main"]);
        commit("base.txt", "base2\n", "Base advances");
        git(&["checkout", "feature/pr"]);

        // Auto-fixes disabled: creation is blocked with the shared hint.
        let result = create_pull_request_output(workspace.path(), false, false, Forge::GitHub);
        let msg = result.expect_err("PR creation should be blocked").to_string();
        assert!(msg.contains("This pull request needs attention"), "{msg}");
        assert!(msg.contains("run /rebase"), "{msg}");
        assert!(msg.contains("run /squash"), "{msg}");
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

    fn git_run(dir: &Path, args: &[&str]) {
        let ok = std::process::Command::new("git")
            .current_dir(dir)
            .args(args)
            .status()
            .expect("run git")
            .success();
        assert!(ok, "git {args:?} failed");
    }

    fn rev_count(dir: &Path, rev: &str) -> usize {
        let output = std::process::Command::new("git")
            .current_dir(dir)
            .args(["rev-list", "--count", rev])
            .output()
            .expect("git rev-list");
        String::from_utf8_lossy(&output.stdout)
            .trim()
            .parse()
            .unwrap_or(0)
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

    #[test]
    fn resolve_editor_command_handles_plain_terminal_editors() {
        let _env_lock = process_env_lock().lock().unwrap_or_else(|p| p.into_inner());
        let workspace = tempdir().expect("workspace");

        for editor in ["vim", "vi", "nano", "emacs"] {
            let _editor = EnvVarGuard::set_value("EDITOR", editor);
            let (program, args, path) =
                resolve_editor_command(workspace.path(), "file.txt").expect("resolve");
            assert_eq!(program, editor);
            assert!(args.is_empty(), "unexpected args for {editor}: {args:?}");
            assert!(path.ends_with("file.txt"), "unexpected path: {path:?}");
        }
    }

    #[test]
    fn resolve_editor_command_splits_arguments() {
        let _env_lock = process_env_lock().lock().unwrap_or_else(|p| p.into_inner());
        let workspace = tempdir().expect("workspace");

        let _editor = EnvVarGuard::set_value("EDITOR", "emacs -nw");
        let (program, args, _path) =
            resolve_editor_command(workspace.path(), "src/main.rs").expect("resolve");
        assert_eq!(program, "emacs");
        assert_eq!(args, vec!["-nw".to_string()]);
    }

    #[test]
    fn resolve_editor_command_rejects_empty_editor() {
        let _env_lock = process_env_lock().lock().unwrap_or_else(|p| p.into_inner());
        let workspace = tempdir().expect("workspace");
        let _editor = EnvVarGuard::set_value("EDITOR", "");
        assert!(resolve_editor_command(workspace.path(), "file.txt").is_err());
    }

    #[test]
    fn editor_needs_terminal_classifies_editors() {
        // Terminal editors → need a window.
        for editor in [
            "vim",
            "vi",
            "nvim",
            "nano",
            "micro",
            "hx",
            "helix",
            "/usr/bin/vim",
        ] {
            assert!(
                editor_needs_terminal(editor, &[]),
                "{editor} should need a terminal"
            );
        }
        // GUI / self-windowing editors → launched directly.
        for editor in [
            "code",
            "codium",
            "subl",
            "gvim",
            "gedit",
            "emacs",
            "emacsclient",
        ] {
            assert!(
                !editor_needs_terminal(editor, &[]),
                "{editor} should not need a terminal"
            );
        }
        // emacs only counts as a terminal editor when explicitly asked.
        assert!(editor_needs_terminal("emacs", &["-nw".to_string()]));
        assert!(editor_needs_terminal("emacsclient", &["-t".to_string()]));
    }

    #[test]
    fn editor_launch_wraps_terminal_editors_and_passes_through_gui() {
        let _env_lock = process_env_lock().lock().unwrap_or_else(|p| p.into_inner());
        let workspace = tempdir().expect("workspace");
        // Use a deterministic configured terminal launcher.
        let terminal = "myterm --run";

        // A terminal editor is wrapped: <launcher> <editor> <args> <path>.
        let _editor = EnvVarGuard::set_value("EDITOR", "vim");
        let argv = editor_launch_argv(workspace.path(), "file.txt", terminal).expect("argv");
        assert_eq!(argv[0], "myterm");
        assert_eq!(argv[1], "--run");
        assert_eq!(argv[2], "vim");
        assert!(argv.last().unwrap().ends_with("file.txt"));

        // A GUI editor is launched directly (no terminal launcher prefix).
        let _editor = EnvVarGuard::set_value("EDITOR", "code --wait");
        let argv = editor_launch_argv(workspace.path(), "file.txt", terminal).expect("argv");
        assert_eq!(argv[0], "code");
        assert_eq!(argv[1], "--wait");
        assert!(argv.last().unwrap().ends_with("file.txt"));
        assert!(!argv.contains(&"myterm".to_string()));
    }

    #[test]
    fn terminal_launcher_honors_configured_terminal() {
        assert_eq!(
            terminal_launcher("gnome-terminal --"),
            Some(vec!["gnome-terminal".to_string(), "--".to_string()])
        );
    }
}
