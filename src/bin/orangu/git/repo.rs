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

/// Every branch, local and remote-tracking, by the name after `refs/heads/`
/// or `refs/remotes/` (`main`, `origin/main`). Not `%(refname:short)`: that
/// spells a local branch `heads/bob/main` once a remote `bob` also has a
/// `main`, which is exactly what `/add_repository` sets up.
pub fn git_branch_names(repo_root: &Path) -> Vec<String> {
    let Ok(output) = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["branch", "--all", "--format=%(refname:lstrip=2)"])
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
        .args(["branch", "--format=%(refname:lstrip=2)"])
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

/// The branches known for `remote` — the names under `refs/remotes/<remote>/`
/// without that prefix (so `main`, not `origin/main`), `HEAD` excluded, sorted.
/// Empty when nothing has been fetched from it or the command fails.
pub fn git_remote_branch_names(repo_root: &Path, remote: &str) -> Vec<String> {
    let Ok(output) = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args([
            "for-each-ref",
            "--format=%(refname)",
            &format!("refs/remotes/{remote}/"),
        ])
        .output()
    else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    let prefix = format!("refs/remotes/{remote}/");
    let mut branches: Vec<String> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| line.trim().strip_prefix(prefix.as_str()))
        .filter(|name| !name.is_empty() && *name != "HEAD")
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

/// Every file `git` tracks under `workspace` (`git ls-files`, scoped to `workspace`
/// so a nested workspace only lists its own subtree), as paths relative to
/// `workspace`. Untracked files (including anything `.gitignore` excludes) never
/// appear — this is `git`'s own bookkeeping, not a filesystem walk. Used by
/// `/auto_review all` to review the whole project without picking up scratch or
/// ignored files. Returns an empty vector outside a Git repository or on error.
pub fn git_tracked_files(workspace: &Path) -> Vec<String> {
    let Ok(output) = std::process::Command::new("git")
        .arg("-C")
        .arg(workspace)
        .args(["ls-files"])
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
    repository_name_from_url(&git_remote_url(repo_root, "origin")?)
}

/// The URL of the configured remote `remote` (`git remote get-url`), trimmed.
/// `None` when the remote does not exist or the command fails.
pub fn git_remote_url(repo_root: &Path, remote: &str) -> Option<String> {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["remote", "get-url", remote])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Where a repository's `origin` lives — the layouts `/add_repository` knows
/// how to find another user's copy of the same project in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RepositoryHost {
    /// `github.com` or a `github.*` instance: forks live directly under the
    /// user.
    GitHub,
    /// `gitlab.com` or a `gitlab.*` instance: forks live directly under the
    /// user, whatever group the original sits in.
    GitLab,
    /// Any other host reached over the network: the other user's copy is the
    /// sibling directory named after them.
    Network,
    /// A filesystem path (or `file://` URL): likewise the sibling directory.
    Local,
}

impl RepositoryHost {
    /// The name shown in `/add_repository`'s report.
    pub fn name(self) -> &'static str {
        match self {
            RepositoryHost::GitHub => "GitHub",
            RepositoryHost::GitLab => "GitLab",
            RepositoryHost::Network => "network",
            RepositoryHost::Local => "local",
        }
    }

    /// Whether the copy is found by replacing the whole owner path with the
    /// user (a forge fork) rather than only the directory the project sits in.
    fn is_forge(self) -> bool {
        matches!(self, RepositoryHost::GitHub | RepositoryHost::GitLab)
    }
}

/// Another user's copy of the project behind an `origin` URL, as
/// [`sibling_repository`] derives it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SiblingRepository {
    pub host: RepositoryHost,
    /// The project name, e.g. `pgmoneta` — the last path segment of `origin`
    /// without `.git`.
    pub project: String,
    /// The URL to fetch the other user's copy from: `origin`'s scheme, host and
    /// project with the owner replaced.
    pub url: String,
}

/// Locate `user`'s copy of the project that `origin_url` points at: the same
/// host, reached the same way (HTTPS, SSH, or a filesystem path), the same
/// project name, and `user` in place of the owner. So
/// `https://github.com/alice/pgmoneta.git` yields
/// `https://github.com/bob/pgmoneta.git` for `bob`, `git@gitlab.com:group/sub/pgmoneta.git`
/// yields `git@gitlab.com:bob/pgmoneta.git` (a forge fork lives directly under
/// the user), while `/srv/git/alice/pgmoneta.git` and
/// `ssh://git.example.com/srv/git/alice/pgmoneta.git` yield the same path with
/// `bob` in place of `alice` (only the directory the project sits in is
/// renamed).
///
/// `github.com`/`github.*` and `gitlab.com`/`gitlab.*` are recognised by name;
/// any other host is a plain network host and a path-less URL is local. Errors
/// when the URL has no owner segment to replace, since there is then nowhere
/// to look.
pub fn sibling_repository(origin_url: &str, user: &str) -> Result<SiblingRepository> {
    let url = origin_url.trim();
    if url.is_empty() {
        return Err(anyhow!("the 'origin' remote has no URL"));
    }
    // A Windows drive path (`C:\...` or `C:/...`) is a filesystem path even
    // though it has the `x:` shape of an scp-style host.
    let drive_path = {
        let bytes = url.as_bytes();
        bytes.len() >= 3
            && bytes[0].is_ascii_alphabetic()
            && bytes[1] == b':'
            && (bytes[2] == b'\\' || bytes[2] == b'/')
    };
    // Split the URL into what comes before the path (scheme and authority, or
    // the scp-style `user@host:`), the path itself, and the host it names —
    // `None` for a filesystem path.
    let (prefix, path, host): (String, &str, Option<&str>) =
        if let Some(rest) = url.strip_prefix("file://") {
            ("file://".to_string(), rest, None)
        } else if let Some((scheme, rest)) = url.split_once("://") {
            let (authority, path) = rest.split_once('/').unwrap_or((rest, ""));
            let host = authority.rsplit_once('@').map_or(authority, |(_, h)| h);
            let host = host.rsplit_once(':').map_or(host, |(h, _)| h);
            (format!("{scheme}://{authority}/"), path, Some(host))
        } else if !drive_path
            && let Some((authority, path)) = url.split_once(':')
            && !authority.contains('/')
            && !authority.contains('\\')
            && !authority.is_empty()
            && !url.starts_with('.')
            && !url.starts_with('/')
            && !url.starts_with('~')
        {
            let host = authority.rsplit_once('@').map_or(authority, |(_, h)| h);
            (format!("{authority}:"), path, Some(host))
        } else {
            (String::new(), url, None)
        };

    let host = match host {
        None => RepositoryHost::Local,
        Some(host) => {
            let host = host.to_ascii_lowercase();
            let host = host.strip_prefix("www.").unwrap_or(&host);
            if host == "github.com" || host.starts_with("github.") {
                RepositoryHost::GitHub
            } else if host == "gitlab.com" || host.starts_with("gitlab.") {
                RepositoryHost::GitLab
            } else {
                RepositoryHost::Network
            }
        }
    };

    // A local path keeps whichever separator it came with; everything reached
    // over the network is `/`-separated.
    let separator = if host == RepositoryHost::Local && path.contains('\\') && !path.contains('/') {
        "\\"
    } else {
        "/"
    };
    let path = path.trim_end_matches(['/', '\\']);
    let (path, suffix) = match path.strip_suffix(".git") {
        Some(path) => (path, ".git"),
        None => (path, ""),
    };
    let mut segments: Vec<&str> = if host == RepositoryHost::Local {
        path.split(['/', '\\']).collect()
    } else {
        path.split('/').collect()
    };
    let project = segments.pop().filter(|p| !p.is_empty()).ok_or_else(|| {
        anyhow!("could not read a project name from the 'origin' URL '{origin_url}'")
    })?;
    let owner = segments.last().copied().filter(|o| !o.is_empty());
    if owner.is_none() {
        return Err(anyhow!(
            "the 'origin' URL '{origin_url}' names no owner to replace with '{user}'"
        ));
    }
    // A forge fork lives directly under the user, whatever group nesting the
    // original had; any other copy keeps the directories above the owner.
    let mut rebuilt: Vec<&str> = if host.is_forge() {
        Vec::new()
    } else {
        segments[..segments.len() - 1].to_vec()
    };
    rebuilt.push(user);
    rebuilt.push(project);
    Ok(SiblingRepository {
        host,
        project: project.to_string(),
        url: format!("{prefix}{}{suffix}", rebuilt.join(separator)),
    })
}

/// A repository's web home on a known forge, derived from its `origin` remote.
pub struct ForgeWeb {
    /// The repository's web base, e.g. `https://github.com/owner/repo` (no
    /// trailing slash, no `.git`).
    pub base: String,
    /// `true` for GitLab (whose blob path is `/-/blob/…` and whose multi-line
    /// fragment is `#L10-20`), `false` for GitHub (`/blob/…`, `#L10-L20`).
    pub gitlab: bool,
}

impl ForgeWeb {
    /// The web URL for `path` (repository-root-relative, forward slashes) at
    /// `git_ref`, highlighting lines `start`–`end` (1-based). A single line when
    /// `start == end`.
    pub fn blob_url(&self, git_ref: &str, path: &str, start: usize, end: usize) -> String {
        let fragment = if start == end {
            format!("#L{start}")
        } else if self.gitlab {
            format!("#L{start}-{end}")
        } else {
            format!("#L{start}-L{end}")
        };
        let blob = if self.gitlab { "/-/blob/" } else { "/blob/" };
        format!("{}{blob}{git_ref}/{path}{fragment}", self.base)
    }
}

/// The GitHub/GitLab web home for the repository at `repo_root`, parsed from its
/// `origin` remote URL. Returns `None` when there is no `origin`, the URL cannot
/// be parsed, or the host is neither `github.com` nor `gitlab.com` (self-hosted
/// instances are not assumed) — callers then leave source references as plain
/// text.
pub fn forge_web_from_origin(repo_root: &Path) -> Option<ForgeWeb> {
    forge_web_from_url(&git_remote_url(repo_root, "origin")?)
}

/// Parse a remote URL into a [`ForgeWeb`]. Handles HTTPS (`https://host/owner/repo`),
/// scp-style SSH (`git@host:owner/repo`), and `ssh://git@host/owner/repo` forms,
/// each optionally `.git`-suffixed. Only `github.com` and `gitlab.com` are
/// recognised.
pub fn forge_web_from_url(url: &str) -> Option<ForgeWeb> {
    let url = url.trim();
    // Reduce every form to `host/owner/repo…` (no scheme, no user@).
    let without_scheme = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
        .or_else(|| url.strip_prefix("ssh://"))
        .unwrap_or(url);
    let without_user = without_scheme
        .rsplit_once('@')
        .map(|(_, rest)| rest)
        .unwrap_or(without_scheme);
    // scp syntax uses `host:owner/repo`; normalise the first `:` to `/`.
    let normalised = without_user.replacen(':', "/", 1);
    let path = normalised.trim_end_matches('/');
    let path = path.strip_suffix(".git").unwrap_or(path);

    let (host, owner_repo) = path.split_once('/')?;
    if owner_repo.is_empty() || !owner_repo.contains('/') {
        return None;
    }
    let gitlab = match host {
        "github.com" => false,
        "gitlab.com" => true,
        _ => return None,
    };
    Some(ForgeWeb {
        base: format!("https://{host}/{owner_repo}"),
        gitlab,
    })
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
    orangu::pdf::decode_text(file_path, &output.stdout).context("failed to read git show output")
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

/// One commit's contribution to `/statistics`: the day it was authored (days
/// since the Unix epoch, UTC), the author's name, and the lines it added and
/// removed (summed across every file the commit touched; binary files, which
/// `git log --numstat` reports as `-`, contribute zero).
#[derive(Debug, Clone, PartialEq)]
pub struct CommitStat {
    pub day: u64,
    pub author: String,
    pub additions: usize,
    pub deletions: usize,
}

/// Every commit reachable from `HEAD`, oldest first. Used by `/statistics` for
/// the "By author" breakdown (commits plus lines added/removed, styled like
/// `/export pr`'s changed-files list) and to give the heatmap something to
/// show on a repository with commit history predating any orangu-recorded
/// activity. Returns an empty list outside a Git repository or if `git log`
/// fails.
pub fn commit_history(workspace: &Path) -> Vec<CommitStat> {
    let Some(repo_root) = discover_git_root(workspace) else {
        return Vec::new();
    };
    // `\x01` (a byte that never appears in a timestamp or name) marks the
    // start of each commit's header line, so a commit's numstat block can be
    // told apart from the next commit's header even though `git log` puts no
    // other separator between them.
    let Ok(output) = std::process::Command::new("git")
        .arg("-C")
        .arg(&repo_root)
        .args(["log", "--reverse", "--numstat", "--format=%x01%at%x09%an"])
        .output()
    else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    String::from_utf8_lossy(&output.stdout)
        .split('\x01')
        .filter_map(|chunk| {
            let mut lines = chunk.lines();
            let (secs, author) = lines.next()?.split_once('\t')?;
            let day = secs.trim().parse::<u64>().ok()? / 86_400;
            let (mut additions, mut deletions) = (0usize, 0usize);
            for line in lines {
                let mut fields = line.splitn(3, '\t');
                if let (Some(added), Some(removed)) = (fields.next(), fields.next()) {
                    additions += added.parse::<usize>().unwrap_or(0);
                    deletions += removed.parse::<usize>().unwrap_or(0);
                }
            }
            Some(CommitStat {
                day,
                author: author.to_string(),
                additions,
                deletions,
            })
        })
        .collect()
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
    fn sibling_repository_replaces_the_owner_and_keeps_the_rest() {
        use RepositoryHost::*;
        for (origin, host, url) in [
            // Forge forks live directly under the user; scheme and `.git` stay.
            (
                "https://github.com/alice/pgmoneta.git",
                GitHub,
                "https://github.com/bob/pgmoneta.git",
            ),
            (
                "https://github.com/alice/pgmoneta",
                GitHub,
                "https://github.com/bob/pgmoneta",
            ),
            (
                "git@github.com:alice/pgmoneta.git",
                GitHub,
                "git@github.com:bob/pgmoneta.git",
            ),
            (
                "ssh://git@github.com/alice/pgmoneta.git",
                GitHub,
                "ssh://git@github.com/bob/pgmoneta.git",
            ),
            (
                "https://GitHub.com/alice/pgmoneta/",
                GitHub,
                "https://GitHub.com/bob/pgmoneta",
            ),
            (
                "https://github.example.com/alice/pgmoneta.git",
                GitHub,
                "https://github.example.com/bob/pgmoneta.git",
            ),
            // GitLab group nesting collapses to the user.
            (
                "git@gitlab.com:group/sub/pgmoneta.git",
                GitLab,
                "git@gitlab.com:bob/pgmoneta.git",
            ),
            (
                "https://gitlab.example.com:8443/group/pgmoneta.git",
                GitLab,
                "https://gitlab.example.com:8443/bob/pgmoneta.git",
            ),
            // Any other network host and the local filesystem rename only the
            // directory the project sits in.
            (
                "ssh://git.example.com/srv/git/alice/pgmoneta.git",
                Network,
                "ssh://git.example.com/srv/git/bob/pgmoneta.git",
            ),
            (
                "git@codeberg.org:alice/pgmoneta.git",
                Network,
                "git@codeberg.org:bob/pgmoneta.git",
            ),
            (
                "/srv/git/alice/pgmoneta.git",
                Local,
                "/srv/git/bob/pgmoneta.git",
            ),
            (
                "file:///srv/git/alice/pgmoneta",
                Local,
                "file:///srv/git/bob/pgmoneta",
            ),
            ("../../alice/pgmoneta", Local, "../../bob/pgmoneta"),
            ("~/src/alice/pgmoneta", Local, "~/src/bob/pgmoneta"),
            // Windows drive paths, with either separator, are local — not an
            // scp-style host called `C`.
            (
                "C:\\Users\\me\\alice\\pgmoneta",
                Local,
                "C:\\Users\\me\\bob\\pgmoneta",
            ),
            (
                "C:/Users/me/alice/pgmoneta.git",
                Local,
                "C:/Users/me/bob/pgmoneta.git",
            ),
            (
                "file:///C:/Users/me/alice/pgmoneta",
                Local,
                "file:///C:/Users/me/bob/pgmoneta",
            ),
        ] {
            let sibling =
                sibling_repository(origin, "bob").unwrap_or_else(|e| panic!("{origin}: {e}"));
            assert_eq!(sibling.host, host, "{origin}");
            assert_eq!(sibling.project, "pgmoneta", "{origin}");
            assert_eq!(sibling.url, url, "{origin}");
        }
    }

    #[test]
    fn sibling_repository_needs_an_owner_to_replace() {
        for origin in [
            "",
            "pgmoneta.git",
            "/pgmoneta",
            "https://github.com/pgmoneta.git",
        ] {
            assert!(
                sibling_repository(origin, "bob").is_err(),
                "{origin:?} should have no owner"
            );
        }
    }

    #[test]
    fn git_remote_branch_names_lists_one_remote_without_its_prefix() {
        let workspace = tempdir().expect("workspace");
        init_git_for_test(workspace.path());
        git_run(workspace.path(), &["checkout", "-B", "main"]);
        git_run(workspace.path(), &["commit", "--allow-empty", "-m", "one"]);
        // A remote-tracking ref for `bob` and one for `origin`; only `bob`'s
        // is listed, and without the `bob/` prefix.
        git_run(
            workspace.path(),
            &["update-ref", "refs/remotes/bob/muse", "HEAD"],
        );
        git_run(
            workspace.path(),
            &["update-ref", "refs/remotes/bob/main", "HEAD"],
        );
        git_run(
            workspace.path(),
            &["update-ref", "refs/remotes/origin/main", "HEAD"],
        );
        git_run(
            workspace.path(),
            &[
                "symbolic-ref",
                "refs/remotes/bob/HEAD",
                "refs/remotes/bob/main",
            ],
        );
        assert_eq!(
            git_remote_branch_names(workspace.path(), "bob"),
            vec!["main", "muse"]
        );
        assert!(git_remote_branch_names(workspace.path(), "nobody").is_empty());
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
    fn commit_history_reads_day_and_author_oldest_first() {
        let workspace = tempdir().expect("workspace");
        init_git_for_test(workspace.path());

        std::fs::write(workspace.path().join("a.txt"), "one\n").expect("write");
        git_run(workspace.path(), &["add", "."]);
        git_run(workspace.path(), &["commit", "-m", "first"]);

        std::fs::write(workspace.path().join("a.txt"), "two\n").expect("write");
        git_run(workspace.path(), &["add", "."]);
        git_run(
            workspace.path(),
            &["-c", "user.name=Someone Else", "commit", "-m", "second"],
        );

        let history = commit_history(workspace.path());
        assert_eq!(history.len(), 2);
        // Oldest first: the "first" commit (author "Orangu Tests" from
        // `init_git_for_test`) before "second" (author "Someone Else").
        assert_eq!(history[0].author, "Orangu Tests");
        assert_eq!(history[1].author, "Someone Else");
        // Both commits happened today.
        assert_eq!(history[0].day, crate::activity_log::today());
        // "one\n" -> "two\n" is a one-line change: 1 added, 1 removed.
        assert_eq!(history[1].additions, 1);
        assert_eq!(history[1].deletions, 1);
        // The first commit only adds a line (a new file).
        assert_eq!(history[0].additions, 1);
        assert_eq!(history[0].deletions, 0);
    }

    #[test]
    fn commit_history_is_empty_outside_a_git_repository() {
        let workspace = tempdir().expect("workspace");
        assert!(commit_history(workspace.path()).is_empty());
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
