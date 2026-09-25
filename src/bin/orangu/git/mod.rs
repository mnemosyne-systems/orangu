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

use std::{
    fs,
    path::{Path, PathBuf},
};

mod diff;
mod editor;
mod forge;
mod ops;
mod repo;

pub use diff::*;
pub use editor::*;
pub use forge::*;
pub use ops::*;
pub use repo::*;

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
    // Windows Git defaults to autocrlf=true, which rewrites checked-out files with CRLF.
    assert!(
        std::process::Command::new("git")
            .args(["config", "core.autocrlf", "false"])
            .current_dir(workspace)
            .status()
            .expect("git config autocrlf")
            .success()
    );
}

/// Test helper: run a git command in `dir`, asserting success.
#[cfg(test)]
pub(crate) fn git_run(dir: &Path, args: &[&str]) {
    let ok = std::process::Command::new("git")
        .current_dir(dir)
        .args(args)
        .status()
        .expect("run git")
        .success();
    assert!(ok, "git {args:?} failed");
}

/// Test helper: count commits reachable from `rev` in `dir`.
#[cfg(test)]
pub(crate) fn rev_count(dir: &Path, rev: &str) -> usize {
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

/// Test helper: set an environment variable for the lifetime of the guard,
/// restoring the previous value (or removing it) on drop.
#[cfg(test)]
pub(crate) struct EnvVarGuard {
    key: &'static str,
    original: Option<std::ffi::OsString>,
}

#[cfg(test)]
impl EnvVarGuard {
    pub(crate) fn set_path(key: &'static str, value: &Path) -> Self {
        let original = std::env::var_os(key);
        unsafe {
            std::env::set_var(key, value);
        }
        Self { key, original }
    }

    pub(crate) fn set_value(key: &'static str, value: &str) -> Self {
        let original = std::env::var_os(key);
        unsafe {
            std::env::set_var(key, value);
        }
        Self { key, original }
    }
}

#[cfg(test)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

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
}
