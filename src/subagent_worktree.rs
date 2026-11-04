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

//! A temporary git worktree for a subagent with `isolation: worktree`: it
//! works on its own checkout of `HEAD`, on a branch of its own, so nothing it
//! does touches the workspace. The worktree lives in the repository's git
//! directory (`.git/orangu-worktrees/<name>`), out of `git status` and on
//! the repository's own disk. After each turn, a worktree the subagent left
//! as it found it is removed with its branch, and the report says it changed
//! nothing; one with changes stays, and the report says where.

use anyhow::{Context, Result, anyhow};
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Worktree {
    /// The repository the worktree belongs to (its main checkout).
    repo: PathBuf,
    path: PathBuf,
    branch: String,
    /// The commit it started from.
    base: String,
    /// Where the workspace sits inside the repository (`git rev-parse
    /// --show-prefix`), so the subagent works in the same place.
    prefix: PathBuf,
}

fn git(dir: &Path, args: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .context("failed to run git")?;
    if !output.status.success() {
        return Err(anyhow!(
            "git {}: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

impl Worktree {
    /// A new worktree of `workspace`'s repository at `HEAD`, for `agent`.
    pub fn create(workspace: &Path, agent: &str) -> Result<Self> {
        let repo = PathBuf::from(
            git(workspace, &["rev-parse", "--show-toplevel"])
                .context("isolation: worktree needs the workspace to be a git repository")?,
        );
        let base = git(workspace, &["rev-parse", "HEAD"])
            .context("isolation: worktree needs a commit to start from")?;
        let prefix = PathBuf::from(git(workspace, &["rev-parse", "--show-prefix"])?);
        let common = PathBuf::from(git(
            workspace,
            &["rev-parse", "--path-format=absolute", "--git-common-dir"],
        )?);
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_millis());
        let safe: String = agent
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' {
                    c
                } else {
                    '-'
                }
            })
            .collect();
        let name = format!("{safe}-{stamp}-{}", std::process::id());
        let path = common.join("orangu-worktrees").join(&name);
        let branch = format!("orangu/{name}");
        let path_text = path.to_string_lossy().to_string();
        git(
            &repo,
            &[
                "worktree", "add", "--quiet", "-b", &branch, &path_text, &base,
            ],
        )?;
        Ok(Self {
            repo,
            path,
            branch,
            base,
            prefix,
        })
    }

    /// Puts back a worktree [`Self::settle`] removed, at the same place and
    /// on the same branch from the same commit, for a task continued later.
    pub fn reopen(&self) -> Result<()> {
        if self.path.exists() {
            return Ok(());
        }
        let path = self.path.to_string_lossy().to_string();
        git(
            &self.repo,
            &[
                "worktree",
                "add",
                "--quiet",
                "-b",
                &self.branch,
                &path,
                &self.base,
            ],
        )
        .map(|_| ())
    }

    /// Where the subagent works: the workspace's place in the worktree.
    pub fn workspace(&self) -> PathBuf {
        self.path.join(&self.prefix)
    }

    /// After a turn, what the report adds about the worktree. One the
    /// subagent left as it started is removed with its branch, and the note
    /// says nothing changed, so a claim to have changed files can be weighed
    /// against it. `None` when the worktree is already gone.
    pub fn settle(&self) -> Result<Option<String>> {
        if !self.path.exists() {
            return Ok(None);
        }
        let changed = !git(&self.path, &["status", "--porcelain"])?.is_empty();
        let committed = git(&self.path, &["rev-parse", "HEAD"])? != self.base;
        if !changed && !committed {
            let path = self.path.to_string_lossy().to_string();
            git(&self.repo, &["worktree", "remove", "--force", &path])?;
            git(&self.repo, &["branch", "-D", "--quiet", &self.branch])?;
            return Ok(Some(
                "[Worked in an isolated worktree and changed no files there; the workspace is \
                 unchanged too, and the worktree has been removed.]"
                    .to_string(),
            ));
        }
        let state = match (committed, changed) {
            (true, true) => "commits and uncommitted changes",
            (true, false) => "commits",
            _ => "uncommitted changes",
        };
        Ok(Some(format!(
            "[Worked in an isolated worktree: its {state} are in {} on branch {} (from {}); the \
             workspace is unchanged. Merge the branch or copy what is wanted, then remove it with \
             `git worktree remove {}` and `git branch -D {}`.]",
            self.path.display(),
            self.branch,
            &self.base[..self.base.len().min(12)],
            self.path.display(),
            self.branch
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repo() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let run = |args: &[&str]| git(dir.path(), args).unwrap();
        run(&["init", "--quiet"]);
        std::fs::create_dir_all(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("sub/a.txt"), "a\n").unwrap();
        run(&["add", "."]);
        run(&[
            "-c",
            "user.email=t@example.com",
            "-c",
            "user.name=t",
            "commit",
            "--quiet",
            "-m",
            "init",
        ]);
        dir
    }

    #[test]
    fn an_unchanged_worktree_is_removed_and_a_changed_one_kept() {
        let repo = repo();
        let workspace = repo.path().join("sub");

        let untouched = Worktree::create(&workspace, "a:b").unwrap();
        assert!(
            untouched.workspace().join("a.txt").exists(),
            "same place in the tree"
        );
        assert!(untouched.branch.starts_with("orangu/a-b-"));
        assert!(
            git(repo.path(), &["status", "--porcelain"])
                .unwrap()
                .is_empty()
        );
        let note = untouched.settle().unwrap().expect("a note");
        assert!(note.contains("changed no files"), "{note}");
        assert!(!untouched.path.exists());
        assert_eq!(untouched.settle().unwrap(), None, "already gone");
        assert!(!git(repo.path(), &["branch"]).unwrap().contains("orangu/"));
        untouched.reopen().unwrap();
        assert!(
            untouched.workspace().join("a.txt").exists(),
            "back for a continuation"
        );
        assert!(untouched.settle().unwrap().is_some());

        let changed = Worktree::create(&workspace, "writer").unwrap();
        std::fs::write(changed.workspace().join("a.txt"), "b\n").unwrap();
        let note = changed.settle().unwrap().expect("a note");
        assert!(note.contains("uncommitted changes"), "{note}");
        assert!(note.contains(&changed.branch));
        assert!(changed.path.exists());
        // The workspace itself never changed.
        assert_eq!(
            std::fs::read_to_string(workspace.join("a.txt")).unwrap(),
            "a\n"
        );
    }

    #[test]
    fn a_workspace_outside_git_cannot_be_isolated() {
        let dir = tempfile::tempdir().unwrap();
        let error = Worktree::create(dir.path(), "a").unwrap_err();
        assert!(format!("{error:#}").contains("git repository"), "{error:#}");
    }
}
