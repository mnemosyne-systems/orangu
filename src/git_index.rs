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

//! The Git side of the file-lifecycle API (`http::files`): when the
//! workspace is inside a repository, a file is created, modified, moved and
//! deleted **with its Git command** — `git add`, `git mv`, `git rm` — so the
//! change lands in the index rather than only on disk.
//!
//! **Nothing is ever committed.** Every operation here stops at the index;
//! what to commit, when, and with what message is the user's call, and this
//! API deliberately gives no way to make that decision for them.
//!
//! `gh`/`glab` are detected when present (the same two CLIs `orangu`'s own
//! Git commands prefer) and reported back as the repository's forge, so a
//! client knows which platform it is working against. Neither CLI can touch
//! the index — there is no `gh add` — so the staging itself always runs
//! through `git`; the forge CLI is what forge-level operations would go
//! through if they are added later.
//!
//! Kept self-contained rather than shared with `orangu`'s own much larger
//! `git` module (`src/bin/orangu/git/`), for the same reason each `--init`
//! wizard is: these are separate binaries, and this needs a small, fixed set
//! of index operations rather than that module's fetch/rebase/merge/PR
//! machinery.

use serde::Serialize;
use std::path::{Path, PathBuf};
use std::process::Command;

/// A code-hosting platform whose CLI is installed on this machine. Mirrors
/// `orangu`'s own `Forge` — `gh` for GitHub, `glab` for GitLab — but is only
/// ever `Some` here when the matching CLI is actually on `PATH`, since a
/// forge orangu-server cannot talk to is not worth reporting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Forge {
    GitHub,
    GitLab,
}

impl Forge {
    pub fn label(self) -> &'static str {
        match self {
            Forge::GitHub => "github",
            Forge::GitLab => "gitlab",
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

/// The repository a workspace sits in.
#[derive(Clone, Debug)]
pub struct Repo {
    root: PathBuf,
    forge: Option<Forge>,
}

/// What the Git side of one file operation did, reported alongside the
/// operation's own result so a client can see exactly what reached the index
/// — and, when nothing did, why.
#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct GitOutcome {
    /// The repository root the workspace sits in.
    pub repo_root: String,
    /// `"github"`/`"gitlab"` when the matching CLI (`gh`/`glab`) is
    /// installed, otherwise `null`.
    pub forge: Option<&'static str>,
    /// Whether the change is now in the index.
    pub staged: bool,
    /// The Git command that was run, verbatim.
    pub command: Option<String>,
    /// Why nothing was staged, when `staged` is `false`: `"untracked"` (Git
    /// has no record of this path, so a move or delete is a plain filesystem
    /// operation), `"ignored"` (`.gitignore` covers it), or
    /// `"nothing_to_stage"` (Git tracks no directories of its own).
    pub skipped: Option<&'static str>,
    /// Git's own stderr, when the command failed. The file operation itself
    /// still succeeded — see `http::files`' own documentation.
    pub error: Option<String>,
}

impl GitOutcome {
    fn base(repo: &Repo) -> Self {
        Self {
            repo_root: repo.root.display().to_string(),
            forge: repo.forge.map(Forge::label),
            staged: false,
            command: None,
            skipped: None,
            error: None,
        }
    }

    fn skipped(repo: &Repo, reason: &'static str) -> Self {
        Self {
            skipped: Some(reason),
            ..Self::base(repo)
        }
    }
}

/// The repository `workspace` sits in, if any — the nearest ancestor holding
/// a `.git` entry, which is a directory in an ordinary clone and a file in a
/// worktree or submodule.
pub fn discover(workspace: &Path) -> Option<Repo> {
    let root = workspace
        .ancestors()
        .find(|ancestor| ancestor.join(".git").exists())?
        .to_path_buf();
    let forge = detect_forge(&root);
    Some(Repo { root, forge })
}

/// Which forge this repository lives on, from `origin`'s URL — and only when
/// that forge's CLI is installed, since the point of naming it is to say
/// which tool can act on it.
fn detect_forge(root: &Path) -> Option<Forge> {
    let origin = run(root, ["remote", "get-url", "origin"]).ok()?;
    let host = origin.to_lowercase();
    let forge = if host.contains("github") {
        Forge::GitHub
    } else if host.contains("gitlab") {
        Forge::GitLab
    } else {
        return None;
    };
    cli_available(forge.cli()).then_some(forge)
}

/// Whether `name` is an executable on `PATH` — how "if available" is decided
/// for `gh`/`glab`, without running either of them.
fn cli_available(name: &str) -> bool {
    let Some(path) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&path).any(|dir| {
        let candidate = dir.join(name);
        candidate.is_file() || candidate.with_extension("exe").is_file()
    })
}

/// Run a Git command in `root`, returning its trimmed stdout, or its stderr
/// as the error.
fn run<I, S>(root: &Path, args: I) -> Result<String, String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<std::ffi::OsStr>,
{
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .map_err(|err| format!("failed to run git: {err}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(if stderr.is_empty() {
            format!("git exited with {}", output.status)
        } else {
            stderr
        });
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

impl Repo {
    /// Whether Git has a record of this path — an untracked file is not
    /// `git mv`-able or `git rm`-able, so its move/delete stays a plain
    /// filesystem operation.
    pub fn is_tracked(&self, path: &Path) -> bool {
        run(
            &self.root,
            ["ls-files", "--error-unmatch", "--", &path_arg(path)],
        )
        .is_ok()
    }

    /// Whether this directory holds anything Git tracks. A directory is not
    /// itself an object Git knows about, so "tracked" for one means "has
    /// tracked content".
    pub fn holds_tracked_files(&self, path: &Path) -> bool {
        run(&self.root, ["ls-files", "--", &path_arg(path)])
            .map(|out| !out.is_empty())
            .unwrap_or(false)
    }

    /// Whether `.gitignore` (or any other ignore rule) covers this path.
    /// `git add` refuses an ignored path outright, so those are skipped
    /// rather than turned into an error a caller can do nothing about.
    pub fn is_ignored(&self, path: &Path) -> bool {
        run(&self.root, ["check-ignore", "-q", "--", &path_arg(path)]).is_ok()
    }

    /// `git add <path>` — stages a newly created or modified file.
    pub fn add(&self, path: &Path) -> GitOutcome {
        if self.is_ignored(path) {
            return GitOutcome::skipped(self, "ignored");
        }
        self.stage(
            ["add", "--", &path_arg(path)],
            format!("git add {}", self.show(path)),
        )
    }

    /// `git mv <from> <to>` — moves the file *and* stages the move, so the
    /// rename is recorded rather than showing up as a delete plus an add.
    /// `force` (the endpoint's own `overwrite`) becomes `-f`, which is what
    /// lets the destination be replaced.
    pub fn mv(&self, from: &Path, to: &Path, force: bool) -> GitOutcome {
        let mut args = vec!["mv".to_string()];
        if force {
            args.push("-f".to_string());
        }
        args.push("--".to_string());
        args.push(path_arg(from));
        args.push(path_arg(to));
        let display = format!(
            "git mv {}{} {}",
            if force { "-f " } else { "" },
            self.show(from),
            self.show(to)
        );
        self.stage(args, display)
    }

    /// `git rm -f <path>` — deletes the file *and* stages the deletion.
    /// `-f` because the endpoint's contract is that the file goes away;
    /// without it Git refuses whenever the working copy differs from the
    /// index, which would make deletion fail exactly when it is most likely
    /// to be wanted.
    pub fn rm(&self, path: &Path) -> GitOutcome {
        self.stage(
            ["rm", "-f", "--", &path_arg(path)],
            format!("git rm -f {}", self.show(path)),
        )
    }

    /// Nothing for Git to do: it tracks files, not directories, so creating
    /// or removing an (empty) directory leaves the index untouched.
    pub fn nothing_to_stage(&self) -> GitOutcome {
        GitOutcome::skipped(self, "nothing_to_stage")
    }

    /// The path is real but Git has no record of it, so its move or delete
    /// happened on the filesystem alone.
    pub fn untracked(&self) -> GitOutcome {
        GitOutcome::skipped(self, "untracked")
    }

    fn stage<I, S>(&self, args: I, display: String) -> GitOutcome
    where
        I: IntoIterator<Item = S>,
        S: AsRef<std::ffi::OsStr>,
    {
        match run(&self.root, args) {
            Ok(_) => GitOutcome {
                staged: true,
                command: Some(display),
                ..GitOutcome::base(self)
            },
            Err(error) => GitOutcome {
                command: Some(display),
                error: Some(error),
                ..GitOutcome::base(self)
            },
        }
    }
}

/// Paths are handed to Git as absolute strings (it accepts them anywhere a
/// pathspec is expected), so nothing depends on what the process's current
/// directory happens to be — `-C <root>` sets where the command runs, and
/// the path says exactly which file it means.
fn path_arg(path: &Path) -> String {
    path.display().to_string()
}

impl Repo {
    /// How a path appears in the reported `command`: relative to the
    /// repository root, the way a user would type it themselves.
    fn show(&self, path: &Path) -> String {
        path.strip_prefix(&self.root)
            .unwrap_or(path)
            .display()
            .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// A repository with one committed file, `tracked.txt`, and a
    /// `.gitignore` covering `ignored/`.
    fn repo_with_a_commit() -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("repo");
        let root = dir.path();
        for args in [
            vec!["init", "-q"],
            vec!["config", "user.email", "test@example.com"],
            vec!["config", "user.name", "Test"],
            vec!["config", "commit.gpgsign", "false"],
        ] {
            run(root, args).expect("git setup");
        }
        fs::write(root.join("tracked.txt"), "tracked\n").expect("write");
        fs::write(root.join(".gitignore"), "ignored/\n").expect("write");
        run(root, ["add", "tracked.txt", ".gitignore"]).expect("add");
        run(root, ["commit", "-q", "-m", "initial"]).expect("commit");
        dir
    }

    fn staged_paths(root: &Path) -> String {
        run(root, ["diff", "--cached", "--name-status"]).expect("diff --cached")
    }

    #[test]
    fn discover_finds_the_repository_root_from_a_subdirectory() {
        let dir = repo_with_a_commit();
        let nested = dir.path().join("a/b");
        fs::create_dir_all(&nested).expect("nested");

        let repo = discover(&nested).expect("repo");

        assert_eq!(
            repo.root.canonicalize().unwrap(),
            dir.path().canonicalize().unwrap()
        );
    }

    #[test]
    fn discover_returns_none_outside_a_repository() {
        let dir = tempfile::tempdir().expect("plain directory");
        // A temp dir under /tmp has no repository above it; if the test
        // machine somehow puts one there, this would be a false failure —
        // hence the explicit check rather than an unwrap of the negative.
        assert!(
            discover(dir.path()).is_none() || discover(dir.path()).unwrap().root != dir.path(),
            "a bare temp dir should not be its own repository root"
        );
    }

    #[test]
    fn add_stages_a_new_file() {
        let dir = repo_with_a_commit();
        let repo = discover(dir.path()).expect("repo");
        let path = dir.path().join("new.txt");
        fs::write(&path, "new\n").expect("write");

        let outcome = repo.add(&path);

        assert!(outcome.staged, "{outcome:?}");
        assert!(outcome.command.as_deref().unwrap().starts_with("git add "));
        assert!(outcome.error.is_none());
        assert!(staged_paths(dir.path()).contains("new.txt"));
    }

    #[test]
    fn an_ignored_path_is_skipped_rather_than_failing() {
        let dir = repo_with_a_commit();
        let repo = discover(dir.path()).expect("repo");
        fs::create_dir(dir.path().join("ignored")).expect("dir");
        let path = dir.path().join("ignored/a.txt");
        fs::write(&path, "a\n").expect("write");

        let outcome = repo.add(&path);

        assert!(!outcome.staged);
        assert_eq!(outcome.skipped, Some("ignored"));
        assert!(outcome.error.is_none(), "{outcome:?}");
    }

    #[test]
    fn mv_moves_the_file_and_records_the_rename() {
        let dir = repo_with_a_commit();
        let repo = discover(dir.path()).expect("repo");
        let from = dir.path().join("tracked.txt");
        let to = dir.path().join("renamed.txt");

        let outcome = repo.mv(&from, &to, false);

        assert!(outcome.staged, "{outcome:?}");
        assert!(!from.exists(), "git mv moved the file itself");
        assert!(to.exists());
        // Recorded as a rename, which is the whole point of using `git mv`
        // rather than renaming and staging both sides separately.
        assert!(staged_paths(dir.path()).starts_with('R'));
    }

    #[test]
    fn rm_deletes_the_file_and_stages_the_deletion() {
        let dir = repo_with_a_commit();
        let repo = discover(dir.path()).expect("repo");
        let path = dir.path().join("tracked.txt");
        // Modified against the index: `git rm` alone would refuse, which is
        // why the deletion is forced.
        fs::write(&path, "changed\n").expect("write");

        let outcome = repo.rm(&path);

        assert!(outcome.staged, "{outcome:?}");
        assert!(!path.exists());
        assert!(staged_paths(dir.path()).contains("D\ttracked.txt"));
    }

    #[test]
    fn tracking_and_ignore_state_are_reported_per_path() {
        let dir = repo_with_a_commit();
        let repo = discover(dir.path()).expect("repo");
        let untracked = dir.path().join("loose.txt");
        fs::write(&untracked, "loose\n").expect("write");

        assert!(repo.is_tracked(&dir.path().join("tracked.txt")));
        assert!(!repo.is_tracked(&untracked));
        assert!(repo.is_ignored(&dir.path().join("ignored/a.txt")));
        assert!(!repo.is_ignored(&untracked));
        assert!(repo.holds_tracked_files(dir.path()));
        assert!(!repo.holds_tracked_files(&dir.path().join("nonexistent")));
    }

    /// A failing command is reported, not swallowed — and never as a
    /// success.
    #[test]
    fn a_failed_git_command_is_reported() {
        let dir = repo_with_a_commit();
        let repo = discover(dir.path()).expect("repo");

        let outcome = repo.mv(
            &dir.path().join("nope.txt"),
            &dir.path().join("x.txt"),
            false,
        );

        assert!(!outcome.staged);
        assert!(outcome.error.is_some(), "{outcome:?}");
        assert!(outcome.command.is_some());
    }

    /// Nothing here ever commits: after a full create/move/delete cycle the
    /// repository still has exactly the one commit it started with.
    #[test]
    fn no_operation_ever_commits() {
        let dir = repo_with_a_commit();
        let repo = discover(dir.path()).expect("repo");
        let before = run(dir.path(), ["rev-list", "--count", "HEAD"]).expect("count");

        let created = dir.path().join("new.txt");
        fs::write(&created, "new\n").expect("write");
        repo.add(&created);
        repo.mv(&created, &dir.path().join("moved.txt"), false);
        repo.rm(&dir.path().join("moved.txt"));

        assert_eq!(
            run(dir.path(), ["rev-list", "--count", "HEAD"]).expect("count"),
            before,
            "the commit count must not change"
        );
    }
}
