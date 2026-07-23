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

//! The file life cycle: create, modify, move, delete and show a file, plus
//! the directory operations around it — one function per operation, each
//! taking a workspace root and a request struct, and each confined to that
//! workspace.
//!
//! This is the shared implementation behind both surfaces orangu exposes it
//! through, so they cannot drift: `orangu-server`'s HTTP endpoints
//! (`/v1/create_file`, `/v1/modify_file`, …) and `orangu`'s own local tools
//! (`create_file`, `modify_file`, …) are two front ends over exactly these
//! functions, with the same field names, the same defaults and the same
//! errors.
//!
//! Every path in a request is resolved against the workspace and rejected if
//! it lands outside it, so neither surface can reach a file it was not
//! pointed at. The check is both lexical (`..` segments folded away, via
//! [`crate::tools::resolve_workspace_path`]) and physical (the nearest
//! existing ancestor is canonicalized, so a symlink pointing out of the tree
//! cannot be used as a way around it).
//!
//! The unified diff [`modify`] returns carries zero context lines (what
//! `diff -U0` prints): the caller said exactly which lines it was replacing,
//! so each edit is one exact hunk, and adjacent edits never end up with two
//! hunks fighting over the same context lines.
//!
//! When the workspace is inside a Git repository, each change is made *with*
//! its Git command — `git add`, `git mv`, `git rm` (see [`crate::git_index`])
//! — so it lands in the index; `git: false` on a request opts back out to a
//! plain filesystem change. **Nothing is ever committed**: what to commit,
//! and when, stays the user's decision.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::git_index::{GitOutcome, Repo};

/// `overwrite` defaults to on for [`create`]: writing a file that is
/// already there is an override, not a mistake to be refused. A caller that
/// wants create-if-absent passes `overwrite: false` and gets an
/// `already_exists` error instead.
pub fn overwrite_default() -> bool {
    true
}

/// `git` defaults to on: inside a repository, a file's life cycle *is* a
/// sequence of Git operations, and a caller that wants the plain filesystem
/// has to say so.
pub fn git_default() -> bool {
    true
}

/// The repository to act in for this request: none when the workspace isn't
/// in one, or when the request opted out with `git: false`.
fn repo_for(workspace: &Path, git: bool) -> Option<Repo> {
    git.then(|| crate::git_index::discover(workspace)).flatten()
}

/// Everything that can go wrong, mapped to one HTTP status and one stable
/// `code` string per variant so a client can branch on the code rather than
/// on message text.
#[derive(Debug)]
pub enum FileError {
    /// The path resolves outside the workspace root.
    OutsideWorkspace(String),
    /// Nothing exists at the path.
    NotFound(String),
    /// Something already exists at the path and the request didn't allow
    /// replacing it.
    AlreadyExists(String),
    /// The path exists but isn't a regular file (a directory, a socket, …).
    NotAFile(String),
    /// The path exists but isn't a directory — the mirror of
    /// [`NotAFile`](Self::NotAFile), for the two directory endpoints.
    NotADirectory(String),
    /// The directory still has something in it, and `delete_directory` only
    /// removes empty ones.
    NotEmpty(String),
    /// The request itself doesn't make sense: an empty path, a line range
    /// running off the end of the file, an unparsable mode, …
    BadRequest(String),
    /// The file isn't valid UTF-8, so it has no line structure to edit and
    /// no content to return as JSON.
    NotUtf8(String),
    /// The filesystem said no.
    Io(String),
}

/// Rendered as `<code>: <message>` — how a caller with one error line to
/// spend (a tool result, a log) shows it. A surface that carries the code
/// as its own field (the HTTP endpoints) uses [`FileError::message`]
/// instead.
impl std::fmt::Display for FileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code(), self.message())
    }
}

impl std::error::Error for FileError {}

impl FileError {
    pub fn code(&self) -> &'static str {
        match self {
            FileError::OutsideWorkspace(_) => "outside_workspace",
            FileError::NotFound(_) => "not_found",
            FileError::AlreadyExists(_) => "already_exists",
            FileError::NotAFile(_) => "not_a_file",
            FileError::NotADirectory(_) => "not_a_directory",
            FileError::NotEmpty(_) => "not_empty",
            FileError::BadRequest(_) => "bad_request",
            FileError::NotUtf8(_) => "not_utf8",
            FileError::Io(_) => "io_error",
        }
    }

    /// The message alone, for a surface that carries the [`code`](Self::code)
    /// separately (as the HTTP endpoints do in their JSON body).
    pub fn message(&self) -> &str {
        match self {
            FileError::OutsideWorkspace(m)
            | FileError::NotFound(m)
            | FileError::AlreadyExists(m)
            | FileError::NotAFile(m)
            | FileError::NotADirectory(m)
            | FileError::NotEmpty(m)
            | FileError::BadRequest(m)
            | FileError::NotUtf8(m)
            | FileError::Io(m) => m,
        }
    }
}

pub type FileResult<T> = std::result::Result<T, FileError>;

/// A file mode, accepted either as an octal string (`"0644"`, `"644"`) or as
/// the raw number `chmod` takes (`420`, i.e. `0o644` — JSON has no octal
/// literals, so a number is the value *after* octal parsing, exactly what
/// `stat`'s `st_mode & 0o7777` holds).
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum Mode {
    Text(String),
    Bits(u32),
}

impl Mode {
    /// The permission bits this mode names, rejecting anything that doesn't
    /// fit in the low 12 bits (`0o7777`) — the range `chmod` itself accepts.
    fn bits(&self) -> FileResult<u32> {
        let bits = match self {
            Mode::Text(text) => {
                let trimmed = text.trim();
                u32::from_str_radix(trimmed.trim_start_matches("0o"), 8).map_err(|_| {
                    FileError::BadRequest(format!(
                        "mode {text:?} is not an octal permission string (e.g. \"0644\")"
                    ))
                })?
            }
            Mode::Bits(bits) => *bits,
        };
        if bits > 0o7777 {
            return Err(FileError::BadRequest(format!(
                "mode {bits:#o} is out of range (expected at most 0o7777)"
            )));
        }
        Ok(bits)
    }
}

/// `POST /v1/create_file`.
#[derive(Debug, Deserialize)]
pub struct CreateFileRequest {
    /// Workspace-relative (or workspace-absolute) path of the file to write.
    pub path: String,
    /// The file's full content. Omitted means an empty file.
    #[serde(default)]
    pub content: String,
    /// Permission bits to set on the new file. Unset leaves the process
    /// umask to decide, exactly as an ordinary `open`/`create` would.
    #[serde(default)]
    pub mode: Option<Mode>,
    /// Replace the file if it already exists — **the default**. Creating a
    /// file that is already there is an override, the same on every surface:
    /// the endpoint, the tool, and the typed command. Pass `false` for
    /// create-if-absent instead, which turns an existing path into an
    /// `already_exists` error.
    #[serde(default = "overwrite_default")]
    pub overwrite: bool,
    /// Create any missing parent directories. Without this, a missing parent
    /// is a `404 not_found`.
    #[serde(default)]
    pub parents: bool,
    /// Perform the change with its Git command (`git add`/`git mv`/`git
    /// rm`) when the workspace is inside a repository, leaving it staged.
    /// Set to `false` for a plain filesystem change that leaves the index
    /// alone. Nothing is ever committed either way.
    #[serde(default = "git_default")]
    pub git: bool,
}

#[derive(Debug, Serialize)]
pub struct CreateFileResponse {
    pub path: String,
    pub bytes_written: usize,
    pub mode: Option<String>,
    /// `true` when an existing file was replaced (only possible with
    /// `overwrite`), `false` when the file is new.
    pub overwritten: bool,
    /// What Git did, or `null` when the workspace isn't a repository (or
    /// the request passed `"git": false`).
    pub git: Option<GitOutcome>,
}

/// `POST /v1/modify_file`.
#[derive(Debug, Deserialize)]
pub struct ModifyFileRequest {
    pub path: String,
    /// The changes to apply, each naming the lines it replaces. Ranges are
    /// 1-based and inclusive, must not overlap, and are applied to the file
    /// as it is *now* — every range refers to the original line numbering,
    /// not to the numbering left behind by an earlier edit in the same
    /// request.
    pub edits: Vec<LineEdit>,
    /// Perform the change with its Git command (`git add`/`git mv`/`git
    /// rm`) when the workspace is inside a repository, leaving it staged.
    /// Set to `false` for a plain filesystem change that leaves the index
    /// alone. Nothing is ever committed either way.
    #[serde(default = "git_default")]
    pub git: bool,
}

#[derive(Debug, Deserialize)]
pub struct LineEdit {
    /// First line replaced, 1-based.
    pub start_line: usize,
    /// Last line replaced, 1-based and inclusive. `start_line - 1` inserts
    /// before `start_line` without replacing anything.
    pub end_line: usize,
    /// The lines to put in their place. An empty string deletes the range.
    /// A trailing newline is not required — the replacement is spliced in as
    /// whole lines either way.
    #[serde(default)]
    pub replacement: String,
}

#[derive(Debug, Serialize)]
pub struct ModifyFileResponse {
    pub path: String,
    pub lines_before: usize,
    pub lines_after: usize,
    pub edits_applied: usize,
    /// A zero-context unified diff of exactly what changed.
    pub diff: String,
    /// What Git did, or `null` when the workspace isn't a repository (or
    /// the request passed `"git": false`).
    pub git: Option<GitOutcome>,
}

/// `POST /v1/move_file`.
#[derive(Debug, Deserialize)]
pub struct MoveFileRequest {
    pub from: String,
    pub to: String,
    /// Permission bits to set on the file at its new path. Unset keeps
    /// whatever it already had.
    #[serde(default)]
    pub mode: Option<Mode>,
    /// Replace the destination if it already exists.
    #[serde(default)]
    pub overwrite: bool,
    /// Create any missing parent directories of the destination.
    #[serde(default)]
    pub parents: bool,
    /// Perform the change with its Git command (`git add`/`git mv`/`git
    /// rm`) when the workspace is inside a repository, leaving it staged.
    /// Set to `false` for a plain filesystem change that leaves the index
    /// alone. Nothing is ever committed either way.
    #[serde(default = "git_default")]
    pub git: bool,
}

#[derive(Debug, Serialize)]
pub struct MoveFileResponse {
    pub from: String,
    pub to: String,
    pub mode: Option<String>,
    pub overwritten: bool,
    /// What Git did, or `null` when the workspace isn't a repository (or
    /// the request passed `"git": false`).
    pub git: Option<GitOutcome>,
}

/// `POST /v1/delete_file`.
#[derive(Debug, Deserialize)]
pub struct DeleteFileRequest {
    pub path: String,
    /// Perform the change with its Git command (`git add`/`git mv`/`git
    /// rm`) when the workspace is inside a repository, leaving it staged.
    /// Set to `false` for a plain filesystem change that leaves the index
    /// alone. Nothing is ever committed either way.
    #[serde(default = "git_default")]
    pub git: bool,
}

#[derive(Debug, Serialize)]
pub struct DeleteFileResponse {
    pub path: String,
    pub deleted: bool,
    /// What Git did, or `null` when the workspace isn't a repository (or
    /// the request passed `"git": false`).
    pub git: Option<GitOutcome>,
}

/// `POST /v1/show_file`.
#[derive(Debug, Deserialize)]
pub struct ShowFileRequest {
    pub path: String,
}

#[derive(Debug, Serialize)]
pub struct ShowFileResponse {
    pub path: String,
    pub content: String,
    pub bytes: usize,
    pub lines: usize,
    pub mode: Option<String>,
}

/// `POST /v1/create_directory`.
#[derive(Debug, Deserialize)]
pub struct CreateDirectoryRequest {
    pub path: String,
    /// Permission bits for the new directory, as an octal string (`"0755"`)
    /// or the number `chmod` takes (`493`). Unset leaves the process umask
    /// to decide, exactly as an ordinary `mkdir` would.
    #[serde(default)]
    pub mode: Option<Mode>,
    /// Create any missing parent directories too. They are created with the
    /// umask's own permissions — `mode` applies to the directory actually
    /// named by `path`, the same way `mkdir -p -m` behaves.
    #[serde(default)]
    pub parents: bool,
    /// Perform the change with its Git command (`git add`/`git mv`/`git
    /// rm`) when the workspace is inside a repository, leaving it staged.
    /// Set to `false` for a plain filesystem change that leaves the index
    /// alone. Nothing is ever committed either way.
    #[serde(default = "git_default")]
    pub git: bool,
}

#[derive(Debug, Serialize)]
pub struct CreateDirectoryResponse {
    pub path: String,
    pub mode: Option<String>,
    /// What Git did, or `null` when the workspace isn't a repository (or
    /// the request passed `"git": false`).
    pub git: Option<GitOutcome>,
}

/// `POST /v1/move_directory`.
#[derive(Debug, Deserialize)]
pub struct MoveDirectoryRequest {
    pub from: String,
    pub to: String,
    /// Permission bits to set on the directory at its new path. Unset keeps
    /// whatever it already had. Only the moved directory itself is touched —
    /// never anything inside it.
    #[serde(default)]
    pub mode: Option<Mode>,
    /// Create any missing parent directories of the destination.
    #[serde(default)]
    pub parents: bool,
    /// Perform the change with its Git command (`git add`/`git mv`/`git
    /// rm`) when the workspace is inside a repository, leaving it staged.
    /// Set to `false` for a plain filesystem change that leaves the index
    /// alone. Nothing is ever committed either way.
    #[serde(default = "git_default")]
    pub git: bool,
}

#[derive(Debug, Serialize)]
pub struct MoveDirectoryResponse {
    pub from: String,
    pub to: String,
    pub mode: Option<String>,
    /// What Git did, or `null` when the workspace isn't a repository (or
    /// the request passed `"git": false`).
    pub git: Option<GitOutcome>,
}

/// `POST /v1/delete_directory`.
#[derive(Debug, Deserialize)]
pub struct DeleteDirectoryRequest {
    pub path: String,
    /// Perform the change with its Git command (`git add`/`git mv`/`git
    /// rm`) when the workspace is inside a repository, leaving it staged.
    /// Set to `false` for a plain filesystem change that leaves the index
    /// alone. Nothing is ever committed either way.
    #[serde(default = "git_default")]
    pub git: bool,
}

#[derive(Debug, Serialize)]
pub struct DeleteDirectoryResponse {
    pub path: String,
    pub deleted: bool,
    /// What Git did, or `null` when the workspace isn't a repository (or
    /// the request passed `"git": false`).
    pub git: Option<GitOutcome>,
}

/// Resolve a request's path against `workspace`, refusing anything that
/// lands outside it.
///
/// Two checks, because either alone has a hole. The lexical one
/// (`orangu::tools::resolve_workspace_path`, the same resolution `orangu`'s
/// own file tools use) folds `..` away before comparing, so a path can't
/// climb out with parent segments — but it can't see a symlink. The physical
/// one canonicalizes the nearest ancestor that actually exists and compares
/// that against the canonicalized workspace, which catches a symlink inside
/// the tree pointing out of it. The *nearest existing* ancestor, rather than
/// the path itself, is what makes this work for `create_file`/
/// `create_directory`, whose target doesn't exist yet by definition.
fn resolve(workspace: &Path, raw: &str) -> FileResult<PathBuf> {
    if raw.trim().is_empty() {
        return Err(FileError::BadRequest("path must not be empty".to_string()));
    }
    let resolved = crate::tools::resolve_workspace_path(workspace, raw)
        .map_err(|err| FileError::OutsideWorkspace(format!("{raw:?}: {err}")))?;

    let canonical_workspace = canonical(workspace)?;
    let mut existing = resolved.as_path();
    let anchor = loop {
        if existing.exists() {
            break existing;
        }
        match existing.parent() {
            Some(parent) => existing = parent,
            None => {
                return Err(FileError::OutsideWorkspace(format!(
                    "{raw:?}: no existing parent directory inside the workspace"
                )));
            }
        }
    };
    if !canonical(anchor)?.starts_with(&canonical_workspace) {
        return Err(FileError::OutsideWorkspace(format!(
            "{raw:?}: path escapes the configured workspace"
        )));
    }
    Ok(resolved)
}

/// How a resolved path is echoed back: relative to the workspace, so a
/// client sees the same shape it sent rather than the server's absolute
/// layout. Falls back to the absolute path if the two share no prefix, which
/// [`resolve`] has already ruled out for anything that gets this far.
fn display_path(workspace: &Path, path: &Path) -> String {
    path.strip_prefix(workspace)
        .unwrap_or(path)
        .display()
        .to_string()
}

#[cfg(unix)]
fn read_mode(path: &Path) -> Option<String> {
    use std::os::unix::fs::PermissionsExt;
    let metadata = std::fs::metadata(path).ok()?;
    Some(format!("{:04o}", metadata.permissions().mode() & 0o7777))
}

/// Permission bits are a Unix concept; on every other platform the `mode`
/// field is reported as `null` and setting one is refused rather than
/// silently ignored (see [`set_mode`]).
#[cfg(not(unix))]
fn read_mode(_path: &Path) -> Option<String> {
    None
}

#[cfg(unix)]
fn set_mode(path: &Path, bits: u32) -> FileResult<()> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(bits))
        .map_err(|err| FileError::Io(format!("{}: {err}", path.display())))
}

#[cfg(not(unix))]
fn set_mode(_path: &Path, _bits: u32) -> FileResult<()> {
    Err(FileError::BadRequest(
        "file permissions are only supported on Unix-like platforms".to_string(),
    ))
}

/// Create the parent directory of `path` when `parents` is set; otherwise
/// require it to already be there, so a typo'd directory name fails loudly
/// instead of quietly growing a new tree.
fn prepare_parent(workspace: &Path, path: &Path, parents: bool) -> FileResult<()> {
    let Some(parent) = path.parent() else {
        return Ok(());
    };
    if parent.is_dir() {
        return Ok(());
    }
    let display = display_path(workspace, parent);
    if !parents {
        return Err(FileError::NotFound(format!(
            "{display}: parent directory does not exist (pass \"parents\": true to create it)"
        )));
    }
    std::fs::create_dir_all(parent)
        .map_err(|err| FileError::Io(format!("failed to create directory {display}: {err}")))
}

pub fn create(workspace: &Path, request: CreateFileRequest) -> FileResult<CreateFileResponse> {
    let path = resolve(workspace, &request.path)?;
    let existed = path.exists();
    if existed {
        if !path.is_file() {
            return Err(FileError::NotAFile(format!(
                "{}: exists and is not a regular file",
                display_path(workspace, &path)
            )));
        }
        if !request.overwrite {
            return Err(FileError::AlreadyExists(format!(
                "{}: already exists (the request passed \"overwrite\": false)",
                display_path(workspace, &path)
            )));
        }
    }
    let mode = request.mode.as_ref().map(Mode::bits).transpose()?;

    prepare_parent(workspace, &path, request.parents)?;
    std::fs::write(&path, request.content.as_bytes())
        .map_err(|err| FileError::Io(format!("{}: {err}", path.display())))?;
    if let Some(bits) = mode {
        set_mode(&path, bits)?;
    }

    // Staged after the write, since `git add` records the content that is
    // there now.
    let git = repo_for(workspace, request.git).map(|repo| repo.add(&path));

    Ok(CreateFileResponse {
        path: display_path(workspace, &path),
        bytes_written: request.content.len(),
        mode: read_mode(&path),
        overwritten: existed,
        git,
    })
}

pub fn modify(workspace: &Path, request: ModifyFileRequest) -> FileResult<ModifyFileResponse> {
    let path = resolve(workspace, &request.path)?;
    let display = display_path(workspace, &path);
    if !path.exists() {
        return Err(FileError::NotFound(format!("{display}: no such file")));
    }
    if !path.is_file() {
        return Err(FileError::NotAFile(format!(
            "{display}: not a regular file"
        )));
    }
    if request.edits.is_empty() {
        return Err(FileError::BadRequest(
            "edits must contain at least one change".to_string(),
        ));
    }

    let original =
        std::fs::read(&path).map_err(|err| FileError::Io(format!("{display}: {err}")))?;
    let original = String::from_utf8(original)
        .map_err(|_| FileError::NotUtf8(format!("{display}: not valid UTF-8")))?;
    let (lines, trailing_newline) = split_lines(&original);

    let mut edits: Vec<&LineEdit> = request.edits.iter().collect();
    edits.sort_by_key(|edit| edit.start_line);
    validate_edits(&edits, lines.len(), &display)?;

    // Applied last-first so an earlier edit's line numbers stay valid while
    // the ones after it are still being spliced in — every range in the
    // request refers to the file as it was read.
    let mut updated = lines.clone();
    for edit in edits.iter().rev() {
        let replacement = replacement_lines(&edit.replacement);
        let start = edit.start_line - 1;
        let end = edit.end_line; // exclusive; == start for a pure insert
        updated.splice(start..end, replacement);
    }

    let diff = unified_diff(&display, &lines, &edits);
    let mut content = updated.join("\n");
    if trailing_newline && !content.is_empty() {
        content.push('\n');
    }
    std::fs::write(&path, content.as_bytes())
        .map_err(|err| FileError::Io(format!("{display}: {err}")))?;
    let git = repo_for(workspace, request.git).map(|repo| repo.add(&path));

    Ok(ModifyFileResponse {
        path: display,
        lines_before: lines.len(),
        lines_after: updated.len(),
        edits_applied: edits.len(),
        diff,
        git,
    })
}

pub fn move_(workspace: &Path, request: MoveFileRequest) -> FileResult<MoveFileResponse> {
    let from = resolve(workspace, &request.from)?;
    let to = resolve(workspace, &request.to)?;
    let from_display = display_path(workspace, &from);
    let to_display = display_path(workspace, &to);

    if !from.exists() {
        return Err(FileError::NotFound(format!("{from_display}: no such file")));
    }
    if !from.is_file() {
        return Err(FileError::NotAFile(format!(
            "{from_display}: not a regular file"
        )));
    }
    let overwritten = to.exists();
    if overwritten {
        if !to.is_file() {
            return Err(FileError::NotAFile(format!(
                "{to_display}: exists and is not a regular file"
            )));
        }
        if !request.overwrite {
            return Err(FileError::AlreadyExists(format!(
                "{to_display}: already exists (pass \"overwrite\": true to replace it)"
            )));
        }
    }
    let mode = request.mode.as_ref().map(Mode::bits).transpose()?;

    prepare_parent(workspace, &to, request.parents)?;
    // `git mv` performs the move itself, so a tracked file is renamed *by*
    // Git — the index records a rename rather than a delete plus an add.
    // An untracked file has nothing for Git to rename, so it falls back to
    // a plain filesystem move; so does a request that opted out.
    let git = match repo_for(workspace, request.git) {
        Some(repo) if repo.is_tracked(&from) => {
            let outcome = repo.mv(&from, &to, request.overwrite);
            if let Some(error) = &outcome.error {
                return Err(FileError::Io(format!(
                    "{from_display} -> {to_display}: {error}"
                )));
            }
            Some(outcome)
        }
        repo => {
            std::fs::rename(&from, &to)
                .map_err(|err| FileError::Io(format!("{from_display} -> {to_display}: {err}")))?;
            repo.map(|repo| repo.untracked())
        }
    };
    if let Some(bits) = mode {
        set_mode(&to, bits)?;
    }

    Ok(MoveFileResponse {
        from: from_display,
        to: to_display,
        mode: read_mode(&to),
        overwritten,
        git,
    })
}

pub fn delete(workspace: &Path, request: DeleteFileRequest) -> FileResult<DeleteFileResponse> {
    let path = resolve(workspace, &request.path)?;
    let display = display_path(workspace, &path);
    if !path.exists() {
        return Err(FileError::NotFound(format!("{display}: no such file")));
    }
    // Directories are out of scope on purpose: this API is a file's life
    // cycle, and a recursive delete behind one JSON field is a much bigger
    // gun than anything else here hands out.
    if !path.is_file() {
        return Err(FileError::NotAFile(format!(
            "{display}: not a regular file"
        )));
    }
    // `git rm` deletes the file itself, so a tracked file goes away *and*
    // the deletion is staged in one step. An untracked file is Git's to
    // ignore, so it is simply unlinked.
    let git = match repo_for(workspace, request.git) {
        Some(repo) if repo.is_tracked(&path) => {
            let outcome = repo.rm(&path);
            if let Some(error) = &outcome.error {
                return Err(FileError::Io(format!("{display}: {error}")));
            }
            Some(outcome)
        }
        repo => {
            std::fs::remove_file(&path)
                .map_err(|err| FileError::Io(format!("{display}: {err}")))?;
            repo.map(|repo| repo.untracked())
        }
    };

    Ok(DeleteFileResponse {
        path: display,
        deleted: true,
        git,
    })
}

pub fn show(workspace: &Path, request: ShowFileRequest) -> FileResult<ShowFileResponse> {
    let path = resolve(workspace, &request.path)?;
    let display = display_path(workspace, &path);
    if !path.exists() {
        return Err(FileError::NotFound(format!("{display}: no such file")));
    }
    if !path.is_file() {
        return Err(FileError::NotAFile(format!(
            "{display}: not a regular file"
        )));
    }
    let bytes = std::fs::read(&path).map_err(|err| FileError::Io(format!("{display}: {err}")))?;
    let content = String::from_utf8(bytes)
        .map_err(|_| FileError::NotUtf8(format!("{display}: not valid UTF-8")))?;
    let (lines, _) = split_lines(&content);

    Ok(ShowFileResponse {
        path: display,
        bytes: content.len(),
        lines: lines.len(),
        mode: read_mode(&path),
        content,
    })
}

pub fn create_dir(
    workspace: &Path,
    request: CreateDirectoryRequest,
) -> FileResult<CreateDirectoryResponse> {
    let path = resolve(workspace, &request.path)?;
    let display = display_path(workspace, &path);
    if path.exists() {
        // No `overwrite` counterpart here: replacing a directory that is
        // already there would mean deleting whatever it holds, which is
        // exactly the recursive delete `delete_directory` refuses to be.
        return Err(FileError::AlreadyExists(format!(
            "{display}: already exists"
        )));
    }
    let mode = request.mode.as_ref().map(Mode::bits).transpose()?;

    prepare_parent(workspace, &path, request.parents)?;
    std::fs::create_dir(&path).map_err(|err| FileError::Io(format!("{display}: {err}")))?;
    if let Some(bits) = mode {
        set_mode(&path, bits)?;
    }

    // Git tracks files, not directories: a new (necessarily empty) one has
    // nothing to stage. It becomes visible to Git with the first file
    // created inside it.
    let git = repo_for(workspace, request.git).map(|repo| repo.nothing_to_stage());

    Ok(CreateDirectoryResponse {
        mode: read_mode(&path),
        path: display,
        git,
    })
}

pub fn move_dir(
    workspace: &Path,
    request: MoveDirectoryRequest,
) -> FileResult<MoveDirectoryResponse> {
    let from = resolve(workspace, &request.from)?;
    let to = resolve(workspace, &request.to)?;
    let from_display = display_path(workspace, &from);
    let to_display = display_path(workspace, &to);

    if !from.exists() {
        return Err(FileError::NotFound(format!(
            "{from_display}: no such directory"
        )));
    }
    if !from.is_dir() {
        return Err(FileError::NotADirectory(format!(
            "{from_display}: not a directory"
        )));
    }
    if canonical(&from)? == canonical(workspace)? {
        return Err(FileError::BadRequest(
            "the workspace root itself cannot be moved".to_string(),
        ));
    }
    // No `overwrite` counterpart, unlike `move_file`: replacing a directory
    // means deleting whatever it holds, and nothing in this API deletes a
    // non-empty directory. An existing destination is the caller's to clear.
    if to.exists() {
        return Err(FileError::AlreadyExists(format!(
            "{to_display}: already exists"
        )));
    }
    // `rename` itself refuses this (EINVAL), but "Invalid argument" is a
    // poor way to learn that a directory can't be moved inside itself.
    if to.starts_with(&from) {
        return Err(FileError::BadRequest(format!(
            "{to_display} is inside {from_display}: a directory cannot be moved into itself"
        )));
    }
    let mode = request.mode.as_ref().map(Mode::bits).transpose()?;

    prepare_parent(workspace, &to, request.parents)?;
    // `git mv` on a directory moves the whole subtree and stages every
    // tracked file's rename with it. Either way the move is a single
    // `rename` underneath, so it is atomic — and, on a crossing of
    // filesystems, fails outright (`EXDEV`) rather than half-copying a tree.
    // A directory holding nothing Git tracks is moved on the filesystem
    // alone, since there are no index entries to rewrite.
    let git = match repo_for(workspace, request.git) {
        Some(repo) if repo.holds_tracked_files(&from) => {
            let outcome = repo.mv(&from, &to, false);
            if let Some(error) = &outcome.error {
                return Err(FileError::Io(format!(
                    "{from_display} -> {to_display}: {error}"
                )));
            }
            Some(outcome)
        }
        repo => {
            std::fs::rename(&from, &to)
                .map_err(|err| FileError::Io(format!("{from_display} -> {to_display}: {err}")))?;
            repo.map(|repo| repo.untracked())
        }
    };
    if let Some(bits) = mode {
        set_mode(&to, bits)?;
    }

    Ok(MoveDirectoryResponse {
        from: from_display,
        to: to_display,
        mode: read_mode(&to),
        git,
    })
}

pub fn delete_dir(
    workspace: &Path,
    request: DeleteDirectoryRequest,
) -> FileResult<DeleteDirectoryResponse> {
    let path = resolve(workspace, &request.path)?;
    let display = display_path(workspace, &path);
    if !path.exists() {
        return Err(FileError::NotFound(format!("{display}: no such directory")));
    }
    if !path.is_dir() {
        return Err(FileError::NotADirectory(format!(
            "{display}: not a directory"
        )));
    }
    // The workspace root itself is not the API's to remove: it is what every
    // other request is resolved against, and a server whose root vanished
    // mid-run would fail every call after it in a much more confusing way.
    // Compared canonically, since the request may have reached the same
    // directory by a different (symlinked) route than the root the server
    // holds.
    if canonical(&path)? == canonical(workspace)? {
        return Err(FileError::BadRequest(
            "the workspace root itself cannot be deleted".to_string(),
        ));
    }
    // Emptiness is checked explicitly rather than left to `remove_dir`'s own
    // errno, so the refusal is one stable `not_empty` code on every platform
    // instead of a passed-through OS message.
    let mut entries =
        std::fs::read_dir(&path).map_err(|err| FileError::Io(format!("{display}: {err}")))?;
    if entries.next().is_some() {
        return Err(FileError::NotEmpty(format!(
            "{display}: directory is not empty (only empty directories can be deleted)"
        )));
    }
    std::fs::remove_dir(&path).map_err(|err| FileError::Io(format!("{display}: {err}")))?;
    // An empty directory holds nothing Git tracks, so — as with creating one
    // — there is no index entry to remove.
    let git = repo_for(workspace, request.git).map(|repo| repo.nothing_to_stage());

    Ok(DeleteDirectoryResponse {
        path: display,
        deleted: true,
        git,
    })
}

/// An existing path with every symlink and `..` resolved, for comparing two
/// paths that may name the same directory by different routes.
fn canonical(path: &Path) -> FileResult<PathBuf> {
    path.canonicalize()
        .map_err(|err| FileError::Io(format!("{}: {err}", path.display())))
}

/// Split file content into lines, reporting separately whether the file
/// ended with a newline — so writing the result back can preserve it (or its
/// absence) instead of silently adding or dropping one.
fn split_lines(content: &str) -> (Vec<String>, bool) {
    if content.is_empty() {
        return (Vec::new(), true);
    }
    match content.strip_suffix('\n') {
        Some(body) => (body.split('\n').map(str::to_string).collect(), true),
        None => (content.split('\n').map(str::to_string).collect(), false),
    }
}

/// The lines an edit's `replacement` contributes. An empty replacement
/// contributes nothing (the range is deleted); a trailing newline is
/// tolerated rather than turned into a stray empty line, since a caller
/// sending whole lines will naturally include one.
fn replacement_lines(replacement: &str) -> Vec<String> {
    if replacement.is_empty() {
        return Vec::new();
    }
    replacement
        .strip_suffix('\n')
        .unwrap_or(replacement)
        .split('\n')
        .map(str::to_string)
        .collect()
}

/// Reject line ranges that don't address real lines, or that overlap each
/// other. `edits` is already sorted by `start_line`, so overlap is just a
/// comparison against the previous edit's end.
fn validate_edits(edits: &[&LineEdit], line_count: usize, display: &str) -> FileResult<()> {
    let mut previous_end = 0usize;
    for edit in edits {
        if edit.start_line == 0 {
            return Err(FileError::BadRequest(
                "start_line is 1-based and must be at least 1".to_string(),
            ));
        }
        if edit.end_line + 1 < edit.start_line {
            return Err(FileError::BadRequest(format!(
                "end_line {} is before start_line {} (use end_line = start_line - 1 to insert)",
                edit.end_line, edit.start_line
            )));
        }
        // A pure insert may sit just past the last line (append); a
        // replacement may not.
        if edit.start_line > line_count + 1 {
            return Err(FileError::BadRequest(format!(
                "{display}: start_line {} is past the end of the file ({line_count} lines)",
                edit.start_line
            )));
        }
        if edit.end_line > line_count {
            return Err(FileError::BadRequest(format!(
                "{display}: end_line {} is past the end of the file ({line_count} lines)",
                edit.end_line
            )));
        }
        if edit.start_line <= previous_end {
            return Err(FileError::BadRequest(format!(
                "edits overlap at line {} (ranges must be disjoint)",
                edit.start_line
            )));
        }
        previous_end = edit.end_line;
    }
    Ok(())
}

/// A zero-context unified diff of the applied edits, in file order.
///
/// No diff algorithm is involved: the caller said exactly which lines it was
/// replacing, so each edit is one hunk verbatim. `+++`'s line numbers track
/// the running length change from the edits before it, the same way real
/// unified diff output does.
fn unified_diff(display: &str, original: &[String], edits: &[&LineEdit]) -> String {
    let mut diff = format!("--- a/{display}\n+++ b/{display}\n");
    let mut offset: isize = 0;
    for edit in edits {
        let removed = &original[edit.start_line - 1..edit.end_line];
        let added = replacement_lines(&edit.replacement);
        // A hunk that adds or removes nothing starts *after* the anchor line
        // rather than at it, which is how `diff -U0` numbers an insertion.
        let old_start = if removed.is_empty() {
            edit.start_line - 1
        } else {
            edit.start_line
        };
        let new_start = if added.is_empty() {
            (edit.start_line as isize + offset - 1).max(0)
        } else {
            edit.start_line as isize + offset
        };
        diff.push_str(&format!(
            "@@ -{},{} +{},{} @@\n",
            old_start,
            removed.len(),
            new_start,
            added.len()
        ));
        for line in removed {
            diff.push_str(&format!("-{line}\n"));
        }
        for line in &added {
            diff.push_str(&format!("+{line}\n"));
        }
        offset += added.len() as isize - removed.len() as isize;
    }
    diff
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn workspace_with(files: &[(&str, &str)]) -> TempDir {
        let dir = tempfile::tempdir().expect("workspace");
        for (name, content) in files {
            let path = dir.path().join(name);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).expect("parent");
            }
            fs::write(path, content).expect("seed file");
        }
        dir
    }

    /// A workspace that is also a Git repository, with every seeded file
    /// committed — so the Git-aware paths have something tracked to act on.
    fn git_workspace_with(files: &[(&str, &str)]) -> TempDir {
        let dir = workspace_with(files);
        let git = |args: &[&str]| {
            let status = std::process::Command::new("git")
                .arg("-C")
                .arg(dir.path())
                .args(args)
                .output()
                .expect("run git");
            assert!(
                status.status.success(),
                "git {args:?}: {}",
                String::from_utf8_lossy(&status.stderr)
            );
        };
        git(&["init", "-q"]);
        git(&["config", "user.email", "test@example.com"]);
        git(&["config", "user.name", "Test"]);
        git(&["config", "commit.gpgsign", "false"]);
        if !files.is_empty() {
            git(&["add", "-A"]);
            git(&["commit", "-q", "-m", "initial"]);
        }
        dir
    }

    /// `git diff --cached --name-status`: exactly what is staged, and how.
    fn staged(workspace: &Path) -> String {
        let output = std::process::Command::new("git")
            .arg("-C")
            .arg(workspace)
            .args(["diff", "--cached", "--name-status"])
            .output()
            .expect("git diff --cached");
        String::from_utf8_lossy(&output.stdout).trim().to_string()
    }

    fn commit_count(workspace: &Path) -> String {
        let output = std::process::Command::new("git")
            .arg("-C")
            .arg(workspace)
            .args(["rev-list", "--count", "HEAD"])
            .output()
            .expect("git rev-list");
        String::from_utf8_lossy(&output.stdout).trim().to_string()
    }

    // Request builders. Every one defaults `git` to `false` so a test says
    // explicitly when it is exercising the Git side; the endpoints' own
    // default is the opposite (`git_default`), which
    // `the_git_flag_opts_out_of_staging` covers.
    fn create_request(path: &str, content: &str) -> CreateFileRequest {
        CreateFileRequest {
            path: path.to_string(),
            content: content.to_string(),
            mode: None,
            overwrite: overwrite_default(),
            parents: false,
            git: false,
        }
    }

    fn modify_request(path: &str, edits: Vec<LineEdit>) -> ModifyFileRequest {
        ModifyFileRequest {
            path: path.to_string(),
            edits,
            git: false,
        }
    }

    fn move_request(from: &str, to: &str) -> MoveFileRequest {
        MoveFileRequest {
            from: from.to_string(),
            to: to.to_string(),
            mode: None,
            overwrite: false,
            parents: false,
            git: false,
        }
    }

    fn delete_request(path: &str) -> DeleteFileRequest {
        DeleteFileRequest {
            path: path.to_string(),
            git: false,
        }
    }

    fn show_request(path: &str) -> ShowFileRequest {
        ShowFileRequest {
            path: path.to_string(),
        }
    }

    fn create_dir_request(path: &str) -> CreateDirectoryRequest {
        CreateDirectoryRequest {
            path: path.to_string(),
            mode: None,
            parents: false,
            git: false,
        }
    }

    fn move_dir_request(from: &str, to: &str) -> MoveDirectoryRequest {
        MoveDirectoryRequest {
            from: from.to_string(),
            to: to.to_string(),
            mode: None,
            parents: false,
            git: false,
        }
    }

    fn delete_dir_request(path: &str) -> DeleteDirectoryRequest {
        DeleteDirectoryRequest {
            path: path.to_string(),
            git: false,
        }
    }

    fn edit(start_line: usize, end_line: usize, replacement: &str) -> LineEdit {
        LineEdit {
            start_line,
            end_line,
            replacement: replacement.to_string(),
        }
    }

    #[test]
    fn create_writes_the_file_and_reports_it_relative_to_the_workspace() {
        let dir = workspace_with(&[]);

        let err = create(dir.path(), create_request("src/main.rs", "fn main() {}\n"))
            .expect_err("missing parent");
        assert!(matches!(err, FileError::NotFound(_)));
        // Reported the way the caller addressed it, not as the server's own
        // absolute layout.
        assert!(
            err.message().starts_with("src:"),
            "unexpected message: {}",
            err.message()
        );

        let mut request = create_request("src/main.rs", "fn main() {}\n");
        request.parents = true;
        let response = create(dir.path(), request).expect("create");

        assert_eq!(response.path, "src/main.rs");
        assert_eq!(response.bytes_written, 13);
        assert!(!response.overwritten);
        assert_eq!(
            fs::read_to_string(dir.path().join("src/main.rs")).unwrap(),
            "fn main() {}\n"
        );
    }

    /// Creating a file that is already there is an override — the same on
    /// every surface — and `overwrite: false` is how a caller asks for
    /// create-if-absent instead.
    #[test]
    fn create_overwrites_an_existing_file_by_default() {
        let dir = workspace_with(&[("a.txt", "old\n")]);

        let response = create(dir.path(), create_request("a.txt", "new\n")).expect("overwrite");
        assert!(response.overwritten);
        assert_eq!(
            fs::read_to_string(dir.path().join("a.txt")).unwrap(),
            "new\n"
        );

        let mut request = create_request("a.txt", "newer\n");
        request.overwrite = false;
        let err = create(dir.path(), request).expect_err("create-if-absent");
        assert!(matches!(err, FileError::AlreadyExists(_)));
        assert_eq!(err.code(), "already_exists");
        assert_eq!(
            fs::read_to_string(dir.path().join("a.txt")).unwrap(),
            "new\n"
        );
    }

    /// The wire default is what an omitted field gives a real request.
    #[test]
    fn overwrite_defaults_to_on_when_the_field_is_omitted() {
        let request: CreateFileRequest =
            serde_json::from_str(r#"{"path": "a.txt", "content": "x"}"#).expect("parse");

        assert!(request.overwrite);
    }

    #[cfg(unix)]
    #[test]
    fn create_applies_the_requested_permissions_in_either_notation() {
        let dir = workspace_with(&[]);

        let mut request = create_request("script.sh", "#!/bin/sh\n");
        request.mode = Some(Mode::Text("0750".to_string()));
        assert_eq!(
            create(dir.path(), request).expect("create").mode.as_deref(),
            Some("0750")
        );

        let mut request = create_request("plain.txt", "hi\n");
        request.mode = Some(Mode::Bits(0o600));
        assert_eq!(
            create(dir.path(), request).expect("create").mode.as_deref(),
            Some("0600")
        );
    }

    #[test]
    fn an_unparsable_mode_is_rejected_before_anything_is_written() {
        let dir = workspace_with(&[]);
        let mut request = create_request("a.txt", "hi\n");
        request.mode = Some(Mode::Text("rwxr-xr-x".to_string()));

        let err = create(dir.path(), request).expect_err("bad mode");

        assert!(matches!(err, FileError::BadRequest(_)));
        assert!(!dir.path().join("a.txt").exists(), "nothing was written");
    }

    #[test]
    fn modify_replaces_the_named_lines_and_returns_a_diff() {
        let dir = workspace_with(&[("a.txt", "one\ntwo\nthree\n")]);

        let response = modify(
            dir.path(),
            modify_request("a.txt", vec![edit(2, 2, "TWO\n")]),
        )
        .expect("modify");

        assert_eq!(
            fs::read_to_string(dir.path().join("a.txt")).unwrap(),
            "one\nTWO\nthree\n"
        );
        assert_eq!(response.lines_before, 3);
        assert_eq!(response.lines_after, 3);
        assert_eq!(
            response.diff,
            "--- a/a.txt\n+++ b/a.txt\n@@ -2,1 +2,1 @@\n-two\n+TWO\n"
        );
    }

    /// Several edits in one request all address the file as it was read, so
    /// a caller never has to re-number ranges around its own earlier edits.
    #[test]
    fn multiple_edits_use_the_original_line_numbering() {
        let dir = workspace_with(&[("a.txt", "1\n2\n3\n4\n5\n")]);

        let response = modify(
            dir.path(),
            modify_request(
                "a.txt",
                vec![edit(4, 5, "four+five\n"), edit(1, 1, "one\nuno\n")],
            ),
        )
        .expect("modify");

        assert_eq!(
            fs::read_to_string(dir.path().join("a.txt")).unwrap(),
            "one\nuno\n2\n3\nfour+five\n"
        );
        assert_eq!(response.edits_applied, 2);
        assert_eq!(response.lines_after, 5);
        // The second hunk's `+` line number carries the first hunk's growth.
        assert_eq!(
            response.diff,
            "--- a/a.txt\n+++ b/a.txt\n\
             @@ -1,1 +1,2 @@\n-1\n+one\n+uno\n\
             @@ -4,2 +5,1 @@\n-4\n-5\n+four+five\n"
        );
    }

    #[test]
    fn an_insert_uses_an_empty_range_and_an_empty_replacement_deletes() {
        let dir = workspace_with(&[("a.txt", "one\ntwo\n")]);

        // end_line = start_line - 1: insert before line 2.
        modify(
            dir.path(),
            modify_request("a.txt", vec![edit(2, 1, "one-and-a-half\n")]),
        )
        .expect("insert");
        assert_eq!(
            fs::read_to_string(dir.path().join("a.txt")).unwrap(),
            "one\none-and-a-half\ntwo\n"
        );

        modify(dir.path(), modify_request("a.txt", vec![edit(1, 1, "")])).expect("delete line");
        assert_eq!(
            fs::read_to_string(dir.path().join("a.txt")).unwrap(),
            "one-and-a-half\ntwo\n"
        );
    }

    /// Appending is an insert just past the last line — the one case where
    /// `start_line` is allowed to be `line_count + 1`.
    #[test]
    fn appending_past_the_last_line_is_allowed() {
        let dir = workspace_with(&[("a.txt", "one\n")]);

        modify(
            dir.path(),
            modify_request("a.txt", vec![edit(2, 1, "two\n")]),
        )
        .expect("append");

        assert_eq!(
            fs::read_to_string(dir.path().join("a.txt")).unwrap(),
            "one\ntwo\n"
        );
    }

    #[test]
    fn modify_rejects_ranges_that_are_out_of_bounds_or_overlap() {
        let dir = workspace_with(&[("a.txt", "one\ntwo\n")]);
        let bad = |edits: Vec<LineEdit>| {
            modify(dir.path(), modify_request("a.txt", edits)).expect_err("invalid edits")
        };

        assert!(matches!(
            bad(vec![edit(0, 1, "x\n")]),
            FileError::BadRequest(_)
        ));
        assert!(matches!(
            bad(vec![edit(1, 9, "x\n")]),
            FileError::BadRequest(_)
        ));
        assert!(matches!(
            bad(vec![edit(9, 9, "x\n")]),
            FileError::BadRequest(_)
        ));
        assert!(matches!(
            bad(vec![edit(1, 2, "x\n"), edit(2, 2, "y\n")]),
            FileError::BadRequest(_)
        ));
        assert!(matches!(bad(Vec::new()), FileError::BadRequest(_)));
        // Untouched by any of the rejected requests.
        assert_eq!(
            fs::read_to_string(dir.path().join("a.txt")).unwrap(),
            "one\ntwo\n"
        );
    }

    /// A file that doesn't end in a newline keeps that property; one that
    /// does keeps that too.
    #[test]
    fn modify_preserves_the_files_trailing_newline_state() {
        let dir = workspace_with(&[("with.txt", "a\nb\n"), ("without.txt", "a\nb")]);

        for name in ["with.txt", "without.txt"] {
            modify(dir.path(), modify_request(name, vec![edit(1, 1, "A")])).expect("modify");
        }

        assert_eq!(
            fs::read_to_string(dir.path().join("with.txt")).unwrap(),
            "A\nb\n"
        );
        assert_eq!(
            fs::read_to_string(dir.path().join("without.txt")).unwrap(),
            "A\nb"
        );
    }

    #[test]
    fn move_renames_the_file_and_can_set_its_permissions() {
        let dir = workspace_with(&[("a.txt", "content\n")]);
        let mut request = move_request("a.txt", "sub/b.txt");
        request.mode = Some(Mode::Text("0640".to_string()));
        request.parents = true;

        let response = move_(dir.path(), request).expect("move");

        assert_eq!(response.from, "a.txt");
        assert_eq!(response.to, "sub/b.txt");
        assert!(!dir.path().join("a.txt").exists());
        assert_eq!(
            fs::read_to_string(dir.path().join("sub/b.txt")).unwrap(),
            "content\n"
        );
        #[cfg(unix)]
        assert_eq!(response.mode.as_deref(), Some("0640"));
    }

    #[test]
    fn move_refuses_an_existing_destination_unless_asked() {
        let dir = workspace_with(&[("a.txt", "a\n"), ("b.txt", "b\n")]);
        let request = |overwrite| {
            let mut request = move_request("a.txt", "b.txt");
            request.overwrite = overwrite;
            request
        };

        let err = move_(dir.path(), request(false)).expect_err("destination exists");
        assert!(matches!(err, FileError::AlreadyExists(_)));

        let response = move_(dir.path(), request(true)).expect("overwrite");
        assert!(response.overwritten);
        assert_eq!(fs::read_to_string(dir.path().join("b.txt")).unwrap(), "a\n");
    }

    #[test]
    fn delete_removes_a_file_but_never_a_directory() {
        let dir = workspace_with(&[("sub/a.txt", "a\n")]);

        let err = delete(dir.path(), delete_request("sub")).expect_err("directory");
        assert!(matches!(err, FileError::NotAFile(_)));

        let response = delete(dir.path(), delete_request("sub/a.txt")).expect("delete");
        assert!(response.deleted);
        assert!(!dir.path().join("sub/a.txt").exists());

        let err = delete(dir.path(), delete_request("sub/a.txt")).expect_err("already gone");
        assert!(matches!(err, FileError::NotFound(_)));
        assert_eq!(err.code(), "not_found");
    }

    #[test]
    fn show_returns_the_whole_file_with_its_line_and_byte_counts() {
        let dir = workspace_with(&[("a.txt", "one\ntwo\n")]);

        let response = show(dir.path(), show_request("a.txt")).expect("show");

        assert_eq!(response.content, "one\ntwo\n");
        assert_eq!(response.bytes, 8);
        assert_eq!(response.lines, 2);
        assert_eq!(response.path, "a.txt");
    }

    #[test]
    fn a_file_that_is_not_utf8_is_reported_rather_than_mangled() {
        let dir = workspace_with(&[]);
        fs::write(dir.path().join("blob.bin"), [0xff, 0xfe, 0x00]).expect("write");

        let err = show(dir.path(), show_request("blob.bin")).expect_err("binary");

        assert!(matches!(err, FileError::NotUtf8(_)));
        assert_eq!(err.code(), "not_utf8");
    }

    /// Every endpoint refuses a path outside the workspace, whether it
    /// climbs out with `..` or names an absolute path elsewhere.
    #[test]
    fn paths_outside_the_workspace_are_refused_by_every_endpoint() {
        let dir = workspace_with(&[("a.txt", "a\n")]);
        let outside = workspace_with(&[("secret.txt", "secret\n")]);
        let escapes = [
            "../secret.txt".to_string(),
            outside.path().join("secret.txt").display().to_string(),
        ];

        for path in escapes {
            assert!(matches!(
                create(dir.path(), create_request(&path, "x\n")).expect_err("create"),
                FileError::OutsideWorkspace(_)
            ));
            assert!(matches!(
                modify(dir.path(), modify_request(&path, vec![edit(1, 1, "x\n")]))
                    .expect_err("modify"),
                FileError::OutsideWorkspace(_)
            ));
            assert!(matches!(
                move_(dir.path(), move_request("a.txt", &path)).expect_err("move to"),
                FileError::OutsideWorkspace(_)
            ));
            assert!(matches!(
                move_(dir.path(), move_request(&path, "b.txt")).expect_err("move from"),
                FileError::OutsideWorkspace(_)
            ));
            assert!(matches!(
                delete(dir.path(), delete_request(&path)).expect_err("delete"),
                FileError::OutsideWorkspace(_)
            ));
            assert!(matches!(
                show(dir.path(), show_request(&path)).expect_err("show"),
                FileError::OutsideWorkspace(_)
            ));
        }

        assert_eq!(
            fs::read_to_string(outside.path().join("secret.txt")).unwrap(),
            "secret\n",
            "the file outside the workspace was left alone"
        );
    }

    /// The lexical check can't see a symlink, so a link *inside* the
    /// workspace pointing out of it is caught by canonicalizing the nearest
    /// existing ancestor instead.
    #[cfg(unix)]
    #[test]
    fn a_symlink_pointing_out_of_the_workspace_is_refused() {
        let dir = workspace_with(&[]);
        let outside = workspace_with(&[("secret.txt", "secret\n")]);
        std::os::unix::fs::symlink(outside.path(), dir.path().join("escape")).expect("symlink");

        let err = show(dir.path(), show_request("escape/secret.txt")).expect_err("symlinked out");
        assert!(matches!(err, FileError::OutsideWorkspace(_)));

        let err =
            create(dir.path(), create_request("escape/new.txt", "x\n")).expect_err("symlinked out");
        assert!(matches!(err, FileError::OutsideWorkspace(_)));
        assert!(!outside.path().join("new.txt").exists());
    }

    #[test]
    fn create_directory_makes_one_directory_and_can_set_its_permissions() {
        let dir = workspace_with(&[]);

        let err = create_dir(dir.path(), create_dir_request("a/b/c")).expect_err("missing parents");
        assert!(matches!(err, FileError::NotFound(_)));

        let mut request = create_dir_request("a/b/c");
        request.mode = Some(Mode::Text("0700".to_string()));
        request.parents = true;
        let response = create_dir(dir.path(), request).expect("create");

        assert_eq!(response.path, "a/b/c");
        assert!(dir.path().join("a/b/c").is_dir());
        #[cfg(unix)]
        {
            assert_eq!(response.mode.as_deref(), Some("0700"));
            // `mode` applies to the directory named by `path`; the parents
            // created along the way keep the umask's own permissions, the
            // same way `mkdir -p -m` behaves.
            assert_ne!(read_mode(&dir.path().join("a")).as_deref(), Some("0700"));
        }
    }

    #[test]
    fn create_directory_refuses_a_path_that_already_exists() {
        let dir = workspace_with(&[("sub/a.txt", "a\n")]);

        for path in ["sub", "sub/a.txt"] {
            let err = create_dir(dir.path(), create_dir_request(path)).expect_err("exists");
            assert!(matches!(err, FileError::AlreadyExists(_)), "{path}");
        }
    }

    #[test]
    fn move_directory_moves_the_whole_subtree() {
        let dir = workspace_with(&[("src/deep/a.txt", "a\n"), ("src/b.txt", "b\n")]);
        let mut request = move_dir_request("src", "lib/src");
        request.mode = Some(Mode::Text("0750".to_string()));
        request.parents = true;

        let response = move_dir(dir.path(), request).expect("move");

        assert_eq!(response.from, "src");
        assert_eq!(response.to, "lib/src");
        assert!(!dir.path().join("src").exists());
        assert_eq!(
            fs::read_to_string(dir.path().join("lib/src/deep/a.txt")).unwrap(),
            "a\n"
        );
        assert_eq!(
            fs::read_to_string(dir.path().join("lib/src/b.txt")).unwrap(),
            "b\n"
        );
        #[cfg(unix)]
        assert_eq!(response.mode.as_deref(), Some("0750"));
    }

    #[test]
    fn move_directory_refuses_an_existing_destination_or_a_move_into_itself() {
        let dir = workspace_with(&[("src/a.txt", "a\n"), ("docs/b.txt", "b\n")]);
        let attempt = |from: &str, to: &str| {
            let mut request = move_dir_request(from, to);
            request.parents = true;
            move_dir(dir.path(), request)
        };

        // No `overwrite` here: clearing a non-empty destination is the
        // caller's to do, not something one JSON field triggers.
        assert!(matches!(
            attempt("src", "docs").expect_err("destination exists"),
            FileError::AlreadyExists(_)
        ));
        assert!(matches!(
            attempt("src", "src/nested").expect_err("into itself"),
            FileError::BadRequest(_)
        ));
        assert!(matches!(
            attempt("src/a.txt", "elsewhere").expect_err("a file"),
            FileError::NotADirectory(_)
        ));
        assert!(matches!(
            attempt("gone", "elsewhere").expect_err("missing"),
            FileError::NotFound(_)
        ));
        assert!(matches!(
            attempt(".", "moved-root").expect_err("workspace root"),
            FileError::BadRequest(_)
        ));

        // Every refusal left the tree exactly as it was.
        assert_eq!(
            fs::read_to_string(dir.path().join("src/a.txt")).unwrap(),
            "a\n"
        );
        assert_eq!(
            fs::read_to_string(dir.path().join("docs/b.txt")).unwrap(),
            "b\n"
        );
    }

    #[test]
    fn delete_directory_only_removes_an_empty_directory() {
        let dir = workspace_with(&[("sub/a.txt", "a\n")]);
        let delete_at = |path: &str| delete_dir(dir.path(), delete_dir_request(path));

        let err = delete_at("sub").expect_err("not empty");
        assert!(matches!(err, FileError::NotEmpty(_)));
        assert_eq!(err.code(), "not_empty");
        assert!(dir.path().join("sub/a.txt").exists());

        // A regular file is not a directory, and neither is a missing path.
        assert!(matches!(
            delete_at("sub/a.txt").expect_err("a file"),
            FileError::NotADirectory(_)
        ));
        assert!(matches!(
            delete_at("gone").expect_err("missing"),
            FileError::NotFound(_)
        ));

        fs::remove_file(dir.path().join("sub/a.txt")).expect("empty it");
        assert!(delete_at("sub").expect("delete").deleted);
        assert!(!dir.path().join("sub").exists());
    }

    /// Deleting the workspace root would leave every later request resolving
    /// against a directory that no longer exists.
    #[test]
    fn delete_directory_refuses_the_workspace_root() {
        let dir = workspace_with(&[]);

        for path in [".", dir.path().display().to_string().as_str()] {
            let err = delete_dir(dir.path(), delete_dir_request(path)).expect_err("workspace root");
            assert!(matches!(err, FileError::BadRequest(_)), "{path}");
        }
        assert!(dir.path().is_dir());
    }

    #[test]
    fn the_directory_endpoints_stay_inside_the_workspace_too() {
        let dir = workspace_with(&[]);
        let outside = workspace_with(&[]);
        fs::create_dir(outside.path().join("empty")).expect("seed");

        let mut escaping_create = create_dir_request("../escaped");
        escaping_create.parents = true;
        assert!(matches!(
            create_dir(dir.path(), escaping_create).expect_err("create outside"),
            FileError::OutsideWorkspace(_)
        ));
        assert!(matches!(
            delete_dir(
                dir.path(),
                delete_dir_request(&outside.path().join("empty").display().to_string()),
            )
            .expect_err("delete outside"),
            FileError::OutsideWorkspace(_)
        ));
        fs::create_dir(dir.path().join("inside")).expect("seed");
        let mut escaping_move = move_dir_request("inside", "../escaped");
        escaping_move.parents = true;
        assert!(matches!(
            move_dir(dir.path(), escaping_move).expect_err("move outside"),
            FileError::OutsideWorkspace(_)
        ));
        assert!(outside.path().join("empty").is_dir());
        assert!(dir.path().join("inside").is_dir());
    }

    #[test]
    fn an_empty_path_is_a_bad_request() {
        let dir = workspace_with(&[]);

        let err = show(dir.path(), show_request("  ")).expect_err("empty path");

        assert!(matches!(err, FileError::BadRequest(_)));
    }

    // --- Git ---------------------------------------------------------------
    //
    // In a repository a file is created, modified, moved and deleted *with*
    // its Git command, so the change lands in the index. Nothing is ever
    // committed; `nothing_is_ever_committed` is the standing check on that.

    #[test]
    fn create_and_modify_stage_the_file_with_git_add() {
        let dir = git_workspace_with(&[("seed.txt", "seed\n")]);

        let mut request = create_request("new.txt", "new\n");
        request.git = true;
        let response = create(dir.path(), request).expect("create");

        let git = response.git.expect("git outcome");
        assert!(git.staged, "{git:?}");
        assert_eq!(git.command.as_deref(), Some("git add new.txt"));
        assert!(git.error.is_none());
        assert!(staged(dir.path()).contains("new.txt"));

        let mut request = modify_request("seed.txt", vec![edit(1, 1, "changed\n")]);
        request.git = true;
        let git = modify(dir.path(), request)
            .expect("modify")
            .git
            .expect("git outcome");

        assert!(git.staged, "{git:?}");
        assert_eq!(git.command.as_deref(), Some("git add seed.txt"));
        assert!(staged(dir.path()).contains("M\tseed.txt"));
    }

    /// `git mv` rather than a rename plus two index updates, so the move is
    /// recorded as a rename.
    #[test]
    fn move_uses_git_mv_and_the_index_records_a_rename() {
        let dir = git_workspace_with(&[("a.txt", "content\n")]);

        let mut request = move_request("a.txt", "sub/b.txt");
        request.parents = true;
        request.git = true;
        let response = move_(dir.path(), request).expect("move");

        let git = response.git.expect("git outcome");
        assert!(git.staged, "{git:?}");
        assert_eq!(git.command.as_deref(), Some("git mv a.txt sub/b.txt"));
        assert!(!dir.path().join("a.txt").exists());
        assert_eq!(
            fs::read_to_string(dir.path().join("sub/b.txt")).unwrap(),
            "content\n"
        );
        assert!(
            staged(dir.path()).starts_with('R'),
            "expected a rename, got: {}",
            staged(dir.path())
        );
    }

    #[test]
    fn delete_uses_git_rm_and_stages_the_deletion() {
        let dir = git_workspace_with(&[("a.txt", "content\n")]);

        let mut request = delete_request("a.txt");
        request.git = true;
        let git = delete(dir.path(), request)
            .expect("delete")
            .git
            .expect("git outcome");

        assert!(git.staged, "{git:?}");
        assert_eq!(git.command.as_deref(), Some("git rm -f a.txt"));
        assert!(!dir.path().join("a.txt").exists());
        assert!(staged(dir.path()).contains("D\ta.txt"));
    }

    /// A directory move takes every tracked file under it along, as renames.
    #[test]
    fn move_directory_uses_git_mv_for_a_tracked_tree() {
        let dir = git_workspace_with(&[("src/a.txt", "a\n"), ("src/deep/b.txt", "b\n")]);

        let mut request = move_dir_request("src", "lib/src");
        request.parents = true;
        request.git = true;
        let git = move_dir(dir.path(), request)
            .expect("move")
            .git
            .expect("git outcome");

        assert!(git.staged, "{git:?}");
        assert_eq!(git.command.as_deref(), Some("git mv src lib/src"));
        let staged = staged(dir.path());
        assert!(staged.contains("lib/src/a.txt"), "{staged}");
        assert!(staged.contains("lib/src/deep/b.txt"), "{staged}");
    }

    /// Git tracks files, not directories, so creating or removing an empty
    /// one leaves the index alone — reported rather than silently absent.
    #[test]
    fn directory_creation_and_removal_have_nothing_to_stage() {
        let dir = git_workspace_with(&[("seed.txt", "seed\n")]);

        let mut request = create_dir_request("empty");
        request.git = true;
        let git = create_dir(dir.path(), request)
            .expect("create")
            .git
            .expect("git outcome");
        assert!(!git.staged);
        assert_eq!(git.skipped, Some("nothing_to_stage"));

        let mut request = delete_dir_request("empty");
        request.git = true;
        let git = delete_dir(dir.path(), request)
            .expect("delete")
            .git
            .expect("git outcome");
        assert!(!git.staged);
        assert_eq!(git.skipped, Some("nothing_to_stage"));
        assert_eq!(staged(dir.path()), "", "the index was left alone");
    }

    /// An untracked file has nothing for Git to move or delete, so those
    /// stay plain filesystem operations — reported as `untracked` rather
    /// than as a failure.
    #[test]
    fn an_untracked_file_is_moved_and_deleted_on_the_filesystem_alone() {
        let dir = git_workspace_with(&[("seed.txt", "seed\n")]);
        fs::write(dir.path().join("loose.txt"), "loose\n").expect("write");

        let mut request = move_request("loose.txt", "moved.txt");
        request.git = true;
        let git = move_(dir.path(), request)
            .expect("move")
            .git
            .expect("git outcome");
        assert!(!git.staged);
        assert_eq!(git.skipped, Some("untracked"));
        assert!(dir.path().join("moved.txt").exists());

        let mut request = delete_request("moved.txt");
        request.git = true;
        let git = delete(dir.path(), request)
            .expect("delete")
            .git
            .expect("git outcome");
        assert!(!git.staged);
        assert_eq!(git.skipped, Some("untracked"));
        assert!(!dir.path().join("moved.txt").exists());
        assert_eq!(staged(dir.path()), "", "the index was left alone");
    }

    /// `git add` refuses an ignored path outright; that is reported as a
    /// skip, not an error the caller can do nothing about.
    #[test]
    fn an_ignored_file_is_written_but_not_staged() {
        let dir = git_workspace_with(&[(".gitignore", "build/\n")]);
        fs::create_dir(dir.path().join("build")).expect("dir");

        let mut request = create_request("build/out.txt", "out\n");
        request.git = true;
        let git = create(dir.path(), request)
            .expect("create")
            .git
            .expect("git outcome");

        assert!(!git.staged);
        assert_eq!(git.skipped, Some("ignored"));
        assert!(git.error.is_none(), "{git:?}");
        assert_eq!(
            fs::read_to_string(dir.path().join("build/out.txt")).unwrap(),
            "out\n"
        );
    }

    /// `"git": false` is the way back to a pure filesystem change; the
    /// endpoints' own default is the opposite.
    #[test]
    fn the_git_flag_opts_out_of_staging() {
        let dir = git_workspace_with(&[("a.txt", "a\n")]);

        // The serde default is what an omitted field gives a real request.
        let parsed: CreateFileRequest =
            serde_json::from_str(r#"{"path": "b.txt", "content": "b\n"}"#).expect("parse");
        assert!(parsed.git, "git defaults to on");

        let response = create(dir.path(), create_request("c.txt", "c\n")).expect("create");
        assert!(response.git.is_none(), "opted out: no git outcome");
        assert_eq!(staged(dir.path()), "", "the index was left alone");
        assert!(dir.path().join("c.txt").exists());
    }

    /// Outside a repository every endpoint behaves exactly as before, with
    /// no Git outcome to report.
    #[test]
    fn outside_a_repository_there_is_no_git_outcome() {
        let dir = workspace_with(&[("a.txt", "a\n")]);

        let mut request = create_request("b.txt", "b\n");
        request.git = true;
        let response = create(dir.path(), request).expect("create");

        assert!(response.git.is_none());
        assert!(dir.path().join("b.txt").exists());
    }

    /// The whole point of the Git wiring: it stops at the index. A full
    /// create/modify/move/delete cycle leaves the commit count untouched.
    #[test]
    fn nothing_is_ever_committed() {
        let dir = git_workspace_with(&[("a.txt", "a\n")]);
        let before = commit_count(dir.path());

        let mut request = create_request("b.txt", "b\n");
        request.git = true;
        create(dir.path(), request).expect("create");

        let mut request = modify_request("a.txt", vec![edit(1, 1, "changed\n")]);
        request.git = true;
        modify(dir.path(), request).expect("modify");

        let mut request = move_request("a.txt", "moved.txt");
        request.git = true;
        move_(dir.path(), request).expect("move");

        let mut request = delete_request("moved.txt");
        request.git = true;
        delete(dir.path(), request).expect("delete");

        assert_eq!(
            commit_count(dir.path()),
            before,
            "the API must never create a commit"
        );
        assert!(
            !staged(dir.path()).is_empty(),
            "the changes are staged, waiting for the user's own commit"
        );
    }
}
