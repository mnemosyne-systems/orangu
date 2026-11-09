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

//! Checkpoints of the files the agent writes, one per turn, so `/undo` and
//! `/rewind` can put them back.
//!
//! Before a file tool first touches a path in a turn, the path's content,
//! permissions and Git index entry are copied under the session directory
//! (`<session>/checkpoints/<turn>/`), apart from Git's history and index.
//! When the turn ends the state the agent left is fingerprinted, so a later
//! restore can tell when someone changed a file since. Restoring a turn
//! writes every saved file back, removes the files and directories the turn
//! created, and puts the index entries back as they were.

use std::collections::HashSet;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};

use anyhow::{Context as _, Result, anyhow};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The most file content one turn's checkpoint copies; files past it are
/// listed as not saved and left alone by a restore.
pub const MAX_TURN_BYTES: u64 = 64 * 1024 * 1024;

/// How many turns with changes a session keeps; older ones are dropped.
pub const MAX_TURNS: usize = 100;

const MANIFEST: &str = "manifest.json";

/// One path's state before a turn first changed it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FileEntry {
    /// Absolute path.
    pub path: PathBuf,
    /// The copy of its content (a file name in the turn's directory), or
    /// `None` when the path did not exist.
    #[serde(default)]
    pub blob: Option<String>,
    /// Where it pointed, when it was a symbolic link.
    #[serde(default)]
    pub link: Option<PathBuf>,
    /// Its permission bits.
    #[serde(default)]
    pub mode: Option<u32>,
    /// Its Git index entry (`<mode> <object>`), `None` when untracked.
    #[serde(default)]
    pub index: Option<String>,
    /// The state the turn left, filled in when the turn ends.
    #[serde(default)]
    pub after: Option<Fingerprint>,
}

/// A directory a turn created, or one that held saved files.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DirEntry {
    pub path: PathBuf,
    /// Whether it existed before the turn.
    pub existed: bool,
}

/// What a path holds, for telling whether it changed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Fingerprint {
    Absent,
    Link(PathBuf),
    Content(String),
    Directory,
}

/// One turn's checkpoint, as stored in `manifest.json`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub id: u64,
    /// The first line of the prompt that started the turn.
    pub prompt: String,
    /// Seconds since the epoch.
    pub started_at: u64,
    pub files: Vec<FileEntry>,
    #[serde(default)]
    pub dirs: Vec<DirEntry>,
    /// Files the turn changed that were too large to save.
    #[serde(default)]
    pub not_saved: Vec<PathBuf>,
    /// Whether the Git index was saved too.
    #[serde(default)]
    pub git: bool,
    /// Whether the turn has ended (and `after` is filled in).
    #[serde(default)]
    pub finished: bool,
}

/// A turn as `/rewind` lists it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TurnSummary {
    pub id: u64,
    pub prompt: String,
    pub started_at: u64,
    /// Workspace-relative paths the turn changed.
    pub files: Vec<String>,
    /// Paths it changed but could not save.
    pub not_saved: Vec<String>,
}

/// What a restore did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Restored {
    /// The turns restored, newest first.
    pub turns: Vec<TurnSummary>,
    /// Workspace-relative paths written back.
    pub written: Vec<String>,
    /// Workspace-relative paths removed (the turns created them).
    pub removed: Vec<String>,
    /// Paths changed since the turn that were overwritten (`force`).
    pub overwritten: Vec<String>,
    /// Paths the turns changed that were never saved and stay as they are.
    pub not_saved: Vec<String>,
}

/// Why a restore did nothing.
#[derive(Debug, PartialEq, Eq)]
pub enum RestoreError {
    /// No turn with changes is saved.
    Nothing,
    /// No saved turn has this number.
    NoSuchTurn(u64),
    /// These workspace-relative paths changed after the agent wrote them;
    /// restoring would discard that.
    Changed(Vec<String>),
    /// Reading or writing failed.
    Failed(String),
}

impl std::fmt::Display for RestoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Nothing => write!(f, "no agent changes to undo in this session"),
            Self::NoSuchTurn(id) => write!(f, "no saved turn {id}"),
            Self::Changed(paths) => write!(
                f,
                "changed since the agent wrote them: {}",
                paths.join(", ")
            ),
            Self::Failed(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for RestoreError {}

/// The checkpoints of one session's workspace, shared by every clone of
/// the session's tool executor and its subagents.
#[derive(Clone)]
pub struct Checkpoints {
    inner: Arc<Mutex<State>>,
}

struct State {
    workspace: PathBuf,
    /// `<session>/checkpoints`; `None` keeps no checkpoints.
    dir: Option<PathBuf>,
    open: Option<OpenTurn>,
    /// The repository's top level, once asked.
    git_root: Option<Option<PathBuf>>,
    /// What to tell the model about the last restore.
    notice: Option<String>,
}

struct OpenTurn {
    prompt: String,
    started_at: u64,
    /// Written on the turn's first change.
    manifest: Option<Manifest>,
    seen: HashSet<PathBuf>,
    seen_dirs: HashSet<PathBuf>,
    bytes: u64,
}

impl Checkpoints {
    /// Checkpoints for `workspace`, kept under `dir`; `None` keeps none.
    pub fn new(workspace: &Path, dir: Option<PathBuf>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(State {
                workspace: workspace.to_path_buf(),
                dir,
                open: None,
                git_root: None,
                notice: None,
            })),
        }
    }

    /// Whether anything is kept at all.
    pub fn enabled(&self) -> bool {
        self.inner.lock().is_ok_and(|state| state.dir.is_some())
    }

    /// Starts a turn for `prompt`, ending the one before it.
    pub fn begin_turn(&self, prompt: &str) {
        let Ok(mut state) = self.inner.lock() else {
            return;
        };
        state.finish();
        if state.dir.is_none() {
            return;
        }
        state.open = Some(OpenTurn {
            prompt: first_line(prompt),
            started_at: unix_now(),
            manifest: None,
            seen: HashSet::new(),
            seen_dirs: HashSet::new(),
            bytes: 0,
        });
    }

    /// Ends the open turn, recording the state it left.
    pub fn finish_turn(&self) {
        if let Ok(mut state) = self.inner.lock() {
            state.finish();
        }
    }

    /// Saves `paths` (files or whole directories) before they change, the
    /// first time this turn touches each.
    pub fn record(&self, paths: &[PathBuf]) {
        if let Ok(mut state) = self.inner.lock() {
            state.record(paths);
        }
    }

    /// Saves both ends of a move from `from` to `to`.
    pub fn record_move(&self, from: &Path, to: &Path) {
        let mut paths = vec![from.to_path_buf(), to.to_path_buf()];
        if from.is_dir() {
            paths.extend(
                walk(from)
                    .into_iter()
                    .filter_map(|(path, _)| path.strip_prefix(from).ok().map(|rel| to.join(rel))),
            );
        }
        self.record(&paths);
    }

    /// The saved turns, oldest first.
    pub fn list(&self) -> Vec<TurnSummary> {
        let Ok(mut state) = self.inner.lock() else {
            return Vec::new();
        };
        state.finish();
        state
            .manifests()
            .iter()
            .map(|manifest| state.summary(manifest))
            .collect()
    }

    /// Puts back the files the last turn with changes wrote.
    pub fn undo(&self, force: bool) -> Result<Restored, RestoreError> {
        let Ok(mut state) = self.inner.lock() else {
            return Err(RestoreError::Failed("checkpoint lock poisoned".into()));
        };
        state.finish();
        let last = state
            .manifests()
            .last()
            .map(|manifest| manifest.id)
            .ok_or(RestoreError::Nothing)?;
        state.restore_from(last, force)
    }

    /// Puts the files back as they were before turn `id`, undoing it and
    /// every later turn.
    pub fn rewind(&self, id: u64, force: bool) -> Result<Restored, RestoreError> {
        let Ok(mut state) = self.inner.lock() else {
            return Err(RestoreError::Failed("checkpoint lock poisoned".into()));
        };
        state.finish();
        state.restore_from(id, force)
    }

    /// What to tell the model about the last restore, once.
    pub fn take_notice(&self) -> Option<String> {
        self.inner
            .lock()
            .ok()
            .and_then(|mut state| state.notice.take())
    }
}

impl State {
    fn turn_dir(&self, id: u64) -> Option<PathBuf> {
        self.dir.as_ref().map(|dir| dir.join(id.to_string()))
    }

    fn relative(&self, path: &Path) -> String {
        crate::files::to_slash(path.strip_prefix(&self.workspace).unwrap_or(path))
    }

    fn summary(&self, manifest: &Manifest) -> TurnSummary {
        TurnSummary {
            id: manifest.id,
            prompt: manifest.prompt.clone(),
            started_at: manifest.started_at,
            files: manifest
                .files
                .iter()
                .map(|entry| self.relative(&entry.path))
                .collect(),
            not_saved: manifest
                .not_saved
                .iter()
                .map(|path| self.relative(path))
                .collect(),
        }
    }

    /// Every saved turn, oldest first.
    fn manifests(&self) -> Vec<Manifest> {
        let Some(dir) = &self.dir else {
            return Vec::new();
        };
        let Ok(entries) = fs::read_dir(dir) else {
            return Vec::new();
        };
        let mut manifests: Vec<Manifest> = entries
            .flatten()
            .filter_map(|entry| fs::read(entry.path().join(MANIFEST)).ok())
            .filter_map(|bytes| serde_json::from_slice(&bytes).ok())
            .collect();
        manifests.sort_by_key(|manifest: &Manifest| manifest.id);
        manifests
    }

    fn git_root(&mut self) -> Option<PathBuf> {
        if self.git_root.is_none() {
            let root = Command::new("git")
                .arg("-C")
                .arg(&self.workspace)
                .args(["rev-parse", "--show-toplevel"])
                .stderr(Stdio::null())
                .output()
                .ok()
                .filter(|output| output.status.success())
                .and_then(|output| String::from_utf8(output.stdout).ok())
                .map(|root| PathBuf::from(root.trim_end_matches(['\n', '\r'])))
                .filter(|root| !root.as_os_str().is_empty());
            self.git_root = Some(root);
        }
        self.git_root.clone().flatten()
    }

    /// `path` relative to the repository's top level, `/`-separated as git
    /// lists it. Both sides are canonicalized: git prints its top level in
    /// its own form, which on Windows shares no prefix with a canonical path.
    fn repo_relative(&mut self, path: &Path) -> Option<String> {
        let root = self.git_root()?.canonicalize().ok()?;
        let workspace = self.workspace.canonicalize().ok()?;
        let inside = workspace.join(path.strip_prefix(&self.workspace).ok()?);
        inside.strip_prefix(&root).ok().map(crate::files::to_slash)
    }

    /// The index entries (`<mode> <object>`) of `paths`, by path.
    fn index_entries(&mut self, paths: &[PathBuf]) -> Vec<(PathBuf, Option<String>)> {
        let Some(root) = self.git_root() else {
            return paths.iter().map(|path| (path.clone(), None)).collect();
        };
        let relative: Vec<(PathBuf, String)> = paths
            .iter()
            .filter_map(|path| Some((path.clone(), self.repo_relative(path)?)))
            .collect();
        let mut found = std::collections::HashMap::new();
        if !relative.is_empty() {
            let output = Command::new("git")
                .arg("-C")
                .arg(&root)
                .args(["ls-files", "-s", "-z", "--"])
                .args(relative.iter().map(|(_, rel)| rel))
                .stderr(Stdio::null())
                .output();
            if let Ok(output) = output {
                for record in output.stdout.split(|&byte| byte == 0) {
                    let record = String::from_utf8_lossy(record);
                    let Some((meta, path)) = record.split_once('\t') else {
                        continue;
                    };
                    let fields: Vec<&str> = meta.split(' ').collect();
                    if let [mode, object, "0"] = fields.as_slice() {
                        found.insert(path.to_string(), format!("{mode} {object}"));
                    }
                }
            }
        }
        paths
            .iter()
            .map(|path| {
                let entry = relative
                    .iter()
                    .find(|(candidate, _)| candidate == path)
                    .and_then(|(_, rel)| found.get(rel).cloned());
                (path.clone(), entry)
            })
            .collect()
    }

    fn record(&mut self, paths: &[PathBuf]) {
        if self.dir.is_none() {
            return;
        }
        if self.open.is_none() {
            // A background subagent writing between prompts.
            self.open = Some(OpenTurn {
                prompt: "(background work)".into(),
                started_at: unix_now(),
                manifest: None,
                seen: HashSet::new(),
                seen_dirs: HashSet::new(),
                bytes: 0,
            });
        }
        let mut files = Vec::new();
        let mut dirs = Vec::new();
        for path in paths {
            if !path.starts_with(&self.workspace) {
                continue;
            }
            match fs::symlink_metadata(path) {
                Ok(meta) if meta.is_dir() => {
                    dirs.push(path.clone());
                    for (inner, is_dir) in walk(path) {
                        if is_dir {
                            dirs.push(inner);
                        } else {
                            files.push(inner);
                        }
                    }
                }
                Ok(_) => files.push(path.clone()),
                Err(_) => {
                    // Absent: the tool may make it a file or a directory.
                    files.push(path.clone());
                    dirs.push(path.clone());
                    let mut parent = path.parent();
                    while let Some(dir) = parent {
                        if dir.exists() || !dir.starts_with(&self.workspace) {
                            break;
                        }
                        dirs.push(dir.to_path_buf());
                        parent = dir.parent();
                    }
                }
            }
        }
        let open = self.open.as_mut().expect("opened above");
        files.retain(|path| open.seen.insert(path.clone()));
        dirs.retain(|path| open.seen_dirs.insert(path.clone()));
        if files.is_empty() && dirs.is_empty() {
            return;
        }
        if let Err(err) = self.save(files, dirs) {
            log::warn!("checkpoint: {err:#}");
        }
    }

    fn save(&mut self, files: Vec<PathBuf>, dirs: Vec<PathBuf>) -> Result<()> {
        let index = self.index_entries(&files);
        let git = self.git_root().is_some();
        let next_id = self
            .manifests()
            .last()
            .map_or(1, |manifest| manifest.id + 1);
        let open = self.open.as_mut().expect("a turn is open");
        let manifest = open.manifest.get_or_insert_with(|| Manifest {
            id: next_id,
            prompt: open.prompt.clone(),
            started_at: open.started_at,
            files: Vec::new(),
            dirs: Vec::new(),
            not_saved: Vec::new(),
            git,
            finished: false,
        });
        let dir = self
            .dir
            .as_ref()
            .expect("checkpoints are kept")
            .join(manifest.id.to_string());
        fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
        for (path, index) in index {
            let meta = fs::symlink_metadata(&path).ok();
            let mut entry = FileEntry {
                path: path.clone(),
                blob: None,
                link: None,
                mode: meta.as_ref().map(file_mode),
                index,
                after: None,
            };
            match meta {
                Some(meta) if meta.file_type().is_symlink() => {
                    entry.link = fs::read_link(&path).ok();
                }
                Some(meta) => {
                    if open.bytes + meta.len() > MAX_TURN_BYTES {
                        manifest.not_saved.push(path);
                        continue;
                    }
                    let blob = manifest.files.len().to_string();
                    fs::copy(&path, dir.join(&blob))
                        .with_context(|| format!("saving {}", path.display()))?;
                    open.bytes += meta.len();
                    entry.blob = Some(blob);
                }
                None => {}
            }
            manifest.files.push(entry);
        }
        for path in dirs {
            let existed = path.is_dir();
            manifest.dirs.push(DirEntry { path, existed });
        }
        write_manifest(&dir, manifest)?;
        if manifest.files.len() + manifest.dirs.len() > 0 {
            self.prune();
        }
        Ok(())
    }

    /// Drops the oldest turns past [`MAX_TURNS`].
    fn prune(&self) {
        let manifests = self.manifests();
        if manifests.len() <= MAX_TURNS {
            return;
        }
        for manifest in &manifests[..manifests.len() - MAX_TURNS] {
            if let Some(dir) = self.turn_dir(manifest.id) {
                let _ = fs::remove_dir_all(dir);
            }
        }
    }

    fn finish(&mut self) {
        let Some(open) = self.open.take() else {
            return;
        };
        let Some(mut manifest) = open.manifest else {
            return;
        };
        for entry in &mut manifest.files {
            entry.after = Some(fingerprint(&entry.path));
        }
        manifest.finished = true;
        if let Some(dir) = self.turn_dir(manifest.id) {
            let _ = write_manifest(&dir, &manifest);
        }
    }

    fn restore_from(&mut self, id: u64, force: bool) -> Result<Restored, RestoreError> {
        let manifests = self.manifests();
        if manifests.is_empty() {
            return Err(RestoreError::Nothing);
        }
        if !manifests.iter().any(|manifest| manifest.id == id) {
            return Err(RestoreError::NoSuchTurn(id));
        }
        // Newest first: each turn puts back what the one before it left.
        let turns: Vec<Manifest> = manifests
            .into_iter()
            .rev()
            .take_while(|manifest| manifest.id >= id)
            .collect();

        // A path must hold what the newest turn touching it left.
        let mut checked = HashSet::new();
        let mut changed = Vec::new();
        for manifest in &turns {
            for entry in &manifest.files {
                if !checked.insert(entry.path.clone()) {
                    continue;
                }
                if let Some(after) = &entry.after
                    && fingerprint(&entry.path) != *after
                {
                    changed.push(self.relative(&entry.path));
                }
            }
        }
        if !changed.is_empty() && !force {
            return Err(RestoreError::Changed(changed));
        }

        let mut restored = Restored {
            overwritten: changed,
            ..Restored::default()
        };
        for manifest in &turns {
            let dir = self
                .turn_dir(manifest.id)
                .ok_or_else(|| RestoreError::Failed("checkpoints are off".into()))?;
            self.restore_turn(&dir, manifest, &mut restored)
                .map_err(|err| RestoreError::Failed(format!("{err:#}")))?;
            restored.turns.push(self.summary(manifest));
            restored
                .not_saved
                .extend(manifest.not_saved.iter().map(|path| self.relative(path)));
            let _ = fs::remove_dir_all(&dir);
        }
        restored.written.sort();
        restored.written.dedup();
        restored.removed.sort();
        restored.removed.dedup();
        restored
            .removed
            .retain(|path| restored.written.binary_search(path).is_err());
        self.notice = Some(notice(&restored));
        Ok(restored)
    }

    fn restore_turn(
        &mut self,
        dir: &Path,
        manifest: &Manifest,
        restored: &mut Restored,
    ) -> Result<()> {
        // Directories that held files first, so the files can go back in.
        for entry in manifest.dirs.iter().filter(|entry| entry.existed) {
            fs::create_dir_all(&entry.path)
                .with_context(|| format!("creating {}", entry.path.display()))?;
        }
        for entry in &manifest.files {
            let path = &entry.path;
            let current = fs::symlink_metadata(path).ok();
            if current.as_ref().is_some_and(|meta| meta.is_dir()) {
                if entry.blob.is_none() && entry.link.is_none() {
                    continue;
                }
                fs::remove_dir_all(path).with_context(|| format!("removing {}", path.display()))?;
            }
            if let Some(blob) = &entry.blob {
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent)?;
                }
                if current
                    .as_ref()
                    .is_some_and(|meta| meta.file_type().is_symlink())
                {
                    fs::remove_file(path)?;
                }
                fs::copy(dir.join(blob), path)
                    .with_context(|| format!("restoring {}", path.display()))?;
                set_mode(path, entry.mode);
                restored.written.push(self.relative(path));
            } else if let Some(target) = &entry.link {
                if current.is_some() {
                    fs::remove_file(path)?;
                }
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent)?;
                }
                make_link(target, path)?;
                restored.written.push(self.relative(path));
            } else if current.is_some() {
                fs::remove_file(path).with_context(|| format!("removing {}", path.display()))?;
                restored.removed.push(self.relative(path));
            }
        }
        // Directories the turn created go if nothing else is in them now.
        let mut created: Vec<&DirEntry> = manifest
            .dirs
            .iter()
            .filter(|entry| !entry.existed)
            .collect();
        created.sort_by_key(|entry| std::cmp::Reverse(entry.path.components().count()));
        for entry in created {
            let _ = fs::remove_dir(&entry.path);
        }
        if manifest.git {
            self.restore_index(manifest)?;
        }
        Ok(())
    }

    /// Puts the index entries of `manifest`'s files back as they were.
    fn restore_index(&mut self, manifest: &Manifest) -> Result<()> {
        let Some(root) = self.git_root() else {
            return Ok(());
        };
        let paths: Vec<PathBuf> = manifest
            .files
            .iter()
            .map(|entry| entry.path.clone())
            .collect();
        let now = self.index_entries(&paths);
        let mut set = String::new();
        let mut remove = Vec::new();
        for (entry, (_, current)) in manifest.files.iter().zip(now) {
            if entry.index == current {
                continue;
            }
            let Some(rel) = self.repo_relative(&entry.path) else {
                continue;
            };
            match &entry.index {
                Some(index) => set.push_str(&format!("{index}\t{rel}\n")),
                None => remove.push(rel),
            }
        }
        if !set.is_empty() {
            let mut child = Command::new("git")
                .arg("-C")
                .arg(&root)
                .args(["update-index", "--index-info"])
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .context("running git update-index")?;
            child
                .stdin
                .take()
                .ok_or_else(|| anyhow!("git update-index has no stdin"))?
                .write_all(set.as_bytes())?;
            child.wait()?;
        }
        if !remove.is_empty() {
            Command::new("git")
                .arg("-C")
                .arg(&root)
                .args(["update-index", "--force-remove", "--"])
                .args(&remove)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .context("running git update-index")?;
        }
        Ok(())
    }
}

/// What the model is told about a restore, before the user's next prompt.
fn notice(restored: &Restored) -> String {
    let turns: Vec<String> = restored
        .turns
        .iter()
        .map(|turn| format!("\"{}\"", turn.prompt))
        .collect();
    let mut text = format!(
        "[The user undid the file changes you made for {}.",
        turns.join(", ")
    );
    if !restored.written.is_empty() {
        text.push_str(&format!(
            " These files are back as they were before: {}.",
            restored.written.join(", ")
        ));
    }
    if !restored.removed.is_empty() {
        text.push_str(&format!(
            " These files you created are gone: {}.",
            restored.removed.join(", ")
        ));
    }
    text.push(']');
    text
}

fn write_manifest(dir: &Path, manifest: &Manifest) -> Result<()> {
    let body = serde_json::to_vec_pretty(manifest)?;
    let temp = dir.join(format!("{MANIFEST}.tmp"));
    fs::write(&temp, body)?;
    fs::rename(&temp, dir.join(MANIFEST))?;
    Ok(())
}

/// Every file and directory under `root`, without following links.
fn walk(root: &Path) -> Vec<(PathBuf, bool)> {
    walkdir::WalkDir::new(root)
        .follow_links(false)
        .min_depth(1)
        .into_iter()
        .flatten()
        .map(|entry| (entry.path().to_path_buf(), entry.file_type().is_dir()))
        .collect()
}

/// What `path` holds now.
pub fn fingerprint(path: &Path) -> Fingerprint {
    match fs::symlink_metadata(path) {
        Err(_) => Fingerprint::Absent,
        Ok(meta) if meta.file_type().is_symlink() => {
            Fingerprint::Link(fs::read_link(path).unwrap_or_default())
        }
        Ok(meta) if meta.is_dir() => Fingerprint::Directory,
        Ok(_) => match fs::read(path) {
            Ok(bytes) => Fingerprint::Content(
                Sha256::digest(&bytes)
                    .iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect(),
            ),
            Err(_) => Fingerprint::Absent,
        },
    }
}

fn first_line(prompt: &str) -> String {
    let line = prompt
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("");
    let line = line.trim();
    match line.char_indices().nth(72) {
        Some((at, _)) => format!("{}…", &line[..at]),
        None => line.to_string(),
    }
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or_default()
}

#[cfg(unix)]
fn file_mode(meta: &fs::Metadata) -> u32 {
    use std::os::unix::fs::PermissionsExt;
    meta.permissions().mode() & 0o7777
}

#[cfg(not(unix))]
fn file_mode(meta: &fs::Metadata) -> u32 {
    u32::from(meta.permissions().readonly())
}

#[cfg(unix)]
fn set_mode(path: &Path, mode: Option<u32>) {
    use std::os::unix::fs::PermissionsExt;
    if let Some(mode) = mode {
        let _ = fs::set_permissions(path, fs::Permissions::from_mode(mode));
    }
}

#[cfg(not(unix))]
fn set_mode(path: &Path, mode: Option<u32>) {
    if let (Some(mode), Ok(meta)) = (mode, fs::metadata(path)) {
        let mut permissions = meta.permissions();
        permissions.set_readonly(mode == 1);
        let _ = fs::set_permissions(path, permissions);
    }
}

#[cfg(unix)]
fn make_link(target: &Path, path: &Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(target, path)
}

#[cfg(windows)]
fn make_link(target: &Path, path: &Path) -> std::io::Result<()> {
    std::os::windows::fs::symlink_file(target, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixture {
        workspace: tempfile::TempDir,
        _session: tempfile::TempDir,
        checkpoints: Checkpoints,
    }

    fn fixture() -> Fixture {
        let workspace = tempfile::tempdir().unwrap();
        let session = tempfile::tempdir().unwrap();
        let checkpoints =
            Checkpoints::new(workspace.path(), Some(session.path().join("checkpoints")));
        Fixture {
            workspace,
            _session: session,
            checkpoints,
        }
    }

    impl Fixture {
        fn path(&self, rel: &str) -> PathBuf {
            self.workspace.path().join(rel)
        }

        fn write(&self, rel: &str, body: &str) {
            let path = self.path(rel);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, body).unwrap();
        }

        fn read(&self, rel: &str) -> Option<String> {
            fs::read_to_string(self.path(rel)).ok()
        }

        fn git(&self, args: &[&str]) -> String {
            let output = Command::new("git")
                .arg("-C")
                .arg(self.workspace.path())
                .args(args)
                .output()
                .unwrap();
            String::from_utf8(output.stdout).unwrap()
        }
    }

    #[test]
    fn undo_puts_back_edits_and_removes_created_files() {
        let f = fixture();
        f.write("a.txt", "one\n");
        f.checkpoints.begin_turn("change things\nmore detail");
        f.checkpoints.record(&[f.path("a.txt")]);
        f.write("a.txt", "two\n");
        f.checkpoints.record(&[f.path("new/dir/b.txt")]);
        f.write("new/dir/b.txt", "b\n");
        // A second write to the same file keeps the first copy.
        f.checkpoints.record(&[f.path("a.txt")]);
        f.write("a.txt", "three\n");
        f.checkpoints.finish_turn();

        let restored = f.checkpoints.undo(false).unwrap();
        assert_eq!(f.read("a.txt").as_deref(), Some("one\n"));
        assert!(!f.path("new").exists());
        assert_eq!(restored.written, vec!["a.txt"]);
        assert_eq!(restored.removed, vec!["new/dir/b.txt"]);
        assert_eq!(restored.turns[0].prompt, "change things");
        assert!(f.checkpoints.take_notice().unwrap().contains("a.txt"));
        assert_eq!(f.checkpoints.undo(false), Err(RestoreError::Nothing));
    }

    #[test]
    fn a_turn_without_changes_saves_nothing() {
        let f = fixture();
        f.checkpoints.begin_turn("just a question");
        f.checkpoints.finish_turn();
        assert!(f.checkpoints.list().is_empty());
    }

    #[test]
    fn rewind_undoes_every_turn_from_the_one_named() {
        let f = fixture();
        f.write("a.txt", "v0");
        for (turn, body) in ["v1", "v2", "v3"].iter().enumerate() {
            f.checkpoints.begin_turn(&format!("turn {turn}"));
            f.checkpoints.record(&[f.path("a.txt")]);
            f.write("a.txt", body);
        }
        f.checkpoints.finish_turn();
        let turns = f.checkpoints.list();
        assert_eq!(
            turns.iter().map(|t| t.id).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert_eq!(turns[1].files, vec!["a.txt"]);

        let restored = f.checkpoints.rewind(2, false).unwrap();
        assert_eq!(f.read("a.txt").as_deref(), Some("v1"));
        assert_eq!(
            restored.turns.iter().map(|t| t.id).collect::<Vec<_>>(),
            vec![3, 2]
        );
        assert_eq!(f.checkpoints.list().len(), 1);
        assert_eq!(
            f.checkpoints.rewind(7, false),
            Err(RestoreError::NoSuchTurn(7))
        );
    }

    #[test]
    fn a_file_changed_since_the_turn_needs_force() {
        let f = fixture();
        f.write("a.txt", "mine");
        f.checkpoints.begin_turn("edit");
        f.checkpoints.record(&[f.path("a.txt")]);
        f.write("a.txt", "agent");
        f.checkpoints.finish_turn();
        f.write("a.txt", "mine again, edited by hand");

        assert_eq!(
            f.checkpoints.undo(false),
            Err(RestoreError::Changed(vec!["a.txt".into()]))
        );
        assert_eq!(
            f.read("a.txt").as_deref(),
            Some("mine again, edited by hand")
        );
        let restored = f.checkpoints.undo(true).unwrap();
        assert_eq!(restored.overwritten, vec!["a.txt"]);
        assert_eq!(f.read("a.txt").as_deref(), Some("mine"));
    }

    #[test]
    fn deleted_and_moved_directories_come_back() {
        let f = fixture();
        f.write("src/one.rs", "1");
        f.write("src/deep/two.rs", "2");
        f.write("docs/readme.md", "r");
        f.checkpoints.begin_turn("reorganise");
        f.checkpoints.record_move(&f.path("src"), &f.path("lib"));
        fs::rename(f.path("src"), f.path("lib")).unwrap();
        f.checkpoints.record(&[f.path("docs")]);
        fs::remove_dir_all(f.path("docs")).unwrap();
        f.checkpoints.finish_turn();

        f.checkpoints.undo(false).unwrap();
        assert_eq!(f.read("src/one.rs").as_deref(), Some("1"));
        assert_eq!(f.read("src/deep/two.rs").as_deref(), Some("2"));
        assert_eq!(f.read("docs/readme.md").as_deref(), Some("r"));
        assert!(!f.path("lib").exists());
    }

    #[test]
    fn checkpoints_survive_a_restart() {
        let f = fixture();
        f.write("a.txt", "before");
        f.checkpoints.begin_turn("edit");
        f.checkpoints.record(&[f.path("a.txt")]);
        f.write("a.txt", "after");
        f.checkpoints.finish_turn();

        let dir = f.checkpoints.inner.lock().unwrap().dir.clone();
        let reopened = Checkpoints::new(f.workspace.path(), dir);
        assert_eq!(reopened.list().len(), 1);
        reopened.undo(false).unwrap();
        assert_eq!(f.read("a.txt").as_deref(), Some("before"));
    }

    #[test]
    fn without_a_session_directory_nothing_is_kept() {
        let workspace = tempfile::tempdir().unwrap();
        let checkpoints = Checkpoints::new(workspace.path(), None);
        checkpoints.begin_turn("edit");
        checkpoints.record(&[workspace.path().join("a.txt")]);
        assert!(!checkpoints.enabled());
        assert_eq!(checkpoints.undo(false), Err(RestoreError::Nothing));
    }

    #[test]
    fn the_index_goes_back_as_it_was() {
        let f = fixture();
        f.git(&["init", "-q"]);
        f.write("tracked.txt", "t0");
        f.git(&["add", "tracked.txt"]);
        let staged = f.git(&["ls-files", "-s", "tracked.txt"]);

        f.checkpoints.begin_turn("edit and add");
        f.checkpoints
            .record(&[f.path("tracked.txt"), f.path("added.txt")]);
        f.write("tracked.txt", "t1");
        f.write("added.txt", "new");
        f.git(&["add", "tracked.txt", "added.txt"]);
        f.checkpoints.finish_turn();

        f.checkpoints.undo(false).unwrap();
        assert_eq!(f.git(&["ls-files", "-s", "tracked.txt"]), staged);
        assert_eq!(f.git(&["ls-files", "added.txt"]), "");
        assert!(!f.path("added.txt").exists());
        assert_eq!(f.read("tracked.txt").as_deref(), Some("t0"));
    }

    #[test]
    fn first_lines_are_kept_short() {
        assert_eq!(first_line("\n  fix the bug  \nthen more"), "fix the bug");
        assert_eq!(first_line(&"x".repeat(100)).chars().count(), 73);
    }
}
