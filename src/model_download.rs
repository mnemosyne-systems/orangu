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

//! `orangu-server download <user>/<model>[:quant]`: downloads a GGUF model
//! from the Hugging Face Hub into the configured `models` directory, laid
//! out exactly the way llama.cpp's own `-hf`/`--hf-repo` downloads into —
//! `models--<user>--<model>/{blobs,refs,snapshots}` — so `list`/`show`
//! already read what this writes, and llama.cpp itself recognizes it as
//! already downloaded rather than fetching it again.
//!
//! Mirrors llama.cpp's own `common/download.cpp`/`common/hf-cache.cpp`
//! (verified directly against that source, not guessed): the same two Hub
//! API calls (`/api/models/<repo>/refs` for the commit, `/api/models/<repo>/tree/<commit>?recursive=true`
//! for the file listing), the same file-selection rules (excluding
//! `mmproj`/`imatrix`/`mtp-` files from being treated as "the model", the
//! same `["Q4_K_M", "Q8_0"]` default tag preference when no `:quant` is
//! given, the same shard-sibling collection for a multi-part model), the
//! same best-matching-`mmproj`-sibling selection (`find_best_sibling`:
//! prefer the deepest directory shared with the model, then the closest
//! quantization bit-depth) — llama-server's own `-hf` already auto-fetches
//! this file the first time a vision-capable model is launched with an
//! image-related flag, so fetching it up front means `LLAMA_CACHE=<models>`
//! already has everything ready offline — and the same on-disk layout
//! (content-addressed blobs, a relative symlink per snapshot file). Not
//! mirrored: `--mtp` companion downloads, `preset.ini` repos, and Docker
//! registry sources — all out of scope for a first version of a "download
//! the model" command.
//!
//! A multi-part model's shards (and a bundled `mmproj`, when present)
//! download concurrently rather than one at a time — bounded by rayon's
//! global thread pool — each reporting its own progress line on a shared
//! [`ProgressBoard`]. Every file gets a line from the first draw onwards,
//! whether or not a thread has picked it up yet, so all of them stay visible
//! at once until the last one is done.

use anyhow::{Context, Result, anyhow, bail};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, OpenOptions},
    io::{IsTerminal, Read, Write},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

const HUB_ENDPOINT: &str = "https://huggingface.co";
/// The same fallback preference order llama.cpp's own `find_best_model`
/// uses when a `download` target names a repo but no `:quant` — asked for
/// in that order, first match wins.
const DEFAULT_TAG_PREFERENCE: &[&str] = &["Q4_K_M", "Q8_0"];

/// One file's line of a [`DownloadSnapshot`] — the structured form of the
/// text [`Slot::line`] renders for a terminal.
#[derive(Clone, Serialize)]
pub struct DownloadFile {
    /// The repo-relative path of the file, as [`Slot::label`].
    pub label: String,
    /// Its real size from the repository listing.
    pub size: u64,
    /// How many of those bytes are on disk right now.
    pub downloaded: u64,
    /// `queued`, `downloading`, `retrying`, or `done`.
    pub state: &'static str,
    /// Which retry attempt this file is waiting out, when `state` is
    /// `retrying`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry: Option<u32>,
}

/// Where a download is right now, for a caller with no terminal to draw a
/// [`ProgressBoard`] on — the same numbers the board's own `Total` line is
/// built from, as data rather than text.
#[derive(Clone, Default, Serialize)]
pub struct DownloadSnapshot {
    pub files: Vec<DownloadFile>,
    pub total_bytes: u64,
    pub done_bytes: u64,
    pub percent: u64,
    /// Seconds remaining at the rate this run has actually transferred at,
    /// or `None` before there is enough of one to extrapolate from — the
    /// same rule [`ProgressBoard::eta`] uses for its own text.
    pub eta_secs: Option<u64>,
}

/// A live, shared view of an in-progress [`download_model_reporting`] run.
/// Passing one both redirects the [`ProgressBoard`] away from stdout
/// entirely (a server has no terminal to draw an in-place-updating block
/// on, and would only spam its log) and makes the same state readable at
/// any moment from another thread — which is what the web console's model
/// manager polls.
#[derive(Default)]
pub struct DownloadProgress {
    snapshot: Mutex<DownloadSnapshot>,
}

impl DownloadProgress {
    pub fn snapshot(&self) -> DownloadSnapshot {
        self.snapshot.lock().unwrap().clone()
    }

    fn publish(&self, snapshot: DownloadSnapshot) {
        *self.snapshot.lock().unwrap() = snapshot;
    }
}

/// Downloads `spec` (`<user>/<model>[:quant]`) from the Hugging Face Hub
/// into `models_dir`, and returns the local path of the primary model file
/// (the first shard, for a multi-part model) once every selected file
/// (every shard, plus a bundled `mmproj` sidecar if the repo has one) is in
/// place. Used both by `orangu-server download` (which only cares that the
/// files land on disk) and model-spec resolution ahead of serving (which
/// also needs the resulting path to load).
pub fn download_model(models_dir: &Path, spec: &str) -> Result<PathBuf> {
    download_model_reporting(models_dir, spec, None)
}

/// [`download_model`], reporting into `progress` as it goes instead of onto
/// a terminal. Blocking exactly as `download_model` is — a caller inside an
/// async runtime has to run it on a blocking thread, and reads `progress`
/// from wherever it likes meanwhile.
pub fn download_model_reporting(
    models_dir: &Path,
    spec: &str,
    progress: Option<Arc<DownloadProgress>>,
) -> Result<PathBuf> {
    let (repo, tag) = split_repo_tag(spec)?;
    let client = build_client(None)?;
    let token = std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty());

    let commit = resolve_commit(&client, &repo, token.as_deref())?;
    let files = list_repo_files(&client, &repo, &commit, token.as_deref())?;
    let mut selected = select_files_to_download(&files, tag.as_deref())
        .with_context(|| format!("no matching GGUF file in {repo}"))?;
    let primary_path = selected[0].path.clone();

    if let Some(mmproj) = find_best_mmproj(&files, &selected[0].path) {
        selected.push(mmproj);
    }

    let repo_dir = models_dir.join(repo_folder_name(&repo));
    let blobs_dir = repo_dir.join("blobs");
    let snapshot_dir = repo_dir.join("snapshots").join(&commit);
    fs::create_dir_all(&blobs_dir)
        .with_context(|| format!("failed to create {}", blobs_dir.display()))?;
    fs::create_dir_all(&snapshot_dir)
        .with_context(|| format!("failed to create {}", snapshot_dir.display()))?;
    let refs_dir = repo_dir.join("refs");
    fs::create_dir_all(&refs_dir)
        .with_context(|| format!("failed to create {}", refs_dir.display()))?;
    fs::write(refs_dir.join("main"), &commit)
        .with_context(|| format!("failed to write {}", refs_dir.join("main").display()))?;

    // Every selected file gets a board line, in selection order, whether or
    // not this run has to fetch it — the skipped ones included, so the
    // `[index/total]` counter runs over one visible block instead of naming
    // a file that scrolled past above it. Every file's size, and how much of
    // it is already on disk, is known here — before a single byte is
    // fetched — so the `Total` line starts out telling the truth instead of
    // discovering it as threads free up.
    let mut slots = Vec::with_capacity(selected.len());
    let mut tasks = Vec::new();
    for (index, file) in selected.iter().enumerate() {
        let blob_path = blobs_dir.join(&file.oid);
        let already_downloaded =
            blob_path.is_file() && fs::metadata(&blob_path)?.len() == file.size;
        // What an interrupted earlier run left behind, which this one resumes
        // from rather than re-fetches. Measured against the file's real size
        // from the repository listing, never against anything on disk: a
        // `.part` at or past that size is a stale leftover that
        // `download_attempt` throws away and re-fetches from zero, so it's
        // worth nothing here either.
        let resumed = match fs::metadata(part_path(&blob_path)).map(|m| m.len()) {
            Ok(len) if len < file.size => len,
            _ => 0,
        };

        slots.push(Slot {
            label: file.path.clone(),
            size: file.size,
            downloaded: if already_downloaded {
                file.size
            } else {
                resumed
            },
            state: if already_downloaded {
                SlotState::Skipped
            } else {
                SlotState::Queued
            },
        });
        if !already_downloaded {
            tasks.push(DownloadTask {
                label: file.path.clone(),
                url: format!(
                    "{HUB_ENDPOINT}/{repo}/resolve/{commit}/{}",
                    urlencode_path(&file.path)
                ),
                blob_path,
                size: file.size,
                slot: index,
            });
        }
    }

    // What this run still has to write, before it writes any of it: every
    // file's real size less whatever of it is already on disk. Checked
    // against the free space up front rather than discovered part-way
    // through a multi-hour download that then dies with ENOSPC — and with
    // dozens of shards streaming at once, the one that fails first is rarely
    // the one that filled the disk.
    let needed: u64 = slots.iter().map(|s| s.size - s.downloaded).sum();
    check_space(&blobs_dir, needed, crate::os::available_space(&blobs_dir))?;

    let board = Mutex::new(ProgressBoard::new(slots, progress));
    // Drawn even when there's nothing left to fetch: a repo that's already
    // fully downloaded still says so, file by file.
    board.lock().unwrap().draw();
    if !tasks.is_empty() {
        download_all(&client, &tasks, token.as_deref(), &board)?;
    }

    for file in &selected {
        let blob_path = blobs_dir.join(&file.oid);
        let snapshot_path = snapshot_dir.join(&file.path);
        if let Some(parent) = snapshot_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        if !snapshot_path.exists() {
            link_or_copy(&blob_path, &snapshot_path, &file.oid, &file.path)?;
        }
    }

    Ok(snapshot_dir.join(primary_path))
}

fn build_client(timeout: Option<std::time::Duration>) -> Result<reqwest::blocking::Client> {
    let mut builder = reqwest::blocking::Client::builder()
        .user_agent(concat!("orangu-server/", env!("CARGO_PKG_VERSION")));
    if let Some(timeout) = timeout {
        builder = builder.timeout(timeout);
    }
    builder.build().context("failed to build HTTP client")
}

/// Resolves `main`'s live commit for each of `repos` (every distinct
/// Hugging Face repo id `list` found under the models directory — deduped
/// by repo, not by `(repo, commit)`, so a repo with several `:quant` rows
/// cached at different commits is still only queried once), in parallel,
/// returning a `repo -> commit` map. `orangu-server list` compares each
/// row's own `local_commit` against this map when rendering, so only the
/// rows actually behind get marked `(Refresh)` — a repo with one stale and
/// one current row doesn't mark both just because they share a repo id.
///
/// Every failure — no network, DNS, a rate limit, a repo that's since gone
/// private — is swallowed per-repo rather than propagated: unlike
/// `download`, `list` must still print its table when offline, just without
/// any refresh markers, so one flaky lookup (or no connectivity at all)
/// can't fail the whole command. Each request is capped with a short
/// timeout for the same reason — an unreachable Hub shouldn't make `list`
/// hang.
pub fn latest_commits(repos: &[String]) -> std::collections::HashMap<String, String> {
    if repos.is_empty() {
        return std::collections::HashMap::new();
    }
    let client = match build_client(Some(std::time::Duration::from_secs(5))) {
        Ok(client) => client,
        Err(_) => return std::collections::HashMap::new(),
    };
    let token = std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty());

    repos
        .par_iter()
        .filter_map(|repo| {
            let commit = resolve_commit(&client, repo, token.as_deref()).ok()?;
            Some((repo.clone(), commit))
        })
        .collect()
}

/// Splits a `download` argument into `(repo, tag)`, e.g.
/// `"unsloth/gemma-4-26B-A4B-it-qat-GGUF:UD-Q4_K_XL"` ->
/// `("unsloth/gemma-4-26B-A4B-it-qat-GGUF", Some("UD-Q4_K_XL"))`. `repo`
/// must have exactly one `/`, the same `<user>/<model>` shape llama.cpp's
/// own `-hf` flag requires.
fn split_repo_tag(spec: &str) -> Result<(String, Option<String>)> {
    let (repo, tag) = match spec.split_once(':') {
        Some((repo, tag)) => (repo.to_string(), Some(tag.to_string())),
        None => (spec.to_string(), None),
    };
    if repo.matches('/').count() != 1 {
        bail!("'{spec}' is not a valid <user>/<model>[:quant] reference");
    }
    Ok((repo, tag))
}

/// `models--<user>--<model>`, the Hugging Face hub cache's own directory
/// naming convention (`repo_id.replace("/", "--")`, prefixed) — the same
/// one `models::hf_repo_id_from_path` reverses when reading a cache back.
fn repo_folder_name(repo: &str) -> String {
    format!("models--{}", repo.replace('/', "--"))
}

#[derive(Deserialize)]
struct RefsResponse {
    branches: Vec<Branch>,
}

#[derive(Deserialize)]
struct Branch {
    name: String,
    #[serde(rename = "targetCommit")]
    target_commit: String,
}

/// Resolves `repo`'s `main` branch to a commit sha via
/// `GET /api/models/<repo>/refs`, falling back to the first branch listed
/// if there's no `main` (mirrors `hf_cache::get_repo_commit`).
fn resolve_commit(
    client: &reqwest::blocking::Client,
    repo: &str,
    token: Option<&str>,
) -> Result<String> {
    let url = format!("{HUB_ENDPOINT}/api/models/{repo}/refs");
    let response = authed_get(client, &url, token)
        .send()
        .with_context(|| format!("failed to reach Hugging Face for {repo}"))?;
    // A repo that doesn't exist at all can 401 rather than 404 when
    // unauthenticated — Hugging Face returns the same status for "doesn't
    // exist" as for "exists but is private", to avoid leaking which. Only
    // read that way without a token already in hand; with one, a 401 means
    // the token itself was rejected, not that the repo is missing.
    match (response.status(), token) {
        (reqwest::StatusCode::NOT_FOUND, _) => bail!("repository not found: {repo}"),
        (reqwest::StatusCode::UNAUTHORIZED, None) => {
            bail!("repository not found: {repo} (if it's private or gated, set HF_TOKEN)")
        }
        (reqwest::StatusCode::UNAUTHORIZED, Some(_)) => {
            bail!("authentication failed for {repo} — check HF_TOKEN")
        }
        _ => {}
    }
    let response = response
        .error_for_status()
        .with_context(|| format!("failed to list refs for {repo}"))?;
    let refs: RefsResponse = response
        .json()
        .with_context(|| format!("unexpected response listing refs for {repo}"))?;

    refs.branches
        .iter()
        .find(|b| b.name == "main")
        .or_else(|| refs.branches.first())
        .map(|b| b.target_commit.clone())
        .ok_or_else(|| anyhow!("{repo} has no branches to download from"))
}

#[derive(Deserialize)]
struct TreeEntry {
    #[serde(rename = "type")]
    kind: String,
    path: String,
    oid: Option<String>,
    size: Option<u64>,
    lfs: Option<LfsInfo>,
}

#[derive(Deserialize)]
struct LfsInfo {
    oid: String,
    size: u64,
}

#[derive(Debug)]
pub struct RepoFile {
    pub path: String,
    /// The content hash this file is stored under — the LFS oid (sha256)
    /// for large files, the plain git blob oid (sha1) otherwise. Doubles as
    /// the blob's filename in the cache, exactly like the real Hugging Face
    /// hub cache.
    pub oid: String,
    pub size: u64,
}

/// Lists every file in `repo`@`commit` via `GET /api/models/<repo>/tree/<commit>?recursive=true`.
fn list_repo_files(
    client: &reqwest::blocking::Client,
    repo: &str,
    commit: &str,
    token: Option<&str>,
) -> Result<Vec<RepoFile>> {
    let url = format!("{HUB_ENDPOINT}/api/models/{repo}/tree/{commit}?recursive=true");
    let response = authed_get(client, &url, token)
        .send()
        .with_context(|| format!("failed to list files in {repo}"))?
        .error_for_status()
        .with_context(|| format!("failed to list files in {repo}"))?;
    let entries: Vec<TreeEntry> = response
        .json()
        .with_context(|| format!("unexpected response listing files in {repo}"))?;

    Ok(entries
        .into_iter()
        .filter(|entry| entry.kind == "file")
        .filter_map(|entry| {
            let (oid, size) = match entry.lfs {
                Some(lfs) => (lfs.oid, lfs.size),
                None => (entry.oid?, entry.size.unwrap_or(0)),
            };
            Some(RepoFile {
                path: entry.path,
                oid,
                size,
            })
        })
        .collect())
}

fn authed_get(
    client: &reqwest::blocking::Client,
    url: &str,
    token: Option<&str>,
) -> reqwest::blocking::RequestBuilder {
    let request = client.get(url).header("Accept", "application/json");
    match token {
        Some(token) => request.bearer_auth(token),
        None => request,
    }
}

/// Whether `path` names a standalone model file rather than a companion
/// sidecar — excludes multimodal projectors, imatrix calibration data, and
/// multi-token-prediction draft heads, exactly like llama.cpp's own
/// `gguf_filename_is_model`.
fn is_model_gguf(path: &str) -> bool {
    if !path.to_lowercase().ends_with(".gguf") {
        return false;
    }
    let filename = path.rsplit('/').next().unwrap_or(path).to_lowercase();
    !filename.contains("mmproj") && !filename.contains("imatrix") && !filename.starts_with("mtp-")
}

/// Parses a GGUF shard suffix (`-NNNNN-of-NNNNN.gguf`), returning
/// `(prefix, index, total)` — e.g. `"model-00002-of-00004.gguf"` ->
/// `("model", 2, 4)`. `None` for an unsharded file, which callers treat as
/// shard 1 of 1.
fn shard_info(path: &str) -> Option<(String, u32, u32)> {
    let stem = path.strip_suffix(".gguf")?;
    static SHARD_SUFFIX: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let pattern =
        SHARD_SUFFIX.get_or_init(|| regex::Regex::new(r"^(.+)-(\d{5})-of-(\d{5})$").unwrap());
    let captures = pattern.captures(stem)?;
    Some((
        captures[1].to_string(),
        captures[2].parse().ok()?,
        captures[3].parse().ok()?,
    ))
}

/// Picks which file(s) to download: the primary model file matching `tag`
/// (or, when `tag` is `None`, the first of [`DEFAULT_TAG_PREFERENCE`] that
/// exists, falling further back to the first model file found at all), plus
/// every other shard belonging to that same multi-part model. Mirrors
/// llama.cpp's `find_best_model` + `get_split_files`.
fn select_files_to_download<'a>(
    files: &'a [RepoFile],
    tag: Option<&str>,
) -> Result<Vec<&'a RepoFile>> {
    let model_files: Vec<&RepoFile> = files.iter().filter(|f| is_model_gguf(&f.path)).collect();
    if model_files.is_empty() {
        bail!("no GGUF model files found in this repository");
    }

    let primary = match tag {
        Some(tag) => find_by_tag(&model_files, tag).ok_or_else(|| {
            anyhow!(
                "no file matching quant '{tag}'; available: {}",
                available_tags(&model_files)
            )
        })?,
        None => DEFAULT_TAG_PREFERENCE
            .iter()
            .find_map(|tag| find_by_tag(&model_files, tag))
            .or_else(|| first_primary_shard(&model_files))
            .ok_or_else(|| anyhow!("no downloadable model file found"))?,
    };

    let Some((prefix, _, total)) = shard_info(&primary.path) else {
        return Ok(vec![primary]);
    };
    let mut shards: Vec<(&RepoFile, u32)> = files
        .iter()
        .filter_map(|f| match shard_info(&f.path) {
            Some((p, index, t)) if p == prefix && t == total => Some((f, index)),
            _ => None,
        })
        .collect();
    shards.sort_by_key(|(_, index)| *index);
    Ok(shards.into_iter().map(|(f, _)| f).collect())
}

/// A file matches `tag` when the tag text appears in its path immediately
/// followed by `.` or `-` (so `"Q4_K_M"` matches `"model-Q4_K_M.gguf"` and
/// `"model-Q4_K_M-00001-of-00004.gguf"`, the same substring rule llama.cpp
/// uses), and it's shard 1 (or unsharded) — never a later shard on its own.
fn find_by_tag<'a>(model_files: &[&'a RepoFile], tag: &str) -> Option<&'a RepoFile> {
    let tag_lower = tag.to_lowercase();
    model_files
        .iter()
        .find(|f| {
            let path_lower = f.path.to_lowercase();
            path_lower.match_indices(&tag_lower).any(|(index, _)| {
                matches!(
                    path_lower.as_bytes().get(index + tag_lower.len()),
                    Some(b'.') | Some(b'-')
                )
            }) && matches!(shard_info(&f.path), None | Some((_, 1, _)))
        })
        .copied()
}

fn first_primary_shard<'a>(model_files: &[&'a RepoFile]) -> Option<&'a RepoFile> {
    model_files
        .iter()
        .find(|f| matches!(shard_info(&f.path), None | Some((_, 1, _))))
        .copied()
}

/// The trailing quant tag of a (possibly sharded) path, e.g.
/// `"model-Q4_K_M-00001-of-00003.gguf"` -> `Some("Q4_K_M")` — mirrors
/// llama.cpp's `get_gguf_split_info`'s own `tag` field (shard suffix
/// stripped first, then the last `-`/`.`-delimited segment).
fn trailing_tag(path: &str) -> Option<String> {
    let prefix = match shard_info(path) {
        Some((prefix, _, _)) => prefix,
        None => path.strip_suffix(".gguf").unwrap_or(path).to_string(),
    };
    static TAG_SUFFIX: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let pattern = TAG_SUFFIX.get_or_init(|| regex::Regex::new(r"[-.]([A-Za-z0-9_]+)$").unwrap());
    pattern.captures(&prefix).map(|c| c[1].to_uppercase())
}

/// The quantization's bit depth extracted from its tag, e.g. `"Q4_K_M"` ->
/// `4`, `"BF16"`/`"F16"` -> `16`, `"F32"` -> `32` — mirrors llama.cpp's
/// `extract_quant_bits` (first run of digits in the tag).
fn extract_quant_bits(path: &str) -> i64 {
    let Some(tag) = trailing_tag(path) else {
        return 0;
    };
    let digits: String = tag
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().unwrap_or(0)
}

/// Picks the best sibling GGUF whose path contains `keyword` (e.g.
/// `"mmproj"`) — preferring the deepest directory shared with `model_path`,
/// then the closest quantization bit-depth. Mirrors llama.cpp's own
/// `find_best_sibling`/`find_best_mmproj` exactly, so this selects the same
/// file llama-server's own `-hf` would auto-fetch anyway when it needs one.
fn find_best_sibling<'a>(
    files: &'a [RepoFile],
    model_path: &str,
    keyword: &str,
) -> Option<&'a RepoFile> {
    let model_parts: Vec<&str> = model_path.split('/').collect();
    let model_dir = &model_parts[..model_parts.len().saturating_sub(1)];
    let model_bits = extract_quant_bits(model_path);

    let mut best: Option<&RepoFile> = None;
    let mut best_depth = 0usize;
    let mut best_diff = i64::MAX;

    for f in files {
        let path_lower = f.path.to_lowercase();
        if !path_lower.ends_with(".gguf") || !path_lower.contains(keyword) {
            continue;
        }
        let sib_parts: Vec<&str> = f.path.split('/').collect();
        let sib_dir = &sib_parts[..sib_parts.len().saturating_sub(1)];

        let depth = model_dir
            .iter()
            .zip(sib_dir.iter())
            .take_while(|(a, b)| a == b)
            .count();
        if depth != sib_dir.len() {
            // sib_dir isn't a prefix of model_dir — not a valid sibling.
            continue;
        }

        let diff = (extract_quant_bits(&f.path) - model_bits).abs();
        if best.is_none() || depth > best_depth || (depth == best_depth && diff < best_diff) {
            best = Some(f);
            best_depth = depth;
            best_diff = diff;
        }
    }
    best
}

fn find_best_mmproj<'a>(files: &'a [RepoFile], model_path: &str) -> Option<&'a RepoFile> {
    find_best_sibling(files, model_path, "mmproj")
}

/// Lists the quant tags found among `model_files`'s own filenames (via the
/// same trailing-tag convention `models::hf_tag_from_label` extracts),
/// shown in an error when a requested `:quant` doesn't exist.
fn available_tags(model_files: &[&RepoFile]) -> String {
    let mut tags: Vec<String> = model_files
        .iter()
        .filter_map(|f| {
            let stem = f.path.rsplit('/').next()?.strip_suffix(".gguf")?;
            let stem = shard_info(&f.path)
                .map(|(prefix, _, _)| prefix)
                .unwrap_or_else(|| stem.to_string());
            let separator = stem.rfind(['-', '.'])?;
            Some(stem[separator + 1..].to_uppercase())
        })
        .collect();
    tags.sort();
    tags.dedup();
    if tags.is_empty() {
        "(none found)".to_string()
    } else {
        tags.join(", ")
    }
}

/// Percent-encodes a repo-relative path for use in a URL, leaving `/`
/// itself unescaped (each segment is encoded, the separators are not).
fn urlencode_path(path: &str) -> String {
    path.split('/')
        .map(percent_encode)
        .collect::<Vec<_>>()
        .join("/")
}

fn percent_encode(segment: &str) -> String {
    let mut out = String::with_capacity(segment.len());
    for byte in segment.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char);
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

/// One file still needing a fetch, everything [`download_with_resume`]
/// needs to run independently of the others on its own thread: `label` is
/// shown in progress text (the repo-relative path), `slot` is which line of
/// the [`ProgressBoard`] this file reports on — its 0-based position among
/// *every* selected file, not among the ones being fetched, so a file
/// skipped as already downloaded still owns its own line.
struct DownloadTask {
    label: String,
    url: String,
    blob_path: PathBuf,
    size: u64,
    slot: usize,
}

/// Where one file is in its own download, which is what its line on the
/// [`ProgressBoard`] says. A file that hasn't started yet is [`Queued`], not
/// blank: rayon runs only as many downloads at once as it has threads, so a
/// 34-shard model otherwise showed a couple of dozen empty lines between the
/// handful actually in flight.
///
/// [`Queued`]: SlotState::Queued
#[derive(Clone, Copy)]
enum SlotState {
    /// Already on disk at the right size, so this run never fetches it. Reads
    /// exactly like [`Done`] — from the outside a file that's on disk is
    /// downloaded, however it got there — and is a state of its own only so
    /// non-interactive output can log these up front, since they never pass
    /// through [`ProgressBoard::update`].
    ///
    /// [`Done`]: SlotState::Done
    Skipped,
    Queued,
    /// Being fetched. How far along is [`Slot::downloaded`], not part of the
    /// state — a failed attempt waiting to be retried is still this state,
    /// with `retry` set, since it's the same file with the same bytes on
    /// disk, and the retry resumes from them rather than starting over.
    Downloading {
        retry: Option<Retry>,
    },
    Done,
}

/// An attempt that failed and is waiting out [`DOWNLOAD_RETRY_DELAY`] before
/// the next one.
#[derive(Clone, Copy)]
struct Retry {
    attempt: u32,
    secs: u64,
}

/// One file's line on the board: everything needed to render it, so the
/// board never has to parse text it printed earlier to know what a file is
/// doing. Every selected file has one, including the skipped ones — a
/// `[35/35]` that scrolls past above the board is a file the user then has
/// to go looking for.
struct Slot {
    label: String,
    /// This file's real size, from the repository listing (an LFS object's
    /// own size for the files that matter here) — never anything measured on
    /// disk.
    size: u64,
    /// How many of those bytes are on disk right now: whatever an interrupted
    /// earlier run left in the `.part` to begin with, then the running count
    /// as this run streams it, and `size` once it's complete. Bytes, not a
    /// percentage, because `Total` subtracts them from the real total — a
    /// per-file percentage would round away up to a gigabyte a shard.
    downloaded: u64,
    state: SlotState,
}

impl Slot {
    fn line(&self, index: usize, total: usize) -> String {
        let Slot { label, state, .. } = self;
        let percent = self.percent();
        match state {
            // A file waiting on a thread with nothing on disk yet is just
            // queued; one that a previous run got partway through says how
            // far, since that's already progress against the total.
            SlotState::Queued if self.downloaded == 0 => {
                format!("Queued {label} [{index}/{total}]")
            }
            SlotState::Queued => format!("Queued {label}: {percent}% [{index}/{total}]"),
            SlotState::Downloading { retry: None } => {
                format!("Downloading {label}: {percent}% [{index}/{total}]")
            }
            SlotState::Downloading {
                retry: Some(Retry { attempt, secs }),
            } => format!(
                "Downloading {label}: {percent}% (retry {attempt}/{DOWNLOAD_RETRIES} in {secs}s) [{index}/{total}]"
            ),
            SlotState::Done | SlotState::Skipped => {
                format!("Downloaded {label}: 100% [{index}/{total}]")
            }
        }
    }

    /// This file's own progress, for its own line — its bytes on disk against
    /// its real size.
    fn percent(&self) -> u64 {
        (self.downloaded * 100)
            .checked_div(self.size)
            .unwrap_or(0)
            .min(100)
    }
}

/// Tracks one in-place-updating terminal line per file being downloaded, so
/// several can report progress at once without their redraws clobbering each
/// other. A single [`Mutex`] around the whole board (rather than one per
/// line) means each update is an atomic "set this file's state, then redraw
/// every line" — no interleaving of two threads' writes.
struct ProgressBoard {
    /// One per selected file, in the order they were selected — the shards
    /// in shard order, then a bundled `mmproj` last — so a line's position
    /// in the block is also its `[index/total]`.
    slots: Vec<Slot>,
    /// How many lines the previous draw actually wrote, and so how far up the
    /// cursor has to move to overwrite them. Zero before the first draw,
    /// which must not move up at all — there's nothing above it yet.
    drawn: usize,
    /// Whether progress is being drawn to a terminal at all. Piped or
    /// redirected output gets one plain line per *completed* file instead:
    /// cursor movement means nothing in a file, and a redraw per percent per
    /// shard would bury the log in thousands of lines.
    interactive: bool,
    /// When this run started, for the `Total` line's ETA.
    started: std::time::Instant,
    /// Bytes this run has actually pulled off the network, counted as they're
    /// read. The ETA divides by *this*, not by how much of the model is now
    /// complete: a file already on disk, and the bytes a resumed `.part`
    /// contributes the moment its first percentage lands, are progress this
    /// run never spent any time on. Crediting them to its first seconds
    /// projected a download speed nothing had reached — a 1.4 TiB model
    /// resuming from a third of a terabyte read as minutes remaining.
    ///
    /// An atomic rather than a plain field: it's incremented per 64 KiB chunk
    /// by every download thread at once, which would otherwise mean taking
    /// the board's mutex thousands of times a second.
    transferred: Arc<AtomicU64>,
    /// Set when the caller wants the state as data rather than as output
    /// (see [`DownloadProgress`]). Its presence turns *all* printing off,
    /// interactive and logged alike: the one caller that passes it is a
    /// running server, where the in-place redraws would be escape-sequence
    /// noise and the per-file log lines would go somewhere nobody is
    /// watching.
    sink: Option<Arc<DownloadProgress>>,
}

impl ProgressBoard {
    fn new(slots: Vec<Slot>, sink: Option<Arc<DownloadProgress>>) -> Self {
        Self {
            slots,
            drawn: 0,
            interactive: sink.is_none() && std::io::stdout().is_terminal(),
            started: std::time::Instant::now(),
            transferred: Arc::new(AtomicU64::new(0)),
            sink,
        }
    }

    /// Hands the sink, if there is one, everything its holder could want to
    /// know right now. Called from every state change (via [`Self::update`])
    /// and from the one up-front [`Self::draw`], so a poll never sees a
    /// board older than the last transition.
    fn publish(&self) {
        let Some(sink) = &self.sink else { return };
        let elapsed = self.started.elapsed();
        let total_bytes: u64 = self.slots.iter().map(|s| s.size).sum();
        let done_bytes: u64 = self.slots.iter().map(|s| s.downloaded).sum();
        sink.publish(DownloadSnapshot {
            files: self
                .slots
                .iter()
                .map(|slot| DownloadFile {
                    label: slot.label.clone(),
                    size: slot.size,
                    downloaded: slot.downloaded,
                    state: match slot.state {
                        SlotState::Skipped | SlotState::Done => "done",
                        SlotState::Queued => "queued",
                        SlotState::Downloading { retry: None } => "downloading",
                        SlotState::Downloading { retry: Some(_) } => "retrying",
                    },
                    retry: match slot.state {
                        SlotState::Downloading {
                            retry: Some(Retry { attempt, .. }),
                        } => Some(attempt),
                        _ => None,
                    },
                })
                .collect(),
            total_bytes,
            done_bytes,
            percent: (done_bytes * 100).checked_div(total_bytes).unwrap_or(0),
            eta_secs: self
                .eta(done_bytes, total_bytes, elapsed)
                .map(|eta| eta.as_secs()),
        });
    }

    /// `slot` now has `downloaded` of its bytes on disk and is streaming —
    /// which also clears any retry marker the line was carrying, since bytes
    /// are moving again.
    fn progress(&mut self, slot: usize, downloaded: u64) {
        self.slots[slot].downloaded = downloaded;
        self.update(slot, SlotState::Downloading { retry: None });
    }

    /// `slot`'s file is complete: all of its real size is on disk.
    fn finished(&mut self, slot: usize) {
        self.slots[slot].downloaded = self.slots[slot].size;
        self.update(slot, SlotState::Done);
    }

    /// Moves `slot` to `state` and redraws every line, so the whole board
    /// is one flush behind whatever just changed.
    fn update(&mut self, slot: usize, state: SlotState) {
        self.slots[slot].state = state;
        self.publish();
        if self.sink.is_some() {
            return;
        }
        if !self.interactive {
            // A file finishing, and a download stalling into a retry, are the
            // only two transitions worth a line in a log — the percentages in
            // between would be thousands of them.
            let worth_logging = matches!(
                state,
                SlotState::Done | SlotState::Downloading { retry: Some(_), .. }
            );
            if worth_logging {
                println!("{}", self.slots[slot].line(slot + 1, self.slots.len()));
            }
            return;
        }
        self.draw();
    }

    /// Marks `slot`'s attempt as failed and waiting `secs` before the next
    /// one. Its bytes on disk are untouched — the retry resumes from them
    /// rather than restarting the file — so the line keeps its percentage.
    fn set_retry(&mut self, slot: usize, attempt: u32, secs: u64) {
        self.update(
            slot,
            SlotState::Downloading {
                retry: Some(Retry { attempt, secs }),
            },
        );
    }

    /// Writes the whole board and flushes it. Called once before any file
    /// starts — so every file queued for this run is on screen from the
    /// outset rather than appearing only as a thread frees up — and again on
    /// every state change after that.
    fn draw(&mut self) {
        self.publish();
        if self.sink.is_some() {
            return;
        }
        if !self.interactive {
            // Only ever reached by the single up-front call — `update`
            // returns before drawing when output isn't a terminal. Files
            // already on disk never reach `update` at all, so this is where
            // they're logged.
            for (i, slot) in self.slots.iter().enumerate() {
                if matches!(slot.state, SlotState::Skipped) {
                    println!("{}", slot.line(i + 1, self.slots.len()));
                }
            }
            return;
        }
        let frame = self.frame(
            terminal_rows().map_or(usize::MAX, |rows| rows.saturating_sub(1)),
            self.started.elapsed(),
        );
        let mut out = std::io::stdout();
        if self.drawn > 0 {
            // Move the cursor back up to the first line so every line below
            // gets overwritten rather than appended below the last draw.
            write!(out, "\x1b[{}A", self.drawn).ok();
        }
        for line in &frame {
            // \x1b[2K clears the line first — a shorter new line (e.g. once
            // a percentage's digit count shrinks, which can't happen here,
            // but also just a differently-sized final "Downloaded" message)
            // otherwise leaves stray trailing characters from the old one.
            writeln!(out, "\r\x1b[2K{line}").ok();
        }
        if frame.len() < self.drawn {
            // The frame just got shorter — the terminal was resized down to
            // where only the aggregate line fits. Erase everything below the
            // cursor, or the tail of the taller previous draw stays on screen
            // forever, frozen at whatever it last said.
            write!(out, "\x1b[J").ok();
        }
        self.drawn = frame.len();
        out.flush().ok();
    }

    /// The block of lines to draw, given how many rows there is `room` to
    /// draw into: one line per file, every file, so nothing being downloaded
    /// is invisible, closed by a `Total` line for the run as a whole.
    ///
    /// The one case that can't have them all is a terminal with fewer rows
    /// than there are files — redrawing in place needs the whole block to
    /// stay on screen, since a block that scrolls off the top can't be moved
    /// back up to, and the redraws would smear down the screen instead. There
    /// the per-file lines drop away and the `Total` line stands alone: it's
    /// the one line that still accounts for every file.
    fn frame(&self, room: usize, elapsed: std::time::Duration) -> Vec<String> {
        let total = self.slots.len();
        let mut lines: Vec<String> = self
            .slots
            .iter()
            .enumerate()
            .map(|(i, slot)| slot.line(i + 1, total))
            .collect();
        lines.push(self.total_line(elapsed));
        if lines.len() <= room {
            return lines;
        }
        vec![self.total_line(elapsed)]
    }

    /// The run as a whole on one line: bytes on disk against the real total
    /// (so a 5 GiB shard half fetched counts for more than a finished
    /// 200 MiB one), how the files themselves are spread across
    /// done/active/queued, and an ETA once there's enough of a download rate
    /// to project one from.
    fn total_line(&self, elapsed: std::time::Duration) -> String {
        let total_bytes: u64 = self.slots.iter().map(|s| s.size).sum();
        let done_bytes: u64 = self.slots.iter().map(|s| s.downloaded).sum();
        let percent = (done_bytes * 100).checked_div(total_bytes).unwrap_or(0);
        let done = self
            .slots
            .iter()
            .filter(|s| matches!(s.state, SlotState::Done | SlotState::Skipped))
            .count();
        let queued = self
            .slots
            .iter()
            .filter(|s| matches!(s.state, SlotState::Queued))
            .count();
        let active = self.slots.len() - done - queued;
        let eta = match self.eta(done_bytes, total_bytes, elapsed) {
            Some(eta) => format!(", ETA {}", format_eta(eta)),
            None => String::new(),
        };
        format!(
            "Total {percent}%: {done}/{} files ({} of {}), {active} active, {queued} queued{eta}",
            self.slots.len(),
            crate::format::format_bytes(done_bytes),
            crate::format::format_bytes(total_bytes),
        )
    }

    /// How long the bytes still outstanding — every file's, queued ones
    /// included — should take at the rate this run has actually transferred
    /// at. `None` until there's something to extrapolate from (the first
    /// seconds of a run project wildly) and once there's nothing left.
    fn eta(
        &self,
        done_bytes: u64,
        total_bytes: u64,
        elapsed: std::time::Duration,
    ) -> Option<std::time::Duration> {
        const MIN_ELAPSED: std::time::Duration = std::time::Duration::from_secs(5);

        let remaining = total_bytes.checked_sub(done_bytes).filter(|r| *r > 0)?;
        let transferred = self.transferred.load(Ordering::Relaxed);
        if elapsed < MIN_ELAPSED || transferred == 0 {
            return None;
        }
        let per_second = transferred as f64 / elapsed.as_secs_f64();
        Some(std::time::Duration::from_secs_f64(
            remaining as f64 / per_second,
        ))
    }
}

/// An ETA as `2h:05m`, or just `23m` under an hour — a download measured
/// in seconds isn't worth a number, so anything under a minute is `<1m`.
fn format_eta(eta: std::time::Duration) -> String {
    let minutes = eta.as_secs() / 60;
    match (minutes / 60, minutes % 60) {
        (0, 0) => "<1m".to_string(),
        (0, minutes) => format!("{minutes}m"),
        (hours, minutes) => format!("{hours}h:{minutes:02}m"),
    }
}

/// Refuses a download that wouldn't fit: `needed` bytes to write into `dir`,
/// with `available` bytes free there (`None` where the platform can't say,
/// which passes — an unanswerable question is not a reason to refuse work
/// that may well fit).
///
/// The margin is deliberately none: this catches the download that can't
/// possibly fit, not the one that fits with little to spare. Anything else
/// writing to the same filesystem meanwhile is beyond what a check before
/// the fact can promise.
fn check_space(dir: &Path, needed: u64, available: Option<u64>) -> Result<()> {
    let Some(available) = available else {
        return Ok(());
    };
    if needed <= available {
        return Ok(());
    }
    bail!(
        "not enough free space in {}: {} needed, {} free (short by {})",
        dir.display(),
        crate::format::format_bytes(needed),
        crate::format::format_bytes(available),
        crate::format::format_bytes(needed - available),
    )
}

/// Where a blob's partial download lives until it's complete. Blob filenames
/// are bare content hashes with no extension of their own for
/// `Path::with_extension` to replace, so just append directly.
fn part_path(blob: &Path) -> PathBuf {
    PathBuf::from(format!("{}.part", blob.display()))
}

/// The terminal's height in rows, or `None` when output isn't a terminal at
/// all (nothing to fit the board into).
fn terminal_rows() -> Option<usize> {
    terminal_size::terminal_size().map(|(_, terminal_size::Height(rows))| usize::from(rows))
}

/// Downloads every task concurrently — bounded by rayon's global thread
/// pool rather than one thread per file, so a model with dozens of shards
/// doesn't open dozens of simultaneous connections — each reporting into its
/// own line of the shared `board` (which already has a line for every
/// selected file, this run's tasks and the files it had nothing to do
/// among them) so every file's progress stays visible at once until all are
/// done. Returns the first error encountered, if any; other in-flight
/// downloads still run to completion (each writes its own `.part` file, so a
/// later retry only re-fetches whatever actually failed).
fn download_all(
    client: &reqwest::blocking::Client,
    tasks: &[DownloadTask],
    token: Option<&str>,
    board: &Mutex<ProgressBoard>,
) -> Result<()> {
    tasks
        .par_iter()
        .try_for_each(|task| download_with_resume(client, task, token, task.slot, board))
}

/// How many times a single file's download is retried after a transient
/// failure (a dropped connection, a rate limit, a truncated stream) before
/// giving up, and how long to wait between attempts. A large multi-shard
/// model streams for long enough that at least one shard hitting a blip is
/// the norm, not the exception; each retry resumes from the `.part` file via
/// a `Range` request rather than restarting, so a blip near the end costs
/// only the tail, not the whole file.
const DOWNLOAD_RETRIES: u32 = 5;
const DOWNLOAD_RETRY_DELAY: std::time::Duration = std::time::Duration::from_secs(30);

/// Downloads `task.url` into `task.blob_path`, resuming from a `.part` file
/// left over from an interrupted attempt (via an HTTP `Range` request), and
/// reporting percentage progress against `task.size` into `board`'s `slot`
/// line as it goes.
///
/// A failed attempt is retried quietly up to [`DOWNLOAD_RETRIES`] times with
/// [`DOWNLOAD_RETRY_DELAY`] between tries — each retry picks up where the
/// last left off, since the partial bytes are already on disk. The retry
/// status shows in place on this file's own progress line rather than
/// scrolling new output, so a stall stays visible without becoming noise.
/// Only once every retry is exhausted does the error propagate and fail the
/// whole command.
fn download_with_resume(
    client: &reqwest::blocking::Client,
    task: &DownloadTask,
    token: Option<&str>,
    slot: usize,
    board: &Mutex<ProgressBoard>,
) -> Result<()> {
    let DownloadTask {
        blob_path: dest, ..
    } = task;
    let part_path = part_path(dest);

    let mut attempt = 0;
    loop {
        match download_attempt(client, task, token, slot, board, &part_path) {
            Ok(()) => break,
            Err(_) if attempt < DOWNLOAD_RETRIES => {
                attempt += 1;
                let secs = DOWNLOAD_RETRY_DELAY.as_secs();
                board.lock().unwrap().set_retry(slot, attempt, secs);
                std::thread::sleep(DOWNLOAD_RETRY_DELAY);
            }
            Err(err) => return Err(err),
        }
    }

    board.lock().unwrap().finished(slot);

    fs::rename(&part_path, dest)
        .with_context(|| format!("failed to finalize {}", dest.display()))?;
    Ok(())
}

/// One attempt at fetching `task` into `part_path` — sends the request
/// (resuming from any existing partial bytes), streams the body to disk, and
/// verifies the finished `.part` is the expected size. Returns an error on
/// any network, HTTP, I/O, or short-read failure so [`download_with_resume`]
/// can retry it; the partial bytes it leaves behind are what the next
/// attempt resumes from.
fn download_attempt(
    client: &reqwest::blocking::Client,
    task: &DownloadTask,
    token: Option<&str>,
    slot: usize,
    board: &Mutex<ProgressBoard>,
    part_path: &Path,
) -> Result<()> {
    let DownloadTask {
        label,
        url,
        size: expected_size,
        ..
    } = task;
    let expected_size = *expected_size;
    let mut resume_from = fs::metadata(part_path).map(|m| m.len()).unwrap_or(0);
    if resume_from >= expected_size && expected_size > 0 {
        // A stale/complete .part from an interrupted run that never got
        // renamed; nothing left to fetch.
        resume_from = 0;
        fs::remove_file(part_path).ok();
    }

    let mut request = authed_get(client, url, token);
    if resume_from > 0 {
        request = request.header("Range", format!("bytes={resume_from}-"));
    }
    let mut response = request
        .send()
        .with_context(|| format!("failed to download {label}"))?;

    if resume_from > 0 && response.status() != reqwest::StatusCode::PARTIAL_CONTENT {
        // Server ignored the Range request; restart from scratch.
        resume_from = 0;
        fs::remove_file(part_path).ok();
        response = authed_get(client, url, token)
            .send()
            .with_context(|| format!("failed to download {label}"))?;
    }
    let response = response
        .error_for_status()
        .with_context(|| format!("failed to download {label}"))?;

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(part_path)
        .with_context(|| format!("failed to open {}", part_path.display()))?;

    let mut downloaded = resume_from;
    let mut buf = [0u8; 65536];
    let mut last_printed = None;
    let mut body = response;
    // Taken once: the read loop adds to it per chunk without going anywhere
    // near the board's mutex, which every other download thread is also
    // contending for.
    let transferred = board.lock().unwrap().transferred.clone();
    loop {
        let read = body
            .read(&mut buf)
            .with_context(|| format!("failed to download {label}"))?;
        if read == 0 {
            break;
        }
        file.write_all(&buf[..read])
            .with_context(|| format!("failed to write {}", part_path.display()))?;
        downloaded += read as u64;
        // Only what this attempt actually pulled down — bytes a previous run
        // left in the `.part` (`resume_from`) are not this run's throughput.
        transferred.fetch_add(read as u64, Ordering::Relaxed);

        // Reported on each whole percent rather than each chunk: the board
        // redraws every line on every update, and 64 KiB at a time would be
        // tens of thousands of redraws a shard. The byte count is what's
        // handed over, though — `Total` subtracts it from the real total.
        if let Some(percent) = (downloaded * 100)
            .checked_div(expected_size)
            .map(|p| p.min(100))
            && last_printed != Some(percent)
        {
            board.lock().unwrap().progress(slot, downloaded);
            last_printed = Some(percent);
        }
    }

    // A connection that drops mid-stream often ends with a clean `read == 0`
    // rather than an error, leaving a truncated `.part`. Treat a short file
    // as a failure so it's retried (and resumed) rather than finalized as if
    // it were complete.
    if expected_size > 0 && downloaded != expected_size {
        bail!(
            "download of {label} ended early ({downloaded} of {expected_size} bytes) — will retry"
        );
    }
    Ok(())
}

/// Points `link` (a `snapshots/<commit>/<file>` path) at `blob` (a
/// `blobs/<oid>` path) with a relative symlink — exactly how the real
/// Hugging Face hub cache does it, so the file is portable if the whole
/// `models` directory is moved. Falls back to a plain copy if symlinks
/// aren't available (e.g. Windows without developer mode enabled).
fn link_or_copy(blob: &Path, link: &Path, oid: &str, file_path: &str) -> Result<()> {
    // From `snapshots/<commit>/<file_path>`, `..` once per path component of
    // `file_path` (including the filename) reaches `snapshots/<commit>/`,
    // then two more reach the repo root, then descend into `blobs/<oid>`.
    let ups = "../".repeat(file_path.matches('/').count() + 2);
    let target = format!("{ups}blobs/{oid}");
    // Windows only resolves symlink targets with backslash separators: a
    // forward-slash target creates a link that Windows itself cannot follow
    // (every native read fails with "the filename, directory name, or volume
    // label syntax is incorrect"), so the model would be invisible to `list`
    // and unreadable by llama-server, even though POSIX-emulating shells
    // resolve it fine.
    #[cfg(windows)]
    let target = target.replace('/', "\\");

    #[cfg(unix)]
    let symlink_result = std::os::unix::fs::symlink(&target, link);
    #[cfg(windows)]
    let symlink_result = std::os::windows::fs::symlink_file(&target, link);
    #[cfg(not(any(unix, windows)))]
    let symlink_result: std::io::Result<()> = Err(std::io::Error::other("symlinks unsupported"));

    if symlink_result.is_ok() {
        return Ok(());
    }
    fs::copy(blob, link)
        .map(|_| ())
        .with_context(|| format!("failed to place {}", link.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1024 * 1024;

    /// A board of equally-sized 1 MiB files, each given as the state it's in
    /// and how many of its bytes are on disk — built the way the real one is,
    /// with every file's real size and starting position known at
    /// construction.
    fn board_with(files: Vec<(SlotState, u64)>) -> ProgressBoard {
        let count = files.len();
        ProgressBoard::new(
            files
                .into_iter()
                .enumerate()
                .map(|(i, (state, downloaded))| Slot {
                    label: format!("model-{:05}-of-{count:05}.gguf", i + 1),
                    size: MIB,
                    downloaded,
                    state,
                })
                .collect(),
            None,
        )
    }

    fn board_of(count: usize) -> ProgressBoard {
        board_with(vec![(SlotState::Queued, 0); count])
    }

    /// Long enough for the ETA to be worth projecting; short enough that the
    /// arithmetic in these tests stays obvious.
    const MINUTE: std::time::Duration = std::time::Duration::from_secs(60);

    /// Every file has a line from the very first draw, before any of them
    /// has started — rayon runs only as many downloads at once as it has
    /// threads, and the waiting ones used to render as blank lines.
    #[test]
    fn every_file_has_a_line_before_anything_starts() {
        let board = board_of(34);

        let frame = board.frame(usize::MAX, std::time::Duration::ZERO);

        assert_eq!(frame.len(), 35, "34 files plus the Total line");
        assert!(
            frame[..34].iter().all(|line| line.starts_with("Queued ")),
            "{frame:?}"
        );
        assert_eq!(frame[17], "Queued model-00018-of-00034.gguf [18/34]");
    }

    /// A file's own line carries its own state; the others stay put. A file
    /// waiting on a retry stays a `Downloading` line, at the percentage it
    /// had reached, with the retry as a note on that same line. The `Total`
    /// line closes the block whatever they say.
    #[test]
    fn each_line_follows_its_own_files_state() {
        let board = board_with(vec![
            (SlotState::Skipped, MIB),
            (SlotState::Downloading { retry: None }, MIB / 2),
            (
                SlotState::Downloading {
                    retry: Some(Retry {
                        attempt: 2,
                        secs: 30,
                    }),
                },
                MIB / 4,
            ),
        ]);
        board.transferred.store(MIB, Ordering::Relaxed);

        assert_eq!(
            board.frame(usize::MAX, MINUTE),
            vec![
                // A file already on disk is downloaded, however it got there.
                "Downloaded model-00001-of-00003.gguf: 100% [1/3]",
                "Downloading model-00002-of-00003.gguf: 50% [2/3]",
                "Downloading model-00003-of-00003.gguf: 25% (retry 2/5 in 30s) [3/3]",
                "Total 58%: 1/3 files (1.75 MiB of 3.00 MiB), 2 active, 0 queued, ETA 1m",
            ]
        );
    }

    /// A retry keeps the bytes the file had already reached — it resumes from
    /// them — and the marker clears again once bytes move.
    #[test]
    fn a_retry_annotates_the_line_it_is_already_on() {
        let mut board = board_with(vec![(SlotState::Downloading { retry: None }, MIB / 2)]);

        board.set_retry(0, 1, 30);
        assert_eq!(
            board.slots[0].line(1, 1),
            "Downloading model-00001-of-00001.gguf: 50% (retry 1/5 in 30s) [1/1]"
        );
        assert_eq!(board.slots[0].downloaded, MIB / 2, "no bytes were dropped");

        board.progress(0, MIB * 3 / 4);
        assert_eq!(
            board.slots[0].line(1, 1),
            "Downloading model-00001-of-00001.gguf: 75% [1/1]"
        );
    }

    /// The ETA covers every byte still outstanding — the queued files
    /// included, not just whatever is in flight.
    #[test]
    fn the_eta_accounts_for_every_file_still_outstanding() {
        // 1 MiB transferred in a minute, 2 MiB still to go.
        let two_queued = board_with(vec![
            (SlotState::Skipped, MIB),
            (SlotState::Done, MIB),
            (SlotState::Queued, 0),
            (SlotState::Queued, 0),
        ]);
        two_queued.transferred.store(MIB, Ordering::Relaxed);
        assert_eq!(
            two_queued.frame(usize::MAX, MINUTE).pop().unwrap(),
            "Total 50%: 2/4 files (2.00 MiB of 4.00 MiB), 0 active, 2 queued, ETA 2m"
        );

        // The same run with one fewer file still to fetch, at the same rate:
        // an ETA that only covered what was in flight wouldn't move at all.
        let one_queued = board_with(vec![
            (SlotState::Skipped, MIB),
            (SlotState::Done, MIB),
            (SlotState::Queued, 0),
        ]);
        one_queued.transferred.store(MIB, Ordering::Relaxed);
        assert_eq!(
            one_queued.frame(usize::MAX, MINUTE).pop().unwrap(),
            "Total 66%: 2/3 files (2.00 MiB of 3.00 MiB), 0 active, 1 queued, ETA 1m"
        );
    }

    /// Bytes that were already on disk are progress, but they are not
    /// throughput: dividing by them projected minutes left on a download with
    /// a terabyte to go. Only what this run pulled off the network counts.
    #[test]
    fn bytes_already_on_disk_are_not_this_runs_throughput() {
        let board = board_with(vec![
            (SlotState::Downloading { retry: None }, MIB * 3 / 4);
            4
        ]);
        // 3 MiB of 4 MiB is on disk, but only 64 KiB of it came down the wire
        // this minute — so the 1 MiB left is a quarter of an hour away, not
        // the seconds that 3 MiB/min would have projected.
        board.transferred.store(64 * 1024, Ordering::Relaxed);

        assert_eq!(
            board.frame(usize::MAX, MINUTE).pop().unwrap(),
            "Total 75%: 0/4 files (3.00 MiB of 4.00 MiB), 4 active, 0 queued, ETA 16m"
        );
    }

    /// A file an earlier run got partway through is counted from the `.part`
    /// on disk, before any thread picks it up — the `Total` line starts out
    /// telling the truth instead of climbing as threads free up.
    #[test]
    fn a_resumed_file_counts_before_a_thread_picks_it_up() {
        let board = board_with(vec![(SlotState::Queued, MIB / 4), (SlotState::Queued, 0)]);

        let frame = board.frame(usize::MAX, MINUTE);

        assert_eq!(frame[0], "Queued model-00001-of-00002.gguf: 25% [1/2]");
        assert_eq!(frame[1], "Queued model-00002-of-00002.gguf [2/2]");
        assert_eq!(
            frame[2],
            "Total 12%: 0/2 files (256.00 KiB of 2.00 MiB), 0 active, 2 queued"
        );
    }

    /// No ETA is shown until this run has actually transferred something to
    /// project from, and none once there's nothing left to fetch.
    #[test]
    fn an_eta_needs_both_a_rate_and_something_left_to_fetch() {
        let mut board = board_of(2);
        board.progress(0, MIB / 2);

        let nothing_transferred = board.frame(usize::MAX, MINUTE);
        assert!(
            !nothing_transferred.last().unwrap().contains("ETA"),
            "{nothing_transferred:?}"
        );

        board.transferred.store(MIB / 2, Ordering::Relaxed);
        let too_early = board.frame(usize::MAX, std::time::Duration::from_secs(1));
        assert!(!too_early.last().unwrap().contains("ETA"), "{too_early:?}");

        board.finished(0);
        board.finished(1);
        let finished = board.frame(usize::MAX, MINUTE);
        assert!(!finished.last().unwrap().contains("ETA"), "{finished:?}");
    }

    /// A download that can't fit is refused before it writes anything —
    /// filling the disk halfway through a multi-hour fetch of a model this
    /// size is a slow, confusing way to find out.
    #[test]
    fn a_download_that_cannot_fit_is_refused_up_front() {
        let dir = std::path::Path::new("/models");

        assert!(
            check_space(dir, 10 * MIB, Some(10 * MIB)).is_ok(),
            "exact fit"
        );
        assert!(check_space(dir, 10 * MIB, Some(11 * MIB)).is_ok());
        // Nothing left to fetch always fits, even on a full disk.
        assert!(check_space(dir, 0, Some(0)).is_ok());
        // A platform that can't answer is not a reason to refuse.
        assert!(check_space(dir, u64::MAX, None).is_ok());

        let err = check_space(dir, 11 * MIB, Some(10 * MIB)).expect_err("does not fit");
        assert_eq!(
            err.to_string(),
            "not enough free space in /models: 11.00 MiB needed, 10.00 MiB free (short by 1.00 MiB)"
        );
    }

    /// What's checked is what's left to write, not the model's full size: a
    /// resumed run only needs room for the bytes it hasn't fetched yet.
    #[test]
    fn the_space_needed_is_what_is_left_to_write() {
        let board = board_with(vec![
            (SlotState::Skipped, MIB),
            (SlotState::Queued, MIB / 4),
            (SlotState::Queued, 0),
        ]);

        let needed: u64 = board.slots.iter().map(|s| s.size - s.downloaded).sum();

        assert_eq!(
            needed,
            MIB + MIB * 3 / 4,
            "3 MiB of model, 1.25 MiB on disk"
        );
    }

    #[test]
    fn an_eta_is_hours_and_minutes_only_past_the_hour() {
        use std::time::Duration;

        assert_eq!(format_eta(Duration::from_secs(30)), "<1m");
        assert_eq!(format_eta(Duration::from_secs(23 * 60)), "23m");
        assert_eq!(format_eta(Duration::from_secs(59 * 60 + 59)), "59m");
        assert_eq!(format_eta(Duration::from_secs(60 * 60)), "1h:00m");
        assert_eq!(format_eta(Duration::from_secs(2 * 3600 + 5 * 60)), "2h:05m");
        assert_eq!(format_eta(Duration::from_secs(30 * 3600)), "30h:00m");
    }

    /// A board taller than the terminal can't be redrawn in place — it would
    /// scroll off the top and smear — so it collapses to the `Total` line,
    /// which still accounts for every file.
    #[test]
    fn a_board_taller_than_the_terminal_collapses_to_the_total_line() {
        let mut board = board_of(34);
        board.finished(0);
        board.progress(1, MIB / 2);
        board.transferred.store(MIB + MIB / 2, Ordering::Relaxed);

        let frame = board.frame(23, MINUTE);

        assert_eq!(
            frame,
            vec!["Total 4%: 1/34 files (1.50 MiB of 34.00 MiB), 1 active, 32 queued, ETA 21m"]
        );
        assert_eq!(
            board.frame(35, MINUTE).len(),
            35,
            "35 rows of room fits 34 files plus the Total line"
        );
        assert_eq!(
            board.frame(34, MINUTE).len(),
            1,
            "one row short of the whole block is not enough"
        );
    }

    #[test]
    fn split_repo_tag_separates_an_optional_quant() {
        assert_eq!(
            split_repo_tag("unsloth/gemma-4-26B-A4B-it-qat-GGUF:UD-Q4_K_XL").unwrap(),
            (
                "unsloth/gemma-4-26B-A4B-it-qat-GGUF".to_string(),
                Some("UD-Q4_K_XL".to_string())
            )
        );
        assert_eq!(
            split_repo_tag("Qwen/Qwen3.6-35B-A3B").unwrap(),
            ("Qwen/Qwen3.6-35B-A3B".to_string(), None)
        );
    }

    #[test]
    fn split_repo_tag_rejects_anything_without_exactly_one_slash() {
        assert!(split_repo_tag("no-slash-at-all").is_err());
        assert!(split_repo_tag("too/many/slashes").is_err());
    }

    #[test]
    fn repo_folder_name_matches_the_hub_cache_convention() {
        assert_eq!(
            repo_folder_name("ggml-org/embeddinggemma-300M-GGUF"),
            "models--ggml-org--embeddinggemma-300M-GGUF"
        );
    }

    #[test]
    fn is_model_gguf_excludes_sidecar_files() {
        assert!(is_model_gguf("model-Q4_K_M.gguf"));
        assert!(is_model_gguf("sub/model-Q4_K_M.gguf"));
        assert!(!is_model_gguf("mmproj-model-bf16.gguf"));
        assert!(!is_model_gguf("model.imatrix.gguf"));
        assert!(!is_model_gguf("mtp-model-q8_0.gguf"));
        assert!(!is_model_gguf("README.md"));
    }

    #[test]
    fn shard_info_parses_the_suffix_and_leaves_unsharded_files_alone() {
        assert_eq!(
            shard_info("model-00002-of-00004.gguf"),
            Some(("model".to_string(), 2, 4))
        );
        assert_eq!(shard_info("model-Q4_K_M.gguf"), None);
        // Not a valid shard suffix (not 5 digits) — left untouched.
        assert_eq!(shard_info("model-1-of-4.gguf"), None);
    }

    fn file(path: &str, oid: &str, size: u64) -> RepoFile {
        RepoFile {
            path: path.to_string(),
            oid: oid.to_string(),
            size,
        }
    }

    #[test]
    fn select_files_to_download_honors_an_explicit_tag() {
        let files = vec![
            file("model-Q4_K_M.gguf", "a", 1),
            file("model-Q8_0.gguf", "b", 2),
        ];
        let selected = select_files_to_download(&files, Some("Q8_0")).unwrap();
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].path, "model-Q8_0.gguf");
    }

    #[test]
    fn select_files_to_download_errors_when_the_requested_tag_is_absent() {
        let files = vec![file("model-Q4_K_M.gguf", "a", 1)];
        let err = select_files_to_download(&files, Some("Q8_0")).unwrap_err();
        assert!(err.to_string().contains("Q4_K_M"), "{err}");
    }

    #[test]
    fn select_files_to_download_prefers_q4_k_m_then_q8_0_by_default() {
        let files = vec![
            file("model-Q8_0.gguf", "a", 1),
            file("model-Q4_K_M.gguf", "b", 2),
            file("model-F16.gguf", "c", 3),
        ];
        let selected = select_files_to_download(&files, None).unwrap();
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].path, "model-Q4_K_M.gguf");
    }

    #[test]
    fn select_files_to_download_falls_back_to_the_first_model_file() {
        let files = vec![
            file("mmproj-model-bf16.gguf", "a", 1),
            file("model-F16.gguf", "b", 2),
        ];
        let selected = select_files_to_download(&files, None).unwrap();
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].path, "model-F16.gguf");
    }

    #[test]
    fn select_files_to_download_collects_every_shard_in_order() {
        let files = vec![
            file("model-Q4_K_M-00002-of-00003.gguf", "b", 2),
            file("model-Q4_K_M-00003-of-00003.gguf", "c", 3),
            file("model-Q4_K_M-00001-of-00003.gguf", "a", 1),
            file("other-Q4_K_M-00001-of-00002.gguf", "x", 9),
        ];
        let selected = select_files_to_download(&files, Some("Q4_K_M")).unwrap();
        assert_eq!(
            selected.iter().map(|f| f.path.as_str()).collect::<Vec<_>>(),
            vec![
                "model-Q4_K_M-00001-of-00003.gguf",
                "model-Q4_K_M-00002-of-00003.gguf",
                "model-Q4_K_M-00003-of-00003.gguf",
            ]
        );
    }

    #[test]
    fn select_files_to_download_never_picks_a_non_first_shard_as_primary() {
        // Only shard 2 of 2 happens to mention the tag in this contrived
        // case; the algorithm must not treat it as a standalone primary.
        let files = vec![file("model-Q4_K_M-00002-of-00002.gguf", "b", 2)];
        assert!(select_files_to_download(&files, Some("Q4_K_M")).is_err());
    }

    #[test]
    fn select_files_to_download_errors_without_any_model_files() {
        let files = vec![file("README.md", "a", 1), file("mmproj-x.gguf", "b", 2)];
        assert!(select_files_to_download(&files, None).is_err());
    }

    #[test]
    fn extract_quant_bits_reads_the_first_digit_run_in_the_tag() {
        assert_eq!(extract_quant_bits("model-Q4_K_M.gguf"), 4);
        assert_eq!(extract_quant_bits("mmproj-BF16.gguf"), 16);
        assert_eq!(extract_quant_bits("mmproj-F32.gguf"), 32);
        assert_eq!(extract_quant_bits("model-Q8_0-00001-of-00003.gguf"), 8);
        assert_eq!(extract_quant_bits("no-digits-here.gguf"), 0);
    }

    #[test]
    fn find_best_mmproj_prefers_the_closest_quant_bit_depth() {
        // Mirrors the real unsloth/Qwen3.6-35B-A3B-GGUF layout: three
        // top-level mmproj variants alongside the selected top-level model
        // file — llama-server's own `-hf` picks BF16 here too (closest bit
        // depth to Q4_K_M's 4, tie broken by listing order).
        let files = vec![
            file("Qwen3.6-35B-A3B-UD-Q4_K_M.gguf", "m", 1),
            file("mmproj-BF16.gguf", "a", 2),
            file("mmproj-F16.gguf", "b", 3),
            file("mmproj-F32.gguf", "c", 4),
        ];
        let best = find_best_mmproj(&files, "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf").unwrap();
        assert_eq!(best.path, "mmproj-BF16.gguf");
    }

    #[test]
    fn find_best_mmproj_prefers_a_deeper_shared_directory() {
        let files = vec![
            file("Q4_K_M/model-Q4_K_M.gguf", "m", 1),
            file("mmproj-F16.gguf", "a", 2),
            file("Q4_K_M/mmproj-F16.gguf", "b", 3),
        ];
        let best = find_best_mmproj(&files, "Q4_K_M/model-Q4_K_M.gguf").unwrap();
        assert_eq!(best.path, "Q4_K_M/mmproj-F16.gguf");
    }

    #[test]
    fn find_best_mmproj_returns_none_without_a_sidecar() {
        let files = vec![file("model-Q4_K_M.gguf", "m", 1)];
        assert!(find_best_mmproj(&files, "model-Q4_K_M.gguf").is_none());
    }

    #[test]
    fn urlencode_path_escapes_special_characters_but_not_slashes() {
        assert_eq!(
            urlencode_path("sub/model file.gguf"),
            "sub/model%20file.gguf"
        );
        assert_eq!(
            urlencode_path("bartowski/model-Q4_K_M.gguf"),
            "bartowski/model-Q4_K_M.gguf"
        );
    }
}
