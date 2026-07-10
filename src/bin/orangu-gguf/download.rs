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

//! `orangu-gguf download <user>/<model>[:quant]`: downloads a GGUF model
//! from the Hugging Face Hub into the configured `models` directory, laid
//! out exactly the way llama.cpp's own `-hf`/`--hf-repo` downloads into —
//! `models--<user>--<model>/{blobs,refs,snapshots}` — so `list`/`show`/the
//! role wizard already read what this writes, and llama.cpp itself
//! recognizes it as already downloaded rather than fetching it again.
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

use crate::config::GgufConfiguration;
use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;
use std::{
    fs::{self, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
};

const HUB_ENDPOINT: &str = "https://huggingface.co";
/// The same fallback preference order llama.cpp's own `find_best_model`
/// uses when a `download` target names a repo but no `:quant` — asked for
/// in that order, first match wins.
const DEFAULT_TAG_PREFERENCE: &[&str] = &["Q4_K_M", "Q8_0"];

pub fn run_download(config: &GgufConfiguration, spec: &str) -> Result<()> {
    let (repo, tag) = split_repo_tag(spec)?;
    let client = build_client()?;
    let token = std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty());

    let commit = resolve_commit(&client, &repo, token.as_deref())?;
    let files = list_repo_files(&client, &repo, &commit, token.as_deref())?;
    let mut selected = select_files_to_download(&files, tag.as_deref())
        .with_context(|| format!("no matching GGUF file in {repo}"))?;

    if let Some(mmproj) = find_best_mmproj(&files, &selected[0].path) {
        selected.push(mmproj);
    }

    let repo_dir = config.models.join(repo_folder_name(&repo));
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

    let total = selected.len();
    for (index, file) in selected.iter().enumerate() {
        let position = (index + 1, total);
        let blob_path = blobs_dir.join(&file.oid);
        let snapshot_path = snapshot_dir.join(&file.path);

        if blob_path.is_file() && fs::metadata(&blob_path)?.len() == file.size {
            println!(
                "{} already downloaded — skipping [{}/{total}]",
                file.path, position.0
            );
        } else {
            let url = format!(
                "{HUB_ENDPOINT}/{repo}/resolve/{commit}/{}",
                urlencode_path(&file.path)
            );
            download_with_resume(
                &client,
                &url,
                &blob_path,
                token.as_deref(),
                &file.path,
                file.size,
                position,
            )?;
        }

        if let Some(parent) = snapshot_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        if !snapshot_path.exists() {
            link_or_copy(&blob_path, &snapshot_path, &file.oid, &file.path)?;
        }
    }

    Ok(())
}

fn build_client() -> Result<reqwest::blocking::Client> {
    reqwest::blocking::Client::builder()
        .user_agent(concat!("orangu-gguf/", env!("CARGO_PKG_VERSION")))
        .build()
        .context("failed to build HTTP client")
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

/// Downloads `url` into `dest`, resuming from a `.part` file left over from
/// an interrupted attempt (via an HTTP `Range` request), and printing
/// percentage progress against `expected_size` as it goes.
fn download_with_resume(
    client: &reqwest::blocking::Client,
    url: &str,
    dest: &Path,
    token: Option<&str>,
    label: &str,
    expected_size: u64,
    position: (usize, usize),
) -> Result<()> {
    let (index, total) = position;
    // Blob filenames are bare content hashes with no extension of their own
    // for `Path::with_extension` to replace, so just append directly.
    let part_path = PathBuf::from(format!("{}.part", dest.display()));
    let mut resume_from = fs::metadata(&part_path).map(|m| m.len()).unwrap_or(0);
    if resume_from >= expected_size && expected_size > 0 {
        // A stale/complete .part from an interrupted run that never got
        // renamed; nothing left to fetch.
        resume_from = 0;
        fs::remove_file(&part_path).ok();
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
        fs::remove_file(&part_path).ok();
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
        .open(&part_path)
        .with_context(|| format!("failed to open {}", part_path.display()))?;

    let mut downloaded = resume_from;
    let mut buf = [0u8; 65536];
    let mut last_printed = None;
    let mut body = response;
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

        if let Some(percent) = (downloaded * 100)
            .checked_div(expected_size)
            .map(|p| p.min(100))
            && last_printed != Some(percent)
        {
            print!("\rDownloading {label}: {percent}% [{index}/{total}]");
            std::io::stdout().flush().ok();
            last_printed = Some(percent);
        }
    }
    println!();

    fs::rename(&part_path, dest)
        .with_context(|| format!("failed to finalize {}", dest.display()))?;
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
