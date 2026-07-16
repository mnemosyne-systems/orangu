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

//! The persistent embedding index and hybrid search.
//!
//! The index lives under `~/.orangu/workspace/<hash>/embeddings/`, keyed by a
//! hash of the workspace path so it is shared across sessions without cluttering
//! the workspace tree. Vectors are stored append-only in `chunks.json`, with a
//! small `meta.json` sidecar (version + per-file hashes) and a `processed.log`.
//! A per-file sha256 map drives incremental rebuilds, so only changed files are
//! re-embedded. Search embeds the query, ranks chunks by cosine similarity, then
//! expands the top matches along the knowledge graph's call edges — a semantic
//! seed followed by structural expansion.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::chunk::chunk_file;
use super::client::EmbeddingClient;
use crate::graph::extract::{GraphExtractor, SupportedLanguage};
use crate::graph::store::GraphStore;

const INDEX_VERSION: u32 = 1;

/// Approximate character-per-token ratio used to size embedding batches. Coarse
/// (real tokenizers vary), but conservative enough to leave headroom under a
/// typical llama.cpp server's default physical batch (`-b`/`--batch-size 512`
/// tokens) without requiring users to raise it.
const CHARS_PER_TOKEN_ESTIMATE: usize = 4;

/// Target token budget per `/v1/embeddings` request — comfortably under the
/// llama.cpp default physical batch of 512 tokens, so a batch built from several
/// chunks does not trip "input is too large to process" on a stock server.
const EMBED_BATCH_TOKEN_BUDGET: usize = 350;
const EMBED_BATCH_CHAR_BUDGET: usize = EMBED_BATCH_TOKEN_BUDGET * CHARS_PER_TOKEN_ESTIMATE;

/// A safety cap on how many chunks join one request even when they are all
/// tiny, so a file with hundreds of one-line functions still amortises HTTP
/// overhead without producing an unreasonably large request.
const EMBED_BATCH_MAX_COUNT: usize = 64;

/// How many files embed concurrently. The upload to the embedding server is the
/// bottleneck, so several requests are kept in flight to keep the server busy;
/// this also bounds peak memory to the files currently being embedded.
const EMBED_CONCURRENCY: usize = 8;

/// Score multiplier applied to a chunk pulled in only because it neighbours a
/// strong semantic hit in the knowledge graph. Below `1.0` so a graph-expanded
/// result never outranks the semantic hit that surfaced it.
const EXPANSION_DECAY: f32 = 0.6;

// ── On-disk format ────────────────────────────────────────────────────────────

/// One embedded code chunk: a symbol plus its vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddedChunk {
    pub id: String,
    pub symbol: String,
    pub file: String,
    pub start_line: usize,
    pub end_line: usize,
    pub vector: Vec<f32>,
}

/// The full index: version, per-file hashes for incremental rebuilds, and the
/// embedded chunks.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct EmbeddingIndex {
    #[serde(default)]
    version: u32,
    #[serde(default)]
    file_hashes: HashMap<String, String>,
    #[serde(default)]
    pub module_sketches: HashMap<String, super::sketch::EllipsoidSketch>,
    #[serde(default)]
    chunks: Vec<EmbeddedChunk>,
}

/// One ranked search result.
#[derive(Debug, Clone)]
pub struct SearchHit {
    pub id: String,
    pub symbol: String,
    pub file: String,
    pub start_line: usize,
    pub end_line: usize,
    pub score: f32,
    /// `None` for a direct semantic hit, or `Some(symbol)` when this chunk was
    /// pulled in because it neighbours that semantic hit in the graph.
    pub expanded_from: Option<String>,
}

/// A completed file's embedding result: its workspace-relative path, content
/// hash, embedded chunks, and total chunk-text bytes (for progress weighting).
type FileEmbed = (String, String, Vec<EmbeddedChunk>, u64);

/// The small metadata sidecar: the version and per-file hashes. Kept apart from
/// the (potentially large) chunk vectors so it can be rewritten cheaply after
/// each file without re-serialising every vector.
#[derive(Debug, Default, Serialize, Deserialize)]
struct Meta {
    #[serde(default)]
    version: u32,
    #[serde(default)]
    file_hashes: HashMap<String, String>,
    #[serde(default)]
    module_sketches: HashMap<String, super::sketch::EllipsoidSketch>,
}

impl EmbeddingIndex {
    /// The cache directory for `workspace`, kept out of the workspace tree in the
    /// global, per-workspace orangu directory:
    ///
    /// ```text
    /// ~/.orangu/workspace/<sha256(path)>/embeddings/
    /// ```
    ///
    /// It holds `chunks.json` (one embedded chunk per line, appended as files
    /// are embedded), `meta.json` (version and per-file hashes), and
    /// `processed.log` (each file's path and completion time). Falls back to the
    /// workspace tree when no home directory resolves.
    pub fn cache_dir(workspace: &Path) -> PathBuf {
        crate::workspace_cache::workspace_cache_dir(workspace, "embeddings")
    }

    fn chunks_path(dir: &Path) -> PathBuf {
        dir.join("chunks.json")
    }
    fn meta_path(dir: &Path) -> PathBuf {
        dir.join("meta.json")
    }
    fn manifest_path(dir: &Path) -> PathBuf {
        dir.join("processed.log")
    }

    /// Load the index for `workspace`, or an empty index when the cache is
    /// missing or a stale version (which forces a full rebuild).
    pub fn load(workspace: &Path) -> Self {
        let dir = Self::cache_dir(workspace);
        let meta: Meta = std::fs::read_to_string(Self::meta_path(&dir))
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        if meta.version != INDEX_VERSION {
            return Self::default();
        }
        // Read chunks line by line so a truncated final line (e.g. after a hard
        // kill mid-append) is skipped rather than discarding the whole cache.
        let chunks = std::fs::read_to_string(Self::chunks_path(&dir))
            .map(|s| {
                s.lines()
                    .filter_map(|line| serde_json::from_str::<EmbeddedChunk>(line).ok())
                    .collect()
            })
            .unwrap_or_default();
        Self {
            version: INDEX_VERSION,
            file_hashes: meta.file_hashes,
            module_sketches: meta.module_sketches,
            chunks,
        }
    }

    /// Number of embedded chunks currently in the index.
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    /// Whether the index holds no chunks.
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Build or incrementally update the index for `workspace`, re-embedding
    /// only files whose content hash changed since the last build and dropping
    /// chunks for files that no longer exist — so the cache always reflects the
    /// current workspace.
    ///
    /// Runs in two phases, each half of the reported progress:
    ///
    /// 1. **Local** (0–50%): parse every stale file into chunks, in parallel
    ///    across `compile_workers` threads (`0` = every CPU thread). All the
    ///    local work is done here, up front; nothing is uploaded. When it
    ///    finishes, the parsed files are logged to `processed.log` and the full
    ///    `meta.json` (every file's hash) and reused chunks are written.
    /// 2. **Upload** (50–100%): embed the parsed chunks, keeping several requests
    ///    in flight at once so the embedding server (the real bottleneck) stays
    ///    busy. Files finish out of order and their chunks are appended to
    ///    `chunks.json` as they complete.
    ///
    /// Progress and a total-time estimate are published to the `progress`
    /// (permille) and `eta` (estimated total milliseconds) atomics. `cancel` is
    /// polled while requests are in flight and aborts them promptly.
    ///
    /// Persistence is append-based, so it is cheap no matter how large the index
    /// grows and it survives a hard kill: a later run reuses the files whose
    /// embedded chunks are present and re-embeds the rest. The up-to-date index
    /// is also returned so the caller can search it without re-loading.
    pub async fn build_or_update(
        workspace: &Path,
        client: &EmbeddingClient,
        cancel: &AtomicBool,
        compile_workers: usize,
        progress: &std::sync::atomic::AtomicU64,
        eta: &std::sync::atomic::AtomicU64,
    ) -> Result<Self> {
        let started = std::time::Instant::now();
        let mut previous = Self::load(workspace);

        // Classify every current file. Its hash always goes into `new_hashes`
        // (so the saved `meta.json` lists every file once the local phase is
        // done); an unchanged file that still has cached chunks is reused, the
        // rest are queued for (re-)embedding.
        let files = collect_source_files(workspace);
        let mut new_hashes: HashMap<String, String> = HashMap::new();
        let mut kept: Vec<EmbeddedChunk> = Vec::new();
        let mut stale_files: Vec<(PathBuf, String, String)> = Vec::new();

        // Index previous chunks by file for reuse.
        let mut prev_by_file: HashMap<String, Vec<EmbeddedChunk>> = HashMap::new();
        for chunk in std::mem::take(&mut previous.chunks) {
            prev_by_file
                .entry(chunk.file.clone())
                .or_default()
                .push(chunk);
        }

        for path in files {
            let rel = path
                .strip_prefix(workspace)
                .unwrap_or(&path)
                .to_string_lossy()
                .to_string();
            let Ok(content) = std::fs::read_to_string(&path) else {
                continue;
            };
            let hash = sha256(&content);
            new_hashes.insert(rel.clone(), hash.clone());
            // Reuse only when unchanged AND the chunks are actually cached — a
            // file that was analysed but not embedded (no chunks) is re-embedded.
            if previous.file_hashes.get(&rel) == Some(&hash)
                && let Some(chunks) = prev_by_file.get(&rel)
            {
                kept.extend(chunks.iter().cloned());
            } else {
                stale_files.push((path, rel, hash));
            }
        }

        let dir = Self::cache_dir(workspace);
        if previous.file_hashes.is_empty() {
            let _ = std::fs::remove_file(Self::manifest_path(&dir));
        }

        // ── Phase 1 (0–50%): parse every stale file locally, up front ─────────
        // All the local work — reading and Tree-sitter parsing — happens here, in
        // parallel across `compile_workers` threads (`0` = every CPU thread).
        // Nothing is uploaded yet; the parsed chunks are held for phase 2.
        let threads = if compile_workers > 0 {
            compile_workers
        } else {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        };
        let extractor = GraphExtractor::new()?;
        let analysed = std::sync::atomic::AtomicUsize::new(0);
        let total_files = stale_files.len().max(1);
        let parse_one = |(_path, rel, hash): &(PathBuf, String, String)| {
            let content = std::fs::read_to_string(_path).ok()?;
            let chunks = chunk_file(&extractor, _path, rel, &content);
            let done = analysed.fetch_add(1, Ordering::Relaxed) + 1;
            progress.store(
                (done as u64 * 500 / total_files as u64).min(500),
                Ordering::Relaxed,
            );
            Some((rel.clone(), hash.clone(), chunks))
        };
        let parsed: Vec<(String, String, Vec<super::chunk::Chunk>)> =
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()?
                .install(|| stale_files.par_iter().filter_map(parse_one).collect());
        // Log the parsed files to `processed.log` in one sequential write — the
        // parallel parse above must not touch the file, or concurrent appends
        // interleave and corrupt it.
        append_manifest(
            &Self::manifest_path(&dir),
            parsed.iter().map(|(rel, _, _)| rel.as_str()),
        );
        let total_bytes: u64 = parsed
            .iter()
            .map(|(_, _, chunks)| chunks.iter().map(|c| c.text.len() as u64).sum::<u64>())
            .sum::<u64>()
            .max(1);

        // Local phase done: seed the cache with the reused chunks and write the
        // full meta (every file), so the on-disk state lists all local files
        // before any upload starts.
        let mut index = Self {
            version: INDEX_VERSION,
            file_hashes: new_hashes,
            module_sketches: HashMap::new(),
            chunks: kept,
        };
        index.rewrite_cache(&dir)?;

        // ── Phase 2 (50–100%): upload — embed the parsed chunks concurrently ──
        // The upload to the embedding server is the bottleneck, so keep up to
        // `EMBED_CONCURRENCY` files in flight — a llama-server started with
        // `-np N` runs them in parallel, and even with one slot the next request
        // is ready the moment the server frees up. Files finish out of order and
        // their chunks are appended as they complete, so a cancel or kill keeps
        // whatever is done. A double-`Esc` aborts the in-flight requests at once.
        let chunks_path = Self::chunks_path(&dir);
        let phase2_start = std::time::Instant::now();
        let mut done_bytes: u64 = 0;
        let mut iter = parsed.into_iter();
        let mut set: tokio::task::JoinSet<Result<FileEmbed>> = tokio::task::JoinSet::new();

        macro_rules! spawn_next {
            () => {
                if let Some((rel, hash, chunks)) = iter.next() {
                    let client = client.clone();
                    set.spawn(async move {
                        let bytes: u64 = chunks.iter().map(|c| c.text.len() as u64).sum();
                        let mut embedded = Vec::with_capacity(chunks.len());
                        for batch in batches_within_budget(&chunks) {
                            let inputs: Vec<String> =
                                batch.iter().map(|c| c.text.clone()).collect();
                            let vectors = client.embed(&inputs).await?;
                            for (chunk, vector) in batch.iter().zip(vectors) {
                                embedded.push(EmbeddedChunk {
                                    id: chunk.id.clone(),
                                    symbol: chunk.symbol.clone(),
                                    file: chunk.file.clone(),
                                    start_line: chunk.start_line,
                                    end_line: chunk.end_line,
                                    vector,
                                });
                            }
                        }
                        Ok((rel, hash, embedded, bytes))
                    });
                }
            };
        }

        for _ in 0..EMBED_CONCURRENCY {
            spawn_next!();
        }
        loop {
            // Poll for a finished file, but wake at least every 150ms so a
            // double-Esc is honoured promptly even while requests are in flight.
            tokio::select! {
                maybe = set.join_next() => {
                    let Some(joined) = maybe else { break };
                    let (_rel, _hash, embedded, bytes) = joined??;
                    // The file's hash is already in meta; append its chunks.
                    append_chunks(&chunks_path, &embedded)?;
                    index.chunks.extend(embedded);
                    done_bytes = done_bytes.saturating_add(bytes);
                    let permille = (500 + done_bytes.saturating_mul(500) / total_bytes).min(1000);
                    progress.store(permille, Ordering::Relaxed);
                    // Estimate remaining time from the overall embedding rate (a
                    // steadier signal than any single burst); the run loop counts
                    // down from it. Held until a couple of seconds of data exist.
                    let p2 = phase2_start.elapsed().as_millis();
                    if p2 > 2_000 {
                        let remaining_ms = p2
                            .saturating_mul(total_bytes.saturating_sub(done_bytes) as u128)
                            .checked_div(done_bytes.max(1) as u128)
                            .unwrap_or(0) as u64;
                        eta.store(started.elapsed().as_millis() as u64 + remaining_ms, Ordering::Relaxed);
                    }
                    spawn_next!();
                }
                _ = tokio::time::sleep(std::time::Duration::from_millis(150)) => {}
            }
            if cancel.load(Ordering::Relaxed) {
                // Abort the in-flight requests immediately rather than draining.
                set.shutdown().await;
                break;
            }
        }

        let mut chunks_by_file: HashMap<String, Vec<&[f32]>> = HashMap::new();
        for chunk in &index.chunks {
            chunks_by_file
                .entry(chunk.file.clone())
                .or_default()
                .push(&chunk.vector);
        }
        for (file, vectors) in chunks_by_file {
            index
                .module_sketches
                .insert(file, super::sketch::EllipsoidSketch::compute(&vectors));
        }
        index.write_meta(&dir)?;

        Ok(index)
    }

    /// Rewrite `chunks.json` (one chunk per line) and `meta.json` from scratch —
    /// used once at the end of the local phase to seed the cache with the reused
    /// chunks and record every file's hash.
    fn rewrite_cache(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        let mut body = String::new();
        for chunk in &self.chunks {
            if let Ok(line) = serde_json::to_string(chunk) {
                body.push_str(&line);
                body.push('\n');
            }
        }
        std::fs::write(Self::chunks_path(dir), body)?;
        self.write_meta(dir)?;
        Ok(())
    }

    /// Write the small `meta.json` sidecar (version + per-file hashes).
    fn write_meta(&self, dir: &Path) -> Result<()> {
        let meta = Meta {
            version: self.version,
            file_hashes: self.file_hashes.clone(),
            module_sketches: self.module_sketches.clone(),
        };
        let json = serde_json::to_string(&meta)?;
        std::fs::create_dir_all(dir)?;
        std::fs::write(Self::meta_path(dir), json)?;
        Ok(())
    }

    /// Hybrid search: rank chunks by cosine similarity to `query_vector`, then,
    /// when a graph is available, expand the strongest hits along their call
    /// edges. Returns up to `top_k` results, most relevant first.
    pub fn search(
        &self,
        query_vector: &[f32],
        graph: Option<&GraphStore>,
        top_k: usize,
        semantic_budget_tokens: usize,
    ) -> Vec<SearchHit> {
        if self.chunks.is_empty() || query_vector.is_empty() {
            return Vec::new();
        }

        // Occupancy-Aware Sketch Filtering
        // Determine candidate modules that are close enough to the query.
        // Recall safety net: always force-include the 3 most recently modified files
        // to prevent over-aggressive pruning of the developer's immediate context.
        let mut candidate_files = std::collections::HashSet::new();

        let mut mtimes: Vec<(String, std::time::SystemTime)> = Vec::new();
        for file in self.module_sketches.keys() {
            if let Ok(mtime) = std::fs::metadata(file).and_then(|m| m.modified()) {
                mtimes.push((file.clone(), mtime));
            }
        }
        mtimes.sort_by_key(|b| std::cmp::Reverse(b.1));
        for (file, _) in mtimes.into_iter().take(3) {
            candidate_files.insert(file);
        }

        let threshold = 0.35;
        for (file, sketch) in &self.module_sketches {
            let match_result = sketch.matches(query_vector, threshold);
            if match_result.inside {
                candidate_files.insert(file.clone());
            }
        }

        // Track divergence/recall in debug builds
        #[cfg(debug_assertions)]
        let _unfiltered_count = self.chunks.len();

        let filter_candidates = !candidate_files.is_empty();

        // Semantic pass: cosine of the query against every chunk.
        let mut scored: Vec<(usize, f32)> = self
            .chunks
            .iter()
            .enumerate()
            .filter(|(_, c)| !filter_candidates || candidate_files.contains(&c.file))
            .map(|(i, c)| (i, cosine(query_vector, &c.vector)))
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));

        // Best score seen per chunk id, so expansion never demotes a direct hit.
        let mut best: HashMap<String, SearchHit> = HashMap::new();
        let id_to_index: HashMap<&str, usize> = self
            .chunks
            .iter()
            .enumerate()
            .map(|(i, c)| (c.id.as_str(), i))
            .collect();

        // Take the strongest semantic hits as seeds for expansion.
        // Pluribus Diversity-Aware Planner: track selected seeds to penalize redundant semantics
        let seed_count = top_k.saturating_mul(2).max(top_k);
        let mut selected_seed_indices: Vec<usize> = Vec::new();

        const DIVERSITY_THRESHOLD: f32 = 0.85;
        const DIVERSITY_PENALTY: f32 = 0.5;

        let mut penalized_scores: Vec<(usize, f32)> = Vec::with_capacity(scored.len());

        for &(idx, original_score) in &scored {
            let chunk = &self.chunks[idx];
            let mut score = original_score;
            let mut penalized = false;
            for &s_idx in &selected_seed_indices {
                let sim = cosine(&chunk.vector, &self.chunks[s_idx].vector);
                if sim > DIVERSITY_THRESHOLD {
                    score *= DIVERSITY_PENALTY;
                    penalized = true;
                    break;
                }
            }
            if !penalized {
                selected_seed_indices.push(idx);
            }
            penalized_scores.push((idx, score));
        }

        penalized_scores.sort_by(|a, b| b.1.total_cmp(&a.1));

        for &(idx, score) in penalized_scores.iter().take(seed_count) {
            let chunk = &self.chunks[idx];
            insert_hit(&mut best, hit_from(chunk, score, None));

            // Structural expansion: pull the graph neighbours of this symbol.
            if let Some(graph) = graph {
                for result in graph.lookup(&chunk.symbol) {
                    if result.node.id != chunk.id {
                        continue;
                    }
                    let neighbours = result.callers.iter().chain(result.callees.iter());
                    for edge in neighbours {
                        if let Some(&nidx) = id_to_index.get(edge.node_id.as_str()) {
                            let n = &self.chunks[nidx];
                            let boosted = score * EXPANSION_DECAY;
                            insert_hit(&mut best, hit_from(n, boosted, Some(chunk.symbol.clone())));
                        }
                    }
                }
            }
        }

        let mut hits: Vec<SearchHit> = best.into_values().collect();
        hits.sort_by(|a, b| b.score.total_cmp(&a.score));

        let mut total_tokens = 0;
        let mut final_hits = Vec::new();
        for hit in hits {
            let chunk_lines = hit.end_line.saturating_sub(hit.start_line) + 1;
            let chunk_tokens = chunk_lines * 10;
            if total_tokens + chunk_tokens > semantic_budget_tokens && !final_hits.is_empty() {
                break;
            }
            total_tokens += chunk_tokens;
            final_hits.push(hit);
            if final_hits.len() >= top_k {
                break;
            }
        }

        final_hits
    }
}

/// Insert `hit` into `best` unless an equal-or-better score for the same id is
/// already recorded.
fn insert_hit(best: &mut HashMap<String, SearchHit>, hit: SearchHit) {
    match best.get(&hit.id) {
        Some(existing) if existing.score >= hit.score => {}
        _ => {
            best.insert(hit.id.clone(), hit);
        }
    }
}

fn hit_from(chunk: &EmbeddedChunk, score: f32, expanded_from: Option<String>) -> SearchHit {
    SearchHit {
        id: chunk.id.clone(),
        symbol: chunk.symbol.clone(),
        file: chunk.file.clone(),
        start_line: chunk.start_line,
        end_line: chunk.end_line,
        score,
        expanded_from,
    }
}

/// Cosine similarity of two vectors. Returns `0.0` for a length mismatch or a
/// zero-magnitude vector.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Group `chunks` into request-sized batches that stay under
/// [`EMBED_BATCH_CHAR_BUDGET`] (and [`EMBED_BATCH_MAX_COUNT`]), so a request
/// never trips a llama.cpp server's default physical batch size. A single chunk
/// that alone exceeds the budget (should not happen given `MAX_CHUNK_CHARS`, but
/// guarded regardless) still goes out alone rather than being dropped.
fn batches_within_budget(chunks: &[super::chunk::Chunk]) -> Vec<&[super::chunk::Chunk]> {
    let mut batches = Vec::new();
    let mut start = 0;
    let mut running_chars = 0usize;
    for (i, chunk) in chunks.iter().enumerate() {
        let len = chunk.text.len();
        let would_exceed = running_chars > 0
            && (running_chars + len > EMBED_BATCH_CHAR_BUDGET
                || i - start >= EMBED_BATCH_MAX_COUNT);
        if would_exceed {
            batches.push(&chunks[start..i]);
            start = i;
            running_chars = 0;
        }
        running_chars += len;
    }
    if start < chunks.len() {
        batches.push(&chunks[start..]);
    }
    batches
}

/// Append embedded chunks to `chunks.json`, one JSON object per line. Appending
/// keeps per-file persistence cheap — the existing vectors are never rewritten.
fn append_chunks(path: &Path, chunks: &[EmbeddedChunk]) -> Result<()> {
    use std::io::Write;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut body = String::new();
    for chunk in chunks {
        if let Ok(line) = serde_json::to_string(chunk) {
            body.push_str(&line);
            body.push('\n');
        }
    }
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    f.write_all(body.as_bytes())?;
    Ok(())
}

/// Append `<unix-timestamp>\t<path>` lines to the processed-files manifest, one
/// per file, in a single write. Must be called from a single thread — concurrent
/// appends interleave and corrupt the file. Errors are ignored; the manifest is
/// a progress aid, never load-bearing.
fn append_manifest<'a>(path: &Path, files: impl Iterator<Item = &'a str>) {
    use std::io::Write;
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut body = String::new();
    for file in files {
        body.push_str(&format!("{ts}\t{file}\n"));
    }
    if body.is_empty() {
        return;
    }
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = f.write_all(body.as_bytes());
    }
}

/// Compute the sha256 hex digest of `content`.
fn sha256(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Collect the workspace's source files, honouring `.gitignore`, restricted to
/// languages the Tree-sitter extractor understands.
fn collect_source_files(workspace: &Path) -> Vec<PathBuf> {
    ignore::WalkBuilder::new(workspace)
        .hidden(true)
        .git_ignore(true)
        .build()
        .filter_map(|entry| {
            let path = entry.ok()?.into_path();
            if !path.is_file() {
                return None;
            }
            let ext = path.extension().and_then(|e| e.to_str())?;
            SupportedLanguage::from_extension(ext).map(|_| path)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk_of_len(chars: usize) -> super::super::chunk::Chunk {
        super::super::chunk::Chunk {
            id: "f::c".into(),
            symbol: "c".into(),
            file: "f.rs".into(),
            start_line: 1,
            end_line: 1,
            text: "x".repeat(chars),
        }
    }

    #[test]
    fn batches_within_budget_splits_when_char_budget_exceeded() {
        // Three chunks each at 60% of the char budget: batching all three would
        // overshoot, so they must land in separate requests.
        let big = EMBED_BATCH_CHAR_BUDGET * 6 / 10;
        let chunks = vec![chunk_of_len(big), chunk_of_len(big), chunk_of_len(big)];
        let batches = batches_within_budget(&chunks);
        assert_eq!(batches.len(), 3);
        for batch in &batches {
            assert_eq!(batch.len(), 1);
        }
    }

    #[test]
    fn batches_within_budget_groups_small_chunks_together() {
        // Many tiny chunks should still be grouped into as few requests as the
        // budget allows, not one request per chunk.
        let chunks: Vec<_> = (0..10).map(|_| chunk_of_len(10)).collect();
        let batches = batches_within_budget(&chunks);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 10);
    }

    #[test]
    fn batches_within_budget_respects_max_count_even_when_small() {
        // More chunks than EMBED_BATCH_MAX_COUNT, all tiny — the count cap must
        // still split them into multiple requests.
        let chunks: Vec<_> = (0..(EMBED_BATCH_MAX_COUNT + 5))
            .map(|_| chunk_of_len(1))
            .collect();
        let batches = batches_within_budget(&chunks);
        assert!(batches.len() >= 2);
        assert!(batches.iter().all(|b| b.len() <= EMBED_BATCH_MAX_COUNT));
    }

    #[test]
    fn cosine_of_identical_vectors_is_one() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_of_orthogonal_vectors_is_zero() {
        assert_eq!(cosine(&[1.0, 0.0], &[0.0, 1.0]), 0.0);
    }

    #[test]
    fn cosine_handles_length_mismatch_and_zero() {
        assert_eq!(cosine(&[1.0, 2.0], &[1.0]), 0.0);
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    }

    #[test]
    fn search_ranks_closest_chunk_first() {
        let index = EmbeddingIndex {
            version: INDEX_VERSION,
            file_hashes: HashMap::new(),
            module_sketches: HashMap::new(),
            chunks: vec![
                EmbeddedChunk {
                    id: "a::one".into(),
                    symbol: "one".into(),
                    file: "a.rs".into(),
                    start_line: 1,
                    end_line: 2,
                    vector: vec![1.0, 0.0],
                },
                EmbeddedChunk {
                    id: "a::two".into(),
                    symbol: "two".into(),
                    file: "a.rs".into(),
                    start_line: 3,
                    end_line: 4,
                    vector: vec![0.0, 1.0],
                },
            ],
        };
        let hits = index.search(&[0.9, 0.1], None, 5, 16384);
        assert_eq!(hits.first().unwrap().symbol, "one");
    }

    #[test]
    fn search_on_empty_index_returns_nothing() {
        let index = EmbeddingIndex::default();
        assert!(index.search(&[1.0, 0.0], None, 5, 16384).is_empty());
    }

    #[test]
    fn search_diversity_aware_planner_penalizes_redundant_semantics() {
        let index = EmbeddingIndex {
            version: INDEX_VERSION,
            file_hashes: HashMap::new(),
            module_sketches: HashMap::new(),
            chunks: vec![
                EmbeddedChunk {
                    id: "a::one".into(),
                    symbol: "one".into(),
                    file: "a.rs".into(),
                    start_line: 1,
                    end_line: 2,
                    vector: vec![1.0, 0.0],
                },
                EmbeddedChunk {
                    id: "a::two".into(),
                    symbol: "two".into(),
                    file: "a.rs".into(),
                    start_line: 3,
                    end_line: 4,
                    // Highly similar to "one" (cosine > 0.85)
                    vector: vec![0.99, 0.14],
                },
                EmbeddedChunk {
                    id: "a::three".into(),
                    symbol: "three".into(),
                    file: "a.rs".into(),
                    start_line: 5,
                    end_line: 6,
                    // Less similar to "one" but stronger than penalized "two"
                    vector: vec![0.8, 0.6],
                },
            ],
        };
        // Query is exactly [1.0, 0.0]
        // "one" has score 1.0
        // "two" has score ~0.99. Similarity to "one" > 0.85 -> Penalty applied -> Score 0.495
        // "three" has score 0.8. Similarity to "one" = 0.8 < 0.85 -> No penalty -> Score 0.8
        let hits = index.search(&[1.0, 0.0], None, 2, 16384);

        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].symbol, "one");
        // "three" should beat "two" due to the diversity penalty
        assert_eq!(hits[1].symbol, "three");
    }
}
