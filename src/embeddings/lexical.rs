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

//! The lexical BM25 sidecar used by hybrid search.
//!
//! The sidecar lives next to the embedding cache as `lexical.json` and has its
//! own version, so tokenizer or postings changes never force a re-embedding.
//! It indexes the full extracted Tree-sitter symbol range (the embedding index
//! keeps its own 1,600-character truncation) and shares file discovery and
//! SHA-256 computation with [`super::index::EmbeddingIndex`], while tracking
//! which files *it* has successfully processed independently.
//!
//! Postings and per-document lengths are persisted rather than source text.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::Result;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::chunk::lexical_chunks;
use super::index::{EmbeddingIndex, SearchHit, budget_cut, hit_from_doc, insert_hit};
use super::source::{SourceFile, scan_workspace};
use crate::graph::extract::GraphExtractor;
use crate::graph::store::GraphStore;

/// Independent sidecar version. Bump when the tokenizer, postings format, or
/// BM25 fields change — never when the embedding format changes.
pub const LEXICAL_INDEX_VERSION: u32 = 1;

const BM25_K1: f32 = 1.2;
const BM25_B: f32 = 0.75;

/// How many BM25 candidates to source-verify for a quoted phrase or explicit
/// identifier. Common phrases can match many documents; verifying a larger set
/// is cheap because each candidate file is read at most once per query.
const MAX_VERIFY_CANDIDATES: usize = 5_000;

/// Token repetition weights approximate field weighting without a multi-field
/// BM25 implementation: a symbol name counts more than a file path, which in
/// turn counts more than the body.
const NAME_WEIGHT: usize = 3;
const PATH_WEIGHT: usize = 2;

/// One indexed symbol. The line range lets query-time verification inspect only
/// the symbol's source rather than the whole file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LexicalDoc {
    pub id: String,
    pub symbol: String,
    pub file: String,
    pub start_line: usize,
    pub end_line: usize,
    /// Weighted token count, used for BM25 length normalisation.
    pub length: u32,
}

/// One term occurrence in one document. Serialised as `[doc, tf]` to keep the
/// sidecar compact.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Posting(pub u32, pub u32);

/// The persisted lexical index: version, per-file hashes for incremental
/// updates, document metadata, and the term postings.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct LexicalIndex {
    #[serde(default)]
    version: u32,
    #[serde(default)]
    file_hashes: HashMap<String, String>,
    #[serde(default)]
    docs: Vec<LexicalDoc>,
    #[serde(default)]
    postings: HashMap<String, Vec<Posting>>,
    #[serde(default)]
    avg_length: f32,
}

/// One ranked lexical result, before graph expansion and budgeting.
#[derive(Debug, Clone)]
pub struct LexicalHit {
    pub id: String,
    pub symbol: String,
    pub file: String,
    pub start_line: usize,
    pub end_line: usize,
    pub score: f32,
    /// Verified as an exact identifier or quoted-phrase match in the source.
    pub exact: bool,
}

impl LexicalHit {
    fn to_search_hit(&self, expanded_from: Option<String>) -> SearchHit {
        SearchHit {
            id: self.id.clone(),
            symbol: self.symbol.clone(),
            file: self.file.clone(),
            start_line: self.start_line,
            end_line: self.end_line,
            score: self.score,
            expanded_from,
            exact: self.exact,
        }
    }
}

/// One document together with its term frequencies, used while building.
type DocWithTerms = (LexicalDoc, Vec<(String, u32)>);

/// One parsed file's documents.
type ParsedFile = Vec<DocWithTerms>;

impl LexicalIndex {
    /// The lexical sidecar path, next to the embedding index:
    ///
    /// ```text
    /// ~/.orangu/workspace/<sha256(path)>/embeddings/lexical.json
    /// ```
    pub fn cache_path(workspace: &Path) -> PathBuf {
        EmbeddingIndex::cache_dir(workspace).join("lexical.json")
    }

    /// Load the sidecar, or an empty index when it is missing or stale.
    pub fn load(workspace: &Path) -> Self {
        let Ok(text) = std::fs::read_to_string(Self::cache_path(workspace)) else {
            return Self::default();
        };
        let Ok(index) = serde_json::from_str::<Self>(&text) else {
            return Self::default();
        };
        if index.version != LEXICAL_INDEX_VERSION {
            return Self::default();
        }
        index
    }

    /// Number of indexed symbols.
    pub fn len(&self) -> usize {
        self.docs.len()
    }

    /// Whether the index holds no symbols.
    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    /// A map from document id to its metadata, for graph expansion by id.
    pub fn doc_by_id(&self) -> HashMap<&str, &LexicalDoc> {
        self.docs.iter().map(|doc| (doc.id.as_str(), doc)).collect()
    }

    /// Build or incrementally update the lexical index for `workspace`.
    ///
    /// Only files whose content hash changed are re-parsed and re-tokenised;
    /// unchanged documents and their term frequencies are carried over. The
    /// sidecar is then rewritten in full because postings are positional and
    /// corpus statistics change whenever any file changes.
    pub fn build_or_update(workspace: &Path, compile_workers: usize) -> Result<Self> {
        let scan = scan_workspace(workspace);
        Self::build_or_update_from_scan(workspace, &scan, compile_workers, None)
    }

    /// [`Self::build_or_update`] over a scan shared with the embedding index,
    /// optionally publishing progress (permille) for the lexical-only path,
    /// where no embedding build follows to report it.
    pub fn build_or_update_with_progress(
        workspace: &Path,
        compile_workers: usize,
        progress: Option<&std::sync::atomic::AtomicU64>,
    ) -> Result<Self> {
        let scan = scan_workspace(workspace);
        Self::build_or_update_from_scan(workspace, &scan, compile_workers, progress)
    }

    /// Build or update the sidecar from an existing workspace scan.
    pub fn build_or_update_from_scan(
        workspace: &Path,
        scan: &[SourceFile],
        compile_workers: usize,
        progress: Option<&std::sync::atomic::AtomicU64>,
    ) -> Result<Self> {
        use std::sync::atomic::Ordering;
        let report = |permille: u64| {
            if let Some(progress) = progress {
                progress.store(permille.min(1000), Ordering::Relaxed);
            }
        };
        report(0);

        let previous = Self::load(workspace);
        let previous_terms = previous.terms_by_index();
        let mut previous_by_file: HashMap<&str, Vec<usize>> = HashMap::new();
        for (index, doc) in previous.docs.iter().enumerate() {
            previous_by_file
                .entry(doc.file.as_str())
                .or_default()
                .push(index);
        }

        let current_files: HashSet<&str> = scan.iter().map(|file| file.rel.as_str()).collect();
        let mut new_hashes: HashMap<String, String> = HashMap::new();
        let mut kept: Vec<DocWithTerms> = Vec::new();
        let mut stale_files: Vec<SourceFile> = Vec::new();

        for file in scan {
            let unchanged = previous.file_hashes.get(&file.rel) == Some(&file.hash);
            if unchanged {
                // Keep every document previously emitted for this file. A file
                // with no symbols legitimately keeps none. Terms are carried by
                // document *position*, not id: one file can emit two symbols
                // with the same id (repeated method names), and keying by id
                // would give both copies the same term frequencies.
                new_hashes.insert(file.rel.clone(), file.hash.clone());
                if let Some(indices) = previous_by_file.get(file.rel.as_str()) {
                    for &index in indices {
                        let doc = previous.docs[index].clone();
                        let terms = previous_terms[index].clone();
                        kept.push((doc, terms));
                    }
                }
            } else {
                stale_files.push(file.clone());
            }
        }
        report(250);

        let removed_files = previous
            .file_hashes
            .keys()
            .any(|file| !current_files.contains(file.as_str()));
        if stale_files.is_empty() && !removed_files && previous.version == LEXICAL_INDEX_VERSION {
            report(1000);
            return Ok(previous);
        }

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
        let parse_one = |file: &SourceFile| {
            let content = std::fs::read_to_string(&file.path).ok()?;
            let chunks = lexical_chunks(&extractor, &file.path, &file.rel, &content);
            let docs: ParsedFile = chunks
                .into_iter()
                .map(|chunk| {
                    let (terms, length) = doc_terms(&chunk.symbol, &chunk.file, &chunk.body);
                    (
                        LexicalDoc {
                            id: chunk.id,
                            symbol: chunk.symbol,
                            file: chunk.file,
                            start_line: chunk.start_line,
                            end_line: chunk.end_line,
                            length,
                        },
                        terms,
                    )
                })
                .collect();
            let done = analysed.fetch_add(1, Ordering::Relaxed) + 1;
            report(250 + (done as u64 * 650 / total_files as u64).min(650));
            Some((file.rel.clone(), file.hash.clone(), docs))
        };
        let parsed: Vec<(String, String, ParsedFile)> = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()?
            .install(|| stale_files.par_iter().filter_map(parse_one).collect());

        // Record a hash only after that file parsed successfully, so a failed
        // parse is retried on the next run instead of being treated as done.
        for (rel, hash, _) in &parsed {
            new_hashes.insert(rel.clone(), hash.clone());
        }

        let mut docs_with_terms = kept;
        for (_, _, file_docs) in parsed {
            docs_with_terms.extend(file_docs);
        }

        let mut docs = Vec::with_capacity(docs_with_terms.len());
        let mut postings: HashMap<String, Vec<Posting>> = HashMap::new();
        let mut total_length: u64 = 0;
        for (doc_index, (doc, terms)) in docs_with_terms.into_iter().enumerate() {
            total_length += doc.length as u64;
            for (term, tf) in terms {
                postings
                    .entry(term)
                    .or_default()
                    .push(Posting(doc_index as u32, tf));
            }
            docs.push(doc);
        }
        // Deterministic sidecar output: postings order is already stable for a
        // given document order, but sorting each list by doc makes it explicit.
        for list in postings.values_mut() {
            list.sort_by_key(|posting| posting.0);
        }

        let index = Self {
            version: LEXICAL_INDEX_VERSION,
            file_hashes: new_hashes,
            avg_length: if docs.is_empty() {
                0.0
            } else {
                total_length as f32 / docs.len() as f32
            },
            docs,
            postings,
        };
        index.save(workspace)?;
        report(1000);
        Ok(index)
    }

    /// Rewrite the sidecar in full.
    fn save(&self, workspace: &Path) -> Result<()> {
        let path = Self::cache_path(workspace);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Reconstruct per-document term frequencies from the postings so unchanged
    /// files can be carried over without re-tokenising them. Indexed by document
    /// position, because ids are not unique within a file.
    fn terms_by_index(&self) -> Vec<Vec<(String, u32)>> {
        let mut terms: Vec<Vec<(String, u32)>> = vec![Vec::new(); self.docs.len()];
        for (term, postings) in &self.postings {
            for posting in postings {
                if let Some(slot) = terms.get_mut(posting.0 as usize) {
                    slot.push((term.clone(), posting.1));
                }
            }
        }
        terms
    }

    /// Rank documents for `query` with BM25, giving verified exact identifier
    /// and quoted-phrase matches priority.
    pub fn rank(&self, workspace: &Path, query: &str) -> Vec<LexicalHit> {
        if self.docs.is_empty() {
            return Vec::new();
        }
        let parsed = parse_query(query);
        if parsed.terms.is_empty() {
            return Vec::new();
        }
        let scores = self.bm25(&parsed.terms);

        let mut exact: HashSet<usize> = HashSet::new();
        if let Some(identifier) = &parsed.identifier {
            let candidates = self.identifier_candidates(identifier, &scores);
            exact.extend(self.verify(workspace, &candidates, &[], Some(identifier)));
        }
        for phrase in &parsed.phrases {
            let candidates = self.phrase_candidates(phrase, &scores);
            exact.extend(self.verify(workspace, &candidates, std::slice::from_ref(phrase), None));
        }

        let mut hits = Vec::new();
        for (index, score) in scores.into_iter().enumerate() {
            let is_exact = exact.contains(&index);
            if score <= 0.0 && !is_exact {
                continue;
            }
            let doc = &self.docs[index];
            hits.push(LexicalHit {
                id: doc.id.clone(),
                symbol: doc.symbol.clone(),
                file: doc.file.clone(),
                start_line: doc.start_line,
                end_line: doc.end_line,
                score,
                exact: is_exact,
            });
        }
        hits.sort_by(|a, b| {
            b.exact
                .cmp(&a.exact)
                .then_with(|| b.score.total_cmp(&a.score))
                .then_with(|| a.id.cmp(&b.id))
        });
        hits
    }

    /// Lexical-only retrieval: rank, expand graph neighbours, and apply the
    /// token budget. Used when no embeddings server is available.
    pub fn search(
        &self,
        workspace: &Path,
        query: &str,
        graph: Option<&GraphStore>,
        top_k: usize,
        semantic_budget_tokens: usize,
    ) -> Vec<SearchHit> {
        let ranked = self.rank(workspace, query);
        if ranked.is_empty() {
            return Vec::new();
        }
        let id_to_index: HashMap<&str, usize> = self
            .docs
            .iter()
            .enumerate()
            .map(|(index, doc)| (doc.id.as_str(), index))
            .collect();

        let mut best: HashMap<String, SearchHit> = HashMap::new();
        let seed_count = top_k.saturating_mul(2).max(top_k);
        for hit in ranked.iter().take(seed_count) {
            insert_hit(&mut best, hit.to_search_hit(None));
        }

        if let Some(graph) = graph {
            for hit in ranked.iter().take(seed_count) {
                for result in graph.lookup(&hit.symbol) {
                    if result.node.id != hit.id {
                        continue;
                    }
                    let neighbours = result.callers.iter().chain(result.callees.iter());
                    for edge in neighbours {
                        if let Some(&index) = id_to_index.get(edge.node_id.as_str()) {
                            let doc = &self.docs[index];
                            insert_hit(
                                &mut best,
                                hit_from_doc(
                                    doc,
                                    hit.score * super::index::EXPANSION_DECAY,
                                    Some(hit.symbol.clone()),
                                    false,
                                ),
                            );
                        }
                    }
                }
            }
        }

        let mut hits: Vec<SearchHit> = best.into_values().collect();
        hits.sort_by(|a, b| {
            b.exact
                .cmp(&a.exact)
                .then_with(|| b.score.total_cmp(&a.score))
        });
        budget_cut(hits, top_k, semantic_budget_tokens)
    }

    /// BM25 score for every document. Returns a zero for documents containing
    /// none of the query terms.
    fn bm25(&self, query_terms: &[String]) -> Vec<f32> {
        let mut scores = vec![0.0f32; self.docs.len()];
        let n = self.docs.len() as f32;
        let avg_length = self.avg_length.max(1.0);
        let mut seen: HashSet<&str> = HashSet::new();

        for term in query_terms {
            if !seen.insert(term.as_str()) {
                continue;
            }
            let Some(postings) = self.postings.get(term) else {
                continue;
            };
            let df = postings.len() as f32;
            let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
            for posting in postings {
                let Some(doc) = self.docs.get(posting.0 as usize) else {
                    continue;
                };
                let tf = posting.1 as f32;
                let length = doc.length as f32;
                let denominator = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * length / avg_length);
                if denominator > 0.0 {
                    scores[posting.0 as usize] += idf * (tf * (BM25_K1 + 1.0)) / denominator;
                }
            }
        }
        scores
    }

    /// Candidate documents for an explicit identifier: its exact-token postings
    /// plus symbol-name matches. These are already narrow, so unlike phrase
    /// candidates they do not fall back to the broader BM25 ranking.
    fn identifier_candidates(&self, identifier: &str, scores: &[f32]) -> Vec<usize> {
        let tokens = unique(tokenize(identifier));
        let mut candidates: HashSet<usize> = HashSet::new();

        if let Some(full) = tokens.first()
            && let Some(postings) = self.postings.get(full)
        {
            candidates.extend(postings.iter().map(|posting| posting.0 as usize));
        }
        if candidates.is_empty() {
            candidates.extend(self.intersect_postings(&tokens[1..]));
        }
        for (index, doc) in self.docs.iter().enumerate() {
            if doc.symbol.eq_ignore_ascii_case(identifier) {
                candidates.insert(index);
            }
        }
        self.cap_candidates(candidates, scores)
    }

    /// Candidate documents for a quoted phrase: documents containing every
    /// phrase term, capped to the strongest BM25 candidates.
    fn phrase_candidates(&self, phrase: &[String], scores: &[f32]) -> Vec<usize> {
        let candidates = self.intersect_postings(phrase);
        self.cap_candidates(candidates, scores)
    }

    fn intersect_postings(&self, terms: &[String]) -> HashSet<usize> {
        let mut unique_terms = unique(terms.to_vec());
        if unique_terms.is_empty() {
            return HashSet::new();
        }
        let Some(first) = self.postings.get(&unique_terms[0]) else {
            return HashSet::new();
        };
        let mut result: HashSet<usize> = first.iter().map(|posting| posting.0 as usize).collect();
        for term in unique_terms.drain(1..) {
            let Some(postings) = self.postings.get(&term) else {
                return HashSet::new();
            };
            let docs: HashSet<usize> = postings.iter().map(|posting| posting.0 as usize).collect();
            result.retain(|doc| docs.contains(doc));
            if result.is_empty() {
                break;
            }
        }
        result
    }

    fn cap_candidates(&self, candidates: HashSet<usize>, scores: &[f32]) -> Vec<usize> {
        let mut candidates: Vec<usize> = candidates.into_iter().collect();
        candidates.sort_by(|a, b| {
            scores[*b]
                .total_cmp(&scores[*a])
                .then_with(|| self.docs[*a].id.cmp(&self.docs[*b].id))
        });
        candidates.truncate(MAX_VERIFY_CANDIDATES);
        candidates
    }

    /// Verify candidate documents against the current source. Reads each file at
    /// most once per query and checks only the symbol's line range.
    fn verify(
        &self,
        workspace: &Path,
        candidates: &[usize],
        phrases: &[Vec<String>],
        identifier: Option<&str>,
    ) -> HashSet<usize> {
        let mut by_file: HashMap<&str, Vec<usize>> = HashMap::new();
        for &index in candidates {
            if let Some(doc) = self.docs.get(index) {
                by_file.entry(doc.file.as_str()).or_default().push(index);
            }
        }

        let mut verified = HashSet::new();
        for (file, indices) in by_file {
            let Ok(content) = std::fs::read_to_string(workspace.join(file)) else {
                continue;
            };
            let lines: Vec<&str> = content.lines().collect();
            for index in indices {
                let doc = &self.docs[index];
                let Some(slice) = slice_lines(&lines, doc.start_line, doc.end_line) else {
                    continue;
                };
                if let Some(identifier) = identifier {
                    let symbol_match = doc.symbol.eq_ignore_ascii_case(identifier);
                    let source_match = contains_identifier(&slice, identifier);
                    if symbol_match || source_match {
                        verified.insert(index);
                        continue;
                    }
                }
                let haystack = phrase_tokens(&slice);
                if phrases
                    .iter()
                    .any(|phrase| contains_sequence(&haystack, phrase))
                {
                    verified.insert(index);
                }
            }
        }
        verified
    }
}

/// One parsed query: BM25 terms, quoted phrases, and an explicit identifier.
struct ParsedQuery {
    terms: Vec<String>,
    phrases: Vec<Vec<String>>,
    identifier: Option<String>,
}

fn parse_query(query: &str) -> ParsedQuery {
    let mut phrases = Vec::new();
    let mut outside = String::new();
    let mut inside = String::new();
    let mut in_quote = false;

    for character in query.chars() {
        if character == '"' {
            if in_quote {
                let phrase = phrase_tokens(&inside);
                if !phrase.is_empty() {
                    phrases.push(phrase);
                }
                inside.clear();
            }
            in_quote = !in_quote;
        } else if in_quote {
            inside.push(character);
        } else {
            outside.push(character);
        }
    }
    // An unmatched quote is treated as a phrase rather than dropped.
    if in_quote {
        let phrase = phrase_tokens(&inside);
        if !phrase.is_empty() {
            phrases.push(phrase);
        }
    }

    let mut terms = tokenize(&outside);
    for phrase in &phrases {
        terms.extend(phrase.iter().cloned());
    }
    let terms = unique(terms);

    let trimmed = query.trim();
    let identifier = if phrases.is_empty() && looks_like_identifier(trimmed) {
        Some(trimmed.to_string())
    } else {
        None
    };

    ParsedQuery {
        terms,
        phrases,
        identifier,
    }
}

/// A single identifier-like query with no spaces: `MAX_CHUNK_CHARS`,
/// `GraphStore::lookup`, `src/embeddings/index.rs`.
fn looks_like_identifier(query: &str) -> bool {
    !query.is_empty()
        && !query.chars().any(char::is_whitespace)
        && !query.contains('"')
        && (query.contains('_')
            || query.contains("::")
            || query.contains('.')
            || query.contains('/')
            || (query.chars().any(|c| c.is_uppercase()) && query.chars().any(|c| c.is_lowercase())))
}

/// Tokenise text for phrase verification: only lowercase separator/camel-case
/// parts, so `graph_store` and `GraphStore` compare as the same two words.
fn phrase_tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for span in spans(text) {
        for segment in split_segments(&span) {
            tokens.extend(camel_parts(&segment));
        }
    }
    tokens
}

fn unique(tokens: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    tokens
        .into_iter()
        .filter(|token| seen.insert(token.clone()))
        .collect()
}

/// Whether `haystack` contains `needle` as a whole identifier-like token.
fn contains_identifier(haystack: &str, needle: &str) -> bool {
    let needle = needle.to_lowercase();
    spans(haystack)
        .iter()
        .any(|span| span.to_lowercase() == needle)
}

fn slice_lines(lines: &[&str], start: usize, end: usize) -> Option<String> {
    if start == 0 || end < start || start > lines.len() {
        return None;
    }
    let end = end.min(lines.len());
    Some(lines[start - 1..end].join("\n"))
}

fn contains_sequence(haystack: &[String], needle: &[String]) -> bool {
    if needle.is_empty() || needle.len() > haystack.len() {
        return false;
    }
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}

/// Term frequencies for one document, with symbol name and path weighted above
/// the body. The full body is indexed, not the embedding truncation.
fn doc_terms(symbol: &str, file: &str, body: &str) -> (Vec<(String, u32)>, u32) {
    let mut weighted = Vec::new();
    push_weighted(&mut weighted, &tokenize(symbol), NAME_WEIGHT);
    push_weighted(&mut weighted, &tokenize(file), PATH_WEIGHT);
    push_weighted(&mut weighted, &tokenize(body), 1);

    let length = weighted.len() as u32;
    let mut counts: HashMap<String, u32> = HashMap::new();
    for token in weighted {
        *counts.entry(token).or_default() += 1;
    }
    let mut terms: Vec<(String, u32)> = counts.into_iter().collect();
    terms.sort_by(|a, b| a.0.cmp(&b.0));
    (terms, length)
}

fn push_weighted(out: &mut Vec<String>, tokens: &[String], weight: usize) {
    for _ in 0..weight {
        out.extend(tokens.iter().cloned());
    }
}

/// Tokenise text for BM25: keep each original identifier-like span (so an exact
/// identifier is its own token) and add lowercase separator/camel-case parts.
fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for span in spans(text) {
        tokens.extend(span_tokens(&span));
    }
    tokens
}

fn spans(text: &str) -> Vec<String> {
    let mut spans = Vec::new();
    let mut current = String::new();
    for character in text.chars() {
        if character.is_alphanumeric()
            || matches!(character, '_' | ':' | '.' | '/' | '\\' | '$' | '-')
        {
            current.push(character);
        } else {
            push_span(&mut spans, &mut current);
        }
    }
    push_span(&mut spans, &mut current);
    spans
}

fn push_span(spans: &mut Vec<String>, current: &mut String) {
    let trimmed = current.trim_matches(|c| matches!(c, '.' | '/' | ':' | '-' | '\\'));
    if !trimmed.is_empty() {
        spans.push(trimmed.to_string());
    }
    current.clear();
}

fn span_tokens(span: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut seen = HashSet::new();
    push_unique(&mut tokens, &mut seen, span.to_lowercase());
    for segment in split_segments(span) {
        push_unique(&mut tokens, &mut seen, segment.to_lowercase());
        for part in camel_parts(&segment) {
            push_unique(&mut tokens, &mut seen, part);
        }
    }
    tokens
}

fn push_unique(tokens: &mut Vec<String>, seen: &mut HashSet<String>, token: String) {
    if !token.is_empty() && seen.insert(token.clone()) {
        tokens.push(token);
    }
}

fn split_segments(span: &str) -> Vec<String> {
    span.split(['_', ':', '.', '/', '\\', '-'])
        .filter(|segment| !segment.is_empty())
        .map(str::to_string)
        .collect()
}

/// Split `GraphStore` into `graph`, `store` and `HTTP2Server` into `http`, `2`,
/// `server`.
fn camel_parts(segment: &str) -> Vec<String> {
    let chars: Vec<char> = segment.chars().collect();
    if chars.is_empty() {
        return Vec::new();
    }
    let mut parts = Vec::new();
    let mut start = 0;
    for index in 1..chars.len() {
        let previous = chars[index - 1];
        let current = chars[index];
        let next = chars.get(index + 1);
        let boundary = (previous.is_lowercase() && current.is_uppercase())
            || (previous.is_uppercase()
                && current.is_uppercase()
                && next.is_some_and(|c| c.is_lowercase()))
            || (previous.is_alphabetic() && current.is_numeric())
            || (previous.is_numeric() && current.is_alphabetic());
        if boundary {
            parts.push(
                chars[start..index]
                    .iter()
                    .collect::<String>()
                    .to_lowercase(),
            );
            start = index;
        }
    }
    parts.push(chars[start..].iter().collect::<String>().to_lowercase());
    parts.retain(|part| !part.is_empty());
    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_keeps_full_identifier_and_parts() {
        let tokens = tokenize("EMBED_BATCH_TOKEN_BUDGET");
        assert!(tokens.contains(&"embed_batch_token_budget".to_string()));
        assert!(tokens.contains(&"embed".to_string()));
        assert!(tokens.contains(&"budget".to_string()));
    }

    #[test]
    fn tokenizer_splits_camel_case_and_paths() {
        let tokens = tokenize("GraphStore::lookup src/embeddings/index.rs");
        assert!(tokens.contains(&"graphstore::lookup".to_string()));
        assert!(tokens.contains(&"graph".to_string()));
        assert!(tokens.contains(&"store".to_string()));
        assert!(tokens.contains(&"embeddings".to_string()));
        assert!(tokens.contains(&"index".to_string()));
    }

    #[test]
    fn phrase_tokens_normalise_separators_and_case() {
        assert_eq!(
            phrase_tokens("graph_store mutex poisoned"),
            vec!["graph", "store", "mutex", "poisoned"]
        );
        assert_eq!(
            phrase_tokens("GraphStore mutex poisoned"),
            vec!["graph", "store", "mutex", "poisoned"]
        );
    }

    #[test]
    fn quoted_phrase_is_verified_against_source_and_prioritised() {
        let workspace = tempfile::tempdir().unwrap();
        std::fs::write(
            workspace.path().join("a.rs"),
            "fn first() {\n    log::error!(\"graph_store mutex poisoned\");\n}\n",
        )
        .unwrap();
        std::fs::write(
            workspace.path().join("b.rs"),
            "fn second() {\n    // graph store and mutex poisoned in prose\n}\n",
        )
        .unwrap();
        let index = LexicalIndex::build_or_update(workspace.path(), 1).unwrap();
        let ranked = index.rank(workspace.path(), "\"graph_store mutex poisoned\"");
        assert_eq!(ranked.first().unwrap().symbol, "first");
        assert!(ranked.first().unwrap().exact);
    }

    #[test]
    fn explicit_identifier_is_verified_and_prioritised() {
        let workspace = tempfile::tempdir().unwrap();
        std::fs::write(
            workspace.path().join("a.rs"),
            "fn lookup() {\n    store.lookup(\"x\");\n}\n\
             fn lookup_extra() {\n    store.lookup_extra(\"y\");\n}\n",
        )
        .unwrap();
        let index = LexicalIndex::build_or_update(workspace.path(), 1).unwrap();
        let ranked = index.rank(workspace.path(), "store.lookup");
        assert_eq!(ranked.first().unwrap().symbol, "lookup");
        assert!(ranked.first().unwrap().exact);
    }

    #[test]
    fn identifier_verification_uses_token_boundaries() {
        let workspace = tempfile::tempdir().unwrap();
        std::fs::write(
            workspace.path().join("a.rs"),
            "pub fn rrf_k() {}\npub fn rrf_k_extra() {}\n",
        )
        .unwrap();
        let index = LexicalIndex::build_or_update(workspace.path(), 1).unwrap();
        let ranked = index.rank(workspace.path(), "rrf_k");
        let exact: Vec<&str> = ranked
            .iter()
            .filter(|hit| hit.exact)
            .map(|hit| hit.symbol.as_str())
            .collect();
        assert_eq!(exact, vec!["rrf_k"]);
    }

    #[test]
    fn sidecar_round_trips_and_persists_no_source_text() {
        let workspace = tempfile::tempdir().unwrap();
        std::fs::write(
            workspace.path().join("a.rs"),
            "pub fn retry_backoff() {\n    let marker = \"secret_body_text\";\n}\n",
        )
        .unwrap();

        let built = LexicalIndex::build_or_update(workspace.path(), 1).unwrap();
        assert_eq!(built.len(), 1);
        assert_eq!(built.docs[0].symbol, "retry_backoff");
        assert!(built.postings.contains_key("retry_backoff"));

        let loaded = LexicalIndex::load(workspace.path());
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.docs[0].id, built.docs[0].id);

        let raw = std::fs::read_to_string(LexicalIndex::cache_path(workspace.path())).unwrap();
        // Postings store individual terms, never the source line or function body.
        assert!(!raw.contains("let marker"));
        assert!(!raw.contains("pub fn retry_backoff"));
    }

    #[test]
    fn version_mismatch_invalidates_the_sidecar() {
        let workspace = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(EmbeddingIndex::cache_dir(workspace.path())).unwrap();
        std::fs::write(
            LexicalIndex::cache_path(workspace.path()),
            r#"{"version":999,"file_hashes":{},"docs":[],"postings":{},"avg_length":0.0}"#,
        )
        .unwrap();
        assert!(LexicalIndex::load(workspace.path()).is_empty());
    }

    #[test]
    fn changed_and_removed_files_update_only_their_documents() {
        let workspace = tempfile::tempdir().unwrap();
        let first = workspace.path().join("a.rs");
        let second = workspace.path().join("b.rs");
        std::fs::write(&first, "pub fn one() {}\n").unwrap();
        std::fs::write(&second, "pub fn two() {}\n").unwrap();

        let built = LexicalIndex::build_or_update(workspace.path(), 1).unwrap();
        assert_eq!(built.len(), 2);

        std::fs::write(&first, "pub fn three() {}\n").unwrap();
        let updated = LexicalIndex::build_or_update(workspace.path(), 1).unwrap();
        let symbols: Vec<_> = updated.docs.iter().map(|doc| doc.symbol.as_str()).collect();
        assert!(symbols.contains(&"three"));
        assert!(symbols.contains(&"two"));
        assert!(!symbols.contains(&"one"));

        std::fs::remove_file(&second).unwrap();
        let updated = LexicalIndex::build_or_update(workspace.path(), 1).unwrap();
        assert_eq!(updated.len(), 1);
        assert_eq!(updated.docs[0].symbol, "three");
    }

    #[test]
    fn warm_rebuild_keeps_duplicate_id_documents_distinct() {
        let workspace = tempfile::tempdir().unwrap();
        // Two methods with the same name in one file share an id, but not a
        // body, so they must keep their own weights across a warm rebuild.
        std::fs::write(
            workspace.path().join("a.rs"),
            "impl A {\n    pub fn run(&self) {\n        one();\n    }\n}\n\n\
             impl B {\n    pub fn run(&self) {\n        two(); three(); four(); five(); six();\n    }\n}\n",
        )
        .unwrap();
        let cold = LexicalIndex::build_or_update(workspace.path(), 1).unwrap();
        let runs = cold.docs.iter().filter(|doc| doc.symbol == "run").count();
        assert_eq!(runs, 2, "both `run` methods should be indexed");
        let mut cold_lengths: Vec<u32> = cold.docs.iter().map(|doc| doc.length).collect();
        cold_lengths.sort_unstable();

        let warm = LexicalIndex::build_or_update(workspace.path(), 1).unwrap();
        let mut warm_lengths: Vec<u32> = warm.docs.iter().map(|doc| doc.length).collect();
        warm_lengths.sort_unstable();
        assert_eq!(cold_lengths, warm_lengths);
    }

    #[test]
    fn unchanged_file_reuses_cached_documents() {
        let workspace = tempfile::tempdir().unwrap();
        std::fs::write(workspace.path().join("a.rs"), "pub fn one() {}\n").unwrap();
        let built = LexicalIndex::build_or_update(workspace.path(), 1).unwrap();

        let updated = LexicalIndex::build_or_update(workspace.path(), 1).unwrap();
        assert_eq!(updated.len(), built.len());
        assert_eq!(updated.docs[0].id, built.docs[0].id);
        assert_eq!(updated.docs[0].length, built.docs[0].length);
    }

    #[test]
    fn lexical_index_covers_the_full_symbol_range() {
        let workspace = tempfile::tempdir().unwrap();
        let body = format!(
            "pub fn long() {{\n    let _ = \"{}\";\n}}\n",
            "word ".repeat(2_000)
        );
        std::fs::write(workspace.path().join("a.rs"), body).unwrap();

        let index = LexicalIndex::build_or_update(workspace.path(), 1).unwrap();
        assert_eq!(index.len(), 1);
        assert!(index.docs[0].length > 500);
        assert!(index.postings.contains_key("word"));
    }

    #[test]
    fn bm25_ranks_exact_identifier_above_unrelated_docs() {
        let workspace = tempfile::tempdir().unwrap();
        std::fs::write(
            workspace.path().join("a.rs"),
            "pub fn retry() {}\n\
             pub fn retry_backoff() {\n    retry();\n}\n\
             pub fn other() {}\n",
        )
        .unwrap();
        let index = LexicalIndex::build_or_update(workspace.path(), 1).unwrap();
        let ranked = index.rank(workspace.path(), "retry_backoff");
        assert_eq!(ranked.first().unwrap().symbol, "retry_backoff");
        assert!(!ranked.iter().any(|hit| hit.symbol == "other"));
    }

    #[test]
    fn lexical_search_returns_ranked_hits_without_vectors() {
        let workspace = tempfile::tempdir().unwrap();
        std::fs::write(
            workspace.path().join("a.rs"),
            "pub fn retry_backoff() {\n    sleep(1);\n}\n",
        )
        .unwrap();
        let index = LexicalIndex::build_or_update(workspace.path(), 1).unwrap();
        let hits = index.search(workspace.path(), "retry_backoff", None, 5, 16_384);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].symbol, "retry_backoff");
        assert!(hits[0].score > 0.0);
    }
}
