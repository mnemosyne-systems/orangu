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
use super::index::EmbeddingIndex;
use super::source::{SourceFile, scan_workspace};
use crate::graph::extract::GraphExtractor;

/// Independent sidecar version. Bump when the tokenizer, postings format, or
/// BM25 fields change — never when the embedding format changes.
pub const LEXICAL_INDEX_VERSION: u32 = 1;

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
}
