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

//! Splitting a source file into embeddable chunks.
//!
//! Reuses the Tree-sitter symbol extraction that backs `/graph` and
//! `/duplicates`: each extracted definition (function, class, struct, …) becomes
//! one chunk, sliced from the file by the `L<start>-L<end>` line range the
//! extractor records. The chunk's `id` matches the graph node id
//! (`<file_stem>::<symbol>`) so semantic hits can be expanded along graph edges.

use crate::graph::extract::GraphExtractor;

/// One embeddable unit of code: a single top-level symbol.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// Graph node id (`<file_stem>::<symbol>`), matching the knowledge graph.
    pub id: String,
    /// Symbol name, for display.
    pub symbol: String,
    /// Workspace-relative source file path.
    pub file: String,
    /// 1-indexed inclusive start line.
    pub start_line: usize,
    /// 1-indexed inclusive end line.
    pub end_line: usize,
    /// The source text embedded for this chunk (symbol name + truncated body).
    pub text: String,
    /// The full, untruncated symbol body. Kept separate from [`Self::text`] so
    /// the lexical index can cover the whole extracted symbol range without
    /// changing what gets embedded.
    pub body: String,
}

/// Parse the `L<start>-L<end>` location string the extractor records into a
/// 1-indexed inclusive `(start, end)` line pair.
fn parse_location(location: &str) -> Option<(usize, usize)> {
    let (start, end) = location.split_once('-')?;
    let start = start.trim_start_matches('L').parse().ok()?;
    let end = end.trim_start_matches('L').parse().ok()?;
    Some((start, end))
}

/// Slice the 1-indexed inclusive line range `[start, end]` out of `content`.
fn slice_lines(content: &str, start: usize, end: usize) -> String {
    content
        .lines()
        .skip(start.saturating_sub(1))
        .take(end.saturating_sub(start).saturating_add(1))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Cap embedded chunk text so a single huge function cannot dominate a request,
/// and so a single chunk alone never exceeds a typical orangu-server's default
/// physical batch size (`-b`/`--batch-size 512` tokens). Characters, not tokens —
/// a coarse ~4 chars/token estimate, not an exact budget — chosen to leave
/// headroom under that default rather than requiring users to raise it.
const MAX_CHUNK_CHARS: usize = 1_600;

/// Truncate `text` to at most `limit` bytes without splitting a UTF-8 code
/// point. [`String::truncate`] panics when the boundary falls inside a
/// multi-byte character, which is easy to hit in files containing comments or
/// string literals outside ASCII.
fn truncate_on_char_boundary(text: &mut String, limit: usize) {
    if text.len() <= limit {
        return;
    }
    let mut cut = limit;
    while cut > 0 && !text.is_char_boundary(cut) {
        cut -= 1;
    }
    text.truncate(cut);
}

/// Extract embeddable chunks from one file's content. Returns an empty vector
/// for unsupported languages or files with no extractable symbols. Bodies are
/// truncated to [`MAX_CHUNK_CHARS`] for embedding; [`lexical_chunks`] keeps the
/// full extracted symbol range for lexical indexing.
pub fn chunk_file(
    extractor: &GraphExtractor,
    path: &std::path::Path,
    rel_path: &str,
    content: &str,
) -> Vec<Chunk> {
    chunks_with_body_limit(extractor, path, rel_path, content, Some(MAX_CHUNK_CHARS))
}

/// Extract chunks with the full symbol body, for the lexical index. The
/// embedding index intentionally keeps its own truncation, so the two caches
/// share extraction but not the indexed text boundary.
pub fn lexical_chunks(
    extractor: &GraphExtractor,
    path: &std::path::Path,
    rel_path: &str,
    content: &str,
) -> Vec<Chunk> {
    chunks_with_body_limit(extractor, path, rel_path, content, None)
}

fn chunks_with_body_limit(
    extractor: &GraphExtractor,
    path: &std::path::Path,
    rel_path: &str,
    content: &str,
    body_limit: Option<usize>,
) -> Vec<Chunk> {
    let Ok((nodes, _edges)) = extractor.extract_from_file(path, rel_path, content) else {
        return Vec::new();
    };

    nodes
        .into_iter()
        .filter_map(|node| {
            let (start, end) = parse_location(&node.source_location)?;
            let body = slice_lines(content, start, end);
            let mut embedded_body = body.clone();
            if let Some(limit) = body_limit {
                truncate_on_char_boundary(&mut embedded_body, limit);
            }
            // Prefix the symbol name so the embedding captures intent even when
            // the body is truncated or terse.
            let text = format!("{} in {}\n{}", node.label, node.source_file, embedded_body);
            Some(Chunk {
                id: node.id,
                symbol: node.label,
                file: node.source_file,
                start_line: start,
                end_line: end,
                text,
                body,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_location_reads_line_range() {
        assert_eq!(parse_location("L3-L10"), Some((3, 10)));
        assert_eq!(parse_location("L1-L1"), Some((1, 1)));
        assert_eq!(parse_location("garbage"), None);
    }

    #[test]
    fn slice_lines_extracts_inclusive_range() {
        let src = "a\nb\nc\nd\ne";
        assert_eq!(slice_lines(src, 2, 4), "b\nc\nd");
        assert_eq!(slice_lines(src, 1, 1), "a");
    }

    #[test]
    fn truncate_on_char_boundary_never_splits_a_code_point() {
        // "é" is two bytes, so a 1_600-byte cut can land inside it.
        let mut text = "a".repeat(1_599) + "é" + "tail";
        truncate_on_char_boundary(&mut text, 1_600);
        assert_eq!(text.len(), 1_599);
        assert!(text.is_char_boundary(text.len()));

        let mut short = "abc".to_string();
        truncate_on_char_boundary(&mut short, 1_600);
        assert_eq!(short, "abc");
    }
}
