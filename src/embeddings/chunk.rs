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
    /// The source text embedded for this chunk (symbol name + body).
    pub text: String,
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
/// and so a single chunk alone never exceeds a typical llama.cpp server's default
/// physical batch size (`-b`/`--batch-size 512` tokens). Characters, not tokens —
/// a coarse ~4 chars/token estimate, not an exact budget — chosen to leave
/// headroom under that default rather than requiring users to raise it.
const MAX_CHUNK_CHARS: usize = 1_600;

/// Extract embeddable chunks from one file's content. Returns an empty vector
/// for unsupported languages or files with no extractable symbols.
pub fn chunk_file(
    extractor: &GraphExtractor,
    path: &std::path::Path,
    rel_path: &str,
    content: &str,
) -> Vec<Chunk> {
    let Ok((nodes, _edges)) = extractor.extract_from_file(path, rel_path, content) else {
        return Vec::new();
    };

    nodes
        .into_iter()
        .filter_map(|node| {
            let (start, end) = parse_location(&node.source_location)?;
            let mut body = slice_lines(content, start, end);
            body.truncate(MAX_CHUNK_CHARS);
            // Prefix the symbol name so the embedding captures intent even when
            // the body is truncated or terse.
            let text = format!("{} in {}\n{}", node.label, node.source_file, body);
            Some(Chunk {
                id: node.id,
                symbol: node.label,
                file: node.source_file,
                start_line: start,
                end_line: end,
                text,
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
}
