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

//! Review-aware context packing for `/create_patch`.
//!
//! Review reports are user-facing Markdown, so forwarding one verbatim wastes
//! prompt space on empty categories, approval prose, and generator metadata.
//! This module extracts only actionable findings, deduplicates them, gives the
//! initial prompt a hard token budget, and stores overflow as a bounded tree in
//! the existing reverse-compression cache. Each `expand_context` call therefore
//! reveals at most one small index or finding chunk instead of restoring the
//! entire original report at once.

use std::collections::{HashMap, HashSet};

use orangu::compression_cache::CompressionStore;
use tiktoken_rs::{CoreBPE, cl100k_base};

use crate::review::ReviewFeedbackRecord;

/// Review findings may consume at most this many tokens in the initial patch
/// prompt. Source code is deliberately excluded; the model reads current files
/// with tools after it chooses a finding to verify.
pub(crate) const INITIAL_REVIEW_TOKENS: usize = 2_048;

/// Maximum size of every reverse-compression leaf or index. Expansion remains
/// safe even when the original review is enormous.
pub(crate) const EXPANSION_NODE_TOKENS: usize = 768;

#[derive(Debug)]
pub(crate) struct PackedReviewContext {
    pub(crate) content: String,
    pub(crate) original_findings: usize,
    pub(crate) included_findings: usize,
    pub(crate) omitted_findings: usize,
}

#[derive(Debug)]
struct FindingGroup {
    label: String,
    priority: u8,
    findings: Vec<String>,
}

#[derive(Debug)]
struct CacheNode {
    id: String,
    summary: String,
}

/// Produce the compact review payload used by `/create_patch`.
pub(crate) fn pack_review_context(
    kind: &str,
    report: &str,
    feedback: &[ReviewFeedbackRecord],
    store: &CompressionStore,
) -> PackedReviewContext {
    let tokenizer = cl100k_base().ok();
    let mut groups = report_groups(kind, report);
    groups.extend(feedback_groups(feedback));
    deduplicate(&mut groups);
    groups.sort_by_key(|group| group.priority);

    let original_findings = groups.iter().map(|group| group.findings.len()).sum();
    let mut visible = String::new();
    let mut included_findings = 0;
    let mut overflow: Vec<(String, String)> = Vec::new();

    for group in groups {
        let mut heading_written = false;
        for finding in group.findings {
            let heading = if heading_written || visible.contains(&format!("## {}\n", group.label)) {
                String::new()
            } else {
                format!("## {}\n", group.label)
            };
            let candidate = format!("{heading}- {}\n", indent_continuations(&finding));
            if token_count(&format!("{visible}{candidate}"), tokenizer.as_ref())
                <= INITIAL_REVIEW_TOKENS
            {
                visible.push_str(&candidate);
                heading_written = true;
                included_findings += 1;
            } else {
                overflow.push((group.label.clone(), finding));
            }
        }
    }

    let omitted_findings = overflow.len();
    let mut content = format!(
        "Actionable review context ({included_findings}/{original_findings} findings included):"
    );
    if visible.is_empty() {
        content.push_str("\nNo concise finding fit in the initial review budget.");
    } else {
        content.push_str("\n\n");
        content.push_str(visible.trim_end());
    }

    if omitted_findings > 0 {
        match store_overflow_tree(&overflow, store, tokenizer.as_ref()) {
            Some(root) => content.push_str(&format!(
                "\n\n{omitted_findings} additional finding(s) were omitted to keep this prompt \
                 under {INITIAL_REVIEW_TOKENS} review tokens. They are stored in bounded \
                 reverse-compression nodes (each at most {EXPANSION_NODE_TOKENS} tokens). \
                 Expand only relevant paths/categories, starting with \
                 expand_context(id=\"{}\").",
                root.id
            )),
            None => content.push_str(&format!(
                "\n\n{omitted_findings} additional finding(s) were omitted because this run has \
                 no writable session compression cache."
            )),
        }
    }

    PackedReviewContext {
        content,
        original_findings,
        included_findings,
        omitted_findings,
    }
}

fn report_groups(kind: &str, report: &str) -> Vec<FindingGroup> {
    let mut grouped: HashMap<String, FindingGroup> = HashMap::new();
    let mut order = Vec::new();
    let mut section = "Overall".to_string();
    let mut current: Option<String> = None;

    let flush = |current: &mut Option<String>,
                 section: &str,
                 grouped: &mut HashMap<String, FindingGroup>,
                 order: &mut Vec<String>| {
        let Some(finding) = current.take() else {
            return;
        };
        let finding = finding.trim().to_string();
        if finding.is_empty() || boilerplate(&finding) {
            return;
        }
        let location = finding_location(&finding).unwrap_or("whole patch");
        let label = format!("{section} — {location}");
        let priority = if section == "Conclusion" || kind == "review" {
            0
        } else {
            1
        };
        if !grouped.contains_key(&label) {
            order.push(label.clone());
            grouped.insert(
                label.clone(),
                FindingGroup {
                    label: label.clone(),
                    priority,
                    findings: Vec::new(),
                },
            );
        }
        grouped
            .get_mut(&label)
            .expect("finding group inserted")
            .findings
            .push(finding);
    };

    for line in report.lines() {
        if let Some(heading) = line.trim().strip_prefix("## ") {
            flush(&mut current, &section, &mut grouped, &mut order);
            section = heading.trim().to_string();
            continue;
        }
        if let Some(item) = line.trim_start().strip_prefix("- ") {
            flush(&mut current, &section, &mut grouped, &mut order);
            current = Some(item.trim().to_string());
            continue;
        }
        if (line.starts_with("  ") || line.starts_with('\t'))
            && let Some(finding) = current.as_mut()
        {
            finding.push('\n');
            finding.push_str(line.trim());
            continue;
        }
        flush(&mut current, &section, &mut grouped, &mut order);
    }
    flush(&mut current, &section, &mut grouped, &mut order);

    order
        .into_iter()
        .filter_map(|label| grouped.remove(&label))
        .collect()
}

fn feedback_groups(feedback: &[ReviewFeedbackRecord]) -> Vec<FindingGroup> {
    feedback
        .iter()
        .filter(|record| !record.response.trim().is_empty())
        .map(|record| {
            let mut finding = String::new();
            if let Some(question) = record.question.as_deref().filter(|q| !q.trim().is_empty()) {
                finding.push_str("Question: ");
                finding.push_str(question.trim());
                finding.push('\n');
            }
            finding.push_str(record.response.trim());
            FindingGroup {
                label: format!("Interactive model feedback — {}", record.path),
                // User-curated line comments and explicit rejections lead;
                // popup advice follows them as supplemental context.
                priority: 1,
                findings: vec![finding],
            }
        })
        .collect()
}

fn boilerplate(finding: &str) -> bool {
    let plain = normalize(finding);
    plain.is_empty()
        || plain == "no issues found"
        || plain == "patch approved"
        || plain.contains("approves this patch")
        || plain.starts_with("generated by:")
}

fn finding_location(finding: &str) -> Option<&str> {
    let start = finding.find("**")? + 2;
    let end = finding[start..].find("**")? + start;
    let location = finding[start..end].trim();
    (!location.is_empty()).then_some(location)
}

fn deduplicate(groups: &mut Vec<FindingGroup>) {
    let mut seen = HashSet::new();
    for group in groups.iter_mut() {
        group
            .findings
            .retain(|finding| seen.insert(normalize(finding)));
    }
    groups.retain(|group| !group.findings.is_empty());
}

fn normalize(text: &str) -> String {
    text.chars()
        .map(|ch| match ch {
            '*' | '`' | '[' | ']' | '(' | ')' | '#' => ' ',
            _ => ch.to_ascii_lowercase(),
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn indent_continuations(finding: &str) -> String {
    finding.lines().collect::<Vec<_>>().join("\n  ")
}

fn store_overflow_tree(
    overflow: &[(String, String)],
    store: &CompressionStore,
    tokenizer: Option<&CoreBPE>,
) -> Option<CacheNode> {
    let mut by_label: Vec<(String, Vec<String>)> = Vec::new();
    for (label, finding) in overflow {
        match by_label.last_mut() {
            Some((last, findings)) if last == label => findings.push(finding.clone()),
            _ => by_label.push((label.clone(), vec![finding.clone()])),
        }
    }

    let mut nodes = Vec::new();
    for (label, findings) in by_label {
        let label = one_line_preview(&label, 80);
        let body = findings
            .iter()
            .map(|finding| format!("- {}", indent_continuations(finding)))
            .collect::<Vec<_>>()
            .join("\n");
        // Reserve suffix space before splitting, then verify against the exact
        // generated titles. This keeps the guarantee true even for unusually
        // token-dense Markdown headings and paths.
        let prefix = format!("## {label}\n");
        let mut body_budget = EXPANSION_NODE_TOKENS
            .saturating_sub(token_count(&prefix, tokenizer))
            .saturating_sub(32)
            .max(1);
        let parts = loop {
            let parts = split_to_budget(&body, body_budget, tokenizer);
            let part_count = parts.len();
            let largest = parts
                .iter()
                .enumerate()
                .map(|(index, part)| {
                    let title = part_title(&label, index, part_count);
                    token_count(&format!("## {title}\n{part}"), tokenizer)
                })
                .max()
                .unwrap_or_default();
            if largest <= EXPANSION_NODE_TOKENS {
                break parts;
            }
            body_budget = body_budget
                .saturating_sub(largest - EXPANSION_NODE_TOKENS + 1)
                .max(1);
        };
        let part_count = parts.len();
        for (index, part) in parts.into_iter().enumerate() {
            let title = part_title(&label, index, part_count);
            let content = format!("## {title}\n{part}");
            let id = store.store(&content)?;
            nodes.push(CacheNode { id, summary: title });
        }
    }

    build_index_tree(nodes, store, tokenizer)
}

fn build_index_tree(
    mut nodes: Vec<CacheNode>,
    store: &CompressionStore,
    tokenizer: Option<&CoreBPE>,
) -> Option<CacheNode> {
    if nodes.len() == 1 {
        return nodes.pop();
    }
    let mut depth = 1;
    const INDEX_PREAMBLE: &str = "Bounded review-context index. Expand only entries relevant to \
                                  the files or categories being fixed.\n";
    let index_budget = EXPANSION_NODE_TOKENS
        .saturating_sub(token_count(INDEX_PREAMBLE, tokenizer))
        .max(1);
    while nodes.len() > 1 {
        let entries = nodes
            .iter()
            .map(|node| {
                format!(
                    "- {}: expand_context(id=\"{}\")",
                    one_line_preview(&node.summary, 80),
                    node.id
                )
            })
            .collect::<Vec<_>>();
        let mut next = Vec::new();
        for (index, chunk) in pack_lines(&entries, index_budget, tokenizer)
            .into_iter()
            .enumerate()
        {
            let summary = format!("review overflow index level {depth}, part {}", index + 1);
            let content = format!("{INDEX_PREAMBLE}{chunk}");
            debug_assert!(token_count(&content, tokenizer) <= EXPANSION_NODE_TOKENS);
            let id = store.store(&content)?;
            next.push(CacheNode { id, summary });
        }
        nodes = next;
        depth += 1;
    }
    nodes.pop()
}

fn part_title(label: &str, index: usize, part_count: usize) -> String {
    if part_count == 1 {
        label.to_string()
    } else {
        format!("{label} (part {}/{part_count})", index + 1)
    }
}

fn pack_lines(lines: &[String], budget: usize, tokenizer: Option<&CoreBPE>) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    for line in lines {
        let candidate = if current.is_empty() {
            line.clone()
        } else {
            format!("{current}\n{line}")
        };
        if !current.is_empty() && token_count(&candidate, tokenizer) > budget {
            chunks.push(current);
            current = line.clone();
        } else {
            current = candidate;
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

fn split_to_budget(text: &str, budget: usize, tokenizer: Option<&CoreBPE>) -> Vec<String> {
    if token_count(text, tokenizer) <= budget {
        return vec![text.to_string()];
    }
    let mut remaining = text;
    let mut parts = Vec::new();
    while !remaining.is_empty() {
        let mut low = 1;
        let mut high = remaining.chars().count();
        let mut best = 1;
        while low <= high {
            let middle = low + (high - low) / 2;
            let end = char_boundary(remaining, middle);
            if token_count(&remaining[..end], tokenizer) <= budget {
                best = middle;
                low = middle + 1;
            } else {
                high = middle.saturating_sub(1);
            }
        }
        let end = char_boundary(remaining, best);
        parts.push(remaining[..end].to_string());
        remaining = &remaining[end..];
    }
    parts
}

fn char_boundary(text: &str, characters: usize) -> usize {
    text.char_indices()
        .nth(characters)
        .map(|(index, _)| index)
        .unwrap_or(text.len())
}

fn token_count(text: &str, tokenizer: Option<&CoreBPE>) -> usize {
    tokenizer
        .map(|tokenizer| tokenizer.encode_with_special_tokens(text).len())
        // Fail closed if the exact tokenizer cannot be constructed. BPE token
        // counts cannot exceed the UTF-8 byte count, so this remains bounded at
        // the cost of under-filling nodes.
        .unwrap_or_else(|| text.len())
}

fn one_line_preview(text: &str, max_chars: usize) -> String {
    let plain = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if plain.chars().count() <= max_chars {
        plain
    } else {
        format!("{}…", plain.chars().take(max_chars).collect::<String>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use tempfile::tempdir;

    fn store() -> (tempfile::TempDir, CompressionStore) {
        let dir = tempdir().expect("session");
        let store = CompressionStore::new(Some(dir.path().to_path_buf()));
        (dir, store)
    }

    #[test]
    fn packing_keeps_actionable_findings_and_drops_report_boilerplate() {
        let (_dir, store) = store();
        let report = "## Code\n\n- **src/a.rs:7**: handle the error\n- **src/a.rs:7**: handle the error\n\n## Security\n\nNo issues found\n\n## Conclusion\n\n**Patch rejected**\n\n- Rejected: **src/a.rs**\n\nGenerated by: **orangu**";
        let packed = pack_review_context("review", report, &[], &store);

        assert_eq!(packed.original_findings, 2);
        assert_eq!(packed.included_findings, 2);
        assert_eq!(packed.omitted_findings, 0);
        assert!(packed.content.contains("handle the error"));
        assert!(packed.content.contains("Rejected:"));
        assert!(!packed.content.contains("No issues found"));
        assert!(!packed.content.contains("Generated by"));
        assert_eq!(packed.content.matches("handle the error").count(), 1);
    }

    #[test]
    fn interactive_feedback_is_available_without_entering_the_public_report() {
        let (_dir, store) = store();
        let feedback = [ReviewFeedbackRecord {
            path: "src/a.rs".to_string(),
            question: Some("Is this thread safe?".to_string()),
            response: "The shared counter needs a lock.".to_string(),
        }];
        let packed = pack_review_context("review", "## Code\n\nNo issues found", &feedback, &store);

        assert!(
            packed
                .content
                .contains("Interactive model feedback — src/a.rs")
        );
        assert!(packed.content.contains("Is this thread safe?"));
        assert!(packed.content.contains("needs a lock"));
    }

    #[test]
    fn overflow_uses_bounded_reverse_compression_nodes() {
        let (_dir, store) = store();
        let mut report = String::from("## Code\n\n");
        for index in 0..300 {
            report.push_str(&format!(
                "- **src/file{}.rs:{}**: {}\n",
                index % 12,
                index + 1,
                "verify this detailed behavior ".repeat(12)
            ));
        }
        let packed = pack_review_context("auto_review", &report, &[], &store);
        assert!(packed.omitted_findings > 0);
        assert!(token_count(&packed.content, cl100k_base().ok().as_ref()) < 2_300);

        let root_id = packed
            .content
            .split("expand_context(id=\"")
            .nth(1)
            .and_then(|rest| rest.split('\"').next())
            .expect("root expansion id");
        let root = store.retrieve(root_id).expect("root node");
        let tokenizer = cl100k_base().ok();
        let mut pending = vec![(root_id.to_string(), root)];
        let mut visited = HashSet::new();
        while let Some((id, node)) = pending.pop() {
            if !visited.insert(id) {
                continue;
            }
            let tokens = token_count(&node, tokenizer.as_ref());
            assert!(
                tokens <= EXPANSION_NODE_TOKENS,
                "expanded node contained {tokens} tokens"
            );
            for child in node.split("expand_context(id=\"").skip(1) {
                let child_id = child.split('"').next().expect("child id");
                pending.push((
                    child_id.to_string(),
                    store.retrieve(child_id).expect("child node"),
                ));
            }
        }
    }
}
