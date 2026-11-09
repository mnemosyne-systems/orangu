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

//! `@` mentions in a prompt: `@src/main.rs` attaches the file, `@src/` a
//! listing of the folder, and `@Symbol` the definition the Knowledge Graph
//! knows by that name. A mention that names none of these stays plain text.

use std::path::{Path, PathBuf};

use orangu::tools::ToolExecutor;

/// The most of one file attached; the rest is cut with a note.
pub(crate) const MAX_FILE_BYTES: usize = 256 * 1024;

/// The most entries a folder listing shows.
pub(crate) const MAX_LISTING_ENTRIES: usize = 400;

/// How deep a folder listing goes.
const LISTING_DEPTH: usize = 3;

/// The lines of a definition attached past where it starts, when the graph
/// does not say where it ends.
const DEFINITION_LINES: usize = 60;

/// One attached mention.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Attachment {
    /// What the summary line calls it: `src/main.rs`, `src/ (listing)`.
    pub label: String,
    /// The text added to the prompt.
    pub body: String,
}

impl Attachment {
    /// A rough token count: four bytes a token.
    pub fn tokens(&self) -> usize {
        self.body.len().div_ceil(4)
    }
}

/// The `@` mentions in `prompt`, in order and without repeats: words that
/// start with `@` at a word boundary, less trailing punctuation.
/// `@agent-<name>` belongs to the subagents and is left out.
pub(crate) fn mentions(prompt: &str) -> Vec<&str> {
    let mut found: Vec<&str> = Vec::new();
    for word in prompt.split_whitespace() {
        let Some(name) = word.strip_prefix('@') else {
            continue;
        };
        let name = name.trim_end_matches([',', '.', ';', ':', ')', '!', '?', '"', '\'']);
        if name.is_empty() || name.starts_with("agent-") || found.contains(&name) {
            continue;
        }
        found.push(name);
    }
    found
}

/// `prompt` with what its mentions name attached after it, and the
/// attachments. Unchanged when nothing is attached.
pub(crate) fn attach(
    prompt: &str,
    workspace: &Path,
    tools: &ToolExecutor,
) -> (String, Vec<Attachment>) {
    let attachments: Vec<Attachment> = mentions(prompt)
        .into_iter()
        .filter_map(|name| {
            path_attachment(name, workspace).or_else(|| symbol_attachment(name, workspace, tools))
        })
        .collect();
    if attachments.is_empty() {
        return (prompt.to_string(), attachments);
    }
    let bodies: Vec<&str> = attachments.iter().map(|a| a.body.as_str()).collect();
    (format!("{prompt}\n\n{}", bodies.join("\n\n")), attachments)
}

/// The output-window line naming what was attached:
/// `Attached src/main.rs (~1.2k tokens), src/ (listing, ~300 tokens)`.
pub(crate) fn summary(attachments: &[Attachment]) -> String {
    let items: Vec<String> = attachments
        .iter()
        .map(|attachment| {
            let tokens = format!(
                "~{} tokens",
                orangu::tui::context_gauge::short_count(attachment.tokens() as u64)
            );
            match attachment.label.strip_suffix(" (listing)") {
                Some(folder) => format!("{folder} (listing, {tokens})"),
                None => format!("{} ({tokens})", attachment.label),
            }
        })
        .collect();
    format!("Attached {}", items.join(", "))
}

/// A file or folder in the workspace named `name`.
fn path_attachment(name: &str, workspace: &Path) -> Option<Attachment> {
    let path = orangu::tools::resolve_workspace_path(workspace, name).ok()?;
    let relative = relative_to(&path, workspace);
    if path.is_dir() {
        let label = format!("{}/", relative.trim_end_matches('/'));
        let listing = listing(&path, workspace);
        return Some(Attachment {
            body: format!("<attached_folder path=\"{label}\">\n{listing}\n</attached_folder>"),
            label: format!("{label} (listing)"),
        });
    }
    let bytes = std::fs::read(&path).ok()?;
    if bytes.contains(&0) {
        return None;
    }
    let cut = bytes.len() > MAX_FILE_BYTES;
    let shown = &bytes[..bytes.len().min(MAX_FILE_BYTES)];
    let mut text = String::from_utf8_lossy(shown).into_owned();
    if cut {
        text.push_str(&format!(
            "\n[… cut at {} KiB of {} KiB; read the rest with show_file]",
            MAX_FILE_BYTES / 1024,
            bytes.len() / 1024
        ));
    }
    Some(Attachment {
        body: format!(
            "<attached_file path=\"{relative}\">\n{}\n</attached_file>",
            text.trim_end_matches('\n')
        ),
        label: relative,
    })
}

/// The files and folders under `dir`, one a line, skipping what Git ignores.
fn listing(dir: &Path, workspace: &Path) -> String {
    let mut entries: Vec<String> = ignore::WalkBuilder::new(dir)
        .max_depth(Some(LISTING_DEPTH))
        .hidden(false)
        .filter_entry(|entry| entry.file_name() != ".git")
        .build()
        .flatten()
        .filter(|entry| entry.depth() > 0)
        .map(|entry| {
            let rel = relative_to(entry.path(), workspace);
            if entry.file_type().is_some_and(|kind| kind.is_dir()) {
                format!("{rel}/")
            } else {
                rel
            }
        })
        .collect();
    entries.sort();
    let total = entries.len();
    entries.truncate(MAX_LISTING_ENTRIES);
    let mut text = entries.join("\n");
    if total > MAX_LISTING_ENTRIES {
        text.push_str(&format!(
            "\n[… {} more entries]",
            total - MAX_LISTING_ENTRIES
        ));
    }
    text
}

/// The definition the Knowledge Graph has for `name`, when the graph is
/// already built and knows that exact name.
fn symbol_attachment(name: &str, workspace: &Path, tools: &ToolExecutor) -> Option<Attachment> {
    if !name
        .chars()
        .all(|ch| ch.is_alphanumeric() || matches!(ch, '_' | ':'))
    {
        return None;
    }
    let node = {
        let guard = tools.graph_store.lock().ok()?;
        let store = guard.as_ref()?;
        store
            .lookup(name)
            .into_iter()
            .find(|result| result.node.label == name)?
            .node
    };
    let file = {
        let path = PathBuf::from(&node.source_file);
        if path.is_absolute() {
            path
        } else {
            workspace.join(path)
        }
    };
    let (start, end) = line_range(&node.source_location)?;
    let source = std::fs::read_to_string(&file).ok()?;
    let end = end.unwrap_or(start + DEFINITION_LINES - 1);
    let lines: Vec<&str> = source
        .lines()
        .skip(start.saturating_sub(1))
        .take(end.saturating_sub(start) + 1)
        .collect();
    if lines.is_empty() {
        return None;
    }
    let relative = relative_to(&file, workspace);
    let location = format!("{relative}:{start}-{}", start + lines.len() - 1);
    Some(Attachment {
        body: format!(
            "<attached_definition symbol=\"{name}\" path=\"{location}\">\n{}\n</attached_definition>",
            lines.join("\n")
        ),
        label: format!("{name} ({location})"),
    })
}

/// `L12-L40` as (12, Some(40)); `12` as (12, None).
fn line_range(location: &str) -> Option<(usize, Option<usize>)> {
    let mut parts = location.split('-');
    let start = parts.next()?.trim_start_matches('L').parse().ok()?;
    let end = parts
        .next()
        .and_then(|end| end.trim_start_matches('L').parse().ok());
    Some((start, end))
}

fn relative_to(path: &Path, workspace: &Path) -> String {
    path.strip_prefix(workspace)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn mentions_are_words_that_start_with_an_at_sign() {
        assert_eq!(
            mentions(
                "look at @src/main.rs, and @Cargo.toml. Mail me@example.com @agent-explore @src/main.rs"
            ),
            vec!["src/main.rs", "Cargo.toml"]
        );
        assert!(mentions("no mentions @ here").is_empty());
    }

    #[test]
    fn files_and_folders_are_attached_and_the_rest_stays_text() {
        let workspace = tempfile::tempdir().unwrap();
        fs::create_dir_all(workspace.path().join("src/inner")).unwrap();
        fs::write(workspace.path().join("src/a.rs"), "fn a() {}\n").unwrap();
        fs::write(workspace.path().join("src/inner/b.rs"), "fn b() {}\n").unwrap();
        fs::write(workspace.path().join("blob.bin"), [0u8, 1, 2]).unwrap();
        let tools = ToolExecutor::new(workspace.path());

        let (prompt, attached) = attach(
            "explain @src/a.rs and @src/ but not @missing.rs or @blob.bin or @../outside",
            workspace.path(),
            &tools,
        );
        assert_eq!(attached.len(), 2);
        assert_eq!(attached[0].label, "src/a.rs");
        assert_eq!(attached[1].label, "src/ (listing)");
        assert!(prompt.starts_with("explain @src/a.rs and @src/"));
        assert!(prompt.contains("<attached_file path=\"src/a.rs\">\nfn a() {}\n</attached_file>"));
        assert!(prompt.contains("src/inner/\nsrc/inner/b.rs"));
        assert_eq!(
            summary(&attached),
            format!(
                "Attached src/a.rs (~{} tokens), src/ (listing, ~{} tokens)",
                attached[0].tokens(),
                attached[1].tokens()
            )
        );

        let (unchanged, none) = attach("nothing @here", workspace.path(), &tools);
        assert_eq!(unchanged, "nothing @here");
        assert!(none.is_empty());
    }

    #[test]
    fn a_large_file_is_cut() {
        let workspace = tempfile::tempdir().unwrap();
        fs::write(
            workspace.path().join("big.txt"),
            "x".repeat(MAX_FILE_BYTES + 10),
        )
        .unwrap();
        let attachment = path_attachment("big.txt", workspace.path()).unwrap();
        assert!(
            attachment
                .body
                .contains("[… cut at 256 KiB of 256 KiB; read the rest with show_file]")
        );
    }

    #[test]
    fn a_symbol_attaches_its_definition_from_the_graph() {
        use orangu::graph::extract::ExtractedNode;
        use orangu::graph::store::GraphStore;
        let workspace = tempfile::tempdir().unwrap();
        fs::write(
            workspace.path().join("lib.rs"),
            "// header\npub fn retry_backoff() {\n    sleep();\n}\n",
        )
        .unwrap();
        let tools = ToolExecutor::new(workspace.path());
        let mut store = GraphStore::new();
        store.add_node(ExtractedNode {
            id: "lib::retry_backoff".to_string(),
            label: "retry_backoff".to_string(),
            source_file: "lib.rs".to_string(),
            source_location: "L2-L4".to_string(),
            kind: "function".to_string(),
        });
        *tools.graph_store.lock().unwrap() = Some(store);

        let (_, attached) = attach("why does @retry_backoff loop", workspace.path(), &tools);
        assert_eq!(attached.len(), 1);
        assert_eq!(attached[0].label, "retry_backoff (lib.rs:2-4)");
        assert!(
            attached[0]
                .body
                .contains("pub fn retry_backoff() {\n    sleep();\n}")
        );
    }
}
