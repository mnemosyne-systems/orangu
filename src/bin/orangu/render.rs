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

use anyhow::{Context, Result, anyhow};
use markdown::{
    ParseOptions,
    mdast::{Image, Link, List, ListItem, Node},
    to_mdast,
};
use std::{collections::HashMap, fs, path::Path, sync::OnceLock};
use syntect::{
    easy::HighlightLines,
    highlighting::{Theme, ThemeSet},
    parsing::SyntaxSet,
    util::{LinesWithEndings, as_24_bit_terminal_escaped},
};

use super::commands::{LocalError, ShowFileOptions, shell_words};
use super::git::{discover_git_root, git_show_file_content};
use orangu::tools::{ToolExecutor, resolve_workspace_path};

pub const ANSI_BOLD_ON: &str = "\x1b[1m";
pub const ANSI_BOLD_OFF: &str = "\x1b[22m";
pub const ANSI_ITALIC_ON: &str = "\x1b[3m";
pub const ANSI_ITALIC_OFF: &str = "\x1b[23m";
pub const ANSI_STRIKETHROUGH_ON: &str = "\x1b[9m";
pub const ANSI_STRIKETHROUGH_OFF: &str = "\x1b[29m";
pub const ANSI_FG_CODE: &str = "\x1b[38;2;255;215;120m";
pub const ANSI_FG_LINK: &str = "\x1b[38;2;102;178;255m";
pub const ANSI_FG_LIGHT_GREEN: &str = "\x1b[38;2;170;255;170m";
pub const ANSI_FG_LIGHT_RED: &str = "\x1b[38;2;255;170;170m";
pub const ANSI_FG_SUBTLE: &str = "\x1b[38;2;180;190;205m";
pub const ANSI_FG_RESET: &str = "\x1b[39m";
pub const ANSI_RESET: &str = "\x1b[0m";

pub struct SyntaxHighlightAssets {
    pub syntaxes: SyntaxSet,
    pub dark_theme: Theme,
    pub light_theme: Theme,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GitLineMetadata {
    pub hash: String,
    pub author: String,
}

pub fn syntax_highlight_assets() -> &'static SyntaxHighlightAssets {
    static ASSETS: OnceLock<SyntaxHighlightAssets> = OnceLock::new();
    ASSETS.get_or_init(|| {
        let syntaxes = SyntaxSet::load_defaults_newlines();
        let themes = ThemeSet::load_defaults();
        let dark_theme = themes
            .themes
            .get("base16-ocean.dark")
            .cloned()
            .or_else(|| themes.themes.values().next().cloned())
            .unwrap_or_default();
        let light_theme = themes
            .themes
            .get("base16-ocean.light")
            .or_else(|| themes.themes.get("InspiredGitHub"))
            .cloned()
            .unwrap_or_default();
        SyntaxHighlightAssets {
            syntaxes,
            dark_theme,
            light_theme,
        }
    })
}

pub fn show_file_output(workspace: &Path, raw_args: &str, virtual_width: usize) -> Result<String> {
    let (path, options, rev) = parse_show_file_arguments(raw_args)?;
    let resolved_path = resolve_workspace_path(workspace, &path)?;

    if let Some(ref rev) = rev {
        let content = git_show_file_content(workspace, &resolved_path, rev)?;
        let blame = if options.show_hash || options.show_author {
            Some(git_blame_metadata_at_rev(
                workspace,
                &resolved_path,
                Some(rev),
            )?)
        } else {
            None
        };
        return render_show_file_content(&resolved_path, &content, blame.as_deref(), options);
    }

    if !options.show_hash
        && !options.show_author
        && let Some(output) = show_file_output_with_bat(&resolved_path, virtual_width)?
    {
        return Ok(output);
    }
    let content = fs::read_to_string(&resolved_path)
        .with_context(|| format!("failed to read {}", resolved_path.display()))?;
    let blame = if options.show_hash || options.show_author {
        Some(git_blame_metadata(workspace, &resolved_path)?)
    } else {
        None
    };

    render_show_file_content(&resolved_path, &content, blame.as_deref(), options)
}

pub fn show_file_output_with_bat(path: &Path, virtual_width: usize) -> Result<Option<String>> {
    let output = match std::process::Command::new("bat")
        .arg("--paging=never")
        .arg("--color=always")
        .arg("--style=numbers")
        .arg("--terminal-width")
        .arg(virtual_width.to_string())
        .arg(path)
        .output()
    {
        Ok(output) => output,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => {
            return Err(err).with_context(|| format!("failed to run bat for {}", path.display()));
        }
    };

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(anyhow!(
            "bat failed{}",
            if stderr.is_empty() {
                String::new()
            } else {
                format!(": {stderr}")
            }
        ));
    }

    String::from_utf8(output.stdout)
        .map(Some)
        .with_context(|| format!("bat output for {} was not UTF-8", path.display()))
}

pub fn parse_show_file_arguments(
    raw_args: &str,
) -> Result<(String, ShowFileOptions, Option<String>)> {
    let args = shell_words(raw_args)
        .map_err(|_| LocalError::Usage(show_file_usage_message().to_string()))?;
    let mut options = ShowFileOptions::default();
    let mut path = None;
    let mut rev = None;

    for arg in args {
        match arg.as_str() {
            "--hash" => options.show_hash = true,
            "--author" => options.show_author = true,
            _ if arg.starts_with('-') => {
                return Err(LocalError::Usage(format!(
                    "Unknown option '{arg}'. {}",
                    show_file_usage_message()
                ))
                .into());
            }
            _ if path.is_none() => path = Some(arg),
            _ if rev.is_none() => rev = Some(arg),
            _ => {
                return Err(LocalError::Usage(show_file_usage_message().to_string()).into());
            }
        }
    }

    let path = path.ok_or_else(|| LocalError::Usage(show_file_usage_message().to_string()))?;
    Ok((path, options, rev))
}

pub fn show_file_usage_message() -> &'static str {
    "Usage: /show_file [--hash] [--author] <path> [<ref>]. Use /help to see available commands."
}

pub fn render_show_file_content(
    path: &Path,
    content: &str,
    blame: Option<&[GitLineMetadata]>,
    options: ShowFileOptions,
) -> Result<String> {
    let assets = syntax_highlight_assets();
    let syntax = assets
        .syntaxes
        .find_syntax_for_file(path)
        .ok()
        .flatten()
        .unwrap_or_else(|| assets.syntaxes.find_syntax_plain_text());
    let theme = if orangu::tui::Theme::is_dark() {
        &assets.dark_theme
    } else {
        &assets.light_theme
    };
    let mut highlighter = HighlightLines::new(syntax, theme);
    let line_count = content.lines().count().max(1);
    let line_number_width = line_count.to_string().len();
    let mut rendered = Vec::new();

    if content.is_empty() {
        rendered.push(format_show_file_line(
            1,
            "",
            blame.and_then(|metadata| metadata.first()),
            options,
            line_number_width,
        ));
        return Ok(rendered.join("\n"));
    }

    for (index, line) in LinesWithEndings::from(content).enumerate() {
        let line_no = index + 1;
        let line_without_newline = line.trim_end_matches(['\r', '\n']);
        let highlighted = highlight_source_line(&mut highlighter, &assets.syntaxes, line)?;
        let highlighted = highlighted.trim_end_matches(['\r', '\n']);
        let rendered_line = if line_without_newline.is_empty() {
            String::new()
        } else {
            highlighted.to_string()
        };
        rendered.push(format_show_file_line(
            line_no,
            &rendered_line,
            blame.and_then(|metadata| metadata.get(index)),
            options,
            line_number_width,
        ));
    }

    Ok(rendered.join("\n"))
}

pub fn highlight_source_line(
    highlighter: &mut HighlightLines<'_>,
    syntaxes: &SyntaxSet,
    line: &str,
) -> Result<String> {
    let ranges = highlighter
        .highlight_line(line, syntaxes)
        .map_err(|err| anyhow!("failed to highlight source line: {err}"))?;
    Ok(as_24_bit_terminal_escaped(&ranges, false))
}

pub fn format_show_file_line(
    line_no: usize,
    line: &str,
    metadata: Option<&GitLineMetadata>,
    options: ShowFileOptions,
    line_number_width: usize,
) -> String {
    let mut parts = vec![format!("{line_no:>line_number_width$}")];
    if options.show_hash
        && let Some(metadata) = metadata
    {
        parts.push(metadata.hash.clone());
    }
    if options.show_author
        && let Some(metadata) = metadata
    {
        parts.push(metadata.author.clone());
    }
    if !line.is_empty() {
        parts.push(format!("{ANSI_RESET}{line}{ANSI_RESET}"));
    }
    parts.join(" ")
}

pub fn git_blame_metadata(workspace: &Path, path: &Path) -> Result<Vec<GitLineMetadata>> {
    git_blame_metadata_at_rev(workspace, path, None)
}

pub fn git_blame_metadata_at_rev(
    workspace: &Path,
    path: &Path,
    rev: Option<&str>,
) -> Result<Vec<GitLineMetadata>> {
    let repo_root = discover_git_root(workspace)
        .ok_or_else(|| anyhow!("Git blame metadata is only available inside a Git repository"))?;
    let relative_path = path
        .strip_prefix(&repo_root)
        .with_context(|| format!("{} is outside the Git repository", path.display()))?;
    let mut cmd = std::process::Command::new("git");
    cmd.arg("-C")
        .arg(&repo_root)
        .arg("blame")
        .arg("--line-porcelain")
        .arg("--abbrev=8");
    if let Some(rev) = rev {
        cmd.arg(rev);
    }
    let output = cmd
        .arg("--")
        .arg(relative_path)
        .output()
        .context("failed to run git blame")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(anyhow!(
            "git blame failed{}",
            if stderr.is_empty() {
                String::new()
            } else {
                format!(": {stderr}")
            }
        ));
    }

    let stdout = String::from_utf8(output.stdout).context("git blame output was not UTF-8")?;
    let mut metadata = Vec::new();
    let mut current_hash = String::new();
    let mut current_author = String::new();
    for line in stdout.lines() {
        if let Some(content) = line.strip_prefix('\t') {
            let _ = content;
            metadata.push(GitLineMetadata {
                hash: current_hash.clone(),
                author: current_author.clone(),
            });
            continue;
        }

        if let Some(author) = line.strip_prefix("author ") {
            current_author = author.to_string();
            continue;
        }

        let mut parts = line.split_whitespace();
        if let (Some(hash), Some(_orig), Some(_final)) = (parts.next(), parts.next(), parts.next())
            && hash.chars().all(|ch| ch.is_ascii_hexdigit())
            && hash.len() >= 8
        {
            current_hash = hash.chars().take(8).collect();
            current_author.clear();
        }
    }

    Ok(metadata)
}

pub enum MarkdownChunk {
    Text(String),
    Code {
        language: Option<String>,
        content: String,
    },
}

pub fn parse_markdown_chunks(text: &str) -> Vec<MarkdownChunk> {
    if text.is_empty() {
        return vec![];
    }

    match to_mdast(text, &ParseOptions::default()) {
        Ok(mut tree) => {
            resolve_reference_links(&mut tree);
            let mut chunks = Vec::new();
            if let Node::Root(root) = tree {
                for child in root.children {
                    match child {
                        Node::Code(code) => {
                            chunks.push(MarkdownChunk::Code {
                                language: code.lang.clone(),
                                content: code.value.clone(),
                            });
                        }
                        _ => {
                            let rendered = render_markdown_node(&child);
                            if !rendered.trim().is_empty() {
                                chunks.push(MarkdownChunk::Text(rendered));
                            }
                        }
                    }
                }
            } else {
                chunks.push(MarkdownChunk::Text(render_markdown_node(&tree)));
            }
            chunks
        }
        Err(_) => vec![MarkdownChunk::Text(text.to_string())],
    }
}

pub fn render_markdown_for_console(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }

    match to_mdast(text, &ParseOptions::default()) {
        Ok(mut tree) => {
            resolve_reference_links(&mut tree);
            render_markdown_node(&tree)
        }
        Err(_) => text.to_string(),
    }
}

/// Replace reference-style links and images (`[label][id]`, with a matching
/// `[id]: url` definition elsewhere in the document) by their inline
/// equivalents, so they render with their URLs instead of as bare labels.
fn resolve_reference_links(tree: &mut Node) {
    let mut definitions = HashMap::new();
    collect_link_definitions(tree, &mut definitions);
    if !definitions.is_empty() {
        replace_reference_nodes(tree, &definitions);
    }
}

fn collect_link_definitions(node: &Node, definitions: &mut HashMap<String, String>) {
    if let Node::Definition(definition) = node {
        definitions.insert(definition.identifier.clone(), definition.url.clone());
    }
    if let Some(children) = node.children() {
        for child in children {
            collect_link_definitions(child, definitions);
        }
    }
}

fn replace_reference_nodes(node: &mut Node, definitions: &HashMap<String, String>) {
    let Some(children) = node.children_mut() else {
        return;
    };
    for child in children {
        let replacement = match child {
            Node::LinkReference(reference) => definitions.get(&reference.identifier).map(|url| {
                Node::Link(Link {
                    children: std::mem::take(&mut reference.children),
                    position: None,
                    url: url.clone(),
                    title: None,
                })
            }),
            Node::ImageReference(reference) => definitions.get(&reference.identifier).map(|url| {
                Node::Image(Image {
                    position: None,
                    alt: std::mem::take(&mut reference.alt),
                    url: url.clone(),
                    title: None,
                })
            }),
            _ => None,
        };
        if let Some(replacement) = replacement {
            *child = replacement;
        }
        replace_reference_nodes(child, definitions);
    }
}

pub fn render_markdown_node(node: &Node) -> String {
    match node {
        Node::Root(root) => render_block_nodes(&root.children, false),
        Node::Paragraph(paragraph) => render_inline_nodes(&paragraph.children),
        Node::Heading(heading) => format!(
            "{ANSI_BOLD_ON}{} {}{ANSI_BOLD_OFF}",
            "#".repeat(heading.depth.into()),
            render_inline_nodes(&heading.children)
        ),
        Node::Blockquote(blockquote) => {
            prefix_lines(&render_block_nodes(&blockquote.children, false), "> ")
        }
        Node::List(list) => render_list(list),
        Node::ListItem(item) => render_list_item(item, "-", 2),
        Node::Code(code) => render_code_block(code.lang.as_deref(), &code.value),
        Node::ThematicBreak(_) => "-".repeat(40),
        Node::Table(table) => render_table(&table.children),
        Node::Definition(_) => String::new(),
        Node::Break(_) => "\n".to_string(),
        _ => render_inline_node(node),
    }
}

pub fn render_block_nodes(nodes: &[Node], compact: bool) -> String {
    let separator = if compact { "\n" } else { "\n\n" };
    nodes
        .iter()
        .map(render_markdown_node)
        .filter(|rendered| !rendered.trim().is_empty())
        .collect::<Vec<_>>()
        .join(separator)
}

pub fn render_inline_nodes(nodes: &[Node]) -> String {
    nodes.iter().map(render_inline_node).collect()
}

pub fn render_inline_node(node: &Node) -> String {
    match node {
        Node::Text(text) => text.value.clone(),
        Node::Strong(strong) => format!(
            "{ANSI_BOLD_ON}{}{ANSI_BOLD_OFF}",
            render_inline_nodes(&strong.children)
        ),
        Node::Emphasis(emphasis) => format!(
            "{ANSI_ITALIC_ON}{}{ANSI_ITALIC_OFF}",
            render_inline_nodes(&emphasis.children)
        ),
        Node::Delete(delete) => format!(
            "{ANSI_STRIKETHROUGH_ON}{}{ANSI_STRIKETHROUGH_OFF}",
            render_inline_nodes(&delete.children)
        ),
        Node::InlineCode(code) => {
            format!("{ANSI_FG_CODE}`{}{ANSI_FG_RESET}`", code.value)
        }
        Node::InlineMath(math) => {
            format!("{ANSI_FG_CODE}${}{ANSI_FG_RESET}$", math.value)
        }
        Node::Link(link) => render_link(&render_inline_nodes(&link.children), &link.url),
        Node::LinkReference(link) => render_inline_nodes(&link.children),
        Node::Image(image) => format!("[image: {}] ({})", image.alt, image.url),
        Node::ImageReference(image) => format!("[image: {}]", image.alt),
        Node::FootnoteReference(reference) => format!("[^{}]", reference.identifier),
        Node::Break(_) => "\n".to_string(),
        Node::Html(html) => html.value.clone(),
        Node::Math(math) => math.value.clone(),
        Node::MdxFlowExpression(expression) => expression.value.clone(),
        Node::MdxTextExpression(expression) => expression.value.clone(),
        Node::MdxjsEsm(esm) => esm.value.clone(),
        Node::Toml(toml) => toml.value.clone(),
        Node::Yaml(yaml) => yaml.value.clone(),
        _ => render_markdown_node(node),
    }
}

/// Render `label` as an OSC 8 terminal hyperlink to `url`: a supporting
/// terminal shows only `label`, made clickable, and the URL is not printed.
/// The label keeps the link colour and underline so it still reads as a link
/// where OSC 8 is unsupported (there the control sequence is ignored and the
/// styled label remains). An empty label falls back to showing the URL itself.
pub fn render_link(label: &str, url: &str) -> String {
    let shown = if label.is_empty() { url } else { label };
    format!("\x1b]8;;{url}\x1b\\{ANSI_FG_LINK}{shown}{ANSI_FG_RESET}\x1b]8;;\x1b\\")
}

pub fn render_list(list: &List) -> String {
    let start = list.start.unwrap_or(1);
    list.children
        .iter()
        .enumerate()
        .filter_map(|(index, child)| match child {
            Node::ListItem(item) => {
                let marker = if list.ordered {
                    format!("{}.", start + index as u32)
                } else {
                    "-".to_string()
                };
                Some(render_list_item(item, &marker, marker.len() + 1))
            }
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn render_list_item(item: &ListItem, marker: &str, indent: usize) -> String {
    let body = render_block_nodes(&item.children, !item.spread);
    indent_lines(&body, &format!("{marker} "), &" ".repeat(indent))
}

pub fn render_code_block(language: Option<&str>, value: &str) -> String {
    let mut lines = Vec::new();
    let opener = match language {
        Some(language) if !language.is_empty() => format!("```{language}"),
        _ => "```".to_string(),
    };
    lines.push(format!("{ANSI_FG_CODE}{opener}{ANSI_FG_RESET}"));
    if value.is_empty() {
        lines.push(String::new());
    } else {
        lines.extend(render_syntax_highlighted_code(language, value));
    }
    lines.push(format!("{ANSI_FG_CODE}```{ANSI_FG_RESET}"));
    lines.join("\n")
}

pub fn render_syntax_highlighted_code(language: Option<&str>, value: &str) -> Vec<String> {
    let language = language.and_then(|language| {
        let trimmed = language.trim();
        (!trimmed.is_empty()).then_some(trimmed)
    });
    let Some(language) = language else {
        return render_plain_code_lines(value);
    };

    let assets = syntax_highlight_assets();
    let Some(syntax) = assets
        .syntaxes
        .find_syntax_by_token(language)
        .or_else(|| assets.syntaxes.find_syntax_by_extension(language))
    else {
        return render_plain_code_lines(value);
    };

    let theme = if orangu::tui::Theme::is_dark() {
        &assets.dark_theme
    } else {
        &assets.light_theme
    };
    let mut highlighter = HighlightLines::new(syntax, theme);
    let mut rendered = Vec::new();
    for line in LinesWithEndings::from(value) {
        match highlighter.highlight_line(line, &assets.syntaxes) {
            Ok(ranges) => {
                let mut escaped = as_24_bit_terminal_escaped(&ranges, false);
                while escaped.ends_with('\n') {
                    escaped.pop();
                }
                rendered.push(escaped);
            }
            Err(_) => return render_plain_code_lines(value),
        }
    }
    if rendered.is_empty() {
        render_plain_code_lines(value)
    } else {
        rendered
    }
}

pub fn render_plain_code_lines(value: &str) -> Vec<String> {
    if value.is_empty() {
        return vec![String::new()];
    }

    value
        .lines()
        .map(|line| format!("{ANSI_FG_CODE}{line}{ANSI_FG_RESET}"))
        .collect()
}

pub fn render_table(rows: &[Node]) -> String {
    let rendered_rows = rows
        .iter()
        .filter_map(|row| match row {
            Node::TableRow(row) => Some(
                row.children
                    .iter()
                    .filter_map(|cell| match cell {
                        Node::TableCell(cell) => Some(render_inline_nodes(&cell.children)),
                        _ => None,
                    })
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        })
        .collect::<Vec<_>>();

    if rendered_rows.is_empty() {
        return String::new();
    }

    let mut lines = Vec::with_capacity(rendered_rows.len() + 1);
    for (index, row) in rendered_rows.iter().enumerate() {
        lines.push(format!("| {} |", row.join(" | ")));
        if index == 0 {
            lines.push(format!(
                "| {} |",
                row.iter().map(|_| "---").collect::<Vec<_>>().join(" | ")
            ));
        }
    }
    lines.join("\n")
}

pub fn prefix_lines(text: &str, prefix: &str) -> String {
    text.lines()
        .map(|line| format!("{prefix}{line}"))
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn indent_lines(text: &str, first_prefix: &str, rest_prefix: &str) -> String {
    text.lines()
        .enumerate()
        .map(|(index, line)| {
            let prefix = if index == 0 {
                first_prefix
            } else {
                rest_prefix
            };
            format!("{prefix}{line}")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn format_tools(tools: &ToolExecutor) -> String {
    tools
        .definitions()
        .into_iter()
        .map(|tool| format!("- {}: {}", tool.function.name, tool.function.description))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::process_env_lock;
    use tempfile::tempdir;

    struct EnvVarGuard {
        key: &'static str,
        original: Option<std::ffi::OsString>,
    }
    impl EnvVarGuard {
        fn set_value(key: &'static str, value: &str) -> Self {
            let original = std::env::var_os(key);
            unsafe {
                std::env::set_var(key, value);
            }
            Self { key, original }
        }
    }
    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            unsafe {
                match &self.original {
                    Some(value) => std::env::set_var(self.key, value),
                    None => std::env::remove_var(self.key),
                }
            }
        }
    }

    #[test]
    fn renders_markdown_emphasis_for_console() {
        let rendered = render_markdown_for_console("Hello **bold** and *italic*.");

        assert!(rendered.contains("\x1b[1mbold\x1b[22m"));
        assert!(rendered.contains("\x1b[3mitalic\x1b[23m"));
    }

    #[test]
    fn renders_markdown_blocks_for_console() {
        let rendered = render_markdown_for_console(
            "# Title\n\n- one\n- two\n\n`code`\n\n[docs](https://example.com)",
        );

        assert!(rendered.contains("\x1b[1m# Title\x1b[22m"));
        assert!(rendered.contains("- one"));
        assert!(rendered.contains("- two"));
        assert!(rendered.contains("\x1b[38;2;255;215;120m`code\x1b[39m`"));
        assert!(rendered.contains("docs"));
        assert!(rendered.contains("https://example.com"));
    }

    #[test]
    fn resolves_reference_links_to_their_definitions() {
        let rendered = render_markdown_for_console(
            "See [the repo][repo].\n\n[repo]: https://example.com/repo",
        );

        assert!(rendered.contains("the repo"));
        assert!(rendered.contains("https://example.com/repo"));
    }

    #[test]
    fn renders_fenced_code_blocks_with_syntax_highlighting() {
        let rendered = render_markdown_for_console("```c\nprintf(\"Hello World !\\\\n\");\n```");

        assert!(rendered.contains("```c"));
        assert!(rendered.contains("printf"));
        assert!(rendered.contains("\x1b["));
    }

    #[test]
    fn renders_unknown_fenced_code_blocks_with_plain_code_color() {
        let rendered = render_markdown_for_console("```unknownlang\nplain text\n```");

        assert!(rendered.contains("```unknownlang"));
        assert!(rendered.contains("\x1b[38;2;255;215;120mplain text\x1b[39m"));
    }

    #[test]
    fn show_file_formatting_bounds_ansi_to_source_column() {
        let metadata = GitLineMetadata {
            hash: "deadbeef".to_string(),
            author: "Alice".to_string(),
        };

        let rendered = format_show_file_line(
            7,
            "\x1b[38;2;1;2;3mlet x = 1;",
            Some(&metadata),
            ShowFileOptions {
                show_hash: true,
                show_author: true,
            },
            2,
        );

        assert_eq!(
            rendered,
            format!(" 7 deadbeef Alice {ANSI_RESET}\x1b[38;2;1;2;3mlet x = 1;{ANSI_RESET}")
        );
    }

    #[test]
    fn show_file_outputs_line_numbers_and_syntax_highlighting() {
        let _env_lock = process_env_lock().lock().unwrap_or_else(|p| p.into_inner());
        let workspace = tempdir().expect("workspace");
        std::fs::write(
            workspace.path().join("main.rs"),
            "fn main() {\n    println!(\"hello\");\n}\n",
        )
        .expect("source file");

        let _path_guard = EnvVarGuard::set_value("PATH", "");
        let output = show_file_output(workspace.path(), "main.rs", 512).expect("show file");
        assert!(output.contains("1 "));
        assert!(output.contains("2 "));
        assert!(output.contains("\u{1b}["));
        assert!(output.contains("println!"));
    }

    #[test]
    fn show_file_uses_bat_when_available_without_metadata_columns() {
        let _env_lock = process_env_lock().lock().unwrap_or_else(|p| p.into_inner());
        let workspace = tempdir().expect("workspace");
        std::fs::write(workspace.path().join("main.rs"), "fn main() {}\n").expect("source file");

        let tools_dir = tempdir().expect("tools dir");
        let bat = tools_dir.path().join("bat");
        std::fs::write(&bat, "#!/bin/sh\nprintf 'BAT:%s\\n' \"$*\"\n").expect("bat script");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut permissions = std::fs::metadata(&bat).expect("bat metadata").permissions();
            permissions.set_mode(0o755);
            std::fs::set_permissions(&bat, permissions).expect("bat permissions");
        }
        let path_value = format!(
            "{}:{}",
            tools_dir.path().display(),
            std::env::var("PATH").unwrap_or_default()
        );
        let _path_guard = EnvVarGuard::set_value("PATH", &path_value);
        let _columns_guard = EnvVarGuard::set_value("COLUMNS", "123");

        let output = show_file_output(workspace.path(), "main.rs", 512).expect("show file");
        assert!(output.contains("BAT:"));
        assert!(output.contains("--paging=never"));
        assert!(output.contains("--color=always"));
        assert!(output.contains("--style=numbers"));
        assert!(output.contains("--terminal-width"));
        assert!(output.contains(workspace.path().join("main.rs").to_string_lossy().as_ref()));
    }

    #[test]
    fn show_file_bypasses_bat_when_metadata_columns_are_requested() {
        let _env_lock = process_env_lock().lock().unwrap_or_else(|p| p.into_inner());
        let workspace = tempdir().expect("workspace");
        super::super::git::init_git_for_test(workspace.path());
        std::fs::write(workspace.path().join("README.md"), "alpha\nbeta\n").expect("write file");
        assert!(
            std::process::Command::new("git")
                .args(["add", "README.md"])
                .current_dir(workspace.path())
                .status()
                .expect("git add")
                .success()
        );
        assert!(
            std::process::Command::new("git")
                .args(["commit", "-m", "initial"])
                .current_dir(workspace.path())
                .status()
                .expect("git commit")
                .success()
        );

        let tools_dir = tempdir().expect("tools dir");
        let bat = tools_dir.path().join("bat");
        std::fs::write(&bat, "#!/bin/sh\nprintf 'BAT:%s\\n' \"$*\"\n").expect("bat script");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut permissions = std::fs::metadata(&bat).expect("bat metadata").permissions();
            permissions.set_mode(0o755);
            std::fs::set_permissions(&bat, permissions).expect("bat permissions");
        }
        let path_value = format!(
            "{}:{}",
            tools_dir.path().display(),
            std::env::var("PATH").unwrap_or_default()
        );
        let _path_guard = EnvVarGuard::set_value("PATH", &path_value);

        let output =
            show_file_output(workspace.path(), "--hash README.md", 512).expect("show file");
        assert!(!output.contains("BAT:"));
        assert!(output.contains("alpha"));
        assert!(output.contains("beta"));
    }

    #[test]
    fn show_file_can_include_git_hash_and_author() {
        let _env_lock = process_env_lock().lock().unwrap_or_else(|p| p.into_inner());
        let workspace = tempdir().expect("workspace");
        super::super::git::init_git_for_test(workspace.path());
        std::fs::write(workspace.path().join("README.md"), "alpha\nbeta\n").expect("write file");
        assert!(
            std::process::Command::new("git")
                .args(["add", "README.md"])
                .current_dir(workspace.path())
                .status()
                .expect("git add")
                .success()
        );
        assert!(
            std::process::Command::new("git")
                .args(["commit", "-m", "initial"])
                .current_dir(workspace.path())
                .status()
                .expect("git commit")
                .success()
        );

        let hash_output = std::process::Command::new("git")
            .args(["rev-parse", "--short=8", "HEAD"])
            .current_dir(workspace.path())
            .output()
            .expect("git rev-parse");
        let expected_hash = String::from_utf8(hash_output.stdout)
            .expect("hash output")
            .trim()
            .to_string();

        let output = show_file_output(workspace.path(), "--hash --author README.md", 512)
            .expect("show file");
        assert!(output.contains(&expected_hash));
        assert!(output.contains("Orangu Tests"));
        assert!(output.contains("1 "));
        assert!(output.contains("2 "));
    }

    use crate::commands::{LocalCommand, parse_local_command};

    #[test]
    fn parses_show_file_commands() {
        match parse_local_command("/show_file README.md") {
            Some(LocalCommand::ShowFile(args)) => assert_eq!(args.as_ref(), "README.md"),
            _ => panic!("expected show file slash command"),
        }

        let (path, options, rev) =
            parse_show_file_arguments("--hash --author \"docs/user guide.md\"")
                .expect("show file args");
        assert_eq!(path, "docs/user guide.md");
        assert!(options.show_hash);
        assert!(options.show_author);
        assert!(rev.is_none());
        let (path2, _, rev2) = parse_show_file_arguments("src/main.rs abc1234").expect("path+rev");
        assert_eq!(path2, "src/main.rs");
        assert_eq!(rev2.as_deref(), Some("abc1234"));
    }
}
