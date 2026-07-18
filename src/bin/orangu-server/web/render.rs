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

//! Renders a chat message's markdown to HTML, with fenced code blocks
//! syntax-highlighted — the web UI's equivalent of `orangu`'s own
//! `render_markdown_for_console` (`src/bin/orangu/render.rs`), same
//! `to_mdast`-then-walk structure, targeting HTML instead of ANSI.
//!
//! Every text-ish node is HTML-escaped; the one exception is `Node::Html`
//! (a model emitting a literal `<tag>` in its markdown), which is escaped
//! too rather than passed through — chat output is untrusted content, and
//! this is a chat window, not a document renderer.
//!
//! Reference-style links/images (`[text][id]` with a separate `[id]: url`
//! definition) are not resolved — direct `[text](url)` links and fenced
//! code, which cover the overwhelming majority of real LLM markdown
//! output, are what this targets.
//!
//! `$...$`/`$$...$$` math is parsed (off by default in the `markdown`
//! crate's own GFM preset, turned on explicitly in [`parse_options`]) into
//! `<span class="katex-source" data-tex="...">`/`<div class="katex-source
//! katex-block" data-tex="...">` placeholders, holding the raw TeX
//! source both as a `data-tex` attribute and as escaped fallback text.
//! `app.js` finds these after inserting the HTML and calls `katex.render`
//! on each in place — actual typesetting is a client-side, JS-only step
//! (no server-side TeX engine exists in Rust); the escaped fallback text
//! is what stays visible if that JS step is ever skipped (KaTeX fails to
//! load, `render()` throws on malformed TeX, ...) rather than an empty
//! element.

use markdown::{
    ParseOptions,
    mdast::{List, ListItem, Node},
    to_mdast,
};
use std::sync::OnceLock;
use syntect::{
    highlighting::{Theme, ThemeSet},
    html::highlighted_html_for_string,
    parsing::SyntaxSet,
};

struct HighlightAssets {
    syntaxes: SyntaxSet,
    theme: Theme,
}

fn highlight_assets() -> &'static HighlightAssets {
    static ASSETS: OnceLock<HighlightAssets> = OnceLock::new();
    ASSETS.get_or_init(|| {
        let syntaxes = SyntaxSet::load_defaults_newlines();
        let themes = ThemeSet::load_defaults();
        let theme = themes
            .themes
            .get("base16-ocean.dark")
            .cloned()
            .or_else(|| themes.themes.values().next().cloned())
            .unwrap_or_default();
        HighlightAssets { syntaxes, theme }
    })
}

/// `ParseOptions::gfm()` plus math (`math_text`/`math_flow`), which GFM
/// itself leaves off — without these, `$...$`/`$$...$$` isn't recognized
/// as math at all and passes through as plain literal text (backslashes,
/// braces and all), which is what made LaTeX-heavy replies unreadable
/// before this.
fn parse_options() -> ParseOptions {
    let mut options = ParseOptions::gfm();
    options.constructs.math_text = true;
    options.constructs.math_flow = true;
    options
}

/// Renders `text` (a chat message's raw content) to an HTML fragment safe
/// to inject into the transcript. Falls back to escaped plain text wrapped
/// in a `<p>` if the markdown fails to parse.
pub fn render_markdown_to_html(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    match to_mdast(text, &parse_options()) {
        Ok(tree) => render_node(&tree),
        Err(_) => format!("<p>{}</p>", escape_html(text)),
    }
}

fn render_node(node: &Node) -> String {
    match node {
        Node::Root(root) => render_block_nodes(&root.children),
        Node::Paragraph(paragraph) => {
            format!("<p>{}</p>", render_inline_nodes(&paragraph.children))
        }
        Node::Heading(heading) => {
            let level = heading.depth.clamp(1, 6);
            format!(
                "<h{level}>{}</h{level}>",
                render_inline_nodes(&heading.children)
            )
        }
        Node::Blockquote(blockquote) => {
            format!(
                "<blockquote>{}</blockquote>",
                render_block_nodes(&blockquote.children)
            )
        }
        Node::List(list) => render_list(list),
        Node::ListItem(item) => render_list_item(item),
        Node::Code(code) => render_code_block(code.lang.as_deref(), &code.value),
        Node::ThematicBreak(_) => "<hr>".to_string(),
        Node::Table(table) => render_table(&table.children),
        Node::Definition(_) => String::new(),
        Node::Break(_) => "<br>".to_string(),
        _ => render_inline_node(node),
    }
}

fn render_block_nodes(nodes: &[Node]) -> String {
    nodes.iter().map(render_node).collect()
}

fn render_inline_nodes(nodes: &[Node]) -> String {
    nodes.iter().map(render_inline_node).collect()
}

fn render_inline_node(node: &Node) -> String {
    match node {
        Node::Text(text) => escape_html(&text.value),
        Node::Strong(strong) => {
            format!("<strong>{}</strong>", render_inline_nodes(&strong.children))
        }
        Node::Emphasis(emphasis) => format!("<em>{}</em>", render_inline_nodes(&emphasis.children)),
        Node::Delete(delete) => format!("<del>{}</del>", render_inline_nodes(&delete.children)),
        Node::InlineCode(code) => format!("<code>{}</code>", escape_html(&code.value)),
        Node::InlineMath(math) => render_math(&math.value, false),
        Node::Link(link) => format!(
            "<a href=\"{}\" target=\"_blank\" rel=\"noopener noreferrer\">{}</a>",
            escape_attr(&link.url),
            render_inline_nodes(&link.children)
        ),
        Node::LinkReference(link) => render_inline_nodes(&link.children),
        Node::Image(image) => format!(
            "<img src=\"{}\" alt=\"{}\">",
            escape_attr(&image.url),
            escape_attr(&image.alt)
        ),
        Node::ImageReference(image) => format!("[image: {}]", escape_html(&image.alt)),
        Node::FootnoteReference(reference) => {
            format!("[^{}]", escape_html(&reference.identifier))
        }
        Node::Break(_) => "<br>".to_string(),
        Node::Html(html) => escape_html(&html.value),
        Node::Math(math) => render_math(&math.value, true),
        _ => render_node(node),
    }
}

/// Shared by both math node kinds — `display` picks `$$...$$`'s block
/// `<div>` (KaTeX's centered, enlarged display mode) vs. `$...$`'s inline
/// `<span>`. `katex-block`, not `katex-display` — KaTeX's own generated
/// markup uses `katex-display` internally for display-mode output, and
/// `katex.render()` inserts that markup as a *child* of this element
/// rather than replacing it, so reusing the same class name here would
/// leave two different `.katex-display` elements nested inside each
/// other. See this module's own doc comment for what `app.js` does with
/// the `katex-source`/`data-tex` markers this produces.
fn render_math(tex: &str, display: bool) -> String {
    let tag = if display { "div" } else { "span" };
    let class = if display {
        "katex-source katex-block"
    } else {
        "katex-source"
    };
    format!(
        "<{tag} class=\"{class}\" data-tex=\"{}\">{}</{tag}>",
        escape_attr(tex),
        escape_html(tex)
    )
}

fn render_list(list: &List) -> String {
    let tag = if list.ordered { "ol" } else { "ul" };
    let items: String = list
        .children
        .iter()
        .filter_map(|child| match child {
            Node::ListItem(item) => Some(render_list_item(item)),
            _ => None,
        })
        .collect();
    format!("<{tag}>{items}</{tag}>")
}

fn render_list_item(item: &ListItem) -> String {
    // A "tight" list item (no blank lines between items in the source) whose
    // content is a single paragraph renders that paragraph's inline content
    // directly, skipping the <p> wrapper — matches how CommonMark HTML
    // renderers distinguish tight from loose lists, instead of every
    // one-line item picking up a paragraph's extra vertical margin.
    if !item.spread
        && let [Node::Paragraph(paragraph)] = item.children.as_slice()
    {
        return format!("<li>{}</li>", render_inline_nodes(&paragraph.children));
    }
    format!("<li>{}</li>", render_block_nodes(&item.children))
}

fn render_code_block(language: Option<&str>, value: &str) -> String {
    let language = language.and_then(|l| {
        let trimmed = l.trim();
        (!trimmed.is_empty()).then_some(trimmed)
    });

    let assets = highlight_assets();
    let syntax = language.and_then(|language| {
        assets
            .syntaxes
            .find_syntax_by_token(language)
            .or_else(|| assets.syntaxes.find_syntax_by_extension(language))
    });

    match syntax {
        Some(syntax) => {
            match highlighted_html_for_string(value, &assets.syntaxes, syntax, &assets.theme) {
                Ok(html) => format!("<div class=\"code-block\">{html}</div>"),
                Err(_) => format!("<pre><code>{}</code></pre>", escape_html(value)),
            }
        }
        None => format!("<pre><code>{}</code></pre>", escape_html(value)),
    }
}

fn render_table(rows: &[Node]) -> String {
    let mut html = String::from("<table>");
    for (index, row) in rows.iter().enumerate() {
        let Node::TableRow(row) = row else { continue };
        let cell_tag = if index == 0 { "th" } else { "td" };
        html.push_str("<tr>");
        for cell in &row.children {
            let Node::TableCell(cell) = cell else {
                continue;
            };
            html.push_str(&format!(
                "<{cell_tag}>{}</{cell_tag}>",
                render_inline_nodes(&cell.children)
            ));
        }
        html.push_str("</tr>");
    }
    html.push_str("</table>");
    html
}

fn escape_html(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn escape_attr(text: &str) -> String {
    escape_html(text).replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_emphasis_and_paragraphs() {
        let html = render_markdown_to_html("Hello **bold** and *italic*.");
        assert!(html.contains("<strong>bold</strong>"));
        assert!(html.contains("<em>italic</em>"));
        assert!(html.starts_with("<p>"));
    }

    #[test]
    fn renders_headings_lists_and_links() {
        let html =
            render_markdown_to_html("# Title\n\n- one\n- two\n\n[docs](https://example.com)");
        assert!(html.contains("<h1>Title</h1>"));
        assert!(html.contains("<li>one</li>"));
        assert!(html.contains("<li>two</li>"));
        assert!(html.contains("href=\"https://example.com\""));
        assert!(html.contains("target=\"_blank\""));
    }

    #[test]
    fn renders_multi_paragraph_list_items_with_paragraph_wrappers() {
        // A list item with more than one block (two paragraphs, blank line
        // between them) keeps each paragraph's <p> wrapper — only a tight
        // item whose sole content is one paragraph gets unwrapped.
        let html = render_markdown_to_html("- one\n\n  two");
        assert!(html.contains("<p>one</p>"));
        assert!(html.contains("<p>two</p>"));
    }

    #[test]
    fn renders_fenced_code_with_syntax_highlighting() {
        let html = render_markdown_to_html("```rust\nfn main() {}\n```");
        assert!(html.contains("code-block"));
        assert!(html.contains("fn"));
    }

    #[test]
    fn renders_unknown_language_code_as_plain_pre() {
        let html = render_markdown_to_html("```notalanguage\nplain text\n```");
        assert!(html.contains("<pre><code>"));
        assert!(html.contains("plain text"));
    }

    #[test]
    fn escapes_html_in_plain_text_to_prevent_injection() {
        let html = render_markdown_to_html("<script>alert(1)</script>");
        assert!(!html.contains("<script>"));
        assert!(html.contains("&lt;script&gt;"));
    }

    #[test]
    fn escapes_html_inside_inline_and_block_code() {
        let html = render_markdown_to_html("`<b>x</b>`");
        assert!(!html.contains("<code><b>"));
        assert!(html.contains("&lt;b&gt;"));
    }

    #[test]
    fn escapes_quotes_in_link_and_image_attributes() {
        let html = render_markdown_to_html("![alt\"](http://x/\"y)");
        assert!(!html.contains("\"y\""));
        assert!(html.contains("&quot;"));
    }

    #[test]
    fn renders_inline_math_as_a_katex_source_span() {
        let html = render_markdown_to_html(r"The set $A \to B$ is finite.");
        assert!(html.contains(r#"<span class="katex-source" data-tex="A \to B">"#));
        // Escaped fallback content too, so the raw TeX is at least legible
        // if the client-side katex.render() pass never runs.
        assert!(html.contains(r"A \to B</span>"));
        assert!(!html.contains("katex-block"));
    }

    #[test]
    fn renders_block_math_as_a_katex_block_div() {
        let html = render_markdown_to_html("$$\n\\sum_{i=0}^n i\n$$");
        assert!(html.contains(r#"<div class="katex-source katex-block" data-tex="#));
        assert!(html.contains(r"\sum_{i=0}^n i"));
    }

    #[test]
    fn escapes_html_special_characters_inside_math_source() {
        let html = render_markdown_to_html(r"$a < b \& c > d$");
        assert!(html.contains("data-tex=\"a &lt; b \\&amp; c &gt; d\""));
        assert!(!html.contains("$a < b"));
    }
}
