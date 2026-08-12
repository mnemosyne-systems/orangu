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
//!
//! Mermaid and PlantUML blocks go the other way and are drawn here on the
//! server by [`super::mermaid`] and [`super::plantuml`]. A block that doesn't
//! parse falls back to the ordinary highlighted code block.

use super::{mermaid, plantuml};
use markdown::{
    ParseOptions,
    mdast::{Code, List, ListItem, Node},
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
    let renderer = Renderer {
        open_fence_start: unterminated_fence_start(text),
    };
    match to_mdast(text, &parse_options()) {
        Ok(tree) => renderer.render_node(&tree),
        Err(_) => format!("<p>{}</p>", escape_html(text)),
    }
}

/// Carries the one piece of whole-document context the walk needs: where a
/// still-unclosed fence begins, if the text ends inside one. Everything else
/// a node needs to render is local to it.
struct Renderer {
    /// Byte offset of the opening fence of a code block that never closes —
    /// which, mid-stream, is every code block the model is still typing.
    /// See [`unterminated_fence_start`].
    open_fence_start: Option<usize>,
}

/// Finds the opening fence of a trailing, still-unclosed code block.
///
/// This exists for streaming. The transcript is re-rendered on every token,
/// so a ```` ```mermaid ```` block is seen dozens of times while it fills in
/// one line at a time, and CommonMark says an unclosed fence runs to the end
/// of the document — the AST alone can't tell a half-typed diagram from a
/// finished one.
///
/// The failure this prevents is worse than it first looks. A half-written
/// diagram usually still *parses* — `flowchart TD` plus one edge is valid
/// Mermaid — so without this the reader watches a diagram redraw, reflow and
/// jump on every token until the fence closes, and each of those throwaway
/// states costs a full layout that the cache can never hit (the source is
/// different every time).
///
/// Returns the offset of the opening fence line's first backtick/tilde,
/// which matches the `position.start.offset` mdast gives that block.
fn unterminated_fence_start(text: &str) -> Option<usize> {
    // (offset of the opening fence, its fence character, its length)
    let mut open: Option<(usize, char, usize)> = None;
    let mut offset = 0;

    for line in text.split_inclusive('\n') {
        let content = line.trim_end_matches(['\n', '\r']);
        let indent = content.len() - content.trim_start_matches(' ').len();
        let rest = &content[indent..];
        // Four spaces or more makes it an indented code block, not a fence.
        if indent <= 3
            && let Some(fence) = rest.chars().next().filter(|c| *c == '`' || *c == '~')
        {
            let len = rest.chars().take_while(|c| *c == fence).count();
            if len >= 3 {
                match open {
                    // A closing fence uses the same character, is at least
                    // as long, and carries no info string.
                    Some((_, open_fence, open_len)) => {
                        if fence == open_fence && len >= open_len && rest[len..].trim().is_empty() {
                            open = None;
                        }
                    }
                    None => open = Some((offset + indent, fence, len)),
                }
            }
        }
        offset += line.len();
    }

    open.map(|(start, _, _)| start)
}

impl Renderer {
    fn render_node(&self, node: &Node) -> String {
        match node {
            Node::Root(root) => self.render_block_nodes(&root.children),
            Node::Paragraph(paragraph) => {
                format!("<p>{}</p>", self.render_inline_nodes(&paragraph.children))
            }
            Node::Heading(heading) => {
                let level = heading.depth.clamp(1, 6);
                format!(
                    "<h{level}>{}</h{level}>",
                    self.render_inline_nodes(&heading.children)
                )
            }
            Node::Blockquote(blockquote) => {
                format!(
                    "<blockquote>{}</blockquote>",
                    self.render_block_nodes(&blockquote.children)
                )
            }
            Node::List(list) => self.render_list(list),
            Node::ListItem(item) => self.render_list_item(item),
            Node::Code(code) => self.render_code(code),
            Node::ThematicBreak(_) => "<hr>".to_string(),
            Node::Table(table) => self.render_table(&table.children),
            Node::Definition(_) => String::new(),
            Node::Break(_) => "<br>".to_string(),
            _ => self.render_inline_node(node),
        }
    }

    fn render_block_nodes(&self, nodes: &[Node]) -> String {
        nodes.iter().map(|node| self.render_node(node)).collect()
    }

    fn render_inline_nodes(&self, nodes: &[Node]) -> String {
        nodes
            .iter()
            .map(|node| self.render_inline_node(node))
            .collect()
    }

    fn render_inline_node(&self, node: &Node) -> String {
        match node {
            Node::Text(text) => escape_html(&text.value),
            Node::Strong(strong) => {
                format!(
                    "<strong>{}</strong>",
                    self.render_inline_nodes(&strong.children)
                )
            }
            Node::Emphasis(emphasis) => {
                format!("<em>{}</em>", self.render_inline_nodes(&emphasis.children))
            }
            Node::Delete(delete) => {
                format!("<del>{}</del>", self.render_inline_nodes(&delete.children))
            }
            Node::InlineCode(code) => format!("<code>{}</code>", escape_html(&code.value)),
            Node::InlineMath(math) => render_math(&math.value, false),
            Node::Link(link) => format!(
                "<a href=\"{}\" target=\"_blank\" rel=\"noopener noreferrer\">{}</a>",
                escape_attr(&link.url),
                self.render_inline_nodes(&link.children)
            ),
            Node::LinkReference(link) => self.render_inline_nodes(&link.children),
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
            _ => self.render_node(node),
        }
    }

    fn render_list(&self, list: &List) -> String {
        let tag = if list.ordered { "ol" } else { "ul" };
        let items: String = list
            .children
            .iter()
            .filter_map(|child| match child {
                Node::ListItem(item) => Some(self.render_list_item(item)),
                _ => None,
            })
            .collect();
        format!("<{tag}>{items}</{tag}>")
    }

    fn render_list_item(&self, item: &ListItem) -> String {
        // A "tight" list item (no blank lines between items in the source) whose
        // content is a single paragraph renders that paragraph's inline content
        // directly, skipping the <p> wrapper — matches how CommonMark HTML
        // renderers distinguish tight from loose lists, instead of every
        // one-line item picking up a paragraph's extra vertical margin.
        if !item.spread
            && let [Node::Paragraph(paragraph)] = item.children.as_slice()
        {
            return format!("<li>{}</li>", self.render_inline_nodes(&paragraph.children));
        }
        format!("<li>{}</li>", self.render_block_nodes(&item.children))
    }

    /// Routes a fenced block to a diagram renderer or the syntax highlighter.
    ///
    /// A `mermaid` block is only offered to [`super::mermaid`] once its
    /// fence has closed — see [`unterminated_fence_start`] for why a
    /// still-streaming block must not be drawn even though it would often
    /// parse. Until then, and for anything that fails to render, it stays an
    /// ordinary code block.
    ///
    /// Two ways in. A `mermaid`/`mmd` tag is taken at its word. An
    /// **untagged** fence is checked against [`mermaid::looks_like_diagram`]
    /// first, because models do emit diagrams into a bare ```` ``` ````
    /// fence and a diagram served as a wall of code is the failure worth
    /// avoiding. A fence tagged anything else is left alone: an explicit
    /// `bash` or `json` is the model saying what it wrote, and second-
    /// guessing it is how a shell transcript ends up drawn as a flowchart.
    fn render_code(&self, code: &Code) -> String {
        let language = code
            .lang
            .as_deref()
            .map(str::trim)
            .filter(|l| !l.is_empty());
        let diagram_language = language.map(str::to_ascii_lowercase);

        if self.fence_is_closed(code) {
            match diagram_language.as_deref() {
                Some("mermaid" | "mmd") => {
                    if let Some(diagram) = mermaid::render(&code.value) {
                        return render_diagram(diagram, &code.value);
                    }
                }
                Some("plantuml" | "puml" | "pu") => {
                    if let Some(diagram) = plantuml::render(&code.value) {
                        return render_plantuml_diagram(&diagram, &code.value);
                    }
                }
                None if plantuml::looks_like_diagram(&code.value) => {
                    if let Some(diagram) = plantuml::render(&code.value) {
                        return render_plantuml_diagram(&diagram, &code.value);
                    }
                }
                None if mermaid::looks_like_diagram(&code.value) => {
                    if let Some(diagram) = mermaid::render(&code.value) {
                        return render_diagram(diagram, &code.value);
                    }
                }
                _ => {}
            }
        }

        render_code_block(language, &code.value)
    }

    /// True unless this block is the one the document ends inside of.
    ///
    /// A node with no position info is treated as closed: positions are
    /// always present for parsed input, and guessing "still streaming" for a
    /// block that isn't would suppress the diagram permanently rather than
    /// for one more token.
    fn fence_is_closed(&self, code: &Code) -> bool {
        let Some(open_fence_start) = self.open_fence_start else {
            return true;
        };
        code.position
            .as_ref()
            .is_none_or(|position| position.start.offset != open_fence_start)
    }

    fn render_table(&self, rows: &[Node]) -> String {
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
                    self.render_inline_nodes(&cell.children)
                ));
            }
            html.push_str("</tr>");
        }
        html.push_str("</table>");
        html
    }
}

/// Wraps a rendered diagram: one `<img>` per theme (`app.css` shows the one
/// matching the console's current theme — an `<img>` can't inherit the
/// page's CSS variables, so the choice has to be made between two finished
/// pictures) plus the source in a collapsed `<details>`.
///
/// The `<details>` is not decoration. Turning the code block into an image
/// would otherwise take away both the screen-reader-legible content and the
/// reader's only way to copy the diagram source back out.
fn render_diagram(diagram: &mermaid::Diagram, source: &str) -> String {
    // The diagram's own viewBox size. Not to force full size — `app.css`
    // scales it to the message — but so the browser knows the aspect ratio
    // and reserves the right box before the image decodes, instead of
    // reflowing the transcript when it lands.
    let size = if diagram.width > 0.0 {
        format!(
            " width=\"{:.0}\" height=\"{:.0}\"",
            diagram.width, diagram.height
        )
    } else {
        String::new()
    };
    format!(
        "<figure class=\"mermaid-diagram\">\
         <img class=\"mermaid-light\" src=\"{light}\" alt=\"Mermaid diagram\"{size}>\
         <img class=\"mermaid-dark\" src=\"{dark}\" alt=\"Mermaid diagram\"{size}>\
         <div class=\"diagram-actions\">{}{}</div>\
         <details class=\"mermaid-source\"><summary>Diagram source</summary>\
         <pre><code>{}</code></pre></details>\
         </figure>",
        download_link(&diagram.light, "diagram-dl-light"),
        download_link(&diagram.dark, "diagram-dl-dark"),
        escape_html(source),
        light = escape_attr(&diagram.light),
        dark = escape_attr(&diagram.dark),
    )
}

fn render_plantuml_diagram(diagram: &plantuml::Diagram, source: &str) -> String {
    let size = if diagram.width > 0.0 {
        format!(
            " width=\"{:.0}\" height=\"{:.0}\"",
            diagram.width, diagram.height
        )
    } else {
        String::new()
    };
    format!(
        "<figure class=\"plantuml-diagram\">\
         <img class=\"plantuml-light\" src=\"{light}\" alt=\"PlantUML diagram\"{size}>\
         <img class=\"plantuml-dark\" src=\"{dark}\" alt=\"PlantUML diagram\"{size}>\
         <div class=\"diagram-actions\">{}{}{}{}</div>\
         <details class=\"diagram-source\"><summary>Diagram source</summary>\
         <pre><code>{}</code></pre></details>\
         </figure>",
        format_download_link(&diagram.light, "diagram-dl-light", "svg"),
        format_download_link(&diagram.light_png, "diagram-dl-light", "png"),
        format_download_link(&diagram.dark, "diagram-dl-dark", "svg"),
        format_download_link(&diagram.dark_png, "diagram-dl-dark", "png"),
        escape_html(source),
        light = escape_attr(&diagram.light),
        dark = escape_attr(&diagram.dark),
    )
}

/// The download control on a diagram, mirroring the answer footer's
/// Save-as-Markdown button.
///
/// The picture in the transcript is scaled to the message, which for a large
/// diagram means well below its real resolution — this is how the full-size
/// original gets out. It's a plain anchor onto the same `data:` URI the
/// `<img>` already carries, so saving needs no JavaScript and no round trip,
/// and the file is the exact SVG being displayed.
///
/// One per theme, toggled by the same rules that pick the image, so the
/// saved file matches what's on screen rather than always being the light
/// version.
fn download_link(data_uri: &str, class: &str) -> String {
    format!(
        "<a class=\"diagram-dl {class}\" href=\"{}\" download=\"orangu-diagram.svg\" \
         title=\"Download SVG\" aria-label=\"Download diagram as SVG\">{SAVE_ICON}</a>",
        escape_attr(data_uri)
    )
}

fn format_download_link(data_uri: &str, class: &str, format: &str) -> String {
    let upper = format.to_ascii_uppercase();
    format!(
        "<a class=\"diagram-dl {class}\" href=\"{}\" download=\"orangu-plantuml.{format}\" \
         title=\"Download {upper}\" aria-label=\"Download diagram as {upper}\">{SAVE_ICON}<span>{upper}</span></a>",
        escape_attr(data_uri)
    )
}

/// The same glyph `app.js` uses for its own save buttons — kept identical so
/// a diagram's download reads as the same action as an answer's.
const SAVE_ICON: &str = "<svg viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" \
     stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\">\
     <path d=\"M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4\"/>\
     <polyline points=\"7 10 12 15 17 10\"/><line x1=\"12\" y1=\"15\" x2=\"12\" y2=\"3\"/></svg>";

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
    use base64::Engine as _;

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
    fn renders_a_closed_mermaid_block_as_a_diagram() {
        let html = render_markdown_to_html("```mermaid\nflowchart TD\n    A --> B\n```");
        assert!(html.contains("<figure class=\"mermaid-diagram\">"));
        assert!(html.contains("<img class=\"mermaid-light\" src=\"data:image/svg+xml;base64,"));
        assert!(html.contains("<img class=\"mermaid-dark\" src=\"data:image/svg+xml;base64,"));
        // The source survives as copyable, screen-reader-legible text.
        assert!(html.contains("flowchart TD"));
    }

    #[test]
    fn renders_plantuml_with_svg_and_png_downloads() {
        let html =
            render_markdown_to_html("```plantuml\n@startuml\nAlice -> Bob: hello\n@enduml\n```");
        assert!(
            html.contains("<figure class=\"plantuml-diagram\">"),
            "{html}"
        );
        assert!(html.contains("class=\"plantuml-light\" src=\"/api/diagrams/"));
        assert!(html.contains("download=\"orangu-plantuml.svg\""));
        assert!(html.contains("download=\"orangu-plantuml.png\""));
        assert!(html.contains("/light.png"));
        assert!(html.contains("Alice -&gt; Bob") || html.contains("Alice -&gt; Bob: hello"));
    }

    #[test]
    fn plantuml_fence_tags_are_case_insensitive() {
        let html =
            render_markdown_to_html("```PlantUML\n@startuml\nAlice -> Bob: hello\n@enduml\n```");
        assert!(
            html.contains("<figure class=\"plantuml-diagram\">"),
            "{html}"
        );
    }

    #[test]
    fn plantuml_streaming_and_fallback_match_mermaid_behaviour() {
        let open = render_markdown_to_html("```plantuml\n@startuml\nAlice -> Bob: hi\n@enduml");
        assert!(!open.contains("plantuml-diagram"), "{open}");
        let unsupported = render_markdown_to_html(
            "```plantuml\n@startuml\n!include https://example.com/theme.puml\nA -> B\n@enduml\n```",
        );
        assert!(!unsupported.contains("plantuml-diagram"), "{unsupported}");
        assert!(unsupported.contains("!include"));
    }

    #[test]
    fn leaves_a_still_streaming_mermaid_block_as_code() {
        // Exactly what arrives token by token: an opening fence with no
        // closing one yet. This fragment is *valid* Mermaid — that is the
        // point. Drawing it would show a diagram that reshapes on every
        // token, at a full layout each time, so the guard has to key on the
        // fence rather than on whether the source happens to parse.
        let html = render_markdown_to_html("```mermaid\nflowchart TD\n    A --> B");
        assert!(!html.contains("mermaid-diagram"), "{html}");
        assert!(html.contains("flowchart TD"));
    }

    #[test]
    fn renders_a_diagram_once_the_closing_fence_arrives() {
        // The same block one token later — the transition the test above
        // guards has to actually complete, or diagrams would never appear.
        let html = render_markdown_to_html("```mermaid\nflowchart TD\n    A --> B\n```\n");
        assert!(html.contains("mermaid-diagram"), "{html}");
    }

    #[test]
    fn keeps_an_earlier_diagram_while_a_later_block_streams() {
        // Only the trailing unclosed fence is held back; a diagram the model
        // already finished must not blink out while it types the next block.
        let html = render_markdown_to_html(
            "```mermaid\nflowchart TD\n    A --> B\n```\n\nthen\n\n```rust\nfn main() {",
        );
        assert!(html.contains("mermaid-diagram"), "{html}");
    }

    #[test]
    fn falls_back_to_a_code_block_when_the_diagram_does_not_parse() {
        // Models emit near-miss Mermaid often enough that this is the
        // difference between a readable reply and a blank frame.
        let html = render_markdown_to_html("```mermaid\nflowchart TD\n    A[[[--->>> ???\n```");
        assert!(!html.contains("mermaid-diagram"), "{html}");
        assert!(html.contains("A[[["), "{html}");
    }

    #[test]
    fn leaves_non_mermaid_code_blocks_alone() {
        let html = render_markdown_to_html("```rust\nfn main() {}\n```");
        assert!(!html.contains("mermaid-diagram"));
        assert!(html.contains("code-block"));
    }

    #[test]
    fn every_diagram_carries_a_full_resolution_download() {
        let html = render_markdown_to_html("```mermaid\nflowchart TD\n    A --> B\n```");
        // One control per theme, so the saved file matches the visible
        // image rather than always being the light one.
        assert!(html.contains("diagram-dl diagram-dl-light"), "{html}");
        assert!(html.contains("diagram-dl diagram-dl-dark"), "{html}");
        assert!(html.contains("download=\"orangu-diagram.svg\""), "{html}");

        // The link has to be the same data URI as the image it sits under —
        // the picture is scaled to the message, so this is the only route
        // to full resolution, and a broken href would fail silently.
        let href = html
            .split("<a class=\"diagram-dl diagram-dl-light\" href=\"")
            .nth(1)
            .and_then(|rest| rest.split('"').next())
            .expect("download href");
        assert!(html.contains(&format!("class=\"mermaid-light\" src=\"{href}\"")));

        // And it must decode to a whole SVG document, not a truncated one.
        let svg = String::from_utf8(
            base64::engine::general_purpose::STANDARD
                .decode(href.strip_prefix("data:image/svg+xml;base64,").unwrap())
                .expect("valid base64"),
        )
        .expect("utf-8");
        assert!(svg.starts_with("<svg"), "{svg}");
        assert!(svg.trim_end().ends_with("</svg>"));
    }

    #[test]
    fn diagram_images_declare_their_aspect_ratio() {
        // width/height come from the viewBox so the browser reserves the
        // right box before the image decodes; `app.css` scales it down from
        // there. Without them a large diagram reflows the transcript when
        // it lands.
        let html = render_markdown_to_html("```mermaid\nflowchart TD\n    A --> B\n```");
        assert!(
            html.contains(" width=\"") && html.contains(" height=\""),
            "{html}"
        );
    }

    #[test]
    fn renders_a_diagram_from_an_untagged_fence() {
        // Models don't always tag the fence; a diagram served as a wall of
        // code is the failure this avoids.
        let html = render_markdown_to_html("```\nflowchart TD\n    A --> B\n```");
        assert!(html.contains("mermaid-diagram"), "{html}");
    }

    #[test]
    fn leaves_an_untagged_fence_that_is_not_a_diagram_as_code() {
        for source in [
            "```\n$ npm install\n$ npm run build\n```",
            // merman renders this one; only the header gate stops it.
            "```\ngraph is a data structure of nodes and edges\n```",
            "```\ninfo: build succeeded\n```",
        ] {
            let html = render_markdown_to_html(source);
            assert!(!html.contains("mermaid-diagram"), "{source} -> {html}");
        }
    }

    #[test]
    fn respects_an_explicit_non_mermaid_tag() {
        // Tagged `bash` but holding valid Mermaid: the tag is the model
        // saying what it wrote, and overriding it is how a shell transcript
        // ends up drawn as a flowchart.
        let html = render_markdown_to_html("```bash\ngraph LR\n    A --> B\n```");
        assert!(!html.contains("mermaid-diagram"), "{html}");
    }

    #[test]
    fn finds_only_a_trailing_unterminated_fence() {
        assert_eq!(unterminated_fence_start("```rust\nfn main() {}\n```"), None);
        assert_eq!(
            unterminated_fence_start("text\n\n```rust\nfn main() {"),
            Some(6)
        );
        // A tilde fence isn't closed by a backtick one, and a longer fence
        // isn't closed by a shorter one.
        assert_eq!(unterminated_fence_start("~~~\nx\n```"), Some(0));
        assert_eq!(unterminated_fence_start("````\nx\n```"), Some(0));
        // A closing fence takes no info string.
        assert_eq!(unterminated_fence_start("```\nx\n``` js"), Some(0));
        assert_eq!(unterminated_fence_start("no code here"), None);
    }

    #[test]
    fn escapes_html_special_characters_inside_math_source() {
        let html = render_markdown_to_html(r"$a < b \& c > d$");
        assert!(html.contains("data-tex=\"a &lt; b \\&amp; c &gt; d\""));
        assert!(!html.contains("$a < b"));
    }
}
