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
//!
//! Every code block that stays a code block is wrapped as a *code window*
//! — the source under its licence header (`orangu::license`), highlighted,
//! over a footer naming the file it saves as and a `.code-dl` button. The
//! name is derived here (see [`Renderer::code_file_name`]); the saving is
//! `app.js`'s, which reads the `<pre>`'s text back out on click rather than
//! the server embedding a second, base64 copy of it in every streamed
//! frame. Since the licence is in that text already, what is saved is
//! exactly what the reader was looking at.

use super::{mermaid, plantuml};
use markdown::{
    ParseOptions,
    mdast::{Code, List, ListItem, Node},
    to_mdast,
};
use std::{cell::Cell, sync::OnceLock};
use syntect::{
    highlighting::{Theme, ThemeSet},
    html::highlighted_html_for_string,
    parsing::{SyntaxReference, SyntaxSet},
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
pub fn render_markdown_to_html(
    text: &str,
    project_licence: Option<&orangu::license::Project>,
) -> String {
    if text.is_empty() {
        return String::new();
    }
    let renderer = Renderer {
        open_fence_start: unterminated_fence_start(text),
        code_blocks_seen: Cell::new(0),
        project_licence,
    };
    match to_mdast(text, &parse_options()) {
        Ok(tree) => renderer.render_node(&tree),
        Err(_) => format!("<p>{}</p>", escape_html(text)),
    }
}

/// Carries the one piece of whole-document context the walk needs: where a
/// still-unclosed fence begins, if the text ends inside one. Everything else
/// a node needs to render is local to it.
struct Renderer<'a> {
    /// The licence of the workspace this server is rooted at, or `None` when
    /// it has none this program can write a header for. A downloaded code
    /// block is a file for *that* project, so it carries that project's
    /// licence or no licence at all — never a default one.
    project_licence: Option<&'a orangu::license::Project>,
    /// Byte offset of the opening fence of a code block that never closes —
    /// which, mid-stream, is every code block the model is still typing.
    /// See [`unterminated_fence_start`].
    open_fence_start: Option<usize>,
    /// How many code blocks have been rendered so far, which is what numbers
    /// the ones with no name of their own (`orangu-snippet-2.py`). Counted
    /// during the walk rather than passed down, because the walk is
    /// recursive and a code block can sit at any depth (inside a list item,
    /// a blockquote, ...) — a `Cell` keeps `render_node` taking `&self`.
    code_blocks_seen: Cell<usize>,
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

impl Renderer<'_> {
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

        self.render_code_block(code, language)
    }

    /// A fenced block rendered as a *code window*: the licensed source,
    /// highlighted, over a footer naming the file it saves as and the button
    /// that saves it.
    ///
    /// The licence header (`orangu::license`) is put on here, in the block
    /// the reader sees, rather than added on the way out of the download
    /// button. It is part of the code as far as the console is concerned:
    /// visible, highlighted, selectable, and in the text the download reads
    /// back out of the `<pre>` — so what gets saved is exactly what was on
    /// screen. Only the rendering is affected; the message stored in the
    /// session and replayed as context stays what the model actually wrote.
    ///
    /// The footer sits under the block on the right, dimmed, the same shape
    /// and place the answer footer's own save control has — a download in
    /// this console is a small icon at the lower right of the thing it
    /// saves, and a code block is no exception.
    ///
    /// The text is already on screen, so this is purely about getting it
    /// back out. Selecting it by hand is where that goes wrong — a code
    /// block scrolls in both directions (`.message pre` in `app.css`), so a
    /// drag over a long listing selects a window of it, silently. The button
    /// hands over the block verbatim.
    ///
    /// The markup is a header plus the body; `app.js` reads the text back
    /// out of the `<pre>` on click rather than the server embedding a second
    /// copy of it as a `data:` URI. The transcript's HTML is re-rendered and
    /// re-sent on *every streamed token*, and a `data:` URI would put a
    /// base64 copy of every code block in the reply into each of those
    /// frames.
    fn render_code_block(&self, code: &Code, language: Option<&str>) -> String {
        let assets = highlight_assets();
        let syntax = language.and_then(|language| {
            assets
                .syntaxes
                .find_syntax_by_token(language)
                .or_else(|| assets.syntaxes.find_syntax_by_extension(language))
        });

        // The name has to be read from what the model wrote, before the
        // licence goes on: `file_name_from_first_line` looks at line one,
        // and line one is about to become `// MIT License`.
        let name = self.code_file_name(code, language, syntax);
        let source = orangu::license::apply(&code.value, &name, self.project_licence);

        let plain = || format!("<pre><code>{}</code></pre>", escape_html(&source));
        let body = match syntax {
            Some(syntax) => {
                highlighted_html_for_string(&source, &assets.syntaxes, syntax, &assets.theme)
                    .unwrap_or_else(|_| plain())
            }
            None => plain(),
        };

        let (text, attr) = (escape_html(&name), escape_attr(&name));
        format!(
            "<div class=\"code-block\">{body}\
             <div class=\"code-footer\"><span class=\"code-name\">{text}</span>\
             <button type=\"button\" class=\"code-dl\" data-file-name=\"{attr}\" \
             title=\"Download {attr}\" aria-label=\"Download code as {attr}\">{SAVE_ICON}</button>\
             </div></div>"
        )
    }

    /// The name a code block downloads as.
    ///
    /// Preference order is how sure the name is: what the model wrote on the
    /// fence, then what it wrote in the block's first-line comment, then a
    /// generated `orangu-snippet-<n>.<ext>`. The counter advances for every
    /// block, named or not, so the number is the block's position in the
    /// reply — two blocks never race for the same generated name, and a
    /// stable name means re-rendering mid-stream doesn't renumber the ones
    /// already on screen.
    fn code_file_name(
        &self,
        code: &Code,
        language: Option<&str>,
        syntax: Option<&SyntaxReference>,
    ) -> String {
        let index = self.code_blocks_seen.get() + 1;
        self.code_blocks_seen.set(index);
        declared_file_name(language, code.meta.as_deref())
            .or_else(|| file_name_from_first_line(&code.value))
            .unwrap_or_else(|| {
                format!(
                    "orangu-snippet-{index}.{}",
                    default_extension(language, syntax)
                )
            })
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
/// a diagram's download, a code block's, and an answer's all read as the
/// same action.
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

/// Extension-less file names common enough to be worth recognising — the
/// only names accepted without a `.ext`, so that a bare `rust` or `bash`
/// info string stays a language rather than becoming a file.
const BARE_FILE_NAMES: [&str; 8] = [
    "Makefile",
    "Dockerfile",
    "Rakefile",
    "Gemfile",
    "Justfile",
    "Vagrantfile",
    "Jenkinsfile",
    "Procfile",
];

/// A file name the model spelled out on the fence itself.
///
/// Three shapes are in circulation and all three are read here:
/// ```` ```rust src/main.rs ````, ```` ```rust:src/main.rs ```` and
/// ```` ```rust title="src/main.rs" ````. The info string's first word is
/// what mdast calls `lang`; whatever follows it is `meta`.
fn declared_file_name(language: Option<&str>, meta: Option<&str>) -> Option<String> {
    if let Some(language) = language {
        if let Some((_, rest)) = language.split_once(':')
            && let Some(name) = clean_file_name(rest)
        {
            return Some(name);
        }
        // A fence tagged with the file name and nothing else — ```Makefile,
        // ```main.rs. A plain ```rust has no dot and isn't a known bare
        // name, so it falls through.
        if let Some(name) = clean_file_name(language) {
            return Some(name);
        }
    }

    meta?.split_whitespace().find_map(|token| {
        let value = token
            .split_once('=')
            .filter(|(key, _)| {
                matches!(
                    key.to_ascii_lowercase().as_str(),
                    "title" | "file" | "filename" | "name" | "path"
                )
            })
            .map_or(token, |(_, value)| value);
        clean_file_name(value)
    })
}

/// A file name in the block's own first line, which is how a model labels a
/// snippet when the fence carries only the language: `// src/main.rs`,
/// `# app/models.py`, `<!-- index.html -->`.
///
/// Only a comment whose body is a *single* token is considered. That guard
/// is what keeps `# Install the dependencies first` from naming a file, and
/// it's why an ordinary explanatory comment on line one costs nothing —
/// the block just falls back to a generated name.
fn file_name_from_first_line(value: &str) -> Option<String> {
    let line = value.lines().find(|line| !line.trim().is_empty())?.trim();
    // Longest marker first, so `#!` isn't read as `#` and `;;` not as `;`.
    let markers = [
        "<!--", "\"\"\"", "/*", "//", "--", "#!", ";;", "#", ";", "%",
    ];
    let body = markers
        .iter()
        .find_map(|marker| line.strip_prefix(marker))?
        .trim()
        .trim_end_matches("-->")
        .trim_end_matches("*/")
        .trim();

    // `// File: src/main.rs` is as common as the bare form.
    let body = ["filename:", "file:", "path:", "name:"]
        .iter()
        .find_map(|label| {
            body.get(..label.len())
                .filter(|head| head.eq_ignore_ascii_case(label))
                .map(|_| body[label.len()..].trim())
        })
        .unwrap_or(body);

    let mut tokens = body.split_whitespace();
    let token = tokens.next()?;
    if tokens.next().is_some() {
        return None;
    }
    clean_file_name(token.trim_end_matches([':', ',', ';', '.']))
}

/// Reduces a candidate to a bare, safe file name, or turns it down.
///
/// Only the last path component survives: the button saves one file, and a
/// `download` attribute containing a separator is ignored by the browser
/// anyway. Everything that isn't plainly a file name is rejected rather
/// than guessed at — a confidently wrong name on a saved file is worse
/// than a numbered one.
fn clean_file_name(raw: &str) -> Option<String> {
    let raw = raw.trim().trim_matches(['"', '\'', '`']);
    // A URL's last component can look exactly like a file name.
    if raw.is_empty() || raw.contains("://") {
        return None;
    }
    let name = raw.rsplit(['/', '\\']).next()?.trim();
    if name.is_empty()
        || name.len() > 64
        || !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | '+'))
    {
        return None;
    }
    if BARE_FILE_NAMES
        .iter()
        .any(|known| known.eq_ignore_ascii_case(name))
    {
        return Some(name.to_string());
    }

    let (stem, extension) = name.rsplit_once('.')?;
    // A dotfile (`.gitignore`) is the one shape with no stem. Requiring one
    // otherwise, and requiring the extension to start with a letter, is what
    // keeps a comment reading `# 3.14` from naming a file `3.14`.
    let named = !stem.is_empty() || name.starts_with('.');
    // Looser than [`is_plausible_extension`], which sizes a *generated*
    // extension: `.gitignore` is a real name and nine characters long.
    let plausible = (1..=12).contains(&extension.len())
        && extension.starts_with(|c: char| c.is_ascii_alphabetic())
        && extension.chars().all(|c| c.is_ascii_alphanumeric());
    (named && plausible).then(|| name.to_string())
}

/// Fence tags that are not their own extension and that the highlighter
/// can't resolve either — it ships Sublime's syntax definitions, which
/// cover neither TypeScript nor Kotlin nor PowerShell, and a tag it can't
/// resolve would otherwise be taken at its word (`.typescript`).
///
/// Only tags that are *wrong* without an entry belong here; anything the
/// highlighter resolves (`rust`, `bash`, `haskell`) already comes out right
/// and stays out of the table. `web::license` keys the comment syntax off
/// the extension this produces, so a missing entry costs the snippet its
/// licence header as well as its name — see that module's own consistency
/// test.
const LANGUAGE_EXTENSIONS: &[(&str, &str)] = &[
    ("batch", "bat"),
    ("csharp", "cs"),
    ("elisp", "el"),
    ("elixir", "ex"),
    ("emacs-lisp", "el"),
    ("fortran", "f90"),
    ("fsharp", "fs"),
    ("golang", "go"),
    ("julia", "jl"),
    ("kotlin", "kt"),
    ("matlab", "m"),
    ("objc", "m"),
    ("objective-c", "m"),
    ("ocaml", "ml"),
    ("pascal", "pas"),
    ("plaintext", "txt"),
    ("powershell", "ps1"),
    ("protobuf", "proto"),
    ("racket", "rkt"),
    ("scheme", "scm"),
    ("shell", "sh"),
    ("terraform", "tf"),
    ("text", "txt"),
    ("typescript", "ts"),
];

/// The extension a generated name gets.
///
/// Taken from the resolved syntax definition, which already carries the
/// real-world extensions for every language the highlighter knows, so
/// `rust` becomes `.rs` without a hand-written table. [`LANGUAGE_EXTENSIONS`]
/// covers the tags it can't resolve and that aren't their own extension;
/// any other unresolved tag is used as its own extension where it reads
/// like one, which is right far more often than `.txt` is.
fn default_extension(language: Option<&str>, syntax: Option<&SyntaxReference>) -> String {
    let language = language.map(|language| language.to_ascii_lowercase());
    let language = language.as_deref();

    // A linear scan, not a binary search: the table is small, and this way
    // it cannot go silently wrong if the alphabetical order below slips.
    if let Some(extension) = language.and_then(|language| {
        LANGUAGE_EXTENSIONS
            .iter()
            .find(|(tag, _)| *tag == language)
            .map(|(_, extension)| *extension)
    }) {
        return extension.to_string();
    }

    syntax
        .and_then(|syntax| {
            syntax
                .file_extensions
                .iter()
                .map(String::as_str)
                .find(|extension| is_plausible_extension(extension))
        })
        .or_else(|| language.filter(|language| is_plausible_extension(language)))
        .unwrap_or("txt")
        .to_ascii_lowercase()
}

fn is_plausible_extension(candidate: &str) -> bool {
    !candidate.is_empty()
        && candidate.len() <= 8
        && candidate.starts_with(|c: char| c.is_ascii_alphabetic())
        && candidate.chars().all(|c| c.is_ascii_alphanumeric())
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
    /// An MIT project to render against, built on disk because
    /// `license::Project` is only ever constructed by detecting one — there
    /// is deliberately no way to assert a licence a project does not have.
    fn mit_workspace() -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("workspace");
        std::fs::write(
            dir.path().join("Cargo.toml"),
            "[package]\nlicense = \"MIT\"\nauthors = [\"Example Holder\"]\n",
        )
        .expect("manifest");
        dir
    }

    fn mit_project() -> orangu::license::Project {
        orangu::license::Project::detect(mit_workspace().path()).expect("detected")
    }

    /// The rendering tests are about markup, not licensing, so they all run
    /// against the same MIT project.
    fn render_markdown_to_html_t(text: &str) -> String {
        render_markdown_to_html(text, Some(&mit_project()))
    }

    use super::*;
    use base64::Engine as _;

    #[test]
    fn renders_emphasis_and_paragraphs() {
        let html = render_markdown_to_html_t("Hello **bold** and *italic*.");
        assert!(html.contains("<strong>bold</strong>"));
        assert!(html.contains("<em>italic</em>"));
        assert!(html.starts_with("<p>"));
    }

    #[test]
    fn renders_headings_lists_and_links() {
        let html =
            render_markdown_to_html_t("# Title\n\n- one\n- two\n\n[docs](https://example.com)");
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
        let html = render_markdown_to_html_t("- one\n\n  two");
        assert!(html.contains("<p>one</p>"));
        assert!(html.contains("<p>two</p>"));
    }

    #[test]
    fn renders_fenced_code_with_syntax_highlighting() {
        let html = render_markdown_to_html_t("```rust\nfn main() {}\n```");
        assert!(html.contains("code-block"));
        assert!(html.contains("fn"));
    }

    #[test]
    fn renders_unknown_language_code_as_plain_pre() {
        let html = render_markdown_to_html_t("```notalanguage\nplain text\n```");
        assert!(html.contains("<pre><code>"));
        assert!(html.contains("plain text"));
    }

    /// Every code block, highlighted or not, is a downloadable window.
    #[test]
    fn every_code_block_carries_a_download_button() {
        for source in [
            "```rust\nfn main() {}\n```",
            "```notalanguage\nplain text\n```",
            "```\nno tag at all\n```",
        ] {
            let html = render_markdown_to_html_t(source);
            assert!(html.contains("class=\"code-block\""), "{html}");
            assert!(html.contains("class=\"code-dl\""), "{html}");
            assert!(html.contains("data-file-name="), "{html}");
        }
    }

    /// The licence is in the block the reader sees, not bolted on at save
    /// time — so it is highlighted, selectable, and already in the text the
    /// download button reads back out.
    #[test]
    fn a_code_block_is_rendered_with_its_licence_header() {
        let html = render_markdown_to_html_t("```rust\nfn main() {}\n```");
        assert!(html.contains("MIT License"), "{html}");
        assert!(html.contains("Copyright (c)"), "{html}");
        assert!(html.contains("fn"), "{html}");
    }

    /// A format with nowhere to put a comment is shown as written — a
    /// displayed `.json` that doesn't parse would be worse than an
    /// unlicensed one.
    #[test]
    fn a_block_that_cannot_carry_a_comment_is_rendered_untouched() {
        let html = render_markdown_to_html_t("```json\n{\"a\": 1}\n```");
        assert!(!html.contains("MIT License"), "{html}");

        let untagged = render_markdown_to_html_t("```\nplain text\n```");
        assert!(!untagged.contains("MIT License"), "{untagged}");
    }

    /// The name is read off line one, and line one is exactly what the
    /// licence header displaces — so it has to be taken first.
    #[test]
    fn the_licence_does_not_hide_a_first_line_file_name() {
        let html = render_markdown_to_html_t("```rust\n// src/lib.rs\nfn f() {}\n```");
        assert!(html.contains("data-file-name=\"lib.rs\""), "{html}");
        assert!(html.contains("MIT License"), "{html}");
    }

    /// A model that wrote its own header must not get a second one.
    #[test]
    fn a_block_that_already_carries_the_licence_gets_no_second_copy() {
        let source = orangu::license::apply("fn main() {}\n", "main.rs", Some(&mit_project()));
        let html = render_markdown_to_html_t(&format!("```rust\n{source}```"));
        assert_eq!(html.matches("MIT License").count(), 1, "{html}");
    }

    /// A diagram is not a code window and never reaches the licensing path.
    #[test]
    fn a_diagram_is_not_licensed() {
        let html = render_markdown_to_html_t("```mermaid\nflowchart TD\n    A --> B\n```");
        assert!(html.contains("mermaid-diagram"), "{html}");
        assert!(!html.contains("MIT License"), "{html}");
    }

    /// The three ways a model spells a file name onto the fence.
    #[test]
    fn takes_the_file_name_off_the_fence() {
        for source in [
            "```rust src/main.rs\nfn main() {}\n```",
            "```rust:src/main.rs\nfn main() {}\n```",
            "```rust title=\"src/main.rs\"\nfn main() {}\n```",
        ] {
            let html = render_markdown_to_html_t(source);
            assert!(html.contains("data-file-name=\"main.rs\""), "{html}");
            assert!(html.contains(">main.rs</span>"), "{html}");
        }
    }

    /// A fence that is *only* a file name, with the language left implied.
    #[test]
    fn takes_a_bare_file_name_fence() {
        let html = render_markdown_to_html_t("```Makefile\nall:\n\techo hi\n```");
        assert!(html.contains("data-file-name=\"Makefile\""), "{html}");
    }

    /// Failing that, the block's own first-line comment.
    #[test]
    fn takes_the_file_name_off_a_first_line_comment() {
        let cases = [
            ("```rust\n// src/lib.rs\nfn f() {}\n```", "lib.rs"),
            ("```python\n# File: app/models.py\nx = 1\n```", "models.py"),
            ("```html\n<!-- index.html -->\n<p>hi</p>\n```", "index.html"),
            (
                "```c\n/* main.c */\nint main(void) { return 0; }\n```",
                "main.c",
            ),
        ];
        for (source, expected) in cases {
            let html = render_markdown_to_html_t(source);
            assert!(
                html.contains(&format!("data-file-name=\"{expected}\"")),
                "{source} -> {html}"
            );
        }
    }

    /// The guard that keeps the first-line rule from firing on prose. A
    /// wrong name on a saved file is worse than a numbered one.
    #[test]
    fn a_prose_comment_does_not_name_the_file() {
        for source in [
            "```bash\n# Install the dependencies first\nnpm i\n```",
            "```python\n# roughly 3.14\nx = 1\n```",
            "```js\n// see https://example.com/docs.html\nlet x = 1;\n```",
            "```c\n#include <stdio.h>\nint main(void) { return 0; }\n```",
        ] {
            let html = render_markdown_to_html_t(source);
            assert!(
                html.contains("data-file-name=\"orangu-snippet-1."),
                "{html}"
            );
        }
    }

    /// With no name anywhere, the extension comes from the highlighter's own
    /// syntax definitions and the number from the block's position.
    #[test]
    fn generated_names_number_and_extend_by_language() {
        let html = render_markdown_to_html_t(
            "```rust\nfn main() {}\n```\n\ntext\n\n```python\nx = 1\n```",
        );
        assert!(
            html.contains("data-file-name=\"orangu-snippet-1.rs\""),
            "{html}"
        );
        assert!(
            html.contains("data-file-name=\"orangu-snippet-2.py\""),
            "{html}"
        );
    }

    /// A tag the highlighter has no definition for still beats `.txt` as an
    /// extension when it reads like one; a tag that doesn't, doesn't.
    #[test]
    fn unknown_language_falls_back_to_the_tag_then_txt() {
        let html = render_markdown_to_html_t("```zigzag\nconst x = 1;\n```");
        assert!(
            html.contains("data-file-name=\"orangu-snippet-1.zigzag\""),
            "{html}"
        );

        let html = render_markdown_to_html_t("```++\nwhat\n```");
        assert!(
            html.contains("data-file-name=\"orangu-snippet-1.txt\""),
            "{html}"
        );
    }

    /// A name is attacker-supplied text like everything else in a reply.
    #[test]
    fn rejects_a_file_name_that_is_not_one() {
        let html = render_markdown_to_html_t("```rust ../../etc/passwd\nfn main() {}\n```");
        assert!(!html.contains("passwd"), "{html}");

        let html =
            render_markdown_to_html_t("```rust \"><script>alert(1)</script>\nfn f() {}\n```");
        assert!(!html.contains("<script>"), "{html}");
    }

    /// A diagram block doesn't consume a snippet number — it isn't a code
    /// window and has its own download.
    #[test]
    fn a_diagram_does_not_take_a_snippet_number() {
        let html = render_markdown_to_html_t(
            "```mermaid\nflowchart TD\n    A --> B\n```\n\n```rust\nfn main() {}\n```",
        );
        assert!(html.contains("mermaid-diagram"), "{html}");
        assert!(
            html.contains("data-file-name=\"orangu-snippet-1.rs\""),
            "{html}"
        );
    }

    /// The two halves have to agree. this module picks the file name from the
    /// fence's language; `orangu::license` picks the comment from that name.
    /// If they drift, a snippet in a mainstream language saves with no
    /// licence header at all and nothing else notices.
    #[test]
    fn every_name_render_generates_for_a_common_language_gets_a_header() {
        for language in [
            "rust",
            "python",
            "javascript",
            "typescript",
            "tsx",
            "jsx",
            "go",
            "golang",
            "java",
            "c",
            "c++",
            "cpp",
            "csharp",
            "c#",
            "fsharp",
            "bash",
            "sh",
            "shell",
            "zsh",
            "ruby",
            "php",
            "sql",
            "yaml",
            "yml",
            "toml",
            "ini",
            "html",
            "xml",
            "css",
            "scss",
            "lua",
            "kotlin",
            "swift",
            "scala",
            "haskell",
            "perl",
            "r",
            "erlang",
            "elixir",
            "clojure",
            "racket",
            "scheme",
            "elisp",
            "ocaml",
            "pascal",
            "fortran",
            "julia",
            "nim",
            "zig",
            "dart",
            "vue",
            "svelte",
            "proto",
            "protobuf",
            "graphql",
            "terraform",
            "powershell",
            "batch",
            "latex",
            "tex",
            "markdown",
            "make",
            "makefile",
            "dockerfile",
        ] {
            let html = render_markdown_to_html_t(&format!("```{language}\nplaceholder\n```"));
            let name = html
                .split("data-file-name=\"")
                .nth(1)
                .and_then(|rest| rest.split('"').next())
                .unwrap_or_else(|| panic!("{language}: no file name in {html}"));
            assert!(
                orangu::license::header_for(name, &mit_project()).is_some(),
                "{language} saves as {name}, which has no licence header"
            );
        }
    }

    #[test]
    fn escapes_html_in_plain_text_to_prevent_injection() {
        let html = render_markdown_to_html_t("<script>alert(1)</script>");
        assert!(!html.contains("<script>"));
        assert!(html.contains("&lt;script&gt;"));
    }

    #[test]
    fn escapes_html_inside_inline_and_block_code() {
        let html = render_markdown_to_html_t("`<b>x</b>`");
        assert!(!html.contains("<code><b>"));
        assert!(html.contains("&lt;b&gt;"));
    }

    #[test]
    fn escapes_quotes_in_link_and_image_attributes() {
        let html = render_markdown_to_html_t("![alt\"](http://x/\"y)");
        assert!(!html.contains("\"y\""));
        assert!(html.contains("&quot;"));
    }

    #[test]
    fn renders_inline_math_as_a_katex_source_span() {
        let html = render_markdown_to_html_t(r"The set $A \to B$ is finite.");
        assert!(html.contains(r#"<span class="katex-source" data-tex="A \to B">"#));
        // Escaped fallback content too, so the raw TeX is at least legible
        // if the client-side katex.render() pass never runs.
        assert!(html.contains(r"A \to B</span>"));
        assert!(!html.contains("katex-block"));
    }

    #[test]
    fn renders_block_math_as_a_katex_block_div() {
        let html = render_markdown_to_html_t("$$\n\\sum_{i=0}^n i\n$$");
        assert!(html.contains(r#"<div class="katex-source katex-block" data-tex="#));
        assert!(html.contains(r"\sum_{i=0}^n i"));
    }

    #[test]
    fn renders_a_closed_mermaid_block_as_a_diagram() {
        let html = render_markdown_to_html_t("```mermaid\nflowchart TD\n    A --> B\n```");
        assert!(html.contains("<figure class=\"mermaid-diagram\">"));
        assert!(html.contains("<img class=\"mermaid-light\" src=\"data:image/svg+xml;base64,"));
        assert!(html.contains("<img class=\"mermaid-dark\" src=\"data:image/svg+xml;base64,"));
        // The source survives as copyable, screen-reader-legible text.
        assert!(html.contains("flowchart TD"));
    }

    #[test]
    fn renders_plantuml_with_svg_and_png_downloads() {
        let html =
            render_markdown_to_html_t("```plantuml\n@startuml\nAlice -> Bob: hello\n@enduml\n```");
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
            render_markdown_to_html_t("```PlantUML\n@startuml\nAlice -> Bob: hello\n@enduml\n```");
        assert!(
            html.contains("<figure class=\"plantuml-diagram\">"),
            "{html}"
        );
    }

    #[test]
    fn plantuml_streaming_and_fallback_match_mermaid_behaviour() {
        let open = render_markdown_to_html_t("```plantuml\n@startuml\nAlice -> Bob: hi\n@enduml");
        assert!(!open.contains("plantuml-diagram"), "{open}");
        let unsupported = render_markdown_to_html_t(
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
        let html = render_markdown_to_html_t("```mermaid\nflowchart TD\n    A --> B");
        assert!(!html.contains("mermaid-diagram"), "{html}");
        assert!(html.contains("flowchart TD"));
    }

    #[test]
    fn renders_a_diagram_once_the_closing_fence_arrives() {
        // The same block one token later — the transition the test above
        // guards has to actually complete, or diagrams would never appear.
        let html = render_markdown_to_html_t("```mermaid\nflowchart TD\n    A --> B\n```\n");
        assert!(html.contains("mermaid-diagram"), "{html}");
    }

    #[test]
    fn keeps_an_earlier_diagram_while_a_later_block_streams() {
        // Only the trailing unclosed fence is held back; a diagram the model
        // already finished must not blink out while it types the next block.
        let html = render_markdown_to_html_t(
            "```mermaid\nflowchart TD\n    A --> B\n```\n\nthen\n\n```rust\nfn main() {",
        );
        assert!(html.contains("mermaid-diagram"), "{html}");
    }

    #[test]
    fn falls_back_to_a_code_block_when_the_diagram_does_not_parse() {
        // Models emit near-miss Mermaid often enough that this is the
        // difference between a readable reply and a blank frame.
        let html = render_markdown_to_html_t("```mermaid\nflowchart TD\n    A[[[--->>> ???\n```");
        assert!(!html.contains("mermaid-diagram"), "{html}");
        assert!(html.contains("A[[["), "{html}");
    }

    #[test]
    fn leaves_non_mermaid_code_blocks_alone() {
        let html = render_markdown_to_html_t("```rust\nfn main() {}\n```");
        assert!(!html.contains("mermaid-diagram"));
        assert!(html.contains("code-block"));
    }

    #[test]
    fn every_diagram_carries_a_full_resolution_download() {
        let html = render_markdown_to_html_t("```mermaid\nflowchart TD\n    A --> B\n```");
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
        let html = render_markdown_to_html_t("```mermaid\nflowchart TD\n    A --> B\n```");
        assert!(
            html.contains(" width=\"") && html.contains(" height=\""),
            "{html}"
        );
    }

    #[test]
    fn renders_a_diagram_from_an_untagged_fence() {
        // Models don't always tag the fence; a diagram served as a wall of
        // code is the failure this avoids.
        let html = render_markdown_to_html_t("```\nflowchart TD\n    A --> B\n```");
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
            let html = render_markdown_to_html_t(source);
            assert!(!html.contains("mermaid-diagram"), "{source} -> {html}");
        }
    }

    #[test]
    fn respects_an_explicit_non_mermaid_tag() {
        // Tagged `bash` but holding valid Mermaid: the tag is the model
        // saying what it wrote, and overriding it is how a shell transcript
        // ends up drawn as a flowchart.
        let html = render_markdown_to_html_t("```bash\ngraph LR\n    A --> B\n```");
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
        let html = render_markdown_to_html_t(r"$a < b \& c > d$");
        assert!(html.contains("data-tex=\"a &lt; b \\&amp; c &gt; d\""));
        assert!(!html.contains("$a < b"));
    }
}
