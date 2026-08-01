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

//! Turns a ```` ```mermaid ```` code block's source into an SVG diagram, for
//! [`super::render`] to embed in the transcript.
//!
//! Rendering is headless and local (the `merman` crate): no Node, no
//! browser, no network, so a diagram renders on the same offline box the
//! model runs on. Unlike the `$...$` math path — which ships raw TeX to the
//! client for KaTeX to typeset, because no server-side TeX engine exists in
//! Rust — this runs entirely server-side and the client only ever sees a
//! finished picture.
//!
//! Two things follow from chat output being untrusted, and they decide the
//! whole shape of this module:
//!
//! 1. **The SVG is embedded as an `<img>` data URI, never inlined.** Merman
//!    strips scripts and event-handler attributes from diagram labels, but
//!    it does *not* escape a literal `</svg>` in one — a label of
//!    `A["</svg>..."]` emits that tag raw. Inlined, the HTML parser would
//!    treat it as the real end tag, close the diagram early and let the rest
//!    of the label escape into the page. Inside `<img>` the SVG is a
//!    separate, isolated document instead: scripts are inert, `id`s can't
//!    collide with the page or with another diagram, and nothing can break
//!    out into the transcript. This keeps the same "escape everything"
//!    stance [`super::render`] takes for every other node kind.
//! 2. **The SVG has to stand alone.** An `<img>` document gets no HTML
//!    layout engine, so Mermaid's usual `<foreignObject>` HTML labels would
//!    render as nothing at all. [`HostThemeOutput::resvg_safe_editor`] is
//!    the setting that turns those into plain SVG `<text>`, which is why it
//!    is not optional here.
//!
//! Because an `<img>` also can't inherit the page's CSS variables, each
//! diagram is rendered twice, once per theme, and `app.css` shows whichever
//! matches. The two SVGs are built in one pass and cached together.

use base64::Engine as _;
use merman::render::{HeadlessRenderer, HostThemeOutput, HostThemeProfile, HostThemeRoles};
use rustc_hash::FxHasher;
use std::{
    collections::HashMap,
    hash::{Hash as _, Hasher as _},
    sync::{Mutex, OnceLock},
};

/// A finished diagram: one picture per console theme, plus its natural size.
pub struct Diagram {
    pub light: String,
    pub dark: String,
    /// Natural size in CSS pixels, from the SVG's `viewBox`.
    ///
    /// Not used to force full size — `app.css` scales a diagram to the
    /// message — but so the browser reserves the right aspect ratio before
    /// the image decodes rather than reflowing the transcript when it lands.
    pub width: f64,
    pub height: f64,
}

impl Diagram {
    fn new(light: &str, dark: &str) -> Self {
        let (width, height) = viewbox_size(light).unwrap_or((0.0, 0.0));
        Self {
            light: data_uri(light),
            dark: data_uri(dark),
            width,
            height,
        }
    }
}

/// A diagram found inside an attached file, with the source it came from.
pub struct Found {
    pub diagram: &'static Diagram,
    pub source: String,
}

/// Pulls `width`/`height` out of `viewBox="minX minY width height"`.
///
/// Falls back to `None` rather than guessing: an `<img>` with no dimensions
/// still displays, whereas wrong dimensions would distort the diagram.
fn viewbox_size(svg: &str) -> Option<(f64, f64)> {
    let rest = svg.split_once("viewBox=\"")?.1;
    let value = rest.split_once('"')?.0;
    let mut parts = value.split_whitespace();
    let (_, _, width, height) = (parts.next()?, parts.next()?, parts.next()?, parts.next()?);
    let (width, height) = (width.parse().ok()?, height.parse().ok()?);
    (width > 0.0 && height > 0.0).then_some((width, height))
}

/// base64 rather than percent-encoded UTF-8: an SVG is full of `#`, `<`, `"`
/// and `&`, each of which would need escaping in a raw data URI (and a
/// missed one silently truncates the image), whereas base64 is inert in both
/// an attribute and a URL at the cost of ~33% size.
fn data_uri(svg: &str) -> String {
    format!(
        "data:image/svg+xml;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(svg.as_bytes())
    )
}

/// Cap on distinct diagrams held in memory. A diagram is a few tens of KiB
/// across both themes, so this is a bounded handful of MiB; the map is
/// cleared wholesale on overflow rather than evicted in LRU order, which
/// costs one re-render of anything still on screen and keeps this to a plain
/// `HashMap` (the entries are cheap to rebuild — that is the whole premise
/// of the cache).
const CACHE_LIMIT: usize = 256;

type Cache = HashMap<u64, Option<&'static Diagram>>;

fn cache() -> &'static Mutex<Cache> {
    static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Maps the console's CSS custom properties (`app.css` `:root`) onto
/// Merman's semantic theme roles, so a diagram sits in the transcript as if
/// it were styled by the same stylesheet as everything around it rather than
/// arriving in Mermaid's stock lavender.
fn roles(dark: bool) -> HostThemeRoles {
    if dark {
        HostThemeRoles {
            canvas: Some("#23272e".into()),        // --assistant-bubble
            surface: Some("#2a2f38".into()),       // --bg-input
            surface_alt: Some("#2b3a55".into()),   // --user-bubble
            surface_muted: Some("#23272e".into()), // --bg-raised
            text: Some("#e6e8eb".into()),          // --text
            subtle_text: Some("#9aa3b2".into()),   // --text-dim
            border: Some("#5b9dff".into()),        // --accent
            line: Some("#9aa3b2".into()),          // --text-dim
            edge_label_background: Some("#23272e".into()),
            cluster_background: Some("#2a2f38".into()),
            cluster_border: Some("#343a45".into()), // --border
            note_background: Some("#2b3a55".into()),
            note_border: Some("#5b9dff".into()),
            note_text: Some("#e6e8eb".into()),
            actor_background: Some("#2a2f38".into()),
            actor_border: Some("#5b9dff".into()),
            actor_text: Some("#e6e8eb".into()),
            error: Some("#ff6b6b".into()), // --error
            ..HostThemeRoles::default()
        }
    } else {
        HostThemeRoles {
            canvas: Some("#ffffff".into()),        // --assistant-bubble
            surface: Some("#eef0f3".into()),       // --bg-input
            surface_alt: Some("#dbe6ff".into()),   // --user-bubble
            surface_muted: Some("#f5f6f8".into()), // --bg
            text: Some("#1b1e23".into()),          // --text
            subtle_text: Some("#62697a".into()),   // --text-dim
            border: Some("#2563eb".into()),        // --accent
            line: Some("#62697a".into()),          // --text-dim
            edge_label_background: Some("#ffffff".into()),
            cluster_background: Some("#f5f6f8".into()),
            cluster_border: Some("#d7dbe0".into()), // --border
            note_background: Some("#dbe6ff".into()),
            note_border: Some("#2563eb".into()),
            note_text: Some("#1b1e23".into()),
            actor_background: Some("#eef0f3".into()),
            actor_border: Some("#2563eb".into()),
            actor_text: Some("#1b1e23".into()),
            error: Some("#d92d20".into()), // --error
            ..HostThemeRoles::default()
        }
    }
}

/// `resvg_safe_editor` is the load-bearing part — see this module's doc
/// comment for why `<foreignObject>` labels can't survive the `<img>`
/// embedding. The font stack matches `app.css`'s body font so diagram text
/// looks like transcript text.
fn renderer(dark: bool) -> &'static HeadlessRenderer {
    static RENDERERS: OnceLock<[HeadlessRenderer; 2]> = OnceLock::new();
    let renderers = RENDERERS.get_or_init(|| {
        [false, true].map(|dark| {
            let profile = HostThemeProfile::builder()
                .font_family(
                    "system-ui, -apple-system, \"Segoe UI\", Roboto, \"Helvetica Neue\", sans-serif",
                )
                .roles(roles(dark))
                .output(HostThemeOutput::resvg_safe_editor())
                .build();
            HeadlessRenderer::new()
                .with_host_theme(&profile)
                // Every diagram is its own isolated `<img>` document, so a
                // fixed id is fine and keeps output byte-identical across
                // renders of the same source.
                .with_diagram_id("orangu-diagram")
        })
    });
    &renderers[usize::from(dark)]
}

/// Every Mermaid diagram header, with the tokens each may legally be
/// followed by on its own line. An empty slice means the keyword stands
/// alone.
///
/// This table is what [`looks_like_diagram`] gates on, and it exists because
/// Mermaid's own parsers — merman's included — are far too permissive to use
/// as a detector. Handed the sentence `graph is a data structure of nodes
/// and edges`, merman detects a flowchart and renders one, with `is` as a
/// node. `classDiagram is what you want` and a log line reading `info: build
/// succeeded` do the same thing. All three are things a model or a text file
/// plausibly contains, and all three would otherwise turn into nonsense
/// pictures. Requiring the first line to be a *bare header* rejects them
/// while still admitting every real diagram, including `pie title A Very
/// Long Descriptive Title`.
const HEADERS: &[(&str, &[&str])] = &[
    ("flowchart", &["TD", "TB", "BT", "RL", "LR"]),
    ("graph", &["TD", "TB", "BT", "RL", "LR"]),
    ("sequenceDiagram", &[]),
    ("classDiagram", &[]),
    ("classDiagram-v2", &[]),
    ("stateDiagram", &[]),
    ("stateDiagram-v2", &[]),
    ("erDiagram", &[]),
    ("journey", &[]),
    ("gantt", &[]),
    ("pie", &["title", "showData"]),
    ("quadrantChart", &[]),
    ("requirementDiagram", &[]),
    ("gitGraph", &["TB", "BT", "LR", "RL"]),
    ("mindmap", &[]),
    ("timeline", &[]),
    ("sankey-beta", &[]),
    ("xychart-beta", &["horizontal"]),
    ("block-beta", &[]),
    ("packet-beta", &[]),
    ("radar-beta", &[]),
    ("treemap", &[]),
    ("treemap-beta", &[]),
    ("architecture-beta", &[]),
    ("kanban", &[]),
    ("C4Context", &[]),
    ("C4Container", &[]),
    ("C4Component", &[]),
    ("C4Dynamic", &[]),
    ("C4Deployment", &[]),
    ("info", &["showInfo"]),
];

/// The first line that carries a diagram header: front matter and leading
/// `%%` comments are both legal above one, and both appear in real files.
fn first_meaningful_line(text: &str) -> Option<&str> {
    let mut lines = text.lines().peekable();
    if lines.peek().map(|line| line.trim()) == Some("---") {
        lines.next();
        for line in lines.by_ref() {
            if line.trim() == "---" {
                break;
            }
        }
    }
    lines
        .map(str::trim)
        .find(|line| !line.is_empty() && !line.starts_with("%%"))
}

/// Whether `text` opens with a Mermaid diagram header — the cheap, strict
/// test used to decide whether untagged content is worth handing to the
/// renderer at all.
///
/// Deliberately conservative: a false negative costs a diagram that stays a
/// code block, while a false positive turns someone's shell transcript into
/// a picture. See [`HEADERS`] for the cases that forced this.
pub fn looks_like_diagram(text: &str) -> bool {
    let Some(line) = first_meaningful_line(text) else {
        return false;
    };
    let line = line.strip_suffix(';').unwrap_or(line).trim_end();

    HEADERS.iter().any(|(keyword, allowed)| {
        let Some(rest) = line.strip_prefix(keyword) else {
            return false;
        };
        // `gitGraph:` and `gitGraph TB:` are both legal openings.
        let trimmed = rest.strip_prefix(':').unwrap_or(rest).trim();
        if trimmed.is_empty() {
            return true;
        }
        // A real word boundary, so `graphql` isn't read as `graph`.
        if !rest.starts_with([' ', '\t', ':']) {
            return false;
        }
        let next = trimmed.split_whitespace().next().unwrap_or("");
        allowed.contains(&next.strip_suffix(':').unwrap_or(next))
    })
}

/// Renders `source` for both themes, or returns `None` if it isn't a
/// diagram this can draw.
///
/// `None` is the honest and common answer, not an edge case: while a reply
/// is streaming, the caller only reaches this once the fence has closed, but
/// models still emit Mermaid with typos, invented syntax, or a dialect newer
/// than the vendored parser. Every one of those lands here, and the caller
/// falls back to showing the source as an ordinary code block — strictly
/// better than a blank frame where a picture should be.
pub fn render(source: &str) -> Option<&'static Diagram> {
    let mut hasher = FxHasher::default();
    source.hash(&mut hasher);
    let key = hasher.finish();

    // The transcript is re-rendered from scratch on every streamed token, so
    // a reply that produced a diagram early would otherwise re-lay it out
    // hundreds of times over the rest of the stream. Failures are cached
    // too: a code block the model labelled `mermaid` but never closes into
    // valid syntax is retried just as often as a good one.
    if let Ok(cache) = cache().lock()
        && let Some(hit) = cache.get(&key)
    {
        return *hit;
    }

    let rendered = render_uncached(source);

    if let Ok(mut cache) = cache().lock() {
        if cache.len() >= CACHE_LIMIT {
            cache.clear();
        }
        cache.insert(key, rendered);
    }
    rendered
}

/// Leaks the rendered pair deliberately: entries live in a process-wide
/// cache for the life of the server anyway, and `&'static` lets [`render`]
/// hand out a reference without cloning tens of KiB of SVG per token or
/// holding the cache lock across the caller's use of it.
fn render_uncached(source: &str) -> Option<&'static Diagram> {
    // `render_svg_sync` distinguishes "no diagram here" (`Ok(None)`, e.g. a
    // block mislabelled `mermaid`) from "a diagram, but it doesn't parse"
    // (`Err`). Both mean the same thing to the caller, so both collapse to
    // `None` here.
    let light = renderer(false).render_svg_sync(source).ok().flatten()?;
    let dark = renderer(true).render_svg_sync(source).ok().flatten()?;
    Some(Box::leak(Box::new(Diagram::new(&light, &dark))))
}

/// Upper bound on diagrams drawn from one attachment. A design document can
/// legitimately hold dozens; past this they stay as text rather than turning
/// one upload into an unbounded amount of layout work. The cap is reported
/// to the reader (see `mod.rs`'s `AttachmentView`) rather than silently
/// truncating the list.
pub const MAX_PER_ATTACHMENT: usize = 32;

/// Finds every Mermaid diagram in an attached file's extracted text.
///
/// Two shapes matter, because both are what people actually attach:
///
/// * **The whole file is a diagram** — a `.mmd`/`.mermaid` export, or a
///   `.txt` someone pasted a diagram into. There is no fence to key on, so
///   this is exactly the case [`looks_like_diagram`] exists for.
/// * **A document containing diagrams** — a `.md` design doc with
///   ```` ```mermaid ```` blocks in it, which is the common one.
///
/// Fenced blocks are found by parsing the text as markdown rather than by
/// scanning for backticks, so fence lengths, indentation and info strings
/// follow the same CommonMark rules the transcript renderer already uses. A
/// block tagged `mermaid`/`mmd` is trusted on its tag; an untagged one has
/// to pass [`looks_like_diagram`] first. A block tagged as something else is
/// left alone — an explicit `rust` or `bash` tag is the author telling us
/// what it is.
pub fn find_in_text(text: &str) -> Vec<Found> {
    // A file that is nothing but a diagram: no fence, so nothing to walk.
    if looks_like_diagram(text)
        && let Some(diagram) = render(text.trim())
    {
        return vec![Found {
            diagram,
            source: text.trim().to_string(),
        }];
    }

    let Ok(tree) = markdown::to_mdast(text, &markdown::ParseOptions::gfm()) else {
        return Vec::new();
    };
    let mut found = Vec::new();
    collect_from_nodes(&tree, &mut found);
    found
}

fn collect_from_nodes(node: &markdown::mdast::Node, found: &mut Vec<Found>) {
    if found.len() >= MAX_PER_ATTACHMENT {
        return;
    }
    if let markdown::mdast::Node::Code(code) = node {
        let tagged_mermaid = matches!(code.lang.as_deref().map(str::trim), Some("mermaid" | "mmd"));
        let untagged = code.lang.as_deref().map(str::trim).unwrap_or("").is_empty();
        if (tagged_mermaid || (untagged && looks_like_diagram(&code.value)))
            && let Some(diagram) = render(&code.value)
        {
            found.push(Found {
                diagram,
                source: code.value.clone(),
            });
        }
        return;
    }
    if let Some(children) = node.children() {
        for child in children {
            collect_from_nodes(child, found);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FLOWCHART: &str = "flowchart TD\n    A[Start] --> B[Done]";

    #[test]
    fn renders_a_flowchart_for_both_themes() {
        let diagram = render(FLOWCHART).expect("flowchart renders");
        assert!(diagram.light.starts_with("data:image/svg+xml;base64,"));
        assert!(diagram.dark.starts_with("data:image/svg+xml;base64,"));
        // Same diagram, different palette — if these ever match, the host
        // theme stopped being applied and dark mode silently shows the
        // light diagram.
        assert_ne!(diagram.light, diagram.dark);
    }

    #[test]
    fn embedded_svg_is_standalone_and_themed() {
        let diagram = render("sequenceDiagram\n    A->>B: hi").expect("sequence renders");
        let svg = decode(&diagram.dark);
        // `<foreignObject>` labels render as nothing inside an `<img>`; the
        // resvg-safe pipeline is what keeps them out. This is the assertion
        // that catches that setting being dropped.
        assert!(!svg.contains("foreignObject"), "{svg}");
        assert!(!svg.contains("<script"), "{svg}");
        // No external fetches: the console has to work fully offline, and
        // an `<img>` that reaches out would leak the diagram's existence to
        // whatever host it reached. Checked on the reference forms rather
        // than on "http" anywhere, which would only match the `xmlns`
        // namespace URIs — those name the SVG dialect, they aren't fetched.
        assert!(!svg.contains("href=\"http"), "{svg}");
        assert!(!svg.contains("url(http"), "{svg}");
        assert!(!svg.contains("@import"), "{svg}");
        assert!(svg.contains("#23272e"), "dark canvas colour missing: {svg}");
    }

    #[test]
    fn rejects_non_mermaid_and_malformed_sources() {
        // A code block mislabelled `mermaid`...
        assert!(render("fn main() { println!(\"hi\"); }").is_none());
        // ...and one that opens a real diagram but never parses. Both have
        // to be refused rather than rendered as an empty frame, so the
        // caller can fall back to showing the source.
        assert!(render("flowchart TD\n    A[[[--->>> ???").is_none());
        assert!(render("").is_none());
    }

    #[test]
    fn reads_the_viewbox() {
        assert_eq!(
            viewbox_size(r#"<svg viewBox="0 0 240 120">"#),
            Some((240.0, 120.0))
        );
        // A negative origin is common (merman emits one); only the last two
        // numbers are the size.
        assert_eq!(
            viewbox_size(r#"<svg viewBox="-50 -10 450 215">"#),
            Some((450.0, 215.0))
        );
        assert_eq!(viewbox_size("<svg>"), None);
        assert_eq!(viewbox_size(r#"<svg viewBox="0 0 0 0">"#), None);
    }

    #[test]
    fn caches_repeated_renders() {
        // Streaming re-renders the same completed diagram once per token,
        // so a repeat lookup has to be a cache hit, not a re-layout.
        let first = render(FLOWCHART).expect("renders");
        let second = render(FLOWCHART).expect("renders");
        assert!(std::ptr::eq(first, second));
    }

    #[test]
    fn labels_cannot_break_out_of_the_svg_document() {
        // The `</svg>` here is emitted raw by merman inside a label. It is
        // harmless only because the SVG is base64'd into an `<img>` — this
        // test pins that it never reaches the transcript as live markup.
        let diagram =
            render("flowchart TD\n    A[\"</svg><script>alert(1)</script>\"]").expect("renders");
        assert!(!diagram.light.contains("</svg>"));
        assert!(!diagram.light.contains("<script"));
        let svg = decode(&diagram.light);
        assert!(!svg.contains("<script"), "{svg}");
        assert!(!svg.contains("alert(1)"), "{svg}");
        // Exactly one `</svg>`: the document's own closing tag.
        assert_eq!(svg.matches("</svg>").count(), 1, "{svg}");
    }

    #[test]
    fn header_gate_admits_every_diagram_family() {
        for source in [
            "flowchart TD\n  A --> B",
            "graph LR\n  A-->B",
            "sequenceDiagram\n  A->>B: hi",
            "classDiagram\n  class R",
            "stateDiagram-v2\n  [*] --> Idle",
            "erDiagram\n  A ||--o{ B : has",
            "gantt\n  title R",
            "pie title Tokens\n  \"a\" : 1",
            // A long free-text title must not be mistaken for prose.
            "pie title A Very Long Descriptive Title Here\n  \"a\" : 1",
            "mindmap\n  root((r))",
            "gitGraph\n  commit",
            "gitGraph:\n  commit",
            "journey\n  title D",
            "timeline\n  title H",
            "quadrantChart\n  title R",
            "sankey-beta\n\nA,B,10",
            "xychart-beta\n  title \"t\"",
            "block-beta\n  columns 3",
            "requirementDiagram\n  requirement R1 {",
            "C4Context\n  title S",
            "packet-beta\n0-15: \"Src\"",
            "radar-beta\n  axis a, b",
            "treemap-beta\n\"Root\"",
            // Front matter and a leading `%%` comment are both legal above
            // the header.
            "---\ntitle: X\n---\nflowchart TD\n  A-->B",
            "%% a note\nflowchart TD\n  A-->B",
        ] {
            assert!(looks_like_diagram(source), "rejected: {source:?}");
        }
    }

    #[test]
    fn header_gate_rejects_prose_and_code_that_merman_would_render() {
        // The first three are the reason the gate exists at all: merman
        // detects and happily *renders* each one, so without this they would
        // become nonsense pictures.
        for source in [
            "graph is a data structure of nodes and edges",
            "classDiagram is what you want",
            "info: build succeeded\ninfo: 3 files written",
            "pie chart data below:\n  a, 1",
            "timeline of the project:\n  - phase one",
            "$ npm install\n$ npm run build",
            "{\n  \"name\": \"x\"\n}",
            "fn main() { println!(\"hi\"); }",
            "SELECT * FROM users;",
            "GET /api/models HTTP/1.1",
            "| a | b |\n|---|---|",
            "state: running\nport: 8080",
            // Word-boundary cases: neither is a `graph`/`info` header.
            "graphql query { user { id } }",
            "graphData = loadGraph()",
            "",
        ] {
            assert!(!looks_like_diagram(source), "accepted: {source:?}");
        }
    }

    #[test]
    fn finds_a_whole_file_that_is_one_diagram() {
        // A `.mmd` export: no fence to key on, which is the case the header
        // gate exists for.
        let found = find_in_text("flowchart TD\n    A[Start] --> B[Done]\n");
        assert_eq!(found.len(), 1);
        assert!(found[0].source.starts_with("flowchart TD"));
    }

    #[test]
    fn finds_diagrams_inside_an_attached_markdown_document() {
        let doc = "# Design\n\nIntro text.\n\n```mermaid\nflowchart TD\n    A --> B\n```\n\nMore prose.\n\n```\nsequenceDiagram\n    A->>B: hi\n```\n\n```bash\ngraph LR\n    A --> B\n```\n";
        let found = find_in_text(doc);
        // The tagged block and the untagged-but-detected one, but *not* the
        // block the author tagged `bash` — an explicit tag is respected even
        // when the contents would parse as a diagram.
        assert_eq!(
            found.len(),
            2,
            "{:?}",
            found.iter().map(|f| &f.source).collect::<Vec<_>>()
        );
        assert!(found[0].source.contains("flowchart TD"));
        assert!(found[1].source.contains("sequenceDiagram"));
    }

    #[test]
    fn finds_nothing_in_an_ordinary_document() {
        let doc = "# Notes\n\nJust prose here.\n\n```rust\nfn main() {}\n```\n";
        assert!(find_in_text(doc).is_empty());
    }

    fn decode(data_uri: &str) -> String {
        let b64 = data_uri
            .strip_prefix("data:image/svg+xml;base64,")
            .expect("data URI prefix");
        String::from_utf8(
            base64::engine::general_purpose::STANDARD
                .decode(b64)
                .expect("valid base64"),
        )
        .expect("valid utf-8")
    }
}
