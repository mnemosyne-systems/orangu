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

//! The `/export` tool.
//!
//! Renders either the console output window (`export console`) or the last
//! review report (`export review`) to a PDF saved in the root of the
//! workspace as `{repository}-{branch}-console.pdf` or
//! `{repository}-{branch}-review.pdf`.
//!
//! Every page carries a header band with `{repository}-{branch}` and a footer
//! band with `orangu {version} ({model})` — both in white on the orangu brand
//! colour, the footer's `orangu` linking to the project site.
//!
//! Text is set in **Red Hat Text** (embedded from `assets/fonts`, SIL OFL), so
//! the brand typeface ships in the binary and needs no system fonts. If the
//! embedded font cannot be loaded the export falls back to the closest
//! printpdf-native face, Helvetica. The PDF keeps the Markdown formatting as
//! much as it can: the console export prints the output window line for line
//! (ANSI styling removed); the review export renders the report's Markdown with
//! brand-coloured headings, bold/italic emphasis, lists, code blocks, block
//! quotes, and tables. Lines wrap to the page using real glyph metrics — prose
//! on word boundaries, code hard at the margin — across as many pages as needed.
//! A review export (both `/review` and `/auto_review`) adds a final source
//! appendix: the code around each finding (the `/show_file` view, 3 lines before
//! and after) with line numbers, grouped by category. Only the finding's
//! recorded line(s) are syntax-highlighted and bold; the context lines are left
//! plain.

use anyhow::{Context, Result};
use markdown::{
    ParseOptions,
    mdast::{Code, Heading, List, ListItem, Node, Paragraph},
    to_mdast,
};
use printpdf::{
    Actions, BorderArray, BuiltinFont, Color, Line, LinePoint, LinkAnnotation, Mm, Op, PaintMode,
    ParsedFont, PdfDocument, PdfFontHandle, PdfPage, PdfSaveOptions, Point, Pt, Rect, Rgb,
    TextItem,
};
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::OnceLock,
    time::{SystemTime, UNIX_EPOCH},
};

use syntect::easy::HighlightLines;
use syntect::highlighting::{Theme, ThemeSet};

use orangu::tui::TranscriptLine;

use crate::VERSION;
use crate::git::{discover_git_root, git_repository_name, workspace_branch_name};
use crate::render::{SyntaxHighlightAssets, syntax_highlight_assets};

/// One finding in the `/auto_review` source appendix: the category it belongs
/// to, the finding's Markdown text, and the source code around the finding's
/// line (the `/show_file` view — 3 lines before and after), as plain lines with
/// the line number they start at and the file path (for syntax detection).
/// `code` is empty when the finding has no resolvable file or line. Built by
/// `AutoReviewState::export_appendix`.
#[derive(Clone, Debug)]
pub struct AutoReviewAppendixEntry {
    pub category: String,
    pub finding: String,
    /// The file path, for syntax detection (empty when the finding has none).
    pub path: String,
    /// The 1-based line number of the first row of `code` (0 when `code` is empty).
    pub start_line: usize,
    /// The plain source lines around the finding (the ±3-line window).
    pub code: Vec<String>,
    /// The inclusive 1-based line range the finding recorded — the line(s)
    /// syntax-highlighted in the appendix; the surrounding context is left
    /// plain. `None` when the finding carries no line.
    pub highlight: Option<(usize, usize)>,
}

// --- Embedded brand font (Red Hat Text, SIL OFL — see assets/fonts/LICENSE) ---
const FONT_REGULAR: &[u8] = include_bytes!("../../../assets/fonts/RedHatText-Regular.otf");
const FONT_BOLD: &[u8] = include_bytes!("../../../assets/fonts/RedHatText-Bold.otf");
const FONT_ITALIC: &[u8] = include_bytes!("../../../assets/fonts/RedHatText-Italic.otf");
const FONT_BOLD_ITALIC: &[u8] = include_bytes!("../../../assets/fonts/RedHatText-BoldItalic.otf");

// --- Page geometry (A4, in millimetres) ---
const PAGE_WIDTH_MM: f32 = 210.0;
const PAGE_HEIGHT_MM: f32 = 297.0;
const MARGIN_MM: f32 = 18.0;
const USABLE_WIDTH_MM: f32 = PAGE_WIDTH_MM - 2.0 * MARGIN_MM;

const PT_TO_MM: f32 = 25.4 / 72.0;

const BODY_SIZE: f32 = 10.0;
const CODE_SIZE: f32 = 9.0;
const BAND_TEXT_SIZE: f32 = 11.0;
/// Indent (mm) added per level of list/quote nesting.
const INDENT_MM: f32 = 6.0;

// --- Header/footer bands ---
const HEADER_BAND_MM: f32 = 11.0;
const FOOTER_BAND_MM: f32 = 11.0;
/// Gap between a band and the page content.
const CONTENT_GAP_MM: f32 = 5.0;
/// First content baseline / lowest content baseline.
const CONTENT_TOP_MM: f32 = PAGE_HEIGHT_MM - HEADER_BAND_MM - CONTENT_GAP_MM;
const CONTENT_BOTTOM_MM: f32 = FOOTER_BAND_MM + CONTENT_GAP_MM;

/// The orangu brand colour (`ORANGU_BROWN`, rgb 139/90/43): the band fill, plus
/// the title and headings, so the PDF matches the terminal banner.
const BRAND_COLOR: (f32, f32, f32) = (139.0 / 255.0, 90.0 / 255.0, 43.0 / 255.0);
const TEXT_COLOR: (f32, f32, f32) = (0.0, 0.0, 0.0);
const WHITE: (f32, f32, f32) = (1.0, 1.0, 1.0);
/// Overall-status banner colours (the terminal's status green/red).
const STATUS_GREEN: (f32, f32, f32) = (80.0 / 255.0, 200.0 / 255.0, 120.0 / 255.0);
const STATUS_RED: (f32, f32, f32) = (220.0 / 255.0, 80.0 / 255.0, 80.0 / 255.0);
const GRID_COLOR: (f32, f32, f32) = (0.6, 0.6, 0.6);

const ORANGU_URL: &str = "https://mnemosyne-systems.github.io/orangu/";

/// The overall patch verdict shown on the review export's first page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verdict {
    Approved,
    Rejected,
}

/// One run of text with a single style. A style differs by weight and slant,
/// which select a Red Hat Text variant (or its Helvetica fallback).
#[derive(Clone)]
struct Span {
    text: String,
    bold: bool,
    italic: bool,
    /// An explicit RGB fill colour. `None` uses the block's default colour
    /// (brand for headings, black for body); set only for syntax-highlighted
    /// appendix code so its runs carry their own colours.
    color: Option<(f32, f32, f32)>,
}

impl Span {
    fn plain(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            bold: false,
            italic: false,
            color: None,
        }
    }
}

/// A logical line to be laid out: its styled spans, the font size, the leading
/// indent (mm) of the first and continuation (wrapped) lines, and the gap left
/// after it. `word_wrap` breaks on spaces for prose; code lines wrap hard at the
/// margin so their layout is preserved.
#[derive(Clone)]
struct Block {
    spans: Vec<Span>,
    size: f32,
    indent_mm: f32,
    hanging_mm: f32,
    word_wrap: bool,
    space_after_mm: f32,
}

impl Block {
    fn paragraph(spans: Vec<Span>) -> Self {
        Self {
            spans,
            size: BODY_SIZE,
            indent_mm: 0.0,
            hanging_mm: 0.0,
            word_wrap: true,
            space_after_mm: BODY_SIZE * 0.5 * PT_TO_MM,
        }
    }
}

#[derive(Clone, Copy)]
struct StyledChar {
    ch: char,
    bold: bool,
    italic: bool,
    color: Option<(f32, f32, f32)>,
}

/// Export the console output window to a PDF in the workspace root.
pub fn export_console(
    workspace: &Path,
    transcript: &[TranscriptLine],
    model: &str,
) -> Result<PathBuf> {
    let mut blocks = Vec::new();
    if transcript.is_empty() {
        blocks.push(Block::paragraph(vec![Span::plain(
            "(the output window is empty)",
        )]));
    }
    for line in transcript {
        let text = strip_ansi(line.as_str());
        let bold = matches!(line, TranscriptLine::UserInput(_));
        blocks.push(Block {
            spans: vec![Span {
                text,
                bold,
                italic: false,
                color: None,
            }],
            size: BODY_SIZE,
            indent_mm: 0.0,
            hanging_mm: 0.0,
            // Mirror the terminal: hard-wrap long lines instead of reflowing words.
            word_wrap: false,
            space_after_mm: 0.0,
        });
    }
    let mut pdf = Pdf::new(&header_label(workspace), model)?;
    pdf.draw_blocks(&blocks);
    let path = export_file_path(workspace, "console");
    pdf.save(&path)?;
    Ok(path)
}

/// Export the last review report (Markdown) to a PDF in the workspace root.
///
/// The first page is a summary: repository, branch, date/time, an entry count
/// per category, and a green/red Approved/Rejected banner. The second page is a
/// table of contents. Each category then starts on its own page, so the first
/// category (`Overall`) opens on page 3. A non-empty `appendix` (built for both
/// `/review` and `/auto_review`) adds a final source appendix — the code around
/// each finding, the recorded line syntax-highlighted and bold — on its own
/// page, listed in the contents; an empty `appendix` adds none.
pub fn export_review(
    workspace: &Path,
    markdown: &str,
    model: &str,
    appendix: &[AutoReviewAppendixEntry],
) -> Result<PathBuf> {
    let sections = parse_sections(markdown);
    let verdict = overall_verdict(markdown);
    let repository = repository_display(workspace);
    let branch = branch_display(workspace);

    let mut pdf = Pdf::new(&header_label(workspace), model)?;

    // Page 1 — summary.
    pdf.draw_info_page(&repository, &branch, &sections, verdict);

    // The categories start on page 3, each on its own page; compute where each
    // one lands so the table of contents (page 2) can point at it.
    let mut starts = Vec::with_capacity(sections.len());
    let mut page = 3;
    for section in &sections {
        starts.push(page);
        page += paginate(&section.blocks, &pdf.fonts);
    }

    // The appendix (when present) follows the categories on its own page.
    let appendix_blocks = build_appendix_blocks(appendix);
    let appendix_start = (!appendix_blocks.is_empty()).then_some(page);

    // Page 2 — table of contents: every category, then the appendix.
    let mut toc_rows: Vec<(&str, usize)> = sections
        .iter()
        .map(|section| section.title.as_str())
        .zip(starts.iter().copied())
        .collect();
    if let Some(start) = appendix_start {
        toc_rows.push(("Appendix", start));
    }
    pdf.new_page();
    pdf.draw_toc(&toc_rows);

    // Pages 3+ — one category per page, then the appendix.
    for section in &sections {
        pdf.new_page();
        pdf.draw_blocks(&section.blocks);
    }
    if !appendix_blocks.is_empty() {
        pdf.new_page();
        pdf.draw_blocks(&appendix_blocks);
    }

    let path = export_file_path(workspace, "review");
    pdf.save(&path)?;
    Ok(path)
}

/// The `{repository}-{branch}` shown in the page header band, using the real
/// (unsanitized) names.
fn header_label(workspace: &Path) -> String {
    format!(
        "{}-{}",
        repository_display(workspace),
        branch_display(workspace)
    )
}

/// `{repository}-{branch}-{kind}.pdf` in the workspace root, with the names
/// sanitized for use in a filename.
fn export_file_path(workspace: &Path, kind: &str) -> PathBuf {
    let repository = non_empty(sanitize(&repository_display(workspace)), "workspace");
    let branch = non_empty(sanitize(&branch_display(workspace)), "nobranch");
    workspace.join(format!("{repository}-{branch}-{kind}.pdf"))
}

/// The repository name, for display. Taken from the `origin` remote (so a repo
/// cloned into a differently named directory still exports under its own name),
/// falling back to the Git root — else the workspace — directory name.
fn repository_display(workspace: &Path) -> String {
    let root = discover_git_root(workspace).unwrap_or_else(|| workspace.to_path_buf());
    let name = git_repository_name(&root).unwrap_or_else(|| {
        root.file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default()
    });
    non_empty(name, "workspace")
}

/// The current branch name, for display (`nobranch` when not on one).
fn branch_display(workspace: &Path) -> String {
    non_empty(
        workspace_branch_name(workspace).unwrap_or_default(),
        "nobranch",
    )
}

fn non_empty(value: String, fallback: &str) -> String {
    if value.is_empty() {
        fallback.to_string()
    } else {
        value
    }
}

/// Make a string safe for a filename: keep alphanumerics, `-`, `_`, and `.`;
/// turn everything else (including the `/` in `feature/x`) into `-`, and
/// collapse runs of `-`.
fn sanitize(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '.' {
            out.push(ch);
        } else if !out.ends_with('-') {
            out.push('-');
        }
    }
    out.trim_matches('-').to_string()
}

// --- Markdown -> blocks ---

/// A top-level section of the review report (a category), with its rendered
/// blocks and the number of entries (list items) it contains.
struct Section {
    title: String,
    blocks: Vec<Block>,
    entry_count: usize,
}

/// Split the report Markdown into sections at its top-most heading level — for
/// the auto review report that is each `## Category`. Content before the first
/// such heading (or a report with no headings) becomes a single section.
fn parse_sections(markdown: &str) -> Vec<Section> {
    let root = match to_mdast(markdown, &ParseOptions::gfm()) {
        Ok(Node::Root(root)) => root,
        // A document we cannot parse still exports as one verbatim section.
        _ => {
            let blocks = markdown
                .lines()
                .map(|line| Block::paragraph(vec![Span::plain(line)]))
                .collect();
            return vec![Section {
                title: "Review".to_string(),
                blocks,
                entry_count: 0,
            }];
        }
    };

    let min_depth = root
        .children
        .iter()
        .filter_map(|node| match node {
            Node::Heading(heading) => Some(heading.depth),
            _ => None,
        })
        .min();

    let mut sections: Vec<Section> = Vec::new();
    let mut nodes: Vec<&Node> = Vec::new();
    let mut title: Option<String> = None;
    for node in &root.children {
        let is_break = matches!((node, min_depth), (Node::Heading(h), Some(d)) if h.depth == d);
        if is_break {
            if title.is_some() || !nodes.is_empty() {
                sections.push(build_section(title.take(), &nodes));
                nodes.clear();
            }
            if let Node::Heading(heading) = node {
                title = Some(heading_text(&heading.children));
            }
        }
        nodes.push(node);
    }
    if title.is_some() || !nodes.is_empty() {
        sections.push(build_section(title.take(), &nodes));
    }
    if sections.is_empty() {
        sections.push(Section {
            title: "Review".to_string(),
            blocks: Vec::new(),
            entry_count: 0,
        });
    }
    sections
}

fn build_section(title: Option<String>, nodes: &[&Node]) -> Section {
    let mut blocks = Vec::new();
    for node in nodes {
        render_block_node(node, 0, &mut blocks);
    }
    let entry_count = nodes.iter().copied().map(count_list_items).sum();
    Section {
        title: title.unwrap_or_else(|| "Report".to_string()),
        blocks,
        entry_count,
    }
}

/// Count the list items (entries) anywhere within `node`, each once.
fn count_list_items(node: &Node) -> usize {
    let mut count = 0;
    if let Node::List(list) = node {
        count += list
            .children
            .iter()
            .filter(|child| matches!(child, Node::ListItem(_)))
            .count();
    }
    if let Some(children) = node.children() {
        for child in children {
            count += count_list_items(child);
        }
    }
    count
}

/// The plain text of a heading's inline children, for a section title.
fn heading_text(children: &[Node]) -> String {
    let mut spans = Vec::new();
    collect_inline(children, false, false, &mut spans);
    spans
        .into_iter()
        .map(|span| span.text)
        .collect::<String>()
        .trim()
        .to_string()
}

/// The overall patch verdict, read from the report's conclusion text.
fn overall_verdict(markdown: &str) -> Verdict {
    let lower = markdown.to_lowercase();
    if lower.contains("rejects this patch") || lower.contains("patch rejected") {
        Verdict::Rejected
    } else if lower.contains("approves this patch") || lower.contains("patch approved") {
        Verdict::Approved
    } else if lower.contains("reject") {
        Verdict::Rejected
    } else {
        Verdict::Approved
    }
}

/// The current UTC date and time as `YYYY-MM-DD HH:MM:SS UTC`.
fn format_timestamp() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0) as i64;
    let days = secs.div_euclid(86_400);
    let tod = secs.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    let (hour, minute, second) = (tod / 3600, (tod % 3600) / 60, tod % 60);
    format!("{year:04}-{month:02}-{day:02} {hour:02}:{minute:02}:{second:02} UTC")
}

/// Convert a count of days since the Unix epoch to a `(year, month, day)`
/// proleptic-Gregorian date (Howard Hinnant's `civil_from_days`).
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    (year + i64::from(month <= 2), month, day)
}

fn render_block_nodes(nodes: &[Node], level: usize, blocks: &mut Vec<Block>) {
    for node in nodes {
        render_block_node(node, level, blocks);
    }
}

fn render_block_node(node: &Node, level: usize, blocks: &mut Vec<Block>) {
    match node {
        Node::Heading(heading) => blocks.push(heading_block(heading, level)),
        Node::Paragraph(paragraph) => {
            blocks.push(paragraph_block(&paragraph.children, level));
        }
        Node::List(list) => render_list(list, level, blocks),
        Node::Code(code) => render_code(code, level, blocks),
        Node::Blockquote(quote) => render_blockquote(&quote.children, level, blocks),
        Node::ThematicBreak(_) => blocks.push(Block {
            spans: vec![Span::plain("—".repeat(40))],
            size: BODY_SIZE,
            indent_mm: level as f32 * INDENT_MM,
            hanging_mm: level as f32 * INDENT_MM,
            word_wrap: false,
            space_after_mm: BODY_SIZE * 0.5 * PT_TO_MM,
        }),
        Node::Table(_) => render_table(node, level, blocks),
        // Definitions carry no printable text; anything else is treated as inline.
        Node::Definition(_) => {}
        _ => {
            let spans = inline_spans_of(node);
            if !spans.is_empty() {
                blocks.push(paragraph_block_from_spans(spans, level));
            }
        }
    }
}

fn heading_block(heading: &Heading, level: usize) -> Block {
    let size = match heading.depth {
        1 => BODY_SIZE + 5.0,
        2 => BODY_SIZE + 3.5,
        3 => BODY_SIZE + 2.0,
        4 => BODY_SIZE + 1.0,
        _ => BODY_SIZE + 0.5,
    };
    let mut spans = Vec::new();
    collect_inline(&heading.children, true, false, &mut spans);
    Block {
        spans,
        size,
        indent_mm: level as f32 * INDENT_MM,
        hanging_mm: level as f32 * INDENT_MM,
        word_wrap: true,
        space_after_mm: size * 0.45 * PT_TO_MM,
    }
}

fn paragraph_block(children: &[Node], level: usize) -> Block {
    let mut spans = Vec::new();
    collect_inline(children, false, false, &mut spans);
    paragraph_block_from_spans(spans, level)
}

fn paragraph_block_from_spans(spans: Vec<Span>, level: usize) -> Block {
    Block {
        spans,
        indent_mm: level as f32 * INDENT_MM,
        hanging_mm: level as f32 * INDENT_MM,
        ..Block::paragraph(Vec::new())
    }
}

fn render_list(list: &List, level: usize, blocks: &mut Vec<Block>) {
    let mut number = list.start.unwrap_or(1);
    for child in &list.children {
        let Node::ListItem(item) = child else {
            continue;
        };
        let marker = if list.ordered {
            let marker = format!("{number}.");
            number += 1;
            marker
        } else {
            "•".to_string()
        };
        render_list_item(item, &marker, level, blocks);
    }
}

fn render_list_item(item: &ListItem, marker: &str, level: usize, blocks: &mut Vec<Block>) {
    // Continuation lines hang under the text, past an approximate marker width.
    let marker_allowance = (marker.chars().count() + 1) as f32 * BODY_SIZE * 0.5 * PT_TO_MM;
    let indent_mm = level as f32 * INDENT_MM;
    let mut marker_used = false;
    for child in &item.children {
        match child {
            Node::Paragraph(Paragraph { children, .. }) => {
                let mut spans = Vec::new();
                if !marker_used {
                    spans.push(Span::plain(format!("{marker} ")));
                    marker_used = true;
                }
                collect_inline(children, false, false, &mut spans);
                blocks.push(Block {
                    spans,
                    size: BODY_SIZE,
                    indent_mm,
                    hanging_mm: indent_mm + marker_allowance,
                    word_wrap: true,
                    space_after_mm: BODY_SIZE * 0.3 * PT_TO_MM,
                });
            }
            Node::List(sub) => render_list(sub, level + 1, blocks),
            other => render_block_node(other, level + 1, blocks),
        }
    }
    if !marker_used {
        blocks.push(Block {
            spans: vec![Span::plain(marker.to_string())],
            size: BODY_SIZE,
            indent_mm,
            hanging_mm: indent_mm + marker_allowance,
            word_wrap: true,
            space_after_mm: BODY_SIZE * 0.3 * PT_TO_MM,
        });
    }
}

fn render_code(code: &Code, level: usize, blocks: &mut Vec<Block>) {
    let base = level as f32 * INDENT_MM + INDENT_MM;
    let lines: Vec<&str> = if code.value.is_empty() {
        vec![""]
    } else {
        code.value.lines().collect()
    };
    let last = lines.len().saturating_sub(1);
    for (index, line) in lines.iter().enumerate() {
        blocks.push(Block {
            spans: vec![Span::plain((*line).to_string())],
            size: CODE_SIZE,
            indent_mm: base,
            hanging_mm: base,
            word_wrap: false,
            space_after_mm: if index == last {
                CODE_SIZE * 0.5 * PT_TO_MM
            } else {
                0.0
            },
        });
    }
}

fn render_blockquote(children: &[Node], level: usize, blocks: &mut Vec<Block>) {
    let start = blocks.len();
    render_block_nodes(children, level, blocks);
    // Prefix every line the quote produced with a "> " marker span.
    for block in &mut blocks[start..] {
        block.spans.insert(0, Span::plain("> "));
        block.hanging_mm += BODY_SIZE * PT_TO_MM;
    }
}

fn render_table(node: &Node, level: usize, blocks: &mut Vec<Block>) {
    let rendered = crate::render::render_table(node.children().map(Vec::as_slice).unwrap_or(&[]));
    let base = level as f32 * INDENT_MM;
    for line in rendered.lines() {
        blocks.push(Block {
            spans: vec![Span::plain(line.to_string())],
            size: CODE_SIZE,
            indent_mm: base,
            hanging_mm: base,
            word_wrap: false,
            space_after_mm: 0.0,
        });
    }
    if !rendered.is_empty()
        && let Some(last) = blocks.last_mut()
    {
        last.space_after_mm = BODY_SIZE * 0.5 * PT_TO_MM;
    }
}

// --- Auto review source appendix ---

/// Render the `/auto_review` source appendix: an `Appendix` page, then per
/// category that has findings a heading, and under each finding its Markdown
/// text followed by the syntax-highlighted source code around the finding (the
/// `/show_file` view — 3 lines before and after). Empty when there are no
/// entries, so the export adds no appendix for an interactive `/review`.
fn build_appendix_blocks(appendix: &[AutoReviewAppendixEntry]) -> Vec<Block> {
    if appendix.is_empty() {
        return Vec::new();
    }
    // The assets are resolved once; the syntax is per file (so it varies by
    // entry) and is resolved inside `highlight_code_blocks`.
    let assets = syntax_highlight_assets();

    let mut blocks = vec![heading("Appendix", BODY_SIZE + 5.0)];
    let mut current: Option<&str> = None;
    for entry in appendix {
        if current != Some(entry.category.as_str()) {
            blocks.push(heading(&entry.category, BODY_SIZE + 3.5));
            current = Some(entry.category.as_str());
        }
        blocks.push(finding_block(&entry.finding));
        blocks.extend(highlight_code_blocks(entry, assets));
    }
    blocks
}

/// A finding's Markdown text as a body paragraph block, so its `**path:line**`
/// location renders bold like in the report.
fn finding_block(finding: &str) -> Block {
    paragraph_block_from_spans(inline_markdown_spans(finding), 0)
}

/// Parse one line of inline Markdown into styled spans, reusing the report's
/// `collect_inline`. Falls back to a single plain span when it does not parse.
fn inline_markdown_spans(text: &str) -> Vec<Span> {
    match to_mdast(text, &ParseOptions::gfm()) {
        Ok(Node::Root(root)) => {
            let mut spans = Vec::new();
            for node in &root.children {
                if let Node::Paragraph(paragraph) = node {
                    collect_inline(&paragraph.children, false, false, &mut spans);
                }
            }
            if spans.is_empty() {
                vec![Span::plain(text)]
            } else {
                spans
            }
        }
        _ => vec![Span::plain(text)],
    }
}

/// A light syntect theme for the appendix — dark, readable colours on the white
/// page — loaded once. The shared `syntax_highlight_assets` theme is tuned for
/// the dark terminal, so on paper its colours are too faint; this picks a light
/// theme (GitHub-style) instead. `None` when no light theme can be loaded, in
/// which case the recorded line falls back to bold black.
fn appendix_theme() -> Option<&'static Theme> {
    static THEME: OnceLock<Option<Theme>> = OnceLock::new();
    THEME
        .get_or_init(|| {
            let themes = ThemeSet::load_defaults();
            ["InspiredGitHub", "Solarized (light)", "base16-ocean.light"]
                .iter()
                .find_map(|name| themes.themes.get(*name).cloned())
        })
        .as_ref()
}

/// Syntax-highlight a finding's source-code window into code blocks, with a grey
/// line-number gutter like `/show_file`. Only the recorded line is emphasised —
/// drawn bold and, when a light theme is available, syntax-coloured in the
/// file's language (otherwise bold black) — while the surrounding context is
/// left plain. Each line is one hard-wrapped `CODE_SIZE` block, mirroring
/// `render_code`. Empty when the entry has no code.
fn highlight_code_blocks(
    entry: &AutoReviewAppendixEntry,
    assets: &SyntaxHighlightAssets,
) -> Vec<Block> {
    if entry.code.is_empty() {
        return Vec::new();
    }
    let extension = Path::new(&entry.path)
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or("");
    // A light theme keeps the colours readable on the page; without one the
    // recorded line is drawn bold black.
    let mut highlighter = appendix_theme().map(|theme| {
        let syntax = assets
            .syntaxes
            .find_syntax_by_extension(extension)
            .unwrap_or_else(|| assets.syntaxes.find_syntax_plain_text());
        HighlightLines::new(syntax, theme)
    });

    let last_line = entry.start_line + entry.code.len() - 1;
    let gutter_width = last_line.to_string().len();
    let base = INDENT_MM;
    let last = entry.code.len().saturating_sub(1);
    let mut blocks = Vec::with_capacity(entry.code.len());
    for (index, line) in entry.code.iter().enumerate() {
        let line_no = entry.start_line + index;
        let recorded = entry
            .highlight
            .is_some_and(|(start, end)| line_no >= start && line_no <= end);
        // The grey line-number gutter, then the source. Every line is fed to the
        // highlighter so its parse state stays correct; only the recorded line
        // keeps the (bold) syntax colours, the context is left plain.
        let mut spans = vec![Span {
            text: format!("{line_no:>gutter_width$}  "),
            bold: false,
            italic: false,
            color: Some(GRID_COLOR),
        }];
        let ranges = highlighter
            .as_mut()
            .and_then(|highlighter| highlighter.highlight_line(line, &assets.syntaxes).ok());
        match (recorded, ranges) {
            // Recorded line with a light theme: bold, syntax-coloured.
            (true, Some(ranges)) => spans.extend(ranges.iter().map(|(style, text)| Span {
                text: (*text).to_string(),
                bold: true,
                italic: false,
                color: Some(syntect_rgb(style.foreground)),
            })),
            // Recorded line without a theme: bold black so it still stands out.
            (true, None) => spans.push(Span {
                text: line.clone(),
                bold: true,
                italic: false,
                color: None,
            }),
            // Context line: plain.
            (false, _) => spans.push(Span::plain(line.clone())),
        }
        blocks.push(Block {
            spans,
            size: CODE_SIZE,
            indent_mm: base,
            hanging_mm: base,
            word_wrap: false,
            space_after_mm: if index == last {
                CODE_SIZE * 0.5 * PT_TO_MM
            } else {
                0.0
            },
        });
    }
    blocks
}

/// A syntect highlight colour as a 0..1 RGB triple for printpdf.
fn syntect_rgb(color: syntect::highlighting::Color) -> (f32, f32, f32) {
    (
        f32::from(color.r) / 255.0,
        f32::from(color.g) / 255.0,
        f32::from(color.b) / 255.0,
    )
}

fn inline_spans_of(node: &Node) -> Vec<Span> {
    let mut spans = Vec::new();
    collect_inline(std::slice::from_ref(node), false, false, &mut spans);
    spans
}

fn collect_inline(nodes: &[Node], bold: bool, italic: bool, out: &mut Vec<Span>) {
    for node in nodes {
        match node {
            Node::Text(text) => push_span(out, &text.value, bold, italic),
            Node::InlineCode(code) => push_span(out, &code.value, bold, italic),
            Node::InlineMath(math) => push_span(out, &math.value, bold, italic),
            Node::Strong(strong) => collect_inline(&strong.children, true, italic, out),
            Node::Emphasis(emphasis) => collect_inline(&emphasis.children, bold, true, out),
            Node::Delete(delete) => collect_inline(&delete.children, bold, italic, out),
            Node::Link(link) => {
                collect_inline(&link.children, bold, italic, out);
                if !link.url.is_empty() {
                    push_span(out, &format!(" ({})", link.url), bold, italic);
                }
            }
            Node::Image(image) => {
                push_span(
                    out,
                    &format!("[image: {}] ({})", image.alt, image.url),
                    bold,
                    italic,
                );
            }
            Node::Break(_) => push_span(out, " ", bold, italic),
            other => {
                if let Some(children) = other.children() {
                    collect_inline(children, bold, italic, out);
                } else if let Some(value) = inline_value(other) {
                    push_span(out, value, bold, italic);
                }
            }
        }
    }
}

fn inline_value(node: &Node) -> Option<&str> {
    match node {
        Node::Html(html) => Some(&html.value),
        Node::Math(math) => Some(&math.value),
        _ => None,
    }
}

/// Append `text` to `out`, turning newlines into spaces and merging into a
/// trailing span when the style matches.
fn push_span(out: &mut Vec<Span>, text: &str, bold: bool, italic: bool) {
    let normalized = text.replace(['\n', '\r'], " ");
    if normalized.is_empty() {
        return;
    }
    match out.last_mut() {
        Some(last) if last.bold == bold && last.italic == italic => last.text.push_str(&normalized),
        _ => out.push(Span {
            text: normalized,
            bold,
            italic,
            color: None,
        }),
    }
}

// --- Fonts & measurement ---

/// The four faces used for drawing, plus the matching width measurer. Either
/// the embedded Red Hat Text family or, if it cannot be loaded, the builtin
/// Helvetica family.
struct DocFonts {
    regular: PdfFontHandle,
    bold: PdfFontHandle,
    italic: PdfFontHandle,
    bold_italic: PdfFontHandle,
    measurer: Measurer,
}

impl DocFonts {
    fn font(&self, bold: bool, italic: bool) -> &PdfFontHandle {
        match (bold, italic) {
            (true, true) => &self.bold_italic,
            (true, false) => &self.bold,
            (false, true) => &self.italic,
            (false, false) => &self.regular,
        }
    }

    fn char_width_mm(&self, ch: char, bold: bool, italic: bool, size: f32) -> f32 {
        self.measurer.char_width_mm(ch, bold, italic, size)
    }

    fn text_width_mm(&self, text: &str, bold: bool, italic: bool, size: f32) -> f32 {
        text.chars()
            .map(|ch| self.char_width_mm(ch, bold, italic, size))
            .sum()
    }
}

/// Measures glyph advances for line breaking and positioning.
enum Measurer {
    /// Real metrics parsed from the embedded Red Hat Text faces, indexed by
    /// `(bold, italic)` as `bold + italic * 2`.
    Embedded(Box<[ttf_parser::Face<'static>; 4]>),
    /// No font metrics available (Helvetica fallback): approximate every glyph
    /// as half an em, which is close to a proportional sans-serif average.
    Approximate,
}

impl Measurer {
    fn char_width_mm(&self, ch: char, bold: bool, italic: bool, size: f32) -> f32 {
        match self {
            Measurer::Embedded(faces) => {
                let face = &faces[usize::from(bold) + usize::from(italic) * 2];
                let units = f32::from(face.units_per_em());
                let advance = face
                    .glyph_index(ch)
                    .and_then(|glyph| face.glyph_hor_advance(glyph))
                    .map_or(units * 0.5, f32::from);
                advance / units * size * PT_TO_MM
            }
            Measurer::Approximate => size * 0.5 * PT_TO_MM,
        }
    }
}

/// Load the embedded Red Hat Text family into `doc`, falling back to the closest
/// printpdf-native face (Helvetica) if it cannot be embedded or parsed.
fn load_fonts(doc: &mut PdfDocument) -> Result<DocFonts> {
    if let Some(fonts) = load_embedded_fonts(doc) {
        return Ok(fonts);
    }
    Ok(DocFonts {
        regular: PdfFontHandle::Builtin(BuiltinFont::Helvetica),
        bold: PdfFontHandle::Builtin(BuiltinFont::HelveticaBold),
        italic: PdfFontHandle::Builtin(BuiltinFont::HelveticaOblique),
        bold_italic: PdfFontHandle::Builtin(BuiltinFont::HelveticaBoldOblique),
        measurer: Measurer::Approximate,
    })
}

/// Parse and register one embedded face, returning its document font handle.
fn add_external_font(doc: &mut PdfDocument, bytes: &'static [u8]) -> Option<PdfFontHandle> {
    let parsed = ParsedFont::from_bytes(bytes, 0, &mut Vec::new())?;
    Some(PdfFontHandle::External(doc.add_font(&parsed)))
}

fn load_embedded_fonts(doc: &mut PdfDocument) -> Option<DocFonts> {
    let regular = add_external_font(doc, FONT_REGULAR)?;
    let bold = add_external_font(doc, FONT_BOLD)?;
    let italic = add_external_font(doc, FONT_ITALIC)?;
    let bold_italic = add_external_font(doc, FONT_BOLD_ITALIC)?;
    let faces = Box::new([
        ttf_parser::Face::parse(FONT_REGULAR, 0).ok()?,
        ttf_parser::Face::parse(FONT_BOLD, 0).ok()?,
        ttf_parser::Face::parse(FONT_ITALIC, 0).ok()?,
        ttf_parser::Face::parse(FONT_BOLD_ITALIC, 0).ok()?,
    ]);
    Some(DocFonts {
        regular,
        bold,
        italic,
        bold_italic,
        measurer: Measurer::Embedded(faces),
    })
}

// --- Layout & PDF output ---

/// Incremental PDF builder: owns the document and fonts, tracks the current
/// page and the content cursor, and draws every page's header/footer bands.
struct Pdf {
    doc: PdfDocument,
    fonts: DocFonts,
    header: String,
    footer: String,
    /// Drawing operations accumulated for the page currently being built.
    ops: Vec<Op>,
    /// Pages already finished (flushed by `new_page`/`save`), in order.
    pages: Vec<PdfPage>,
    cursor_y: f32,
}

impl Pdf {
    fn new(header: &str, model: &str) -> Result<Self> {
        let mut doc = PdfDocument::new("orangu export");
        let fonts = load_fonts(&mut doc)?;
        let mut pdf = Pdf {
            doc,
            fonts,
            header: header.to_string(),
            footer: format!("orangu {VERSION} ({model})"),
            ops: Vec::new(),
            pages: Vec::new(),
            cursor_y: CONTENT_TOP_MM,
        };
        pdf.draw_furniture();
        Ok(pdf)
    }

    /// Finish the page under construction, pushing its accumulated ops as a new
    /// `PdfPage`.
    fn finish_page(&mut self) {
        let ops = std::mem::take(&mut self.ops);
        self.pages
            .push(PdfPage::new(Mm(PAGE_WIDTH_MM), Mm(PAGE_HEIGHT_MM), ops));
    }

    /// Start a fresh page (with its bands) and reset the content cursor.
    fn new_page(&mut self) {
        self.finish_page();
        self.cursor_y = CONTENT_TOP_MM;
        self.draw_furniture();
    }

    fn draw_blocks(&mut self, blocks: &[Block]) {
        for block in blocks {
            self.draw_block(block);
        }
    }

    fn draw_block(&mut self, block: &Block) {
        let line_height = block.size * 1.35 * PT_TO_MM;
        // Headings (the only blocks larger than the body) carry the brand
        // colour. A span with its own colour (syntax-highlighted appendix code)
        // overrides this per run; everything else uses the block default.
        let default_color = if block.size > BODY_SIZE {
            BRAND_COLOR
        } else {
            TEXT_COLOR
        };
        for (index, line) in wrap_block(block, &self.fonts).into_iter().enumerate() {
            if self.cursor_y - line_height < CONTENT_BOTTOM_MM {
                self.new_page();
            }
            self.cursor_y -= line_height;
            let indent = if index == 0 {
                block.indent_mm
            } else {
                block.hanging_mm
            };
            draw_line(
                &mut self.ops,
                &line,
                indent,
                block.size,
                self.cursor_y,
                &self.fonts,
                default_color,
            );
        }
        self.cursor_y -= block.space_after_mm;
    }

    /// Draw a single line of text at `(x, baseline)` in the given colour.
    fn text(
        &mut self,
        text: &str,
        bold: bool,
        x: f32,
        baseline: f32,
        size: f32,
        color: (f32, f32, f32),
    ) {
        let font = self.fonts.font(bold, false).clone();
        emit_text(&mut self.ops, text, &font, size, x, baseline, color);
    }

    fn fill_rect(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, color: (f32, f32, f32)) {
        let (r, g, b) = color;
        let rect = Rect {
            x: Mm(x0).into(),
            y: Mm(y0).into(),
            width: Mm(x1 - x0).into(),
            height: Mm(y1 - y0).into(),
            mode: Some(PaintMode::Fill),
            winding_order: None,
        };
        self.ops.push(Op::SetFillColor {
            col: Color::Rgb(Rgb::new(r, g, b, None)),
        });
        self.ops.push(Op::DrawPolygon {
            polygon: rect.to_polygon(),
        });
    }

    fn rule(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, color: (f32, f32, f32), thickness: f32) {
        let (r, g, b) = color;
        self.ops.push(Op::SetOutlineColor {
            col: Color::Rgb(Rgb::new(r, g, b, None)),
        });
        self.ops.push(Op::SetOutlineThickness { pt: Pt(thickness) });
        self.ops.push(Op::DrawLine {
            line: Line {
                points: vec![line_point(x0, y0), line_point(x1, y1)],
                is_closed: false,
            },
        });
    }

    /// Page 1: repository / branch / date, the per-category entry counts, and
    /// the green/red Approved/Rejected banner.
    fn draw_info_page(
        &mut self,
        repository: &str,
        branch: &str,
        sections: &[Section],
        verdict: Verdict,
    ) {
        self.draw_block(&heading("Review", BODY_SIZE + 5.0));
        let mut rows = vec![
            ("Repository".to_string(), repository.to_string()),
            ("Branch".to_string(), branch.to_string()),
            ("Generated".to_string(), format_timestamp()),
        ];
        for section in sections {
            rows.push((section.title.clone(), section.entry_count.to_string()));
        }
        self.draw_kv_table(&rows);
        self.draw_status_banner(verdict);
    }

    /// A two-column table: bold labels on the left, values on the right.
    fn draw_kv_table(&mut self, rows: &[(String, String)]) {
        let x0 = MARGIN_MM;
        let label_width = rows
            .iter()
            .map(|(key, _)| self.fonts.text_width_mm(key, true, false, BODY_SIZE))
            .fold(0.0_f32, f32::max);
        let divider = x0 + 6.0 + label_width + 6.0;
        let x1 = (divider + 70.0).min(PAGE_WIDTH_MM - MARGIN_MM);
        let row_height = BODY_SIZE * 1.9 * PT_TO_MM;
        let top = self.cursor_y;
        for (index, (key, value)) in rows.iter().enumerate() {
            let baseline = top - (index as f32 + 0.68) * row_height;
            self.text(key, true, x0 + 3.0, baseline, BODY_SIZE, TEXT_COLOR);
            self.text(value, false, divider + 3.0, baseline, BODY_SIZE, TEXT_COLOR);
        }
        let bottom = top - rows.len() as f32 * row_height;
        for index in 0..=rows.len() {
            let y = top - index as f32 * row_height;
            self.rule(x0, y, x1, y, GRID_COLOR, 0.4);
        }
        self.rule(x0, top, x0, bottom, GRID_COLOR, 0.4);
        self.rule(divider, top, divider, bottom, GRID_COLOR, 0.4);
        self.rule(x1, top, x1, bottom, GRID_COLOR, 0.4);
        self.cursor_y = bottom - 8.0;
    }

    fn draw_status_banner(&mut self, verdict: Verdict) {
        let (label, color) = match verdict {
            Verdict::Approved => ("Approved", STATUS_GREEN),
            Verdict::Rejected => ("Rejected", STATUS_RED),
        };
        let height = 16.0;
        let top = self.cursor_y - 4.0;
        let bottom = top - height;
        self.fill_rect(MARGIN_MM, bottom, PAGE_WIDTH_MM - MARGIN_MM, top, color);
        let size = 16.0;
        let width = self.fonts.text_width_mm(label, true, false, size);
        let x = ((PAGE_WIDTH_MM - width) / 2.0).max(MARGIN_MM);
        let cap = size * 0.7 * PT_TO_MM;
        self.text(
            label,
            true,
            x,
            bottom + height / 2.0 - cap / 2.0,
            size,
            WHITE,
        );
        self.cursor_y = bottom - 8.0;
    }

    /// Page 2: the table of contents — each entry (categories, then the
    /// appendix) and its starting page.
    fn draw_toc(&mut self, rows: &[(&str, usize)]) {
        self.draw_block(&heading("Table of Contents", BODY_SIZE + 5.0));
        let size = BODY_SIZE + 1.0;
        let row_height = size * 1.8 * PT_TO_MM;
        for &(title, page) in rows {
            if self.cursor_y - row_height < CONTENT_BOTTOM_MM {
                self.new_page();
            }
            self.cursor_y -= row_height;
            self.text(title, false, MARGIN_MM, self.cursor_y, size, TEXT_COLOR);
            let number = page.to_string();
            let width = self.fonts.text_width_mm(&number, false, false, size);
            self.text(
                &number,
                false,
                PAGE_WIDTH_MM - MARGIN_MM - width,
                self.cursor_y,
                size,
                TEXT_COLOR,
            );
        }
    }

    /// Draw the header and footer bands (brand fill, centered white text) on the
    /// current page, with the footer's `orangu` underlined and linked.
    fn draw_furniture(&mut self) {
        self.fill_rect(
            0.0,
            PAGE_HEIGHT_MM - HEADER_BAND_MM,
            PAGE_WIDTH_MM,
            PAGE_HEIGHT_MM,
            BRAND_COLOR,
        );
        self.fill_rect(0.0, 0.0, PAGE_WIDTH_MM, FOOTER_BAND_MM, BRAND_COLOR);

        let cap = BAND_TEXT_SIZE * 0.7 * PT_TO_MM;
        let header_baseline = (PAGE_HEIGHT_MM - HEADER_BAND_MM / 2.0) - cap / 2.0;
        let footer_baseline = FOOTER_BAND_MM / 2.0 - cap / 2.0;

        // Center the band text horizontally.
        let header_width = self
            .fonts
            .text_width_mm(&self.header, true, false, BAND_TEXT_SIZE);
        let header_x = ((PAGE_WIDTH_MM - header_width) / 2.0).max(MARGIN_MM);
        let footer_width = self
            .fonts
            .text_width_mm(&self.footer, false, false, BAND_TEXT_SIZE);
        let footer_x = ((PAGE_WIDTH_MM - footer_width) / 2.0).max(MARGIN_MM);

        let header = self.header.clone();
        let footer = self.footer.clone();
        self.text(
            &header,
            true,
            header_x,
            header_baseline,
            BAND_TEXT_SIZE,
            WHITE,
        );
        self.text(
            &footer,
            false,
            footer_x,
            footer_baseline,
            BAND_TEXT_SIZE,
            WHITE,
        );

        // Underline "orangu" (it opens the footer) and make it a link.
        let orangu_width = self
            .fonts
            .text_width_mm("orangu", false, false, BAND_TEXT_SIZE);
        let (wr, wg, wb) = WHITE;
        self.ops.push(Op::SetOutlineColor {
            col: Color::Rgb(Rgb::new(wr, wg, wb, None)),
        });
        self.ops.push(Op::SetOutlineThickness { pt: Pt(0.6) });
        self.ops.push(Op::DrawLine {
            line: Line {
                points: vec![
                    line_point(footer_x, footer_baseline - 1.2),
                    line_point(footer_x + orangu_width, footer_baseline - 1.2),
                ],
                is_closed: false,
            },
        });
        self.ops.push(Op::LinkAnnotation {
            link: LinkAnnotation::new(
                Rect {
                    x: Mm(footer_x).into(),
                    y: Mm(1.0).into(),
                    width: Mm(orangu_width).into(),
                    height: Mm(FOOTER_BAND_MM - 2.0).into(),
                    mode: None,
                    winding_order: None,
                },
                Actions::uri(ORANGU_URL.to_string()),
                Some(BorderArray::Solid([0.0, 0.0, 0.0])),
                None,
                None,
            ),
        });
    }

    fn save(mut self, path: &Path) -> Result<()> {
        self.finish_page();
        let pages = std::mem::take(&mut self.pages);
        self.doc.with_pages(pages);
        let bytes = self.doc.save(&PdfSaveOptions::default(), &mut Vec::new());
        let file =
            File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
        BufWriter::new(file)
            .write_all(&bytes)
            .with_context(|| format!("failed to write {}", path.display()))?;
        Ok(())
    }
}

/// A synthetic heading block (used for the info and contents pages).
fn heading(text: &str, size: f32) -> Block {
    Block {
        spans: vec![Span {
            text: text.to_string(),
            bold: true,
            italic: false,
            color: None,
        }],
        size,
        indent_mm: 0.0,
        hanging_mm: 0.0,
        word_wrap: true,
        space_after_mm: size * 0.6 * PT_TO_MM,
    }
}

/// The number of pages `blocks` occupy when laid out from a fresh page — used
/// to compute table-of-contents page numbers without drawing.
fn paginate(blocks: &[Block], fonts: &DocFonts) -> usize {
    let mut pages = 1;
    let mut cursor_y = CONTENT_TOP_MM;
    for block in blocks {
        let line_height = block.size * 1.35 * PT_TO_MM;
        for _ in 0..wrap_block(block, fonts).len() {
            if cursor_y - line_height < CONTENT_BOTTOM_MM {
                pages += 1;
                cursor_y = CONTENT_TOP_MM;
            }
            cursor_y -= line_height;
        }
        cursor_y -= block.space_after_mm;
    }
    pages
}

/// Wrap a block into visual lines at the page width.
fn wrap_block(block: &Block, fonts: &DocFonts) -> Vec<Vec<StyledChar>> {
    let first_width = (USABLE_WIDTH_MM - block.indent_mm).max(1.0);
    let cont_width = (USABLE_WIDTH_MM - block.hanging_mm).max(1.0);
    wrap(
        &block_chars(block),
        first_width,
        cont_width,
        block.word_wrap,
        fonts,
        block.size,
    )
}

fn block_chars(block: &Block) -> Vec<StyledChar> {
    let mut chars = Vec::new();
    for span in &block.spans {
        for ch in span.text.chars() {
            chars.push(StyledChar {
                ch,
                bold: span.bold,
                italic: span.italic,
                color: span.color,
            });
        }
    }
    chars
}

/// Break a styled line into visual lines that fit the page width (mm). The first
/// line uses `first_width`, the rest `cont_width`. With `word_wrap`, breaks fall
/// on spaces (an over-long word is split hard); without it, every line is cut
/// hard at the width.
fn wrap(
    chars: &[StyledChar],
    first_width: f32,
    cont_width: f32,
    word_wrap: bool,
    fonts: &DocFonts,
    size: f32,
) -> Vec<Vec<StyledChar>> {
    let width_of = |sc: &StyledChar| fonts.char_width_mm(sc.ch, sc.bold, sc.italic, size);

    let mut lines: Vec<Vec<StyledChar>> = Vec::new();
    let mut line: Vec<StyledChar> = Vec::new();
    let mut line_width = 0.0_f32;
    let mut last_space: Option<usize> = None;
    let mut budget = first_width;

    let mut push_line = |line: &mut Vec<StyledChar>, carry: &[StyledChar]| {
        lines.push(std::mem::take(line));
        line.extend_from_slice(carry);
    };

    for &sc in chars {
        let w = width_of(&sc);
        if !line.is_empty() && line_width + w > budget {
            match (word_wrap, last_space) {
                // Break at the last space: the text after it carries to the next line.
                (true, Some(at)) if at > 0 => {
                    let carry: Vec<StyledChar> = line[at + 1..].to_vec();
                    line.truncate(at);
                    push_line(&mut line, &carry);
                }
                // No usable break point: cut hard before this character.
                _ => push_line(&mut line, &[]),
            }
            budget = cont_width;
            line_width = line.iter().map(width_of).sum();
            last_space = line.iter().rposition(|c| c.ch == ' ');
        }
        if sc.ch == ' ' {
            last_space = Some(line.len());
        }
        line.push(sc);
        line_width += w;
    }
    if !line.is_empty() || lines.is_empty() {
        lines.push(line);
    }
    lines
}

/// The style of one drawn run: weight, slant, and resolved RGB fill colour.
/// Runs break whenever any of these change.
type RunStyle = (bool, bool, (f32, f32, f32));

fn draw_line(
    ops: &mut Vec<Op>,
    line: &[StyledChar],
    indent_mm: f32,
    size: f32,
    y: f32,
    fonts: &DocFonts,
    default_color: (f32, f32, f32),
) {
    let mut x = MARGIN_MM + indent_mm;
    let mut run = String::new();
    let mut run_style: Option<RunStyle> = None;

    for sc in line {
        // A span's own colour (highlighted code) wins; everything else takes
        // the block default resolved here, so the run carries a concrete colour.
        let style = (sc.bold, sc.italic, sc.color.unwrap_or(default_color));
        if run_style != Some(style) {
            x = flush_run(ops, &mut run, x, y, size, run_style, fonts);
            run_style = Some(style);
        }
        run.push(sc.ch);
    }
    flush_run(ops, &mut run, x, y, size, run_style, fonts);
}

/// Draw `run` at `(x, y)` (mm) in its style's colour and return the x just past
/// it.
fn flush_run(
    ops: &mut Vec<Op>,
    run: &mut String,
    x: f32,
    y: f32,
    size: f32,
    style: Option<RunStyle>,
    fonts: &DocFonts,
) -> f32 {
    if run.is_empty() {
        return x;
    }
    let (bold, italic, color) = style.unwrap_or((false, false, TEXT_COLOR));
    let width = fonts.text_width_mm(run, bold, italic, size);
    emit_text(ops, run, fonts.font(bold, italic), size, x, y, color);
    run.clear();
    x + width
}

/// A non-bezier line vertex at `(x, y)` millimetres.
fn line_point(x: f32, y: f32) -> LinePoint {
    LinePoint {
        p: Point::new(Mm(x), Mm(y)),
        bezier: false,
    }
}

/// Append the ops drawing one absolutely-positioned text run: a self-contained
/// text section (`BT … ET`) whose single `Td` places the baseline at `(x, y)`
/// millimetres from the page's bottom-left. Each run is its own section so the
/// text matrix resets and `Td` acts as an absolute move.
fn emit_text(
    ops: &mut Vec<Op>,
    text: &str,
    font: &PdfFontHandle,
    size: f32,
    x: f32,
    y: f32,
    color: (f32, f32, f32),
) {
    if text.is_empty() {
        return;
    }
    let (r, g, b) = color;
    ops.push(Op::StartTextSection);
    ops.push(Op::SetFillColor {
        col: Color::Rgb(Rgb::new(r, g, b, None)),
    });
    ops.push(Op::SetFont {
        font: font.clone(),
        size: Pt(size),
    });
    ops.push(Op::SetTextCursor {
        pos: Point::new(Mm(x), Mm(y)),
    });
    ops.push(Op::ShowText {
        items: vec![TextItem::Text(text.to_string())],
    });
    ops.push(Op::EndTextSection);
}

/// Remove ANSI escape sequences (CSI, plus the `ESC O x` cursor keys) so the
/// console export shows the plain text the terminal rendered.
fn strip_ansi(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars().peekable();
    'outer: while let Some(ch) = chars.next() {
        if ch == '\x1b' {
            match chars.peek() {
                Some(&'[') => {
                    chars.next();
                    loop {
                        match chars.next() {
                            Some(c) if c.is_ascii_alphabetic() || c == '~' || c == '@' => break,
                            Some(_) => {}
                            None => break 'outer,
                        }
                    }
                }
                Some(&'O') => {
                    chars.next();
                    chars.next();
                }
                // An OSC sequence (e.g. an OSC 8 hyperlink) prints nothing, so
                // drop it: `ESC ] ... ST`, terminated by BEL or `ESC \`. The
                // hyperlink's visible label is left in place.
                Some(&']') => {
                    chars.next();
                    loop {
                        match chars.next() {
                            Some('\x07') => break,
                            Some('\x1b') => {
                                if chars.peek() == Some(&'\\') {
                                    chars.next();
                                }
                                break;
                            }
                            Some(_) => {}
                            None => break 'outer,
                        }
                    }
                }
                _ => {}
            }
            continue;
        }
        out.push(ch);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn sanitize_replaces_path_separators() {
        assert_eq!(sanitize("feature/my-pr"), "feature-my-pr");
        assert_eq!(sanitize("a  b"), "a-b");
        assert_eq!(sanitize("--weird--"), "weird");
        assert_eq!(sanitize("v0.7.0"), "v0.7.0");
    }

    #[test]
    fn strip_ansi_removes_styling() {
        assert_eq!(strip_ansi("\x1b[1mbold\x1b[22m text"), "bold text");
        assert_eq!(strip_ansi("\x1b[38;2;1;2;3mx\x1b[0m"), "x");
        assert_eq!(strip_ansi("plain"), "plain");
    }

    #[test]
    fn embedded_fonts_measure_real_glyph_widths() {
        // The Red Hat Text faces must parse and report sane advances: a wide 'm'
        // is wider than a narrow 'i'.
        let faces = match load_test_measurer() {
            Measurer::Embedded(faces) => faces,
            Measurer::Approximate => panic!("embedded Red Hat Text must parse"),
        };
        let measurer = Measurer::Embedded(faces);
        let m = measurer.char_width_mm('m', false, false, 10.0);
        let i = measurer.char_width_mm('i', false, false, 10.0);
        assert!(m > i, "'m' ({m}) should be wider than 'i' ({i})");
        assert!(i > 0.0);
    }

    fn load_test_measurer() -> Measurer {
        match ttf_parser::Face::parse(FONT_REGULAR, 0) {
            Ok(_) => Measurer::Embedded(Box::new([
                ttf_parser::Face::parse(FONT_REGULAR, 0).unwrap(),
                ttf_parser::Face::parse(FONT_BOLD, 0).unwrap(),
                ttf_parser::Face::parse(FONT_ITALIC, 0).unwrap(),
                ttf_parser::Face::parse(FONT_BOLD_ITALIC, 0).unwrap(),
            ])),
            Err(_) => Measurer::Approximate,
        }
    }

    #[test]
    fn export_console_writes_a_pdf() {
        let workspace = tempdir().expect("workspace");
        let transcript = vec![
            TranscriptLine::UserInput("> hello".to_string()),
            TranscriptLine::Plain("\x1b[1mworld\x1b[22m".to_string()),
        ];
        let path = export_console(workspace.path(), &transcript, "test-model").expect("export");
        assert!(path.exists());
        assert!(
            path.file_name()
                .unwrap()
                .to_string_lossy()
                .ends_with("-console.pdf")
        );
        let bytes = std::fs::read(&path).expect("read pdf");
        assert!(bytes.starts_with(b"%PDF"));
    }

    #[test]
    fn export_review_renders_markdown() {
        let workspace = tempdir().expect("workspace");
        let markdown = "# Title\n\nSome **bold** text.\n\n- one\n- two\n\n```\ncode\n```";
        let path = export_review(workspace.path(), markdown, "gemma", &[]).expect("export");
        assert!(path.exists());
        assert!(
            path.file_name()
                .unwrap()
                .to_string_lossy()
                .ends_with("-review.pdf")
        );
        let bytes = std::fs::read(&path).expect("read pdf");
        assert!(bytes.starts_with(b"%PDF"));
    }

    #[test]
    fn export_review_with_appendix_writes_a_pdf() {
        let workspace = tempdir().expect("workspace");
        let markdown = "## Code\n\n- **a.rs:2**: boom\n\n## Conclusion\n\n\
                        **orangu rejects this patch**";
        let appendix = vec![AutoReviewAppendixEntry {
            category: "Code".to_string(),
            finding: "**a.rs:2**: boom".to_string(),
            path: "a.rs".to_string(),
            start_line: 1,
            code: vec![
                "fn a() {}".to_string(),
                "fn b() {}".to_string(),
                "fn c() {}".to_string(),
            ],
            highlight: Some((2, 2)),
        }];
        let path = export_review(workspace.path(), markdown, "model", &appendix).expect("export");
        assert!(path.exists());
        let bytes = std::fs::read(&path).expect("read pdf");
        assert!(bytes.starts_with(b"%PDF"));
    }

    #[test]
    fn build_appendix_blocks_groups_by_category_with_heading() {
        // No entries: no appendix at all (an interactive `/review`).
        assert!(build_appendix_blocks(&[]).is_empty());

        let appendix = vec![
            AutoReviewAppendixEntry {
                category: "Code".to_string(),
                finding: "**a.rs:2**: boom".to_string(),
                path: "a.rs".to_string(),
                start_line: 1,
                code: vec!["fn b() {}".to_string()],
                highlight: Some((2, 2)),
            },
            AutoReviewAppendixEntry {
                category: "Code".to_string(),
                finding: "**a.rs:5**: bang".to_string(),
                path: "a.rs".to_string(),
                start_line: 4,
                code: vec!["fn c() {}".to_string()],
                highlight: Some((5, 5)),
            },
        ];
        let blocks = build_appendix_blocks(&appendix);
        // The "Appendix" title plus one "Code" heading (shared by both
        // findings, not repeated), each finding, and each code line.
        let headings: Vec<&str> = blocks
            .iter()
            .filter(|block| block.size > BODY_SIZE)
            .flat_map(|block| block.spans.iter().map(|span| span.text.as_str()))
            .collect();
        assert_eq!(headings, vec!["Appendix", "Code"]);
        // A code span carries an explicit colour (syntax highlighting).
        assert!(
            blocks
                .iter()
                .flat_map(|block| &block.spans)
                .any(|span| span.color.is_some())
        );
    }

    #[test]
    fn build_appendix_blocks_highlights_only_the_recorded_line() {
        // A window of `let` lines; only line 2 (the recorded line) is
        // syntax-highlighted, the ±context lines stay plain.
        let appendix = vec![AutoReviewAppendixEntry {
            category: "Code".to_string(),
            finding: "**a.rs:2**: boom".to_string(),
            path: "a.rs".to_string(),
            start_line: 1,
            code: vec![
                "let a = 1;".to_string(),
                "let b = 2;".to_string(),
                "let c = 3;".to_string(),
            ],
            highlight: Some((2, 2)),
        }];
        let blocks = build_appendix_blocks(&appendix);
        // The code-line blocks are the `CODE_SIZE` ones (not the headings or the
        // body-size finding line), in source order.
        let code_blocks: Vec<&Block> = blocks
            .iter()
            .filter(|block| (block.size - CODE_SIZE).abs() < f32::EPSILON)
            .collect();
        assert_eq!(code_blocks.len(), 3);
        // Each line carries a grey gutter span; a line is "highlighted" when its
        // code spans (beyond that gutter) are coloured and bold.
        let highlighted = |block: &Block| {
            block
                .spans
                .iter()
                .skip(1)
                .any(|s| s.color.is_some() && s.bold)
        };
        assert!(!highlighted(code_blocks[0]), "context line 1 must be plain");
        assert!(
            highlighted(code_blocks[1]),
            "recorded line 2 must be highlighted and bold"
        );
        assert!(!highlighted(code_blocks[2]), "context line 3 must be plain");
        // Context lines stay non-bold.
        assert!(
            code_blocks[0].spans.iter().all(|s| !s.bold),
            "context line 1 must not be bold"
        );
    }

    #[test]
    fn appendix_uses_a_light_theme_with_dark_text() {
        // A light theme must load so the recorded line's colours are readable on
        // the white page (the dark terminal theme would be too faint).
        let theme = appendix_theme().expect("a light syntect theme is available");
        // Its default foreground is dark (well under mid-grey), not light.
        let fg = theme.settings.foreground.expect("theme has a foreground");
        let brightness = (u32::from(fg.r) + u32::from(fg.g) + u32::from(fg.b)) / 3;
        assert!(
            brightness < 128,
            "foreground should be dark, got {brightness}"
        );
    }

    #[test]
    fn parse_sections_splits_categories_and_counts_entries() {
        let report = "## Overall\n\nNo issues found\n\n## Code\n\n\
                      - src/main.rs:1: a\n- src/main.rs:2: b\n\n## Conclusion\n\n\
                      **orangu rejects this patch**";
        let sections = parse_sections(report);
        let titles: Vec<&str> = sections.iter().map(|s| s.title.as_str()).collect();
        assert_eq!(titles, vec!["Overall", "Code", "Conclusion"]);
        assert_eq!(sections[0].entry_count, 0);
        assert_eq!(sections[1].entry_count, 2);
        // Each category's blocks lead with its bold, larger heading.
        assert!(
            sections[1]
                .blocks
                .iter()
                .any(|b| b.size > BODY_SIZE && b.spans.iter().any(|s| s.bold))
        );
    }

    #[test]
    fn overall_verdict_reads_the_conclusion() {
        assert_eq!(
            overall_verdict("## Conclusion\n\n**orangu approves this patch**"),
            Verdict::Approved
        );
        assert_eq!(
            overall_verdict("## Conclusion\n\n**orangu rejects this patch**"),
            Verdict::Rejected
        );
        assert_eq!(overall_verdict("**Patch approved**"), Verdict::Approved);
        assert_eq!(overall_verdict("**Patch rejected**"), Verdict::Rejected);
    }

    #[test]
    fn export_review_with_unreviewed_files_is_rejected() {
        use orangu::tui::{ReviewEntry, ReviewStatus};

        // A `/review` (or `/auto_review`) the user left with files not reviewed
        // still has a report to export, and that report rejects the patch.
        let files = vec![
            ReviewEntry {
                path: "a.txt".to_string(),
                status: ReviewStatus::Approved,
                diff_lines: vec!["+x".to_string()],
                patch: String::new(),
            },
            ReviewEntry {
                path: "b.txt".to_string(),
                status: ReviewStatus::Unreviewed,
                diff_lines: vec!["+y".to_string()],
                patch: String::new(),
            },
        ];
        let (_lines, markdown) = crate::review::review_exit_output(&files, &[], &[]);
        assert!(markdown.contains("Not reviewed: **b.txt**"));
        // A file left unreviewed rejects the patch (so the banner is red).
        assert_eq!(overall_verdict(&markdown), Verdict::Rejected);

        // The export still succeeds and writes a PDF.
        let workspace = tempdir().expect("workspace");
        let path = export_review(workspace.path(), &markdown, "model", &[]).expect("export");
        assert!(path.exists());
        let bytes = std::fs::read(&path).expect("read pdf");
        assert!(bytes.starts_with(b"%PDF"));
    }

    #[test]
    fn civil_from_days_matches_known_dates() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        // 2021-01-01 is 18628 days after the epoch.
        assert_eq!(civil_from_days(18628), (2021, 1, 1));
    }
}
