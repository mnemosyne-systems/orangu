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

//! The project's own documents, built from their Markdown sources by the same
//! engine that draws the PDFs `/export` writes — the page bands, the brand
//! colour and the embedded Red Hat Text faces all come from [`crate::export`],
//! so the cheat sheet and a review report are visibly the same family.
//!
//! This is a development tool for this repository, reached through the hidden
//! `--build-cheatsheet` and `--build-manual` flags and driven by
//! `doc/build.sh`; it is not part of what orangu does for a workspace. Both
//! documents are drawn with printpdf, like every other PDF the project
//! produces — there is no LaTeX, and no second toolchain, in the PDF path.
//!
//! # The cheat sheet's Markdown
//!
//! `doc/cheatsheet/en` holds one file per page, named `??-*.md` and built in
//! that order. Within a file:
//!
//! - `# Title` opens a **box** — one focus, with a brand title bar — that runs
//!   until the next `# Title` or the end of the file.
//! - `## Title` is a bold subheading inside the current box.
//! - A paragraph is prose, wrapped to the box.
//! - A two-column table is a **command list**: the left cell is the command,
//!   the right what it does. The header row is not drawn (`Command | What it
//!   does` above every list is noise on a card), but Markdown requires one.
//! - A fenced code block is drawn verbatim on the code tint.
//! - A bullet list is drawn with bullets.
//!
//! A page holds what it holds: if a file's boxes do not fit on one page the
//! build fails, naming the file, rather than spilling onto a fifth page.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use markdown::mdast::{Node, Table};

use crate::export::{
    BRAND_COLOR, Block, CODE_SIZE, CONTENT_BOTTOM_MM, CONTENT_TOP_MM, MARGIN_MM, PAGE_HEIGHT_MM,
    PAGE_WIDTH_MM, PT_TO_MM, Pdf, Span, TEXT_COLOR, USABLE_WIDTH_MM, WHITE, inline_spans_of,
    parse_markdown, render_block_nodes,
};

/// The band text: the reports put `{repository}-{branch}` here.
const HEADER: &str = "orangu-cheatsheet";
/// The footer band, and the site the whole line links to.
const FOOTER: &str = "2026 mnemosyne-systems.ai";
const FOOTER_URL: &str = "https://mnemosyne-systems.ai/";

/// The box's title bar: its height, and the inset of the title text.
const TITLE_BAR_MM: f32 = 7.0;
const TITLE_SIZE: f32 = 12.0;
/// Padding between a box's edge and its content, and the gap between boxes.
const BOX_PAD_MM: f32 = 3.0;
const BOX_GAP_MM: f32 = 4.0;
/// The card sets its text a little smaller than a report's.
const CARD_SIZE: f32 = 9.4;
const SUB_SIZE: f32 = 10.4;
/// The gap between a command and what it does.
const COLUMN_GAP_MM: f32 = 4.0;
/// The command column is sized to its content, within these bounds (as a
/// fraction of the box's inner width), so a box of long commands and a box of
/// short ones both read well.
const COMMAND_COLUMN_MIN: f32 = 0.20;
const COMMAND_COLUMN_MAX: f32 = 0.46;

/// The tints derived from the brand colour: a box's fill and rule, and the
/// ground a code block sits on.
const BOX_BG: (f32, f32, f32) = (0.984, 0.969, 0.949);
const BOX_RULE: (f32, f32, f32) = (0.851, 0.776, 0.690);
const CODE_BG: (f32, f32, f32) = (0.953, 0.918, 0.878);

/// Build `source_dir`'s Markdown into the cheat sheet at `output`, one page per
/// source file. Returns the path written.
pub fn build_cheatsheet(source_dir: &Path, output: &Path) -> Result<PathBuf> {
    let sources = source_files(source_dir)?;
    let pages: Vec<Page> = sources
        .iter()
        .map(|path| parse_page(path))
        .collect::<Result<_>>()?;

    let mut pdf = Pdf::with_footer(HEADER, FOOTER, (FOOTER, FOOTER_URL))?;
    for (index, page) in pages.iter().enumerate() {
        if index > 0 {
            pdf.new_page();
        }
        draw_page(&mut pdf, page)?;
    }

    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    pdf.save(output)?;
    Ok(output.to_path_buf())
}

/// The `??-*.md` sources of a document, in name order — one page each.
fn source_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut sources: Vec<PathBuf> = fs::read_dir(dir)
        .with_context(|| format!("failed to read {}", dir.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.extension().is_some_and(|ext| ext == "md")
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.len() > 3 && name.as_bytes()[2] == b'-')
        })
        .collect();
    sources.sort();
    if sources.is_empty() {
        bail!("no sources found in {} matching ??-*.md", dir.display());
    }
    Ok(sources)
}

/// One page: the file it came from, and the boxes on it.
struct Page {
    name: String,
    boxes: Vec<Focus>,
}

/// One box: a title bar and the items under it.
struct Focus {
    title: String,
    items: Vec<Item>,
}

/// What a box can hold.
enum Item {
    /// A bold subheading (`## Title`).
    Sub(String),
    /// A paragraph.
    Prose(Vec<Span>),
    /// A command list: (command, what it does) per row.
    Commands(Vec<(Vec<Span>, Vec<Span>)>),
    /// A bullet list.
    Bullets(Vec<Vec<Span>>),
    /// A fenced code block, drawn verbatim.
    Code(Vec<String>),
}

fn parse_page(path: &Path) -> Result<Page> {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_string();
    let markdown =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let root = parse_markdown(&markdown);
    let children = root.children().map(Vec::as_slice).unwrap_or(&[]);

    let mut boxes: Vec<Focus> = Vec::new();
    for node in children {
        match node {
            Node::Heading(heading) if heading.depth == 1 => boxes.push(Focus {
                title: plain_text(&inline_spans_of(node)),
                items: Vec::new(),
            }),
            other => {
                let Some(current) = boxes.last_mut() else {
                    bail!(
                        "{name}: content before the first `# ` heading — every page starts a box"
                    );
                };
                if let Some(item) = parse_item(other) {
                    current.items.push(item);
                }
            }
        }
    }

    if boxes.is_empty() {
        bail!("{name}: no `# ` heading, so the page has no box");
    }
    Ok(Page { name, boxes })
}

fn parse_item(node: &Node) -> Option<Item> {
    match node {
        Node::Heading(_) => Some(Item::Sub(plain_text(&inline_spans_of(node)))),
        Node::Paragraph(paragraph) => Some(Item::Prose(spans_of(&paragraph.children))),
        Node::Table(table) => Some(Item::Commands(command_rows(table))),
        Node::List(list) => Some(Item::Bullets(
            list.children
                .iter()
                .map(|item| spans_of_node(item))
                .collect(),
        )),
        Node::Code(code) => Some(Item::Code(code.value.lines().map(str::to_string).collect())),
        _ => None,
    }
}

/// The body rows of a two-column table, as (command, description). The header
/// row is dropped: Markdown needs it, the card does not.
fn command_rows(table: &Table) -> Vec<(Vec<Span>, Vec<Span>)> {
    table
        .children
        .iter()
        .skip(1)
        .filter_map(|row| {
            let cells = row.children()?;
            let command = spans_of_node(cells.first()?);
            let description = cells.get(1).map(spans_of_node).unwrap_or_default();
            Some((command, description))
        })
        .collect()
}

/// A table cell's or list item's inline content as spans. Inline code is set
/// bold: the engine embeds no monospaced face, and a command has to stand out
/// from the prose around it.
fn spans_of_node(node: &Node) -> Vec<Span> {
    let children = node.children().map(Vec::as_slice).unwrap_or(&[]);
    if children.len() == 1
        && let Some(Node::Paragraph(paragraph)) = children.first()
    {
        return spans_of(&paragraph.children);
    }
    spans_of(children)
}

fn spans_of(nodes: &[Node]) -> Vec<Span> {
    let mut spans = Vec::new();
    for node in nodes {
        match node {
            Node::InlineCode(code) => spans.push(Span::styled(&code.value, true, false)),
            other => spans.extend(inline_spans_of(other)),
        }
    }
    spans
}

fn plain_text(spans: &[Span]) -> String {
    spans.iter().map(Span::text).collect()
}

// --- Layout ---

/// Draw a page's boxes down the page, failing if they do not fit on it.
fn draw_page(pdf: &mut Pdf, page: &Page) -> Result<()> {
    let mut y = CONTENT_TOP_MM;
    for focus in &page.boxes {
        y = draw_focus(pdf, focus, y);
        y -= BOX_GAP_MM;
    }
    if y < CONTENT_BOTTOM_MM {
        bail!(
            "{}: its boxes are {:.0} mm past the bottom of the page — tighten the text",
            page.name,
            CONTENT_BOTTOM_MM - y
        );
    }
    Ok(())
}

/// Draw one box with its top edge at `top`, returning the bottom edge.
fn draw_focus(pdf: &mut Pdf, focus: &Focus, top: f32) -> f32 {
    let inner_width = USABLE_WIDTH_MM - 2.0 * BOX_PAD_MM;
    let body_height = focus
        .items
        .iter()
        .map(|item| item_height(pdf, item, inner_width))
        .sum::<f32>();
    let body_top = top - TITLE_BAR_MM;
    let bottom = body_top - body_height - 2.0 * BOX_PAD_MM;

    // The title bar, then the body it sits on: both are drawn before the text
    // so the text lands on top of them.
    pdf.fill_rect(
        MARGIN_MM,
        body_top,
        PAGE_WIDTH_MM - MARGIN_MM,
        top,
        BRAND_COLOR,
    );
    pdf.fill_rect(
        MARGIN_MM,
        bottom,
        PAGE_WIDTH_MM - MARGIN_MM,
        body_top,
        BOX_BG,
    );
    pdf.rule(
        MARGIN_MM,
        bottom,
        PAGE_WIDTH_MM - MARGIN_MM,
        bottom,
        BOX_RULE,
        0.4,
    );
    pdf.rule(MARGIN_MM, bottom, MARGIN_MM, body_top, BOX_RULE, 0.4);
    pdf.rule(
        PAGE_WIDTH_MM - MARGIN_MM,
        bottom,
        PAGE_WIDTH_MM - MARGIN_MM,
        body_top,
        BOX_RULE,
        0.4,
    );

    let title_baseline = body_top + (TITLE_BAR_MM - TITLE_SIZE * 0.7 * PT_TO_MM) / 2.0;
    pdf.text(
        &focus.title,
        true,
        MARGIN_MM + BOX_PAD_MM,
        title_baseline,
        TITLE_SIZE,
        WHITE,
    );

    let mut y = body_top - BOX_PAD_MM;
    for item in &focus.items {
        y = draw_item(pdf, item, y, inner_width);
    }
    bottom
}

/// What an item consumes vertically (mm), including the gap after it.
fn item_height(pdf: &Pdf, item: &Item, width: f32) -> f32 {
    match item {
        Item::Sub(title) => {
            pdf.spans_height_mm(&[Span::styled(title, true, false)], SUB_SIZE, width)
                + gap_after(item)
        }
        Item::Prose(spans) => pdf.spans_height_mm(spans, CARD_SIZE, width) + gap_after(item),
        Item::Commands(rows) => {
            let command_width = command_column_width(pdf, rows, width);
            let description_width = width - command_width - COLUMN_GAP_MM;
            rows.iter()
                .map(|(command, description)| {
                    let left = pdf.spans_height_mm(command, CARD_SIZE, command_width);
                    let right = pdf.spans_height_mm(description, CARD_SIZE, description_width);
                    left.max(right) + ROW_GAP_MM
                })
                .sum::<f32>()
                + gap_after(item)
        }
        Item::Bullets(items) => {
            items
                .iter()
                .map(|spans| {
                    pdf.spans_height_mm(spans, CARD_SIZE, width - BULLET_INDENT_MM) + ROW_GAP_MM
                })
                .sum::<f32>()
                + gap_after(item)
        }
        Item::Code(lines) => {
            lines.len() as f32 * CODE_SIZE * 1.35 * PT_TO_MM + 2.0 * CODE_PAD_MM + gap_after(item)
        }
    }
}

/// The gap left below an item, before the next one.
fn gap_after(item: &Item) -> f32 {
    match item {
        Item::Sub(_) => 1.2,
        _ => 2.4,
    }
}

/// The vertical step between the rows of a command list or a bullet list.
const ROW_GAP_MM: f32 = 1.4;
/// How far a bullet's text is indented from the box's inner edge.
const BULLET_INDENT_MM: f32 = 4.0;
/// Padding between a code block's tint and its text.
const CODE_PAD_MM: f32 = 2.0;

/// The width of the command column: wide enough for the longest command it can
/// give a line to, within the bounds that keep either column readable.
fn command_column_width(pdf: &Pdf, rows: &[(Vec<Span>, Vec<Span>)], width: f32) -> f32 {
    let longest = rows
        .iter()
        .map(|(command, _)| {
            command
                .iter()
                .map(|span| pdf.text_width_mm(span.text(), span.bold(), CARD_SIZE))
                .sum::<f32>()
        })
        .fold(0.0_f32, f32::max);
    longest.clamp(width * COMMAND_COLUMN_MIN, width * COMMAND_COLUMN_MAX)
}

/// Draw an item with its top at `y`, returning the new top.
fn draw_item(pdf: &mut Pdf, item: &Item, y: f32, width: f32) -> f32 {
    let x = MARGIN_MM + BOX_PAD_MM;
    let bottom = match item {
        Item::Sub(title) => pdf.draw_spans_at(
            &[Span::styled(title, true, false)],
            SUB_SIZE,
            x,
            width,
            y,
            TEXT_COLOR,
        ),
        Item::Prose(spans) => pdf.draw_spans_at(spans, CARD_SIZE, x, width, y, TEXT_COLOR),
        Item::Commands(rows) => {
            let command_width = command_column_width(pdf, rows, width);
            let description_width = width - command_width - COLUMN_GAP_MM;
            let mut row_y = y;
            for (command, description) in rows {
                let left =
                    pdf.draw_spans_at(command, CARD_SIZE, x, command_width, row_y, BRAND_COLOR);
                let right = pdf.draw_spans_at(
                    description,
                    CARD_SIZE,
                    x + command_width + COLUMN_GAP_MM,
                    description_width,
                    row_y,
                    TEXT_COLOR,
                );
                row_y = left.min(right) - ROW_GAP_MM;
            }
            row_y
        }
        Item::Bullets(items) => {
            let mut row_y = y;
            for spans in items {
                pdf.text(
                    "-",
                    false,
                    x,
                    row_y - CARD_SIZE * 1.35 * PT_TO_MM,
                    CARD_SIZE,
                    TEXT_COLOR,
                );
                row_y = pdf.draw_spans_at(
                    spans,
                    CARD_SIZE,
                    x + BULLET_INDENT_MM,
                    width - BULLET_INDENT_MM,
                    row_y,
                    TEXT_COLOR,
                ) - ROW_GAP_MM;
            }
            row_y
        }
        Item::Code(lines) => {
            let height = lines.len() as f32 * CODE_SIZE * 1.35 * PT_TO_MM + 2.0 * CODE_PAD_MM;
            pdf.fill_rect(x, y - height, x + width, y, CODE_BG);
            let mut line_y = y - CODE_PAD_MM;
            for line in lines {
                line_y -= CODE_SIZE * 1.35 * PT_TO_MM;
                pdf.text(line, false, x + CODE_PAD_MM, line_y, CODE_SIZE, TEXT_COLOR);
            }
            y - height
        }
    };
    bottom - gap_after(item)
}

// --- The manual ---

/// The manual's band text, and how deep its table of contents goes.
const MANUAL_HEADER: &str = "orangu-manual";
const TOC_DEPTH: u8 = 2;
/// The cover's wordmark and tagline sizes.
const COVER_TITLE_SIZE: f32 = 64.0;
const COVER_SUBTITLE_SIZE: f32 = 20.0;
/// A table's cell padding, and the rule under its header row.
const CELL_PAD_MM: f32 = 2.0;
const TABLE_RULE: f32 = 0.4;

/// Build `source_dir`'s Markdown into the manual at `output`: a brand cover, a
/// table of contents, then one chapter per source file. Returns the path
/// written.
pub fn build_manual(source_dir: &Path, output: &Path) -> Result<PathBuf> {
    let chapters = read_chapters(source_dir)?;
    let (title, subtitle) = front_matter(source_dir)?;

    // Two passes: the first learns which page each contents entry lands on,
    // knowing only that every chapter starts a fresh page, so the numbers hold
    // once the cover and the contents are pushed in front of them.
    let mut probe = Pdf::with_footer(MANUAL_HEADER, FOOTER, (FOOTER, FOOTER_URL))?;
    let entries = draw_chapters(&mut probe, &chapters)?;
    let toc_pages = toc_page_count(entries.len());
    let offset = 1 + toc_pages;

    let mut pdf = Pdf::with_footer(MANUAL_HEADER, FOOTER, (FOOTER, FOOTER_URL))?;
    draw_cover(&mut pdf, &title, &subtitle);
    pdf.new_page();
    draw_contents(&mut pdf, &entries, offset);
    while pdf.current_page() < offset {
        pdf.new_page();
    }
    pdf.new_page();
    draw_chapters(&mut pdf, &chapters)?;

    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    pdf.save(output)?;
    Ok(output.to_path_buf())
}

/// One chapter: the pieces of one source file, in order.
struct Chapter {
    pieces: Vec<Piece>,
}

/// What a chapter is made of. Prose, lists and code go through the engine's
/// own Markdown rendering; tables and images are laid out here, because a
/// reference manual's tables have to be real columns rather than the reports'
/// monospaced approximation.
enum Piece {
    Blocks(Vec<Block>),
    /// A heading: its depth, its numbered text, and the blocks that draw it.
    Heading(u8, String, Vec<Block>),
    Table(Vec<Vec<Vec<Span>>>),
    Image(PathBuf),
    PageBreak,
}

fn read_chapters(dir: &Path) -> Result<Vec<Chapter>> {
    let mut numbers = [0usize; 6];
    source_files(dir)?
        .iter()
        .map(|path| read_chapter(path, &mut numbers))
        .collect()
}

fn read_chapter(path: &Path, numbers: &mut [usize; 6]) -> Result<Chapter> {
    let markdown =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let root = parse_markdown(strip_front_matter(&markdown));
    let dir = path.parent().unwrap_or(Path::new("."));

    let mut pieces = Vec::new();
    for node in root.children().map(Vec::as_slice).unwrap_or(&[]) {
        match node {
            Node::Heading(heading) => {
                let text =
                    number_heading(heading.depth, &plain_text(&inline_spans_of(node)), numbers);
                let mut blocks = Vec::new();
                render_block_nodes(std::slice::from_ref(node), 0, &mut blocks);
                if let Some(block) = blocks.first_mut() {
                    block.set_text(&text);
                }
                pieces.push(Piece::Heading(heading.depth, text, blocks));
            }
            Node::Table(table) => pieces.push(Piece::Table(table_cells(table))),
            // A paragraph holding nothing but an image is a figure; a lone
            // `\newpage` is the page break the sources use between chapters.
            Node::Paragraph(paragraph) => match paragraph.children.as_slice() {
                [Node::Image(image)] => pieces.push(Piece::Image(resolve_asset(dir, &image.url))),
                _ if plain_text(&inline_spans_of(node)).trim() == "\\newpage" => {
                    pieces.push(Piece::PageBreak);
                }
                _ => pieces.push(blocks_piece(node)),
            },
            other => pieces.push(blocks_piece(other)),
        }
    }
    Ok(Chapter { pieces })
}

/// Where an image referenced from a chapter actually lives: beside the
/// chapter, or one or two directories up. `doc/manual/en` refers to
/// `images/orangu-terminal.png`, which sits in `doc/images`.
fn resolve_asset(dir: &Path, url: &str) -> PathBuf {
    let mut root = Some(dir);
    while let Some(base) = root {
        let candidate = base.join(url);
        if candidate.is_file() {
            return candidate;
        }
        root = base.parent();
    }
    dir.join(url)
}

fn blocks_piece(node: &Node) -> Piece {
    let mut blocks = Vec::new();
    render_block_nodes(std::slice::from_ref(node), 0, &mut blocks);
    Piece::Blocks(blocks)
}

/// Number a heading the way the manual has always been numbered: chapters
/// `1`, sections `1.2`, and so on, resetting every deeper counter.
fn number_heading(depth: u8, text: &str, numbers: &mut [usize; 6]) -> String {
    let index = (depth as usize).min(numbers.len()) - 1;
    numbers[index] += 1;
    for deeper in numbers.iter_mut().skip(index + 1) {
        *deeper = 0;
    }
    let number = numbers[..=index]
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(".");
    format!("{number} {text}")
}

/// A table's cells, row by row, with the header row first.
fn table_cells(table: &Table) -> Vec<Vec<Vec<Span>>> {
    table
        .children
        .iter()
        .filter_map(|row| {
            Some(
                row.children()?
                    .iter()
                    .map(spans_of_node)
                    .collect::<Vec<_>>(),
            )
        })
        .collect()
}

/// The title and subtitle for the cover, from the first source's YAML front
/// matter.
fn front_matter(dir: &Path) -> Result<(String, String)> {
    let first = source_files(dir)?.remove(0);
    let text = fs::read_to_string(&first)
        .with_context(|| format!("failed to read {}", first.display()))?;
    let mut title = String::from("orangu");
    let mut subtitle = String::new();
    for line in text.lines().take_while(|line| !line.starts_with("...")) {
        if let Some(value) = line.strip_prefix("title:") {
            title = value.trim().trim_matches('"').to_string();
        } else if let Some(value) = line.strip_prefix("subtitle:") {
            subtitle = value.trim().trim_matches('"').to_string();
        }
    }
    Ok((title, subtitle))
}

/// Drop a leading `---`/`...` YAML block: it is metadata for the cover, not
/// content, and the Markdown parser would set it as a table.
fn strip_front_matter(markdown: &str) -> &str {
    let trimmed = markdown.trim_start();
    if !trimmed.starts_with("---") {
        return markdown;
    }
    trimmed
        .split_once('\n')
        .and_then(|(_, rest)| {
            rest.find("\n...")
                .or_else(|| rest.find("\n---"))
                .map(|at| &rest[at + 4..])
        })
        .unwrap_or(markdown)
}

/// The cover: the wordmark and its tagline, white on a full-bleed brand page.
fn draw_cover(pdf: &mut Pdf, title: &str, subtitle: &str) {
    pdf.fill_rect(0.0, 0.0, PAGE_WIDTH_MM, PAGE_HEIGHT_MM, BRAND_COLOR);
    let title_width = pdf.text_width_mm(title, true, COVER_TITLE_SIZE);
    pdf.text(
        title,
        true,
        (PAGE_WIDTH_MM - title_width) / 2.0,
        PAGE_HEIGHT_MM / 2.0,
        COVER_TITLE_SIZE,
        WHITE,
    );
    let subtitle_width = pdf.text_width_mm(subtitle, false, COVER_SUBTITLE_SIZE);
    pdf.text(
        subtitle,
        false,
        (PAGE_WIDTH_MM - subtitle_width) / 2.0,
        PAGE_HEIGHT_MM / 2.0 - 18.0,
        COVER_SUBTITLE_SIZE,
        WHITE,
    );
    let footer_width = pdf.text_width_mm(FOOTER, false, CARD_SIZE);
    pdf.text(
        FOOTER,
        false,
        (PAGE_WIDTH_MM - footer_width) / 2.0,
        30.0,
        CARD_SIZE,
        WHITE,
    );
}

/// A contents entry: its depth, its numbered title, and the content page it
/// starts on before the cover and contents are counted in.
struct Entry {
    depth: u8,
    title: String,
    page: usize,
}

const TOC_SIZE: f32 = 10.0;
const TOC_ROW_MM: f32 = TOC_SIZE * 1.7 * PT_TO_MM;

fn toc_page_count(entries: usize) -> usize {
    let rows_per_page = ((CONTENT_TOP_MM - CONTENT_BOTTOM_MM - 12.0) / TOC_ROW_MM) as usize;
    entries.div_ceil(rows_per_page.max(1)).max(1)
}

fn draw_contents(pdf: &mut Pdf, entries: &[Entry], offset: usize) {
    let mut blocks = Vec::new();
    render_block_nodes(
        parse_markdown("# Table of Contents")
            .children()
            .map(Vec::as_slice)
            .unwrap_or(&[]),
        0,
        &mut blocks,
    );
    pdf.draw_blocks(&blocks);

    for entry in entries {
        if pdf.room_left_mm() < TOC_ROW_MM {
            pdf.new_page();
        }
        let y = pdf.cursor() - TOC_ROW_MM;
        pdf.set_cursor(y);
        let indent = (entry.depth.saturating_sub(1)) as f32 * 6.0;
        pdf.text(
            &entry.title,
            entry.depth == 1,
            MARGIN_MM + indent,
            y,
            TOC_SIZE,
            BRAND_COLOR,
        );
        let page = (entry.page + offset).to_string();
        let width = pdf.text_width_mm(&page, false, TOC_SIZE);
        pdf.text(
            &page,
            false,
            PAGE_WIDTH_MM - MARGIN_MM - width,
            y,
            TOC_SIZE,
            BRAND_COLOR,
        );
        pdf.link_to_page(entry.page + offset, y - 1.0, TOC_SIZE * PT_TO_MM + 2.0);
    }
}

/// Draw every chapter, returning the contents entries with the page each one
/// started on. Each chapter opens a fresh page, so these numbers stay true
/// when the cover and contents are added in front of them.
fn draw_chapters(pdf: &mut Pdf, chapters: &[Chapter]) -> Result<Vec<Entry>> {
    let mut entries = Vec::new();
    let mut first = true;
    for chapter in chapters {
        if !first {
            page_break(pdf);
        }
        first = false;
        for piece in &chapter.pieces {
            match piece {
                Piece::Heading(depth, title, blocks) => {
                    if *depth <= TOC_DEPTH {
                        entries.push(Entry {
                            depth: *depth,
                            title: title.clone(),
                            page: pdf.current_page(),
                        });
                    }
                    pdf.draw_blocks(blocks);
                }
                Piece::Blocks(blocks) => pdf.draw_blocks(blocks),
                Piece::Table(rows) => draw_table(pdf, rows),
                Piece::Image(path) => match fs::read(path) {
                    Ok(bytes) => pdf.draw_image(&bytes, USABLE_WIDTH_MM)?,
                    Err(error) => {
                        bail!("failed to read {}: {error}", path.display());
                    }
                },
                Piece::PageBreak => page_break(pdf),
            }
        }
    }
    Ok(entries)
}

/// Start a new page, unless nothing has been drawn on this one yet. Every
/// chapter file opens with a `\newpage`, and a chapter already begins on a
/// fresh page, so taking both at face value would leave a blank page between
/// every pair of chapters.
fn page_break(pdf: &mut Pdf) {
    if pdf.cursor() < CONTENT_TOP_MM {
        pdf.new_page();
    }
}

/// Draw a Markdown table as real columns: widths from the content, cells
/// wrapped inside them, the header row in bold over a rule, and a rule under
/// every row. A table that outruns the page continues on the next one under a
/// repeat of its header.
fn draw_table(pdf: &mut Pdf, rows: &[Vec<Vec<Span>>]) {
    let Some(header) = rows.first() else {
        return;
    };
    let widths = column_widths(pdf, rows);
    let row_of = |pdf: &Pdf, cells: &[Vec<Span>]| -> f32 {
        cells
            .iter()
            .zip(&widths)
            .map(|(cell, width)| pdf.spans_height_mm(cell, CODE_SIZE, width - 2.0 * CELL_PAD_MM))
            .fold(0.0_f32, f32::max)
            + 2.0 * CELL_PAD_MM
    };

    let draw_row = |pdf: &mut Pdf, cells: &[Vec<Span>], bold: bool| {
        let height = row_of(pdf, cells);
        if pdf.room_left_mm() < height {
            pdf.new_page();
        }
        let top = pdf.cursor();
        let mut x = MARGIN_MM;
        for (cell, width) in cells.iter().zip(&widths) {
            let spans: Vec<Span> = if bold {
                cell.iter()
                    .map(|span| Span::styled(span.text(), true, false))
                    .collect()
            } else {
                cell.clone()
            };
            pdf.draw_spans_at(
                &spans,
                CODE_SIZE,
                x + CELL_PAD_MM,
                width - 2.0 * CELL_PAD_MM,
                top - CELL_PAD_MM,
                TEXT_COLOR,
            );
            x += width;
        }
        let bottom = top - height;
        pdf.rule(
            MARGIN_MM,
            bottom,
            MARGIN_MM + widths.iter().sum::<f32>(),
            bottom,
            BOX_RULE,
            TABLE_RULE,
        );
        pdf.set_cursor(bottom);
    };

    draw_row(pdf, header, true);
    for cells in rows.iter().skip(1) {
        draw_row(pdf, cells, false);
    }
    pdf.set_cursor(pdf.cursor() - CODE_SIZE * 0.6 * PT_TO_MM);
}

/// Column widths for a table: each column asks for what its longest cell needs
/// on one line, and the excess over the text column is taken back in
/// proportion, so a column of short flags keeps its width and a column of
/// prose gives way.
fn column_widths(pdf: &Pdf, rows: &[Vec<Vec<Span>>]) -> Vec<f32> {
    let columns = rows.iter().map(Vec::len).max().unwrap_or(0);
    let mut wanted = vec![0.0_f32; columns];
    for row in rows {
        for (index, cell) in row.iter().enumerate() {
            let width: f32 = cell
                .iter()
                .map(|span| pdf.text_width_mm(span.text(), span.bold(), CODE_SIZE))
                .sum();
            wanted[index] = wanted[index].max(width + 2.0 * CELL_PAD_MM);
        }
    }
    let total: f32 = wanted.iter().sum();
    if total <= USABLE_WIDTH_MM {
        // Spread what is left over the columns, so the table fills the page.
        let extra = (USABLE_WIDTH_MM - total) / columns as f32;
        return wanted.iter().map(|width| width + extra).collect();
    }
    // Too wide: shrink proportionally, but never below a readable minimum.
    let minimum = (USABLE_WIDTH_MM / columns as f32 / 3.0).max(14.0);
    let scale = USABLE_WIDTH_MM / total;
    let mut widths: Vec<f32> = wanted
        .iter()
        .map(|width| (width * scale).max(minimum))
        .collect();
    let over: f32 = widths.iter().sum::<f32>() - USABLE_WIDTH_MM;
    if over > 0.0 {
        // Take the overshoot from the widest column, which is the prose one.
        if let Some(widest) = widths
            .iter_mut()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        {
            *widest -= over;
        }
    }
    widths
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse `markdown` as a page. Each caller names its own file: the tests
    /// run in one process, in parallel, so a shared name would have them read
    /// each other's content.
    fn page(name: &str, markdown: &str) -> Page {
        let dir = std::env::temp_dir().join(format!("orangu-docs-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("00-{name}.md"));
        std::fs::write(&path, markdown).unwrap();
        let page = parse_page(&path).unwrap();
        std::fs::remove_file(&path).ok();
        page
    }

    #[test]
    fn a_level_one_heading_opens_a_box() {
        let page = page("boxes", "# Setup\n\nProse.\n\n# Coding\n\nMore.\n");
        assert_eq!(page.boxes.len(), 2);
        assert_eq!(page.boxes[0].title, "Setup");
        assert_eq!(page.boxes[1].title, "Coding");
    }

    #[test]
    fn a_table_becomes_a_command_list_without_its_header_row() {
        let page = page(
            "commands",
            "# Setup\n\n| Command | What it does |\n| --- | --- |\n| `orangu -i` | Configure it. |\n",
        );
        let Item::Commands(rows) = &page.boxes[0].items[0] else {
            panic!("expected a command list");
        };
        assert_eq!(rows.len(), 1);
        assert_eq!(plain_text(&rows[0].0), "orangu -i");
        assert_eq!(plain_text(&rows[0].1), "Configure it.");
    }

    #[test]
    fn inline_code_in_a_command_list_is_set_bold() {
        let page = page(
            "bold",
            "# Setup\n\n| A | B |\n| --- | --- |\n| `orangu` | plain |\n",
        );
        let Item::Commands(rows) = &page.boxes[0].items[0] else {
            panic!("expected a command list");
        };
        assert!(rows[0].0.iter().all(Span::bold));
        assert!(rows[0].1.iter().all(|span| !span.bold()));
    }

    #[test]
    fn headings_are_numbered_by_depth() {
        let mut numbers = [0usize; 6];
        assert_eq!(
            number_heading(1, "Introduction", &mut numbers),
            "1 Introduction"
        );
        assert_eq!(number_heading(2, "Features", &mut numbers), "1.1 Features");
        assert_eq!(number_heading(3, "A stack", &mut numbers), "1.1.1 A stack");
        assert_eq!(number_heading(2, "Tools", &mut numbers), "1.2 Tools");
        assert_eq!(
            number_heading(1, "Getting started", &mut numbers),
            "2 Getting started"
        );
    }

    #[test]
    fn front_matter_is_not_content() {
        let markdown = "---\ntitle: \"orangu\"\nsubtitle: \"Advanced\"\n...\n\n# Introduction\n";
        assert_eq!(
            strip_front_matter(markdown).trim_start(),
            "# Introduction\n"
        );
    }

    #[test]
    fn a_document_without_front_matter_is_left_alone() {
        let markdown = "# Introduction\n\nProse.\n";
        assert_eq!(strip_front_matter(markdown), markdown);
    }

    #[test]
    fn content_before_the_first_heading_is_rejected() {
        let dir = std::env::temp_dir().join(format!("orangu-docs-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("00-loose.md");
        std::fs::write(&path, "Loose prose.\n\n# Setup\n").unwrap();
        let error = match parse_page(&path) {
            Err(error) => error.to_string(),
            Ok(_) => panic!("expected the page to be rejected"),
        };
        std::fs::remove_file(&path).ok();
        assert!(error.contains("before the first"), "{error}");
    }
}
