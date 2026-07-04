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
    Actions, BorderArray, BuiltinFont, Color, Destination, Line, LinePoint, LinkAnnotation, Mm, Op,
    PaintMode, ParsedFont, PdfDocument, PdfFontHandle, PdfPage, PdfSaveOptions, Point, Pt, Rect,
    Rgb, TextItem,
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

use orangu::duplicates::{DuplicatesReport, FunctionLocation};
use orangu::tui::TranscriptLine;

use crate::VERSION;
use crate::git::{
    ForgeWeb, PullRequestDetail, PullRequestReviewer, discover_git_root, forge_web_from_origin,
    git_repository_name, workspace_branch_name,
};
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
    /// When set, every drawn line of this block is underlined and made a
    /// clickable hyperlink to this URL (used for the duplicates export's source
    /// links). `None` for ordinary text.
    link: Option<String>,
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
            link: None,
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
            link: None,
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
    let mut toc_rows: Vec<(&str, usize, Option<bool>)> = sections
        .iter()
        .map(|section| section.title.as_str())
        .zip(starts.iter().copied())
        .map(|(title, start)| (title, start, None))
        .collect();
    if let Some(start) = appendix_start {
        toc_rows.push(("Appendix", start, None));
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

/// Export a duplicate-code report to a PDF in the workspace root as
/// `{repository}-{branch}-duplicates.pdf`.
///
/// Laid out like the review export:
///
/// - **Page 1 — summary.** A table of the repository, branch, generation
///   date/time, the threshold used, and the file/function/pair counts.
/// - **Page 2 — table of contents.** One entry per similarity chapter with the
///   page it starts on.
/// - **Page 3 onward — the chapters.** The candidate pairs are grouped by their
///   similarity percentage; each `{n}% similar` chapter starts on its own page
///   and lists its pairs (the two function names and their source locations).
///
/// A report with no pairs is a summary page followed by a short note.
pub fn export_duplicates(
    workspace: &Path,
    report: &DuplicatesReport,
    model: &str,
) -> Result<PathBuf> {
    let repository = repository_display(workspace);
    let branch = branch_display(workspace);
    let mut pdf = Pdf::new(&header_label(workspace), model)?;

    // Page 1 — summary statistics.
    pdf.draw_duplicates_info_page(&repository, &branch, report);

    if report.pairs.is_empty() {
        pdf.new_page();
        pdf.draw_blocks(&[Block::paragraph(vec![Span::plain(format!(
            "No function pairs met the {}% similarity threshold.",
            report.threshold_percent()
        ))])]);
        let path = export_file_path(workspace, "duplicates");
        pdf.save(&path)?;
        return Ok(path);
    }

    // One chapter (title + blocks) per similarity percentage, each on its own
    // page. Source locations link to the forge when one is known.
    let linker = SourceLinker::new(workspace);
    let chapters = build_duplicate_chapters(report, &linker);

    // The chapters start on page 3, each on its own page; compute where each one
    // lands so the table of contents (page 2) can point at it.
    let mut starts = Vec::with_capacity(chapters.len());
    let mut page = 3;
    for (_, blocks) in &chapters {
        starts.push(page);
        page += paginate(blocks, &pdf.fonts);
    }

    // Page 2 — table of contents (each entry links to its chapter).
    let toc_rows: Vec<(&str, usize, Option<bool>)> = chapters
        .iter()
        .map(|(title, _)| title.as_str())
        .zip(starts.iter().copied())
        .map(|(title, start)| (title, start, None))
        .collect();
    pdf.new_page();
    pdf.draw_toc(&toc_rows);

    // Pages 3+ — one chapter per page.
    for (_, blocks) in &chapters {
        pdf.new_page();
        pdf.draw_blocks(blocks);
    }

    let path = export_file_path(workspace, "duplicates");
    pdf.save(&path)?;
    Ok(path)
}

/// Export a report of every open pull/merge request to `{repository}-pr.pdf`
/// in the workspace root. Unlike the other exports, the filename carries no
/// branch — the report covers the whole repository, not one branch.
///
/// `prs` is fetched from the forge by the caller.
///
/// - **Page 1 — pull request status.** The repository name, generation time,
///   and the open pull requests broken down by status: open, ready for
///   review, and conflicting.
/// - **Page 2 — table of contents.** One entry per open pull request, `#N
///   Title`, linking to its page.
/// - **Page 3 onward — one page per pull request** (more when it changes
///   many files): its title linking to the forge, a table of author, dates,
///   branches, draft/conflict status, comment count, assignees, reviewers
///   (with their review state), and labels, then the changed files — full
///   path, one per line, with additions in green and deletions in red.
///
/// A repository with no open pull requests still gets its status page,
/// followed by a short note instead of a table of contents and PR pages.
pub fn export_pr(workspace: &Path, prs: &[PullRequestDetail], model: &str) -> Result<PathBuf> {
    let repository = repository_display(workspace);
    let mut pdf = Pdf::new(&header_label(workspace), model)?;

    pdf.draw_pr_stats_page(&repository, prs);

    let path = export_pr_file_path(workspace);
    if prs.is_empty() {
        pdf.new_page();
        pdf.draw_blocks(&[Block::paragraph(vec![Span::plain(
            "No open pull requests.",
        )])]);
        pdf.save(&path)?;
        return Ok(path);
    }

    // One page (more when a pull request touches many files) per pull
    // request; compute where each lands so the table of contents (page 2)
    // can point at it. The header (title + field table) is a fixed height
    // drawn ahead of the changed-files list, so the files are paginated from
    // however much room is left on that first page; the "Last comment" table
    // follows, adding one more page only if it doesn't fit where the files
    // list left off.
    let titles: Vec<String> = prs
        .iter()
        .map(|pr| format!("#{} {}", pr.number, pr.title))
        .collect();
    let mut starts = Vec::with_capacity(prs.len());
    let mut page = 3;
    for pr in prs {
        starts.push(page);
        let header_height = pr_header_height_mm(pr, &pdf.fonts);
        let start_cursor = (CONTENT_TOP_MM - header_height).max(CONTENT_BOTTOM_MM);
        let (mut pages_for_pr, cursor_after_files) =
            paginate_from(&build_changed_file_blocks(pr), &pdf.fonts, start_cursor);
        let comment_table_height = last_comment_table_height_mm(pr, &pdf.fonts);
        if cursor_after_files - comment_table_height < CONTENT_BOTTOM_MM {
            pages_for_pr += 1;
        }
        page += pages_for_pr;
    }

    let toc_rows: Vec<(&str, usize, Option<bool>)> = titles
        .iter()
        .map(String::as_str)
        .zip(starts.iter().copied())
        .zip(prs.iter())
        .map(|((title, start), pr)| (title, start, Some(pr_is_ready(pr))))
        .collect();
    pdf.new_page();
    pdf.draw_toc(&toc_rows);

    for pr in prs {
        pdf.new_page();
        pdf.draw_pr_detail_page(pr);
    }

    pdf.save(&path)?;
    Ok(path)
}

/// The linked title heading (`#N Title`, linking to the forge) for a pull
/// request's detail page.
fn pr_title_block(pr: &PullRequestDetail) -> Block {
    let title_size = BODY_SIZE + 3.5;
    Block {
        spans: vec![Span {
            text: format!("#{} {}", pr.number, pr.title),
            bold: true,
            italic: false,
            color: None,
        }],
        size: title_size,
        indent_mm: 0.0,
        hanging_mm: 0.0,
        word_wrap: true,
        space_after_mm: title_size * 0.6 * PT_TO_MM,
        link: (!pr.url.is_empty()).then(|| pr.url.clone()),
    }
}

/// A [`Pdf::draw_kv_table`] row's value: plain text for most rows; a URL for
/// a pull request's "Link" row, drawn brand-coloured and clickable; a URL
/// followed by plain text (for the stats page's "Oldest"/"Newest" rows — the
/// pull request's link, then its creation date); or — for a pull request's
/// "Reviewers" row — the reviewer list itself, drawn with a status icon per
/// name instead of the review state spelled out.
#[derive(Clone, Debug, PartialEq, Eq)]
enum RowValue {
    Text(String),
    Link(String),
    LinkWithText(String, String),
    Reviewers(Vec<PullRequestReviewer>),
}

/// A reviewer's status, reduced to the icon [`Pdf::draw_review_icon`] draws:
/// a green checkmark for an approval, a red "X" for a change request (or, on
/// GitLab, a still-outstanding review request), and a "?" for anything else —
/// a comment, a still-pending GitHub review request, or a dismissed review,
/// none of which are a meaningful verdict either way.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReviewIcon {
    Check,
    Cross,
    Question,
}

fn review_icon(state: &str) -> ReviewIcon {
    match state {
        "Approved" => ReviewIcon::Check,
        "Changes requested" | "Requested" => ReviewIcon::Cross,
        _ => ReviewIcon::Question,
    }
}

/// Whether a pull request is ready to merge as far as its own state goes:
/// not a draft, and not reported as conflicting (a `None`/unknown mergeable
/// state is not treated as a conflict). Drives the table of contents' green
/// checkmark (`true`) versus red "X" (`false`) next to each entry.
fn pr_is_ready(pr: &PullRequestDetail) -> bool {
    !pr.draft && pr.conflicting != Some(true)
}

/// The rows of a pull request's header table: author, dates, branch,
/// draft/conflict status, comment count, assignees, reviewers (as a status
/// icon per name), and labels — whatever the forge returned. Each row carries
/// whether its value should render bold: `Draft`/`Conflicts` are bold when
/// `Yes`, so a draft or a conflict catches the eye.
/// The rows of the pull request export's page 1 status table: the
/// repository, generation time, the open/ready/conflicting counts, and — when
/// there is at least one open pull request with a known creation date — the
/// oldest and newest, each a clickable link to the pull request followed by
/// its creation date.
fn pr_stats_rows(repository: &str, prs: &[PullRequestDetail]) -> Vec<(String, RowValue, bool)> {
    let ready = prs.iter().filter(|pr| !pr.draft).count();
    let conflicts = prs.iter().filter(|pr| pr.conflicting == Some(true)).count();
    let mut rows = vec![
        (
            "Repository".to_string(),
            RowValue::Text(repository.to_string()),
            false,
        ),
        (
            "Generated".to_string(),
            RowValue::Text(format_timestamp()),
            false,
        ),
        (
            "Open".to_string(),
            RowValue::Text(prs.len().to_string()),
            false,
        ),
        (
            "Ready".to_string(),
            RowValue::Text(ready.to_string()),
            false,
        ),
        (
            "Conflicts".to_string(),
            RowValue::Text(conflicts.to_string()),
            false,
        ),
    ];
    let (oldest, newest) = oldest_and_newest(prs);
    // With exactly one open pull request, "Oldest" and "Newest" would be the
    // same entry — showing it twice is redundant, so "Oldest" is left empty
    // rather than repeating "Newest". With none at all, both fall back to
    // "N/A" via `pr_link_or_na`.
    let oldest_value = if prs.len() == 1 {
        RowValue::Text(String::new())
    } else {
        pr_link_or_na(oldest)
    };
    rows.push(("Oldest".to_string(), oldest_value, false));
    rows.push(("Newest".to_string(), pr_link_or_na(newest), false));
    rows
}

/// A pull request as an "Oldest"/"Newest" row value: its link and creation
/// date, or `"N/A"` when there is none (no open pull requests, or none with
/// a known creation date).
fn pr_link_or_na(pr: Option<&PullRequestDetail>) -> RowValue {
    match pr {
        Some(pr) => RowValue::LinkWithText(pr.url.clone(), pr.created_at.clone()),
        None => RowValue::Text("N/A".to_string()),
    }
}

/// The oldest and newest pull request by `created_at` (an ISO-ish
/// `YYYY-MM-DD HH:MM:SS` string, which sorts lexicographically) — `None` for
/// either when no pull request has a known creation date. The same pull
/// request is returned for both when there is only one.
fn oldest_and_newest(
    prs: &[PullRequestDetail],
) -> (Option<&PullRequestDetail>, Option<&PullRequestDetail>) {
    let mut oldest: Option<&PullRequestDetail> = None;
    let mut newest: Option<&PullRequestDetail> = None;
    for pr in prs {
        if pr.created_at.is_empty() {
            continue;
        }
        if oldest.is_none_or(|current| pr.created_at < current.created_at) {
            oldest = Some(pr);
        }
        if newest.is_none_or(|current| pr.created_at > current.created_at) {
            newest = Some(pr);
        }
    }
    (oldest, newest)
}

fn pr_header_rows(pr: &PullRequestDetail) -> Vec<(String, RowValue, bool)> {
    let conflicts = match pr.conflicting {
        Some(true) => "Yes",
        Some(false) => "No",
        None => "Unknown",
    };
    let draft = if pr.draft { "Yes" } else { "No" };
    vec![
        (
            "Author".to_string(),
            RowValue::Text(non_empty(pr.author.clone(), "unknown")),
            false,
        ),
        ("Link".to_string(), RowValue::Link(pr.url.clone()), false),
        (
            "Created".to_string(),
            RowValue::Text(non_empty(pr.created_at.clone(), "unknown")),
            false,
        ),
        (
            "Updated".to_string(),
            RowValue::Text(non_empty(pr.updated_at.clone(), "unknown")),
            false,
        ),
        ("Branch".to_string(), RowValue::Text(pr.head.clone()), false),
        (
            "Draft".to_string(),
            RowValue::Text(draft.to_string()),
            draft == "Yes",
        ),
        (
            "Conflicts".to_string(),
            RowValue::Text(conflicts.to_string()),
            conflicts == "Yes",
        ),
        (
            "Comments".to_string(),
            RowValue::Text(pr.comment_count.to_string()),
            false,
        ),
        (
            "Assignees".to_string(),
            RowValue::Text(list_or_none(&pr.assignees)),
            false,
        ),
        (
            "Reviewers".to_string(),
            RowValue::Reviewers(pr.reviewers.clone()),
            false,
        ),
        (
            "Labels".to_string(),
            RowValue::Text(list_or_none(&pr.labels)),
            false,
        ),
    ]
}

/// The vertical space (mm) a pull request page's fixed header — the title
/// heading plus the field table — occupies, before the changed-files list
/// starts. Mirrors [`Pdf::draw_pr_detail_page`]'s layout exactly, so the
/// table of contents can compute accurate page numbers without drawing.
fn pr_header_height_mm(pr: &PullRequestDetail, fonts: &DocFonts) -> f32 {
    let title = pr_title_block(pr);
    let title_line_height = title.size * 1.35 * PT_TO_MM;
    let title_lines = wrap_block(&title, fonts).len() as f32;
    let title_height = title_lines * title_line_height + title.space_after_mm;

    let row_height = BODY_SIZE * 1.9 * PT_TO_MM;
    let table_height = pr_header_rows(pr).len() as f32 * row_height + 8.0;

    title_height + table_height
}

/// One line per changed file: its full path, then its added-line count in
/// green and removed-line count in red. Empty when the forge did not return a
/// diff (GitLab's merge-request list endpoint never does).
/// Extra breathing room (mm), on top of a changed-file line's normal
/// trailing gap, left after the last one — so the "Last comment" table that
/// follows doesn't crowd the changed files. Added to the last block's
/// `space_after_mm` so both the real draw (`draw_blocks`) and the table of
/// contents' page-count simulation (`paginate_from`, which also sums
/// `space_after_mm`) account for it identically.
const LAST_COMMENT_GAP_MM: f32 = 6.0;

fn build_changed_file_blocks(pr: &PullRequestDetail) -> Vec<Block> {
    let mut blocks: Vec<Block> = pr
        .files
        .iter()
        .map(|file| Block {
            spans: vec![
                Span::plain(format!("{}  ", file.path)),
                Span {
                    text: format!("+{}", file.additions),
                    bold: false,
                    italic: false,
                    color: Some(STATUS_GREEN),
                },
                Span::plain("  "),
                Span {
                    text: format!("-{}", file.deletions),
                    bold: false,
                    italic: false,
                    color: Some(STATUS_RED),
                },
            ],
            size: BODY_SIZE,
            indent_mm: 0.0,
            hanging_mm: 4.0,
            word_wrap: false,
            space_after_mm: BODY_SIZE * 0.25 * PT_TO_MM,
            link: None,
        })
        .collect();
    if let Some(last) = blocks.last_mut() {
        last.space_after_mm += LAST_COMMENT_GAP_MM;
    }
    blocks
}

fn list_or_none(items: &[String]) -> String {
    if items.is_empty() {
        "none".to_string()
    } else {
        items.join(", ")
    }
}

/// The comment body length above which [`truncate_comment`] cuts it off —
/// bounds the "Last comment" table's height so it stays small and its size
/// predictable ahead of drawing (see [`last_comment_table_height_mm`]).
const LAST_COMMENT_MAX_CHARS: usize = 500;

/// Cut `body` to [`LAST_COMMENT_MAX_CHARS`] characters, appending an ellipsis
/// when it was longer. Left whole otherwise.
fn truncate_comment(body: &str) -> String {
    if body.chars().count() <= LAST_COMMENT_MAX_CHARS {
        return body.to_string();
    }
    let mut truncated: String = body.chars().take(LAST_COMMENT_MAX_CHARS).collect();
    truncated.push('…');
    truncated
}

/// The "Last comment" table's geometry (the author/comment column divider,
/// the row height) and the comment's wrapped, truncated lines — shared by
/// [`Pdf::draw_last_comment_table`] and [`last_comment_table_height_mm`] so
/// they can never drift apart.
fn last_comment_table_layout(
    author: &str,
    body: &str,
    fonts: &DocFonts,
) -> (f32, f32, Vec<String>) {
    let x0 = MARGIN_MM;
    let x1 = PAGE_WIDTH_MM - MARGIN_MM;
    let row_height = BODY_SIZE * 1.9 * PT_TO_MM;
    let author_width = fonts.text_width_mm(author, false, false, BODY_SIZE);
    let divider = (x0 + 6.0 + author_width.max(20.0) + 6.0).min(x1 - 40.0);
    let comment_width = (x1 - divider - 6.0).max(20.0);
    let chars: Vec<StyledChar> = truncate_comment(body)
        .chars()
        .map(|ch| StyledChar {
            ch,
            bold: false,
            italic: false,
            color: None,
        })
        .collect();
    let lines = wrap(&chars, comment_width, comment_width, true, fonts, BODY_SIZE)
        .into_iter()
        .map(|line| line.into_iter().map(|styled| styled.ch).collect())
        .collect();
    (divider, row_height, lines)
}

/// The vertical space (mm) the "Last comment" table occupies, for a pull
/// request with (or without) a comment — `"N/A"`/`"N/A"` when there is none,
/// matching what [`Pdf::draw_pr_detail_page`] actually draws.
fn last_comment_table_height_mm(pr: &PullRequestDetail, fonts: &DocFonts) -> f32 {
    let (author, body) = match &pr.last_comment {
        Some(comment) => (comment.author.as_str(), comment.body.as_str()),
        None => ("N/A", "N/A"),
    };
    let (_, row_height, lines) = last_comment_table_layout(author, body, fonts);
    row_height + lines.len().max(1) as f32 * row_height + 8.0
}

/// `{repository}-pr.pdf` in the workspace root — no branch, since the report
/// covers every open pull request in the repository, not one branch.
fn export_pr_file_path(workspace: &Path) -> PathBuf {
    let repository = non_empty(sanitize(&repository_display(workspace)), "workspace");
    workspace.join(format!("{repository}-pr.pdf"))
}

/// Resolves source locations to forge (GitHub/GitLab) web URLs. Built once per
/// export: the forge web base, the ref to link at, and the workspace's path
/// relative to the repository root (so locations relative to the scan root map
/// to repository-relative paths).
struct SourceLinker {
    forge: Option<ForgeWeb>,
    git_ref: String,
    /// The workspace relative to the git root, forward-slashed, with a trailing
    /// `/` (empty when the workspace is the git root).
    prefix: String,
}

impl SourceLinker {
    fn new(workspace: &Path) -> Self {
        let git_root = discover_git_root(workspace);
        let forge = git_root.as_deref().and_then(forge_web_from_origin);
        // Link at the current branch; `HEAD` (the default branch on both forges)
        // when not on one.
        let git_ref = workspace_branch_name(workspace)
            .filter(|branch| !branch.is_empty())
            .unwrap_or_else(|| "HEAD".to_string());
        let prefix = git_root
            .as_deref()
            .and_then(|root| workspace.strip_prefix(root).ok())
            .map(|relative| {
                let text = relative.to_string_lossy().replace('\\', "/");
                if text.is_empty() {
                    String::new()
                } else {
                    format!("{text}/")
                }
            })
            .unwrap_or_default();
        SourceLinker {
            forge,
            git_ref,
            prefix,
        }
    }

    /// The forge URL for a scan-root-relative location, or `None` when there is
    /// no recognised forge.
    fn url(&self, location: &FunctionLocation) -> Option<String> {
        let forge = self.forge.as_ref()?;
        let path = format!(
            "{}{}",
            self.prefix,
            location.path.to_string_lossy().replace('\\', "/")
        );
        Some(forge.blob_url(&self.git_ref, &path, location.start_line, location.end_line))
    }
}

/// Group a report's pairs (already sorted most-similar first, so equal
/// percentages are contiguous) into `(title, blocks)` chapters — one per
/// distinct similarity percentage. Each chapter opens with its `{n}% similar`
/// heading, then per pair a `a <-> b` sub-heading and the two source-location
/// lines (`path:start–end`, linked to the forge when available).
fn build_duplicate_chapters(
    report: &DuplicatesReport,
    linker: &SourceLinker,
) -> Vec<(String, Vec<Block>)> {
    let mut chapters: Vec<(String, Vec<Block>)> = Vec::new();
    for pair in &report.pairs {
        let title = format!("{}% similar", pair.percent());
        if chapters.last().is_none_or(|(last, _)| last != &title) {
            let heading = heading(&title, BODY_SIZE + 3.5);
            chapters.push((title, vec![heading]));
        }
        let blocks = &mut chapters.last_mut().expect("chapter pushed above").1;
        blocks.push(heading(
            &format!("{}  <->  {}", pair.a.name, pair.b.name),
            BODY_SIZE + 1.0,
        ));
        blocks.push(location_block(&pair.a, linker));
        blocks.push(location_block(&pair.b, linker));
    }
    chapters
}

/// A single source-location line: `path:start–end` (a single line number when
/// the range is one line), drawn in the brand colour and linked to the forge
/// when `linker` resolves a URL, otherwise plain text.
fn location_block(location: &FunctionLocation, linker: &SourceLinker) -> Block {
    let range = if location.start_line == location.end_line {
        location.start_line.to_string()
    } else {
        format!("{}–{}", location.start_line, location.end_line)
    };
    let text = format!("{}:{}", location.path.display(), range);
    let url = linker.url(location);
    Block {
        spans: vec![Span {
            text,
            bold: false,
            italic: false,
            color: url.as_ref().map(|_| BRAND_COLOR),
        }],
        size: BODY_SIZE,
        indent_mm: 0.0,
        hanging_mm: 0.0,
        word_wrap: false,
        space_after_mm: BODY_SIZE * 0.25 * PT_TO_MM,
        link: url,
    }
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
pub(crate) fn repository_display(workspace: &Path) -> String {
    let root = discover_git_root(workspace).unwrap_or_else(|| workspace.to_path_buf());
    let name = git_repository_name(&root).unwrap_or_else(|| {
        root.file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default()
    });
    non_empty(name, "workspace")
}

/// The current branch name, for display (`nobranch` when not on one).
pub(crate) fn branch_display(workspace: &Path) -> String {
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
pub(crate) fn sanitize(value: &str) -> String {
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
            link: None,
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
        link: None,
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
                    link: None,
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
            link: None,
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
            link: None,
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
            link: None,
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
            link: None,
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
            // A linked block attaches a clickable URI annotation over the drawn
            // text (the brand colour already marks it as a link).
            if let Some(url) = &block.link {
                let text: String = line.iter().map(|styled| styled.ch).collect();
                let width = self.fonts.text_width_mm(&text, false, false, block.size);
                let baseline = self.cursor_y;
                self.ops.push(Op::LinkAnnotation {
                    link: LinkAnnotation::new(
                        Rect {
                            x: Mm(MARGIN_MM + indent).into(),
                            y: Mm(baseline - 1.5).into(),
                            width: Mm(width).into(),
                            height: Mm(block.size * PT_TO_MM + 2.5).into(),
                            mode: None,
                            winding_order: None,
                        },
                        Actions::uri(url.clone()),
                        Some(BorderArray::Solid([0.0, 0.0, 0.0])),
                        None,
                        None,
                    ),
                });
            }
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
            (
                "Repository".to_string(),
                RowValue::Text(repository.to_string()),
                false,
            ),
            (
                "Branch".to_string(),
                RowValue::Text(branch.to_string()),
                false,
            ),
            (
                "Generated".to_string(),
                RowValue::Text(format_timestamp()),
                false,
            ),
        ];
        for section in sections {
            rows.push((
                section.title.clone(),
                RowValue::Text(section.entry_count.to_string()),
                false,
            ));
        }
        self.draw_kv_table(&rows);
        self.draw_status_banner(verdict);
    }

    /// Page 1 of the duplicates export: the repository/branch/date and the scan
    /// statistics (threshold, files, functions, candidate pairs).
    fn draw_duplicates_info_page(
        &mut self,
        repository: &str,
        branch: &str,
        report: &DuplicatesReport,
    ) {
        self.draw_block(&heading("Duplicate code report", BODY_SIZE + 5.0));
        let mut rows = vec![
            (
                "Repository".to_string(),
                RowValue::Text(repository.to_string()),
                false,
            ),
            (
                "Branch".to_string(),
                RowValue::Text(branch.to_string()),
                false,
            ),
            (
                "Generated".to_string(),
                RowValue::Text(format_timestamp()),
                false,
            ),
            (
                "Threshold".to_string(),
                RowValue::Text(format!("{}%", report.threshold_percent())),
                false,
            ),
            (
                "Files scanned".to_string(),
                RowValue::Text(report.files_scanned.to_string()),
                false,
            ),
            (
                "Functions analysed".to_string(),
                RowValue::Text(report.functions_analyzed.to_string()),
                false,
            ),
        ];
        // On a branch, record what the analysis was restricted to.
        if let orangu::duplicates::Scope::Patch {
            base,
            new_functions,
        } = &report.scope
        {
            rows.push((
                "Compared against".to_string(),
                RowValue::Text(base.clone()),
                false,
            ));
            rows.push((
                "New/changed functions".to_string(),
                RowValue::Text(new_functions.to_string()),
                false,
            ));
        }
        rows.push((
            "Candidate pairs".to_string(),
            RowValue::Text(report.pairs.len().to_string()),
            false,
        ));
        self.draw_kv_table(&rows);
    }

    /// Page 1 of the pull request export: the repository name, generation
    /// time, the open pull requests broken down by status — how many are
    /// open in total, how many are ready for review (not a draft), and how
    /// many have merge conflicts — and the oldest/newest open pull request.
    fn draw_pr_stats_page(&mut self, repository: &str, prs: &[PullRequestDetail]) {
        self.draw_block(&heading("Pull Requests", BODY_SIZE + 5.0));
        self.draw_kv_table(&pr_stats_rows(repository, prs));
    }

    /// One pull request's detail page: the linked title heading, a field
    /// table (author, dates, branch, draft/conflict status, comment count,
    /// assignees, reviewers, labels), then the changed files — one line per
    /// file, full path, additions in green and deletions in red. The page is
    /// assumed freshly started (`new_page` just called); the changed-files
    /// list spills onto further pages on its own via `draw_blocks`' normal
    /// overflow handling when there are many files.
    fn draw_pr_detail_page(&mut self, pr: &PullRequestDetail) {
        self.draw_block(&pr_title_block(pr));
        self.draw_kv_table(&pr_header_rows(pr));
        if !pr.files.is_empty() {
            self.draw_blocks(&build_changed_file_blocks(pr));
        }
        // A single atomic page break if the table (small and bounded — see
        // `last_comment_table_height_mm`) does not fit in what's left.
        if self.cursor_y - last_comment_table_height_mm(pr, &self.fonts) < CONTENT_BOTTOM_MM {
            self.new_page();
        }
        let (author, body) = match &pr.last_comment {
            Some(comment) => (comment.author.as_str(), comment.body.as_str()),
            None => ("N/A", "N/A"),
        };
        self.draw_last_comment_table(author, body);
    }

    /// The "Last comment" table following a pull request's changed files: a
    /// header row spanning both columns, then one data row with the
    /// comment's author (left) and its text (right, word-wrapped to fit).
    /// `"N/A"`/`"N/A"` when the pull request has no comment.
    fn draw_last_comment_table(&mut self, author: &str, body: &str) {
        let (divider, row_height, lines) = last_comment_table_layout(author, body, &self.fonts);
        let x0 = MARGIN_MM;
        let x1 = PAGE_WIDTH_MM - MARGIN_MM;
        let top = self.cursor_y;
        let header_bottom = top - row_height;
        let bottom = header_bottom - lines.len().max(1) as f32 * row_height;

        self.text(
            "Last comment",
            true,
            x0 + 3.0,
            top - row_height * 0.68,
            BODY_SIZE,
            TEXT_COLOR,
        );
        self.text(
            author,
            false,
            x0 + 3.0,
            header_bottom - row_height * 0.68,
            BODY_SIZE,
            TEXT_COLOR,
        );
        for (index, line) in lines.iter().enumerate() {
            let baseline = header_bottom - (index as f32 + 0.68) * row_height;
            self.text(line, false, divider + 3.0, baseline, BODY_SIZE, TEXT_COLOR);
        }

        // Outer box, the rule under the header, and the column divider
        // (alongside the data row only — the header spans both columns).
        self.rule(x0, top, x1, top, GRID_COLOR, 0.4);
        self.rule(x0, header_bottom, x1, header_bottom, GRID_COLOR, 0.4);
        self.rule(x0, bottom, x1, bottom, GRID_COLOR, 0.4);
        self.rule(x0, top, x0, bottom, GRID_COLOR, 0.4);
        self.rule(x1, top, x1, bottom, GRID_COLOR, 0.4);
        self.rule(divider, header_bottom, divider, bottom, GRID_COLOR, 0.4);

        self.cursor_y = bottom - 8.0;
    }

    /// A two-column table: bold labels on the left, values on the right. Each
    /// row is `(label, value, bold)`; `bold` renders that row's value in bold
    /// too (used to draw attention to e.g. a pull request's `Draft`/
    /// `Conflicts` value when it's `Yes`) — plain for the common case.
    fn draw_kv_table(&mut self, rows: &[(String, RowValue, bool)]) {
        let x0 = MARGIN_MM;
        let label_width = rows
            .iter()
            .map(|(key, _, _)| self.fonts.text_width_mm(key, true, false, BODY_SIZE))
            .fold(0.0_f32, f32::max);
        let divider = x0 + 6.0 + label_width + 6.0;
        // Span the full content width (page width minus margins) rather than
        // a fixed value-column width, matching `draw_last_comment_table`.
        let x1 = PAGE_WIDTH_MM - MARGIN_MM;
        let row_height = BODY_SIZE * 1.9 * PT_TO_MM;
        let top = self.cursor_y;
        for (index, (key, value, bold)) in rows.iter().enumerate() {
            let baseline = top - (index as f32 + 0.68) * row_height;
            self.text(key, true, x0 + 3.0, baseline, BODY_SIZE, TEXT_COLOR);
            match value {
                RowValue::Text(text) => {
                    self.text(text, *bold, divider + 3.0, baseline, BODY_SIZE, TEXT_COLOR);
                }
                RowValue::Link(url) if !url.is_empty() => {
                    self.draw_link(url, divider + 3.0, baseline, *bold);
                }
                RowValue::Link(_) => {
                    self.text(
                        "unknown",
                        *bold,
                        divider + 3.0,
                        baseline,
                        BODY_SIZE,
                        TEXT_COLOR,
                    );
                }
                RowValue::LinkWithText(url, text) if !url.is_empty() => {
                    self.draw_link(url, divider + 3.0, baseline, *bold);
                    let link_width = self.fonts.text_width_mm(url, *bold, false, BODY_SIZE);
                    self.text(
                        &format!(" {text}"),
                        false,
                        divider + 3.0 + link_width,
                        baseline,
                        BODY_SIZE,
                        TEXT_COLOR,
                    );
                }
                RowValue::LinkWithText(_, text) => {
                    self.text(text, *bold, divider + 3.0, baseline, BODY_SIZE, TEXT_COLOR);
                }
                RowValue::Reviewers(reviewers) => {
                    self.draw_reviewer_list(reviewers, divider + 3.0, baseline);
                }
            }
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

    /// Draw a clickable URL: the text in the brand colour (so it reads as a
    /// link, matching the duplicates export's source-location links), with a
    /// `Op::LinkAnnotation` covering the drawn text.
    fn draw_link(&mut self, url: &str, x: f32, baseline: f32, bold: bool) {
        self.text(url, bold, x, baseline, BODY_SIZE, BRAND_COLOR);
        let width = self.fonts.text_width_mm(url, bold, false, BODY_SIZE);
        self.ops.push(Op::LinkAnnotation {
            link: LinkAnnotation::new(
                Rect {
                    x: Mm(x).into(),
                    y: Mm(baseline - 1.5).into(),
                    width: Mm(width).into(),
                    height: Mm(BODY_SIZE * PT_TO_MM + 2.5).into(),
                    mode: None,
                    winding_order: None,
                },
                Actions::uri(url.to_string()),
                Some(BorderArray::Solid([0.0, 0.0, 0.0])),
                None,
                None,
            ),
        });
    }

    /// Draw a pull request's reviewers, comma-separated, each name followed
    /// by its status icon (see [`draw_review_icon`](Self::draw_review_icon))
    /// instead of the review state spelled out. `"none"` when there are no
    /// reviewers.
    fn draw_reviewer_list(&mut self, reviewers: &[PullRequestReviewer], x: f32, baseline: f32) {
        if reviewers.is_empty() {
            self.text("none", false, x, baseline, BODY_SIZE, TEXT_COLOR);
            return;
        }
        let mut cursor_x = x;
        for (index, reviewer) in reviewers.iter().enumerate() {
            if index > 0 {
                let separator = ", ";
                self.text(separator, false, cursor_x, baseline, BODY_SIZE, TEXT_COLOR);
                cursor_x += self.fonts.text_width_mm(separator, false, false, BODY_SIZE);
            }
            self.text(
                &reviewer.login,
                false,
                cursor_x,
                baseline,
                BODY_SIZE,
                TEXT_COLOR,
            );
            cursor_x += self
                .fonts
                .text_width_mm(&reviewer.login, false, false, BODY_SIZE);
            cursor_x += 1.5;
            cursor_x += self.draw_review_icon(review_icon(&reviewer.state), cursor_x, baseline);
        }
    }

    /// Draw one reviewer's status icon at `(x, baseline)`, returning the mm
    /// width it occupies so the caller can advance past it. A green
    /// checkmark for an approval — drawn as two short vector strokes, since
    /// the embedded Red Hat Text font has no checkmark glyph — a bold red
    /// "X" for a change request, or a plain "?" for anything else.
    fn draw_review_icon(&mut self, icon: ReviewIcon, x: f32, baseline: f32) -> f32 {
        match icon {
            ReviewIcon::Check => self.draw_checkmark(x, baseline, BODY_SIZE, STATUS_GREEN),
            ReviewIcon::Cross => {
                self.text("X", true, x, baseline, BODY_SIZE, STATUS_RED);
                self.fonts.text_width_mm("X", true, false, BODY_SIZE)
            }
            ReviewIcon::Question => {
                self.text("?", false, x, baseline, BODY_SIZE, TEXT_COLOR);
                self.fonts.text_width_mm("?", false, false, BODY_SIZE)
            }
        }
    }

    /// Draw a small checkmark (two vector strokes — a short down-stroke then
    /// a longer up-stroke) at `(x, baseline)`, sized to `size`, returning its
    /// mm width.
    fn draw_checkmark(&mut self, x: f32, baseline: f32, size: f32, color: (f32, f32, f32)) -> f32 {
        let h = size * 0.5 * PT_TO_MM;
        let w = h * 1.1;
        let mid_x = x + w * 0.4;
        self.rule(x, baseline + h * 0.35, mid_x, baseline, color, 1.0);
        self.rule(mid_x, baseline, x + w, baseline + h * 0.8, color, 1.0);
        w
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
    /// appendix) and its starting page. `ok` is `Some(true)`/`Some(false)`
    /// to draw a green checkmark/red "X" right after the title (used by the
    /// `/export pr` table of contents to flag a draft or conflicting pull
    /// request at a glance), or `None` to draw no icon at all.
    fn draw_toc(&mut self, rows: &[(&str, usize, Option<bool>)]) {
        self.draw_block(&heading("Table of Contents", BODY_SIZE + 5.0));
        let size = BODY_SIZE + 1.0;
        let row_height = size * 1.8 * PT_TO_MM;
        for &(title, page, ok) in rows {
            if self.cursor_y - row_height < CONTENT_BOTTOM_MM {
                self.new_page();
            }
            self.cursor_y -= row_height;
            // The entry (title and page number) links to the start of its
            // chapter, drawn in the brand colour to read as a link.
            self.text(title, false, MARGIN_MM, self.cursor_y, size, BRAND_COLOR);
            if let Some(ok) = ok {
                let icon_x = MARGIN_MM + self.fonts.text_width_mm(title, false, false, size) + 3.0;
                if ok {
                    self.draw_checkmark(icon_x, self.cursor_y, size, STATUS_GREEN);
                } else {
                    self.text("X", true, icon_x, self.cursor_y, size, STATUS_RED);
                }
            }
            let number = page.to_string();
            let width = self.fonts.text_width_mm(&number, false, false, size);
            self.text(
                &number,
                false,
                PAGE_WIDTH_MM - MARGIN_MM - width,
                self.cursor_y,
                size,
                BRAND_COLOR,
            );
            self.link_to_page(page, self.cursor_y - 1.0, size * PT_TO_MM + 2.0);
        }
    }

    /// Attach an internal "go to page" link spanning the content width at the
    /// given baseline (`page` is 1-based). Used by the table of contents.
    fn link_to_page(&mut self, page: usize, y_mm: f32, height_mm: f32) {
        // The destination's `top` is the page height in points so the target
        // page is shown from its top edge.
        let top = PAGE_HEIGHT_MM / PT_TO_MM;
        self.ops.push(Op::LinkAnnotation {
            link: LinkAnnotation::new(
                Rect {
                    x: Mm(MARGIN_MM).into(),
                    y: Mm(y_mm).into(),
                    width: Mm(USABLE_WIDTH_MM).into(),
                    height: Mm(height_mm).into(),
                    mode: None,
                    winding_order: None,
                },
                Actions::go_to(Destination::Xyz {
                    page,
                    left: Some(0.0),
                    top: Some(top),
                    zoom: None,
                }),
                Some(BorderArray::Solid([0.0, 0.0, 0.0])),
                None,
                None,
            ),
        });
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
        link: None,
    }
}

/// The number of pages `blocks` occupy when laid out from a fresh page — used
/// to compute table-of-contents page numbers without drawing.
fn paginate(blocks: &[Block], fonts: &DocFonts) -> usize {
    paginate_from(blocks, fonts, CONTENT_TOP_MM).0
}

/// Like [`paginate`], but the first page starts with `start_cursor_y` mm of
/// vertical room already used — by a fixed-height header drawn ahead of these
/// blocks — instead of a fresh page's full height. Returns the page count and
/// the cursor position after the last block, so a fixed-height element that
/// follows (like the "Last comment" table) can be accounted for too, without
/// redrawing.
fn paginate_from(blocks: &[Block], fonts: &DocFonts, start_cursor_y: f32) -> (usize, f32) {
    let mut pages = 1;
    let mut cursor_y = start_cursor_y;
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
    (pages, cursor_y)
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

    fn duplicate_location(
        path: &str,
        name: &str,
        start: usize,
    ) -> orangu::duplicates::FunctionLocation {
        orangu::duplicates::FunctionLocation {
            path: PathBuf::from(path),
            name: name.to_string(),
            language: "Rust",
            start_line: start,
            end_line: start + 8,
        }
    }

    #[test]
    fn export_duplicates_renders_chapters() {
        let workspace = tempdir().expect("workspace");
        // Two percentage groups, so the export must build chapters and a TOC.
        let report = DuplicatesReport {
            root: workspace.path().to_path_buf(),
            threshold: 0.80,
            files_scanned: 3,
            functions_analyzed: 6,
            scope: orangu::duplicates::Scope::Project,
            pairs: vec![
                orangu::duplicates::DuplicatePair {
                    a: duplicate_location("a.rs", "foo", 1),
                    b: duplicate_location("b.rs", "bar", 20),
                    similarity: 1.0,
                },
                orangu::duplicates::DuplicatePair {
                    a: duplicate_location("c.rs", "baz", 5),
                    b: duplicate_location("d.rs", "qux", 40),
                    similarity: 0.92,
                },
            ],
        };
        let path = export_duplicates(workspace.path(), &report, "gemma").expect("export");
        assert!(path.exists());
        assert!(
            path.file_name()
                .unwrap()
                .to_string_lossy()
                .ends_with("-duplicates.pdf")
        );
        let bytes = std::fs::read(&path).expect("read pdf");
        assert!(bytes.starts_with(b"%PDF"));
    }

    #[test]
    fn export_duplicates_with_no_pairs_writes_a_pdf() {
        let workspace = tempdir().expect("workspace");
        let report = DuplicatesReport {
            root: workspace.path().to_path_buf(),
            threshold: 0.80,
            files_scanned: 1,
            functions_analyzed: 2,
            scope: orangu::duplicates::Scope::Project,
            pairs: vec![],
        };
        let path = export_duplicates(workspace.path(), &report, "gemma").expect("export");
        let bytes = std::fs::read(&path).expect("read pdf");
        assert!(bytes.starts_with(b"%PDF"));
    }

    fn sample_pull_request() -> PullRequestDetail {
        PullRequestDetail {
            number: 42,
            title: "Add pull completion".to_string(),
            author: "alice".to_string(),
            created_at: "2026-06-01 12:30:45".to_string(),
            updated_at: "2026-06-02 08:00:00".to_string(),
            base: "main".to_string(),
            head: "feature/completion".to_string(),
            draft: false,
            conflicting: Some(true),
            comment_count: 3,
            reviewers: vec![
                crate::git::PullRequestReviewer {
                    login: "bob".to_string(),
                    state: "Approved".to_string(),
                },
                crate::git::PullRequestReviewer {
                    login: "carol".to_string(),
                    state: "Pending".to_string(),
                },
            ],
            assignees: vec!["alice".to_string()],
            labels: vec!["enhancement".to_string()],
            files: vec![
                crate::git::ChangedFile {
                    path: "src/main.rs".to_string(),
                    additions: 8,
                    deletions: 2,
                },
                crate::git::ChangedFile {
                    path: "src/lib.rs".to_string(),
                    additions: 2,
                    deletions: 2,
                },
            ],
            last_comment: Some(crate::git::PullRequestComment {
                author: "dave".to_string(),
                body: "Looks good, thanks!".to_string(),
            }),
            url: "https://github.com/o/r/pull/42".to_string(),
        }
    }

    #[test]
    fn export_pr_renders_stats_toc_and_pages() {
        let workspace = tempdir().expect("workspace");
        let prs = vec![sample_pull_request()];
        let path = export_pr(workspace.path(), &prs, "gemma").expect("export");
        assert!(path.exists());
        assert!(
            path.file_name()
                .unwrap()
                .to_string_lossy()
                .ends_with("-pr.pdf")
        );
        // Unlike the other exports, the branch is not part of the filename.
        assert!(
            !path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .contains("nobranch")
        );
        let bytes = std::fs::read(&path).expect("read pdf");
        assert!(bytes.starts_with(b"%PDF"));
    }

    #[test]
    fn export_pr_with_no_pull_requests_still_writes_a_pdf() {
        let workspace = tempdir().expect("workspace");
        let path = export_pr(workspace.path(), &[], "gemma").expect("export");
        let bytes = std::fs::read(&path).expect("read pdf");
        assert!(bytes.starts_with(b"%PDF"));
    }

    #[test]
    fn pr_stats_page_counts_ready_and_conflicting() {
        let mut draft = sample_pull_request();
        draft.number = 1;
        draft.draft = true;
        draft.conflicting = Some(false);
        let mut ready_clean = sample_pull_request();
        ready_clean.number = 2;
        ready_clean.draft = false;
        ready_clean.conflicting = Some(false);
        let mut ready_conflicting = sample_pull_request();
        ready_conflicting.number = 3;
        ready_conflicting.draft = false;
        ready_conflicting.conflicting = Some(true);
        let prs = vec![draft, ready_clean, ready_conflicting];

        // Mirrors the counts draw_pr_stats_page computes: 3 open, 2 ready
        // (non-draft), 1 with conflicts.
        let ready = prs.iter().filter(|pr| !pr.draft).count();
        let conflicts = prs.iter().filter(|pr| pr.conflicting == Some(true)).count();
        assert_eq!((prs.len(), ready, conflicts), (3, 2, 1));

        let workspace = tempdir().expect("workspace");
        let path = export_pr(workspace.path(), &prs, "gemma").expect("export");
        let bytes = std::fs::read(&path).expect("read pdf");
        assert!(bytes.starts_with(b"%PDF"));
    }

    #[test]
    fn oldest_and_newest_pick_by_created_at_regardless_of_order() {
        let mut middle = sample_pull_request();
        middle.number = 1;
        middle.created_at = "2026-06-15 00:00:00".to_string();
        middle.url = "https://github.com/o/r/pull/1".to_string();
        let mut oldest = sample_pull_request();
        oldest.number = 2;
        oldest.created_at = "2026-01-01 00:00:00".to_string();
        oldest.url = "https://github.com/o/r/pull/2".to_string();
        let mut newest = sample_pull_request();
        newest.number = 3;
        newest.created_at = "2026-12-31 00:00:00".to_string();
        newest.url = "https://github.com/o/r/pull/3".to_string();
        // Deliberately out of chronological order, so a naive first/last
        // pick would get this wrong.
        let prs = vec![middle, oldest.clone(), newest.clone()];

        let (found_oldest, found_newest) = oldest_and_newest(&prs);
        assert_eq!(found_oldest.map(|pr| pr.number), Some(2));
        assert_eq!(found_newest.map(|pr| pr.number), Some(3));
    }

    #[test]
    fn oldest_and_newest_skip_missing_created_at() {
        let mut unknown = sample_pull_request();
        unknown.created_at = String::new();
        let (oldest, newest) = oldest_and_newest(std::slice::from_ref(&unknown));
        assert!(oldest.is_none());
        assert!(newest.is_none());
    }

    #[test]
    fn pr_stats_rows_include_oldest_and_newest_as_links() {
        let mut oldest = sample_pull_request();
        oldest.number = 1;
        oldest.created_at = "2026-01-01 00:00:00".to_string();
        oldest.url = "https://github.com/o/r/pull/1".to_string();
        let mut newest = sample_pull_request();
        newest.number = 2;
        newest.created_at = "2026-12-31 00:00:00".to_string();
        newest.url = "https://github.com/o/r/pull/2".to_string();
        let prs = vec![oldest, newest];

        let rows = pr_stats_rows("orangu", &prs);
        let find = |label: &str| {
            rows.iter()
                .find(|(key, _, _)| key == label)
                .map(|(_, value, _)| value)
        };
        assert_eq!(
            find("Oldest"),
            Some(&RowValue::LinkWithText(
                "https://github.com/o/r/pull/1".to_string(),
                "2026-01-01 00:00:00".to_string(),
            ))
        );
        assert_eq!(
            find("Newest"),
            Some(&RowValue::LinkWithText(
                "https://github.com/o/r/pull/2".to_string(),
                "2026-12-31 00:00:00".to_string(),
            ))
        );
    }

    #[test]
    fn pr_stats_rows_use_na_for_oldest_and_newest_when_empty() {
        let rows = pr_stats_rows("orangu", &[]);
        let find = |label: &str| {
            rows.iter()
                .find(|(key, _, _)| key == label)
                .map(|(_, value, _)| value)
        };
        assert_eq!(find("Oldest"), Some(&RowValue::Text("N/A".to_string())));
        assert_eq!(find("Newest"), Some(&RowValue::Text("N/A".to_string())));
    }

    #[test]
    fn pr_stats_rows_oldest_is_empty_with_a_single_pull_request() {
        let mut pr = sample_pull_request();
        pr.created_at = "2026-03-01 00:00:00".to_string();
        pr.url = "https://github.com/o/r/pull/1".to_string();
        let rows = pr_stats_rows("orangu", std::slice::from_ref(&pr));
        let find = |label: &str| {
            rows.iter()
                .find(|(key, _, _)| key == label)
                .map(|(_, value, _)| value)
        };
        // "Oldest" and "Newest" would be the same entry with only one pull
        // request, so "Oldest" is left empty instead of repeating it.
        assert_eq!(find("Oldest"), Some(&RowValue::Text(String::new())));
        assert_eq!(
            find("Newest"),
            Some(&RowValue::LinkWithText(
                "https://github.com/o/r/pull/1".to_string(),
                "2026-03-01 00:00:00".to_string(),
            ))
        );
    }

    #[test]
    fn pr_title_block_links_to_the_forge() {
        let pr = sample_pull_request();
        let title = pr_title_block(&pr);
        assert_eq!(title.spans[0].text, "#42 Add pull completion");
        assert_eq!(
            title.link.as_deref(),
            Some("https://github.com/o/r/pull/42")
        );
    }

    /// Look up a `pr_header_rows` row's text value by label — `None` for the
    /// "Reviewers" row, whose value is the raw reviewer list, not text.
    fn find_row_text<'a>(
        rows: &'a [(String, RowValue, bool)],
        label: &str,
    ) -> Option<(&'a str, bool)> {
        rows.iter()
            .find(|(key, _, _)| key == label)
            .and_then(|(_, value, bold)| match value {
                RowValue::Text(text) | RowValue::Link(text) => Some((text.as_str(), *bold)),
                RowValue::LinkWithText(_, _) | RowValue::Reviewers(_) => None,
            })
    }

    #[test]
    fn pr_header_rows_report_every_field() {
        let pr = sample_pull_request();
        let rows = pr_header_rows(&pr);
        assert_eq!(find_row_text(&rows, "Author"), Some(("alice", false)));
        let link_row = rows
            .iter()
            .find(|(key, _, _)| key == "Link")
            .map(|(_, value, _)| value);
        assert_eq!(
            link_row,
            Some(&RowValue::Link(
                "https://github.com/o/r/pull/42".to_string()
            ))
        );
        assert_eq!(find_row_text(&rows, "Draft"), Some(("No", false)));
        // "Yes" is bolded, so a draft or a conflict catches the eye.
        assert_eq!(find_row_text(&rows, "Conflicts"), Some(("Yes", true)));
        assert_eq!(find_row_text(&rows, "Comments"), Some(("3", false)));
        assert_eq!(find_row_text(&rows, "Labels"), Some(("enhancement", false)));
        // Reviewers carries the raw list — icon selection happens at draw
        // time (`review_icon`/`draw_review_icon`), not as formatted text.
        let reviewers_row = rows
            .iter()
            .find(|(key, _, _)| key == "Reviewers")
            .map(|(_, value, _)| value);
        assert_eq!(
            reviewers_row,
            Some(&RowValue::Reviewers(pr.reviewers.clone()))
        );
        // The changed-files table has its own section — no longer duplicated
        // as a row in the header table.
        assert!(find_row_text(&rows, "Changed files").is_none());
    }

    #[test]
    fn pr_header_rows_bolds_draft_yes_and_leaves_no_plain() {
        let mut pr = sample_pull_request();
        pr.draft = true;
        pr.conflicting = Some(false);
        let rows = pr_header_rows(&pr);
        assert_eq!(find_row_text(&rows, "Draft"), Some(("Yes", true)));
        assert_eq!(find_row_text(&rows, "Conflicts"), Some(("No", false)));
    }

    #[test]
    fn pr_header_rows_link_falls_back_to_unknown_without_a_url() {
        let mut pr = sample_pull_request();
        pr.url = String::new();
        let rows = pr_header_rows(&pr);
        assert_eq!(
            rows.iter()
                .find(|(key, _, _)| key == "Link")
                .map(|(_, value, _)| value),
            Some(&RowValue::Link(String::new()))
        );
        // draw_kv_table renders an empty Link as plain "unknown" text rather
        // than an empty, unclickable link.
        let workspace = tempdir().expect("workspace");
        let path = export_pr(workspace.path(), std::slice::from_ref(&pr), "gemma").expect("export");
        let bytes = std::fs::read(&path).expect("read pdf");
        assert!(bytes.starts_with(b"%PDF"));
    }

    #[test]
    fn review_icon_maps_states_to_icons() {
        assert_eq!(review_icon("Approved"), ReviewIcon::Check);
        assert_eq!(review_icon("Changes requested"), ReviewIcon::Cross);
        // GitLab's "Requested" (a review requested but not yet given) is not
        // a positive verdict, so it gets the same icon as an explicit change
        // request rather than the ambiguous "?".
        assert_eq!(review_icon("Requested"), ReviewIcon::Cross);
        assert_eq!(review_icon("Commented"), ReviewIcon::Question);
        assert_eq!(review_icon("Pending"), ReviewIcon::Question);
        // Anything else (e.g. a dismissed review) is not a meaningful
        // verdict either, so it falls back to the same "?".
        assert_eq!(review_icon("Dismissed"), ReviewIcon::Question);
        assert_eq!(review_icon("Unknown"), ReviewIcon::Question);
    }

    #[test]
    fn pr_is_ready_requires_not_draft_and_not_conflicting() {
        let mut pr = sample_pull_request();
        pr.draft = false;
        pr.conflicting = Some(false);
        assert!(pr_is_ready(&pr));

        pr.conflicting = None;
        assert!(
            pr_is_ready(&pr),
            "an unknown mergeable state is not a conflict"
        );

        pr.conflicting = Some(true);
        assert!(!pr_is_ready(&pr));

        pr.conflicting = Some(false);
        pr.draft = true;
        assert!(!pr_is_ready(&pr));
    }

    #[test]
    fn changed_file_blocks_color_additions_green_and_deletions_red() {
        let pr = sample_pull_request();
        let blocks = build_changed_file_blocks(&pr);
        assert_eq!(blocks.len(), 2);
        let main_rs = &blocks[0];
        assert_eq!(main_rs.spans[0].text, "src/main.rs  ");
        assert_eq!(main_rs.spans[1].text, "+8");
        assert_eq!(main_rs.spans[1].color, Some(STATUS_GREEN));
        assert_eq!(main_rs.spans[3].text, "-2");
        assert_eq!(main_rs.spans[3].color, Some(STATUS_RED));
    }

    #[test]
    fn no_changed_files_yields_no_blocks() {
        let mut pr = sample_pull_request();
        pr.files.clear();
        assert!(build_changed_file_blocks(&pr).is_empty());
    }

    #[test]
    fn last_comment_table_layout_wraps_the_body_and_places_the_divider() {
        let fonts = Pdf::new("t", "m").expect("pdf").fonts;
        let (divider, row_height, lines) =
            last_comment_table_layout("dave", "Looks good, thanks!", &fonts);
        assert!(divider > MARGIN_MM);
        assert!(row_height > 0.0);
        assert_eq!(lines.join(" "), "Looks good, thanks!");
    }

    #[test]
    fn last_comment_table_truncates_very_long_bodies() {
        let fonts = Pdf::new("t", "m").expect("pdf").fonts;
        let long_body = "x".repeat(LAST_COMMENT_MAX_CHARS + 200);
        let (_, _, lines) = last_comment_table_layout("dave", &long_body, &fonts);
        let joined: String = lines.join("");
        // Truncated to the cap plus the ellipsis marker, not the full length.
        assert!(joined.chars().count() <= LAST_COMMENT_MAX_CHARS + 1);
        assert!(joined.ends_with('…'));
    }

    #[test]
    fn last_comment_table_height_matches_no_comment_and_with_comment() {
        let fonts = Pdf::new("t", "m").expect("pdf").fonts;
        let mut pr = sample_pull_request();
        let with_comment = last_comment_table_height_mm(&pr, &fonts);
        pr.last_comment = None;
        let without_comment = last_comment_table_height_mm(&pr, &fonts);
        // Both a real comment and the "N/A" placeholder are one line, so the
        // heights should match (both render one header row + one data row).
        assert!((with_comment - without_comment).abs() < 0.01);
    }

    #[test]
    fn export_pr_draws_last_comment_table_with_na_fallback() {
        let workspace = tempdir().expect("workspace");
        let mut pr = sample_pull_request();
        pr.last_comment = None;
        let path = export_pr(workspace.path(), std::slice::from_ref(&pr), "gemma").expect("export");
        let bytes = std::fs::read(&path).expect("read pdf");
        assert!(bytes.starts_with(b"%PDF"));
    }

    #[test]
    fn duplicate_chapters_group_by_percentage() {
        let workspace = tempdir().expect("workspace");
        let report = DuplicatesReport {
            root: workspace.path().to_path_buf(),
            threshold: 0.80,
            files_scanned: 4,
            functions_analyzed: 8,
            scope: orangu::duplicates::Scope::Project,
            pairs: vec![
                orangu::duplicates::DuplicatePair {
                    a: duplicate_location("a.rs", "foo", 1),
                    b: duplicate_location("b.rs", "bar", 20),
                    similarity: 1.0,
                },
                orangu::duplicates::DuplicatePair {
                    a: duplicate_location("c.rs", "qux", 5),
                    b: duplicate_location("d.rs", "quux", 40),
                    similarity: 1.0,
                },
                orangu::duplicates::DuplicatePair {
                    a: duplicate_location("e.rs", "baz", 5),
                    b: duplicate_location("f.rs", "corge", 40),
                    similarity: 0.90,
                },
            ],
        };
        // No git repo in the tempdir, so locations are plain (unlinked) text.
        let linker = SourceLinker::new(workspace.path());
        let chapters = build_duplicate_chapters(&report, &linker);
        // Two distinct percentages → two chapters, titled by percentage.
        assert_eq!(chapters.len(), 2);
        assert_eq!(chapters[0].0, "100% similar");
        assert_eq!(chapters[1].0, "90% similar");
        // First chapter: heading + 2 pairs × (sub-heading + 2 locations) = 5 blocks.
        assert_eq!(chapters[0].1.len(), 1 + 2 * 3);
        // The location text uses the `path:start–end` form.
        let location_texts: Vec<&str> = chapters[1].1[1..]
            .iter()
            .flat_map(|block| block.spans.iter())
            .map(|span| span.text.as_str())
            .collect();
        assert!(location_texts.contains(&"e.rs:5–13"));
    }

    #[test]
    fn forge_links_use_github_and_gitlab_formats() {
        use crate::git::forge_web_from_url as parse;
        let gh = parse("https://github.com/owner/repo.git").expect("github");
        assert_eq!(
            gh.blob_url("main", "src/a.rs", 10, 20),
            "https://github.com/owner/repo/blob/main/src/a.rs#L10-L20"
        );
        let ssh = parse("git@github.com:owner/repo.git").expect("github ssh");
        assert_eq!(
            ssh.blob_url("main", "src/a.rs", 7, 7),
            "https://github.com/owner/repo/blob/main/src/a.rs#L7"
        );
        let gl = parse("https://gitlab.com/group/proj").expect("gitlab");
        assert_eq!(
            gl.blob_url("dev", "x.rs", 3, 9),
            "https://gitlab.com/group/proj/-/blob/dev/x.rs#L3-9"
        );
        // Unknown hosts are not linked.
        assert!(parse("https://example.com/owner/repo.git").is_none());
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
