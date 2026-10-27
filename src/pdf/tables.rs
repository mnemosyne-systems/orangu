// Copyright (C) 2026 The orangu community
// SPDX-License-Identifier: GPL-3.0-or-later

//! Table geometry comes from pdfni. This adapter validates its cells, preserves
//! spans, and renders complete tables so context limits cannot cut a value in half.

use pdfni::{BBox, ExtractOptions, PagePart};
use serde::{Deserialize, Serialize};
use std::ops::Range;

pub(super) const MAX_GLYPHS: usize = 60_000;
pub(super) const MAX_EDGES: usize = 512;
const MAX_CELLS: usize = 4096;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PdfTableCell {
    /// One-based row and column, including covered positions of spanning cells.
    pub row: usize,
    pub column: usize,
    pub row_span: usize,
    pub column_span: usize,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PdfTable {
    pub rows: usize,
    pub columns: usize,
    pub cells: Vec<PdfTableCell>,
    pub bounds: BBox,
    pub method: String,
    pub warning: Option<String>,
}

#[derive(Default)]
pub(super) struct PageContent {
    pub text: String,
    pub tables: Vec<PdfTable>,
    pub body_text: Option<String>,
    pub warning: Option<String>,
}

impl PageContent {
    pub fn plain(text: String) -> Self {
        Self {
            text,
            ..Default::default()
        }
    }
}

impl PdfTable {
    pub fn context_html(&self, number: usize) -> String {
        let mut out = format!(
            "<table>\n<caption>Table {number}: {} rows, {} columns</caption>\n",
            self.rows, self.columns
        );
        for row in 1..=self.rows {
            out.push_str("<tr>");
            for cell in self.cells.iter().filter(|cell| cell.row == row) {
                out.push_str(&format!(
                    "<td rowspan=\"{}\" colspan=\"{}\">{}</td>",
                    cell.row_span,
                    cell.column_span,
                    escape_html(&cell.text)
                ));
            }
            out.push_str("</tr>\n");
        }
        out.push_str("</table>\n");
        if let Some(warning) = &self.warning {
            out.push_str(&format!("[Table extraction warning: {warning}]\n"));
        }
        out
    }
}

fn escape_html(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\r', "")
        .replace('\n', "<br/>")
}

/// Apply a caller's character limit without exposing half of a table.
pub(crate) fn truncate_context(text: &str, max_chars: usize) -> String {
    let Some((mut end, _)) = text.char_indices().nth(max_chars) else {
        return text.to_owned();
    };
    let mut omitted_table = false;
    for (start, _) in text.match_indices("<table>\n") {
        let Some(close) = text[start..].find("</table>") else {
            continue;
        };
        if start < end && end < start + close + "</table>".len() {
            end = start;
            omitted_table = true;
            break;
        }
    }
    let reason = if omitted_table {
        "An entire table and remaining content were omitted at the character limit; no partial table values are provided"
    } else {
        "PDF content truncated at the character limit"
    };
    format!("{}\n[{reason}]", &text[..end])
}

/// Character ranges refer to the original plain text. Removing table ranges
/// keeps narrative text without repeating a misleading flattened table.
pub(super) fn reconstruct(
    text: String,
    part: PagePart,
    ranges: Vec<(Range<usize>, BBox)>,
    overflow: bool,
) -> PageContent {
    if overflow {
        return PageContent {
            text,
            warning: Some("Table layout limits exceeded; cell associations are unavailable".into()),
            ..Default::default()
        };
    }
    let possible_grid = part
        .edges
        .iter()
        .filter(|e| e.orientation == pdfni::Orientation::Horizontal)
        .count()
        >= 2
        && part
            .edges
            .iter()
            .filter(|e| e.orientation == pdfni::Orientation::Vertical)
            .count()
            >= 2;
    let mut options = ExtractOptions::default();
    // Preserve actual border geometry. Text wrapped inside a cell must not
    // create extra rows or split an intentional merged cell.
    options.lattice.refine_rows = false;
    options.lattice.refine_cols = false;
    options.lattice.col_check = false;
    let result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pdfni::extract_from_parts(vec![part.clone()], &options)
    })) {
        Ok(result) => result,
        Err(_) => {
            return PageContent {
                text,
                warning: Some(
                    "Table analysis failed; flattened text cannot establish cell associations"
                        .into(),
                ),
                ..Default::default()
            };
        }
    };
    let mut tables = Vec::new();
    let mut rejected = false;
    let mut remaining_cells = MAX_CELLS;
    if let Some(page) = result.pages.first() {
        for raw in page.tables.iter().take(32) {
            match validate(raw, &part) {
                Some(table) if table.cells.len() <= remaining_cells => {
                    remaining_cells -= table.cells.len();
                    tables.push(table);
                }
                _ => rejected = true,
            }
        }
        rejected |= page.tables.len() > 32;
    }
    let body_text = if tables.is_empty() {
        None
    } else {
        let mut body = String::new();
        let mut cursor = 0;
        for (range, bounds) in ranges {
            if range.start < cursor || range.end > text.len() {
                continue;
            }
            body.push_str(&text[cursor..range.start]);
            if !tables.iter().any(|t| t.bounds.contains_center(&bounds)) {
                body.push_str(&text[range.clone()]);
            }
            cursor = range.end;
        }
        body.push_str(&text[cursor..]);
        Some(
            body.lines()
                .filter(|line| !line.trim().is_empty())
                .collect::<Vec<_>>()
                .join("\n"),
        )
    };
    let warning = (rejected || (possible_grid && tables.is_empty())).then(||
        "A possible table could not be validated. Flattened text may not preserve rows, columns, or merged cells; do not infer cell associations".to_string());
    PageContent {
        text,
        tables,
        body_text,
        warning,
    }
}

fn valid_bounds(b: &BBox) -> bool {
    [b.x0, b.x1, b.top, b.bottom].iter().all(|v| v.is_finite())
        && b.width() > 0.0
        && b.height() > 0.0
}

fn axes(mut values: Vec<f64>) -> Vec<f64> {
    values.sort_by(f64::total_cmp);
    values.dedup_by(|a, b| (*a - *b).abs() < 0.5);
    values
}

fn axis_index(axis: &[f64], value: f64) -> Option<usize> {
    axis.iter().position(|v| (*v - value).abs() < 0.5)
}

fn validate(raw: &pdfni::Table, part: &PagePart) -> Option<PdfTable> {
    if !valid_bounds(&raw.bbox)
        || raw.n_rows < 2
        || raw.n_cols < 2
        || raw.n_rows.checked_mul(raw.n_cols)? > MAX_CELLS
    {
        return None;
    }
    let ruled = raw.extraction_method == "lattice";
    let geometric: Vec<_> = raw
        .data
        .iter()
        .flatten()
        // pdfni uses the entire table bounds for empty borderless cells.
        // They carry no glyphs; include them later by their row/column index.
        .filter(|cell| valid_bounds(&cell.bbox) && (ruled || !cell.text.trim().is_empty()))
        .collect();
    if geometric.len() < 4 {
        return None;
    }
    // Every source character inside the table must belong to exactly one
    // cell. Reject overlapping or dropped text instead of choosing a value.
    for glyph in part.glyphs.iter().filter(|g| !g.ch.is_whitespace()) {
        let bounds = BBox {
            x0: glyph.left,
            x1: glyph.right,
            top: glyph.top,
            bottom: glyph.bottom,
        };
        if raw.bbox.contains_center(&bounds)
            && geometric
                .iter()
                .filter(|cell| cell.bbox.contains_center(&bounds))
                .count()
                != 1
        {
            return None;
        }
    }
    let (rows, columns, cells) = if ruled {
        let xs = axes(
            geometric
                .iter()
                .flat_map(|c| [c.bbox.x0, c.bbox.x1])
                .collect(),
        );
        let ys = axes(
            geometric
                .iter()
                .flat_map(|c| [c.bbox.top, c.bbox.bottom])
                .collect(),
        );
        let rows = ys.len().checked_sub(1)?;
        let columns = xs.len().checked_sub(1)?;
        if rows < 2 || columns < 2 {
            return None;
        }
        if rows.checked_mul(columns)? > MAX_CELLS {
            return None;
        }
        let mut coverage = vec![false; rows * columns];
        let mut cells = Vec::new();
        for cell in geometric {
            let r = axis_index(&ys, cell.bbox.top)?;
            let c = axis_index(&xs, cell.bbox.x0)?;
            let end_r = axis_index(&ys, cell.bbox.bottom)?;
            let end_c = axis_index(&xs, cell.bbox.x1)?;
            if end_r <= r || end_c <= c {
                return None;
            }
            for rr in r..end_r {
                for cc in c..end_c {
                    if std::mem::replace(&mut coverage[rr * columns + cc], true) {
                        return None;
                    }
                }
            }
            cells.push(PdfTableCell {
                row: r + 1,
                column: c + 1,
                row_span: end_r - r,
                column_span: end_c - c,
                text: cell.text.clone(),
            });
        }
        // Missing geometry is not evidence for an empty cell.
        if coverage.iter().any(|covered| !covered) {
            return None;
        }
        (rows, columns, cells)
    } else {
        if raw.data.len() != raw.n_rows || raw.data.iter().any(|r| r.len() != raw.n_cols) {
            return None;
        }
        // A lone spanning header in an unruled table is ambiguous.
        if raw
            .data
            .iter()
            .any(|r| r.iter().filter(|c| !c.text.trim().is_empty()).count() == 1)
        {
            return None;
        }
        let cells = raw
            .data
            .iter()
            .enumerate()
            .flat_map(|(r, row)| {
                row.iter().enumerate().map(move |(c, cell)| PdfTableCell {
                    row: r + 1,
                    column: c + 1,
                    row_span: 1,
                    column_span: 1,
                    text: cell.text.clone(),
                })
            })
            .collect();
        (raw.n_rows, raw.n_cols, cells)
    };
    let mut cells: Vec<PdfTableCell> = cells;
    cells.sort_by_key(|c| (c.row, c.column));
    Some(PdfTable { rows, columns, cells, bounds: raw.bbox, method: raw.extraction_method.into(), warning: (!ruled).then(|| "Column structure inferred from text alignment; merged cells cannot be verified without borders".into()) })
}

/// Use imageproc's directional morphology to isolate scanned grid lines.
/// Coordinates are converted back to PDF points before table detection.
pub(super) fn raster_edges(
    rgb: &[u8],
    width: u32,
    height: u32,
    scale: f64,
) -> Vec<pdfni::EdgePart> {
    use image::{GrayImage, Luma};
    use imageproc::morphology::{Mask, grayscale_open};
    let mut ink = GrayImage::new(width, height);
    for (pixel, channels) in ink.pixels_mut().zip(rgb.as_chunks::<3>().0) {
        pixel.0[0] = if channels.iter().map(|v| u16::from(*v)).sum::<u16>() < 3 * 190 {
            255
        } else {
            0
        };
    }
    let ratio = (1600.0 / f64::from(width.max(height))).min(1.0);
    let ink = if ratio < 1.0 {
        image::imageops::resize(
            &ink,
            (f64::from(width) * ratio).round().max(1.0) as u32,
            (f64::from(height) * ratio).round().max(1.0) as u32,
            image::imageops::FilterType::Triangle,
        )
    } else {
        ink
    };
    let (width, height) = ink.dimensions();
    let scale = scale * ratio;
    // Measure in page points so downsampling a full-size page does not
    // discard short border segments around merged cells.
    let kernel = ((scale * 22.0).ceil() as u32).clamp(21, 121) | 1;
    let center = (kernel / 2) as u8;
    let mut edges = Vec::new();
    for horizontal in [true, false] {
        let mask_image = GrayImage::from_pixel(
            if horizontal { kernel } else { 1 },
            if horizontal { 1 } else { kernel },
            Luma([255]),
        );
        let mask = Mask::from_image(
            &mask_image,
            if horizontal { center } else { 0 },
            if horizontal { 0 } else { center },
        );
        let lines = grayscale_open(&ink, &mask);
        let (outer, inner) = if horizontal {
            (height, width)
        } else {
            (width, height)
        };
        for a in 0..outer {
            let mut start = None;
            for b in 0..=inner {
                let dark = b < inner
                    && lines
                        .get_pixel(
                            if horizontal { b } else { a },
                            if horizontal { a } else { b },
                        )
                        .0[0]
                        > 127;
                if dark && start.is_none() {
                    start = Some(b);
                }
                if !dark
                    && let Some(s) = start.take()
                    && b - s >= kernel
                {
                    let (x0, y0, x1, y1) = if horizontal {
                        (s, a, b, a)
                    } else {
                        (a, s, a, b)
                    };
                    let edge = pdfni::EdgePart {
                        orientation: if horizontal {
                            pdfni::Orientation::Horizontal
                        } else {
                            pdfni::Orientation::Vertical
                        },
                        left: f64::from(x0) / scale,
                        right: f64::from(x1) / scale,
                        top: f64::from(y0) / scale,
                        bottom: f64::from(y1) / scale,
                    };
                    // Collapse adjacent pixels of one thick border.
                    if !edges.iter().any(|e: &pdfni::EdgePart| {
                        e.orientation == edge.orientation
                            && (e.left - edge.left).abs() < 1.5
                            && (e.right - edge.right).abs() < 1.5
                            && (e.top - edge.top).abs() < 1.5
                            && (e.bottom - edge.bottom).abs() < 1.5
                    }) {
                        edges.push(edge);
                    }
                    if edges.len() > MAX_EDGES {
                        return edges;
                    }
                }
            }
        }
    }
    edges
}

/// Keep border geometry for structure, but do not let OCR mistake grid lines
/// for letters or digits. Cell text is recognized from a separate cleaned image.
pub(super) fn without_grid(
    rgb: &[u8],
    width: u32,
    height: u32,
    scale: f64,
    edges: &[pdfni::EdgePart],
) -> Vec<u8> {
    let mut cleaned = rgb.to_vec();
    if edges.len() > MAX_EDGES
        || edges
            .iter()
            .filter(|e| e.orientation == pdfni::Orientation::Horizontal)
            .count()
            < 2
        || edges
            .iter()
            .filter(|e| e.orientation == pdfni::Orientation::Vertical)
            .count()
            < 2
    {
        return cleaned;
    }
    for edge in edges {
        let pad = scale.max(1.0);
        let x0 = (edge.left * scale - pad).max(0.0) as u32;
        let y0 = (edge.top * scale - pad).max(0.0) as u32;
        let x1 = (edge.right * scale + pad + 1.0).min(f64::from(width)) as u32;
        let y1 = (edge.bottom * scale + pad + 1.0).min(f64::from(height)) as u32;
        for y in y0..y1 {
            for x in x0..x1 {
                let index = (y as usize * width as usize + x as usize) * 3;
                cleaned[index..index + 3].fill(255);
            }
        }
    }
    cleaned
}

#[cfg(test)]
pub(super) mod tests {
    use pdf_extract::{
        Document, Object, Stream,
        content::{Content, Operation},
        dictionary,
    };

    pub(in crate::pdf) fn table_pdf(merged: bool, ruled: bool) -> Vec<u8> {
        let xs = [30., 130., 230., 330.];
        let ys = [50., 80., 110., 140., 170.];
        let mut ops = Vec::new();
        if ruled {
            for (i, &y) in ys.iter().enumerate() {
                let left = if merged && i == 3 { xs[1] } else { xs[0] };
                line(&mut ops, left, y, xs[3], y);
            }
            for (i, &x) in xs.iter().enumerate() {
                line(
                    &mut ops,
                    x,
                    if merged && (i == 1 || i == 2) {
                        ys[1]
                    } else {
                        ys[0]
                    },
                    x,
                    ys[4],
                );
            }
        }
        let values = if merged {
            vec![
                (0, 0, "Quarterly report"),
                (1, 0, "Item"),
                (1, 1, "Q1"),
                (1, 2, "Q2"),
                (2, 0, "Apples"),
                (2, 1, "12"),
                (2, 2, "13"),
                (3, 1, "22"),
                (3, 2, "23"),
            ]
        } else {
            vec![
                (0, 0, "Item"),
                (0, 1, "Q1"),
                (0, 2, "Q2"),
                (1, 0, "Apples"),
                (1, 1, "12"),
                (1, 2, "13"),
                (2, 0, "Berries"),
                (2, 2, "23"),
                (3, 0, "Total"),
                (3, 1, "12"),
                (3, 2, "36"),
            ]
        };
        // Deliberately write in reverse column order, not reading order.
        let mut values = values;
        values.sort_by_key(|(r, c, _)| (std::cmp::Reverse(*c), std::cmp::Reverse(*r)));
        for (r, c, value) in values {
            text(&mut ops, xs[c] + 8.0, ys[r] + 20.0, value);
        }
        text(&mut ops, 30., 220., "Narrative outside the table.");
        fixture(ops)
    }

    fn line(ops: &mut Vec<Operation>, x0: f64, y0: f64, x1: f64, y1: f64) {
        ops.extend([
            Operation::new("m", vec![x0.into(), (260.0 - y0).into()]),
            Operation::new("l", vec![x1.into(), (260.0 - y1).into()]),
            Operation::new("S", vec![]),
        ]);
    }
    fn text(ops: &mut Vec<Operation>, x: f64, y: f64, text: &str) {
        ops.extend([
            Operation::new("BT", vec![]),
            Operation::new("Tf", vec![Object::Name(b"F1".to_vec()), 14.into()]),
            Operation::new(
                "Tm",
                vec![
                    1.into(),
                    0.into(),
                    0.into(),
                    1.into(),
                    x.into(),
                    (260.0 - y).into(),
                ],
            ),
            Operation::new("Tj", vec![Object::string_literal(text)]),
            Operation::new("ET", vec![]),
        ]);
    }
    fn fixture(operations: Vec<Operation>) -> Vec<u8> {
        let mut doc = Document::with_version("1.5");
        let pages = doc.new_object_id();
        let font = doc.add_object(
            dictionary! { "Type" => "Font", "Subtype" => "Type1", "BaseFont" => "Helvetica" },
        );
        let stream = doc.add_object(Stream::new(
            dictionary! {},
            Content { operations }.encode().unwrap(),
        ));
        let page = doc.add_object(dictionary! { "Type" => "Page", "Parent" => pages, "MediaBox" => vec![0.into(),0.into(),360.into(),260.into()],
            "Resources" => dictionary! { "Font" => dictionary! { "F1" => font } }, "Contents" => stream });
        doc.objects.insert(pages, Object::Dictionary(dictionary! { "Type" => "Pages", "Kids" => vec![Object::Reference(page)], "Count" => 1 }));
        let catalog = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages });
        doc.trailer.set("Root", catalog);
        let mut bytes = Vec::new();
        doc.save_to(&mut bytes).unwrap();
        bytes
    }

    fn read(bytes: &[u8]) -> super::PageContent {
        super::super::layout::extract_page(&Document::load_mem(bytes).unwrap(), 1).unwrap()
    }

    #[test]
    fn native_grid_keeps_headers_empty_cells_and_numeric_columns() {
        let page = read(&table_pdf(false, true));
        assert_eq!(page.tables.len(), 1, "{}; {:?}", page.text, page.warning);
        let table = &page.tables[0];
        assert_eq!((table.rows, table.columns), (4, 3));
        let cell = |r, c| {
            table
                .cells
                .iter()
                .find(|cell| cell.row == r && cell.column == c)
                .unwrap()
                .text
                .as_str()
        };
        assert_eq!(cell(1, 2), "Q1");
        assert_eq!(cell(2, 2), "12");
        assert_eq!(cell(2, 3), "13");
        assert_eq!(cell(3, 2), "");
        assert_eq!(cell(3, 3), "23");
        assert!(
            page.body_text
                .as_ref()
                .unwrap()
                .contains("Narrative outside")
        );
        assert!(!page.body_text.as_ref().unwrap().contains("Berries"));
    }

    #[test]
    fn ruled_merged_headers_and_row_spans_survive() {
        let page = read(&table_pdf(true, true));
        assert_eq!(page.tables.len(), 1, "{:?}", page.warning);
        let table = &page.tables[0];
        let title = table
            .cells
            .iter()
            .find(|c| c.text == "Quarterly report")
            .unwrap();
        assert_eq!((title.row, title.column, title.column_span), (1, 1, 3));
        let apples = table.cells.iter().find(|c| c.text == "Apples").unwrap();
        assert_eq!(apples.row_span, 2);
        let html = table.context_html(1);
        assert!(html.contains("colspan=\"3\""));
        assert!(html.contains("rowspan=\"2\""));
    }

    #[test]
    fn borderless_numeric_table_is_labelled_inferred() {
        let page = read(&table_pdf(false, false));
        assert!(!page.tables.is_empty(), "{:?}", page.warning);
        assert!(
            page.tables[0]
                .warning
                .as_ref()
                .unwrap()
                .contains("inferred")
        );
        assert!(
            page.tables[0]
                .cells
                .iter()
                .any(|cell| cell.text == "23" && cell.column == 3)
        );
    }

    #[test]
    fn ordinary_prose_is_not_promoted_to_a_table() {
        let bytes = crate::pdf::encode_text("Prose", "This is a paragraph about a project.\nA second sentence explains the project.\nA third sentence adds useful context.\nThe last sentence closes the paragraph.").unwrap();
        assert!(read(&bytes).tables.is_empty());
    }

    #[test]
    fn table_markup_escapes_document_content() {
        let mut page = read(&table_pdf(false, true));
        page.tables[0].cells[0].text = "<script> & \"value\"\nnext".into();
        let html = page.tables[0].context_html(1);
        assert!(!html.contains("<script>"));
        assert!(html.contains("&lt;script&gt;"));
        assert!(html.contains("<br/>"));
    }

    #[test]
    fn multiline_cell_does_not_create_a_new_row() {
        let mut ops = Vec::new();
        for y in [40., 90., 140.] {
            line(&mut ops, 30., y, 330., y);
        }
        for x in [30., 180., 330.] {
            line(&mut ops, x, 40., x, 140.);
        }
        for (x, y, value) in [
            (38., 58., "First line"),
            (38., 78., "Second line"),
            (188., 58., "12.50"),
            (38., 110., "Other"),
            (188., 110., "-3,000"),
        ] {
            text(&mut ops, x, y, value);
        }
        let page = read(&fixture(ops));
        assert_eq!(page.tables.len(), 1, "{:?}", page.warning);
        let table = &page.tables[0];
        assert_eq!((table.rows, table.columns), (2, 2));
        assert!(table.cells[0].text.contains("First line"));
        assert!(table.cells[0].text.contains("Second line"));
        assert_eq!(table.cells[1].text, "12.50");
        assert_eq!(table.cells[3].text, "-3,000");
    }

    #[test]
    fn overlapping_cells_are_rejected_instead_of_assigning_a_value_twice() {
        let page = read(&table_pdf(false, true));
        let b = page.tables[0].bounds;
        let raw = pdfni::Table {
            extraction_method: "lattice",
            bbox: b,
            n_rows: 2,
            n_cols: 2,
            data: vec![
                vec![
                    pdfni::Cell {
                        text: "12".into(),
                        bbox: b
                    };
                    2
                ];
                2
            ],
        };
        let part = pdfni::PagePart {
            width: 360.,
            height: 260.,
            glyphs: Vec::new(),
            edges: Vec::new(),
            graphics: Vec::new(),
            norm_rotate: 0,
        };
        assert!(super::validate(&raw, &part).is_none());
    }
}
