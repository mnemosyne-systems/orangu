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

//! `--report FILE.pdf`: the run as one document — what was measured, what
//! produced it, and the pictures.
//!
//! A benchmark result travels. It goes into a pull request, an issue, a mail
//! to someone with different hardware — and the parts that make it *readable*
//! are exactly the parts a terminal cannot carry: the throughput chart and the
//! flamegraph. Up to now those were separate files, and a number quoted
//! without them is a number nobody can check.
//!
//! So the report is a PDF, not a markdown table: **a PDF can fold the PNGs
//! in**. One file holds the provenance (which build, which model, which
//! device, at which clocks), the measurements, and the two images, and it is
//! the same file months later.
//!
//! It is deliberately a small, self-contained layout engine rather than a
//! reuse of `orangu`'s own exporter: that one lives inside the client binary
//! and is built around transcripts, reviews and pull requests. What is shared
//! is what should be — the brand font, the page geometry, the header/footer
//! bands and the `subset_fonts: false` workaround — so the two look like one
//! product.

use std::path::{Path, PathBuf};

use printpdf::{
    BuiltinFont, Color, Line, LinePoint, Mm, Op, PaintMode, ParsedFont, PdfDocument, PdfFontHandle,
    PdfPage, PdfSaveOptions, Point, Pt, RawImage, Rect, Rgb, TextItem, XObjectTransform,
};

/// Red Hat Text (SIL OFL — see `assets/fonts/LICENSE`), the same faces
/// `orangu`'s own PDF export embeds.
const FONT_REGULAR: &[u8] = include_bytes!("../../../assets/fonts/RedHatText-Regular.ttf");
const FONT_BOLD: &[u8] = include_bytes!("../../../assets/fonts/RedHatText-Bold.ttf");

const PAGE_WIDTH_MM: f32 = 210.0;
const PAGE_HEIGHT_MM: f32 = 297.0;
const MARGIN_MM: f32 = 18.0;
const USABLE_WIDTH_MM: f32 = PAGE_WIDTH_MM - 2.0 * MARGIN_MM;
const PT_TO_MM: f32 = 25.4 / 72.0;

const HEADER_BAND_MM: f32 = 11.0;
const FOOTER_BAND_MM: f32 = 11.0;
const CONTENT_GAP_MM: f32 = 5.0;
const CONTENT_TOP_MM: f32 = PAGE_HEIGHT_MM - HEADER_BAND_MM - CONTENT_GAP_MM;
const CONTENT_BOTTOM_MM: f32 = FOOTER_BAND_MM + CONTENT_GAP_MM;

const TITLE_SIZE: f32 = 15.0;
const HEADING_SIZE: f32 = 11.5;
const BODY_SIZE: f32 = 9.5;
const BAND_TEXT_SIZE: f32 = 11.0;
const LINE_MM: f32 = 5.0;
const ROW_MM: f32 = 5.2;

/// The orangu brand brown, as `orangu`'s exporter uses it.
const BRAND: (f32, f32, f32) = (139.0 / 255.0, 90.0 / 255.0, 43.0 / 255.0);
const TEXT: (f32, f32, f32) = (0.0, 0.0, 0.0);
const DIM: (f32, f32, f32) = (0.42, 0.42, 0.45);
const RULE: (f32, f32, f32) = (0.72, 0.72, 0.75);
const WHITE: (f32, f32, f32) = (1.0, 1.0, 1.0);

/// How a column's cells sit under its heading. Numbers right, words left —
/// a rate column that is not right-aligned cannot be scanned down.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Align {
    Left,
    Right,
}

pub struct Column {
    pub title: String,
    pub align: Align,
}

impl Column {
    pub fn left(title: &str) -> Column {
        Column {
            title: title.to_string(),
            align: Align::Left,
        }
    }

    pub fn right(title: &str) -> Column {
        Column {
            title: title.to_string(),
            align: Align::Right,
        }
    }
}

/// One piece of a report, in the order it is laid out.
pub enum Block {
    Heading(String),
    /// Label/value pairs — the provenance blocks. Laid out as two columns, not
    /// as prose, because they are read by scanning for one of them.
    Fields(Vec<(String, String)>),
    Table {
        columns: Vec<Column>,
        rows: Vec<Vec<String>>,
    },
    /// A PNG, fitted to the page width. Missing or undecodable files are
    /// **skipped with a note in their place** rather than failing the report:
    /// the measurements are the deliverable, and a run that produced no
    /// rasterizer output should still hand back a document.
    Image {
        caption: String,
        path: PathBuf,
    },
    Note(String),
}

/// Write `blocks` to `path` as a PDF.
pub fn write(
    path: &Path,
    title: &str,
    subtitle: &str,
    footer: &str,
    blocks: &[Block],
) -> anyhow::Result<()> {
    let mut doc = PdfDocument::new(title);
    let fonts = Fonts::load(&mut doc);
    let mut pdf = Pdf {
        doc,
        fonts,
        header: title.to_string(),
        footer: footer.to_string(),
        ops: Vec::new(),
        pages: Vec::new(),
        y: CONTENT_TOP_MM,
    };
    pdf.furniture();
    pdf.title(title, subtitle);

    for block in blocks {
        pdf.block(block);
    }
    pdf.save(path)
}

struct Fonts {
    regular: PdfFontHandle,
    bold: PdfFontHandle,
    /// Real glyph advances, for right-aligning a column and centering the
    /// band text. `None` when the embedded faces could not be parsed, in
    /// which case every glyph is assumed half an em — the same fallback
    /// `orangu`'s exporter makes, and for the same reason: a report with
    /// slightly loose columns beats no report.
    faces: Option<Box<[ttf_parser::Face<'static>; 2]>>,
}

impl Fonts {
    fn load(doc: &mut PdfDocument) -> Fonts {
        let external = |doc: &mut PdfDocument, bytes: &'static [u8]| {
            ParsedFont::from_bytes(bytes, 0, &mut Vec::new())
                .map(|parsed| PdfFontHandle::External(doc.add_font(&parsed)))
        };
        match (external(doc, FONT_REGULAR), external(doc, FONT_BOLD)) {
            (Some(regular), Some(bold)) => Fonts {
                regular,
                bold,
                faces: ttf_parser::Face::parse(FONT_REGULAR, 0)
                    .ok()
                    .zip(ttf_parser::Face::parse(FONT_BOLD, 0).ok())
                    .map(|(r, b)| Box::new([r, b])),
            },
            _ => Fonts {
                regular: PdfFontHandle::Builtin(BuiltinFont::Helvetica),
                bold: PdfFontHandle::Builtin(BuiltinFont::HelveticaBold),
                faces: None,
            },
        }
    }

    fn handle(&self, bold: bool) -> &PdfFontHandle {
        if bold { &self.bold } else { &self.regular }
    }

    fn width_mm(&self, text: &str, bold: bool, size: f32) -> f32 {
        let text = self.shape(text);
        let Some(faces) = &self.faces else {
            return text.chars().count() as f32 * size * 0.5 * PT_TO_MM;
        };
        let face = &faces[usize::from(bold)];
        let per_em = f32::from(face.units_per_em());
        text.chars()
            .map(|ch| {
                let advance = face
                    .glyph_index(ch)
                    .and_then(|glyph| face.glyph_hor_advance(glyph))
                    .unwrap_or_else(|| face.units_per_em() / 2);
                f32::from(advance) / per_em * size * PT_TO_MM
            })
            .sum()
    }

    /// Replace characters the embedded faces have no glyph for.
    ///
    /// A missing glyph is not a missing character — it draws as a filled box,
    /// so "cmp-a → cmp-b" became "cmp-a ▮ cmp-b" in a document whose whole job
    /// is to be read by someone else. Red Hat Text covers `±`, `·` and `—`
    /// (which is why the tables look right) but not the arrows, and the next
    /// caller cannot be expected to know which is which. So the substitution
    /// happens here, once, and only for what is genuinely absent: a face that
    /// *has* the glyph keeps it.
    fn shape(&self, text: &str) -> String {
        let Some(faces) = &self.faces else {
            return text.to_string();
        };
        if text.is_ascii() {
            return text.to_string();
        }
        let face = &faces[0];
        text.chars()
            .map(|ch| {
                if ch.is_ascii() || face.glyph_index(ch).is_some() {
                    return ch.to_string();
                }
                match ch {
                    '→' => "->".to_string(),
                    '←' => "<-".to_string(),
                    '—' | '–' => "-".to_string(),
                    '±' => "+/-".to_string(),
                    '·' => "-".to_string(),
                    '×' => "x".to_string(),
                    '≈' => "~".to_string(),
                    // Anything else unknown: dropped rather than drawn as a
                    // box. A box in a report reads as corruption.
                    _ => String::new(),
                }
            })
            .collect()
    }
}

struct Pdf {
    doc: PdfDocument,
    fonts: Fonts,
    header: String,
    footer: String,
    ops: Vec<Op>,
    pages: Vec<PdfPage>,
    /// The baseline the next line is drawn on, descending down the page.
    y: f32,
}

impl Pdf {
    fn block(&mut self, block: &Block) {
        match block {
            Block::Heading(text) => self.heading(text),
            Block::Fields(fields) => self.fields(fields),
            Block::Table { columns, rows } => self.table(columns, rows),
            Block::Image { caption, path } => self.image(caption, path),
            Block::Note(text) => self.note(text),
        }
    }

    fn heading(&mut self, text: &str) {
        self.space(4.0);
        self.ensure(LINE_MM * 2.0);
        self.y -= HEADING_SIZE * PT_TO_MM;
        self.text(text, true, MARGIN_MM, self.y, HEADING_SIZE, TEXT);
        self.y -= 1.6;
        self.rule(MARGIN_MM, self.y, PAGE_WIDTH_MM - MARGIN_MM, self.y, RULE);
        self.y -= 3.0;
    }

    fn fields(&mut self, fields: &[(String, String)]) {
        // One column width for every label, so the values line up in a single
        // straight edge — the whole reason this is not prose.
        let label_width = fields
            .iter()
            .map(|(k, _)| self.fonts.width_mm(k, false, BODY_SIZE))
            .fold(0.0_f32, f32::max);
        let value_x = MARGIN_MM + label_width + 6.0;

        // Kept together where it can be. A field group is read as a unit —
        // "which build, which device, which clocks" — and two of its rows
        // stranded at the top of the next page read as belonging to whatever
        // heading is up there. Only when the group would not fit a whole page
        // either is it allowed to split.
        let height: f32 = fields
            .iter()
            .map(|(_, value)| {
                self.wrap(value, PAGE_WIDTH_MM - MARGIN_MM - value_x, BODY_SIZE)
                    .len() as f32
                    * LINE_MM
            })
            .sum();
        if self.y - height < CONTENT_BOTTOM_MM && height <= CONTENT_TOP_MM - CONTENT_BOTTOM_MM {
            self.new_page();
        }

        for (key, value) in fields {
            self.ensure(LINE_MM);
            self.y -= LINE_MM;
            self.text(key, false, MARGIN_MM, self.y, BODY_SIZE, DIM);
            // A long value (a path, a kernel summary) wraps under itself
            // rather than running into the margin.
            for (i, line) in self
                .wrap(value, PAGE_WIDTH_MM - MARGIN_MM - value_x, BODY_SIZE)
                .into_iter()
                .enumerate()
            {
                if i > 0 {
                    self.ensure(LINE_MM);
                    self.y -= LINE_MM;
                }
                self.text(&line, false, value_x, self.y, BODY_SIZE, TEXT);
            }
        }
    }

    fn table(&mut self, columns: &[Column], rows: &[Vec<String>]) {
        if columns.is_empty() {
            return;
        }
        // Column widths from the widest cell, headings included, then the
        // slack spread evenly. Measured rather than guessed, so a long label
        // never overlaps the column beside it.
        let mut widths: Vec<f32> = columns
            .iter()
            .map(|c| self.fonts.width_mm(&c.title, true, BODY_SIZE))
            .collect();
        for row in rows {
            for (i, cell) in row.iter().enumerate().take(widths.len()) {
                widths[i] = widths[i].max(self.fonts.width_mm(cell, false, BODY_SIZE));
            }
        }
        let gap = 6.0;
        let total: f32 = widths.iter().sum::<f32>() + gap * (widths.len() - 1) as f32;
        if total < USABLE_WIDTH_MM {
            let extra = (USABLE_WIDTH_MM - total) / widths.len() as f32;
            for w in &mut widths {
                *w += extra;
            }
        }

        self.space(1.5);
        self.header_row(columns, &widths, gap);
        for row in rows {
            self.ensure(ROW_MM);
            // A table split across a page break repeats its heading; a run of
            // bare numbers under no columns is unreadable.
            if self.y - ROW_MM < CONTENT_BOTTOM_MM {
                self.new_page();
                self.header_row(columns, &widths, gap);
            }
            self.y -= ROW_MM;
            let mut x = MARGIN_MM;
            for (i, width) in widths.iter().enumerate() {
                let cell = row.get(i).map(String::as_str).unwrap_or("");
                let at = match columns[i].align {
                    Align::Left => x,
                    Align::Right => x + width - self.fonts.width_mm(cell, false, BODY_SIZE),
                };
                self.text(cell, false, at, self.y, BODY_SIZE, TEXT);
                x += width + gap;
            }
        }
        self.space(1.0);
    }

    fn header_row(&mut self, columns: &[Column], widths: &[f32], gap: f32) {
        self.ensure(ROW_MM * 2.0);
        self.y -= ROW_MM;
        let mut x = MARGIN_MM;
        for (i, width) in widths.iter().enumerate() {
            let title = columns[i].title.clone();
            let at = match columns[i].align {
                Align::Left => x,
                Align::Right => x + width - self.fonts.width_mm(&title, true, BODY_SIZE),
            };
            self.text(&title, true, at, self.y, BODY_SIZE, DIM);
            x += width + gap;
        }
        self.y -= 1.4;
        self.rule(MARGIN_MM, self.y, PAGE_WIDTH_MM - MARGIN_MM, self.y, RULE);
        self.y -= 1.0;
    }

    fn note(&mut self, text: &str) {
        self.space(1.0);
        for line in self.wrap(text, USABLE_WIDTH_MM, BODY_SIZE) {
            self.ensure(LINE_MM);
            self.y -= LINE_MM;
            self.text(&line, false, MARGIN_MM, self.y, BODY_SIZE, DIM);
        }
    }

    /// Place a PNG, fitted to the page width and never taller than one page.
    ///
    /// A flamegraph is 1200×N and a chart 900×N; both are wider than they are
    /// tall at the sizes this produces, so fitting the width is what keeps the
    /// frame labels legible. An image that would not fit the space left starts
    /// a page of its own rather than being shrunk to whatever is left.
    fn image(&mut self, caption: &str, path: &Path) {
        let bytes = match std::fs::read(path) {
            Ok(bytes) => bytes,
            Err(e) => {
                self.note(&format!("{caption}: not embedded ({e})"));
                return;
            }
        };
        let image = match RawImage::decode_from_bytes(&bytes, &mut Vec::new()) {
            Ok(image) => image,
            Err(e) => {
                self.note(&format!("{caption}: not embedded ({e})"));
                return;
            }
        };
        let (px_w, px_h) = (image.width as f32, image.height as f32);
        if px_w <= 0.0 || px_h <= 0.0 {
            return;
        }

        let mut width_mm = USABLE_WIDTH_MM;
        let mut height_mm = width_mm * px_h / px_w;
        let caption_mm = LINE_MM + 1.0;
        let full_page = CONTENT_TOP_MM - CONTENT_BOTTOM_MM - caption_mm;
        if height_mm > full_page {
            // Taller than a page even at full width: scale to the page.
            height_mm = full_page;
            width_mm = height_mm * px_w / px_h;
        }

        self.space(2.0);
        if self.y - (height_mm + caption_mm) < CONTENT_BOTTOM_MM {
            self.new_page();
        }
        self.y -= LINE_MM;
        self.text(caption, true, MARGIN_MM, self.y, BODY_SIZE, DIM);
        self.y -= 1.0 + height_mm;

        // `UseXobject` sizes an image from its pixel count and a DPI, so the
        // DPI is how a target width in millimetres is expressed.
        let width_pt = width_mm / PT_TO_MM;
        let dpi = px_w * 72.0 / width_pt;
        let id = self.doc.add_image(&image);
        self.ops.push(Op::UseXobject {
            id,
            transform: XObjectTransform {
                translate_x: Some(Mm(MARGIN_MM).into()),
                translate_y: Some(Mm(self.y).into()),
                dpi: Some(dpi),
                ..Default::default()
            },
        });
        self.y -= 2.0;
    }

    fn title(&mut self, title: &str, subtitle: &str) {
        self.y -= TITLE_SIZE * PT_TO_MM + 2.0;
        self.text(title, true, MARGIN_MM, self.y, TITLE_SIZE, TEXT);
        self.y -= LINE_MM;
        self.text(subtitle, false, MARGIN_MM, self.y, BODY_SIZE, DIM);
        self.y -= 2.0;
        self.rule(MARGIN_MM, self.y, PAGE_WIDTH_MM - MARGIN_MM, self.y, BRAND);
        self.y -= 2.0;
    }

    /// Break `text` to `width`, on spaces only — a URL, a path or a commit is
    /// one word, and half of any of them is useless.
    fn wrap(&self, text: &str, width: f32, size: f32) -> Vec<String> {
        let mut lines = Vec::new();
        let mut current = String::new();
        for word in text.split_whitespace() {
            let candidate = if current.is_empty() {
                word.to_string()
            } else {
                format!("{current} {word}")
            };
            if !current.is_empty() && self.fonts.width_mm(&candidate, false, size) > width {
                lines.push(std::mem::take(&mut current));
                current = word.to_string();
            } else {
                current = candidate;
            }
        }
        if !current.is_empty() {
            lines.push(current);
        }
        if lines.is_empty() {
            lines.push(String::new());
        }
        lines
    }

    fn space(&mut self, mm: f32) {
        self.y -= mm;
    }

    /// Start a new page when `needed` millimetres do not remain.
    fn ensure(&mut self, needed: f32) {
        if self.y - needed < CONTENT_BOTTOM_MM {
            self.new_page();
        }
    }

    fn new_page(&mut self) {
        let ops = std::mem::take(&mut self.ops);
        self.pages
            .push(PdfPage::new(Mm(PAGE_WIDTH_MM), Mm(PAGE_HEIGHT_MM), ops));
        self.y = CONTENT_TOP_MM;
        self.furniture();
    }

    /// The brand bands, top and bottom, on every page.
    fn furniture(&mut self) {
        self.fill(
            0.0,
            PAGE_HEIGHT_MM - HEADER_BAND_MM,
            PAGE_WIDTH_MM,
            PAGE_HEIGHT_MM,
            BRAND,
        );
        self.fill(0.0, 0.0, PAGE_WIDTH_MM, FOOTER_BAND_MM, BRAND);

        let cap = BAND_TEXT_SIZE * 0.7 * PT_TO_MM;
        let header_baseline = (PAGE_HEIGHT_MM - HEADER_BAND_MM / 2.0) - cap / 2.0;
        let footer_baseline = FOOTER_BAND_MM / 2.0 - cap / 2.0;
        let header = self.header.clone();
        let footer = self.footer.clone();
        let header_x = ((PAGE_WIDTH_MM - self.fonts.width_mm(&header, true, BAND_TEXT_SIZE)) / 2.0)
            .max(MARGIN_MM);
        let footer_x = ((PAGE_WIDTH_MM - self.fonts.width_mm(&footer, false, BAND_TEXT_SIZE))
            / 2.0)
            .max(MARGIN_MM);
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
    }

    fn text(&mut self, text: &str, bold: bool, x: f32, y: f32, size: f32, color: (f32, f32, f32)) {
        // Shaped here, where every drawn string passes through — and measured
        // the same way in `width_mm`, so a substitution can never move a
        // right-aligned column off its edge.
        let text = &self.fonts.shape(text);
        if text.is_empty() {
            return;
        }
        let (r, g, b) = color;
        self.ops.push(Op::StartTextSection);
        self.ops.push(Op::SetFillColor {
            col: Color::Rgb(Rgb::new(r, g, b, None)),
        });
        self.ops.push(Op::SetFont {
            font: self.fonts.handle(bold).clone(),
            size: Pt(size),
        });
        self.ops.push(Op::SetTextCursor {
            pos: Point::new(Mm(x), Mm(y)),
        });
        self.ops.push(Op::ShowText {
            items: vec![TextItem::Text(text.to_string())],
        });
        self.ops.push(Op::EndTextSection);
    }

    fn fill(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, color: (f32, f32, f32)) {
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

    fn rule(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, color: (f32, f32, f32)) {
        let (r, g, b) = color;
        self.ops.push(Op::SetOutlineColor {
            col: Color::Rgb(Rgb::new(r, g, b, None)),
        });
        self.ops.push(Op::SetOutlineThickness { pt: Pt(0.4) });
        self.ops.push(Op::DrawLine {
            line: Line {
                points: vec![
                    LinePoint {
                        p: Point::new(Mm(x0), Mm(y0)),
                        bezier: false,
                    },
                    LinePoint {
                        p: Point::new(Mm(x1), Mm(y1)),
                        bezier: false,
                    },
                ],
                is_closed: false,
            },
        });
    }

    fn save(mut self, path: &Path) -> anyhow::Result<()> {
        let ops = std::mem::take(&mut self.ops);
        self.pages
            .push(PdfPage::new(Mm(PAGE_WIDTH_MM), Mm(PAGE_HEIGHT_MM), ops));
        let pages = std::mem::take(&mut self.pages);
        self.doc.with_pages(pages);
        // Not subset: printpdf's glyph renumbering scrambles these faces'
        // outlines — the layout and the copy/paste text stay right while the
        // glyphs on screen go wrong. `orangu`'s own exporter carries the same
        // workaround and the same comment.
        let opts = PdfSaveOptions {
            subset_fonts: false,
            ..PdfSaveOptions::default()
        };
        let bytes = self.doc.save(&opts, &mut Vec::new());
        std::fs::write(path, bytes)
            .map_err(|e| anyhow::anyhow!("could not write {}: {e}", path.display()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("orangu-bench-report-{name}"));
        let _ = std::fs::create_dir_all(&dir);
        dir.join("report.pdf")
    }

    /// The report has to *say the numbers it was given*. Rendering a PDF that
    /// opens but carries different figures than the run measured would be the
    /// one failure nobody checks for, so the text is read back out.
    #[test]
    fn the_report_carries_the_measurements_and_the_provenance() {
        let path = temp("content");
        write(
            &path,
            "orangu-bench report",
            "decode · 2026-08-04",
            "orangu-bench 1.2.0",
            &[
                Block::Heading("What produced it".to_string()),
                Block::Fields(vec![
                    ("build".to_string(), "1.2.0 (52c04435f)".to_string()),
                    ("model".to_string(), "gemma-4-E2B-it:Q4_K_M".to_string()),
                ]),
                Block::Heading("What it measured".to_string()),
                Block::Table {
                    columns: vec![
                        Column::left("measurement"),
                        Column::right("n"),
                        Column::right("best"),
                    ],
                    rows: vec![vec![
                        "decode".to_string(),
                        "512".to_string(),
                        "41.37".to_string(),
                    ]],
                },
            ],
        )
        .expect("writes");

        let bytes = std::fs::read(&path).expect("read back");
        assert!(bytes.starts_with(b"%PDF"), "not a PDF");
        let text = pdf_extract::extract_text_from_mem(&bytes).expect("extract");
        for expected in [
            "orangu-bench report",
            "1.2.0 (52c04435f)",
            "gemma-4-E2B-it:Q4_K_M",
            "What it measured",
            "decode",
            "512",
            "41.37",
        ] {
            assert!(text.contains(expected), "missing {expected:?} in:\n{text}");
        }
        let _ = std::fs::remove_dir_all(path.parent().expect("dir"));
    }

    /// A missing or unreadable image must not cost the reader the report. The
    /// rasterizer is optional (`rsvg-convert`), so this is the ordinary case
    /// on a machine that does not have it — not an edge case.
    #[test]
    fn a_missing_image_leaves_a_note_not_an_error() {
        let path = temp("missing-image");
        write(
            &path,
            "orangu-bench report",
            "decode",
            "orangu-bench 1.2.0",
            &[
                Block::Image {
                    caption: "Throughput".to_string(),
                    path: PathBuf::from("/nonexistent/chart.png"),
                },
                Block::Table {
                    columns: vec![Column::left("measurement"), Column::right("best")],
                    rows: vec![vec!["decode".to_string(), "41.37".to_string()]],
                },
            ],
        )
        .expect("writes anyway");
        let bytes = std::fs::read(&path).expect("read back");
        let text = pdf_extract::extract_text_from_mem(&bytes).expect("extract");
        assert!(text.contains("not embedded"), "{text}");
        // And the measurements after it still made it in.
        assert!(text.contains("41.37"), "{text}");
        let _ = std::fs::remove_dir_all(path.parent().expect("dir"));
    }

    /// A character the font has no glyph for must not reach the page: it
    /// draws as a filled box, which in a document sent to someone else reads
    /// as corruption rather than as one missing arrow.
    #[test]
    fn a_glyph_the_font_lacks_is_replaced_not_drawn_as_a_box() {
        let path = temp("glyphs");
        write(
            &path,
            "orangu-bench comparison",
            "cmp-a → cmp-b",
            "orangu-bench 1.2.0",
            &[Block::Fields(vec![
                ("arrow".to_string(), "old → new".to_string()),
                // Present in Red Hat Text, so these must survive untouched —
                // the substitution is for what is absent, not for everything
                // non-ASCII.
                ("kept".to_string(), "41.37 ± 0.26 · best".to_string()),
            ])],
        )
        .expect("writes");
        let bytes = std::fs::read(&path).expect("read back");
        let text = pdf_extract::extract_text_from_mem(&bytes).expect("extract");
        assert!(text.contains("old -> new"), "arrow not replaced:\n{text}");
        assert!(!text.contains('→'), "the arrow reached the page:\n{text}");
        assert!(text.contains('±'), "± was replaced needlessly:\n{text}");
        assert!(text.contains('·'), "· was replaced needlessly:\n{text}");
        let _ = std::fs::remove_dir_all(path.parent().expect("dir"));
    }

    /// Long values wrap instead of running off the page — and wrap on spaces
    /// only, so a path or a URL survives intact.
    #[test]
    fn a_long_value_wraps_without_breaking_a_path() {
        let path = temp("wrap");
        let long_path = "/home/someone/.orangu/orangu-bench/runs/1785813435883/flamegraph.svg";
        write(
            &path,
            "orangu-bench report",
            "decode",
            "orangu-bench 1.2.0",
            &[Block::Fields(vec![
                ("artifact".to_string(), long_path.to_string()),
                (
                    "note".to_string(),
                    "a sentence long enough that it has to be broken across more than one line \
                     of the value column, which is what this is checking"
                        .to_string(),
                ),
            ])],
        )
        .expect("writes");
        let bytes = std::fs::read(&path).expect("read back");
        let text = pdf_extract::extract_text_from_mem(&bytes).expect("extract");
        assert!(text.contains(long_path), "the path was broken up:\n{text}");
        let _ = std::fs::remove_dir_all(path.parent().expect("dir"));
    }
}
