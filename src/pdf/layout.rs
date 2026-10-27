// Copyright (C) 2026 The orangu community
// SPDX-License-Identifier: GPL-3.0-or-later

//! Adapt pdf-extract's positioned output to pdfni; parsing remains in the library.

use super::tables::{self, MAX_EDGES, MAX_GLYPHS, PageContent};
use pdf_extract::{OutputDev, OutputError, Transform};
use pdfni::{BBox, EdgePart, GlyphPart, Orientation, PagePart};
use std::{cell::RefCell, fmt, ops::Range, rc::Rc};

#[derive(Clone)]
struct TextBuffer(Rc<RefCell<String>>);
impl fmt::Write for TextBuffer {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        self.0.borrow_mut().push_str(text);
        Ok(())
    }
}
impl pdf_extract::ConvertToFmt for TextBuffer {
    type Writer = Self;
    fn convert(self) -> Self {
        self
    }
}

struct LayoutOutput {
    plain: pdf_extract::PlainTextOutput<TextBuffer>,
    buffer: TextBuffer,
    part: PagePart,
    ranges: Vec<(Range<usize>, BBox)>,
    left: f64,
    top: f64,
    overflow: bool,
}

pub(super) fn extract_page(
    document: &pdf_extract::Document,
    number: u32,
) -> anyhow::Result<PageContent> {
    let buffer = TextBuffer(Rc::new(RefCell::new(String::new())));
    let mut output = LayoutOutput {
        plain: pdf_extract::PlainTextOutput::new(buffer.clone()),
        buffer,
        part: PagePart {
            width: 0.0,
            height: 0.0,
            glyphs: Vec::new(),
            edges: Vec::new(),
            norm_rotate: 0,
            graphics: Vec::new(),
        },
        ranges: Vec::new(),
        left: 0.0,
        top: 0.0,
        overflow: false,
    };
    pdf_extract::output_doc_page(document, &mut output, number)?;
    let text = output.buffer.0.borrow().clone();
    Ok(tables::reconstruct(
        text,
        output.part,
        output.ranges,
        output.overflow,
    ))
}

impl LayoutOutput {
    fn point(&self, ctm: &Transform, x: f64, y: f64) -> (f64, f64) {
        (
            ctm.m11 * x + ctm.m21 * y + ctm.m31 - self.left,
            self.top - (ctm.m12 * x + ctm.m22 * y + ctm.m32),
        )
    }

    fn edge(&mut self, from: (f64, f64), to: (f64, f64)) {
        if self.part.edges.len() >= MAX_EDGES {
            self.overflow = true;
            return;
        }
        let orientation = if (from.1 - to.1).abs() < 0.5 {
            Orientation::Horizontal
        } else if (from.0 - to.0).abs() < 0.5 {
            Orientation::Vertical
        } else {
            return;
        };
        if (from.0 - to.0).abs().max((from.1 - to.1).abs()) < 3.0 {
            return;
        }
        self.part.edges.push(EdgePart {
            orientation,
            left: from.0.min(to.0),
            right: from.0.max(to.0),
            top: from.1.min(to.1),
            bottom: from.1.max(to.1),
        });
    }

    fn path(&mut self, ctm: &Transform, path: &pdf_extract::Path, filled: bool) {
        use pdf_extract::PathOp;
        let mut current = None;
        let mut start = None;
        for operation in &path.ops {
            match *operation {
                PathOp::MoveTo(x, y) => {
                    current = Some(self.point(ctm, x, y));
                    start = current;
                }
                PathOp::LineTo(x, y) => {
                    let next = self.point(ctm, x, y);
                    if !filled && let Some(from) = current {
                        self.edge(from, next);
                    }
                    current = Some(next);
                }
                PathOp::Close => {
                    if !filled && let (Some(from), Some(to)) = (current, start) {
                        self.edge(from, to);
                    }
                    current = start;
                }
                PathOp::Rect(x, y, width, height) => {
                    let points = [
                        self.point(ctm, x, y),
                        self.point(ctm, x + width, y),
                        self.point(ctm, x + width, y + height),
                        self.point(ctm, x, y + height),
                    ];
                    if !filled {
                        for i in 0..4 {
                            self.edge(points[i], points[(i + 1) % 4]);
                        }
                    } else if (points[0].0 - points[1].0).abs() <= 1.5 {
                        self.edge(
                            ((points[0].0 + points[1].0) / 2.0, points[0].1),
                            ((points[2].0 + points[3].0) / 2.0, points[2].1),
                        );
                    } else if (points[0].1 - points[3].1).abs() <= 1.5 {
                        self.edge(
                            (points[0].0, (points[0].1 + points[3].1) / 2.0),
                            (points[1].0, (points[1].1 + points[2].1) / 2.0),
                        );
                    }
                }
                PathOp::CurveTo(_, _, _, _, _, _) => {
                    current = None;
                }
            }
        }
    }
}

impl OutputDev for LayoutOutput {
    fn begin_page(
        &mut self,
        number: u32,
        media: &pdf_extract::MediaBox,
        art: Option<(f64, f64, f64, f64)>,
    ) -> Result<(), OutputError> {
        self.left = media.llx;
        self.top = media.ury;
        self.part.width = media.urx - media.llx;
        self.part.height = media.ury - media.lly;
        self.plain.begin_page(number, media, art)
    }
    fn end_page(&mut self) -> Result<(), OutputError> {
        self.plain.end_page()
    }
    fn begin_word(&mut self) -> Result<(), OutputError> {
        self.plain.begin_word()
    }
    fn end_word(&mut self) -> Result<(), OutputError> {
        self.plain.end_word()
    }
    fn end_line(&mut self) -> Result<(), OutputError> {
        self.plain.end_line()
    }
    fn output_character(
        &mut self,
        trm: &Transform,
        width: f64,
        spacing: f64,
        font_size: f64,
        text: &str,
    ) -> Result<(), OutputError> {
        let before = self.buffer.0.borrow().len();
        self.plain
            .output_character(trm, width, spacing, font_size, text)?;
        if self.part.glyphs.len() + text.chars().count() > MAX_GLYPHS {
            self.overflow = true;
            return Ok(());
        }
        let height = trm.m21.hypot(trm.m22) * font_size.abs();
        let advance = trm.m11.hypot(trm.m12) * width * font_size.abs();
        let x = trm.m31 - self.left;
        let baseline = self.top - trm.m32;
        let bounds = BBox {
            x0: x,
            x1: x + advance,
            top: baseline - height * 0.8,
            bottom: baseline + height * 0.2,
        };
        let count = text.chars().count().max(1) as f64;
        for (i, ch) in text.chars().enumerate() {
            self.part.glyphs.push(GlyphPart {
                ch,
                left: x + i as f64 * advance / count,
                right: x + (i + 1) as f64 * advance / count,
                top: bounds.top,
                bottom: bounds.bottom,
                font_size: Some(height),
                upright: trm.m12.abs() < 0.001
                    && trm.m21.abs() < 0.001
                    && trm.m11 > 0.0
                    && trm.m22 > 0.0,
                rot: 0,
            });
        }
        self.ranges
            .push((before..self.buffer.0.borrow().len(), bounds));
        Ok(())
    }
    fn stroke(
        &mut self,
        ctm: &Transform,
        _: &pdf_extract::ColorSpace,
        _: &[f64],
        path: &pdf_extract::Path,
    ) -> Result<(), OutputError> {
        self.path(ctm, path, false);
        Ok(())
    }
    fn fill(
        &mut self,
        ctm: &Transform,
        _: &pdf_extract::ColorSpace,
        _: &[f64],
        path: &pdf_extract::Path,
    ) -> Result<(), OutputError> {
        self.path(ctm, path, true);
        Ok(())
    }
}
