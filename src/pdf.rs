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

//! PDF text input and output shared by the terminal, file tools and web uploads.
//! Reading extracts embedded text; scanned pages require OCR outside orangu.

use anyhow::{Context, Result, anyhow, bail};
use printpdf::{
    Mm, Op, ParsedFont, PdfDocument, PdfFontHandle, PdfPage, PdfSaveOptions, Point, Pt, TextItem,
};
use std::path::Path;

pub fn is_pdf_path(path: &Path) -> bool {
    path.extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("pdf"))
}

pub fn is_pdf(path: &Path, bytes: &[u8]) -> bool {
    is_pdf_path(path) || bytes.starts_with(b"%PDF-")
}

/// Extract text without letting a malformed document's parser panic escape.
pub fn extract_text(bytes: &[u8]) -> Result<String> {
    let text = std::panic::catch_unwind(|| pdf_extract::extract_text_from_mem(bytes))
        .map_err(|_| anyhow!("PDF parser failed while reading this document"))?
        .context("Cannot extract PDF text; the document may be damaged or password-protected")?;
    let text = text.trim();
    if text.is_empty() {
        bail!(
            "PDF contains no extractable text; scanned documents require OCR before they can be queried"
        );
    }
    Ok(text.to_owned())
}

pub fn read_text(path: &Path) -> Result<String> {
    let bytes =
        std::fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    decode_text(path, &bytes)
}

pub fn decode_text(path: &Path, bytes: &[u8]) -> Result<String> {
    if is_pdf(path, bytes) {
        extract_text(bytes).map_err(|err| anyhow!("{}: {err:#}", path.display()))
    } else {
        String::from_utf8(bytes.to_vec()).context("not valid UTF-8")
    }
}

/// Save plain text as a searchable PDF, wrapping by font metrics and paginating.
/// This creates a new document; it does not edit the layout of an existing PDF.
pub fn encode_text(title: &str, text: &str) -> Result<Vec<u8>> {
    const FONT: &[u8] = include_bytes!("../assets/fonts/RedHatText-Regular.ttf");
    const SIZE: f32 = 10.0;
    const LINE_MM: f32 = 5.0;
    const TOP: f32 = 277.0;
    const BOTTOM: f32 = 20.0;
    const WIDTH_PT: f32 = 170.0 * 72.0 / 25.4;
    let mut doc = PdfDocument::new(title);
    // An empty embedded font has no glyph mapping and is rejected by PDF
    // readers. Blank documents need only a page, with no font resources.
    if text.trim().is_empty() {
        doc.pages
            .push(PdfPage::new(Mm(210.0), Mm(297.0), Vec::new()));
        return Ok(doc.save(&PdfSaveOptions::default(), &mut Vec::new()));
    }
    let face = ttf_parser::Face::parse(FONT, 0).context("failed to load PDF font metrics")?;
    let parsed = ParsedFont::from_bytes(FONT, 0, &mut Vec::new())
        .ok_or_else(|| anyhow!("failed to load PDF font"))?;
    let font = PdfFontHandle::External(doc.add_font(&parsed));
    let mut ops = Vec::new();
    let mut y = TOP;
    let normalized = text
        .replace("\r\n", "\n")
        .replace('\r', "\n")
        .replace('\t', "    ");
    for paragraph in normalized.split('\n') {
        // Hard-wrap long tokens too, so code and URLs cannot run off the page.
        let mut lines = Vec::new();
        let mut line = String::new();
        let mut width = 0.0;
        for ch in paragraph.chars() {
            let advance = face
                .glyph_index(ch)
                .and_then(|id| face.glyph_hor_advance(id))
                .unwrap_or(face.units_per_em() / 2) as f32
                / face.units_per_em() as f32
                * SIZE;
            if width + advance > WIDTH_PT && !line.is_empty() {
                lines.push(std::mem::take(&mut line));
                width = 0.0;
            }
            line.push(ch);
            width += advance;
        }
        lines.push(line);
        for line in lines {
            if y < BOTTOM {
                doc.pages
                    .push(PdfPage::new(Mm(210.0), Mm(297.0), std::mem::take(&mut ops)));
                y = TOP;
            }
            ops.extend([
                Op::StartTextSection,
                Op::SetFont {
                    font: font.clone(),
                    size: Pt(SIZE),
                },
                Op::SetTextCursor {
                    pos: Point::new(Mm(20.0), Mm(y)),
                },
                Op::ShowText {
                    items: vec![TextItem::Text(line)],
                },
                Op::EndTextSection,
            ]);
            y -= LINE_MM;
        }
    }
    doc.pages.push(PdfPage::new(Mm(210.0), Mm(297.0), ops));
    Ok(doc.save(&PdfSaveOptions::default(), &mut Vec::new()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pdf_text_round_trip_and_detection() {
        let bytes = encode_text("Report", "Revenue rose to 42 million.\nCafé résumé.").unwrap();
        assert!(bytes.starts_with(b"%PDF-"));
        let text = decode_text(Path::new("REPORT.PDF"), &bytes).unwrap();
        assert!(text.contains("Revenue rose to 42 million."), "{text}");
        assert!(text.contains("Café résumé."), "{text}");
        assert!(
            decode_text(Path::new("report.bin"), &bytes)
                .unwrap()
                .contains("42 million")
        );
    }

    #[test]
    fn pdf_errors_do_not_become_document_content() {
        assert!(extract_text(b"%PDF-invalid").is_err());
        for text in ["", " \n\t\r\n"] {
            let blank = encode_text("Empty", text).unwrap();
            let error = extract_text(&blank).unwrap_err().to_string();
            assert!(error.contains("OCR"), "{error}");
        }
    }

    #[test]
    fn pdf_pagination_keeps_last_line() {
        let text = format!("{}\nLast page sentinel", "A long report line\n".repeat(130));
        let bytes = encode_text("Long report", &text).unwrap();
        assert!(extract_text(&bytes).unwrap().contains("Last page sentinel"));
    }
}
