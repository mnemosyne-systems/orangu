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

//! Local OCR using existing Rust render/inference libraries. No downloads occur here.

use super::tables::{self, PageContent};
use anyhow::{Context, Result, anyhow, bail};
use hayro::{
    PixmapSettings, RenderCache, RenderSettings, hayro_interpret::InterpreterSettings,
    hayro_syntax::Pdf,
};
use ocrs::TextItem as _;
use ocrs::{ImageSource, OcrEngine, OcrEngineParams};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex, OnceLock},
};

pub(super) trait PageOcr {
    fn recognize(&mut self, page: usize) -> Result<String>;
    fn recognize_layout(&mut self, page: usize) -> Result<PageContent> {
        self.recognize(page).map(PageContent::plain)
    }
}

type CachedEngine = Option<(PathBuf, Arc<OcrEngine>)>;
static ENGINE: OnceLock<Mutex<CachedEngine>> = OnceLock::new();

pub(super) struct LocalOcr<'a> {
    bytes: &'a [u8],
    pdf: Option<Pdf>,
    engine: Option<Arc<OcrEngine>>,
    load_error: Option<String>,
}

impl<'a> LocalOcr<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            pdf: None,
            engine: None,
            load_error: None,
        }
    }

    fn engine(&mut self) -> Result<Arc<OcrEngine>> {
        if let Some(engine) = &self.engine {
            return Ok(engine.clone());
        }
        if let Some(error) = &self.load_error {
            bail!("{error}");
        }
        let loaded = load_engine();
        match loaded {
            Ok(engine) => {
                self.engine = Some(engine.clone());
                Ok(engine)
            }
            Err(error) => {
                self.load_error = Some(format!("{error:#}"));
                Err(error)
            }
        }
    }
}

fn load_engine() -> Result<Arc<OcrEngine>> {
    let directory = std::env::var_os("ORANGU_OCR_MODEL_DIR")
        .map(PathBuf::from)
        .or_else(|| home::home_dir().map(|p| p.join(".orangu/ocr")))
        .ok_or_else(|| anyhow!("OCR model directory is unavailable; set ORANGU_OCR_MODEL_DIR"))?;
    let mut cache = ENGINE
        .get_or_init(|| Mutex::new(None))
        .lock()
        .map_err(|_| anyhow!("OCR model cache is unavailable"))?;
    if let Some((cached_directory, engine)) = &*cache
        && cached_directory == &directory
    {
        return Ok(engine.clone());
    }
    let detection = directory.join("text-detection.onnx");
    let recognition = directory.join("text-recognition.onnx");
    if !detection.is_file() || !recognition.is_file() {
        bail!(
            "OCR models are missing in {}. Install text-detection.onnx and text-recognition.onnx there, or set ORANGU_OCR_MODEL_DIR. See the PDF section of the tools manual",
            directory.display()
        );
    }
    let engine = Arc::new(
        OcrEngine::new(OcrEngineParams {
            detection_model: Some(
                rten::Model::load_file(&detection).context("Cannot load OCR detection model")?,
            ),
            recognition_model: Some(
                rten::Model::load_file(&recognition)
                    .context("Cannot load OCR recognition model")?,
            ),
            ..Default::default()
        })
        .context("Cannot initialize local OCR")?,
    );
    *cache = Some((directory, engine.clone()));
    Ok(engine)
}

impl PageOcr for LocalOcr<'_> {
    fn recognize(&mut self, page: usize) -> Result<String> {
        Ok(self.recognize_layout(page)?.text)
    }

    fn recognize_layout(&mut self, page: usize) -> Result<PageContent> {
        let engine = self.engine()?;
        if self.pdf.is_none() {
            self.pdf = Some(
                Pdf::new(self.bytes.to_vec())
                    .map_err(|e| anyhow!("Cannot render PDF for OCR: {e:?}"))?,
            );
        }
        let pdf = self.pdf.as_ref().unwrap();
        let page = pdf
            .pages()
            .get(page - 1)
            .ok_or_else(|| anyhow!("PDF page is unavailable for OCR"))?;
        let (width, height) = page.render_dimensions();
        let scale = render_scale(width, height)?;
        let pixmap = hayro::render(
            page,
            &RenderCache::new(),
            &InterpreterSettings::default(),
            &RenderSettings::default(),
            &PixmapSettings {
                x_scale: scale,
                y_scale: scale,
                bg_color: hayro::vello_cpu::color::palette::css::WHITE,
            },
        );
        // The white opaque background makes premultiplied RGB identical to RGB.
        let rgb: Vec<u8> = pixmap
            .data_as_u8_slice()
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|pixel| [pixel[0], pixel[1], pixel[2]])
            .collect();
        let edges = tables::raster_edges(
            &rgb,
            u32::from(pixmap.width()),
            u32::from(pixmap.height()),
            f64::from(scale),
        );
        let cleaned = tables::without_grid(
            &rgb,
            u32::from(pixmap.width()),
            u32::from(pixmap.height()),
            f64::from(scale),
            &edges,
        );
        let image = ImageSource::from_bytes(
            &cleaned,
            (u32::from(pixmap.width()), u32::from(pixmap.height())),
        )?;
        let input = engine.prepare_input(image)?;
        let words = engine.detect_words(&input)?;
        let lines = engine.find_text_lines(&input, &words);
        let recognized = engine.recognize_text(&input, &lines)?;
        let mut text = String::new();
        let mut glyphs = Vec::new();
        let mut ranges = Vec::new();
        for line in recognized.iter().flatten() {
            for ch in line.chars() {
                let before = text.len();
                text.push(ch.char);
                let rect = &ch.rect;
                let scale = f64::from(scale);
                let bounds = pdfni::BBox {
                    x0: f64::from(rect.left()) / scale,
                    x1: f64::from(rect.right()) / scale,
                    top: f64::from(rect.top()) / scale,
                    bottom: f64::from(rect.bottom()) / scale,
                };
                glyphs.push(pdfni::GlyphPart {
                    ch: ch.char,
                    left: bounds.x0,
                    right: bounds.x1,
                    top: bounds.top,
                    bottom: bounds.bottom,
                    font_size: Some(bounds.height()),
                    upright: true,
                    rot: 0,
                });
                ranges.push((before..text.len(), bounds));
            }
            text.push('\n');
        }
        let overflow = glyphs.len() > tables::MAX_GLYPHS || edges.len() > tables::MAX_EDGES;
        let part = pdfni::PagePart {
            width: f64::from(width),
            height: f64::from(height),
            glyphs,
            edges,
            graphics: Vec::new(),
            norm_rotate: 0,
        };
        Ok(tables::reconstruct(text, part, ranges, overflow))
    }
}

fn render_scale(width: f32, height: f32) -> Result<f32> {
    if !width.is_finite() || !height.is_finite() || width <= 0.0 || height <= 0.0 {
        bail!("PDF page has invalid dimensions");
    }
    // Aim for 200 DPI, with bounded memory even for oversized page dimensions.
    let scale = (200.0_f32 / 72.0)
        .min(4096.0 / width.max(height))
        .min((12_000_000.0 / (f64::from(width) * f64::from(height))).sqrt() as f32);
    if width * scale < 1.0 || height * scale < 1.0 {
        bail!("PDF page dimensions cannot be rendered safely");
    }
    Ok(scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pdf_extract::dictionary;

    #[test]
    #[ignore = "requires local text-detection.onnx and text-recognition.onnx models"]
    fn scanned_pdf_recovers_text_with_real_local_models() {
        // Rasterize real text, then replace the page with only that image.
        // Native extraction must be empty, so passing proves OCR was used.
        let native = crate::pdf::encode_text(
            "Harvest",
            "Apricot harvest report.\nProduction reached forty two tonnes.",
        )
        .unwrap();
        let bytes = image_only_pdf(&native, Some(300));
        assert!(
            pdf_extract::extract_text_from_mem(&bytes)
                .unwrap()
                .trim()
                .is_empty()
        );
        let result = crate::pdf::extract_document(&bytes, &Default::default()).unwrap();
        assert_eq!(result.pages[0].source, crate::pdf::PageSource::Ocr);
        assert!(
            result.pages[0].text.to_lowercase().contains("apricot"),
            "{}",
            result.pages[0].text
        );
        assert!(result.context_text().unwrap().contains("Page 1 | OCR"));
    }

    fn image_only_pdf(native: &[u8], crop_height: Option<usize>) -> Vec<u8> {
        let rendered_pdf = Pdf::new(native.to_vec()).unwrap();
        let pixmap = hayro::render(
            &rendered_pdf.pages()[0],
            &RenderCache::new(),
            &InterpreterSettings::default(),
            &RenderSettings::default(),
            &PixmapSettings {
                x_scale: 2.0,
                y_scale: 2.0,
                bg_color: hayro::vello_cpu::color::palette::css::WHITE,
            },
        );
        let width = usize::from(pixmap.width());
        let height = crop_height
            .unwrap_or(usize::from(pixmap.height()))
            .min(usize::from(pixmap.height()));
        let rgb: Vec<u8> = pixmap.data_as_u8_slice()[..width * height * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect();
        let mut scanned = pdf_extract::Document::load_mem(native).unwrap();
        let image = scanned.add_object(pdf_extract::Stream::new(
            dictionary! {
                "Type" => "XObject", "Subtype" => "Image", "Width" => width as i64,
                "Height" => height as i64, "ColorSpace" => "DeviceRGB", "BitsPerComponent" => 8
            },
            rgb,
        ));
        let contents = format!(
            "q {} 0 0 {} 0 0 cm /Scan Do Q",
            width as f32 / 2.0,
            height as f32 / 2.0
        );
        let contents = scanned.add_object(pdf_extract::Stream::new(
            dictionary! {},
            contents.into_bytes(),
        ));
        let page_id = *scanned.get_pages().values().next().unwrap();
        let page = scanned
            .get_object_mut(page_id)
            .unwrap()
            .as_dict_mut()
            .unwrap();
        page.set(
            "Resources",
            dictionary! { "XObject" => dictionary! { "Scan" => image } },
        );
        page.set("Contents", contents);
        page.set(
            "MediaBox",
            vec![
                pdf_extract::Object::Integer(0),
                0.into(),
                (width as f32 / 2.0).into(),
                (height as f32 / 2.0).into(),
            ],
        );
        let mut bytes = Vec::new();
        scanned.save_to(&mut bytes).unwrap();
        bytes
    }

    #[test]
    #[ignore = "requires local text-detection.onnx and text-recognition.onnx models"]
    fn scanned_tables_keep_values_in_their_cells_with_real_local_models() {
        for merged in [false, true] {
            let mut native = super::super::tables::tests::table_pdf(merged, true);
            if merged {
                // Exercise the downsampled grid detector on a full-size page.
                let mut document = pdf_extract::Document::load_mem(&native).unwrap();
                let id = *document.get_pages().values().next().unwrap();
                document
                    .get_object_mut(id)
                    .unwrap()
                    .as_dict_mut()
                    .unwrap()
                    .set("MediaBox", vec![0.into(), 0.into(), 612.into(), 792.into()]);
                native.clear();
                document.save_to(&mut native).unwrap();
            }
            let bytes = image_only_pdf(&native, None);
            let report = crate::pdf::extract_document(&bytes, &Default::default()).unwrap();
            let page = &report.pages[0];
            assert_eq!(page.source, crate::pdf::PageSource::Ocr);
            assert_eq!(page.tables.len(), 1, "{}; {:?}", page.text, page.warning);
            let table = &page.tables[0];
            let cell = |r, c| {
                table
                    .cells
                    .iter()
                    .find(|cell| cell.row == r && cell.column == c)
                    .unwrap()
            };
            if merged {
                assert_eq!(cell(1, 1).column_span, 3);
                assert_eq!(cell(3, 1).row_span, 2);
                assert_eq!(cell(3, 2).text, "12");
            } else {
                assert_eq!(cell(2, 2).text, "12");
                assert_eq!(cell(2, 3).text, "13");
                assert_eq!(cell(3, 2).text, "");
                assert_eq!(cell(3, 3).text, "23");
            }
            assert!(report.context_text().unwrap().contains("<table>"));
        }
    }

    #[test]
    fn rendering_is_bounded_for_large_pages_and_rejects_invalid_dimensions() {
        for (width, height) in [(595.0, 842.0), (100_000.0, 200_000.0)] {
            let scale = render_scale(width, height).unwrap();
            assert!(width.max(height) * scale <= 4096.1);
            assert!(f64::from(width * scale) * f64::from(height * scale) <= 12_000_001.0);
        }
        assert!(render_scale(f32::NAN, 100.0).is_err());
        assert!(render_scale(100.0, 0.0).is_err());
    }
}
