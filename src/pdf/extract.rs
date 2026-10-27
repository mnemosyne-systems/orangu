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

use super::ocr::{LocalOcr, PageOcr};
use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashSet,
    panic::{AssertUnwindSafe, catch_unwind},
};

const MAX_FILE_BYTES: usize = 64 * 1024 * 1024;
const DEFAULT_PAGE_WINDOW: usize = 20;
const MAX_QUERY_PAGES: usize = 200;
const MAX_OCR_PAGES: usize = 20;
const MAX_PAGE_CHARS: usize = 30_000;
const MAX_CONTEXT_CHARS: usize = 60_000;

/// Page ranges are inclusive and start at one. A query selects up to six
/// matching pages by keyword overlap, retaining their original page numbers.
#[derive(Debug, Default, Clone, Deserialize, Serialize)]
pub struct PdfReadOptions {
    pub start_page: Option<usize>,
    pub end_page: Option<usize>,
    pub query: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PageSource {
    EmbeddedText,
    Ocr,
    Empty,
    Unreadable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PdfPageText {
    pub number: usize,
    pub source: PageSource,
    pub text: String,
    pub warning: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tables: Vec<super::PdfTable>,
    /// Narrative text without table glyphs. None preserves legacy page indexes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body_text: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedPdf {
    pub total_pages: usize,
    pub inspected_start: usize,
    pub inspected_end: usize,
    pub query: Option<String>,
    pub pages: Vec<PdfPageText>,
}

impl ExtractedPdf {
    /// Select source pages again for a follow-up without repeating OCR.
    pub fn context_for_query(&self, query: &str) -> Result<String> {
        if let Some(number) = requested_page(query) {
            let Some(page) = self.pages.iter().find(|page| page.number == number) else {
                return Ok(format!(
                    "[Requested PDF page {number} is not in the stored page index. The index covers pages {}-{} of {}. No content from the requested page is available.]",
                    self.inspected_start, self.inspected_end, self.total_pages
                ));
            };
            if page.text.is_empty() {
                return Ok(format!(
                    "[Requested PDF page {number} is unavailable: {}. Do not infer its contents.]",
                    page.warning.as_deref().unwrap_or("no text was detected")
                ));
            }
            let selected = Self {
                total_pages: self.total_pages,
                inspected_start: self.inspected_start,
                inspected_end: self.inspected_end,
                query: None,
                pages: vec![page.clone()],
            };
            return Ok(format!(
                "Selected the requested PDF page; other pages are omitted.\n{}",
                selected.context_text()?
            ));
        }
        let terms = words(query);
        let mut scores: Vec<_> = self
            .pages
            .iter()
            .enumerate()
            .filter(|(_, p)| !p.text.is_empty())
            .map(|(i, p)| (terms.intersection(&page_words(p)).count(), i))
            .collect();
        scores.sort_by_key(|&(score, i)| (std::cmp::Reverse(score), i));
        if scores.iter().any(|(score, _)| *score > 0) {
            scores.retain(|(score, _)| *score > 0);
        }
        let selected: HashSet<_> = scores.into_iter().take(6).map(|(_, i)| i).collect();
        let mut doc = self.clone();
        doc.pages = self
            .pages
            .iter()
            .enumerate()
            .filter(|(i, _)| selected.contains(i))
            .map(|(_, p)| p.clone())
            .collect();
        doc.query = None;
        let unreadable: Vec<_> = self
            .pages
            .iter()
            .filter(|p| p.source == PageSource::Unreadable || p.warning.is_some())
            .map(|p| p.number.to_string())
            .collect();
        let mut context = format!(
            "Selected up to six stored PDF pages for this question; other stored pages are omitted.\n{}",
            doc.context_text()?
        );
        if !unreadable.is_empty() {
            context.push_str(&format!(
                "\n[PDF pages with unavailable or partial extraction: {}]\n",
                unreadable.join(", ")
            ));
        }
        Ok(context)
    }

    /// Both the model and the UI see the source and any omissions. OCR text is
    /// labelled as uncertain rather than assigned an invented confidence score.
    pub fn context_text(&self) -> Result<String> {
        if self.pages.iter().all(|page| page.text.is_empty()) {
            let reason = self
                .pages
                .iter()
                .find_map(|page| page.warning.as_deref())
                .unwrap_or("the selected pages are blank");
            bail!(
                "PDF has no readable text in pages {}-{} of {}. OCR may be required. {reason}",
                self.inspected_start,
                self.inspected_end,
                self.total_pages
            );
        }
        let mut out = format!(
            "PDF: {} pages. Inspected pages {}-{}. Cite the original page numbers below.\n",
            self.total_pages, self.inspected_start, self.inspected_end
        );
        if self.inspected_start > 1 || self.inspected_end < self.total_pages {
            out.push_str(
                "Other pages were not inspected. Use start_page/end_page to read another range.\n",
            );
        }
        if self.query.is_some() {
            out.push_str("Showing up to six pages matched by query keywords. Unselected pages are omitted.\n");
        }
        let mut remaining = MAX_CONTEXT_CHARS;
        for page in &self.pages {
            let source = match page.source {
                PageSource::EmbeddedText => "embedded text",
                PageSource::Ocr => "OCR; may contain recognition errors",
                PageSource::Empty => "no text detected",
                PageSource::Unreadable => "unreadable",
            };
            out.push_str(&format!("\n[Page {} | {source}]\n", page.number));
            if let Some(warning) = &page.warning {
                out.push_str(&format!("[Extraction warning: {warning}]\n"));
            }
            if remaining == 0 {
                out.push_str("[PDF context limit reached; remaining pages are omitted. Read a narrower page range.]\n");
                break;
            }
            if !page.tables.is_empty() {
                out.push_str("[Use the structured cells for table values. Empty cells remain empty; spanning cells cover the specified rows/columns.]\n");
            }
            for (index, table) in page.tables.iter().enumerate() {
                let html = table.context_html(index + 1);
                let size = html.chars().count();
                if size > remaining {
                    out.push_str("[An entire table was omitted because it exceeds the remaining context budget. Do not infer its cell values from flattened text.]\n");
                } else {
                    out.push_str(&html);
                    remaining -= size;
                }
            }
            let page_body = page.body_text.as_deref().unwrap_or(&page.text);
            let body: String = page_body.chars().take(remaining).collect();
            remaining -= body.chars().count();
            out.push_str(&body);
            out.push('\n');
            if body.len() < page_body.len() {
                out.push_str(
                    "[Page text truncated at the PDF context limit. Read a narrower range.]\n",
                );
            }
        }
        Ok(out)
    }
}

pub fn extract_document(bytes: &[u8], options: &PdfReadOptions) -> Result<ExtractedPdf> {
    catch_unwind(AssertUnwindSafe(|| {
        let mut ocr = LocalOcr::new(bytes);
        extract_with(bytes, options, &mut ocr)
    }))
    .map_err(|_| anyhow!("PDF parser failed while reading this document"))?
}

/// Retain a bounded page index for upload follow-ups without repeating OCR.
pub fn extract_upload(bytes: &[u8]) -> Result<ExtractedPdf> {
    catch_unwind(AssertUnwindSafe(|| {
        let mut ocr = LocalOcr::new(bytes);
        extract_with_window(bytes, &PdfReadOptions::default(), &mut ocr, MAX_QUERY_PAGES)
    }))
    .map_err(|_| anyhow!("PDF parser failed while reading this document"))?
}

fn extract_with(
    bytes: &[u8],
    options: &PdfReadOptions,
    ocr: &mut impl PageOcr,
) -> Result<ExtractedPdf> {
    extract_with_window(bytes, options, ocr, DEFAULT_PAGE_WINDOW)
}

fn extract_with_window(
    bytes: &[u8],
    options: &PdfReadOptions,
    ocr: &mut impl PageOcr,
    default_window: usize,
) -> Result<ExtractedPdf> {
    if bytes.len() > MAX_FILE_BYTES {
        bail!("PDF exceeds the 64 MiB input limit");
    }
    if options
        .query
        .as_ref()
        .is_some_and(|query| words(query).is_empty())
    {
        bail!("PDF query must contain at least one word");
    }
    let mut document = pdf_extract::Document::load_mem(bytes)
        .context("Cannot open PDF; the document may be damaged")?;
    if document.is_encrypted() {
        document
            .decrypt("")
            .context("Cannot read password-protected PDF without its password")?;
    }
    let page_ids = document.get_pages();
    let total_pages = page_ids.len();
    if total_pages == 0 {
        bail!("PDF has no pages");
    }
    let start = options.start_page.unwrap_or(1);
    let window = if options.query.is_some() {
        MAX_QUERY_PAGES
    } else {
        default_window
    };
    let end = options
        .end_page
        .unwrap_or(start.saturating_add(window - 1).min(total_pages));
    if start == 0 || end < start || end > total_pages {
        bail!("PDF page range must be within 1-{total_pages}, with start_page <= end_page");
    }
    if end - start + 1 > MAX_QUERY_PAGES {
        bail!("Read at most {MAX_QUERY_PAGES} PDF pages per request; use a narrower page range");
    }
    let mut pages = Vec::new();
    let mut ocr_pages = 0;
    let mut remaining_chars = 200_000;
    for (&number, &id) in page_ids.iter().skip(start - 1).take(end - start + 1) {
        if remaining_chars == 0 {
            pages.push(PdfPageText {
                number: number as usize, source: PageSource::Unreadable, text: String::new(),
                warning: Some("Document extraction reached 200000 characters. Read a narrower page range to extract this page".into()),
                tables: Vec::new(), body_text: None,
            });
            continue;
        }
        // The by-pages convenience API stops at the first failed page. Use
        // the library's page API to retain later readable pages and failures.
        let embedded = catch_unwind(AssertUnwindSafe(|| {
            super::layout::extract_page(&document, number)
        }))
        .unwrap_or_else(|_| Err(anyhow!("Embedded text parser failed on this page")));
        let mut warning = embedded.as_ref().err().map(|e| format!("{e:#}"));
        let embedded = embedded.unwrap_or_default();
        if embedded.warning.is_some() {
            warning = embedded.warning;
        }
        let mut text = embedded.text;
        let mut tables = embedded.tables;
        let mut body_text = embedded.body_text;
        let mut source = if text.is_empty() {
            PageSource::Unreadable
        } else {
            PageSource::EmbeddedText
        };
        let has_images = catch_unwind(AssertUnwindSafe(|| {
            document
                .get_page_images(id)
                .is_ok_and(|images| !images.is_empty())
        }))
        .unwrap_or(false);
        if text.is_empty()
            || (text.chars().filter(|ch| ch.is_alphanumeric()).count() < 32
                && has_images
                && tables.is_empty())
        {
            let recognized = if ocr_pages >= MAX_OCR_PAGES {
                Err(anyhow!(
                    "OCR page limit reached; read a narrower page range to process this page"
                ))
            } else {
                ocr_pages += 1;
                catch_unwind(AssertUnwindSafe(|| ocr.recognize_layout(number as usize)))
                    .unwrap_or_else(|_| Err(anyhow!("OCR failed on this page")))
            };
            match recognized {
                Ok(recognized) if !recognized.text.trim().is_empty() => {
                    // Sparse native text may just be the footer on a scan.
                    // Keep it too if the OCR result did not recover it.
                    let recognized_text = recognized.text.trim();
                    body_text = recognized.body_text;
                    tables = recognized.tables;
                    if !text.is_empty() && !recognized_text.contains(&text) {
                        if let Some(body) = &mut body_text {
                            body.push_str(&format!("\n[Additional embedded text]\n{text}"));
                        }
                        text = format!("{recognized_text}\n[Additional embedded text]\n{text}");
                    } else {
                        text = recognized_text.to_owned();
                    }
                    source = PageSource::Ocr;
                    warning = recognized.warning;
                }
                Ok(_) if text.is_empty() => {
                    source = PageSource::Empty;
                }
                Ok(_) => {
                    warning = Some(
                        "OCR found no additional text; only embedded text is available".into(),
                    );
                }
                Err(error) => {
                    warning = Some(format!("{error:#}"));
                }
            }
        }
        let limit = MAX_PAGE_CHARS.min(remaining_chars);
        if text.chars().count() > limit {
            text = text.chars().take(limit).collect();
            add_warning(
                &mut warning,
                format!("Page text truncated at {limit} characters"),
            );
        }
        // Store whole tables, never partial cells. The narrative was filtered
        // before truncation, so omitted tables cannot leak back as flat text.
        let mut table_budget = limit;
        tables.retain(|table| {
            let size: usize = table.cells.iter().map(|cell| cell.text.chars().count()).sum();
            if size > table_budget {
                add_warning(&mut warning, "An entire table exceeded the stored text limit and was omitted; read a narrower page range".into());
                false
            } else {
                table_budget -= size;
                true
            }
        });
        if let Some(body) = &mut body_text {
            *body = body.chars().take(limit).collect();
        }
        remaining_chars -= text.chars().count();
        pages.push(PdfPageText {
            number: number as usize,
            source,
            text,
            warning,
            tables,
            body_text,
        });
    }
    if let Some(query) = &options.query {
        let terms = words(query);
        if terms.is_empty() {
            bail!("PDF query must contain at least one word");
        }
        let mut scored: Vec<_> = pages
            .iter()
            .enumerate()
            .map(|(i, page)| {
                let tokens = page_words(page);
                (terms.intersection(&tokens).count(), i)
            })
            .filter(|(score, _)| *score > 0)
            .collect();
        scored.sort_by_key(|&(score, i)| (std::cmp::Reverse(score), i));
        let selected: HashSet<_> = scored.into_iter().take(6).map(|(_, i)| i).collect();
        if selected.is_empty() {
            let unavailable = pages
                .iter()
                .filter(|p| p.source == PageSource::Unreadable)
                .count();
            bail!(
                "No PDF pages matched the query in pages {start}-{end}. {unavailable} pages were unreadable. Try different keywords or another page range"
            );
        }
        // Preserve extraction warnings for unreadable pages even when they
        // cannot match a query. Otherwise partial results would look complete.
        pages = pages
            .into_iter()
            .enumerate()
            .filter(|(i, p)| selected.contains(i) || p.warning.is_some())
            .map(|(_, p)| p)
            .collect();
    }
    Ok(ExtractedPdf {
        total_pages,
        inspected_start: start,
        inspected_end: end,
        query: options.query.clone(),
        pages,
    })
}

fn add_warning(warning: &mut Option<String>, message: String) {
    if let Some(existing) = warning {
        existing.push_str("; ");
        existing.push_str(&message);
    } else {
        *warning = Some(message);
    }
}

fn words(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
        .collect()
}

fn page_words(page: &PdfPageText) -> HashSet<String> {
    let mut terms = words(&page.text);
    for cell in page.tables.iter().flat_map(|table| &table.cells) {
        terms.extend(words(&cell.text));
    }
    terms
}

fn requested_page(query: &str) -> Option<usize> {
    let mut after_page = false;
    for word in query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
    {
        if after_page && let Ok(number) = word.parse() {
            return Some(number);
        }
        after_page = word.eq_ignore_ascii_case("page");
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StubOcr {
        calls: Vec<usize>,
        fail: bool,
    }
    impl PageOcr for StubOcr {
        fn recognize(&mut self, page: usize) -> Result<String> {
            self.calls.push(page);
            if self.fail {
                bail!("OCR models missing for this test");
            }
            Ok(format!("Scanned content from page {page}"))
        }
    }

    fn mixed_document() -> Vec<u8> {
        let bytes =
            crate::pdf::encode_text("Mixed", "Native text is read without optical recognition.")
                .unwrap();
        let mut doc = pdf_extract::Document::load_mem(&bytes).unwrap();
        let first = *doc.get_pages().values().next().unwrap();
        let parent = doc
            .get_dictionary(first)
            .unwrap()
            .get(b"Parent")
            .unwrap()
            .as_reference()
            .unwrap();
        let blank = doc.add_object(pdf_extract::dictionary! { "Type" => "Page", "Parent" => parent, "MediaBox" => vec![0.into(), 0.into(), 595.into(), 842.into()] });
        let parent = doc.get_object_mut(parent).unwrap().as_dict_mut().unwrap();
        parent.set(
            "Kids",
            vec![
                pdf_extract::Object::Reference(first),
                pdf_extract::Object::Reference(blank),
            ],
        );
        parent.set("Count", 2);
        let mut bytes = Vec::new();
        doc.save_to(&mut bytes).unwrap();
        bytes
    }

    #[test]
    fn mixed_pages_keep_numbers_and_only_missing_text_uses_ocr() {
        let mut ocr = StubOcr {
            calls: Vec::new(),
            fail: false,
        };
        let doc = extract_with(&mixed_document(), &PdfReadOptions::default(), &mut ocr).unwrap();
        assert_eq!(ocr.calls, [2]);
        assert_eq!(doc.pages[0].source, PageSource::EmbeddedText);
        assert_eq!(doc.pages[1].source, PageSource::Ocr);
        let context = doc.context_text().unwrap();
        assert!(context.contains("[Page 2 | OCR"));
        assert!(context.contains("Scanned content from page 2"));
    }

    #[test]
    fn failed_ocr_keeps_native_pages_and_reports_the_missing_page() {
        let mut ocr = StubOcr {
            calls: Vec::new(),
            fail: true,
        };
        let doc = extract_with(&mixed_document(), &PdfReadOptions::default(), &mut ocr).unwrap();
        let context = doc.context_text().unwrap();
        assert!(context.contains("Native text is read"));
        assert!(context.contains("[Page 2 | unreadable]"));
        assert!(context.contains("OCR models missing"));
        let focused = doc.context_for_query("What is on page 2?").unwrap();
        assert!(focused.contains("page 2 is unavailable"));
        assert!(focused.contains("OCR models missing"));
    }

    #[test]
    fn page_ranges_and_queries_preserve_original_page_references() {
        let bytes = mixed_document();
        let mut ocr = StubOcr {
            calls: Vec::new(),
            fail: false,
        };
        let options = PdfReadOptions {
            start_page: Some(2),
            end_page: Some(2),
            query: Some("Scanned".into()),
        };
        let doc = extract_with(&bytes, &options, &mut ocr).unwrap();
        assert_eq!(doc.pages[0].number, 2);
        assert!(
            doc.context_text()
                .unwrap()
                .contains("Other pages were not inspected")
        );
        let options = PdfReadOptions {
            start_page: Some(0),
            ..Default::default()
        };
        assert!(extract_with(&bytes, &options, &mut ocr).is_err());
    }

    #[test]
    fn follow_up_selects_new_pages_from_the_persisted_index() {
        let mut pages: Vec<_> = (1..=9)
            .map(|number| PdfPageText {
                number,
                source: PageSource::EmbeddedText,
                text: format!("Ordinary content on page {number}"),
                warning: None,
                tables: Vec::new(),
                body_text: None,
            })
            .collect();
        pages[8].text = "Apricot harvest reached forty two tonnes".into();
        let document = ExtractedPdf {
            total_pages: 9,
            inspected_start: 1,
            inspected_end: 9,
            query: None,
            pages,
        };
        let persisted = serde_json::to_string(&document).unwrap();
        let restored: ExtractedPdf = serde_json::from_str(&persisted).unwrap();
        let context = restored.context_for_query("apricot harvest").unwrap();
        assert!(context.contains("[Page 9 | embedded text]"));
        assert!(context.contains("forty two tonnes"));
        assert!(!context.contains("[Page 8 |"));
        let focused = restored.context_for_query("What does page 9 say?").unwrap();
        assert!(focused.contains("Apricot harvest"));
        assert!(!focused.contains("Ordinary content"));
    }

    #[test]
    fn tables_survive_persistence_and_follow_up_queries_without_models() {
        let mut ocr = StubOcr {
            calls: Vec::new(),
            fail: true,
        };
        let bytes = super::super::tables::tests::table_pdf(true, true);
        let doc = extract_with(&bytes, &Default::default(), &mut ocr).unwrap();
        assert!(ocr.calls.is_empty());
        let restored: ExtractedPdf =
            serde_json::from_str(&serde_json::to_string(&doc).unwrap()).unwrap();
        let context = restored.context_for_query("Apples Q1").unwrap();
        assert!(context.contains("colspan=\"3\""));
        assert!(context.contains("rowspan=\"2\""));
        assert_eq!(context.matches("Quarterly report").count(), 1);
        assert!(context.contains("Narrative outside the table."));
    }

    #[test]
    fn context_budget_omits_whole_tables_without_flattened_fallback() {
        let mut ocr = StubOcr {
            calls: Vec::new(),
            fail: true,
        };
        let bytes = super::super::tables::tests::table_pdf(false, true);
        let mut doc = extract_with(&bytes, &Default::default(), &mut ocr).unwrap();
        doc.pages[0].tables[0].cells[0].text = "X".repeat(MAX_CONTEXT_CHARS);
        let context = doc.context_text().unwrap();
        assert!(context.contains("An entire table was omitted"));
        assert!(!context.contains("<table>"));
        assert!(!context.contains("Berries"));
        assert!(context.contains("Narrative outside the table."));
    }

    #[test]
    fn legacy_page_indexes_still_load() {
        let page: PdfPageText = serde_json::from_str(
            r#"{"number":1,"source":"embedded_text","text":"Legacy content","warning":null}"#,
        )
        .unwrap();
        assert!(page.tables.is_empty());
        assert!(page.body_text.is_none());
    }
}
