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

//! Turning user-uploaded files into text the engine can actually read.
//!
//! The engine is text-only — it has no vision path — so an attachment only
//! reaches the model as text. This module decodes the browser-supplied
//! bytes and pulls plain text out of the formats worth supporting:
//! UTF-8 text/code, PDF, and the OOXML Office trio (Word/Excel/PowerPoint,
//! plus OpenDocument spreadsheets via the same reader). Anything else is
//! kept as a metadata-only reference — its name/type/size are mentioned to
//! the model, but there are no bytes to feed it.
//!
//! The extracted text is stored on the message (see [`super::sessions`]) so
//! that a follow-up turn keeps the document in context, and folded into the
//! prompt by [`compose_content`].

use std::io::{Cursor, Read};

use anyhow::{Context, Result};
use base64::Engine as _;
use serde::Deserialize;

use super::sessions::Attachment;

/// An attachment as it arrives from the browser: the raw file bytes,
/// base64-encoded, plus the name and MIME type the browser reported.
#[derive(Deserialize)]
pub struct IncomingAttachment {
    pub name: String,
    #[serde(default)]
    pub mime: String,
    /// Base64 of the file's raw bytes (no `data:` URL prefix).
    pub data: String,
}

/// Upper bound on extracted text per attachment, in characters. A single
/// oversized document shouldn't be able to blow out the context window (or
/// the on-disk session) — past this the text is truncated with a marker.
const MAX_TEXT_CHARS: usize = 200_000;

/// Decode one incoming attachment and pull whatever text we can out of it.
/// Fails only when the base64 itself is malformed — an unreadable *format*
/// is not an error, it just yields an attachment with `text: None`.
pub fn extract(incoming: &IncomingAttachment) -> Result<Attachment> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(incoming.data.trim())
        .with_context(|| format!("attachment {}: invalid base64", incoming.name))?;
    let size = bytes.len() as u64;

    let mime = if incoming.mime.is_empty() {
        mime_guess::from_path(&incoming.name)
            .first_raw()
            .unwrap_or("application/octet-stream")
            .to_string()
    } else {
        incoming.mime.clone()
    };

    let text = extract_text(&incoming.name, &mime, &bytes).map(cap_text);

    Ok(Attachment {
        name: incoming.name.clone(),
        mime,
        size,
        text,
    })
}

/// Fold a message's typed text and its attachments into the single string
/// the chat template sees. Extractable attachments are inlined as fenced
/// blocks; the rest are named so the model at least knows a file was sent.
pub fn compose_content(text: &str, attachments: &[Attachment]) -> String {
    let text = text.trim();
    if attachments.is_empty() {
        return text.to_string();
    }

    let mut out = String::new();
    if !text.is_empty() {
        out.push_str(text);
    }
    for att in attachments {
        if !out.is_empty() {
            out.push_str("\n\n");
        }
        match &att.text {
            Some(body) => {
                let fence = "`".repeat(fence_width(body));
                out.push_str(&format!(
                    "Attached document \"{}\" ({}):\n{fence}\n{}\n{fence}",
                    att.name, att.mime, body
                ));
            }
            None => out.push_str(&format!(
                "[Attached file \"{}\" ({}, {}) — binary content, not included]",
                att.name,
                att.mime,
                human_size(att.size)
            )),
        }
    }
    out
}

/// How many backticks the fence around an attachment's body needs.
///
/// Three is only safe for a body that contains no fences of its own, and
/// documents worth attaching routinely do — a Markdown file with a
/// ```` ```mermaid ```` block in it being the obvious case. CommonMark ends
/// a fenced block at the first fence *at least as long* as the opening one,
/// so a 3-backtick wrapper around such a file is closed by the file's own
/// closing fence: everything after it escapes the block, and the wrapper's
/// real closing fence goes on to open a new, unterminated one. The model
/// then receives the document with its structure scrambled — which is a
/// good way to get a description of a diagram instead of the diagram.
///
/// One longer than the longest run in the body, and never fewer than three.
fn fence_width(body: &str) -> usize {
    let mut longest = 0;
    let mut run = 0;
    for ch in body.chars() {
        if ch == '`' {
            run += 1;
            longest = longest.max(run);
        } else {
            run = 0;
        }
    }
    longest.max(2) + 1
}

enum Kind {
    Text,
    Pdf,
    /// Word/PowerPoint — OOXML whose text lives in `<*:t>` elements.
    OoxmlRuns,
    /// Excel and OpenDocument spreadsheets — read via calamine.
    Spreadsheet,
    Binary,
}

fn classify(name: &str, mime: &str, bytes: &[u8]) -> Kind {
    let name = name.to_ascii_lowercase();
    let ends = |ext: &str| name.ends_with(ext);

    if ends(".pdf") || mime == "application/pdf" {
        return Kind::Pdf;
    }
    if ends(".docx")
        || ends(".pptx")
        || mime.contains("wordprocessingml")
        || mime.contains("presentationml")
    {
        return Kind::OoxmlRuns;
    }
    if ends(".xlsx")
        || ends(".xlsm")
        || ends(".xlsb")
        || ends(".xls")
        || ends(".ods")
        || mime.contains("spreadsheetml")
        || mime.contains("ms-excel")
        || mime.contains("opendocument.spreadsheet")
    {
        return Kind::Spreadsheet;
    }
    if is_texty(mime, bytes) {
        return Kind::Text;
    }
    Kind::Binary
}

fn extract_text(name: &str, mime: &str, bytes: &[u8]) -> Option<String> {
    let raw = match classify(name, mime, bytes) {
        Kind::Text => Some(String::from_utf8_lossy(bytes).into_owned()),
        Kind::Pdf => extract_pdf(bytes),
        Kind::OoxmlRuns => extract_ooxml_runs(bytes),
        Kind::Spreadsheet => extract_spreadsheet(bytes),
        Kind::Binary => None,
    }?;
    let trimmed = raw.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

/// A file counts as text if its MIME says so, or — for the many code/config
/// files browsers report as `application/octet-stream` — if the bytes are
/// valid UTF-8 with no NUL bytes.
fn is_texty(mime: &str, bytes: &[u8]) -> bool {
    if mime.starts_with("text/") {
        return true;
    }
    if bytes.iter().take(8192).any(|&b| b == 0) {
        return false;
    }
    std::str::from_utf8(bytes).is_ok()
}

/// PDF text extraction. Wrapped in `catch_unwind` because the underlying
/// parser can panic on malformed files, and a bad upload must not take the
/// request handler (or the connection) down — a panic just means "no text".
fn extract_pdf(bytes: &[u8]) -> Option<String> {
    let owned = bytes.to_vec();
    std::panic::catch_unwind(move || pdf_extract::extract_text_from_mem(&owned).ok())
        .ok()
        .flatten()
}

/// Word (`word/document.xml`) and PowerPoint (`ppt/slides/slideN.xml`) both
/// store their visible text as runs in `<*:t>` elements — collect those,
/// breaking a line at every paragraph (`<*:p>`).
fn extract_ooxml_runs(bytes: &[u8]) -> Option<String> {
    let mut zip = zip::ZipArchive::new(Cursor::new(bytes)).ok()?;

    let mut parts: Vec<(String, String)> = Vec::new();
    for i in 0..zip.len() {
        let name = match zip.by_index(i) {
            Ok(f) => f.name().to_string(),
            Err(_) => continue,
        };
        let is_doc = name == "word/document.xml";
        let is_slide = name.starts_with("ppt/slides/slide") && name.ends_with(".xml");
        if !is_doc && !is_slide {
            continue;
        }
        let mut xml = String::new();
        if let Ok(mut entry) = zip.by_name(&name)
            && entry.read_to_string(&mut xml).is_ok()
        {
            parts.push((name, ooxml_runs_to_text(&xml)));
        }
    }
    // Slides sort lexically as slide1/slide10/slide2 — order the numeric
    // suffix so the deck reads top to bottom.
    parts.sort_by_key(|(name, _)| slide_ordinal(name));

    let joined = parts
        .into_iter()
        .map(|(_, text)| text)
        .collect::<Vec<_>>()
        .join("\n\n");
    Some(joined)
}

fn slide_ordinal(name: &str) -> (bool, u64) {
    // (is_slide, number) — non-slides (the single Word doc) keep index 0.
    let digits: String = name
        .trim_end_matches(".xml")
        .chars()
        .rev()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    (name.contains("/slide"), digits.parse().unwrap_or(0))
}

fn ooxml_runs_to_text(xml: &str) -> String {
    use quick_xml::events::Event;

    let mut reader = quick_xml::Reader::from_str(xml);
    let mut out = String::new();
    let mut in_run = false;
    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) => match e.local_name().as_ref() {
                b"t" => in_run = true,
                b"p" => out.push('\n'),
                _ => {}
            },
            Ok(Event::End(e)) => {
                if e.local_name().as_ref() == b"t" {
                    in_run = false;
                }
            }
            Ok(Event::Text(t)) if in_run => {
                if let Ok(decoded) = t.decode() {
                    out.push_str(&decoded);
                }
            }
            // quick-xml surfaces entities (`&amp;`, `&#233;`, …) as their own
            // events rather than folding them into the surrounding text.
            Ok(Event::GeneralRef(r)) if in_run => {
                if let Ok(Some(c)) = r.resolve_char_ref() {
                    out.push(c);
                } else if let Ok(name) = r.decode() {
                    let resolved = match name.as_ref() {
                        "amp" => Some('&'),
                        "lt" => Some('<'),
                        "gt" => Some('>'),
                        "quot" => Some('"'),
                        "apos" => Some('\''),
                        _ => None,
                    };
                    if let Some(c) = resolved {
                        out.push(c);
                    }
                }
            }
            Ok(Event::Eof) | Err(_) => break,
            _ => {}
        }
    }
    out
}

/// Excel / OpenDocument spreadsheets: dump every sheet as tab-separated
/// rows under its name — plenty for the model to reason over the data.
fn extract_spreadsheet(bytes: &[u8]) -> Option<String> {
    use calamine::Reader;

    let mut workbook = calamine::open_workbook_auto_from_rs(Cursor::new(bytes.to_vec())).ok()?;
    let mut out = String::new();
    for name in workbook.sheet_names() {
        let Ok(range) = workbook.worksheet_range(&name) else {
            continue;
        };
        if range.is_empty() {
            continue;
        }
        out.push_str(&format!("# {name}\n"));
        for row in range.rows() {
            let cells: Vec<String> = row.iter().map(|c| c.to_string()).collect();
            out.push_str(&cells.join("\t"));
            out.push('\n');
        }
        out.push('\n');
    }
    Some(out)
}

fn cap_text(text: String) -> String {
    if text.chars().count() <= MAX_TEXT_CHARS {
        return text;
    }
    let capped: String = text.chars().take(MAX_TEXT_CHARS).collect();
    format!("{capped}\n\n[… attachment truncated at {MAX_TEXT_CHARS} characters]")
}

fn human_size(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = 1024.0 * 1024.0;
    let b = bytes as f64;
    if b < KB {
        format!("{bytes} B")
    } else if b < MB {
        format!("{:.1} KB", b / KB)
    } else {
        format!("{:.1} MB", b / MB)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn incoming(name: &str, mime: &str, bytes: &[u8]) -> IncomingAttachment {
        IncomingAttachment {
            name: name.to_string(),
            mime: mime.to_string(),
            data: base64::engine::general_purpose::STANDARD.encode(bytes),
        }
    }

    #[test]
    fn extracts_plain_text_and_reports_size() {
        let att = extract(&incoming("notes.txt", "text/plain", b"hello there")).unwrap();
        assert_eq!(att.text.as_deref(), Some("hello there"));
        assert_eq!(att.size, 11);
        assert_eq!(att.mime, "text/plain");
    }

    #[test]
    fn code_file_with_octet_stream_mime_is_still_text() {
        // Browsers often report unknown extensions as octet-stream — the
        // UTF-8 sniff should still treat a source file as text.
        let att = extract(&incoming(
            "main.rs",
            "application/octet-stream",
            b"fn main() {}",
        ))
        .unwrap();
        assert_eq!(att.text.as_deref(), Some("fn main() {}"));
    }

    #[test]
    fn binary_has_no_text() {
        let att = extract(&incoming(
            "blob.bin",
            "application/octet-stream",
            &[0u8, 1, 2, 3, 255],
        ))
        .unwrap();
        assert!(att.text.is_none());
        assert_eq!(att.size, 5);
    }

    #[test]
    fn invalid_base64_is_an_error() {
        let bad = IncomingAttachment {
            name: "x".into(),
            mime: String::new(),
            data: "not valid base64!!!".into(),
        };
        assert!(extract(&bad).is_err());
    }

    #[test]
    fn missing_mime_is_guessed_from_the_name() {
        let att = extract(&incoming("readme.md", "", b"# Title")).unwrap();
        assert!(att.mime.contains("markdown") || att.mime.starts_with("text/"));
        assert_eq!(att.text.as_deref(), Some("# Title"));
    }

    #[test]
    fn extracts_docx_run_text() {
        let mut buf = Vec::new();
        {
            let mut zw = zip::ZipWriter::new(Cursor::new(&mut buf));
            zw.start_file(
                "word/document.xml",
                zip::write::SimpleFileOptions::default(),
            )
            .unwrap();
            zw.write_all(
                br#"<?xml version="1.0"?><w:document xmlns:w="ns"><w:body>
                    <w:p><w:r><w:t>Hello</w:t></w:r></w:p>
                    <w:p><w:r><w:t xml:space="preserve">World &amp; more</w:t></w:r></w:p>
                    </w:body></w:document>"#,
            )
            .unwrap();
            zw.finish().unwrap();
        }
        let text = extract_ooxml_runs(&buf).unwrap();
        assert!(text.contains("Hello"), "got: {text:?}");
        assert!(
            text.contains("World & more"),
            "entities unescaped: {text:?}"
        );
    }

    #[test]
    fn compose_inlines_documents_and_notes_binaries() {
        let atts = vec![
            Attachment {
                name: "spec.txt".into(),
                mime: "text/plain".into(),
                size: 4,
                text: Some("BODY".into()),
            },
            Attachment {
                name: "img.png".into(),
                mime: "image/png".into(),
                size: 2048,
                text: None,
            },
        ];
        let out = compose_content("look at these", &atts);
        assert!(out.starts_with("look at these"));
        assert!(out.contains("Attached document \"spec.txt\""));
        assert!(out.contains("BODY"));
        assert!(out.contains("[Attached file \"img.png\" (image/png, 2.0 KB)"));
    }

    #[test]
    fn compose_without_attachments_is_just_the_text() {
        assert_eq!(compose_content("hi", &[]), "hi");
    }

    #[test]
    fn fence_width_outgrows_the_body() {
        assert_eq!(fence_width("plain text"), 3);
        assert_eq!(fence_width("a ``` fence"), 4);
        assert_eq!(fence_width("````\nx\n````"), 5);
        // Inline code shouldn't inflate the fence past what's needed.
        assert_eq!(fence_width("use `foo` here"), 3);
    }

    #[test]
    fn a_document_containing_fences_stays_one_block() {
        // The real shape that broke this: a Markdown file with a mermaid
        // block in it. The wrapper has to outlive the file's own fences, or
        // the document arrives at the model in pieces.
        let atts = vec![Attachment {
            name: "entity-diagram.md".into(),
            mime: "text/markdown".into(),
            size: 0,
            text: Some("# Entities\n\n```mermaid\nerDiagram\n    A ||--o{ B : has\n```\n".into()),
        }];
        let out = compose_content("Please, render this", &atts);

        // Parsing the composed prompt back must yield exactly one code
        // block holding the whole document — not the three fragments a
        // 3-backtick wrapper produced.
        let tree = markdown::to_mdast(&out, &markdown::ParseOptions::gfm()).unwrap();
        let markdown::mdast::Node::Root(root) = &tree else {
            panic!("root")
        };
        let blocks: Vec<_> = root
            .children
            .iter()
            .filter_map(|n| match n {
                markdown::mdast::Node::Code(c) => Some(c),
                _ => None,
            })
            .collect();
        assert_eq!(blocks.len(), 1, "composed prompt: {out}");
        assert!(
            blocks[0].value.contains("```mermaid"),
            "{}",
            blocks[0].value
        );
        assert!(blocks[0].value.contains("erDiagram"), "{}", blocks[0].value);
    }

    #[test]
    fn human_size_scales() {
        assert_eq!(human_size(512), "512 B");
        assert_eq!(human_size(1536), "1.5 KB");
        assert_eq!(human_size(3 * 1024 * 1024), "3.0 MB");
    }
}
