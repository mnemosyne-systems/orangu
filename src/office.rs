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

//! LibreOffice documents, written and converted in pure Rust through
//! [`libreoffice_pure`]: no LibreOffice install is needed.
//!
//! A model cannot write an ODF package byte by byte, so every document starts
//! as text in a format it can write, which is rendered into the target:
//!
//! | Source format | Target |
//! | :-- | :-- |
//! | `md`, `html`, `txt` | `odt`, `docx`, `pdf`, … |
//! | `csv` (`=` starts a formula) | `ods`, `xlsx`, `odb`, `pdf`, … |
//! | `outline` (`# slide`, `- bullet`) | `odp`, `pptx`, `pdf`, … |
//! | `svg` | `odg`, `pdf` |
//! | `latex`, `mathml` | `odf`, `pdf` |
//!
//! Any existing document converts the same way, by its extension, which is
//! how a document becomes a PDF.

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Maps a format name or file extension to the name used here.
pub fn canonical_format(format: &str) -> String {
    let format = format.trim().trim_start_matches('.').to_ascii_lowercase();
    match format.as_str() {
        "markdown" => "md".to_string(),
        "htm" => "html".to_string(),
        "text" => "txt".to_string(),
        "mml" => "mathml".to_string(),
        "tex" => "latex".to_string(),
        _ => format,
    }
}

/// The format a path's extension names, if it has one.
pub fn format_of(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(canonical_format)
}

/// Renders or converts `input`, in format `from`, to format `to`.
///
/// `base` is the directory relative image references in Markdown resolve
/// against when it is rendered straight to PDF.
pub fn convert(input: &[u8], from: &str, to: &str, base: &Path) -> Result<Vec<u8>> {
    let from = canonical_format(from);
    let to = canonical_format(to);
    if from == to {
        bail!("source and target are both '{to}'");
    }
    // An HTML comment never renders in Markdown, but the importer prints it
    // as text, and new `.md` files carry their licence header in one.
    let stripped;
    let input = if from == "md" {
        let markdown = std::str::from_utf8(input).context("Markdown is not UTF-8")?;
        stripped = strip_html_comments(markdown);
        stripped.as_bytes()
    } else {
        input
    };
    let result = match (from.as_str(), to.as_str()) {
        // A slide outline is plain text to the importer, but only the
        // presentation family reads it as slides.
        ("outline", _) => libreoffice_pure::impress_convert_bytes(input, "txt", &to),
        // CSV is a spreadsheet unless a database was asked for.
        ("csv", "odb") => libreoffice_pure::base_convert_bytes(input, "csv", "odb"),
        // The richer Markdown renderer: Mermaid, math, embedded images.
        ("md", "pdf") => {
            let markdown = std::str::from_utf8(input).context("Markdown is not UTF-8")?;
            return markdown_to_pdf(markdown, base);
        }
        _ => libreoffice_pure::convert_bytes(input, &from, &to),
    };
    result.map_err(|err| anyhow!("{from} to {to}: {err}"))
}

/// `markdown` without its HTML comments (`<!-- … -->`), except inside fenced
/// code blocks and inline code spans, where they are literal text.
pub fn strip_html_comments(markdown: &str) -> String {
    let mut out = String::with_capacity(markdown.len());
    let mut fence: Option<&str> = None;
    let mut in_comment = false;
    for line in markdown.split_inclusive('\n') {
        let trimmed = line.trim_start();
        if !in_comment {
            if let Some(marker) = fence {
                if trimmed.starts_with(marker) {
                    fence = None;
                }
                out.push_str(line);
                continue;
            }
            if let Some(marker) = ["```", "~~~"].into_iter().find(|m| trimmed.starts_with(m)) {
                fence = Some(marker);
                out.push_str(line);
                continue;
            }
        }
        let mut rest = line;
        let mut in_code_span = false;
        while !rest.is_empty() {
            if in_comment {
                match rest.find("-->") {
                    Some(end) => {
                        rest = &rest[end + 3..];
                        in_comment = false;
                    }
                    None => rest = "",
                }
                continue;
            }
            let open = rest.find("<!--");
            let tick = rest.find('`');
            match (open, tick) {
                (Some(open), Some(tick)) if tick < open || in_code_span => {
                    out.push_str(&rest[..=tick]);
                    rest = &rest[tick + 1..];
                    in_code_span = !in_code_span;
                }
                (Some(open), _) if !in_code_span => {
                    out.push_str(&rest[..open]);
                    rest = &rest[open + 4..];
                    in_comment = true;
                }
                _ => {
                    out.push_str(rest);
                    rest = "";
                }
            }
        }
    }
    out
}

fn markdown_to_pdf(markdown: &str, base: &Path) -> Result<Vec<u8>> {
    // `markdown_to_pdf_bytes` takes the Markdown file's own path and resolves
    // images next to it; any file name in `base` gives the same base.
    libreoffice_pure::markdown_to_pdf_bytes(base.join("document.md"), markdown.as_bytes())
        .map_err(|err| anyhow!("md to pdf: {err}"))
}

/// The `convert_document` tool's request.
#[derive(Debug, Deserialize)]
pub struct ConvertRequest {
    /// Where the document is written; its extension is the target format.
    pub target: String,
    /// An existing workspace file to convert.
    pub source: Option<String>,
    /// Text to render, instead of `source`.
    pub content: Option<String>,
    /// The format of `content`, or overrides `source`'s extension.
    pub format: Option<String>,
    #[serde(default = "crate::files::overwrite_default")]
    pub overwrite: bool,
}

#[derive(Debug, Serialize)]
pub struct ConvertResponse {
    pub target: String,
    pub format: String,
    pub bytes_written: usize,
    pub overwritten: bool,
}

/// Runs one request against `workspace`, confining `source` and `target` to
/// it first: what the `convert_document` tool and `/convert` both do.
pub fn convert_in_workspace(workspace: &Path, request: &ConvertRequest) -> Result<ConvertResponse> {
    let target = crate::tools::resolve_workspace_path(workspace, &request.target)?;
    let source = request
        .source
        .as_deref()
        .map(|source| crate::tools::resolve_workspace_path(workspace, source))
        .transpose()?;
    convert_file(workspace, request, source.as_deref(), &target)
}

/// The target `/convert <source> <target>` writes. A `target` that is only
/// a format (`pdf`, `.odt`) means the source's path with that extension.
pub fn command_target(source: &str, target: &str) -> String {
    let bare_format = !target.contains(['/', '\\'])
        && target
            .trim_start_matches('.')
            .chars()
            .all(|c| c.is_ascii_alphanumeric())
        && !target.trim_start_matches('.').is_empty()
        && (target.starts_with('.') || !target.contains('.'));
    if bare_format {
        Path::new(source)
            .with_extension(target.trim_start_matches('.'))
            .to_string_lossy()
            .into_owned()
    } else {
        target.to_string()
    }
}

/// Runs one `convert_document` request against `workspace`. `source` and
/// `target` are already confined to it by the caller.
pub fn convert_file(
    workspace: &Path,
    request: &ConvertRequest,
    source: Option<&Path>,
    target: &Path,
) -> Result<ConvertResponse> {
    let to = format_of(target)
        .with_context(|| format!("target '{}' has no extension", request.target))?;
    let (input, from) =
        match (source, &request.content) {
            (Some(_), Some(_)) => bail!("give either 'source' or 'content', not both"),
            (None, None) => bail!("give 'source' or 'content'"),
            (Some(source), None) => {
                let from = match &request.format {
                    Some(format) => canonical_format(format),
                    None => format_of(source)
                        .with_context(|| "source has no extension; pass 'format'")?,
                };
                let input = std::fs::read(source)
                    .with_context(|| format!("failed to read {}", source.display()))?;
                (input, from)
            }
            (None, Some(content)) => {
                let from =
                    request.format.as_deref().map(canonical_format).context(
                        "'content' needs a 'format' (md, html, csv, outline, svg, latex)",
                    )?;
                (content.clone().into_bytes(), from)
            }
        };

    let existed = target.exists();
    if existed && !request.overwrite {
        bail!(
            "{} already exists (the request passed \"overwrite\": false)",
            request.target
        );
    }
    let base = source
        .and_then(Path::parent)
        .or_else(|| target.parent())
        .unwrap_or(workspace);
    let bytes = convert(&input, &from, &to, base)?;

    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    std::fs::write(target, &bytes)
        .with_context(|| format!("failed to write {}", target.display()))?;

    Ok(ConvertResponse {
        target: target
            .strip_prefix(workspace)
            .unwrap_or(target)
            .to_string_lossy()
            .replace('\\', "/"),
        format: to,
        bytes_written: bytes.len(),
        overwritten: existed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_zip_with_mimetype(bytes: &[u8], mimetype: &str) -> bool {
        bytes.starts_with(b"PK\x03\x04")
            && bytes
                .windows(mimetype.len())
                .any(|window| window == mimetype.as_bytes())
    }

    fn is_pdf(bytes: &[u8]) -> bool {
        bytes.starts_with(b"%PDF-")
    }

    fn render(content: &str, from: &str, to: &str) -> Vec<u8> {
        convert(content.as_bytes(), from, to, Path::new(".")).unwrap()
    }

    #[test]
    fn html_comments_are_stripped_from_markdown_outside_code() {
        let markdown = "<!--\nSPDX licence\n-->\n# Title\n\nText <!-- note --> here.\n\n\
                        Inline `<!-- kept -->` code.\n\n```html\n<!-- kept too -->\n```\n";
        let stripped = strip_html_comments(markdown);
        assert!(!stripped.contains("SPDX"), "{stripped}");
        assert!(!stripped.contains("note"), "{stripped}");
        assert!(stripped.contains("# Title"), "{stripped}");
        assert!(stripped.contains("Text  here."), "{stripped}");
        assert!(stripped.contains("`<!-- kept -->`"), "{stripped}");
        assert!(stripped.contains("<!-- kept too -->"), "{stripped}");

        // And the licence header no longer reaches the document.
        let pdf = render(markdown, "md", "pdf");
        let text = libreoffice_pure::pdf_to_txt_bytes(&pdf).unwrap();
        assert!(!String::from_utf8_lossy(&text).contains("SPDX"));
    }

    #[test]
    fn a_bare_format_targets_the_source_with_that_extension() {
        assert_eq!(command_target("doc/report.md", "pdf"), "doc/report.pdf");
        assert_eq!(command_target("report.md", ".ODT"), "report.ODT");
        assert_eq!(
            command_target("report.md", "out/report.pdf"),
            "out/report.pdf"
        );
        assert_eq!(command_target("report.md", "final.pdf"), "final.pdf");
        assert_eq!(command_target("slides.odp", "out/pdf"), "out/pdf");
    }

    #[test]
    fn convert_in_workspace_confines_both_paths() {
        let workspace = tempfile::tempdir().unwrap();
        std::fs::write(workspace.path().join("a.md"), "# A\n").unwrap();
        let request = |source: &str, target: &str| ConvertRequest {
            target: target.to_string(),
            source: Some(source.to_string()),
            content: None,
            format: None,
            overwrite: true,
        };
        let response = convert_in_workspace(workspace.path(), &request("a.md", "a.pdf")).unwrap();
        assert_eq!(response.target, "a.pdf");
        assert!(convert_in_workspace(workspace.path(), &request("a.md", "../a.pdf")).is_err());
        assert!(
            convert_in_workspace(workspace.path(), &request("/etc/hosts.md", "b.pdf")).is_err()
        );
    }

    #[test]
    fn canonical_format_folds_aliases_and_dots() {
        assert_eq!(canonical_format(".Markdown"), "md");
        assert_eq!(canonical_format("ODT"), "odt");
        assert_eq!(canonical_format("tex"), "latex");
        assert_eq!(format_of(Path::new("a/b.PPTX")).as_deref(), Some("pptx"));
        assert_eq!(format_of(Path::new("README")), None);
    }

    #[test]
    fn markdown_renders_to_a_text_document_and_pdf() {
        let markdown = "# Report\n\nSome *text*.\n\n- one\n- two\n";
        let odt = render(markdown, "md", "odt");
        assert!(is_zip_with_mimetype(
            &odt,
            "application/vnd.oasis.opendocument.text"
        ));
        assert!(is_pdf(&render(markdown, "md", "pdf")));
        assert!(render(markdown, "md", "docx").starts_with(b"PK\x03\x04"));
    }

    #[test]
    fn csv_renders_to_a_spreadsheet_with_formulas() {
        let csv = "Item,Cost\nA,2\nB,3\nTotal,=SUM(B2:B3)\n";
        let ods = render(csv, "csv", "ods");
        assert!(is_zip_with_mimetype(
            &ods,
            "application/vnd.oasis.opendocument.spreadsheet"
        ));
        assert!(is_pdf(&render(csv, "csv", "pdf")));
        assert!(render(csv, "csv", "xlsx").starts_with(b"PK\x03\x04"));
    }

    #[test]
    fn an_outline_renders_to_a_presentation() {
        let outline = "# Welcome\n- first point\n- second point\n# Next\n- more\n";
        let odp = render(outline, "outline", "odp");
        assert!(is_zip_with_mimetype(
            &odp,
            "application/vnd.oasis.opendocument.presentation"
        ));
        assert!(is_pdf(&render(outline, "outline", "pdf")));
        // Slides and their bullets are in the package. (Reading an `.odp`
        // back drops list text, so this looks at the XML, not a round trip.)
        let mut content = String::new();
        std::io::Read::read_to_string(
            &mut zip::ZipArchive::new(std::io::Cursor::new(&odp))
                .unwrap()
                .by_name("content.xml")
                .unwrap(),
            &mut content,
        )
        .unwrap();
        assert!(content.contains(r#"draw:name="Welcome""#), "{content}");
        assert!(
            content.contains("<text:p>second point</text:p>"),
            "{content}"
        );
        assert!(content.contains(r#"draw:name="Next""#), "{content}");
    }

    #[test]
    fn svg_renders_to_a_drawing() {
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100"><rect x="10" y="10" width="80" height="40" fill="blue"/><text x="20" y="80">Hi</text></svg>"#;
        let odg = render(svg, "svg", "odg");
        assert!(is_zip_with_mimetype(
            &odg,
            "application/vnd.oasis.opendocument.graphics"
        ));
        assert!(is_pdf(&render(svg, "svg", "pdf")));
    }

    #[test]
    fn a_document_converts_to_pdf() {
        let odt = render("# Title\n\nBody.\n", "md", "odt");
        assert!(is_pdf(
            &convert(&odt, "odt", "pdf", Path::new(".")).unwrap()
        ));
    }

    #[test]
    fn unsupported_conversions_are_errors() {
        assert!(convert(b"x", "md", "md", Path::new(".")).is_err());
        assert!(convert(b"x", "bogus", "pdf", Path::new(".")).is_err());
    }

    #[test]
    fn convert_file_writes_the_target_and_refuses_to_overwrite_when_asked() {
        let workspace = tempfile::tempdir().unwrap();
        let target = workspace.path().join("out/report.odt");
        let request = ConvertRequest {
            target: "out/report.odt".to_string(),
            source: None,
            content: Some("# Hello\n".to_string()),
            format: Some("markdown".to_string()),
            overwrite: false,
        };
        let response = convert_file(workspace.path(), &request, None, &target).unwrap();
        assert_eq!(response.target, "out/report.odt");
        assert_eq!(response.format, "odt");
        assert!(!response.overwritten);
        assert_eq!(
            std::fs::metadata(&target).unwrap().len() as usize,
            response.bytes_written
        );

        let err = convert_file(workspace.path(), &request, None, &target).unwrap_err();
        assert!(err.to_string().contains("already exists"), "{err}");

        // The document it wrote converts on to PDF.
        let pdf = workspace.path().join("out/report.pdf");
        let to_pdf = ConvertRequest {
            target: "out/report.pdf".to_string(),
            source: Some("out/report.odt".to_string()),
            content: None,
            format: None,
            overwrite: true,
        };
        convert_file(workspace.path(), &to_pdf, Some(&target), &pdf).unwrap();
        assert!(is_pdf(&std::fs::read(&pdf).unwrap()));
    }

    #[test]
    fn convert_file_needs_exactly_one_input() {
        let workspace = tempfile::tempdir().unwrap();
        let target = workspace.path().join("a.odt");
        let mut request = ConvertRequest {
            target: "a.odt".to_string(),
            source: None,
            content: None,
            format: Some("md".to_string()),
            overwrite: true,
        };
        assert!(convert_file(workspace.path(), &request, None, &target).is_err());
        request.content = Some("# x".to_string());
        let source = workspace.path().join("a.md");
        std::fs::write(&source, "# x").unwrap();
        assert!(convert_file(workspace.path(), &request, Some(&source), &target).is_err());
        let content_without_format = ConvertRequest {
            target: "a.odt".to_string(),
            source: None,
            content: Some("# x".to_string()),
            format: None,
            overwrite: true,
        };
        let err =
            convert_file(workspace.path(), &content_without_format, None, &target).unwrap_err();
        assert!(err.to_string().contains("format"), "{err}");
    }
}
