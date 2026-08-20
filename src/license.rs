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

//! The licence header generated code is written with.
//!
//! Two surfaces produce code and both go through here: `orangu`'s
//! `create_file` tool, which writes a new file into the workspace, and the
//! web console's per-block download, which saves one code block out of a
//! reply. One module so they cannot disagree about the year, the licence,
//! or where the header goes.
//!
//! The text is [`TEMPLATE`], a string constant in this file — not a data
//! file read from disk, and not an `include_str!` of one either. There is
//! nothing to ship beside the binary, nothing to go missing from an
//! install, and nothing a running server can be pointed at; editing that
//! constant is the whole of "use a different licence". Placeholders in it
//! are filled in at write time ([`PLACEHOLDERS`]), which is what keeps
//! `<YEAR>` right in a process left running across New Year's Eve rather
//! than frozen at whatever it was when the binary was built.
//!
//! The header is wrapped in the file's own comment syntax
//! ([`CommentStyle`]), because a licence header that isn't a comment is a
//! syntax error. An extension whose comment syntax isn't known produces
//! **no header** rather than a guessed one: a `#` at the top of a JSON
//! file, or a `%` at the top of Objective-C, doesn't produce a
//! differently-licensed file — it produces a broken one.

/// The licence text, with placeholders still in it.
///
/// A raw string rather than a file: `"Software"` appears throughout, so the
/// literal has to be `r#"..."#`, and there is no `#` anywhere in the licence
/// for the delimiter to collide with.
const TEMPLATE: &str = r#"MIT License

Copyright (c) <YEAR> orangu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"#;

/// What each placeholder in [`TEMPLATE`] is replaced with, in order.
///
/// A table rather than a chain of `.replace()` calls so that adding one is
/// a line here and a token in [`TEMPLATE`], with no code to touch.
///
/// Only values that *can't* be static belong here. The copyright holder is
/// written into [`TEMPLATE`] directly, because it is one — routing a constant
/// through a `fn() -> String` would buy nothing and would stop the file
/// reading as the licence it actually emits. The year can't be: a server
/// left running past midnight on 31 December would otherwise keep stamping
/// last year onto everything it saves.
type Placeholder = (&'static str, fn() -> String);

const PLACEHOLDERS: [Placeholder; 1] = [("<YEAR>", || current_year().to_string())];

/// The current UTC calendar year — computed from the Unix clock rather than
/// pulling in a full date/time crate for one integer.
///
/// Public because the web console's page footer wants the same number, and
/// two implementations of "what year is it" is one too many.
pub fn current_year() -> i64 {
    let mut days = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0)
        / 86400;
    let mut year = 1970i64;
    loop {
        let is_leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
        let days_in_year = if is_leap { 366 } else { 365 };
        if days < days_in_year {
            return year;
        }
        days -= days_in_year;
        year += 1;
    }
}

/// How a language spells a comment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommentStyle {
    /// One marker per line: `// text`, `# text`. A blank line in the
    /// licence becomes a bare marker rather than a marker plus a trailing
    /// space, which some formatters and linters strip back out again.
    Line(&'static str),
    /// A delimited block: an opening line, a marker on each line between,
    /// and a closing line — `/*`, ` *`, ` */`. `line` carries no trailing
    /// space; the one separating it from the text is added per line, so a
    /// blank licence line doesn't end up as trailing whitespace.
    Block {
        open: &'static str,
        line: &'static str,
        close: &'static str,
    },
}

/// The comment syntax for a file, or `None` when it isn't known.
///
/// Keyed off the saved file's extension rather than the fence's language
/// tag, because the extension is what the file on disk will actually be
/// read as. Extension-less names that are nonetheless a known file get
/// matched whole, mirroring the web console's own bare-name list.
///
/// Genuinely ambiguous extensions are absent on purpose. `.m` is
/// Objective-C (`//`) about as often as it is MATLAB (`%`), and `.s` is
/// assembly whose comment marker depends on the assembler — those save
/// without a header instead of with a wrong one.
fn comment_style(file_name: &str) -> Option<CommentStyle> {
    use CommentStyle::{Block, Line};

    let lower = file_name.to_ascii_lowercase();
    match lower.as_str() {
        "makefile" | "dockerfile" | "rakefile" | "gemfile" | "justfile" | "vagrantfile"
        | "procfile" => return Some(Line("#")),
        "jenkinsfile" => return Some(Line("//")),
        _ => {}
    }

    // `.gitignore` and friends split into an empty stem and the whole name
    // as the "extension", which is exactly the key wanted here.
    let extension = lower.rsplit_once('.')?.1;
    Some(match extension {
        // C family and everything that borrowed its comment.
        "c" | "cc" | "cpp" | "cxx" | "c++" | "h" | "hh" | "hpp" | "hxx" | "cu" | "cuh" | "cs"
        | "d" | "dart" | "go" | "java" | "js" | "jsx" | "mjs" | "cjs" | "kt" | "kts" | "rs"
        | "scala" | "sc" | "swift" | "ts" | "tsx" | "mts" | "cts" | "php" | "zig" | "proto"
        | "gradle" | "groovy" | "less" | "sass" | "scss" | "v" | "sv" | "svh" | "wgsl" | "glsl"
        | "vert" | "frag" | "comp" | "hlsl" | "metal" | "rego" | "jsonc" | "json5" | "fs"
        | "fsi" | "fsx" => Line("//"),
        // Shells, scripting languages, and the configuration formats that
        // took `#` from them.
        "sh" | "bash" | "zsh" | "ksh" | "fish" | "py" | "pyi" | "pyw" | "rb" | "pl" | "pm"
        | "t" | "r" | "jl" | "nim" | "cr" | "ex" | "exs" | "eex" | "yaml" | "yml" | "toml"
        | "ini" | "cfg" | "conf" | "properties" | "env" | "gitignore" | "gitattributes"
        | "dockerignore" | "editorconfig" | "tf" | "tfvars" | "hcl" | "mk" | "make" | "cmake"
        | "ps1" | "psm1" | "psd1" | "awk" | "tcl" | "coffee" | "nix" | "pp" | "service"
        | "desktop" | "gemspec" | "podspec" | "rake" | "graphql" | "gql" => Line("#"),
        // The `--` family.
        "sql" | "lua" | "hs" | "lhs" | "elm" | "purs" | "ada" | "adb" | "ads" | "vhd" | "vhdl"
        | "applescript" | "moon" => Line("--"),
        // Lisps.
        "lisp" | "lsp" | "cl" | "el" | "clj" | "cljs" | "cljc" | "edn" | "scm" | "ss" | "rkt"
        | "fnl" => Line(";;"),
        // TeX and Erlang both took `%`.
        "tex" | "latex" | "sty" | "cls" | "bib" | "dtx" | "erl" | "hrl" | "escript" => Line("%"),
        "bat" | "cmd" => Line("REM"),
        "f" | "for" | "f77" | "f90" | "f95" | "f03" | "f08" => Line("!"),
        "vb" | "vbs" | "bas" | "frm" => Line("'"),
        // Markup, where a line comment doesn't exist at all.
        "html" | "htm" | "xhtml" | "xml" | "xsl" | "xslt" | "svg" | "vue" | "svelte" | "md"
        | "markdown" | "plist" | "xaml" | "resx" | "wxs" => Block {
            open: "<!--",
            line: "",
            close: "-->",
        },
        "css" => Block {
            open: "/*",
            line: " *",
            close: " */",
        },
        "ml" | "mli" | "pas" | "dpr" | "sml" => Block {
            open: "(*",
            line: " *",
            close: " *)",
        },
        _ => return None,
    })
}

/// The finished licence header for a file, comment markers and all, ending
/// in the blank line that separates it from the code — or `None` when the
/// file's comment syntax isn't known.
pub fn header_for(file_name: &str) -> Option<String> {
    let style = comment_style(file_name)?;
    let mut text = TEMPLATE.to_string();
    for (placeholder, value) in PLACEHOLDERS {
        if text.contains(placeholder) {
            text = text.replace(placeholder, &value());
        }
    }

    let (open, marker, close) = match style {
        CommentStyle::Line(marker) => (None, marker, None),
        CommentStyle::Block { open, line, close } => (Some(open), line, Some(close)),
    };

    let mut header = String::with_capacity(text.len() + 128);
    let mut push = |line: &str| {
        header.push_str(line);
        header.push('\n');
    };

    if let Some(open) = open {
        push(open);
    }
    for line in text.lines() {
        match (marker.is_empty(), line.is_empty()) {
            // A block comment whose body carries no marker of its own.
            (true, _) => push(line),
            // No trailing space on an otherwise empty comment line.
            (false, true) => push(marker),
            (false, false) => push(&format!("{marker} {line}")),
        }
    }
    if let Some(close) = close {
        push(close);
    }
    header.push('\n');
    Some(header)
}

/// The first line of [`TEMPLATE`], used to recognise a licence that is
/// already there. Taken from the template rather than hard-coded, so
/// replacing [`TEMPLATE`] with a different licence keeps the check honest.
fn licence_marker() -> &'static str {
    TEMPLATE
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("MIT License")
}

/// True when `content` already carries this licence, whoever put it there —
/// a model that wrote its own header, or a file being rewritten that had one
/// already. Prepending a second copy is the failure this prevents.
pub fn already_present(content: &str) -> bool {
    content.contains(licence_marker())
}

/// `content` with the licence header on top, ready to write as `file_name`.
///
/// Returns it unchanged when the file's comment syntax isn't known or the
/// licence is already there.
///
/// A shebang and an XML declaration keep the first line — a licence above
/// either one leaves a script the kernel won't exec and a document no
/// parser will accept — so the header goes directly beneath them. This is
/// the one implementation of that rule; the web console reaches it over
/// `POST /api/license-header` rather than repeating it in JavaScript.
pub fn apply(content: &str, file_name: &str) -> String {
    if already_present(content) {
        return content.to_string();
    }
    let Some(header) = header_for(file_name) else {
        return content.to_string();
    };

    let preamble = first_line_preamble(content);
    let (kept, rest) = content.split_at(preamble);
    format!("{kept}{header}{rest}")
}

/// The byte length of a leading shebang or XML declaration line, including
/// its newline — `0` when there is neither.
fn first_line_preamble(content: &str) -> usize {
    let Some(line_end) = content.find('\n') else {
        return 0;
    };
    let first = &content[..line_end];
    let is_shebang = first.starts_with("#!");
    let is_xml_prolog = first.trim_start().starts_with("<?xml") && first.trim_end().ends_with("?>");
    if is_shebang || is_xml_prolog {
        line_end + 1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The header has to *be* a comment — the whole point of the exercise.
    #[test]
    fn wraps_the_licence_in_each_line_comment_style() {
        let cases = [
            ("main.rs", "// MIT License"),
            ("app.py", "# MIT License"),
            ("schema.sql", "-- MIT License"),
            ("core.clj", ";; MIT License"),
            ("paper.tex", "% MIT License"),
            ("setup.bat", "REM MIT License"),
            ("solver.f90", "! MIT License"),
            ("Form1.vb", "' MIT License"),
            ("Makefile", "# MIT License"),
            (".gitignore", "# MIT License"),
        ];
        for (file, expected) in cases {
            let header = header_for(file).unwrap_or_else(|| panic!("no header for {file}"));
            assert!(header.starts_with(expected), "{file}: {header}");
        }
    }

    /// A markup file has no line comment, so the licence is delimited.
    #[test]
    fn wraps_the_licence_in_block_comments_where_there_is_no_line_comment() {
        let html = header_for("index.html").expect("html header");
        assert!(html.starts_with("<!--\nMIT License\n"), "{html}");
        assert!(html.trim_end().ends_with("-->"), "{html}");

        let css = header_for("style.css").expect("css header");
        assert!(css.starts_with("/*\n * MIT License\n"), "{css}");
        assert!(css.trim_end().ends_with(" */"), "{css}");

        let ml = header_for("parser.ml").expect("ml header");
        assert!(ml.starts_with("(*\n * MIT License\n"), "{ml}");
        assert!(ml.trim_end().ends_with(" *)"), "{ml}");
    }

    /// A guessed comment marker doesn't relicense a file, it breaks it.
    #[test]
    fn refuses_to_guess_an_unknown_comment_syntax() {
        for file in [
            "data.json",
            "rows.csv",
            "orangu-snippet-1.txt",
            "thing.m",
            "boot.s",
            "whatever",
        ] {
            assert!(header_for(file).is_none(), "{file} should get no header");
        }
    }

    /// The year is filled in at download time, not baked in at build time.
    #[test]
    fn substitutes_the_year() {
        let header = header_for("main.rs").expect("header");
        assert!(!header.contains("<YEAR>"), "{header}");
        assert!(
            header.contains(&format!("Copyright (c) {} orangu", current_year())),
            "{header}"
        );
    }

    /// Catches a placeholder added to [`TEMPLATE`] and never wired into
    /// [`PLACEHOLDERS`], which would otherwise ship `<SOMETHING>` at the top
    /// of every file someone saves. A line-comment header has no angle
    /// bracket of its own, so the absence of one is the whole check.
    #[test]
    fn no_placeholder_survives_into_a_saved_file() {
        let header = header_for("main.rs").expect("header");
        assert!(!header.contains('<'), "unsubstituted placeholder: {header}");
    }

    /// A blank licence line stays blank rather than becoming `"// "`, which
    /// editors and linters strip anyway.
    #[test]
    fn blank_lines_carry_a_bare_marker() {
        let header = header_for("main.rs").expect("header");
        assert!(header.contains("\n//\n"), "{header}");
        assert!(!header.contains("// \n"), "{header}");
    }

    /// It has to end separated from the code, or the first line of the
    /// snippet lands against the licence.
    #[test]
    fn ends_with_a_blank_line() {
        assert!(header_for("main.rs").expect("header").ends_with("\n\n"));
        assert!(header_for("index.html").expect("header").ends_with("\n\n"));
    }

    /// A block comment that contains its own terminator ends early and
    /// spills the rest of the licence into the file as code.
    #[test]
    fn the_licence_text_cannot_close_a_block_comment_early() {
        assert!(!TEMPLATE.contains("-->"), "would end an HTML comment");
        assert!(!TEMPLATE.contains("*/"), "would end a C comment");
        assert!(!TEMPLATE.contains("*)"), "would end an OCaml comment");
    }

    /// The whole job, as `create_file` and the console's download both use
    /// it: header on top, code below, one blank line between.
    #[test]
    fn apply_puts_the_header_above_the_code() {
        let out = apply("fn main() {}\n", "main.rs");
        assert!(out.starts_with("// MIT License\n"), "{out}");
        assert!(out.ends_with("\nfn main() {}\n"), "{out}");
    }

    /// A shebang stops being a shebang the moment anything precedes it, and
    /// an XML declaration stops being well-formed.
    #[test]
    fn apply_keeps_a_shebang_or_xml_declaration_on_the_first_line() {
        let script = apply("#!/usr/bin/env bash\nset -e\n", "deploy.sh");
        assert!(
            script.starts_with("#!/usr/bin/env bash\n# MIT License\n"),
            "{script}"
        );
        assert!(script.ends_with("\nset -e\n"), "{script}");

        let xml = apply("<?xml version=\"1.0\"?>\n<rss/>\n", "feed.xml");
        assert!(xml.starts_with("<?xml version=\"1.0\"?>\n<!--\n"), "{xml}");

        // Not every first line is a preamble.
        let plain = apply("<rss/>\n", "feed.xml");
        assert!(plain.starts_with("<!--\n"), "{plain}");
    }

    /// Writing the licence twice is the failure a model emitting its own
    /// header would otherwise cause, and the one a rewrite would.
    #[test]
    fn apply_does_not_stack_a_second_licence() {
        let once = apply("fn main() {}\n", "main.rs");
        assert_eq!(apply(&once, "main.rs"), once);
        assert!(already_present(&once));
        assert!(!already_present("fn main() {}\n"));
    }

    /// A format with nowhere to put a comment comes back untouched, so a
    /// generated `.json` is still parseable.
    #[test]
    fn apply_leaves_a_file_it_cannot_comment_alone() {
        let json = "{\"a\": 1}\n";
        assert_eq!(apply(json, "data.json"), json);
        assert_eq!(apply(json, "no-extension"), json);
    }

    /// Empty content is what `create_file` sends for an empty file; it must
    /// not panic on the missing newline.
    #[test]
    fn apply_handles_content_with_no_newline() {
        assert!(apply("", "main.rs").starts_with("// MIT License"));
        assert!(apply("x = 1", "app.py").ends_with("\nx = 1"));
    }

    /// Extensions are matched case-insensitively — a fence naming
    /// `Main.RS` still gets Rust's comment.
    #[test]
    fn matches_the_extension_case_insensitively() {
        assert!(header_for("Main.RS").expect("header").starts_with("// "));
        assert!(header_for("MAKEFILE").expect("header").starts_with("# "));
    }
}
