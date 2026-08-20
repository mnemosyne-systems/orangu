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
//! **The licence is the project's, not this program's.** orangu writes files
//! into somebody else's repository, so the header has to be the one that
//! repository already uses — read from its `Cargo.toml`/`package.json`/
//! `pyproject.toml` `license` field, or sniffed from its `LICENSE`/`COPYING`
//! file. This used to be a single hard-coded MIT constant, which meant every
//! file generated into a GPL project (orangu's own repository among them)
//! arrived carrying an MIT header attributed to "orangu": false licensing
//! metadata, `git add`ed, on somebody else's code. `files::create` already
//! refused to put a header on an *existing* file for exactly that reason —
//! "a project's licensing is not this tool's to decide" — and a file being
//! created is no more this tool's to decide.
//!
//! What cannot be determined produces **no header at all**. A project whose
//! licence is unrecognised, dual (`MIT OR Apache-2.0` — which of the two
//! would the header be?), or absent gets none; so does one that names no
//! copyright holder, because the alternative is inventing an attribution.
//! Nothing is a strictly better answer than something wrong here, and the
//! caller is told which happened ([`CreateFileResponse::licensed`]).
//!
//! The texts are constants in this file — not data files read from disk, and
//! not `include_str!` of any. There is nothing to ship beside the binary and
//! nothing a running server can be pointed at. Placeholders in them are
//! filled in at write time ([`PLACEHOLDERS`] and the holder), which is what
//! keeps `<YEAR>` right in a process left running across New Year's Eve
//! rather than frozen at whatever it was when the binary was built.
//!
//! The header is wrapped in the file's own comment syntax
//! ([`CommentStyle`]), because a licence header that isn't a comment is a
//! syntax error. An extension whose comment syntax isn't known produces
//! **no header** rather than a guessed one: a `#` at the top of a JSON
//! file, or a `%` at the top of Objective-C, doesn't produce a
//! differently-licensed file — it produces a broken one.

use std::path::Path;

/// A licence this program knows how to write a header for.
///
/// The licences on <https://opensource.org/license>'s popular list, with the
/// GNU family split the way SPDX splits it — `-only` and `-or-later` are
/// different grants, and a header has to say which. Deliberately not every
/// OSI-approved licence: each entry needs a *per-file header* that is well
/// established, because inventing one is how a file ends up claiming terms
/// nobody chose.
///
/// Ordered by how often a project is under it, because this order is what
/// `/license`'s completion offers and what its ghost cycles through — see
/// [`Licence::ALL`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Licence {
    Mit,
    Apache2,
    Bsd3Clause,
    Bsd2Clause,
    Gpl3OrLater,
    Gpl3Only,
    Gpl2OrLater,
    Gpl2Only,
    Lgpl3OrLater,
    Lgpl3Only,
    Lgpl21OrLater,
    Lgpl21Only,
    Agpl3OrLater,
    Agpl3Only,
    Mpl2,
    Epl2,
    Bsl1,
    Cddl1,
}

impl Licence {
    /// Every licence, in the order `/license` offers them.
    pub const ALL: [Licence; 18] = [
        Licence::Mit,
        Licence::Apache2,
        Licence::Bsd3Clause,
        Licence::Bsd2Clause,
        Licence::Gpl3OrLater,
        Licence::Gpl3Only,
        Licence::Gpl2OrLater,
        Licence::Gpl2Only,
        Licence::Lgpl3OrLater,
        Licence::Lgpl3Only,
        Licence::Lgpl21OrLater,
        Licence::Lgpl21Only,
        Licence::Agpl3OrLater,
        Licence::Agpl3Only,
        Licence::Mpl2,
        Licence::Epl2,
        Licence::Bsl1,
        Licence::Cddl1,
    ];

    /// The SPDX identifier — what a manifest declares, what `/license` takes,
    /// and what a session stores.
    pub fn spdx(self) -> &'static str {
        match self {
            Licence::Mit => "MIT",
            Licence::Apache2 => "Apache-2.0",
            Licence::Bsd3Clause => "BSD-3-Clause",
            Licence::Bsd2Clause => "BSD-2-Clause",
            Licence::Gpl3OrLater => "GPL-3.0-or-later",
            Licence::Gpl3Only => "GPL-3.0-only",
            Licence::Gpl2OrLater => "GPL-2.0-or-later",
            Licence::Gpl2Only => "GPL-2.0-only",
            Licence::Lgpl3OrLater => "LGPL-3.0-or-later",
            Licence::Lgpl3Only => "LGPL-3.0-only",
            Licence::Lgpl21OrLater => "LGPL-2.1-or-later",
            Licence::Lgpl21Only => "LGPL-2.1-only",
            Licence::Agpl3OrLater => "AGPL-3.0-or-later",
            Licence::Agpl3Only => "AGPL-3.0-only",
            Licence::Mpl2 => "MPL-2.0",
            Licence::Epl2 => "EPL-2.0",
            Licence::Bsl1 => "BSL-1.0",
            Licence::Cddl1 => "CDDL-1.0",
        }
    }

    /// The licence's own name, for a line of prose about it.
    pub fn title(self) -> &'static str {
        match self {
            Licence::Mit => "MIT License",
            Licence::Apache2 => "Apache License 2.0",
            Licence::Bsd3Clause => "BSD 3-Clause License",
            Licence::Bsd2Clause => "BSD 2-Clause License",
            Licence::Gpl3OrLater => "GNU General Public License v3.0 or later",
            Licence::Gpl3Only => "GNU General Public License v3.0 only",
            Licence::Gpl2OrLater => "GNU General Public License v2.0 or later",
            Licence::Gpl2Only => "GNU General Public License v2.0 only",
            Licence::Lgpl3OrLater => "GNU Lesser General Public License v3.0 or later",
            Licence::Lgpl3Only => "GNU Lesser General Public License v3.0 only",
            Licence::Lgpl21OrLater => "GNU Lesser General Public License v2.1 or later",
            Licence::Lgpl21Only => "GNU Lesser General Public License v2.1 only",
            Licence::Agpl3OrLater => "GNU Affero General Public License v3.0 or later",
            Licence::Agpl3Only => "GNU Affero General Public License v3.0 only",
            Licence::Mpl2 => "Mozilla Public License 2.0",
            Licence::Epl2 => "Eclipse Public License 2.0",
            Licence::Bsl1 => "Boost Software License 1.0",
            Licence::Cddl1 => "Common Development and Distribution License 1.0",
        }
    }

    /// The header text, with `<YEAR>` and `<HOLDER>` still in it.
    ///
    /// For the short licences that is the whole text, which is how those are
    /// conventionally carried per file. For the GNU family, Apache, Mozilla
    /// and Eclipse it is the short notice each licence's own text tells you
    /// to use — the full terms live in the project's `LICENSE`, and a copy of
    /// the GPL on top of every source file is not what "match the project"
    /// means.
    ///
    /// A `String` rather than a `&'static str` because the ten GNU variants
    /// differ only in a name and a version clause, and ten near-identical
    /// constants is ten places for one of them to be wrong.
    pub fn header_text(self) -> String {
        match self {
            Licence::Mit => MIT.to_string(),
            Licence::Apache2 => APACHE_2.to_string(),
            Licence::Bsd3Clause => BSD_3_CLAUSE.to_string(),
            Licence::Bsd2Clause => BSD_2_CLAUSE.to_string(),
            Licence::Mpl2 => MPL_2.to_string(),
            Licence::Epl2 => EPL_2.to_string(),
            Licence::Bsl1 => BSL_1.to_string(),
            Licence::Cddl1 => CDDL_1.to_string(),
            Licence::Gpl3OrLater => gnu_notice("General Public License", "3", true),
            Licence::Gpl3Only => gnu_notice("General Public License", "3", false),
            Licence::Gpl2OrLater => gnu_notice("General Public License", "2", true),
            Licence::Gpl2Only => gnu_notice("General Public License", "2", false),
            Licence::Lgpl3OrLater => gnu_notice("Lesser General Public License", "3", true),
            Licence::Lgpl3Only => gnu_notice("Lesser General Public License", "3", false),
            Licence::Lgpl21OrLater => gnu_notice("Lesser General Public License", "2.1", true),
            Licence::Lgpl21Only => gnu_notice("Lesser General Public License", "2.1", false),
            Licence::Agpl3OrLater => gnu_notice("Affero General Public License", "3", true),
            Licence::Agpl3Only => gnu_notice("Affero General Public License", "3", false),
        }
    }
}

/// The GNU family's standard "how to apply these terms" notice.
///
/// One generator for ten licences: they differ only in which General Public
/// License they name and whether the reader may choose a later version. The
/// modern URL form of the last paragraph is used throughout, including for
/// version 2, whose original text pointed at a postal address the FSF left
/// decades ago.
fn gnu_notice(name: &str, version: &str, or_later: bool) -> String {
    let terms = if or_later {
        format!(
            "the Free Software Foundation, either version {version} of the License, or\n\
             (at your option) any later version."
        )
    } else {
        format!("the Free Software Foundation, version {version} of the License.")
    };
    format!(
        "Copyright (C) <YEAR> <HOLDER>\n\
         \n\
         This program is free software: you can redistribute it and/or modify\n\
         it under the terms of the GNU {name} as published by\n\
         {terms}\n\
         \n\
         This program is distributed in the hope that it will be useful,\n\
         but WITHOUT ANY WARRANTY; without even the implied warranty of\n\
         MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the\n\
         GNU {name} for more details.\n\
         \n\
         You should have received a copy of the GNU {name}\n\
         along with this program. If not, see <https://www.gnu.org/licenses/>.\n"
    )
}

/// Phrases that identify *a* licence already present in a file.
///
/// Used only by [`already_present`], which asks "does this carry a licence at
/// all" — not which one. `General Public License` is deliberately unqualified
/// so that it catches the Lesser and Affero variants too.
const PRESENT_MARKERS: [&str; 9] = [
    "MIT License",
    "Permission is hereby granted, free of charge",
    "Licensed under the Apache License",
    "General Public License",
    "Redistribution and use in source and binary forms",
    "Mozilla Public License",
    "Eclipse Public License",
    "Boost Software License",
    "Common Development and Distribution License",
];

const MIT: &str = r#"MIT License

Copyright (c) <YEAR> <HOLDER>

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

const APACHE_2: &str = r#"Copyright <YEAR> <HOLDER>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"#;

const BSD_3_CLAUSE: &str = r#"Copyright (c) <YEAR> <HOLDER>

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"#;

const BSD_2_CLAUSE: &str = r#"Copyright (c) <YEAR> <HOLDER>

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"#;

const MPL_2: &str = r#"Copyright (c) <YEAR> <HOLDER>

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"#;

const EPL_2: &str = r#"Copyright (c) <YEAR> <HOLDER>

This program and the accompanying materials are made available under the
terms of the Eclipse Public License 2.0 which is available at
https://www.eclipse.org/legal/epl-2.0/

SPDX-License-Identifier: EPL-2.0
"#;

const BSL_1: &str = r#"Copyright (c) <YEAR> <HOLDER>

Distributed under the Boost Software License, Version 1.0.
(See accompanying file LICENSE_1_0.txt or copy at
https://www.boost.org/LICENSE_1_0.txt)
"#;

const CDDL_1: &str = r#"Copyright (c) <YEAR> <HOLDER>

The contents of this file are subject to the terms of the Common Development
and Distribution License, Version 1.0 only (the "License"). You may not use
this file except in compliance with the License.

You can obtain a copy of the license at
https://opensource.org/licenses/CDDL-1.0

See the License for the specific language governing permissions and
limitations under the License.
"#;

/// What each *process-wide* placeholder is replaced with, in order.
///
/// A table rather than a chain of `.replace()` calls so that adding one is
/// a line here and a token in the templates, with no code to touch.
///
/// Only the year lives here. It can't be static: a server left running past
/// midnight on 31 December would otherwise keep stamping last year onto
/// everything it saves. `<HOLDER>` is *not* here — it is a property of the
/// project being written into, not of this process, and is substituted from
/// [`Project::holder`].
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

/// What a file generated into a project should carry at the top of it.
///
/// Built by [`Project::detect`] from the workspace root, or by
/// [`Project::resolve`] with `/license`'s answer laid over it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Project {
    source: Source,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Source {
    /// A licence this program has a header for, and the copyright holder to
    /// name in it.
    Known { licence: Licence, holder: String },
    /// The project's own `LICENSE` file, used **verbatim**.
    ///
    /// For a licence outside [`Licence::ALL`] — a custom corporate notice, or
    /// simply one this set does not cover. The project has said what its
    /// licence is; that it is not one of the eighteen here is no reason to
    /// put a different licence on its files, and the file's own text is by
    /// definition the right words. It carries its own copyright line, so
    /// nothing is substituted into it and no holder has to be found.
    ///
    /// The whole file, however long it is: truncating a licence is not a
    /// licence. `/license <spdx>` is the way out if that is not wanted.
    File(String),
}

impl Project {
    /// The licence a file generated into `workspace` should carry.
    ///
    /// In order: what the manifest declares, what the `LICENSE` file is,
    /// that file *verbatim* when it is a licence this program does not know,
    /// and [`DEFAULT_LICENCE`] when the project says nothing at all. A
    /// default is what makes the common case — a new project with nothing
    /// declared yet — still produce a licensed file; `/license` is how to say
    /// otherwise, including saying "no header".
    ///
    /// `None` only when a *known* licence was resolved and no copyright
    /// holder could be found, because a header naming nobody would mean
    /// inventing an attribution.
    pub fn detect(workspace: &Path) -> Option<Self> {
        Self::resolve(workspace, None, None)
    }

    /// [`Self::detect`] with `/license`'s answer laid over it: `licence`
    /// and/or `holder` given explicitly, anything not given still detected.
    ///
    /// Naming a licence always produces a known header — it is an explicit
    /// instruction, and overrides even an unrecognised `LICENSE` file.
    pub fn resolve(
        workspace: &Path,
        licence: Option<Licence>,
        holder: Option<&str>,
    ) -> Option<Self> {
        let holder_given = holder.map(str::trim).filter(|h| !h.is_empty());
        if let Some(licence) = licence.or_else(|| declared_licence(workspace)) {
            return Self::known(workspace, licence, holder_given);
        }
        match licence_file(workspace) {
            Some((_, Some(known))) => Self::known(workspace, known, holder_given),
            Some((text, None)) => Some(Self {
                source: Source::File(text),
            }),
            None => Self::known(workspace, DEFAULT_LICENCE, holder_given),
        }
    }

    fn known(workspace: &Path, licence: Licence, holder: Option<&str>) -> Option<Self> {
        let holder = match holder {
            Some(given) => given.to_string(),
            None => self::holder(workspace)?,
        };
        Some(Self {
            source: Source::Known { licence, holder },
        })
    }

    /// The licence the project *declares*, with nothing defaulted — what
    /// `/license` reports as auto-detected, as opposed to what it falls back
    /// to. `None` means the project says nothing this program recognises,
    /// which includes having a `LICENSE` file it cannot name.
    pub fn declared(workspace: &Path) -> Option<Licence> {
        declared_licence(workspace).or_else(|| licence_file(workspace)?.1)
    }

    /// Whether the workspace has a `LICENSE` file whose licence is not one
    /// of [`Licence::ALL`] — the case whose text is used verbatim.
    pub fn has_unknown_licence_file(workspace: &Path) -> bool {
        matches!(licence_file(workspace), Some((_, None)))
    }

    /// The licence, when it is one this program knows. `None` when the
    /// project's own `LICENSE` file is being used verbatim.
    pub fn licence(&self) -> Option<Licence> {
        match &self.source {
            Source::Known { licence, .. } => Some(*licence),
            Source::File(_) => None,
        }
    }

    /// The copyright holder the header will name, or `None` when the header
    /// is a verbatim `LICENSE` file carrying its own.
    pub fn holder(&self) -> Option<&str> {
        match &self.source {
            Source::Known { holder, .. } => Some(holder),
            Source::File(_) => None,
        }
    }

    /// One phrase naming this licence, for a line of prose about it.
    pub fn label(&self) -> String {
        match &self.source {
            Source::Known { licence, .. } => format!("{} ({})", licence.spdx(), licence.title()),
            Source::File(_) => "the project's own LICENSE file".to_string(),
        }
    }

    /// The header text with its placeholders still in it.
    fn text(&self) -> String {
        match &self.source {
            Source::Known { licence, .. } => licence.header_text(),
            Source::File(text) => text.clone(),
        }
    }

    fn holder_value(&self) -> &str {
        self.holder().unwrap_or_default()
    }
}

/// What a project that says nothing about its licensing gets.
pub const DEFAULT_LICENCE: Licence = Licence::Mit;

/// What `/license` pinned for a session, laid over what the workspace says.
///
/// Held by the `ToolExecutor` and consulted on every `create_file`, so a
/// choice made mid-session applies to the next file the model writes without
/// anything being rebuilt.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum Choice {
    /// Nothing pinned: the workspace decides, defaulting to
    /// [`DEFAULT_LICENCE`].
    #[default]
    Auto,
    /// A licence the user named, and optionally a holder to go with it.
    Use {
        licence: Licence,
        holder: Option<String>,
    },
    /// `/license none` — generated files carry no header in this session.
    None,
}

/// The workspace's `LICENSE` file: its text, and which licence it is when
/// that can be told.
fn licence_file(workspace: &Path) -> Option<(String, Option<Licence>)> {
    for name in [
        "LICENSE",
        "LICENSE.md",
        "LICENSE.txt",
        "LICENCE",
        "COPYING",
        "COPYING.md",
    ] {
        let Ok(text) = std::fs::read_to_string(workspace.join(name)) else {
            continue;
        };
        if text.trim().is_empty() {
            continue;
        }
        let licence = licence_of(&text);
        return Some((text, licence));
    }
    None
}

/// The `license` field of whichever manifest the project carries.
///
/// Read with a line scan rather than a TOML/JSON parser: the field is a
/// single quoted string on its own line in every manifest that has one, and
/// this file has no business pulling a parser in — or failing on a manifest
/// some other tool would accept — for one lookup.
fn declared_licence(workspace: &Path) -> Option<Licence> {
    for (file, key) in [
        ("Cargo.toml", "license"),
        ("pyproject.toml", "license"),
        ("package.json", "\"license\""),
    ] {
        let Ok(text) = std::fs::read_to_string(workspace.join(file)) else {
            continue;
        };
        for line in text.lines() {
            if let Some(value) = quoted_value_after(line.trim(), key)
                && let Some(licence) = from_spdx(&value)
            {
                return Some(licence);
            }
        }
    }
    None
}

/// The first quoted string after `key` on a manifest line, when the line
/// assigns one.
///
/// `license = "MIT"`, `"license": "MIT"` and pyproject's
/// `license = { text = "MIT" }` all reduce to that, which is why one
/// helper covers three manifest formats.
fn quoted_value_after(line: &str, key: &str) -> Option<String> {
    let rest = line.strip_prefix(key)?;
    if !rest.trim_start().starts_with(['=', ':']) {
        return None;
    }
    let open = rest.find('"')?;
    let close = rest[open + 1..].find('"')?;
    Some(rest[open + 1..open + 1 + close].to_string())
}

/// An SPDX identifier, as a manifest declares it or as `/license` takes it.
///
/// Case-insensitive, and the deprecated bare `GPL-3.0` spellings are read as
/// SPDX defines them — `-only`. An *expression* — `MIT OR Apache-2.0`,
/// `Apache-2.0 WITH LLVM-exception` — is refused rather than resolved to one
/// of its halves: a dual-licensed project's per-file header is a choice its
/// maintainers make, not one to be made for them by taking whichever
/// identifier came first.
pub fn from_spdx(value: &str) -> Option<Licence> {
    let upper = value.trim().to_ascii_uppercase();
    if upper.contains(" OR ") || upper.contains(" AND ") || upper.contains(" WITH ") {
        return None;
    }
    // The bare, deprecated GNU spellings, which SPDX defines as `-only`.
    let upper = match upper.as_str() {
        "GPL-2.0" => "GPL-2.0-ONLY",
        "GPL-3.0" => "GPL-3.0-ONLY",
        "LGPL-2.1" => "LGPL-2.1-ONLY",
        "LGPL-3.0" => "LGPL-3.0-ONLY",
        "AGPL-3.0" => "AGPL-3.0-ONLY",
        "GPL-2.0+" => "GPL-2.0-OR-LATER",
        "GPL-3.0+" => "GPL-3.0-OR-LATER",
        "LGPL-2.1+" => "LGPL-2.1-OR-LATER",
        "LGPL-3.0+" => "LGPL-3.0-OR-LATER",
        "AGPL-3.0+" => "AGPL-3.0-OR-LATER",
        other => other,
    };
    Licence::ALL
        .into_iter()
        .find(|licence| licence.spdx().eq_ignore_ascii_case(upper))
}

/// Which licence a `LICENSE` file's text is.
fn licence_of(text: &str) -> Option<Licence> {
    let has = |needle: &str| text.contains(needle);
    // An older version has to say so; anything else is the current one. A
    // GNU text that named no version at all used to fall through to the
    // *default* licence, which then made its `Copyright (C) 2007 Free
    // Software Foundation` line eligible as a holder.
    let gnu_version = |current: Licence, older: Licence| {
        if has("Version 2") { older } else { current }
    };
    if has("GNU AFFERO GENERAL PUBLIC LICENSE") || has("GNU Affero General Public License") {
        return Some(Licence::Agpl3OrLater);
    }
    if has("GNU LESSER GENERAL PUBLIC LICENSE") || has("GNU Lesser General Public License") {
        return Some(gnu_version(Licence::Lgpl3OrLater, Licence::Lgpl21OrLater));
    }
    if has("GNU GENERAL PUBLIC LICENSE") || has("GNU General Public License") {
        return Some(gnu_version(Licence::Gpl3OrLater, Licence::Gpl2OrLater));
    }
    if has("Apache License") && has("Version 2.0") {
        return Some(Licence::Apache2);
    }
    if has("Mozilla Public License") {
        return Some(Licence::Mpl2);
    }
    if has("Eclipse Public License") {
        return Some(Licence::Epl2);
    }
    if has("Boost Software License") {
        return Some(Licence::Bsl1);
    }
    if has("Common Development and Distribution License") {
        return Some(Licence::Cddl1);
    }
    if has("Redistribution and use in source and binary forms") {
        return Some(if has("Neither the name") {
            Licence::Bsd3Clause
        } else {
            Licence::Bsd2Clause
        });
    }
    if has("MIT License") || has("Permission is hereby granted, free of charge") {
        return Some(Licence::Mit);
    }
    None
}

/// The project's copyright holder.
///
/// The manifest comes first, and for the GNU family and Apache it is the only
/// *file* consulted: those projects ship the licence's own boilerplate as
/// `LICENSE`, and its copyright line names the Free Software Foundation or
/// the Apache Software Foundation rather than anybody who wrote this code.
/// Reading it would attribute every generated file to the FSF — measured on
/// orangu's own repository, whose `LICENSE` opens `Copyright (C) 2007 Free
/// Software Foundation, Inc.`. MIT and the BSDs put the project's real holder
/// in that line, so for those it is a good source and is used when the
/// manifest has nothing.
///
/// Failing both, the name Git commits here are made under. That is not a
/// guess: it is the identity already attached to every change in this
/// repository, and it is what a person who has not filled in a manifest would
/// put in the header themselves. `/license <spdx> <holder>` overrides it.
fn holder(workspace: &Path) -> Option<String> {
    manifest_holder(workspace)
        .or_else(|| licence_file_holder(workspace))
        .or_else(|| git_user_name(workspace))
}

/// Whether a `LICENSE` file of this licence carries the *project's* copyright
/// line or the licence steward's.
///
/// MIT and the BSDs are filled in per project, so their copyright line names
/// whoever wrote the code. Everything else ships as boilerplate naming the
/// Free Software Foundation, the Apache Software Foundation, Mozilla, and so
/// on — reading a holder out of one of those attributes the project's files to
/// a foundation that has never seen them.
fn names_the_project(licence: Licence) -> bool {
    matches!(
        licence,
        Licence::Mit | Licence::Bsd3Clause | Licence::Bsd2Clause
    )
}

/// `git config user.name`, as configured for `workspace`.
fn git_user_name(workspace: &Path) -> Option<String> {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(workspace)
        .args(["config", "user.name"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let name = String::from_utf8_lossy(&output.stdout).trim().to_string();
    (!name.is_empty()).then_some(name)
}

fn manifest_holder(workspace: &Path) -> Option<String> {
    for (file, key) in [
        ("Cargo.toml", "authors"),
        ("pyproject.toml", "authors"),
        ("package.json", "\"author\""),
    ] {
        let Ok(text) = std::fs::read_to_string(workspace.join(file)) else {
            continue;
        };
        for line in text.lines() {
            if let Some(value) = quoted_value_after(line.trim(), key) {
                let name = strip_email(&value);
                if !name.is_empty() {
                    return Some(name);
                }
            }
        }
    }
    None
}

fn licence_file_holder(workspace: &Path) -> Option<String> {
    for name in ["LICENSE", "LICENSE.md", "LICENSE.txt", "LICENCE", "COPYING"] {
        let Ok(text) = std::fs::read_to_string(workspace.join(name)) else {
            continue;
        };
        // Decided from the file's *own* text, not from the licence finally
        // resolved: a project that declares `MIT` in its manifest but ships
        // the GPL as `LICENSE` must still not attribute anything to the FSF.
        if !licence_of(&text).is_some_and(names_the_project) {
            continue;
        }
        if let Some(holder) = text.lines().take(20).find_map(holder_after_copyright) {
            return Some(holder);
        }
    }
    None
}

/// The holder named by a `Copyright (c) 2024 Someone` line, or `None` when
/// the line is not one.
fn holder_after_copyright(line: &str) -> Option<String> {
    let rest = line.trim().strip_prefix("Copyright")?.trim_start();
    let rest = rest
        .strip_prefix("(c)")
        .or_else(|| rest.strip_prefix("(C)"))
        .unwrap_or(rest)
        .trim_start();
    // Drop a year or a year range, however it is punctuated.
    let rest = rest
        .trim_start_matches(|c: char| c.is_ascii_digit() || c == '-' || c == ',' || c == ' ')
        .trim();
    let holder = strip_email(rest);
    (!holder.is_empty()).then_some(holder)
}

/// `Name <mail@example.com>` — and the trailing `<…>` URL a licence line can
/// carry — reduced to `Name`.
fn strip_email(value: &str) -> String {
    match value.split_once('<') {
        Some((name, _)) => name.trim().to_string(),
        None => value.trim().to_string(),
    }
}

/// The finished licence header for a file, comment markers and all, ending
/// in the blank line that separates it from the code.
///
/// `None` when the file's comment syntax isn't known — a licence header that
/// isn't a comment is a syntax error, and a `#` at the top of a JSON file
/// produces a broken file rather than a differently-licensed one.
pub fn header_for(file_name: &str, project: &Project) -> Option<String> {
    let style = comment_style(file_name)?;
    let mut text = project.text();
    for (placeholder, value) in PLACEHOLDERS {
        if text.contains(placeholder) {
            text = text.replace(placeholder, &value());
        }
    }
    text = text.replace("<HOLDER>", project.holder_value());

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

/// True when `content` already carries *any* licence this program knows,
/// whoever put it there — a model that wrote its own header, or a file being
/// rewritten that had one already.
///
/// Every marker is checked, not only the project's own: stacking an MIT
/// header on top of a GPL one the model wrote is worse than stacking a
/// second copy of the same licence, and both are prevented here.
pub fn already_present(content: &str) -> bool {
    PRESENT_MARKERS
        .iter()
        .any(|marker| content.contains(marker))
}

/// `content` with the project's licence header on top, ready to write as
/// `file_name`.
///
/// Returns it unchanged when the project's licence could not be established
/// (`project` is `None`), when the file's comment syntax isn't known, or
/// when a licence is already there.
///
/// A shebang and an XML declaration keep the first line — a licence above
/// either one leaves a script the kernel won't exec and a document no
/// parser will accept — so the header goes directly beneath them. This is
/// the one implementation of that rule; the web console reaches it over
/// `POST /api/license-header` rather than repeating it in JavaScript.
pub fn apply(content: &str, file_name: &str, project: Option<&Project>) -> String {
    let Some(project) = project else {
        return content.to_string();
    };
    let Some(header) = header_for(file_name, project) else {
        return content.to_string();
    };
    // A licence already there is not covered with another — whether it is one
    // of the ones this program knows, or the project's own text arriving for
    // the second time (a verbatim `LICENSE` file has no marker in the list).
    if already_present(content) || header_already_there(content, &header) {
        return content.to_string();
    }

    let preamble = first_line_preamble(content);
    let (kept, rest) = content.split_at(preamble);
    format!("{kept}{header}{rest}")
}

/// Whether `content` already opens with the header being applied — matched on
/// the header's own first non-blank line of text, which is what a verbatim
/// `LICENSE` file has instead of a marker in [`PRESENT_MARKERS`].
fn header_already_there(content: &str, header: &str) -> bool {
    let Some(first) = header
        .lines()
        .map(|line| line.trim_start_matches(['/', '#', '-', ';', '%', '*', '!', '\'', ' ']))
        .map(str::trim)
        .find(|line| line.len() > 12)
    else {
        return false;
    };
    content.contains(first)
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

    /// A project fixture. The header tests are about *shape* — comment
    /// markers, blank lines, placeholder substitution — so they all run
    /// against one licence and one holder rather than each building their
    /// own.
    fn project(licence: Licence) -> Project {
        Project {
            source: Source::Known {
                licence,
                holder: "Example Holder".to_string(),
            },
        }
    }

    fn mit() -> Project {
        project(Licence::Mit)
    }

    /// A workspace on disk with the given files in it.
    fn workspace_with(files: &[(&str, &str)]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("workspace");
        for (name, body) in files {
            std::fs::write(dir.path().join(name), body).expect("write");
        }
        dir
    }

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
            let header = header_for(file, &mit()).unwrap_or_else(|| panic!("no header for {file}"));
            assert!(header.starts_with(expected), "{file}: {header}");
        }
    }

    /// A markup file has no line comment, so the licence is delimited.
    #[test]
    fn wraps_the_licence_in_block_comments_where_there_is_no_line_comment() {
        let html = header_for("index.html", &mit()).expect("html header");
        assert!(html.starts_with("<!--\nMIT License\n"), "{html}");
        assert!(html.trim_end().ends_with("-->"), "{html}");

        let css = header_for("style.css", &mit()).expect("css header");
        assert!(css.starts_with("/*\n * MIT License\n"), "{css}");
        assert!(css.trim_end().ends_with(" */"), "{css}");

        let ml = header_for("parser.ml", &mit()).expect("ml header");
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
            assert!(
                header_for(file, &mit()).is_none(),
                "{file} should get no header"
            );
        }
    }

    /// The year is filled in at download time, not baked in at build time.
    #[test]
    fn substitutes_the_year() {
        let header = header_for("main.rs", &mit()).expect("header");
        assert!(!header.contains("<YEAR>"), "{header}");
        assert!(
            header.contains(&format!("Copyright (c) {} Example Holder", current_year())),
            "{header}"
        );
    }

    /// Catches a placeholder added to a template and never wired into
    /// [`PLACEHOLDERS`], which would otherwise ship `<SOMETHING>` at the top
    /// of every file someone saves.
    ///
    /// Not "contains no `<`": the GNU family's notice ends by pointing at
    /// `<https://www.gnu.org/licenses/>`, and every text is free to contain a
    /// URL that way. A placeholder is specifically `<ALL_CAPS>`.
    #[test]
    fn no_placeholder_survives_into_a_saved_file() {
        for licence in Licence::ALL {
            let header = header_for("main.rs", &project(licence)).expect("header");
            assert!(
                !contains_placeholder(&header),
                "{licence:?} kept an unsubstituted placeholder: {header}"
            );
        }
    }

    /// `<ALL_CAPS>`, which is what every placeholder in this file looks like
    /// and what no licence text contains for any other reason.
    fn contains_placeholder(text: &str) -> bool {
        text.split('<').skip(1).any(|rest| {
            rest.split_once('>').is_some_and(|(inner, _)| {
                !inner.is_empty() && inner.chars().all(|c| c.is_ascii_uppercase() || c == '_')
            })
        })
    }

    /// A blank licence line stays blank rather than becoming `"// "`, which
    /// editors and linters strip anyway.
    #[test]
    fn blank_lines_carry_a_bare_marker() {
        let header = header_for("main.rs", &mit()).expect("header");
        assert!(header.contains("\n//\n"), "{header}");
        assert!(!header.contains("// \n"), "{header}");
    }

    /// It has to end separated from the code, or the first line of the
    /// snippet lands against the licence.
    #[test]
    fn ends_with_a_blank_line() {
        assert!(
            header_for("main.rs", &mit())
                .expect("header")
                .ends_with("\n\n")
        );
        assert!(
            header_for("index.html", &mit())
                .expect("header")
                .ends_with("\n\n")
        );
    }

    /// A block comment that contains its own terminator ends early and
    /// spills the rest of the licence into the file as code.
    #[test]
    fn the_licence_text_cannot_close_a_block_comment_early() {
        for licence in Licence::ALL {
            let text = licence.header_text();
            assert!(
                !text.contains("-->"),
                "{licence:?} would end an HTML comment"
            );
            assert!(!text.contains("*/"), "{licence:?} would end a C comment");
            assert!(
                !text.contains("*)"),
                "{licence:?} would end an OCaml comment"
            );
        }
    }

    /// The whole job, as `create_file` and the console's download both use
    /// it: header on top, code below, one blank line between.
    #[test]
    fn apply_puts_the_header_above_the_code() {
        let out = apply("fn main() {}\n", "main.rs", Some(&mit()));
        assert!(out.starts_with("// MIT License\n"), "{out}");
        assert!(out.ends_with("\nfn main() {}\n"), "{out}");
    }

    /// A shebang stops being a shebang the moment anything precedes it, and
    /// an XML declaration stops being well-formed.
    #[test]
    fn apply_keeps_a_shebang_or_xml_declaration_on_the_first_line() {
        let script = apply("#!/usr/bin/env bash\nset -e\n", "deploy.sh", Some(&mit()));
        assert!(
            script.starts_with("#!/usr/bin/env bash\n# MIT License\n"),
            "{script}"
        );
        assert!(script.ends_with("\nset -e\n"), "{script}");

        let xml = apply(
            "<?xml version=\"1.0\"?>\n<rss/>\n",
            "feed.xml",
            Some(&mit()),
        );
        assert!(xml.starts_with("<?xml version=\"1.0\"?>\n<!--\n"), "{xml}");

        // Not every first line is a preamble.
        let plain = apply("<rss/>\n", "feed.xml", Some(&mit()));
        assert!(plain.starts_with("<!--\n"), "{plain}");
    }

    /// Writing the licence twice is the failure a model emitting its own
    /// header would otherwise cause, and the one a rewrite would.
    #[test]
    fn apply_does_not_stack_a_second_licence() {
        let once = apply("fn main() {}\n", "main.rs", Some(&mit()));
        assert_eq!(apply(&once, "main.rs", Some(&mit())), once);
        assert!(already_present(&once));
        assert!(!already_present("fn main() {}\n"));
    }

    /// The whole point of the change: a file generated into a GPL project
    /// carries *that* project's notice, not a default one. This is orangu's
    /// own repository, where every generated file used to arrive claiming
    /// MIT.
    #[test]
    fn a_gpl_project_gets_a_gpl_header() {
        let ws = workspace_with(&[(
            "Cargo.toml",
            "[package]\nname = \"thing\"\nlicense = \"GPL-3.0-or-later\"\nauthors = [\"Jane Roe <jane@example.com>\"]\n",
        )]);
        let project = Project::detect(ws.path()).expect("detected");
        assert_eq!(project.licence(), Some(Licence::Gpl3OrLater));
        assert_eq!(project.holder(), Some("Jane Roe"));

        let out = apply("fn main() {}\n", "main.rs", Some(&project));
        assert!(out.contains("GNU General Public License"), "{out}");
        assert!(out.contains("any later version"), "{out}");
        assert!(!out.contains("MIT"), "{out}");
    }

    /// `-only` and `-or-later` are different licences and get different
    /// notices; the deprecated bare `GPL-3.0` is SPDX's `-only`.
    #[test]
    fn the_two_gpl_threes_are_not_interchangeable() {
        assert_eq!(from_spdx("GPL-3.0-only"), Some(Licence::Gpl3Only));
        assert_eq!(from_spdx("GPL-3.0"), Some(Licence::Gpl3Only));
        assert_eq!(from_spdx("GPL-3.0-or-later"), Some(Licence::Gpl3OrLater));
        let only = header_for("main.rs", &project(Licence::Gpl3Only)).expect("header");
        assert!(only.contains("version 3 of the License."), "{only}");
        assert!(!only.contains("any later version"), "{only}");
        let or_later = header_for("main.rs", &project(Licence::Gpl3OrLater)).expect("header");
        assert!(or_later.contains("any later version"), "{or_later}");
    }

    /// An MIT project still gets MIT — the old behaviour, now because it was
    /// read off the project rather than assumed.
    #[test]
    fn an_mit_project_still_gets_mit() {
        let ws = workspace_with(&[(
            "LICENSE",
            "MIT License\n\nCopyright (c) 2024 Acme Corp\n\nPermission is hereby granted, free of charge...\n",
        )]);
        let project = Project::detect(ws.path()).expect("detected");
        assert_eq!(project.licence(), Some(Licence::Mit));
        assert_eq!(project.holder(), Some("Acme Corp"));
    }

    /// **The trap.** A GPL project ships the FSF's own text as `LICENSE`,
    /// whose copyright line names the Free Software Foundation. Reading the
    /// holder from it would attribute every generated file to the FSF, so
    /// for GPL and Apache the manifest is the only source — and when the
    /// manifest names nobody, there is no header rather than a wrong one.
    #[test]
    fn a_gpl_licence_file_never_supplies_the_copyright_holder() {
        let ws = workspace_with(&[(
            "LICENSE",
            "                    GNU General Public License\n\n \
             Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>\n",
        )]);
        assert_ne!(
            Project::detect(ws.path()).and_then(|p| p.holder().map(str::to_string)),
            Some("Free Software Foundation, Inc.".to_string()),
            "the FSF is not this project's copyright holder"
        );

        // With a manifest naming one, that is who the header names.
        std::fs::write(
            ws.path().join("Cargo.toml"),
            "[package]\nauthors = [\"Jane Roe <jane@example.com>\"]\n",
        )
        .expect("write");
        let project = Project::detect(ws.path()).expect("detected");
        assert_eq!(project.licence(), Some(Licence::Gpl3OrLater));
        assert_eq!(project.holder(), Some("Jane Roe"));
    }

    /// A `LICENSE` this program cannot name is still the project's licence,
    /// so it is used as written rather than replaced with the default. It
    /// carries its own copyright line, so no holder has to be found.
    #[test]
    fn an_unrecognised_licence_file_is_used_verbatim() {
        let text = "ACME INTERNAL SOURCE LICENSE\n\nCopyright (c) 2031 Acme Ltd.\n\n\
                    Use of this file is permitted only by employees of Acme Ltd.\n";
        let ws = workspace_with(&[("LICENSE", text)]);
        assert!(Project::has_unknown_licence_file(ws.path()));
        assert_eq!(Project::declared(ws.path()), None);

        let project = Project::detect(ws.path()).expect("the file is the licence");
        assert_eq!(project.licence(), None, "not one of the known set");
        assert_eq!(project.holder(), None, "the text carries its own");

        let out = apply("fn main() {}\n", "main.rs", Some(&project));
        assert!(
            out.starts_with("// ACME INTERNAL SOURCE LICENSE\n"),
            "{out}"
        );
        assert!(out.contains("// Use of this file is permitted"), "{out}");
        assert!(out.ends_with("\nfn main() {}\n"), "{out}");
        assert!(
            !out.contains("MIT"),
            "the default must not have been used: {out}"
        );

        // And it is not stacked a second time on a file that already has it.
        assert_eq!(apply(&out, "main.rs", Some(&project)), out);
    }

    /// Naming a licence is an explicit instruction and beats even a `LICENSE`
    /// file this program cannot read.
    #[test]
    fn an_explicit_choice_beats_an_unrecognised_licence_file() {
        let ws = workspace_with(&[("LICENSE", "ACME INTERNAL SOURCE LICENSE\n")]);
        let project =
            Project::resolve(ws.path(), Some(Licence::Mit), Some("Acme Ltd")).expect("resolve");
        assert_eq!(project.licence(), Some(Licence::Mit));
        assert_eq!(project.holder(), Some("Acme Ltd"));
    }

    /// A dual-licensed project's per-file header is its maintainers' choice,
    /// so neither half is picked for them — the expression is refused, and
    /// the project reads as declaring nothing.
    #[test]
    fn a_dual_licence_is_not_resolved_to_one_of_its_halves() {
        assert_eq!(from_spdx("MIT OR Apache-2.0"), None);
        assert_eq!(from_spdx("Apache-2.0 WITH LLVM-exception"), None);
        let ws = workspace_with(&[(
            "Cargo.toml",
            "[package]\nlicense = \"MIT OR Apache-2.0\"\nauthors = [\"Jane Roe\"]\n",
        )]);
        assert_eq!(Project::declared(ws.path()), None);
        // Which leaves the default, not a coin flip between the two halves.
        assert_eq!(
            Project::detect(ws.path()).and_then(|p| p.licence()),
            Some(DEFAULT_LICENCE)
        );
    }

    /// A project that says nothing about its licensing gets the default.
    #[test]
    fn a_project_with_no_licence_gets_the_default() {
        let ws = workspace_with(&[("Cargo.toml", "[package]\nauthors = [\"Jane Roe\"]\n")]);
        assert_eq!(Project::declared(ws.path()), None);
        let project = Project::detect(ws.path()).expect("the default applies");
        assert_eq!(project.licence(), Some(DEFAULT_LICENCE));
        assert_eq!(project.licence(), Some(Licence::Mit));
        assert_eq!(project.holder(), Some("Jane Roe"));
    }

    /// `/license` lays its answer over whatever the workspace says, and can
    /// name the holder too.
    #[test]
    fn an_explicit_choice_overrides_what_the_workspace_declares() {
        let ws = workspace_with(&[(
            "Cargo.toml",
            "[package]\nlicense = \"MIT\"\nauthors = [\"Jane Roe\"]\n",
        )]);
        let project =
            Project::resolve(ws.path(), Some(Licence::Apache2), Some("Acme Ltd")).expect("resolve");
        assert_eq!(project.licence(), Some(Licence::Apache2));
        assert_eq!(project.holder(), Some("Acme Ltd"));

        // Only the licence given: the holder is still the project's.
        let project =
            Project::resolve(ws.path(), Some(Licence::Agpl3OrLater), None).expect("resolve");
        assert_eq!(project.licence(), Some(Licence::Agpl3OrLater));
        assert_eq!(project.holder(), Some("Jane Roe"));
    }

    /// Every licence in the set round-trips through its own SPDX identifier,
    /// which is what `/license` takes and what a session stores.
    #[test]
    fn every_licence_round_trips_through_its_spdx_id() {
        for licence in Licence::ALL {
            assert_eq!(from_spdx(licence.spdx()), Some(licence), "{licence:?}");
            assert_eq!(
                from_spdx(&licence.spdx().to_ascii_lowercase()),
                Some(licence),
                "{licence:?} is not case-insensitive"
            );
            let header = header_for("main.rs", &project(licence)).expect("header");
            assert!(!contains_placeholder(&header), "{licence:?}: {header}");
        }
    }

    /// A header the model wrote itself is not doubled, whichever licence it
    /// picked — an MIT header on top of a GPL one is worse than two of the
    /// same.
    #[test]
    fn a_licence_the_model_wrote_is_not_covered_with_another() {
        let gpl = "// GNU General Public License\nfn main() {}\n";
        assert_eq!(apply(gpl, "main.rs", Some(&mit())), gpl);
    }

    /// A format with nowhere to put a comment comes back untouched, so a
    /// generated `.json` is still parseable.
    #[test]
    fn apply_leaves_a_file_it_cannot_comment_alone() {
        let json = "{\"a\": 1}\n";
        assert_eq!(apply(json, "data.json", Some(&mit())), json);
        assert_eq!(apply(json, "no-extension", Some(&mit())), json);
    }

    /// Empty content is what `create_file` sends for an empty file; it must
    /// not panic on the missing newline.
    #[test]
    fn apply_handles_content_with_no_newline() {
        assert!(apply("", "main.rs", Some(&mit())).starts_with("// MIT License"));
        assert!(apply("x = 1", "app.py", Some(&mit())).ends_with("\nx = 1"));
    }

    /// Extensions are matched case-insensitively — a fence naming
    /// `Main.RS` still gets Rust's comment.
    #[test]
    fn matches_the_extension_case_insensitively() {
        assert!(
            header_for("Main.RS", &mit())
                .expect("header")
                .starts_with("// ")
        );
        assert!(
            header_for("MAKEFILE", &mit())
                .expect("header")
                .starts_with("# ")
        );
    }
}
