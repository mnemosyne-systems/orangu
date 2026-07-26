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

//! Duplicate-code detection across many languages.
//!
//! The workspace is walked (respecting `.gitignore`), and every file whose
//! extension matches a language in the [registry](LANGUAGES) is parsed into a
//! [tree-sitter](https://tree-sitter.github.io/) AST. Each function definition
//! (the [`LanguageSpec::function_kinds`] for that language, including nested
//! ones) becomes a [`FunctionUnit`]: a multiset of *bigrams* — adjacent pairs of
//! named AST node kinds visited in pre-order. Because the comparison is over node
//! *kinds*, not identifiers or literals, two functions that differ only in their
//! names, variables, or constants still score as structurally identical.
//!
//! Every pair of functions **in the same language** is then scored with the
//! Sørensen–Dice coefficient over those bigram multisets:
//!
//! ```text
//! similarity = 2 * |A ∩ B| / (|A| + |B|)
//! ```
//!
//! where the intersection counts shared bigrams with multiplicity. The result
//! ranges from `0.0` (nothing in common) to `1.0` (identical structure). Pairs
//! at or above the threshold (default [`DEFAULT_THRESHOLD`]) are collected into a
//! [`DuplicatesReport`], sorted most-similar first, ready to render to the
//! console ([`DuplicatesReport::to_console`]) or, via `/export duplicates`, to a
//! grouped-by-percentage PDF.
//!
//! There are two scan modes ([`Scope`]). [`scan_duplicates`] compares the whole
//! project against itself. [`scan_duplicates_in_patch`] compares only the
//! functions a branch adds or changes (those overlapping a [`ChangedRegions`]
//! set the caller computes from the branch's diff) against the whole project —
//! answering whether the branch introduces code that already exists.
//!
//! Adding a language is a single [`LanguageSpec`] entry in [`LANGUAGES`]: a
//! display name, the file extensions, the tree-sitter grammar, and the node
//! kind(s) that denote a function or method. Nothing else changes — the scoring
//! is entirely language-agnostic.
//!
//! The report is a *starting point for human review*, not a verdict: the
//! developer is expected to open each pair and decide whether the functions are
//! genuinely duplicated.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::Result;
use ignore::WalkBuilder;
use tree_sitter::{Language, Node, Parser};

/// The default similarity threshold (`0.0`–`1.0`): pairs scoring below this are
/// not reported. `0.80` surfaces strong structural duplicates without drowning
/// the report in coincidental matches.
pub const DEFAULT_THRESHOLD: f64 = 0.80;

/// Functions whose AST has fewer named nodes than this are skipped. Very small
/// functions (one-line getters, trivial wrappers) are structurally
/// interchangeable and would otherwise flood the report with uninteresting
/// "duplicates".
const MIN_FUNCTION_NODES: usize = 20;

/// One language orangu can analyse for duplicates.
pub struct LanguageSpec {
    /// The human-readable name shown in reports (e.g. `Rust`, `C++`).
    pub name: &'static str,
    /// The lower-case file extensions (without the dot) that select this
    /// language.
    pub extensions: &'static [&'static str],
    /// The tree-sitter grammar for the language. A bare `fn` pointer (a
    /// non-capturing closure) so the table stays a `static`.
    grammar: fn() -> Language,
    /// The AST node kinds that denote a function or method definition. A
    /// function whose kind is in this list (anywhere in the tree, so nested and
    /// member functions count) becomes a [`FunctionUnit`].
    function_kinds: &'static [&'static str],
}

/// The supported languages. Each entry is self-contained, so a new language is
/// one row here plus its `tree-sitter-*` dependency in `Cargo.toml`.
///
/// `.h` is treated as C (the common case); C++ headers use the `.hpp`/`.hh`/…
/// extensions. JavaScript covers `.jsx`; TSX is a separate grammar from
/// TypeScript because the two parse `<...>` differently.
pub static LANGUAGES: &[LanguageSpec] = &[
    LanguageSpec {
        name: "Rust",
        extensions: &["rs"],
        grammar: || tree_sitter_rust::LANGUAGE.into(),
        function_kinds: &["function_item"],
    },
    LanguageSpec {
        name: "C",
        extensions: &["c", "h"],
        grammar: || tree_sitter_c::LANGUAGE.into(),
        function_kinds: &["function_definition"],
    },
    LanguageSpec {
        name: "C++",
        extensions: &["cc", "cpp", "cxx", "c++", "hpp", "hh", "hxx"],
        grammar: || tree_sitter_cpp::LANGUAGE.into(),
        function_kinds: &["function_definition"],
    },
    LanguageSpec {
        name: "C#",
        extensions: &["cs"],
        grammar: || tree_sitter_c_sharp::LANGUAGE.into(),
        function_kinds: &[
            "method_declaration",
            "constructor_declaration",
            "local_function_statement",
        ],
    },
    LanguageSpec {
        name: "Go",
        extensions: &["go"],
        grammar: || tree_sitter_go::LANGUAGE.into(),
        function_kinds: &["function_declaration", "method_declaration"],
    },
    LanguageSpec {
        name: "Java",
        extensions: &["java"],
        grammar: || tree_sitter_java::LANGUAGE.into(),
        function_kinds: &["method_declaration", "constructor_declaration"],
    },
    LanguageSpec {
        name: "Python",
        extensions: &["py", "pyi"],
        grammar: || tree_sitter_python::LANGUAGE.into(),
        function_kinds: &["function_definition"],
    },
    LanguageSpec {
        name: "JavaScript",
        extensions: &["js", "mjs", "cjs", "jsx"],
        grammar: || tree_sitter_javascript::LANGUAGE.into(),
        function_kinds: &[
            "function_declaration",
            "generator_function_declaration",
            "method_definition",
        ],
    },
    LanguageSpec {
        name: "TypeScript",
        extensions: &["ts", "mts", "cts"],
        grammar: || tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        function_kinds: &[
            "function_declaration",
            "generator_function_declaration",
            "method_definition",
        ],
    },
    LanguageSpec {
        name: "TSX",
        extensions: &["tsx"],
        grammar: || tree_sitter_typescript::LANGUAGE_TSX.into(),
        function_kinds: &[
            "function_declaration",
            "generator_function_declaration",
            "method_definition",
        ],
    },
    LanguageSpec {
        name: "Ruby",
        extensions: &["rb"],
        grammar: || tree_sitter_ruby::LANGUAGE.into(),
        function_kinds: &["method", "singleton_method"],
    },
    LanguageSpec {
        name: "PHP",
        extensions: &["php"],
        grammar: || tree_sitter_php::LANGUAGE_PHP.into(),
        function_kinds: &["function_definition", "method_declaration"],
    },
    LanguageSpec {
        name: "Bash",
        extensions: &["sh", "bash"],
        grammar: || tree_sitter_bash::LANGUAGE.into(),
        function_kinds: &["function_definition"],
    },
    LanguageSpec {
        name: "Scala",
        extensions: &["scala", "sc"],
        grammar: || tree_sitter_scala::LANGUAGE.into(),
        function_kinds: &["function_definition"],
    },
    LanguageSpec {
        name: "OCaml",
        extensions: &["ml"],
        grammar: || tree_sitter_ocaml::LANGUAGE_OCAML.into(),
        function_kinds: &["value_definition"],
    },
    LanguageSpec {
        name: "Julia",
        extensions: &["jl"],
        grammar: || tree_sitter_julia::LANGUAGE.into(),
        function_kinds: &["function_definition"],
    },
    LanguageSpec {
        name: "Lua",
        extensions: &["lua"],
        grammar: || tree_sitter_lua::LANGUAGE.into(),
        function_kinds: &["function_declaration"],
    },
    LanguageSpec {
        name: "R",
        extensions: &["r"],
        grammar: || tree_sitter_r::LANGUAGE.into(),
        function_kinds: &["function_definition"],
    },
    LanguageSpec {
        name: "Zig",
        extensions: &["zig"],
        grammar: || tree_sitter_zig::LANGUAGE.into(),
        function_kinds: &["function_declaration"],
    },
    LanguageSpec {
        name: "Swift",
        extensions: &["swift"],
        grammar: || tree_sitter_swift::LANGUAGE.into(),
        function_kinds: &["function_declaration"],
    },
    LanguageSpec {
        name: "Dart",
        extensions: &["dart"],
        grammar: || tree_sitter_dart::LANGUAGE.into(),
        function_kinds: &["function_declaration", "method_signature"],
    },
    LanguageSpec {
        name: "Erlang",
        extensions: &["erl"],
        grammar: || tree_sitter_erlang::LANGUAGE.into(),
        function_kinds: &["function_clause"],
    },
];

/// The names of every supported language, for help text and documentation.
pub fn supported_languages() -> Vec<&'static str> {
    LANGUAGES.iter().map(|spec| spec.name).collect()
}

/// Map each registered extension to its language index in [`LANGUAGES`].
fn extension_index() -> HashMap<&'static str, usize> {
    let mut map = HashMap::new();
    for (id, spec) in LANGUAGES.iter().enumerate() {
        for extension in spec.extensions {
            map.insert(*extension, id);
        }
    }
    map
}

/// Where a function lives, for display in a report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FunctionLocation {
    /// Path relative to the scanned root.
    pub path: PathBuf,
    /// The function's name (best-effort; `<anonymous>` when none can be read).
    pub name: String,
    /// The language's display name (from [`LanguageSpec::name`]).
    pub language: &'static str,
    /// 1-based first line of the definition.
    pub start_line: usize,
    /// 1-based last line of the definition.
    pub end_line: usize,
}

impl FunctionLocation {
    /// `path:start–end` followed by the function name, e.g.
    /// `src/foo.rs:10–42 bar`.
    fn display(&self) -> String {
        format!(
            "{}:{}–{} {}",
            self.path.display(),
            self.start_line,
            self.end_line,
            self.name
        )
    }
}

/// A single analysed function: its location plus the bigram multiset used for
/// scoring.
#[derive(Clone, Debug)]
struct FunctionUnit {
    location: FunctionLocation,
    /// Index into [`LANGUAGES`]; only functions sharing it are compared.
    lang_id: usize,
    /// Count of each adjacent (kind, kind) pair in the pre-order named-node
    /// traversal.
    bigrams: HashMap<(u16, u16), u32>,
    /// The total number of bigrams (the sum of the map's counts).
    total: u32,
}

/// Two functions found to be structurally similar.
#[derive(Clone, Debug)]
pub struct DuplicatePair {
    pub a: FunctionLocation,
    pub b: FunctionLocation,
    /// Sørensen–Dice similarity in `0.0`–`1.0`.
    pub similarity: f64,
}

impl DuplicatePair {
    /// The similarity as a whole-number percentage (`0`–`100`).
    pub fn percent(&self) -> u32 {
        (self.similarity * 100.0).round() as u32
    }
}

/// What a scan compared.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Scope {
    /// Every function in the project compared against every other
    /// ([`scan_duplicates`]).
    Project,
    /// Only the functions a branch adds or changes, each compared against the
    /// whole project ([`scan_duplicates_in_patch`]).
    Patch {
        /// The base ref the branch was diffed against (e.g. `origin/main`).
        base: String,
        /// How many new/changed functions were compared against the project.
        new_functions: usize,
    },
}

/// The set of lines a branch adds, by file, used to pick out the functions a
/// patch introduces or changes. Paths are relative to the scan root, matching
/// the paths the walk records for each function.
#[derive(Clone, Debug, Default)]
pub struct ChangedRegions {
    files: HashMap<PathBuf, Vec<(usize, usize)>>,
}

impl ChangedRegions {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that lines `start..=end` (1-based) of `path` were added.
    pub fn add_range(&mut self, path: impl Into<PathBuf>, start: usize, end: usize) {
        self.files
            .entry(path.into())
            .or_default()
            .push((start, end));
    }

    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    /// Whether any recorded range for `path` overlaps `start..=end`.
    fn overlaps(&self, path: &Path, start: usize, end: usize) -> bool {
        self.files.get(path).is_some_and(|ranges| {
            ranges
                .iter()
                .any(|&(range_start, range_end)| start <= range_end && range_start <= end)
        })
    }
}

/// The outcome of a duplicate-code scan.
#[derive(Clone, Debug)]
pub struct DuplicatesReport {
    /// The scanned root, for display.
    pub root: PathBuf,
    /// The threshold the scan used (`0.0`–`1.0`).
    pub threshold: f64,
    /// How many source files were parsed.
    pub files_scanned: usize,
    /// How many functions were extracted (the comparison universe).
    pub functions_analyzed: usize,
    /// What was compared against what.
    pub scope: Scope,
    /// The similar pairs, sorted most-similar first.
    pub pairs: Vec<DuplicatePair>,
}

/// The Sørensen–Dice coefficient over two bigram multisets: `2 * shared /
/// (total_a + total_b)`, where `shared` sums the per-bigram minimum count. `0.0`
/// when both are empty.
fn dice(a: &FunctionUnit, b: &FunctionUnit) -> f64 {
    let total = a.total + b.total;
    if total == 0 {
        return 0.0;
    }
    // Iterate the smaller map and probe the larger one.
    let (small, large) = if a.bigrams.len() <= b.bigrams.len() {
        (&a.bigrams, &b.bigrams)
    } else {
        (&b.bigrams, &a.bigrams)
    };
    let mut shared = 0u32;
    for (bigram, &count) in small {
        if let Some(&other) = large.get(bigram) {
            shared += count.min(other);
        }
    }
    2.0 * f64::from(shared) / f64::from(total)
}

/// Collect, in pre-order, the `kind_id` of every *named* node under `node`
/// (anonymous tokens such as punctuation are skipped, so formatting and
/// delimiters do not sway the score).
fn collect_named_kinds(node: Node, out: &mut Vec<u16>) {
    if node.is_named() {
        out.push(node.kind_id());
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_named_kinds(child, out);
    }
}

/// The first identifier-like leaf found in pre-order under `node`, as text. A
/// node counts when its kind contains `identifier` (covering `identifier`,
/// `field_identifier`, `type_identifier`, …) or is an `atom`/`name`, which
/// between them name a function across every supported grammar.
fn first_name_like<'a>(node: Node, source: &'a [u8]) -> Option<&'a str> {
    let kind = node.kind();
    if kind.contains("identifier") || kind == "atom" || kind == "name" {
        return node.utf8_text(source).ok();
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if let Some(text) = first_name_like(child, source) {
            return Some(text);
        }
    }
    None
}

/// Read a function definition's name, best-effort and language-agnostic: prefer
/// the `name` field, then a `declarator` field (C/C++), then the first
/// identifier-like leaf anywhere in the node.
fn function_name(node: Node, source: &[u8]) -> String {
    if let Some(name) = node.child_by_field_name("name")
        && let Some(text) = first_name_like(name, source)
    {
        return text.to_string();
    }
    if let Some(declarator) = node.child_by_field_name("declarator")
        && let Some(text) = first_name_like(declarator, source)
    {
        return text.to_string();
    }
    first_name_like(node, source)
        .unwrap_or("<anonymous>")
        .to_string()
}

/// Build a [`FunctionUnit`] from a function-definition node, or `None` when it
/// is too small to score meaningfully ([`MIN_FUNCTION_NODES`]).
fn build_unit(node: Node, source: &[u8], path: &Path, lang_id: usize) -> Option<FunctionUnit> {
    let mut kinds = Vec::new();
    collect_named_kinds(node, &mut kinds);
    if kinds.len() < MIN_FUNCTION_NODES {
        return None;
    }
    let mut bigrams: HashMap<(u16, u16), u32> = HashMap::new();
    for window in kinds.windows(2) {
        *bigrams.entry((window[0], window[1])).or_insert(0) += 1;
    }
    let total = (kinds.len() - 1) as u32;
    if total == 0 {
        return None;
    }
    Some(FunctionUnit {
        location: FunctionLocation {
            path: path.to_path_buf(),
            name: function_name(node, source),
            language: LANGUAGES[lang_id].name,
            start_line: node.start_position().row + 1,
            end_line: node.end_position().row + 1,
        },
        lang_id,
        bigrams,
        total,
    })
}

/// Walk a parsed tree and append a [`FunctionUnit`] for every function
/// definition (including nested ones).
fn extract_functions(
    root: Node,
    source: &[u8],
    path: &Path,
    lang_id: usize,
    out: &mut Vec<FunctionUnit>,
) {
    let kinds = LANGUAGES[lang_id].function_kinds;
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if kinds.contains(&node.kind())
            && let Some(unit) = build_unit(node, source, path, lang_id)
        {
            out.push(unit);
        }
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            stack.push(child);
        }
    }
}

/// Walk `root` and extract every function from each supported-language file,
/// returning the units and the number of files parsed. The walk honours
/// `.gitignore`; unreadable or unparseable files are skipped.
fn collect_units(root: &Path) -> (Vec<FunctionUnit>, usize) {
    let extensions = extension_index();
    // A parser is created lazily per language and reused across its files.
    let mut parsers: HashMap<usize, Parser> = HashMap::new();

    let mut units: Vec<FunctionUnit> = Vec::new();
    let mut files_scanned = 0usize;

    for entry in WalkBuilder::new(root).build() {
        let Ok(entry) = entry else { continue };
        if !entry.file_type().is_some_and(|kind| kind.is_file()) {
            continue;
        }
        let path = entry.path();
        let Some(extension) = path.extension().and_then(|ext| ext.to_str()) else {
            continue;
        };
        let lowered = extension.to_ascii_lowercase();
        let Some(&lang_id) = extensions.get(lowered.as_str()) else {
            continue;
        };
        let Ok(source) = std::fs::read_to_string(path) else {
            continue;
        };

        // Get (or build) the parser for this language; a grammar that refuses to
        // load is skipped for the rest of the scan.
        let parser = match parsers.entry(lang_id) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                let mut parser = Parser::new();
                if parser
                    .set_language(&(LANGUAGES[lang_id].grammar)())
                    .is_err()
                {
                    continue;
                }
                entry.insert(parser)
            }
        };

        files_scanned += 1;
        let Some(tree) = parser.parse(source.as_bytes(), None) else {
            continue;
        };
        let relative = path.strip_prefix(root).unwrap_or(path);
        extract_functions(
            tree.root_node(),
            source.as_bytes(),
            relative,
            lang_id,
            &mut units,
        );
    }

    (units, files_scanned)
}

/// Sort pairs most-similar first, with location tie-breakers, for a stable
/// report across runs.
fn sort_pairs(pairs: &mut [DuplicatePair]) {
    pairs.sort_by(|x, y| {
        y.similarity
            .partial_cmp(&x.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| x.a.path.cmp(&y.a.path))
            .then_with(|| x.a.start_line.cmp(&y.a.start_line))
            .then_with(|| x.b.path.cmp(&y.b.path))
            .then_with(|| x.b.start_line.cmp(&y.b.start_line))
    });
}

/// Scan `root` for duplicated functions, reporting every same-language pair
/// scoring at or above `threshold` (`0.0`–`1.0`) — the whole project compared
/// against itself ([`Scope::Project`]).
pub fn scan_duplicates(root: &Path, threshold: f64) -> Result<DuplicatesReport> {
    let (units, files_scanned) = collect_units(root);

    let mut pairs = Vec::new();
    for (i, left) in units.iter().enumerate() {
        for right in &units[i + 1..] {
            if left.lang_id != right.lang_id {
                continue;
            }
            let similarity = dice(left, right);
            if similarity >= threshold {
                pairs.push(DuplicatePair {
                    a: left.location.clone(),
                    b: right.location.clone(),
                    similarity,
                });
            }
        }
    }
    sort_pairs(&mut pairs);

    Ok(DuplicatesReport {
        root: root.to_path_buf(),
        threshold,
        files_scanned,
        functions_analyzed: units.len(),
        scope: Scope::Project,
        pairs,
    })
}

/// Scan `root`, but compare only the functions a branch adds or changes (those
/// overlapping `changed`) against the *whole* project ([`Scope::Patch`]) — the
/// duplicates a branch introduces. Each reported pair is oriented with the
/// new/changed function as `a`. `base` labels the ref the branch was diffed
/// against, for the report.
pub fn scan_duplicates_in_patch(
    root: &Path,
    threshold: f64,
    changed: &ChangedRegions,
    base: &str,
) -> Result<DuplicatesReport> {
    let (units, files_scanned) = collect_units(root);

    // The functions the patch touches: those whose span overlaps an added range.
    let new_indices: Vec<usize> = units
        .iter()
        .enumerate()
        .filter(|(_, unit)| {
            changed.overlaps(
                &unit.location.path,
                unit.location.start_line,
                unit.location.end_line,
            )
        })
        .map(|(index, _)| index)
        .collect();

    // Each new function against every other function (same language), de-duped so
    // a pair of two new functions is reported once.
    let mut pairs = Vec::new();
    let mut seen: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
    for &i in &new_indices {
        for j in 0..units.len() {
            if i == j || units[i].lang_id != units[j].lang_id {
                continue;
            }
            if !seen.insert((i.min(j), i.max(j))) {
                continue;
            }
            let similarity = dice(&units[i], &units[j]);
            if similarity >= threshold {
                pairs.push(DuplicatePair {
                    a: units[i].location.clone(),
                    b: units[j].location.clone(),
                    similarity,
                });
            }
        }
    }
    sort_pairs(&mut pairs);

    Ok(DuplicatesReport {
        root: root.to_path_buf(),
        threshold,
        files_scanned,
        functions_analyzed: units.len(),
        scope: Scope::Patch {
            base: base.to_string(),
            new_functions: new_indices.len(),
        },
        pairs,
    })
}

impl DuplicatesReport {
    /// The threshold as a whole-number percentage (`0`–`100`).
    pub fn threshold_percent(&self) -> u32 {
        (self.threshold * 100.0).round() as u32
    }

    /// A plain-text report for the console output window.
    pub fn to_console(&self) -> String {
        let mut out = String::new();
        out.push_str("Duplicate code report\n");
        match &self.scope {
            Scope::Project => out.push_str(&format!(
                "Scanned {} file(s), analysed {} function(s), threshold {}%.\n",
                self.files_scanned,
                self.functions_analyzed,
                self.threshold_percent()
            )),
            Scope::Patch {
                base,
                new_functions,
            } => out.push_str(&format!(
                "Scanned {} file(s); compared {} new/changed function(s) on this branch \
                 against {} project function(s) (base {}), threshold {}%.\n",
                self.files_scanned,
                new_functions,
                self.functions_analyzed,
                base,
                self.threshold_percent()
            )),
        }
        if self.pairs.is_empty() {
            out.push_str(&format!(
                "No function pairs met the {}% similarity threshold.\n",
                self.threshold_percent()
            ));
            return out;
        }
        out.push_str(&format!(
            "Found {} candidate pair(s) — review each to confirm.\n",
            self.pairs.len()
        ));
        for (index, pair) in self.pairs.iter().enumerate() {
            out.push('\n');
            out.push_str(&format!(
                "{}. {}% — {} ↔ {}\n",
                index + 1,
                pair.percent(),
                pair.a.name,
                pair.b.name,
            ));
            out.push_str(&format!("   {}\n", pair.a.display()));
            out.push_str(&format!("   {}\n", pair.b.display()));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn write(dir: &Path, name: &str, contents: &str) {
        fs::write(dir.join(name), contents).unwrap();
    }

    #[test]
    fn dice_of_identical_units_is_one() {
        let dir = tempfile::tempdir().unwrap();
        let body = r#"
            fn process(items: &[i32]) -> i32 {
                let mut total = 0;
                for item in items {
                    if *item > 0 {
                        total += item;
                    } else {
                        total -= item;
                    }
                }
                total
            }
        "#;
        // Same structure, different identifiers — should score ~100%.
        let twin = body
            .replace("process", "compute")
            .replace("items", "values")
            .replace("total", "sum")
            .replace("item", "value");
        write(dir.path(), "a.rs", body);
        write(dir.path(), "b.rs", &twin);

        let report = scan_duplicates(dir.path(), DEFAULT_THRESHOLD).unwrap();
        assert_eq!(report.files_scanned, 2);
        assert!(
            report.pairs.iter().any(|pair| pair.percent() >= 99),
            "expected a near-identical pair, got {:?}",
            report.pairs
        );
    }

    #[test]
    fn threshold_is_honoured_and_recorded() {
        // Two functions that share structure but are not identical: a higher
        // threshold must drop the pair a lower one keeps. This is the contract
        // `/export duplicates` relies on when it replays the last run's
        // threshold, so the report must both record it and filter by it.
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "a.rs",
            r#"
            fn classify(items: &[i32]) -> i32 {
                let mut total = 0;
                for item in items {
                    if *item > 0 {
                        total += item;
                    } else {
                        total -= item;
                    }
                }
                total
            }
            "#,
        );
        write(
            dir.path(),
            "b.rs",
            r#"
            fn summarize(values: &[i32]) -> i32 {
                let mut sum = 0;
                for value in values {
                    sum += value;
                }
                sum
            }
            "#,
        );

        // Establish the actual score, then bracket it.
        let loose = scan_duplicates(dir.path(), 0.0).unwrap();
        let score = loose
            .pairs
            .iter()
            .map(|pair| pair.similarity)
            .next()
            .expect("the two functions form a pair at threshold 0");
        assert!(
            (0.0..1.0).contains(&score),
            "expected a partial match, got {score}"
        );

        let below = (score - 0.05).max(0.0);
        let above = (score + 0.05).min(1.0);

        let kept = scan_duplicates(dir.path(), below).unwrap();
        assert_eq!(kept.threshold, below);
        assert!(
            !kept.pairs.is_empty(),
            "a threshold below the score ({below}) should keep the pair"
        );

        let dropped = scan_duplicates(dir.path(), above).unwrap();
        assert_eq!(dropped.threshold, above);
        assert_eq!(dropped.threshold_percent(), (above * 100.0).round() as u32);
        assert!(
            dropped.pairs.is_empty(),
            "a threshold above the score ({above}) should drop the pair"
        );
    }

    #[test]
    fn dissimilar_functions_are_not_reported() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "a.rs",
            r#"
            fn fibonacci(n: u64) -> u64 {
                let mut a = 0u64;
                let mut b = 1u64;
                for _ in 0..n {
                    let next = a + b;
                    a = b;
                    b = next;
                }
                a
            }
            "#,
        );
        write(
            dir.path(),
            "b.rs",
            r#"
            fn greet(people: &[String]) {
                for person in people {
                    println!("Hello, {}!", person);
                }
            }
            "#,
        );
        let report = scan_duplicates(dir.path(), DEFAULT_THRESHOLD).unwrap();
        assert!(
            report.pairs.is_empty(),
            "unrelated functions should not match: {:?}",
            report.pairs
        );
    }

    #[test]
    fn detects_cross_file_c_duplicates() {
        let dir = tempfile::tempdir().unwrap();
        let body = r#"
            int accumulate(const int *items, int count) {
                int total = 0;
                for (int i = 0; i < count; i++) {
                    if (items[i] > 0) {
                        total += items[i];
                    } else {
                        total -= items[i];
                    }
                }
                return total;
            }
        "#;
        let twin = body
            .replace("accumulate", "reduce")
            .replace("items", "values")
            .replace("total", "sum");
        write(dir.path(), "a.c", body);
        write(dir.path(), "b.c", &twin);
        let report = scan_duplicates(dir.path(), DEFAULT_THRESHOLD).unwrap();
        assert!(
            report.pairs.iter().any(|pair| pair.percent() >= 99),
            "expected the C twins to match: {:?}",
            report.pairs
        );
    }

    #[test]
    fn detects_python_duplicates() {
        let dir = tempfile::tempdir().unwrap();
        let body = "def accumulate(items):\n\
                    \x20   total = 0\n\
                    \x20   for item in items:\n\
                    \x20       if item > 0:\n\
                    \x20           total += item\n\
                    \x20       else:\n\
                    \x20           total -= item\n\
                    \x20   return total\n";
        let twin = body
            .replace("accumulate", "reduce")
            .replace("items", "values")
            .replace("total", "sum")
            .replace("item", "value");
        write(dir.path(), "a.py", body);
        write(dir.path(), "b.py", &twin);
        let report = scan_duplicates(dir.path(), DEFAULT_THRESHOLD).unwrap();
        assert!(
            report
                .pairs
                .iter()
                .any(|pair| pair.percent() >= 99 && pair.a.language == "Python"),
            "expected the Python twins to match: {:?}",
            report.pairs
        );
    }

    #[test]
    fn detects_go_method_duplicates() {
        let dir = tempfile::tempdir().unwrap();
        let body = "package main\n\
                    func accumulate(items []int) int {\n\
                    \ttotal := 0\n\
                    \tfor _, item := range items {\n\
                    \t\tif item > 0 {\n\
                    \t\t\ttotal += item\n\
                    \t\t} else {\n\
                    \t\t\ttotal -= item\n\
                    \t\t}\n\
                    \t}\n\
                    \treturn total\n\
                    }\n";
        let twin = body
            .replace("accumulate", "reduce")
            .replace("items", "values")
            .replace("total", "sum")
            .replace("item", "value");
        write(dir.path(), "a.go", body);
        write(dir.path(), "b.go", &twin);
        let report = scan_duplicates(dir.path(), DEFAULT_THRESHOLD).unwrap();
        assert!(
            report
                .pairs
                .iter()
                .any(|pair| pair.percent() >= 99 && pair.a.language == "Go"),
            "expected the Go twins to match: {:?}",
            report.pairs
        );
    }

    #[test]
    fn different_languages_are_compared_separately() {
        // A Rust and a C function never form a pair even if both are present.
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "only.rs",
            r#"
            fn solo(value: i32) -> i32 {
                let mut acc = value;
                for step in 0..value {
                    acc = acc + step;
                }
                acc
            }
            "#,
        );
        write(
            dir.path(),
            "only.c",
            r#"
            int solo(int value) {
                int acc = value;
                for (int step = 0; step < value; step++) {
                    acc = acc + step;
                }
                return acc;
            }
            "#,
        );
        let report = scan_duplicates(dir.path(), 0.10).unwrap();
        assert!(
            report
                .pairs
                .iter()
                .all(|pair| pair.a.language == pair.b.language),
            "cross-language pairs should never be reported: {:?}",
            report.pairs
        );
    }

    #[test]
    fn tiny_functions_are_skipped() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "a.rs", "fn x() -> i32 { 1 }\n");
        write(dir.path(), "b.rs", "fn y() -> i32 { 2 }\n");
        let report = scan_duplicates(dir.path(), DEFAULT_THRESHOLD).unwrap();
        assert_eq!(report.functions_analyzed, 0);
        assert!(report.pairs.is_empty());
    }

    #[test]
    fn console_renders() {
        let dir = tempfile::tempdir().unwrap();
        let report = scan_duplicates(dir.path(), DEFAULT_THRESHOLD).unwrap();
        let console = report.to_console();
        assert!(console.contains("Duplicate code report"));
        assert!(console.contains("threshold 80%"));
    }

    // Three structurally identical functions (different identifiers): only the
    // one on the "branch" is compared against the project.
    fn three_twins(dir: &Path) {
        let body = r#"
            fn accumulate(items: &[i32]) -> i32 {
                let mut total = 0;
                for item in items {
                    if *item > 0 { total += item; } else { total -= item; }
                }
                total
            }
        "#;
        write(dir, "a.rs", &body.replace("accumulate", "alpha"));
        write(dir, "b.rs", &body.replace("accumulate", "beta"));
        write(dir, "c.rs", &body.replace("accumulate", "newcomer"));
    }

    #[test]
    fn patch_mode_compares_only_new_functions_against_the_project() {
        let dir = tempfile::tempdir().unwrap();
        three_twins(dir.path());

        // Only c.rs (the "newcomer") is new on the branch.
        let mut changed = ChangedRegions::new();
        changed.add_range(PathBuf::from("c.rs"), 1, 100);

        let report =
            scan_duplicates_in_patch(dir.path(), DEFAULT_THRESHOLD, &changed, "origin/main")
                .unwrap();

        // The whole project is the comparison universe (3 functions), but only
        // the 1 new function is compared.
        assert_eq!(report.functions_analyzed, 3);
        assert_eq!(
            report.scope,
            Scope::Patch {
                base: "origin/main".to_string(),
                new_functions: 1
            }
        );
        // newcomer vs alpha and newcomer vs beta — two pairs, always oriented
        // with the new function as `a`.
        assert_eq!(report.pairs.len(), 2);
        assert!(
            report
                .pairs
                .iter()
                .all(|pair| pair.a.path == Path::new("c.rs")),
            "the new function should be `a`: {:?}",
            report.pairs
        );
        // The two pre-existing twins (alpha vs beta) are NOT reported, since
        // neither is new on the branch.
        assert!(
            !report.pairs.iter().any(|pair| {
                let names = [pair.a.name.as_str(), pair.b.name.as_str()];
                names.contains(&"alpha") && names.contains(&"beta")
            }),
            "pre-existing duplicates must not be reported: {:?}",
            report.pairs
        );
    }

    #[test]
    fn patch_mode_with_no_new_functions_reports_nothing() {
        let dir = tempfile::tempdir().unwrap();
        three_twins(dir.path());
        // An empty change set: nothing on the branch is new.
        let changed = ChangedRegions::new();
        let report =
            scan_duplicates_in_patch(dir.path(), DEFAULT_THRESHOLD, &changed, "main").unwrap();
        assert!(matches!(
            report.scope,
            Scope::Patch {
                new_functions: 0,
                ..
            }
        ));
        assert!(report.pairs.is_empty());
        assert!(report.to_console().contains("0 new/changed function"));
    }

    #[test]
    fn every_grammar_loads_and_extensions_are_unique() {
        // Each registered grammar must load, and no extension may be claimed by
        // two languages (which one wins would be undefined).
        let mut seen = std::collections::HashSet::new();
        for spec in LANGUAGES {
            let mut parser = Parser::new();
            assert!(
                parser.set_language(&(spec.grammar)()).is_ok(),
                "grammar for {} failed to load",
                spec.name
            );
            for extension in spec.extensions {
                assert!(
                    seen.insert(*extension),
                    "extension .{extension} is claimed twice (at {})",
                    spec.name
                );
            }
        }
        assert!(supported_languages().contains(&"Rust"));
    }
}
