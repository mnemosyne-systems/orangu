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

//! The documentation's tables, against the code they describe: the HTTP routes,
//! the architecture families, and the weight types each backend reads. A table
//! kept by hand beside the code it describes goes stale without anything
//! noticing, so a change to one side fails here until the other follows.

use regex::Regex;
use std::collections::BTreeSet;

const HTTP_DOC: &str = include_str!("../../../doc/manual/en/80-http.md");
const ENGINE_DOC: &str = include_str!("../../../doc/ENGINE.md");
const SERVER_DOC: &str = include_str!("../../../doc/SERVER.md");
const SERVER_MANUAL: &str = include_str!("../../../doc/manual/en/46-server.md");
const README: &str = include_str!("../../../README.md");
const LOADER: &str = include_str!("engine/loader.rs");

/// The text of `doc`'s section under `heading`, up to the next heading of the
/// same depth or shallower.
fn section<'a>(doc: &'a str, heading: &str) -> &'a str {
    let depth = heading.chars().take_while(|c| *c == '#').count();
    let start = doc
        .find(&format!("\n{heading}\n"))
        .unwrap_or_else(|| panic!("no section {heading}"))
        + heading.len()
        + 2;
    let rest = &doc[start..];
    let end = rest
        .match_indices("\n#")
        .find(|(at, _)| {
            let line = rest[at + 1..].lines().next().unwrap_or_default();
            let hashes = line.chars().take_while(|c| *c == '#').count();
            hashes <= depth && line[hashes..].starts_with(' ')
        })
        .map_or(rest.len(), |(at, _)| at);
    &rest[..end]
}

/// Every path a `.route("…"` in `source` serves.
fn routes_in(source: &str) -> Vec<String> {
    Regex::new(r#"\.route\(\s*"([^"]+)""#)
        .unwrap()
        .captures_iter(source)
        .map(|c| c[1].to_string())
        .collect()
}

/// `source` up to its unit tests, whose routers serve nothing.
fn without_tests(source: &str) -> &str {
    source.split("#[cfg(test)]").next().unwrap_or(source)
}

fn served_routes() -> BTreeSet<String> {
    routes_in(without_tests(include_str!("http/mod.rs")))
        .into_iter()
        .chain(routes_in(without_tests(include_str!(
            "../../files_http.rs"
        ))))
        .collect()
}

/// **Every route a client can call is in the table of endpoints.** A row may
/// cover a family with a trailing `…`, and the file-lifecycle endpoints have a
/// table of their own.
#[test]
fn every_route_is_in_the_table_of_endpoints() {
    let table = section(HTTP_DOC, "### Endpoints at a glance");
    let lifecycle = section(HTTP_DOC, "### File-lifecycle API");
    let families: Vec<String> = Regex::new(r"`(/[^` ]*)…")
        .unwrap()
        .captures_iter(table)
        .map(|c| c[1].to_string())
        .collect();
    let missing: Vec<String> = served_routes()
        .into_iter()
        .filter(|path| {
            let spelled = format!(" {path}`");
            let bare = format!("`{path}`");
            !table.contains(&spelled)
                && !table.contains(&bare)
                && !lifecycle.contains(&spelled)
                && !families.iter().any(|family| path.starts_with(family))
        })
        .collect();
    assert!(
        missing.is_empty(),
        "served but not in `Endpoints at a glance` of doc/manual/en/80-http.md: {missing:?}"
    );
}

/// And the other way: a row names a route that exists.
#[test]
fn every_endpoint_in_the_table_is_served() {
    let served = served_routes();
    let table = section(HTTP_DOC, "### Endpoints at a glance");
    let gone: Vec<String> = Regex::new(r"`(?:GET|POST|DELETE) (/[^` ?]*)")
        .unwrap()
        .captures_iter(table)
        .map(|c| c[1].to_string())
        .filter(|path| !served.contains(path))
        .collect();
    assert!(
        gone.is_empty(),
        "in `Endpoints at a glance` of doc/manual/en/80-http.md but not served: {gone:?}"
    );
}

/// One table of endpoints: `SERVER.md` points at it instead of keeping a copy,
/// and a copy is how it came to describe `json_schema` wrongly.
#[test]
fn the_server_guide_links_to_the_endpoints_rather_than_copying_them() {
    let guide = section(SERVER_DOC, "## Endpoint reference");
    assert!(
        guide.contains("manual/en/80-http.md"),
        "no link to the table"
    );
    assert!(
        !guide.contains("| `POST /v1/chat/completions`"),
        "doc/SERVER.md has its own copy of the endpoint table again"
    );
}

/// `SERVER.md` is a map of the manual, not a second copy of it: every link
/// lands on a real heading, and it stays short enough to read as a map.
#[test]
fn the_server_guide_is_a_map_of_the_manual() {
    let lines = SERVER_DOC.lines().count();
    assert!(lines < 120, "doc/SERVER.md has grown to {lines} lines");
    let docs = [
        ("manual/en/46-server.md", SERVER_MANUAL),
        ("manual/en/80-http.md", HTTP_DOC),
    ];
    let mut links = 0;
    for (path, text) in docs {
        let anchors: Vec<String> = text
            .lines()
            .filter(|line| line.starts_with('#'))
            .map(|line| heading_anchor(line.trim_start_matches('#')))
            .collect();
        for piece in SERVER_DOC.split(&format!("]({path}#")).skip(1) {
            let anchor = piece.split(')').next().unwrap();
            assert!(
                anchors.iter().any(|a| a == anchor),
                "doc/SERVER.md links to {path}#{anchor}, which is not a heading there"
            );
            links += 1;
        }
    }
    assert!(links >= 10, "doc/SERVER.md links to only {links} sections");
}

/// The anchor a Markdown renderer gives a heading.
fn heading_anchor(heading: &str) -> String {
    heading
        .trim()
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || matches!(c, ' ' | '-' | '_'))
        .map(|c| if c == ' ' { '-' } else { c })
        .collect()
}

fn arch_family_variants() -> Vec<String> {
    let body = LOADER
        .split("pub enum ArchFamily {")
        .nth(1)
        .and_then(|rest| rest.split("\n}").next())
        .expect("ArchFamily in engine/loader.rs");
    let variant = Regex::new(r"^    ([A-Z][A-Za-z0-9]*),?$").unwrap();
    body.lines()
        .filter_map(|line| variant.captures(line).map(|c| c[1].to_string()))
        .collect()
}

/// `n` as the documentation spells it.
fn in_words(n: usize) -> String {
    const ONES: [&str; 10] = [
        "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    ];
    assert!((20..40).contains(&n), "{n} is outside what this spells");
    let tens = if n < 30 { "twenty" } else { "thirty" };
    match n % 10 {
        0 => tens.to_string(),
        ones => format!("{tens}-{}", ONES[ones]),
    }
}

fn collapse(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Every count of the architecture families, against `ArchFamily`. A variant
/// added without them is how the README, the guide and the engine map came to
/// say three different numbers.
#[test]
fn the_documented_family_counts_match_arch_family() {
    let variants = arch_family_variants();
    assert!(variants.contains(&"DFlash".to_string()));
    let all = variants.len();
    // `dflash` is a draft sidecar: a variant that resolves to the model it
    // drafts for rather than serving itself.
    let servable = all - 1;
    let (all_words, servable_words) = (in_words(all), in_words(servable));
    let capitalized = {
        let mut chars = all_words.chars();
        let first = chars.next().unwrap().to_uppercase().to_string();
        first + chars.as_str()
    };
    let engine = collapse(ENGINE_DOC);
    for expected in [
        format!("across {servable_words} servable architecture families"),
        format!("{capitalized} `ArchFamily` variants, {servable_words} *servable* families"),
    ] {
        assert!(
            engine.contains(&expected),
            "doc/ENGINE.md must say: {expected}"
        );
    }
    let scope = format!("for {servable_words} servable architecture families");
    assert!(
        collapse(SERVER_DOC).contains(&scope),
        "doc/SERVER.md: {scope}"
    );
    assert!(
        collapse(SERVER_MANUAL).contains(&scope),
        "doc/manual/en/46-server.md: {scope}"
    );
    let readme = format!("Runs {servable} servable model families");
    assert!(README.contains(&readme), "README.md must say: {readme}");
}

/// The names of the weight types `supported` answers yes for, in id order.
fn type_names(supported: impl Fn(u32) -> bool) -> String {
    (0..128)
        .filter(|&id| supported(id))
        .map(orangu::gguf::ggml_type_name)
        .collect::<Vec<_>>()
        .join(", ")
}

/// Which weight types each backend reads, against what each answers from
/// `supports_type`. The vendor backends share one list; Metal and DX12 are the
/// `wgpu` backend, and the Rockchip NPU reads what it is given through the CPU.
#[test]
fn the_documented_weight_types_match_the_backends() {
    use crate::engine::backend::{vendor_shaders, vulkan};
    let table = section(ENGINE_DOC, "### Weight types by backend");
    let rows = [
        ("CPU", type_names(crate::engine::quant::supports_type)),
        (
            "Vulkan, Metal, DX12",
            type_names(|id| vulkan::SUPPORTED_TYPES.contains(&id)),
        ),
        (
            "CUDA, ROCm, OpenCL",
            type_names(|id| vendor_shaders::SUPPORTED_TYPES.contains(&id)),
        ),
    ];
    let missing: Vec<String> = rows
        .iter()
        .map(|(row, types)| format!("| {row} | {types} |"))
        .filter(|expected| !table.contains(expected))
        .collect();
    assert!(
        missing.is_empty(),
        "doc/ENGINE.md, Weight types by backend, must have:\n{}",
        missing.join("\n")
    );
}
