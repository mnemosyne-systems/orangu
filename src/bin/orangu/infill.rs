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

//! `/infill <file>:<line>[:<column>]`: what the model would write at a place in
//! a file, through the server's `POST /infill`. It shows the completion and
//! changes nothing.

use anyhow::{Context, Result, anyhow, bail};
use orangu::config::LlmConfiguration;
use orangu::graph::store::GraphStore;
use orangu::llm::normalized_openai_endpoint;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// How many other files a request carries for context, and how much of each.
/// The server keeps what fits; this bounds what is read and sent.
const CONTEXT_FILES: usize = 3;
const CONTEXT_FILE_BYTES: usize = 16 * 1024;

const REQUEST_TIMEOUT: Duration = Duration::from_secs(300);

pub(crate) const USAGE: &str = "Usage: /infill <file>:<line>[:<column>]";

/// A place in a file: both 1-based, the column in characters. With no column,
/// the end of the line.
#[derive(Debug, PartialEq)]
pub(crate) struct Location {
    pub(crate) path: String,
    pub(crate) line: usize,
    pub(crate) column: Option<usize>,
}

/// Reads `file:line` or `file:line:column`; the path may itself hold a `:`.
pub(crate) fn parse_location(argument: &str) -> Option<Location> {
    let number = |text: &str| text.parse::<usize>().ok().filter(|&n| n > 0);
    let (head, last) = argument.trim().rsplit_once(':')?;
    let last = number(last)?;
    let (path, line, column) = match head.rsplit_once(':') {
        Some((path, line)) if number(line).is_some() => (path, number(line)?, Some(last)),
        _ => (head, last, None),
    };
    (!path.is_empty()).then(|| Location {
        path: path.to_string(),
        line,
        column,
    })
}

/// `text` cut at `location`.
fn split_at_cursor<'a>(text: &'a str, location: &Location) -> Result<(&'a str, &'a str)> {
    let mut start = 0;
    for _ in 1..location.line {
        let Some(newline) = text[start..].find('\n') else {
            bail!(
                "{} has {} lines, so there is no line {}",
                location.path,
                text.lines().count(),
                location.line
            );
        };
        start += newline + 1;
    }
    let end = text[start..]
        .find('\n')
        .map_or(text.len(), |newline| start + newline);
    let line = text[start..end].trim_end_matches('\r');
    let at = match location.column {
        None => start + line.len(),
        Some(column) => {
            let characters = line.chars().count();
            if column > characters + 1 {
                bail!(
                    "line {} of {} has {characters} characters, so there is no column {column}",
                    location.line,
                    location.path
                );
            }
            start
                + line
                    .char_indices()
                    .nth(column - 1)
                    .map_or(line.len(), |(at, _)| at)
        }
    };
    Ok(text.split_at(at))
}

/// `path` under `workspace`; nothing outside it is read.
fn inside_workspace(workspace: &Path, path: &str) -> Result<PathBuf> {
    let root = workspace
        .canonicalize()
        .with_context(|| format!("reading the workspace {}", workspace.display()))?;
    let full = root
        .join(path)
        .canonicalize()
        .with_context(|| format!("reading {path}"))?;
    if !full.starts_with(&root) {
        bail!("{path} is outside the workspace");
    }
    Ok(full)
}

/// The first `limit` bytes of `text`, ending on a whole line where there is one.
fn head_of(text: &str, limit: usize) -> &str {
    if text.len() <= limit {
        return text;
    }
    let mut end = limit;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    match text[..end].rfind('\n') {
        Some(newline) => &text[..=newline],
        None => &text[..end],
    }
}

/// The files the knowledge graph couples most strongly to `path`, as
/// `input_extra` chunks. Empty until the graph has been built.
fn related_files(
    workspace: &Path,
    path: &str,
    graph: &Arc<Mutex<Option<GraphStore>>>,
) -> Vec<Value> {
    let related = match graph.lock() {
        Ok(store) => store
            .as_ref()
            .map_or(Vec::new(), |store| store.predictive_group_vectors(path)),
        Err(_) => Vec::new(),
    };
    related
        .into_iter()
        .filter(|file| file != path)
        .filter_map(|file| {
            let full = inside_workspace(workspace, &file).ok()?;
            let text = std::fs::read_to_string(full).ok()?;
            Some(json!({"filename": file, "text": head_of(&text, CONTEXT_FILE_BYTES)}))
        })
        .take(CONTEXT_FILES)
        .collect()
}

/// The body of the request, with `model` for a coordinator to route on.
fn request_body(
    model: &str,
    path: &str,
    prefix: &str,
    suffix: &str,
    input_extra: Vec<Value>,
) -> Value {
    json!({
        "model": model,
        "filename": path,
        "input_prefix": prefix,
        "input_suffix": suffix,
        "input_extra": input_extra,
        "temperature": 0,
    })
}

/// Asks the server what belongs at `location`. The answer is what to show: the
/// place, and the completion.
pub(crate) async fn complete_at(
    profile: &LlmConfiguration,
    workspace: &Path,
    location: &Location,
    graph: &Arc<Mutex<Option<GraphStore>>>,
) -> Result<String> {
    let full = inside_workspace(workspace, &location.path)?;
    let text =
        std::fs::read_to_string(&full).with_context(|| format!("reading {}", location.path))?;
    let (prefix, suffix) = split_at_cursor(&text, location)?;
    let body = request_body(
        &profile.model,
        &location.path,
        prefix,
        suffix,
        related_files(workspace, &location.path, graph),
    );
    let url = format!("{}/infill", normalized_openai_endpoint(&profile.endpoint));
    let mut request = reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()?
        .post(&url)
        .json(&body);
    if let Some(api_key) = &profile.api_key {
        request = request.bearer_auth(api_key);
    }
    let response = request
        .send()
        .await
        .with_context(|| format!("asking {url}"))?;
    let status = response.status();
    let answer = response.text().await?;
    if !status.is_success() {
        return Err(anyhow!("{}: {}", status.as_u16(), answer.trim()));
    }
    let answer: Value =
        serde_json::from_str(&answer).with_context(|| format!("reading what {url} answered"))?;
    let completion = answer["content"].as_str().unwrap_or_default();
    Ok(if completion.trim().is_empty() {
        format!("The model has nothing to add at {}.", describe(location))
    } else {
        format!("At {}:\n\n{completion}", describe(location))
    })
}

fn describe(location: &Location) -> String {
    match location.column {
        Some(column) => format!("{}:{}:{column}", location.path, location.line),
        None => format!("{}:{}", location.path, location.line),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(line: usize, column: Option<usize>) -> Location {
        Location {
            path: "f".to_string(),
            line,
            column,
        }
    }

    #[test]
    fn a_location_is_a_path_a_line_and_maybe_a_column() {
        let location = |path: &str, line, column| {
            Some(Location {
                path: path.to_string(),
                line,
                column,
            })
        };
        assert_eq!(parse_location("a.rs:3"), location("a.rs", 3, None));
        assert_eq!(parse_location("a.rs:3:7"), location("a.rs", 3, Some(7)));
        assert_eq!(
            parse_location(" dir/a b.rs:3 "),
            location("dir/a b.rs", 3, None)
        );
        assert_eq!(
            parse_location("c:weird.rs:2"),
            location("c:weird.rs", 2, None)
        );
        for refused in ["", "a.rs", "a.rs:", "a.rs:x", "a.rs:0", ":3", "a.rs:1:0"] {
            assert_eq!(parse_location(refused), None, "{refused}");
        }
    }

    #[test]
    fn the_cursor_cuts_the_text_where_it_says() {
        let text = "one\ntwo\nthree\n";
        assert_eq!(
            split_at_cursor(text, &at(1, None)).unwrap(),
            ("one", "\ntwo\nthree\n")
        );
        assert_eq!(
            split_at_cursor(text, &at(2, Some(1))).unwrap(),
            ("one\n", "two\nthree\n")
        );
        assert_eq!(
            split_at_cursor(text, &at(2, Some(2))).unwrap(),
            ("one\nt", "wo\nthree\n")
        );
        assert_eq!(
            split_at_cursor(text, &at(2, Some(4))).unwrap(),
            ("one\ntwo", "\nthree\n")
        );
        assert_eq!(
            split_at_cursor(text, &at(3, None)).unwrap(),
            ("one\ntwo\nthree", "\n")
        );
    }

    /// A column counts characters, and an end of line can be two bytes.
    #[test]
    fn the_cursor_counts_characters_and_stays_before_a_carriage_return() {
        assert_eq!(
            split_at_cursor("héllo", &at(1, Some(3))).unwrap(),
            ("hé", "llo")
        );
        assert_eq!(
            split_at_cursor("ab\r\ncd", &at(1, None)).unwrap(),
            ("ab", "\r\ncd")
        );
    }

    #[test]
    fn a_place_the_file_does_not_have_is_refused_by_name() {
        let err = split_at_cursor("one\n", &at(5, None))
            .unwrap_err()
            .to_string();
        assert!(err.contains("no line 5"), "{err}");
        let err = split_at_cursor("one\n", &at(1, Some(9)))
            .unwrap_err()
            .to_string();
        assert!(err.contains("no column 9"), "{err}");
    }

    #[test]
    fn nothing_outside_the_workspace_is_read() {
        let dir = tempfile::tempdir().unwrap();
        let workspace = dir.path().join("w");
        std::fs::create_dir(&workspace).unwrap();
        std::fs::write(workspace.join("in.rs"), "x").unwrap();
        std::fs::write(dir.path().join("out.rs"), "x").unwrap();
        assert!(inside_workspace(&workspace, "in.rs").is_ok());
        let err = inside_workspace(&workspace, "../out.rs")
            .unwrap_err()
            .to_string();
        assert!(err.contains("outside the workspace"), "{err}");
    }

    #[test]
    fn a_files_head_ends_on_a_whole_line_and_a_whole_character() {
        assert_eq!(head_of("short", 100), "short");
        assert_eq!(head_of("ab\ncd\nef", 7), "ab\ncd\n");
        assert_eq!(head_of("ééé", 3), "é");
    }

    /// The coordinator routes on `model`, and the server reads the rest.
    #[test]
    fn the_request_has_what_the_endpoint_reads() {
        let body = request_body(
            "code",
            "a.rs",
            "p",
            "s",
            vec![json!({"filename": "b", "text": "t"})],
        );
        assert_eq!(body["model"], "code");
        assert_eq!(body["filename"], "a.rs");
        assert_eq!(body["input_prefix"], "p");
        assert_eq!(body["input_suffix"], "s");
        assert_eq!(body["input_extra"][0]["filename"], "b");
    }

    /// With no graph built there is simply no extra context.
    #[test]
    fn no_graph_means_no_related_files() {
        let graph = Arc::new(Mutex::new(None));
        assert!(related_files(Path::new("."), "a.rs", &graph).is_empty());
    }
}
