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

//! Interactive `--init` flow that writes `~/.orangu/orangu.conf`.
//!
//! It asks for the LLM URL, auto-detects a model the server advertises, and
//! then walks every `[orangu]` and server option, showing its default. A value
//! left at its default is omitted from the file, so the generated config stays
//! minimal; only what the user changed (plus the always-required keys) is
//! written. The provider is assumed to be `llama.cpp`.

use crate::quotes::QUOTE_OPTIONS;
use anyhow::{Context, Result, anyhow};
use orangu::{
    config::{
        CLIENT_SECTION, DEFAULT_PLATFORM, default_llm_max_tool_rounds, default_timeout,
        default_virtual_width,
    },
    llm::normalized_openai_endpoint,
};
use rustyline::{
    Config, Context as RlContext, Editor, Helper,
    completion::{Completer, Pair},
    error::ReadlineError,
    highlight::Highlighter,
    hint::Hinter,
    history::DefaultHistory,
    validate::Validator,
};
use serde::Deserialize;
use std::{
    io::{self, Write},
    time::Duration,
};

/// The single server section the wizard creates; `[orangu].server` points at it.
const SERVER_SECTION: &str = "main-server";

/// Valid values for `[orangu].banner`, offered for completion and validated.
const BANNER_OPTIONS: &[&str] = &["left", "center", "right"];

/// Valid values for `[orangu].platform`, offered for completion and validated.
const PLATFORM_OPTIONS: &[&str] = &["github", "gitlab"];

/// Subset of an OpenAI-compatible `/v1/models` response, enough to pull out the
/// first advertised model id for pre-filling the `Model` prompt.
#[derive(Debug, Default, Deserialize)]
struct ModelsResponse {
    #[serde(default)]
    data: Vec<ModelEntry>,
    #[serde(default)]
    models: Vec<ModelEntry>,
}

#[derive(Debug, Deserialize)]
struct ModelEntry {
    #[serde(default)]
    id: String,
    #[serde(default)]
    model: String,
    #[serde(default)]
    name: String,
}

/// Run the interactive configuration wizard and write `~/.orangu/orangu.conf`.
pub async fn run_init() -> Result<()> {
    println!("orangu configuration");
    println!("====================\n");

    let url = prompt_required("LLM URL: ")?;

    // Ask the server which models it serves and pre-fill the first one. The
    // user can accept it or type a different identifier.
    let model = match detect_model(&url).await {
        Some(detected) => prompt_with_default("Model", &detected)?,
        None => {
            println!("Could not auto-detect a model from {url}; please enter it manually.");
            prompt_required("Model: ")?
        }
    };

    // `[orangu]` client-wide options. Each carries the default the loader would
    // apply; values left unchanged are not written to the file below. Numeric
    // and fixed-option keys are validated so the wizard never writes a config
    // the loader would later reject.
    let timeout = prompt_number::<u64>("timeout", default_timeout())?;
    let max_tool_rounds = prompt_number::<usize>("max_tool_rounds", default_llm_max_tool_rounds())?;
    let quotes = prompt_with_options("quotes", "none", QUOTE_OPTIONS)?;
    let width = prompt_number::<usize>("width", default_virtual_width())?;
    let banner = prompt_with_options("banner", "left", BANNER_OPTIONS)?;
    let feedback = prompt_bool("feedback", false)?;
    let auto_rebase = prompt_bool("auto_rebase", false)?;
    let auto_squash = prompt_bool("auto_squash", false)?;
    let terminal = prompt_with_default("terminal", "")?;
    let platform = prompt_with_options("platform", DEFAULT_PLATFORM, PLATFORM_OPTIONS)?;

    // Server-section option.
    let api_key = prompt_with_default("api_key", "")?;

    // Build the `[orangu]` section. `server` and `model` are always written;
    // every other key is added only when it differs from its default.
    let mut client = vec![
        format!("server = {SERVER_SECTION}"),
        format!("model = {model}"),
    ];
    if timeout != default_timeout() {
        client.push(format!("timeout = {timeout}"));
    }
    if max_tool_rounds != default_llm_max_tool_rounds() {
        client.push(format!("max_tool_rounds = {max_tool_rounds}"));
    }
    if quotes != "none" {
        client.push(format!("quotes = {quotes}"));
    }
    if width != default_virtual_width() {
        client.push(format!("width = {width}"));
    }
    if banner != "left" {
        client.push(format!("banner = {banner}"));
    }
    if feedback {
        client.push("feedback = on".to_string());
    }
    if auto_rebase {
        client.push("auto_rebase = on".to_string());
    }
    if auto_squash {
        client.push("auto_squash = on".to_string());
    }
    if !terminal.is_empty() {
        client.push(format!("terminal = {terminal}"));
    }
    if platform != DEFAULT_PLATFORM {
        client.push(format!("platform = {platform}"));
    }

    // Build the server section. `provider` and `endpoint` are always written;
    // the model is inherited from `[orangu].model`, so it is not repeated here.
    // `api_key` is added only when one was supplied.
    let mut server = vec![
        "provider = llama.cpp".to_string(),
        format!("endpoint = {url}"),
    ];
    if !api_key.is_empty() {
        server.push(format!("api_key = {api_key}"));
    }

    let contents = format!(
        "[{CLIENT_SECTION}]\n{}\n\n[{SERVER_SECTION}]\n{}\n",
        client.join("\n"),
        server.join("\n"),
    );

    println!("\nConfiguration to write:\n");
    println!("{contents}");

    if !prompt_bool("Write this configuration?", true)? {
        println!("Aborted. No changes written.");
        return Ok(());
    }

    let dir = home::home_dir()
        .context("failed to resolve home directory")?
        .join(".orangu");
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("failed to create directory {}", dir.display()))?;
    let path = dir.join("orangu.conf");
    std::fs::write(&path, contents)
        .with_context(|| format!("failed to write {}", path.display()))?;
    println!("Wrote {}", path.display());

    Ok(())
}

/// Query the server's `/v1/models` endpoint and return the first advertised
/// model id. Any failure (unreachable host, non-success status, empty list) is
/// reported as `None` so the caller can fall back to a manual prompt.
async fn detect_model(url: &str) -> Option<String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .ok()?;
    let endpoint = normalized_openai_endpoint(url);
    let response = client
        .get(format!("{endpoint}/v1/models"))
        .send()
        .await
        .ok()?;
    if !response.status().is_success() {
        return None;
    }
    let models = response.json::<ModelsResponse>().await.ok()?;
    models
        .data
        .iter()
        .chain(models.models.iter())
        .find_map(|entry| {
            if !entry.id.is_empty() {
                Some(entry.id.clone())
            } else if !entry.model.is_empty() {
                Some(entry.model.clone())
            } else if !entry.name.is_empty() {
                Some(entry.name.clone())
            } else {
                None
            }
        })
}

/// Read a line from stdin after printing `label`. A closed stdin (EOF, e.g.
/// Ctrl-D) is reported as an error rather than an empty line, so callers abort
/// instead of looping forever (`prompt_required`) or silently accepting every
/// default — including the final write confirmation — on truncated input.
fn prompt(label: &str) -> Result<String> {
    print!("{label}");
    io::stdout().flush()?;
    let mut line = String::new();
    let read = io::stdin()
        .read_line(&mut line)
        .context("failed to read from standard input")?;
    if read == 0 {
        return Err(anyhow!("aborted: reached end of input"));
    }
    Ok(line.trim().to_string())
}

/// Prompt until a non-empty value is entered.
fn prompt_required(label: &str) -> Result<String> {
    loop {
        let value = prompt(label)?;
        if !value.is_empty() {
            return Ok(value);
        }
        println!("A value is required.");
    }
}

/// Prompt showing `default` in brackets; an empty entry keeps the default.
fn prompt_with_default(label: &str, default: &str) -> Result<String> {
    let value = prompt(&format!("{label} [{default}]: "))?;
    if value.is_empty() {
        Ok(default.to_string())
    } else {
        Ok(value)
    }
}

/// Prompt for a value that must parse as `T` (e.g. a `u64`/`usize`), re-prompting
/// on anything that does not. EOF aborts, an empty entry keeps `default`.
fn prompt_number<T>(label: &str, default: T) -> Result<T>
where
    T: std::str::FromStr + std::fmt::Display,
{
    loop {
        let value = prompt(&format!("{label} [{default}]: "))?;
        if value.is_empty() {
            return Ok(default);
        }
        match value.parse::<T>() {
            Ok(parsed) => return Ok(parsed),
            Err(_) => println!("'{value}' is not a valid whole number."),
        }
    }
}

/// Build a line editor that offers TAB completion over `options`.
fn option_editor(options: &[&str]) -> Result<Editor<OptionCompleter, DefaultHistory>> {
    let config = Config::builder()
        .completion_type(rustyline::CompletionType::List)
        .build();
    let mut editor: Editor<OptionCompleter, DefaultHistory> = Editor::with_config(config)?;
    editor.set_helper(Some(OptionCompleter {
        options: options.iter().map(|s| s.to_string()).collect(),
    }));
    Ok(editor)
}

/// Read one trimmed line from `editor`, mapping a closed stdin (EOF / Ctrl-C)
/// to the same abort error the plain prompts raise.
fn read_line(editor: &mut Editor<OptionCompleter, DefaultHistory>, prompt: &str) -> Result<String> {
    match editor.readline(prompt) {
        Ok(line) => Ok(line.trim().to_string()),
        Err(ReadlineError::Eof | ReadlineError::Interrupted) => {
            Err(anyhow!("aborted: reached end of input"))
        }
        Err(err) => Err(err.into()),
    }
}

/// Like [`prompt_with_default`], but offers TAB completion over `options` and
/// accepts only one of them (case-insensitively), re-prompting otherwise. Used
/// for keys with a fixed set of valid values (e.g. `quotes`, `banner`,
/// `platform`). EOF aborts, an empty entry keeps `default`.
fn prompt_with_options(label: &str, default: &str, options: &[&str]) -> Result<String> {
    let mut editor = option_editor(options)?;
    loop {
        let value = read_line(&mut editor, &format!("{label} [{default}]: "))?;
        if value.is_empty() {
            return Ok(default.to_string());
        }
        if options
            .iter()
            .any(|option| option.eq_ignore_ascii_case(&value))
        {
            return Ok(value);
        }
        println!(
            "'{value}' is not valid; choose one of: {}",
            options.join(", ")
        );
    }
}

/// A rustyline helper that completes the whole line against a fixed option set,
/// matching the typed prefix case-insensitively (so `y` completes to `Yes`).
struct OptionCompleter {
    options: Vec<String>,
}

impl Completer for OptionCompleter {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &RlContext<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        let prefix = line[..pos].to_lowercase();
        let matches = self
            .options
            .iter()
            .filter(|option| option.to_lowercase().starts_with(&prefix))
            .map(|option| Pair {
                display: option.clone(),
                replacement: option.clone(),
            })
            .collect();
        Ok((0, matches))
    }
}

impl Hinter for OptionCompleter {
    type Hint = String;
}

impl Highlighter for OptionCompleter {}
impl Validator for OptionCompleter {}
impl Helper for OptionCompleter {}

/// Prompt for a Yes/No value, accepting `Yes`/`Y`/`No`/`N` case-insensitively
/// with TAB completion over `Yes`/`No`. An empty entry keeps `default`.
fn prompt_bool(label: &str, default: bool) -> Result<bool> {
    let default_label = if default { "Yes" } else { "No" };
    let mut editor = option_editor(&["Yes", "No"])?;
    loop {
        let value = read_line(
            &mut editor,
            &format!("{label} (Yes/No) [{default_label}]: "),
        )?;
        if value.is_empty() {
            return Ok(default);
        }
        match value.to_lowercase().as_str() {
            "yes" | "y" => return Ok(true),
            "no" | "n" => return Ok(false),
            _ => println!("Please answer Yes/Y or No/N."),
        }
    }
}
