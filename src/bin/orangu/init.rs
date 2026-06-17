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
        CLIENT_SECTION, DEFAULT_PLATFORM, default_code_max_tokens, default_llm_max_tool_rounds,
        default_review_max_tokens, default_timeout, default_virtual_width,
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

/// Valid values for `[orangu].workspaces` (the workspace tab placement), offered
/// for completion and validated. Must stay in step with the placements the
/// loader accepts (see `orangu::workspaces::WorkspacePlacement`).
const WORKSPACE_OPTIONS: &[&str] = &["top", "bottom", "left", "right"];

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
    let review_max_tokens = prompt_number::<u32>("review_max_tokens", default_review_max_tokens())?;
    let code_max_tokens = prompt_number::<u32>("code_max_tokens", default_code_max_tokens())?;
    let quotes = prompt_with_options("quotes", "none", QUOTE_OPTIONS)?;
    let width = prompt_number::<usize>("width", default_virtual_width())?;
    let banner = prompt_with_options("banner", "left", BANNER_OPTIONS)?;
    let workspaces = prompt_with_options("workspaces", "top", WORKSPACE_OPTIONS)?;
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
    if review_max_tokens != default_review_max_tokens() {
        client.push(format!("review_max_tokens = {review_max_tokens}"));
    }
    if code_max_tokens != default_code_max_tokens() {
        client.push(format!("code_max_tokens = {code_max_tokens}"));
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
    if workspaces != "top" {
        client.push(format!("workspaces = {workspaces}"));
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

    report_optional_tools(&platform);

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

/// Print whether each optional external tool documented under "Optional
/// external tools" in the manual is detected on this system, and whether it is
/// actually wired up to be used. Shown just before the configuration preview so
/// the user can see which integrations orangu will pick up:
///
/// * `git lg` for `/log` — used when the `lg` alias is set in `~/.gitconfig`.
/// * `delta` for `/diff` — used when it is the configured Git diff pager.
/// * `bat` for `/show_file` — used automatically whenever it is installed.
/// * `gh`/`glab` for the forge commands — used for the selected `platform`.
///
/// Each line reads `No` when the tool is absent, `Yes (Used)` when installed
/// and active, or `Yes (Not used)` when installed but not configured to be
/// used.
fn report_optional_tools(platform: &str) {
    let delta_installed = command_available("delta");
    let bat_installed = command_available("bat");
    let gh_installed = command_available("gh");
    let glab_installed = command_available("glab");

    println!("\nDetected optional tools:\n");
    // `git lg` has no separate binary: the alias in `~/.gitconfig` both
    // installs and activates it, so it is only ever absent or used.
    let lg = git_lg_configured();
    println!("  git lg: {}", tool_status(lg, lg));
    // delta is only used when it resolves as the Git diff pager.
    println!(
        "  delta:  {}",
        tool_status(
            delta_installed,
            delta_installed && delta_is_git_diff_pager()
        )
    );
    // bat needs no configuration; orangu uses it whenever it is installed.
    println!("  bat:    {}", tool_status(bat_installed, bat_installed));
    // gh/glab are selected by `[orangu].platform`.
    println!(
        "  gh:     {}",
        tool_status(gh_installed, gh_installed && platform == "github")
    );
    println!(
        "  glab:   {}",
        tool_status(glab_installed, glab_installed && platform == "gitlab")
    );
}

/// Format a detection result for the wizard: `No` when the tool is not
/// installed, `Yes (Used)` when installed and active, or `Yes (Not used)` when
/// installed but not configured to be used.
fn tool_status(installed: bool, used: bool) -> &'static str {
    match (installed, used) {
        (false, _) => "No",
        (true, true) => "Yes (Used)",
        (true, false) => "Yes (Not used)",
    }
}

/// Whether the `lg` Git alias is configured globally, matching the check `/log`
/// uses to decide between `git lg` and the plain `git log` fallback.
fn git_lg_configured() -> bool {
    std::process::Command::new("git")
        .args(["config", "--global", "--get", "alias.lg"])
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

/// Whether `command --version` runs successfully, used to detect an optional
/// executable on `PATH`. A missing binary (or any spawn failure) reports
/// `false`.
fn command_available(command: &str) -> bool {
    std::process::Command::new(command)
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

/// Whether `delta` is the effective Git diff pager in `~/.gitconfig`, mirroring
/// how `/diff` resolves a pager: `pager.diff` wins over `core.pager`, and an
/// interactive pager (`less`, `more`, …) is skipped so the next candidate is
/// considered. Only the global config is inspected, since the wizard runs
/// outside any particular repository.
fn delta_is_git_diff_pager() -> bool {
    for key in ["pager.diff", "core.pager"] {
        let Some(value) = git_global_config(key) else {
            continue;
        };
        let value = value.trim();
        if value.is_empty() {
            continue;
        }
        let executable = pager_executable(value);
        // An interactive pager cannot run non-interactively, so orangu ignores
        // it and falls through to the next key.
        if matches!(executable, "less" | "more" | "most" | "lv") {
            continue;
        }
        return executable == "delta";
    }
    false
}

/// Read a single value from the global Git configuration (`~/.gitconfig`),
/// returning `None` when the key is unset or Git cannot be run.
fn git_global_config(key: &str) -> Option<String> {
    let output = std::process::Command::new("git")
        .args(["config", "--global", "--get", key])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// The bare executable name of a configured pager command, dropping any
/// directory prefix and arguments (e.g. `/usr/bin/delta --side-by-side` →
/// `delta`).
fn pager_executable(command: &str) -> &str {
    let first = command.split_whitespace().next().unwrap_or(command);
    first.rsplit(['/', '\\']).next().unwrap_or(first)
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

#[cfg(test)]
mod tests {
    use super::{WORKSPACE_OPTIONS, pager_executable, tool_status};

    #[test]
    fn workspace_options_are_all_valid_placements() {
        // Every value the wizard offers (and Tab-completes) must parse as a
        // placement the loader accepts, so the wizard never writes a config the
        // loader would later reject and the two lists never drift apart.
        for option in WORKSPACE_OPTIONS {
            assert!(
                option
                    .parse::<orangu::workspaces::WorkspacePlacement>()
                    .is_ok(),
                "wizard offers an invalid workspaces value: {option}"
            );
        }
    }

    #[test]
    fn tool_status_reports_install_and_usage() {
        assert_eq!(tool_status(false, false), "No");
        assert_eq!(tool_status(true, true), "Yes (Used)");
        assert_eq!(tool_status(true, false), "Yes (Not used)");
    }

    #[test]
    fn pager_executable_strips_path_and_arguments() {
        assert_eq!(pager_executable("delta"), "delta");
        assert_eq!(pager_executable("delta --side-by-side"), "delta");
        assert_eq!(pager_executable("/usr/bin/delta --width=90"), "delta");
        assert_eq!(pager_executable("less"), "less");
    }
}
