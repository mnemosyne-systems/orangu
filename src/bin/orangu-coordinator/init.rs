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

//! Interactive `--init` flow that writes `~/.orangu/orangu-coordinator.conf`.
//!
//! It walks every `[orangu-coordinator]` option, showing its default, then
//! asks for a model and a port for each role in turn. `all` is mandatory —
//! it's the fallback profile a loaded config must always have — the rest
//! (`code`, `review`, `explorer`, `embeddings`) are skipped by leaving the
//! model prompt blank. Each role that gets a model becomes its own section,
//! named after the role, and its own `orangu-server` (see `process::
//! Coordinator::start`).

use crate::config::{
    default_host, default_max_body_bytes, default_port, default_profile_port,
    default_startup_timeout,
};
use anyhow::{Context, Result, anyhow};
use rustyline::{
    Config, Context as RlContext, Editor, Helper,
    completion::{Completer, FilenameCompleter, Pair},
    error::ReadlineError,
    highlight::Highlighter,
    hint::Hinter,
    history::DefaultHistory,
    validate::Validator,
};
use std::borrow::Cow;
use std::io::{self, Write};

/// Roles offered after the mandatory `all`, in the order `orangu.conf` itself
/// documents them.
const OPTIONAL_ROLES: &[&str] = &["code", "review", "explorer", "embeddings"];

/// Grey ANSI truecolor used for inline ghost-text hints (`DirCompleter`'s
/// own `highlight_hint`) — the same color `src/tui/screen.rs`'s own
/// `GHOST_TEXT` uses for orangu's main chat REPL, duplicated here rather
/// than exported from there since it's a one-line constant and each
/// `--init` wizard is already its own self-contained binary (see
/// `OptionCompleter`'s doc comment in `orangu-server`'s own `init.rs` for
/// the same reasoning applied to a different helper).
const GHOST_TEXT: &str = "\x1b[38;2;120;120;120m";
const ANSI_RESET: &str = "\x1b[0m";

pub async fn run_init() -> Result<()> {
    println!("orangu-coordinator configuration");
    println!("=================================\n");

    let host = prompt_with_default("host", &default_host())?;
    let port = prompt_number::<u16>("port", default_port())?;
    let models = prompt_models_dir("models")?;
    let startup_timeout = prompt_number::<u64>("startup_timeout", default_startup_timeout())?;
    let max_body_bytes = prompt_number::<usize>("max_body_bytes", default_max_body_bytes())?;
    let idle_timeout = prompt_optional_number::<u64>("idle_timeout")?;
    let shutdown_token = prompt_optional_string("shutdown_token")?;

    let model_options = orangu::model_spec::scan_models_dir(std::path::Path::new(&models))
        .map(|found| model_completion_options(&orangu::model_spec::group_models(&found)))
        .unwrap_or_default();

    let (all_model, all_host, all_port) = prompt_required_role("all", &model_options)?;
    let mut roles = vec![("all".to_string(), all_model, all_host, all_port)];
    for role in OPTIONAL_ROLES {
        if let Some((model, host, port)) = prompt_optional_role(role, &model_options)? {
            roles.push((role.to_string(), model, host, port));
        }
    }

    let contents = render_config(
        &host,
        port,
        &models,
        startup_timeout,
        max_body_bytes,
        idle_timeout,
        shutdown_token.as_deref(),
        &roles,
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
    let path = dir.join("orangu-coordinator.conf");
    std::fs::write(&path, contents)
        .with_context(|| format!("failed to write {}", path.display()))?;
    println!("Wrote {}", path.display());

    Ok(())
}

/// Renders `orangu-coordinator.conf`'s contents from `--init`'s collected
/// answers. `host`/`port` are always written, even at their default — the
/// two values someone skimming the file most wants to see at a glance for
/// what's otherwise just a proxy address. `models` is always written too,
/// but for a different reason: it has no default to compare against at all
/// (see `prompt_models_dir`'s own doc comment), so there is no "matches the
/// default" case to omit. Every other value here — including every
/// per-profile `role`/`host`/`port` — is left out entirely when it matches
/// what the loader would already default to on its own
/// (`load_coordinator_configuration` applies the exact same defaults back),
/// so omitting them changes nothing about how the written file behaves,
/// only how much of it a reader has to look at. `model` is the one
/// per-profile exception, always written since — like `models` above — it
/// has no default.
///
/// Pulled out of `run_init` itself so this terseness logic is directly
/// unit-testable without needing to fake an interactive rustyline session.
#[allow(clippy::too_many_arguments)]
fn render_config(
    host: &str,
    port: u16,
    models: &str,
    startup_timeout: u64,
    max_body_bytes: usize,
    idle_timeout: Option<u64>,
    shutdown_token: Option<&str>,
    roles: &[(String, String, String, u16)],
) -> String {
    let mut client = vec![format!("host = {host}"), format!("port = {port}")];
    client.push(format!("models = {models}"));
    if startup_timeout != default_startup_timeout() {
        client.push(format!("startup_timeout = {startup_timeout}"));
    }
    if max_body_bytes != default_max_body_bytes() {
        client.push(format!("max_body_bytes = {max_body_bytes}"));
    }
    if let Some(t) = idle_timeout {
        client.push(format!("idle_timeout = {t}"));
    }
    if let Some(tok) = shutdown_token {
        client.push(format!("shutdown_token = {tok}"));
    }

    let mut contents = format!("[orangu-coordinator]\n{}\n", client.join("\n"));
    for (role, model, host, port) in roles {
        let mut section = format!("\n[{role}]\n");
        if role.as_str() != "all" {
            section.push_str(&format!("role = {role}\n"));
        }
        section.push_str(&format!("model = {model}\n"));
        if host != &default_host() {
            section.push_str(&format!("host = {host}\n"));
        }
        if *port != default_profile_port() {
            section.push_str(&format!("port = {port}\n"));
        }
        contents.push_str(&section);
    }
    contents
}

/// Prompts for the mandatory `all` role's model, host, and port,
/// re-prompting on an empty model or an invalid port. `options` (see
/// `model_completion_options`) drives the model prompt's ghost-text/TAB
/// completion. `host`/`port` both fall back to the same defaults
/// `CoordinatorLlmEntry::host`/`port` themselves default to when a config
/// omits them (`127.0.0.1`/`8100`) — sharing those defaults across every
/// role is fine, not a footgun, since only one profile's `orangu-server` is
/// ever active at a time (see `CoordinatorLlmEntry::port`'s own doc
/// comment).
fn prompt_required_role(role: &str, options: &[String]) -> Result<(String, String, u16)> {
    let model = loop {
        let value = prompt_model_line(&format!("model/{role}: "), options)?;
        if value.is_empty() {
            println!("A model is required for the mandatory 'all' role.");
            continue;
        }
        break value;
    };
    let host = prompt_with_default(&format!("host/{role}"), &default_host())?;
    let port = prompt_number::<u16>(&format!("port/{role}"), default_profile_port())?;
    Ok((model, host, port))
}

/// Prompts for an optional role's model, host, and port. A blank model
/// entry skips the role entirely (`Ok(None)`); a non-blank model always
/// continues on to `host`/`port`. `options` drives the model prompt's
/// ghost-text/TAB completion, same as [`prompt_required_role`].
fn prompt_optional_role(role: &str, options: &[String]) -> Result<Option<(String, String, u16)>> {
    let value = prompt_model_line(&format!("model/{role} []: "), options)?;
    if value.is_empty() {
        return Ok(None);
    }
    let host = prompt_with_default(&format!("host/{role}"), &default_host())?;
    let port = prompt_number::<u16>(&format!("port/{role}"), default_profile_port())?;
    Ok(Some((value, host, port)))
}

/// Read a line from stdin after printing `label`. A closed stdin (EOF, e.g.
/// Ctrl-D) is reported as an error rather than an empty line, so callers abort
/// instead of looping forever or silently accepting every default.
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

/// Prompt showing `default` in brackets; an empty entry keeps the default.
fn prompt_with_default(label: &str, default: &str) -> Result<String> {
    let value = prompt(&format!("{label} [{default}]: "))?;
    if value.is_empty() {
        Ok(default.to_string())
    } else {
        Ok(value)
    }
}

/// TAB-completes filesystem paths (via `FilenameCompleter`) for the
/// `models` prompt, and — the same underlying candidates — shows the first
/// match as a greyed-out inline ghost-text suggestion while typing, so a
/// user can see (and Right-Arrow-accept, or Tab-cycle) an existing
/// directory under the current path without needing to press Tab first.
/// Mirrors `orangu-server`'s own `DirCompleter` (`src/bin/orangu-server/
/// init.rs`) for TAB completion, duplicated here per that struct's own doc
/// comment's reasoning (each `--init` wizard is a separate, self-contained
/// binary) — but adds a real `hint()`/`highlight_hint()` body, which
/// nothing in this codebase's existing rustyline helpers does yet.
struct DirCompleter {
    inner: FilenameCompleter,
}

impl Completer for DirCompleter {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        ctx: &RlContext<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        self.inner.complete(line, pos, ctx)
    }
}

impl Hinter for DirCompleter {
    type Hint = String;

    fn hint(&self, line: &str, pos: usize, ctx: &RlContext<'_>) -> Option<String> {
        // Only hint when the cursor is at the end of the line — matching
        // `rustyline`'s own `HistoryHinter` convention: a hint previewing
        // what comes *after* the cursor makes no sense while editing
        // earlier in the middle of an already-typed path.
        if pos < line.len() {
            return None;
        }
        let (start, candidates) = self.inner.complete(line, pos, ctx).ok()?;
        let candidate = candidates.first()?;
        let typed = &line[start..pos];
        candidate
            .replacement
            .strip_prefix(typed)
            .filter(|suffix| !suffix.is_empty())
            .map(str::to_string)
    }
}

impl Highlighter for DirCompleter {
    fn highlight_hint<'h>(&self, hint: &'h str) -> Cow<'h, str> {
        Cow::Owned(format!("{GHOST_TEXT}{hint}{ANSI_RESET}"))
    }
}
impl Validator for DirCompleter {}
impl Helper for DirCompleter {}

/// Prompts for a value that must not be left blank, re-prompting on an
/// empty entry — used for `models`, which unlike `orangu-server`'s own
/// `--init` has no single obviously-right directory to *default* to since
/// it's shared across every profile (so Enter on an empty line still
/// re-prompts rather than silently accepting a guess) — but still benefits
/// from the same filesystem ghost-text/TAB-completion `orangu-server`'s
/// `models` prompt has, once the user starts typing a real path.
fn prompt_models_dir(label: &str) -> Result<String> {
    let config = Config::builder()
        .completion_type(rustyline::CompletionType::List)
        .build();
    let mut editor: Editor<DirCompleter, DefaultHistory> = Editor::with_config(config)?;
    editor.set_helper(Some(DirCompleter {
        inner: FilenameCompleter::new(),
    }));

    loop {
        let value = match editor.readline(&format!("{label}: ")) {
            Ok(line) => line.trim().to_string(),
            Err(ReadlineError::Eof | ReadlineError::Interrupted) => {
                return Err(anyhow!("aborted: reached end of input"));
            }
            Err(err) => return Err(err.into()),
        };
        if !value.is_empty() {
            return Ok(value);
        }
        println!("A value is required.");
    }
}

/// TAB-completes a role's `model` prompt over the models already installed
/// under the shared `models` directory — every `NR` *and* every `MODEL`
/// label, matched against the whole typed line case-insensitively — and
/// ghost-suggests the first matching option's remainder while typing.
/// Mirrors `orangu-server`'s own `OptionCompleter` (`src/bin/orangu-server/
/// init.rs`) for TAB completion (duplicated per that struct's own doc
/// comment's reasoning), but — like this file's own `DirCompleter` — adds a
/// real `hint()`/`highlight_hint()` body.
struct ModelCompleter {
    options: Vec<String>,
}

impl Completer for ModelCompleter {
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

impl Hinter for ModelCompleter {
    type Hint = String;

    fn hint(&self, line: &str, pos: usize, _ctx: &RlContext<'_>) -> Option<String> {
        // Same "only at the end of the line" rule as `DirCompleter::hint`.
        if pos < line.len() {
            return None;
        }
        let prefix = line.to_lowercase();
        let candidate = self
            .options
            .iter()
            .find(|option| option.to_lowercase().starts_with(&prefix))?;
        candidate
            .get(line.len()..)
            .map(str::to_string)
            .filter(|suffix| !suffix.is_empty())
    }
}

impl Highlighter for ModelCompleter {
    fn highlight_hint<'h>(&self, hint: &'h str) -> Cow<'h, str> {
        Cow::Owned(format!("{GHOST_TEXT}{hint}{ANSI_RESET}"))
    }
}
impl Validator for ModelCompleter {}
impl Helper for ModelCompleter {}

/// Turns `group_models`'s output into TAB-completion/ghost-text candidates:
/// every group's `MODEL` label (the same user-facing Hugging Face-style
/// name, e.g. `unsloth/gemma-4-E2B-it-GGUF:Q4_K_M`, `orangu-server list`
/// prints in its `MODEL` column) — unlike `orangu-server`'s own `--init`
/// wizard (`model_completion_options` in `src/bin/orangu-server/init.rs`),
/// this deliberately does **not** also offer each group's `NR` shorthand.
/// `orangu-server`'s `model` key is re-resolved fresh every time that one
/// process starts, so a stale `NR` only risks pointing at the wrong model
/// within a single, already-running deployment; a coordinator profile's
/// `model` is written once into `orangu-coordinator.conf` and read back
/// indefinitely — and it's also the literal string clients match against
/// (`process::match_hint`) — so a scan-order-dependent `NR` baked in here
/// would silently start resolving to a *different* model the moment the
/// `models` directory's contents change, with no error either way. The
/// label alone is a stable identifier either use is safe with.
fn model_completion_options(groups: &[orangu::model_spec::ModelGroup]) -> Vec<String> {
    groups.iter().map(|group| group.label.clone()).collect()
}

/// Reads one line for a role's `model` prompt, ghost-texting/TAB-completing
/// over `options` — freely accepts anything typed regardless of whether it
/// matches (a local path or a not-yet-downloaded `<user>/<model>[:quant]`
/// Hugging Face spec is equally valid; `options` only *assists* typing, it
/// never constrains it).
fn prompt_model_line(label: &str, options: &[String]) -> Result<String> {
    let config = Config::builder()
        .completion_type(rustyline::CompletionType::List)
        .build();
    let mut editor: Editor<ModelCompleter, DefaultHistory> = Editor::with_config(config)?;
    editor.set_helper(Some(ModelCompleter {
        options: options.to_vec(),
    }));

    match editor.readline(label) {
        Ok(line) => Ok(line.trim().to_string()),
        Err(ReadlineError::Eof | ReadlineError::Interrupted) => {
            Err(anyhow!("aborted: reached end of input"))
        }
        Err(err) => Err(err.into()),
    }
}

/// Prompt for a value that must parse as `T` (e.g. a `u64`/`u16`/`usize`),
/// re-prompting on anything that does not. An empty entry keeps `default`.
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
            Err(_) => println!("'{value}' is not a valid number."),
        }
    }
}

/// Prompt for an optional value that must parse as `T` (e.g. a `u64`),
/// re-prompting on anything that does not. An empty entry returns `None`.
fn prompt_optional_number<T>(label: &str) -> Result<Option<T>>
where
    T: std::str::FromStr + std::fmt::Display,
{
    loop {
        let value = prompt(&format!("{label} [none]: "))?;
        if value.is_empty() {
            return Ok(None);
        }
        match value.parse::<T>() {
            Ok(parsed) => return Ok(Some(parsed)),
            Err(_) => println!("'{value}' is not a valid number."),
        }
    }
}

/// Prompt for an optional string value. An empty entry returns `None`.
fn prompt_optional_string(label: &str) -> Result<Option<String>> {
    let value = prompt(&format!("{label} [none]: "))?;
    if value.is_empty() {
        Ok(None)
    } else {
        Ok(Some(value))
    }
}

/// Prompt for a Yes/No value, accepting `Yes`/`Y`/`No`/`N` case-insensitively.
/// An empty entry keeps `default`.
fn prompt_bool(label: &str, default: bool) -> Result<bool> {
    let default_label = if default { "Yes" } else { "No" };
    loop {
        let value = prompt(&format!("{label} (Yes/No) [{default_label}]: "))?;
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
    use super::*;
    use rustyline::history::DefaultHistory;

    /// `host`/`port`/`models` are always written, and nothing else is, when
    /// every other answer matches its own default — the minimal case
    /// `render_config`'s own doc comment describes.
    #[test]
    fn omits_every_value_that_matches_its_default() {
        let roles = vec![(
            "all".to_string(),
            "org/gemma".to_string(),
            default_host(),
            default_profile_port(),
        )];
        let contents = render_config(
            &default_host(),
            default_port(),
            "/srv/models",
            default_startup_timeout(),
            default_max_body_bytes(),
            None,
            None,
            &roles,
        );
        assert_eq!(
            contents,
            format!(
                "[orangu-coordinator]\nhost = {}\nport = {}\nmodels = /srv/models\n\n[all]\nmodel = org/gemma\n",
                default_host(),
                default_port()
            )
        );
    }

    /// `host`/`port` are still written even when they exactly match their
    /// own default — unlike everything else, they're never omitted.
    #[test]
    fn always_writes_host_and_port_even_at_their_default() {
        let contents = render_config(
            &default_host(),
            default_port(),
            "/srv/models",
            default_startup_timeout(),
            default_max_body_bytes(),
            None,
            None,
            &[],
        );
        assert!(contents.contains(&format!("host = {}\n", default_host())));
        assert!(contents.contains(&format!("port = {}\n", default_port())));
    }

    /// Every value that differs from its default is written — the
    /// complement of `omits_every_value_that_matches_its_default`.
    #[test]
    fn writes_every_value_that_differs_from_its_default() {
        let roles = vec![
            (
                "all".to_string(),
                "org/gemma".to_string(),
                "192.168.1.1".to_string(),
                9999,
            ),
            (
                "explorer".to_string(),
                "org/qwen".to_string(),
                default_host(),
                default_profile_port(),
            ),
        ];
        let contents = render_config(
            "0.0.0.0",
            9100,
            "/srv/models",
            60,
            1024,
            Some(300),
            Some("secret"),
            &roles,
        );
        assert!(contents.contains("host = 0.0.0.0\n"));
        assert!(contents.contains("port = 9100\n"));
        assert!(contents.contains("models = /srv/models\n"));
        assert!(contents.contains("startup_timeout = 60\n"));
        assert!(contents.contains("max_body_bytes = 1024\n"));
        assert!(contents.contains("idle_timeout = 300\n"));
        assert!(contents.contains("shutdown_token = secret\n"));
        assert!(contents.contains("[all]\n"));
        assert!(contents.contains("model = org/gemma\n"));
        assert!(contents.contains("host = 192.168.1.1\n"));
        assert!(contents.contains("port = 9999\n"));
        assert!(contents.contains("[explorer]\nrole = explorer\nmodel = org/qwen\n"));
        // `all`'s own role and `explorer`'s own host/port all match their
        // defaults and must not be written.
        assert!(!contents.contains("role = all"));
        assert!(!contents.contains(&format!("host = {}", default_host())));
        assert!(!contents.contains(&format!("port = {}\n", default_profile_port())));
    }

    fn hinter() -> DirCompleter {
        DirCompleter {
            inner: FilenameCompleter::new(),
        }
    }

    /// Typing a real subdirectory's prefix ghost-suggests the rest of its
    /// name — the exact scenario the `models` prompt needs: point at a
    /// directory tree and see (without pressing TAB) what's actually there.
    /// `FilenameCompleter` appends a trailing separator to directory
    /// candidates, which carries through into the hint — a small extra cue
    /// that what's suggested is itself a directory. It uses the host's own
    /// separator, so what the hint ends with is `/` here and `\` on Windows.
    #[test]
    fn hints_the_remainder_of_a_matching_directory_entry() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("gguf-models")).unwrap();
        let history = DefaultHistory::new();
        let ctx = RlContext::new(&history);

        let prefix = dir.path().join("gguf-mod");
        let line = prefix.to_str().unwrap();
        let hint = hinter().hint(line, line.len(), &ctx);
        let expected = format!("els{}", std::path::MAIN_SEPARATOR);
        assert_eq!(hint.as_deref(), Some(expected.as_str()));
    }

    /// No hint once the typed text already exactly matches the only
    /// candidate (trailing separator included) — there's nothing left to
    /// suggest.
    #[test]
    fn no_hint_once_the_entry_is_fully_typed() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("models")).unwrap();
        let history = DefaultHistory::new();
        let ctx = RlContext::new(&history);

        let prefix = dir.path().join("models");
        // The host's own separator, so this really is "already fully typed"
        // rather than a path the completer simply fails to match.
        let line = format!("{}{}", prefix.to_str().unwrap(), std::path::MAIN_SEPARATOR);
        let hint = hinter().hint(&line, line.len(), &ctx);
        assert_eq!(hint, None);
    }

    /// A hint previews what comes *after* the cursor, so editing in the
    /// middle of an already-typed path (cursor not at the end) must never
    /// show one — matching `rustyline`'s own `HistoryHinter` convention.
    #[test]
    fn no_hint_when_the_cursor_is_not_at_the_end_of_the_line() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("gguf-models")).unwrap();
        let history = DefaultHistory::new();
        let ctx = RlContext::new(&history);

        let prefix = dir.path().join("gguf-mod");
        let line = prefix.to_str().unwrap();
        let hint = hinter().hint(line, line.len() - 1, &ctx);
        assert_eq!(hint, None);
    }

    /// `highlight_hint` wraps the raw hint text in the same grey truecolor
    /// escape (and reset) `src/tui/screen.rs`'s own ghost text uses
    /// elsewhere in the app, for a visually consistent "suggestion, not
    /// real input" look.
    #[test]
    fn highlight_hint_wraps_the_text_in_grey() {
        let highlighted = hinter().highlight_hint("els");
        assert_eq!(highlighted, format!("{GHOST_TEXT}els{ANSI_RESET}"));
    }

    fn group(label: &str) -> orangu::model_spec::ModelGroup {
        orangu::model_spec::ModelGroup {
            label: label.to_string(),
            size_bytes: 0,
            quantization: None,
            errors: Vec::new(),
            representative_path: std::path::PathBuf::new(),
            paths: Vec::new(),
            hf_repo: None,
            local_commit: None,
        }
    }

    /// Only each group's `MODEL` label is offered — no `NR` shorthand (see
    /// `model_completion_options`'s own doc comment for why a coordinator
    /// profile deliberately excludes it, unlike `orangu-server`'s own
    /// `--init` wizard).
    #[test]
    fn offers_only_labels_no_nr_shorthand() {
        let groups = vec![
            group("Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M"),
            group("unsloth/gemma-4-E2B-it-GGUF:Q4_K_M"),
        ];
        assert_eq!(
            model_completion_options(&groups),
            vec![
                "Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M".to_string(),
                "unsloth/gemma-4-E2B-it-GGUF:Q4_K_M".to_string(),
            ]
        );
    }

    #[test]
    fn empty_groups_give_no_completion_options() {
        assert!(model_completion_options(&[]).is_empty());
    }

    fn model_hinter() -> ModelCompleter {
        ModelCompleter {
            options: model_completion_options(&[
                group("unsloth/gemma-4-E2B-it-GGUF:Q4_K_M"),
                group("unsloth/gemma-4-E4B-it-GGUF:Q4_K_M"),
            ]),
        }
    }

    /// Typing a prefix of an installed model's user-facing Hugging Face
    /// label — the same label `orangu-server list`'s `MODEL` column prints
    /// — ghost-suggests the rest of it, matching case-insensitively against
    /// the whole typed line (not just a filesystem path segment, unlike
    /// `DirCompleter`).
    #[test]
    fn hints_the_remainder_of_a_matching_model_label() {
        let ctx_history = DefaultHistory::new();
        let ctx = RlContext::new(&ctx_history);
        let line = "unsloth/gemma-4-E2B";
        let hint = model_hinter().hint(line, line.len(), &ctx);
        assert_eq!(hint.as_deref(), Some("-it-GGUF:Q4_K_M"));
    }

    /// An empty line ghost-suggests the first offered label outright — and,
    /// since `NR` shorthand is deliberately excluded from the option list
    /// (see `model_completion_options`'s own doc comment), that first
    /// suggestion is always a real, stable model label, never a bare
    /// number.
    #[test]
    fn hints_the_first_label_on_an_empty_line() {
        let ctx_history = DefaultHistory::new();
        let ctx = RlContext::new(&ctx_history);
        let hint = model_hinter().hint("", 0, &ctx);
        assert_eq!(hint.as_deref(), Some("unsloth/gemma-4-E2B-it-GGUF:Q4_K_M"));
    }

    /// No hint once the cursor isn't at the end of the line, and no hint
    /// when nothing typed matches any option — same guarantees
    /// `DirCompleter::hint` makes.
    #[test]
    fn no_hint_when_cursor_is_mid_line_or_nothing_matches() {
        let ctx_history = DefaultHistory::new();
        let ctx = RlContext::new(&ctx_history);
        let line = "unsloth/gemma-4-E2B";
        assert_eq!(model_hinter().hint(line, line.len() - 1, &ctx), None);
        assert_eq!(model_hinter().hint("nonexistent/model", 17, &ctx), None);
    }

    #[test]
    fn model_completer_highlight_hint_wraps_the_text_in_grey() {
        let highlighted = model_hinter().highlight_hint("-it-GGUF:Q4_K_M");
        assert_eq!(
            highlighted,
            format!("{GHOST_TEXT}-it-GGUF:Q4_K_M{ANSI_RESET}")
        );
    }
}
