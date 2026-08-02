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
    HOST_ALL, HOST_ALL_ALIAS, default_host, default_max_body_bytes, default_port,
    default_profile_port, default_startup_timeout,
};
use anyhow::{Context, Result, anyhow};
use orangu::model_spec::ModelGroup;
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
use std::net::IpAddr;
use std::path::{Path, PathBuf};

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

    let host = prompt_host("host", &default_host())?;
    let port = prompt_number::<u16>("port", default_port())?;
    let models = prompt_models_dir("models")?;
    let startup_timeout = prompt_number::<u64>("startup_timeout", default_startup_timeout())?;
    let max_body_bytes = prompt_number::<usize>("max_body_bytes", default_max_body_bytes())?;
    let idle_timeout = prompt_optional_number::<u64>("idle_timeout")?;
    let shutdown_token = prompt_optional_string("shutdown_token")?;

    let groups = orangu::model_spec::scan_models_dir(Path::new(&models))
        .map(|found| orangu::model_spec::group_models(&found))
        .unwrap_or_default();

    let (all_model, all_host, all_port) = prompt_required_role("all", &groups)?;
    let mut roles = vec![("all".to_string(), all_model, all_host, all_port)];
    for role in OPTIONAL_ROLES {
        if let Some((model, host, port)) = prompt_optional_role(role, &groups)? {
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
/// re-prompting on an empty model or an invalid port. `groups` (the models
/// already installed under the shared `models` directory) drives the model
/// prompt's ghost-text/TAB completion. `host`/`port` both fall back to the
/// same defaults `CoordinatorLlmEntry::host`/`port` themselves default to
/// when a config omits them (`all`/`8100`) — sharing those defaults across
/// every role is fine, not a footgun, since only one profile's
/// `orangu-server` is ever active at a time (see
/// `CoordinatorLlmEntry::port`'s own doc comment).
///
/// The one case that isn't asked at all is a `models` directory holding
/// exactly one model: there is nothing to choose between, so it's taken and
/// echoed as a plain `model/all: <label>` line — the same shortcut
/// `orangu-server`'s own `--init` takes (see [`sole_model`]). It applies to
/// this role only: an optional role's blank answer is how it gets skipped,
/// so filling one in automatically would configure profiles nobody asked
/// for.
fn prompt_required_role(role: &str, groups: &[ModelGroup]) -> Result<(String, String, u16)> {
    let model = match sole_model(groups) {
        // Echoed in the same `key: value` shape as the prompts around it,
        // so the transcript reads as if it had been answered.
        Some(only) => {
            let label = qualified_label(only);
            println!("model/{role}: {label}");
            label
        }
        None => loop {
            let value = prompt_model_line(&format!("model/{role}: "), groups)?;
            if value.is_empty() {
                println!("A model is required for the mandatory 'all' role.");
                continue;
            }
            break value;
        },
    };
    let host = prompt_host(&format!("host/{role}"), &default_host())?;
    let port = prompt_number::<u16>(&format!("port/{role}"), default_profile_port())?;
    Ok((model, host, port))
}

/// Prompts for an optional role's model, host, and port. A blank model
/// entry skips the role entirely (`Ok(None)`); a non-blank model always
/// continues on to `host`/`port`. `groups` drives the model prompt's
/// ghost-text/TAB completion, same as [`prompt_required_role`].
fn prompt_optional_role(
    role: &str,
    groups: &[ModelGroup],
) -> Result<Option<(String, String, u16)>> {
    let value = prompt_model_line(&format!("model/{role} []: "), groups)?;
    if value.is_empty() {
        return Ok(None);
    }
    let host = prompt_host(&format!("host/{role}"), &default_host())?;
    let port = prompt_number::<u16>(&format!("port/{role}"), default_profile_port())?;
    Ok(Some((value, host, port)))
}

/// Reads one line after printing `label`, with `rustyline`'s ordinary line
/// editing (arrow keys, Ctrl-A/E, ...) but no completion or ghost text of
/// its own — the plain-value prompts (`port`, `startup_timeout`, ...), same
/// as `orangu-server`'s own `--init` uses for those. A closed stdin (EOF,
/// e.g. Ctrl-D) — or Ctrl-C — is reported as an error rather than an empty
/// line, so callers abort instead of looping forever or silently accepting
/// every default.
fn prompt(label: &str) -> Result<String> {
    let mut editor: Editor<(), DefaultHistory> = Editor::new()?;
    match editor.readline(label) {
        Ok(line) => Ok(line.trim().to_string()),
        Err(ReadlineError::Eof | ReadlineError::Interrupted) => {
            Err(anyhow!("aborted: reached end of input"))
        }
        Err(err) => Err(err.into()),
    }
}

/// One candidate offered at a `host` prompt: `value` is what lands in the
/// config file (and what the ghost text completes to), `display` is the
/// annotated form the TAB list shows — an address on its own says nothing
/// about which interface it belongs to. Mirrors `orangu-server`'s own
/// `HostOption`/[`HostCompleter`] (`src/bin/orangu-server/init.rs`),
/// duplicated here per [`DirCompleter`]'s own reasoning: each `--init`
/// wizard is a separate, self-contained binary.
#[derive(Clone, Debug, PartialEq, Eq)]
struct HostOption {
    value: String,
    display: String,
}

/// TAB-completes (and ghost-suggests) a `host` prompt over
/// [`host_completion_options`]'s candidates: [`HOST_ALL`], its `*` alias,
/// and every address this machine's network interfaces actually have.
struct HostCompleter {
    options: Vec<HostOption>,
}

impl HostCompleter {
    /// Every candidate whose value starts with what's typed so far, shared
    /// by `complete` and `hint` so the TAB list and the ghost can never
    /// disagree about what matches.
    fn matches<'a>(&'a self, prefix: &str) -> impl Iterator<Item = &'a HostOption> + 'a {
        let prefix = prefix.to_lowercase();
        self.options
            .iter()
            .filter(move |option| option.value.to_lowercase().starts_with(&prefix))
    }
}

impl Completer for HostCompleter {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &RlContext<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        let matches = self
            .matches(&line[..pos])
            .map(|option| Pair {
                display: option.display.clone(),
                replacement: option.value.clone(),
            })
            .collect();
        Ok((0, matches))
    }
}

impl Hinter for HostCompleter {
    type Hint = String;

    fn hint(&self, line: &str, pos: usize, _ctx: &RlContext<'_>) -> Option<String> {
        // Same "only at the end of the line" rule as `DirCompleter::hint`.
        if pos < line.len() {
            return None;
        }
        let candidate = self.matches(line).next()?;
        candidate
            .value
            .get(line.len()..)
            .filter(|suffix| !suffix.is_empty())
            .map(str::to_string)
    }
}

impl Highlighter for HostCompleter {
    fn highlight_hint<'h>(&self, hint: &'h str) -> Cow<'h, str> {
        Cow::Owned(format!("{GHOST_TEXT}{hint}{ANSI_RESET}"))
    }
}
impl Validator for HostCompleter {}
impl Helper for HostCompleter {}

/// This machine's interfaces as `(name, address)` pairs, or nothing at all
/// if the platform won't say — a `host` prompt only *assists* typing, so a
/// failed enumeration costs the address candidates and nothing else
/// ([`HOST_ALL`] and its alias are added by [`host_completion_options`]
/// regardless).
fn local_interfaces() -> Vec<(String, IpAddr)> {
    if_addrs::get_if_addrs()
        .unwrap_or_default()
        .into_iter()
        .map(|interface| (interface.name.clone(), interface.ip()))
        .collect()
}

/// Orders a `host` prompt's candidates: [`HOST_ALL`] first — it's the
/// default, so it's also what ghosts on an empty line — then its `*` alias,
/// then every routable interface address (IPv4 before IPv6), and finally the
/// loopback addresses, the narrowest choice and so the least likely one to
/// be after at a prompt whose whole point is picking what to expose.
/// Duplicates collapse, keeping the earliest occurrence.
///
/// Split out from [`prompt_host`] so the ordering is unit-testable without a
/// live rustyline session or a machine with any particular interfaces.
fn host_completion_options(interfaces: &[(String, IpAddr)]) -> Vec<HostOption> {
    let mut addresses: Vec<&(String, IpAddr)> = interfaces.iter().collect();
    addresses
        .sort_by_key(|(name, ip)| (ip.is_loopback(), ip.is_ipv6(), name.clone(), ip.to_string()));

    let mut options = vec![
        HostOption {
            value: HOST_ALL.to_string(),
            display: format!("{HOST_ALL} (every network interface)"),
        },
        HostOption {
            value: HOST_ALL_ALIAS.to_string(),
            display: format!("{HOST_ALL_ALIAS} (alias for {HOST_ALL})"),
        },
    ];
    for (name, ip) in addresses {
        let value = ip.to_string();
        if options.iter().any(|option| option.value == value) {
            continue;
        }
        options.push(HostOption {
            display: format!("{value} ({name})"),
            value,
        });
    }
    options
}

/// Prompts for a `host` — the coordinator's own listen address, or a
/// profile's — ghost-texting/TAB-completing over
/// [`host_completion_options`], with `default` kept on an empty entry.
/// Anything typed is accepted as-is: a hostname the machine resolves, or an
/// address on an interface that only exists once this config is deployed
/// elsewhere, are both legitimate, and `bind` reports the ones that aren't
/// at startup.
fn prompt_host(label: &str, default: &str) -> Result<String> {
    let config = Config::builder()
        .completion_type(rustyline::CompletionType::List)
        .build();
    let mut editor: Editor<HostCompleter, DefaultHistory> = Editor::with_config(config)?;
    editor.set_helper(Some(HostCompleter {
        options: host_completion_options(&local_interfaces()),
    }));

    let value = match editor.readline(&format!("{label} [{default}]: ")) {
        Ok(line) => line.trim().to_string(),
        Err(ReadlineError::Eof | ReadlineError::Interrupted) => {
            return Err(anyhow!("aborted: reached end of input"));
        }
        Err(err) => return Err(err.into()),
    };
    Ok(if value.is_empty() {
        default.to_string()
    } else {
        value
    })
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

/// Where Hugging Face downloads land by default (`~/.cache/huggingface/hub`
/// on Linux/macOS, `%USERPROFILE%\.cache\huggingface\hub` on Windows) —
/// offered as the `models` prompt's default so pointing every profile at
/// whatever is likely already there is just pressing Enter. The same default
/// `orangu-server`'s own `--init` offers, since it is the same directory
/// both would scan.
fn huggingface_cache_dir() -> Option<PathBuf> {
    Some(
        home::home_dir()?
            .join(".cache")
            .join("huggingface")
            .join("hub"),
    )
}

/// Prompts for the shared `models` directory every profile's
/// `orangu-server` is started against, with the same filesystem
/// ghost-text/TAB completion — and the same behaviour on an answer —
/// `orangu-server`'s own `--init` `models` prompt has: [`huggingface_cache_dir`]
/// is the default an empty entry takes, and a directory that isn't there yet
/// is simply created (only a failure is worth a line, since that's the one
/// case the prompt comes back for). A machine with no resolvable home
/// directory has no default to offer, so there an empty entry re-prompts
/// instead.
fn prompt_models_dir(label: &str) -> Result<String> {
    let default_display = huggingface_cache_dir().map(|d| d.display().to_string());
    let config = Config::builder()
        .completion_type(rustyline::CompletionType::List)
        .build();
    let mut editor: Editor<DirCompleter, DefaultHistory> = Editor::with_config(config)?;
    editor.set_helper(Some(DirCompleter {
        inner: FilenameCompleter::new(),
    }));

    loop {
        let prompt_label = match &default_display {
            Some(d) => format!("{label} [{d}]: "),
            None => format!("{label} []: "),
        };
        let value = match editor.readline(&prompt_label) {
            Ok(line) => line.trim().to_string(),
            Err(ReadlineError::Eof | ReadlineError::Interrupted) => {
                return Err(anyhow!("aborted: reached end of input"));
            }
            Err(err) => return Err(err.into()),
        };
        let value = if value.is_empty() {
            match &default_display {
                Some(d) => d.clone(),
                None => {
                    println!("A value is required.");
                    continue;
                }
            }
        } else {
            value
        };
        let path = PathBuf::from(&value);
        if !path.is_dir()
            && let Err(err) = std::fs::create_dir_all(&path)
        {
            println!("failed to create '{value}': {err}");
            continue;
        }
        return Ok(value);
    }
}

/// TAB-completes a role's `model` prompt over the models already installed
/// under the shared `models` directory — every `NR` *and* every
/// `MODEL:QUANT` label ([`model_completion_options`]), matched against the
/// whole typed line case-insensitively — while ghost-suggesting from the
/// labels alone ([`model_hint_options`]). Mirrors `orangu-server`'s own
/// `ModelCompleter` (`src/bin/orangu-server/init.rs`) exactly, including
/// that split: an `NR` is a two-keystroke shorthand someone types on
/// purpose, a model name is the thing worth previewing, so an empty line
/// opens already ghosting the first installed model rather than the digit
/// `1`. Duplicated rather than shared per [`DirCompleter`]'s own reasoning.
struct ModelCompleter {
    options: Vec<String>,
    labels: Vec<String>,
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
            .labels
            .iter()
            .find(|label| label.to_lowercase().starts_with(&prefix))?;
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

/// Turns `group_models`'s output into TAB-completion candidates: `NR` (its
/// 1-based position, counted exactly as `orangu-server list` prints it)
/// immediately followed by that row's `MODEL:QUANT` ([`qualified_label`]),
/// for every group in turn — the same NR-then-MODEL pairing, in the same
/// order, `orangu-server`'s own `--init` offers.
///
/// An `NR` is offered here only as something to *type*: unlike
/// `orangu-server`'s `model` key, a coordinator profile's is also the
/// literal string clients match against (`process::match_hint`), and it is
/// read back for as long as the file lives, so a scan-order-dependent digit
/// baked into it would silently start resolving to a different model the
/// moment the `models` directory's contents change. [`resolve_model_answer`]
/// is what keeps that from happening: an `NR` answer is written out as the
/// row's own stable `MODEL:QUANT` label.
fn model_completion_options(groups: &[ModelGroup]) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    groups
        .iter()
        .enumerate()
        .flat_map(|(index, group)| [(index + 1).to_string(), qualified_label(group)])
        // Two snapshots of one repo at one quantization would still collide;
        // each distinct candidate is offered once, in first-seen order.
        // (Every such row keeps its own `NR` regardless.)
        .filter(|option| seen.insert(option.clone()))
        .collect()
}

/// What a `model` prompt ghost-suggests from: the `MODEL:QUANT` labels only,
/// in `group_models` order — so the first one is the first row
/// `orangu-server list` prints, and it's what an empty line previews.
/// Deliberately not [`model_completion_options`]'s NR-and-label
/// interleaving: its first entry is the digit `1`, which is a shorthand to
/// type, not a model to preview.
fn model_hint_options(groups: &[ModelGroup]) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    groups
        .iter()
        .map(qualified_label)
        .filter(|label| seen.insert(label.clone()))
        .collect()
}

/// A group's `MODEL` column with its `QUANT` appended —
/// `unsloth/gemma-4-E2B-it-GGUF:Q4_K_M` — which is what a `model` prompt
/// offers and what an answer gets written to the config as.
///
/// The bare label is not enough. A repo with several quantizations on disk
/// prints the same `MODEL` on every one of their rows, so offering that
/// would list one name once per quantization and — worse — write a `model =`
/// value that resolves to whichever of them comes first, rather than the one
/// picked. `<repo>:<quant>` is a spelling
/// [`orangu::model_spec::ModelGroup::matches_label`] already accepts, so it
/// resolves straight back to this exact row.
///
/// Falls back to the plain label where that spelling wouldn't resolve: a
/// model outside the Hugging Face cache layout has no repo id to qualify,
/// and one whose file says nothing about its scheme has no quantization to
/// qualify it with. Mirrors `orangu-server`'s own `qualified_label`.
fn qualified_label(group: &ModelGroup) -> String {
    match (&group.hf_repo, &group.quantization) {
        (Some(repo), Some(quant)) => format!("{repo}:{quant}"),
        _ => group.label.clone(),
    }
}

/// The model to take without asking: `Some` only when the models directory
/// holds exactly one — the same one-row-per-model grouping `orangu-server
/// list` prints, so "one model" means one *label*, however many `.gguf`
/// files (a sharded model's parts) back it. Mirrors `orangu-server`'s own
/// `sole_model`, which its `--init` uses for exactly the same shortcut.
fn sole_model(groups: &[ModelGroup]) -> Option<&ModelGroup> {
    match groups {
        [only] => Some(only),
        _ => None,
    }
}

/// What a `model` answer is actually written into the config as: an `NR` —
/// offered for typing by [`model_completion_options`], and meaningless
/// outside the scan it came from — becomes that row's own `MODEL:QUANT`
/// label ([`qualified_label`]), which names the same model for as long as
/// the file lives. Everything else is kept exactly as typed: a label, a
/// local `.gguf` path, and a `<user>/<model>[:quant]` repo that isn't
/// downloaded yet are all valid `model` values, and none of them is this
/// wizard's to rewrite.
///
/// An out-of-range `NR` (nothing installed, or a number past the last row)
/// has no row to name, so it too is left as typed — the config loader and
/// the eventual `orangu-server` start report it far better than a wizard
/// silently substituting something else could.
fn resolve_model_answer(value: &str, groups: &[ModelGroup]) -> String {
    if let Ok(nr) = value.parse::<usize>()
        && let Some(group) = nr.checked_sub(1).and_then(|index| groups.get(index))
    {
        return qualified_label(group);
    }
    value.to_string()
}

/// Reads one line for a role's `model` prompt, ghost-texting/TAB-completing
/// over the installed `groups` — freely accepts anything typed regardless of
/// whether it matches (a local path or a not-yet-downloaded
/// `<user>/<model>[:quant]` Hugging Face spec is equally valid; the
/// candidates only *assist* typing, they never constrain it). The answer
/// comes back as [`resolve_model_answer`] would write it.
fn prompt_model_line(label: &str, groups: &[ModelGroup]) -> Result<String> {
    let config = Config::builder()
        .completion_type(rustyline::CompletionType::List)
        .build();
    let mut editor: Editor<ModelCompleter, DefaultHistory> = Editor::with_config(config)?;
    editor.set_helper(Some(ModelCompleter {
        options: model_completion_options(groups),
        labels: model_hint_options(groups),
    }));

    match editor.readline(label) {
        Ok(line) => Ok(resolve_model_answer(line.trim(), groups)),
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

/// A yes/no prompt whose default is `default` — shown as `[Y/n]` or
/// `[y/N]` so which way Enter goes is visible, the same convention
/// `orangu-server`'s own `--init` (and `delete`/`refresh`'s confirmations)
/// use.
fn prompt_bool(label: &str, default: bool) -> Result<bool> {
    let hint = if default { "Y/n" } else { "y/N" };
    loop {
        let value = prompt(&format!("{label} [{hint}]: "))?.to_lowercase();
        match value.as_str() {
            "" => return Ok(default),
            "y" | "yes" => return Ok(true),
            "n" | "no" => return Ok(false),
            _ => println!("Please answer yes or no."),
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

    fn group(label: &str) -> ModelGroup {
        ModelGroup {
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

    /// A real hub-cache group, whose `MODEL` column is the bare repo and
    /// whose quantization is a column of its own.
    fn hub_group(repo: &str, quant: &str) -> ModelGroup {
        ModelGroup {
            label: repo.to_string(),
            size_bytes: 0,
            quantization: Some(quant.to_string()),
            errors: Vec::new(),
            representative_path: std::path::PathBuf::new(),
            paths: Vec::new(),
            hf_repo: Some(repo.to_string()),
            local_commit: None,
        }
    }

    /// The same `["1", "<first label>", "2", ...]` pairing `orangu-server`'s
    /// own `--init` offers — its `NR` column against its `MODEL` one, in
    /// `group_models` order.
    #[test]
    fn pairs_each_nr_with_its_label_in_group_models_order() {
        let groups = vec![
            group("Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M"),
            group("unsloth/gemma-4-E2B-it-GGUF:Q4_K_M"),
        ];
        assert_eq!(
            model_completion_options(&groups),
            vec![
                "1".to_string(),
                "Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M".to_string(),
                "2".to_string(),
                "unsloth/gemma-4-E2B-it-GGUF:Q4_K_M".to_string(),
            ]
        );
    }

    /// A repo with several quantizations on disk prints the same bare
    /// `MODEL` on every one of their rows, so each is offered — and written
    /// — as `MODEL:QUANT`, which names exactly one of them. Without the
    /// quantization, a profile's `model` would resolve to whichever row came
    /// first rather than the one picked, and the completion list would show
    /// one name three times over.
    #[test]
    fn each_quantization_is_offered_as_its_own_qualified_label() {
        let repo = "unsloth/gemma-4-E2B-it-GGUF";
        let groups = vec![
            hub_group(repo, "Q4_K_M"),
            hub_group(repo, "Q6_K"),
            hub_group(repo, "Q8_0"),
        ];

        assert_eq!(
            model_completion_options(&groups),
            vec![
                "1".to_string(),
                format!("{repo}:Q4_K_M"),
                "2".to_string(),
                format!("{repo}:Q6_K"),
                "3".to_string(),
                format!("{repo}:Q8_0"),
            ]
        );
        assert_eq!(
            model_hint_options(&groups),
            vec![
                format!("{repo}:Q4_K_M"),
                format!("{repo}:Q6_K"),
                format!("{repo}:Q8_0"),
            ]
        );
    }

    /// `<repo>:<quant>` only resolves for a model that *has* a repo and a
    /// quantization; anything else keeps the plain label, which is the
    /// spelling that does resolve for it.
    #[test]
    fn a_model_with_nothing_to_qualify_keeps_its_plain_label() {
        assert_eq!(qualified_label(&group("my-local-model")), "my-local-model");

        let mut no_quant = hub_group("user/model", "Q4_K_M");
        no_quant.quantization = None;
        assert_eq!(qualified_label(&no_quant), "user/model");
    }

    /// An `NR` is a shorthand for typing, never something to persist: a
    /// coordinator profile's `model` is read back for as long as the file
    /// lives *and* is the literal string clients match against, so the digit
    /// is written out as the row's own stable `MODEL:QUANT` label instead.
    #[test]
    fn an_nr_answer_is_written_as_its_stable_label() {
        let repo = "unsloth/gemma-4-E2B-it-GGUF";
        let groups = vec![hub_group(repo, "Q4_K_M"), hub_group(repo, "Q6_K")];
        assert_eq!(resolve_model_answer("1", &groups), format!("{repo}:Q4_K_M"));
        assert_eq!(resolve_model_answer("2", &groups), format!("{repo}:Q6_K"));
    }

    /// Everything that isn't an `NR` naming a real row is kept exactly as
    /// typed — a label, a local path, an undownloaded Hugging Face spec, and
    /// a number with no row behind it are all things this wizard has no
    /// business rewriting.
    #[test]
    fn every_other_answer_is_kept_verbatim() {
        let groups = vec![hub_group("unsloth/gemma-4-E2B-it-GGUF", "Q4_K_M")];
        for typed in [
            "unsloth/gemma-4-E2B-it-GGUF:Q4_K_M",
            "/srv/models/local.gguf",
            "bartowski/not-downloaded-GGUF:Q4_K_M",
            // No such row: out of range, and `0`, which is no NR at all.
            "2",
            "0",
        ] {
            assert_eq!(resolve_model_answer(typed, &groups), typed);
        }
        assert_eq!(resolve_model_answer("1", &[]), "1");
    }

    /// One installed model is not a choice — the mandatory `all` role takes
    /// it instead of prompting, exactly as `orangu-server`'s own `--init`
    /// does. Two, or none, is a real prompt.
    #[test]
    fn a_single_installed_model_is_selected_without_asking() {
        let groups = vec![hub_group("unsloth/gemma-4-E2B-it-GGUF", "Q4_K_M")];
        assert_eq!(
            sole_model(&groups).map(qualified_label).as_deref(),
            Some("unsloth/gemma-4-E2B-it-GGUF:Q4_K_M")
        );
        assert!(
            sole_model(&[
                group("bartowski/gemma-4-12B-it-GGUF:Q4_K_M"),
                group("unsloth/gemma-4-E2B-it-GGUF:Q4_K_M"),
            ])
            .is_none()
        );
        assert!(sole_model(&[]).is_none());
    }

    #[test]
    fn empty_groups_give_no_completion_options() {
        assert!(model_completion_options(&[]).is_empty());
        assert!(model_hint_options(&[]).is_empty());
    }

    fn model_hinter() -> ModelCompleter {
        let groups = [
            group("unsloth/gemma-4-E2B-it-GGUF:Q4_K_M"),
            group("unsloth/gemma-4-E4B-it-GGUF:Q4_K_M"),
        ];
        ModelCompleter {
            options: model_completion_options(&groups),
            labels: model_hint_options(&groups),
        }
    }

    /// Both `NR`s and labels TAB-complete, even though only the labels are
    /// ghosted.
    #[test]
    fn completes_both_an_nr_and_a_label() {
        let history = DefaultHistory::new();
        let ctx = RlContext::new(&history);
        let completer = model_hinter();
        let (_, nr) = completer.complete("2", 1, &ctx).unwrap();
        assert_eq!(
            nr.iter().map(|pair| &pair.replacement).collect::<Vec<_>>(),
            vec!["2"]
        );
        let (_, label) = completer.complete("unsloth/gemma-4-E4B", 19, &ctx).unwrap();
        assert_eq!(
            label
                .iter()
                .map(|pair| &pair.replacement)
                .collect::<Vec<_>>(),
            vec!["unsloth/gemma-4-E4B-it-GGUF:Q4_K_M"]
        );
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

    /// An empty line ghost-suggests the first *model*, not the `1` that
    /// `model_completion_options` puts first — the prompt opens already
    /// previewing something worth pressing Enter or Tab for.
    #[test]
    fn hints_the_first_label_on_an_empty_line() {
        let ctx_history = DefaultHistory::new();
        let ctx = RlContext::new(&ctx_history);
        let hint = model_hinter().hint("", 0, &ctx);
        assert_eq!(hint.as_deref(), Some("unsloth/gemma-4-E2B-it-GGUF:Q4_K_M"));
    }

    /// No hint once the cursor isn't at the end of the line, no hint when
    /// nothing typed matches any option — same guarantees
    /// `DirCompleter::hint` makes — and no hint for an `NR`, which no label
    /// starts with, rather than a misleading one. (TAB still completes it.)
    #[test]
    fn no_hint_when_cursor_is_mid_line_or_nothing_matches() {
        let ctx_history = DefaultHistory::new();
        let ctx = RlContext::new(&ctx_history);
        let line = "unsloth/gemma-4-E2B";
        assert_eq!(model_hinter().hint(line, line.len() - 1, &ctx), None);
        assert_eq!(model_hinter().hint("nonexistent/model", 17, &ctx), None);
        assert_eq!(model_hinter().hint("2", 1, &ctx), None);
    }

    #[test]
    fn model_completer_highlight_hint_wraps_the_text_in_grey() {
        let highlighted = model_hinter().highlight_hint("-it-GGUF:Q4_K_M");
        assert_eq!(
            highlighted,
            format!("{GHOST_TEXT}-it-GGUF:Q4_K_M{ANSI_RESET}")
        );
    }

    fn interface(name: &str, ip: &str) -> (String, IpAddr) {
        (name.to_string(), ip.parse().unwrap())
    }

    fn host_values(options: &[HostOption]) -> Vec<&str> {
        options.iter().map(|option| option.value.as_str()).collect()
    }

    /// The exact order `host_completion_options`'s doc comment claims —
    /// `all`, `*`, routable IPv4, routable IPv6, loopback last — the same
    /// order `orangu-server`'s own `host` prompt offers.
    #[test]
    fn offers_all_first_and_loopback_last() {
        let interfaces = [
            interface("lo", "127.0.0.1"),
            interface("wlan0", "192.168.1.10"),
            interface("lo", "::1"),
            interface("eth0", "10.0.0.5"),
            interface("eth0", "2001:db8::5"),
        ];
        assert_eq!(
            host_values(&host_completion_options(&interfaces)),
            vec![
                "all",
                "*",
                "10.0.0.5",
                "192.168.1.10",
                "2001:db8::5",
                "127.0.0.1",
                "::1",
            ]
        );
    }

    /// A machine that reports the same address twice — the same interface
    /// listed once per address family, or two interfaces bridged onto one
    /// address — must not offer it twice.
    #[test]
    fn collapses_duplicate_addresses() {
        let interfaces = [
            interface("eth0", "10.0.0.5"),
            interface("eth0", "10.0.0.5"),
            interface("docker0", "10.0.0.5"),
        ];
        assert_eq!(
            host_values(&host_completion_options(&interfaces)),
            vec!["all", "*", "10.0.0.5"]
        );
    }

    /// The TAB list names the interface an address belongs to; the value
    /// written into the config is the bare address.
    #[test]
    fn labels_each_address_with_its_interface() {
        let options = host_completion_options(&[interface("wlan0", "192.168.1.10")]);
        let address = options.last().unwrap();
        assert_eq!(address.value, "192.168.1.10");
        assert_eq!(address.display, "192.168.1.10 (wlan0)");
    }

    /// Enumeration failing (or a machine with no interfaces at all) still
    /// leaves the two values that don't come from the machine.
    #[test]
    fn always_offers_all_and_its_alias() {
        assert_eq!(host_values(&host_completion_options(&[])), vec!["all", "*"]);
    }

    fn host_hinter() -> HostCompleter {
        HostCompleter {
            options: host_completion_options(&[interface("wlan0", "192.168.1.10")]),
        }
    }

    /// An empty line ghosts `all` — which is both the first candidate and
    /// the default every `host` prompt here opens with, so what Enter is
    /// about to accept is visible before it is pressed.
    #[test]
    fn hints_all_on_an_empty_host_line() {
        let history = DefaultHistory::new();
        let ctx = RlContext::new(&history);
        assert_eq!(host_hinter().hint("", 0, &ctx).as_deref(), Some(HOST_ALL));
        assert_eq!(default_host(), HOST_ALL);
    }

    /// Typing a real address's prefix ghosts the rest of it — the point of
    /// enumerating the interfaces in the first place. Nothing matching
    /// ghosts nothing, rather than something misleading.
    #[test]
    fn hints_the_remainder_of_a_matching_address() {
        let history = DefaultHistory::new();
        let ctx = RlContext::new(&history);
        let hinter = host_hinter();
        assert_eq!(hinter.hint("192.", 4, &ctx).as_deref(), Some("168.1.10"));
        assert_eq!(hinter.hint("172.", 4, &ctx), None);
        assert_eq!(hinter.hint("192.168.1.10", 3, &ctx), None);
    }

    #[test]
    fn host_completer_highlight_hint_wraps_the_text_in_grey() {
        let rendered = host_hinter().highlight_hint("all");
        assert_eq!(rendered, format!("{GHOST_TEXT}all{ANSI_RESET}"));
    }
}
