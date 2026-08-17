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

//! Interactive `--init` flow that writes `~/.orangu/orangu-server.conf`.

use crate::config::{
    HOST_ALL, HOST_ALL_ALIAS, Role, WEB_SECTION, default_delete, default_host, default_port,
    default_reexec, default_web_port,
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
use std::io::{IsTerminal, Write};
use std::net::IpAddr;
use std::path::{Path, PathBuf};

/// Grey ANSI truecolor used for the `model` and `host` prompts' inline
/// ghost-text hints (`ModelCompleter`/`HostCompleter`'s own
/// `highlight_hint`) — the same color `src/tui/screen.rs`'s
/// own `GHOST_TEXT` uses for orangu's main chat REPL, duplicated here rather
/// than exported from there for the same reason `DirCompleter` is
/// duplicated from `orangu-coordinator`'s wizard: it's a one-line constant
/// and each `--init` wizard is its own self-contained binary.
const GHOST_TEXT: &str = "\x1b[38;2;120;120;120m";
const ANSI_RESET: &str = "\x1b[0m";

pub fn run_init() -> Result<()> {
    println!("orangu-server configuration");
    println!("============================\n");

    let models = prompt_dir("models", huggingface_cache_dir().as_deref())?;
    let model = prompt_model(Path::new(&models))?;
    let role = prompt_role(
        &format!("role [{}]: ", Role::default().label()),
        Role::default(),
    )?;
    let host = prompt_host(&default_host())?;
    let port = prompt_line("port", &default_port().to_string())?;
    // Asked only when the answer above made it matter. On loopback the server
    // is reachable by whoever is already on the machine and a key buys
    // nothing; the moment the address is widened it is the difference between
    // a private tool and a published inference engine — and this wizard is
    // where that decision is actually being made, so it is the place to ask.
    // Empty leaves the server open, which stays the default rather than
    // becoming a thing you have to know to ask for.
    let api_key = if is_public_host(&host) {
        prompt_line(
            "api_key (blank = no authentication, and the server is reachable off this machine)",
            "",
        )?
    } else {
        String::new()
    };
    // The web console is a section of its own, so it is one yes/no question
    // rather than a port that has to be guessed at (and `0` remembered as
    // the way to decline). Declining writes no `[web]` section at all.
    let web = if prompt_bool("Add web console", true)? {
        // Defaults to the address the API just took, which is also what
        // `[web].host` falls back to when the key is absent — so pressing
        // Enter through this section puts the console wherever the API is,
        // and answering differently is how the two get separated.
        let web_host = prompt_host(&host)?;
        let web_port = prompt_line("port", &default_web_port().to_string())?;
        let reexec = prompt_bool("reexec", default_reexec())?;
        let delete = prompt_bool("delete", default_delete())?;
        Some((web_host, web_port, reexec, delete))
    } else {
        None
    };

    let mut contents = format!("[orangu-server]\nmodels = {models}\n");
    if !model.is_empty() {
        contents.push_str(&format!("model = {model}\n"));
    }
    if role != Role::default() {
        contents.push_str(&format!("role = {}\n", role.label()));
    }
    contents.push_str(&format!("host = {host}\nport = {port}\n"));
    if !api_key.trim().is_empty() {
        contents.push_str(&format!("api_key = {}\n", api_key.trim()));
    }
    if let Some((web_host, web_port, reexec, delete)) = &web {
        // `host` and `port` together, the same pair `[orangu-server]` writes
        // unconditionally above — a section that names where it listens
        // reads better than one that leaves half of it implied.
        contents.push_str(&format!(
            "\n[{WEB_SECTION}]\nhost = {web_host}\nport = {web_port}\n"
        ));
        // Written only when it differs from the default, matching how `role`
        // is handled above and how `--init` stays terse generally.
        if *reexec != default_reexec() {
            contents.push_str(&format!(
                "reexec = {}\n",
                if *reexec { "yes" } else { "no" }
            ));
        }
        if *delete != default_delete() {
            contents.push_str(&format!(
                "delete = {}\n",
                if *delete { "yes" } else { "no" }
            ));
        }
    }

    println!("\nConfiguration to write:\n");
    println!("{contents}");

    if !prompt_bool_yes_default("Write this configuration?")? {
        println!("Aborted. No changes written.");
        return Ok(());
    }

    let dir = home::home_dir()
        .context("failed to resolve home directory")?
        .join(".orangu");
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("failed to create directory {}", dir.display()))?;
    let path = dir.join("orangu-server.conf");
    std::fs::write(&path, contents)
        .with_context(|| format!("failed to write {}", path.display()))?;
    println!("Wrote {}", path.display());

    Ok(())
}

/// Where Hugging Face downloads land by default (`~/.cache/huggingface/hub`
/// on Linux/macOS, `%USERPROFILE%\.cache\huggingface\hub` on Windows) — the
/// same directory llama.cpp's own `-hf` falls back to when
/// `LLAMA_CACHE`/`HF_HUB_CACHE`/etc. aren't set. Offered as `--init`'s
/// default `models` value so pointing `orangu-server` at whatever's likely
/// already there is just pressing Enter — and taken outright, for the same
/// reason, as the models directory of a bundled server started with no
/// config file (`config::bundled_configuration`) and of a `bundle` run that
/// has no config file to read one from.
pub(crate) fn huggingface_cache_dir() -> Option<PathBuf> {
    Some(
        home::home_dir()?
            .join(".cache")
            .join("huggingface")
            .join("hub"),
    )
}

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

    /// Ghost-suggests the rest of the directory being typed, from the same
    /// [`FilenameCompleter`] TAB completes with — so what the grey text
    /// previews and what TAB fills in can never disagree.
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
        let typed = line.get(start..pos)?;
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

fn prompt_dir(label: &str, default: Option<&std::path::Path>) -> Result<String> {
    let default_display = default.map(|d| d.display().to_string());
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
        // A directory that isn't there yet is simply created, silently —
        // only a failure is worth a line, since that's the one case the
        // prompt comes back for.
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

/// A rustyline helper that TAB-completes the whole line against a fixed set
/// of options — the five role names for [`prompt_role`], this file's only
/// prompt with no ghost text of its own — matching the typed prefix case-
/// insensitively. Mirrors `orangu`'s own `OptionCompleter`
/// (`src/bin/orangu/init.rs`), duplicated here rather than shared since
/// each `--init` wizard is a separate, self-contained binary.
struct OptionCompleter {
    options: Vec<String>,
    /// What an empty line means, ghosted so pressing Enter has a visible
    /// effect before it is pressed. `None` leaves an empty line unhinted.
    default: Option<String>,
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

    fn hint(&self, line: &str, pos: usize, _ctx: &RlContext<'_>) -> Option<String> {
        // Same end-of-line rule as every other hinter here.
        if pos < line.len() {
            return None;
        }
        if line.is_empty() {
            return self.default.clone();
        }
        let prefix = line.to_lowercase();
        let candidate = self
            .options
            .iter()
            .find(|option| option.to_lowercase().starts_with(&prefix))?;
        candidate
            .get(line.len()..)
            .filter(|suffix| !suffix.is_empty())
            .map(str::to_string)
    }
}

impl Highlighter for OptionCompleter {
    fn highlight_hint<'h>(&self, hint: &'h str) -> Cow<'h, str> {
        Cow::Owned(format!("{GHOST_TEXT}{hint}{ANSI_RESET}"))
    }
}
impl Validator for OptionCompleter {}
impl Helper for OptionCompleter {}

/// Prompts for the optional `model` key — only consulted in `--daemon`
/// mode — TAB-completing over the models already
/// installed under `models_dir`: every `NR` *and* every `MODEL` label,
/// both in exactly the order `orangu-server list` prints them (both call
/// the same `group_models`, which sorts by label — nothing here re-sorts),
/// and the same pairing `orangu-server`'s own shell completion uses for
/// `show`/`download`'s argument. The `MODEL` labels also drive an inline
/// grey ghost suggestion while typing (see [`ModelCompleter`]), so the
/// prompt opens already previewing the first installed model. Like
/// [`prompt_dir`], doesn't require the typed
/// value to be one of them: a local path or a `<user>/<model>[:quant]`
/// Hugging Face spec not yet downloaded is equally valid, and an empty
/// entry is fine too — daemon mode is the only thing that needs it.
///
/// The one case that isn't asked at all is a `models_dir` holding exactly
/// one model (see [`sole_model`]) — there's nothing to choose between, so
/// it's taken and echoed as a plain `model: <label>` line rather than
/// typed out.
fn prompt_model(models_dir: &Path) -> Result<String> {
    let groups = orangu::model_spec::scan_models_dir(models_dir)
        .map(|models| orangu::model_spec::group_models(&models))
        .unwrap_or_default();

    if let Some(only) = sole_model(&groups) {
        // Echoed in the same `key: value` shape as the prompts around it,
        // so the transcript reads as if it had been answered.
        println!("model: {}", only.label);
        return Ok(only.label.clone());
    }

    let config = Config::builder()
        .completion_type(rustyline::CompletionType::List)
        .build();
    let mut editor: Editor<ModelCompleter, DefaultHistory> = Editor::with_config(config)?;
    editor.set_helper(Some(ModelCompleter {
        options: model_completion_options(&groups),
        labels: model_hint_options(&groups),
    }));

    match editor.readline("model []: ") {
        Ok(line) => Ok(line.trim().to_string()),
        Err(ReadlineError::Eof | ReadlineError::Interrupted) => {
            Err(anyhow!("aborted: reached end of input"))
        }
        Err(err) => Err(err.into()),
    }
}

/// TAB-completes the `model` prompt over `options` — every `NR` and every
/// `MODEL` label, per [`model_completion_options`] — while ghost-suggesting
/// from `labels` alone ([`model_hint_options`]). The split is deliberate: an
/// `NR` is a two-keystroke shorthand someone types on purpose, a model name
/// is the thing worth previewing, so an empty line opens already ghosting
/// the first installed model rather than the digit `1`.
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
        // Same "only at the end of the line" rule as `HostCompleter::hint`.
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
            .filter(|suffix| !suffix.is_empty())
            .map(str::to_string)
    }
}

impl Highlighter for ModelCompleter {
    fn highlight_hint<'h>(&self, hint: &'h str) -> Cow<'h, str> {
        Cow::Owned(format!("{GHOST_TEXT}{hint}{ANSI_RESET}"))
    }
}
impl Validator for ModelCompleter {}
impl Helper for ModelCompleter {}

/// The model to take without asking: `Some` only when the models directory
/// holds exactly one — the same one-row-per-model grouping `orangu-server
/// list` prints, so "one model" means one *label*, however many `.gguf`
/// files (a sharded model's parts) back it. Two or more is a real choice
/// and gets prompted for; none is handled by each caller in turn — here,
/// leaving the `model` key empty (equally valid: only `--daemon` consults
/// it at all); in `main.rs`, the "no models found" error.
///
/// Shared with `main.rs`'s own `select_model_interactively`, so a plain
/// `orangu-server` run and the `--init` wizard agree on when there's
/// nothing to ask.
pub(crate) fn sole_model(
    groups: &[orangu::model_spec::ModelGroup],
) -> Option<&orangu::model_spec::ModelGroup> {
    match groups {
        [only] => Some(only),
        _ => None,
    }
}

/// Turns `group_models`'s output into TAB-completion candidates: `NR` (its
/// 1-based position — `resolve_show_target`'s own NR resolution counts the
/// exact same way) immediately followed by that row's `MODEL:QUANT`
/// ([`qualified_label`]), for every group in turn — the same NR-then-MODEL pairing, in the same
/// order, `orangu-server`'s own shell completion for `show`/`download`
/// prints from `orangu-server list`'s output (`awk 'NR>1 {print $1; print
/// $2}'`). Split out from [`prompt_model`] so this ordering claim is
/// actually checked, not just asserted in a doc comment.
fn model_completion_options(groups: &[orangu::model_spec::ModelGroup]) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    groups
        .iter()
        .enumerate()
        .flat_map(|(index, group)| [(index + 1).to_string(), qualified_label(group)])
        // Two snapshots of one repo at one quantization would still collide;
        // each distinct candidate is offered once, in first-seen order, so
        // the NR-then-MODEL ordering above holds for everything that
        // survives. (Every such row keeps its own `NR` regardless.)
        .filter(|option| seen.insert(option.clone()))
        .collect()
}

/// A group's `MODEL` column with its `QUANT` appended —
/// `unsloth/gemma-4-E2B-it-GGUF:Q4_K_M` — which is what the prompt offers
/// and what an answer gets written to the config as.
///
/// The bare label is not enough. A repo with several quantizations on disk
/// prints the same `MODEL` on every one of their rows, so offering that
/// would list one name eight times over and — worse — write a `model =`
/// value that resolves to whichever of them comes first, rather than the one
/// picked. `<repo>:<quant>` is a spelling
/// [`orangu::model_spec::ModelGroup::matches_label`] already accepts, so it
/// resolves straight back to this exact row.
///
/// Falls back to the plain label where that spelling wouldn't resolve: a
/// model outside the Hugging Face cache layout has no repo id to qualify,
/// and one whose file says nothing about its scheme has no quantization to
/// qualify it with.
fn qualified_label(group: &orangu::model_spec::ModelGroup) -> String {
    match (&group.hf_repo, &group.quantization) {
        (Some(repo), Some(quant)) => format!("{repo}:{quant}"),
        _ => group.label.clone(),
    }
}

/// What the `model` prompt ghost-suggests from: the `MODEL:QUANT` labels
/// only ([`qualified_label`]), in `group_models` order — so the first one is the first row `orangu-server
/// list` prints, and it's what an empty line previews. Deliberately not
/// [`model_completion_options`]'s NR-and-label interleaving: its first entry
/// is the digit `1`, which is a shorthand to type, not a model to preview.
fn model_hint_options(groups: &[orangu::model_spec::ModelGroup]) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    groups
        .iter()
        .map(qualified_label)
        .filter(|label| seen.insert(label.clone()))
        .collect()
}

/// The model prompt of `orangu-server`'s own interactive startup picker,
/// with `default` (the `NR` of whatever `[orangu-server].model` names, when
/// it names something installed) pre-selected — ghosted on the empty line,
/// and returned when Enter is pressed on nothing.
///
/// The pre-selection shows *only* as the ghost: no `[61]` after the label
/// as well. The bracketed form is right for a prompt whose default can't be
/// previewed any other way (`--init`'s `port`, `host`, and the rest), but
/// here it would print the same value twice on one line, once in a place the
/// answer can't be typed and once where it can.
///
/// TAB completes over every `NR` *and* every `MODEL:QUANT`
/// ([`model_completion_options`]) — the same candidates `--init`'s own
/// `model` prompt offers — so anything the prompt suggests is also something
/// the caller accepts. Returns the answer as typed; resolving it is the
/// caller's job, since it accepts more spellings than are offered.
pub(crate) fn prompt_model_nr(
    groups: &[orangu::model_spec::ModelGroup],
    default: Option<&str>,
) -> Result<String> {
    // The blank line separating the table above from the prompt, printed
    // rather than folded into the prompt string — `echo_answer` rewrites the
    // prompt's own line afterwards, and a leading newline in it would make
    // that rewrite land a line too high.
    println!();
    let label = "Model: ";
    let config = Config::builder()
        .completion_type(rustyline::CompletionType::List)
        .build();
    let mut editor: Editor<OptionCompleter, DefaultHistory> = Editor::with_config(config)?;
    editor.set_helper(Some(OptionCompleter {
        options: model_completion_options(groups),
        default: default.map(str::to_string),
    }));

    let value = match editor.readline(label) {
        Ok(line) => line.trim().to_string(),
        Err(ReadlineError::Eof | ReadlineError::Interrupted) => {
            return Err(anyhow!("aborted: reached end of input"));
        }
        Err(err) => return Err(err.into()),
    };
    match (value.is_empty(), default) {
        (true, Some(default)) => Ok(default.to_string()),
        (true, None) => Err(anyhow!("no model selected")),
        (false, _) => Ok(value),
    }
}

/// Rewrites the line a prompt was just answered on so it reads
/// `<label><value>`.
///
/// Without this, answering a ghost-only prompt with Enter leaves `Model: `
/// with nothing after it: the ghost is a hint, not part of the line, so it
/// vanishes the moment Enter is pressed and takes the record of what was
/// chosen with it. The startup transcript is the only place that says which
/// model and role this server came up on before the banner prints, so it has
/// to survive.
///
/// Idempotent for a typed answer — rewriting `Model: 61` as `Model: 61`
/// changes nothing on screen — so the caller doesn't have to know which of
/// the two happened. Skipped entirely when stdout isn't a terminal: there is
/// no line to move back up to in a pipe or a log file, and the escape codes
/// would be the only thing that landed there.
pub(crate) fn echo_answer(label: &str, value: &str) {
    if !std::io::stdout().is_terminal() {
        return;
    }
    print!("{}", answer_line(label, value));
    let _ = std::io::stdout().flush();
}

/// The escape sequence [`echo_answer`] writes: up onto the line `readline`
/// just left behind, clear it, write it out again with the value that was
/// actually taken. Split out so the cursor arithmetic is testable — it is
/// the part that quietly ruins the line above if `label` ever grows a
/// newline of its own.
fn answer_line(label: &str, value: &str) -> String {
    debug_assert!(
        !label.contains('\n'),
        "a label with a newline would put the rewrite on the wrong line"
    );
    format!("\x1b[1A\r\x1b[2K{label}{value}\n")
}

/// Where `spec` sits in `groups` as an `NR`, or `None` when it names
/// nothing installed — a repo that hasn't been downloaded yet, most often,
/// which has no row to point at and is left to the caller to fetch.
///
/// Accepts every spelling the listing itself does: an `NR`, a `MODEL` label,
/// the `MODEL:QUANT` spelling [`qualified_label`] offers, and a path to any
/// of a model's shards.
pub(crate) fn nr_of(groups: &[orangu::model_spec::ModelGroup], spec: &str) -> Option<usize> {
    if let Ok(nr) = spec.parse::<usize>() {
        return (nr >= 1 && nr <= groups.len()).then_some(nr);
    }
    let path = std::fs::canonicalize(spec).ok();
    groups
        .iter()
        .position(|group| {
            group.matches_label(spec)
                || qualified_label(group) == spec
                || path.as_ref().is_some_and(|path| {
                    group
                        .paths
                        .iter()
                        .any(|shard| std::fs::canonicalize(shard).is_ok_and(|s| s == *path))
                })
        })
        .map(|index| index + 1)
}

/// Prompts for a [`Role`], TAB-completing over the five valid role names
/// (dropdown-style: an empty `TAB` press lists every option, matching
/// `rustyline`'s `CompletionType::List`) and defaulting to [`Role::All`] on
/// an empty entry. `prompt` is the exact readline prompt text to show —
/// callers word it for their own context: `run_init`'s wizard (`role`'s
/// only consulted in `--daemon` mode) versus
/// `main.rs`'s plain interactive startup (`select_role_interactively`,
/// where the chosen role takes effect immediately for this run). Unlike
/// `model`'s free-form spec, `role` has a fixed, small set of valid
/// values, so (unlike [`prompt_dir`], which just creates a path that isn't
/// there yet) an unrecognized entry here just re-prompts:
/// there's no sensible way to "use" a role that isn't one of the five
/// [`Role`] actually implements.
pub(crate) fn prompt_role(prompt: &str, default: Role) -> Result<Role> {
    let options: Vec<String> = [
        Role::All,
        Role::Code,
        Role::Review,
        Role::Explorer,
        Role::Embedding,
    ]
    .iter()
    .map(|role| role.label().to_string())
    .collect();

    let config = Config::builder()
        .completion_type(rustyline::CompletionType::List)
        .build();
    let mut editor: Editor<OptionCompleter, DefaultHistory> = Editor::with_config(config)?;
    editor.set_helper(Some(OptionCompleter {
        options,
        default: Some(default.label().to_string()),
    }));

    loop {
        let value = match editor.readline(prompt) {
            Ok(line) => line.trim().to_string(),
            Err(ReadlineError::Eof | ReadlineError::Interrupted) => {
                return Err(anyhow!("aborted: reached end of input"));
            }
            Err(err) => return Err(err.into()),
        };
        if value.is_empty() {
            return Ok(default);
        }
        match Role::parse(&value) {
            Ok(role) => return Ok(role),
            Err(err) => {
                println!("{err}");
                continue;
            }
        }
    }
}

/// One candidate offered at the `host` prompt: `value` is what lands in the
/// config file (and what the ghost text completes to), `display` is the
/// annotated form the TAB list shows — an address on its own says nothing
/// about which interface it belongs to.
#[derive(Clone, Debug, PartialEq, Eq)]
struct HostOption {
    value: String,
    display: String,
}

/// TAB-completes (and ghost-suggests) the `host` prompt over
/// [`host_completion_options`]'s candidates: [`HOST_ALL`], its `*` alias,
/// and every address this machine's network interfaces actually have.
/// Matches the whole typed line case-insensitively, like `OptionCompleter`
/// — but, unlike it, also renders a real `hint()`, so an empty line
/// previews `all` in grey and pressing Enter takes it.
struct HostCompleter {
    options: Vec<HostOption>,
}

impl HostCompleter {
    /// The first candidate whose value starts with everything typed so far,
    /// shared by `complete`'s single-candidate case and `hint`.
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
        // Same "only at the end of the line" rule `orangu-coordinator`'s own
        // hinting helpers use: a preview of what comes after the cursor is
        // meaningless while editing earlier in the line.
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
/// if the platform won't say — the `host` prompt only *assists* typing, so a
/// failed enumeration costs the address candidates and nothing else
/// ([`HOST_ALL`] and its alias are added by [`host_completion_options`]
/// regardless). IPv6 link-local (`fe80::`) addresses are already filtered
/// out by `if_addrs` itself (its `link-local` feature, left off): they can't
/// be bound without a scope id, so they'd only be noise here.
fn local_interfaces() -> Vec<(String, IpAddr)> {
    if_addrs::get_if_addrs()
        .unwrap_or_default()
        .into_iter()
        .map(|interface| (interface.name.clone(), interface.ip()))
        .collect()
}

/// Orders the `host` prompt's candidates: [`HOST_ALL`] first — it's the
/// default, so it's also what ghosts on an empty line — then its `*` alias,
/// then every routable interface address (IPv4 before IPv6, since that's
/// what a `host = ` line almost always holds), and finally the loopback
/// addresses, which are the *narrowest* choice and so the least likely one
/// to be after at a prompt whose whole point is picking what to expose.
/// Duplicates collapse (an interface may report the same address more than
/// once, and a second interface may repeat the first's) keeping the earliest
/// occurrence, so the list stays as short as what the machine really offers.
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

/// Prompts for `host`, ghost-texting/TAB-completing over
/// [`host_completion_options`]. Anything typed is accepted as-is — a
/// hostname the machine resolves, or an address on an interface that only
/// exists once this config is deployed elsewhere, are both legitimate, and
/// `bind` reports the ones that aren't at startup.
/// Whether this bind address makes the server reachable from another machine.
///
/// `all` and `0.0.0.0`/`::` obviously do; a literal address that is not a
/// loopback one does too, and answering `192.168.1.10` is a perfectly ordinary
/// way to reach this state. Anything unparseable is treated as public, because
/// the failure of guessing wrong in that direction is one extra question,
/// where guessing wrong the other way is silence about an exposed server.
fn is_public_host(host: &str) -> bool {
    let host = host.trim();
    if host.eq_ignore_ascii_case(HOST_ALL) || host.eq_ignore_ascii_case(HOST_ALL_ALIAS) {
        return true;
    }
    match host.parse::<std::net::IpAddr>() {
        Ok(ip) => !ip.is_loopback(),
        Err(_) => true,
    }
}

fn prompt_host(default: &str) -> Result<String> {
    let config = Config::builder()
        .completion_type(rustyline::CompletionType::List)
        .build();
    let mut editor: Editor<HostCompleter, DefaultHistory> = Editor::with_config(config)?;
    editor.set_helper(Some(HostCompleter {
        options: host_completion_options(&local_interfaces()),
    }));

    let value = match editor.readline(&format!("host [{default}]: ")) {
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

/// Prompts for a plain value (no filesystem completion), reusing `default`
/// on an empty entry.
fn prompt_line(label: &str, default: &str) -> Result<String> {
    let mut editor: Editor<(), DefaultHistory> = Editor::new()?;
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

/// A yes/no prompt whose default is `default` — shown as `[Y/n]` or
/// `[y/N]` so which way Enter goes is visible, the same convention
/// `delete`/`refresh`'s own confirmations use.
fn prompt_bool(label: &str, default: bool) -> Result<bool> {
    let hint = if default { "Y/n" } else { "y/N" };
    let mut editor: Editor<(), DefaultHistory> = Editor::new()?;
    loop {
        let value = match editor.readline(&format!("{label} [{hint}]: ")) {
            Ok(line) => line.trim().to_lowercase(),
            Err(ReadlineError::Eof | ReadlineError::Interrupted) => {
                return Err(anyhow!("aborted: reached end of input"));
            }
            Err(err) => return Err(err.into()),
        };
        match value.as_str() {
            "" => return Ok(default),
            "y" | "yes" => return Ok(true),
            "n" | "no" => return Ok(false),
            _ => println!("Please answer yes or no."),
        }
    }
}

fn prompt_bool_yes_default(label: &str) -> Result<bool> {
    let mut editor: Editor<(), DefaultHistory> = Editor::new()?;
    let value = match editor.readline(&format!("{label} [Y/n]: ")) {
        Ok(line) => line.trim().to_lowercase(),
        Err(ReadlineError::Eof | ReadlineError::Interrupted) => {
            return Err(anyhow!("aborted: reached end of input"));
        }
        Err(err) => return Err(err.into()),
    };
    Ok(value.is_empty() || value == "y" || value == "yes")
}

#[cfg(test)]
mod tests {
    /// The wizard asks for a key exactly when the address it just wrote makes
    /// one matter.
    ///
    /// Getting this wrong in one direction costs a question nobody needed; in
    /// the other it means the wizard walked an operator into an exposed server
    /// without mentioning it, which is the whole reason the prompt exists.
    #[test]
    fn a_key_is_offered_only_for_an_address_reachable_off_this_machine() {
        for public in [
            "all",
            "0.0.0.0",
            "::",
            "192.168.1.10",
            "10.0.0.1",
            "nonsense",
        ] {
            assert!(is_public_host(public), "{public:?} should prompt for a key");
        }
        for private in ["127.0.0.1", "::1", "  127.0.0.1  "] {
            assert!(!is_public_host(private), "{private:?} should not prompt");
        }
    }

    use super::*;
    use orangu::model_spec::ModelGroup;

    fn group(label: &str) -> ModelGroup {
        ModelGroup {
            label: label.to_string(),
            size_bytes: 0,
            quantization: None,
            errors: Vec::new(),
            representative_path: PathBuf::new(),
            paths: Vec::new(),
            hf_repo: None,
            local_commit: None,
        }
    }

    /// The exact claim `model_completion_options`'s own doc comment makes:
    /// `["1", "<first label>", "2", "<second label>", ...]` — matching
    /// `orangu-server list`'s NR column (1-based position in `group_models`'s
    /// already-sorted-by-label output) paired with its MODEL column, the
    /// same pairing order `orangu-server -s`'s own bash/zsh/fish completion
    /// scripts use for `show`/`download`'s argument.
    #[test]
    fn pairs_each_nr_with_its_label_in_group_models_order() {
        let groups = vec![
            group("Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M"),
            group("bartowski/gemma-4-12B-it-GGUF:Q4_K_M"),
            group("unsloth/gemma-4-E2B-it-GGUF:Q4_K_M"),
        ];
        assert_eq!(
            model_completion_options(&groups),
            vec![
                "1".to_string(),
                "Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M".to_string(),
                "2".to_string(),
                "bartowski/gemma-4-12B-it-GGUF:Q4_K_M".to_string(),
                "3".to_string(),
                "unsloth/gemma-4-E2B-it-GGUF:Q4_K_M".to_string(),
            ]
        );
    }

    /// A real hub-cache group, whose `MODEL` column is the bare repo and
    /// whose quantization is a column of its own.
    fn hub_group(repo: &str, quant: &str) -> ModelGroup {
        ModelGroup {
            label: repo.to_string(),
            size_bytes: 0,
            quantization: Some(quant.to_string()),
            errors: Vec::new(),
            representative_path: PathBuf::new(),
            paths: Vec::new(),
            hf_repo: Some(repo.to_string()),
            local_commit: None,
        }
    }

    /// A repo with several quantizations on disk prints the same bare
    /// `MODEL` on every one of their rows. Offering *that* would list one
    /// name once per quantization — and write a `model =` value resolving to
    /// whichever came first rather than the one picked. Each row is offered
    /// as `MODEL:QUANT` instead, which names it exactly.
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
        // And nothing is offered twice.
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
        // A `.gguf` copied in by hand: no hub-cache repo id.
        assert_eq!(qualified_label(&group("my-local-model")), "my-local-model");

        // A hub-cache model whose file says nothing about its scheme.
        let mut no_quant = hub_group("user/model", "Q4_K_M");
        no_quant.quantization = None;
        assert_eq!(qualified_label(&no_quant), "user/model");
    }

    #[test]
    fn empty_groups_give_no_completion_options() {
        assert!(model_completion_options(&[]).is_empty());
        assert!(model_hint_options(&[]).is_empty());
    }

    /// A configuration that already names a model and a role should cost one
    /// Enter each at startup, not a retype — so `nr_of` has to find the row
    /// whatever spelling the config used.
    #[test]
    fn finds_the_row_a_configured_model_names_however_it_is_spelled() {
        let repo = "bartowski/Llama-3.2-3B-Instruct-GGUF";
        let groups = vec![
            hub_group("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "Q4_K_M"),
            hub_group(repo, "IQ3_M"),
            hub_group(repo, "Q4_K_M"),
        ];

        // The `MODEL:QUANT` spelling the prompt itself offers.
        assert_eq!(nr_of(&groups, &format!("{repo}:Q4_K_M")), Some(3));
        assert_eq!(nr_of(&groups, &format!("{repo}:IQ3_M")), Some(2));
        // A bare `MODEL`, which two rows share — first match, exactly as
        // every other bare-label resolution in the codebase.
        assert_eq!(nr_of(&groups, repo), Some(2));
        // An `NR` as written.
        assert_eq!(nr_of(&groups, "1"), Some(1));
    }

    /// Out-of-range numbers and unknown names have no row to point at, and
    /// must say so rather than picking one.
    #[test]
    fn a_configured_model_that_names_no_row_has_no_nr() {
        let groups = vec![hub_group("user/model", "Q4_K_M")];
        assert_eq!(nr_of(&groups, "2"), None);
        assert_eq!(nr_of(&groups, "0"), None);
        assert_eq!(nr_of(&groups, "user/not-installed-yet"), None);
        assert_eq!(nr_of(&[], "1"), None);
    }

    /// Enter on a ghost leaves the prompt line blank — the hint was never
    /// part of the line. The answer has to be written back onto it, one line
    /// up from where `readline` left the cursor, or the startup transcript
    /// never says which model and role this server came up on.
    #[test]
    fn rewrites_the_answered_line_one_line_up() {
        assert_eq!(answer_line("Model: ", "61"), "\x1b[1A\r\x1b[2KModel: 61\n");
        assert_eq!(answer_line("Role: ", "all"), "\x1b[1A\r\x1b[2KRole: all\n");
    }

    fn option_hinter(default: Option<&str>) -> OptionCompleter {
        OptionCompleter {
            options: ["all", "code", "review", "explorer", "embedding"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            default: default.map(str::to_string),
        }
    }

    /// The pre-selected value is ghosted on an empty line, so what Enter is
    /// about to do is visible before it is pressed.
    #[test]
    fn ghosts_the_pre_selected_option_on_an_empty_line() {
        let history = DefaultHistory::new();
        let ctx = RlContext::new(&history);
        assert_eq!(
            option_hinter(Some("review")).hint("", 0, &ctx).as_deref(),
            Some("review")
        );
        // Nothing pre-selected, nothing ghosted.
        assert_eq!(option_hinter(None).hint("", 0, &ctx), None);
    }

    /// Once something is typed the ghost follows the options, not the
    /// pre-selection — otherwise typing `e` under a `review` default would
    /// preview `review`, which is not what Enter would then do.
    #[test]
    fn typing_ghosts_the_matching_option_not_the_default() {
        let history = DefaultHistory::new();
        let ctx = RlContext::new(&history);
        let hinter = option_hinter(Some("review"));
        // `explorer` precedes `embedding` in the offered order, so a bare
        // `e` previews the first of the two rather than either at random.
        assert_eq!(hinter.hint("e", 1, &ctx).as_deref(), Some("xplorer"));
        assert_eq!(hinter.hint("emb", 3, &ctx).as_deref(), Some("edding"));
        assert_eq!(hinter.hint("zzz", 3, &ctx), None);
        assert_eq!(hinter.hint("review", 3, &ctx), None, "not at end of line");
    }

    fn dir_hinter() -> DirCompleter {
        DirCompleter {
            inner: FilenameCompleter::new(),
        }
    }

    /// The `models` prompt ghost-suggests the rest of the directory being
    /// typed, from the same completer TAB fills in from — so pointing at a
    /// models directory is a prefix and a keypress, not a full path typed
    /// out.
    #[test]
    fn hints_the_remainder_of_a_directory_being_typed() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("models-directory")).unwrap();
        // `MAIN_SEPARATOR`, not a literal `/`: rustyline's own
        // `filename_complete` splits the typed path on that constant alone,
        // so on Windows a `/` is not a separator at all — the whole string
        // reads as one file name in the current directory, nothing matches,
        // and there is no hint to assert about.
        let sep = std::path::MAIN_SEPARATOR;
        let typed = format!("{}{sep}models-dir", dir.path().display());

        let history = DefaultHistory::new();
        let ctx = RlContext::new(&history);
        let hint = dir_hinter()
            .hint(&typed, typed.len(), &ctx)
            .expect("a real subdirectory prefix should ghost the rest of its name");
        // A directory's completion carries a trailing separator of its own,
        // so this is a prefix check rather than an equality one.
        assert!(
            format!("{typed}{hint}")
                .starts_with(&format!("{}{sep}models-directory", dir.path().display())),
            "hint {hint:?} did not complete {typed:?}"
        );
    }

    /// A `Hinter` with no `Highlighter` behind it produces a hint nobody can
    /// see — which is exactly what the `models` prompt used to do. Assert
    /// the grey is actually applied.
    #[test]
    fn renders_the_directory_hint_in_grey() {
        let rendered = dir_hinter().highlight_hint("ectory");
        assert_eq!(rendered, format!("{GHOST_TEXT}ectory{ANSI_RESET}"));
    }

    /// Same end-of-line rule the other hinters follow: a preview of what
    /// comes *after* the cursor means nothing while editing before it.
    #[test]
    fn does_not_hint_a_directory_before_the_end_of_the_line() {
        let history = DefaultHistory::new();
        let ctx = RlContext::new(&history);
        assert_eq!(dir_hinter().hint("/tmp/x", 2, &ctx), None);
    }

    fn model_hinter() -> ModelCompleter {
        let groups = vec![
            group("Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M"),
            group("unsloth/gemma-4-E2B-it-GGUF:Q4_K_M"),
        ];
        ModelCompleter {
            options: model_completion_options(&groups),
            labels: model_hint_options(&groups),
        }
    }

    /// An empty line ghosts the first *model*, not the `1` that
    /// `model_completion_options` puts first — the prompt opens already
    /// previewing something worth pressing Tab for.
    #[test]
    fn hints_the_first_model_on_an_empty_line() {
        let history = DefaultHistory::new();
        let ctx = RlContext::new(&history);
        assert_eq!(
            model_hinter().hint("", 0, &ctx).as_deref(),
            Some("Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M")
        );
    }

    /// Typing a label's prefix ghosts the rest of it, case-insensitively;
    /// an `NR` (which no label starts with) ghosts nothing rather than
    /// something misleading, and TAB still completes it.
    #[test]
    fn hints_the_remainder_of_a_matching_model_label() {
        let history = DefaultHistory::new();
        let ctx = RlContext::new(&history);
        let hinter = model_hinter();
        assert_eq!(
            hinter.hint("unsloth/", 8, &ctx).as_deref(),
            Some("gemma-4-E2B-it-GGUF:Q4_K_M")
        );
        assert_eq!(
            hinter.hint("qwen/", 5, &ctx).as_deref(),
            Some("Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M")
        );
        assert_eq!(hinter.hint("2", 1, &ctx), None);
        assert_eq!(hinter.hint("bartowski/", 10, &ctx), None);
    }

    /// Both `NR`s and labels still TAB-complete, even though only the labels
    /// are ghosted.
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
        let (_, label) = completer.complete("unsl", 4, &ctx).unwrap();
        assert_eq!(
            label
                .iter()
                .map(|pair| &pair.replacement)
                .collect::<Vec<_>>(),
            vec!["unsloth/gemma-4-E2B-it-GGUF:Q4_K_M"]
        );
    }

    /// Editing mid-line suppresses the ghost, same rule as the `host`
    /// prompt's own hinter.
    #[test]
    fn does_not_hint_a_model_before_the_end_of_the_line() {
        let history = DefaultHistory::new();
        let ctx = RlContext::new(&history);
        assert_eq!(model_hinter().hint("unsloth/", 3, &ctx), None);
    }

    /// One installed model is not a choice — both interactive paths take it
    /// instead of prompting for it.
    #[test]
    fn a_single_installed_model_is_selected_without_asking() {
        let groups = vec![group("unsloth/gemma-4-E2B-it-GGUF:Q4_K_M")];
        assert_eq!(
            sole_model(&groups).map(|group| group.label.as_str()),
            Some("unsloth/gemma-4-E2B-it-GGUF:Q4_K_M")
        );
    }

    /// Two models — or none at all — still go to the prompt: picking one of
    /// several is exactly the choice `--init` is there to ask, and an empty
    /// directory has nothing to offer.
    #[test]
    fn several_or_no_installed_models_still_prompt() {
        let groups = vec![
            group("bartowski/gemma-4-12B-it-GGUF:Q4_K_M"),
            group("unsloth/gemma-4-E2B-it-GGUF:Q4_K_M"),
        ];
        assert!(sole_model(&groups).is_none());
        assert!(sole_model(&[]).is_none());
    }

    fn interface(name: &str, ip: &str) -> (String, IpAddr) {
        (name.to_string(), ip.parse().unwrap())
    }

    fn values(options: &[HostOption]) -> Vec<&str> {
        options.iter().map(|option| option.value.as_str()).collect()
    }

    /// The exact order `host_completion_options`'s doc comment claims:
    /// `all`, `*`, routable IPv4, routable IPv6, then loopback last.
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
            values(&host_completion_options(&interfaces)),
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
            values(&host_completion_options(&interfaces)),
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
        assert_eq!(values(&host_completion_options(&[])), vec!["all", "*"]);
    }

    fn host_hinter() -> HostCompleter {
        HostCompleter {
            options: host_completion_options(&[interface("wlan0", "192.168.1.10")]),
        }
    }

    /// An empty line ghosts the default outright, so `--init`'s `host`
    /// prompt shows what Enter is about to accept.
    #[test]
    fn hints_the_default_on_an_empty_line() {
        let history = DefaultHistory::new();
        let ctx = RlContext::new(&history);
        assert_eq!(host_hinter().hint("", 0, &ctx).as_deref(), Some("all"));
    }

    /// Typing a real address's prefix ghosts the rest of it — the point of
    /// enumerating the interfaces in the first place.
    #[test]
    fn hints_the_remainder_of_a_matching_address() {
        let history = DefaultHistory::new();
        let ctx = RlContext::new(&history);
        let hinter = host_hinter();
        assert_eq!(hinter.hint("192.", 4, &ctx).as_deref(), Some("168.1.10"));
        // Nothing this machine has starts with `172.` — no ghost at all,
        // rather than a misleading one.
        assert_eq!(hinter.hint("172.", 4, &ctx), None);
    }

    /// Editing mid-line suppresses the hint, matching `rustyline`'s own
    /// `HistoryHinter` convention.
    #[test]
    fn does_not_hint_before_the_end_of_the_line() {
        let history = DefaultHistory::new();
        let ctx = RlContext::new(&history);
        assert_eq!(host_hinter().hint("192.168.1.10", 3, &ctx), None);
    }

    /// The ghost is rendered in grey and resets afterwards, so the rest of
    /// the prompt isn't left tinted.
    #[test]
    fn renders_the_hint_in_grey() {
        let rendered = host_hinter().highlight_hint("all");
        assert_eq!(rendered, format!("{GHOST_TEXT}all{ANSI_RESET}"));
    }
}
