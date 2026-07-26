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

use crate::config::{HOST_ALL, HOST_ALL_ALIAS, Role, default_host, default_port, default_web};
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
    let role = prompt_role(&format!(
        "role (optional, only used with --daemon) [{}]: ",
        Role::default().label()
    ))?;
    let host = prompt_host(&default_host())?;
    let port = prompt_line("port", &default_port().to_string())?;
    let web = prompt_line("web", &default_web().to_string())?;

    let mut contents = format!("[orangu-server]\nmodels = {models}\n");
    if !model.is_empty() {
        contents.push_str(&format!("model = {model}\n"));
    }
    if role != Role::default() {
        contents.push_str(&format!("role = {}\n", role.label()));
    }
    contents.push_str(&format!("host = {host}\nport = {port}\nweb = {web}\n"));

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
/// already there is just pressing Enter.
fn huggingface_cache_dir() -> Option<PathBuf> {
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
}

impl Highlighter for DirCompleter {}
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

    match editor.readline("model (optional, only used with --daemon) []: ") {
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
/// exact same way) immediately followed by that row's `MODEL` label, for
/// every group in turn — the same NR-then-MODEL pairing, in the same
/// order, `orangu-server`'s own shell completion for `show`/`download`
/// prints from `orangu-server list`'s output (`awk 'NR>1 {print $1; print
/// $2}'`). Split out from [`prompt_model`] so this ordering claim is
/// actually checked, not just asserted in a doc comment.
fn model_completion_options(groups: &[orangu::model_spec::ModelGroup]) -> Vec<String> {
    groups
        .iter()
        .enumerate()
        .flat_map(|(index, group)| [(index + 1).to_string(), group.label.clone()])
        .collect()
}

/// What the `model` prompt ghost-suggests from: the `MODEL` labels only, in
/// `group_models` order — so the first one is the first row `orangu-server
/// list` prints, and it's what an empty line previews. Deliberately not
/// [`model_completion_options`]'s NR-and-label interleaving: its first entry
/// is the digit `1`, which is a shorthand to type, not a model to preview.
fn model_hint_options(groups: &[orangu::model_spec::ModelGroup]) -> Vec<String> {
    groups.iter().map(|group| group.label.clone()).collect()
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
pub(crate) fn prompt_role(prompt: &str) -> Result<Role> {
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
    editor.set_helper(Some(OptionCompleter { options }));

    loop {
        let value = match editor.readline(prompt) {
            Ok(line) => line.trim().to_string(),
            Err(ReadlineError::Eof | ReadlineError::Interrupted) => {
                return Err(anyhow!("aborted: reached end of input"));
            }
            Err(err) => return Err(err.into()),
        };
        if value.is_empty() {
            return Ok(Role::default());
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

    #[test]
    fn empty_groups_give_no_completion_options() {
        assert!(model_completion_options(&[]).is_empty());
        assert!(model_hint_options(&[]).is_empty());
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
