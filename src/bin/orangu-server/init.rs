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

use crate::config::{Role, default_host, default_port, default_web};
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
use std::path::{Path, PathBuf};

pub fn run_init() -> Result<()> {
    println!("orangu-server configuration");
    println!("============================\n");

    let models = prompt_dir("models", huggingface_cache_dir().as_deref())?;
    let model = prompt_model(Path::new(&models))?;
    let role = prompt_role(&format!(
        "role (optional, only used with --daemon) [{}]: ",
        Role::default().label()
    ))?;
    let host = prompt_line("host", &default_host())?;
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
        let path = PathBuf::from(&value);
        if !path.is_dir() {
            println!("'{value}' does not exist or is not a directory.");
            if !prompt_bool_yes_default("Use it anyway?")? {
                continue;
            }
        }
        return Ok(value);
    }
}

/// A rustyline helper that TAB-completes the whole line against a fixed set
/// of options (installed-model labels for [`prompt_model`], the five role
/// names for [`prompt_role`]), matching the typed prefix case-
/// insensitively — mirrors `orangu`'s own `OptionCompleter`
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
/// `show`/`download`'s argument. Like [`prompt_dir`], doesn't require the typed
/// value to be one of them: a local path or a `<user>/<model>[:quant]`
/// Hugging Face spec not yet downloaded is equally valid, and an empty
/// entry is fine too — daemon mode is the only thing that needs it.
fn prompt_model(models_dir: &Path) -> Result<String> {
    let options = orangu::model_spec::scan_models_dir(models_dir)
        .map(|models| model_completion_options(&orangu::model_spec::group_models(&models)))
        .unwrap_or_default();

    let config = Config::builder()
        .completion_type(rustyline::CompletionType::List)
        .build();
    let mut editor: Editor<OptionCompleter, DefaultHistory> = Editor::with_config(config)?;
    editor.set_helper(Some(OptionCompleter { options }));

    match editor.readline("model (optional, only used with --daemon) []: ") {
        Ok(line) => Ok(line.trim().to_string()),
        Err(ReadlineError::Eof | ReadlineError::Interrupted) => {
            Err(anyhow!("aborted: reached end of input"))
        }
        Err(err) => Err(err.into()),
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

/// Prompts for a [`Role`], TAB-completing over the five valid role names
/// (dropdown-style: an empty `TAB` press lists every option, matching
/// `rustyline`'s `CompletionType::List`) and defaulting to [`Role::All`] on
/// an empty entry. `prompt` is the exact readline prompt text to show —
/// callers word it for their own context: `run_init`'s wizard (`role`'s
/// only consulted in `--daemon` mode) versus
/// `main.rs`'s plain interactive startup (`select_role_interactively`,
/// where the chosen role takes effect immediately for this run). Unlike
/// `model`'s free-form spec, `role` has a fixed, small set of valid
/// values, so (unlike [`prompt_dir`]'s "use it anyway?" escape hatch for
/// an unrecognized path) an unrecognized entry here just re-prompts:
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
    }
}
