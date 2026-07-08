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
//! asks for a `llamacpp` command for each role in turn. `all` is mandatory —
//! it's the fallback profile a loaded config must always have — the rest
//! (`code`, `review`, `explorer`, `embeddings`) are skipped by leaving the
//! prompt blank. Each role that gets a command becomes its own section, named
//! after the role.

use crate::config::{
    default_host, default_max_body_bytes, default_port, default_startup_timeout, extract_model_id,
};
use anyhow::{Context, Result, anyhow};
use std::io::{self, Write};

/// Roles offered after the mandatory `all`, in the order `orangu.conf` itself
/// documents them.
const OPTIONAL_ROLES: &[&str] = &["code", "review", "explorer", "embeddings"];

pub async fn run_init() -> Result<()> {
    println!("orangu-coordinator configuration");
    println!("=================================\n");

    let host = prompt_with_default("host", &default_host())?;
    let port = prompt_number::<u16>("port", default_port())?;
    let startup_timeout = prompt_number::<u64>("startup_timeout", default_startup_timeout())?;
    let max_body_bytes = prompt_number::<usize>("max_body_bytes", default_max_body_bytes())?;
    let idle_timeout = prompt_optional_number::<u64>("idle_timeout")?;
    let shutdown_token = prompt_optional_string("shutdown_token")?;

    let mut roles = vec![("all".to_string(), prompt_required_llamacpp("all")?)];
    for role in OPTIONAL_ROLES {
        if let Some(llamacpp) = prompt_optional_llamacpp(role)? {
            roles.push((role.to_string(), llamacpp));
        }
    }

    // `[orangu-coordinator]` values are always written, even when left at
    // their default, so the generated file documents every mandatory
    // property explicitly.
    let mut client = vec![
        format!("host = {host}"),
        format!("port = {port}"),
        format!("startup_timeout = {startup_timeout}"),
        format!("max_body_bytes = {max_body_bytes}"),
    ];
    if let Some(t) = idle_timeout {
        client.push(format!("idle_timeout = {t}"));
    }
    if let Some(tok) = shutdown_token {
        client.push(format!("shutdown_token = {tok}"));
    }

    let mut contents = format!("[orangu-coordinator]\n{}\n", client.join("\n"));
    for (role, llamacpp) in &roles {
        contents.push_str(&format!(
            "\n[{role}]\nrole = {role}\nllamacpp = {llamacpp}\n"
        ));
    }

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

/// Checks that a `llamacpp` value is a valid command line naming a model, the
/// same two things `load_coordinator_configuration` requires, so the wizard
/// never writes a config the loader would later reject.
fn validate_llamacpp(value: &str) -> std::result::Result<(), String> {
    let argv =
        shell_words::split(value).map_err(|err| format!("not a valid command line: {err}"))?;
    if extract_model_id(&argv).is_none() {
        return Err("must specify a model via -hf/--hf-repo or -m/--model".to_string());
    }
    Ok(())
}

/// Prompts for the mandatory `all` role's `llamacpp` command, re-prompting on
/// an empty or invalid value.
fn prompt_required_llamacpp(role: &str) -> Result<String> {
    loop {
        let value = prompt(&format!("llamacpp/{role} []: "))?;
        if value.is_empty() {
            println!("A value is required for the mandatory 'all' role.");
            continue;
        }
        match validate_llamacpp(&value) {
            Ok(()) => return Ok(value),
            Err(err) => println!("Invalid llamacpp command: {err}"),
        }
    }
}

/// Prompts for an optional role's `llamacpp` command. An empty entry skips
/// the role entirely (`Ok(None)`); an invalid non-empty entry re-prompts.
fn prompt_optional_llamacpp(role: &str) -> Result<Option<String>> {
    loop {
        let value = prompt(&format!("llamacpp/{role} []: "))?;
        if value.is_empty() {
            return Ok(None);
        }
        match validate_llamacpp(&value) {
            Ok(()) => return Ok(Some(value)),
            Err(err) => println!("Invalid llamacpp command: {err}"),
        }
    }
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

    #[test]
    fn validates_llamacpp_requires_a_model_flag() {
        assert!(validate_llamacpp("llama-server -hf org/gemma --port 8100").is_ok());
        assert!(validate_llamacpp("llama-server -m /models/gemma.gguf").is_ok());
        assert!(validate_llamacpp("llama-server --port 8100").is_err());
    }

    #[test]
    fn validates_llamacpp_rejects_malformed_command_lines() {
        let err =
            validate_llamacpp("llama-server --chat-template-kwargs '{\"unterminated").unwrap_err();
        assert!(err.contains("not a valid command line"), "{err}");
    }
}
