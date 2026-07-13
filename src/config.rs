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

use anyhow::{Context, Result, anyhow};
use serde::Serialize;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    str::FromStr,
};

use crate::tui::Banner;
use crate::workspaces::WorkspacePlacement;

#[derive(Clone, Debug, Serialize)]
pub struct ClientAppConfiguration {
    pub default_server: String,
    pub default_model: Option<String>,
    pub llms: HashMap<String, LlmConfiguration>,
    pub compression: bool,
    pub auto_downsample_lines: usize,
    pub diff_file_cap: usize,
    pub quotes: String,
    pub width: usize,
    #[serde(skip)]
    pub banner: Banner,
    pub feedback: bool,
    pub auto_rebase: bool,
    pub auto_squash: bool,
    pub compile_workers: usize,
    pub terminal: String,
    pub platform: String,
    pub workspaces: WorkspacePlacement,
    pub drop_down: bool,
    /// Enable mouse capture in the terminal. When `true` (default), the TUI
    /// handles mouse scroll (for output scrolling) and double-click (to toggle
    /// collapsibles). Hold **Shift** while clicking/dragging to do native text
    /// selection and copy. Set to `false` to disable all mouse handling.
    pub mouse: bool,
    pub semantic_budget_tokens: usize,
    pub theme: String,
    pub auto_dark_theme: String,
    pub auto_light_theme: String,
}

impl ClientAppConfiguration {
    /// Returns the name of the first server configured with the given role.
    /// If no server specifies this role, returns the `default_server`.
    pub fn find_server_for_role(&self, role: &str) -> String {
        self.find_server_for_role_or(role, &self.default_server)
    }

    /// Returns the name of the first server configured with the given role,
    /// or `fallback` if no server specifies this role. Lets a caller prefer a
    /// dedicated role (e.g. `review`) while falling back to something more
    /// specific than the static config default — such as whatever server the
    /// interactive session is currently connected to — when no server opts
    /// into that role.
    pub fn find_server_for_role_or(&self, role: &str, fallback: &str) -> String {
        for (name, profile) in &self.llms {
            if profile.role == role {
                return name.clone();
            }
        }
        fallback.to_string()
    }

    /// The server section that can serve embeddings for semantic `/search`,
    /// resolved by role: a section whose `role` is `embeddings` wins; otherwise
    /// the default server serves it as part of the `all` role, and failing that
    /// any `all` server does. Returns `None` when no section qualifies.
    ///
    /// Whether the endpoint actually serves embeddings is confirmed by probing
    /// it at startup — this only names the candidate.
    pub fn embeddings_server(&self) -> Option<String> {
        for (name, profile) in &self.llms {
            if profile.role == "embeddings" {
                return Some(name.clone());
            }
        }
        if let Some(profile) = self.llms.get(&self.default_server)
            && matches!(profile.role.as_str(), "all" | "embeddings")
        {
            return Some(self.default_server.clone());
        }
        self.llms
            .iter()
            .find(|(_, profile)| profile.role == "all")
            .map(|(name, _)| name.clone())
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct LlmConfiguration {
    pub endpoint: String,
    pub model: String,
    pub role: String,
    pub api_key: Option<String>,
    pub request_timeout_seconds: u64,
    pub max_tool_rounds: usize,
    /// Response-token cap for `/auto_review` requests; `0` disables the cap.
    pub review_max_tokens: u32,
    /// Response-token cap for normal chat/tool responses; `0` disables the cap.
    pub code_max_tokens: u32,
    pub system_prompt: String,
    pub model_verbosity: Option<String>,
    /// Minimum confidence score (0–100) for `/auto_review` findings; findings
    /// below this threshold are silently dropped. `0` disables filtering.
    pub review_confidence_threshold: u32,
}

pub const CLIENT_SECTION: &str = "orangu";

pub fn default_virtual_width() -> usize {
    512
}

pub fn default_timeout() -> u64 {
    1_800
}

pub fn default_llm_max_tool_rounds() -> usize {
    10
}

/// Default `/auto_review` response cap: a verdict plus at most five one-line
/// findings fits comfortably. Raise it (with model thinking enabled) for
/// deeper reviews; `0` disables the cap.
pub fn default_review_max_tokens() -> u32 {
    512
}

pub fn default_review_confidence_threshold() -> u32 {
    80
}

/// Default chat-response cap: `0`, no cap — normal coding responses are
/// open-ended.
pub fn default_code_max_tokens() -> u32 {
    0
}

pub fn default_drop_down() -> bool {
    true
}

pub fn default_theme() -> String {
    "classic".to_string()
}

pub fn default_auto_dark_theme() -> String {
    "classic".to_string()
}

pub fn default_auto_light_theme() -> String {
    "oranguday".to_string()
}

pub fn default_auto_downsample_lines() -> usize {
    300
}

pub fn default_diff_file_cap() -> usize {
    20
}

pub fn default_compression() -> bool {
    true
}

/// Default `/build` parallelism: `0`, meaning unused — no `-j`/`--jobs` flag is
/// passed and each toolchain falls back to its own default (Cargo's own
/// automatic parallelism, a bare serial `make`, ...).
pub fn default_compile_workers() -> usize {
    0
}

pub fn parse_feedback_bool(s: &str) -> bool {
    matches!(s.trim().to_lowercase().as_str(), "on" | "true" | "1")
}

pub fn load_client_configuration(path: &Path) -> Result<ClientAppConfiguration> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read configuration {}", path.display()))?;
    let mut sections = parse_ini_sections(&contents)
        .with_context(|| format!("failed to parse configuration {}", path.display()))?;

    // The [orangu] section holds client-wide settings; every other section is
    // a named LLM profile.
    let client = sections.remove(CLIENT_SECTION).unwrap_or_default();

    let timeout = parse_client_field(&client, "timeout", default_timeout)?;
    let max_tool_rounds =
        parse_client_field(&client, "max_tool_rounds", default_llm_max_tool_rounds)?;
    let review_max_tokens =
        parse_client_field(&client, "review_max_tokens", default_review_max_tokens)?;
    let review_confidence_threshold = parse_client_field(
        &client,
        "review_confidence_threshold",
        default_review_confidence_threshold,
    )?;
    let code_max_tokens = parse_client_field(&client, "code_max_tokens", default_code_max_tokens)?;
    let compression = client
        .get("compression")
        .map(|value| parse_feedback_bool(value))
        .unwrap_or_else(default_compression);
    let width = parse_client_field(&client, "width", default_virtual_width)?;
    let workspaces = parse_client_field(&client, "workspaces", WorkspacePlacement::default)?;
    let drop_down = client
        .get("drop_down")
        .map(|value| parse_feedback_bool(value))
        .unwrap_or_else(default_drop_down);
    let mouse = client
        .get("mouse")
        .map(|value| parse_feedback_bool(value))
        .unwrap_or(true);
    let auto_downsample_lines = parse_client_field(
        &client,
        "auto_downsample_lines",
        default_auto_downsample_lines,
    )?;
    let diff_file_cap = parse_client_field(&client, "diff_file_cap", default_diff_file_cap)?;
    let compile_workers = parse_client_field(&client, "compile_workers", default_compile_workers)?;
    let system_prompt = client.get("system_prompt").cloned().unwrap_or_default();

    let semantic_budget_tokens = client
        .get("semantic_budget_tokens")
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(16384);

    let theme = client.get("theme").cloned().unwrap_or_else(default_theme);
    let auto_dark_theme = client
        .get("auto_dark_theme")
        .cloned()
        .unwrap_or_else(default_auto_dark_theme);
    let auto_light_theme = client
        .get("auto_light_theme")
        .cloned()
        .unwrap_or_else(default_auto_light_theme);

    // `[orangu].model` is the general default model id; a server section's own
    // `model` takes precedence over it.
    let default_model = client
        .get("model")
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());

    normalize_client_configuration(ClientAppConfiguration {
        default_server: client
            .get("server")
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .unwrap_or_default(),
        default_model: default_model.clone(),
        llms: parse_llm_profiles(
            sections,
            timeout,
            max_tool_rounds,
            review_max_tokens,
            review_confidence_threshold,
            code_max_tokens,
            system_prompt,
            default_model,
        )?,
        compression,
        auto_downsample_lines,
        diff_file_cap,
        quotes: client.get("quotes").cloned().unwrap_or_default(),
        width,
        banner: client
            .get("banner")
            .map(|value| value.parse().unwrap_or_default())
            .unwrap_or_default(),
        feedback: parse_feedback_bool(client.get("feedback").map(String::as_str).unwrap_or("")),
        auto_rebase: parse_feedback_bool(
            client.get("auto_rebase").map(String::as_str).unwrap_or(""),
        ),
        auto_squash: parse_feedback_bool(
            client.get("auto_squash").map(String::as_str).unwrap_or(""),
        ),
        semantic_budget_tokens,
        compile_workers,
        terminal: client.get("terminal").cloned().unwrap_or_default(),
        platform: client.get("platform").cloned().unwrap_or_default(),
        workspaces,
        drop_down,
        mouse,
        theme,
        auto_dark_theme,
        auto_light_theme,
    })
}

pub const DEFAULT_PLATFORM: &str = "github";

/// Parse a typed value from the `[orangu]` section, falling back to `default`
/// when the key is absent. Profile names and values may freely contain `.`
/// and `:` — unlike a generic INI loader, this parser never treats those as
/// nested-key separators.
pub fn set_client_theme(path: &Path, new_theme: &str) -> Result<()> {
    let contents = std::fs::read_to_string(path)?;
    let mut lines: Vec<String> = contents.lines().map(|s| s.to_string()).collect();

    let mut in_orangu_section = false;
    let mut theme_line_index = None;
    let mut insert_index = None;

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            let section = &trimmed[1..trimmed.len() - 1].trim();
            if *section == CLIENT_SECTION {
                in_orangu_section = true;
                insert_index = Some(i + 1); // After the section header
            } else if in_orangu_section {
                // Leaving the section
                break;
            }
        } else if in_orangu_section && !trimmed.is_empty() && !trimmed.starts_with('#') {
            insert_index = Some(i + 1); // Track the last active line in the section
            if let Some(stripped) = trimmed.strip_prefix("theme") {
                let after_key = &stripped.trim_start();
                if after_key.starts_with('=') {
                    theme_line_index = Some(i);
                    break;
                }
            }
        }
    }
    let new_line = format!("theme = {}", new_theme);

    if let Some(idx) = theme_line_index {
        lines[idx] = new_line;
    } else if let Some(idx) = insert_index {
        lines.insert(idx, new_line);
    } else {
        // [orangu] section doesn't exist at all, append it
        if !lines.is_empty() && !lines.last().unwrap().is_empty() {
            lines.push(String::new());
        }
        lines.push(format!("[{}]", CLIENT_SECTION));
        lines.push(new_line);
    }

    let new_contents = lines.join("\n") + "\n";
    std::fs::write(path, new_contents)?;
    Ok(())
}

fn parse_client_field<T: FromStr>(
    client: &HashMap<String, String>,
    key: &str,
    default: impl Fn() -> T,
) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    match client.get(key) {
        Some(value) => value
            .trim()
            .parse::<T>()
            .map_err(|err| anyhow!("invalid value for [{CLIENT_SECTION}].{key}: {err}")),
        None => Ok(default()),
    }
}

/// Minimal INI parser: `[section]` headers and `key = value` lines, with `#`
/// and `;` full-line comments and blank lines ignored. Section names and
/// values are taken literally, so model identifiers containing `.` or `:`
/// (e.g. `Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M`) round-trip unchanged.
pub fn parse_ini_sections(contents: &str) -> Result<HashMap<String, HashMap<String, String>>> {
    let mut sections: HashMap<String, HashMap<String, String>> = HashMap::new();
    let mut current: Option<String> = None;

    for (index, raw) in contents.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
            continue;
        }

        if let Some(rest) = line.strip_prefix('[') {
            let name = rest
                .strip_suffix(']')
                .ok_or_else(|| anyhow!("line {}: malformed section header '{}'", index + 1, raw))?;
            let name = name.trim().to_string();
            current = Some(name.clone());
            sections.entry(name).or_default();
            continue;
        }

        let (key, value) = line.split_once('=').ok_or_else(|| {
            anyhow!(
                "line {}: expected 'key = value' but found '{}'",
                index + 1,
                raw
            )
        })?;
        let section = current.as_ref().ok_or_else(|| {
            anyhow!(
                "line {}: key '{}' appears before any [section]",
                index + 1,
                key.trim()
            )
        })?;
        sections
            .get_mut(section)
            .expect("current section was inserted")
            .insert(key.trim().to_string(), value.trim().to_string());
    }

    Ok(sections)
}

pub fn default_client_config_path() -> Option<PathBuf> {
    let cwd_path = std::env::current_dir().ok()?.join("orangu.conf");
    if cwd_path.exists() {
        return Some(cwd_path);
    }

    let config_path = home::home_dir()?.join(".orangu/orangu.conf");
    config_path.exists().then_some(config_path)
}

fn normalize_client_configuration(
    mut conf: ClientAppConfiguration,
) -> Result<ClientAppConfiguration> {
    conf.default_server = conf.default_server.trim().to_string();

    conf.platform = conf.platform.trim().to_lowercase();
    if conf.platform.is_empty() {
        conf.platform = DEFAULT_PLATFORM.to_string();
    }
    match conf.platform.as_str() {
        "github" | "gitlab" => {}
        other => {
            return Err(anyhow!(
                "Unsupported [{CLIENT_SECTION}].platform '{other}'; expected 'github' or 'gitlab'"
            ));
        }
    }

    if conf.llms.is_empty() {
        return Err(anyhow!("At least one named LLM profile must be defined"));
    }

    for llm in conf.llms.values_mut() {
        normalize_llm_configuration(llm)?;
    }

    // Each chat server must resolve to a distinct (endpoint, model) pair: a
    // server is one host serving one model, and `/model` cycles the models
    // that single host offers. Endpoints are compared canonically, so
    // `http://x` and `http://x/v1/` count as the same host. Two servers *may*
    // share a host as long as their `model` differs — this is how Ollama
    // serves several models off one endpoint, and how an orangu-coordinator
    // instance fronts several role-specific models behind a single proxy
    // address. An `embeddings` server is exempt entirely — it is selected by
    // role, never in the `/server` rotation, so it may share both endpoint and
    // model with a chat server.
    let mut endpoints: HashMap<(&str, &str), &str> = HashMap::new();
    for (name, llm) in &conf.llms {
        if llm.role == "embeddings" {
            continue;
        }
        let canonical = llm.endpoint.strip_suffix("/v1").unwrap_or(&llm.endpoint);
        let key = (canonical, llm.model.as_str());
        if let Some(existing) = endpoints.insert(key, name.as_str()) {
            return Err(anyhow!(
                "Servers '{existing}' and '{name}' share endpoint '{}' and model '{}'; servers sharing an endpoint must use distinct models",
                llm.endpoint,
                llm.model
            ));
        }
    }

    if conf.default_server.is_empty() {
        if conf.llms.len() == 1 {
            conf.default_server = conf
                .llms
                .keys()
                .next()
                .cloned()
                .ok_or_else(|| anyhow!("Missing server definition"))?;
        } else {
            return Err(anyhow!(
                "Client configuration must define [orangu].server when multiple servers are configured"
            ));
        }
    }

    if !conf.llms.contains_key(&conf.default_server) {
        return Err(anyhow!(
            "Server '{}' is not defined in the configuration",
            conf.default_server
        ));
    }

    Ok(conf)
}

#[allow(clippy::too_many_arguments)]
fn parse_llm_profiles(
    sections: HashMap<String, HashMap<String, String>>,
    timeout: u64,
    max_tool_rounds: usize,
    review_max_tokens: u32,
    review_confidence_threshold: u32,
    code_max_tokens: u32,
    system_prompt: String,
    default_model: Option<String>,
) -> Result<HashMap<String, LlmConfiguration>> {
    sections
        .into_iter()
        .map(|(name, values)| {
            // A server section's own `model` overrides the general
            // `[orangu].model`; fall back to it when the section omits one.
            let model = values
                .get("model")
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
                .or_else(|| default_model.clone())
                .unwrap_or_default();
            let api_key = values
                .get("api_key")
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty());
            let role = values
                .get("role")
                .map(|value| value.trim().to_lowercase())
                .filter(|r| !r.is_empty())
                .unwrap_or_else(|| "all".to_string());

            let model_verbosity = values
                .get("model_verbosity")
                .map(|value| value.trim().to_lowercase())
                .filter(|value| !value.is_empty());

            Ok((
                name,
                LlmConfiguration {
                    endpoint: values.get("endpoint").cloned().unwrap_or_default(),
                    model,
                    role,
                    api_key,
                    request_timeout_seconds: timeout,
                    max_tool_rounds,
                    review_max_tokens,
                    code_max_tokens,
                    system_prompt: system_prompt.clone(),
                    model_verbosity,
                    review_confidence_threshold,
                },
            ))
        })
        .collect()
}

fn normalize_llm_configuration(llm: &mut LlmConfiguration) -> Result<()> {
    llm.endpoint = llm.endpoint.trim().trim_end_matches('/').to_string();
    llm.model = llm.model.trim().to_string();
    llm.system_prompt = llm.system_prompt.trim().to_string();

    if let Some(ref v) = llm.model_verbosity {
        match v.as_str() {
            "terse" | "normal" | "verbose" => {}
            other => {
                return Err(anyhow!(
                    "model_verbosity must be terse, normal, or verbose (got '{other}')"
                ));
            }
        }
    }

    if llm.endpoint.is_empty() {
        return Err(anyhow!("LLM endpoint must not be empty"));
    }
    if llm.model.is_empty() {
        return Err(anyhow!("LLM model must not be empty"));
    }

    Ok(())
}

pub fn load_agents_instructions(workspace: &std::path::Path) -> String {
    let global_agents = home::home_dir()
        .map(|h| h.join(".orangu/AGENTS.md"))
        .unwrap_or_default();
    let local_agents = workspace.join("AGENTS.md");

    let mut agents_content = String::new();
    if global_agents.exists()
        && let Ok(content) = std::fs::read_to_string(&global_agents)
    {
        agents_content.push_str(&content);
        agents_content.push('\n');
    }
    if local_agents.exists()
        && let Ok(content) = std::fs::read_to_string(&local_agents)
    {
        agents_content.push_str(&content);
        agents_content.push('\n');
    }

    if agents_content.is_empty() {
        String::new()
    } else {
        format!(
            "\n\n# AGENTS.md instructions for project\n<INSTRUCTIONS>\n{}</INSTRUCTIONS>\n",
            agents_content
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn loads_ini_style_profiles() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu]\nserver = gemma\ntimeout = 45\nmax_tool_rounds = 12\n\n[gemma]\nendpoint = http://localhost:8100/v1\nmodel = ggml-org/gemma-4-E4B-it-GGUF\n\n[qwen]\nendpoint = http://localhost:8101/v1\nmodel = unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF\n"
        )
        .unwrap();

        let conf = load_client_configuration(file.path()).unwrap();
        assert_eq!(conf.default_server, "gemma");
        assert_eq!(conf.llms.len(), 2);
        assert_eq!(conf.llms["gemma"].request_timeout_seconds, 45);
        assert_eq!(conf.llms["gemma"].max_tool_rounds, 12);
        assert!(conf.compression);
        // Absent platform defaults to GitHub.
        assert_eq!(conf.platform, "github");
        // Absent theme keeps the historical dark UI as the named classic theme.
        assert_eq!(conf.theme, "classic");
        assert_eq!(conf.auto_dark_theme, "classic");
        assert_eq!(conf.auto_light_theme, "oranguday");
    }

    #[test]
    fn parses_theme_defaults_and_overrides() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu]\nserver = a\ntheme = auto\nauto_dark_theme = tokyonight\nauto_light_theme = oranguday\n\n[a]\nendpoint = http://x/v1\nmodel = m\n"
        )
        .unwrap();
        let conf = load_client_configuration(file.path()).unwrap();
        assert_eq!(conf.theme, "auto");
        assert_eq!(conf.auto_dark_theme, "tokyonight");
        assert_eq!(conf.auto_light_theme, "oranguday");
    }

    #[test]
    fn parses_review_and_code_max_tokens_with_defaults() {
        // Absent keys: reviews capped at 512 tokens, chat responses uncapped.
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu]\nserver = a\n\n[a]\nendpoint = http://x/v1\nmodel = m\n"
        )
        .unwrap();
        let conf = load_client_configuration(file.path()).unwrap();
        assert_eq!(conf.llms["a"].review_max_tokens, 512);
        assert_eq!(conf.llms["a"].code_max_tokens, 0);

        // Explicit values land on every profile.
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu]\nserver = a\nreview_max_tokens = 2048\ncode_max_tokens = 4096\n\n[a]\nendpoint = http://x/v1\nmodel = m\n"
        )
        .unwrap();
        let conf = load_client_configuration(file.path()).unwrap();
        assert_eq!(conf.llms["a"].review_max_tokens, 2048);
        assert_eq!(conf.llms["a"].code_max_tokens, 4096);
    }

    #[test]
    fn parses_compile_workers_with_default_and_override() {
        // Absent key defaults to 0 (unused: no job flag is passed).
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu]\nserver = a\n\n[a]\nendpoint = http://x/v1\nmodel = m\n"
        )
        .unwrap();
        let conf = load_client_configuration(file.path()).unwrap();
        assert_eq!(conf.compile_workers, 0);

        // An explicit 0 is accepted too, with the same meaning as the default.
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu]\nserver = a\ncompile_workers = 0\n\n[a]\nendpoint = http://x/v1\nmodel = m\n"
        )
        .unwrap();
        let conf = load_client_configuration(file.path()).unwrap();
        assert_eq!(conf.compile_workers, 0);

        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu]\nserver = a\ncompile_workers = 12\n\n[a]\nendpoint = http://x/v1\nmodel = m\n"
        )
        .unwrap();
        let conf = load_client_configuration(file.path()).unwrap();
        assert_eq!(conf.compile_workers, 12);
    }

    #[test]
    fn parses_compression_with_default_and_override() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu]\nserver = a\n\n[a]\nendpoint = http://x/v1\nmodel = m\n"
        )
        .unwrap();
        let conf = load_client_configuration(file.path()).unwrap();
        assert!(conf.compression);

        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu]\nserver = a\ncompression = off\n\n[a]\nendpoint = http://x/v1\nmodel = m\n"
        )
        .unwrap();
        let conf = load_client_configuration(file.path()).unwrap();
        assert!(!conf.compression);
    }

    #[test]
    fn parses_workspaces_placement_with_default_and_validation() {
        // Absent key defaults to the top placement.
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu]\nserver = a\n\n[a]\nendpoint = http://x/v1\nmodel = m\n"
        )
        .unwrap();
        let conf = load_client_configuration(file.path()).unwrap();
        assert_eq!(conf.workspaces, WorkspacePlacement::Top);

        // An explicit value is parsed case-insensitively.
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu]\nserver = a\nworkspaces = Bottom\n\n[a]\nendpoint = http://x/v1\nmodel = m\n"
        )
        .unwrap();
        let conf = load_client_configuration(file.path()).unwrap();
        assert_eq!(conf.workspaces, WorkspacePlacement::Bottom);

        // An unknown value is rejected, naming the key.
        let mut bad = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            bad,
            "[orangu]\nserver = a\nworkspaces = middle\n\n[a]\nendpoint = http://x/v1\nmodel = m\n"
        )
        .unwrap();
        let err = load_client_configuration(bad.path()).unwrap_err();
        assert!(
            format!("{err:#}").contains("invalid value for [orangu].workspaces"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn parses_and_validates_platform() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu]\nserver = a\nplatform = GitLab\n\n[a]\nendpoint = http://x/v1\nmodel = m\n"
        )
        .unwrap();
        let conf = load_client_configuration(file.path()).unwrap();
        assert_eq!(conf.platform, "gitlab");

        let mut bad = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            bad,
            "[orangu]\nserver = a\nplatform = bitbucket\n\n[a]\nendpoint = http://x/v1\nmodel = m\n"
        )
        .unwrap();
        let err = load_client_configuration(bad.path()).unwrap_err();
        assert!(
            err.to_string().contains("Unsupported [orangu].platform"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn loads_profiles_with_dots_and_colons_in_section_names() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu]\nserver = Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M\n\n[Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M]\nendpoint = http://localhost:8100/v1\nmodel = bartowski/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M\n\n[Qwen_Qwen3.6-35B-A3B-GGUF]\nendpoint = http://localhost:8101/v1\nmodel = bartowski/Qwen_Qwen3.6-35B-A3B-GGUF\n"
        )
        .unwrap();

        let conf = load_client_configuration(file.path()).unwrap();
        assert_eq!(conf.default_server, "Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M");
        assert_eq!(conf.llms.len(), 2);
        assert!(
            conf.llms
                .contains_key("Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M")
        );
        assert!(conf.llms.contains_key("Qwen_Qwen3.6-35B-A3B-GGUF"));
        assert_eq!(
            conf.llms["Qwen_Qwen3.6-35B-A3B-GGUF"].model,
            "bartowski/Qwen_Qwen3.6-35B-A3B-GGUF"
        );
    }

    #[test]
    fn reports_invalid_numeric_client_field() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu]\nserver = a\ntimeout = soon\n\n[a]\nendpoint = http://x/v1\nmodel = m\n"
        )
        .unwrap();

        let err = load_client_configuration(file.path()).unwrap_err();
        assert!(
            format!("{err:#}").contains("invalid value for [orangu].timeout"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn parse_feedback_bool_recognises_truthy_and_falsy_values() {
        assert!(super::parse_feedback_bool("on"));
        assert!(super::parse_feedback_bool("true"));
        assert!(super::parse_feedback_bool("1"));
        assert!(!super::parse_feedback_bool("off"));
        assert!(!super::parse_feedback_bool("false"));
        assert!(!super::parse_feedback_bool("0"));
        assert!(!super::parse_feedback_bool(""));
    }

    #[test]
    fn requires_default_server_when_multiple_profiles_exist() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu]\n\n[gemma]\nendpoint = http://localhost:8100/v1\nmodel = ggml-org/gemma-4-E4B-it-GGUF\n\n[qwen]\nendpoint = http://localhost:8101/v1\nmodel = unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF\n"
        )
        .unwrap();

        let err = load_client_configuration(file.path()).unwrap_err();
        assert!(err.to_string().contains("must define [orangu].server"));
    }

    #[test]
    fn server_model_overrides_general_default_model() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu]\nserver = main\nmodel = general-default\n\n[main]\nendpoint = http://localhost:8100/v1\nmodel = server-specific\n\n[fallback]\nendpoint = http://localhost:8101/v1\n"
        )
        .unwrap();

        let conf = load_client_configuration(file.path()).unwrap();
        assert_eq!(conf.default_server, "main");
        assert_eq!(conf.default_model.as_deref(), Some("general-default"));
        // The server's own model wins over [orangu].model.
        assert_eq!(conf.llms["main"].model, "server-specific");
        // A server without its own model inherits [orangu].model.
        assert_eq!(conf.llms["fallback"].model, "general-default");
    }

    #[test]
    fn rejects_servers_sharing_an_endpoint_and_model() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu]\nserver = a\n\n[a]\nendpoint = http://localhost:8100/v1\nmodel = one\n\n[b]\nendpoint = http://localhost:8100/v1\nmodel = one\n"
        )
        .unwrap();

        let err = load_client_configuration(file.path()).unwrap_err();
        assert!(
            err.to_string().contains("distinct models"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn endpoint_uniqueness_ignores_trailing_slash_and_v1_suffix() {
        // `http://x` and `http://x/v1/` normalise to the same endpoint.
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu]\nserver = a\n\n[a]\nendpoint = http://localhost:8100\nmodel = one\n\n[b]\nendpoint = http://localhost:8100/v1/\nmodel = one\n"
        )
        .unwrap();

        let err = load_client_configuration(file.path()).unwrap_err();
        assert!(
            err.to_string().contains("distinct models"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn servers_may_share_an_endpoint_with_distinct_models() {
        // An orangu-coordinator instance fronts several role-specific models
        // behind one proxy address; servers distinguish themselves by `model`
        // rather than by endpoint in that setup.
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu]\nserver = a\n\n[a]\nendpoint = http://localhost:9000/v1\nmodel = one\nrole = all\n\n[b]\nendpoint = http://localhost:9000/v1\nmodel = two\nrole = explorer\n"
        )
        .unwrap();

        let conf = load_client_configuration(file.path()).expect("shared coordinator endpoint");
        assert_eq!(conf.find_server_for_role("explorer"), "b");
        assert_eq!(conf.llms["a"].endpoint, conf.llms["b"].endpoint);
    }

    #[test]
    fn embeddings_server_may_share_a_chat_server_endpoint() {
        // Ollama serves chat and embeddings on one endpoint; an `embeddings`
        // server is exempt from the unique-endpoint rule, so this must load.
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu]\nserver = chat\n\n[chat]\nendpoint = http://localhost:11434/v1\nmodel = llama3.1\n\n[embed]\nrole = embeddings\nendpoint = http://localhost:11434/v1\nmodel = nomic-embed-text\n"
        )
        .unwrap();
        let conf = load_client_configuration(file.path()).expect("shared embeddings endpoint");
        assert_eq!(conf.embeddings_server().as_deref(), Some("embed"));
    }

    #[test]
    fn parses_and_finds_roles() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu]\nserver = a\n\n[a]\nendpoint = http://x/v1\nmodel = m\nrole = all\n\n[b]\nendpoint = http://y/v1\nmodel = n\nrole = explorer\n"
        )
        .unwrap();
        let conf = load_client_configuration(file.path()).unwrap();
        assert_eq!(conf.find_server_for_role("explorer"), "b");
        assert_eq!(conf.find_server_for_role("review"), "a");
        assert_eq!(conf.find_server_for_role("missing"), "a"); // fallback

        // A custom fallback lets a caller prefer "whatever is currently
        // active" over the static config default when no server opts into
        // the role — e.g. review falling back to the live chat connection.
        assert_eq!(conf.find_server_for_role_or("explorer", "custom"), "b");
        assert_eq!(conf.find_server_for_role_or("review", "custom"), "custom");
    }

    #[test]
    fn embeddings_server_prefers_embeddings_role_then_default_all() {
        // An explicit `embeddings` role wins over the default `all` server.
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu]\nserver = a\n\n[a]\nendpoint = http://x/v1\nmodel = m\nrole = all\n\n[e]\nendpoint = http://z/v1\nmodel = k\nrole = embeddings\n"
        )
        .unwrap();
        let conf = load_client_configuration(file.path()).unwrap();
        assert_eq!(conf.embeddings_server().as_deref(), Some("e"));

        // With no `embeddings` server, the default `all` server serves it.
        let mut file2 = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file2,
            "[orangu]\nserver = a\n\n[a]\nendpoint = http://x/v1\nmodel = m\nrole = all\n"
        )
        .unwrap();
        let conf2 = load_client_configuration(file2.path()).unwrap();
        assert_eq!(conf2.embeddings_server().as_deref(), Some("a"));
    }
}
