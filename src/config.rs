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
    pub quotes: String,
    pub width: usize,
    #[serde(skip)]
    pub banner: Banner,
    pub feedback: bool,
    pub auto_rebase: bool,
    pub auto_squash: bool,
    pub terminal: String,
    pub platform: String,
    /// Where the workspace tab bar is drawn.
    pub workspaces: WorkspacePlacement,
}

#[derive(Clone, Debug, Serialize)]
pub struct LlmConfiguration {
    pub provider: String,
    pub endpoint: String,
    pub model: String,
    pub api_key: Option<String>,
    pub request_timeout_seconds: u64,
    pub max_tool_rounds: usize,
    /// Response-token cap for `/auto_review` requests; `0` disables the cap.
    pub review_max_tokens: u32,
    /// Response-token cap for normal chat/tool responses; `0` disables the cap.
    pub code_max_tokens: u32,
    pub system_prompt: String,
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

/// Default chat-response cap: `0`, no cap — normal coding responses are
/// open-ended.
pub fn default_code_max_tokens() -> u32 {
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
    let code_max_tokens = parse_client_field(&client, "code_max_tokens", default_code_max_tokens)?;
    let width = parse_client_field(&client, "width", default_virtual_width)?;
    let workspaces = parse_client_field(&client, "workspaces", WorkspacePlacement::default)?;
    let system_prompt = client.get("system_prompt").cloned().unwrap_or_default();

    // `[orangu].model` is the general default model id; a server section's own
    // `model` takes precedence over it.
    let default_model = client
        .get("model")
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());

    normalize_client_configuration(ClientAppConfiguration {
        default_server: client.get("server").cloned().unwrap_or_default(),
        default_model: default_model.clone(),
        llms: parse_llm_profiles(
            sections,
            timeout,
            max_tool_rounds,
            review_max_tokens,
            code_max_tokens,
            system_prompt,
            default_model,
        )?,
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
        terminal: client.get("terminal").cloned().unwrap_or_default(),
        platform: client.get("platform").cloned().unwrap_or_default(),
        workspaces,
    })
}

pub const DEFAULT_PLATFORM: &str = "github";

/// Parse a typed value from the `[orangu]` section, falling back to `default`
/// when the key is absent. Profile names and values may freely contain `.`
/// and `:` — unlike a generic INI loader, this parser never treats those as
/// nested-key separators.
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
fn parse_ini_sections(contents: &str) -> Result<HashMap<String, HashMap<String, String>>> {
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

    // Each server must point at a distinct endpoint: a server is one host, and
    // `/model` cycles the models that single host offers. Endpoints are compared
    // canonically, so `http://x` and `http://x/v1/` count as the same host.
    let mut endpoints: HashMap<&str, &str> = HashMap::new();
    for (name, llm) in &conf.llms {
        let canonical = llm.endpoint.strip_suffix("/v1").unwrap_or(&llm.endpoint);
        if let Some(existing) = endpoints.insert(canonical, name.as_str()) {
            return Err(anyhow!(
                "Servers '{existing}' and '{name}' share endpoint '{}'; each server must use a unique endpoint",
                llm.endpoint
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

            Ok((
                name,
                LlmConfiguration {
                    provider: values.get("provider").cloned().unwrap_or_default(),
                    endpoint: values.get("endpoint").cloned().unwrap_or_default(),
                    model,
                    api_key,
                    request_timeout_seconds: timeout,
                    max_tool_rounds,
                    review_max_tokens,
                    code_max_tokens,
                    system_prompt: system_prompt.clone(),
                },
            ))
        })
        .collect()
}

fn normalize_llm_configuration(llm: &mut LlmConfiguration) -> Result<()> {
    llm.provider = llm.provider.trim().to_string();
    llm.endpoint = llm.endpoint.trim().trim_end_matches('/').to_string();
    llm.model = llm.model.trim().to_string();
    llm.system_prompt = llm.system_prompt.trim().to_string();

    if llm.provider.is_empty() {
        return Err(anyhow!("LLM provider must not be empty"));
    }
    if llm.endpoint.is_empty() {
        return Err(anyhow!("LLM endpoint must not be empty"));
    }
    if llm.model.is_empty() {
        return Err(anyhow!("LLM model must not be empty"));
    }

    match llm.provider.to_lowercase().as_str() {
        "llama.cpp" | "openai" => Ok(()),
        _ => Err(anyhow!("Unsupported LLM provider '{}'", llm.provider)),
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
            "[orangu]\nserver = gemma\ntimeout = 45\nmax_tool_rounds = 12\n\n[gemma]\nprovider = llama.cpp\nendpoint = http://localhost:8100/v1\nmodel = ggml-org/gemma-4-E4B-it-GGUF\n\n[qwen]\nprovider = llama.cpp\nendpoint = http://localhost:8101/v1\nmodel = unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF\n"
        )
        .unwrap();

        let conf = load_client_configuration(file.path()).unwrap();
        assert_eq!(conf.default_server, "gemma");
        assert_eq!(conf.llms.len(), 2);
        assert_eq!(conf.llms["gemma"].provider, "llama.cpp");
        assert_eq!(conf.llms["gemma"].request_timeout_seconds, 45);
        assert_eq!(conf.llms["gemma"].max_tool_rounds, 12);
        // Absent platform defaults to GitHub.
        assert_eq!(conf.platform, "github");
    }

    #[test]
    fn parses_review_and_code_max_tokens_with_defaults() {
        // Absent keys: reviews capped at 512 tokens, chat responses uncapped.
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu]\nserver = a\n\n[a]\nprovider = llama.cpp\nendpoint = http://x/v1\nmodel = m\n"
        )
        .unwrap();
        let conf = load_client_configuration(file.path()).unwrap();
        assert_eq!(conf.llms["a"].review_max_tokens, 512);
        assert_eq!(conf.llms["a"].code_max_tokens, 0);

        // Explicit values land on every profile.
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu]\nserver = a\nreview_max_tokens = 2048\ncode_max_tokens = 4096\n\n[a]\nprovider = llama.cpp\nendpoint = http://x/v1\nmodel = m\n"
        )
        .unwrap();
        let conf = load_client_configuration(file.path()).unwrap();
        assert_eq!(conf.llms["a"].review_max_tokens, 2048);
        assert_eq!(conf.llms["a"].code_max_tokens, 4096);
    }

    #[test]
    fn parses_workspaces_placement_with_default_and_validation() {
        // Absent key defaults to the top placement.
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu]\nserver = a\n\n[a]\nprovider = llama.cpp\nendpoint = http://x/v1\nmodel = m\n"
        )
        .unwrap();
        let conf = load_client_configuration(file.path()).unwrap();
        assert_eq!(conf.workspaces, WorkspacePlacement::Top);

        // An explicit value is parsed case-insensitively.
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu]\nserver = a\nworkspaces = Bottom\n\n[a]\nprovider = llama.cpp\nendpoint = http://x/v1\nmodel = m\n"
        )
        .unwrap();
        let conf = load_client_configuration(file.path()).unwrap();
        assert_eq!(conf.workspaces, WorkspacePlacement::Bottom);

        // An unknown value is rejected, naming the key.
        let mut bad = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            bad,
            "[orangu]\nserver = a\nworkspaces = middle\n\n[a]\nprovider = llama.cpp\nendpoint = http://x/v1\nmodel = m\n"
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
            "[orangu]\nserver = a\nplatform = GitLab\n\n[a]\nprovider = llama.cpp\nendpoint = http://x/v1\nmodel = m\n"
        )
        .unwrap();
        let conf = load_client_configuration(file.path()).unwrap();
        assert_eq!(conf.platform, "gitlab");

        let mut bad = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            bad,
            "[orangu]\nserver = a\nplatform = bitbucket\n\n[a]\nprovider = llama.cpp\nendpoint = http://x/v1\nmodel = m\n"
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
            "[orangu]\nserver = Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M\n\n[Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M]\nprovider = llama.cpp\nendpoint = http://localhost:8100/v1\nmodel = bartowski/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M\n\n[Qwen_Qwen3.6-35B-A3B-GGUF]\nprovider = llama.cpp\nendpoint = http://localhost:8101/v1\nmodel = bartowski/Qwen_Qwen3.6-35B-A3B-GGUF\n"
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
            "[orangu]\nserver = a\ntimeout = soon\n\n[a]\nprovider = llama.cpp\nendpoint = http://x/v1\nmodel = m\n"
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
            "[orangu]\n\n[gemma]\nprovider = llama.cpp\nendpoint = http://localhost:8100/v1\nmodel = ggml-org/gemma-4-E4B-it-GGUF\n\n[qwen]\nprovider = llama.cpp\nendpoint = http://localhost:8101/v1\nmodel = unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF\n"
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
            "[orangu]\nserver = main\nmodel = general-default\n\n[main]\nprovider = llama.cpp\nendpoint = http://localhost:8100/v1\nmodel = server-specific\n\n[fallback]\nprovider = llama.cpp\nendpoint = http://localhost:8101/v1\n"
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
    fn rejects_servers_sharing_an_endpoint() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu]\nserver = a\n\n[a]\nprovider = llama.cpp\nendpoint = http://localhost:8100/v1\nmodel = one\n\n[b]\nprovider = llama.cpp\nendpoint = http://localhost:8100/v1\nmodel = two\n"
        )
        .unwrap();

        let err = load_client_configuration(file.path()).unwrap_err();
        assert!(
            err.to_string().contains("unique endpoint"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn endpoint_uniqueness_ignores_trailing_slash_and_v1_suffix() {
        // `http://x` and `http://x/v1/` normalise to the same endpoint.
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu]\nserver = a\n\n[a]\nprovider = llama.cpp\nendpoint = http://localhost:8100\nmodel = one\n\n[b]\nprovider = llama.cpp\nendpoint = http://localhost:8100/v1/\nmodel = two\n"
        )
        .unwrap();

        let err = load_client_configuration(file.path()).unwrap_err();
        assert!(
            err.to_string().contains("unique endpoint"),
            "unexpected error: {err:#}"
        );
    }
}
