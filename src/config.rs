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
use config::{Config, FileFormat};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use crate::tui::Banner;

#[derive(Clone, Debug, Serialize)]
pub struct ClientAppConfiguration {
    pub default_model: String,
    pub llms: HashMap<String, LlmConfiguration>,
    pub quotes: String,
    pub width: usize,
    #[serde(skip)]
    pub banner: Banner,
    pub feedback: bool,
    pub auto_rebase: bool,
    pub auto_squash: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct LlmConfiguration {
    pub provider: String,
    pub endpoint: String,
    pub model: String,
    pub api_key: Option<String>,
    pub request_timeout_seconds: u64,
    pub max_tool_rounds: usize,
    pub system_prompt: String,
}

#[derive(Clone, Debug, Deserialize)]
struct ClientConfiguration {
    #[serde(default)]
    model: String,
    #[serde(default = "default_timeout")]
    timeout: u64,
    #[serde(default = "default_llm_max_tool_rounds")]
    max_tool_rounds: usize,
    #[serde(default)]
    system_prompt: String,
    #[serde(default)]
    quotes: String,
    #[serde(default = "default_virtual_width")]
    width: usize,
    #[serde(default)]
    banner: String,
    #[serde(default)]
    feedback: String,
    #[serde(default)]
    auto_rebase: String,
    #[serde(default)]
    auto_squash: String,
}

pub fn default_virtual_width() -> usize {
    512
}

#[derive(Debug, Deserialize)]
struct ClientConfRoot {
    #[serde(rename = "orangu")]
    client: ClientConfiguration,
    #[serde(flatten)]
    llms: HashMap<String, HashMap<String, String>>,
}

pub fn default_timeout() -> u64 {
    1_800
}

pub fn default_llm_max_tool_rounds() -> usize {
    10
}

pub fn parse_feedback_bool(s: &str) -> bool {
    matches!(s.trim().to_lowercase().as_str(), "on" | "true" | "1")
}

pub fn load_client_configuration(path: &Path) -> Result<ClientAppConfiguration> {
    let conf = Config::builder()
        .add_source(config::File::from(path).format(FileFormat::Ini))
        .build()
        .with_context(|| format!("failed to read configuration {}", path.display()))?;
    let root = conf
        .try_deserialize::<ClientConfRoot>()
        .with_context(|| format!("failed to parse configuration {}", path.display()))?;

    normalize_client_configuration(ClientAppConfiguration {
        default_model: root.client.model,
        llms: parse_llm_profiles(
            root.llms,
            root.client.timeout,
            root.client.max_tool_rounds,
            root.client.system_prompt,
        )?,
        quotes: root.client.quotes,
        width: root.client.width,
        banner: root.client.banner.parse().unwrap_or_default(),
        feedback: parse_feedback_bool(&root.client.feedback),
        auto_rebase: parse_feedback_bool(&root.client.auto_rebase),
        auto_squash: parse_feedback_bool(&root.client.auto_squash),
    })
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
    conf.default_model = conf.default_model.trim().to_string();

    if conf.llms.is_empty() {
        return Err(anyhow!("At least one named LLM profile must be defined"));
    }

    for llm in conf.llms.values_mut() {
        normalize_llm_configuration(llm)?;
    }

    if conf.default_model.is_empty() {
        if conf.llms.len() == 1 {
            conf.default_model = conf
                .llms
                .keys()
                .next()
                .cloned()
                .ok_or_else(|| anyhow!("Missing LLM model definition"))?;
        } else {
            return Err(anyhow!(
                "Client configuration must define [orangu].model when multiple LLM profiles are configured"
            ));
        }
    }

    if !conf.llms.contains_key(&conf.default_model) {
        return Err(anyhow!(
            "Client model '{}' is not defined in the configuration",
            conf.default_model
        ));
    }

    Ok(conf)
}

fn parse_llm_profiles(
    sections: HashMap<String, HashMap<String, String>>,
    timeout: u64,
    max_tool_rounds: usize,
    system_prompt: String,
) -> Result<HashMap<String, LlmConfiguration>> {
    sections
        .into_iter()
        .map(|(name, values)| {
            let api_key = values
                .get("api_key")
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty());
            let api_key_env = values
                .get("api_key_env")
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty());
            let api_key = match (api_key, api_key_env) {
                (Some(key), _) => Some(key),
                (None, Some(env_var)) => std::env::var(&env_var).ok(),
                (None, None) => None,
            };

            Ok((
                name,
                LlmConfiguration {
                    provider: values.get("provider").cloned().unwrap_or_default(),
                    endpoint: values.get("endpoint").cloned().unwrap_or_default(),
                    model: values.get("model").cloned().unwrap_or_default(),
                    api_key,
                    request_timeout_seconds: timeout,
                    max_tool_rounds,
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
            "[orangu]\nmodel = gemma\ntimeout = 45\nmax_tool_rounds = 12\n\n[gemma]\nprovider = llama.cpp\nendpoint = http://localhost:8100/v1\nmodel = ggml-org/gemma-4-E4B-it-GGUF\n\n[qwen]\nprovider = llama.cpp\nendpoint = http://localhost:8100/v1\nmodel = unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF\n"
        )
        .unwrap();

        let conf = load_client_configuration(file.path()).unwrap();
        assert_eq!(conf.default_model, "gemma");
        assert_eq!(conf.llms.len(), 2);
        assert_eq!(conf.llms["gemma"].provider, "llama.cpp");
        assert_eq!(conf.llms["gemma"].request_timeout_seconds, 45);
        assert_eq!(conf.llms["gemma"].max_tool_rounds, 12);
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
    fn requires_default_model_when_multiple_profiles_exist() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu]\n\n[gemma]\nprovider = llama.cpp\nendpoint = http://localhost:8100/v1\nmodel = ggml-org/gemma-4-E4B-it-GGUF\n\n[qwen]\nprovider = llama.cpp\nendpoint = http://localhost:8100/v1\nmodel = unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF\n"
        )
        .unwrap();

        let err = load_client_configuration(file.path()).unwrap_err();
        assert!(err.to_string().contains("must define [orangu].model"));
    }
}
