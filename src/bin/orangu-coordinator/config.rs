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

//! Configuration for `orangu-coordinator`: a single `[orangu-coordinator]`
//! client section plus one section per llama.cpp-backed model, mirroring the
//! shape of `orangu.conf`'s server sections.

use anyhow::{Context, Result, anyhow};
use orangu::config::parse_ini_sections;
use std::{collections::HashMap, path::Path, path::PathBuf};

pub const CLIENT_SECTION: &str = "orangu-coordinator";

/// The conventional roles `orangu.conf` itself documents, in the order they
/// are listed there. Used to report a model for every role in
/// [`CoordinatorConfiguration::models_by_role`], not just the ones a given
/// `orangu-coordinator.conf` happens to define profiles for.
pub const KNOWN_ROLES: &[&str] = &["all", "code", "review", "explorer", "embeddings"];

#[derive(Clone, Debug)]
pub struct CoordinatorConfiguration {
    /// Host the proxy listens on, e.g. `127.0.0.1`.
    pub host: String,
    /// Port the proxy listens on.
    pub port: u16,
    /// How long to wait for a newly started llama.cpp to answer `/v1/models`
    /// before giving up and reporting an error to the caller.
    pub startup_timeout_seconds: u64,
    /// Request/response body size cap in bytes.
    pub max_body_bytes: usize,
    pub llms: HashMap<String, CoordinatorLlmEntry>,
    /// Name of the section whose `role` is `all`; used whenever a request's
    /// `model` field is absent or matches no configured entry.
    pub default_entry: String,
}

impl CoordinatorConfiguration {
    /// The `host:port` string to bind the proxy's listener to.
    pub fn listen_addr(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }

    /// The model each of [`KNOWN_ROLES`] resolves to: the model of the
    /// lexicographically-first profile tagged with that role (matching how
    /// [`crate::process::Coordinator::resolve_entry`] breaks ties), or, when
    /// no profile defines that role, the `all`-role default's model.
    pub fn models_by_role(&self) -> Vec<(&str, &str)> {
        let default_model = self.llms[&self.default_entry].model.as_str();
        KNOWN_ROLES
            .iter()
            .map(|&role| {
                let mut candidates: Vec<&CoordinatorLlmEntry> = self
                    .llms
                    .values()
                    .filter(|entry| entry.role == role)
                    .collect();
                candidates.sort_unstable_by(|a, b| a.name.cmp(&b.name));
                let model = candidates
                    .first()
                    .map(|entry| entry.model.as_str())
                    .unwrap_or(default_model);
                (role, model)
            })
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct CoordinatorLlmEntry {
    pub name: String,
    pub role: String,
    /// The model id a client request must carry (in its JSON `model` field)
    /// to be routed to this entry. Not configured directly — it's read off
    /// `llamacpp`'s own `-hf`/`--hf-repo` or `-m`/`--model` argument, so there
    /// is exactly one place a profile's model is named.
    pub model: String,
    /// Host llama.cpp will listen on once started via `llamacpp`. Read off
    /// `llamacpp`'s own `--host` argument (defaulting to `127.0.0.1`, same as
    /// llama.cpp itself, when absent). May differ between entries — e.g. one
    /// profile's model runs on another machine — even though at most one is
    /// ever active at a time.
    pub host: String,
    /// Port llama.cpp will listen on once started via `llamacpp`. Read off
    /// `llamacpp`'s own `--port` argument (defaulting to `8080`, same as
    /// llama.cpp itself, when absent).
    pub port: u16,
    /// Shell-style command line used to start llama.cpp for this entry, e.g.
    /// `llama-server -hf org/Model-GGUF --port 8100 --ctx-size 32768`.
    pub llamacpp: String,
    /// API key sent as `Authorization: Bearer <key>` on requests the
    /// coordinator makes to this llama.cpp once it is running. Required only
    /// when `llamacpp` itself starts the server with `--api-key`.
    pub api_key: Option<String>,
}

impl CoordinatorLlmEntry {
    /// The origin (`http://host:port`) requests are proxied to once this
    /// entry's llama.cpp is active.
    pub fn origin(&self) -> String {
        format!("http://{}:{}", self.host, self.port)
    }
}

pub(crate) fn default_startup_timeout() -> u64 {
    180
}

pub(crate) fn default_max_body_bytes() -> usize {
    64 * 1024 * 1024
}

pub(crate) fn default_host() -> String {
    "127.0.0.1".to_string()
}

pub(crate) fn default_port() -> u16 {
    9000
}

/// llama.cpp's own default port when `--port` is omitted.
fn default_llama_port() -> u16 {
    8080
}

fn parse_port(values: &HashMap<String, String>, name: &str, section: &str) -> Result<Option<u16>> {
    match values.get(name) {
        Some(value) => value
            .trim()
            .parse::<u16>()
            .map(Some)
            .map_err(|err| anyhow!("invalid value for [{section}].{name}: {err}")),
        None => Ok(None),
    }
}

/// Flags that name the model a `llama-server` invocation loads, in the order
/// they're preferred: an explicit Hugging Face repo id is what callers
/// naturally send as `model`, a local model path is the fallback.
const MODEL_REPO_FLAGS: &[&str] = &["-hf", "--hf-repo"];
const MODEL_PATH_FLAGS: &[&str] = &["-m", "--model"];

fn find_flag_value(argv: &[String], flags: &[&str]) -> Option<String> {
    for (index, arg) in argv.iter().enumerate() {
        for flag in flags {
            if arg == flag {
                return argv.get(index + 1).cloned();
            }
            if let Some(value) = arg.strip_prefix(&format!("{flag}=")) {
                return Some(value.to_string());
            }
        }
    }
    None
}

/// Reads the model id a profile's `llamacpp` command will load, so profiles
/// don't need to name their model a second time in a separate config key.
pub(crate) fn extract_model_id(argv: &[String]) -> Option<String> {
    find_flag_value(argv, MODEL_REPO_FLAGS).or_else(|| find_flag_value(argv, MODEL_PATH_FLAGS))
}

const HOST_FLAGS: &[&str] = &["--host"];
const PORT_FLAGS: &[&str] = &["--port"];

/// Reads the host and port a profile's `llamacpp` command will bind to, so
/// profiles don't need to repeat them in separate config keys. Falls back to
/// llama.cpp's own defaults (`127.0.0.1:8080`) when `--host`/`--port` are
/// absent from the command line.
fn extract_host_and_port(name: &str, argv: &[String]) -> Result<(String, u16)> {
    let host = find_flag_value(argv, HOST_FLAGS).unwrap_or_else(default_host);
    let port = match find_flag_value(argv, PORT_FLAGS) {
        Some(value) => value
            .parse::<u16>()
            .map_err(|err| anyhow!("[{name}].llamacpp has an invalid --port value: {err}"))?,
        None => default_llama_port(),
    };
    Ok((host, port))
}

pub fn default_coordinator_config_path() -> Option<PathBuf> {
    let cwd_path = std::env::current_dir()
        .ok()?
        .join("orangu-coordinator.conf");
    if cwd_path.exists() {
        return Some(cwd_path);
    }

    let config_path = home::home_dir()?.join(".orangu/orangu-coordinator.conf");
    config_path.exists().then_some(config_path)
}

pub fn load_coordinator_configuration(path: &Path) -> Result<CoordinatorConfiguration> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read configuration {}", path.display()))?;
    let mut sections = parse_ini_sections(&contents)
        .with_context(|| format!("failed to parse configuration {}", path.display()))?;

    let client = sections.remove(CLIENT_SECTION).unwrap_or_default();

    let host = client
        .get("host")
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(default_host);
    let port = parse_port(&client, "port", CLIENT_SECTION)?.unwrap_or_else(default_port);

    let startup_timeout_seconds = match client.get("startup_timeout") {
        Some(value) => value.trim().parse::<u64>().map_err(|err| {
            anyhow!("invalid value for [{CLIENT_SECTION}].startup_timeout: {err}")
        })?,
        None => default_startup_timeout(),
    };

    let max_body_bytes = match client.get("max_body_bytes") {
        Some(value) => value
            .trim()
            .parse::<usize>()
            .map_err(|err| anyhow!("invalid value for [{CLIENT_SECTION}].max_body_bytes: {err}"))?,
        None => default_max_body_bytes(),
    };

    if sections.is_empty() {
        return Err(anyhow!("At least one named LLM profile must be defined"));
    }

    let llms = parse_llm_profiles(sections)?;

    let default_entry = {
        let mut all_entries: Vec<&str> = llms
            .values()
            .filter(|entry| entry.role == "all")
            .map(|entry| entry.name.as_str())
            .collect();
        all_entries.sort_unstable();
        all_entries.first().map(|name| name.to_string()).ok_or_else(|| {
            anyhow!(
                "At least one profile must specify (or default to) role = all, to serve as the fallback when a request's model doesn't match a specific profile"
            )
        })?
    };

    Ok(CoordinatorConfiguration {
        host,
        port,
        startup_timeout_seconds,
        max_body_bytes,
        llms,
        default_entry,
    })
}

fn parse_llm_profiles(
    sections: HashMap<String, HashMap<String, String>>,
) -> Result<HashMap<String, CoordinatorLlmEntry>> {
    sections
        .into_iter()
        .map(|(name, values)| {
            let role = values
                .get("role")
                .map(|value| value.trim().to_lowercase())
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| "all".to_string());

            let llamacpp = values
                .get("llamacpp")
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
                .ok_or_else(|| anyhow!("[{name}].llamacpp must not be empty"))?;
            // Fail fast on a malformed command line rather than at first use.
            let argv = shell_words::split(&llamacpp)
                .map_err(|err| anyhow!("[{name}].llamacpp is not a valid command line: {err}"))?;
            let model = extract_model_id(&argv).ok_or_else(|| {
                anyhow!(
                    "[{name}].llamacpp must specify a model via -hf/--hf-repo or -m/--model, so requests can be routed to it"
                )
            })?;
            let (host, port) = extract_host_and_port(&name, &argv)?;

            let api_key = values
                .get("api_key")
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty());

            Ok((
                name.clone(),
                CoordinatorLlmEntry {
                    name,
                    role,
                    model,
                    host,
                    port,
                    llamacpp,
                    api_key,
                },
            ))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn loads_minimal_configuration_with_defaults() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-coordinator]\n\n[main]\nrole = all\nllamacpp = llama-server -hf org/gemma --port 8100\n"
        )
        .unwrap();

        let conf = load_coordinator_configuration(file.path()).unwrap();
        assert_eq!(conf.listen_addr(), "127.0.0.1:9000");
        assert_eq!(conf.startup_timeout_seconds, 180);
        assert_eq!(conf.default_entry, "main");
        assert_eq!(conf.llms["main"].host, "127.0.0.1");
        assert_eq!(conf.llms["main"].origin(), "http://127.0.0.1:8100");
        assert_eq!(conf.llms["main"].model, "org/gemma");
    }

    #[test]
    fn requires_at_least_one_all_role() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-coordinator]\n\n[explorer]\nrole = explorer\nllamacpp = llama-server -hf org/qwen --port 8200\n"
        )
        .unwrap();

        let err = load_coordinator_configuration(file.path()).unwrap_err();
        assert!(
            err.to_string().contains("role = all"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn role_defaults_to_all() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-coordinator]\n\n[main]\nllamacpp = llama-server -hf org/gemma --port 8100\n"
        )
        .unwrap();

        let conf = load_coordinator_configuration(file.path()).unwrap();
        assert_eq!(conf.llms["main"].role, "all");
        assert_eq!(conf.default_entry, "main");
    }

    #[test]
    fn allows_profiles_sharing_a_model() {
        // Two profiles may reference the same model (e.g. one per role, even
        // when the underlying model happens to be identical); this is not an
        // error, just an ambiguity `resolve_entry` breaks deterministically.
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-coordinator]\n\n[a]\nrole = all\nllamacpp = llama-server -hf org/gemma --port 8100\n\n[b]\nrole = explorer\nllamacpp = llama-server -hf org/gemma --port 8200\n"
        )
        .unwrap();

        let conf = load_coordinator_configuration(file.path()).unwrap();
        assert_eq!(conf.llms["a"].model, "org/gemma");
        assert_eq!(conf.llms["b"].model, "org/gemma");
    }

    #[test]
    fn rejects_malformed_llamacpp_command() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-coordinator]\n\n[main]\nrole = all\nllamacpp = llama-server --chat-template-kwargs '{{\"unterminated\n"
        )
        .unwrap();

        let err = load_coordinator_configuration(file.path()).unwrap_err();
        assert!(
            err.to_string().contains("llamacpp"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn rejects_llamacpp_command_without_a_model() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-coordinator]\n\n[main]\nrole = all\nllamacpp = llama-server --port 8100\n"
        )
        .unwrap();

        let err = load_coordinator_configuration(file.path()).unwrap_err();
        assert!(
            err.to_string().contains("must specify a model"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn rejects_invalid_port_in_llamacpp() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-coordinator]\n\n[main]\nrole = all\nllamacpp = llama-server -hf org/gemma --port not-a-port\n"
        )
        .unwrap();

        let err = load_coordinator_configuration(file.path()).unwrap_err();
        assert!(
            err.to_string().contains("invalid --port"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn host_and_port_default_when_llamacpp_omits_them() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-coordinator]\n\n[main]\nrole = all\nllamacpp = llama-server -hf org/gemma\n"
        )
        .unwrap();

        let conf = load_coordinator_configuration(file.path()).unwrap();
        assert_eq!(conf.llms["main"].host, "127.0.0.1");
        assert_eq!(conf.llms["main"].port, 8080);
    }

    #[test]
    fn parses_multiple_roles_with_distinct_hosts() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-coordinator]\nhost = 0.0.0.0\nport = 9100\nstartup_timeout = 30\n\n[main]\nrole = all\nllamacpp = llama-server -hf org/gemma --port 8100\n\n[explorer]\nrole = explorer\nllamacpp = llama-server -hf org/qwen --host 192.168.1.20 --port 8200\napi_key = secret\n"
        )
        .unwrap();

        let conf = load_coordinator_configuration(file.path()).unwrap();
        assert_eq!(conf.listen_addr(), "0.0.0.0:9100");
        assert_eq!(conf.startup_timeout_seconds, 30);
        assert_eq!(conf.llms.len(), 2);
        assert_eq!(conf.llms["main"].host, "127.0.0.1");
        assert_eq!(conf.llms["explorer"].origin(), "http://192.168.1.20:8200");
        assert_eq!(conf.llms["explorer"].model, "org/qwen");
        assert_eq!(conf.llms["explorer"].api_key.as_deref(), Some("secret"));
    }

    #[test]
    fn models_by_role_falls_back_to_all_for_undefined_roles() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-coordinator]\n\n[main]\nrole = all\nllamacpp = llama-server -hf org/gemma --port 8100\n\n[explorer]\nrole = explorer\nllamacpp = llama-server -hf org/qwen --port 8200\n"
        )
        .unwrap();

        let conf = load_coordinator_configuration(file.path()).unwrap();
        let models: std::collections::HashMap<&str, &str> =
            conf.models_by_role().into_iter().collect();
        assert_eq!(models.len(), KNOWN_ROLES.len());
        assert_eq!(models["all"], "org/gemma");
        assert_eq!(models["explorer"], "org/qwen");
        // code/review/embeddings have no profile of their own — fall back to
        // the `all`-role default's model.
        assert_eq!(models["code"], "org/gemma");
        assert_eq!(models["review"], "org/gemma");
        assert_eq!(models["embeddings"], "org/gemma");
    }

    #[test]
    fn extracts_model_id_from_hf_repo_and_model_path_flags() {
        assert_eq!(
            extract_model_id(
                &shell_words::split("llama-server -hf org/gemma --port 8100").unwrap()
            ),
            Some("org/gemma".to_string())
        );
        assert_eq!(
            extract_model_id(
                &shell_words::split("llama-server --hf-repo=org/gemma --port 8100").unwrap()
            ),
            Some("org/gemma".to_string())
        );
        assert_eq!(
            extract_model_id(
                &shell_words::split("llama-server -m /models/gemma.gguf --port 8100").unwrap()
            ),
            Some("/models/gemma.gguf".to_string())
        );
        assert_eq!(
            extract_model_id(&shell_words::split("llama-server --port 8100").unwrap()),
            None
        );
    }
}
