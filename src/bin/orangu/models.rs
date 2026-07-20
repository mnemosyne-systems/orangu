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

use crate::*;

#[derive(Debug, Deserialize)]
pub(crate) struct ModelsResponse {
    #[serde(default)]
    data: Vec<ModelEntry>,
    #[serde(default)]
    models: Vec<ModelEntry>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ModelEntry {
    #[serde(default)]
    id: String,
    #[serde(default)]
    model: String,
    #[serde(default)]
    name: String,
}

impl ModelsResponse {
    /// Every advertised model id, preferring `id`, then `model`, then `name`
    /// for entries that only carry one of those fields.
    ///
    /// `data` and `models` are two different shapes a `/v1/models` response
    /// might use (plain OpenAI vs. orangu-server); most
    /// servers populate only one, leaving the other empty, but a response
    /// that happens to fill in both would otherwise list every model twice —
    /// once from each field — so entries are deduplicated by their resolved
    /// id, keeping the first occurrence.
    pub(crate) fn model_ids(&self) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        self.data
            .iter()
            .chain(self.models.iter())
            .filter_map(|entry| {
                if !entry.id.is_empty() {
                    Some(entry.id.clone())
                } else if !entry.model.is_empty() {
                    Some(entry.model.clone())
                } else if !entry.name.is_empty() {
                    Some(entry.name.clone())
                } else {
                    None
                }
            })
            .filter(|id| seen.insert(id.clone()))
            .collect()
    }
}

/// Build a GET request to a server's `/v1/models` endpoint, attaching the
/// optional bearer token. OpenAI-compatible servers — including an orangu-server
/// server started with `--api-key` — require `Authorization: Bearer <key>` on
/// every `/v1/*` endpoint, not just chat completions.
pub(crate) fn models_request(
    http_client: &reqwest::Client,
    endpoint: &str,
    api_key: Option<&str>,
) -> reqwest::RequestBuilder {
    let url = format!("{}/v1/models", normalized_openai_endpoint(endpoint));
    let request = http_client.get(url);
    match api_key {
        Some(key) => request.bearer_auth(key),
        None => request,
    }
}

/// Probe the active server and return its header status together with the list
/// of model ids it advertises (used for `/model` completion). `model_ok` is set
/// when the active wire model id is among the advertised models.
///
/// An orangu-coordinator endpoint is checked first (see [`probe_coordinator`])
/// and, when confirmed, `/v1/models` is never called: against a coordinator
/// that request is not a side-effect-free status check — an unmatched `model`
/// falls back to the `all` role and can start (or wait on) a different
/// profile's llama.cpp, which would both stall this per-cycle refresh and
/// fight whatever model real requests are actively using. `server_ok` and
/// `model_ok` are simply `true` in that case: the coordinator answering *is*
/// the health check, and which model is actually loaded is automatic, so
/// there is no fixed "wrong model" state to report.
pub(crate) async fn probe_header_status(
    http_client: &reqwest::Client,
    workspace: &Path,
    active_model_id: &str,
    profile: &LlmConfiguration,
    endpoint: Option<&str>,
) -> (orangu::tui::HeaderStatus, Vec<String>) {
    let workspace_ok = workspace.exists();
    let mut server_ok = false;
    let mut model_ok = false;
    let mut available_models = Vec::new();
    let mut is_coordinator = false;

    if let Some(endpoint) = endpoint {
        let reachability =
            probe_server_reachability(http_client, endpoint, profile.api_key.as_deref()).await;
        is_coordinator = reachability.is_coordinator;
        server_ok = reachability.reachable;
        available_models = reachability.available_models;
        model_ok = is_coordinator || available_models.iter().any(|id| id == active_model_id);
    }

    (
        orangu::tui::HeaderStatus {
            workspace_ok,
            server_ok: orangu::tui::ConnStatus::from_bool(server_ok),
            model_ok: orangu::tui::ConnStatus::from_bool(model_ok),
            is_coordinator,
        },
        available_models,
    )
}

/// Whether a server endpoint responds at all, and what it advertises —
/// independent of which model is currently pinned. Shared by
/// [`probe_header_status`] (which additionally checks whether the pinned
/// model is among what's advertised) and startup server auto-failover
/// (`resolve_startup_server`), which only cares whether a candidate server
/// answers at all.
pub(crate) struct ServerReachability {
    pub is_coordinator: bool,
    pub reachable: bool,
    pub available_models: Vec<String>,
}

async fn probe_server_reachability(
    http_client: &reqwest::Client,
    endpoint: &str,
    api_key: Option<&str>,
) -> ServerReachability {
    let coordinator = probe_coordinator(http_client, endpoint, api_key).await;
    if coordinator.is_coordinator {
        return ServerReachability {
            is_coordinator: true,
            reachable: true,
            available_models: coordinator.models,
        };
    }
    match models_request(http_client, endpoint, api_key).send().await {
        Ok(response) if response.status().is_success() => {
            let available_models = response
                .json::<ModelsResponse>()
                .await
                .map(|models| models.model_ids())
                .unwrap_or_default();
            ServerReachability {
                is_coordinator: false,
                reachable: true,
                available_models,
            }
        }
        _ => ServerReachability {
            is_coordinator: false,
            reachable: false,
            available_models: Vec::new(),
        },
    }
}

/// What `GET /v1/coordinator` reports.
struct CoordinatorProbe {
    /// Whether the endpoint answered with `"orangu_coordinator": true` — the
    /// marker an orangu-coordinator proxy exposes and neither a direct orangu-server nor a
    /// generic OpenAI-compatible server does.
    is_coordinator: bool,
    /// Every distinct model named in the response's `models` map (the model
    /// each conventional role currently resolves to), deduplicated — used for
    /// `/model` completion in place of a `/v1/models` probe.
    models: Vec<String>,
}

/// `GET /v1/coordinator`, side-effect-free and safe to call whether or not
/// any profile's orangu-server is currently active. Any failure (unreachable,
/// non-success status, unexpected body) is treated as "not a coordinator"
/// rather than an error, since this is just an identity probe alongside the
/// main status refresh. Thin wrapper over the shared library probe (also
/// used by [`crate::explorer`]-style code with no `HeaderStatus` of its own)
/// that adapts its `Option<Vec<String>>` into this struct's shape.
async fn probe_coordinator(
    http_client: &reqwest::Client,
    endpoint: &str,
    api_key: Option<&str>,
) -> CoordinatorProbe {
    match orangu::llm::probe_coordinator(http_client, endpoint, api_key).await {
        Some(models) => CoordinatorProbe {
            is_coordinator: true,
            models,
        },
        None => CoordinatorProbe {
            is_coordinator: false,
            models: Vec::new(),
        },
    }
}

/// Decide whether an idle refresh should switch the pinned model. When the
/// server is up and advertising models but no longer serves the one we are
/// pinned to (e.g. an orangu-server swapped the loaded model while we sat
/// idle), return the model id to switch to so the header banner can reflect the
/// change; otherwise `None`. Reuses the model list the header probe already
/// fetched, so no extra request is made.
pub(crate) fn idle_model_switch_target(
    status: orangu::tui::HeaderStatus,
    available_models: &[String],
) -> Option<&str> {
    if status.server_ok == orangu::tui::ConnStatus::Ok
        && status.model_ok == orangu::tui::ConnStatus::Failed
    {
        available_models.first().map(String::as_str)
    } else {
        None
    }
}

/// If the active server is not serving the configured model at startup, switch
/// to a model the server actually advertises. Returns `(old, new)` model ids
/// when a switch happened. The server (endpoint, provider, system prompt) is
/// unchanged — only the wire model id moves.
pub(crate) async fn try_startup_model_switch(
    http_client: &reqwest::Client,
    profile: &LlmConfiguration,
    active_model_id: &mut String,
    endpoint: Option<&str>,
) -> Option<(String, String)> {
    let endpoint = endpoint?;
    let response = models_request(http_client, endpoint, profile.api_key.as_deref())
        .send()
        .await
        .ok()?;
    if !response.status().is_success() {
        return None;
    }
    let models = response.json::<ModelsResponse>().await.ok()?;
    let available = models.model_ids();

    // The server already serves the configured model — nothing to switch.
    if available.iter().any(|model| model == active_model_id) {
        return None;
    }

    // Otherwise move to the first model the server actually offers.
    let new_model = available.into_iter().next()?;
    let old = std::mem::replace(active_model_id, new_model.clone());
    Some((old, new_model))
}

/// Whether `active_model`'s connection is confirmed to be an
/// orangu-coordinator proxy rather than a directly-connected orangu-server
/// — see [`orangu::llm::probe_coordinator`]. `false` if the model
/// isn't configured or nothing is currently connected.
pub(crate) async fn is_active_connection_a_coordinator(
    http_client: &reqwest::Client,
    config: &ClientAppConfiguration,
    active_model: &str,
    endpoint: Option<&str>,
) -> bool {
    match (config.llms.get(active_model), endpoint) {
        (Some(profile), Some(endpoint)) => {
            orangu::llm::probe_coordinator(http_client, endpoint, profile.api_key.as_deref())
                .await
                .is_some()
        }
        _ => false,
    }
}

/// Resolves which server semantic `/search` should use, probing it to
/// confirm it is actually embeddings-capable right now. Returns the
/// resolved server's name (a key into `config.llms`) if the probe succeeds,
/// or an empty string otherwise — matching `CommandContext::embeddings_server`'s
/// existing "empty means unavailable" convention.
///
/// Once a coordinator is confirmed (`is_coordinator`), it alone owns every
/// model/role decision: a local `role = embeddings` section in orangu.conf
/// is never consulted — the active connection is reused as-is with `.model`
/// forced to `embeddings`, and the coordinator resolves that to whatever
/// real model actually backs it (falling back to `all` if it has none).
///
/// Otherwise, [`ClientAppConfiguration::embeddings_server`]'s existing
/// resolution (a dedicated `role = embeddings` section, or the default
/// server if it's `all`/`embeddings`) is used, exactly as before this
/// existed. Called both at startup and whenever `/server`/`/reload` selects
/// a new connection, since which server is embeddings-capable can change
/// along with it.
pub(crate) async fn detect_embeddings_server(
    config: &ClientAppConfiguration,
    active_model: &str,
    is_coordinator: bool,
) -> String {
    if is_coordinator {
        let Some(profile) = config.llms.get(active_model) else {
            return String::new();
        };
        let mut embeddings_profile = profile.clone();
        embeddings_profile.model = "embeddings".to_string();
        return match orangu::embeddings::probe_endpoint(&embeddings_profile).await {
            Ok(()) => active_model.to_string(),
            Err(_) => String::new(),
        };
    }
    match config.embeddings_server() {
        Some(name) => match config.llms.get(&name) {
            Some(profile) => match orangu::embeddings::probe_endpoint(profile).await {
                Ok(()) => name,
                Err(_) => String::new(),
            },
            None => String::new(),
        },
        None => String::new(),
    }
}

/// Which server startup connectivity resolution settled on, and how.
pub(crate) struct StartupServerResolution {
    /// The chosen server's config name — the default if it (or nothing else
    /// configured) responded, otherwise the first alternate that did.
    pub server: String,
    /// Whether a different server than the configured default was chosen.
    pub switched: bool,
    pub is_coordinator: bool,
    /// Models the chosen server advertised (empty if none responded at all).
    pub available_models: Vec<String>,
}

/// The order candidate servers should be tried in at startup: the default
/// first, then every other configured server from `server_names` in order,
/// skipping the default if it appears there too (it was already tried).
fn startup_server_candidates<'a>(default: &'a str, server_names: &'a [String]) -> Vec<&'a str> {
    std::iter::once(default)
        .chain(
            server_names
                .iter()
                .map(String::as_str)
                .filter(|name| *name != default),
        )
        .collect()
}

/// If the default configured server doesn't respond, cycles through the
/// other configured server sections (in `server_names` order, which excludes
/// nothing — the default is simply skipped when encountered) until one
/// does, and uses that instead. If none respond, stays on the default, which
/// will simply show red status once probed normally from then on.
pub(crate) async fn resolve_startup_server(
    http_client: &reqwest::Client,
    config: &ClientAppConfiguration,
    server_names: &[String],
) -> StartupServerResolution {
    let default = config.default_server.clone();
    for (index, name) in startup_server_candidates(&default, server_names)
        .into_iter()
        .enumerate()
    {
        let Some(profile) = config.llms.get(name) else {
            continue;
        };
        let reachability =
            probe_server_reachability(http_client, &profile.endpoint, profile.api_key.as_deref())
                .await;
        if reachability.reachable {
            return StartupServerResolution {
                server: name.to_string(),
                switched: index > 0,
                is_coordinator: reachability.is_coordinator,
                available_models: reachability.available_models,
            };
        }
    }
    StartupServerResolution {
        server: default,
        switched: false,
        is_coordinator: false,
        available_models: Vec::new(),
    }
}

/// Everything startup connectivity resolution figures out, ready for the
/// main loop to apply in one shot once the background task running
/// [`resolve_startup_state`] completes.
pub(crate) struct StartupResolution {
    pub server: String,
    pub switched_server: bool,
    pub active_model_id: String,
    /// `Some((old, new))` if the resolved server didn't serve the model id
    /// its section configures, and orangu switched to one it does advertise.
    pub model_switched: Option<(String, String)>,
    pub embeddings_server: String,
}

/// Runs the full startup connectivity resolution: which server to use (with
/// auto-failover, see [`resolve_startup_server`]), whether its pinned model
/// needs switching to one it actually advertises, and semantic `/search`'s
/// embeddings detection. Meant to be run as a single background task so the
/// UI can render immediately instead of blocking on it — see `main.rs`.
pub(crate) async fn resolve_startup_state(
    http_client: reqwest::Client,
    config: ClientAppConfiguration,
    server_names: Vec<String>,
) -> StartupResolution {
    let server_resolution = resolve_startup_server(&http_client, &config, &server_names).await;
    let profile = config
        .llms
        .get(&server_resolution.server)
        .expect("resolve_startup_server only returns configured server names");

    let mut active_model_id = profile.model.clone();
    let mut model_switched = None;
    // Reuses the model list resolve_startup_server's own reachability probe
    // already fetched, rather than probing `/v1/models` a second time.
    if !server_resolution.is_coordinator
        && !server_resolution.available_models.is_empty()
        && !server_resolution
            .available_models
            .contains(&active_model_id)
        && let Some(new_model) = server_resolution.available_models.first()
    {
        model_switched = Some((active_model_id.clone(), new_model.clone()));
        active_model_id = new_model.clone();
    }

    let embeddings_server = detect_embeddings_server(
        &config,
        &server_resolution.server,
        server_resolution.is_coordinator,
    )
    .await;

    StartupResolution {
        server: server_resolution.server,
        switched_server: server_resolution.switched,
        active_model_id,
        model_switched,
        embeddings_server,
    }
}

/// Resolves which server a `/review`/`/auto_review` request should use.
///
/// Once a coordinator is confirmed (`is_coordinator`), it alone owns every
/// model/role decision: orangu.conf's local `role = review` sections are
/// never consulted — the active connection is reused as-is with `.model`
/// forced to `review`, and the coordinator resolves that to whatever real
/// model actually backs it (falling back to `all` if it has none).
///
/// Otherwise (no coordinator), a dedicated `role = review` section is used
/// if one is configured, otherwise whatever the interactive chat is
/// currently connected to (unchanged from before this existed) — with
/// `endpoint`/`model` taken from the live, possibly-`/server`-or-`/model`-
/// switched values rather than the static config in that fallback case,
/// exactly like every other chat request; a dedicated review section
/// instead uses its own configured `endpoint`/`model` untouched, since it
/// may be a genuinely different server.
///
/// `None` only if `active_model` itself somehow isn't in `config.llms`,
/// which should not happen in practice.
pub(crate) fn review_prompt_profile(
    config: &ClientAppConfiguration,
    active_model: &str,
    active_model_id: &str,
    current_endpoint: &str,
    is_coordinator: bool,
) -> Option<LlmConfiguration> {
    if is_coordinator {
        let profile = config.llms.get(active_model)?;
        return Some(coordinator_role_profile(
            profile,
            current_endpoint,
            "review",
        ));
    }
    let review_server = config.find_server_for_role_or("review", active_model);
    let profile = config.llms.get(&review_server)?;
    let mut prompt_profile = profile.clone();
    if review_server == active_model {
        prompt_profile.endpoint = current_endpoint.to_string();
        prompt_profile.model = active_model_id.to_string();
    }
    Some(prompt_profile)
}

/// Clones `profile` with `.endpoint` pinned to the live connection and
/// `.model` forced to `role` (a conventional role name: `all`, `review`,
/// `explorer`, `embeddings`, ...) — used once a coordinator is confirmed,
/// since it alone decides which real model backs each role, regardless of
/// what orangu.conf's active section itself names.
pub(crate) fn coordinator_role_profile(
    profile: &LlmConfiguration,
    endpoint: &str,
    role: &str,
) -> LlmConfiguration {
    let mut prompt_profile = profile.clone();
    prompt_profile.endpoint = endpoint.to_string();
    prompt_profile.model = role.to_string();
    prompt_profile
}

/// Sends orangu-coordinator a "start warming this role up now" hint (`POST
/// /v1/coordinator/activate`) and returns without waiting for it to
/// complete — see doc/COORDINATOR.md's pre-warming section. Fired once at
/// the start of `/review`/`/auto_review` so a coordinator can begin
/// swapping to the `review` role's model in parallel with whatever local
/// work (diff collection, the auto-review prestart screen, ...) happens
/// before the first real request, instead of only starting the swap then.
///
/// Fire-and-forget by design: the real request that follows will trigger
/// (and wait for) the same swap on its own if this hint didn't get there
/// first or failed for any reason (not actually a coordinator, unknown
/// role, connection error, ...), so there is nothing to check or retry
/// here.
pub(crate) fn spawn_coordinator_activation_hint(
    http_client: reqwest::Client,
    endpoint: String,
    model_hint: String,
    api_key: Option<String>,
) {
    tokio::spawn(async move {
        let url = format!(
            "{}/v1/coordinator/activate",
            normalized_openai_endpoint(&endpoint)
        );
        let mut request = http_client
            .post(url)
            .json(&serde_json::json!({ "model": model_hint }));
        if let Some(key) = api_key {
            request = request.bearer_auth(key);
        }
        let _ = request.send().await;
    });
}

#[cfg(test)]
mod tests {
    use orangu::config::load_client_configuration;
    use orangu::tui::HeaderStatus;
    use std::io::Write;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// A minimal stub that answers `/v1/models` (and any other path) with a
    /// 200 JSON body naming `model_id`, standing in for a reachable
    /// orangu-server without needing a real one. Its
    /// body never sets `"orangu_coordinator": true`, so it's correctly
    /// treated as a plain server, not a coordinator.
    async fn spawn_reachable_stub(model_id: &str) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let body = format!(r#"{{"object":"list","data":[{{"id":"{model_id}"}}]}}"#);
        tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                let body = body.clone();
                tokio::spawn(async move {
                    let mut buf = [0u8; 4096];
                    let _ = stream.read(&mut buf).await;
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    );
                    let _ = stream.write_all(response.as_bytes()).await;
                    let _ = stream.write_all(body.as_bytes()).await;
                    let _ = stream.shutdown().await;
                });
            }
        });
        format!("http://{addr}/v1")
    }

    /// A port nothing listens on, standing in for an unreachable server —
    /// connections to it are refused immediately, no real timeout needed.
    async fn unreachable_endpoint() -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        format!("http://{addr}/v1")
    }

    fn status(server_ok: bool, model_ok: bool) -> HeaderStatus {
        HeaderStatus {
            workspace_ok: true,
            server_ok: orangu::tui::ConnStatus::from_bool(server_ok),
            model_ok: orangu::tui::ConnStatus::from_bool(model_ok),
            is_coordinator: false,
        }
    }

    #[test]
    fn model_ids_deduplicates_entries_shared_by_data_and_models() {
        // A server whose `/v1/models` response fills in both the OpenAI-style
        // `data` array and a `models` array with the same entries would
        // otherwise list every model twice — once per field.
        let response = super::ModelsResponse {
            data: vec![super::ModelEntry {
                id: "gemma".to_string(),
                model: String::new(),
                name: String::new(),
            }],
            models: vec![super::ModelEntry {
                id: "gemma".to_string(),
                model: String::new(),
                name: String::new(),
            }],
        };
        assert_eq!(response.model_ids(), vec!["gemma".to_string()]);
    }

    #[test]
    fn idle_switch_targets_first_model_when_pinned_model_unserved() {
        let available = vec!["a".to_string(), "b".to_string()];
        assert_eq!(
            super::idle_model_switch_target(status(true, false), &available),
            Some("a")
        );
    }

    #[test]
    fn idle_switch_skips_when_model_still_served() {
        let available = vec!["a".to_string()];
        assert_eq!(
            super::idle_model_switch_target(status(true, true), &available),
            None
        );
    }

    #[test]
    fn idle_switch_skips_when_server_down() {
        // Server down: no advertised models to switch to, so leave the banner
        // showing the pinned model with its red indicator.
        assert_eq!(
            super::idle_model_switch_target(status(false, false), &[]),
            None
        );
    }

    #[test]
    fn idle_switch_skips_when_server_up_but_advertises_nothing() {
        assert_eq!(
            super::idle_model_switch_target(status(true, false), &[]),
            None
        );
    }

    #[test]
    fn models_request_attaches_optional_bearer_token() {
        let client = reqwest::Client::new();

        let with_key = super::models_request(&client, "http://localhost:8100/v1", Some("secret"))
            .build()
            .expect("build request");
        assert_eq!(with_key.url().as_str(), "http://localhost:8100/v1/models");
        assert_eq!(
            with_key
                .headers()
                .get(reqwest::header::AUTHORIZATION)
                .and_then(|value| value.to_str().ok()),
            Some("Bearer secret")
        );

        let without_key = super::models_request(&client, "http://localhost:8100/v1", None)
            .build()
            .expect("build request");
        assert!(
            without_key
                .headers()
                .get(reqwest::header::AUTHORIZATION)
                .is_none()
        );
    }

    #[test]
    fn review_prompt_profile_uses_the_dedicated_review_server_untouched() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu]\nserver = main\n\n[main]\nendpoint = http://localhost:9000/v1\nmodel = all\nrole = all\n\n[reviewer]\nendpoint = http://localhost:9000/v1\nmodel = review\nrole = review\n"
        )
        .unwrap();
        let config = load_client_configuration(file.path()).unwrap();

        // Even though the interactive chat is "connected" elsewhere, a
        // dedicated `role = review` server must be used with its own
        // endpoint/model, not whatever the live chat connection happens to
        // be pinned to right now.
        let profile = super::review_prompt_profile(
            &config,
            "main",
            "some-other-model-picked-via-slash-model",
            "http://localhost:9999/v1",
            false,
        )
        .expect("review server resolves");
        assert_eq!(profile.endpoint, "http://localhost:9000/v1");
        assert_eq!(profile.model, "review");
    }

    #[test]
    fn review_prompt_profile_falls_back_to_the_active_connection_when_no_review_role_exists() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu]\nserver = main\n\n[main]\nendpoint = http://localhost:9000/v1\nmodel = all\nrole = all\n"
        )
        .unwrap();
        let config = load_client_configuration(file.path()).unwrap();

        // No dedicated review server — reuse the live connection's own
        // endpoint/model exactly as chat requests do, since those may have
        // moved via /server or /model since the config was loaded.
        let profile = super::review_prompt_profile(
            &config,
            "main",
            "picked-via-slash-model",
            "http://localhost:9999/v1",
            false,
        )
        .expect("falls back to the active server");
        assert_eq!(profile.endpoint, "http://localhost:9999/v1");
        assert_eq!(profile.model, "picked-via-slash-model");
    }

    #[test]
    fn review_prompt_profile_under_a_coordinator_ignores_local_role_sections() {
        // Once a coordinator is confirmed, it alone owns model/role
        // decisions: even a dedicated `role = review` section (with its own
        // distinct endpoint/model) must be ignored in favor of the live
        // active connection with `.model` forced to the role name "review",
        // matching the "coordinator controls everything" principle.
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu]\nserver = main\n\n[main]\nendpoint = http://localhost:9000/v1\nmodel = all\nrole = all\n\n[reviewer]\nendpoint = http://localhost:9111/v1\nmodel = review\nrole = review\n"
        )
        .unwrap();
        let config = load_client_configuration(file.path()).unwrap();

        let profile = super::review_prompt_profile(
            &config,
            "main",
            "whatever-was-picked-via-slash-model",
            "http://localhost:9000/v1",
            true,
        )
        .expect("active connection resolves");
        assert_eq!(profile.endpoint, "http://localhost:9000/v1");
        assert_eq!(profile.model, "review");
    }

    #[test]
    fn startup_server_candidates_tries_default_first_then_the_rest_in_order() {
        let server_names = vec!["alt1".to_string(), "alt2".to_string(), "main".to_string()];
        assert_eq!(
            super::startup_server_candidates("main", &server_names),
            vec!["main", "alt1", "alt2"]
        );
    }

    #[test]
    fn startup_server_candidates_skips_the_default_if_listed_elsewhere_too() {
        // The default shouldn't be tried twice even if server_names lists it
        // again in its natural (non-default-first) position.
        let server_names = vec!["main".to_string(), "alt1".to_string()];
        assert_eq!(
            super::startup_server_candidates("main", &server_names),
            vec!["main", "alt1"]
        );
    }

    #[tokio::test]
    async fn resolve_startup_server_fails_over_to_a_reachable_alternate() {
        let unreachable = unreachable_endpoint().await;
        let reachable = spawn_reachable_stub("org/alt-model").await;
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu]\nserver = main\n\n[main]\nendpoint = {unreachable}\nmodel = org/main-model\n\n[alt]\nendpoint = {reachable}\nmodel = org/alt-model\n"
        )
        .unwrap();
        let config = load_client_configuration(file.path()).unwrap();
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(500))
            .build()
            .unwrap();
        let server_names = vec!["alt".to_string(), "main".to_string()];

        let resolution = super::resolve_startup_server(&http_client, &config, &server_names).await;
        assert_eq!(resolution.server, "alt");
        assert!(resolution.switched);
        assert_eq!(
            resolution.available_models,
            vec!["org/alt-model".to_string()]
        );
    }

    #[tokio::test]
    async fn resolve_startup_server_stays_on_default_when_nothing_responds() {
        let unreachable_main = unreachable_endpoint().await;
        let unreachable_alt = unreachable_endpoint().await;
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu]\nserver = main\n\n[main]\nendpoint = {unreachable_main}\nmodel = org/main-model\n\n[alt]\nendpoint = {unreachable_alt}\nmodel = org/alt-model\n"
        )
        .unwrap();
        let config = load_client_configuration(file.path()).unwrap();
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(500))
            .build()
            .unwrap();
        let server_names = vec!["alt".to_string(), "main".to_string()];

        let resolution = super::resolve_startup_server(&http_client, &config, &server_names).await;
        assert_eq!(resolution.server, "main");
        assert!(!resolution.switched);
        assert!(resolution.available_models.is_empty());
    }

    #[tokio::test]
    async fn resolve_startup_server_keeps_the_default_when_it_already_responds() {
        let reachable = spawn_reachable_stub("org/main-model").await;
        let unreachable = unreachable_endpoint().await;
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu]\nserver = main\n\n[main]\nendpoint = {reachable}\nmodel = org/main-model\n\n[alt]\nendpoint = {unreachable}\nmodel = org/alt-model\n"
        )
        .unwrap();
        let config = load_client_configuration(file.path()).unwrap();
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(500))
            .build()
            .unwrap();
        let server_names = vec!["alt".to_string(), "main".to_string()];

        let resolution = super::resolve_startup_server(&http_client, &config, &server_names).await;
        assert_eq!(resolution.server, "main");
        assert!(!resolution.switched);
    }

    #[tokio::test]
    async fn resolve_startup_state_switches_the_model_to_one_the_resolved_server_advertises() {
        // The default server responds, but doesn't advertise the model id
        // its own section configures (e.g. it swapped models since the
        // config was written) — resolve_startup_state should pick up the
        // advertised model instead, without a second /v1/models round trip.
        let reachable = spawn_reachable_stub("org/actually-loaded").await;
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu]\nserver = main\n\n[main]\nendpoint = {reachable}\nmodel = org/configured-but-stale\n"
        )
        .unwrap();
        let config = load_client_configuration(file.path()).unwrap();
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(500))
            .build()
            .unwrap();

        let resolution =
            super::resolve_startup_state(http_client, config, vec!["main".to_string()]).await;
        assert_eq!(resolution.server, "main");
        assert!(!resolution.switched_server);
        assert_eq!(resolution.active_model_id, "org/actually-loaded");
        assert_eq!(
            resolution.model_switched,
            Some((
                "org/configured-but-stale".to_string(),
                "org/actually-loaded".to_string()
            ))
        );
    }
}
