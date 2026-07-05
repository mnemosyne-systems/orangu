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
    /// might use (plain OpenAI vs. some llama.cpp/Ollama-style servers); most
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
/// optional bearer token. OpenAI-compatible servers — including a llama.cpp
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

    if let Some(endpoint) = endpoint
        && let Ok(response) = models_request(http_client, endpoint, profile.api_key.as_deref())
            .send()
            .await
        && response.status().is_success()
    {
        server_ok = true;
        if let Ok(models) = response.json::<ModelsResponse>().await {
            available_models = models.model_ids();
            model_ok = available_models.iter().any(|id| id == active_model_id);
        }
    }

    (
        orangu::tui::HeaderStatus {
            workspace_ok,
            server_ok,
            model_ok,
        },
        available_models,
    )
}

/// Decide whether an idle refresh should switch the pinned model. When the
/// server is up and advertising models but no longer serves the one we are
/// pinned to (e.g. a llama.cpp server swapped the loaded model while we sat
/// idle), return the model id to switch to so the header banner can reflect the
/// change; otherwise `None`. Reuses the model list the header probe already
/// fetched, so no extra request is made.
pub(crate) fn idle_model_switch_target(
    status: orangu::tui::HeaderStatus,
    available_models: &[String],
) -> Option<&str> {
    if status.server_ok && !status.model_ok {
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

#[cfg(test)]
mod tests {
    use orangu::tui::HeaderStatus;

    fn status(server_ok: bool, model_ok: bool) -> HeaderStatus {
        HeaderStatus {
            workspace_ok: true,
            server_ok,
            model_ok,
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
}
