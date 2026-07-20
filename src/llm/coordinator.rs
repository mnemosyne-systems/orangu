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

//! Detecting whether an endpoint is an orangu-coordinator proxy rather than a
//! plain orangu-server — shared by both the `orangu`
//! binary (the header status probe, startup embeddings detection, `/server`
//! re-detection) and library code that needs it without access to any of
//! that (the explorer subagent, which reloads its own config from disk).

use crate::llm::normalized_openai_endpoint;
use serde_json::Value;

/// `GET /v1/coordinator`: `Some(models)` when the endpoint confirms itself as
/// an orangu-coordinator proxy (`"orangu_coordinator": true`), carrying every
/// distinct model each conventional role (`all`/`code`/`review`/`explorer`/
/// `embeddings`) currently resolves to, deduplicated; `None` for anything
/// else — unreachable, a non-success status, an unexpected body, or a plain
/// orangu-server, neither of which exposes this path.
pub async fn probe_coordinator(
    http_client: &reqwest::Client,
    endpoint: &str,
    api_key: Option<&str>,
) -> Option<Vec<String>> {
    let url = format!("{}/v1/coordinator", normalized_openai_endpoint(endpoint));
    let mut request = http_client.get(url);
    if let Some(key) = api_key {
        request = request.bearer_auth(key);
    }
    let response = request.send().await.ok()?;
    if !response.status().is_success() {
        return None;
    }
    let body: Value = response.json().await.ok()?;
    parse_coordinator_body(&body)
}

/// Parses a `GET /v1/coordinator` response body: `Some(models)` — every
/// distinct model named in its `models` map, deduplicated — when
/// `orangu_coordinator` is `true`, `None` otherwise. Split out of
/// [`probe_coordinator`] so the parsing itself is testable without a live
/// server.
fn parse_coordinator_body(body: &Value) -> Option<Vec<String>> {
    if !body
        .get("orangu_coordinator")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return None;
    }

    let mut models: Vec<String> = body
        .get("models")
        .and_then(Value::as_object)
        .map(|roles| {
            roles
                .values()
                .filter_map(|model| model.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    models.sort_unstable();
    models.dedup();
    Some(models)
}

#[cfg(test)]
mod tests {
    #[test]
    fn parse_coordinator_body_reads_and_dedupes_models() {
        let body = serde_json::json!({
            "orangu_coordinator": true,
            "version": "0.11.0",
            "models": {
                "all": "org/gemma",
                "code": "org/gemma",
                "explorer": "org/qwen",
            },
        });
        let mut models = super::parse_coordinator_body(&body).expect("is a coordinator");
        models.sort_unstable();
        assert_eq!(
            models,
            vec!["org/gemma".to_string(), "org/qwen".to_string()]
        );
    }

    #[test]
    fn parse_coordinator_body_rejects_non_coordinator_responses() {
        assert_eq!(super::parse_coordinator_body(&serde_json::json!({})), None);
        assert_eq!(
            super::parse_coordinator_body(&serde_json::json!({"status": "ok"})),
            None
        );
        assert_eq!(
            super::parse_coordinator_body(&serde_json::json!({"orangu_coordinator": false})),
            None
        );
    }
}
