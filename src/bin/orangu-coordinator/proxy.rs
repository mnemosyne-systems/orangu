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

//! The HTTP proxy handler: inspect the request's `model` field, make sure
//! that model's llama.cpp is the active process, then forward the request
//! through unchanged and stream the response back.

use crate::process::Coordinator;
use axum::{
    Json,
    body::{Body, Bytes},
    extract::State,
    http::{HeaderMap, HeaderName, Method, StatusCode, Uri},
    response::{IntoResponse, Response},
};
use serde_json::{Value, json};
use std::sync::Arc;

/// `GET /v1/coordinator` — a fixed, side-effect-free identity marker orangu
/// (or any other client) can probe to tell an orangu-coordinator proxy apart
/// from a plain llama.cpp or OpenAI-compatible server, neither of which
/// exposes this path. Unlike every other request, it is answered directly
/// and never proxied: it must work even when no profile is active yet.
///
/// `models` reports the model each conventional role
/// (`all`/`code`/`review`/`explorer`/`embeddings`) currently resolves to, so
/// a caller can see what `model` to send for a given role without needing
/// its own copy of `orangu-coordinator.conf` — a role with no profile of its
/// own falls back to the `all`-role default's model, same as routing does.
pub async fn coordinator_info(State(coordinator): State<Arc<Coordinator>>) -> Json<Value> {
    let models: serde_json::Map<String, Value> = coordinator
        .models_by_role()
        .into_iter()
        .map(|(role, model)| (role.to_string(), Value::String(model.to_string())))
        .collect();
    Json(json!({
        "orangu_coordinator": true,
        "version": crate::VERSION,
        "models": models,
    }))
}

/// `POST /v1/coordinator/activate` — a pre-warming hint a caller can send
/// *before* the request that actually needs a model, naming a `model` (a
/// real model id or a role name, matched exactly like ordinary routing) to
/// start swapping to right away. Answered directly, never proxied — this
/// never reaches any backend llama.cpp itself.
///
/// The swap is spawned detached and NOT awaited here: this must return
/// immediately so the swap survives the caller disconnecting early or not
/// waiting for a response at all (that's the whole point of a hint sent
/// ahead of the real request), and keeps this endpoint from ever blocking
/// on a slow cold load itself. A caller that does want to fail loudly should
/// just send its real request instead, which does wait.
///
/// Unlike ordinary routing, an unmatched `model` is reported as an error
/// (`404`) rather than silently falling back to `all` or "currently
/// active": those fallbacks exist so a request that must be answered
/// somehow always is, but an explicit "activate X" call has no such
/// obligation, and silently activating the wrong thing would be worse than
/// saying so.
pub async fn activate(State(coordinator): State<Arc<Coordinator>>, body: Bytes) -> Response {
    let Some(hint) = extract_model_field(&body) else {
        return (
            StatusCode::BAD_REQUEST,
            "orangu-coordinator: request body must be a JSON object with a \"model\" field naming a model id or role to activate",
        )
            .into_response();
    };
    let Some(entry) = coordinator.match_hint(&hint) else {
        return (
            StatusCode::NOT_FOUND,
            format!("orangu-coordinator: no profile matches model or role '{hint}'"),
        )
            .into_response();
    };

    let name = entry.name.clone();
    let background_coordinator = coordinator.clone();
    tokio::spawn(async move {
        let _ = background_coordinator.ensure_active(&entry).await;
    });

    (StatusCode::ACCEPTED, Json(json!({ "activating": name }))).into_response()
}

/// Headers that are specific to one hop of the connection and must not be
/// blindly forwarded to (or from) the upstream llama.cpp server.
const HOP_BY_HOP: &[&str] = &[
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
];

fn is_hop_by_hop(name: &HeaderName) -> bool {
    HOP_BY_HOP.iter().any(|hop| name.as_str() == *hop)
}

/// Reads the JSON body's top-level `model` field, if the body is a JSON
/// object and that field is a string. Any other shape (non-JSON body, GET
/// request with no body, missing field) yields `None`, and the caller falls
/// back to the default `all`-role entry.
fn extract_model_field(body: &[u8]) -> Option<String> {
    let value: serde_json::Value = serde_json::from_slice(body).ok()?;
    value.get("model")?.as_str().map(str::to_string)
}

/// The role a request's own path implies, independent of whatever `model`
/// it did or didn't name — currently just `/v1/embeddings`, the one
/// endpoint that names a distinct capability rather than being usable by
/// any chat-capable role. Matched by suffix so a request through a mount
/// point or reverse proxy prefix still resolves the same way.
pub(crate) fn implied_role_for_path(path: &str) -> Option<&'static str> {
    path.ends_with("/v1/embeddings").then_some("embeddings")
}

pub async fn proxy(
    State(coordinator): State<Arc<Coordinator>>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let model_hint = extract_model_field(&body);
    let implied_role = implied_role_for_path(uri.path());
    let entry = coordinator
        .resolve_entry(model_hint.as_deref(), implied_role)
        .await
        .clone();

    let origin = match coordinator.ensure_active(&entry).await {
        Ok(origin) => origin,
        Err(err) => {
            return (
                StatusCode::BAD_GATEWAY,
                format!(
                    "orangu-coordinator: failed to start '{}': {err:#}",
                    entry.name
                ),
            )
                .into_response();
        }
    };

    let path_and_query = uri.path_and_query().map(|pq| pq.as_str()).unwrap_or("/");
    let target = format!("{origin}{path_and_query}");

    let mut request = coordinator
        .http_client()
        .request(method, &target)
        .body(body);
    for (name, value) in headers.iter() {
        if is_hop_by_hop(name) {
            continue;
        }
        request = request.header(name, value);
    }
    if let Some(api_key) = &entry.api_key {
        request = request.bearer_auth(api_key);
    }

    let upstream = match request.send().await {
        Ok(response) => response,
        Err(err) => {
            return (
                StatusCode::BAD_GATEWAY,
                format!("orangu-coordinator: failed to reach {target}: {err}"),
            )
                .into_response();
        }
    };

    let status = upstream.status();
    let mut response_headers = HeaderMap::new();
    for (name, value) in upstream.headers().iter() {
        if is_hop_by_hop(name) {
            continue;
        }
        response_headers.insert(name.clone(), value.clone());
    }

    let mut response = Response::new(Body::from_stream(upstream.bytes_stream()));
    *response.status_mut() = status;
    *response.headers_mut() = response_headers;
    response
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_model_field_from_json_body() {
        let body = br#"{"model":"gemma","messages":[]}"#;
        assert_eq!(extract_model_field(body).as_deref(), Some("gemma"));
    }

    #[test]
    fn returns_none_for_missing_or_malformed_body() {
        assert_eq!(extract_model_field(b""), None);
        assert_eq!(extract_model_field(b"not json"), None);
        assert_eq!(extract_model_field(br#"{"messages":[]}"#), None);
    }

    #[test]
    fn identifies_hop_by_hop_headers_case_insensitively() {
        assert!(is_hop_by_hop(&HeaderName::from_static("connection")));
        assert!(is_hop_by_hop(&HeaderName::from_static("content-length")));
        assert!(!is_hop_by_hop(&HeaderName::from_static("content-type")));
        assert!(!is_hop_by_hop(&HeaderName::from_static("authorization")));
    }

    #[test]
    fn implied_role_for_path_recognizes_embeddings_requests() {
        assert_eq!(implied_role_for_path("/v1/embeddings"), Some("embeddings"));
        assert_eq!(
            implied_role_for_path("/some/prefix/v1/embeddings"),
            Some("embeddings")
        );
    }

    #[test]
    fn implied_role_for_path_is_none_for_everything_else() {
        assert_eq!(implied_role_for_path("/v1/chat/completions"), None);
        assert_eq!(implied_role_for_path("/v1/models"), None);
        assert_eq!(implied_role_for_path("/health"), None);
        assert_eq!(implied_role_for_path("/v1/coordinator"), None);
    }
}
