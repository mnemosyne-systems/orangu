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
//! that model's `orangu-server` is the active process, then forward the
//! request through unchanged and stream the response back.

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
/// from a a plain OpenAI-compatible server, neither of which
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
/// never reaches any backend `orangu-server` itself.
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
/// blindly forwarded to (or from) the upstream `orangu-server`.
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

/// One attempt at forwarding the request to `target`, with every header
/// that isn't hop-by-hop carried over unchanged.
///
/// Split out of [`proxy`] so the same attempt can be made twice — see its
/// retry path — from one description of what "forward it" means, rather
/// than two copies that could drift apart in which headers they pass on.
async fn send_upstream(
    coordinator: &Coordinator,
    method: &Method,
    target: &str,
    headers: &HeaderMap,
    body: Bytes,
) -> Result<reqwest::Response, reqwest::Error> {
    let mut request = coordinator
        .http_client()
        .request(method.clone(), target)
        .body(body);
    for (name, value) in headers.iter() {
        if is_hop_by_hop(name) {
            continue;
        }
        request = request.header(name, value);
    }
    request.send().await
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
        .await;

    let origin = match coordinator.ensure_active(&entry).await {
        Ok(origin) => origin,
        Err(err) => {
            let default_entry = coordinator.default_entry();
            if entry.name != default_entry.name {
                eprintln!(
                    "warning: failed to start '{}': {err:#}; falling back to '{}'",
                    entry.name, default_entry.name
                );
                match coordinator.ensure_active(&default_entry).await {
                    Ok(origin) => origin,
                    Err(fallback_err) => {
                        return (
                            StatusCode::BAD_GATEWAY,
                            format!("orangu-coordinator: failed to start both '{}' and fallback '{}': {fallback_err:#}", entry.name, default_entry.name),
                        ).into_response();
                    }
                }
            } else {
                return (
                    StatusCode::BAD_GATEWAY,
                    format!(
                        "orangu-coordinator: failed to start default profile '{}': {err:#}",
                        entry.name
                    ),
                )
                    .into_response();
            }
        }
    };

    let path_and_query = uri.path_and_query().map(|pq| pq.as_str()).unwrap_or("/");
    let target = format!("{origin}{path_and_query}");

    let upstream = match send_upstream(&coordinator, &method, &target, &headers, body.clone()).await
    {
        Ok(response) => response,
        // The child was alive when `ensure_active` checked and gone by the
        // time this request reached it — the window a profile's
        // `orangu-server` exiting on its own (a lost GPU device, an OOM
        // kill, an operator's `kill`) always leaves. Nothing has been
        // written back to the client yet, so the request is still whole and
        // safe to send again: bring the profile back up and retry it
        // exactly once. A second failure is reported rather than retried —
        // at that point the profile is not coming back on its own, and
        // retrying forever would just hold the caller open.
        //
        // `ensure_reachable`, not `ensure_active`: the child is dead but
        // has not necessarily been *reported* dead yet this soon after the
        // connection failed, and asking the question that can lag is how
        // the retry ends up in the same closed port. See its doc comment.
        Err(first) => {
            let Ok(origin) = coordinator.ensure_reachable(&entry).await else {
                return (
                    StatusCode::BAD_GATEWAY,
                    format!("orangu-coordinator: failed to reach {target}: {first}"),
                )
                    .into_response();
            };
            let target = format!("{origin}{path_and_query}");
            match send_upstream(&coordinator, &method, &target, &headers, body).await {
                Ok(response) => response,
                Err(retry) => {
                    return (
                        StatusCode::BAD_GATEWAY,
                        format!(
                            "orangu-coordinator: failed to reach {target}: {retry} \
                             (after restarting '{}', which had stopped: {first})",
                            entry.name
                        ),
                    )
                        .into_response();
                }
            }
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
    // Only the two Unix-only retry tests below drive a socket by hand.
    #[cfg(unix)]
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// A stand-in `orangu-server`: answers every request `200 OK` — except
    /// the `nth_to_drop`-th request to `/v1/chat/completions` and the
    /// `nth_probe_to_drop`-th to `/v1/models`, whose connections it closes
    /// without a response, which is exactly what a child that has just
    /// exited (a lost GPU device, an OOM kill, a stray `kill`) looks like
    /// from the coordinator's side. Pass `0` for either to never drop it.
    ///
    /// Handles requests one at a time, reading each one's `Content-Length`
    /// body in full — a proxied POST that got a response before its body
    /// was read would report as its own kind of connection error and prove
    /// nothing about the retry.
    #[cfg(unix)]
    async fn flaky_upstream(
        listener: tokio::net::TcpListener,
        nth_to_drop: usize,
        nth_probe_to_drop: usize,
    ) {
        let mut chat_requests = 0usize;
        let mut probes = 0usize;
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            loop {
                let mut request = Vec::new();
                let mut byte = [0u8; 1];
                // Read exactly the request head; the body (if any) follows.
                while !request.ends_with(b"\r\n\r\n") {
                    match stream.read(&mut byte).await {
                        Ok(0) | Err(_) => return,
                        Ok(_) => request.push(byte[0]),
                    }
                }
                let head = String::from_utf8_lossy(&request).to_string();
                let content_length = head
                    .lines()
                    .find_map(|line| {
                        line.strip_prefix("content-length: ")
                            .or_else(|| line.strip_prefix("Content-Length: "))
                    })
                    .and_then(|v| v.trim().parse::<usize>().ok())
                    .unwrap_or(0);
                let mut body = vec![0u8; content_length];
                if content_length > 0 && stream.read_exact(&mut body).await.is_err() {
                    return;
                }

                if head.starts_with("POST /v1/chat/completions") {
                    chat_requests += 1;
                    if chat_requests == nth_to_drop {
                        // Gone: no response at all, connection closed.
                        drop(stream);
                        break;
                    }
                }
                if head.starts_with("GET /v1/models") {
                    probes += 1;
                    if probes == nth_probe_to_drop {
                        drop(stream);
                        break;
                    }
                }
                let payload = b"{\"ok\":true}";
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
                    payload.len()
                );
                if stream.write_all(response.as_bytes()).await.is_err()
                    || stream.write_all(payload).await.is_err()
                {
                    return;
                }
            }
        }
    }

    /// A request that reaches a profile's `orangu-server` just as it goes
    /// away is sent again once, not reported as a failure.
    ///
    /// This is the window `ensure_active`'s liveness check cannot close:
    /// the child is alive when it is checked and gone microseconds later,
    /// which is precisely what `orangu-server` exiting on a lost GPU device
    /// does under a request. Nothing has been written back to the caller at
    /// that point, so the request is still whole — the fix is to restart and
    /// resend it, and what the caller sees is a slow answer rather than a
    /// `502`.
    ///
    /// Unix-only: the stand-in server it spawns is a shell script, which
    /// Windows cannot execute. What is under test is the coordinator's retry
    /// logic, which is platform-independent.
    #[tokio::test]
    #[cfg(unix)]
    async fn a_request_that_loses_its_server_is_retried_once() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        // The first chat request is dropped; every `/v1/models` probe is
        // answered, so the profile is still reachable and must NOT be
        // restarted — the retry alone is the whole recovery.
        tokio::spawn(flaky_upstream(listener, 1, 0));

        let mut file = tempfile::NamedTempFile::new().unwrap();
        use std::io::Write;
        writeln!(
            file,
            "[orangu-coordinator]\nmodels = /srv/models\nstartup_timeout = 10\n\n\
             [main]\nrole = all\nmodel = org/gemma\nhost = 127.0.0.1\nport = {port}\n"
        )
        .unwrap();
        let config = crate::config::load_coordinator_configuration(file.path()).unwrap();

        // A "server" that just stays alive: the fake upstream above is what
        // actually answers, so the spawned child only has to not exit.
        let script = crate::process::fake_server_script("sleep 30\n");

        let coordinator = Arc::new(Coordinator::new(config, true, Some(script.clone())).unwrap());
        let response = proxy(
            State(coordinator.clone()),
            Method::POST,
            "/v1/chat/completions".parse::<Uri>().unwrap(),
            HeaderMap::new(),
            Bytes::from_static(br#"{"model":"org/gemma"}"#),
        )
        .await;

        assert_eq!(
            response.status(),
            StatusCode::OK,
            "the dropped request should have been retried, not reported as a 502"
        );

        coordinator.shutdown().await;
        std::fs::remove_file(&script).ok();
    }

    /// The same retry, when the profile really is gone: the request fails,
    /// the reachability probe that follows fails too, so the profile is
    /// restarted before the request is sent again.
    ///
    /// This is the case `try_wait` alone gets wrong. A child killed
    /// milliseconds ago is not reported as exited yet, so
    /// `ensure_active` — which trusts that answer — hands back the origin of
    /// a process that is already gone, and the retry lands in the same
    /// closed port. Observed against a real profile before
    /// `ensure_reachable` existed; this test is that observation, made
    /// cheap: the second `/v1/models` probe (the reachability check) is
    /// refused, exactly as a dead child would refuse it.
    ///
    /// Unix-only for the same reason as the test above.
    #[tokio::test]
    #[cfg(unix)]
    async fn a_profile_that_stopped_answering_is_restarted_before_the_retry() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        // Probe 1 is the initial startup health check (answered); probe 2 is
        // the reachability check after the dropped chat request (refused,
        // so a restart follows); probe 3 is the restarted profile's own
        // health check (answered).
        tokio::spawn(flaky_upstream(listener, 1, 2));

        let mut file = tempfile::NamedTempFile::new().unwrap();
        use std::io::Write;
        writeln!(
            file,
            "[orangu-coordinator]\nmodels = /srv/models\nstartup_timeout = 10\n\n\
             [main]\nrole = all\nmodel = org/gemma\nhost = 127.0.0.1\nport = {port}\n"
        )
        .unwrap();
        let config = crate::config::load_coordinator_configuration(file.path()).unwrap();

        let script = crate::process::fake_server_script("sleep 30\n");

        let coordinator = Arc::new(Coordinator::new(config, true, Some(script.clone())).unwrap());
        let response = proxy(
            State(coordinator.clone()),
            Method::POST,
            "/v1/chat/completions".parse::<Uri>().unwrap(),
            HeaderMap::new(),
            Bytes::from_static(br#"{"model":"org/gemma"}"#),
        )
        .await;

        assert_eq!(
            response.status(),
            StatusCode::OK,
            "an unreachable profile must be restarted, then the request re-sent"
        );

        coordinator.shutdown().await;
        std::fs::remove_file(&script).ok();
    }

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
