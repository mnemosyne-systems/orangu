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

//! Router assembly and shared state. Endpoint handlers live in
//! `http::openai` (OpenAI-compatible) and `http::native`
//! (llama.cpp-native); shutdown is handled here since it's neither.

pub mod native;
pub mod openai;

use crate::engine::backend::Backend;
use crate::engine::generate::Engine;
use axum::{
    Router,
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use std::{net::SocketAddr, path::PathBuf, sync::Arc, time::Instant};
use tokio::sync::mpsc;

impl orangu::files_http::WorkspaceState for AppState {
    fn workspace(&self) -> &std::path::Path {
        &self.workspace
    }
}

pub struct AppState {
    pub engine: Arc<Engine>,
    /// The bearer token every request must carry, or `None` for an open
    /// server — see [`require_api_key`].
    pub api_key: Option<String>,
    /// What `general.name`/the resolved model spec reports as the model's
    /// "id" in `/v1/models` and `/props` — not necessarily a real file path,
    /// so a client can display it directly.
    pub model_label: String,
    /// Backend and device this model is running on, exactly as the startup
    /// banner prints it (e.g. `Vulkan/AMD Radeon RX 5500M (RADV NAVI14)`).
    /// Reported by `/props` so a benchmark can record *what* it measured
    /// alongside the numbers — a throughput figure with no device attached
    /// to it cannot be compared against anything later.
    pub backend_label: String,
    /// Which GPU kernels and tuning constants this device actually came up
    /// with — `VulkanBackend::tuning_report`, or `None` for a backend that
    /// has no such selection to make (CPU/CUDA/OpenCL/ROCm). Reported by
    /// `/props` for the same reason `backend_label` is, one level deeper:
    /// the label says *which device*, this says *which of its kernels*, and
    /// on a GPU whose defaults were swept on different hardware that is the
    /// difference between a comparable number and an anecdote.
    pub gpu_tuning: Option<serde_json::Value>,
    /// The `wgpu` engine, when this backend is one — so `/gpu-timings` can
    /// drain `VulkanBackend::take_timings`. `None` for CPU/CUDA/OpenCL/ROCm,
    /// which have no GPU timestamp queries to report.
    pub wgpu_backend: Option<Arc<dyn Backend>>,
    /// The root directory this server operates in (`-w`/`--workspace`, or
    /// the current working directory). Reported by `/props` so a client can
    /// see which tree it is talking to.
    pub workspace: PathBuf,
    pub started_at: Instant,
    pub shutdown_tx: mpsc::Sender<()>,
}

/// Paths that stay reachable without a key.
///
/// Only the probes. `/health` says the process is up and nothing else — it
/// names no model, reports no load, and returns the same bytes to everyone —
/// so requiring a secret for it buys nothing and costs the thing every
/// deployment needs: a probe that works before credentials are distributed,
/// from a load balancer that has none.
///
/// `/ready` is here for the same reason and is a deliberate widening of it,
/// because it does disclose something `/health` does not: how loaded this
/// server is. That is the one fact the probe exists to report, it is bounded
/// (queue depth and slot counts, no model name and no request content), and
/// anyone able to send a request learns the same thing from the `503` they
/// would get instead. A readiness probe that needed a credential would fail
/// closed at exactly the moment a balancer most needs an answer.
///
/// Everything else is closed, `/v1/models` included. It is tempting to leave
/// that open because the coordinator probes it, and the coordinator was fixed
/// instead: an HTTP `401` proves a process is answering just as well as a
/// `200`, and a probe that needed a secret to establish liveness would have
/// been the wrong shape.
const OPEN_PATHS: &[&str] = &["/health", "/ready"];

/// Establishes who is asking, and whether they may.
///
/// No key configured means no check, which is the behaviour before this
/// existed and the right default for the loopback address the server also
/// defaults to. The check matters exactly when `host` is widened, and the
/// pairing is deliberate: nothing about binding to a network should silently
/// also mean publishing an inference engine. `Authorization: Bearer <key>`
/// because that is what the orangu client already sends (`orangu::llm`'s
/// `bearer_auth`) and what every OpenAI-shaped client sends.
async fn require_api_key(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let Some(expected) = state.api_key.as_deref() else {
        return next.run(request).await;
    };
    if OPEN_PATHS.contains(&request.uri().path()) {
        return next.run(request).await;
    }
    let presented = request
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .unwrap_or("");

    if !constant_time_eq(presented.as_bytes(), expected.as_bytes()) {
        return unauthorized();
    }
    next.run(request).await
}

/// Compares two byte strings without leaking where they first differ.
///
/// A `==` on secrets returns as soon as it finds a mismatch, so the time it
/// takes reports how many leading bytes were right — enough, over many
/// attempts, to recover a key one byte at a time.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn unauthorized() -> axum::response::Response {
    (
        axum::http::StatusCode::UNAUTHORIZED,
        [(axum::http::header::WWW_AUTHENTICATE, "Bearer")],
        "unauthorized: this server requires an Authorization: Bearer <key> header\n",
    )
        .into_response()
}

/// What a refused request is told, on every endpoint.
///
/// One string so the three transports cannot describe the same condition
/// differently — a caller reading an SSE `error` and a caller reading a `503`
/// body are looking at the same server state.
pub const OVERLOADED_MESSAGE: &str =
    "server busy: the request queue is full ([orangu-server].queue_limit). Retry shortly.";

/// The HTTP answer to a full queue.
///
/// `503` with `Retry-After`, not `500`: this request could be served later,
/// and the distinction is the whole point of bounding the queue rather than
/// letting it grow. A client that sees `500` files a bug; one that sees `503`
/// backs off.
pub fn overloaded_response() -> axum::response::Response {
    use axum::response::IntoResponse;
    (
        axum::http::StatusCode::SERVICE_UNAVAILABLE,
        [(axum::http::header::RETRY_AFTER, "1")],
        OVERLOADED_MESSAGE,
    )
        .into_response()
}

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(native::health))
        .route("/ready", get(native::ready))
        .route("/props", get(native::props))
        .route("/gpu-timings", get(native::gpu_timings))
        .route("/moe-stats", get(native::moe_stats))
        .route("/model-cache", get(native::model_cache))
        .route("/model-cache/drop", post(native::drop_model_cache))
        .route("/slots", get(native::slots))
        .route("/slots/{id_slot}", post(native::slot_action))
        .route("/metrics", get(native::metrics))
        .route("/tokenize", post(native::tokenize))
        .route("/detokenize", post(native::detokenize))
        .route("/completion", post(native::completion))
        .route("/embedding", post(native::embedding))
        .route("/apply-template", post(native::apply_template))
        .route("/v1/models", get(openai::list_models))
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/completions", post(openai::completions))
        .route("/v1/embeddings", post(openai::embeddings))
        .route("/v1/shutdown", post(shutdown))
        // The file-lifecycle API, mounted from the shared router
        // `orangu-coordinator` mounts too, so both front doors serve the
        // same eight endpoints over the same implementation.
        .merge(orangu::files_http::router::<AppState>())
        .route_layer(axum::middleware::from_fn_with_state(
            state.clone(),
            require_api_key,
        ))
        .with_state(state)
}

/// Loopback-only, like `orangu-coordinator`'s own shutdown endpoint — a
/// server bound to a non-loopback `host` must not let an arbitrary network
/// peer kill it with an unauthenticated POST.
async fn shutdown(
    axum::extract::ConnectInfo(addr): axum::extract::ConnectInfo<SocketAddr>,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    if !addr.ip().is_loopback() {
        return (
            StatusCode::FORBIDDEN,
            "shutdown is only available from localhost\n",
        );
    }
    let _ = state.shutdown_tx.send(()).await;
    (StatusCode::OK, "shutting down\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The probes stay reachable without a key, and nothing else does.
    ///
    /// `/v1/models` is deliberately *not* on the list even though the
    /// coordinator probes it: the coordinator was changed to treat any HTTP
    /// answer as proof of life, because a probe that needs a secret to
    /// establish liveness is the wrong shape.
    ///
    /// The list has grown once, by `/ready`, and the reasoning grew with it —
    /// see [`OPEN_PATHS`]. Anything added here after that has to justify not
    /// only that a probe needs it but what it discloses, which is why this
    /// asserts the exact list rather than a subset.
    #[test]
    fn only_the_probes_are_reachable_without_a_key() {
        assert_eq!(OPEN_PATHS, &["/health", "/ready"]);
        for closed in [
            "/v1/models",
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/embeddings",
            "/v1/shutdown",
            "/metrics",
            "/slots",
            "/props",
            "/completion",
        ] {
            assert!(!OPEN_PATHS.contains(&closed), "{closed} must require a key");
        }
    }
}
