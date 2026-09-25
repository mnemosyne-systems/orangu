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

pub mod images;
pub mod native;
pub mod openai;

use crate::engine::backend::Backend;
use crate::engine::generate::Engine;
use axum::{
    Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
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
    /// The served file's `general.architecture`, as the banner prints it.
    /// Not always `engine.model.config().architecture`: on a `qwen_image`
    /// server the engine's model is the text encoder (`qwen2vl`), and this
    /// is what says so.
    pub architecture: String,
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
    /// Build, host, process and in-flight-request metrics for `/metrics`.
    pub process_metrics: Arc<crate::engine::metrics::ProcessMetrics>,
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

/// The HTTP answer to a request this server cannot serve **as sent** — today,
/// one whose context is longer than the device has room for.
///
/// `400`, not `500`: nothing is broken, and the difference decides what the
/// caller does next. A client that sees `500` retries the same request or
/// files a bug; one that sees `400` shortens the prompt — which is the only
/// thing that can work, and which `/auto_review` does automatically by falling
/// back to a file's diff alone.
pub fn too_long_response(message: String) -> axum::response::Response {
    use axum::response::IntoResponse;
    (axum::http::StatusCode::BAD_REQUEST, message).into_response()
}

/// Refuses a request whose *prompt* is longer than this device can hold, before
/// any of it is processed — `None` when it fits, or when there is no device
/// ceiling to apply.
///
/// The prompt alone, not the prompt plus `max_tokens`. `max_tokens` is a cap
/// on the answer, not a reservation, and the engine already clamps it to the
/// model's own context length; the device ceiling clamps it the same way (see
/// `engine::generate`). Counting it here refused every request from a client
/// that asks for a generous cap without knowing the card — the web console
/// asks for 32768 on every turn, and on a device whose room is 16384 tokens
/// that made the console unusable on any prompt at all.
///
/// **A `400`, not a `500`.** Nothing is broken and retrying changes nothing:
/// the only thing that can work is a shorter prompt, and the status code is
/// what tells a client to try that rather than file a bug. `/auto_review` does
/// exactly that, dropping a file's whole-file context and reviewing its diff.
///
/// Asked here rather than only inside the engine because *here* is where an
/// answer is still free. Past this point the prompt is on its way to a device
/// that answers an oversized request by stalling until its driver resets it,
/// taking every other request on the process with it.
pub fn reject_oversized_context(prompt_tokens: usize) -> Option<Response> {
    let ceiling = crate::engine::generate::kv_device_token_ceiling()?;
    if crate::engine::generate::prompt_fits_device(prompt_tokens, ceiling) {
        return None;
    }
    // `CONTEXT_TOO_LONG_MARKER` is what a client matches on to tell this
    // apart from every other `400` — see its own doc comment.
    Some(too_long_response(format!(
        "prompt ({prompt_tokens} tokens) plus one generated token needs {} {}, more than \
         the {ceiling} this server has room for on its device — send a shorter prompt, or \
         serve this model on a device with more memory\n",
        prompt_tokens.saturating_add(1),
        orangu::llm::CONTEXT_TOO_LONG_MARKER
    )))
}

pub fn build_router(state: Arc<AppState>) -> Router {
    let metrics = state.process_metrics.clone();
    // An image server takes a picture to start from as base64 in the JSON,
    // which axum's default 2 MB body cap would refuse long before
    // `images::MAX_IMAGE_BYTES`. Everything else keeps that default.
    let body_limit = axum::extract::DefaultBodyLimit::max(if state.engine.image.is_some() {
        images::IMAGE_BODY_LIMIT
    } else {
        2 * 1024 * 1024
    });
    let router = Router::new()
        .route("/health", get(native::health))
        .route("/ready", get(native::ready))
        .route("/props", get(native::props).post(native::set_props))
        .route("/gpu-timings", get(native::gpu_timings))
        .route("/moe-stats", get(native::moe_stats))
        .route("/decode-stages", get(native::decode_stages))
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
        .route("/v1/images/generations", post(images::generations))
        .route("/v1/shutdown", post(shutdown))
        // The file-lifecycle API, mounted from the shared router
        // `orangu-coordinator` mounts too, so both front doors serve the
        // same eight endpoints over the same implementation.
        .merge(orangu::files_http::router::<AppState>())
        .route_layer(axum::middleware::from_fn_with_state(
            state.clone(),
            require_api_key,
        ))
        .layer(body_limit)
        .with_state(state);
    count_requests(router, metrics)
}

/// Adds request counting to a listener's router, for `orangu_server_http_requests`.
/// The API and the web console get it; the dedicated metrics listener does not,
/// so scraping it never shows up in the figures.
pub fn count_requests(
    router: Router,
    metrics: Arc<crate::engine::metrics::ProcessMetrics>,
) -> Router {
    router.layer(axum::middleware::from_fn_with_state(
        metrics,
        count_in_flight,
    ))
}

/// Holds a request in flight until its response *body* has been sent or
/// dropped, not just until the headers are ready: a streamed completion is
/// in flight for as long as it is generating. Also tallies the status, for
/// requests the key check turns away as much as for the rest.
async fn count_in_flight(
    State(metrics): State<Arc<crate::engine::metrics::ProcessMetrics>>,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    let guard = metrics.track_request();
    let response = next.run(request).await;
    metrics.record_response(response.status().as_u16());
    let (parts, body) = response.into_parts();
    Response::from_parts(
        parts,
        axum::body::Body::new(TrackedBody {
            inner: body,
            _guard: guard,
        }),
    )
}

struct TrackedBody {
    inner: axum::body::Body,
    _guard: crate::engine::metrics::InFlightGuard,
}

impl http_body::Body for TrackedBody {
    type Data = <axum::body::Body as http_body::Body>::Data;
    type Error = <axum::body::Body as http_body::Body>::Error;

    fn poll_frame(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Result<http_body::Frame<Self::Data>, Self::Error>>> {
        std::pin::Pin::new(&mut self.inner).poll_frame(cx)
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.inner.size_hint()
    }
}

/// The dedicated metrics listener's router (see `main.rs`) — `/metrics` and
/// a static landing page at `/`, with no [`require_api_key`] layer. Keep it
/// that way: don't merge this into [`build_router`] or add other routes to it.
pub fn build_metrics_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(native::metrics_index))
        .route("/metrics", get(native::metrics))
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

    /// Requests through a counted router show up in the figures, whichever
    /// status they get, and none is left in flight once it has been answered.
    #[tokio::test]
    async fn counted_routers_tally_requests_by_status() {
        let metrics = Arc::new(crate::engine::metrics::ProcessMetrics::new());
        let router = count_requests(
            Router::new().route("/ok", get(|| async { "hi" })),
            metrics.clone(),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, router).await;
        });

        for path in ["ok", "ok", "missing"] {
            let response = reqwest::get(format!("http://{addr}/{path}")).await.unwrap();
            response.bytes().await.unwrap();
        }

        // The guard drops as the server finishes with the body, which can be a
        // moment after the client has finished reading it.
        for _ in 0..50 {
            if metrics.in_flight() == 0 {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        let body = metrics.render(std::time::Duration::ZERO);
        for expected in [
            "orangu_server_http_requests{stat=\"in_flight\"} 0
",
            "orangu_server_http_requests{stat=\"total\"} 3
",
            "orangu_server_http_requests{stat=\"client_errors\"} 1
",
            "orangu_server_http_requests{stat=\"server_errors\"} 0
",
        ] {
            assert!(
                body.contains(expected),
                "{expected}
{body}"
            );
        }
    }

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
