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
use crate::tenant::{Denied, InFlight, TenantMeter, TenantRegistry};
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
    /// The named keys from `[tenant:<name>]`, each with its own limits and its
    /// own live accounting. Empty for every deployment that has not declared
    /// one, in which case nothing here costs anything.
    pub tenants: Arc<TenantRegistry>,
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

/// The paths a tenant's limits apply to: everything that runs the model.
///
/// The rest of the API is metadata — `/props`, `/slots`, `/metrics`,
/// `/v1/models` — and metering it would be worse than leaving it alone in both
/// directions. A monitoring scrape holding a tenant key would spend a
/// generation budget on nothing, and a `requests_per_minute` of 60 would then
/// mean "sixty seconds of Prometheus, and no inference at all". These limits
/// are denominated in the machine's time, so they are charged where the
/// machine's time goes.
///
/// `/tokenize` and `/apply-template` are on the metadata side deliberately:
/// they touch the tokenizer and the template, not the model, and their cost is
/// already bounded by the size of the body the caller sent.
const METERED_PATHS: &[&str] = &[
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
    "/completion",
    "/embedding",
];

/// Establishes who is asking, and whether they may.
///
/// Two jobs, and they are one function because the second is meaningless
/// without the first: a limit needs a subject, and the subject is whatever the
/// presented key resolves to.
///
/// **Authentication.** No key configured and no tenant declared means no
/// check, which is the behaviour before this existed and the right default for
/// the loopback address the server also defaults to. The check matters exactly
/// when `host` is widened, and the pairing is deliberate: nothing about
/// binding to a network should silently also mean publishing an inference
/// engine. `Authorization: Bearer <key>` because that is what the orangu
/// client already sends (`orangu::llm`'s `bearer_auth`) and what every
/// OpenAI-shaped client sends.
///
/// **Metering.** A request that authenticated *as a tenant* takes a place
/// against that tenant's limits before the handler runs, and holds it until
/// the response body is finished — see [`hold_until_finished`], which is where
/// the interesting part is. `[orangu-server].api_key` is deliberately not a
/// tenant: it is the operator's own key, it predates this, and a deployment
/// that adds tenants around it should not discover that its own key acquired
/// limits it never set.
async fn require_api_key(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    mut request: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    if state.api_key.is_none() && state.tenants.is_empty() {
        return next.run(request).await;
    }
    if OPEN_PATHS.contains(&request.uri().path()) {
        return next.run(request).await;
    }
    let presented = request
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .unwrap_or("")
        .to_string();

    let is_operator = state.api_key.as_deref().is_some_and(|expected| {
        crate::tenant::constant_time_eq(presented.as_bytes(), expected.as_bytes())
    });
    let meter = if is_operator {
        None
    } else {
        match state.tenants.resolve(&presented) {
            Some(meter) => Some(meter),
            None => return unauthorized(),
        }
    };

    let Some(meter) = meter else {
        return next.run(request).await;
    };
    // Available to the handlers, which is how a finished generation knows
    // whose token budget to charge (`engine::generate::GenerateRequest`).
    request.extensions_mut().insert(meter.clone());

    if !METERED_PATHS.contains(&request.uri().path()) {
        return next.run(request).await;
    }
    let guard = match meter.admit() {
        Ok(guard) => guard,
        Err(denied) => return denied_response(&denied),
    };
    hold_until_finished(next.run(request).await, guard)
}

fn unauthorized() -> axum::response::Response {
    (
        axum::http::StatusCode::UNAUTHORIZED,
        [(axum::http::header::WWW_AUTHENTICATE, "Bearer")],
        "unauthorized: this server requires an Authorization: Bearer <key> header\n",
    )
        .into_response()
}

/// What a metered refusal looks like on the wire.
///
/// `429`, not the `503` a full queue gets: they are different conditions and a
/// client should treat them differently. `503` means the *server* is saturated
/// and any client would see it; `429` means this caller is over its own bound
/// while the server may be idle. Backing off is right for both; reporting them
/// as the same outage is not.
/// Which of the three bounds was hit is carried in a header as well as in the
/// prose, because they call for different responses and a client should not
/// have to parse a sentence to tell them apart: concurrency clears as soon as
/// the caller's own requests finish, a rate window only clears with time.
fn denied_response(denied: &Denied) -> axum::response::Response {
    (
        axum::http::StatusCode::TOO_MANY_REQUESTS,
        [
            (
                axum::http::header::RETRY_AFTER,
                denied.retry_after.to_string(),
            ),
            (RATE_LIMIT_HEADER, denied.limit.label().to_string()),
        ],
        format!("{}\n", denied.message),
    )
        .into_response()
}

/// Names the bound a `429` came from: `concurrency`, `requests` or `tokens`.
const RATE_LIMIT_HEADER: axum::http::HeaderName =
    axum::http::HeaderName::from_static("x-orangu-rate-limit");

/// Keeps a tenant's place until the response has actually been delivered.
///
/// **This is the part that would be wrong if it were simpler.** Dropping the
/// guard when the handler returns looks equivalent and is not: a streaming
/// generation returns its response as soon as the SSE stream *exists*, and
/// then generates for however long the answer takes. A guard released there
/// would count every streamed request as instantaneous — so a
/// `max_concurrent` of 1 would admit a thousand of them, and the one shape of
/// request the limit exists for would be the one shape it did not bound.
///
/// So the guard rides on the body and is dropped when the body is dropped:
/// when the last frame is sent, or when the client disconnects and the
/// connection tears its body down. Both are exactly when the work stops.
///
/// The wrapper forwards `size_hint` and `is_end_stream` rather than
/// re-streaming the body, so an ordinary JSON response keeps its
/// `Content-Length` and its framing. Re-wrapping it as a stream of unknown
/// length would have quietly moved every response on the server to chunked
/// transfer-encoding to add a `Drop`.
fn hold_until_finished(
    response: axum::response::Response,
    guard: InFlight,
) -> axum::response::Response {
    response.map(|body| axum::body::Body::new(Guarded { body, guard }))
}

/// Whose token budget this request's work belongs to.
///
/// An extractor rather than an `Option<Extension<…>>` at five call sites,
/// because the absence is the common case and has to read as ordinary: an open
/// server, the operator's own key, and the web console all generate with
/// nobody to charge, and none of those is a missing extension to be handled.
pub struct Charge(pub Option<Arc<TenantMeter>>);

impl<S: Send + Sync> axum::extract::FromRequestParts<S> for Charge {
    type Rejection = std::convert::Infallible;

    async fn from_request_parts(
        parts: &mut axum::http::request::Parts,
        _state: &S,
    ) -> Result<Self, Self::Rejection> {
        Ok(Self(parts.extensions.get::<Arc<TenantMeter>>().cloned()))
    }
}

/// A response body carrying one tenant's in-flight place.
struct Guarded {
    body: axum::body::Body,
    #[allow(dead_code, reason = "held for its Drop, which releases the place")]
    guard: InFlight,
}

impl http_body::Body for Guarded {
    type Data = axum::body::Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Result<http_body::Frame<Self::Data>, Self::Error>>> {
        // `axum::body::Body` is `Unpin` (it is a boxed, already-pinned body),
        // so the projection needs no `unsafe` and no `pin_project`.
        std::pin::Pin::new(&mut self.get_mut().body).poll_frame(cx)
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.body.size_hint()
    }

    fn is_end_stream(&self) -> bool {
        self.body.is_end_stream()
    }
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

    /// Every metered path is a real route, and every route that runs the
    /// model is metered.
    ///
    /// The second half is the one that bites: a generating endpoint added
    /// later and left off this list is unlimited, silently, and nothing about
    /// the server would say so. Written against the router's own list rather
    /// than a copy of it, so it fails when the routes change and not when
    /// somebody remembers to update a fixture.
    #[test]
    fn every_endpoint_that_runs_the_model_is_metered() {
        // The generating and embedding endpoints, from `build_router`.
        let model_paths = [
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/embeddings",
            "/completion",
            "/embedding",
        ];
        for path in model_paths {
            assert!(METERED_PATHS.contains(&path), "{path} must be metered");
        }
        for path in METERED_PATHS {
            assert!(model_paths.contains(path), "{path} is not a model path");
        }
        // Metadata must not be: a scrape holding a tenant key would otherwise
        // spend that tenant's generation budget on nothing.
        for path in ["/metrics", "/props", "/slots", "/v1/models", "/tokenize"] {
            assert!(!METERED_PATHS.contains(&path), "{path} must not be metered");
        }
    }

    /// A metered path is a closed path. Metering an endpoint reachable
    /// without a key would be a limit with no subject.
    #[test]
    fn nothing_metered_is_reachable_without_a_key() {
        for path in METERED_PATHS {
            assert!(
                !OPEN_PATHS.contains(path),
                "{path} is both open and metered"
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
