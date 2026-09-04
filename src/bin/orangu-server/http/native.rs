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

//! llama.cpp-native endpoints: `/health`, `/props`, `/slots`, `/metrics`,
//! `/completion`, `/tokenize`, `/detokenize`, `/embedding`,
//! `/apply-template`. Response shapes approximate llama.cpp's own —
//! close enough for `orangu`'s `/information` probe and `curl` inspection,
//! not a byte-for-byte schema match.

use axum::{
    Json,
    body::Bytes,
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::AppState;
use crate::engine::chat_template::{ChatMessage, ChatTemplate};
use crate::engine::generate::{FinishReason, GenerateRequest, StreamEvent};
use crate::engine::sampling::SamplingParams;

pub async fn health() -> impl IntoResponse {
    Json(serde_json::json!({"status": "ok"}))
}

/// Whether this server should be sent traffic *right now*.
///
/// Distinct from `/health`, and the distinction is the point. `/health` asks
/// "is this process alive" — the answer a supervisor uses to decide whether to
/// restart it, and one that must stay `200` while the server is merely busy,
/// because restarting a loaded server under load is the worst possible
/// response to load. `/ready` asks "would a request sent now be served" — the
/// answer a load balancer uses to decide where to route, where "busy" is
/// exactly the case worth reporting.
///
/// Two things make it `503`:
///
/// - **The admission queue is full.** A new unpinned request would be refused
///   with `503` anyway (see `[orangu-server].queue_limit`), so saying so
///   up front lets a balancer send it somewhere that can take it instead of
///   spending a round trip to find out. Only meaningful with a `queue_limit`
///   set; an unbounded queue never refuses and so is never unready for this
///   reason.
/// - **The GPU device was lost.** This process is on its way out
///   (`crate::device_lost`) and every request from here on is answered with
///   one sentence about a driver reset. It has not stopped being *alive* —
///   `/health` keeps saying so, and a supervisor restarting it is precisely
///   the intended recovery — but nothing should be routed to it meanwhile.
///
/// The body names which, because a probe that only flips a status code leaves
/// an operator with the alert and none of the reason.
pub async fn ready(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let queued = state.engine.slots.queued();
    let limit = state.engine.slots.queue_limit();
    let (status, reason) = readiness(crate::device_lost::is_lost(), queued, limit);
    (
        status,
        Json(serde_json::json!({
            "status": reason,
            "queue_depth": queued,
            "queue_limit": limit,
            "slots_busy": state.engine.slots.busy_count(),
            "slots_total": state.engine.slots.total(),
        })),
    )
}

/// The readiness decision, apart from the state it reads.
///
/// Separated so the rule can be tested at all: the alternative is standing up
/// an `AppState` — a loaded model and a backend — to assert five lines of
/// comparison, which is why rules like this usually go untested.
fn readiness(device_lost: bool, queued: usize, limit: usize) -> (StatusCode, &'static str) {
    if device_lost {
        // Checked first: a lost device makes every answer wrong, including a
        // cheerful one about an empty queue.
        (StatusCode::SERVICE_UNAVAILABLE, "device lost")
    } else if limit > 0 && queued >= limit {
        (StatusCode::SERVICE_UNAVAILABLE, "queue full")
    } else {
        (StatusCode::OK, "ok")
    }
}

pub async fn props(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let cfg = state.engine.model.config();
    Json(serde_json::json!({
        "model": state.model_label,
        "backend": state.backend_label,
        // Which build answered. `version` dates the release; `commit` is the
        // only field that tells two builds of one version apart, which during
        // performance work is every pair of builds that matters — see
        // `orangu::build_info`. A benchmark archives both (`orangu-bench`'s
        // bundle) so a stored result says what produced it without anyone
        // having remembered to write it down.
        "version": orangu::build_info::VERSION,
        "commit": orangu::build_info::COMMIT,
        // `null` on a backend with no kernel selection to report — see
        // `AppState::gpu_tuning`.
        "gpu": state.gpu_tuning,
        "architecture": cfg.architecture,
        "n_ctx": cfg.n_ctx_train,
        "n_vocab": state.engine.tokenizer.vocab_size(),
        "n_embd": cfg.n_embd,
        "total_slots": state.engine.slots.total(),
        "chat_template": state.engine.chat_template_source,
        "workspace": state.workspace.display().to_string(),
        "uptime_seconds": state.started_at.elapsed().as_secs(),
        // Identifies *which* server process answered. Paired with
        // `uptime_seconds` this is what lets a benchmark prove it is talking to
        // the build it just launched rather than one left over from a previous
        // run — see `orangu-bench`'s `report_environment`.
        "pid": std::process::id(),
    }))
}

/// `GET /gpu-timings` — the accumulated GPU timestamp breakdown since the last
/// call, and **reset**.
///
/// Read-and-reset so a client can bracket a window: read once before the
/// workload to discard whatever the warmup left, run it, read again to get
/// exactly that window. See `VulkanBackend::take_timings`.
///
/// This is the answer to "where did the time go" on a platform with no
/// `perf` — which is every macOS machine, and so every machine the Metal
/// backend is interesting on. `steps: 0` means no timestamped decode step
/// happened in the window: either `ORANGU_GPU_TIMESTAMPS=1` is not set, or
/// this adapter has no timestamp query. Reported as zero steps rather than
/// zero milliseconds so "not measured" cannot be read as "took no time".
///
/// `unavailable` extends that rule one step further out. A **split** model has
/// no `wgpu` backend to ask at all — `Backend::as_wgpu` answers `None` on the
/// multi-device wrapper by design, because a timestamp query set belongs to one
/// device and a split resolves none of them — so this endpoint used to answer
/// `enabled: false` and nothing else. That is indistinguishable from "you
/// forgot to set `ORANGU_GPU_TIMESTAMPS`", and a client that reports nothing
/// when it gets nothing (as `orangu-bench` did) leaves a split run looking like
/// a run whose GPU stages cost zero. Naming the reason is what stops that: the
/// caller can say *why* there is no breakdown, and point at the profiler that
/// does still work.
pub async fn gpu_timings(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let timings = state
        .wgpu_backend
        .as_deref()
        .and_then(orangu_backend_as_wgpu)
        .map(|v| v.take_timings().to_json());
    // Read off the tuning report rather than carried as a second flag: that
    // object *is* the placement plan on a split (see `SplitReport::to_json`),
    // so there is one source for "was this run split" and it cannot drift.
    let split = state
        .gpu_tuning
        .as_ref()
        .and_then(|gpu| gpu.get("split"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let unavailable = match (timings.is_some(), split) {
        (true, _) => None,
        (false, true) => Some("split"),
        (false, false) => Some("no_wgpu_backend"),
    };
    Json(serde_json::json!({
        "enabled": timings.is_some(),
        "timings": timings,
        "unavailable": unavailable,
    }))
}

/// `GET /moe-stats` — the accumulated mixture-of-experts counters since the
/// last call, and **reset**.
///
/// Same drain-on-read contract as [`gpu_timings`], for the same reason: read
/// once before the workload to discard the warmup, run it, read again to get
/// exactly that window.
///
/// This is what makes work on models too large for RAM scoreable. Throughput
/// cannot separate a change that moves fewer bytes from one that moves the
/// same bytes faster, and on a model whose weights stream from disk it cannot
/// see either one over the I/O. `stats.union_ratio` is the redundancy in
/// today's per-token expert loop, and `process.major_faults_window` is the
/// only honest signal that the weights came from the disk rather than the
/// page cache — see `engine::moe_stats` and `BIG.md`.
///
/// `stats.layer_calls: 0` means no MoE layer ran in the window — a dense
/// model, or an empty window. Reported as zero *calls* rather than zero bytes
/// so "not measured" cannot be read as "moved nothing", exactly as
/// [`gpu_timings`] reports zero steps. `process` is `null` off Linux, where
/// there is no `/proc/self` to read it from.
pub async fn moe_stats() -> impl IntoResponse {
    let stats = crate::engine::moe_stats::take();
    let process = crate::engine::moe_stats::take_process_memory();
    let store = crate::engine::expert_store::global().take_stats();
    let route_ahead = crate::engine::route_ahead::take();
    Json(serde_json::json!({
        "stats": stats.to_json(),
        "process": process.map(|p| p.to_json()),
        // Where the expert weights were when they were asked for. `hit_rate`
        // is `null` unless `ORANGU_EXPERT_RESIDENCY=1` asked the kernel — see
        // `engine::expert_store`, which reports "not measured" rather than
        // "not resident" when it did not look.
        "store": store.to_json(),
        // How well the next layer's routing could be guessed one layer early.
        // `accuracy` is `null` unless `ORANGU_ROUTE_AHEAD=1` — see
        // `engine::route_ahead`, which measures this before anything is built
        // on it.
        "route_ahead": route_ahead.to_json(),
        // Layers released behind the sweep, or null when no dense residency
        // window is configured. Null and zero are different claims: one says
        // the policy is off, the other that it ran and freed nothing.
        "dense_residency": crate::engine::dense_residency::stats().map(|(layers, bytes)| {
            serde_json::json!({"released_layers": layers, "released_bytes": bytes})
        }),
    }))
}

/// `GET /decode-stages` — where a forward pass's *elapsed* time went, since
/// the last call, and **reset**.
///
/// Same drain-on-read contract as [`gpu_timings`] and [`moe_stats`], for the
/// same reason: read once before the workload to discard the warmup, run it,
/// read again to get exactly that window.
///
/// This answers the question a CPU profile cannot. A sampling profile counts
/// cycles, so a stage that runs alone on one core and a stage that runs the
/// same arithmetic across sixteen look alike in it — while the first costs
/// sixteen times the wall clock. `stages[].ms` is time on the thread that ran
/// the stage, so the breakdown is in the same units as the token latency it
/// explains.
///
/// `enabled: false` means `ORANGU_DECODE_STAGES=1` was not set, and
/// `passes: 0` that no forward pass finished in the window — reported as zero
/// *passes* rather than zero milliseconds, so "not measured" cannot be read as
/// "took no time", exactly as [`gpu_timings`] reports zero steps. See
/// `engine::decode_stages` for which stages are measured for every
/// architecture and which are per-architecture.
pub async fn decode_stages() -> impl IntoResponse {
    Json(crate::engine::decode_stages::take_json())
}

/// `GET /model-cache` — how much of the model's weights are in RAM right now.
///
/// Not drain-on-read: this is a state, not a window. A benchmark records it
/// beside its numbers so a later reader can tell a cold run from a warm one
/// instead of assuming — on a model larger than memory those are different
/// experiments and their rates are not comparable.
///
/// `resident_bytes` is `null` where the platform cannot measure it, never
/// zero: "nothing is cached" and "this machine cannot say" would otherwise be
/// the same answer, and only one of them means the run was cold.
pub async fn model_cache() -> impl IntoResponse {
    let shards = crate::engine::page_cache::residency();
    let (bytes, resident) = crate::engine::page_cache::residency_totals(&shards);
    Json(serde_json::json!({
        "model_bytes": bytes,
        "resident_bytes": resident,
        "shards": shards.iter().map(crate::engine::page_cache::ShardResidency::to_json).collect::<Vec<_>>(),
    }))
}

/// `POST /model-cache/drop` — evict the model's weights from the page cache,
/// so the next request reads them from the disk.
///
/// Loopback-only, like `/v1/shutdown`: it makes the server dramatically slower
/// on purpose, which is not something an arbitrary network peer should be able
/// to do to it.
///
/// The response reports residency **before and after** rather than a success
/// flag, because a partial drop is the realistic failure and it reads exactly
/// like a successful one from the outside — see `engine::page_cache` for why
/// dropping takes two different calls in a specific order.
pub async fn drop_model_cache(
    axum::extract::ConnectInfo(addr): axum::extract::ConnectInfo<std::net::SocketAddr>,
) -> impl IntoResponse {
    if !addr.ip().is_loopback() {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({
                "error": "dropping the model cache is only available from localhost",
            })),
        );
    }
    (
        StatusCode::OK,
        Json(crate::engine::page_cache::drop_model_page_cache().to_json()),
    )
}

/// `&dyn Backend` → the `wgpu` engine behind it, if it is one.
///
/// A free function rather than an inline closure only because
/// `Backend::as_wgpu` returns a borrow tied to the trait object, and naming
/// the lifetime relationship is clearer here than at the call site.
fn orangu_backend_as_wgpu(
    backend: &dyn crate::engine::backend::Backend,
) -> Option<&crate::engine::backend::VulkanBackend> {
    backend.as_wgpu()
}

pub async fn slots(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(state.engine.slots.snapshot())
}

/// `?action=save|restore` on the slot-action endpoint.
#[derive(Deserialize)]
pub struct SlotActionQuery {
    action: String,
}

/// `POST /slots/{id_slot}?action=save|restore` with a JSON body naming the
/// file: persists a slot's KV cache to, or restores it from,
/// `~/.orangu/server/<fingerprint>/slots/<filename>`. This is orangu-server's
/// equivalent of llama.cpp's `--slot-save-path` endpoints, and speaks the
/// same request shape the orangu client (`orangu::llm::SlotRegistry`) already
/// sends.
///
/// When durable slot persistence is disabled (`ORANGU_NO_SLOT_SAVE` set, or no
/// resolvable home directory), this reports "not supported" exactly as a
/// llama.cpp server started without `--slot-save-path` does — the orangu
/// client already degrades gracefully against that, falling back to a full
/// reprefill. A missing or model-incompatible saved file is *not* an error:
/// `restore` succeeds with `n_restored: 0` so a stale sidecar never trips the
/// client's "persistence unavailable" notice.
pub async fn slot_action(
    State(state): State<Arc<AppState>>,
    Path(id_slot): Path<usize>,
    Query(query): Query<SlotActionQuery>,
    body: Bytes,
) -> impl IntoResponse {
    let Some(store) = state.engine.slot_store.as_deref() else {
        return (
            StatusCode::NOT_IMPLEMENTED,
            "This server does not support slots action (disabled by ORANGU_NO_SLOT_SAVE)\n"
                .to_string(),
        )
            .into_response();
    };

    if id_slot >= state.engine.slots.total() {
        return (
            StatusCode::BAD_REQUEST,
            format!(
                "id_slot {id_slot} out of range (server has {} slots)\n",
                state.engine.slots.total()
            ),
        )
            .into_response();
    }

    let filename = match serde_json::from_slice::<SlotActionBody>(&body) {
        Ok(b) if !b.filename.is_empty() => b.filename,
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                "request body must be a JSON object with a non-empty \"filename\" field\n"
                    .to_string(),
            )
                .into_response();
        }
    };

    match query.action.as_str() {
        "save" => match store.save(id_slot, &filename) {
            Ok(n_saved) => Json(serde_json::json!({
                "id_slot": id_slot,
                "filename": filename,
                "n_saved": n_saved,
            }))
            .into_response(),
            Err(err) => (StatusCode::BAD_REQUEST, format!("{err}\n")).into_response(),
        },
        "restore" => match store.restore(id_slot, &filename) {
            Ok(n_restored) => Json(serde_json::json!({
                "id_slot": id_slot,
                "filename": filename,
                "n_restored": n_restored,
            }))
            .into_response(),
            Err(err) => (StatusCode::BAD_REQUEST, format!("{err}\n")).into_response(),
        },
        other => (
            StatusCode::BAD_REQUEST,
            format!("unknown slot action {other:?} (expected \"save\" or \"restore\")\n"),
        )
            .into_response(),
    }
}

#[derive(Deserialize)]
struct SlotActionBody {
    #[serde(default)]
    filename: String,
}

pub async fn metrics(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let snapshot = state.engine.slots.snapshot();
    let busy = snapshot.iter().filter(|s| s.busy).count();
    let mut body = format!(
        "# HELP orangu_server_slots_total Configured concurrent request slots.\n\
         # TYPE orangu_server_slots_total gauge\n\
         orangu_server_slots_total {}\n\
         # HELP orangu_server_slots_busy Slots currently generating.\n\
         # TYPE orangu_server_slots_busy gauge\n\
         orangu_server_slots_busy {busy}\n\
         # HELP orangu_server_queue_depth Requests waiting for a slot.\n\
         # TYPE orangu_server_queue_depth gauge\n\
         orangu_server_queue_depth {}\n\
         # HELP orangu_server_queue_limit Waiting requests allowed before refusing; 0 is unbounded.\n\
         # TYPE orangu_server_queue_limit gauge\n\
         orangu_server_queue_limit {}\n",
        state.engine.slots.total(),
        state.engine.slots.queued(),
        state.engine.slots.queue_limit(),
    );
    body.push_str(&state.engine.metrics.render());
    (
        StatusCode::OK,
        [("Content-Type", "text/plain; version=0.0.4")],
        body,
    )
}

#[derive(Deserialize)]
pub struct TokenizeRequest {
    content: String,
    #[serde(default)]
    add_special: bool,
}

#[derive(Serialize)]
pub struct TokenizeResponse {
    tokens: Vec<u32>,
}

pub async fn tokenize(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TokenizeRequest>,
) -> impl IntoResponse {
    let tokens = state.engine.tokenizer.encode(&req.content, req.add_special);
    Json(TokenizeResponse { tokens })
}

#[derive(Deserialize)]
pub struct DetokenizeRequest {
    tokens: Vec<u32>,
}

#[derive(Serialize)]
pub struct DetokenizeResponse {
    content: String,
}

pub async fn detokenize(
    State(state): State<Arc<AppState>>,
    Json(req): Json<DetokenizeRequest>,
) -> impl IntoResponse {
    let tokenizer = &state.engine.tokenizer;
    let content = tokenizer.clean_up_tokenization_spaces(&tokenizer.decode(&req.tokens));
    Json(DetokenizeResponse { content })
}

#[derive(Deserialize)]
pub struct CompletionRequest {
    prompt: String,
    #[serde(default = "default_n_predict")]
    n_predict: usize,
    /// See `openai::ChatCompletionRequest::cache_prompt` — llama.cpp's field
    /// name and default (`true`) on its own native endpoint too.
    #[serde(default = "super::openai::default_cache_prompt")]
    cache_prompt: bool,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    repeat_penalty: Option<f32>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    stream: bool,
    /// See `openai::ChatCompletionRequest::id_slot`.
    #[serde(default)]
    id_slot: Option<usize>,
}

fn default_n_predict() -> usize {
    256
}

pub async fn completion(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> axum::response::Response {
    if !state.engine.role.allows_generation() {
        return (
            StatusCode::NOT_IMPLEMENTED,
            format!(
                "this server is running in --{} mode; generation endpoints are disabled",
                state.engine.role.label()
            ),
        )
            .into_response();
    }
    if let Some(rejection) = super::openai::reject_unknown_slot(&state, req.id_slot) {
        return rejection;
    }
    let tokens = state.engine.tokenizer.encode(&req.prompt, true);
    let sampling = sampling_from(&req, state.engine.role);
    let stop_token_ids = state.engine.tokenizer.stop_token_ids();
    let mut rx = state
        .engine
        .generate(GenerateRequest {
            // These endpoints have no structured-output field of their own.
            json_output: false,
            prompt_tokens: tokens,
            sampling,
            max_tokens: req.n_predict,
            stop_token_ids,
            cache_prompt: req.cache_prompt,
            id_slot: req.id_slot,
            timings_per_token: false,
        })
        .await;

    if !req.stream {
        let mut content = String::new();
        let mut timings = serde_json::Value::Null;
        while let Some(event) = rx.recv().await {
            match event {
                StreamEvent::PromptProgress { .. } | StreamEvent::Timings(_) => {}
                // `/completion` is llama.cpp's raw endpoint: one `content`
                // field, no message shape to split reasoning out into, so
                // the thinking stays where it has always been. The chat
                // endpoints are where a caller gets the two apart.
                StreamEvent::Token(text) | StreamEvent::Reasoning(text) => content.push_str(&text),
                StreamEvent::Done { stats, .. } => {
                    timings = super::openai::timings_json(&stats);
                    break;
                }
                StreamEvent::Overloaded => return crate::http::overloaded_response(),
                StreamEvent::Error(err) => {
                    return (StatusCode::INTERNAL_SERVER_ERROR, err).into_response();
                }
            }
        }
        let content = state
            .engine
            .tokenizer
            .clean_up_tokenization_spaces(&content);
        return Json(serde_json::json!({"content": content, "stop": true, "timings": timings}))
            .into_response();
    }

    let stream = async_stream::stream! {
        while let Some(event) = rx.recv().await {
            match event {
                StreamEvent::PromptProgress { .. } | StreamEvent::Timings(_) => {}
                // See the non-streaming arm above: one field, no shape.
                StreamEvent::Token(text) | StreamEvent::Reasoning(text) => {
                    yield Ok::<_, std::convert::Infallible>(
                        axum::response::sse::Event::default()
                            .data(serde_json::json!({"content": text, "stop": false}).to_string()),
                    );
                }
                StreamEvent::Done { finish_reason, stats } => {
                    yield Ok(axum::response::sse::Event::default().data(
                        serde_json::json!({
                            "content": "",
                            "stop": true,
                            "finish_reason": finish_reason_str(finish_reason),
                            "timings": super::openai::timings_json(&stats),
                            "prompt_progress": super::openai::prompt_progress_json(&stats),
                        })
                        .to_string(),
                    ));
                }
                StreamEvent::Overloaded => {
                    yield Ok(axum::response::sse::Event::default()
                        .data(serde_json::json!({"error": crate::http::OVERLOADED_MESSAGE}).to_string()));
                    break;
                }
                StreamEvent::Error(err) => {
                    yield Ok(axum::response::sse::Event::default()
                        .data(serde_json::json!({"error": err}).to_string()));
                }
            }
        }
    };
    axum::response::sse::Sse::new(stream).into_response()
}

fn sampling_from(req: &CompletionRequest, role: crate::config::Role) -> SamplingParams {
    let mut sampling = SamplingParams::default_for_role(role);
    if let Some(v) = req.temperature {
        sampling.temperature = v;
    }
    if let Some(v) = req.top_p {
        sampling.top_p = v;
    }
    if let Some(v) = req.top_k {
        sampling.top_k = v;
    }
    if let Some(v) = req.min_p {
        sampling.min_p = v;
    }
    if let Some(v) = req.repeat_penalty {
        sampling.repeat_penalty = v;
    }
    if let Some(v) = req.seed {
        sampling.seed = v;
    }
    sampling
}

pub fn finish_reason_str(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
    }
}

#[derive(Deserialize)]
pub struct EmbeddingRequest {
    content: String,
}

#[derive(Serialize)]
pub struct EmbeddingResponse {
    embedding: Vec<f32>,
}

pub async fn embedding(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbeddingRequest>,
) -> axum::response::Response {
    match super::openai::pooled_embedding(&state, &req.content).await {
        // llama.cpp's native `/embedding` carries the vector and nothing else,
        // so the token count `PooledEmbedding` also returns is dropped on
        // purpose; `/v1/embeddings` is where it surfaces, as `usage`.
        Ok(pooled) => Json(EmbeddingResponse {
            embedding: pooled.embedding,
        })
        .into_response(),
        Err(err) => (StatusCode::INTERNAL_SERVER_ERROR, err).into_response(),
    }
}

#[derive(Deserialize)]
pub struct ApplyTemplateRequest {
    messages: Vec<ChatMessage>,
}

#[derive(Serialize)]
pub struct ApplyTemplateResponse {
    prompt: String,
}

pub async fn apply_template(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ApplyTemplateRequest>,
) -> axum::response::Response {
    let Some(source) = &state.engine.chat_template_source else {
        return (
            StatusCode::NOT_IMPLEMENTED,
            "model has no tokenizer.chat_template",
        )
            .into_response();
    };
    let template = ChatTemplate::new(source.clone());
    match template.render(&req.messages, true, "", "", state.engine.reasoning()) {
        Ok(mut prompt) => {
            // Mirror `openai::chat_completions`'s own reasoning-suppression
            // prefill, so this endpoint's whole point — showing exactly
            // what will be sent to the model — stays accurate for `Role::
            // Review`.
            super::openai::append_reasoning_suppression(
                &mut prompt,
                state.engine.role,
                &state.engine.tokenizer,
            );
            Json(ApplyTemplateResponse { prompt }).into_response()
        }
        Err(err) => (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An unbounded queue is never "full", so a server without a
    /// `queue_limit` must never report itself unready for depth — it would
    /// take itself out of rotation for a condition it does not have.
    #[test]
    fn an_unbounded_queue_is_always_ready_however_deep_it_gets() {
        for queued in [0, 1, 1000] {
            assert_eq!(readiness(false, queued, 0).0, StatusCode::OK, "{queued}");
        }
    }

    /// Ready right up to the limit and not past it, matching exactly when
    /// `SlotPool::try_acquire` starts refusing. A boundary off by one here
    /// either takes a server out of rotation while it can still serve, or
    /// keeps sending it requests it will answer with `503`.
    #[test]
    fn readiness_flips_at_the_same_depth_the_queue_starts_refusing() {
        assert_eq!(readiness(false, 1, 2).0, StatusCode::OK);
        assert_eq!(readiness(false, 2, 2).0, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(readiness(false, 3, 2).1, "queue full");
    }

    /// A lost device outranks everything: an empty queue on a dead GPU is
    /// still a server nothing should be routed to.
    #[test]
    fn a_lost_device_is_unready_even_with_an_empty_queue() {
        let (status, reason) = readiness(true, 0, 0);
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(reason, "device lost");
    }

    /// Every reason is a distinct string — a dashboard groups by it.
    #[test]
    fn each_reason_names_itself() {
        assert_eq!(readiness(false, 0, 0).1, "ok");
        assert_ne!(readiness(true, 0, 0).1, readiness(false, 5, 5).1);
    }
}
