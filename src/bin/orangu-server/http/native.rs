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
//! not a byte-for-byte schema match; see `doc/SERVER_ROADMAP.md`.

use axum::{Json, extract::State, http::StatusCode, response::IntoResponse};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::AppState;
use crate::engine::chat_template::{ChatMessage, ChatTemplate};
use crate::engine::generate::{FinishReason, GenerateRequest, StreamEvent};
use crate::engine::sampling::SamplingParams;

pub async fn health() -> impl IntoResponse {
    Json(serde_json::json!({"status": "ok"}))
}

pub async fn props(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let cfg = state.engine.model.config();
    Json(serde_json::json!({
        "model": state.model_label,
        "architecture": cfg.architecture,
        "n_ctx": cfg.n_ctx_train,
        "n_vocab": state.engine.tokenizer.vocab_size(),
        "n_embd": cfg.n_embd,
        "total_slots": state.engine.slots.total(),
        "chat_template": state.engine.chat_template_source,
        "uptime_seconds": state.started_at.elapsed().as_secs(),
    }))
}

pub async fn slots(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(state.engine.slots.snapshot())
}

pub async fn metrics(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let snapshot = state.engine.slots.snapshot();
    let busy = snapshot.iter().filter(|s| s.busy).count();
    let body = format!(
        "# HELP orangu_server_slots_total Configured concurrent request slots.\n\
         # TYPE orangu_server_slots_total gauge\n\
         orangu_server_slots_total {}\n\
         # HELP orangu_server_slots_busy Slots currently generating.\n\
         # TYPE orangu_server_slots_busy gauge\n\
         orangu_server_slots_busy {busy}\n",
        state.engine.slots.total(),
    );
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
    let tokens = state.engine.tokenizer.encode(&req.prompt, true);
    let sampling = sampling_from(&req, state.engine.role);
    let stop_token_ids = state.engine.tokenizer.eos_token.into_iter().collect();
    let mut rx = state
        .engine
        .generate(GenerateRequest {
            prompt_tokens: tokens,
            sampling,
            max_tokens: req.n_predict,
            stop_token_ids,
        })
        .await;

    if !req.stream {
        let mut content = String::new();
        while let Some(event) = rx.recv().await {
            match event {
                StreamEvent::Token(text) => content.push_str(&text),
                StreamEvent::Done { .. } => break,
                StreamEvent::Error(err) => {
                    return (StatusCode::INTERNAL_SERVER_ERROR, err).into_response();
                }
            }
        }
        let content = state
            .engine
            .tokenizer
            .clean_up_tokenization_spaces(&content);
        return Json(serde_json::json!({"content": content, "stop": true})).into_response();
    }

    let stream = async_stream::stream! {
        while let Some(event) = rx.recv().await {
            match event {
                StreamEvent::Token(text) => {
                    yield Ok::<_, std::convert::Infallible>(
                        axum::response::sse::Event::default()
                            .data(serde_json::json!({"content": text, "stop": false}).to_string()),
                    );
                }
                StreamEvent::Done { finish_reason, .. } => {
                    yield Ok(axum::response::sse::Event::default().data(
                        serde_json::json!({
                            "content": "",
                            "stop": true,
                            "finish_reason": finish_reason_str(finish_reason),
                        })
                        .to_string(),
                    ));
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
        Ok(embedding) => Json(EmbeddingResponse { embedding }).into_response(),
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
    match template.render(
        &req.messages,
        true,
        "",
        "",
        state.engine.role.enable_thinking(),
    ) {
        Ok(mut prompt) => {
            // Mirror `openai::chat_completions`'s own reasoning-suppression
            // prefill, so this endpoint's whole point — showing exactly
            // what will be sent to the model — stays accurate for `Role::
            // Review`.
            if state.engine.role.suppresses_reasoning() {
                prompt.push_str(super::openai::EMPTY_THINK_BLOCK);
            }
            Json(ApplyTemplateResponse { prompt }).into_response()
        }
        Err(err) => (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response(),
    }
}
