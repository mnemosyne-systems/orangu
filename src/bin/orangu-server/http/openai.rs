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

//! OpenAI-compatible endpoints: `/v1/models`, `/v1/chat/completions`,
//! `/v1/completions`, `/v1/embeddings`.

use axum::{Json, extract::State, http::StatusCode, response::IntoResponse};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use super::AppState;
use super::native::finish_reason_str;
use crate::engine::chat_template::{ChatMessage, ChatTemplate};
use crate::engine::generate::{GenerateRequest, GenerateStats, StreamEvent};
use crate::engine::loader::PoolingType;
use crate::engine::sampling::SamplingParams;

/// OpenAI's `usage` object.
pub(crate) fn usage_json(stats: &GenerateStats) -> serde_json::Value {
    json!({
        "prompt_tokens": stats.prompt_tokens,
        "completion_tokens": stats.generated_tokens,
        "total_tokens": stats.prompt_tokens + stats.generated_tokens,
        // OpenAI's own field for the part of the prompt served from cache;
        // `prompt_progress` below carries the same number in llama.cpp's shape.
        "prompt_tokens_details": {"cached_tokens": stats.cached_tokens},
    })
}

/// llama.cpp's `timings` object, field for field, so a client (this
/// project's own `llm::openai`, `orangu-bench`, or anything written against
/// llama-server) reads prompt- and decode-rate the same way from either
/// server. Rates come from [`GenerateStats`], which is also what the
/// per-request console log prints — one source of truth for "how fast was
/// that", rather than a wall-clock guess at the far end of an HTTP stream.
pub(crate) fn timings_json(stats: &GenerateStats) -> serde_json::Value {
    let prompt_ms = stats.prompt_time.as_secs_f64() * 1000.0;
    let predicted_ms = stats.generate_time.as_secs_f64() * 1000.0;
    json!({
        "prompt_n": stats.prompt_tokens,
        "prompt_ms": prompt_ms,
        "prompt_per_token_ms": prompt_ms / (stats.prompt_tokens.max(1) as f64),
        "prompt_per_second": stats.prompt_tokens_per_second(),
        "predicted_n": stats.generated_tokens,
        "predicted_ms": predicted_ms,
        "predicted_per_token_ms": predicted_ms / (stats.generated_tokens.max(1) as f64),
        "predicted_per_second": stats.generate_tokens_per_second(),
    })
}

/// llama.cpp's `prompt_progress` object. llama-server emits it repeatedly
/// *during* prefill; this server has no mid-prefill progress event, so it is
/// sent once, with the finished request's totals — enough to tell a cache hit
/// from real prompt processing, which is what it is read for.
pub(crate) fn prompt_progress_json(stats: &GenerateStats) -> serde_json::Value {
    json!({
        "total": stats.prompt_tokens,
        "cache": stats.cached_tokens,
        "processed": stats.prefilled_tokens(),
        "time_ms": stats.prompt_time.as_millis() as i64,
    })
}

pub async fn list_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(json!({
        "object": "list",
        "data": [{
            "id": state.model_label,
            "object": "model",
            "created": unix_now(),
            "owned_by": "orangu-server",
        }]
    }))
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[derive(Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    messages: Vec<ChatMessage>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    seed: Option<u64>,
    /// llama.cpp's field name and default: reuse an already-computed KV cache
    /// for whatever prefix of this prompt one exists for. `false` forces a
    /// full prefill.
    #[serde(default = "default_cache_prompt")]
    cache_prompt: bool,
}

pub(crate) fn default_cache_prompt() -> bool {
    true
}

/// `Role::Review`'s reasoning-suppression approximation: real llama-server
/// (`--reasoning-budget 0`) truncates a reasoning model's thinking phase by
/// pre-filling an *empty, already-closed* thinking block right after the
/// rendered prompt, so generation resumes immediately past it rather than
/// entering one at all — this is the same mechanism, without llama.cpp's
/// own reasoning-parsing machinery behind it. `<think>`/`</think>` is a
/// near-universal convention (DeepSeek-R1, QwQ, Qwen3, GLM, and this
/// project's own real-model testing) but not a guaranteed one — a model
/// using a different tag, or no explicit tag at all, won't be affected by
/// this (the `enable_thinking: false` template kwarg, passed separately,
/// is the other half of this approximation and *does* generalize to any
/// template that checks for it).
pub(crate) const EMPTY_THINK_BLOCK: &str = "<think>\n\n</think>\n\n";

pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
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
    let Some(template_source) = &state.engine.chat_template_source else {
        return (
            StatusCode::NOT_IMPLEMENTED,
            "model has no tokenizer.chat_template; use /v1/completions instead",
        )
            .into_response();
    };
    let template = ChatTemplate::new(template_source.clone());
    let (bos, eos) = (
        state
            .engine
            .tokenizer
            .bos_token
            .and_then(|id| state.engine.tokenizer.token_text(id))
            .unwrap_or(""),
        state
            .engine
            .tokenizer
            .eos_token
            .and_then(|id| state.engine.tokenizer.token_text(id))
            .unwrap_or(""),
    );
    let mut prompt = match template.render(
        &req.messages,
        true,
        bos,
        eos,
        state.engine.role.enable_thinking(),
    ) {
        Ok(p) => p,
        Err(err) => return (StatusCode::BAD_REQUEST, err.to_string()).into_response(),
    };
    if state.engine.role.suppresses_reasoning() {
        prompt.push_str(EMPTY_THINK_BLOCK);
    }
    let tokens = state.engine.tokenizer.encode(&prompt, false);

    let mut sampling = SamplingParams::default_for_role(state.engine.role);
    if let Some(v) = req.temperature {
        sampling.temperature = v;
    }
    if let Some(v) = req.top_p {
        sampling.top_p = v;
    }
    if let Some(v) = req.seed {
        sampling.seed = v;
    }
    let max_tokens = req.max_tokens.unwrap_or(512);
    let stop_token_ids = state.engine.tokenizer.stop_token_ids();
    let created = unix_now();
    let model = state.model_label.clone();

    let mut rx = state
        .engine
        .generate(GenerateRequest {
            prompt_tokens: tokens,
            sampling,
            max_tokens,
            stop_token_ids,
            cache_prompt: req.cache_prompt,
        })
        .await;

    if !req.stream {
        let mut content = String::new();
        let mut finish_reason = "stop";
        let mut usage = json!({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0});
        let mut timings = serde_json::Value::Null;
        while let Some(event) = rx.recv().await {
            match event {
                StreamEvent::Token(text) => content.push_str(&text),
                StreamEvent::Done {
                    finish_reason: fr,
                    stats,
                } => {
                    finish_reason = finish_reason_str(fr);
                    usage = usage_json(&stats);
                    timings = timings_json(&stats);
                    break;
                }
                StreamEvent::Error(err) => {
                    return (StatusCode::INTERNAL_SERVER_ERROR, err).into_response();
                }
            }
        }
        let content = state
            .engine
            .tokenizer
            .clean_up_tokenization_spaces(&content);
        return Json(json!({
            "id": format!("chatcmpl-{created}"),
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }],
            "usage": usage,
            "timings": timings,
        }))
        .into_response();
    }

    let stream = async_stream::stream! {
        let id = format!("chatcmpl-{created}");
        loop {
            let Some(event) = rx.recv().await else { break };
            match event {
                StreamEvent::Token(text) => {
                    let chunk = json!({
                        "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": null}],
                    });
                    yield Ok::<_, std::convert::Infallible>(axum::response::sse::Event::default().data(chunk.to_string()));
                }
                StreamEvent::Done { finish_reason, stats } => {
                    // The final chunk carries what the request cost, in both
                    // OpenAI's shape (`usage`) and llama.cpp's (`timings`,
                    // `prompt_progress`) — a streaming client otherwise has
                    // only its own wall clock, which cannot separate prefill
                    // from decode or a cache hit from real work.
                    let chunk = json!({
                        "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason_str(finish_reason)}],
                        "usage": usage_json(&stats),
                        "timings": timings_json(&stats),
                        "prompt_progress": prompt_progress_json(&stats),
                    });
                    yield Ok(axum::response::sse::Event::default().data(chunk.to_string()));
                    yield Ok(axum::response::sse::Event::default().data("[DONE]"));
                    break;
                }
                StreamEvent::Error(err) => {
                    yield Ok(axum::response::sse::Event::default().data(json!({"error": err}).to_string()));
                    break;
                }
            }
        }
    };
    axum::response::sse::Sse::new(stream).into_response()
}

#[derive(Deserialize)]
pub struct CompletionsRequest {
    prompt: String,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    stream: bool,
    /// Keep generating to `max_tokens` even if the model emits EOS (llama.cpp's
    /// field name). Used by benchmarks (`orangu-bench --depths`, `llama-bench
    /// -d`) to time a fixed number of decode steps at a given context depth
    /// regardless of what the model would otherwise stop on.
    #[serde(default)]
    ignore_eos: bool,
    /// See [`ChatCompletionRequest::cache_prompt`]. `orangu-bench --pp` sets
    /// this `false` so each timed run prefills for real.
    #[serde(default = "default_cache_prompt")]
    cache_prompt: bool,
}

pub async fn completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionsRequest>,
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
    let mut sampling = SamplingParams::default_for_role(state.engine.role);
    if let Some(v) = req.temperature {
        sampling.temperature = v;
    }
    let max_tokens = req.max_tokens.unwrap_or(256);
    // `ignore_eos` drops the EOS stop token so generation runs the full
    // `max_tokens` — the "measure decode, not content" contract benchmarks need.
    let stop_token_ids: Vec<u32> = if req.ignore_eos {
        Vec::new()
    } else {
        state.engine.tokenizer.stop_token_ids()
    };
    let created = unix_now();
    let model = state.model_label.clone();

    let mut rx = state
        .engine
        .generate(GenerateRequest {
            prompt_tokens: tokens,
            sampling,
            max_tokens,
            stop_token_ids,
            cache_prompt: req.cache_prompt,
        })
        .await;

    if req.stream {
        let stream = async_stream::stream! {
            loop {
                let Some(event) = rx.recv().await else { break };
                match event {
                    StreamEvent::Token(text) => {
                        let chunk = json!({
                            "id": format!("cmpl-{created}"), "object": "text_completion", "created": created,
                            "model": model, "choices": [{"index": 0, "text": text, "finish_reason": null}],
                        });
                        yield Ok::<_, std::convert::Infallible>(axum::response::sse::Event::default().data(chunk.to_string()));
                    }
                    StreamEvent::Done { stats, .. } => {
                        // A final chunk before `[DONE]`, carrying the same
                        // cost figures the chat endpoint reports — this is
                        // the endpoint `orangu-bench` measures through, so
                        // it is where prefill numbers have to come from.
                        let chunk = json!({
                            "id": format!("cmpl-{created}"), "object": "text_completion", "created": created,
                            "model": model, "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
                            "usage": usage_json(&stats),
                            "timings": timings_json(&stats),
                            "prompt_progress": prompt_progress_json(&stats),
                        });
                        yield Ok(axum::response::sse::Event::default().data(chunk.to_string()));
                        yield Ok(axum::response::sse::Event::default().data("[DONE]"));
                        break;
                    }
                    StreamEvent::Error(err) => {
                        yield Ok(axum::response::sse::Event::default().data(json!({"error": err}).to_string()));
                        break;
                    }
                }
            }
        };
        return axum::response::sse::Sse::new(stream).into_response();
    }

    let mut text = String::new();
    let mut finish_reason = "stop";
    let mut usage = json!({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0});
    let mut timings = serde_json::Value::Null;
    while let Some(event) = rx.recv().await {
        match event {
            StreamEvent::Token(t) => text.push_str(&t),
            StreamEvent::Done {
                finish_reason: fr,
                stats,
            } => {
                finish_reason = finish_reason_str(fr);
                usage = usage_json(&stats);
                timings = timings_json(&stats);
                break;
            }
            StreamEvent::Error(err) => {
                return (StatusCode::INTERNAL_SERVER_ERROR, err).into_response();
            }
        }
    }
    let text = state.engine.tokenizer.clean_up_tokenization_spaces(&text);
    Json(json!({
        "id": format!("cmpl-{created}"),
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "text": text, "finish_reason": finish_reason}],
        "usage": usage,
        "timings": timings,
    }))
    .into_response()
}

#[derive(Deserialize)]
#[serde(untagged)]
enum EmbeddingsInput {
    One(String),
    Many(Vec<String>),
}

#[derive(Deserialize)]
pub struct EmbeddingsRequest {
    input: EmbeddingsInput,
}

#[derive(Serialize)]
struct EmbeddingDatum {
    object: &'static str,
    embedding: Vec<f32>,
    index: usize,
}

pub async fn embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbeddingsRequest>,
) -> axum::response::Response {
    let inputs = match req.input {
        EmbeddingsInput::One(s) => vec![s],
        EmbeddingsInput::Many(v) => v,
    };
    let mut data = Vec::with_capacity(inputs.len());
    for (index, text) in inputs.into_iter().enumerate() {
        match pooled_embedding(&state, &text).await {
            Ok(embedding) => data.push(EmbeddingDatum {
                object: "embedding",
                embedding,
                index,
            }),
            Err(err) => return (StatusCode::INTERNAL_SERVER_ERROR, err).into_response(),
        }
    }
    Json(json!({"object": "list", "data": data, "model": state.model_label})).into_response()
}

/// Pools a model's per-token final hidden states per its own `<arch>.
/// pooling_type` ([`PoolingType`] — `Mean`, e.g. `gemma-embedding`, or
/// `Last`, e.g. `qwen3vl`-embedding models; every other value falls back
/// to `Mean`, see that type's own doc comment), runs the model's own
/// [`ModelForward::post_pool_projection`] (identity for most
/// architectures; `gemma-embedding`'s `dense_2`/`dense_3` sentence-
/// transformers adapters for that one), then L2-normalizes the result.
pub(crate) async fn pooled_embedding(
    state: &Arc<AppState>,
    text: &str,
) -> Result<Vec<f32>, String> {
    let tokens = state.engine.tokenizer.encode_for_embedding(text);
    let model = state.engine.model.clone();
    let n_embd = model.config().n_embd;
    let pooling_type = model.config().pooling_type;
    let hidden = tokio::task::spawn_blocking({
        let model = model.clone();
        move || model.forward_hidden_states(&tokens)
    })
    .await
    .map_err(|err| err.to_string())?
    .map_err(|err| err.to_string())?;
    let n_tokens = (hidden.len() / n_embd).max(1);

    let pooled = match pooling_type {
        PoolingType::Last => hidden[(n_tokens - 1) * n_embd..].to_vec(),
        PoolingType::Mean => {
            let mut pooled = vec![0f32; n_embd];
            for row in hidden.chunks(n_embd) {
                for (p, v) in pooled.iter_mut().zip(row.iter()) {
                    *p += v;
                }
            }
            for v in pooled.iter_mut() {
                *v /= n_tokens as f32;
            }
            pooled
        }
    };

    let mut pooled = tokio::task::spawn_blocking(move || model.post_pool_projection(pooled))
        .await
        .map_err(|err| err.to_string())?
        .map_err(|err| err.to_string())?;

    let norm = pooled.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in pooled.iter_mut() {
            *v /= norm;
        }
    }
    Ok(pooled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn stats() -> GenerateStats {
        GenerateStats {
            prompt_tokens: 200,
            cached_tokens: 50,
            prompt_time: Duration::from_millis(4000),
            generated_tokens: 30,
            generate_time: Duration::from_millis(1000),
        }
    }

    #[test]
    fn usage_reports_totals_and_the_cached_share_of_the_prompt() {
        let usage = usage_json(&stats());
        assert_eq!(usage["prompt_tokens"], 200);
        assert_eq!(usage["completion_tokens"], 30);
        assert_eq!(usage["total_tokens"], 230);
        assert_eq!(usage["prompt_tokens_details"]["cached_tokens"], 50);
    }

    #[test]
    fn timings_report_prompt_and_decode_rates_in_llama_cpp_field_names() {
        let timings = timings_json(&stats());
        assert_eq!(timings["prompt_n"], 200);
        assert_eq!(timings["prompt_ms"], 4000.0);
        // 200 tokens in 4s, 30 tokens in 1s.
        assert_eq!(timings["prompt_per_second"], 50.0);
        assert_eq!(timings["predicted_per_second"], 30.0);
        assert_eq!(timings["prompt_per_token_ms"], 20.0);
    }

    /// A prompt served entirely from cache must not read as instant prefill:
    /// `processed` is what actually went through a forward pass.
    #[test]
    fn prompt_progress_separates_cached_tokens_from_processed_ones() {
        let progress = prompt_progress_json(&stats());
        assert_eq!(progress["total"], 200);
        assert_eq!(progress["cache"], 50);
        assert_eq!(progress["processed"], 150);
        assert_eq!(progress["time_ms"], 4000);
    }

    /// Rate helpers divide by the token count; an empty generation (a prompt
    /// that stopped immediately) must not produce a division by zero.
    #[test]
    fn timings_survive_a_request_that_generated_nothing() {
        let empty = GenerateStats {
            prompt_tokens: 0,
            cached_tokens: 0,
            prompt_time: Duration::ZERO,
            generated_tokens: 0,
            generate_time: Duration::ZERO,
        };
        let timings = timings_json(&empty);
        assert_eq!(timings["prompt_per_token_ms"], 0.0);
        assert_eq!(timings["predicted_per_token_ms"], 0.0);
    }
}
