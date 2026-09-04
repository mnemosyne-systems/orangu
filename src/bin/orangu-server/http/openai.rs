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
use crate::engine::tool_calls;

/// How long an answer may get when the request does not say — the
/// `max_tokens` default for `/v1/chat/completions`, clamped afterwards to
/// what is left of the model's context window.
///
/// It has to be *this* large because the endpoint's real workload is an agent
/// writing a file: the file's content travels inside a `<|tool_call>` span,
/// and a span cut off by the cap never closes, so `engine::tool_calls`
/// correctly refuses to read it as a call and it lands in the answer as prose
/// instead. The tool is never run and nothing reaches disk. 512 — the default
/// this replaces — is about forty lines of code, and a request that hit it
/// wrote no file at all; 4096 was still short of a page of HTML with its game
/// loop in it. This is a long source file with room for the model to think
/// first.
///
/// A cap has to exist all the same, and the reason is **time**, not memory:
/// `LayerCache::new_strided` only *reserves* its buffers, so an oversized
/// capacity costs address space rather than RSS (see its own comment). What an
/// unbounded default would cost is a runaway generation holding the slot every
/// other request queues behind until the whole context window is full — on the
/// order of an hour at a local decode rate, against a couple of minutes here.
///
/// Reaching this is no longer silent either way: the response carries
/// `finish_reason: "length"`, and the client says so — see
/// `llm::StreamMetrics::truncated`.
const DEFAULT_MAX_TOKENS: usize = 8192;

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
    /// The rest of the sampler, under the same names `/completion` uses.
    ///
    /// These are not OpenAI's fields, and that is the point: the sampler
    /// has knobs OpenAI's schema has no word for, and a request that names
    /// one it cannot reach is **silently ignored** rather than rejected —
    /// `serde` drops unknown keys. That failure mode is worse than the gap
    /// it comes from: a caller sending `repeat_penalty` and seeing no change
    /// concludes the penalty does nothing, when in fact the value never
    /// arrived. It cost a real debugging session exactly that way.
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    repeat_penalty: Option<f32>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    seed: Option<u64>,
    /// llama.cpp's field name and default: reuse an already-computed KV cache
    /// for whatever prefix of this prompt one exists for. `false` forces a
    /// full prefill.
    #[serde(default = "default_cache_prompt")]
    cache_prompt: bool,
    /// Pin this request to one specific slot (llama.cpp's field name), so a
    /// conversation returns to the slot holding its own KV cache instead of
    /// reprefilling on a stranger's. See `engine::generate::GenerateRequest`.
    #[serde(default)]
    id_slot: Option<usize>,
    /// llama.cpp's field: attach a `timings` object to every streamed chunk,
    /// not only the last one. A client showing a live decode rate has no
    /// other honest source for it.
    #[serde(default)]
    timings_per_token: bool,
    /// llama.cpp's field: emit `prompt_progress` chunks *during* prefill, so
    /// a client can show how far a long prompt has got instead of an
    /// indefinite spinner.
    #[serde(default)]
    return_progress: bool,
    /// OpenAI's structured-output request. `{"type": "json_object"}`
    /// constrains generation so that only tokens keeping the output a valid
    /// JSON prefix can be sampled, and so that generation cannot stop until
    /// the document is complete — see `engine::constraint`.
    #[serde(default)]
    response_format: Option<serde_json::Value>,
    /// OpenAI's tool array, handed to the chat template as `tools`. Kept as
    /// raw JSON: what a tool declaration must contain is the template's
    /// business, and every field this server invented an opinion about would
    /// be one a model's own template could no longer see.
    #[serde(default)]
    tools: Option<serde_json::Value>,
}

pub(crate) fn default_cache_prompt() -> bool {
    true
}

/// Whether `response_format` asks for JSON.
///
/// OpenAI's field is an object, `{"type": "json_object"}`. `json_schema` is
/// recognised too and treated as `json_object` — the output is constrained to
/// valid JSON, but **not** to the schema's shape, which is a strictly larger
/// job. Answering "the type I do not implement is simply ignored" would be
/// worse: a caller asking for a schema and receiving free text has no way to
/// tell that the field never arrived, which is the exact failure this file's
/// own comment about silently-dropped sampler fields already records.
///
/// Anything else — including `{"type": "text"}`, the default — is
/// unconstrained.
fn wants_json(response_format: Option<&serde_json::Value>) -> bool {
    response_format
        .and_then(|v| v.get("type"))
        .and_then(|v| v.as_str())
        .is_some_and(|t| matches!(t, "json_object" | "json_schema"))
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

/// Appends [`EMPTY_THINK_BLOCK`] to a rendered prompt when — and only
/// when — that approximation is the right tool for this model.
///
/// It is not always. A `<|start|>…<|message|>`-framed model
/// (`Tokenizer::message_framing`, `muse-glimmer`) ends its generation
/// prompt *mid-header*, at `<|start|>assistant`, waiting for the model to
/// write its own recipient. Appending anything there lands inside the
/// header rather than after it, and the model then continues from a
/// malformed turn: the observed result was a reply that never wrote
/// `<|message|>` at all, which `engine::generate`'s own header filter then
/// withheld in full — an empty answer.
///
/// The same holds for a model that types each message body with a control
/// token (`Tokenizer::content_kinds`, `inkling`): its generation prompt
/// ends at `<|message_model|>`, and a `<think>` block appended there is
/// body text arriving before the marker that says what kind of body this
/// is.
///
/// Those models need no approximation anyway. Their reasoning is a whole
/// separate message — addressed `to=self`, or opened with
/// `<|content_thinking|>` — and a suppressing role drops it exactly (see
/// `MessageHeader`) rather than trying to talk the model out of producing
/// one.
pub(crate) fn append_reasoning_suppression(
    prompt: &mut String,
    role: crate::config::Role,
    tokenizer: &crate::engine::tokenizer::Tokenizer,
) {
    if role.suppresses_reasoning()
        && tokenizer.message_framing().is_none()
        && tokenizer.content_kinds().is_none()
    {
        prompt.push_str(EMPTY_THINK_BLOCK);
    }
}

/// Reject an `id_slot` this server has no slot for, rather than silently
/// ignoring it and quietly reprefilling every turn — the exact failure the
/// field was added to end. Same shape `POST /slots/{id}` already answers with.
pub(crate) fn reject_unknown_slot(
    state: &AppState,
    id_slot: Option<usize>,
) -> Option<axum::response::Response> {
    let index = id_slot?;
    (index >= state.engine.slots.total()).then(|| {
        (
            StatusCode::BAD_REQUEST,
            format!(
                "id_slot {index} out of range (server has {} slots)\n",
                state.engine.slots.total()
            ),
        )
            .into_response()
    })
}

/// OpenAI's `tool_calls` array. `index` is what a streaming client
/// accumulates deltas by, so it counts calls across the whole response rather
/// than restarting per chunk. Each call carries a whole `function` object in
/// one delta — this server recognises a call only once it is completely
/// written, so there is never a partial one to stream.
fn tool_calls_json_from(
    calls: &[tool_calls::ParsedToolCall],
    created: u64,
    first_index: usize,
) -> serde_json::Value {
    serde_json::Value::Array(
        calls
            .iter()
            .enumerate()
            .map(|(n, call)| {
                let index = first_index + n;
                json!({
                    "index": index,
                    "id": format!("call-{created}-{index}"),
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                })
            })
            .collect(),
    )
}

fn tool_calls_json(calls: &[tool_calls::ParsedToolCall], created: u64) -> serde_json::Value {
    tool_calls_json_from(calls, created, 0)
}

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
    if let Some(rejection) = reject_unknown_slot(&state, req.id_slot) {
        return rejection;
    }
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
    // An empty `tools: []` is treated as no tools: a template gating on
    // `{%- if tools -%}` would agree, and passing it through only invites a
    // template to emit an empty declaration block.
    let tools = req
        .tools
        .as_ref()
        .filter(|t| !matches!(t.as_array(), Some(a) if a.is_empty()));
    let mut prompt = match template.render_with_tools(
        &req.messages,
        true,
        bos,
        eos,
        state.engine.reasoning(),
        tools,
    ) {
        Ok(p) => p,
        Err(err) => return (StatusCode::BAD_REQUEST, err.to_string()).into_response(),
    };
    append_reasoning_suppression(&mut prompt, state.engine.role, &state.engine.tokenizer);
    let tokens = state.engine.tokenizer.encode(&prompt, false);

    let sampling = sampling_for(
        state.engine.role,
        RequestSampling {
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k,
            min_p: req.min_p,
            repeat_penalty: req.repeat_penalty,
            seed: req.seed,
        },
    );
    let max_tokens = req.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
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
            json_output: wants_json(req.response_format.as_ref()),
            cache_prompt: req.cache_prompt,
            id_slot: req.id_slot,
            timings_per_token: req.stream && req.timings_per_token,
        })
        .await;

    if !req.stream {
        let mut content = String::new();
        let mut reasoning = String::new();
        let mut finish_reason = "stop";
        let mut usage = json!({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0});
        let mut timings = serde_json::Value::Null;
        while let Some(event) = rx.recv().await {
            match event {
                StreamEvent::Token(text) => content.push_str(&text),
                StreamEvent::Reasoning(text) => reasoning.push_str(&text),
                // Progress and per-token timings only mean something to a
                // reader watching the stream; a whole response already
                // carries the final `timings`/`usage`.
                StreamEvent::PromptProgress { .. } | StreamEvent::Timings(_) => {}
                StreamEvent::Done {
                    finish_reason: fr,
                    stats,
                } => {
                    finish_reason = finish_reason_str(fr);
                    usage = usage_json(&stats);
                    timings = timings_json(&stats);
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
        let split = tool_calls::split(&content);
        let mut message = json!({"role": "assistant", "content": split.content});
        // `reasoning_content` is where DeepSeek put a reasoning model's
        // thinking and where llama-server puts it under
        // `--reasoning-format deepseek`; a client that does not know the
        // field ignores it and sees the answer alone, which is the point.
        // Absent rather than empty when there was no thinking, so "this
        // model does not reason" and "it reasoned about nothing" stay
        // distinguishable.
        if !reasoning.is_empty() {
            message["reasoning_content"] = json!(
                state
                    .engine
                    .tokenizer
                    .clean_up_tokenization_spaces(&reasoning)
            );
        }
        if split.has_calls() {
            message["tool_calls"] = tool_calls_json(&split.calls, created);
            // OpenAI's contract: a turn that called tools finishes for that
            // reason, whatever the sampler stopped on. A client keying off
            // `"stop"` would treat the calls as a finished answer.
            finish_reason = "tool_calls";
        }
        return Json(json!({
            "id": format!("chatcmpl-{created}"),
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }],
            "usage": usage,
            "timings": timings,
        }))
        .into_response();
    }

    let return_progress = req.return_progress;
    let stream = async_stream::stream! {
        let id = format!("chatcmpl-{created}");
        // Tool-call syntax has to be recognised before it is forwarded, and a
        // delimiter arrives a few tokens at a time. `pending` holds back text
        // that might still turn into a call; everything else streams straight
        // through, so an ordinary answer is not delayed at all.
        let mut pending = String::new();
        let mut emitted_calls = 0usize;
        let mut saw_calls = false;
        loop {
            let Some(event) = rx.recv().await else { break };
            match event {
                StreamEvent::PromptProgress { total, cached, processed, elapsed } => {
                    if return_progress {
                        // llama.cpp's shape, so a client reads mid-prefill
                        // progress from either server the same way.
                        let chunk = json!({
                            "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": null}],
                            "prompt_progress": {
                                "total": total,
                                "cache": cached,
                                "processed": processed,
                                "time_ms": elapsed.as_millis() as i64,
                            },
                        });
                        yield Ok::<_, std::convert::Infallible>(axum::response::sse::Event::default().data(chunk.to_string()));
                    }
                }
                StreamEvent::Timings(stats) => {
                    // A chunk of its own rather than riding along with the
                    // next content delta: content can be held back for a
                    // while by the tool-call splitter above, and a decode
                    // rate that arrives in bursts is worse than none.
                    let chunk = json!({
                        "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": null}],
                        "timings": timings_json(&stats),
                    });
                    yield Ok(axum::response::sse::Event::default().data(chunk.to_string()));
                }
                // Reasoning streams straight through in its own delta field,
                // bypassing the tool-call splitter entirely: a call is
                // something the model addresses to the caller, never part of
                // a body it addressed to itself, and holding thinking back
                // to see whether it turns into one would delay the only
                // thing there is to show during a long think.
                StreamEvent::Reasoning(text) => {
                    let chunk = json!({
                        "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                        "choices": [{"index": 0, "delta": {"reasoning_content": text}, "finish_reason": null}],
                    });
                    yield Ok::<_, std::convert::Infallible>(axum::response::sse::Event::default().data(chunk.to_string()));
                }
                StreamEvent::Token(text) => {
                    pending.push_str(&text);
                    if tool_calls::may_be_partial(&pending) {
                        continue;
                    }
                    let split = tool_calls::split(&pending);
                    pending.clear();
                    if !split.content.is_empty() {
                        let chunk = json!({
                            "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                            "choices": [{"index": 0, "delta": {"content": split.content}, "finish_reason": null}],
                        });
                        yield Ok::<_, std::convert::Infallible>(axum::response::sse::Event::default().data(chunk.to_string()));
                    }
                    if split.has_calls() {
                        saw_calls = true;
                        let calls = tool_calls_json_from(&split.calls, created, emitted_calls);
                        emitted_calls += split.calls.len();
                        let chunk = json!({
                            "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                            "choices": [{"index": 0, "delta": {"tool_calls": calls}, "finish_reason": null}],
                        });
                        yield Ok(axum::response::sse::Event::default().data(chunk.to_string()));
                    }
                }
                StreamEvent::Done { finish_reason, stats } => {
                    // Whatever is still held back never became a call. It is
                    // the model's output and belongs in the answer.
                    if !pending.is_empty() {
                        let split = tool_calls::split(&pending);
                        if !split.content.is_empty() {
                            let chunk = json!({
                                "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                                "choices": [{"index": 0, "delta": {"content": split.content}, "finish_reason": null}],
                            });
                            yield Ok(axum::response::sse::Event::default().data(chunk.to_string()));
                        }
                        if split.has_calls() {
                            saw_calls = true;
                            let calls = tool_calls_json_from(&split.calls, created, emitted_calls);
                            let chunk = json!({
                                "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                                "choices": [{"index": 0, "delta": {"tool_calls": calls}, "finish_reason": null}],
                            });
                            yield Ok(axum::response::sse::Event::default().data(chunk.to_string()));
                        }
                    }
                    // The final chunk carries what the request cost, in both
                    // OpenAI's shape (`usage`) and llama.cpp's (`timings`,
                    // `prompt_progress`) — a streaming client otherwise has
                    // only its own wall clock, which cannot separate prefill
                    // from decode or a cache hit from real work.
                    let reason = if saw_calls { "tool_calls" } else { finish_reason_str(finish_reason) };
                    let chunk = json!({
                        "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
                        "usage": usage_json(&stats),
                        "timings": timings_json(&stats),
                        "prompt_progress": prompt_progress_json(&stats),
                    });
                    yield Ok(axum::response::sse::Event::default().data(chunk.to_string()));
                    yield Ok(axum::response::sse::Event::default().data("[DONE]"));
                    break;
                }
                StreamEvent::Overloaded => {
                    yield Ok(axum::response::sse::Event::default()
                        .data(serde_json::json!({"error": crate::http::OVERLOADED_MESSAGE}).to_string()));
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

/// The sampler fields an HTTP request may override, gathered so the two
/// OpenAI-compatible endpoints apply them through one place.
///
/// Every field is `Option`: `None` keeps the role's default rather than
/// meaning "zero". A `temperature` of `0.0` is greedy and a `repeat_penalty`
/// of `1.0` is off, so the difference between "unset" and "set to the
/// neutral value" is real and cannot be collapsed.
struct RequestSampling {
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    min_p: Option<f32>,
    repeat_penalty: Option<f32>,
    seed: Option<u64>,
}

/// The role's defaults with a request's overrides applied.
///
/// One function rather than a copy per endpoint: the two of them had drifted
/// — chat honoured `temperature`/`top_p`/`seed` and completions honoured
/// `temperature` alone — and nothing about either endpoint made that
/// deliberate. A field added to one and forgotten in the other is not a
/// visible bug, because the ignored value simply has no effect.
fn sampling_for(role: crate::config::Role, req: RequestSampling) -> SamplingParams {
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

#[derive(Deserialize)]
pub struct CompletionsRequest {
    prompt: String,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    /// The rest of the sampler — see [`ChatCompletionRequest::top_k`] for why
    /// these are here and why their absence was worse than a gap. This
    /// endpoint had *only* `temperature`, so a request setting `top_k` or
    /// `repeat_penalty` here was accepted and discarded.
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
    /// Not part of OpenAI's completions schema, accepted anyway for the same
    /// reason the sampler fields above are: a caller that reaches for it and
    /// is silently ignored concludes the feature does not work.
    #[serde(default)]
    response_format: Option<serde_json::Value>,
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
    /// See [`ChatCompletionRequest::id_slot`].
    #[serde(default)]
    id_slot: Option<usize>,
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
    if let Some(rejection) = reject_unknown_slot(&state, req.id_slot) {
        return rejection;
    }
    let tokens = state.engine.tokenizer.encode(&req.prompt, true);
    let sampling = sampling_for(
        state.engine.role,
        RequestSampling {
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k,
            min_p: req.min_p,
            repeat_penalty: req.repeat_penalty,
            seed: req.seed,
        },
    );
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
            json_output: wants_json(req.response_format.as_ref()),
            cache_prompt: req.cache_prompt,
            id_slot: req.id_slot,
            timings_per_token: false,
        })
        .await;

    if req.stream {
        let stream = async_stream::stream! {
            loop {
                let Some(event) = rx.recv().await else { break };
                match event {
                    StreamEvent::PromptProgress { .. } | StreamEvent::Timings(_) => {}
                    // A raw completion has one field to put text in and no
                    // message shape to split it across, so reasoning stays
                    // in it — a caller that prefilled `<think>` here asked
                    // for exactly that text back.
                    StreamEvent::Token(text) | StreamEvent::Reasoning(text) => {
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
                    StreamEvent::Overloaded => {
                        yield Ok(axum::response::sse::Event::default()
                            .data(serde_json::json!({"error": crate::http::OVERLOADED_MESSAGE}).to_string()));
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
            StreamEvent::PromptProgress { .. } | StreamEvent::Timings(_) => {}
            // See the streaming arm above: one field, no message shape.
            StreamEvent::Token(t) | StreamEvent::Reasoning(t) => text.push_str(&t),
            StreamEvent::Done {
                finish_reason: fr,
                stats,
            } => {
                finish_reason = finish_reason_str(fr);
                usage = usage_json(&stats);
                timings = timings_json(&stats);
                break;
            }
            StreamEvent::Overloaded => return crate::http::overloaded_response(),
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
    let mut prompt_tokens = 0usize;
    for (index, text) in inputs.into_iter().enumerate() {
        match pooled_embedding(&state, &text).await {
            Ok(pooled) => {
                prompt_tokens += pooled.prompt_tokens;
                data.push(EmbeddingDatum {
                    object: "embedding",
                    embedding: pooled.embedding,
                    index,
                });
            }
            Err(err) => return (StatusCode::INTERNAL_SERVER_ERROR, err).into_response(),
        }
    }
    Json(json!({
        "object": "list",
        "data": data,
        "model": state.model_label,
        // OpenAI's embeddings `usage`, which has these two fields and not the
        // `completion_tokens` of `usage_json` above — there is no completion.
        // Summed across a batched `input`, as OpenAI's is.
        //
        // Not cosmetic: the token count is not recoverable from the response
        // otherwise, so anything measuring embedding throughput had to
        // tokenize the input itself with a second tool to learn what it just
        // paid for. `doc/perf/embed_bench.sh` shelled out to `llama-tokenize`
        // for exactly this, and `orangu-bench --embed` reads this field
        // instead — from either server, since llama-server sends it too.
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens,
        },
    }))
    .into_response()
}

/// One embedding and the prompt length that produced it.
///
/// The token count is carried out rather than recomputed by the caller
/// because this is where it is known exactly — `encode_for_embedding` applies
/// the model's own embedding-specific framing, so a caller re-tokenizing the
/// same string could easily disagree with the forward pass that was actually
/// run. It is what `/v1/embeddings` reports as `usage.prompt_tokens`.
pub(crate) struct PooledEmbedding {
    pub embedding: Vec<f32>,
    pub prompt_tokens: usize,
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
) -> Result<PooledEmbedding, String> {
    let tokens = state.engine.tokenizer.encode_for_embedding(text);
    let prompt_tokens = tokens.len();
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
    Ok(PooledEmbedding {
        embedding: pooled,
        prompt_tokens,
    })
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

    /// Every sampler field a request can name must actually reach the
    /// sampler, on **both** OpenAI-compatible endpoints.
    ///
    /// This is the test whose absence cost a debugging session. `serde`
    /// drops unknown keys, so an endpoint that lacks a field accepts a
    /// request naming it and silently ignores the value — indistinguishable,
    /// from the client's side, from a knob that genuinely does nothing.
    /// `/v1/completions` honoured `temperature` alone; a `repeat_penalty` or
    /// `top_k` sent to it went nowhere, and the conclusion drawn was that the
    /// repetition penalty was not the cause of a garbling it was in fact
    /// causing.
    ///
    /// Deserializing real JSON rather than constructing the structs is the
    /// point: a field that exists on the struct but is spelled differently on
    /// the wire fails here and nowhere else.
    #[test]
    fn every_sampler_field_a_request_names_reaches_the_sampler() {
        let body = serde_json::json!({
            "prompt": "hi",
            "messages": [],
            "temperature": 0.25,
            "top_p": 0.6,
            "top_k": 7,
            "min_p": 0.125,
            "repeat_penalty": 1.5,
            "seed": 99,
        });
        let role = crate::config::Role::All;

        let chat: ChatCompletionRequest =
            serde_json::from_value(body.clone()).expect("chat request parses");
        let completions: CompletionsRequest =
            serde_json::from_value(body).expect("completions request parses");

        for (label, got) in [
            (
                "chat",
                sampling_for(
                    role,
                    RequestSampling {
                        temperature: chat.temperature,
                        top_p: chat.top_p,
                        top_k: chat.top_k,
                        min_p: chat.min_p,
                        repeat_penalty: chat.repeat_penalty,
                        seed: chat.seed,
                    },
                ),
            ),
            (
                "completions",
                sampling_for(
                    role,
                    RequestSampling {
                        temperature: completions.temperature,
                        top_p: completions.top_p,
                        top_k: completions.top_k,
                        min_p: completions.min_p,
                        repeat_penalty: completions.repeat_penalty,
                        seed: completions.seed,
                    },
                ),
            ),
        ] {
            assert_eq!(got.temperature, 0.25, "{label}: temperature");
            assert_eq!(got.top_p, 0.6, "{label}: top_p");
            assert_eq!(got.top_k, 7, "{label}: top_k");
            assert_eq!(got.min_p, 0.125, "{label}: min_p");
            assert_eq!(got.repeat_penalty, 1.5, "{label}: repeat_penalty");
            assert_eq!(got.seed, 99, "{label}: seed");
        }
    }

    /// An unset field keeps the role's default rather than collapsing to
    /// zero — and the default penalty is **off**.
    ///
    /// `1.0` is the neutral value, not the absent one, so "unset" and "set to
    /// neutral" must both leave the penalty off while remaining distinct for
    /// every other field. The default itself is pinned here because it is a
    /// behavioural promise: a penalty applied per token id falls hardest on
    /// the newline, and turning it on by default corrupts generated code.
    #[test]
    fn an_unset_field_keeps_the_role_default_and_the_penalty_is_off() {
        let empty: CompletionsRequest =
            serde_json::from_value(serde_json::json!({"prompt": "hi"})).expect("parses");
        let got = sampling_for(
            crate::config::Role::All,
            RequestSampling {
                temperature: empty.temperature,
                top_p: empty.top_p,
                top_k: empty.top_k,
                min_p: empty.min_p,
                repeat_penalty: empty.repeat_penalty,
                seed: empty.seed,
            },
        );
        let default = SamplingParams::default_for_role(crate::config::Role::All);
        assert_eq!(got.temperature, default.temperature);
        assert_eq!(got.top_k, default.top_k);
        assert_eq!(
            got.repeat_penalty, 1.0,
            "the repetition penalty must default to off"
        );
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
