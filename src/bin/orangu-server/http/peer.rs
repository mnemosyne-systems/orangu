//! Generation one node of a workers channel asks of another, for a request
//! that did not arrive as an API call: the web console's chat. The prompt
//! travels as tokens — every node a request is routed to holds the same
//! model (`workers::node::route`), so its tokenizer reads them the same way
//! — and the answer comes back as the engine's own events, one JSON object
//! a line, so the asking side runs the rest of its turn exactly as it would
//! for a local answer.

use super::AppState;
use crate::engine::generate::{FinishReason, GenerateRequest, GenerateStats, StreamEvent};
use crate::engine::sampling::SamplingParams;
use axum::response::IntoResponse;
use serde::{Deserialize, Serialize};
use std::{convert::Infallible, sync::Arc, time::Duration};
use tokio::sync::mpsc::{self, UnboundedReceiver};

/// Where a node serves [`generate`].
pub const PATH: &str = "/v1/workers/generate";

/// A prompt, already tokenized, and how to sample its answer.
#[derive(Debug, Serialize, Deserialize)]
pub struct PeerGenerate {
    pub prompt_tokens: Vec<u32>,
    pub temperature: f32,
    pub top_p: f32,
    pub min_p: f32,
    pub max_tokens: usize,
    pub cache_point: Option<usize>,
}

/// One [`StreamEvent`] on the wire. Progress, timings and log
/// probabilities are not sent: the console shows none of them. The finished
/// answer does carry where its forwards spent their time, which the
/// console's topology view of the channel overlays.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum PeerEvent {
    Token(String),
    Reasoning(String),
    Done {
        prompt_tokens: usize,
        cached_tokens: usize,
        prompt_ms: u64,
        generated_tokens: usize,
        generate_ms: u64,
        length: bool,
        /// [`GenerateStats::workers`] in microseconds. Absent from a node
        /// that does not send it, which reads as no tree.
        #[serde(default)]
        workers: Vec<(String, u64)>,
    },
    Overloaded,
    NoRoom(String),
    Error(String),
}

impl PeerEvent {
    fn from_event(event: StreamEvent) -> Option<Self> {
        Some(match event {
            StreamEvent::Token(text) => Self::Token(text),
            StreamEvent::Reasoning(text) => Self::Reasoning(text),
            StreamEvent::Done {
                stats,
                finish_reason,
            } => Self::Done {
                prompt_tokens: stats.prompt_tokens,
                cached_tokens: stats.cached_tokens,
                prompt_ms: stats.prompt_time.as_millis() as u64,
                generated_tokens: stats.generated_tokens,
                generate_ms: stats.generate_time.as_millis() as u64,
                length: finish_reason == FinishReason::Length,
                workers: stats
                    .workers
                    .iter()
                    .map(|(node, t)| (node.clone(), t.as_micros() as u64))
                    .collect(),
            },
            StreamEvent::Overloaded => Self::Overloaded,
            StreamEvent::NoRoom(text) => Self::NoRoom(text),
            StreamEvent::Error(text) => Self::Error(text),
            StreamEvent::PromptProgress { .. }
            | StreamEvent::Timings(_)
            | StreamEvent::Logprob(_) => return None,
        })
    }

    fn into_event(self) -> StreamEvent {
        match self {
            Self::Token(text) => StreamEvent::Token(text),
            Self::Reasoning(text) => StreamEvent::Reasoning(text),
            Self::Done {
                prompt_tokens,
                cached_tokens,
                prompt_ms,
                generated_tokens,
                generate_ms,
                length,
                workers,
            } => StreamEvent::Done {
                stats: GenerateStats {
                    prompt_tokens,
                    cached_tokens,
                    prompt_time: Duration::from_millis(prompt_ms),
                    generated_tokens,
                    generate_time: Duration::from_millis(generate_ms),
                    dropped_tokens: 0,
                    workers: workers
                        .into_iter()
                        .map(|(node, us)| (node, Duration::from_micros(us)))
                        .collect(),
                },
                finish_reason: if length {
                    FinishReason::Length
                } else {
                    FinishReason::Stop
                },
            },
            Self::Overloaded => StreamEvent::Overloaded,
            Self::NoRoom(text) => StreamEvent::NoRoom(text),
            Self::Error(text) => StreamEvent::Error(text),
        }
    }

    fn ends(&self) -> bool {
        !matches!(self, Self::Token(_) | Self::Reasoning(_))
    }
}

/// The request [`PeerGenerate`] describes, as this engine runs it.
pub fn generate_request(req: PeerGenerate, stop_token_ids: Vec<u32>) -> GenerateRequest {
    let defaults = SamplingParams::default();
    GenerateRequest {
        json_output: false,
        json_schema: None,
        prompt_tokens: req.prompt_tokens,
        sampling: SamplingParams {
            temperature: req.temperature,
            top_p: req.top_p,
            min_p: req.min_p,
            ..defaults
        },
        max_tokens: req.max_tokens,
        stop_token_ids,
        role: None,
        cache_point: req.cache_point,
        background: false,
        reasoning_budget: None,
        cache_prompt: true,
        id_slot: None,
        timings_per_token: false,
        ..Default::default()
    }
}

/// `POST /v1/workers/generate`: runs [`PeerGenerate`] here, whatever this
/// node's load — the asking node already chose it — and streams the
/// answer back a [`PeerEvent`] a line.
pub async fn generate(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    axum::Json(req): axum::Json<PeerGenerate>,
) -> axum::response::Response {
    let stop_token_ids = state.engine.tokenizer.stop_token_ids();
    let mut rx = state
        .engine
        .generate(generate_request(req, stop_token_ids))
        .await;
    let lines = async_stream::stream! {
        while let Some(event) = rx.recv().await {
            let Some(event) = PeerEvent::from_event(event) else {
                continue;
            };
            let ends = event.ends();
            let mut line = serde_json::to_string(&event).unwrap_or_default();
            line.push('\n');
            yield Ok::<_, Infallible>(line);
            if ends {
                break;
            }
        }
    };
    (
        [(axum::http::header::CONTENT_TYPE, "application/x-ndjson")],
        axum::body::Body::from_stream(lines),
    )
        .into_response()
}

/// Asks the node whose API is at `base` to run `req`, and hands back its
/// answer as engine events. `Err` when that node cannot be reached or
/// refuses, so the caller runs the request here instead; a node lost once
/// its answer has started ends the answer with an [`StreamEvent::Error`].
pub async fn generate_on(
    base: &str,
    api_key: Option<&str>,
    req: &PeerGenerate,
) -> Result<UnboundedReceiver<StreamEvent>, String> {
    static CLIENT: std::sync::OnceLock<reqwest::Client> = std::sync::OnceLock::new();
    let client = CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(2))
            .build()
            .unwrap_or_default()
    });
    let mut post = client.post(format!("{base}{PATH}")).json(req);
    if let Some(key) = api_key {
        post = post.bearer_auth(key);
    }
    let mut answer = post.send().await.map_err(|e| e.to_string())?;
    if !answer.status().is_success() {
        return Err(format!("{} answered {}", base, answer.status()));
    }
    let (tx, rx) = mpsc::unbounded_channel();
    let base = base.to_string();
    tokio::spawn(async move {
        let mut pending: Vec<u8> = Vec::new();
        loop {
            let chunk = match answer.chunk().await {
                Ok(Some(chunk)) => chunk,
                Ok(None) => break,
                Err(e) => {
                    let _ = tx.send(StreamEvent::Error(format!(
                        "the node serving this answer ({base}) was lost: {e}"
                    )));
                    return;
                }
            };
            pending.extend_from_slice(&chunk);
            while let Some(end) = pending.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = pending.drain(..=end).collect();
                let Ok(event) = serde_json::from_slice::<PeerEvent>(&line) else {
                    continue;
                };
                let ends = event.ends();
                if tx.send(event.into_event()).is_err() || ends {
                    return;
                }
            }
        }
        let _ = tx.send(StreamEvent::Error(format!(
            "the node serving this answer ({base}) ended it early"
        )));
    });
    Ok(rx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_survive_the_wire() {
        let stats = GenerateStats {
            prompt_tokens: 12,
            cached_tokens: 4,
            prompt_time: Duration::from_millis(30),
            generated_tokens: 7,
            generate_time: Duration::from_millis(90),
            dropped_tokens: 0,
            workers: vec![("b:8400".into(), Duration::from_micros(31_250))],
        };
        let sent = [
            StreamEvent::Reasoning("hm".into()),
            StreamEvent::Token("Hi".into()),
            StreamEvent::Done {
                stats,
                finish_reason: FinishReason::Length,
            },
        ];
        for event in sent {
            let line = serde_json::to_string(&PeerEvent::from_event(event).unwrap()).unwrap();
            let back = serde_json::from_str::<PeerEvent>(&line)
                .unwrap()
                .into_event();
            match back {
                StreamEvent::Reasoning(t) => assert_eq!(t, "hm"),
                StreamEvent::Token(t) => assert_eq!(t, "Hi"),
                StreamEvent::Done {
                    stats,
                    finish_reason,
                } => {
                    assert_eq!(finish_reason, FinishReason::Length);
                    assert_eq!((stats.prompt_tokens, stats.cached_tokens), (12, 4));
                    assert_eq!(stats.generated_tokens, 7);
                    assert_eq!(stats.generate_time, Duration::from_millis(90));
                    assert_eq!(
                        stats.workers,
                        vec![("b:8400".to_string(), Duration::from_micros(31_250))]
                    );
                }
                _ => panic!("an event changed kind on the wire"),
            }
        }
    }

    /// A node that sends no stage times still ends its answer: it reads as
    /// one served without a tree.
    #[test]
    fn a_done_without_stage_times_reads_as_no_tree() {
        let line = r#"{"done":{"prompt_tokens":1,"cached_tokens":0,"prompt_ms":2,"generated_tokens":3,"generate_ms":4,"length":false}}"#;
        match serde_json::from_str::<PeerEvent>(line)
            .unwrap()
            .into_event()
        {
            StreamEvent::Done { stats, .. } => assert!(stats.workers.is_empty()),
            _ => panic!("a done event changed kind on the wire"),
        }
    }

    #[test]
    fn progress_stays_home() {
        let progress = StreamEvent::PromptProgress {
            total: 1,
            cached: 0,
            processed: 1,
            elapsed: Duration::ZERO,
        };
        assert!(PeerEvent::from_event(progress).is_none());
    }
}
