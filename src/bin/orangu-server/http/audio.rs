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

//! Text to speech over HTTP, for a `qwen3tts` model: OpenAI's
//! `POST /v1/audio/speech`, the voices it speaks with
//! (`GET /v1/audio/voices`), and its defaults through `/props`.

use axum::{
    Json,
    extract::State,
    http::{HeaderValue, StatusCode, header},
    response::IntoResponse,
};
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;

use super::AppState;
use crate::engine::audio::{AudioDefaults, AudioFormat, Pipeline, Speech, SpeechRequest};

fn error(status: StatusCode, message: &str) -> axum::response::Response {
    (status, Json(json!({"error": {"message": message}}))).into_response()
}

/// The pipeline, or the response that says this server has none.
pub fn pipeline_or_reject(
    state: &AppState,
) -> Result<Arc<Pipeline>, Box<axum::response::Response>> {
    match &state.engine.audio {
        Some(pipeline) => Ok(pipeline.clone()),
        None => Err(Box::new(error(
            StatusCode::NOT_IMPLEMENTED,
            "this server is not serving a speech model (a qwen3tts GGUF); /v1/audio/speech \
             needs one",
        ))),
    }
}

/// The answer a text endpoint gives on a speech model, which has no text
/// to generate or embed: `None` on any other model.
pub fn reject_text_endpoint(state: &AppState, endpoint: &str) -> Option<axum::response::Response> {
    state.engine.audio.is_some().then(|| {
        error(
            StatusCode::NOT_IMPLEMENTED,
            &format!(
                "this server is serving a speech model (qwen3tts); {endpoint} needs a language \
                 model — use /v1/audio/speech, or /v1/chat/completions for a spoken reply"
            ),
        )
    })
}

/// The per-request knobs a clip may override, over the pipeline's defaults.
#[derive(Default)]
pub struct SpeechParams<'a> {
    pub text: &'a str,
    pub voice: Option<&'a str>,
    pub language: Option<&'a str>,
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub repetition_penalty: Option<f32>,
    pub max_seconds: Option<f64>,
    pub seed: Option<u64>,
}

/// `defaults` with `p` over them, checked as the configuration is.
pub fn build_request(
    defaults: &AudioDefaults,
    p: SpeechParams<'_>,
) -> Result<SpeechRequest, String> {
    let mut r = SpeechRequest::from_defaults(p.text, defaults);
    if let Some(voice) = p.voice.map(str::trim).filter(|v| !v.is_empty()) {
        r.voice = Some(voice.to_string());
    }
    if let Some(language) = p.language.map(str::trim).filter(|v| !v.is_empty()) {
        r.language = Some(language.to_ascii_lowercase());
    }
    if let Some(t) = p.temperature {
        if !(t.is_finite() && (0.0..=2.0).contains(&t)) {
            return Err(format!("temperature must be from 0 to 2, not {t}"));
        }
        r.temperature = t;
    }
    if let Some(k) = p.top_k {
        if k == 0 {
            return Err("top_k must be 1 or more".to_string());
        }
        r.top_k = k;
    }
    if let Some(penalty) = p.repetition_penalty {
        if !(penalty.is_finite() && (1.0..=2.0).contains(&penalty)) {
            return Err(format!(
                "repetition_penalty must be from 1 to 2, not {penalty}"
            ));
        }
        r.repetition_penalty = penalty;
    }
    if let Some(s) = p.max_seconds {
        if !(s.is_finite() && s > 0.0 && s <= crate::engine::audio::MAX_SECONDS) {
            return Err(format!(
                "max_seconds must be above 0 and at most {}, not {s}",
                crate::engine::audio::MAX_SECONDS
            ));
        }
        r.max_seconds = s;
    }
    r.seed = p.seed;
    Ok(r)
}

/// `response_format`: one of the formats this server encodes, or the
/// default; OpenAI's compressed ones are refused by name.
pub fn parse_format(value: Option<&str>, default: AudioFormat) -> Result<AudioFormat, String> {
    match value.map(str::trim).filter(|v| !v.is_empty()) {
        None => Ok(default),
        Some(v) => AudioFormat::parse(v).ok_or_else(|| {
            format!(
                "response_format '{v}' is not one this server encodes (expected {})",
                AudioFormat::NAMES
            )
        }),
    }
}

/// One line in the log per clip.
pub fn log_speech(speech: &Speech, elapsed: std::time::Duration, how: &str) {
    log::info!(
        "orangu-server: [audio] {:.1} s of speech ({} frames, voice {}, language {}, seed {}) in \
         {:.1} s{how} — prompt ({} rows) {:.1} s, speaking {:.1} s, decoding {:.1} s{}",
        speech.seconds(),
        speech.frames,
        speech.voice,
        speech.language,
        speech.seed,
        elapsed.as_secs_f64(),
        speech.prompt_rows,
        speech.timings.prompt.as_secs_f64(),
        speech.timings.speak.as_secs_f64(),
        speech.timings.decode.as_secs_f64(),
        if speech.finished {
            ""
        } else {
            " (cut at the length limit)"
        }
    );
}

/// `POST /v1/audio/speech`'s body: OpenAI's fields, and this server's
/// own beside them.
#[derive(Deserialize)]
pub struct SpeechBody {
    /// Ignored: the server speaks with the model it serves.
    #[serde(default)]
    #[allow(dead_code)]
    model: Option<String>,
    input: String,
    /// A voice's name, or `{"id": name}`.
    #[serde(default)]
    voice: Option<serde_json::Value>,
    /// A Base model is not steered by instructions; refused when given,
    /// rather than silently ignored.
    #[serde(default)]
    instructions: Option<String>,
    #[serde(default)]
    response_format: Option<String>,
    #[serde(default)]
    speed: Option<f32>,
    #[serde(default)]
    stream_format: Option<String>,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    repetition_penalty: Option<f32>,
    #[serde(default)]
    max_seconds: Option<f64>,
    #[serde(default)]
    seed: Option<u64>,
}

fn voice_name(value: Option<&serde_json::Value>) -> Result<Option<String>, String> {
    match value {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::String(s)) => Ok(Some(s.clone())),
        Some(serde_json::Value::Object(o)) => match o.get("id").or_else(|| o.get("name")) {
            Some(serde_json::Value::String(s)) => Ok(Some(s.clone())),
            _ => Err("voice must be a name, or an object with an \"id\"".to_string()),
        },
        Some(_) => Err("voice must be a name, or an object with an \"id\"".to_string()),
    }
}

pub async fn speech(
    State(state): State<Arc<AppState>>,
    Json(body): Json<SpeechBody>,
) -> axum::response::Response {
    let pipeline = match pipeline_or_reject(&state) {
        Ok(pipeline) => pipeline,
        Err(response) => return *response,
    };
    let defaults = pipeline.defaults();
    let format = match parse_format(body.response_format.as_deref(), defaults.format) {
        Ok(format) => format,
        Err(err) => return error(StatusCode::BAD_REQUEST, &err),
    };
    if let Some(speed) = body.speed
        && (speed - 1.0).abs() > 1e-3
    {
        return error(
            StatusCode::BAD_REQUEST,
            &format!(
                "speed {speed} is not supported; this server speaks at the voice's own pace (1.0)"
            ),
        );
    }
    if body
        .stream_format
        .as_deref()
        .is_some_and(|f| !f.eq_ignore_ascii_case("audio"))
    {
        return error(
            StatusCode::BAD_REQUEST,
            "stream_format must be \"audio\": the clip is sent whole",
        );
    }
    if body
        .instructions
        .as_deref()
        .is_some_and(|i| !i.trim().is_empty())
    {
        return error(
            StatusCode::BAD_REQUEST,
            "instructions are not supported: a Qwen3-TTS Base model speaks in the voice it is \
             given and cannot be steered by a description",
        );
    }
    let voice = match voice_name(body.voice.as_ref()) {
        Ok(voice) => voice,
        Err(err) => return error(StatusCode::BAD_REQUEST, &err),
    };
    let request = match build_request(
        &defaults,
        SpeechParams {
            text: &body.input,
            voice: voice.as_deref(),
            language: body.language.as_deref(),
            temperature: body.temperature,
            top_k: body.top_k,
            repetition_penalty: body.repetition_penalty,
            max_seconds: body.max_seconds,
            seed: body.seed,
        },
    ) {
        Ok(request) => request,
        Err(err) => return error(StatusCode::BAD_REQUEST, &err),
    };
    if request.text.trim().is_empty() {
        return error(StatusCode::BAD_REQUEST, "input is empty");
    }
    let started = std::time::Instant::now();
    let speech = match pipeline.speak(request).await {
        Ok(speech) => speech,
        Err(err) if err.contains("no voice") || err.contains("unknown language") => {
            return error(StatusCode::BAD_REQUEST, &err);
        }
        Err(err) => return error(StatusCode::INTERNAL_SERVER_ERROR, &err),
    };
    log_speech(&speech, started.elapsed(), "");
    let bytes = format.encode(&speech.samples, speech.sample_rate);
    let mut response = (StatusCode::OK, bytes).into_response();
    let headers = response.headers_mut();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static(format.mime()),
    );
    for (name, value) in [
        ("x-orangu-voice", speech.voice.clone()),
        ("x-orangu-seed", speech.seed.to_string()),
        ("x-orangu-frames", speech.frames.to_string()),
        ("x-orangu-sample-rate", speech.sample_rate.to_string()),
        ("x-orangu-finished", speech.finished.to_string()),
    ] {
        if let Ok(value) = HeaderValue::from_str(&value) {
            headers.insert(name, value);
        }
    }
    response
}

/// `GET /v1/audio/voices`: who can speak, in which languages, and in what.
pub async fn voices(State(state): State<Arc<AppState>>) -> axum::response::Response {
    match pipeline_or_reject(&state) {
        Ok(pipeline) => Json(voices_json(&pipeline)).into_response(),
        Err(response) => *response,
    }
}

fn voices_json(pipeline: &Pipeline) -> serde_json::Value {
    let defaults = pipeline.defaults();
    let voices = pipeline.voices();
    json!({
        "object": "list",
        "data": voices.iter().map(|v| json!({
            "id": v.name,
            "object": "voice",
            "description": v.description,
        })).collect::<Vec<_>>(),
        "default": defaults.voice.clone().or_else(|| voices.first().map(|v| v.name.clone())),
        "languages": pipeline.languages(),
        "formats": AudioFormat::NAMES.split(", ").collect::<Vec<_>>(),
    })
}

/// `POST /props`' `audio`: new defaults, refused whole when one is out of
/// range. `reset` starts from the configuration's.
#[derive(Deserialize, Default)]
pub struct AudioSettings {
    #[serde(default)]
    pub reset: bool,
    #[serde(default)]
    pub voice: Option<String>,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub format: Option<String>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub max_seconds: Option<f64>,
}

pub fn apply_settings(pipeline: &Pipeline, settings: &AudioSettings) -> Result<(), String> {
    let base = if settings.reset {
        pipeline.configured.clone()
    } else {
        pipeline.defaults()
    };
    let format = parse_format(settings.format.as_deref(), base.format)?;
    let r = build_request(
        &base,
        SpeechParams {
            text: "",
            voice: settings.voice.as_deref(),
            language: settings.language.as_deref(),
            temperature: settings.temperature,
            top_k: settings.top_k,
            repetition_penalty: settings.repetition_penalty,
            max_seconds: settings.max_seconds,
            seed: None,
        },
    )?;
    pipeline
        .set_defaults(AudioDefaults {
            voice: r.voice,
            language: r.language.unwrap_or_else(|| "auto".to_string()),
            format,
            temperature: r.temperature,
            top_k: r.top_k,
            repetition_penalty: r.repetition_penalty,
            max_seconds: r.max_seconds,
        })
        .map_err(|e| e.to_string())
}

fn defaults_json(d: &AudioDefaults) -> serde_json::Value {
    json!({
        "voice": d.voice,
        "language": d.language,
        "format": d.format.label(),
        "temperature": d.temperature,
        "top_k": d.top_k,
        "repetition_penalty": d.repetition_penalty,
        "max_seconds": d.max_seconds,
    })
}

/// What `/props` says about a speech model.
pub fn props_json(pipeline: &Pipeline) -> serde_json::Value {
    json!({
        "architecture": orangu::model_spec::QWEN3TTS_ARCHITECTURE,
        "codec": pipeline.companions.codec.display().to_string(),
        "voice_packs": pipeline
            .companions
            .voices
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>(),
        "voices": pipeline.voices(),
        "languages": pipeline.languages(),
        "formats": AudioFormat::NAMES.split(", ").collect::<Vec<_>>(),
        "sample_rate": pipeline.sample_rate(),
        "frame_rate": pipeline.frame_rate(),
        "talker_layers": pipeline.talker_layers(),
        "predictor_layers": pipeline.predictor_layers(),
        "defaults": defaults_json(&pipeline.defaults()),
        "configured": defaults_json(&pipeline.configured),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_compressed_formats_are_refused_by_name() {
        let err = parse_format(Some("mp3"), AudioFormat::Wav).unwrap_err();
        assert!(err.contains("mp3") && err.contains("wav, flac, pcm"));
        assert_eq!(parse_format(None, AudioFormat::Flac), Ok(AudioFormat::Flac));
        assert_eq!(
            parse_format(Some("pcm"), AudioFormat::Wav),
            Ok(AudioFormat::Pcm)
        );
    }

    #[test]
    fn a_voice_is_a_name_or_an_object_with_an_id() {
        assert_eq!(voice_name(None), Ok(None));
        assert_eq!(
            voice_name(Some(&json!("clear"))),
            Ok(Some("clear".to_string()))
        );
        assert_eq!(
            voice_name(Some(&json!({"id": "clear"}))),
            Ok(Some("clear".to_string()))
        );
        assert!(voice_name(Some(&json!(3))).is_err());
    }

    #[test]
    fn a_request_is_checked_as_the_configuration_is() {
        let d = AudioDefaults::default();
        assert!(
            build_request(
                &d,
                SpeechParams {
                    text: "hi",
                    temperature: Some(5.0),
                    ..Default::default()
                }
            )
            .is_err()
        );
        let r = build_request(
            &d,
            SpeechParams {
                text: "hi",
                voice: Some("clear"),
                language: Some("English"),
                seed: Some(3),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(r.voice.as_deref(), Some("clear"));
        assert_eq!(r.language.as_deref(), Some("english"));
        assert_eq!(r.seed, Some(3));
    }
}
