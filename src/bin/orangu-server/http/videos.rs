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

//! `/v1/videos` — OpenAI's Videos API, on an `ltxv` model.
//!
//! A clip is minutes of work, so the API is a job: `POST /v1/videos`
//! answers at once with a `video` object (`status: queued`), `GET
//! /v1/videos/{id}` reports its `progress` (and this server's
//! `eta_seconds`), and `GET /v1/videos/{id}/content` returns the MP4 once it
//! is `completed`. `DELETE /v1/videos/{id}` cancels a running clip or drops
//! a finished one; `GET /v1/videos` lists them.
//!
//! The request is OpenAI's (`prompt`, `seconds`, `size`) plus the knobs a
//! local model has: `frames` (in place of `seconds`), `fps`, `seed`,
//! `audio`, `two_stage`, `quality`. `stream: true` keeps the connection
//! open instead and sends server-sent events — `video.progress` per phase
//! and step, then `video.completed` with the job — for a client that would
//! rather not poll.
//!
//! Jobs live in memory: the last [`KEEP_JOBS`] are kept, and a restart
//! forgets them all.

use axum::{
    Json,
    body::Body,
    extract::{Path, State},
    http::{HeaderMap, StatusCode, header},
    response::IntoResponse,
};
use serde::Deserialize;
use serde_json::json;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, OnceLock};

use super::AppState;
use crate::engine::video::{
    CancelOnDrop, GeneratedVideo, Pipeline, Progress, VideoDefaults, VideoEvent, VideoRequest,
};

/// Finished and running jobs kept in memory, oldest dropped first.
pub const KEEP_JOBS: usize = 16;

#[derive(Deserialize, Default)]
pub struct VideoCreateRequest {
    #[serde(default)]
    pub prompt: String,
    /// OpenAI's: a string (`"4"`) or a number of seconds.
    #[serde(default)]
    pub seconds: Option<serde_json::Value>,
    /// `WIDTHxHEIGHT`.
    #[serde(default)]
    pub size: Option<String>,
    #[serde(default)]
    pub frames: Option<usize>,
    #[serde(default)]
    pub fps: Option<u32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub audio: Option<bool>,
    #[serde(default)]
    pub two_stage: Option<bool>,
    #[serde(default)]
    pub quality: Option<usize>,
    #[serde(default)]
    pub stream: bool,
}

/// The knobs a chat turn, a console turn and an API request share,
/// resolved against the server's defaults.
#[derive(Default, Clone)]
pub struct VideoParams<'a> {
    pub prompt: &'a str,
    pub seconds: Option<f64>,
    pub size: Option<&'a str>,
    pub frames: Option<usize>,
    pub fps: Option<u32>,
    pub seed: Option<u64>,
    pub audio: Option<bool>,
    pub two_stage: Option<bool>,
    pub quality: Option<usize>,
}

pub fn build_request(defaults: &VideoDefaults, p: VideoParams<'_>) -> Result<VideoRequest, String> {
    let prompt = p.prompt.trim();
    if prompt.is_empty() {
        return Err("prompt is empty".to_string());
    }
    let mut r = VideoRequest::from_defaults(prompt, defaults);
    if let Some(size) = p
        .size
        .map(str::trim)
        .filter(|s| !s.is_empty() && *s != "auto")
    {
        let (w, h) = crate::config::parse_video_size(size)
            .ok_or_else(|| format!("size '{size}' is not WIDTHxHEIGHT in multiples of 32"))?;
        r.width = w;
        r.height = h;
    }
    if let Some(fps) = p.fps {
        if !(1..=60).contains(&fps) {
            return Err("fps must be between 1 and 60".to_string());
        }
        r.fps = fps;
    }
    match (p.frames, p.seconds) {
        (Some(_), Some(_)) => return Err("give frames or seconds, not both".to_string()),
        (Some(frames), None) => r.frames = crate::engine::video::snap_frames(frames),
        (None, Some(seconds)) => {
            if !(seconds.is_finite() && seconds > 0.0 && seconds <= 60.0) {
                return Err("seconds must be more than 0 and at most 60".to_string());
            }
            r.frames = crate::engine::video::snap_frames((seconds * r.fps as f64).round() as usize);
        }
        (None, None) => {}
    }
    r.seed = p.seed;
    if let Some(audio) = p.audio {
        r.audio = audio;
    }
    if let Some(two) = p.two_stage {
        r.two_stage = two;
    }
    if let Some(q) = p.quality {
        if q > 255 {
            return Err("quality is the AV1 quantizer, 0 to 255".to_string());
        }
        r.quality = q;
    }
    Ok(r)
}

fn seconds_of(value: &serde_json::Value) -> Option<f64> {
    match value {
        serde_json::Value::Number(n) => n.as_f64(),
        serde_json::Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

/// The `video` object of `/props` (and the console's `/api/props`).
pub fn props_json(pipeline: &Pipeline) -> serde_json::Value {
    let tc = pipeline.transformer_config();
    let defaults = pipeline.defaults();
    let rate = pipeline.rate_model();
    let request = VideoRequest::from_defaults("", &defaults);
    let defaults_json = |d: &VideoDefaults| {
        json!({
            "size": format!("{}x{}", d.width, d.height),
            "frames": d.frames,
            "fps": d.fps,
            "seconds": d.frames as f64 / d.fps.max(1) as f64,
            "audio": d.audio,
            "two_stage": d.two_stage,
            "quality": d.quality,
        })
    };
    json!({
        "architecture": orangu::model_spec::LTXV_ARCHITECTURE,
        // Width and height must be multiples of this: 64 when the clip is
        // drawn at half size first (two stages), 32 otherwise.
        "size_unit": if pipeline.has_upscaler() && defaults.two_stage { 64 } else { 32 },
        "text_encoder": pipeline.companions.text_encoder.display().to_string(),
        "vae": pipeline.companions.vae.display().to_string(),
        "audio_vae": pipeline.companions.audio_vae.as_ref().map(|p| p.display().to_string()),
        "upscaler": pipeline.companions.upscaler.as_ref().map(|p| p.display().to_string()),
        "n_layer": tc.n_layer,
        "dim": tc.dim,
        "head_dim": tc.head_dim,
        "audio_dim": tc.audio_dim,
        "audio_head_dim": tc.audio_head_dim,
        "defaults": defaults_json(&defaults),
        "configured": defaults_json(&pipeline.configured),
        "estimated_default_seconds": pipeline.estimate(&request).ok(),
        // seconds = encode + Σ_stage steps × (per_step + per_token·n +
        // per_token_sq·n²) + pixels × (decode_per_pixel + other_per_pixel)
        // + audio_per_second × seconds of sound; n the stage's video tokens
        // (frames' latent count × height/32 × width/32, a quarter of that in
        // stage 1 of two).
        "rate": {
            "encode": rate.encode,
            "per_step": rate.per_step,
            "per_token": rate.per_token,
            "per_token_sq": rate.per_token_sq,
            "decode_per_pixel": rate.decode_per_pixel,
            "other_per_pixel": rate.other_per_pixel,
            "audio_per_second": rate.audio_per_second,
        },
    })
}

/// `POST /props`' `video` object: defaults to change, every key optional;
/// `reset` puts them back to the configuration first.
#[derive(Deserialize, Default, Debug)]
pub struct VideoSettings {
    #[serde(default)]
    pub reset: bool,
    #[serde(default)]
    pub size: Option<String>,
    #[serde(default)]
    pub frames: Option<usize>,
    #[serde(default)]
    pub seconds: Option<f64>,
    #[serde(default)]
    pub fps: Option<u32>,
    #[serde(default)]
    pub audio: Option<bool>,
    #[serde(default)]
    pub two_stage: Option<bool>,
    #[serde(default)]
    pub quality: Option<usize>,
}

/// Applies `settings` to the pipeline's defaults, refusing them whole when
/// any is out of range — the same checks a request gets.
pub fn apply_settings(pipeline: &Pipeline, settings: &VideoSettings) -> Result<(), String> {
    let base = if settings.reset {
        pipeline.configured.clone()
    } else {
        pipeline.defaults()
    };
    let r = build_request(
        &base,
        VideoParams {
            prompt: "settings",
            seconds: settings.seconds,
            size: settings.size.as_deref(),
            frames: settings.frames,
            fps: settings.fps,
            audio: settings.audio,
            two_stage: settings.two_stage,
            quality: settings.quality,
            seed: None,
        },
    )?;
    crate::engine::video::Plan::new(&r, pipeline.has_upscaler()).map_err(|e| e.to_string())?;
    pipeline.set_defaults(VideoDefaults {
        width: r.width,
        height: r.height,
        frames: r.frames,
        fps: r.fps,
        audio: r.audio,
        two_stage: r.two_stage,
        quality: r.quality,
    });
    Ok(())
}

/// The pipeline, or the response that says this server has none.
pub fn pipeline_or_reject(
    state: &AppState,
) -> Result<Arc<Pipeline>, Box<axum::response::Response>> {
    match &state.engine.video {
        Some(pipeline) => Ok(pipeline.clone()),
        None => Err(Box::new(error(
            StatusCode::NOT_IMPLEMENTED,
            "this server is not serving a video model (an ltxv GGUF); /v1/videos needs one",
        ))),
    }
}

// ---------------------------------------------------------------------------
// Jobs
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Status {
    Queued,
    InProgress,
    Completed,
    Failed,
}

impl Status {
    fn name(self) -> &'static str {
        match self {
            Status::Queued => "queued",
            Status::InProgress => "in_progress",
            Status::Completed => "completed",
            Status::Failed => "failed",
        }
    }
}

pub struct Job {
    pub id: String,
    pub model: String,
    pub created_at: u64,
    pub completed_at: Option<u64>,
    pub status: Status,
    pub request: VideoRequest,
    pub last: Option<Progress>,
    pub video: Option<Arc<GeneratedVideo>>,
    pub error: Option<String>,
    cancel: Option<CancelOnDrop>,
}

impl Job {
    pub fn json(&self) -> serde_json::Value {
        let r = &self.request;
        let progress = match (self.status, self.last) {
            (Status::Completed, _) => 100,
            (_, Some(p)) if p.steps > 0 => (p.step * 100 / (p.steps + 1)).min(99),
            _ => 0,
        };
        json!({
            "id": self.id,
            "object": "video",
            "model": self.model,
            "status": self.status.name(),
            "progress": progress,
            "phase": self.last.map(|p| p.phase),
            "eta_seconds": self.last.map(|p| p.eta_seconds),
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "prompt": r.prompt,
            "size": format!("{}x{}", r.width, r.height),
            "seconds": format!("{}", (r.frames as f64 / r.fps.max(1) as f64 * 100.0).round() / 100.0),
            "frames": crate::engine::video::snap_frames(r.frames),
            "fps": r.fps,
            "seed": self.video.as_ref().map(|v| v.seed).or(r.seed),
            "audio": self.video.as_ref().map(|v| v.has_audio),
            "bytes": self.video.as_ref().map(|v| v.bytes.len()),
            "generation_ms": self.video.as_ref().map(|v| v.elapsed.as_millis() as u64),
            "timings": self.video.as_ref().map(|v| timings_json(&v.timings)),
            "error": self.error.as_ref().map(|e| json!({"message": e})),
        })
    }
}

fn timings_json(t: &crate::engine::video::Timings) -> serde_json::Value {
    json!({
        "encode_ms": t.encode.as_millis() as u64,
        "stage_1_ms": t.stage_1.as_millis() as u64,
        "upscale_ms": t.upscale.as_millis() as u64,
        "stage_2_ms": t.stage_2.as_millis() as u64,
        "decode_ms": t.decode.as_millis() as u64,
        "audio_ms": t.audio.as_millis() as u64,
        "mux_ms": t.mux.as_millis() as u64,
    })
}

fn jobs() -> &'static Mutex<VecDeque<Job>> {
    static JOBS: OnceLock<Mutex<VecDeque<Job>>> = OnceLock::new();
    JOBS.get_or_init(|| Mutex::new(VecDeque::new()))
}

fn with_job<T>(id: &str, f: impl FnOnce(&mut Job) -> T) -> Option<T> {
    let mut jobs = jobs().lock().unwrap_or_else(|p| p.into_inner());
    jobs.iter_mut().find(|j| j.id == id).map(f)
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Starts `request` as a job and returns its id. The job's progress and
/// result land in the job table; `listener` hears every event too.
pub fn start_job(
    pipeline: &Arc<Pipeline>,
    model: &str,
    request: VideoRequest,
    listener: Option<tokio::sync::mpsc::Sender<VideoEvent>>,
) -> String {
    let id = format!("video_{:016x}", rand_id());
    let (mut rx, cancel) = pipeline.spawn(request.clone());
    {
        let mut jobs = jobs().lock().unwrap_or_else(|p| p.into_inner());
        while jobs.len() >= KEEP_JOBS {
            // The oldest finished one goes first; a table of running jobs
            // only ever grows past the limit by the one being added.
            match jobs
                .iter()
                .position(|j| matches!(j.status, Status::Completed | Status::Failed))
            {
                Some(i) => {
                    jobs.remove(i);
                }
                None => break,
            }
        }
        jobs.push_back(Job {
            id: id.clone(),
            model: model.to_string(),
            created_at: now(),
            completed_at: None,
            status: Status::Queued,
            request,
            last: None,
            video: None,
            error: None,
            cancel: Some(cancel),
        });
    }
    let job_id = id.clone();
    tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            let forward = match &event {
                VideoEvent::Progress(p) => {
                    let p = *p;
                    with_job(&job_id, |j| {
                        j.status = Status::InProgress;
                        j.last = Some(p);
                    });
                    Some(VideoEvent::Progress(p))
                }
                VideoEvent::Done(video) => {
                    let video: Arc<GeneratedVideo> = Arc::new((**video).clone());
                    with_job(&job_id, |j| {
                        j.status = Status::Completed;
                        j.completed_at = Some(now());
                        j.video = Some(video.clone());
                        j.cancel = None;
                    });
                    Some(VideoEvent::Done(Box::new((*video).clone())))
                }
                VideoEvent::Error(err) => {
                    let err = err.clone();
                    with_job(&job_id, |j| {
                        j.status = Status::Failed;
                        j.completed_at = Some(now());
                        j.error = Some(err.clone());
                        j.cancel = None;
                    });
                    Some(VideoEvent::Error(err))
                }
            };
            if let (Some(tx), Some(event)) = (&listener, forward) {
                let _ = tx.send(event).await;
            }
        }
    });
    id
}

fn rand_id() -> u64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let c = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut x = nanos ^ c.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^ (x >> 33)
}

/// Job `id` as its API object.
pub fn job_json(id: &str) -> Option<serde_json::Value> {
    with_job(id, |j| j.json())
}

/// The finished clip of job `id`, if there is one.
pub fn job_video(id: &str) -> Option<Arc<GeneratedVideo>> {
    with_job(id, |j| j.video.clone()).flatten()
}

/// OpenAI's error shape, `{"error": {"message"}}`, for every refusal here.
fn error(status: StatusCode, message: &str) -> axum::response::Response {
    (status, Json(json!({"error": {"message": message}}))).into_response()
}

fn not_found(id: &str) -> axum::response::Response {
    (
        StatusCode::NOT_FOUND,
        Json(json!({"error": {"message": format!("no video '{id}'")}})),
    )
        .into_response()
}

pub async fn create(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<VideoCreateRequest>,
) -> axum::response::Response {
    let _ = super::openai::request_role(&state, &headers);
    let pipeline = match pipeline_or_reject(&state) {
        Ok(pipeline) => pipeline,
        Err(response) => return *response,
    };
    let seconds = match &req.seconds {
        Some(v) => match seconds_of(v) {
            Some(s) => Some(s),
            None => return error(StatusCode::BAD_REQUEST, "seconds is not a number"),
        },
        None => None,
    };
    let request = match build_request(
        &pipeline.defaults(),
        VideoParams {
            prompt: &req.prompt,
            seconds,
            size: req.size.as_deref(),
            frames: req.frames,
            fps: req.fps,
            seed: req.seed,
            audio: req.audio,
            two_stage: req.two_stage,
            quality: req.quality,
        },
    ) {
        Ok(request) => request,
        Err(err) => return error(StatusCode::BAD_REQUEST, &err),
    };
    if let Err(err) = crate::engine::video::Plan::new(&request, pipeline.has_upscaler()) {
        return error(StatusCode::BAD_REQUEST, &err.to_string());
    }
    if !req.stream {
        let id = start_job(&pipeline, &state.model_label, request, None);
        let body = with_job(&id, |j| j.json()).unwrap_or_default();
        return Json(body).into_response();
    }
    let (tx, mut rx) = tokio::sync::mpsc::channel(64);
    let id = start_job(&pipeline, &state.model_label, request, Some(tx));
    let stream = async_stream::stream! {
        // Leaving the stream does not stop the job: it is in the table, and
        // `DELETE /v1/videos/{id}` is how a client cancels one.
        let start = with_job(&id, |j| j.json()).unwrap_or_default();
        yield Ok::<_, std::convert::Infallible>(axum::response::sse::Event::default()
            .data(json!({"type": "video.created", "video": start}).to_string()));
        while let Some(event) = rx.recv().await {
            match event {
                VideoEvent::Progress(p) => {
                    let chunk = json!({
                        "type": "video.progress",
                        "id": id,
                        "step": p.step, "steps": p.steps, "phase": p.phase,
                        "eta_seconds": p.eta_seconds,
                    });
                    yield Ok(axum::response::sse::Event::default().data(chunk.to_string()));
                }
                VideoEvent::Done(_) => {
                    let job = with_job(&id, |j| j.json()).unwrap_or_default();
                    yield Ok(axum::response::sse::Event::default()
                        .data(json!({"type": "video.completed", "video": job}).to_string()));
                    yield Ok(axum::response::sse::Event::default().data("[DONE]"));
                    break;
                }
                VideoEvent::Error(err) => {
                    yield Ok(axum::response::sse::Event::default()
                        .data(json!({"type": "video.failed", "id": id, "error": {"message": err}}).to_string()));
                    break;
                }
            }
        }
    };
    axum::response::sse::Sse::new(stream).into_response()
}

pub async fn retrieve(Path(id): Path<String>) -> axum::response::Response {
    match with_job(&id, |j| j.json()) {
        Some(body) => Json(body).into_response(),
        None => not_found(&id),
    }
}

pub async fn list() -> axum::response::Response {
    let jobs = jobs().lock().unwrap_or_else(|p| p.into_inner());
    let data: Vec<serde_json::Value> = jobs.iter().rev().map(Job::json).collect();
    Json(json!({"object": "list", "data": data})).into_response()
}

pub async fn delete(Path(id): Path<String>) -> axum::response::Response {
    let mut jobs = jobs().lock().unwrap_or_else(|p| p.into_inner());
    match jobs.iter().position(|j| j.id == id) {
        // Dropping the job drops its `CancelOnDrop`, which stops a running
        // clip at its next block.
        Some(i) => {
            jobs.remove(i);
            Json(json!({"id": id, "object": "video.deleted", "deleted": true})).into_response()
        }
        None => not_found(&id),
    }
}

/// The MP4 of a completed job, with `Range` support so a `<video>` element
/// can seek in it.
pub async fn content(Path(id): Path<String>, headers: HeaderMap) -> axum::response::Response {
    let Some(status) = with_job(&id, |j| j.status) else {
        return not_found(&id);
    };
    let Some(video) = job_video(&id) else {
        return (
            StatusCode::CONFLICT,
            Json(json!({"error": {"message": format!("video '{id}' is {}", status.name())}})),
        )
            .into_response();
    };
    mp4_response(&video.bytes, &headers)
}

/// `bytes` as `video/mp4`, honouring a single `Range: bytes=a-b`.
pub fn mp4_response(bytes: &[u8], headers: &HeaderMap) -> axum::response::Response {
    ranged_response(bytes, headers, "video/mp4")
}

/// `bytes` as `content_type`, the part a `Range` header asks for when it
/// asks — what lets a player seek in a clip or a recording.
pub fn ranged_response(
    bytes: &[u8],
    headers: &HeaderMap,
    content_type: &'static str,
) -> axum::response::Response {
    let total = bytes.len();
    let range = headers
        .get(header::RANGE)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("bytes="))
        .and_then(|v| {
            let (a, b) = v.split_once('-')?;
            let start: usize = if a.is_empty() {
                total.saturating_sub(b.parse().ok()?)
            } else {
                a.parse().ok()?
            };
            let end: usize = if a.is_empty() || b.is_empty() {
                total.checked_sub(1)?
            } else {
                b.parse::<usize>().ok()?.min(total - 1)
            };
            (start <= end && end < total).then_some((start, end))
        });
    match range {
        Some((start, end)) => axum::response::Response::builder()
            .status(StatusCode::PARTIAL_CONTENT)
            .header(header::CONTENT_TYPE, content_type)
            .header(header::ACCEPT_RANGES, "bytes")
            .header(
                header::CONTENT_RANGE,
                format!("bytes {start}-{end}/{total}"),
            )
            .header(header::CONTENT_LENGTH, end - start + 1)
            .body(Body::from(bytes[start..=end].to_vec()))
            .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response()),
        None => axum::response::Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, content_type)
            .header(header::ACCEPT_RANGES, "bytes")
            .header(header::CONTENT_LENGTH, total)
            .body(Body::from(bytes.to_vec()))
            .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_request_takes_seconds_or_frames_and_snaps_them() {
        let d = VideoDefaults::default();
        let r = build_request(
            &d,
            VideoParams {
                prompt: "x",
                seconds: Some(2.0),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(r.frames, 49);
        let r = build_request(
            &d,
            VideoParams {
                prompt: "x",
                frames: Some(30),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(r.frames, 33);
        assert!(
            build_request(
                &d,
                VideoParams {
                    prompt: "x",
                    frames: Some(9),
                    seconds: Some(1.0),
                    ..Default::default()
                }
            )
            .is_err()
        );
        assert!(
            build_request(
                &d,
                VideoParams {
                    prompt: " ",
                    ..Default::default()
                }
            )
            .is_err()
        );
        assert!(
            build_request(
                &d,
                VideoParams {
                    prompt: "x",
                    size: Some("100x100"),
                    ..Default::default()
                }
            )
            .is_err()
        );
    }

    #[test]
    fn openai_seconds_may_be_a_string() {
        assert_eq!(seconds_of(&json!("4")), Some(4.0));
        assert_eq!(seconds_of(&json!(8)), Some(8.0));
        assert_eq!(seconds_of(&json!(true)), None);
    }

    #[test]
    fn a_range_request_gets_its_bytes() {
        let mut h = HeaderMap::new();
        h.insert(header::RANGE, "bytes=2-4".parse().unwrap());
        let r = mp4_response(b"0123456789", &h);
        assert_eq!(r.status(), StatusCode::PARTIAL_CONTENT);
        assert_eq!(r.headers()[header::CONTENT_RANGE], "bytes 2-4/10");
    }
}
