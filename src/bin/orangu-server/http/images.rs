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

//! `/v1/images/generations` — OpenAI's Images API, on a `qwen_image` model.
//!
//! The request is OpenAI's (`prompt`, `n`, `size`, `output_format`,
//! `response_format`) plus what a local diffusion model has knobs for and
//! OpenAI's schema has no word for: `steps`, `cfg_scale`, `negative_prompt`,
//! `seed`, and — in place of the separate multipart `/v1/images/edits` —
//! an `image` (a data URL or bare base64 of a PNG, JPEG, GIF, WebP or SVG) with a
//! `strength`, which starts the picture from that one. The answer is
//! OpenAI's `{"created", "data": [{"b64_json"}]}`; there is no `url` form,
//! since this server keeps nothing to link to.
//!
//! `stream: true` turns the response into server-sent events — one
//! `image_generation.progress` per step and an `image_generation.completed`
//! carrying the picture — because a 50-step picture is minutes on ordinary
//! hardware and a client with no progress has no way to tell working from
//! stuck.
//!
//! The chat-completions path for the same model lives in `http::openai`,
//! which turns a chat turn into one of these requests.

use axum::{
    Json,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
};
use base64::Engine as _;
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;

use super::AppState;
use crate::engine::image::{
    ImageDefaults, ImageEvent, ImageFormat, ImageRequest, InitImage, Pipeline, codec,
};

/// The largest picture, in bytes once decoded, an image server takes to
/// start from — on the API and in the web console alike.
pub const MAX_IMAGE_BYTES: usize = 50 * 1024 * 1024;

/// The request body cap for an image server: one [`MAX_IMAGE_BYTES`]
/// picture as base64 (4 characters per 3 bytes), plus room for the rest of
/// the JSON around it.
pub const IMAGE_BODY_LIMIT: usize = MAX_IMAGE_BYTES.div_ceil(3) * 4 + 1024 * 1024;

#[derive(Deserialize, Default)]
pub struct ImageGenerationRequest {
    #[serde(default)]
    prompt: String,
    #[serde(default)]
    negative_prompt: Option<String>,
    /// `WIDTHxHEIGHT`, or `auto` — the server default, or the attached
    /// picture's own proportions when there is one.
    #[serde(default)]
    size: Option<String>,
    #[serde(default)]
    n: Option<usize>,
    #[serde(default)]
    response_format: Option<String>,
    /// `png` (the default), `jpeg`, `gif` or `webp`.
    #[serde(default)]
    output_format: Option<String>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    steps: Option<usize>,
    #[serde(default)]
    cfg_scale: Option<f32>,
    /// A picture to start from: a `data:` URL or bare base64.
    #[serde(default)]
    image: Option<String>,
    #[serde(default)]
    strength: Option<f32>,
    #[serde(default)]
    stream: bool,
}

/// The `image` object of `/props` (and the console's `/api/props`): the
/// companions and the adapter, the transformer's shape, the request
/// defaults as they stand and as configured, and the measured rate — what
/// a client needs to know how a picture will come out, and how long it
/// will take, before asking for one.
pub fn props_json(pipeline: &Pipeline) -> serde_json::Value {
    let tc = pipeline.transformer_config();
    let defaults_json = |d: &ImageDefaults| {
        json!({
            "size": format!("{}x{}", d.width, d.height),
            "steps": d.steps,
            "cfg_scale": d.cfg_scale,
            "negative_prompt": d.negative_prompt,
            "strength": d.strength,
            "format": d.format.name(),
        })
    };
    json!({
        // `qwen_image` or `qwen_image_2_1`, and the pixels a side must be a
        // multiple of for it.
        "architecture": pipeline.variant.architecture(),
        "size_unit": pipeline.size_unit(),
        "text_encoder": pipeline.companions.text_encoder.display().to_string(),
        "vae": pipeline.companions.vae.display().to_string(),
        // The projector an attached picture is edited with — `null` when an
        // attachment is a starting point instead.
        "vision": pipeline
            .edits()
            .then(|| pipeline.companions.vision.as_ref().map(|p| p.display().to_string()))
            .flatten(),
        // The adapter in the weights, with the step count its name says it
        // was distilled for — `null` for the base model.
        "lora": pipeline.adapter.as_ref().map(|path| json!({
            "path": path.display().to_string(),
            "steps": path.file_name().and_then(|n| n.to_str()).and_then(orangu::model_spec::lightning_steps),
        })),
        "n_layer": tc.n_layer,
        "n_head": tc.n_head,
        "head_dim": tc.head_dim,
        "dim": tc.dim,
        "defaults": defaults_json(&pipeline.defaults()),
        "configured": defaults_json(&pipeline.configured),
        // The last measured rate and what the defaults cost at it, for a
        // client that wants to say how long before it sends; `rate` is the
        // model behind both, so a client can cost any settings it is about
        // to choose: seconds = encode + steps × step_share × n × passes × (linear +
        // attention × n / attention_tokens) + decode_per_pixel × pixels,
        // with n the latent tokens (width/16 × height/16) and passes 2 under
        // guidance, 1 without. An edit attends over n + r keys (r the
        // reference's tokens) and encodes in edit_encode_per_token × r.
        "token_passes_per_second": pipeline.token_passes_per_second(),
        "estimated_default_seconds": pipeline.estimated_default_seconds(),
        "rate": pipeline.rate_model().map(|m| json!({
            "encode": m.encode,
            "linear_per_token_pass": m.linear_per_token_pass,
            "attention_per_token_pass": m.attention_per_token_pass,
            "attention_tokens": m.attention_tokens,
            "decode_per_pixel": m.decode_per_pixel,
            "edit_encode_per_token": m.edit_encode_per_token,
            "step_share": m.step_share(),
        })),
    })
}

/// `POST /props`' `image` object: the picture defaults to change. Every
/// key is optional and the rest keep their value; `reset` puts all of
/// them back to the configuration first.
#[derive(Deserialize, Default, Debug)]
pub struct ImageSettings {
    #[serde(default)]
    pub reset: bool,
    #[serde(default)]
    pub size: Option<String>,
    #[serde(default)]
    pub steps: Option<usize>,
    #[serde(default)]
    pub cfg_scale: Option<f32>,
    #[serde(default)]
    pub negative_prompt: Option<String>,
    #[serde(default)]
    pub strength: Option<f32>,
    #[serde(default)]
    pub format: Option<String>,
}

/// Applies `settings` to the pipeline's defaults, checked as the config
/// loader checks the same keys — a size that is not multiples of the
/// model's unit (16, or 32 for Qwen-Image 2.1), zero
/// steps, a negative guidance, a strength outside 0–1 are refused whole,
/// so the defaults never hold a value a request could not be built from.
pub fn apply_settings(pipeline: &Pipeline, settings: &ImageSettings) -> Result<(), String> {
    let mut d = if settings.reset {
        pipeline.configured.clone()
    } else {
        pipeline.defaults()
    };
    if let Some(size) = settings.size.as_deref().map(str::trim) {
        let unit = pipeline.size_unit();
        let (width, height) = crate::config::parse_image_size(size)
            .filter(|(w, h)| w.is_multiple_of(unit) && h.is_multiple_of(unit))
            .ok_or_else(|| format!("size '{size}' is not WIDTHxHEIGHT in multiples of {unit}"))?;
        d.width = width;
        d.height = height;
    }
    if let Some(steps) = settings.steps {
        if steps == 0 {
            return Err("steps must be at least 1".to_string());
        }
        d.steps = steps;
    }
    if let Some(cfg_scale) = settings.cfg_scale {
        if !cfg_scale.is_finite() || cfg_scale < 0.0 {
            return Err("cfg_scale must be a number of 0 or more".to_string());
        }
        d.cfg_scale = cfg_scale;
    }
    if let Some(prompt) = &settings.negative_prompt {
        d.negative_prompt = prompt.clone();
    }
    if let Some(strength) = settings.strength {
        if !(0.0..=1.0).contains(&strength) {
            return Err("strength must be between 0 and 1".to_string());
        }
        d.strength = strength;
    }
    if let Some(name) = settings.format.as_deref() {
        d.format = ImageFormat::parse(name)
            .ok_or_else(|| format!("format '{name}' is not png, jpeg, gif, webp or svg"))?;
    }
    pipeline.set_defaults(d);
    Ok(())
}

/// The knobs a chat turn and an images request share, resolved against the
/// server's defaults into one [`ImageRequest`].
pub struct ImageParams<'a> {
    pub prompt: &'a str,
    pub negative_prompt: Option<&'a str>,
    pub size: Option<&'a str>,
    pub output_format: Option<&'a str>,
    pub seed: Option<u64>,
    pub steps: Option<usize>,
    pub cfg_scale: Option<f32>,
    /// A picture to start from, already decoded from its transport.
    pub init: Option<(String, Vec<u8>)>,
    pub strength: Option<f32>,
}

/// One data URL (`data:image/png;base64,...`) or bare base64 string to its
/// MIME type (empty when bare) and bytes.
pub fn decode_image_payload(payload: &str) -> Result<(String, Vec<u8>), String> {
    let payload = payload.trim();
    let (mime, data) = match payload.strip_prefix("data:") {
        Some(rest) => {
            let (header, data) = rest
                .split_once(',')
                .ok_or_else(|| "malformed data URL: no comma".to_string())?;
            let mime = header.split(';').next().unwrap_or("").to_string();
            if !header.contains(";base64") {
                return Err("only base64 data URLs are accepted".to_string());
            }
            (mime, data)
        }
        None => {
            if payload.starts_with("http://") || payload.starts_with("https://") {
                return Err("image URLs are not fetched; send the picture as a data URL".into());
            }
            (String::new(), payload)
        }
    };
    let data = data.trim();
    // Refused before decoding: the base64 length bounds the bytes it holds.
    if data.len() / 4 * 3 > MAX_IMAGE_BYTES + 2 {
        return Err(format!(
            "image is larger than the {} MB this server takes",
            MAX_IMAGE_BYTES / (1024 * 1024)
        ));
    }
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(data)
        .map_err(|err| format!("image is not valid base64: {err}"))?;
    if bytes.len() > MAX_IMAGE_BYTES {
        return Err(format!(
            "image is larger than the {} MB this server takes",
            MAX_IMAGE_BYTES / (1024 * 1024)
        ));
    }
    Ok((mime, bytes))
}

/// Resolves the parameters into a request, or the reason they cannot be.
/// `unit` is the pixels a side must be a multiple of — the pipeline's
/// [`Pipeline::size_unit`].
pub fn build_request(
    defaults: &ImageDefaults,
    unit: usize,
    params: ImageParams<'_>,
) -> Result<ImageRequest, String> {
    if params.prompt.trim().is_empty() {
        return Err("prompt is required".to_string());
    }
    let format = match params.output_format {
        Some(name) => ImageFormat::parse(name)
            .ok_or_else(|| format!("output_format '{name}' is not png, jpeg, gif or webp"))?,
        None => params
            .init
            .as_ref()
            .and_then(|(mime, _)| ImageFormat::from_mime(mime))
            .unwrap_or(defaults.format),
    };
    let init = match &params.init {
        Some((_, bytes)) => Some(InitImage {
            bytes: bytes.clone(),
            strength: params.strength.unwrap_or(defaults.strength),
        }),
        None => None,
    };
    if let Some(strength) = params.strength
        && !(0.0..=1.0).contains(&strength)
    {
        return Err("strength must be between 0 and 1".to_string());
    }
    let (width, height) = match params.size.map(str::trim) {
        Some(size) if !size.is_empty() && !size.eq_ignore_ascii_case("auto") => {
            crate::config::parse_image_size(size)
                .filter(|(w, h)| w.is_multiple_of(unit) && h.is_multiple_of(unit))
                .ok_or_else(|| {
                    format!("size '{size}' is not WIDTHxHEIGHT in multiples of {unit}")
                })?
        }
        _ => match &init {
            // A picture to start from sets the proportions; the server's
            // configured size bounds the longer side.
            Some(init) => codec::dimensions(&init.bytes)
                .map(|dims| {
                    codec::fit_generation_size(dims, defaults.width.max(defaults.height), unit)
                })
                .map_err(|err| format!("{err:#}"))?,
            None => (defaults.width, defaults.height),
        },
    };
    let steps = params.steps.unwrap_or(defaults.steps);
    if steps == 0 {
        return Err("steps must be at least 1".to_string());
    }
    let cfg_scale = params.cfg_scale.unwrap_or(defaults.cfg_scale);
    if !cfg_scale.is_finite() || cfg_scale < 0.0 {
        return Err("cfg_scale must be a non-negative number".to_string());
    }
    Ok(ImageRequest {
        prompt: params.prompt.to_string(),
        negative_prompt: params.negative_prompt.map(str::to_string),
        width,
        height,
        steps,
        cfg_scale,
        seed: params.seed,
        format,
        init,
    })
}

/// A generation's phase timings as the reply carries them, in
/// milliseconds: `encode` (the prompts through the text encoder), `steps`
/// (the denoising loop) and `decode` (the VAE and the file format). They add
/// up to `generation_ms` less the little between phases. `stages` splits the
/// loop's transformer passes by operation class (`engine::image::
/// transformer::Stages`).
fn timings_json(timings: &crate::engine::image::Timings) -> serde_json::Value {
    let st = &timings.stages;
    json!({
        "encode_ms": timings.encode.as_millis() as u64,
        "steps_ms": timings.denoise.as_millis() as u64,
        "decode_ms": timings.decode.as_millis() as u64,
        "stages": {
            "passes": st.passes,
            "modulation_ms": st.modulation.as_millis() as u64,
            "qkv_ms": st.qkv.as_millis() as u64,
            "attention_ms": st.attention.as_millis() as u64,
            "out_ms": st.out.as_millis() as u64,
            "mlp_ms": st.mlp.as_millis() as u64,
            "other_ms": st.other.as_millis() as u64,
            "lora_ms": st.lora.as_millis() as u64,
        },
    })
}

/// The pipeline, or the response that says this server has none.
pub fn pipeline_or_reject(
    state: &AppState,
) -> Result<Arc<Pipeline>, Box<axum::response::Response>> {
    match &state.engine.image {
        Some(pipeline) => Ok(pipeline.clone()),
        None => Err(Box::new(
            (
                StatusCode::NOT_IMPLEMENTED,
                "this server is not serving an image model (a qwen_image GGUF); \
                 /v1/images/generations needs one",
            )
                .into_response(),
        )),
    }
}

/// Markdown for a finished picture, as a data URL the console and any
/// markdown-rendering chat client show inline.
pub fn markdown_image(image: &crate::engine::image::GeneratedImage, alt: &str) -> String {
    let alt = alt.replace(['[', ']', '\n'], " ");
    format!(
        "![{alt}](data:{};base64,{})",
        image.format.mime(),
        base64::engine::general_purpose::STANDARD.encode(&image.bytes)
    )
}

pub async fn generations(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<ImageGenerationRequest>,
) -> axum::response::Response {
    let _ = super::openai::request_role(&state, &headers);
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
    let pipeline = match pipeline_or_reject(&state) {
        Ok(pipeline) => pipeline,
        Err(response) => return *response,
    };
    if let Some(format) = req.response_format.as_deref()
        && format != "b64_json"
    {
        return (
            StatusCode::BAD_REQUEST,
            "response_format must be b64_json; this server keeps no files to return a url to",
        )
            .into_response();
    }
    let n = req.n.unwrap_or(1);
    if n == 0 || n > 10 {
        return (StatusCode::BAD_REQUEST, "n must be between 1 and 10").into_response();
    }
    let init = match req.image.as_deref() {
        Some(payload) => match decode_image_payload(payload) {
            Ok(init) => Some(init),
            Err(err) => return (StatusCode::BAD_REQUEST, err).into_response(),
        },
        None => None,
    };
    let request = match build_request(
        &pipeline.defaults(),
        pipeline.size_unit(),
        ImageParams {
            prompt: &req.prompt,
            negative_prompt: req.negative_prompt.as_deref(),
            size: req.size.as_deref(),
            output_format: req.output_format.as_deref(),
            seed: req.seed,
            steps: req.steps,
            cfg_scale: req.cfg_scale,
            init,
            strength: req.strength,
        },
    ) {
        Ok(request) => request,
        Err(err) => return (StatusCode::BAD_REQUEST, err).into_response(),
    };
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    if !req.stream {
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            let mut one = request.clone();
            // Each picture of a set gets its own seed, derived from the
            // first so the set is reproducible from one number.
            one.seed = request.seed.map(|seed| seed.wrapping_add(i as u64));
            let (mut rx, _cancel) = pipeline.spawn(one);
            let image = loop {
                match rx.recv().await {
                    Some(ImageEvent::Progress(_)) => {}
                    Some(ImageEvent::Done(image)) => break image,
                    Some(ImageEvent::Error(err)) => {
                        return (StatusCode::INTERNAL_SERVER_ERROR, err).into_response();
                    }
                    None => {
                        return (StatusCode::INTERNAL_SERVER_ERROR, "generation ended early")
                            .into_response();
                    }
                }
            };
            data.push(json!({
                "b64_json": base64::engine::general_purpose::STANDARD.encode(&image.bytes),
                "output_format": image.format.extension(),
                "size": format!("{}x{}", image.width, image.height),
                "seed": image.seed,
                "steps": image.steps,
                "generation_ms": image.elapsed.as_millis() as u64,
                "timings": timings_json(&image.timings),
            }));
        }
        return Json(json!({
            "created": created,
            "model": state.model_label,
            "output_format": request.format.extension(),
            "size": format!("{}x{}", request.width, request.height),
            "data": data,
        }))
        .into_response();
    }

    // Streaming: one picture per request (`n` is ignored past the first),
    // progress per step, then the picture.
    let (mut rx, cancel) = pipeline.spawn(request);
    let model = state.model_label.clone();
    let stream = async_stream::stream! {
        // Held for the stream's life, so a client that goes away stops the
        // work — see `CancelOnDrop`.
        let _cancel = cancel;
        loop {
            let Some(event) = rx.recv().await else { break };
            match event {
                ImageEvent::Progress(p) => {
                    let eta = p.eta_seconds();
                    let chunk = json!({
                        "type": "image_generation.progress",
                        "created": created, "model": model,
                        "step": p.step, "steps": p.steps,
                        "seconds_per_step": p.seconds_per_step,
                        "eta_seconds": eta,
                    });
                    yield Ok::<_, std::convert::Infallible>(axum::response::sse::Event::default().data(chunk.to_string()));
                }
                ImageEvent::Done(image) => {
                    let chunk = json!({
                        "type": "image_generation.completed",
                        "created": created, "model": model,
                        "b64_json": base64::engine::general_purpose::STANDARD.encode(&image.bytes),
                        "output_format": image.format.extension(),
                        "size": format!("{}x{}", image.width, image.height),
                        "seed": image.seed,
                        "steps": image.steps,
                        "generation_ms": image.elapsed.as_millis() as u64,
                        "timings": timings_json(&image.timings),
                    });
                    yield Ok(axum::response::sse::Event::default().data(chunk.to_string()));
                    yield Ok(axum::response::sse::Event::default().data("[DONE]"));
                    break;
                }
                ImageEvent::Error(err) => {
                    yield Ok(axum::response::sse::Event::default().data(json!({"error": err}).to_string()));
                    break;
                }
            }
        }
    };
    axum::response::sse::Sse::new(stream).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_data_url_and_bare_base64_both_decode() {
        let (mime, bytes) = decode_image_payload("data:image/png;base64,AQID").unwrap();
        assert_eq!(mime, "image/png");
        assert_eq!(bytes, vec![1, 2, 3]);
        let (mime, bytes) = decode_image_payload("AQID").unwrap();
        assert_eq!(mime, "");
        assert_eq!(bytes, vec![1, 2, 3]);
        assert!(decode_image_payload("https://example.com/a.png").is_err());
        assert!(decode_image_payload("data:image/png,raw").is_err());
    }

    #[test]
    fn a_picture_up_to_the_cap_decodes_and_one_past_it_does_not() {
        let engine = base64::engine::general_purpose::STANDARD;
        let at_cap = engine.encode(vec![0u8; MAX_IMAGE_BYTES]);
        assert!(
            at_cap.len() <= IMAGE_BODY_LIMIT,
            "the body cap fits the picture cap"
        );
        assert_eq!(
            decode_image_payload(&at_cap).unwrap().1.len(),
            MAX_IMAGE_BYTES
        );
        let past_cap = engine.encode(vec![0u8; MAX_IMAGE_BYTES + 1]);
        assert!(
            decode_image_payload(&past_cap)
                .unwrap_err()
                .contains("50 MB")
        );
    }

    fn params<'a>(prompt: &'a str) -> ImageParams<'a> {
        ImageParams {
            prompt,
            negative_prompt: None,
            size: None,
            output_format: None,
            seed: None,
            steps: None,
            cfg_scale: None,
            init: None,
            strength: None,
        }
    }

    #[test]
    fn defaults_fill_what_the_request_leaves_out() {
        let defaults = ImageDefaults {
            width: 512,
            height: 256,
            steps: 8,
            cfg_scale: 2.5,
            negative_prompt: "blurry".into(),
            strength: 0.7,
            format: ImageFormat::Webp,
        };
        let request = build_request(&defaults, 16, params("a cat")).unwrap();
        assert_eq!((request.width, request.height), (512, 256));
        assert_eq!(request.steps, 8);
        assert_eq!(request.cfg_scale, 2.5);
        // The configured format, absent a request's own or an attachment's.
        assert_eq!(request.format, ImageFormat::Webp);
        assert!(request.negative_prompt.is_none());
        assert!(request.init.is_none());

        let mut p = params("a cat");
        p.size = Some("768x512");
        p.steps = Some(4);
        p.output_format = Some("jpeg");
        let request = build_request(&defaults, 16, p).unwrap();
        assert_eq!((request.width, request.height), (768, 512));
        assert_eq!(request.steps, 4);
        assert_eq!(request.format, ImageFormat::Jpeg);
    }

    #[test]
    fn bad_parameters_are_named() {
        let defaults = ImageDefaults::default();
        assert!(build_request(&defaults, 16, params("  ")).is_err());
        let mut p = params("x");
        p.size = Some("100x100");
        assert!(
            build_request(&defaults, 16, p)
                .unwrap_err()
                .contains("size")
        );
        let mut p = params("x");
        p.steps = Some(0);
        assert!(
            build_request(&defaults, 16, p)
                .unwrap_err()
                .contains("steps")
        );
        let mut p = params("x");
        p.output_format = Some("bmp");
        assert!(
            build_request(&defaults, 16, p)
                .unwrap_err()
                .contains("output_format")
        );
    }

    /// An attached JPEG comes back as JPEG, at its own size snapped to the
    /// 16-pixel unit — never scaled up — and bounded by the configured size
    /// when it is larger than that.
    #[test]
    fn an_attached_picture_sets_the_format_and_proportions() {
        let defaults = ImageDefaults {
            width: 512,
            height: 512,
            ..ImageDefaults::default()
        };
        let feature =
            crate::engine::image::vae::Feature::new(200, 400, 3, vec![0.0; 200 * 400 * 3]);
        let jpeg = codec::encode(&feature, ImageFormat::Jpeg).unwrap();
        let mut p = params("brighter");
        p.init = Some(("image/jpeg".into(), jpeg));
        let request = build_request(&defaults, 16, p).unwrap();
        assert_eq!(request.format, ImageFormat::Jpeg);
        assert_eq!((request.width, request.height), (400, 208));
        assert_eq!(request.init.as_ref().unwrap().strength, defaults.strength);

        let feature =
            crate::engine::image::vae::Feature::new(600, 1200, 3, vec![0.0; 600 * 1200 * 3]);
        let png = codec::encode(&feature, ImageFormat::Png).unwrap();
        let mut p = params("brighter");
        p.init = Some(("image/png".into(), png));
        let request = build_request(&defaults, 16, p).unwrap();
        assert_eq!(request.format, ImageFormat::Png);
        assert_eq!((request.width, request.height), (512, 256));
    }

    #[test]
    fn markdown_images_escape_the_alt_text() {
        let image = crate::engine::image::GeneratedImage {
            bytes: vec![1, 2, 3],
            format: ImageFormat::Png,
            width: 16,
            height: 16,
            seed: 1,
            steps: 1,
            elapsed: std::time::Duration::ZERO,
            timings: Default::default(),
        };
        assert_eq!(
            markdown_image(&image, "a [cat]\nhere"),
            "![a  cat  here](data:image/png;base64,AQID)"
        );
    }
}
