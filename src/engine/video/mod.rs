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

//! Video generation: LTX-2.5 (`ltxv`), a prompt to a clip with its sound.
//!
//! The pipeline is Lightricks' distilled two-stage recipe
//! (`ltx_pipelines.distilled`), run on the engine's own kernels:
//!
//! 1. **The prompt** through LTX's Gemma 4 text encoder, its projection
//!    heads and the two connectors ([`text`]) — once per clip.
//! 2. **Stage 1**: the clip's latent at half its width and height, and the
//!    sound's latent, drawn from noise in eight steps of the audio-video
//!    transformer ([`transformer`]) on the fixed distilled schedule, with
//!    the ancestral Euler sampler LTX-2.5 is made for.
//! 3. **Stage 2**, when the latent upscaler is there ([`upscaler`]): the
//!    video latent doubled in width and height, both latents noised back to
//!    σ = 0.909375, and refined in three more steps at the full size.
//!    Without the upscaler the clip is drawn at its full size in stage 1.
//! 4. **The decoders**: the video VAE ([`vae`]) to frames, the audio VAE and
//!    vocoder ([`audio`]) to a stereo waveform.
//! 5. **The file**: AV1 and FLAC in an MP4 ([`mp4`], [`flac`]).
//!
//! The transformer's GGUF (`unsloth/LTX-2.5-GGUF`) carries the transformer
//! and the connectors. The rest are its *companions*, found under the models
//! directory by their contents ([`Companions`]).

pub mod audio;
pub mod convert;
pub mod flac;
pub mod mp4;
pub mod ops;
pub mod text;
pub mod transformer;
pub mod upscaler;
pub mod vae;

use crate::engine::arch::ModelForward;
use crate::engine::backend::Backend;
use crate::engine::loader::LoadedModel;
use crate::engine::tokenizer::Tokenizer;
use anyhow::{Context, Result, anyhow, bail, ensure};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use transformer::{Geometry, LtxTransformer, StreamInput};

/// The distilled schedule's sigmas for stage 1 (eight steps).
pub const STAGE_1_SIGMAS: [f32; 9] = [
    1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0,
];
/// The distilled schedule's sigmas for stage 2 (three steps).
pub const STAGE_2_SIGMAS: [f32; 4] = [0.909375, 0.725, 0.421875, 0.0];

/// Pixel frames per latent frame, and pixels per latent pixel.
pub const TIME_SCALE: usize = 8;
pub const SPACE_SCALE: usize = 32;
/// Audio latent frames per second: 16 kHz, a hop of 160, four mel frames a
/// latent.
pub const AUDIO_LATENTS_PER_SECOND: f64 = 16000.0 / 160.0 / 4.0;

/// The longest clip a request or the configuration may ask for.
pub const MAX_SECONDS: f64 = 60.0;

/// Seed offsets of the ancestral sampler's noise, per stage, as the
/// reference draws them.
const STAGE_1_NOISE_OFFSET: u64 = 10000;
const STAGE_2_NOISE_OFFSET: u64 = 20000;

/// Where `orangu-server download` fetches the companions from.
pub const TEXT_ENCODER_FILE: &str = crate::model_download::LTX_TEXT_ENCODER_FILE;
pub const COMPANION_REPO: &str = crate::model_download::LTX_COMPANION_REPO;
pub const VIDEO_VAE_FILE: &str = crate::model_download::LTX_VIDEO_VAE_FILE;

/// Whether the request's own flag, or the server's, says stop — the same
/// shutdown flag a picture in flight watches.
pub fn cancelled(cancel: Option<&AtomicBool>) -> bool {
    crate::engine::image::cancelled(cancel)
}

/// The files an `ltxv` model generates with.
#[derive(Debug, Clone)]
pub struct Companions {
    /// LTX's Gemma 4 with its projection heads (a GGUF).
    pub text_encoder: PathBuf,
    /// The convolutional video VAE.
    pub vae: PathBuf,
    /// The audio VAE and vocoder; without it a clip is silent.
    pub audio_vae: Option<PathBuf>,
    /// The ×2 latent upscaler; without it a clip is drawn in one stage.
    pub upscaler: Option<PathBuf>,
}

fn file_spec(models_dir: &Path, key: &str, spec: &str) -> Result<PathBuf> {
    let path = PathBuf::from(spec);
    let path = if path.is_absolute() {
        path
    } else {
        models_dir.join(path)
    };
    ensure!(path.is_file(), "{key} {} is not a file", path.display());
    Ok(path)
}

/// The GGUF of the `safetensors` text encoder at `source`: converted now
/// when there is none yet (see [`convert`]), a few minutes once.
fn gguf_text_encoder(models_dir: &Path, source: &Path) -> Result<PathBuf> {
    ensure!(
        crate::model_spec::is_ltx_text_encoder_safetensors(source),
        "{} is not LTX's text encoder (a Gemma 4 with its projection heads)",
        source.display()
    );
    let target = convert::converted_path(models_dir, source);
    if crate::model_spec::is_ltx_text_encoder(&target) {
        return Ok(target);
    }
    log::info!(
        "orangu-server: converting the ltxv text encoder {} to {} (Q6_K) — once, a few minutes",
        source.display(),
        target.display()
    );
    let started = Instant::now();
    let last = std::sync::atomic::AtomicUsize::new(0);
    convert::convert(source, &target, &|done, total| {
        let pct = done * 100 / total.max(1);
        if pct >= last.load(Ordering::Relaxed) + 10 {
            last.store(pct, Ordering::Relaxed);
            log::info!("orangu-server: converting the ltxv text encoder: {pct}%");
        }
    })
    .with_context(|| format!("converting {}", source.display()))?;
    log::info!(
        "orangu-server: converted the ltxv text encoder in {:.0} s",
        started.elapsed().as_secs_f64()
    );
    Ok(target)
}

impl Companions {
    /// The configured file for each companion when there is one, otherwise
    /// the models directory searched for it by its contents. `none` turns
    /// the optional two off.
    pub fn locate(
        models_dir: &Path,
        text_encoder: Option<&str>,
        vae: Option<&str>,
        audio_vae: Option<&str>,
        upscaler: Option<&str>,
    ) -> Result<Self> {
        let text_encoder = match text_encoder {
            Some(spec) if spec.ends_with(".safetensors") => {
                gguf_text_encoder(models_dir, &file_spec(models_dir, "text_encoder", spec)?)?
            }
            Some(spec) => {
                crate::model_spec::resolve_load_target(models_dir, spec)
                    .with_context(|| format!("resolving text_encoder '{spec}'"))?
                    .0
            }
            None => match crate::model_spec::find_ltx_text_encoder(models_dir) {
                Some(path) => path,
                None => match crate::model_spec::find_ltx_text_encoder_safetensors(models_dir) {
                    Some(source) => gguf_text_encoder(models_dir, &source)?,
                    None => bail!(
                        "no LTX text encoder (Gemma 4 12B with LTX's projection heads) found \
                         under {}. An ltxv model is conditioned on every layer of it; download \
                         {COMPANION_REPO}'s {TEXT_ENCODER_FILE} (converted to a GGUF at the \
                         first start) or point [orangu-server].text_encoder at it",
                        models_dir.display()
                    ),
                },
            },
        };
        let vae = match vae {
            Some(spec) => {
                let path = file_spec(models_dir, "vae", spec)?;
                ensure!(
                    crate::model_spec::is_ltx_video_vae(&path),
                    "vae {} is not LTX-2's convolutional video VAE",
                    path.display()
                );
                path
            }
            None => crate::model_spec::find_ltx_video_vae(models_dir).ok_or_else(|| {
                anyhow!(
                    "no LTX video VAE found under {}. It turns the model's latents into frames; \
                     download {COMPANION_REPO}'s {VIDEO_VAE_FILE}, or point [orangu-server].vae at it",
                    models_dir.display()
                )
            })?,
        };
        let audio_vae = match audio_vae {
            Some("none") => None,
            Some(spec) => Some(file_spec(models_dir, "audio_vae", spec)?),
            None => crate::model_spec::find_ltx_audio_vae(models_dir),
        };
        let upscaler = match upscaler {
            Some("none") => None,
            Some(spec) => Some(file_spec(models_dir, "upscaler", spec)?),
            None => crate::model_spec::find_ltx_spatial_upscaler(models_dir),
        };
        Ok(Self {
            text_encoder,
            vae,
            audio_vae,
            upscaler,
        })
    }
}

/// What a clip gets when the request does not say: the `video_*` keys.
#[derive(Debug, Clone, PartialEq)]
pub struct VideoDefaults {
    pub width: usize,
    pub height: usize,
    /// Pixel frames: `8 k + 1`.
    pub frames: usize,
    pub fps: u32,
    /// Sound, when the audio VAE is there.
    pub audio: bool,
    /// The two-stage recipe, when the upscaler is there.
    pub two_stage: bool,
    /// The AV1 encoder's quantizer, 0 (best) to 255.
    pub quality: usize,
}

impl Default for VideoDefaults {
    fn default() -> Self {
        Self {
            width: 768,
            height: 448,
            frames: 121,
            fps: 24,
            audio: true,
            two_stage: true,
            quality: 80,
        }
    }
}

/// The frame count nearest `frames` that the latent grid holds: `8 k + 1`.
pub fn snap_frames(frames: usize) -> usize {
    let k = (frames.max(1) - 1 + TIME_SCALE / 2) / TIME_SCALE;
    k * TIME_SCALE + 1
}

/// One clip to generate.
#[derive(Debug, Clone)]
pub struct VideoRequest {
    pub prompt: String,
    pub width: usize,
    pub height: usize,
    pub frames: usize,
    pub fps: u32,
    pub seed: Option<u64>,
    pub audio: bool,
    pub two_stage: bool,
    pub quality: usize,
}

impl VideoRequest {
    pub fn from_defaults(prompt: &str, d: &VideoDefaults) -> Self {
        Self {
            prompt: prompt.to_string(),
            width: d.width,
            height: d.height,
            frames: d.frames,
            fps: d.fps,
            seed: None,
            audio: d.audio,
            two_stage: d.two_stage,
            quality: d.quality,
        }
    }
}

/// Where the time of a clip went.
#[derive(Debug, Clone, Default)]
pub struct Timings {
    pub encode: Duration,
    pub stage_1: Duration,
    pub upscale: Duration,
    pub stage_2: Duration,
    pub decode: Duration,
    pub audio: Duration,
    pub mux: Duration,
}

/// A finished clip.
#[derive(Debug, Clone)]
pub struct GeneratedVideo {
    /// The MP4.
    pub bytes: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub frames: usize,
    pub fps: u32,
    pub seed: u64,
    pub has_audio: bool,
    pub elapsed: Duration,
    pub timings: Timings,
}

impl GeneratedVideo {
    pub fn seconds(&self) -> f64 {
        self.frames as f64 / self.fps.max(1) as f64
    }
}

/// Where a generation is, for a progress bar: `step` of `steps`
/// transformer steps across both stages; `step = 0` is the announcement,
/// its `pending_seconds` the estimate of everything.
#[derive(Debug, Clone, Copy)]
pub struct Progress {
    pub step: usize,
    pub steps: usize,
    /// What is running: `encoding`, `stage 1`, `upscaling`, `stage 2`,
    /// `decoding`, `sound`, `encoding the file`.
    pub phase: &'static str,
    /// Seconds still ahead, estimated.
    pub eta_seconds: f64,
}

/// What the pipeline has learned about its own speed.
///
/// A step's cost is a part that grows with the video tokens, a part that
/// does not (the prompt's keys and values, recomputed every step since they
/// are modulated by σ; the sound), and attention, which grows with the
/// square of the tokens. The decoders cost per output pixel and per second
/// of sound. Seeded by a startup calibration, replaced by every finished
/// clip's measurements.
#[derive(Debug, Clone, Copy)]
pub struct RateModel {
    /// Seconds per video-token step.
    pub per_token: f64,
    /// Seconds per step that do not depend on the clip.
    pub per_step: f64,
    /// Seconds per step per token², for attention.
    pub per_token_sq: f64,
    pub encode: f64,
    /// Seconds per output pixel per frame.
    pub decode_per_pixel: f64,
    /// Seconds per second of sound.
    pub audio_per_second: f64,
    /// Seconds per output pixel per frame, for the upscaler and the file.
    pub other_per_pixel: f64,
}

impl RateModel {
    /// From the calibrated cost of one video token through one block.
    pub fn from_calibration(per_token_block: f64, blocks: usize) -> Self {
        let per_token = per_token_block * blocks as f64;
        Self {
            per_token,
            // The prompt's 1024 keys and values: about 160 tokens' worth.
            per_step: 160.0 * per_token,
            // 2 · n · dim multiply-adds of attention against a token's
            // ~268 M in the linears, at the same rate.
            per_token_sq: per_token * 2.0 * 4096.0 / 268e6,
            encode: 20.0,
            decode_per_pixel: 2e-6,
            audio_per_second: 2.0,
            other_per_pixel: 0.3e-6,
        }
    }

    pub fn seconds_per_step(&self, tokens: usize) -> f64 {
        let n = tokens as f64;
        self.per_step + self.per_token * n + self.per_token_sq * n * n
    }
}

/// A clip's shape at both stages.
#[derive(Debug, Clone, Copy)]
pub struct Plan {
    pub width: usize,
    pub height: usize,
    pub frames: usize,
    pub latent_frames: usize,
    pub audio_latents: usize,
    pub fps: u32,
    pub two_stage: bool,
    /// Stage 1's latent rows and columns.
    pub stage_1: (usize, usize),
}

impl Plan {
    pub fn new(r: &VideoRequest, upscaler: bool) -> Result<Self> {
        let two_stage = r.two_stage && upscaler;
        let unit = if two_stage {
            2 * SPACE_SCALE
        } else {
            SPACE_SCALE
        };
        ensure!(
            r.width >= unit
                && r.height >= unit
                && r.width.is_multiple_of(unit)
                && r.height.is_multiple_of(unit),
            "a clip's width and height must be multiples of {unit}{}; {}x{} is not",
            if two_stage {
                " (two-stage: it is drawn at half the size first)"
            } else {
                ""
            },
            r.width,
            r.height
        );
        ensure!(r.fps >= 1 && r.fps <= 60, "fps must be between 1 and 60");
        let frames = snap_frames(r.frames);
        ensure!(
            frames as f64 / r.fps as f64 <= MAX_SECONDS,
            "a clip is at most {MAX_SECONDS} seconds; {frames} frames at {} fps is longer",
            r.fps
        );
        let latent_frames = (frames - 1) / TIME_SCALE + 1;
        let seconds = frames as f64 / r.fps as f64;
        let audio_latents = (seconds * AUDIO_LATENTS_PER_SECOND).round() as usize;
        let scale = if two_stage { 2 } else { 1 };
        Ok(Self {
            width: r.width,
            height: r.height,
            frames,
            latent_frames,
            audio_latents: audio_latents.max(1),
            fps: r.fps,
            two_stage,
            stage_1: (
                r.height / SPACE_SCALE / scale,
                r.width / SPACE_SCALE / scale,
            ),
        })
    }

    /// Video tokens at each stage.
    pub fn tokens(&self) -> (usize, Option<usize>) {
        let (h, w) = self.stage_1;
        let one = self.latent_frames * h * w;
        (one, self.two_stage.then_some(one * 4))
    }

    pub fn steps(&self) -> usize {
        STAGE_1_SIGMAS.len() - 1
            + if self.two_stage {
                STAGE_2_SIGMAS.len() - 1
            } else {
                0
            }
    }

    /// Seconds the whole clip will take, by `rate`.
    pub fn estimate(&self, rate: &RateModel, audio: bool) -> f64 {
        let (one, two) = self.tokens();
        let mut s = rate.encode + rate.seconds_per_step(one) * (STAGE_1_SIGMAS.len() - 1) as f64;
        if let Some(two) = two {
            s += rate.seconds_per_step(two) * (STAGE_2_SIGMAS.len() - 1) as f64;
        }
        let pixels = (self.width * self.height * self.frames) as f64;
        s += pixels * (rate.decode_per_pixel + rate.other_per_pixel);
        if audio {
            s += rate.audio_per_second * self.frames as f64 / self.fps.max(1) as f64;
        }
        s
    }
}

/// The midpoint positions of every video token — `(seconds, row pixel,
/// column pixel)` — with the causal VAE's first frame covering one pixel
/// frame (`causal_fix`).
pub fn video_positions(latent_frames: usize, rows: usize, cols: usize, fps: f64) -> Vec<f32> {
    let time = |i: usize| ((i * TIME_SCALE) as f64 + 1.0 - TIME_SCALE as f64).max(0.0);
    let mut out = Vec::with_capacity(latent_frames * rows * cols * 3);
    for f in 0..latent_frames {
        let t = ((time(f) + time(f + 1)) / 2.0 / fps) as f32;
        for y in 0..rows {
            for x in 0..cols {
                out.push(t);
                out.push((y * SPACE_SCALE) as f32 + SPACE_SCALE as f32 / 2.0);
                out.push((x * SPACE_SCALE) as f32 + SPACE_SCALE as f32 / 2.0);
            }
        }
    }
    out
}

/// The midpoint, in seconds, of every audio latent frame — four mel frames
/// of 10 ms each, the first covering one (causal).
pub fn audio_positions(n: usize) -> Vec<f32> {
    let sec = |i: usize| ((i * 4) as f64 + 1.0 - 4.0).max(0.0) * 160.0 / 16000.0;
    (0..n)
        .map(|i| ((sec(i) + sec(i + 1)) / 2.0) as f32)
        .collect()
}

/// One ancestral Euler step (`EulerAncestralDiffusionStep`, η = 1) from
/// `x` at `sigma` to `sigma_next`, given the denoised prediction; at
/// `sigma_next = 0` the prediction itself.
/// Whether every value is a number: a NaN or an infinity anywhere in a latent
/// spreads to every pixel downstream, so a stage that makes one fails the
/// clip rather than writing a blank file.
fn finite(values: &[f32]) -> bool {
    values.iter().all(|v| v.is_finite())
}

pub fn ancestral_step(x: &mut [f32], denoised: &[f32], sigma: f32, sigma_next: f32, noise: &[f32]) {
    if sigma_next == 0.0 {
        x.copy_from_slice(denoised);
        return;
    }
    let down = sigma_next * (1.0 + (sigma_next / sigma - 1.0));
    let ratio = down / sigma;
    let alpha_next = 1.0 - sigma_next;
    let alpha_down = 1.0 - down;
    let renoise = (sigma_next * sigma_next
        - down * down * alpha_next * alpha_next / (alpha_down * alpha_down))
        .max(0.0)
        .sqrt();
    let scale = alpha_next / alpha_down;
    for ((x, d), n) in x.iter_mut().zip(denoised).zip(noise) {
        let next = ratio * *x + (1.0 - ratio) * d;
        *x = scale * next + n * renoise;
    }
}

/// Everything that turns a prompt's conditioning into a clip: the
/// transformer, the upscaler, the decoders, the file.
pub struct Renderer {
    pub backend: Arc<dyn Backend>,
    pub transformer: LtxTransformer,
    pub vae: vae::VideoDecoder,
    pub audio: Option<audio::AudioPipeline>,
    pub upscaler: Option<upscaler::LatentUpscaler>,
}

pub struct Pipeline {
    renderer: Renderer,
    prompt: text::PromptEncoder,
    defaults: RwLock<VideoDefaults>,
    /// One clip at a time.
    busy: Mutex<()>,
    rate: Mutex<RateModel>,
    pub companions: Companions,
    /// The defaults as the configuration set them, for a reset.
    pub configured: VideoDefaults,
}

impl Pipeline {
    /// Loads everything but the text encoder's language model, which is
    /// the server's own `model` (loaded and placed like any other).
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        transformer: &LoadedModel,
        companions: Companions,
        encoder: Arc<dyn ModelForward>,
        tokenizer: Arc<Tokenizer>,
        backend: Arc<dyn Backend>,
        defaults: VideoDefaults,
        vae_precision: crate::engine::image::vae::VaePrecision,
    ) -> Result<Self> {
        let model = LtxTransformer::load(transformer, backend.clone())?;
        let c = model.config.clone();
        let encoder_weights = LoadedModel::open(&companions.text_encoder)
            .with_context(|| format!("opening {}", companions.text_encoder.display()))?;
        let width = encoder.config().n_embd;
        let layers = encoder.n_trunk_layer() + 1;
        let heads = text::FeatureHeads::load(&encoder_weights, width, layers)?;
        ensure!(
            heads.video_dim() == c.dim && heads.audio_dim() == c.audio_dim,
            "the text encoder's heads project to {} and {}, and the transformer reads {} and {}",
            heads.video_dim(),
            heads.audio_dim(),
            c.dim,
            c.audio_dim
        );
        let video_connector = text::Connector::load(
            transformer,
            "video_embeddings_connector",
            c.connector_heads,
            c.connector_max_pos,
            c.theta,
            c.eps,
        )?;
        let audio_connector = text::Connector::load(
            transformer,
            "audio_embeddings_connector",
            c.audio_connector_heads,
            c.connector_max_pos,
            c.theta,
            c.eps,
        )?;
        let vae = vae::VideoDecoder::load(&companions.vae, backend.clone(), vae_precision)
            .with_context(|| format!("loading the video VAE {}", companions.vae.display()))?;
        let audio = companions
            .audio_vae
            .as_deref()
            .map(|p| {
                audio::AudioPipeline::load(p, backend.clone())
                    .with_context(|| format!("loading the audio VAE {}", p.display()))
            })
            .transpose()?;
        if let Some(a) = &audio {
            ensure!(
                a.token_width() == c.audio_in_channels,
                "the audio VAE's latent is {} wide and the transformer's {}",
                a.token_width(),
                c.audio_in_channels
            );
        }
        let upscaler = companions
            .upscaler
            .as_deref()
            .map(|p| {
                upscaler::LatentUpscaler::load(p, backend.clone())
                    .with_context(|| format!("loading the latent upscaler {}", p.display()))
            })
            .transpose()?;
        if let Some(version) = &c.gemma_version {
            log::info!(
                "orangu-server: ltxv was trained against the text encoder '{version}'; serving {}",
                companions.text_encoder.display()
            );
        }
        let rate = calibrate(&model, &*backend);
        Ok(Self {
            renderer: Renderer {
                backend,
                transformer: model,
                vae,
                audio,
                upscaler,
            },
            prompt: text::PromptEncoder {
                model: encoder,
                tokenizer,
                heads,
                video: video_connector,
                audio: audio_connector,
            },
            configured: defaults.clone(),
            defaults: RwLock::new(defaults),
            busy: Mutex::new(()),
            rate: Mutex::new(rate),
            companions,
        })
    }

    pub fn defaults(&self) -> VideoDefaults {
        self.defaults
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    pub fn set_defaults(&self, defaults: VideoDefaults) {
        *self.defaults.write().unwrap_or_else(|p| p.into_inner()) = defaults;
    }

    pub fn rate_model(&self) -> RateModel {
        *self.rate.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub fn has_audio(&self) -> bool {
        self.renderer.audio.is_some()
    }

    pub fn has_upscaler(&self) -> bool {
        self.renderer.upscaler.is_some()
    }

    pub fn transformer_config(&self) -> &transformer::LtxConfig {
        &self.renderer.transformer.config
    }

    /// The seconds a clip of `request`'s shape takes here, by the current
    /// rate.
    pub fn estimate(&self, request: &VideoRequest) -> Result<f64> {
        let plan = Plan::new(request, self.has_upscaler())?;
        Ok(plan.estimate(&self.rate_model(), request.audio && self.has_audio()))
    }

    /// Generates one clip. Blocks for its whole run; `progress` hears after
    /// every phase and step.
    pub fn generate(
        &self,
        request: &VideoRequest,
        progress: &mut dyn FnMut(Progress),
        cancel: &AtomicBool,
    ) -> Result<GeneratedVideo> {
        let _busy = self.busy.lock().unwrap_or_else(|p| p.into_inner());
        let started = Instant::now();
        let plan = Plan::new(request, self.has_upscaler())?;
        let audio_on = request.audio && self.has_audio();
        let seed = request.seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64 & 0xFFFF_FFFF)
                .unwrap_or(42)
        });
        let rate = self.rate_model();
        let total_steps = plan.steps();
        let eta = plan.estimate(&rate, audio_on);
        let mut report = |step: usize, phase: &'static str, eta: f64| {
            progress(Progress {
                step,
                steps: total_steps,
                phase,
                eta_seconds: eta.max(0.0),
            })
        };
        report(0, "encoding", eta);
        let mut timings = Timings::default();
        let t = Instant::now();
        let conditioning =
            self.prompt
                .encode(&*self.renderer.backend, &request.prompt, Some(cancel))?;
        timings.encode = t.elapsed();
        ensure!(
            finite(&conditioning.video) && finite(&conditioning.audio),
            "the text encoder gave non-finite hidden states"
        );
        log::info!(
            "orangu-server: [video] prompt: {} tokens, encoded in {:.1} s",
            conditioning.tokens,
            timings.encode.as_secs_f64()
        );
        let (bytes, frames) = self.renderer.render(
            request,
            &plan,
            conditioning,
            seed,
            &rate,
            eta - rate.encode,
            &mut report,
            &mut timings,
            cancel,
        )?;
        self.learn(&plan, &timings, audio_on);
        Ok(GeneratedVideo {
            bytes,
            width: plan.width,
            height: plan.height,
            frames,
            fps: request.fps,
            seed,
            has_audio: audio_on,
            elapsed: started.elapsed(),
            timings,
        })
    }

    /// Replaces the rate model's parts this clip measured.
    fn learn(&self, plan: &Plan, t: &Timings, audio: bool) {
        let mut rate = self.rate.lock().unwrap_or_else(|p| p.into_inner());
        rate.encode = t.encode.as_secs_f64();
        let (one, two) = plan.tokens();
        // Scale the per-step model so it reproduces stage 1's measurement.
        let s1 = t.stage_1.as_secs_f64() / (STAGE_1_SIGMAS.len() - 1) as f64;
        let predicted = rate.seconds_per_step(one);
        if predicted > 0.0 && s1 > 0.0 {
            let k = s1 / predicted;
            if let Some(two) = two {
                // Two sizes measured: keep the shape, fit the scale on both.
                let s2 = t.stage_2.as_secs_f64() / (STAGE_2_SIGMAS.len() - 1) as f64;
                let p2 = rate.seconds_per_step(two);
                let k2 = if p2 > 0.0 { s2 / p2 } else { k };
                let k = (k + k2) / 2.0;
                rate.per_token *= k;
                rate.per_step *= k;
                rate.per_token_sq *= k;
            } else {
                rate.per_token *= k;
                rate.per_step *= k;
                rate.per_token_sq *= k;
            }
        }
        let pixels = (plan.width * plan.height * plan.frames) as f64;
        rate.decode_per_pixel = t.decode.as_secs_f64() / pixels;
        rate.other_per_pixel = (t.upscale + t.mux).as_secs_f64() / pixels;
        if audio {
            rate.audio_per_second =
                t.audio.as_secs_f64() / (plan.frames as f64 / plan.fps.max(1) as f64).max(0.1);
        }
    }
}

impl Renderer {
    /// Runs one stage of the sampler over `video` and `audio`, in place.
    #[allow(clippy::too_many_arguments)]
    fn run_stage(
        &self,
        geometry: &Geometry,
        video: &mut [f32],
        audio: &mut [f32],
        conditioning: &text::Conditioning,
        sigmas: &[f32],
        noise_seed: u64,
        mut on_step: impl FnMut(Duration) -> Result<()>,
        cancel: &AtomicBool,
    ) -> Result<()> {
        let mut rng = vae::Rng::new(noise_seed);
        for i in 0..sigmas.len() - 1 {
            let started = Instant::now();
            let (sigma, next) = (sigmas[i], sigmas[i + 1]);
            let (v_vel, a_vel) = self.transformer.forward(
                geometry,
                &StreamInput {
                    latent: video,
                    timesteps: &vec![sigma; geometry.n_video],
                    sigma,
                    context: &conditioning.video,
                },
                &StreamInput {
                    latent: audio,
                    timesteps: &vec![sigma; geometry.n_audio],
                    sigma,
                    context: &conditioning.audio,
                },
                Some(cancel),
            )?;
            ensure!(
                finite(&v_vel) && finite(&a_vel),
                "the transformer gave non-finite velocities at step {} of {} (sigma {sigma})",
                i + 1,
                sigmas.len() - 1
            );
            let denoised = |x: &[f32], v: &[f32]| -> Vec<f32> {
                x.iter().zip(v).map(|(x, v)| x - v * sigma).collect()
            };
            let (dv, da) = (denoised(video, &v_vel), denoised(audio, &a_vel));
            let (nv, na) = if next > 0.0 {
                (rng.fill(video.len()), rng.fill(audio.len()))
            } else {
                (Vec::new(), Vec::new())
            };
            ancestral_step(video, &dv, sigma, next, &nv);
            ancestral_step(audio, &da, sigma, next, &na);
            on_step(started.elapsed())?;
        }
        Ok(())
    }

    /// The clip of `request` from its prompt's `conditioning`: both stages,
    /// the decoders and the file. Returns the MP4 and its frame count;
    /// `timings` gets each phase. `eta` is the estimate of all of it, which
    /// `report` hears count down.
    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &self,
        request: &VideoRequest,
        plan: &Plan,
        conditioning: text::Conditioning,
        seed: u64,
        rate: &RateModel,
        mut eta: f64,
        report: &mut dyn FnMut(usize, &'static str, f64),
        timings: &mut Timings,
        cancel: &AtomicBool,
    ) -> Result<(Vec<u8>, usize)> {
        let c = self.transformer.config.clone();
        let fps = request.fps as f64;
        let audio_on = request.audio && self.audio.is_some();

        // Stage 1.
        let (h1, w1) = plan.stage_1;
        let lf = plan.latent_frames;
        let mut init = vae::Rng::new(seed);
        let mut video = init.fill(lf * h1 * w1 * c.in_channels);
        let mut audio = init.fill(plan.audio_latents * c.audio_in_channels);
        let geometry = Geometry::new(
            &c,
            &video_positions(lf, h1, w1, fps),
            &audio_positions(plan.audio_latents),
            h1 * w1,
        );
        let mut step = 0usize;
        let (one, two) = plan.tokens();
        let t = Instant::now();
        let per_step_1 = rate.seconds_per_step(one);
        self.run_stage(
            &geometry,
            &mut video,
            &mut audio,
            &conditioning,
            &STAGE_1_SIGMAS,
            seed + STAGE_1_NOISE_OFFSET,
            |_| {
                step += 1;
                eta -= per_step_1;
                report(step, "stage 1", eta);
                Ok(())
            },
            cancel,
        )?;
        timings.stage_1 = t.elapsed();
        drop(geometry);

        let (rows, cols) = if plan.two_stage {
            let upscaler = self
                .upscaler
                .as_ref()
                .expect("a two-stage plan has an upscaler");
            report(step, "upscaling", eta);
            let t = Instant::now();
            let up = upscaler.upscale(
                &video,
                (lf, h1, w1),
                self.vae.latent_statistics(),
                Some(cancel),
            )?;
            ensure!(finite(&up), "the latent upscaler gave non-finite values");
            timings.upscale = t.elapsed();
            let (h2, w2) = (h1 * 2, w1 * 2);
            // Both latents noised back to the first σ of stage 2 with fresh
            // noise, as `GaussianNoiser` does with the stage's noise scale.
            let s = STAGE_2_SIGMAS[0];
            let mut video2: Vec<f32> = up
                .iter()
                .map(|&x| x * (1.0 - s) + init.normal() * s)
                .collect();
            audio
                .iter_mut()
                .for_each(|x| *x = *x * (1.0 - s) + init.normal() * s);
            let geometry = Geometry::new(
                &c,
                &video_positions(lf, h2, w2, fps),
                &audio_positions(plan.audio_latents),
                h2 * w2,
            );
            let t = Instant::now();
            let per_step_2 = rate.seconds_per_step(two.unwrap_or(one));
            self.run_stage(
                &geometry,
                &mut video2,
                &mut audio,
                &conditioning,
                &STAGE_2_SIGMAS,
                seed + STAGE_2_NOISE_OFFSET,
                |_| {
                    step += 1;
                    eta -= per_step_2;
                    report(step, "stage 2", eta);
                    Ok(())
                },
                cancel,
            )?;
            timings.stage_2 = t.elapsed();
            video = video2;
            (h2, w2)
        } else {
            (h1, w1)
        };
        drop(conditioning);

        report(step, "decoding", eta);
        let t = Instant::now();
        let clip = self
            .vae
            .decode(&video, (lf, rows, cols), seed, Some(cancel))?;
        ensure!(
            finite(&clip.data),
            "the video VAE decoded non-finite pixels"
        );
        timings.decode = t.elapsed();
        ensure!(
            clip.width == plan.width && clip.height == plan.height,
            "the VAE decoded {}x{}, expected {}x{}",
            clip.width,
            clip.height,
            plan.width,
            plan.height
        );
        let frames = clip.frames.min(plan.frames);
        let frame_len = clip.width * clip.height * 3;
        let frame_data: Vec<Vec<f32>> = (0..frames)
            .map(|f| clip.data[f * frame_len..(f + 1) * frame_len].to_vec())
            .collect();
        drop(clip);

        let waveform = if audio_on {
            report(step, "sound", eta);
            let t = Instant::now();
            let w = self.audio.as_ref().expect("audio_on").decode(
                &audio,
                plan.audio_latents,
                Some(cancel),
            )?;
            timings.audio = t.elapsed();
            Some(w)
        } else {
            None
        };

        report(step, "encoding the file", eta);
        let t = Instant::now();
        // The sound cut to the clip's length.
        let audio_track = waveform.as_ref().map(|w| {
            let want = (frames as f64 / fps * w.sample_rate as f64).round() as usize * w.channels;
            let n = want.min(w.samples.len());
            (&w.samples[..n], w.channels, w.sample_rate as u32)
        });
        let bytes = mp4::encode_clip(
            &frame_data,
            plan.width,
            plan.height,
            request.fps,
            audio_track,
            mp4::Av1Settings {
                quantizer: request.quality,
                ..Default::default()
            },
            Some(cancel),
        )?;
        timings.mux = t.elapsed();
        Ok((bytes, frames))
    }
}

/// One block's feed-forward over 256 random tokens, timed: the startup
/// seed of the [`RateModel`].
fn calibrate(model: &LtxTransformer, backend: &dyn Backend) -> RateModel {
    let (per_token_block, blocks) = model.calibrate(backend, 256);
    RateModel::from_calibration(per_token_block, blocks)
}

/// What a background generation reports, in order: progress, then exactly
/// one of `Done` or `Error`.
#[derive(Debug)]
pub enum VideoEvent {
    Progress(Progress),
    Done(Box<GeneratedVideo>),
    Error(String),
}

/// Stops a background generation when dropped.
pub struct CancelOnDrop(Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

impl Pipeline {
    /// Runs `request` on a blocking thread, streaming [`VideoEvent`]s back.
    pub fn spawn(
        self: &Arc<Self>,
        request: VideoRequest,
    ) -> (tokio::sync::mpsc::Receiver<VideoEvent>, CancelOnDrop) {
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let cancel = Arc::new(AtomicBool::new(false));
        let pipeline = self.clone();
        let flag = cancel.clone();
        tokio::task::spawn_blocking(move || {
            let progress_tx = tx.clone();
            let mut progress = |p: Progress| {
                let _ = progress_tx.try_send(VideoEvent::Progress(p));
            };
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                pipeline.generate(&request, &mut progress, &flag)
            }));
            let event = match result {
                Ok(Ok(video)) => VideoEvent::Done(Box::new(video)),
                Ok(Err(err)) => VideoEvent::Error(format!("{err:#}")),
                Err(_) => VideoEvent::Error(
                    crate::panic_capture::take_last_panic_detail()
                        .unwrap_or_else(|| "video generation panicked".to_string()),
                ),
            };
            let _ = tx.blocking_send(event);
        });
        (rx, CancelOnDrop(cancel))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_snap_to_the_latent_grid() {
        assert_eq!(snap_frames(121), 121);
        assert_eq!(snap_frames(120), 121);
        assert_eq!(snap_frames(1), 1);
        assert_eq!(snap_frames(100), 97);
    }

    #[test]
    fn the_first_latent_frame_covers_one_pixel_frame() {
        let p = video_positions(2, 1, 1, 24.0);
        // Frame 0: [0, 1) → 0.5 / 24 s; frame 1: [1, 9) → 5 / 24 s.
        assert!((p[0] - 0.5 / 24.0).abs() < 1e-6);
        assert!((p[3] - 5.0 / 24.0).abs() < 1e-6);
        assert_eq!(p[1], 16.0);
        let a = audio_positions(2);
        // Latent 0: mel [0, 1) → 5 ms; latent 1: mel [1, 5) → 30 ms.
        assert!((a[0] - 0.005).abs() < 1e-6);
        assert!((a[1] - 0.03).abs() < 1e-6);
    }

    #[test]
    fn the_last_step_lands_on_the_prediction() {
        let mut x = vec![1.0, 2.0];
        ancestral_step(&mut x, &[0.5, 0.5], 0.42, 0.0, &[]);
        assert_eq!(x, vec![0.5, 0.5]);
    }

    #[test]
    fn without_noise_an_ancestral_step_interpolates_toward_the_prediction() {
        // sd = σn²/σ; x' = (sd/σ)x + (1 − sd/σ)·x0, then rescaled.
        let (s, n) = (1.0f32, 0.5f32);
        let mut x = vec![1.0];
        ancestral_step(&mut x, &[0.0], s, n, &[0.0]);
        let sd = n * n / s;
        let expect = (sd / s) * (1.0 - n) / (1.0 - sd);
        assert!((x[0] - expect).abs() < 1e-6);
    }

    #[test]
    fn a_two_stage_plan_draws_at_half_size_first() {
        let r = VideoRequest::from_defaults("x", &VideoDefaults::default());
        let p = Plan::new(&r, true).unwrap();
        assert_eq!(p.stage_1, (448 / 64, 768 / 64));
        assert_eq!(p.latent_frames, 16);
        assert_eq!(p.audio_latents, 126);
        assert_eq!(p.steps(), 11);
        let p = Plan::new(&r, false).unwrap();
        assert_eq!(p.stage_1, (448 / 32, 768 / 32));
        assert_eq!(p.steps(), 8);
        let mut odd = r.clone();
        odd.width = 800;
        assert!(Plan::new(&odd, true).is_err());
    }
}

#[cfg(test)]
mod probe {
    use super::*;

    /// With `ORANGU_LTXV` naming the transformer: both text connectors over
    /// random features, then one stage-1 step of the default clip — shapes,
    /// finiteness, and the time each takes, against the rate model's
    /// estimate from the calibration.
    #[test]
    #[ignore]
    fn a_default_stage_one_step_over_real_weights() {
        let Ok(path) = std::env::var("ORANGU_LTXV") else {
            return;
        };
        let loaded = LoadedModel::open(Path::new(&path)).unwrap();
        let backend: Arc<dyn Backend> = Arc::new(crate::engine::backend::CpuBackend);
        let model = LtxTransformer::load(&loaded, backend.clone()).unwrap();
        let c = model.config.clone();
        let rate = calibrate(&model, &*backend);
        eprintln!("calibration: {rate:?}");
        let mut rng = vae::Rng::new(1);
        let t = Instant::now();
        let video_connector = text::Connector::load(
            &loaded,
            "video_embeddings_connector",
            c.connector_heads,
            c.connector_max_pos,
            c.theta,
            c.eps,
        )
        .unwrap();
        let audio_connector = text::Connector::load(
            &loaded,
            "audio_embeddings_connector",
            c.audio_connector_heads,
            c.connector_max_pos,
            c.theta,
            c.eps,
        )
        .unwrap();
        let ctx = video_connector
            .forward(&*backend, &rng.fill(40 * c.dim), None)
            .unwrap();
        let actx = audio_connector
            .forward(&*backend, &rng.fill(40 * c.audio_dim), None)
            .unwrap();
        eprintln!("connectors: {:.1} s", t.elapsed().as_secs_f64());
        assert_eq!(ctx.len(), text::PROMPT_TOKENS * c.dim);
        assert!(ctx.iter().chain(&actx).all(|v| v.is_finite()));
        let r = VideoRequest::from_defaults("x", &VideoDefaults::default());
        let plan = Plan::new(&r, true).unwrap();
        let (h, w) = plan.stage_1;
        let lf = plan.latent_frames;
        let geometry = Geometry::new(
            &c,
            &video_positions(lf, h, w, 24.0),
            &audio_positions(plan.audio_latents),
            h * w,
        );
        let video = rng.fill(lf * h * w * c.in_channels);
        let audio = rng.fill(plan.audio_latents * c.audio_in_channels);
        let n = lf * h * w;
        let t = Instant::now();
        let (v, a) = model
            .forward(
                &geometry,
                &StreamInput {
                    latent: &video,
                    timesteps: &vec![1.0; n],
                    sigma: 1.0,
                    context: &ctx,
                },
                &StreamInput {
                    latent: &audio,
                    timesteps: &vec![1.0; plan.audio_latents],
                    sigma: 1.0,
                    context: &actx,
                },
                None,
            )
            .unwrap();
        let took = t.elapsed().as_secs_f64();
        eprintln!(
            "stage-1 step at {n} video + {} audio tokens: {took:.1} s (estimated {:.1} s)",
            plan.audio_latents,
            rate.seconds_per_step(n)
        );
        assert!(v.iter().chain(&a).all(|x| x.is_finite()));
        // The velocity of pure noise at σ = 1 is about −noise + x0: its
        // RMS should be of the order of the noise's.
        let rms = (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
        eprintln!("velocity rms {rms}");
    }
}

#[cfg(test)]
mod end_to_end {
    use super::*;

    /// With `ORANGU_LTXV` naming the transformer and `ORANGU_LTX_DIR` a
    /// directory holding the VAEs and the upscaler: a small clip drawn from
    /// an *empty* prompt — the connectors' registers alone, the conditioning
    /// LTX's own pipeline gives no text — through both stages, the decoders
    /// and the file, written to `ORANGU_VIDEO_OUT`. Everything but the text
    /// encoder, on the real weights.
    #[test]
    #[ignore]
    fn an_unconditioned_clip_over_real_weights() {
        let (Ok(path), Ok(dir)) = (
            std::env::var("ORANGU_LTXV"),
            std::env::var("ORANGU_LTX_DIR"),
        ) else {
            return;
        };
        let out = std::env::var("ORANGU_VIDEO_OUT").unwrap_or_else(|_| "unconditioned.mp4".into());
        let dir = Path::new(&dir);
        let backend: Arc<dyn Backend> = Arc::new(crate::engine::backend::CpuBackend);
        let loaded = LoadedModel::open(Path::new(&path)).unwrap();
        let transformer = LtxTransformer::load(&loaded, backend.clone()).unwrap();
        let c = transformer.config.clone();
        let t = Instant::now();
        let vae = vae::VideoDecoder::load(
            &crate::model_spec::find_ltx_video_vae(dir).unwrap(),
            backend.clone(),
            crate::engine::image::vae::VaePrecision::Int8,
        )
        .unwrap();
        let audio = audio::AudioPipeline::load(
            &crate::model_spec::find_ltx_audio_vae(dir).unwrap(),
            backend.clone(),
        )
        .unwrap();
        let upscaler = upscaler::LatentUpscaler::load(
            &crate::model_spec::find_ltx_spatial_upscaler(dir).unwrap(),
            backend.clone(),
        )
        .unwrap();
        eprintln!("companions loaded in {:.1} s", t.elapsed().as_secs_f64());
        let vc = text::Connector::load(
            &loaded,
            "video_embeddings_connector",
            c.connector_heads,
            c.connector_max_pos,
            c.theta,
            c.eps,
        )
        .unwrap();
        let ac = text::Connector::load(
            &loaded,
            "audio_embeddings_connector",
            c.audio_connector_heads,
            c.connector_max_pos,
            c.theta,
            c.eps,
        )
        .unwrap();
        let conditioning = text::Conditioning {
            video: vc.forward(&*backend, &[], None).unwrap(),
            audio: ac.forward(&*backend, &[], None).unwrap(),
            tokens: 0,
        };
        let renderer = Renderer {
            backend: backend.clone(),
            transformer,
            vae,
            audio: Some(audio),
            upscaler: Some(upscaler),
        };
        let mut request = VideoRequest::from_defaults("", &VideoDefaults::default());
        request.width = 512;
        request.height = 320;
        request.frames = 25;
        let plan = Plan::new(&request, true).unwrap();
        let rate = calibrate(&renderer.transformer, &*backend);
        let mut timings = Timings::default();
        let cancel = AtomicBool::new(false);
        let mut report = |step: usize, phase: &'static str, eta: f64| {
            eprintln!(
                "[{:.0} s] {phase} step {step}/{} eta {eta:.0} s",
                t.elapsed().as_secs_f64(),
                plan.steps()
            );
        };
        let (bytes, frames) = renderer
            .render(
                &request,
                &plan,
                conditioning,
                7,
                &rate,
                plan.estimate(&rate, true),
                &mut report,
                &mut timings,
                &cancel,
            )
            .unwrap();
        eprintln!("{frames} frames, {} bytes, {timings:?}", bytes.len());
        std::fs::write(&out, &bytes).unwrap();
    }
}
