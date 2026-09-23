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

//! Text-to-image and image-to-image with Qwen-Image — the pipeline a
//! `qwen_image` GGUF is served through.
//!
//! A `qwen_image` file is one third of a model. It holds the diffusion
//! transformer ([`transformer`]), which denoises a *latent* picture under
//! the guidance of a prompt's hidden states. Those hidden states come from a
//! Qwen2.5-VL-7B text encoder, an ordinary `qwen2vl` GGUF served through
//! the usual `arch::llama` path and read through
//! `ModelForward::forward_hidden_states` — the same call an embeddings
//! request makes, which is why the whole language-model engine is reused
//! rather than a second encoder written. And the latent becomes pixels
//! through the Qwen-Image VAE ([`vae`]), a `safetensors` file. The server
//! finds the two companions in the models directory ([`Companions`]), or
//! where the configuration points.
//!
//! Generation is diffusers' `QwenImagePipeline`, step for step:
//! encode the prompt (and the negative one), start from seeded Gaussian
//! noise (or from the attached picture's latent, noised to the chosen
//! `strength`), and walk the flow-matching schedule ([`scheduler`]) — at
//! every step one transformer pass per prompt, classifier-free guidance
//! combining the two with the norm-preserving rescale Qwen-Image uses, one
//! Euler step — then decode and write PNG or JPEG ([`codec`]).
//!
//! One picture at a time: the transformer is 20 billion parameters and a
//! step is a full pass over them, so two requests interleaved would be
//! slower than two in turn. Concurrent callers queue on the pipeline's lock.
//!
//! **Qwen-Image 2.1** (`qwen_image_2_1`, [`Variant::QwenImage21`]) goes
//! through the same pipeline with three different parts: a 7-billion-
//! parameter single-stream transformer ([`transformer21`]) conditioned on a
//! Qwen3-VL-8B encoder's hidden states *before* its final norm, and its own
//! RGBA VAE ([`vae21`]) with 64 latent channels at a sixteenth of the
//! picture. Its latent tokens are the VAE's cells as they are (no 2x2
//! packing), its prompt is run through the transformer once per picture
//! rather than once per step (see [`transformer21`]'s module doc), and its
//! guidance is plain classifier-free guidance — the release samples without
//! any, forty steps.

pub mod codec;
pub mod lora;
pub mod qwen3vl;
pub mod safetensors;
pub mod scheduler;
pub mod stepcache;
pub mod transformer;
pub mod transformer21;
pub mod vae;
pub mod vae21;

use crate::engine::arch::ModelForward;
use crate::engine::backend::Backend;
use crate::engine::loader::LoadedModel;
use crate::engine::tokenizer::Tokenizer;
use anyhow::{Context, Result, bail, ensure};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

pub use codec::ImageFormat;
use scheduler::{Noise, ScheduleConfig};
use transformer::{ForwardInput, QwenImageTransformer};
use transformer21::QwenImage21Transformer;
use vae::{Feature, LATENTS_MEAN, LATENTS_STD, QwenImageVae, SPATIAL_COMPRESSION, Z_DIM};
use vae21::QwenImage21Vae;

/// The text encoder's chat framing around a prompt — diffusers'
/// `prompt_template_encode`. The hidden states of the framing itself are
/// dropped ([`Pipeline::encode_prompt`]); only the prompt's own tokens
/// condition the picture, but they are encoded *in this context*.
const PROMPT_TEMPLATE_PREFIX: &str = "<|im_start|>system\nDescribe the image by detailing the \
     color, shape, size, texture, quantity, text, spatial relationships of the objects and \
     background:<|im_end|>\n<|im_start|>user\n";
const PROMPT_TEMPLATE_SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n";
/// diffusers' `tokenizer_max_length`: how much of a prompt is read.
const MAX_PROMPT_TOKENS: usize = 1024;

/// Qwen-Image 2.1's framing — `QwenImage21Pipeline.prompt_template_t2i`.
/// Only the system turn is dropped from the hidden states: the user turn's
/// own framing and the assistant cue stay, as diffusers keeps them.
const PROMPT21_SYSTEM: &str =
    "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";
const PROMPT21_USER_PREFIX: &str = "<|im_start|>user\n";

/// Which picture model the pipeline serves — decided by the transformer
/// file's architecture, and deciding every companion and default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    /// `qwen_image`: Qwen-Image (`2512`, `Edit-2511`) — dual-stream,
    /// 20 billion parameters, Qwen2.5-VL-7B encoder, RGB VAE.
    QwenImage,
    /// `qwen_image_2_1`: Qwen-Image 2.1 — single-stream and block-causal,
    /// 7 billion parameters, Qwen3-VL-8B encoder, RGBA VAE.
    QwenImage21,
}

impl Variant {
    /// The variant a transformer's `general.architecture` names.
    pub fn from_architecture(architecture: &str) -> Option<Self> {
        match architecture {
            "qwen_image" => Some(Self::QwenImage),
            orangu::model_spec::QWEN_IMAGE_21_ARCHITECTURE => Some(Self::QwenImage21),
            _ => None,
        }
    }

    pub fn architecture(self) -> &'static str {
        match self {
            Self::QwenImage => "qwen_image",
            Self::QwenImage21 => orangu::model_spec::QWEN_IMAGE_21_ARCHITECTURE,
        }
    }

    /// Pixels one latent token covers along each side, which a picture's
    /// size must be a multiple of: Qwen-Image's VAE cell is 8 pixels and its
    /// transformer packs 2x2 cells into a token; Qwen-Image 2.1's cell is 16
    /// pixels, one token each, and diffusers still rounds to 32.
    pub fn size_unit(self) -> usize {
        match self {
            Self::QwenImage => 2 * SPATIAL_COMPRESSION,
            Self::QwenImage21 => 2 * vae21::SPATIAL_COMPRESSION,
        }
    }

    /// The architecture and width of the text encoder the transformer reads.
    pub fn text_encoder(self) -> (&'static str, u64) {
        match self {
            Self::QwenImage => (TEXT_ENCODER_ARCHITECTURE, TEXT_ENCODER_WIDTH),
            Self::QwenImage21 => ("qwen3vl", 4096),
        }
    }

    /// Where `orangu-server download` fetches the companions from.
    fn text_encoder_download(self) -> (&'static str, &'static str) {
        match self {
            Self::QwenImage => (TEXT_ENCODER_REPO, TEXT_ENCODER_DEFAULT_TAG),
            Self::QwenImage21 => (
                orangu::model_download::QWEN_IMAGE_21_TEXT_ENCODER_REPO,
                orangu::model_download::QWEN_IMAGE_21_TEXT_ENCODER_TAG,
            ),
        }
    }

    fn vae_download(self) -> (&'static str, &'static str) {
        match self {
            Self::QwenImage => (VAE_REPO, VAE_FILE),
            Self::QwenImage21 => (
                orangu::model_download::QWEN_IMAGE_21_VAE_REPO,
                orangu::model_download::QWEN_IMAGE_21_VAE_FILE,
            ),
        }
    }

    /// The model's own release settings, before the configuration: see
    /// [`ImageDefaults::default`] for Qwen-Image's. Qwen-Image 2.1's are
    /// forty steps without guidance (`true_cfg_scale = 1.0`, "meant to be
    /// sampled without guidance").
    pub fn release_steps_and_cfg(self) -> (usize, f32) {
        match self {
            Self::QwenImage => {
                let d = ImageDefaults::default();
                (d.steps, d.cfg_scale)
            }
            Self::QwenImage21 => (40, 1.0),
        }
    }

    /// Multiply-adds per image token per transformer pass — see
    /// [`MACS_PER_TOKEN_PASS`].
    fn macs_per_token_pass(self) -> f64 {
        match self {
            Self::QwenImage => MACS_PER_TOKEN_PASS,
            // `4 × 4096² + 3 × 4096 × 12288` per block, thirty-two blocks.
            Self::QwenImage21 => 7.0e9,
        }
    }
}

/// The architecture string of the text encoder the pipeline needs.
pub const TEXT_ENCODER_ARCHITECTURE: &str = "qwen2vl";
/// The width of its hidden states, which is what tells a Qwen2.5-VL-7B
/// apart from the 3B (2048) and 72B (8192) in a models directory.
pub const TEXT_ENCODER_WIDTH: u64 = 3584;

/// Where the pipeline gets its companions from — the same places
/// `orangu-server download` fetches them from beside a `qwen_image` model.
pub const TEXT_ENCODER_REPO: &str = orangu::model_download::QWEN_IMAGE_TEXT_ENCODER_REPO;
pub const TEXT_ENCODER_DEFAULT_TAG: &str = orangu::model_download::QWEN_IMAGE_TEXT_ENCODER_TAG;
pub const VAE_REPO: &str = orangu::model_download::QWEN_IMAGE_VAE_REPO;
pub const VAE_FILE: &str = orangu::model_download::QWEN_IMAGE_VAE_FILE;

/// The two files a picture model cannot generate without — and, for
/// Qwen-Image 2.1, the one that lets it edit a picture.
#[derive(Debug, Clone)]
pub struct Companions {
    pub text_encoder: PathBuf,
    pub vae: PathBuf,
    /// The text encoder's vision projector (`mmproj-*.gguf`): with it, a
    /// picture attached to a `qwen_image_2_1` request is *read* — the model
    /// edits it — rather than drawn over. `None` for Qwen-Image, or when
    /// there is none (or `[orangu-server].vision = none`).
    pub vision: Option<PathBuf>,
}

impl Companions {
    /// Finds both companions of a `variant` model: the configured path or
    /// spec when there is one, otherwise the models directory is searched —
    /// for a text encoder GGUF of the right architecture and width (the
    /// largest, when there are several quantizations), and for the VAE by
    /// its tensors rather than its name.
    pub fn locate(
        models_dir: &Path,
        variant: Variant,
        text_encoder: Option<&str>,
        vae: Option<&str>,
        vision: Option<&str>,
    ) -> Result<Self> {
        let text_encoder = match text_encoder {
            Some(spec) => {
                orangu::model_spec::resolve_load_target(models_dir, spec)
                    .with_context(|| format!("resolving text_encoder '{spec}'"))?
                    .0
            }
            None => find_text_encoder(models_dir, variant)?,
        };
        let vae = match vae {
            Some(path) => {
                let path = PathBuf::from(path);
                let path = if path.is_absolute() {
                    path
                } else {
                    models_dir.join(path)
                };
                ensure!(path.is_file(), "vae {} is not a file", path.display());
                path
            }
            None => find_vae(models_dir, variant)?,
        };
        let vision = match (variant, vision) {
            (Variant::QwenImage, _) | (_, Some("none")) => None,
            (Variant::QwenImage21, Some(path)) => {
                let path = PathBuf::from(path);
                let path = if path.is_absolute() {
                    path
                } else {
                    models_dir.join(path)
                };
                ensure!(
                    orangu::model_spec::is_qwen3vl_8b_projector(&path),
                    "vision {} is not a Qwen3-VL-8B vision projector (an mmproj GGUF)",
                    path.display()
                );
                Some(path)
            }
            (Variant::QwenImage21, None) => {
                orangu::model_spec::find_qwen3vl_8b_projector(models_dir, &text_encoder)
            }
        };
        Ok(Self {
            text_encoder,
            vae,
            vision,
        })
    }
}

fn find_text_encoder(models_dir: &Path, variant: Variant) -> Result<PathBuf> {
    let found = match variant {
        Variant::QwenImage => orangu::model_spec::find_qwen_image_text_encoder(models_dir),
        Variant::QwenImage21 => orangu::model_spec::find_qwen_image21_text_encoder(models_dir),
    };
    let (architecture, width) = variant.text_encoder();
    let (repo, tag) = variant.text_encoder_download();
    let name = match variant {
        Variant::QwenImage => "Qwen2.5-VL-7B",
        Variant::QwenImage21 => "Qwen3-VL-8B",
    };
    found.ok_or_else(|| {
        anyhow::anyhow!(
            "no {name} text encoder ({architecture}, width {width}) found under {}. A {} model \
             is conditioned on that encoder's hidden states; download one with `orangu-server \
             download {repo}:{tag}` or point [orangu-server].text_encoder at it",
            models_dir.display(),
            variant.architecture(),
        )
    })
}

fn find_vae(models_dir: &Path, variant: Variant) -> Result<PathBuf> {
    let (found, what) = match variant {
        Variant::QwenImage => (
            orangu::model_spec::find_qwen_image_vae(models_dir),
            "Qwen-Image VAE (a .safetensors with the Wan decoder)",
        ),
        Variant::QwenImage21 => (
            orangu::model_spec::find_qwen_image21_vae(models_dir),
            "Qwen-Image 2.1 VAE (a .safetensors with a 64-channel, RGBA decoder)",
        ),
    };
    let (repo, file) = variant.vae_download();
    found.ok_or_else(|| {
        anyhow::anyhow!(
            "no {what} found under {}. It turns the model's latents into pixels; \
             `orangu-server download` of a {} model fetches {repo}'s {file} beside it, or point \
             [orangu-server].vae at the file",
            models_dir.display(),
            variant.architecture(),
        )
    })
}

/// One linear of the transformer, timed on a device and on the CPU — the
/// measurement `backend = auto` makes before committing a picture pipeline
/// to a GPU.
///
/// A picture is prefill-shaped work, hundreds of tokens through every
/// linear, and whether a given GPU beats the CPU at it is a question about
/// the two kernels on the two chips, not about the GPU existing: on a
/// board whose integrated Mali shares the CPU's memory and whose CPU has
/// `i8mm`, the device ran a 256-token step 7.8× *slower* than the CPU.
/// So rather than a rule about device classes, the pipeline runs the same
/// linear both ways and keeps the faster. See [`calibrate`].
#[derive(Debug, Clone, Copy)]
pub struct Calibration {
    /// The tensor's shape, for the log line.
    pub in_dim: usize,
    pub out_dim: usize,
    pub n_tokens: usize,
    /// The second run on each: the first pays for pipeline compilation and
    /// the weight upload, which a picture pays once too but which is not
    /// the per-step cost being compared. `device` is `None` when no device
    /// was offered — an explicit `backend = cpu`, where only the estimate
    /// below is wanted.
    pub device: Option<std::time::Duration>,
    pub cpu: std::time::Duration,
    /// The model's multiply-adds per image token per pass, which scales the
    /// linear's time up to a whole pass.
    pub macs_per_token_pass: f64,
}

impl Calibration {
    pub fn cpu_wins(&self) -> bool {
        self.device.is_none_or(|device| self.cpu <= device)
    }

    /// A first guess at the pipeline's rate in latent-token passes per
    /// second, from the winner's time on the calibration linear. A pass
    /// over one image token is about 6.8 G multiply-adds through the
    /// token-wide linears (`doc/PERF-IMAGE.md`, *Arithmetic*), and the
    /// calibration linear is `in_dim × out_dim` of them per token — so the
    /// ratio scales the measured time up to a whole pass. Attention adds
    /// to it at larger pictures, so this is a floor on the time, refined
    /// by the first real picture ([`Pipeline::rate`]).
    pub fn token_passes_per_second(&self) -> f64 {
        let winner = match self.device {
            Some(device) if device < self.cpu => device,
            _ => self.cpu,
        };
        let linear_macs = (self.in_dim * self.out_dim) as f64;
        let pass_seconds_per_token =
            winner.as_secs_f64() / self.n_tokens as f64 * (self.macs_per_token_pass / linear_macs);
        1.0 / pass_seconds_per_token.max(1e-9)
    }
}

/// Multiply-adds per image token per transformer pass, the token-wide
/// linears of every block: `3 × 3072² + 3072² + 2 × 3072 × 12288` per block,
/// sixty blocks.
const MACS_PER_TOKEN_PASS: f64 = 6.8e9;

/// Which tensor [`calibrate`] times: the first block's image MLP input
/// projection, the widest linear a step runs and the largest share of it
/// (`mlp` is over half of every pass). Qwen-Image 2.1's is the fused
/// `gate_up` (or, in diffusers' own layout, its `proj` half).
fn calibration_tensor(transformer: &LoadedModel) -> &'static str {
    [
        "transformer_blocks.0.img_mlp.net.0.proj.weight",
        "transformer_blocks.0.img_mlp.gate_up.weight",
        "transformer_blocks.0.img_mlp.proj.weight",
    ]
    .into_iter()
    .find(|name| transformer.has_tensor(name))
    .unwrap_or("transformer_blocks.0.img_mlp.net.0.proj.weight")
}
/// The token count [`calibrate`] runs — a 256×256 picture's worth, the
/// smallest size worth generating.
const CALIBRATION_TOKENS: usize = 256;

/// Times [`CALIBRATION_TENSOR`] through `device` (when one is offered) and
/// through the CPU at [`CALIBRATION_TOKENS`] tokens, on synthetic
/// activations. Each is run twice and the second timed; the whole thing is
/// well under a second on the CPU and a few seconds on a slow device — a
/// startup cost that saves hours when it says no, and that seeds the wait
/// estimate a picture is announced with.
pub fn calibrate(transformer: &LoadedModel, device: Option<&dyn Backend>) -> Result<Calibration> {
    let tensor = calibration_tensor(transformer);
    let w = transformer
        .matrix(tensor)
        .with_context(|| format!("picture model calibration tensor {tensor}"))?;
    let macs_per_token_pass = Variant::from_architecture(&transformer.config.architecture)
        .unwrap_or(Variant::QwenImage)
        .macs_per_token_pass();
    let n_tokens = CALIBRATION_TOKENS;
    let x: Vec<f32> = (0..n_tokens * w.in_dim)
        .map(|i| ((i * 37 % 23) as f32 - 11.0) * 0.031)
        .collect();
    let time = |backend: &dyn Backend| -> std::time::Duration {
        let op = crate::engine::backend::MatmulOp {
            x: &x,
            n_tokens,
            w: &w,
        };
        let _ = backend.matmul_batch(std::slice::from_ref(&op));
        let started = Instant::now();
        let _ = backend.matmul_batch(std::slice::from_ref(&op));
        started.elapsed()
    };
    Ok(Calibration {
        in_dim: w.in_dim,
        out_dim: w.out_dim,
        n_tokens,
        device: device.map(time),
        cpu: time(&crate::engine::backend::CpuBackend),
        macs_per_token_pass,
    })
}

/// Where `[orangu-server].image_lora` points: a path as given, a path
/// under the models directory, or a `<user>/<repo>:<file>` reference
/// resolved through the hub-cache layout `download` writes
/// (`models--<user>--<repo>/snapshots/<commit>/<file>`).
pub fn resolve_lora(models_dir: &Path, spec: &str) -> Result<PathBuf> {
    let direct = PathBuf::from(spec);
    if direct.is_file() {
        return Ok(direct);
    }
    let under_models = models_dir.join(spec);
    if under_models.is_file() {
        return Ok(under_models);
    }
    if let Some((repo, file)) = spec.split_once(':')
        && repo.matches('/').count() == 1
    {
        let snapshots = models_dir
            .join(format!("models--{}", repo.replace('/', "--")))
            .join("snapshots");
        if let Ok(entries) = std::fs::read_dir(&snapshots) {
            for entry in entries.flatten() {
                let candidate = entry.path().join(file);
                if candidate.is_file() {
                    return Ok(candidate);
                }
            }
        }
        bail!(
            "image_lora {spec}: {file} is not under {}; fetch it with `orangu-server download {spec}`",
            snapshots.display()
        );
    }
    bail!(
        "image_lora {spec}: no such file, as given or under {}",
        models_dir.display()
    )
}

/// Server-wide defaults a request may override — `[orangu-server].image_*`.
#[derive(Debug, Clone)]
pub struct ImageDefaults {
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    pub cfg_scale: f32,
    pub negative_prompt: String,
    pub strength: f32,
    /// The container a picture comes back in when the request names none
    /// and no attachment sets it.
    pub format: ImageFormat,
}

impl Default for ImageDefaults {
    /// Qwen-Image's own release settings: 1024x1024, 50 steps, true CFG 4.0
    /// against a blank negative prompt, and diffusers' image-to-image
    /// `strength` of 0.6.
    fn default() -> Self {
        Self {
            width: 1024,
            height: 1024,
            steps: 50,
            cfg_scale: 4.0,
            negative_prompt: " ".to_string(),
            strength: 0.6,
            format: ImageFormat::Png,
        }
    }
}

/// One generation.
#[derive(Debug, Clone)]
pub struct ImageRequest {
    pub prompt: String,
    /// `None` takes the server default; `Some("")` is a blank prompt too.
    pub negative_prompt: Option<String>,
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    /// `<= 1` runs the positive prompt alone — half the work, no guidance.
    pub cfg_scale: f32,
    /// `None` draws one, and the result reports which.
    pub seed: Option<u64>,
    pub format: ImageFormat,
    /// A picture to start from, for image-to-image.
    pub init: Option<InitImage>,
}

#[derive(Debug, Clone)]
pub struct InitImage {
    /// PNG, JPEG or SVG bytes; resized to the request's size.
    pub bytes: Vec<u8>,
    /// How much of the schedule to run: `1.0` ignores the picture's content
    /// entirely (pure noise), `0.0` returns it unchanged.
    pub strength: f32,
}

#[derive(Debug, Clone)]
pub struct GeneratedImage {
    pub bytes: Vec<u8>,
    pub format: ImageFormat,
    pub width: usize,
    pub height: usize,
    pub seed: u64,
    pub steps: usize,
    pub elapsed: std::time::Duration,
    pub timings: Timings,
}

/// Where a generation's time went, phase by phase — what `orangu-bench
/// --image` reads to say which of the three models a slow picture is slow
/// in, and what the log line prints beside the total.
#[derive(Debug, Clone, Copy, Default)]
pub struct Timings {
    /// Encoding the prompt (and the negative one, under guidance) through
    /// the text encoder — plus, for image-to-image, encoding the attached
    /// picture through the VAE.
    pub encode: std::time::Duration,
    /// The denoising loop: every transformer pass, both prompts when
    /// guided, and the Euler steps between them.
    pub denoise: std::time::Duration,
    /// Unpacking the latent, the VAE decode, and writing the file format.
    pub decode: std::time::Duration,
    /// The denoising loop's transformer passes by operation class — see
    /// [`transformer::Stages`].
    pub stages: transformer::Stages,
}

/// Where a generation is, for a progress bar. `step` is `0` for the
/// announcement before the first step, whose `seconds_per_step` is the
/// estimate from [`Pipeline::seconds_per_step`].
#[derive(Debug, Clone, Copy)]
pub struct Progress {
    /// Steps finished so far, and the total the request will run.
    pub step: usize,
    pub steps: usize,
    /// Seconds per step so far, for an estimate of what is left.
    pub seconds_per_step: f64,
    /// Seconds of work still ahead of the first step — the step-0 event's
    /// estimate of the encode (an edit's is most of a step's worth); `0`
    /// once steps run.
    pub pending_seconds: f64,
}

impl Progress {
    /// The estimated seconds left.
    pub fn eta_seconds(&self) -> f64 {
        self.pending_seconds + self.seconds_per_step * self.steps.saturating_sub(self.step) as f64
    }
}

/// What the pipeline has learned about its own speed, for saying how long
/// a picture will take before it starts.
///
/// A transformer pass is not one rate: the linears cost the same per token
/// at every size, attention costs more per token the more tokens there are
/// (it is quadratic), so a rate measured on a 256-token picture is
/// optimistic for a 4,096-token one by nearly 2×. The model here is the
/// two parts kept apart — seconds per token pass for everything but
/// attention, and seconds per token pass for attention *at the size it was
/// measured*, scaled by the ratio of sizes — plus the VAE decode per pixel
/// and the prompt encode, which the steps' rate never covered. Each
/// finished picture replaces all of it from its own `Timings`; the startup
/// calibration seeds the linear part and takes the rest from this board's
/// measured shares.
#[derive(Debug, Clone, Copy)]
pub struct RateModel {
    /// Seconds per latent-token pass, attention excluded.
    pub linear_per_token_pass: f64,
    /// Seconds per latent-token pass spent in attention, measured at
    /// `attention_tokens` tokens; at `n` tokens it is `× n /
    /// attention_tokens`.
    pub attention_per_token_pass: f64,
    pub attention_tokens: f64,
    /// Seconds of VAE decode per output pixel.
    pub decode_per_pixel: f64,
    /// Seconds of prompt encoding per picture.
    pub encode: f64,
    /// Seconds of an edit's encode — vision tower, VAE encode, the prompts
    /// with the picture in them, the prefixes — per reference latent
    /// token, once an edit was measured.
    pub edit_encode_per_token: Option<f64>,
    /// The share of steps that ran the transformer in the last picture —
    /// the rest reused its passes (`image_cache`); `1` with the cache off.
    pub run_share: f64,
}

/// The share of steps expected to run the transformer before a picture
/// has measured it under `image_cache = easy`: 15–17 of 40 on this board
/// (`doc/PERF-IMAGE.md`, task 4), rounded up.
const RUN_SHARE_SEED: f64 = 0.45;

/// The fewest latent tokens a picture may have to teach the [`RateModel`]:
/// below this the pass is weight-bound and says nothing about the per-token
/// rate a real picture runs at.
const RATE_MODEL_MIN_TOKENS: usize = 64;

impl RateModel {
    /// From the startup calibration alone: its rate is the linears'; the
    /// rest are this board's shares at 256 tokens — attention a tenth of
    /// the linears' time there, the VAE 30 µs a pixel, the encoder two
    /// seconds — refined by the first picture.
    pub fn from_calibration(token_passes_per_second: f64) -> Self {
        let linear = 1.0 / token_passes_per_second.max(1e-9);
        Self {
            linear_per_token_pass: linear,
            attention_per_token_pass: 0.1 * linear,
            attention_tokens: 256.0,
            decode_per_pixel: 30e-6,
            encode: 2.0,
            edit_encode_per_token: None,
            run_share: RUN_SHARE_SEED,
        }
    }

    /// The share of a picture's steps that will run the transformer: `1`
    /// with the step cache off, the last picture's share with it on.
    pub fn step_share(&self) -> f64 {
        if stepcache::configured() == stepcache::ImageCache::Off {
            1.0
        } else {
            self.run_share.clamp(0.05, 1.0)
        }
    }

    /// Seconds one denoising step takes at `n_tokens` latent tokens and
    /// `passes_per_step` transformer passes.
    pub fn seconds_per_step(&self, n_tokens: usize, passes_per_step: f64) -> f64 {
        self.seconds_per_step_over(n_tokens, n_tokens, passes_per_step)
    }

    /// [`RateModel::seconds_per_step`] with each token attending to `keys`
    /// — an edit's picture tokens also read the reference's.
    pub fn seconds_per_step_over(&self, n_tokens: usize, keys: usize, passes_per_step: f64) -> f64 {
        let n = n_tokens as f64;
        let attention =
            self.attention_per_token_pass * keys as f64 / self.attention_tokens.max(1.0);
        n * passes_per_step * (self.linear_per_token_pass + attention)
    }

    /// Seconds of an edit's encode with a reference of `reference_tokens`
    /// latent tokens: as the last edit measured it, or before one, 1.7
    /// passes over the reference per prompt (this board at 1024²: 104 s
    /// against a 63 s step).
    pub fn edit_encode(&self, reference_tokens: usize, passes_per_step: f64) -> f64 {
        match self.edit_encode_per_token {
            Some(per_token) => per_token * reference_tokens as f64,
            None => 1.7 * self.seconds_per_step(reference_tokens, passes_per_step),
        }
    }

    /// The whole picture: encode, the steps, the decode.
    pub fn seconds_for(&self, width: usize, height: usize, steps: usize, passes: f64) -> f64 {
        let n_tokens = (width / 16) * (height / 16);
        self.encode
            + self.seconds_per_step(n_tokens, passes) * steps as f64 * self.step_share()
            + self.decode_per_pixel * (width * height) as f64
    }

    /// Latent-token passes per second at `n_tokens`, for `/props`.
    pub fn token_passes_per_second(&self, n_tokens: usize) -> f64 {
        let per_step = self.seconds_per_step(n_tokens, 1.0);
        if per_step > 0.0 {
            n_tokens as f64 / per_step
        } else {
            0.0
        }
    }
}

impl Pipeline {
    /// The latent tokens of the reference an attached picture would be
    /// edited from — `0` when it would not be (no vision tower, or not
    /// readable), as the step-0 estimate counts it.
    fn reference_tokens(&self, request: &ImageRequest) -> usize {
        let Some(init) = request.init.as_ref().filter(|_| self.edits()) else {
            return 0;
        };
        codec::dimensions(&init.bytes).map_or(0, |dims| {
            let (w, h) = reference_size(dims, request.width * request.height);
            (w / 16) * (h / 16)
        })
    }

    /// The current [`RateModel`], if any.
    pub fn rate_model(&self) -> Option<RateModel> {
        *self
            .rate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The rate at the defaults' size, latent-token passes per second, for
    /// `/props`.
    pub fn token_passes_per_second(&self) -> Option<f64> {
        let d = self.defaults();
        self.rate_model()
            .map(|m| m.token_passes_per_second((d.width / 16) * (d.height / 16)))
    }

    /// The estimated wall time of a whole picture at the server defaults
    /// — the wait the console's user is looking at — or `None` before any
    /// rate is known.
    pub fn estimated_default_seconds(&self) -> Option<f64> {
        let d = self.defaults();
        let passes = if d.cfg_scale > 1.0 { 2.0 } else { 1.0 };
        self.rate_model()
            .map(|m| m.seconds_for(d.width, d.height, d.steps, passes))
    }
}

/// The transformer and VAE of one [`Variant`]. There is one per server, so
/// the variants' difference in size costs nothing.
#[allow(clippy::large_enum_variant)]
enum Model {
    QwenImage {
        transformer: QwenImageTransformer,
        vae: QwenImageVae,
    },
    QwenImage21 {
        transformer: QwenImage21Transformer,
        vae: QwenImage21Vae,
    },
}

/// A prompt as the transformer reads it: Qwen-Image's text-encoder hidden
/// states, re-read every step, or Qwen-Image 2.1's prompt already run
/// through every block (see [`transformer21`]).
enum Condition {
    Hidden(Vec<f32>),
    Prefix(transformer21::Prefix),
}

/// Qwen3-VL's vision tower and the text model that reads its tokens — what
/// Qwen-Image 2.1 edits with.
struct Vision {
    tower: qwen3vl::VisionTower,
    text: qwen3vl::TextModel,
}

/// A picture Qwen-Image 2.1 edits, prepared once per request: as the
/// vision tower read it, and as VAE latents for the transformer.
struct Reference {
    vision: qwen3vl::VisionOutput,
    /// `[rows * cols, 64]`, normalised, row-major.
    latents: Vec<f32>,
    rows: usize,
    cols: usize,
}

/// The size a reference picture is read at: its own proportions at the
/// picture's area, sides rounded to 32 — diffusers' `calculate_dimensions`,
/// at the target's area rather than its fixed 1024², so a small (fast)
/// picture reads a small reference.
pub fn reference_size((width, height): (usize, usize), area: usize) -> (usize, usize) {
    let area = match reference_cap() {
        ReferenceCap::Output => area,
        // Its own pixels at most — upsampling it adds none — though never
        // below 256², where the vision tower's grid grows thin.
        ReferenceCap::Source => area.min((width * height).max(256 * 256)),
        ReferenceCap::Area(cap) => area.min(cap),
    };
    let ratio = width.max(1) as f64 / height.max(1) as f64;
    let w = (area as f64 * ratio).sqrt();
    let h = w / ratio;
    let round = |v: f64| ((v / 32.0).round().max(1.0) as usize) * 32;
    (round(w), round(h))
}

/// `[orangu-server].image_reference_size`: the area an edit's reference
/// is read at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReferenceCap {
    /// The picture's own area (diffusers reads 1024² whatever it gets).
    Output,
    /// The picture's area, but no more than the attached picture's own —
    /// the default: upsampling a reference adds no information, only
    /// prefix tokens and keys in every step (`doc/PERF-IMAGE.md`, task 8).
    #[default]
    Source,
    /// The picture's area, but no more than this many pixels.
    Area(usize),
}

impl ReferenceCap {
    /// `output`, `source`, or `WIDTHxHEIGHT`.
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "output" => Some(Self::Output),
            "source" => Some(Self::Source),
            other => crate::config::parse_image_size(other).map(|(w, h)| Self::Area(w * h)),
        }
    }
}

static REFERENCE_CAP: std::sync::Mutex<ReferenceCap> = std::sync::Mutex::new(ReferenceCap::Source);

/// Set once by `main`.
pub fn set_reference_cap(cap: ReferenceCap) {
    *REFERENCE_CAP.lock().unwrap_or_else(|p| p.into_inner()) = cap;
}

/// The cap, `ORANGU_IMAGE_REFERENCE` over the configuration.
fn reference_cap() -> ReferenceCap {
    std::env::var("ORANGU_IMAGE_REFERENCE")
        .ok()
        .and_then(|v| ReferenceCap::parse(&v))
        .unwrap_or_else(|| *REFERENCE_CAP.lock().unwrap_or_else(|p| p.into_inner()))
}

pub struct Pipeline {
    model: Model,
    pub variant: Variant,
    /// Present when a Qwen-Image 2.1 model has its vision projector: an
    /// attached picture is then edited.
    vision: Option<Vision>,
    /// The text encoder — the server's `ModelForward`, shared with the
    /// `Engine` that also answers embeddings with it.
    encoder: Arc<dyn ModelForward>,
    tokenizer: Arc<Tokenizer>,
    /// What a request gets when it does not say — the config's picture
    /// keys as the server came up, and since then whatever `POST /props`
    /// (the console's Image settings) last set. Read with [`Pipeline::
    /// defaults`], a clone: a request is built from one consistent set.
    defaults: RwLock<ImageDefaults>,
    /// The defaults as configured — what a reset goes back to.
    pub configured: ImageDefaults,
    pub companions: Companions,
    /// The adapter the transformer carries, for `/props` — set by `prepare`
    /// after the load, which is where the file was resolved.
    pub adapter: Option<PathBuf>,
    /// One generation at a time — see the module doc.
    busy: Mutex<()>,
    /// Latent-token passes per second, as last measured: seeded from the
    /// startup calibration, replaced by every finished picture's own rate.
    /// What the step-0 progress event and `/props` announce a wait from.
    rate: Mutex<Option<RateModel>>,
}

impl Pipeline {
    // Eight parameters: the three models, the two companions' record, the
    // backend, the defaults, the seed rate and the adapter — each a thing
    // `prepare` resolved separately, and a struct for them would only move
    // the same eight names one line down.
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        transformer: &LoadedModel,
        companions: Companions,
        encoder: Arc<dyn ModelForward>,
        tokenizer: Arc<Tokenizer>,
        backend: Arc<dyn Backend>,
        defaults: ImageDefaults,
        rate: Option<RateModel>,
        lora: Option<lora::Lora>,
        merge_lora: bool,
        mut merged_cache: Option<lora::MergedCache>,
        vae_precision: vae::VaePrecision,
    ) -> Result<Self> {
        let variant =
            Variant::from_architecture(&transformer.config.architecture).ok_or_else(|| {
                anyhow::anyhow!(
                    "{} is not a picture model this pipeline serves",
                    transformer.config.architecture
                )
            })?;
        let (_, text_width) = variant.text_encoder();
        if variant == Variant::QwenImage21 {
            ensure!(
                lora.is_none(),
                "no adapter is published for {}: set [orangu-server].image_lora = none",
                variant.architecture()
            );
            let transformer = QwenImage21Transformer::load(transformer, backend.clone())
                .context("building the qwen_image_2_1 transformer")?;
            ensure!(
                transformer.config.txt_dim as u64 == text_width
                    && encoder.config().n_embd as u64 == text_width,
                "the transformer reads {}-wide text states but the text encoder produces {}",
                transformer.config.txt_dim,
                encoder.config().n_embd
            );
            let vae = QwenImage21Vae::load(&companions.vae, backend.clone(), vae_precision)
                .context("loading the VAE")?;
            let vision = match &companions.vision {
                Some(path) => {
                    let tower = qwen3vl::VisionTower::load(path, backend.clone())?;
                    ensure!(
                        tower.out_width as u64 == text_width,
                        "the vision projector writes {}-wide tokens for a {text_width}-wide \
                         text encoder",
                        tower.out_width
                    );
                    let weights = LoadedModel::open(&companions.text_encoder)
                        .context("mapping the text encoder for the vision path")?;
                    let text = qwen3vl::TextModel::load(&weights, backend)?;
                    Some(Vision { tower, text })
                }
                None => None,
            };
            let mut pipeline = Self::assemble(
                Model::QwenImage21 { transformer, vae },
                variant,
                encoder,
                tokenizer,
                defaults,
                companions,
                rate,
            );
            pipeline.vision = vision;
            return Ok(pipeline);
        }
        let started = Instant::now();
        let merging = lora.is_some() && merge_lora;
        let hit = merged_cache.as_ref().is_some_and(|c| c.is_hit());
        let transformer = QwenImageTransformer::load(
            transformer,
            backend.clone(),
            lora,
            merge_lora,
            merged_cache.as_mut(),
        )
        .context("building the qwen_image transformer")?;
        if merging {
            if hit {
                log::info!(
                    "orangu-server: [image] LoRA merge read from {} in {:.0} s",
                    merged_cache
                        .as_ref()
                        .map(|c| c.path().display().to_string())
                        .unwrap_or_default(),
                    started.elapsed().as_secs_f64()
                );
            } else {
                log::info!(
                    "orangu-server: [image] LoRA merged into the transformer's weights in {:.0} s",
                    started.elapsed().as_secs_f64()
                );
                if let Some(cache) = merged_cache.as_mut() {
                    let writing = Instant::now();
                    match cache.flush() {
                        Ok(bytes) if bytes > 0 => log::info!(
                            "orangu-server: [image] merged weights cached at {} ({}, {:.0} s) — \
                             the next start skips the merge",
                            cache.path().display(),
                            orangu::format::format_bytes(bytes),
                            writing.elapsed().as_secs_f64()
                        ),
                        Ok(_) => {}
                        Err(err) => log::warn!(
                            "orangu-server: [image] the merged weights could not be cached at {}: \
                             {err:#} — every start will merge again",
                            cache.path().display()
                        ),
                    }
                }
            }
        }
        ensure!(
            transformer.config.txt_dim as u64 == text_width
                && encoder.config().n_embd as u64 == text_width,
            "the transformer reads {}-wide text states but the text encoder produces {}",
            transformer.config.txt_dim,
            encoder.config().n_embd
        );
        let vae = QwenImageVae::load(&companions.vae, backend, vae_precision)
            .context("loading the VAE")?;
        Ok(Self::assemble(
            Model::QwenImage { transformer, vae },
            variant,
            encoder,
            tokenizer,
            defaults,
            companions,
            rate,
        ))
    }

    fn assemble(
        model: Model,
        variant: Variant,
        encoder: Arc<dyn ModelForward>,
        tokenizer: Arc<Tokenizer>,
        defaults: ImageDefaults,
        companions: Companions,
        rate: Option<RateModel>,
    ) -> Self {
        Self {
            model,
            variant,
            vision: None,
            encoder,
            tokenizer,
            configured: defaults.clone(),
            defaults: RwLock::new(defaults),
            companions,
            adapter: None,
            busy: Mutex::new(()),
            rate: Mutex::new(rate),
        }
    }

    pub fn transformer_config(&self) -> &transformer::TransformerConfig {
        match &self.model {
            Model::QwenImage { transformer, .. } => &transformer.config,
            Model::QwenImage21 { transformer, .. } => &transformer.config,
        }
    }

    /// See [`Variant::size_unit`].
    pub fn size_unit(&self) -> usize {
        self.variant.size_unit()
    }

    /// The current picture defaults.
    pub fn defaults(&self) -> ImageDefaults {
        self.defaults
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Replaces the picture defaults — `POST /props`. Validation is the
    /// caller's (`http::images::apply_settings`): what arrives here is a
    /// set a request could be built from.
    pub fn set_defaults(&self, defaults: ImageDefaults) {
        *self
            .defaults
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = defaults;
    }

    /// A prompt's conditioning: the text encoder's final hidden states for
    /// the prompt's own tokens, `[n, 3584]`.
    fn encode_prompt(&self, prompt: &str) -> Result<Vec<f32>> {
        let prefix = self.tokenizer.encode(PROMPT_TEMPLATE_PREFIX, false);
        let text = format!("{PROMPT_TEMPLATE_PREFIX}{prompt}{PROMPT_TEMPLATE_SUFFIX}");
        let mut tokens = self.tokenizer.encode(&text, false);
        ensure!(
            tokens.starts_with(&prefix),
            "the prompt template did not tokenize as a prefix of the prompt"
        );
        // diffusers truncates the whole rendered text to `max_length +
        // drop_idx`, so the suffix goes first on a long prompt.
        tokens.truncate(prefix.len() + MAX_PROMPT_TOKENS);
        let width = self.encoder.config().n_embd;
        let hidden = self
            .encoder
            .forward_hidden_states(&tokens)
            .context("running the text encoder")?;
        ensure!(
            hidden.len() == tokens.len() * width,
            "text encoder returned the wrong shape"
        );
        let mut out = hidden;
        out.drain(..prefix.len() * width);
        ensure!(!out.is_empty(), "the prompt encoded to no tokens");
        Ok(out)
    }

    /// Qwen-Image 2.1's conditioning: the Qwen3-VL encoder's last decoder
    /// layer, *before* its final norm, for everything after the system turn
    /// — `[n, 4096]`. A prompt longer than [`MAX_PROMPT_TOKENS`] is cut there
    /// (diffusers reads any length; this bounds a pass's memory).
    fn encode_prompt21(&self, prompt: &str) -> Result<Vec<f32>> {
        let system = self.tokenizer.encode(PROMPT21_SYSTEM, false);
        let text =
            format!("{PROMPT21_SYSTEM}{PROMPT21_USER_PREFIX}{prompt}{PROMPT_TEMPLATE_SUFFIX}");
        let mut tokens = self.tokenizer.encode(&text, false);
        ensure!(
            tokens.starts_with(&system),
            "the prompt template did not tokenize as a prefix of the prompt"
        );
        tokens.truncate(system.len() + MAX_PROMPT_TOKENS);
        let width = self.encoder.config().n_embd;
        let mut hidden = self
            .encoder
            .forward_hidden_states_pre_norm(&tokens)
            .context("running the text encoder")?;
        ensure!(
            hidden.len() == tokens.len() * width,
            "text encoder returned the wrong shape"
        );
        hidden.drain(..system.len() * width);
        ensure!(!hidden.is_empty(), "the prompt encoded to no tokens");
        Ok(hidden)
    }

    /// A prompt as the transformer reads it (see [`Condition`]).
    fn condition(&self, prompt: &str, cancel: &AtomicBool) -> Result<Condition> {
        match &self.model {
            Model::QwenImage { .. } => Ok(Condition::Hidden(self.encode_prompt(prompt)?)),
            Model::QwenImage21 { transformer, .. } => {
                let hidden = self.encode_prompt21(prompt)?;
                Ok(Condition::Prefix(
                    transformer.prefill(&hidden, Some(cancel))?,
                ))
            }
        }
    }

    /// Whether an attached picture is edited — read by the text encoder's
    /// vision tower and placed in the transformer's prompt — rather than
    /// drawn over from a noised copy of it (`strength`).
    pub fn edits(&self) -> bool {
        self.vision.is_some()
    }

    /// An attached picture prepared for editing a `(width, height)`
    /// picture: read by the vision tower, and encoded (the posterior's
    /// mean, as diffusers takes it) to latents.
    fn prepare_reference(
        &self,
        bytes: &[u8],
        width: usize,
        height: usize,
        cancel: &AtomicBool,
    ) -> Result<Reference> {
        let (Some(vision), Model::QwenImage21 { vae, .. }) = (&self.vision, &self.model) else {
            bail!("this picture model does not edit pictures");
        };
        let dims = codec::dimensions(bytes).context("reading the attached picture")?;
        let (w, h) = reference_size(dims, width * height);
        // The encoder sees the picture on white, as the processor flattens
        // an RGBA one; the VAE reads all four channels.
        let rgb = codec::decode_to_feature(bytes, w, h).context("reading the attached picture")?;
        let started = Instant::now();
        let seen = vision
            .tower
            .encode(&rgb, Some(cancel))
            .context("reading the attached picture through the vision tower")?;
        let tower = started.elapsed();
        let rgba =
            codec::decode_to_rgba_feature(bytes, w, h).context("reading the attached picture")?;
        let (mean, _) = vae
            .encode_unless(&rgba, Some(cancel))
            .context("encoding the attached picture")?;
        log::info!(
            "orangu-server: [image] reference {w}x{h}: vision tower {:.1}s, VAE encode {:.1}s",
            tower.as_secs_f64(),
            (started.elapsed() - tower).as_secs_f64()
        );
        let mut latents = mean.data;
        normalize_latent21(&mut latents);
        Ok(Reference {
            vision: seen,
            latents,
            rows: h / vae21::SPATIAL_COMPRESSION,
            cols: w / vae21::SPATIAL_COMPRESSION,
        })
    }

    /// Qwen-Image 2.1's conditioning for an edit: diffusers'
    /// `prompt_template_ti2i` — the picture's `<|image_pad|>` run inside
    /// the user turn — through the text encoder with the vision tokens in
    /// it, then through the transformer with the picture's latents where
    /// the encoder read it.
    fn condition_edit(
        &self,
        prompt: &str,
        reference: &Reference,
        cancel: &AtomicBool,
    ) -> Result<Condition> {
        let (Some(vision), Model::QwenImage21 { transformer, .. }) = (&self.vision, &self.model)
        else {
            bail!("this picture model does not edit pictures");
        };
        let system = self.tokenizer.encode(PROMPT21_SYSTEM, false);
        let text = format!(
            "{PROMPT21_SYSTEM}{PROMPT21_USER_PREFIX}<image1><|vision_start|><|image_pad|>\
             <|vision_end|>{prompt}{PROMPT_TEMPLATE_SUFFIX}"
        );
        let mut tokens = self.tokenizer.encode(&text, false);
        ensure!(
            tokens.starts_with(&system),
            "the prompt template did not tokenize as a prefix of the prompt"
        );
        let pad = self.tokenizer.encode("<|image_pad|>", false);
        ensure!(
            pad.len() == 1,
            "the text encoder's tokenizer has no <|image_pad|> token"
        );
        let at = tokens
            .iter()
            .position(|&t| t == pad[0])
            .context("the prompt template lost its picture")?;
        let (rows, cols) = reference.vision.grid;
        let slots = rows * cols;
        ensure!(
            slots * 4 == reference.rows * reference.cols,
            "the vision tower read {slots} cells of a {}x{} latent",
            reference.rows,
            reference.cols
        );
        tokens.splice(at..at + 1, std::iter::repeat_n(pad[0], slots));
        let started = Instant::now();
        let hidden = vision.text.forward_pre_norm(
            &tokens,
            &[qwen3vl::PromptPicture {
                start: at,
                vision: &reference.vision,
            }],
            Some(cancel),
        )?;
        let width = self.encoder.config().n_embd;
        ensure!(
            hidden.len() == tokens.len() * width,
            "text encoder returned the wrong shape"
        );
        let mut segments = Vec::new();
        if at > system.len() {
            segments.push(transformer21::Segment::Text(
                &hidden[system.len() * width..at * width],
            ));
        }
        segments.push(transformer21::Segment::Picture {
            latents: &reference.latents,
            rows: reference.rows,
            cols: reference.cols,
        });
        if at + slots < tokens.len() {
            segments.push(transformer21::Segment::Text(
                &hidden[(at + slots) * width..],
            ));
        }
        let text = started.elapsed();
        let prefix = transformer.prefill_segments(&segments, Some(cancel))?;
        log::info!(
            "orangu-server: [image] edit prompt: text encoder {:.1}s over {} tokens, \
             transformer prefix {:.1}s over {} tokens",
            text.as_secs_f64(),
            tokens.len(),
            (started.elapsed() - text).as_secs_f64(),
            prefix.n_txt
        );
        Ok(Condition::Prefix(prefix))
    }

    fn take_stages(&self) -> transformer::Stages {
        match &self.model {
            Model::QwenImage { transformer, .. } => transformer.take_stages(),
            Model::QwenImage21 { transformer, .. } => transformer.take_stages(),
        }
    }

    /// One velocity prediction for `latents` under `condition`.
    fn velocity(
        &self,
        latents: &[f32],
        grid: (usize, usize),
        condition: &Condition,
        sigma: f32,
        cancel: &AtomicBool,
    ) -> Result<Vec<f32>> {
        match (&self.model, condition) {
            (Model::QwenImage { transformer, .. }, Condition::Hidden(txt)) => {
                transformer.forward(&ForwardInput {
                    img: latents,
                    grid,
                    txt,
                    sigma,
                    cancel: Some(cancel),
                })
            }
            (Model::QwenImage21 { transformer, .. }, Condition::Prefix(prefix)) => transformer
                .forward(&transformer21::ForwardInput {
                    img: latents,
                    grid,
                    prefix,
                    sigma,
                    cancel: Some(cancel),
                }),
            _ => bail!("a prompt encoded for one picture model was given to the other"),
        }
    }

    /// Classifier-free guidance of `positive` by `negative`: Qwen-Image's
    /// with its norm rescale ([`guide`]), Qwen-Image 2.1's plain.
    fn guide(&self, positive: &mut [f32], negative: &[f32], scale: f32, token_width: usize) {
        match self.variant {
            Variant::QwenImage => guide(positive, negative, scale, token_width),
            Variant::QwenImage21 => {
                for (p, n) in positive.iter_mut().zip(negative) {
                    *p = n + scale * (*p - n);
                }
            }
        }
    }

    /// The clean latent tokens of an attached picture at `(width,
    /// height)`: encoded, sampled from the posterior as diffusers draws it,
    /// normalised, and laid out as the transformer's tokens.
    fn latent_from_picture(
        &self,
        bytes: &[u8],
        width: usize,
        height: usize,
        noise: &mut Noise,
        cancel: &AtomicBool,
    ) -> Result<Vec<f32>> {
        let sample = |mean: Feature, logvar: Feature, noise: &mut Noise| -> Vec<f32> {
            let mut z = mean.data;
            for ((v, lv), n) in z
                .iter_mut()
                .zip(&logvar.data)
                .zip(noise.normals(logvar.data.len()))
            {
                *v += (0.5 * lv.clamp(-30.0, 20.0)).exp() * n;
            }
            z
        };
        match &self.model {
            Model::QwenImage { vae, .. } => {
                let rgb = codec::decode_to_feature(bytes, width, height)
                    .context("reading the attached picture")?;
                let (mean, logvar) = vae
                    .encode_unless(&rgb, Some(cancel))
                    .context("encoding the attached picture")?;
                let mut z = sample(mean, logvar, noise);
                normalize_latent(&mut z);
                Ok(pack_latent(
                    &z,
                    height / SPATIAL_COMPRESSION,
                    width / SPATIAL_COMPRESSION,
                ))
            }
            Model::QwenImage21 { vae, .. } => {
                let rgba = codec::decode_to_rgba_feature(bytes, width, height)
                    .context("reading the attached picture")?;
                let (mean, logvar) = vae
                    .encode_unless(&rgba, Some(cancel))
                    .context("encoding the attached picture")?;
                let mut z = sample(mean, logvar, noise);
                normalize_latent21(&mut z);
                Ok(z)
            }
        }
    }

    /// The picture a finished latent decodes to: RGB for Qwen-Image, RGBA
    /// for Qwen-Image 2.1, in `[-1, 1]`.
    fn picture_from_latent(
        &self,
        latents: &[f32],
        width: usize,
        height: usize,
        cancel: &AtomicBool,
    ) -> Result<Feature> {
        match &self.model {
            Model::QwenImage { vae, .. } => {
                let (h8, w8) = (height / SPATIAL_COMPRESSION, width / SPATIAL_COMPRESSION);
                let mut z = unpack_latent(latents, h8, w8);
                denormalize_latent(&mut z);
                vae.decode_unless(&Feature::new(h8, w8, Z_DIM, z), Some(cancel))
            }
            Model::QwenImage21 { vae, .. } => {
                let (h16, w16) = (
                    height / vae21::SPATIAL_COMPRESSION,
                    width / vae21::SPATIAL_COMPRESSION,
                );
                let mut z = latents.to_vec();
                denormalize_latent21(&mut z);
                vae.decode_unless(&Feature::new(h16, w16, vae21::Z_DIM, z), Some(cancel))
            }
        }
    }

    /// Runs one request to a finished picture, reporting after every step
    /// and stopping early (with an error) once `cancel` is set.
    pub fn generate(
        &self,
        request: &ImageRequest,
        progress: &mut dyn FnMut(Progress),
        cancel: &AtomicBool,
    ) -> Result<GeneratedImage> {
        let _turn = self
            .busy
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let started = Instant::now();
        let unit = self.size_unit();
        ensure!(
            request.width >= unit
                && request.height >= unit
                && request.width.is_multiple_of(unit)
                && request.height.is_multiple_of(unit),
            "image size must be a multiple of {unit} pixels on each side (got {}x{})",
            request.width,
            request.height
        );
        ensure!(request.steps > 0, "steps must be at least 1");
        ensure!(!request.prompt.trim().is_empty(), "the prompt is empty");
        // One token per 16 pixels along each side for both models: 2x2
        // eight-pixel cells packed, or one sixteen-pixel cell.
        let grid = (request.height / 16, request.width / 16);
        let n_tokens = grid.0 * grid.1;
        let seed = request.seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0)
        });

        let mut timings = Timings::default();
        // The wait, before any of it is spent: a step-0 progress event
        // carrying the per-step estimate from the last measured rate, so a
        // console can show "about 40 minutes" the moment the prompt is sent
        // rather than after the first step (which at 1024 pixels is itself
        // minutes away). `steps` here is the requested count; image-to-image
        // trims it below, and the real events from the loop take over.
        // An edit's picture tokens attend to the reference's as well, and
        // its encode runs the reference through the vision tower, the VAE
        // and every block before the first step.
        let passes_per_step = if request.cfg_scale > 1.0 { 2.0 } else { 1.0 };
        let reference_tokens = self.reference_tokens(request);
        let keys = n_tokens + reference_tokens;
        if let Some(rate) = self.rate_model() {
            progress(Progress {
                step: 0,
                steps: request.steps,
                // A reused step costs next to nothing: the share that runs.
                seconds_per_step: rate.seconds_per_step_over(n_tokens, keys, passes_per_step)
                    * rate.step_share(),
                pending_seconds: if reference_tokens > 0 {
                    rate.edit_encode(reference_tokens, passes_per_step)
                } else {
                    rate.encode
                },
            });
        }
        // Whatever a previous, cancelled pass left behind is not this
        // picture's.
        let _ = self.take_stages();
        // An attached picture under Qwen-Image 2.1 with its vision tower is
        // edited: read once here, placed in both prompts' prefixes, and the
        // picture drawn from noise over the whole schedule.
        let reference = match (&request.init, self.edits()) {
            (Some(init), true) => {
                Some(self.prepare_reference(&init.bytes, request.width, request.height, cancel)?)
            }
            _ => None,
        };
        let condition = |prompt: &str| -> Result<Condition> {
            match &reference {
                Some(reference) => self.condition_edit(prompt, reference, cancel),
                None => self.condition(prompt, cancel),
            }
        };
        let positive = condition(&request.prompt)?;
        let guided = request.cfg_scale > 1.0;
        let negative = if guided {
            let text = request
                .negative_prompt
                .clone()
                .unwrap_or_else(|| self.defaults().negative_prompt);
            // An empty negative prompt still encodes to something: the
            // template frames it. A single space is what Qwen's own examples
            // pass, so a blank stays a blank rather than an error.
            let text = if text.is_empty() {
                " ".to_string()
            } else {
                text
            };
            Some(condition(&text)?)
        } else {
            None
        };
        // Qwen-Image 2.1's prompt pass is encoding, not a step: out of the
        // steps' account.
        let _ = self.take_stages();
        if cancelled(Some(cancel)) {
            bail!("cancelled");
        }

        let sigmas = scheduler::sigmas(&ScheduleConfig::QWEN_IMAGE, request.steps, n_tokens);
        let mut noise = Noise::seeded(seed);
        let token_width = self.transformer_config().in_channels;
        let mut latents = noise.normals(n_tokens * token_width);
        let mut first_step = 0;
        if let Some(init) = request.init.as_ref().filter(|_| reference.is_none()) {
            let clean = self.latent_from_picture(
                &init.bytes,
                request.width,
                request.height,
                &mut noise,
                cancel,
            )?;
            first_step = scheduler::first_step_for_strength(request.steps, init.strength);
            if first_step >= request.steps {
                // Strength 0: nothing to run; the picture comes straight back
                // through the VAE.
                latents = clean;
            } else {
                latents = scheduler::scale_noise(&clean, &latents, sigmas[first_step]);
            }
        }

        let steps_to_run = request.steps - first_step;
        timings.encode = started.elapsed();
        let loop_started = Instant::now();
        // Steps that may reuse the last pass's residual (task 4): the
        // guided velocity is the cached function, so a reused step skips
        // both passes.
        let mut cache = stepcache::StepCache::new(stepcache::configured(), steps_to_run);
        for (done, i) in (first_step..request.steps).enumerate() {
            if cancelled(Some(cancel)) {
                bail!("cancelled");
            }
            let sigma = sigmas[i];
            let reused = cache.as_mut().and_then(|c| c.reuse(done, &latents));
            let velocity = match reused {
                Some(velocity) => velocity,
                None => {
                    let mut velocity = self.velocity(&latents, grid, &positive, sigma, cancel)?;
                    if let Some(negative) = &negative {
                        let unguided = self.velocity(&latents, grid, negative, sigma, cancel)?;
                        self.guide(&mut velocity, &unguided, request.cfg_scale, token_width);
                    }
                    if let Some(cache) = cache.as_mut() {
                        cache.record(done, &latents, &velocity);
                    }
                    velocity
                }
            };
            scheduler::euler_step(&mut latents, &velocity, sigma, sigmas[i + 1]);
            progress(Progress {
                step: done + 1,
                steps: steps_to_run,
                seconds_per_step: loop_started.elapsed().as_secs_f64() / (done + 1) as f64,
                pending_seconds: 0.0,
            });
        }

        timings.denoise = loop_started.elapsed();
        let skipped = cache.as_ref().map_or(0, |c| c.skipped);
        if skipped > 0 {
            log::info!(
                "orangu-server: [image] step cache: {skipped} of {steps_to_run} step(s) reused \
                 the last pass"
            );
        }
        timings.stages = self.take_stages();
        // The measured rates, for the next picture's estimate — the
        // denoising loop split into attention and the rest by the stage
        // account, the decode per pixel, the encode as it was.
        // Only from a picture big enough to be compute-bound: a 16-pixel
        // warmup is one token, whose pass is a read of every weight — a
        // per-token cost 200× the real one, which once extrapolated to a
        // 256-pixel picture as an hour and a half.
        if steps_to_run > 0
            && timings.denoise.as_secs_f64() > 0.0
            && n_tokens >= RATE_MODEL_MIN_TOKENS
        {
            let passed = steps_to_run - skipped;
            let token_passes = (n_tokens * passed.max(1)) as f64 * passes_per_step;
            let attention = timings.stages.attention.as_secs_f64();
            let linear = (timings.denoise.as_secs_f64() - attention).max(0.0);
            let mut rate = self
                .rate
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let previous = *rate;
            // An edit's encode is its own figure: the text-to-image one
            // stays what the last plain picture measured.
            let edited = reference.as_ref().map(|r| r.rows * r.cols);
            let encode = timings.encode.as_secs_f64();
            *rate = Some(RateModel {
                linear_per_token_pass: linear / token_passes,
                attention_per_token_pass: attention / token_passes,
                attention_tokens: (n_tokens + edited.unwrap_or(0)) as f64,
                decode_per_pixel: timings.decode.as_secs_f64()
                    / (request.width * request.height) as f64,
                encode: match (edited, previous) {
                    (Some(_), Some(p)) => p.encode,
                    _ => encode,
                },
                edit_encode_per_token: match edited {
                    Some(tokens) if tokens > 0 => Some(encode / tokens as f64),
                    _ => previous.and_then(|p| p.edit_encode_per_token),
                },
                run_share: match cache.as_ref() {
                    Some(_) => passed as f64 / steps_to_run as f64,
                    None => previous.map_or(RUN_SHARE_SEED, |p| p.run_share),
                },
            });
        }

        let decode_started = Instant::now();
        let picture = self
            .picture_from_latent(&latents, request.width, request.height, cancel)
            .context("decoding the picture")?;
        let bytes = codec::encode(&picture, request.format)?;
        timings.decode = decode_started.elapsed();
        log::info!(
            "orangu-server: [image] {}x{} in {:.1}s: encode {:.1}s, {} step(s) {:.1}s ({:.1}s each), \
             decode {:.1}s",
            request.width,
            request.height,
            started.elapsed().as_secs_f64(),
            timings.encode.as_secs_f64(),
            steps_to_run,
            timings.denoise.as_secs_f64(),
            timings.denoise.as_secs_f64() / steps_to_run.max(1) as f64,
            timings.decode.as_secs_f64(),
        );
        let st = &timings.stages;
        let pct =
            |d: std::time::Duration| 100.0 * d.as_secs_f64() / st.total().as_secs_f64().max(1e-9);
        log::info!(
            "orangu-server: [image] {} transformer pass(es): modulation {:.0}%, qkv {:.0}%, \
             attention {:.0}%, out {:.0}%, mlp {:.0}%, other {:.0}%, lora {:.0}%",
            st.passes,
            pct(st.modulation),
            pct(st.qkv),
            pct(st.attention),
            pct(st.out),
            pct(st.mlp),
            pct(st.other),
            pct(st.lora),
        );
        Ok(GeneratedImage {
            bytes,
            format: request.format,
            width: request.width,
            height: request.height,
            seed,
            steps: steps_to_run,
            elapsed: started.elapsed(),
            timings,
        })
    }
}

/// What a background generation reports, in order: progress after every
/// step, then exactly one of `Done` or `Error`.
#[derive(Debug)]
pub enum ImageEvent {
    Progress(Progress),
    /// Boxed: the picture carries its bytes and its timings, and every
    /// progress event would otherwise be sized for them.
    Done(Box<GeneratedImage>),
    Error(String),
}

/// Set once the server is shutting down: every picture in flight stops at
/// its next block or VAE layer — seconds — rather than at its next step,
/// which at 1024 px is minutes the exit would otherwise wait for (the
/// runtime joins the blocking thread a picture runs on).
static SHUTDOWN: AtomicBool = AtomicBool::new(false);

/// Tells every running and future picture to stop.
pub fn request_shutdown() {
    SHUTDOWN.store(true, Ordering::Relaxed);
}

/// Whether the request's own flag, or the server's, says stop.
pub fn cancelled(cancel: Option<&AtomicBool>) -> bool {
    SHUTDOWN.load(Ordering::Relaxed) || cancel.is_some_and(|flag| flag.load(Ordering::Relaxed))
}

/// Stops a background generation when dropped — held by whoever is reading
/// its events, so a client that disconnects mid-picture stops the work at
/// the next step rather than running the remaining minutes for nobody.
pub struct CancelOnDrop(Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

impl Pipeline {
    /// Runs `request` on a blocking thread, streaming [`ImageEvent`]s back.
    /// Both HTTP endpoints and the web console read the same channel.
    pub fn spawn(
        self: &Arc<Self>,
        request: ImageRequest,
    ) -> (tokio::sync::mpsc::Receiver<ImageEvent>, CancelOnDrop) {
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let cancel = Arc::new(AtomicBool::new(false));
        let pipeline = self.clone();
        let flag = cancel.clone();
        tokio::task::spawn_blocking(move || {
            let progress_tx = tx.clone();
            let mut progress = |p: Progress| {
                // A full channel means the reader is slower than the steps;
                // dropping a progress frame is fine, the next one supersedes it.
                let _ = progress_tx.try_send(ImageEvent::Progress(p));
            };
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                pipeline.generate(&request, &mut progress, &flag)
            }));
            let event = match result {
                Ok(Ok(image)) => ImageEvent::Done(Box::new(image)),
                Ok(Err(err)) => ImageEvent::Error(format!("{err:#}")),
                Err(_) => ImageEvent::Error(
                    crate::panic_capture::take_last_panic_detail()
                        .unwrap_or_else(|| "image generation panicked".to_string()),
                ),
            };
            let _ = tx.blocking_send(event);
        });
        (rx, CancelOnDrop(cancel))
    }
}

/// Classifier-free guidance with Qwen-Image's norm rescale: the guided
/// velocity `neg + scale * (pos - neg)` is scaled back, per token, to the
/// length of the positive prediction.
fn guide(positive: &mut [f32], negative: &[f32], scale: f32, token_width: usize) {
    for (pos, neg) in positive
        .chunks_mut(token_width)
        .zip(negative.chunks(token_width))
    {
        let cond_norm = pos.iter().map(|v| v * v).sum::<f32>().sqrt();
        for (p, n) in pos.iter_mut().zip(neg) {
            *p = n + scale * (*p - n);
        }
        let noise_norm = pos.iter().map(|v| v * v).sum::<f32>().sqrt();
        if noise_norm > 0.0 {
            let k = cond_norm / noise_norm;
            for p in pos.iter_mut() {
                *p *= k;
            }
        }
    }
}

/// `(z - mean) / std`, per channel, on a channel-last latent.
fn normalize_latent(z: &mut [f32]) {
    for px in z.chunks_mut(Z_DIM) {
        for ((v, m), s) in px.iter_mut().zip(&LATENTS_MEAN).zip(&LATENTS_STD) {
            *v = (*v - m) / s;
        }
    }
}

fn denormalize_latent(z: &mut [f32]) {
    for px in z.chunks_mut(Z_DIM) {
        for ((v, m), s) in px.iter_mut().zip(&LATENTS_MEAN).zip(&LATENTS_STD) {
            *v = *v * s + m;
        }
    }
}

/// [`normalize_latent`] for Qwen-Image 2.1's 64 channels.
fn normalize_latent21(z: &mut [f32]) {
    for px in z.chunks_mut(vae21::Z_DIM) {
        for ((v, m), s) in px
            .iter_mut()
            .zip(&vae21::LATENTS_MEAN)
            .zip(&vae21::LATENTS_STD)
        {
            *v = (*v - m) / s;
        }
    }
}

fn denormalize_latent21(z: &mut [f32]) {
    for px in z.chunks_mut(vae21::Z_DIM) {
        for ((v, m), s) in px
            .iter_mut()
            .zip(&vae21::LATENTS_MEAN)
            .zip(&vae21::LATENTS_STD)
        {
            *v = *v * s + m;
        }
    }
}

/// diffusers' `_pack_latents`: a channel-last `[h8 * w8, 16]` latent into
/// `[(h8/2) * (w8/2), 64]` tokens, each a 2x2 patch with features ordered
/// `(channel, dy, dx)`.
pub(crate) fn pack_latent(z: &[f32], h8: usize, w8: usize) -> Vec<f32> {
    debug_assert_eq!(z.len(), h8 * w8 * Z_DIM);
    let (rows, cols) = (h8 / 2, w8 / 2);
    let mut out = vec![0.0f32; rows * cols * Z_DIM * 4];
    for r in 0..rows {
        for q in 0..cols {
            let token = &mut out[(r * cols + q) * Z_DIM * 4..(r * cols + q + 1) * Z_DIM * 4];
            for c in 0..Z_DIM {
                for dy in 0..2 {
                    for dx in 0..2 {
                        let px = (2 * r + dy) * w8 + (2 * q + dx);
                        token[c * 4 + dy * 2 + dx] = z[px * Z_DIM + c];
                    }
                }
            }
        }
    }
    out
}

/// The inverse of [`pack_latent`].
pub(crate) fn unpack_latent(tokens: &[f32], h8: usize, w8: usize) -> Vec<f32> {
    let (rows, cols) = (h8 / 2, w8 / 2);
    debug_assert_eq!(tokens.len(), rows * cols * Z_DIM * 4);
    let mut z = vec![0.0f32; h8 * w8 * Z_DIM];
    for r in 0..rows {
        for q in 0..cols {
            let token = &tokens[(r * cols + q) * Z_DIM * 4..(r * cols + q + 1) * Z_DIM * 4];
            for c in 0..Z_DIM {
                for dy in 0..2 {
                    for dx in 0..2 {
                        let px = (2 * r + dy) * w8 + (2 * q + dx);
                        z[px * Z_DIM + c] = token[c * 4 + dy * 2 + dx];
                    }
                }
            }
        }
    }
    z
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rate model scales attention with the token count and the rest
    /// linearly: a step at four times the tokens of the measurement costs
    /// four times the linears and sixteen times the attention, and the
    /// whole picture adds the encode and a per-pixel decode.
    #[test]
    fn the_rate_model_grows_attention_with_the_token_count() {
        let m = RateModel {
            linear_per_token_pass: 0.02,
            attention_per_token_pass: 0.004,
            attention_tokens: 1024.0,
            decode_per_pixel: 30e-6,
            encode: 2.0,
            edit_encode_per_token: None,
            run_share: 1.0,
        };
        let at_1024 = m.seconds_per_step(1024, 1.0);
        assert!((at_1024 - 1024.0 * 0.024).abs() < 1e-9);
        let at_4096 = m.seconds_per_step(4096, 1.0);
        assert!((at_4096 - 4096.0 * (0.02 + 0.016)).abs() < 1e-6);
        assert!(at_4096 > 4.0 * at_1024);
        // Guidance doubles the passes, and the picture adds encode and decode.
        let picture = m.seconds_for(1024, 1024, 8, 2.0);
        assert!((picture - (2.0 + 8.0 * 2.0 * at_4096 + 30e-6 * 1024.0 * 1024.0)).abs() < 1e-3);
        // Seeded from a calibration alone, the shape is the same.
        let seeded = RateModel::from_calibration(40.0);
        assert!((seeded.linear_per_token_pass - 0.025).abs() < 1e-9);
        assert!(seeded.seconds_per_step(256, 1.0) > 256.0 * 0.025);
        // An edit's picture tokens read the reference too: attention over
        // twice the keys, the linears unchanged.
        let edit = m.seconds_per_step_over(4096, 8192, 1.0);
        assert!((edit - 4096.0 * (0.02 + 0.032)).abs() < 1e-6);
        // Its encode: 1.7 passes over the reference until one is measured.
        assert!((m.edit_encode(4096, 1.0) - 1.7 * at_4096).abs() < 1e-6);
        let measured = RateModel {
            edit_encode_per_token: Some(0.025),
            ..m
        };
        assert!((measured.edit_encode(4096, 1.0) - 102.4).abs() < 1e-9);
    }

    #[test]
    fn packing_is_a_bijection_with_the_diffusers_feature_order() {
        let (h8, w8) = (4, 6);
        let z: Vec<f32> = (0..h8 * w8 * Z_DIM).map(|i| i as f32).collect();
        let tokens = pack_latent(&z, h8, w8);
        assert_eq!(tokens.len(), (h8 / 2) * (w8 / 2) * 64);
        // Token (0,0), channel c=1, dy=1, dx=0 — feature c*4 + dy*2 + dx —
        // is pixel (y=1, x=0) channel 1.
        let (c, dy, dx, y, x) = (1, 1, 0, 1, 0);
        assert_eq!(tokens[c * 4 + dy * 2 + dx], z[(y * w8 + x) * Z_DIM + c]);
        // Token (row 1, col 2) of the 2x3 grid sits at pixels (2..4, 4..6).
        let (row, col, cols) = (1, 2, 3);
        let t = &tokens[(row * cols + col) * 64..(row * cols + col + 1) * 64];
        assert_eq!(t[5 * 4 + 3], z[(3 * w8 + 5) * Z_DIM + 5]);
        assert_eq!(unpack_latent(&tokens, h8, w8), z);
    }

    #[test]
    fn guidance_keeps_the_positive_predictions_length() {
        let mut pos = vec![3.0f32, 4.0, 0.0, 0.0];
        let neg = vec![0.0f32, 0.0, 0.0, 0.0];
        guide(&mut pos, &neg, 4.0, 4);
        // 4x the positive, scaled back to length 5: the same vector.
        assert!((pos[0] - 3.0).abs() < 1e-5 && (pos[1] - 4.0).abs() < 1e-5);
        let mut pos = vec![1.0f32, 0.0];
        let neg = vec![0.0f32, 1.0];
        guide(&mut pos, &neg, 2.0, 2);
        // neg + 2 (pos - neg) = (2, -1), length sqrt(5), scaled to 1.
        let len = (pos[0] * pos[0] + pos[1] * pos[1]).sqrt();
        assert!((len - 1.0).abs() < 1e-5);
        assert!(pos[0] > 0.0 && pos[1] < 0.0);
    }

    #[test]
    fn latent_normalisation_round_trips() {
        let mut z: Vec<f32> = (0..Z_DIM * 2).map(|i| i as f32 * 0.25 - 1.0).collect();
        let original = z.clone();
        normalize_latent(&mut z);
        assert!((z[0] - (original[0] - LATENTS_MEAN[0]) / LATENTS_STD[0]).abs() < 1e-6);
        denormalize_latent(&mut z);
        for (a, b) in z.iter().zip(&original) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    /// One small picture end to end on the CPU: prompt in, PNG out. Slow
    /// (minutes: two transformer passes over 20B parameters at 256 tokens)
    /// and needing three real files, so ignored by default.
    ///
    /// Run with `ORANGU_TEST_QWEN_IMAGE_MODEL=/path/to/qwen-image-2512-Q4_K_M.gguf
    /// cargo test --release --bin orangu-server a_small_picture -- --ignored`
    /// — or a `qwen-image-2.1-*.gguf`, whose companions are its own; the
    /// encoder and VAE are found under `ORANGU_TEST_QWEN_IMAGE_MODELS_DIR`,
    /// or the hub-cache root the model sits in. `ORANGU_TEST_QWEN_IMAGE_STEPS`
    /// and `ORANGU_TEST_QWEN_IMAGE_PROMPT` change the two steps and the
    /// prompt, for a look at a real picture.
    #[test]
    #[ignore]
    fn a_small_picture_is_generated_end_to_end_on_the_cpu() {
        let model = std::env::var("ORANGU_TEST_QWEN_IMAGE_MODEL")
            .expect("set ORANGU_TEST_QWEN_IMAGE_MODEL");
        let model = PathBuf::from(model);
        let models_dir = std::env::var("ORANGU_TEST_QWEN_IMAGE_MODELS_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                // `<models>/models--x--y/snapshots/<commit>/<file>`.
                model
                    .ancestors()
                    .nth(4)
                    .expect("the model sits in a hub-cache layout")
                    .to_path_buf()
            });
        let transformer = LoadedModel::open(&model).expect("transformer");
        let variant =
            Variant::from_architecture(&transformer.config.architecture).expect("a picture model");
        let companions =
            Companions::locate(&models_dir, variant, None, None, None).expect("companions");
        let backend: Arc<dyn Backend> = Arc::new(crate::engine::backend::CpuBackend);
        let encoder_loaded = LoadedModel::open(&companions.text_encoder).expect("encoder");
        let encoder: Arc<dyn ModelForward> = Arc::new(
            crate::engine::arch::llama::LlamaModel::load_with_backend(
                &encoder_loaded,
                backend.clone(),
            )
            .expect("build encoder"),
        );
        let gguf = orangu::gguf::GgufFile::open(&companions.text_encoder).expect("encoder gguf");
        let tokenizer = Arc::new(Tokenizer::from_gguf(&gguf).expect("tokenizer"));
        let pipeline = Pipeline::load(
            &transformer,
            companions,
            encoder,
            tokenizer,
            backend,
            ImageDefaults::default(),
            None,
            None,
            true,
            None,
            vae::VaePrecision::default(),
        )
        .expect("pipeline");

        let steps = std::env::var("ORANGU_TEST_QWEN_IMAGE_STEPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2);
        let mut steps_seen = 0;
        let image = pipeline
            .generate(
                &ImageRequest {
                    prompt: std::env::var("ORANGU_TEST_QWEN_IMAGE_PROMPT")
                        .unwrap_or_else(|_| "a red circle on a white background".into()),
                    negative_prompt: None,
                    width: 256,
                    height: 256,
                    steps,
                    cfg_scale: 1.0,
                    seed: Some(1),
                    format: ImageFormat::Png,
                    init: None,
                },
                &mut |p| {
                    steps_seen = p.step;
                    eprintln!(
                        "step {}/{} ({:.1}s per step)",
                        p.step, p.steps, p.seconds_per_step
                    );
                },
                &AtomicBool::new(false),
            )
            .expect("generate");
        assert_eq!(steps_seen, steps);
        assert_eq!((image.width, image.height), (256, 256));
        assert_eq!(codec::dimensions(&image.bytes).unwrap(), (256, 256));
        let out = std::env::temp_dir().join("orangu-qwen-image-test.png");
        std::fs::write(&out, &image.bytes).unwrap();
        eprintln!("wrote {} in {:?}", out.display(), image.elapsed);
        // A generated picture is not a constant: every channel must vary.
        let rgb = codec::decode_to_feature(&image.bytes, 256, 256).unwrap();
        for c in 0..3 {
            let values: Vec<f32> = rgb.data.iter().skip(c).step_by(3).copied().collect();
            let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            assert!(max - min > 0.05, "channel {c} is flat ({min}..{max})");
        }
    }

    #[test]
    fn the_prompt_template_is_diffusers_own() {
        // The exact string, since the encoder's hidden states depend on it
        // and a changed word would silently condition every picture on
        // different context.
        assert!(PROMPT_TEMPLATE_PREFIX.starts_with("<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"));
        assert_eq!(
            PROMPT_TEMPLATE_SUFFIX,
            "<|im_end|>\n<|im_start|>assistant\n"
        );
    }
}
