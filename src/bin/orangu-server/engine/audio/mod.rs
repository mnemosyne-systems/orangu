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

//! Text to speech with Qwen3-TTS (`qwen3tts`).
//!
//! Three files make a voice:
//!
//! - **The model** (`qwen3tts`): the *talker*, a Qwen3 decoder that reads
//!   the text and writes one frame of speech at a time as the first of the
//!   codec's sixteen codebooks, and the *code predictor*, a five-layer
//!   decoder that fills in the other fifteen for each frame from the
//!   talker's hidden state ([`stack`]).
//! - **The codec** (`qwen3tts_tokenizer`): sixteen codes a frame, twelve
//!   and a half frames a second, back to 24 kHz sound ([`codec`]).
//! - **A voice pack** (`qwen3tts.voicepack`): who is speaking — a speaker
//!   embedding, and a short reference recording as its transcript and its
//!   codec codes.
//!
//! The talker is prompted *in context* (the reference's `icl_mode`): the
//! speaker embedding among a few control codes, then the reference
//! transcript followed by the text to speak, laid over the reference
//! recording's frames, so the next frames it writes continue that voice
//! saying the new text. The text runs ahead of the frames one token a
//! frame (the reference's streaming layout), and what is left over is fed
//! in as the frames are written. The reference recording's frames are
//! decoded with the new ones, for the codec's context, and their sound cut
//! off the front.

pub mod codec;
pub mod placeholder;
pub mod stack;

use crate::engine::backend::{Backend, MatmulOp};
use crate::engine::loader::{LoadedModel, QuantMatrix};
use crate::engine::tensor;
use crate::engine::tokenizer::Tokenizer;
use anyhow::{Context, Result, anyhow, bail, ensure};
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Where `orangu-server download` fetches the companions from.
pub const CODEC_REPO: &str = orangu::model_download::QWEN3TTS_CODEC_REPO;
pub const CODEC_FILE: &str = orangu::model_download::QWEN3TTS_CODEC_FILE;
pub const VOICES_REPO: &str = orangu::model_download::QWEN3TTS_VOICES_REPO;

/// The longest clip a request or the configuration may ask for.
pub const MAX_SECONDS: f64 = 600.0;

/// Whether the request's own flag, or the server's, says stop.
pub fn cancelled(cancel: Option<&AtomicBool>) -> bool {
    crate::engine::image::cancelled(cancel)
}

/// Raw metadata of a companion file, whose keys need not start with its
/// architecture (the codec's are `qwen3tts_codec.*`).
pub(crate) mod meta {
    use crate::engine::loader::LoadedModel;
    use orangu::gguf::GgufValue;

    fn get<'a>(loaded: &'a LoadedModel, key: &str) -> Option<&'a GgufValue> {
        loaded
            .metadata
            .iter()
            .find_map(|(k, v)| (k == key).then_some(v))
    }

    pub fn u64(loaded: &LoadedModel, key: &str) -> Option<u64> {
        get(loaded, key).and_then(|v| v.as_u64())
    }

    pub fn f32(loaded: &LoadedModel, key: &str) -> Option<f32> {
        match get(loaded, key)? {
            GgufValue::F32(f) => Some(*f),
            GgufValue::F64(f) => Some(*f as f32),
            other => other.as_u64().map(|v| v as f32),
        }
    }

    pub fn string(loaded: &LoadedModel, key: &str) -> Option<String> {
        match get(loaded, key)? {
            GgufValue::String(s) => Some(s.clone()),
            _ => None,
        }
    }

    pub fn u64_array(loaded: &LoadedModel, key: &str) -> Option<Vec<u64>> {
        match get(loaded, key)? {
            GgufValue::Array(items) => Some(items.iter().filter_map(|i| i.as_u64()).collect()),
            _ => None,
        }
    }

    pub fn strings(loaded: &LoadedModel, key: &str) -> Option<Vec<String>> {
        match get(loaded, key)? {
            GgufValue::Array(items) => Some(
                items
                    .iter()
                    .filter_map(|i| match i {
                        GgufValue::String(s) => Some(s.clone()),
                        _ => None,
                    })
                    .collect(),
            ),
            _ => None,
        }
    }
}

/// The files a `qwen3tts` model speaks with.
#[derive(Debug, Clone)]
pub struct Companions {
    /// The 12 Hz speech codec.
    pub codec: PathBuf,
    /// The voice packs, in path order.
    pub voices: Vec<PathBuf>,
}

fn file_spec(models_dir: &Path, key: &str, spec: &str) -> Result<PathBuf> {
    let path = PathBuf::from(spec);
    let path = if path.is_absolute() {
        path
    } else {
        models_dir.join(path)
    };
    ensure!(path.exists(), "{key} {} does not exist", path.display());
    Ok(path)
}

impl Companions {
    /// The configured codec and voices when there are, otherwise the models
    /// directory searched for them by their contents. `voices` is a voice
    /// pack, or a directory searched for them.
    pub fn locate(models_dir: &Path, codec: Option<&str>, voices: Option<&str>) -> Result<Self> {
        let codec = match codec {
            Some(spec) => {
                let path = file_spec(models_dir, "codec", spec)?;
                ensure!(
                    orangu::model_spec::is_qwen3tts_codec(&path),
                    "codec {} is not the Qwen3-TTS 12 Hz speech codec (qwen3tts_tokenizer)",
                    path.display()
                );
                path
            }
            None => orangu::model_spec::find_qwen3tts_codec(models_dir).ok_or_else(|| {
                anyhow!(
                    "no Qwen3-TTS speech codec found under {}. It turns the model's codes into \
                     sound; download {CODEC_REPO}'s {CODEC_FILE}, or point \
                     [orangu-server].codec at it",
                    models_dir.display()
                )
            })?,
        };
        let voices = match voices {
            Some(spec) => {
                let path = file_spec(models_dir, "voices", spec)?;
                if path.is_dir() {
                    orangu::model_spec::find_qwen3tts_voicepacks(&path)
                } else {
                    ensure!(
                        orangu::model_spec::is_qwen3tts_voicepack(&path),
                        "voices {} is not a Qwen3-TTS voice pack (qwen3tts.voicepack)",
                        path.display()
                    );
                    vec![path]
                }
            }
            None => orangu::model_spec::find_qwen3tts_voicepacks(models_dir),
        };
        ensure!(
            !voices.is_empty(),
            "no Qwen3-TTS voice pack found under {}. A voice pack is who speaks; download one \
             from {VOICES_REPO}, or point [orangu-server].voices at one",
            models_dir.display()
        );
        Ok(Self { codec, voices })
    }
}

/// How a clip is handed back.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AudioFormat {
    /// 16-bit PCM in a RIFF/WAVE container.
    #[default]
    Wav,
    /// FLAC, lossless.
    Flac,
    /// Bare 16-bit little-endian PCM, mono, at the codec's rate.
    Pcm,
}

impl AudioFormat {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "wav" | "wave" => Some(Self::Wav),
            "flac" => Some(Self::Flac),
            "pcm" => Some(Self::Pcm),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Flac => "flac",
            Self::Pcm => "pcm",
        }
    }

    pub fn mime(self) -> &'static str {
        match self {
            Self::Wav => "audio/wav",
            Self::Flac => "audio/flac",
            Self::Pcm => "audio/pcm",
        }
    }

    /// The formats a request may name, for an error that lists them.
    pub const NAMES: &'static str = "wav, flac, pcm";

    /// `samples` (mono, in `[-1, 1]`) in this format.
    pub fn encode(self, samples: &[f32], sample_rate: u32) -> Vec<u8> {
        match self {
            Self::Wav => wav(samples, sample_rate),
            Self::Flac => flac(samples, sample_rate),
            Self::Pcm => pcm16(samples),
        }
    }
}

fn pcm16(samples: &[f32]) -> Vec<u8> {
    samples
        .iter()
        .flat_map(|&v| (((v.clamp(-1.0, 1.0) * 32767.0).round()) as i16).to_le_bytes())
        .collect()
}

/// A mono 16-bit RIFF/WAVE file.
pub fn wav(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let data = pcm16(samples);
    let mut out = Vec::with_capacity(44 + data.len());
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(&data);
    out
}

/// A FLAC file: the marker, `STREAMINFO` as the only (last) metadata block,
/// and the frames.
pub fn flac(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let stream = crate::engine::video::flac::encode(samples, 1, sample_rate);
    let mut out = Vec::new();
    out.extend_from_slice(b"fLaC");
    let len = stream.streaminfo.len() as u32;
    out.push(0x80); // last block, type 0 (STREAMINFO)
    out.extend_from_slice(&len.to_be_bytes()[1..]);
    out.extend_from_slice(&stream.streaminfo);
    for (frame, _) in &stream.frames {
        out.extend_from_slice(frame);
    }
    out
}

/// What a clip gets when the request does not say: the `audio_*` keys.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioDefaults {
    /// The voice, by name; `None` is the first one loaded.
    pub voice: Option<String>,
    /// A language the model names, or `auto`.
    pub language: String,
    pub format: AudioFormat,
    /// Sampling for the talker's codes; the code predictor's are the
    /// reference's own and fixed.
    pub temperature: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    /// The longest clip one request makes.
    pub max_seconds: f64,
}

impl Default for AudioDefaults {
    fn default() -> Self {
        Self {
            voice: None,
            language: "auto".to_string(),
            format: AudioFormat::Wav,
            temperature: 0.9,
            top_k: 50,
            repetition_penalty: 1.05,
            max_seconds: 120.0,
        }
    }
}

/// One clip to speak.
#[derive(Debug, Clone)]
pub struct SpeechRequest {
    pub text: String,
    pub voice: Option<String>,
    pub language: Option<String>,
    pub temperature: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub max_seconds: f64,
    pub seed: Option<u64>,
}

impl SpeechRequest {
    pub fn from_defaults(text: &str, d: &AudioDefaults) -> Self {
        Self {
            text: text.to_string(),
            voice: d.voice.clone(),
            language: Some(d.language.clone()),
            temperature: d.temperature,
            top_k: d.top_k,
            repetition_penalty: d.repetition_penalty,
            max_seconds: d.max_seconds,
            seed: None,
        }
    }
}

/// Where a clip is.
#[derive(Debug, Clone, Copy)]
pub enum Progress {
    /// The prompt is read.
    Prompt,
    /// `frames` of speech written.
    Speaking { frames: usize },
    /// The codec has turned `frames` of `of` into sound.
    Decoding { frames: usize, of: usize },
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Timings {
    pub prompt: Duration,
    pub speak: Duration,
    pub decode: Duration,
}

/// A spoken clip.
pub struct Speech {
    /// Mono, in `[-1, 1]`.
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    /// Frames of codes written.
    pub frames: usize,
    pub voice: String,
    pub language: String,
    pub seed: u64,
    /// Rows of the talker's prompt: the control codes and speaker, the
    /// reference transcript and the text over the reference frames.
    pub prompt_rows: usize,
    /// Whether the talker ended the clip itself, rather than the length
    /// limit cutting it.
    pub finished: bool,
    pub timings: Timings,
}

impl Speech {
    pub fn seconds(&self) -> f64 {
        self.samples.len() as f64 / self.sample_rate as f64
    }
}

/// A voice, as a pack holds it.
pub struct Voice {
    pub name: String,
    /// What the pack says about it, when it says.
    pub description: Option<String>,
    pub file: PathBuf,
    speaker: Vec<f32>,
    ref_text: String,
    /// The reference recording's frames, `n_code_groups` codes each.
    ref_codes: Vec<Vec<u32>>,
}

/// A voice as a listing shows it.
#[derive(Debug, Clone, serde::Serialize)]
pub struct VoiceInfo {
    pub name: String,
    pub description: Option<String>,
    /// The voice pack it is in.
    pub pack: String,
}

/// Every voice in `paths` the talker (`width` wide, `groups` codebooks) can
/// speak with; a pack made for another width is skipped with a line in the
/// log. Two voices of one name are told apart by their file.
fn load_voices(paths: &[PathBuf], width: usize, groups: usize) -> Result<Vec<Voice>> {
    let mut voices: Vec<Voice> = Vec::new();
    for path in paths {
        let pack = LoadedModel::open_companion(path)
            .with_context(|| format!("opening the voice pack {}", path.display()))?;
        let names = meta::strings(&pack, "voicepack.names").unwrap_or_default();
        let texts = meta::strings(&pack, "voicepack.ref_texts").unwrap_or_default();
        let description = meta::string(&pack, "crispasr.voice.selected_description");
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("voice")
            .to_string();
        for (i, name) in names.iter().enumerate() {
            let (speaker, _) = pack
                .tensor(&format!("voicepack.spk.{name}.embd"))
                .with_context(|| format!("{}: voice '{name}'", path.display()))?;
            if speaker.len() != width {
                log::info!(
                    "orangu-server: voice '{name}' in {} is for a {}-wide talker, this one is \
                     {width} wide; skipped",
                    path.display(),
                    speaker.len()
                );
                continue;
            }
            let (codes, dims) = pack
                .tensor_i32(&format!("voicepack.code.{name}.codes"))
                .with_context(|| format!("{}: voice '{name}'", path.display()))?;
            ensure!(
                dims.len() == 2 && dims[0] as usize == groups,
                "{}: voice '{name}' has codes of shape {dims:?}, expected [{groups}, frames]",
                path.display()
            );
            let ref_codes = codes
                .chunks(groups)
                .map(|f| f.iter().map(|&c| c.max(0) as u32).collect())
                .collect();
            let ref_text = texts.get(i).cloned().unwrap_or_default();
            ensure!(
                !ref_text.trim().is_empty(),
                "{}: voice '{name}' has no reference transcript",
                path.display()
            );
            let name = if voices.iter().any(|v| v.name == *name) || name == "default" {
                // A pack names its only voice `default`; the file says which.
                stem.strip_prefix("qwen3-tts-voice-")
                    .unwrap_or(&stem)
                    .to_string()
            } else {
                name.clone()
            };
            let name = if voices.iter().any(|v| v.name == name) {
                format!("{name}-{}", voices.len() + 1)
            } else {
                name
            };
            voices.push(Voice {
                name,
                description: description.clone(),
                file: path.clone(),
                speaker,
                ref_text,
                ref_codes,
            });
        }
    }
    Ok(voices)
}

/// The ids the prompt is built from, off `qwen3tts.*`.
#[derive(Debug, Clone)]
struct Ids {
    im_start: u32,
    assistant: u32,
    newline: u32,
    tts_bos: u32,
    tts_eos: u32,
    tts_pad: u32,
    codec_bos: u32,
    codec_eos: u32,
    codec_pad: u32,
    think: u32,
    nothink: u32,
    think_bos: u32,
    think_eos: u32,
}

/// The talker: text embedding and its projection, the codec embedding, the
/// stack and its head.
struct Talker {
    text_embd: QuantMatrix,
    fc1: QuantMatrix,
    fc1_b: Vec<f32>,
    fc2: QuantMatrix,
    fc2_b: Vec<f32>,
    codec_embd: QuantMatrix,
    stack: stack::Stack,
    head: QuantMatrix,
}

/// The code predictor: an embedding table and a head per codebook after
/// the first, and the projection from the talker's width to its own.
struct Predictor {
    embd: Vec<QuantMatrix>,
    to_small: Option<(QuantMatrix, Vec<f32>)>,
    stack: stack::Stack,
    heads: Vec<QuantMatrix>,
}

/// A small, seedable generator for the sampler — SplitMix64.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Top-`k` sampling at `temperature` (greedy at zero); `-inf` logits are
/// never drawn.
fn sample(logits: &[f32], temperature: f32, top_k: usize, rng: &mut Rng) -> u32 {
    let mut order: Vec<usize> = (0..logits.len())
        .filter(|&i| logits[i].is_finite())
        .collect();
    if order.is_empty() {
        return 0;
    }
    order.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    if temperature <= 0.0 || top_k == 1 {
        return order[0] as u32;
    }
    order.truncate(top_k.max(1));
    let top = logits[order[0]];
    let weights: Vec<f32> = order
        .iter()
        .map(|&i| ((logits[i] - top) / temperature).exp())
        .collect();
    let total: f32 = weights.iter().sum();
    let mut r = rng.uniform() * total;
    for (&i, w) in order.iter().zip(&weights) {
        if r < *w {
            return i as u32;
        }
        r -= w;
    }
    *order.last().expect("not empty") as u32
}

pub struct Pipeline {
    backend: Arc<dyn Backend>,
    tokenizer: Tokenizer,
    talker: Talker,
    predictor: Predictor,
    codec: codec::Codec,
    voices: Vec<Voice>,
    /// `(name, codec id)` per language the model knows.
    languages: Vec<(String, u32)>,
    ids: Ids,
    groups: usize,
    max_pos: usize,
    defaults: RwLock<AudioDefaults>,
    /// One clip at a time.
    busy: Mutex<()>,
    pub companions: Companions,
    /// The defaults as the configuration set them, for a reset.
    pub configured: AudioDefaults,
}

impl Pipeline {
    pub fn load(
        model: &Path,
        companions: Companions,
        backend: Arc<dyn Backend>,
        defaults: AudioDefaults,
    ) -> Result<Self> {
        let loaded =
            LoadedModel::open(model).with_context(|| format!("opening {}", model.display()))?;
        let gguf = orangu::gguf::GgufFile::open(model)
            .with_context(|| format!("reading {}", model.display()))?;
        let tokenizer = Tokenizer::from_gguf(&gguf).context("building the qwen3tts tokenizer")?;
        drop(gguf);
        let key = |s: &str| format!("qwen3tts.{s}");
        let need = |s: &str| -> Result<u32> {
            meta::u64(&loaded, &key(s))
                .map(|v| v as u32)
                .ok_or_else(|| anyhow!("qwen3tts: missing {}", key(s)))
        };
        let width = need("talker.d_model")? as usize;
        let groups = need("talker.n_code_groups")? as usize;
        let talker_shape = stack::Shape {
            heads: need("talker.n_heads")? as usize,
            kv_heads: need("talker.n_kv_heads")? as usize,
            head_dim: need("talker.head_dim")? as usize,
            eps: meta::f32(&loaded, &key("talker.rms_norm_eps")).unwrap_or(1e-6),
            theta: meta::f32(&loaded, &key("talker.rope_theta")).unwrap_or(1e6),
        };
        let text_embd = loaded
            .matrix("talker.token_embd_text.weight")
            .context("qwen3tts: loading talker.token_embd_text.weight")?;
        let text_width = text_embd.in_dim;
        let fc1 = stack::matrix(
            &loaded,
            "talker.text_proj.fc1.weight",
            text_width,
            text_width,
        )?;
        let fc2 = stack::matrix(&loaded, "talker.text_proj.fc2.weight", text_width, width)?;
        let codec_embd = loaded
            .matrix("talker.token_embd.weight")
            .context("qwen3tts: loading talker.token_embd.weight")?;
        ensure!(
            codec_embd.in_dim == width,
            "qwen3tts: the codec embedding is {} wide, the talker {width}",
            codec_embd.in_dim
        );
        let codec_vocab = codec_embd.out_dim;
        let talker = Talker {
            fc1_b: stack::vector(&loaded, "talker.text_proj.fc1.bias", text_width)?,
            fc2_b: stack::vector(&loaded, "talker.text_proj.fc2.bias", width)?,
            stack: stack::Stack::load(&loaded, "talker", width, talker_shape)?,
            head: stack::matrix(&loaded, "talker.output.weight", width, codec_vocab)?,
            text_embd,
            fc1,
            fc2,
            codec_embd,
        };

        let cp_width = need("code_pred.d_model")? as usize;
        let cp_head_dim = loaded
            .tensor("code_pred.blk.0.attn_q_norm.weight")
            .context("qwen3tts: loading code_pred.blk.0.attn_q_norm.weight")?
            .0
            .len();
        let cp_shape = stack::Shape {
            heads: need("code_pred.n_heads")? as usize,
            kv_heads: need("code_pred.n_kv_heads")? as usize,
            head_dim: cp_head_dim,
            eps: talker_shape.eps,
            theta: talker_shape.theta,
        };
        let cp_vocab = need("code_pred.vocab_size")? as usize;
        let mut embd = Vec::with_capacity(groups - 1);
        let mut heads = Vec::with_capacity(groups - 1);
        for g in 0..groups - 1 {
            embd.push(stack::matrix(
                &loaded,
                &format!("code_pred.token_embd.{g}.weight"),
                width,
                cp_vocab,
            )?);
            heads.push(stack::matrix(
                &loaded,
                &format!("code_pred.output.{g}.weight"),
                cp_width,
                cp_vocab,
            )?);
        }
        let to_small = if loaded.has_tensor("code_pred.small_to_mtp.weight") {
            Some((
                stack::matrix(&loaded, "code_pred.small_to_mtp.weight", width, cp_width)?,
                stack::vector(&loaded, "code_pred.small_to_mtp.bias", cp_width)?,
            ))
        } else {
            ensure!(
                width == cp_width,
                "qwen3tts: the talker is {width} wide and the code predictor {cp_width}, with \
                 no projection between them"
            );
            None
        };
        let predictor = Predictor {
            embd,
            to_small,
            stack: stack::Stack::load(&loaded, "code_pred", cp_width, cp_shape)?,
            heads,
        };

        let ids = Ids {
            im_start: need("im_start_token_id")?,
            assistant: need("assistant_token_id")?,
            newline: tokenizer
                .encode("\n", false)
                .first()
                .copied()
                .unwrap_or(198),
            tts_bos: need("tts_bos_token_id")?,
            tts_eos: need("tts_eos_token_id")?,
            tts_pad: need("tts_pad_token_id")?,
            codec_bos: need("talker.codec_bos_id")?,
            codec_eos: need("talker.codec_eos_token_id")?,
            codec_pad: need("talker.codec_pad_id")?,
            think: need("talker.codec_think_id")?,
            nothink: need("talker.codec_nothink_id")?,
            think_bos: need("talker.codec_think_bos_id")?,
            think_eos: need("talker.codec_think_eos_id")?,
        };
        let names = meta::strings(&loaded, &key("codec_language_names")).unwrap_or_default();
        let lang_ids = meta::u64_array(&loaded, &key("codec_language_ids")).unwrap_or_default();
        let languages = names
            .into_iter()
            .zip(lang_ids)
            .map(|(n, id)| (n, id as u32))
            .collect();

        let codec_file = LoadedModel::open_companion(&companions.codec)
            .with_context(|| format!("opening the codec {}", companions.codec.display()))?;
        let codec = codec::Codec::load(&codec_file)
            .with_context(|| format!("loading the codec {}", companions.codec.display()))?;
        ensure!(
            codec.shape.n_quantizers == groups,
            "the codec reads {} codebooks and the model writes {groups}",
            codec.shape.n_quantizers
        );
        let voices = load_voices(&companions.voices, width, groups)?;
        ensure!(
            !voices.is_empty(),
            "none of the voice packs ({}) is for a {width}-wide talker; download the one made \
             for this model from {VOICES_REPO}",
            companions
                .voices
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let pipeline = Self {
            backend,
            tokenizer,
            talker,
            predictor,
            codec,
            voices,
            languages,
            ids,
            groups,
            max_pos: need("talker.max_pos").map(|v| v as usize).unwrap_or(32768),
            configured: defaults.clone(),
            defaults: RwLock::new(defaults),
            busy: Mutex::new(()),
            companions,
        };
        pipeline.check_defaults(&pipeline.defaults())?;
        Ok(pipeline)
    }

    pub fn defaults(&self) -> AudioDefaults {
        self.defaults
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    pub fn set_defaults(&self, defaults: AudioDefaults) -> Result<()> {
        self.check_defaults(&defaults)?;
        *self.defaults.write().unwrap_or_else(|p| p.into_inner()) = defaults;
        Ok(())
    }

    fn check_defaults(&self, d: &AudioDefaults) -> Result<()> {
        if let Some(voice) = &d.voice {
            self.voice(Some(voice))?;
        }
        self.language_id(&d.language)?;
        Ok(())
    }

    pub fn voices(&self) -> Vec<VoiceInfo> {
        self.voices
            .iter()
            .map(|v| VoiceInfo {
                name: v.name.clone(),
                description: v.description.clone(),
                pack: v.file.display().to_string(),
            })
            .collect()
    }

    /// The languages a request may name, `auto` first.
    pub fn languages(&self) -> Vec<String> {
        std::iter::once("auto".to_string())
            .chain(self.languages.iter().map(|(n, _)| n.clone()))
            .collect()
    }

    pub fn sample_rate(&self) -> u32 {
        self.codec.shape.sample_rate
    }

    pub fn frame_rate(&self) -> f64 {
        self.codec.frame_rate()
    }

    pub fn talker_layers(&self) -> usize {
        self.talker.stack.n_layers()
    }

    pub fn predictor_layers(&self) -> usize {
        self.predictor.stack.n_layers()
    }

    fn voice(&self, name: Option<&str>) -> Result<&Voice> {
        match name.map(str::trim).filter(|n| !n.is_empty()) {
            None => Ok(&self.voices[0]),
            Some(name) => self
                .voices
                .iter()
                .find(|v| v.name.eq_ignore_ascii_case(name))
                .ok_or_else(|| {
                    anyhow!(
                        "no voice '{name}' (voices: {})",
                        self.voices
                            .iter()
                            .map(|v| v.name.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }),
        }
    }

    /// The codec id for `language`, `None` for `auto`.
    fn language_id(&self, language: &str) -> Result<Option<u32>> {
        let language = language.trim().to_ascii_lowercase();
        if language.is_empty() || language == "auto" {
            return Ok(None);
        }
        // ISO 639-1 codes for the names the model uses.
        let named = match language.as_str() {
            "zh" => "chinese",
            "en" => "english",
            "fr" => "french",
            "de" => "german",
            "it" => "italian",
            "ja" => "japanese",
            "ko" => "korean",
            "pt" => "portuguese",
            "ru" => "russian",
            "es" => "spanish",
            other => other,
        };
        self.languages
            .iter()
            .find(|(n, _)| n == named)
            .map(|(_, id)| Some(*id))
            .ok_or_else(|| {
                anyhow!(
                    "unknown language '{language}' (languages: {})",
                    self.languages().join(", ")
                )
            })
    }

    fn matmul(&self, x: &[f32], n: usize, w: &QuantMatrix) -> Vec<f32> {
        self.backend
            .matmul_batch(&[MatmulOp { x, n_tokens: n, w }])
            .pop()
            .expect("one op in, one result out")
    }

    /// Text tokens through the talker's text embedding and its projection:
    /// `[ids.len(), width]`.
    fn text_rows(&self, ids: &[u32]) -> Vec<f32> {
        let t = &self.talker;
        let n = ids.len();
        let embedded: Vec<f32> = ids
            .par_iter()
            .flat_map_iter(|&id| t.text_embd.row(id as usize))
            .collect();
        let mut h = self.matmul(&embedded, n, &t.fc1);
        tensor::add_bias_per_row(&mut h, &t.fc1_b, n);
        h.par_iter_mut().for_each(|v| *v = tensor::silu(*v));
        let mut out = self.matmul(&h, n, &t.fc2);
        tensor::add_bias_per_row(&mut out, &t.fc2_b, n);
        out
    }

    fn codec_rows(&self, codes: &[u32]) -> Vec<f32> {
        codes
            .iter()
            .flat_map(|&c| self.talker.codec_embd.row(c as usize))
            .collect()
    }

    /// A frame's sixteen codes as one talker input row: codebook 0 through
    /// the talker's codec embedding, the rest through the code predictor's.
    fn frame_row(&self, codes: &[u32]) -> Vec<f32> {
        let mut row = self.talker.codec_embd.row(codes[0] as usize);
        for (g, &c) in codes.iter().enumerate().skip(1) {
            tensor::add_inplace(&mut row, &self.predictor.embd[g - 1].row(c as usize));
        }
        row
    }

    /// The prompt's rows, and the text rows left to feed one a frame.
    fn prompt(
        &self,
        voice: &Voice,
        text: &[u32],
        language: Option<u32>,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let width = self.talker.stack.dim;
        let ids = &self.ids;
        let specials = self.text_rows(&[ids.tts_bos, ids.tts_eos, ids.tts_pad]);
        let (tts_bos, rest) = specials.split_at(width);
        let (tts_eos, tts_pad) = rest.split_at(width);

        // `<|im_start|>assistant\n`.
        let mut rows = self.text_rows(&[ids.im_start, ids.assistant, ids.newline]);
        // The control codes and the speaker, over `tts_pad … tts_bos`.
        let control: Vec<u32> = match language {
            Some(lang) => vec![ids.think, ids.think_bos, lang, ids.think_eos],
            None => vec![ids.nothink, ids.think_bos, ids.think_eos],
        };
        let mut codec = self.codec_rows(&control);
        codec.extend_from_slice(&voice.speaker);
        codec.extend_from_slice(&self.codec_rows(&[ids.codec_pad]));
        let n_codec = codec.len() / width;
        for (i, row) in codec.chunks(width).enumerate() {
            let over = if i + 1 < n_codec { tts_pad } else { tts_bos };
            rows.extend(row.iter().zip(over).map(|(a, b)| a + b));
        }

        // In context: the reference transcript and the text, then
        // `tts_eos`, over `codec_bos` and the reference frames.
        let ref_ids = self.tokenizer.encode(&voice.ref_text, false);
        let mut all: Vec<u32> = ref_ids;
        all.extend_from_slice(text);
        let mut text_rows = self.text_rows(&all);
        text_rows.extend_from_slice(tts_eos);
        let mut frames = self.codec_rows(&[ids.codec_bos]);
        for f in &voice.ref_codes {
            frames.extend(self.frame_row(f));
        }
        let (n_text, n_frames) = (text_rows.len() / width, frames.len() / width);
        let mut trailing = Vec::new();
        for i in 0..n_frames {
            let over: &[f32] = if i < n_text {
                &text_rows[i * width..][..width]
            } else {
                tts_pad
            };
            rows.extend(
                frames[i * width..][..width]
                    .iter()
                    .zip(over)
                    .map(|(a, b)| a + b),
            );
        }
        if n_text > n_frames {
            trailing.extend_from_slice(&text_rows[n_frames * width..]);
        }
        (rows, trailing, tts_pad.to_vec())
    }

    /// Codebooks 1–15 of a frame whose codebook 0 is `first`, from the
    /// talker's last hidden state.
    fn predict(
        &self,
        hidden: &[f32],
        first: u32,
        cache: &mut stack::Cache,
        rng: &mut Rng,
    ) -> Vec<u32> {
        let p = &self.predictor;
        let small = |x: &[f32], n: usize| -> Vec<f32> {
            match &p.to_small {
                Some((w, b)) => {
                    let mut y = self.matmul(x, n, w);
                    tensor::add_bias_per_row(&mut y, b, n);
                    y
                }
                None => x.to_vec(),
            }
        };
        cache.clear();
        let mut input = hidden.to_vec();
        input.extend(self.talker.codec_embd.row(first as usize));
        let mut x = small(&input, 2);
        let mut n = 2;
        let mut codes = Vec::with_capacity(self.groups);
        codes.push(first);
        for g in 0..self.groups - 1 {
            let h = p.stack.forward(&*self.backend, &x, n, cache);
            let last = &h[(n - 1) * p.stack.dim..];
            let logits = self.matmul(last, 1, &p.heads[g]);
            // The reference's own sub-talker sampling: top 50 at 0.9.
            let code = sample(&logits, 0.9, 50, rng);
            codes.push(code);
            if g + 1 < self.groups - 1 {
                x = small(&p.embd[g].row(code as usize), 1);
                n = 1;
            }
        }
        codes
    }

    /// Speaks one clip. Blocks for its whole run; `progress` hears as it
    /// goes.
    pub fn synthesize(
        &self,
        request: &SpeechRequest,
        progress: &mut dyn FnMut(Progress),
        cancel: Option<&AtomicBool>,
    ) -> Result<Speech> {
        let _busy = self.busy.lock().unwrap_or_else(|p| p.into_inner());
        let text = request.text.trim();
        ensure!(!text.is_empty(), "there is no text to speak");
        ensure!(
            request.max_seconds > 0.0 && request.max_seconds <= MAX_SECONDS,
            "max_seconds must be above 0 and at most {MAX_SECONDS}"
        );
        let voice = self.voice(request.voice.as_deref())?;
        let language_name = request
            .language
            .clone()
            .unwrap_or_else(|| "auto".to_string());
        let language = self.language_id(&language_name)?;
        let seed = request.seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0)
        });
        let mut rng = Rng(seed);
        let backend = &*self.backend;
        let width = self.talker.stack.dim;
        let started = Instant::now();

        let tokens = self.tokenizer.encode(text, false);
        let (prompt, trailing, tts_pad) = self.prompt(voice, &tokens, language);
        let n_prompt = prompt.len() / width;
        progress(Progress::Prompt);
        // A runaway guard beside the limit asked for: speech runs at a few
        // tokens a second, so a clip far longer than its text is the talker
        // failing to end it.
        let guard_seconds = 8.0 + 1.2 * tokens.len() as f64;
        let limit = ((request.max_seconds.min(guard_seconds) * self.frame_rate()).ceil() as usize)
            .min(self.max_pos.saturating_sub(n_prompt + 1))
            .max(1);

        let mut cache = self.talker.stack.new_cache();
        let h = self
            .talker
            .stack
            .forward(backend, &prompt, n_prompt, &mut cache);
        let mut hidden = h[(n_prompt - 1) * width..].to_vec();
        let mut logits = self.matmul(&hidden, 1, &self.talker.head);
        let prompt_time = started.elapsed();
        if cancelled(cancel) {
            bail!("cancelled");
        }

        let vocab = logits.len();
        let eos = self.ids.codec_eos as usize;
        let mut cp_cache = self.predictor.stack.new_cache();
        let mut frames: Vec<Vec<u32>> = Vec::new();
        let mut finished = false;
        let speak_started = Instant::now();
        let mut said: Vec<bool> = vec![false; vocab];
        for step in 0..limit {
            if cancelled(cancel) {
                bail!("cancelled");
            }
            // The control codes (the vocabulary's top 1024) are never
            // drawn, except the end — and that only after two frames.
            for (i, v) in logits.iter_mut().enumerate() {
                if i >= vocab.saturating_sub(1024) && (i != eos || step < 2) {
                    *v = f32::NEG_INFINITY;
                } else if said[i] && v.is_finite() {
                    *v = if *v > 0.0 {
                        *v / request.repetition_penalty
                    } else {
                        *v * request.repetition_penalty
                    };
                }
            }
            let first = sample(&logits, request.temperature, request.top_k, &mut rng);
            if first as usize == eos {
                finished = true;
                break;
            }
            said[first as usize] = true;
            let codes = self.predict(&hidden, first, &mut cp_cache, &mut rng);
            let mut input = self.frame_row(&codes);
            frames.push(codes);
            let over = if step < trailing.len() / width {
                &trailing[step * width..][..width]
            } else {
                &tts_pad[..]
            };
            tensor::add_inplace(&mut input, over);
            let h = self.talker.stack.forward(backend, &input, 1, &mut cache);
            hidden = h;
            logits = self.matmul(&hidden, 1, &self.talker.head);
            progress(Progress::Speaking {
                frames: frames.len(),
            });
        }
        let speak_time = speak_started.elapsed();
        if frames.is_empty() {
            bail!("the model wrote no speech for this text");
        }
        if !finished {
            log::info!(
                "orangu-server: speech stopped at its limit ({} frames, {:.1} s) before the \
                 model ended it",
                frames.len(),
                frames.len() as f64 / self.frame_rate()
            );
        }

        // The reference frames first, for the codec's context.
        let decode_started = Instant::now();
        let ref_len = voice.ref_codes.len();
        let mut all = voice.ref_codes.clone();
        all.extend(frames.iter().cloned());
        let total = all.len();
        let wav = self.codec.decode(
            backend,
            &all,
            &mut |done| {
                progress(Progress::Decoding {
                    frames: done.saturating_sub(ref_len).min(frames.len()),
                    of: frames.len(),
                })
            },
            cancel,
        )?;
        let cut = ref_len * wav.len() / total.max(1);
        let samples = wav[cut..].to_vec();
        Ok(Speech {
            samples,
            sample_rate: self.sample_rate(),
            frames: frames.len(),
            voice: voice.name.clone(),
            language: language_name,
            seed,
            prompt_rows: n_prompt,
            finished,
            timings: Timings {
                prompt: prompt_time,
                speak: speak_time,
                decode: decode_started.elapsed(),
            },
        })
    }
}

/// What a clip in flight reports.
pub enum SpeechEvent {
    Progress(Progress),
    Done(Box<Speech>),
    Error(String),
}

/// Stops a clip in flight when dropped — held by whoever reads its events,
/// so a client that goes away stops the work at the next frame.
pub struct CancelOnDrop(Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

impl Pipeline {
    /// Runs `request` on a blocking thread, streaming [`SpeechEvent`]s back.
    pub fn spawn(
        self: &Arc<Self>,
        request: SpeechRequest,
    ) -> (tokio::sync::mpsc::Receiver<SpeechEvent>, CancelOnDrop) {
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let cancel = Arc::new(AtomicBool::new(false));
        let pipeline = self.clone();
        let flag = cancel.clone();
        tokio::task::spawn_blocking(move || {
            let progress_tx = tx.clone();
            let mut progress = |p: Progress| {
                let _ = progress_tx.try_send(SpeechEvent::Progress(p));
            };
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                pipeline.synthesize(&request, &mut progress, Some(&flag))
            }));
            let event = match result {
                Ok(Ok(speech)) => SpeechEvent::Done(Box::new(speech)),
                Ok(Err(err)) => SpeechEvent::Error(format!("{err:#}")),
                Err(_) => SpeechEvent::Error(
                    crate::panic_capture::take_last_panic_detail()
                        .unwrap_or_else(|| "speech generation panicked".to_string()),
                ),
            };
            let _ = tx.blocking_send(event);
        });
        (rx, CancelOnDrop(cancel))
    }

    /// Runs `request` to its end: the clip, or why there is none.
    pub async fn speak(self: &Arc<Self>, request: SpeechRequest) -> Result<Speech, String> {
        let (mut rx, _cancel) = self.spawn(request);
        while let Some(event) = rx.recv().await {
            match event {
                SpeechEvent::Progress(_) => {}
                SpeechEvent::Done(speech) => return Ok(*speech),
                SpeechEvent::Error(err) => return Err(err),
            }
        }
        Err("speech generation ended early".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_sampling_takes_the_largest_finite_logit() {
        let mut rng = Rng(1);
        let logits = [0.5, f32::NEG_INFINITY, 2.0, 1.0];
        assert_eq!(sample(&logits, 0.0, 50, &mut rng), 2);
        assert_eq!(sample(&logits, 0.9, 1, &mut rng), 2);
    }

    #[test]
    fn top_k_sampling_never_draws_outside_the_top_k() {
        let mut rng = Rng(7);
        let logits = [3.0, 2.9, -1.0, -2.0, f32::NEG_INFINITY];
        for _ in 0..200 {
            let i = sample(&logits, 1.0, 2, &mut rng);
            assert!(i == 0 || i == 1);
        }
    }

    #[test]
    fn a_wav_file_has_its_header_and_one_sample_per_two_bytes() {
        let bytes = wav(&[0.0, 1.0, -1.0], 24000);
        assert_eq!(&bytes[..4], b"RIFF");
        assert_eq!(&bytes[8..16], b"WAVEfmt ");
        assert_eq!(bytes.len(), 44 + 6);
        assert_eq!(i16::from_le_bytes([bytes[46], bytes[47]]), 32767);
        assert_eq!(i16::from_le_bytes([bytes[48], bytes[49]]), -32767);
    }

    #[test]
    fn a_flac_file_starts_with_its_marker_and_streaminfo() {
        let bytes = flac(&vec![0.25; 5000], 24000);
        assert_eq!(&bytes[..4], b"fLaC");
        assert_eq!(bytes[4], 0x80);
        assert_eq!(u32::from_be_bytes([0, bytes[5], bytes[6], bytes[7]]), 34);
    }

    /// The real model, codec and voices under `ORANGU_QWEN3TTS_DIR`: the
    /// voice's reference codes decoded to `ref.wav` and one sentence spoken
    /// to `speech.wav`, both in `ORANGU_QWEN3TTS_OUT`.
    #[test]
    #[ignore = "needs ORANGU_QWEN3TTS_DIR (a models directory with a qwen3tts model, its codec \
                and a voice pack) and ORANGU_QWEN3TTS_OUT"]
    fn speaks_with_the_real_model() {
        let dir = PathBuf::from(std::env::var("ORANGU_QWEN3TTS_DIR").expect("a models directory"));
        let out = PathBuf::from(std::env::var("ORANGU_QWEN3TTS_OUT").expect("an output directory"));
        let model = walkdir::WalkDir::new(&dir)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
            .map(|e| e.into_path())
            .find(|p| {
                p.extension().is_some_and(|e| e == "gguf")
                    && orangu::gguf::GgufFile::open_summary(p)
                        .ok()
                        .and_then(|g| orangu::model_spec::architecture_of(&g))
                        .as_deref()
                        == Some("qwen3tts")
            })
            .expect("a qwen3tts model");
        let companions = Companions::locate(&dir, None, None).expect("companions");
        let backend: Arc<dyn Backend> = Arc::new(crate::engine::backend::CpuBackend);
        let pipeline = Pipeline::load(
            &model,
            companions,
            backend.clone(),
            AudioDefaults::default(),
        )
        .expect("load");
        let voice = &pipeline.voices[0];
        let started = Instant::now();
        let reference = pipeline
            .codec
            .decode(&*backend, &voice.ref_codes, &mut |_| {}, None)
            .expect("decode");
        eprintln!(
            "decoded {} reference frames to {} samples in {:.2} s",
            voice.ref_codes.len(),
            reference.len(),
            started.elapsed().as_secs_f64()
        );
        std::fs::write(out.join("ref.wav"), wav(&reference, pipeline.sample_rate())).unwrap();
        let text = std::env::var("ORANGU_QWEN3TTS_TEXT").unwrap_or_else(|_| {
            "Hello! This is orangu speaking, with a voice cloned from a short recording.".into()
        });
        let mut request = SpeechRequest::from_defaults(&text, &AudioDefaults::default());
        request.seed = Some(1);
        let speech = pipeline
            .synthesize(&request, &mut |_| {}, None)
            .expect("speech");
        eprintln!(
            "{} frames, {:.2} s of speech (finished: {}); prompt {:.2} s, speak {:.2} s, decode \
             {:.2} s",
            speech.frames,
            speech.seconds(),
            speech.finished,
            speech.timings.prompt.as_secs_f64(),
            speech.timings.speak.as_secs_f64(),
            speech.timings.decode.as_secs_f64()
        );
        std::fs::write(
            out.join("speech.wav"),
            wav(&speech.samples, speech.sample_rate),
        )
        .unwrap();
        assert!(speech.finished);
    }

    #[test]
    fn formats_parse_by_name() {
        assert_eq!(AudioFormat::parse("WAV"), Some(AudioFormat::Wav));
        assert_eq!(AudioFormat::parse("flac"), Some(AudioFormat::Flac));
        assert_eq!(AudioFormat::parse("pcm"), Some(AudioFormat::Pcm));
        assert_eq!(AudioFormat::parse("mp3"), None);
    }
}
