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

//! The serving settings the engine reads: the role a server runs in, how its
//! KV cache is stored, the expert read size and picture sizes. Parsed from
//! `orangu-server.conf` by the server and handed to the engine.

use anyhow::{Result, anyhow};

/// `[orangu-server].read_size` when the key is absent, in KiB.
///
/// 8 MiB, and the number is measured rather than round. Storage throughput is
/// closer to a step than a slope: on the drive this was sized against, reads
/// at or below 512 KiB ran at 15-28 MB/s and reads from 1 MiB up ran at
/// 206-214 MB/s, because the block layer splits a larger request into several
/// commands and issues them together where a smaller one pays a full round
/// trip. 8 MiB sits well inside that plateau.
///
/// End to end, on a mixture-of-experts model read cold, widening to 8 MiB
/// measured **+36% decode tok/s** against not widening — and, just as usefully,
/// it made the result *stable*: three repetitions at 8 MiB spread 7.89-8.44
/// tok/s where three at one page spread 2.25-8.65. Large sequential reads keep
/// the device in its fast regime; scattered small ones let it fall out. Warm,
/// where nothing reaches the disk, the two are within noise of each other.
///
/// **The arms have to be interleaved to see any of this.** Run sequentially,
/// this same comparison reported widening as 2.2x *slower* — the drive
/// degrades measurably across a session, so whichever arm ran second lost.
pub const DEFAULT_READ_SIZE: usize = 8192;

/// A hint at which of `orangu-server`'s features matter for this
/// deployment — set via one of `--all`/`--code`/`--review`/`--explorer`/
/// `--embedding` (mutually exclusive; `--all` is the default) or the
/// config file's `role` key. Unlike a real `llama-server` process (a
/// distinct binary per deployment, so `orangu`'s own conventional roles —
/// `all`/`code`/`review`/`explorer`/`embeddings` — pick model *and* a whole
/// flag set), a single `orangu-server` process serves whatever model it's
/// given; this only adjusts the
/// handful of things that are actually role-specific in a from-scratch
/// engine that doesn't have `--fit`/`--tools`/`--webui-mcp-proxy`/`-sm`/
/// `--cache-reuse`/`-ctk`/`-ctv` equivalents at all: the default slot
/// count, default sampling parameters, whether the generation endpoints
/// are even served, and (`Review` only) reasoning suppression.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Role {
    #[default]
    All,
    Code,
    Review,
    Explorer,
    Embedding,
    /// Picture generation — the role of a `qwen_image` model, and the only
    /// one such a model is served in. Unlike the other five it is not a
    /// tuning chosen for a model: it is what the model *is*, so an image
    /// model comes up in it without being asked (see
    /// [`Role::fixed_by_model`]) and a text model cannot take it.
    Image,
    /// Video generation — the role of an `ltxv` model, fixed by the model
    /// exactly as [`Role::Image`] is.
    Video,
    /// Text to speech — the role of a `qwen3tts` model, fixed by the model
    /// exactly as [`Role::Image`] is.
    Audio,
}

impl Role {
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_lowercase().as_str() {
            "all" => Ok(Role::All),
            "code" => Ok(Role::Code),
            "review" => Ok(Role::Review),
            "explorer" => Ok(Role::Explorer),
            "embedding" => Ok(Role::Embedding),
            "image" => Ok(Role::Image),
            "video" => Ok(Role::Video),
            "audio" => Ok(Role::Audio),
            other => Err(anyhow!(
                "invalid role '{other}' (expected all, code, review, explorer, embedding, image, \
                 video, or audio)"
            )),
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Role::All => "all",
            Role::Code => "code",
            Role::Review => "review",
            Role::Explorer => "explorer",
            Role::Embedding => "embedding",
            Role::Image => "image",
            Role::Video => "video",
            Role::Audio => "audio",
        }
    }

    /// Whether this is the role a model's architecture decides — [`Role::
    /// Image`], [`Role::Video`] or [`Role::Audio`], which a picture, video or
    /// speech generator always has and nothing else can have — as opposed to
    /// the five an operator picks between for a language model. A fixed role
    /// is never prompted for and never narrowed by an `x-orangu-role` header.
    pub fn fixed_by_model(&self) -> bool {
        matches!(self, Role::Image | Role::Video | Role::Audio)
    }

    /// The role a model must be served in, from its `general.architecture`
    /// — `Some(Role::Image)` for a picture generator, `Some(Role::Video)` for
    /// a video generator, `Some(Role::Audio)` for a speech generator, `None`
    /// for a language model, whose role is the operator's choice.
    pub fn required_by(architecture: Option<&str>) -> Option<Role> {
        let architecture = architecture?;
        if crate::model_spec::is_image_architecture(architecture) {
            Some(Role::Image)
        } else if crate::model_spec::is_video_architecture(architecture) {
            Some(Role::Video)
        } else if crate::model_spec::is_audio_architecture(architecture) {
            Some(Role::Audio)
        } else {
            None
        }
    }

    /// Default request-queue depth per slot before a new request is
    /// rejected rather than queued, when the config file doesn't set
    /// `slots` explicitly. `Embedding` defaults higher (matching the
    /// mapped `llama-server -np 8`): embedding requests are typically
    /// short, cheap, and bursty compared to open-ended generation, so
    /// serving more of them concurrently is the right default; every
    /// other role keeps the previous flat default of `1`.
    pub fn default_slots(&self) -> usize {
        match self {
            Role::Embedding => 8,
            // Generation defaults to four slots: one user's agents run
            // several requests at once, and the server serves them together.
            // Decoding slots share one forward pass a step
            // (`engine::decode_batch`), so four streams make about twice the
            // tokens of one; a request whose prompt another is building waits
            // for it and shares it (`engine::prefix_index`); and a prompt makes
            // way for the others' decode steps between its layers. A slot
            // costs a single stream nothing: the KV pool is shared, not split.
            //
            // A review is one request at a time, and an image, video or
            // speech model makes one picture, clip or recording at a time
            // (`engine::image::Pipeline`, `engine::video::Pipeline` and
            // `engine::audio::Pipeline` each hold a single busy lock), so a
            // second slot there would only queue.
            Role::All | Role::Code | Role::Explorer => 4,
            Role::Review | Role::Image | Role::Video | Role::Audio => 1,
        }
    }

    /// Whether `/v1/chat/completions`, `/v1/completions`, and `/completion`
    /// should even be served. Only `Embedding` disables them — the one
    /// role that's a genuinely different use case (an embeddings-only
    /// model's `forward_hidden_states` path) from the other four, which
    /// are all ordinary text generation with different tuning.
    pub fn allows_generation(&self) -> bool {
        !matches!(self, Role::Embedding)
    }

    /// Whether a chat-completion request should suppress a reasoning-
    /// capable model's thinking phase — the `Review` role's mapped
    /// `--reasoning-budget 0 --reasoning off`. See `http::openai::
    /// chat_completions`'s own doc comment for exactly how this is
    /// approximated without llama.cpp's own reasoning-parsing machinery.
    pub fn suppresses_reasoning(&self) -> bool {
        matches!(self, Role::Review)
    }

    /// The `enable_thinking` value to pass to `engine::chat_template::
    /// ChatTemplate::render` for this role — `Some(false)` for `Review`
    /// (see [`Role::suppresses_reasoning`]), `None` (leave the template's
    /// own default/auto-detection alone) for every other role.
    pub fn enable_thinking(&self) -> Option<bool> {
        self.suppresses_reasoning().then_some(false)
    }
}

/// `[orangu-server].kv_cache`: how the GPU-side KV mirror is stored.
///
/// The KV cache is re-read in full by every attention dispatch, so its storage
/// width is a direct multiplier on attention's memory traffic and grows with
/// context — which makes it one of the few knobs that trades *quality* for
/// *both* memory and speed at once, rather than one for the other.
///
/// On the Vulkan-family backends (Vulkan, Metal, DX12) this is the device
/// mirror's storage. On the CPU backend only `q8_0` changes anything, storing
/// the sealed host pages that way; those are `f32` otherwise. On CUDA, OpenCL
/// and ROCm this has no effect.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KvCache {
    /// Half precision — the default, and what the reference engines default
    /// to. Requires the adapter to support `SHADER_F16`; without it the
    /// engine falls back to `f32` on its own.
    #[default]
    F16,
    /// 8-bit block-quantized, about 44% smaller than `f16`. **Lossy**, which
    /// is why it is not the default: it changes generated text, unlike every
    /// other storage choice here. Buys context and concurrent slots on a
    /// machine where memory is the binding constraint, and cuts attention's
    /// read bandwidth at long context.
    Q8_0,
    /// Full precision. Larger and slower than `f16` for no quality gain that
    /// has ever been measured here — kept because it is the fallback an
    /// adapter without `SHADER_F16` gets anyway, and naming it makes that
    /// state reachable deliberately rather than only by accident.
    F32,
}

impl KvCache {
    /// Every value, in the order the error message lists them.
    pub const ALL: [KvCache; 3] = [KvCache::F16, KvCache::Q8_0, KvCache::F32];

    /// The accepted spellings, comma-separated — built from [`ALL`](Self::ALL)
    /// so a value added to the enum cannot be missing from the message that
    /// tells an operator what they may write.
    pub fn accepted() -> String {
        Self::ALL
            .iter()
            .map(|value| value.tag())
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// The spelling this value answers to in the config file and in
    /// `ORANGU_KV_CACHE`.
    pub fn tag(self) -> &'static str {
        match self {
            KvCache::F16 => "f16",
            KvCache::Q8_0 => "q8_0",
            KvCache::F32 => "f32",
        }
    }

    /// This value as a format for host pages. `f16` stays `f32` there: it is
    /// the one format that makes host attention slower.
    pub fn row_format(self) -> crate::engine::kv_rows::RowFormat {
        use crate::engine::kv_rows::RowFormat;
        match self {
            KvCache::Q8_0 => RowFormat::Q8_0,
            KvCache::F16 | KvCache::F32 => RowFormat::F32,
        }
    }

    /// Parses one of the names, or `None` for anything else.
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_lowercase().as_str() {
            "f16" => Some(KvCache::F16),
            "q8_0" | "q8" => Some(KvCache::Q8_0),
            "f32" => Some(KvCache::F32),
            _ => None,
        }
    }
}

/// The smallest and largest side a picture size may name. A size between
/// them is drawn at the model's unit (rounded up, then cropped back) and
/// capped to the largest area the model draws.
pub const MIN_IMAGE_SIDE: usize = 16;
pub const MAX_IMAGE_SIDE: usize = 16384;

/// `WIDTHxHEIGHT` (or `WIDTH` alone for a square), each side from
/// [`MIN_IMAGE_SIDE`] to [`MAX_IMAGE_SIDE`] pixels. Any side in that range
/// is a size: one off the model's unit is drawn rounded up and cropped
/// back, so 1920x1080 is 1920x1080.
pub fn parse_image_size(value: &str) -> Option<(usize, usize)> {
    let value = value.trim().to_ascii_lowercase();
    let (w, h) = match value.split_once(['x', '*']) {
        Some((w, h)) => (
            w.trim().parse::<usize>().ok()?,
            h.trim().parse::<usize>().ok()?,
        ),
        None => {
            let side = value.parse::<usize>().ok()?;
            (side, side)
        }
    };
    let side = MIN_IMAGE_SIDE..=MAX_IMAGE_SIDE;
    (side.contains(&w) && side.contains(&h)).then_some((w, h))
}
